# 15M Training Guide — V3.3
# THE HARDEST TIMEFRAME. 227K rows, 10M+ features, 150-220GB NPZ.
# Read EVERY section. Another Claude session will use this as its sole reference.
# Last in warm-start cascade: 1w -> 1d -> 4h -> 1h -> **15m**

---

## Machine Requirements

- **RAM:** 1.5TB+ cgroup (NOT host RAM -- containers cap lower)
- **Verify cgroup:** `cat /sys/fs/cgroup/memory/memory.limit_in_bytes` -- must show >= 1,610,612,736,000 (1.5TB)
  - If cgroup shows `9223372036854775807` (max int64), there is no cgroup limit -- use `free -g` instead
- **GPU:** REQUIRED. CUDA 12.x for full GPU acceleration
  - If CUDA 13 (driver 580+): CPU only, 3-5x slower, still works with `ALLOW_CPU=1`
- **Cores:** 128+ (cross gen multi-threaded, Optuna n_jobs=4)
- **CPU Score:** 800+ (cores x base GHz). Higher = faster cross gen and Optuna search
- **Disk:** 300GB+ free (NPZ alone is 150-220GB)
- **USER PICKS THIS MACHINE** from vast.ai lineup. Never auto-select.

### Why One Machine for Everything
The NPZ file is 150-220GB. Transferring it between machines is impractical (30 min at 1 Gbps, 5 hrs at 100 Mbps). Pick a single cost-effective machine that runs cross gen + training + Optuna. Never plan to transfer the NPZ.

---

## GPU vs CPU Per Step

| Step | Engine | Notes |
|------|--------|-------|
| Feature build | CPU (pandas) | cuDF if CUDA 12.x available, otherwise pandas. ~2-5 min. |
| Cross gen | GPU + memmap | Per-type NPZ checkpointing. ~1.5-3 hrs GPU, ~6-8 hrs CPU. |
| Optuna search | CPU (n_jobs=4) | 15% row subsample (227K -> ~34K). CPU parallel search. |
| Final CPCV | GPU histogram fork | 3x faster than CPU at 227K rows. enable_bundle=False. |
| Meta + PBO + SHAP | CPU | ~30 min. |
| LSTM | Run locally (13900K + 3090) | Cloud H200 has weak CPU for DataLoader. |

### GPU Histogram Fork
The GPU histogram fork (`gpu_histogram_fork/`) provides `device_type="cuda_sparse"` for LightGBM. This is 3x faster than CPU for 227K rows because GPU histogram building parallelizes across the large dataset. The fork reads sparse CSR directly via `set_external_csr` -- no dense conversion needed.

### CUDA Version Handling
- **CUDA 12.x (driver 535-575):** Full GPU acceleration. cuDF, CuPy, GPU histogram fork all work.
- **CUDA 13.0+ (driver 580+):** cuDF/CuPy SEGFAULT on CUDA 13. Auto-detected. Feature build falls back to pandas CPU. Training still works with `ALLOW_CPU=1` but is 3-5x slower. Prefer CUDA 12.x machines.

---

## Data Profile

- **Rows:** ~227,000 (15m bars, 2017-11-01 to 2026, Binance BTC/USDT)
- **Base features:** ~4,000+ cols after feature_library.py
- **Cross features:** ~10M+ (min_nonzero=3)
- **NPZ size:** 150-220GB on disk
- **NNZ estimate:** ~20-30B (227K rows x 10M cols x ~1% density) -- massively exceeds int32 limit
- **int64 indptr is the PRIMARY fix** for NNZ > 2^31. Row-partitioned boosting REJECTED.

---

## Triple-Barrier Labels (from feature_library.py)

| Parameter | Value |
|-----------|-------|
| tp_atr_mult | 2.0 |
| sl_atr_mult | 1.8 |
| max_hold_bars | 32 |

Asymmetric barriers fix 0% SHORT precision on upward-biased BTC.

---

## LightGBM Config (from config.py)

| Parameter | Value | Source |
|-----------|-------|--------|
| min_data_in_leaf | 15 | TF_MIN_DATA_IN_LEAF['15m'] |
| num_leaves | 511 | TF_NUM_LEAVES['15m'] -- deep trees viable at 227K rows |
| max_bin | 255 | V3_LGBM_PARAMS (binary crosses get 2 bins regardless) |
| CPCV | (6,2) = 15 splits | TF_CPCV_GROUPS['15m'] -- 5 unique paths, 67% train |
| enable_bundle | False | MANDATORY for >1M features with >40K rows (EFB intractable) |
| feature_pre_filter | False | CRITICAL -- True silently kills rare esoteric features |
| is_enable_sparse | True | Sparse CSR fed directly, no dense conversion (would be 6.8TB) |

### Why enable_bundle=False
EFB (Exclusive Feature Bundling) scans all feature pairs for mutual exclusivity. At 10M+ features with 227K rows, this scan is intractable. Disabling it skips the bundle construction entirely. Binary features still train correctly -- they just get individual histogram bins instead of shared bundles. Training is slightly slower per tree but the Dataset construction phase drops from hours to minutes.

### Dense vs Sparse
At 227K rows x 10M features: dense = 227,000 x 10,000,000 x 4 bytes = **~8.5 TB**. Impossible. Stays sparse CSR. LightGBM trains directly on sparse with int64 indptr.

---

## CPCV Configuration

- **(6,2):** 6 groups, 2 test groups per split
- **Splits:** C(6,2) = 15
- **Unique paths:** 5 (phi = (2/6) x 15 = 5)
- **Train fraction:** 67% (4/6 groups per split)
- **Sequential CPCV:** Parallel disabled for >1M features (pickle bottleneck). One fold at a time.
- **Embargo/Purge:** max_hold_bars=32 used for purge window between train/test boundaries

---

## Optuna Configuration

- **Warm-started from 1h** (last in cascade: 1w -> 1d -> 4h -> 1h -> 15m)
- **Phase 1:** 30 trials (2 seeded + 8 random + 20 TPE), 2-fold CPCV, LR=0.15
- **Row subsample:** 15% (227K -> ~34K rows) -- search only, final model uses ALL rows
- **Validation gate:** Top 3 re-evaluated with 4-fold CPCV
- **Final retrain:** Full (6,2) CPCV, 800 rounds, LR=0.03
- **n_jobs:** Auto (total_cores // 8), or env `OPTUNA_N_JOBS`
- **Engine:** CPU for search (row subsample makes GPU marginal). GPU for final retrain only.

---

## Environment Variables (15m-specific)

```bash
export V2_RIGHT_CHUNK=500     # Cross gen chunk size (500 is safe for 1.5TB+)
export V2_BATCH_MAX=500       # Cap dense intermediate arrays per worker
export ALLOW_CPU=1            # Only if CUDA 13 (no GPU fallback otherwise)
## OMP_NUM_THREADS / NUMBA_NUM_THREADS — set dynamically by cloud_run_tf.py per phase
```

### Why V2_RIGHT_CHUNK=500
At 227K rows, each right chunk materializes arrays of (227K x chunk_size x 4 bytes). RC=500 peaks at ~500-700GB on 1.5TB+. RC=200 is safer but slower. If RAM < 1.5TB, lower to RC=200.

### min_nonzero=3
Default in v2_cross_generator.py. Matches min_data_in_leaf logic for rare esoteric signals. Can override via `V2_MIN_CO_OCCURRENCE=3`.

---

## CRITICAL: NNZ int32 Overflow

LightGBM sparse CSR uses int32 for `indptr` and `indices`. Max NNZ = 2^31 - 1 = 2,147,483,647.

15m at 227K rows x 10M+ features has NNZ far exceeding this limit.

### How ml_multi_tf.py handles it (v3.3):
`_ensure_lgbm_sparse_dtypes()` applied after NPZ load AND after hstack:
1. `indptr` cast to int64 -- handles cumulative NNZ values > 2^31
2. `indices` cast to int32 -- column IDs (max ~10M, fits int32)
3. LightGBM C API accepts int64 indptr since PR #1719 (2018)
4. If NNZ > int32 max: logs info, forces sequential CPCV, training proceeds normally

### Risk if int64 indptr not applied
LightGBM trains on corrupted int32-overflowed sparse matrix. Model looks trained but predictions are garbage. There is NO error message -- completely silent.

### Row-partitioned boosting: REJECTED
Perplexity confirmed: row partitioning kills rare signals by splitting occurrence counts below min_data_in_leaf per chunk. 227K rows / 13 chunks = ~17K rows/chunk. Features firing 15 times globally = ~1.1 per chunk = below min_data_in_leaf=15. LightGBM NEVER splits on rare esoteric signals. **NEVER use row-partitioned boosting.**

---

## Cross Gen Checkpointing

Cross gen at 15m takes 1.5-8 hours and produces a 150-220GB NPZ. If OOM kills the process at cross type 12, types 1-11 are recoverable from per-type checkpoint files.

### How it works:
- Each completed cross type saves: `_cross_checkpoint_15m_{prefix}.npz` + `_cross_checkpoint_15m_{prefix}_names.json`
- On restart, existing checkpoints are loaded and only remaining cross types are computed
- After all types complete, checkpoints are merged into `v2_crosses_BTC_15m.npz` and checkpoint files are cleaned up

### Checkpoint files location:
```
/workspace/v3.3/_cross_checkpoint_15m_dx.npz        # Trend crosses
/workspace/v3.3/_cross_checkpoint_15m_dx_names.json
/workspace/v3.3/_cross_checkpoint_15m_ax.npz        # Astro crosses
/workspace/v3.3/_cross_checkpoint_15m_ax_names.json
... (one pair per cross type)
```

### If OOM during cross gen:
1. Check `dmesg | tail -20` for "oom-kill"
2. Lower V2_RIGHT_CHUNK (500 -> 200 -> 100)
3. Restart -- checkpoints will resume from where it stopped
4. Do NOT delete checkpoint files unless doing a full nuclear clean

---

## Required Databases (ALL 16 -- ZERO MISSING)

**From project root (-> /workspace/):**
```
btc_prices.db          # 1.3GB -- BTC OHLCV 2010-2026
tweets.db              # tweet text + gematria
news_articles.db       # news headlines
astrology_full.db      # planetary positions
ephemeris_cache.db     # ephemeris calculations
fear_greed.db          # fear/greed index
sports_results.db      # sports event correlations
space_weather.db       # Kp index, solar flux
macro_data.db          # macro indicators
onchain_data.db        # on-chain metrics
funding_rates.db       # funding rates
open_interest.db       # open interest
google_trends.db       # google trends
llm_cache.db           # LLM feature cache
```

**From v3.3/ (-> /workspace/v3.3/):**
```
multi_asset_prices.db  # 1.3GB -- multi-asset data
v2_signals.db          # DeFi TVL, BTC dominance, mining stats
```

**Also required (non-DB):**
```
kp_history_gfz.txt     # historical Kp index data (in /workspace/ or /workspace/v3.3/)
astrology_engine.py    # must be in v3.3/ directory (feature_library.py imports it)
```

### Verify Script (run BEFORE launching -- ALL must say OK)
```bash
echo "=== DB Verification ==="
FAIL=0
for db in btc_prices.db tweets.db news_articles.db astrology_full.db ephemeris_cache.db \
  fear_greed.db sports_results.db space_weather.db macro_data.db \
  onchain_data.db funding_rates.db open_interest.db google_trends.db \
  llm_cache.db; do
  if [ -f /workspace/$db ] || [ -f /workspace/v3.3/$db ]; then
    echo "OK   $db"
  else
    echo "MISS $db"; FAIL=1
  fi
done
for db in multi_asset_prices.db v2_signals.db; do
  if [ -f /workspace/v3.3/$db ]; then
    echo "OK   v3.3/$db"
  else
    echo "MISS v3.3/$db"; FAIL=1
  fi
done
[ -f /workspace/v3.3/kp_history_gfz.txt ] || [ -f /workspace/kp_history_gfz.txt ] && echo "OK   kp_history_gfz.txt" || { echo "MISS kp_history_gfz.txt"; FAIL=1; }
[ -f /workspace/v3.3/astrology_engine.py ] && echo "OK   astrology_engine.py" || { echo "MISS astrology_engine.py"; FAIL=1; }
echo ""
if [ $FAIL -eq 1 ]; then echo "STOP: Missing files. Upload before launching."; else echo "ALL OK -- safe to launch."; fi
```
**If ANY says MISS -> STOP. Upload the missing file first. Missing DB = broken matrix = invalid model.**

---

## Nuclear Clean (MANDATORY before first run)

Delete ALL old artifacts. Old NPZs built with min_nonzero=8 produce fewer features. Old cross names JSON truncates column count. DELETE BOTH.

```bash
cd /workspace && rm -f *.npz *.json *.pkl *.parquet *.log DONE_* RUNNING_* *.lock 2>/dev/null
# NOTE: no *.txt -- would kill kp_history_gfz.txt
cd /workspace/v3.3 && rm -f \
  v2_crosses_*.npz v2_cross_names_*.json _cross_checkpoint_*.npz _cross_checkpoint_*_names.json \
  v2_base_*.parquet features_BTC_*.parquet features_*_all.json \
  model_*.json platt_*.pkl cpcv_oos_*.pkl \
  feature_importance_*.json shap_analysis_*.json validation_report_*.json \
  meta_model_*.pkl ml_multi_tf_*.* optuna_configs_all.json \
  lstm_*.pt lgbm_dataset_*.bin DONE_* RUNNING_* *.lock 2>/dev/null
echo '{"steps": {}, "version": "3.3"}' > pipeline_manifest.json
echo 'Nuclear clean done'
```

**CRITICAL:** Both `v2_crosses_*.npz` AND `v2_cross_names_*.json` must be deleted together. Also delete checkpoint files (`_cross_checkpoint_*`) to avoid mixing old/new cross types.

---

## Verify Parquet Freshness

```bash
python3 -c "
import pandas as pd, os
p = '/workspace/v3.3/features_BTC_15m.parquet'
if os.path.exists(p):
    df = pd.read_parquet(p)
    print(f'Parquet: {len(df)} rows x {len(df.columns)} cols')
    print(f'Modified: {pd.Timestamp(os.path.getmtime(p), unit=\"s\")}')
else:
    print('No parquet found -- will be built fresh')
"
```

---

## Install Dependencies

```bash
pip install -q lightgbm scikit-learn scipy ephem astropy pytz joblib pandas numpy \
  pyarrow optuna hmmlearn numba tqdm pyyaml alembic cmaes colorlog sqlalchemy \
  threadpoolctl psutil 2>&1 | tail -3
python -c "import pandas, numpy, scipy, sklearn, lightgbm, ephem, astropy, pyarrow, optuna, numba, hmmlearn, yaml, tqdm; print('ALL OK')"
```

---

## Setup Script (Memory Optimizations)

Run `setup.sh` before launching the pipeline. It configures:
1. tcmalloc (google-perftools) -- 5-15% speedup from per-thread malloc caches
2. Transparent Huge Pages -- 5-20% from reduced dTLB misses
3. vm.swappiness=1 -- prevents kernel stealing hot pages
4. NUMA topology detection -- binding recommendation for multi-socket machines
5. vm.overcommit_memory=1 -- needed for large sparse CSR allocations

```bash
cd /workspace/v3.3 && bash setup.sh
```

Expected output: "ALL IMPORTS OK" at the end. If tcmalloc/numactl fail to install (no root in container), pip packages still install correctly.

---

## Pipeline Steps with Estimated Times

| Step | What | Output | Est. Time (GPU) | Est. Time (CPU) |
|------|------|--------|-----------------|-----------------|
| 1 | Feature build | features_BTC_15m.parquet | ~2-5 min | ~5-10 min |
| 2 | Cross gen (GPU + memmap, checkpointed) | v2_crosses_BTC_15m.npz | ~1.5-3 hrs | ~6-8 hrs |
| 3 | Optuna search (15% subsample, n_jobs=4) | optuna_configs_all.json | ~2-4 hrs | ~4-8 hrs |
| 4 | Final CPCV (GPU histogram fork) | model_15m.json | ~3-5 hrs | ~10-15 hrs |
| 5 | Meta-labeling + PBO + SHAP | meta_model_15m.pkl + reports | ~30 min | ~30 min |
| 6 | LSTM | lstm_15m.pt + platt_15m.pkl | **Run locally** | 13900K + 3090 |
| **TOTAL (GPU, with Optuna)** | | | **~8-13 hrs** | |
| **TOTAL (CPU, with Optuna)** | | | | **~22-32 hrs** |
| **Without Optuna (GPU)** | | | **~5-9 hrs** | |

---

## Launch Command

```bash
cd /workspace/v3.3 && \
  export SAVAGE22_DB_DIR=/workspace && \
  export V30_DATA_DIR=/workspace/v3.3 && \
  export PYTHONUNBUFFERED=1 && \
  export V2_RIGHT_CHUNK=500 && \
  export V2_BATCH_MAX=500 && \
  nohup python -u cloud_run_tf.py --symbol BTC --tf 15m > /workspace/15m_log.txt 2>&1 &
```

**All env vars explained:**
- `SAVAGE22_DB_DIR=/workspace` -- where V1 DBs live (tweets.db, btc_prices.db, etc.)
- `V30_DATA_DIR=/workspace/v3.3` -- where to read/write parquets, NPZs, models. MUST be v3.3, NOT v3.0!
- `PYTHONUNBUFFERED=1` -- real-time log output (no buffering)
- `V2_RIGHT_CHUNK=500` -- cross gen chunk size (lower to 200 if RAM < 1.5TB)
- `V2_BATCH_MAX=500` -- cap dense intermediate arrays per worker

---

## Verification After Launch (first 30 lines)

```bash
sleep 30 && head -30 /workspace/15m_log.txt
```

Must see:
- "All 16 databases present" or zero "MISS" lines
- Row count: ~227,000
- Feature count: ~4,000+ base cols
- Correct paths: DB_DIR=/workspace, V30_DATA_DIR=/workspace/v3.3
- No "WARNING: DB missing" for any esoteric DB

---

## Verify Cross Features After Cross Gen

After cross gen completes and training starts:
```bash
grep -E "Sparse crosses loaded|cross.*cols|feature_cols.*len" /workspace/15m_log.txt
```

**Expected:** ~10M+ cross feature cols. If significantly less, old cross_names JSON may not have been deleted.

```bash
python3 -c "
import scipy.sparse as sp, json, os
X = sp.load_npz('/workspace/v3.3/v2_crosses_BTC_15m.npz')
print(f'NPZ shape: {X.shape}')
print(f'NNZ: {X.nnz:,}')
print(f'NNZ > int32 max: {X.nnz > 2**31 - 1}')
jp = '/workspace/v3.3/v2_cross_names_BTC_15m.json'
if os.path.exists(jp):
    names = json.load(open(jp))
    print(f'JSON names: {len(names):,}')
    if len(names) != X.shape[1]:
        print(f'MISMATCH! NPZ has {X.shape[1]} cols but JSON has {len(names)} names. DELETE JSON and restart.')
    else:
        print('MATCH OK')
"
```

---

## Verify int64 indptr Applied

After training starts, check the log:
```bash
grep -i "int64\|indptr\|ensure.*sparse.*dtype" /workspace/15m_log.txt
```

Must see confirmation that indptr was cast to int64. If missing, NNZ overflow causes SILENT corruption.

---

## Memory Monitoring During Cross Gen

Cross gen is the most RAM-intensive step. Monitor continuously.

### Watch command (run in separate SSH session):
```bash
watch -n 10 'echo "=== RAM ===" && free -g && echo "" && echo "=== Process ===" && ps aux | grep cloud_run_tf | grep -v grep && echo "" && echo "=== Cgroup ===" && cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null | awk "{printf \"Cgroup used: %.1f GB\n\", \$1/1073741824}" && echo "" && echo "=== Checkpoints ===" && ls -la /workspace/v3.3/_cross_checkpoint_15m_*.npz 2>/dev/null | wc -l && echo "completed cross types"'
```

### What to watch for:
- RAM usage climbing past 80% of cgroup limit -> approaching OOM kill
- If OOM killed, process disappears silently -- check `dmesg | tail -20` for "oom-kill"
- Cross gen logs batch progress: `Parallel cross: N pairs, M batches, T threads`
- Checkpoint count increasing = cross types completing successfully
- If single-threaded (load avg ~1.0 on 128+ core machine), something is wrong

---

## Download Partial Results After Each Step (MACHINES DIE)

vast.ai machines die without warning. Download after EVERY critical step.

### After feature build (Step 1):
```bash
scp -i ~/.ssh/vast_key -P {PORT} root@{HOST}:/workspace/v3.3/features_BTC_15m.parquet ./v3.3/
```

### After cross gen (Step 2) -- THIS IS THE BIG ONE:
```bash
# NPZ is 150-220GB. Start download IMMEDIATELY when cross gen finishes.
scp -i ~/.ssh/vast_key -P {PORT} root@{HOST}:/workspace/v3.3/v2_crosses_BTC_15m.npz ./v3.3/
scp -i ~/.ssh/vast_key -P {PORT} root@{HOST}:/workspace/v3.3/v2_cross_names_BTC_15m.json ./v3.3/
scp -i ~/.ssh/vast_key -P {PORT} root@{HOST}:/workspace/v3.3/v2_base_BTC_15m.parquet ./v3.3/
```
**WARNING:** 150-220GB transfer. At 1 Gbps = ~30 min. At 100 Mbps = ~5 hrs. If machine dies during training, you only lose training -- cross gen is preserved locally.

**Alternative: download checkpoints incrementally** during cross gen:
```bash
# Check which checkpoints exist
ssh -i ~/.ssh/vast_key -p {PORT} root@{HOST} "ls -lh /workspace/v3.3/_cross_checkpoint_15m_*.npz 2>/dev/null"
```

### After Optuna (Step 3):
```bash
scp -i ~/.ssh/vast_key -P {PORT} root@{HOST}:/workspace/v3.3/optuna_configs_all.json ./v3.3/
```

### After training (Step 4):
```bash
scp -i ~/.ssh/vast_key -P {PORT} root@{HOST}:/workspace/v3.3/model_15m.json ./v3.3/
scp -i ~/.ssh/vast_key -P {PORT} root@{HOST}:/workspace/v3.3/cpcv_oos_15m.pkl ./v3.3/
```

### After meta-labeling + PBO (Step 5):
```bash
scp -i ~/.ssh/vast_key -P {PORT} root@{HOST}:/workspace/v3.3/meta_model_15m.pkl ./v3.3/
scp -i ~/.ssh/vast_key -P {PORT} root@{HOST}:/workspace/v3.3/validation_report_15m.json ./v3.3/
scp -i ~/.ssh/vast_key -P {PORT} root@{HOST}:/workspace/v3.3/feature_importance_*.json ./v3.3/
```

### Download everything at once (if machine is still alive):
```bash
ssh -i ~/.ssh/vast_key -p {PORT} root@{HOST} "cd /workspace/v3.3 && tar czf /workspace/15m_results.tar.gz \
  model_15m.json optuna_configs_all.json meta_model_15m.pkl cpcv_oos_15m.pkl \
  validation_report_15m.json feature_importance_*.json \
  v2_cross_names_BTC_15m.json features_BTC_15m.parquet 2>/dev/null"
scp -i ~/.ssh/vast_key -P {PORT} root@{HOST}:/workspace/15m_results.tar.gz ./v3.3/
# NOTE: NPZ not included in tar (150-220GB). Download separately.
scp -i ~/.ssh/vast_key -P {PORT} root@{HOST}:/workspace/v3.3/v2_crosses_BTC_15m.npz ./v3.3/
```

---

## Monitor Commands

```bash
# Live log tail
tail -f /workspace/15m_log.txt

# Check for errors
grep -iE "error|traceback|fail|critical|exception" /workspace/15m_log.txt

# Check pipeline progress (which step is running)
grep -E "Step [0-9]|DONE|RUNNING|COMPLETE|Fold [0-9]" /workspace/15m_log.txt

# Check system resources
uptime && free -g && df -h /workspace

# Check for OOM kills
dmesg | tail -20 | grep -i oom

# Check cross gen progress (checkpoint count)
ls -la /workspace/v3.3/_cross_checkpoint_15m_*.npz 2>/dev/null

# Check training fold progress
grep -E "Fold [0-9]|fold.*complete|accuracy|int64" /workspace/15m_log.txt

# Verify multi-threaded execution (load avg should be > cores x 0.3)
uptime

# Check process RSS
ps aux | grep cloud_run_tf | grep -v grep | awk '{print $6/1024/1024 " GB RSS"}'
```

---

## Verify Multi-Threaded Training

After CPCV training starts, check load average:
```bash
uptime
```

**Expected:** load avg > (total_cores x 0.3). For a 128-core machine, load avg > 38.
If load avg is ~1.0, training is SINGLE-THREADED. Check:

1. **is_enable_sparse mismatch:** 15m stays sparse (dense is 8.5TB). `is_enable_sparse=True` is correct. Sparse histogram builder works multi-core with EFB disabled.
2. **OMP_NUM_THREADS not set:** LightGBM defaults to 1 thread without this.
3. **enable_bundle still True:** If EFB is running on 10M+ features, it will appear hung. Verify `enable_bundle=False` in log.

---

## Optuna on Same Machine (MANDATORY for 15m)

### Why Same Machine
- NPZ is 150-220GB -- impractical to transfer
- Optuna uses 15% row subsample (227K -> ~34K rows) for search -- much faster than full training
- GPU for final retrain only (after search selects best config)
- Warm-started from 1h: fewer trials needed than cold start

### Required Files for Optuna
All already on the machine after cross gen + training:
```
features_BTC_15m.parquet         # base features
v2_crosses_BTC_15m.npz           # cross features (150-220GB -- already on disk)
v2_cross_names_BTC_15m.json      # cross feature column names
optuna_configs_1h.json           # warm-start cascade source (upload from 1h training)
model_15m.json                   # trained model (warm-start seed, if available)
All 16 .db files                 # already uploaded
```

### Upload 1h Optuna Results for Warm-Start
```bash
scp -i ~/.ssh/vast_key -P {PORT} ./v3.3/optuna_configs_1h.json root@{HOST}:/workspace/v3.3/
```

---

## Deployment Steps (Step-by-Step)

### Step 1: Pick Machine from vast.ai
User picks personally. Filter criteria:
```bash
vastai search offers 'gpu_ram >= 24 cpu_cores_effective >= 128 cpu_ram >= 1500 reliability > 0.95 cuda_vers >= 12.0 cuda_vers < 13.0' -o 'dph'
```
Prefer: CUDA 12.x, 1.5TB+ RAM, 128+ cores, GPU (A100/H100/4090), high CPU Score (cores x GHz).

### Step 2: SSH and Run Setup
```bash
SSH="ssh -i ~/.ssh/vast_key -o StrictHostKeyChecking=no"
$SSH -p {PORT} root@{HOST} "mkdir -p /workspace/v3.3"
```

### Step 3: Upload Code + DBs
```bash
# Upload code tar
scp -i ~/.ssh/vast_key -o StrictHostKeyChecking=no -P {PORT} /tmp/v33_code.tar.gz root@{HOST}:/workspace/
$SSH -p {PORT} root@{HOST} "cd /workspace && tar xzf v33_code.tar.gz -C v3.3/"

# Upload DB tar
scp -i ~/.ssh/vast_key -o StrictHostKeyChecking=no -P {PORT} /tmp/v33_dbs.tar.gz root@{HOST}:/workspace/
$SSH -p {PORT} root@{HOST} "cd /workspace && tar xzf v33_dbs.tar.gz && ln -sf /workspace/*.db /workspace/v3.3/"

# Upload large DBs separately (btc_prices.db, multi_asset_prices.db)
scp -i ~/.ssh/vast_key -P {PORT} "C:/Users/C/Documents/Savage22 Server/v3.3/btc_prices.db" root@{HOST}:/workspace/v3.3/
scp -i ~/.ssh/vast_key -P {PORT} "C:/Users/C/Documents/Savage22 Server/v3.3/multi_asset_prices.db" root@{HOST}:/workspace/v3.3/
$SSH -p {PORT} root@{HOST} "ln -sf /workspace/v3.3/btc_prices.db /workspace/ && ln -sf /workspace/v3.3/multi_asset_prices.db /workspace/"

# Upload 1h Optuna config for warm-start cascade
scp -i ~/.ssh/vast_key -P {PORT} "C:/Users/C/Documents/Savage22 Server/v3.3/optuna_configs_1h.json" root@{HOST}:/workspace/v3.3/
```

### Step 4: Run Setup Script
```bash
$SSH -p {PORT} root@{HOST} "cd /workspace/v3.3 && bash setup.sh"
```

### Step 5: Verify DBs
Run the verify script from the "Required Databases" section above.

### Step 6: Nuclear Clean
Run the nuclear clean script above.

### Step 7: Verify Parquet Freshness
Run the parquet check above.

### Step 8: Launch Pipeline
Run the launch command above.

### Step 9: Verify First 30 Lines
```bash
sleep 30 && $SSH -p {PORT} root@{HOST} "head -30 /workspace/15m_log.txt"
```

### Step 10: Monitor
Run monitor commands in a separate SSH session. Check every 30 seconds during cross gen.

### Step 11: Download After Each Step
Follow the download commands above after each critical step completes.

---

## Failure Modes and What to Check

| Symptom | Cause | Fix |
|---------|-------|-----|
| "WARNING: DB missing" in log | Missing database file | Upload the missing .db, re-run verify script |
| OOM during cross gen | V2_RIGHT_CHUNK too large | Lower to 200 (or 100 if < 1.5TB RAM). Checkpoints preserved. |
| "ModuleNotFoundError: astrology_engine" | astrology_engine.py not in v3.3/ | Copy from project root |
| Feature count mismatch | Stale parquet from old feature_library.py | Delete features_BTC_15m.parquet, restart |
| V30_DATA_DIR shows v3.0 path | Env var not set | Verify `export V30_DATA_DIR=/workspace/v3.3` |
| NNZ overflow (silent corruption) | int32 indptr on >2B NNZ | Verify `_ensure_lgbm_sparse_dtypes()` applied. Check log for int64 indptr. |
| Cross gen very slow (>12 hrs) | 227K rows x 10M+ crosses | Expected on CPU. GPU+memmap should be 1.5-3 hrs. Check GPU utilization. |
| Process disappears silently | OOM kill | Check `dmesg | tail -20 | grep -i oom`. Lower RIGHT_CHUNK. Checkpoints safe. |
| LSTM crashes with NaN | Features have NaN not imputed | Run LSTM locally (13900K + 3090). ml_multi_tf.py imputes NaN->0 for LSTM. |
| enable_bundle hanging | EFB scan on 10M+ features | Verify enable_bundle=False in config/log. |
| Single-threaded training (load ~1.0) | OMP_NUM_THREADS not set or enable_bundle=True | Check log for thread count. Verify enable_bundle=False. |
| Cross names JSON truncated | Old JSON left over from previous run | Delete BOTH npz AND json, delete checkpoints, restart. |
| Optuna missing warm-start | optuna_configs_1h.json not uploaded | Upload from 1h training results. |

---

## v3.2 Baseline

**NONE.** 15m has never completed training in any previous version. This is the first-ever 15m model.

Expected v3.3 targets (based on other TF patterns):
- Accuracy: 55-58% (more noise at 15m, but 227K rows help generalization)
- PBO: DEPLOY
- Any validated result is a milestone.

---

## LAUNCH CHECKLIST

1. [ ] Machine with 1.5TB+ cgroup RAM rented (user picked, CUDA 12.x, GPU present)
2. [ ] Setup script run (tcmalloc, THP, deps installed)
3. [ ] All 16 DBs verified present (verify script shows zero MISS)
4. [ ] Nuclear clean run (all old artifacts deleted)
5. [ ] Parquet freshness verified (or no parquet = will build fresh)
6. [ ] optuna_configs_1h.json uploaded (warm-start cascade)
7. [ ] Environment variables set (V2_RIGHT_CHUNK=500, V2_BATCH_MAX=500)
8. [ ] Pipeline launched with PYTHONUNBUFFERED=1
9. [ ] First 30 lines verified (row count ~227K, all DBs present, correct paths)
10. [ ] Memory monitoring active in separate SSH session
11. [ ] Download plan ready for each step (cross gen NPZ is priority)

---

## STATUS
**READY** -- (6,2)=15 splits, GPU histogram fork, Optuna warm-started from 1h. Single machine. User picks from vast.ai.
