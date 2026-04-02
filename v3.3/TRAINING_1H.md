# 1H Training Guide — V3.3
# Copy-paste ready for a fresh Claude session. Read TRAINING_PLAN.md first.

Status: reference/caveat doc only. Maintained deployment authority is `CLOUD_DEPLOYMENT_FRAMEWORK.md`, `CLOUD_1H_PROFILE.md`, `deploy_1h.sh`, and `contracts/deploy_profiles.json`.

---

## Machine Requirements
- **RAM:** 768GB+ MINIMUM (ideally 1TB+). Cross gen needs streaming/memmap for RAM management.
- **Cores:** 128+ (cross gen + Optuna n_jobs=4 + LightGBM OpenMP)
- **CPU Score:** 400+ (cores x GHz). Higher = faster cross gen and Optuna search.
- **Disk:** 80GB+
- **GPU:** REQUIRED with CUDA 12.x (driver < 580). GPU histogram fork is 2.5x faster than CPU at 91K rows.
- **RIGHT_CHUNK:** `export V2_RIGHT_CHUNK=500` (default in cloud_run_tf.py). Lower to 300 if RAM < 1TB.

### CUDA 12.x vs CUDA 13 (Driver 580+)

| Component | CUDA 12.x (driver < 580) | CUDA 13 (driver 580+) |
|-----------|--------------------------|----------------------|
| Cross gen | GPU (cuSPARSE SpGEMM) | CPU fallback (`ALLOW_CPU=1`) |
| Feature build | GPU (cuDF) | CPU (pandas) — cuDF SEGFAULTs on CUDA 13 |
| CPCV training | GPU histogram fork (2.5x faster) | CPU only (`ALLOW_CPU=1`) — 3-4x slower |
| Optuna search | CPU (n_jobs=4) | CPU (n_jobs=4) — same |
| Optuna final retrain | GPU histogram fork | CPU only — 3-4x slower |

**CUDA 13 machines work but are 3-4x slower overall. STRONGLY prefer CUDA 12.x.**

### Machine Recommendation: Cloud 768GB-1TB+ RAM, GPU with CUDA 12.x
- **vast.ai or Lambda:** A100/H100 80GB, 768GB-1TB RAM, CUDA 12.x, ~$3-5/hr.
- **CPU Score 400+** for cross gen (CPU-bound matmul) and Optuna search (n_jobs=4).
- **SAME machine for Optuna and training.** Upload is ~54GB — impractical to transfer.
- **Cost:** ~$15 without Optuna (GPU), ~$25 with Optuna (GPU), ~$65 with Optuna (CPU).

### GPU vs CPU Per Step
| Step | Engine (CUDA 12.x) | Engine (CUDA 13 / CPU) | Reason |
|------|--------------------|-----------------------|--------|
| Feature build | GPU (cuDF) | CPU (pandas) | Rolling/ewm on GPU. ~30 min GPU, ~60 min CPU. |
| Cross gen | GPU (cuSPARSE SpGEMM) | CPU (scipy sparse) | ~1 hr GPU, ~2-3 hrs CPU. Streaming/memmap for RAM. |
| Optuna search | CPU (n_jobs=4) | CPU (n_jobs=4) | CPU parallel search stage. 20% row subsample (91K -> 18K). |
| Final CPCV | GPU (histogram fork) | CPU | 91K rows = GPU sweet spot. ~2 hrs GPU vs ~5 hrs CPU. |
| Optuna final retrain | GPU (histogram fork) | CPU | Best config retrained with full CPCV. |
| Meta + PBO + SHAP | CPU | CPU | Small overhead. |

### Optuna Strategy
- Run on SAME machine as training — 54GB upload is impractical to transfer.
- Warm-started from 4h: Phase 1 (15 trials, 2-fold CPCV) + Validation Gate (top-2, 4-fold).
- Row subsample 20% for search (91K -> 18K rows). Final retrain uses ALL rows.
- CPU Score 400+ for search stage (n_jobs = total_cores // 8).
- GPU for final retrain only (after search selects best config).

---

## Revised ETAs

### CUDA 12.x Machine (768GB+ RAM, A100/H100 GPU, Score 400+)
| Stage | Time | Status |
|-------|------|--------|
| Feature build (cuDF GPU) | 30 min | PENDING |
| Cross gen (cuSPARSE + streaming) | ~1 hr | PENDING |
| save_binary | 30 min | PENDING |
| CPCV (6,2)=15 splits, GPU hist | ~2 hrs | PENDING |
| Optuna (warm from 4h, n_jobs=4, 20% subsample) | ~3 hrs | PENDING |
| Meta + PBO + SHAP | 20 min | PENDING |
| **TOTAL (GPU)** | **~7 hrs (~$25)** | |
| **Without Optuna (GPU)** | **~4 hrs (~$15)** | |

### CUDA 13 / CPU-Only Machine (768GB+ RAM, Score 400+)
| Stage | Time | Status |
|-------|------|--------|
| Feature build (pandas CPU) | 60 min | PENDING |
| Cross gen (scipy sparse CPU) | ~2-3 hrs | PENDING |
| save_binary | 30 min | PENDING |
| CPCV (6,2)=15 splits, CPU | ~5 hrs | PENDING |
| Optuna (warm from 4h, n_jobs=4, 20% subsample) | ~8 hrs | PENDING |
| Meta + PBO + SHAP | 20 min | PENDING |
| **TOTAL (CPU)** | **~17 hrs (~$65)** | |
| **Without Optuna (CPU)** | **~9 hrs (~$35)** | |

---

## Data
- **Rows:** ~91,000 (1h bars, 2017-08-17 to 2026, Binance BTC/USDT)
- **Base features:** ~3,968 cols
- **Cross features:** ~7-8M expected (min_nonzero=3)
- **Dense matrix:** ~2.3TB (91K rows x 8M features x 4 bytes). IMPOSSIBLE — stays SPARSE.
- **NNZ estimate:** ~8B+ (exceeds int32 limit). int64 indptr handles NNZ > 2^31.
- **Barriers:** tp=2.0 ATR, sl=1.5 ATR, max_hold=24 bars (asymmetric)

---

## Required Databases (ALL 16 -- ZERO MISSING)

```
FROM PROJECT ROOT (-> /workspace/):
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

FROM v3.3/ (-> /workspace/v3.3/):
  multi_asset_prices.db  # 1.3GB -- multi-asset data
  v2_signals.db          # DeFi TVL, BTC dominance, mining stats
```

**Also required (non-DB):**
- `kp_history_gfz.txt` (in /workspace/ or /workspace/v3.3/)
- `astrology_engine.py` (in /workspace/v3.3/ -- feature_library.py imports it)

### Verify ALL 16 DBs (run BEFORE launch):
```bash
FAIL=0
for db in btc_prices.db tweets.db news_articles.db sports_results.db space_weather.db onchain_data.db macro_data.db astrology_full.db ephemeris_cache.db fear_greed.db funding_rates.db google_trends.db open_interest.db llm_cache.db; do
  if [ -f /workspace/$db ] || [ -f /workspace/v3.3/$db ]; then echo "OK   $db"; else echo "MISSING: $db"; FAIL=$((FAIL+1)); fi
done
for db in multi_asset_prices.db v2_signals.db; do
  if [ -f /workspace/v3.3/$db ]; then echo "OK   v3.3/$db"; else echo "MISSING: v3.3/$db"; FAIL=$((FAIL+1)); fi
done
[ -f /workspace/kp_history_gfz.txt ] || [ -f /workspace/v3.3/kp_history_gfz.txt ] && echo "OK   kp_history_gfz.txt" || { echo "MISSING: kp_history_gfz.txt"; FAIL=$((FAIL+1)); }
[ -f /workspace/v3.3/astrology_engine.py ] && echo "OK   astrology_engine.py" || { echo "MISSING: astrology_engine.py"; FAIL=$((FAIL+1)); }
echo ""
echo "DB check: $FAIL missing"
[ $FAIL -eq 0 ] && echo "ALL 16 DBs + kp_history + astrology_engine PRESENT" || { echo "STOP — fix missing files before proceeding"; exit 1; }
```
**ALL must say OK. If ANY says MISSING -> STOP. Upload the missing file first.**

---

## Nuclear Clean (MANDATORY before first run)

Delete ALL old artifacts. Old NPZs were built with min_nonzero=8 and produce fewer features.
Old cross names JSON truncates column count (cross_matrix has N cols but JSON only names M < N).

```bash
cd /workspace && rm -f *.npz *.json *.pkl *.parquet *.log DONE_* RUNNING_* *.lock 2>/dev/null  # NOTE: no *.txt — would kill kp_history_gfz.txt
cd /workspace/v3.3 && rm -f \
  v2_crosses_*.npz v2_cross_names_*.json \
  v2_base_*.parquet features_BTC_*.parquet features_*_all.json \
  model_*.json platt_*.pkl cpcv_oos_*.pkl \
  feature_importance_*.json shap_analysis_*.json validation_report_*.json \
  meta_model_*.pkl ml_multi_tf_*.* optuna_configs_all.json \
  lstm_*.pt DONE_* RUNNING_* *.lock 2>/dev/null
echo '{"steps": {}, "version": "3.3"}' > pipeline_manifest.json
echo "Clean done"
```

**CRITICAL:** Both `v2_crosses_*.npz` AND `v2_cross_names_*.json` must be deleted together.
If you delete the NPZ but leave the old JSON, ml_multi_tf.py loads the JSON and truncates
cross feature columns (e.g., JSON has 6M names but new NPZ has 8M cols -> 2M+ features
get generic `cross_N` names instead of real names, confusing SHAP and audit).

---

## Verify Parquet Freshness
```bash
python3 -c "import pandas as pd; df=pd.read_parquet('/workspace/v3.3/features_BTC_1h.parquet'); print(f'Cols: {len(df.columns)}, Rows: {len(df)}')"
# Expected: ~3,968 columns, ~91,000 rows
```

---

## Pipeline Steps
1. Feature build (~30-60 min) -> `features_BTC_1h.parquet`
2. Cross gen (streaming/memmap, min_nonzero=3) -> `v2_crosses_BTC_1h.npz` + `v2_cross_names_BTC_1h.json`
3. Stays SPARSE (~2.3TB dense impossible). SPARSE + SEQUENTIAL CPCV.
4. LightGBM CPCV ((6,2)=15 splits, sequential) -> `model_1h.json`
5. Optuna (warm from 4h, Phase 1 + Validation Gate) -> `optuna_configs_1h.json`
6. Meta-labeling -> `meta_model_1h.pkl`
7. LSTM -> `lstm_1h.pt` + `platt_1h.pkl`
8. PBO/Audit -> `validation_report_1h.json`

---

## SPARSE TRAINING (1H always stays sparse)

Dense matrix would be ~2.3TB -- impossible on any available machine. 1H ALWAYS stays sparse.
- `is_enable_sparse=True` stays set (default from config.py)
- LightGBM trains on sparse CSR with EFB bundling
- SPARSE + SEQUENTIAL CPCV (parallel disabled for >1M features -- pickle bottleneck)
- int64 indptr handles NNZ > 2^31 (applied by `_ensure_lgbm_sparse_dtypes()`)
- Verify in log: `"Keeping SPARSE (dense would need XXX GB, only YYY GB avail)"`
- Row-partitioned boosting is BANNED (kills rare signals -- Perplexity confirmed)

---

## Install Dependencies
```bash
pip install -q lightgbm scikit-learn scipy ephem astropy pytz joblib pandas numpy \
  pyarrow optuna hmmlearn numba tqdm pyyaml alembic cmaes colorlog sqlalchemy \
  threadpoolctl psutil 2>&1 | tail -3
python -c "import pandas, numpy, scipy, sklearn, lightgbm, ephem, astropy, pyarrow, optuna, numba, hmmlearn, yaml, tqdm; print('ALL OK')"
```

---

## Detect CUDA Version (MANDATORY -- determines GPU vs CPU path)

```bash
python3 -c "
try:
    import subprocess
    out = subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).strip()
    driver = int(out.split('.')[0])
    print(f'Driver: {out} (major: {driver})')
    if driver >= 580:
        print('CUDA 13 DETECTED — GPU cuDF/cuSPARSE will SEGFAULT')
        print('USE: export ALLOW_CPU=1 before launch (3-4x slower)')
        print('Cross gen: CPU scipy sparse matmul')
        print('Training: CPU LightGBM (no GPU histogram fork)')
    else:
        print('CUDA 12.x — FULL GPU PATH AVAILABLE')
        print('Cross gen: GPU cuSPARSE SpGEMM')
        print('Training: GPU histogram fork (2.5x faster)')
except Exception as e:
    print(f'nvidia-smi failed: {e}')
    print('No GPU detected — CPU only')
"
```

---

## Verify Cgroup RAM BEFORE Launch

```bash
python3 -c "
import os
try:
    with open('/sys/fs/cgroup/memory/memory.limit_in_bytes') as f:
        ram = int(f.read().strip())
    print(f'Cgroup RAM: {ram/1e9:.0f} GB')
    if ram < 768e9:
        print(f'WARNING: < 768GB — cross gen may OOM. Need 768GB+ (ideally 1TB+).')
    elif ram < 1000e9:
        print(f'OK — 768GB+ available. Set V2_RIGHT_CHUNK=300 to be safe.')
    else:
        print('OK — 1TB+ available for cross gen. V2_RIGHT_CHUNK=500 is fine.')
except:
    import psutil
    print(f'System RAM: {psutil.virtual_memory().total/1e9:.0f} GB')
"
```

---

## Launch Command

### CUDA 12.x Path (FULL GPU)
```bash
cd /workspace/v3.3 && \
  export SAVAGE22_DB_DIR=/workspace && \
  export V30_DATA_DIR=/workspace/v3.3 && \
  export PYTHONUNBUFFERED=1 && \
  nohup python -u cloud_run_tf.py --symbol BTC --tf 1h > /workspace/1h_log.txt 2>&1 &
```

### CUDA 13 Path (CPU FALLBACK)
```bash
cd /workspace/v3.3 && \
  export SAVAGE22_DB_DIR=/workspace && \
  export V30_DATA_DIR=/workspace/v3.3 && \
  export PYTHONUNBUFFERED=1 && \
  export ALLOW_CPU=1 && \
  nohup python -u cloud_run_tf.py --symbol BTC --tf 1h > /workspace/1h_log.txt 2>&1 &
```

**All env vars explained:**
- `SAVAGE22_DB_DIR=/workspace` -- where V1 DBs live (tweets.db, btc_prices.db, etc.)
- `V30_DATA_DIR=/workspace/v3.3` -- where to read/write parquets, NPZs, models. MUST be v3.3, NOT v3.0!
- `PYTHONUNBUFFERED=1` -- real-time log output (no buffering)
- `ALLOW_CPU=1` -- CUDA 13 only: forces CPU path for cross gen and training
- `OMP_NUM_THREADS` / `NUMBA_NUM_THREADS` -- set dynamically by cloud_run_tf.py per phase

---

## Verify Launch (first 30 seconds)

```bash
sleep 30 && head -30 /workspace/1h_log.txt
```

**Must see:**
- "All 16 databases present" (or zero "MISS" / "WARNING: DB missing" lines)
- Row count: ~91,000 rows
- Base feature count: ~3,968 cols
- Correct data directory: V30_DATA_DIR should show `/workspace/v3.3` not `/workspace/v3.0 (LGBM)`

**For CUDA 12.x:** Also verify GPU detection:
```bash
grep -i "gpu\|cuda\|device" /workspace/1h_log.txt | head -10
```
Should show GPU device recognized and cuSPARSE/histogram fork enabled.

**For CUDA 13:** Verify CPU fallback active:
```bash
grep -i "ALLOW_CPU\|cpu.*fallback\|pandas" /workspace/1h_log.txt | head -10
```
Should show CPU mode active, no cuDF/cuSPARSE attempted.

---

## Verify Cross Features After Cross Gen (Step 2)

```bash
python3 -c "
import scipy.sparse as sp
X = sp.load_npz('/workspace/v3.3/v2_crosses_BTC_1h.npz')
print(f'Cross shape: {X.shape}')
print(f'Expected: ~7-8M columns (min_nonzero=3)')
print(f'NNZ: {X.nnz:,}')
if X.shape[1] < 6_000_000:
    print('WARNING: Fewer crosses than expected — min_nonzero may not be 3')
import json
with open('/workspace/v3.3/v2_cross_names_BTC_1h.json') as f:
    names = json.load(f)
print(f'Cross names: {len(names)}')
assert X.shape[1] == len(names), 'MISMATCH: NPZ cols != JSON names count'
print('OK — NPZ and JSON match')
"
```

---

## Verify Multi-Threaded Execution

After training starts (Step 4 -- CPCV), check load average:
```bash
uptime
# Expected: load average > 30 on 128+ core machine
# If load average ~ 1.0, training is SINGLE-THREADED — critical bug

# 1H always stays sparse (no dense conversion). Check log for "Keeping SPARSE":
grep -i "sparse\|dense\|is_enable_sparse" /workspace/1h_log.txt | tail -5

# Check RSS:
ps aux | grep cloud_run_tf | grep -v grep | awk '{print $6/1024 " MB"}'
```

**Key log lines to look for:**
```
Keeping SPARSE (dense would need XXX GB, only YYY GB avail)
```
This confirms sparse mode is active. 1H NEVER converts to dense.

---

## Monitor Commands

```bash
# Live log tail
tail -f /workspace/1h_log.txt

# Check for errors
grep -iE "error|traceback|fail|critical|exception" /workspace/1h_log.txt

# Check pipeline progress (which step is running)
grep -E "Step [0-9]|DONE|RUNNING|COMPLETE|fold" /workspace/1h_log.txt

# Check process is alive
ps aux | grep cloud_run_tf | grep -v grep

# Check memory usage (stays sparse — no dense conversion)
free -g

# Check disk space
df -h /workspace

# Check load average (should be >> 1.0 during training)
uptime

# Check for OOM kills:
dmesg | tail -20 | grep -i oom

# Check training fold progress
grep -E "Fold [0-9]|fold.*complete|accuracy|split" /workspace/1h_log.txt
```

---

## LightGBM Config (from config.py)

| Parameter | Value | Source |
|-----------|-------|--------|
| min_data_in_leaf | 8 | TF_MIN_DATA_IN_LEAF['1h'] |
| num_leaves | 511 | TF_NUM_LEAVES['1h'] (91K rows -- deep trees viable) |
| max_bin | 255 | V3_LGBM_PARAMS (binary crosses always get 2 bins regardless) |
| CPCV folds | (6,2) = 15 splits, 5 unique paths, 67% train | TF_CPCV_GROUPS['1h'] |
| feature_pre_filter | False | CRITICAL -- True silently kills rare esoteric features |
| is_enable_sparse | True | Stays sparse (dense = 2.3TB impossible) |
| force_col_wise | True | Required for sparse CSR with EFB |
| enable_bundle | False | EFB intractable at 7-8M features with 91K rows — skips bundle scan |
| path_smooth | 2.0 | Regularization for sparse signal stability |
| Barriers | tp=2.0 ATR, sl=1.5 ATR, max_hold=24 bars | TRIPLE_BARRIER_CONFIG['1h'] (asymmetric) |

---

## Expected Feature Count (min_nonzero=3)

| Component | Count |
|-----------|-------|
| Base features | ~3,968 |
| Cross features (min_nonzero=3) | ~7-8M (up from ~6M at min_nonzero=8) |
| Total | ~7-8M |

The increase from min_nonzero=8 to min_nonzero=3 preserves rare esoteric crosses that
fire 3-7 times in the dataset. These rare signals ARE the edge. LightGBM's
min_gain_to_split=2.0 guards against noise from low-support features.

---

## Optuna Config (from config.py)

| Parameter | Value |
|-----------|-------|
| Phase 1 trials | 25 (2 seeded + 8 random + 15 TPE) |
| Phase 1 CPCV | 2-fold (fast search) |
| Phase 1 rounds | 60 max (ES fires at ~30) |
| Phase 1 LR | 0.15 (5x final for fast convergence) |
| Validation top-K | 3 (re-eval with 4-fold CPCV) |
| Warm-start from 4h | Phase 1: 15 trials, Validation: top-2 |
| Row subsample | 20% (91K -> 18K rows) for search only |
| Final retrain | Full rows, LR=0.03, 800 rounds, full CPCV (6,2) |

---

## v3.2 Baseline (targets to beat)
| Metric | v3.2 Value |
|--------|-----------|
| Accuracy | 58.1% (OOS mean across 15 CPCV paths) |
| Dir Acc @>70% conf | 84.9% |
| Meta AUC | 0.648 |
| PBO | DEPLOY (0.14) |
| Cross Features | 6,061,813 (min_nonzero=8) |

---

## Failure Modes and What to Check

| Symptom | Cause | Fix |
|---------|-------|-----|
| "WARNING: DB missing" in log | Missing database file | Upload the missing .db, re-run verify script |
| Cross cols < 6M | Old v2_cross_names JSON left over (min_nonzero=8) | Delete BOTH npz AND json, restart |
| OOM during cross gen | V2_RIGHT_CHUNK too large for RAM | Set V2_RIGHT_CHUNK=300 (lower to 200 if still OOMing) |
| cuDF/cuSPARSE SEGFAULT | CUDA 13 (driver 580+) | Set ALLOW_CPU=1, restart |
| "ModuleNotFoundError: astrology_engine" | astrology_engine.py not in v3.3/ | Copy from project root |
| Feature count mismatch | Stale parquet from old feature_library.py | Delete features_BTC_1h.parquet, restart |
| V30_DATA_DIR shows v3.0 path | Env var not set | Verify `export V30_DATA_DIR=/workspace/v3.3` |
| Load avg ~1.0 during training | Single-threaded sparse training | Check OMP_NUM_THREADS set. Sparse sequential CPCV is expected behavior -- load avg is per-fold, not overall. |
| Cross gen very slow (>5 hrs) | CPU fallback on CUDA 13 | Expected on CPU. GPU cuSPARSE is ~2x faster. |
| LSTM crashes with NaN | Features have NaN not imputed for LSTM | ml_multi_tf.py imputes NaN->0 for LSTM only. Check log. |
| NNZ overflow (silent corruption) | int32 indptr on >2B NNZ | Verify `_ensure_lgbm_sparse_dtypes()` applied. Check log for int64 indptr. |
| Optuna crashes | Missing optuna/cmaes/colorlog | Run full pip install command |
| "No module named 'lightgbm'" | pip install not run | Run install command above |

---

## OPTUNA DEPLOYMENT

### Why Same Machine (RECOMMENDED)
- **Total upload if separate machine: ~54 GB** (parquet + NPZ ~45-50GB + cross_names JSON + all DBs)
- At 100 Mbps upload = ~1.2 hrs transfer. At 1 Gbps = ~7 min.
- **RECOMMENDED: Keep Optuna on the same machine as training.** Avoids 54GB transfer entirely.
- Pick a value machine that handles BOTH cross gen/CPCV AND Optuna.

### Optuna Timing
| Scenario | Time |
|----------|------|
| Warm from 4h, GPU hist fork (CUDA 12.x) | ~3 hrs |
| Warm from 4h, CPU only (CUDA 13) | ~8 hrs |
| Cold (no warm-start) | Add ~50% more time |

### If Running Optuna on a Separate Machine (NOT recommended)
Upload these files:
```
features_BTC_1h.parquet          # base feature parquet
v2_crosses_BTC_1h.npz            # cross feature sparse matrix (~45-50 GB)
v2_cross_names_BTC_1h.json       # cross feature column names
lgbm_dataset_1h.bin              # save_binary output (if available — skips EFB rebuild)
model_1h.json                    # trained model (warm-start seed)
optuna_configs_4h.json           # 4h Optuna results (warm-start cascade source)
All 16 .db files + kp_history_gfz.txt + astrology_engine.py
v33_code.tar.gz                  # all v3.3/*.py code
```
Total: **~54 GB.** This is why same machine is recommended.

---

## Download Results When Done

**Download partial results after each critical step (vast.ai machines die without warning).**
Especially download `model_1h.json` immediately after Step 4 completes -- this is the
most valuable artifact and takes the longest to produce.

```bash
# Check if pipeline completed
grep "DONE\|pipeline complete\|All steps" /workspace/1h_log.txt

# Download all artifacts
scp -P {PORT} root@{HOST}:/workspace/v3.3/model_1h.json .
scp -P {PORT} root@{HOST}:/workspace/v3.3/optuna_configs_1h.json .
scp -P {PORT} root@{HOST}:/workspace/v3.3/meta_model_1h.pkl .
scp -P {PORT} root@{HOST}:/workspace/v3.3/lstm_1h.pt .
scp -P {PORT} root@{HOST}:/workspace/v3.3/platt_1h.pkl .
scp -P {PORT} root@{HOST}:/workspace/v3.3/validation_report_1h.json .
scp -P {PORT} root@{HOST}:/workspace/v3.3/feature_importance_*.json .
scp -P {PORT} root@{HOST}:/workspace/v3.3/inference_1h_base_cols.json .
scp -P {PORT} root@{HOST}:/workspace/v3.3/inference_1h_cross_names.json .
scp -P {PORT} root@{HOST}:/workspace/v3.3/inference_1h_cross_pairs.npz .
scp -P {PORT} root@{HOST}:/workspace/v3.3/inference_1h_ctx_names.json .
scp -P {PORT} root@{HOST}:/workspace/v3.3/inference_1h_thresholds.json .
scp -P {PORT} root@{HOST}:/workspace/1h_log.txt .
```

---

## Deployment Steps (step-by-step)

### Step 1: Detect CUDA version on target machine
```bash
nvidia-smi --query-gpu=driver_version --format=csv,noheader
# If major >= 580: CUDA 13 — use ALLOW_CPU=1 path
# If major < 580: CUDA 12.x — full GPU path
```

### Step 2: Rent machine (vast.ai example)
```bash
# CUDA 12.x preferred:
vastai search offers 'cpu_cores_effective >= 128 cpu_ram >= 768 gpu_ram >= 40 reliability > 0.95 driver_version < 580 cuda_vers >= 12.0 dph <= 5.0' -o 'cpu_cores_effective-'

# If CUDA 12.x unavailable (fallback to CUDA 13 — 3-4x slower):
vastai search offers 'cpu_cores_effective >= 128 cpu_ram >= 768 reliability > 0.95 dph <= 5.0' -o 'cpu_cores_effective-'
```
Pick machine with highest CPU Score (cores x GHz). Ask user before renting.

### Step 3: Upload code + DBs
```bash
SSH="ssh -i ~/.ssh/vast_key -o StrictHostKeyChecking=no"
scp -i ~/.ssh/vast_key -o StrictHostKeyChecking=no -P {PORT} /tmp/v33_code.tar.gz root@{HOST}:/workspace/
$SSH -p {PORT} root@{HOST} "cd /workspace && tar xzf v33_code.tar.gz -C v3.3/"
scp -i ~/.ssh/vast_key -o StrictHostKeyChecking=no -P {PORT} /tmp/v33_dbs.tar.gz root@{HOST}:/workspace/
$SSH -p {PORT} root@{HOST} "cd /workspace && tar xzf v33_dbs.tar.gz && ln -sf /workspace/*.db /workspace/v3.3/"
scp -i ~/.ssh/vast_key -P {PORT} "C:/Users/C/Documents/Savage22 Server/v3.3/btc_prices.db" root@{HOST}:/workspace/v3.3/
scp -i ~/.ssh/vast_key -P {PORT} "C:/Users/C/Documents/Savage22 Server/v3.3/multi_asset_prices.db" root@{HOST}:/workspace/v3.3/
$SSH -p {PORT} root@{HOST} "ln -sf /workspace/v3.3/btc_prices.db /workspace/ && ln -sf /workspace/v3.3/multi_asset_prices.db /workspace/"
```

### Step 4: Verify DBs (see verify script above)

### Step 5: Verify cgroup RAM (see verify script above)

### Step 6: Detect CUDA (see detect script above)

### Step 7: Nuclear clean (see clean script above)

### Step 8: Install deps (see install command above)

### Step 9: Launch pipeline
- CUDA 12.x: use GPU launch command
- CUDA 13: use CPU fallback launch command (with `ALLOW_CPU=1`)

### Step 10: Verify first 30 lines (see verify launch above)

### Step 11: Monitor (see monitor commands above)

### Step 12: Download results after each critical step

---

## STATUS
**PENDING** -- Waiting for 4h Optuna to complete (warm-start cascade: 1w -> 1d -> 4h -> 1h).

---

## Notes
- min_data_in_leaf=8 for 1h (per TF_MIN_DATA_IN_LEAF in config.py)
- CPCV (6,2) = 15 splits, 5 unique paths -- real PBO with combinatorial diversity.
- NNZ will exceed int32 limit (~2B). int64 indptr fixes NNZ > 2^31.
- Row-partitioned boosting is BANNED (kills rare signals -- Perplexity confirmed).
- Subsampling code REMOVED from ml_multi_tf.py. Training uses ALL 91K rows. No data loss.
- Dense matrix (~2.3TB) is impossible -- training ALWAYS stays sparse for 1h.
- Download ALL artifacts after each step -- vast.ai machines die without warning.
- GPU histogram fork provides 2.5x speedup at 91K rows (CUDA 12.x only).
- Optuna row subsample (20%) is for SEARCH phase only -- final retrain uses ALL rows.
