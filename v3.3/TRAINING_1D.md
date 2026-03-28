# 1D Training Guide — V3.3
# Copy-paste ready for a fresh Claude session. Read TRAINING_PLAN.md first.

---

## Machine Requirements
- **RAM:** 512GB+ MINIMUM (944GB confirmed working, 503GB OOM'd at RC=500)
- **Cores:** 128+ (SPARSE + SEQUENTIAL CPCV + cross gen benefit from high core count)
- **CPU Score:** 400+ (cores x GHz). Cross gen + training both CPU-bound.
- **Disk:** 50GB+
- **RIGHT_CHUNK:** `export V2_RIGHT_CHUNK=200` (MANDATORY — auto=2000 OOMs, 500 OOMs on 503GB)

### KNOWN BOTTLENECK: Parallel CPCV Pickle Serialization
With 6M features, parallel CPCV via ProcessPoolExecutor is a TRAP:
- Dense conversion (138GB) fits in RAM → code converts sparse→dense
- Parallel path converts dense→sparse for pickle transport → each worker converts sparse→dense again
- 4 workers × ~100GB pickle data = ~400GB through one IPC pipe = **hours wasted**
- Observed: load 1.04 on 192 cores, 8+ hours with no fold results
- **FIX NEEDED:** For 1M+ features, stay sparse OR train CPCV sequentially (no ProcessPoolExecutor)
- Perplexity confirmed: sparse histogram is O(2 × NNZ), efficient for 99% sparse binary crosses
- LightGBM docs warn: num_threads > 64 on < 10K rows causes poor scaling

### Machine Recommendation: LOCAL or cheap cloud (128c, 256GB+ RAM)
CPU training faster than GPU at 5,733 rows. Marginal GPU benefit. ~$4 without Optuna, ~$12 with.
- **Local option:** 13900K + 3090 IF you have 256GB+ RAM. Otherwise rent cheap cloud.
- **Cloud option:** vast.ai 128c CPU machine, ~$1.75/hr. No GPU needed for training.
- **GPU:** Not needed (LightGBM CPU-only for sparse/dense training at this row count)

### GPU vs CPU Per Step
| Step | Engine | Reason |
|------|--------|--------|
| Feature build | GPU (cuDF) | Rolling/ewm on GPU. ~15 min. |
| Cross gen | GPU (cuSPARSE SpGEMM) | 10 min GPU. DONE (6M features). |
| Optuna search | CPU (n_jobs=4) | 5,733 rows = marginal GPU benefit. 4 parallel TPE trials on CPU. |
| Final CPCV | CPU | Sparse + sequential. ~2 hrs (~30 min/fold). |
| Trade optimizer | GPU | cuDF-accelerated parameter sweep. |

### Optuna Machine Strategy
- CPU-heavy machine (128+ cores, no GPU needed for Optuna). CPU Score 1000+ ideal.
- n_jobs=4 workers. Each trial trains on sparse CSR with 5,733 rows.
- Warm-started from 1w: 50+30 trials (vs 100+50 cold). ~4 hrs.
- Same machine for Optuna and training (no need to separate).
- Separate Optuna machine only if parallelizing across TFs (run 1d Optuna while 4h cross gen runs elsewhere).

### Revised ETAs (Machine A: 128c CPU, Score 1000+)
| Stage | Time | Status |
|-------|------|--------|
| Feature build | 15 min | PENDING |
| Cross gen (cuSPARSE SpGEMM) | 10 min | DONE (6M features) |
| save_binary | 5 min (~0.9-1.2GB file) | PENDING |
| CPCV (4 folds, sparse seq) | 2 hrs (~30 min/fold) | PENDING |
| Optuna (50+30 warm, n_jobs=4, pruning+ES fix) | ~4 hrs | PENDING |
| Meta + PBO + SHAP | 15 min | PENDING |
| **TOTAL** | **~7 hrs ($12)** | |
| **Without Optuna** | **~2.5 hrs ($4)** | |

---

## Data
- **Rows:** 5,733 (daily bars, 2010-2026)
- **Base features:** ~3,756 cols
- **Cross features:** 6,039,797 crosses confirmed (6,043,553 total)
- **NNZ:** 497,919,431 NNZ, 1.4% density
- **Total:** ~6.04M features
- **Dense matrix size:** 138.6GB observed (5,733 x 6.04M x 4 bytes). Needs 512GB+ RAM.
- **min_data_in_leaf:** 3 (rare esoteric signals fire 10-20x on daily)
- **NPZ from v3.2/v3.3 cloud_results:** STALE (built with min_nonzero=8). Must regenerate.

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
cross feature columns (e.g., JSON has 4.46M names but new NPZ has 5.5M cols -> 1M+ features
get generic `cross_N` names instead of real names, confusing SHAP and audit).

---

## Verify Parquet Freshness
```bash
python3 -c "import pandas as pd; df=pd.read_parquet('/workspace/v3.3/features_BTC_1d.parquet'); print(f'Cols: {len(df.columns)}, Rows: {len(df)}')"
# Expected: columns should match feature_library.py output count
```

## Pipeline Steps
1. Feature build (~40s) -> `features_BTC_1d.parquet`
2. Cross gen (~30-45 min, min_nonzero=3) -> `v2_crosses_BTC_1d.npz` + `v2_cross_names_BTC_1d.json`
3. LightGBM CPCV (SPARSE + SEQUENTIAL — parallel disabled for >1M features) -> `model_1d.json`
4. Optuna (200 trials) -> `optuna_configs_1d.json` (have v3.2 config as starting point)
5. Meta-labeling -> `meta_model_1d.pkl`
6. LSTM -> `lstm_1d.pt` + `platt_1d.pkl`
7. PBO/Audit -> `validation_report_1d.json`

---

## Install Dependencies
```bash
pip install -q lightgbm scikit-learn scipy ephem astropy pytz joblib pandas numpy \
  pyarrow optuna hmmlearn numba tqdm pyyaml alembic cmaes colorlog sqlalchemy \
  threadpoolctl 2>&1 | tail -3
python -c "import lightgbm; print(f'LightGBM {lightgbm.__version__} OK')"
```

---

## Launch Command

```bash
cd /workspace/v3.3 && \
  export SAVAGE22_DB_DIR=/workspace && \
  export V30_DATA_DIR=/workspace/v3.3 && \
  export PYTHONUNBUFFERED=1 && \
  nohup python -u cloud_run_tf.py --symbol BTC --tf 1d > /workspace/1d_log.txt 2>&1 &
```

**All env vars explained:**
- `SAVAGE22_DB_DIR=/workspace` -- where V1 DBs live (tweets.db, btc_prices.db, etc.)
- `V30_DATA_DIR=/workspace/v3.3` -- where to read/write parquets, NPZs, models. MUST be v3.3, NOT v3.0!
- `PYTHONUNBUFFERED=1` -- real-time log output (no buffering)
- `OMP_NUM_THREADS` / `NUMBA_NUM_THREADS` -- set dynamically by cloud_run_tf.py per phase (cross gen vs training vs Optuna)

---

## Verify Launch (first 30 seconds)

```bash
sleep 30 && head -30 /workspace/1d_log.txt
```

**Must see:**
- "All 16 databases present" (or zero "MISS" lines)
- Row count: ~5,733 rows
- Base feature count: ~3,756 cols
- Correct data directory: V30_DATA_DIR should show `/workspace/v3.3` not `/workspace/v3.0 (LGBM)`

---

## Verify Cross Features Loaded Correctly

After cross gen completes and training starts, check the log for:
```bash
grep -E "Sparse crosses loaded|cross.*cols|feature_cols.*len" /workspace/1d_log.txt
```

**Expected:** ~6.04M cross feature cols loaded (min_nonzero=3). If you see ~4.46M (the old
min_nonzero=8 count), the old cross names JSON was not deleted. STOP, delete both
`v2_crosses_BTC_1d.npz` and `v2_cross_names_BTC_1d.json`, and restart.

**How to double-check:**
```bash
# Count columns in the NPZ (quick Python check):
python3 -c "
import scipy.sparse as sp
X = sp.load_npz('/workspace/v3.3/v2_crosses_BTC_1d.npz')
print(f'NPZ shape: {X.shape}')
import json, os
jp = '/workspace/v3.3/v2_cross_names_BTC_1d.json'
if os.path.exists(jp):
    names = json.load(open(jp))
    print(f'JSON names: {len(names)}')
    if len(names) != X.shape[1]:
        print(f'MISMATCH! NPZ has {X.shape[1]} cols but JSON has {len(names)} names. DELETE JSON and restart.')
    else:
        print('MATCH OK')
"
```

---

## Verify Multi-Threaded Training

After CPCV training starts (Step 3 in log), check load average:

```bash
# On the machine:
uptime
# Or:
cat /proc/loadavg
```

**Expected:** load avg > (total_cores x 0.3). For a 128-core machine, load avg should be > 38.
If load avg is ~1.0, training is SINGLE-THREADED. Check:

1. **is_enable_sparse mismatch:** If data was converted to dense but `is_enable_sparse` is still True,
   LightGBM's sparse histogram builder serializes OpenMP -> single-threaded.
   Look for `"is_enable_sparse=False"` in the log. If missing and data is dense, this is the bug.

2. **OMP_NUM_THREADS not set:** LightGBM defaults to 1 thread without this env var.

3. **RSS too small:** Dense 1d matrix should be ~138.6GB.
   ```bash
   ps aux | grep cloud_run_tf | grep -v grep | awk '{print $6/1024 " MB"}'
   ```
   If RSS is < 20GB on a 512GB machine, the sparse->dense conversion did not happen.

**Key log lines to look for:**
```
is_enable_sparse=False (data converted to dense -- enables multi-core LightGBM)
```
or
```
Keeping SPARSE (dense would need XXX GB, only YYY GB avail)
```
If keeping sparse: training is slower per fold but sequential CPCV trains one fold at a time.

---

## SPARSE + SEQUENTIAL CPCV (parallel disabled for >1M features — pickle bottleneck)

For 1d with 6M+ features, parallel CPCV via ProcessPoolExecutor is disabled:
- Dense conversion (138.6GB) may fit in RAM, but parallel path converts dense->sparse for pickle transport, then each worker converts sparse->dense again
- 4 workers x ~100GB pickle data = ~400GB through one IPC pipe = hours wasted
- **FIX:** Sequential CPCV (one fold at a time) with sparse CSR input
- LightGBM trains on sparse CSR with EFB bundling — slower per fold but no pickle overhead
- num_threads capped to 32 for <10K rows (active cap, not just warning)

---

## is_enable_sparse Lesson (CRITICAL)

`config.py` has `is_enable_sparse: True` as the default LightGBM param.
When ml_multi_tf.py converts sparse to dense (because RAM allows it), it MUST set
`is_enable_sparse=False`. Otherwise LightGBM assumes sparse input format on dense data,
causing SINGLE-THREADED training (the sparse histogram builder serializes OpenMP).

This is handled automatically in ml_multi_tf.py line ~780:
```python
if _converted_to_dense:
    _base_lgb_params['is_enable_sparse'] = False
```

Verify by checking the log for "is_enable_sparse=False". If this line is missing and data
is dense, training will be 10-50x slower than expected.

---

## Monitor Commands

```bash
# Live log tail
tail -f /workspace/1d_log.txt

# Check for errors
grep -iE "error|traceback|fail|critical|exception" /workspace/1d_log.txt

# Check pipeline progress (which step is running)
grep -E "Step [0-9]|DONE|RUNNING|COMPLETE" /workspace/1d_log.txt

# Check system resources
uptime && free -h && df -h /workspace

# Check for OOM kills:
dmesg | tail -20 | grep -i oom

# Check if cross gen is still running (CPU-bound, watch load avg)
uptime  # load avg should be high during cross gen

# Check training fold progress
grep -E "Fold [0-9]|fold.*complete|accuracy" /workspace/1d_log.txt
```

---

## LightGBM Config (from config.py)

| Parameter | Value | Source |
|-----------|-------|--------|
| min_data_in_leaf | 3 | TF_MIN_DATA_IN_LEAF['1d'] |
| num_leaves | 95 | TF_NUM_LEAVES['1d'] |
| max_bin | 255 | V3_LGBM_PARAMS (binary crosses always get 2 bins regardless) |
| CPCV folds | (4,1) = 4 folds | TF_CPCV_GROUPS['1d'] |
| RIGHT_CHUNK | 200 | MANDATORY -- 500 OOM'd on 503GB, 200 safe on 512GB+ |
| save_binary | Not recommended | Dense matrix ~138.6GB, binary cache would be massive |

---

## Expected Feature Count (min_nonzero=3)

| Component | Count |
|-----------|-------|
| Base features | ~3,756 |
| Cross features (min_nonzero=3) | 6,039,797 confirmed (was 4.46M at min_nonzero=8) |
| Total | 6,043,553 |

The increase from min_nonzero=8 to min_nonzero=3 preserves rare esoteric crosses that
fire 3-7 times in the dataset. These rare signals ARE the edge. LightGBM's
min_gain_to_split=2.0 guards against noise from low-support features.

---

## v3.2 Baseline (targets to beat)
| Metric | v3.2 Value |
|--------|------------|
| Accuracy | 62.6% |
| Precision Long | 64.0% |
| Optuna config | num_leaves=99, feature_fraction=0.113 |
| PBO | DEPLOY |

---

## Failure Modes and What to Check

| Symptom | Cause | Fix |
|---------|-------|-----|
| "WARNING: DB missing" in log | Missing database file | Upload the missing .db, re-run verify script |
| Cross cols ~4.46M instead of ~6.04M | Old v2_cross_names JSON left over (min_nonzero=8) | Delete BOTH npz AND json, restart |
| Load avg ~1.0 during training | is_enable_sparse=True on dense data | Check log for "is_enable_sparse=False" |
| "ModuleNotFoundError: astrology_engine" | astrology_engine.py not in v3.3/ | Copy from project root |
| OOM during dense conversion | Not enough RAM for ~138.6GB dense | Rent 512GB+ machine, or keep sparse (slower but works) |
| Cross gen very slow (>2 hrs) | Sparse matmul single-threaded bottleneck | Normal for 5M+ features. Wait. |
| "No module named 'lightgbm'" | pip install not run | Run install command above |
| Feature count mismatch vs v3.2 | Stale parquet from old feature_library.py | Delete features_BTC_1d.parquet, restart |
| V30_DATA_DIR shows v3.0 path | Env var not set | Verify `export V30_DATA_DIR=/workspace/v3.3` |
| LSTM crashes with NaN | Features have NaN not imputed for LSTM | ml_multi_tf.py imputes NaN->0 for LSTM only. Check log. |
| Optuna errors | Missing optuna/cmaes/colorlog | Run full pip install command |
| "Keeping SPARSE" on 512GB machine | Dense matrix too large for 70% threshold | Expected. Training works on sparse, just slower per fold. |

---

## OPTUNA DEPLOYMENT

### Upload Size for Optuna
- **Total upload: ~3.5 GB** (parquet + NPZ + cross_names JSON + all DBs)
- Can run on a separate machine (small upload, manageable transfer)
- Warm-started from 1w: 50+30 trials (vs 100+50 cold)

### Optuna Timing
- Optuna takes ~2-3x final training time (target after optimizations)
- 1d CPCV = ~2 hrs, so Optuna = ~4 hrs with 50+30 warm trials
- save_binary bridge eliminates redundant EFB construction (Dataset parsed once, reused across all trials)

### If Running Optuna on a Separate Machine
Upload these files:
```
features_BTC_1d.parquet          # base feature parquet
v2_crosses_BTC_1d.npz            # cross feature sparse matrix (~2-3 GB)
v2_cross_names_BTC_1d.json       # cross feature column names
lgbm_dataset_1d.bin              # save_binary output (if available — skips EFB rebuild)
model_1d.json                    # trained model (warm-start seed)
optuna_configs_1w.json           # 1w Optuna results (warm-start cascade source)
All 16 .db files + kp_history_gfz.txt + astrology_engine.py
v33_code.tar.gz                  # all v3.3/*.py code
```
Total: ~3.5 GB. Transferable in ~5-10 min on decent connection.

---

## Download Results When Done

```bash
# Check if pipeline completed
grep "DONE\|pipeline complete\|All steps" /workspace/1d_log.txt

# Download all artifacts
scp -P {PORT} root@{HOST}:/workspace/v3.3/model_1d.json .
scp -P {PORT} root@{HOST}:/workspace/v3.3/optuna_configs_1d.json .
scp -P {PORT} root@{HOST}:/workspace/v3.3/meta_model_1d.pkl .
scp -P {PORT} root@{HOST}:/workspace/v3.3/lstm_1d.pt .
scp -P {PORT} root@{HOST}:/workspace/v3.3/platt_1d.pkl .
scp -P {PORT} root@{HOST}:/workspace/v3.3/validation_report_1d.json .
scp -P {PORT} root@{HOST}:/workspace/v3.3/feature_importance_*.json .
scp -P {PORT} root@{HOST}:/workspace/v3.3/inference_1d_base_cols.json .
scp -P {PORT} root@{HOST}:/workspace/v3.3/inference_1d_cross_names.json .
scp -P {PORT} root@{HOST}:/workspace/v3.3/inference_1d_cross_pairs.npz .
scp -P {PORT} root@{HOST}:/workspace/v3.3/inference_1d_ctx_names.json .
scp -P {PORT} root@{HOST}:/workspace/v3.3/inference_1d_thresholds.json .
scp -P {PORT} root@{HOST}:/workspace/1d_log.txt .
```

**Download partial results after each critical step (vast.ai machines die without warning).**
Especially download `model_1d.json` immediately after Step 3 completes -- this is the
most valuable artifact and takes the longest to produce.

---

## STATUS
Cross gen COMPLETE. CPCV was in progress (sparse+sequential) when machine destroyed. Artifacts downloaded.

---

## Deployment Steps (matches TRAINING_PLAN.md)

### Step 1: Rent machine
```bash
vastai search offers 'cpu_cores_effective >= 128 cpu_ram >= 512 reliability > 0.95 dph <= 2.0' -o 'dph'
```
Pick the machine with highest CPU Score (cores x GHz).

### Step 2: Upload code + DBs
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

### Step 3: Verify DBs (see verify script above)

### Step 4: Nuclear clean (see clean script above)

### Step 5: Install deps (see install command above)

### Step 6: Launch pipeline (see launch command above)

### Step 7: Verify first 30 lines (see verify launch above)

### Step 8: Monitor (see monitor commands above)
