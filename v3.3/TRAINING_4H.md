# 4H Training Guide -- V3.3
# Copy-paste ready for a fresh Claude session. Read TRAINING_PLAN.md first.

Status: reference/caveat doc only. Maintained deployment authority is `CLOUD_DEPLOYMENT_FRAMEWORK.md`, `CLOUD_4H_PROFILE.md`, `deploy_4h.sh`, and `contracts/deploy_profiles.json`.

---

## Machine Requirements
- **RAM:** 512GB+ (cross gen peaks high with 23K rows x 3-4M features)
- **Cores:** 128+ (CPCV has 15 splits, Optuna n_jobs=4, cross gen benefits from high core count)
- **CPU Score:** 450+ recommended (cores x GHz). Training is CPU-bound at 23K rows.
- **Disk:** 50GB+
- **RIGHT_CHUNK:** `export V2_RIGHT_CHUNK=500` (MANDATORY -- auto=2000 OOMs. 500 safe on 512GB+)

### Machine Recommendation: Cloud, 512GB+ RAM, CPU Score 450+
GPU starts to help at 23K rows but benefit is marginal -- CPU is fine for training.
GPU helps cross gen IF CUDA 12.x available (cuSPARSE SpGEMM). CUDA 13+ = CPU only (ALLOW_CPU=1).
- **Cloud option:** vast.ai m:55891 ($0.46/hr, 128c, 504GB, CUDA 13 -- ALLOW_CPU=1)
- **GPU:** Marginal at 23K rows. CPU training is acceptable. GPU cross gen only if CUDA 12.x.
- **Cost estimate:** ~$5-8 without Optuna, ~$12-18 with Optuna

### GPU vs CPU Per Step
| Step | Engine | Reason |
|------|--------|--------|
| Feature build | CPU (ALLOW_CPU=1 on CUDA 13) | Rolling/ewm via pandas. ~15 min. |
| Cross gen | CPU or GPU (cuSPARSE if CUDA 12.x) | CPU ~30-45 min. GPU ~10-15 min if available. |
| Optuna search | CPU (n_jobs=auto) | 23K rows subsampled to 8K. CPU parallel TPE trials. |
| Final CPCV | CPU | 15 splits sequential. ~4-6 hrs on 128c Score 450+. |
| Trade optimizer | CPU | Parameter sweep on 23K rows. |

### Optuna Machine Strategy
- CPU-heavy machine (128+ cores, CPU Score 450+). GPU optional.
- n_jobs = auto (total_cores // 8). Each trial trains on ~8K rows (35% subsample).
- Warm-started from 1d: Phase 1 = 15 trials, Validation = top 2. Much faster than cold.
- Same machine for Optuna and training -- no need to separate.
- Separate Optuna machine only if parallelizing across TFs.

### Revised ETAs (Machine: 128c, Score 450+, CPU training)
| Stage | Time | Status |
|-------|------|--------|
| Feature build | 15 min | PENDING |
| Cross gen (CPU, 128 threads) | 30-45 min | PENDING |
| save_binary | 10 min | PENDING |
| CPCV (15 splits, sparse seq) | 4-6 hrs | PENDING |
| Optuna (warm from 1d, Phase1=15 + Val=2) | ~3-5 hrs | PENDING |
| Meta + PBO + SHAP | 15 min | PENDING |
| **TOTAL** | **~9-13 hrs ($5-6)** | |
| **Without Optuna** | **~5-7 hrs ($3-4)** | |

---

## Key Differences from 1W/1D

| Parameter | 1W | 1D | **4H** |
|-----------|----|----|--------|
| Rows | 818 | 5,733 | **~23,000** |
| CPCV config | (5,2) = 10 splits | (5,2) = 10 splits | **(6,2) = 15 splits** |
| PBO unique paths | 4 paths | 4 paths | **5 paths** |
| Train fraction | 60% | 60% | **67%** |
| min_data_in_leaf | 5 | 5 | **5** |
| num_leaves | 31 | 127 | **255** |
| Optuna row subsample | 1.0 (all) | 1.0 (all) | **0.35 (23K -> 8K)** |
| Warm-start source | ROOT (cold) | 1w | **1d** |
| Triple barrier TP | 3.5x ATR | 3.5x ATR | **3.0x ATR** |
| Triple barrier SL | 1.2x ATR | 1.5x ATR | **1.5x ATR** |
| Triple barrier max_hold | 4 bars | 8 bars | **12 bars** |

**The 15-split CPCV is the biggest cost difference.** Each split trains a full LightGBM model
on 3-4M+ features. Budget 1.5x more wall-time than 1d for the same machine.

---

## Data
- **Rows:** ~23,000 (4h bars, 2017-08-17 to 2026, Binance global API)
- **Base features:** ~3,900+ cols
- **Cross features:** ~3-4M expected (min_nonzero=3)
- **Total:** ~3-4M features (SPARSE -- kept sparse for training)
- **min_data_in_leaf:** 5 (esoteric signals fire more often at 4h than daily)
- **Triple barrier:** tp=3.0x ATR, sl=1.5x ATR, max_hold=12 bars (asymmetric -- fixes SHORT precision)

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
[ $FAIL -eq 0 ] && echo "ALL 16 DBs + kp_history + astrology_engine PRESENT" || { echo "STOP -- fix missing files before proceeding"; exit 1; }
```
**ALL must say OK. If ANY says MISSING -> STOP. Upload the missing file first.**

---

## Nuclear Clean (MANDATORY before first run)

Delete ALL old artifacts. Old NPZs were built with min_nonzero=8 and produce fewer features.
Old cross names JSON truncates column count (cross_matrix has N cols but JSON only names M < N).

```bash
cd /workspace && rm -f *.npz *.json *.pkl *.parquet *.log DONE_* RUNNING_* *.lock 2>/dev/null  # NOTE: no *.txt -- would kill kp_history_gfz.txt
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
cross feature columns (e.g., JSON has 2M names but new NPZ has 3.5M cols -> 1.5M features
get generic `cross_N` names instead of real names, confusing SHAP and audit).

---

## Verify Parquet Freshness
```bash
python3 -c "import pandas as pd; df=pd.read_parquet('/workspace/v3.3/features_BTC_4h.parquet'); print(f'Cols: {len(df.columns)}, Rows: {len(df)}')"
# Expected: ~3,900+ columns, ~23,000 rows
```

---

## Pipeline Steps
1. Feature build (~15 min) -> `features_BTC_4h.parquet`
2. Cross gen (~30-45 min CPU, min_nonzero=3) -> `v2_crosses_BTC_4h.npz` + `v2_cross_names_BTC_4h.json`
3. LightGBM CPCV (SPARSE + SEQUENTIAL -- 15 splits, (6,2) config) -> `model_4h.json`
4. Optuna (warm from 1d, Phase1 + Validation) -> `optuna_configs_4h.json`
5. Meta-labeling -> `meta_model_4h.pkl`
6. LSTM -> `lstm_4h.pt` + `platt_4h.pkl`
7. PBO/Audit -> `validation_report_4h.json`

---

## Quick Launch (Automated)

```bash
# On cloud machine, after uploading code.tar.gz + dbs.tar.gz:
bash setup.sh
cd /workspace/v3.3 && lgbm-run python -u cloud_run_tf.py --tf 4h 2>&1 | tee /workspace/4h_log.txt
```

This runs setup.sh (installs deps, tcmalloc, THP, NUMA detection, creates lgbm-run wrapper),
then launches the full 7-step pipeline with memory optimizations applied.

**Notes on CUDA 13 / ALLOW_CPU:**
- vast.ai m:55891 has CUDA 13 (driver 580+). cuDF/CuPy compiled for CUDA 12.x will not work.
- setup.sh auto-detects this and sets `ALLOW_CPU=1` so GPU-or-nothing guards do not crash.
- Feature build uses pandas CPU. Cross gen uses scipy sparse CPU. Training uses LightGBM CPU.
- This is fine -- 4h is CPU-dominated anyway at 23K rows.

---

## Manual Step-by-Step Launch

### Step 1: Rent Machine
```bash
# vast.ai search for 512GB+ RAM, 128+ core machines
vastai search offers 'cpu_cores_effective >= 128 cpu_ram >= 500 reliability > 0.95 dph <= 2.0' -o 'dph'
```
Pick the machine with highest CPU Score (cores x GHz). Recommended: m:55891 ($0.46/hr, 128c, 504GB).

### Step 2: Upload Code + DBs
```bash
SSH="ssh -i ~/.ssh/vast_key -o StrictHostKeyChecking=no"
# Upload code
scp -i ~/.ssh/vast_key -o StrictHostKeyChecking=no -P {PORT} /tmp/v33_code.tar.gz root@{HOST}:/workspace/
$SSH -p {PORT} root@{HOST} "cd /workspace && tar xzf v33_code.tar.gz"
# Upload DBs
scp -i ~/.ssh/vast_key -o StrictHostKeyChecking=no -P {PORT} /tmp/v33_dbs.tar.gz root@{HOST}:/workspace/
$SSH -p {PORT} root@{HOST} "cd /workspace && tar xzf v33_dbs.tar.gz"
# Upload setup.sh
scp -i ~/.ssh/vast_key -P {PORT} "C:/Users/C/Documents/Savage22 Server/v3.3/setup.sh" root@{HOST}:/workspace/
```

### Step 3: Run Setup
```bash
$SSH -p {PORT} root@{HOST} "cd /workspace && bash setup.sh"
```

### Step 4: Verify DBs (see verify script above)

### Step 5: Nuclear Clean (see clean script above)

### Step 6: Install Dependencies (if setup.sh did not run)
```bash
pip install -q lightgbm scikit-learn scipy ephem astropy pytz joblib pandas numpy \
  pyarrow optuna hmmlearn numba tqdm pyyaml alembic cmaes colorlog sqlalchemy \
  threadpoolctl psutil 2>&1 | tail -3
python -c "import pandas, numpy, scipy, sklearn, lightgbm, ephem, astropy, pyarrow, optuna, numba, hmmlearn, yaml, tqdm; print('ALL OK')"
```

### Step 7: Launch Pipeline
```bash
cd /workspace/v3.3 && \
  export SAVAGE22_DB_DIR=/workspace && \
  export V30_DATA_DIR=/workspace/v3.3 && \
  export PYTHONUNBUFFERED=1 && \
  export V2_RIGHT_CHUNK=500 && \
  export ALLOW_CPU=1 && \
  nohup lgbm-run python -u cloud_run_tf.py --symbol BTC --tf 4h > /workspace/4h_log.txt 2>&1 &
```

**All env vars explained:**
- `SAVAGE22_DB_DIR=/workspace` -- where V1 DBs live (tweets.db, btc_prices.db, etc.)
- `V30_DATA_DIR=/workspace/v3.3` -- where to read/write parquets, NPZs, models. MUST be v3.3, NOT v3.0!
- `PYTHONUNBUFFERED=1` -- real-time log output (no buffering)
- `V2_RIGHT_CHUNK=500` -- mandatory OOM prevention for cross gen
- `ALLOW_CPU=1` -- required on CUDA 13+ machines (cuDF/CuPy unavailable)
- `OMP_NUM_THREADS` / `NUMBA_NUM_THREADS` -- set dynamically by cloud_run_tf.py per phase

---

## Verify Launch (first 30 seconds)

```bash
sleep 30 && head -30 /workspace/4h_log.txt
```

**Must see:**
- "All 16 databases present" (or zero "MISS" lines)
- Row count: ~23,000 rows
- Base feature count: ~3,900+ cols
- Correct data directory: V30_DATA_DIR should show `/workspace/v3.3` not `/workspace/v3.0 (LGBM)`
- ALLOW_CPU=1 acknowledged (if on CUDA 13 machine)

**CHECK ALL CONFIG PATHS IN FIRST 10 LOG LINES:** DB_DIR, V30_DATA_DIR, SAVAGE22_DB_DIR, PROJECT_DIR.
Wrong paths = training on wrong/stale data = wasted time and money.

---

## Verify Cross Features Loaded Correctly

After cross gen completes and training starts, check the log for:
```bash
grep -E "Sparse crosses loaded|cross.*cols|feature_cols.*len" /workspace/4h_log.txt
```

**Expected:** ~3-4M cross feature cols loaded (min_nonzero=3). If you see < 2M,
the old cross names JSON was not deleted. STOP, delete both
`v2_crosses_BTC_4h.npz` and `v2_cross_names_BTC_4h.json`, and restart.

**How to double-check:**
```bash
python3 -c "
import scipy.sparse as sp
X = sp.load_npz('/workspace/v3.3/v2_crosses_BTC_4h.npz')
print(f'NPZ shape: {X.shape}')
import json, os
jp = '/workspace/v3.3/v2_cross_names_BTC_4h.json'
if os.path.exists(jp):
    names = json.load(open(jp))
    print(f'JSON names: {len(names)}')
    if len(names) != X.shape[1]:
        print(f'MISMATCH! NPZ has {X.shape[1]} cols but JSON has {len(names)} names. DELETE JSON and restart.')
    else:
        print('MATCH OK')
print(f'NNZ: {X.nnz:,}')
"
```
If cross count is < 2M, something is wrong (min_nonzero may not be 3). Check logs.

---

## Verify Multi-Threaded Training

After CPCV training starts, check load average:

```bash
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

3. **RSS too small:** Check process memory usage:
   ```bash
   ps aux | grep cloud_run_tf | grep -v grep | awk '{print $6/1024 " MB"}'
   ```

**Key log lines to look for:**
```
is_enable_sparse=False (data converted to dense -- enables multi-core LightGBM)
```
or
```
Keeping SPARSE (dense would need XXX GB, only YYY GB avail)
```
If keeping sparse: training is slower per fold but sequential CPCV trains one fold at a time.
With 23K rows x 3-4M features, sparse is likely (dense matrix would be ~300GB+).

---

## SPARSE + SEQUENTIAL CPCV (15 splits, parallel disabled for >1M features)

For 4h with 3-4M features, parallel CPCV via ProcessPoolExecutor is disabled:
- Dense conversion would need 300GB+ (23K x 3-4M x 4 bytes). May not fit in 504GB.
- Even if it fits, parallel path converts dense->sparse for pickle transport, then each worker converts sparse->dense again
- 4+ workers x ~200GB pickle data = pickle bottleneck
- **Mode:** Sequential CPCV (one split at a time) with sparse CSR input
- LightGBM trains on sparse CSR with EFB bundling -- slower per split but no pickle overhead
- (6,2) config = 15 splits total, 5 unique PBO paths, 67% train fraction per split
- num_threads capped for optimal scaling at this row count

---

## is_enable_sparse Lesson (CRITICAL)

`config.py` has `is_enable_sparse: True` as the default LightGBM param.
When ml_multi_tf.py converts sparse to dense (because RAM allows it), it MUST set
`is_enable_sparse=False`. Otherwise LightGBM assumes sparse input format on dense data,
causing SINGLE-THREADED training (the sparse histogram builder serializes OpenMP).

This is handled automatically in ml_multi_tf.py:
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
tail -f /workspace/4h_log.txt

# Check for errors
grep -iE "error|traceback|fail|critical|exception" /workspace/4h_log.txt

# Check pipeline progress (which step is running)
grep -E "Step [0-9]|DONE|RUNNING|COMPLETE" /workspace/4h_log.txt

# Check training fold/split progress
grep -E "Fold [0-9]|Split [0-9]|fold.*complete|accuracy" /workspace/4h_log.txt

# Check system resources
uptime && free -h && df -h /workspace

# Check for OOM kills
dmesg | tail -20 | grep -i oom

# Check if process is alive
ps aux | grep cloud_run_tf | grep -v grep

# Check memory usage detail
free -g
```

---

## LightGBM Config (from config.py)

| Parameter | Value | Source |
|-----------|-------|--------|
| min_data_in_leaf | 5 | TF_MIN_DATA_IN_LEAF['4h'] |
| num_leaves | 255 | TF_NUM_LEAVES['4h'] |
| max_bin | 255 | V3_LGBM_PARAMS (binary crosses always get 2 bins regardless) |
| CPCV config | (6,2) = 15 splits, 5 paths | TF_CPCV_GROUPS['4h'] |
| Train fraction | 67% (4 of 6 groups) | (N-K)/N = (6-2)/6 |
| RIGHT_CHUNK | 500 | MANDATORY -- auto=2000 OOMs |
| feature_fraction | 0.1 | V3_LGBM_PARAMS |
| learning_rate | 0.03 | V3_LGBM_PARAMS |
| save_binary | Feasible | NPZ + base parquet = manageable at 3-4M features |
| Optuna row subsample | 0.35 | 23K -> ~8K rows for search speed |
| Optuna Phase 1 trials | 25 (cold) / 15 (warm) | OPTUNA_TF_PHASE1_TRIALS['4h'] |

---

## Expected Feature Count (min_nonzero=3)

| Component | Count |
|-----------|-------|
| Base features | ~3,900+ |
| Cross features (min_nonzero=3) | ~3-4M (untested at min_nonzero=3 for 4h) |
| Total | ~3-4M |

The increase from min_nonzero=8 to min_nonzero=3 preserves rare esoteric crosses that
fire 3-7 times in the dataset. These rare signals ARE the edge. LightGBM's
min_gain_to_split=2.0 guards against noise from low-support features.

---

## v3.2 Baseline (targets to beat)
| Metric | v3.2 Value |
|--------|-----------|
| Meta AUC | 0.616 |
| PBO | REJECT (17,520 rows -- old row count) |

---

## OPTUNA DEPLOYMENT

### Upload Size for Optuna
- **Total upload: ~5 GB** (parquet + NPZ + cross_names JSON + all DBs)
- Can run on a separate machine (manageable upload)
- Warm-started from 1d: Phase 1 = 15 trials + Validation = top 2 (vs 25+3 cold)

### Optuna Config (from config.py)
| Parameter | Value |
|-----------|-------|
| Phase 1 trials | 15 (warm from 1d) or 25 (cold) |
| Phase 1 CPCV groups | 2 (fast evaluation) |
| Phase 1 rounds | 60 (ES fires at ~30) |
| Phase 1 LR | 0.15 (5x final for fast convergence) |
| Phase 1 ES patience | 15 (aggressive) |
| Validation top-K | 2 (warm) or 3 (cold) |
| Validation CPCV groups | 4 |
| Validation rounds | 200 |
| Row subsample | 0.35 (23K -> ~8K rows) |
| Final LR | 0.03 |
| Final rounds | 800 |

### Optuna Timing
- Phase 1 (15 warm trials, 2-fold CPCV, 8K rows): ~30-60 min
- Validation (top 2, 4-fold CPCV, 8K rows): ~30-60 min
- Final retrain (full CPCV, all 23K rows): part of main pipeline
- **Total Optuna: ~1-2 hrs (warm) / ~2-4 hrs (cold)**

### If Running Optuna on a Separate Machine
Upload these files:
```
features_BTC_4h.parquet          # base feature parquet
v2_crosses_BTC_4h.npz            # cross feature sparse matrix (~3-4 GB)
v2_cross_names_BTC_4h.json       # cross feature column names
lgbm_dataset_4h.bin              # save_binary output (if available -- skips EFB rebuild)
model_4h.json                    # trained model (warm-start seed)
optuna_configs_1d.json           # 1d Optuna results (warm-start cascade source)
All 16 .db files + kp_history_gfz.txt + astrology_engine.py
v33_code.tar.gz                  # all v3.3/*.py code
```
Total: ~5 GB. Transferable in ~10-15 min on decent connection.

---

## Download Results When Done

```bash
# Check if pipeline completed
grep "DONE\|pipeline complete\|All steps" /workspace/4h_log.txt

# Download all artifacts
scp -P {PORT} root@{HOST}:/workspace/v3.3/model_4h.json .
scp -P {PORT} root@{HOST}:/workspace/v3.3/optuna_configs_4h.json .
scp -P {PORT} root@{HOST}:/workspace/v3.3/meta_model_4h.pkl .
scp -P {PORT} root@{HOST}:/workspace/v3.3/lstm_4h.pt .
scp -P {PORT} root@{HOST}:/workspace/v3.3/platt_4h.pkl .
scp -P {PORT} root@{HOST}:/workspace/v3.3/validation_report_4h.json .
scp -P {PORT} root@{HOST}:/workspace/v3.3/feature_importance_*.json .
scp -P {PORT} root@{HOST}:/workspace/v3.3/inference_4h_base_cols.json .
scp -P {PORT} root@{HOST}:/workspace/v3.3/inference_4h_cross_names.json .
scp -P {PORT} root@{HOST}:/workspace/v3.3/inference_4h_cross_pairs.npz .
scp -P {PORT} root@{HOST}:/workspace/v3.3/inference_4h_ctx_names.json .
scp -P {PORT} root@{HOST}:/workspace/v3.3/inference_4h_thresholds.json .
scp -P {PORT} root@{HOST}:/workspace/4h_log.txt .
```

**Download partial results after each critical step (vast.ai machines die without warning).**
Especially download `model_4h.json` immediately after CPCV completes -- this is the
most valuable artifact and takes the longest to produce.

**Critical download points:**
1. After cross gen: download `v2_crosses_BTC_4h.npz` + `v2_cross_names_BTC_4h.json` (~4GB)
2. After CPCV: download `model_4h.json` (most expensive artifact)
3. After Optuna: download `optuna_configs_4h.json`
4. After full pipeline: download everything above

---

## Failure Modes and What to Check

| Symptom | Cause | Fix |
|---------|-------|-----|
| "WARNING: DB missing" in log | Missing database file | Upload the missing .db, re-run verify script |
| Cross cols < 2M | Old v2_cross_names JSON left over (min_nonzero=8) | Delete BOTH npz AND json, restart |
| Load avg ~1.0 during training | is_enable_sparse=True on dense data | Check log for "is_enable_sparse=False" |
| "ModuleNotFoundError: astrology_engine" | astrology_engine.py not in v3.3/ | Copy from project root |
| OOM during cross gen | V2_RIGHT_CHUNK too large for RAM | Verify V2_RIGHT_CHUNK=500. Lower to 300 if still OOMing |
| Feature count mismatch | Stale parquet from old feature_library.py | Delete features_BTC_4h.parquet, restart |
| V30_DATA_DIR shows v3.0 path | Env var not set | Verify `export V30_DATA_DIR=/workspace/v3.3` |
| Cross gen very slow (>2 hrs) | Single-threaded sparse matmul | Normal for 23K rows x 3-4M crosses on CPU. Wait. |
| LSTM crashes with NaN | Features have NaN not imputed for LSTM | ml_multi_tf.py imputes NaN->0 for LSTM only. Check log. |
| "No module named 'lightgbm'" | pip install not run | Run install command above |
| Optuna errors | Missing optuna/cmaes/colorlog | Run full pip install command |
| "Keeping SPARSE" on 504GB machine | Dense matrix too large for 70% threshold | Expected. Training works on sparse, just slower per fold. |
| cuDF/CuPy SEGFAULT | CUDA 13+ driver with CUDA 12.x compiled libs | Set ALLOW_CPU=1 (setup.sh does this automatically) |
| PBO shows REJECT | 23K rows is borderline for PBO significance | Not necessarily a blocker -- check if accuracy + Sortino are acceptable |

---

## Notes
- 4h has ~23,000 rows (2017-08-17 to 2026, Binance global API)
- (6,2) CPCV = 15 splits, 5 unique PBO paths, 67% train fraction -- production model trains on ALL data
- min_data_in_leaf=5 for 4h (per TF_MIN_DATA_IN_LEAF)
- num_leaves=255 for 4h (23K rows can handle more complexity than 1d/1w)
- Full cross gen from scratch -- no pre-built NPZ exists for 4h
- With 23K rows, PBO should be more reliable than 1w (818) or 1d (5,733)
- Warm-start cascade: 1w -> 1d -> **4h** -> 1h -> 15m. Requires optuna_configs_1d.json.

---

## STATUS
Not yet trained in v3.3. Previous v3.2 run: Meta AUC 0.616, PBO REJECT.
Cross gen step 5/13 was in progress when machine was destroyed (only parquet saved).
