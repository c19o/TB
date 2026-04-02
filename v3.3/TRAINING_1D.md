# 1D Training Guide — V3.3
# Copy-paste ready for a fresh Claude session. Read TRAINING_PLAN.md first.

Status: reference/caveat doc only. Maintained deployment authority is `CLOUD_DEPLOYMENT_FRAMEWORK.md`, `CLOUD_1D_PROFILE.md`, `deploy_1d.sh`, and `contracts/deploy_profiles.json`.

---

## Prerequisites

**1W must be completed first.** 1D is warm-started from 1W Optuna results.

Required from 1W training:
- `optuna_configs_1w.json` — warm-start source for 1D Optuna (enqueues 1W best params, narrows ranges)

If running 1D on a different machine than 1W, upload `optuna_configs_1w.json` before starting.

---

## Machine Requirements

- **RAM:** 256GB+ (dense matrix ~130GB for ~3M features x 5,733 rows)
- **Cores:** 128+ (SPARSE + SEQUENTIAL CPCV + cross gen benefit from high core count)
- **CPU Score:** 250+ (cores x GHz)
- **Disk:** 50GB+
- **GPU:** NOT needed. CPU is faster for all steps at 5,733 rows. Set `ALLOW_CPU=1`.

### Machine Recommendation

| Option | Details |
|--------|---------|
| **LOCAL** | 13900K + 64GB+ (cross gen ~8 min, training ~2 hr). Must have 256GB+ RAM for dense conversion. |
| **Cloud** | vast.ai m:55891 ($0.46/hr, 128c, 504GB RAM, CUDA 13 = no GPU). `ALLOW_CPU=1`. |
| **Any 128c+ CPU** | No GPU needed. CPU Score 250+ recommended. |

**Why CPU?** 5,733 rows is too few for GPU utilization. LightGBM GPU histogram overhead exceeds the compute benefit. All steps run faster on CPU at this row count.

### Engine Per Step

| Step | Engine | Time (local) | Time (cloud 128c) |
|------|--------|-------------|-------------------|
| 0. Prerequisites | N/A | N/A | N/A |
| 1. Feature build | CPU (ALLOW_CPU=1) | ~37s | ~25s |
| 2. Cross gen | CPU (sparse matmul) | ~8 min | ~2.5 min |
| 3. Train without Optuna | CPU (LightGBM) | ~2 hr | ~2 hr |
| 4. Optuna search | CPU (n_jobs=auto) | ~25 min | ~25 min |
| 5. Retrain with Optuna params | CPU (LightGBM) | ~2 hr | ~2 hr |
| 6. Trade optimizer | CPU | ~5 min | ~5 min |
| 7. Meta + LSTM + PBO | CPU (parallel) | ~15 min | ~15 min |

### Revised ETAs

| Stage | Time | Cost (cloud) |
|-------|------|-------------|
| Feature build | 30s | - |
| Cross gen | 2.5-8 min | - |
| Train (no Optuna) | ~2 hr | - |
| Optuna (Phase 1 + Validation Gate, warm) | ~25 min | - |
| Retrain (with Optuna params) | ~2 hr | - |
| Meta + PBO + SHAP | 15 min | - |
| **TOTAL (with Optuna)** | **~5 hr** | **~$2.30 @ $0.46/hr** |
| **TOTAL (without Optuna)** | **~2.5 hr** | **~$1.15 @ $0.46/hr** |

---

## Data

- **Rows:** 5,733 (daily bars, 2010-2026)
- **Base features:** ~3,800 columns
- **Cross features:** ~3M (targeted crossing, 4-tier binarization)
- **Total:** ~3M features
- **Dense matrix size:** ~80-130GB (5,733 x 3M x 4 bytes). Fits in 256GB+ RAM.
- **Triple-barrier labels:** Asymmetric — tp=3.5x ATR, sl=1.5x ATR, max_hold=8 bars

---

## LightGBM Config (from config.py)

| Parameter | Value | Source |
|-----------|-------|--------|
| min_data_in_leaf | 5 | TF_MIN_DATA_IN_LEAF['1d'] |
| num_leaves | 127 | TF_NUM_LEAVES['1d'] |
| max_bin | 255 | V3_LGBM_PARAMS (binary crosses get 2 bins regardless) |
| CPCV | (5, 2) = 10 splits, 4 paths, 60% train | TF_CPCV_GROUPS['1d'] |
| class_weight | balanced | TF_CLASS_WEIGHT['1d'] |
| learning_rate | 0.03 | V3_LGBM_PARAMS (final), 0.15 (Optuna Phase 1) |
| feature_pre_filter | False | CRITICAL — True silently kills rare esoteric features |
| is_enable_sparse | auto | Set to False when dense conversion happens |

---

## Required Databases (ALL 16 — ZERO MISSING)

```
FROM PROJECT ROOT (-> /workspace/):
  btc_prices.db          # 1.3GB — BTC OHLCV 2010-2026
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
  multi_asset_prices.db  # 1.3GB — multi-asset data
  v2_signals.db          # DeFi TVL, BTC dominance, mining stats
```

**Also required (non-DB):**
- `kp_history_gfz.txt` (in /workspace/ or /workspace/v3.3/)
- `astrology_engine.py` (in /workspace/v3.3/ — feature_library.py imports it)
- `optuna_configs_1w.json` (in /workspace/v3.3/ — warm-start source)

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
[ -f /workspace/v3.3/optuna_configs_1w.json ] && echo "OK   optuna_configs_1w.json (warm-start)" || echo "NOTE: optuna_configs_1w.json missing — Optuna will run COLD (more trials needed)"
echo ""
echo "DB check: $FAIL missing"
[ $FAIL -eq 0 ] && echo "ALL 16 DBs + kp_history + astrology_engine PRESENT" || { echo "STOP — fix missing files before proceeding"; exit 1; }
```
**ALL must say OK. If ANY says MISSING -> STOP. Upload the missing file first.**

---

## Nuclear Clean (MANDATORY before first run)

Delete ALL old artifacts. Old NPZs were built with min_nonzero=8 and produce fewer features.
Old cross names JSON truncates column count.

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
cross feature columns (e.g., JSON has 4.46M names but new NPZ has 3M cols -> name mismatch).

**DO NOT delete `optuna_configs_1w.json` — it is the warm-start source.**

---

## Install Dependencies

```bash
pip install -q lightgbm scikit-learn scipy ephem astropy pytz joblib pandas numpy \
  pyarrow optuna hmmlearn numba tqdm pyyaml alembic cmaes colorlog sqlalchemy \
  threadpoolctl 2>&1 | tail -3
python -c "import lightgbm; print(f'LightGBM {lightgbm.__version__} OK')"
python -c "import pandas, numpy, scipy, sklearn, lightgbm, ephem, astropy, pyarrow, optuna, numba, hmmlearn, yaml, tqdm; print('ALL OK')"
```

---

## Launch Command

```bash
cd /workspace/v3.3 && \
  export SAVAGE22_DB_DIR=/workspace && \
  export V30_DATA_DIR=/workspace/v3.3 && \
  export ALLOW_CPU=1 && \
  export PYTHONUNBUFFERED=1 && \
  nohup python -u cloud_run_tf.py --symbol BTC --tf 1d > /workspace/1d_log.txt 2>&1 &
```

**All env vars explained:**
- `SAVAGE22_DB_DIR=/workspace` — where V1 DBs live (tweets.db, btc_prices.db, etc.)
- `V30_DATA_DIR=/workspace/v3.3` — where to read/write parquets, NPZs, models. MUST be v3.3, NOT v3.0!
- `ALLOW_CPU=1` — permits CPU-only mode (no GPU required for 1D)
- `PYTHONUNBUFFERED=1` — real-time log output (no buffering)
- `OMP_NUM_THREADS` / `NUMBA_NUM_THREADS` — set dynamically by cloud_run_tf.py per phase

### RIGHT_CHUNK Setting
```bash
# LOCAL (68GB RAM): MUST set before launch
export V2_RIGHT_CHUNK=200

# CLOUD (256GB+ RAM): can use larger chunks
export V2_RIGHT_CHUNK=500
```
If not set, auto defaults to 2000 which will OOM. Always set explicitly.

---

## Verify Launch (first 30 seconds)

```bash
sleep 30 && head -30 /workspace/1d_log.txt
```

**Must see:**
- "All 16 databases present" (or zero "MISS" lines)
- Row count: ~5,733 rows
- Base feature count: ~3,800 cols
- Correct data directory: V30_DATA_DIR should show `/workspace/v3.3` not `/workspace/v3.0 (LGBM)`
- `ALLOW_CPU=1` acknowledged (no GPU error)

---

## Step-by-Step Pipeline Detail

### Step 1: Build Base Features (~37s local, ~25s cloud)

Runs `build_1d_features.py`. Produces `features_BTC_1d.parquet`.

- 5,733 rows x ~3,800 columns
- Asymmetric triple-barrier labels: tp=3.5x ATR, sl=1.5x ATR, max_hold=8 bars
- All 16 DB sources used (esoteric + TA + macro + space weather + astro)
- `ALLOW_CPU=1` runs pandas CPU path (no cuDF needed)

**Verify:**
```bash
python3 -c "import pandas as pd; df=pd.read_parquet('/workspace/v3.3/features_BTC_1d.parquet'); print(f'Cols: {len(df.columns)}, Rows: {len(df)}')"
# Expected: ~3800 cols, 5733 rows
```

### Step 2: Cross Gen (~8 min local, ~2.5 min cloud)

Runs `v2_cross_generator.py --tf 1d --symbol BTC --save-sparse`. Produces:
- `v2_crosses_BTC_1d.npz` — sparse CSR matrix (~2-3 GB)
- `v2_cross_names_BTC_1d.json` — column name mapping
- `inference_1d_*.json` — inference artifacts for live trading

- ~3M features expected (targeted crossing, 4-tier binarization)
- `OMP_NUM_THREADS=4` during cross gen (prevents thread exhaustion)
- RIGHT_CHUNK controls memory: 200 for 68GB, 500 for 256GB+

**Verify cross features loaded correctly:**
```bash
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

### Step 3: Train Without Optuna (~2 hr, 10 K=2 splits)

Runs `ml_multi_tf.py --tf 1d`. Produces `model_1d.json`.

- CPCV (5,2) = 10 splits, 4 unique paths, 60% train fraction
- Uses config.py defaults (or optuna_configs_1d.json if it exists from prior run)
- Sequential CPCV (parallel disabled for >1M features — pickle bottleneck)
- Sparse CSR input with EFB bundling
- ~30 min per split, 10 splits = ~2 hr total
- Auto-backed up to `model_1d_cpcv_backup.json` after completion

**CRITICAL: Download model_1d.json immediately after Step 3 completes.**
This is the most valuable artifact. vast.ai machines die without warning.

**Verify multi-threaded training:**
```bash
uptime  # load avg should be > cores x 0.3
cat /proc/loadavg
```
If load avg ~1.0 on a 128-core machine, training is SINGLE-THREADED. Check:
1. Log for `is_enable_sparse=False` — must appear if data was converted to dense
2. `OMP_NUM_THREADS` unset — cloud_run_tf.py unsets it before training

**Check training progress:**
```bash
grep -E "Fold [0-9]|fold.*complete|accuracy|mean_accuracy" /workspace/1d_log.txt
```

### Step 4: Optuna Search (~25 min, warm-started from 1W)

Runs `run_optuna_local.py --tf 1d`. Produces `optuna_configs_1d.json`.

**Warm-start from 1W:**
- If `optuna_configs_1w.json` exists, Optuna runs WARM:
  - Enqueues 1W best params as seed trial 1 (adapted to 1D num_leaves cap)
  - 15 Phase 1 trials (vs 25 cold) — 2-fold CPCV, LR=0.15, 60 rounds
  - Top 2 validated (vs top 3 cold) — 4-fold CPCV, LR=0.08, 200 rounds
  - Narrowed search ranges around 1W optimum
- If `optuna_configs_1w.json` is missing, Optuna runs COLD:
  - 25 Phase 1 trials (per-TF override)
  - Top 3 validated
  - Full search ranges

**Phase 1 + Validation Gate structure:**
1. **Phase 1 (rapid):** 2 seeded + random + TPE trials. 2-fold CPCV, fast LR=0.15, ES at 15 rounds. ~15 min.
2. **Validation Gate:** Top-K from Phase 1 re-evaluated with 4-fold CPCV, LR=0.08, 200 rounds, ES at 50. ~10 min.
3. **Output:** Best validated params saved to `optuna_configs_1d.json`.

**Does NOT produce a model.** Only saves params for Step 5.

**Key search ranges (1D-specific):**
- num_leaves: 4 to 127 (cap from TF_NUM_LEAVES['1d'])
- min_data_in_leaf: starts at 5 (TF_MIN_DATA_IN_LEAF['1d'])
- feature_fraction: wide range (EFB handles sparsity)

### Step 5: Retrain With Optuna Params (~2 hr)

Runs `ml_multi_tf.py --tf 1d` again. This time it reads `optuna_configs_1d.json` and uses the optimized params.

- Same CPCV (5,2) = 10 splits as Step 3
- Full 800 rounds, LR=0.03 (production quality)
- Produces final `model_1d.json` (overwrites Step 3 model)
- Backup saved to `model_1d_cpcv_backup.json`

**Download model_1d.json immediately after Step 5.**

### Step 6: Trade Optimizer

Runs `exhaustive_optimizer.py --tf 1d`. Optimizes leverage, risk %, stop ATR, R:R, max hold, exit type, confidence threshold.

- 13D search space, 500 TPE trials, Sortino objective
- Produces optimal trade parameters for live trading
- Non-fatal — pipeline continues if this fails

### Step 7: Meta + LSTM + PBO (parallel)

Three tasks run simultaneously:
- **Meta-labeling** (`meta_labeling.py --tf 1d`) -> `meta_model_1d.pkl`
- **LSTM** (`lstm_sequence_model.py --tf 1d --train`) -> `lstm_1d.pt` + `platt_1d.pkl`
- **PBO** (`backtest_validation.py --tf 1d`) -> `validation_report_1d.json`

All non-fatal. LSTM may crash on NaN (imputes NaN->0 internally).

### Step 8: SHAP + Audit

- SHAP cross feature validation (split importance, active feature count)
- Backtesting audit (`backtesting_audit.py --tf 1d`)

---

## SPARSE + SEQUENTIAL CPCV

For 1D with ~3M features, parallel CPCV via ProcessPoolExecutor is disabled:
- Dense conversion (~80-130GB) may fit in RAM, but parallel path converts dense->sparse for pickle transport, then each worker converts sparse->dense again
- Multiple workers x ~80GB pickle data = hundreds of GB through one IPC pipe
- **Sequential CPCV** (one fold at a time) with sparse CSR input avoids this
- LightGBM trains on sparse CSR with EFB bundling — slower per fold but no pickle overhead

---

## is_enable_sparse Lesson (CRITICAL)

`config.py` has `is_enable_sparse: True` as the default LightGBM param.
When ml_multi_tf.py converts sparse to dense (because RAM allows it), it MUST set
`is_enable_sparse=False`. Otherwise LightGBM assumes sparse input format on dense data,
causing SINGLE-THREADED training.

**Verify:** Check the log for `is_enable_sparse=False`. If this line is missing and data
is dense, training will be 10-50x slower than expected.

---

## Monitor Commands

```bash
# Live log tail
tail -f /workspace/1d_log.txt

# Check for errors
grep -iE "error|traceback|fail|critical|exception" /workspace/1d_log.txt

# Check pipeline progress (which step is running)
grep -E "Step [0-9]|DONE|RUNNING|COMPLETE|Fold|accuracy" /workspace/1d_log.txt

# Check system resources
uptime && free -h && df -h /workspace

# Check for OOM kills
dmesg | tail -20 | grep -i oom

# Check training fold progress
grep -E "Fold [0-9]|fold.*complete|accuracy|mean" /workspace/1d_log.txt

# Check Optuna progress
grep -E "Trial|Phase 1|Validation|best|mlogloss" /workspace/1d_log.txt

# Check RSS (dense matrix should be ~80-130GB)
ps aux | grep cloud_run_tf | grep -v grep | awk '{print $6/1024 " MB"}'
```

---

## OPTUNA DEPLOYMENT (if running on separate machine)

### Upload Size: ~3.5 GB

Upload these files:
```
features_BTC_1d.parquet          # base feature parquet
v2_crosses_BTC_1d.npz            # cross feature sparse matrix (~2-3 GB)
v2_cross_names_BTC_1d.json       # cross feature column names
optuna_configs_1w.json           # 1W warm-start source (CRITICAL)
All 16 .db files + kp_history_gfz.txt + astrology_engine.py
v33_code.tar.gz                  # all v3.3/*.py code
```
Total: ~3.5 GB. Transferable in ~5-10 min on decent connection.

### Optuna-Only Run
```bash
cd /workspace/v3.3 && \
  export SAVAGE22_DB_DIR=/workspace && \
  export V30_DATA_DIR=/workspace/v3.3 && \
  export ALLOW_CPU=1 && \
  export PYTHONUNBUFFERED=1 && \
  python -u run_optuna_local.py --tf 1d 2>&1 | tee /workspace/optuna_1d_log.txt
```

After Optuna completes, download `optuna_configs_1d.json` and copy to the training machine.

---

## Deployment Steps (cloud)

### Step 1: Rent machine
```bash
# vast.ai specific machine (known working):
# m:55891 — 128c, 504GB RAM, $0.46/hr, CUDA 13 (GPU disabled, CPU-only)

# Or search for cheap CPU machines:
vastai search offers 'cpu_cores_effective >= 128 cpu_ram >= 256 reliability > 0.95 dph <= 1.0' -o 'dph'
```
Pick the machine with highest CPU Score (cores x GHz). No GPU needed.

### Step 2: Upload code + DBs
```bash
SSH="ssh -i ~/.ssh/vast_key -o StrictHostKeyChecking=no"
SCP="scp -i ~/.ssh/vast_key -o StrictHostKeyChecking=no"

# Upload code
$SCP -P {PORT} /tmp/v33_code.tar.gz root@{HOST}:/workspace/
$SSH -p {PORT} root@{HOST} "cd /workspace && tar xzf v33_code.tar.gz -C v3.3/"

# Upload DBs
$SCP -P {PORT} /tmp/v33_dbs.tar.gz root@{HOST}:/workspace/
$SSH -p {PORT} root@{HOST} "cd /workspace && tar xzf v33_dbs.tar.gz && ln -sf /workspace/*.db /workspace/v3.3/"

# Upload big DBs separately
$SCP -P {PORT} "C:/Users/C/Documents/Savage22 Server/v3.3/btc_prices.db" root@{HOST}:/workspace/v3.3/
$SCP -P {PORT} "C:/Users/C/Documents/Savage22 Server/v3.3/multi_asset_prices.db" root@{HOST}:/workspace/v3.3/
$SSH -p {PORT} root@{HOST} "ln -sf /workspace/v3.3/btc_prices.db /workspace/ && ln -sf /workspace/v3.3/multi_asset_prices.db /workspace/"

# Upload 1W warm-start config
$SCP -P {PORT} "C:/Users/C/Documents/Savage22 Server/v3.3/optuna_configs_1w.json" root@{HOST}:/workspace/v3.3/
```

### Step 3: Verify DBs (run verify script above)

### Step 4: Nuclear clean (run clean script above — preserve optuna_configs_1w.json)

### Step 5: Install deps (run install command above)

### Step 6: Set RIGHT_CHUNK + Launch
```bash
export V2_RIGHT_CHUNK=500  # 504GB machine
# Then run launch command above
```

### Step 7: Verify first 30 lines (run verify launch above)

### Step 8: Monitor (run monitor commands above)

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
scp -P {PORT} root@{HOST}:/workspace/v3.3/model_1d_cpcv_backup.json .
scp -P {PORT} root@{HOST}:/workspace/1d_log.txt .
```

**Download partial results after each critical step (vast.ai machines die without warning).**
Priority download order:
1. `model_1d.json` — after Step 3 (most expensive artifact)
2. `optuna_configs_1d.json` — after Step 4 (needed for retrain + downstream TFs)
3. `model_1d.json` again — after Step 5 (final retrained model)
4. Everything else — after pipeline completes

---

## Failure Modes and What to Check

| Symptom | Cause | Fix |
|---------|-------|-----|
| "WARNING: DB missing" in log | Missing database file | Upload the missing .db, re-run verify script |
| Cross cols much less than ~3M | Old v2_cross_names JSON (min_nonzero=8) | Delete BOTH npz AND json, restart |
| Load avg ~1.0 during training | is_enable_sparse=True on dense data | Check log for "is_enable_sparse=False" |
| "ModuleNotFoundError: astrology_engine" | astrology_engine.py not in v3.3/ | Copy from project root |
| OOM during dense conversion | Not enough RAM for dense matrix | Rent 256GB+ machine, or keep sparse (slower but works) |
| OOM during cross gen | RIGHT_CHUNK too large | Set `V2_RIGHT_CHUNK=200` (68GB) or `500` (256GB+) |
| Cross gen very slow (>30 min) | Normal for sparse matmul with 3M+ features | Wait. CPU single-threaded bottleneck. |
| "No module named 'lightgbm'" | pip install not run | Run install command above |
| Feature count mismatch | Stale parquet from old feature_library.py | Delete features_BTC_1d.parquet, restart |
| V30_DATA_DIR shows v3.0 path | Env var not set | Verify `export V30_DATA_DIR=/workspace/v3.3` |
| LSTM crashes with NaN | Features have NaN not imputed for LSTM | ml_multi_tf.py imputes NaN->0 for LSTM only. Non-fatal. |
| Optuna cold (no warm-start) | optuna_configs_1w.json missing | Upload from 1W training. Runs cold if missing (more trials). |
| "Keeping SPARSE" on 256GB machine | Dense matrix too large for 70% threshold | Expected. Training works on sparse, slower per fold. |
| GPU error on CUDA 13 machine | cuDF not compatible with CUDA 13 | Set `ALLOW_CPU=1` (already in launch command) |

---

## v3.2 Baseline (targets to beat)

| Metric | v3.2 Value |
|--------|------------|
| Accuracy | 62.6% |
| Precision Long | 64.0% |
| Optuna config | num_leaves=99, feature_fraction=0.113 |
| PBO | DEPLOY |

---

## Warm-Start Cascade

```
1W (ROOT, cold Optuna) -> 1D (warm from 1W) -> 4H (warm from 1D) -> 1H (warm from 4H) -> 15m (warm from 1H)
```

1D is the second link in the cascade. After 1D Optuna completes:
- Download `optuna_configs_1d.json`
- Upload it to the 4H training machine as the warm-start source

---

## STATUS

Pending. 1W complete (77.64% accuracy). 1D cross gen previously completed (6M features with old method). Needs re-run with targeted crossing (~3M features).
