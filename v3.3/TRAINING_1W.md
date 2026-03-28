# 1W Training Guide — V3.3
# Copy-paste ready for a fresh Claude session. Read TRAINING_PLAN.md first.

---

## Machine Requirements
- **RAM:** 64GB+ (dense matrix is ~6.1GB for 2.2M features x 818 rows)
- **Cores:** 64+ (LightGBM parallel + 4 CPCV folds via ProcessPoolExecutor)
- **CPU Score:** 120+ (cores x GHz). Training is fast, ~15-20 min total.
- **Disk:** 30GB+
- **GPU:** RTX 3090 available but NOT needed (CPU Score 120 sufficient, 818 rows too few for GPU utilization)

### Machine Recommendation: LOCAL (13900K + RTX 3090)
**NO cloud needed.** 1w is the smallest dataset (818 rows). CPU trains in 13 min. Local 3090 available for GPU histogram fork but provides negligible benefit at this row count. Total pipeline ~2 hrs with Optuna, ~45 min without.

### GPU vs CPU Per Step
| Step | Engine | Reason |
|------|--------|--------|
| Feature build | GPU (cuDF) | Rolling/ewm on GPU. ~5 min. |
| Cross gen | GPU (cuSPARSE SpGEMM) | 15 sec GPU vs 7 min CPU. DONE. |
| Optuna search | CPU (n_jobs=4) | Too few rows for GPU utilization. 4 parallel TPE trials. |
| Final CPCV | CPU | 818 rows = no GPU benefit. 13 min total. |
| Trade optimizer | GPU | cuDF-accelerated parameter sweep. |

### Optuna Machine Strategy
- Run locally on 13900K. CPU Score 120 is sufficient for 818-row dataset.
- n_jobs=4 workers (4 parallel Optuna trials). Each trial trains on 818 rows = fast.
- Warm-start: 1w is the ROOT of the cascade (1w -> 1d -> 4h -> 1h -> 15m). Run 100+50 cold trials.
- No separate Optuna machine needed. Same local machine handles everything.

### Revised ETAs (local 13900K + 3090)
| Stage | Time | Status |
|-------|------|--------|
| Feature build | 5 min | DONE |
| Cross gen (cuSPARSE SpGEMM) | 15 sec | DONE |
| save_binary | 30 sec | DONE |
| CPCV (4 folds, dense) | 13 min | DONE (77.64% acc) |
| Optuna (100+50 cold, n_jobs=4, pruning+ES fix) | ~1.5 hrs | PENDING |
| Meta + PBO + SHAP | 8 min | PENDING |
| **TOTAL** | **~2 hrs ($0)** | |
| **Without Optuna** | **~45 min ($0)** | |

---

## Data
- **Rows:** 818 (weekly bars, 2010-2026)
- **Base features:** ~3,331 cols
- **Cross features:** ~2.2M cols (min_nonzero=3, was ~1.1M at min_nonzero=8)
- **Total:** ~2.2M features (SPARSE -> converted to dense, fits easily in 64GB+)
- **Dense matrix size:** ~6.1GB (818 x 2.2M x 4 bytes)
- **min_data_in_leaf:** 3 (rare esoteric signals fire 10-20x on weekly)

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
cross feature columns (e.g., JSON has 1.1M names but new NPZ has 2.2M cols -> half the features
get generic `cross_N` names instead of real names, confusing SHAP and audit).

---

## Verify Parquet Freshness
```bash
python3 -c "import pandas as pd; df=pd.read_parquet('/workspace/v3.3/features_BTC_1w.parquet'); print(f'Cols: {len(df.columns)}, Rows: {len(df)}')"
# Expected: columns should match feature_library.py output count
```

## Pipeline Steps
1. Feature build (~30s) -> `features_BTC_1w.parquet`
2. Cross gen (~7 min, 128 threads, min_nonzero=3) -> `v2_crosses_BTC_1w.npz` + `v2_cross_names_BTC_1w.json`
3. LightGBM CPCV (4 folds parallel) -> `model_1w.json`
4. Optuna (200 trials) -> `optuna_configs_1w.json`
5. Meta-labeling -> `meta_model_1w.pkl`
6. LSTM -> `lstm_1w.pt` + `platt_1w.pkl`
7. PBO/Audit -> `validation_report_1w.json`

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
  nohup python -u cloud_run_tf.py --symbol BTC --tf 1w > /workspace/1w_log.txt 2>&1 &
```

**All env vars explained:**
- `SAVAGE22_DB_DIR=/workspace` -- where V1 DBs live (tweets.db, btc_prices.db, etc.)
- `V30_DATA_DIR=/workspace/v3.3` -- where to read/write parquets, NPZs, models. MUST be v3.3, NOT v3.0!
- `PYTHONUNBUFFERED=1` -- real-time log output (no buffering)
- `OMP_NUM_THREADS` / `NUMBA_NUM_THREADS` -- set dynamically by cloud_run_tf.py per phase (cross gen vs training vs Optuna)

---

## Verify Launch (first 30 seconds)

```bash
sleep 30 && head -30 /workspace/1w_log.txt
```

**Must see:**
- "All 16 databases present" (or zero "MISS" lines)
- Row count: ~818 rows
- Base feature count: ~3,331 cols
- Correct data directory: V30_DATA_DIR should show `/workspace/v3.3` not `/workspace/v3.0 (LGBM)`

---

## Verify Cross Features Loaded Correctly

After cross gen completes and training starts, check the log for:
```bash
grep -E "Sparse crosses loaded|cross.*cols|feature_cols.*len" /workspace/1w_log.txt
```

**Expected:** ~2.2M cross feature cols loaded. If you see a number close to 1.1M, the old
cross names JSON was not deleted and is truncating the column list. STOP, delete both
`v2_crosses_BTC_1w.npz` and `v2_cross_names_BTC_1w.json`, and restart.

---

## Verify Multi-Threaded Training

After CPCV training starts (Step 3 in log), check load average:

```bash
# On the machine:
uptime
# Or:
cat /proc/loadavg
```

**Expected:** load avg > (total_cores x 0.3). For a 64-core machine, load avg should be > 19.
If load avg is ~1.0, training is SINGLE-THREADED. This means:
1. `is_enable_sparse` is True but data was converted to dense -> mismatch. Check log for "is_enable_sparse=False" line.
2. OMP_NUM_THREADS not set -> LightGBM uses 1 thread.

**Key log line to look for:**
```
is_enable_sparse=False (data converted to dense -- enables multi-core LightGBM)
```
This confirms dense conversion happened and LightGBM will use multiple cores.
1w data ALWAYS converts to dense (6.1GB << any reasonable machine's RAM).

**Also check RSS matches expected dense matrix size:**
```bash
ps aux | grep cloud_run_tf | grep -v grep | awk '{print $6/1024 " MB"}'
```
RSS should be ~6-10GB (dense matrix + overhead). If RSS is < 1GB, something went wrong.

---

## Parallel CPCV (sparse AND dense)

Parallel CPCV uses ProcessPoolExecutor to train folds concurrently. It works on BOTH:
- **Dense data:** `is_enable_sparse=False` set automatically when sparse->dense conversion happens.
  Each worker gets a slice of the dense numpy array. Multi-core per worker.
- **Sparse data:** CSR matrices pickle correctly. Each worker gets a CSR slice.
  LightGBM uses `is_enable_sparse=True` (default from config.py).

For 1w, data always converts to dense (6.1GB is tiny). is_enable_sparse is automatically set to False.

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
tail -f /workspace/1w_log.txt

# Check for errors
grep -iE "error|traceback|fail|critical|exception" /workspace/1w_log.txt

# Check pipeline progress (which step is running)
grep -E "Step [0-9]|DONE|RUNNING|COMPLETE" /workspace/1w_log.txt

# Check system resources
uptime && free -h && df -h /workspace

# Check for OOM kills:
dmesg | tail -20 | grep -i oom
```

---

## LightGBM Config (from config.py)

| Parameter | Value | Source |
|-----------|-------|--------|
| min_data_in_leaf | 3 | TF_MIN_DATA_IN_LEAF['1w'] |
| num_leaves | 31 | TF_NUM_LEAVES['1w'] |
| max_bin | 255 | V3_LGBM_PARAMS (binary crosses always get 2 bins regardless) |
| CPCV folds | (4,1) = 4 folds | TF_CPCV_GROUPS['1w'] |
| RIGHT_CHUNK | auto (default 500) | 818 rows + 2.2M features fits easily |
| save_binary | Feasible | Dense matrix only 6.1GB, save_binary works fine |

---

## Expected Feature Count (min_nonzero=3)

| Component | Count |
|-----------|-------|
| Base features | ~3,331 |
| Cross features (min_nonzero=3) | ~2.2M (was 1.1M at min_nonzero=8) |
| Total | ~2.2M |

---

## v3.2 Baseline (targets to beat)
| Metric | v3.2 Value |
|--------|------------|
| Accuracy | 73.4% |
| Precision Long | 75.6% |
| Meta AUC | 0.670 |
| PBO | REJECT (small sample, 818 rows) |

---

## V3.3 Actual Results (completed 2026-03-26)
- **CPCV Accuracy**: 71.9% mean (73.3%, 66.7%, 73.0%, 74.6%)
- **PrecL**: 92.0%, 61.9%, 62.5%, 0.0% (varies by fold)
- **PrecS**: 0.0% all folds (only 56 shorts in 818 rows)
- **Trees**: 96, 385, 446, 52 per fold (final: 1041 trees, 347 per class)
- **Model**: 113MB, 2,198,427 features
- **Meta-labeling**: AUC=0.250, acc=0.643 (28 trades — sample too small for valid meta-labeling)
- **PBO**: FAILED (is_metrics not passed — code bug, non-fatal)
- **Machine**: vast.ai RTX 5060 Ti, destroyed after download
- **Status**: COMPLETE. First-ever completed v3.3 model.

**Note:** min_nonzero=3 produces ~2.2M crosses (was ~1.1M at min_nonzero=8)

---

## Failure Modes and What to Check

| Symptom | Cause | Fix |
|---------|-------|-----|
| "WARNING: DB missing" in log | Missing database file | Upload the missing .db, re-run verify script |
| Cross cols ~1.1M instead of ~2.2M | Old v2_cross_names JSON left over | Delete BOTH npz AND json, restart |
| Load avg ~1.0 during training | is_enable_sparse=True on dense data | Check log for "is_enable_sparse=False" line |
| "ModuleNotFoundError: astrology_engine" | astrology_engine.py not in v3.3/ | Copy from project root |
| LSTM fails | features_1w table missing from DB | Non-critical. LSTM needs features DB populated. |
| PBO shows REJECT | Small sample size (818 rows) | Expected for 1w. Not a blocker. |
| "No module named 'lightgbm'" | pip install not run | Run install command above |
| Feature count mismatch vs v3.2 | Stale parquet from old feature_library.py | Delete features_BTC_1w.parquet, restart |
| V30_DATA_DIR shows v3.0 path | Env var not set | Verify `export V30_DATA_DIR=/workspace/v3.3` |
| Optuna errors | Missing optuna/cmaes/colorlog | Run full pip install command |

---

## OPTUNA DEPLOYMENT

### Upload Size for Optuna
- **Total upload: ~500 MB** (parquet + NPZ + cross_names JSON + all DBs)
- Can run locally or on any machine (small upload, fast transfer)
- **Root of warm-start cascade** — 1w must complete Optuna first (1w -> 1d -> 4h -> 1h -> 15m)

### Optuna Timing
- Optuna takes ~2-3x final training time (target after optimizations)
- 1w training = ~13 min, so Optuna = ~1.5 hrs with 100+50 cold trials
- save_binary bridge eliminates redundant EFB construction (Dataset parsed once, reused across all trials)

### If Running Optuna on a Separate Machine
Upload these files:
```
features_BTC_1w.parquet          # base feature parquet
v2_crosses_BTC_1w.npz            # cross feature sparse matrix
v2_cross_names_BTC_1w.json       # cross feature column names
lgbm_dataset_1w.bin              # save_binary output (if available — skips EFB rebuild)
model_1w.json                    # trained model (warm-start seed)
All 16 .db files + kp_history_gfz.txt + astrology_engine.py
v33_code.tar.gz                  # all v3.3/*.py code
```
Total: ~500 MB. Trivial to transfer.

---

## Download Results When Done

```bash
# Check if pipeline completed
grep "DONE\|pipeline complete\|All steps" /workspace/1w_log.txt

# Download all artifacts
scp -P {PORT} root@{HOST}:/workspace/v3.3/model_1w.json .
scp -P {PORT} root@{HOST}:/workspace/v3.3/optuna_configs_1w.json .
scp -P {PORT} root@{HOST}:/workspace/v3.3/meta_model_1w.pkl .
scp -P {PORT} root@{HOST}:/workspace/v3.3/lstm_1w.pt .
scp -P {PORT} root@{HOST}:/workspace/v3.3/platt_1w.pkl .
scp -P {PORT} root@{HOST}:/workspace/v3.3/validation_report_1w.json .
scp -P {PORT} root@{HOST}:/workspace/v3.3/feature_importance_*.json .
scp -P {PORT} root@{HOST}:/workspace/1w_log.txt .
```

**Download partial results after each critical step (vast.ai machines die without warning).**
