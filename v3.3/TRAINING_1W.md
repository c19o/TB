# 1W Training Guide — V3.3 (LOCAL)

Complete step-by-step pipeline for training the 1W (weekly) BTC model locally.
No cloud needed. No questions needed. Follow each step exactly.

---

## Machine

| Component | Spec |
|-----------|------|
| CPU | Intel 13900K (24 cores, 32 threads) |
| GPU | RTX 3090 24GB |
| RAM | 68GB DDR5 |
| OS | Windows 11 |
| Python | 3.10+ |

**ENGINE: CPU for ALL steps.** GPU is 2.2x SLOWER on 818 rows. The GPU histogram fork
is installed but NOT used. Set `ALLOW_CPU=1` to bypass GPU-required checks.

---

## Why 1W First

1W is the ROOT of the warm-start cascade: `1w -> 1d -> 4h -> 1h -> 15m`.
It is the ONLY timeframe that runs Optuna cold (no parent params to inherit).
All downstream timeframes warm-start from 1w's best params.

---

## Step 0: Prerequisites

### 0a. Open a terminal

Open Git Bash, WSL, or any bash-compatible terminal. All commands below use bash syntax.
Set your working directory:

```bash
cd "C:/Users/C/Documents/Savage22 Server/v3.3"
```

### 0b. Delete stale artifacts

Old parquets may have symmetric labels (before the asymmetric barrier fix).
Old NPZs may have different min_nonzero settings. Delete them all.

```bash
rm -f features_BTC_1w.parquet lgbm_dataset_1w.bin
rm -f v2_crosses_BTC_1w.npz v2_cross_names_BTC_1w.json
rm -f model_1w.json model_1w_prev.json
rm -f optuna_configs_1w.json
rm -f meta_model_1w.pkl cpcv_oos_1w.pkl
rm -f shap_analysis_1w.json feature_importance_1w.json
rm -f validation_report_1w.json
echo "Stale artifacts deleted"
```

**CRITICAL:** Both `v2_crosses_BTC_1w.npz` AND `v2_cross_names_BTC_1w.json` must be
deleted together. If you delete one but not the other, ml_multi_tf.py loads the stale
JSON and truncates cross feature columns (e.g., JSON has 1.1M names but NPZ has 2.2M
cols -- half the features get generic `cross_N` names).

### 0c. Verify all 16 databases exist

All databases must be present. The V1 databases (tweets.db, astrology_full.db, etc.)
live in the parent directory. V3.3-specific databases (multi_asset_prices.db, v2_signals.db)
live in v3.3/.

```bash
PROJ="C:/Users/C/Documents/Savage22 Server"
V33="$PROJ/v3.3"
FAIL=0

# V1 databases (parent directory)
for db in btc_prices.db tweets.db news_articles.db sports_results.db space_weather.db \
          onchain_data.db macro_data.db astrology_full.db ephemeris_cache.db \
          fear_greed.db funding_rates.db google_trends.db open_interest.db llm_cache.db; do
  if [ -f "$PROJ/$db" ] || [ -f "$V33/$db" ]; then
    echo "OK   $db"
  else
    echo "MISSING: $db"
    FAIL=$((FAIL+1))
  fi
done

# V3.3 databases
for db in multi_asset_prices.db v2_signals.db; do
  if [ -f "$V33/$db" ]; then
    echo "OK   v3.3/$db"
  else
    echo "MISSING: v3.3/$db"
    FAIL=$((FAIL+1))
  fi
done

# Non-DB required files
[ -f "$PROJ/kp_history_gfz.txt" ] || [ -f "$V33/kp_history_gfz.txt" ] && echo "OK   kp_history_gfz.txt" || { echo "MISSING: kp_history_gfz.txt"; FAIL=$((FAIL+1)); }
[ -f "$V33/astrology_engine.py" ] && echo "OK   astrology_engine.py" || { echo "MISSING: astrology_engine.py"; FAIL=$((FAIL+1)); }

echo ""
echo "DB check: $FAIL missing"
if [ $FAIL -eq 0 ]; then
  echo "ALL 16 DBs + kp_history + astrology_engine PRESENT"
else
  echo "STOP -- fix missing files before proceeding"
fi
```

**ALL must say OK. If ANY says MISSING, stop and find the missing file.**

### 0d. Set environment variables

```bash
export ALLOW_CPU=1
export PYTHONUNBUFFERED=1
export V30_DATA_DIR="C:/Users/C/Documents/Savage22 Server/v3.3"
```

**Why each variable matters:**
- `ALLOW_CPU=1` -- bypasses the GPU-required check. CPU is faster than GPU on 818 rows.
- `PYTHONUNBUFFERED=1` -- real-time log output (no Python buffering).
- `V30_DATA_DIR` -- tells config.py where to read/write parquets, NPZs, and models.
  Without this, it defaults to `v3.0 (LGBM)/` which has OLD data. ALWAYS verify.

### 0e. Install dependencies (if not already installed)

```bash
pip install lightgbm scikit-learn scipy ephem astropy pytz joblib pandas numpy \
  pyarrow optuna hmmlearn numba tqdm pyyaml colorlog sqlalchemy threadpoolctl 2>&1 | tail -3
python -c "import lightgbm; print(f'LightGBM {lightgbm.__version__} OK')"
python -c "import optuna; print(f'Optuna {optuna.__version__} OK')"
```

### 0f. Windows import deadlock workaround

On Windows with Python 3.10, some imports can deadlock due to DLL loading.
If you see the process hang at import time, use the `os.write(1, b'.')` workaround
or upgrade to Python 3.11+.

---

## Step 1: Build Base Features (~37 seconds)

This builds the base feature parquet from all 16 databases. Generates ~3800+ columns
of TA, esoteric, astro, gematria, numerology, space weather, and market signal features.

```bash
cd "C:/Users/C/Documents/Savage22 Server/v3.3" && \
python -u build_features_v2.py --symbol BTC --tf 1w
```

### Verify Step 1

```bash
python -c "
import pandas as pd
df = pd.read_parquet('features_BTC_1w.parquet')
print(f'Rows: {len(df)}')
print(f'Columns: {len(df.columns)}')
has_tb = 'triple_barrier_label' in df.columns
print(f'Triple-barrier labels present: {has_tb}')
if has_tb:
    import numpy as np
    labels = pd.to_numeric(df['triple_barrier_label'], errors='coerce')
    vc = labels.value_counts().sort_index()
    print(f'Label distribution: {dict(vc)}')
    print(f'  0=SHORT: {vc.get(0,0)}, 1=FLAT: {vc.get(1,0)}, 2=LONG: {vc.get(2,0)}')
"
```

**Expected output:**
- Rows: ~818
- Columns: ~3800+ (varies with feature_library.py version)
- Triple-barrier labels present: True
- Asymmetric barriers: tp=3.5 ATR, sl=1.2 ATR, max_hold=4 bars
- SHORT labels should be ~20-30% (was only 7% with old symmetric barriers)

**If columns < 3000:** feature_library.py is not finding all databases. Check Step 0c.

**If triple_barrier_label is missing:** build_features_v2.py did not compute labels.
This is OK -- ml_multi_tf.py will compute them on-the-fly.

---

## Step 2: Build Cross Features (~4 min GPU SpGEMM, ~8 min CPU)

Generates targeted cross features: every binary signal crossed with every other binary signal.
4-tier binarization on ALL numeric columns. Targeted crossing (not ALL x ALL) reduces
noise by 80% while preserving the matrix thesis.

```bash
cd "C:/Users/C/Documents/Savage22 Server/v3.3" && \
python -u v2_cross_generator.py --tf 1w --symbol BTC --save-sparse
```

**Flags explained:**
- `--tf 1w` -- timeframe
- `--symbol BTC` -- asset
- `--save-sparse` -- saves as sparse NPZ (required for ml_multi_tf.py)

If GPU (RTX 3090) is available and CuPy is installed, it will use GPU SpGEMM (~4 min).
If not, CPU fallback (~8 min). Both produce identical results.

Cross gen uses **per-type NPZ checkpointing**: each cross type (dx, ax, etc.) saves a
checkpoint file. If OOM kills the process, restart resumes from the last completed type.

### Verify Step 2

```bash
python -c "
import scipy.sparse as sp
import json, os
X = sp.load_npz('v2_crosses_BTC_1w.npz')
print(f'Cross matrix shape: {X.shape}')
print(f'Non-zero entries: {X.nnz:,}')
print(f'Density: {X.nnz / (X.shape[0] * X.shape[1]) * 100:.4f}%')
if os.path.exists('v2_cross_names_BTC_1w.json'):
    with open('v2_cross_names_BTC_1w.json') as f:
        names = json.load(f)
    print(f'Cross names count: {len(names)}')
    assert len(names) == X.shape[1], f'MISMATCH: names={len(names)} vs cols={X.shape[1]}'
    print('Names match columns: OK')
"
```

**Expected output:**
- Cross matrix shape: (818, ~1,000,000 to 2,000,000)
- Cross names count matches column count exactly
- Density: very low (<1%) -- this is correct for sparse binary crosses

**If shape mismatch between names and columns:** Delete both NPZ and JSON, re-run Step 2.

---

## Step 3: Train WITHOUT Optuna (~15 min)

First training run uses default LightGBM params from config.py. This establishes
a baseline before Optuna optimization.

```bash
cd "C:/Users/C/Documents/Savage22 Server/v3.3" && \
python -u ml_multi_tf.py --tf 1w --boost-rounds 800
```

**What happens internally:**
1. Loads `features_BTC_1w.parquet` (base features)
2. Loads `v2_crosses_BTC_1w.npz` (cross features)
3. Computes triple-barrier labels if not in parquet (tp=3.5, sl=1.2, max_hold=4)
4. Runs CPCV with (5, 2) = 10 splits, 4 unique paths, 60% train fraction
5. Trains LightGBM with 800 boost rounds, early stopping
6. Reports per-split accuracy and mean accuracy

**LightGBM params used (from config.py):**

| Parameter | Value |
|-----------|-------|
| boosting_type | gbdt |
| device | cpu |
| num_leaves | 31 (TF_NUM_LEAVES['1w']) |
| min_data_in_leaf | 5 (TF_MIN_DATA_IN_LEAF['1w']) |
| max_bin | 255 |
| learning_rate | 0.03 |
| feature_fraction | 0.1 |
| num_boost_round | 800 |
| force_col_wise | True |
| feature_pre_filter | False |
| is_enable_sparse | True |
| class_weight | balanced |

### Verify Step 3

```bash
# Check model exists
ls -la model_1w.json

# Check log for accuracy
# Look for lines like "CPCV mean accuracy: XX.X%"
# Or check the model file size
python -c "
import os
size_mb = os.path.getsize('model_1w.json') / (1024*1024)
print(f'Model size: {size_mb:.1f} MB')
"
```

**Expected output:**
- `model_1w.json` exists, ~50-100 MB
- CPCV accuracy: ~65-75% (with default params, before Optuna)
- 10 CPCV splits reported (from 5 groups, K=2)
- SHORT precision should be >0% (was always 0% with old symmetric barriers)

**If accuracy < 60%:** Something is wrong. Check that V30_DATA_DIR is set correctly
and that the parquet has the new asymmetric labels (not old symmetric ones).

**If model_1w.json does not exist:** Check for errors in the output. Common causes:
- Missing database files (check Step 0c)
- Missing optuna_configs_1w.json is OK at this step -- it only matters for Step 5

---

## Step 4: Run Optuna (~8 min)

Optuna searches for optimal LightGBM hyperparameters using a two-phase approach:

**Phase 1 (rapid search):** 20 trials (2 seeded + 8 random + 10 TPE), 2-fold CPCV,
60 rounds max, LR=0.15, early stopping at 15 rounds.

**Validation Gate:** Top-3 configs re-evaluated with 4-fold CPCV, 200 rounds, LR=0.08,
early stopping at 50 rounds.

```bash
cd "C:/Users/C/Documents/Savage22 Server/v3.3" && \
python -u run_optuna_local.py --tf 1w
```

**What Optuna searches:**
- num_leaves: [4, 31] (v3.2's best was num_leaves=7)
- min_data_in_leaf: [3, 15]
- feature_fraction: [0.02, 0.3] (lower bound 0.02, not 0.01)
- feature_fraction_bynode: [0.1, 1.0]
- bagging_fraction: [0.5, 1.0]
- lambda_l1, lambda_l2: [0, 10]
- min_gain_to_split: [0, 5]
- path_smooth: [0, 10]
- max_depth: [3, 12]

**1W is the root of warm-start cascade.** It runs ALL trials cold (no parent params).
Downstream TFs (1d, 4h, etc.) will inherit 1w's best params as seeded trials.

**Optuna implementation details:**
- **PatientPruner** wraps MedianPruner with patience=5 and min_delta=0.001 (5 stagnant reports = prune)
- **Dataset.subset()** used for Optuna fold construction (1000x faster than `reference=` re-parsing)
- **save_binary bridge:** Optuna saves `lgbm_dataset_1w.bin` via `save_binary()` which the final training step reuses (skips EFB rebuild)
- **40% accuracy floor:** Model is NOT saved if CPCV mean accuracy < 40% (prevents garbage models)

### Verify Step 4

```bash
python -c "
import json
with open('optuna_configs_1w.json') as f:
    cfg = json.load(f)
if 'best_params' in cfg:
    print('Best params found:')
    for k, v in cfg['best_params'].items():
        print(f'  {k}: {v}')
elif '1w' in cfg:
    print('Best params found:')
    for k, v in cfg['1w'].items():
        print(f'  {k}: {v}')
else:
    print('Keys in config:', list(cfg.keys()))
"
```

**Expected output:**
- `optuna_configs_1w.json` exists with `best_params` section
- Phase 1: 20 trials completed (~4 min)
- Validation Gate: top-3 validated (~4 min)
- Best accuracy should be higher than Step 3 baseline

---

## Step 5: Retrain with Optuna Params (~15 min)

Now retrain the model using the optimized params from Step 4. ml_multi_tf.py
automatically reads `optuna_configs_1w.json` and uses those params instead of defaults.

```bash
cd "C:/Users/C/Documents/Savage22 Server/v3.3" && \
python -u ml_multi_tf.py --tf 1w --boost-rounds 800
```

**What changes vs Step 3:**
- LightGBM params come from `optuna_configs_1w.json` instead of config.py defaults
- Old model backed up to `model_1w_cpcv_backup.json` automatically
- Full CPCV with (5, 2) = 10 splits, K=2
- Learning rate: 0.03 (final rate, not Optuna's 0.15 search rate)

### Verify Step 5

```bash
python -c "
import os
size_mb = os.path.getsize('model_1w.json') / (1024*1024)
print(f'Model size: {size_mb:.1f} MB')
print(f'Model updated: {os.path.getmtime(\"model_1w.json\")}')
"
```

**Expected output:**
- `model_1w.json` updated (check timestamp)
- CPCV accuracy: ~67-77% (should improve 2-5% over Step 3 default params)
- SHORT precision should still be >0%
- Model size: ~50-100 MB

---

## Step 6: Trade Optimizer (~5 min)

Optimizes trade execution parameters (confidence thresholds, position sizing,
stop-loss/take-profit multipliers) using Optuna TPE with Sortino objective.

```bash
cd "C:/Users/C/Documents/Savage22 Server/v3.3" && \
python -u exhaustive_optimizer.py --tf 1w --n-trials 200
```

**What it optimizes:**
- Confidence threshold (when to enter trades)
- Leverage per regime (bull/bear/sideways/crash)
- Risk per trade
- Stop-loss and take-profit multipliers
- Hold period adjustments
- Regime-dependent sizing

### Verify Step 6

```bash
python -c "
import json
with open('optuna_configs_1w.json') as f:
    cfg = json.load(f)
print('Config keys:', list(cfg.keys()))
# Trade params should now be present alongside model params
"
```

**Expected output:**
- `optuna_configs_1w.json` updated with trade execution parameters
- 200 trials completed
- Best Sortino ratio reported

---

## Step 7: Meta-Labeling (~3 min)

Trains a secondary classifier that answers: "Given the base model says LONG/SHORT,
should we actually take this trade?" Uses CPCV out-of-sample predictions as training
data (no leakage).

```bash
cd "C:/Users/C/Documents/Savage22 Server/v3.3" && \
python -u meta_labeling.py --tf 1w
```

### Verify Step 7

```bash
ls -la meta_model_1w.pkl cpcv_oos_1w.pkl 2>/dev/null
python -c "
import os
for f in ['meta_model_1w.pkl', 'cpcv_oos_1w.pkl']:
    if os.path.exists(f):
        print(f'{f}: {os.path.getsize(f)/1024:.1f} KB')
    else:
        print(f'{f}: NOT FOUND')
"
```

**Expected output:**
- `meta_model_1w.pkl` exists
- `cpcv_oos_1w.pkl` exists (CPCV out-of-sample predictions)
- Meta AUC reported (may be low with only 818 rows -- this is expected)
- Meta accuracy reported

**Note:** With only 818 weekly bars, meta-labeling has very few OOS trades to learn from.
A low AUC (0.3-0.6) is expected and not a blocker. It becomes more meaningful on
higher-frequency timeframes with more samples.

---

## Complete Artifact Checklist

After all 7 steps, verify ALL artifacts exist:

```bash
echo "=== ARTIFACT CHECK ==="
for f in features_BTC_1w.parquet \
         v2_crosses_BTC_1w.npz \
         v2_cross_names_BTC_1w.json \
         model_1w.json \
         optuna_configs_1w.json \
         meta_model_1w.pkl \
         cpcv_oos_1w.pkl; do
  if [ -f "$f" ]; then
    SIZE=$(ls -la "$f" | awk '{print $5}')
    echo "OK   $f ($SIZE bytes)"
  else
    echo "MISSING: $f"
  fi
done
```

**All 7 files must be present before proceeding to 1d training.**

---

## Expected Outcomes

| Metric | Expected Range |
|--------|---------------|
| Base features | ~3800+ columns |
| Cross features | ~1-2M columns (targeted crossing, 4-tier binarization) |
| Total features | ~1-2M |
| CPCV accuracy (default) | ~65-75% |
| CPCV accuracy (Optuna) | ~67-77% |
| SHORT precision | 10-50% (was 0% with old symmetric barriers) |
| LONG precision | 60-90% |
| Model size | ~50-100 MB |
| Total pipeline time | ~50 min (Steps 1-7) |

---

## Data Specifications

| Property | Value |
|----------|-------|
| Rows | ~818 (weekly bars, 2010-2026) |
| Asymmetric barriers | tp=3.5 ATR, sl=1.2 ATR, max_hold=4 bars |
| CPCV config | (5, 2) = 10 splits, 4 unique paths, 60% train |
| min_data_in_leaf | 5 (config: TF_MIN_DATA_IN_LEAF['1w']) |
| num_leaves | 31 (config: TF_NUM_LEAVES['1w']) |
| Dense matrix RAM | ~6 GB (818 x 2M x 4 bytes) -- fits easily in 68GB |
| Optuna Phase 1 | 20 trials, 2-fold, 60 rounds, LR=0.15 |
| Optuna Validation | top-3, 4-fold, 200 rounds, LR=0.08 |

---

## Timing Breakdown

| Step | Description | Time |
|------|-------------|------|
| 0 | Prerequisites (delete + verify) | 1 min |
| 1 | Build base features | ~37 sec |
| 2 | Build cross features | ~4-8 min |
| 3 | Train (default params) | ~15 min |
| 4 | Optuna search | ~8 min |
| 5 | Retrain (Optuna params) | ~15 min |
| 6 | Trade optimizer | ~5 min |
| 7 | Meta-labeling | ~3 min |
| **TOTAL** | | **~50 min** |

---

## Failure Modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Hang at import | Windows DLL deadlock | Use Python 3.11+ or `os.write(1, b'.')` workaround |
| "GPU REQUIRED" error | ALLOW_CPU not set | `export ALLOW_CPU=1` |
| Columns < 3000 in parquet | Missing databases | Run Step 0c, fix missing DBs |
| Cross cols ~1.1M not ~2M | Old min_nonzero setting or stale JSON | Delete BOTH npz AND json, re-run Step 2 |
| "V30_DATA_DIR shows v3.0" | Env var not set | `export V30_DATA_DIR="C:/Users/C/Documents/Savage22 Server/v3.3"` |
| 0% SHORT precision | Stale parquet with old symmetric labels | Delete parquet (Step 0b), rebuild from Step 1 |
| "ModuleNotFoundError: astrology_engine" | astrology_engine.py not in v3.3/ | Copy from project root |
| "No module named lightgbm" | Deps not installed | Run Step 0e |
| OOM during training | Unlikely on 68GB for 1w | Check no other heavy processes running |
| Optuna hangs | n_jobs too high | Lower n_jobs or run single-threaded |
| Load avg ~1.0 during training | is_enable_sparse on dense data (single-thread) | Check log for "is_enable_sparse=False" |
| Feature count mismatch | Stale parquet from old feature_library.py | Delete parquet, rebuild from Step 1 |
| model_1w.json not created | Check stderr for traceback | Fix the reported error, re-run |

---

## Important Notes

1. **1W is the ROOT of warm-start cascade.** It must complete fully (all 7 steps)
   before starting 1d training. 1d inherits 1w's Optuna params as seeded trials.

2. **GPU histogram fork is installed but NOT used.** CPU is 2.2x faster than GPU
   on 818 rows because GPU kernel launch overhead dominates at this scale.

3. **Sparse CSR semantics.** Cross features are pure 0/1 binary. Structural zeros
   in CSR = 0.0 = "feature OFF" (conditions not met). This is correct, not missing.
   Base features can have NaN = missing (LightGBM learns split direction).

4. **feature_pre_filter=False is CRITICAL.** Setting True silently kills rare esoteric
   features at LightGBM Dataset construction. This is set in config.py and must never
   be changed.

5. **class_weight='balanced' for 1w.** Automatically reweights by inverse class
   frequency to handle the natural BTC upward bias in label distribution.

6. **Upload size if running Optuna separately:** ~500MB (parquet + NPZ + JSON + DBs + code).
   But for 1w there is no reason to run remotely -- local is fast enough.

---

## After 1W: Start 1D

Once all 7 artifacts are verified, proceed to 1d training. The 1d Optuna will
warm-start from 1w's best params (fewer trials needed: 15 instead of 20).

```bash
# Verify 1w is complete
ls model_1w.json optuna_configs_1w.json meta_model_1w.pkl
# Then follow TRAINING_1D.md
```
