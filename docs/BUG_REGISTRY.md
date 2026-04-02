# Bug Registry — Savage22 V3.3

All bugs found and fixed (or pending fix) in V3.3 pipeline.

Legend: **[FIXED]** = applied in codebase | **[OPEN]** = validated, not yet applied | **[P]** = Perplexity-validated | **[T]** = Internal think tank

---

## SESSION: 2026-03-29 — Full Pipeline Audit (17 bugs fixed, git: `15bdb43`)

### TIER 1: TRAINING-BLOCKING (Bugs 1–10)

| Bug ID | Description | Root Cause | Fix Applied | Files Changed | Perplexity Validated? | Date |
|--------|-------------|------------|-------------|---------------|----------------------|------|
| BUG-T1-01 | `feature_pre_filter=True` silently kills rare features before trees grow | LightGBM default — features firing < min_data_in_leaf in ANY fold are permanently dropped. Kills esoteric signals on 1h/15m. | Added `"feature_pre_filter": False` to V3_LGBM_PARAMS in config.py | `config.py` | Yes [P] | 2026-03-29 |
| BUG-T1-02 | `OMP_NUM_THREADS=4` persists into training phase — caps parallelism | TRAINING_*.md launch commands export OMP=4 for cross gen (correct) but never unset before LightGBM training (wrong) | Remove OMP/NUMBA exports from launch commands. Set dynamically per phase in cloud_run_tf.py | `cloud_run_tf.py`, all `TRAINING_*.md` | Yes [P] | 2026-03-29 |
| BUG-T1-03 | `NUMBA_NUM_THREADS=4` throttles cross gen prange to 4 cores | Same cause as BUG-T1-02. Numba parallel loops capped at 4 on 192-core machine | Set dynamically per phase (4 for matmul loop, cpu_count for binarization) | `cloud_run_tf.py` | Yes [P] | 2026-03-29 |
| BUG-T1-04 | `num_threads=-1` invalid for core LightGBM API | Undocumented behavior — defers to OMP_NUM_THREADS=4 on cloud | Changed `"num_threads": -1` → `"num_threads": 0` in config.py (0 = auto-detect) | `config.py` | Yes [P] | 2026-03-29 |
| BUG-T1-05 | `is_enable_sparse=True` default → single-threaded when data stays sparse | LightGBM sparse mode serializes OpenMP. 15m/1h always stay sparse → effectively single-threaded | Changed `"is_enable_sparse": True` → `False` in config.py. Workers convert fold slices to dense. | `config.py` | Yes [P] | 2026-03-29 |
| BUG-T1-06 | Final model never sets `num_threads` explicitly | Gets -1 from config, caps at 4 OMP threads | Set `final_params['num_threads'] = os.cpu_count()` after copying V3_LGBM_PARAMS | `ml_multi_tf.py:1263` | Yes [P] | 2026-03-29 |
| BUG-T1-07 | Sequential CPCV path never sets `num_threads` | 15m sequential path gets 4 threads on 512-core machine | Same fix as BUG-T1-06, applied to sequential CPCV path | `ml_multi_tf.py:1046` | Yes [P] | 2026-03-29 |
| BUG-T1-08 | RAM-per-worker underestimates by 3x in cross gen | `* 4` multiplier should be `* 12` at v2_cross_generator.py:652 | Changed `* 4` → `* 12` | `v2_cross_generator.py:652` | No | 2026-03-29 |
| BUG-T1-09 | `_X_all_is_sparse=True` set unconditionally → `.tocsr()` crash on 1w/1d sequential path | Line 630 set `_X_all_is_sparse = True` regardless of whether data was converted to dense | Changed to `_X_all_is_sparse = not _converted_to_dense` | `ml_multi_tf.py:630, 929` | No | 2026-03-29 |
| BUG-T1-10 | `run_optuna_local.py:230` unconditional `.toarray()` → OOM on 15m (8TB) and 1h (425GB) | No RAM check before converting sparse to dense | Added RAM check matching ml_multi_tf.py:617-621 pattern | `run_optuna_local.py:230` | No | 2026-03-29 |

### TRAINING OVERHAUL (git: `9770f53` — "balanced labels, sparse training, cycle features")

| Bug ID | Description | Root Cause | Fix Applied | Files Changed | Perplexity Validated? | Date |
|--------|-------------|------------|-------------|---------------|----------------------|------|
| BUG-T3-01 | Final model trains on last CPCV split (75% data) not full dataset | `ml_multi_tf.py:1236` used last fold's training indices instead of `range(len(df))` | Fixed to train on full dataset | `ml_multi_tf.py:1236` | No | 2026-03-29 |
| BUG-T3-02 | class_weight='balanced' defined in config but NEVER used in training | Config had TF_CLASS_WEIGHT but ml_multi_tf.py never read it | Wired TF_CLASS_WEIGHT → is_unbalance param in lgb.train() | `config.py`, `ml_multi_tf.py` | No | 2026-03-29 |
| BUG-T3-03 | CPCV n_groups/n_test_groups hardcoded, not in config.py | Magic numbers in ml_multi_tf.py:729-737 | Moved to config.py TF_CPCV_CONFIG dict | `ml_multi_tf.py:729-737`, `config.py` | No | 2026-03-29 |
| BUG-T3-04 | 1w num_leaves=63 — overfits 818 rows | Should be 31 for 818-row dataset | Changed to 31 in config.py | `config.py:245` | No | 2026-03-29 |
| BUG-T3-05 | Triple-barrier labels missing — was using fixed label column | HMM sample weights applied unconditionally even when HMM disabled | Removed HMM pre-weighting (MC-1); compute triple-barrier labels in feature builder | `ml_multi_tf.py`, `build_1w_features.py` | No | 2026-03-29 |
| BUG-T3-06 | Cross gen thread cap hardcoded = 128 | `v2_cross_generator.py:654` — never scales to 192+ core machines | Changed to `min(os.cpu_count(), RIGHT_CHUNK_THREADS)` | `v2_cross_generator.py:654` | No | 2026-03-29 |
| BUG-T3-07 | DONE marker written on partial failures | `cloud_run_tf.py:560` writes DONE even if some steps failed | Added success-check before DONE write | `cloud_run_tf.py:560` | No | 2026-03-29 |

---

## SESSION: 2026-03-30 — CEO Agent 6-Department Audit (MASTER_FIX_PLAN.md)

> Status: DISCOVERED — Not yet applied. Requires user approval + Perplexity validation before fix.

### TIER 1: CRITICAL (Fix Before ANY Training)

| Bug ID | Description | Root Cause | Fix Plan | Files Affected | Perplexity Validated? | Date Found |
|--------|-------------|------------|----------|----------------|----------------------|------------|
| BUG-M1-01 | CPCV Purge Mismatch — LEAKAGE | purge=6 bars but max_hold_bars=50 for 1w. Bars 7-50 leak training labels into test. ALL 1w results (57.9% CPCV) are SUSPECT. | `purge = TRIPLE_BARRIER_CONFIG[tf]['max_hold_bars']` + embargo of equal length. Add check to validate.py. | `ml_multi_tf.py`, `config.py` | **YES [P]** — "57.9% must be treated as suspect until re-evaluated with purge=50" | 2026-03-30 |
| BUG-M1-02 | Class Weight Array Misalignment — SHORT under-weighted | sample_weights for ALL rows, _cw_arr for non-NaN only. np.pad misaligns. SHORT 3x upweighting applied to wrong rows. | Apply class weights at CPCV fold level AFTER filtering NaN. `lgb.Dataset(weight=final_weights)` | `ml_multi_tf.py:1057` | **YES [P]** — "most dangerous bug for alpha" | 2026-03-30 |
| BUG-M1-03 | CPCV Uniqueness Divergence Between Files | Optuna computes uniqueness WITHOUT +1, training WITH +1. Different loss landscapes. | Single source of truth — import from ml_multi_tf.py. +1 version is correct (inclusive end bar). | `run_optuna_local.py:133`, `ml_multi_tf.py:181` | **YES [P]** — "t1+1 is correct for inclusive end bar" | 2026-03-30 |
| BUG-M1-04 | feature_fraction=0.05 hardcoded in GPU Fork scripts | 5% of 23K EFB bundles = 1,150 per tree. Rare esoteric signals ~5% chance per tree. Alpha-killing. | `V3_LGBM_PARAMS.get('feature_fraction', 0.9)` — never hardcode | `gpu_histogram_fork/train_1w_gpu.py:222,530,600`, `gpu_histogram_fork/cupy_gpu_train.py:605` | **YES [P]** — "0.05 is a severe alpha-killing bug" | 2026-03-30 |
| BUG-M1-05 | feature_pre_filter Not Set at Dataset Construction | `feature_pre_filter=False` in train params too late if Dataset reused across Optuna trials. Features dropped on first build are gone forever. | `lgb.Dataset(X, label=y, params={'feature_pre_filter': False, 'max_bin': 255})` | `feature_classifier.py:163` | **YES [P]** | 2026-03-30 |
| BUG-M1-06 | min_data_in_leaf upper bound reaches 50 in Optuna | Upper bound `max(50, _tf_mdil + 40)`. Rare signals fire 10-20x — min_data_in_leaf=50 means they can NEVER form a leaf. | Cap at 15: `trial.suggest_int('min_data_in_leaf', max(3, _tf_mdil), 15)` | `run_optuna_local.py:462` | **YES [P]** — "min_data_in_leaf applies per-LEAF, not per-EFB-bundle" | 2026-03-30 |

### TIER 2: HIGH (Fix Before Next Cloud Deploy)

| Bug ID | Description | Files Affected | Perplexity Validated? |
|--------|-------------|----------------|----------------------|
| BUG-M2-01 | device='cpu' leaks into GPU path | `ml_multi_tf.py:_train_gpu()` | YES [P] |
| BUG-M2-02 | is_enable_sparse overwritten to False in Optuna (should be True) | `run_optuna_local.py:473` | YES [P] |
| BUG-M2-03 | Optuna n_jobs formula: `total_cores // 96` → catastrophically slow (fix: `// 16`) | `run_optuna_local.py:1439` | YES [P] — 6-30x speedup |
| BUG-M2-04 | model_to_string() called per-improvement (87MB × 200 calls per trial) | `ml_multi_tf.py:376-380, 503-507` | YES [P] |
| BUG-M2-05 | OMP_NUM_THREADS=4 during cross gen binarization phase | `cloud_run_tf.py:460-462` | YES [P] |
| BUG-M2-06 | Astrology engine bare excepts return False instead of NaN | `astrology_engine.py:698,748,862,899,958,999,1018,1039,1074,1286` | YES [P] — "False=0.0 means signal definitely not active (WRONG)" |

---

## SESSION: 2026-03-21 — V2 Philosophy Audit (10-agent, 23 bugs fixed)

> Historical — V2 pipeline bugs. Reference `AUDIT_FIX_PLAN.md` for full detail.

| Bug ID | Description | Fixed? |
|--------|-------------|--------|
| BUG-V2-01 | 5m/15m SQL missing quote_volume, trades, taker_buy_quote columns | YES |
| BUG-V2-02 | Timestamped onchain data never loaded in build scripts | YES |
| BUG-V2-03 | News schema inversion — loads poor `articles` table instead of rich `streamer_articles` | YES |
| BUG-V2-04 | Space weather format mismatch — RangeIndex instead of DatetimeIndex → all NaN | YES |
| BUG-V2-05 | HMM state mapping inverted in live_trader — bull/bear probabilities scrambled | YES |
| BUG-V2-06 | MI pre-screening inside CPCV folds (leakage + philosophy violation) | YES |
| BUG-V2-07 | Zero-variance filter inside CPCV folds (kills rare signals) | YES |
| BUG-V2-08 | dx_ support=50 filter in feature_library.py | YES |
| BUG-V2-09 | V2_CROSS + V2_LAYERS not mandatory imports in live_trader (silent degradation) | YES |
| BUG-V2-10 | Blanket try/except around V2 layers in live_trader | YES |

---

## Open Bugs (Pending Fix)

| Priority | Bug ID | One-liner | Impact |
|----------|--------|-----------|--------|
| P0 | BUG-M1-01 | 1w CPCV purge=6 < max_hold=50 (leakage) | ALL current 1w results invalid |
| P0 | BUG-M1-02 | SHORT class weight misalignment | PrecS=0.000 on all runs |
| P0 | BUG-M1-04 | feature_fraction=0.05 in GPU fork | Kills esoteric signals in GPU training |
| P0 | BUG-M1-05 | feature_pre_filter not set at Dataset construction | Drops rare signals on Optuna first trial |
| P0 | BUG-M1-06 | min_data_in_leaf cap=50 in Optuna | Rare signals can never form a leaf |
| P1 | BUG-M1-03 | Uniqueness +1 mismatch between files | Different loss landscapes in Optuna vs training |
| P1 | BUG-M2-01 | device='cpu' leaks into GPU path | GPU fork may silently fall back to CPU |
| P1 | BUG-M2-02 | is_enable_sparse wrong in Optuna | EFB disabled during HPO |
| P1 | BUG-M2-03 | Optuna n_jobs formula 6-30x too slow | Wastes cloud GPU hours |
| P1 | BUG-M2-06 | Astrology bare excepts → False not NaN | Esoteric signals polluted with 0.0 |
