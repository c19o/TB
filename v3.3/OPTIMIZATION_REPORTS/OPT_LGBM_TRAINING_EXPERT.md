# LightGBM Training Expert Audit Report

**Date:** 2026-03-30
**Auditor:** LightGBM Training Expert (Claude Opus 4.6)
**Scope:** All training paths in ml_multi_tf.py, config.py, run_optuna_local.py + auxiliary trainers
**Matrix thesis:** 2.9M features, sparse CSR, EFB, cuda_sparse fork, rare signals (10-20 fires) are the edge

---

## EXECUTIVE SUMMARY

**Overall status: PASS with 5 findings (2 MEDIUM, 3 LOW)**

The core training pipeline is well-hardened. All critical parameters (feature_fraction >= 0.7, feature_pre_filter=False, min_data_in_bin=1, is_enable_sparse=True) are correctly set in the primary training paths. The cuda_sparse GPU path is properly configured. Two medium findings relate to missing `min_data_in_bin` in a parent Dataset construction and missing `params=` in auxiliary trainer Dataset calls.

---

## 1. feature_fraction >= 0.7

### PASS -- All paths verified

| Location | Value | Status |
|----------|-------|--------|
| config.py V3_LGBM_PARAMS (L335) | 0.9 | PASS |
| config.py feature_fraction_bynode (L336) | 0.8 | PASS |
| run_optuna_local.py objective (L553) | suggest_float(0.7, 1.0) | PASS -- floor is 0.7 |
| run_optuna_local.py feature_fraction_bynode (L554) | suggest_float(0.7, 1.0) | PASS -- floor is 0.7 |
| ml_multi_tf.py Optuna overlay (L1584-1594) | Copies from Optuna best_params (floored at 0.7 by Optuna) | PASS |
| feature_classifier.py (L152) | 0.8 | PASS |
| leakage_check.py (L195) | 0.9 | PASS |

**No path can produce feature_fraction < 0.7.** Optuna's search space is hard-floored at 0.7 (L553), and config.py default is 0.9.

---

## 2. feature_pre_filter=False in ALL lgb.Dataset() calls

### PASS (primary paths) / MEDIUM finding (auxiliary paths)

**Primary training paths -- ALL PASS:**

| File:Line | Dataset params | Status |
|-----------|---------------|--------|
| ml_multi_tf.py _cpcv_split_worker (L542) | `{'feature_pre_filter': False, 'max_bin': ..., 'min_data_in_bin': 1}` | PASS |
| ml_multi_tf.py _isolated_fold_worker (L696) | `{'feature_pre_filter': False, 'max_bin': ..., 'min_data_in_bin': 1}` | PASS |
| ml_multi_tf.py _gpu_fold_worker (L843) | `{'feature_pre_filter': False, 'max_bin': ..., 'min_data_in_bin': 1}` | PASS |
| ml_multi_tf.py sequential fold (L2303) | `{'feature_pre_filter': False, 'max_bin': ..., 'min_data_in_bin': 1}` | PASS |
| ml_multi_tf.py final retrain (L2607) | `{'feature_pre_filter': False, 'max_bin': ..., 'min_data_in_bin': 1}` | PASS |
| ml_multi_tf.py parent_ds from binary (L2056) | `{'feature_pre_filter': False, 'max_bin': ..., 'min_data_in_bin': 1}` | PASS |
| ml_multi_tf.py parent_ds fallback (L2092) | `{'feature_pre_filter': False, 'max_bin': ...}` | PASS (but see Finding #1) |
| run_optuna_local.py _parallel_dataset_construct (L1115, 1130, 1160) | All have `feature_pre_filter: False` | PASS |
| run_optuna_local.py parent_ds from binary (L1285-1288) | `feature_pre_filter: False` | PASS |
| feature_classifier.py (L163) | `{'feature_pre_filter': False, 'max_bin': 255}` | PASS |
| meta_labeling.py (L155) | `params={'feature_pre_filter': False}` | PASS |
| validate.py test datasets (L543, L572) | `params={'feature_pre_filter': False}` | PASS |

**FINDING #2 (MEDIUM): v2_multi_asset_trainer.py -- lgb.Dataset() calls MISSING `params=` kwarg**

- **Lines 461-464, 627-630:** `lgb.Dataset(X_train_es, label=y_train_es, weight=w_train_es, feature_name=..., free_raw_data=False)` -- NO `params=` dict.
- feature_pre_filter defaults to True when not in params= kwarg. The V3_LGBM_PARAMS has it, but it's passed to `lgb.train()`, not `lgb.Dataset()`. **LightGBM bakes feature_pre_filter at Dataset construction time, not at train time.** If the parent params dict has it, LightGBM may or may not pick it up depending on version.
- **Impact:** This is the multi-asset trainer (v2_multi_asset_trainer.py), which may not be in the active v3.3 pipeline. If used, rare features could be silently killed.
- **Fix:** Add `params={'feature_pre_filter': False, 'max_bin': 255, 'min_data_in_bin': 1}` to both Dataset calls.

**leakage_check.py (L201-202):** lgb.Dataset() calls have no `params=` kwarg. However, this is a diagnostic tool, not production training. LOW priority.

---

## 3. min_data_in_bin=1 everywhere

### PASS (primary paths) / FINDING #1 (MEDIUM)

All primary training paths correctly set `min_data_in_bin=1` in Dataset params. One exception:

**FINDING #1 (MEDIUM): ml_multi_tf.py L2092 -- parent_ds fallback missing `min_data_in_bin`**

```python
params={'feature_pre_filter': False, 'max_bin': _base_lgb_params.get('max_bin', 255)},
```

This is the single-threaded fallback path for parent Dataset construction (only hit when parallel construction fails AND no binary cache exists). Missing `min_data_in_bin=1` means LightGBM defaults to `min_data_in_bin=5`, which could merge rare binary feature bins.

- **Risk:** LOW-MEDIUM. This is a fallback of a fallback. The primary parallel construction path (L1114-1120 in run_optuna_local.py) correctly has `min_data_in_bin=1`. The binary cache path (L2056) also has it.
- **Fix:** Add `'min_data_in_bin': 1` to the params dict at L2092.

All other locations verified:
- config.py V3_LGBM_PARAMS (L327): `"min_data_in_bin": 1` -- PASS
- All _cpcv_split_worker, _isolated_fold_worker, _gpu_fold_worker Dataset params: PASS
- run_optuna_local.py _parallel_dataset_construct (L1115-1116, L1133, L1160): PASS
- feature_classifier.py: NOT SET -- uses LightGBM default (5). LOW priority (not main pipeline).

---

## 4. is_enable_sparse=True

### PASS

- config.py V3_LGBM_PARAMS (L326): `"is_enable_sparse": True` -- PASS
- All training paths copy from V3_LGBM_PARAMS, inheriting this setting.
- run_optuna_local.py objective (L566): explicitly sets `'is_enable_sparse': True` -- PASS
- run_optuna_local.py validate_config (L744): explicitly sets it -- PASS
- run_optuna_local.py final_retrain (L947): explicitly sets it -- PASS
- run_optuna_local.py _parallel_dataset_construct (L1116, L1131, L1160): explicitly sets it -- PASS

**No path converts sparse to dense before passing to LightGBM.** is_enable_sparse flows correctly through all paths.

---

## 5. num_threads=0 for lgb.train()

### PASS with correct adaptive logic

- config.py V3_LGBM_PARAMS (L323): `"num_threads": 0` -- PASS (0 = OpenMP auto-detect)
- ml_multi_tf.py (L1561-1566): Caps to 32 for < 10K rows -- **CORRECT** (LightGBM docs: >64 threads on <10K rows = poor scaling)
- ml_multi_tf.py (L1856-1858): Sets per-worker threads for parallel CPCV -- **CORRECT** (avoids oversubscription)
- ml_multi_tf.py final_params (L2576): Sets `num_threads = _total_cores` -- **CORRECT** (explicit cgroup-aware count)
- run_optuna_local.py objective (L567): `num_threads = max(1, cpu_count // OPTUNA_N_JOBS)` -- **CORRECT** (fair share per trial)
- run_optuna_local.py validate_config (L745): `num_threads = 0` -- PASS
- run_optuna_local.py final_retrain (L949): `num_threads = 0` -- PASS

**FINDING #3 (LOW): feature_importance_pipeline.py uses `n_jobs=-1` in LGBMClassifier**

Lines 411, 484, 546 use `lgb.LGBMClassifier(n_jobs=-1, ...)`. In the sklearn API, n_jobs=-1 maps to num_threads=-1 internally. While LightGBM's C++ library treats -1 the same as 0 (all cores), the CLAUDE.md rule states "NEVER use -1". This is a non-production auxiliary tool, so impact is nil.

---

## 6. cuda_sparse GPU params

### PASS -- All GPU paths correctly configured

**_train_gpu() (ml_multi_tf.py L354-452):**
- Sets `device_type = 'cuda_sparse'` -- PASS
- Sets `gpu_device_id` from param -- PASS
- Removes `force_col_wise` (conflicts with GPU histogram builder) -- PASS
- Sets `histogram_pool_size = 512` if not present -- PASS
- Removes `device` alias (prevents conflict) -- PASS

**_cpcv_split_worker GPU path (L548-581):**
- Sets `device_type = 'cuda_sparse'` -- PASS
- Removes `force_col_wise` -- PASS
- Sets `histogram_pool_size = 512` if missing -- PASS
- Uses `lgb.Booster + set_external_csr()` -- PASS

**_gpu_fold_worker (L768-943):**
- Sets `CUDA_VISIBLE_DEVICES` BEFORE importing LightGBM -- PASS (critical for isolation)
- Sets `gpu_device_id = 0` (correct -- CUDA_VISIBLE_DEVICES remaps) -- PASS
- Removes `force_col_wise`, `force_row_wise`, `device` -- PASS
- Sets `histogram_pool_size = 512` -- PASS

**run_optuna_local.py objective GPU path (L590-601):**
- Sets `device_type = 'cuda_sparse'` -- PASS
- Sets `gpu_device_id = trial.number % n_gpus` -- PASS
- Sets `histogram_pool_size = 512` -- PASS
- Removes `force_col_wise`, `force_row_wise`, `device` -- PASS

---

## 7. EFB bundle efficiency with max_bin=255 and binary features

### PASS -- Correctly configured

- config.py (L321): `"max_bin": 255` -- PASS
- EFB_PREBUNDLE_ENABLED (L367-373): All TFs set to True -- PASS
- With max_bin=255, LightGBM's EFB can pack up to 127 binary features per bundle (each binary feature uses 2 bins, and 127 * 2 = 254 <= 255).
- For 2.9M binary cross features, this produces ~23K EFB bundles -- a 128x histogram reduction.
- `bin_construct_sample_cnt = 5000` (L328): Correct optimization for binary features (default 200K is overkill).
- run_optuna_local.py objective (L577): `max_bin = 255` locked -- PASS

**No path overrides max_bin.** All Dataset params consistently pass `max_bin: 255`.

---

## 8. No dense conversion of sparse CSR

### PASS -- No problematic dense conversions

The only `.toarray()` calls in ml_multi_tf.py:

1. **L1421:** `X_all[:, esoteric_indices].toarray()` -- Extracts only base esoteric feature columns (small subset, ~50 cols) for computing sample weights. NOT passed to LightGBM. **Safe.**

2. **L2007:** `_hmm_slice.toarray()` -- Extracts 4 HMM columns as dense overlay (4 cols). These are separated from the main CSR matrix and hstacked back as sparse before training. **Safe.**

**No path converts the full feature matrix to dense.** The sparse CSR flows untouched through all training paths to LightGBM/cuda_sparse.

---

## FINDINGS SUMMARY

| # | Severity | File:Line | Issue | Fix |
|---|----------|-----------|-------|-----|
| 1 | MEDIUM | ml_multi_tf.py:2092 | Parent Dataset fallback missing `min_data_in_bin=1` in params | Add `'min_data_in_bin': 1` to params dict |
| 2 | MEDIUM | v2_multi_asset_trainer.py:461-464, 627-630 | lgb.Dataset() calls missing `params=` kwarg -- feature_pre_filter defaults to True at Dataset construction | Add `params={'feature_pre_filter': False, 'max_bin': 255, 'min_data_in_bin': 1}` |
| 3 | LOW | feature_importance_pipeline.py:411,484,546 | Uses `n_jobs=-1` instead of `num_threads=0` convention | Change to `n_jobs=0` or leave (non-production tool) |
| 4 | LOW | feature_classifier.py:163 | Dataset params missing `min_data_in_bin=1` | Add `'min_data_in_bin': 1` to _ds_params |
| 5 | LOW | leakage_check.py:201-202 | lgb.Dataset() missing `params=` kwarg entirely | Add `params={'feature_pre_filter': False, 'min_data_in_bin': 1}` (diagnostic tool) |

---

## PARAMETER FLOW VERIFICATION

```
config.py V3_LGBM_PARAMS
    |
    +--> ml_multi_tf.py: V2_LGBM_PARAMS = V3_LGBM_PARAMS.copy() (L1111)
    |       |
    |       +--> _base_lgb_params = V2_LGBM_PARAMS.copy() (L1530)
    |       |       |
    |       |       +--> Per-TF overlays: min_data_in_leaf, num_leaves, class_weight, force_row_wise
    |       |       +--> Optuna overlay: reads optuna_configs_{tf}.json, applies tunable keys
    |       |       +--> Passed to all fold workers via lgb_params arg
    |       |
    |       +--> final_params = V2_LGBM_PARAMS.copy() (L2574)
    |               +--> Optuna keys copied from _base_lgb_params
    |
    +--> run_optuna_local.py: V3_LGBM_PARAMS.copy() in objective (L564), validate_config (L742), final_retrain (L945)
            +--> Each explicitly re-applies is_enable_sparse=True, num_threads
            +--> Optuna trial params overlaid (all floored at 0.7 for feature_fraction)
```

**All paths inherit from V3_LGBM_PARAMS.** No path constructs params from scratch that could miss critical settings.

---

## VERDICT

The training pipeline is production-ready. The two MEDIUM findings are in edge-case paths (fallback parent Dataset construction, auxiliary multi-asset trainer). The core CPCV training loop across all three worker types (_cpcv_split_worker, _isolated_fold_worker, _gpu_fold_worker), the sequential fold path, the final retrain, and the Optuna search are all correctly configured.

**Rare signal protection is intact across all primary paths:**
- feature_fraction >= 0.7 (Optuna floor + config default 0.9)
- feature_pre_filter=False in all production Dataset() calls
- min_data_in_bin=1 in all production Dataset() calls
- is_enable_sparse=True flows from config through all paths
- No dense conversion of the full feature matrix anywhere
- cuda_sparse GPU path correctly strips CPU-only params and sets histogram_pool_size
