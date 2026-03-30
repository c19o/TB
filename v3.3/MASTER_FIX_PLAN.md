# V3.3 Master Fix Plan — Perplexity-Validated + Internal Think Tank
*Generated 2026-03-30 by CEO Agent from 6 department reports*

## Validation Status Key
- **[P]** = Perplexity-validated with full matrix context
- **[T]** = Internal think tank (GPU fork proprietary — Perplexity has no knowledge)
- **[N]** = No validation needed (analysis/documentation only)

---

## TIER 1 — CRITICAL (Fix Before ANY Training)

### 1.1 [P] CPCV Purge Mismatch — LEAKAGE
**Source**: 1w Trade Analyst + Perplexity confirmed
**File**: `ml_multi_tf.py` (purge parameter) + `config.py` (TRIPLE_BARRIER_CONFIG)
**Bug**: purge=6 bars but max_hold_bars=50 for 1w. Bars 7-50 of fold boundary have labels leaking into test period.
**Impact**: ALL 1w results (57.9% CPCV, 75.4%@0.70) are suspect. May apply to other TFs too.
**Fix**: `purge = TRIPLE_BARRIER_CONFIG[tf]['max_hold_bars']` for ALL TFs. Add embargo of equal length.
**Validate.py check**: Assert `cpcv_purge >= triple_barrier_max_hold` for each TF.
**Perplexity says**: "purge ≥ max_hold_bars. The 57.9% must be treated as suspect until re-evaluated with purge=50."

### 1.2 [P] Class Weight Array Misalignment — SHORT Signals Under-Weighted
**Source**: VP QA (BUG-H1) + Perplexity confirmed "most dangerous bug for alpha"
**File**: `ml_multi_tf.py:1057`
**Bug**: sample_weights for ALL rows (incl NaN), _cw_arr for non-NaN only. np.pad misaligns. SHORT 3x upweighting applied to wrong rows.
**Fix**: Apply class weights at CPCV fold level AFTER filtering NaN. Use `lgb.Dataset(weight=final_weights)`.
**Perplexity says**: "If SHORT samples are getting weight 1.0 instead of 3.0, your model is systematically under-weighting the exact rare signal combinations you want."

### 1.3 [P] CPCV Uniqueness Divergence Between Files
**Source**: VP QA (BUG-C3) + Perplexity confirmed
**Files**: `run_optuna_local.py:133` vs `ml_multi_tf.py:181`
**Bug**: Optuna computes uniqueness WITHOUT +1, training WITH +1. Different loss landscapes.
**Fix**: Single source of truth — import from ml_multi_tf.py. The +1 version is correct (inclusive end bar).
**Perplexity says**: "t1+1 is correct for inclusive end bar in half-open Python range convention."

### 1.4 [P] feature_fraction=0.05 Hardcoded in GPU Fork Scripts
**Source**: Matrix Guardian + Perplexity confirmed
**Files**: `gpu_histogram_fork/train_1w_gpu.py:222,530,600` + `gpu_histogram_fork/cupy_gpu_train.py:605`
**Bug**: 5% of 23K EFB bundles = 1,150 per tree. Rare esoteric signals ~5% chance per tree.
**Fix**: `V3_LGBM_PARAMS.get('feature_fraction', 0.9)` — never hardcode.
**Perplexity says**: "feature_fraction operates on EFB bundles. 0.05 is a severe alpha-killing bug."

### 1.5 [P] feature_pre_filter Not Set at Dataset Construction
**Source**: Matrix Guardian + Perplexity confirmed (nuanced)
**File**: `feature_classifier.py:163`
**Bug**: `feature_pre_filter=False` in train params is too late if Dataset is reused across Optuna trials. Features dropped on first build are gone forever.
**Fix**: `lgb.Dataset(X, label=y, params={'feature_pre_filter': False, 'max_bin': 255})`
**Perplexity says**: "Python API builds lazily — params propagate on first build. But if Dataset is reused, features permanently dropped."

### 1.6 [P] min_data_in_leaf Upper Bound Reaches 50 in Optuna
**Source**: Matrix Guardian + Perplexity confirmed
**File**: `run_optuna_local.py:462`
**Bug**: Upper bound `max(50, _tf_mdil + 40)`. Rare signals fire 10-20x, min_data_in_leaf=50 means they can NEVER form a leaf.
**Fix**: Cap at 15: `trial.suggest_int('min_data_in_leaf', max(3, _tf_mdil), 15)`
**Perplexity says**: "min_data_in_leaf applies per-LEAF, not per-EFB-bundle. EFB provides NO protection."

---

## TIER 2 — HIGH (Fix Before Next Cloud Deploy)

### 2.1 [P] device='cpu' Leaks Into GPU Path
**Source**: VP QA (BUG-C1) + Perplexity confirmed
**File**: `ml_multi_tf.py:_train_gpu()`
**Fix**: `params.pop('device', None)` then explicitly set `params['device_type'] = 'cuda_sparse'`
**Perplexity says**: "device is an alias for device_type. Both present = conflict. Version-dependent which wins."

### 2.2 [P] is_enable_sparse Overwritten to False
**Source**: VP QA (BUG-C2) + Perplexity confirmed
**File**: `run_optuna_local.py:473`
**Fix**: Always `'is_enable_sparse': True` regardless of is_sparse variable.
**Perplexity says**: "It's a permission gate, not coercion. True on dense data is a safe no-op. Required for EFB."

### 2.3 [P] Optuna n_jobs Formula Catastrophically Slow
**Source**: Speed Auditor + Perplexity approved
**File**: `run_optuna_local.py:1439`
**Fix**: `n_jobs = max(1, total_cores // 16)` instead of `total_cores // 96`
**Impact**: 6-30x Phase 1 speedup on cloud. ONE LINE CHANGE.
**Perplexity says**: "Each trial is isolated. Thread count doesn't change which features are eligible to split. No rare signal risk."

### 2.4 [P] model_to_string() Called Per-Improvement (87MB × 200 calls)
**Source**: Speed Auditor + Perplexity approved
**Files**: `ml_multi_tf.py:376-380,503-507`
**Fix**: Track `best_iter` integer only, use `booster.predict(num_iteration=best_iter)` at end.
**Perplexity says**: "LightGBM never modifies earlier trees. predict(num_iteration=N) is mathematically identical. This is the canonical pattern."

### 2.5 [P] OMP_NUM_THREADS=4 During Cross Gen Binarization
**Source**: Speed Auditor + Perplexity approved
**File**: `cloud_run_tf.py:460-462`
**Fix**: `NUMBA_NUM_THREADS=cpu_count` for binarization, then 4 for cross matmul loop.
**Perplexity says**: "Percentile-based binarization is deterministic regardless of thread count. Pin np.percentile method='linear'."

### 2.6 [P] Astrology Engine — Bare Excepts Return False Instead of NaN
**Source**: Matrix Guardian + Perplexity confirmed critical
**File**: `astrology_engine.py:698,748,862,899,958,999,1018,1039,1074,1286`
**Fix**: Return `np.nan` not `False`. Catch specific PyEphem exceptions. Log warnings.
**Perplexity says**: "False=0.0 means 'signal definitely not active' (wrong). NaN means 'missing' (correct). LightGBM handles NaN correctly."

### 2.7 [P] subsample=0.8 in build_4h_features.py
**Source**: Matrix Guardian + Perplexity conditional
**File**: `build_4h_features.py:531`
**Fix**: Remove `subsample=0.8` — set to 1.0. But first check if importance scores are used downstream to filter features.
**Perplexity says**: "Harmless IF importance scores aren't used to filter features. Audit downstream usage."

### 2.8 [P] feature_importance_pipeline.py — colsample_bytree=0.01
**Source**: Matrix Guardian
**File**: `feature_importance_pipeline.py:547`
**Fix**: Use `colsample_bytree=0.9` for noise floor test.

---

## TIER 3 — GPU FORK THINK TANK (Proprietary — No Perplexity)

### 3.1 [T] set_external_csr() Not Loading on Cloud Machines
**Source**: Belgium machine crash — `AttributeError: 'Booster' object has no attribute 'set_external_csr'`
**Root cause**: Standard LightGBM .so loaded instead of cuda_sparse fork .so. The swap into site-packages didn't persist or wasn't done correctly.
**Think tank needed**: Reliable .so swap mechanism that survives pip upgrades and container restarts.

### 3.2 [T] Multi-GPU Parallel CPCV via cuda_sparse Fork
**Source**: Speed Auditor fix #2 (Perplexity rejected based on standard LGBM — doesn't apply to our fork)
**Question**: Our cuda_sparse fork handles sparse CSR natively on GPU. Does set_external_csr() produce deterministic results across GPUs? Does our fork use float32 or float64 for histogram construction?
**Think tank needed**: Test 2 GPUs on same fold, compare split decisions on rare features.

### 3.3 [T] GPU Fork Dense Conversion Blocks
**Source**: Matrix Guardian — `train_1w_gpu.py:241-246`, `cupy_gpu_train.py:560-567`
**Bug**: RAM-conditional .toarray() dense conversion before training. Violates sparse-throughout rule.
**Think tank needed**: Is this necessary for the cuda_sparse path? Or can we pass CSR directly?

### 3.4 [T] GPU Fork _train_gpu() Best Model Restore (BUG-H2)
**Source**: VP QA
**Question**: When ds_val is None in _train_gpu(), best_model_str is never set. Is this reachable in our fork's code paths?

### 3.5 [T] GPU Predict Iteration Mismatch (BUG-H3)
**Source**: VP QA
**File**: `run_optuna_local.py:569-572`
**Bug**: GPU predict uses `num_iteration=best_iter` only when `use_gpu`, not `n_gpus`. Edge case.

---

## TIER 4 — 1w RETRAINING PLAN (After Tier 1 Fixes)

### 4.1 [P] Fix CPCV purge=50 and retrain 1w
- All Tier 1 fixes must be applied first
- Retrain locally on 13900K + RTX 3090 (1w fits easily)
- Report per-asset accuracy, not aggregated
- Compare new CPCV with purge=50 to old purge=6 results

### 4.2 [P] Replace Platt Calibration
**Perplexity says**: Use temperature scaling (1 parameter, stable on small sets) or raw probabilities with threshold sweep. Isotonic regression needs MORE data than Platt, not less.

### 4.3 [P] Deploy 1w as LONG-only
**Perplexity says**: "Statistically honest. Add a macro regime filter (price < 20w MA → halt LONG entries) rather than trying to flip SHORT."

---

## TIER 5 — SPEED OPTIMIZATIONS (After Tier 1-2 Verified)

### 5.1 [P] Sequential Validation Gate → Parallel (with fork+CoW)
**File**: `run_optuna_local.py:1308-1328`
**Fix**: Use fork-based Pool (Linux cloud) with copy-on-write CSR. NOT pickle.
**Perplexity says**: "Pickle may fail on 2.9M-feature CSR. Fork+CoW = zero serialization."

### 5.2 [P] Sequential CPCV Final Retrain → Parallel
**File**: `run_optuna_local.py:858-968`
**Same approach as 5.1 for cloud. Single-trial on local.**

### 5.3 [P] Pipeline Step 6 (Optimizer) Blocking Steps 7-9
**File**: `cloud_run_tf.py:603-634`
**Fix**: Launch Steps 6,7,8,9 in parallel. Wait for Step 6 only before Step 10 (audit).

### 5.4 [N] RIGHT_CHUNK=500 for 64GB Local
**File**: `v2_cross_generator.py:144-148`
**Fix**: Increase default for 64GB from 200 to 500. 2.5x less loop overhead.

### 5.5 [N] Local Training: Unset OMP_NUM_THREADS Before Training
**Fix**: Add to local run script or run_optuna_local.py startup.

---

## TIER 6 — DOCUMENTATION (Parallel with fixes)

### 6.1 [N] Create docs/ folder structure
### 6.2 [N] Create MODEL_STATUS.md tracker
### 6.3 [N] Create BUG_REGISTRY.md
### 6.4 [N] Move stale root docs to archive/
### 6.5 [N] Clean memory index (1 duplicate, ~6 stale entries)

---

## EXECUTION ORDER

```
PHASE 1: Tier 1 fixes (all 6 items — blocks everything)
  → Each fix: Perplexity re-check → code change → grep cascade → validate.py check added
  → All in parallel via worktree agents

PHASE 2: Tier 2 fixes (8 items — blocks cloud deploy)
  → Same process, parallel agents

PHASE 3: GPU Fork Think Tank (5 items — blocks GPU training)
  → Internal review, no Perplexity
  → Test on local 3090

PHASE 4: Speed optimizations + cloud deploy prep
  → Maximize compute before spending time on retraining

PHASE 5: 1w Retrain locally with all fixes
  → purge=50, per-asset accuracy, temperature scaling
  → Training on a fully optimized pipeline = faster iteration

PHASE 6: Documentation (parallel with everything)
```
