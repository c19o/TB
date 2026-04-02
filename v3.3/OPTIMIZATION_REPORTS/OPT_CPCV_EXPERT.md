# CPCV Expert Audit Report -- ml_multi_tf.py

**Auditor:** CPCV Validation Expert (Lopez de Prado methodology)
**Date:** 2026-03-30
**File:** `v3.3/ml_multi_tf.py` (2773 lines)
**Scope:** All 6 CPCV execution paths + split generation + dispatch logic

---

## EXECUTIVE SUMMARY

The CPCV implementation is **SOUND** on all critical leakage dimensions. Purge, embargo, HMM per-fold fitting, IS/OOS metric separation, and `feature_pre_filter=False` are all correct. Two minor findings (one cosmetic, one low-risk) are documented below but neither constitutes data leakage or accuracy corruption.

**Verdict: PASS -- safe to train.**

---

## 1. PURGE AND EMBARGO VERIFICATION

### 1.1 Purge = max_hold_bars per TF (PASS)

`TRIPLE_BARRIER_CONFIG` in `feature_library.py` defines per-TF `max_hold_bars`:
- 15m: 24, 1h: 48, 4h: 72, 1d: 90, 1w: 50

At line 1448, `max_hold` is loaded from `TRIPLE_BARRIER_CONFIG[tf_name]` and passed to `_generate_cpcv_splits()` as `max_hold_bars`. The split generator (line 280-291) uses this for purging:
- With t0/t1 arrays: purges train samples whose label window `[t0, t1]` overlaps any test group
- Without t0/t1 (fallback): purges samples where `[i, i + max_hold_bars]` overlaps any test group
- Per-group purging (line 274): each test group is checked independently -- correct for non-contiguous groups

**Verified: purge uses TF-specific max_hold_bars, not a fixed constant.**

### 1.2 Embargo = max(0.01, max_hold/n) (PASS)

Line 1484:
```python
_embargo_pct = max(0.01, max_hold / n)
```

This matches the Lopez de Prado formula. It is also computed inside `_generate_cpcv_splits()` at line 223:
```python
effective_pct = max(embargo_pct, max_hold_bars / n_samples)
```

Both the caller and callee apply `max(0.01, max_hold/n)`. The embargo is applied per test group boundary (line 295-299), removing training samples in the embargo zone after each test group end. **Correct.**

### 1.3 Per-Group Purge (Non-Contiguous Test Groups) (PASS)

Line 274 iterates over each `g in test_group_ids` independently. This prevents under-purging when test groups are non-contiguous (e.g., groups 2 and 5 selected as test). The comment at line 269 correctly explains why min/max of combined test set would be wrong.

---

## 2. CUDA_VISIBLE_DEVICES ISOLATION (PASS)

### _gpu_fold_worker (line 768-943)

Lines 778-779:
```python
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
```

Lines 784: `import lightgbm as lgb` comes AFTER environment variable setting.

The function signature starts with `import os` at line 776, sets env vars, then imports LightGBM. Since this runs via `mp.get_context('spawn').Process`, the subprocess starts fresh -- no parent LightGBM import contaminates it. **Correct.**

GPU ID mapping at line 855: `gpu_device_id = 0` after CUDA_VISIBLE_DEVICES remapping. **Correct.**

---

## 3. DATA LEAKAGE ANALYSIS (NO LEAKAGE FOUND)

### 3.1 Mmap Sharing (PASS)

The GPU-parallel and subprocess-isolated paths share data via mmap'd `.npy` files:
- `save_csr_npy()` writes CSR components + y + weights to shared_dir (line 1688-1690)
- Workers load with `mmap_mode='r'` (line 790-792, 643-645) -- read-only, no cross-process mutation
- Each worker slices `X_all[train_idx]` and `X_all[test_idx]` independently
- No worker writes back to the shared arrays

**No data leakage via mmap.**

### 3.2 Sample Weight Sharing (PASS)

Sample weights (uniqueness + esoteric + class weights) are computed once on the full dataset and shared. These weights are **not target-derived** -- they come from label overlap geometry (uniqueness), esoteric column activity counts, and static class weight configs. No leakage.

### 3.3 Parent Dataset EFB Reuse (PASS -- with note)

The sequential path builds a `_parent_ds` on valid rows (line 2063-2094) for EFB/bin threshold reuse. This means bin thresholds are computed on the full dataset, not per-fold. This is **acceptable** for LightGBM because:
- Bin thresholds are data-agnostic (quantile-based on feature values, not target-correlated)
- Lopez de Prado's purging concerns label leakage, not histogram bin boundaries
- LightGBM's `reference=` parameter is designed for this exact use case

### 3.4 SharedMemory IPC (PASS)

The CPU-parallel path (line 1862-1887) places CSR arrays in SharedMemory with `copy=False` on the worker side (line 489). Workers reconstruct CSR from shared buffers. No target information leaks between folds since each worker operates on its own `train_idx`/`test_idx` slice.

---

## 4. HMM OVERLAY -- PER-FOLD, NO LOOKAHEAD (PASS)

All 5 execution paths handle HMM correctly:

### 4.1 GPU-Parallel Path (line 1694-1716)
- `fit_hmm_on_window(_train_end_gpu)` called per fold using `timestamps[train_idx_j[-1]]`
- HMM fitted only on daily data up to train end date (line 1063: `mask = common_daily_idx <= end_date`)
- Overlay saved as `hmm_overlay_fold{fold_id}.npy` -- each fold gets its own

### 4.2 CPU-Parallel Path (line 1816-1838)
- Same pattern: `fit_hmm_on_window(_train_end_par)` per fold
- `_fold_hmm_overlays[wi]` stored per fold, passed to workers

### 4.3 Sequential Path (line 2124-2156)
- `fit_hmm_on_window(train_end_date)` called at the start of each fold loop iteration
- `_hmm_overlay` updated in-place per fold before train/test extraction

### 4.4 _gpu_fold_worker + _isolated_fold_worker
- Both load `hmm_overlay_fold{fold_id}.npy` specific to their fold
- No cross-fold HMM contamination

### 4.5 HMM Column Stripping (PASS)
All parallel/GPU paths strip full-history HMM columns from X_all before the fold loop (lines 1671-1678, 1802-1811, 1996-2017), replacing them with per-fold overlays. This eliminates the lookahead bias that would occur from using HMM fitted on the entire dataset.

---

## 5. IS METRICS ON TRAIN SET, NOT TEST (PASS)

Verified in all 5 paths:

| Path | IS prediction line | Data used |
|------|-------------------|-----------|
| _gpu_fold_worker | 905 | `model.predict(X_train, ...)` |
| _cpcv_split_worker | 602 | `model.predict(X_train, ...)` |
| _isolated_fold_worker | 718 | `model.predict(X_train, ...)` |
| Sequential | 2347 | `_predict_chunked(model, X_train, ...)` |
| Sequential subprocess | (delegates to _isolated_fold_worker) | same |

All IS metrics (`is_acc`, `is_mlogloss`, `is_sharpe`) are computed on the **full training fold** (`X_train`, `y_train`), not the test set or early-stopping validation split. OOS metrics (`acc`, `prec_long`, `prec_short`, `mlogloss`) are computed on `X_test`, `y_test`. **Correct separation.**

---

## 6. CHECKPOINT/RESUME (PASS)

### 6.1 Checkpoint Storage
All paths save checkpoints after each fold:
- `oos_predictions`, `window_results`, `completed_folds`, `best_acc` (lines 1753-1768, 1955-1972, 2389-2406)
- Atomic save via `atomic_save_pickle` with ImportError fallback to `pickle.dump`

### 6.2 Resume Logic (line 1516-1527)
- Loads checkpoint from `cpcv_checkpoint_{tf_name}.pkl`
- Restores `_completed_folds` set, `oos_predictions`, `window_results`, `best_acc`
- Each fold loop checks `if wi in _completed_folds: continue`

### 6.3 GPU-Parallel Resume (line 964-965)
- `pending` list filters out `completed_folds`: `if wi not in completed_folds`

### 6.4 Cleanup (line 2432-2435)
Checkpoint file is deleted after all folds complete.

**No resume correctness issues found.**

---

## 7. feature_pre_filter=False (PASS)

Verified in ALL Dataset constructor calls:

| Location | Line | Setting |
|----------|------|---------|
| _cpcv_split_worker | 542 | `'feature_pre_filter': False` |
| _isolated_fold_worker | 696 | `'feature_pre_filter': False` |
| _gpu_fold_worker | 843 | `'feature_pre_filter': False` |
| Parent Dataset (binary) | 2056 | `'feature_pre_filter': False` |
| Parent Dataset (construct) | 2092 | `'feature_pre_filter': False` |
| Sequential fold | 2303 | `'feature_pre_filter': False` |

**All paths enforce `feature_pre_filter=False`. Rare esoteric cross signals are protected.**

---

## 8. DISPATCH LOGIC (lines 1633-1654)

```
if GPU_SPARSE and n_gpus > 1  --> run_cpcv_gpu_parallel (multi-GPU subprocess isolation)
elif GPU_SPARSE               --> sequential CPCV (single GPU, _use_parallel_splits=False)
else (CPU):
    if NNZ > int32             --> sequential (can't pickle large CSR)
    if features > 1M + sparse  --> sequential (pickle IPC bottleneck)
    else                       --> parallel CPU (ProcessPoolExecutor)
```

This dispatch is correct:
- Multi-GPU gets subprocess isolation with CUDA_VISIBLE_DEVICES
- Single GPU avoids ProcessPoolExecutor (would fight over one GPU)
- CPU path degrades gracefully based on data size

---

## FINDINGS

### FINDING 1: Parent Dataset missing `min_data_in_bin` (LOW RISK)

**Location:** Line 2092 (sequential path, fallback parent Dataset construction)

The fallback `_parent_ds` construction at line 2092 uses:
```python
params={'feature_pre_filter': False, 'max_bin': _base_lgb_params.get('max_bin', 255)}
```

Missing `'min_data_in_bin': 1` which is present in all per-fold Dataset constructors (lines 542, 696, 843, 2303) and in the binary-load path (line 2056).

**Impact:** LightGBM defaults `min_data_in_bin=3`, which could discard rare esoteric signal bins during the parent EFB construction. Per-fold Datasets with `reference=_parent_ds` inherit these bin boundaries. In practice, this affects only the sequential non-binary path and only if rare features have 1-2 samples per bin.

**Recommendation:** Add `'min_data_in_bin': 1` to line 2092 for consistency:
```python
params={'feature_pre_filter': False, 'max_bin': _base_lgb_params.get('max_bin', 255), 'min_data_in_bin': 1},
```

### FINDING 2: CPU-parallel path does not log train_size (COSMETIC)

**Location:** Line 1938

```python
'train_size': 0, 'test_size': len(y_test),
```

The CPU-parallel worker `_cpcv_split_worker` does not return `train_size` in its result tuple, so the parent sets it to 0. This is cosmetic -- it affects logging/reporting only, not model quality. The GPU-parallel and sequential paths correctly report `train_size`.

---

## VERIFICATION MATRIX

| Check | GPU-Parallel | CPU-Parallel | Isolated Subprocess | Sequential | Sequential+Subprocess |
|-------|:---:|:---:|:---:|:---:|:---:|
| Purge = max_hold_bars/TF | PASS | PASS | PASS | PASS | PASS |
| Embargo = max(0.01, hold/n) | PASS | PASS | PASS | PASS | PASS |
| CUDA_VISIBLE_DEVICES before import | PASS | N/A | N/A | N/A | N/A |
| No mmap data leakage | PASS | PASS (shm) | PASS | N/A | PASS |
| HMM per-fold, train-end-date only | PASS | PASS | PASS | PASS | PASS |
| IS on train, OOS on test | PASS | PASS | PASS | PASS | PASS |
| Checkpoint/resume | PASS | PASS | PASS | PASS | PASS |
| feature_pre_filter=False | PASS | PASS | PASS | PASS | PASS |
| min_data_in_bin=1 | PASS | PASS | PASS | **MISS*** | PASS |

*Only in parent Dataset fallback construction (line 2092). Per-fold Datasets are correct.

---

## CONCLUSION

The CPCV implementation faithfully follows Lopez de Prado's methodology:
1. **Combinatorial paths** with configurable groups/test-groups per TF
2. **Purge** correctly removes train samples whose label windows overlap test groups (per-group, handles non-contiguous)
3. **Embargo** correctly computed as `max(1% of n, max_hold_bars/n)` and applied per test group boundary
4. **HMM re-fitted per fold** on train-end-date only -- no lookahead
5. **IS/OOS properly separated** in all 5 execution paths
6. **feature_pre_filter=False** enforced universally
7. **Checkpoint/resume** correct with atomic saves

One actionable fix: add `min_data_in_bin: 1` to the parent Dataset fallback at line 2092.
