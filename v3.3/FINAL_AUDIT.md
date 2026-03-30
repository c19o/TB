# FINAL AUDIT REPORT — v3.3 Feature Branches

**Date:** 2026-03-30
**Auditor:** Error-Checker Agent (READ-ONLY)
**Base branch:** `v3.3` (commit 218f4e9)

---

## 1. BRANCH INVENTORY

| # | Branch | Commits | Description |
|---|--------|---------|-------------|
| 1 | `ceo/backend-dev-22d6ed1e` | 2 | Numba CSC intersection + CUDA-Speed GPU kernels |
| 2 | `ceo/backend-dev-b161bc3a` | 1 | Parallel cross steps + adaptive RIGHT_CHUNK controller |
| 3 | `ceo/backend-dev-702e95d2` | 1 | Bitpacked POPCNT co-occurrence pre-filter |
| 4 | `ceo/backend-dev-106b1d97` | 1 | Multi-GPU Optuna trial parallelism |
| 5 | `ceo/backend-dev-e443f758` | 1 | Training-Enhance (CSC, extra_trees, GC, NUMA, wave 3 fixes) |

**Empty/stale branches (no commits):** afb08116, b3bc55dc, b94865a8, cf941cf8, da8e680c — skip.

---

## 2. CASCADE ANALYSIS

### 2.1 Modified Files Per Branch

| File | 22d6ed1e | b161bc3a | 702e95d2 | 106b1d97 | e443f758 |
|------|:--------:|:--------:|:--------:|:--------:|:--------:|
| `v2_cross_generator.py` | **X** | **X** | **X** | | |
| `numba_cross_kernels.py` | NEW | | | | |
| `bitpack_utils.py` | | | NEW | | |
| `gpu_histogram.cu` | X | | | | |
| `gpu_histogram_cusparse.py` | X | | | | |
| `histogram_cusparse.py` | X | | | | |
| `leaf_gradient_scatter.py` | X | | | | |
| `multi_gpu_optuna.py` | | | | NEW | |
| `run_optuna_local.py` | | | | **X** | **X** |
| `ml_multi_tf.py` | | | | | X |
| `cloud_run_tf.py` | | | | | X |

### 2.2 Conflict Files

**CONFLICT 1: `v2_cross_generator.py`** — 3 branches touch this file

| Branch | Sections Modified |
|--------|-------------------|
| 22d6ed1e | Imports (top), new `_numba_cross_chunk()` function, dispatch branch in `gpu_batch_cross()` |
| b161bc3a | New `AdaptiveChunkController` class, `_get_mem_*` helpers, refactored `generate_all_crosses()` to step descriptors + parallel dispatch, `gpu_batch_cross()` inner loop changed to adaptive while |
| 702e95d2 | Imports (top), new `_compute_cooccurrence_pairs()` function, replaced co-occurrence blocks in `_gpu_cross_chunk()` and `_cpu_cross_chunk()` |

**Overlap analysis:**
- 22d6ed1e and 702e95d2 both add imports at top of file — **textual conflict**, easy merge
- 22d6ed1e adds numba dispatch in `gpu_batch_cross()`, b161bc3a restructures `gpu_batch_cross()` inner loop — **structural conflict**, manual merge required
- 702e95d2 replaces co-occurrence block in `_gpu_cross_chunk()` and `_cpu_cross_chunk()`, 22d6ed1e adds `_numba_cross_chunk()` which has its own co-occurrence block — **no conflict** (different functions), but the new numba function should also use the bitpack path

**CONFLICT 2: `run_optuna_local.py`** — 2 branches touch this file

| Branch | Sections Modified |
|--------|-------------------|
| 106b1d97 | Imports, `build_phase1_objective()` (gpu_cfg param), `run_search_for_tf()` (GPU config, n_jobs, sampler), `main()` (n_jobs auto-detect) |
| e443f758 | Imports, `build_phase1_objective()` (CSR→CSC, extra_trees, gc.disable), `run_search_for_tf()` (threadpoolctl, force_row_wise), pruner changes |

**Overlap analysis:**
- Both modify `build_phase1_objective()` — **structural conflict**. 106b1d97 adds `gpu_cfg` parameter; e443f758 adds CSC conversion and extra_trees. Different sections of the function, resolvable.
- Both modify `run_search_for_tf()` — **structural conflict**. 106b1d97 overrides n_jobs; e443f758 adds threadpoolctl. Must merge carefully to preserve both.

---

## 3. MATRIX THESIS COMPLIANCE

| Check | 22d6ed1e | b161bc3a | 702e95d2 | 106b1d97 | e443f758 |
|-------|:--------:|:--------:|:--------:|:--------:|:--------:|
| `feature_pre_filter=False` | PASS | N/A | N/A | PASS | PASS |
| `feature_fraction >= 0.7` | PASS | N/A | N/A | PASS | PASS |
| No row subsampling | PASS | PASS | PASS | PASS | PASS |
| No feature filtering/dropping | PASS | PASS | PASS | PASS | PASS |
| int64 indptr preserved | PASS | PASS | PASS | PASS | PASS |
| Binary features (0/1) exact | PASS | PASS | PASS | PASS | PASS |
| NaN preserved (not→0) | PASS | PASS | PASS | PASS | PASS |
| Rare signals form leaf splits | PASS | PASS | PASS | PASS | PASS |

**MATRIX THESIS: ALL PASS across all branches.**

---

## 4. PERFORMANCE AUDIT

| Check | 22d6ed1e | b161bc3a | 702e95d2 | 106b1d97 | e443f758 |
|-------|:--------:|:--------:|:--------:|:--------:|:--------:|
| No O(n²) or worse | PASS | PASS | PASS | PASS | PASS |
| Memory-aware | PASS | PASS | PASS | PASS | PASS |
| Thread-safe | PASS | FAIL | PASS | PASS | PASS |
| No unnecessary CPU↔GPU | FAIL | PASS | PASS | PASS | PASS |
| Env var toggles | PASS | PASS | PASS | PASS | PASS |

---

## 5. BUGS FOUND — BY SEVERITY

### CRITICAL (must fix before merge)

#### BUG-C1: Warp Reduction Kernel Produces Wrong Histograms
- **Branch:** 22d6ed1e
- **File:** `gpu_histogram.cu` — `sparse_hist_build_warp_kernel`
- **Issue:** Warp shuffle (`__shfl_down_sync`) assumes all 32 lanes process the same CSR loop iteration. Each thread has a different number of nonzeros per row — threads exit the loop at different times. Dead lanes return undefined values via shuffle. `full_mask = 0xffffffff` is used even when threads have returned early.
- **Impact:** When `CUDA_WARP_REDUCE=1`, histogram gradient sums are silently wrong → training accuracy degrades unpredictably.
- **Mitigation:** Defaults OFF. Safe if never enabled. **Must fix before anyone sets CUDA_WARP_REDUCE=1.**

#### BUG-C2: `device_type='cuda'` Replaces `'cuda_sparse'` — GPU Silently Falls Back to CPU
- **Branch:** 106b1d97
- **File:** `multi_gpu_optuna.py` line 135, `run_optuna_local.py` line ~521
- **Issue:** Hardcoded `device_type = 'cuda'` instead of `'cuda_sparse'`. Standard CUDA silently falls back to CPU on sparse data. The entire multi-GPU feature does nothing for sparse CSR (the primary code path).
- **Impact:** Multi-GPU Optuna is broken. All trials run on CPU regardless of GPU assignment.
- **Fix:** Change to `'cuda_sparse'` or auto-detect based on fork availability.

### HIGH (should fix before merge)

#### BUG-H1: GPU Contention in Parallel Cross Mode
- **Branch:** b161bc3a
- **File:** `v2_cross_generator.py` — parallel step dispatch
- **Issue:** When `PARALLEL_CROSS_STEPS=1` and GPU=True, multiple threads call `gpu_batch_cross()` concurrently. No VRAM guard, no CUDA stream isolation. Will crash or corrupt.
- **Mitigation:** Defaults OFF. Documentation warns against GPU+parallel. **Should add runtime guard.**

#### BUG-H2: Single GPU Auto-Enables and Overrides n_jobs
- **Branch:** 106b1d97
- **File:** `multi_gpu_optuna.py` line 129
- **Issue:** `enabled = num_gpus >= 1` means a single RTX 3090 triggers multi-GPU mode, forcing `n_jobs=1` and `device_type='cuda'` (CPU fallback per BUG-C2). Breaks existing single-GPU workflow.
- **Fix:** `enabled = num_gpus >= 2` or make activation explicit.

### MODERATE

#### BUG-M1: Legacy COO Column Rebase Produces Negative Indices in Parallel Mode
- **Branch:** b161bc3a
- **File:** `v2_cross_generator.py` — `_collect_cross()`
- **Issue:** In parallel mode, `gpu_batch_cross` is called with `col_offset=0`, but collection uses the running `col_offset`. `_c_local = _c - col_offset` produces negative column indices.
- **Mitigation:** Only triggers if a parallel step returns legacy COO (rare with current NPZ/CSR paths).

#### BUG-M2: Fake Caching in cuSPARSE Path
- **Branch:** 22d6ed1e
- **File:** `gpu_histogram_cusparse.py`
- **Issue:** `.T.tocsr()` recomputed every call despite "cache" comment. `_DUAL_CSR` flag imported but never read.
- **Impact:** No correctness issue. Dead optimization code.

#### BUG-M3: CuPy Warp Reduce Kernel Has Same Structural Issue as BUG-C1
- **Branch:** 22d6ed1e
- **File:** `leaf_gradient_scatter.py` — `_WARP_REDUCE_SCATTER_F64`
- **Impact:** Dead code (only called from dead `build_histogram_batch()`). No runtime impact.

### LOW

#### BUG-L1: `build_histogram_batch()` is Dead Code
- **Branch:** 22d6ed1e — `leaf_gradient_scatter.py`
- Batch kernel launches then falls through to sequential loop. Never called from any path.

#### BUG-L2: Failed Parallel Steps Silently Dropped
- **Branch:** b161bc3a — `_mem_aware_submit()`
- Exception prints traceback but step is missing from results with no error.

#### BUG-L3: `_results_lock` Declared But Never Used
- **Branch:** b161bc3a — dead threading.Lock()

#### BUG-L4: `gpu_oom_handler` Catches Too Broadly
- **Branch:** 106b1d97 — catches any RuntimeError containing 'cuda'

#### BUG-L5: `_gpu_trial_map` Never Cleared Between Timeframes
- **Branch:** 106b1d97 — logging shows trials from all previous TFs

#### BUG-L6: `_pack_column()` Dead Code
- **Branch:** 702e95d2 — `bitpack_utils.py`, defined but never called

#### BUG-L7: `gc.disable()` Scope Too Wide in ml_multi_tf.py
- **Branch:** e443f758 — 660-line span without try/finally protection

#### BUG-L8: CSC Conversion Wasted for 15m (force_row_wise=True)
- **Branch:** e443f758 — `tocsc()` applied then nullified by row-wise flag

---

## 6. RECOMMENDED MERGE ORDER

### Order: e443f758 → 702e95d2 → 22d6ed1e → b161bc3a → 106b1d97

**Rationale:**

| Step | Branch | Why This Order | Conflict Resolution |
|------|--------|----------------|---------------------|
| 1 | **e443f758** (Training-Enhance) | Clean base. Touches `ml_multi_tf.py`, `run_optuna_local.py`, `cloud_run_tf.py`. No cross-gen changes. | No conflicts with v3.3 base. |
| 2 | **702e95d2** (Bitpack POPCNT) | Cleanest cross-gen branch. Adds `bitpack_utils.py` (new) and modifies co-occurrence blocks only. | Merge `v2_cross_generator.py` imports — trivial. |
| 3 | **22d6ed1e** (Numba + CUDA-Speed) | Adds numba path + GPU kernels. After bitpack so the new `_numba_cross_chunk()` can adopt the bitpack co-occurrence path. | Merge imports at top of `v2_cross_generator.py`. Update `_numba_cross_chunk()` to call `_compute_cooccurrence_pairs()` from step 2 instead of copy-pasted code. |
| 4 | **b161bc3a** (Parallel + Adaptive) | Most invasive cross-gen changes. Restructures `generate_all_crosses()` and `gpu_batch_cross()` inner loop. Must come after all other cross-gen changes are in place. | Manual merge of `gpu_batch_cross()` — integrate numba dispatch (step 3) into adaptive while loop. Fix BUG-H1 (add GPU guard). Fix BUG-M1 (COO rebase). |
| 5 | **106b1d97** (Multi-GPU Optuna) | Depends on e443f758's run_optuna_local.py changes being present. | Merge `run_optuna_local.py` — integrate GPU config with CSC/extra_trees/gc changes. **MUST fix BUG-C2** (`cuda` → `cuda_sparse`) and **BUG-H2** (`enabled = num_gpus >= 2`) before merge. |

---

## 7. PER-BRANCH SIGN-OFF

| Branch | Matrix | Perf | Errors | Verdict |
|--------|:------:|:----:|:------:|---------|
| **22d6ed1e** (Numba+CUDA) | PASS | FAIL (dead code) | FAIL (BUG-C1) | **CONDITIONAL PASS** — Numba path is clean and safe. CUDA-Speed kernels (commit 9d425fd) have broken warp reduce. Safe to merge if warp reduce stays OFF or is fixed. |
| **b161bc3a** (Parallel+Adaptive) | PASS | FAIL (BUG-H1) | FAIL (BUG-M1, L2, L3) | **CONDITIONAL PASS** — Sequential mode (default) works correctly. Parallel mode has GPU contention + COO rebase bugs. Safe if `PARALLEL_CROSS_STEPS` stays 0 or bugs are fixed. |
| **702e95d2** (Bitpack POPCNT) | PASS | PASS | PASS | **PASS** — Clean, correct, well-toggled. Safe to merge as-is. |
| **106b1d97** (Multi-GPU Optuna) | PASS | PASS | FAIL (BUG-C2, H2) | **FAIL** — `cuda_sparse` regression makes the feature non-functional. Single-GPU auto-enable breaks existing workflow. Must fix before merge. |
| **e443f758** (Training-Enhance) | PASS | PASS | PASS (minor) | **PASS** — No blocking issues. Minor gc scope and dead code items. Safe to merge as-is. |

---

## 8. FINAL RECOMMENDATION

### Merge NOW (no fixes needed):
- **e443f758** (Training-Enhance) ✓
- **702e95d2** (Bitpack POPCNT) ✓

### Merge AFTER fixes:
- **22d6ed1e** — Fix or remove warp reduce kernel (BUG-C1). Numba path is clean.
- **b161bc3a** — Add GPU guard for parallel mode (BUG-H1). Fix COO rebase (BUG-M1).
- **106b1d97** — Fix `device_type='cuda_sparse'` (BUG-C2). Fix `enabled = num_gpus >= 2` (BUG-H2).

### Total bugs found: 16
- Critical: 2 (C1, C2)
- High: 2 (H1, H2)
- Moderate: 3 (M1, M2, M3)
- Low: 8 (L1-L8)
- Dead code instances: 5

### Matrix Thesis: FULLY PROTECTED across all branches. No violations found.
