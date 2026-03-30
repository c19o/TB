# QA Cascade & Integration Audit — All CEO Branches

**Date:** 2026-03-29
**Auditor:** checklist-agent (READ-ONLY)
**Base:** v3.3

---

## 1. Branch Inventory

| Branch | Codename | Files Modified | Status |
|--------|----------|---------------|--------|
| `ceo/backend-dev-22d6ed1e` | **Numba-CrossGen** | 7 files (numba_cross_kernels.py, v2_cross_generator.py, GPU fork files, CHANGES) | ACTIVE |
| `ceo/backend-dev-702e95d2` | **Bitpack-CoOccurrence** | 3 files (bitpack_utils.py, v2_cross_generator.py, CHANGES) | ACTIVE |
| `ceo/backend-dev-b161bc3a` | **Parallel-Adaptive** | 2 files (v2_cross_generator.py, CHANGES) | ACTIVE |
| `ceo/backend-dev-106b1d97` | **MultiGPU-Optuna** | 3 files (multi_gpu_optuna.py, run_optuna_local.py, CHANGES) | ACTIVE |
| `ceo/backend-dev-e443f758` | **Training-Enhance** | 4 files (run_optuna_local.py, ml_multi_tf.py, cloud_run_tf.py, CHANGES) | ACTIVE |
| `ceo/backend-dev-afb08116` | (empty) | 0 files | SKIP |
| `ceo/backend-dev-b3bc55dc` | (empty) | 0 files | SKIP |
| `ceo/backend-dev-b94865a8` | (empty) | 0 files | SKIP |

---

## 2. Conflict Files

### `v3.3/v2_cross_generator.py` — 3 branches touch this file

| Branch | Section Modified | Lines (approx) |
|--------|-----------------|----------------|
| **Numba-CrossGen** | Import block (~L70), `gpu_batch_cross` elif (~L695), new `_numba_cross_chunk` function (~L988+) | Adds 80+ lines |
| **Bitpack-CoOccurrence** | Import block (~L39), new `_compute_cooccurrence_pairs` function (~L226), refactors `_cpu_cross_chunk` + `_gpu_cross_chunk` to call it | Adds 70+ lines |
| **Parallel-Adaptive** | Modifies RIGHT_CHUNK comment (~L213), adds `AdaptiveChunkController` class + helpers (~L220-330), **rewrites `gpu_batch_cross` loop** (~L669-780) | Major rewrite |

**Conflict severity:**
- **Numba + Bitpack:** LOW — different import locations, different new functions. Numba adds at L70 and L988; Bitpack at L39 and L226. Auto-merge likely succeeds.
- **Numba + Parallel:** MEDIUM — both modify `gpu_batch_cross`. Numba adds an `elif _USE_NUMBA_CROSS` branch inside the loop. Parallel rewrites the entire loop (static → adaptive chunks). Manual resolution needed to insert Numba's elif into Parallel's rewritten loop.
- **Bitpack + Parallel:** MEDIUM — Bitpack adds `_compute_cooccurrence_pairs` near L226; Parallel adds `AdaptiveChunkController` in the same region (~L220+). Adjacent insertions will likely conflict on context lines. Bitpack also changes `_cpu_cross_chunk` internals which Parallel leaves alone (safe).

### `v3.3/run_optuna_local.py` — 2 branches touch this file

| Branch | Section Modified |
|--------|-----------------|
| **MultiGPU-Optuna** | Adds `from multi_gpu_optuna import ...` at imports; adds `gpu_cfg` param to `build_phase1_objective`; replaces GPU params block in objective |
| **Training-Enhance** | Adds threadpoolctl at top; adds WilcoxonPruner import; adds `extra_trees` param; adds `TF_FORCE_ROW_WISE` block; CSR→CSC in `load_tf_data` |

**Conflict severity:** MEDIUM-HIGH
Both modify `build_phase1_objective`:
- Training-Enhance adds `extra_trees` param at ~L486 and `force_row_wise` logic at ~L519 (just ABOVE the GPU params block)
- MultiGPU replaces the GPU params block at ~L505-520

These are adjacent/overlapping hunks. Git will NOT auto-merge. Manual resolution required: keep both `extra_trees` + `force_row_wise` AND the new `gpu_cfg` logic.

---

## 3. Config.py

**NO branches modify config.py.** No env var conflicts.

Training-Enhance references `TF_FORCE_ROW_WISE` imported from config — this must already exist in v3.3 base. MultiGPU reads `LGBM_NUM_GPUS` env var. No naming collisions.

---

## 4. New Module Import Chains

| Module | Imports From | Imported By |
|--------|-------------|-------------|
| `numba_cross_kernels.py` | numpy, numba, llvmlite (stdlib only) | v2_cross_generator.py (conditional) |
| `bitpack_utils.py` | numpy, numba, llvmlite (stdlib only) | v2_cross_generator.py (conditional) |
| `multi_gpu_optuna.py` | os, logging, threading (stdlib only) | run_optuna_local.py |

**No circular imports.** All new modules are leaf dependencies importing only from stdlib/third-party.

---

## 5. Duplicate Function Names

**`_llvm_ctpop_i64`** — exists in BOTH `numba_cross_kernels.py` AND `bitpack_utils.py`

- numba_cross_kernels: `types.int64(types.int64)` signature, uses `declare_intrinsic('llvm.ctpop', [ir.IntType(64)], fn_type)`
- bitpack_utils: `types.int64(types.uint64)` signature, uses `declare_intrinsic('llvm.ctpop.i64', fnty=fn_type)`

**Different signatures AND different intrinsic calls.** No runtime conflict (separate modules), but should be deduplicated into a shared utils module to avoid divergence. The `types.uint64` vs `types.int64` input type difference could cause subtle bugs if one is wrong.

---

## 6. Recommended Merge Order

### Phase 1: Independent merges (no cross-conflicts)
1. **Training-Enhance** (`e443f758`) → merge to v3.3 first
   - Touches `run_optuna_local.py`, `ml_multi_tf.py`, `cloud_run_tf.py`
   - No overlap with cross-gen branches

2. **MultiGPU-Optuna** (`106b1d97`) → merge second
   - Conflict with Training-Enhance on `run_optuna_local.py` (resolve manually)
   - Keep both: Training-Enhance's extra_trees/force_row_wise + MultiGPU's gpu_cfg

### Phase 2: Cross-generator chain (order matters)
3. **Bitpack-CoOccurrence** (`702e95d2`) → merge third
   - Adds `_compute_cooccurrence_pairs` helper + `bitpack_utils.py`
   - Clean addition, minimal conflict surface

4. **Numba-CrossGen** (`22d6ed1e`) → merge fourth
   - Adds `numba_cross_kernels.py` + elif branch in `gpu_batch_cross`
   - Minor conflict with Bitpack in imports area (auto-resolvable)

5. **Parallel-Adaptive** (`b161bc3a`) → merge LAST
   - **Rewrites the `gpu_batch_cross` loop entirely**
   - Must manually re-insert Numba's `elif _USE_NUMBA_CROSS` branch into the rewritten loop
   - Must verify Bitpack's `_compute_cooccurrence_pairs` still called correctly after loop rewrite

### Phase 3: Empty branches
6. Delete empty branches: `afb08116`, `b3bc55dc`, `b94865a8`

---

## 7. Expected Conflicts & Resolution Strategy

| Merge Step | File | Conflict Type | Resolution |
|-----------|------|--------------|------------|
| MultiGPU after Training-Enhance | `run_optuna_local.py` | Adjacent hunks in `build_phase1_objective` | Keep both: extra_trees+force_row_wise above gpu_cfg block |
| Numba after Bitpack | `v2_cross_generator.py` imports | Adjacent import blocks | Accept both import blocks |
| Parallel after Numba+Bitpack | `v2_cross_generator.py:gpu_batch_cross` | Full loop rewrite vs elif insertion | Manually add Numba elif + Bitpack call into Parallel's adaptive loop |
| Parallel after Bitpack | `v2_cross_generator.py` ~L220 area | Adjacent insertions | Ensure AdaptiveChunkController and _compute_cooccurrence_pairs don't interleave |

---

## 8. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Parallel branch rewrites gpu_batch_cross loop | **HIGH** | Merge last, test all code paths (GPU, Numba, CPU, Bitpack) |
| Duplicate `_llvm_ctpop_i64` with different signatures | **LOW** | Deduplicate post-merge into shared numba_intrinsics.py |
| MultiGPU + Training-Enhance both modify objective | **MEDIUM** | Careful manual merge, test with/without multi-GPU |
| Training-Enhance adds CSR→CSC conversion | **LOW** | Verify all downstream code handles CSC (LightGBM does natively) |

---

## Summary

- **5 active branches**, 3 empty (delete)
- **2 conflict files**: `v2_cross_generator.py` (3 branches), `run_optuna_local.py` (2 branches)
- **No config.py conflicts**, no circular imports
- **1 duplicate function** (`_llvm_ctpop_i64`) — different implementations, should deduplicate
- **Merge order is critical**: Training-Enhance → MultiGPU → Bitpack → Numba → Parallel (last)
- **Parallel-Adaptive must merge last** — it rewrites the main loop that 2 other branches modify
