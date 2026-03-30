# QA Training Audit — Matrix Thesis Compliance
**Date**: 2026-03-29
**Auditor**: Error Checker Agent
**Base branch**: v3.3

## Branches Audited
1. `ceo/backend-dev-e443f758` — **Training-Enhance**: CSC, WilcoxonPruner, extra_trees, GC, NUMA, wave3 recovery
2. `ceo/backend-dev-106b1d97` — **MultiGPU Optuna**: multi_gpu_optuna.py
3. `ceo/backend-dev-22d6ed1e` — **CUDA-Speed**: 4 kernel optimizations + Numba cross kernels

---

## Check 1: `feature_pre_filter=False` on ALL `lgb.Dataset` calls

| Branch | Result | Details |
|--------|--------|---------|
| Training-Enhance | **PASS** | All Dataset calls in `run_optuna_local.py` (lines 1050, 1065, 1095, 1215) and `ml_multi_tf.py` (lines 513, 1491, 1527, 1643, 1941) have `feature_pre_filter: False`. No new Dataset calls added. |
| MultiGPU Optuna | **PASS** | No new `lgb.Dataset` calls introduced. Existing calls unchanged (verified lines 1038, 1050, 1065, 1084, 1206). `multi_gpu_optuna.py` does not create Datasets. |
| CUDA-Speed | **PASS** | No `lgb.Dataset` calls in CUDA kernels or `numba_cross_kernels.py`. Cross gen does not touch LightGBM Datasets. |

## Check 2: `feature_fraction >= 0.7` (not lowered)

| Branch | Result | Details |
|--------|--------|---------|
| Training-Enhance | **PASS** | `feature_fraction = trial.suggest_float('feature_fraction', 0.7, 1.0)` — lower bound is 0.7 (line 496). No change from base. |
| MultiGPU Optuna | **PASS** | Same range `(0.7, 1.0)` preserved (line 486). `apply_gpu_params()` does not touch feature_fraction. |
| CUDA-Speed | **PASS** | No feature_fraction parameters in CUDA kernels or Numba cross gen. |

## Check 3: No row subsampling introduced (subsample=1.0 or bagging_fraction >= 0.95)

| Branch | Result | Details |
|--------|--------|---------|
| Training-Enhance | **PASS** | `bagging_fraction` range is `(0.5, 1.0)` — unchanged from base. `OPTUNA_TF_ROW_SUBSAMPLE` unchanged in config.py (1h=0.50, 15m=0.25 — pre-existing, not introduced by this branch). |
| MultiGPU Optuna | **PASS** | Same `bagging_fraction` range `(0.5, 1.0)`. No new row subsampling. Each trial sees ALL rows per CHANGES doc. |
| CUDA-Speed | **PASS** | No training params modified. Cross gen processes all rows. |

> **NOTE**: `bagging_fraction` lower bound 0.5 and `OPTUNA_TF_ROW_SUBSAMPLE` of 0.50/0.25 for 1h/15m are **pre-existing** in the base v3.3 branch. None of the audited branches introduced or lowered these values. If this is a concern, it should be addressed in a separate fix on the base branch.

## Check 4: No feature filtering in Optuna objective

| Branch | Result | Details |
|--------|--------|---------|
| Training-Enhance | **PASS** | No feature selection/filtering logic added. `extra_trees` is a LightGBM splitting parameter, not a filter. All features passed through. |
| MultiGPU Optuna | **PASS** | `apply_gpu_params()` only sets device_type, gpu_device_id, histogram_pool_size, num_threads. No feature filtering. |
| CUDA-Speed | **PASS** | No Optuna objective changes. Numba cross gen preserves all cross features. |

## Check 5: WilcoxonPruner: p_threshold conservative (>=0.1), n_startup_steps >= 2

| Branch | Result | Details |
|--------|--------|---------|
| Training-Enhance | **PASS** | `WilcoxonPruner(p_threshold=0.1, n_startup_steps=2)` — exactly at thresholds. Reports fold-level mlogloss (not round-level). Fallback to MedianPruner if WilcoxonPruner unavailable. |
| MultiGPU Optuna | **N/A** | Does not modify pruner logic. Uses existing pruner from base. |
| CUDA-Speed | **N/A** | Does not modify pruner logic. |

## Check 6: GPU histograms EXACT (no approximations in CUDA code)

| Branch | Result | Details |
|--------|--------|---------|
| Training-Enhance | **N/A** | No CUDA code changes. |
| MultiGPU Optuna | **N/A** | No CUDA code changes. Uses standard LightGBM CUDA device. |
| CUDA-Speed | **PASS** | All 4 optimizations produce exact results: (A) Batched H2D — same atomicAdd logic, just concatenated rows. (B) Warp-cooperative — `__shfl_down_sync` reduces matching keys before atomicAdd, mathematically identical (associative float64 addition in same order). (C) Vectorized launches — same kernel math, fewer Python calls. (D) Dual CSR — pre-computed `.T.tocsr()` instead of lazy, identical SpMV result. CHANGES doc explicitly states "EXACT same results". |

## Check 7: EFB bundle offsets preserved in CUDA changes

| Branch | Result | Details |
|--------|--------|---------|
| Training-Enhance | **N/A** | No CUDA code changes. `max_bin=255` LOCKED in Optuna params. |
| MultiGPU Optuna | **N/A** | No CUDA code changes. |
| CUDA-Speed | **PASS** | All kernels handle EFB mode correctly: `sparse_hist_build_warp_kernel` checks `if (bin == 0) continue` and uses `data[j]` for EFB bin values (lines 383-397). `sparse_hist_build_batched_kernel` same pattern (lines 449-456). Raw binary mode (data==nullptr) preserved. Bundle offsets stored in `indices[j]` (column index) — unchanged. |

## Check 8: Multi-GPU: each trial sees ALL features and ALL rows

| Branch | Result | Details |
|--------|--------|---------|
| Training-Enhance | **N/A** | No multi-GPU changes. |
| MultiGPU Optuna | **PASS** | Architecture is trial-level parallelism (NOT data-parallel). Each trial gets the full X_all and y arrays. `apply_gpu_params()` only sets `device_type`, `gpu_device_id`, `histogram_pool_size`, `num_threads` — no data partitioning. CHANGES doc: "Each trial sees ALL features and ALL rows — no partitioning". Round-robin `gpu_id = trial.number % num_gpus` assigns device only. |
| CUDA-Speed | **N/A** | Not multi-GPU related. |

## Check 9: GC re-enabled after training loops (no permanent disable)

| Branch | Result | Details |
|--------|--------|---------|
| Training-Enhance | **PASS** | `run_optuna_local.py`: `gc.disable()` before `study.optimize()`, `gc.enable(); gc.collect()` in `finally` block (lines 1333-1340). `ml_multi_tf.py`: `gc.disable()` before CPCV (line 1220), `gc.enable(); gc.collect()` after feature importance (lines 1880-1882). Both have proper re-enable. |
| MultiGPU Optuna | **PASS** | No GC changes. |
| CUDA-Speed | **PASS** | No GC changes. `_free_gpu_memory()` in OOM handler calls `gc.collect()` — safe cleanup, not permanent disable. |

## Check 10: NUMA binding doesn't restrict available cores

| Branch | Result | Details |
|--------|--------|---------|
| Training-Enhance | **FAIL** | `numactl --cpunodebind=0 --membind=0` binds training to **node 0 only**. On a 2-socket machine with 64 cores per socket, this restricts training to 50% of available cores. While memory locality improves, the core restriction may be significant. Single-node systems are unaffected (no binding applied). |
| MultiGPU Optuna | **N/A** | No NUMA changes. |
| CUDA-Speed | **N/A** | No NUMA changes. |

---

## Summary

| Check | Training-Enhance | MultiGPU Optuna | CUDA-Speed |
|-------|:---:|:---:|:---:|
| 1. feature_pre_filter=False | PASS | PASS | PASS |
| 2. feature_fraction >= 0.7 | PASS | PASS | PASS |
| 3. No row subsampling introduced | PASS | PASS | PASS |
| 4. No feature filtering in objective | PASS | PASS | PASS |
| 5. WilcoxonPruner thresholds | PASS | N/A | N/A |
| 6. GPU histograms EXACT | N/A | N/A | PASS |
| 7. EFB bundle offsets preserved | N/A | N/A | PASS |
| 8. Multi-GPU: ALL features + rows | N/A | PASS | N/A |
| 9. GC re-enabled after loops | PASS | PASS | PASS |
| 10. NUMA doesn't restrict cores | **FAIL** | N/A | N/A |

### Failures

**Training-Enhance Check 10 — NUMA binding restricts to node 0**:
- `cloud_run_tf.py` lines 560-571: `numactl --cpunodebind=0 --membind=0` on multi-node systems
- This restricts training to only node 0's cores (e.g., 64 of 128 on 2-socket)
- **Fix options**: (a) Use `numactl --interleave=all` for memory without CPU restriction, (b) bind to ALL nodes, (c) use `--preferred=0` instead of `--membind=0` for soft preference
- **Severity**: Medium — only affects multi-socket cloud machines, single-socket (most vast.ai) unaffected

### Warnings (pre-existing, not introduced by audited branches)

1. **bagging_fraction lower bound 0.5** — Optuna can sample `bagging_fraction=0.5`, dropping 50% of rows per boosting iteration. This is in the base v3.3 config, not introduced by any audited branch.
2. **OPTUNA_TF_ROW_SUBSAMPLE** for 1h (0.50) and 15m (0.25) — pre-existing in config.py. Reduces training data significantly for speed. Also base v3.3.
