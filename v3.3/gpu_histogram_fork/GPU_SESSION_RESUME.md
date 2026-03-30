# V3.3 GPU Histogram Fork — Session Resume

## INSTRUCTION TO NEW SESSION
Read this file completely. It contains the full state of training, cloud machines, and what needs doing next.

## ACTIVE MACHINES (2026-03-30 ~01:00 UTC)

**NO ACTIVE MACHINES**

**DESTROYED**: Norway (m:33781954), NJ (m:33791577), Belgium (m:33799816), Poland (m:33802014)

- Belgium destroyed 2026-03-30 01:00 UTC — 4h Optuna CRASHED: `set_external_csr` not found (cuda_sparse .so not loaded). CPU-only trials completed but GPU trials all failed. Artifacts saved to cloud_results_4h/
- Poland destroyed 2026-03-30 01:00 UTC — Running bench_trial.py for 24+ hrs instead of Optuna. 4.1GB swap thrashing. GPU stalled at 0%. Artifacts saved to cloud_results_1d/

**Total cost: $0/hr**

## TRAINING STATUS

| TF | Status | Machine | Accuracy | Notes |
|----|--------|---------|----------|-------|
| 1w | **DONE v5** | destroyed | 57.9% CPCV, 75.4% LONG@0.70, 67% SHORT | All fixes applied. min_data=8, ff=0.9. |
| 1d | **FAILED** | Poland (destroyed) | — | bench_trial.py ran 24hrs, swap thrashing. Never started Optuna. |
| 4h | **FAILED** | Belgium (destroyed) | — | cuda_sparse .so not loaded. GPU trials crashed. CPU trials ran but results invalid (pre-bug-fix). |
| 1h | NOT STARTED | — | — | Need machine + cross gen |
| 15m | NOT STARTED | — | — | Need machine + cross gen |

## CRITICAL BUGS FOUND & FIXED THIS SESSION

### 1. feature_fraction killed rare esoteric signals (BIGGEST)
- OLD Optuna range: 0.005-0.05 = each tree saw only 1-5% of EFB bundles
- With EFB bundling 5.56M→23K, feature_fraction=0.01 = 150 bundles per tree, ~8 per node
- Rare cross features (moon × RSI, gematria × volume) systematically excluded
- Perplexity confirmed: must be >= 0.7 for sparse binary features
- **FIX**: feature_fraction range [0.7, 1.0], V3_LGBM_PARAMS default 0.9
- Previous 1d (0.753) and 4h (0.657) results INVALID

### 2. min_data_in_leaf killed rare signals (SAME CLASS)
- OLD: 1w=30, 1d=50, 4h=20 — rare signals fire 10-20x, can't make leaf with 50 min
- **FIX**: 1w=8, 1d=10, 4h=10, 1h=10, 15m=10

### 3. Standard LightGBM CUDA silently falls back to CPU on sparse
- `device='cuda'` on sparse CSR → warning "sparse features not supported" → runs on CPU
- Must use custom fork with `device_type='cuda_sparse'` + `set_external_csr()`
- `lgb.train()` CANNOT use cuda_sparse — need manual Booster + update() loop

### 4. set_external_csr() is NOT thread-safe
- Cannot run parallel Optuna trials in threads with cuda_sparse
- Must use 1 process per GPU (separate processes, shared Optuna DB)
- Perplexity confirmed: fork() with LightGBM GPU = deadlock risk, use spawn

### 5. cmake 3.30 injects /utf-8 MSVC flags into CUDA builds
- Causes gcc "cannot specify -o with multiple files" error
- Cannot fix via CMAKE_CUDA_FLAGS (cmake appends AFTER override)
- Cannot fix via sed (make re-invokes cmake which regenerates)
- **FIX**: gcc/g++ wrapper scripts that strip /utf-8 from argv
- Also must strip /FORCE:MULTIPLE from linker flags (MSVC leak in CMakeLists)

### 6. min_data_in_bin mismatch on .bin reload
- Dataset .bin caches params at build time (min_data_in_bin, max_bin)
- Loading .bin without matching params → crash
- **FIX**: pass params={min_data_in_bin:1, max_bin:255} to ALL lgb.Dataset() calls including .bin loads
- Always `rm lgbm_dataset_*.bin` when config changes

### 7. Parallel Dataset construction fork-bombs without __main__ guard
- _parallel_dataset_construct() uses ProcessPoolExecutor(spawn)
- Standalone scripts without `if __name__ == '__main__'` trigger recursive spawning
- run_optuna_local.py has the guard; bench_trial.py did not
- **FIX**: always use __main__ guard, or call from run_optuna_local.py directly

## VALIDATION SYSTEM (NEW)
- `validate.py` — 74 deterministic pre-flight checks. Runs before ALL training.
- `runtime_checks.py` — 3-layer runtime monitoring:
  1. `preflight_training()` — memory, sparse integrity, labels, GPU VRAM budget, all-GPU smoke test
  2. `TrainingMonitor` — daemon thread: CPU%, RSS, swap, GPU utilization, multi-GPU active count
  3. `post_trial_check()` — NaN detection, timing anomalies, esoteric features in top-100
- validate.py is the SINGLE SOURCE OF TRUTH for all parameter constraints

## GPU DEPLOYMENT NOTES

### Belgium (A40, CUDA 12.6) — WORKING
- cuda_sparse fork built with gcc wrapper (strips /utf-8)
- NCCL: `apt-get install libnccl2 libnccl-dev` → /usr/include/nccl.h
- Build: see memory/project_gpu_fork_deploy_fixes.md for full cmake command
- .so at: `/workspace/v3.3/gpu_histogram_fork/_build/LightGBM/lib_lightgbm.so`
- Swapped into: `$(python -c "import lightgbm; ...")/lib/lib_lightgbm.so`
- CPU backup: `.cpu_backup` alongside

### Poland (RTX 5090, CUDA 12.9) — WORKING
- Fork .so copied from Belgium (sm_89 included in build, works on sm_120 via PTX)
- Must set LD_LIBRARY_PATH for NCCL + CUDA runtime:
  ```
  export LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cusparse/lib:/opt/conda/lib/python3.11/site-packages/nvidia/nvjitlink/lib:/opt/conda/lib/python3.11/site-packages/nvidia/nccl/lib:/usr/lib/x86_64-linux-gnu
  ```
- Also needs: `apt-get install libgomp1`

## 1w RESULTS (FINAL — v5 with all fixes)

| Confidence | Accuracy | Samples | % of Total |
|-----------|----------|---------|-----------|
| >= 0.50 | 59.3% | 4,181 | 91% |
| >= 0.60 | 64.9% | 2,604 | 57% |
| >= 0.65 | 68.2% | 1,593 | 35% |
| >= 0.70 | **75.4%** | 994 | 22% |
| >= 0.75 | 73.8% | 600 | 13% |

LONG@0.70: 75.4% (994 samples). SHORT@0.60: 62.9% (35 samples).
Best params: num_leaves=7, min_data_in_leaf=8, feature_fraction=0.80, lambda_l2=12.6

## ARTIFACTS BACKED UP LOCALLY

All critical artifacts downloaded to local machine:
- `v3.3/cloud_results_1d/` — 1.4GB (12 files: NPZ, parquets, inference JSONs, cross names)
- `v3.3/cloud_results_4h/` — 3.8GB (15 files: NPZ, parquets, inference JSONs, cross names, ml_config)

## NEXT STEPS
1. **Wait for 1d + 4h training** to finish (or at least first trial results)
2. **Verify GPU actually fires** on Belgium 4h — check for `[CUDASparseHist]` in logs
3. If GPU works: scale to 4 processes per machine (1 per GPU, shared Optuna DB)
4. If GPU doesn't work: stay CPU, use parallel Dataset construction for speedup
5. **1h**: rent machine after 4h results. Need 768GB+ RAM for cross gen.
6. **15m**: after 1h. Need 1TB+ RAM.
7. **Update deploy_vastai.sh** with gcc wrapper fix + all deployment lessons

## KEY LESSONS THIS SESSION
- **feature_fraction is the #1 silent killer** — Optuna optimizes by killing the matrix edge
- **validate.py catches bugs BEFORE spending money** — added 74 checks
- **runtime_checks.py catches bugs DURING training** — GPU idle detection, memory leaks
- **Standard CUDA silently falls back to CPU on sparse** — only cuda_sparse fork works
- **Multi-GPU requires separate processes** — not Optuna n_jobs threading
- **cmake 3.30 is broken for CUDA** — gcc wrapper is the reliable fix
- **Always test the REAL pipeline** — smoke tests with tiny data miss real bottlenecks
- **Plan before launching cloud** — we burned hours on 0% GPU utilization
