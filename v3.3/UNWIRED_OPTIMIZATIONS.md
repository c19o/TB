# UNWIRED OPTIMIZATIONS REPORT
**Date:** 2026-03-30
**Branch:** v3.3
**Method:** Full scan of v3.3 + all 21 ceo/* branches + audit reports

---

## EXECUTIVE SUMMARY

| Priority | Count | Description |
|----------|-------|-------------|
| CRITICAL | 3 | Bugs that corrupt training results if deployed as-is |
| HIGH | 7 | Coded optimizations that exist on branches but are not merged |
| MEDIUM | 5 | Designed modules that exist but aren't connected to the pipeline |
| LOW | 3 | Minor optimizations or future-planned features |

**18 total unwired items.** The CEO branches contain ~2,500 lines of working code that has never been merged to v3.3.

---

## CRITICAL — Fix Before Any Training

### C1. Sortino Denominator Bug (CORRECTNESS)
- **File:** `v3.3/exhaustive_optimizer.py:686-687`
- **Bug:** `sum_neg_sq / total_trades` should be `sum_neg_sq / count_neg`
- **Impact:** Sortino inflated up to 5x when most trades are profitable. Optimizer picks wrong configs.
- **Fix exists on:** `ceo/backend-dev-cf941cf8` (line 695-696 uses `count_neg`)
- **Wire:** Merge the Sortino fix from cf941cf8

### C2. Optuna feature_fraction_bynode Floor = 0.5 (Should Be 0.7)
- **File:** `v3.3/run_optuna_local.py:483`
- **Bug:** `trial.suggest_float('feature_fraction_bynode', 0.5, 1.0)` allows 0.5
- **Impact:** Kills rare esoteric cross signals at the per-node level
- **Fix exists on:** `ceo/backend-dev-8dedf3ff` (changes floor to 0.7)
- **Wire:** Merge param floor fix from 8dedf3ff

### C3. Optuna bagging_fraction Floor = 0.5 (Should Be 0.7)
- **File:** `v3.3/run_optuna_local.py:484`
- **Bug:** `trial.suggest_float('bagging_fraction', 0.5, 1.0)` allows 0.5
- **Impact:** At 0.5, P(10-fire signal in bag) = 0.1%. Destroys rare signal detection.
- **Fix exists on:** `ceo/backend-dev-8dedf3ff` (changes floor to 0.7)
- **Wire:** Merge param floor fix from 8dedf3ff

---

## HIGH — Coded on Branches, Not Merged to v3.3

### H1. Sobol Trade Optimizer (Game-Changer #3)
- **File:** `ceo/backend-dev-cf941cf8:v3.3/exhaustive_optimizer.py`
- **What:** Full Sobol quasi-random sweep + Bayesian refinement (scipy.stats.qmc.Sobol). Two-phase optimizer with GPU-batched evaluation.
- **v3.3 status:** Uses TPE sampler only (no Sobol). `OPTIMIZER_MODE` env var doesn't exist.
- **Wire:** Merge exhaustive_optimizer.py from cf941cf8. Contains Sobol + fixed Sortino.

### H2. lleaves Compiler (Game-Changer #4)
- **File:** `ceo/backend-dev-cf941cf8:v3.3/lleaves_compiler.py` (189 lines)
- **What:** Compiles LightGBM model to native LLVM code via lleaves for 5.4x inference speedup.
- **v3.3 status:** File does NOT exist on v3.3. Not imported by any pipeline file.
- **Wire:** Merge file, add auto-compile call after model save in `cloud_run_tf.py` and `ml_multi_tf.py`. Wire compiled model loading in `live_trader.py`.

### H3. Inference Pruner (Game-Changer #4 companion)
- **File:** `ceo/backend-dev-cf941cf8:v3.3/inference_pruner.py` (260 lines)
- **What:** Extracts split-count > 0 features from trained model, creates pruned deployment model.
- **v3.3 status:** File does NOT exist on v3.3. Not imported anywhere.
- **Wire:** Merge file, call after training in `cloud_run_tf.py` (step 7). Wire into `live_trader.py` model loading.

### H4. Memmap Streaming Merge (Optimization #1E)
- **File:** `ceo/backend-dev-cf941cf8:v3.3/memmap_merge.py` (236 lines)
- **What:** Two-pass streaming CSC merge via memory-mapped files. Reduces 1h RAM from 1.8TB to 5-10GB, 15m from 3TB+ to 10-15GB.
- **v3.3 status:** File does NOT exist on v3.3. Cross gen still uses in-memory concatenation.
- **Wire:** Merge file, call from `v2_cross_generator.py` final merge when TF is 1h/15m (controlled by `MEMMAP_CROSS_GEN` env var).

### H5. Multi-GPU Optuna Trial Parallelism
- **File:** `ceo/backend-dev-106b1d97:v3.3/multi_gpu_optuna.py` (278 lines)
- **What:** Detects GPUs, assigns each Optuna trial to a separate GPU via round-robin. 4 GPUs = 4 concurrent trials.
- **v3.3 status:** File does NOT exist on v3.3. `run_optuna_local.py` has no multi-GPU awareness.
- **Wire:** Merge file. Branch 106b1d97 already has `run_optuna_local.py` wired with imports (line 87-89) and `apply_gpu_params()` calls.
- **Note:** The branch also has 4 bug fixes for device_type, threshold, OOM handler, and trial map.

### H6. EFB Pre-Bundler (Game-Changer #1)
- **File:** `ceo/backend-dev-605acb8a:v3.3/efb_prebundler.py` (460 lines)
- **What:** External pre-bundling of binary cross features. 127 features per bundle, 128x histogram reduction (10M features -> 79K bundles).
- **v3.3 status:** File does NOT exist on v3.3. Double-audit confirmed "NOT IMPLEMENTED".
- **Wire on branch:** Branch 605acb8a DOES wire it into `v2_cross_generator.py:1999-2000` via `prebundle_from_files()`. Still needs merge.
- **Training wire:** Must also call from `ml_multi_tf.py` / `run_optuna_local.py` to pass pre-bundled matrix to LightGBM with `enable_bundle=False`.

### H7. Bitpacked POPCNT Co-occurrence Pre-Filter (Optimization #1D)
- **File:** `ceo/backend-dev-702e95d2:v3.3/bitpack_utils.py` (107 lines)
- **What:** Hardware POPCNT via LLVM intrinsic for 8-21ms co-occurrence counting (replaces sparse matmul).
- **v3.3 status:** File does NOT exist on v3.3.
- **Wire on branch:** Branch 702e95d2 DOES wire it into `v2_cross_generator.py:45,262` with graceful fallback. Needs merge.

---

## MEDIUM — Config/Feature Gaps on v3.3

### M1. .npy Memmap Loading (Optimization #7)
- **File:** Design only in `EXPERT_NVME_IO.md:62-69`
- **What:** `save_csr_npy` / `load_csr_npy` with `mmap_mode='r'` for zero-copy sparse array loading. Current `.npz` format silently ignores mmap_mode.
- **v3.3 status:** All saves use `sparse.save_npz()`. No implementation exists.
- **Wire:** Replace `save_npz` with separate `.npy` saves (data, indices, indptr) in `v2_cross_generator.py` and `atomic_io.py`. Load via `np.load(mmap_mode='r')` in training data loader.
- **Note:** Branch da8e680c has `NPZ_INDICES_ONLY` optimization (indices-only storage for binary crosses) but NOT the mmap format.

### M2. Parallel Cross Steps (Optimization #1B)
- **File:** `ceo/backend-dev-a8040695:v3.3/v2_cross_generator.py`
- **What:** All 12 cross steps run concurrently via ThreadPoolExecutor with memory-aware scheduling (70% RAM ceiling).
- **v3.3 status:** Cross steps run sequentially. No `PARALLEL_CROSS_STEPS` env var exists.
- **Wire:** Merge from branch a8040695. Enabled via `PARALLEL_CROSS_STEPS=1`.

### M3. Adaptive RIGHT_CHUNK Controller (Optimization #1C)
- **File:** `ceo/backend-dev-a8040695:v3.3/v2_cross_generator.py`
- **What:** Rolling RSS-based chunk sizing replaces static per-TF values. MemoryError recovery with auto-halve.
- **v3.3 status:** Static `V2_RIGHT_CHUNK` only. No `AdaptiveChunkController`.
- **Wire:** Merge from branch a8040695. V2_RIGHT_CHUNK becomes max cap.

### M4. force_row_wise for 15m Timeframe (Optimization #9 variant)
- **File:** `v3.3/config.py:359` defines `TF_FORCE_ROW_WISE = frozenset(['15m'])`
- **What:** 15m has 294K rows / ~23K bundles (ratio 12.8) -- row-wise is faster.
- **v3.3 status:** Config EXISTS but `ml_multi_tf.py` NEVER reads `TF_FORCE_ROW_WISE`. Always uses `force_col_wise=True`.
- **Wire:** Add check in `ml_multi_tf.py` training param setup: if tf in TF_FORCE_ROW_WISE, set `force_row_wise=True` and remove `force_col_wise`.
- **Also wire in:** `run_optuna_local.py` (same pattern needed).

### M5. bin_construct_sample_cnt = 5000
- **File:** Not set anywhere in production code
- **What:** LightGBM default is 200,000 (40x higher). Lower value speeds up Dataset construction.
- **v3.3 status:** Not configured. Falls back to LightGBM default.
- **Fix exists on:** `ceo/backend-dev-8dedf3ff` adds to config
- **Wire:** Add to `V3_LGBM_PARAMS` in `config.py`, verify in `validate.py`.

---

## LOW — Minor Optimizations or Future Features

### L1. Warp-Cooperative CUDA Kernel (__ballot_sync)
- **File:** `ceo/backend-dev-22d6ed1e:v3.3/gpu_histogram_fork/src/gpu_histogram.cu`
- **What:** New `sparse_hist_build_warp_kernel` with proper `__ballot_sync` for active lane masking.
- **v3.3 status:** No warp kernel exists. Env var `CUDA_WARP_REDUCE=1` does nothing.
- **Wire:** Merge kernel from branch 22d6ed1e (NOT 4181eede -- that branch has a correctness regression).
- **Priority:** Low -- opt-in via env var, affects GPU histogram only.

### L2. Numba CSC Intersection (Optimization #1A)
- **File:** `ceo/backend-dev-22d6ed1e:v3.3/v2_cross_generator.py`
- **What:** Two-pointer sorted intersection on CSC indptr/indices. Replaces scipy sparse matmul.
- **v3.3 status:** Not implemented. Cross gen uses scipy sparse matmul.
- **Wire:** Merge from branch 22d6ed1e.

### L3. Indices-Only NPZ + Intra-Step Time Flush (Optimization #8, #7H)
- **File:** `ceo/backend-dev-da8e680c:v3.3/atomic_io.py`, `v3.3/v2_cross_generator.py`
- **What:** Binary crosses store only indices (40% smaller files). Time-based flush every 20 min.
- **v3.3 status:** Neither optimization exists.
- **Wire:** Merge from branch da8e680c.

---

## STANDALONE FILES ON v3.3 — NOT IMPORTED BY PIPELINE

These .py files exist on v3.3 but are standalone modules never imported by the main pipeline (`cloud_run_tf.py` / `ml_multi_tf.py` / `run_optuna_local.py`):

| File | Purpose | Called By |
|------|---------|----------|
| `cost_sensitive_obj.py` | Cost-sensitive LightGBM objective function | Self-reference only (docstring example) |
| `gpu_cross_builder.py` | GPU-accelerated cross feature builder | Not imported by v2_cross_generator.py |
| `pipeline_orchestrator.py` | Multi-TF pipeline orchestrator | Not imported (standalone script) |
| `backtest_validation.py` | Walk-forward backtesting validation | Not imported |
| `backtesting_audit.py` | Backtesting audit tool | Not imported |
| `feature_importance_pipeline.py` | SHAP/importance analysis | Not imported by training pipeline |
| `portfolio_aggregator.py` | Multi-TF portfolio aggregation | Not imported by live_trader.py |

---

## TRAINING ENHANCEMENTS ON BRANCHES — NOT MERGED

Branch `ceo/backend-dev-e443f758` contains 5 optimizations that are NOT on v3.3:

| Optimization | What | Impact |
|-------------|------|--------|
| OPT-9: CSR->CSC conversion before training | Convert to CSC for force_col_wise cache locality | 10-20% faster histogram build |
| OPT-10: WilcoxonPruner | Statistical inter-fold pruning | 20-30% fewer wasted trials |
| OPT-11: extra_trees in Optuna | Adds extra_trees to search space | Diversity injection |
| OPT-13: GC disable during training | gc.disable() during CPCV loops | Reduces GC overhead |
| SharedMemory CPCV IPC | Zero-copy CSR passing to parallel workers | Eliminates pickle bottleneck for 15m |

---

## RECOMMENDED MERGE ORDER

### Phase 1 — Critical Fixes (must merge before ANY training)
1. `ceo/backend-dev-8dedf3ff` — Optuna param floors (0.5 -> 0.7) + bin_construct_sample_cnt
2. `ceo/backend-dev-cf941cf8` — Sortino fix + Sobol optimizer + lleaves + inference pruner + memmap merge + CPCV fold reduction

### Phase 2 — High-Impact Optimizations
3. `ceo/backend-dev-e443f758` — Training enhancements (CSC, WilcoxonPruner, extra_trees, GC, SharedMemory)
4. `ceo/backend-dev-106b1d97` — Multi-GPU Optuna trial parallelism
5. `ceo/backend-dev-605acb8a` — EFB pre-bundler

### Phase 3 — Cross-Gen Speed
6. `ceo/backend-dev-702e95d2` — Bitpacked POPCNT co-occurrence
7. `ceo/backend-dev-a8040695` — Parallel cross steps + adaptive chunk controller
8. `ceo/backend-dev-da8e680c` — Indices-only NPZ + time flush
9. `ceo/backend-dev-22d6ed1e` — CUDA warp kernel + Numba CSC intersection

### Phase 4 — Manual Wiring (no branch exists)
10. Wire `TF_FORCE_ROW_WISE` from config.py into ml_multi_tf.py and run_optuna_local.py
11. Implement .npy memmap format (no code exists, only design docs)

---

## BRANCH CONFLICT RISK

| Branch | Files Modified | Conflict Risk |
|--------|---------------|---------------|
| 8dedf3ff | run_optuna_local.py, config.py, validate.py | LOW (small changes) |
| cf941cf8 | exhaustive_optimizer.py, ml_multi_tf.py, v2_cross_generator.py, + 4 new files | MEDIUM |
| e443f758 | ml_multi_tf.py, run_optuna_local.py, cloud_run_tf.py | HIGH (overlaps cf941cf8) |
| 106b1d97 | run_optuna_local.py + 1 new file | HIGH (overlaps e443f758) |
| 605acb8a | v2_cross_generator.py, config.py + 1 new file | MEDIUM |
| 702e95d2 | v2_cross_generator.py + 1 new file | MEDIUM (overlaps 605acb8a) |
| a8040695 | v2_cross_generator.py | HIGH (major rewrite, overlaps all cross-gen branches) |
| da8e680c | v2_cross_generator.py, atomic_io.py | MEDIUM |
| 22d6ed1e | gpu_histogram.cu, v2_cross_generator.py | LOW (CUDA file unique) |

**Recommendation:** Merge sequentially in the order above. Resolve conflicts at each step before proceeding.
