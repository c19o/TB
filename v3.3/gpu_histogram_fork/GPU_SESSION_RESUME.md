# V3.3 GPU Histogram Fork — Session Resume

## INSTRUCTION TO NEW SESSION
Read this file completely. Then read RESEARCH.md, ARCHITECTURE.md, and IMPLEMENTATION_PLAN.md in this directory. These files are the complete GPU fork context. The main pipeline context is in v3.3/SESSION_RESUME.md and v3.3/CLAUDE.md.

**CRITICAL: The GPU fork is ISOLATED in v3.3/gpu_histogram_fork/. NEVER modify main pipeline files (ml_multi_tf.py, cloud_run_tf.py, config.py) for GPU fork work.**

---

## STATUS: PHASE 3 — BLOCKED on EFB Histogram Offset Mapping

### What Works
- LightGBM fork builds successfully (42/42 objects + linking)
- `device_type="cuda_sparse"` accepted by LightGBM
- GPU detected: RTX 3090, sm_86, 82 SMs, 24GB VRAM
- `SetExternalCSR()` C API exists (LGBM_BoosterSetExternalCSR in c_api.cpp)
- `booster.set_external_csr(X_csr)` Python method exists
- cuSPARSE SpMV histogram benchmark: **99x speedup on real 2.2M features** (1.68ms vs 166.5ms)
- All 2.2M feature NPZ rebuilt with min_nonzero=3 (downloaded from cloud)
- Pre-built DLL at: `_build/LightGBM/lib_lightgbm.dll`
- CSR bridge between C API global and tree learner instance (Phase 3 fix)
- DLL rebuilt with ninja (build_utf8 directory)
- CPU training works: **73.9% accuracy in 5 minutes for 1w** — GPU fork is a nice-to-have optimization, not a blocker

### Phase 2 Bug (RESOLVED) — Init() Crash
**Was:** `Init()` crashed before `set_external_csr()` could be called. Deferred upload fix applied in Phase 2.

### Phase 3 Bugs

**1. CSR Bridge Bug (FIXED):**
The C API stored CSR in a global struct (`g_external_csr`) but the tree learner only checked its own member variable (`has_external_csr_`). The global was set by `LGBM_BoosterSetExternalCSR()` but the tree learner's `ConstructHistograms()` never read from it.

**Fix:** Added `extern "C"` declarations for the getter functions (`LGBM_GetExternalCSRIndptr`, `LGBM_GetExternalCSRIndices`, etc.) and a bridge in `ConstructHistograms()` that reads from the global and calls `SetExternalCSR()` on the tree learner instance.

**2. Dangling Pointer Bug (FIXED):**
`basic.py`'s `set_external_csr()` created local numpy arrays whose pointers were stored in the C API global but could be garbage-collected after the method returned. The C++ side held raw pointers to freed memory.

**Fix:** Store as Booster instance attributes (`self._external_csr_indptr`, `self._external_csr_indices`) so they live as long as the Booster object.

**3. enable_bundle=False (APPLIED but INSUFFICIENT):**
Disabling EFB bundling was attempted to make feature IDs map 1:1 to histogram bins. However, LightGBM still has a feature-to-bin offset mapping even with `enable_bundle=False` — constant/unused features get 0 bins, others get 2 (for binary features). The SpMV output is indexed by raw feature ID, but the histogram buffer is indexed by cumulative bin offsets.

**4. EFB Histogram Offset Mismatch (BLOCKED):**
This is DEEPER than expected. Even without bundling, LightGBM's histogram buffer layout uses cumulative bin offsets, not raw feature indices. The mapping works as follows:
- Feature 0 (constant, 0 bins) → offset 0
- Feature 1 (binary, 2 bins) → offset 0
- Feature 2 (constant, 0 bins) → offset 2
- Feature 3 (binary, 2 bins) → offset 2
- ...and so on for 2.2M features

The SpMV kernel writes gradients to `histogram[feature_id]`, but LightGBM expects them at `histogram[cumulative_bin_offset[feature_id]]`. Without this remapping, gradient/hessian values land in the wrong histogram bins, producing garbage splits.

**The fix would require:**
1. Extracting the `feature_hist_offsets` array from LightGBM's `share_state_` (the cumulative bin offset table)
2. Uploading it to GPU memory
3. Using it in the CUDA kernel to remap the SpMV output index before writing to the histogram buffer
4. Keeping this array synchronized across boosting iterations (features may be dropped/rebinned)

This is a significant kernel rewrite, not a simple patch.

**5. Build Note:** `lightgbm.exe` target doesn't link `c_api` symbols — must build `_lightgbm` target (DLL) only.

### Decision: GPU Fork Deferred
CPU training delivers 73.9% accuracy in 5 minutes for 1w. The GPU fork would speed up histogram construction but is not needed to produce trading models. All timeframes will be trained via the CPU pipeline first. GPU fork can be revisited after all TF models are trained and verified for live trading.

### Build Environment (Windows 11, user's local machine)
- CUDA Toolkit 12.6 at `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6`
- VS Build Tools 2025 (MSVC 19.50) — needs `--allow-unsupported-compiler`
- CMake via VS Build Tools
- Ninja via Python pip
- Build command: `vcvarsall.bat x64` then `cmake -G Ninja -DUSE_CUDA_SPARSE=ON ...`
- DLL installed to: `site-packages/lightgbm/bin/lib_lightgbm.dll` AND `site-packages/lightgbm/lib/lib_lightgbm.dll` (BOTH locations needed!)
- Patched `basic.py` with `set_external_csr()` method

### Build Gotchas (LEARNED THE HARD WAY)
1. **CMakeLists.txt**: CUDA sparse source MUST be added to LGBM_SOURCES BEFORE `add_library(lightgbm_objs OBJECT)` at line 495. Appending at end = file never compiled.
2. **FMT_UNICODE=0**: Must define in .cu file before includes. nvcc + MSVC 19.50 doesn't set `_MSVC_EXECUTION_CHARACTER_SET` correctly.
3. **FORCE:MULTIPLE**: Linker needs this for OMP_NUM_THREADS symbol conflict between openmp_wrapper.cpp and our .cu.
4. **config.cpp**: Must add "cuda_sparse" to `GetDeviceType()`, device enum, and force_row_wise block. The options comment in config.h is NOT enough.
5. **DLL location**: LightGBM Python searches `bin/` before `lib/`. Must copy to BOTH.
6. **CUDA headers in non-CUDA TUs**: Guard with `#ifdef __CUDACC__` and provide forward decls for c_api.cpp.

### Files Modified in LightGBM Fork
```
_build/LightGBM/
  src/treelearner/cuda_sparse_hist_tree_learner.cu  (NEW — 1232 lines CUDA)
  src/treelearner/cuda_sparse_hist_tree_learner.h   (NEW — header)
  src/treelearner/tree_learner.cpp                  (EDIT — factory dispatch)
  src/io/config.cpp                                 (EDIT — cuda_sparse validation)
  src/io/dataset.cpp                                (EDIT — preserve sparse for cuda_sparse)
  src/c_api.cpp                                     (EDIT — SetExternalCSR C API)
  src/c_api.h                                       (EDIT — SetExternalCSR declaration)
  src/boosting/gbdt.h                               (EDIT — GetTreeLearner() public getter)
  src/application/application.cpp                   (EDIT — CUDA allocator for cuda_sparse)
  include/LightGBM/config.h                         (EDIT — device_type option)
  CMakeLists.txt                                    (EDIT — USE_CUDA_SPARSE build option)
  python-package/lightgbm/basic.py                  (EDIT — set_external_csr method)
```

### Real Data Available Locally
- `v3.3/v2_crosses_BTC_1w.npz` — 818 rows × 2,195,129 features, 48M NNZ (FRESH, min_nonzero=3)
- `v3.3/v2_cross_names_BTC_1w.json` — 2,195,129 feature names
- `v3.3/features_BTC_1w.parquet` — 818 rows × 3,331 base features
- All 16 DBs present locally

### Benchmark Results (measured on RTX 3090)
| Data | CPU (scipy) | GPU (cuSPARSE) | Speedup |
|------|------------|----------------|---------|
| 1w real (2.2M features) | 166.5 ms | 1.68 ms | **99x** |
| 1w old (1.1M features) | 46.4 ms | 0.65 ms | **71x** |
| 1d synthetic (6M features) | 901 ms | 1.90 ms | **473x** |

### Previous 1w Model Baselines
- v3.3 cloud (CPCV, 2.2M features): 67.7% accuracy (4-fold avg, model failed to save due to _parent_ds bug)
- v3.3 CPU (CPCV, 2.2M features): 73.9% accuracy in 5 minutes
- v3.2 (1.1M features): 71.9% accuracy (OLD model, stale NPZ)
- v3.0 (658K features): unknown

### Git Commits
- `1f7db7c` — GPU fork Phase 1 (57 files, 25,537 lines)
- `d6e30c3` — GPU fork Phase 2 (39 files, 9,763 lines)

### Project Structure
```
gpu_histogram_fork/          ~70 files, ~30,000 lines
├── Docs: README, RESEARCH, ARCHITECTURE, IMPLEMENTATION_PLAN, STATUS, VASTAI_DEPLOY, BUILD_WINDOWS
├── Core fork: src/treelearner/ (cuda_sparse_hist_tree_learner.cu/.h, tree_learner.cpp patch)
├── Python GPU: src/ (histogram_cusparse, histogram_atomic, memory_manager, leaf_partition, lgbm_csr_bridge, train_pipeline, cuda_compat, multi_gpu, cloud_gpu_integration)
├── Benchmarks: benchmark/ + cupy_spmv_benchmark.py, benchmark_real_1w.py, quick_bench.py
├── Tests: tests/ (equivalence, stress, integration, accuracy_validator, histogram_output_mapper)
├── Deploy: deploy_vastai.sh, setup_universal.sh, vastai_oneliner.sh, build_linux_wheel.sh, Dockerfile
├── Training: train_1w_gpu.py, train_1w_cached.py, cupy_gpu_train.py, test_real_1w.py, test_1w_end_to_end.py
└── Build: CMakeLists.txt, Makefile, build_and_test.sh/.ps1, build_wheel.sh, build_windows.ps1
```

---

## NEXT STEPS

1. [DONE] CSR bridge fix — global-to-instance bridge in ConstructHistograms()
2. [DONE] Dangling pointer fix — store CSR arrays as Booster instance attributes
3. [DONE] enable_bundle=False — applied but does not solve the offset mapping problem
4. [BLOCKED] feature_hist_offsets mapping needed for GPU histograms — requires extracting cumulative bin offset table from share_state_, uploading to GPU, and remapping SpMV output indices in the CUDA kernel. Significant kernel rewrite.
5. [DECISION] CPU training works at 73.9% accuracy in 5 minutes for 1w — GPU fork deferred to after all TF models are trained and verified for live trading
6. **Current focus: train all TFs via CPU pipeline, verify models for trading**

## KEY FILES
- Binary cache: `v3.3/lgbm_dataset_1w.bin` (252MB, skips 4.5min EFB)
- Fresh NPZ: `v3.3/v2_crosses_BTC_1w.npz` (75MB, 2.2M features)
- Fork DLL: `_build/LightGBM/lib_lightgbm.dll`
- Must go to BOTH: `site-packages/lightgbm/bin/` AND `site-packages/lightgbm/lib/`
- Training script: `gpu_histogram_fork/train_1w_cached.py`
- CPU training script: `v3.3/cloud_run_tf.py` (production pipeline)
