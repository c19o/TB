# V3.3 GPU Histogram Fork — Session Resume

## INSTRUCTION TO NEW SESSION
Read this file completely. Then read RESEARCH.md, ARCHITECTURE.md, and IMPLEMENTATION_PLAN.md in this directory. These files are the complete GPU fork context. The main pipeline context is in v3.3/SESSION_RESUME.md and v3.3/CLAUDE.md.

**CRITICAL: The GPU fork is ISOLATED in v3.3/gpu_histogram_fork/. NEVER modify main pipeline files (ml_multi_tf.py, cloud_run_tf.py, config.py) for GPU fork work.**

---

## STATUS: Phase 4 COMPLETE — GPU histogram training WORKS with EFB

### What Works
- LightGBM fork builds successfully (42/42 objects + linking)
- `device_type="cuda_sparse"` accepted by LightGBM
- GPU detected: RTX 3090, sm_86, 82 SMs, 24GB VRAM
- `SetExternalCSR()` C API exists (LGBM_BoosterSetExternalCSR in c_api.cpp)
- `booster.set_external_csr(X_csr)` Python method exists
- cuSPARSE SpMV histogram benchmark: **99x speedup on real 2.2M features** (1.68ms vs 166.5ms) — Phase 1 standalone only; integrated Phase 4 result is 78x
- All 2.2M feature NPZ rebuilt with min_nonzero=3 (downloaded from cloud)
- Pre-built DLL at: `_build/LightGBM/lib_lightgbm.dll`
- CSR bridge between C API global and tree learner instance (Phase 3 fix)
- DLL rebuilt with ninja (build_utf8 directory)
- CPU training works: **77.64% accuracy** (GPU = CPU, exact match on real labels, verified). GPU fork is a nice-to-have optimization, not a blocker

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

**4. Atomic Kernel Multiclass Stride Bug (FIXED — THE Bug 4 root cause):**
Investigation revealed the scatter kernel and atomic kernel had dangerously wrong comments claiming `d_gradients_` contained "ordered" gradients when it actually contains RAW per-sample gradients indexed by original row ID. `gradients_` (from `SerialTreeLearner`) is set directly from GBDT::TrainOneIter() which pre-slices per class — it is NOT `ordered_gradients_` (which is a separate CPU reordering buffer used by `Dataset::ConstructHistograms`).

**Root cause:** The atomic kernel used `gradients[row * num_classes + class_id]` stride indexing, but GBDT pre-slices gradients per class before passing to the tree learner. The gradient buffer already contains only the current class's gradients — no stride needed. This caused out-of-bounds reads for multiclass (worked by accident when num_classes=1).

**Fixes applied:**
1. All comments corrected: "ordered" → "RAW" throughout both .cu and .h files
2. Atomic kernel indexing fixed: `gradients[row * num_classes + class_id]` → `gradients[row]` (GBDT pre-slices, no stride needed)
3. Parameter names corrected: `ordered_grad` → `raw_grad` / `d_raw_grad` in all forward declarations, definitions, and launch wrappers
4. Overview comment block updated to document that we use `gradients_` (raw) not `ordered_gradients_` (CPU reorder buffer)

**4b. EFB Histogram Offset Mismatch (FIXED — `feature_hist_offsets` remapping):**
The `interleave_grad_kernel` now uses `d_feature_hist_offsets_` to map SpMV feature indices to cumulative bin offsets. Unused features have offset `UINT32_MAX` and are skipped.

**NOTE: bin 1 vs bin 0 offset resolved as part of the extended offset table fix (step 5).**

**5. Build Note:** `lightgbm.exe` target doesn't link `c_api` symbols — must build `_lightgbm` target (DLL) only.

### Phase 4: Full GPU Pipeline Acceleration (COMPLETE)
All GPU histogram bugs resolved. 10/10 training rounds completed on 1w (2.2M features, EFB enabled).

**Phase 4 Results:**
- cuSPARSE SpMV: **78x faster** than CPU for SpMV only, 1.34e-14 relative error (machine precision)
- Performance: 7.46s/round GPU vs ~7s/round CPU — SpMV is fast but round-level GPU is roughly equal to CPU on 1w (small dataset, split finding still CPU)
- GPU memory: 87MB allocated, properly freed
- All 10 training rounds completed successfully with EFB bundling active
- **GPU vs CPU accuracy: EXACT MATCH (77.64% on real labels, verified)**

### Previous Decision (Phase 3): GPU Fork Was Deferred
CPU training delivers 77.64% accuracy for 1w. Phase 3 deferred GPU fork because the EFB histogram mismatch was blocking. Phase 4 fixes the EFB mismatch — SpMV is 78x faster but round-level GPU is roughly equal to CPU on small 1w data (818 rows). GPU benefit grows with larger datasets (1d/4h/1h/15m).

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
  src/treelearner/cuda_sparse_hist_tree_learner.cu  (NEW — 1470 lines CUDA)
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
- All 16+ DBs present locally

### Benchmark Results (measured on RTX 3090)
| Data | CPU (scipy) | GPU (cuSPARSE) | Speedup |
|------|------------|----------------|---------|
| 1w real (2.2M features) | 166.5 ms | 1.68 ms | **99x** |
| 1w old (1.1M features) | 46.4 ms | 0.65 ms | **71x** |
| 1d synthetic (6M features) | 901 ms | 1.90 ms | **473x** |

### Previous 1w Model Baselines
- v3.3 cloud (CPCV, 2.2M features): 67.7% accuracy (4-fold avg, model failed to save due to _parent_ds bug)
- v3.3 CPU (CPCV, 2.2M features): 77.64% accuracy (GPU = CPU exact match verified)
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
3. [DONE] enable_bundle=False (later removed after offset fix)
4. [DONE] feature_hist_offsets mapping — extract cumulative bin offset table from share_state_, upload to GPU, remap SpMV output indices in CUDA kernel
5. [DONE] Extended offset table (used vs total features) — bin 1 vs bin 0 resolved
6. [DONE] Gradient ordering comments (raw, not ordered) — all comments corrected throughout .cu and .h
7. [DONE] Atomic kernel multiclass stride fix (THE Bug 4 root cause) — `gradients[row * num_classes]` → `gradients[row]` (GBDT pre-slices per class)
8. [DONE] Integrate GPU path into ml_multi_tf.py for CPCV training
9. [DONE] Integrate into run_optuna_local.py for GPU Optuna
10. [NEXT] Deploy to cloud, verify on 1d/4h/1h/15m

## KEY FILES
- Binary cache: `v3.3/lgbm_dataset_1w.bin` (252MB, skips 4.5min EFB)
- Fresh NPZ: `v3.3/v2_crosses_BTC_1w.npz` (75MB, 2.2M features)
- Fork DLL: `_build/LightGBM/lib_lightgbm.dll`
- Must go to BOTH: `site-packages/lightgbm/bin/` AND `site-packages/lightgbm/lib/`
- Training script: `gpu_histogram_fork/train_1w_cached.py`
- CPU training script: `v3.3/cloud_run_tf.py` (production pipeline)
