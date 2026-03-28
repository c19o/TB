# V3.3 GPU Histogram Fork — Session Resume

## INSTRUCTION TO NEW SESSION
Read this file completely. Then read RESEARCH.md, ARCHITECTURE.md, and IMPLEMENTATION_PLAN.md in this directory. These files are the complete GPU fork context. The main pipeline context is in v3.3/SESSION_RESUME.md and v3.3/CLAUDE.md.

**CRITICAL: The GPU fork is ISOLATED in v3.3/gpu_histogram_fork/. NEVER modify main pipeline files (ml_multi_tf.py, cloud_run_tf.py, config.py) for GPU fork work.**

---

## STATUS: PHASE 2 — LightGBM C++ Fork Compiles, Needs Init() Fix

### What Works
- LightGBM fork builds successfully (42/42 objects + linking)
- `device_type="cuda_sparse"` accepted by LightGBM
- GPU detected: RTX 3090, sm_86, 82 SMs, 24GB VRAM
- `SetExternalCSR()` C API exists (LGBM_BoosterSetExternalCSR in c_api.cpp)
- `booster.set_external_csr(X_csr)` Python method exists
- cuSPARSE SpMV histogram benchmark: **99x speedup on real 2.2M features** (1.68ms vs 166.5ms)
- All 2.2M feature NPZ rebuilt with min_nonzero=3 (downloaded from cloud)
- Pre-built DLL at: `_build/LightGBM/lib_lightgbm.dll`

### What's Broken — THE ONE BUG TO FIX
**`Init()` crashes before `set_external_csr()` can be called.**

The sequence is:
1. `lgb.Booster(params, ds)` → calls `Init()` → calls `UploadCSR()` → FAILS (no CSR set yet)
2. `booster.set_external_csr(X)` → never reached

**Fix needed**: In `cuda_sparse_hist_tree_learner.cu`, make `UploadCSR()` a no-op if `has_external_csr_ == false`. Defer the actual upload to the first `ConstructHistograms()` call. The CSR will be set between Booster creation and the first `update()` call.

Specifically in the .cu file:
```cpp
void CUDASparseHistTreeLearner::Init(...) {
    SerialTreeLearner::Init(train_data, is_constant_hessian);
    InitGPU();  // detect GPU, allocate streams
    // DON'T call UploadCSR() here — CSR not set yet
    // UploadCSR() will be called from first ConstructHistograms()
}

void CUDASparseHistTreeLearner::ConstructHistograms(...) {
    if (!csr_uploaded_ && has_external_csr_) {
        UploadCSR();  // First call — upload CSR now
        csr_uploaded_ = true;
    }
    // ... rest of histogram building
}
```

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

## NEXT STEPS (in order)

1. **Fix Init() crash** — defer UploadCSR to first ConstructHistograms
2. **Rebuild DLL** — cmake + ninja in _build/LightGBM/build_link
3. **Test locally** — train_1w_cached.py on 2.2M feature data
4. **Verify model quality** — compare accuracy vs 67.7% cloud baseline
5. **Train 1d, 4h** locally (fit in 3090 VRAM)
6. **Rent cloud machine** for 1h/15m (need 80GB+ VRAM)
