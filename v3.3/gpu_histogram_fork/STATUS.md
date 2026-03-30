# GPU Histogram Fork -- Status Dashboard

**Project:** GPU-accelerated sparse histogram builder for LightGBM
**Goal:** Replace LightGBM's CPU histogram pass with GPU SpMV for 2M+ sparse feature matrices
**Branch:** v3.3
**Last Updated:** 2026-03-27

---

## Phase Overview

| Phase | Name | Status |
|-------|------|--------|
| 1 | Proof of Concept | COMPLETE |
| 2 | LightGBM Fork | IN PROGRESS |
| 3 | Optimization | NOT STARTED |
| 4 | Production | NOT STARTED |

---

## Phase 1: Proof of Concept -- COMPLETE

- [x] cuSPARSE SpMV approach implemented (`src/histogram_cusparse.py`)
- [x] Atomic scatter kernel implemented (`src/histogram_atomic.py`)
- [x] CPU reference histogram builder (`src/cpu_histogram_reference.py`, `benchmark/cpu_histogram_reference.py`)
- [x] Test data generator (`src/generate_test_data.py`, `benchmark/generate_test_data.py`)
- [x] Benchmark framework (`benchmark/bench_kernel_speed.py`, `benchmark/bench_end_to_end.py`)
- [x] Histogram equivalence tests (32 tests, all pass on CPU)
- [x] Stress tests (7 tests)
- [x] MEASURED: 473x SpMV speedup on RTX 3090 (1d: 5727 x 6M)
- [x] MEASURED: 160x SpMV speedup on RTX 3090 (1w: 818 x 2.2M)
- [x] All correctness checks PASS (relative error < 5e-16)
- [x] Memory: 6M features fits in 2.6GB (11% of RTX 3090 VRAM)

## Phase 2: LightGBM Fork -- IN PROGRESS

- [x] cuda_sparse_hist_tree_learner.h (C++ header)
- [ ] cuda_sparse_hist_tree_learner.cu (CUDA implementation)
- [ ] config.h patch (add `cuda_sparse` device type)
- [ ] tree_learner.cpp patch (factory dispatch)
- [ ] CMakeLists.txt patch (`USE_CUDA_SPARSE` build option)
- [ ] Build on Windows (RTX 3090 local)
- [ ] Build on Linux (cloud deployment)
- [ ] Test on synthetic data
- [ ] Test on real 1w data (818 rows x 2.2M features)

## Phase 3: Optimization -- NOT STARTED

- [ ] Histogram subtraction on GPU
- [ ] GPU-side partition updates
- [ ] 3-stream pipeline overlap
- [ ] Multi-GPU support

## Phase 4: Production -- NOT STARTED

- [ ] Pre-built pip wheel
- [ ] cloud_run_tf.py integration
- [ ] All TF benchmarks

---

## Benchmark Results (RTX 3090)

| Matrix | CPU scipy | GPU SpMV | Speedup | BW Util |
|--------|-----------|----------|---------|---------|
| 1w (818 x 2.2M) | 38ms | 0.24ms | 160x | 45% |
| 1d (5727 x 3M) | 359ms | 0.98ms | 367x | 73% |
| 1d (5727 x 6M) | 901ms | 1.90ms | 473x | 75% |

---

## GPU Compatibility

VRAM determines which timeframes fit. Rows x features drives matrix size.

| GPU | VRAM | 1w | 1d | 4h | 1h | 15m |
|-----|------|----|----|----|----|-----|
| RTX 3090 (24GB) | 24GB | YES | YES | YES | NO | NO |
| A100 (80GB) | 80GB | YES | YES | YES | YES | YES |
| H100 (80GB) | 80GB | YES | YES | YES | YES | YES |

---

## Files

### Root

| File | Description |
|------|-------------|
| `README.md` | Project overview and quickstart |
| `ARCHITECTURE.md` | Detailed architecture design and data flow |
| `RESEARCH.md` | Background research on GPU histogram approaches |
| `IMPLEMENTATION_PLAN.md` | Step-by-step implementation roadmap |
| `BUILD_WINDOWS.md` | Windows build instructions for RTX 3090 local dev |
| `STATUS.md` | This file -- live status dashboard |
| `CMakeLists.txt` | CMake build configuration for CUDA/C++ components |
| `Makefile` | Make targets for build convenience |
| `setup.py` | Python package setup for pip install |
| `__init__.py` | Package init, exports top-level API |
| `cupy_spmv_benchmark.py` | Standalone CuPy SpMV benchmark script |

### src/

| File | Description |
|------|-------------|
| `__init__.py` | Source package init |
| `histogram_cusparse.py` | cuSPARSE-based SpMV histogram kernel (primary approach) |
| `histogram_atomic.py` | Atomic scatter histogram kernel (alternative approach) |
| `gpu_histogram_cusparse.py` | GPU histogram wrapper using cuSPARSE backend |
| `gpu_histogram_atomic.py` | GPU histogram wrapper using atomic backend |
| `gpu_histogram_wrapper.py` | Unified GPU histogram API (dispatches to cusparse or atomic) |
| `gpu_histogram.h` | C++ header for CUDA histogram kernels |
| `gpu_histogram.cu` | CUDA kernel implementations (SpMV, atomic, binning) |
| `histogram_output_mapper.py` | Maps raw GPU histogram output to LightGBM format |
| `cpu_histogram_reference.py` | CPU reference implementation for correctness validation |
| `generate_test_data.py` | Generates synthetic sparse matrices matching real TF shapes |
| `lgbm_integration.py` | LightGBM integration layer (hooks into training loop) |
| `memory_manager.py` | GPU memory allocation, pooling, and OOM prevention |
| `leaf_partition.py` | Leaf-level data partitioning for tree building |

### src/treelearner/

| File | Description |
|------|-------------|
| `cuda_sparse_hist_tree_learner.h` | C++ header for the custom LightGBM tree learner |

### benchmark/

| File | Description |
|------|-------------|
| `__init__.py` | Benchmark package init |
| `bench_kernel_speed.py` | Microbenchmark: raw kernel latency (SpMV vs atomic vs CPU) |
| `bench_end_to_end.py` | End-to-end benchmark: full histogram build including transfers |
| `cpu_histogram_reference.py` | CPU baseline for benchmark comparisons |
| `generate_test_data.py` | Test data generator for benchmark matrices |

### tests/

| File | Description |
|------|-------------|
| `__init__.py` | Test package init |
| `conftest.py` | Pytest fixtures (sparse matrix generators, GPU skip markers) |
| `test_histogram_equivalence.py` | 32 equivalence tests: GPU histograms match CPU reference |
| `test_stress.py` | 7 stress tests: OOM handling, large matrices, edge cases |
