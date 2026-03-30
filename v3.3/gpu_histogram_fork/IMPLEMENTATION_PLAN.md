# GPU Histogram Fork — Implementation Plan

## NON-NEGOTIABLE CONSTRAINTS (Matrix Thesis)

These apply to every phase, every line of code, every decision:

- **NO feature filtering/removal** — ALL features preserved on GPU
- **NO row subsampling** — ALL rows processed
- **Sparse binary cross features ARE the edge** — they stay sparse
- **`feature_pre_filter=False`** must be maintained
- **EFB bundling** stays as-is (CPU-side bundling, GPU builds histograms from bundles)
- **Production model accuracy IDENTICAL to CPU training** — verified per fold

If any optimization violates these constraints, it is rejected. No exceptions.

---

## Phase 1: Standalone Proof of Concept (Weeks 1-2)

### Goal
Write a CUDA kernel that takes CSR sparse data + gradient/hessian vectors and produces histograms identical to LightGBM's CPU output.

### Approach: cuSPARSE SpMV First
Start with the simplest correct approach before optimizing:
1. Use `cusparseSpMV` to multiply CSR feature matrix by gradient vector
2. This gives per-bin gradient sums — exactly what LightGBM's histogram building does
3. No custom kernels yet, just validate the math

### Deliverables

**`test_data_generator.py`**
- Generate synthetic sparse binary CSR matching real data profiles
- Profiles: 4h (17K rows x 3M features), 1d (2.5K x 2M), 1w (400 x 500K)
- Configurable sparsity ratio (target: 99.7%+ zeros, matching real crosses)
- Output: scipy CSR matrix + random gradient/hessian vectors
- Also generate matching dense array for CPU reference

**`histogram_cuda.py`**
- Load CSR to GPU via CuPy or raw CUDA
- Compute per-feature histogram bins using cuSPARSE SpMV
- Handle `max_bin=2` (binary features only need 2 bins)
- Return histograms to CPU as numpy arrays

**`test_histogram_equivalence.py`**
- Build histograms via CPU (replicate LightGBM's logic exactly)
- Build histograms via GPU
- Assert: `np.allclose(cpu_hist, gpu_hist, atol=1e-5)`
- Run on all synthetic profiles (4h, 1d, 1w)
- Log max absolute difference per feature

**`bench_kernel_speed.py`**
- CUDA event timing (not wall clock)
- Measure: H2D transfer, kernel execution, D2H transfer
- Compare against LightGBM CPU histogram time (extracted from logs)
- Report speedup factor

### Target Hardware
- Local RTX 3090 (24GB VRAM)
- 4h profile must fit entirely in VRAM
- 1d and 1w profiles trivially fit

### Success Gate
- Histogram equivalence passes on all profiles
- GPU is at least 2x faster than single-core CPU histogram
- No precision drift across 1000 random gradient vectors

---

## Phase 2: LightGBM Integration (Weeks 3-5)

### Goal
Fork LightGBM and replace the histogram construction hot path with GPU calls while keeping everything else (split finding, tree growing, EFB) on CPU.

### Fork Strategy
1. Fork `microsoft/LightGBM` on GitHub
2. Branch: `gpu-histogram-sparse`
3. Modify ~5 files, target ~500 lines total diff
4. Keep diff minimal for maintainability

### Files to Modify

**`src/treelearner/serial_tree_learner.cpp`**
- `ConstructHistograms()` — the main target
- Add GPU path: if `use_cuda_histogram` is set, call into CUDA wrapper
- CPU path remains untouched as fallback

**`src/treelearner/cuda/cuda_histogram.cu`** (new file)
- CUDA kernel implementation
- CSR data resident on GPU, gradients transferred per round
- Per-node histogram construction

**`src/treelearner/cuda/cuda_histogram.h`** (new file)
- Host-side interface: `init()`, `build_histogram()`, `cleanup()`
- Memory management lifecycle

**`src/io/config.cpp`**
- Add `use_cuda_histogram` parameter
- Auto-detect: if CUDA available and VRAM sufficient, enable

**`CMakeLists.txt`**
- Add `-DUSE_CUDA=ON` option
- CUDA compilation rules
- Fat binary targets: `sm_80` (A100), `sm_86` (3090), `sm_89` (4090), `sm_90` (H100), `sm_100` (5090)

### Multi-Class Handling
LightGBM trains 3 trees per round for 3-class (long/short/hold). Each gets its own gradient/hessian slice:
- Transfer all 3 gradient slices to GPU at round start
- Build histograms for active class only per tree
- No wasted transfers — gradient memory is tiny vs feature data

### EFB Bundle Metadata
- EFB bundling happens on CPU during data load (this is fine, it's one-time)
- Bundle mapping metadata transferred to GPU once at init
- GPU kernel respects bundle boundaries when accumulating histograms
- EFB result is identical — bundles are just column remapping

### GPU Memory Lifecycle
```
init():
  - Allocate CSR on GPU (indices, indptr, data) — stays resident
  - Allocate gradient/hessian buffers (reused each round)
  - Allocate histogram output buffer

per_round():
  - Copy current gradients to GPU (small: num_rows * num_classes * 8 bytes)

per_node():
  - Launch histogram kernel for node's row subset
  - Copy histogram back to CPU for split finding

cleanup():
  - Free all GPU allocations
```

### VRAM Budget (Estimates)
| Component | 4h (17K x 3M) | 1h (80K x 5M) | 15m (294K x 10M) |
|-----------|---------------|----------------|-------------------|
| CSR indices (int32) | ~200 MB | ~1.5 GB | ~8 GB |
| CSR indptr (int64) | ~0.13 MB | ~0.6 MB | ~2.3 MB |
| Gradients (fp32) | ~0.2 MB | ~1 MB | ~3.5 MB |
| Histograms (fp64) | ~48 MB | ~80 MB | ~160 MB |
| **Total** | **~250 MB** | **~1.6 GB** | **~8.2 GB** |

All fit on RTX 3090 (24GB). 1h/15m fit on A100/H100 (80GB).

### Auto-Detect + Fallback
```python
def should_use_gpu():
    if not cuda_available():
        return False  # No GPU
    vram = get_gpu_vram_mb()
    estimated_need = estimate_csr_gpu_bytes(n_rows, nnz)
    if estimated_need > vram * 0.85:  # 85% threshold
        log.warning("CSR too large for GPU VRAM, falling back to CPU")
        return False
    return True
```

### Build System
```bash
# Build the fork
git clone https://github.com/<user>/LightGBM.git
cd LightGBM
mkdir build && cd build
cmake .. -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90;100"
make -j$(nproc)

# Build pip wheel
cd ../python-package
pip install -e .
```

### Testing on Real Data
- Train 4h model (1 fold) with GPU histogram
- Compare against CPU baseline:
  - Histogram values per feature per node (logged)
  - Final model accuracy
  - Feature importance rankings
- Must pass: accuracy within 0.1% of CPU

### Success Gate
- Real 4h data trains successfully on GPU histogram path
- Model accuracy identical to CPU (within 0.1%)
- No NaN/Inf in any histogram
- Clean fallback to CPU when GPU unavailable

---

## Phase 3: Optimization (Weeks 6-7)

### Goal
Replace cuSPARSE with a custom CUDA kernel tuned for our specific data pattern (ultra-sparse binary CSR, max_bin=2), and add pipelining.

### Custom CUDA Kernel
The cuSPARSE SpMV approach works but is generic. Our data has special properties:
- **Binary features** (values are 0 or 1) — no multiply needed, just accumulate
- **max_bin=2** — only 2 histogram bins per feature
- **Ultra-sparse** (~99.7% zeros) — row-parallel is better than column-parallel

**Kernel Design: Row-Parallel with Shared Memory**
```
__global__ void build_histogram_sparse_binary(
    const int* csr_indices,    // column indices
    const int64_t* csr_indptr, // row pointers (int64 for >2B nnz)
    const float* gradients,
    const float* hessians,
    double* hist_grad,         // [num_features, 2]
    double* hist_hess,         // [num_features, 2]
    const int* row_indices,    // which rows belong to this node
    int num_node_rows
) {
    // Each thread processes one row
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_node_rows) return;

    int row = row_indices[tid];
    float g = gradients[row];
    float h = hessians[row];

    // For each non-zero in this row, atomicAdd to bin=1
    for (int64_t j = csr_indptr[row]; j < csr_indptr[row+1]; j++) {
        int col = csr_indices[j];
        atomicAdd(&hist_grad[col * 2 + 1], (double)g);
        atomicAdd(&hist_hess[col * 2 + 1], (double)h);
    }
}
// bin=0 computed by subtraction: total - bin1
```

**Why row-parallel works here:**
- Each row has ~9K non-zeros out of 3M features (0.3% density)
- Atomic contention is extremely low — probability of two threads hitting same column simultaneously is near zero
- No shared memory bank conflicts

### Histogram Subtraction on GPU
LightGBM's histogram subtraction trick: `hist(sibling) = hist(parent) - hist(child)`. Do this on GPU to avoid D2H + CPU subtract + H2D:
```
__global__ void histogram_subtract(
    const double* parent_hist,
    const double* child_hist,
    double* sibling_hist,
    int num_bins  // num_features * 2
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_bins) {
        sibling_hist[i] = parent_hist[i] - child_hist[i];
    }
}
```

### GPU-Side Row Partitioning
After split finding (CPU), the row indices for left/right children are computed. Do this on GPU:
- Avoids transferring row masks back and forth
- Parallel partition with CUB `DevicePartition`

### 3-Stream Pipeline
Overlap data transfer with computation:
```
Stream 0: H2D gradients for round N+1
Stream 1: Compute histograms for round N
Stream 2: D2H histograms from round N-1
```
Effective overlap depends on transfer size vs compute time. For 4h data, compute dominates (good). For 1w data, transfer dominates (pipeline less beneficial but data is tiny anyway).

### Benchmark Matrix
| Timeframe | Rows | Features | NNZ | Expected Speedup |
|-----------|------|----------|-----|-----------------|
| 1w | 400 | 500K | 1.2M | 2-3x (data too small for GPU efficiency) |
| 1d | 2.5K | 2M | 22M | 3-5x |
| 4h | 17K | 3M | 150M | 5-8x |
| 1h | 80K | 5M | 1.2B | 8-10x |
| 15m | 294K | 10M | 8.8B | 8-10x (if fits in VRAM) |

### Multi-GPU Feature Partitioning
If a single GPU cannot hold the full CSR (15m on 24GB cards):
- Partition features across GPUs
- Each GPU builds histograms for its feature subset
- Merge on CPU (simple concatenation)
- Only useful for multi-GPU rigs; single large GPU (80GB) handles all TFs

Evaluate cost/benefit before implementing — communication overhead may negate gains.

### Success Gate
- Custom kernel is faster than cuSPARSE path on all profiles
- Histogram equivalence still passes (atol=1e-5)
- 3-stream pipeline shows measurable overlap on 4h+ data
- No regressions in model accuracy

---

## Phase 4: Production Integration (Week 8)

### Goal
Make GPU histogram training a drop-in replacement in the existing pipeline with zero friction.

### Python Wrapper: `gpu_train()`
```python
def gpu_train(params, train_set, num_boost_round, folds, ...):
    """Drop-in replacement for lgb.train with GPU histogram.

    Automatically enables GPU histogram if:
    1. CUDA is available
    2. VRAM is sufficient for the dataset
    3. Fork wheel is installed

    Falls back to CPU transparently on any failure.
    """
    params = params.copy()
    if should_use_gpu():
        params['use_cuda_histogram'] = True
        params['device_type'] = 'cpu'  # everything else stays CPU
        log.info(f"GPU histogram enabled (VRAM: {get_vram_mb()}MB)")
    else:
        log.info("GPU histogram unavailable, using CPU")

    return lgb.train(params, train_set, num_boost_round, ...)
```

### `cloud_run_tf.py` Integration
Minimal changes:
- Replace `lgb.train(...)` call with `gpu_train(...)`
- No other changes needed — all config, CPCV, feature building stays the same
- The fork's LightGBM is a drop-in; `import lightgbm as lgb` works as before

### Pre-Built Wheel Deployment
```bash
# Build wheel once on build machine
cd LightGBM/python-package
python setup.py bdist_wheel
# Output: lightgbm-4.x.x+gpuhist-cp310-linux_x86_64.whl

# Include in code tarball
tar czf v33_code.tar.gz v3.3/ lightgbm-4.x.x+gpuhist-*.whl

# On cloud machine (5-second install):
pip install lightgbm-4.x.x+gpuhist-cp310-linux_x86_64.whl --force-reinstall
```

### Deployment Checklist
- [ ] Wheel built for Python 3.10 + CUDA 12.x
- [ ] Fat binary covers sm_80/86/89/90/100
- [ ] `pip install wheel` works on fresh vast.ai machine
- [ ] `import lightgbm; lightgbm.__version__` shows custom build
- [ ] `gpu_train()` auto-detects GPU on cloud
- [ ] `gpu_train()` falls back to CPU on local (if VRAM insufficient)
- [ ] 4h model trained with GPU matches CPU baseline
- [ ] All 5 TFs trained successfully on appropriate hardware

### Documentation Updates
- `FOLD_STRATEGY.md` — add GPU histogram section
- `TRAINING_*.md` — update with GPU timing estimates
- `SESSION_RESUME.md` — document fork location and build instructions

### Success Gate
- Cloud deploy takes < 30 seconds (wheel install)
- Training completes on all TFs without manual intervention
- Model accuracy identical to CPU on all TFs
- Graceful fallback confirmed on CPU-only machines

---

## Benchmark Framework (Standalone Scripts)

All scripts live in `v3.3/gpu_histogram_fork/benchmarks/`:

| Script | Purpose |
|--------|---------|
| `generate_test_data.py` | Synthetic sparse binary CSR for all TF profiles |
| `test_histogram_equivalence.py` | CPU vs GPU histogram comparison (pass/fail) |
| `bench_kernel_speed.py` | CUDA event timing, speedup factor |
| `bench_end_to_end.py` | Full fold training: GPU fork vs stock LightGBM |
| `test_stress.py` | 15m scale (294K x 10M), int64 indptr, OOM recovery |

### `test_stress.py` Details
- Generate 15m-scale CSR with int64 indptr (NNZ > 2^31)
- Attempt GPU histogram — should either succeed or gracefully fall back
- Test OOM recovery: allocate more than VRAM, verify fallback triggers
- Test int64 indptr handling (critical — our real data exceeds int32 limits)

---

## Risk Register

| # | Risk | Impact | Likelihood | Mitigation |
|---|------|--------|------------|------------|
| 1 | CSR too large for GPU VRAM on 1h/15m | Can't use GPU for largest TFs | Medium | VRAM check + automatic CPU fallback. 80GB GPUs (A100/H100) handle all TFs. |
| 2 | Atomic contention in custom kernel | Lower speedup than expected | Low | Ultra-sparse data means near-zero contention. Fallback to cuSPARSE if needed. |
| 3 | FP precision differences (GPU vs CPU) | Model divergence, different accuracy | Medium | Assert histogram equivalence every fold. Use fp64 atomics. Fail loud if atol>1e-5. |
| 4 | LightGBM upstream changes break fork | Maintenance burden | Low | Minimal diff (~500 lines). Pin to specific LightGBM version. Rebase quarterly. |
| 5 | EFB not reproducible on GPU | Wrong histograms, wrong model | Low | EFB stays 100% on CPU. Only post-EFB histogram building moves to GPU. |
| 6 | CUDA driver incompatibility on cloud | Wheel fails to load | Medium | Fat binary (sm_80-100). Require driver 535+. Test on 3 different cloud providers. |
| 7 | int64 indptr not handled in kernel | Crash on 15m data | High if missed | Use int64 throughout. Test explicitly in stress tests. |
| 8 | Build system breaks on different OS | Can't build wheel | Low | Linux-only target (all training is on Linux). Pre-built wheel eliminates build step. |

---

## Timeline

| Week | Phase | Deliverable | Verification |
|------|-------|-------------|-------------|
| 1 | Phase 1 | `test_data_generator.py`, cuSPARSE histogram kernel | Synthetic data generates correctly |
| 2 | Phase 1 | `test_histogram_equivalence.py`, `bench_kernel_speed.py` | Equivalence passes, speedup measured |
| 3 | Phase 2 | LightGBM fork, CMake build, basic GPU histogram path | Fork builds with CUDA, stock tests pass |
| 4 | Phase 2 | Multi-class support, EFB metadata transfer | 3-class training works on synthetic data |
| 5 | Phase 2 | Real 4h data training, auto-detect, fallback | 4h accuracy matches CPU baseline |
| 6 | Phase 3 | Custom CUDA kernel, histogram subtraction | Faster than cuSPARSE, equivalence holds |
| 7 | Phase 3 | 3-stream pipeline, benchmark all TFs | Speedup numbers for all 5 TFs |
| 8 | Phase 4 | `gpu_train()` wrapper, wheel, cloud deploy, docs | End-to-end cloud training works |

---

## Decision Log

Decisions will be recorded here as the project progresses.

| Date | Decision | Rationale |
|------|----------|-----------|
| — | cuSPARSE first, custom kernel second | Get correct histograms before optimizing |
| — | EFB stays on CPU | One-time cost, avoids reproducing complex bundling logic on GPU |
| — | Row-parallel kernel (not column-parallel) | Ultra-sparse rows mean minimal atomic contention |
| — | fp64 atomics for histogram accumulation | Matches LightGBM CPU precision, prevents drift |
| — | Fat binary not JIT | 5-second cloud install, no NVCC needed on target machine |
| — | Single LightGBM fork, not a plugin | Simpler integration, no IPC overhead, direct function call |
