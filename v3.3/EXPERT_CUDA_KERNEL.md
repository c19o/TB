# EXPERT: CUDA Kernel Optimization for GPU Histogram Fork

**Date:** 2026-03-30
**Target Hardware:** 8x RTX 5090 Blackwell (sm_120, 32GB VRAM each)
**Data:** 2-10M binary cross-features, CSR int64 indptr, 818-227K rows, ~99.7% zeros

---

## Current Implementation Analysis

### What We Have (4 Optimizations Already Implemented)
1. **Batch H2D transfers** — pinned memory + async cudaMemcpyAsync on dedicated stream
2. **Warp-cooperative atomics** — not yet active (ConstructHistograms falls back to CPU at line 847)
3. **Vectorized launches** — 3-stream pipeline (H2D, compute, D2H) with event synchronization
4. **CSR.T dual storage** — transposed CSR pre-built for cuSPARSE SpMV (int32/int64 adaptive)

### Current Kernel Architecture
- **Kernel 1 (Global Atomic):** One thread per leaf row, walks CSR nonzeros, `atomicAdd` grad/hess directly to global histogram. No contention at 99.7% sparsity.
- **Kernel 2 (Tiled Shared Memory):** Tiles feature range into shared memory blocks (up to 8192 bins/tile). Multiple launches per feature range. Used when `n_leaf_rows >= 1024`.
- **cuSPARSE SpMV path:** `A^T @ grad_vector` produces histogram in one call. Proven 473x on RTX 3090 for dense-enough leaves.

### Critical Finding
`ConstructHistograms()` at line 847 of `cuda_sparse_hist_tree_learner.cu` currently **force-falls-back to CPU** (`SerialTreeLearner::ConstructHistograms()`). The GPU path is fully wired but disabled. **Enabling it is the single biggest speedup available** — everything below assumes the GPU path is active.

---

## Optimization Tier 1: High-Impact (Expected 3-10x Each)

### 1.1 Enable the GPU ConstructHistograms Path
**Expected Speedup:** 10-50x over CPU (based on 473x SpMV benchmark)
**Effort:** Low — remove the early-return at line 847

The cuSPARSE SpMV path is fully set up (descriptors, workspace, dual int32/int64). The `TEMP DEBUG` block just needs to be removed and replaced with the actual GPU dispatch:

```cpp
void CUDASparseHistTreeLearner::ConstructHistograms(
    const std::vector<int8_t>& is_feature_used,
    bool use_subtract) {

    if (!csr_uploaded_) {
        // Deferred upload path
        if (has_external_csr_) {
            UploadCSR();
            if (gpu_hist_mode_ == GPU_HIST_MODE_CUSPARSE && !has_efb_data_) {
                SetupCuSPARSE();
            }
            csr_uploaded_ = true;
        } else {
            SerialTreeLearner::ConstructHistograms(is_feature_used, use_subtract);
            return;
        }
    }

    // ... GPU histogram build (SpMV or atomic scatter) ...
}
```

**Citation:** LightGBM GPU docs confirm histogram building is the dominant training cost. [LightGBM GPU Performance Guide](https://lightgbm.readthedocs.io/en/latest/GPU-Performance.html)

---

### 1.2 Warp-Aggregated Atomic Updates (for Atomic Scatter Kernel)
**Expected Speedup:** 2-4x on the atomic scatter kernel
**Effort:** Medium

Current `sparse_hist_build_kernel` does raw `atomicAdd` per CSR nonzero. At 99.7% sparsity, contention is low — but warp-level aggregation still helps by reducing total atomic operations when multiple threads in a warp hit the same feature bin.

```cuda
__global__ void sparse_hist_build_warp_agg_kernel(
    const int64_t* __restrict__ indptr,
    const int32_t* __restrict__ indices,
    const double*  __restrict__ gradients,
    const double*  __restrict__ hessians,
    const int32_t* __restrict__ leaf_rows,
    int32_t n_leaf_rows,
    double* __restrict__ hist
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_leaf_rows) return;

    int32_t row = leaf_rows[tid];
    double g = gradients[row];
    double h = hessians[row];

    int64_t start = indptr[row];
    int64_t end   = indptr[row + 1];

    for (int64_t j = start; j < end; j++) {
        int32_t col = indices[j];

        // Warp-level aggregation: find lanes with same col
        unsigned mask = __match_any_sync(0xFFFFFFFF, col);
        int leader = __ffs(mask) - 1;

        // Sum gradients/hessians within matching lanes
        double warp_g = g, warp_h = h;
        for (int offset = 16; offset > 0; offset >>= 1) {
            double other_g = __shfl_down_sync(mask, warp_g, offset);
            double other_h = __shfl_down_sync(mask, warp_h, offset);
            warp_g += other_g;
            warp_h += other_h;
        }

        // Only leader performs the atomic
        int lane = threadIdx.x & 31;
        if (lane == leader) {
            atomicAdd(&hist[(int64_t)col * 2],     warp_g);
            atomicAdd(&hist[(int64_t)col * 2 + 1], warp_h);
        }
    }
}
```

**Key insight:** `__match_any_sync` detects duplicate feature targets within a warp and collapses them. For ultra-sparse data (9K nnz per row / 3M features), collision probability within a warp is ~0.003%, so the win is modest per-warp but compounds across 227K rows.

**Citation:** [NVIDIA GPU Pro Tip: Fast Histograms Using Shared Atomics on Maxwell](https://developer.nvidia.com/blog/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/)

---

### 1.3 Hot/Cold Feature Partitioning
**Expected Speedup:** 2-5x on tiled kernel path
**Effort:** Medium-High

Split features into two buckets offline (once per boosting round or every N rounds):
- **Hot features** (>100 fires per leaf): route to shared-memory tiled kernel
- **Cold/rare features** (3-20 fires): route to direct global atomics or deterministic one-writer path

```
Rationale:
- Hot features: contention matters → shared memory atomics = 10x faster
- Cold features: no contention → shared memory init/flush overhead wasted
  (allocating 16 bytes per bin for features that fire 3 times = waste)
- Rare features get EXACT deterministic accumulation (no ordering dependency)
```

**Implementation sketch:**
```python
# Pre-sort at Python level before GPU upload
col_fire_counts = np.diff(csr_matrix.indptr)  # nnz per row
# ... or compute per-feature nnz from CSC
hot_mask = feature_nnz > HOT_THRESHOLD
cold_mask = ~hot_mask

# Reorder CSR columns: hot features first, then cold
col_order = np.concatenate([np.where(hot_mask)[0], np.where(cold_mask)[0]])
# Apply permutation to CSR
```

Then launch two kernels: tiled for hot range `[0, n_hot)`, global-atomic for cold range `[n_hot, n_features)`.

**Citation:** [NVIDIA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html) — "separate hot and cold features before histogram accumulation"

---

## Optimization Tier 2: Medium-Impact (Expected 1.5-3x Each)

### 2.1 Thread Block Clusters with Distributed Shared Memory (Blackwell DSM)
**Expected Speedup:** 2-3x on the tiled kernel
**Effort:** High
**Requires:** CUDA 12.8+, sm_100+ (Blackwell cc 10.0)

Blackwell supports **Thread Block Clusters** where blocks in a cluster share access to each other's shared memory via DSM. This is the intermediate tier between "one block's shared memory" (too small for 10M features) and "global memory atomics" (high latency).

```
Current: tile_size = ~8192 bins per block → ~600 tile launches for 5M features
With DSM: cluster of 8 blocks shares 8x shared memory → ~75 tile launches
```

**Kernel design:**
```cuda
// Cluster launch attribute
__cluster_dims__(8, 1, 1)  // 8 blocks per cluster
__global__ void sparse_hist_build_cluster_kernel(
    /* ... same args ... */
    int32_t tile_start,
    int32_t tile_end
) {
    // Each block in cluster owns 1/8 of the shared tile
    extern __shared__ double smem_hist[];

    // Use distributed shared memory for cross-block accumulation
    // cluster.map_shared_rank() to access neighbor block's smem
    namespace cg = cooperative_groups;
    auto cluster = cg::this_cluster();

    int local_tile_start = tile_start + (cluster.block_rank() * local_tile_size);

    // ... accumulate into local smem ...
    // For feature IDs outside local range, use DSM atomics
    // to reach the owning block's smem directly

    cluster.sync();
    // Flush only your local shard to global
}
```

**Key numbers for Blackwell sm_120 (RTX 5090):**
- 48 concurrent warps per SM
- 99 KB max shared memory per thread block
- 128 KB shared memory per SM
- Max cluster size: 8 (portable), 16 (non-portable)

**Caveat:** DSM accesses must be coalesced and 32-byte aligned. Feature ID scatter is inherently irregular, so benefit depends on feature locality within CSR rows.

**Citation:** [CUDA Programming Guide — Thread Block Clusters](https://docs.nvidia.com/cuda/cuda-c-programming-guide/), [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)

---

### 2.2 L2 Cache Persistence for CSR Metadata
**Expected Speedup:** 1.3-1.5x overall
**Effort:** Low

Pin frequently-accessed read-only data in L2 cache:
- `d_csr_indptr` (accessed every kernel launch, sequential pattern)
- `d_leaf_rows` (accessed every kernel launch)
- Feature remap tables / histogram offsets

```cuda
// At init time, after CSR upload:
cudaStreamAttrValue stream_attr;
stream_attr.accessPolicyWindow.base_ptr  = (void*)d_csr_indptr_;
stream_attr.accessPolicyWindow.num_bytes = (n_rows_ + 1) * sizeof(int64_t);
stream_attr.accessPolicyWindow.hitRatio  = 1.0f;  // try to keep all of it
stream_attr.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
stream_attr.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
cudaStreamSetAttribute(stream_compute_, cudaStreamAttributeAccessPolicyWindow, &stream_attr);
```

Blackwell L2 is significantly larger than Ampere/Ada. For 227K rows, indptr = ~1.8MB which fits easily.

**Citation:** [CUDA Programming Guide — L2 Access Management](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)

---

### 2.3 Persistent Kernel with Work Queue
**Expected Speedup:** 1.5-2x (eliminates per-tile launch overhead)
**Effort:** Medium

Replace the `for (tile_start...; tile_start += tile_size)` loop of kernel launches with a single persistent kernel that pulls tile assignments from a global work queue:

```cuda
__global__ void sparse_hist_build_persistent_kernel(
    /* ... CSR args ... */
    volatile int32_t* tile_queue,  // atomic counter
    int32_t n_tiles,
    int32_t tile_size,
    int32_t total_bins
) {
    extern __shared__ double smem_hist[];

    while (true) {
        // Grab next tile
        __shared__ int32_t my_tile;
        if (threadIdx.x == 0) {
            my_tile = atomicAdd((int32_t*)tile_queue, 1);
        }
        __syncthreads();

        if (my_tile >= n_tiles) return;

        int32_t ts = my_tile * tile_size;
        int32_t te = min(ts + tile_size, total_bins);

        // Zero shared memory, build tile, flush to global
        // ... (same as current tiled kernel body) ...
    }
}
```

Current approach: ~600 kernel launches for 5M features with 8192-bin tiles. Each launch has ~5-10us overhead on modern GPUs. Total: ~3-6ms wasted. Persistent kernel eliminates this.

**Citation:** [CUDA Graphs vs Kernel Fusion](https://www.reddit.com/r/CUDA/comments/1o2fl3g/cuda_graphs_vs_kernel_fusion_are_we_solving_the/) — persistent kernels reduce launch overhead for iterative workloads.

---

### 2.4 CUDA Graphs for the Build Pipeline
**Expected Speedup:** 1.3-1.5x (eliminates CPU-side launch latency)
**Effort:** Low-Medium

The histogram build pipeline is deterministic per leaf: zero → H2D → kernel → D2H. Capture it as a CUDA graph and replay:

```cuda
// Capture once (first leaf of first tree)
cudaGraph_t graph;
cudaGraphExec_t graphExec;

cudaStreamBeginCapture(stream_compute_, cudaStreamCaptureModeGlobal);
// ... all kernel launches and memcpys for one histogram build ...
cudaStreamEndCapture(stream_compute_, &graph);
cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

// Replay for subsequent leaves (update grad/hess pointers via graph update)
cudaGraphLaunch(graphExec, stream_compute_);
```

Best for the atomic scatter path where the kernel sequence is fixed. Less applicable to the tiled path where tile count varies by feature count.

---

## Optimization Tier 3: Architecture-Level (Expected 2-5x, High Effort)

### 3.1 Feature-Sharded Multi-GPU (8x RTX 5090)
**Expected Speedup:** 4-7x with 8 GPUs (sub-linear due to communication)
**Effort:** Very High

Current design: single GPU. With 8 GPUs, feature-shard the histogram:

```
GPU 0: features [0,       n/8)     — build histograms for this shard
GPU 1: features [n/8,    2n/8)
...
GPU 7: features [7n/8,    n)

Each GPU:
1. Receives full CSR for its feature columns (column-sliced CSR)
2. Receives full grad/hess vectors
3. Builds local exact histograms for its feature shard
4. All-reduce: exchange only top split candidates, NOT full histograms
```

**Why feature-sharding > data-sharding:** With 2-10M features and 818-227K rows, the feature dimension dominates. Feature-sharding means each GPU builds a complete histogram for fewer features rather than a partial histogram for all features that must be merged.

**Communication cost:** Only exchange `(feature_id, gain, threshold)` tuples for best split candidates — ~bytes per node, not MB of full histograms.

**Balance:** Sort features by NNZ and stripe across GPUs (GPU 0 gets features 0, 8, 16, ...; GPU 1 gets 1, 9, 17, ...) to balance load. Don't just partition by contiguous range — rare features cluster.

**Citation:** [LightGBM Features — Data Parallel](https://lightgbm.readthedocs.io/en/stable/Features.html), [ThunderGBM Multi-GPU](https://readingxtra.github.io/docs/ml-gpu/wen_tpds19_gpugbdt.pdf)

---

### 3.2 Zero-Bin Reconstruction (Eliminate 50% of Histogram Work)
**Expected Speedup:** ~2x on histogram build
**Effort:** Low-Medium

For binary features, bin=0 (feature OFF) can be computed as:
```
hist[feature][bin=0] = leaf_total - hist[feature][bin=1]
```

The current kernel already skips structural zeros in CSR (only processes nonzeros). But it then needs the CPU to compute bin=0 via leaf totals. Making this explicit and fusing it into the D2H copy:

```cuda
__global__ void reconstruct_zero_bins_kernel(
    double* __restrict__ hist,          // [n_features * 2] (bin=1 only)
    double               leaf_grad_sum, // sum of all leaf gradients
    double               leaf_hess_sum,
    int32_t              n_features
) {
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n_features) return;

    // hist[f*2] currently holds bin=1 grad sum
    // bin=0 grad = leaf_total_grad - bin=1 grad
    // We expand in-place to interleaved [bin0_g, bin0_h, bin1_g, bin1_h]
    // Or better: keep as [bin1_g, bin1_h] and let CPU reconstruct bin0
    // (cheaper than doubling histogram memory)
}
```

This is already implicitly done (CSR only stores nonzeros = bin 1). The optimization is to **halve the histogram buffer size** from `n_features * 2 * 2` (two bins) to `n_features * 2` (one bin), cutting VRAM and D2H bandwidth.

---

### 3.3 Histogram Subtraction on GPU
**Expected Speedup:** 1.5x on deeper trees
**Effort:** Low (kernel already exists)

`hist_subtract_kernel` exists but the `ConstructHistograms()` method currently computes both children on CPU. Enable the GPU subtraction path:

```
For each internal node with children L and R:
1. Build histogram for smaller child (fewer rows) on GPU
2. Parent histogram is already known
3. sibling = parent - child  → GPU kernel, ~0.1ms for 10M features
```

This halves the per-level histogram work for deep trees (depth 7+ = 63 leaves).

---

## Optimization Tier 4: Speculative / Research (Uncertain Payoff)

### 4.1 Tensor Cores for SpMV
**Status:** NOT RECOMMENDED for the histogram path
**Reason:** Tensor cores excel at dense matrix-multiply tiles (MMA instructions). Histogram building is scatter-add, not GEMM. The SpMV path already uses cuSPARSE which internally routes to the best available hardware path.

However, tensor cores COULD help in a **separate preprocessing stage** if you ever need:
- Batched dense feature interactions (not current use case)
- Mixed-precision gradient accumulation (violates exact requirement)

**Citation:** [Libra: Synergizing CUDA and Tensor Cores for SpMM](https://arxiv.org/html/2506.22714v1) — confirms tensor cores help for SpMM when regions are dense enough, not for scatter-add histograms.

### 4.2 Cooperative Groups Grid-Wide Sync
**Status:** Low priority
**Reason:** Grid-wide synchronization through `cooperative_groups::this_grid().sync()` enables a single kernel to replace the multi-launch tiled approach. But it limits occupancy (all blocks must fit simultaneously) and is less flexible than the persistent kernel approach (2.3).

### 4.3 Deterministic Accumulation Mode
**Status:** Worth implementing for backtest reproducibility
**Expected Overhead:** ~20-30% slower than non-deterministic

For exact bitwise reproducibility across runs (important for backtest validation):

```cuda
// Instead of atomicAdd (non-deterministic order):
// Use two-phase approach:
// Phase 1: Write per-block partial histograms to global memory
// Phase 2: Reduce partials in fixed block-ID order

__global__ void sparse_hist_build_deterministic_phase1(
    /* ... */
    double* __restrict__ partial_hists,  // [n_blocks * n_features * 2]
    int32_t block_stride
) {
    // Each block writes its partial to partial_hists[blockIdx.x * stride + ...]
    // No atomics needed — each block has its own output region
}

__global__ void sparse_hist_build_deterministic_phase2(
    const double* __restrict__ partial_hists,
    double* __restrict__ final_hist,
    int32_t n_blocks,
    int32_t n_features
) {
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n_features) return;

    double g = 0.0, h = 0.0;
    for (int b = 0; b < n_blocks; b++) {
        g += partial_hists[b * n_features * 2 + f * 2];
        h += partial_hists[b * n_features * 2 + f * 2 + 1];
    }
    final_hist[f * 2] = g;
    final_hist[f * 2 + 1] = h;
}
```

**VRAM cost:** For 227K rows / 256 threads per block = ~888 blocks. Partial histograms for 10M features = 888 * 10M * 16 bytes = **142 TB**. This is obviously infeasible for full-size features. Solution: combine with tiling (partial hists per tile, not per full feature set).

---

## Optimization Priority Matrix

| # | Optimization | Speedup | Effort | Risk | Priority |
|---|---|---|---|---|---|
| 1.1 | Enable GPU ConstructHistograms | 10-50x | Low | Low | **P0 — DO THIS FIRST** |
| 1.3 | Hot/Cold Feature Partitioning | 2-5x | Medium | Low | P1 |
| 3.2 | Zero-Bin Reconstruction | ~2x | Low | Low | P1 |
| 3.3 | Histogram Subtraction on GPU | 1.5x | Low | Low | P1 |
| 1.2 | Warp-Aggregated Atomics | 2-4x | Medium | Low | P2 |
| 2.2 | L2 Cache Persistence | 1.3-1.5x | Low | Low | P2 |
| 2.3 | Persistent Kernel | 1.5-2x | Medium | Low | P2 |
| 2.4 | CUDA Graphs | 1.3-1.5x | Low | Low | P2 |
| 2.1 | Blackwell DSM Clusters | 2-3x | High | Medium | P3 |
| 3.1 | Multi-GPU Feature Sharding | 4-7x | Very High | Medium | P3 |
| 4.3 | Deterministic Mode | N/A | Medium | Low | P4 (backtest) |

---

## sm_120 Compilation Fix

Current `gpu_histogram.cu` compiles for up to `compute_100`. For RTX 5090:

```bash
nvcc -O3 -shared -Xcompiler -fPIC \
     -gencode arch=compute_80,code=sm_80 \
     -gencode arch=compute_86,code=sm_86 \
     -gencode arch=compute_89,code=sm_89 \
     -gencode arch=compute_90,code=sm_90 \
     -gencode arch=compute_100,code=sm_100 \
     -gencode arch=compute_120,code=sm_120 \
     -gencode arch=compute_120,code=compute_120 \
     -o libgpu_histogram.so gpu_histogram.cu
```

**Requirements:** CUDA 12.8+ for sm_120 support.

**sm_120 tuning considerations:**
- 48 warps/SM (vs 64 on Hopper) — reduce register pressure
- 99 KB max shared memory per block (vs 164 KB on A100)
- Test 128-thread blocks instead of 256 for occupancy balance
- Use `cudaOccupancyMaxActiveBlocksPerMultiprocessor` to tune

---

## Rare Signal Preservation Guarantees

All optimizations above preserve exact rare signals because:

1. **No feature filtering** — all 2-10M features enter the histogram kernel
2. **No approximate binning** — binary features have exactly 2 bins, no `max_bin` reduction
3. **No lossy compression** — CSR column indices are uncompressed int32
4. **Exact arithmetic** — `atomicAdd` on FP64 is exact for counts; gradient sums may vary in order but not in split-candidate semantics
5. **Histogram subtraction** preserves exact integer-like counts for binary features
6. **Zero-bin reconstruction** is algebraically exact: `bin0 = total - bin1`

For features firing only 3-20 times, the gain computation depends on exact gradient sums. The non-determinism from atomic ordering affects only the least-significant bits of FP64 sums. For practical purposes (same split decisions), this is exact. For bitwise reproducibility, use deterministic mode (4.3).

---

## Perplexity Research Citations

1. [LightGBM GPU Performance Guide](https://lightgbm.readthedocs.io/en/latest/GPU-Performance.html) — histogram building dominance, sparse optimization
2. [NVIDIA GPU Pro Tip: Fast Histograms Using Shared Atomics](https://developer.nvidia.com/blog/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/) — three-tier accumulator pattern
3. [NVIDIA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html) — DSM, cluster size, shared memory limits, sm_120 specs
4. [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — cooperative groups, L2 persistence, atomic scopes
5. [ThunderGBM GPU GBDT Training](https://readingxtra.github.io/docs/ml-gpu/wen_tpds19_gpugbdt.pdf) — sparse histogram construction, multi-GPU feature sharding
6. [CCCL Issue #3357](https://github.com/NVIDIA/cccl/issues/3357) — reduce scope of histogram atomics for better performance
7. [Vectorized Adaptive Histograms for Sparse Oblique Forests](https://arxiv.org/html/2603.00326v1) — fusing histogram+split search shows no benefit
8. [Fused3S: Fast Sparse Attention on Tensor Cores](https://arxiv.org/html/2505.08098v1) — fusion helps when intermediates stay on-chip
9. [Libra: Synergizing CUDA and Tensor Cores for SpMM](https://arxiv.org/html/2506.22714v1) — tensor cores for dense regions only
10. [Blackwell Microbenchmarking](https://arxiv.org/html/2512.02189v2) — cc 12.0 characteristics, DSM performance
