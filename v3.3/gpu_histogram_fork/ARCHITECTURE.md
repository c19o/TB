# GPU Histogram Co-Processor Fork of LightGBM
# Technical Architecture Document
# v3.3 — 2026-03-27

## Matrix Thesis Constraints (NON-NEGOTIABLE)

- NO feature filtering/removal — the model decides via tree splits
- NO row subsampling — ALL rows used (5,733 on 1d, 227K on 15m)
- Sparse binary cross features ARE the edge (gematria, astrology, numerology, space weather x TA)
- EFB bundling preserves all features in compressed bundles (~23K bundles for 6M features)
- feature_pre_filter=False always
- max_bin=255 for maximum EFB compression (254 features per bundle)
- Binary cross features get num_bin=2 regardless of max_bin

---

## 1. Problem Statement

LightGBM trains CPU-only on sparse CSR with EFB. Histogram building is the bottleneck:
- 6M raw features -> ~18K-23K EFB bundles (max_bin=255)
- Each tree level builds histograms: for every leaf node, scan all rows, accumulate
  gradient/hessian into per-bundle-bin counters
- num_leaves=63 -> up to 62 histogram builds per tree (minus subtraction trick)
- 800 boosting rounds x 3 classes x ~31 histogram builds/tree = ~74,400 histogram ops

The histogram operation is: for each row in this leaf, iterate its nonzero CSR entries,
map feature index to EFB bundle+bin, atomically add gradient and hessian to that bin's
accumulator. This is embarrassingly parallel across rows — a perfect GPU workload.

---

## 2. Integration Point

Replace ONLY the histogram building step. Everything else stays on CPU.

```
LightGBM C++ Call Graph (simplified):
  GBDT::Train()
    -> Boosting::TrainOneIter()
      -> Tree::Train()
        -> SerialTreeLearner::Train()
          -> SerialTreeLearner::ConstructHistograms()   <-- GPU REPLACES THIS
          -> SerialTreeLearner::FindBestSplits()        <-- stays CPU
          -> SerialTreeLearner::Split()                 <-- stays CPU
```

### What the GPU replaces:
- `SerialTreeLearner::ConstructHistograms()` in `src/treelearner/serial_tree_learner.cpp`
- Specifically the inner loop that iterates CSR rows and accumulates into histogram bins

### What stays on CPU (and why):
- **EFB bundling** (`src/io/dataset.cpp`): one-time O(features^2) conflict graph. Runs once at
  Dataset construction. Too irregular for GPU, and only runs once.
- **Split finding** (`src/treelearner/feature_histogram.hpp`): scans histogram bins for best
  gain. Only ~18K-23K bins per feature — trivial on CPU.
- **Tree structure updates**: pointer-based tree construction, not parallelizable.
- **Row partitioning**: updating which rows belong to which leaf after a split.
- **Gradient computation**: objective function computes gradients. Small array (N x 3 classes).
- **EFB bin mapping**: pre-computed at Dataset construction, stored as lookup table.

---

## 3. Data Flow

```
                         ONE-TIME INIT
  +=============================================================+
  |                                                             |
  |  CPU: load CSR (scipy) --> EFB bundle (LightGBM Dataset)   |
  |       3M raw features --> ~18K-23K EFB bundles              |
  |       CSR stored as: indptr[], indices[], data[]            |
  |       data[] = pre-computed EFB bundle bin index (uint8)    |
  |                                                             |
  |  GPU: cudaMemcpy H2D (async, stream 0):                    |
  |       indptr   int64  [N+1]           ~1.8 MB  (1d)        |
  |       indices  int32  [nnz]           ~4.0 GB  (1d, 6M)    |
  |       data     uint8  [nnz]           ~1.0 GB  (1d, 6M)    |
  |       bin_offsets  int32  [n_bundles]  ~72 KB               |
  |                                                             |
  |  Total GPU-resident CSR: 2-40 GB depending on TF           |
  |    1w:  818 rows x 2.2M features  ->  ~2 GB                |
  |    1d:  5733 rows x 6M features   ->  ~5 GB                |
  |    4h:  23K rows x 4M features    ->  ~12 GB               |
  |    1h:  100K rows x 10M features  ->  ~25 GB               |
  |    15m: 227K rows x 10M features  ->  ~40 GB               |
  +=============================================================+


                   PER BOOSTING ROUND (x800)
  +=============================================================+
  |                                                             |
  |  CPU: compute gradients for all N rows                      |
  |       grad[]  float32 [N x 3]    ~69 KB  (1d)              |
  |       hess[]  float32 [N x 3]    ~69 KB  (1d)              |
  |                                                             |
  |  H2D: cudaMemcpyAsync (stream 0, pinned memory)            |
  |       grad + hess = ~138 KB per round (1d)                  |
  |       Latency: <0.1 ms (PCIe 4.0 x16 = 25 GB/s)           |
  |                                                             |
  +=============================================================+


               PER TREE NODE (up to 62 per tree)
  +=============================================================+
  |                                                             |
  |  CPU -> GPU: leaf row indices OR split decision             |
  |    Option A: send row_indices[] for this leaf  (~11 KB avg) |
  |    Option B: send split (feature, threshold)   (12 bytes)   |
  |              GPU maintains its own partition bitmap          |
  |                                                             |
  |  GPU KERNEL: histogram accumulation                         |
  |    For each row in leaf:                                    |
  |      For each nonzero in CSR row:                           |
  |        bundle_bin = data[offset]    // pre-computed by EFB  |
  |        atomicAdd(hist[bundle_bin].grad, grad[row])          |
  |        atomicAdd(hist[bundle_bin].hess, hess[row])          |
  |                                                             |
  |  GPU -> CPU: histogram buffer                               |
  |    hist[]  float64 [n_bundles x max_bin x 2]                |
  |    = 18K bundles x 255 bins x 16 bytes = ~73 MB (worst)    |
  |    Actual: most bundles have 2 bins -> ~47 MB typical       |
  |    D2H latency: ~2 ms (PCIe 4.0)                           |
  |                                                             |
  |  CPU: FindBestSplit on histogram -> (feature, threshold,    |
  |        gain, left_count, right_count)                       |
  |                                                             |
  +=============================================================+


  FULL ROUND TIMELINE (1d, 6M features, num_leaves=63):
  +----+--------+----------+---------+--------+----------+-----
  | H2D grad   | kernel   | D2H hist| split  | H2D part | ...
  | 0.1ms      | 1-5ms    | 2ms     | 0.5ms  | <0.1ms   |
  +----+--------+----------+---------+--------+----------+-----
       ^                   ^
       |-- overlapped via streams --|
```

---

## 4. CUDA Kernel Architecture

### 4.1 Kernel Signature

```cuda
__global__ void histogram_build(
    // CSR matrix (GPU-resident, read-only)
    const int64_t* __restrict__ indptr,     // [N+1]
    const int32_t* __restrict__ indices,    // [nnz] - feature index
    const uint8_t* __restrict__ data,       // [nnz] - EFB bundle bin index
    // Gradients (updated per round)
    const float*   __restrict__ grad,       // [N x num_class]
    const float*   __restrict__ hess,       // [N x num_class]
    // Leaf membership
    const int32_t* __restrict__ leaf_rows,  // [n_leaf_rows]
    int32_t n_leaf_rows,
    // Bundle mapping
    const int32_t* __restrict__ bundle_offsets, // [n_bundles+1]
    int32_t n_bundles,
    int32_t num_class,
    // Output histogram
    double* __restrict__ hist_buf           // [n_bundles * max_bin * 2 * num_class]
);
```

### 4.2 Row-Parallel Execution

Each CUDA thread processes ONE leaf row's CSR nonzeros:

```
Thread i:
  row = leaf_rows[i]
  g = grad[row * num_class + class_idx]
  h = hess[row * num_class + class_idx]
  for offset in range(indptr[row], indptr[row+1]):
      bundle_bin = data[offset]
      // bundle_bin encodes both which bundle AND which bin within it
      atomicAdd(&hist_buf[bundle_bin * 2 + 0], (double)g)
      atomicAdd(&hist_buf[bundle_bin * 2 + 1], (double)h)
```

### 4.3 Bundle Tiling for Shared Memory

The histogram buffer is too large for shared memory on any current GPU.
Solution: tile over bundles.

```
Grid dimensions:  (row_blocks, bundle_tiles)
Block dimensions: (256, 1, 1)   // 256 threads per block

Each block:
  1. Zero shared memory histogram for this tile's bundles
  2. Each thread processes its row, but ONLY accumulates into
     bundles in range [tile_start, tile_end)
  3. __syncthreads()
  4. Flush shared memory histogram to global memory (atomicAdd)
```

**Adaptive tile sizing per GPU:**

```
Shared memory per SM:
  RTX 3090:  100 KB  -> 1752 bins/tile  (100KB / (2 * 8 * 3 + padding))
  A40:       100 KB  -> 1752 bins/tile
  A100:      164 KB  -> 3413 bins/tile   (configurable shared mem)
  H100:      228 KB  -> 4747 bins/tile
  B200:      228 KB  -> 4747 bins/tile

Bins per tile = shared_mem_bytes / (2 doubles * num_class)
             = shared_mem_bytes / (2 * 8 * 3)
             = shared_mem_bytes / 48

Tile count for 18K bundles x 255 max_bin:
  Actual total bins ~ 18K * 2 (binary avg) = 36K bins
  RTX 3090: ceil(36K / 1752) = 21 tiles
  A100:     ceil(36K / 3413) = 11 tiles
  H100:     ceil(36K / 4747) = 8 tiles
```

### 4.4 Binary Feature Optimization

Cross features are binary (0/1). After EFB bundling:
- Each bundle of K binary features has K+1 possible bin values (0..K)
- data[] stores the pre-computed bundle bin index (0 = all features OFF in bundle)
- **Skip bin 0**: if data[offset] == 0, the row contributes to no feature in this bundle.
  The bin-0 histogram is computed by subtraction (total - sum of other bins).
- This eliminates ~95% of atomicAdd calls for sparse data (most entries are 0).

```cuda
// Inner loop optimization for binary crosses:
for (int64_t j = indptr[row]; j < indptr[row+1]; j++) {
    uint8_t bin = data[j];
    if (bin == 0) continue;  // structural zero in this bundle -> skip
    int idx = bin * 2;       // grad at even, hess at odd
    atomicAdd(&smem_hist[idx + 0], (double)g);
    atomicAdd(&smem_hist[idx + 1], (double)h);
}
```

### 4.5 3-Class Multiclass Handling

LightGBM multiclass trains separate trees per class. For num_class=3:
- Round i trains tree for class (i % 3)
- Gradients are indexed: `grad[row * 3 + class_idx]`
- Histogram accumulates grad/hess for ONE class at a time
- No change to kernel structure, just different gradient slice per round

---

## 5. Memory Management

### 5.1 CUDA Stream Architecture

```
Stream 0 (H2D):    grad/hess upload, partition updates
Stream 1 (Compute): histogram kernel execution
Stream 2 (D2H):    histogram download to CPU

Timeline per node:
  Stream 0: |--H2D grad--|                    |--H2D partition--|
  Stream 1:               |---kernel launch---|
  Stream 2:                                    |---D2H hist---|
                                                              |--CPU split--|
```

### 5.2 Double-Buffered Pinned Gradients

```
pinned_grad_A[N * 3]   // CPU writes round R gradients here
pinned_grad_B[N * 3]   // GPU reads round R-1 gradients from here
// Swap A<->B each round. Zero copy overhead.
```

### 5.3 GPU-Side Row Partitioning

Instead of sending full row index arrays per leaf:
1. GPU maintains a `leaf_id[N]` array (int8, one byte per row)
2. CPU sends split decision: `(leaf_to_split, feature_idx, threshold)` = 12 bytes
3. GPU kernel updates `leaf_id[]` in-place: rows in old leaf get reassigned

```
Benefits:
  - 12 bytes per split vs ~11 KB average row index array
  - GPU already has the CSR data to evaluate the split condition
  - leaf_id[N] = 227 KB for 15m (trivial)
```

### 5.4 Histogram Buffer Pool

```
Pre-allocated at init:
  max_leaves = 63 (num_leaves from config)
  hist_size = n_total_bins * 2 * sizeof(double) * num_class
            = 36K bins * 2 * 8 * 3 = ~1.7 MB per leaf (typical)
            = 4.6M bins * 2 * 8 * 3 = ~220 MB per leaf (worst, all 255 bins)

  Pool: 63 buffers x ~1.7 MB = ~107 MB typical
        63 buffers x ~220 MB = ~13.9 GB worst case

  Actual for our data (18K bundles, avg 2 bins):
    Per buffer: 18K * 2 * 2 * 8 * 3 = ~1.7 MB
    Pool: 63 * 1.7 MB = ~107 MB
```

### 5.5 Total GPU Memory Budget

```
                        1w       1d       4h       1h       15m
  CSR resident:         2 GB     5 GB     12 GB    25 GB    40 GB
  Histogram pool:       0.1 GB   0.1 GB   0.1 GB   0.1 GB   0.1 GB
  Gradients (x2 buf):   <1 MB    <1 MB    <1 MB    2 MB     4 MB
  leaf_id array:        <1 MB    <1 MB    <1 MB    <1 MB    <1 MB
  Kernel overhead:      ~0.5 GB  ~0.5 GB  ~0.5 GB  ~0.5 GB  ~0.5 GB
  ─────────────────────────────────────────────────────────────────
  TOTAL:                ~3 GB    ~6 GB    ~13 GB   ~26 GB   ~41 GB

  GPU VRAM Available:
    RTX 3090:  24 GB  -> 1w, 1d OK. 4h tight. 1h/15m NO.
    A40:       48 GB  -> 1w-4h OK. 1h tight. 15m NO.
    A100 80GB: 80 GB  -> All TFs OK.
    H100:      80 GB  -> All TFs OK.
    B200:     192 GB  -> All TFs OK with room to spare.
```

### 5.6 Memory Layout Diagram

```
GPU GLOBAL MEMORY
+================================================================+
|                                                                |
|  CSR REGION (read-only, allocated once at init)                |
|  +----------------------------------------------------------+ |
|  | indptr   [N+1]      int64    | contiguous                | |
|  | indices  [nnz]      int32    | contiguous                | |
|  | data     [nnz]      uint8    | contiguous                | |
|  | bundle_offsets [B+1] int32   | contiguous                | |
|  +----------------------------------------------------------+ |
|                                                                |
|  GRADIENT REGION (double-buffered, pinned host mirror)         |
|  +----------------------------------------------------------+ |
|  | grad_buf_A [N * num_class]   float32                      | |
|  | grad_buf_B [N * num_class]   float32                      | |
|  | hess_buf_A [N * num_class]   float32                      | |
|  | hess_buf_B [N * num_class]   float32                      | |
|  +----------------------------------------------------------+ |
|                                                                |
|  PARTITION REGION                                              |
|  +----------------------------------------------------------+ |
|  | leaf_id  [N]         int8    | updated per split          | |
|  | leaf_count [max_leaves] int32| row counts per leaf        | |
|  +----------------------------------------------------------+ |
|                                                                |
|  HISTOGRAM POOL (ring buffer, reused across tree levels)       |
|  +----------------------------------------------------------+ |
|  | hist[0]  [total_bins * 2 * C]  float64  | leaf 0         | |
|  | hist[1]  [total_bins * 2 * C]  float64  | leaf 1         | |
|  | ...                                                       | |
|  | hist[62] [total_bins * 2 * C]  float64  | leaf 62        | |
|  +----------------------------------------------------------+ |
|                                                                |
|  (no dynamic allocation after init — all sizes known)          |
|                                                                |
+================================================================+

HOST PINNED MEMORY
+================================================================+
|  pinned_grad_A [N * C]  float32  |  CPU writes, DMA to GPU    |
|  pinned_grad_B [N * C]  float32  |  swap each round           |
|  pinned_hess_A [N * C]  float32  |                            |
|  pinned_hess_B [N * C]  float32  |                            |
|  pinned_hist   [total_bins * 2 * C]  float64  |  DMA from GPU |
+================================================================+
```

---

## 6. Histogram Subtraction Trick

LightGBM's key optimization: only build the histogram for the SMALLER child.
The larger child's histogram = parent - smaller child.

### 6.1 Implementation

```
After split of leaf P into children L (larger) and S (smaller):
  1. GPU builds hist[S] by scanning rows in S   (fast: fewer rows)
  2. CPU computes hist[L] = hist[P] - hist[S]   (vectorized subtract)
  3. hist[P] buffer reassigned to L              (zero-copy pointer swap)

Cost savings:
  Without trick: scan ALL rows per level
  With trick:    scan only smaller-half rows per level
  Speedup:       ~2x per tree level
```

### 6.2 Parent Buffer Reassignment (Zero-Copy)

```
Before split:         After split:
  hist_pool[P] -> parent hist    hist_pool[P] -> (now L's hist, computed by subtraction)
                                 hist_pool[S] -> S's hist (built by GPU)
                                 hist_pool[P] is NOT freed — it IS L's hist

The "larger child inherits parent buffer" is a pointer swap, not a copy.
No memcpy, no allocation, no GPU involvement.
```

### 6.3 Ring Buffer with Depth Bounding

```
Tree depth d has at most 2^d leaves. num_leaves=63 -> max depth ~6.
At any tree level, we need:
  - Current level's histograms (up to 2^d)
  - Parent level's histograms (for subtraction at next split)
  - Working buffer for the smaller child being built

hist_pool depth bound = max_trick_depth = 6
  -> max 2 * 2^6 = 128 buffers needed
  -> but num_leaves=63 caps it at 63 buffers total

Ring buffer reclaims parent buffers after both children are computed.
```

---

## 7. GPU-Side Row Partition Update

After CPU finds the best split, it sends the split decision to the GPU
so the GPU can update its `leaf_id[]` array without receiving full row lists.

```
Split message from CPU (12 bytes):
  struct SplitDecision {
      int32_t leaf_to_split;    // which leaf is being split
      int32_t feature_idx;      // which feature to split on
      float   threshold;        // split threshold
  };

GPU partition update kernel:
  __global__ void update_partition(
      const int64_t* indptr, const int32_t* indices, const uint8_t* data,
      int8_t* leaf_id, int32_t* leaf_count,
      int32_t old_leaf, int32_t new_leaf_left, int32_t new_leaf_right,
      int32_t split_feature, float split_threshold
  ) {
      int row = blockIdx.x * blockDim.x + threadIdx.x;
      if (row >= N) return;
      if (leaf_id[row] != old_leaf) return;

      // Find split_feature's value for this row by scanning CSR
      float val = 0.0f;  // default = left (missing goes left in LightGBM default)
      for (int64_t j = indptr[row]; j < indptr[row+1]; j++) {
          if (indices[j] == split_feature) {
              val = (float)data[j];
              break;
          }
      }
      leaf_id[row] = (val <= split_threshold) ? new_leaf_left : new_leaf_right;
  }

Cost: O(N * avg_nnz_per_row) worst case, but only rows in the split leaf
      are checked (early exit via leaf_id check).
```

---

## 8. GPU Compatibility

### 8.1 Auto-Detection at Init

```python
def detect_gpu_config():
    """Auto-detect GPU capabilities for histogram co-processor."""
    import ctypes
    props = cuda.cudaGetDeviceProperties(0)
    return {
        'compute_capability': (props.major, props.minor),
        'sm_count': props.multiProcessorCount,
        'shared_mem_per_sm': props.sharedMemPerMultiprocessor,
        'vram_bytes': props.totalGlobalMem,
        'name': props.name,
        # Derived
        'bins_per_tile': props.sharedMemPerMultiprocessor // 48,  # 48 = 2*8*3
        'max_csr_bytes': int(props.totalGlobalMem * 0.70),  # 70% for CSR
    }
```

### 8.2 VRAM Threshold and CPU Fallback

```
If CSR size > 70% of VRAM:
  -> Fall back to CPU histogram (standard LightGBM path)
  -> Log warning with CSR size vs VRAM
  -> No crash, no degradation of results — just slower

The 70% threshold reserves:
  30% for histogram pool + gradients + partition array + CUDA overhead
```

### 8.3 Fat Binary Compilation

```cmake
set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90;100")

# sm_80: A100, A30
# sm_86: RTX 3090, RTX 3080, A40
# sm_89: RTX 4090, L40
# sm_90: H100, H200
# sm_100: B200, B100

# Compile with: nvcc -gencode=arch=compute_80,code=sm_80 \
#                    -gencode=arch=compute_86,code=sm_86 \
#                    -gencode=arch=compute_89,code=sm_89 \
#                    -gencode=arch=compute_90,code=sm_90 \
#                    -gencode=arch=compute_100,code=sm_100 \
#                    -gencode=arch=compute_100,code=compute_100
# Last line: PTX for forward compat with future GPUs
```

---

## 9. Phase 1: cuSPARSE SpMV Approach (Recommended First)

Before building a custom CUDA kernel, use cuSPARSE for a simpler implementation
that validates the integration and data flow.

### 9.1 Key Insight: Binary Histogram = Sparse Matrix-Vector Product

For binary cross features (our dominant case), the histogram for one leaf is:

```
hist_grad[bundle_bin] = sum over (rows in leaf where feature has that bin) of grad[row]

Equivalently:
  Let M = CSR matrix of EFB-encoded features, shape (N, total_bins)
      M[row, bin] = 1 if row has that bin value, 0 otherwise
  Let g = gradient vector, shape (N,)
  Let mask = leaf membership vector, shape (N,), binary

  Then: hist_grad = M.T @ (g * mask)     <-- SpMV!
        hist_hess = M.T @ (h * mask)     <-- SpMV!
```

Two SpMV calls per leaf per class = 2 * 3 = 6 SpMV calls per node.

### 9.2 cuSPARSE SpMM: All Leaves at Once

Better: batch all leaves into a single SpMM call.

```
Let G = gradient matrix, shape (N, n_active_leaves * 2 * num_class)
    Each column = grad or hess for one leaf, masked to that leaf's rows

Then: H = M.T @ G    <-- single SpMM call!
    shape: (total_bins, n_active_leaves * 2 * num_class)

cuSPARSE SpMM on CSR:
  cusparseSpMM(handle,
    CUSPARSE_OPERATION_TRANSPOSE,  // M.T
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    &alpha, M_descr, G_descr,
    &beta, H_descr,
    CUSPARSE_SPMM_ALG_DEFAULT, buffer);
```

### 9.3 Performance Estimate

```
cuSPARSE SpMM throughput (measured on A100):
  CSR with 1B nonzeros, 100 RHS columns: ~15 GB/s effective
  Our data (1d): ~500M nnz, ~6 RHS (3 classes x 2 grad/hess)
  Estimated: 3-15 ms per tree level

  Full tree (31 levels avg): 93-465 ms per tree
  Full training (800 rounds x 3 classes): 223-1116 seconds = 4-19 minutes

  vs CPU baseline (1d): ~45-90 minutes for histogram building
  Speedup: 4-20x on histogram step alone
```

### 9.4 Leaf Batching for Memory

With 63 leaves, the RHS matrix G has 63 * 2 * 3 = 378 columns, each of length N.
That is 378 * N * 4 bytes = 378 * 5733 * 4 = ~8.7 MB for 1d. Trivial.

For 15m (227K rows): 378 * 227K * 4 = ~344 MB. Still fits.

Leaf batching only needed if VRAM-constrained AND many active leaves AND large N.
Threshold: if G matrix > 10% of free VRAM, batch leaves in groups of 16.

### 9.5 Limitations of SpMV Approach

1. **EFB bin encoding**: The CSR stores pre-computed bundle bin indices, not a
   simple binary matrix. Must construct the "indicator matrix" M from the EFB
   encoding. One-time cost at Dataset construction.
2. **Non-binary bins**: Continuous base features have >2 bins. The indicator matrix
   must expand each multi-bin feature into multiple binary columns. Increases M's
   column count but M is still very sparse.
3. **Histogram subtraction**: Must be done AFTER SpMM, on the output H. CPU-side
   subtraction is fast (H is small).

### 9.6 Phase 1 Implementation Plan

```
Step 1: Fork LightGBM, add GPU histogram option
Step 2: At Dataset::Construct(), build indicator matrix M on GPU
Step 3: Replace ConstructHistograms() with cuSPARSE SpMM call
Step 4: Validate histogram outputs match CPU exactly (bit-for-bit)
Step 5: Benchmark on 1w (smallest) then 1d
Step 6: Profile and identify bottlenecks
Step 7: If SpMM is within 2x of theoretical peak -> ship it
         If not -> move to Phase 2 (custom kernel)
```

---

## 10. Phase 2: Custom CUDA Kernel

If cuSPARSE SpMM leaves performance on the table (likely due to the transpose
and multi-class overhead), build the custom kernel described in Section 4.

### 10.1 Advantages over SpMV

- No indicator matrix construction (works directly on EFB-encoded CSR)
- Native multi-class support (single kernel pass for all 3 classes)
- Shared memory tiling tuned for our exact histogram shape
- Skip-zero optimization for sparse binary data (~95% of entries)
- Fused gradient lookup (no separate mask multiply)

### 10.2 Expected Performance

```
Custom kernel performance model (1d, A100):

  Data movement per tree level:
    Read CSR indices+data for leaf rows: ~500 MB (amortized, cached)
    Read gradients for leaf rows: ~69 KB
    Write histogram: ~1.7 MB
    Total effective: ~500 MB

  A100 memory bandwidth: 2 TB/s
  Roofline: 500 MB / 2 TB/s = 0.25 ms per level (memory-bound)

  With 31 levels per tree:
    0.25 * 31 = 7.75 ms per tree
    800 rounds * 3 classes: 18.6 seconds total histogram time

  vs CPU: 45-90 minutes
  Speedup: 145-290x on histogram step

  Caveats:
    - atomicAdd contention reduces effective bandwidth
    - Shared memory tiling adds overhead
    - Realistic: 30-100x speedup (not pure roofline)
```

---

## 11. Multi-GPU (Future — Feature-Parallel)

For 1h/15m where CSR exceeds single-GPU VRAM.

### 11.1 Architecture

```
  GPU 0: owns EFB bundles 0..B/2         GPU 1: owns EFB bundles B/2..B
  +----------------------------+         +----------------------------+
  | CSR columns for bundles    |         | CSR columns for bundles    |
  | 0..B/2 (half the features) |         | B/2..B (other half)        |
  | Full gradient copy         |         | Full gradient copy         |
  | Full leaf_id copy          |         | Full leaf_id copy          |
  +----------------------------+         +----------------------------+
         |                                        |
         | local best split                       | local best split
         | (feature, threshold, gain)             | (feature, threshold, gain)
         | = 12 bytes                             | = 12 bytes
         +------------------+---------------------+
                            |
                      CPU: pick global best split
                      (compare 2 candidates)
                            |
                      Broadcast split decision
                      (12 bytes to each GPU)
```

### 11.2 Communication Cost

```
Per tree node:
  Each GPU sends: (feature_idx: int32, threshold: float32, gain: float64) = 16 bytes
  CPU broadcasts: same 16 bytes back

  Total: 32 bytes per node per GPU
  63 nodes * 2 GPUs * 32 bytes = 4 KB per tree
  800 rounds * 3 classes * 4 KB = 9.6 MB total

  Latency: ~1 us per PCIe transfer of 16 bytes
  Total latency overhead: 63 * 2 * 800 * 3 * 1us = 0.3 seconds

  -> Communication is NEGLIGIBLE. No histogram merging needed because
     each GPU owns disjoint features.
```

### 11.3 Scaling

```
2 GPUs: each holds ~50% of CSR columns. 2x memory, ~1.8x speed.
4 GPUs: each holds ~25% of CSR columns. 4x memory, ~3.2x speed.
8 GPUs: each holds ~12.5%. 8x memory, ~5x speed (diminishing returns from CPU bottleneck).

15m on 4x A100 (80GB each):
  40 GB CSR / 4 = 10 GB per GPU. Fits easily.
  Histogram time: ~18s / 3.2 = ~6 seconds for 800 rounds.
```

---

## 12. Build System Integration

### 12.1 CMake Changes

```cmake
# In LightGBM's CMakeLists.txt, add:
option(USE_GPU_HISTOGRAM "Use GPU co-processor for histogram building" OFF)

if(USE_GPU_HISTOGRAM)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90;100")

    add_library(lgbm_gpu_hist STATIC
        src/treelearner/cuda/histogram_kernel.cu
        src/treelearner/cuda/partition_kernel.cu
        src/treelearner/cuda/gpu_histogram_builder.cpp
    )
    target_link_libraries(lgbm_gpu_hist PRIVATE CUDA::cudart CUDA::cusparse)
    target_link_libraries(lightgbm PRIVATE lgbm_gpu_hist)
    target_compile_definitions(lightgbm PRIVATE USE_GPU_HISTOGRAM)
endif()
```

### 12.2 Python API

```python
# Usage in ml_multi_tf.py:
lgb_params = {
    **V3_LGBM_PARAMS,
    "device": "gpu_histogram",         # new device type
    "gpu_histogram_device_id": 0,      # which GPU
    "gpu_histogram_vram_limit": 0.70,  # fraction of VRAM for CSR
}

# Falls back to CPU if:
#   - GPU not detected
#   - CSR exceeds VRAM limit
#   - CUDA compilation not available
# Controlled by: device="gpu_histogram" (explicit opt-in, never silent)
```

---

## 13. Correctness Validation

### 13.1 Bit-Exact Verification

The GPU histogram MUST produce bit-exact results compared to CPU.
This is achievable because:
- Accumulations use float64 (same as CPU LightGBM)
- Deterministic reduction order (sorted row indices per leaf)
- No floating-point non-associativity issues (sum in same order)

```python
def validate_histograms(cpu_hist, gpu_hist, atol=0.0):
    """Bit-exact comparison. atol=0.0 means EXACT match required."""
    assert np.array_equal(cpu_hist, gpu_hist), \
        f"Histogram mismatch! Max diff: {np.max(np.abs(cpu_hist - gpu_hist))}"
```

### 13.2 Training Equivalence Test

```
1. Train model with device="cpu" on 1w dataset -> model_cpu
2. Train model with device="gpu_histogram" on same 1w dataset -> model_gpu
3. Assert: model_cpu.model_to_string() == model_gpu.model_to_string()
   (byte-for-byte identical model)
```

### 13.3 Determinism Guarantee

LightGBM with `deterministic=True` (our config) requires ordered reduction.
The GPU kernel must process rows in index order within each leaf.
Sort `leaf_rows[]` before kernel launch (or maintain sorted invariant via
partition update kernel).

---

## 14. Expected Impact on Training Time

```
Component breakdown for 1d training (800 rounds, 6M features, 4 CPCV folds):

                        CPU-only        GPU histogram
  Dataset construct:    15 min          15 min (same, CPU)
  Histogram building:   45-90 min       30-90 sec (30-100x faster)
  Split finding:        5 min           5 min (same, CPU)
  Tree structure:       2 min           2 min (same, CPU)
  Gradient compute:     3 min           3 min (same, CPU)
  I/O + overhead:       5 min           5 min (same, CPU)
  ────────────────────────────────────────────────────────
  Total per fold:       75-120 min      30-35 min
  Total 4 folds:        5-8 hours       2-2.5 hours
  + Optuna (200 trials): 50-80 hours   20-25 hours

With Phase 2A (.subset() for CPCV):
  Dataset construct:    15 min (once)   15 min (once, CPU)
  Per fold:             60-105 min      15-20 min
  Total 4 folds:        4-7 hours       1-1.5 hours
  + Optuna:             40-70 hours     10-15 hours

Net speedup: 3-5x on full training pipeline.
The histogram step goes from 60-75% of training time to <5%.
```

---

## 15. File Structure

```
v3.3/gpu_histogram_fork/
  ARCHITECTURE.md           <- this document
  README.md                 <- build/install instructions
  CMakeLists.txt            <- build system additions
  src/
    treelearner/
      cuda/
        histogram_kernel.cu       <- Phase 2 custom CUDA kernel
        partition_kernel.cu       <- GPU-side row partitioning
        gpu_histogram_builder.h   <- C++ interface class
        gpu_histogram_builder.cpp <- C++ host-side logic
        cusparse_histogram.cu     <- Phase 1 cuSPARSE SpMV approach
        gpu_config.h              <- auto-detection, tile sizing
  python/
    gpu_histogram_config.py       <- Python-side GPU detection + params
  tests/
    test_histogram_exact.py       <- bit-exact validation
    test_training_equiv.py        <- full training equivalence
    bench_histogram.py            <- performance benchmarks
```

---

## 16. Implementation Order

```
Phase 1 (cuSPARSE, 2-3 weeks):
  1. Fork LightGBM, add gpu_histogram device type
  2. Build indicator matrix from EFB encoding
  3. Implement cuSPARSE SpMM histogram
  4. Bit-exact validation on 1w
  5. Benchmark on 1w, 1d
  6. Ship if within 2x of custom kernel estimate

Phase 2 (Custom kernel, 2-3 weeks):
  1. Implement row-parallel histogram kernel with shared memory tiling
  2. Add skip-zero optimization for binary features
  3. GPU-side partition update kernel
  4. Double-buffered streaming
  5. Bit-exact validation on all TFs
  6. Adaptive tile sizing per GPU

Phase 3 (Multi-GPU, 1-2 weeks):
  1. Feature-parallel CSR distribution
  2. Per-GPU histogram + local best split
  3. CPU-side global split selection
  4. Split decision broadcast
  5. Validate on 1h with 2+ GPUs

Phase 4 (Production hardening, 1 week):
  1. Fat binary compilation
  2. CPU fallback path
  3. VRAM auto-detection and threshold
  4. Error handling (GPU OOM, driver crash)
  5. Integration with cloud_run_tf.py
  6. Documentation + build instructions
```
