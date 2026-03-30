# Multi-GPU Training Architecture: Breaking the Single-GPU Ceiling

**Date**: 2026-03-30
**Problem**: Final Retrain is single-GPU. 15m TF takes ~6hr on one RTX 5090. 7 GPUs sit idle.
**Target**: Sub-1hr Final Retrain using 8x RTX 5090.

---

## Current Architecture

```
┌─────────────────────────────────────────────────┐
│  ml_multi_tf.py  (orchestrator)                 │
│                                                 │
│  TF-level parallelism:                          │
│    GPU 0 → 1w   (3 GB)                          │
│    GPU 1 → 1d   (6 GB)                          │
│    GPU 2 → 4h   (13 GB)                         │
│    GPU 3 → 1h   (26 GB)                         │
│    GPU 4 → 15m  (41 GB)  ← BOTTLENECK           │
│    GPU 5-7 → idle                               │
│                                                 │
│  Within each TF:                                │
│    CPCV folds → sequential (single GPU)         │
│    Final retrain → single GPU, all data         │
└─────────────────────────────────────────────────┘
```

**15m TF profile** (the bottleneck):
- 227K rows, ~10M sparse binary features
- ~23,600 EFB bundles (254 features/bundle, max_bin=255)
- ~47K total histogram bins (2 effective bins per bundle for binary)
- CPCV: (2,1) = 2 folds, run sequentially
- Final Retrain: single model on all 227K rows
- VRAM: 41 GB (fits in one 5090's 32GB only with streaming)

---

## Three Strategies (Ranked by ROI)

### Strategy 1: Fold-Parallel + Multi-GPU Final Retrain (RECOMMENDED)

**Complexity**: Medium | **Speedup**: 4-6x | **Risk**: Low

This is a two-part approach that gets the most bang for the least engineering:

#### Part A: Fold-Parallel CPCV

Run each CPCV fold on a separate GPU simultaneously. No NCCL needed — folds are independent.

```
CPCV Phase (current: sequential)          CPCV Phase (proposed: parallel)
─────────────────────────────────         ─────────────────────────────────
GPU 0: [fold 0 ████████████████]          GPU 0: [fold 0 ████████████████]
GPU 0: [fold 1 ████████████████]          GPU 1: [fold 1 ████████████████]
                                          GPU 2: [fold 2 ████████████████]
Total: 2x fold time                       GPU 3: [fold 3 ████████████████]
                                          Total: 1x fold time (2-4x speedup)
```

**Implementation** (in `ml_multi_tf.py`):
```python
import multiprocessing as mp

def _train_fold_on_gpu(gpu_id, fold_idx, train_idx, test_idx,
                        X_csr, y, params, result_queue):
    """Train one CPCV fold on a specific GPU."""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    params['gpu_platform_id'] = 0
    params['gpu_device_id'] = 0

    dtrain = lgb.Dataset(X_csr[train_idx], y[train_idx])
    dval = lgb.Dataset(X_csr[test_idx], y[test_idx])
    model = lgb.train(params, dtrain, valid_sets=[dval], ...)

    oof_preds = model.predict(X_csr[test_idx])
    result_queue.put((fold_idx, oof_preds, model))

def run_cpcv_parallel(X_csr, y, cpcv_splits, params, available_gpus):
    """Run all CPCV folds in parallel across GPUs."""
    result_queue = mp.Queue()
    processes = []

    for fold_idx, (train_idx, test_idx) in enumerate(cpcv_splits):
        gpu_id = available_gpus[fold_idx % len(available_gpus)]
        p = mp.Process(target=_train_fold_on_gpu,
                       args=(gpu_id, fold_idx, train_idx, test_idx,
                             X_csr, y, params, result_queue))
        processes.append(p)
        p.start()

    # Collect results
    results = {}
    for _ in processes:
        fold_idx, preds, model = result_queue.get()
        results[fold_idx] = (preds, model)

    for p in processes:
        p.join()

    return results
```

**Constraints**:
- 15m TF uses (2,1) CPCV = only 2 folds → 2 GPUs max for fold-parallel
- Optuna trials can ALSO be parallelized across GPUs (each trial = different GPU)
- Each fold needs ~41GB VRAM → only one fold per 5090 (32GB), need streaming or A100/H100

#### Part B: Data-Parallel Final Retrain via NCCL Histogram AllReduce

The Final Retrain is the real 6hr bottleneck. This requires NCCL-based histogram aggregation.

```
Final Retrain (current)                   Final Retrain (proposed: 8-GPU data-parallel)
───────────────────────                   ──────────────────────────────────────────────
GPU 0: [all 227K rows ████████████ 6hr]   GPU 0: [28K rows ██] ─┐
                                          GPU 1: [28K rows ██]  │ NCCL AllReduce
                                          GPU 2: [28K rows ██]  │ histograms
                                          GPU 3: [28K rows ██]  ├─→ split search
                                          GPU 4: [28K rows ██]  │   → broadcast
                                          GPU 5: [28K rows ██]  │   → next node
                                          GPU 6: [28K rows ██]  │
                                          GPU 7: [28K rows ██] ─┘
                                          Total: ~45min-1hr (6-8x speedup)
```

---

### Strategy 2: Pure NCCL Data-Parallel (All Phases)

**Complexity**: High | **Speedup**: 6-8x | **Risk**: Medium

Use NCCL histogram allreduce for ALL training: CPCV folds AND Final Retrain.
Same mechanism as Strategy 1 Part B, but applied everywhere.

**Advantage**: Maximum GPU utilization at all times.
**Disadvantage**: More complex, NCCL overhead on small TFs (1w/1d) where single-GPU is already fast.

### Strategy 3: Switch Final Retrain to XGBoost Multi-GPU

**Complexity**: Low | **Speedup**: 4-6x | **Risk**: HIGH — NOT RECOMMENDED

Use Dask XGBoost `gpu_hist` for Final Retrain only, keeping LightGBM for CPCV.

**Why NOT**:
- LightGBM is architecturally correct for the matrix (EFB, sparse CSR, max_bin=2)
- XGBoost accuracy dropped 12% without LightGBM's EFB (see `project_v33_must_use_lgbm_optuna.md`)
- Model semantics differ — hyperparams tuned on LightGBM CPCV don't transfer to XGBoost
- Breaks the principle: never compromise the signal matrix

---

## Detailed Design: NCCL Histogram AllReduce

This is the core of Strategy 1 Part B (and Strategy 2).

### Communication Cost Analysis

```
Per-node histogram size:
  23,600 bundles × 255 max_bin × 2 (grad+hess) × 8 bytes = 96 MB (worst case)
  23,600 bundles × 2 bins × 2 (grad+hess) × 8 bytes     = 0.75 MB (binary-only)

  Reality: ~47K total_bins × 2 × 8 = 0.75 MB per node (confirmed from code)

NCCL AllReduce cost (8x GPUs, PCIe 5.0):
  Ring allreduce effective BW ≈ 20-30 GB/s
  0.75 MB / 25 GB/s ≈ 0.03 ms per node

  Per tree (63 leaves, ~31 internal nodes):
    31 × 0.03 ms ≈ 1 ms per tree (negligible!)

  Per boosting round (with histogram subtraction):
    ~16 allreduces × 0.03 ms ≈ 0.5 ms

Conclusion: NCCL overhead is NEGLIGIBLE for our histogram sizes.
Even without NVLink, PCIe 5.0 handles this easily.
```

### Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     NCCL Communicator (8 GPUs)                   │
│                                                                  │
│  Init: ncclCommInitAll(comms, 8, {0,1,2,3,4,5,6,7})            │
│  One CUDA stream per GPU for compute + comms                     │
│                                                                  │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐     ┌─────────┐          │
│  │  GPU 0  │ │  GPU 1  │ │  GPU 2  │ ... │  GPU 7  │          │
│  │         │ │         │ │         │     │         │          │
│  │ CSR     │ │ CSR     │ │ CSR     │     │ CSR     │          │
│  │ shard 0 │ │ shard 1 │ │ shard 2 │     │ shard 7 │          │
│  │ (28K    │ │ (28K    │ │ (28K    │     │ (28K    │          │
│  │  rows)  │ │  rows)  │ │  rows)  │     │  rows)  │          │
│  │         │ │         │ │         │     │         │          │
│  │ hist[]  │ │ hist[]  │ │ hist[]  │     │ hist[]  │          │
│  │ (local) │ │ (local) │ │ (local) │     │ (local) │          │
│  └────┬────┘ └────┬────┘ └────┬────┘     └────┬────┘          │
│       │           │           │               │                │
│       └───────────┴─────┬─────┴───────────────┘                │
│                         │                                       │
│              ncclAllReduce(ncclSum)                              │
│                         │                                       │
│                   ┌─────┴─────┐                                 │
│                   │  Global   │                                 │
│                   │ Histogram │                                 │
│                   │ (on all   │                                 │
│                   │  GPUs)    │                                 │
│                   └─────┬─────┘                                 │
│                         │                                       │
│                  Split Search                                   │
│                  (identical on all GPUs)                         │
│                         │                                       │
│                  Broadcast split decision                        │
│                  (implicit — all GPUs compute same result)       │
│                         │                                       │
│                  Update local row→leaf mapping                   │
│                  (each GPU updates its shard only)               │
└──────────────────────────────────────────────────────────────────┘
```

### Implementation Plan

#### Layer 1: NCCL Manager (`gpu_histogram_fork/src/nccl_manager.py`)

```python
"""
Manages NCCL communicators for multi-GPU histogram allreduce.
Single-process, multi-device (no MPI needed).
"""
import ctypes
import os

class NCCLManager:
    def __init__(self, gpu_ids: list[int]):
        self.num_gpus = len(gpu_ids)
        self.gpu_ids = gpu_ids
        self._lib = ctypes.CDLL("libnccl.so")  # or link into libgpu_histogram.so

        # Initialize communicators
        self.comms = (ctypes.c_void_p * self.num_gpus)()
        self._lib.ncclCommInitAll(self.comms, self.num_gpus,
                                  (ctypes.c_int * self.num_gpus)(*gpu_ids))

        # Create per-GPU streams
        self.streams = []
        for gpu_id in gpu_ids:
            # cudaSetDevice + cudaStreamCreate
            stream = self._create_stream(gpu_id)
            self.streams.append(stream)

    def allreduce_histograms(self, local_hists: list, hist_size: int):
        """
        AllReduce histogram buffers across all GPUs.
        local_hists: list of device pointers (one per GPU)
        hist_size: number of doubles in histogram
        """
        # ncclGroupStart
        self._lib.ncclGroupStart()
        for i in range(self.num_gpus):
            self._lib.ncclAllReduce(
                local_hists[i],    # sendbuf (in-place)
                local_hists[i],    # recvbuf
                hist_size,
                2,                 # ncclFloat64
                0,                 # ncclSum
                self.comms[i],
                self.streams[i]
            )
        self._lib.ncclGroupEnd()
        # Sync handled by caller

    def destroy(self):
        for comm in self.comms:
            self._lib.ncclCommDestroy(comm)
```

#### Layer 2: Data Partitioner (`gpu_histogram_fork/src/data_partitioner.py`)

```python
"""
Partitions CSR matrix rows across GPUs for data-parallel training.
Each GPU gets a contiguous slice of rows with all features.
"""
import numpy as np
from scipy.sparse import csr_matrix

class DataPartitioner:
    def __init__(self, X_csr: csr_matrix, num_gpus: int):
        self.n_rows = X_csr.shape[0]
        self.num_gpus = num_gpus

        # Equal row partitioning (contiguous for cache locality)
        rows_per_gpu = self.n_rows // num_gpus
        self.partitions = []

        for i in range(num_gpus):
            start = i * rows_per_gpu
            end = self.n_rows if i == num_gpus - 1 else (i + 1) * rows_per_gpu

            # Slice CSR (efficient — just adjust indptr)
            shard = X_csr[start:end]
            self.partitions.append({
                'row_start': start,
                'row_end': end,
                'n_rows': end - start,
                'csr': shard,
                'nnz': shard.nnz
            })

    def get_shard(self, gpu_idx: int) -> dict:
        return self.partitions[gpu_idx]

    def get_labels_shard(self, y: np.ndarray, gpu_idx: int) -> np.ndarray:
        p = self.partitions[gpu_idx]
        return y[p['row_start']:p['row_end']]
```

#### Layer 3: Modified Histogram Builder (`gpu_histogram.cu` changes)

```c
// NEW: Multi-GPU histogram build context
typedef struct {
    // Existing fields...
    GpuHistContext base_ctx;

    // NCCL fields
    ncclComm_t nccl_comm;
    int gpu_rank;
    int num_gpus;
    cudaStream_t nccl_stream;

    // Row partition info
    int local_n_rows;
    int row_offset;  // global row index of local row 0
} MultiGpuHistContext;

// MODIFIED: Build histogram for local rows only
__global__ void sparse_hist_build_kernel_multi(
    const int64_t* indptr,
    const int32_t* indices,
    const uint8_t* data,
    const double*  gradients,
    const int*     leaf_ids,      // local leaf IDs
    double*        histogram,     // local partial histogram
    int            local_n_rows,
    int            target_leaf
) {
    // Same kernel as before — operates on local rows only
    // No code change needed if CSR is already sliced per GPU
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= local_n_rows) return;
    if (leaf_ids[row] != target_leaf) return;

    int64_t start = indptr[row];
    int64_t end   = indptr[row + 1];

    for (int64_t j = start; j < end; j++) {
        int bin = data[j];
        int feature = indices[j];
        int hist_idx = bundle_offset[feature] + bin;

        atomicAdd(&histogram[hist_idx * 2],     gradients[row * 2]);
        atomicAdd(&histogram[hist_idx * 2 + 1], gradients[row * 2 + 1]);
    }
}

// NEW: AllReduce after local histogram build
void multi_gpu_hist_build_and_reduce(
    MultiGpuHistContext* ctx,
    int target_leaf,
    double* out_histogram
) {
    // 1. Build local histogram
    sparse_hist_build_kernel_multi<<<grid, block, 0, ctx->base_ctx.stream>>>(
        ctx->base_ctx.indptr,
        ctx->base_ctx.indices,
        ctx->base_ctx.data,
        ctx->base_ctx.gradients,
        ctx->base_ctx.leaf_ids,
        ctx->base_ctx.histogram,  // local partial
        ctx->local_n_rows,
        target_leaf
    );

    // 2. AllReduce to sum partials across GPUs
    ncclAllReduce(
        ctx->base_ctx.histogram,   // in-place
        ctx->base_ctx.histogram,
        ctx->base_ctx.n_total_bins * 2,
        ncclFloat64,
        ncclSum,
        ctx->nccl_comm,
        ctx->nccl_stream
    );

    // 3. Sync
    cudaStreamSynchronize(ctx->nccl_stream);

    // 4. Copy to output (or use in-place)
    cudaMemcpy(out_histogram, ctx->base_ctx.histogram,
               ctx->base_ctx.n_total_bins * 2 * sizeof(double),
               cudaMemcpyDeviceToHost);
}
```

#### Layer 4: Integration into Training Loop (`lgbm_integration.py` changes)

```python
class MultiGPUHistogramProvider(GPUHistogramProvider):
    """
    Drop-in replacement for GPUHistogramProvider that uses
    NCCL AllReduce across multiple GPUs.
    """

    def __init__(self, X_csr, gpu_ids, **kwargs):
        self.num_gpus = len(gpu_ids)
        self.gpu_ids = gpu_ids

        # Partition data
        self.partitioner = DataPartitioner(X_csr, self.num_gpus)

        # Initialize NCCL
        self.nccl = NCCLManager(gpu_ids)

        # Initialize per-GPU histogram contexts
        self.contexts = []
        for i, gpu_id in enumerate(gpu_ids):
            shard = self.partitioner.get_shard(i)
            ctx = self._init_gpu_context(
                gpu_id=gpu_id,
                csr_shard=shard['csr'],
                nccl_comm=self.nccl.comms[i],
                gpu_rank=i
            )
            self.contexts.append(ctx)

    def build_histogram(self, leaf_id, gradients):
        """Build histogram across all GPUs and allreduce."""
        # 1. Scatter gradients to each GPU's shard
        for i in range(self.num_gpus):
            shard = self.partitioner.get_shard(i)
            local_grads = gradients[shard['row_start']:shard['row_end']]
            self._upload_gradients(self.contexts[i], local_grads)

        # 2. Launch histogram kernels on all GPUs (parallel)
        for ctx in self.contexts:
            self._launch_hist_kernel(ctx, leaf_id)

        # 3. NCCL AllReduce
        local_hists = [ctx.histogram_ptr for ctx in self.contexts]
        self.nccl.allreduce_histograms(local_hists, self.n_total_bins * 2)

        # 4. Sync and return (all GPUs now have identical global histogram)
        self._sync_all()
        return self._download_histogram(self.contexts[0])
```

#### Layer 5: Orchestrator Changes (`ml_multi_tf.py`)

```python
def _train_final_retrain_multi_gpu(params, X_csr, y, available_gpus):
    """
    Final Retrain using all available GPUs via data-parallel
    histogram AllReduce.
    """
    num_gpus = len(available_gpus)

    # Configure multi-GPU histogram provider
    params['_gpu_histogram_multi_gpu'] = True
    params['_gpu_histogram_gpu_ids'] = available_gpus
    params['_gpu_histogram_nccl'] = True

    # Create multi-GPU provider
    provider = MultiGPUHistogramProvider(X_csr, available_gpus)

    # Train with co-processor mode using multi-GPU provider
    dtrain = lgb.Dataset(X_csr, y)
    model = lgb.train(
        params, dtrain,
        callbacks=[provider.as_callback()],
        ...
    )

    provider.cleanup()
    return model
```

---

## VRAM Budget (8x RTX 5090 @ 32GB each)

### Data-Parallel: 15m TF distributed across 8 GPUs

```
Per-GPU allocation (227K rows / 8 = 28K rows per GPU):

CSR shard:
  indptr:    28K × 8 bytes        =   0.22 MB
  indices:   ~NNZ/8 × 4 bytes     =   varies (~0.5-2 GB)
  data:      ~NNZ/8 × 1 byte      =   varies (~0.1-0.5 GB)

Gradients:
  28K × 2 × 8 bytes              =   0.44 MB

Histograms:
  63 leaves × 47K bins × 16 bytes =   47 MB

Total per GPU:                     ≈  1-5 GB (down from 41 GB!)

Headroom:                          27-31 GB FREE per GPU
```

**Key insight**: Data-parallel SOLVES the VRAM problem too. 15m TF currently needs 41GB (doesn't fit one 5090). Split across 8 GPUs, each needs only ~2-5GB.

---

## Speedup Estimates

### Histogram Build (compute)
```
Current:  227K rows on 1 GPU
Proposed: 28K rows on each of 8 GPUs (parallel)

Histogram kernel scales linearly with rows (row-parallel atomicAdd).
Expected compute speedup: ~8x (near-linear)
```

### NCCL AllReduce (communication overhead)
```
Histogram payload: 47K bins × 16 bytes = 0.75 MB per node
Allreduce latency: ~0.03 ms per node (PCIe 5.0)
Per tree (31 nodes): ~1 ms
Per boosting round: ~1 ms

Fraction of round time: 1ms / (compute_time/8 + 1ms)
If single-GPU round = 100ms → multi-GPU round = 100/8 + 1 = 13.5ms
Efficiency: 12.5/13.5 = 93%
```

### End-to-End
```
Phase              Current    Proposed              Speedup
─────────────────  ─────────  ────────────────────  ───────
Optuna CPCV        ~2hr       fold-parallel (8GPU)  4-8x
Final Retrain      ~4hr       data-parallel (8GPU)  6-7x
─────────────────  ─────────  ────────────────────  ───────
Total 15m TF       ~6hr       ~45min - 1hr          6-8x
```

---

## Implementation Phases

### Phase 1: Fold-Parallel CPCV (1-2 days)
- Modify `ml_multi_tf.py` to spawn fold training on separate GPUs via `multiprocessing`
- No CUDA changes needed — each fold is independent
- Immediate 2-4x speedup on CPCV phase
- Also parallelize Optuna trials across GPUs

### Phase 2: NCCL Histogram AllReduce (3-5 days)
- Add `nccl_manager.py` — NCCL communicator init/destroy
- Add `data_partitioner.py` — CSR row-sharding
- Modify `gpu_histogram.cu` — add allreduce after kernel
- Modify `lgbm_integration.py` — `MultiGPUHistogramProvider`
- Modify `ml_multi_tf.py` — route Final Retrain through multi-GPU path

### Phase 3: Optimization (2-3 days)
- Overlap compute and communication (pipeline node k+1 build with node k allreduce)
- FP16 histogram accumulation option (halves allreduce bandwidth)
- Histogram subtraction (only build one child, derive sibling from parent)
- Level-wise batched allreduce (one allreduce per tree depth, not per node)

### Phase 4: Validation (1-2 days)
- Bitwise comparison: single-GPU vs multi-GPU histograms
- Model accuracy comparison: ensure no degradation
- Benchmark: measure actual speedup vs estimates
- Stress test: 15m TF full pipeline

---

## Key Design Decisions

### Q: Data-parallel (split rows) or Feature-parallel (split features)?

**Answer: Data-parallel.**

Rationale:
- Histogram size is tiny (0.75 MB per node) → allreduce overhead negligible
- Row partitioning is trivial with CSR (just slice indptr)
- Feature-parallel requires coordinating which GPU "owns" which features — complex bookkeeping
- Data-parallel produces mathematically identical results to single-GPU
- XGBoost validated this approach with `gpu_hist` — proven architecture

### Q: Single-process multi-device or multi-process?

**Answer: Single-process, multi-device.**

Rationale:
- Avoids IPC overhead for gradient/histogram sharing
- NCCL supports `ncclCommInitAll` for single-process multi-GPU
- Simpler debugging, no MPI dependency
- All 8 GPUs managed from one Python process via ctypes

### Q: What about Optuna trial-level parallelism?

**Answer: Use BOTH fold-parallel AND trial-parallel.**

```
Optuna Phase 1 (fast search):
  Trial 0 → GPU 0,1 (fold-parallel within trial)
  Trial 1 → GPU 2,3
  Trial 2 → GPU 4,5
  Trial 3 → GPU 6,7
  = 4 trials running simultaneously

Optuna Validation (thorough):
  Trial → all 8 GPUs (4-fold CPCV, 2 GPUs per fold)

Final Retrain:
  → all 8 GPUs (data-parallel histogram allreduce)
```

### Q: Can we use histogram subtraction to reduce allreduce count?

**Answer: Yes — halves the allreduce count.**

LightGBM already computes sibling histograms by subtraction:
```
parent_hist - child_hist = sibling_hist
```
In multi-GPU mode: only allreduce the smaller child's histogram. Derive the sibling locally from cached parent histogram. This cuts allreduce calls from ~31/tree to ~16/tree.

### Q: What if GPUs have different VRAM (mixed fleet)?

**Answer: Proportional row partitioning.**

```python
# Distribute rows proportional to free VRAM
vram = [get_free_vram(gpu) for gpu in gpu_ids]
total_vram = sum(vram)
rows_per_gpu = [int(n_rows * v / total_vram) for v in vram]
```

This handles mixed GPU fleets (e.g., 4x 5090 + 4x 4090).

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| NCCL version incompatibility | Build fails | Pin NCCL version in fork's CMakeLists.txt |
| Numerical divergence (FP ordering) | Different model | Use `deterministic=True`, validate histograms |
| PCIe bandwidth saturation | Slower than expected | Histograms are only 0.75MB — not a risk |
| GPU memory fragmentation | OOM on some GPUs | Use CUDA memory pools, pre-allocate |
| LightGBM callback API limitations | Can't intercept histogram build | Already solved — co-processor mode bypasses this |
| Row partition imbalance (leaf routing) | Some GPUs idle | Monitor and rebalance if skew > 2x |

---

## Dependencies

- NCCL 2.18+ (already available in CUDA 12+ toolkit)
- Our custom `libgpu_histogram.so` (already built and working)
- `gpu_histogram.cu` (modification — add allreduce call)
- `lgbm_integration.py` (modification — multi-GPU provider)
- `ml_multi_tf.py` (modification — orchestration)

No new external libraries. No changes to LightGBM core. All changes are in our fork layer.

---

## Summary

The single-GPU ceiling is breakable with **data-parallel histogram allreduce via NCCL**. The histogram payload (0.75 MB/node) is so small that NCCL overhead is negligible even on PCIe. The primary speedup comes from distributing the histogram BUILD (row-parallel kernel) across 8 GPUs, not from any communication trick.

**Expected result**: 15m TF Final Retrain drops from **6 hours → ~45 minutes**.

Phase 1 (fold-parallel) can ship in 1-2 days with zero CUDA changes. Phase 2 (NCCL allreduce) takes 3-5 days but delivers the full 6-8x speedup on the bottleneck Final Retrain step.
