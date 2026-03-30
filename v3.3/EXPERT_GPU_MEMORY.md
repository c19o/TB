# EXPERT: GPU Memory & PCIe Bus Architecture

**Scope**: GPU memory hierarchy, PCIe data movement, and host-device transfer optimization for the custom CUDA LightGBM fork.
**Hardware Target**: 8x RTX 5090 Blackwell (32GB GDDR7 each), PCIe 5.0 x16 (64 GB/s per GPU)
**Data Profile**: CSR matrices up to 10M features, 227K rows, int64 indptr, int32 indices, binary cross features
**Date**: 2026-03-30

---

## Table of Contents

1. [Current Architecture Assessment](#1-current-architecture-assessment)
2. [Pinned Memory for Gradient Uploads](#2-pinned-memory-for-gradient-uploads)
3. [CUDA Stream Overlap: H2D + Compute](#3-cuda-stream-overlap-h2d--compute)
4. [Unified Memory on Blackwell: Verdict](#4-unified-memory-on-blackwell-verdict)
5. [Minimizing PCIe Bottleneck with 8 GPUs](#5-minimizing-pcie-bottleneck-with-8-gpus)
6. [Memory Pool vs malloc Per-Round](#6-memory-pool-vs-malloc-per-round)
7. [Concrete Recommendations](#7-concrete-recommendations)

---

## 1. Current Architecture Assessment

### What's Already Right

The existing `memory_manager.py` and `leaf_gradient_scatter.py` are well-architected:

- **Gradient double-buffering** (`GradientBuffers`): Two pinned host buffers (A/B) + one GPU buffer. CPU fills the back buffer while GPU reads the front. This is textbook correct.
- **Pinned histogram D2H**: `allocate_histogram_pool()` creates pinned host buffers for async D2H of histogram results. Correct.
- **CSR stays GPU-resident**: Uploaded once via `upload_csr()`, used across all boosting rounds. No per-round H2D of the feature matrix.
- **Histogram pool with subtraction trick**: Pre-allocated `max_leaves` histogram buffer pairs. Avoids per-node malloc churn.
- **Adaptive strategy selection**: `LeafGradientScatter` auto-selects masked SpMV (large leaves) vs atomic scatter (small leaves) at 25% crossover. The atomic kernel has near-zero contention at 2.2M features — collision probability ~0.04%.
- **SIGTERM cleanup + atexit**: GPU memory is reliably freed on crash/kill.

### Gaps to Address

| Gap | Location | Impact |
|-----|----------|--------|
| CSR upload uses `cp.asarray()` (pageable path) | `memory_manager.py:230-232` | ~2.2x slower H2D for initial upload |
| No per-GPU pinned staging buffers for multi-GPU | `memory_manager.py` (single device) | Serialized DMA across 8 GPUs |
| `get_histogram_result()` copies then syncs | `memory_manager.py:493-498` | Blocks stream; could pipeline |
| No CUDA stream separation for H2D vs compute | `leaf_gradient_scatter.py` | Cannot overlap gradient upload with previous histogram |
| CuPy default memory pool used everywhere | Both files | Fragmentation risk over 1000+ boosting rounds |
| Dual CSR storage (CSR + CSR.T) without shared indices | `leaf_gradient_scatter.py:555-564` | Double VRAM for index storage |

---

## 2. Pinned Memory for Gradient Uploads

### Answer: YES — already partially done, extend to CSR upload

**Current state**: Gradients already use pinned memory (correct). CSR upload does not.

**The numbers on PCIe 5.0 x16**:
- Pinned H2D: **~50 GB/s** measured on RTX 5090
- Pageable H2D: **~20-23 GB/s** (2.2x slower due to hidden double-copy through 4MB driver staging buffer)
- Theoretical PCIe 5.0 x16 peak: ~63 GB/s bidirectional

**Why pageable is bad for CSR**: The driver's 4MB staging buffer chunks the transfer. A CSR `indptr` array for 227K rows = 1.8MB (fits one chunk), but `indices` for 10M NNZ = 40MB and `data` = 80MB — these get fragmented across ~10-20 DMA chunks with synchronization overhead between each.

### Recommendation for CSR upload

```python
# CURRENT (pageable — slow):
gpu_indptr = cp.asarray(scipy_csr.indptr)  # copies through driver staging

# RECOMMENDED (pinned — 2.2x faster):
pin_indptr = cp.cuda.alloc_pinned_memory(scipy_csr.indptr.nbytes)
np_pin = np.frombuffer(pin_indptr, dtype=np.int64)[:len(scipy_csr.indptr)]
np_pin[:] = scipy_csr.indptr
gpu_indptr = cp.empty(len(scipy_csr.indptr), dtype=cp.int64)
gpu_indptr.set(np_pin)  # DMA from pinned → device
```

**For write-once arrays** (CSR `data[]` and `indices[]`): Use `cudaHostAllocWriteCombined` — CPU writes sequentially, GPU reads only. WC memory coalesces CPU writes for faster DMA. Do NOT use WC for `indptr` if CPU needs random reads during upload validation.

**Impact**: CSR upload is a one-time cost per training run, so the 2.2x speedup matters less than for gradients. But on 15m (227K rows, potentially billions of NNZ), shaving upload from 3s to 1.4s is worthwhile. **Priority: Medium.**

---

## 3. CUDA Stream Overlap: H2D + Compute

### Answer: YES — but the biggest win is device-side, not host-device

**Key insight**: The CSR matrix is already GPU-resident. The per-round H2D traffic is only gradients + hessians — small arrays (~227K * 8 bytes * 2 = 3.6MB). At 50 GB/s pinned, that's **0.07ms**. The histogram kernel itself takes 1-10ms. So H2D overlap saves <1% of round time.

**Where overlap actually matters**:

#### A) Overlap gradient upload with previous histogram readback

```
Stream 0 (compute):  [histogram kernel round N] → [histogram kernel round N+1]
Stream 1 (transfer): [D2H histogram N] ←→ [H2D gradients N+1]
```

The current code does `s.synchronize()` inside `get_histogram_result()` — this blocks the compute stream. Instead:

1. Launch histogram D2H on stream 1
2. Launch gradient H2D on stream 1
3. Launch next histogram kernel on stream 0
4. Synchronize stream 1 only when CPU needs the histogram result

**Estimated saving**: 0.1-0.3ms per round × 1000 rounds = 100-300ms total. Modest but free.

#### B) Overlap within the histogram kernel (bigger win)

For the atomic scatter path, the kernel walks CSR entries sequentially per row. With `cuda::memcpy_async` or double-buffered shared memory:

1. Warp 0 prefetches next row's CSR indices from global → shared
2. Warp 1 processes current row's indices from shared memory
3. Pipeline hides HBM latency (~500 cycle round-trip) behind compute

**This is where the real bandwidth utilization improvement lives.** The RTX 5090 has 1.79 TB/s HBM bandwidth, but irregular CSR access patterns achieve maybe 20-30% utilization. Double-buffered prefetch can push this to 40-50%.

**Estimated saving**: 2-5x kernel speedup for atomic scatter path. **Priority: High.**

#### C) Multi-leaf pipelining

When building histograms for multiple leaves in the same tree level, pipeline them:

```
Stream 0: [leaf 0 histogram] → [leaf 2 histogram] → ...
Stream 1: [leaf 1 histogram] → [leaf 3 histogram] → ...
```

LightGBM's subtraction trick means only the smaller child needs a full histogram build. Pipeline the independent leaf computations across 2-4 streams.

---

## 4. Unified Memory on Blackwell: Verdict

### Answer: DO NOT USE for the training hot path

**Why not**:

1. **On-demand page migration stalls**: Unified Memory faults on first access. For irregular CSR access patterns, this means thousands of 64KB page faults per kernel launch. Each fault costs ~10-50us. With 10M features across hundreds of pages, startup penalty alone could exceed the entire kernel time.

2. **No GPUDirect RDMA compatibility**: NVIDIA explicitly documents that Unified Memory is not supported with GPUDirect RDMA — the GPU copy can become incoherent with the writable non-GPU copy. If you ever want inter-GPU direct access, UM blocks that path.

3. **No correctness benefit for our data**: Our CSR structure is static per training run. We know exactly what goes where. Explicit placement is both faster and simpler.

4. **Prefetch hints don't help irregular access**: `cudaMemPrefetchAsync` works for bulk sequential access. CSR column-index access is scattered — the prefetcher can't predict which pages will be needed.

**Where UM could be acceptable** (non-critical paths only):
- Cold metadata (feature names, config)
- Debugging / profiling buffers
- VRAM oversubscription experiments (15m with >32GB CSR)

**For the 15m timeframe VRAM overflow case**: Use explicit host-pinned staging with manual paging (load CSR chunks on-demand) rather than UM. You control the eviction policy; UM's eviction is opaque and can thrash.

---

## 5. Minimizing PCIe Bottleneck with 8 GPUs

### Critical: RTX 5090 has NO NVLink

GeForce RTX 5090 does **not** support NVLink. All inter-GPU communication goes through PCIe 5.0 + CPU/chipset. This is the single biggest constraint for multi-GPU training.

### PCIe Topology Matters

```
Ideal topology (8x GPUs, dual-socket):
  CPU0 ← PCIe switch → GPU0, GPU1, GPU2, GPU3
  CPU1 ← PCIe switch → GPU4, GPU5, GPU6, GPU7

Worst case: all 8 GPUs behind one root complex
  → shared 128 GB/s for all cross-GPU traffic
```

**Action item**: Run `nvidia-smi topo -m` on the target machine before deploying. GPUs sharing a PCIe switch get x8 lanes (32 GB/s) instead of x16 (64 GB/s). This changes all batching math.

### Bandwidth Budget

| Transfer Type | Size | Time @ 50 GB/s | Frequency |
|--------------|------|-----------------|-----------|
| Gradient H2D (per GPU) | 3.6 MB | 0.07 ms | Every round |
| Histogram all-reduce (per level) | ~160 MB (10M bins × 16B) | 3.2 ms | Every tree level |
| CSR upload (one-time) | Up to 32 GB | 640 ms | Once at init |

**The histogram all-reduce is the bottleneck.** At 10M features × 16 bytes (grad+hess) × 8 GPUs, a naive all-reduce moves ~1.3 GB per tree level through PCIe. With ~63 levels per tree and 1000 trees, that's 80 TB of PCIe traffic.

### Mitigation Strategies (ordered by impact)

#### 1. Feature-sharded reduce-scatter (CRITICAL)

Partition features across GPUs. Each GPU owns 1.25M features (10M / 8). Build local histograms for ALL features, then:
- **Reduce-scatter**: Each GPU sends only its non-owned feature histograms to the owner
- **Broadcast**: Owner sends merged histogram back
- Traffic per GPU per level: ~140 MB (7/8 of 160 MB) instead of 1.3 GB

This is LightGBM's native data-parallel design. **Must implement.**

#### 2. Sparse histogram compression

For binary features, most histogram bins have the same gradient sum as the parent (feature=0 for most rows). Only bins where feature=1 have distinct values. With 99.7% sparsity, only ~0.3% of bins differ from parent.

**Transmit only non-trivial bins**: 10M × 0.3% = 30K bins × 16 bytes = 480 KB instead of 160 MB. **333x reduction.**

This requires a custom NCCL-free reduction using packed sparse deltas. Significant engineering but transformative for PCIe-bound 8-GPU training.

#### 3. Histogram subtraction across GPUs

LightGBM's subtraction trick: `child_small = parent - child_large`. Only build the histogram for the smaller child; derive the other by subtraction. This halves communication per level.

#### 4. Async histogram reduction

Overlap level N's all-reduce with level N+1's local histogram build for independent subtrees:

```
Stream 0 (compute): [build hist level N+1, left subtree]
Stream 1 (comms):   [all-reduce level N results]
```

Requires dependency tracking but can hide 50-80% of communication latency.

#### 5. Row sharding as alternative

Instead of feature sharding, each GPU owns a subset of rows. Local histograms are partial sums. All-reduce is a simple element-wise sum.

**Pros**: Simpler implementation, natural load balance
**Cons**: Every GPU must communicate ALL features' histograms. With 10M features this is prohibitive on PCIe.

**Verdict**: Feature sharding + sparse compression is the correct architecture for PCIe-only 8-GPU with 10M sparse features.

---

## 6. Memory Pool vs malloc Per-Round

### Answer: Three-tier allocation strategy

**Do NOT use one-size-fits-all allocation.** The workload has three distinct lifetime classes:

#### Tier 1: Persistent (Arena — allocate once, never free)
- CSR indptr, indices, data (uploaded at init)
- CSR.T (computed at init)
- Feature metadata, bin boundaries
- **Strategy**: Page-aligned VA reservation via `cuMemAddressReserve` + `cuMemCreate` + `cuMemMap`. Or simply CuPy's default pool with manual pinning.
- **Size**: Up to 32GB per GPU. Allocated once. Never reallocated.

#### Tier 2: Semi-persistent (Slab allocator — reuse across rounds)
- Gradient/hessian device buffers (same size every round)
- Histogram pool (pre-allocated per `max_leaves`)
- Row index arrays (change per node but bounded by `n_rows`)
- Node partition buffers
- **Strategy**: Fixed-size slab classes (1MB, 4MB, 16MB, 64MB). Allocate at round 0, reuse thereafter. The current `allocate_histogram_pool()` already does this correctly.
- **Size**: ~500MB per GPU. Shape-stable after first tree.

#### Tier 3: Ephemeral scratch (Bump allocator — bulk reset per round)
- Temporary reduction buffers
- Sort/partition scratch space
- Scan temporaries
- **Strategy**: Single large scratch buffer, bump-allocated forward during a round, bulk-reset to zero at round end. No individual frees.
- **Size**: ~200MB per GPU. Grows to high-water mark, stays there.

### CuPy Memory Pool Tuning

CuPy's default memory pool already acts as a caching allocator (similar to PyTorch's CUDA caching allocator). For our workload:

```python
# RECOMMENDED: Increase allocation unit to reduce fragmentation
pool = cp.get_default_memory_pool()
pool.set_limit(size=28 * 1024**3)  # Cap at 28GB (leave 4GB headroom on 32GB 5090)

# For pinned memory:
pinned_pool = cp.get_default_pinned_memory_pool()
pinned_pool.set_limit(size=4 * 1024**3)  # 4GB pinned host per GPU
```

**Key rule**: Never call `pool.free_all_blocks()` between boosting rounds — this forces re-allocation. Only call it during `cleanup()` at end of training. The current code in `cleanup()` is correct. Just ensure nobody calls it mid-training.

### Why NOT `cudaMalloc` per round

Each `cudaMalloc` call:
- Acquires a driver mutex (serializes across all threads)
- Potentially triggers a CUDA context synchronization
- Costs 10-100us per call

At 1000 rounds × 63 nodes × 2 mallocs (grad+hess) = 126,000 malloc calls. At 50us each = **6.3 seconds wasted** on memory management alone. The current histogram pool design avoids this. Keep it.

---

## 7. Concrete Recommendations

### Priority 1: Must-do for 8x RTX 5090 deployment

| # | Change | File | Impact |
|---|--------|------|--------|
| 1 | Feature-sharded histogram reduce-scatter | New: `multi_gpu_reducer.py` | Enables 8-GPU scaling |
| 2 | Sparse histogram delta compression | New: `sparse_hist_compress.py` | 333x less PCIe traffic |
| 3 | Double-buffered shared memory in atomic scatter kernel | `leaf_gradient_scatter.py:298-326` | 2-5x kernel speedup |
| 4 | Per-GPU memory manager instances | `memory_manager.py` | Independent DMA engines |

### Priority 2: Should-do for performance

| # | Change | File | Impact |
|---|--------|------|--------|
| 5 | Pinned memory for CSR upload | `memory_manager.py:229-232` | 2.2x faster init |
| 6 | Separate CUDA streams for transfer vs compute | `memory_manager.py` | 0.1-0.3ms/round saved |
| 7 | Remove `.synchronize()` from `get_histogram_result` | `memory_manager.py:495` | Unblock compute stream |
| 8 | Slab allocator for tier-2 buffers | `memory_manager.py` | Eliminate fragmentation |

### Priority 3: Nice-to-have

| # | Change | File | Impact |
|---|--------|------|--------|
| 9 | WC pinned memory for CSR data/indices upload | `memory_manager.py` | ~10% faster upload |
| 10 | Topology-aware GPU assignment | New: `topo_detect.py` | Optimal PCIe routing |
| 11 | CUDA events for fine-grained profiling | Both files | Optimization feedback |

### What NOT to do

- **Do not use Unified Memory** for CSR data, gradients, or histograms
- **Do not use GPUDirect RDMA** between 5090s (no NVLink, PCIe peer access unreliable across root complexes)
- **Do not share pinned host buffers across GPUs** (serializes DMA engines)
- **Do not call `pool.free_all_blocks()` mid-training** (forces re-allocation)
- **Do not use `cudaHostAllocMapped` (zero-copy)** for bulk CSR data — every GPU read traverses PCIe
- **Do not densify sparse histograms for communication** — transmit only non-trivial bins

### Verification Checklist

Before deploying to 8x RTX 5090:

- [ ] `nvidia-smi topo -m` confirms each GPU has x16 PCIe 5.0 lanes
- [ ] CUDA `bandwidthTest --mode=shmoo` confirms >45 GB/s pinned H2D per GPU
- [ ] No two GPUs share a PCIe switch (or batching math adjusted if they do)
- [ ] Gradient double-buffering validated: CPU fills buffer B while GPU reads A
- [ ] Histogram pool allocated once at init, reused across all rounds
- [ ] No `cudaMalloc`/`cudaFree` calls in the boosting hot loop
- [ ] NCCL or custom reduce-scatter verified for histogram merging
- [ ] Total VRAM usage per GPU: CSR + CSR.T + gradients + histograms < 28GB (85% of 32GB)

---

## Appendix: Bandwidth Reference

| Interconnect | Bandwidth (bidirectional) | Latency | Available on 5090? |
|-------------|--------------------------|---------|-------------------|
| PCIe 5.0 x16 | 128 GB/s | ~1 us | YES |
| PCIe 5.0 x8 (shared switch) | 64 GB/s | ~1 us | Possible |
| NVLink 4 (Hopper) | 900 GB/s | ~0.5 us | NO |
| NVLink 5 (Blackwell datacenter) | 1.8 TB/s | ~0.3 us | NO (datacenter only) |
| GDDR7 local bandwidth (5090) | 1,792 GB/s | — | YES (on-die) |

**Key takeaway**: Local VRAM bandwidth (1.79 TB/s) is 14x faster than PCIe (128 GB/s). Every byte kept GPU-resident instead of transferred saves 14x. The architecture must be designed around minimizing PCIe traffic, not maximizing it.
