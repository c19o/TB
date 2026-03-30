# CrossGen-Numba Changes — Optimizations #1 + #6

## What Changed

### New File: `v3.3/numba_cross_kernels.py`
Contains all Numba-accelerated cross gen kernels:

| Function | Purpose |
|----------|---------|
| `_intersect_sorted()` | Two-pointer sorted intersection on CSC indices (count-only) |
| `_intersect_sorted_fill()` | Two-pointer intersection that fills output array |
| `_count_intersections()` | **Pass 1**: `@njit(parallel=True)` prange over all pairs, counts intersection sizes |
| `_fill_intersections()` | **Pass 2**: `@njit(parallel=True)` prange fills pre-allocated output with row indices |
| `sort_pairs_l2_friendly()` | **Opt #6**: Lexsort pairs by (left_col, right_col) for L2 cache reuse |
| `numba_csc_cross()` | High-level API: CSC conversion -> pair sort -> 2-pass kernel -> CSR output |
| `warmup_numba_kernels()` | Pre-compiles kernels with tiny arrays to avoid JIT latency on first real call |
| `_llvm_ctpop_i64()` | LLVM intrinsic for popcount (available for future co-occurrence counting) |

### Modified File: `v3.3/v2_cross_generator.py`
- Added `USE_NUMBA_CROSS` env var check + import of `numba_cross_kernels`
- Added `_numba_cross_chunk()` function — drop-in replacement for `_cpu_cross_chunk`
- Wired numba path into `gpu_batch_cross()` dispatch (GPU -> Numba -> CPU fallback)
- **No existing functions modified or deleted**

## How to Enable

```bash
# Enable Numba CSC intersection for CPU cross gen
export USE_NUMBA_CROSS=1
python v2_cross_generator.py --tf 1d

# Disable (default) — uses existing dense multiply path
unset USE_NUMBA_CROSS
python v2_cross_generator.py --tf 1d
```

## How It Works

### Old Path (`_cpu_cross_chunk`)
1. Extract dense columns from left/right matrices for each pair batch
2. Element-wise multiply (Numba prange) -> dense (N, batch) array
3. `np.nonzero()` to find nonzero entries -> COO arrays
4. Convert COO -> CSR chunks

**Bottleneck**: Allocates dense (N, batch) arrays even though result is ~5-15% dense. `np.nonzero` scans all N*batch elements.

### New Path (`_numba_cross_chunk` + `numba_csc_cross`)
1. Convert left/right to CSC sparse (one-time, O(NNZ))
2. **Opt #6**: Sort pairs by left column index (lexsort) — same left column stays in L2 cache
3. **Pass 1**: Numba prange counts intersection size per pair (two-pointer on sorted CSC indices)
4. Prefix sum to compute output offsets
5. **Pass 2**: Numba prange fills pre-allocated int32 array with intersection row indices
6. Build CSR in chunks from the flat output array

**Why faster**: No dense allocation, no np.nonzero scan. Two-pointer intersection is O(NNZ_left + NNZ_right) per pair vs O(N) for dense multiply. L2 tiling avoids reloading same column 100x.

## Expected Speedup

| Optimization | Source | Expected |
|-------------|--------|----------|
| #1: Numba CSC intersection | Eliminates dense alloc + np.nonzero | 3-8x |
| #6: L2 pair sorting | Same left col reused across ~100 partners | 2-5x additional on 15m |

Combined: **6-40x** on large TFs (1h, 15m), **3-8x** on small TFs (1w, 1d).

## Compatibility

- **Output format**: Identical CSR with int64 indptr (downstream-safe)
- **Feature values**: Binary 0/1 (AND = intersection). Lossless.
- **Co-occurrence pre-filter**: Same sparse matmul as before (GPU cuSPARSE or CPU MKL)
- **Toggle**: `USE_NUMBA_CROSS=0` (default) uses original path. Zero risk to existing runs.
- **Dependencies**: numba (already required), llvmlite (numba dependency)

## What Was NOT Changed
- NPZ save/load logic
- Step orchestration / ThreadPoolExecutor
- Co-occurrence counting (sparse matmul pre-filter)
- GPU cross path (`_gpu_cross_chunk`)
- Any file outside v3.3/
