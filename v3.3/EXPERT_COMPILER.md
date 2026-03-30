# EXPERT: Compiler & Numba/LLVM Specialist

**Context:** Numba JIT for sparse binary feature cross-generation, sorted-index intersection on CSC, AMD EPYC AVX-512. Research conducted 2026-03-30.

---

## 1. Numba LLVM Optimization for Sorted-Index Intersection

### Core Finding
Sorted-index intersection on CSC columns is **branch-heavy and memory-latency-bound**. LLVM autovectorization helps far less than people expect for this pattern. The gains come from algorithm shape and memory layout, not from compiler magic.

### Why LLVM Struggles with Sparse Intersection
- Two-pointer merge on sorted arrays is dominated by **unpredictable branches**, **dependent loads**, and **irregular memory access**
- LLVM's vectorizer needs straight-line arithmetic loops; branchy pointer-chasing code resists autovectorization
- Even on AVX-512-capable hardware, the scalar merge loop often does not emit `zmm` (512-bit) instructions
- Sparse intersection scales better with **thread-level parallelism** than with SIMD width

### Practical Kernel Strategy
| Situation | Best Approach |
|---|---|
| Similar nnz per CSC column | Plain two-pointer intersection in `@njit` nopython mode |
| Highly imbalanced column nnz (10:1+) | Galloping/exponential search from shorter side |
| Huge number of independent feature pairs | `prange` on outer pair loop; keep each task coarse |
| Need AVX-512 payoff | Recast into blockwise masks, counts, or dense mini-batches |
| Unsorted sparse columns | Sort first or enforce `has_sorted_indices` invariant |

### CSC Prerequisite
SciPy's `has_sorted_indices` must be True. If column slices are not sorted, two-pointer or galloping intersection loses its advantage and **can produce incorrect results** if the algorithm assumes monotone advancement. Sort once at construction time, never repeatedly.

### Current V3.3 Status
The cross generator (`v2_cross_generator.py`) currently uses:
- `@njit(parallel=True, cache=True)` for binarization (`_binarize_batch_4tier`, `_binarize_batch_2tier`)
- `@njit(parallel=True, cache=True)` for `_parallel_cross_multiply` (dense element-wise)
- Sparse matmul pre-filter for co-occurrence counts, then dense block multiply
- This is NOT doing CSC sorted-index intersection -- it converts to dense blocks first

**Implication:** The current approach avoids the sparse intersection bottleneck entirely by using dense blocks + sparse matmul pre-filter. A Numba CSC intersection kernel would only help if we wanted to avoid the dense conversion step, which matters most for 15m/1h where memory is tight.

---

## 2. Numba `fastmath` and SIMD Safety

### `fastmath=True` Safety Assessment

**Safe for binary cross-gen:** YES, conditionally.
- Binary cross features are integer presence/absence operations
- The hot path is integer index traversal and counting
- `fastmath` mainly affects floating-point reassociation in reductions (irrelevant for integer counting)
- For the `_parallel_cross_multiply` kernel (float32 multiply of 0.0/1.0 values): `fastmath` is safe because there are no `inf`/`nan` inputs (binarized masks are clean 0.0/1.0)

**NOT safe for:**
- Any kernel that processes raw feature values with NaN (binarization handles NaN explicitly)
- Reductions where IEEE float ordering matters

### Recommended Defaults
```python
# Binary cross multiply -- fastmath is safe (0/1 inputs only)
@njit(parallel=True, cache=True, fastmath=True)
def _parallel_cross_multiply(left, right, out):
    n_rows, n_pairs = left.shape
    for j in prange(n_pairs):
        for i in range(n_rows):
            out[i, j] = left[i, j] * right[i, j]

# Binarization -- fastmath NOT safe (processes NaN values)
@njit(parallel=True, cache=True)  # NO fastmath
def _binarize_batch_4tier(values, n_cols):
    ...
```

### SIMD Verification
To confirm LLVM emitted AVX-512 instructions, inspect Numba's generated assembly:
```python
from numba import njit
@njit
def kernel(...): ...
kernel.inspect_asm(kernel.signatures[0])
# Look for zmm registers = AVX-512, ymm = AVX2, xmm = SSE
```

On AMD EPYC:
- **Zen 4 (9004 series):** AVX-512 via dual 256-bit datapaths (2 cycles per 512-bit op)
- **Zen 5 (9005 series):** Full 512-bit native datapath (1 cycle), up to 37% IPC improvement for suitable workloads
- **Reality for sparse intersection:** Neither gen benefits much because the bottleneck is branches/memory, not arithmetic throughput

---

## 3. LightGBM: Compile from Source with PGO/LTO/-march=native

### Expected Gains
No definitive published benchmark exists for PGO+LTO+`-march=native` on LightGBM specifically (as of 2026-03). Microsoft's LightGBM benchmark framework supports testing custom builds but has not published canonical numbers. Based on analogous projects (MySQL PGO/LTO benchmarks, GCC workloads):

| Optimization | Expected Gain | Confidence |
|---|---|---|
| `-O3 -march=native` (vs pip wheel `-O2 -march=x86-64`) | 5-15% training speed | Medium-high |
| LTO (Link-Time Optimization) | 3-8% additional | Medium |
| PGO (Profile-Guided Optimization) | 5-15% additional | Medium |
| Combined PGO+LTO+native | 10-30% total vs pip wheel | Medium |

### Build Commands (Linux, GCC 12+)

**Standard optimized build:**
```bash
cd LightGBM && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-O3 -march=native -mtune=native" \
    -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native" \
    -DUSE_CUDA=ON
make -j$(nproc)
```

**With LTO:**
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-O3 -march=native -flto=auto" \
    -DCMAKE_CXX_FLAGS="-O3 -march=native -flto=auto" \
    -DCMAKE_EXE_LINKER_FLAGS="-flto=auto" \
    -DUSE_CUDA=ON
make -j$(nproc)
```

**With PGO (two-pass):**
```bash
# Pass 1: Instrumented build
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-O3 -march=native -fprofile-generate=/tmp/lgbm_pgo" \
    -DCMAKE_CXX_FLAGS="-O3 -march=native -fprofile-generate=/tmp/lgbm_pgo"
make -j$(nproc)

# Run representative training workload
python -c "
import lightgbm as lgb
import numpy as np
from scipy.sparse import random as sparse_random
# Generate workload matching our matrix shape
X = sparse_random(50000, 500000, density=0.001, format='csr', dtype=np.float32)
y = np.random.randint(0, 3, 50000)
ds = lgb.Dataset(X, label=y, free_raw_data=False)
params = {'objective': 'multiclass', 'num_class': 3, 'num_leaves': 255,
          'max_bin': 2, 'feature_fraction': 0.8, 'verbose': -1}
lgb.train(params, ds, num_boost_round=50)
"

# Pass 2: Optimized build using profile data
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-O3 -march=native -flto=auto -fprofile-use=/tmp/lgbm_pgo -fprofile-correction" \
    -DCMAKE_CXX_FLAGS="-O3 -march=native -flto=auto -fprofile-use=/tmp/lgbm_pgo -fprofile-correction"
make -j$(nproc)
```

### GPU Fork Consideration
The V3.3 `gpu_histogram_fork` already builds from source. Adding `-march=native` to the CMake flags in `build_linux.sh` is a one-line change. PGO requires a two-pass build which complicates the deployment script but is worth it for 15m training (hours per fold).

### What `-march=native` Unlocks on EPYC
- AVX-512 codegen for histogram accumulation, gradient computation
- BMI2 for bit manipulation in EFB (Exclusive Feature Bundling)
- Vectorized float comparisons in split finding
- The pip wheel ships with `-march=x86-64` (SSE2 baseline) -- leaving 50%+ of hardware capability unused

---

## 4. Numba Parallel `prange` Scheduling

### Current V3.3 Behavior
The cross generator sets `NUMBA_NUM_THREADS` dynamically (line 60):
```python
_numba_threads = max(4, min(_ncpu, 64))
```
And uses `@njit(parallel=True)` with `prange` for binarization and cross multiply.

### Scheduling Options

**Default (static):** Numba's `prange` uses approximately equal-sized chunks. Works when iterations cost the same.

**Dynamic (via `set_parallel_chunksize`):** Threads take next available chunk after finishing. Requires `tbb` backend (OpenMP backend does NOT support dynamic chunking as of Numba 0.60).

### Problem: Cross-Gen Has Extreme Load Imbalance
CSC pairwise cross-generation has wildly varying per-pair cost:
- Column with 50 nonzeros x column with 50 nonzeros = trivial
- Column with 50,000 nonzeros x column with 50,000 nonzeros = expensive
- Static scheduling over raw pair IDs leaves threads idle while one gets the heavy pairs

### Recommended Fix: Cost-Sorted Worklist
```python
# Before entering the prange kernel:
# 1. Estimate cost per pair
costs = nnz_left[pairs[:, 0]] + nnz_right[pairs[:, 1]]
# 2. Sort descending (heavy pairs first = better load balance)
order = np.argsort(-costs)
pairs_sorted = pairs[order]
# 3. Feed sorted pairs to prange -- even static scheduling balances well
```

This is more reliable than dynamic chunking because:
- Works on ALL threading backends (not just TBB)
- No scheduler overhead
- Heavy pairs start first, light pairs fill in the tail

### Chunk Size Tuning (if using TBB)
```python
from numba import set_parallel_chunksize

# Test these values, benchmark end-to-end:
# 0 = default static (baseline)
# 4-16 = sweet spot for sparse workloads
# 1 = maximum balance but high overhead
# 32+ = approaches static behavior
set_parallel_chunksize(8)  # Good starting point
```

**Caveat:** Chunk size is thread-local and resets on entry to each parallel region. Set it immediately before the kernel call.

### Nested `prange` Warning
Numba serializes inner `prange` under an outer parallel loop. Never nest:
```python
# WRONG -- inner prange is serialized
@njit(parallel=True)
def bad(pairs, data):
    for p in prange(len(pairs)):      # outer parallel
        for i in prange(n_rows):       # inner SERIALIZED
            ...

# RIGHT -- single outer prange
@njit(parallel=True)
def good(pairs, data):
    for p in prange(len(pairs)):      # outer parallel
        for i in range(n_rows):        # inner sequential (fast)
            ...
```

---

## 5. Hybrid Representation: When to Switch from CSC to Bitsets

### Key Insight from Research
For truly binary features (0/1 only), a different representation can beat CSC intersection entirely for denser columns:

| Column Density | Best Representation | Why |
|---|---|---|
| < 1% nonzero | CSC sorted indices | Compact, cache-friendly for two-pointer |
| 1-10% nonzero | CSC or bitset (benchmark) | Crossover zone |
| > 10% nonzero | Bitset (64-bit blocks) | popcount intersection is SIMD-friendly |

### Bitset Intersection
```python
@njit(cache=True, fastmath=True)
def bitset_intersect_count(a, b, n_words):
    """Count intersection of two bitset arrays using popcount."""
    count = 0
    for i in range(n_words):
        count += _popcount64(a[i] & b[i])
    return count
```

This pattern:
- Is perfectly regular (no branches)
- LLVM vectorizes it trivially with AVX-512 VPOPCNTDQ (on Zen 4+)
- Processes 512 bits (64 rows) per instruction
- Is the ONLY way to actually exploit AVX-512 for binary feature intersection

### V3.3 Applicability
The current pipeline uses dense blocks for the actual multiply, so bitsets would only help if we moved to a fully sparse intersection path. For 15m/1h with memory pressure, a hybrid CSC+bitset approach could avoid dense conversion entirely.

---

## 6. Concrete Recommendations for V3.3

### Immediate (No Code Change Required)
1. **Add `fastmath=True` to `_parallel_cross_multiply`** -- safe for binary 0/1 values, may give 5-10% on the dense multiply step
2. **Verify TBB backend:** `python -c "from numba.core.runtime import _nrt_python; import numba; print(numba.config.THREADING_LAYER)"` -- if not TBB, set `NUMBA_THREADING_LAYER=tbb`

### Short-Term (GPU Fork Build)
3. **Add `-march=native` to `build_linux.sh`** -- one CMake flag change, unlocks AVX-512 for LightGBM histogram kernels
4. **Cost-sort the cross pair worklist** before entering prange loops -- eliminates load imbalance without backend dependency

### Medium-Term (If 15m Memory Remains Tight)
5. **Numba CSC intersection kernel** for direct sparse-to-sparse cross-gen (avoids dense conversion)
6. **Hybrid bitset path** for columns with >1% density (SIMD-friendly popcount intersection)
7. **PGO build of LightGBM** for 15m training (hours per fold, 10-30% speedup compounds)

### NOT Recommended
- Expecting AVX-512 to magically speed up the sparse merge loop (it won't -- branches dominate)
- Using `fastmath` on binarization kernels (they handle NaN explicitly)
- Over-tuning `prange` chunk size without first sorting the worklist by cost
- Nested `prange` (inner loop gets serialized silently)

---

## Sources

- Numba Performance Tips: https://numba.pydata.org/numba-doc/dev/user/performance-tips.html
- Numba Automatic Parallelization: https://numba.readthedocs.io/en/stable/user/parallel.html
- Numba `set_parallel_chunksize` (TBB only): https://github.com/numba/numba/issues/8284
- SIMD Autovectorization in Numba: https://tbetcke.github.io/hpc_lecture_notes/simd.html
- 100x Faster Sorted Array Intersections: https://softwaredoug.com/blog/2024/05/05/faster-intersect
- Fast Sorted-Set Intersection using SIMD (ADMS paper): https://www.adms-conf.org/p1-SCHLEGEL.pdf
- AMD EPYC Zen 5 AVX-512: https://www.amd.com/en/blogs/2025/leadership-hpc-performance-with-5th-generation-amd.html
- Zen 5 AVX-512 Frequency Behavior: https://chipsandcheese.com/p/zen-5s-avx-512-frequency-behavior
- SciPy `has_sorted_indices`: https://docs.scipy.org/doc/scipy-1.15.1/reference/generated/scipy.sparse.csc_array.has_sorted_indices.html
- LightGBM Benchmark Framework: https://microsoft.github.io/lightgbm-benchmark/lightgbm-benchmark-project/
- Numba Sparse Matrix Discussion: https://github.com/numba/numba-scipy/issues/29
- Finch Sparse Tensor Programming (2025): https://willowahrens.net/assets/documents/ahrens_finch_2025.pdf
- MySQL PGO/LTO Benchmark: http://smalldatum.blogspot.com/2024/10/the-impact-of-pgo-lto-and-more-for.html
