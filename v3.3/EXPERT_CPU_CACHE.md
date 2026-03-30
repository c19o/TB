# EXPERT: CPU Cache & NUMA Optimization for Sparse Cross-Gen

## Target Hardware: AMD EPYC 9654 (96 cores, 12 CCDs, Zen 4)

### Architecture Facts
- **12 CCDs**, each with 1 CCX of 8 cores
- **32 MB shared L3 per CCD** (384 MB total L3)
- **1 MB private L2 per core** (96 MB total L2)
- **12 memory channels** per socket
- **Cache line**: 64 bytes
- BIOS can expose 1 to 12 NUMA domains
- "L3 Cache as NUMA Domain" makes each CCD its own NUMA node

---

## 1. NUMA Topology & BIOS Settings

### Recommended BIOS Configuration
| Setting | Value | Rationale |
|---|---|---|
| NPS (Nodes Per Socket) | **4** | Divides socket into 4 quadrants (3 CCDs + 3 memory controllers each) |
| L3 Cache as NUMA Domain | **Enabled** | Exposes each CCD as separate NUMA node for fine-grained binding |
| SMT | **Off** for cross-gen | Memory-bound sparse kernels gain more from per-thread cache/bandwidth than from sibling threads |
| CPU Governor | **performance** | No frequency scaling during hot path |
| NUMA Balancing | **Disabled** | Prevents OS from migrating pages away from manually-pinned workers |
| Deterministic Performance | **Enabled** | Consistent clock behavior |

### Why NPS=4 + L3-as-NUMA
- NPS=4 gives 4 quadrants with local memory controllers
- L3-as-NUMA on top of that gives 12 locality domains matching the 12 physical L3 caches
- Sparse CSC traversal benefits from keeping hot working sets inside one 32 MB L3 island
- Benchmark NPS=4+L3NUMA vs plain NPS=4 vs NPS=1 to confirm

---

## 2. Optimal Thread Count Per CCD

### Key Finding: More Cores != Better for Sparse Traversal

Sparse binary feature intersection is **memory-bound and latency-bound**, not compute-bound. Adding threads saturates memory bandwidth or amplifies LLC misses before arithmetic becomes the bottleneck.

### Recommended Testing Matrix
| Threads | CCD Coverage | Use Case |
|---|---|---|
| **8** | 1 CCD | Baseline -- fits entirely in one L3 |
| **16** | 2 CCDs | First scale-up test |
| **24** | 1 NPS quadrant (3 CCDs) | Memory-channel saturation test |
| **48** | Half socket | Bandwidth ceiling test |
| **96** | Full socket | Verify scaling vs 48 |

**Always use multiples of 8** so each step maps cleanly to whole CCDs.

### Per-CCD Worker Design
- **Target: 4-8 threads per CCD** for sparse traversal
- 8 threads = full CCD, maximizes L3 sharing but contends on memory controller
- 4 threads = half CCD, more L2 per thread (2 MB effective), less bandwidth pressure
- Start with 8, drop to 4 if LLC miss rate is high

---

## 3. Cache-Friendly CSC Column Access

### L2-First Design (1 MB per core)

The active intersection state for each worker must fit in L2:
- `indptr` metadata for the active tile
- `indices` blocks for candidate columns being compared
- Thread-local accumulators and scratch buffers

**Mental model:**
- **L2** (1 MB): Current tile metadata + front of several posting lists
- **L3** (32 MB): Read-mostly CSC arrays shared among 8 CCD cores
- **DRAM**: Only feeds new tiles, not every compare step

### Column Blocking Strategy

Do NOT iterate all 2-10M columns globally. Block them:

```
# Pseudocode: Blocked CSC traversal
BLOCK_SIZE = estimate_columns_for_L2()  # ~500-2000 cols depending on avg nnz

for block_start in range(0, n_features, BLOCK_SIZE):
    block_end = min(block_start + BLOCK_SIZE, n_features)
    # All columns in this block share nearby indptr/indices memory
    process_column_block(csc_matrix, block_start, block_end)
```

### Why Blocking Matters
- Raw global CSC gives poor locality when cross-gen touches columns from all over memory
- Column blocking keeps `indptr` and `indices` ranges contiguous in cache
- Each block reuses a narrower section of weights and output accumulators
- Hardware prefetchers handle sequential block walks well but fail on giant monolithic posting lists with sparse jumps

---

## 4. Hardware Prefetch in Numba for CSC Access

### What Works
- **Sequential `indices` scanning**: Hardware prefetchers (including Zen 4's 2D stride prefetcher) handle regular stride patterns well
- **Block-sequential column access**: Processing columns in contiguous blocks allows L2/L3 prefetch to work
- **Merge-scan patterns**: Linear merge through two sorted index arrays is prefetch-friendly

### What Does NOT Work
- **Random binary search probes**: Pointer-chasing through sparse indices defeats hardware prefetch
- **Galloping search jumps**: Exponential search with irregular strides confuses prefetchers
- **Cross-column random access**: Touching accumulators at arbitrary row positions

### Software Prefetch Guidance
- Use software prefetch ONLY for predictable next-block access in merge scans
- Do NOT prefetch inside irregular galloping branches
- Numba does not expose `__builtin_prefetch` directly; use `ctypes` or restructure access patterns instead
- **Best approach**: Make access patterns sequential rather than trying to prefetch irregular ones

### Intersection Dispatch by NNZ Ratio

| Case | Best Kernel | Cache Behavior |
|---|---|---|
| `nnz_a ~ nnz_b` | Branch-light merge | Sequential scans, prefetch-friendly |
| `nnz_big / nnz_small > 8-10x` | Galloping/exponential search | Less predictable, keep small list in L2 |
| Very dense features | Bitset/bitmap tile | Wordwise AND, excellent cache use |

```python
# Dispatch logic
@njit
def intersect_dispatch(indices_a, indices_b):
    ratio = len(indices_a) / max(len(indices_b), 1)
    if 0.1 < ratio < 10.0:
        return merge_intersect(indices_a, indices_b)
    elif ratio >= 10.0:
        return galloping_intersect(indices_a, indices_b)  # a is bigger
    else:
        return galloping_intersect(indices_b, indices_a)  # b is bigger
```

---

## 5. Tiling Pair Processing for 12 CCDs

### Sharding Strategy

Partition the feature space into **12 locality-aware shards** (one per CCD):

```
Total features: N (2-10M)
Shard size: N / 12
CCD 0: features [0, N/12)
CCD 1: features [N/12, 2N/12)
...
CCD 11: features [11N/12, N)
```

### Per-CCD Resource Budget
- **32 MB L3** shared across 8 cores in the CCD
- Each shard's CSC slice (indptr + indices + data) should target < 24 MB to leave room for accumulators
- For 2M features / 12 CCDs = ~167K features per shard
- With avg 50 nnz per feature: 167K * 50 * 4 bytes (int32) = ~33 MB -- tight but workable
- For 10M features: need sub-sharding within each CCD

### Work Balancing

Do NOT use static `prange` over columns -- later iterations become lighter, causing idle cores.

**Cost-balanced tiling:**
```python
# Pre-compute cost per column pair
cost[i,j] = min(nnz_i, nnz_j)  # for galloping
# OR
cost[i,j] = nnz_i + nnz_j      # for merge

# Group into tiles with ~equal total cost
# Assign tiles to CCDs respecting column locality
```

### CCD-Pinned Worker Launch Pattern (Linux)

```bash
# Discover CCD core groups
for i in /sys/devices/system/cpu/cpu*/cache/index3/shared_cpu_list; do
    cat $i
done | sort -u

# Example output for 96-core EPYC 9654:
# 0-7      (CCD 0)
# 8-15     (CCD 1)
# 16-23    (CCD 2)
# ...
# 88-95    (CCD 11)

# Launch one worker per CCD
for ccd in 0 1 2 3 4 5 6 7 8 9 10 11; do
    start=$((ccd * 8))
    end=$((start + 7))
    numactl --physcpubind=${start}-${end} --localalloc \
        python cross_gen_shard.py --shard=$ccd --total-shards=12 &
done
```

### Python Multiprocessing NUMA Binding

```python
import multiprocessing as mp
import subprocess

def launch_numa_worker(ccd_id, shard_data_path, result_queue):
    """Launch a worker pinned to a specific CCD."""
    start_core = ccd_id * 8
    end_core = start_core + 7

    # Use numactl to pin the subprocess
    cmd = [
        'numactl',
        f'--physcpubind={start_core}-{end_core}',
        '--localalloc',
        'python', 'cross_gen_worker.py',
        '--shard', str(ccd_id),
        '--data', shard_data_path
    ]
    subprocess.run(cmd)

# Alternative: os.sched_setaffinity within the worker process
import os
def worker_init(ccd_id):
    cores = list(range(ccd_id * 8, ccd_id * 8 + 8))
    os.sched_setaffinity(0, set(cores))
```

---

## 6. First-Touch Memory Allocation

### Critical Rule
Linux allocates physical pages on first access, local to the CPU that faults them in. If you initialize arrays serially from one thread, ALL pages land on that thread's NUMA node.

### Correct Pattern
```python
# WRONG: Serial init places all pages on node 0
data = np.zeros(huge_size)  # All pages on whoever calls this

# RIGHT: First-touch under pinned parallel layout
@njit(parallel=True)
def first_touch_init(arr, n_shards, shard_size):
    for shard in prange(n_shards):
        start = shard * shard_size
        end = min(start + shard_size, len(arr))
        for i in range(start, end):
            arr[i] = 0  # Pages fault onto the core running this shard
```

Initialize large CSC arrays, row-index arrays, and per-thread work buffers **under the same pinned parallel layout** used during compute.

---

## 7. False Sharing Prevention

### The Problem
Multiple `prange` workers updating adjacent counters/accumulators in a shared array cause cache line bouncing (64 bytes = 16 float32 or 8 float64 values).

### Solution: Padded Thread-Local Buffers

```python
# WRONG: Adjacent threads share cache lines
shared_output = np.zeros(n_outputs)  # Workers i and i+1 bounce lines

# RIGHT: Padded per-thread buffers
CACHE_LINE = 64
PAD = CACHE_LINE // 8  # 8 for float64
per_thread = np.zeros((n_threads, n_outputs + PAD))

@njit(parallel=True)
def compute(per_thread, ...):
    for t in prange(n_threads):
        local = per_thread[t]  # No false sharing -- rows are padded
        # ... accumulate into local ...

    # Reduce once after parallel section
    result = np.zeros(n_outputs)
    for t in range(n_threads):
        result += per_thread[t, :n_outputs]
```

### Rules
1. Thread-private accumulators, never shared atomic writes
2. Pad buffers to 64-byte alignment between threads
3. Reduce ONCE after the parallel loop
4. Coarse-grained ownership: one thread owns one output shard for the entire inner loop

---

## 8. Numba Threading Layer Configuration

### Required Environment Variables

```bash
# Use OpenMP backend (required for affinity controls)
export NUMBA_THREADING_LAYER=omp

# Thread binding
export OMP_PROC_BIND=close        # Keep threads near each other (cache sharing)
export OMP_PLACES=ll_caches       # Bind to last-level cache groups (= CCDs on EPYC)

# OR for bandwidth-bound independent work:
export OMP_PROC_BIND=spread       # Spread across NUMA nodes
export OMP_PLACES=numa_domains    # One thread per NUMA domain

# Thread count (match to CCD assignment)
export OMP_NUM_THREADS=8          # Per-CCD worker
export NUMBA_NUM_THREADS=8        # Numba equivalent
```

### Caution
- Numba `prange` may use fewer threads than requested (known issue in some OpenMP layer configs)
- Verify actual thread count with a simple parallel test before production runs
- `prange` scaling often flattens or regresses when overhead, reductions, or memory traffic dominate
- Treat thread count as an empirical tuning knob, not monotonic accelerator

---

## 9. Huge Pages / THP Guidance

### Be Careful with Sparse Data
- THP (Transparent Huge Pages) can inflate memory footprint for sparsely-addressed data
- Large sparse mappings where VSZ >> RSS cause premature reclaim or remote-node fallback
- On EPYC, node spillover congests Infinity Fabric links

### Recommendation
1. **Benchmark THP on vs off** for your CSC arrays
2. If VSZ is much larger than RSS, or access is sparse/noncontiguous: **disable THP**
3. If THP measurably lowers TLB pressure without causing node spillover: keep it
4. For dense accumulators and thread-local buffers: THP is usually fine

```bash
# Disable THP for the cross-gen process
echo madvise > /sys/kernel/mm/transparent_hugepage/enabled
# Then in Python: madvise(MADV_NOHUGEPAGE) on sparse arrays
```

---

## 10. Index Compression: uint32

### Halve Cache Footprint
If row IDs fit in 32 bits (< 4.3B rows -- always true for your data):

```python
# Convert indices to uint32
csc.indices = csc.indices.astype(np.uint32)
# indptr can stay int64 for safety (NNZ > 2^31 possible)
```

- Halves `indices` array cache footprint
- More indices fit per cache line (16 vs 8)
- Directly improves L2/L3 residency for intersection kernels

---

## 11. Complete Optimization Recipe

### Phase 1: BIOS & OS (do once)
1. Set NPS=4, L3-as-NUMA enabled, SMT off, performance governor
2. Disable NUMA auto-balancing: `echo 0 > /proc/sys/kernel/numa_balancing`
3. Discover CCD topology: `lscpu --all --extended` and `numactl -H`

### Phase 2: Data Layout
1. Convert `indices` to `uint32`
2. Keep `indptr` as `int64` (NNZ > 2^31 protection)
3. Sort features into locality-aware shards (12 shards for 12 CCDs)
4. Build blocked CSC view within each shard

### Phase 3: Worker Architecture
1. Launch 12 process-level workers (one per CCD)
2. Pin each with `numactl --physcpubind=N-N+7 --localalloc`
3. First-touch allocate all shard data under the pinned binding
4. Each worker uses `prange` with 8 threads (one CCD)
5. Set `NUMBA_THREADING_LAYER=omp`, `OMP_PROC_BIND=close`, `OMP_PLACES=ll_caches`

### Phase 4: Kernel Design
1. Ratio-dispatch: merge for similar nnz, galloping for skewed
2. Column-blocked traversal: tile columns to fit working set in L2
3. Thread-private padded accumulators (64-byte aligned)
4. Single reduction pass after parallel section
5. Cost-balanced tile assignment (not static column split)

### Phase 5: Benchmark & Tune
Measure for each configuration:
- **LLC miss rate** (`perf stat -e LLC-load-misses`)
- **NUMA remote traffic** (`perf stat -e node-load-misses`)
- **Memory bandwidth** (`perf stat -e bus-cycles`)
- **End-to-end epoch time** (the real metric)

Test matrix (priority order):
1. 12 CCD-pinned workers vs 4 quadrant workers vs 1 flat worker
2. SMT off vs on
3. Column-blocked vs full-sweep CSC
4. Merge-only vs ratio-dispatched merge+galloping
5. THP on vs off for CSC arrays
6. 4 vs 8 threads per CCD

### Fallback Strategy
If 12 CCD-pinned workers underperform due to cross-shard synchronization or duplicated state:
- Fall back to **4 workers pinned by NPS quadrant** (24 cores each)
- Larger shards, less coordination overhead
- 3 CCDs share one memory controller per quadrant = natural locality boundary

---

## 12. Expected Gains

| Optimization | Expected Impact | Confidence |
|---|---|---|
| CCD-pinned workers + first-touch | 2-4x for NUMA-heavy workloads | High |
| Column blocking (L2-fit tiles) | 1.5-3x less LLC misses | High |
| False sharing elimination | 1.2-2x on shared accumulator paths | High |
| uint32 indices | 1.3-1.5x better cache residency | High |
| Ratio-dispatched intersection | 1.5-2x for skewed nnz distributions | Medium |
| SMT off | 1.0-1.2x (workload dependent) | Medium |
| THP tuning | 1.0-1.3x (depends on access pattern) | Low-Medium |

**Combined potential: 3-8x improvement** over naive `prange` with default OS scheduling on the same hardware.

---

## Sources
- AMD EPYC 9654 Product Page & Architecture White Paper
- AMD BIOS & Workload Tuning Guide (9004 Series)
- SUSE EPYC 9004 Linux Optimization Guide (SBP-AMD-EPYC-4-SLES15SP4)
- NUMA-Aware Optimization of Sparse Matrix-Vector Multiplication (Yu et al., 2020)
- "Generating Data Locality to Accelerate Sparse Matrix-Matrix Multiplication" (arXiv:2501.07056)
- "Faster Set Intersection with SIMD instructions by Reducing Branch Mispredictions" (VLDB 2015)
- "SIMD Compression and the Intersection of Sorted Integers" (Lemire et al., 2014)
- "Multi-Strided Access Patterns to Boost Hardware Prefetching" (arXiv:2412.16001)
- "Evaluating the impact of the L3 cache size of AMD EPYC CPUs" (arXiv:2505.17934)
- Numba Threading Layer Documentation
- OpenMP Affinity (OMP_PROC_BIND / OMP_PLACES) Specification
- Red Hat Performance Tuning Guide (CPU chapter)
- Intel Thread Affinity Interface Documentation
