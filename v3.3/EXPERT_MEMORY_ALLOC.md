# EXPERT: Memory Allocator Optimization for Sparse Matrix ML Pipeline

**Context:** LightGBM pipeline with 2-10M sparse binary features (CSR format), repeated alloc/free of large matrices across CPCV folds, 512GB-1TB RAM cloud machines.

---

## 1. Allocator Ranking for This Workload

### Winner: tcmalloc (Google Temeraire)

tcmalloc's Temeraire design is purpose-built for this exact pattern: repeated large alloc/free cycles on high-RAM hosts. Key advantages:

- **Hugepage-aware allocation**: Temeraire packs allocations to maximize hugepage coverage, reducing TLB misses on large CSR traversals
- **Size-class separation**: Handles sub-hugepage (indptr arrays), mid-sized (indices arrays), and very large (data arrays) allocations through separate code paths -- maps directly to CSR's three-array structure
- **~50% pageheap savings** in fragmentation-heavy workloads (Google's own measurements)
- **Best throughput for allocations >1KB** in 2025 independent benchmarks

**Risk**: May retain RSS higher than jemalloc between training folds. Monitor with `/proc/self/status` VmRSS.

### Fallback: jemalloc

jemalloc wins if RSS return-to-OS is the operational pain point (e.g., running multiple TFs on the same machine):

- **Best RSS recovery** after freeing large objects -- the only tested allocator that consistently drops back near starting RSS
- Stable across mixed allocation sizes and thread counts
- Gives up some peak throughput on large allocations vs tcmalloc

### Third Option: mimalloc

- Strong for small temporary allocations (Python/C++ boundary objects)
- **Not recommended as default** -- falls off quickly for allocations >1KB in recent benchmarks
- Worth one benchmark pass but unlikely to beat tcmalloc/jemalloc for large CSR churn

| Allocator | Large Alloc Throughput | RSS Recovery | Fragmentation Resistance | Best For |
|-----------|----------------------|--------------|--------------------------|----------|
| tcmalloc  | Best                 | Good         | Best (Temeraire)         | Peak training speed |
| jemalloc  | Good                 | Best         | Good                     | Multi-task memory sharing |
| mimalloc  | Fair (>1KB)          | Good         | Fair                     | Small alloc heavy workloads |

---

## 2. Memory Fragmentation: Root Causes & Fixes

### Why RSS Stays High After Freeing CSR Matrices

A SciPy CSR matrix is three contiguous arrays (`data`, `indices`, `indptr`). Repeated create/destroy of varying-size matrices causes:

1. **Heap fragmentation**: Small live allocations pin freed regions, preventing return to OS
2. **Allocator arena bloat**: Multiple arenas (one per thread) each hold freed-but-unreturned pages
3. **LightGBM internal buffers**: `LGBM_BoosterFree()` releases memory to the allocator, not the OS (confirmed LightGBM issue #6421)

### High-Value Fixes (Priority Order)

#### A. Eliminate Repeated CSR Rebuilds
```python
# BAD: Rebuild CSR every fold
for fold in folds:
    X = build_csr_matrix(features)  # allocates ~40GB
    dataset = lgb.Dataset(X)
    # X stays in memory alongside dataset

# GOOD: Build once, reuse LightGBM binary format
X = build_csr_matrix(features)
dataset = lgb.Dataset(X, free_raw_data=True)  # frees X after binning
dataset.save_binary("train.bin")
# Subsequent folds:
dataset = lgb.Dataset("train.bin")  # fast binary load, no CSR rebuild
```

#### B. Compact CSR Dtypes
```python
# Binary features: use uint8 not float64 (8x memory savings on data array)
data = np.ones(nnz, dtype=np.uint8)

# indices: int32 if max_col < 2^31 (~2.1B) -- YES for 2-10M features
# indptr: int64 REQUIRED if nnz > 2^31 (your 15m/1h TFs hit this)
```

**CRITICAL**: Never downcast indptr to int32 when nnz > 2^31. This is already documented in `feedback_lgbm_int32_nnz_limit.md`.

#### C. Process Isolation for Memory Reclamation
```python
# Each fold trains in a subprocess -- OS reclaims ALL memory on exit
from multiprocessing import Process

def train_fold(fold_id, data_path):
    dataset = lgb.Dataset(data_path)  # binary load
    booster = lgb.train(params, dataset)
    booster.save_model(f"model_fold_{fold_id}.txt")
    # Process exits here -- OS reclaims everything

for fold_id in range(n_folds):
    p = Process(target=train_fold, args=(fold_id, "train.bin"))
    p.start()
    p.join()  # Wait, then OS reclaims all memory
```

This is the **single most reliable** anti-fragmentation pattern. No allocator tuning needed when the process dies.

#### D. Explicit Trim Between Stages (glibc fallback)
```python
import ctypes
import gc

def force_memory_release():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except:
        pass  # Not on glibc
```

Call between pipeline stages (after cross_gen, before training, between folds). This is a **mitigation, not a cure** -- process isolation is better.

---

## 3. jemalloc Arena Tuning

### Recommended Configuration for This Workload

```bash
# Set via environment variable before launching Python
export MALLOC_CONF="background_thread:true,narenas:4,dirty_decay_ms:5000,muzzy_decay_ms:5000,tcache_max:4096,metadata_thp:auto"
```

### Parameter Explanations

| Parameter | Value | Why |
|-----------|-------|-----|
| `background_thread:true` | Enable | Background purging thread -- reduces latency spikes from inline purging |
| `narenas:4` | 4 (not default 4x CPUs) | Fewer arenas = less fragmentation. LightGBM training is not malloc-contention-bound |
| `dirty_decay_ms:5000` | 5 seconds | Aggressive return-to-OS. Default 10s is too slow between fold boundaries |
| `muzzy_decay_ms:5000` | 5 seconds | Also purge "muzzy" (lazily freed) pages faster |
| `tcache_max:4096` | 4KB | Thread-local cache cap. Large CSR arrays bypass tcache anyway |
| `metadata_thp:auto` | Auto | Let jemalloc use transparent hugepages for its own metadata |

### Aggressive Memory-Return Profile (Multi-TF on Same Machine)

```bash
export MALLOC_CONF="background_thread:true,narenas:1,dirty_decay_ms:1000,muzzy_decay_ms:0,tcache_max:1024"
```

This trades some CPU for minimal RSS footprint -- useful when running 1h after 4h on the same vast.ai instance.

### CPU-Focused Profile (Single TF, Maximum Speed)

```bash
export MALLOC_CONF="background_thread:true,metadata_thp:auto,dirty_decay_ms:30000,muzzy_decay_ms:30000"
```

Keep freed pages around longer to avoid re-faulting during next fold's allocation.

---

## 4. Transparent Huge Pages (THP) Configuration

### Recommended Setup

```bash
# Set THP to madvise mode (opt-in per allocation, not globally)
echo madvise > /sys/kernel/mm/transparent_hugepage/enabled

# Set defrag to defer+madvise (best balance)
echo defer+madvise > /sys/kernel/mm/transparent_hugepage/defrag

# Tune khugepaged scan interval (reduce CPU overhead)
echo 30000 > /sys/kernel/mm/transparent_hugepage/khugepaged/scan_sleep_millisecs
```

### Why madvise, NOT always

- `always` mode: khugepaged promotes ALL 4KB pages to 2MB hugepages, including sparse regions. A matrix that touches only a few bytes per 2MB range gets **512x memory bloat**
- `madvise` mode: Only regions explicitly marked with `MADV_HUGEPAGE` get promotion. Safe for sparse workloads

### Why defer+madvise for defrag

- Direct reclaim + compaction only for `MADV_HUGEPAGE` regions
- Background kswapd/kcompactd for everything else
- No stalls on non-critical allocations

### Applying MADV_HUGEPAGE to CSR Arrays

tcmalloc (Temeraire) does this automatically for large allocations. For jemalloc or manual control:

```python
import ctypes
import ctypes.util

MADV_HUGEPAGE = 14

def enable_hugepages_for_array(arr):
    """Mark a numpy array's backing memory for hugepage promotion."""
    libc = ctypes.CDLL(ctypes.util.find_library("c"))
    addr = arr.ctypes.data
    size = arr.nbytes
    # Align to page boundary
    page_size = 4096
    aligned_addr = addr & ~(page_size - 1)
    aligned_size = size + (addr - aligned_addr)
    libc.madvise(ctypes.c_void_p(aligned_addr), ctypes.c_size_t(aligned_size), MADV_HUGEPAGE)

# Apply to CSR backing arrays after construction
X_csr = build_csr_matrix(features)
enable_hugepages_for_array(X_csr.data)
enable_hugepages_for_array(X_csr.indices)
# indptr is small, skip it
```

### THP + Allocator Interaction

| Allocator | THP Behavior |
|-----------|-------------|
| tcmalloc (Temeraire) | **Automatic** hugepage management. Best THP integration. No madvise calls needed. |
| jemalloc | Set `metadata_thp:auto` in MALLOC_CONF. Does NOT auto-request hugepages for user allocations. |
| glibc malloc | No hugepage awareness. Manual madvise required. |

---

## 5. LD_PRELOAD Deployment

### tcmalloc

```bash
# Install
apt-get install -y google-perftools libgoogle-perftools-dev
# OR build from source for Temeraire (recommended):
# git clone https://github.com/google/tcmalloc && cd tcmalloc && bazel build //tcmalloc:libtcmalloc.so

# Deploy
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4
python cloud_run_tf.py --symbol BTC --tf 1h
```

### jemalloc

```bash
# Install
apt-get install -y libjemalloc-dev

# Deploy with tuning
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
export MALLOC_CONF="background_thread:true,narenas:4,dirty_decay_ms:5000,muzzy_decay_ms:5000,metadata_thp:auto"
python cloud_run_tf.py --symbol BTC --tf 1h
```

### Verification

```bash
# Confirm allocator is loaded
python -c "
import ctypes, ctypes.util
try:
    je = ctypes.CDLL('libjemalloc.so.2')
    print('jemalloc loaded')
except:
    pass
try:
    tc = ctypes.CDLL('libtcmalloc.so.4')
    print('tcmalloc loaded')
except:
    pass
"

# Monitor RSS during training
watch -n 5 'grep -E "VmRSS|VmHWM" /proc/$(pgrep -f cloud_run_tf)/status'
```

---

## 6. Benchmark Protocol

Run this before committing to an allocator:

```bash
#!/bin/bash
# benchmark_allocators.sh

ALLOCATORS=(
    "glibc|"
    "tcmalloc|/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4"
    "jemalloc|/usr/lib/x86_64-linux-gnu/libjemalloc.so.2"
)

for entry in "${ALLOCATORS[@]}"; do
    IFS='|' read -r name lib <<< "$entry"
    echo "=== Testing $name ==="

    if [ -n "$lib" ]; then
        export LD_PRELOAD="$lib"
    else
        unset LD_PRELOAD
    fi

    # Run 3 consecutive fold cycles to expose fragmentation
    /usr/bin/time -v python -c "
import scipy.sparse as sp
import numpy as np
import gc, resource

for cycle in range(3):
    # Simulate CSR build + LightGBM dataset + free
    rows, cols, nnz = 50000, 2000000, 100000000
    data = np.ones(nnz, dtype=np.uint8)
    indices = np.random.randint(0, cols, nnz, dtype=np.int32)
    indptr = np.sort(np.random.randint(0, nnz, rows+1).astype(np.int64))
    indptr[0] = 0; indptr[-1] = nnz
    X = sp.csr_matrix((data, indices, indptr), shape=(rows, cols))

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f'Cycle {cycle}: peak RSS = {peak // 1024} MB')

    del X, data, indices, indptr
    gc.collect()

    post = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f'Cycle {cycle}: post-free RSS = {post // 1024} MB')
" 2>&1 | tee "bench_${name}.log"

    unset LD_PRELOAD
    echo ""
done
```

Measure: **peak RSS**, **post-free RSS**, **wall time per cycle**, **RSS delta across 3 cycles** (fragmentation indicator).

---

## 7. Recommended Configuration for cloud_run_tf.py

Add to the cloud deployment template (before Python launch):

```bash
# --- Memory Allocator Setup ---
# Install tcmalloc (primary) and jemalloc (fallback)
apt-get install -y google-perftools libjemalloc-dev 2>/dev/null

# Use tcmalloc by default (Temeraire hugepage-aware allocator)
if [ -f /usr/lib/x86_64-linux-gnu/libtcmalloc.so.4 ]; then
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4
    echo "ALLOCATOR: tcmalloc (Temeraire)"
elif [ -f /usr/lib/x86_64-linux-gnu/libjemalloc.so.2 ]; then
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
    export MALLOC_CONF="background_thread:true,narenas:4,dirty_decay_ms:5000,muzzy_decay_ms:5000,metadata_thp:auto"
    echo "ALLOCATOR: jemalloc (tuned)"
else
    echo "ALLOCATOR: glibc (default -- install tcmalloc for better performance)"
fi

# --- THP Setup ---
echo madvise > /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null
echo defer+madvise > /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null
echo 30000 > /sys/kernel/mm/transparent_hugepage/khugepaged/scan_sleep_millisecs 2>/dev/null
```

---

## 8. Decision Matrix

| Scenario | Allocator | THP | Process Isolation |
|----------|-----------|-----|-------------------|
| Single TF, max speed | tcmalloc | madvise + defer+madvise | Optional |
| Multiple TFs, same machine | jemalloc (aggressive decay) | madvise + defer+madvise | Required |
| 15m/1h (huge matrices, OOM risk) | tcmalloc | madvise + defer+madvise | Required (subprocess per fold) |
| 1w/1d (smaller matrices) | tcmalloc or jemalloc | madvise | Optional |

---

## Sources

- [tcmalloc Temeraire: Hugepage-Aware Allocator](https://google.github.io/tcmalloc/temeraire.html)
- [Allocator Benchmarks 2025](https://dev.to/frosnerd/libmalloc-jemalloc-tcmalloc-mimalloc-exploring-different-memory-allocators-4lp3)
- [LightGBM Memory Not Returned (Issue #6421)](https://github.com/microsoft/LightGBM/issues/6421)
- [LightGBM Python-package Documentation](https://lightgbm.readthedocs.io/en/stable/Python-Intro.html)
- [jemalloc TUNING.md](https://github.com/jemalloc/jemalloc/blob/dev/TUNING.md)
- [Linux THP Kernel Documentation](https://www.kernel.org/doc/html/latest/admin-guide/mm/transhuge.html)
- [THP madvise vs always (Phoronix)](https://www.phoronix.com/review/thp-madvise-always)
- [SciPy CSR int64 indptr (Issue #16774)](https://github.com/scipy/scipy/issues/16774)
- [malloc_trim for Python](https://www.softwareatscale.dev/p/run-python-servers-more-efficiently)
- [Joblib Large Memory Growth (Issue #781)](https://github.com/joblib/joblib/issues/781)
