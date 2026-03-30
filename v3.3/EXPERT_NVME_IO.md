# EXPERT: NVMe I/O Pipeline for Sparse Feature Matrix

Research date: 2026-03-30
Sources: Perplexity (internet focus), NumPy/SciPy docs, Linux kernel docs, NVMe benchmark papers

---

## 1. Buffered I/O vs Direct I/O for Memmap

### Verdict: Buffered I/O (default memmap) wins for this pipeline

NumPy memmap uses OS page cache via `mmap()` -- it is inherently buffered I/O. Direct I/O (`O_DIRECT`) bypasses page cache entirely and requires custom aligned buffer management.

**When buffered wins (our case):**
- Feature shards read mostly sequentially per batch
- Restarts benefit from warm page cache (no re-read from disk)
- Locality can be improved by grouping co-accessed features
- NVMe SSDs reduce page fault penalty vs spinning disks

**When direct I/O would win (not our case):**
- Working set vastly exceeds RAM with random access patterns
- Page cache thrashing from concurrent processes competing for cache
- Database-style random lookups where cache eviction is unpredictable

**NVMe benchmark data (2025):**
- Random direct `pread()` on Samsung PM9A1: flat latency 512B-4KB
- Throughput ramps: ~1.0 GB/s at 128KB blocks, ~3.1 GB/s at 1MB blocks
- 128KB is the diminishing-returns inflection point for sequential NVMe reads
- Tiny random reads severely underutilize NVMe -- batch/cluster reads instead

**Key rule:** On NVMe, shaping data for locality and reading in bigger chunks matters more than switching I/O mode.

---

## 2. Storage Format: NPZ vs Separate NPY Files

### Verdict: Store CSR component arrays as separate .npy files for runtime; keep NPZ for archival only

**The NPZ problem:**
- `scipy.sparse.load_npz` loads from a NumPy `.npz` archive -- not memory-mappable
- `save_npz` defaults to `compressed=True` -- extra CPU decompression on every load
- `.npz` is an archive container, not a raw array layout for zero-copy reopening
- NumPy's `NpzFile` does not expose members as memmaps

**Optimal CSR storage layout:**

```
checkpoint_1h_fold0/
    data.npy        # CSR.data array (float32 or int8 for binary)
    indices.npy     # CSR.indices array (int32 or int64)
    indptr.npy      # CSR.indptr array (int64 for >2^31 NNZ)
    metadata.json   # shape, dtype, format, feature_count, timestamp
```

**Loading pattern:**

```python
import numpy as np
from scipy.sparse import csr_matrix
import json

def load_csr_memmap(checkpoint_dir, mmap_mode='r'):
    """Load CSR from separate .npy files with memory mapping."""
    with open(f"{checkpoint_dir}/metadata.json") as f:
        meta = json.load(f)

    data = np.load(f"{checkpoint_dir}/data.npy", mmap_mode=mmap_mode)
    indices = np.load(f"{checkpoint_dir}/indices.npy", mmap_mode=mmap_mode)
    indptr = np.load(f"{checkpoint_dir}/indptr.npy", mmap_mode=mmap_mode)

    return csr_matrix(
        (data, indices, indptr),
        shape=tuple(meta['shape'])
    )
```

**Saving pattern:**

```python
import numpy as np
import json, os, tempfile, shutil

def save_csr_atomic(matrix, checkpoint_dir):
    """Atomic CSR checkpoint -- crash-safe via temp dir + rename."""
    tmp_dir = tempfile.mkdtemp(dir=os.path.dirname(checkpoint_dir))
    try:
        np.save(f"{tmp_dir}/data.npy", matrix.data)
        np.save(f"{tmp_dir}/indices.npy", matrix.indices)
        np.save(f"{tmp_dir}/indptr.npy", matrix.indptr)

        meta = {
            'shape': list(matrix.shape),
            'dtype': str(matrix.data.dtype),
            'format': 'csr',
            'nnz': int(matrix.nnz),
        }
        with open(f"{tmp_dir}/metadata.json", 'w') as f:
            json.dump(meta, f)

        # Atomic swap -- if crash happens mid-write, old checkpoint survives
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
        os.rename(tmp_dir, checkpoint_dir)
    except:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
```

**Format decision table:**

| Use Case | Format | Why |
|---|---|---|
| Fast training restart | Separate `.npy` + mmap | Zero-copy, page-cache friendly |
| Long-term snapshot | `save_npz(compressed=True)` | Portable, smaller on disk |
| Cross-machine transfer | `save_npz(compressed=True)` | Single file, easy to SCP |
| Local checkpoint | Separate `.npy` + metadata JSON | Atomic swap, fast recovery |
| Cloud download artifact | `.npz` compressed | Bandwidth-limited transfer |

---

## 3. Linux Kernel Tuning for NVMe + Memmap

### 3a. Readahead (`blockdev --setra`)

Default is often 128KB -- too small for sequential sparse matrix streaming.

**Tuning protocol:**

```bash
# Check current readahead (in 512-byte sectors)
blockdev --getra /dev/nvme0n1

# Start at 1MB (2048 sectors of 512B each)
blockdev --setra 2048 /dev/nvme0n1

# Test range: 1MB / 2MB / 4MB
# blockdev --setra 2048   # 1 MB
# blockdev --setra 4096   # 2 MB
# blockdev --setra 8192   # 4 MB
```

**Guidance:**
- Start at **1MB** (2048 sectors)
- Sparse CSR access is "mostly sequential with skips" -- oversized readahead overfetches useless pages
- Benchmark with `fio` on representative files at each setting
- 4MB is the upper bound for mixed sequential/skip patterns
- For pure sequential streaming, 4MB or even 8MB can help

### 3b. Dirty Page Settings

Training is **read-dominated** -- dirty settings matter less than readahead.

```bash
# Conservative defaults -- avoid writeback stalls during training
sysctl -w vm.dirty_ratio=10
sysctl -w vm.dirty_background_ratio=5

# For machines with >128GB RAM, use byte-based limits instead:
sysctl -w vm.dirty_bytes=1073741824        # 1 GB
sysctl -w vm.dirty_background_bytes=268435456  # 256 MB
```

**Key point:** Ratio-based and byte-based knobs are mutually exclusive. On large-RAM machines, byte-based is more predictable.

### 3c. Transparent Huge Pages (THP)

```bash
# Use madvise, NOT always
echo madvise > /sys/kernel/mm/transparent_hugepage/enabled

# Verify
cat /sys/kernel/mm/transparent_hugepage/enabled
```

**Why `madvise` not `always`:**
- THP currently applies to **anonymous memory and tmpfs/shmem**, not file-backed mmap
- THP helps LightGBM's in-memory working structures, not the memmap itself
- `always` over-allocates when processes map large regions but sparsely touch them
- `madvise` lets LightGBM opt in for dense hot buffers only
- Avoids compaction stalls from blanket THP promotion

**Monitoring THP effectiveness:**

```bash
grep -E 'thp_fault_alloc|thp_fault_fallback|compact_stall|AnonHugePages' /proc/vmstat /proc/meminfo
```

### 3d. NUMA Binding

Critical for multi-socket machines (most cloud GPU boxes).

```bash
# Find which NUMA node owns the NVMe
cat /sys/block/nvme0n1/device/numa_node

# Bind training to local NUMA node (example: node 0)
numactl --cpunodebind=0 --membind=0 python train.py

# If dataset exceeds one node's RAM, test interleave:
numactl --interleave=all python train.py
```

**Rules:**
- Start with strict local binding (`--cpunodebind` + `--membind`)
- Only use `--interleave` if local node runs out of memory headroom
- Remote NUMA accesses can erase all gains from faster storage

---

## 4. Parallel File Loading

### ThreadPoolExecutor vs ProcessPoolExecutor

**ThreadPoolExecutor (preferred for I/O-bound NPY loading):**
- `np.load()` releases the GIL during the actual disk read
- No IPC overhead, no pickling, shared page cache
- 4-8 threads typically saturates a single NVMe (3-7 GB/s)
- Best for loading separate .npy component files in parallel

**ProcessPoolExecutor (use only for CPU-bound decompression):**
- Required if loading compressed `.npz` (decompression is CPU-bound, holds GIL)
- IPC overhead from pickling results back to main process
- Each process gets its own page cache view (wasteful for shared data)

**Recommended loading pattern:**

```python
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.sparse import csr_matrix
import json

def load_fold_parallel(fold_dirs, max_workers=4):
    """Load multiple fold checkpoints in parallel."""
    def _load_one(checkpoint_dir):
        with open(f"{checkpoint_dir}/metadata.json") as f:
            meta = json.load(f)
        data = np.load(f"{checkpoint_dir}/data.npy")
        indices = np.load(f"{checkpoint_dir}/indices.npy")
        indptr = np.load(f"{checkpoint_dir}/indptr.npy")
        return csr_matrix((data, indices, indptr), shape=tuple(meta['shape']))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(_load_one, fold_dirs))
```

### Saturating NVMe Bandwidth

A single NVMe drive typically provides:
- Sequential read: 3-7 GB/s (PCIe Gen4 x4)
- Random 4K read: 500K-1M IOPS
- Queue depth matters: single-threaded `read()` underutilizes the device

**To saturate NVMe:**
- Use 4-8 concurrent readers (threads) with large reads (128KB+ chunks)
- Memmap naturally benefits from kernel readahead merging nearby faults
- Avoid many tiny random reads -- batch into contiguous regions
- Pre-sort access patterns by file offset when possible

### io_uring (Linux 5.1+, advanced)

- Kernel-side async I/O with minimal syscall overhead
- Python support via `liburing` bindings (still immature as of 2025)
- Best gains for high-IOPS random patterns, less impactful for sequential
- **Not recommended** for this pipeline -- ThreadPoolExecutor + memmap is simpler and sufficient

---

## 5. Checkpoint Recovery Protocol

### Atomic Checkpoint Strategy

```
checkpoints/
    1h_fold0/           # Complete checkpoint (verified)
    1h_fold0.tmp/       # In-progress write (ignored on recovery)
    1h_fold1/           # Complete checkpoint
    manifest.json       # Points to latest valid checkpoints
```

**Recovery rules:**
1. Write new checkpoint to `.tmp` directory
2. Flush all files: `os.fsync(fd)` on each .npy file
3. Atomic rename `.tmp` -> final name (rename is atomic on same filesystem)
4. Update manifest.json (also atomic via write-tmp-then-rename)
5. On crash recovery: ignore any `.tmp` directories, load from manifest

**Memmap flush semantics:**
- `memmap.flush()` writes modified pages back to disk but does NOT guarantee durability
- For true durability: `flush()` then `os.fsync()` on the underlying file descriptor
- NumPy provides no API to close the underlying mmap directly -- let GC handle it
- For checkpoint writes, use `np.save()` (not memmap mutation) for crash safety

### Recovery loader:

```python
import os, json, glob

def find_latest_checkpoint(checkpoint_root, timeframe, fold):
    """Find the latest valid (non-tmp) checkpoint."""
    pattern = f"{checkpoint_root}/{timeframe}_fold{fold}"
    if os.path.isdir(pattern):
        meta_path = f"{pattern}/metadata.json"
        if os.path.exists(meta_path):
            return pattern

    # Clean up any orphaned .tmp dirs
    for tmp in glob.glob(f"{checkpoint_root}/{timeframe}_fold{fold}.tmp*"):
        shutil.rmtree(tmp, ignore_errors=True)

    return None
```

---

## 6. Recommended Configuration for This Pipeline

### vast.ai / Cloud Machine Setup Script

```bash
#!/bin/bash
# NVMe I/O tuning for sparse feature training pipeline

# 1. Readahead: 1MB start
NVME_DEV=$(lsblk -d -n -o NAME | grep nvme | head -1)
if [ -n "$NVME_DEV" ]; then
    blockdev --setra 2048 /dev/$NVME_DEV
    echo "Readahead set to 1MB on /dev/$NVME_DEV"
fi

# 2. THP: madvise only
echo madvise > /sys/kernel/mm/transparent_hugepage/enabled
echo "THP set to madvise"

# 3. Dirty pages: conservative for read-heavy workload
sysctl -w vm.dirty_ratio=10
sysctl -w vm.dirty_background_ratio=5

# 4. Swappiness: low (keep feature data in cache)
sysctl -w vm.swappiness=10

# 5. NUMA: detect and report
if command -v numactl &>/dev/null; then
    echo "NUMA topology:"
    numactl --hardware
    echo ""
    echo "Launch training with: numactl --cpunodebind=0 --membind=0 python ..."
fi

# 6. Verify
echo "=== I/O Configuration ==="
blockdev --getra /dev/$NVME_DEV 2>/dev/null && echo "Readahead: $(blockdev --getra /dev/$NVME_DEV) sectors"
cat /sys/kernel/mm/transparent_hugepage/enabled
sysctl vm.dirty_ratio vm.dirty_background_ratio vm.swappiness
```

### Pipeline Integration Points

| Pipeline Stage | I/O Pattern | Optimization |
|---|---|---|
| Feature build (cross gen) | Sequential write, heavy | `V2_RIGHT_CHUNK=500`, flush after each chunk |
| NPZ -> NPY conversion | Read NPZ, write .npy x3 | One-time cost, compress NPZ for transfer |
| Training data load | Sequential read, memmap | Readahead 1MB, NUMA-local, ThreadPool |
| CPCV fold iteration | Repeated sequential scan | Warm page cache after first fold |
| Checkpoint save | Burst write | Atomic tmp+rename, fsync before rename |
| Checkpoint recovery | Cold read | Load from manifest, skip .tmp dirs |

---

## 7. Migration Path from NPZ to NPY-Memmap

If the current pipeline uses `.npz` throughout, migrate incrementally:

1. **Phase 1:** After cross-gen produces NPZ, add a conversion step that extracts to separate .npy files
2. **Phase 2:** Training scripts load from .npy with `mmap_mode='r'` instead of `load_npz`
3. **Phase 3:** Cross-gen writes .npy directly (skip NPZ intermediate)
4. **Phase 4:** Keep NPZ only for cloud transfer / archival snapshots

**Expected gains:**
- Load time: 2-5x faster (no decompression, no archive unpacking)
- Memory: lower peak (mmap pages in on demand vs full materialization)
- Recovery: instant (no need to re-decompress on restart)
- Multi-fold: page cache shared across folds reading same feature arrays

---

## Summary of Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| I/O mode | Buffered (memmap) | Sequential pattern, warm cache helps restarts |
| Storage format | Separate .npy arrays | Memory-mappable, zero-copy, atomic checkpoints |
| Readahead | 1MB start, test to 4MB | NVMe sweet spot for sequential-with-skips |
| THP | `madvise` | Avoids over-allocation on sparse-touch regions |
| Dirty ratio | 10% / 5% background | Read-heavy, minimize writeback stalls |
| NUMA | Local bind first | Remote accesses erase NVMe speed gains |
| Parallel loading | ThreadPoolExecutor(4-8) | GIL released during disk reads, no IPC overhead |
| Checkpoint | Atomic tmp+rename+fsync | Crash-safe, no partial state corruption |
| Compression | Only for transfer/archival | CPU cost not worth it for local training I/O |
