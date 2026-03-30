# Expert: Linux Kernel & OS Tuning for Training Pipeline

**Target**: vast.ai / Lambda cloud boxes — AMD EPYC 96-128c, 512GB-1TB RAM, NVMe, LightGBM with 2-10M sparse CSR features.

**Current baseline** (via `cuda_compat.py` + manual setup):
- tcmalloc preloaded via `LD_PRELOAD` + `PYTHONMALLOC=malloc`
- `OMP_NUM_THREADS=4` during cross gen, unset (all cores) during training
- No sysctl tuning in `cloud_run_tf.py` (relies on host defaults)

---

## 1. SYSCTL Tuning Script

Add this to `cloud_run_tf.py` Step 0, after dep install:

```bash
#!/bin/bash
# === OS TUNING FOR ML TRAINING ===

# --- Virtual Memory ---
sysctl -w vm.swappiness=1                    # Almost never swap (we want OOM not swap thrash)
sysctl -w vm.dirty_ratio=40                  # Allow 40% RAM dirty before sync (large writes)
sysctl -w vm.dirty_background_ratio=10       # Background writeback starts at 10%
sysctl -w vm.vfs_cache_pressure=50           # Keep dentries/inodes cached (NVMe file reloads)
sysctl -w vm.min_free_kbytes=2097152         # 2GB free reserve — prevents OOM during large allocs
sysctl -w vm.overcommit_memory=0             # Heuristic overcommit (safer than =1 for training)
sysctl -w vm.zone_reclaim_mode=0             # CRITICAL for NUMA: don't reclaim local before using remote
sysctl -w vm.max_map_count=1048576           # Needed for large mmap'd files and many sparse matrices

# --- NUMA ---
sysctl -w kernel.numa_balancing=0            # Disable auto-balancing (we pin manually)

# --- Scheduler ---
sysctl -w kernel.sched_migration_cost_ns=2000000  # 2ms — reduce thread bouncing between cores
sysctl -w kernel.sched_autogroup_enabled=0         # Disable autogroup (single-user training box)

# --- CPU Governor ---
for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance > "$gov" 2>/dev/null
done
```

### Why each setting matters:

| Setting | Why | Risk if wrong |
|---|---|---|
| `swappiness=1` | Training should never swap — swap = 100x slower than OOM restart | Swapping during histogram build kills throughput |
| `zone_reclaim_mode=0` | **CRITICAL**: Default (=1) on some NUMA systems forces local reclaim before using remote RAM, causing premature OOM on one node while other nodes have free RAM | Training dies with 50% RAM free on other socket |
| `min_free_kbytes=2GB` | Large sparse matrix allocs can fail if free memory is fragmented | `ENOMEM` during CSR construction despite "sufficient" RAM |
| `numa_balancing=0` | We pin with numactl; auto-migration wastes cycles scanning pages | 5-15% throughput loss from migration overhead on EPYC |
| `sched_migration_cost_ns=2M` | LightGBM OpenMP threads share L3 cache; bouncing kills cache locality | Threads migrate between CCDs, cold cache on every iteration |
| `max_map_count=1M` | Multiple large memmap files + sparse matrices can exceed 65K default | `mmap failed` errors loading NPZ/memmap files |

---

## 2. Transparent Huge Pages (THP)

### Recommendation: `enabled=madvise`, `defrag=defer+madvise`

**NOT `always`** — our sparse workloads have mixed allocation patterns. Large contiguous histogram buffers benefit from THP, but sparse CSR metadata (indptr, indices) is irregularly accessed and wastes physical memory at 2MB granularity.

```bash
# THP configuration
echo madvise > /sys/kernel/mm/transparent_hugepage/enabled
echo defer+madvise > /sys/kernel/mm/transparent_hugepage/defrag
echo 0 > /sys/kernel/mm/transparent_hugepage/khugepaged/defrag
```

### What this does:
- `madvise`: Only allocate THP for regions explicitly marked with `MADV_HUGEPAGE`
- `defer+madvise`: Background compaction for madvise'd regions, no synchronous stalls
- tcmalloc (already preloaded) uses `madvise(MADV_HUGEPAGE)` on large allocations automatically

### Verification:
```bash
# After a training run, check:
grep -E 'thp_fault|compact_stall|AnonHuge' /proc/vmstat /proc/meminfo
# Good: thp_fault_alloc rising, compact_stall staying low
# Bad: compact_stall >> thp_fault_alloc (switch to defrag=defer)
```

---

## 3. NUMA Policy

### Decision tree:

```
Dataset + LightGBM working set fits in one socket's RAM?
  YES → numactl --cpunodebind=0 --membind=0   (best latency)
  NO  → numactl --cpunodebind=0,1 --interleave=all  (current default, correct)
```

For our workload (2-10M features, CSR matrices often 20-80GB):
- **1w/1d/4h**: Usually fits one socket on 512GB+ boxes → try `membind` first
- **1h/15m**: Often exceeds single socket → `interleave=all` is correct

### Benchmark command:
```bash
# A/B test: run identical 50-round training, compare wall time
# Test A: interleave (current)
numactl --interleave=all python -u ml_multi_tf.py --tf 1w --max-rounds 50

# Test B: single-socket bind
numactl --cpunodebind=0 --membind=0 python -u ml_multi_tf.py --tf 1w --max-rounds 50

# Measure with:
numastat -p $(pgrep -f ml_multi_tf)  # Remote vs local memory ratio
```

### Key insight from research:
LightGBM has a known NUMA sensitivity issue (GitHub #1441). Histogram building repeatedly touches shared data structures — local memory access wins unless capacity-constrained. The `zone_reclaim_mode=0` sysctl above is **critical** to making interleave work properly (prevents one node from reclaiming cache while other nodes have free RAM).

---

## 4. File I/O: mmap vs read() for NPZ/Memmap

### Recommendation: **Eliminate compressed NPZ from hot path**

**Key finding**: `numpy.load(mmap_mode='r')` does NOT memory-map arrays inside `.npz` files. The `mmap_mode` parameter is silently ignored for `.npz` — it only works for standalone `.npy` files.

### Optimal data path:
```
Current (slow):  scipy.sparse.load_npz() → decompress → construct CSR
Better:          Save CSR components as separate .npy → mmap individually → construct CSR
Best:            Save as LightGBM binary dataset (.bin) → lgb.Dataset(data='file.bin')
```

### For cross generator output:
```python
# Instead of: scipy.sparse.save_npz('crosses.npz', csr_matrix)
# Save components directly:
np.save('crosses_data.npy', csr.data)
np.save('crosses_indices.npy', csr.indices)
np.save('crosses_indptr.npy', csr.indptr)

# Load with true mmap:
data = np.load('crosses_data.npy', mmap_mode='r')
indices = np.load('crosses_indices.npy', mmap_mode='r')
indptr = np.load('crosses_indptr.npy', mmap_mode='r')
csr = scipy.sparse.csr_matrix((data, indices, indptr), shape=shape)
```

### NVMe readahead for mmap:
```bash
# Increase readahead for NVMe (default 128 sectors = 64KB, too small for sequential mmap)
for dev in /sys/block/nvme*; do
    echo 2048 > "$dev/queue/read_ahead_kb"  # 2MB readahead
done
```

### io_uring verdict:
Not worth it for this pipeline. mmap with proper readahead + page cache covers our access pattern (large sequential reads). io_uring's advantage is for many concurrent small random reads — not our case.

---

## 5. CFS Scheduler Tuning

### Recommendation: Do NOT use `isolcpus` / `nohz_full`

These are for latency-sensitive real-time workloads (one task per isolated core). LightGBM is throughput-oriented with OpenMP thread cooperation — isolation would hurt, not help.

### What to do instead:

```bash
# 1. Disable autogroup (training box, not desktop)
sysctl -w kernel.sched_autogroup_enabled=0

# 2. Increase migration cost (keep threads on their CCD)
sysctl -w kernel.sched_migration_cost_ns=2000000

# 3. Set CPU governor to performance
cpupower frequency-set -g performance 2>/dev/null || \
    for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        echo performance > "$g" 2>/dev/null
    done
```

### For Numba phases (cross generator binarization):
Numba's thread pool is separate from OpenMP. The existing approach of setting `NUMBA_NUM_THREADS` to core count for binarization and 4 for matmul is correct. No scheduler changes needed — Numba uses its own threadpool, not CFS-managed threads.

### SMT (Hyperthreading) consideration:
On EPYC, SMT provides marginal benefit for memory-bandwidth-bound workloads like histogram building. If the box has 128 physical + 128 SMT:
- `num_threads=0` (auto) will use all 256 — this can hurt due to SMT contention on shared FP units
- Test `num_threads=128` (physical only) vs `num_threads=0` — physical-only often wins by 5-15%

---

## 6. Cgroup v2 OOM Prevention

### Recommendation: Use cgroup `memory.high` as throttle, `memory.max` as safety net

```bash
# Create training cgroup (if systemd, use systemd-run instead)
TRAIN_CG="/sys/fs/cgroup/training"
mkdir -p "$TRAIN_CG"

# For 512GB host: reserve 64GB for OS/cache, give training 400GB budget
echo $((380 * 1024 * 1024 * 1024)) > "$TRAIN_CG/memory.high"  # Throttle at 380GB
echo $((420 * 1024 * 1024 * 1024)) > "$TRAIN_CG/memory.max"   # Hard kill at 420GB

# For 1TB host: reserve 96GB for OS/cache
# echo $((850 * 1024 * 1024 * 1024)) > "$TRAIN_CG/memory.high"
# echo $((920 * 1024 * 1024 * 1024)) > "$TRAIN_CG/memory.max"

# Move training process into cgroup
echo $$ > "$TRAIN_CG/cgroup.procs"
```

### Simpler alternative with systemd-run:
```bash
# Launch training in a memory-limited scope
systemd-run --scope -p MemoryHigh=380G -p MemoryMax=420G \
    python -u cloud_run_tf.py --tf 1w
```

### Monitor during training:
```bash
# Watch for pressure (add to monitoring loop)
cat /sys/fs/cgroup/training/memory.pressure
# "some avg10=0.00" = fine
# "some avg10=5.00+" = approaching throttle, reduce concurrency

cat /sys/fs/cgroup/training/memory.events
# Watch 'high' counter — rising means throttling active
# Watch 'oom_kill' counter — should stay 0
```

### LightGBM-specific memory control:
- `histogram_pool_size` in LightGBM params caps histogram memory (MB). Set to ~30% of budget.
- For 2M+ features, histogram memory alone can be 50-100GB. This is the main control.

---

## 7. Complete Setup Script for cloud_run_tf.py

Drop-in replacement for Step 0 system tuning (add after pip install):

```python
def setup_linux_tuning():
    """OS-level tuning for ML training. Safe to call multiple times."""
    import shutil
    cmds = [
        # Virtual memory
        'sysctl -w vm.swappiness=1',
        'sysctl -w vm.dirty_ratio=40',
        'sysctl -w vm.dirty_background_ratio=10',
        'sysctl -w vm.vfs_cache_pressure=50',
        'sysctl -w vm.min_free_kbytes=2097152',
        'sysctl -w vm.overcommit_memory=0',
        'sysctl -w vm.zone_reclaim_mode=0',
        'sysctl -w vm.max_map_count=1048576',
        # NUMA
        'sysctl -w kernel.numa_balancing=0',
        # Scheduler
        'sysctl -w kernel.sched_migration_cost_ns=2000000',
        'sysctl -w kernel.sched_autogroup_enabled=0',
    ]
    for cmd in cmds:
        os.system(cmd + ' 2>/dev/null')

    # THP: madvise + defer+madvise (sparse-friendly)
    for path, val in [
        ('/sys/kernel/mm/transparent_hugepage/enabled', 'madvise'),
        ('/sys/kernel/mm/transparent_hugepage/defrag', 'defer+madvise'),
    ]:
        try:
            with open(path, 'w') as f: f.write(val)
        except (OSError, PermissionError):
            pass

    # NVMe readahead (2MB for mmap performance)
    import glob as _glob
    for dev in _glob.glob('/sys/block/nvme*/queue/read_ahead_kb'):
        try:
            with open(dev, 'w') as f: f.write('2048')
        except (OSError, PermissionError):
            pass

    # CPU governor: performance
    for gov in _glob.glob('/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor'):
        try:
            with open(gov, 'w') as f: f.write('performance')
        except (OSError, PermissionError):
            pass

    # Install tcmalloc if not present (cuda_compat.py handles LD_PRELOAD)
    if not any(os.path.exists(p) for p in [
        '/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4',
        '/usr/lib/libtcmalloc_minimal.so.4',
    ]):
        os.system('apt-get install -y -qq libgoogle-perftools4 2>/dev/null || '
                  'yum install -y -q gperftools-libs 2>/dev/null')

    log("Linux tuning applied: vm, NUMA, scheduler, THP=madvise, NVMe readahead=2MB")
```

---

## 8. Priority Ranking (Impact vs Effort)

| # | Tuning | Expected Impact | Effort | When |
|---|---|---|---|---|
| 1 | `zone_reclaim_mode=0` | **HIGH** — prevents false OOM on NUMA | 1 line | Always |
| 2 | `numa_balancing=0` + manual pin | **HIGH** — 5-15% throughput | 2 lines | Always |
| 3 | THP `madvise` + `defer+madvise` | **MEDIUM** — eliminates compaction stalls | 3 lines | Always |
| 4 | `sched_migration_cost_ns=2M` | **MEDIUM** — reduces thread bouncing | 1 line | Always |
| 5 | `min_free_kbytes=2GB` | **MEDIUM** — prevents alloc failures | 1 line | Always |
| 6 | NVMe readahead 2MB | **LOW-MEDIUM** — faster mmap loads | 1 line | Always |
| 7 | CPU governor `performance` | **LOW** — boxes usually default this | Loop | Always |
| 8 | cgroup `memory.high` | **MEDIUM** — prevents hard OOM kill | 5 lines | 15m/1h TFs |
| 9 | Eliminate compressed NPZ | **HIGH** — faster data loading | Code change | Future |
| 10 | `num_threads=128` (no SMT) | **LOW-MEDIUM** — test empirically | 1 param | Benchmark |

---

## 9. Verification Checklist

Run after applying tuning, before training:

```bash
#!/bin/bash
echo "=== Linux Tuning Verification ==="
echo "swappiness:     $(cat /proc/sys/vm/swappiness) (want: 1)"
echo "zone_reclaim:   $(cat /proc/sys/vm/zone_reclaim_mode) (want: 0)"
echo "numa_balancing:  $(cat /proc/sys/kernel/numa_balancing) (want: 0)"
echo "migration_cost: $(cat /proc/sys/kernel/sched_migration_cost_ns) (want: 2000000)"
echo "max_map_count:  $(cat /proc/sys/vm/max_map_count) (want: 1048576)"
echo "min_free_kB:    $(cat /proc/sys/vm/min_free_kbytes) (want: 2097152)"
echo "THP enabled:    $(cat /sys/kernel/mm/transparent_hugepage/enabled) (want: [madvise])"
echo "THP defrag:     $(cat /sys/kernel/mm/transparent_hugepage/defrag) (want: [defer+madvise])"
echo "CPU governor:   $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo N/A) (want: performance)"
echo "NUMA nodes:     $(numactl --hardware 2>/dev/null | grep 'available' || echo 'numactl not installed')"
echo "NVMe readahead: $(cat /sys/block/nvme0n1/queue/read_ahead_kb 2>/dev/null || echo N/A) (want: 2048)"
echo "tcmalloc:       $(ls /usr/lib/x86_64-linux-gnu/libtcmalloc* 2>/dev/null | head -1 || echo 'NOT FOUND')"
echo "LD_PRELOAD:     ${LD_PRELOAD:-not set}"
```

---

## 10. What NOT To Do

| Bad idea | Why |
|---|---|
| `isolcpus` / `nohz_full` | Designed for RT, not throughput. LightGBM OpenMP threads cooperate — isolation hurts. |
| `vm.overcommit_memory=1` | Hides real OOM until process gets SIGKILL mid-training. Keep =0 for early warning. |
| THP `enabled=always` | Synchronous compaction stalls during sparse matrix alloc. Use `madvise`. |
| `vm.zone_reclaim_mode=1` | Causes false OOM: node 0 reclaims aggressively while node 1 has free RAM. |
| Swap file for "safety" | Training on swap = 100x slower. Better to OOM and restart than swap-thrash for hours. |
| `feature_fraction < 0.7` | NOT a kernel setting, but often "tuned" alongside — kills rare esoteric cross signals. |
| io_uring for data loading | Over-engineering. mmap + readahead covers our sequential-read pattern. |

---

## Sources

- AMD EPYC NVMe Tuning Guide (docs.amd.com)
- SUSE EPYC Optimization Guide (SLES 15 SP1/SP6)
- LightGBM Parameters Tuning docs (lightgbm.readthedocs.io)
- LightGBM GitHub #1441 (NUMA performance impact)
- LightGBM GitHub #5478 (sparse dataset loading)
- Linux Kernel Documentation: THP, cgroup v2, NUMA memory policy
- Perplexity research queries (2026-03-30), 7 queries with full matrix thesis context
