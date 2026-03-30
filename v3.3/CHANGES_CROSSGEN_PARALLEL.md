# CrossGen-Parallel Changes (Optimizations #2 + #3)

## File Modified
- `v3.3/v2_cross_generator.py`

---

## Optimization #2: Parallel Cross Step Execution

### What Changed
All 12 cross-feature generation steps (DOY x contexts, Astro x TA, etc.) can now run concurrently via `ThreadPoolExecutor` instead of sequentially.

### How It Works
1. Each cross step is defined as a self-contained descriptor dict with its left/right arrays and metadata
2. A unified `_execute_one_step()` function runs any step independently
3. In parallel mode, steps are sorted by estimated RAM usage (ascending) and submitted with memory-aware scheduling:
   - Steps are submitted only while system RAM is below 70% ceiling
   - When ceiling is hit, the scheduler waits for running steps to complete before submitting more
   - Max 6 concurrent workers (prevents total RAM exhaustion)
4. Results are collected and assembled in the original step order after all steps complete
5. Each step uses `col_offset=0` internally; final offsets are assigned during collection

### Safety
- Each step reads from the **same immutable base matrix** (binarized contexts, TA arrays, etc.) -- thread-safe
- Numba releases GIL, so `ThreadPoolExecutor` achieves true parallelism for the inner compute
- Checkpoint/resume system still works: each step saves its own NPZ checkpoint after completion

### Configuration
```bash
# Enable parallel cross steps (default: off)
export PARALLEL_CROSS_STEPS=1

# Adjust RAM ceiling percentage (default: 70%)
export V2_RAM_CEILING_PCT=70
```

### When to Use
- **CPU mode with many cores**: Parallel steps let different cross types utilize different cores simultaneously
- **Large RAM machines (128GB+)**: Multiple steps can coexist in memory
- **NOT recommended for GPU mode with limited VRAM**: GPU steps compete for VRAM; sequential is safer

---

## Optimization #3: Adaptive RIGHT_CHUNK Controller

### What Changed
The static per-TF `RIGHT_CHUNK` value is replaced with a rolling RSS-based adaptive controller (`AdaptiveChunkController` class).

### How It Works
1. **Pilot phase**: First 2 chunks use RC=16 to measure actual bytes/column
2. **Sizing phase**: Subsequent chunks are sized from the worst-case of the last 3 RSS measurements
3. **MemoryError recovery**: On `MemoryError`, chunk size is halved and the same index is retried
4. **V2_RIGHT_CHUNK env var** now serves as a **max cap** (not a fixed value)
5. Target: stay below 70% of available RAM per chunk

### Class: `AdaptiveChunkController`
- `get_chunk_size()` -- returns current recommended chunk size
- `record_chunk(n_cols, rss_before, rss_after)` -- records measurement, triggers resize after pilot
- `halve()` -- called on MemoryError, halves chunk size

### Configuration
```bash
# Max cap for adaptive chunks (default: auto-detected from RAM)
export V2_RIGHT_CHUNK=500

# RAM ceiling for chunk sizing (default: 70%)
export V2_RAM_CEILING_PCT=70
```

---

## Memory Logging

Every cross step now logs memory at start and end:
```
[MEM] Cross 5 (ex2) START: RSS=4.21GB, System=32.1% (used=20.5GB / total=63.8GB, avail=43.3GB)
[MEM] Cross 5 (ex2) END (45.2s): RSS=5.83GB, System=38.7% (...)
```

Helper functions added:
- `_get_mem_percent()` -- system RAM usage %
- `_get_rss_bytes()` -- process RSS in bytes
- `_log_memory(label)` -- formatted memory log line

---

## Matrix Thesis Compliance
- ALL features preserved -- no filtering, no subsampling
- int64 indptr maintained (NNZ > 2^31)
- Every cross step must complete -- no partial runs accepted
- Element-wise multiply is order-independent -- parallel execution produces identical results

## What Was NOT Touched
- Inner cross computation (Numba kernels)
- NPZ save/load logic
- Co-occurrence counting
- Training files (run_optuna_local.py, ml_multi_tf.py)
