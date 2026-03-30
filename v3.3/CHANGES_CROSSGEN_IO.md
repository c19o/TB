# CrossGen-IO Optimizations (#7, #8, #7H)

## Optimization #7 — Atomic NPZ checkpointing

All `sparse.save_npz` calls now use temp file + `os.replace()` for atomic writes.
If the process crashes mid-write, no corrupt `.npz` files are left behind.

**Changed locations:**
- `gpu_batch_cross._flush_chunks_to_disk()` — sub-checkpoint flush
- `_gpu_cross_chunk._flush_csr_to_disk()` — GPU CSR disk flush
- `_save_checkpoint()` — per-cross-type checkpoint (NPZ + JSON both atomic)
- `atomic_save_npz()` in `atomic_io.py` — already atomic (final save)

**Pattern:** write to `path.tmp` -> `os.replace(tmp, final)`. On failure, temp is cleaned up.

## Optimization #8 — Indices-only NPZ storage

Binary crosses are all `1.0`. The `data` array is redundant (~40% of file size).
When `NPZ_INDICES_ONLY=1` (default ON), final save stores only:
- `indptr` (int64 for NNZ > 2^31)
- `indices` (int32)
- `shape` (int64 array)
- `_indices_only` marker (int8)

On load, `data = np.ones(len(indices), dtype=np.float32)` reconstructs exact IEEE 754 1.0 values.

**Backward compatible:** `load_npz_auto()` detects format by checking for `_indices_only` key.
Old scipy NPZ files load normally via `sparse.load_npz()`.

**Env var:** `NPZ_INDICES_ONLY=0` to disable and use full scipy format.

**Files:**
- `atomic_io.py` — `save_npz_indices_only()`, `load_npz_auto()`
- `v2_cross_generator.py` — final save uses indices-only; checkpoint resume uses `load_npz_auto()`

## Optimization #7H — Intra-step time flush

Long-running cross types (esp. 15m with 200K+ rows) can run for hours.
Without periodic flush, a crash loses all accumulated work within that step.

**Mechanism:** Track wall clock time. Flush accumulated CSR chunks to disk every
`V2_FLUSH_INTERVAL_SEC` seconds (default 1200 = 20 min), even if the chunk count
threshold hasn't been reached.

**Env var:** `V2_FLUSH_INTERVAL_SEC=1200` (default). Set lower for more frequent saves.

**Locations:**
- `gpu_batch_cross()` RIGHT_CHUNK loop — time-based sub-checkpoint flush
- `generate_all_crosses()` — `_maybe_time_flush()` between cross types

## Matrix thesis compliance

- ALL features preserved. No data loss in save/load
- int64 indptr preserved through indices-only format
- float32 data reconstructed as exact IEEE 754 1.0
- Structural zeros in CSR remain structural zeros (not stored)
- Backward compatible with all existing NPZ files
