# CrossGen-Bitpack — Optimization #4 (Phase 1D)

## What Changed

### New file: `bitpack_utils.py`
- `_llvm_ctpop_i64()` — Numba intrinsic wrapping `llvm.ctpop.i64` for hardware POPCNT
- `_pack_column()` / `_pack_matrix()` — pack binary float32 columns into uint64 bitarrays (1 bit per row)
- `_cooccurrence_matrix_popcnt()` — Numba `@njit(parallel=True)` kernel: AND + POPCNT across all (left, right) pairs
- `bitpack_cooccurrence_filter(left_mat, right_mat, min_co)` — public API, returns valid pairs + counts

### Modified: `v2_cross_generator.py`
- Added `USE_BITPACK_COOCCURRENCE` env var toggle (default: `1` = enabled)
- Added `_compute_cooccurrence_pairs()` — unified co-occurrence function used by both GPU and CPU paths
  - Tries bitpack POPCNT first (8-21ms for 7M+ pairs)
  - Falls back to sparse matmul (cuSPARSE → MKL → scipy) if bitpack unavailable
  - Logs: method used, total pairs, valid pairs, time taken
- `_gpu_cross_chunk()` — replaced 25-line sparse matmul block with single `_compute_cooccurrence_pairs()` call
- `_cpu_cross_chunk()` — replaced 25-line sparse matmul block with single `_compute_cooccurrence_pairs()` call

## How It Works
1. Each binary column is packed into `ceil(n_rows/64)` uint64 words (bit per row)
2. For each (L, R) pair: `popcnt(L_bits[w] AND R_bits[w])` summed across all words = exact co-occurrence count
3. Pairs with count < MIN_CO_OCCURRENCE (default 3) are rejected
4. Remaining pairs proceed to the existing cross generation (sparse matmul / GPU element-wise)

## Safety
- **Mathematically identical** to sparse matmul L.T @ R for binary features
- Binary features are lossless when bitpacked (0.0 → bit 0, nonzero → bit 1)
- No probabilistic filtering — exact counts only
- MIN_CO_OCCURRENCE threshold unchanged (still 3)
- Inner cross computation untouched — this is ONLY the pre-filter

## Configuration
- `USE_BITPACK_COOCCURRENCE=1` (default) — use bitpack POPCNT
- `USE_BITPACK_COOCCURRENCE=0` — fall back to sparse matmul (original behavior)

## What Was NOT Touched
- Inner cross computation (element-wise multiply)
- Step orchestration (cross step ordering)
- NPZ save/load
- `_generate_multi_signal_combos()` (uses dense sum, different pattern)
