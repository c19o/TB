# CrossGen Memmap Merge — Optimization 1E

## Problem
Final assembly in `v2_cross_generator.py` uses `_streaming_csc_splice()` which `np.concatenate()`s all data/indices/indptr arrays in RAM. For large timeframes:
- **1h**: ~1.8TB peak RAM (OOM on any machine)
- **15m**: ~3TB+ peak RAM (impossible)

## Solution
New module `memmap_merge.py` implements two-pass streaming CSC merge via memory-mapped files:

### Pass 1 — Scan
- Load each checkpoint file one at a time
- Count total NNZ and columns
- No accumulation in RAM

### Pass 2 — Fill
- Pre-allocate memmap files on NVMe (data, indices, indptr)
- Fill from each checkpoint sequentially
- Each checkpoint loaded, copied to memmap, then freed

### Assembly
- Construct `scipy.sparse.csc_matrix` backed by memmap arrays (no RAM spike)
- Convert to CSR via OS page cache streaming
- `save_npz()` streams from memmap — no full materialization

## RAM Reduction
| TF | Before | After | Reduction |
|----|--------|-------|-----------|
| 1h | ~1.8TB | ~5-10GB | 99.5% |
| 15m | ~3TB+ | ~10-15GB | 99.7% |

## Controls
| Env Var | Effect |
|---------|--------|
| `MEMMAP_CROSS_GEN=1` | Force memmap for ALL timeframes |
| `MEMMAP_CROSS_GEN=0` | Force in-memory for ALL timeframes |
| Unset | Auto-enable for 1h/15m only |

## Files Changed
- **`v3.3/memmap_merge.py`** — NEW: Two-pass streaming CSC merge module
- **`v3.3/v2_cross_generator.py`** — Modified: final assembly uses memmap path for 1h/15m

## Matrix Compliance
- ALL cross features preserved — no filtering during merge
- int64 indptr (NNZ > 2^31 safe)
- Resume-safe: per-type checkpoints unchanged
- Structural zeros = 0.0, not NaN
- In-memory path unchanged for 1w/1d/4h (no regression risk)
