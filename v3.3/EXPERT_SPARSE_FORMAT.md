# Expert Report: Sparse Matrix Format for Binary Feature Matrices

**Date**: 2026-03-30
**Context**: 10M+ pure binary (0/1) sparse features in CSR, int64 indptr, 5-15% density, LightGBM histogram construction. No data loss allowed.

---

## Executive Summary

**CSR remains the correct format.** No alternative (CSC, BSR, ELLPACK, HYB, roaring bitmap, bitarray) offers a clear win for our specific workload: LightGBM histogram construction on 10M+ binary sparse columns with 227K rows at 5-15% density. The highest-value optimizations are not format changes but **parameter tuning** and **leveraging what we already have** (indices-only storage, EFB bundling, col-wise histograms).

---

## 1. Format-by-Format Analysis

### 1.1 CSR (Current) — KEEP

| Metric | Value |
|---|---|
| Memory per row | `8 * nnz_per_row + 8` bytes (int64 indptr + int64 indices, no data array) |
| LightGBM compat | Native. `scipy.sparse.csr_matrix` accepted directly |
| Histogram cost | O(2 * #non_zero_data) with `is_enable_sparse=True` |
| Int64 support | Yes — required when NNZ > 2^31 |

**Why it works**: LightGBM's sparse histogram path iterates only nonzeros. CSR matches our row-oriented pipeline (cross gen builds row-by-row). Our indices-only NPZ already eliminates the data array (all values are implicitly 1), achieving the best compression ratio possible within CSR semantics.

**Current config is already correct**:
- `force_col_wise=True` (except 15m which uses row-wise due to 294K rows / ~23K bundles = 12.8 ratio)
- `is_enable_sparse=True`
- `max_bin=255` (enables EFB bundling of ~254 mutually exclusive binary features per bundle)
- `feature_pre_filter=False` (protects rare signals)

### 1.2 CSC (Column-Compressed) — MARGINAL WIN, HIGH COST

| Metric | Value |
|---|---|
| Histogram access pattern | Better — column-major matches per-feature histogram construction |
| indptr size | **10M+ entries** (one per column) vs 227K entries for CSR |
| Conversion cost | O(nnz) time + double memory during conversion |

**Verdict: Not worth it.** CSC's indptr array at 10M+ columns costs ~80MB just for pointers, vs ~1.8MB for CSR's 227K-row indptr. LightGBM internally transposes as needed for col-wise mode. The conversion overhead and memory spike during CSR→CSC is worse than letting LightGBM handle it internally. Our EFB bundles reduce effective feature count to ~23K anyway, making the column-access pattern moot after bundling.

### 1.3 BSR (Block Sparse Row) — REJECTED

| Metric | Value |
|---|---|
| Best case | Dense sub-blocks within sparse structure |
| Our case | No block structure — cross features are scattered, not grouped |

**Verdict: Hard reject.** BSR requires evenly-dividing block sizes and only pays off when blocks are internally dense. Our binary cross features have no natural block structure. BSR would store mostly-zero blocks, wasting memory. SciPy docs explicitly say BSR is "considerably more efficient" only with "dense sub-matrices."

### 1.4 ELLPACK / SELL / HYB — REJECTED FOR TRAINING

| Metric | Value |
|---|---|
| ELLPACK | Pads all rows to max row length — catastrophic at 10M+ columns |
| SELL (Sliced ELL) | Better with row bucketing, but still pads within slices |
| HYB (ELL + COO overflow) | Overflow portion adds conversion cost and dual traversal |

**Verdict: Hard reject for our workload.** These formats exist for GPU SpMV (sparse matrix-vector multiply), not histogram construction. With variable-density rows (5-15% = 500K to 1.5M nonzeros per row at 10M features), padding waste is enormous. XGBoost uses ELLPACK internally for GPU because it does SpMV-style operations; LightGBM's histogram approach doesn't benefit from fixed-stride memory layout.

Additionally: **LightGBM CUDA currently warns "Using sparse features with CUDA is not supported"** (GitHub issue #6725). The sparse GPU path is not mature enough to architect around.

### 1.5 Roaring Bitmap — REJECTED

| Metric | Value |
|---|---|
| Designed for | Fast set algebra (union/intersection) on posting lists |
| Our need | Gradient/hessian accumulation per feature bin |

**Verdict: Wrong abstraction.** Roaring excels at boolean set operations (OLAP filters, search engine posting lists). LightGBM needs to accumulate floating-point gradient/hessian sums into histogram bins — not intersect sets. At 5-15% density, many roaring chunks would be bitmap containers (not compressed arrays), adding container dispatch overhead with no histogram benefit. Would require a complete custom trainer.

### 1.6 Dense Bitarray (1-bit per entry) — REJECTED

| Metric | Value |
|---|---|
| Memory | 10M features * 227K rows / 8 = ~284 GB (dense bitmap) |
| CSR memory | ~8 * nnz bytes (at 10% density: ~8 * 227B = ~1.8 TB... but with EFB: much less) |
| Scan cost | O(n_rows * n_features) — must scan zeros |

**Verdict: Worse in every way.** Dense bitmap forces scanning ALL entries (including zeros), while CSR touches only nonzeros. At 5-15% density, 85-95% of scanned bits are wasted work. LightGBM has no bitmap-native histogram kernel. Would require complete custom implementation with SIMD popcount throughout.

---

## 2. Memory Analysis: Current vs Alternatives

### Our actual matrix (example: 4h timeframe)
- Rows: ~23,000
- Features: ~6M binary cross features
- Density: ~10%
- NNZ: ~1.38 billion

### Memory comparison (indices-only CSR vs alternatives)

| Format | Storage Formula | Estimated Size |
|---|---|---|
| **CSR indices-only (current)** | `8*nnz + 8*(rows+1)` | **~11.0 GB** |
| CSR with float64 data | `16*nnz + 8*(rows+1)` | ~22.1 GB |
| CSR with float32 data | `12*nnz + 8*(rows+1)` | ~16.6 GB |
| CSC indices-only | `8*nnz + 8*(cols+1)` | ~11.0 GB + 48MB indptr |
| Dense bitmap | `rows * cols / 8` | ~17.3 GB (stores zeros!) |
| Dense float32 | `4 * rows * cols` | ~552 GB |

**Key insight**: Our indices-only CSR is already near-optimal. The data array elimination saves 50% of the nnz payload (we already do this). The remaining cost is dominated by int64 column indices.

---

## 3. The Real Optimization Opportunities

Instead of changing formats, these parameter/pipeline changes yield actual speedups:

### 3.1 `bin_construct_sample_cnt` — MISSING FROM CONFIG

**Finding**: LightGBM docs explicitly state "increase this for very sparse data" and warn that too-small values cause "poor accuracy or unexpected errors during bin construction." Default is 200,000.

**Recommendation**: Set `bin_construct_sample_cnt=2000000` (10x default). With binary features this mostly affects how LightGBM discovers the {0, 1} bin boundaries — with too few samples, sparse features with <1% density may not have enough 1s in the sample to construct proper bins.

**Risk of NOT doing this**: Rare esoteric signals (firing 10-20 times in 23K rows) might get incorrect bin boundaries from a 200K subsample.

### 3.2 Index Dtype Optimization — PARTIAL WIN

Currently: `indptr=int64, indices=int64` (per memory file).

**Per-chunk opportunity**: During cross gen, individual NPZ chunks have NNZ well below 2^31. Could use `int32` indices within chunks and only promote to `int64` after merge. Saves 50% on index storage during build phase.

**Training**: The merged matrix requires int64 indptr (NNZ > 2^31), but `indices` (column positions) could stay `int32` if max column index < 2^31 (~2.1B). With 6-10M features, int32 indices are safe. **This alone saves ~5.5 GB on the 4h example.**

**Current CLAUDE.md says**: "indptr=int64 (NNZ > 2^31 fix), indices=int32" — so this is already the intended design. Verify actual implementation matches.

### 3.3 EFB Bundle Awareness

LightGBM's Exclusive Feature Bundling (EFB) is the single most impactful optimization for our matrix. With `max_bin=255`, up to 254 mutually exclusive binary features are packed into one bundle. For 6M features, this reduces effective feature count to ~23K bundles.

**This means**: After EFB, LightGBM internally works with ~23K "meta-features" of up to 255 bins each. The sparse format only matters for the initial Dataset construction and EFB detection phase. Training-time histograms operate on the bundled representation.

### 3.4 `zero_as_missing=False` — VERIFY

**Critical**: Must be `False` (default). If `True`, structural zeros in CSR (feature OFF) would be treated as missing values instead of "value is zero." This would fundamentally break binary cross-feature semantics where 0 = "condition not met" (a real signal) vs NaN = "data unavailable."

**Current config**: Not explicitly set. Default is `False`. Recommend making it explicit.

---

## 4. GPU Format Considerations (Future)

### 4.1 Current State
- LightGBM CUDA: **Does not support sparse features** (GitHub #6631, #6725)
- Our GPU fork: Uses CuPy for SpMV acceleration of histogram construction
- XGBoost GPU: Uses ELLPACK internally (irrelevant — we use LightGBM)

### 4.2 Best GPU Representation for Binary Features

If building a custom GPU histogram kernel (which our fork partially does):

| Approach | Description | Fit |
|---|---|---|
| **CSR on GPU** | cuSPARSE supports int32/int64 CSR natively | Good baseline |
| **Feature posting lists** | Per-feature: sorted array of active row indices | Best for histogram accumulation |
| **Implicit-value CSR** | CSR without data array; kernel assumes all values = 1 | Our current approach — correct |
| **Blocked feature groups** | Group similar-density features, process blocks together | Good for warp utilization |

**The winning GPU design** (already partially implemented in our fork): Store compressed active-index lists per feature. For each active entry, add gradient/hessian to the feature's "1-bin." Derive the "0-bin" by subtraction from node totals. This is exactly LightGBM's histogram-subtraction philosophy applied to binary features.

### 4.3 cuSPARSE Compatibility
- cuSPARSE generic API supports both int32 and int64 indices
- CSR is a first-class format in cuSPARSE 13.x
- No need to convert to ELLPACK/HYB for our use case

---

## 5. Recommendations Summary

### Do Now (No Code Changes)
1. **Add `bin_construct_sample_cnt=2000000`** to `V3_LGBM_PARAMS` in config.py
2. **Add `zero_as_missing=False`** explicitly to `V3_LGBM_PARAMS`
3. **Verify** indices are int32 (not int64) in the actual sparse matrices — CLAUDE.md says this is intended but grep confirms it should be validated

### Do Not Do
1. Do NOT switch to CSC — LightGBM handles internal transposition; CSC indptr at 10M+ columns wastes memory
2. Do NOT implement roaring bitmap — wrong abstraction for histogram accumulation
3. Do NOT use BSR — no block structure in cross features
4. Do NOT use ELLPACK/HYB — designed for SpMV, not histogram construction; padding waste is catastrophic
5. Do NOT use dense bitarray — forces scanning zeros, 6-17x worse than sparse at our density
6. Do NOT switch to row-wise mode (except 15m which already does) — col-wise is correct for high feature count

### Future Investigation
1. **Custom GPU kernel**: Feature-posting-list format with implicit value=1 semantics (partially done in gpu_histogram_fork)
2. **Index compression**: Delta-encoding or variable-length encoding of column indices within CSR (research-grade, not production-ready)
3. **Monitor LightGBM CUDA sparse support**: GitHub issues #6631, #6725 — when resolved, our CSR matrices should work directly on GPU without custom fork

---

## 6. Format Decision Matrix

| Format | Memory | LightGBM Compat | Histogram Speed | Binary Exploit | Data Loss Risk | Verdict |
|---|---|---|---|---|---|---|
| **CSR indices-only** | Optimal | Native | O(nnz) | Yes (no data array) | None | **KEEP** |
| CSC indices-only | +48MB indptr | Native | O(nnz) | Yes | None | Not worth conversion cost |
| BSR | Worse (block padding) | Native | O(nnz + padding) | No | None | Reject — no block structure |
| ELLPACK | Catastrophic padding | Not supported | N/A | No | None | Reject |
| HYB | Bad (overflow) | Not supported | N/A | No | None | Reject |
| Roaring bitmap | Good compression | Not supported | Custom only | No | None | Reject — wrong abstraction |
| Dense bitarray | 17GB+ (stores zeros) | Not supported | O(rows*features) | Yes (1-bit) | None | Reject — scans zeros |
| Dense float32 | 552GB | Native | O(rows*features) | No | None | Reject — impossible |

---

## Sources

- LightGBM Parameters Documentation (v4.6.0)
- LightGBM Features Documentation — sparse optimization, EFB
- LightGBM GitHub Issues: #6352 (sparse arrays), #6631 (CUDA sparse), #6725 (CUDA warning), #7101 (int64 segfault), #5478 (sparse load times), #1689 (NNZ > 2^31)
- SciPy Sparse Documentation (v1.17.0) — CSR, CSC, BSR format specs
- cuSPARSE 13.2 Documentation — GPU sparse format support
- Roaring Bitmap Paper (Chambi et al., 2016)
- XGBoost GPU Documentation — ELLPACK internal format
- "Compiling Generalized Histograms for GPU" (SC'20)
