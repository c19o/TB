# EXPERT: cuSPARSE & Sparse Linear Algebra — Architecture Report

**Date:** 2026-03-30
**Scope:** GPU histogram construction for custom LightGBM fork
**Matrix:** 2–10M sparse binary cross-features, CSR (int64 indptr, int32 indices, float32 data=all 1.0)
**Hardware:** 8× RTX 5090 target, validated on RTX 3090 (78× SpMV speedup achieved)

---

## 1. Current Architecture

### What We Have
- **`histogram_cusparse.py`** — `CuSparseHistogramBuilder` class using CuPy's `cupyx.scipy.sparse`
- **`gpu_histogram_cusparse.py`** — Standalone function doing `CSR.T @ grad_vector`
- **Approach:** Upload CSR to GPU once, pre-compute CSR transpose (`.T.tocsr()`), reuse for all SpMV calls
- **Leaf masking:** Zero-out non-leaf rows via elementwise multiply, then SpMV on full matrix
- **SpMM batching:** `build_all_leaves()` constructs dense gradient matrices `G[row, leaf]` and does `CSR.T @ G`

### Current Performance Profile
- 78× speedup over CPU reference (validated)
- 7.46s/round on RTX 3090
- Memory: CSR + CSR.T + grad/hess buffers stored on GPU

---

## 2. Is CSR Optimal?

### Verdict: **CSR is correct. Stay with CSR.**

| Format | Fit | Reason |
|--------|-----|--------|
| **CSR** | ✅ Best default | Compact, robust to skewed row-length distribution, no padding tax, natural for row-sharded training. cuSPARSE supports CSR with int64 indptr natively. |
| **CSC** | ✅ Good secondary | Already used implicitly (CSR.T = CSC of original). Useful for feature-major histogram passes. Keep the pre-computed transpose. |
| **SELL (Sliced ELL)** | ⚠️ Situational | Only wins if rows can be bucketed into slices with very similar nnz. Cross-features typically have heavy-tail row-length distribution → SELL padding wastes bandwidth. |
| **BSR** | ❌ Poor fit | Requires dense block structure in nonzeros. Binary cross-features have isolated singleton activations — BSR adds padding overhead with no block reuse. |
| **HYB (ELL+COO)** | ❌ Avoid | Overflow and conversion overhead hurt when row lengths vary. Published GPU results show degradation as overflow grows. |
| **Blocked-ELL** | ❌ Avoid | Designed for Tensor Core block-sparse SpMM (structured sparsity). Not a fit for irregular singleton binary features. |

**Key insight:** Cross-features create highly skewed row/column occupancies (many tiny rows, a few dense ones). CSR is the least sensitive format to this distribution. Utah GPU research confirmed CSR has the smallest memory footprint while HYB was only favorable with low row-length variance.

---

## 3. Exploiting the All-1.0 Data Array

### The Single Biggest Optimization Opportunity

Every value in our sparse matrix is exactly 1.0. This means:

```
histogram[feature] = CSR.T @ gradient_vector
                   = Σ gradient[row]  for each row where feature == 1
```

**The multiply by 1.0 is a no-op.** The operation is pure gather-and-accumulate, not multiply-accumulate.

### What This Enables

1. **Eliminate value array reads entirely** — Custom kernel skips loading `data[]` array. For 2M features at 0.3% density, this saves ~24MB of wasted memory bandwidth per SpMV call.

2. **Implicit-value CSR format** — Store only `indptr` + `indices`, drop `data` array completely. Cuts CSR memory by ~33% (one fewer array of same length as indices).

3. **Replace SpMV with sparse gather-accumulate** — Instead of `result[col] += data[k] * x[row]`, just do `result[col] += x[row]`. This is a fundamentally simpler operation.

4. **Integer accumulation path** — For counting operations (bin counts), use integer atomics instead of floating-point, which are faster and exact on GPU.

### cuSPARSE Cannot Do This

cuSPARSE's generic API always reads the value array — it has no "implicit ones" mode. This is where a **custom kernel definitively beats cuSPARSE** for our workload. The library is designed for general sparse matrices with arbitrary values; our constraint (all 1.0) is a domain-specific optimization that no library will exploit.

---

## 4. cuSPARSE Generic API — Algorithm Selection

### For Cases Where We Still Use cuSPARSE

| Algorithm | Speed | Determinism | Use Case |
|-----------|-------|-------------|----------|
| `CUSPARSE_SPMV_ALG_DEFAULT` | Baseline | No guarantee | Never use — just a generic fallback |
| `CUSPARSE_SPMV_CSR_ALG1` | **Fastest** | Non-deterministic | Production training (tiny FP accumulation order differences) |
| `CUSPARSE_SPMV_CSR_ALG2` | Slower | **Bitwise deterministic** | Debugging, regression testing, validation runs |

### Key API Features
- **`cusparseSpMV_preprocess()`** — Amortizes analysis cost when reusing same sparsity pattern. Critical for our use case (same CSR structure, different gradient vectors each round).
- **64-bit index support** — Our int64 indptr is directly compatible.
- **Non-transpose is much faster than transpose** — NVIDIA explicitly documents this. Our current code does `CSR.T @ vector` which is a transpose operation. Better: store CSC (= CSR of transpose) and do non-transpose SpMV on that.

### Recommendation
Our current CuPy path (`gpu_csr_t @ g_gpu`) already pre-computes the transpose as CSR, so we're effectively doing non-transpose SpMV on the transposed matrix. This is already the right approach. Adding `cusparseSpMV_preprocess()` via raw cuSPARSE calls could squeeze out additional gains for repeated calls.

**However:** A custom binary-specialized kernel should beat all cuSPARSE paths because it eliminates value reads entirely. Use cuSPARSE ALG1 only as a performance baseline.

---

## 5. GPU-Side Bitpacking

### The Nuclear Option — Highest Potential Gain

Since every feature is binary (0 or 1), the entire matrix can be represented as a **bitfield** instead of float arrays.

#### B2SR (Binary Block Sparse Row) Format
Published research (B2SR) shows a two-level blocked format:
- **Top level:** Block-sparse row structure (which blocks are non-empty)
- **Bottom level:** Bit-packed tiles (32×32 or 16×16 bits per block)

#### Kernel Primitive
```
// Instead of: result[col] += data[k] * x[row]
// Do:         result[col] += __popc(tile_bits & x_bits)
```

Uses `__popc()` (population count), `__ballot_sync()`, and `__shfl_sync()` — warp-level intrinsics that are extremely fast on NVIDIA GPUs.

#### Storage Savings
| Representation | Per-nonzero storage | For 100M nnz |
|----------------|-------------------|--------------|
| CSR float32 (current) | 4 bytes (value) + 4 bytes (index) = 8 bytes | 800 MB |
| CSR implicit-ones | 4 bytes (index only) | 400 MB |
| Bitpacked tiles (32×32) | ~0.125 bytes (1 bit) + block overhead | ~50–100 MB |

**Up to 8–16× memory reduction** → more features fit in VRAM → larger matrices feasible.

#### Caveats
1. **Block fill ratio matters** — If cross-features don't cluster into co-activated groups, most 32×32 blocks will have only 1–2 set bits, wasting 1022–1023 bits of padding per block.
2. **Column reordering required** — Must sort/cluster features by co-activation frequency to maximize block fill. This is a preprocessing step, not a runtime cost.
3. **Implementation complexity** — Requires custom CUDA C kernels, not achievable through CuPy alone.
4. **Histogram integration** — The popcount result gives a count, but we need weighted sums (gradient accumulation). Need a mixed path: bitpacked structure identification + float gradient gather.

#### Feasibility Assessment
- **For histogram building:** Partially applicable. Can use bitpacking to identify which features are active in a row, then gather gradients. But the gather step still needs float arithmetic.
- **For cross-gen (SpGEMM):** Excellent fit. Binary × binary = binary, and popcount gives exact results.
- **For the fork overall:** Worth prototyping for cross-gen first, then evaluating for histograms.

---

## 6. Custom Kernel vs cuSPARSE — Which Wins?

### At Our Scale: Custom Kernel Wins

| Aspect | cuSPARSE | Custom Binary Kernel |
|--------|----------|---------------------|
| Value array reads | Always reads float data[] | Skips entirely (implicit 1.0) |
| Memory bandwidth | Full CSR bandwidth | ~33% less (no value array) |
| Histogram specialization | Generic SpMV output | Writes directly to histogram buffers |
| Bin-0 reconstruction | Separate subtraction pass | Fused: total − bin1 in same kernel |
| Leaf masking | Separate mask multiply | Fused: skip non-leaf rows in traversal |
| Atomics | Generic float atomics | Can use faster int atomics for counts |
| Warp scheduling | Library-chosen | Tuned for our sparsity pattern |
| Preprocessing | `cusparseSpMV_preprocess()` | Custom row-binning by nnz |

### Recommended Kernel Architecture

```
__global__ void binary_histogram_kernel(
    const int64_t* indptr,      // CSR row pointers (int64)
    const int32_t* indices,     // CSR column indices (int32)
    const float*   gradients,   // Per-row gradient values
    const float*   hessians,    // Per-row hessian values
    const int32_t* leaf_mask,   // 1 if row in current leaf, 0 otherwise
    float*         hist_grad,   // Output: per-feature gradient sum
    float*         hist_hess,   // Output: per-feature hessian sum
    int64_t        n_rows
) {
    // Warp-cooperative: each warp processes one row
    // No value array read — every nonzero contributes gradient[row] directly
    // Leaf mask checked once per row, not per nonzero
    // atomicAdd to histogram bins (or warp-private + reduce)
}
```

### Row-Length Binning
Cross-features have highly variable row lengths. Optimal strategy:
- **Short rows (nnz < 32):** One warp per row, each thread handles one nonzero
- **Medium rows (32–512):** One warp per row, threads loop over nonzeros
- **Long rows (512+):** Multiple warps per row, reduce across warps

This adaptive binning (similar to DCSR/CSR-Adaptive) prevents warp divergence and maximizes occupancy.

---

## 7. Multi-GPU Strategy (8× RTX 5090)

### Feature Partitioning (Recommended)
- Partition features (columns) across 8 GPUs
- Each GPU holds ~250K–1.25M features in CSR format
- Each GPU independently computes histogram for its feature partition
- All-reduce only the split-gain statistics (tiny: one float per feature candidate)

### Why Not Row Partitioning
- Row partitioning requires each GPU to have ALL features (duplicated)
- Gradient vectors must be synchronized across GPUs each iteration
- Feature partitioning keeps data local and only communicates split decisions

### Communication Pattern
```
Per boosting round:
1. Broadcast gradient/hessian vectors to all GPUs (~N×8 bytes, small)
2. Each GPU: histogram build on local feature partition (parallel, no communication)
3. Each GPU: find local best split candidate
4. All-reduce: global best split (one float + feature ID)
5. Broadcast: split decision to all GPUs
```

This is communication-efficient because histogram data (the large payload) never leaves the GPU.

---

## 8. Prioritized Optimization Roadmap

### Phase 1: Implicit-Value CSR (Easiest, ~30% bandwidth savings)
- Remove value array from GPU-side CSR representation
- Modify CuPy-based SpMV to custom kernel that skips value reads
- **Effort:** Medium (custom CUDA kernel via CuPy RawKernel or Numba CUDA)
- **Expected gain:** ~30% SpMV speedup from bandwidth reduction
- **Risk:** Low — same algorithm, just fewer memory reads

### Phase 2: Fused Histogram Kernel (Medium effort, ~2× over Phase 1)
- Fuse leaf masking + SpMV + bin-0 subtraction into single kernel
- Add row-length binning for warp scheduling
- Write directly to histogram buffer layout expected by LightGBM
- **Effort:** High (custom CUDA C kernel)
- **Expected gain:** 2–3× over current CuPy SpMV approach
- **Risk:** Medium — requires careful integration with LightGBM training loop

### Phase 3: cuSPARSE Preprocessing Baseline (Low effort)
- Add `cusparseSpMV_preprocess()` to current CuPy path as comparison baseline
- Use `CSR_ALG1` for speed, `CSR_ALG2` for validation
- **Effort:** Low (CuPy exposes raw cuSPARSE handles)
- **Expected gain:** 10–20% over current CuPy default algorithm
- **Risk:** Low

### Phase 4: Bitpacked Binary Format (Highest effort, highest ceiling)
- Implement B2SR-style bitpacked tiles for cross-gen SpGEMM first
- Profile block fill ratio on real cross-feature matrices
- If fill ratio > 10%, extend to histogram building
- **Effort:** Very high (custom CUDA C, new format converter, new kernels)
- **Expected gain:** 8–16× memory reduction, potentially 3–5× kernel speedup
- **Risk:** High — depends on co-activation structure of cross-features

### Phase 5: Multi-GPU Feature Partitioning (For 8× RTX 5090)
- Implement feature-parallel histogram building across GPUs
- NCCL all-reduce for split candidates
- **Effort:** High
- **Expected gain:** Near-linear scaling to 8 GPUs
- **Risk:** Medium — NCCL integration with custom training loop

---

## 9. Key Decisions Summary

| Question | Answer | Confidence |
|----------|--------|------------|
| Is CSR optimal? | **Yes** — CSR is the right base format for our skewed binary features | High |
| Should we switch to BSR/ELL/HYB? | **No** — all are worse for irregular singleton binary activations | High |
| Can we exploit all-1.0 values? | **Yes — this is the #1 optimization.** Skip value reads entirely | High |
| cuSPARSE vs custom kernel? | **Custom wins** at our scale due to implicit-value + fused histogram | High |
| Bitpacking worth it? | **Maybe** — depends on block fill ratio. Prototype on cross-gen first | Medium |
| Which cuSPARSE algorithm? | ALG1 for speed, ALG2 for deterministic validation only | High |
| Multi-GPU strategy? | **Feature partitioning** — row partitioning duplicates too much data | High |
| `cusparseSpMV_preprocess()`? | **Yes** — free speedup for repeated same-structure SpMV | High |

---

## 10. What NOT to Do

1. **Never convert to dense** — 2–10M features × N rows = hundreds of GB. Dense is impossible.
2. **Never use ELL for skewed sparsity** — padding overhead destroys bandwidth savings.
3. **Never hash or prune features** — rare signals are the entire edge. Exact preservation is non-negotiable.
4. **Never use float16 for histogram accumulation** — gradient sums need float32 minimum for split-gain precision.
5. **Never row-partition sparse data across GPUs** — duplicates the full feature matrix on every GPU.
6. **Never rely on stock LightGBM CUDA path** — it doesn't properly support sparse features (confirmed by maintainer issues #6725, #6631).

---

*Research sources: cuSPARSE 12.6/13.2 documentation, LightGBM GPU performance guide, B2SR binary sparse format, Utah DCSR research, NVIDIA Developer Forums, academic SpMV optimization literature (2024–2025).*
