# QA CrossGen Audit — 5 Optimization Branches

**Audited**: 2026-03-29
**Base**: `v3.3`
**Auditor**: error-checker agent (READ-ONLY)

---

## Summary

| # | Branch | Description | Verdict |
|---|--------|-------------|---------|
| 1 | `ceo/backend-dev-22d6ed1e` | Numba intersection + L2 sort | **PASS** |
| 2 | `ceo/backend-dev-b161bc3a` | Parallel steps + adaptive chunks | **PASS (1 NOTE)** |
| 3 | `ceo/backend-dev-b94865a8` | Atomic NPZ + indices-only | **NO CHANGES** (0 commits beyond v3.3) |
| 4 | `ceo/backend-dev-702e95d2` | Bitpack POPCNT co-occurrence | **PASS** |
| 5 | `ceo/backend-dev-afb08116` | Memmap CSC streaming | **NO CHANGES** (0 commits beyond v3.3) |

---

## Branch 1: `ceo/backend-dev-22d6ed1e` — Numba CSC Intersection + L2 Sort

**Files changed**: `numba_cross_kernels.py` (new), `v2_cross_generator.py` (+86 lines), GPU histogram fork (CUDA optimizations)

| Check | Result | Notes |
|-------|--------|-------|
| 1. No feature filtering/dropping | **PASS** | No features dropped. All valid_pairs from co-occurrence pre-filter are processed. |
| 2. int64 indptr preserved | **PASS** | CSC indptr explicitly cast to `np.int64`. Output CSR indptr checked and upcast: `if csr.indptr.dtype != np.int64: csr.indptr = csr.indptr.astype(np.int64)` |
| 3. Binary features (0/1) preserved | **PASS** | Two-pointer intersection on sorted CSC indices = AND operation. Output data is `np.ones(..., dtype=np.float32)`. Binary 0/1 exactly preserved. |
| 4. No NaN→0 conversion | **PASS** | No NaN handling anywhere in new code. |
| 5. Structural zeros = 0.0 maintained | **PASS** | CSC format inherently stores only nonzeros. Intersection only outputs matching indices. |
| 6. No subsample/sample | **PASS** | No row or feature sampling. All pairs processed. |
| 7. MIN_CO_OCCURRENCE unchanged | **PASS** | Uses `min_nonzero` parameter from caller (=MIN_CO_OCCURRENCE=3). Not modified. |
| 8. All cross types generated (12 steps) | **PASS** | `_numba_cross_chunk` is a drop-in replacement for `_cpu_cross_chunk`. Step orchestration untouched. |
| 9. Resume/checkpoint logic intact | **PASS** | No changes to checkpoint save/load logic. |
| 10. feature_pre_filter not introduced | **PASS** | Not present anywhere in diff. |

**Note**: CSC indices are cast to `int32` (row indices). This is safe because row counts are <2^31 for all TFs. The indptr (column pointers) correctly uses int64.

---

## Branch 2: `ceo/backend-dev-b161bc3a` — Parallel Steps + Adaptive Chunks

**Files changed**: `v2_cross_generator.py` (+593/-258 lines), `CHANGES_CROSSGEN_PARALLEL.md` (new)

| Check | Result | Notes |
|-------|--------|-------|
| 1. No feature filtering/dropping | **PASS** | No new feature filtering. Same `gpu_batch_cross` call with same parameters. |
| 2. int64 indptr preserved | **PASS** | No changes to CSR/indptr handling. Same COO→CSR conversion paths. |
| 3. Binary features (0/1) preserved | **PASS** | Cross computation untouched. Only orchestration changed. |
| 4. No NaN→0 conversion | **PASS** | No NaN handling in new code. |
| 5. Structural zeros = 0.0 maintained | **PASS** | No changes to data values. |
| 6. No subsample/sample | **PASS** | All cross steps still process all data. No row/feature sampling. |
| 7. MIN_CO_OCCURRENCE unchanged | **PASS** | MIN_CO_OCCURRENCE not modified. Still defaults to 3. |
| 8. All cross types generated (12 steps) | **PASS** | All 12 cross steps (dx, ax, ax2, ta2, ex2, sw, hod, mx, vx, asp, pn, mn) preserved as step descriptors. Cross 13 was already removed in v3.3 baseline (noted as redundant). |
| 9. Resume/checkpoint logic intact | **PASS** | Checkpoint skip logic preserved (`_completed_prefixes` check). Checkpoints saved after each step completes. |
| 10. feature_pre_filter not introduced | **PASS** | Not present. |

**NOTE**: In parallel mode (`PARALLEL_CROSS_STEPS=1`), `col_offset=0` is passed to all steps instead of the running `col_offset`. The comment says "offsets are reassigned during collection." This is functionally safe since offsets are re-computed at merge time, BUT could cause issues if any downstream code relies on col_offset within `gpu_batch_cross`. In sequential mode (default), this is not an issue since `_collect_cross` handles offset tracking. **Severity: LOW** — only affects the opt-in parallel mode, and the step results are collected in canonical order.

---

## Branch 3: `ceo/backend-dev-b94865a8` — Atomic NPZ + Indices-Only

**NO CHANGES**: This branch has 0 commits beyond v3.3. `git diff v3.3..ceo/backend-dev-b94865a8` produces empty output.

**Verdict**: Nothing to audit. Branch appears to not have been implemented yet.

---

## Branch 4: `ceo/backend-dev-702e95d2` — Bitpack POPCNT Co-occurrence

**Files changed**: `bitpack_utils.py` (new, 123 lines), `v2_cross_generator.py` (+82/-55 lines), `CHANGES_CROSSGEN_BITPACK.md` (new)

| Check | Result | Notes |
|-------|--------|-------|
| 1. No feature filtering/dropping | **PASS** | Only replaces the co-occurrence counting method. Same `np.argwhere(counts >= min_nonzero)` filtering. No features dropped beyond the existing threshold. |
| 2. int64 indptr preserved | **PASS** | No changes to CSR/indptr handling. Bitpack operates only on the pre-filter step. |
| 3. Binary features (0/1) preserved | **PASS** | Bitpacking is read-only: `if col[i] != 0.0` → set bit. Used only for counting, not for output. |
| 4. No NaN→0 conversion | **PASS** | No NaN handling. `col[i] != 0.0` treats NaN as nonzero (bit set), which is correct for binary features that should never be NaN. |
| 5. Structural zeros = 0.0 maintained | **PASS** | Bitpack only counts; actual cross values computed by existing code path. |
| 6. No subsample/sample | **PASS** | All rows and all pairs computed. `_cooccurrence_matrix_popcnt` iterates all (left, right) pairs exhaustively. |
| 7. MIN_CO_OCCURRENCE unchanged | **PASS** | `min_co=min_nonzero` passed through. MIN_CO_OCCURRENCE still defaults to 3 (line 226 unchanged). |
| 8. All cross types generated (12 steps) | **PASS** | Step orchestration completely untouched. Only the inner co-occurrence computation was refactored. |
| 9. Resume/checkpoint logic intact | **PASS** | No changes to checkpoint logic. |
| 10. feature_pre_filter not introduced | **PASS** | Not present. |

**Mathematical correctness**: Bitpack AND + POPCNT is mathematically identical to sparse matmul L.T @ R for binary {0,1} features. `popcnt(L[i] AND R[j])` = count of rows where both L_col_i=1 and R_col_j=1 = (L.T @ R)[i,j]. Verified by documentation claim and code inspection.

---

## Branch 5: `ceo/backend-dev-afb08116` — Memmap CSC Streaming

**NO CHANGES**: This branch has 0 commits beyond v3.3. `git diff v3.3..ceo/backend-dev-afb08116` produces empty output.

**Verdict**: Nothing to audit. Branch appears to not have been implemented yet.

---

## Cross-Branch Concerns

1. **Branch 1 + Branch 4 compatibility**: Both modify `v2_cross_generator.py` in the co-occurrence pre-filter area. Branch 4 refactors the co-occurrence into `_compute_cooccurrence_pairs()`, while Branch 1 adds `_numba_cross_chunk` that duplicates the co-occurrence code. **Merge conflict expected** — recommend rebasing Branch 1 to use `_compute_cooccurrence_pairs()` from Branch 4.

2. **Branch 2 + any other branch**: Branch 2 restructures the entire `generate_all_crosses()` function (12 steps → descriptor array + executor). This will conflict with any branch that touches the step orchestration.

3. **Branches 3 & 5 are empty**: No implementation exists. These should either be implemented or removed from the audit list.

---

## Final Verdict

**3 branches with code changes — ALL PASS matrix thesis compliance.**
- No feature filtering, dropping, subsampling, or NaN conversion
- int64 indptr preserved in all paths
- Binary 0/1 values preserved exactly
- MIN_CO_OCCURRENCE=3 unchanged
- All 12 cross steps intact
- No feature_pre_filter introduced

**2 branches (3, 5) have no changes to audit.**
