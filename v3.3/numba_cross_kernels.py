"""
numba_cross_kernels.py — Numba-accelerated CSC sorted-index intersection for cross gen
=======================================================================================
Optimizations #1 (Numba prange sorted-index intersection) and #6 (L2 cache-friendly
pair sorting) from OPTIMIZATION_PLAN.md Phase 1A and 1F.

Enable with: USE_NUMBA_CROSS=1 environment variable.

Binary features (0/1) — cross = AND = sorted-index intersection on CSC columns.
Eliminates dense allocation + np.nonzero overhead from _process_cross_block.

Single-pass fusion: upper-bound pre-allocate + one parallel kernel (no count pass).
"""

import numpy as np
from numba import njit, prange, types
from numba.extending import intrinsic
from llvmlite import ir


# ============================================================
# LLVM intrinsics for popcount (co-occurrence counting)
# ============================================================

@intrinsic
def _llvm_ctpop_i64(typingctx, x):
    """LLVM ctpop.i64 — count set bits in a 64-bit integer."""
    sig = types.int64(types.int64)
    def codegen(context, builder, signature, args):
        fn_type = ir.FunctionType(ir.IntType(64), [ir.IntType(64)])
        fn = builder.module.declare_intrinsic('llvm.ctpop', [ir.IntType(64)], fn_type)
        return builder.call(fn, args)
    return sig, codegen


# ============================================================
# SINGLE-PASS KERNEL: Two-pointer merge writes matches directly
# ============================================================

@njit(parallel=True, cache=True)
def _intersect_single_pass(left_indptr, left_indices, right_indptr, right_indices,
                            pair_left, pair_right, offsets, out_rows, counts):
    """Single-pass: for each pair, two-pointer merge writes matches directly."""
    n_pairs = len(pair_left)
    for p in prange(n_pairs):
        li = pair_left[p]
        ri = pair_right[p]
        a_start, a_end = left_indptr[li], left_indptr[li + 1]
        b_start, b_end = right_indptr[ri], right_indptr[ri + 1]
        write_pos = offsets[p]
        i, j = a_start, b_start
        c = 0
        while i < a_end and j < b_end:
            if left_indices[i] == right_indices[j]:
                out_rows[write_pos + c] = left_indices[i]
                c += 1
                i += 1
                j += 1
            elif left_indices[i] < right_indices[j]:
                i += 1
            else:
                j += 1
        counts[p] = c


# ============================================================
# OPTIMIZATION #6: L2 cache-friendly pair sorting
# ============================================================

def sort_pairs_l2_friendly(pair_left, pair_right):
    """
    Sort cross-feature pairs so the same left column stays hot in L2 cache
    across its right-side partners.

    Groups pairs by left column index. Within each group, right indices
    are sorted for sequential access. This means each left CSC column
    is loaded once and reused for all its right-side partners (~100 on avg).

    Without this: left column reloaded 100x (once per right partner, scattered).
    With this: left column loaded once, stays in L2 for all partners.

    Parameters
    ----------
    pair_left : int32 array of left column indices
    pair_right : int32 array of right column indices

    Returns
    -------
    sorted_left : int32 array, sorted by (left_col, right_col)
    sorted_right : int32 array, matching sorted order
    original_indices : int32 array, maps sorted position -> original position
    """
    # Lexicographic sort: primary key = left, secondary = right
    sort_order = np.lexsort((pair_right, pair_left))
    return (pair_left[sort_order].astype(np.int32),
            pair_right[sort_order].astype(np.int32),
            sort_order.astype(np.int32))


# ============================================================
# HIGH-LEVEL API: Numba CSC cross generation
# ============================================================

def numba_csc_cross(left_mat, right_mat, valid_pairs, all_names,
                    n_rows, min_nonzero=3, progress=True):
    """
    Generate cross features using Numba CSC sorted-index intersection.

    Replaces _process_cross_block's dense multiply + np.nonzero path.

    Parameters
    ----------
    left_mat : ndarray (n_rows, n_left) — dense binary (0/1) float32
    right_mat : ndarray (n_rows, n_right) — dense binary (0/1) float32
    valid_pairs : ndarray (n_pairs, 2) — [left_col_idx, right_col_idx] pairs
                  (already filtered by co-occurrence >= min_nonzero)
    all_names : list of str — pre-computed feature names for each pair
    n_rows : int — number of data rows
    min_nonzero : int — minimum intersection size (pairs below this are empty)
    progress : bool — print progress updates

    Returns
    -------
    names : list of str — feature names for non-empty crosses
    csr_chunks : list of scipy CSR matrices — cross feature chunks
    n_features : int — total number of features generated
    """
    from scipy import sparse
    import gc
    import time

    n_pairs = len(valid_pairs)
    if n_pairs == 0:
        return [], [], 0

    t0 = time.time()

    # ── Convert dense binary matrices to CSC sparse ──
    # CSC gives us sorted row indices per column — perfect for two-pointer intersection
    left_csc = sparse.csc_matrix(left_mat.astype(np.float32))
    right_csc = sparse.csc_matrix(right_mat.astype(np.float32))

    # Extract CSC arrays (contiguous int32 indices, int64 indptr)
    l_indptr = np.ascontiguousarray(left_csc.indptr.astype(np.int64))
    l_indices = np.ascontiguousarray(left_csc.indices.astype(np.int32))
    r_indptr = np.ascontiguousarray(right_csc.indptr.astype(np.int64))
    r_indices = np.ascontiguousarray(right_csc.indices.astype(np.int32))

    pair_left = valid_pairs[:, 0].astype(np.int32)
    pair_right = valid_pairs[:, 1].astype(np.int32)

    # ── Optimization #6: Sort pairs for L2 cache reuse ──
    sorted_left, sorted_right, orig_indices = sort_pairs_l2_friendly(pair_left, pair_right)
    # Map names to sorted order
    sorted_names = [all_names[orig_indices[i]] for i in range(n_pairs)]

    if progress:
        t_csc = time.time()
        print(f"[numba_cross] CSC conversion + pair sort: {t_csc - t0:.2f}s "
              f"({n_pairs:,} pairs, L={left_mat.shape[1]} cols, R={right_mat.shape[1]} cols)",
              flush=True)

    # Free dense matrices from caller's perspective (CSC holds the data now)
    del left_csc, right_csc

    # ── Single-pass fusion: upper-bound allocate + intersect in one kernel ──
    # Compute upper bounds per pair: min(nnz_left, nnz_right)
    nnz_left = np.empty(n_pairs, dtype=np.int64)
    nnz_right = np.empty(n_pairs, dtype=np.int64)
    for p in range(n_pairs):
        nnz_left[p] = l_indptr[sorted_left[p] + 1] - l_indptr[sorted_left[p]]
        nnz_right[p] = r_indptr[sorted_right[p] + 1] - r_indptr[sorted_right[p]]
    ub = np.minimum(nnz_left, nnz_right)

    # Compute offsets from upper bounds
    offsets = np.zeros(n_pairs + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(ub)
    total_ub = int(offsets[n_pairs])

    if total_ub == 0:
        return [], [], 0

    # Pre-allocate to upper bound and run single-pass kernel
    out_rows = np.empty(total_ub, dtype=np.int32)
    counts = np.empty(n_pairs, dtype=np.int64)
    _intersect_single_pass(l_indptr, l_indices, r_indptr, r_indices,
                           sorted_left, sorted_right, offsets, out_rows, counts)

    if progress:
        t_fill = time.time()
        n_nonempty = np.count_nonzero(counts)
        total_nnz_val = int(counts.sum())
        print(f"[numba_cross] Single-pass intersect: {t_fill - t_csc:.2f}s — "
              f"{n_nonempty:,}/{n_pairs:,} non-empty, {total_nnz_val:,} NNZ "
              f"(upper-bound alloc: {total_ub:,})", flush=True)
    else:
        total_nnz_val = int(counts.sum())

    if total_nnz_val == 0:
        del out_rows, offsets, counts, ub, nnz_left, nnz_right
        gc.collect()
        return [], [], 0

    # ── Build CSR output in chunks to bound memory ──
    # Process in chunks of CHUNK_SIZE features to avoid a single massive COO
    CHUNK_SIZE = max(5000, min(50000, n_pairs // 4))

    csr_chunks = []
    names = []
    n_features = 0

    nonempty_mask = counts > 0
    nonempty_indices = np.where(nonempty_mask)[0]
    n_nonempty = len(nonempty_indices)

    for chunk_start in range(0, n_nonempty, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, n_nonempty)
        chunk_pairs = nonempty_indices[chunk_start:chunk_end]
        n_chunk = len(chunk_pairs)

        # Build COO arrays for this chunk
        chunk_rows_list = []
        chunk_cols_list = []
        chunk_data_list = []
        chunk_names = []

        for local_col, p_idx in enumerate(chunk_pairs):
            p_start = int(offsets[p_idx])
            p_end = p_start + int(counts[p_idx])
            if p_end > p_start:
                rows_slice = out_rows[p_start:p_end]
                chunk_rows_list.append(rows_slice)
                chunk_cols_list.append(np.full(len(rows_slice), local_col, dtype=np.int32))
                chunk_data_list.append(np.ones(len(rows_slice), dtype=np.float32))
                chunk_names.append(sorted_names[p_idx])

        if chunk_rows_list:
            all_r = np.concatenate(chunk_rows_list)
            all_c = np.concatenate(chunk_cols_list)
            all_d = np.concatenate(chunk_data_list)
            csr = sparse.coo_matrix((all_d, (all_r, all_c)),
                                    shape=(n_rows, n_chunk)).tocsr()
            # Ensure int64 indptr for NNZ > 2^31 safety
            if csr.indptr.dtype != np.int64:
                csr.indptr = csr.indptr.astype(np.int64)
            csr_chunks.append(csr)
            names.extend(chunk_names)
            n_features += len(chunk_names)

            del all_r, all_c, all_d, csr

        del chunk_rows_list, chunk_cols_list, chunk_data_list, chunk_names

    if progress:
        t_end = time.time()
        print(f"[numba_cross] CSR build: {t_end - t_fill:.2f}s — "
              f"{n_features:,} features in {len(csr_chunks)} chunks", flush=True)
        print(f"[numba_cross] Total: {t_end - t0:.2f}s "
              f"(was dense multiply + np.nonzero)", flush=True)

    del out_rows, offsets, counts, ub, nnz_left, nnz_right
    gc.collect()

    return names, csr_chunks, n_features


def warmup_numba_kernels():
    """Pre-compile Numba kernels with tiny arrays to avoid JIT overhead on first real call."""
    dummy_indptr = np.array([0, 2, 4], dtype=np.int64)
    dummy_indices = np.array([0, 1, 1, 2], dtype=np.int32)
    pair_l = np.array([0, 1], dtype=np.int32)
    pair_r = np.array([1, 0], dtype=np.int32)

    # Upper bounds: min(nnz_left, nnz_right) for each pair
    offsets = np.array([0, 2, 4], dtype=np.int64)
    out = np.empty(4, dtype=np.int32)
    counts = np.empty(2, dtype=np.int64)

    # Trigger compilation of single-pass kernel
    _intersect_single_pass(dummy_indptr, dummy_indices,
                           dummy_indptr, dummy_indices,
                           pair_l, pair_r, offsets, out, counts)
