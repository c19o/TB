#!/usr/bin/env python
"""
bitpack_utils.py — Bitpacked POPCNT co-occurrence pre-filter
=============================================================
Packs binary feature columns into uint64 bitarrays, then uses
hardware POPCNT (via Numba LLVM intrinsic) to count co-occurrences
in 8-21ms for millions of pairs.

Mathematically identical to sparse matmul L.T @ R for binary features.
Used as a fast pre-filter BEFORE expensive cross generation.

Usage:
    from bitpack_utils import bitpack_cooccurrence_filter
    valid_pairs = bitpack_cooccurrence_filter(left_mat, right_mat, min_co=3)
"""

import numpy as np
from numba import njit, prange, types
from numba.extending import intrinsic
from llvmlite import ir


# ── Hardware POPCNT via LLVM intrinsic ──

@intrinsic
def _llvm_ctpop_i64(typingctx, val):
    """Emit llvm.ctpop.i64 — hardware POPCNT on x86."""
    sig = types.int64(types.uint64)
    def codegen(context, builder, signature, args):
        [val] = args
        fn_type = ir.FunctionType(ir.IntType(64), [ir.IntType(64)])
        fn = builder.module.declare_intrinsic('llvm.ctpop.i64', fnty=fn_type)
        return builder.call(fn, [val])
    return sig, codegen


# ── Pack a dense binary column into uint64 bitarray ──


@njit(parallel=True, cache=True)
def _pack_matrix(mat, n_cols, n_words_padded):
    """Pack all columns of a dense binary matrix into bitarrays.
    mat: float32 (n_rows, n_cols) — binary 0/1 values
    n_words_padded: padded to multiple of 8 for cache-aligned unroll
    Returns: uint64 (n_cols, n_words_padded) — each row is a packed column
    Extra padding words stay zero: AND(x, 0)=0, POPCNT(0)=0 — mathematically neutral.
    Outer loop parallelized over columns via prange (each column is independent).
    """
    packed = np.zeros((n_cols, n_words_padded), dtype=np.uint64)
    n_rows = mat.shape[0]
    for c in prange(n_cols):
        for i in range(n_rows):
            if mat[i, c] != 0.0:
                word_idx = i >> 6
                bit_idx = i & 63
                packed[c, word_idx] |= np.uint64(1) << np.uint64(bit_idx)
    return packed


# ── POPCNT-based co-occurrence counting for all pairs ──

@njit(parallel=True, cache=True)
def _cooccurrence_matrix_popcnt_naive(left_packed, right_packed, n_left, n_right, n_words):
    """LEGACY: Naive co-occurrence kernel. Kept for reference/fallback.
    left_packed:  uint64 (n_left, n_words)
    right_packed: uint64 (n_right, n_words)
    Returns: int32 (n_left, n_right) — co-occurrence counts
    """
    counts = np.zeros((n_left, n_right), dtype=np.int32)
    for li in prange(n_left):
        for ri in range(n_right):
            c = np.int32(0)
            for w in range(n_words):
                bits = left_packed[li, w] & right_packed[ri, w]
                c += np.int32(_llvm_ctpop_i64(bits))
            counts[li, ri] = c
    return counts


@njit(parallel=True, cache=True)
def _cooccurrence_matrix_popcnt_tiled(left_packed, right_packed, n_words, counts):
    """POPCNT co-occurrence kernel with unroll-by-8 inner loop.

    - prange over left rows (outermost) — Numba parfors-safe
    - Unroll-by-8: n_words padded to multiple of 8, so no remainder loop needed
    - AND + POPCNT on each word pair, accumulate per (li, ri) pair

    Parameters
    ----------
    left_packed  : uint64 (n_left, n_words)
    right_packed : uint64 (n_right, n_words)
    n_words      : int — MUST be multiple of 8 (padded by caller)
    counts       : int32 (n_left, n_right) — output, pre-allocated zeros
    """
    n_left = left_packed.shape[0]
    n_right = right_packed.shape[0]

    for li in prange(n_left):
        for ri in range(n_right):
            c = 0
            # Unroll-by-8: n_words/8 iterations, no remainder
            for w in range(0, n_words, 8):
                c += _llvm_ctpop_i64(left_packed[li, w] & right_packed[ri, w])
                c += _llvm_ctpop_i64(left_packed[li, w + 1] & right_packed[ri, w + 1])
                c += _llvm_ctpop_i64(left_packed[li, w + 2] & right_packed[ri, w + 2])
                c += _llvm_ctpop_i64(left_packed[li, w + 3] & right_packed[ri, w + 3])
                c += _llvm_ctpop_i64(left_packed[li, w + 4] & right_packed[ri, w + 4])
                c += _llvm_ctpop_i64(left_packed[li, w + 5] & right_packed[ri, w + 5])
                c += _llvm_ctpop_i64(left_packed[li, w + 6] & right_packed[ri, w + 6])
                c += _llvm_ctpop_i64(left_packed[li, w + 7] & right_packed[ri, w + 7])
            counts[li, ri] = c


def bitpack_cooccurrence_filter(left_mat, right_mat, min_co=3):
    """
    Fast co-occurrence pre-filter using bitpacked AND + hardware POPCNT.

    Parameters
    ----------
    left_mat : ndarray (n_rows, n_left) — binary 0/1 float32
    right_mat : ndarray (n_rows, n_right) — binary 0/1 float32
    min_co : int — minimum co-occurrence threshold (default 3)

    Returns
    -------
    valid_pairs : ndarray (n_valid, 2) — indices of pairs meeting threshold
        Column 0 = left index, column 1 = right index
    """
    n_rows = left_mat.shape[0]
    n_left = left_mat.shape[1]
    n_right = right_mat.shape[1]
    n_words_raw = (n_rows + 63) >> 6  # ceil(n_rows / 64)
    n_words = ((n_words_raw + 7) >> 3) << 3  # round up to multiple of 8 for cache-aligned unroll

    # Pack both matrices into bitarrays (padded width — extra words stay zero)
    left_packed = _pack_matrix(left_mat, n_left, n_words)
    right_packed = _pack_matrix(right_mat, n_right, n_words)

    # Compute all co-occurrence counts via tiled POPCNT kernel
    counts = np.zeros((n_left, n_right), dtype=np.int32)
    _cooccurrence_matrix_popcnt_tiled(left_packed, right_packed, n_words, counts)

    # Extract valid pairs
    valid_pairs = np.argwhere(counts >= min_co)

    valid_counts = counts[valid_pairs[:, 0], valid_pairs[:, 1]] if len(valid_pairs) > 0 else np.array([], dtype=counts.dtype)
    return valid_pairs, valid_counts
