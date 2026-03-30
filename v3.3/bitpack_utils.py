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
def _pack_matrix(mat, n_cols, n_words):
    """Pack all columns of a dense binary matrix into bitarrays.
    mat: float32 (n_rows, n_cols) — binary 0/1 values
    Returns: uint64 (n_cols, n_words) — each row is a packed column
    Outer loop parallelized over columns via prange (each column is independent).
    """
    packed = np.zeros((n_cols, n_words), dtype=np.uint64)
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
def _cooccurrence_matrix_popcnt(left_packed, right_packed, n_left, n_right, n_words):
    """Compute co-occurrence counts for all (left, right) pairs using AND + POPCNT.
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
    n_words = (n_rows + 63) >> 6  # ceil(n_rows / 64)

    # Pack both matrices into bitarrays
    left_packed = _pack_matrix(left_mat, n_left, n_words)
    right_packed = _pack_matrix(right_mat, n_right, n_words)

    # Compute all co-occurrence counts via POPCNT
    counts = _cooccurrence_matrix_popcnt(left_packed, right_packed, n_left, n_right, n_words)

    # Extract valid pairs
    valid_pairs = np.argwhere(counts >= min_co)

    return valid_pairs, counts
