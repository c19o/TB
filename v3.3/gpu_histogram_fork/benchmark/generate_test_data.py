#!/usr/bin/env python3
"""
Generate synthetic sparse binary CSR data matching real training profiles.

Produces benchmark data for GPU histogram kernel testing:
- Sparse binary (0/1) CSR matrices with int32 indices, int64 indptr
- 3-class softmax gradient/hessian arrays
- Leaf assignment masks (depth-6 tree)
- EFB bundle mappings (255 features per bundle)

Profiles match actual v3.3 cross-feature dimensions per timeframe.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse


# ---------------------------------------------------------------------------
# Real training profiles  (rows, features, density)
# ---------------------------------------------------------------------------
PROFILES = {
    "1w":  {"rows":    818, "cols":  2_200_000, "density": 0.003},
    "1d":  {"rows":  5_733, "cols":  6_000_000, "density": 0.003},
    "4h":  {"rows": 17_520, "cols":  3_000_000, "density": 0.003},
    "1h":  {"rows": 75_405, "cols":  8_000_000, "density": 0.002},
    "15m": {"rows": 293_980, "cols": 10_000_000, "density": 0.0015},
}


def get_profile(name: str) -> dict:
    """Return profile dict for a timeframe name."""
    if name not in PROFILES:
        raise ValueError(f"Unknown profile '{name}'. Choose from: {list(PROFILES.keys())}")
    return PROFILES[name].copy()


# ---------------------------------------------------------------------------
# Sparse binary matrix
# ---------------------------------------------------------------------------
def generate_sparse_binary(rows: int, cols: int, density: float,
                           seed: int = 42) -> sparse.csr_matrix:
    """
    Generate a sparse binary CSR matrix.

    Uses int32 indices and int64 indptr so the matrix can represent
    >2^31 non-zeros (needed for 15m with 10M cols).

    Parameters
    ----------
    rows : int       Number of rows (samples).
    cols : int       Number of columns (features).
    density : float  Fraction of non-zero entries (0-1).
    seed : int       RNG seed for reproducibility.

    Returns
    -------
    scipy.sparse.csr_matrix with dtype=np.float32, int32 indices, int64 indptr.
    """
    rng = np.random.RandomState(seed)
    nnz_per_row = max(1, int(cols * density))
    total_nnz = rows * nnz_per_row

    # Build COO-style arrays in chunks to keep peak memory bounded
    CHUNK = 500_000  # rows per chunk
    indptr = np.zeros(rows + 1, dtype=np.int64)
    indices_parts = []
    offset = 0

    for start in range(0, rows, CHUNK):
        end = min(start + CHUNK, rows)
        chunk_rows = end - start

        # Random column indices per row (no duplicates within a row)
        chunk_indices = np.empty(chunk_rows * nnz_per_row, dtype=np.int32)
        for i in range(chunk_rows):
            chunk_indices[i * nnz_per_row:(i + 1) * nnz_per_row] = \
                rng.choice(cols, size=nnz_per_row, replace=False).astype(np.int32)

        indices_parts.append(chunk_indices)

        # Fill indptr for this chunk
        for i in range(chunk_rows):
            indptr[start + i + 1] = indptr[start + i] + nnz_per_row

    indices = np.concatenate(indices_parts)
    data = np.ones(len(indices), dtype=np.float32)

    mat = sparse.csr_matrix((data, indices, indptr), shape=(rows, cols))
    # Ensure correct dtypes
    mat.indices = mat.indices.astype(np.int32)
    mat.indptr = mat.indptr.astype(np.int64)
    return mat


# ---------------------------------------------------------------------------
# Gradients / hessians  (softmax cross-entropy style)
# ---------------------------------------------------------------------------
def generate_gradients(rows: int, num_class: int = 3,
                       seed: int = 42) -> tuple:
    """
    Generate realistic gradient and hessian arrays for multi-class softmax.

    For softmax CE:  g_k = p_k - y_k   (range ~ [-1, 1])
                     h_k = p_k(1-p_k)  (range ~ [0, 0.25], we scale to [0,2])

    Parameters
    ----------
    rows : int        Number of samples.
    num_class : int   Number of classes (default 3).
    seed : int        RNG seed.

    Returns
    -------
    (gradients, hessians) each shape (rows, num_class), dtype float32.
    """
    rng = np.random.RandomState(seed)

    # Simulate softmax probabilities
    logits = rng.randn(rows, num_class).astype(np.float32)
    logits -= logits.max(axis=1, keepdims=True)  # numerical stability
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    # One-hot labels
    labels = rng.randint(0, num_class, size=rows)
    one_hot = np.zeros((rows, num_class), dtype=np.float32)
    one_hot[np.arange(rows), labels] = 1.0

    gradients = (probs - one_hot).astype(np.float32)           # [-1, 1]
    hessians = (2.0 * probs * (1.0 - probs)).astype(np.float32)  # [0, 2]

    return gradients, hessians


# ---------------------------------------------------------------------------
# Leaf masks  (depth-6 tree => up to 63 internal nodes, 64 leaves)
# ---------------------------------------------------------------------------
def generate_leaf_masks(rows: int, num_leaves: int = 63,
                        seed: int = 42) -> tuple:
    """
    Simulate leaf assignments for a depth-6 tree.

    Parameters
    ----------
    rows : int         Number of samples.
    num_leaves : int   Number of leaves (default 63).
    seed : int         RNG seed.

    Returns
    -------
    (leaf_assignment, leaf_indices)
        leaf_assignment : ndarray shape (rows,) int32 — leaf id per row.
        leaf_indices    : list of num_leaves ndarrays — row indices per leaf.
    """
    rng = np.random.RandomState(seed)

    # Non-uniform distribution: some leaves get more rows (realistic)
    weights = rng.dirichlet(np.ones(num_leaves) * 0.5)
    leaf_assignment = rng.choice(num_leaves, size=rows, p=weights).astype(np.int32)

    leaf_indices = []
    for leaf_id in range(num_leaves):
        leaf_indices.append(np.where(leaf_assignment == leaf_id)[0].astype(np.int32))

    return leaf_assignment, leaf_indices


# ---------------------------------------------------------------------------
# EFB bundle mapping
# ---------------------------------------------------------------------------
def generate_efb_mapping(n_cols: int,
                         features_per_bundle: int = 255) -> tuple:
    """
    Generate Exclusive Feature Bundling (EFB) mapping arrays.

    Maps each column to a bundle id and a bin offset within that bundle.
    LightGBM packs mutually exclusive binary features into uint8 bins.

    Parameters
    ----------
    n_cols : int              Total number of features.
    features_per_bundle : int Max features per bundle (default 255 for uint8).

    Returns
    -------
    (col_to_bundle, col_to_bin_offset) each shape (n_cols,) int32.
    """
    col_to_bundle = np.arange(n_cols, dtype=np.int32) // features_per_bundle
    col_to_bin_offset = (np.arange(n_cols, dtype=np.int32) % features_per_bundle).astype(np.int32)
    return col_to_bundle, col_to_bin_offset


# ---------------------------------------------------------------------------
# Disk I/O helpers
# ---------------------------------------------------------------------------
def _save_profile(out_dir: Path, name: str, mat, grad, hess,
                  leaf_assign, leaf_idx, col2bundle, col2bin):
    """Save all arrays for one profile to disk."""
    d = out_dir / name
    d.mkdir(parents=True, exist_ok=True)

    sparse.save_npz(str(d / "X.npz"), mat)
    np.save(str(d / "gradients.npy"), grad)
    np.save(str(d / "hessians.npy"), hess)
    np.save(str(d / "leaf_assignment.npy"), leaf_assign)
    np.savez(str(d / "leaf_indices.npz"),
             **{str(i): arr for i, arr in enumerate(leaf_idx)})
    np.save(str(d / "col_to_bundle.npy"), col2bundle)
    np.save(str(d / "col_to_bin_offset.npy"), col2bin)


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic sparse binary benchmark data "
                    "matching real v3.3 training profiles."
    )
    parser.add_argument(
        "--profiles", nargs="+", default=list(PROFILES.keys()),
        choices=list(PROFILES.keys()),
        help="Which timeframe profiles to generate (default: all)."
    )
    parser.add_argument(
        "--out-dir", type=str, default="benchmark_data",
        help="Output directory (default: benchmark_data)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)."
    )
    parser.add_argument(
        "--num-class", type=int, default=3,
        help="Number of classes for gradient arrays (default: 3)."
    )
    parser.add_argument(
        "--num-leaves", type=int, default=63,
        help="Number of tree leaves for leaf masks (default: 63)."
    )
    parser.add_argument(
        "--features-per-bundle", type=int, default=255,
        help="Features per EFB bundle (default: 255)."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print sizes without generating data."
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    print("=" * 72)
    print("GPU Histogram Benchmark — Synthetic Data Generator")
    print("=" * 72)

    for name in args.profiles:
        prof = get_profile(name)
        rows, cols, density = prof["rows"], prof["cols"], prof["density"]
        nnz = int(rows * cols * density)
        n_bundles = (cols + args.features_per_bundle - 1) // args.features_per_bundle

        # Size estimates
        csr_bytes = (nnz * 4) + (nnz * 4) + ((rows + 1) * 8)  # data + indices + indptr
        grad_bytes = rows * args.num_class * 4 * 2              # grad + hess
        leaf_bytes = rows * 4                                    # assignment
        efb_bytes = cols * 4 * 2                                 # bundle + offset

        total = csr_bytes + grad_bytes + leaf_bytes + efb_bytes

        print(f"\n--- {name.upper()} ---")
        print(f"  Rows:       {rows:>12,}")
        print(f"  Cols:       {cols:>12,}")
        print(f"  Density:    {density:>12.4%}")
        print(f"  NNZ:        {nnz:>12,}")
        print(f"  Bundles:    {n_bundles:>12,}")
        print(f"  CSR size:   {_human_bytes(csr_bytes):>12}")
        print(f"  Grad+Hess:  {_human_bytes(grad_bytes):>12}")
        print(f"  Leaf masks: {_human_bytes(leaf_bytes):>12}")
        print(f"  EFB maps:   {_human_bytes(efb_bytes):>12}")
        print(f"  TOTAL:      {_human_bytes(total):>12}")

        if args.dry_run:
            continue

        t0 = time.time()

        print(f"  Generating sparse matrix ...", end=" ", flush=True)
        mat = generate_sparse_binary(rows, cols, density, seed=args.seed)
        print(f"done ({time.time() - t0:.1f}s, nnz={mat.nnz:,})")

        print(f"  Generating gradients ...", end=" ", flush=True)
        grad, hess = generate_gradients(rows, args.num_class, seed=args.seed)
        print("done")

        print(f"  Generating leaf masks ...", end=" ", flush=True)
        leaf_assign, leaf_idx = generate_leaf_masks(
            rows, args.num_leaves, seed=args.seed
        )
        print(f"done (leaves with rows: {sum(len(x) > 0 for x in leaf_idx)}/{args.num_leaves})")

        print(f"  Generating EFB mapping ...", end=" ", flush=True)
        col2bundle, col2bin = generate_efb_mapping(
            cols, args.features_per_bundle
        )
        print(f"done ({n_bundles:,} bundles)")

        print(f"  Saving to {out_dir / name}/ ...", end=" ", flush=True)
        _save_profile(out_dir, name, mat, grad, hess,
                      leaf_assign, leaf_idx, col2bundle, col2bin)
        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s total)")

    if args.dry_run:
        print("\n[dry-run] No data written.")
    else:
        print(f"\nAll profiles saved to: {out_dir.resolve()}")

    print("=" * 72)


if __name__ == "__main__":
    main()
