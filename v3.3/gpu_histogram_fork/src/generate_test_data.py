"""
Synthetic sparse binary CSR data generator for histogram equivalence tests.

Generates data matching real training profiles:
- Binary cross features (0/1) in CSR format
- Gradient/hessian vectors (float32)
- Leaf row-index arrays
- 3-class multiclass gradients

Profiles match v3.3 training data characteristics:
  1w:  818 rows x 2.2M features, 0.3% density
  1d:  5733 rows x 6M features,  0.3% density
  4h:  17520 rows x 3M features, 0.3% density
  1h:  100K rows x 10M features, 0.3% density
  15m: 227K rows x 10M features, 0.3% density
"""

import numpy as np
import scipy.sparse as sp


def generate_sparse_binary_csr(
    n_rows: int,
    n_features: int,
    density: float = 0.003,
    seed: int = 42,
) -> sp.csr_matrix:
    """Generate a sparse binary CSR matrix.

    Parameters
    ----------
    n_rows : int
        Number of rows (data points / candles).
    n_features : int
        Number of columns (cross features).
    density : float
        Fraction of nonzero entries. Default 0.3% matches real crosses.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    sp.csr_matrix
        Binary (0/1) CSR matrix with float64 dtype for histogram precision.
    """
    rng = np.random.default_rng(seed)
    nnz = int(n_rows * n_features * density)
    # Generate random nonzero positions
    row_idx = rng.integers(0, n_rows, size=nnz)
    col_idx = rng.integers(0, n_features, size=nnz)
    data = np.ones(nnz, dtype=np.float64)
    mat = sp.csr_matrix((data, (row_idx, col_idx)), shape=(n_rows, n_features))
    # Eliminate duplicates (sum -> clip to 1)
    mat.data = np.clip(mat.data, 0, 1)
    mat.eliminate_zeros()
    return mat


def generate_gradients(
    n_rows: int,
    num_class: int = 3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate random gradient and hessian vectors.

    Parameters
    ----------
    n_rows : int
        Number of rows.
    num_class : int
        Number of classes (3 for long/short/hold).
    seed : int
        Random seed.

    Returns
    -------
    grad : np.ndarray, shape (n_rows, num_class), float32
    hess : np.ndarray, shape (n_rows, num_class), float32
    """
    rng = np.random.default_rng(seed)
    grad = rng.standard_normal((n_rows, num_class)).astype(np.float32)
    hess = np.abs(rng.standard_normal((n_rows, num_class))).astype(np.float32) + 0.01
    return grad, hess


def generate_leaf_indices(
    n_rows: int,
    n_leaves: int = 63,
    seed: int = 42,
) -> list[np.ndarray]:
    """Assign rows to leaves (disjoint partition).

    Parameters
    ----------
    n_rows : int
        Total rows.
    n_leaves : int
        Number of leaves.
    seed : int
        Random seed.

    Returns
    -------
    list of np.ndarray
        Each element is a sorted int32 array of row indices for that leaf.
    """
    rng = np.random.default_rng(seed)
    assignments = rng.integers(0, n_leaves, size=n_rows)
    leaves = []
    for leaf_id in range(n_leaves):
        rows = np.where(assignments == leaf_id)[0].astype(np.int32)
        leaves.append(rows)
    return leaves


def generate_parent_child_split(
    n_rows: int,
    split_ratio: float = 0.4,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate parent leaf row indices and split into left/right children.

    Parameters
    ----------
    n_rows : int
        Total rows in dataset.
    split_ratio : float
        Fraction of parent rows going to left child.
    seed : int
        Random seed.

    Returns
    -------
    parent_rows : np.ndarray (int32)
    left_rows : np.ndarray (int32)
    right_rows : np.ndarray (int32)
    """
    rng = np.random.default_rng(seed)
    # Parent has ~50% of total rows
    parent_size = n_rows // 2
    parent_rows = np.sort(rng.choice(n_rows, size=parent_size, replace=False)).astype(np.int32)
    # Split parent into left and right
    n_left = int(len(parent_rows) * split_ratio)
    perm = rng.permutation(len(parent_rows))
    left_rows = np.sort(parent_rows[perm[:n_left]])
    right_rows = np.sort(parent_rows[perm[n_left:]])
    return parent_rows, left_rows, right_rows
