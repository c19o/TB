"""
CPU reference implementation for histogram building.

Replicates LightGBM's histogram accumulation logic exactly:
  For each row in the leaf, iterate CSR nonzeros, accumulate
  gradient and hessian into per-feature bin counters.

Binary cross features (max_bin=2): bin 0 = feature OFF, bin 1 = feature ON.
Bin 0 is computed by subtraction (total - bin1) for efficiency, but this
reference computes it directly for validation.

All accumulation in float64 to match LightGBM's internal precision.
"""

import numpy as np
import scipy.sparse as sp


def cpu_build_histogram(
    csr: sp.csr_matrix,
    grad: np.ndarray,
    hess: np.ndarray,
    row_indices: np.ndarray,
    class_idx: int = 0,
    n_bins: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-feature histogram on CPU — reference implementation.

    Parameters
    ----------
    csr : sp.csr_matrix
        Sparse binary feature matrix, shape (N, n_features).
    grad : np.ndarray
        Gradient array, shape (N,) or (N, num_class). float32.
    hess : np.ndarray
        Hessian array, shape (N,) or (N, num_class). float32.
    row_indices : np.ndarray
        Rows belonging to this leaf. int32.
    class_idx : int
        Which class gradient to use (0-based). Ignored if grad is 1D.
    n_bins : int
        Number of bins per feature. Default 2 (binary).

    Returns
    -------
    hist_grad : np.ndarray, shape (n_features, n_bins), float64
        Gradient sums per feature per bin.
    hist_hess : np.ndarray, shape (n_features, n_bins), float64
        Hessian sums per feature per bin.
    """
    n_features = csr.shape[1]
    hist_grad = np.zeros((n_features, n_bins), dtype=np.float64)
    hist_hess = np.zeros((n_features, n_bins), dtype=np.float64)

    # Extract class-specific gradients if multiclass
    if grad.ndim == 2:
        g_vec = grad[:, class_idx]
        h_vec = hess[:, class_idx]
    else:
        g_vec = grad
        h_vec = hess

    indptr = csr.indptr
    indices = csr.indices
    data = csr.data

    # Accumulate bin=1 (feature ON) for each nonzero entry
    for row in row_indices:
        g = float(g_vec[row])
        h = float(h_vec[row])
        start = indptr[row]
        end = indptr[row + 1]
        for j in range(start, end):
            col = indices[j]
            val = data[j]
            if val > 0:
                # bin 1 = feature ON
                hist_grad[col, 1] += g
                hist_hess[col, 1] += h

    # bin 0 = total - bin 1 (rows where feature is OFF)
    total_g = np.sum(g_vec[row_indices]).astype(np.float64)
    total_h = np.sum(h_vec[row_indices]).astype(np.float64)
    hist_grad[:, 0] = total_g - hist_grad[:, 1]
    hist_hess[:, 0] = total_h - hist_hess[:, 1]

    return hist_grad, hist_hess


def cpu_build_histogram_vectorized(
    csr: sp.csr_matrix,
    grad: np.ndarray,
    hess: np.ndarray,
    row_indices: np.ndarray,
    class_idx: int = 0,
    n_bins: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized CPU histogram build — faster reference for larger data.

    Uses scipy sparse slicing + np.bincount for speed. Produces identical
    results to cpu_build_histogram (the row-loop version).

    Parameters and returns are identical to cpu_build_histogram.
    """
    n_features = csr.shape[1]

    if grad.ndim == 2:
        g_vec = grad[:, class_idx].astype(np.float64)
        h_vec = hess[:, class_idx].astype(np.float64)
    else:
        g_vec = grad.astype(np.float64)
        h_vec = hess.astype(np.float64)

    # Slice CSR to leaf rows
    leaf_csr = csr[row_indices]
    leaf_g = g_vec[row_indices]
    leaf_h = h_vec[row_indices]

    # For each nonzero entry: col index tells which feature, row tells which gradient
    coo = leaf_csr.tocoo()
    # bin 1 histogram: sum gradients for each feature where value > 0
    hist_grad_bin1 = np.bincount(
        coo.col, weights=leaf_g[coo.row] * (coo.data > 0).astype(np.float64),
        minlength=n_features
    )
    hist_hess_bin1 = np.bincount(
        coo.col, weights=leaf_h[coo.row] * (coo.data > 0).astype(np.float64),
        minlength=n_features
    )

    # bin 0 = total - bin 1
    total_g = np.sum(leaf_g)
    total_h = np.sum(leaf_h)

    hist_grad = np.zeros((n_features, n_bins), dtype=np.float64)
    hist_hess = np.zeros((n_features, n_bins), dtype=np.float64)
    hist_grad[:, 1] = hist_grad_bin1
    hist_hess[:, 1] = hist_hess_bin1
    hist_grad[:, 0] = total_g - hist_grad_bin1
    hist_hess[:, 0] = total_h - hist_hess_bin1

    return hist_grad, hist_hess
