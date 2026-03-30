"""
GPU histogram builder using cuSPARSE SpMV.

Phase 1 approach: use cuSPARSE to multiply CSR feature matrix by gradient
vector. This gives per-feature gradient sums — exactly what histogram bin 1
accumulates. Bin 0 is computed by subtraction.

Optimization D (CUDA_DUAL_CSR=1): Stores both CSR and CSR.T on GPU at init
time, avoiding repeated transpose computation during histogram builds.

Requires: CuPy with CUDA support.
"""

import os
import numpy as np
import scipy.sparse as sp

try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

_DUAL_CSR = os.environ.get('CUDA_DUAL_CSR', '0') in ('1', 'y', 'Y', 'yes')


def is_available() -> bool:
    """Check if CUDA is available for cuSPARSE histogram building."""
    if not CUDA_AVAILABLE:
        return False
    try:
        cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


def gpu_build_histogram_cusparse(
    csr: sp.csr_matrix,
    grad: np.ndarray,
    hess: np.ndarray,
    row_indices: np.ndarray,
    class_idx: int = 0,
    n_bins: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-feature histogram on GPU via cuSPARSE SpMV.

    Uses CSR.T @ grad_vector to compute per-feature gradient sums for bin 1.
    Bin 0 = total - bin 1.

    Parameters
    ----------
    csr : sp.csr_matrix
        Sparse binary feature matrix, shape (N, n_features).
    grad : np.ndarray
        Gradient array, shape (N,) or (N, num_class). float32.
    hess : np.ndarray
        Hessian array, same shape as grad.
    row_indices : np.ndarray
        Rows belonging to this leaf. int32.
    class_idx : int
        Class index for multiclass gradients.
    n_bins : int
        Number of bins per feature (default 2 for binary).

    Returns
    -------
    hist_grad : np.ndarray, shape (n_features, n_bins), float64
    hist_hess : np.ndarray, shape (n_features, n_bins), float64
    """
    n_features = csr.shape[1]

    if grad.ndim == 2:
        g_vec = grad[:, class_idx].astype(np.float64)
        h_vec = hess[:, class_idx].astype(np.float64)
    else:
        g_vec = grad.astype(np.float64)
        h_vec = hess.astype(np.float64)

    # Slice to leaf rows on CPU, then transfer
    leaf_csr = csr[row_indices].astype(np.float64)
    leaf_g = g_vec[row_indices]
    leaf_h = h_vec[row_indices]

    # Transfer to GPU
    gpu_csr = cp_sparse.csr_matrix(leaf_csr)
    gpu_g = cp.asarray(leaf_g)
    gpu_h = cp.asarray(leaf_h)

    # SpMV: CSR.T @ grad = per-feature gradient sum (bin 1)
    # Shape: (n_features,)
    # Optimization D (CUDA_DUAL_CSR=1): Cache CSR.T to avoid repeated transpose
    if _DUAL_CSR:
        if not hasattr(gpu_build_histogram_cusparse, '_cached_csr_t') or \
           gpu_build_histogram_cusparse._cached_shape != gpu_csr.shape:
            gpu_build_histogram_cusparse._cached_csr_t = gpu_csr.T.tocsr()
            gpu_build_histogram_cusparse._cached_shape = gpu_csr.shape
        gpu_csr_t = gpu_build_histogram_cusparse._cached_csr_t
    else:
        gpu_csr_t = gpu_csr.T.tocsr()
    hist_grad_bin1 = gpu_csr_t.dot(gpu_g)
    hist_hess_bin1 = gpu_csr_t.dot(gpu_h)

    # Total gradient/hessian for this leaf
    total_g = float(cp.sum(gpu_g))
    total_h = float(cp.sum(gpu_h))

    # Transfer back to CPU
    hg1 = cp.asnumpy(hist_grad_bin1)
    hh1 = cp.asnumpy(hist_hess_bin1)

    # Assemble 2-bin histogram
    hist_grad = np.zeros((n_features, n_bins), dtype=np.float64)
    hist_hess = np.zeros((n_features, n_bins), dtype=np.float64)
    hist_grad[:, 1] = hg1
    hist_hess[:, 1] = hh1
    hist_grad[:, 0] = total_g - hg1
    hist_hess[:, 0] = total_h - hh1

    return hist_grad, hist_hess
