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
import hashlib
import numpy as np
import scipy.sparse as sp

try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

_DUAL_CSR = os.environ.get('CUDA_DUAL_CSR', '0') in ('1', 'y', 'Y', 'yes')

# Cache for the full GPU CSR and its transpose (uploaded once, reused every call)
_gpu_csr_cache = {}   # keyed on matrix identity hash -> gpu_csr
_gpu_csr_t_cache = {} # keyed on matrix identity hash -> gpu_csr.T


def _matrix_identity_hash(csr: sp.csr_matrix) -> str:
    """Compute a stable identity hash for a CSR matrix based on its data pointers.
    Uses id() of the underlying arrays — fast O(1), changes only if matrix is rebuilt."""
    return f"{id(csr.indptr)}_{id(csr.indices)}_{id(csr.data)}_{csr.shape}"


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

    # Upload full CSR to GPU once (stays resident). Cache keyed on matrix identity.
    mat_key = _matrix_identity_hash(csr)
    if mat_key not in _gpu_csr_cache:
        _gpu_csr_cache.clear()  # evict old matrix if any
        _gpu_csr_t_cache.clear()
        _gpu_csr_cache[mat_key] = cp_sparse.csr_matrix(csr.astype(np.float64))
    gpu_csr_full = _gpu_csr_cache[mat_key]

    # Build full-size masked gradient vector on GPU (non-leaf rows = 0)
    # This avoids expensive CPU-side csr[row_indices] slicing per call
    gpu_g_full = cp.zeros(csr.shape[0], dtype=cp.float64)
    gpu_h_full = cp.zeros(csr.shape[0], dtype=cp.float64)
    gpu_row_idx = cp.asarray(row_indices.astype(np.int64))
    gpu_g_full[gpu_row_idx] = cp.asarray(g_vec[row_indices])
    gpu_h_full[gpu_row_idx] = cp.asarray(h_vec[row_indices])

    # Cache CSR.T on GPU (keyed on matrix identity, not shape)
    if mat_key not in _gpu_csr_t_cache:
        _gpu_csr_t_cache[mat_key] = gpu_csr_full.T.tocsr()
    gpu_csr_t = _gpu_csr_t_cache[mat_key]

    # SpMV: CSR.T @ masked_grad = per-feature gradient sum (bin 1)
    hist_grad_bin1 = gpu_csr_t.dot(gpu_g_full)
    hist_hess_bin1 = gpu_csr_t.dot(gpu_h_full)

    # Total gradient/hessian for this leaf (sum of masked vectors = leaf sum)
    total_g = float(cp.sum(gpu_g_full))
    total_h = float(cp.sum(gpu_h_full))

    # Transfer results back to CPU
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
