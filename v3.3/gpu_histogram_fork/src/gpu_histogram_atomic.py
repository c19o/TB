"""
GPU histogram builder using custom atomic kernel (row-parallel).

Phase 3 optimized approach: each CUDA thread processes one leaf row's CSR
nonzeros and uses atomicAdd to accumulate into per-feature histogram bins.

For binary features (max_bin=2): only bin 1 is accumulated via atomics.
Bin 0 = total - bin 1 (computed after kernel).

Atomic contention is near-zero because data is ultra-sparse (0.3% density):
probability of two threads hitting the same feature column is ~0.003^2.

Requires: CuPy with CUDA support.
"""

import numpy as np
import scipy.sparse as sp

try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


# Raw CUDA kernel for row-parallel histogram accumulation
_HISTOGRAM_KERNEL_SRC = r"""
extern "C" __global__
void histogram_build_atomic(
    const long long* indptr,       // int64 [N+1]
    const int*       indices,      // int32 [nnz]
    const double*    data,         // float64 [nnz]
    const double*    grad,         // float64 [n_leaf_rows]
    const double*    hess,         // float64 [n_leaf_rows]
    const int*       leaf_rows,    // int32 [n_leaf_rows]
    int              n_leaf_rows,
    double*          hist_grad,    // float64 [n_features]  (bin 1 only)
    double*          hist_hess     // float64 [n_features]  (bin 1 only)
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_leaf_rows) return;

    int row = leaf_rows[tid];
    double g = grad[tid];
    double h = hess[tid];

    long long start = indptr[row];
    long long end   = indptr[row + 1];

    for (long long j = start; j < end; j++) {
        if (data[j] > 0.0) {
            int col = indices[j];
            atomicAdd(&hist_grad[col], g);
            atomicAdd(&hist_hess[col], h);
        }
    }
}
"""


def is_available() -> bool:
    """Check if CUDA is available for atomic histogram kernel."""
    if not CUDA_AVAILABLE:
        return False
    try:
        cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


def gpu_build_histogram_atomic(
    csr: sp.csr_matrix,
    grad: np.ndarray,
    hess: np.ndarray,
    row_indices: np.ndarray,
    class_idx: int = 0,
    n_bins: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-feature histogram on GPU via custom atomic kernel.

    Row-parallel: each thread processes one leaf row's CSR nonzeros.
    Uses fp64 atomicAdd for precision matching CPU reference.

    Parameters and returns identical to cpu_build_histogram.
    """
    n_features = csr.shape[1]

    if grad.ndim == 2:
        g_vec = grad[:, class_idx].astype(np.float64)
        h_vec = hess[:, class_idx].astype(np.float64)
    else:
        g_vec = grad.astype(np.float64)
        h_vec = hess.astype(np.float64)

    # Leaf-specific gradients
    leaf_g = g_vec[row_indices]
    leaf_h = h_vec[row_indices]
    n_leaf_rows = len(row_indices)

    # Transfer CSR structure to GPU (full matrix, kernel indexes by leaf_rows)
    gpu_indptr = cp.asarray(csr.indptr.astype(np.int64))
    gpu_indices = cp.asarray(csr.indices.astype(np.int32))
    gpu_data = cp.asarray(csr.data.astype(np.float64))
    gpu_grad = cp.asarray(leaf_g)
    gpu_hess = cp.asarray(leaf_h)
    gpu_leaf_rows = cp.asarray(row_indices.astype(np.int32))

    # Output buffers (bin 1 only — bin 0 computed by subtraction)
    gpu_hist_grad = cp.zeros(n_features, dtype=np.float64)
    gpu_hist_hess = cp.zeros(n_features, dtype=np.float64)

    # Compile and launch kernel
    kernel = cp.RawKernel(_HISTOGRAM_KERNEL_SRC, "histogram_build_atomic")
    block_size = 256
    grid_size = (n_leaf_rows + block_size - 1) // block_size

    kernel(
        (grid_size,), (block_size,),
        (
            gpu_indptr, gpu_indices, gpu_data,
            gpu_grad, gpu_hess, gpu_leaf_rows,
            np.int32(n_leaf_rows),
            gpu_hist_grad, gpu_hist_hess,
        ),
    )
    cp.cuda.Device().synchronize()

    # Transfer back
    hg1 = cp.asnumpy(gpu_hist_grad)
    hh1 = cp.asnumpy(gpu_hist_hess)

    # Compute totals and assemble 2-bin histogram
    total_g = np.sum(leaf_g)
    total_h = np.sum(leaf_h)

    hist_grad = np.zeros((n_features, n_bins), dtype=np.float64)
    hist_hess = np.zeros((n_features, n_bins), dtype=np.float64)
    hist_grad[:, 1] = hg1
    hist_hess[:, 1] = hh1
    hist_grad[:, 0] = total_g - hg1
    hist_hess[:, 0] = total_h - hh1

    return hist_grad, hist_hess
