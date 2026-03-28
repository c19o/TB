"""
Leaf Gradient Scatter — gradient mapping strategies for per-leaf GPU histograms.

The Problem:
    LightGBM computes ordered_gradients[i] = gradients[data_indices[i]]
    where data_indices = row indices belonging to the current leaf.

    For SpMV (CSR_AT @ gradient_vector), we need a FULL-SIZE gradient vector
    indexed by original row IDs, with non-leaf rows zeroed out.

    For the atomic scatter kernel, we directly iterate leaf rows using
    data_indices — no gradient remapping needed.

Three Approaches:
    A) Masked Gradient Vector — zero non-leaf rows, then SpMV (CSR.T @ masked)
    B) Sparse Gradient Vector — sparse vector with only leaf rows (NOT usable
       with cuSPARSE SpMV which requires a dense RHS vector)
    C) Atomic Scatter — CUDA kernel walks CSR per leaf row, atomicAdd to histogram

Crossover Analysis (1w: 818 rows x 2.2M features):
    - Small leaf (10% of rows, ~82 rows): masking wastes 90% of SpMV compute.
      Atomic scatter wins because it only touches 82 rows' CSR entries.
    - Large leaf (50% of rows, ~409 rows): masking overhead is small relative
      to SpMV work. SpMV wins due to cuSPARSE library-optimized kernels.
    - Root leaf (100%, 818 rows): masking = no-op, SpMV is ideal (one call,
      full matrix utilization, cuSPARSE tuned code paths).

Matrix Thesis Context:
    Binary cross features in sparse CSR. ALL features preserved (2.2M+).
    Structural zero = 0.0 = feature OFF. No filtering. No subsampling.
    Atomic contention is LOW because we have millions of features —
    different rows rarely hit the same column in the same warp.

Requires: CuPy with CUDA support.
"""

import time
import logging
from typing import Optional

import numpy as np
import scipy.sparse as sp_sparse

try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BLOCK_SIZE = 256  # threads per block — standard for modern GPUs

# Crossover threshold: if leaf has fewer than this fraction of total rows,
# atomic scatter is faster than masked SpMV. Empirically ~25% on RTX 3090
# for 818x2.2M matrices. Adjustable via set_crossover_fraction().
_CROSSOVER_FRACTION = 0.25


def set_crossover_fraction(frac: float):
    """Override the crossover threshold for strategy auto-selection."""
    global _CROSSOVER_FRACTION
    if not 0.0 < frac < 1.0:
        raise ValueError(f"Crossover fraction must be in (0, 1), got {frac}")
    _CROSSOVER_FRACTION = frac


# ---------------------------------------------------------------------------
# Approach A: Masked Gradient Vector (for SpMV path)
# ---------------------------------------------------------------------------

class MaskedGradientMapper:
    """
    Creates a full-size gradient vector with non-leaf rows zeroed out.

    For SpMV: histogram = CSR.T @ masked_gradients.

    Cost: O(n_total_rows) to zero + O(n_leaf_rows) to scatter.
    The SpMV itself touches all NNZ entries regardless of masking,
    so small leaves waste most of the SpMV compute on zero-multiplies.

    Best for: large leaves (>25% of rows) and root node (100%).
    """

    def __init__(self, n_total_rows: int, dtype: str = "float64"):
        if not HAS_CUPY:
            raise RuntimeError("CuPy required. Install: pip install cupy-cuda12x")

        self.n_total_rows = n_total_rows
        self.cp_dtype = cp.float32 if dtype == "float32" else cp.float64
        self.dtype_str = dtype

        # Pre-allocate reusable masked buffers on GPU
        self._masked_grad = cp.zeros(n_total_rows, dtype=self.cp_dtype)
        self._masked_hess = cp.zeros(n_total_rows, dtype=self.cp_dtype)

    def create_masked_gradients(
        self,
        d_gradients: "cp.ndarray",
        d_hessians: "cp.ndarray",
        d_leaf_row_indices: "cp.ndarray",
    ) -> tuple:
        """
        Create full-size gradient/hessian vectors with non-leaf rows zeroed.

        Parameters
        ----------
        d_gradients : cp.ndarray, shape (n_total_rows,)
            Full gradient vector on GPU.
        d_hessians : cp.ndarray, shape (n_total_rows,)
            Full hessian vector on GPU.
        d_leaf_row_indices : cp.ndarray, shape (n_leaf_rows,), dtype int32
            Row indices belonging to this leaf, on GPU.

        Returns
        -------
        masked_grad : cp.ndarray, shape (n_total_rows,)
        masked_hess : cp.ndarray, shape (n_total_rows,)
        """
        # Zero the buffers
        self._masked_grad.fill(0)
        self._masked_hess.fill(0)

        if len(d_leaf_row_indices) == 0:
            return self._masked_grad, self._masked_hess

        # Scatter leaf gradients into full-size vector
        # CuPy fancy indexing is GPU-accelerated
        idx = d_leaf_row_indices.astype(cp.int64)
        self._masked_grad[idx] = d_gradients[idx]
        self._masked_hess[idx] = d_hessians[idx]

        return self._masked_grad, self._masked_hess

    def build_histogram_spmv(
        self,
        gpu_csr_t: "cp_sparse.csr_matrix",
        d_gradients: "cp.ndarray",
        d_hessians: "cp.ndarray",
        d_leaf_row_indices: "cp.ndarray",
    ) -> tuple:
        """
        Build histogram via masked SpMV: CSR.T @ masked_gradients.

        Parameters
        ----------
        gpu_csr_t : cupyx.scipy.sparse.csr_matrix
            Transpose of the feature matrix, already on GPU.
            Shape: (n_features, n_total_rows).
        d_gradients, d_hessians : cp.ndarray, shape (n_total_rows,)
        d_leaf_row_indices : cp.ndarray, shape (n_leaf_rows,), int32

        Returns
        -------
        hist_grad : cp.ndarray, shape (n_features,)
        hist_hess : cp.ndarray, shape (n_features,)
        elapsed_ms : float
        """
        start_ev = cp.cuda.Event()
        end_ev = cp.cuda.Event()
        start_ev.record()

        masked_g, masked_h = self.create_masked_gradients(
            d_gradients, d_hessians, d_leaf_row_indices
        )

        # SpMV: CSR.T @ masked_vector = per-feature sums for leaf rows
        hist_grad = gpu_csr_t @ masked_g
        hist_hess = gpu_csr_t @ masked_h

        end_ev.record()
        end_ev.synchronize()
        elapsed_ms = cp.cuda.get_elapsed_time(start_ev, end_ev)

        return hist_grad, hist_hess, elapsed_ms

    def cleanup(self):
        """Free GPU buffers."""
        del self._masked_grad, self._masked_hess
        self._masked_grad = None
        self._masked_hess = None


# ---------------------------------------------------------------------------
# Approach B: Sparse Gradient Vector
# ---------------------------------------------------------------------------

class SparseGradientMapper:
    """
    Creates a sparse vector with only leaf-row entries.

    NOTE: cuSPARSE SpMV requires a DENSE right-hand-side vector.
    This approach CANNOT be used with the SpMV path directly.

    It CAN be used with:
      - Manual sparse-sparse multiply (very slow for vector RHS)
      - Converting back to dense (defeats the purpose)
      - Custom CUDA kernel that reads sparse indices

    Included for completeness and benchmarking. In practice, Approach A
    (masked dense) or Approach C (atomic scatter) are always faster.

    Best for: nothing in practice. Documented for reference.
    """

    def __init__(self, n_total_rows: int, dtype: str = "float64"):
        if not HAS_CUPY:
            raise RuntimeError("CuPy required. Install: pip install cupy-cuda12x")

        self.n_total_rows = n_total_rows
        self.cp_dtype = cp.float32 if dtype == "float32" else cp.float64
        self.dtype_str = dtype

    def create_sparse_gradients(
        self,
        d_gradients: "cp.ndarray",
        d_leaf_row_indices: "cp.ndarray",
    ) -> "cp_sparse.csr_matrix":
        """
        Create a sparse CSR vector with only leaf-row gradient values.

        Parameters
        ----------
        d_gradients : cp.ndarray, shape (n_total_rows,)
        d_leaf_row_indices : cp.ndarray, shape (n_leaf_rows,), int32

        Returns
        -------
        sparse_grad : cupyx.scipy.sparse.csr_matrix, shape (n_total_rows, 1)
            Sparse column vector with non-zeros only at leaf row positions.
        """
        n_leaf = len(d_leaf_row_indices)
        if n_leaf == 0:
            return cp_sparse.csr_matrix(
                (self.n_total_rows, 1), dtype=self.cp_dtype
            )

        idx = d_leaf_row_indices.astype(cp.int64)
        values = d_gradients[idx]
        col_indices = cp.zeros(n_leaf, dtype=cp.int32)

        sparse_grad = cp_sparse.csr_matrix(
            (values, (idx, col_indices)),
            shape=(self.n_total_rows, 1),
            dtype=self.cp_dtype,
        )
        return sparse_grad

    def build_histogram_sparse(
        self,
        gpu_csr_t: "cp_sparse.csr_matrix",
        d_gradients: "cp.ndarray",
        d_hessians: "cp.ndarray",
        d_leaf_row_indices: "cp.ndarray",
    ) -> tuple:
        """
        Build histogram via sparse-sparse multiply (slow — for benchmarking only).

        Converts sparse gradient vector to dense before SpMV because
        cuSPARSE SpMV requires dense RHS. This defeats the purpose of
        sparse representation but shows the overhead.

        Returns
        -------
        hist_grad, hist_hess : cp.ndarray, shape (n_features,)
        elapsed_ms : float
        """
        start_ev = cp.cuda.Event()
        end_ev = cp.cuda.Event()
        start_ev.record()

        sparse_g = self.create_sparse_gradients(d_gradients, d_leaf_row_indices)
        sparse_h = self.create_sparse_gradients(d_hessians, d_leaf_row_indices)

        # Must convert to dense for SpMV — cuSPARSE limitation
        dense_g = sparse_g.toarray().ravel()
        dense_h = sparse_h.toarray().ravel()

        hist_grad = gpu_csr_t @ dense_g
        hist_hess = gpu_csr_t @ dense_h

        end_ev.record()
        end_ev.synchronize()
        elapsed_ms = cp.cuda.get_elapsed_time(start_ev, end_ev)

        return hist_grad, hist_hess, elapsed_ms


# ---------------------------------------------------------------------------
# Approach C: Atomic Scatter Kernel (no SpMV)
# ---------------------------------------------------------------------------

_ATOMIC_SCATTER_F64 = r"""
extern "C" __global__
void leaf_scatter_histogram_f64(
    const long long*  __restrict__ indptr,
    const int*        __restrict__ indices,
    const double*     __restrict__ gradients,
    const double*     __restrict__ hessians,
    const int*        __restrict__ leaf_rows,
    double*           __restrict__ hist_grad,
    double*           __restrict__ hist_hess,
    int n_leaf_rows
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_leaf_rows) return;

    int row = leaf_rows[tid];
    double g = gradients[row];
    double h = hessians[row];

    long long start = indptr[row];
    long long end   = indptr[row + 1];

    for (long long j = start; j < end; j++) {
        int col = indices[j];
        atomicAdd(&hist_grad[col], g);
        atomicAdd(&hist_hess[col], h);
    }
}
"""

_ATOMIC_SCATTER_F32 = r"""
extern "C" __global__
void leaf_scatter_histogram_f32(
    const long long* __restrict__ indptr,
    const int*       __restrict__ indices,
    const float*     __restrict__ gradients,
    const float*     __restrict__ hessians,
    const int*       __restrict__ leaf_rows,
    float*           __restrict__ hist_grad,
    float*           __restrict__ hist_hess,
    int n_leaf_rows
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_leaf_rows) return;

    int row = leaf_rows[tid];
    float g = gradients[row];
    float h = hessians[row];

    long long start = indptr[row];
    long long end   = indptr[row + 1];

    for (long long j = start; j < end; j++) {
        int col = indices[j];
        atomicAdd(&hist_grad[col], g);
        atomicAdd(&hist_hess[col], h);
    }
}
"""

_compiled_kernels = {}


def _get_kernel(name: str, source: str) -> "cp.RawKernel":
    """Compile and cache a CuPy RawKernel."""
    if name not in _compiled_kernels:
        _compiled_kernels[name] = cp.RawKernel(source, name)
    return _compiled_kernels[name]


class AtomicScatterMapper:
    """
    Builds per-leaf histograms via CUDA atomic scatter — no SpMV needed.

    Each CUDA thread processes one leaf row: reads the row's gradient,
    walks the CSR entries for that row, and atomicAdd's into the
    histogram bins.

    Uses data_indices directly — no gradient remapping or masking.

    Atomic contention is LOW for our workload:
        - 2.2M features = 2.2M histogram bins
        - 818 rows (1w) = 818 threads
        - Probability of two threads hitting the same bin simultaneously
          is ~818/2.2M = 0.04% per NNZ entry. Effectively zero contention.

    Best for: small leaves (<25% of rows) where SpMV wastes compute
    on zero-multiplied rows.
    """

    def __init__(
        self,
        d_indptr: "cp.ndarray",
        d_indices: "cp.ndarray",
        n_features: int,
        dtype: str = "float64",
    ):
        """
        Parameters
        ----------
        d_indptr : cp.ndarray, int64
            CSR indptr array, already on GPU.
        d_indices : cp.ndarray, int32
            CSR column indices, already on GPU.
        n_features : int
            Number of features (histogram output size).
        dtype : str
            'float32' or 'float64'.
        """
        if not HAS_CUPY:
            raise RuntimeError("CuPy required. Install: pip install cupy-cuda12x")

        self.d_indptr = d_indptr
        self.d_indices = d_indices
        self.n_features = n_features
        self.dtype_str = dtype
        self.cp_dtype = cp.float32 if dtype == "float32" else cp.float64

        # Pre-allocate histogram output buffers
        self.d_hist_grad = cp.zeros(n_features, dtype=self.cp_dtype)
        self.d_hist_hess = cp.zeros(n_features, dtype=self.cp_dtype)

        # Select kernel
        if dtype == "float32":
            self._kernel = _get_kernel(
                "leaf_scatter_histogram_f32", _ATOMIC_SCATTER_F32
            )
        else:
            self._kernel = _get_kernel(
                "leaf_scatter_histogram_f64", _ATOMIC_SCATTER_F64
            )

    def build_histogram_atomic(
        self,
        d_gradients: "cp.ndarray",
        d_hessians: "cp.ndarray",
        d_leaf_row_indices: "cp.ndarray",
    ) -> tuple:
        """
        Build histogram via atomic scatter kernel.

        Parameters
        ----------
        d_gradients : cp.ndarray, shape (n_total_rows,)
            Full gradient vector on GPU (indexed by original row ID).
        d_hessians : cp.ndarray, shape (n_total_rows,)
        d_leaf_row_indices : cp.ndarray, shape (n_leaf_rows,), int32
            Row indices belonging to this leaf.

        Returns
        -------
        hist_grad : cp.ndarray, shape (n_features,)
        hist_hess : cp.ndarray, shape (n_features,)
        elapsed_ms : float
        """
        n_leaf_rows = len(d_leaf_row_indices)
        if n_leaf_rows == 0:
            self.d_hist_grad.fill(0)
            self.d_hist_hess.fill(0)
            return self.d_hist_grad, self.d_hist_hess, 0.0

        # Zero output buffers
        self.d_hist_grad.fill(0)
        self.d_hist_hess.fill(0)

        grid_size = (n_leaf_rows + BLOCK_SIZE - 1) // BLOCK_SIZE

        start_ev = cp.cuda.Event()
        end_ev = cp.cuda.Event()
        start_ev.record()

        self._kernel(
            (grid_size,),
            (BLOCK_SIZE,),
            (
                self.d_indptr,
                self.d_indices,
                d_gradients,
                d_hessians,
                d_leaf_row_indices,
                self.d_hist_grad,
                self.d_hist_hess,
                np.int32(n_leaf_rows),
            ),
        )

        end_ev.record()
        end_ev.synchronize()
        elapsed_ms = cp.cuda.get_elapsed_time(start_ev, end_ev)

        return self.d_hist_grad, self.d_hist_hess, elapsed_ms

    def cleanup(self):
        """Free GPU histogram buffers."""
        del self.d_hist_grad, self.d_hist_hess
        self.d_hist_grad = None
        self.d_hist_hess = None


# ---------------------------------------------------------------------------
# Adaptive Strategy Selector
# ---------------------------------------------------------------------------

class LeafGradientScatter:
    """
    Unified interface that auto-selects the best gradient mapping strategy
    based on leaf size relative to total rows.

    Strategy selection:
        - leaf_fraction >= crossover (default 0.25): Masked SpMV (Approach A)
        - leaf_fraction < crossover: Atomic Scatter (Approach C)
        - leaf_fraction == 1.0 (root): Masked SpMV with no masking overhead

    Approach B (sparse vector) is never auto-selected — it always loses to
    A or C in benchmarks due to the cuSPARSE dense-RHS requirement.

    Parameters
    ----------
    csr_matrix : scipy.sparse.csr_matrix
        Feature matrix (n_rows, n_features). Binary cross features.
        Must have int64 indptr for 15m (NNZ > 2^31).
    dtype : str
        'float32' or 'float64'. LightGBM uses float64 internally.
    device_id : int
        CUDA device.
    """

    def __init__(
        self,
        csr_matrix: sp_sparse.csr_matrix,
        dtype: str = "float64",
        device_id: int = 0,
    ):
        if not HAS_CUPY:
            raise RuntimeError("CuPy required. Install: pip install cupy-cuda12x")

        self.n_rows = csr_matrix.shape[0]
        self.n_features = csr_matrix.shape[1]
        self.dtype_str = dtype
        self.cp_dtype = cp.float32 if dtype == "float32" else cp.float64
        self.device_id = device_id

        # Ensure CSR with int64 indptr
        csr = csr_matrix.tocsr()
        indptr_np = np.asarray(csr.indptr, dtype=np.int64)
        indices_np = np.asarray(csr.indices, dtype=np.int32)

        with cp.cuda.Device(device_id):
            # Upload CSR to GPU
            t0 = time.perf_counter()
            self._d_indptr = cp.asarray(indptr_np)
            self._d_indices = cp.asarray(indices_np)
            cp.cuda.Device().synchronize()
            h2d_ms = (time.perf_counter() - t0) * 1000.0

            # Upload CSR with data for SpMV path
            cpu_csr = csr.astype(np.float64 if dtype == "float64" else np.float32)
            self._gpu_csr = cp_sparse.csr_matrix(
                (
                    cp.array(cpu_csr.data),
                    cp.array(cpu_csr.indices),
                    cp.array(cpu_csr.indptr.astype(np.int64)),
                ),
                shape=cpu_csr.shape,
                dtype=self.cp_dtype,
            )
            self._gpu_csr_t = self._gpu_csr.T.tocsr()

            # Initialize both strategy backends
            self._masked_mapper = MaskedGradientMapper(self.n_rows, dtype)
            self._atomic_mapper = AtomicScatterMapper(
                self._d_indptr, self._d_indices, self.n_features, dtype
            )
            self._sparse_mapper = SparseGradientMapper(self.n_rows, dtype)

        self.nnz = int(indptr_np[-1])

        logger.info(
            "LeafGradientScatter initialized: %d rows x %d features, "
            "NNZ=%s, dtype=%s, H2D=%.1fms, crossover=%.0f%%",
            self.n_rows, self.n_features, f"{self.nnz:,}",
            dtype, h2d_ms, _CROSSOVER_FRACTION * 100,
        )

    def build_histogram(
        self,
        d_gradients: "cp.ndarray",
        d_hessians: "cp.ndarray",
        d_leaf_row_indices: "cp.ndarray",
        strategy: Optional[str] = None,
    ) -> tuple:
        """
        Build per-leaf histogram using the optimal strategy.

        Parameters
        ----------
        d_gradients : cp.ndarray, shape (n_rows,)
            Full gradient vector on GPU.
        d_hessians : cp.ndarray, shape (n_rows,)
        d_leaf_row_indices : cp.ndarray, shape (n_leaf_rows,), int32
        strategy : str or None
            Force a specific strategy: 'masked', 'atomic', 'sparse'.
            If None, auto-selects based on leaf fraction.

        Returns
        -------
        hist_grad : cp.ndarray, shape (n_features,)
        hist_hess : cp.ndarray, shape (n_features,)
        elapsed_ms : float
        strategy_used : str
            Which strategy was actually used.
        """
        n_leaf = len(d_leaf_row_indices)
        leaf_fraction = n_leaf / self.n_rows if self.n_rows > 0 else 1.0

        # Auto-select strategy
        if strategy is None:
            if leaf_fraction >= _CROSSOVER_FRACTION:
                strategy = "masked"
            else:
                strategy = "atomic"

        if strategy == "masked":
            hg, hh, ms = self._masked_mapper.build_histogram_spmv(
                self._gpu_csr_t, d_gradients, d_hessians, d_leaf_row_indices
            )
        elif strategy == "atomic":
            hg, hh, ms = self._atomic_mapper.build_histogram_atomic(
                d_gradients, d_hessians, d_leaf_row_indices
            )
        elif strategy == "sparse":
            hg, hh, ms = self._sparse_mapper.build_histogram_sparse(
                self._gpu_csr_t, d_gradients, d_hessians, d_leaf_row_indices
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'masked', 'atomic', or 'sparse'.")

        return hg, hh, ms, strategy

    def cleanup(self):
        """Free all GPU memory."""
        self._masked_mapper.cleanup()
        self._atomic_mapper.cleanup()
        del self._gpu_csr, self._gpu_csr_t
        del self._d_indptr, self._d_indices
        self._gpu_csr = None
        self._gpu_csr_t = None

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        logger.info("LeafGradientScatter: GPU memory freed.")

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CPU Reference (for validation)
# ---------------------------------------------------------------------------

def cpu_histogram_reference(
    csr_matrix: sp_sparse.csr_matrix,
    gradients: np.ndarray,
    hessians: np.ndarray,
    leaf_row_indices: np.ndarray,
) -> tuple:
    """
    CPU reference: iterate leaf rows, accumulate grad/hess per feature.

    Intentionally simple — correctness reference only.
    """
    n_features = csr_matrix.shape[1]
    hist_grad = np.zeros(n_features, dtype=np.float64)
    hist_hess = np.zeros(n_features, dtype=np.float64)

    csr = csr_matrix.tocsr()
    indptr = csr.indptr
    indices = csr.indices

    for row in leaf_row_indices:
        g = gradients[row]
        h = hessians[row]
        start = indptr[row]
        end = indptr[row + 1]
        cols = indices[start:end]
        hist_grad[cols] += g
        hist_hess[cols] += h

    return hist_grad, hist_hess


# ---------------------------------------------------------------------------
# Benchmark Suite
# ---------------------------------------------------------------------------

def run_benchmark(
    n_rows: int = 818,
    n_features: int = 2_200_000,
    sparsity: float = 0.997,
    leaf_fractions: Optional[list] = None,
    n_warmup: int = 3,
    n_trials: int = 10,
    dtype: str = "float64",
):
    """
    Benchmark all three approaches across different leaf sizes.

    Parameters
    ----------
    n_rows : int
        Number of rows (default 818 for 1w timeframe).
    n_features : int
        Number of features (default 2.2M for 1w).
    sparsity : float
        Fraction of zeros (default 0.997 matching real crosses).
    leaf_fractions : list of float
        Leaf sizes to test as fraction of n_rows.
        Default: [0.05, 0.10, 0.25, 0.50, 0.75, 1.00]
    n_warmup : int
        Warmup iterations before timing.
    n_trials : int
        Timed iterations per strategy per leaf size.
    dtype : str
        'float32' or 'float64'.

    Returns
    -------
    results : list of dict
        One entry per (leaf_fraction, strategy) with timing info.
    """
    if not HAS_CUPY:
        raise RuntimeError("CuPy required for benchmarks.")

    if leaf_fractions is None:
        leaf_fractions = [0.05, 0.10, 0.25, 0.50, 0.75, 1.00]

    print("=" * 78)
    print("Leaf Gradient Scatter — Benchmark Suite")
    print("=" * 78)
    print(f"Matrix: {n_rows:,} rows x {n_features:,} features")
    print(f"Sparsity: {sparsity * 100:.1f}%, dtype: {dtype}")
    print(f"GPU: {cp.cuda.Device().name}")
    free, total = cp.cuda.Device().mem_info
    print(f"VRAM: {free / 1e9:.1f} GB free / {total / 1e9:.1f} GB total")
    print(f"Warmup: {n_warmup}, Trials: {n_trials}")

    # --- Generate synthetic data matching real 1w profile ---
    print("\nGenerating synthetic sparse CSR...")
    rng = np.random.default_rng(42)
    nnz_expected = int(n_rows * n_features * (1 - sparsity))
    row_idx = rng.integers(0, n_rows, size=nnz_expected)
    col_idx = rng.integers(0, n_features, size=nnz_expected)
    data = np.ones(nnz_expected, dtype=np.float32)
    csr = sp_sparse.csr_matrix(
        (data, (row_idx, col_idx)), shape=(n_rows, n_features)
    )
    csr.sum_duplicates()
    csr.indptr = csr.indptr.astype(np.int64)
    print(f"Actual NNZ: {csr.nnz:,}")

    # Gradients / hessians
    np_dtype = np.float32 if dtype == "float32" else np.float64
    gradients = rng.standard_normal(n_rows).astype(np_dtype)
    hessians = (np.abs(rng.standard_normal(n_rows)) + 0.1).astype(np_dtype)

    # --- Initialize the unified scatter ---
    print("\nInitializing LeafGradientScatter...")
    scatter = LeafGradientScatter(csr, dtype=dtype)

    # Upload gradients to GPU once
    d_grad = cp.asarray(gradients)
    d_hess = cp.asarray(hessians)

    strategies = ["masked", "atomic", "sparse"]
    results = []

    print(f"\n{'Leaf%':>6} {'Rows':>6} {'Strategy':>10} "
          f"{'Mean ms':>10} {'Std ms':>8} {'Min ms':>8} {'Max ms':>8} {'Status':>8}")
    print("-" * 78)

    for frac in leaf_fractions:
        n_leaf = max(1, int(n_rows * frac))
        leaf_rows = rng.choice(n_rows, size=n_leaf, replace=False).astype(np.int32)
        leaf_rows.sort()
        d_leaf = cp.asarray(leaf_rows)

        # CPU reference for validation
        hg_cpu, hh_cpu = cpu_histogram_reference(csr, gradients, hessians, leaf_rows)

        for strat in strategies:
            timings = []

            # Warmup
            for _ in range(n_warmup):
                scatter.build_histogram(d_grad, d_hess, d_leaf, strategy=strat)

            # Timed trials
            for _ in range(n_trials):
                hg, hh, ms, _ = scatter.build_histogram(
                    d_grad, d_hess, d_leaf, strategy=strat
                )
                timings.append(ms)

            # Validate against CPU reference
            hg_np = cp.asnumpy(hg)
            hh_np = cp.asnumpy(hh)
            grad_diff = np.max(np.abs(hg_np - hg_cpu))
            hess_diff = np.max(np.abs(hh_np - hh_cpu))
            tol = 1e-6 if dtype == "float64" else 1e-1
            valid = grad_diff < tol and hess_diff < tol
            status = "PASS" if valid else f"FAIL(g={grad_diff:.1e})"

            mean_ms = np.mean(timings)
            std_ms = np.std(timings)
            min_ms = np.min(timings)
            max_ms = np.max(timings)

            print(f"{frac*100:5.0f}% {n_leaf:6d} {strat:>10} "
                  f"{mean_ms:10.3f} {std_ms:8.3f} {min_ms:8.3f} {max_ms:8.3f} "
                  f"{status:>8}")

            results.append({
                "leaf_fraction": frac,
                "n_leaf_rows": n_leaf,
                "strategy": strat,
                "mean_ms": float(mean_ms),
                "std_ms": float(std_ms),
                "min_ms": float(min_ms),
                "max_ms": float(max_ms),
                "valid": valid,
                "grad_max_diff": float(grad_diff),
            })

        print()  # blank line between leaf sizes

    # --- Recommendations ---
    print("=" * 78)
    print("RECOMMENDATIONS")
    print("=" * 78)

    for frac in leaf_fractions:
        frac_results = [r for r in results if r["leaf_fraction"] == frac and r["valid"]]
        if not frac_results:
            print(f"  {frac*100:.0f}%: NO VALID RESULTS")
            continue
        best = min(frac_results, key=lambda r: r["mean_ms"])
        print(
            f"  {frac*100:5.0f}% ({best['n_leaf_rows']:,} rows): "
            f"{best['strategy']:>8} @ {best['mean_ms']:.3f} ms"
        )

    # Find crossover point
    for i in range(len(leaf_fractions) - 1):
        f_lo = leaf_fractions[i]
        f_hi = leaf_fractions[i + 1]
        best_lo = min(
            [r for r in results if r["leaf_fraction"] == f_lo and r["valid"]],
            key=lambda r: r["mean_ms"], default=None
        )
        best_hi = min(
            [r for r in results if r["leaf_fraction"] == f_hi and r["valid"]],
            key=lambda r: r["mean_ms"], default=None
        )
        if best_lo and best_hi:
            if best_lo["strategy"] != best_hi["strategy"]:
                print(
                    f"\n  CROSSOVER between {f_lo*100:.0f}% and {f_hi*100:.0f}%: "
                    f"{best_lo['strategy']} -> {best_hi['strategy']}"
                )
                print(
                    f"  Suggested _CROSSOVER_FRACTION = {(f_lo + f_hi) / 2:.2f}"
                )

    print(f"\n  Current auto-select threshold: {_CROSSOVER_FRACTION:.0%}")
    print("=" * 78)

    scatter.cleanup()
    return results


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

def _run_test():
    """Correctness + basic performance test."""
    print("=" * 70)
    print("LeafGradientScatter — Correctness Test")
    print("=" * 70)

    if not HAS_CUPY:
        print("SKIP: CuPy not available")
        return

    print(f"GPU: {cp.cuda.Device().name}")
    free, total = cp.cuda.Device().mem_info
    print(f"VRAM: {free / 1e9:.1f} GB free / {total / 1e9:.1f} GB total")

    # Small test matrix
    n_rows = 818
    n_features = 100_000
    sparsity = 0.997

    rng = np.random.default_rng(42)
    nnz = int(n_rows * n_features * (1 - sparsity))
    row_idx = rng.integers(0, n_rows, size=nnz)
    col_idx = rng.integers(0, n_features, size=nnz)
    data = np.ones(nnz, dtype=np.float32)
    csr = sp_sparse.csr_matrix(
        (data, (row_idx, col_idx)), shape=(n_rows, n_features)
    )
    csr.sum_duplicates()
    csr.indptr = csr.indptr.astype(np.int64)
    print(f"Matrix: {n_rows} x {n_features:,}, NNZ={csr.nnz:,}")

    gradients = rng.standard_normal(n_rows).astype(np.float64)
    hessians = (np.abs(rng.standard_normal(n_rows)) + 0.1).astype(np.float64)

    scatter = LeafGradientScatter(csr, dtype="float64")
    d_grad = cp.asarray(gradients)
    d_hess = cp.asarray(hessians)

    test_cases = [
        ("root (100%)", np.arange(n_rows, dtype=np.int32)),
        ("large (50%)", rng.choice(n_rows, size=n_rows // 2, replace=False).astype(np.int32)),
        ("medium (25%)", rng.choice(n_rows, size=n_rows // 4, replace=False).astype(np.int32)),
        ("small (10%)", rng.choice(n_rows, size=n_rows // 10, replace=False).astype(np.int32)),
        ("tiny (1%)", rng.choice(n_rows, size=max(1, n_rows // 100), replace=False).astype(np.int32)),
        ("empty (0)", np.array([], dtype=np.int32)),
    ]

    all_passed = True
    for name, leaf_rows in test_cases:
        leaf_rows.sort()
        d_leaf = cp.asarray(leaf_rows)

        # CPU reference
        if len(leaf_rows) > 0:
            hg_cpu, hh_cpu = cpu_histogram_reference(csr, gradients, hessians, leaf_rows)
        else:
            hg_cpu = np.zeros(n_features, dtype=np.float64)
            hh_cpu = np.zeros(n_features, dtype=np.float64)

        print(f"\n--- {name}: {len(leaf_rows)} rows ---")

        for strat in ["masked", "atomic", "sparse"]:
            hg, hh, ms, used = scatter.build_histogram(
                d_grad, d_hess, d_leaf, strategy=strat
            )
            hg_np = cp.asnumpy(hg)
            hh_np = cp.asnumpy(hh)

            grad_diff = np.max(np.abs(hg_np - hg_cpu))
            hess_diff = np.max(np.abs(hh_np - hh_cpu))
            ok = grad_diff < 1e-6 and hess_diff < 1e-6
            status = "PASS" if ok else "FAIL"
            if not ok:
                all_passed = False

            print(f"  {strat:>8}: {ms:7.3f} ms | "
                  f"grad_diff={grad_diff:.2e} hess_diff={hess_diff:.2e} | {status}")

        # Auto-select test
        hg_auto, hh_auto, ms_auto, strat_auto = scatter.build_histogram(
            d_grad, d_hess, d_leaf
        )
        print(f"  {'auto':>8}: {ms_auto:7.3f} ms | strategy={strat_auto}")

    scatter.cleanup()

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 70)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if "--benchmark" in sys.argv:
        # Full benchmark: python leaf_gradient_scatter.py --benchmark
        run_benchmark()
    else:
        # Quick correctness test
        _run_test()
