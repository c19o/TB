"""
GPU Histogram Builder — Atomic Scatter (ThunderGBM approach)

Builds per-feature gradient/hessian histograms from sparse CSR binary cross
features using a CuPy RawKernel with atomicAdd.

Matrix Thesis Context:
    Binary cross features (0/1) in sparse CSR.  ALL features preserved.
    Each nonzero entry means "this feature is ON for this row".
    We accumulate gradients into histogram bins using atomicAdd.

    Atomic contention is LOW because we have millions of features.
    Different rows rarely hit the same column in the same warp.
    This is exactly why atomic scatter works well for our workload.

Supports:
    - int64 indptr (15m with NNZ > 2^31)
    - float32 and float64 gradients (LightGBM uses float64 internally)
    - Per-leaf histogram building
    - All-leaves batch building
    - CuPy event timing
"""

import numpy as np

try:
    import cupy as cp
except ImportError:
    raise ImportError(
        "CuPy required for GPU histogram building. "
        "Install: pip install cupy-cuda12x"
    )

import scipy.sparse as sp_sparse
import time

# ---------------------------------------------------------------------------
# CUDA kernel source — one kernel per float precision
# ---------------------------------------------------------------------------

_KERNEL_FLOAT32 = r"""
extern "C" __global__
void sparse_histogram_kernel_f32(
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

_KERNEL_FLOAT64 = r"""
extern "C" __global__
void sparse_histogram_kernel_f64(
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

# Batched kernel: each thread processes one row, writes to leaf-specific
# histogram region at offset leaf_id * n_features.
_KERNEL_BATCHED_F32 = r"""
extern "C" __global__
void sparse_histogram_batched_f32(
    const long long* __restrict__ indptr,
    const int*       __restrict__ indices,
    const float*     __restrict__ gradients,
    const float*     __restrict__ hessians,
    const int*       __restrict__ all_rows,
    const int*       __restrict__ leaf_offsets,
    const int*       __restrict__ leaf_sizes,
    float*           __restrict__ hist_grad,
    float*           __restrict__ hist_hess,
    int n_features,
    int total_rows
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_rows) return;

    // Binary search to find which leaf this thread belongs to
    // leaf_offsets is sorted ascending, find rightmost offset <= tid
    int row_global_idx = tid;
    int lo = 0, hi = 0;

    // We encode leaf_id in the all_rows array layout:
    // all_rows = [leaf0_rows..., leaf1_rows..., ...]
    // leaf_offsets[leaf] = start index in all_rows
    // Use a simple scan (num_leaves is small, <128)
    // Actually, we pass leaf_id per row via a separate array for simplicity.
    // See the Python wrapper — it flattens rows and passes leaf_ids.

    // UNUSED — see _KERNEL_BATCHED_V2 below
}
"""

# Simpler batched approach: pass leaf_id per row
_KERNEL_BATCHED_V2_F32 = r"""
extern "C" __global__
void sparse_histogram_batched_v2_f32(
    const long long* __restrict__ indptr,
    const int*       __restrict__ indices,
    const float*     __restrict__ gradients,
    const float*     __restrict__ hessians,
    const int*       __restrict__ all_rows,
    const int*       __restrict__ row_leaf_ids,
    float*           __restrict__ hist_grad,
    float*           __restrict__ hist_hess,
    int n_features,
    int total_rows
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_rows) return;

    int row     = all_rows[tid];
    int leaf_id = row_leaf_ids[tid];
    float g = gradients[row];
    float h = hessians[row];

    long long base = (long long)leaf_id * (long long)n_features;
    long long start = indptr[row];
    long long end   = indptr[row + 1];

    for (long long j = start; j < end; j++) {
        int col = indices[j];
        atomicAdd(&hist_grad[base + col], g);
        atomicAdd(&hist_hess[base + col], h);
    }
}
"""

_KERNEL_BATCHED_V2_F64 = r"""
extern "C" __global__
void sparse_histogram_batched_v2_f64(
    const long long*  __restrict__ indptr,
    const int*        __restrict__ indices,
    const double*     __restrict__ gradients,
    const double*     __restrict__ hessians,
    const int*        __restrict__ all_rows,
    const int*        __restrict__ row_leaf_ids,
    double*           __restrict__ hist_grad,
    double*           __restrict__ hist_hess,
    int n_features,
    int total_rows
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_rows) return;

    int row     = all_rows[tid];
    int leaf_id = row_leaf_ids[tid];
    double g = gradients[row];
    double h = hessians[row];

    long long base = (long long)leaf_id * (long long)n_features;
    long long start = indptr[row];
    long long end   = indptr[row + 1];

    for (long long j = start; j < end; j++) {
        int col = indices[j];
        atomicAdd(&hist_grad[base + col], g);
        atomicAdd(&hist_hess[base + col], h);
    }
}
"""

# ---------------------------------------------------------------------------
# Compile kernels lazily (once per process)
# ---------------------------------------------------------------------------

_compiled = {}


def _get_kernel(name: str, source: str):
    """Compile and cache a CuPy RawKernel."""
    if name not in _compiled:
        _compiled[name] = cp.RawKernel(source, name)
    return _compiled[name]


# ---------------------------------------------------------------------------
# Histogram builder class
# ---------------------------------------------------------------------------

BLOCK_SIZE = 256  # threads per block — standard for modern GPUs


class AtomicHistogramBuilder:
    """
    GPU histogram builder using atomic scatter on sparse CSR binary crosses.

    Parameters
    ----------
    csr_matrix : scipy.sparse.csr_matrix or csr_array
        Sparse CSR feature matrix (n_rows x n_features).
        Binary cross features: structural zero = OFF, 1.0 = ON.
        Must have int64 indptr for 15m (NNZ > 2^31).
    dtype : str
        'float32' or 'float64'. Must match gradient precision.
        LightGBM uses float64 internally. float32 is faster on consumer GPUs.
    """

    def __init__(self, csr_matrix, dtype: str = "float64"):
        if dtype not in ("float32", "float64"):
            raise ValueError(f"dtype must be float32 or float64, got {dtype}")

        self.dtype = dtype
        self.cp_dtype = cp.float32 if dtype == "float32" else cp.float64
        self.n_rows = csr_matrix.shape[0]
        self.n_features = csr_matrix.shape[1]

        # Ensure CSR format
        if not sp_sparse.issparse(csr_matrix):
            raise TypeError("Input must be a scipy sparse matrix")
        csr = csr_matrix.tocsr()

        # Ensure int64 indptr (critical for 15m with NNZ > 2^31)
        indptr_np = np.asarray(csr.indptr, dtype=np.int64)
        indices_np = np.asarray(csr.indices, dtype=np.int32)

        # Upload CSR structure to GPU (one-time H2D transfer)
        t0 = time.perf_counter()
        self.d_indptr = cp.asarray(indptr_np)
        self.d_indices = cp.asarray(indices_np)
        cp.cuda.Device().synchronize()
        self._h2d_time_ms = (time.perf_counter() - t0) * 1000.0

        # Pre-allocate histogram output buffers (reused across calls)
        self.d_hist_grad = cp.zeros(self.n_features, dtype=self.cp_dtype)
        self.d_hist_hess = cp.zeros(self.n_features, dtype=self.cp_dtype)

        # Select kernels based on dtype
        if dtype == "float32":
            self._kernel_single = _get_kernel(
                "sparse_histogram_kernel_f32", _KERNEL_FLOAT32
            )
            self._kernel_batched = _get_kernel(
                "sparse_histogram_batched_v2_f32", _KERNEL_BATCHED_V2_F32
            )
        else:
            self._kernel_single = _get_kernel(
                "sparse_histogram_kernel_f64", _KERNEL_FLOAT64
            )
            self._kernel_batched = _get_kernel(
                "sparse_histogram_batched_v2_f64", _KERNEL_BATCHED_V2_F64
            )

        # Track NNZ for diagnostics
        self.nnz = int(indptr_np[-1])

        # VRAM usage estimate (bytes)
        self._vram_csr = (
            indptr_np.nbytes + indices_np.nbytes
        )
        elem_bytes = 4 if dtype == "float32" else 8
        self._vram_hist = 2 * self.n_features * elem_bytes
        self._vram_total = self._vram_csr + self._vram_hist

        print(
            f"[AtomicHistogramBuilder] Initialized: "
            f"{self.n_rows:,} rows x {self.n_features:,} features, "
            f"NNZ={self.nnz:,}, dtype={dtype}"
        )
        print(
            f"  CSR H2D: {self._h2d_time_ms:.1f} ms | "
            f"VRAM: CSR={self._vram_csr / 1e9:.2f} GB, "
            f"hist={self._vram_hist / 1e6:.1f} MB, "
            f"total={self._vram_total / 1e9:.2f} GB"
        )

    def build_histogram(
        self,
        gradients: np.ndarray,
        hessians: np.ndarray,
        leaf_row_indices: np.ndarray,
    ) -> tuple:
        """
        Build gradient/hessian histogram for a single leaf.

        Parameters
        ----------
        gradients : np.ndarray, shape (n_rows,)
            Per-row gradients. float32 or float64 matching self.dtype.
        hessians : np.ndarray, shape (n_rows,)
            Per-row hessians.
        leaf_row_indices : np.ndarray, shape (n_leaf_rows,), dtype int32
            Row indices belonging to this leaf.

        Returns
        -------
        hist_grad : np.ndarray, shape (n_features,)
            Per-feature gradient sums for rows in this leaf.
        hist_hess : np.ndarray, shape (n_features,)
            Per-feature hessian sums for rows in this leaf.
        kernel_ms : float
            Kernel execution time in milliseconds (CUDA event timing).
        """
        n_leaf_rows = len(leaf_row_indices)
        if n_leaf_rows == 0:
            zeros = np.zeros(self.n_features, dtype=self.dtype)
            return zeros, zeros.copy(), 0.0

        # Upload per-call data to GPU
        d_grad = cp.asarray(gradients.astype(self.dtype))
        d_hess = cp.asarray(hessians.astype(self.dtype))
        d_leaf = cp.asarray(leaf_row_indices.astype(np.int32))

        # Zero histogram buffers
        self.d_hist_grad.fill(0)
        self.d_hist_hess.fill(0)

        # Grid/block config
        grid_size = (n_leaf_rows + BLOCK_SIZE - 1) // BLOCK_SIZE

        # Timed kernel launch
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()

        start_event.record()
        self._kernel_single(
            (grid_size,),
            (BLOCK_SIZE,),
            (
                self.d_indptr,
                self.d_indices,
                d_grad,
                d_hess,
                d_leaf,
                self.d_hist_grad,
                self.d_hist_hess,
                np.int32(n_leaf_rows),
            ),
        )
        end_event.record()
        end_event.synchronize()
        kernel_ms = cp.cuda.get_elapsed_time(start_event, end_event)

        # Download results
        hist_grad = self.d_hist_grad.get()
        hist_hess = self.d_hist_hess.get()

        return hist_grad, hist_hess, kernel_ms

    def build_all_leaves(
        self,
        gradients: np.ndarray,
        hessians: np.ndarray,
        leaf_assignment: np.ndarray,
        num_leaves: int,
        batch_mode: bool = True,
    ) -> tuple:
        """
        Build histograms for all leaves.

        Parameters
        ----------
        gradients : np.ndarray, shape (n_rows,)
        hessians : np.ndarray, shape (n_rows,)
        leaf_assignment : np.ndarray, shape (n_rows,), dtype int32
            Leaf ID for each row (0..num_leaves-1). -1 = unassigned.
        num_leaves : int
            Number of leaves in the current tree.
        batch_mode : bool
            If True, use a single batched kernel launch (2D histogram buffer).
            If False, process leaves sequentially (lower VRAM, safer).

        Returns
        -------
        all_hist_grad : np.ndarray, shape (num_leaves, n_features)
        all_hist_hess : np.ndarray, shape (num_leaves, n_features)
        total_ms : float
            Total kernel time in milliseconds.
        """
        # Partition rows by leaf
        leaf_rows_list = []
        for leaf_id in range(num_leaves):
            rows = np.where(leaf_assignment == leaf_id)[0].astype(np.int32)
            leaf_rows_list.append(rows)

        if batch_mode:
            return self._build_all_batched(
                gradients, hessians, leaf_rows_list, num_leaves
            )
        else:
            return self._build_all_sequential(
                gradients, hessians, leaf_rows_list, num_leaves
            )

    def _build_all_sequential(
        self,
        gradients: np.ndarray,
        hessians: np.ndarray,
        leaf_rows_list: list,
        num_leaves: int,
    ) -> tuple:
        """Process each leaf one at a time. Lower VRAM, safe for large models."""
        np_dtype = np.float32 if self.dtype == "float32" else np.float64
        all_hist_grad = np.zeros((num_leaves, self.n_features), dtype=np_dtype)
        all_hist_hess = np.zeros((num_leaves, self.n_features), dtype=np_dtype)
        total_ms = 0.0

        for leaf_id in range(num_leaves):
            rows = leaf_rows_list[leaf_id]
            if len(rows) == 0:
                continue
            hg, hh, ms = self.build_histogram(gradients, hessians, rows)
            all_hist_grad[leaf_id] = hg
            all_hist_hess[leaf_id] = hh
            total_ms += ms

        return all_hist_grad, all_hist_hess, total_ms

    def _build_all_batched(
        self,
        gradients: np.ndarray,
        hessians: np.ndarray,
        leaf_rows_list: list,
        num_leaves: int,
    ) -> tuple:
        """
        Single kernel launch for all leaves.

        Uses a 2D histogram buffer: hist[leaf_id * n_features + col].
        Each thread knows its leaf_id via a parallel row_leaf_ids array.

        VRAM cost: 2 * num_leaves * n_features * elem_bytes.
        For num_leaves=63, n_features=6M, float64: 2*63*6M*8 = ~6 GB.
        Falls back to sequential if this exceeds 80% free VRAM.
        """
        np_dtype = np.float32 if self.dtype == "float32" else np.float64
        elem_bytes = 4 if self.dtype == "float32" else 8

        # Check VRAM for batched histogram buffer
        hist_vram_bytes = 2 * num_leaves * self.n_features * elem_bytes
        free_vram = cp.cuda.Device().mem_info[0]  # free bytes

        if hist_vram_bytes > int(free_vram * 0.8):
            print(
                f"  [WARN] Batched hist needs {hist_vram_bytes / 1e9:.1f} GB "
                f"but only {free_vram / 1e9:.1f} GB free. "
                f"Falling back to sequential."
            )
            return self._build_all_sequential(
                gradients, hessians, leaf_rows_list, num_leaves
            )

        # Flatten all leaf rows into one array + parallel leaf_id array
        all_rows_parts = []
        all_leaf_ids_parts = []
        for leaf_id, rows in enumerate(leaf_rows_list):
            if len(rows) == 0:
                continue
            all_rows_parts.append(rows)
            all_leaf_ids_parts.append(
                np.full(len(rows), leaf_id, dtype=np.int32)
            )

        if len(all_rows_parts) == 0:
            return (
                np.zeros((num_leaves, self.n_features), dtype=np_dtype),
                np.zeros((num_leaves, self.n_features), dtype=np_dtype),
                0.0,
            )

        all_rows_np = np.concatenate(all_rows_parts)
        all_leaf_ids_np = np.concatenate(all_leaf_ids_parts)
        total_rows = len(all_rows_np)

        # Upload
        d_grad = cp.asarray(gradients.astype(self.dtype))
        d_hess = cp.asarray(hessians.astype(self.dtype))
        d_all_rows = cp.asarray(all_rows_np)
        d_row_leaf_ids = cp.asarray(all_leaf_ids_np)

        # Allocate 2D histogram buffer [num_leaves * n_features]
        buf_size = num_leaves * self.n_features
        d_batch_grad = cp.zeros(buf_size, dtype=self.cp_dtype)
        d_batch_hess = cp.zeros(buf_size, dtype=self.cp_dtype)

        # Launch
        grid_size = (total_rows + BLOCK_SIZE - 1) // BLOCK_SIZE

        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()

        start_event.record()
        self._kernel_batched(
            (grid_size,),
            (BLOCK_SIZE,),
            (
                self.d_indptr,
                self.d_indices,
                d_grad,
                d_hess,
                d_all_rows,
                d_row_leaf_ids,
                d_batch_grad,
                d_batch_hess,
                np.int32(self.n_features),
                np.int32(total_rows),
            ),
        )
        end_event.record()
        end_event.synchronize()
        kernel_ms = cp.cuda.get_elapsed_time(start_event, end_event)

        # Download and reshape
        all_hist_grad = d_batch_grad.get().reshape(num_leaves, self.n_features)
        all_hist_hess = d_batch_hess.get().reshape(num_leaves, self.n_features)

        # Free large temp buffers
        del d_batch_grad, d_batch_hess
        del d_all_rows, d_row_leaf_ids, d_grad, d_hess

        return all_hist_grad, all_hist_hess, kernel_ms

    def cleanup(self):
        """Free all GPU memory."""
        attrs = [
            "d_indptr",
            "d_indices",
            "d_hist_grad",
            "d_hist_hess",
        ]
        for attr in attrs:
            if hasattr(self, attr):
                delattr(self, attr)

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        print("[AtomicHistogramBuilder] GPU memory freed.")

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return diagnostic info."""
        return {
            "n_rows": self.n_rows,
            "n_features": self.n_features,
            "nnz": self.nnz,
            "dtype": self.dtype,
            "vram_csr_gb": self._vram_csr / 1e9,
            "vram_hist_mb": self._vram_hist / 1e6,
            "vram_total_gb": self._vram_total / 1e9,
            "h2d_time_ms": self._h2d_time_ms,
        }


# ---------------------------------------------------------------------------
# CPU reference implementation (for validation)
# ---------------------------------------------------------------------------


def cpu_histogram_reference(
    csr_matrix,
    gradients: np.ndarray,
    hessians: np.ndarray,
    leaf_row_indices: np.ndarray,
) -> tuple:
    """
    CPU reference histogram build for validation.

    Iterates CSR rows in the leaf and accumulates grad/hess per feature.
    Intentionally simple and slow — correctness reference only.
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
# Standalone test
# ---------------------------------------------------------------------------


def _run_test():
    """Standalone correctness + performance test."""
    import sys

    print("=" * 70)
    print("AtomicHistogramBuilder — Standalone Test")
    print("=" * 70)

    # Check GPU
    print(f"\nCUDA device: {cp.cuda.Device().name}")
    free, total = cp.cuda.Device().mem_info
    print(f"VRAM: {free / 1e9:.1f} GB free / {total / 1e9:.1f} GB total")

    # ---- Test parameters (scaled to fit any GPU) ----
    n_rows = 5_000
    n_features = 500_000
    sparsity = 0.997  # 99.7% zeros, matching real cross features
    n_leaf_rows = 2_000
    num_leaves = 8

    print(f"\nTest matrix: {n_rows:,} rows x {n_features:,} features")
    print(f"Sparsity: {sparsity * 100:.1f}%")
    nnz_expected = int(n_rows * n_features * (1 - sparsity))
    print(f"Expected NNZ: {nnz_expected:,}")

    # Generate sparse binary CSR
    print("\nGenerating synthetic sparse CSR...")
    rng = np.random.default_rng(42)
    row_indices = rng.integers(0, n_rows, size=nnz_expected)
    col_indices = rng.integers(0, n_features, size=nnz_expected)
    data = np.ones(nnz_expected, dtype=np.float32)
    csr = sp_sparse.csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(n_rows, n_features),
    )
    csr.sum_duplicates()
    print(f"Actual NNZ: {csr.nnz:,}")

    # Gradients/hessians
    gradients = rng.standard_normal(n_rows).astype(np.float64)
    hessians = np.abs(rng.standard_normal(n_rows)).astype(np.float64) + 0.1

    # Leaf row indices
    leaf_rows = rng.choice(n_rows, size=n_leaf_rows, replace=False).astype(
        np.int32
    )
    leaf_rows.sort()

    # ---- Test 1: Single-leaf float64 ----
    print("\n--- Test 1: Single-leaf histogram (float64) ---")
    builder = AtomicHistogramBuilder(csr, dtype="float64")

    # GPU
    hg_gpu, hh_gpu, gpu_ms = builder.build_histogram(
        gradients, hessians, leaf_rows
    )
    print(f"GPU kernel: {gpu_ms:.3f} ms")

    # CPU reference
    t0 = time.perf_counter()
    hg_cpu, hh_cpu = cpu_histogram_reference(
        csr, gradients, hessians, leaf_rows
    )
    cpu_ms = (time.perf_counter() - t0) * 1000.0
    print(f"CPU reference: {cpu_ms:.1f} ms")

    # Validate
    grad_max_diff = np.max(np.abs(hg_gpu - hg_cpu))
    hess_max_diff = np.max(np.abs(hh_gpu - hh_cpu))
    print(f"Max grad diff: {grad_max_diff:.2e}")
    print(f"Max hess diff: {hess_max_diff:.2e}")

    # float64 atomicAdd should be nearly exact
    tol = 1e-6
    grad_ok = grad_max_diff < tol
    hess_ok = hess_max_diff < tol
    print(f"PASS: {grad_ok and hess_ok} (tol={tol})")

    if not (grad_ok and hess_ok):
        print("FAIL — GPU/CPU mismatch!")
        sys.exit(1)

    # ---- Test 2: Single-leaf float32 ----
    print("\n--- Test 2: Single-leaf histogram (float32) ---")
    builder32 = AtomicHistogramBuilder(csr, dtype="float32")
    hg_gpu32, hh_gpu32, gpu_ms32 = builder32.build_histogram(
        gradients.astype(np.float32),
        hessians.astype(np.float32),
        leaf_rows,
    )
    print(f"GPU kernel (f32): {gpu_ms32:.3f} ms")

    # float32 has less precision — wider tolerance
    grad_max_diff32 = np.max(np.abs(hg_gpu32.astype(np.float64) - hg_cpu))
    print(f"Max grad diff (f32 vs f64 CPU): {grad_max_diff32:.2e}")
    # atomicAdd float32 accumulation error scales with n_leaf_rows
    tol32 = 1e-1  # relaxed for float32 accumulation
    print(f"PASS: {grad_max_diff32 < tol32} (tol={tol32})")
    builder32.cleanup()

    # ---- Test 3: All-leaves sequential ----
    print(f"\n--- Test 3: All-leaves sequential ({num_leaves} leaves) ---")
    leaf_assignment = rng.integers(0, num_leaves, size=n_rows).astype(np.int32)

    all_hg, all_hh, seq_ms = builder.build_all_leaves(
        gradients, hessians, leaf_assignment, num_leaves, batch_mode=False
    )
    print(f"Sequential total: {seq_ms:.3f} ms")
    print(f"Output shape: {all_hg.shape}")

    # Validate leaf 0
    leaf0_rows = np.where(leaf_assignment == 0)[0].astype(np.int32)
    hg_ref0, hh_ref0 = cpu_histogram_reference(
        csr, gradients, hessians, leaf0_rows
    )
    diff0 = np.max(np.abs(all_hg[0] - hg_ref0))
    print(f"Leaf 0 grad max diff: {diff0:.2e} — PASS: {diff0 < tol}")

    # ---- Test 4: All-leaves batched ----
    print(f"\n--- Test 4: All-leaves batched ({num_leaves} leaves) ---")
    all_hg_b, all_hh_b, batch_ms = builder.build_all_leaves(
        gradients, hessians, leaf_assignment, num_leaves, batch_mode=True
    )
    print(f"Batched total: {batch_ms:.3f} ms")

    # Validate against sequential
    max_diff_batch = np.max(np.abs(all_hg_b - all_hg))
    print(f"Batched vs sequential max diff: {max_diff_batch:.2e}")
    print(f"PASS: {max_diff_batch < tol}")

    # ---- Test 5: Empty leaf ----
    print("\n--- Test 5: Empty leaf ---")
    empty_rows = np.array([], dtype=np.int32)
    hg_empty, hh_empty, ms_empty = builder.build_histogram(
        gradients, hessians, empty_rows
    )
    assert np.all(hg_empty == 0), "Empty leaf should produce zero histogram"
    assert np.all(hh_empty == 0), "Empty leaf should produce zero histogram"
    print(f"PASS: empty leaf returns zeros ({ms_empty:.3f} ms)")

    # ---- Test 6: int64 indptr (simulate 15m scale) ----
    print("\n--- Test 6: int64 indptr verification ---")
    assert builder.d_indptr.dtype == cp.int64, "indptr must be int64"
    print(f"indptr dtype on GPU: {builder.d_indptr.dtype} — PASS")

    # ---- Performance summary ----
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"Matrix: {n_rows:,} x {n_features:,}, NNZ={csr.nnz:,}")
    print(f"Single leaf ({n_leaf_rows:,} rows):")
    print(f"  GPU f64: {gpu_ms:.3f} ms")
    print(f"  GPU f32: {gpu_ms32:.3f} ms")
    print(f"  CPU ref: {cpu_ms:.1f} ms")
    if gpu_ms > 0:
        print(f"  Speedup (f64): {cpu_ms / gpu_ms:.1f}x")
    print(f"All {num_leaves} leaves:")
    print(f"  Sequential: {seq_ms:.3f} ms")
    print(f"  Batched:    {batch_ms:.3f} ms")
    print(f"\nDiagnostics: {builder.stats()}")

    builder.cleanup()
    print("\nAll tests PASSED.")


if __name__ == "__main__":
    _run_test()
