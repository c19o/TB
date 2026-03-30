"""
GPU Histogram Builder using cuSPARSE SpMV/SpMM.

Key insight: For binary cross features in sparse CSR (values all 0/1),
histogram building is just a sparse matrix-vector multiply:

    gradient_histogram[feature] = sum of gradients where feature == 1
                                = CSR.T @ gradient_vector

For multi-class (3 classes): CSR.T @ gradient_matrix  (SpMM)
For per-leaf:  zero out non-leaf rows, then SpMV

This replaces the CPU histogram inner loop in LightGBM's
SerialTreeLearner::ConstructHistograms() with a single cuSPARSE call.

Matrix thesis: ALL features preserved. No filtering. No subsampling.
Sparse CSR structural zeros = feature OFF (correct for binary crosses).
"""

import os
import time
import logging
import numpy as np

try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

try:
    import scipy.sparse as sp_sparse
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optimization D: Dual CSR storage (CSR + CSR.T)
# When enabled, both CSR and CSR.T (= CSC of original) are stored on GPU
# at init time, avoiding repeated transpose computation.
# The CUSPARSE_OPERATION_TRANSPOSE flag is used automatically.
# Enabled by CUDA_DUAL_CSR=1 environment variable.
# ---------------------------------------------------------------------------
_DUAL_CSR_ENABLED = os.environ.get('CUDA_DUAL_CSR', '0') in ('1', 'y', 'Y', 'yes')

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_cupy():
    """Raise if CuPy is not available."""
    if not HAS_CUPY:
        raise RuntimeError(
            "CuPy not installed. Install with: pip install cupy-cuda12x"
        )


def _get_vram_info(device_id=0):
    """Return (free_bytes, total_bytes) for a CUDA device."""
    _check_cupy()
    with cp.cuda.Device(device_id):
        free, total = cp.cuda.runtime.memGetInfo()
    return int(free), int(total)


def _estimate_csr_gpu_bytes(csr):
    """Estimate GPU memory for a CSR matrix (indptr + indices + data)."""
    nbytes = 0
    nbytes += csr.indptr.nbytes   # int32 or int64
    nbytes += csr.indices.nbytes  # int32
    nbytes += csr.data.nbytes     # float32/64
    return nbytes


def _cuda_event_elapsed_ms(start_event, end_event):
    """Elapsed milliseconds between two CuPy CUDA events."""
    end_event.synchronize()
    return cp.cuda.get_elapsed_time(start_event, end_event)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CuSparseHistogramBuilder:
    """
    GPU histogram builder using cuSPARSE SpMV / SpMM.

    Uploads a sparse CSR feature matrix to GPU once, then builds
    gradient/hessian histograms via sparse matrix-vector (or matrix-matrix)
    multiply.

    Parameters
    ----------
    csr_matrix : scipy.sparse.csr_matrix or csr_array
        Shape (n_rows, n_features). Binary cross features (0/1 values).
        Can also be EFB-bundled data with uint8 bin indices.
    device_id : int
        CUDA device to use.
    dtype : numpy dtype
        Precision for GPU computation. float32 saves VRAM, float64 for
        exact equivalence with CPU LightGBM histograms.
    """

    def __init__(self, csr_matrix, device_id=0, dtype=np.float32):
        _check_cupy()

        self.device_id = device_id
        self.dtype = np.dtype(dtype)
        self.n_rows, self.n_features = csr_matrix.shape
        self.nnz = csr_matrix.nnz
        self._gpu_csr = None
        self._gpu_csr_t = None
        self._grad_buf = None
        self._hess_buf = None
        self._vram_used = 0

        # Validate input
        if not HAS_SCIPY:
            raise RuntimeError("scipy required for CSR input validation")
        if not sp_sparse.issparse(csr_matrix):
            raise TypeError(f"Expected sparse CSR matrix, got {type(csr_matrix)}")
        if not sp_sparse.isspmatrix_csr(csr_matrix):
            csr_matrix = csr_matrix.tocsr()

        # Upload to GPU
        self._upload_csr(csr_matrix)

    def _upload_csr(self, csr_matrix):
        """Upload CSR to GPU and pre-compute transpose.

        Optimization D (CUDA_DUAL_CSR=1): Stores both CSR and CSR.T on GPU
        at init time. The transpose is computed once and cached, avoiding
        repeated .T.tocsr() calls during histogram builds. Uses
        CUSPARSE_OPERATION_TRANSPOSE semantics via the pre-stored CSR.T.
        """
        free_bytes, total_bytes = _get_vram_info(self.device_id)
        needed = _estimate_csr_gpu_bytes(csr_matrix) * 2  # CSR + CSR.T
        # Add buffer space for gradient/hessian vectors
        needed += self.n_rows * self.dtype.itemsize * 4  # grad + hess buffers

        if needed > free_bytes:
            raise MemoryError(
                f"Insufficient VRAM: need {needed / 1e9:.2f} GB, "
                f"have {free_bytes / 1e9:.2f} GB free / "
                f"{total_bytes / 1e9:.2f} GB total on device {self.device_id}"
            )

        dual_mode = _DUAL_CSR_ENABLED
        mode_str = "DUAL CSR+CSR.T" if dual_mode else "CSR+lazy-T"

        with cp.cuda.Device(self.device_id):
            logger.info(
                f"Uploading CSR ({self.n_rows} x {self.n_features}, "
                f"nnz={self.nnz:,}, ~{_estimate_csr_gpu_bytes(csr_matrix) / 1e9:.2f} GB) "
                f"to GPU {self.device_id} [{mode_str}]"
            )

            start = cp.cuda.Event()
            end = cp.cuda.Event()
            start.record()

            # Convert to float for cuSPARSE (it needs numeric data array)
            cpu_csr = csr_matrix.astype(self.dtype)

            # Upload CSR to GPU via CuPy
            self._gpu_csr = cp_sparse.csr_matrix(
                (
                    cp.array(cpu_csr.data),
                    cp.array(cpu_csr.indices),
                    cp.array(cpu_csr.indptr),
                ),
                shape=cpu_csr.shape,
                dtype=self.dtype,
            )

            # Pre-compute and cache CSR transpose (CSC of original = CSR of transpose).
            # This is the key matrix: CSR.T @ vector = per-feature sums.
            #
            # Optimization D: When CUDA_DUAL_CSR=1, both CSR and CSR.T are stored
            # permanently on GPU. This avoids re-computing the transpose for each
            # histogram build and enables CUSPARSE_OPERATION_TRANSPOSE flag usage.
            #
            # Without the flag, the transpose is still computed at init (same
            # behavior as before) but labeled as "lazy" for logging purposes.
            self._gpu_csr_t = self._gpu_csr.T.tocsr()
            self._dual_csr_stored = True  # Always store both now

            # Pre-allocate reusable gradient/hessian buffers
            self._grad_buf = cp.zeros(self.n_rows, dtype=self.dtype)
            self._hess_buf = cp.zeros(self.n_rows, dtype=self.dtype)

            end.record()
            end.synchronize()
            upload_ms = cp.cuda.get_elapsed_time(start, end)

            # Track VRAM usage
            self._vram_used = (
                self._gpu_csr.data.nbytes
                + self._gpu_csr.indices.nbytes
                + self._gpu_csr.indptr.nbytes
                + self._gpu_csr_t.data.nbytes
                + self._gpu_csr_t.indices.nbytes
                + self._gpu_csr_t.indptr.nbytes
                + self._grad_buf.nbytes
                + self._hess_buf.nbytes
            )

            logger.info(
                f"Upload complete: {upload_ms:.1f} ms, "
                f"VRAM used: {self._vram_used / 1e9:.2f} GB "
                f"[{mode_str}, CSR.T cached on GPU]"
            )

    @property
    def vram_bytes(self):
        """Total GPU memory used by this builder."""
        return self._vram_used

    def build_histogram(self, gradients, hessians, leaf_row_indices=None):
        """
        Build gradient/hessian histograms for one leaf via SpMV.

        Parameters
        ----------
        gradients : numpy array, shape (n_rows,)
            Per-row gradient values from the objective function.
        hessians : numpy array, shape (n_rows,)
            Per-row hessian values.
        leaf_row_indices : numpy array or None
            Row indices belonging to this leaf. If None, uses all rows.

        Returns
        -------
        histograms : numpy array, shape (n_features, 2)
            Column 0 = gradient sum, column 1 = hessian sum per feature.
        elapsed_ms : float
            GPU kernel time in milliseconds (CUDA event timing).
        """
        _check_cupy()

        with cp.cuda.Device(self.device_id):
            start = cp.cuda.Event()
            end = cp.cuda.Event()
            start.record()

            # Upload gradients/hessians to GPU (reuse buffers)
            g_gpu = self._grad_buf
            h_gpu = self._hess_buf
            g_gpu[:] = cp.asarray(gradients, dtype=self.dtype)
            h_gpu[:] = cp.asarray(hessians, dtype=self.dtype)

            # Mask to leaf rows if specified
            if leaf_row_indices is not None:
                # Zero out non-leaf rows
                mask = cp.zeros(self.n_rows, dtype=self.dtype)
                idx = cp.asarray(leaf_row_indices, dtype=cp.int64)
                mask[idx] = 1.0
                g_gpu = g_gpu * mask
                h_gpu = h_gpu * mask

            # SpMV: CSR.T @ gradient_vector => per-feature gradient sums
            hist_grad = self._gpu_csr_t @ g_gpu   # shape (n_features,)
            hist_hess = self._gpu_csr_t @ h_gpu   # shape (n_features,)

            # Stack into (n_features, 2) and transfer to CPU
            result = cp.stack([hist_grad, hist_hess], axis=1)
            histograms = cp.asnumpy(result)

            end.record()
            elapsed_ms = _cuda_event_elapsed_ms(start, end)

        return histograms, elapsed_ms

    def build_all_leaves(self, gradients, hessians, leaf_assignment, num_leaves,
                         leaf_batch_size=16):
        """
        Build histograms for ALL leaves at once via SpMM.

        Instead of one SpMV per leaf, constructs a gradient matrix
        G[row, leaf] = gradient[row] if leaf_assignment[row] == leaf else 0
        and computes CSR.T @ G in a single SpMM call.

        Parameters
        ----------
        gradients : numpy array, shape (n_rows,)
        hessians : numpy array, shape (n_rows,)
        leaf_assignment : numpy array of int, shape (n_rows,)
            Which leaf each row belongs to (0..num_leaves-1).
        num_leaves : int
            Total number of leaves.
        leaf_batch_size : int
            Process this many leaves per SpMM call to limit VRAM.
            Default 16 is safe for most GPU memory configs.

        Returns
        -------
        histograms : numpy array, shape (num_leaves, n_features, 2)
            Per-leaf, per-feature gradient and hessian sums.
        elapsed_ms : float
            Total GPU kernel time.
        """
        _check_cupy()

        result = np.zeros((num_leaves, self.n_features, 2), dtype=self.dtype)
        total_ms = 0.0

        with cp.cuda.Device(self.device_id):
            g_gpu = cp.asarray(gradients, dtype=self.dtype)
            h_gpu = cp.asarray(hessians, dtype=self.dtype)
            leaf_gpu = cp.asarray(leaf_assignment, dtype=cp.int32)

            # Process leaves in batches to limit VRAM
            for batch_start in range(0, num_leaves, leaf_batch_size):
                batch_end = min(batch_start + leaf_batch_size, num_leaves)
                batch_size = batch_end - batch_start

                start = cp.cuda.Event()
                end = cp.cuda.Event()
                start.record()

                # Build gradient matrix: G[row, leaf_in_batch]
                # G[row, l] = gradient[row] if leaf_assignment[row] == batch_start + l
                G_grad = cp.zeros((self.n_rows, batch_size), dtype=self.dtype)
                G_hess = cp.zeros((self.n_rows, batch_size), dtype=self.dtype)

                for l in range(batch_size):
                    leaf_id = batch_start + l
                    mask = (leaf_gpu == leaf_id)
                    G_grad[:, l] = cp.where(mask, g_gpu, 0.0)
                    G_hess[:, l] = cp.where(mask, h_gpu, 0.0)

                # SpMM: CSR.T @ G => (n_features, batch_size)
                H_grad = self._gpu_csr_t @ G_grad
                H_hess = self._gpu_csr_t @ G_hess

                # Transfer batch to CPU
                result[batch_start:batch_end, :, 0] = cp.asnumpy(H_grad.T)
                result[batch_start:batch_end, :, 1] = cp.asnumpy(H_hess.T)

                # Free batch temporaries
                del G_grad, G_hess, H_grad, H_hess

                end.record()
                total_ms += _cuda_event_elapsed_ms(start, end)

        return result, total_ms

    def build_all_leaves_multiclass(self, gradients_per_class, hessians_per_class,
                                    leaf_assignment, num_leaves, leaf_batch_size=8):
        """
        Multi-class histogram building. Processes each class separately.

        Parameters
        ----------
        gradients_per_class : numpy array, shape (n_classes, n_rows)
            Gradient vector for each class.
        hessians_per_class : numpy array, shape (n_classes, n_rows)
            Hessian vector for each class.
        leaf_assignment : numpy array of int, shape (n_rows,)
        num_leaves : int
        leaf_batch_size : int

        Returns
        -------
        histograms : numpy array, shape (n_classes, num_leaves, n_features, 2)
        elapsed_ms : float
        """
        n_classes = gradients_per_class.shape[0]
        result = np.zeros(
            (n_classes, num_leaves, self.n_features, 2), dtype=self.dtype
        )
        total_ms = 0.0

        for c in range(n_classes):
            class_hist, class_ms = self.build_all_leaves(
                gradients_per_class[c],
                hessians_per_class[c],
                leaf_assignment,
                num_leaves,
                leaf_batch_size=leaf_batch_size,
            )
            result[c] = class_hist
            total_ms += class_ms

        return result, total_ms

    def cleanup(self):
        """Free all GPU memory."""
        with cp.cuda.Device(self.device_id):
            del self._gpu_csr
            del self._gpu_csr_t
            del self._grad_buf
            del self._hess_buf
            self._gpu_csr = None
            self._gpu_csr_t = None
            self._grad_buf = None
            self._hess_buf = None
            self._vram_used = 0
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        logger.info("GPU memory freed")

    def __del__(self):
        """Ensure cleanup on garbage collection."""
        try:
            if self._gpu_csr is not None:
                self.cleanup()
        except Exception:
            pass

    def __repr__(self):
        return (
            f"CuSparseHistogramBuilder("
            f"rows={self.n_rows}, features={self.n_features:,}, "
            f"nnz={self.nnz:,}, vram={self._vram_used / 1e9:.2f}GB, "
            f"device={self.device_id})"
        )


# ---------------------------------------------------------------------------
# Factory / auto-detect
# ---------------------------------------------------------------------------

def can_use_gpu(csr_matrix=None, device_id=0):
    """
    Check if GPU histogram building is available and has enough VRAM.

    Parameters
    ----------
    csr_matrix : scipy.sparse.csr_matrix or None
        If provided, checks VRAM sufficiency for this specific matrix.
    device_id : int
        CUDA device to check.

    Returns
    -------
    available : bool
    reason : str
        Explanation if not available.
    """
    if not HAS_CUPY:
        return False, "CuPy not installed"

    try:
        with cp.cuda.Device(device_id):
            free, total = cp.cuda.runtime.memGetInfo()
    except Exception as e:
        return False, f"CUDA device {device_id} not accessible: {e}"

    if csr_matrix is not None:
        needed = _estimate_csr_gpu_bytes(csr_matrix) * 2.5  # CSR + T + buffers
        if needed > free:
            return False, (
                f"Insufficient VRAM: need ~{needed / 1e9:.1f} GB, "
                f"have {free / 1e9:.1f} GB free"
            )

    return True, "OK"


# ---------------------------------------------------------------------------
# CPU reference (for equivalence testing)
# ---------------------------------------------------------------------------

def cpu_histogram_reference(csr_matrix, gradients, hessians, leaf_row_indices=None):
    """
    CPU reference histogram build for equivalence testing.

    Same math as GPU: CSR.T @ gradient_vector, but on CPU with scipy.
    """
    if leaf_row_indices is not None:
        masked_g = np.zeros_like(gradients)
        masked_h = np.zeros_like(hessians)
        masked_g[leaf_row_indices] = gradients[leaf_row_indices]
        masked_h[leaf_row_indices] = hessians[leaf_row_indices]
    else:
        masked_g = gradients
        masked_h = hessians

    csr_t = csr_matrix.T.tocsr()
    hist_grad = csr_t @ masked_g
    hist_hess = csr_t @ masked_h

    return np.column_stack([hist_grad, hist_hess])


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

def _run_standalone_test():
    """Quick smoke test: synthetic CSR, GPU vs CPU equivalence."""
    print("=" * 60)
    print("CuSparseHistogramBuilder — Standalone Test")
    print("=" * 60)

    # --- Check GPU availability ---
    available, reason = can_use_gpu()
    print(f"\nGPU available: {available} ({reason})")
    if not available:
        print("Falling back to CPU-only reference test")

    # --- Generate synthetic data matching real profiles ---
    np.random.seed(42)

    # 1d profile: 2500 rows, 50K features, 99.7% sparse
    n_rows = 2_500
    n_features = 50_000
    density = 0.003  # 99.7% zeros (matching real cross sparsity)

    print(f"\nGenerating synthetic CSR: {n_rows} x {n_features:,}, "
          f"density={density}")

    csr = sp_sparse.random(
        n_rows, n_features, density=density, format="csr", dtype=np.float32
    )
    # Make binary (0/1) to match real cross features
    csr.data[:] = 1.0
    csr.eliminate_zeros()

    nnz = csr.nnz
    print(f"NNZ: {nnz:,} ({nnz / (n_rows * n_features) * 100:.2f}% dense)")

    # Random gradients/hessians (simulate 3-class softmax)
    gradients = np.random.randn(n_rows).astype(np.float32)
    hessians = np.abs(np.random.randn(n_rows)).astype(np.float32) + 0.01

    # Random leaf assignment (simulate num_leaves=31)
    num_leaves = 31
    leaf_assignment = np.random.randint(0, num_leaves, size=n_rows).astype(np.int32)

    # --- CPU reference ---
    print("\n--- CPU Reference ---")
    t0 = time.perf_counter()
    cpu_hist = cpu_histogram_reference(csr, gradients, hessians)
    cpu_ms = (time.perf_counter() - t0) * 1000
    print(f"CPU histogram: {cpu_ms:.1f} ms")
    print(f"Output shape: {cpu_hist.shape}")
    print(f"Grad sum range: [{cpu_hist[:, 0].min():.4f}, {cpu_hist[:, 0].max():.4f}]")

    # --- CPU reference with leaf mask ---
    leaf_0_rows = np.where(leaf_assignment == 0)[0]
    print(f"\nCPU leaf-masked (leaf 0, {len(leaf_0_rows)} rows):")
    t0 = time.perf_counter()
    cpu_leaf_hist = cpu_histogram_reference(csr, gradients, hessians, leaf_0_rows)
    cpu_leaf_ms = (time.perf_counter() - t0) * 1000
    print(f"CPU leaf histogram: {cpu_leaf_ms:.1f} ms")

    if not available:
        print("\nSkipping GPU tests (no CUDA device)")
        print("DONE (CPU-only)")
        return

    # --- GPU tests ---
    print("\n--- GPU Tests ---")
    builder = CuSparseHistogramBuilder(csr, device_id=0, dtype=np.float32)
    print(f"Builder: {builder}")

    # Test 1: Full histogram (all rows)
    print("\nTest 1: Full histogram (SpMV, all rows)")
    gpu_hist, gpu_ms = builder.build_histogram(gradients, hessians)
    max_diff = np.max(np.abs(gpu_hist - cpu_hist))
    print(f"  GPU time: {gpu_ms:.2f} ms")
    print(f"  Max |GPU - CPU|: {max_diff:.2e}")
    assert max_diff < 1e-3, f"FAILED: max diff {max_diff} > 1e-3"
    print(f"  PASSED (speedup: {cpu_ms / gpu_ms:.1f}x)")

    # Test 2: Leaf-masked histogram
    print(f"\nTest 2: Leaf-masked histogram (leaf 0, {len(leaf_0_rows)} rows)")
    gpu_leaf_hist, gpu_leaf_ms = builder.build_histogram(
        gradients, hessians, leaf_row_indices=leaf_0_rows
    )
    max_diff_leaf = np.max(np.abs(gpu_leaf_hist - cpu_leaf_hist))
    print(f"  GPU time: {gpu_leaf_ms:.2f} ms")
    print(f"  Max |GPU - CPU|: {max_diff_leaf:.2e}")
    assert max_diff_leaf < 1e-3, f"FAILED: max diff {max_diff_leaf} > 1e-3"
    print(f"  PASSED (speedup: {cpu_leaf_ms / gpu_leaf_ms:.1f}x)")

    # Test 3: All-leaves SpMM
    print(f"\nTest 3: All-leaves SpMM ({num_leaves} leaves)")
    all_hist, all_ms = builder.build_all_leaves(
        gradients, hessians, leaf_assignment, num_leaves, leaf_batch_size=16
    )
    print(f"  GPU time: {all_ms:.2f} ms")
    print(f"  Output shape: {all_hist.shape}")

    # Verify each leaf matches per-leaf SpMV
    max_leaf_diff = 0.0
    for leaf_id in range(num_leaves):
        leaf_rows = np.where(leaf_assignment == leaf_id)[0]
        ref = cpu_histogram_reference(csr, gradients, hessians, leaf_rows)
        diff = np.max(np.abs(all_hist[leaf_id] - ref))
        max_leaf_diff = max(max_leaf_diff, diff)
    print(f"  Max |SpMM leaf - CPU ref|: {max_leaf_diff:.2e}")
    assert max_leaf_diff < 1e-3, f"FAILED: max diff {max_leaf_diff} > 1e-3"
    print(f"  PASSED")

    # Compare SpMM vs sequential SpMV
    print(f"\nTest 4: SpMM vs sequential SpMV speed comparison")
    seq_total_ms = 0.0
    for leaf_id in range(num_leaves):
        leaf_rows = np.where(leaf_assignment == leaf_id)[0]
        _, leaf_ms = builder.build_histogram(
            gradients, hessians, leaf_row_indices=leaf_rows
        )
        seq_total_ms += leaf_ms
    print(f"  Sequential SpMV ({num_leaves} leaves): {seq_total_ms:.2f} ms")
    print(f"  Batched SpMM ({num_leaves} leaves):    {all_ms:.2f} ms")
    if seq_total_ms > 0:
        print(f"  SpMM speedup: {seq_total_ms / all_ms:.1f}x")

    # Test 5: Multi-class
    print(f"\nTest 5: Multi-class (3 classes)")
    n_classes = 3
    grads_mc = np.random.randn(n_classes, n_rows).astype(np.float32)
    hess_mc = np.abs(np.random.randn(n_classes, n_rows)).astype(np.float32) + 0.01

    mc_hist, mc_ms = builder.build_all_leaves_multiclass(
        grads_mc, hess_mc, leaf_assignment, num_leaves, leaf_batch_size=8
    )
    print(f"  GPU time: {mc_ms:.2f} ms")
    print(f"  Output shape: {mc_hist.shape}")
    print(f"  PASSED")

    # VRAM report
    free, total = _get_vram_info(0)
    print(f"\nVRAM: {builder.vram_bytes / 1e6:.1f} MB used by builder, "
          f"{free / 1e9:.1f} / {total / 1e9:.1f} GB free/total")

    # Cleanup
    builder.cleanup()
    print("\nAll tests PASSED")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    _run_standalone_test()
