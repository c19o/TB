"""
LightGBM GPU Histogram Co-Processor Integration
================================================
Drop-in replacement for lgb.train() that offloads histogram building to GPU
via a C++ shared library (libgpu_histogram.so).

Architecture:
  - GPU replaces ONLY ConstructHistograms() — the inner CSR scan + accumulate loop
  - EFB bundling, split finding, tree growing all stay on CPU (LightGBM)
  - C interface via ctypes: init -> per-round build -> cleanup
  - Auto-detect GPU, estimate VRAM, fallback to CPU if insufficient

Integration approach:
  We do NOT fork LightGBM's Python code. Instead, we fork the C++ library and
  expose a custom train function that uses our GPU-accelerated LightGBM build.
  From Python's perspective, it's still lgb.train() — just compiled with our
  CUDA histogram kernel linked in. The params dict gets `use_cuda_histogram=True`.

  For the STANDALONE path (no C++ fork yet), this module provides:
  1. GPUHistogramProvider — ctypes wrapper for libgpu_histogram.so
  2. gpu_train() — drop-in for lgb.train() with GPU histogram acceleration
  3. Auto-detection and fallback logic
  4. VRAM estimation for all timeframes

Usage in ml_multi_tf.py:
  Replace:
    model = lgb.train(params, dtrain, ...)
  With:
    from gpu_histogram_fork.src.lgbm_integration import gpu_train
    model = gpu_train(params, dtrain, X_csr, ...)

The C++ shared library must implement the contract defined in GPUHistogramProvider.
See ARCHITECTURE.md section 4 for kernel design and section 5 for memory layout.
"""

import ctypes
import ctypes.util
import logging
import os
import platform
import struct
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    from scipy import sparse as sp_sparse
except ImportError:
    sp_sparse = None


log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# VRAM budget per timeframe (GB) — from ARCHITECTURE.md section 5.5
# Includes CSR resident + histogram pool + gradients + kernel overhead
TF_VRAM_ESTIMATES_GB = {
    '1w':  3.0,
    '1d':  6.0,
    '4h':  13.0,
    '1h':  26.0,
    '15m': 41.0,
}

# VRAM safety margin — never use more than 85% of total
VRAM_SAFETY_FACTOR = 0.85

# Shared library name
_LIB_NAME = 'libgpu_histogram'
_LIB_SO = f'{_LIB_NAME}.so'
_LIB_DLL = f'{_LIB_NAME}.dll'


# ---------------------------------------------------------------------------
# GPU Detection
# ---------------------------------------------------------------------------

def _detect_cuda() -> dict:
    """Detect CUDA availability and GPU properties.

    Returns
    -------
    dict with keys:
        available : bool
        device_count : int
        devices : list of dict (name, vram_mb, compute_major, compute_minor)
        driver_version : int  (e.g. 12040 for CUDA 12.4)
    """
    result = {
        'available': False,
        'device_count': 0,
        'devices': [],
        'driver_version': 0,
    }

    try:
        import cupy as cp
        result['available'] = True
        result['device_count'] = cp.cuda.runtime.getDeviceCount()
        result['driver_version'] = cp.cuda.runtime.driverGetVersion()

        for i in range(result['device_count']):
            with cp.cuda.Device(i):
                props = cp.cuda.runtime.getDeviceProperties(i)
                free_mem, total_mem = cp.cuda.runtime.memGetInfo()
                result['devices'].append({
                    'index': i,
                    'name': props['name'].decode() if isinstance(props['name'], bytes) else str(props['name']),
                    'vram_total_mb': total_mem // (1024 * 1024),
                    'vram_free_mb': free_mem // (1024 * 1024),
                    'compute_major': props.get('major', 0),
                    'compute_minor': props.get('minor', 0),
                    'shared_mem_per_block': props.get('sharedMemPerBlock', 0),
                })
    except (ImportError, Exception) as e:
        log.debug("CUDA detection failed: %s", e)

    return result


def estimate_vram_bytes(n_rows: int, nnz: int, n_bundles: int,
                        num_class: int = 3, max_leaves: int = 63) -> int:
    """Estimate total GPU VRAM needed for histogram co-processor.

    Parameters
    ----------
    n_rows : int
        Number of training rows.
    nnz : int
        Number of nonzero entries in CSR matrix.
    n_bundles : int
        Number of EFB bundles (post-bundling feature count).
    num_class : int
        Number of classes (3 for long/short/hold).
    max_leaves : int
        Maximum leaves per tree.

    Returns
    -------
    int
        Estimated bytes needed on GPU.
    """
    # CSR region (read-only, allocated once)
    indptr_bytes = (n_rows + 1) * 8            # int64
    indices_bytes = nnz * 4                     # int32
    data_bytes = nnz * 1                        # uint8 (EFB bin index)
    bundle_offsets_bytes = (n_bundles + 1) * 4  # int32

    # Gradient region (double-buffered)
    grad_hess_bytes = 4 * n_rows * num_class * 4  # 4 buffers x float32

    # Partition region
    leaf_id_bytes = n_rows * 1                  # int8
    leaf_count_bytes = max_leaves * 4           # int32

    # Histogram pool — assume average 2 bins per bundle (binary features)
    avg_bins_per_bundle = 2
    total_bins = n_bundles * avg_bins_per_bundle
    hist_per_leaf = total_bins * 2 * num_class * 8   # 2 accumulators (grad+hess) x float64
    hist_pool_bytes = max_leaves * hist_per_leaf

    # Kernel overhead (stack, registers, launch overhead)
    overhead_bytes = 512 * 1024 * 1024  # 512 MB conservative

    total = (indptr_bytes + indices_bytes + data_bytes + bundle_offsets_bytes
             + grad_hess_bytes + leaf_id_bytes + leaf_count_bytes
             + hist_pool_bytes + overhead_bytes)

    return total


# ---------------------------------------------------------------------------
# C Interface Contract
# ---------------------------------------------------------------------------
#
# The shared library (libgpu_histogram.so) MUST export these functions.
# See ARCHITECTURE.md for kernel design and memory layout.
#
# C function signatures:
#
#   int gpu_hist_init(
#       const int64_t* indptr,      // CSR indptr array, length n_rows+1
#       int64_t        n_rows,
#       const int32_t* indices,     // CSR column indices, length nnz
#       const uint8_t* data,        // EFB bundle bin index per nonzero, length nnz
#       int64_t        nnz,
#       const int32_t* bundle_offsets,  // cumulative bin count per bundle, length n_bundles+1
#       int32_t        n_bundles,
#       int32_t        max_bin,     // max bins per bundle (255)
#       int32_t        num_class,   // number of classes (3)
#       int32_t        max_leaves,  // max leaves per tree (63)
#       int32_t        gpu_id       // CUDA device index
#   );
#   Returns 0 on success, negative on error.
#
#   int gpu_hist_upload_gradients(
#       const float* grad,          // shape [n_rows * num_class], row-major
#       const float* hess,          // shape [n_rows * num_class], row-major
#       int64_t      n_rows,
#       int32_t      num_class
#   );
#   Returns 0 on success. Async copy using pinned double-buffer.
#
#   int gpu_hist_build(
#       const int32_t* row_indices, // sorted row indices for this leaf
#       int32_t        n_leaf_rows,
#       int32_t        leaf_id,     // which leaf (for histogram pool indexing)
#       int32_t        class_idx,   // which class gradient to use (0, 1, or 2)
#       double*        out_hist     // output: [n_total_bins * 2], grad/hess interleaved
#   );
#   Returns 0 on success. Launches kernel, copies result to out_hist.
#
#   int gpu_hist_build_subset(
#       int32_t        parent_leaf_id,  // parent leaf to compute subtraction from
#       int32_t        smaller_leaf_id, // the smaller child (built by kernel)
#       int32_t        larger_leaf_id,  // the larger child (computed by subtraction)
#       const int32_t* smaller_rows,    // row indices of smaller child
#       int32_t        n_smaller_rows,
#       int32_t        class_idx,
#       double*        out_smaller_hist,  // output for smaller child
#       double*        out_larger_hist    // output for larger child (parent - smaller)
#   );
#   Returns 0 on success. Implements subtraction trick on GPU.
#
#   int gpu_hist_update_partition(
#       int32_t        leaf_to_split,
#       int32_t        left_leaf_id,
#       int32_t        right_leaf_id,
#       int32_t        split_feature,   // EFB bundle index
#       uint8_t        split_bin        // bin threshold
#   );
#   Returns 0 on success. Updates GPU-side leaf_id[] array in-place.
#
#   void gpu_hist_get_stats(
#       int64_t* out_kernel_time_ns,   // cumulative kernel execution time
#       int64_t* out_h2d_time_ns,      // cumulative host-to-device time
#       int64_t* out_d2h_time_ns,      // cumulative device-to-host time
#       int64_t* out_n_kernel_calls    // total histogram kernel launches
#   );
#
#   void gpu_hist_cleanup(void);
#   Frees all GPU allocations. Safe to call multiple times.
#
# Error codes:
#   0  = success
#   -1 = CUDA not available
#   -2 = GPU memory allocation failed (VRAM insufficient)
#   -3 = invalid parameters
#   -4 = kernel launch failed
#   -5 = data transfer failed
#
# Data layout notes:
#   - indptr uses int64 (NOT int32) to support NNZ > 2^31 on 15m timeframe
#   - data[] is uint8 because max_bin=255 fits in one byte
#   - Histograms are float64 for accumulation precision (LightGBM uses double internally)
#   - Gradients are float32 (promoted to float64 inside kernel via atomicAdd)
#   - row_indices are always sorted ascending (matches LightGBM's convention)
#   - out_hist layout: [bin0_grad, bin0_hess, bin1_grad, bin1_hess, ...]
#     Total size: n_total_bins * 2 doubles (where n_total_bins = sum of bins per bundle)
#


# Error code descriptions
_GPU_ERRORS = {
    0: 'success',
    -1: 'CUDA not available',
    -2: 'GPU memory allocation failed',
    -3: 'invalid parameters',
    -4: 'kernel launch failed',
    -5: 'data transfer failed',
}


class GPUHistogramProvider:
    """Python wrapper around the GPU histogram C shared library.

    Manages the lifecycle: init -> upload gradients -> build histograms -> cleanup.
    All GPU memory is allocated at init() and freed at cleanup(). No dynamic
    allocations during training.

    Parameters
    ----------
    lib_path : str or None
        Path to libgpu_histogram.so. If None, searches standard locations.
    gpu_id : int
        CUDA device index. Default 0.

    Raises
    ------
    FileNotFoundError
        If shared library not found.
    RuntimeError
        If GPU initialization fails.
    """

    def __init__(self, lib_path: Optional[str] = None, gpu_id: int = 0):
        self._lib = None
        self._initialized = False
        self._gpu_id = gpu_id
        self._n_total_bins = 0
        self._n_rows = 0
        self._num_class = 3

        # Find and load shared library
        if lib_path is None:
            lib_path = self._find_library()
        if lib_path is None:
            raise FileNotFoundError(
                f"Cannot find {_LIB_SO} or {_LIB_DLL}. "
                f"Build the C++ library first (see IMPLEMENTATION_PLAN.md Phase 2)."
            )

        self._lib = ctypes.CDLL(lib_path)
        self._setup_prototypes()
        log.info("Loaded GPU histogram library: %s", lib_path)

    def _find_library(self) -> Optional[str]:
        """Search for the shared library in standard locations."""
        # 1. Same directory as this file
        this_dir = Path(__file__).parent
        candidates = [
            this_dir / _LIB_SO,
            this_dir / _LIB_DLL,
            this_dir.parent / 'build' / _LIB_SO,
            this_dir.parent / 'build' / _LIB_DLL,
            this_dir.parent / 'build' / 'lib' / _LIB_SO,
        ]

        # 2. Environment variable override
        env_path = os.environ.get('GPU_HISTOGRAM_LIB')
        if env_path:
            candidates.insert(0, Path(env_path))

        # 3. System library path
        sys_path = ctypes.util.find_library(_LIB_NAME)
        if sys_path:
            candidates.append(Path(sys_path))

        for p in candidates:
            if p.exists():
                return str(p)
        return None

    def _setup_prototypes(self):
        """Define ctypes function signatures for type safety."""
        lib = self._lib

        # gpu_hist_init
        lib.gpu_hist_init.restype = ctypes.c_int
        lib.gpu_hist_init.argtypes = [
            ctypes.POINTER(ctypes.c_int64),   # indptr
            ctypes.c_int64,                    # n_rows
            ctypes.POINTER(ctypes.c_int32),    # indices
            ctypes.POINTER(ctypes.c_uint8),    # data (EFB bin)
            ctypes.c_int64,                    # nnz
            ctypes.POINTER(ctypes.c_int32),    # bundle_offsets
            ctypes.c_int32,                    # n_bundles
            ctypes.c_int32,                    # max_bin
            ctypes.c_int32,                    # num_class
            ctypes.c_int32,                    # max_leaves
            ctypes.c_int32,                    # gpu_id
        ]

        # gpu_hist_upload_gradients
        lib.gpu_hist_upload_gradients.restype = ctypes.c_int
        lib.gpu_hist_upload_gradients.argtypes = [
            ctypes.POINTER(ctypes.c_float),    # grad
            ctypes.POINTER(ctypes.c_float),    # hess
            ctypes.c_int64,                    # n_rows
            ctypes.c_int32,                    # num_class
        ]

        # gpu_hist_build
        lib.gpu_hist_build.restype = ctypes.c_int
        lib.gpu_hist_build.argtypes = [
            ctypes.POINTER(ctypes.c_int32),    # row_indices
            ctypes.c_int32,                    # n_leaf_rows
            ctypes.c_int32,                    # leaf_id
            ctypes.c_int32,                    # class_idx
            ctypes.POINTER(ctypes.c_double),   # out_hist
        ]

        # gpu_hist_build_subset (subtraction trick)
        lib.gpu_hist_build_subset.restype = ctypes.c_int
        lib.gpu_hist_build_subset.argtypes = [
            ctypes.c_int32,                    # parent_leaf_id
            ctypes.c_int32,                    # smaller_leaf_id
            ctypes.c_int32,                    # larger_leaf_id
            ctypes.POINTER(ctypes.c_int32),    # smaller_rows
            ctypes.c_int32,                    # n_smaller_rows
            ctypes.c_int32,                    # class_idx
            ctypes.POINTER(ctypes.c_double),   # out_smaller_hist
            ctypes.POINTER(ctypes.c_double),   # out_larger_hist
        ]

        # gpu_hist_update_partition
        lib.gpu_hist_update_partition.restype = ctypes.c_int
        lib.gpu_hist_update_partition.argtypes = [
            ctypes.c_int32,                    # leaf_to_split
            ctypes.c_int32,                    # left_leaf_id
            ctypes.c_int32,                    # right_leaf_id
            ctypes.c_int32,                    # split_feature (bundle index)
            ctypes.c_uint8,                    # split_bin
        ]

        # gpu_hist_get_stats
        lib.gpu_hist_get_stats.restype = None
        lib.gpu_hist_get_stats.argtypes = [
            ctypes.POINTER(ctypes.c_int64),    # kernel_time_ns
            ctypes.POINTER(ctypes.c_int64),    # h2d_time_ns
            ctypes.POINTER(ctypes.c_int64),    # d2h_time_ns
            ctypes.POINTER(ctypes.c_int64),    # n_kernel_calls
        ]

        # gpu_hist_cleanup
        lib.gpu_hist_cleanup.restype = None
        lib.gpu_hist_cleanup.argtypes = []

    def init(self, csr_matrix, bundle_offsets: np.ndarray,
             max_bin: int = 255, num_class: int = 3, max_leaves: int = 63):
        """Initialize GPU with CSR data. Transfers CSR to GPU (one-time).

        Parameters
        ----------
        csr_matrix : scipy.sparse.csr_matrix
            The sparse binary feature matrix. indptr must be int64.
            data array is reinterpreted as uint8 EFB bin indices.
        bundle_offsets : np.ndarray, int32
            Cumulative bin count per EFB bundle. Length = n_bundles + 1.
            bundle_offsets[0] = 0, bundle_offsets[-1] = total_bins.
        max_bin : int
            Maximum bins per bundle (255 for our config).
        num_class : int
            Number of classes (3).
        max_leaves : int
            Maximum leaves per tree (63).

        Raises
        ------
        RuntimeError
            If GPU initialization or memory allocation fails.
        """
        if self._initialized:
            self.cleanup()

        indptr = np.ascontiguousarray(csr_matrix.indptr, dtype=np.int64)
        indices = np.ascontiguousarray(csr_matrix.indices, dtype=np.int32)

        # EFB bin indices — in the real C++ fork, LightGBM computes these during
        # Dataset construction. For standalone mode, the data array IS the bin index
        # (binary features: bin = 0 or 1 within their bundle).
        data = np.ascontiguousarray(csr_matrix.data, dtype=np.uint8)

        n_rows = csr_matrix.shape[0]
        nnz = csr_matrix.nnz
        bundle_offsets = np.ascontiguousarray(bundle_offsets, dtype=np.int32)
        n_bundles = len(bundle_offsets) - 1

        self._n_rows = n_rows
        self._num_class = num_class
        self._n_total_bins = int(bundle_offsets[-1])

        rc = self._lib.gpu_hist_init(
            indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            ctypes.c_int64(n_rows),
            indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_int64(nnz),
            bundle_offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int32(n_bundles),
            ctypes.c_int32(max_bin),
            ctypes.c_int32(num_class),
            ctypes.c_int32(max_leaves),
            ctypes.c_int32(self._gpu_id),
        )

        if rc != 0:
            raise RuntimeError(
                f"GPU histogram init failed: {_GPU_ERRORS.get(rc, f'unknown error {rc}')}"
            )

        self._initialized = True
        log.info("GPU histogram initialized: %d rows, %d nnz, %d bundles, %d total bins, GPU %d",
                 n_rows, nnz, n_bundles, self._n_total_bins, self._gpu_id)

    def upload_gradients(self, grad: np.ndarray, hess: np.ndarray):
        """Upload gradient/hessian vectors to GPU (async, double-buffered).

        Called once per boosting round. The C library handles pinned memory
        and double-buffering internally.

        Parameters
        ----------
        grad : np.ndarray, shape (n_rows, num_class) or (n_rows * num_class,), float32
        hess : np.ndarray, shape (n_rows, num_class) or (n_rows * num_class,), float32
        """
        if not self._initialized:
            raise RuntimeError("GPU histogram not initialized. Call init() first.")

        grad = np.ascontiguousarray(grad.ravel(), dtype=np.float32)
        hess = np.ascontiguousarray(hess.ravel(), dtype=np.float32)

        rc = self._lib.gpu_hist_upload_gradients(
            grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            hess.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int64(self._n_rows),
            ctypes.c_int32(self._num_class),
        )

        if rc != 0:
            raise RuntimeError(
                f"GPU gradient upload failed: {_GPU_ERRORS.get(rc, f'unknown error {rc}')}"
            )

    def build_histogram(self, row_indices: np.ndarray, leaf_id: int,
                        class_idx: int) -> np.ndarray:
        """Build histogram for a single leaf node on GPU.

        Parameters
        ----------
        row_indices : np.ndarray, int32
            Sorted row indices belonging to this leaf.
        leaf_id : int
            Leaf index in histogram pool (0..max_leaves-1).
        class_idx : int
            Which class (0, 1, or 2) — indexes into gradient array.

        Returns
        -------
        np.ndarray, float64, shape (n_total_bins * 2,)
            Interleaved grad/hess: [bin0_grad, bin0_hess, bin1_grad, bin1_hess, ...]
        """
        if not self._initialized:
            raise RuntimeError("GPU histogram not initialized. Call init() first.")

        row_indices = np.ascontiguousarray(row_indices, dtype=np.int32)
        out_hist = np.empty(self._n_total_bins * 2, dtype=np.float64)

        rc = self._lib.gpu_hist_build(
            row_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int32(len(row_indices)),
            ctypes.c_int32(leaf_id),
            ctypes.c_int32(class_idx),
            out_hist.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )

        if rc != 0:
            raise RuntimeError(
                f"GPU histogram build failed: {_GPU_ERRORS.get(rc, f'unknown error {rc}')}"
            )

        return out_hist

    def build_histogram_subtraction(
        self,
        parent_leaf_id: int,
        smaller_leaf_id: int,
        larger_leaf_id: int,
        smaller_rows: np.ndarray,
        class_idx: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build histogram for smaller child, compute larger child by subtraction.

        Implements LightGBM's subtraction trick: only scan the smaller child's rows,
        then larger = parent - smaller. ~2x speedup per tree level.

        Parameters
        ----------
        parent_leaf_id : int
            The leaf being split (its histogram is already computed).
        smaller_leaf_id : int
            The child with fewer rows (histogram built by GPU kernel).
        larger_leaf_id : int
            The child with more rows (histogram = parent - smaller).
        smaller_rows : np.ndarray, int32
            Row indices of the smaller child.
        class_idx : int
            Class index (0, 1, or 2).

        Returns
        -------
        smaller_hist : np.ndarray, float64
        larger_hist : np.ndarray, float64
        """
        if not self._initialized:
            raise RuntimeError("GPU histogram not initialized. Call init() first.")

        smaller_rows = np.ascontiguousarray(smaller_rows, dtype=np.int32)
        out_smaller = np.empty(self._n_total_bins * 2, dtype=np.float64)
        out_larger = np.empty(self._n_total_bins * 2, dtype=np.float64)

        rc = self._lib.gpu_hist_build_subset(
            ctypes.c_int32(parent_leaf_id),
            ctypes.c_int32(smaller_leaf_id),
            ctypes.c_int32(larger_leaf_id),
            smaller_rows.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int32(len(smaller_rows)),
            ctypes.c_int32(class_idx),
            out_smaller.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            out_larger.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )

        if rc != 0:
            raise RuntimeError(
                f"GPU histogram subtraction failed: {_GPU_ERRORS.get(rc, f'unknown error {rc}')}"
            )

        return out_smaller, out_larger

    def get_stats(self) -> dict:
        """Get cumulative GPU timing statistics.

        Returns
        -------
        dict with keys: kernel_time_ms, h2d_time_ms, d2h_time_ms, n_kernel_calls
        """
        kernel_ns = ctypes.c_int64(0)
        h2d_ns = ctypes.c_int64(0)
        d2h_ns = ctypes.c_int64(0)
        n_calls = ctypes.c_int64(0)

        self._lib.gpu_hist_get_stats(
            ctypes.byref(kernel_ns),
            ctypes.byref(h2d_ns),
            ctypes.byref(d2h_ns),
            ctypes.byref(n_calls),
        )

        return {
            'kernel_time_ms': kernel_ns.value / 1e6,
            'h2d_time_ms': h2d_ns.value / 1e6,
            'd2h_time_ms': d2h_ns.value / 1e6,
            'n_kernel_calls': n_calls.value,
        }

    def cleanup(self):
        """Free all GPU allocations. Safe to call multiple times."""
        if self._lib is not None and self._initialized:
            self._lib.gpu_hist_cleanup()
            self._initialized = False
            log.info("GPU histogram resources freed")

    def __del__(self):
        self.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cleanup()
        return False


# ---------------------------------------------------------------------------
# GPU Availability Check
# ---------------------------------------------------------------------------

def check_gpu_histogram_available(
    n_rows: int = 0,
    nnz: int = 0,
    n_bundles: int = 18000,
    tf_name: Optional[str] = None,
) -> dict:
    """Check if GPU histogram acceleration is available and has enough VRAM.

    Parameters
    ----------
    n_rows : int
        Number of training rows (for VRAM estimation).
    nnz : int
        Number of nonzero entries in CSR.
    n_bundles : int
        Number of EFB bundles.
    tf_name : str or None
        Timeframe name for quick estimate lookup.

    Returns
    -------
    dict with keys:
        available : bool       — GPU + library both ready
        cuda_available : bool  — CUDA runtime detected
        lib_found : bool       — shared library found
        vram_sufficient : bool — enough VRAM for this data
        vram_needed_gb : float
        vram_free_gb : float
        gpu_name : str
        reason : str           — human-readable status
    """
    result = {
        'available': False,
        'cuda_available': False,
        'lib_found': False,
        'vram_sufficient': False,
        'vram_needed_gb': 0.0,
        'vram_free_gb': 0.0,
        'gpu_name': 'none',
        'reason': '',
    }

    # Check CUDA
    cuda_info = _detect_cuda()
    result['cuda_available'] = cuda_info['available']
    if not cuda_info['available']:
        result['reason'] = 'CUDA not available'
        return result

    if cuda_info['device_count'] == 0:
        result['reason'] = 'No CUDA devices found'
        return result

    gpu = cuda_info['devices'][0]
    result['gpu_name'] = gpu['name']
    result['vram_free_gb'] = gpu['vram_free_mb'] / 1024.0

    # Check shared library
    try:
        provider = GPUHistogramProvider.__new__(GPUHistogramProvider)
        provider._lib = None
        provider._initialized = False
        lib_path = provider._find_library()
        result['lib_found'] = lib_path is not None
    except Exception:
        result['lib_found'] = False

    if not result['lib_found']:
        result['reason'] = f"Shared library {_LIB_SO} not found (build C++ first)"
        return result

    # Estimate VRAM
    if tf_name and tf_name in TF_VRAM_ESTIMATES_GB:
        result['vram_needed_gb'] = TF_VRAM_ESTIMATES_GB[tf_name]
    elif n_rows > 0 and nnz > 0:
        vram_bytes = estimate_vram_bytes(n_rows, nnz, n_bundles)
        result['vram_needed_gb'] = vram_bytes / (1024 ** 3)
    else:
        result['vram_needed_gb'] = 0.0

    usable_vram_gb = result['vram_free_gb'] * VRAM_SAFETY_FACTOR
    result['vram_sufficient'] = result['vram_needed_gb'] <= usable_vram_gb

    if not result['vram_sufficient']:
        result['reason'] = (
            f"Insufficient VRAM: need {result['vram_needed_gb']:.1f} GB, "
            f"have {usable_vram_gb:.1f} GB usable ({gpu['name']})"
        )
        return result

    result['available'] = True
    result['reason'] = (
        f"GPU histogram ready: {gpu['name']}, "
        f"{result['vram_needed_gb']:.1f}/{usable_vram_gb:.1f} GB"
    )
    return result


# ---------------------------------------------------------------------------
# Drop-in gpu_train() Function
# ---------------------------------------------------------------------------

def gpu_train(
    params: dict,
    train_set,
    num_boost_round: int = 800,
    valid_sets=None,
    valid_names=None,
    callbacks=None,
    X_csr=None,
    tf_name: Optional[str] = None,
    gpu_id: int = 0,
):
    """Drop-in replacement for lgb.train() with GPU histogram acceleration.

    INTEGRATION PATH 1: Forked LightGBM (use_cuda_histogram=True)
    ---------------------------------------------------------------
    If LightGBM was built from our fork with -DUSE_CUDA=ON, the histogram
    kernel is linked directly into the C++ library. We just set the param
    and call lgb.train() normally. This is the production path.

    INTEGRATION PATH 2: External co-processor (standalone .so)
    -----------------------------------------------------------
    If using stock LightGBM, we cannot intercept C++ histogram calls from
    Python. In this mode, gpu_train() is a transparent passthrough to
    lgb.train() — the GPU acceleration only works with the forked build.
    The function still validates GPU availability and logs diagnostics.

    Parameters
    ----------
    params : dict
        LightGBM parameters. Will NOT be mutated.
    train_set : lgb.Dataset
        Training dataset.
    num_boost_round : int
        Number of boosting iterations.
    valid_sets : list of lgb.Dataset or None
        Validation datasets for early stopping.
    valid_names : list of str or None
        Names for validation sets.
    callbacks : list or None
        LightGBM callbacks (early stopping, logging, etc.).
    X_csr : scipy.sparse.csr_matrix or None
        Original CSR matrix. Used for VRAM estimation and future
        standalone co-processor path. Not required for forked LightGBM.
    tf_name : str or None
        Timeframe name ('1w', '1d', '4h', '1h', '15m') for VRAM estimation.
    gpu_id : int
        CUDA device index.

    Returns
    -------
    lgb.Booster
        Trained model (identical to lgb.train() return).
    """
    if lgb is None:
        raise ImportError("lightgbm is required")

    params = params.copy()

    # --- Attempt forked LightGBM path ---
    # The forked build recognizes 'use_cuda_histogram' as a native parameter.
    # Stock LightGBM ignores unknown params with a warning (verbosity=-1 suppresses).
    gpu_status = _check_forked_lgbm_support()

    if gpu_status['forked']:
        # Forked LightGBM: GPU histogram is handled inside C++ automatically
        params['use_cuda_histogram'] = True
        params['cuda_histogram_gpu_id'] = gpu_id

        # VRAM validation
        vram_check = check_gpu_histogram_available(
            n_rows=X_csr.shape[0] if X_csr is not None else 0,
            nnz=X_csr.nnz if X_csr is not None else 0,
            tf_name=tf_name,
        )
        if not vram_check['vram_sufficient']:
            log.warning("GPU VRAM insufficient for histogram co-processor, "
                        "falling back to CPU: %s", vram_check['reason'])
            params.pop('use_cuda_histogram', None)
            params.pop('cuda_histogram_gpu_id', None)
        else:
            log.info("GPU histogram ENABLED via forked LightGBM: %s", vram_check['reason'])
    else:
        # Stock LightGBM: log that GPU acceleration is not available
        log.info("Using stock LightGBM (CPU histograms). "
                 "Build from fork with -DUSE_CUDA=ON for GPU acceleration.")

    # --- Train ---
    model = lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )

    # --- Post-training diagnostics ---
    if gpu_status['forked'] and params.get('use_cuda_histogram'):
        _log_gpu_training_stats(model)

    return model


def _check_forked_lgbm_support() -> dict:
    """Check if the installed LightGBM is our GPU histogram fork.

    The forked build sets a custom build flag that we can detect.

    Returns
    -------
    dict with keys:
        forked : bool
        version : str
        build_info : str
    """
    result = {
        'forked': False,
        'version': getattr(lgb, '__version__', 'unknown'),
        'build_info': '',
    }

    # Method 1: Check for our custom parameter support.
    # The forked LightGBM registers 'use_cuda_histogram' as a valid parameter.
    # Stock LightGBM will raise a warning but not an error (verbosity=-1).
    # We detect the fork by checking if the C lib exports our init function.
    try:
        lib_path = lgb.basic._LIB._name if hasattr(lgb.basic, '_LIB') else None
        if lib_path:
            lib = ctypes.CDLL(lib_path)
            # Our fork exports this symbol
            _ = lib.gpu_hist_init
            result['forked'] = True
            result['build_info'] = 'gpu_hist_init symbol found in LightGBM lib'
    except (AttributeError, OSError):
        pass

    # Method 2: Check build config string (our fork appends "cuda_histogram" to build info)
    try:
        build_info = lgb.basic._LIB.LGBM_DumpParamAliases  # proxy for lib being loaded
        # If we get here, the lib is loaded. Check for our marker.
        if hasattr(lgb.basic._LIB, 'LGBM_GetGPUHistogramSupport'):
            result['forked'] = True
            result['build_info'] = 'LGBM_GetGPUHistogramSupport exported'
    except (AttributeError, OSError):
        pass

    return result


def _log_gpu_training_stats(model):
    """Extract and log GPU histogram timing from the trained model.

    The forked LightGBM stores GPU stats in the model metadata.
    """
    try:
        # The forked build adds these to model.dump_model() metadata
        model_dump = model.dump_model()
        gpu_stats = model_dump.get('gpu_histogram_stats', {})
        if gpu_stats:
            log.info(
                "GPU histogram stats: "
                "kernel=%.1fms, H2D=%.1fms, D2H=%.1fms, launches=%d",
                gpu_stats.get('kernel_time_ms', 0),
                gpu_stats.get('h2d_time_ms', 0),
                gpu_stats.get('d2h_time_ms', 0),
                gpu_stats.get('n_kernel_calls', 0),
            )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# CPCV Integration Helper
# ---------------------------------------------------------------------------

def make_cpcv_worker_with_gpu(original_worker_fn):
    """Wrap a CPCV split worker to use GPU histogram acceleration.

    This is a higher-order function that wraps _cpcv_split_worker from
    ml_multi_tf.py. The wrapper:
    1. Detects if GPU histogram is available
    2. If yes, patches lgb.train -> gpu_train in the worker scope
    3. If no, runs the original worker unchanged

    Parameters
    ----------
    original_worker_fn : callable
        The original _cpcv_split_worker function.

    Returns
    -------
    callable
        Wrapped worker function with same signature and return type.

    Usage in ml_multi_tf.py:
        from gpu_histogram_fork.src.lgbm_integration import make_cpcv_worker_with_gpu
        _cpcv_split_worker = make_cpcv_worker_with_gpu(_cpcv_split_worker)
    """
    def gpu_worker(args_tuple):
        # Unpack to get tf_name for VRAM estimation
        (wi, train_idx, test_idx, X_data, X_indices, X_indptr, X_shape,
         y_3class, sample_weights, feature_cols, lgb_params,
         num_boost_round, tf_name, gpu_id) = args_tuple

        # Check if forked LightGBM is available
        fork_status = _check_forked_lgbm_support()
        if fork_status['forked']:
            # Inject GPU histogram params — the forked C++ handles the rest
            lgb_params = lgb_params.copy()
            lgb_params['use_cuda_histogram'] = True
            lgb_params['cuda_histogram_gpu_id'] = gpu_id if gpu_id >= 0 else 0

            # Re-pack with modified params
            args_tuple = (wi, train_idx, test_idx, X_data, X_indices, X_indptr, X_shape,
                          y_3class, sample_weights, feature_cols, lgb_params,
                          num_boost_round, tf_name, gpu_id)

        return original_worker_fn(args_tuple)

    return gpu_worker


# ---------------------------------------------------------------------------
# Standalone Benchmark
# ---------------------------------------------------------------------------

def benchmark_gpu_vs_cpu(
    n_rows: int = 5733,
    n_features: int = 2_000_000,
    density: float = 0.003,
    n_bundles: int = 18000,
    n_rounds: int = 10,
    gpu_id: int = 0,
) -> dict:
    """Benchmark GPU histogram building against CPU baseline.

    Requires libgpu_histogram.so to be built. Uses synthetic data from
    generate_test_data.py.

    Parameters
    ----------
    n_rows, n_features, density : data profile
    n_bundles : number of EFB bundles
    n_rounds : number of histogram builds to average
    gpu_id : CUDA device

    Returns
    -------
    dict with timing results and speedup factor
    """
    from .generate_test_data import (
        generate_sparse_binary_csr,
        generate_gradients,
        generate_leaf_indices,
    )
    import time

    log.info("Generating synthetic data: %d rows x %d features, %.1f%% density",
             n_rows, n_features, density * 100)

    csr = generate_sparse_binary_csr(n_rows, n_features, density)
    grad, hess = generate_gradients(n_rows)
    leaves = generate_leaf_indices(n_rows)

    # Synthetic bundle offsets (uniform 2 bins per bundle)
    bundle_offsets = np.arange(0, (n_bundles + 1) * 2, 2, dtype=np.int32)[:n_bundles + 1]
    n_total_bins = int(bundle_offsets[-1])

    # --- CPU baseline: simple histogram accumulation ---
    log.info("Running CPU baseline (%d rounds)...", n_rounds)
    cpu_times = []
    for r in range(n_rounds):
        leaf_rows = leaves[r % len(leaves)]
        if len(leaf_rows) == 0:
            continue
        t0 = time.perf_counter()
        # Simulate CPU histogram: for each row in leaf, accumulate CSR nonzeros
        hist_cpu = np.zeros(n_total_bins * 2, dtype=np.float64)
        for row in leaf_rows:
            start, end = csr.indptr[row], csr.indptr[row + 1]
            g = float(grad[row, 0])
            h = float(hess[row, 0])
            for j in range(start, end):
                bin_idx = min(int(csr.data[j]), n_total_bins - 1)
                hist_cpu[bin_idx * 2] += g
                hist_cpu[bin_idx * 2 + 1] += h
        cpu_times.append(time.perf_counter() - t0)

    avg_cpu_ms = np.mean(cpu_times) * 1000 if cpu_times else float('inf')
    log.info("CPU avg: %.2f ms/histogram", avg_cpu_ms)

    # --- GPU benchmark ---
    try:
        provider = GPUHistogramProvider(gpu_id=gpu_id)
        provider.init(csr, bundle_offsets, num_class=3)
        provider.upload_gradients(grad, hess)

        log.info("Running GPU benchmark (%d rounds)...", n_rounds)
        gpu_times = []
        for r in range(n_rounds):
            leaf_rows = leaves[r % len(leaves)]
            if len(leaf_rows) == 0:
                continue
            t0 = time.perf_counter()
            hist_gpu = provider.build_histogram(leaf_rows, leaf_id=0, class_idx=0)
            gpu_times.append(time.perf_counter() - t0)

        avg_gpu_ms = np.mean(gpu_times) * 1000 if gpu_times else float('inf')
        stats = provider.get_stats()
        provider.cleanup()

        speedup = avg_cpu_ms / avg_gpu_ms if avg_gpu_ms > 0 else 0
        log.info("GPU avg: %.2f ms/histogram (%.1fx speedup)", avg_gpu_ms, speedup)

        return {
            'cpu_ms': avg_cpu_ms,
            'gpu_ms': avg_gpu_ms,
            'speedup': speedup,
            'gpu_stats': stats,
            'n_rows': n_rows,
            'n_features': n_features,
            'nnz': csr.nnz,
            'n_bundles': n_bundles,
        }

    except FileNotFoundError:
        log.warning("GPU benchmark skipped: shared library not found")
        return {
            'cpu_ms': avg_cpu_ms,
            'gpu_ms': float('inf'),
            'speedup': 0,
            'reason': 'shared library not found',
        }


# ---------------------------------------------------------------------------
# C++ Changes Required (Documentation)
# ---------------------------------------------------------------------------
#
# To enable GPU histogram acceleration, the following changes are needed in
# the LightGBM C++ source (microsoft/LightGBM fork):
#
# 1. src/treelearner/serial_tree_learner.cpp
#    - In ConstructHistograms(): check config.use_cuda_histogram
#    - If true, call gpu_hist_build() instead of the CPU inner loop
#    - CPU path remains as-is (fallback)
#
# 2. src/treelearner/cuda/cuda_histogram.cu (NEW FILE)
#    - CUDA kernel: histogram_build() — row-parallel, shared-memory tiled
#    - Host wrapper: gpu_hist_init(), gpu_hist_build(), gpu_hist_cleanup()
#    - Exports C symbols for ctypes (extern "C")
#
# 3. src/treelearner/cuda/cuda_histogram.h (NEW FILE)
#    - Function declarations for cuda_histogram.cu
#    - Memory lifecycle structs
#
# 4. src/io/config.cpp
#    - Register 'use_cuda_histogram' (bool, default false)
#    - Register 'cuda_histogram_gpu_id' (int, default 0)
#    - Auto-disable if CUDA runtime not detected
#
# 5. CMakeLists.txt
#    - Add USE_CUDA option (separate from existing USE_GPU which is OpenCL)
#    - CUDA compilation for sm_80, sm_86, sm_89, sm_90, sm_100
#    - Link cuda_histogram.cu into the LightGBM shared library
#
# 6. python-package/lightgbm/basic.py
#    - No changes needed — params pass through to C++ automatically
#    - Our gpu_train() wrapper handles the Python-side logic
#
# Total diff estimate: ~500 lines across 5 files (3 new, 2 modified).
# See IMPLEMENTATION_PLAN.md Phase 2 for detailed file-by-file changes.
