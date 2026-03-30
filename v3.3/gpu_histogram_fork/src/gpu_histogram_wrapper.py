"""
Python ctypes wrapper for libgpu_histogram.so — the custom CUDA histogram
co-processor for LightGBM sparse binary cross features.

Replaces LightGBM's CPU SerialTreeLearner::ConstructHistograms() with GPU
kernels. The CSR feature matrix stays GPU-resident; only gradients (tiny)
are transferred per boosting round. Histograms are built via row-parallel
CUDA kernels with shared-memory tiling and returned to the CPU for split
finding.

Matrix thesis: ALL features preserved. NO filtering. NO subsampling.
Sparse binary cross features ARE the edge. EFB bundling stays on CPU;
only the histogram accumulation loop moves to GPU.

C API contract (gpu_histogram.h):
    gpu_hist_init         — upload CSR to GPU, allocate buffers
    gpu_hist_update_grads — transfer new gradient/hessian vectors
    gpu_hist_build        — build histogram for one leaf + one class
    gpu_hist_subtract     — parent - child histogram subtraction on GPU
    gpu_hist_vram_usage   — query VRAM consumption
    gpu_hist_cleanup      — free all GPU memory
    gpu_hist_version      — library version string
    gpu_hist_cuda_available — check CUDA runtime reachability
"""

import atexit
import ctypes
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import numpy.ctypeslib as npct

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Library version this wrapper was written against.  gpu_hist_version() must
# return a string starting with this major.minor prefix or we refuse to load.
# ---------------------------------------------------------------------------
_REQUIRED_LIB_VERSION_PREFIX = "1."

# ---------------------------------------------------------------------------
# ctypes type aliases matching gpu_histogram.h
# ---------------------------------------------------------------------------
c_int32 = ctypes.c_int32
c_int64 = ctypes.c_int64
c_double = ctypes.c_double
c_float = ctypes.c_float
c_uint8 = ctypes.c_uint8
c_size_t = ctypes.c_size_t
c_char_p = ctypes.c_char_p
c_void_p = ctypes.c_void_p
c_bool = ctypes.c_bool

# Pointer types for numpy arrays
_PTR_INT32 = npct.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
_PTR_INT64 = npct.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")
_PTR_FLOAT64 = npct.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
_PTR_FLOAT32 = npct.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
_PTR_UINT8 = npct.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")


# ---------------------------------------------------------------------------
# Opaque handle type — the C library returns a void* context handle
# ---------------------------------------------------------------------------
class _GpuHistHandle(ctypes.c_void_p):
    """Opaque handle to GPU histogram context (allocated by C library)."""
    pass


# ---------------------------------------------------------------------------
# Return code constants (must match gpu_histogram.h enum)
# ---------------------------------------------------------------------------
GPU_HIST_OK = 0
GPU_HIST_ERR_NO_DEVICE = 1
GPU_HIST_ERR_OOM = 2
GPU_HIST_ERR_INVALID_ARG = 3
GPU_HIST_ERR_CUDA = 4
GPU_HIST_ERR_NOT_INIT = 5

_RC_MESSAGES = {
    GPU_HIST_OK: "OK",
    GPU_HIST_ERR_NO_DEVICE: "No CUDA-capable GPU found or device_id out of range",
    GPU_HIST_ERR_OOM: "GPU out of memory",
    GPU_HIST_ERR_INVALID_ARG: "Invalid argument",
    GPU_HIST_ERR_CUDA: "CUDA runtime error",
    GPU_HIST_ERR_NOT_INIT: "Not initialized (call init first)",
}


def _check_rc(rc: int, func_name: str) -> None:
    """Raise RuntimeError if C function returned a non-zero status."""
    if rc != GPU_HIST_OK:
        msg = _RC_MESSAGES.get(rc, f"Unknown error code {rc}")
        raise RuntimeError(f"gpu_histogram: {func_name}() failed — {msg} (rc={rc})")


# ---------------------------------------------------------------------------
# Library discovery
# ---------------------------------------------------------------------------
_LIB_NAME = "libgpu_histogram.so"


def _find_library() -> Optional[str]:
    """Search standard locations for libgpu_histogram.so.

    Search order:
        1. Same directory as this Python file
        2. ../lib  relative to this file
        3. sys.prefix/lib  (virtualenv / conda)
        4. Directories in LD_LIBRARY_PATH
        5. /usr/local/lib, /usr/lib

    Returns the first path where the library exists, or None.
    """
    candidates = []

    # 1. Same directory as this wrapper
    here = Path(__file__).resolve().parent
    candidates.append(here / _LIB_NAME)

    # 2. ../lib relative to this file
    candidates.append(here.parent / "lib" / _LIB_NAME)

    # 3. sys.prefix/lib
    candidates.append(Path(sys.prefix) / "lib" / _LIB_NAME)

    # 4. LD_LIBRARY_PATH
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    for d in ld_path.split(os.pathsep):
        if d:
            candidates.append(Path(d) / _LIB_NAME)

    # 5. Standard system paths
    candidates.append(Path("/usr/local/lib") / _LIB_NAME)
    candidates.append(Path("/usr/lib") / _LIB_NAME)

    for p in candidates:
        if p.is_file():
            return str(p)

    return None


def _load_library() -> ctypes.CDLL:
    """Load libgpu_histogram.so and configure function signatures.

    Raises RuntimeError with a clear message if the library cannot be found
    or the version does not match.
    """
    lib_path = _find_library()
    if lib_path is None:
        search_dirs = [
            str(Path(__file__).resolve().parent),
            str(Path(__file__).resolve().parent.parent / "lib"),
            str(Path(sys.prefix) / "lib"),
            "LD_LIBRARY_PATH directories",
            "/usr/local/lib",
            "/usr/lib",
        ]
        raise RuntimeError(
            f"Cannot find {_LIB_NAME}. Searched:\n"
            + "\n".join(f"  - {d}" for d in search_dirs)
            + "\n\nBuild the library first (see gpu_histogram_fork/README.md) "
            "or set LD_LIBRARY_PATH."
        )

    logger.info(f"Loading GPU histogram library from {lib_path}")
    lib = ctypes.CDLL(lib_path)

    # ------------------------------------------------------------------
    # Declare function signatures — must match gpu_histogram.h exactly
    # ------------------------------------------------------------------

    # const char* gpu_hist_version(void)
    lib.gpu_hist_version.argtypes = []
    lib.gpu_hist_version.restype = c_char_p

    # int gpu_hist_cuda_available(void)
    lib.gpu_hist_cuda_available.argtypes = []
    lib.gpu_hist_cuda_available.restype = c_int32

    # int gpu_hist_init(
    #     void** handle_out,
    #     const int64_t* indptr,   int64_t  n_rows_plus1,
    #     const int32_t* indices,  int64_t  nnz,
    #     const double*  data,     int64_t  nnz_data,  (same as nnz)
    #     int32_t n_features,
    #     int32_t device_id,
    #     int32_t use_fp64         (1 = float64 accumulation, 0 = float32)
    # )
    lib.gpu_hist_init.argtypes = [
        ctypes.POINTER(_GpuHistHandle),  # handle_out
        _PTR_INT64,                       # indptr
        c_int64,                          # n_rows_plus1
        _PTR_INT32,                       # indices
        c_int64,                          # nnz
        _PTR_FLOAT64,                     # data (float64 CSR values)
        c_int64,                          # nnz_data
        c_int32,                          # n_features
        c_int32,                          # device_id
        c_int32,                          # use_fp64
    ]
    lib.gpu_hist_init.restype = c_int32

    # int gpu_hist_update_grads(
    #     void* handle,
    #     const double* gradients,  int64_t n_grad,
    #     const double* hessians,   int64_t n_hess,
    #     int32_t num_class
    # )
    lib.gpu_hist_update_grads.argtypes = [
        _GpuHistHandle,   # handle
        _PTR_FLOAT64,     # gradients
        c_int64,          # n_grad
        _PTR_FLOAT64,     # hessians
        c_int64,          # n_hess
        c_int32,          # num_class
    ]
    lib.gpu_hist_update_grads.restype = c_int32

    # int gpu_hist_build(
    #     void* handle,
    #     const int32_t* row_indices,  int32_t n_rows,
    #     int32_t class_id,
    #     int32_t n_bins,
    #     double* out_hist,            (n_features * n_bins * 2) doubles
    #     int64_t out_hist_len
    # )
    lib.gpu_hist_build.argtypes = [
        _GpuHistHandle,   # handle
        _PTR_INT32,       # row_indices
        c_int32,          # n_rows
        c_int32,          # class_id
        c_int32,          # n_bins
        _PTR_FLOAT64,     # out_hist
        c_int64,          # out_hist_len
    ]
    lib.gpu_hist_build.restype = c_int32

    # int gpu_hist_subtract(
    #     void* handle,
    #     const double* parent_hist,  int64_t parent_len,
    #     const double* child_hist,   int64_t child_len,
    #     double* out_sibling,        int64_t out_len
    # )
    lib.gpu_hist_subtract.argtypes = [
        _GpuHistHandle,   # handle
        _PTR_FLOAT64,     # parent_hist
        c_int64,          # parent_len
        _PTR_FLOAT64,     # child_hist
        c_int64,          # child_len
        _PTR_FLOAT64,     # out_sibling
        c_int64,          # out_len
    ]
    lib.gpu_hist_subtract.restype = c_int32

    # int gpu_hist_vram_usage(
    #     void* handle,
    #     int64_t* used_bytes_out,
    #     int64_t* total_bytes_out
    # )
    lib.gpu_hist_vram_usage.argtypes = [
        _GpuHistHandle,                     # handle
        ctypes.POINTER(c_int64),            # used_bytes_out
        ctypes.POINTER(c_int64),            # total_bytes_out
    ]
    lib.gpu_hist_vram_usage.restype = c_int32

    # int gpu_hist_cleanup(void* handle)
    lib.gpu_hist_cleanup.argtypes = [_GpuHistHandle]
    lib.gpu_hist_cleanup.restype = c_int32

    # ------------------------------------------------------------------
    # Version check
    # ------------------------------------------------------------------
    raw_version = lib.gpu_hist_version()
    if raw_version is None:
        raise RuntimeError("gpu_hist_version() returned NULL")
    version_str = raw_version.decode("utf-8")
    if not version_str.startswith(_REQUIRED_LIB_VERSION_PREFIX):
        raise RuntimeError(
            f"Library version mismatch: got '{version_str}', "
            f"expected '{_REQUIRED_LIB_VERSION_PREFIX}*'. "
            f"Rebuild the library or update the wrapper."
        )
    logger.info(f"GPU histogram library version: {version_str}")

    return lib


# ---------------------------------------------------------------------------
# Singleton library handle (loaded lazily on first use)
# ---------------------------------------------------------------------------
_lib: Optional[ctypes.CDLL] = None


def _get_lib() -> ctypes.CDLL:
    """Return the loaded library, loading it on first call."""
    global _lib
    if _lib is None:
        _lib = _load_library()
    return _lib


# ---------------------------------------------------------------------------
# Main wrapper class
# ---------------------------------------------------------------------------

class GPUHistogramBuilder:
    """GPU-accelerated histogram builder for LightGBM sparse binary features.

    Uploads a scipy CSR matrix to the GPU once at construction. Subsequent
    calls to ``build_histogram()`` only transfer the small gradient/hessian
    vectors — the feature matrix stays GPU-resident.

    Designed for the matrix thesis feature pipeline: millions of sparse
    binary cross features (gematria x TA, astrology x TA, etc.) where
    histogram building is the training bottleneck.

    Parameters
    ----------
    csr_matrix : scipy.sparse.csr_matrix or csr_array
        Shape ``(n_rows, n_features)``.  The sparse feature matrix.
        Must have int64 indptr (required for NNZ > 2^31 on 15m data).
        Data array is converted to float64 for accumulation precision.
    device_id : int, default 0
        CUDA device ordinal.
    use_fp64 : bool, default True
        Use float64 for histogram accumulation.  True matches LightGBM's
        CPU precision exactly.  False uses float32 (saves VRAM, slight
        precision loss).

    Examples
    --------
    >>> import scipy.sparse as sp
    >>> csr = sp.random(5000, 2_000_000, density=0.003, format="csr")
    >>> with GPUHistogramBuilder(csr) as gpu:
    ...     gpu.update_gradients(grad, hess, num_class=3)
    ...     hist = gpu.build_histogram(leaf_row_indices, class_id=0)
    ...     print(hist.shape)  # (2_000_000, 2)

    Notes
    -----
    - Thread safety: NOT thread-safe.  Each thread needs its own builder.
    - The builder registers a SIGTERM handler to clean up GPU memory on
      unexpected termination (cloud preemption, kill signals).
    - Use the context manager (``with`` statement) to guarantee cleanup.
    """

    def __init__(
        self,
        csr_matrix,
        device_id: int = 0,
        use_fp64: bool = True,
    ):
        import scipy.sparse as sp

        self._lib = _get_lib()
        self._handle = _GpuHistHandle()
        self._cleaned_up = False
        self._device_id = device_id
        self._use_fp64 = use_fp64

        # ----- Validate and normalize CSR input -----
        if not sp.issparse(csr_matrix):
            raise TypeError(
                f"Expected scipy sparse matrix, got {type(csr_matrix).__name__}"
            )
        if not sp.isspmatrix_csr(csr_matrix):
            logger.info("Converting sparse matrix to CSR format")
            csr_matrix = csr_matrix.tocsr()

        self._n_rows, self._n_features = csr_matrix.shape
        self._nnz = csr_matrix.nnz

        # Ensure int64 indptr — critical for 15m data where NNZ > 2^31
        indptr = np.ascontiguousarray(csr_matrix.indptr, dtype=np.int64)

        # Ensure int32 indices (column indices fit in int32 for all TFs)
        if csr_matrix.indices.dtype != np.int32:
            if self._n_features > np.iinfo(np.int32).max:
                raise ValueError(
                    f"n_features={self._n_features:,} exceeds int32 max. "
                    "Column-partitioned multi-GPU required."
                )
            indices = np.ascontiguousarray(csr_matrix.indices, dtype=np.int32)
        else:
            indices = np.ascontiguousarray(csr_matrix.indices)

        # Data array in float64 for precise accumulation
        data = np.ascontiguousarray(csr_matrix.data, dtype=np.float64)

        # ----- Call C init -----
        logger.info(
            f"Initializing GPU histogram builder: "
            f"{self._n_rows:,} rows x {self._n_features:,} features, "
            f"NNZ={self._nnz:,}, device={device_id}, "
            f"fp64={use_fp64}"
        )

        rc = self._lib.gpu_hist_init(
            ctypes.byref(self._handle),
            indptr,
            c_int64(len(indptr)),
            indices,
            c_int64(self._nnz),
            data,
            c_int64(len(data)),
            c_int32(self._n_features),
            c_int32(device_id),
            c_int32(1 if use_fp64 else 0),
        )
        _check_rc(rc, "gpu_hist_init")

        # Register cleanup on process exit and SIGTERM
        atexit.register(self.cleanup)
        self._prev_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, self._sigterm_handler)

        used, total = self.get_vram_usage()
        logger.info(
            f"GPU histogram builder ready. "
            f"VRAM: {used / 1e9:.2f} GB used / {total / 1e9:.2f} GB total"
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_rows(self) -> int:
        """Number of rows in the feature matrix."""
        return self._n_rows

    @property
    def n_features(self) -> int:
        """Number of features (columns) in the feature matrix."""
        return self._n_features

    @property
    def nnz(self) -> int:
        """Number of non-zero entries in the CSR matrix."""
        return self._nnz

    @property
    def device_id(self) -> int:
        """CUDA device ordinal."""
        return self._device_id

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def update_gradients(
        self,
        gradients: np.ndarray,
        hessians: np.ndarray,
        num_class: int = 3,
    ) -> None:
        """Transfer new gradient and hessian vectors to the GPU.

        Called once per boosting round (before building histograms for
        that round's tree nodes).

        Parameters
        ----------
        gradients : np.ndarray, shape ``(n_rows,)`` or ``(n_rows * num_class,)``
            Gradient vector from the objective function.  For multiclass,
            the array is laid out as ``[row0_c0, row0_c1, row0_c2, row1_c0, ...]``
            (row-major interleaved by class).
        hessians : np.ndarray, same shape as gradients
            Hessian vector.
        num_class : int, default 3
            Number of classes (3 for long/short/hold).

        Raises
        ------
        RuntimeError
            If the C library returns an error (e.g., size mismatch).
        """
        self._assert_alive()

        gradients = np.ascontiguousarray(gradients, dtype=np.float64)
        hessians = np.ascontiguousarray(hessians, dtype=np.float64)

        expected_len = self._n_rows * num_class
        if gradients.size != expected_len:
            # Allow flat (n_rows,) for single-class or class-at-a-time
            if gradients.size != self._n_rows:
                raise ValueError(
                    f"gradients.size={gradients.size}, expected "
                    f"{expected_len} (n_rows*num_class) or "
                    f"{self._n_rows} (n_rows for single class)"
                )
            # Single-class mode: num_class=1 internally
            num_class = 1

        if hessians.size != gradients.size:
            raise ValueError(
                f"hessians.size={hessians.size} != gradients.size={gradients.size}"
            )

        rc = self._lib.gpu_hist_update_grads(
            self._handle,
            gradients.ravel(),
            c_int64(gradients.size),
            hessians.ravel(),
            c_int64(hessians.size),
            c_int32(num_class),
        )
        _check_rc(rc, "gpu_hist_update_grads")

    def build_histogram(
        self,
        row_indices: np.ndarray,
        class_id: int = 0,
        n_bins: Optional[int] = None,
    ) -> np.ndarray:
        """Build a gradient/hessian histogram for one tree leaf.

        This is the hot path — called up to 62 times per tree (minus
        subtraction-trick savings), 800 rounds x 3 classes.

        Parameters
        ----------
        row_indices : np.ndarray of int32
            Row indices belonging to this leaf node.
        class_id : int, default 0
            Which class's gradients to accumulate (0-indexed).
        n_bins : int or None
            Number of histogram bins per feature.  ``None`` defaults to 2
            (binary cross features: bin 0 = OFF, bin 1 = ON).

        Returns
        -------
        histogram : np.ndarray, shape ``(n_features, n_bins, 2)``
            Last axis: ``[gradient_sum, hessian_sum]`` per bin per feature.
            For binary features (n_bins=2):
              - ``[:, 0, :]`` = feature OFF (computed by subtraction)
              - ``[:, 1, :]`` = feature ON  (accumulated from CSR nonzeros)

        Raises
        ------
        RuntimeError
            If the GPU kernel fails (CUDA error, OOM, etc.).
        """
        self._assert_alive()

        if n_bins is None:
            n_bins = 2

        row_indices = np.ascontiguousarray(row_indices, dtype=np.int32)
        n_leaf_rows = len(row_indices)

        # Output buffer: n_features * n_bins * 2 (grad + hess per bin)
        out_len = self._n_features * n_bins * 2
        out_hist = np.empty(out_len, dtype=np.float64)

        rc = self._lib.gpu_hist_build(
            self._handle,
            row_indices,
            c_int32(n_leaf_rows),
            c_int32(class_id),
            c_int32(n_bins),
            out_hist,
            c_int64(out_len),
        )
        _check_rc(rc, "gpu_hist_build")

        # Reshape to (n_features, n_bins, 2)
        return out_hist.reshape(self._n_features, n_bins, 2)

    def subtract(
        self,
        parent_hist: np.ndarray,
        child_hist: np.ndarray,
    ) -> np.ndarray:
        """Compute sibling histogram via GPU-side subtraction.

        Implements LightGBM's histogram subtraction trick:
        ``sibling_hist = parent_hist - child_hist``.

        Done on GPU to avoid a CPU round-trip for large histogram arrays.

        Parameters
        ----------
        parent_hist : np.ndarray, shape ``(n_features, n_bins, 2)``
            The parent node's histogram.
        child_hist : np.ndarray, shape ``(n_features, n_bins, 2)``
            The smaller child's histogram (built by ``build_histogram``).

        Returns
        -------
        sibling_hist : np.ndarray, same shape as parent_hist
            The larger child's histogram.
        """
        self._assert_alive()

        parent_flat = np.ascontiguousarray(parent_hist.ravel(), dtype=np.float64)
        child_flat = np.ascontiguousarray(child_hist.ravel(), dtype=np.float64)

        if parent_flat.size != child_flat.size:
            raise ValueError(
                f"Shape mismatch: parent has {parent_flat.size} elements, "
                f"child has {child_flat.size}"
            )

        out_sibling = np.empty_like(parent_flat)

        rc = self._lib.gpu_hist_subtract(
            self._handle,
            parent_flat,
            c_int64(parent_flat.size),
            child_flat,
            c_int64(child_flat.size),
            out_sibling,
            c_int64(out_sibling.size),
        )
        _check_rc(rc, "gpu_hist_subtract")

        return out_sibling.reshape(parent_hist.shape)

    def get_vram_usage(self) -> Tuple[int, int]:
        """Query GPU VRAM usage.

        Returns
        -------
        used_bytes : int
            Bytes currently allocated by this builder on the GPU.
        total_bytes : int
            Total VRAM on the device.
        """
        self._assert_alive()

        used = c_int64(0)
        total = c_int64(0)

        rc = self._lib.gpu_hist_vram_usage(
            self._handle,
            ctypes.byref(used),
            ctypes.byref(total),
        )
        _check_rc(rc, "gpu_hist_vram_usage")

        return int(used.value), int(total.value)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Free all GPU memory held by this builder.

        Safe to call multiple times.  Called automatically by ``__del__``,
        ``atexit``, and the SIGTERM handler.
        """
        if self._cleaned_up:
            return
        if self._handle.value is None:
            self._cleaned_up = True
            return

        try:
            rc = self._lib.gpu_hist_cleanup(self._handle)
            if rc != GPU_HIST_OK:
                logger.warning(
                    f"gpu_hist_cleanup returned rc={rc} "
                    f"({_RC_MESSAGES.get(rc, 'unknown')})"
                )
        except Exception as e:
            logger.warning(f"Exception during GPU cleanup: {e}")
        finally:
            self._cleaned_up = True
            self._handle = _GpuHistHandle()  # null out
            logger.info("GPU histogram builder cleaned up")

    def __del__(self):
        """Release GPU memory on garbage collection."""
        self.cleanup()

    def __enter__(self):
        """Context manager entry — returns self."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit — releases GPU memory."""
        self.cleanup()
        return False  # do not suppress exceptions

    def __repr__(self) -> str:
        status = "active" if not self._cleaned_up else "cleaned up"
        return (
            f"GPUHistogramBuilder("
            f"rows={self._n_rows:,}, "
            f"features={self._n_features:,}, "
            f"nnz={self._nnz:,}, "
            f"device={self._device_id}, "
            f"fp64={self._use_fp64}, "
            f"status={status})"
        )

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    def _sigterm_handler(self, signum, frame):
        """Clean up GPU memory on SIGTERM (cloud preemption, kill)."""
        logger.warning("SIGTERM received — cleaning up GPU memory")
        self.cleanup()
        # Re-raise via previous handler if there was one
        if callable(self._prev_sigterm):
            self._prev_sigterm(signum, frame)
        else:
            sys.exit(128 + signum)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_alive(self) -> None:
        """Raise if cleanup() has already been called."""
        if self._cleaned_up or self._handle.value is None:
            raise RuntimeError(
                "GPUHistogramBuilder has been cleaned up. "
                "Create a new instance to continue."
            )

    # ------------------------------------------------------------------
    # Static utility
    # ------------------------------------------------------------------

    @staticmethod
    def is_gpu_available() -> bool:
        """Check if CUDA is reachable and libgpu_histogram.so is loadable.

        Returns True only if both conditions are met:
        1. The shared library can be found and loaded.
        2. The library reports CUDA is available (driver + device present).

        This does NOT allocate any GPU memory.
        """
        try:
            lib = _get_lib()
        except (RuntimeError, OSError):
            return False

        try:
            return bool(lib.gpu_hist_cuda_available())
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def is_gpu_available() -> bool:
    """Module-level shortcut for ``GPUHistogramBuilder.is_gpu_available()``."""
    return GPUHistogramBuilder.is_gpu_available()
