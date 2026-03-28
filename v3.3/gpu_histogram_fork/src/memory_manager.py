"""
GPU memory manager for the histogram builder.

Handles CSR upload, gradient double-buffering, histogram allocation/pooling,
and cleanup. Designed for sparse CSR matrices ranging from 2-40GB depending
on timeframe. Auto-detects VRAM and provides graceful fallback when data
doesn't fit.

Works on any CUDA GPU: RTX 3090 (24GB) through B200 (192GB).

Requires: CuPy with CUDA support.
"""

import atexit
import logging
import signal
import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.sparse as sp

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

logger = logging.getLogger(__name__)

# Reserve 15% of VRAM as headroom — never allocate into the last 15%
_VRAM_HEADROOM = 0.85


@dataclass
class CSRHandle:
    """Reference to a CSR matrix uploaded to GPU."""
    indptr: "cp.ndarray"    # int64
    indices: "cp.ndarray"   # int32
    data: "cp.ndarray"      # float64
    n_rows: int
    n_cols: int
    nnz: int
    bytes_used: int


@dataclass
class GradientBuffers:
    """Double-buffered pinned host + GPU gradient arrays."""
    # Two host-side pinned buffers (A/B) for async overlap
    host_a: np.ndarray
    host_b: np.ndarray
    # GPU-side buffer (current active)
    device: "cp.ndarray"
    # Which host buffer is "front" (being filled by CPU)
    front: int = 0  # 0 = A is front, 1 = B is front
    n_rows: int = 0
    num_class: int = 3


class GPUMemoryManager:
    """Manages all GPU memory for the histogram co-processor.

    Thread-safe: all public methods acquire _lock before touching GPU state.
    Registers SIGTERM cleanup handler and atexit hook.

    Usage:
        mgr = GPUMemoryManager(device_id=0)
        handle = mgr.upload_csr(scipy_csr)
        mgr.allocate_gradient_buffers(n_rows, num_class=3)
        mgr.allocate_histogram_pool(n_bins=23000, max_leaves=63)
        ...
        mgr.cleanup()
    """

    def __init__(self, device_id: int = 0):
        if not CUDA_AVAILABLE:
            raise RuntimeError(
                "CuPy not available — cannot initialize GPUMemoryManager. "
                "Install cupy-cuda12x or equivalent."
            )

        self._lock = threading.Lock()
        self._device_id = device_id
        self._device = cp.cuda.Device(device_id)

        with self._device:
            props = cp.cuda.runtime.getDeviceProperties(device_id)
            self._gpu_name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
            self._vram_total = props["totalGlobalMem"]
            self._compute_capability = (props["major"], props["minor"])

        logger.info(
            "GPUMemoryManager: device=%d name=%s VRAM=%.1fGB CC=%d.%d",
            device_id, self._gpu_name,
            self._vram_total / (1024 ** 3),
            *self._compute_capability,
        )

        # Tracked allocations
        self._csr_handle: Optional[CSRHandle] = None
        self._grad_buffers: Optional[GradientBuffers] = None
        self._hist_pool: list = []          # list of cp.ndarray (GPU histogram buffers)
        self._hist_host_pin: Optional[np.ndarray] = None  # pinned host for D2H
        self._hist_n_bins: int = 0
        self._streams: list = []
        self._cleaned_up = False

        # Register cleanup
        self._prev_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, self._sigterm_handler)
        atexit.register(self.cleanup)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def detect_gpus() -> list[dict]:
        """List all CUDA devices with name, VRAM, and compute capability.

        Returns list of dicts: [{"id": 0, "name": ..., "vram_bytes": ...,
                                  "vram_gb": ..., "compute_capability": (M, m)}, ...]
        """
        if not CUDA_AVAILABLE:
            return []
        gpus = []
        n = cp.cuda.runtime.getDeviceCount()
        for i in range(n):
            props = cp.cuda.runtime.getDeviceProperties(i)
            name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
            vram = props["totalGlobalMem"]
            gpus.append({
                "id": i,
                "name": name,
                "vram_bytes": vram,
                "vram_gb": round(vram / (1024 ** 3), 1),
                "compute_capability": (props["major"], props["minor"]),
            })
        return gpus

    # ------------------------------------------------------------------
    # VRAM queries
    # ------------------------------------------------------------------

    def get_vram_info(self) -> tuple[int, int, int]:
        """Return (used, free, total) VRAM in bytes."""
        with self._lock:
            with self._device:
                free, total = cp.cuda.runtime.memGetInfo()
                used = total - free
                return used, free, total

    def estimate_csr_footprint(self, n_rows: int, nnz: int) -> int:
        """Estimate GPU bytes needed for a CSR matrix.

        indptr: (n_rows+1) * 8  (int64)
        indices: nnz * 4        (int32)
        data: nnz * 8           (float64)
        overhead: ~16MB for CuPy internals
        """
        indptr_bytes = (n_rows + 1) * 8
        indices_bytes = nnz * 4
        data_bytes = nnz * 8
        overhead = 16 * 1024 * 1024  # 16MB
        return indptr_bytes + indices_bytes + data_bytes + overhead

    def can_fit(self, n_rows: int, nnz: int) -> bool:
        """Check if a CSR matrix fits in 85% of free VRAM."""
        needed = self.estimate_csr_footprint(n_rows, nnz)
        _, free, _ = self.get_vram_info()
        fits = needed <= int(free * _VRAM_HEADROOM)
        if not fits:
            logger.warning(
                "CSR won't fit: need %.2fGB, free %.2fGB (%.0f%% headroom = %.2fGB usable)",
                needed / (1024 ** 3), free / (1024 ** 3),
                _VRAM_HEADROOM * 100, free * _VRAM_HEADROOM / (1024 ** 3),
            )
        return fits

    # ------------------------------------------------------------------
    # CSR upload
    # ------------------------------------------------------------------

    def upload_csr(self, scipy_csr: sp.csr_matrix) -> CSRHandle:
        """Upload a scipy CSR matrix to GPU memory.

        Validates dtypes (int64 indptr, int32 indices) and checks VRAM
        before uploading. Returns a CSRHandle for kernel use.

        Raises ValueError if dtypes are wrong, RuntimeError if VRAM insufficient.
        """
        with self._lock:
            return self._upload_csr_locked(scipy_csr)

    def _upload_csr_locked(self, scipy_csr: sp.csr_matrix) -> CSRHandle:
        # Free previous CSR if any
        if self._csr_handle is not None:
            self._free_csr_locked()

        n_rows, n_cols = scipy_csr.shape
        nnz = scipy_csr.nnz

        # Validate dtypes
        if scipy_csr.indptr.dtype != np.int64:
            raise ValueError(
                f"indptr must be int64, got {scipy_csr.indptr.dtype}. "
                "15m+ timeframes need int64 for NNZ > 2^31."
            )
        if scipy_csr.indices.dtype != np.int32:
            raise ValueError(
                f"indices must be int32, got {scipy_csr.indices.dtype}."
            )

        # Check VRAM
        needed = self.estimate_csr_footprint(n_rows, nnz)
        _, free, total = self.get_vram_info()
        usable = int(free * _VRAM_HEADROOM)
        if needed > usable:
            raise RuntimeError(
                f"CSR needs {needed / (1024**3):.2f}GB but only "
                f"{usable / (1024**3):.2f}GB usable VRAM "
                f"({free / (1024**3):.2f}GB free, {_VRAM_HEADROOM:.0%} headroom). "
                f"GPU: {self._gpu_name} ({total / (1024**3):.1f}GB total)."
            )

        # Upload
        with self._device:
            gpu_indptr = cp.asarray(scipy_csr.indptr)       # already int64
            gpu_indices = cp.asarray(scipy_csr.indices)      # already int32
            gpu_data = cp.asarray(scipy_csr.data.astype(np.float64, copy=False))

        handle = CSRHandle(
            indptr=gpu_indptr,
            indices=gpu_indices,
            data=gpu_data,
            n_rows=n_rows,
            n_cols=n_cols,
            nnz=nnz,
            bytes_used=needed,
        )
        self._csr_handle = handle

        used_after, free_after, _ = self.get_vram_info()
        logger.info(
            "CSR uploaded: %d rows x %d cols, nnz=%s, %.2fGB used. "
            "VRAM: %.2fGB used / %.2fGB free",
            n_rows, n_cols, f"{nnz:,}",
            needed / (1024 ** 3),
            used_after / (1024 ** 3), free_after / (1024 ** 3),
        )
        return handle

    def _free_csr_locked(self):
        if self._csr_handle is not None:
            del self._csr_handle.indptr
            del self._csr_handle.indices
            del self._csr_handle.data
            self._csr_handle = None

    # ------------------------------------------------------------------
    # Gradient double-buffering
    # ------------------------------------------------------------------

    def allocate_gradient_buffers(self, n_rows: int, num_class: int = 3) -> GradientBuffers:
        """Allocate double-buffered pinned memory for gradient transfer.

        Two pinned host buffers (A and B) allow CPU to fill one while GPU
        reads from the other. GPU buffer receives the active copy.

        Each buffer shape: (n_rows, num_class * 2)  — grad + hess interleaved.
        """
        with self._lock:
            return self._alloc_grad_locked(n_rows, num_class)

    def _alloc_grad_locked(self, n_rows: int, num_class: int) -> GradientBuffers:
        if self._grad_buffers is not None:
            self._free_grad_locked()

        # Each buffer holds both gradients and hessians: (n_rows, num_class * 2)
        buf_shape = (n_rows, num_class * 2)

        with self._device:
            # Pinned host memory for async H2D
            host_a = cp.cuda.alloc_pinned_memory(
                int(np.prod(buf_shape) * 8)  # float64
            )
            host_b = cp.cuda.alloc_pinned_memory(
                int(np.prod(buf_shape) * 8)
            )
            # Wrap as numpy arrays over pinned memory
            np_a = np.frombuffer(host_a, dtype=np.float64).reshape(buf_shape)
            np_b = np.frombuffer(host_b, dtype=np.float64).reshape(buf_shape)

            # GPU buffer
            gpu_buf = cp.zeros(buf_shape, dtype=np.float64)

        bufs = GradientBuffers(
            host_a=np_a,
            host_b=np_b,
            device=gpu_buf,
            front=0,
            n_rows=n_rows,
            num_class=num_class,
        )
        # Keep refs to pinned memory so they aren't GC'd
        bufs._pin_a = host_a
        bufs._pin_b = host_b
        self._grad_buffers = bufs

        bytes_used = int(np.prod(buf_shape) * 8 * 3)  # 2 host + 1 device
        logger.info(
            "Gradient buffers allocated: n_rows=%d num_class=%d "
            "shape=%s (%.1fMB total, double-buffered + GPU)",
            n_rows, num_class, buf_shape,
            bytes_used / (1024 ** 2),
        )
        return bufs

    def _free_grad_locked(self):
        if self._grad_buffers is not None:
            del self._grad_buffers.device
            # Pinned memory freed when refs dropped
            self._grad_buffers = None

    def update_gradients(
        self,
        gradients: np.ndarray,
        hessians: np.ndarray,
        stream: Optional["cp.cuda.Stream"] = None,
    ):
        """Async copy gradients + hessians to GPU, swapping double buffer.

        Args:
            gradients: (n_rows,) or (n_rows, num_class) float64
            hessians: same shape as gradients
            stream: CUDA stream for async copy. If None, uses default stream.
        """
        with self._lock:
            self._update_grad_locked(gradients, hessians, stream)

    def _update_grad_locked(self, gradients, hessians, stream):
        bufs = self._grad_buffers
        if bufs is None:
            raise RuntimeError("Gradient buffers not allocated. Call allocate_gradient_buffers first.")

        # Select the back buffer (not being read by GPU)
        back = bufs.host_b if bufs.front == 0 else bufs.host_a
        nc = bufs.num_class

        # Fill the back buffer
        if gradients.ndim == 1:
            back[:, 0] = gradients
            back[:, nc] = hessians
        else:
            back[:, :nc] = gradients
            back[:, nc:] = hessians

        # Async copy to GPU
        with self._device:
            s = stream or cp.cuda.get_current_stream()
            bufs.device.set(cp.asarray(back), stream=s)

        # Swap front/back
        bufs.front = 1 - bufs.front

    # ------------------------------------------------------------------
    # Histogram allocation
    # ------------------------------------------------------------------

    def allocate_histogram(self, n_bins: int) -> tuple:
        """Allocate a single histogram output on GPU + pinned host for D2H.

        Returns (gpu_hist_grad, gpu_hist_hess, host_pinned_grad, host_pinned_hess).
        Each is shape (n_bins,) float64.
        """
        with self._lock:
            return self._alloc_hist_locked(n_bins)

    def _alloc_hist_locked(self, n_bins: int):
        with self._device:
            gpu_g = cp.zeros(n_bins, dtype=np.float64)
            gpu_h = cp.zeros(n_bins, dtype=np.float64)

            pin_g_mem = cp.cuda.alloc_pinned_memory(n_bins * 8)
            pin_h_mem = cp.cuda.alloc_pinned_memory(n_bins * 8)
            np_g = np.frombuffer(pin_g_mem, dtype=np.float64)
            np_h = np.frombuffer(pin_h_mem, dtype=np.float64)

        bytes_used = n_bins * 8 * 4  # 2 GPU + 2 host pinned
        logger.info(
            "Histogram allocated: n_bins=%d (%.1fMB GPU + %.1fMB pinned host)",
            n_bins, n_bins * 8 * 2 / (1024 ** 2),
            n_bins * 8 * 2 / (1024 ** 2),
        )
        # Store pinned memory refs to prevent GC
        self._hist_host_pin = (pin_g_mem, pin_h_mem, np_g, np_h)
        self._hist_n_bins = n_bins
        return gpu_g, gpu_h, np_g, np_h

    def allocate_histogram_pool(
        self, n_bins: int, max_leaves: int = 63
    ) -> list[tuple]:
        """Allocate a pool of histogram buffers for subtraction trick.

        One (grad, hess) pair per potential leaf. The subtraction trick
        reuses parent histogram: child_small = parent - child_large,
        so we need at most max_leaves buffers.

        Returns list of (gpu_hist_grad, gpu_hist_hess) tuples.
        """
        with self._lock:
            return self._alloc_pool_locked(n_bins, max_leaves)

    def _alloc_pool_locked(self, n_bins: int, max_leaves: int):
        # Free previous pool
        self._free_pool_locked()

        pool = []
        with self._device:
            for _ in range(max_leaves):
                gpu_g = cp.zeros(n_bins, dtype=np.float64)
                gpu_h = cp.zeros(n_bins, dtype=np.float64)
                pool.append((gpu_g, gpu_h))

        # Also allocate pinned host buffer for D2H (shared, reused)
        with self._device:
            pin_g_mem = cp.cuda.alloc_pinned_memory(n_bins * 8)
            pin_h_mem = cp.cuda.alloc_pinned_memory(n_bins * 8)
            np_g = np.frombuffer(pin_g_mem, dtype=np.float64)
            np_h = np.frombuffer(pin_h_mem, dtype=np.float64)

        self._hist_pool = pool
        self._hist_host_pin = (pin_g_mem, pin_h_mem, np_g, np_h)
        self._hist_n_bins = n_bins

        bytes_gpu = max_leaves * n_bins * 8 * 2
        bytes_pin = n_bins * 8 * 2
        _, free, _ = self.get_vram_info()
        logger.info(
            "Histogram pool allocated: %d leaves x %d bins = %.1fMB GPU + %.1fMB pinned. "
            "VRAM free: %.2fGB",
            max_leaves, n_bins,
            bytes_gpu / (1024 ** 2), bytes_pin / (1024 ** 2),
            free / (1024 ** 3),
        )
        return pool

    def _free_pool_locked(self):
        for pair in self._hist_pool:
            del pair
        self._hist_pool.clear()
        self._hist_host_pin = None

    # ------------------------------------------------------------------
    # Histogram result retrieval
    # ------------------------------------------------------------------

    def get_histogram_result(
        self,
        gpu_hist_grad: "cp.ndarray",
        gpu_hist_hess: "cp.ndarray",
        stream: Optional["cp.cuda.Stream"] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Async copy histogram from GPU to pinned host and return numpy arrays.

        Uses the shared pinned host buffer from allocate_histogram or
        allocate_histogram_pool.

        Args:
            gpu_hist_grad: GPU histogram grad buffer (n_bins,)
            gpu_hist_hess: GPU histogram hess buffer (n_bins,)
            stream: CUDA stream. If None, uses default (synchronous).

        Returns:
            (hist_grad, hist_hess) as numpy float64 arrays.
        """
        with self._lock:
            return self._get_hist_locked(gpu_hist_grad, gpu_hist_hess, stream)

    def _get_hist_locked(self, gpu_g, gpu_h, stream):
        if self._hist_host_pin is None:
            raise RuntimeError(
                "No pinned host buffer. Call allocate_histogram or "
                "allocate_histogram_pool first."
            )
        _, _, np_g, np_h = self._hist_host_pin

        with self._device:
            s = stream or cp.cuda.get_current_stream()
            # Async D2H into pinned memory
            gpu_g.get(out=np_g, stream=s)
            gpu_h.get(out=np_h, stream=s)
            s.synchronize()

        # Return copies so caller owns the data (pinned buffer gets reused)
        return np_g.copy(), np_h.copy()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        """Free all GPU allocations. Safe to call multiple times."""
        with self._lock:
            if self._cleaned_up:
                return
            self._cleaned_up = True

            logger.info("GPUMemoryManager: cleaning up all allocations")

            self._free_csr_locked()
            self._free_grad_locked()
            self._free_pool_locked()

            # Free any streams
            self._streams.clear()

            # Force CuPy memory pool to release
            with self._device:
                pool = cp.get_default_memory_pool()
                pool.free_all_blocks()
                pinned_pool = cp.get_default_pinned_memory_pool()
                pinned_pool.free_all_blocks()

            logger.info("GPUMemoryManager: cleanup complete")

    def _sigterm_handler(self, signum, frame):
        """Handle SIGTERM: clean up GPU memory, then call previous handler."""
        logger.warning("GPUMemoryManager: SIGTERM received, cleaning up GPU memory")
        self.cleanup()
        # Chain to previous handler
        if callable(self._prev_sigterm):
            self._prev_sigterm(signum, frame)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def gpu_name(self) -> str:
        return self._gpu_name

    @property
    def vram_total(self) -> int:
        return self._vram_total

    @property
    def compute_capability(self) -> tuple[int, int]:
        return self._compute_capability

    @property
    def device_id(self) -> int:
        return self._device_id

    @property
    def csr_handle(self) -> Optional[CSRHandle]:
        return self._csr_handle
