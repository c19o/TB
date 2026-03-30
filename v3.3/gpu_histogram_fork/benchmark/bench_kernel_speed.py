#!/usr/bin/env python3
"""
Micro-benchmark: CPU vs GPU histogram building speed.

Compares three approaches for the core LightGBM histogram operation
(scatter-add gradients into per-bundle bins for rows in a leaf):

  1. CPU Numba @njit reference
  2. GPU cuSPARSE SpMV (sparse matrix-vector multiply to accumulate)
  3. GPU atomic scatter kernel (custom CuPy RawKernel)

Profiles match real v3.3 training dimensions (VRAM-adjusted for RTX 3090):
  1w:  818 rows,    500K features
  1d:  5,733 rows,  2M features
  4h:  17,520 rows, 3M features
  1h:  75,405 rows, 5M features  (reduced from 8M for 24GB VRAM)
  15m: 100,000 rows, 5M features (reduced from 10M for 24GB VRAM)

Usage:
  python bench_kernel_speed.py                        # all profiles
  python bench_kernel_speed.py --profile 1w 1d        # specific profiles
  python bench_kernel_speed.py --scaling-features      # feature scaling test
  python bench_kernel_speed.py --scaling-rows           # row scaling test
  python bench_kernel_speed.py --scaling-leaves         # leaf size scaling test
  python bench_kernel_speed.py --output results.json   # save JSON results

Requires: numpy, scipy, numba, cupy (CUDA GPU)
"""

import argparse
import gc
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from scipy import sparse

# ---------------------------------------------------------------------------
# Lazy imports for GPU (fail gracefully if no GPU)
# ---------------------------------------------------------------------------
_cp = None
_cupyx = None
_GPU_AVAILABLE = False
_GPU_NAME = "N/A"
_GPU_VRAM_BYTES = 0
_GPU_MEM_BW_GBS = 0.0  # peak memory bandwidth in GB/s


def _init_gpu(gpu_id: int = 0):
    """Initialize CuPy and detect GPU specs."""
    global _cp, _cupyx, _GPU_AVAILABLE, _GPU_NAME, _GPU_VRAM_BYTES, _GPU_MEM_BW_GBS
    try:
        import cupy as cp
        import cupyx

        cp.cuda.Device(gpu_id).use()
        _cp = cp
        _cupyx = cupyx
        _GPU_AVAILABLE = True

        props = cp.cuda.runtime.getDeviceProperties(gpu_id)
        _GPU_NAME = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
        _GPU_VRAM_BYTES = props["totalGlobalMem"]
        # Memory bandwidth = memoryClockRate (kHz) * memoryBusWidth (bits) * 2 (DDR) / 8 (bits->bytes)
        clock_khz = props["memoryClockRate"]
        bus_width_bits = props["memoryBusWidth"]
        _GPU_MEM_BW_GBS = (clock_khz * 1e3 * bus_width_bits * 2) / (8 * 1e9)
    except Exception as e:
        print(f"[WARN] GPU init failed: {e}")
        _GPU_AVAILABLE = False


# ---------------------------------------------------------------------------
# Numba CPU import
# ---------------------------------------------------------------------------
try:
    from numba import njit, prange

    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    print("[WARN] Numba not available, CPU benchmark will use numpy fallback")


# ===================================================================
# BENCHMARK PROFILES (VRAM-adjusted for RTX 3090 24GB)
# ===================================================================
PROFILES = OrderedDict([
    ("1w",  {"rows":    818, "cols":    500_000, "density": 0.003}),
    ("1d",  {"rows":  5_733, "cols":  2_000_000, "density": 0.003}),
    ("4h",  {"rows": 17_520, "cols":  3_000_000, "density": 0.003}),
    ("1h",  {"rows": 75_405, "cols":  5_000_000, "density": 0.002}),
    ("15m", {"rows": 100_000, "cols": 5_000_000, "density": 0.0015}),
])

# EFB config: 255 max_bin -> 254 features per bundle
FEATURES_PER_BUNDLE = 254
NUM_LEAVES = 63
NUM_CLASS = 3
WARMUP_ITERS = 10
TIMED_ITERS = 50


# ===================================================================
# DATA GENERATION (inline, no disk I/O dependency)
# ===================================================================
def make_sparse_binary(rows, cols, density, seed=42):
    """Generate sparse binary CSR matrix with int32 indices, int64 indptr."""
    rng = np.random.RandomState(seed)
    nnz_per_row = max(1, int(cols * density))

    indptr = np.zeros(rows + 1, dtype=np.int64)
    indices_parts = []
    for start in range(0, rows, 100_000):
        end = min(start + 100_000, rows)
        chunk_rows = end - start
        chunk_idx = np.empty(chunk_rows * nnz_per_row, dtype=np.int32)
        for i in range(chunk_rows):
            chunk_idx[i * nnz_per_row:(i + 1) * nnz_per_row] = \
                rng.choice(cols, size=nnz_per_row, replace=False).astype(np.int32)
        indices_parts.append(chunk_idx)
        for i in range(chunk_rows):
            indptr[start + i + 1] = indptr[start + i] + nnz_per_row

    indices = np.concatenate(indices_parts)
    data = np.ones(len(indices), dtype=np.float32)
    mat = sparse.csr_matrix((data, indices, indptr), shape=(rows, cols))
    mat.indices = mat.indices.astype(np.int32)
    mat.indptr = mat.indptr.astype(np.int64)
    return mat


def make_gradients(rows, num_class=3, seed=42):
    """Softmax-style gradients and hessians."""
    rng = np.random.RandomState(seed)
    logits = rng.randn(rows, num_class).astype(np.float32)
    logits -= logits.max(axis=1, keepdims=True)
    exp_l = np.exp(logits)
    probs = exp_l / exp_l.sum(axis=1, keepdims=True)
    labels = rng.randint(0, num_class, size=rows)
    one_hot = np.zeros((rows, num_class), dtype=np.float32)
    one_hot[np.arange(rows), labels] = 1.0
    grad = (probs - one_hot).astype(np.float32)
    hess = (2.0 * probs * (1.0 - probs)).astype(np.float32)
    return grad, hess


def make_leaf_mask(rows, num_leaves=63, seed=42):
    """Non-uniform leaf assignment (Dirichlet weights)."""
    rng = np.random.RandomState(seed)
    weights = rng.dirichlet(np.ones(num_leaves) * 0.5)
    assignment = rng.choice(num_leaves, size=rows, p=weights).astype(np.int32)
    return assignment


def make_efb_mapping(n_cols, fpb=FEATURES_PER_BUNDLE):
    """Feature -> EFB bundle + bin offset."""
    col_to_bundle = np.arange(n_cols, dtype=np.int32) // fpb
    col_to_bin = (np.arange(n_cols, dtype=np.int32) % fpb).astype(np.int32)
    n_bundles = int(col_to_bundle.max()) + 1
    return col_to_bundle, col_to_bin, n_bundles


# ===================================================================
# CPU HISTOGRAM KERNEL (Numba @njit)
# ===================================================================
if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _cpu_histogram_kernel(
        indptr, indices, data,
        grad_flat, hess_flat,
        row_indices,
        col_to_bundle, col_to_bin,
        n_bundles, num_class
    ):
        """
        Build histogram for one leaf on CPU.

        For each row in this leaf, iterate its CSR nonzeros,
        map feature -> bundle+bin, accumulate grad/hess.

        Output shape: (n_bundles, max_bin_per_bundle, num_class, 2)
        Flattened to (n_bundles * 256 * num_class * 2) for simplicity.
        """
        # Histogram: [bundle][bin][class][grad/hess]
        hist = np.zeros(n_bundles * 256 * num_class * 2, dtype=np.float64)

        n_rows = row_indices.shape[0]
        for ri in range(n_rows):
            row = row_indices[ri]
            start = indptr[row]
            end = indptr[row + 1]
            for j in range(start, end):
                col = indices[j]
                bundle = col_to_bundle[col]
                bin_id = col_to_bin[col]
                # data[j] is always 1.0 for binary, but multiply anyway
                val = data[j]
                for c in range(num_class):
                    g = grad_flat[row * num_class + c]
                    h = hess_flat[row * num_class + c]
                    base = (bundle * 256 + bin_id) * num_class * 2 + c * 2
                    hist[base] += val * g
                    hist[base + 1] += val * h
        return hist


def cpu_histogram_numpy(indptr, indices, data, grad, hess,
                        row_indices, col_to_bundle, col_to_bin,
                        n_bundles, num_class):
    """Numpy fallback if Numba unavailable (very slow, reference only)."""
    hist = np.zeros((n_bundles, 256, num_class, 2), dtype=np.float64)
    for ri in range(len(row_indices)):
        row = row_indices[ri]
        start, end = indptr[row], indptr[row + 1]
        cols = indices[start:end]
        vals = data[start:end]
        bundles = col_to_bundle[cols]
        bins = col_to_bin[cols]
        for c in range(num_class):
            g = grad[row, c]
            h = hess[row, c]
            for k in range(len(cols)):
                hist[bundles[k], bins[k], c, 0] += vals[k] * g
                hist[bundles[k], bins[k], c, 1] += vals[k] * h
    return hist.ravel()


# ===================================================================
# GPU KERNELS
# ===================================================================
_ATOMIC_SCATTER_KERNEL_CODE = r"""
extern "C" __global__
void histogram_atomic_scatter(
    const long long* __restrict__ indptr,      // int64 [N+1]
    const int*       __restrict__ indices,     // int32 [nnz]
    const float*     __restrict__ data,        // float32 [nnz]
    const float*     __restrict__ grad,        // float32 [N * num_class]
    const float*     __restrict__ hess,        // float32 [N * num_class]
    const int*       __restrict__ row_indices, // int32 [n_leaf_rows]
    const int*       __restrict__ col_to_bundle, // int32 [n_cols]
    const int*       __restrict__ col_to_bin,    // int32 [n_cols]
    double*          __restrict__ hist,        // float64 [n_bundles * 256 * num_class * 2]
    int n_leaf_rows,
    int num_class
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= n_leaf_rows) return;

    int row = row_indices[tid];
    long long start = indptr[row];
    long long end   = indptr[row + 1];

    for (long long j = start; j < end; j++) {
        int col = indices[j];
        int bundle = col_to_bundle[col];
        int bin_id = col_to_bin[col];
        float val  = data[j];

        for (int c = 0; c < num_class; c++) {
            float g = grad[row * num_class + c];
            float h = hess[row * num_class + c];
            int base = (bundle * 256 + bin_id) * num_class * 2 + c * 2;
            atomicAdd(&hist[base],     (double)(val * g));
            atomicAdd(&hist[base + 1], (double)(val * h));
        }
    }
}
"""


def _compile_atomic_kernel():
    """Compile the atomic scatter histogram kernel."""
    return _cp.RawKernel(_ATOMIC_SCATTER_KERNEL_CODE, "histogram_atomic_scatter")


# ===================================================================
# TIMING UTILITIES
# ===================================================================
class TimingResult:
    """Stores timing measurements for a single benchmark."""

    def __init__(self, name):
        self.name = name
        self.times = []
        self.h2d_times = []
        self.kernel_times = []
        self.d2h_times = []

    def add(self, total, h2d=0.0, kernel=0.0, d2h=0.0):
        self.times.append(total)
        self.h2d_times.append(h2d)
        self.kernel_times.append(kernel)
        self.d2h_times.append(d2h)

    def stats(self, arr):
        if not arr or all(x == 0.0 for x in arr):
            return {"mean": 0, "std": 0, "min": 0, "max": 0, "p50": 0, "p95": 0}
        a = np.array(arr)
        return {
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
            "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
        }

    def to_dict(self):
        return {
            "name": self.name,
            "total": self.stats(self.times),
            "h2d": self.stats(self.h2d_times),
            "kernel": self.stats(self.kernel_times),
            "d2h": self.stats(self.d2h_times),
            "n_samples": len(self.times),
        }


def _gpu_vram_used():
    """Return (used_bytes, total_bytes) on current GPU."""
    if not _GPU_AVAILABLE:
        return 0, 0
    free = _cp.cuda.runtime.memGetInfo()[0]
    total = _GPU_VRAM_BYTES
    return total - free, total


def _human_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _human_ms(sec):
    if sec < 0.001:
        return f"{sec * 1e6:.1f} us"
    if sec < 1.0:
        return f"{sec * 1e3:.2f} ms"
    return f"{sec:.3f} s"


# ===================================================================
# BENCHMARK RUNNERS
# ===================================================================
def bench_cpu(mat, grad, hess, leaf_rows, col_to_bundle, col_to_bin,
              n_bundles, num_class, warmup, iters):
    """Benchmark CPU histogram kernel."""
    result = TimingResult("cpu_numba")
    grad_flat = grad.ravel().astype(np.float32)
    hess_flat = hess.ravel().astype(np.float32)

    if _NUMBA_AVAILABLE:
        fn = _cpu_histogram_kernel
    else:
        result.name = "cpu_numpy"
        fn = None

    # Warmup (triggers Numba JIT)
    for _ in range(warmup):
        if fn is not None:
            fn(mat.indptr, mat.indices, mat.data, grad_flat, hess_flat,
               leaf_rows, col_to_bundle, col_to_bin, n_bundles, num_class)
        else:
            cpu_histogram_numpy(mat.indptr, mat.indices, mat.data, grad, hess,
                                leaf_rows, col_to_bundle, col_to_bin, n_bundles, num_class)

    # Timed
    for _ in range(iters):
        t0 = time.perf_counter()
        if fn is not None:
            fn(mat.indptr, mat.indices, mat.data, grad_flat, hess_flat,
               leaf_rows, col_to_bundle, col_to_bin, n_bundles, num_class)
        else:
            cpu_histogram_numpy(mat.indptr, mat.indices, mat.data, grad, hess,
                                leaf_rows, col_to_bundle, col_to_bin, n_bundles, num_class)
        elapsed = time.perf_counter() - t0
        result.add(elapsed)

    return result


def bench_gpu_cusparse_spmv(mat, grad, hess, leaf_rows, col_to_bundle, col_to_bin,
                            n_bundles, num_class, warmup, iters):
    """
    GPU cuSPARSE SpMV approach.

    Strategy: extract leaf rows from the CSR, then use SpMV to multiply
    the submatrix by a gradient vector for each class. This produces
    per-feature gradient sums, which we then scatter into histogram bins.
    """
    result = TimingResult("gpu_cusparse_spmv")

    # Build leaf submatrix on CPU
    sub = mat[leaf_rows]
    sub_cp = _cupyx.scipy.sparse.csr_matrix(
        (_cp.array(sub.data), _cp.array(sub.indices.astype(np.int32)),
         _cp.array(sub.indptr.astype(np.int64))),
        shape=sub.shape, dtype=_cp.float32
    )

    # Pre-upload bundle mapping
    d_col_to_bundle = _cp.array(col_to_bundle)
    d_col_to_bin = _cp.array(col_to_bin)

    # Grad/hess for leaf rows: shape (n_leaf_rows, num_class)
    leaf_grad = grad[leaf_rows]
    leaf_hess = hess[leaf_rows]

    for wi in range(warmup + iters):
        _cp.cuda.Device().synchronize()
        gc.collect()

        # --- H2D ---
        ev_h2d_start = _cp.cuda.Event()
        ev_h2d_end = _cp.cuda.Event()
        ev_h2d_start.record()

        d_grad = _cp.array(leaf_grad, dtype=_cp.float32)
        d_hess = _cp.array(leaf_hess, dtype=_cp.float32)

        ev_h2d_end.record()
        ev_h2d_end.synchronize()
        h2d_ms = _cp.cuda.get_elapsed_time(ev_h2d_start, ev_h2d_end)

        # --- Kernel (SpMV per class) ---
        ev_k_start = _cp.cuda.Event()
        ev_k_end = _cp.cuda.Event()
        ev_k_start.record()

        hist = _cp.zeros(n_bundles * 256 * num_class * 2, dtype=_cp.float64)
        n_features = mat.shape[1]

        for c in range(num_class):
            # SpMV: sub_cp.T @ d_grad[:, c] -> per-feature grad sum
            gvec = d_grad[:, c]
            hvec = d_hess[:, c]
            feat_grad_sum = sub_cp.T @ gvec  # shape (n_features,)
            feat_hess_sum = sub_cp.T @ hvec

            # Scatter into histogram bins
            # hist[(bundle*256 + bin)*num_class*2 + c*2] += feat_grad_sum[f]
            # Vectorized scatter using advanced indexing
            hist_idx_g = (d_col_to_bundle[:n_features] * 256 +
                          d_col_to_bin[:n_features]) * num_class * 2 + c * 2
            hist_idx_h = hist_idx_g + 1
            _cupyx.scatter_add(hist, hist_idx_g, feat_grad_sum.astype(_cp.float64))
            _cupyx.scatter_add(hist, hist_idx_h, feat_hess_sum.astype(_cp.float64))

        ev_k_end.record()
        ev_k_end.synchronize()
        kernel_ms = _cp.cuda.get_elapsed_time(ev_k_start, ev_k_end)

        # --- D2H ---
        ev_d2h_start = _cp.cuda.Event()
        ev_d2h_end = _cp.cuda.Event()
        ev_d2h_start.record()

        _ = hist.get()

        ev_d2h_end.record()
        ev_d2h_end.synchronize()
        d2h_ms = _cp.cuda.get_elapsed_time(ev_d2h_start, ev_d2h_end)

        if wi >= warmup:
            total_ms = h2d_ms + kernel_ms + d2h_ms
            result.add(total_ms / 1000, h2d_ms / 1000, kernel_ms / 1000, d2h_ms / 1000)

    return result


def bench_gpu_atomic_scatter(mat, grad, hess, leaf_rows, col_to_bundle, col_to_bin,
                             n_bundles, num_class, warmup, iters):
    """
    GPU atomic scatter kernel approach.

    Each thread handles one leaf row, iterates its CSR nonzeros,
    and does atomicAdd into the histogram. Direct port of CPU logic.
    """
    result = TimingResult("gpu_atomic_scatter")
    kernel = _compile_atomic_kernel()

    # Pre-upload CSR structure (stays on GPU between iterations)
    d_indptr = _cp.array(mat.indptr)
    d_indices = _cp.array(mat.indices.astype(np.int32))
    d_data = _cp.array(mat.data)
    d_col_to_bundle = _cp.array(col_to_bundle)
    d_col_to_bin = _cp.array(col_to_bin)
    d_grad_full = _cp.array(grad.ravel(), dtype=_cp.float32)
    d_hess_full = _cp.array(hess.ravel(), dtype=_cp.float32)

    n_leaf = len(leaf_rows)
    block = 256
    grid = (n_leaf + block - 1) // block

    for wi in range(warmup + iters):
        _cp.cuda.Device().synchronize()

        # --- H2D (leaf row indices) ---
        ev_h2d_start = _cp.cuda.Event()
        ev_h2d_end = _cp.cuda.Event()
        ev_h2d_start.record()

        d_leaf_rows = _cp.array(leaf_rows.astype(np.int32))

        ev_h2d_end.record()
        ev_h2d_end.synchronize()
        h2d_ms = _cp.cuda.get_elapsed_time(ev_h2d_start, ev_h2d_end)

        # --- Kernel ---
        ev_k_start = _cp.cuda.Event()
        ev_k_end = _cp.cuda.Event()
        ev_k_start.record()

        d_hist = _cp.zeros(n_bundles * 256 * num_class * 2, dtype=_cp.float64)
        kernel((grid,), (block,),
               (d_indptr, d_indices, d_data,
                d_grad_full, d_hess_full,
                d_leaf_rows, d_col_to_bundle, d_col_to_bin,
                d_hist, np.int32(n_leaf), np.int32(num_class)))

        ev_k_end.record()
        ev_k_end.synchronize()
        kernel_ms = _cp.cuda.get_elapsed_time(ev_k_start, ev_k_end)

        # --- D2H ---
        ev_d2h_start = _cp.cuda.Event()
        ev_d2h_end = _cp.cuda.Event()
        ev_d2h_start.record()

        _ = d_hist.get()

        ev_d2h_end.record()
        ev_d2h_end.synchronize()
        d2h_ms = _cp.cuda.get_elapsed_time(ev_d2h_start, ev_d2h_end)

        if wi >= warmup:
            total_ms = h2d_ms + kernel_ms + d2h_ms
            result.add(total_ms / 1000, h2d_ms / 1000, kernel_ms / 1000, d2h_ms / 1000)

    return result


# ===================================================================
# MEMORY ESTIMATION
# ===================================================================
def estimate_gpu_memory(rows, cols, density, n_bundles, num_class):
    """Estimate GPU VRAM needed for the atomic scatter approach."""
    nnz = int(rows * cols * density)
    mem = {
        "indptr": (rows + 1) * 8,             # int64
        "indices": nnz * 4,                     # int32
        "data": nnz * 4,                        # float32
        "grad": rows * num_class * 4,           # float32
        "hess": rows * num_class * 4,           # float32
        "col_to_bundle": cols * 4,              # int32
        "col_to_bin": cols * 4,                 # int32
        "hist": n_bundles * 256 * num_class * 2 * 8,  # float64
        "leaf_rows": rows * 4,                  # int32 (worst case: all rows in one leaf)
    }
    return mem


def estimate_bandwidth_util(bytes_accessed, time_sec):
    """Estimate fraction of peak memory bandwidth utilized."""
    if time_sec <= 0 or _GPU_MEM_BW_GBS <= 0:
        return 0.0
    actual_bw = bytes_accessed / time_sec / 1e9  # GB/s
    return actual_bw / _GPU_MEM_BW_GBS


# ===================================================================
# SINGLE PROFILE BENCHMARK
# ===================================================================
def run_profile(name, rows, cols, density, warmup=WARMUP_ITERS, iters=TIMED_ITERS,
                num_class=NUM_CLASS, num_leaves=NUM_LEAVES, verbose=True):
    """Run full benchmark for one profile. Returns dict of results."""

    if verbose:
        print(f"\n{'='*72}")
        print(f"  PROFILE: {name}  ({rows:,} rows x {cols:,} cols, density={density})")
        print(f"{'='*72}")

    # Generate data
    if verbose:
        print("  Generating sparse matrix ...", end=" ", flush=True)
    t0 = time.perf_counter()
    mat = make_sparse_binary(rows, cols, density)
    if verbose:
        print(f"done ({time.perf_counter()-t0:.1f}s, nnz={mat.nnz:,})")

    grad, hess = make_gradients(rows, num_class)
    leaf_assign = make_leaf_mask(rows, num_leaves)

    col_to_bundle, col_to_bin, n_bundles = make_efb_mapping(cols)

    # Pick a representative leaf (the largest one, worst case)
    leaf_sizes = np.bincount(leaf_assign, minlength=num_leaves)
    biggest_leaf = int(np.argmax(leaf_sizes))
    leaf_rows = np.where(leaf_assign == biggest_leaf)[0].astype(np.int32)

    if verbose:
        print(f"  Biggest leaf: {biggest_leaf} with {len(leaf_rows):,} rows "
              f"({100*len(leaf_rows)/rows:.1f}% of total)")
        print(f"  EFB bundles: {n_bundles:,}")

    # Memory estimates
    mem_est = estimate_gpu_memory(rows, cols, density, n_bundles, num_class)
    total_gpu_mem = sum(mem_est.values())
    if verbose:
        print(f"  Estimated GPU memory: {_human_bytes(total_gpu_mem)}")
        if _GPU_AVAILABLE:
            used, total = _gpu_vram_used()
            print(f"  GPU VRAM: {_human_bytes(used)} / {_human_bytes(total)} "
                  f"({100*used/total:.1f}% used)")

    # Bytes accessed per histogram build (for bandwidth calc)
    nnz_leaf = int(len(leaf_rows) * cols * density)
    bytes_per_hist = (
        nnz_leaf * 4 +                         # indices reads
        nnz_leaf * 4 +                         # data reads
        (len(leaf_rows) + 1) * 8 +             # indptr reads
        len(leaf_rows) * num_class * 4 +       # grad reads
        len(leaf_rows) * num_class * 4 +       # hess reads
        nnz_leaf * 4 +                         # col_to_bundle reads
        nnz_leaf * 4 +                         # col_to_bin reads
        nnz_leaf * num_class * 2 * 8           # hist writes (atomic)
    )

    results = {
        "profile": name,
        "rows": rows,
        "cols": cols,
        "density": density,
        "nnz": mat.nnz,
        "n_bundles": n_bundles,
        "leaf_rows": len(leaf_rows),
        "bytes_per_hist": bytes_per_hist,
        "gpu_mem_estimate_bytes": total_gpu_mem,
        "benchmarks": {},
    }

    # --- CPU benchmark ---
    if verbose:
        print(f"\n  [CPU] Running {warmup} warmup + {iters} timed iterations ...")
    cpu_res = bench_cpu(mat, grad, hess, leaf_rows, col_to_bundle, col_to_bin,
                        n_bundles, num_class, warmup, iters)
    results["benchmarks"]["cpu"] = cpu_res.to_dict()
    cpu_mean = cpu_res.to_dict()["total"]["mean"]
    if verbose:
        print(f"  [CPU] mean={_human_ms(cpu_mean)}, "
              f"p50={_human_ms(cpu_res.to_dict()['total']['p50'])}, "
              f"p95={_human_ms(cpu_res.to_dict()['total']['p95'])}")

    # --- GPU benchmarks ---
    if _GPU_AVAILABLE:
        # Check if we can fit this on GPU
        used, total_vram = _gpu_vram_used()
        available = total_vram - used
        if total_gpu_mem > available * 0.9:
            if verbose:
                print(f"\n  [GPU] SKIPPED — needs {_human_bytes(total_gpu_mem)}, "
                      f"only {_human_bytes(available)} available")
            results["benchmarks"]["gpu_cusparse"] = {"skipped": "OOM"}
            results["benchmarks"]["gpu_atomic"] = {"skipped": "OOM"}
        else:
            # cuSPARSE SpMV
            if verbose:
                print(f"\n  [GPU cuSPARSE] Running {warmup} warmup + {iters} timed ...")
            try:
                spmv_res = bench_gpu_cusparse_spmv(
                    mat, grad, hess, leaf_rows, col_to_bundle, col_to_bin,
                    n_bundles, num_class, warmup, iters)
                results["benchmarks"]["gpu_cusparse"] = spmv_res.to_dict()
                spmv_mean = spmv_res.to_dict()["total"]["mean"]
                bw_util = estimate_bandwidth_util(bytes_per_hist, spmv_mean)
                results["benchmarks"]["gpu_cusparse"]["bandwidth_util"] = bw_util
                if verbose:
                    print(f"  [GPU cuSPARSE] mean={_human_ms(spmv_mean)}, "
                          f"speedup={cpu_mean/spmv_mean:.1f}x, "
                          f"BW util={100*bw_util:.1f}%")
                    used_now, _ = _gpu_vram_used()
                    print(f"  [GPU cuSPARSE] VRAM after: {_human_bytes(used_now)}")
            except Exception as e:
                if verbose:
                    print(f"  [GPU cuSPARSE] FAILED: {e}")
                results["benchmarks"]["gpu_cusparse"] = {"error": str(e)}

            # Free cuSPARSE intermediates
            _cp.get_default_memory_pool().free_all_blocks()
            gc.collect()

            # Atomic scatter
            if verbose:
                print(f"\n  [GPU atomic] Running {warmup} warmup + {iters} timed ...")
            try:
                atomic_res = bench_gpu_atomic_scatter(
                    mat, grad, hess, leaf_rows, col_to_bundle, col_to_bin,
                    n_bundles, num_class, warmup, iters)
                results["benchmarks"]["gpu_atomic"] = atomic_res.to_dict()
                atomic_mean = atomic_res.to_dict()["total"]["mean"]
                bw_util = estimate_bandwidth_util(bytes_per_hist, atomic_mean)
                results["benchmarks"]["gpu_atomic"]["bandwidth_util"] = bw_util
                if verbose:
                    print(f"  [GPU atomic] mean={_human_ms(atomic_mean)}, "
                          f"speedup={cpu_mean/atomic_mean:.1f}x, "
                          f"BW util={100*bw_util:.1f}%")
                    used_now, _ = _gpu_vram_used()
                    print(f"  [GPU atomic] VRAM after: {_human_bytes(used_now)}")
            except Exception as e:
                if verbose:
                    print(f"  [GPU atomic] FAILED: {e}")
                results["benchmarks"]["gpu_atomic"] = {"error": str(e)}

            _cp.get_default_memory_pool().free_all_blocks()
    else:
        results["benchmarks"]["gpu_cusparse"] = {"skipped": "no GPU"}
        results["benchmarks"]["gpu_atomic"] = {"skipped": "no GPU"}

    # Cleanup
    del mat, grad, hess, leaf_assign, col_to_bundle, col_to_bin
    gc.collect()
    if _GPU_AVAILABLE:
        _cp.get_default_memory_pool().free_all_blocks()

    return results


# ===================================================================
# SCALING TESTS
# ===================================================================
def run_scaling_features(warmup, iters, verbose=True):
    """Fixed rows (17,520), vary features: 100K, 500K, 1M, 3M, 5M."""
    fixed_rows = 17_520
    feature_counts = [100_000, 500_000, 1_000_000, 3_000_000, 5_000_000]
    results = []
    for fc in feature_counts:
        r = run_profile(f"scale_feat_{fc//1000}K", fixed_rows, fc, 0.003,
                        warmup=warmup, iters=iters, verbose=verbose)
        results.append(r)
    return results


def run_scaling_rows(warmup, iters, verbose=True):
    """Fixed features (3M), vary rows: 1K, 5K, 10K, 17K, 50K, 100K."""
    fixed_cols = 3_000_000
    row_counts = [1_000, 5_000, 10_000, 17_520, 50_000, 100_000]
    results = []
    for rc in row_counts:
        density = 0.003 if rc <= 17_520 else 0.002
        r = run_profile(f"scale_rows_{rc//1000}K", rc, fixed_cols, density,
                        warmup=warmup, iters=iters, verbose=verbose)
        results.append(r)
    return results


def run_scaling_leaves(warmup, iters, verbose=True):
    """
    Fixed full matrix (17,520 rows, 3M cols), vary leaf size:
    10%, 25%, 50%, 75%, 100% of rows.
    """
    rows = 17_520
    cols = 3_000_000
    density = 0.003
    pcts = [0.10, 0.25, 0.50, 0.75, 1.00]

    if verbose:
        print(f"\n{'='*72}")
        print(f"  LEAF SIZE SCALING  ({rows:,} rows x {cols:,} cols)")
        print(f"{'='*72}")

    # Generate matrix once
    mat = make_sparse_binary(rows, cols, density)
    grad, hess = make_gradients(rows, NUM_CLASS)
    col_to_bundle, col_to_bin, n_bundles = make_efb_mapping(cols)

    results = []
    for pct in pcts:
        n_leaf = max(1, int(rows * pct))
        leaf_rows = np.arange(n_leaf, dtype=np.int32)

        tag = f"leaf_{int(pct*100)}pct"
        if verbose:
            print(f"\n  --- {tag}: {n_leaf:,} rows ({pct*100:.0f}%) ---")

        sub_results = {"profile": tag, "rows": rows, "cols": cols,
                       "leaf_rows": n_leaf, "leaf_pct": pct, "benchmarks": {}}

        # CPU
        cpu_res = bench_cpu(mat, grad, hess, leaf_rows, col_to_bundle, col_to_bin,
                            n_bundles, NUM_CLASS, warmup, iters)
        sub_results["benchmarks"]["cpu"] = cpu_res.to_dict()
        cpu_mean = cpu_res.to_dict()["total"]["mean"]
        if verbose:
            print(f"    CPU: {_human_ms(cpu_mean)}")

        if _GPU_AVAILABLE:
            try:
                atomic_res = bench_gpu_atomic_scatter(
                    mat, grad, hess, leaf_rows, col_to_bundle, col_to_bin,
                    n_bundles, NUM_CLASS, warmup, iters)
                sub_results["benchmarks"]["gpu_atomic"] = atomic_res.to_dict()
                atomic_mean = atomic_res.to_dict()["total"]["mean"]
                if verbose:
                    print(f"    GPU atomic: {_human_ms(atomic_mean)} "
                          f"(speedup {cpu_mean/atomic_mean:.1f}x)")
            except Exception as e:
                if verbose:
                    print(f"    GPU atomic: FAILED ({e})")
                sub_results["benchmarks"]["gpu_atomic"] = {"error": str(e)}
            _cp.get_default_memory_pool().free_all_blocks()

        results.append(sub_results)

    del mat, grad, hess
    gc.collect()
    if _GPU_AVAILABLE:
        _cp.get_default_memory_pool().free_all_blocks()

    return results


# ===================================================================
# ASCII TABLE OUTPUT
# ===================================================================
def print_summary_table(all_results):
    """Print ASCII summary table of all benchmark results."""
    print(f"\n{'='*100}")
    print("  BENCHMARK SUMMARY")
    print(f"{'='*100}")
    print(f"  GPU: {_GPU_NAME}" if _GPU_AVAILABLE else "  GPU: not available")
    if _GPU_AVAILABLE:
        print(f"  VRAM: {_human_bytes(_GPU_VRAM_BYTES)}, "
              f"Peak BW: {_GPU_MEM_BW_GBS:.0f} GB/s")
    print(f"  Iterations: {WARMUP_ITERS} warmup + {TIMED_ITERS} timed")
    print(f"{'='*100}")

    hdr = (f"  {'Profile':<20} {'Rows':>8} {'Cols':>8} {'LeafRows':>9} "
           f"{'CPU':>10} {'cuSPARSE':>10} {'Atomic':>10} "
           f"{'SpMV/CPU':>9} {'Atom/CPU':>9}")
    print(hdr)
    print(f"  {'-'*98}")

    for r in all_results:
        profile = r.get("profile", "?")
        rows = r.get("rows", 0)
        cols = r.get("cols", 0)
        n_leaf = r.get("leaf_rows", 0)

        cpu_d = r.get("benchmarks", {}).get("cpu", {})
        spmv_d = r.get("benchmarks", {}).get("gpu_cusparse", {})
        atom_d = r.get("benchmarks", {}).get("gpu_atomic", {})

        cpu_mean = cpu_d.get("total", {}).get("mean", 0) if isinstance(cpu_d, dict) and "total" in cpu_d else 0
        spmv_mean = spmv_d.get("total", {}).get("mean", 0) if isinstance(spmv_d, dict) and "total" in spmv_d else 0
        atom_mean = atom_d.get("total", {}).get("mean", 0) if isinstance(atom_d, dict) and "total" in atom_d else 0

        spmv_str = _human_ms(spmv_mean) if spmv_mean > 0 else "N/A"
        atom_str = _human_ms(atom_mean) if atom_mean > 0 else "N/A"

        spmv_speedup = f"{cpu_mean/spmv_mean:.1f}x" if spmv_mean > 0 and cpu_mean > 0 else "N/A"
        atom_speedup = f"{cpu_mean/atom_mean:.1f}x" if atom_mean > 0 and cpu_mean > 0 else "N/A"

        cols_str = f"{cols//1_000_000}M" if cols >= 1_000_000 else f"{cols//1_000}K"

        print(f"  {profile:<20} {rows:>8,} {cols_str:>8} {n_leaf:>9,} "
              f"{_human_ms(cpu_mean):>10} {spmv_str:>10} {atom_str:>10} "
              f"{spmv_speedup:>9} {atom_speedup:>9}")

    print(f"{'='*100}")

    # Detailed breakdown for GPU atomic (usually the better kernel)
    print(f"\n  GPU Atomic Scatter — Timing Breakdown")
    print(f"  {'-'*80}")
    print(f"  {'Profile':<20} {'H2D':>10} {'Kernel':>10} {'D2H':>10} {'Total':>10} {'BW Util':>8}")
    print(f"  {'-'*80}")
    for r in all_results:
        profile = r.get("profile", "?")
        atom_d = r.get("benchmarks", {}).get("gpu_atomic", {})
        if not isinstance(atom_d, dict) or "total" not in atom_d:
            continue
        h2d = atom_d.get("h2d", {}).get("mean", 0)
        kern = atom_d.get("kernel", {}).get("mean", 0)
        d2h = atom_d.get("d2h", {}).get("mean", 0)
        total = atom_d.get("total", {}).get("mean", 0)
        bw = atom_d.get("bandwidth_util", 0)
        print(f"  {profile:<20} {_human_ms(h2d):>10} {_human_ms(kern):>10} "
              f"{_human_ms(d2h):>10} {_human_ms(total):>10} {100*bw:>7.1f}%")
    print(f"  {'-'*80}")


# ===================================================================
# MAIN
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Micro-benchmark: CPU vs GPU histogram building speed.\n"
                    "Compare Numba CPU, cuSPARSE SpMV, and atomic scatter kernel.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--profile", nargs="+", default=None,
        choices=list(PROFILES.keys()),
        help="Which TF profiles to benchmark (default: all)."
    )
    parser.add_argument("--gpu-id", type=int, default=0,
                        help="CUDA device ID (default: 0).")
    parser.add_argument("--output", type=str, default=None,
                        help="Path for JSON results file.")
    parser.add_argument("--warmup", type=int, default=WARMUP_ITERS,
                        help=f"Warmup iterations (default: {WARMUP_ITERS}).")
    parser.add_argument("--iters", type=int, default=TIMED_ITERS,
                        help=f"Timed iterations (default: {TIMED_ITERS}).")
    parser.add_argument("--scaling-features", action="store_true",
                        help="Run feature count scaling test.")
    parser.add_argument("--scaling-rows", action="store_true",
                        help="Run row count scaling test.")
    parser.add_argument("--scaling-leaves", action="store_true",
                        help="Run leaf size scaling test.")
    parser.add_argument("--all-scaling", action="store_true",
                        help="Run all scaling tests.")
    args = parser.parse_args()

    print("=" * 72)
    print("  GPU Histogram Kernel Speed Benchmark")
    print("  Matrix thesis: 2-10M sparse binary features, 17K-294K rows")
    print("=" * 72)

    # Init GPU
    _init_gpu(args.gpu_id)
    if _GPU_AVAILABLE:
        print(f"  GPU: {_GPU_NAME}")
        print(f"  VRAM: {_human_bytes(_GPU_VRAM_BYTES)}")
        print(f"  Peak BW: {_GPU_MEM_BW_GBS:.0f} GB/s")
    else:
        print("  GPU: NOT AVAILABLE (CPU-only benchmark)")

    if not _NUMBA_AVAILABLE:
        print("  WARNING: Numba not available, CPU benchmark will be very slow")

    all_results = []
    output_data = {
        "gpu_name": _GPU_NAME,
        "gpu_vram_bytes": _GPU_VRAM_BYTES,
        "gpu_peak_bw_gbs": _GPU_MEM_BW_GBS,
        "gpu_available": _GPU_AVAILABLE,
        "warmup_iters": args.warmup,
        "timed_iters": args.iters,
        "profiles": [],
        "scaling_features": [],
        "scaling_rows": [],
        "scaling_leaves": [],
    }

    # --- Profile benchmarks ---
    profiles_to_run = args.profile if args.profile else list(PROFILES.keys())
    run_scaling = args.scaling_features or args.scaling_rows or args.scaling_leaves or args.all_scaling

    if not run_scaling or args.profile:
        for name in profiles_to_run:
            p = PROFILES[name]
            r = run_profile(name, p["rows"], p["cols"], p["density"],
                            warmup=args.warmup, iters=args.iters)
            all_results.append(r)
            output_data["profiles"].append(r)

    # --- Scaling tests ---
    if args.scaling_features or args.all_scaling:
        print(f"\n\n{'#'*72}")
        print("  SCALING TEST: Features (fixed rows=17,520)")
        print(f"{'#'*72}")
        sf = run_scaling_features(args.warmup, args.iters)
        all_results.extend(sf)
        output_data["scaling_features"] = sf

    if args.scaling_rows or args.all_scaling:
        print(f"\n\n{'#'*72}")
        print("  SCALING TEST: Rows (fixed cols=3M)")
        print(f"{'#'*72}")
        sr = run_scaling_rows(args.warmup, args.iters)
        all_results.extend(sr)
        output_data["scaling_rows"] = sr

    if args.scaling_leaves or args.all_scaling:
        print(f"\n\n{'#'*72}")
        print("  SCALING TEST: Leaf Size (17,520 rows x 3M cols)")
        print(f"{'#'*72}")
        sl = run_scaling_leaves(args.warmup, args.iters)
        all_results.extend(sl)
        output_data["scaling_leaves"] = sl

    # --- Summary ---
    if all_results:
        print_summary_table(all_results)

    # --- VRAM summary ---
    if _GPU_AVAILABLE:
        print(f"\n  GPU Memory Summary:")
        used, total = _gpu_vram_used()
        print(f"    Final VRAM: {_human_bytes(used)} / {_human_bytes(total)} "
              f"({100*used/total:.1f}% used)")

    # --- Save JSON ---
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\n  Results saved to: {out_path.resolve()}")

    print(f"\n{'='*72}")
    print("  DONE")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
