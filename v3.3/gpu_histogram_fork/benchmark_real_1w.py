#!/usr/bin/env python3
"""
Benchmark cuSPARSE SpMV on REAL 1w Cross Feature Data
=====================================================

Loads the actual v2_crosses_BTC_1w.npz sparse matrix and benchmarks
GPU-accelerated histogram building (CSR.T @ gradient_vector) using CuPy.

No LightGBM fork needed -- pure CuPy cuSPARSE.

Usage:
    cd v3.3/gpu_histogram_fork
    python benchmark_real_1w.py

Requires: cupy, scipy, numpy
"""

import os
import sys
import time
import numpy as np
import scipy.sparse as sp

# ─── Locate the real NPZ ─────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
V33_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(V33_DIR)

# Search paths in priority order
NPZ_CANDIDATES = [
    os.path.join(V33_DIR, "v2_crosses_BTC_1w.npz"),
    os.path.join(PROJECT_ROOT, "v3.2_2.9M_Features", "v2_crosses_BTC_1w.npz"),
    os.path.join(PROJECT_ROOT, "v3.0 (LGBM)", "v2_crosses_BTC_1w.npz"),
]

CROSS_NAMES_CANDIDATES = [
    os.path.join(V33_DIR, "v2_cross_names_BTC_1w.json"),
    os.path.join(PROJECT_ROOT, "v3.2_2.9M_Features", "v2_cross_names_BTC_1w.json"),
    os.path.join(PROJECT_ROOT, "v3.0 (LGBM)", "v2_cross_names_BTC_1w.json"),
]

# ─── Check CuPy ──────────────────────────────────────────────────────────────
try:
    import cupy as cp
    from cupyx.scipy import sparse as cp_sparse
    VRAM_TOTAL = cp.cuda.Device(0).mem_info[1] / (1024**3)
    try:
        props = cp.cuda.runtime.getDeviceProperties(0)
        GPU_NAME = props.get("name", "Unknown GPU")
    except Exception:
        GPU_NAME = "Unknown GPU"
    print(f"GPU: {GPU_NAME}")
    print(f"VRAM: {VRAM_TOTAL:.1f} GB")
    print(f"CuPy: {cp.__version__}")
except ImportError:
    print("ERROR: CuPy not installed. pip install cupy-cuda12x")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: GPU not available: {e}")
    sys.exit(1)


def find_file(candidates, label):
    """Find first existing file from candidates list."""
    for path in candidates:
        if os.path.isfile(path):
            return path
    print(f"\nERROR: {label} not found. Searched:")
    for p in candidates:
        print(f"  {p}")
    return None


def load_real_data():
    """Load real 1w cross feature NPZ and cross names."""
    npz_path = find_file(NPZ_CANDIDATES, "v2_crosses_BTC_1w.npz")
    if npz_path is None:
        return None, None

    print(f"\nLoading: {npz_path}")
    t0 = time.perf_counter()
    X = sp.load_npz(npz_path)
    load_time = time.perf_counter() - t0

    n_rows, n_cols = X.shape
    density = X.nnz / (n_rows * n_cols) if n_rows * n_cols > 0 else 0

    print(f"  Shape: {n_rows:,} x {n_cols:,}")
    print(f"  NNZ: {X.nnz:,}")
    print(f"  Density: {density:.4%}")
    print(f"  Load time: {load_time:.2f}s")
    print(f"  CSR size: {(X.data.nbytes + X.indices.nbytes + X.indptr.nbytes) / 1e6:.1f} MB")
    print(f"  dtype: data={X.data.dtype}, indices={X.indices.dtype}, indptr={X.indptr.dtype}")

    # Load cross names if available
    names_path = find_file(CROSS_NAMES_CANDIDATES, "v2_cross_names_BTC_1w.json")
    n_names = None
    if names_path:
        import json
        with open(names_path, "r") as f:
            names = json.load(f)
        n_names = len(names)
        print(f"  Cross names: {n_names:,} (from {os.path.basename(names_path)})")
        if n_names != n_cols:
            print(f"  WARNING: name count ({n_names:,}) != column count ({n_cols:,})")

    return X, names_path


def generate_realistic_gradients(n_rows):
    """Generate realistic LightGBM 3-class gradients and hessians."""
    rng = np.random.default_rng(42)
    grad = rng.standard_normal(n_rows).astype(np.float64)
    hess = np.abs(rng.standard_normal(n_rows)).astype(np.float64) + 0.1
    return grad, hess


def benchmark_cpu(X_T_cpu, grad, n_iters=50):
    """CPU baseline: scipy SpMV."""
    print(f"\n{'='*70}")
    print(f"CPU BASELINE (scipy SpMV, {n_iters} iterations)")
    print(f"{'='*70}")

    # Warmup
    _ = X_T_cpu @ grad

    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        result = X_T_cpu @ grad
        times.append(time.perf_counter() - t0)

    med_ms = np.median(times) * 1000
    min_ms = np.min(times) * 1000
    max_ms = np.max(times) * 1000
    mean_ms = np.mean(times) * 1000

    print(f"  Median: {med_ms:.2f} ms")
    print(f"  Min:    {min_ms:.2f} ms")
    print(f"  Max:    {max_ms:.2f} ms")
    print(f"  Mean:   {mean_ms:.2f} ms")

    return result, med_ms


def benchmark_gpu(X, grad, n_warmup=10, n_iters=50):
    """GPU cuSPARSE benchmark with detailed timing."""
    n_rows, n_cols = X.shape

    print(f"\n{'='*70}")
    print(f"GPU cuSPARSE (CuPy SpMV, {n_iters} iterations)")
    print(f"{'='*70}")

    # ── H2D Transfer ──────────────────────────────────────────────────────
    print("\n  --- Host-to-Device Transfer ---")
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    X_gpu = cp_sparse.csr_matrix(X)
    cp.cuda.Stream.null.synchronize()
    h2d_matrix_ms = (time.perf_counter() - t0) * 1000
    print(f"  CSR matrix upload: {h2d_matrix_ms:.1f} ms")

    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    grad_gpu = cp.asarray(grad)
    cp.cuda.Stream.null.synchronize()
    h2d_grad_ms = (time.perf_counter() - t0) * 1000
    print(f"  Gradient vector upload: {h2d_grad_ms:.2f} ms")

    # ── Pre-transpose (one-time cost) ────────────────────────────────────
    print("\n  --- Pre-transpose (one-time) ---")
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    X_T_gpu = X_gpu.T.tocsr()
    cp.cuda.Stream.null.synchronize()
    transpose_ms = (time.perf_counter() - t0) * 1000
    print(f"  X.T.tocsr() on GPU: {transpose_ms:.1f} ms")
    print(f"  X_T shape: {X_T_gpu.shape[0]:,} x {X_T_gpu.shape[1]:,}")

    total_h2d_ms = h2d_matrix_ms + h2d_grad_ms + transpose_ms
    print(f"  Total setup (H2D + transpose): {total_h2d_ms:.1f} ms")

    # ── Warmup ────────────────────────────────────────────────────────────
    for _ in range(n_warmup):
        _ = X_T_gpu @ grad_gpu
        cp.cuda.Stream.null.synchronize()

    # ── Kernel timing with CuPy events ────────────────────────────────────
    print(f"\n  --- SpMV Kernel Timing ({n_iters} iterations) ---")
    times_wall = []
    times_event = []

    for _ in range(n_iters):
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()

        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        start_event.record()

        result_gpu = X_T_gpu @ grad_gpu

        end_event.record()
        end_event.synchronize()
        wall_ms = (time.perf_counter() - t0) * 1000
        event_ms = cp.cuda.get_elapsed_time(start_event, end_event)

        times_wall.append(wall_ms)
        times_event.append(event_ms)

    med_wall = np.median(times_wall)
    min_wall = np.min(times_wall)
    med_event = np.median(times_event)
    min_event = np.min(times_event)

    print(f"  Wall-clock:  Median={med_wall:.3f} ms | Min={min_wall:.3f} ms")
    print(f"  CUDA event:  Median={med_event:.3f} ms | Min={min_event:.3f} ms")

    # ── D2H Transfer ──────────────────────────────────────────────────────
    print("\n  --- Device-to-Host Transfer ---")
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    result_cpu = result_gpu.get()
    d2h_ms = (time.perf_counter() - t0) * 1000
    print(f"  Result download ({n_cols:,} floats): {d2h_ms:.2f} ms")

    # ── Bandwidth ─────────────────────────────────────────────────────────
    nnz = X.nnz
    bytes_read = nnz * (8 + 4) + (n_cols + 1) * 8 + n_rows * 8  # data+indices+indptr+vector
    bytes_written = n_cols * 8
    total_bytes = bytes_read + bytes_written
    eff_bw = total_bytes / (med_event / 1000) / 1e9

    print(f"\n  --- Effective Bandwidth ---")
    print(f"  Data moved: {total_bytes / 1e6:.1f} MB")
    print(f"  Bandwidth: {eff_bw:.1f} GB/s")
    print(f"  RTX 3090 peak: 936 GB/s -> {100 * eff_bw / 936:.1f}% utilization")

    return result_cpu, med_event, total_h2d_ms, d2h_ms, X_T_gpu, grad_gpu


def verify_correctness(cpu_result, gpu_result):
    """Compare GPU vs CPU histograms."""
    print(f"\n{'='*70}")
    print("CORRECTNESS VERIFICATION")
    print(f"{'='*70}")

    max_abs_err = np.max(np.abs(cpu_result - gpu_result))
    max_val = np.max(np.abs(cpu_result))
    rel_err = max_abs_err / (max_val + 1e-15)

    print(f"  Max absolute error: {max_abs_err:.2e}")
    print(f"  Max relative error: {rel_err:.2e}")
    print(f"  Max |CPU result|:   {max_val:.4f}")

    is_close = np.allclose(cpu_result, gpu_result, rtol=1e-10, atol=1e-12)
    if is_close:
        print("  np.allclose: PASS")
    else:
        n_mismatch = np.sum(~np.isclose(cpu_result, gpu_result, rtol=1e-10, atol=1e-12))
        print(f"  np.allclose: FAIL ({n_mismatch:,} / {len(cpu_result):,} elements differ)")

    # Looser check
    is_close_loose = np.allclose(cpu_result, gpu_result, rtol=1e-6, atol=1e-8)
    if not is_close and is_close_loose:
        print("  np.allclose (rtol=1e-6): PASS (acceptable for training)")

    assert is_close_loose, "GPU result diverges from CPU -- check data types"
    return is_close


def simulate_full_training(gpu_ms_per_call, cpu_ms_per_call, n_cols):
    """Simulate full 1w LightGBM training time from histogram calls."""
    print(f"\n{'='*70}")
    print("FULL TRAINING TIME SIMULATION")
    print(f"{'='*70}")

    # LightGBM 3-class training parameters
    n_rounds = 800
    n_classes = 3
    n_leaves = 63  # max_leaves default
    # Per tree: build histograms for ~half the leaves (rest via subtraction trick)
    n_hist_leaves = 32  # ~half of 63
    n_spmv_per_hist = 2  # grad + hess

    total_calls = n_rounds * n_classes * n_hist_leaves * n_spmv_per_hist
    print(f"\n  Training parameters:")
    print(f"    Boosting rounds:   {n_rounds}")
    print(f"    Classes:           {n_classes}")
    print(f"    Leaves per tree:   {n_leaves}")
    print(f"    Histogram leaves:  {n_hist_leaves} (rest via subtraction)")
    print(f"    SpMV per histogram: {n_spmv_per_hist} (grad + hess)")
    print(f"    ---")
    print(f"    Total SpMV calls:  {total_calls:,}")

    gpu_total_s = total_calls * gpu_ms_per_call / 1000
    cpu_total_s = total_calls * cpu_ms_per_call / 1000

    gpu_total_min = gpu_total_s / 60
    cpu_total_min = cpu_total_s / 60

    print(f"\n  Estimated histogram time (SpMV only):")
    print(f"    GPU: {total_calls:,} x {gpu_ms_per_call:.3f} ms = {gpu_total_s:.1f}s ({gpu_total_min:.1f} min)")
    print(f"    CPU: {total_calls:,} x {cpu_ms_per_call:.2f} ms = {cpu_total_s:.1f}s ({cpu_total_min:.1f} min)")
    print(f"    Speedup: {cpu_total_s / gpu_total_s:.1f}x")

    # CPCV multiplier (5 folds x 5 combinations = 25 total trains, but we run ~10)
    n_cpcv_folds = 10
    print(f"\n  With CPCV ({n_cpcv_folds} fold-combos):")
    print(f"    GPU: {gpu_total_min * n_cpcv_folds:.1f} min ({gpu_total_min * n_cpcv_folds / 60:.1f} hr)")
    print(f"    CPU: {cpu_total_min * n_cpcv_folds:.1f} min ({cpu_total_min * n_cpcv_folds / 60:.1f} hr)")

    # Note: this is histogram time only -- tree construction, EFB, etc. are separate
    print(f"\n  NOTE: This is histogram computation time ONLY.")
    print(f"  Actual training includes EFB bundling, tree construction,")
    print(f"  split finding, data subsetting, etc. Histogram building is")
    print(f"  typically 40-60% of total LightGBM training time.")

    # Estimate total training time
    hist_fraction = 0.50  # histogram is ~50% of training
    gpu_total_train_min = gpu_total_min / hist_fraction
    cpu_total_train_min = cpu_total_min / hist_fraction

    print(f"\n  Estimated TOTAL training time (assuming histograms = 50%):")
    print(f"    GPU: ~{gpu_total_train_min:.1f} min per full train")
    print(f"    CPU: ~{cpu_total_train_min:.1f} min per full train")
    print(f"    GPU + CPCV: ~{gpu_total_train_min * n_cpcv_folds:.0f} min ({gpu_total_train_min * n_cpcv_folds / 60:.1f} hr)")
    print(f"    CPU + CPCV: ~{cpu_total_train_min * n_cpcv_folds:.0f} min ({cpu_total_train_min * n_cpcv_folds / 60:.1f} hr)")


def memory_report(X):
    """Report VRAM usage and 3090 fit assessment."""
    n_rows, n_cols = X.shape
    nnz = X.nnz

    print(f"\n{'='*70}")
    print("MEMORY REPORT")
    print(f"{'='*70}")

    # CSR(A) storage
    data_bytes = nnz * 8  # float64
    indices_bytes = nnz * 4  # int32
    indptr_bytes = (n_rows + 1) * 8  # int64
    csr_a_bytes = data_bytes + indices_bytes + indptr_bytes

    # CSR(A^T) storage (same nnz, but indptr has n_cols+1 entries)
    indptr_t_bytes = (n_cols + 1) * 8
    csr_at_bytes = data_bytes + indices_bytes + indptr_t_bytes

    # Vectors
    grad_bytes = n_rows * 8
    hess_bytes = n_rows * 8
    output_bytes = n_cols * 8  # histogram result

    # Total for both approaches
    total_both = csr_a_bytes + csr_at_bytes + grad_bytes + hess_bytes + output_bytes
    total_at_only = csr_at_bytes + grad_bytes + hess_bytes + output_bytes

    # Actual VRAM measurement
    vram_free, vram_total = cp.cuda.Device(0).mem_info
    pool = cp.get_default_memory_pool()
    vram_used_pool = pool.used_bytes()
    vram_total_pool = pool.total_bytes()

    print(f"\n  Theoretical VRAM:")
    print(f"    CSR(A) data:       {data_bytes / 1e6:>8.1f} MB")
    print(f"    CSR(A) indices:    {indices_bytes / 1e6:>8.1f} MB")
    print(f"    CSR(A) indptr:     {indptr_bytes / 1e6:>8.1f} MB")
    print(f"    CSR(A) total:      {csr_a_bytes / 1e6:>8.1f} MB")
    print(f"    CSR(A^T) total:    {csr_at_bytes / 1e6:>8.1f} MB")
    print(f"    Gradient vector:   {grad_bytes / 1e6:>8.1f} MB")
    print(f"    Hessian vector:    {hess_bytes / 1e6:>8.1f} MB")
    print(f"    Output histogram:  {output_bytes / 1e6:>8.1f} MB")
    print(f"    ─────────────────────────────────")
    print(f"    Total (keep both): {total_both / 1e6:>8.1f} MB")
    print(f"    Total (A^T only):  {total_at_only / 1e6:>8.1f} MB")

    print(f"\n  Actual VRAM:")
    print(f"    Pool used:         {vram_used_pool / 1e6:>8.1f} MB")
    print(f"    Pool total:        {vram_total_pool / 1e6:>8.1f} MB")
    print(f"    Device free:       {vram_free / 1e6:>8.1f} MB")
    print(f"    Device total:      {vram_total / 1e6:>8.1f} MB")

    vram_3090 = 24 * 1024  # 24 GB in MB
    fits = total_at_only / 1e6 < vram_3090 * 0.85  # 85% to leave headroom

    print(f"\n  RTX 3090 (24 GB):")
    print(f"    Required (A^T only): {total_at_only / 1e6:.0f} MB")
    print(f"    Available (85%):     {vram_3090 * 0.85:.0f} MB")
    print(f"    Fits on 3090:        {'YES' if fits else 'NO'}")

    if not fits:
        print(f"    Overflow:            {total_at_only / 1e6 - vram_3090 * 0.85:.0f} MB over limit")
    else:
        headroom = vram_3090 * 0.85 - total_at_only / 1e6
        print(f"    Headroom:            {headroom:.0f} MB remaining")

    return fits


def main():
    print("=" * 70)
    print("REAL 1w DATA -- cuSPARSE SpMV Benchmark")
    print("GPU Histogram Building for Savage22 Binary Cross Features")
    print("=" * 70)

    # ── 1. Load real NPZ ──────────────────────────────────────────────────
    X, names_path = load_real_data()
    if X is None:
        print("\nCannot proceed without NPZ data. Exiting.")
        sys.exit(1)

    # Ensure float64 for histogram accuracy
    if X.dtype != np.float64:
        print(f"\n  Converting from {X.dtype} to float64...")
        X = X.astype(np.float64)

    # ── 2. Generate realistic gradients ───────────────────────────────────
    print(f"\n{'='*70}")
    print("GENERATING REALISTIC GRADIENTS (3-class LightGBM)")
    print(f"{'='*70}")
    grad, hess = generate_realistic_gradients(X.shape[0])
    print(f"  grad: shape={grad.shape}, dtype={grad.dtype}, range=[{grad.min():.3f}, {grad.max():.3f}]")
    print(f"  hess: shape={hess.shape}, dtype={hess.dtype}, range=[{hess.min():.3f}, {hess.max():.3f}]")

    # ── 3. CPU baseline ───────────────────────────────────────────────────
    print("\nPre-transposing on CPU...")
    t0 = time.perf_counter()
    X_T_cpu = X.T.tocsr()
    cpu_transpose_ms = (time.perf_counter() - t0) * 1000
    print(f"  CPU X.T.tocsr(): {cpu_transpose_ms:.1f} ms")
    print(f"  X_T shape: {X_T_cpu.shape[0]:,} x {X_T_cpu.shape[1]:,}")

    cpu_result, cpu_median_ms = benchmark_cpu(X_T_cpu, grad, n_iters=50)

    # ── 4. GPU cuSPARSE ──────────────────────────────────────────────────
    gpu_result, gpu_median_ms, h2d_ms, d2h_ms, X_T_gpu, grad_gpu = benchmark_gpu(
        X, grad, n_warmup=10, n_iters=50
    )

    # ── 5. Correctness ────────────────────────────────────────────────────
    verify_correctness(cpu_result, gpu_result)

    # ── 6. Hessian SpMV (both grad + hess needed per histogram) ──────────
    print(f"\n{'='*70}")
    print("HESSIAN SpMV (second half of histogram)")
    print(f"{'='*70}")

    hess_gpu = cp.asarray(hess)

    # Warmup
    for _ in range(10):
        _ = X_T_gpu @ hess_gpu
        cp.cuda.Stream.null.synchronize()

    times_hess = []
    for _ in range(50):
        start_ev = cp.cuda.Event()
        end_ev = cp.cuda.Event()
        cp.cuda.Stream.null.synchronize()
        start_ev.record()
        _ = X_T_gpu @ hess_gpu
        end_ev.record()
        end_ev.synchronize()
        times_hess.append(cp.cuda.get_elapsed_time(start_ev, end_ev))

    med_hess = np.median(times_hess)
    print(f"  Hessian SpMV: Median={med_hess:.3f} ms (vs grad {gpu_median_ms:.3f} ms)")
    print(f"  Combined (grad+hess): {gpu_median_ms + med_hess:.3f} ms per histogram")

    # ── 7. Simulate full training ─────────────────────────────────────────
    simulate_full_training(gpu_median_ms, cpu_median_ms, X.shape[1])

    # ── 8. Memory report ──────────────────────────────────────────────────
    fits = memory_report(X)

    # ── Summary ───────────────────────────────────────────────────────────
    speedup = cpu_median_ms / gpu_median_ms
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Matrix:    {X.shape[0]:,} x {X.shape[1]:,} ({X.nnz:,} NNZ)")
    print(f"  CPU SpMV:  {cpu_median_ms:.2f} ms (scipy, pre-transposed)")
    print(f"  GPU SpMV:  {gpu_median_ms:.3f} ms (cuSPARSE, pre-transposed)")
    print(f"  Speedup:   {speedup:.0f}x")
    print(f"  H2D setup: {h2d_ms:.1f} ms (one-time)")
    print(f"  D2H:       {d2h_ms:.2f} ms")
    print(f"  Fits 3090: {'YES' if fits else 'NO'}")
    print(f"{'='*70}")

    # Cleanup
    del X_T_gpu, grad_gpu, hess_gpu
    cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
