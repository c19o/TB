#!/usr/bin/env python3
"""Savage22 GPU Histogram Quick Benchmark — standalone, <30s, any machine."""
import os, sys, time, platform
import numpy as np
from scipy import sparse

# Windows DLL path for CUDA
if platform.system() == "Windows":
    for p in [r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
              r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",
              r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin"]:
        if os.path.isdir(p):
            os.add_dll_directory(p)

ROWS, COLS, DENSITY = 1_000, 100_000, 0.003
NNZ = int(ROWS * COLS * DENSITY)
CPU_ITERS, GPU_ITERS = 10, 50

TF_ESTIMATES = {  # (rows, cols) for histogram ratio estimation
    "1w":  (260,   800_000),
    "1d":  (1800,  1_500_000),
    "4h":  (10800, 1_200_000),
}

def make_sparse(rows, cols, nnz):
    rng = np.random.default_rng(42)
    r = rng.integers(0, rows, nnz)
    c = rng.integers(0, cols, nnz)
    d = rng.standard_normal(nnz).astype(np.float64)
    return sparse.csr_matrix((d, (r, c)), shape=(rows, cols))

def cpu_bench(mat, vec, iters):
    # Warm up
    mat.T @ vec
    t0 = time.perf_counter()
    for _ in range(iters):
        mat.T @ vec
    return (time.perf_counter() - t0) / iters

def gpu_bench(mat, vec, iters):
    import cupy as cp
    from cupyx.scipy import sparse as csp
    g_mat = csp.csr_matrix(mat.astype(np.float32)).T.tocsr()
    g_vec = cp.asarray(vec.astype(np.float32))
    # Warm up
    for _ in range(5):
        g_mat @ g_vec
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        g_mat @ g_vec
    cp.cuda.Device().synchronize()
    elapsed = (time.perf_counter() - t0) / iters
    result = cp.asnumpy(g_mat @ g_vec)
    return elapsed, result

def get_gpu_info():
    import cupy as cp
    dev = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
    mem_gb = props["totalGlobalMem"] / 1e9
    sm = f"sm_{props['major']}{props['minor']}"
    bw = 2 * props["memoryClockRate"] * 1e3 * (props["memoryBusWidth"] / 8) / 1e9
    return name, mem_gb, sm, bw

def main():
    print("=== Savage22 GPU Histogram Quick Benchmark ===")

    # Build matrix
    mat = make_sparse(ROWS, COLS, NNZ)
    actual_nnz = mat.nnz
    vec = np.random.default_rng(99).standard_normal(ROWS)
    print(f"Matrix: {ROWS:,} x {COLS:,} ({actual_nnz:,} nnz)\n")

    # CPU
    cpu_ms = cpu_bench(mat, vec, CPU_ITERS) * 1000
    cpu_result = (mat.T @ vec)
    print(f"CPU (scipy):    {cpu_ms:>8.2f} ms/call")

    # GPU
    has_gpu = False
    try:
        import cupy as cp
        name, mem_gb, sm, peak_bw = get_gpu_info()
        print(f"GPU: {name} ({mem_gb:.0f}GB, {sm})")
        gpu_ms, gpu_result = gpu_bench(mat, vec, GPU_ITERS)
        gpu_ms *= 1000
        has_gpu = True
    except Exception as e:
        print(f"GPU: not available ({e})")

    if has_gpu:
        speedup = cpu_ms / gpu_ms
        # Bandwidth: bytes moved = nnz*(8+4) + rows*8 + cols*8 for SpMV
        data_bytes = actual_nnz * (4 + 4) + ROWS * 4 + COLS * 4
        bw_gbs = (data_bytes / (gpu_ms / 1000)) / 1e9
        bw_pct = (bw_gbs / peak_bw * 100) if peak_bw > 0 else 0
        max_err = np.max(np.abs(cpu_result - gpu_result.astype(np.float64)))
        correct = "PASS" if max_err < 1e-4 else "FAIL"

        print(f"GPU (cuSPARSE): {gpu_ms:>8.4f} ms/call")
        print(f"Speedup: {speedup:.0f}x")
        print(f"Bandwidth: {bw_gbs:.0f} GB/s ({bw_pct:.0f}% of peak)")
        print(f"Correctness: {correct} (max error: {max_err:.1e})\n")

        print("Estimated training speedup per TF:")
        for tf, (r, c) in TF_ESTIMATES.items():
            # Histogram is ~40-60% of LightGBM time; GPU speeds up that portion
            ratio = min(speedup * (r / ROWS), speedup * 2)
            hist_frac = 0.5
            total_speedup = 1 / (1 - hist_frac + hist_frac / ratio)
            print(f"  {tf}: ~{ratio:.0f}x histogram -> ~{total_speedup:.1f}x total")
    else:
        print("\nSkipping GPU benchmark (no CUDA device).")
        print(f"CPU-only result: {cpu_ms:.2f} ms/call for {COLS:,}-col SpMV")

if __name__ == "__main__":
    main()
