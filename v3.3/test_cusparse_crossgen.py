#!/usr/bin/env python3
"""
cuSPARSE Cross-Gen Benchmark
Tests CPU (scipy) vs GPU (cuSPARSE via CuPy) sparse matrix multiplication
simulating cross-feature generation: C = A.T @ B
"""

import time
import sys
import numpy as np
import scipy.sparse as sp

# Hard fail if CuPy not available - NO fallback
import cupy as cp
import cupyx.scipy.sparse as cpx_sp


def create_synthetic_binary_csr(rows, cols, density, seed=42):
    """Create a binary sparse CSR matrix with given density."""
    rng = np.random.RandomState(seed)
    nnz = int(rows * cols * density)
    row_idx = rng.randint(0, rows, size=nnz)
    col_idx = rng.randint(0, cols, size=nnz)
    data = np.ones(nnz, dtype=np.float64)
    mat = sp.csr_matrix((data, (row_idx, col_idx)), shape=(rows, cols))
    # Deduplicate and ensure binary
    mat.data[:] = 1.0
    mat.sum_duplicates()
    return mat


def get_gpu_mem():
    """Return (used_mb, total_mb) VRAM."""
    free, total = cp.cuda.runtime.memGetInfo()
    used = (total - free) / (1024 ** 2)
    total_mb = total / (1024 ** 2)
    return used, total_mb


def benchmark_cpu(A, B, warmup=1, runs=3):
    """Benchmark CPU sparse matmul: A.T @ B"""
    # Warmup
    for _ in range(warmup):
        _ = (A.T @ B).tocsr()

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        C = (A.T @ B).tocsr()
        times.append(time.perf_counter() - t0)
    return C, times


def benchmark_gpu(A, B, warmup=2, runs=5):
    """Benchmark GPU sparse matmul: A.T @ B via cuSPARSE"""
    A_gpu = cpx_sp.csr_matrix(A.astype(np.float32))
    B_gpu = cpx_sp.csr_matrix(B.astype(np.float32))
    cp.cuda.Stream.null.synchronize()

    # Warmup
    for _ in range(warmup):
        _ = (A_gpu.T @ B_gpu).tocsr()
        cp.cuda.Stream.null.synchronize()

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        C_gpu = (A_gpu.T @ B_gpu).tocsr()
        cp.cuda.Stream.null.synchronize()
        times.append(time.perf_counter() - t0)
    return C_gpu, A_gpu, B_gpu, times


def verify_results(C_cpu, C_gpu):
    """Verify CPU and GPU results match exactly."""
    C_check = C_gpu.get().tocsr().astype(np.float64)
    diff = (C_cpu - C_check)
    diff.eliminate_zeros()
    assert diff.nnz == 0, f"RESULTS DIFFER! {diff.nnz} elements mismatch"
    print("  VERIFIED: CPU and GPU results match exactly")


def report(label, A, B, C, cpu_times, gpu_times, vram_before, vram_after):
    """Print benchmark results."""
    cpu_med = np.median(cpu_times)
    gpu_med = np.median(gpu_times)
    speedup = cpu_med / gpu_med if gpu_med > 0 else float('inf')

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  A shape:     {A.shape}  NNZ: {A.nnz:,}")
    print(f"  B shape:     {B.shape}  NNZ: {B.nnz:,}")
    print(f"  C shape:     {C.shape}  NNZ: {C.nnz:,}")
    print(f"  A density:   {A.nnz / (A.shape[0] * A.shape[1]) * 100:.3f}%")
    print(f"  B density:   {B.nnz / (B.shape[0] * B.shape[1]) * 100:.3f}%")
    print(f"{'─' * 60}")
    print(f"  CPU times:   {[f'{t:.4f}s' for t in cpu_times]}")
    print(f"  CPU median:  {cpu_med:.4f}s")
    print(f"  GPU times:   {[f'{t:.4f}s' for t in gpu_times]}")
    print(f"  GPU median:  {gpu_med:.4f}s")
    print(f"  SPEEDUP:     {speedup:.1f}x")
    print(f"{'─' * 60}")
    print(f"  VRAM before: {vram_before:.0f} MB")
    print(f"  VRAM after:  {vram_after:.0f} MB")
    print(f"  VRAM delta:  {vram_after - vram_before:.0f} MB")
    print(f"{'=' * 60}")
    return speedup


def run_benchmark(label, rows, cols, density, seed=42):
    """Run full CPU vs GPU benchmark for given matrix dimensions."""
    print(f"\n>>> Creating matrices: {rows:,} x {cols:,} @ {density*100:.1f}% density ...")
    A = create_synthetic_binary_csr(rows, cols, density, seed=seed)
    B = create_synthetic_binary_csr(rows, cols, density, seed=seed + 1)
    print(f"    A: {A.shape}, NNZ={A.nnz:,}  |  B: {B.shape}, NNZ={B.nnz:,}")

    # CPU benchmark
    print(">>> Running CPU benchmark (scipy) ...")
    C_cpu, cpu_times = benchmark_cpu(A, B)

    # GPU benchmark
    vram_before, vram_total = get_gpu_mem()
    print(f">>> Running GPU benchmark (cuSPARSE) ... [VRAM total: {vram_total:.0f} MB]")
    C_gpu, A_gpu, B_gpu, gpu_times = benchmark_gpu(A, B)
    vram_after, _ = get_gpu_mem()

    # Verify
    print(">>> Verifying results ...")
    verify_results(C_cpu, C_gpu)

    # Report
    speedup = report(label, A, B, C_cpu, cpu_times, gpu_times, vram_before, vram_after)

    # Cleanup GPU memory
    del A_gpu, B_gpu, C_gpu
    cp.get_default_memory_pool().free_all_blocks()

    return speedup


def main():
    print("cuSPARSE Cross-Gen Benchmark")
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    free, total = cp.cuda.runtime.memGetInfo()
    print(f"VRAM: {free / 1024**2:.0f} MB free / {total / 1024**2:.0f} MB total")
    print(f"CuPy version: {cp.__version__}")
    import scipy; print(f"SciPy version: {scipy.__version__}")

    results = {}

    # --- Test 1: 1d-scale (small) ---
    results['1d'] = run_benchmark(
        label="1D SCALE (small cross gen)",
        rows=5733, cols=1000, density=0.003,
    )

    # --- Test 2: 4h-scale (medium) ---
    results['4h'] = run_benchmark(
        label="4H SCALE (medium cross gen)",
        rows=23000, cols=1500, density=0.003,
    )

    # --- Test 3: 15m-scale (large - actual bottleneck) ---
    # Check if we have enough VRAM for the large test
    free_mb = cp.cuda.runtime.memGetInfo()[0] / (1024 ** 2)
    if free_mb < 4000:
        print(f"\n>>> SKIPPING 15m scale: only {free_mb:.0f} MB VRAM free (need ~4 GB)")
    else:
        results['15m'] = run_benchmark(
            label="15M SCALE (actual bottleneck)",
            rows=227000, cols=2300, density=0.003,
        )

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    for scale, speedup in results.items():
        print(f"  {scale:>4s}: {speedup:.1f}x GPU speedup")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
