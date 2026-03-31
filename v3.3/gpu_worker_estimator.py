"""Pre-launch memory estimator for multi-GPU cross generation workers.
Calculates peak VRAM and host RAM per worker based on data dimensions.
Used by v2_cross_generator.py to set max_concurrent BEFORE launching workers."""

import numpy as np
import os

# Empirical constants (calibrate on first run)
_CUDA_CTX_MB = 620       # CUDA context overhead per process
_POOL_FRAG_FACTOR = 1.18  # CuPy memory pool fragmentation
_POOL_BASELINE_MB = 256   # CuPy pool baseline reserve
_PYTHON_OVERHEAD_MB = 205  # Python + scipy + numpy per process
_KERNEL_SAFETY = 3.0       # worst-case column nnz / average nnz


def estimate_worker_memory(n_rows, n_cols_left, n_cols_right, density, n_valid_pairs,
                           use_sparse_kernel=True):
    """Estimate peak VRAM and host RAM per worker in GB.

    Args:
        n_rows: number of data rows (5727 for 1d, 228000 for 15m)
        n_cols_left: left signal columns
        n_cols_right: right signal columns
        density: fraction of nonzeros (typically 0.1-0.3 for binary features)
        n_valid_pairs: number of valid cross pairs per worker
        use_sparse_kernel: True for new CSC two-pointer kernel, False for dense

    Returns:
        dict with 'vram_gb', 'host_ram_gb', 'total_gb'
    """
    n_cols = n_cols_left + n_cols_right
    nnz = int(n_rows * n_cols * density)
    avg_nnz_per_col = int(n_rows * density)

    if use_sparse_kernel:
        # Sparse kernel: O(nnz) memory
        # CSC upload: indptr + indices + data
        csc_bytes = (n_cols + 1) * 8 + nnz * 4 + nnz * 4
        # Kernel workspace: result buffer per batch
        max_nnz = int(avg_nnz_per_col * _KERNEL_SAFETY)
        batch_size = min(25000, n_valid_pairs)
        kernel_bytes = batch_size * max_nnz * 4 * 3  # result + counts + pairs
        vram_bytes = csc_bytes + kernel_bytes
    else:
        # Dense kernel: O(rows * cols) memory
        vram_bytes = n_rows * n_cols_right * 4 * 3  # left + right + result

    # Add CUDA context + pool overhead
    vram_bytes = int(vram_bytes * _POOL_FRAG_FACTOR)
    vram_bytes += (_CUDA_CTX_MB + _POOL_BASELINE_MB) * 1024**2

    # Host RAM: scipy CSC staging + python overhead
    host_bytes = nnz * 12 + (n_cols + 1) * 8  # CSC on host
    host_bytes += _PYTHON_OVERHEAD_MB * 1024**2

    return {
        'vram_gb': vram_bytes / 1024**3,
        'host_ram_gb': host_bytes / 1024**3,
        'total_gb': (vram_bytes + host_bytes) / 1024**3,
        'n_rows': n_rows,
        'nnz': nnz,
        'use_sparse_kernel': use_sparse_kernel,
    }


def plan_workers(n_rows, n_cols_left, n_cols_right, density, n_valid_pairs,
                 n_gpus=8, gpu_vram_gb=24, host_ram_gb=774,
                 host_headroom_fraction=0.05, use_sparse_kernel=True):
    """Calculate max safe concurrent workers for given hardware.

    Returns:
        dict with 'max_concurrent', 'per_worker', 'total_8_workers', 'fits_all'
    """
    est = estimate_worker_memory(n_rows, n_cols_left, n_cols_right, density,
                                  n_valid_pairs, use_sparse_kernel)

    host_usable = host_ram_gb * (1 - host_headroom_fraction)
    max_by_host = int(host_usable / max(est['host_ram_gb'], 0.1))
    max_by_vram = int(gpu_vram_gb / max(est['vram_gb'], 0.1))
    max_concurrent = min(n_gpus, max_by_host, max_by_vram * n_gpus)

    return {
        'max_concurrent': max_concurrent,
        'per_worker_vram_gb': est['vram_gb'],
        'per_worker_host_gb': est['host_ram_gb'],
        'total_8_workers_host_gb': est['host_ram_gb'] * 8,
        'total_8_workers_vram_gb': est['vram_gb'] * 8,
        'fits_all_8': max_concurrent >= n_gpus,
        'host_headroom_gb': host_usable - est['host_ram_gb'] * min(max_concurrent, n_gpus),
    }


def preflight_check(tf_name, n_rows, n_cols_left, n_cols_right, density, n_valid_pairs,
                    n_gpus=8, gpu_vram_gb=24, host_ram_gb=None):
    """Run before cross gen. Prints estimate and returns max_concurrent.
    Auto-detects host RAM if not provided."""
    if host_ram_gb is None:
        try:
            import psutil
            host_ram_gb = psutil.virtual_memory().total / 1024**3
        except ImportError:
            host_ram_gb = 64  # conservative fallback

    plan = plan_workers(n_rows, n_cols_left, n_cols_right, density, n_valid_pairs,
                        n_gpus=n_gpus, gpu_vram_gb=gpu_vram_gb, host_ram_gb=host_ram_gb)

    print(f"  [MEMORY EST] {tf_name}: {n_rows} rows, {n_valid_pairs:,} pairs, "
          f"density={density:.2f}")
    print(f"    Per worker: VRAM={plan['per_worker_vram_gb']:.2f}GB, "
          f"Host={plan['per_worker_host_gb']:.2f}GB")
    print(f"    8 workers: VRAM={plan['total_8_workers_vram_gb']:.1f}GB, "
          f"Host={plan['total_8_workers_host_gb']:.1f}GB")
    print(f"    Machine: {host_ram_gb:.0f}GB host, {n_gpus}x {gpu_vram_gb}GB GPU")
    print(f"    → Max concurrent: {plan['max_concurrent']} workers "
          f"({'ALL 8' if plan['fits_all_8'] else 'THROTTLED'})")

    return plan['max_concurrent']
