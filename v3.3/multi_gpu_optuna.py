#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
multi_gpu_optuna.py -- Multi-GPU trial-level parallelism for Optuna HPO (v3.3)
================================================================================
Detects available CUDA GPUs and assigns each Optuna trial to a separate GPU
via round-robin. 4 GPUs = 4 concurrent trials, each on its own device.

Architecture: Trial-level parallelism (NOT data-parallel training).
Each trial sees ALL features and ALL rows — no partitioning.

Usage:
    from multi_gpu_optuna import detect_gpus, get_multi_gpu_config, apply_gpu_params

    gpu_cfg = get_multi_gpu_config()
    # gpu_cfg.num_gpus, gpu_cfg.enabled, gpu_cfg.n_jobs, gpu_cfg.device_type

    params = apply_gpu_params(params, trial_number, gpu_cfg)
    # Sets device_type, gpu_device_id, thread count

Env vars:
    MULTI_GPU=1         Force enable (0 = force disable, unset = auto-detect)
    LGBM_NUM_GPUS=N     Override GPU count (skips detection)
"""
import os
import logging
import threading

log = logging.getLogger(__name__)

# Thread-local storage for GPU assignment logging
_gpu_lock = threading.Lock()
_gpu_trial_map = {}  # trial_number -> gpu_id


class MultiGPUConfig:
    """Configuration for multi-GPU Optuna execution."""
    __slots__ = ('num_gpus', 'enabled', 'device_type', 'n_jobs',
                 'threads_per_trial', 'gpu_names')

    def __init__(self, num_gpus=0, enabled=False, device_type='cpu',
                 n_jobs=1, threads_per_trial=0, gpu_names=None):
        self.num_gpus = num_gpus
        self.enabled = enabled
        self.device_type = device_type
        self.n_jobs = n_jobs
        self.threads_per_trial = threads_per_trial
        self.gpu_names = gpu_names or []

    def __repr__(self):
        return (f"MultiGPUConfig(gpus={self.num_gpus}, enabled={self.enabled}, "
                f"device={self.device_type}, n_jobs={self.n_jobs}, "
                f"threads/trial={self.threads_per_trial})")


def detect_gpus():
    """Detect available CUDA GPUs via torch.cuda or nvidia-smi fallback.

    Returns:
        (num_gpus, gpu_names) tuple. num_gpus=0 if no CUDA GPUs found.
    """
    # Method 1: PyTorch (most reliable)
    try:
        import torch
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            names = [torch.cuda.get_device_name(i) for i in range(n)]
            return n, names
    except ImportError:
        pass

    # Method 2: nvidia-smi (fallback for non-PyTorch environments)
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            names = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
            return len(names), names
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    return 0, []


def _detect_lgbm_device_type():
    """Detect the best available LightGBM GPU device type.

    Tests cuda_sparse first (our GPU fork), then gpu (OpenCL), then cpu.
    Raises RuntimeError if no GPU device works — no silent CPU fallback.
    """
    import numpy as np
    import lightgbm as lgb

    test_X = np.random.rand(20, 10).astype(np.float32)
    test_y = np.random.randint(0, 2, 20)
    test_ds = lgb.Dataset(test_X, label=test_y, params={'feature_pre_filter': False})

    for dtype in ('cuda_sparse', 'gpu'):
        try:
            lgb.train({
                'objective': 'binary', 'device_type': dtype,
                'gpu_device_id': 0, 'num_iterations': 1, 'verbose': -1,
            }, test_ds)
            log.info(f"  LightGBM device_type='{dtype}' verified working")
            return dtype
        except Exception as e:
            log.info(f"  LightGBM device_type='{dtype}' not available: {e}")

    raise RuntimeError(
        "LightGBM has NO working GPU device type (tried cuda_sparse, gpu). "
        "Install GPU fork (cmake -DUSE_CUDA_SPARSE=ON) or pip install lightgbm --config-settings=cmake.define.USE_GPU=ON. "
        "CPU fallback is not allowed."
    )


def get_multi_gpu_config(total_cores=None):
    """Build multi-GPU configuration based on environment and hardware.

    Respects env vars:
        MULTI_GPU=1/0     Force enable/disable (unset = auto-detect)
        LGBM_NUM_GPUS=N   Override detected GPU count

    Args:
        total_cores: CPU core count (for thread allocation). Auto-detected if None.

    Returns:
        MultiGPUConfig instance
    """
    if total_cores is None:
        total_cores = os.cpu_count() or 8

    # Check force toggle
    multi_gpu_env = os.environ.get('MULTI_GPU')
    if multi_gpu_env == '0':
        log.info("  MULTI_GPU=0: multi-GPU disabled by env var")
        return MultiGPUConfig(num_gpus=0, enabled=False, device_type='cpu',
                              n_jobs=max(1, total_cores // 8))

    # Check explicit GPU count override
    lgbm_num_gpus = int(os.environ.get('LGBM_NUM_GPUS', '0'))
    if lgbm_num_gpus > 0:
        num_gpus = lgbm_num_gpus
        gpu_names = [f'GPU-{i}' for i in range(num_gpus)]
        log.info(f"  LGBM_NUM_GPUS={num_gpus}: using override count")
    else:
        num_gpus, gpu_names = detect_gpus()

    if num_gpus == 0:
        if multi_gpu_env == '1':
            log.warning("  MULTI_GPU=1 but no GPUs detected — falling back to CPU")
        return MultiGPUConfig(num_gpus=0, enabled=False, device_type='cpu',
                              n_jobs=max(1, total_cores // 8))

    # Auto-detect: enable if >1 GPU, or if MULTI_GPU=1 with 1 GPU
    if multi_gpu_env == '1':
        enabled = True
    else:
        # Auto: enable for 2+ GPUs only; single GPU doesn't need multi-GPU scheduling
        enabled = num_gpus >= 2

    # Determine device type — cuda_sparse for sparse CSR data (our default)
    # Verify cuda_sparse is actually supported by this LightGBM build
    device_type = _detect_lgbm_device_type()

    # Thread allocation: split CPU cores across GPU trials
    # Each trial gets a fair share of cores for data loading / preprocessing
    n_jobs = num_gpus if enabled else 1
    threads_per_trial = max(1, total_cores // n_jobs)

    log.info(f"  Multi-GPU: {num_gpus} GPUs detected, n_jobs={n_jobs}, "
             f"{threads_per_trial} threads/trial")
    for i, name in enumerate(gpu_names):
        log.info(f"    GPU {i}: {name}")

    return MultiGPUConfig(
        num_gpus=num_gpus,
        enabled=enabled,
        device_type=device_type,
        n_jobs=n_jobs,
        threads_per_trial=threads_per_trial,
        gpu_names=gpu_names,
    )


def apply_gpu_params(params, trial_number, gpu_cfg):
    """Apply GPU-specific parameters to a LightGBM param dict for one trial.

    Round-robin assigns trial to GPU. Logs the assignment.

    Args:
        params: LightGBM parameter dict (modified in-place and returned)
        trial_number: Optuna trial.number (for round-robin)
        gpu_cfg: MultiGPUConfig instance

    Returns:
        params dict with GPU settings applied
    """
    if not gpu_cfg.enabled or gpu_cfg.num_gpus == 0:
        return params

    gpu_id = trial_number % gpu_cfg.num_gpus
    gpu_name = gpu_cfg.gpu_names[gpu_id] if gpu_id < len(gpu_cfg.gpu_names) else f'GPU-{gpu_id}'

    # Log GPU assignment (thread-safe)
    with _gpu_lock:
        _gpu_trial_map[trial_number] = gpu_id
        log.info(f"  Trial #{trial_number} → GPU {gpu_id} ({gpu_name})")

    params['device_type'] = gpu_cfg.device_type
    params['gpu_device_id'] = gpu_id
    params['histogram_pool_size'] = 512  # MB — GPU histogram memory

    # GPU mode is incompatible with force_col_wise/force_row_wise
    params.pop('force_col_wise', None)
    params.pop('force_row_wise', None)
    params.pop('device', None)  # remove legacy 'device' key

    # Thread count: fair share of CPU cores
    params['num_threads'] = gpu_cfg.threads_per_trial

    return params


def create_gpu_safe_sampler(seed, n_startup_trials):
    """Create TPESampler with constant_liar for thread-safe parallel sampling.

    constant_liar=True prevents the sampler from waiting for trial completion
    before suggesting the next trial — essential for multi-GPU parallelism.

    Args:
        seed: Random seed for reproducibility
        n_startup_trials: Number of random trials before TPE kicks in

    Returns:
        optuna.samplers.TPESampler instance
    """
    import optuna

    return optuna.samplers.TPESampler(
        seed=seed,
        n_startup_trials=n_startup_trials,
        multivariate=True,
        group=True,
        constant_liar=True,
    )


def gpu_oom_handler(func):
    """Decorator that catches CUDA OOM errors and reports them as failed Optuna trials.

    On OOM: logs the error, frees GPU memory, returns infinity (worst score).
    The trial is NOT pruned — it's marked as failed so Optuna avoids that region.
    """
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            err_str = str(e).lower()
            if 'out of memory' in err_str or ('cuda' in err_str and ('alloc' in err_str or 'memory' in err_str)):
                log.warning(f"  GPU OOM in trial: {e}")
                _free_gpu_memory()
                return float('inf')  # worst possible score
            raise
        except Exception as e:
            err_str = str(e).lower()
            if 'cuda' in err_str and ('memory' in err_str or 'alloc' in err_str):
                log.warning(f"  GPU memory error in trial: {e}")
                _free_gpu_memory()
                return float('inf')
            raise

    return wrapper


def _free_gpu_memory():
    """Best-effort GPU memory cleanup after OOM."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def clear_gpu_trial_map():
    """Clear GPU trial assignments between timeframes to prevent stale state."""
    with _gpu_lock:
        _gpu_trial_map.clear()
    log.debug("  GPU trial map cleared")


def get_gpu_trial_summary():
    """Return a summary of GPU assignments for logging.

    Returns:
        dict mapping gpu_id -> list of trial numbers that ran on it
    """
    with _gpu_lock:
        summary = {}
        for trial_num, gpu_id in _gpu_trial_map.items():
            summary.setdefault(gpu_id, []).append(trial_num)
        return summary
