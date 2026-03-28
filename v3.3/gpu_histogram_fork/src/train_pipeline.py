"""
GPU Histogram Training Pipeline Integration
============================================
High-level integration that makes GPU histogram training work with
ml_multi_tf.py's CPCV pipeline.

Two integration paths:
  1. NATIVE: LightGBM built with USE_CUDA_SPARSE — set device_type='cuda_sparse'
     in params. LightGBM handles everything internally (EFB, histogram, etc.)
  2. CO-PROCESSOR: Our custom libgpu_histogram.so replaces ConstructHistograms()
     only. Used via gpu_train() from lgbm_integration.py.

This module provides:
  - get_training_params()  — the main entry point for ml_multi_tf.py
  - dry_run_report()       — reports what WOULD happen without using GPU
  - estimate_vram_need()   — VRAM estimation from CSR matrix properties
  - GPU detection and fallback logging

Usage in ml_multi_tf.py (minimal change):
  # Before CPCV loop, after params setup:
  from gpu_histogram_fork.src.train_pipeline import get_training_params
  params = get_training_params(_base_lgb_params, X_all, tf_name=tf_name)
  # Everything else stays the same — lgb.train() with the modified params
"""

import logging
import os
import sys
from typing import Any, Optional, Union

import numpy as np

try:
    import scipy.sparse as sp_sparse
except ImportError:
    sp_sparse = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# VRAM budgets per timeframe from ARCHITECTURE.md section 5.5
# Includes: CSR resident + histogram pool + gradients + kernel overhead
# ---------------------------------------------------------------------------
TF_VRAM_BUDGET_GB = {
    '1w':   3.0,    #  818 rows x 2.2M features  ->  ~2 GB CSR + overhead
    '1d':   6.0,    # 5733 rows x 6M features    ->  ~5 GB CSR + overhead
    '4h':  13.0,    #  23K rows x 4M features     -> ~12 GB CSR + overhead
    '1h':  26.0,    # 100K rows x 10M features    -> ~25 GB CSR + overhead
    '15m': 41.0,    # 227K rows x 10M features    -> ~40 GB CSR + overhead
}

# GPU recommendations per timeframe
TF_GPU_RECOMMENDATIONS = {
    '1w':  'RTX 3090 (24GB) or higher',
    '1d':  'RTX 3090 (24GB) or higher',
    '4h':  'A40 (48GB) or higher',
    '1h':  'A100 80GB or H100',
    '15m': 'A100 80GB, H100, or B200 (192GB)',
}

# Safety margin — never allocate more than 85% of VRAM
_VRAM_SAFETY = 0.85

# Environment variable to force CPU (useful for debugging)
_ENV_FORCE_CPU = 'GPU_HISTOGRAM_FORCE_CPU'


# ---------------------------------------------------------------------------
# GPU Detection
# ---------------------------------------------------------------------------

def _detect_cuda_devices() -> list[dict]:
    """Detect all CUDA devices with properties.

    Returns list of dicts with keys:
        id, name, vram_total_bytes, vram_free_bytes, vram_total_gb,
        vram_free_gb, compute_capability
    """
    try:
        import cupy as cp
    except ImportError:
        return []

    devices = []
    try:
        n = cp.cuda.runtime.getDeviceCount()
    except Exception:
        return []

    for i in range(n):
        try:
            with cp.cuda.Device(i):
                props = cp.cuda.runtime.getDeviceProperties(i)
                free, total = cp.cuda.runtime.memGetInfo()
                name = props['name']
                if isinstance(name, bytes):
                    name = name.decode()
                devices.append({
                    'id': i,
                    'name': str(name),
                    'vram_total_bytes': total,
                    'vram_free_bytes': free,
                    'vram_total_gb': round(total / (1024 ** 3), 1),
                    'vram_free_gb': round(free / (1024 ** 3), 1),
                    'compute_capability': (props.get('major', 0), props.get('minor', 0)),
                })
        except Exception as e:
            log.debug("Failed to query GPU %d: %s", i, e)

    return devices


def _get_free_vram(device_id: int = 0) -> int:
    """Get free VRAM in bytes for a specific device. Returns 0 if unavailable."""
    try:
        import cupy as cp
        with cp.cuda.Device(device_id):
            free, _ = cp.cuda.runtime.memGetInfo()
            return free
    except Exception:
        return 0


def _check_cuda_sparse_available() -> bool:
    """Check if LightGBM was built with CUDA sparse histogram support.

    Tests by creating a tiny dataset and attempting to train with
    device_type='cuda_sparse'. If the build doesn't support it,
    LightGBM raises an exception.

    Returns True if cuda_sparse is available, False otherwise.
    """
    if lgb is None:
        return False

    try:
        import cupy as cp
        # Just check CUDA is accessible at all
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        return False

    try:
        # Tiny test to probe LightGBM build capabilities
        rng = np.random.RandomState(42)
        X_test = rng.rand(20, 3).astype(np.float32)
        y_test = rng.randint(0, 3, size=20)
        ds = lgb.Dataset(X_test, label=y_test, free_raw_data=False)

        test_params = {
            'objective': 'multiclass',
            'num_class': 3,
            'device_type': 'cuda_sparse',
            'num_threads': 1,
            'verbose': -1,
            'num_iterations': 1,
            'num_leaves': 4,
        }
        lgb.train(test_params, ds, num_boost_round=1)
        return True
    except Exception as e:
        log.debug("cuda_sparse not available: %s", e)
        return False


def _check_coprocessor_available() -> bool:
    """Check if our custom GPU histogram co-processor library is available."""
    try:
        from pathlib import Path
        import ctypes.util

        lib_name = 'libgpu_histogram'
        lib_so = f'{lib_name}.so'
        lib_dll = f'{lib_name}.dll'

        this_dir = Path(__file__).parent
        candidates = [
            this_dir / lib_so,
            this_dir / lib_dll,
            this_dir.parent / 'build' / lib_so,
            this_dir.parent / 'build' / lib_dll,
            this_dir.parent / 'build' / 'lib' / lib_so,
        ]

        env_path = os.environ.get('GPU_HISTOGRAM_LIB')
        if env_path:
            candidates.insert(0, Path(env_path))

        sys_path = ctypes.util.find_library(lib_name)
        if sys_path:
            candidates.append(Path(sys_path))

        for p in candidates:
            if p.exists():
                return True
        return False
    except Exception:
        return False


# ---------------------------------------------------------------------------
# VRAM Estimation
# ---------------------------------------------------------------------------

def estimate_vram_need(X_all, n_bundles: Optional[int] = None,
                       num_class: int = 3, max_leaves: int = 63) -> int:
    """Estimate total GPU VRAM needed for training with this data.

    Parameters
    ----------
    X_all : scipy.sparse.csr_matrix or np.ndarray
        The feature matrix (all rows, all features).
    n_bundles : int or None
        Number of EFB bundles. If None, estimated from feature count
        assuming binary cross features with max_bin=255.
    num_class : int
        Number of output classes (3 for long/short/hold).
    max_leaves : int
        Maximum leaves per tree.

    Returns
    -------
    int
        Estimated bytes needed on GPU.
    """
    if sp_sparse is not None and sp_sparse.issparse(X_all):
        n_rows, n_cols = X_all.shape
        nnz = X_all.nnz
    else:
        n_rows, n_cols = X_all.shape
        nnz = n_rows * n_cols  # dense = every element is "nonzero"

    # Estimate EFB bundles if not provided
    # Binary cross features: max_bin=255 -> up to 254 features per bundle
    # Most features are binary, so EFB compression is high
    if n_bundles is None:
        features_per_bundle = 254  # max_bin - 1
        n_bundles = max(1, n_cols // features_per_bundle)

    # CSR region (read-only, allocated once)
    indptr_bytes = (n_rows + 1) * 8            # int64
    indices_bytes = nnz * 4                     # int32
    data_bytes = nnz * 1                        # uint8 (EFB bin index)
    bundle_offsets_bytes = (n_bundles + 1) * 4  # int32

    # Gradient region (double-buffered pinned + device copy)
    # 4 buffers: 2 pinned host (A/B) + 1 device = 3, plus margin
    grad_hess_bytes = 4 * n_rows * num_class * 4  # float32

    # Leaf partition (GPU-side leaf_id array)
    leaf_id_bytes = n_rows * 1  # int8
    leaf_count_bytes = max_leaves * 4  # int32

    # Histogram pool — binary features average 2 bins per bundle
    avg_bins_per_bundle = 2
    total_bins = n_bundles * avg_bins_per_bundle
    # Each histogram entry: grad (float64) + hess (float64) per class
    hist_per_leaf = total_bins * 2 * num_class * 8  # bytes
    hist_pool_bytes = max_leaves * hist_per_leaf

    # Kernel overhead (CUDA context, stack, registers, launch overhead)
    overhead_bytes = 512 * 1024 * 1024  # 512 MB conservative

    total = (indptr_bytes + indices_bytes + data_bytes + bundle_offsets_bytes
             + grad_hess_bytes + leaf_id_bytes + leaf_count_bytes
             + hist_pool_bytes + overhead_bytes)

    return total


def _estimate_from_tf(tf_name: str) -> float:
    """Get pre-computed VRAM estimate (GB) for a timeframe."""
    return TF_VRAM_BUDGET_GB.get(tf_name, 0.0)


# ---------------------------------------------------------------------------
# Main Integration: get_training_params()
# ---------------------------------------------------------------------------

def get_training_params(
    base_params: dict[str, Any],
    X_all,
    device_id: int = 0,
    tf_name: Optional[str] = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Add GPU histogram params if available and data fits in VRAM.

    This is the main entry point for ml_multi_tf.py. Call it after
    building _base_lgb_params and before the CPCV loop.

    Parameters
    ----------
    base_params : dict
        Base LightGBM parameters (from config.V3_LGBM_PARAMS + per-TF overrides).
    X_all : scipy.sparse.csr_matrix or np.ndarray
        The full feature matrix.
    device_id : int
        CUDA device to use. Default 0.
    tf_name : str or None
        Timeframe name ('1w', '1d', '4h', '1h', '15m'). Used for pre-computed
        VRAM estimates and logging. If None, estimates from X_all directly.
    dry_run : bool
        If True, only log what would happen. Return base_params unchanged.

    Returns
    -------
    dict
        Modified params with GPU settings, or original params if GPU
        unavailable or VRAM insufficient.
    """
    params = base_params.copy()

    # Check environment override
    if os.environ.get(_ENV_FORCE_CPU, '').lower() in ('1', 'true', 'yes'):
        log.info("GPU histogram DISABLED by %s environment variable", _ENV_FORCE_CPU)
        return params

    # Detect GPUs
    devices = _detect_cuda_devices()
    if not devices:
        log.info("GPU histogram: no CUDA devices detected. Using CPU.")
        return params

    if device_id >= len(devices):
        log.warning(
            "GPU histogram: requested device_id=%d but only %d devices found. Using CPU.",
            device_id, len(devices)
        )
        return params

    gpu = devices[device_id]
    gpu_name = gpu['name']
    vram_total_gb = gpu['vram_total_gb']
    vram_free_gb = gpu['vram_free_gb']
    vram_free_bytes = gpu['vram_free_bytes']
    usable_bytes = int(vram_free_bytes * _VRAM_SAFETY)
    usable_gb = usable_bytes / (1024 ** 3)

    # Estimate VRAM need
    estimated_bytes = estimate_vram_need(X_all)
    estimated_gb = estimated_bytes / (1024 ** 3)

    # Cross-check with pre-computed TF budget if available
    if tf_name and tf_name in TF_VRAM_BUDGET_GB:
        tf_budget_gb = TF_VRAM_BUDGET_GB[tf_name]
        # Use the larger of computed and pre-computed (conservative)
        if tf_budget_gb > estimated_gb:
            log.debug(
                "Using TF budget %.1fGB (> computed %.1fGB) for %s",
                tf_budget_gb, estimated_gb, tf_name
            )
            estimated_gb = tf_budget_gb
            estimated_bytes = int(tf_budget_gb * (1024 ** 3))

    # Log the assessment
    tf_label = f" [{tf_name}]" if tf_name else ""
    log.info(
        "GPU histogram%s: %s (%.1fGB total, %.1fGB free, %.1fGB usable). "
        "Estimated need: %.1fGB.",
        tf_label, gpu_name, vram_total_gb, vram_free_gb, usable_gb, estimated_gb
    )

    # Check fit
    if estimated_bytes > usable_bytes:
        log.info(
            "GPU histogram: CSR too large for GPU (%.1fGB > %.1fGB usable). Using CPU.%s",
            estimated_gb, usable_gb,
            f" Recommended: {TF_GPU_RECOMMENDATIONS.get(tf_name, 'larger GPU')}"
            if tf_name else ""
        )
        if dry_run:
            _log_dry_run(gpu, estimated_gb, usable_gb, tf_name, fit=False)
        return params

    if dry_run:
        _log_dry_run(gpu, estimated_gb, usable_gb, tf_name, fit=True)
        return params

    # Try native cuda_sparse first (preferred — LightGBM handles everything)
    if _check_cuda_sparse_available():
        log.info(
            "GPU histogram ENABLED (native cuda_sparse): "
            "%.1fGB / %.1fGB VRAM on %s",
            estimated_gb, usable_gb, gpu_name
        )
        params['device_type'] = 'cuda_sparse'
        params['gpu_device_id'] = device_id

        # Remove CPU-specific params that conflict with cuda_sparse
        params.pop('force_col_wise', None)
        params.pop('force_row_wise', None)
        # device_type supersedes the old 'device' key
        params.pop('device', None)

        return params

    # Try co-processor path (our custom libgpu_histogram.so)
    if _check_coprocessor_available():
        log.info(
            "GPU histogram ENABLED (co-processor): "
            "%.1fGB / %.1fGB VRAM on %s. "
            "Use gpu_train() from lgbm_integration.py instead of lgb.train().",
            estimated_gb, usable_gb, gpu_name
        )
        # Co-processor mode: params stay CPU because LightGBM runs on CPU.
        # The GPU acceleration happens inside gpu_train() which wraps lgb.train().
        # Tag the params so ml_multi_tf.py knows to call gpu_train().
        params['_gpu_histogram_coprocessor'] = True
        params['_gpu_histogram_device_id'] = device_id
        return params

    # Neither path available
    log.info(
        "GPU histogram: CUDA detected (%s, %.1fGB) but neither cuda_sparse "
        "LightGBM build nor libgpu_histogram.so found. Using CPU.",
        gpu_name, vram_total_gb
    )
    return params


# ---------------------------------------------------------------------------
# Dry Run Report
# ---------------------------------------------------------------------------

def _log_dry_run(gpu: dict, estimated_gb: float, usable_gb: float,
                 tf_name: Optional[str], fit: bool):
    """Log a dry-run report line."""
    status = "WOULD USE GPU" if fit else "WOULD FALL BACK TO CPU"
    log.info(
        "DRY RUN: %s. GPU=%s (%.1fGB free). Need=%.1fGB. TF=%s.",
        status, gpu['name'], gpu['vram_free_gb'], estimated_gb,
        tf_name or 'unknown'
    )


def dry_run_report(
    X_all=None,
    tf_names: Optional[list[str]] = None,
    device_id: int = 0,
) -> str:
    """Generate a human-readable report of what GPU histogram would do.

    Call with either X_all (actual data) or tf_names (pre-computed estimates).
    Returns a formatted string suitable for printing.

    Parameters
    ----------
    X_all : sparse matrix or None
        If provided, estimate from actual data.
    tf_names : list of str or None
        If provided, report for each timeframe using pre-computed budgets.
    device_id : int
        CUDA device to check.

    Returns
    -------
    str
        Formatted report.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("GPU HISTOGRAM DRY-RUN REPORT")
    lines.append("=" * 70)

    # GPU detection
    devices = _detect_cuda_devices()
    if not devices:
        lines.append("")
        lines.append("  NO CUDA DEVICES DETECTED")
        lines.append("  All training will use CPU.")
        lines.append("")
        lines.append("=" * 70)
        return "\n".join(lines)

    lines.append("")
    lines.append(f"  Detected {len(devices)} GPU(s):")
    for d in devices:
        cc = d['compute_capability']
        lines.append(
            f"    [{d['id']}] {d['name']}  "
            f"{d['vram_total_gb']}GB total, {d['vram_free_gb']}GB free  "
            f"CC {cc[0]}.{cc[1]}"
        )

    target_gpu = devices[device_id] if device_id < len(devices) else None
    if target_gpu is None:
        lines.append(f"\n  ERROR: device_id={device_id} not found.")
        lines.append("=" * 70)
        return "\n".join(lines)

    usable_gb = target_gpu['vram_free_gb'] * _VRAM_SAFETY
    lines.append(f"\n  Target device: [{device_id}] {target_gpu['name']}")
    lines.append(f"  Usable VRAM (85% of free): {usable_gb:.1f}GB")

    # LightGBM build check
    cuda_sparse = _check_cuda_sparse_available()
    coprocessor = _check_coprocessor_available()
    lines.append("")
    lines.append(f"  LightGBM cuda_sparse build: {'YES' if cuda_sparse else 'NO'}")
    lines.append(f"  Co-processor (libgpu_histogram.so): {'YES' if coprocessor else 'NO'}")

    if not cuda_sparse and not coprocessor:
        lines.append("")
        lines.append("  WARNING: No GPU histogram backend available.")
        lines.append("  Install cuda_sparse LightGBM or build libgpu_histogram.so.")

    # Per-TF assessment
    if tf_names is None:
        tf_names = ['1w', '1d', '4h', '1h', '15m']

    lines.append("")
    lines.append("  Per-Timeframe Assessment:")
    lines.append(f"  {'TF':<6} {'Need (GB)':<12} {'Fits?':<8} {'Recommendation'}")
    lines.append(f"  {'-' * 6} {'-' * 12} {'-' * 8} {'-' * 30}")

    for tf in tf_names:
        budget = TF_VRAM_BUDGET_GB.get(tf, 0)
        fits = budget <= usable_gb and (cuda_sparse or coprocessor)
        rec = TF_GPU_RECOMMENDATIONS.get(tf, '')
        marker = 'YES' if fits else 'NO'
        lines.append(f"  {tf:<6} {budget:<12.1f} {marker:<8} {rec}")

    # Actual data estimate if provided
    if X_all is not None:
        est_bytes = estimate_vram_need(X_all)
        est_gb = est_bytes / (1024 ** 3)
        fits = est_gb <= usable_gb and (cuda_sparse or coprocessor)

        if sp_sparse is not None and sp_sparse.issparse(X_all):
            shape_str = f"{X_all.shape[0]:,} x {X_all.shape[1]:,} (sparse, nnz={X_all.nnz:,})"
        else:
            shape_str = f"{X_all.shape[0]:,} x {X_all.shape[1]:,} (dense)"

        lines.append("")
        lines.append(f"  Actual data: {shape_str}")
        lines.append(f"  Estimated VRAM: {est_gb:.1f}GB")
        lines.append(f"  Fits in GPU: {'YES' if fits else 'NO'}")

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helper: should we use gpu_train() instead of lgb.train()?
# ---------------------------------------------------------------------------

def should_use_gpu_train(params: dict) -> bool:
    """Check if params indicate co-processor mode (gpu_train needed).

    After calling get_training_params(), check this to decide whether
    to call gpu_train() from lgbm_integration.py or standard lgb.train().

    Returns True if the co-processor path was selected and gpu_train()
    should be used instead of lgb.train().
    """
    return params.get('_gpu_histogram_coprocessor', False)


def get_gpu_device_id(params: dict) -> int:
    """Extract GPU device ID from params (set by get_training_params)."""
    return params.get('_gpu_histogram_device_id', params.get('gpu_device_id', 0))


def is_gpu_enabled(params: dict) -> bool:
    """Check if GPU histogram is enabled in any mode (native or co-processor)."""
    if params.get('device_type') == 'cuda_sparse':
        return True
    if params.get('_gpu_histogram_coprocessor', False):
        return True
    return False


def clean_params_for_lgb(params: dict) -> dict:
    """Remove internal tags before passing to lgb.train().

    Strips _gpu_histogram_* keys that LightGBM doesn't understand.
    Call this before passing params to lgb.train() or lgb.Dataset().
    """
    cleaned = params.copy()
    cleaned.pop('_gpu_histogram_coprocessor', None)
    cleaned.pop('_gpu_histogram_device_id', None)
    return cleaned


# ---------------------------------------------------------------------------
# Convenience: wrap the full train call
# ---------------------------------------------------------------------------

def train(
    params: dict[str, Any],
    dtrain: "lgb.Dataset",
    X_csr=None,
    **kwargs,
) -> "lgb.Booster":
    """Unified train function — routes to GPU or CPU automatically.

    If get_training_params() set cuda_sparse mode, this calls lgb.train()
    directly (LightGBM handles GPU internally).

    If co-processor mode was set, this calls gpu_train() from
    lgbm_integration.py which wraps lgb.train() with GPU histogram
    acceleration via the C shared library.

    Otherwise, falls back to standard lgb.train().

    Parameters
    ----------
    params : dict
        LightGBM params (output of get_training_params).
    dtrain : lgb.Dataset
        Training dataset.
    X_csr : scipy.sparse.csr_matrix or None
        Required for co-processor mode. The raw CSR matrix.
    **kwargs
        Additional arguments passed to lgb.train() (num_boost_round,
        valid_sets, callbacks, etc.)

    Returns
    -------
    lgb.Booster
        Trained model.
    """
    if lgb is None:
        raise ImportError("LightGBM is not installed.")

    clean = clean_params_for_lgb(params)

    if should_use_gpu_train(params):
        # Co-processor path
        if X_csr is None:
            raise ValueError(
                "Co-processor GPU histogram mode requires X_csr (the raw sparse "
                "CSR matrix). Pass it to train_pipeline.train()."
            )
        try:
            from gpu_histogram_fork.src.lgbm_integration import gpu_train
            device_id = get_gpu_device_id(params)
            return gpu_train(clean, dtrain, X_csr, gpu_id=device_id, **kwargs)
        except Exception as e:
            log.warning(
                "GPU co-processor failed: %s. Falling back to CPU lgb.train().", e
            )
            return lgb.train(clean, dtrain, **kwargs)

    # Native cuda_sparse or CPU — lgb.train() handles both
    return lgb.train(clean, dtrain, **kwargs)


# ---------------------------------------------------------------------------
# Module self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )

    print(dry_run_report(tf_names=['1w', '1d', '4h', '1h', '15m']))
