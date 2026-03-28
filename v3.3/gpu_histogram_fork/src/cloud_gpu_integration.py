"""
Cloud GPU Integration for GPU Histogram Training
=================================================
Bridges cloud_run_tf.py pipeline to GPU histogram fork.

This module is the ONLY file cloud_run_tf.py needs to import. It wraps
train_pipeline.py's detection/estimation/param-modification logic into
cloud-friendly functions with explicit logging that matches cloud_run_tf.py's
log format.

Integration points:
  Step 3 (training) in cloud_run_tf.py is the only step affected.
  Two lines added to cloud_run_tf.py, just before the CPCV training loop
  in ml_multi_tf.py is invoked:

    # In cloud_run_tf.py, after params setup, before lgb.train():
    from gpu_histogram_fork.src.cloud_gpu_integration import gpu_cloud_integration
    params = gpu_cloud_integration(params, tf_name=TF)

  OR, for the full setup including SetExternalCSR on the booster:

    from gpu_histogram_fork.src.cloud_gpu_integration import (
        gpu_cloud_integration, setup_gpu_for_training, should_use_gpu
    )
    if should_use_gpu(TF, X_csr):
        params = gpu_cloud_integration(params, tf_name=TF)
        # After lgb.train(), optionally call setup_gpu_for_training()

Design rules:
  - GPU is OPTIONAL. CPU fallback always works. Model is identical either way.
  - ALL features preserved. No subsetting, no row partitioning.
  - No changes to cloud_run_tf.py's pipeline structure.
  - No changes to ml_multi_tf.py's CPCV logic.
  - Params are modified in-place (copy returned). lgb.train() picks up device_type.
  - Logging uses print() to match cloud_run_tf.py style (not logging module).
"""

import os
import sys
import time
from typing import Optional

import numpy as np

try:
    from scipy import sparse as sp_sparse
except ImportError:
    sp_sparse = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# VRAM budget per timeframe (GB) — from ARCHITECTURE.md section 5.5
# Conservative estimates including CSR + histograms + gradients + overhead
_TF_VRAM_BUDGET_GB = {
    '1w':   3.0,    #  818 rows x 2.2M features
    '1d':   6.0,    # 5733 rows x 6M features
    '4h':  13.0,    #  23K rows x 4M features
    '1h':  26.0,    # 100K rows x 10M features
    '15m': 41.0,    # 227K rows x 10M features
}

_TF_GPU_RECS = {
    '1w':  'RTX 3090 24GB+',
    '1d':  'RTX 3090 24GB+',
    '4h':  'A40 48GB+',
    '1h':  'A100 80GB / H100',
    '15m': 'A100 80GB / H100 / B200 192GB',
}

# Safety factor: never use more than 85% of VRAM
_VRAM_SAFETY = 0.85

# Env var to force CPU even when GPU is available
_FORCE_CPU_ENV = 'GPU_HISTOGRAM_FORCE_CPU'


# ---------------------------------------------------------------------------
# Internal: GPU detection
# ---------------------------------------------------------------------------

def _detect_gpu() -> dict:
    """Detect primary CUDA GPU with VRAM info.

    Returns dict with keys: available, name, vram_total_gb, vram_free_gb,
    compute_capability, device_id.
    Returns {'available': False} if no GPU or CuPy not installed.
    """
    result = {'available': False}
    try:
        import cupy as cp
        n = cp.cuda.runtime.getDeviceCount()
        if n == 0:
            return result

        with cp.cuda.Device(0):
            props = cp.cuda.runtime.getDeviceProperties(0)
            free, total = cp.cuda.runtime.memGetInfo()
            name = props['name']
            if isinstance(name, bytes):
                name = name.decode()

            result.update({
                'available': True,
                'name': str(name),
                'device_id': 0,
                'vram_total_gb': round(total / (1024 ** 3), 1),
                'vram_free_gb': round(free / (1024 ** 3), 1),
                'vram_free_bytes': free,
                'compute_capability': (props.get('major', 0), props.get('minor', 0)),
            })
    except Exception:
        pass
    return result


def _check_lgbm_cuda_sparse() -> bool:
    """Check if LightGBM build supports device_type='cuda_sparse'.

    Runs a tiny 1-iteration probe. Returns True if the build supports it.
    """
    if lgb is None:
        return False
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        return False

    try:
        rng = np.random.RandomState(42)
        X = rng.rand(20, 3).astype(np.float32)
        y = rng.randint(0, 3, size=20)
        ds = lgb.Dataset(X, label=y, free_raw_data=False)
        lgb.train(
            {'objective': 'multiclass', 'num_class': 3,
             'device_type': 'cuda_sparse', 'num_threads': 1,
             'verbose': -1, 'num_iterations': 1, 'num_leaves': 4},
            ds, num_boost_round=1,
        )
        return True
    except Exception:
        return False


def _check_coprocessor_lib() -> bool:
    """Check if our custom libgpu_histogram shared library is available."""
    try:
        from pathlib import Path
        import ctypes.util

        lib_name = 'libgpu_histogram'
        this_dir = Path(__file__).parent
        candidates = [
            this_dir / f'{lib_name}.so',
            this_dir / f'{lib_name}.dll',
            this_dir.parent / 'build' / f'{lib_name}.so',
            this_dir.parent / 'build' / f'{lib_name}.dll',
            this_dir.parent / 'build' / 'lib' / f'{lib_name}.so',
            this_dir.parent / '_build' / f'{lib_name}.so',
        ]
        env_path = os.environ.get('GPU_HISTOGRAM_LIB')
        if env_path:
            candidates.insert(0, Path(env_path))
        sys_path = ctypes.util.find_library(lib_name)
        if sys_path:
            candidates.append(Path(sys_path))

        return any(p.exists() for p in candidates)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# VRAM estimation
# ---------------------------------------------------------------------------

def _estimate_vram_bytes(X_csr, num_class: int = 3, max_leaves: int = 63) -> int:
    """Estimate GPU VRAM needed for the full training data.

    Accounts for: CSR arrays, gradient double-buffers, leaf partitions,
    histogram pool, and CUDA overhead.

    Parameters
    ----------
    X_csr : scipy.sparse.csr_matrix or np.ndarray
        Full feature matrix.
    num_class : int
        Number of output classes.
    max_leaves : int
        Max leaves per tree.

    Returns
    -------
    int
        Estimated bytes.
    """
    if sp_sparse is not None and sp_sparse.issparse(X_csr):
        n_rows, n_cols = X_csr.shape
        nnz = X_csr.nnz
    else:
        n_rows, n_cols = X_csr.shape
        nnz = n_rows * n_cols

    # EFB bundles: binary crosses -> up to 254 per bundle (max_bin=255)
    features_per_bundle = 254
    n_bundles = max(1, n_cols // features_per_bundle)

    # CSR region (read-only on GPU, allocated once)
    indptr_bytes = (n_rows + 1) * 8        # int64
    indices_bytes = nnz * 4                 # int32
    data_bytes = nnz * 1                    # uint8 (EFB bin index)
    bundle_offsets = (n_bundles + 1) * 4    # int32

    # Gradients: 4 buffers (2 pinned host A/B + 1 device + margin)
    grad_bytes = 4 * n_rows * num_class * 4  # float32

    # Leaf partition
    leaf_id_bytes = n_rows * 1              # int8
    leaf_count_bytes = max_leaves * 4       # int32

    # Histogram pool: binary features average 2 bins per bundle
    avg_bins = 2
    total_bins = n_bundles * avg_bins
    hist_per_leaf = total_bins * 2 * num_class * 8  # grad+hess, float64, per class
    hist_bytes = max_leaves * hist_per_leaf

    # CUDA overhead (context, stack, registers, launch)
    overhead = 512 * 1024 * 1024  # 512 MB conservative

    return (indptr_bytes + indices_bytes + data_bytes + bundle_offsets
            + grad_bytes + leaf_id_bytes + leaf_count_bytes
            + hist_bytes + overhead)


# ---------------------------------------------------------------------------
# Public API: should_use_gpu()
# ---------------------------------------------------------------------------

def should_use_gpu(tf_name: str, X_csr=None) -> bool:
    """Check if GPU histogram training is viable for this timeframe.

    Tests three conditions:
      1. CUDA available (CuPy imports, device detected)
      2. LightGBM cuda_sparse build OR libgpu_histogram.so available
      3. VRAM sufficient for this TF's data

    Parameters
    ----------
    tf_name : str
        Timeframe ('1w', '1d', '4h', '1h', '15m').
    X_csr : scipy.sparse.csr_matrix or None
        If provided, computes VRAM estimate from actual data.
        If None, uses pre-computed per-TF budgets.

    Returns
    -------
    bool
        True if GPU histogram training should be used.
    """
    # Environment override
    if os.environ.get(_FORCE_CPU_ENV, '').lower() in ('1', 'true', 'yes'):
        _cloud_log(f"GPU histogram DISABLED by {_FORCE_CPU_ENV}")
        return False

    gpu = _detect_gpu()
    if not gpu['available']:
        _cloud_log("GPU histogram: no CUDA device detected")
        return False

    usable_bytes = int(gpu['vram_free_bytes'] * _VRAM_SAFETY)

    # Estimate VRAM need
    if X_csr is not None:
        need_bytes = _estimate_vram_bytes(X_csr)
    elif tf_name in _TF_VRAM_BUDGET_GB:
        need_bytes = int(_TF_VRAM_BUDGET_GB[tf_name] * (1024 ** 3))
    else:
        _cloud_log(f"GPU histogram: unknown TF '{tf_name}', no VRAM estimate")
        return False

    need_gb = need_bytes / (1024 ** 3)
    usable_gb = usable_bytes / (1024 ** 3)

    if need_bytes > usable_bytes:
        rec = _TF_GPU_RECS.get(tf_name, 'larger GPU')
        _cloud_log(
            f"GPU histogram: {tf_name} needs {need_gb:.1f}GB, "
            f"only {usable_gb:.1f}GB usable on {gpu['name']}. "
            f"Recommended: {rec}. Using CPU."
        )
        return False

    # Check backend availability
    has_cuda_sparse = _check_lgbm_cuda_sparse()
    has_coprocessor = _check_coprocessor_lib()

    if not has_cuda_sparse and not has_coprocessor:
        _cloud_log(
            f"GPU histogram: CUDA detected ({gpu['name']}, {gpu['vram_total_gb']}GB) "
            f"but no backend available (need cuda_sparse LightGBM or libgpu_histogram.so). "
            f"Using CPU."
        )
        return False

    backend = 'cuda_sparse' if has_cuda_sparse else 'co-processor'
    _cloud_log(
        f"GPU histogram VIABLE: {tf_name} on {gpu['name']} "
        f"({need_gb:.1f}GB / {usable_gb:.1f}GB usable) via {backend}"
    )
    return True


# ---------------------------------------------------------------------------
# Public API: gpu_cloud_integration()
# ---------------------------------------------------------------------------

def gpu_cloud_integration(
    base_params: dict,
    tf_name: str,
    X_csr=None,
    device_id: int = 0,
) -> dict:
    """Modify LightGBM training params to use GPU histograms if viable.

    This is the main entry point for cloud_run_tf.py. Call after building
    base params from config.V3_LGBM_PARAMS and before passing to lgb.train().

    If GPU is not available, not viable, or VRAM is insufficient, returns
    params UNCHANGED. CPU training proceeds identically. The model output
    is the same regardless of GPU/CPU — only histogram build speed differs.

    Parameters
    ----------
    base_params : dict
        LightGBM params from config.V3_LGBM_PARAMS (copied, not mutated).
    tf_name : str
        Timeframe ('1w', '1d', '4h', '1h', '15m').
    X_csr : scipy.sparse.csr_matrix or None
        If provided, used for precise VRAM estimation.
        If None, uses pre-computed per-TF budgets.
    device_id : int
        CUDA device ID (default 0).

    Returns
    -------
    dict
        Modified params dict. If GPU enabled, contains device_type='cuda_sparse'
        or _gpu_histogram_coprocessor=True. If CPU fallback, original params
        with device='cpu' intact.

    Integration in cloud_run_tf.py (2 lines):
    ------------------------------------------
    Add these lines in cloud_run_tf.py BEFORE the train step (Step 4),
    after environment setup but before invoking ml_multi_tf.py:

        # === GPU HISTOGRAM INTEGRATION (optional, auto-detects) ===
        os.environ['GPU_HISTOGRAM_TF'] = TF
        # ml_multi_tf.py picks this up and calls gpu_cloud_integration() internally

    OR, if modifying ml_multi_tf.py params directly:

        from gpu_histogram_fork.src.cloud_gpu_integration import gpu_cloud_integration
        params = gpu_cloud_integration(params, tf_name=TF, X_csr=X_all)
    """
    t0 = time.time()
    params = base_params.copy()

    _cloud_log(f"=== GPU Histogram Detection [{tf_name}] ===")

    # --- Check environment override ---
    if os.environ.get(_FORCE_CPU_ENV, '').lower() in ('1', 'true', 'yes'):
        _cloud_log(f"  DISABLED by {_FORCE_CPU_ENV} env var. Using CPU.")
        return params

    # --- Detect GPU ---
    gpu = _detect_gpu()
    if not gpu['available']:
        _cloud_log("  No CUDA device detected. Using CPU.")
        return params

    gpu_name = gpu['name']
    vram_total = gpu['vram_total_gb']
    vram_free = gpu['vram_free_gb']
    usable_gb = round(gpu['vram_free_bytes'] * _VRAM_SAFETY / (1024 ** 3), 1)

    _cloud_log(f"  GPU: {gpu_name} ({vram_total}GB total, {vram_free}GB free, {usable_gb}GB usable)")

    # --- Estimate VRAM need ---
    if X_csr is not None:
        need_bytes = _estimate_vram_bytes(X_csr)
        if sp_sparse is not None and sp_sparse.issparse(X_csr):
            shape_str = f"{X_csr.shape[0]:,} x {X_csr.shape[1]:,} sparse (nnz={X_csr.nnz:,})"
        else:
            shape_str = f"{X_csr.shape[0]:,} x {X_csr.shape[1]:,} dense"
        _cloud_log(f"  Data: {shape_str}")
    else:
        budget = _TF_VRAM_BUDGET_GB.get(tf_name)
        if budget is None:
            _cloud_log(f"  Unknown TF '{tf_name}', no VRAM estimate. Using CPU.")
            return params
        need_bytes = int(budget * (1024 ** 3))

    need_gb = round(need_bytes / (1024 ** 3), 1)
    usable_bytes = int(gpu['vram_free_bytes'] * _VRAM_SAFETY)

    _cloud_log(f"  VRAM need: {need_gb}GB, usable: {usable_gb}GB")

    # --- Check VRAM fit ---
    if need_bytes > usable_bytes:
        rec = _TF_GPU_RECS.get(tf_name, 'larger GPU')
        _cloud_log(f"  VRAM INSUFFICIENT ({need_gb}GB > {usable_gb}GB). Recommended: {rec}")
        _cloud_log(f"  Falling back to CPU. Model output is IDENTICAL.")
        return params

    # --- Check backend availability ---
    has_cuda_sparse = _check_lgbm_cuda_sparse()
    has_coprocessor = _check_coprocessor_lib()

    if has_cuda_sparse:
        # Native path: LightGBM handles GPU internally
        params['device_type'] = 'cuda_sparse'
        params['gpu_device_id'] = device_id

        # Remove CPU-specific params that conflict with cuda_sparse
        params.pop('force_col_wise', None)
        params.pop('force_row_wise', None)
        params.pop('device', None)

        dt = time.time() - t0
        _cloud_log(
            f"  GPU ENABLED (native cuda_sparse) on {gpu_name}: "
            f"{need_gb}GB / {usable_gb}GB ({dt:.1f}s)"
        )
        _cloud_log(f"  LightGBM will build histograms on GPU. EFB + tree logic stays CPU.")
        return params

    if has_coprocessor:
        # Co-processor path: our libgpu_histogram.so
        # Params stay CPU-mode because LightGBM still runs on CPU.
        # gpu_train() wraps lgb.train() with GPU histogram acceleration.
        params['_gpu_histogram_coprocessor'] = True
        params['_gpu_histogram_device_id'] = device_id

        dt = time.time() - t0
        _cloud_log(
            f"  GPU ENABLED (co-processor libgpu_histogram) on {gpu_name}: "
            f"{need_gb}GB / {usable_gb}GB ({dt:.1f}s)"
        )
        _cloud_log(f"  Use train_pipeline.train() instead of lgb.train() for GPU acceleration.")
        return params

    # Neither backend available
    _cloud_log(
        f"  GPU detected ({gpu_name}, {vram_total}GB) but no backend available."
    )
    _cloud_log(f"  Install cuda_sparse LightGBM build or compile libgpu_histogram.so.")
    _cloud_log(f"  Using CPU. Model output is IDENTICAL.")
    return params


# ---------------------------------------------------------------------------
# Public API: setup_gpu_for_training()
# ---------------------------------------------------------------------------

def setup_gpu_for_training(X_csr, booster) -> bool:
    """Upload CSR to GPU and call SetExternalCSR via C API.

    This is called AFTER lgb.train() creates the booster, for the
    co-processor path only. For native cuda_sparse, this is a no-op
    (LightGBM handles CSR internally).

    Parameters
    ----------
    X_csr : scipy.sparse.csr_matrix
        The training data in CSR format.
    booster : lgb.Booster
        The LightGBM booster (after train).

    Returns
    -------
    bool
        True if GPU setup succeeded, False if not applicable or failed.

    Notes
    -----
    For the native cuda_sparse path, LightGBM manages its own GPU memory
    and CSR upload. This function only applies to the co-processor path
    where we manage GPU memory ourselves via memory_manager.py.
    """
    if booster is None or X_csr is None:
        return False

    if sp_sparse is None or not sp_sparse.issparse(X_csr):
        _cloud_log("  setup_gpu_for_training: X_csr is not sparse, skipping")
        return False

    # Check if co-processor library is available
    if not _check_coprocessor_lib():
        _cloud_log("  setup_gpu_for_training: libgpu_histogram.so not found, skipping")
        return False

    try:
        from .memory_manager import GPUMemoryManager

        mgr = GPUMemoryManager()

        # Upload CSR to GPU
        handle = mgr.upload_csr(X_csr)
        if handle is None:
            _cloud_log("  setup_gpu_for_training: CSR upload failed (VRAM?)")
            return False

        _cloud_log(
            f"  GPU CSR uploaded: {handle.n_rows:,} x {handle.n_cols:,} "
            f"(nnz={handle.nnz:,}, {handle.bytes_used / (1024**3):.1f}GB)"
        )

        # SetExternalCSR via ctypes — requires the C library to expose this
        try:
            from .lgbm_integration import GPUHistogramProvider
            provider = GPUHistogramProvider()
            if hasattr(provider, 'set_external_csr'):
                provider.set_external_csr(
                    handle.indptr, handle.indices, handle.data,
                    handle.n_rows, handle.n_cols
                )
                _cloud_log("  SetExternalCSR: success")
                return True
            else:
                _cloud_log("  SetExternalCSR: not available in this build")
                return False
        except Exception as e:
            _cloud_log(f"  SetExternalCSR failed: {e}")
            return False

    except ImportError as e:
        _cloud_log(f"  setup_gpu_for_training: import error: {e}")
        return False
    except Exception as e:
        _cloud_log(f"  setup_gpu_for_training: {e}")
        return False


# ---------------------------------------------------------------------------
# Public API: get_gpu_status_report()
# ---------------------------------------------------------------------------

def get_gpu_status_report(tf_name: str, X_csr=None) -> str:
    """Generate a human-readable GPU status report for cloud logging.

    Parameters
    ----------
    tf_name : str
        Timeframe name.
    X_csr : sparse matrix or None
        If provided, includes actual VRAM estimate.

    Returns
    -------
    str
        Multi-line report string.
    """
    lines = ["GPU Histogram Status:"]

    gpu = _detect_gpu()
    if not gpu['available']:
        lines.append("  No CUDA device detected. Training will use CPU.")
        lines.append("  Model output is identical regardless of GPU/CPU.")
        return "\n".join(lines)

    usable_gb = round(gpu['vram_free_bytes'] * _VRAM_SAFETY / (1024 ** 3), 1)
    lines.append(f"  GPU: {gpu['name']} ({gpu['vram_total_gb']}GB total, {usable_gb}GB usable)")

    if X_csr is not None:
        need_bytes = _estimate_vram_bytes(X_csr)
        need_gb = round(need_bytes / (1024 ** 3), 1)
        fits = need_bytes <= int(gpu['vram_free_bytes'] * _VRAM_SAFETY)
        lines.append(f"  VRAM need for {tf_name}: {need_gb}GB {'<= usable (FITS)' if fits else '> usable (TOO LARGE)'}")
    elif tf_name in _TF_VRAM_BUDGET_GB:
        budget = _TF_VRAM_BUDGET_GB[tf_name]
        fits = budget <= usable_gb
        lines.append(f"  VRAM budget for {tf_name}: {budget}GB {'<= usable (FITS)' if fits else '> usable (TOO LARGE)'}")

    has_cs = _check_lgbm_cuda_sparse()
    has_cp = _check_coprocessor_lib()
    lines.append(f"  LightGBM cuda_sparse: {'YES' if has_cs else 'NO'}")
    lines.append(f"  Co-processor lib: {'YES' if has_cp else 'NO'}")

    if not has_cs and not has_cp:
        lines.append("  No GPU backend available. Training will use CPU.")
    lines.append("  Model output is identical regardless of GPU/CPU.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Utility: clean params for lgb.train()
# ---------------------------------------------------------------------------

def clean_params_for_lgb(params: dict) -> dict:
    """Strip internal GPU tags before passing to lgb.train().

    LightGBM does not recognize _gpu_histogram_* keys.
    Call this before lgb.train() if using co-processor mode.

    For native cuda_sparse mode, params are already clean.
    """
    cleaned = params.copy()
    cleaned.pop('_gpu_histogram_coprocessor', None)
    cleaned.pop('_gpu_histogram_device_id', None)
    return cleaned


def is_gpu_enabled(params: dict) -> bool:
    """Check if GPU histogram is active in the params dict."""
    if params.get('device_type') == 'cuda_sparse':
        return True
    if params.get('_gpu_histogram_coprocessor', False):
        return True
    return False


def is_coprocessor_mode(params: dict) -> bool:
    """Check if co-processor mode is active (need gpu_train instead of lgb.train)."""
    return params.get('_gpu_histogram_coprocessor', False)


# ---------------------------------------------------------------------------
# Logging helper (matches cloud_run_tf.py print-based logging)
# ---------------------------------------------------------------------------

_START_TIME = time.time()

def _cloud_log(msg: str):
    """Log in cloud_run_tf.py format: [Xs] message."""
    elapsed = time.time() - _START_TIME
    print(f"[{elapsed:.0f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Module self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("  Cloud GPU Integration Self-Test")
    print("=" * 60)

    # Test detection
    gpu = _detect_gpu()
    print(f"\nGPU detected: {gpu['available']}")
    if gpu['available']:
        print(f"  Name: {gpu['name']}")
        print(f"  VRAM: {gpu['vram_total_gb']}GB total, {gpu['vram_free_gb']}GB free")

    # Test should_use_gpu for each TF
    print("\nPer-TF viability (pre-computed budgets):")
    for tf in ['1w', '1d', '4h', '1h', '15m']:
        viable = should_use_gpu(tf)
        budget = _TF_VRAM_BUDGET_GB.get(tf, 0)
        print(f"  {tf}: {'YES' if viable else 'NO'} (budget: {budget}GB)")

    # Test gpu_cloud_integration with dummy params
    print("\nIntegration test (dry):")
    from config import V3_LGBM_PARAMS
    for tf in ['1w', '1d']:
        result = gpu_cloud_integration(V3_LGBM_PARAMS.copy(), tf_name=tf)
        gpu_on = is_gpu_enabled(result)
        print(f"  {tf}: GPU={'ON' if gpu_on else 'OFF'}, device_type={result.get('device_type', 'cpu')}")

    # Full status report
    print()
    print(get_gpu_status_report('1w'))
