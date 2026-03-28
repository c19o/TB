"""
LightGBM CSR Bridge — Pass scipy CSR to CUDASparseHistTreeLearner via C API
=============================================================================

Problem:
    LightGBM's Python API doesn't expose custom tree learner methods.
    CUDASparseHistTreeLearner::SetExternalCSR() lives inside the forked
    LightGBM C++ and must be called BETWEEN Booster creation and the
    first booster.update() call.

Solution:
    1. Our forked LightGBM exports LGBM_BoosterSetExternalCSR() in c_api.cpp
    2. This module calls it via ctypes through lgb.basic._LIB
    3. The C function casts Booster -> GBDT -> GetTreeLearner() ->
       dynamic_cast<CUDASparseHistTreeLearner*> -> SetExternalCSR()
    4. The CSR data is stored host-side; UploadCSR() happens automatically
       on the first ConstructHistograms() call

Timing in training loop:
    Dataset construction (lgb.Dataset)
         |
    Booster creation (lgb.Booster) -- creates tree_learner, calls Init()
         |
    >>> lgbm_csr_bridge.set_external_csr(booster, X_csr) <<<  <-- HERE
         |
    booster.update() loop -- first update triggers UploadCSR() + GPU kernels

Matrix thesis: ALL features preserved. The CSR is our sparse binary cross
features (2-10M columns). Structural zeros = feature OFF. No filtering.

Copyright (c) Savage22 Server Project. Licensed under MIT.
"""

import ctypes
import logging
import os
import sys
from typing import Optional, Union

import numpy as np

try:
    import scipy.sparse as sp_sparse
except ImportError:
    sp_sparse = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

# EarlyStopException moved between LightGBM versions.
# Resolve it once here to avoid version-dependent imports everywhere.
_EarlyStopException = Exception  # fallback: catch nothing specific
if lgb is not None:
    for _loc in ('lgb.callback.EarlyStopException',
                 'lgb.early_stopping.EarlyStopException'):
        try:
            _EarlyStopException = eval(_loc)
            break
        except (AttributeError, NameError):
            continue

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error codes from our forked c_api.cpp (LGBM_BoosterSetExternalCSR)
# These must match the C side.
# ---------------------------------------------------------------------------
_CSR_OK = 0
_CSR_ERR_NOT_CUDA_SPARSE = -1    # tree_learner is not CUDASparseHistTreeLearner
_CSR_ERR_INVALID_HANDLE = -2     # booster handle is NULL
_CSR_ERR_INVALID_ARG = -3        # NULL pointers or bad dimensions
_CSR_ERR_CAST_FAILED = -4        # dynamic_cast to CUDASparseHistTreeLearner failed

_CSR_ERROR_MESSAGES = {
    _CSR_OK: "success",
    _CSR_ERR_NOT_CUDA_SPARSE: (
        "Tree learner is not CUDASparseHistTreeLearner. "
        "Set device_type='cuda_sparse' in params."
    ),
    _CSR_ERR_INVALID_HANDLE: "Booster handle is NULL or invalid.",
    _CSR_ERR_INVALID_ARG: "Invalid argument (NULL pointer or bad dimensions).",
    _CSR_ERR_CAST_FAILED: (
        "dynamic_cast to CUDASparseHistTreeLearner failed. "
        "LightGBM was not built with -DUSE_CUDA_SPARSE=ON."
    ),
}


# ---------------------------------------------------------------------------
# C API function signature declaration
# ---------------------------------------------------------------------------

_api_declared = False


def _declare_c_api(lib: ctypes.CDLL) -> bool:
    """Declare LGBM_BoosterSetExternalCSR on the LightGBM C library.

    Our forked LightGBM adds this function to c_api.cpp:

        int LGBM_BoosterSetExternalCSR(
            BoosterHandle    handle,
            const int64_t*   indptr,
            const int32_t*   indices,
            int64_t          nnz,
            int32_t          n_rows,
            int32_t          n_features
        );

    Returns True if the symbol exists, False if not (stock LightGBM).
    """
    global _api_declared
    if _api_declared:
        return True

    try:
        func = lib.LGBM_BoosterSetExternalCSR
    except AttributeError:
        return False

    func.restype = ctypes.c_int
    func.argtypes = [
        ctypes.c_void_p,                        # BoosterHandle
        ctypes.POINTER(ctypes.c_int64),          # indptr
        ctypes.POINTER(ctypes.c_int32),          # indices
        ctypes.c_int64,                          # nnz
        ctypes.c_int32,                          # n_rows
        ctypes.c_int32,                          # n_features
    ]

    _api_declared = True
    return True


# ---------------------------------------------------------------------------
# Fork detection
# ---------------------------------------------------------------------------

def is_fork_available() -> bool:
    """Check if the installed LightGBM is our cuda_sparse fork.

    Returns True if LGBM_BoosterSetExternalCSR is exported.
    """
    if lgb is None:
        return False
    try:
        lib = lgb.basic._LIB
        return _declare_c_api(lib)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# CUDA availability check (no CuPy dependency — uses ctypes directly)
# ---------------------------------------------------------------------------

def _check_cuda_available() -> dict:
    """Check for CUDA runtime without importing CuPy.

    Returns dict with keys: available, device_count, driver_version,
    free_vram_bytes (device 0), total_vram_bytes (device 0), device_name.
    """
    result = {
        'available': False,
        'device_count': 0,
        'driver_version': 0,
        'free_vram_bytes': 0,
        'total_vram_bytes': 0,
        'device_name': '',
    }

    # Try CuPy first (already imported in most of our pipeline)
    try:
        import cupy as cp
        n = cp.cuda.runtime.getDeviceCount()
        if n == 0:
            return result
        result['available'] = True
        result['device_count'] = n
        result['driver_version'] = cp.cuda.runtime.driverGetVersion()
        with cp.cuda.Device(0):
            free, total = cp.cuda.runtime.memGetInfo()
            result['free_vram_bytes'] = free
            result['total_vram_bytes'] = total
            props = cp.cuda.runtime.getDeviceProperties(0)
            name = props.get('name', b'')
            result['device_name'] = name.decode() if isinstance(name, bytes) else str(name)
        return result
    except Exception:
        pass

    # Fallback: try loading libcudart directly
    try:
        if sys.platform == 'win32':
            cudart = ctypes.CDLL('cudart64_12.dll')
        else:
            cudart = ctypes.CDLL('libcudart.so')
        count = ctypes.c_int(0)
        if cudart.cudaGetDeviceCount(ctypes.byref(count)) == 0 and count.value > 0:
            result['available'] = True
            result['device_count'] = count.value
    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# VRAM estimation for the CSR transfer
# ---------------------------------------------------------------------------

# Bytes per element in GPU memory
_INDPTR_DTYPE_SIZE = 8   # int64
_INDICES_DTYPE_SIZE = 4  # int32
_OVERHEAD_BYTES = 256 * 1024 * 1024  # 256 MB for CUDA context + kernel scratch

# VRAM safety factor — never allocate more than 85% of free VRAM
_VRAM_SAFETY = 0.85


def estimate_csr_vram_bytes(n_rows: int, nnz: int, n_features: int,
                            num_class: int = 3, max_leaves: int = 63) -> int:
    """Estimate GPU VRAM needed for CSR + histogram buffers.

    Parameters
    ----------
    n_rows : int
        Number of training rows.
    nnz : int
        Number of nonzero entries in CSR.
    n_features : int
        Number of features (columns).
    num_class : int
        Number of classes (3 for long/short/hold).
    max_leaves : int
        Maximum leaves per tree.

    Returns
    -------
    int
        Estimated bytes needed on GPU.
    """
    # CSR arrays (uploaded once, stay resident)
    csr_bytes = (
        (n_rows + 1) * _INDPTR_DTYPE_SIZE    # indptr
        + nnz * _INDICES_DTYPE_SIZE           # indices
    )

    # Gradient/hessian buffers (double-buffered for async upload)
    # 2 buffers x (grad + hess) x n_rows x num_class x float64
    grad_bytes = 4 * n_rows * num_class * 8

    # Histogram buffer pool
    # Binary features: 2 bins per feature (ON/OFF)
    # Each bin: grad (float64) + hess (float64) per class
    hist_per_leaf = n_features * 2 * 2 * num_class * 8
    hist_bytes = max_leaves * hist_per_leaf

    # Leaf row indices buffer
    leaf_idx_bytes = n_rows * 4  # int32

    return csr_bytes + grad_bytes + hist_bytes + leaf_idx_bytes + _OVERHEAD_BYTES


# ---------------------------------------------------------------------------
# Main API: set_external_csr()
# ---------------------------------------------------------------------------

def set_external_csr(
    booster: "lgb.Booster",
    X_csr: "sp_sparse.csr_matrix",
    validate_vram: bool = True,
    device_id: int = 0,
) -> None:
    """Pass a scipy CSR matrix to LightGBM's CUDASparseHistTreeLearner.

    Must be called AFTER lgb.Booster() creation and BEFORE the first
    booster.update() call.

    The CSR data is stored host-side by the tree learner. GPU upload
    happens automatically on the first ConstructHistograms() call
    (triggered by the first booster.update()).

    Parameters
    ----------
    booster : lgb.Booster
        A freshly created Booster with device_type='cuda_sparse'.
    X_csr : scipy.sparse.csr_matrix
        The sparse binary cross feature matrix. Must have:
        - int64 indptr  (required for NNZ > 2^31 on 15m timeframe)
        - int32 indices (feature indices)
        - Shape (n_rows, n_features) matching the training Dataset.
    validate_vram : bool, default True
        If True, estimate VRAM requirement and warn if it may not fit.
        Does NOT prevent the call — the C++ side handles OOM gracefully.
    device_id : int, default 0
        CUDA device for VRAM estimation. The actual device is set in
        LightGBM's params (gpu_device_id).

    Raises
    ------
    ImportError
        If lightgbm or scipy is not installed.
    RuntimeError
        If the forked LightGBM is not available (LGBM_BoosterSetExternalCSR
        not exported), or if the C call fails.
    TypeError
        If X_csr is not a scipy CSR matrix.
    ValueError
        If X_csr has invalid dimensions or dtypes.
    """
    # --- Dependency checks ---
    if lgb is None:
        raise ImportError("lightgbm is not installed.")
    if sp_sparse is None:
        raise ImportError("scipy is not installed.")

    # --- Type validation ---
    if not sp_sparse.issparse(X_csr):
        raise TypeError(
            f"Expected scipy sparse matrix, got {type(X_csr).__name__}. "
            "Pass the raw CSR from the cross generator."
        )
    if not sp_sparse.isspmatrix_csr(X_csr):
        log.info("Converting sparse matrix to CSR format (was %s)",
                 type(X_csr).__name__)
        X_csr = X_csr.tocsr()

    n_rows, n_features = X_csr.shape
    nnz = X_csr.nnz

    if n_rows == 0 or n_features == 0:
        raise ValueError(
            f"CSR matrix has shape ({n_rows}, {n_features}) — "
            "cannot be empty."
        )
    if nnz == 0:
        raise ValueError(
            "CSR matrix has zero nonzero entries. "
            "Cross features must have at least some active signals."
        )

    # --- Check fork availability ---
    lib = lgb.basic._LIB
    if not _declare_c_api(lib):
        raise RuntimeError(
            "LGBM_BoosterSetExternalCSR not found in LightGBM library. "
            "The installed LightGBM is stock, not our cuda_sparse fork.\n"
            "\n"
            "To fix:\n"
            "  1. Build LightGBM from v3.3/gpu_histogram_fork/_build/LightGBM\n"
            "     with: cmake -DUSE_CUDA_SPARSE=ON -DUSE_CUDA=ON ..\n"
            "  2. pip install --no-deps -e .\n"
            "\n"
            "Or use the co-processor path via lgbm_integration.gpu_train() "
            "which uses our standalone libgpu_histogram.so."
        )

    # --- VRAM pre-check (advisory, not blocking) ---
    if validate_vram:
        _validate_vram(n_rows, nnz, n_features, device_id)

    # --- Prepare CSR arrays with correct dtypes ---
    # int64 indptr — critical for 15m (NNZ > 2^31)
    indptr = np.ascontiguousarray(X_csr.indptr, dtype=np.int64)
    if len(indptr) != n_rows + 1:
        raise ValueError(
            f"indptr length {len(indptr)} != n_rows + 1 ({n_rows + 1}). "
            "Corrupt CSR matrix."
        )

    # int32 indices — feature indices (column count fits in int32 for all TFs)
    if n_features > np.iinfo(np.int32).max:
        raise ValueError(
            f"n_features={n_features:,} exceeds int32 max ({np.iinfo(np.int32).max:,}). "
            "Column-partitioned multi-GPU not yet supported."
        )
    indices = np.ascontiguousarray(X_csr.indices, dtype=np.int32)
    if len(indices) != nnz:
        raise ValueError(
            f"indices length {len(indices)} != nnz ({nnz}). "
            "Corrupt CSR matrix."
        )

    # --- Get booster handle ---
    # lgb.Booster stores the C handle as self.handle (ctypes.c_void_p)
    handle = booster.handle
    if handle is None or (hasattr(handle, 'value') and handle.value is None):
        raise RuntimeError(
            "Booster handle is NULL. The Booster may have been freed."
        )
    # Convert to raw void* for ctypes call
    if isinstance(handle, ctypes.c_void_p):
        handle_ptr = handle
    else:
        handle_ptr = ctypes.c_void_p(handle)

    # --- Call the C API ---
    log.info(
        "Setting external CSR: %s rows, %s features, %s nnz (%.2f%% dense)",
        f"{n_rows:,}", f"{n_features:,}", f"{nnz:,}",
        100.0 * nnz / (n_rows * n_features) if n_rows * n_features > 0 else 0
    )

    rc = lib.LGBM_BoosterSetExternalCSR(
        handle_ptr,
        indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_int64(nnz),
        ctypes.c_int32(n_rows),
        ctypes.c_int32(n_features),
    )

    if rc != _CSR_OK:
        msg = _CSR_ERROR_MESSAGES.get(rc, f"Unknown error code {rc}")
        raise RuntimeError(
            f"LGBM_BoosterSetExternalCSR failed (rc={rc}): {msg}"
        )

    log.info(
        "External CSR set successfully. GPU upload will occur on first "
        "ConstructHistograms() call (first booster.update())."
    )


# ---------------------------------------------------------------------------
# VRAM validation helper
# ---------------------------------------------------------------------------

def _validate_vram(n_rows: int, nnz: int, n_features: int,
                   device_id: int = 0) -> None:
    """Estimate VRAM need and warn if it may not fit.

    This is advisory only — it does NOT prevent the CSR from being set.
    The C++ side handles OOM by logging an error and falling back.
    """
    est_bytes = estimate_csr_vram_bytes(n_rows, nnz, n_features)
    est_gb = est_bytes / (1024 ** 3)

    cuda_info = _check_cuda_available()
    if not cuda_info['available']:
        log.warning(
            "CUDA not detected. CSR will be set but GPU histogram "
            "building will fail at runtime. Estimated VRAM need: %.1f GB",
            est_gb
        )
        return

    free_bytes = cuda_info['free_vram_bytes']
    total_bytes = cuda_info['total_vram_bytes']
    usable_bytes = int(free_bytes * _VRAM_SAFETY)

    if est_bytes > usable_bytes:
        log.warning(
            "VRAM may be insufficient for GPU histogram. "
            "Estimated: %.1f GB, available: %.1f GB (%.1f GB free x 85%% safety) "
            "on %s. The C++ side will attempt allocation and fall back to CPU "
            "histogram on OOM.",
            est_gb,
            usable_bytes / (1024 ** 3),
            free_bytes / (1024 ** 3),
            cuda_info['device_name'] or f"device {device_id}",
        )
    else:
        log.info(
            "VRAM check OK: estimated %.1f GB, available %.1f GB on %s",
            est_gb,
            usable_bytes / (1024 ** 3),
            cuda_info['device_name'] or f"device {device_id}",
        )


# ---------------------------------------------------------------------------
# High-level training function: gpu_sparse_train()
# ---------------------------------------------------------------------------

def gpu_sparse_train(
    params: dict,
    train_data: "lgb.Dataset",
    X_csr: "sp_sparse.csr_matrix",
    num_boost_round: int = 800,
    valid_sets: Optional[list] = None,
    valid_names: Optional[list] = None,
    callbacks: Optional[list] = None,
    feval: Optional[callable] = None,
) -> "lgb.Booster":
    """Train LightGBM with GPU sparse histograms via the forked build.

    Same interface as lgb.train() but additionally takes X_csr (scipy CSR)
    and passes it to the CUDASparseHistTreeLearner via our C API bridge.

    The training loop is run manually (booster.update() in a loop) to
    inject the SetExternalCSR call between Booster creation and the
    first training iteration.

    Parameters
    ----------
    params : dict
        LightGBM parameters. device_type will be forced to 'cuda_sparse'.
    train_data : lgb.Dataset
        Training dataset (already constructed).
    X_csr : scipy.sparse.csr_matrix
        The sparse binary cross feature matrix. Must match train_data's
        row count and feature count.
    num_boost_round : int, default 800
        Number of boosting iterations.
    valid_sets : list of lgb.Dataset or None
        Validation datasets for early stopping / eval.
    valid_names : list of str or None
        Names for validation sets in logging.
    callbacks : list or None
        LightGBM callbacks (early_stopping, log_evaluation, etc.).
    feval : callable or None
        Custom evaluation function.

    Returns
    -------
    lgb.Booster
        Trained model.

    Raises
    ------
    RuntimeError
        If the forked LightGBM is not available, CUDA is not present,
        or training fails.
    ImportError
        If lightgbm or scipy is not installed.
    """
    if lgb is None:
        raise ImportError("lightgbm is required")
    if sp_sparse is None:
        raise ImportError("scipy is required")

    # --- Force cuda_sparse device type ---
    params = params.copy()
    params['device_type'] = 'cuda_sparse'
    # Remove conflicting params
    params.pop('device', None)
    params.pop('force_col_wise', None)
    params.pop('force_row_wise', None)

    # --- Validate fork availability before doing anything expensive ---
    if not is_fork_available():
        raise RuntimeError(
            "Forked LightGBM with LGBM_BoosterSetExternalCSR not available. "
            "Cannot use gpu_sparse_train(). Either:\n"
            "  1. Build the fork: cmake -DUSE_CUDA_SPARSE=ON && make\n"
            "  2. Use lgbm_integration.gpu_train() for co-processor path\n"
            "  3. Use standard lgb.train() for CPU histogram path"
        )

    # --- CUDA check ---
    cuda_info = _check_cuda_available()
    if not cuda_info['available']:
        raise RuntimeError(
            "CUDA is not available. gpu_sparse_train() requires a CUDA GPU.\n"
            f"Device count: {cuda_info['device_count']}, "
            f"driver version: {cuda_info['driver_version']}"
        )

    # --- Construct datasets ---
    # train_data must be constructed before Booster creation
    train_data.construct()
    if valid_sets is not None:
        for vs in valid_sets:
            vs.construct()

    # --- Create Booster ---
    # This calls LGBM_BoosterCreate -> Booster() -> GBDT::Init() ->
    # CreateTreeLearner("serial", "cuda_sparse") -> CUDASparseHistTreeLearner
    # -> Init(train_data). At this point the tree learner exists but has
    # no CSR data yet (has_external_csr_ = false).
    booster = lgb.Booster(params, train_data)

    # Add validation sets
    if valid_sets is not None:
        if valid_names is None:
            valid_names = [f'valid_{i}' for i in range(len(valid_sets))]
        for name, vs in zip(valid_names, valid_sets):
            booster.add_valid(vs, name)

    # --- Inject CSR data ---
    # This is the critical bridge call. The tree learner stores the CSR
    # host-side. GPU upload happens on the first ConstructHistograms().
    set_external_csr(booster, X_csr)

    # --- Training loop ---
    for i in range(num_boost_round):
        # Update one iteration (triggers ConstructHistograms -> GPU kernel)
        booster.update(fobj=None)

        # Evaluate
        eval_results = []
        if valid_sets is not None:
            eval_results = booster.eval_valid(feval=feval)

        # Run callbacks (version-agnostic)
        if callbacks:
            env = _make_callback_env(
                booster, params, i, num_boost_round, eval_results
            )
            try:
                for cb in callbacks:
                    cb(env)
            except _EarlyStopException as e:
                log.info(
                    "Early stopping at iteration %d (best: %d)",
                    i, getattr(e, 'best_iteration', i)
                )
                booster.best_iteration = getattr(e, 'best_iteration', i)
                booster.best_score = getattr(e, 'best_score', {})
                break

    return booster


# ---------------------------------------------------------------------------
# Simplified training function using lgb.train() internal callback machinery
# ---------------------------------------------------------------------------

def gpu_sparse_train_simple(
    params: dict,
    train_data: "lgb.Dataset",
    X_csr: "sp_sparse.csr_matrix",
    num_boost_round: int = 800,
    valid_sets: Optional[list] = None,
    valid_names: Optional[list] = None,
    callbacks: Optional[list] = None,
) -> "lgb.Booster":
    """Simplified GPU sparse training that patches lgb.train() internally.

    Uses a pre-training callback to inject the CSR data after Booster
    creation. This avoids reimplementing the full training loop and
    preserves all of LightGBM's callback/early-stopping machinery.

    Falls back to standard lgb.train() (CPU) if the fork is not available,
    with a warning.

    Parameters
    ----------
    params : dict
        LightGBM parameters. device_type forced to 'cuda_sparse' if
        fork is available.
    train_data : lgb.Dataset
        Training dataset.
    X_csr : scipy.sparse.csr_matrix
        Sparse binary cross feature matrix.
    num_boost_round : int
        Boosting iterations.
    valid_sets : list of lgb.Dataset or None
        Validation sets.
    valid_names : list of str or None
        Validation set names.
    callbacks : list or None
        LightGBM callbacks.

    Returns
    -------
    lgb.Booster
        Trained model.
    """
    if lgb is None:
        raise ImportError("lightgbm is required")

    params = params.copy()
    use_gpu = False

    if is_fork_available():
        cuda_info = _check_cuda_available()
        if cuda_info['available']:
            params['device_type'] = 'cuda_sparse'
            params.pop('device', None)
            params.pop('force_col_wise', None)
            params.pop('force_row_wise', None)
            use_gpu = True
            log.info(
                "GPU sparse histogram enabled: %s (%s GB free)",
                cuda_info['device_name'],
                f"{cuda_info['free_vram_bytes'] / (1024**3):.1f}"
            )
        else:
            log.warning(
                "Fork available but CUDA not detected. "
                "Falling back to CPU histogram."
            )
    else:
        log.info(
            "Forked LightGBM not available. Using standard CPU histogram."
        )

    if use_gpu:
        # Use the init_model callback trick: lgb.train() creates the Booster
        # internally, but we can't access it before training starts.
        # Instead, we create the Booster manually and pass it as init_model.
        train_data.construct()
        if valid_sets:
            for vs in valid_sets:
                vs.construct()

        booster = lgb.Booster(params, train_data)
        if valid_sets:
            if valid_names is None:
                valid_names = [f'valid_{i}' for i in range(len(valid_sets))]
            for name, vs in zip(valid_names, valid_sets):
                booster.add_valid(vs, name)

        # Inject CSR before any training
        set_external_csr(booster, X_csr)

        # Use booster's built-in training loop
        for i in range(num_boost_round):
            booster.update()

            # Run callbacks
            if callbacks:
                eval_res = booster.eval_valid() if valid_sets else []
                env = _make_callback_env(
                    booster, params, i, num_boost_round, eval_res
                )
                try:
                    for cb in callbacks:
                        cb(env)
                except _EarlyStopException as e:
                    log.info("Early stopping at iteration %d", i)
                    booster.best_iteration = getattr(e, 'best_iteration', i)
                    booster.best_score = getattr(e, 'best_score', {})
                    break

        return booster
    else:
        # Standard CPU path — lgb.train() handles everything
        return lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )


def _make_callback_env(booster, params, iteration, end_iteration,
                       evaluation_result_list):
    """Create a callback environment compatible with LightGBM's protocol.

    LightGBM's callback API changed across versions. This handles both
    the old CallbackEnv namedtuple and the newer dict-based approach.
    """
    try:
        # LightGBM >= 4.0 uses a simple namespace
        env = lgb.callback.CallbackEnv(
            model=booster,
            params=params,
            iteration=iteration,
            begin_iteration=0,
            end_iteration=end_iteration,
            evaluation_result_list=evaluation_result_list,
        )
    except (TypeError, AttributeError):
        # Older versions or different callback protocol — use a namespace
        class _Env:
            pass
        env = _Env()
        env.model = booster
        env.params = params
        env.iteration = iteration
        env.begin_iteration = 0
        env.end_iteration = end_iteration
        env.evaluation_result_list = evaluation_result_list
    return env


# ---------------------------------------------------------------------------
# Module self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )

    print("=" * 70)
    print("LightGBM CSR Bridge — Diagnostics")
    print("=" * 70)

    # Check LightGBM
    if lgb is None:
        print("  LightGBM: NOT INSTALLED")
    else:
        print(f"  LightGBM version: {lgb.__version__}")
        print(f"  Library path: {lgb.basic._LIB._name}")

    # Check fork
    fork_ok = is_fork_available()
    print(f"  Fork available (LGBM_BoosterSetExternalCSR): {fork_ok}")

    # Check CUDA
    cuda_info = _check_cuda_available()
    print(f"  CUDA available: {cuda_info['available']}")
    if cuda_info['available']:
        print(f"  Device count: {cuda_info['device_count']}")
        print(f"  Device name: {cuda_info['device_name']}")
        print(f"  Driver version: {cuda_info['driver_version']}")
        print(f"  Free VRAM: {cuda_info['free_vram_bytes'] / (1024**3):.1f} GB")
        print(f"  Total VRAM: {cuda_info['total_vram_bytes'] / (1024**3):.1f} GB")

    # Estimate VRAM for each TF
    print()
    print("  Per-Timeframe VRAM Estimates:")
    tf_specs = {
        '1w':  (818,    2_200_000, 6_600),
        '1d':  (5_733,  6_000_000, 18_000),
        '4h':  (23_000, 4_000_000, 12_000),
        '1h':  (100_000, 10_000_000, 30_000),
        '15m': (227_000, 10_000_000, 30_000),
    }
    for tf, (nr, nf, nnz_est) in tf_specs.items():
        nnz = int(nr * nf * 0.003)  # ~0.3% density
        est = estimate_csr_vram_bytes(nr, nnz, nf)
        print(f"    {tf:>4s}: {est / (1024**3):6.1f} GB  "
              f"({nr:>7,} rows x {nf:>10,} features)")

    # Check scipy
    if sp_sparse is None:
        print("\n  scipy: NOT INSTALLED")
    else:
        print(f"\n  scipy available: yes")

    print()
    if fork_ok and cuda_info['available']:
        print("  STATUS: READY for GPU sparse training")
    elif fork_ok and not cuda_info['available']:
        print("  STATUS: Fork available but no CUDA — CPU only")
    elif not fork_ok and cuda_info['available']:
        print("  STATUS: CUDA available but stock LightGBM — build the fork")
    else:
        print("  STATUS: No fork, no CUDA — CPU only")

    print("=" * 70)
