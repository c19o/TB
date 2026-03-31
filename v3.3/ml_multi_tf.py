#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ml_multi_tf.py -- Multi-Timeframe ML Trading System (v4 - LightGBM)
====================================================================
Pipeline:
1. HMM re-fitted per walk-forward window (no future leakage)
2. Triple-barrier labels: LONG(2)/FLAT(1)/SHORT(0) via ATR barriers
3. Rolling windows (not expanding) -- better for crypto regime drift
4. LightGBM GBDT with force_col_wise, max_bin=255 (EFB optimized), sparse-native
5. Nested validation: GA on inner fold, final metrics on outer untouched fold
6. ALL features used -- no SHAP pruning. LightGBM handles feature selection via tree splits.
7. multiclass objective with 3 classes (long_prob, flat_prob, short_prob)
"""

import sys, os, io, time, random, warnings, json, pickle, gc
# DO NOT re-wrap stdout — it kills python -u unbuffered mode on cloud
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, io.UnsupportedOperation):
    pass  # Already configured or not a TextIOWrapper
warnings.filterwarnings('ignore')

# Windows: add CUDA toolkit DLL directory so LightGBM can find cudart64_12.dll etc.
if sys.platform == 'win32':
    _cuda_bins = [
        os.path.join(os.environ.get('CUDA_PATH', r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6'), 'bin'),
    ]
    for _cb in _cuda_bins:
        if os.path.isdir(_cb):
            os.add_dll_directory(_cb)
            break

import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime
from numba import njit

def _fix_binary_preds(preds):
    """Convert binary 1D predictions to 2D [P(DOWN), P(UP)] for compat with 3-class eval code."""
    if preds.ndim == 1:
        return np.column_stack([1 - preds, preds])
    return preds

DB_DIR = os.environ.get('SAVAGE22_DB_DIR', os.path.dirname(os.path.abspath(__file__)))
# v3.1: resolve feature data from v3.0 shared dir — import from config (single source of truth)
try:
    from config import V30_DATA_DIR
except ImportError:
    # Fallback: use PROJECT_DIR (self-contained v3.3) or env var
    V30_DATA_DIR = os.environ.get("V30_DATA_DIR",
        os.path.dirname(os.path.abspath(__file__)))
START_TIME = time.time()
RESULTS = []

def elapsed():
    return f"[{time.time()-START_TIME:.0f}s]"

def log(msg):
    print(msg)
    RESULTS.append(msg)

# ============================================================
# INSTALL DEPS IF NEEDED
# ============================================================
try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    os.system("pip install hmmlearn")
    from hmmlearn.hmm import GaussianHMM

import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, log_loss
from scipy import stats
from scipy import sparse as sp_sparse
from feature_library import compute_triple_barrier_labels, TRIPLE_BARRIER_CONFIG

# Wave 3: Lift MKL/OpenBLAS thread caps — lets LightGBM's OpenMP use all cores
# Guard: skip in spawn'd CPCV workers (they set their own per-worker caps)
if os.environ.get('_CPCV_WORKER') != '1':
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=os.cpu_count() or 64, user_api='blas')
    except ImportError:
        pass

# ============================================================
# GPU SPARSE HISTOGRAM DETECTION
# The GPU fork adds set_external_csr() to lgb.Booster and supports
# device_type='cuda_sparse'. Detect availability at import time.
# ============================================================
_GPU_SPARSE_AVAILABLE = hasattr(lgb.Booster, 'set_external_csr')
_ALLOW_CPU = os.environ.get('ALLOW_CPU', '0') == '1'
if _GPU_SPARSE_AVAILABLE:
    log(f"LightGBM: GPU sparse histograms AVAILABLE (cuda_sparse fork detected)")
    if _ALLOW_CPU:
        log(f"  ALLOW_CPU=1 — will use CPU despite GPU availability")
else:
    log(f"LightGBM: CPU mode (force_col_wise=True) — GPU fork not detected")


# ============================================================
# MC-5: SIGTERM CHECKPOINT CALLBACK
# Saves model every 100 rounds + on SIGTERM for crash recovery
# ============================================================
import signal, threading

_SIGTERM_FLAG = threading.Event()

def _sigterm_handler(signum, frame):
    _SIGTERM_FLAG.set()
    log(f"  [SIGTERM] Received signal {signum} — will save checkpoint at next callback")

# Install handler (only in main process)
if threading.current_thread() is threading.main_thread():
    try:
        signal.signal(signal.SIGTERM, _sigterm_handler)
        signal.signal(signal.SIGINT, _sigterm_handler)
    except (OSError, ValueError):
        pass  # can't set signal handler in non-main thread or on Windows


class CheckpointCallback:
    """LightGBM callback: saves model every `period` rounds and on SIGTERM."""
    def __init__(self, save_path, period=100):
        self.save_path = save_path
        self.period = period
        self.order = 100  # run after early_stopping (order=10) and log_evaluation (order=20)

    def __call__(self, env):
        iteration = env.iteration + 1
        save_now = (iteration % self.period == 0) or _SIGTERM_FLAG.is_set()
        if save_now:
            try:
                import tempfile
                tmp_path = self.save_path + f'.tmp_{os.getpid()}'
                env.model.save_model(tmp_path)
                os.replace(tmp_path, self.save_path)
                if _SIGTERM_FLAG.is_set():
                    log(f"  [SIGTERM] Checkpoint saved at iteration {iteration}: {self.save_path}")
                    raise KeyboardInterrupt("SIGTERM checkpoint saved — stopping training")
            except KeyboardInterrupt:
                raise
            except Exception as e:
                log(f"  WARNING: checkpoint save failed at iter {iteration}: {e}")


# ============================================================
# COMBINATORIAL PURGED CROSS-VALIDATION (CPCV)
# Implementation based on Lopez de Prado's framework.
# ============================================================

from itertools import combinations


@njit(cache=True)
def _compute_uniqueness_inner(starts, ends, n_bars):
    """Numba-compiled inner loop for sample uniqueness (50-100x faster)."""
    concurrent = np.ones(n_bars, dtype=np.float64)
    for i in range(len(starts)):
        s, e = starts[i], min(ends[i], n_bars)
        for j in range(s, e):
            concurrent[j] += 1.0
    uniqueness = np.empty(len(starts), dtype=np.float64)
    for i in range(len(starts)):
        s, e = starts[i], min(ends[i], n_bars)
        total = 0.0
        count = 0
        for j in range(s, e):
            total += 1.0 / concurrent[j]
            count += 1
        uniqueness[i] = total / max(count, 1)
    return uniqueness


def _compute_sample_uniqueness(t0_arr, t1_arr, n_bars):
    """Compute average uniqueness per sample (Lopez de Prado method).

    For each event i with window [t0[i], t1[i]], uniqueness = average of
    1/N_t across all bars t in the window, where N_t = number of concurrent
    events at bar t.

    Args:
        t0_arr: array of event start indices
        t1_arr: array of event end indices (t0 + max_hold_bars)
        n_bars: total number of bars in the dataset

    Returns:
        uniqueness: array of shape (n_events,) with values in (0, 1]
    """
    if len(t0_arr) == 0:
        return np.ones(0, dtype=np.float64)
    starts = np.asarray(t0_arr, dtype=np.int64)
    ends = np.asarray(t1_arr, dtype=np.int64) + 1  # exclusive end
    return _compute_uniqueness_inner(starts, ends, n_bars)


def _generate_cpcv_splits(n_samples, n_groups=6, n_test_groups=2,
                          t0_arr=None, t1_arr=None, max_hold_bars=None,
                          embargo_pct=0.01, sample_paths=None, seed=42):
    """Generate Combinatorial Purged Cross-Validation splits.

    Args:
        n_samples: total number of samples
        n_groups: number of contiguous groups to split data into (default 6)
        n_test_groups: number of groups used as test in each path (default 2)
        t0_arr: event start indices (for purging). If None, uses sample index.
        t1_arr: event end indices (for purging). If None, t0 + max_hold_bars.
        max_hold_bars: maximum label horizon (for purging when t0/t1 not provided)
        embargo_pct: fraction of samples to embargo after each test boundary
        sample_paths: if set, sample this many paths (deterministic). None = exhaustive.
                      Guarantees every group appears in at least one test fold.
        seed: RNG seed for deterministic path sampling (default 42)

    Returns:
        list of (train_indices, test_indices) tuples, one per CPCV path
    """
    # Split into n_groups contiguous groups
    group_size = n_samples // n_groups
    groups = []
    for g in range(n_groups):
        start = g * group_size
        end = (g + 1) * group_size if g < n_groups - 1 else n_samples
        groups.append(np.arange(start, end))

    if max_hold_bars is not None:
        # Embargo must be >= max_hold_bars bars (prevents leakage from forward label horizon)
        # Formula: max(embargo_pct * n, max_hold_bars) — López de Prado
        effective_pct = max(embargo_pct, max_hold_bars / n_samples)
    else:
        effective_pct = embargo_pct
    embargo_size = max(1, int(n_samples * effective_pct))

    # Generate all combinatorial test paths
    all_paths = list(combinations(range(n_groups), n_test_groups))

    # ── Path sampling: deterministic subset with full group coverage ──
    if sample_paths is not None and sample_paths < len(all_paths):
        rng = np.random.RandomState(seed)
        # Step 1: greedily select paths to cover all groups
        uncovered = set(range(n_groups))
        coverage_paths = []
        remaining = list(range(len(all_paths)))
        rng.shuffle(remaining)  # randomize coverage order for diversity
        for idx in remaining:
            if not uncovered:
                break
            path_groups = set(all_paths[idx])
            if path_groups & uncovered:  # covers at least one new group
                coverage_paths.append(idx)
                uncovered -= path_groups
        # Step 2: fill remaining slots with random paths
        coverage_set = set(coverage_paths)
        extra_pool = [i for i in range(len(all_paths)) if i not in coverage_set]
        n_extra = sample_paths - len(coverage_paths)
        if n_extra > 0 and extra_pool:
            extra = rng.choice(extra_pool, size=min(n_extra, len(extra_pool)), replace=False)
            coverage_paths.extend(extra.tolist())
        # Sort for deterministic order
        coverage_paths.sort()
        all_paths = [all_paths[i] for i in coverage_paths]

    splits = []
    for test_group_ids in all_paths:
        # Test indices = union of selected groups
        test_idx = np.concatenate([groups[g] for g in test_group_ids])

        # Train indices = all other groups
        train_group_ids = [g for g in range(n_groups) if g not in test_group_ids]
        train_idx = np.concatenate([groups[g] for g in train_group_ids])

        # --- Purging (per-group: handles non-contiguous test groups) ---
        # CRITICAL: check each test group independently — using min/max of the
        # combined test set treats the gap between non-contiguous groups as test
        # range, which either over-purges (t0/t1) or under-purges (boundary).
        if t0_arr is not None and t1_arr is not None:
            train_t0 = t0_arr[train_idx]
            train_t1 = t1_arr[train_idx]
            purge_mask = np.zeros(len(train_idx), dtype=bool)
            for g in test_group_ids:
                g_start = groups[g][0]
                g_end = groups[g][-1]
                # Purge if label window [t0, t1] overlaps this test group [g_start, g_end]
                purge_mask |= (train_t1 >= g_start) & (train_t0 <= g_end)
            train_idx = train_idx[~purge_mask]
        elif max_hold_bars is not None:
            # Purge training samples whose forward label window overlaps ANY test group.
            # For sample at index i, label window = [i, i + max_hold_bars].
            # For test group g with range [g_start, g_end]:
            #   overlap iff i + max_hold_bars >= g_start AND i <= g_end
            #   → i >= g_start - max_hold_bars AND i <= g_end
            purge_mask = np.zeros(len(train_idx), dtype=bool)
            for g in test_group_ids:
                g_start = groups[g][0]
                g_end = groups[g][-1]
                purge_mask |= (train_idx >= g_start - max_hold_bars) & (train_idx <= g_end)
            train_idx = train_idx[~purge_mask]

        # --- Embargo ---
        # Remove training samples in embargo zone after each test group boundary
        for g in test_group_ids:
            test_end = groups[g][-1]
            embargo_start = test_end + 1
            embargo_end = test_end + embargo_size
            embargo_mask = (train_idx >= embargo_start) & (train_idx <= embargo_end)
            train_idx = train_idx[~embargo_mask]

        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))

    return splits


INT32_MAX = 2_147_483_647


def _ensure_lgbm_sparse_dtypes(X, label="matrix"):
    """Enforce correct CSR dtypes for LightGBM. One path, all TFs.
    - indptr: always int64 (row pointers — values can exceed int32 when NNZ > 2^31)
    - indices: always int32 (column IDs — values < n_features, always fits int32)
    LightGBM C API: indptr accepts int32/int64, indices requires int32.
    PR #1719 (2018) fixed silent corruption with large NNZ.
    No fallbacks. No conditional logic. Crash loud if assumptions violated.
    """
    if not hasattr(X, 'indptr'):
        return X
    if X.indptr.dtype != np.int64:
        X.indptr = X.indptr.astype(np.int64)
    if X.indices.dtype != np.int32:
        assert X.nnz == 0 or X.indices.max() <= INT32_MAX, (
            f"FATAL: {label} has column index {X.indices.max()} > int32 max. "
            f"This means > 2B features — LightGBM cannot handle this.")
        X.indices = X.indices.astype(np.int32)
    return X


def _predict_chunked(model, X, chunk_size=50000, num_iteration=None):
    """Predict in row chunks for large sparse matrices.
    Each chunk's CSR is small enough for LightGBM predict.
    Used for IS predictions where full train set NNZ may exceed int32.
    num_iteration: passed through to model.predict() for best-iteration scoring.
    """
    predict_kwargs = {} if num_iteration is None else {'num_iteration': num_iteration}
    if not hasattr(X, 'nnz') or X.nnz <= INT32_MAX:
        return model.predict(X, **predict_kwargs)
    n = X.shape[0]
    preds = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = X[start:end]
        preds.append(model.predict(chunk, **predict_kwargs))
    return np.vstack(preds)


def _use_gpu_sparse():
    """Check if GPU sparse training should be used."""
    return _GPU_SPARSE_AVAILABLE and not _ALLOW_CPU


def _train_gpu(params, ds_train, ds_val, X_train_csr, num_boost_round,
               early_stopping_rounds, checkpoint_cb=None, log_period=100,
               gpu_device_id=0):
    """Train with GPU sparse histograms via the CUDA fork.

    Uses lgb.Booster + manual update() loop with set_external_csr().
    Falls back to CPU lgb.train() if GPU init fails.

    Parameters
    ----------
    params : dict
        LightGBM params (device_type will be set to cuda_sparse).
    ds_train : lgb.Dataset
        Training dataset.
    ds_val : lgb.Dataset or None
        Validation dataset for early stopping.
    X_train_csr : scipy.sparse.csr_matrix
        CSR matrix for GPU histogram building (must match ds_train rows).
    num_boost_round : int
        Maximum boosting rounds.
    early_stopping_rounds : int
        Stop if no improvement for this many rounds.
    checkpoint_cb : CheckpointCallback or None
        Optional checkpoint callback.
    log_period : int
        Log evaluation every N rounds.

    Returns
    -------
    booster : lgb.Booster
        Trained model. Has .best_iteration set.
    """
    params = dict(params)
    params.pop('device', None)  # remove alias — 'device'='cpu' from V3_LGBM_PARAMS conflicts with device_type
    params['device_type'] = 'cuda_sparse'
    params['gpu_device_id'] = gpu_device_id
    # Remove force_col_wise — GPU path uses its own histogram builder
    params.pop('force_col_wise', None)
    # GPU fork needs histogram_pool_size to avoid OOM on large num_leaves
    if 'histogram_pool_size' not in params:
        params['histogram_pool_size'] = 512

    # No CPU fallback — GPU-or-nothing
    # If set_external_csr fails, crash loud
    booster = lgb.Booster(params, ds_train)
    booster.set_external_csr(X_train_csr)

    # Add validation set
    if ds_val is not None:
        booster.add_valid(ds_val, 'val')

    best_score = None
    best_iter = 0
    _higher_better = None  # auto-detected from first eval
    for i in range(num_boost_round):
        booster.update()

        # Early stopping check on validation set
        if ds_val is not None:
            val_result = booster.eval_valid()  # returns list of (ds_name, metric_name, value, higher_better)
            if val_result:
                score = val_result[0][2]  # first metric value
                if _higher_better is None:
                    _higher_better = val_result[0][3]  # auto-detect from LightGBM
                    best_score = score
                    best_iter = i + 1
                elif (_higher_better and score > best_score) or (not _higher_better and score < best_score):
                    best_score = score
                    best_iter = i + 1
                elif (i + 1) - best_iter >= early_stopping_rounds:
                    if log_period > 0:
                        log(f"    GPU early stopping at round {i+1} (best={best_iter})")
                    break

        # Log periodically
        if log_period > 0 and (i + 1) % log_period == 0:
            train_result = booster.eval_train()
            tr_str = f"train={train_result[0][2]:.4f}" if train_result else ""
            val_str = f"val={best_score:.4f}" if ds_val is not None else ""
            log(f"    [GPU] Round {i+1}/{num_boost_round}: {tr_str} {val_str}")

        # Checkpoint callback
        if checkpoint_cb is not None:
            class _FakeEnv:
                pass
            _env = _FakeEnv()
            _env.iteration = i
            _env.model = booster
            try:
                checkpoint_cb(_env)
            except KeyboardInterrupt:
                raise

        # SIGTERM check
        if _SIGTERM_FLAG.is_set():
            log(f"    [GPU] SIGTERM at round {i+1}")
            break

    # best_iter tracks the exact round with best val score.
    # LightGBM trees are immutable once added — predict(num_iteration=best_iter)
    # is mathematically identical to restoring from model_to_string() at that round
    # and avoids 87MB × N serializations per training run.
    booster.best_iteration = best_iter if best_iter > 0 else (i + 1)
    return booster


def _cpcv_worker_initializer(_nth_str):
    """Pre-load heavy modules once per spawn'd worker process.
    Called by ProcessPoolExecutor(initializer=...) — runs exactly once per worker,
    before any _cpcv_split_worker calls. This avoids re-importing numpy/scipy/lightgbm/
    sklearn on every fold, saving ~2-4s per worker startup."""
    import os
    os.environ['OMP_NUM_THREADS'] = _nth_str
    os.environ['MKL_NUM_THREADS'] = _nth_str
    os.environ['OPENBLAS_NUM_THREADS'] = _nth_str
    os.environ['NUMEXPR_NUM_THREADS'] = _nth_str
    os.environ['_CPCV_WORKER'] = '1'

    # Pre-import all heavy libraries into this worker's module cache
    import numpy          # noqa: F401
    from scipy import sparse  # noqa: F401
    import lightgbm       # noqa: F401
    from sklearn.metrics import accuracy_score, precision_score, log_loss  # noqa: F401
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=int(_nth_str), user_api='blas')
    except ImportError:
        pass
    try:
        from multiprocessing.shared_memory import SharedMemory  # noqa: F401
    except ImportError:
        pass


def _cpcv_split_worker(args_tuple):
    """Train a single LightGBM CPCV split.
    Runs in a subprocess for parallel CPU training.
    Returns: (wi, acc, prec_long, prec_short, mlogloss, best_iter,
              model_bytes, preds_3c, y_test, test_idx_valid,
              importance, is_acc, is_mlogloss, is_sharpe)
    On error: returns (wi, None, ..., None) with error logged to stderr.
    """
    (wi, train_idx, test_idx, X_data, X_indices, X_indptr, X_shape,
     y_3class, sample_weights, feature_cols, lgb_params,
     num_boost_round, tf_name, gpu_id,
     hmm_overlay, hmm_overlay_names) = args_tuple

    import os, sys, traceback

    # Cap ALL thread env vars BEFORE importing numerical libraries.
    # With spawn context, module-level code re-runs during import — setting these
    # env vars first prevents threadpool_limits(128) and OpenMP auto-detect(128)
    # from giving each worker 128 threads (causes oversubscription crashes).
    _nth = str(lgb_params.get('num_threads', 1))
    os.environ['OMP_NUM_THREADS'] = _nth
    os.environ['MKL_NUM_THREADS'] = _nth
    os.environ['OPENBLAS_NUM_THREADS'] = _nth
    os.environ['NUMEXPR_NUM_THREADS'] = _nth
    os.environ['_CPCV_WORKER'] = '1'  # tells module-level code to skip threadpool_limits(128)

    # These imports are near-instant if _cpcv_worker_initializer already loaded them
    import numpy as np
    from scipy import sparse
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score, precision_score, log_loss

    # Re-apply thread cap after imports (some libs reset on import)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=int(_nth), user_api='blas')
    except ImportError:
        pass

    try:
        # Wave 3: SharedMemory IPC — reconstruct CSR from shared memory if available
        if isinstance(X_data, dict):
            # X_data is actually _shm_info dict; X_indices/X_indptr/X_shape are None
            from multiprocessing.shared_memory import SharedMemory as _SHM
            _shm_info = X_data
            _shm_d = _SHM(name=_shm_info['data_name'], create=False)
            _shm_i = _SHM(name=_shm_info['indices_name'], create=False)
            _shm_p = _SHM(name=_shm_info['indptr_name'], create=False)
            _data = np.ndarray(_shm_info['data_shape'], dtype=np.dtype(_shm_info['data_dtype']), buffer=_shm_d.buf)
            _indices = np.ndarray(_shm_info['indices_shape'], dtype=np.dtype(_shm_info['indices_dtype']), buffer=_shm_i.buf)
            _indptr = np.ndarray(_shm_info['indptr_shape'], dtype=np.dtype(_shm_info['indptr_dtype']), buffer=_shm_p.buf)
            X_all = sparse.csr_matrix((_data, _indices, _indptr), shape=_shm_info['matrix_shape'], copy=False)
            _shm_d.close()
            _shm_i.close()
            _shm_p.close()
        else:
            # Reconstruct matrix in worker from pickle'd arrays
            X_all = sparse.csr_matrix((X_data, X_indices, X_indptr), shape=X_shape)

        y_train_raw = y_3class[train_idx]
        y_test_raw = y_3class[test_idx]
        train_valid = ~np.isnan(y_train_raw)
        test_valid = ~np.isnan(y_test_raw)

        X_train = X_all[train_idx][train_valid]
        y_train = y_train_raw[train_valid].astype(int)
        X_test = X_all[test_idx][test_valid]
        y_test = y_test_raw[test_valid].astype(int)

        # T-2 FIX: Apply per-fold HMM overlay (fitted on train-end-date only, no lookahead)
        # hmm_overlay is (N, n_hmm_cols) float32 pre-computed by the parent process per fold.
        # hstack only on the subset rows — cheap since overlay is small (4 cols).
        if hmm_overlay is not None and len(hmm_overlay_names) > 0:
            _Xtr_hmm = sparse.csr_matrix(hmm_overlay[train_idx][train_valid])
            X_train = sparse.hstack([X_train, _Xtr_hmm], format='csr')
            del _Xtr_hmm
            _Xte_hmm = sparse.csr_matrix(hmm_overlay[test_idx][test_valid])
            X_test = sparse.hstack([X_test, _Xte_hmm], format='csr')
            del _Xte_hmm
            feature_cols = list(feature_cols) + list(hmm_overlay_names)
        test_idx_valid = test_idx[test_valid]

        min_train = 50 if tf_name in ('1w', '1d') else 300
        min_test = 20 if tf_name in ('1w', '1d') else 50
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]
        if n_train < min_train or n_test < min_test:
            return (wi, None, None, None, None, None, None, None, None, None, None, None, None, None)

        w_train = sample_weights[train_idx][train_valid]

        params = lgb_params.copy()

        # 85/15 train/val split for early stopping — scale floor to dataset size
        # For 1w (443 train rows), floor=100 steals 23% leaving only 343 for training.
        # Use 15% target with floor scaled to max(20, n_train//10) instead.
        _val_floor = max(20, n_train // 10)
        val_size = max(int(n_train * 0.15), _val_floor)
        if val_size >= n_train - _val_floor:
            val_size = max(n_train // 5, 20)
        X_val_es = X_train[-val_size:]
        y_val_es = y_train[-val_size:]
        w_val_es = w_train[-val_size:]
        X_train_es = X_train[:-val_size]
        y_train_es = y_train[:-val_size]
        w_train_es = w_train[:-val_size]

        _w_ds_params = {'feature_pre_filter': False, 'max_bin': params.get('max_bin', 7), 'min_data_in_bin': 1}
        dtrain = lgb.Dataset(X_train_es, label=y_train_es, weight=w_train_es,
                             feature_name=feature_cols, free_raw_data=False, params=_w_ds_params)
        dval = lgb.Dataset(X_val_es, label=y_val_es, weight=w_val_es, feature_name=feature_cols, free_raw_data=False, params=_w_ds_params)
        _es_rounds_w = max(50, int(100 * (0.1 / params.get('learning_rate', 0.03))))
        from config import TF_ES_PATIENCE as _tf_es_cfg
        if tf_name in _tf_es_cfg:
            _es_rounds_w = _tf_es_cfg[tf_name]

        # GPU sparse path: detect in-worker (subprocess doesn't inherit parent globals)
        _worker_gpu = hasattr(lgb.Booster, 'set_external_csr') and os.environ.get('ALLOW_CPU', '0') != '1'
        if _worker_gpu and sparse.issparse(X_train_es):
            _X_csr_w = X_train_es.tocsr() if not isinstance(X_train_es, sparse.csr_matrix) else X_train_es
            _gpu_params = dict(params)
            _gpu_params['device_type'] = 'cuda_sparse'
            _gpu_params.pop('force_col_wise', None)
            if 'histogram_pool_size' not in _gpu_params:
                _gpu_params['histogram_pool_size'] = 512
            # No CPU fallback — GPU-or-nothing
            # If set_external_csr fails, crash loud
            booster = lgb.Booster(_gpu_params, dtrain)
            booster.set_external_csr(_X_csr_w)
            booster.add_valid(dval, 'val')
            best_score_w = None
            best_iter_w = 0
            _hb_w = None  # higher_better flag
            for _ri in range(num_boost_round):
                booster.update()
                val_result = booster.eval_valid()
                if val_result:
                    _sc = val_result[0][2]
                    if _hb_w is None:
                        _hb_w = val_result[0][3]
                        best_score_w = _sc
                        best_iter_w = _ri + 1
                    elif (_hb_w and _sc > best_score_w) or (not _hb_w and _sc < best_score_w):
                        best_score_w = _sc
                        best_iter_w = _ri + 1
                    elif (_ri + 1) - best_iter_w >= _es_rounds_w:
                        break
            model = booster
            model.best_iteration = best_iter_w if best_iter_w > 0 else (_ri + 1)
            del _X_csr_w
        else:
            model = lgb.train(
                params, dtrain,
                num_boost_round=num_boost_round,
                valid_sets=[dtrain, dval],
                valid_names=['train', 'val'],
                callbacks=[lgb.early_stopping(_es_rounds_w), lgb.log_evaluation(0)],
            )

        # OOS predictions — use best_iteration so GPU path (which no longer truncates via
        # model_to_string) evaluates only the first best_iter trees, identical to the
        # old model_to_string() restore but without 87MB × N serializations.
        preds_3c = model.predict(X_test, num_iteration=model.best_iteration)
        pred_labels = np.argmax(preds_3c, axis=1)
        acc = float(accuracy_score(y_test, pred_labels))
        prec_long = float(precision_score(y_test, pred_labels, labels=[2], average='macro', zero_division=0))
        prec_short = float(precision_score(y_test, pred_labels, labels=[0], average='macro', zero_division=0))
        mlogloss = float(log_loss(y_test, preds_3c, labels=[0, 1, 2]))

        # IS metrics for PBO
        is_preds_3c = model.predict(X_train, num_iteration=model.best_iteration)
        is_pred_labels = np.argmax(is_preds_3c, axis=1)
        is_acc = float(accuracy_score(y_train, is_pred_labels))
        is_mlogloss = float(log_loss(y_train, is_preds_3c, labels=[0, 1, 2]))
        _is_sim_ret = np.where(is_pred_labels == y_train, 1.0, -1.0)
        _is_std = np.std(_is_sim_ret, ddof=1)
        is_sharpe = float(np.mean(_is_sim_ret) / max(_is_std, 1e-10) * np.sqrt(252))

        importance = dict(zip(model.feature_name(), model.feature_importance(importance_type='gain')))

        # Serialize model
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix='.txt', delete=False)
        tmp.close()
        model.save_model(tmp.name)
        with open(tmp.name, 'rb') as f:
            model_bytes = f.read()
        import os as _os
        _os.unlink(tmp.name)

        return (wi, acc, prec_long, prec_short, mlogloss, model.best_iteration,
                model_bytes, preds_3c.copy(), y_test.copy(), test_idx_valid.copy(),
                importance, is_acc, is_mlogloss, is_sharpe)

    except Exception as _worker_err:
        # Log to stderr (visible in parent) and return graceful failure tuple
        print(f"[CPCV WORKER {wi}] CRASH: {type(_worker_err).__name__}: {_worker_err}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return (wi, None, None, None, None, None, None, None, None, None, None, None, None, None)


def _isolated_fold_worker(shared_dir, wi, train_idx, test_idx,
                          lgb_params, feature_cols, num_boost_round, tf_name,
                          hmm_overlay_names, result_path, db_dir):
    """Train a single CPCV fold in an isolated subprocess.

    Loads X_all from .npy files (mmap), trains one fold, saves results to pickle,
    then exits. OS reclaims ALL memory — no allocator fragmentation across folds.
    """
    import os, pickle, gc
    import numpy as np
    from scipy import sparse
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score, precision_score, log_loss

    # Load shared data — mmap for zero-copy read, OS pages in on demand
    from atomic_io import load_csr_npy
    X_all = load_csr_npy(shared_dir, mmap_mode='r')
    y_3class = np.load(os.path.join(shared_dir, 'y.npy'), mmap_mode='r')
    sample_weights = np.load(os.path.join(shared_dir, 'weights.npy'), mmap_mode='r')

    # Load HMM overlay if present
    _hmm_path = os.path.join(shared_dir, f'hmm_overlay_fold{wi}.npy')
    hmm_overlay = np.load(_hmm_path) if os.path.exists(_hmm_path) else None

    # Extract train/test
    y_train_raw = y_3class[train_idx]
    y_test_raw = y_3class[test_idx]
    train_valid = ~np.isnan(y_train_raw)
    test_valid = ~np.isnan(y_test_raw)

    X_train = X_all[train_idx][train_valid]
    X_test = X_all[test_idx][test_valid]
    y_train = np.array(y_train_raw[train_valid]).astype(int)
    y_test = np.array(y_test_raw[test_valid]).astype(int)

    # Apply per-fold HMM overlay
    if hmm_overlay is not None and len(hmm_overlay_names) > 0:
        _Xtr_hmm = sparse.csr_matrix(hmm_overlay[train_idx][train_valid])
        X_train = sparse.hstack([X_train, _Xtr_hmm], format='csr')
        del _Xtr_hmm
        _Xte_hmm = sparse.csr_matrix(hmm_overlay[test_idx][test_valid])
        X_test = sparse.hstack([X_test, _Xte_hmm], format='csr')
        del _Xte_hmm
        feature_cols = list(feature_cols) + list(hmm_overlay_names)
    test_idx_valid = test_idx[test_valid]

    min_train = 50 if tf_name in ('1w', '1d') else 300
    min_test = 20 if tf_name in ('1w', '1d') else 50
    if X_train.shape[0] < min_train or X_test.shape[0] < min_test:
        with open(result_path, 'wb') as f:
            pickle.dump({'wi': wi, 'skip': True}, f)
        return

    w_train = np.array(sample_weights[train_idx][train_valid])
    params = lgb_params.copy()

    # 85/15 train/val split for early stopping — floor scales with dataset size
    n_tr = X_train.shape[0]
    _val_floor_i = max(20, n_tr // 10)
    val_size = max(int(n_tr * 0.15), _val_floor_i)
    if val_size >= n_tr - _val_floor_i:
        val_size = max(n_tr // 5, 20)
    X_val_es = X_train[-val_size:]
    y_val_es = y_train[-val_size:]
    w_val_es = w_train[-val_size:]
    X_train_es = X_train[:-val_size]
    y_train_es = y_train[:-val_size]
    w_train_es = w_train[:-val_size]

    _ds_params = {'feature_pre_filter': False, 'max_bin': params.get('max_bin', 7), 'min_data_in_bin': 1}
    dtrain = lgb.Dataset(X_train_es, label=y_train_es, weight=w_train_es,
                         feature_name=feature_cols, free_raw_data=False, params=_ds_params)
    dval = lgb.Dataset(X_val_es, label=y_val_es, weight=w_val_es,
                       feature_name=feature_cols, free_raw_data=False, params=_ds_params)
    _es_rounds = max(50, int(100 * (0.1 / params.get('learning_rate', 0.03))))
    from config import TF_ES_PATIENCE as _tf_es_cfg2
    if tf_name in _tf_es_cfg2:
        _es_rounds = _tf_es_cfg2[tf_name]

    model = lgb.train(
        params, dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dtrain, dval],
        valid_names=['train', 'val'],
        callbacks=[lgb.early_stopping(_es_rounds), lgb.log_evaluation(100)],
    )

    # OOS predictions
    preds_3c = model.predict(X_test, num_iteration=model.best_iteration)
    pred_labels = np.argmax(preds_3c, axis=1)
    acc = float(accuracy_score(y_test, pred_labels))
    prec_long = float(precision_score(y_test, pred_labels, labels=[2], average='macro', zero_division=0))
    prec_short = float(precision_score(y_test, pred_labels, labels=[0], average='macro', zero_division=0))
    mlogloss = float(log_loss(y_test, preds_3c, labels=[0, 1, 2]))

    # IS metrics for PBO
    is_preds_3c = model.predict(X_train, num_iteration=model.best_iteration)
    is_pred_labels = np.argmax(is_preds_3c, axis=1)
    is_acc = float(accuracy_score(y_train, is_pred_labels))
    is_mlogloss = float(log_loss(y_train, is_preds_3c, labels=[0, 1, 2]))
    _is_sim_ret = np.where(is_pred_labels == y_train, 1.0, -1.0)
    _is_std = np.std(_is_sim_ret, ddof=1)
    is_sharpe = float(np.mean(_is_sim_ret) / max(_is_std, 1e-10) * np.sqrt(252))

    importance = dict(zip(model.feature_name(), model.feature_importance(importance_type='gain')))

    # Save fold model
    _fold_model_path = os.path.join(db_dir, f'model_{tf_name}_fold{wi}.txt')
    model.save_model(_fold_model_path)

    # Save results for parent
    with open(result_path, 'wb') as f:
        pickle.dump({
            'wi': wi, 'skip': False,
            'acc': acc, 'prec_long': prec_long, 'prec_short': prec_short,
            'mlogloss': mlogloss, 'best_iteration': model.best_iteration,
            'preds_3c': preds_3c, 'y_test': y_test,
            'test_idx_valid': test_idx_valid,
            'importance': importance,
            'train_size': X_train.shape[0], 'test_size': X_test.shape[0],
            'is_acc': is_acc, 'is_mlogloss': is_mlogloss, 'is_sharpe': is_sharpe,
        }, f)
    # Subprocess exits here — OS reclaims ALL heap memory (no fragmentation)


# ============================================================
# FOLD-PARALLEL CPCV ACROSS MULTIPLE GPUs
# Each fold trains in a separate subprocess with its own GPU via
# CUDA_VISIBLE_DEVICES isolation. Data shared via mmap'd .npy files.
# ============================================================

def _detect_gpu_count():
    """Detect number of NVIDIA GPUs available."""
    from config import CPCV_PARALLEL_GPUS
    if CPCV_PARALLEL_GPUS > 0:
        return CPCV_PARALLEL_GPUS
    try:
        import subprocess as _sp
        _nv = _sp.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=5)
        if _nv.returncode == 0 and _nv.stdout.strip():
            return _nv.stdout.strip().count('\n') + 1
    except Exception:
        pass
    return 1


def _gpu_fold_worker(fold_id, gpu_id, shared_dir, train_idx, test_idx,
                     lgb_params, feature_cols, num_boost_round, tf_name,
                     hmm_overlay_names, result_path, db_dir):
    """Train a single CPCV fold on a specific GPU in an isolated subprocess.

    CUDA_VISIBLE_DEVICES is set BEFORE importing LightGBM, ensuring clean
    GPU isolation per process. Data is loaded via mmap'd .npy files (zero-copy).
    """
    import os
    # CRITICAL: Set CUDA_VISIBLE_DEVICES BEFORE importing LightGBM/CUDA
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    import pickle, gc
    import numpy as np
    from scipy import sparse
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score, precision_score, log_loss

    try:
        # Load shared data — mmap for zero-copy read, OS pages in on demand
        from atomic_io import load_csr_npy
        X_all = load_csr_npy(shared_dir, mmap_mode='r')
        y_3class = np.load(os.path.join(shared_dir, 'y.npy'), mmap_mode='r')
        sample_weights = np.load(os.path.join(shared_dir, 'weights.npy'), mmap_mode='r')

        # Load HMM overlay if present
        _hmm_path = os.path.join(shared_dir, f'hmm_overlay_fold{fold_id}.npy')
        hmm_overlay = np.load(_hmm_path) if os.path.exists(_hmm_path) else None

        # Extract train/test
        y_train_raw = y_3class[train_idx]
        y_test_raw = y_3class[test_idx]
        train_valid = ~np.isnan(y_train_raw)
        test_valid = ~np.isnan(y_test_raw)

        X_train = X_all[train_idx][train_valid]
        X_test = X_all[test_idx][test_valid]
        y_train = np.array(y_train_raw[train_valid]).astype(int)
        y_test = np.array(y_test_raw[test_valid]).astype(int)

        # Apply per-fold HMM overlay
        if hmm_overlay is not None and len(hmm_overlay_names) > 0:
            _Xtr_hmm = sparse.csr_matrix(hmm_overlay[train_idx][train_valid])
            X_train = sparse.hstack([X_train, _Xtr_hmm], format='csr')
            del _Xtr_hmm
            _Xte_hmm = sparse.csr_matrix(hmm_overlay[test_idx][test_valid])
            X_test = sparse.hstack([X_test, _Xte_hmm], format='csr')
            del _Xte_hmm
            feature_cols = list(feature_cols) + list(hmm_overlay_names)
        test_idx_valid = test_idx[test_valid]

        min_train = 50 if tf_name in ('1w', '1d') else 300
        min_test = 20 if tf_name in ('1w', '1d') else 50
        if X_train.shape[0] < min_train or X_test.shape[0] < min_test:
            with open(result_path, 'wb') as f:
                pickle.dump({'wi': fold_id, 'skip': True}, f)
            return

        w_train = np.array(sample_weights[train_idx][train_valid])
        params = lgb_params.copy()

        # 85/15 train/val split for early stopping — floor scales with dataset size
        n_tr = X_train.shape[0]
        _val_floor_g = max(20, n_tr // 10)
        val_size = max(int(n_tr * 0.15), _val_floor_g)
        if val_size >= n_tr - _val_floor_g:
            val_size = max(n_tr // 5, 20)
        X_val_es = X_train[-val_size:]
        y_val_es = y_train[-val_size:]
        w_val_es = w_train[-val_size:]
        X_train_es = X_train[:-val_size]
        y_train_es = y_train[:-val_size]
        w_train_es = w_train[:-val_size]

        _ds_params = {'feature_pre_filter': False, 'max_bin': params.get('max_bin', 7), 'min_data_in_bin': 1}
        dtrain = lgb.Dataset(X_train_es, label=y_train_es, weight=w_train_es,
                             feature_name=feature_cols, free_raw_data=False, params=_ds_params)
        dval = lgb.Dataset(X_val_es, label=y_val_es, weight=w_val_es,
                           feature_name=feature_cols, free_raw_data=False, params=_ds_params)
        _es_rounds = max(50, int(100 * (0.1 / params.get('learning_rate', 0.03))))
        from config import TF_ES_PATIENCE as _tf_es_cfg3
        if tf_name in _tf_es_cfg3:
            _es_rounds = _tf_es_cfg3[tf_name]

        # GPU cuda_sparse training — after CUDA_VISIBLE_DEVICES isolation,
        # the visible GPU is always device 0 from this process's perspective
        _worker_has_gpu = hasattr(lgb.Booster, 'set_external_csr')
        if _worker_has_gpu and sparse.issparse(X_train_es):
            _X_csr = X_train_es.tocsr() if not isinstance(X_train_es, sparse.csr_matrix) else X_train_es
            _gpu_params = dict(params)
            _gpu_params['device_type'] = 'cuda_sparse'
            _gpu_params['gpu_device_id'] = 0  # always 0 — CUDA_VISIBLE_DEVICES maps physical GPU
            _gpu_params.pop('force_col_wise', None)
            _gpu_params.pop('force_row_wise', None)
            _gpu_params.pop('device', None)
            if 'histogram_pool_size' not in _gpu_params:
                _gpu_params['histogram_pool_size'] = 512

            booster = lgb.Booster(_gpu_params, dtrain)
            booster.set_external_csr(_X_csr)
            booster.add_valid(dval, 'val')

            best_score = None
            best_iter = 0
            _hb = None
            for ri in range(num_boost_round):
                booster.update()
                val_result = booster.eval_valid()
                if val_result:
                    sc = val_result[0][2]
                    if _hb is None:
                        _hb = val_result[0][3]
                        best_score = sc
                        best_iter = ri + 1
                    elif (_hb and sc > best_score) or (not _hb and sc < best_score):
                        best_score = sc
                        best_iter = ri + 1
                    elif (ri + 1) - best_iter >= _es_rounds:
                        break
            model = booster
            model.best_iteration = best_iter if best_iter > 0 else (ri + 1)
            del _X_csr
        else:
            # CPU fallback (no GPU fork in this worker)
            model = lgb.train(
                params, dtrain,
                num_boost_round=num_boost_round,
                valid_sets=[dtrain, dval],
                valid_names=['train', 'val'],
                callbacks=[lgb.early_stopping(_es_rounds), lgb.log_evaluation(100)],
            )

        # OOS predictions
        preds_3c = model.predict(X_test, num_iteration=model.best_iteration)
        pred_labels = np.argmax(preds_3c, axis=1)
        acc = float(accuracy_score(y_test, pred_labels))
        prec_long = float(precision_score(y_test, pred_labels, labels=[2], average='macro', zero_division=0))
        prec_short = float(precision_score(y_test, pred_labels, labels=[0], average='macro', zero_division=0))
        mlogloss = float(log_loss(y_test, preds_3c, labels=[0, 1, 2]))

        # IS metrics for PBO
        is_preds_3c = model.predict(X_train, num_iteration=model.best_iteration)
        is_pred_labels = np.argmax(is_preds_3c, axis=1)
        is_acc = float(accuracy_score(y_train, is_pred_labels))
        is_mlogloss = float(log_loss(y_train, is_preds_3c, labels=[0, 1, 2]))
        _is_sim_ret = np.where(is_pred_labels == y_train, 1.0, -1.0)
        _is_std = np.std(_is_sim_ret, ddof=1)
        is_sharpe = float(np.mean(_is_sim_ret) / max(_is_std, 1e-10) * np.sqrt(252))

        importance = dict(zip(model.feature_name(), model.feature_importance(importance_type='gain')))

        # Save fold model
        _fold_model_path = os.path.join(db_dir, f'model_{tf_name}_fold{fold_id}.txt')
        model.save_model(_fold_model_path)

        # Save results for parent
        with open(result_path, 'wb') as f:
            pickle.dump({
                'wi': fold_id, 'skip': False,
                'acc': acc, 'prec_long': prec_long, 'prec_short': prec_short,
                'mlogloss': mlogloss, 'best_iteration': model.best_iteration,
                'preds_3c': preds_3c, 'y_test': y_test,
                'test_idx_valid': test_idx_valid,
                'importance': importance,
                'train_size': X_train.shape[0], 'test_size': X_test.shape[0],
                'is_acc': is_acc, 'is_mlogloss': is_mlogloss, 'is_sharpe': is_sharpe,
            }, f)

        del model, dtrain, dval, X_train, X_test, X_all
        gc.collect()

    except Exception as e:
        import traceback
        with open(result_path, 'wb') as f:
            pickle.dump({
                'wi': fold_id, 'skip': False,
                'error': str(e), 'traceback': traceback.format_exc(),
                'gpu_id': gpu_id,
            }, f)
    # Subprocess exits here — OS reclaims ALL heap memory (no CUDA fragmentation)


def run_cpcv_gpu_parallel(splits, completed_folds, shared_dir, lgb_params,
                          feature_cols, num_boost_round, tf_name,
                          hmm_overlay_names, db_dir, n_gpus=None):
    """Run CPCV folds in parallel across multiple GPUs via subprocess isolation.

    Each fold trains in its own subprocess with CUDA_VISIBLE_DEVICES set before
    importing LightGBM, ensuring clean GPU isolation. Data is pre-saved as
    mmap'd .npy files in shared_dir by the caller.

    Returns list of result dicts (same format as _isolated_fold_worker output).
    """
    import multiprocessing as mp
    import tempfile

    if n_gpus is None or n_gpus <= 0:
        n_gpus = _detect_gpu_count()
    n_gpus = max(1, n_gpus)

    pending = [(wi, train_idx, test_idx) for wi, (train_idx, test_idx)
               in enumerate(splits) if wi not in completed_folds]

    log(f"  GPU-PARALLEL CPCV: {n_gpus} GPUs, {len(splits)} total folds, "
        f"{len(pending)} pending")

    ctx = mp.get_context('spawn')
    all_results = []

    for batch_start in range(0, len(pending), n_gpus):
        batch = pending[batch_start:batch_start + n_gpus]
        batch_num = batch_start // n_gpus + 1
        total_batches = (len(pending) + n_gpus - 1) // n_gpus
        log(f"\n  {elapsed()} Batch {batch_num}/{total_batches}: "
            f"folds {[b[0]+1 for b in batch]}")

        procs = []
        result_paths = []
        for i, (fold_id, train_idx, test_idx) in enumerate(batch):
            gpu_id = i % n_gpus
            result_path = os.path.join(shared_dir, f'gpu_fold_{fold_id}_result.pkl')
            result_paths.append((fold_id, gpu_id, result_path))

            p = ctx.Process(
                target=_gpu_fold_worker,
                args=(fold_id, gpu_id, shared_dir, train_idx, test_idx,
                      lgb_params, feature_cols, num_boost_round, tf_name,
                      hmm_overlay_names, result_path, db_dir),
            )
            p.start()
            procs.append((fold_id, gpu_id, p))
            log(f"    Fold {fold_id+1} → GPU {gpu_id} (PID {p.pid})")

        # Wait for all in batch
        for fold_id, gpu_id, p in procs:
            p.join(timeout=7200)  # 2hr max per fold
            if p.is_alive():
                log(f"    WARNING: Fold {fold_id+1} on GPU {gpu_id} timed out — terminating")
                p.terminate()
                p.join(timeout=30)

        # Collect results from this batch
        for fold_id, gpu_id, result_path in result_paths:
            if not os.path.exists(result_path):
                log(f"    Fold {fold_id+1} (GPU {gpu_id}): NO OUTPUT")
                continue
            with open(result_path, 'rb') as f:
                result = pickle.load(f)
            if result.get('skip'):
                log(f"    Fold {fold_id+1}: SKIP (not enough samples)")
            elif 'error' in result:
                log(f"    Fold {fold_id+1} (GPU {gpu_id}): ERROR — {result['error']}")
                log(f"      {result['traceback'][:500]}")
            else:
                log(f"    Fold {fold_id+1} (GPU {gpu_id}): "
                    f"Acc={result['acc']:.3f} PrecL={result['prec_long']:.3f} "
                    f"PrecS={result['prec_short']:.3f} mlogloss={result['mlogloss']:.4f} "
                    f"Trees={result['best_iteration']}")
            all_results.append(result)

    log(f"\n  {elapsed()} GPU-PARALLEL CPCV complete: "
        f"{sum(1 for r in all_results if not r.get('skip') and 'error' not in r)} succeeded, "
        f"{sum(1 for r in all_results if 'error' in r)} errors")

    return all_results


if __name__ == '__main__':
  # ============================================================
  # LOAD DAILY DATA FOR HMM (will re-fit per window)
  # ============================================================
  log(f"\n{elapsed()} Loading daily closes for HMM...")
  _btc_db = f'{DB_DIR}/btc_prices.db'
  if not os.path.exists(_btc_db) or os.path.getsize(_btc_db) == 0:
      _btc_db = os.path.join(os.path.dirname(DB_DIR), 'btc_prices.db')
  conn = sqlite3.connect(_btc_db)
  daily = pd.read_sql_query("""
      SELECT open_time, close FROM ohlcv
      WHERE timeframe='1d' AND symbol='BTC/USDT' ORDER BY open_time
  """, conn)
  conn.close()
  daily['timestamp'] = pd.to_datetime(daily['open_time'], unit='ms', utc=True)
  daily['close'] = pd.to_numeric(daily['close'], errors='coerce')
  daily = daily.dropna(subset=['close']).set_index('timestamp').sort_index()
  daily_returns = np.log(daily['close'] / daily['close'].shift(1)).dropna()
  daily_abs_ret = daily_returns.abs()
  daily_vol10 = daily_returns.rolling(10).std().dropna()
  common_daily_idx = daily_returns.index.intersection(daily_vol10.index)
  log(f"  Daily data: {len(common_daily_idx)} days for HMM")

  def fit_hmm_on_window(end_date):
      """Fit HMM on daily data up to end_date (no future leakage)."""
      # Ensure timezone compatibility
      if hasattr(common_daily_idx, 'tz') and common_daily_idx.tz is not None:
          if end_date.tzinfo is None:
              end_date = end_date.tz_localize('UTC')
      else:
          if end_date.tzinfo is not None:
              end_date = end_date.tz_localize(None)
      mask = common_daily_idx <= end_date
      idx = common_daily_idx[mask]
      if len(idx) < 200:
          return None
      r = daily_returns.loc[idx].values
      ar = daily_abs_ret.loc[idx].values
      v = daily_vol10.loc[idx].values
      X = np.column_stack([r, ar, v])

      best_score = -np.inf
      best_model = None
      for seed in range(3):
          try:
              model = GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=seed)
              model.fit(X)
              score = model.score(X)
              if score > best_score:
                  best_score = score
                  best_model = model
          except Exception:
              pass

      if best_model is None:
          return None

      probs = best_model.predict_proba(X)
      states = best_model.predict(X)

      # Label states by mean return
      state_means = {}
      for s in range(3):
          state_means[s] = r[states == s].mean() if (states == s).sum() > 0 else 0
      sorted_states = sorted(state_means.keys(), key=lambda s: state_means[s])

      hmm_df = pd.DataFrame({
          'hmm_bull_prob': probs[:, sorted_states[2]],
          'hmm_bear_prob': probs[:, sorted_states[0]],
          'hmm_neutral_prob': probs[:, sorted_states[1]],
          'hmm_state': [sorted_states.index(s) for s in states],
      }, index=idx)
      hmm_df.index = hmm_df.index.normalize()
      return hmm_df

  # ============================================================
  # TF CONFIGS (Perplexity-adjusted regularization)
  # ============================================================
  # ── LightGBM base params — single source of truth from config.py ──
  from config import (V3_LGBM_PARAMS as _CFG_LGBM, TF_MIN_DATA_IN_LEAF as _CFG_MIN_LEAF,
                      TF_LEARNING_RATE as _CFG_TF_LR, TF_NUM_BOOST_ROUND as _CFG_TF_ROUNDS,
                      TF_ES_PATIENCE as _CFG_TF_ES)
  V2_LGBM_PARAMS = _CFG_LGBM.copy()
  _MIN_DATA_IN_LEAF = _CFG_MIN_LEAF.copy()

  TF_CONFIGS = {
      '1w': {
          'db': 'features_1w.db', 'table': 'features_1w',
          'return_col': 'next_1w_return',
          'cost_pct': 0.0025,
          'context_only': False,
          'rolling_window_bars': None,  # use all available (339 rows)
      },
      '1d': {
          'db': 'features_1d.db', 'table': 'features_1d',
          'return_col': 'next_1d_return',
          'cost_pct': 0.0025,
          'context_only': False,
          'rolling_window_bars': None,  # use all available (2368 rows)
      },
      '4h': {
          'db': 'features_4h.db', 'table': 'features_4h',
          'return_col': 'next_4h_return',
          'cost_pct': 0.24,
          'context_only': False,
          'rolling_window_bars': 8760,  # ~18 months of 4H bars
      },
      '1h': {
          'db': 'features_1h.db', 'table': 'features_1h',
          'return_col': 'next_1h_return',
          'cost_pct': 0.23,
          'context_only': False,
          'rolling_window_bars': 25000,  # ~2.8 years of 1H bars
      },
      '15m': {
          'db': 'features_15m.db', 'table': 'features_15m',
          'return_col': 'next_15m_return',
          'cost_pct': 0.22,
          'context_only': False,
          'rolling_window_bars': 70000,  # ~6 months of 15m bars
      },
  }

  # ============================================================
  # CLI ARGS
  # ============================================================
  import argparse
  import multiprocessing as _mp_ctx
  try:
      _mp_ctx.set_start_method('spawn', force=True)  # CRITICAL: fork + LightGBM OpenMP = deadlock
  except RuntimeError:
      pass  # already set
  from concurrent.futures import ProcessPoolExecutor
  from concurrent.futures.process import BrokenProcessPool
  _parser = argparse.ArgumentParser()
  _parser.add_argument('--tf', action='append', help='Only train specific TFs (can repeat)')
  _parser.add_argument('--boost-rounds', type=int, default=800, help='LightGBM num_boost_round (default 800)')
  _parser.add_argument('--n-groups', type=int, default=None, help='Override CPCV n_groups (default: per-TF)')
  _parser.add_argument('--search-mode', action='store_true', default=False,
                        help='Use OPTUNA_PHASE1_CPCV_GROUPS for faster Optuna search trials')
  _parser.add_argument('--parallel-splits', action='store_true', default=False,
                        help='(legacy, now auto-detected) Kept for backward compat')
  # --no-parallel-splits removed: auto-detected from matrix type or env V3_FORCE_SEQUENTIAL=1
  _parser.add_argument('--subprocess-folds', action='store_true', default=False,
                        help='Train each CPCV fold in isolated subprocess (OS reclaims all memory per fold)')
  _args, _unknown = _parser.parse_known_args()  # ignore unknown args (e.g. from smoke_test)
  _tf_filter = set(_args.tf) if _args.tf else None

  # LightGBM CPU mode: parallel splits use ProcessPoolExecutor across CPU cores
  # Auto-detect: env V3_FORCE_SEQUENTIAL=1 or dense data → sequential
  _force_sequential = os.environ.get('V3_FORCE_SEQUENTIAL', '0') == '1'
  _use_parallel_splits = not _force_sequential
  try:
      from hardware_detect import get_cpu_count
      _total_cores = get_cpu_count()
  except ImportError:
      import multiprocessing as _mp
      _total_cores = _mp.cpu_count() or 24
  if _force_sequential:
      log("PARALLEL SPLITS: disabled (V3_FORCE_SEQUENTIAL=1)")
  else:
      log(f"PARALLEL SPLITS: enabled (dynamic workers per TF, {_total_cores} cores detected)")

  # ============================================================
  # MAIN TRAINING LOOP
  # ============================================================
  all_results = {}

  try:
      for tf_name, cfg in TF_CONFIGS.items():
          if _tf_filter and tf_name not in _tf_filter:
              continue
          log(f"\n{'='*70}")
          log(f"TRAINING {tf_name.upper()} MODEL")
          log(f"{'='*70}")

          db_path = f"{DB_DIR}/{cfg['db']}"
          parquet_path = db_path.replace('.db', '.parquet')
          # V2 naming: features_BTC_{tf}.parquet (asset-prefixed)
          v2_parquet = os.path.join(DB_DIR, f'features_BTC_{tf_name}.parquet')
          # v3.1: also check v3.0 shared data dir
          v30_parquet = os.path.join(V30_DATA_DIR, f'features_BTC_{tf_name}.parquet')
          if not os.path.exists(parquet_path) and os.path.exists(v2_parquet):
              parquet_path = v2_parquet
          if not os.path.exists(parquet_path) and os.path.exists(v30_parquet):
              parquet_path = v30_parquet

          # Try parquet first (no column limit), fall back to SQLite, then v3.0
          if os.path.exists(parquet_path):
              df = pd.read_parquet(parquet_path)
              # Downcast float64 → float32 (LightGBM uses float32 histograms, saves 2x RAM)
              _f64 = df.select_dtypes(include=['float64']).columns
              if len(_f64) > 0:
                  df[_f64] = df[_f64].astype(np.float32)
                  log(f"  Downcast {len(_f64)} float64 cols → float32 (saves ~{len(_f64) * len(df) * 4 / 1e6:.0f} MB)")
              log(f"  Loaded from parquet: {parquet_path}")
          elif os.path.exists(db_path):
              conn = sqlite3.connect(db_path)
              # Check if ext table exists (split due to SQLite column limit)
              ext_check = conn.execute(
                  "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                  (cfg['table'] + '_ext',)
              ).fetchone()
              if ext_check:
                  df_main = pd.read_sql_query(f"SELECT * FROM {cfg['table']}", conn)
                  df_ext = pd.read_sql_query(f"SELECT * FROM {cfg['table']}_ext", conn)
                  df_ext = df_ext.drop(columns=['timestamp'], errors='ignore')
                  df = pd.concat([df_main, df_ext], axis=1)
                  log(f"  Loaded from SQLite (split tables): {cfg['table']} + {cfg['table']}_ext")
              else:
                  df = pd.read_sql_query(f"SELECT * FROM {cfg['table']}", conn)
              conn.close()
          else:
              log(f"  SKIP — no data found for {tf_name} (checked local + v3.0)")
              continue

          # Parse timestamp
          if 'timestamp' in df.columns:
              df['timestamp'] = pd.to_datetime(df['timestamp'])
          elif 'date' in df.columns:
              df['timestamp'] = pd.to_datetime(df['date'])

          log(f"  {elapsed()} Loaded: {len(df)} rows x {len(df.columns)} columns")

          # Triple-barrier labels: LONG(2) / FLAT(1) / SHORT(0)
          if 'triple_barrier_label' in df.columns:
              tb_labels = pd.to_numeric(df['triple_barrier_label'], errors='coerce').values
              log(f"  Using pre-computed triple_barrier_label column")
          else:
              log(f"  Computing triple-barrier labels on-the-fly for {tf_name}...")
              tb_labels = compute_triple_barrier_labels(df, tf_name)

          # tb_labels: 0=SHORT, 1=FLAT, 2=LONG, NaN=insufficient data
          # y_3class uses same encoding for LightGBM multiclass (num_class=3)
          y_3class = tb_labels.copy()
          valid_mask = ~np.isnan(y_3class)
          n_long = (y_3class == 2).sum()
          n_short = (y_3class == 0).sum()
          n_flat = (y_3class == 1).sum()
          n_nan = (~valid_mask).sum()
          tb_cfg = TRIPLE_BARRIER_CONFIG.get(tf_name, TRIPLE_BARRIER_CONFIG['1h'])
          log(f"  Triple-barrier labels (tp={tb_cfg['tp_atr_mult']}xATR, sl={tb_cfg['sl_atr_mult']}xATR, hold={tb_cfg['max_hold_bars']}): "
              f"{int(n_long)} LONG, {int(n_short)} SHORT, {int(n_flat)} FLAT, {int(n_nan)} NaN")

          # ── BINARY MODE (per-TF) ──
          from config import LEAN_1W_MODE, BINARY_TF_MODE
          _BINARY_MODE = BINARY_TF_MODE.get(tf_name, False)
          if _BINARY_MODE:
              # Convert 3-class → binary: SHORT(0)→0, FLAT(1)→NaN(drop), LONG(2)→1
              _flat_mask = (y_3class == 1)
              y_3class[_flat_mask] = np.nan  # mark FLAT as NaN → excluded from training
              # Remap: SHORT(0)→0(DOWN), LONG(2)→1(UP)
              _valid_binary = ~np.isnan(y_3class)
              y_3class[_valid_binary] = (y_3class[_valid_binary] == 2).astype(float)
              n_up = int((y_3class == 1).sum())
              n_down = int((y_3class == 0).sum())
              n_dropped = int(_flat_mask.sum())
              log(f"  BINARY MODE: {n_up} UP + {n_down} DOWN = {n_up+n_down} rows (dropped {n_dropped} FLAT)")

          # Also keep old return_col for backward compat logging
          return_col = cfg['return_col']
          if return_col not in df.columns:
              candidates = [c for c in df.columns if 'next' in c.lower() and 'return' in c.lower()]
              if candidates:
                  return_col = candidates[0]

          # MC-1 FIX: No regime weighting — model decides via HMM state feature, not pre-judged weights.
          # Counter-trend esoteric signals (full moon at bull-to-bear transition) need full weight.
          sample_weights = np.ones(len(y_3class), dtype=np.float32)
          log(f"  Sample weights: uniform (HMM state added as input feature, no regime pre-weighting)")

          # Identify feature columns
          meta_cols = {'timestamp', 'date', 'open', 'high', 'low', 'close', 'volume',
                       'quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote', 'open_time',
                       'date_norm'}
          target_like = {c for c in df.columns if 'next_' in c.lower() or 'target' in c.lower()
                         or 'direction' in c.lower() or c == 'triple_barrier_label'}
          exclude_cols = meta_cols | target_like
          feature_cols = [c for c in df.columns if c not in exclude_cols]

          # Per-TF feature filter: drop short-period noise for low-row TFs (1w)
          from config import apply_tf_feature_filter
          _pre_filter = len(feature_cols)
          _df_filtered = apply_tf_feature_filter(df[feature_cols], tf_name)
          feature_cols = list(_df_filtered.columns)
          if len(feature_cols) < _pre_filter:
              log(f"  TF feature filter: {_pre_filter} → {len(feature_cols)} features ({_pre_filter - len(feature_cols)} dropped)")

          # --- Sparse matrix support for 150K+ features ---
          # Load base features as dense (typically a few hundred to a few thousand cols)
          X_base = _df_filtered.values.astype(np.float32)
          del _df_filtered
          # Fix inf values which would break training (NaN kept for LightGBM missing branches)
          X_base = np.where(np.isinf(X_base), np.nan, X_base)
          n_base_features = len(feature_cols)

          # Check for sparse cross .npz file for this TF
          cross_matrix = None
          cross_cols = None
          npz_path = os.path.join(DB_DIR, f'v2_crosses_BTC_{tf_name}.npz')
          # v3.1: check v3.0 shared data dir for cross NPZ
          if not os.path.exists(npz_path):
              npz_v30 = os.path.join(V30_DATA_DIR, f'v2_crosses_BTC_{tf_name}.npz')
              if os.path.exists(npz_v30):
                  npz_path = npz_v30
          if os.path.exists(npz_path):
              try:
                  log(f"  {elapsed()} Loading sparse cross matrix: {npz_path}")
                  cross_matrix = sp_sparse.load_npz(npz_path).tocsr()
                  cross_matrix = _ensure_lgbm_sparse_dtypes(cross_matrix, "cross_matrix")
                  # Load column names (try both naming conventions)
                  cols_path = npz_path.replace('.npz', '_columns.json')
                  if os.path.exists(cols_path):
                      with open(cols_path) as f:
                          cross_cols = json.load(f)
                  else:
                      # Fallback: v2_cross_names_{symbol}_{tf}.json (v2_cross_generator format)
                      _npz_basename = os.path.basename(npz_path)  # e.g. v2_crosses_BTC_1d.npz
                      _parts = _npz_basename.replace('v2_crosses_', '').replace('.npz', '').rsplit('_', 1)
                      _sym, _tfn = (_parts[0], _parts[1]) if len(_parts) == 2 else ('BTC', tf_name)
                      cols_path_alt = os.path.join(DB_DIR, f'v2_cross_names_{_sym}_{_tfn}.json')
                      # v3.1: also check v3.0 for cross names
                      if not os.path.exists(cols_path_alt):
                          cols_path_alt = os.path.join(V30_DATA_DIR, f'v2_cross_names_{_sym}_{_tfn}.json')
                      if os.path.exists(cols_path_alt):
                          with open(cols_path_alt) as f:
                              cross_cols = json.load(f)
                      else:
                          cross_cols = [f'cross_{i}' for i in range(cross_matrix.shape[1])]
                  log(f"  Sparse crosses loaded: {cross_matrix.shape[0]} rows x {cross_matrix.shape[1]} cols "
                      f"({cross_matrix.nnz} non-zeros, {cross_matrix.nnz / max(1, cross_matrix.shape[0] * cross_matrix.shape[1]) * 100:.1f}% dense)")
              except Exception as e:
                  log(f"  WARNING: Failed to load sparse crosses: {e}")
                  cross_matrix = None
                  cross_cols = None

          # Combine base + crosses into X_all
          _X_all_is_sparse = False
          _nnz_exceeds_int32 = False   # set True if NNZ > int32 max (15m with large cross matrix)
          _n_total_features = 0        # set after cross loading, used for sequential CPCV decision
          _converted_to_dense = False  # tracks if sparse→dense conversion happened
          # (subsampling removed — matrix = ALL data, no data loss)
          if cross_matrix is not None and cross_matrix.shape[0] == X_base.shape[0]:
              # Convert base to sparse PRESERVING NaN (LightGBM treats NaN as missing natively)
              # Do NOT use nan_to_num — that would:
              #   1. Change semantics: LightGBM missing-value handling learns split directions for NaN
              #   2. Bloat storage: explicit 0s get stored in sparse matrix, defeating the point
              X_base_sparse = sp_sparse.csr_matrix(X_base)  # NaN stored as explicit entries, true zeros are structural
              X_all = sp_sparse.hstack([X_base_sparse, cross_matrix], format='csr')
              X_all = _ensure_lgbm_sparse_dtypes(X_all, "X_all")
              log(f"  Sparse dtypes: indices={X_all.indices.dtype}, indptr={X_all.indptr.dtype}")
              # ALWAYS keep sparse CSR — LightGBM accepts scipy sparse natively.
              # Dense conversion is unnecessary and OOMs on 4h+ (498GB for 5.3M features).
              # With force_col_wise=True, sparse training is multi-threaded.
              _n_total_features = X_all.shape[1]
              # OPT-9: Convert to optimal sparse format for LightGBM
              # CSC for force_col_wise, CSR for force_row_wise — skip wasteful conversion
              from config import TF_FORCE_ROW_WISE
              if tf_name in TF_FORCE_ROW_WISE:
                  X_all = X_all.tocsr() if not isinstance(X_all, sp_sparse.csr_matrix) else X_all
                  log(f"  Keeping SPARSE CSR ({_n_total_features:,} features, force_row_wise for {tf_name})")
              else:
                  X_all = X_all.tocsc()
                  log(f"  Keeping SPARSE CSC ({_n_total_features:,} features, force_col_wise)")
              feature_cols = feature_cols + cross_cols
              _X_all_is_sparse = True  # ALWAYS sparse — no dense conversion
              nnz = X_all.nnz if hasattr(X_all, 'nnz') else int((X_all != 0).sum())
              if hasattr(X_all, 'nnz') and X_all.nnz > INT32_MAX:
                  log(f"  NNZ={X_all.nnz:,} exceeds int32 max ({INT32_MAX:,})")
                  log(f"  indptr is int64 (LightGBM PR #1719) — training proceeds normally")
                  log(f"  All {X_all.shape[0]} rows x {X_all.shape[1]} features preserved")
                  _nnz_exceeds_int32 = True
              else:
                  _nnz_exceeds_int32 = False
              total = X_all.shape[0] * X_all.shape[1]
              density = nnz / total * 100 if total > 0 else 0
              log(f"  Combined sparse matrix: {X_all.shape[0]} rows x {X_all.shape[1]} cols "
                  f"({n_base_features} base + {len(cross_cols)} crosses) "
                  f"density={density:.4f}% nnz={nnz:,}")
              del X_base_sparse, cross_matrix
          elif cross_matrix is not None:
              log(f"  WARNING: Cross matrix row count ({cross_matrix.shape[0]}) != base ({X_base.shape[0]}), skipping crosses")
              X_all = X_base
              del cross_matrix
          else:
              X_all = X_base
          del X_base

          # Dense→sparse conversion EARLY — ensures consistent code paths through
          # runtime_checks, Optuna, and CPCV. Small TFs (1w) without crosses are dense.
          if not _X_all_is_sparse:
              log(f"  Dense matrix ({X_all.shape}) — converting to sparse CSR early")
              X_all = sp_sparse.csr_matrix(X_all)
              _X_all_is_sparse = True

          timestamps = df['timestamp'].values if 'timestamp' in df.columns else np.arange(len(df))
          closes = pd.to_numeric(df['close'], errors='coerce').values

          log(f"  Features: {len(feature_cols)} ({'SPARSE' if _X_all_is_sparse else 'DENSE'})")

          # Event-aware weights: upweight bars where esoteric signals are active
          ESOTERIC_KEYWORDS = [
              'gem_', 'dr_', 'moon', 'nakshatra', 'vedic', 'bazi', 'tzolkin', 'arabic',
              'tweet', 'sport', 'horse', 'caution', 'cross_', 'eclipse', 'retro', 'shmita',
              'gold_tweet', 'red_tweet', 'misdirection', 'planetary_', 'lot_', 'hebrew',
              'fibonacci', 'gann', 'tesla_369', 'master_', 'contains_113', 'contains_322',
              'contains_93', 'contains_213', 'contains_666', 'contains_777', 'friday_13',
              'palindrome', 'fg_x_', 'onchain_', 'macro_', 'headline_', 'news_sentiment',
              'sentiment_mean', 'caps_', 'excl_', 'upset', 'overtime',
              'vortex_', 'sephirah', 'schumann_', 'chakra_', 'jupiter_', 'mercury_',
              'saros_', 'metonic_', 'news_astro_', 'game_astro_',
              'diwali_', 'ramadan_', 'chinese_new_year', 'angel',
          ]
          esoteric_col_mask = np.array([any(kw in col for kw in ESOTERIC_KEYWORDS) for col in feature_cols])
          if esoteric_col_mask.sum() > 0:
              # For esoteric weight computation, use only base feature columns (first n_base_features)
              # to avoid densifying the huge cross matrix
              esoteric_base_mask = esoteric_col_mask[:n_base_features]
              if esoteric_base_mask.sum() > 0:
                  if _X_all_is_sparse and hasattr(X_all, 'toarray'):
                      esoteric_indices = np.where(esoteric_base_mask)[0]
                      X_esoteric = X_all[:, esoteric_indices].toarray()
                  else:
                      X_esoteric = X_all[:, np.where(esoteric_base_mask)[0] if esoteric_base_mask.dtype == bool else esoteric_base_mask]
                  esoteric_active = np.sum(~np.isnan(X_esoteric) & (X_esoteric != 0), axis=1)
                  esoteric_weight = np.clip(1.0 + 0.5 * np.minimum(esoteric_active, 4), 1.0, 3.0)
                  sample_weights *= esoteric_weight.astype(np.float32)
                  bars_with_esoteric = (esoteric_active > 0).sum()
                  n_rows = X_all.shape[0]
                  log(f"  Esoteric weights: {esoteric_base_mask.sum()} esoteric columns, "
                      f"{bars_with_esoteric}/{n_rows} bars have active signals, "
                      f"weight range [{esoteric_weight.min():.1f}, {esoteric_weight.max():.1f}]")

          # Count NaN density to verify esoteric features are sparse (not all zeros)
          if hasattr(X_all, 'nnz'):
              log(f"  Sparse matrix: {X_all.nnz} non-zeros of {X_all.shape[0] * X_all.shape[1]} total "
                  f"({X_all.nnz / max(1, X_all.shape[0] * X_all.shape[1]) * 100:.2f}% non-zero)")
          else:
              nan_counts = np.isnan(X_all).sum(axis=0)
              nan_pct = nan_counts / X_all.shape[0] * 100
              sparse_features = (nan_pct > 10).sum()
              log(f"  Sparse features (>10% NaN): {sparse_features} of {len(feature_cols)} — these are esoteric signals")

          # ============================================================
          # COMBINATORIAL PURGED CROSS-VALIDATION (CPCV)
          # ============================================================
          n = X_all.shape[0]  # Use subsampled row count, not original df length
          tb_cfg = TRIPLE_BARRIER_CONFIG.get(tf_name, TRIPLE_BARRIER_CONFIG['1h'])
          max_hold = tb_cfg.get('max_hold_bars', 24)

          # Build event start/end arrays for purging
          # t0 = sample index, t1 = sample index + max_hold_bars
          _all_indices = np.arange(n)
          _t0_arr = _all_indices.copy()
          _t1_arr = np.minimum(_all_indices + max_hold, n - 1)

          # Sample uniqueness weighting (Lopez de Prado)
          uniqueness = _compute_sample_uniqueness(_t0_arr, _t1_arr, n)
          # Combine with existing regime/esoteric weights
          sample_weights *= uniqueness.astype(np.float32)
          # Normalize so sum = n_events
          _sw_sum = sample_weights[~np.isnan(y_3class)].sum()
          if _sw_sum > 0:
              sample_weights *= (~np.isnan(y_3class)).sum() / _sw_sum
          log(f"  Sample uniqueness: min={uniqueness.min():.3f} max={uniqueness.max():.3f} mean={uniqueness.mean():.3f}")

          # Generate CPCV splits
          from config import CPCV_SAMPLE_SEED
          _cpcv_sample = None  # None = exhaustive (all paths)
          if _args.n_groups is not None:
              n_groups = _args.n_groups
              n_test_groups = 1
              log(f"  CPCV override: n_groups={n_groups}, n_test=1 (--n-groups flag)")
          elif _args.search_mode:
              # Optuna search mode: fewer CPCV groups for faster evaluation
              from config import OPTUNA_PHASE1_CPCV_GROUPS
              n_groups = OPTUNA_PHASE1_CPCV_GROUPS
              n_test_groups = 1
              log(f"  CPCV search mode: n_groups={n_groups}, n_test=1 (--search-mode flag)")
          else:
              from config import TF_CPCV_GROUPS, CPCV_SAMPLE_PATHS, CPCV_SAMPLE_SEED
              n_groups, n_test_groups = TF_CPCV_GROUPS.get(tf_name, (10, 2))
              _cpcv_sample = CPCV_SAMPLE_PATHS  # sample paths for training (None = exhaustive)

          _embargo_pct = max(0.01, max_hold / n)  # embargo >= max_hold_bars bars (López de Prado)
          _cpcv_seed = CPCV_SAMPLE_SEED if _cpcv_sample else 42
          cpcv_splits = _generate_cpcv_splits(
              n, n_groups=n_groups, n_test_groups=n_test_groups,
              max_hold_bars=max_hold, embargo_pct=_embargo_pct,
              sample_paths=_cpcv_sample, seed=_cpcv_seed,
          )

          # Convert CPCV splits to (train_start, train_end, test_start, test_end) format
          # for compatibility with existing training loop
          splits = []
          for train_idx, test_idx in cpcv_splits:
              splits.append((train_idx, test_idx))

          _total_cpcv = len(list(combinations(range(n_groups), n_test_groups)))
          _sample_tag = f", sampled {len(splits)}/{_total_cpcv}" if _cpcv_sample else ""
          log(f"  {elapsed()} CPCV: {len(splits)} paths (N={n_groups} groups, K={n_test_groups} test, purge={max_hold} bars, embargo={max(1, int(n * _embargo_pct))} bars{_sample_tag})")
          for i, (tr_idx, te_idx) in enumerate(splits[:5]):  # log first 5
              log(f"    Path {i+1}: train={len(tr_idx)} samples, test={len(te_idx)} samples")
          if len(splits) > 5:
              log(f"    ... and {len(splits) - 5} more paths")

          # Storage for OOS predictions across all paths (for meta-labeling + PBO)
          oos_predictions = []  # list of dicts with {indices, y_true, y_pred_probs}

          window_results = []
          best_model_obj = None
          best_acc = 0

          # ── CPCV fold checkpoint: resume from last completed fold on crash ──
          _cpcv_ckpt_path = os.path.join(DB_DIR, f'cpcv_checkpoint_{tf_name}.pkl')
          _completed_folds = set()
          if os.path.exists(_cpcv_ckpt_path):
              try:
                  with open(_cpcv_ckpt_path, 'rb') as _ckf:
                      _ckpt = pickle.load(_ckf)
                  oos_predictions = _ckpt.get('oos_predictions', [])
                  window_results = _ckpt.get('window_results', [])
                  _completed_folds = set(_ckpt.get('completed_folds', []))
                  best_acc = _ckpt.get('best_acc', 0)
                  log(f"  CHECKPOINT LOADED: {len(_completed_folds)}/{len(splits)} folds done, resuming")
              except Exception as _cke:
                  log(f"  WARNING: checkpoint corrupt ({_cke}), starting fresh")
                  _completed_folds = set()

          # Build LightGBM params once (shared by all paths)
          _base_lgb_params = V2_LGBM_PARAMS.copy()
          _base_lgb_params['min_data_in_leaf'] = _MIN_DATA_IN_LEAF.get(tf_name, 3)
          # Apply per-TF num_leaves cap (fix 2.8)
          from config import TF_NUM_LEAVES
          _base_lgb_params['num_leaves'] = TF_NUM_LEAVES.get(tf_name, 63)
          # Apply per-TF learning_rate override (1w: 0.1, others: global 0.03)
          if tf_name in _CFG_TF_LR:
              _base_lgb_params['learning_rate'] = _CFG_TF_LR[tf_name]
              log(f"  learning_rate override: {_CFG_TF_LR[tf_name]} (per-TF config)")
          # Apply class weighting for imbalanced TFs (fix 2.3)
          from config import TF_CLASS_WEIGHT
          _cw = TF_CLASS_WEIGHT.get(tf_name)
          if isinstance(_cw, dict):
              # Explicit per-class weights — apply as sample_weight multiplier
              _class_weight_map = _cw
              # Build full-length class weight array (one entry per row, NaN rows get 1.0).
              # MUST be position-aligned with sample_weights — np.pad was WRONG because non-NaN
              # rows are scattered, not contiguous at the start. Per-fold slicing
              # sample_weights[train_idx][train_valid] then picks the correct weights.
              _cw_arr = np.ones(len(y_3class), dtype=np.float32)
              _cw_mask = ~np.isnan(y_3class)
              _cw_arr[_cw_mask] = np.array([_class_weight_map.get(int(k), 1.0) for k in y_3class[_cw_mask]], dtype=np.float32)
              sample_weights = sample_weights * _cw_arr
              log(f"  Class weights: {_class_weight_map} (SHORT={_class_weight_map.get(0, 1.0)}x)")
          elif _cw == 'balanced':
              _base_lgb_params['is_unbalance'] = True
              log(f"  is_unbalance=True (balanced)")
          # Wave 3: force_row_wise for 15m (294K rows / 23K EFB bundles = 12.8 ratio → row-wise faster)
          from config import TF_FORCE_ROW_WISE
          if tf_name in TF_FORCE_ROW_WISE:
              _base_lgb_params['force_row_wise'] = True
              _base_lgb_params.pop('force_col_wise', None)
              log(f"  force_row_wise=True for {tf_name} (high rows/bundles ratio)")
          else:
              _base_lgb_params['force_col_wise'] = True
          # ── BINARY MODE: override objective ──
          if _BINARY_MODE:
              _base_lgb_params['objective'] = 'binary'
              _base_lgb_params.pop('num_class', None)
              _base_lgb_params['learning_rate'] = 0.3  # higher for binary
              _base_lgb_params['metric'] = 'binary_logloss'
              log(f"  BINARY MODE ({tf_name}): objective=binary, LR=0.3")

          # Cap num_threads for small datasets (LightGBM docs: >64 threads on <10K rows = poor scaling)
          _n_rows = X_all.shape[0] if not hasattr(X_all, 'nnz') else X_all.shape[0]
          if _n_rows < 10_000 and _base_lgb_params.get('num_threads', 0) == 0:
              _capped_threads = min(_total_cores, 32)
              _base_lgb_params['num_threads'] = _capped_threads
              log(f"  num_threads capped to {_capped_threads} (< 10K rows, LightGBM docs)")

          # ── Optuna best params overlay ──
          # If run_optuna_local.py was run first (Step 5 in cloud pipeline), it saves
          # optuna_configs_{tf}.json with best_params. Load and overlay onto _base_lgb_params.
          # This lets Step 4 (training) use Optuna-found params instead of config.py defaults.
          _optuna_config_path = os.path.join(V30_DATA_DIR, f'optuna_configs_{tf_name}.json')
          if not os.path.exists(_optuna_config_path):
              # Also check CWD (cloud deploys may have it in /workspace)
              _optuna_config_path_cwd = f'optuna_configs_{tf_name}.json'
              if os.path.exists(_optuna_config_path_cwd):
                  _optuna_config_path = _optuna_config_path_cwd
          if os.path.exists(_optuna_config_path):
              try:
                  with open(_optuna_config_path) as _ocf:
                      _optuna_cfg = json.load(_ocf)
                  _optuna_best = _optuna_cfg.get('best_params', {})
                  # Keys that Optuna tunes (from run_optuna_local.py objective)
                  _OPTUNA_TUNABLE_KEYS = [
                      'num_leaves', 'min_data_in_leaf', 'feature_fraction',
                      'feature_fraction_bynode', 'bagging_fraction',
                      'lambda_l1', 'lambda_l2', 'min_gain_to_split',
                      'max_depth', 'learning_rate', 'extra_trees',
                  ]
                  _applied = []
                  for _ok in _OPTUNA_TUNABLE_KEYS:
                      if _ok in _optuna_best:
                          _base_lgb_params[_ok] = _optuna_best[_ok]
                          _applied.append(f"{_ok}={_optuna_best[_ok]}")
                  if _applied:
                      log(f"  OPTUNA PARAMS LOADED from {_optuna_config_path}")
                      log(f"    Applied: {', '.join(_applied)}")
                      log(f"    Optuna accuracy: {_optuna_cfg.get('final_mean_accuracy', 'N/A')}, "
                          f"Sortino: {_optuna_cfg.get('final_mean_sortino', 'N/A')}")
                  else:
                      log(f"  WARNING: optuna_configs_{tf_name}.json found but no tunable params in best_params")
              except Exception as _oe:
                  log(f"  WARNING: Failed to load Optuna params: {_oe}")
          else:
              log(f"  No optuna_configs_{tf_name}.json found — using config.py defaults")

          # Interaction constraints — DISABLED for 100K+ features (787K constrained features kills LightGBM perf)
          # LightGBM checks constraint groups per split candidate per round — catastrophic at 787K.
          # Model learns interactions via tree structure instead. Re-enable for <100K features only.
          if _n_total_features < 100_000:
              _fc_index = {f: i for i, f in enumerate(feature_cols)}
              _doy_names = [f for f in feature_cols if f.startswith('doy_')]
              if _doy_names:
                  _trend_kw = ('regime', 'ema50', 'bull', 'bear', 'hmm_', 'trend')
                  _ta_kw = ('rsi_', 'macd', 'bb_', 'atr_', 'sma_', 'ema_', 'adx_', 'stoch_', 'obv', 'vwap', 'cci_', 'mfi_', 'williams', 'ichimoku', 'keltner', 'donchian', 'supertrend', 'sar_')
                  _trend_names = [f for f in feature_cols if any(kw in f for kw in _trend_kw)]
                  _ta_names = [f for f in feature_cols if any(kw in f for kw in _ta_kw)]
                  _constrained = _doy_names + _trend_names + _ta_names
                  if _constrained:
                      _constrained_indices = [_fc_index[f] for f in _constrained if f in _fc_index]
                      _base_lgb_params['interaction_constraints'] = [_constrained_indices]
                  log(f"  Interaction constraints: {len(_constrained)} features constrained")
          else:
              log(f"  Interaction constraints: DISABLED ({_n_total_features:,} features — constraint checking too slow)")

          # Per-TF num_boost_round override (1w: 300, others: CLI default 800)
          _tf_boost_rounds = _CFG_TF_ROUNDS.get(tf_name, _args.boost_rounds)
          if tf_name in _CFG_TF_ROUNDS:
              log(f"  num_boost_round override: {_tf_boost_rounds} (per-TF config, CLI default was {_args.boost_rounds})")

          # Per-TF ES patience override (1w: 50, others: dynamic formula)
          _tf_es_patience = _CFG_TF_ES.get(tf_name)
          if _tf_es_patience:
              log(f"  ES patience override: {_tf_es_patience} (per-TF config)")

          # Default: final feature cols = feature_cols (parallel path keeps HMM in X_all)
          # Sequential sparse path overrides this after stripping HMM into overlay
          _final_feature_cols = feature_cols
          _hmm_overlay = None  # Will be set by sequential sparse path if needed
          _hmm_overlay_names = []
          _parent_ds = None  # Set by sequential path for EFB reuse; parallel path doesn't use it

          # Multi-GPU detection — dispatch to run_cpcv_gpu_parallel() for multi-GPU,
          # fall back to CPU parallel or sequential for single-GPU/CPU
          _num_gpus = _detect_gpu_count()
          _multi_gpu_mode = False
          _use_gpu_parallel_cpcv = False
          if _use_gpu_sparse() and _num_gpus > 1:
              log(f"  Multi-GPU CPCV: {_num_gpus} GPUs detected — fold-parallel via subprocess isolation")
              _use_gpu_parallel_cpcv = True
              _multi_gpu_mode = True
          elif _use_gpu_sparse():
              _use_parallel_splits = False
              _multi_gpu_mode = False
              log(f"  Single GPU — sequential CPCV")
          else:
              # CPU path
              # Tiny dataset guard: subprocess startup + module reimport (numba, lightgbm,
              # feature_library) takes 10-20s per worker. For < 2000 rows, actual training
              # is ~1s — sequential is both faster and avoids spawn/thread crashes.
              _MIN_ROWS_PARALLEL = int(os.environ.get('V3_MIN_ROWS_PARALLEL', 2000))
              if _use_parallel_splits and X_all.shape[0] < _MIN_ROWS_PARALLEL:
                  log(f"  Tiny dataset ({X_all.shape[0]} rows < {_MIN_ROWS_PARALLEL}) "
                      f"— forcing sequential CPCV (subprocess overhead >> training time)")
                  _use_parallel_splits = False

              # Dense matrices must be converted to sparse CSR for parallel workers
              if not _X_all_is_sparse and _use_parallel_splits:
                  import scipy.sparse as _sp_convert
                  log(f"  Converting dense matrix to sparse CSR for parallel CPCV ({X_all.shape})")
                  X_all = _sp_convert.csr_matrix(X_all)
                  _X_all_is_sparse = True

              # SharedMemory IPC copies raw CSR arrays (no pickle),
              # so NNZ>int32 and >1M features are NOT bottlenecks anymore.
              if _nnz_exceeds_int32 and _use_parallel_splits:
                  log(f"  NNZ exceeds int32 — SharedMemory IPC handles this (no pickle)")
              if _n_total_features > 1_000_000 and _use_parallel_splits and _X_all_is_sparse:
                  log(f"  {_n_total_features:,} SPARSE features — SharedMemory IPC (no pickle bottleneck)")

          # OPT-13: Disable GC during CPCV — LightGBM C++ does heavy lifting, Python GC is overhead
          gc.disable()

          if _use_gpu_parallel_cpcv:
              # ── GPU Fold-Parallel CPCV path (multi-GPU, subprocess isolation) ──
              # Each fold trains in its own subprocess with CUDA_VISIBLE_DEVICES set
              # before importing LightGBM. Data shared via mmap'd .npy files.

              # Pre-compute per-fold HMM overlays (same logic as CPU parallel path)
              _HMM_COL_NAMES_GPU = ['hmm_bull_prob', 'hmm_bear_prob', 'hmm_neutral_prob', 'hmm_state']
              _hmm_overlay_names_gpu = []

              if _X_all_is_sparse and 'timestamp' in df.columns:
                  _fc_idx_gpu = {f: i for i, f in enumerate(feature_cols)}
                  _hmm_existing_gpu = [_fc_idx_gpu[hc] for hc in _HMM_COL_NAMES_GPU if hc in _fc_idx_gpu]
                  if _hmm_existing_gpu:
                      _hmm_overlay_names_gpu = [feature_cols[i] for i in _hmm_existing_gpu]
                      _keep_mask_gpu = np.ones(X_all.shape[1], dtype=bool)
                      for _ci in _hmm_existing_gpu:
                          _keep_mask_gpu[_ci] = False
                      _keep_idx_gpu = np.where(_keep_mask_gpu)[0]
                      X_all = X_all[:, _keep_idx_gpu].tocsr()
                      feature_cols = [feature_cols[i] for i in _keep_idx_gpu]
                      log(f"  GPU-PARALLEL HMM: stripped {len(_hmm_existing_gpu)} full-history HMM cols")
                  else:
                      _hmm_overlay_names_gpu = list(_HMM_COL_NAMES_GPU)

              # Save mmap'd data for subprocess workers (same format as _isolated_fold_worker)
              import tempfile
              _gpu_shared_dir = tempfile.mkdtemp(prefix=f'cpcv_gpu_{tf_name}_')
              from atomic_io import save_csr_npy
              _X_csr_gpu = X_all.tocsr() if hasattr(X_all, 'tocsr') else sp_sparse.csr_matrix(X_all)
              save_csr_npy(_X_csr_gpu, _gpu_shared_dir)
              np.save(os.path.join(_gpu_shared_dir, 'y.npy'), y_3class)
              np.save(os.path.join(_gpu_shared_dir, 'weights.npy'), sample_weights)
              log(f"  GPU-PARALLEL: saved mmap data to {_gpu_shared_dir} "
                  f"(CSR={_X_csr_gpu.data.nbytes/1e9:.2f}GB, shape={_X_csr_gpu.shape})")

              # Pre-compute and save per-fold HMM overlays
              if _hmm_overlay_names_gpu and 'timestamp' in df.columns:
                  _date_norm_gpu = pd.to_datetime(timestamps).normalize()
                  if hasattr(_date_norm_gpu, 'tz') and _date_norm_gpu.tz is not None:
                      _date_norm_gpu = _date_norm_gpu.tz_localize(None)
                  _n_rows_gpu = X_all.shape[0]
                  for _wj, (train_idx_j, _) in enumerate(splits):
                      if _wj in _completed_folds:
                          continue
                      _train_end_gpu = pd.Timestamp(timestamps[train_idx_j[-1]])
                      _hmm_df_gpu = fit_hmm_on_window(_train_end_gpu)
                      _fold_ov = np.full((_n_rows_gpu, len(_hmm_overlay_names_gpu)), np.nan, dtype=np.float32)
                      if _hmm_df_gpu is not None:
                          _hmm_notz = _hmm_df_gpu.copy()
                          if _hmm_notz.index.tz is not None:
                              _hmm_notz.index = _hmm_notz.index.tz_localize(None)
                          for _hi, _hcol in enumerate(_hmm_overlay_names_gpu):
                              if _hcol in _hmm_notz.columns:
                                  _fold_ov[:, _hi] = pd.Series(_date_norm_gpu).map(
                                      _hmm_notz[_hcol].to_dict()
                                  ).ffill().values.astype(np.float32)
                      np.save(os.path.join(_gpu_shared_dir, f'hmm_overlay_fold{_wj}.npy'), _fold_ov)
                  log(f"  GPU-PARALLEL HMM: pre-computed per-fold overlays (no lookahead)")

              # FIX-2: Cap per-worker threads for GPU parallel to avoid oversubscription.
              # With num_threads=0 (auto), each of N GPU subprocesses auto-detects ALL cores,
              # spawning N × total_cores threads total. Cap to total_cores / n_gpus.
              _gpu_lgb_params = _base_lgb_params.copy()
              _gpu_threads_per_worker = max(1, _total_cores // _num_gpus)
              _gpu_lgb_params['num_threads'] = _gpu_threads_per_worker
              log(f"  GPU-PARALLEL: num_threads per GPU worker: {_gpu_threads_per_worker} "
                  f"({_num_gpus} GPUs × {_gpu_threads_per_worker} threads = "
                  f"{_num_gpus * _gpu_threads_per_worker} total, {_total_cores} cores)")

              # Run GPU-parallel CPCV
              _gpu_results = run_cpcv_gpu_parallel(
                  splits, _completed_folds, _gpu_shared_dir, _gpu_lgb_params,
                  feature_cols, _tf_boost_rounds, tf_name,
                  _hmm_overlay_names_gpu, DB_DIR, n_gpus=_num_gpus,
              )

              # Integrate results into standard format
              for result in sorted(_gpu_results, key=lambda r: r.get('wi', 0)):
                  if result.get('skip') or 'error' in result:
                      continue
                  wi = result['wi']
                  oos_predictions.append({
                      'path': wi,
                      'test_indices': result['test_idx_valid'],
                      'y_true': result['y_test'],
                      'y_pred_probs': result['preds_3c'],
                      'y_pred_labels': np.argmax(result['preds_3c'], axis=1),
                      'is_accuracy': result['is_acc'],
                      'is_mlogloss': result['is_mlogloss'],
                      'is_sharpe': result['is_sharpe'],
                  })
                  window_results.append({
                      'window': wi + 1, 'accuracy': result['acc'],
                      'prec_long': result['prec_long'], 'prec_short': result['prec_short'],
                      'mlogloss': result['mlogloss'],
                      'train_size': result['train_size'], 'test_size': result['test_size'],
                      'n_trees': result['best_iteration'],
                      'importance': result['importance'],
                  })
                  if result['acc'] > best_acc:
                      best_acc = result['acc']
                  _completed_folds.add(wi)

                  # Checkpoint after each result
                  try:
                      from atomic_io import atomic_save_pickle
                      atomic_save_pickle({
                          'oos_predictions': oos_predictions,
                          'window_results': window_results,
                          'completed_folds': list(_completed_folds),
                          'best_acc': best_acc,
                      }, _cpcv_ckpt_path)
                  except ImportError:
                      with open(_cpcv_ckpt_path, 'wb') as _ckf:
                          pickle.dump({
                              'oos_predictions': oos_predictions,
                              'window_results': window_results,
                              'completed_folds': list(_completed_folds),
                              'best_acc': best_acc,
                          }, _ckf)

              _gpu_errors = [r for r in _gpu_results if 'error' in r]
              if _gpu_errors:
                  log(f"  WARNING: {len(_gpu_errors)} GPU fold errors — check logs above")

              # Cleanup mmap files
              import shutil
              try:
                  shutil.rmtree(_gpu_shared_dir, ignore_errors=True)
              except Exception:
                  pass

              # Load best model from saved fold file
              if window_results:
                  _best_fold = max(window_results, key=lambda r: r['accuracy'])
                  _best_wi = _best_fold['window'] - 1
                  _best_model_path = os.path.join(DB_DIR, f'model_{tf_name}_fold{_best_wi}.txt')
                  if os.path.exists(_best_model_path):
                      best_model_obj = lgb.Booster(model_file=_best_model_path)

          elif _use_parallel_splits:
              # ── Parallel CPCV path (CPU workers) ──

              # T-2 FIX: Pre-compute per-fold HMM overlays BEFORE the parallel loop.
              # Old code baked the full-history HMM into X_all (lookahead bias).
              # Now: strip HMM cols from X_all, compute fold-specific HMM overlay per fold,
              # pass each worker its own (N, 4) overlay fitted only on train-end-date data.
              _HMM_COL_NAMES_PAR = ['hmm_bull_prob', 'hmm_bear_prob', 'hmm_neutral_prob', 'hmm_state']
              _hmm_overlay_names_par = []
              _fold_hmm_overlays = {}

              if _X_all_is_sparse and 'timestamp' in df.columns:
                  # Strip existing HMM cols from sparse X_all (they were fitted on full history)
                  _fc_idx_par = {f: i for i, f in enumerate(feature_cols)}
                  _hmm_existing_par = [_fc_idx_par[hc] for hc in _HMM_COL_NAMES_PAR if hc in _fc_idx_par]
                  if _hmm_existing_par:
                      _hmm_overlay_names_par = [feature_cols[i] for i in _hmm_existing_par]
                      _keep_mask_par = np.ones(X_all.shape[1], dtype=bool)
                      for _ci in _hmm_existing_par:
                          _keep_mask_par[_ci] = False
                      _keep_idx_par = np.where(_keep_mask_par)[0]
                      X_all = X_all[:, _keep_idx_par].tocsr()
                      feature_cols = [feature_cols[i] for i in _keep_idx_par]
                      log(f"  PARALLEL HMM: stripped {len(_hmm_existing_par)} full-history HMM cols from X_all")
                  else:
                      _hmm_overlay_names_par = list(_HMM_COL_NAMES_PAR)

                  # Pre-compute per-fold HMM overlays (fit on train-end-date only)
                  _date_norm_par = pd.to_datetime(timestamps).normalize()
                  if hasattr(_date_norm_par, 'tz') and _date_norm_par.tz is not None:
                      _date_norm_par = _date_norm_par.tz_localize(None)
                  _n_rows_par = X_all.shape[0]
                  for _wj, (train_idx_j, _) in enumerate(splits):
                      if _wj in _completed_folds:
                          _fold_hmm_overlays[_wj] = None
                          continue
                      _train_end_par = pd.Timestamp(timestamps[train_idx_j[-1]])
                      _hmm_df_par = fit_hmm_on_window(_train_end_par)
                      _fold_ov = np.full((_n_rows_par, len(_hmm_overlay_names_par)), np.nan, dtype=np.float32)
                      if _hmm_df_par is not None:
                          _hmm_notz = _hmm_df_par.copy()
                          if _hmm_notz.index.tz is not None:
                              _hmm_notz.index = _hmm_notz.index.tz_localize(None)
                          for _hi, _hcol in enumerate(_hmm_overlay_names_par):
                              if _hcol in _hmm_notz.columns:
                                  _fold_ov[:, _hi] = pd.Series(_date_norm_par).map(
                                      _hmm_notz[_hcol].to_dict()
                                  ).ffill().values.astype(np.float32)
                      _fold_hmm_overlays[_wj] = _fold_ov
                  log(f"  PARALLEL HMM: pre-computed {sum(v is not None for v in _fold_hmm_overlays.values())} per-fold overlays (no lookahead)")

              # Dynamic worker/thread allocation: adapt to actual split count
              # On 13900K (24 cores): 4 splits -> 4 workers x 6 threads
              #                       10 splits -> 10 workers x 2 threads
              #                       15 splits -> 15 workers x 1-2 threads
              _pending_splits = len(splits) - len(_completed_folds)
              # RAM-aware worker cap: each worker copies CSR data (~data_size per worker)
              try:
                  from hardware_detect import get_available_ram_gb
                  _data_gb = (X_all.data.nbytes + X_all.indices.nbytes + X_all.indptr.nbytes) / 1e9 if hasattr(X_all, 'data') else X_all.nbytes / 1e9
                  _ram_workers = max(1, int(get_available_ram_gb() * 0.6 / max(0.1, _data_gb)))
              except (ImportError, Exception):
                  _ram_workers = _total_cores
              # Row-aware worker cap: LightGBM docs say >64 threads on <10K rows = poor scaling.
              # For small datasets, fewer workers with more threads each is better than many workers.
              _n_rows = X_all.shape[0]
              if _n_rows < 2000:
                  _row_workers = max(1, min(2, _pending_splits))  # 1w/1d: 1-2 workers max
              elif _n_rows < 10000:
                  _row_workers = max(1, min(4, _pending_splits))  # 4h: up to 4 workers
              else:
                  _row_workers = _pending_splits  # 1h/15m: full parallelism
              _n_workers = int(os.environ.get('V3_CPCV_WORKERS', min(_pending_splits, _total_cores, _ram_workers, _row_workers)))
              _threads_per_worker = max(1, min(32, _total_cores // _n_workers))
              # Thread cap for small datasets: LightGBM says don't use >64 threads for <10K rows.
              # For CPCV train sets (~54% of N), actual per-fold rows = N * 0.54.
              # Scale thread cap: 1 thread per ~25 train rows, capped at _total_cores//_n_workers.
              _approx_fold_rows = int(_n_rows * 0.54)
              _thread_cap_by_rows = max(1, _approx_fold_rows // 25)
              if _threads_per_worker > _thread_cap_by_rows:
                  _threads_per_worker = _thread_cap_by_rows
                  log(f"  Thread cap: {_thread_cap_by_rows} threads/worker (scaled to ~{_approx_fold_rows} fold rows)")
              log(f"\n  PARALLEL CPCV: {_pending_splits} pending splits, {_n_workers} workers x {_threads_per_worker} threads = {_n_workers * _threads_per_worker} total ({_total_cores} cores)")

              # Set num_threads per worker to avoid oversubscription (each worker gets fair share of cores)
              _base_lgb_params = _base_lgb_params.copy()
              _base_lgb_params['num_threads'] = _threads_per_worker
              log(f"  num_threads per worker: {_base_lgb_params['num_threads']}")

              # FIX-1: Guard against dense data reaching parallel path — tocsr() on dense
              # numpy allocates a full copy (80GB+ spike on 15m). Fail loud instead.
              if not sp_sparse.issparse(X_all):
                  raise RuntimeError(
                      f"PARALLEL CPCV requires sparse X_all but got {type(X_all).__name__} "
                      f"shape={X_all.shape}. Convert to sparse BEFORE training or set "
                      f"V3_FORCE_SEQUENTIAL=1 for dense matrices."
                  )
              X_csr = X_all.tocsr() if not isinstance(X_all, sp_sparse.csr_matrix) else X_all

              # Wave 3: SharedMemory for CPCV IPC — eliminates pickle bottleneck for 15m
              # Place CSR arrays in shared memory so workers attach instead of deserializing copies
              _shm_blocks = []
              _use_shm = False
              try:
                  from multiprocessing.shared_memory import SharedMemory as _SharedMemory
                  _shm_data = _SharedMemory(create=True, size=X_csr.data.nbytes)
                  _shm_indices = _SharedMemory(create=True, size=X_csr.indices.nbytes)
                  _shm_indptr = _SharedMemory(create=True, size=X_csr.indptr.nbytes)
                  # Copy CSR arrays into shared memory
                  np.ndarray(X_csr.data.shape, dtype=X_csr.data.dtype, buffer=_shm_data.buf)[:] = X_csr.data
                  np.ndarray(X_csr.indices.shape, dtype=X_csr.indices.dtype, buffer=_shm_indices.buf)[:] = X_csr.indices
                  np.ndarray(X_csr.indptr.shape, dtype=X_csr.indptr.dtype, buffer=_shm_indptr.buf)[:] = X_csr.indptr
                  _shm_info = {
                      'data_name': _shm_data.name, 'data_shape': X_csr.data.shape, 'data_dtype': str(X_csr.data.dtype),
                      'indices_name': _shm_indices.name, 'indices_shape': X_csr.indices.shape, 'indices_dtype': str(X_csr.indices.dtype),
                      'indptr_name': _shm_indptr.name, 'indptr_shape': X_csr.indptr.shape, 'indptr_dtype': str(X_csr.indptr.dtype),
                      'matrix_shape': X_csr.shape,
                  }
                  _shm_blocks = [_shm_data, _shm_indices, _shm_indptr]
                  _use_shm = True
                  _total_shm_mb = (X_csr.data.nbytes + X_csr.indices.nbytes + X_csr.indptr.nbytes) / 1e6
                  log(f"  SharedMemory IPC: {_total_shm_mb:.0f} MB in 3 blocks (eliminates pickle bottleneck)")
              except (ImportError, OSError) as _shm_err:
                  # FIX-4: SharedMemory is REQUIRED for large sparse matrices (no pickle fallback).
                  # For small matrices (<1M features), pickle IPC is acceptable.
                  if _n_total_features > 1_000_000 or _nnz_exceeds_int32:
                      raise RuntimeError(
                          f"SharedMemory IPC failed ({_shm_err}) and pickle cannot handle "
                          f"{_n_total_features:,} features / NNZ>int32. "
                          f"Fix SharedMemory or increase /dev/shm size."
                      )
                  log(f"  SharedMemory unavailable ({_shm_err}), falling back to pickle IPC (OK for <1M features)")
                  _use_shm = False

              worker_args = []
              for wi, (train_idx, test_idx) in enumerate(splits):
                  if wi in _completed_folds:
                      log(f"  Path {wi+1}/{len(splits)}: SKIP (checkpoint)")
                      continue
                  gpu_id = 0  # unused in LightGBM CPU mode
                  if _use_shm:
                      worker_args.append((
                          wi, train_idx, test_idx,
                          _shm_info, None, None, None,  # SharedMemory info in slot 3, rest None
                          y_3class, sample_weights, feature_cols, _base_lgb_params,
                          _tf_boost_rounds, tf_name, gpu_id,
                          _fold_hmm_overlays.get(wi),
                          _hmm_overlay_names_par,
                      ))
                  else:
                      worker_args.append((
                          wi, train_idx, test_idx,
                          X_csr.data, X_csr.indices, X_csr.indptr, X_csr.shape,
                          y_3class, sample_weights, feature_cols, _base_lgb_params,
                          _tf_boost_rounds, tf_name, gpu_id,
                          _fold_hmm_overlays.get(wi),
                          _hmm_overlay_names_par,
                      ))

              # FIX #44: Pre-compile .pyc + warm OS page cache for heavy libs BEFORE spawning.
              # With 'spawn' context each child re-imports from scratch (3-5s × N workers).
              # Pre-compiling ensures .pyc exists; importing in parent warms filesystem cache.
              import py_compile
              for _precache_mod in ('numpy', 'scipy', 'scipy.sparse', 'lightgbm',
                                    'sklearn', 'sklearn.metrics', 'threadpoolctl'):
                  try:
                      _mod = __import__(_precache_mod)
                      if hasattr(_mod, '__file__') and _mod.__file__:
                          py_compile.compile(_mod.__file__, doraise=False)
                  except ImportError:
                      pass

              # Use 'spawn' context to avoid fork+OpenMP deadlock (GCC bug: forked
              # OpenMP runtime is in undefined state, causes hangs/crashes in LightGBM).
              import multiprocessing as _mp_ctx
              _spawn_ctx = _mp_ctx.get_context('spawn')

              # Set worker env BEFORE spawning — spawn'd children inherit parent env at fork.
              # This prevents module-level threadpool_limits(128) in each worker and caps
              # OMP threads to per-worker allocation (total_cores / n_workers).
              _saved_env = {}
              for _ek in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                          'NUMEXPR_NUM_THREADS', '_CPCV_WORKER'):
                  _saved_env[_ek] = os.environ.get(_ek)
              os.environ['OMP_NUM_THREADS'] = str(_threads_per_worker)
              os.environ['MKL_NUM_THREADS'] = str(_threads_per_worker)
              os.environ['OPENBLAS_NUM_THREADS'] = str(_threads_per_worker)
              os.environ['NUMEXPR_NUM_THREADS'] = str(_threads_per_worker)
              os.environ['_CPCV_WORKER'] = '1'

              _n_failed_folds = 0
              try:
                  with ProcessPoolExecutor(
                      max_workers=_n_workers, mp_context=_spawn_ctx,
                      initializer=_cpcv_worker_initializer,
                      initargs=(str(_threads_per_worker),),
                  ) as executor:
                      for result in executor.map(_cpcv_split_worker, worker_args):
                          (wi, acc, prec_long, prec_short, mlogloss_val, best_iter,
                           model_bytes, preds_3c, y_test, test_idx_valid,
                           importance, is_acc, is_mlogloss, is_sharpe) = result

                          if acc is None:
                              log(f"  Path {wi+1}/{len(splits)}: SKIP -- not enough samples or worker error")
                              _n_failed_folds += 1
                              continue

                          oos_predictions.append({
                              'path': wi,
                              'test_indices': test_idx_valid,
                              'y_true': y_test,
                              'y_pred_probs': preds_3c,
                              'y_pred_labels': np.argmax(preds_3c, axis=1),
                              'is_accuracy': is_acc,
                              'is_mlogloss': is_mlogloss,
                              'is_sharpe': is_sharpe,
                          })
                          window_results.append({
                              'window': wi + 1, 'accuracy': acc,
                              'prec_long': prec_long, 'prec_short': prec_short,
                              'mlogloss': mlogloss_val,
                              'train_size': 0, 'test_size': len(y_test),
                              'n_trees': best_iter,
                              'importance': importance,
                          })
                          log(f"  Path {wi+1}/{len(splits)}: "
                              f"Acc={acc:.3f} PrecL={prec_long:.3f} PrecS={prec_short:.3f} mlogloss={mlogloss_val:.4f} Trees={best_iter}")

                          if acc > best_acc:
                              best_acc = acc
                              # Deserialize best model
                              import tempfile as _tmpmod
                              _tmp = _tmpmod.NamedTemporaryFile(suffix='.txt', delete=False)
                              _tmp.write(model_bytes)
                              _tmp.close()
                              best_model_obj = lgb.Booster(model_file=_tmp.name)
                              os.unlink(_tmp.name)

                          # Checkpoint after each fold (crash-safe resume)
                          _completed_folds.add(wi)
                          try:
                              from atomic_io import atomic_save_pickle
                              atomic_save_pickle({
                                  'oos_predictions': oos_predictions,
                                  'window_results': window_results,
                                  'completed_folds': list(_completed_folds),
                                  'best_acc': best_acc,
                              }, _cpcv_ckpt_path)
                          except ImportError:
                              with open(_cpcv_ckpt_path, 'wb') as _ckf:
                                  pickle.dump({
                                      'oos_predictions': oos_predictions,
                                      'window_results': window_results,
                                      'completed_folds': list(_completed_folds),
                                      'best_acc': best_acc,
                                  }, _ckf)

              except (BrokenProcessPool, Exception) as _pool_err:
                  log(f"\n  PARALLEL CPCV POOL ERROR: {type(_pool_err).__name__}: {_pool_err}")
                  log(f"  Completed {len(_completed_folds)} folds before crash — checkpoint saved")
                  # Cleanup SharedMemory before fallback
                  for _sb in _shm_blocks:
                      try:
                          _sb.close()
                          _sb.unlink()
                      except Exception:
                          pass
                  _shm_blocks = []

                  # Fallback: run remaining folds SEQUENTIALLY in main process
                  _remaining = [a for a in worker_args if a[0] not in _completed_folds]
                  if _remaining:
                      log(f"  FALLBACK: running {len(_remaining)} remaining folds sequentially in main process")
                      for _fb_args in _remaining:
                          _fb_wi = _fb_args[0]
                          # Rebuild args with raw arrays (SharedMemory is cleaned up)
                          if isinstance(_fb_args[3], dict):
                              _fb_args = (
                                  _fb_args[0], _fb_args[1], _fb_args[2],
                                  X_csr.data, X_csr.indices, X_csr.indptr, X_csr.shape,
                              ) + _fb_args[7:]
                          try:
                              result = _cpcv_split_worker(_fb_args)
                              (wi, acc, prec_long, prec_short, mlogloss_val, best_iter,
                               model_bytes, preds_3c, y_test, test_idx_valid,
                               importance, is_acc, is_mlogloss, is_sharpe) = result
                              if acc is None:
                                  log(f"  Fallback fold {_fb_wi+1}: SKIP (not enough samples)")
                                  _n_failed_folds += 1
                                  continue
                              oos_predictions.append({
                                  'path': wi, 'test_indices': test_idx_valid,
                                  'y_true': y_test, 'y_pred_probs': preds_3c,
                                  'y_pred_labels': np.argmax(preds_3c, axis=1),
                                  'is_accuracy': is_acc, 'is_mlogloss': is_mlogloss, 'is_sharpe': is_sharpe,
                              })
                              window_results.append({
                                  'window': wi + 1, 'accuracy': acc,
                                  'prec_long': prec_long, 'prec_short': prec_short,
                                  'mlogloss': mlogloss_val,
                                  'train_size': 0, 'test_size': len(y_test),
                                  'n_trees': best_iter, 'importance': importance,
                              })
                              log(f"  Fallback fold {_fb_wi+1}: Acc={acc:.3f} PrecL={prec_long:.3f} "
                                  f"PrecS={prec_short:.3f} mlogloss={mlogloss_val:.4f}")
                              if acc > best_acc:
                                  best_acc = acc
                                  import tempfile as _tmpmod
                                  _tmp = _tmpmod.NamedTemporaryFile(suffix='.txt', delete=False)
                                  _tmp.write(model_bytes)
                                  _tmp.close()
                                  best_model_obj = lgb.Booster(model_file=_tmp.name)
                                  os.unlink(_tmp.name)
                              _completed_folds.add(wi)
                              try:
                                  from atomic_io import atomic_save_pickle
                                  atomic_save_pickle({
                                      'oos_predictions': oos_predictions,
                                      'window_results': window_results,
                                      'completed_folds': list(_completed_folds),
                                      'best_acc': best_acc,
                                  }, _cpcv_ckpt_path)
                              except ImportError:
                                  with open(_cpcv_ckpt_path, 'wb') as _ckf:
                                      pickle.dump({
                                          'oos_predictions': oos_predictions,
                                          'window_results': window_results,
                                          'completed_folds': list(_completed_folds),
                                          'best_acc': best_acc,
                                      }, _ckf)
                          except Exception as _fb_err:
                              log(f"  Fallback fold {_fb_wi+1} FAILED: {type(_fb_err).__name__}: {_fb_err}")
                              _n_failed_folds += 1
                      log(f"  FALLBACK complete: {len(window_results)} folds succeeded")

                  if not window_results:
                      raise RuntimeError(
                          f"Parallel CPCV failed AND sequential fallback failed: {_pool_err}"
                      ) from _pool_err

              if _n_failed_folds > 0:
                  log(f"  WARNING: {_n_failed_folds} folds returned None (skipped or worker error)")

              # Wave 3: Cleanup SharedMemory blocks after parallel CPCV
              for _sb in _shm_blocks:
                  try:
                      _sb.close()
                      _sb.unlink()
                  except Exception:
                      pass

              # Restore parent env after parallel CPCV
              for _ek, _ev in _saved_env.items():
                  if _ev is None:
                      os.environ.pop(_ek, None)
                  else:
                      os.environ[_ek] = _ev

          else:
              # ── Sequential CPCV path (dense matrix or V3_FORCE_SEQUENTIAL=1) ──

              # ── FIX: Separate HMM columns from the big sparse matrix ──
              # Instead of tolil()/tocsr() on a 1M x 2M matrix EVERY fold (150-450s),
              # we keep HMM columns as a small dense overlay that gets hstacked
              # only on the train/test SUBSET at extraction time (milliseconds).
              _HMM_COL_NAMES = ['hmm_bull_prob', 'hmm_bear_prob', 'hmm_neutral_prob', 'hmm_state']
              _hmm_overlay = None  # (N, 4) dense float32 — updated each fold
              _hmm_overlay_names = []  # feature names for the overlay columns
              _hmm_stripped = False  # whether we removed HMM cols from X_all

              if _X_all_is_sparse:
                  # Find HMM columns already in feature_cols and strip them from X_all
                  _hmm_existing_indices = []
                  _hmm_existing_names = []
                  _fc_idx = {f: i for i, f in enumerate(feature_cols)}  # O(1) lookup
                  for hc in _HMM_COL_NAMES:
                      if hc in _fc_idx:
                          _hmm_existing_indices.append(_fc_idx[hc])
                          _hmm_existing_names.append(hc)

                  if _hmm_existing_indices:
                      # Extract current HMM values as dense overlay
                      _hmm_slice = X_all[:, _hmm_existing_indices]
                      _hmm_overlay = (_hmm_slice.toarray() if hasattr(_hmm_slice, 'toarray') else _hmm_slice).astype(np.float32)
                      _hmm_overlay_names = _hmm_existing_names

                      # Remove HMM columns from X_all (keep only non-HMM columns)
                      _keep_mask = np.ones(X_all.shape[1], dtype=bool)
                      for ci in _hmm_existing_indices:
                          _keep_mask[ci] = False
                      _keep_indices = np.where(_keep_mask)[0]
                      X_all = X_all[:, _keep_indices].tocsr()
                      feature_cols = [feature_cols[i] for i in _keep_indices]
                      _hmm_stripped = True
                      log(f"  HMM overlay: stripped {len(_hmm_existing_indices)} HMM cols from sparse matrix "
                          f"(avoids tolil/tocsr per fold)")

                      # Recompute interaction constraints after column strip (indices shifted)
                      if 'interaction_constraints' in _base_lgb_params:
                          _doy_names_s = [f for f in feature_cols if f.startswith('doy_')]
                          if _doy_names_s:
                              _trend_kw = ('regime', 'ema50', 'bull', 'bear', 'hmm_', 'trend')
                              _ta_kw = ('rsi_', 'macd', 'bb_', 'atr_', 'sma_', 'ema_', 'adx_', 'stoch_', 'obv', 'vwap', 'cci_', 'mfi_', 'williams', 'ichimoku', 'keltner', 'donchian', 'supertrend', 'sar_')
                              _trend_s = [f for f in feature_cols if any(kw in f for kw in _trend_kw)]
                              _ta_s = [f for f in feature_cols if any(kw in f for kw in _ta_kw)]
                              _constrained_s = _doy_names_s + _trend_s + _ta_s
                              _fc_idx2 = {f: i for i, f in enumerate(feature_cols)}
                              _ci_s = [_fc_idx2[f] for f in _constrained_s if f in _fc_idx2]
                              # Add HMM overlay indices (appended at end of _fold_feature_cols)
                              _n_base = len(feature_cols)
                              for oi, on in enumerate(_hmm_overlay_names):
                                  if any(kw in on for kw in _trend_kw):
                                      _ci_s.append(_n_base + oi)
                              _base_lgb_params['interaction_constraints'] = [_ci_s]
                          else:
                              del _base_lgb_params['interaction_constraints']
                  else:
                      # HMM cols don't exist yet — will be appended to overlay
                      _hmm_overlay = np.full((X_all.shape[0], len(_HMM_COL_NAMES)), np.nan, dtype=np.float32)
                      _hmm_overlay_names = list(_HMM_COL_NAMES)
                      log(f"  HMM overlay: initialized {len(_HMM_COL_NAMES)} new cols as dense overlay")

              # ── Build parent Dataset ONCE for EFB/bin reuse across folds ──
              # This avoids recomputing EFB bundles + bin thresholds per fold (~30% time savings).
              # Per-fold Datasets use reference=_parent_ds to inherit bins/EFB.
              # If Optuna already saved a binary Dataset, load it to skip EFB reconstruction entirely.
              _parent_ds = None
              try:
                  _parent_t0 = time.time()
                  bin_path = os.path.join(DB_DIR, f'lgbm_dataset_{tf_name}.bin')
                  if os.path.exists(bin_path):
                      log(f"  Loading parent Dataset from binary: {bin_path}")
                      _parent_ds = lgb.Dataset(bin_path, params={'feature_pre_filter': False, 'max_bin': _base_lgb_params.get('max_bin', 7), 'min_data_in_bin': 1})
                      _parent_ds.construct()
                      log(f"  Parent Dataset loaded from binary in {time.time() - _parent_t0:.1f}s "
                          f"(EFB reconstruction skipped). Folds reuse via reference=.")
                  else:
                      _parent_feature_cols = feature_cols + (_hmm_overlay_names if _hmm_overlay is not None else [])
                      # Build representative sample: valid rows only, with HMM overlay if present
                      _parent_valid = ~np.isnan(y_3class)
                      if _X_all_is_sparse and _hmm_overlay is not None:
                          _Xp_base = X_all[_parent_valid]
                          _Xp_hmm = sp_sparse.csr_matrix(_hmm_overlay[_parent_valid])
                          _Xp = sp_sparse.hstack([_Xp_base, _Xp_hmm], format='csr')
                          del _Xp_base, _Xp_hmm
                      else:
                          _Xp = X_all[_parent_valid]

                      # Parallel construction for large feature sets (10x+ faster)
                      if hasattr(_Xp, 'shape') and _Xp.shape[1] > 100_000:
                          try:
                              from run_optuna_local import _parallel_dataset_construct
                              log(f"  Parallel Dataset construction: {_Xp.shape[1]:,} features...")
                              _parent_ds = _parallel_dataset_construct(
                                  _Xp, y_3class[_parent_valid].astype(int),
                                  sample_weights[_parent_valid],
                              )
                              log(f"  Parallel build done in {time.time() - _parent_t0:.1f}s. "
                                  f"Folds reuse via reference=.")
                          except Exception as _pe:
                              log(f"  Parallel build failed ({_pe}), falling back to single-threaded")
                              _parent_ds = None
                      if _parent_ds is None:
                          _parent_ds = lgb.Dataset(
                              _Xp, label=y_3class[_parent_valid].astype(int),
                              weight=sample_weights[_parent_valid],
                              feature_name=_parent_feature_cols,
                              free_raw_data=True,
                              params={'feature_pre_filter': False, 'max_bin': _base_lgb_params.get('max_bin', 7), 'min_data_in_bin': 1},
                          )
                          _parent_ds.construct()
                      log(f"  Parent Dataset: {_Xp.shape[1]:,} features, EFB bins computed "
                          f"({time.time() - _parent_t0:.1f}s). Folds reuse via reference=.")
                      del _Xp
                  gc.collect()
              except Exception as _pde:
                  log(f"  WARNING: Parent Dataset build failed ({_pde}), folds will construct independently")
                  _parent_ds = None

              # ── Subprocess-per-fold: save shared data for isolated workers ──
              _subprocess_folds = _args.subprocess_folds and not _use_gpu_sparse()
              _shared_fold_dir = None
              if _subprocess_folds:
                  import shutil
                  _shared_fold_dir = os.path.join(DB_DIR, f'_fold_shared_{tf_name}')
                  if os.path.exists(_shared_fold_dir):
                      shutil.rmtree(_shared_fold_dir)
                  from atomic_io import save_csr_npy
                  log(f"  SUBPROCESS FOLDS: saving shared data to {_shared_fold_dir}")
                  save_csr_npy(X_all, _shared_fold_dir)
                  np.save(os.path.join(_shared_fold_dir, 'y.npy'), y_3class)
                  np.save(os.path.join(_shared_fold_dir, 'weights.npy'), sample_weights)
                  log(f"  SUBPROCESS FOLDS: shared data saved. Each fold runs in isolated process.")

              for wi, (train_idx, test_idx) in enumerate(splits):
                  if wi in _completed_folds:
                      log(f"\n  --- CPCV Path {wi+1}/{len(splits)} --- SKIP (checkpoint)")
                      continue
                  log(f"\n  --- CPCV Path {wi+1}/{len(splits)} ---")

                  # HMM: re-fit on training data only
                  if 'timestamp' in df.columns:
                      train_end_date = pd.Timestamp(timestamps[train_idx[-1]])
                      hmm_df = fit_hmm_on_window(train_end_date)
                      if hmm_df is not None:
                          date_norm = pd.to_datetime(timestamps)
                          if date_norm.tz is not None:
                              date_norm = date_norm.tz_localize(None)
                          date_norm = date_norm.normalize()
                          hmm_df_notz = hmm_df.copy()
                          if hmm_df_notz.index.tz is not None:
                              hmm_df_notz.index = hmm_df_notz.index.tz_localize(None)

                          if _X_all_is_sparse and _hmm_overlay is not None:
                              # Fast path: update the small dense overlay (no sparse conversion)
                              for hi, hmm_col in enumerate(_hmm_overlay_names):
                                  if hmm_col in hmm_df_notz.columns:
                                      hmm_mapped = pd.Series(date_norm).map(
                                          hmm_df_notz[hmm_col].to_dict()
                                      ).ffill().values.astype(np.float32)
                                      _hmm_overlay[:, hi] = hmm_mapped
                          else:
                              # Dense matrix path: update columns in-place (cheap for dense)
                              for hmm_col in _HMM_COL_NAMES:
                                  hmm_mapped = pd.Series(date_norm).map(
                                      hmm_df_notz[hmm_col].to_dict()
                                  ).ffill().values.astype(np.float32)
                                  col_idx = {f: i for i, f in enumerate(feature_cols)}.get(hmm_col, -1)
                                  if col_idx >= 0:
                                      X_all[:, col_idx] = hmm_mapped
                                  else:
                                      X_all = np.column_stack([X_all, hmm_mapped])
                                      feature_cols.append(hmm_col)

                  # ── Subprocess isolation path: train fold in isolated process ──
                  if _subprocess_folds and _shared_fold_dir is not None:
                      import multiprocessing as _mp_iso
                      _fold_result_path = os.path.join(DB_DIR, f'_fold_result_{tf_name}_{wi}.pkl')
                      # Save per-fold HMM overlay to disk for subprocess
                      if _hmm_overlay is not None:
                          np.save(os.path.join(_shared_fold_dir, f'hmm_overlay_fold{wi}.npy'), _hmm_overlay)
                      _fold_proc = _mp_iso.Process(
                          target=_isolated_fold_worker,
                          args=(_shared_fold_dir, wi, train_idx, test_idx,
                                _base_lgb_params, feature_cols,
                                _tf_boost_rounds, tf_name,
                                _hmm_overlay_names if _hmm_overlay is not None else [],
                                _fold_result_path, DB_DIR),
                      )
                      _fold_proc.start()
                      # FIX-3: Timeout on subprocess join — hung folds block forever without this.
                      _fold_proc.join(timeout=7200)  # 2hr max per fold
                      if _fold_proc.is_alive():
                          log(f"    WARNING: Fold {wi+1} subprocess timed out after 7200s — terminating")
                          _fold_proc.terminate()
                          _fold_proc.join(timeout=30)
                          raise RuntimeError(
                              f"CPCV fold {wi+1} subprocess hung for >7200s. "
                              f"Check logs for OOM or deadlock."
                          )
                      if _fold_proc.exitcode != 0:
                          raise RuntimeError(
                              f"CPCV fold {wi+1} subprocess crashed with exit code {_fold_proc.exitcode}"
                          )
                      # Read results from subprocess
                      with open(_fold_result_path, 'rb') as _frf:
                          _fold_res = pickle.load(_frf)
                      os.unlink(_fold_result_path)

                      if _fold_res.get('skip', False):
                          log(f"    SKIP -- not enough valid samples (subprocess)")
                          continue

                      acc = _fold_res['acc']
                      prec_long = _fold_res['prec_long']
                      prec_short = _fold_res['prec_short']
                      mlogloss = _fold_res['mlogloss']
                      preds_3c = _fold_res['preds_3c']
                      y_test = _fold_res['y_test']
                      test_idx_valid = _fold_res['test_idx_valid']
                      importance = _fold_res['importance']

                      oos_predictions.append({
                          'path': wi,
                          'test_indices': test_idx_valid,
                          'y_true': y_test,
                          'y_pred_probs': preds_3c,
                          'y_pred_labels': np.argmax(preds_3c, axis=1),
                          'is_accuracy': _fold_res['is_acc'],
                          'is_mlogloss': _fold_res['is_mlogloss'],
                          'is_sharpe': _fold_res['is_sharpe'],
                      })
                      window_results.append({
                          'window': wi + 1, 'accuracy': acc,
                          'prec_long': prec_long, 'prec_short': prec_short,
                          'mlogloss': mlogloss,
                          'train_size': _fold_res['train_size'], 'test_size': _fold_res['test_size'],
                          'n_trees': _fold_res['best_iteration'],
                          'importance': importance,
                      })
                      log(f"    [SUBPROCESS] Acc={acc:.3f} PrecL={prec_long:.3f} PrecS={prec_short:.3f} mlogloss={mlogloss:.4f} Trees={_fold_res['best_iteration']}")

                      if acc > best_acc:
                          best_acc = acc
                          _fold_model_path = os.path.join(DB_DIR, f'model_{tf_name}_fold{wi}.txt')
                          best_model_obj = lgb.Booster(model_file=_fold_model_path)

                      _completed_folds.add(wi)
                      try:
                          from atomic_io import atomic_save_pickle
                          atomic_save_pickle({
                              'oos_predictions': oos_predictions,
                              'window_results': window_results,
                              'completed_folds': list(_completed_folds),
                              'best_acc': best_acc,
                          }, _cpcv_ckpt_path)
                      except ImportError:
                          with open(_cpcv_ckpt_path, 'wb') as _ckf:
                              pickle.dump({
                                  'oos_predictions': oos_predictions,
                                  'window_results': window_results,
                                  'completed_folds': list(_completed_folds),
                                  'best_acc': best_acc,
                              }, _ckf)
                      log(f"    [SUBPROCESS] Process exited — OS reclaimed all fold memory")
                      continue

                  # Extract train/test using CPCV index arrays
                  y_train_raw = y_3class[train_idx]
                  y_test_raw = y_3class[test_idx]

                  # Filter out NaN labels
                  train_valid = ~np.isnan(y_train_raw)
                  test_valid = ~np.isnan(y_test_raw)

                  # Extract train/test — sparse path hstacks HMM overlay at extraction time
                  if _X_all_is_sparse and _hmm_overlay is not None:
                      # hstack only on the SUBSET rows (much smaller than full matrix)
                      _Xtr_base = X_all[train_idx][train_valid]
                      _Xtr_hmm = sp_sparse.csr_matrix(_hmm_overlay[train_idx][train_valid])
                      X_train = sp_sparse.hstack([_Xtr_base, _Xtr_hmm], format='csr')
                      del _Xtr_base, _Xtr_hmm

                      _Xte_base = X_all[test_idx][test_valid]
                      _Xte_hmm = sp_sparse.csr_matrix(_hmm_overlay[test_idx][test_valid])
                      X_test = sp_sparse.hstack([_Xte_base, _Xte_hmm], format='csr')
                      del _Xte_base, _Xte_hmm

                      _fold_feature_cols = feature_cols + _hmm_overlay_names
                  else:
                      X_train = X_all[train_idx][train_valid]
                      X_test = X_all[test_idx][test_valid]
                      _fold_feature_cols = feature_cols

                  y_train = y_train_raw[train_valid].astype(int)
                  y_test = y_test_raw[test_valid].astype(int)
                  test_idx_valid = test_idx[test_valid]  # for OOS prediction storage

                  # NO pre-filtering: LightGBM decides via tree splits, not us.

                  # Lower threshold for sparse TFs
                  min_train = 50 if tf_name in ('1w', '1d') else 300
                  min_test = 20 if tf_name in ('1w', '1d') else 50
                  _n_train = X_train.shape[0] if hasattr(X_train, 'shape') else len(X_train)
                  _n_test = X_test.shape[0] if hasattr(X_test, 'shape') else len(X_test)
                  if _n_train < min_train or _n_test < min_test:
                      log(f"    SKIP -- not enough valid samples (train={_n_train}, test={_n_test})")
                      continue

                  n_tr_long = (y_train == 2).sum()
                  n_tr_short = (y_train == 0).sum()
                  n_tr_flat = (y_train == 1).sum()
                  log(f"    Train: {_n_train} (L={n_tr_long} F={n_tr_flat} S={n_tr_short}), Test: {_n_test}")

                  params = _base_lgb_params.copy()

                  w_train = sample_weights[train_idx][train_valid]

                  # Split training fold into 85% train + 15% validation for early stopping
                  # (never use test set for model selection — preserves OOS integrity)
                  n_tr = X_train.shape[0]
                  _val_floor_s = max(20, n_tr // 10)
                  val_size = max(int(n_tr * 0.15), _val_floor_s)
                  if val_size >= n_tr - _val_floor_s:
                      val_size = max(n_tr // 5, 20)  # fallback for tiny folds
                  X_val_es = X_train[-val_size:]
                  y_val_es = y_train[-val_size:]
                  w_val_es = w_train[-val_size:]
                  X_train_es = X_train[:-val_size]
                  y_train_es = y_train[:-val_size]
                  w_train_es = w_train[:-val_size]

                  _ds_kwargs = dict(feature_name=_fold_feature_cols, free_raw_data=False,
                                    params={'feature_pre_filter': False, 'max_bin': params.get('max_bin', 7), 'min_data_in_bin': 1})
                  if _parent_ds is not None:
                      _ds_kwargs['reference'] = _parent_ds  # reuse EFB bundles + bin thresholds
                  dtrain = lgb.Dataset(X_train_es, label=y_train_es, weight=w_train_es, **_ds_kwargs)
                  dval = lgb.Dataset(X_val_es, label=y_val_es, weight=w_val_es, **_ds_kwargs)
                  _ckpt_path = os.path.join(DB_DIR, f'lgbm_ckpt_{tf_name}_fold{wi}.txt')
                  _es_rounds = max(50, int(100 * (0.1 / params.get('learning_rate', 0.03))))
                  if tf_name in _CFG_TF_ES:
                      _es_rounds = _CFG_TF_ES[tf_name]
                  if _use_gpu_sparse() and hasattr(X_train_es, 'tocsr'):
                      # GPU sparse histogram path — keep CSR, use manual update loop
                      _X_csr_fold = X_train_es.tocsr() if not isinstance(X_train_es, sp_sparse.csr_matrix) else X_train_es
                      _gpu_id = wi % _num_gpus if _multi_gpu_mode else 0
                      model = _train_gpu(
                          params, dtrain, dval, _X_csr_fold,
                          num_boost_round=_tf_boost_rounds,
                          early_stopping_rounds=_es_rounds,
                          checkpoint_cb=CheckpointCallback(_ckpt_path, period=100),
                          log_period=100,
                          gpu_device_id=_gpu_id,
                      )
                      del _X_csr_fold
                  else:
                      if _GPU_SPARSE_AVAILABLE and not hasattr(X_train_es, 'tocsr'):
                          log(f"    WARNING: GPU fork available but data is dense — using CPU training")
                      model = lgb.train(
                          params, dtrain,
                          num_boost_round=_tf_boost_rounds,
                          valid_sets=[dtrain, dval],
                          valid_names=['train', 'val'],
                          callbacks=[
                              lgb.early_stopping(_es_rounds),
                              lgb.log_evaluation(100),
                              CheckpointCallback(_ckpt_path, period=100),
                          ],
                      )

                  # Predict OOS — pass best_iteration so GPU path uses only the best trees
                  preds_3c = _fix_binary_preds(model.predict(X_test, num_iteration=model.best_iteration))
                  _n_classes = preds_3c.shape[1]
                  _labels = list(range(_n_classes))
                  pred_labels = np.argmax(preds_3c, axis=1)
                  acc = accuracy_score(y_test, pred_labels)
                  prec_long = precision_score(y_test, pred_labels, labels=[_n_classes-1], average='macro', zero_division=0)
                  prec_short = precision_score(y_test, pred_labels, labels=[0], average='macro', zero_division=0)
                  mlogloss = log_loss(y_test, preds_3c, labels=_labels)

                  # Evaluate IS (full training data) for proper PBO
                  is_preds_3c = _fix_binary_preds(_predict_chunked(model, X_train, num_iteration=model.best_iteration))
                  is_pred_labels = np.argmax(is_preds_3c, axis=1)
                  is_acc = float(accuracy_score(y_train, is_pred_labels))
                  is_mlogloss = float(log_loss(y_train, is_preds_3c, labels=_labels))
                  # IS Sharpe from simulated returns: +1 correct, -1 wrong
                  _is_sim_ret = np.where(is_pred_labels == y_train, 1.0, -1.0)
                  _is_std = np.std(_is_sim_ret, ddof=1)
                  is_sharpe = float(np.mean(_is_sim_ret) / max(_is_std, 1e-10) * np.sqrt(252))

                  # Store OOS predictions + IS metrics for meta-labeling and PBO
                  oos_predictions.append({
                      'path': wi,
                      'test_indices': test_idx_valid,
                      'y_true': y_test,
                      'y_pred_probs': preds_3c,
                      'y_pred_labels': pred_labels,
                      'is_accuracy': is_acc,
                      'is_mlogloss': is_mlogloss,
                      'is_sharpe': is_sharpe,
                  })

                  # Feature importance for this fold (gain-based)
                  importance = dict(zip(model.feature_name(), model.feature_importance(importance_type='gain')))

                  # Save fold model for feature importance pipeline
                  _fold_model_path = os.path.join(DB_DIR, f'model_{tf_name}_fold{wi}.txt')
                  model.save_model(_fold_model_path)

                  window_results.append({
                      'window': wi + 1, 'accuracy': acc,
                      'prec_long': prec_long, 'prec_short': prec_short,
                      'mlogloss': mlogloss,
                      'train_size': X_train.shape[0], 'test_size': X_test.shape[0],
                      'n_trees': model.best_iteration,
                      'importance': importance,
                  })
                  log(f"    Acc={acc:.3f} PrecL={prec_long:.3f} PrecS={prec_short:.3f} mlogloss={mlogloss:.4f} Trees={model.best_iteration}")

                  if acc > best_acc:
                      best_acc = acc
                      best_model_obj = model

                  # Checkpoint after each fold (crash-safe resume)
                  _completed_folds.add(wi)
                  try:
                      from atomic_io import atomic_save_pickle
                      atomic_save_pickle({
                          'oos_predictions': oos_predictions,
                          'window_results': window_results,
                          'completed_folds': list(_completed_folds),
                          'best_acc': best_acc,
                      }, _cpcv_ckpt_path)
                  except ImportError:
                      with open(_cpcv_ckpt_path, 'wb') as _ckf:
                          pickle.dump({
                              'oos_predictions': oos_predictions,
                              'window_results': window_results,
                              'completed_folds': list(_completed_folds),
                              'best_acc': best_acc,
                          }, _ckf)

                  # Free GPU/CPU memory between sequential folds
                  # (best_model_obj holds its own reference if this was the best fold)
                  del model
                  gc.collect()
                  if _use_gpu_sparse():
                      gc.collect()
                      log(f"    GPU memory cleanup between folds")

          if not window_results:
              log(f"  SKIP -- no valid CPCV paths")
              continue

          avg_acc = np.mean([w['accuracy'] for w in window_results])
          avg_prec_l = np.mean([w['prec_long'] for w in window_results])
          avg_prec_s = np.mean([w['prec_short'] for w in window_results])
          avg_mlogloss = np.mean([w['mlogloss'] for w in window_results])
          log(f"\n  {tf_name.upper()} CPCV AVG ({len(window_results)} paths): Acc={avg_acc:.3f} PrecL={avg_prec_l:.3f} PrecS={avg_prec_s:.3f} mlogloss={avg_mlogloss:.4f}")

          # Clean up subprocess shared data if used
          if _subprocess_folds and _shared_fold_dir and os.path.exists(_shared_fold_dir):
              import shutil
              shutil.rmtree(_shared_fold_dir, ignore_errors=True)
              log(f"  Subprocess shared data cleaned: {_shared_fold_dir}")

          # Clean up CPCV checkpoint — all folds done
          if os.path.exists(_cpcv_ckpt_path):
              os.remove(_cpcv_ckpt_path)
              log(f"  CPCV checkpoint cleaned: {os.path.basename(_cpcv_ckpt_path)}")

          # Save OOS predictions for meta-labeling and PBO
          oos_path = os.path.join(DB_DIR, f'cpcv_oos_predictions_{tf_name}.pkl')
          try:
              import pickle
              with open(oos_path, 'wb') as f:
                  pickle.dump(oos_predictions, f)
              log(f"  Saved OOS predictions: {oos_path} ({len(oos_predictions)} paths)")
          except Exception as e:
              log(f"  WARNING: Could not save OOS predictions: {e}")

          # ============================================================
          # FEATURE IMPORTANCE STABILITY ACROSS CPCV FOLDS
          # ============================================================
          t0_fi = time.time()
          try:
              all_importances = [w.get('importance', {}) for w in window_results if w.get('importance')]
              if len(all_importances) >= 3:
                  # Collect all feature names that appeared in any fold
                  all_feat_names = set()
                  for imp in all_importances:
                      all_feat_names.update(imp.keys())

                  # Build rank matrix: (n_folds, n_features)
                  feat_list = sorted(all_feat_names)
                  rank_matrix = np.zeros((len(all_importances), len(feat_list)))
                  for fold_i, imp in enumerate(all_importances):
                      gains = np.array([imp.get(f, 0) for f in feat_list])
                      rank_matrix[fold_i] = np.argsort(np.argsort(-gains)) + 1  # 3x faster than rankdata

                  # Stability: mean rank + rank CV across folds
                  mean_ranks = rank_matrix.mean(axis=0)
                  std_ranks = rank_matrix.std(axis=0)
                  cv_ranks = std_ranks / np.maximum(mean_ranks, 1)

                  # Top features by mean rank (consistently important)
                  top_k = min(50, len(feat_list))
                  top_idx = np.argsort(mean_ranks)[:top_k]
                  stable_features = []
                  for idx in top_idx:
                      fname = feat_list[idx]
                      stable_features.append({
                          'feature': fname,
                          'mean_rank': float(mean_ranks[idx]),
                          'rank_cv': float(cv_ranks[idx]),
                          'appears_in_folds': int((rank_matrix[:, idx] < len(feat_list) * 0.5).sum()),
                      })

                  # dx_ audit: flag any DOY cross in top 50 with high rank CV
                  dx_in_top = [f for f in stable_features if f['feature'].startswith('dx_')]
                  if dx_in_top:
                      log(f"  Feature stability: {len(dx_in_top)} dx_ crosses in top {top_k}")
                      for dxf in dx_in_top[:5]:
                          log(f"    {dxf['feature']}: mean_rank={dxf['mean_rank']:.0f} cv={dxf['rank_cv']:.2f}")

                  # Save stability report
                  stability_report = {
                      'n_folds': len(all_importances),
                      'n_features_used': len(feat_list),
                      'top_stable': stable_features[:top_k],
                      'dx_in_top': dx_in_top,
                      'n_unstable': int((cv_ranks > 0.5).sum()),
                  }
                  stab_path = os.path.join(DB_DIR, f'feature_importance_stability_{tf_name}.json')
                  with open(stab_path, 'w') as f:
                      json.dump(stability_report, f, indent=2)
                  log(f"  {elapsed()} Feature stability: {len(feat_list)} features analyzed, "
                      f"{stability_report['n_unstable']} unstable (CV>0.5), saved to {stab_path}")
              else:
                  log(f"  Feature stability: skipped (need >= 3 folds, got {len(all_importances)})")
          except Exception as e:
              log(f"  Feature stability failed: {e}")
          log(f"  Feature importance stability: {time.time()-t0_fi:.1f}s")

          # ============================================================
          # ADVANCED FEATURE IMPORTANCE PIPELINE (6M-scale)
          # ============================================================
          try:
              from feature_importance_pipeline import FeatureImportancePipeline
              import lightgbm as _lgb_loader

              # Load saved fold models
              _fold_boosters = []
              for _wi in range(len(window_results)):
                  _fm_path = os.path.join(DB_DIR, f'model_{tf_name}_fold{_wi}.txt')
                  if os.path.exists(_fm_path):
                      _fold_boosters.append(_lgb_loader.Booster(model_file=_fm_path))

              if len(_fold_boosters) >= 3:
                  log(f"  {elapsed()} Running advanced feature importance pipeline ({len(_fold_boosters)} folds)...")
                  _fi_pipeline = FeatureImportancePipeline(
                      fold_boosters=_fold_boosters,
                      feature_names=list(feature_cols) + (list(_hmm_overlay_names) if _hmm_overlay is not None else []),
                      tf_name=tf_name,
                      output_dir=DB_DIR,
                  )
                  _fi_results = _fi_pipeline.run(
                      skip_permutation=True,   # No X_val available here
                      skip_shap=True,          # No X_val available here
                      skip_injection=True,     # No X_val available here
                      skip_viz=True,           # matplotlib may not be on cloud
                  )
                  log(f"  {elapsed()} Advanced feature importance pipeline complete")
              else:
                  log(f"  Advanced FI pipeline: skipped (need >= 3 fold models, got {len(_fold_boosters)})")
          except ImportError:
              log(f"  Advanced FI pipeline: skipped (feature_importance_pipeline.py not found)")
          except Exception as _fi_err:
              log(f"  Advanced FI pipeline failed: {_fi_err}")

          # OPT-13: Re-enable GC after CPCV (H-5: always re-enable, even on exception above)
          # NOTE: gc.disable() at line ~1657 should be in try/finally, but re-indenting
          # ~900 lines is too risky. This gc.enable() is the safety net.
          gc.enable()
          gc.collect()

          # ============================================================
          # FINAL MODEL — ALL FEATURES, NO PRUNING
          # ============================================================
          _n_final_feats = len(feature_cols) + (len(_hmm_overlay_names) if _hmm_overlay is not None else 0)
          log(f"\n  {elapsed()} Training final model on ALL {_n_final_feats} features (no pruning)...")

          # Final model on ALL rows (standard practice after CPCV — scikit-learn GridSearchCV does this)
          # Carve 15% from END for early stopping only (not real validation)
          all_mask = ~np.isnan(y_3class)

          # If HMM overlay was separated (sequential sparse path), hstack it back
          _final_feature_cols = feature_cols
          if _hmm_overlay is not None and _X_all_is_sparse:
              _Xall_base = X_all[all_mask]
              _Xall_hmm = sp_sparse.csr_matrix(_hmm_overlay[all_mask])
              X_final_all = sp_sparse.hstack([_Xall_base, _Xall_hmm], format='csr')
              del _Xall_base, _Xall_hmm
              _final_feature_cols = feature_cols + _hmm_overlay_names
          else:
              X_final_all = X_all[all_mask]

          y_final_all = y_3class[all_mask].astype(int)
          w_final_all = sample_weights[all_mask]

          final_params = V2_LGBM_PARAMS.copy()
          final_params['min_data_in_leaf'] = _MIN_DATA_IN_LEAF.get(tf_name, 3)
          final_params['num_threads'] = _total_cores  # explicit core count (cgroup-aware)
          if tf_name in TF_FORCE_ROW_WISE:
              final_params['force_row_wise'] = True
          else:
              final_params['force_col_wise'] = True  # sparse multi-threaded
          # Copy interaction_constraints from CPCV params (fix 2.5)
          if 'interaction_constraints' in _base_lgb_params:
              final_params['interaction_constraints'] = _base_lgb_params['interaction_constraints']
          # Apply Optuna-found params to final retrain (same overlay as CPCV phase)
          _OPTUNA_TUNABLE_KEYS_FINAL = [
              'num_leaves', 'min_data_in_leaf', 'feature_fraction',
              'feature_fraction_bynode', 'bagging_fraction',
              'lambda_l1', 'lambda_l2', 'min_gain_to_split',
              'max_depth', 'learning_rate', 'extra_trees',
          ]
          for _ok in _OPTUNA_TUNABLE_KEYS_FINAL:
              if _ok in _base_lgb_params and _ok not in ('num_threads',):
                  final_params[_ok] = _base_lgb_params[_ok]

          # Split into 85% train + 15% val for early stopping
          n_final = X_final_all.shape[0]
          val_sz = max(int(n_final * 0.15), 100)
          if val_sz >= n_final:
              val_sz = max(n_final // 5, 20)
          X_val_f = X_final_all[-val_sz:]
          y_val_f = y_final_all[-val_sz:]
          w_val_f = w_final_all[-val_sz:]
          X_tr_f = X_final_all[:-val_sz]
          y_tr_f = y_final_all[:-val_sz]
          w_tr_f = w_final_all[:-val_sz]
          _final_ds_kwargs = dict(feature_name=_final_feature_cols, free_raw_data=False,
                                  params={'feature_pre_filter': False, 'max_bin': final_params.get('max_bin', 7), 'min_data_in_bin': 1})
          if _parent_ds is not None:
              _final_ds_kwargs['reference'] = _parent_ds  # reuse EFB from parent
          dtrain = lgb.Dataset(X_tr_f, label=y_tr_f, weight=w_tr_f, **_final_ds_kwargs)
          dval = lgb.Dataset(X_val_f, label=y_val_f, weight=w_val_f, **_final_ds_kwargs)
          _final_ckpt_path = os.path.join(DB_DIR, f'lgbm_ckpt_{tf_name}_final.txt')
          _final_es_rounds = max(50, int(100 * (0.1 / final_params.get('learning_rate', 0.03))))
          if tf_name in _CFG_TF_ES:
              _final_es_rounds = _CFG_TF_ES[tf_name]
          if _use_gpu_sparse() and hasattr(X_tr_f, 'tocsr'):
              # GPU sparse histogram path for final model
              _X_csr_final = X_tr_f.tocsr() if not isinstance(X_tr_f, sp_sparse.csr_matrix) else X_tr_f
              final_model = _train_gpu(
                  final_params, dtrain, dval, _X_csr_final,
                  num_boost_round=_tf_boost_rounds,
                  early_stopping_rounds=_final_es_rounds,
                  checkpoint_cb=CheckpointCallback(_final_ckpt_path, period=100),
                  log_period=100,
              )
              del _X_csr_final
          else:
              if _GPU_SPARSE_AVAILABLE and not hasattr(X_tr_f, 'tocsr'):
                  log(f"  WARNING: GPU fork available but final data is dense — using CPU training")
              final_model = lgb.train(
                  final_params, dtrain, num_boost_round=_tf_boost_rounds,
                  valid_sets=[dtrain, dval], valid_names=['train', 'val'],
                  callbacks=[
                      lgb.early_stopping(_final_es_rounds),
                      lgb.log_evaluation(100),
                      CheckpointCallback(_final_ckpt_path, period=100),
                  ],
              )

          # Evaluate on val set (held-out 15% from end, used for early stopping)
          final_preds_3c = _fix_binary_preds(final_model.predict(X_val_f, num_iteration=final_model.best_iteration))
          _fn_classes = final_preds_3c.shape[1]
          _fn_labels = list(range(_fn_classes))
          final_labels = np.argmax(final_preds_3c, axis=1)
          final_acc = accuracy_score(y_val_f, final_labels)
          final_prec_l = precision_score(y_val_f, final_labels, labels=[_fn_classes-1], average='macro', zero_division=0)
          final_prec_s = precision_score(y_val_f, final_labels, labels=[0], average='macro', zero_division=0)
          final_mlogloss = log_loss(y_val_f, final_preds_3c, labels=_fn_labels)
          log(f"  FINAL: Acc={final_acc:.3f} PrecL={final_prec_l:.3f} PrecS={final_prec_s:.3f} "
              f"mlogloss={final_mlogloss:.4f} Trees={final_model.best_iteration} Features={len(_final_feature_cols)}")

          # Log feature importance (top 30 by gain — for visibility, not pruning)
          importance = dict(zip(final_model.feature_name(), final_model.feature_importance(importance_type='gain')))
          if importance:
              sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
              log(f"\n  TOP 30 FEATURES BY GAIN (of {len(_final_feature_cols)} total):")
              for i, (fname, gain) in enumerate(sorted_imp[:30]):
                  log(f"    {i+1:3d}. {fname:<45s} gain={gain:.1f}")
              # Count how many esoteric features made it into top 50
              esoteric_in_top50 = sum(1 for fname, _ in sorted_imp[:50]
                                      if any(k in fname for k in ['gem_', 'dr_', 'moon', 'nakshatra', 'vedic',
                                             'bazi', 'tzolkin', 'arabic', 'tweet', 'sport', 'horse',
                                             'caution', 'cross_', 'eclipse', 'retro', 'shmita']))
              log(f"  Esoteric features in top 50 by gain: {esoteric_in_top50}")

          # KNN features kept unconditionally — LightGBM decides via tree splits, not us

          # Save model + feature list (must include HMM overlay names for inference)
          # Accuracy floor — don't save a model worse than random
          ACCURACY_FLOOR = 0.40  # 3-class random = 33%, floor at 40%
          if final_acc < ACCURACY_FLOOR:
              log(f"  *** ACCURACY BELOW FLOOR: {final_acc:.3f} < {ACCURACY_FLOOR}. "
                  f"Model NOT saved. Check Optuna params or data. ***")
              log(f"  FATAL: model_{tf_name}.json will NOT exist — pipeline cannot continue.")
              sys.exit(1)
          else:
              # Atomic save: write to temp then rename (prevents corrupt model on crash)
              _model_path = f'{DB_DIR}/model_{tf_name}.json'
              if os.path.exists(_model_path):
                  import shutil
                  _backup_path = _model_path.replace('.json', '_prev.json')
                  shutil.copy2(_model_path, _backup_path)
                  log(f"  Backed up previous model to {_backup_path}")
              _model_tmp = _model_path + '.tmp'
              final_model.save_model(_model_tmp)
              os.replace(_model_tmp, _model_path)
              _feat_path = f'{DB_DIR}/features_{tf_name}_all.json'
              _feat_tmp = _feat_path + '.tmp'
              with open(_feat_tmp, 'w') as f:
                  json.dump(_final_feature_cols, f, indent=2)
              os.replace(_feat_tmp, _feat_path)
              log(f"  Model saved: {_model_path} (accuracy: {final_acc:.3f})")

          log(f"\n  {elapsed()} Platt calibration (per-class)...")
          from sklearn.linear_model import LogisticRegression

          cal_raw_3c = np.concatenate([p['y_pred_probs'] for p in oos_predictions], axis=0)
          y_cal = np.concatenate([p['y_true'] for p in oos_predictions], axis=0)

          if len(cal_raw_3c) > 50:
              # Platt scaling on the 3-class softprob outputs
              platt = LogisticRegression(max_iter=500)
              platt.fit(cal_raw_3c, y_cal)
              with open(f'{DB_DIR}/platt_{tf_name}.pkl', 'wb') as f:
                  pickle.dump(platt, f)
              log(f"  Platt calibrator saved (platt_{tf_name}.pkl) -- multinomial 3-class")
          else:
              platt = None

          # Confidence threshold validation (using held-out val set from final model)
          log(f"\n  {elapsed()} CONFIDENCE THRESHOLD VALIDATION (on val set)...")
          outer_raw_3c = final_model.predict(X_val_f)  # shape (N, 3)

          if platt is not None:
              outer_cal_3c = platt.predict_proba(outer_raw_3c)
              log(f"  Raw short_prob: [{outer_raw_3c[:, 0].min():.3f}, {outer_raw_3c[:, 0].max():.3f}] "
                  f"flat_prob: [{outer_raw_3c[:, 1].min():.3f}, {outer_raw_3c[:, 1].max():.3f}] "
                  f"long_prob: [{outer_raw_3c[:, 2].min():.3f}, {outer_raw_3c[:, 2].max():.3f}]")
          else:
              outer_cal_3c = outer_raw_3c

          # Check accuracy at various confidence thresholds (max class probability)
          outer_max_prob = outer_cal_3c.max(axis=1)
          outer_pred_class = np.argmax(outer_cal_3c, axis=1)
          for thresh in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
              high_conf = outer_max_prob > thresh
              tradeable = (outer_pred_class[high_conf] == 2) | (outer_pred_class[high_conf] == 0)
              if tradeable.sum() > 0:
                  hc_acc = accuracy_score(y_val_f[high_conf][tradeable], outer_pred_class[high_conf][tradeable])
                  n_long_pred = (outer_pred_class[high_conf][tradeable] == 2).sum()
                  n_short_pred = (outer_pred_class[high_conf][tradeable] == 0).sum()
                  log(f"    conf>{thresh:.2f}: {tradeable.sum()} trades (L={n_long_pred} S={n_short_pred}), Acc={hc_acc:.3f}")

          all_results[tf_name] = {
              'avg_accuracy': avg_acc, 'final_accuracy': final_acc,
              'n_features': len(_final_feature_cols), 'n_samples': len(df),
              'context_only': False, 'window_results': window_results,
              'knn_ab_test': None,  # removed — no pre-filtering
              'label_type': 'triple_barrier',
          }

          # Memory cleanup between TF builds (sparse matrices can be 1-10 GB)
          del df
          del X_all
          gc.collect()
          log(f"  [GC] Memory released after {tf_name}")

  except Exception as _e:
      log(f"\n  TRAINING FAILED: {_e}")
      import traceback
      traceback.print_exc()
      # Propagate failure so cloud_run_tf.py sees non-zero exit
      if not gc.isenabled():
          gc.enable()
          gc.collect()
      sys.exit(1)
  finally:
      # BUG-L7 FIX: ensure GC is always re-enabled even if training crashes
      if not gc.isenabled():
          gc.enable()
          gc.collect()

  # ============================================================
  # FINAL SUMMARY
  # ============================================================
  log(f"\n\n{'='*70}")
  log("MULTI-TIMEFRAME ML TRAINING COMPLETE")
  log(f"{'='*70}")

  for tf, res in all_results.items():
      log(f"\n  {tf.upper()}:")
      log(f"    Samples: {res['n_samples']}, Features: {res['n_features']}")
      log(f"    WF Accuracy: {res['avg_accuracy']:.3f}, Final: {res['final_accuracy']:.3f}")

  log(f"\n  Total time: {elapsed()}")

  with open(f'{DB_DIR}/ml_multi_tf_results.txt', 'w', encoding='utf-8') as f:
      f.write('\n'.join(RESULTS))

  with open(f'{DB_DIR}/ml_multi_tf_configs.json', 'w') as f:
      json.dump(all_results, f, indent=2, default=str)

  log(f"  Saved: ml_multi_tf_results.txt, ml_multi_tf_configs.json")
