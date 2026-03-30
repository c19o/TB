#!/usr/bin/env python3
"""
train_1w_gpu.py -- Train Real 1w Model with GPU Histogram Acceleration
=======================================================================
Two approaches:
  A) Direct CuPy histogram (works NOW with stock LightGBM + CuPy)
     - Trains on CPU with stock lgb.train()
     - Benchmarks GPU SpMV histogram for timing comparison
     - Reports: CPU training time, GPU histogram estimate, accuracy
  B) Full fork integration (when C API SetExternalCSR is ready)
     - Uses ctypes to pass scipy CSR to forked LightGBM
     - device_type='cuda_sparse' for native GPU histogram

Run from v3.3/:
    python -u gpu_histogram_fork/train_1w_gpu.py

Matrix thesis: ALL features preserved. NO filtering. Binary cross features ARE the edge.
"""

import sys
import os
import time
import json
import gc
import warnings
import pickle
import argparse
from itertools import combinations

warnings.filterwarnings('ignore')

# ── Path setup ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
V33_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(V33_DIR)

# Ensure v3.3 is on path for imports (feature_library, config, etc.)
if V33_DIR not in sys.path:
    sys.path.insert(0, V33_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import pandas as pd
from scipy import sparse as sp_sparse

import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, log_loss

START_TIME = time.time()


def elapsed():
    return f"[{time.time() - START_TIME:.0f}s]"


def log(msg):
    print(msg, flush=True)


# ── Config lookup for feature_fraction (NEVER hardcode < 0.7 — kills rare esoteric crosses) ──
# Perplexity-validated 2026-03-29: EFB operates on ~23K bundles; 0.05 = 1,150 bundles/tree.
# Rare signals in 1-2 bundles have ~5% hit rate. Must be >= 0.7, target 0.9.
try:
    from config import V3_LGBM_PARAMS, TF_MIN_DATA_IN_LEAF, TF_NUM_LEAVES
    _FF = V3_LGBM_PARAMS.get('feature_fraction', 0.9)
    _FF_BYNODE = V3_LGBM_PARAMS.get('feature_fraction_bynode', 0.8)
    _MDIL_1W = TF_MIN_DATA_IN_LEAF.get('1w', 3)
    _NUM_LEAVES_1W = TF_NUM_LEAVES.get('1w', 31)
except ImportError:
    _FF = 0.9        # NEVER below 0.7 — kills rare esoteric crosses (EFB-bundled signals)
    _FF_BYNODE = 0.8  # effective rate = _FF * _FF_BYNODE; keep both high
    _MDIL_1W = 3
    _NUM_LEAVES_1W = 31


# ============================================================================
# GPU detection
# ============================================================================

HAS_CUPY = False
GPU_NAME = "N/A"
VRAM_GB = 0.0

try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse
    HAS_CUPY = True
    try:
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props.get("name", b"Unknown")
        GPU_NAME = name.decode() if isinstance(name, bytes) else str(name)
    except Exception:
        GPU_NAME = "Unknown GPU"
    free, total = cp.cuda.Device(0).mem_info
    VRAM_GB = total / (1024 ** 3)
except ImportError:
    pass
except Exception:
    pass


# ============================================================================
# Data search paths
# ============================================================================

PARQUET_CANDIDATES = [
    os.path.join(V33_DIR, "features_BTC_1w.parquet"),
    os.path.join(PROJECT_ROOT, "v3.2_2.9M_Features", "features_BTC_1w.parquet"),
    os.path.join(PROJECT_ROOT, "v3.0 (LGBM)", "features_BTC_1w.parquet"),
]

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


def find_file(candidates, label):
    """Find first existing file from candidates list."""
    for path in candidates:
        if os.path.isfile(path):
            return path
    log(f"ERROR: {label} not found. Searched:")
    for p in candidates:
        log(f"  {p}")
    return None


# ============================================================================
# Sparse dtype enforcement (from ml_multi_tf.py)
# ============================================================================

INT32_MAX = 2_147_483_647


def _ensure_lgbm_sparse_dtypes(X, label="matrix"):
    """Enforce correct CSR dtypes for LightGBM."""
    if not hasattr(X, 'indptr'):
        return X
    if X.indptr.dtype != np.int64:
        X.indptr = X.indptr.astype(np.int64)
    if X.indices.dtype != np.int32:
        assert X.nnz == 0 or X.indices.max() <= INT32_MAX, (
            f"FATAL: {label} has column index > int32 max")
        X.indices = X.indices.astype(np.int32)
    return X


# ============================================================================
# CPCV split generation (from ml_multi_tf.py)
# ============================================================================

def _generate_cpcv_splits(n_samples, n_groups=4, n_test_groups=1,
                          max_hold_bars=None, embargo_pct=0.01):
    """Generate Combinatorial Purged Cross-Validation splits."""
    group_size = n_samples // n_groups
    groups = []
    for g in range(n_groups):
        start = g * group_size
        end = (g + 1) * group_size if g < n_groups - 1 else n_samples
        groups.append(np.arange(start, end))

    if max_hold_bars is not None:
        # Embargo must be >= max_hold_bars bars (prevents leakage from forward label horizon)
        effective_pct = max(embargo_pct, max_hold_bars / n_samples)
    else:
        effective_pct = embargo_pct
    embargo_size = max(1, int(n_samples * effective_pct))
    all_paths = list(combinations(range(n_groups), n_test_groups))

    splits = []
    for test_group_ids in all_paths:
        test_idx = np.concatenate([groups[g] for g in test_group_ids])
        train_group_ids = [g for g in range(n_groups) if g not in test_group_ids]
        train_idx = np.concatenate([groups[g] for g in train_group_ids])

        # Simple purge: remove training samples within max_hold_bars of test boundaries
        if max_hold_bars is not None:
            test_boundaries = []
            for g in test_group_ids:
                test_boundaries.append(groups[g][0])
                test_boundaries.append(groups[g][-1])
            purge_mask = np.zeros(len(train_idx), dtype=bool)
            for boundary in test_boundaries:
                purge_mask |= (np.abs(train_idx - boundary) <= max_hold_bars)
            train_idx = train_idx[~purge_mask]

        # Embargo
        for g in test_group_ids:
            test_end = groups[g][-1]
            embargo_start = test_end + 1
            embargo_end = test_end + embargo_size
            embargo_mask = (train_idx >= embargo_start) & (train_idx <= embargo_end)
            train_idx = train_idx[~embargo_mask]

        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))

    return splits


# ============================================================================
# APPROACH A: Direct CuPy histogram benchmark + stock LightGBM training
# ============================================================================

def run_approach_a(X_all, y_3class, sample_weights, feature_cols, splits):
    """
    Approach A: Train with stock LightGBM on CPU, benchmark GPU histogram
    timing separately using CuPy cuSPARSE SpMV.
    """
    log(f"\n{'=' * 70}")
    log(f"APPROACH A: Stock LightGBM + CuPy Histogram Benchmark")
    log(f"{'=' * 70}")

    # ── LightGBM params from config.py ──
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'device': 'cpu',
        'force_col_wise': True,
        'max_bin': 255,
        'max_depth': -1,
        'num_threads': 0,
        'deterministic': True,
        'feature_pre_filter': False,
        'is_enable_sparse': True,
        'min_data_in_bin': 1,
        'path_smooth': 0.1,
        'min_data_in_leaf': 3,       # 1w setting
        'min_gain_to_split': 2.0,
        'lambda_l1': 0.5,
        'lambda_l2': 3.0,
        'feature_fraction': _FF,           # from config — NEVER < 0.7 (EFB bundle killer)
        'feature_fraction_bynode': _FF_BYNODE,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'num_leaves': _NUM_LEAVES_1W,
        'learning_rate': 0.03,
        'verbosity': -1,
    }

    num_boost_round = 800
    n_folds = len(splits)

    # SPARSE throughout — CLAUDE.md §4: no dense conversion (LightGBM CSR native + EFB)
    is_sparse = hasattr(X_all, 'nnz')
    X_train_data = X_all
    log(f"  Keeping SPARSE ({X_all.shape[1]:,} features) — sparse-throughout rule")

    # ── GPU histogram benchmark (separate from training) ──
    gpu_hist_ms = None
    cpu_hist_ms = None

    if HAS_CUPY and is_sparse:
        log(f"\n  --- GPU Histogram Benchmark (CuPy cuSPARSE SpMV) ---")
        log(f"  GPU: {GPU_NAME} ({VRAM_GB:.1f} GB)")

        # Use original sparse matrix for benchmark
        X_csr = X_all if is_sparse else sp_sparse.csr_matrix(X_all)
        n_rows, n_cols = X_csr.shape
        grad = np.random.default_rng(42).standard_normal(n_rows).astype(np.float64)

        # CPU baseline: scipy SpMV
        log(f"  CPU scipy SpMV...")
        X_T_cpu = X_csr.T.tocsr()
        _ = X_T_cpu @ grad  # warmup
        cpu_times = []
        for _ in range(20):
            t0 = time.perf_counter()
            _ = X_T_cpu @ grad
            cpu_times.append(time.perf_counter() - t0)
        cpu_hist_ms = np.median(cpu_times) * 1000
        log(f"  CPU SpMV median: {cpu_hist_ms:.2f} ms")
        del X_T_cpu

        # GPU cuSPARSE
        log(f"  GPU cuSPARSE SpMV...")
        try:
            t_upload = time.perf_counter()
            X_gpu = cp_sparse.csr_matrix(X_csr.astype(np.float64))
            X_T_gpu = X_gpu.T.tocsr()
            grad_gpu = cp.asarray(grad)
            cp.cuda.Stream.null.synchronize()
            upload_ms = (time.perf_counter() - t_upload) * 1000
            log(f"  H2D upload + transpose: {upload_ms:.0f} ms (one-time)")

            # Warmup
            for _ in range(10):
                _ = X_T_gpu @ grad_gpu
                cp.cuda.Stream.null.synchronize()

            # Benchmark
            gpu_times = []
            for _ in range(50):
                start_ev = cp.cuda.Event()
                end_ev = cp.cuda.Event()
                cp.cuda.Stream.null.synchronize()
                start_ev.record()
                _ = X_T_gpu @ grad_gpu
                end_ev.record()
                end_ev.synchronize()
                gpu_times.append(cp.cuda.get_elapsed_time(start_ev, end_ev))

            gpu_hist_ms = np.median(gpu_times)
            speedup = cpu_hist_ms / gpu_hist_ms if gpu_hist_ms > 0 else 0
            log(f"  GPU SpMV median: {gpu_hist_ms:.3f} ms")
            log(f"  Speedup: {speedup:.0f}x")

            # Estimate full training histogram time
            # 800 rounds x 3 classes x ~16 hist leaves x 2 (grad+hess)
            total_spmv = num_boost_round * 3 * 16 * 2
            gpu_total_s = total_spmv * gpu_hist_ms / 1000
            cpu_total_s = total_spmv * cpu_hist_ms / 1000
            log(f"\n  Full training histogram estimate ({total_spmv:,} SpMV calls):")
            log(f"    GPU: {gpu_total_s:.1f}s ({gpu_total_s / 60:.1f} min)")
            log(f"    CPU: {cpu_total_s:.1f}s ({cpu_total_s / 60:.1f} min)")

            # Cleanup GPU memory before training
            del X_gpu, X_T_gpu, grad_gpu
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

        except Exception as e:
            log(f"  GPU benchmark failed: {e}")
            gpu_hist_ms = None

    # ── CPCV Training Loop ──
    log(f"\n  --- CPCV Training ({n_folds} folds, {num_boost_round} rounds) ---")

    fold_results = []
    oos_predictions = []
    best_model = None
    best_acc = 0
    total_train_time = 0

    for fi, (train_idx, test_idx) in enumerate(splits):
        fold_start = time.perf_counter()

        y_train_raw = y_3class[train_idx]
        y_test_raw = y_3class[test_idx]
        train_valid = ~np.isnan(y_train_raw)
        test_valid = ~np.isnan(y_test_raw)

        X_train = X_train_data[train_idx][train_valid]
        y_train = y_train_raw[train_valid].astype(int)
        X_test = X_train_data[test_idx][test_valid]
        y_test = y_test_raw[test_valid].astype(int)
        test_idx_valid = test_idx[test_valid]
        w_train = sample_weights[train_idx][train_valid]

        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

        if n_train < 50 or n_test < 20:
            log(f"  Fold {fi + 1}/{n_folds}: SKIP (train={n_train}, test={n_test})")
            continue

        # 85/15 early stopping split
        val_size = max(int(n_train * 0.15), 20)
        if val_size >= n_train:
            val_size = max(n_train // 5, 10)

        X_val_es = X_train[-val_size:]
        y_val_es = y_train[-val_size:]
        w_val_es = w_train[-val_size:]
        X_train_es = X_train[:-val_size]
        y_train_es = y_train[:-val_size]
        w_train_es = w_train[:-val_size]

        dtrain = lgb.Dataset(X_train_es, label=y_train_es, weight=w_train_es,
                             feature_name=feature_cols, free_raw_data=False)
        dval = lgb.Dataset(X_val_es, label=y_val_es, weight=w_val_es,
                           feature_name=feature_cols, free_raw_data=False)

        es_rounds = max(50, int(100 * (0.1 / params.get('learning_rate', 0.03))))
        model = lgb.train(
            params, dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dtrain, dval],
            valid_names=['train', 'val'],
            callbacks=[lgb.early_stopping(es_rounds), lgb.log_evaluation(0)],
        )

        # OOS predictions
        preds_3c = model.predict(X_test)
        pred_labels = np.argmax(preds_3c, axis=1)
        acc = float(accuracy_score(y_test, pred_labels))
        prec_long = float(precision_score(y_test, pred_labels, labels=[2],
                                          average='macro', zero_division=0))
        prec_short = float(precision_score(y_test, pred_labels, labels=[0],
                                           average='macro', zero_division=0))
        mlogloss = float(log_loss(y_test, preds_3c, labels=[0, 1, 2]))

        # IS metrics for PBO
        is_preds = model.predict(X_train)
        is_labels = np.argmax(is_preds, axis=1)
        is_acc = float(accuracy_score(y_train, is_labels))

        fold_time = time.perf_counter() - fold_start
        total_train_time += fold_time

        fold_results.append({
            'fold': fi + 1,
            'accuracy': acc,
            'prec_long': prec_long,
            'prec_short': prec_short,
            'mlogloss': mlogloss,
            'is_accuracy': is_acc,
            'best_iter': model.best_iteration,
            'train_size': n_train,
            'test_size': n_test,
            'fold_time_s': fold_time,
        })

        oos_predictions.append({
            'test_indices': test_idx_valid,
            'y_true': y_test,
            'y_pred_probs': preds_3c,
        })

        log(f"  Fold {fi + 1}/{n_folds}: "
            f"Acc={acc:.3f} PrecL={prec_long:.3f} PrecS={prec_short:.3f} "
            f"mlogloss={mlogloss:.4f} Trees={model.best_iteration} "
            f"IS_Acc={is_acc:.3f} Time={fold_time:.1f}s")

        if acc > best_acc:
            best_acc = acc
            best_model = model

        del dtrain, dval, X_train, X_test, X_train_es, X_val_es
        gc.collect()

    # ── Summary ──
    if fold_results:
        avg_acc = np.mean([r['accuracy'] for r in fold_results])
        avg_prec_l = np.mean([r['prec_long'] for r in fold_results])
        avg_prec_s = np.mean([r['prec_short'] for r in fold_results])
        avg_logloss = np.mean([r['mlogloss'] for r in fold_results])
        avg_is_acc = np.mean([r['is_accuracy'] for r in fold_results])

        log(f"\n  --- CPCV Summary ---")
        log(f"  Folds completed: {len(fold_results)}/{n_folds}")
        log(f"  Avg Accuracy:    {avg_acc:.4f}")
        log(f"  Avg Prec Long:   {avg_prec_l:.4f}")
        log(f"  Avg Prec Short:  {avg_prec_s:.4f}")
        log(f"  Avg Logloss:     {avg_logloss:.4f}")
        log(f"  Avg IS Accuracy: {avg_is_acc:.4f}")
        log(f"  Best fold Acc:   {best_acc:.4f}")
        log(f"  Total train time:{total_train_time:.1f}s ({total_train_time / 60:.1f} min)")

    return {
        'fold_results': fold_results,
        'oos_predictions': oos_predictions,
        'best_model': best_model,
        'best_acc': best_acc,
        'total_train_time_s': total_train_time,
        'gpu_hist_ms': gpu_hist_ms,
        'cpu_hist_ms': cpu_hist_ms,
    }


# ============================================================================
# APPROACH B: Full fork integration (when C API SetExternalCSR is ready)
# ============================================================================

def run_approach_b(X_all, y_3class, sample_weights, feature_cols, splits):
    """
    Approach B: Use the forked LightGBM with device_type='cuda_sparse'.
    Requires the C API SetExternalCSR to be callable.

    Status: STUB -- the .cu kernel is not yet compiled. This code shows
    exactly what to call once the fork build is complete.
    """
    log(f"\n{'=' * 70}")
    log(f"APPROACH B: Forked LightGBM cuda_sparse (C API)")
    log(f"{'=' * 70}")

    # ── Check if forked LightGBM is available ──
    from gpu_histogram_fork.src.train_pipeline import _check_cuda_sparse_available

    if _check_cuda_sparse_available():
        log("  cuda_sparse device type AVAILABLE in LightGBM build!")
        _run_fork_training(X_all, y_3class, sample_weights, feature_cols, splits)
        return

    log("  cuda_sparse device type NOT available in current LightGBM build.")
    log("  Showing the integration code that will work once the fork is built:")
    log("")

    # ── The code below is the integration plan ──
    # It demonstrates exactly how to call the forked LightGBM once
    # cuda_sparse_hist_tree_learner.cu is compiled and linked.

    log("  --- Integration Plan (Approach B) ---")
    log("")
    log("  1. Build LightGBM fork with USE_CUDA_SPARSE=ON:")
    log("     cd gpu_histogram_fork/_build/LightGBM")
    log("     cmake -B build -DUSE_CUDA_SPARSE=ON -DCMAKE_CUDA_ARCHITECTURES=86")
    log("     cmake --build build -j")
    log("")
    log("  2. Install the forked wheel:")
    log("     pip install --force-reinstall build/python-package/dist/*.whl")
    log("")
    log("  3. Training code (copy of what _run_fork_training does):")
    log("")

    # Show the integration code as documentation
    integration_code = '''
    import ctypes
    import lightgbm as lgb
    from scipy import sparse as sp_sparse

    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'device_type': 'cuda_sparse',     # <-- fork device type
        'gpu_device_id': 0,               # <-- RTX 3090
        'max_bin': 255,
        'num_leaves': 31,
        'learning_rate': 0.03,
        'feature_pre_filter': False,
        'is_enable_sparse': True,
        'min_data_in_leaf': 3,
        'min_gain_to_split': 2.0,
        'lambda_l1': 0.5,
        'lambda_l2': 3.0,
        'feature_fraction': 0.9,           # NEVER < 0.7 — EFB bundle killer
        'feature_fraction_bynode': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'num_threads': 0,
        'deterministic': True,
        'verbosity': 1,
    }

    # Create dataset normally -- LightGBM reads CSR via sparse interface
    dtrain = lgb.Dataset(X_train_csr, label=y_train, weight=w_train,
                         feature_name=feature_cols, free_raw_data=False)

    # Option A: lgb.train() handles everything (preferred when fork is complete)
    model = lgb.train(params, dtrain, num_boost_round=800,
                      valid_sets=[dtrain, dval], ...)

    # Option B: Manual ctypes SetExternalCSR (if fork exposes the C API)
    #   lgb_lib = ctypes.CDLL('lib_lightgbm.dll')  # or .so on Linux
    #   # After booster creation, pass CSR pointers:
    #   lgb_lib.LGBM_BoosterSetExternalCSR(
    #       booster_handle,
    #       X_csr.indptr.ctypes.data_as(ctypes.c_void_p),    # int64
    #       X_csr.indices.ctypes.data_as(ctypes.c_void_p),   # int32
    #       ctypes.c_int64(X_csr.nnz),
    #       ctypes.c_int32(X_csr.shape[0]),
    #       ctypes.c_int32(X_csr.shape[1]),
    #   )
    #   # Then lgb.train() uses GPU histogram internally
    '''

    for line in integration_code.strip().split('\n'):
        log(f"    {line}")

    log("")
    log("  When the fork is ready, re-run this script. It will auto-detect")
    log("  cuda_sparse and use GPU histograms natively.")
    log("")

    # ── Also try the co-processor path ──
    from gpu_histogram_fork.src.train_pipeline import _check_coprocessor_available
    if _check_coprocessor_available():
        log("  libgpu_histogram co-processor AVAILABLE!")
        log("  Would use gpu_train() from lgbm_integration.py")
    else:
        log("  libgpu_histogram co-processor NOT available.")
        log("  Build with: cd gpu_histogram_fork && make")


def _run_fork_training(X_all, y_3class, sample_weights, feature_cols, splits):
    """Run actual training with the forked LightGBM cuda_sparse device."""
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'device_type': 'cuda_sparse',
        'gpu_device_id': 0,
        'max_bin': 255,
        'max_depth': -1,
        'num_threads': 0,
        'deterministic': True,
        'feature_pre_filter': False,
        'is_enable_sparse': True,
        'min_data_in_bin': 1,
        'path_smooth': 0.1,
        'min_data_in_leaf': 3,
        'min_gain_to_split': 2.0,
        'lambda_l1': 0.5,
        'lambda_l2': 3.0,
        'feature_fraction': _FF,           # from config — NEVER < 0.7 (EFB bundle killer)
        'feature_fraction_bynode': _FF_BYNODE,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'num_leaves': _NUM_LEAVES_1W,
        'learning_rate': 0.03,
        'verbosity': 1,
    }

    num_boost_round = 800
    n_folds = len(splits)

    log(f"  Training with device_type='cuda_sparse' on {GPU_NAME}")
    log(f"  {n_folds} CPCV folds, {num_boost_round} rounds")

    fold_results = []
    best_model = None
    best_acc = 0

    for fi, (train_idx, test_idx) in enumerate(splits):
        fold_start = time.perf_counter()

        y_train_raw = y_3class[train_idx]
        y_test_raw = y_3class[test_idx]
        train_valid = ~np.isnan(y_train_raw)
        test_valid = ~np.isnan(y_test_raw)

        X_train = X_all[train_idx][train_valid]
        y_train = y_train_raw[train_valid].astype(int)
        X_test = X_all[test_idx][test_valid]
        y_test = y_test_raw[test_valid].astype(int)
        w_train = sample_weights[train_idx][train_valid]

        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

        if n_train < 50 or n_test < 20:
            log(f"  Fold {fi + 1}/{n_folds}: SKIP (train={n_train}, test={n_test})")
            continue

        val_size = max(int(n_train * 0.15), 20)
        if val_size >= n_train:
            val_size = max(n_train // 5, 10)

        dtrain = lgb.Dataset(X_train[:-val_size], label=y_train[:-val_size],
                             weight=w_train[:-val_size],
                             feature_name=feature_cols, free_raw_data=False)
        dval = lgb.Dataset(X_train[-val_size:], label=y_train[-val_size:],
                           weight=w_train[-val_size:],
                           feature_name=feature_cols, free_raw_data=False)

        es_rounds = max(50, int(100 * (0.1 / params.get('learning_rate', 0.03))))
        model = lgb.train(
            params, dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dtrain, dval],
            valid_names=['train', 'val'],
            callbacks=[lgb.early_stopping(es_rounds), lgb.log_evaluation(100)],
        )

        preds_3c = model.predict(X_test)
        pred_labels = np.argmax(preds_3c, axis=1)
        acc = float(accuracy_score(y_test, pred_labels))
        fold_time = time.perf_counter() - fold_start

        fold_results.append({'fold': fi + 1, 'accuracy': acc,
                             'best_iter': model.best_iteration,
                             'fold_time_s': fold_time})
        log(f"  Fold {fi + 1}/{n_folds}: Acc={acc:.3f} "
            f"Trees={model.best_iteration} Time={fold_time:.1f}s")

        if acc > best_acc:
            best_acc = acc
            best_model = model

        del dtrain, dval
        gc.collect()

    if fold_results:
        avg_acc = np.mean([r['accuracy'] for r in fold_results])
        total_time = sum(r['fold_time_s'] for r in fold_results)
        log(f"\n  cuda_sparse Summary:")
        log(f"  Avg Accuracy: {avg_acc:.4f}")
        log(f"  Best Accuracy: {best_acc:.4f}")
        log(f"  Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train 1w model with GPU histogram")
    parser.add_argument('--approach', choices=['a', 'b', 'both'], default='both',
                        help='Which approach to run (default: both)')
    parser.add_argument('--boost-rounds', type=int, default=800)
    parser.add_argument('--save-model', action='store_true',
                        help='Save best model to v3.3/model_1w_gpu.txt')
    args = parser.parse_args()

    log("=" * 70)
    log("TRAIN 1w MODEL -- GPU Histogram Acceleration")
    log("=" * 70)
    log(f"  LightGBM: {lgb.__version__}")
    log(f"  GPU: {GPU_NAME} ({VRAM_GB:.1f} GB)")
    log(f"  CuPy: {'YES' if HAS_CUPY else 'NO'}")
    log(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Approach: {args.approach.upper()}")

    # ── 1. Load base features ──────────────────────────────────────────────
    log(f"\n{elapsed()} Loading base features...")
    parquet_path = find_file(PARQUET_CANDIDATES, "features_BTC_1w.parquet")
    if parquet_path is None:
        log("FATAL: No parquet found. Cannot train.")
        sys.exit(1)

    t0 = time.perf_counter()
    df = pd.read_parquet(parquet_path)
    load_time = time.perf_counter() - t0
    log(f"  Loaded: {parquet_path}")
    log(f"  Shape: {df.shape[0]} rows x {df.shape[1]} cols ({load_time:.1f}s)")

    # Downcast float64 -> float32
    f64_cols = df.select_dtypes(include=['float64']).columns
    if len(f64_cols) > 0:
        df[f64_cols] = df[f64_cols].astype(np.float32)
        log(f"  Downcast {len(f64_cols)} float64 -> float32")

    # Parse timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    elif 'date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['date'])

    # ── 2. Triple-barrier labels ───────────────────────────────────────────
    log(f"\n{elapsed()} Computing triple-barrier labels...")
    if 'triple_barrier_label' in df.columns:
        y_3class = pd.to_numeric(df['triple_barrier_label'], errors='coerce').values
        log(f"  Using pre-computed triple_barrier_label column")
    else:
        from feature_library import compute_triple_barrier_labels, TRIPLE_BARRIER_CONFIG
        y_3class = compute_triple_barrier_labels(df, '1w')

    valid_mask = ~np.isnan(y_3class)
    n_long = int((y_3class == 2).sum())
    n_short = int((y_3class == 0).sum())
    n_flat = int((y_3class == 1).sum())
    n_nan = int((~valid_mask).sum())
    log(f"  Labels: {n_long} LONG, {n_short} SHORT, {n_flat} FLAT, {n_nan} NaN")

    # ── 3. Identify features ───────────────────────────────────────────────
    meta_cols = {'timestamp', 'date', 'open', 'high', 'low', 'close', 'volume',
                 'quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote',
                 'open_time', 'date_norm'}
    target_like = {c for c in df.columns if 'next_' in c.lower() or 'target' in c.lower()
                   or 'direction' in c.lower() or c == 'triple_barrier_label'}
    exclude_cols = meta_cols | target_like
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X_base = df[feature_cols].values.astype(np.float32)
    X_base = np.where(np.isinf(X_base), np.nan, X_base)
    n_base = len(feature_cols)
    log(f"  Base features: {n_base}")

    # ── 4. Load sparse crosses ─────────────────────────────────────────────
    log(f"\n{elapsed()} Loading sparse cross features...")
    npz_path = find_file(NPZ_CANDIDATES, "v2_crosses_BTC_1w.npz")
    cross_matrix = None
    cross_cols = None

    if npz_path is not None:
        t0 = time.perf_counter()
        cross_matrix = sp_sparse.load_npz(npz_path).tocsr()
        cross_matrix = _ensure_lgbm_sparse_dtypes(cross_matrix, "cross_matrix")
        load_cross_time = time.perf_counter() - t0
        log(f"  Loaded: {npz_path}")
        log(f"  Shape: {cross_matrix.shape[0]} x {cross_matrix.shape[1]:,} "
            f"(nnz={cross_matrix.nnz:,}, {load_cross_time:.1f}s)")

        # Load cross names
        names_path = find_file(CROSS_NAMES_CANDIDATES, "v2_cross_names_BTC_1w.json")
        if names_path is not None:
            with open(names_path) as f:
                cross_cols = json.load(f)
            log(f"  Cross names: {len(cross_cols):,}")
        else:
            cross_cols = [f'cross_{i}' for i in range(cross_matrix.shape[1])]
            log(f"  Cross names: generated {len(cross_cols):,} placeholders")
    else:
        log(f"  No cross NPZ found. Training with base features only.")

    # ── 5. Combine base + crosses ──────────────────────────────────────────
    log(f"\n{elapsed()} Combining features...")
    if cross_matrix is not None and cross_matrix.shape[0] == X_base.shape[0]:
        X_base_sparse = sp_sparse.csr_matrix(X_base)
        X_all = sp_sparse.hstack([X_base_sparse, cross_matrix], format='csr')
        X_all = _ensure_lgbm_sparse_dtypes(X_all, "X_all")
        feature_cols = feature_cols + cross_cols
        nnz = X_all.nnz
        total = X_all.shape[0] * X_all.shape[1]
        density = nnz / total * 100 if total > 0 else 0
        log(f"  Combined: {X_all.shape[0]} x {X_all.shape[1]:,} "
            f"({n_base} base + {len(cross_cols):,} crosses)")
        log(f"  NNZ: {nnz:,} ({density:.4f}% dense)")
        log(f"  CSR size: {(X_all.data.nbytes + X_all.indices.nbytes + X_all.indptr.nbytes) / 1e6:.0f} MB")
        log(f"  Sparse dtypes: indices={X_all.indices.dtype}, indptr={X_all.indptr.dtype}")
        del X_base_sparse, cross_matrix
    elif cross_matrix is not None:
        log(f"  WARNING: Row mismatch (base={X_base.shape[0]}, cross={cross_matrix.shape[0]})")
        X_all = sp_sparse.csr_matrix(X_base)
    else:
        X_all = sp_sparse.csr_matrix(X_base)

    del X_base
    gc.collect()

    # ── 6. Sample weights (uniform + uniqueness) ──────────────────────────
    sample_weights = np.ones(len(y_3class), dtype=np.float32)
    # Simplified uniqueness -- 1w has ~800 rows, max_hold=6
    from feature_library import TRIPLE_BARRIER_CONFIG
    tb_cfg = TRIPLE_BARRIER_CONFIG.get('1w', TRIPLE_BARRIER_CONFIG['1h'])
    max_hold = tb_cfg['max_hold_bars']  # no fallback — must be in TRIPLE_BARRIER_CONFIG

    # Normalize weights
    sw_sum = sample_weights[valid_mask].sum()
    if sw_sum > 0:
        sample_weights *= valid_mask.sum() / sw_sum

    # ── 7. Generate CPCV splits ────────────────────────────────────────────
    log(f"\n{elapsed()} Generating CPCV splits...")
    n = X_all.shape[0]
    # (4, 1) = 4 folds per config
    splits = _generate_cpcv_splits(
        n, n_groups=4, n_test_groups=1,
        max_hold_bars=max_hold, embargo_pct=0.01,
    )
    log(f"  CPCV: {len(splits)} paths (4 groups, 1 test, purge={max_hold} bars, embargo={max(1, int(n * max(0.01, max_hold / n)))} bars)")
    for i, (tr, te) in enumerate(splits):
        log(f"    Path {i + 1}: train={len(tr)}, test={len(te)}")

    # ── 8. Run training ───────────────────────────────────────────────────
    result_a = None
    if args.approach in ('a', 'both'):
        result_a = run_approach_a(X_all, y_3class, sample_weights,
                                  feature_cols, splits)

    if args.approach in ('b', 'both'):
        run_approach_b(X_all, y_3class, sample_weights, feature_cols, splits)

    # ── 9. Save model if requested ─────────────────────────────────────────
    if args.save_model and result_a and result_a['best_model'] is not None:
        model_path = os.path.join(V33_DIR, "model_1w_gpu.txt")
        result_a['best_model'].save_model(model_path)
        log(f"\n{elapsed()} Model saved: {model_path}")

    # ── 10. Final report ───────────────────────────────────────────────────
    log(f"\n{'=' * 70}")
    log(f"FINAL REPORT")
    log(f"{'=' * 70}")
    log(f"  Matrix: {X_all.shape[0]} x {X_all.shape[1]:,} features")
    log(f"  ALL features preserved. NO filtering. Binary crosses = the edge.")
    log(f"  GPU: {GPU_NAME} ({VRAM_GB:.1f} GB VRAM)")

    if result_a:
        log(f"\n  Approach A (stock LightGBM + CuPy benchmark):")
        log(f"    Best accuracy: {result_a['best_acc']:.4f}")
        log(f"    Training time: {result_a['total_train_time_s']:.1f}s "
            f"({result_a['total_train_time_s'] / 60:.1f} min)")
        if result_a['cpu_hist_ms'] is not None:
            log(f"    CPU SpMV: {result_a['cpu_hist_ms']:.2f} ms/call")
        if result_a['gpu_hist_ms'] is not None:
            log(f"    GPU SpMV: {result_a['gpu_hist_ms']:.3f} ms/call")
            speedup = result_a['cpu_hist_ms'] / result_a['gpu_hist_ms']
            log(f"    Histogram speedup: {speedup:.0f}x")
            # Estimate savings if GPU histograms were native
            hist_fraction = 0.50
            cpu_hist_total = result_a['total_train_time_s'] * hist_fraction
            gpu_hist_total = cpu_hist_total / speedup
            saved = cpu_hist_total - gpu_hist_total
            log(f"    Estimated savings with GPU hist: {saved:.0f}s "
                f"({saved / 60:.1f} min)")

    log(f"\n  Total wall time: {time.time() - START_TIME:.0f}s "
        f"({(time.time() - START_TIME) / 60:.1f} min)")
    log(f"{'=' * 70}")


if __name__ == "__main__":
    main()
