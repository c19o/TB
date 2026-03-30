#!/usr/bin/env python3
"""
cupy_gpu_train.py -- GPU-Accelerated Training with CuPy cuSPARSE Benchmarking
==============================================================================

Trains a LightGBM model on real 1w data using stock LightGBM on CPU,
while simultaneously benchmarking GPU histogram times via CuPy cuSPARSE.

Pipeline:
  1. Load real 1w data (parquet + NPZ cross features)
  2. Compute triple-barrier labels
  3. CPCV (4,1) = 4 folds with purging + embargo
  4. Per-fold: LightGBM train on CPU, timed per component
  5. CuPy cuSPARSE SpMV benchmark on same data for GPU comparison
  6. Final model retrain on all data
  7. Save model as model_1w.json
  8. Report: total time, per-fold breakdown, GPU histogram estimate, accuracy

Environment:
  - Stock LightGBM (pip install lightgbm)
  - CuPy (pip install cupy-cuda12x)
  - RTX 3090 for cuSPARSE benchmarks
  - NO custom C++ fork needed

Usage:
    cd v3.3/gpu_histogram_fork
    python cupy_gpu_train.py
"""

import os
import sys
import time
import json
import gc
import warnings
import pickle

warnings.filterwarnings('ignore')

# ── CUDA DLL path for Windows ──
if os.name == 'nt':
    cuda_bin = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin'
    if os.path.isdir(cuda_bin):
        os.add_dll_directory(cuda_bin)

import numpy as np
import pandas as pd
from scipy import sparse as sp_sparse

# ── Path setup ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
V33_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(V33_DIR)

# Add v3.3 to sys.path FIRST so we import v3.3/config.py (not root config.py)
# PROJECT_ROOT needed for astrology_engine.py etc.
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if V33_DIR not in sys.path:
    sys.path.insert(0, V33_DIR)  # must be index 0 — v3.3/config.py takes priority

import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, log_loss

START_TIME = time.time()
RESULTS = []


def elapsed():
    return f"[{time.time() - START_TIME:.0f}s]"


def log(msg):
    print(msg, flush=True)
    RESULTS.append(str(msg))


# ============================================================
# GPU DETECTION
# ============================================================
HAS_CUPY = False
GPU_NAME = "N/A"
VRAM_GB = 0.0

try:
    import cupy as cp
    from cupyx.scipy import sparse as cp_sparse
    VRAM_GB = cp.cuda.Device(0).mem_info[1] / (1024**3)
    try:
        props = cp.cuda.runtime.getDeviceProperties(0)
        GPU_NAME = props.get("name", "Unknown GPU")
    except Exception:
        GPU_NAME = "Unknown GPU"
    HAS_CUPY = True
    log(f"GPU: {GPU_NAME} ({VRAM_GB:.1f} GB VRAM)")
    log(f"CuPy: {cp.__version__}")
except ImportError:
    log("WARNING: CuPy not installed. GPU benchmarks will be skipped.")
    log("Install with: pip install cupy-cuda12x")
except Exception as e:
    log(f"WARNING: GPU not available: {e}")


# ============================================================
# DATA LOADING
# ============================================================
def find_file(candidates, label):
    """Find first existing file from candidates list."""
    for path in candidates:
        if os.path.isfile(path):
            return path
    log(f"ERROR: {label} not found. Searched:")
    for p in candidates:
        log(f"  {p}")
    return None


def load_1w_data():
    """Load 1w parquet + sparse cross NPZ. Returns (df, cross_matrix, cross_cols)."""
    # Parquet search paths
    parquet_candidates = [
        os.path.join(V33_DIR, "features_BTC_1w.parquet"),
        os.path.join(V33_DIR, "v2_base_1w.parquet"),
        os.path.join(PROJECT_ROOT, "v3.2_2.9M_Features", "features_BTC_1w.parquet"),
        os.path.join(PROJECT_ROOT, "v3.0 (LGBM)", "features_BTC_1w.parquet"),
    ]
    npz_candidates = [
        os.path.join(V33_DIR, "v2_crosses_BTC_1w.npz"),
        os.path.join(PROJECT_ROOT, "v3.2_2.9M_Features", "v2_crosses_BTC_1w.npz"),
        os.path.join(PROJECT_ROOT, "v3.0 (LGBM)", "v2_crosses_BTC_1w.npz"),
    ]
    cross_names_candidates = [
        os.path.join(V33_DIR, "v2_cross_names_BTC_1w.json"),
        os.path.join(PROJECT_ROOT, "v3.2_2.9M_Features", "v2_cross_names_BTC_1w.json"),
        os.path.join(PROJECT_ROOT, "v3.0 (LGBM)", "v2_cross_names_BTC_1w.json"),
    ]

    # Load parquet
    parquet_path = find_file(parquet_candidates, "1w parquet")
    if parquet_path is None:
        return None, None, None
    log(f"\n{elapsed()} Loading parquet: {parquet_path}")
    t0 = time.perf_counter()
    df = pd.read_parquet(parquet_path)
    log(f"  Loaded: {len(df)} rows x {len(df.columns)} cols ({time.perf_counter() - t0:.2f}s)")

    # Downcast float64 -> float32
    f64_cols = df.select_dtypes(include=['float64']).columns
    if len(f64_cols) > 0:
        df[f64_cols] = df[f64_cols].astype(np.float32)
        log(f"  Downcast {len(f64_cols)} float64 cols -> float32")

    # Load NPZ
    npz_path = find_file(npz_candidates, "1w cross NPZ")
    cross_matrix = None
    cross_cols = None
    if npz_path is not None:
        log(f"\n{elapsed()} Loading cross NPZ: {npz_path}")
        t0 = time.perf_counter()
        cross_matrix = sp_sparse.load_npz(npz_path).tocsr()
        load_t = time.perf_counter() - t0
        log(f"  Shape: {cross_matrix.shape[0]:,} x {cross_matrix.shape[1]:,}")
        log(f"  NNZ: {cross_matrix.nnz:,}")
        density = cross_matrix.nnz / max(1, cross_matrix.shape[0] * cross_matrix.shape[1])
        log(f"  Density: {density:.4%}")
        log(f"  Load time: {load_t:.2f}s")
        log(f"  CSR size: {(cross_matrix.data.nbytes + cross_matrix.indices.nbytes + cross_matrix.indptr.nbytes) / 1e6:.1f} MB")

        # Enforce correct dtypes for LightGBM
        if cross_matrix.indptr.dtype != np.int64:
            cross_matrix.indptr = cross_matrix.indptr.astype(np.int64)
        if cross_matrix.indices.dtype != np.int32:
            cross_matrix.indices = cross_matrix.indices.astype(np.int32)

        # Load cross names
        names_path = find_file(cross_names_candidates, "1w cross names")
        if names_path is not None:
            with open(names_path, 'r') as f:
                cross_cols = json.load(f)
            log(f"  Cross names: {len(cross_cols):,}")
        else:
            cross_cols = [f'cross_{i}' for i in range(cross_matrix.shape[1])]

    return df, cross_matrix, cross_cols


# ============================================================
# TRIPLE-BARRIER LABELS
# ============================================================
def compute_labels(df):
    """Compute triple-barrier labels for 1w data."""
    try:
        from feature_library import compute_triple_barrier_labels, TRIPLE_BARRIER_CONFIG
        tb_cfg = TRIPLE_BARRIER_CONFIG['1w']
        log(f"\n{elapsed()} Computing triple-barrier labels "
            f"(tp={tb_cfg['tp_atr_mult']}xATR, sl={tb_cfg['sl_atr_mult']}xATR, hold={tb_cfg['max_hold_bars']})")

        if 'triple_barrier_label' in df.columns:
            labels = pd.to_numeric(df['triple_barrier_label'], errors='coerce').values
            log(f"  Using pre-computed triple_barrier_label column")
        else:
            labels = compute_triple_barrier_labels(df, '1w')

        valid = ~np.isnan(labels)
        n_long = (labels == 2).sum()
        n_short = (labels == 0).sum()
        n_flat = (labels == 1).sum()
        n_nan = (~valid).sum()
        log(f"  Labels: {int(n_long)} LONG, {int(n_short)} SHORT, {int(n_flat)} FLAT, {int(n_nan)} NaN")
        return labels
    except ImportError as e:
        log(f"  WARNING: Could not import feature_library ({e}), using simple return-based labels")
        # Fallback: simple sign of close-to-close return
        close = pd.to_numeric(df['close'], errors='coerce').values
        ret = np.empty(len(close))
        ret[:] = np.nan
        ret[:-1] = (close[1:] - close[:-1]) / close[:-1]
        labels = np.where(ret > 0.02, 2, np.where(ret < -0.02, 0, 1)).astype(float)
        labels[np.isnan(ret)] = np.nan
        log(f"  Fallback labels: {int((labels==2).sum())} LONG, {int((labels==0).sum())} SHORT, {int((labels==1).sum())} FLAT")
        return labels


# ============================================================
# CPCV SPLIT GENERATION
# ============================================================
def generate_cpcv_splits(n_samples, n_groups=4, n_test_groups=1,
                         max_hold_bars=None, embargo_pct=0.01):
    """Generate Combinatorial Purged Cross-Validation splits (4,1)=4 folds."""
    from itertools import combinations

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
        test_boundaries = []
        for g in test_group_ids:
            test_boundaries.append(groups[g][0])
            test_boundaries.append(groups[g][-1])
        purge_mask = np.zeros(len(train_idx), dtype=bool)
        for boundary in test_boundaries:
            purge_mask |= (np.abs(train_idx - boundary) <= max_hold_bars)
        train_idx = train_idx[~purge_mask]

        # Embargo after each test group
        for g in test_group_ids:
            test_end = groups[g][-1]
            embargo_start = test_end + 1
            embargo_end = test_end + embargo_size
            embargo_mask = (train_idx >= embargo_start) & (train_idx <= embargo_end)
            train_idx = train_idx[~embargo_mask]

        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))

    return splits


# ============================================================
# cuSPARSE HISTOGRAM BENCHMARK
# ============================================================
def benchmark_gpu_histogram(X_csr, n_warmup=3, n_trials=20):
    """Benchmark GPU histogram building: CSR.T @ gradient_vector.

    This simulates LightGBM's histogram operation:
    gradient_sum_per_feature = feature_matrix.T @ gradient_vector

    Returns dict with timing results, or None if CuPy unavailable.
    """
    if not HAS_CUPY:
        return None

    n_rows, n_cols = X_csr.shape
    log(f"\n  cuSPARSE Histogram Benchmark ({n_rows:,} x {n_cols:,})")

    # Generate fake gradient/hessian vectors (same shape as training gradients)
    rng = np.random.default_rng(42)
    grad_cpu = rng.standard_normal(n_rows).astype(np.float64)
    hess_cpu = np.abs(rng.standard_normal(n_rows)).astype(np.float64) + 0.1

    # Transfer CSR to GPU
    try:
        t0 = time.perf_counter()
        X_gpu = cp_sparse.csr_matrix(X_csr.astype(np.float64))
        transfer_time = time.perf_counter() - t0
        log(f"  CSR -> GPU transfer: {transfer_time:.3f}s")
    except Exception as e:
        log(f"  GPU transfer failed: {e}")
        return None

    # Pre-compute CSC (transpose) for efficient column-wise access
    try:
        t0 = time.perf_counter()
        X_csc_gpu = X_gpu.tocsc()
        transpose_time = time.perf_counter() - t0
        log(f"  CSR -> CSC conversion (one-time): {transpose_time:.3f}s")
    except Exception as e:
        log(f"  CSC conversion failed: {e}")
        return None

    grad_gpu = cp.asarray(grad_cpu)
    hess_gpu = cp.asarray(hess_cpu)

    results = {}

    # Method 1: CSC.T @ gradient (CSC transpose = CSR, efficient for SpMV)
    # This is: csc_matrix.T @ vector = csr_matrix @ vector (coalesced access)
    log(f"  Method: CSC.T @ gradient (cuSPARSE SpMV, coalesced)")
    gpu_times = []
    for i in range(n_warmup + n_trials):
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        # Histogram = sum of gradients per feature = X.T @ grad
        # Using CSC: X_csc.T is CSR, so X_csc.T @ grad uses CSR SpMV (fast)
        grad_hist = X_csc_gpu.T @ grad_gpu
        hess_hist = X_csc_gpu.T @ hess_gpu
        cp.cuda.Stream.null.synchronize()
        dt = time.perf_counter() - t0
        if i >= n_warmup:
            gpu_times.append(dt)

    gpu_mean = np.mean(gpu_times)
    gpu_std = np.std(gpu_times)
    gpu_min = np.min(gpu_times)
    log(f"    Mean: {gpu_mean * 1000:.2f}ms  Std: {gpu_std * 1000:.2f}ms  Min: {gpu_min * 1000:.2f}ms")
    results['gpu_histogram_ms'] = gpu_mean * 1000
    results['gpu_histogram_min_ms'] = gpu_min * 1000

    # Method 2: CPU scipy baseline (for comparison)
    log(f"  Method: scipy CPU (CSR.T @ gradient)")
    X_csc_cpu = X_csr.tocsc()
    cpu_times = []
    for i in range(min(5, n_trials)):
        t0 = time.perf_counter()
        grad_hist_cpu = X_csc_cpu.T @ grad_cpu
        hess_hist_cpu = X_csc_cpu.T @ hess_cpu
        dt = time.perf_counter() - t0
        cpu_times.append(dt)

    cpu_mean = np.mean(cpu_times)
    cpu_min = np.min(cpu_times)
    log(f"    Mean: {cpu_mean * 1000:.2f}ms  Min: {cpu_min * 1000:.2f}ms")
    results['cpu_histogram_ms'] = cpu_mean * 1000
    results['cpu_histogram_min_ms'] = cpu_min * 1000

    # Speedup
    speedup = cpu_mean / gpu_mean if gpu_mean > 0 else 0
    log(f"  GPU Speedup: {speedup:.1f}x")
    results['speedup'] = speedup

    # Verify correctness
    grad_hist_check = cp.asnumpy(grad_hist)
    grad_hist_ref = (X_csc_cpu.T @ grad_cpu)
    max_err = np.max(np.abs(grad_hist_check - grad_hist_ref))
    rel_err = max_err / (np.max(np.abs(grad_hist_ref)) + 1e-15)
    results['correctness'] = 'PASS' if rel_err < 1e-8 else f'FAIL (rel_err={rel_err:.2e})'
    log(f"  Correctness: {results['correctness']}")

    # Cleanup GPU memory
    del X_gpu, X_csc_gpu, grad_gpu, hess_gpu, grad_hist, hess_hist
    cp.get_default_memory_pool().free_all_blocks()

    return results


# ============================================================
# SINGLE FOLD TRAINING (with detailed timing)
# ============================================================
def train_fold(wi, train_idx, test_idx, X_all, y_3class, sample_weights,
               feature_cols, lgb_params, num_boost_round=800, parent_ds=None):
    """Train a single CPCV fold with detailed timing breakdown.

    Returns dict with metrics, timing, and model.
    """
    timings = {}

    y_train_raw = y_3class[train_idx]
    y_test_raw = y_3class[test_idx]
    train_valid = ~np.isnan(y_train_raw)
    test_valid = ~np.isnan(y_test_raw)

    X_train = X_all[train_idx][train_valid]
    y_train = y_train_raw[train_valid].astype(int)
    X_test = X_all[test_idx][test_valid]
    y_test = y_test_raw[test_valid].astype(int)
    test_idx_valid = test_idx[test_valid]
    w_train = sample_weights[train_idx][train_valid]

    n_train, n_test = X_train.shape[0], X_test.shape[0]
    if n_train < 50 or n_test < 20:
        return None

    # 85/15 train/val split for early stopping
    val_size = max(int(n_train * 0.15), 20)
    if val_size >= n_train:
        val_size = max(n_train // 5, 10)

    X_val_es = X_train[-val_size:]
    y_val_es = y_train[-val_size:]
    w_val_es = w_train[-val_size:]
    X_train_es = X_train[:-val_size]
    y_train_es = y_train[:-val_size]
    w_train_es = w_train[:-val_size]

    # ── Dataset construction timing ──
    t0 = time.perf_counter()
    ds_kwargs = dict(feature_name=feature_cols, free_raw_data=False)
    if parent_ds is not None:
        ds_kwargs['reference'] = parent_ds
    dtrain = lgb.Dataset(X_train_es, label=y_train_es, weight=w_train_es, **ds_kwargs)
    dval = lgb.Dataset(X_val_es, label=y_val_es, weight=w_val_es, **ds_kwargs)
    timings['dataset_construct_s'] = time.perf_counter() - t0

    # ── Training timing ──
    early_stop_rounds = max(50, int(100 * (0.1 / lgb_params.get('learning_rate', 0.03))))
    t0 = time.perf_counter()
    model = lgb.train(
        lgb_params, dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dtrain, dval],
        valid_names=['train', 'val'],
        callbacks=[lgb.early_stopping(early_stop_rounds), lgb.log_evaluation(0)],
    )
    timings['training_s'] = time.perf_counter() - t0
    timings['best_iteration'] = model.best_iteration

    # ── Prediction timing ──
    t0 = time.perf_counter()
    preds_3c = model.predict(X_test)
    timings['predict_s'] = time.perf_counter() - t0

    pred_labels = np.argmax(preds_3c, axis=1)
    acc = float(accuracy_score(y_test, pred_labels))
    prec_long = float(precision_score(y_test, pred_labels, labels=[2], average='macro', zero_division=0))
    prec_short = float(precision_score(y_test, pred_labels, labels=[0], average='macro', zero_division=0))
    mlogloss = float(log_loss(y_test, preds_3c, labels=[0, 1, 2]))

    # IS metrics
    is_preds = model.predict(X_train)
    is_labels = np.argmax(is_preds, axis=1)
    is_acc = float(accuracy_score(y_train, is_labels))

    importance = dict(zip(model.feature_name(), model.feature_importance(importance_type='gain')))

    timings['total_fold_s'] = timings['dataset_construct_s'] + timings['training_s'] + timings['predict_s']

    # Estimate: how much of training time is histogram building?
    # LightGBM histogram building is ~60-80% of training time for sparse data.
    # The rest is split finding + leaf value computation.
    n_trees = model.best_iteration
    est_hist_pct = 0.70  # conservative estimate for sparse
    timings['est_histogram_s'] = timings['training_s'] * est_hist_pct
    timings['est_histogram_per_tree_ms'] = (timings['est_histogram_s'] / max(n_trees, 1)) * 1000

    return {
        'fold': wi,
        'accuracy': acc,
        'prec_long': prec_long,
        'prec_short': prec_short,
        'mlogloss': mlogloss,
        'is_accuracy': is_acc,
        'n_train': n_train,
        'n_test': n_test,
        'n_trees': n_trees,
        'timings': timings,
        'importance': importance,
        'model': model,
        'preds_3c': preds_3c,
        'y_test': y_test,
        'test_idx_valid': test_idx_valid,
    }


# ============================================================
# MAIN
# ============================================================
def main():
    log("=" * 70)
    log("cupy_gpu_train.py -- GPU-Accelerated Training Benchmark")
    log("=" * 70)
    log(f"LightGBM: {lgb.__version__}")
    log(f"Platform: {'Windows' if os.name == 'nt' else 'Linux'}")
    log(f"GPU available: {HAS_CUPY} ({GPU_NAME})")
    log("")

    # ── Load data ──
    df, cross_matrix, cross_cols = load_1w_data()
    if df is None:
        log("FATAL: No data found. Exiting.")
        return

    # ── Compute labels ──
    y_3class = compute_labels(df)

    # ── Build feature matrix ──
    log(f"\n{elapsed()} Building feature matrix...")

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

    # Combine base + crosses
    if cross_matrix is not None and cross_matrix.shape[0] == X_base.shape[0]:
        X_base_sparse = sp_sparse.csr_matrix(X_base)
        X_all = sp_sparse.hstack([X_base_sparse, cross_matrix], format='csr')
        # Enforce LightGBM-compatible dtypes
        if X_all.indptr.dtype != np.int64:
            X_all.indptr = X_all.indptr.astype(np.int64)
        if X_all.indices.dtype != np.int32:
            X_all.indices = X_all.indices.astype(np.int32)
        feature_cols = feature_cols + cross_cols
        is_sparse = True
        nnz = X_all.nnz
        total = X_all.shape[0] * X_all.shape[1]
        density = nnz / total * 100 if total > 0 else 0
        log(f"  Combined: {X_all.shape[0]:,} x {X_all.shape[1]:,} "
            f"({n_base} base + {len(cross_cols):,} crosses)")
        log(f"  NNZ: {nnz:,} ({density:.4f}% dense)")
        log(f"  CSR size: {(X_all.data.nbytes + X_all.indices.nbytes + X_all.indptr.nbytes) / 1e6:.1f} MB")
        del X_base_sparse, cross_matrix
    elif cross_matrix is not None:
        log(f"  WARNING: Cross matrix rows ({cross_matrix.shape[0]}) != base ({X_base.shape[0]}), skipping crosses")
        X_all = X_base
        is_sparse = False
    else:
        X_all = X_base
        is_sparse = False
    del X_base, df
    gc.collect()

    n_features = len(feature_cols)
    log(f"  Total features: {n_features:,} ({'SPARSE' if is_sparse else 'DENSE'})")

    # SPARSE throughout — CLAUDE.md §4: no dense conversion (LightGBM CSR native + EFB)
    X_all_sparse_backup = X_all if is_sparse else sp_sparse.csr_matrix(X_all)

    # ── Sample weights (uniform) ──
    sample_weights = np.ones(len(y_3class), dtype=np.float32)

    # ── LightGBM params ──
    try:
        from config import V3_LGBM_PARAMS, TF_MIN_DATA_IN_LEAF, TF_NUM_LEAVES
        lgb_params = V3_LGBM_PARAMS.copy()
        lgb_params['min_data_in_leaf'] = TF_MIN_DATA_IN_LEAF.get('1w', 3)
        lgb_params['num_leaves'] = TF_NUM_LEAVES.get('1w', 31)
    except ImportError:
        lgb_params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "device": "cpu",
            "force_col_wise": True,
            "max_bin": 255,
            "max_depth": -1,
            "num_threads": 0,
            "deterministic": True,
            "feature_pre_filter": False,
            "is_enable_sparse": True,
            "min_data_in_bin": 1,
            "path_smooth": 0.1,
            "min_data_in_leaf": 3,
            "min_gain_to_split": 2.0,
            "lambda_l1": 0.5,
            "lambda_l2": 3.0,
            "feature_fraction": 0.9,      # NEVER < 0.7 — kills rare esoteric EFB bundles
            "feature_fraction_bynode": 0.8,  # effective = ff * ff_bynode; keep both high
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "num_leaves": 31,
            "learning_rate": 0.03,
            "verbosity": -1,
        }

    log(f"\n{elapsed()} LightGBM params:")
    for k, v in sorted(lgb_params.items()):
        log(f"  {k}: {v}")

    # ── Build parent Dataset (EFB bin reuse across folds) ──
    log(f"\n{elapsed()} Building parent Dataset (EFB bins computed once)...")
    parent_ds = None
    try:
        t0 = time.perf_counter()
        valid_mask = ~np.isnan(y_3class)
        X_parent = X_all[valid_mask]
        parent_ds = lgb.Dataset(
            X_parent, label=y_3class[valid_mask].astype(int),
            weight=sample_weights[valid_mask],
            feature_name=feature_cols,
            free_raw_data=True,
            params={'feature_pre_filter': False, 'max_bin': lgb_params.get('max_bin', 255)},
        )
        parent_ds.construct()
        parent_time = time.perf_counter() - t0
        log(f"  Parent Dataset: {X_parent.shape[1]:,} features, built in {parent_time:.1f}s")
        del X_parent
        gc.collect()
    except Exception as e:
        log(f"  WARNING: Parent Dataset build failed ({e}), folds build independently")
        parent_ds = None

    # ============================================================
    # CPCV TRAINING LOOP
    # ============================================================
    log(f"\n{'=' * 70}")
    log(f"CPCV TRAINING (4 groups, 1 test = 4 folds)")
    log(f"{'=' * 70}")

    n = X_all.shape[0]
    from feature_library import TRIPLE_BARRIER_CONFIG as _TBC
    _max_hold = _TBC.get('1w', _TBC['1h'])['max_hold_bars']
    splits = generate_cpcv_splits(n, n_groups=4, n_test_groups=1, max_hold_bars=_max_hold)
    log(f"  Generated {len(splits)} CPCV paths")
    for i, (tr, te) in enumerate(splits):
        log(f"    Path {i + 1}: train={len(tr)}, test={len(te)}")

    fold_results = []
    best_model = None
    best_acc = 0.0
    total_train_time = 0.0

    for wi, (train_idx, test_idx) in enumerate(splits):
        log(f"\n  --- Fold {wi + 1}/{len(splits)} ---")

        result = train_fold(
            wi, train_idx, test_idx, X_all, y_3class, sample_weights,
            feature_cols, lgb_params, num_boost_round=800, parent_ds=parent_ds,
        )

        if result is None:
            log(f"    SKIP -- not enough samples")
            continue

        t = result['timings']
        total_train_time += t['total_fold_s']

        log(f"    Acc={result['accuracy']:.3f}  PrecL={result['prec_long']:.3f}  "
            f"PrecS={result['prec_short']:.3f}  mlogloss={result['mlogloss']:.4f}  "
            f"Trees={result['n_trees']}")
        log(f"    Timing: Dataset={t['dataset_construct_s']:.2f}s  "
            f"Train={t['training_s']:.2f}s  Predict={t['predict_s']:.3f}s  "
            f"Total={t['total_fold_s']:.2f}s")
        log(f"    Est. histogram time: {t['est_histogram_s']:.2f}s "
            f"({t['est_histogram_per_tree_ms']:.1f}ms/tree)")
        log(f"    IS Accuracy: {result['is_accuracy']:.3f}")

        fold_results.append(result)
        if result['accuracy'] > best_acc:
            best_acc = result['accuracy']
            best_model = result['model']

    # ============================================================
    # GPU HISTOGRAM BENCHMARK
    # ============================================================
    gpu_bench = None
    if HAS_CUPY and X_all_sparse_backup is not None:
        log(f"\n{'=' * 70}")
        log(f"GPU HISTOGRAM BENCHMARK (cuSPARSE)")
        log(f"{'=' * 70}")
        gpu_bench = benchmark_gpu_histogram(X_all_sparse_backup)

    # ============================================================
    # FINAL MODEL RETRAIN ON ALL DATA
    # ============================================================
    log(f"\n{'=' * 70}")
    log(f"FINAL MODEL -- ALL DATA")
    log(f"{'=' * 70}")

    all_mask = ~np.isnan(y_3class)
    X_final = X_all[all_mask]
    y_final = y_3class[all_mask].astype(int)
    w_final = sample_weights[all_mask]

    # 85/15 split for early stopping
    n_final = X_final.shape[0]
    val_sz = max(int(n_final * 0.15), 20)
    X_val_f = X_final[-val_sz:]
    y_val_f = y_final[-val_sz:]
    w_val_f = w_final[-val_sz:]
    X_tr_f = X_final[:-val_sz]
    y_tr_f = y_final[:-val_sz]
    w_tr_f = w_final[:-val_sz]

    final_params = lgb_params.copy()
    import multiprocessing
    total_cores = multiprocessing.cpu_count() or 24
    final_params['num_threads'] = total_cores

    ds_kwargs = dict(feature_name=feature_cols, free_raw_data=False)
    if parent_ds is not None:
        ds_kwargs['reference'] = parent_ds

    log(f"\n{elapsed()} Training final model ({n_final} rows, {len(feature_cols):,} features, "
        f"{total_cores} threads)...")
    t0 = time.perf_counter()
    dtrain = lgb.Dataset(X_tr_f, label=y_tr_f, weight=w_tr_f, **ds_kwargs)
    dval = lgb.Dataset(X_val_f, label=y_val_f, weight=w_val_f, **ds_kwargs)
    final_ds_time = time.perf_counter() - t0
    log(f"  Dataset construction: {final_ds_time:.2f}s")

    early_stop = max(50, int(100 * (0.1 / final_params.get('learning_rate', 0.03))))
    t0 = time.perf_counter()
    final_model = lgb.train(
        final_params, dtrain, num_boost_round=800,
        valid_sets=[dtrain, dval], valid_names=['train', 'val'],
        callbacks=[lgb.early_stopping(early_stop), lgb.log_evaluation(100)],
    )
    final_train_time = time.perf_counter() - t0

    # Evaluate
    final_preds = final_model.predict(X_val_f)
    final_labels = np.argmax(final_preds, axis=1)
    final_acc = float(accuracy_score(y_val_f, final_labels))
    final_prec_l = float(precision_score(y_val_f, final_labels, labels=[2], average='macro', zero_division=0))
    final_prec_s = float(precision_score(y_val_f, final_labels, labels=[0], average='macro', zero_division=0))
    final_mlogloss = float(log_loss(y_val_f, final_preds, labels=[0, 1, 2]))

    log(f"\n  FINAL MODEL:")
    log(f"    Accuracy: {final_acc:.3f}")
    log(f"    Precision LONG: {final_prec_l:.3f}")
    log(f"    Precision SHORT: {final_prec_s:.3f}")
    log(f"    Multi-logloss: {final_mlogloss:.4f}")
    log(f"    Trees: {final_model.best_iteration}")
    log(f"    Training time: {final_train_time:.1f}s")

    # Top 30 features by gain
    importance = dict(zip(final_model.feature_name(),
                          final_model.feature_importance(importance_type='gain')))
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    log(f"\n  TOP 30 FEATURES BY GAIN:")
    for i, (fname, gain) in enumerate(sorted_imp[:30]):
        log(f"    {i + 1:3d}. {fname:<45s} gain={gain:.1f}")

    # Save model
    model_path = os.path.join(V33_DIR, 'model_1w.json')
    model_tmp = model_path + '.tmp'
    final_model.save_model(model_tmp)
    os.replace(model_tmp, model_path)
    model_size_mb = os.path.getsize(model_path) / 1e6
    log(f"\n  Model saved: {model_path} ({model_size_mb:.1f} MB)")

    # Save feature list
    feat_path = os.path.join(V33_DIR, 'features_1w_all.json')
    feat_tmp = feat_path + '.tmp'
    with open(feat_tmp, 'w') as f:
        json.dump(feature_cols, f, indent=2)
    os.replace(feat_tmp, feat_path)
    log(f"  Features saved: {feat_path}")

    # ============================================================
    # FINAL REPORT
    # ============================================================
    log(f"\n{'=' * 70}")
    log(f"TRAINING REPORT")
    log(f"{'=' * 70}")

    # CPCV summary
    if fold_results:
        accs = [r['accuracy'] for r in fold_results]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        log(f"\n  CPCV Results ({len(fold_results)} folds):")
        log(f"    Mean Accuracy: {mean_acc:.3f} +/- {std_acc:.3f}")
        log(f"    Best Fold Accuracy: {max(accs):.3f}")
        log(f"    Worst Fold Accuracy: {min(accs):.3f}")
        log(f"    Baseline comparison: 71.9% (v3.3 1w)")

        log(f"\n  Per-Fold Timing Breakdown:")
        log(f"    {'Fold':<6} {'Dataset':>10} {'Train':>10} {'Predict':>10} "
            f"{'Total':>10} {'Est Hist':>10} {'Trees':>6}")
        total_hist_est = 0
        total_training = 0
        for r in fold_results:
            t = r['timings']
            total_hist_est += t['est_histogram_s']
            total_training += t['training_s']
            log(f"    {r['fold'] + 1:<6} {t['dataset_construct_s']:>9.2f}s "
                f"{t['training_s']:>9.2f}s {t['predict_s']:>9.3f}s "
                f"{t['total_fold_s']:>9.2f}s {t['est_histogram_s']:>9.2f}s "
                f"{r['n_trees']:>6}")

        log(f"\n    Total CPCV training: {total_train_time:.1f}s")
        log(f"    Total estimated histogram time: {total_hist_est:.1f}s "
            f"({total_hist_est / max(total_training, 0.01) * 100:.0f}% of training)")

    # GPU benchmark summary
    if gpu_bench is not None:
        log(f"\n  GPU Histogram Benchmark:")
        log(f"    GPU (cuSPARSE): {gpu_bench['gpu_histogram_ms']:.2f}ms per SpMV")
        log(f"    CPU (scipy):    {gpu_bench['cpu_histogram_ms']:.2f}ms per SpMV")
        log(f"    Speedup:        {gpu_bench['speedup']:.1f}x")
        log(f"    Correctness:    {gpu_bench['correctness']}")

        if fold_results:
            # Estimate: if GPU did histograms, how much time would we save?
            avg_trees = np.mean([r['n_trees'] for r in fold_results])
            cpu_hist_per_tree = total_hist_est / max(sum(r['n_trees'] for r in fold_results), 1) * 1000
            gpu_hist_per_tree = gpu_bench['gpu_histogram_ms']  # SpMV for grad + hess = 2x this
            gpu_hist_per_tree_total = gpu_hist_per_tree * 2  # grad + hess

            log(f"\n  Projected GPU Training Speedup:")
            log(f"    CPU histogram/tree: {cpu_hist_per_tree:.1f}ms (estimated from LightGBM wall time)")
            log(f"    GPU histogram/tree: {gpu_hist_per_tree_total:.1f}ms (measured cuSPARSE SpMV x2)")
            if cpu_hist_per_tree > 0:
                tree_speedup = cpu_hist_per_tree / gpu_hist_per_tree_total
                log(f"    Per-tree histogram speedup: {tree_speedup:.1f}x")
                # Total training: histogram time saved + non-histogram time unchanged
                non_hist_time = total_training - total_hist_est
                gpu_hist_total = sum(r['n_trees'] for r in fold_results) * gpu_hist_per_tree_total / 1000
                projected_total = non_hist_time + gpu_hist_total
                actual_total = total_training
                log(f"    Projected CPCV training: {projected_total:.1f}s (was {actual_total:.1f}s)")
                log(f"    Projected wall-time speedup: {actual_total / max(projected_total, 0.01):.1f}x")

    # Final model stats
    log(f"\n  Final Model:")
    log(f"    Accuracy (val): {final_acc:.3f}")
    log(f"    Trees: {final_model.best_iteration}")
    log(f"    Training time: {final_train_time:.1f}s")
    log(f"    Model size: {model_size_mb:.1f} MB")
    log(f"    Features: {len(feature_cols):,}")

    total_elapsed = time.time() - START_TIME
    log(f"\n  Total wall time: {total_elapsed:.0f}s ({total_elapsed / 60:.1f} min)")

    # Save report
    report_path = os.path.join(SCRIPT_DIR, 'gpu_train_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(RESULTS))
    log(f"\n  Report saved: {report_path}")

    log(f"\n{'=' * 70}")
    log(f"DONE")
    log(f"{'=' * 70}")


if __name__ == '__main__':
    main()
