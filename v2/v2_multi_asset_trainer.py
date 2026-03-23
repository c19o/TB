#!/usr/bin/env python
"""
v2_multi_asset_trainer.py — V2 Multi-Asset Training Pipeline
==============================================================
Trains on ALL 31 assets with ALL features (base + sparse crosses).
NO FILTERING. XGBoost decides via tree splits.

Pipeline:
  1. Load base features (parquet) + sparse crosses (.npz) per asset
  2. Stack all assets into unified sparse training matrix
  3. Triple-barrier labels per asset
  4. CPCV walk-forward (purging + embargo)
  5. XGBoost gpu_hist or LightGBM EFB on sparse CSR — max_bin=32, colsample=0.15
  6. Per-asset models + unified model + per-crypto production models
  7. Meta-labeling + PBO validation

Speed optimizations (4-6M features):
  - max_bin=32 (binary features only need 2 bins) = ~6x speedup
  - colsample_bytree=0.15 (15% features per tree) = ~7x speedup
  - colsample_bynode=0.5 = ~2x on top
  - early_stopping_rounds=50
  - grow_policy='lossguide' with max_leaves=64
  - DeviceQuantileDMatrix for XGBoost GPU (avoids CPU-side quantile sketch)
  - LightGBM EFB alternative for massive sparse binary feature spaces
  - nthread=-1 (all CPU cores)

Usage:
  python v2_multi_asset_trainer.py --mode unified --tf 1d
  python v2_multi_asset_trainer.py --mode per-asset --tf 1d
  python v2_multi_asset_trainer.py --mode production --tf 1d 1h 4h
  python v2_multi_asset_trainer.py --mode all --tf 1d
  python v2_multi_asset_trainer.py --mode unified --tf 1d --engine lightgbm
  python v2_multi_asset_trainer.py --mode unified --tf 1d --engine both
  python v2_multi_asset_trainer.py --mode unified --tf 1d                    # auto-parallel if N_GPUS > 1
  python v2_multi_asset_trainer.py --mode unified --tf 1d --no-parallel-splits  # force sequential
  python v2_multi_asset_trainer.py --mode unified --tf 1d --use-dask        # Dask distributed multi-GPU
"""

import os, sys, time, json, argparse, warnings, pickle, gc, glob
import numpy as np
import pandas as pd
from scipy import sparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from numba import njit

warnings.filterwarnings('ignore')

V2_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, V2_DIR)

import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss

from config import ALL_TRAINING, TRAINING_CRYPTO, TRAINING_STOCKS
from feature_library import compute_triple_barrier_labels, TRIPLE_BARRIER_CONFIG

# Dask-XGBoost for distributed multi-GPU training (optional)
try:
    from dask_cuda import LocalCUDACluster
    from dask.distributed import Client
    import xgboost.dask as dxgb
    import dask.array as da
    HAS_DASK_CUDA = True
except ImportError:
    HAS_DASK_CUDA = False

START_TIME = time.time()

def elapsed():
    return f"[{time.time()-START_TIME:.0f}s]"

def log(msg):
    print(f"{elapsed()} {msg}", flush=True)


# ============================================================
# GPU DETECTION (via hardware_detect.py)
# ============================================================

from hardware_detect import detect_hardware

USE_GPU = False
N_GPUS = 0
_hw = {}
try:
    _test = xgb.DMatrix(np.random.rand(10, 5), label=np.random.randint(0, 2, 10), nthread=-1)
    xgb.train({'tree_method': 'hist', 'device': 'cuda', 'max_depth': 2}, _test, num_boost_round=2)
    USE_GPU = True
    _hw = detect_hardware()
    N_GPUS = _hw['n_gpus'] or 1
except:
    pass
log(f"GPU: {'ENABLED' if USE_GPU else 'CPU'} ({N_GPUS} device{'s' if N_GPUS != 1 else ''})")


# ============================================================
# XGB PARAMS — OPTIMIZED FOR 4-6M SPARSE FEATURES
# ============================================================

V2_XGB_PARAMS = {
    'objective': 'multi:softprob',
    'num_class': 3,
    'eval_metric': 'mlogloss',
    'verbosity': 0,

    # Tree structure
    'grow_policy': 'lossguide',  # Leaf-wise = faster than depth-wise
    'max_leaves': 64,
    'max_depth': 0,  # 0 = no limit when using lossguide + max_leaves
    'min_child_weight': 50,  # Default; overridden per-TF in _apply_tf_params()

    # SPEED: histogram bins — binary features only need 2 bins
    'max_bin': 32,  # 8x faster than default 256, plenty for binary + continuous

    # SPEED: column sampling cascade — 15% × 50% × 50% = 3.75% per split
    'colsample_bytree': 0.15,   # 15% of 4M = 600K features per tree
    'colsample_bylevel': 0.50,  # 50% of 400K = 200K per level
    'colsample_bynode': 0.50,   # 50% of 200K = 100K per split candidate

    # Row sampling
    'subsample': 0.8,

    # Regularization — strong for 4M+ features
    'reg_lambda': 10.0,  # L2
    'reg_alpha': 5.0,    # L1 — prune marginal sparse splits
    'gamma': 2.0,        # Min loss reduction per split

    # Learning
    'learning_rate': 0.03,  # Slow = more robust
    'tree_method': 'hist',
    'nthread': -1,  # ALL CPU cores
}

if USE_GPU:
    V2_XGB_PARAMS['device'] = 'cuda'

# Per-TF min_child_weight: XGBoost decides with sensitivity matched to dataset size.
# Low TFs (1d/1w) have fewer rows → rare esoteric signals need lower thresholds to split.
# High TFs (5m/15m) have millions of rows → need higher thresholds for stability.
TF_MIN_CHILD_WEIGHT = {
    '1w': 10, '1d': 10,
    '4h': 20,
    '1h': 25,
    '15m': 50, '5m': 50,
}

def _apply_tf_params(params, tf_name):
    """Apply per-TF overrides to XGBoost/LightGBM params. Protects rare esoteric signals."""
    params = params.copy()
    mcw = TF_MIN_CHILD_WEIGHT.get(tf_name, 50)
    params['min_child_weight'] = mcw
    return params


# ============================================================
# LGBM PARAMS — EFB FOR MASSIVE SPARSE BINARY FEATURES
# ============================================================

V2_LGBM_PARAMS = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'verbosity': -1,
    'device': 'gpu' if USE_GPU else 'cpu',
    'gpu_use_dp': False,
    'num_leaves': 64,
    'min_child_samples': 50,
    'feature_fraction': 0.15,  # equivalent to colsample_bytree
    'feature_fraction_bynode': 0.50,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l2': 10.0,
    'lambda_l1': 5.0,
    'min_gain_to_split': 2.0,
    'learning_rate': 0.03,
    'max_bin': 32,
    'num_threads': -1,
    'is_enable_sparse': True,  # LightGBM EFB for sparse features
}


# ============================================================
# DATA LOADING
# ============================================================

def load_asset_data(symbol, tf):
    """
    Load base features (parquet) + sparse crosses (.npz) for one asset.
    Returns: (base_df, sparse_crosses, cross_names)
    """
    base_path = os.path.join(V2_DIR, f'features_{symbol}_{tf}.parquet')
    sparse_path = os.path.join(V2_DIR, f'v2_crosses_{symbol}_{tf}.npz')
    names_path = os.path.join(V2_DIR, f'v2_cross_names_{symbol}_{tf}.json')

    if not os.path.exists(base_path):
        return None, None, None

    base_df = pd.read_parquet(base_path)

    sparse_crosses = None
    cross_names = None
    if os.path.exists(sparse_path) and os.path.exists(names_path):
        sparse_crosses = sparse.load_npz(sparse_path)
        with open(names_path, 'r') as f:
            cross_names = json.load(f)

    return base_df, sparse_crosses, cross_names


def _load_and_process_asset(symbol, tf):
    """
    Load + process a single asset for training (I/O-bound parquet + npz reads).
    Returns dict with processed arrays, or None if asset is skipped.
    """
    base_df, sp_crosses, cn = load_asset_data(symbol, tf)
    if base_df is None:
        return None

    # Get feature columns
    meta_cols = {'timestamp', 'date', 'open', 'high', 'low', 'close', 'volume',
                 'quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote',
                 'open_time', 'date_norm'}
    target_cols = {c for c in base_df.columns if 'next_' in c.lower() or 'target' in c.lower()
                   or c == 'triple_barrier_label'}
    exclude = meta_cols | target_cols
    feat_cols = [c for c in base_df.columns if c not in exclude]

    # Base features — convert to sparse CSC immediately to avoid dense RAM blow-up
    n_rows_total = len(base_df)
    X_base_dense = base_df[feat_cols].values.astype(np.float32)
    X_base_dense = np.where(np.isinf(X_base_dense), np.nan, X_base_dense)

    # Labels
    if 'triple_barrier_label' in base_df.columns:
        y = pd.to_numeric(base_df['triple_barrier_label'], errors='coerce').values
    else:
        y = compute_triple_barrier_labels(base_df, tf)

    # Valid mask (non-NaN labels)
    valid = ~np.isnan(y)
    if valid.sum() < 50:
        del X_base_dense, base_df
        gc.collect()
        return None

    # Sample weights
    w = np.ones(len(y), dtype=np.float32)

    # Timestamps
    if 'timestamp' in base_df.columns:
        ts = pd.to_datetime(base_df['timestamp']).values
    else:
        ts = base_df.index.values

    # Asset IDs for tracking
    asset_ids = np.full(len(y), symbol, dtype=object)

    # Filter to valid only — then convert to sparse and free dense
    X_base_valid = X_base_dense[valid]
    del X_base_dense, base_df  # free dense originals immediately
    gc.collect()

    X_base_sparse = sparse.csr_matrix(X_base_valid)
    del X_base_valid  # free the filtered dense copy
    gc.collect()

    y_valid = y[valid].astype(int)
    w = w[valid]
    ts = ts[valid]
    asset_ids = asset_ids[valid]

    # Sparse crosses — filter to same valid rows
    sp_valid = None
    if sp_crosses is not None and sp_crosses.shape[0] == n_rows_total:
        sp_valid = sp_crosses[valid]
    del sp_crosses  # free cross reference
    gc.collect()

    return {
        'symbol': symbol,
        'feat_cols': feat_cols,
        'cross_names': cn,
        'X_base_sparse': X_base_sparse,
        'sp_valid': sp_valid,
        'y_valid': y_valid,
        'w': w,
        'ts': ts,
        'asset_ids': asset_ids,
    }


def prepare_training_data(symbols, tf):
    """
    Load and stack all assets into unified training matrices.
    Uses ThreadPoolExecutor(max_workers=8) for parallel parquet reads (I/O-bound).
    Returns: (X_sparse, y, sample_weights, feature_names, timestamps, asset_ids)
    """
    log(f"Loading {len(symbols)} assets for {tf} (8 parallel readers)...")

    all_base_X = []
    all_sparse = []
    all_y = []
    all_weights = []
    all_timestamps = []
    all_asset_ids = []
    base_feature_names = None
    cross_names = None

    # Parallel I/O: load + process all assets concurrently (parquet reads are I/O-bound)
    with ThreadPoolExecutor(max_workers=8) as io_pool:
        futures = {io_pool.submit(_load_and_process_asset, sym, tf): sym
                   for sym in symbols}
        for future in as_completed(futures):
            sym = futures[future]
            try:
                result = future.result()
            except Exception as e:
                log(f"  {sym}: load error: {e}, skipping")
                continue

            if result is None:
                log(f"  {sym}: no data or <50 valid labels, skipping")
                continue

            if base_feature_names is None:
                base_feature_names = result['feat_cols']
            if result['cross_names'] is not None and cross_names is None:
                cross_names = result['cross_names']

            all_base_X.append(result['X_base_sparse'])
            all_sparse.append(result['sp_valid'])
            all_y.append(result['y_valid'])
            all_weights.append(result['w'])
            all_timestamps.append(result['ts'])
            all_asset_ids.append(result['asset_ids'])

            y_valid = result['y_valid']
            n_long = (y_valid == 2).sum()
            n_short = (y_valid == 0).sum()
            n_flat = (y_valid == 1).sum()
            log(f"  {sym}: {len(y_valid)} samples (L={n_long} F={n_flat} S={n_short})")

    if not all_base_X:
        return None, None, None, None, None, None

    # Stack base features (already sparse CSC per asset)
    log(f"  Stacking {len(all_base_X)} base sparse matrices...")
    X_base_all = sparse.vstack(all_base_X, format='csr')
    del all_base_X  # free individual sparse matrices
    gc.collect()

    y_all = np.concatenate(all_y)
    w_all = np.concatenate(all_weights)
    ts_all = np.concatenate(all_timestamps)
    assets_all = np.concatenate(all_asset_ids)

    # Stack sparse crosses
    if all(sp is not None for sp in all_sparse):
        log(f"  Stacking {len(all_sparse)} cross sparse matrices...")
        X_sparse_all = sparse.vstack(all_sparse, format='csr')
    else:
        X_sparse_all = None
    del all_sparse
    gc.collect()

    # Combine base (sparse) + crosses (sparse) into one sparse matrix
    if X_sparse_all is not None:
        X_combined = sparse.hstack([X_base_all, X_sparse_all], format='csr')
        del X_base_all, X_sparse_all
        gc.collect()
        all_feature_names = base_feature_names + (cross_names or [])
    else:
        X_combined = X_base_all
        del X_base_all
        gc.collect()
        all_feature_names = base_feature_names

    log(f"\nCombined: {X_combined.shape[0]:,} rows × {X_combined.shape[1]:,} features")
    log(f"  Sparse density: {X_combined.nnz / (X_combined.shape[0] * X_combined.shape[1]) * 100:.3f}%")
    log(f"  Memory: {X_combined.data.nbytes / 1e9:.2f} GB (non-zeros only)")
    log(f"  Labels: {(y_all==2).sum()} LONG, {(y_all==1).sum()} FLAT, {(y_all==0).sum()} SHORT")

    # Class balance correction — prevent FLAT domination
    class_counts = np.array([(y_all == c).sum() for c in range(3)], dtype=np.float64)
    class_weights = class_counts.sum() / (3.0 * class_counts)  # inverse frequency
    class_weights /= class_weights.min()  # normalize so smallest = 1.0
    log(f"  Class weights: SHORT={class_weights[0]:.2f}, FLAT={class_weights[1]:.2f}, LONG={class_weights[2]:.2f}")

    # Apply class weights to per-sample weights
    w_all *= class_weights[y_all]

    return X_combined, y_all, w_all, all_feature_names, ts_all, assets_all


# ============================================================
# SAMPLE UNIQUENESS (Lopez de Prado)
# ============================================================

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


def compute_sample_uniqueness(n_samples, max_hold_bars):
    """Compute average uniqueness per sample (Lopez de Prado method).

    For each event i with window [i, i + max_hold_bars], uniqueness = average of
    1/N_t across all bars t in the window, where N_t = number of concurrent
    events at bar t.

    Returns:
        uniqueness: array of shape (n_samples,) with values in (0, 1]
    """
    if n_samples == 0:
        return np.ones(0, dtype=np.float64)
    t0_arr = np.arange(n_samples, dtype=np.int64)
    t1_arr = np.minimum(t0_arr + max_hold_bars, n_samples - 1)
    starts = t0_arr
    ends = t1_arr + 1  # exclusive end
    return _compute_uniqueness_inner(starts, ends, n_samples)


# ============================================================
# CPCV SPLITS (from V1, adapted for multi-asset)
# ============================================================

def generate_cpcv_splits(n_samples, n_groups=6, n_test_groups=2,
                         max_hold_bars=24, embargo_pct=0.01):
    """Generate Combinatorial Purged Cross-Validation splits.

    Per-boundary purging: for EACH test group independently, purge training
    samples whose label windows could leak into that group. This correctly
    handles non-contiguous test groups (e.g., groups 1 and 4) by also purging
    training samples between them whose labels extend into a later test group.
    """
    from itertools import combinations

    group_size = n_samples // n_groups
    embargo_size = max(1, int(n_samples * embargo_pct))

    groups = []
    for g in range(n_groups):
        start = g * group_size
        end = start + group_size if g < n_groups - 1 else n_samples
        groups.append(np.arange(start, end))

    all_idx = np.arange(n_samples)

    splits = []
    for test_combo in combinations(range(n_groups), n_test_groups):
        test_idx = np.concatenate([groups[g] for g in test_combo])
        test_mask = np.zeros(n_samples, dtype=bool)
        test_mask[test_idx] = True

        # Per-boundary purge: for EACH test group, compute purge + embargo zones
        # This correctly handles non-contiguous test groups (e.g., groups 1 and 4)
        # by purging training samples BETWEEN groups whose labels leak forward
        purge_mask = np.zeros(n_samples, dtype=bool)
        for g in test_combo:
            g_start = groups[g][0]
            g_end = groups[g][-1]
            # Purge before this group: training samples whose label window
            # [i, i+max_hold_bars] could extend into this test group
            purge_mask[max(0, g_start - max_hold_bars):g_start] = True
            # Embargo after this group: prevent information leakage from
            # serial correlation immediately after test boundary
            purge_mask[g_end + 1:min(n_samples, g_end + 1 + embargo_size)] = True

        train_idx = all_idx[~test_mask & ~purge_mask]

        if len(train_idx) > 50 and len(test_idx) > 20:
            splits.append((train_idx, test_idx))

    return splits


# ============================================================
# PARALLEL SPLIT WORKERS (multi-GPU)
# ============================================================

def _xgb_split_worker(args):
    """
    Train a single XGBoost CPCV split on a specific GPU.
    Runs in a subprocess — all GPUs visible, device selected via params.
    Returns: (split_idx, acc, loss, best_iteration, model_bytes, preds, y_test, test_idx)
    """
    (wi, train_idx, test_idx, X_data, X_indices, X_indptr, X_shape,
     y, weights, feature_names, params, num_boost_round,
     early_stopping_rounds, gpu_id) = args

    import numpy as np
    from scipy import sparse
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, log_loss

    # Reconstruct sparse matrix in worker
    X_sparse = sparse.csr_matrix((X_data, X_indices, X_indptr), shape=X_shape)

    X_train = X_sparse[train_idx]
    y_train = y[train_idx]
    w_train = weights[train_idx]
    X_test = X_sparse[test_idx]
    y_test = y[test_idx]

    # Split training data into 85% train + 15% validation for early stopping
    # (never use test set for model selection — preserves OOS integrity)
    n_tr = X_train.shape[0]
    val_size = max(int(n_tr * 0.15), 100)
    if val_size >= n_tr:
        val_size = max(n_tr // 5, 20)
    X_val_es = X_train[-val_size:]
    y_val_es = y_train[-val_size:]
    w_train_es = w_train[:-val_size]
    X_train_es = X_train[:-val_size]
    y_train_es = y_train[:-val_size]

    params = params.copy()
    params['device'] = f'cuda:{gpu_id}'

    try:
        dtrain = xgb.DeviceQuantileDMatrix(X_train_es, label=y_train_es, weight=w_train_es,
                                            feature_names=feature_names,
                                            max_bin=params.get('max_bin', 32), nthread=-1)
        dval = xgb.DMatrix(X_val_es, label=y_val_es, feature_names=feature_names, nthread=-1)
    except Exception:
        dtrain = xgb.DMatrix(X_train_es, label=y_train_es, weight=w_train_es,
                             feature_names=feature_names, nthread=-1)
        dval = xgb.DMatrix(X_val_es, label=y_val_es, feature_names=feature_names, nthread=-1)

    model = xgb.train(
        params, dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=0,
    )

    # OOS prediction on the held-out test set (NOT used for early stopping)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names, nthread=-1)
    preds = model.predict(dtest)
    pred_labels = np.argmax(preds, axis=1)
    acc = accuracy_score(y_test, pred_labels)
    loss = log_loss(y_test, preds, labels=[0, 1, 2])

    # IS metrics for proper PBO (Lopez de Prado)
    is_preds = model.predict(dtrain)
    is_labels = np.argmax(is_preds, axis=1)
    is_acc = float(accuracy_score(y_train_es, is_labels))
    is_returns = np.where(is_labels == y_train_es, 1.0, -1.0)
    is_sharpe = float(np.mean(is_returns) / max(np.std(is_returns), 1e-8) * np.sqrt(252))

    # Serialize model to bytes so it can be returned across processes
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix='.ubj', delete=False)
    tmp.close()
    model.save_model(tmp.name)
    with open(tmp.name, 'rb') as f:
        model_bytes = f.read()
    os.unlink(tmp.name)

    return (wi, acc, loss, model.best_iteration, model_bytes, preds.copy(), y_test.copy(), test_idx.copy(), is_acc, is_sharpe)


def _lgbm_split_worker(args):
    """
    Train a single LightGBM CPCV split on a specific GPU.
    Runs in a subprocess — all GPUs visible, device selected via gpu_device_id.
    Returns: (split_idx, acc, loss, best_iteration, model_str, preds, y_test, test_idx)
    """
    (wi, train_idx, test_idx, X_data, X_indices, X_indptr, X_shape,
     y, weights, feature_names, params, num_boost_round,
     early_stopping_rounds, gpu_id) = args

    import numpy as np
    from scipy import sparse
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score, log_loss

    # Reconstruct sparse matrix in worker
    X_sparse = sparse.csr_matrix((X_data, X_indices, X_indptr), shape=X_shape)

    X_train = X_sparse[train_idx]
    y_train = y[train_idx]
    w_train = weights[train_idx]
    X_test = X_sparse[test_idx]
    y_test = y[test_idx]

    # Split training data into 85% train + 15% validation for early stopping
    # (never use test set for model selection — preserves OOS integrity)
    n_tr = X_train.shape[0]
    val_size = max(int(n_tr * 0.15), 100)
    if val_size >= n_tr:
        val_size = max(n_tr // 5, 20)
    X_val_es = X_train[-val_size:]
    y_val_es = y_train[-val_size:]
    w_train_es = w_train[:-val_size]
    X_train_es = X_train[:-val_size]
    y_train_es = y_train[:-val_size]

    params = params.copy()
    params['device'] = 'gpu'
    params['gpu_device_id'] = gpu_id

    dtrain = lgb.Dataset(X_train_es, label=y_train_es, weight=w_train_es,
                         feature_name=feature_names, free_raw_data=False)
    dval = lgb.Dataset(X_val_es, label=y_val_es,
                        feature_name=feature_names, reference=dtrain,
                        free_raw_data=False)

    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    model = lgb.train(
        params, dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dval],
        valid_names=['val'],
        callbacks=callbacks,
    )

    # OOS prediction on the held-out test set (NOT used for early stopping)
    preds = model.predict(X_test)
    pred_labels = np.argmax(preds, axis=1)
    acc = accuracy_score(y_test, pred_labels)
    loss = log_loss(y_test, preds, labels=[0, 1, 2])

    # IS metrics for proper PBO (Lopez de Prado)
    is_preds = model.predict(X_train_es)
    is_labels = np.argmax(is_preds, axis=1)
    is_acc = float(accuracy_score(y_train_es, is_labels))
    is_returns = np.where(is_labels == y_train_es, 1.0, -1.0)
    is_sharpe = float(np.mean(is_returns) / max(np.std(is_returns), 1e-8) * np.sqrt(252))

    # Serialize model to string
    model_str = model.model_to_string()

    return (wi, acc, loss, model.best_iteration, model_str, preds.copy(), y_test.copy(), test_idx.copy(), is_acc, is_sharpe)


# ============================================================
# TRAINING
# ============================================================

def train_model(X_sparse, y, weights, feature_names, params=None,
                n_groups=6, n_test_groups=2, max_hold_bars=24,
                num_boost_round=800, early_stopping_rounds=50,
                parallel_splits=False, tf_name=None, use_dask=False,
                mode=None):
    """
    Train XGBoost on sparse data with CPCV.
    If use_dask=True and HAS_DASK_CUDA and N_GPUS > 1 and mode in ('unified', 'all'),
    uses Dask-XGBoost for distributed multi-GPU training of a single model.
    If parallel_splits=True and N_GPUS > 1, trains splits across GPUs in parallel.
    Returns: (best_model, results_dict)
    """
    if params is None:
        params = V2_XGB_PARAMS.copy()
    if tf_name:
        params = _apply_tf_params(params, tf_name)
        log(f"  Per-TF params: min_child_weight={params['min_child_weight']} (tf={tf_name})")

    # ── V2_BATCH_SIZE env override (set by cloud runner OOM retry) ──
    batch_size = int(os.environ.get('V2_BATCH_SIZE', 0))
    if batch_size > 0:
        ratio = min(1.0, batch_size / max(1, X_sparse.shape[0]))
        params['subsample'] = ratio
        num_boost_round = min(num_boost_round, batch_size)
        log(f"  V2_BATCH_SIZE={batch_size}: subsample={ratio:.4f}, num_boost_round={num_boost_round}")

    n = X_sparse.shape[0]

    # Sample uniqueness weighting (Lopez de Prado)
    uniqueness = compute_sample_uniqueness(n, max_hold_bars)
    weights = weights.copy().astype(np.float64)
    weights *= uniqueness
    # Normalize so sum = n_samples (preserves scale)
    w_sum = weights.sum()
    if w_sum > 0:
        weights *= n / w_sum
    weights = weights.astype(np.float32)
    log(f"  Sample uniqueness: min={uniqueness.min():.3f} max={uniqueness.max():.3f} mean={uniqueness.mean():.3f}")

    # ── Dask-XGBoost distributed path (all GPUs for single model) ──
    # Only for unified/all modes — per-asset already parallelizes folds across GPUs
    dask_eligible = (use_dask and HAS_DASK_CUDA and N_GPUS > 1
                     and mode in ('unified', 'all'))

    # Safety check: Dask requires dense arrays. Sparse CSR → dense can OOM
    # for large feature matrices (200K rows × 4M features = 3.2 TB dense)
    if dask_eligible and sparse.issparse(X_sparse):
        dense_gb = (X_sparse.shape[0] * X_sparse.shape[1] * 4) / (1024**3)
        total_ram = _hw.get('total_ram_gb', 256)
        if dense_gb > total_ram * 0.5:
            log(f"  WARNING: Dask requires dense array ({dense_gb:.0f} GB) > 50% RAM ({total_ram:.0f} GB)")
            log(f"  Falling back to --parallel-splits (CPCV folds across GPUs)")
            dask_eligible = False

    if dask_eligible:
        log(f"Training with Dask-XGBoost across {N_GPUS} GPUs")
        cluster = LocalCUDACluster(n_workers=N_GPUS)
        client = Client(cluster)
        try:
            # Convert sparse to dense for Dask (Dask arrays don't support scipy sparse)
            # Chunk the conversion to avoid peak memory spike from one massive .toarray()
            chunk_rows = max(1, X_sparse.shape[0] // N_GPUS)
            if sparse.issparse(X_sparse):
                n_rows, n_cols = X_sparse.shape
                dense_chunks = []
                for start in range(0, n_rows, chunk_rows):
                    end = min(start + chunk_rows, n_rows)
                    dense_chunks.append(X_sparse[start:end].toarray().astype(np.float32))
                X_dense = np.concatenate(dense_chunks, axis=0)
                del dense_chunks
            else:
                X_dense = np.asarray(X_sparse, dtype=np.float32)
            X_dask = da.from_array(X_dense, chunks=(chunk_rows, X_dense.shape[1]))
            y_dask = da.from_array(y.astype(np.float32), chunks=chunk_rows)
            w_dask = da.from_array(weights, chunks=chunk_rows)

            dtrain = dxgb.DaskDMatrix(client, X_dask, y_dask, weight=w_dask,
                                      feature_names=feature_names)

            # Use same params but remove device (Dask handles GPU assignment)
            dask_params = {k: v for k, v in params.items() if k != 'device'}
            dask_params['tree_method'] = 'hist'

            # Split 15% for early stopping eval set
            val_n = max(int(n * 0.15), 100)
            X_val_da = da.from_array(X_dense[-val_n:], chunks=(val_n, X_dense.shape[1]))
            y_val_da = da.from_array(y[-val_n:].astype(np.float32), chunks=val_n)
            dval = dxgb.DaskDMatrix(client, X_val_da, y_val_da,
                                    feature_names=feature_names)

            output = dxgb.train(client, dask_params, dtrain,
                                num_boost_round=num_boost_round,
                                early_stopping_rounds=early_stopping_rounds,
                                evals=[(dval, 'val')])

            booster = output['booster']

            # Predict OOS on full dataset for downstream validation
            dpred = dxgb.DaskDMatrix(client, X_dask, feature_names=feature_names)
            preds_dask = dxgb.predict(client, booster, dpred)
            preds = preds_dask.compute()

            pred_labels = np.argmax(preds, axis=1)
            acc = accuracy_score(y, pred_labels)
            loss = log_loss(y, preds, labels=[0, 1, 2])

            log(f"  Dask-XGBoost full-data: acc={acc:.4f} loss={loss:.4f} "
                f"rounds={output.get('best_iteration', num_boost_round)}")

            # Save model
            model_path = os.path.join(V2_DIR, f'model_v2_{mode}_{tf_name}.json')
            booster.save_model(model_path)
            log(f"  Dask-XGBoost model saved: {model_path}")

            results = {
                'engine': 'xgboost',
                'mean_acc': acc, 'std_acc': 0.0,
                'mean_loss': loss,
                'all_accs': [acc], 'all_losses': [loss],
                'n_features': X_sparse.shape[1],
                'n_samples': n,
                'n_paths': 1,
                'parallel_gpus': N_GPUS,
                'dask_distributed': True,
                'oos_predictions': [{
                    'path_idx': 0,
                    'path': 0,
                    'y_true': y.copy(),
                    'y_pred_probs': preds.copy(),
                    'test_indices': np.arange(n),
                    'is_accuracy': acc,
                    'is_sharpe': 0.0,
                }],
            }

            del X_dense, X_dask, y_dask, w_dask
            gc.collect()

            return booster, results

        finally:
            client.close()
            cluster.close()

    splits = generate_cpcv_splits(n, n_groups, n_test_groups, max_hold_bars)
    log(f"CPCV: {len(splits)} paths ({n_groups} groups, {n_test_groups} test)")

    use_parallel = parallel_splits and USE_GPU and N_GPUS > 1

    if use_parallel:
        # ── Multi-GPU parallel path ──
        log(f"  PARALLEL: distributing {len(splits)} splits across {N_GPUS} GPUs")

        # Extract CSR components for pickling (scipy sparse is picklable but explicit is safer)
        X_csr = X_sparse.tocsr()
        worker_args = []
        for wi, (train_idx, test_idx) in enumerate(splits):
            gpu_id = wi % N_GPUS
            worker_args.append((
                wi, train_idx, test_idx,
                X_csr.data, X_csr.indices, X_csr.indptr, X_csr.shape,
                y, weights, feature_names, params,
                num_boost_round, early_stopping_rounds, gpu_id
            ))

        all_accs = [0.0] * len(splits)
        all_losses = [0.0] * len(splits)
        best_model = None
        best_acc = 0
        oos_predictions = []

        with ProcessPoolExecutor(max_workers=N_GPUS) as executor:
            for result in executor.map(_xgb_split_worker, worker_args):
                wi, acc, loss, best_iter, model_bytes, preds, y_test, test_idx, is_acc, is_sharpe = result
                all_accs[wi] = acc
                all_losses[wi] = loss

                oos_predictions.append({
                    'path_idx': wi,
                    'path': wi,
                    'y_true': y_test.copy(),
                    'y_pred_probs': preds.copy(),
                    'test_indices': test_idx.copy(),
                    'is_accuracy': is_acc,
                    'is_sharpe': is_sharpe,
                })

                if acc > best_acc:
                    best_acc = acc
                    # Deserialize model
                    import tempfile
                    tmp = tempfile.NamedTemporaryFile(suffix='.ubj', delete=False)
                    tmp.write(model_bytes)
                    tmp.close()
                    best_model = xgb.Booster()
                    best_model.load_model(tmp.name)
                    os.unlink(tmp.name)

                log(f"  Path {wi+1}/{len(splits)} (GPU {wi % N_GPUS}): "
                    f"acc={acc:.4f} loss={loss:.4f} rounds={best_iter}")

    else:
        # ── Sequential path (single GPU or CPU) ──
        best_model = None
        best_acc = 0
        all_accs = []
        all_losses = []
        oos_predictions = []

        for wi, (train_idx, test_idx) in enumerate(splits):
            X_train = X_sparse[train_idx]
            y_train = y[train_idx]
            w_train = weights[train_idx]
            X_test = X_sparse[test_idx]
            y_test = y[test_idx]

            # Split training data into 85% train + 15% validation for early stopping
            # (never use test set for model selection — preserves OOS integrity)
            n_tr = X_train.shape[0]
            val_size = max(int(n_tr * 0.15), 100)
            if val_size >= n_tr:
                val_size = max(n_tr // 5, 20)
            X_val_es = X_train[-val_size:]
            y_val_es = y_train[-val_size:]
            w_train_es = w_train[:-val_size]
            X_train_es = X_train[:-val_size]
            y_train_es = y_train[:-val_size]

            if USE_GPU:
                try:
                    dtrain = xgb.DeviceQuantileDMatrix(X_train_es, label=y_train_es, weight=w_train_es,
                                                        feature_names=feature_names,
                                                        max_bin=params.get('max_bin', 32), nthread=-1)
                    dval = xgb.DMatrix(X_val_es, label=y_val_es, feature_names=feature_names, nthread=-1)
                except Exception:
                    dtrain = xgb.DMatrix(X_train_es, label=y_train_es, weight=w_train_es,
                                         feature_names=feature_names, nthread=-1)
                    dval = xgb.DMatrix(X_val_es, label=y_val_es, feature_names=feature_names, nthread=-1)
            else:
                dtrain = xgb.DMatrix(X_train_es, label=y_train_es, weight=w_train_es,
                                     feature_names=feature_names, nthread=-1)
                dval = xgb.DMatrix(X_val_es, label=y_val_es, feature_names=feature_names, nthread=-1)

            model = xgb.train(
                params, dtrain,
                num_boost_round=num_boost_round,
                evals=[(dtrain, 'train'), (dval, 'val')],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=0,
            )

            # OOS prediction on the held-out test set (NOT used for early stopping)
            dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names, nthread=-1)
            preds = model.predict(dtest)
            pred_labels = np.argmax(preds, axis=1)
            acc = accuracy_score(y_test, pred_labels)
            loss = log_loss(y_test, preds, labels=[0, 1, 2])

            # IS metrics for proper PBO (Lopez de Prado requires IS vs OOS rank comparison)
            is_preds = model.predict(dtrain)
            is_labels = np.argmax(is_preds, axis=1)
            is_acc = accuracy_score(y_train_es, is_labels)
            is_returns = np.where(is_labels == y_train_es, 1.0, -1.0)
            is_sharpe = float(np.mean(is_returns) / max(np.std(is_returns), 1e-8) * np.sqrt(252))

            oos_predictions.append({
                'path_idx': wi,
                'path': wi,
                'y_true': y_test.copy(),
                'y_pred_probs': preds.copy(),
                'test_indices': test_idx.copy(),
                'is_accuracy': is_acc,
                'is_sharpe': is_sharpe,
            })

            all_accs.append(acc)
            all_losses.append(loss)

            if acc > best_acc:
                best_acc = acc
                best_model = model

            log(f"  Path {wi+1}/{len(splits)}: acc={acc:.4f} loss={loss:.4f} "
                f"rounds={model.best_iteration}")

    mean_acc = np.mean(all_accs)
    std_acc = np.std(all_accs)
    mean_loss = np.mean(all_losses)

    log(f"\nCPCV Results: acc={mean_acc:.4f}±{std_acc:.4f} loss={mean_loss:.4f}")

    results = {
        'mean_acc': mean_acc, 'std_acc': std_acc,
        'mean_loss': mean_loss,
        'all_accs': all_accs, 'all_losses': all_losses,
        'n_features': X_sparse.shape[1],
        'n_samples': X_sparse.shape[0],
        'n_paths': len(splits),
        'parallel_gpus': N_GPUS if use_parallel else 1,
        'oos_predictions': oos_predictions,
    }

    return best_model, results


# ============================================================
# LIGHTGBM TRAINING
# ============================================================

def train_model_lgbm(X_sparse, y, weights, feature_names, params=None,
                     n_groups=6, n_test_groups=2, max_hold_bars=24,
                     num_boost_round=800, early_stopping_rounds=50,
                     parallel_splits=False, tf_name=None):
    """
    Train LightGBM on sparse data with CPCV.
    LightGBM's EFB (Exclusive Feature Bundling) is optimized for massive sparse binary features.
    If parallel_splits=True and N_GPUS > 1, trains splits across GPUs in parallel.
    Returns: (best_model, results_dict) or (None, None) if lightgbm not available.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        log("WARNING: lightgbm not installed — skipping LightGBM training")
        log("  Install with: pip install lightgbm --install-option=--gpu")
        return None, None

    if params is None:
        params = V2_LGBM_PARAMS.copy()
    if tf_name:
        mcw = TF_MIN_CHILD_WEIGHT.get(tf_name, 50)
        params['min_child_samples'] = mcw  # LightGBM equivalent of min_child_weight
        log(f"  Per-TF params (LightGBM): min_child_samples={mcw} (tf={tf_name})")

    n = X_sparse.shape[0]

    # Sample uniqueness weighting (Lopez de Prado)
    uniqueness = compute_sample_uniqueness(n, max_hold_bars)
    weights = weights.copy().astype(np.float64)
    weights *= uniqueness
    # Normalize so sum = n_samples (preserves scale)
    w_sum = weights.sum()
    if w_sum > 0:
        weights *= n / w_sum
    weights = weights.astype(np.float32)
    log(f"  Sample uniqueness: min={uniqueness.min():.3f} max={uniqueness.max():.3f} mean={uniqueness.mean():.3f}")

    splits = generate_cpcv_splits(n, n_groups, n_test_groups, max_hold_bars)
    log(f"CPCV (LightGBM): {len(splits)} paths ({n_groups} groups, {n_test_groups} test)")

    use_parallel = parallel_splits and USE_GPU and N_GPUS > 1

    if use_parallel:
        # ── Multi-GPU parallel path ──
        log(f"  PARALLEL: distributing {len(splits)} splits across {N_GPUS} GPUs")

        X_csr = X_sparse.tocsr()
        worker_args = []
        for wi, (train_idx, test_idx) in enumerate(splits):
            gpu_id = wi % N_GPUS
            worker_args.append((
                wi, train_idx, test_idx,
                X_csr.data, X_csr.indices, X_csr.indptr, X_csr.shape,
                y, weights, feature_names, params,
                num_boost_round, early_stopping_rounds, gpu_id
            ))

        all_accs = [0.0] * len(splits)
        all_losses = [0.0] * len(splits)
        best_model = None
        best_acc = 0
        oos_predictions = []

        with ProcessPoolExecutor(max_workers=N_GPUS) as executor:
            for result in executor.map(_lgbm_split_worker, worker_args):
                wi, acc, loss, best_iter, model_str, preds, y_test, test_idx, is_acc, is_sharpe = result
                all_accs[wi] = acc
                all_losses[wi] = loss

                oos_predictions.append({
                    'path_idx': wi,
                    'path': wi,
                    'y_true': y_test.copy(),
                    'y_pred_probs': preds.copy(),
                    'test_indices': test_idx.copy(),
                    'is_accuracy': is_acc,
                    'is_sharpe': is_sharpe,
                })

                if acc > best_acc:
                    best_acc = acc
                    # Deserialize LightGBM model from string
                    best_model = lgb.Booster(model_str=model_str)

                log(f"  Path {wi+1}/{len(splits)} (GPU {wi % N_GPUS}): "
                    f"acc={acc:.4f} loss={loss:.4f} rounds={best_iter}")

    else:
        # ── Sequential path (single GPU or CPU) ──
        best_model = None
        best_acc = 0
        all_accs = []
        all_losses = []
        oos_predictions = []

        for wi, (train_idx, test_idx) in enumerate(splits):
            X_train = X_sparse[train_idx]
            y_train = y[train_idx]
            w_train = weights[train_idx]
            X_test = X_sparse[test_idx]
            y_test = y[test_idx]

            # Split training data into 85% train + 15% validation for early stopping
            # (never use test set for model selection — preserves OOS integrity)
            n_tr = X_train.shape[0]
            val_size = max(int(n_tr * 0.15), 100)
            if val_size >= n_tr:
                val_size = max(n_tr // 5, 20)
            X_val_es = X_train[-val_size:]
            y_val_es = y_train[-val_size:]
            w_train_es = w_train[:-val_size]
            X_train_es = X_train[:-val_size]
            y_train_es = y_train[:-val_size]

            # LightGBM Dataset accepts scipy.sparse directly
            dtrain = lgb.Dataset(X_train_es, label=y_train_es, weight=w_train_es,
                                 feature_name=feature_names, free_raw_data=False)
            dval = lgb.Dataset(X_val_es, label=y_val_es,
                                feature_name=feature_names, reference=dtrain,
                                free_raw_data=False)

            callbacks = [
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ]

            model = lgb.train(
                params, dtrain,
                num_boost_round=num_boost_round,
                valid_sets=[dval],
                valid_names=['val'],
                callbacks=callbacks,
            )

            # OOS prediction on the held-out test set (NOT used for early stopping)
            preds = model.predict(X_test)
            pred_labels = np.argmax(preds, axis=1)
            acc = accuracy_score(y_test, pred_labels)
            loss = log_loss(y_test, preds, labels=[0, 1, 2])

            # IS metrics for proper PBO (Lopez de Prado)
            is_preds = model.predict(X_train_es)
            is_labels = np.argmax(is_preds, axis=1)
            is_acc = float(accuracy_score(y_train_es, is_labels))
            is_returns = np.where(is_labels == y_train_es, 1.0, -1.0)
            is_sharpe = float(np.mean(is_returns) / max(np.std(is_returns), 1e-8) * np.sqrt(252))

            oos_predictions.append({
                'path_idx': wi,
                'path': wi,
                'y_true': y_test.copy(),
                'y_pred_probs': preds.copy(),
                'test_indices': test_idx.copy(),
                'is_accuracy': is_acc,
                'is_sharpe': is_sharpe,
            })

            all_accs.append(acc)
            all_losses.append(loss)

            if acc > best_acc:
                best_acc = acc
                best_model = model

            log(f"  Path {wi+1}/{len(splits)}: acc={acc:.4f} loss={loss:.4f} "
                f"rounds={model.best_iteration}")

    mean_acc = np.mean(all_accs)
    std_acc = np.std(all_accs)
    mean_loss = np.mean(all_losses)

    log(f"\nCPCV Results (LightGBM): acc={mean_acc:.4f}+/-{std_acc:.4f} loss={mean_loss:.4f}")

    results = {
        'engine': 'lightgbm',
        'mean_acc': mean_acc, 'std_acc': std_acc,
        'mean_loss': mean_loss,
        'all_accs': all_accs, 'all_losses': all_losses,
        'n_features': X_sparse.shape[1],
        'n_samples': X_sparse.shape[0],
        'n_paths': len(splits),
        'parallel_gpus': N_GPUS if use_parallel else 1,
        'oos_predictions': oos_predictions,
    }

    return best_model, results


# ============================================================
# ENGINE DISPATCH
# ============================================================

def train_with_engine(engine, X, y, w, feat_names, boost_rounds,
                      parallel_splits=False, tf_name=None,
                      use_dask=False, mode=None):
    """
    Dispatch training to the selected engine(s).
    Returns: (best_model, best_results, winning_engine)
    """
    xgb_model, xgb_results = None, None
    lgbm_model, lgbm_results = None, None

    if engine in ('xgboost', 'both'):
        log(f"\n  >> XGBoost training...")
        xgb_model, xgb_results = train_model(
            X, y, w, feat_names,
            num_boost_round=boost_rounds,
            parallel_splits=parallel_splits,
            tf_name=tf_name,
            use_dask=use_dask,
            mode=mode,
        )
        if xgb_results:
            xgb_results['engine'] = 'xgboost'

    if engine in ('lightgbm', 'both'):
        log(f"\n  >> LightGBM training...")
        lgbm_model, lgbm_results = train_model_lgbm(
            X, y, w, feat_names,
            num_boost_round=boost_rounds,
            parallel_splits=parallel_splits,
            tf_name=tf_name,
        )

    # Pick winner
    if engine == 'both' and xgb_results and lgbm_results:
        log(f"\n  BENCHMARK: XGBoost acc={xgb_results['mean_acc']:.4f} vs "
            f"LightGBM acc={lgbm_results['mean_acc']:.4f}")
        if lgbm_results['mean_acc'] > xgb_results['mean_acc']:
            log(f"  WINNER: LightGBM (+{lgbm_results['mean_acc'] - xgb_results['mean_acc']:.4f})")
            return lgbm_model, lgbm_results, 'lightgbm'
        else:
            log(f"  WINNER: XGBoost (+{xgb_results['mean_acc'] - lgbm_results['mean_acc']:.4f})")
            return xgb_model, xgb_results, 'xgboost'
    elif xgb_model is not None:
        return xgb_model, xgb_results, 'xgboost'
    elif lgbm_model is not None:
        return lgbm_model, lgbm_results, 'lightgbm'
    else:
        return None, None, None


def _save_model(model, engine, path):
    """Save model in engine-appropriate format."""
    if engine == 'xgboost':
        model.save_model(path)
    elif engine == 'lightgbm':
        model.save_model(path)


# ============================================================
# PER-ASSET PARALLEL TRAINING WORKER
# ============================================================

def _train_single_asset_worker(args_tuple):
    """
    Train a single per-asset model in a subprocess.
    ALL GPUs visible — no CUDA_VISIBLE_DEVICES pinning.
    Returns: (symbol, status_str) where status_str is 'OK', 'SKIP', or 'FAIL:reason'.
    """
    symbol, tf, engine, boost_rounds, parallel_splits, resume, mode_prefix, gpu_id = args_tuple

    import os, sys, gc, json
    import numpy as np
    from scipy import sparse

    V2 = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, V2)

    try:
        # Check resume — skip if model already exists
        if resume:
            xgb_path = os.path.join(V2, f'model_v2_{mode_prefix}{symbol}_{tf}.json')
            lgb_path = os.path.join(V2, f'model_v2_{mode_prefix}{symbol}_{tf}.txt')
            if os.path.exists(xgb_path) or os.path.exists(lgb_path):
                print(f"  RESUME: model for {mode_prefix}{symbol} {tf} exists, skipping", flush=True)
                return (symbol, 'SKIP')

        # Load data
        X, y, w, feat_names, ts, assets = prepare_training_data([symbol], tf)
        if X is None:
            return (symbol, 'SKIP:no_data')

        print(f"\nTraining {mode_prefix}{symbol} ({X.shape[0]:,} x {X.shape[1]:,})...", flush=True)
        model, results, winner = train_with_engine(
            engine, X, y, w, feat_names, boost_rounds,
            parallel_splits=parallel_splits, tf_name=tf,
        )

        if model is not None:
            ext = '.json' if winner == 'xgboost' else '.txt'
            out = os.path.join(V2, f'model_v2_{mode_prefix}{symbol}_{tf}{ext}')
            _save_model(model, winner, out)
            print(f"  Saved ({winner}): {out}", flush=True)

            from atomic_io import atomic_save_json
            atomic_save_json(feat_names, os.path.join(V2, f'features_v2_{mode_prefix.rstrip("_") or "per-asset"}_{symbol}_{tf}.json'))

            if results and 'oos_predictions' in results:
                from atomic_io import atomic_save_pickle
                mode_tag = f'{mode_prefix.rstrip("_") or "per-asset"}_{symbol}'
                oos_path = os.path.join(V2, f'oos_predictions_{mode_tag}_{tf}.pkl')
                atomic_save_pickle(results['oos_predictions'], oos_path)
                print(f"  OOS predictions saved: {oos_path}", flush=True)

        del X, y, w, model, results
        gc.collect()
        return (symbol, 'OK')

    except Exception as e:
        print(f"  [FAILED] {mode_prefix}{symbol} {tf}: {e}", flush=True)
        return (symbol, f'FAIL:{e}')


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['per-asset', 'unified', 'production', 'all'],
                        default='all')
    parser.add_argument('--tf', nargs='+', default=['1d'])
    parser.add_argument('--boost-rounds', type=int, default=800)
    parser.add_argument('--engine', choices=['xgboost', 'lightgbm', 'both'],
                        default='xgboost',
                        help='Training engine: xgboost (default), lightgbm, or both (benchmark)')
    parser.add_argument('--parallel-splits', action='store_true', default=False,
                        help='(legacy, now auto-detected) Kept for backward compat with cloud runners')
    parser.add_argument('--no-parallel-splits', action='store_true', default=False,
                        help='Force sequential CPCV splits even with multiple GPUs')
    parser.add_argument('--use-dask', action='store_true', default=False,
                        help='Use Dask-XGBoost for distributed multi-GPU training (unified/all mode only)')
    parser.add_argument('--resume', action='store_true',
                        help='Skip models that already exist')
    args = parser.parse_args()

    # Auto-detect parallel splits: ON by default when multiple GPUs available
    # Use --no-parallel-splits to force sequential
    if args.no_parallel_splits:
        args.parallel_splits = False
        log("PARALLEL SPLITS: disabled (--no-parallel-splits)")
    elif USE_GPU and N_GPUS > 1:
        args.parallel_splits = True
        log(f"PARALLEL SPLITS: auto-enabled across {N_GPUS} GPUs (use --no-parallel-splits to disable)")
    else:
        args.parallel_splits = False
        if N_GPUS <= 1:
            log("PARALLEL SPLITS: off (single GPU detected)")

    # Dask-XGBoost validation
    if args.use_dask:
        if not HAS_DASK_CUDA:
            log("WARNING: --use-dask requested but dask-cuda/dask not installed, falling back to standard training")
            log("  Install with: pip install dask-cuda dask distributed")
            args.use_dask = False
        elif N_GPUS <= 1:
            log("WARNING: --use-dask requested but only 1 GPU detected, falling back to standard training")
            args.use_dask = False
        else:
            log(f"DASK-XGBOOST: enabled for distributed training across {N_GPUS} GPUs (unified/all mode only)")

    for tf in args.tf:
        log(f"\n{'='*70}")
        log(f"V2 TRAINING — {tf.upper()} — engine={args.engine}")
        log(f"{'='*70}")

        # ── Per-asset models (parallel across GPUs) ──
        if args.mode in ('per-asset', 'all'):
            train_workers = min(N_GPUS, 4) if N_GPUS > 1 else 1
            log(f"\n--- PER-ASSET MODELS ({tf}) — {train_workers} parallel workers ---")

            if train_workers > 1:
                # Parallel per-asset training
                worker_args = [
                    (symbol, tf, args.engine, args.boost_rounds,
                     False, args.resume, '', wi % N_GPUS)  # parallel_splits=False per asset (single model)
                    for wi, symbol in enumerate(ALL_TRAINING)
                ]
                with ProcessPoolExecutor(max_workers=train_workers) as pool:
                    for result in pool.map(_train_single_asset_worker, worker_args):
                        symbol_r, status_r = result
                        log(f"  {symbol_r}: {status_r}")
                gc.collect()
            else:
                # Sequential fallback (single GPU)
                for symbol in ALL_TRAINING:
                    if args.resume:
                        xgb_path = os.path.join(V2_DIR, f'model_v2_{symbol}_{tf}.json')
                        lgb_path = os.path.join(V2_DIR, f'model_v2_{symbol}_{tf}.txt')
                        if os.path.exists(xgb_path) or os.path.exists(lgb_path):
                            log(f"  RESUME: model for {symbol} {tf} exists, skipping")
                            continue

                    X, y, w, feat_names, ts, assets = prepare_training_data([symbol], tf)
                    if X is None:
                        continue

                    log(f"\nTraining {symbol} ({X.shape[0]:,} x {X.shape[1]:,})...")
                    model, results, winner = train_with_engine(
                        args.engine, X, y, w, feat_names, args.boost_rounds,
                        parallel_splits=args.parallel_splits, tf_name=tf,
                    )

                    if model is not None:
                        ext = '.json' if winner == 'xgboost' else '.txt'
                        out = os.path.join(V2_DIR, f'model_v2_{symbol}_{tf}{ext}')
                        _save_model(model, winner, out)
                        log(f"  Saved ({winner}): {out}")

                        from atomic_io import atomic_save_json
                        atomic_save_json(feat_names, os.path.join(V2_DIR, f'features_v2_per-asset_{symbol}_{tf}.json'))

                        if results and 'oos_predictions' in results:
                            from atomic_io import atomic_save_pickle
                            mode_tag = f'per-asset_{symbol}'
                            oos_path = os.path.join(V2_DIR, f'oos_predictions_{mode_tag}_{tf}.pkl')
                            atomic_save_pickle(results['oos_predictions'], oos_path)
                            log(f"  OOS predictions saved: {oos_path} ({len(results['oos_predictions'])} paths)")

        # ── Unified model (all assets combined) ──
        if args.mode in ('unified', 'all'):
            log(f"\n--- UNIFIED MODEL ({tf}) ---")

            skip_unified = False
            if args.resume:
                xgb_path = os.path.join(V2_DIR, f'model_v2_unified_{tf}.json')
                lgb_path = os.path.join(V2_DIR, f'model_v2_unified_{tf}.txt')
                if os.path.exists(xgb_path) or os.path.exists(lgb_path):
                    log(f"  RESUME: model for unified {tf} exists, skipping")
                    skip_unified = True

            if not skip_unified:
                X, y, w, feat_names, ts, assets = prepare_training_data(ALL_TRAINING, tf)
                if X is not None:
                    log(f"\nTraining unified ({X.shape[0]:,} × {X.shape[1]:,})...")
                    model, results, winner = train_with_engine(
                        args.engine, X, y, w, feat_names, args.boost_rounds,
                        parallel_splits=args.parallel_splits, tf_name=tf,
                        use_dask=args.use_dask, mode='unified',
                    )

                    if model is not None:
                        ext = '.json' if winner == 'xgboost' else '.txt'
                        out = os.path.join(V2_DIR, f'model_v2_unified_{tf}{ext}')
                        _save_model(model, winner, out)
                        log(f"  Saved ({winner}): {out}")

                        # Save feature names
                        from atomic_io import atomic_save_json
                        atomic_save_json(feat_names, os.path.join(V2_DIR, f'features_v2_unified_{tf}.json'))

                        # Save OOS predictions
                        if results and 'oos_predictions' in results:
                            from atomic_io import atomic_save_pickle
                            mode_tag = 'unified'
                            oos_path = os.path.join(V2_DIR, f'oos_predictions_{mode_tag}_{tf}.pkl')
                            atomic_save_pickle(results['oos_predictions'], oos_path)
                            log(f"  OOS predictions saved: {oos_path} ({len(results['oos_predictions'])} paths)")

                        # Feature importance report
                        if winner == 'xgboost':
                            imp = model.get_score(importance_type='gain')
                        else:
                            imp_vals = model.feature_importance(importance_type='gain')
                            imp_names = model.feature_name()
                            imp = dict(zip(imp_names, imp_vals.tolist()))

                        top_100 = sorted(imp.items(), key=lambda x: -x[1])[:100]
                        log(f"\n  Top 20 features by gain:")
                        for fname, gain in top_100[:20]:
                            log(f"    {fname}: {gain:.4f}")

                        with open(os.path.join(V2_DIR, f'importance_v2_unified_{tf}.json'), 'w') as f:
                            json.dump(dict(sorted(imp.items(), key=lambda x: -x[1])), f, indent=2)

        # ── Production models (per crypto, for live trading — parallel across GPUs) ──
        if args.mode in ('production', 'all'):
            train_workers_prod = min(N_GPUS, 4) if N_GPUS > 1 else 1
            log(f"\n--- PRODUCTION MODELS ({tf}) — {train_workers_prod} parallel workers ---")

            if train_workers_prod > 1:
                worker_args = [
                    (symbol, tf, args.engine, args.boost_rounds,
                     False, args.resume, 'prod_', wi % N_GPUS)
                    for wi, symbol in enumerate(TRAINING_CRYPTO)
                ]
                with ProcessPoolExecutor(max_workers=train_workers_prod) as pool:
                    for result in pool.map(_train_single_asset_worker, worker_args):
                        symbol_r, status_r = result
                        log(f"  {symbol_r}: {status_r}")
                gc.collect()
            else:
                for symbol in TRAINING_CRYPTO:
                    if args.resume:
                        xgb_path = os.path.join(V2_DIR, f'model_v2_prod_{symbol}_{tf}.json')
                        lgb_path = os.path.join(V2_DIR, f'model_v2_prod_{symbol}_{tf}.txt')
                        if os.path.exists(xgb_path) or os.path.exists(lgb_path):
                            log(f"  RESUME: model for {symbol} {tf} exists, skipping")
                            continue

                    X, y, w, feat_names, ts, assets = prepare_training_data([symbol], tf)
                    if X is None:
                        continue

                    log(f"\nTraining {symbol} production ({X.shape[0]:,} x {X.shape[1]:,})...")
                    model, results, winner = train_with_engine(
                        args.engine, X, y, w, feat_names, args.boost_rounds,
                        parallel_splits=args.parallel_splits, tf_name=tf,
                    )

                    if model is not None:
                        ext = '.json' if winner == 'xgboost' else '.txt'
                        out = os.path.join(V2_DIR, f'model_v2_prod_{symbol}_{tf}{ext}')
                        _save_model(model, winner, out)
                        log(f"  Saved ({winner}): {out}")

                        from atomic_io import atomic_save_json
                        atomic_save_json(feat_names, os.path.join(V2_DIR, f'features_v2_production_{symbol}_{tf}.json'))

                        if results and 'oos_predictions' in results:
                            from atomic_io import atomic_save_pickle
                            mode_tag = f'prod_{symbol}'
                            oos_path = os.path.join(V2_DIR, f'oos_predictions_{mode_tag}_{tf}.pkl')
                            atomic_save_pickle(results['oos_predictions'], oos_path)
                            log(f"  OOS predictions saved: {oos_path} ({len(results['oos_predictions'])} paths)")

        # ── Auto-validation and meta-labeling after training ──
        oos_files = glob.glob(os.path.join(V2_DIR, f'oos_predictions_*_{tf}.pkl'))
        for oos_file in oos_files:
            mode_tag = os.path.basename(oos_file).replace('oos_predictions_', '').replace(f'_{tf}.pkl', '')
            try:
                from backtest_validation import validation_report
                oos_data = pickle.load(open(oos_file, 'rb'))
                report = validation_report(oos_data, tf_name=tf)
                report_path = os.path.join(V2_DIR, f'validation_report_{mode_tag}_{tf}.json')
                from atomic_io import atomic_save_json
                atomic_save_json(report, report_path)
                log(f"  Validation [{mode_tag}]: PBO={report['pbo'].get('pbo', 'N/A'):.3f}, rec={report.get('overall', 'N/A')}")

                # Auto meta-labeling if not rejected
                if report.get('overall') != 'REJECT':
                    try:
                        from meta_labeling import train_meta_model
                        meta_result = train_meta_model(oos_data, tf_name=f'{mode_tag}_{tf}', db_dir=V2_DIR)
                        if meta_result:
                            log(f"  Meta-model [{mode_tag}]: AUC={meta_result['metrics']['auc']:.3f}")
                    except Exception as e:
                        log(f"  Meta-labeling failed: {e}")
            except Exception as e:
                log(f"  Validation failed for {oos_file}: {e}")

    log(f"\n{'='*70}")
    log(f"V2 TRAINING COMPLETE")
    log(f"{'='*70}")


if __name__ == '__main__':
    main()
