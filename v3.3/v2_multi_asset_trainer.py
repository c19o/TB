#!/usr/bin/env python
"""
v2_multi_asset_trainer.py — V2 Multi-Asset Training Pipeline (LightGBM)
========================================================================
Trains on ALL 31 assets with ALL features (base + sparse crosses).
NO FILTERING. LightGBM decides via tree splits.

Pipeline:
  1. Load base features (parquet) + sparse crosses (.npz) per asset
  2. Stack all assets into unified sparse training matrix
  3. Triple-barrier labels per asset
  4. CPCV walk-forward (purging + embargo)
  5. LightGBM EFB on sparse CSR — force_col_wise, max_bin=255 (EFB optimized), feature_fraction=0.9
  6. Per-asset models + unified model + per-crypto production models
  7. Meta-labeling + PBO validation

Speed optimizations (4-6M features):
  - max_bin=255 (max EFB bundle size 254/bundle, binary features still get 2 bins)
  - feature_fraction=0.9 (90% EFB bundles per tree — preserves rare esoteric signals)
  - feature_fraction_bynode=0.8
  - early_stopping_rounds=50
  - num_leaves=63 (leaf-wise growth)
  - force_col_wise=True (critical for 100K+ columns)
  - is_enable_sparse=True (native sparse CSR support)
  - num_threads=0 (auto-detect via OpenMP)

Usage:
  python v2_multi_asset_trainer.py --mode unified --tf 1d
  python v2_multi_asset_trainer.py --mode per-asset --tf 1d
  python v2_multi_asset_trainer.py --mode production --tf 1d 1h 4h
  python v2_multi_asset_trainer.py --mode all --tf 1d
  python v2_multi_asset_trainer.py --mode unified --tf 1d                    # auto-parallel if N_GPUS > 1
  V3_FORCE_SEQUENTIAL=1 python v2_multi_asset_trainer.py --mode unified --tf 1d  # force sequential
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

import lightgbm as lgb
from sklearn.metrics import accuracy_score, log_loss

from config import ALL_TRAINING, TRAINING_CRYPTO
from feature_library import compute_triple_barrier_labels, TRIPLE_BARRIER_CONFIG

START_TIME = time.time()

def elapsed():
    return f"[{time.time()-START_TIME:.0f}s]"

def log(msg):
    print(f"{elapsed()} {msg}", flush=True)


# ============================================================
# GPU DETECTION (via hardware_detect.py)
# ============================================================

try:
    from hardware_detect import detect_hardware
except ImportError:
    def detect_hardware():
        import multiprocessing
        return {'cpu_count': multiprocessing.cpu_count() or 1, 'ram_gb': 64.0, 'gpu_count': 0, 'n_gpus': 0}

USE_GPU = False
N_GPUS = 0
_hw = {}
try:
    _hw = detect_hardware()
    N_GPUS = _hw['n_gpus'] or 0
    if N_GPUS > 0:
        USE_GPU = True
except:
    pass
log(f"GPU: {'DETECTED' if USE_GPU else 'NONE'} ({N_GPUS} device{'s' if N_GPUS != 1 else ''}) — LightGBM runs on CPU (sparse not supported on GPU)")


# ============================================================
# LGBM PARAMS — OPTIMIZED FOR 4-6M SPARSE FEATURES
# ============================================================

# LightGBM params — single source of truth from config.py
from config import V3_LGBM_PARAMS as V2_LGBM_PARAMS, TF_MIN_DATA_IN_LEAF

def _apply_tf_params(params, tf_name):
    """Apply per-TF overrides to LightGBM params. Protects rare esoteric signals.

    NOTE: is_enable_sparse=True flows from V3_LGBM_PARAMS in config.py.
    Data stays sparse (CSR) throughout — no .toarray() conversion.
    If data were ever converted to dense, is_enable_sparse must be set to False.
    """
    params = params.copy()
    mdl = TF_MIN_DATA_IN_LEAF.get(tf_name, 3)
    params['min_data_in_leaf'] = mdl
    return params


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
                         max_hold_bars=None, embargo_pct=0.01):
    """Generate Combinatorial Purged Cross-Validation splits.

    Per-boundary purging: for EACH test group independently, purge training
    samples whose label windows could leak into that group. This correctly
    handles non-contiguous test groups (e.g., groups 1 and 4) by also purging
    training samples between them whose labels extend into a later test group.
    """
    from itertools import combinations

    group_size = n_samples // n_groups
    if max_hold_bars is not None:
        # Embargo must be >= max_hold_bars bars (prevents leakage from forward label horizon)
        effective_pct = max(embargo_pct, max_hold_bars / n_samples)
    else:
        effective_pct = embargo_pct
    embargo_size = max(1, int(n_samples * effective_pct))

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
            if max_hold_bars is not None:
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

def _lgbm_split_worker(args):
    """
    Train a single LightGBM CPCV split in a subprocess.
    Returns: (split_idx, acc, loss, best_iteration, model_str, preds, y_test, test_idx, is_acc, is_sharpe)
    """
    (wi, train_idx, test_idx, X_data, X_indices, X_indptr, X_shape,
     y, weights, feature_names, params, num_boost_round,
     early_stopping_rounds, worker_id) = args

    import numpy as np
    from scipy import sparse
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score, log_loss

    X_sparse = sparse.csr_matrix((X_data, X_indices, X_indptr), shape=X_shape)

    X_train = X_sparse[train_idx]
    y_train = y[train_idx]
    w_train = weights[train_idx]
    X_test = X_sparse[test_idx]
    y_test = y[test_idx]

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

    _ds_params = {'feature_pre_filter': False, 'max_bin': 7, 'min_data_in_bin': 1}
    dtrain = lgb.Dataset(X_train_es, label=y_train_es, weight=w_train_es,
                         feature_name=feature_names, free_raw_data=False, params=_ds_params)
    dval = lgb.Dataset(X_val_es, label=y_val_es,
                        feature_name=feature_names, reference=dtrain,
                        free_raw_data=False, params=_ds_params)

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

    preds = model.predict(X_test)
    pred_labels = np.argmax(preds, axis=1)
    acc = accuracy_score(y_test, pred_labels)
    loss = log_loss(y_test, preds, labels=[0, 1, 2])

    is_preds = model.predict(X_train_es)
    is_labels = np.argmax(is_preds, axis=1)
    is_acc = float(accuracy_score(y_train_es, is_labels))
    is_returns = np.where(is_labels == y_train_es, 1.0, -1.0)
    is_sharpe = float(np.mean(is_returns) / max(np.std(is_returns), 1e-8) * np.sqrt(252))

    model_str = model.model_to_string()

    return (wi, acc, loss, model.best_iteration, model_str, preds.copy(), y_test.copy(), test_idx.copy(), is_acc, is_sharpe)


# ============================================================
# TRAINING
# ============================================================

def train_model(X_sparse, y, weights, feature_names, params=None,
                n_groups=6, n_test_groups=2, max_hold_bars=None,
                num_boost_round=800, early_stopping_rounds=50,
                parallel_splits=False, tf_name=None, mode=None):
    """
    Train LightGBM on sparse data with CPCV.
    If parallel_splits=True and N_GPUS > 1, trains splits across CPU workers in parallel.
    Returns: (best_model, results_dict)
    """
    if params is None:
        params = V2_LGBM_PARAMS.copy()
    if tf_name:
        params = _apply_tf_params(params, tf_name)
        log(f"  Per-TF params: min_data_in_leaf={params['min_data_in_leaf']} (tf={tf_name})")

    if max_hold_bars is None:
        from feature_library import TRIPLE_BARRIER_CONFIG
        _tb = TRIPLE_BARRIER_CONFIG.get(tf_name or '1h', TRIPLE_BARRIER_CONFIG['1h'])
        max_hold_bars = _tb['max_hold_bars']
        log(f"  CPCV purge: max_hold_bars={max_hold_bars} (from TRIPLE_BARRIER_CONFIG[{tf_name or '1h'}])")

    batch_size = int(os.environ.get('V2_BATCH_SIZE', 0))
    if batch_size > 0:
        ratio = min(1.0, batch_size / max(1, X_sparse.shape[0]))
        params['bagging_fraction'] = ratio
        num_boost_round = min(num_boost_round, batch_size)
        log(f"  V2_BATCH_SIZE={batch_size}: bagging_fraction={ratio:.4f}, num_boost_round={num_boost_round}")

    cloud_threads = int(os.environ.get('V2_NUM_THREADS', 0))
    if cloud_threads > 0:
        params['num_threads'] = cloud_threads
        log(f"  V2_NUM_THREADS={cloud_threads}: overriding num_threads")

    n = X_sparse.shape[0]

    uniqueness = compute_sample_uniqueness(n, max_hold_bars)
    weights = weights.copy().astype(np.float64)
    weights *= uniqueness
    w_sum = weights.sum()
    if w_sum > 0:
        weights *= n / w_sum
    weights = weights.astype(np.float32)
    log(f"  Sample uniqueness: min={uniqueness.min():.3f} max={uniqueness.max():.3f} mean={uniqueness.mean():.3f}")

    splits = generate_cpcv_splits(n, n_groups, n_test_groups, max_hold_bars)
    log(f"CPCV: {len(splits)} paths ({n_groups} groups, {n_test_groups} test)")

    try:
        from hardware_detect import get_cpu_count
        _total_cores = get_cpu_count()
    except ImportError:
        _total_cores = os.cpu_count() or 24
    # Dynamic worker allocation: match workers to splits, cap at core count
    n_workers = max(1, min(len(splits), _total_cores)) if parallel_splits else 1
    n_workers = int(os.environ.get('V3_CPCV_WORKERS', n_workers))
    use_parallel = parallel_splits and n_workers > 1

    if use_parallel:
        _threads_per = max(1, _total_cores // n_workers)
        log(f"  PARALLEL: {len(splits)} splits, {n_workers} workers x {_threads_per} threads = {n_workers * _threads_per} total ({_total_cores} cores)")
        # Set num_threads per worker to avoid oversubscription (each worker gets fair share of cores)
        params['num_threads'] = _threads_per
        log(f"  num_threads per worker: {params['num_threads']}")

        X_csr = X_sparse.tocsr()
        worker_args = []
        for wi, (train_idx, test_idx) in enumerate(splits):
            worker_args.append((
                wi, train_idx, test_idx,
                X_csr.data, X_csr.indices, X_csr.indptr, X_csr.shape,
                y, weights, feature_names, params,
                num_boost_round, early_stopping_rounds, wi % n_workers
            ))

        all_accs = [0.0] * len(splits)
        all_losses = [0.0] * len(splits)
        best_model = None
        best_acc = 0
        oos_predictions = []

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
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
                    best_model = lgb.Booster(model_str=model_str)

                log(f"  Path {wi+1}/{len(splits)} (worker {wi % n_workers}): "
                    f"acc={acc:.4f} loss={loss:.4f} rounds={best_iter}")

    else:
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

            n_tr = X_train.shape[0]
            val_size = max(int(n_tr * 0.15), 100)
            if val_size >= n_tr:
                val_size = max(n_tr // 5, 20)
            X_val_es = X_train[-val_size:]
            y_val_es = y_train[-val_size:]
            w_train_es = w_train[:-val_size]
            X_train_es = X_train[:-val_size]
            y_train_es = y_train[:-val_size]

            _ds_params2 = {'feature_pre_filter': False, 'max_bin': 7, 'min_data_in_bin': 1}
            dtrain = lgb.Dataset(X_train_es, label=y_train_es, weight=w_train_es,
                                 feature_name=feature_names, free_raw_data=False, params=_ds_params2)
            dval = lgb.Dataset(X_val_es, label=y_val_es,
                                feature_name=feature_names, reference=dtrain,
                                free_raw_data=False, params=_ds_params2)

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

            preds = model.predict(X_test)
            pred_labels = np.argmax(preds, axis=1)
            acc = accuracy_score(y_test, pred_labels)
            loss = log_loss(y_test, preds, labels=[0, 1, 2])

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

    log(f"\nCPCV Results: acc={mean_acc:.4f}+/-{std_acc:.4f} loss={mean_loss:.4f}")

    results = {
        'engine': 'lightgbm',
        'mean_acc': mean_acc, 'std_acc': std_acc,
        'mean_loss': mean_loss,
        'all_accs': all_accs, 'all_losses': all_losses,
        'n_features': X_sparse.shape[1],
        'n_samples': X_sparse.shape[0],
        'n_paths': len(splits),
        'parallel_workers': n_workers if use_parallel else 1,
        'oos_predictions': oos_predictions,
    }

    return best_model, results


# ============================================================
# ENGINE DISPATCH
# ============================================================

def train_with_engine(engine, X, y, w, feat_names, boost_rounds,
                      parallel_splits=False, tf_name=None, mode=None):
    """
    Train LightGBM model.
    Returns: (best_model, best_results, 'lightgbm')
    """
    log(f"\n  >> LightGBM training...")
    model, results = train_model(
        X, y, w, feat_names,
        num_boost_round=boost_rounds,
        parallel_splits=parallel_splits,
        tf_name=tf_name,
        mode=mode,
    )

    if model is not None:
        return model, results, 'lightgbm'
    return None, None, None


def _save_model(model, engine, path):
    """Save LightGBM model."""
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
        if resume:
            lgb_path = os.path.join(V2, f'model_v2_{mode_prefix}{symbol}_{tf}.txt')
            if os.path.exists(lgb_path):
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
            out = os.path.join(V2, f'model_v2_{mode_prefix}{symbol}_{tf}.txt')
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
    parser.add_argument('--engine', choices=['lightgbm'], default='lightgbm',
                        help='Training engine (lightgbm only)')
    parser.add_argument('--parallel-splits', action='store_true', default=False,
                        help='(legacy, now auto-detected) Kept for backward compat with cloud runners')
    # --no-parallel-splits removed: use env V3_FORCE_SEQUENTIAL=1 instead
    parser.add_argument('--resume', action='store_true',
                        help='Skip models that already exist')
    args = parser.parse_args()

    try:
        from hardware_detect import get_cpu_count
        n_cpu_workers = max(1, get_cpu_count() // 4)
    except ImportError:
        n_cpu_workers = max(1, os.cpu_count() // 4)
    _force_seq = os.environ.get('V3_FORCE_SEQUENTIAL', '0') == '1'
    if _force_seq:
        args.parallel_splits = False
        log("PARALLEL SPLITS: disabled (V3_FORCE_SEQUENTIAL=1)")
    elif n_cpu_workers > 1:
        args.parallel_splits = True
        log(f"PARALLEL SPLITS: auto-enabled across {n_cpu_workers} CPU workers (set V3_FORCE_SEQUENTIAL=1 to disable)")
    else:
        args.parallel_splits = False
        log("PARALLEL SPLITS: off (insufficient CPU cores)")

    for tf in args.tf:
        log(f"\n{'='*70}")
        log(f"V2 TRAINING — {tf.upper()} — LightGBM")
        log(f"{'='*70}")

        if args.mode in ('per-asset', 'all'):
            train_workers = min(n_cpu_workers, 4)
            log(f"\n--- PER-ASSET MODELS ({tf}) — {train_workers} parallel workers ---")

            if train_workers > 1:
                worker_args = [
                    (symbol, tf, args.engine, args.boost_rounds,
                     False, args.resume, '', wi % train_workers)
                    for wi, symbol in enumerate(ALL_TRAINING)
                ]
                with ProcessPoolExecutor(max_workers=train_workers) as pool:
                    for result in pool.map(_train_single_asset_worker, worker_args):
                        symbol_r, status_r = result
                        log(f"  {symbol_r}: {status_r}")
                gc.collect()
            else:
                for symbol in ALL_TRAINING:
                    if args.resume:
                        lgb_path = os.path.join(V2_DIR, f'model_v2_{symbol}_{tf}.txt')
                        if os.path.exists(lgb_path):
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
                        out = os.path.join(V2_DIR, f'model_v2_{symbol}_{tf}.txt')
                        _save_model(model, winner, out)
                        log(f"  Saved: {out}")

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
                lgb_path = os.path.join(V2_DIR, f'model_v2_unified_{tf}.txt')
                if os.path.exists(lgb_path):
                    log(f"  RESUME: model for unified {tf} exists, skipping")
                    skip_unified = True

            if not skip_unified:
                X, y, w, feat_names, ts, assets = prepare_training_data(ALL_TRAINING, tf)
                if X is not None:
                    log(f"\nTraining unified ({X.shape[0]:,} x {X.shape[1]:,})...")
                    model, results, winner = train_with_engine(
                        args.engine, X, y, w, feat_names, args.boost_rounds,
                        parallel_splits=args.parallel_splits, tf_name=tf,
                        mode='unified',
                    )

                    if model is not None:
                        out = os.path.join(V2_DIR, f'model_v2_unified_{tf}.txt')
                        _save_model(model, winner, out)
                        log(f"  Saved: {out}")

                        from atomic_io import atomic_save_json
                        atomic_save_json(feat_names, os.path.join(V2_DIR, f'features_v2_unified_{tf}.json'))

                        if results and 'oos_predictions' in results:
                            from atomic_io import atomic_save_pickle
                            mode_tag = 'unified'
                            oos_path = os.path.join(V2_DIR, f'oos_predictions_{mode_tag}_{tf}.pkl')
                            atomic_save_pickle(results['oos_predictions'], oos_path)
                            log(f"  OOS predictions saved: {oos_path} ({len(results['oos_predictions'])} paths)")

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
            train_workers_prod = min(n_cpu_workers, 4)
            log(f"\n--- PRODUCTION MODELS ({tf}) — {train_workers_prod} parallel workers ---")

            if train_workers_prod > 1:
                worker_args = [
                    (symbol, tf, args.engine, args.boost_rounds,
                     False, args.resume, 'prod_', wi % train_workers_prod)
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
                        lgb_path = os.path.join(V2_DIR, f'model_v2_prod_{symbol}_{tf}.txt')
                        if os.path.exists(lgb_path):
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
                        out = os.path.join(V2_DIR, f'model_v2_prod_{symbol}_{tf}.txt')
                        _save_model(model, winner, out)
                        log(f"  Saved: {out}")

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
