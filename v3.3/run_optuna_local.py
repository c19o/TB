#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_optuna_local.py -- Two-Stage Optuna LightGBM Hyperparameter Search (v3.1)
===============================================================================
Stage 1: Coarse search with row subsample + fewer CPCV groups + pruning
Stage 2: Refined search around best region, full CPCV + full data
Final:   Retrain best config with full CPCV + full rounds + original LR

Usage:
    python run_optuna_local.py                    # all TFs
    python run_optuna_local.py --tf 1d            # single TF
    python run_optuna_local.py --tf 1d --tf 4h    # multiple TFs
    python run_optuna_local.py --stage 1          # only stage 1
    python run_optuna_local.py --n-jobs 4         # parallel trials
"""
import os
import sys
import json
import time
import pickle
import argparse
import logging
import multiprocessing
import warnings

try:
    from hardware_detect import get_cpu_count, get_available_ram_gb
except ImportError:
    def get_cpu_count():
        try:
            return len(os.sched_getaffinity(0))
        except AttributeError:
            return os.cpu_count() or 1
    def get_available_ram_gb():
        try:
            import psutil
            return psutil.virtual_memory().available / (1024**3)
        except ImportError:
            return 64.0  # safe default

import numpy as np
import pandas as pd
from scipy import sparse as sp_sparse

warnings.filterwarnings('ignore')

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('SAVAGE22_DB_DIR', PROJECT_DIR)
os.environ.setdefault('SKIP_LLM', '1')

import optuna
from optuna.pruners import HyperbandPruner, MedianPruner
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, log_loss

from config import (
    V3_LGBM_PARAMS, TF_MIN_DATA_IN_LEAF, V30_DATA_DIR,
    OPTUNA_STAGE1_TRIALS, OPTUNA_STAGE2_TRIALS,
    OPTUNA_N_STARTUP_TRIALS, OPTUNA_SEED,
    OPTUNA_PRUNER, OPTUNA_PRUNER_MIN_RESOURCE, OPTUNA_PRUNER_REDUCTION_FACTOR,
    OPTUNA_SEARCH_LR, OPTUNA_SEARCH_ROUNDS,
    OPTUNA_FINAL_LR, OPTUNA_FINAL_ROUNDS,
    OPTUNA_SEARCH_CPCV_GROUPS, OPTUNA_SEARCH_ROW_SUBSAMPLE,
    OPTUNA_TF_ROW_SUBSAMPLE,
    OPTUNA_N_JOBS, TF_CPCV_GROUPS,
)
from feature_library import compute_triple_barrier_labels, TRIPLE_BARRIER_CONFIG

# Suppress Optuna info spam — only show warnings and above
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(PROJECT_DIR, 'optuna_search.log'), mode='a'),
    ],
)
log = logging.getLogger(__name__)

DB_DIR = os.environ.get('SAVAGE22_DB_DIR', PROJECT_DIR)
TF_ORDER = ['1w', '1d', '4h', '1h', '15m']

# CPCV groups per TF — imported from config.TF_CPCV_GROUPS (single source of truth)


# ============================================================
# CPCV SPLIT GENERATOR (copied from ml_multi_tf.py for independence)
# ============================================================
from itertools import combinations
from numba import njit


@njit(cache=True)
def _compute_uniqueness_inner(starts, ends, n_bars):
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
    starts = np.asarray(t0_arr, dtype=np.int64)
    ends = np.asarray(t1_arr, dtype=np.int64)
    return _compute_uniqueness_inner(starts, ends, n_bars)


def _generate_cpcv_splits(n_samples, n_groups=6, n_test_groups=2,
                           max_hold_bars=None, embargo_pct=0.01):
    group_size = n_samples // n_groups
    groups = []
    for g in range(n_groups):
        start = g * group_size
        end = (g + 1) * group_size if g < n_groups - 1 else n_samples
        groups.append(np.arange(start, end))

    embargo_size = max(1, int(n_samples * embargo_pct))
    all_paths = list(combinations(range(n_groups), n_test_groups))

    splits = []
    for test_group_ids in all_paths:
        test_idx = np.concatenate([groups[g] for g in test_group_ids])
        train_group_ids = [g for g in range(n_groups) if g not in test_group_ids]
        train_idx = np.concatenate([groups[g] for g in train_group_ids])

        # Embargo: remove training samples near test boundaries
        if embargo_size > 0:
            test_min = test_idx.min()
            test_max = test_idx.max()
            embargo_mask = np.ones(len(train_idx), dtype=bool)
            for ti in range(len(train_idx)):
                idx = train_idx[ti]
                if abs(idx - test_min) < embargo_size or abs(idx - test_max) < embargo_size:
                    embargo_mask[ti] = False
            train_idx = train_idx[embargo_mask]

        splits.append((train_idx, test_idx))

    return splits


# ============================================================
# DATA LOADING
# ============================================================
def load_tf_data(tf_name):
    """Load features + crosses + labels for a timeframe. Returns (X_all, y, sample_weights, feature_cols, is_sparse)."""
    # Find parquet
    parquet_path = os.path.join(DB_DIR, f'features_{tf_name}.parquet')
    v2_parquet = os.path.join(DB_DIR, f'features_BTC_{tf_name}.parquet')
    v30_parquet = os.path.join(V30_DATA_DIR, f'features_BTC_{tf_name}.parquet')

    for p in [parquet_path, v2_parquet, v30_parquet]:
        if os.path.exists(p):
            parquet_path = p
            break
    else:
        raise FileNotFoundError(f"No feature parquet found for {tf_name}")

    df = pd.read_parquet(parquet_path)
    # Downcast float64 -> float32
    f64 = df.select_dtypes(include=['float64']).columns
    if len(f64) > 0:
        df[f64] = df[f64].astype(np.float32)
    log.info(f"  Loaded {parquet_path}: {len(df)} rows x {len(df.columns)} cols")

    # Parse timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    elif 'date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['date'])

    # Triple-barrier labels
    if 'triple_barrier_label' in df.columns:
        y = pd.to_numeric(df['triple_barrier_label'], errors='coerce').values
    else:
        y = compute_triple_barrier_labels(df, tf_name)

    # Feature columns
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

    # Load sparse crosses
    cross_matrix = None
    cross_cols = None
    npz_path = os.path.join(DB_DIR, f'v2_crosses_BTC_{tf_name}.npz')
    if not os.path.exists(npz_path):
        npz_v30 = os.path.join(V30_DATA_DIR, f'v2_crosses_BTC_{tf_name}.npz')
        if os.path.exists(npz_v30):
            npz_path = npz_v30

    if os.path.exists(npz_path):
        try:
            cross_matrix = sp_sparse.load_npz(npz_path).tocsr()
            # Enforce correct CSR dtypes: indptr=int64, indices=int32 (LightGBM PR #1719)
            INT32_MAX = 2_147_483_647
            if cross_matrix.indptr.dtype != np.int64:
                cross_matrix.indptr = cross_matrix.indptr.astype(np.int64)
            if cross_matrix.indices.dtype != np.int32:
                assert cross_matrix.nnz == 0 or cross_matrix.indices.max() <= INT32_MAX, (
                    f"FATAL: cross_matrix column index > int32 max")
                cross_matrix.indices = cross_matrix.indices.astype(np.int32)
            # Load column names
            _basename = os.path.basename(npz_path)
            _parts = _basename.replace('v2_crosses_', '').replace('.npz', '').rsplit('_', 1)
            _sym, _tfn = (_parts[0], _parts[1]) if len(_parts) == 2 else ('BTC', tf_name)
            for cols_dir in [DB_DIR, V30_DATA_DIR]:
                cols_path = os.path.join(cols_dir, f'v2_cross_names_{_sym}_{_tfn}.json')
                if os.path.exists(cols_path):
                    with open(cols_path) as f:
                        cross_cols = json.load(f)
                    break
            if cross_cols is None:
                cross_cols = [f'cross_{i}' for i in range(cross_matrix.shape[1])]
            log.info(f"  Sparse crosses: {cross_matrix.shape[1]:,} cols, {cross_matrix.nnz:,} nnz")
        except Exception as e:
            log.warning(f"  Failed to load crosses: {e}")
            cross_matrix = None

    # Combine base + crosses
    is_sparse = False
    if cross_matrix is not None and cross_matrix.shape[0] == X_base.shape[0]:
        X_base_sparse = sp_sparse.csr_matrix(X_base)
        X_all = sp_sparse.hstack([X_base_sparse, cross_matrix], format='csr')
        # Enforce correct CSR dtypes after hstack
        if X_all.indptr.dtype != np.int64:
            X_all.indptr = X_all.indptr.astype(np.int64)
        if X_all.indices.dtype != np.int32:
            assert X_all.nnz == 0 or X_all.indices.max() <= INT32_MAX, (
                f"FATAL: X_all column index > int32 max")
            X_all.indices = X_all.indices.astype(np.int32)
        feature_cols = feature_cols + cross_cols
        # RAM check before dense conversion — avoid OOM
        try:
            import psutil
            _dense_bytes = X_all.shape[0] * X_all.shape[1] * 4  # float32 (data is downcast)
            _avail_ram = psutil.virtual_memory().available
            if _dense_bytes < _avail_ram * 0.7:
                log.info(f"  Converting sparse→dense for multi-core Optuna ({_dense_bytes/1e9:.1f} GB, RAM avail: {_avail_ram/1e9:.0f} GB)...")
                X_all = X_all.toarray()
                is_sparse = False
            else:
                log.warning(f"  Dense would need {_dense_bytes/1e9:.1f}GB, only {_avail_ram/1e9:.0f}GB avail — keeping sparse")
                is_sparse = True
        except ImportError:
            log.info(f"  Converting sparse→dense for multi-core Optuna...")
            X_all = X_all.toarray()
            is_sparse = False
        log.info(f"  Combined: {X_all.shape[1]:,} features ({n_base} base + {len(cross_cols):,} crosses) [{'SPARSE' if is_sparse else 'DENSE'}]")
        del X_base_sparse, cross_matrix
    else:
        X_all = X_base
    del X_base

    # Sample weights: regime + esoteric + uniqueness
    sample_weights = np.ones(len(y), dtype=np.float32)

    # MC-1 FIX: No regime weighting — model decides via HMM state feature, not pre-judged weights.
    # Counter-trend esoteric signals need full weight. HMM state is an input feature column.

    # Uniqueness weights
    tb_cfg = TRIPLE_BARRIER_CONFIG.get(tf_name, TRIPLE_BARRIER_CONFIG['1h'])
    max_hold = tb_cfg.get('max_hold_bars', 24)
    n = len(df)
    t0 = np.arange(n)
    t1 = np.minimum(t0 + max_hold, n - 1)
    uniqueness = _compute_sample_uniqueness(t0, t1, n)
    sample_weights *= uniqueness.astype(np.float32)

    # Normalize
    valid_mask = ~np.isnan(y)
    sw_sum = sample_weights[valid_mask].sum()
    if sw_sum > 0:
        sample_weights *= valid_mask.sum() / sw_sum

    return X_all, y, sample_weights, feature_cols, is_sparse, max_hold


# ============================================================
# OBJECTIVE FUNCTION BUILDER
# ============================================================
def build_objective(X_all, y, sample_weights, feature_cols, is_sparse, tf_name,
                    max_hold, n_groups, n_test_groups, row_subsample=1.0,
                    search_lr=None, search_rounds=None, stage=1,
                    narrow_ranges=None):
    """Build an Optuna objective function for LightGBM hyperparameter search.

    Args:
        narrow_ranges: dict of param -> (low, high) for stage 2 narrowing
    """

    def objective(trial):
        # ── Suggest hyperparameters ──
        # Per-TF ranges from config
        from config import TF_MIN_DATA_IN_LEAF, TF_NUM_LEAVES
        _tf_mdil = TF_MIN_DATA_IN_LEAF.get(tf_name, 3)
        _tf_nl_cap = TF_NUM_LEAVES.get(tf_name, 63)

        if narrow_ranges:
            # Stage 2: narrow ranges around best region
            nr = narrow_ranges
            num_leaves = trial.suggest_int('num_leaves', nr['num_leaves'][0], nr['num_leaves'][1])
            min_data_in_leaf = trial.suggest_int('min_data_in_leaf', nr['min_data_in_leaf'][0], nr['min_data_in_leaf'][1])
            feature_fraction = trial.suggest_float('feature_fraction', nr['feature_fraction'][0], nr['feature_fraction'][1], log=True)
            feature_fraction_bynode = trial.suggest_float('feature_fraction_bynode', nr['feature_fraction_bynode'][0], nr['feature_fraction_bynode'][1])
            bagging_fraction = trial.suggest_float('bagging_fraction', nr['bagging_fraction'][0], nr['bagging_fraction'][1])
            lambda_l1 = trial.suggest_float('lambda_l1', nr['lambda_l1'][0], nr['lambda_l1'][1], log=True)
            lambda_l2 = trial.suggest_float('lambda_l2', nr['lambda_l2'][0], nr['lambda_l2'][1], log=True)
            min_gain_to_split = trial.suggest_float('min_gain_to_split', nr['min_gain_to_split'][0], nr['min_gain_to_split'][1])
            max_depth = trial.suggest_int('max_depth', nr.get('max_depth', (4, 12))[0], nr.get('max_depth', (4, 12))[1])
            learning_rate = trial.suggest_float('learning_rate', nr.get('learning_rate', (0.01, 0.1))[0], nr.get('learning_rate', (0.01, 0.1))[1], log=True)
        else:
            # Stage 1: Perplexity-validated wide ranges
            num_leaves = trial.suggest_int('num_leaves', 15, _tf_nl_cap)
            min_data_in_leaf = trial.suggest_int('min_data_in_leaf', max(1, _tf_mdil - 2), _tf_mdil + 10)
            feature_fraction = trial.suggest_float('feature_fraction', 0.005, 0.1, log=True)  # log-scaled (3.16)
            feature_fraction_bynode = trial.suggest_float('feature_fraction_bynode', 0.2, 0.8)
            bagging_fraction = trial.suggest_float('bagging_fraction', 0.5, 0.95)
            lambda_l1 = trial.suggest_float('lambda_l1', 0.01, 1.0, log=True)  # capped at 1.0 (was 10.0)
            lambda_l2 = trial.suggest_float('lambda_l2', 0.1, 20.0, log=True)  # v3.2 best was 13.58
            min_gain_to_split = trial.suggest_float('min_gain_to_split', 0.1, 5.0)  # lowered floor from 0.5
            max_depth = trial.suggest_int('max_depth', 4, 12)  # new (3.9)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)  # new (3.10)
        max_bin = 255  # LOCKED — controls EFB bundle size (254/bundle). Binary features still get 2 bins.

        # Build LightGBM params — start from V3_LGBM_PARAMS baseline (fix 3.11)
        lr = search_lr if search_lr else OPTUNA_FINAL_LR
        rounds = search_rounds if search_rounds else OPTUNA_FINAL_ROUNDS

        params = V3_LGBM_PARAMS.copy()
        params.update({
            'is_enable_sparse': is_sparse,
            'num_threads': max(1, get_cpu_count() // max(1, int(os.environ.get('OPTUNA_N_JOBS', '1')))),
            'num_leaves': num_leaves,
            'min_data_in_leaf': min_data_in_leaf,
            'feature_fraction': feature_fraction,
            'feature_fraction_bynode': feature_fraction_bynode,
            'bagging_fraction': bagging_fraction,
            'bagging_freq': 1,
            'lambda_l1': lambda_l1,
            'lambda_l2': lambda_l2,
            'min_gain_to_split': min_gain_to_split,
            'max_bin': max_bin,
            'max_depth': max_depth,
            'learning_rate': learning_rate if not search_lr else lr,
            'seed': OPTUNA_SEED,
        })
        # ── Row subsample for stage 1 ──
        if row_subsample < 1.0:
            np.random.seed(OPTUNA_SEED)
            valid_mask = ~np.isnan(y)
            valid_indices = np.where(valid_mask)[0]
            n_sample = int(len(valid_indices) * row_subsample)
            subsample_idx = np.sort(np.random.choice(valid_indices, size=n_sample, replace=False))
        else:
            valid_mask = ~np.isnan(y)
            subsample_idx = np.where(valid_mask)[0]

        # ── Generate CPCV splits on subsampled data ──
        n_sub = len(subsample_idx)
        splits = _generate_cpcv_splits(
            n_sub, n_groups=n_groups, n_test_groups=n_test_groups,
            max_hold_bars=max_hold, embargo_pct=0.01,
        )

        # ── Evaluate across CPCV folds ──
        fold_scores = []
        fold_sortinos = []

        for fold_i, (train_rel, test_rel) in enumerate(splits):
            train_idx = subsample_idx[train_rel]
            test_idx = subsample_idx[test_rel]

            y_train = y[train_idx].astype(int)
            y_test = y[test_idx].astype(int)
            w_train = sample_weights[train_idx]

            if is_sparse:
                X_train = X_all[train_idx]
                X_test = X_all[test_idx]
            else:
                X_train = X_all[train_idx]
                X_test = X_all[test_idx]

            if len(y_train) < 50 or len(y_test) < 20:
                continue

            # 85/15 train/val for early stopping
            val_size = max(int(len(y_train) * 0.15), 50)
            if val_size >= len(y_train):
                val_size = max(len(y_train) // 5, 20)

            X_val_es = X_train[-val_size:]
            y_val_es = y_train[-val_size:]
            w_val_es = w_train[-val_size:]
            X_train_es = X_train[:-val_size]
            y_train_es = y_train[:-val_size]
            w_train_es = w_train[:-val_size]

            dtrain = lgb.Dataset(X_train_es, label=y_train_es, weight=w_train_es,
                                 free_raw_data=True)
            dval = lgb.Dataset(X_val_es, label=y_val_es, weight=w_val_es,
                               reference=dtrain, free_raw_data=True)

            # Use pruning callback for stage 1
            # MC-4: scale early stopping inversely with LR — esoteric signals need 800+ trees at low LR
            _es_patience = max(50, int(100 * (0.1 / params.get('learning_rate', 0.03))))
            callbacks = [lgb.early_stopping(_es_patience), lgb.log_evaluation(0)]

            model = lgb.train(
                params, dtrain,
                num_boost_round=rounds,
                valid_sets=[dval],
                valid_names=['val'],
                callbacks=callbacks,
            )

            # OOS predictions
            preds = model.predict(X_test)
            pred_labels = np.argmax(preds, axis=1)
            acc = accuracy_score(y_test, pred_labels)
            mlogloss = log_loss(y_test, preds, labels=[0, 1, 2])

            # Sortino-style metric: reward correct directional trades, penalize wrong
            sim_ret = np.where(pred_labels == y_test, 1.0, -1.0)
            downside = sim_ret[sim_ret < 0]
            downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1.0
            sortino = np.mean(sim_ret) / max(downside_std, 1e-10) * np.sqrt(252)

            fold_scores.append(mlogloss)
            fold_sortinos.append(sortino)

            # Report intermediate value for pruning (after each fold)
            trial.report(np.mean(fold_scores), fold_i)
            if trial.should_prune():
                raise optuna.TrialPruned()

            del model, dtrain, dval, X_train, X_test
            import gc
            gc.collect()

        if not fold_scores:
            return float('inf')  # minimize mlogloss

        # Primary objective: mean OOS multi_logloss (lower = better)
        mean_mlogloss = np.mean(fold_scores)
        mean_sortino = np.mean(fold_sortinos)

        # Store sortino as user attr for analysis
        trial.set_user_attr('mean_sortino', float(mean_sortino))
        trial.set_user_attr('n_folds', len(fold_scores))
        trial.set_user_attr('mean_mlogloss', float(mean_mlogloss))

        return mean_mlogloss

    return objective


# ============================================================
# NARROW RANGES FROM TOP TRIALS
# ============================================================
def compute_narrow_ranges(study, top_k=5):
    """Compute narrowed search ranges from top-K trials of stage 1."""
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value)
    top_trials = completed[:top_k]

    if not top_trials:
        return None

    ranges = {}
    param_names = ['num_leaves', 'min_data_in_leaf', 'feature_fraction',
                   'feature_fraction_bynode', 'bagging_fraction',
                   'lambda_l1', 'lambda_l2', 'min_gain_to_split']

    for pname in param_names:
        values = [t.params[pname] for t in top_trials if pname in t.params]
        if not values:
            continue
        lo = min(values)
        hi = max(values)
        # Expand by 20% on each side
        margin = max((hi - lo) * 0.2, abs(lo) * 0.1)
        ranges[pname] = (max(lo - margin, 0.001), hi + margin)

    # max_bin: LOCKED at 255 — controls EFB bundle size, not searchable
    ranges['max_bin'] = [255]

    # Clamp integer ranges
    if 'num_leaves' in ranges:
        ranges['num_leaves'] = (max(7, int(ranges['num_leaves'][0])), min(255, int(ranges['num_leaves'][1])))
    if 'min_data_in_leaf' in ranges:
        ranges['min_data_in_leaf'] = (max(1, int(ranges['min_data_in_leaf'][0])), max(2, int(ranges['min_data_in_leaf'][1])))

    return ranges


# ============================================================
# FINAL RETRAINING WITH BEST PARAMS
# ============================================================
def final_retrain(X_all, y, sample_weights, feature_cols, is_sparse,
                  tf_name, max_hold, best_params):
    """Retrain with best params using full CPCV + full rounds + final LR."""
    log.info(f"  FINAL RETRAIN: full CPCV, lr={OPTUNA_FINAL_LR}, rounds={OPTUNA_FINAL_ROUNDS}")

    n_groups, n_test_groups = TF_CPCV_GROUPS.get(tf_name, (4, 1))
    valid_mask = ~np.isnan(y)
    valid_indices = np.where(valid_mask)[0]
    n_valid = len(valid_indices)

    splits = _generate_cpcv_splits(
        n_valid, n_groups=n_groups, n_test_groups=n_test_groups,
        max_hold_bars=max_hold, embargo_pct=0.01,
    )

    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'device': 'cpu',
        'force_col_wise': True,
        'is_enable_sparse': is_sparse,  # match actual data format
        'feature_pre_filter': False,   # CRITICAL: never filter rare features
        'verbosity': -1,
        'num_threads': 0,  # auto-detect via OpenMP
        'learning_rate': OPTUNA_FINAL_LR,
        'seed': OPTUNA_SEED,
        'bagging_freq': 1,
        'min_data_in_bin': 1,  # allow bins with 1 sample (rare signals)
        'path_smooth': 0.1,
    }
    # Apply best Optuna params
    for k in ['num_leaves', 'min_data_in_leaf', 'feature_fraction',
              'feature_fraction_bynode', 'bagging_fraction',
              'lambda_l1', 'lambda_l2', 'min_gain_to_split', 'max_bin',
              'max_depth']:
        if k in best_params:
            params[k] = best_params[k]

    oos_predictions = []
    fold_accs = []
    fold_sortinos = []
    best_model_obj = None
    best_acc = 0

    for fold_i, (train_rel, test_rel) in enumerate(splits):
        train_idx = valid_indices[train_rel]
        test_idx = valid_indices[test_rel]

        y_train = y[train_idx].astype(int)
        y_test = y[test_idx].astype(int)
        w_train = sample_weights[train_idx]

        X_train = X_all[train_idx]
        X_test = X_all[test_idx]

        if len(y_train) < 50 or len(y_test) < 20:
            continue

        val_size = max(int(len(y_train) * 0.15), 50)
        if val_size >= len(y_train):
            val_size = max(len(y_train) // 5, 20)

        dtrain = lgb.Dataset(X_train[:-val_size], label=y_train[:-val_size],
                             weight=w_train[:-val_size], feature_name=feature_cols,
                             free_raw_data=False)
        dval = lgb.Dataset(X_train[-val_size:], label=y_train[-val_size:],
                           weight=w_train[-val_size:],
                           feature_name=feature_cols, free_raw_data=False)

        model = lgb.train(
            params, dtrain,
            num_boost_round=OPTUNA_FINAL_ROUNDS,
            valid_sets=[dtrain, dval],
            valid_names=['train', 'val'],
            callbacks=[lgb.early_stopping(max(50, int(100 * (0.1 / params.get('learning_rate', OPTUNA_FINAL_LR))))), lgb.log_evaluation(0)],
        )

        preds = model.predict(X_test)
        pred_labels = np.argmax(preds, axis=1)
        acc = accuracy_score(y_test, pred_labels)
        prec_long = precision_score(y_test, pred_labels, labels=[2], average='macro', zero_division=0)
        prec_short = precision_score(y_test, pred_labels, labels=[0], average='macro', zero_division=0)

        sim_ret = np.where(pred_labels == y_test, 1.0, -1.0)
        downside = sim_ret[sim_ret < 0]
        downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1.0
        sortino = np.mean(sim_ret) / max(downside_std, 1e-10) * np.sqrt(252)

        fold_accs.append(acc)
        fold_sortinos.append(sortino)

        oos_predictions.append({
            'path': fold_i,
            'test_indices': test_idx.tolist(),
            'y_true': y_test.tolist(),
            'y_pred_probs': preds.tolist(),
            'y_pred_labels': pred_labels.tolist(),
        })

        log.info(f"    Fold {fold_i+1}/{len(splits)}: Acc={acc:.3f} PrecL={prec_long:.3f} PrecS={prec_short:.3f} Sortino={sortino:.2f} Trees={model.best_iteration}")

        if acc > best_acc:
            best_acc = acc
            best_model_obj = model

        del dtrain, dval
        import gc
        gc.collect()

    return {
        'best_model': best_model_obj,
        'oos_predictions': oos_predictions,
        'mean_accuracy': float(np.mean(fold_accs)) if fold_accs else 0,
        'mean_sortino': float(np.mean(fold_sortinos)) if fold_sortinos else 0,
        'n_folds': len(fold_accs),
    }


# ============================================================
# MAIN SEARCH FOR ONE TF
# ============================================================
def run_search_for_tf(tf_name, max_stage=2, n_jobs=1):
    """Run two-stage Optuna search + final retrain for one timeframe."""
    log.info(f"\n{'='*70}")
    log.info(f"OPTUNA SEARCH: {tf_name.upper()}")
    log.info(f"{'='*70}")

    total_start = time.time()

    # Load data
    X_all, y, sample_weights, feature_cols, is_sparse, max_hold = load_tf_data(tf_name)
    n_valid = (~np.isnan(y)).sum()
    log.info(f"  Features: {len(feature_cols):,} ({'SPARSE' if is_sparse else 'DENSE'})")
    log.info(f"  Valid samples: {int(n_valid):,} / {len(y):,}")

    # Create pruner
    if OPTUNA_PRUNER == 'hyperband':
        pruner = HyperbandPruner(
            min_resource=OPTUNA_PRUNER_MIN_RESOURCE,
            reduction_factor=OPTUNA_PRUNER_REDUCTION_FACTOR,
        )
    else:
        pruner = MedianPruner(
            n_startup_trials=OPTUNA_N_STARTUP_TRIALS,
            n_warmup_steps=2,
        )

    # Sampler with fixed seed for reproducibility
    sampler = optuna.samplers.TPESampler(
        seed=OPTUNA_SEED,
        n_startup_trials=OPTUNA_N_STARTUP_TRIALS,
        multivariate=True,
        group=True,
    )

    study_name = f'lgbm_{tf_name}_v31'
    storage_path = os.path.join(PROJECT_DIR, f'optuna_{tf_name}.db')
    storage = f'sqlite:///{storage_path}'

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction='minimize',  # minimize mlogloss
        pruner=pruner,
        sampler=sampler,
        load_if_exists=True,
    )

    # ── STAGE 1: Coarse search with subsample + fewer CPCV ──
    log.info(f"\n  STAGE 1: {OPTUNA_STAGE1_TRIALS} trials, {OPTUNA_SEARCH_CPCV_GROUPS} CPCV groups, "
             f"{OPTUNA_SEARCH_ROW_SUBSAMPLE:.0%} row subsample, lr={OPTUNA_SEARCH_LR}")
    stage1_start = time.time()

    # Store n_jobs in env for thread calculation inside objective
    os.environ['OPTUNA_N_JOBS'] = str(n_jobs)

    objective_s1 = build_objective(
        X_all, y, sample_weights, feature_cols, is_sparse, tf_name,
        max_hold,
        n_groups=OPTUNA_SEARCH_CPCV_GROUPS,
        n_test_groups=1,
        row_subsample=OPTUNA_TF_ROW_SUBSAMPLE.get(tf_name, OPTUNA_SEARCH_ROW_SUBSAMPLE),
        search_lr=OPTUNA_SEARCH_LR,
        search_rounds=OPTUNA_SEARCH_ROUNDS,
        stage=1,
    )

    # Count existing completed trials to determine remaining
    existing_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    stage1_remaining = max(0, OPTUNA_STAGE1_TRIALS - existing_completed)

    if stage1_remaining > 0:
        log.info(f"  Running {stage1_remaining} stage 1 trials ({existing_completed} already done)...")
        study.optimize(
            objective_s1,
            n_trials=stage1_remaining,
            n_jobs=n_jobs,
            show_progress_bar=True,
        )
    else:
        log.info(f"  Stage 1 already complete ({existing_completed} trials)")

    stage1_elapsed = time.time() - stage1_start
    best_s1 = study.best_trial
    log.info(f"  Stage 1 done in {stage1_elapsed:.0f}s: best mlogloss={best_s1.value:.4f}")
    log.info(f"  Best params: {best_s1.params}")
    if 'mean_sortino' in best_s1.user_attrs:
        log.info(f"  Best sortino: {best_s1.user_attrs['mean_sortino']:.2f}")

    if max_stage < 2:
        log.info("  Stopping after stage 1 (--stage 1)")
        return {
            'tf': tf_name,
            'stage1_best_value': best_s1.value,
            'stage1_best_params': best_s1.params,
            'stage1_time': stage1_elapsed,
        }

    # ── STAGE 2: Refined search around best region, full CPCV + full data ──
    narrow_ranges = compute_narrow_ranges(study, top_k=5)
    if narrow_ranges is None:
        log.warning("  No completed trials for stage 2 narrowing, skipping")
        return None

    n_groups_full, n_test_full = TF_CPCV_GROUPS.get(tf_name, (4, 1))
    log.info(f"\n  STAGE 2: {OPTUNA_STAGE2_TRIALS} trials, {n_groups_full} CPCV groups (full), "
             f"100% rows, lr={OPTUNA_SEARCH_LR}")
    log.info(f"  Narrowed ranges: {json.dumps({k: v for k, v in narrow_ranges.items() if k != 'max_bin'}, indent=4, default=str)}")

    stage2_start = time.time()

    # New study for stage 2 (or continue with same study and adjusted objective)
    study_s2 = optuna.create_study(
        study_name=f'{study_name}_s2',
        storage=storage,
        direction='minimize',
        pruner=pruner,
        sampler=optuna.samplers.TPESampler(
            seed=OPTUNA_SEED + 1,
            n_startup_trials=5,
            multivariate=True,
        ),
        load_if_exists=True,
    )

    # Enqueue best params from stage 1 as first trial
    study_s2.enqueue_trial(best_s1.params)

    objective_s2 = build_objective(
        X_all, y, sample_weights, feature_cols, is_sparse, tf_name,
        max_hold,
        n_groups=n_groups_full,
        n_test_groups=n_test_full,
        row_subsample=1.0,
        search_lr=OPTUNA_SEARCH_LR,
        search_rounds=OPTUNA_SEARCH_ROUNDS,
        stage=2,
        narrow_ranges=narrow_ranges,
    )

    existing_s2 = len([t for t in study_s2.trials if t.state == optuna.trial.TrialState.COMPLETE])
    s2_remaining = max(0, OPTUNA_STAGE2_TRIALS - existing_s2)

    if s2_remaining > 0:
        log.info(f"  Running {s2_remaining} stage 2 trials ({existing_s2} already done)...")
        study_s2.optimize(
            objective_s2,
            n_trials=s2_remaining,
            n_jobs=n_jobs,
            show_progress_bar=True,
        )
    else:
        log.info(f"  Stage 2 already complete ({existing_s2} trials)")

    stage2_elapsed = time.time() - stage2_start
    best_s2 = study_s2.best_trial
    log.info(f"  Stage 2 done in {stage2_elapsed:.0f}s: best mlogloss={best_s2.value:.4f}")
    log.info(f"  Best params: {best_s2.params}")

    # Pick overall best between stage 1 and stage 2
    if best_s2.value < best_s1.value:
        best_overall = best_s2
        log.info(f"  Stage 2 improved: {best_s1.value:.4f} -> {best_s2.value:.4f}")
    else:
        best_overall = best_s1
        log.info(f"  Stage 1 was better: {best_s1.value:.4f} vs {best_s2.value:.4f}")

    # ── FINAL RETRAIN with best config ──
    log.info(f"\n  FINAL RETRAIN: best params with lr={OPTUNA_FINAL_LR}, rounds={OPTUNA_FINAL_ROUNDS}")
    final_start = time.time()

    final_result = final_retrain(
        X_all, y, sample_weights, feature_cols, is_sparse,
        tf_name, max_hold, best_overall.params,
    )
    final_elapsed = time.time() - final_start

    log.info(f"  Final retrain done in {final_elapsed:.0f}s: "
             f"mean_acc={final_result['mean_accuracy']:.4f} "
             f"mean_sortino={final_result['mean_sortino']:.2f}")

    # Save best model — IMPORTANT: save as optuna_model_{tf}.json to avoid
    # overwriting the production model (model_{tf}.json) from ml_multi_tf.py.
    # Both use LightGBM; live_trader.py loads model_{tf}.json with lgb.Booster.
    if final_result['best_model'] is not None:
        model_path = os.path.join(PROJECT_DIR, f'optuna_model_{tf_name}.json')
        final_result['best_model'].save_model(model_path)
        log.info(f"  LightGBM model saved: {model_path}")
        log.info(f"  NOTE: This is a LightGBM model for analysis only. "
                 f"Production model (model_{tf_name}.json) is LightGBM from ml_multi_tf.py.")

    # Save OOS predictions for meta-labeling
    oos_path = os.path.join(PROJECT_DIR, f'cpcv_oos_predictions_{tf_name}.pkl')
    with open(oos_path, 'wb') as f:
        pickle.dump(final_result['oos_predictions'], f)
    log.info(f"  OOS predictions saved: {oos_path}")

    # Save optimal config
    config_path = os.path.join(PROJECT_DIR, f'optuna_configs_{tf_name}.json')
    with open(config_path, 'w') as f:
        json.dump({
            'best_params': best_overall.params,
            'stage1_best_value': float(best_s1.value),
            'stage2_best_value': float(best_s2.value),
            'final_mean_accuracy': final_result['mean_accuracy'],
            'final_mean_sortino': final_result['mean_sortino'],
            'n_features': len(feature_cols),
            'n_valid_samples': int(n_valid),
            'stage1_time': stage1_elapsed,
            'stage2_time': stage2_elapsed,
            'final_time': final_elapsed,
        }, f, indent=2)
    log.info(f"  Config saved: {config_path}")

    total_elapsed = time.time() - total_start
    return {
        'tf': tf_name,
        'best_params': best_overall.params,
        'stage1_best_value': float(best_s1.value),
        'stage2_best_value': float(best_s2.value),
        'final_mean_accuracy': final_result['mean_accuracy'],
        'final_mean_sortino': final_result['mean_sortino'],
        'total_time': total_elapsed,
    }


# ============================================================
# CLI ENTRYPOINT
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Two-stage Optuna LightGBM hyperparameter search')
    parser.add_argument('--tf', type=str, nargs='*', default=None,
                        help='Timeframes to search (default: all)')
    parser.add_argument('--stage', type=int, default=2, choices=[1, 2],
                        help='Max stage to run (1=coarse only, 2=coarse+refine+final)')
    parser.add_argument('--n-jobs', type=int, default=None,
                        help='Parallel Optuna trials (default: cpu_count // threads_per_trial)')
    args = parser.parse_args()

    timeframes = args.tf if args.tf else TF_ORDER

    # Auto-detect parallel trials
    total_cores = get_cpu_count() or 24
    if args.n_jobs is not None:
        n_jobs = args.n_jobs
    else:
        # Use config constant if set (>0), otherwise auto-calculate
        n_jobs = OPTUNA_N_JOBS if OPTUNA_N_JOBS > 0 else max(1, total_cores // 8)

    log.info(f"Optuna LightGBM Search v3.1")
    log.info(f"  Cores: {total_cores}, Parallel trials: {n_jobs}")
    log.info(f"  Timeframes: {timeframes}")
    log.info(f"  Stage 1: {OPTUNA_STAGE1_TRIALS} trials, {OPTUNA_SEARCH_CPCV_GROUPS} CPCV groups, "
             f"{OPTUNA_SEARCH_ROW_SUBSAMPLE:.0%} subsample, lr={OPTUNA_SEARCH_LR}")
    log.info(f"  Stage 2: {OPTUNA_STAGE2_TRIALS} trials, full CPCV, 100% data, lr={OPTUNA_SEARCH_LR}")
    log.info(f"  Final: full CPCV, lr={OPTUNA_FINAL_LR}, rounds={OPTUNA_FINAL_ROUNDS}")

    all_results = {}
    total_start = time.time()

    for tf in timeframes:
        try:
            result = run_search_for_tf(tf, max_stage=args.stage, n_jobs=n_jobs)
            if result:
                all_results[tf] = result
        except Exception as e:
            log.error(f"FAILED {tf}: {e}", exc_info=True)

    # Save summary
    summary_path = os.path.join(PROJECT_DIR, 'optuna_search_results.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    total_elapsed = time.time() - total_start
    log.info(f"\n{'='*70}")
    log.info(f"ALL DONE in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    log.info(f"{'='*70}")
    for tf, r in all_results.items():
        log.info(f"  {tf}: s1={r.get('stage1_best_value', 'N/A'):.4f} "
                 f"s2={r.get('stage2_best_value', 'N/A'):.4f} "
                 f"acc={r.get('final_mean_accuracy', 0):.4f} "
                 f"sortino={r.get('final_mean_sortino', 0):.2f} "
                 f"({r.get('total_time', 0):.0f}s)")
    log.info(f"Results saved to {summary_path}")


if __name__ == '__main__':
    main()
