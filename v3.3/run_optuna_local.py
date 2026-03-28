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

# ── GPU fork detection ──
_HAS_GPU_FORK = False
try:
    from gpu_histogram_fork.src.cloud_gpu_integration import (
        should_use_gpu, gpu_cloud_integration,
    )
    # Probe: does set_external_csr exist on Booster?
    if hasattr(lgb.Booster, 'set_external_csr'):
        _HAS_GPU_FORK = True
except ImportError:
    pass

from config import (
    V3_LGBM_PARAMS, TF_MIN_DATA_IN_LEAF, V30_DATA_DIR,
    OPTUNA_STAGE1_TRIALS, OPTUNA_STAGE2_TRIALS,
    OPTUNA_WARMSTART_STAGE1_TRIALS, OPTUNA_WARMSTART_STAGE2_TRIALS,
    OPTUNA_N_STARTUP_TRIALS, OPTUNA_SEED,
    OPTUNA_PRUNER, OPTUNA_PRUNER_MIN_RESOURCE, OPTUNA_PRUNER_REDUCTION_FACTOR,
    OPTUNA_SEARCH_LR, OPTUNA_SEARCH_ROUNDS, OPTUNA_SEARCH_ES_PATIENCE,
    OPTUNA_FINAL_LR, OPTUNA_FINAL_ROUNDS,
    OPTUNA_SEARCH_CPCV_GROUPS, OPTUNA_SEARCH_ROW_SUBSAMPLE,
    OPTUNA_TF_ROW_SUBSAMPLE,
    OPTUNA_N_JOBS, TF_CPCV_GROUPS,
    OPTUNA_TF_STAGE1_TRIALS, OPTUNA_TF_STAGE2_TRIALS, OPTUNA_TF_N_STARTUP_TRIALS,
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
                          t0_arr=None, t1_arr=None, max_hold_bars=None,
                          embargo_pct=0.01):
    """Generate Combinatorial Purged Cross-Validation splits.

    Args:
        n_samples: total number of samples
        n_groups: number of contiguous groups to split data into (default 6)
        n_test_groups: number of groups used as test in each path (default 2)
        t0_arr: event start indices (for purging). If None, uses sample index.
        t1_arr: event end indices (for purging). If None, t0 + max_hold_bars.
        max_hold_bars: maximum label horizon (for purging when t0/t1 not provided)
        embargo_pct: fraction of samples to embargo after each test boundary

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

    embargo_size = max(1, int(n_samples * embargo_pct))

    # Generate all combinatorial test paths
    all_paths = list(combinations(range(n_groups), n_test_groups))

    splits = []
    for test_group_ids in all_paths:
        # Test indices = union of selected groups
        test_idx = np.concatenate([groups[g] for g in test_group_ids])

        # Train indices = all other groups
        train_group_ids = [g for g in range(n_groups) if g not in test_group_ids]
        train_idx = np.concatenate([groups[g] for g in train_group_ids])

        # --- Purging ---
        # Remove training samples whose label window overlaps with any test sample
        if t0_arr is not None and t1_arr is not None:
            test_min = test_idx.min()
            test_max = test_idx.max()

            # For each training sample, check if its label window overlaps test range
            # Label window of sample i = [t0_arr[i], t1_arr[i]]
            # Purge if: t0_arr[train_i] <= test_max AND t1_arr[train_i] >= test_min
            train_t0 = t0_arr[train_idx]
            train_t1 = t1_arr[train_idx]
            # Purge: label window overlaps with any test sample's time range
            overlap = (train_t1 >= test_min) & (train_t0 <= test_max)
            train_idx = train_idx[~overlap]
        elif max_hold_bars is not None:
            # Simple purge: remove training samples within max_hold_bars of test boundaries
            test_set = set(test_idx)
            test_boundaries = []
            for g in test_group_ids:
                test_boundaries.append(groups[g][0])   # start of test group
                test_boundaries.append(groups[g][-1])   # end of test group

            purge_mask = np.zeros(len(train_idx), dtype=bool)
            for boundary in test_boundaries:
                purge_mask |= (np.abs(train_idx - boundary) <= max_hold_bars)
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
        # Check if GPU fork will be used — keep sparse for GPU
        if _HAS_GPU_FORK:
            log.info("  GPU fork detected — keeping data sparse for GPU histograms")
            is_sparse = True
            # Skip dense conversion entirely
        else:
            # CPU path: convert to dense for multi-core Optuna if RAM allows
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
# ROUND-LEVEL PRUNING CALLBACK (for CPU lgb.train path)
# ============================================================
class _RoundPruningCallback:
    """Reports val score to Optuna every `interval` rounds for round-level pruning."""
    def __init__(self, trial, fold_i, max_rounds, interval=10):
        self.trial = trial
        self.fold_i = fold_i
        self.max_rounds = max_rounds
        self.interval = interval

    def __call__(self, env):
        if (env.iteration + 1) % self.interval != 0:
            return
        for entry in env.evaluation_result_list:
            if entry[0] == 'val' and entry[1] == 'multi_logloss':
                score = entry[2]
                break
        else:
            return
        step = self.fold_i * self.max_rounds + (env.iteration + 1)
        self.trial.report(score, step=step)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


# ============================================================
# OBJECTIVE FUNCTION BUILDER
# ============================================================
def build_objective(X_all, y, sample_weights, feature_cols, is_sparse, tf_name,
                    max_hold, n_groups, n_test_groups, row_subsample=1.0,
                    search_lr=None, search_rounds=None, stage=1,
                    narrow_ranges=None, use_gpu=False, parent_ds=None):
    """Build an Optuna objective function for LightGBM hyperparameter search.

    Pre-computes CPCV fold data slicing ONCE (valid_mask, subsample, splits,
    X/y/w slices) -- identical across all trials. Eliminates ~0.5-2s of
    redundant slicing per trial.

    Args:
        narrow_ranges: dict of param -> (low, high) for stage 2 narrowing
        use_gpu: bool -- if True, use GPU fork (cuda_sparse + set_external_csr)
        parent_ds: lgb.Dataset -- pre-constructed parent Dataset for EFB reuse
                   (reference= shares EFB bins across all trials, eliminating
                   redundant bin construction)
    """

    # ── Pre-compute CPCV fold data ONCE (identical across all trials) ──
    valid_mask = ~np.isnan(y)
    subsample_idx = np.where(valid_mask)[0]
    if row_subsample < 1.0:
        np.random.seed(OPTUNA_SEED)
        n_sample = int(len(subsample_idx) * row_subsample)
        subsample_idx = np.sort(np.random.choice(subsample_idx, size=n_sample, replace=False))

    n_sub = len(subsample_idx)
    splits = _generate_cpcv_splits(
        n_sub, n_groups=n_groups, n_test_groups=n_test_groups,
        max_hold_bars=max_hold, embargo_pct=0.01,
    )

    # Pre-slice data for each fold -- avoids redundant indexing every trial
    fold_data = []
    for _fi, (train_rel, test_rel) in enumerate(splits):
        abs_train = subsample_idx[train_rel]
        abs_test = subsample_idx[test_rel]

        y_train = y[abs_train].astype(int)
        y_test = y[abs_test].astype(int)
        w_train = sample_weights[abs_train]

        X_train = X_all[abs_train]
        X_test = X_all[abs_test]

        if len(y_train) < 50 or len(y_test) < 20:
            continue

        # 85/15 train/val for early stopping
        val_size = max(int(len(y_train) * 0.15), 50)
        if val_size >= len(y_train):
            val_size = max(len(y_train) // 5, 20)

        X_train_es = X_train[:-val_size]
        X_val_es = X_train[-val_size:]
        y_train_es = y_train[:-val_size]
        y_val_es = y_train[-val_size:]
        w_train_es = w_train[:-val_size]
        w_val_es = w_train[-val_size:]

        fold_data.append((X_train_es, X_val_es, y_train_es, y_val_es, w_train_es,
                          w_val_es, X_test, y_test))

    log.info(f"  Pre-computed {len(fold_data)} CPCV folds (from {len(splits)} splits)")

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
        max_bin = 255  # LOCKED — controls EFB bundle size (254/bundle). Binary features still get 2 bins.

        # Build LightGBM params — start from V3_LGBM_PARAMS baseline (fix 3.11)
        lr = search_lr if search_lr else OPTUNA_FINAL_LR
        rounds = search_rounds if search_rounds else OPTUNA_FINAL_ROUNDS

        params = V3_LGBM_PARAMS.copy()
        params.update({
            'is_enable_sparse': is_sparse,
            'num_threads': max(1, (os.cpu_count() or 4) // max(1, int(os.environ.get('OPTUNA_N_JOBS', '1')))),
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
            'learning_rate': lr,
            'seed': OPTUNA_SEED,
        })

        # ── Evaluate across pre-computed CPCV folds ──
        fold_scores = []
        fold_sortinos = []

        for fold_i, (X_train_es, X_val_es, y_train_es, y_val_es, w_train_es,
                      w_val_es, X_test, y_test) in enumerate(fold_data):

            dtrain = lgb.Dataset(X_train_es, label=y_train_es, weight=w_train_es,
                                 reference=parent_ds, free_raw_data=False)
            dval = lgb.Dataset(X_val_es, label=y_val_es, weight=w_val_es,
                               reference=parent_ds, free_raw_data=False)

            # MC-4: search trials use fast ES patience (OPTUNA_SEARCH_ES_PATIENCE=30);
            # final retrain uses LR-scaled patience for esoteric signals needing 800+ trees
            if search_lr:
                _es_patience = OPTUNA_SEARCH_ES_PATIENCE  # fast cutoff during search
            else:
                _es_patience = max(50, int(100 * (0.1 / params.get('learning_rate', 0.03))))

            if use_gpu and sp_sparse.issparse(X_train_es):
                # GPU fork path: manual Booster loop with set_external_csr
                gpu_params = params.copy()
                gpu_params['device_type'] = 'cuda_sparse'
                gpu_params.pop('force_col_wise', None)
                gpu_params.pop('force_row_wise', None)
                gpu_params.pop('device', None)
                gpu_params['histogram_pool_size'] = 512

                dtrain.construct()
                dval.construct()
                booster = lgb.Booster(gpu_params, dtrain)
                booster.add_valid(dval, 'val')
                booster.set_external_csr(X_train_es)

                best_score = float('inf')
                best_iter = 0
                no_improve = 0
                for rnd in range(rounds):
                    booster.update()
                    val_result = booster.eval_valid()[0]  # (dataset_name, metric_name, value, is_higher_better)
                    val_score = val_result[2]
                    if val_score < best_score:
                        best_score = val_score
                        best_iter = rnd + 1
                        no_improve = 0
                    else:
                        no_improve += 1
                    if no_improve >= _es_patience:
                        break
                    # Round-level pruning for Optuna (every 10 rounds)
                    if (rnd + 1) % 10 == 0:
                        global_step = fold_i * rounds + (rnd + 1)
                        trial.report(val_score, step=global_step)
                        if trial.should_prune():
                            del booster
                            import gc
                            gc.collect()
                            raise optuna.TrialPruned()
                model = booster
            else:
                # CPU path
                _prune_cb = _RoundPruningCallback(trial, fold_i, rounds, interval=10)
                callbacks = [lgb.early_stopping(_es_patience), lgb.log_evaluation(0), _prune_cb]

                model = lgb.train(
                    params, dtrain,
                    num_boost_round=rounds,
                    valid_sets=[dval],
                    valid_names=['val'],
                    callbacks=callbacks,
                )

            # OOS predictions — GPU path uses num_iteration=best_iter to avoid overfitting trees
            if use_gpu and sp_sparse.issparse(X_all):
                preds = model.predict(X_test, num_iteration=best_iter)
            else:
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

            del model, dtrain, dval
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

    import math
    ranges = {}

    # Linear-scale params: expand by 20% margin in linear space
    linear_params = ['num_leaves', 'min_data_in_leaf',
                     'feature_fraction_bynode', 'bagging_fraction',
                     'min_gain_to_split', 'max_depth']
    # Log-scale params: expand by 20% margin in LOG space (these use log=True in Optuna)
    log_params = ['feature_fraction', 'lambda_l1', 'lambda_l2']

    for pname in linear_params:
        values = [t.params[pname] for t in top_trials if pname in t.params]
        if not values:
            continue
        lo = min(values)
        hi = max(values)
        margin = max((hi - lo) * 0.2, abs(lo) * 0.1)
        ranges[pname] = (max(lo - margin, 0.001), hi + margin)

    for pname in log_params:
        values = [t.params[pname] for t in top_trials if pname in t.params]
        if not values:
            continue
        log_vals = [math.log(v) for v in values]
        lo_log = min(log_vals)
        hi_log = max(log_vals)
        # Expand by 20% on each side in log space
        margin = max((hi_log - lo_log) * 0.2, abs(lo_log) * 0.1)
        ranges[pname] = (math.exp(lo_log - margin), math.exp(hi_log + margin))

    # max_bin: LOCKED at 255 — controls EFB bundle size, not searchable
    ranges['max_bin'] = [255]

    # Clamp integer ranges
    if 'num_leaves' in ranges:
        ranges['num_leaves'] = (max(7, int(ranges['num_leaves'][0])), min(255, int(ranges['num_leaves'][1])))
    if 'min_data_in_leaf' in ranges:
        ranges['min_data_in_leaf'] = (max(1, int(ranges['min_data_in_leaf'][0])), max(2, int(ranges['min_data_in_leaf'][1])))
    if 'max_depth' in ranges:
        ranges['max_depth'] = (max(2, int(ranges['max_depth'][0])), min(20, int(ranges['max_depth'][1])))

    return ranges


# ============================================================
# WARM-START: Cross-TF param inheritance
# ============================================================
# Params that transfer across TFs (regularization/sampling — generalizes)
_WARMSTART_TRANSFERABLE = [
    'feature_fraction', 'feature_fraction_bynode', 'bagging_fraction',
    'lambda_l1', 'lambda_l2', 'min_gain_to_split', 'max_depth',
]
# Params that are TF-specific (DO NOT inherit — capped by TF_NUM_LEAVES, TF_MIN_DATA_IN_LEAF)
_WARMSTART_TF_SPECIFIC = ['num_leaves', 'min_data_in_leaf']

# TF cascade order: each TF inherits from the one before it
_TF_WARMSTART_PARENT = {
    '1d': '1w',
    '4h': '1d',
    '1h': '4h',
    '15m': '1h',
}


def load_warmstart_params(tf_name):
    """Load best params from the parent TF's Optuna config (if available).
    Returns (parent_params_dict, parent_tf_name) or (None, None) if not available.
    """
    parent_tf = _TF_WARMSTART_PARENT.get(tf_name)
    if parent_tf is None:
        return None, None

    parent_config_path = os.path.join(PROJECT_DIR, f'optuna_configs_{parent_tf}.json')
    if not os.path.exists(parent_config_path):
        return None, None

    with open(parent_config_path) as f:
        parent_config = json.load(f)

    parent_params = parent_config.get('best_params', {})
    if not parent_params:
        return None, None

    return parent_params, parent_tf


def compute_warmstart_ranges(parent_params, tf_name):
    """Compute narrowed search ranges from parent TF's best params.

    Transferable params get ±20% range. TF-specific params use their
    standard wide ranges (num_leaves, min_data_in_leaf).
    """
    from config import TF_MIN_DATA_IN_LEAF, TF_NUM_LEAVES

    ranges = {}

    for pname in _WARMSTART_TRANSFERABLE:
        if pname not in parent_params:
            continue
        val = parent_params[pname]
        if isinstance(val, int):
            margin = max(int(abs(val) * 0.2), 1)
            lo = max(1, val - margin)
            hi = val + margin
            ranges[pname] = (lo, hi)
        else:
            # Float: ±20%, clamped to sensible minimums
            lo = max(val * 0.8, 0.001)
            hi = val * 1.2
            ranges[pname] = (lo, hi)

    # Clamp max_depth to valid range
    if 'max_depth' in ranges:
        lo, hi = ranges['max_depth']
        ranges['max_depth'] = (max(4, lo), min(12, hi))

    # Clamp feature_fraction to [0.005, 0.1]
    if 'feature_fraction' in ranges:
        lo, hi = ranges['feature_fraction']
        ranges['feature_fraction'] = (max(0.005, lo), min(0.1, hi))

    # Clamp feature_fraction_bynode to [0.2, 0.8]
    if 'feature_fraction_bynode' in ranges:
        lo, hi = ranges['feature_fraction_bynode']
        ranges['feature_fraction_bynode'] = (max(0.2, lo), min(0.8, hi))

    # Clamp bagging_fraction to [0.5, 0.95]
    if 'bagging_fraction' in ranges:
        lo, hi = ranges['bagging_fraction']
        ranges['bagging_fraction'] = (max(0.5, lo), min(0.95, hi))

    # Clamp lambda_l1 to [0.01, 1.0]
    if 'lambda_l1' in ranges:
        lo, hi = ranges['lambda_l1']
        ranges['lambda_l1'] = (max(0.01, lo), min(1.0, hi))

    # Clamp lambda_l2 to [0.1, 20.0]
    if 'lambda_l2' in ranges:
        lo, hi = ranges['lambda_l2']
        ranges['lambda_l2'] = (max(0.1, lo), min(20.0, hi))

    # Clamp min_gain_to_split to [0.1, 5.0]
    if 'min_gain_to_split' in ranges:
        lo, hi = ranges['min_gain_to_split']
        ranges['min_gain_to_split'] = (max(0.1, lo), min(5.0, hi))

    # TF-specific params: use standard wide ranges (NOT inherited)
    _tf_mdil = TF_MIN_DATA_IN_LEAF.get(tf_name, 3)
    _tf_nl_cap = TF_NUM_LEAVES.get(tf_name, 63)
    ranges['num_leaves'] = (15, _tf_nl_cap)
    ranges['min_data_in_leaf'] = (max(1, _tf_mdil - 2), _tf_mdil + 10)

    # max_bin: LOCKED at 255
    ranges['max_bin'] = [255]

    return ranges


def build_warmstart_enqueue_params(parent_params, tf_name):
    """Build a param dict suitable for study.enqueue_trial() from parent TF params.

    Transferable params are copied directly. TF-specific params use the
    TF's default values (mid-range).
    """
    from config import TF_MIN_DATA_IN_LEAF, TF_NUM_LEAVES

    enqueue = {}

    # Copy transferable params
    for pname in _WARMSTART_TRANSFERABLE:
        if pname in parent_params:
            enqueue[pname] = parent_params[pname]

    # TF-specific: use sensible defaults (not inherited)
    _tf_mdil = TF_MIN_DATA_IN_LEAF.get(tf_name, 3)
    _tf_nl_cap = TF_NUM_LEAVES.get(tf_name, 63)
    enqueue['num_leaves'] = min(_tf_nl_cap, max(15, _tf_nl_cap // 2))
    enqueue['min_data_in_leaf'] = _tf_mdil

    return enqueue


# ============================================================
# FINAL RETRAINING WITH BEST PARAMS
# ============================================================
def final_retrain(X_all, y, sample_weights, feature_cols, is_sparse,
                  tf_name, max_hold, best_params, use_gpu=False, parent_ds=None):
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

    params = V3_LGBM_PARAMS.copy()
    params.update({
        'is_enable_sparse': is_sparse,  # match actual data format
        'verbosity': -1,
        'num_threads': 0,  # auto-detect via OpenMP
        'learning_rate': OPTUNA_FINAL_LR,
        'seed': OPTUNA_SEED,
        'bagging_freq': 1,
    })
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

        X_train_fold = X_train[:-val_size]
        dtrain = lgb.Dataset(X_train_fold, label=y_train[:-val_size],
                             weight=w_train[:-val_size], feature_name=feature_cols,
                             reference=parent_ds, free_raw_data=False)
        dval = lgb.Dataset(X_train[-val_size:], label=y_train[-val_size:],
                           weight=w_train[-val_size:],
                           feature_name=feature_cols, reference=parent_ds,
                           free_raw_data=False)

        _es_patience_final = max(50, int(100 * (0.1 / params.get('learning_rate', OPTUNA_FINAL_LR))))

        if use_gpu and sp_sparse.issparse(X_train_fold):
            # GPU fork path
            gpu_params = params.copy()
            gpu_params['device_type'] = 'cuda_sparse'
            gpu_params.pop('force_col_wise', None)
            gpu_params.pop('force_row_wise', None)
            gpu_params.pop('device', None)
            gpu_params['histogram_pool_size'] = 512

            dtrain.construct()
            dval.construct()
            booster = lgb.Booster(gpu_params, dtrain)
            booster.add_valid(dval, 'val')
            booster.set_external_csr(X_train_fold)

            best_score = float('inf')
            best_iter = 0
            no_improve = 0
            for rnd in range(OPTUNA_FINAL_ROUNDS):
                booster.update()
                val_result = booster.eval_valid()[0]
                val_score = val_result[2]
                if val_score < best_score:
                    best_score = val_score
                    best_iter = rnd + 1
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= _es_patience_final:
                    break
            model = booster
        else:
            # CPU path
            model = lgb.train(
                params, dtrain,
                num_boost_round=OPTUNA_FINAL_ROUNDS,
                valid_sets=[dtrain, dval],
                valid_names=['train', 'val'],
                callbacks=[lgb.early_stopping(_es_patience_final), lgb.log_evaluation(0)],
            )

        # GPU path: use num_iteration=best_iter to avoid overfitting trees
        if use_gpu and sp_sparse.issparse(X_all):
            preds = model.predict(X_test, num_iteration=best_iter)
        else:
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

        _n_trees = getattr(model, 'best_iteration', None) or getattr(model, 'current_iteration', lambda: '?')()
        log.info(f"    Fold {fold_i+1}/{len(splits)}: Acc={acc:.3f} PrecL={prec_long:.3f} PrecS={prec_short:.3f} Sortino={sortino:.2f} Trees={_n_trees}")

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
def run_search_for_tf(tf_name, max_stage=2, n_jobs=1, warmstart=True):
    """Run two-stage Optuna search + final retrain for one timeframe.

    Args:
        warmstart: If True, load best params from parent TF and use them to:
                   (1) seed the study with enqueue_trial
                   (2) narrow stage 1 ranges to parent ±20%
                   (3) reduce trial counts (50+50 instead of 100+100)
    """
    log.info(f"\n{'='*70}")
    log.info(f"OPTUNA SEARCH: {tf_name.upper()}")
    log.info(f"{'='*70}")

    total_start = time.time()

    # Load data
    X_all, y, sample_weights, feature_cols, is_sparse, max_hold = load_tf_data(tf_name)
    n_valid = (~np.isnan(y)).sum()
    log.info(f"  Features: {len(feature_cols):,} ({'SPARSE' if is_sparse else 'DENSE'})")
    log.info(f"  Valid samples: {int(n_valid):,} / {len(y):,}")

    # ── GPU fork detection ──
    # Strategy: CPU parallel for search stages (enables n_jobs parallelism),
    # GPU only for the single final retrain (no parallelism needed there)
    gpu_available = False
    if _HAS_GPU_FORK and is_sparse:
        try:
            gpu_available = should_use_gpu(tf_name, X_all)
        except Exception as e:
            log.warning(f"  GPU fork detection failed: {e}")

    search_use_gpu = False  # Always CPU for search = enables n_jobs parallelism
    final_use_gpu = gpu_available  # GPU for the single final retrain

    if gpu_available:
        log.info(f"  GPU available — Stage 1+2: CPU parallel (n_jobs={n_jobs}), Final: GPU")
    else:
        if _HAS_GPU_FORK and is_sparse:
            log.warning(f"  GPU fork available but not viable for {tf_name} — using CPU")
        elif _HAS_GPU_FORK and not is_sparse:
            log.warning(f"  GPU fork requires sparse data — data is dense, using CPU")
        else:
            log.info(f"  GPU fork: not available, using CPU")

    # ── Warm-start: load parent TF params if available ──
    warmstart_params = None
    warmstart_ranges = None
    warmstart_parent_tf = None
    is_warmstarted = False

    if warmstart:
        warmstart_params, warmstart_parent_tf = load_warmstart_params(tf_name)
        if warmstart_params is not None:
            warmstart_ranges = compute_warmstart_ranges(warmstart_params, tf_name)
            is_warmstarted = True
            log.info(f"  WARM-START from {warmstart_parent_tf.upper()}: inheriting {len([p for p in _WARMSTART_TRANSFERABLE if p in warmstart_params])}/{len(_WARMSTART_TRANSFERABLE)} transferable params")
            log.info(f"  Warm-start ranges: {json.dumps({k: v for k, v in warmstart_ranges.items() if k != 'max_bin'}, indent=4, default=str)}")
        else:
            log.info(f"  No warm-start available for {tf_name} (no parent TF config found)")

    # ── Build parent Dataset ONCE for EFB reuse across all trials ──
    valid_mask = ~np.isnan(y)
    log.info("  Building parent Dataset for EFB reuse...")
    t0_ds = time.time()
    _parent_ds = lgb.Dataset(
        X_all[valid_mask], label=y[valid_mask].astype(int),
        weight=sample_weights[valid_mask] if sample_weights is not None else None,
        params={'feature_pre_filter': False, 'max_bin': 255, 'min_data_in_bin': 1},
        free_raw_data=False,
    )
    _parent_ds.construct()
    log.info(f"  Parent Dataset built in {time.time()-t0_ds:.1f}s: "
             f"{_parent_ds.num_data()} rows, {_parent_ds.num_feature()} features "
             f"(EFB cached for all trials)")

    # Determine trial counts: per-TF override > warm-start override > global default
    if is_warmstarted:
        s1_trials = OPTUNA_WARMSTART_STAGE1_TRIALS
        s2_trials = OPTUNA_WARMSTART_STAGE2_TRIALS
    else:
        s1_trials = OPTUNA_TF_STAGE1_TRIALS.get(tf_name, OPTUNA_STAGE1_TRIALS)
        s2_trials = OPTUNA_TF_STAGE2_TRIALS.get(tf_name, OPTUNA_STAGE2_TRIALS)

    # Per-TF startup trials (15m needs more random warm-up due to higher variance)
    _tf_startup = OPTUNA_TF_N_STARTUP_TRIALS.get(tf_name, OPTUNA_N_STARTUP_TRIALS)

    # Create pruner
    if OPTUNA_PRUNER == 'hyperband':
        pruner = HyperbandPruner(
            min_resource=OPTUNA_PRUNER_MIN_RESOURCE,
            reduction_factor=OPTUNA_PRUNER_REDUCTION_FACTOR,
        )
    else:
        pruner = MedianPruner(
            n_startup_trials=_tf_startup,
            n_warmup_steps=30,    # skip first 30 rounds (round-level pruning)
            interval_steps=10,    # match round-level report interval
        )

    # Sampler with fixed seed for reproducibility
    sampler = optuna.samplers.TPESampler(
        seed=OPTUNA_SEED,
        n_startup_trials=_tf_startup,
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

    # Enqueue warm-start params as first trial (TPE evaluates this before exploring)
    if is_warmstarted:
        enqueue_params = build_warmstart_enqueue_params(warmstart_params, tf_name)
        study.enqueue_trial(enqueue_params)
        log.info(f"  Enqueued warm-start trial: {enqueue_params}")

    # ── STAGE 1: Coarse search (narrowed if warm-started) ──
    _ws_tag = " [WARM-START]" if is_warmstarted else ""
    log.info(f"\n  STAGE 1{_ws_tag}: {s1_trials} trials, {OPTUNA_SEARCH_CPCV_GROUPS} CPCV groups, "
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
        narrow_ranges=warmstart_ranges if is_warmstarted else None,
        use_gpu=search_use_gpu,
        parent_ds=_parent_ds,
    )

    # Count existing completed trials to determine remaining
    existing_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    stage1_remaining = max(0, s1_trials - existing_completed)

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
    log.info(f"\n  STAGE 2{_ws_tag}: {s2_trials} trials, {n_groups_full} CPCV groups (full), "
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
        use_gpu=search_use_gpu,
        parent_ds=_parent_ds,
    )

    existing_s2 = len([t for t in study_s2.trials if t.state == optuna.trial.TrialState.COMPLETE])
    s2_remaining = max(0, s2_trials - existing_s2)

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
        use_gpu=final_use_gpu, parent_ds=_parent_ds,
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
    config_out = {
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
        'warmstart_from': warmstart_parent_tf,
        'warmstart_trials': f'{s1_trials}+{s2_trials}' if is_warmstarted else None,
    }
    with open(config_path, 'w') as f:
        json.dump(config_out, f, indent=2)
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
    parser.add_argument('--no-warmstart', action='store_true',
                        help='Disable warm-start from parent TF (use full wide ranges + full trial count)')
    args = parser.parse_args()

    timeframes = args.tf if args.tf else TF_ORDER
    use_warmstart = not args.no_warmstart

    # Auto-detect parallel trials
    total_cores = get_cpu_count() or 24
    if args.n_jobs is not None:
        n_jobs = args.n_jobs
    else:
        # Use config constant if set (>0), otherwise auto-calculate
        n_jobs = OPTUNA_N_JOBS if OPTUNA_N_JOBS > 0 else max(1, min(4, total_cores // 96))

    log.info(f"Optuna LightGBM Search v3.3")
    log.info(f"  Cores: {total_cores}, Parallel trials: {n_jobs}")
    log.info(f"  Timeframes: {timeframes}")
    log.info(f"  Warm-start: {'ENABLED (cascade: 1w->1d->4h->1h->15m)' if use_warmstart else 'DISABLED'}")
    log.info(f"  Stage 1 (cold): per-TF {OPTUNA_TF_STAGE1_TRIALS} | (warm): {OPTUNA_WARMSTART_STAGE1_TRIALS} trials")
    log.info(f"  Stage 2 (cold): per-TF {OPTUNA_TF_STAGE2_TRIALS} | (warm): {OPTUNA_WARMSTART_STAGE2_TRIALS} trials")
    log.info(f"  Search: {OPTUNA_SEARCH_CPCV_GROUPS} CPCV groups, "
             f"{OPTUNA_SEARCH_ROW_SUBSAMPLE:.0%} subsample, lr={OPTUNA_SEARCH_LR}, "
             f"ES patience={OPTUNA_SEARCH_ES_PATIENCE}, rounds={OPTUNA_SEARCH_ROUNDS}")
    log.info(f"  Final: full CPCV, lr={OPTUNA_FINAL_LR}, rounds={OPTUNA_FINAL_ROUNDS}")

    all_results = {}
    total_start = time.time()

    for tf in timeframes:
        try:
            result = run_search_for_tf(tf, max_stage=args.stage, n_jobs=n_jobs,
                                       warmstart=use_warmstart)
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
