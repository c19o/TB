#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_optuna_local.py -- Phase 1 + Validation Gate Optuna LightGBM Search (v3.3)
================================================================================
Phase 1:      Rapid search — 25 trials (2 seeded + 8 random + 15 TPE),
              2-fold CPCV, 60 rounds max, LR=0.15, ES=15, sampled paths
Validation:   Top-3 re-evaluated with 4-fold CPCV, 200 rounds, LR=0.08, sampled paths
Final:        Retrain best config with full CPCV K=2, 800 rounds, LR=0.03, exhaustive paths

Usage:
    python run_optuna_local.py                    # all TFs
    python run_optuna_local.py --tf 1d            # single TF
    python run_optuna_local.py --tf 1d --tf 4h    # multiple TFs
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

# Wave 3: Lift MKL/OpenBLAS thread caps — lets LightGBM's OpenMP use all cores
try:
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=get_cpu_count() or 64, user_api='blas')
except ImportError:
    pass

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('SAVAGE22_DB_DIR', PROJECT_DIR)
os.environ.setdefault('SKIP_LLM', '1')

import optuna
from optuna.pruners import MedianPruner
try:
    from optuna.pruners import WilcoxonPruner
except ImportError:
    WilcoxonPruner = None  # fallback to MedianPruner
try:
    from optuna.pruners import PatientPruner
except ImportError:
    PatientPruner = None
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
    V3_LGBM_PARAMS, TF_MIN_DATA_IN_LEAF, TF_NUM_LEAVES, V30_DATA_DIR,
    OPTUNA_SEED, OPTUNA_SAMPLER,
    OPTUNA_PHASE1_TRIALS, OPTUNA_PHASE1_CPCV_GROUPS,
    OPTUNA_PHASE1_ROUNDS, OPTUNA_PHASE1_LR, OPTUNA_PHASE1_ES_PATIENCE,
    OPTUNA_PHASE1_N_STARTUP,
    OPTUNA_VALIDATION_TOP_K, OPTUNA_VALIDATION_CPCV_GROUPS,
    OPTUNA_VALIDATION_ROUNDS, OPTUNA_VALIDATION_LR, OPTUNA_VALIDATION_ES_PATIENCE,
    OPTUNA_WARMSTART_PHASE1_TRIALS, OPTUNA_WARMSTART_VALIDATION_TOP_K,
    OPTUNA_TF_ROW_SUBSAMPLE, OPTUNA_TF_PHASE1_TRIALS,
    OPTUNA_FINAL_LR, OPTUNA_FINAL_ROUNDS,
    OPTUNA_N_JOBS, TF_CPCV_GROUPS, TF_CLASS_WEIGHT,
    CPCV_OPTUNA_SAMPLE_PATHS, CPCV_SAMPLE_PATHS, CPCV_SAMPLE_SEED,
    CPCV_PARALLEL_GPUS,
    TF_MIN_DATA_IN_LEAF_MAX,
    OPTUNA_TF_PHASE1_LR, OPTUNA_TF_PHASE1_ROUNDS,
    OPTUNA_TF_MAX_DEPTH_RANGE, OPTUNA_TF_LR_SEARCH_RANGE,
    OPTUNA_TF_FINAL_ROUNDS,
)
from feature_library import compute_triple_barrier_labels, TRIPLE_BARRIER_CONFIG
from multi_gpu_optuna import (
    get_multi_gpu_config, apply_gpu_params, create_gpu_safe_sampler,
    gpu_oom_handler, get_gpu_trial_summary, clear_gpu_trial_map, MultiGPUConfig,
    _detect_lgbm_device_type,
)

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
    ends = np.asarray(t1_arr, dtype=np.int64) + 1  # +1: t1 is inclusive end bar, range(s,e) is exclusive
    return _compute_uniqueness_inner(starts, ends, n_bars)


def _detect_n_gpus():
    """Detect GPU count for fold-parallel CPCV in Optuna.
    Returns 0 if MULTI_GPU=0 or LightGBM GPU fork not available."""
    if os.environ.get('MULTI_GPU') == '0':
        return 0
    if not hasattr(lgb.Booster, 'set_external_csr'):
        return 0  # standard LightGBM — no cuda_sparse support
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


def _apply_binary_mode(params, tf_name):
    """Apply binary mode overrides to LightGBM params if tf_name uses binary classification."""
    from config import BINARY_TF_MODE
    if BINARY_TF_MODE.get(tf_name, False):
        params['objective'] = 'binary'
        params['metric'] = 'binary_logloss'
        params.pop('num_class', None)
    return params


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

    # Binary mode: match ml_multi_tf.py — SHORT(0)→0, FLAT(1)→NaN(drop), LONG(2)→1
    from config import BINARY_TF_MODE
    if BINARY_TF_MODE.get(tf_name, False):
        _flat_mask = (y == 1)
        y[_flat_mask] = np.nan
        _valid_binary = ~np.isnan(y)
        y[_valid_binary] = (y[_valid_binary] == 2).astype(float)
        n_up = int((y == 1).sum())
        n_down = int((y == 0).sum())
        n_dropped = int(_flat_mask.sum())
        log.info(f"  BINARY MODE ({tf_name}): {n_up} UP + {n_down} DOWN = {n_up+n_down} rows (dropped {n_dropped} FLAT)")

    # Feature columns
    meta_cols = {'timestamp', 'date', 'open', 'high', 'low', 'close', 'volume',
                 'quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote',
                 'open_time', 'date_norm'}
    target_like = {c for c in df.columns if 'next_' in c.lower() or 'target' in c.lower()
                   or 'direction' in c.lower() or c == 'triple_barrier_label'}
    exclude_cols = meta_cols | target_like
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Per-TF feature filter: drop short-period noise features for low-row TFs (1w)
    from config import apply_tf_feature_filter
    _pre_filter = len(feature_cols)
    _df_filtered = apply_tf_feature_filter(df[feature_cols], tf_name)
    feature_cols = list(_df_filtered.columns)
    if len(feature_cols) < _pre_filter:
        log.info(f"  TF feature filter: {_pre_filter} → {len(feature_cols)} features ({_pre_filter - len(feature_cols)} dropped)")

    X_base = _df_filtered.values.astype(np.float32)
    del _df_filtered
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
        # ALWAYS keep sparse CSR — LightGBM accepts scipy sparse natively via
        # LGBM_DatasetCreateFromCSR. Dense conversion is unnecessary and OOMs on
        # 4h+ (498GB for 5.3M features × 23K rows). With force_col_wise=True,
        # sparse training is multi-threaded. EFB works on sparse input.
        is_sparse = True
        # OPT-9: Convert to CSC for LightGBM column-wise iteration (force_col_wise=True)
        # CSC = columns contiguous in memory → faster histogram building when iterating features
        # BUG-L8 FIX: Skip CSC when force_row_wise — row-wise training ignores column layout
        from config import TF_FORCE_ROW_WISE
        if tf_name not in TF_FORCE_ROW_WISE:
            X_all = X_all.tocsc()
            log.info(f"  Converted to CSC for LightGBM column-wise access")
            _fmt = "CSC"
        else:
            log.info(f"  Keeping CSR for force_row_wise ({tf_name})")
            _fmt = "CSR"
        log.info(f"  Combined: {X_all.shape[1]:,} features ({n_base} base + {len(cross_cols):,} crosses) [SPARSE {_fmt}]")
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

    # T-3 FIX: Apply dict class weights here so Optuna and final training use the same
    # loss landscape. ml_multi_tf.py folds TF_CLASS_WEIGHT into sample_weights;
    # Optuna must match or it optimizes in a different objective than final training.
    # Note: 'balanced' TFs keep is_unbalance=True in params (handled at call sites).
    _cw = TF_CLASS_WEIGHT.get(tf_name)
    if isinstance(_cw, dict):
        _cw_arr = np.ones(len(y), dtype=np.float32)
        _cw_mask = ~np.isnan(y)
        _cw_arr[_cw_mask] = np.array([_cw.get(int(k), 1.0) for k in y[_cw_mask]], dtype=np.float32)
        sample_weights = sample_weights * _cw_arr
        log.info(f"  Class weights: {_cw} (SHORT={_cw.get(0, 1.0)}x) — folded into sample_weights")

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
        self._best_score = float('inf')

    def __call__(self, env):
        if (env.iteration + 1) % self.interval != 0:
            return
        for entry in env.evaluation_result_list:
            if entry[0] == 'val' and entry[1] in ('multi_logloss', 'binary_logloss'):
                score = entry[2]
                break
        else:
            return
        # Report best-so-far score to prevent noisy dips from triggering false prunes
        self._best_score = min(self._best_score, score)
        step = self.fold_i * self.max_rounds + (env.iteration + 1)
        self.trial.report(self._best_score, step=step)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


# ============================================================
# PHASE 1 OBJECTIVE FUNCTION BUILDER
# ============================================================
def build_phase1_objective(X_all, y, sample_weights, feature_cols, is_sparse, tf_name,
                           max_hold, parent_ds, row_subsample=1.0,
                           use_gpu=False, gpu_cfg=None, actual_n_jobs=1):
    """Build an Optuna objective for Phase 1 rapid search.

    Uses LR=0.15 (TF-overrideable), ES=15, max 60 rounds (TF-overrideable), 2-fold CPCV.
    Pre-computes CPCV fold indices ONCE for all trials.
    """
    n_groups = OPTUNA_PHASE1_CPCV_GROUPS
    lr = OPTUNA_TF_PHASE1_LR.get(tf_name, OPTUNA_PHASE1_LR)
    rounds = OPTUNA_TF_PHASE1_ROUNDS.get(tf_name, OPTUNA_PHASE1_ROUNDS)
    es_patience = OPTUNA_PHASE1_ES_PATIENCE
    # Per-TF searchable LR range (None = use fixed lr from phase config)
    _lr_search_range = OPTUNA_TF_LR_SEARCH_RANGE.get(tf_name)
    # Per-TF max_depth search range (default [2, 8])
    _max_depth_lo, _max_depth_hi = OPTUNA_TF_MAX_DEPTH_RANGE.get(tf_name, (2, 8))
    # Per-TF min_data_in_leaf max (default 10)
    _mdil_max = TF_MIN_DATA_IN_LEAF_MAX.get(tf_name, 10)
    log.info(f"  Phase 1 config for {tf_name}: lr={'search '+str(_lr_search_range) if _lr_search_range else lr}, "
             f"rounds={rounds}, max_depth=[{_max_depth_lo},{_max_depth_hi}], mdil_max={_mdil_max}")

    # ── Pre-compute CPCV fold data ONCE ──
    valid_mask = ~np.isnan(y)
    subsample_idx = np.where(valid_mask)[0]
    if row_subsample < 1.0:
        np.random.seed(OPTUNA_SEED)
        n_sample = int(len(subsample_idx) * row_subsample)
        subsample_idx = np.sort(np.random.choice(subsample_idx, size=n_sample, replace=False))

    n_sub = len(subsample_idx)
    _embargo_pct = max(0.01, max_hold / n_sub)  # embargo >= max_hold_bars bars (López de Prado)
    splits = _generate_cpcv_splits(
        n_sub, n_groups=n_groups, n_test_groups=1,
        max_hold_bars=max_hold, embargo_pct=_embargo_pct,
        sample_paths=CPCV_OPTUNA_SAMPLE_PATHS, seed=CPCV_SAMPLE_SEED,
    )

    # Map absolute indices to parent Dataset row positions
    valid_indices_sorted = np.where(valid_mask)[0]

    fold_data = []
    for _fi, (train_rel, test_rel) in enumerate(splits):
        abs_train = subsample_idx[train_rel]
        abs_test = subsample_idx[test_rel]

        if len(abs_train) < 50 or len(abs_test) < 20:
            continue

        parent_train_idx = np.searchsorted(valid_indices_sorted, abs_train)

        # 85/15 train/val split for early stopping
        val_size = max(int(len(parent_train_idx) * 0.15), 50)
        if val_size >= len(parent_train_idx):
            val_size = max(len(parent_train_idx) // 5, 20)

        train_indices = parent_train_idx[:-val_size]
        val_indices = parent_train_idx[-val_size:]

        X_test = X_all[abs_test]
        y_test = y[abs_test].astype(int)

        fold_data.append({
            'train_indices': train_indices.tolist(),
            'val_indices': val_indices.tolist(),
            'X_test': X_test,
            'y_test': y_test,
        })

    log.info(f"  Phase 1: pre-computed {len(fold_data)} CPCV folds (from {len(splits)} splits) [subset() pattern]")

    def objective(trial):
        """Phase 1 objective — returns float score (lower=better) or raises TrialPruned.

        CRITICAL: This function must ALWAYS return a valid float for completed trials.
        Exceptions during scoring are caught and return float('inf') instead of
        propagating (which would mark the trial as FAIL in the ask/tell handler).
        Only optuna.TrialPruned is re-raised (marks trial as PRUNED, not FAIL).
        """
        try:
            return _objective_inner(trial)
        except optuna.TrialPruned:
            raise
        except Exception as _outer_err:
            log.warning(f"  Trial {trial.number} OUTER error (returning inf): "
                        f"{type(_outer_err).__name__}: {_outer_err}")
            import traceback
            log.warning(f"  Traceback: {traceback.format_exc()[-500:]}")
            return float('inf')

    def _objective_inner(trial):
        trial_start = time.time()
        _tf_mdil = TF_MIN_DATA_IN_LEAF.get(tf_name, 3)
        _tf_nl_cap = TF_NUM_LEAVES.get(tf_name, 63)

        # Per-TF aware ranges — aggressive regularization for low-row TFs
        num_leaves = trial.suggest_int('num_leaves', 4, _tf_nl_cap)
        min_data_in_leaf = trial.suggest_int('min_data_in_leaf', max(2, _tf_mdil), _mdil_max)
        feature_fraction = trial.suggest_float('feature_fraction', 0.7, 1.0)
        feature_fraction_bynode = trial.suggest_float('feature_fraction_bynode', 0.7, 1.0)
        bagging_fraction = trial.suggest_float('bagging_fraction', 0.7, 1.0)
        lambda_l1 = trial.suggest_float('lambda_l1', 1e-4, 4.0, log=True)   # T-1: capped at 4.0 — [1,100] zeroed rare signals firing ≤15 times
        lambda_l2 = trial.suggest_float('lambda_l2', 1e-4, 10.0, log=True)  # T-1: capped at 10.0 — log-scale, mass near zero
        min_gain_to_split = trial.suggest_float('min_gain_to_split', 0.0, 5.0)
        max_depth = trial.suggest_int('max_depth', _max_depth_lo, _max_depth_hi)
        # OPT-11: extra_trees as searchable boolean — safe diversity injection
        # For binary features (0/1), extra_trees is a no-op (only 1 possible threshold)
        extra_trees = trial.suggest_categorical('extra_trees', [True, False])

        # Per-TF searchable learning rate (1w: [0.05, 0.3]) or fixed from phase config
        if _lr_search_range is not None:
            trial_lr = trial.suggest_float('learning_rate', _lr_search_range[0], _lr_search_range[1], log=True)
        else:
            trial_lr = lr

        params = V3_LGBM_PARAMS.copy()
        params.update({
            'is_enable_sparse': True,  # always True — permission gate for sparse CSR, not data coercion
            'num_threads': max(1, (get_cpu_count() or 4) // max(1, actual_n_jobs)),
            'num_leaves': num_leaves,
            'min_data_in_leaf': min_data_in_leaf,
            'feature_fraction': feature_fraction,
            'feature_fraction_bynode': feature_fraction_bynode,
            'bagging_fraction': bagging_fraction,
            'bagging_freq': 1,
            'lambda_l1': lambda_l1,
            'lambda_l2': lambda_l2,
            'min_gain_to_split': min_gain_to_split,
            'max_bin': 7,  # LOCKED — binary features need 2 bins, 4-tier needs ~5
            'max_depth': max_depth,
            'extra_trees': extra_trees,
            'learning_rate': trial_lr,
            'seed': OPTUNA_SEED,
        })
        # Binary mode: override objective/metric to match ml_multi_tf.py
        _apply_binary_mode(params, tf_name)
        if params.get('objective') == 'binary' and not _lr_search_range:
            params['learning_rate'] = 0.3  # binary uses higher LR
        # Wave 3: force_row_wise for 15m (high rows/bundles ratio)
        from config import TF_FORCE_ROW_WISE
        if tf_name in TF_FORCE_ROW_WISE:
            params['force_row_wise'] = True
            params.pop('force_col_wise', None)

        # Multi-GPU: assign trial to a GPU via round-robin
        if gpu_cfg is not None and gpu_cfg.enabled:
            apply_gpu_params(params, trial.number, gpu_cfg)
            n_gpus = gpu_cfg.num_gpus
        else:
            n_gpus = int(os.environ.get('LGBM_NUM_GPUS', '0'))
            if n_gpus > 0:
                params['device_type'] = _detect_lgbm_device_type()
                params['gpu_device_id'] = trial.number % n_gpus
                params['histogram_pool_size'] = 512
                params.pop('force_col_wise', None)
                params.pop('force_row_wise', None)
                params.pop('device', None)
        # T-3 FIX: Dict class weights are folded into sample_weights in load_tf_data().
        # Only set is_unbalance for 'balanced' TFs (no current TF uses this).
        if TF_CLASS_WEIGHT.get(tf_name) == 'balanced':
            params['is_unbalance'] = True

        fold_scores = []
        fold_sortinos = []

        try:
            for fold_i, fold in enumerate(fold_data):
                X_test = fold['X_test']
                y_test = fold['y_test']

                dtrain = parent_ds.subset(fold['train_indices'])
                dval = parent_ds.subset(fold['val_indices'])

                if (use_gpu or n_gpus > 0) and is_sparse and _HAS_GPU_FORK:
                    # GPU fork path — cuda_sparse with set_external_csr
                    _gpu_train_data = X_all[np.where(valid_mask)[0][fold['train_indices']]]
                    gpu_params = params.copy()
                    _dt = gpu_cfg.device_type if (gpu_cfg and gpu_cfg.enabled) else params.get('device_type', 'gpu')
                    gpu_params['device_type'] = _dt
                    gpu_params.pop('force_col_wise', None)
                    gpu_params.pop('force_row_wise', None)
                    gpu_params.pop('device', None)
                    gpu_params['histogram_pool_size'] = 512
                    if n_gpus > 0:
                        gpu_params['gpu_device_id'] = trial.number % n_gpus

                    dtrain.construct()
                    dval.construct()
                    booster = lgb.Booster(gpu_params, dtrain)
                    booster.add_valid(dval, 'val')
                    booster.set_external_csr(_gpu_train_data)

                    best_score = float('inf')
                    best_iter = 0
                    no_improve = 0
                    for rnd in range(rounds):
                        booster.update()
                        val_result = booster.eval_valid()[0]
                        val_score = val_result[2]
                        if val_score < best_score:
                            best_score = val_score
                            best_iter = rnd + 1
                            no_improve = 0
                        else:
                            no_improve += 1
                        if no_improve >= es_patience:
                            break
                    model = booster
                elif (use_gpu or n_gpus > 0) and is_sparse:
                    # Standard LightGBM GPU (OpenCL) — fork not available
                    gpu_params = params.copy()
                    _dt = gpu_cfg.device_type if (gpu_cfg and gpu_cfg.enabled) else params.get('device_type', 'gpu')
                    gpu_params['device_type'] = _dt
                    gpu_params.pop('force_col_wise', None)
                    gpu_params.pop('force_row_wise', None)
                    gpu_params.pop('device', None)
                    gpu_params['histogram_pool_size'] = 512
                    if n_gpus > 0:
                        gpu_params['gpu_device_id'] = trial.number % n_gpus

                    model = lgb.train(
                        gpu_params, dtrain,
                        num_boost_round=rounds,
                        valid_sets=[dval],
                        valid_names=['val'],
                        callbacks=[lgb.early_stopping(es_patience), lgb.log_evaluation(0)],
                    )
                else:
                    # CPU path — no round-level pruning callback (it raises TrialPruned
                    # inside lgb.train which aborts training and marks trial as PRUNED
                    # in ask/tell, even when the model is learning well).
                    # Inter-fold pruning at the end of each fold is sufficient.
                    callbacks = [
                        lgb.early_stopping(es_patience),
                        lgb.log_evaluation(0),
                    ]

                    model = lgb.train(
                        params, dtrain,
                        num_boost_round=rounds,
                        valid_sets=[dval],
                        valid_names=['val'],
                        callbacks=callbacks,
                    )

                # OOS predictions — use best_iteration from lgb.train or manual loop
                _best_it = getattr(model, 'best_iteration', None)
                preds = model.predict(X_test, num_iteration=_best_it) if _best_it else model.predict(X_test)
                # Binary mode: preds is 1D (P(class=1)), multiclass: 2D
                _is_binary_obj = (params.get('objective') == 'binary')
                if _is_binary_obj:
                    preds_2d = np.column_stack([1 - preds, preds])
                    pred_labels = (preds > 0.5).astype(int)
                    mlogloss = log_loss(y_test, preds_2d, labels=[0, 1])
                else:
                    pred_labels = np.argmax(preds, axis=1)
                    mlogloss = log_loss(y_test, preds, labels=[0, 1, 2])

                sim_ret = np.where(pred_labels == y_test, 1.0, -1.0)
                downside = sim_ret[sim_ret < 0]
                downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1.0
                sortino = np.mean(sim_ret) / max(downside_std, 1e-10) * np.sqrt(252)

                fold_scores.append(mlogloss)
                fold_sortinos.append(sortino)

                # OPT-10: Inter-fold pruning via WilcoxonPruner
                # Report fold mlogloss to Optuna (step=fold_idx for fold-level pruning)
                trial.report(mlogloss, step=fold_i)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                del model, dtrain, dval
                import gc
                gc.collect()

        except optuna.TrialPruned:
            # Re-raise TrialPruned — ask/tell handler marks as PRUNED
            raise
        except Exception as _obj_err:
            # CRITICAL FIX: Catch all non-pruning exceptions and return float('inf')
            # instead of propagating. This prevents the ask/tell handler from marking
            # the trial as FAIL (which means "no trials completed" if ALL trials fail).
            # float('inf') = worst possible score, but trial is COMPLETE not FAIL.
            log.warning(f"  Trial {trial.number} objective error (returning inf): "
                        f"{type(_obj_err).__name__}: {_obj_err}")
            return float('inf')

        if not fold_scores:
            return float('inf')

        mean_mlogloss = float(np.mean(fold_scores))
        mean_sortino = float(np.mean(fold_sortinos))

        trial.set_user_attr('mean_sortino', mean_sortino)
        trial.set_user_attr('n_folds', len(fold_scores))
        trial.set_user_attr('mean_mlogloss', mean_mlogloss)

        # Runtime post-trial check
        try:
            from runtime_checks import post_trial_check
            post_trial_check(trial.number, mean_mlogloss, trial.params,
                             time.time() - trial_start)
        except Exception:
            pass

        return mean_mlogloss

    # Wrap with GPU OOM handler when multi-GPU is active
    if gpu_cfg is not None and gpu_cfg.enabled:
        objective = gpu_oom_handler(objective)

    return objective


# ============================================================
# VALIDATION GATE
# ============================================================
def _run_single_validation_fold(fold_i, train_rel, test_rel, valid_indices, X_all, y,
                                params, rounds, es_patience, is_sparse, use_gpu,
                                parent_ds, n_gpus, n_total_folds):
    """Worker: train one validation CPCV fold. Returns (fold_i, mlogloss) or None."""
    import gc

    if len(train_rel) < 50 or len(test_rel) < 20:
        return None

    parent_train_idx = train_rel

    val_size = max(int(len(parent_train_idx) * 0.15), 50)
    if val_size >= len(parent_train_idx):
        val_size = max(len(parent_train_idx) // 5, 20)

    train_subset_idx = parent_train_idx[:-val_size].tolist()
    val_subset_idx = parent_train_idx[-val_size:].tolist()

    dtrain = parent_ds.subset(train_subset_idx)
    dval = parent_ds.subset(val_subset_idx)

    abs_test = valid_indices[test_rel]
    X_test = X_all[abs_test]
    y_test = y[abs_test].astype(int)

    # Per-fold GPU assignment + thread capping
    fold_params = params.copy()
    if n_gpus > 1:
        fold_params['num_threads'] = max(1, (get_cpu_count() or 8) // n_gpus)

    if use_gpu and is_sparse and _HAS_GPU_FORK:
        # GPU fork path — cuda_sparse with set_external_csr
        _gpu_train_data = X_all[valid_indices[parent_train_idx[:-val_size]]]
        gpu_params = fold_params.copy()
        gpu_params['device_type'] = _detect_lgbm_device_type()
        gpu_params.pop('force_col_wise', None)
        gpu_params.pop('force_row_wise', None)
        gpu_params.pop('device', None)
        gpu_params['histogram_pool_size'] = 512
        if n_gpus > 0:
            gpu_params['gpu_device_id'] = fold_i % n_gpus

        dtrain.construct()
        dval.construct()
        booster = lgb.Booster(gpu_params, dtrain)
        booster.add_valid(dval, 'val')
        booster.set_external_csr(_gpu_train_data)

        best_score = float('inf')
        best_iter = 0
        no_improve = 0
        for rnd in range(rounds):
            booster.update()
            val_result = booster.eval_valid()[0]
            val_score = val_result[2]
            if val_score < best_score:
                best_score = val_score
                best_iter = rnd + 1
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= es_patience:
                break
        model = booster
    elif (use_gpu or n_gpus > 0) and is_sparse:
        # Standard LightGBM GPU (OpenCL) — fork not available or multi-GPU
        gpu_params = fold_params.copy()
        gpu_params['device_type'] = _detect_lgbm_device_type()
        if n_gpus > 0:
            gpu_params['gpu_device_id'] = fold_i % n_gpus
        gpu_params['histogram_pool_size'] = 512
        gpu_params.pop('force_col_wise', None)
        gpu_params.pop('force_row_wise', None)
        gpu_params.pop('device', None)

        model = lgb.train(
            gpu_params, dtrain,
            num_boost_round=rounds,
            valid_sets=[dval],
            valid_names=['val'],
            callbacks=[lgb.early_stopping(es_patience), lgb.log_evaluation(0)],
        )
    else:
        model = lgb.train(
            fold_params, dtrain,
            num_boost_round=rounds,
            valid_sets=[dval],
            valid_names=['val'],
            callbacks=[lgb.early_stopping(es_patience), lgb.log_evaluation(0)],
        )

    _best_it = getattr(model, 'best_iteration', None)
    preds = model.predict(X_test, num_iteration=_best_it) if _best_it else model.predict(X_test)
    _is_binary_obj = (params.get('objective') == 'binary')
    if _is_binary_obj:
        preds_2d = np.column_stack([1 - preds, preds])
        mlogloss = log_loss(y_test, preds_2d, labels=[0, 1])
    else:
        mlogloss = log_loss(y_test, preds, labels=[0, 1, 2])

    _n_trees = getattr(model, 'best_iteration', None) or getattr(model, 'current_iteration', lambda: '?')()
    _gpu_tag = f" [GPU {fold_i % n_gpus}]" if n_gpus > 0 else ""
    log.info(f"      Val fold {fold_i+1}/{n_total_folds}{_gpu_tag}: mlogloss={mlogloss:.4f} trees={_n_trees}")

    del model, dtrain, dval
    gc.collect()
    return (fold_i, mlogloss)


def validate_config(params_dict, X_all, y, sample_weights, is_sparse, tf_name,
                    max_hold, parent_ds, use_gpu=False, n_val_workers=1):
    """Validate a single config with 4-fold CPCV, 200 rounds, LR=0.08.

    Folds are distributed across GPUs when n_gpus > 1 (round-robin by fold_i).
    Returns mean OOS mlogloss (lower = better). No pruning — full evaluation.
    """
    n_groups = OPTUNA_VALIDATION_CPCV_GROUPS
    lr = OPTUNA_VALIDATION_LR
    rounds = OPTUNA_VALIDATION_ROUNDS
    es_patience = OPTUNA_VALIDATION_ES_PATIENCE

    valid_mask = ~np.isnan(y)
    valid_indices = np.where(valid_mask)[0]
    n_valid = len(valid_indices)

    _embargo_pct = max(0.01, max_hold / n_valid)  # embargo >= max_hold_bars bars
    splits = _generate_cpcv_splits(
        n_valid, n_groups=n_groups, n_test_groups=1,
        max_hold_bars=max_hold, embargo_pct=_embargo_pct,
        sample_paths=CPCV_OPTUNA_SAMPLE_PATHS, seed=CPCV_SAMPLE_SEED,
    )

    params = V3_LGBM_PARAMS.copy()
    _apply_binary_mode(params, tf_name)
    params.update({
        'is_enable_sparse': True,  # always True — sparse CSR is always the data format
        'num_threads': max(1, (get_cpu_count() or 8) // max(1, n_val_workers)),  # FIX #18: cap to prevent oversubscription in parallel validation
        'learning_rate': lr,
        'seed': OPTUNA_SEED,
        'bagging_freq': 1,
    })
    # T-3 FIX: Dict class weights are folded into sample_weights in load_tf_data().
    # Only set is_unbalance for 'balanced' TFs (no current TF uses this).
    if TF_CLASS_WEIGHT.get(tf_name) == 'balanced':
        params['is_unbalance'] = True
    # Apply the trial's tuned params
    for k in ['num_leaves', 'min_data_in_leaf', 'feature_fraction',
              'feature_fraction_bynode', 'bagging_fraction',
              'lambda_l1', 'lambda_l2', 'min_gain_to_split', 'max_depth',
              'extra_trees']:
        if k in params_dict:
            params[k] = params_dict[k]
    params['max_bin'] = 7  # LOCKED — binary features need 2 bins, 4-tier needs ~5

    # Detect GPU count for per-fold distribution
    n_gpus = _detect_n_gpus()
    n_parallel = max(1, n_gpus) if n_gpus > 1 else 1

    fold_scores = []

    if n_parallel > 1:
        # Parallel fold execution across GPUs (LightGBM releases GIL during training)
        from concurrent.futures import ThreadPoolExecutor
        log.info(f"      Distributing {len(splits)} validation folds across {n_gpus} GPUs")
        futures = {}
        with ThreadPoolExecutor(max_workers=n_parallel) as pool:
            for fold_i, (train_rel, test_rel) in enumerate(splits):
                fut = pool.submit(
                    _run_single_validation_fold,
                    fold_i, train_rel, test_rel, valid_indices, X_all, y,
                    params, rounds, es_patience, is_sparse, use_gpu,
                    parent_ds, n_gpus, len(splits),
                )
                futures[fut] = fold_i

            for fut in futures:
                result = fut.result()
                if result is not None:
                    fold_scores.append(result[1])  # result = (fold_i, mlogloss)
    else:
        # Sequential fallback (1 GPU or CPU-only)
        for fold_i, (train_rel, test_rel) in enumerate(splits):
            result = _run_single_validation_fold(
                fold_i, train_rel, test_rel, valid_indices, X_all, y,
                params, rounds, es_patience, is_sparse, use_gpu,
                parent_ds, n_gpus, len(splits),
            )
            if result is not None:
                fold_scores.append(result[1])

    if not fold_scores:
        return float('inf')

    return float(np.mean(fold_scores))


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


def build_warmstart_enqueue_params(parent_params, tf_name):
    """Build a param dict suitable for study.enqueue_trial() from parent TF params.

    Transferable params are copied directly. TF-specific params use the
    TF's default values (mid-range).
    """
    enqueue = {}

    # Copy transferable params
    for pname in _WARMSTART_TRANSFERABLE:
        if pname in parent_params:
            enqueue[pname] = parent_params[pname]

    # TF-specific: use sensible defaults (not inherited)
    _tf_mdil = TF_MIN_DATA_IN_LEAF.get(tf_name, 3)
    _tf_nl_cap = TF_NUM_LEAVES.get(tf_name, 63)
    enqueue['num_leaves'] = min(_tf_nl_cap, max(4, _tf_nl_cap // 2))
    enqueue['min_data_in_leaf'] = _tf_mdil

    # If this TF has searchable LR, seed with the TF-specific Phase 1 LR
    _lr_range = OPTUNA_TF_LR_SEARCH_RANGE.get(tf_name)
    if _lr_range is not None:
        enqueue['learning_rate'] = OPTUNA_TF_PHASE1_LR.get(tf_name, OPTUNA_PHASE1_LR)

    return enqueue


def build_default_enqueue_params(tf_name):
    """Build a param dict from V3_LGBM_PARAMS defaults for seeding."""
    _tf_mdil = TF_MIN_DATA_IN_LEAF.get(tf_name, 3)
    params = {
        'num_leaves': V3_LGBM_PARAMS.get('num_leaves', 63),
        'min_data_in_leaf': _tf_mdil,
        'feature_fraction': V3_LGBM_PARAMS.get('feature_fraction', 0.9),
        'feature_fraction_bynode': V3_LGBM_PARAMS.get('feature_fraction_bynode', 0.8),
        'bagging_fraction': V3_LGBM_PARAMS.get('bagging_fraction', 0.8),
        'lambda_l1': V3_LGBM_PARAMS.get('lambda_l1', 0.5),
        'lambda_l2': V3_LGBM_PARAMS.get('lambda_l2', 3.0),
        'min_gain_to_split': V3_LGBM_PARAMS.get('min_gain_to_split', 2.0),
        'max_depth': -1 if V3_LGBM_PARAMS.get('max_depth', -1) == -1 else V3_LGBM_PARAMS.get('max_depth', 8),
    }
    # If this TF has searchable LR, seed with the TF-specific Phase 1 LR
    _lr_range = OPTUNA_TF_LR_SEARCH_RANGE.get(tf_name)
    if _lr_range is not None:
        params['learning_rate'] = OPTUNA_TF_PHASE1_LR.get(tf_name, OPTUNA_PHASE1_LR)
    return params


# ============================================================
# FINAL RETRAINING WITH BEST PARAMS
# ============================================================
def _run_single_final_fold(fold_i, train_rel, test_rel, valid_indices, X_all, y,
                           params, is_sparse, use_gpu, parent_ds, _final_rounds,
                           n_gpus, n_total_folds):
    """Worker: train one final CPCV fold. Returns dict with fold results."""
    import gc

    test_idx = valid_indices[test_rel]
    y_test = y[test_idx].astype(int)
    X_test = X_all[test_idx]

    if len(train_rel) < 50 or len(test_rel) < 20:
        return None

    parent_train_idx = train_rel

    val_size = max(int(len(parent_train_idx) * 0.15), 50)
    if val_size >= len(parent_train_idx):
        val_size = max(len(parent_train_idx) // 5, 20)

    train_subset_idx = parent_train_idx[:-val_size].tolist()
    val_subset_idx = parent_train_idx[-val_size:].tolist()

    dtrain = parent_ds.subset(train_subset_idx)
    dval = parent_ds.subset(val_subset_idx)

    _es_patience_final = max(50, int(100 * (0.1 / params.get('learning_rate', OPTUNA_FINAL_LR))))

    # FIX #5: Assign each fold to a different GPU (round-robin)
    fold_params = params.copy()
    if n_gpus > 1:
        fold_params['num_threads'] = max(1, (get_cpu_count() or 8) // n_gpus)
    best_iter = None

    if use_gpu and is_sparse and _HAS_GPU_FORK:
        # GPU fork path — cuda_sparse with set_external_csr
        _gpu_train_data = X_all[valid_indices[parent_train_idx[:-val_size]]]
        gpu_params = fold_params.copy()
        gpu_params['device_type'] = _detect_lgbm_device_type()
        gpu_params.pop('force_col_wise', None)
        gpu_params.pop('force_row_wise', None)
        gpu_params.pop('device', None)
        gpu_params['histogram_pool_size'] = 512
        if n_gpus > 0:
            gpu_params['gpu_device_id'] = fold_i % n_gpus

        dtrain.construct()
        dval.construct()
        booster = lgb.Booster(gpu_params, dtrain)
        booster.add_valid(dval, 'val')
        booster.set_external_csr(_gpu_train_data)

        best_score = float('inf')
        best_iter = 0
        no_improve = 0
        for rnd in range(_final_rounds):
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
    elif (use_gpu or n_gpus > 0) and is_sparse:
        # Standard LightGBM GPU (OpenCL) — fork not available or multi-GPU
        gpu_params = fold_params.copy()
        gpu_params['device_type'] = _detect_lgbm_device_type()
        if n_gpus > 0:
            gpu_params['gpu_device_id'] = fold_i % n_gpus
        gpu_params['histogram_pool_size'] = 512
        gpu_params.pop('force_col_wise', None)
        gpu_params.pop('force_row_wise', None)
        gpu_params.pop('device', None)

        model = lgb.train(
            gpu_params, dtrain,
            num_boost_round=_final_rounds,
            valid_sets=[dtrain, dval],
            valid_names=['train', 'val'],
            callbacks=[lgb.early_stopping(_es_patience_final), lgb.log_evaluation(0)],
        )
    else:
        model = lgb.train(
            fold_params, dtrain,
            num_boost_round=_final_rounds,
            valid_sets=[dtrain, dval],
            valid_names=['train', 'val'],
            callbacks=[lgb.early_stopping(_es_patience_final), lgb.log_evaluation(0)],
        )

    _best_it = best_iter if best_iter is not None else getattr(model, 'best_iteration', None)
    preds = model.predict(X_test, num_iteration=_best_it) if _best_it else model.predict(X_test)
    _is_binary = (params.get('objective') == 'binary')
    if _is_binary:
        pred_labels = (preds > 0.5).astype(int)
    else:
        pred_labels = np.argmax(preds, axis=1)
    acc = accuracy_score(y_test, pred_labels)
    _n_cls = 2 if _is_binary else 3
    prec_long = precision_score(y_test, pred_labels, labels=[_n_cls - 1], average='macro', zero_division=0)
    prec_short = precision_score(y_test, pred_labels, labels=[0], average='macro', zero_division=0)

    sim_ret = np.where(pred_labels == y_test, 1.0, -1.0)
    downside = sim_ret[sim_ret < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1.0
    sortino = np.mean(sim_ret) / max(downside_std, 1e-10) * np.sqrt(252)

    _n_trees = getattr(model, 'best_iteration', None) or getattr(model, 'current_iteration', lambda: '?')()
    _gpu_tag = f" [GPU {fold_i % n_gpus}]" if n_gpus > 0 else ""
    log.info(f"    Fold {fold_i+1}/{n_total_folds}{_gpu_tag}: Acc={acc:.3f} PrecL={prec_long:.3f} PrecS={prec_short:.3f} Sortino={sortino:.2f} Trees={_n_trees}")

    result = {
        'fold_i': fold_i,
        'model': model,
        'acc': acc,
        'sortino': sortino,
        'oos': {
            'path': fold_i,
            'test_indices': test_idx.tolist(),
            'y_true': y_test.tolist(),
            'y_pred_probs': preds.tolist(),
            'y_pred_labels': pred_labels.tolist(),
        },
    }

    del dtrain, dval
    gc.collect()
    return result


def final_retrain(X_all, y, sample_weights, feature_cols, is_sparse,
                  tf_name, max_hold, best_params, use_gpu=False, parent_ds=None):
    """Retrain with best params using full CPCV + full rounds + final LR.

    FIX #5: Folds run in parallel across all available GPUs (or CPU threads).
    """
    _final_rounds = OPTUNA_TF_FINAL_ROUNDS.get(tf_name, OPTUNA_FINAL_ROUNDS)
    log.info(f"  FINAL RETRAIN: full CPCV, lr={OPTUNA_FINAL_LR}, rounds={_final_rounds}")

    n_groups, n_test_groups = TF_CPCV_GROUPS.get(tf_name, (10, 2))
    valid_mask = ~np.isnan(y)
    valid_indices = np.where(valid_mask)[0]
    n_valid = len(valid_indices)

    _embargo_pct = max(0.01, max_hold / n_valid)  # embargo >= max_hold_bars bars
    splits = _generate_cpcv_splits(
        n_valid, n_groups=n_groups, n_test_groups=n_test_groups,
        max_hold_bars=max_hold, embargo_pct=_embargo_pct,
    )

    params = V3_LGBM_PARAMS.copy()
    _apply_binary_mode(params, tf_name)
    params.update({
        'is_enable_sparse': True,  # always True — sparse CSR input regardless of cross load status
        'verbosity': -1,
        'num_threads': 0,  # auto-detect via OpenMP
        'learning_rate': OPTUNA_FINAL_LR,
        'seed': OPTUNA_SEED,
        'bagging_freq': 1,
    })
    # T-3 FIX: Dict class weights are folded into sample_weights in load_tf_data().
    # Only set is_unbalance for 'balanced' TFs (no current TF uses this).
    if TF_CLASS_WEIGHT.get(tf_name) == 'balanced':
        params['is_unbalance'] = True
    # Apply best Optuna params
    for k in ['num_leaves', 'min_data_in_leaf', 'feature_fraction',
              'feature_fraction_bynode', 'bagging_fraction',
              'lambda_l1', 'lambda_l2', 'min_gain_to_split', 'max_bin',
              'max_depth', 'extra_trees']:
        if k in best_params:
            params[k] = best_params[k]

    # Detect GPU count for parallel fold distribution
    n_gpus = _detect_n_gpus()
    # Parallelize when: multiple GPUs available AND enough data (>2000 rows).
    # Small datasets (e.g. 1w) train so fast that thread overhead dominates.
    _use_parallel = n_gpus > 1 and n_valid > 2000
    n_parallel = min(n_gpus, len(splits)) if _use_parallel else 1
    if _use_parallel:
        # Partition CPU threads across concurrent folds so they don't over-subscribe
        _threads_per_fold = max(1, (get_cpu_count() or 8) // n_parallel)
        params['num_threads'] = _threads_per_fold
        log.info(f"  PARALLEL FINAL RETRAIN: {len(splits)} folds across {n_gpus} GPUs "
                 f"({n_parallel} concurrent, {_threads_per_fold} threads/fold)")
    else:
        _reason = "single GPU" if n_gpus <= 1 else f"small dataset ({n_valid} rows <= 2000)"
        log.info(f"  SEQUENTIAL FINAL RETRAIN: {len(splits)} folds ({_reason})")

    oos_predictions = []
    fold_accs = []
    fold_sortinos = []
    best_model_obj = None
    best_acc = 0

    if _use_parallel:
        # Parallel fold execution across GPUs using ThreadPoolExecutor.
        # ThreadPool is correct here: LightGBM releases GIL during C++ training,
        # and we avoid serializing the large Dataset/sparse matrix across processes.
        # GPU assignment is handled by gpu_device_id = fold_i % n_gpus in each fold.
        from concurrent.futures import ThreadPoolExecutor, as_completed
        futures = {}
        with ThreadPoolExecutor(max_workers=n_parallel) as pool:
            for fold_i, (train_rel, test_rel) in enumerate(splits):
                fut = pool.submit(
                    _run_single_final_fold,
                    fold_i, train_rel, test_rel, valid_indices, X_all, y,
                    params, is_sparse, use_gpu, parent_ds, _final_rounds,
                    n_gpus, len(splits),
                )
                futures[fut] = fold_i

            # Process results as they complete for faster feedback
            n_done = 0
            for fut in as_completed(futures):
                fold_idx = futures[fut]
                try:
                    result = fut.result()
                except Exception:
                    log.exception(f"    Fold {fold_idx+1} FAILED — skipping")
                    continue
                if result is None:
                    continue
                n_done += 1
                fold_accs.append(result['acc'])
                fold_sortinos.append(result['sortino'])
                oos_predictions.append(result['oos'])
                if result['acc'] > best_acc:
                    best_acc = result['acc']
                    best_model_obj = result['model']
            log.info(f"  Parallel final retrain: {n_done}/{len(splits)} folds completed")
    else:
        # Sequential fallback (1 GPU, CPU-only, or small dataset)
        for fold_i, (train_rel, test_rel) in enumerate(splits):
            result = _run_single_final_fold(
                fold_i, train_rel, test_rel, valid_indices, X_all, y,
                params, is_sparse, use_gpu, parent_ds, _final_rounds,
                n_gpus, len(splits),
            )
            if result is None:
                continue
            fold_accs.append(result['acc'])
            fold_sortinos.append(result['sortino'])
            oos_predictions.append(result['oos'])
            if result['acc'] > best_acc:
                best_acc = result['acc']
                best_model_obj = result['model']

    return {
        'best_model': best_model_obj,
        'oos_predictions': oos_predictions,
        'mean_accuracy': float(np.mean(fold_accs)) if fold_accs else 0,
        'mean_sortino': float(np.mean(fold_sortinos)) if fold_sortinos else 0,
        'n_folds': len(fold_accs),
    }


# ============================================================
# PARALLEL DATASET CONSTRUCTION
# ============================================================
def _build_chunk(chunk_csr, y, weights, params, bin_path):
    """Worker: build one column-chunk Dataset and save binary."""
    ds = lgb.Dataset(chunk_csr, label=y, weight=weights,
                     params=params, free_raw_data=True)
    ds.construct()
    ds.save_binary(bin_path)
    return bin_path


def _parallel_dataset_construct(X_csr, y, sample_weights=None, n_workers=None):
    """Build LightGBM Dataset using parallel column-chunk construction.

    Splits CSR column-wise, builds each chunk's Dataset in parallel via
    ProcessPoolExecutor, merges via add_features_from. 10-50x faster than
    single-threaded PushDataToMultiValBin for millions of sparse features.

    LightGBM maintainer-endorsed approach (GitHub #5205).
    """
    import tempfile
    import shutil

    n_cols = X_csr.shape[1]

    # Skip parallel for small feature counts (overhead not worth it)
    if n_cols < 100_000:
        log.info(f"  Dataset has {n_cols:,} cols -- using single-threaded construction")
        ds = lgb.Dataset(X_csr, label=y, weight=sample_weights,
                         params={'feature_pre_filter': False, 'max_bin': 7,
                                 'min_data_in_bin': 1, 'is_enable_sparse': True,
                                 'force_col_wise': True},
                         free_raw_data=False)
        ds.construct()
        return ds

    if n_workers is None:
        n_workers = min(64, max(4, (get_cpu_count() or 8) // 4))

    chunk_cols = np.array_split(np.arange(n_cols), n_workers)
    tmp_dir = tempfile.mkdtemp(prefix='lgbm_parallel_')

    params = {
        'max_bin': 7,
        'feature_pre_filter': False,
        'is_enable_sparse': True,
        'force_col_wise': True,
        'min_data_in_bin': 1,
        'num_threads': max(1, (get_cpu_count() or 8) // n_workers),
    }

    log.info(f"  Parallel Dataset construction: {n_workers} workers, "
             f"{n_cols:,} cols, ~{n_cols // n_workers:,} cols/worker")

    t0 = time.time()

    # Build chunks in parallel -- MUST use 'spawn' (fork deadlocks with tcmalloc/OpenMP)
    from concurrent.futures import ProcessPoolExecutor
    mp_ctx = multiprocessing.get_context('spawn')

    futures = []
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_ctx) as pool:
        for i, cols in enumerate(chunk_cols):
            chunk_csr = X_csr[:, cols]
            bin_path = os.path.join(tmp_dir, f'chunk_{i}.bin')
            futures.append(pool.submit(_build_chunk, chunk_csr, y,
                                       sample_weights, params, bin_path))

    paths = [f.result() for f in futures]
    build_time = time.time() - t0
    log.info(f"  Parallel chunk build: {build_time:.1f}s ({n_workers} workers)")

    # Sequential merge (fast -- metadata only, no re-binning)
    t1 = time.time()
    _ds_params = {'min_data_in_bin': 1, 'max_bin': 7, 'feature_pre_filter': False, 'is_enable_sparse': True}
    base_ds = lgb.Dataset(paths[0], params=_ds_params).construct()
    for p in paths[1:]:
        chunk_ds = lgb.Dataset(p, params=_ds_params).construct()
        base_ds.add_features_from(chunk_ds)

    merge_time = time.time() - t1
    total_time = time.time() - t0
    log.info(f"  Merge: {merge_time:.1f}s | Total: {total_time:.1f}s "
             f"({base_ds.num_data()} rows, {base_ds.num_feature()} features)")

    # Cleanup temp files
    try:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass

    return base_ds


# ============================================================
# MAIN SEARCH FOR ONE TF
# ============================================================
def run_search_for_tf(tf_name, n_jobs=1, warmstart=True):
    """Run Phase 1 + Validation Gate + Final Retrain for one timeframe.

    Args:
        warmstart: If True, load best params from parent TF and use them to:
                   (1) seed the study with enqueue_trial (warm-start best)
                   (2) reduce trial counts (15 instead of 25)
    """
    log.info(f"\n{'='*70}")
    log.info(f"OPTUNA SEARCH: {tf_name.upper()}")
    log.info(f"{'='*70}")

    total_start = time.time()

    # Clear stale GPU trial assignments from previous timeframe
    clear_gpu_trial_map()

    # Load data
    X_all, y, sample_weights, feature_cols, is_sparse, max_hold = load_tf_data(tf_name)

    # Dense→sparse conversion BEFORE Optuna — LightGBM prefers CSR, and runtime_checks
    # expects sparse for memory estimation. Dense is fine for small TFs (1w) but
    # converting early ensures consistent code paths through the entire pipeline.
    if not is_sparse:
        log.info(f"  Dense matrix ({X_all.shape}) — converting to sparse CSR before Optuna")
        X_all = sp_sparse.csr_matrix(X_all)
        is_sparse = True

    n_valid = (~np.isnan(y)).sum()
    log.info(f"  Features: {len(feature_cols):,} ({'SPARSE' if is_sparse else 'DENSE'})")
    log.info(f"  Valid samples: {int(n_valid):,} / {len(y):,}")

    # ── GPU detection (multi-GPU + fork) ──
    gpu_cfg = get_multi_gpu_config(total_cores=os.cpu_count() or 8)

    gpu_available = False
    if _HAS_GPU_FORK and is_sparse:
        try:
            gpu_available = should_use_gpu(tf_name, X_all)
        except Exception as e:
            log.warning(f"  GPU fork detection failed: {e}")

    if gpu_cfg.enabled and gpu_cfg.num_gpus > 0:
        search_use_gpu = False  # GPU handled via apply_gpu_params in objective
        final_use_gpu = False   # Same — handled via gpu_cfg
        # Override n_jobs to match GPU count for trial-level parallelism
        n_jobs = gpu_cfg.n_jobs
        log.info(f"  MULTI-GPU MODE: {gpu_cfg.num_gpus} GPUs, {n_jobs} parallel trials, "
                 f"device={gpu_cfg.device_type}, {gpu_cfg.threads_per_trial} threads/trial")
    elif gpu_available:
        search_use_gpu = False  # GPU fork for final only
        final_use_gpu = True
        log.info(f"  GPU fork available — Phase 1+Validation: CPU parallel (n_jobs={n_jobs}), Final: GPU")
    else:
        search_use_gpu = False
        final_use_gpu = False
        gpu_cfg = MultiGPUConfig()  # disabled config
        if _HAS_GPU_FORK and is_sparse:
            log.warning(f"  GPU fork available but not viable for {tf_name} — using CPU")
        elif _HAS_GPU_FORK and not is_sparse:
            log.warning(f"  GPU fork requires sparse data — data is dense, using CPU")
        else:
            log.info(f"  GPU fork: not available, using CPU")

    # ── Warm-start: load parent TF params if available ──
    warmstart_params = None
    warmstart_parent_tf = None
    is_warmstarted = False

    if warmstart:
        warmstart_params, warmstart_parent_tf = load_warmstart_params(tf_name)
        if warmstart_params is not None:
            is_warmstarted = True
            log.info(f"  WARM-START from {warmstart_parent_tf.upper()}: inheriting {len([p for p in _WARMSTART_TRANSFERABLE if p in warmstart_params])}/{len(_WARMSTART_TRANSFERABLE)} transferable params")
        else:
            log.info(f"  No warm-start available for {tf_name} (no parent TF config found)")

    # ── Build parent Dataset ONCE for EFB reuse across all trials ──
    # ── Runtime pre-flight checks (after data load, before training) ──
    try:
        from runtime_checks import preflight_training, TrainingMonitor, post_trial_check
        _base_params = dict(V3_LGBM_PARAMS)
        _base_params['num_leaves'] = TF_NUM_LEAVES.get(tf_name, 63)
        _apply_binary_mode(_base_params, tf_name)
        preflight_training(X_all, y, tf_name, _base_params, n_jobs)
        _rt_monitor = TrainingMonitor(interval=30)
        _rt_monitor.start()
    except ImportError:
        log.warning("  runtime_checks.py not found — skipping runtime validation")
        _rt_monitor = None
    except RuntimeError as e:
        log.error(f"  Runtime pre-flight FAILED: {e}")
        sys.exit(1)

    valid_mask = ~np.isnan(y)
    bin_path = os.path.join(PROJECT_DIR, f'lgbm_dataset_{tf_name}.bin')

    # T-3 FIX: If this TF has explicit dict class weights (folded into sample_weights
    # by load_tf_data), any existing binary Dataset was built WITHOUT those weights.
    # Invalidate the cache so it is rebuilt with the correct per-sample weights.
    _has_dict_cw = isinstance(TF_CLASS_WEIGHT.get(tf_name), dict)
    if _has_dict_cw and os.path.exists(bin_path):
        log.info(f"  Invalidating cached Dataset binary (class weights changed): {bin_path}")
        try:
            os.remove(bin_path)
        except OSError as _e:
            log.warning(f"  Could not remove {bin_path}: {_e}")

    # H-4 FIX: Staleness check — if parquet or NPZ is newer than cached binary, invalidate.
    # Prevents training on stale features after a feature rebuild.
    if os.path.exists(bin_path):
        bin_mtime = os.path.getmtime(bin_path)
        _stale_sources = []
        for _src_pattern in [f'features_{tf_name}.parquet', f'features_BTC_{tf_name}.parquet',
                             f'v2_crosses_BTC_{tf_name}.npz']:
            for _src_dir in [DB_DIR, V30_DATA_DIR]:
                _src_path = os.path.join(_src_dir, _src_pattern)
                if os.path.exists(_src_path) and os.path.getmtime(_src_path) > bin_mtime:
                    _stale_sources.append(_src_pattern)
                    break
        if _stale_sources:
            log.info(f"  Invalidating cached Dataset binary (source newer): {', '.join(_stale_sources)}")
            try:
                os.remove(bin_path)
            except OSError as _e:
                log.warning(f"  Could not remove stale {bin_path}: {_e}")

    # Try loading pre-built binary first (instant — skips EFB reconstruction)
    _n_expected_features = X_all.shape[1]
    if os.path.exists(bin_path):
        log.info(f"  Loading parent Dataset from binary: {bin_path}")
        t0_ds = time.time()
        _parent_ds = lgb.Dataset(bin_path, params={
            'min_data_in_bin': 1, 'max_bin': 7,
            'feature_pre_filter': False, 'is_enable_sparse': True,
        })
        _parent_ds.construct()
        # Feature count mismatch check — binary was built with different features
        if _parent_ds.num_feature() != _n_expected_features:
            log.warning(f"  Binary feature count mismatch: binary={_parent_ds.num_feature()}, "
                        f"current={_n_expected_features}. Invalidating stale binary.")
            del _parent_ds
            try:
                os.remove(bin_path)
            except OSError as _e:
                log.warning(f"  Could not remove stale {bin_path}: {_e}")
            # Fall through to rebuild below
            _parent_ds = None
        else:
            log.info(f"  Loaded in {time.time()-t0_ds:.1f}s: "
                     f"{_parent_ds.num_data()} rows, {_parent_ds.num_feature()} features")
    else:
        _parent_ds = None

    if _parent_ds is None:
        # First build — use parallel column-chunk construction for large feature sets
        log.info("  Building parent Dataset (parallel column-chunk for >100K features)...")
        t0_ds = time.time()
        X_valid = X_all[valid_mask]
        y_valid = y[valid_mask].astype(int)
        w_valid = sample_weights[valid_mask] if sample_weights is not None else None
        _parent_ds = _parallel_dataset_construct(X_valid, y_valid, w_valid)
        log.info(f"  Parent Dataset built in {time.time()-t0_ds:.1f}s: "
                 f"{_parent_ds.num_data()} rows, {_parent_ds.num_feature()} features "
                 f"(EFB cached — folds use subset())")
        _parent_ds.save_binary(bin_path)
        log.info(f"  Saved binary: {bin_path}")

    # Determine trial counts
    if is_warmstarted:
        phase1_trials = OPTUNA_WARMSTART_PHASE1_TRIALS
        validation_top_k = OPTUNA_WARMSTART_VALIDATION_TOP_K
    else:
        phase1_trials = OPTUNA_TF_PHASE1_TRIALS.get(tf_name, OPTUNA_PHASE1_TRIALS)
        validation_top_k = OPTUNA_VALIDATION_TOP_K

    row_subsample = OPTUNA_TF_ROW_SUBSAMPLE.get(tf_name, 1.0)

    # OPT-10: WilcoxonPruner for inter-fold statistical pruning
    # After each CPCV fold, test if trial is statistically worse than best.
    # p_threshold=0.1 (conservative), n_startup_steps=2 (need ≥2 folds before pruning)
    # Wrapped with PatientPruner (patience=10) to prevent killing mid-fold trials
    # that discover rare signals — n_warmup_steps=120 ensures both folds complete
    if WilcoxonPruner is not None:
        _inner = WilcoxonPruner(
            p_threshold=0.1,
            n_startup_steps=2,
        )
        if PatientPruner is not None:
            pruner = PatientPruner(
                wrapped_pruner=_inner,
                patience=10,
                min_delta=0.001,
            )
        else:
            pruner = _inner
    else:
        # Fallback for older Optuna versions
        _median = MedianPruner(
            n_startup_trials=OPTUNA_PHASE1_N_STARTUP,
            n_warmup_steps=120,
            interval_steps=1,
        )
        pruner = _median

    # Sampler selection: TPE (default) or CMA-ES (better for continuous spaces)
    # Multi-GPU: use constant_liar=True for thread-safe parallel sampling (TPE only)
    _use_cmaes = OPTUNA_SAMPLER.lower() == 'cmaes'
    if _use_cmaes:
        try:
            from optuna.samplers import CmaEsSampler
            # CMA-ES doesn't support categorical params natively.
            # Use TPESampler as independent_sampler for categoricals (extra_trees).
            _tpe_fallback = optuna.samplers.TPESampler(
                seed=OPTUNA_SEED,
                n_startup_trials=OPTUNA_PHASE1_N_STARTUP,
                multivariate=True,
            )
            sampler = CmaEsSampler(
                seed=OPTUNA_SEED,
                n_startup_trials=OPTUNA_PHASE1_N_STARTUP,
                independent_sampler=_tpe_fallback,
            )
            log.info("  Using CmaEsSampler (TPE fallback for categoricals)")
        except (ImportError, AttributeError):
            log.warning("  CMA-ES sampler unavailable, falling back to TPESampler")
            _use_cmaes = False

    if not _use_cmaes:
        if gpu_cfg.enabled and gpu_cfg.num_gpus > 1:
            sampler = create_gpu_safe_sampler(OPTUNA_SEED, OPTUNA_PHASE1_N_STARTUP)
            log.info(f"  Using TPESampler with constant_liar=True (multi-GPU parallel sampling)")
        else:
            sampler = optuna.samplers.TPESampler(
                seed=OPTUNA_SEED,
                n_startup_trials=OPTUNA_PHASE1_N_STARTUP,
                multivariate=True,
                group=True,
            )

    study_name = f'lgbm_{tf_name}_v33'

    # FIX #41: In-memory storage — eliminates SQLite single-writer bottleneck.
    # SQLite serializes all trial inserts, killing multi-GPU parallel throughput.
    # In-memory is safe here: we save best params to JSON at the end.
    study = optuna.create_study(
        study_name=study_name,
        storage=None,  # in-memory — no SQLite bottleneck
        direction='minimize',  # minimize mlogloss
        pruner=pruner,
        sampler=sampler,
    )

    # ── Enqueue 2 seed trials ──
    # Seed 1: warm-start best (parent TF) OR V3_LGBM_PARAMS defaults
    if is_warmstarted:
        enqueue_ws = build_warmstart_enqueue_params(warmstart_params, tf_name)
        study.enqueue_trial(enqueue_ws)
        log.info(f"  Enqueued seed 1 (warm-start): {enqueue_ws}")
    else:
        # No warm-start — use default params as seed 1
        default_params = build_default_enqueue_params(tf_name)
        # max_depth=-1 not valid for suggest_int range [4,12], clamp to 8
        if default_params.get('max_depth', -1) == -1:
            default_params['max_depth'] = 8
        study.enqueue_trial(default_params)
        log.info(f"  Enqueued seed 1 (defaults): {default_params}")

    # Seed 2: V3_LGBM_PARAMS defaults (always — gives TPE a known baseline)
    default_seed2 = build_default_enqueue_params(tf_name)
    if default_seed2.get('max_depth', -1) == -1:
        default_seed2['max_depth'] = 8
    # Only enqueue if different from seed 1 (avoid duplicate)
    if is_warmstarted:
        study.enqueue_trial(default_seed2)
        log.info(f"  Enqueued seed 2 (defaults): {default_seed2}")
    # If not warmstarted, seed 1 IS the defaults, so skip duplicate

    # ═══════════════════════════════════════════════════════════
    # PHASE 1: RAPID SEARCH
    # ═══════════════════════════════════════════════════════════
    _ws_tag = " [WARM-START]" if is_warmstarted else ""
    log.info(f"\n  PHASE 1{_ws_tag}: {phase1_trials} trials, {OPTUNA_PHASE1_CPCV_GROUPS}-fold CPCV, "
             f"{row_subsample:.0%} row subsample, lr={OPTUNA_PHASE1_LR}, "
             f"ES={OPTUNA_PHASE1_ES_PATIENCE}, rounds={OPTUNA_PHASE1_ROUNDS}")
    phase1_start = time.time()

    objective_p1 = build_phase1_objective(
        X_all, y, sample_weights, feature_cols, is_sparse, tf_name,
        max_hold, parent_ds=_parent_ds,
        row_subsample=row_subsample,
        use_gpu=search_use_gpu,
        gpu_cfg=gpu_cfg if gpu_cfg.enabled else None,
        actual_n_jobs=n_jobs,
    )

    # FIX #1: Ask/tell batch pattern for multi-GPU parallel evaluation.
    # study.optimize() holds the GIL between trials — ask/tell lets us evaluate
    # a full batch of trials concurrently (one per GPU), then tell results.
    # 50-200x speedup on multi-GPU setups.
    log.info(f"  Running {phase1_trials} Phase 1 trials (ask/tell batch, batch_size={n_jobs})...")

    # OPT-13: Disable GC during Optuna search — LightGBM C++ does heavy lifting, Python GC is overhead
    import gc as _gc
    _gc.disable()
    try:
        completed = 0
        from concurrent.futures import ThreadPoolExecutor
        while completed < phase1_trials:
            batch_size = min(n_jobs, phase1_trials - completed)
            # Ask: get batch_size trial suggestions
            trials_batch = []
            for _ in range(batch_size):
                trial = study.ask()
                trials_batch.append(trial)

            # Evaluate batch in parallel (threads — objective uses C++ LightGBM, releases GIL)
            if batch_size > 1:
                futures = {}
                with ThreadPoolExecutor(max_workers=batch_size) as pool:
                    for trial in trials_batch:
                        fut = pool.submit(objective_p1, trial)
                        futures[fut] = trial
                    for fut in futures:
                        trial = futures[fut]
                        try:
                            value = fut.result()
                        except optuna.TrialPruned:
                            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                            continue
                        except Exception as e:
                            log.warning(f"  Trial {trial.number} exception in objective "
                                        f"({type(e).__name__}): {e}")
                            study.tell(trial, state=optuna.trial.TrialState.FAIL)
                            continue
                        # Tell value separately — if study.tell raises, don't double-tell as FAIL
                        try:
                            study.tell(trial, float(value))
                        except Exception as e:
                            log.error(f"  Trial {trial.number} study.tell() failed "
                                      f"({type(e).__name__}): {e}  value={value!r}")
            else:
                # Single trial — no thread overhead
                trial = trials_batch[0]
                try:
                    value = objective_p1(trial)
                except optuna.TrialPruned:
                    study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                    completed += batch_size
                    _n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
                    _n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
                    log.info(f"  Progress: {completed}/{phase1_trials} dispatched, "
                             f"{_n_complete} complete, {_n_pruned} pruned")
                    continue
                except Exception as e:
                    log.warning(f"  Trial {trial.number} exception in objective "
                                f"({type(e).__name__}): {e}")
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                    completed += batch_size
                    _n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
                    _n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
                    log.info(f"  Progress: {completed}/{phase1_trials} dispatched, "
                             f"{_n_complete} complete, {_n_pruned} pruned")
                    continue
                # Tell value separately
                try:
                    study.tell(trial, float(value))
                except Exception as e:
                    log.error(f"  Trial {trial.number} study.tell() failed "
                              f"({type(e).__name__}): {e}  value={value!r}")

            completed += batch_size
            _n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            _n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
            log.info(f"  Progress: {completed}/{phase1_trials} dispatched, "
                     f"{_n_complete} complete, {_n_pruned} pruned")
    finally:
        _gc.enable()
        _gc.collect()

    phase1_elapsed = time.time() - phase1_start

    # Summarize trial states for debugging
    _state_counts = {}
    for _t in study.trials:
        _sname = _t.state.name
        _state_counts[_sname] = _state_counts.get(_sname, 0) + 1
    log.info(f"  Phase 1 trial states: {_state_counts}")

    try:
        best_p1 = study.best_trial
    except ValueError:
        # "No trials are completed yet" — all trials were PRUNED or FAIL
        log.error(f"  Phase 1 FAILED: no completed trials out of {len(study.trials)} "
                  f"(states: {_state_counts}). Cannot proceed.")
        # Log first few trial details for debugging
        for _t in study.trials[:5]:
            log.error(f"    Trial {_t.number}: state={_t.state.name}, "
                      f"params={_t.params}, user_attrs={_t.user_attrs}")
        return None

    log.info(f"  Phase 1 done in {phase1_elapsed:.0f}s: best mlogloss={best_p1.value:.4f}")
    log.info(f"  Best params: {best_p1.params}")
    if 'mean_sortino' in best_p1.user_attrs:
        log.info(f"  Best sortino: {best_p1.user_attrs['mean_sortino']:.2f}")

    # Log multi-GPU trial distribution
    if gpu_cfg.enabled:
        gpu_summary = get_gpu_trial_summary()
        for gpu_id, trials in sorted(gpu_summary.items()):
            log.info(f"  GPU {gpu_id}: ran {len(trials)} trials ({trials[:10]}{'...' if len(trials) > 10 else ''})")

    # ═══════════════════════════════════════════════════════════
    # VALIDATION GATE: Top-K re-evaluated with 4-fold CPCV
    # ═══════════════════════════════════════════════════════════
    log.info(f"\n  VALIDATION GATE: top-{validation_top_k} configs, "
             f"{OPTUNA_VALIDATION_CPCV_GROUPS}-fold CPCV, "
             f"lr={OPTUNA_VALIDATION_LR}, rounds={OPTUNA_VALIDATION_ROUNDS}, "
             f"ES={OPTUNA_VALIDATION_ES_PATIENCE}")
    val_start = time.time()

    # Get top-K completed trials from Phase 1
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed_trials.sort(key=lambda t: t.value)
    top_trials = completed_trials[:validation_top_k]

    if not top_trials:
        log.error("  No completed trials found — cannot validate")
        return None

    best_params = None
    best_val_score = float('inf')
    val_results = []

    # H-3 FIX: Parallelize validation gate — each trial validates independently.
    # Uses ThreadPoolExecutor (not Process) because parent_ds can't be pickled.
    # Each validation uses num_threads=0 (full cores), so we cap workers to avoid
    # oversubscription: max 2 concurrent validations (each gets half the cores).
    _val_max_workers = min(len(top_trials), max(1, n_jobs))
    if _val_max_workers > 1:
        log.info(f"    Parallel validation: {_val_max_workers} workers")
        from concurrent.futures import ThreadPoolExecutor
        _val_futures = {}
        with ThreadPoolExecutor(max_workers=_val_max_workers) as _val_pool:
            for trial in top_trials:
                log.info(f"    Submitting trial #{trial.number} (Phase 1 mlogloss={trial.value:.4f})...")
                fut = _val_pool.submit(
                    validate_config,
                    trial.params, X_all, y, sample_weights, is_sparse, tf_name,
                    max_hold, parent_ds=_parent_ds, use_gpu=search_use_gpu,
                    n_val_workers=_val_max_workers,
                )
                _val_futures[fut] = trial

            for fut in _val_futures:
                trial = _val_futures[fut]
                val_score = fut.result()
                val_results.append({
                    'trial_number': trial.number,
                    'phase1_mlogloss': float(trial.value),
                    'validation_mlogloss': val_score,
                    'params': trial.params,
                })
                log.info(f"    Trial #{trial.number}: validation mlogloss={val_score:.4f} "
                         f"(Phase 1: {trial.value:.4f})")
                if val_score < best_val_score:
                    best_val_score = val_score
                    best_params = trial.params.copy()
    else:
        for rank, trial in enumerate(top_trials):
            log.info(f"    Validating trial #{trial.number} (Phase 1 mlogloss={trial.value:.4f})...")
            val_score = validate_config(
                trial.params, X_all, y, sample_weights, is_sparse, tf_name,
                max_hold, parent_ds=_parent_ds, use_gpu=search_use_gpu,
            )
            val_results.append({
                'trial_number': trial.number,
                'phase1_mlogloss': float(trial.value),
                'validation_mlogloss': val_score,
                'params': trial.params,
            })
            log.info(f"    Trial #{trial.number}: validation mlogloss={val_score:.4f} "
                     f"(Phase 1: {trial.value:.4f})")
            if val_score < best_val_score:
                best_val_score = val_score
                best_params = trial.params.copy()

    val_elapsed = time.time() - val_start
    log.info(f"  Validation Gate done in {val_elapsed:.0f}s: "
             f"best validation mlogloss={best_val_score:.4f}")
    log.info(f"  Winner params: {best_params}")

    # ═══════════════════════════════════════════════════════════
    # FINAL RETRAIN with best config
    # ═══════════════════════════════════════════════════════════
    _tf_final_rounds = OPTUNA_TF_FINAL_ROUNDS.get(tf_name, OPTUNA_FINAL_ROUNDS)
    log.info(f"\n  FINAL RETRAIN: best params with lr={OPTUNA_FINAL_LR}, rounds={_tf_final_rounds}")
    final_start = time.time()

    final_result = final_retrain(
        X_all, y, sample_weights, feature_cols, is_sparse,
        tf_name, max_hold, best_params,
        use_gpu=final_use_gpu, parent_ds=_parent_ds,
    )
    final_elapsed = time.time() - final_start

    log.info(f"  Final retrain done in {final_elapsed:.0f}s: "
             f"mean_acc={final_result['mean_accuracy']:.4f} "
             f"mean_sortino={final_result['mean_sortino']:.2f}")

    # Save best model
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
        'best_params': best_params,
        'phase1_best_value': float(best_p1.value),
        'validation_best_value': best_val_score,
        'validation_results': val_results,
        'final_mean_accuracy': final_result['mean_accuracy'],
        'final_mean_sortino': final_result['mean_sortino'],
        'n_features': len(feature_cols),
        'n_valid_samples': int(n_valid),
        'phase1_time': phase1_elapsed,
        'validation_time': val_elapsed,
        'final_time': final_elapsed,
        'warmstart_from': warmstart_parent_tf,
        'phase1_trials': phase1_trials,
        'validation_top_k': validation_top_k,
        'row_subsample': row_subsample,
    }
    with open(config_path, 'w') as f:
        json.dump(config_out, f, indent=2)
    log.info(f"  Config saved: {config_path}")

    # Stop runtime monitor
    if _rt_monitor is not None:
        try:
            _rt_monitor.stop()
            _rt_monitor.report()
        except Exception:
            pass

    total_elapsed = time.time() - total_start
    return {
        'tf': tf_name,
        'best_params': best_params,
        'phase1_best_value': float(best_p1.value),
        'validation_best_value': best_val_score,
        'final_mean_accuracy': final_result['mean_accuracy'],
        'final_mean_sortino': final_result['mean_sortino'],
        'total_time': total_elapsed,
    }


# ============================================================
# CLI ENTRYPOINT
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Phase 1 + Validation Gate Optuna LightGBM hyperparameter search')
    parser.add_argument('--tf', type=str, nargs='*', default=None,
                        help='Timeframes to search (default: all)')
    parser.add_argument('--n-jobs', type=int, default=None,
                        help='Parallel Optuna trials (default: cpu_count // threads_per_trial)')
    parser.add_argument('--no-warmstart', action='store_true',
                        help='Disable warm-start from parent TF (use full wide ranges + full trial count)')
    args = parser.parse_args()

    timeframes = args.tf if args.tf else TF_ORDER
    use_warmstart = not args.no_warmstart

    # Pre-flight validation — deterministic checks before spending compute
    for _tf in timeframes:
        try:
            import subprocess as _sp
            _sp.check_call([sys.executable, os.path.join(PROJECT_DIR, 'validate.py'), '--tf', _tf, '--local'])
        except _sp.CalledProcessError:
            log.error(f"Pre-flight validation FAILED for {_tf}. Fix issues above before training.")
            sys.exit(1)

    # Auto-detect parallel trials
    total_cores = get_cpu_count() or 24

    # Multi-GPU auto-detection for n_jobs
    _gpu_cfg = get_multi_gpu_config(total_cores)

    if args.n_jobs is not None:
        n_jobs = args.n_jobs
    elif _gpu_cfg.enabled and _gpu_cfg.num_gpus > 0:
        n_jobs = _gpu_cfg.n_jobs
        log.info(f"  Multi-GPU: auto-set n_jobs={n_jobs} (1 trial per GPU)")
    else:
        n_jobs = OPTUNA_N_JOBS if OPTUNA_N_JOBS > 0 else max(1, total_cores // 8)

    log.info(f"Optuna LightGBM Search v3.3 (Phase 1 + Validation Gate)")
    # FIX #18: Cap num_threads to prevent oversubscription (total_cores / n_parallel_workers)
    _threads_per_trial = max(1, total_cores // n_jobs)
    if n_jobs * _threads_per_trial > total_cores:
        _threads_per_trial = max(1, total_cores // n_jobs)
    log.info(f"  Cores: {total_cores}, Parallel trials: {n_jobs}, Threads/trial: {_threads_per_trial}")
    if _gpu_cfg.enabled:
        log.info(f"  GPUs: {_gpu_cfg.num_gpus} ({', '.join(_gpu_cfg.gpu_names[:4])})")
    log.info(f"  Timeframes: {timeframes}")
    log.info(f"  Warm-start: {'ENABLED (cascade: 1w->1d->4h->1h->15m)' if use_warmstart else 'DISABLED'}")
    log.info(f"  Phase 1 (cold): per-TF {OPTUNA_TF_PHASE1_TRIALS} | (warm): {OPTUNA_WARMSTART_PHASE1_TRIALS} trials")
    log.info(f"  Phase 1: {OPTUNA_PHASE1_CPCV_GROUPS}-fold CPCV, "
             f"lr={OPTUNA_PHASE1_LR}, ES={OPTUNA_PHASE1_ES_PATIENCE}, "
             f"rounds={OPTUNA_PHASE1_ROUNDS}")
    log.info(f"  Validation: top-{OPTUNA_VALIDATION_TOP_K} (cold) / "
             f"top-{OPTUNA_WARMSTART_VALIDATION_TOP_K} (warm), "
             f"{OPTUNA_VALIDATION_CPCV_GROUPS}-fold CPCV, "
             f"lr={OPTUNA_VALIDATION_LR}, rounds={OPTUNA_VALIDATION_ROUNDS}")
    log.info(f"  Final: full CPCV, lr={OPTUNA_FINAL_LR}, rounds={OPTUNA_FINAL_ROUNDS} (TF overrides: {OPTUNA_TF_FINAL_ROUNDS})")
    log.info(f"  Row subsample: {OPTUNA_TF_ROW_SUBSAMPLE}")

    all_results = {}
    total_start = time.time()

    for tf in timeframes:
        try:
            result = run_search_for_tf(tf, n_jobs=n_jobs, warmstart=use_warmstart)
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
        log.info(f"  {tf}: p1={r.get('phase1_best_value', 'N/A'):.4f} "
                 f"val={r.get('validation_best_value', 'N/A'):.4f} "
                 f"acc={r.get('final_mean_accuracy', 0):.4f} "
                 f"sortino={r.get('final_mean_sortino', 0):.2f} "
                 f"({r.get('total_time', 0):.0f}s)")
    log.info(f"Results saved to {summary_path}")


if __name__ == '__main__':
    main()
