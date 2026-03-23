#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ml_multi_tf.py -- Multi-Timeframe ML Trading System (v4)
=======================================================
Pipeline:
1. HMM re-fitted per walk-forward window (no future leakage)
2. Triple-barrier labels: LONG(2)/FLAT(1)/SHORT(0) via ATR barriers
3. Rolling windows (not expanding) -- better for crypto regime drift
4. Stronger regularization for 15m/5m (max_depth=3-4, lambda=5-10)
5. Nested validation: GA on inner fold, final metrics on outer untouched fold
6. ALL features used -- no SHAP pruning. XGBoost handles feature selection via tree splits.
7. multi:softprob objective with 3 classes (long_prob, flat_prob, short_prob)
"""

import sys, os, io, time, math, random, warnings, json, pickle
# DO NOT re-wrap stdout — it kills python -u unbuffered mode on cloud
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, io.UnsupportedOperation):
    pass  # Already configured or not a TextIOWrapper
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime
from numba import njit

DB_DIR = os.environ.get('SAVAGE22_DB_DIR', os.path.dirname(os.path.abspath(__file__)))
START_TIME = time.time()
RESULTS = []

def elapsed():
    return f"[{time.time()-START_TIME:.0f}s]"

def log(msg):
    print(msg)
    RESULTS.append(msg)

def compute_rsi_arr(close_arr, period=14):
    delta = np.diff(close_arr, prepend=close_arr[0])
    gain = np.maximum(delta, 0)
    loss = np.maximum(-delta, 0)
    alpha = 1.0 / period
    avg_gain = pd.Series(gain).ewm(alpha=alpha, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=alpha, min_periods=period).mean().values
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    return 100 - (100 / (1 + rs))

# ============================================================
# INSTALL DEPS IF NEEDED
# ============================================================
try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    os.system("pip install hmmlearn")
    from hmmlearn.hmm import GaussianHMM

import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, log_loss
from scipy import stats
from scipy import sparse as sp_sparse
from feature_library import compute_triple_barrier_labels, TRIPLE_BARRIER_CONFIG

# Dask multi-GPU support (optional)
try:
    from dask_cuda import LocalCUDACluster
    from dask.distributed import Client
    import dask.array as da
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

# Check GPU
USE_GPU = False
USE_GPU_XGB = False
N_GPUS = 0
try:
    test_dmat = xgb.DMatrix(np.random.rand(10, 5), label=np.random.randint(0, 2, 10), nthread=-1)
    bst = xgb.train({'tree_method': 'hist', 'device': 'cuda', 'max_depth': 3}, test_dmat, num_boost_round=2)
    USE_GPU = True
    USE_GPU_XGB = True
    try:
        from hardware_detect import detect_hardware
        _hw = detect_hardware()
        N_GPUS = _hw['n_gpus'] or 1
    except Exception:
        N_GPUS = 1
except:
    pass
log(f"GPU: {'ENABLED' if USE_GPU else 'CPU only'} ({N_GPUS} device{'s' if N_GPUS != 1 else ''})")
log(f"Dask multi-GPU: {'AVAILABLE' if DASK_AVAILABLE else 'not installed (pip install dask-cuda)'}")


_dask_client = None
_dask_cluster = None

def _get_dask_client():
    """Get or create singleton Dask CUDA cluster"""
    global _dask_client, _dask_cluster
    if _dask_client is not None:
        try:
            _dask_client.scheduler_info()  # check if still alive
            return _dask_client
        except Exception:
            _dask_client = None
            _dask_cluster = None

    try:
        _dask_cluster = LocalCUDACluster(
            n_workers=None,  # auto-detect GPUs
            device_memory_limit='70GB',  # leave headroom on 80GB H100
        )
        _dask_client = Client(_dask_cluster)
        n_workers = len(_dask_client.scheduler_info()['workers'])
        print(f"  [Dask] Cluster ready: {n_workers} GPU workers")
        return _dask_client
    except Exception as e:
        print(f"  [Dask] Cluster creation failed: {e}")
        if _dask_cluster is not None:
            try:
                _dask_cluster.close()
            except:
                pass
        _dask_cluster = None
        _dask_client = None
        return None


def _shutdown_dask_client():
    global _dask_client, _dask_cluster
    if _dask_client is not None:
        try:
            _dask_client.close()
        except:
            pass
        _dask_client = None
    if _dask_cluster is not None:
        try:
            _dask_cluster.close()
        except:
            pass
        _dask_cluster = None
    print("  [Dask] Cluster shutdown complete")


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


def _cpcv_split_worker(args_tuple):
    """Train a single XGBoost CPCV split on a specific GPU.
    Runs in a subprocess for multi-GPU parallelism.
    Returns: (wi, acc, prec_long, prec_short, mlogloss, best_iter,
              model_bytes, preds_3c, y_test, test_idx_valid,
              importance, is_acc, is_mlogloss, is_sharpe)
    """
    (wi, train_idx, test_idx, X_data, X_indices, X_indptr, X_shape,
     y_3class, sample_weights, feature_cols, xgb_params,
     num_boost_round, tf_name, gpu_id) = args_tuple

    import numpy as np
    from scipy import sparse
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, precision_score, log_loss

    # Reconstruct sparse matrix in worker
    X_all = sparse.csr_matrix((X_data, X_indices, X_indptr), shape=X_shape)

    y_train_raw = y_3class[train_idx]
    y_test_raw = y_3class[test_idx]
    train_valid = ~np.isnan(y_train_raw)
    test_valid = ~np.isnan(y_test_raw)

    X_train = X_all[train_idx][train_valid]
    y_train = y_train_raw[train_valid].astype(int)
    X_test = X_all[test_idx][test_valid]
    y_test = y_test_raw[test_valid].astype(int)
    test_idx_valid = test_idx[test_valid]

    min_train = 50 if tf_name in ('1w', '1d') else 300
    min_test = 20 if tf_name in ('1w', '1d') else 50
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    if n_train < min_train or n_test < min_test:
        return (wi, None, None, None, None, None, None, None, None, None, None, None, None, None)

    w_train = sample_weights[train_idx][train_valid]

    params = xgb_params.copy()
    params['device'] = f'cuda:{gpu_id}'

    # 85/15 train/val split for early stopping
    val_size = max(int(n_train * 0.15), 100)
    if val_size >= n_train:
        val_size = max(n_train // 5, 20)
    X_val_es = X_train[-val_size:]
    y_val_es = y_train[-val_size:]
    w_val_es = w_train[-val_size:]
    X_train_es = X_train[:-val_size]
    y_train_es = y_train[:-val_size]
    w_train_es = w_train[:-val_size]

    dtrain = xgb.DMatrix(X_train_es, label=y_train_es, weight=w_train_es,
                         feature_names=feature_cols, nthread=-1)
    dval = xgb.DMatrix(X_val_es, label=y_val_es, feature_names=feature_cols, nthread=-1)
    model = xgb.train(
        params, dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=0,
    )

    # OOS predictions
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols, nthread=-1)
    preds_3c = model.predict(dtest)
    pred_labels = np.argmax(preds_3c, axis=1)
    acc = float(accuracy_score(y_test, pred_labels))
    prec_long = float(precision_score(y_test, pred_labels, labels=[2], average='macro', zero_division=0))
    prec_short = float(precision_score(y_test, pred_labels, labels=[0], average='macro', zero_division=0))
    mlogloss = float(log_loss(y_test, preds_3c, labels=[0, 1, 2]))

    # IS metrics for PBO
    dtrain_is = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols, nthread=-1)
    is_preds_3c = model.predict(dtrain_is)
    is_pred_labels = np.argmax(is_preds_3c, axis=1)
    is_acc = float(accuracy_score(y_train, is_pred_labels))
    is_mlogloss = float(log_loss(y_train, is_preds_3c, labels=[0, 1, 2]))
    _is_sim_ret = np.where(is_pred_labels == y_train, 1.0, -1.0)
    _is_std = np.std(_is_sim_ret, ddof=1)
    is_sharpe = float(np.mean(_is_sim_ret) / max(_is_std, 1e-10) * np.sqrt(252))

    importance = model.get_score(importance_type='gain')

    # Serialize model
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix='.ubj', delete=False)
    tmp.close()
    model.save_model(tmp.name)
    with open(tmp.name, 'rb') as f:
        model_bytes = f.read()
    import os as _os
    _os.unlink(tmp.name)

    return (wi, acc, prec_long, prec_short, mlogloss, model.best_iteration,
            model_bytes, preds_3c.copy(), y_test.copy(), test_idx_valid.copy(),
            importance, is_acc, is_mlogloss, is_sharpe)


def _train_xgb_dask(client, params, X_train, y_train, X_test, y_test,
                     w_train=None, feature_names=None,
                     num_boost_round=800, early_stopping_rounds=50):
    """Train XGBoost via Dask multi-GPU, with fallback to single GPU."""
    try:
        # Dask arrays don't support scipy sparse — fall back to single GPU
        if sp_sparse.issparse(X_train):
            log(f"    [Dask] Skipping — sparse matrices not supported by Dask, using single GPU")
            return None
        n_workers = len(client.scheduler_info()['workers'])

        # Split training data into 85% train + 15% validation for early stopping
        n_train = X_train.shape[0]
        val_size = max(int(n_train * 0.15), 100)
        X_val_es = X_train[-val_size:]
        y_val_es = y_train[-val_size:]
        X_train_es = X_train[:-val_size]
        y_train_es = y_train[:-val_size]
        w_train_es = w_train[:-val_size] if w_train is not None else None

        chunk_rows = max(10000, X_train_es.shape[0] // n_workers)

        X_train_da = da.from_array(X_train_es, chunks=(chunk_rows, X_train_es.shape[1]))
        y_train_da = da.from_array(y_train_es, chunks=(chunk_rows,))
        X_val_da = da.from_array(X_val_es, chunks=(X_val_es.shape[0], X_val_es.shape[1]))
        y_val_da = da.from_array(y_val_es, chunks=(X_val_es.shape[0],))

        if w_train_es is not None:
            w_train_da = da.from_array(w_train_es, chunks=(chunk_rows,))
            dtrain = xgb.dask.DaskQuantileDMatrix(client, X_train_da, y_train_da, weight=w_train_da)
        else:
            dtrain = xgb.dask.DaskQuantileDMatrix(client, X_train_da, y_train_da)

        dval = xgb.dask.DaskDMatrix(client, X_val_da, y_val_da)

        output = xgb.dask.train(
            client, params, dtrain,
            num_boost_round=num_boost_round,
            evals=[(dval, 'val')],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=100,
        )
        if isinstance(output, dict):
            model = output.get('booster')
        else:
            model = output
        if model is None:
            print(f"    [Dask] Training returned no booster")
            return None
        if feature_names and hasattr(model, 'feature_names'):
            model.feature_names = feature_names
        return model
    except Exception as e:
        log(f"    WARNING: Dask training failed ({e}), falling back to single GPU")
        return None

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
          except:
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
  TF_CONFIGS = {
      # ═══════════════════════════════════════════════════════════════
      # v4 INSTITUTIONAL PARAMS — audited by Perplexity for 150K features
      # Triple colsample cascade: bytree × bylevel × bynode = ~7.5% effective
      # min_child_weight=50: prevent splits on <50 samples (critical for sparse dx_)
      # reg_alpha=5.0 (L1): prune marginal sparse splits
      # learning_rate=0.03: slower learning = more robust
      # Based on production 100K+ sparse feature systems (ads/recs/ranking)
      # ═══════════════════════════════════════════════════════════════
      '1w': {
          'db': 'features_1w.db', 'table': 'features_1w',
          'return_col': 'next_1w_return',
          'cost_pct': 0.0025,
          'xgb': {'max_depth': 3, 'min_child_weight': 10, 'subsample': 0.7,
                  'colsample_bytree': 0.30, 'colsample_bylevel': 0.50, 'colsample_bynode': 0.50,
                  'reg_lambda': 10.0, 'reg_alpha': 5.0, 'gamma': 2.0,
                  'learning_rate': 0.03},
          'context_only': False,
          'rolling_window_bars': None,  # use all available (339 rows)
      },
      '1d': {
          'db': 'features_1d.db', 'table': 'features_1d',
          'return_col': 'next_1d_return',
          'cost_pct': 0.0025,
          'xgb': {'max_depth': 3, 'min_child_weight': 10, 'subsample': 0.7,
                  'colsample_bytree': 0.30, 'colsample_bylevel': 0.50, 'colsample_bynode': 0.50,
                  'reg_lambda': 10.0, 'reg_alpha': 5.0, 'gamma': 2.0,
                  'learning_rate': 0.03},
          'context_only': False,
          'rolling_window_bars': None,  # use all available (2368 rows)
      },
      '4h': {
          'db': 'features_4h.db', 'table': 'features_4h',
          'return_col': 'next_4h_return',
          'cost_pct': 0.24,
          'xgb': {'max_depth': 4, 'min_child_weight': 20, 'subsample': 0.7,
                  'colsample_bytree': 0.30, 'colsample_bylevel': 0.50, 'colsample_bynode': 0.50,
                  'reg_lambda': 10.0, 'reg_alpha': 5.0, 'gamma': 2.0,
                  'learning_rate': 0.03},
          'context_only': False,
          'rolling_window_bars': 8760,  # ~18 months of 4H bars
      },
      '1h': {
          'db': 'features_1h.db', 'table': 'features_1h',
          'return_col': 'next_1h_return',
          'cost_pct': 0.23,
          'xgb': {'max_depth': 4, 'min_child_weight': 25, 'subsample': 0.7,
                  'colsample_bytree': 0.30, 'colsample_bylevel': 0.50, 'colsample_bynode': 0.50,
                  'reg_lambda': 10.0, 'reg_alpha': 5.0, 'gamma': 2.0,
                  'learning_rate': 0.03},
          'context_only': False,
          'rolling_window_bars': 25000,  # ~2.8 years of 1H bars
      },
      '15m': {
          'db': 'features_15m.db', 'table': 'features_15m',
          'return_col': 'next_15m_return',
          'cost_pct': 0.22,
          'xgb': {'max_depth': 4, 'min_child_weight': 50, 'subsample': 0.7,
                  'colsample_bytree': 0.30, 'colsample_bylevel': 0.50, 'colsample_bynode': 0.50,
                  'reg_lambda': 10.0, 'reg_alpha': 5.0, 'gamma': 2.0,
                  'learning_rate': 0.03},
          'context_only': False,
          'rolling_window_bars': 70000,  # ~6 months of 15m bars
      },
      '5m': {
          'db': 'features_5m.db', 'table': 'features_5m',
          'return_col': 'next_5m_return',
          'cost_pct': 0.22,
          'xgb': {'max_depth': 3, 'min_child_weight': 50, 'subsample': 0.6,
                  'colsample_bytree': 0.30, 'colsample_bylevel': 0.50, 'colsample_bynode': 0.50,
                  'reg_lambda': 10.0, 'reg_alpha': 5.0, 'gamma': 2.0,
                  'learning_rate': 0.03},
          'context_only': False,
          'rolling_window_bars': 210000,  # ~6 months of 5m bars
      },
  }

  # GA replaced by exhaustive_optimizer.py — these ranges kept for reference only
  # GA_RANGES = {
  #     '4h': {'lev': (1, 25), 'risk': (0.5, 3.0), 'stop_atr': (0.5, 2.0), 'rr': (1, 4), 'hold': (1, 32), 'ptp': [0, 25, 50, 75], 'conf': (0.55, 0.85)},
  #     '1h': {'lev': (5, 50), 'risk': (0.1, 1.0), 'stop_atr': (0.3, 1.0), 'rr': (1, 3), 'hold': (1, 8), 'ptp': [0, 25, 50, 75], 'conf': (0.55, 0.85)},
  #     '15m': {'lev': (10, 60), 'risk': (0.05, 0.5), 'stop_atr': (0.2, 0.5), 'rr': (1, 2), 'hold': (1, 8), 'ptp': [0, 25, 50, 75], 'conf': (0.55, 0.85)},
  #     '5m': {'lev': (15, 75), 'risk': (0.05, 0.3), 'stop_atr': (0.1, 0.3), 'rr': (1, 1.5), 'hold': (1, 12), 'ptp': [0, 25, 50, 75], 'conf': (0.55, 0.85)},
  # }

  # ============================================================
  # CLI ARGS
  # ============================================================
  import argparse
  from concurrent.futures import ProcessPoolExecutor
  _parser = argparse.ArgumentParser()
  _parser.add_argument('--tf', action='append', help='Only train specific TFs (can repeat)')
  _parser.add_argument('--boost-rounds', type=int, default=800, help='XGBoost num_boost_round (default 800)')
  _parser.add_argument('--n-groups', type=int, default=None, help='Override CPCV n_groups (default: per-TF)')
  _parser.add_argument('--parallel-splits', action='store_true', default=False,
                        help='(legacy, now auto-detected) Kept for backward compat')
  _parser.add_argument('--no-parallel-splits', action='store_true', default=False,
                        help='Force sequential CPCV splits even with multiple GPUs')
  _args, _unknown = _parser.parse_known_args()  # ignore unknown args (e.g. from smoke_test)
  _tf_filter = set(_args.tf) if _args.tf else None

  # Auto-detect parallel splits: ON by default when N_GPUS > 1
  if _args.no_parallel_splits:
      _use_parallel_splits = False
      log("PARALLEL SPLITS: disabled (--no-parallel-splits)")
  elif USE_GPU and N_GPUS > 1:
      _use_parallel_splits = True
      log(f"PARALLEL SPLITS: auto-enabled across {N_GPUS} GPUs (use --no-parallel-splits to disable)")
  else:
      _use_parallel_splits = False
      if N_GPUS <= 1:
          log("PARALLEL SPLITS: off (single GPU detected)")

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
          if not os.path.exists(parquet_path) and os.path.exists(v2_parquet):
              parquet_path = v2_parquet

          # Try parquet first (no column limit), fall back to SQLite
          if os.path.exists(parquet_path):
              df = pd.read_parquet(parquet_path)
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
              log(f"  SKIP — {cfg['db']} not found")
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
          # y_3class uses same encoding for XGBoost multi:softprob (num_class=3)
          y_3class = tb_labels.copy()
          valid_mask = ~np.isnan(y_3class)
          n_long = (y_3class == 2).sum()
          n_short = (y_3class == 0).sum()
          n_flat = (y_3class == 1).sum()
          n_nan = (~valid_mask).sum()
          tb_cfg = TRIPLE_BARRIER_CONFIG.get(tf_name, TRIPLE_BARRIER_CONFIG['1h'])
          log(f"  Triple-barrier labels (tp={tb_cfg['tp_atr_mult']}xATR, sl={tb_cfg['sl_atr_mult']}xATR, hold={tb_cfg['max_hold_bars']}): "
              f"{int(n_long)} LONG, {int(n_short)} SHORT, {int(n_flat)} FLAT, {int(n_nan)} NaN")

          # Also keep old return_col for backward compat logging
          return_col = cfg['return_col']
          if return_col not in df.columns:
              candidates = [c for c in df.columns if 'next' in c.lower() and 'return' in c.lower()]
              if candidates:
                  return_col = candidates[0]

          # Regime-aware sample weights: downweight counter-trend trades (0.15x, Perplexity validated)
          sample_weights = np.ones(len(y_3class), dtype=np.float32)
          if 'ema50_declining' in df.columns and 'ema50_rising' in df.columns:
              ema_dec = pd.to_numeric(df['ema50_declining'], errors='coerce').values
              ema_ris = pd.to_numeric(df['ema50_rising'], errors='coerce').values
              bear_longs = (y_3class == 2) & (ema_dec == 1)  # LONG in bear regime
              bull_shorts = (y_3class == 0) & (ema_ris == 1)  # SHORT in bull regime
              sample_weights[bear_longs] = 0.15
              sample_weights[bull_shorts] = 0.15
              log(f"  Regime weights: {bear_longs.sum()} bear LONGs @ 0.15, {bull_shorts.sum()} bull SHORTs @ 0.15")
          else:
              log(f"  WARNING: ema50_declining/ema50_rising not in features -- no regime weighting applied")

          # Identify feature columns
          meta_cols = {'timestamp', 'date', 'open', 'high', 'low', 'close', 'volume',
                       'quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote', 'open_time',
                       'date_norm'}
          target_like = {c for c in df.columns if 'next_' in c.lower() or 'target' in c.lower()
                         or 'direction' in c.lower() or c == 'triple_barrier_label'}
          exclude_cols = meta_cols | target_like
          feature_cols = [c for c in df.columns if c not in exclude_cols]

          # --- Sparse matrix support for 150K+ features ---
          # Load base features as dense (typically a few hundred to a few thousand cols)
          X_base = df[feature_cols].values.astype(np.float32)
          # Fix inf values which would break training (NaN kept for XGBoost missing branches)
          X_base = np.where(np.isinf(X_base), np.nan, X_base)
          n_base_features = len(feature_cols)

          # Check for sparse cross .npz file for this TF
          cross_matrix = None
          cross_cols = None
          npz_path = os.path.join(DB_DIR, f'v2_crosses_BTC_{tf_name}.npz')
          if os.path.exists(npz_path):
              try:
                  log(f"  {elapsed()} Loading sparse cross matrix: {npz_path}")
                  cross_matrix = sp_sparse.load_npz(npz_path).tocsr()
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
          if cross_matrix is not None and cross_matrix.shape[0] == X_base.shape[0]:
              # Convert base to sparse PRESERVING NaN (XGBoost treats NaN as missing natively)
              # Do NOT use nan_to_num — that would:
              #   1. Change semantics: XGBoost missing-value handling learns split directions for NaN
              #   2. Bloat storage: explicit 0s get stored in sparse matrix, defeating the point
              X_base_sparse = sp_sparse.csr_matrix(X_base)  # NaN stored as explicit entries, true zeros are structural
              X_all = sp_sparse.hstack([X_base_sparse, cross_matrix], format='csr')
              # DO NOT call eliminate_zeros() — it converts explicit 0.0 values into
              # structural zeros, which XGBoost treats as "missing." 0.0 means "the value
              # is zero" (a real signal), not "data is absent." NaN already encodes missing.
              feature_cols = feature_cols + cross_cols
              _X_all_is_sparse = True
              nnz = X_all.nnz
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

          timestamps = df['timestamp'].values if 'timestamp' in df.columns else np.arange(len(df))
          closes = pd.to_numeric(df['close'], errors='coerce').values

          log(f"  Features: {len(feature_cols)} ({'SPARSE' if _X_all_is_sparse else 'DENSE'})")

          # Event-aware weights: upweight bars where esoteric signals are active
          ESOTERIC_KEYWORDS = [
              'gem_', 'dr_', 'moon', 'nakshatra', 'vedic', 'bazi', 'tzolkin', 'arabic',
              'tweet', 'sport', 'horse', 'caution', 'cross_', 'eclipse', 'retro', 'shmita',
              'gold_tweet', 'red_tweet', 'misdirection', 'planetary', 'lot_', 'hebrew',
              'fibonacci', 'gann', 'tesla_369', 'master_', 'contains_113', 'contains_322',
              'contains_93', 'contains_213', 'contains_666', 'contains_777', 'friday_13',
              'palindrome', 'fg_x_', 'onchain_', 'macro_', 'headline_', 'news_sentiment',
              'sentiment_mean', 'caps_', 'excl_', 'upset', 'overtime',
          ]
          esoteric_col_mask = np.array([any(kw in col for kw in ESOTERIC_KEYWORDS) for col in feature_cols])
          if esoteric_col_mask.sum() > 0:
              # For esoteric weight computation, use only base feature columns (first n_base_features)
              # to avoid densifying the huge cross matrix
              esoteric_base_mask = esoteric_col_mask[:n_base_features]
              if esoteric_base_mask.sum() > 0:
                  if _X_all_is_sparse:
                      esoteric_indices = np.where(esoteric_base_mask)[0]
                      X_esoteric = X_all[:, esoteric_indices].toarray()
                  else:
                      X_esoteric = X_all[:, esoteric_base_mask]
                  esoteric_active = np.sum(~np.isnan(X_esoteric) & (X_esoteric != 0), axis=1)
                  esoteric_weight = np.clip(1.0 + 0.5 * np.minimum(esoteric_active, 4), 1.0, 3.0)
                  sample_weights *= esoteric_weight.astype(np.float32)
                  bars_with_esoteric = (esoteric_active > 0).sum()
                  n_rows = X_all.shape[0]
                  log(f"  Esoteric weights: {esoteric_base_mask.sum()} esoteric columns, "
                      f"{bars_with_esoteric}/{n_rows} bars have active signals, "
                      f"weight range [{esoteric_weight.min():.1f}, {esoteric_weight.max():.1f}]")

          # Count NaN density to verify esoteric features are sparse (not all zeros)
          if _X_all_is_sparse:
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
          n = len(df)
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
          if _args.n_groups is not None:
              n_groups = _args.n_groups
              n_test_groups = 1
              log(f"  CPCV override: n_groups={n_groups}, n_test=1 (--n-groups flag)")
          elif tf_name in ('1w', '1d'):
              n_groups = 4
              n_test_groups = 1
          elif tf_name == '4h':
              n_groups = 5
              n_test_groups = 2
          else:
              n_groups = 6
              n_test_groups = 2

          cpcv_splits = _generate_cpcv_splits(
              n, n_groups=n_groups, n_test_groups=n_test_groups,
              max_hold_bars=max_hold, embargo_pct=0.01,
          )

          # Convert CPCV splits to (train_start, train_end, test_start, test_end) format
          # for compatibility with existing training loop
          splits = []
          for train_idx, test_idx in cpcv_splits:
              splits.append((train_idx, test_idx))

          log(f"  {elapsed()} CPCV: {len(splits)} paths (N={n_groups} groups, K={n_test_groups} test, purge={max_hold} bars, embargo=1%)")
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

          # Build XGBoost params once (shared by all paths)
          _base_xgb_params = {**cfg['xgb'], 'objective': 'multi:softprob', 'num_class': 3,
                              'eval_metric': 'mlogloss', 'verbosity': 0,
                              'tree_method': 'hist'}
          if USE_GPU:
              _base_xgb_params['device'] = 'cuda'

          # Interaction constraints (compute once, used by all paths)
          _doy_names = [f for f in feature_cols if f.startswith('doy_')]
          if _doy_names:
              _trend_kw = ('regime', 'ema50', 'bull', 'bear', 'hmm_', 'trend')
              _ta_kw = ('rsi_', 'macd', 'bb_', 'atr_', 'sma_', 'ema_', 'adx_', 'stoch_', 'obv', 'vwap', 'cci_', 'mfi_', 'williams', 'ichimoku', 'keltner', 'donchian', 'supertrend', 'sar_')
              _trend_names = [f for f in feature_cols if any(kw in f for kw in _trend_kw)]
              _ta_names = [f for f in feature_cols if any(kw in f for kw in _ta_kw)]
              _constrained = _doy_names + _trend_names + _ta_names
              if _constrained:
                  _base_xgb_params['interaction_constraints'] = [_constrained]
              log(f"  Interaction constraints: 1 group with DOY({len(_doy_names)}) + TREND({len(_trend_names)}) + TA({len(_ta_names)}) = {len(_constrained)} constrained, rest free")

          if _use_parallel_splits and _X_all_is_sparse:
              # ── Multi-GPU parallel CPCV path ──
              # NOTE: per-fold HMM re-fitting is skipped in parallel mode (HMM fitted once before loop).
              # This is the same tradeoff v2_multi_asset_trainer.py makes.
              log(f"\n  PARALLEL CPCV: distributing {len(splits)} splits across {N_GPUS} GPUs")
              X_csr = X_all.tocsr()
              worker_args = []
              for wi, (train_idx, test_idx) in enumerate(splits):
                  if wi in _completed_folds:
                      log(f"  Path {wi+1}/{len(splits)}: SKIP (checkpoint)")
                      continue
                  gpu_id = wi % N_GPUS
                  worker_args.append((
                      wi, train_idx, test_idx,
                      X_csr.data, X_csr.indices, X_csr.indptr, X_csr.shape,
                      y_3class, sample_weights, feature_cols, _base_xgb_params,
                      _args.boost_rounds, tf_name, gpu_id
                  ))

              with ProcessPoolExecutor(max_workers=N_GPUS) as executor:
                  for result in executor.map(_cpcv_split_worker, worker_args):
                      (wi, acc, prec_long, prec_short, mlogloss_val, best_iter,
                       model_bytes, preds_3c, y_test, test_idx_valid,
                       importance, is_acc, is_mlogloss, is_sharpe) = result

                      if acc is None:
                          log(f"  Path {wi+1}/{len(splits)} (GPU {wi % N_GPUS}): SKIP -- not enough samples")
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
                      log(f"  Path {wi+1}/{len(splits)} (GPU {wi % N_GPUS}): "
                          f"Acc={acc:.3f} PrecL={prec_long:.3f} PrecS={prec_short:.3f} mlogloss={mlogloss_val:.4f} Trees={best_iter}")

                      if acc > best_acc:
                          best_acc = acc
                          # Deserialize best model
                          import tempfile as _tmpmod
                          _tmp = _tmpmod.NamedTemporaryFile(suffix='.ubj', delete=False)
                          _tmp.write(model_bytes)
                          _tmp.close()
                          best_model_obj = xgb.Booster()
                          best_model_obj.load_model(_tmp.name)
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

          else:
              # ── Sequential CPCV path (single GPU or dense matrix) ──
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
                          # Batch HMM column update: ONE format conversion instead of 4
                          _hmm_update_indices = []
                          _hmm_update_values = []
                          _hmm_append_cols = []
                          for hmm_col in ['hmm_bull_prob', 'hmm_bear_prob', 'hmm_neutral_prob', 'hmm_state']:
                              hmm_mapped = pd.Series(date_norm).map(
                                  hmm_df_notz[hmm_col].to_dict()
                              ).ffill().values.astype(np.float32)
                              col_idx = feature_cols.index(hmm_col) if hmm_col in feature_cols else -1
                              if col_idx >= 0:
                                  _hmm_update_indices.append(col_idx)
                                  _hmm_update_values.append(hmm_mapped)
                              else:
                                  _hmm_append_cols.append((hmm_col, hmm_mapped))
                          # Apply batched column updates (single tolil/tocsr round-trip)
                          if _hmm_update_indices:
                              if _X_all_is_sparse:
                                  X_lil = X_all.tolil()
                                  for ci, cv in zip(_hmm_update_indices, _hmm_update_values):
                                      X_lil[:, ci] = cv.reshape(-1, 1)
                                  X_all = X_lil.tocsr()
                                  del X_lil
                              else:
                                  for ci, cv in zip(_hmm_update_indices, _hmm_update_values):
                                      X_all[:, ci] = cv
                          # Append any new HMM columns
                          for hmm_col, hmm_mapped in _hmm_append_cols:
                              if _X_all_is_sparse:
                                  hmm_sparse = sp_sparse.csr_matrix(hmm_mapped.reshape(-1, 1))
                                  X_all = sp_sparse.hstack([X_all, hmm_sparse], format='csr')
                              else:
                                  X_all = np.column_stack([X_all, hmm_mapped])
                              feature_cols.append(hmm_col)

                  # Extract train/test using CPCV index arrays
                  y_train_raw = y_3class[train_idx]
                  y_test_raw = y_3class[test_idx]

                  # Filter out NaN labels
                  train_valid = ~np.isnan(y_train_raw)
                  test_valid = ~np.isnan(y_test_raw)

                  # Extract train/test — sparse indexing uses same [idx] syntax
                  X_train = X_all[train_idx][train_valid]
                  y_train = y_train_raw[train_valid].astype(int)
                  X_test = X_all[test_idx][test_valid]
                  y_test = y_test_raw[test_valid].astype(int)
                  test_idx_valid = test_idx[test_valid]  # for OOS prediction storage

                  # NO pre-filtering: XGBoost decides via tree splits, not us.
                  _fold_feature_cols = feature_cols

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

                  params = _base_xgb_params.copy()

                  w_train = sample_weights[train_idx][train_valid]

                  # Split training fold into 85% train + 15% validation for early stopping
                  # (never use test set for model selection — preserves OOS integrity)
                  n_tr = X_train.shape[0]
                  val_size = max(int(n_tr * 0.15), 100)
                  if val_size >= n_tr:
                      val_size = max(n_tr // 5, 20)  # fallback for tiny folds
                  X_val_es = X_train[-val_size:]
                  y_val_es = y_train[-val_size:]
                  w_val_es = w_train[-val_size:]
                  X_train_es = X_train[:-val_size]
                  y_train_es = y_train[:-val_size]
                  w_train_es = w_train[:-val_size]

                  model = None
                  if DASK_AVAILABLE and USE_GPU_XGB:
                      client = _get_dask_client()
                      if client is not None:
                          model = _train_xgb_dask(
                              client, params, X_train_es, y_train_es, X_test, y_test,
                              w_train=w_train_es, feature_names=_fold_feature_cols,
                              num_boost_round=_args.boost_rounds, early_stopping_rounds=50,
                          )

                  if model is None:
                      dtrain = xgb.DMatrix(X_train_es, label=y_train_es, weight=w_train_es, feature_names=_fold_feature_cols, nthread=-1)
                      dval = xgb.DMatrix(X_val_es, label=y_val_es, feature_names=_fold_feature_cols, nthread=-1)
                      model = xgb.train(
                          params, dtrain,
                          num_boost_round=_args.boost_rounds,
                          evals=[(dtrain, 'train'), (dval, 'val')],
                          early_stopping_rounds=50,
                          verbose_eval=100,
                      )

                  # Predict OOS
                  dtest_pred = xgb.DMatrix(X_test, label=y_test, feature_names=_fold_feature_cols, nthread=-1)
                  preds_3c = model.predict(dtest_pred)  # shape (N, 3)
                  pred_labels = np.argmax(preds_3c, axis=1)
                  acc = accuracy_score(y_test, pred_labels)
                  prec_long = precision_score(y_test, pred_labels, labels=[2], average='macro', zero_division=0)
                  prec_short = precision_score(y_test, pred_labels, labels=[0], average='macro', zero_division=0)
                  mlogloss = log_loss(y_test, preds_3c, labels=[0, 1, 2])

                  # Evaluate IS (full training data) for proper PBO
                  dtrain_is = xgb.DMatrix(X_train, label=y_train, feature_names=_fold_feature_cols, nthread=-1)
                  is_preds_3c = model.predict(dtrain_is)
                  is_pred_labels = np.argmax(is_preds_3c, axis=1)
                  is_acc = float(accuracy_score(y_train, is_pred_labels))
                  is_mlogloss = float(log_loss(y_train, is_preds_3c, labels=[0, 1, 2]))
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
                  importance = model.get_score(importance_type='gain')

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

          if not window_results:
              log(f"  SKIP -- no valid CPCV paths")
              continue

          avg_acc = np.mean([w['accuracy'] for w in window_results])
          avg_prec_l = np.mean([w['prec_long'] for w in window_results])
          avg_prec_s = np.mean([w['prec_short'] for w in window_results])
          avg_mlogloss = np.mean([w['mlogloss'] for w in window_results])
          log(f"\n  {tf_name.upper()} CPCV AVG ({len(window_results)} paths): Acc={avg_acc:.3f} PrecL={avg_prec_l:.3f} PrecS={avg_prec_s:.3f} mlogloss={avg_mlogloss:.4f}")

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
                      rank_matrix[fold_i] = stats.rankdata(-gains)  # higher gain = lower rank

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
          # FINAL MODEL — ALL FEATURES, NO PRUNING
          # ============================================================
          log(f"\n  {elapsed()} Training final model on ALL {len(feature_cols)} features (no pruning)...")

          # Use last CPCV split for final model
          last_train_idx, last_test_idx = splits[-1]
          train_mask = ~np.isnan(y_3class[last_train_idx])
          X_tr = X_all[last_train_idx][train_mask]
          y_tr = y_3class[last_train_idx][train_mask].astype(int)
          test_mask = ~np.isnan(y_3class[last_test_idx])
          X_te = X_all[last_test_idx][test_mask]
          y_te = y_3class[last_test_idx][test_mask].astype(int)

          w_tr = sample_weights[last_train_idx][train_mask]

          final_params = {**cfg['xgb'], 'objective': 'multi:softprob', 'num_class': 3,
                          'eval_metric': 'mlogloss', 'verbosity': 0,
                          **({'tree_method': 'hist', 'device': 'cuda'} if USE_GPU else {'tree_method': 'hist'})}

          final_model = None
          if DASK_AVAILABLE and USE_GPU_XGB:
              client = _get_dask_client()
              if client is not None:
                  final_model = _train_xgb_dask(
                      client, final_params, X_tr, y_tr, X_te, y_te,
                      w_train=w_tr, feature_names=feature_cols,
                      num_boost_round=_args.boost_rounds, early_stopping_rounds=50,
                  )

          if final_model is None:
              print(f"    [Dask fallback] Using single-GPU training")
              # Split train into 85% train + 15% val for early stopping
              n_tr_final = X_tr.shape[0]
              val_sz = max(int(n_tr_final * 0.15), 100)
              if val_sz >= n_tr_final:
                  val_sz = max(n_tr_final // 5, 20)
              X_val_f = X_tr[-val_sz:]
              y_val_f = y_tr[-val_sz:]
              X_tr_f = X_tr[:-val_sz]
              y_tr_f = y_tr[:-val_sz]
              w_tr_f = w_tr[:-val_sz]
              dtrain = xgb.DMatrix(X_tr_f, label=y_tr_f, weight=w_tr_f, feature_names=feature_cols, nthread=-1)
              dval = xgb.DMatrix(X_val_f, label=y_val_f, feature_names=feature_cols, nthread=-1)
              final_model = xgb.train(
                  final_params, dtrain, num_boost_round=_args.boost_rounds,
                  evals=[(dtrain, 'train'), (dval, 'val')], early_stopping_rounds=50, verbose_eval=100,
              )

          dtest_final = xgb.DMatrix(X_te, label=y_te, feature_names=feature_cols, nthread=-1)
          final_preds_3c = final_model.predict(dtest_final)  # shape (N, 3)
          final_labels = np.argmax(final_preds_3c, axis=1)
          final_acc = accuracy_score(y_te, final_labels)
          final_prec_l = precision_score(y_te, final_labels, labels=[2], average='macro', zero_division=0)
          final_prec_s = precision_score(y_te, final_labels, labels=[0], average='macro', zero_division=0)
          final_mlogloss = log_loss(y_te, final_preds_3c, labels=[0, 1, 2])
          log(f"  FINAL: Acc={final_acc:.3f} PrecL={final_prec_l:.3f} PrecS={final_prec_s:.3f} "
              f"mlogloss={final_mlogloss:.4f} Trees={final_model.best_iteration} Features={len(feature_cols)}")

          # Log feature importance (top 30 by gain — for visibility, not pruning)
          importance = final_model.get_score(importance_type='gain')
          if importance:
              sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
              log(f"\n  TOP 30 FEATURES BY GAIN (of {len(feature_cols)} total):")
              for i, (fname, gain) in enumerate(sorted_imp[:30]):
                  log(f"    {i+1:3d}. {fname:<45s} gain={gain:.1f}")
              # Count how many esoteric features made it into top 50
              esoteric_in_top50 = sum(1 for fname, _ in sorted_imp[:50]
                                      if any(k in fname for k in ['gem_', 'dr_', 'moon', 'nakshatra', 'vedic',
                                             'bazi', 'tzolkin', 'arabic', 'tweet', 'sport', 'horse',
                                             'caution', 'cross_', 'eclipse', 'retro', 'shmita']))
              log(f"  Esoteric features in top 50 by gain: {esoteric_in_top50}")

          # KNN features kept unconditionally — XGBoost decides via tree splits, not us

          # Save model + feature list
          final_model.save_model(f'{DB_DIR}/model_{tf_name}.json')
          with open(f'{DB_DIR}/features_{tf_name}_all.json', 'w') as f:
              json.dump(feature_cols, f, indent=2)

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

          # Outer fold validation
          log(f"\n  {elapsed()} OUTER FOLD VALIDATION...")
          dtest_outer = xgb.DMatrix(X_te, feature_names=feature_cols, nthread=-1)
          outer_raw_3c = final_model.predict(dtest_outer)  # shape (N, 3)

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
              # Only count tradeable predictions (LONG=2 or SHORT=0)
              tradeable = (outer_pred_class[high_conf] == 2) | (outer_pred_class[high_conf] == 0)
              if tradeable.sum() > 0:
                  hc_acc = accuracy_score(y_te[high_conf][tradeable], outer_pred_class[high_conf][tradeable])
                  n_long_pred = (outer_pred_class[high_conf][tradeable] == 2).sum()
                  n_short_pred = (outer_pred_class[high_conf][tradeable] == 0).sum()
                  log(f"    conf>{thresh:.2f}: {tradeable.sum()} trades (L={n_long_pred} S={n_short_pred}), Acc={hc_acc:.3f}")

          all_results[tf_name] = {
              'avg_accuracy': avg_acc, 'final_accuracy': final_acc,
              'n_features': len(feature_cols), 'n_samples': len(df),
              'context_only': False, 'window_results': window_results,
              'knn_ab_test': None,  # removed — no pre-filtering
              'label_type': 'triple_barrier',
          }

  finally:
      _shutdown_dask_client()

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
