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

# LightGBM runs on CPU with force_col_wise for max throughput
log(f"LightGBM: CPU mode (force_col_wise=True)")


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


def _predict_chunked(model, X, chunk_size=50000):
    """Predict in row chunks for large sparse matrices.
    Each chunk's CSR is small enough for LightGBM predict.
    Used for IS predictions where full train set NNZ may exceed int32.
    """
    if not hasattr(X, 'nnz') or X.nnz <= INT32_MAX:
        return model.predict(X)
    n = X.shape[0]
    preds = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = X[start:end]
        preds.append(model.predict(chunk))
    return np.vstack(preds)


def _cpcv_split_worker(args_tuple):
    """Train a single LightGBM CPCV split.
    Runs in a subprocess for parallel CPU training.
    Returns: (wi, acc, prec_long, prec_short, mlogloss, best_iter,
              model_bytes, preds_3c, y_test, test_idx_valid,
              importance, is_acc, is_mlogloss, is_sharpe)
    """
    (wi, train_idx, test_idx, X_data, X_indices, X_indptr, X_shape,
     y_3class, sample_weights, feature_cols, lgb_params,
     num_boost_round, tf_name, gpu_id) = args_tuple

    import numpy as np
    from scipy import sparse
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score, precision_score, log_loss

    # Reconstruct matrix in worker
    X_all = sparse.csr_matrix((X_data, X_indices, X_indptr), shape=X_shape)
    # If is_enable_sparse=False, convert to dense for multi-core LightGBM
    if not lgb_params.get('is_enable_sparse', True):
        X_all = X_all.toarray()

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

    params = lgb_params.copy()

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

    dtrain = lgb.Dataset(X_train_es, label=y_train_es, weight=w_train_es,
                         feature_name=feature_cols, free_raw_data=False)
    dval = lgb.Dataset(X_val_es, label=y_val_es, weight=w_val_es, feature_name=feature_cols, free_raw_data=False)
    model = lgb.train(
        params, dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dtrain, dval],
        valid_names=['train', 'val'],
        callbacks=[lgb.early_stopping(max(50, int(100 * (0.1 / params.get('learning_rate', 0.03))))), lgb.log_evaluation(0)],
    )

    # OOS predictions (LightGBM predicts directly on arrays)
    preds_3c = model.predict(X_test)
    pred_labels = np.argmax(preds_3c, axis=1)
    acc = float(accuracy_score(y_test, pred_labels))
    prec_long = float(precision_score(y_test, pred_labels, labels=[2], average='macro', zero_division=0))
    prec_short = float(precision_score(y_test, pred_labels, labels=[0], average='macro', zero_division=0))
    mlogloss = float(log_loss(y_test, preds_3c, labels=[0, 1, 2]))

    # IS metrics for PBO
    is_preds_3c = model.predict(X_train)
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
  from config import V3_LGBM_PARAMS as _CFG_LGBM, TF_MIN_DATA_IN_LEAF as _CFG_MIN_LEAF
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
  _parser = argparse.ArgumentParser()
  _parser.add_argument('--tf', action='append', help='Only train specific TFs (can repeat)')
  _parser.add_argument('--boost-rounds', type=int, default=800, help='LightGBM num_boost_round (default 800)')
  _parser.add_argument('--n-groups', type=int, default=None, help='Override CPCV n_groups (default: per-TF)')
  _parser.add_argument('--search-mode', action='store_true', default=False,
                        help='Use OPTUNA_SEARCH_CPCV_GROUPS for faster Optuna search trials')
  _parser.add_argument('--parallel-splits', action='store_true', default=False,
                        help='(legacy, now auto-detected) Kept for backward compat')
  _parser.add_argument('--no-parallel-splits', action='store_true', default=False,
                        help='Force sequential CPCV splits even with multiple GPUs')
  _args, _unknown = _parser.parse_known_args()  # ignore unknown args (e.g. from smoke_test)
  _tf_filter = set(_args.tf) if _args.tf else None

  # LightGBM CPU mode: parallel splits use ProcessPoolExecutor across CPU cores
  _use_parallel_splits = not _args.no_parallel_splits
  try:
      from hardware_detect import get_cpu_count
      _total_cores = get_cpu_count()
  except ImportError:
      import multiprocessing as _mp
      _total_cores = _mp.cpu_count() or 24
  if _args.no_parallel_splits:
      log("PARALLEL SPLITS: disabled (--no-parallel-splits)")
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

          # --- Sparse matrix support for 150K+ features ---
          # Load base features as dense (typically a few hundred to a few thousand cols)
          X_base = df[feature_cols].values.astype(np.float32)
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
              # Dense conversion decision:
              # - 1M+ features: NEVER convert (pickle serialization bottleneck kills parallel CPCV)
              #   Sparse LightGBM histogram = O(2×NNZ), faster than dense O(rows×features) at low density
              #   Perplexity-confirmed: dense+parallel on 6M features = 12-30h, sparse+sequential = 2-3h
              # - <1M features: convert if fits in 70% of RAM (multi-core dense histogram)
              import psutil
              _n_total_features = X_all.shape[1]
              _dense_bytes = X_all.shape[0] * _n_total_features * 4  # float32
              _avail_ram = psutil.virtual_memory().available
              if _dense_bytes < _avail_ram * 0.5:
                  # Dense fits comfortably — convert for multi-core LightGBM histogram
                  log(f"  Converting sparse to dense ({_dense_bytes/1e9:.1f} GB, RAM avail: {_avail_ram/1e9:.0f} GB)...")
                  X_all = X_all.toarray()
                  _converted_to_dense = True  # disable is_enable_sparse in LightGBM params
              elif _n_total_features > 1_000_000:
                  # Dense doesn't fit comfortably — stay sparse (sequential CPCV, single-thread histogram)
                  log(f"  Keeping SPARSE ({_n_total_features:,} features, dense={_dense_bytes/1e9:.1f} GB > 50% of {_avail_ram/1e9:.0f} GB avail)")
                  log(f"  Sparse histogram is single-threaded — expected ~15 min/fold for 2M+ features")
              else:
                  log(f"  Keeping SPARSE (dense would need {_dense_bytes/1e9:.1f} GB, only {_avail_ram/1e9:.0f} GB avail)")
              feature_cols = feature_cols + cross_cols
              _X_all_is_sparse = not _converted_to_dense  # True if still sparse CSR, False if converted to dense
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
          if _args.n_groups is not None:
              n_groups = _args.n_groups
              n_test_groups = 1
              log(f"  CPCV override: n_groups={n_groups}, n_test=1 (--n-groups flag)")
          elif _args.search_mode:
              # Optuna search mode: fewer CPCV groups for faster evaluation
              from config import OPTUNA_SEARCH_CPCV_GROUPS
              n_groups = OPTUNA_SEARCH_CPCV_GROUPS
              n_test_groups = 1
              log(f"  CPCV search mode: n_groups={n_groups}, n_test=1 (--search-mode flag)")
          else:
              from config import TF_CPCV_GROUPS
              n_groups, n_test_groups = TF_CPCV_GROUPS.get(tf_name, (4, 1))

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

          # Build LightGBM params once (shared by all paths)
          _base_lgb_params = V2_LGBM_PARAMS.copy()
          _base_lgb_params['min_data_in_leaf'] = _MIN_DATA_IN_LEAF.get(tf_name, 3)
          # Apply per-TF num_leaves cap (fix 2.8)
          from config import TF_NUM_LEAVES
          _base_lgb_params['num_leaves'] = TF_NUM_LEAVES.get(tf_name, 63)
          # Apply class weighting for imbalanced TFs (fix 2.3)
          from config import TF_CLASS_WEIGHT
          if tf_name in TF_CLASS_WEIGHT:
              _base_lgb_params['is_unbalance'] = True
              log(f"  is_unbalance=True (TF_CLASS_WEIGHT['{tf_name}'] = '{TF_CLASS_WEIGHT[tf_name]}')")
          if _converted_to_dense:
              _base_lgb_params['is_enable_sparse'] = False  # dense data → use multi-threaded dense histogram
              log(f"  is_enable_sparse=False (data converted to dense — enables multi-core LightGBM)")
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
          _optuna_config_path = os.path.join(PROJECT_DIR, f'optuna_configs_{tf_name}.json')
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
                      'max_depth', 'learning_rate',
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

          # Default: final feature cols = feature_cols (parallel path keeps HMM in X_all)
          # Sequential sparse path overrides this after stripping HMM into overlay
          _final_feature_cols = feature_cols
          _hmm_overlay = None  # Will be set by sequential sparse path if needed
          _hmm_overlay_names = []
          _parent_ds = None  # Set by sequential path for EFB reuse; parallel path doesn't use it

          if _nnz_exceeds_int32 and _use_parallel_splits:
              _use_parallel_splits = False
              log(f"  Forcing sequential CPCV (CSR arrays too large for multiprocess pickle)")

          if _n_total_features > 1_000_000 and _use_parallel_splits and _X_all_is_sparse:
              _use_parallel_splits = False
              log(f"  Forcing sequential CPCV ({_n_total_features:,} SPARSE features — pickle IPC bottleneck > training time)")
          # Dense data with >1M features: parallel OK (numpy arrays serialize fast, no pickle bottleneck)

          if _use_parallel_splits:
              # ── Parallel CPCV path (CPU workers) ──
              # NOTE: per-fold HMM re-fitting is skipped in parallel mode (HMM fitted once before loop).
              # This is the same tradeoff v2_multi_asset_trainer.py makes.

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
              _n_workers = int(os.environ.get('V3_CPCV_WORKERS', min(_pending_splits, _total_cores, _ram_workers)))
              _threads_per_worker = max(1, _total_cores // _n_workers)
              log(f"\n  PARALLEL CPCV: {_pending_splits} pending splits, {_n_workers} workers x {_threads_per_worker} threads = {_n_workers * _threads_per_worker} total ({_total_cores} cores)")

              # Set num_threads per worker to avoid oversubscription (each worker gets fair share of cores)
              _base_lgb_params = _base_lgb_params.copy()
              _base_lgb_params['num_threads'] = _threads_per_worker
              log(f"  num_threads per worker: {_base_lgb_params['num_threads']}")
              X_csr = X_all.tocsr() if hasattr(X_all, 'tocsr') else sp_sparse.csr_matrix(X_all)
              worker_args = []
              for wi, (train_idx, test_idx) in enumerate(splits):
                  if wi in _completed_folds:
                      log(f"  Path {wi+1}/{len(splits)}: SKIP (checkpoint)")
                      continue
                  gpu_id = 0  # unused in LightGBM CPU mode
                  worker_args.append((
                      wi, train_idx, test_idx,
                      X_csr.data, X_csr.indices, X_csr.indptr, X_csr.shape,
                      y_3class, sample_weights, feature_cols, _base_lgb_params,
                      _args.boost_rounds, tf_name, gpu_id
                  ))

              with ProcessPoolExecutor(max_workers=_n_workers) as executor:
                  for result in executor.map(_cpcv_split_worker, worker_args):
                      (wi, acc, prec_long, prec_short, mlogloss_val, best_iter,
                       model_bytes, preds_3c, y_test, test_idx_valid,
                       importance, is_acc, is_mlogloss, is_sharpe) = result

                      if acc is None:
                          log(f"  Path {wi+1}/{len(splits)}: SKIP -- not enough samples")
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

          else:
              # ── Sequential CPCV path (dense matrix or --no-parallel-splits) ──

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
              _parent_ds = None
              try:
                  _parent_t0 = time.time()
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
                  _parent_ds = lgb.Dataset(
                      _Xp, label=y_3class[_parent_valid].astype(int),
                      weight=sample_weights[_parent_valid],
                      feature_name=_parent_feature_cols,
                      free_raw_data=True,
                      params={'feature_pre_filter': False, 'max_bin': _base_lgb_params.get('max_bin', 255)},
                  )
                  _parent_ds.construct()
                  log(f"  Parent Dataset built: {_Xp.shape[1]:,} features, EFB bins computed ONCE "
                      f"({time.time() - _parent_t0:.1f}s). Folds reuse via reference=.")
                  del _Xp
                  gc.collect()
              except Exception as _pde:
                  log(f"  WARNING: Parent Dataset build failed ({_pde}), folds will construct independently")
                  _parent_ds = None

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
                  val_size = max(int(n_tr * 0.15), 100)
                  if val_size >= n_tr:
                      val_size = max(n_tr // 5, 20)  # fallback for tiny folds
                  X_val_es = X_train[-val_size:]
                  y_val_es = y_train[-val_size:]
                  w_val_es = w_train[-val_size:]
                  X_train_es = X_train[:-val_size]
                  y_train_es = y_train[:-val_size]
                  w_train_es = w_train[:-val_size]

                  _ds_kwargs = dict(feature_name=_fold_feature_cols, free_raw_data=False)
                  if _parent_ds is not None:
                      _ds_kwargs['reference'] = _parent_ds  # reuse EFB bundles + bin thresholds
                  dtrain = lgb.Dataset(X_train_es, label=y_train_es, weight=w_train_es, **_ds_kwargs)
                  dval = lgb.Dataset(X_val_es, label=y_val_es, weight=w_val_es, **_ds_kwargs)
                  _ckpt_path = os.path.join(DB_DIR, f'lgbm_ckpt_{tf_name}_fold{wi}.txt')
                  model = lgb.train(
                      params, dtrain,
                      num_boost_round=_args.boost_rounds,
                      valid_sets=[dtrain, dval],
                      valid_names=['train', 'val'],
                      callbacks=[
                          lgb.early_stopping(max(50, int(100 * (0.1 / params.get('learning_rate', 0.03))))),
                          lgb.log_evaluation(100),
                          CheckpointCallback(_ckpt_path, period=100),
                      ],
                  )

                  # Predict OOS (LightGBM predicts directly on arrays)
                  preds_3c = model.predict(X_test)  # shape (N, 3)
                  pred_labels = np.argmax(preds_3c, axis=1)
                  acc = accuracy_score(y_test, pred_labels)
                  prec_long = precision_score(y_test, pred_labels, labels=[2], average='macro', zero_division=0)
                  prec_short = precision_score(y_test, pred_labels, labels=[0], average='macro', zero_division=0)
                  mlogloss = log_loss(y_test, preds_3c, labels=[0, 1, 2])

                  # Evaluate IS (full training data) for proper PBO
                  is_preds_3c = _predict_chunked(model, X_train)
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
          if _converted_to_dense:
              final_params['is_enable_sparse'] = False
          # Copy interaction_constraints from CPCV params (fix 2.5)
          if 'interaction_constraints' in _base_lgb_params:
              final_params['interaction_constraints'] = _base_lgb_params['interaction_constraints']
          # Apply Optuna-found params to final retrain (same overlay as CPCV phase)
          _OPTUNA_TUNABLE_KEYS_FINAL = [
              'num_leaves', 'min_data_in_leaf', 'feature_fraction',
              'feature_fraction_bynode', 'bagging_fraction',
              'lambda_l1', 'lambda_l2', 'min_gain_to_split',
              'max_depth', 'learning_rate',
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
          _final_ds_kwargs = dict(feature_name=_final_feature_cols, free_raw_data=False)
          if _parent_ds is not None:
              _final_ds_kwargs['reference'] = _parent_ds  # reuse EFB from parent
          dtrain = lgb.Dataset(X_tr_f, label=y_tr_f, weight=w_tr_f, **_final_ds_kwargs)
          dval = lgb.Dataset(X_val_f, label=y_val_f, weight=w_val_f, **_final_ds_kwargs)
          _final_ckpt_path = os.path.join(DB_DIR, f'lgbm_ckpt_{tf_name}_final.txt')
          final_model = lgb.train(
              final_params, dtrain, num_boost_round=_args.boost_rounds,
              valid_sets=[dtrain, dval], valid_names=['train', 'val'],
              callbacks=[
                  lgb.early_stopping(max(50, int(100 * (0.1 / final_params.get('learning_rate', 0.03))))),
                  lgb.log_evaluation(100),
                  CheckpointCallback(_final_ckpt_path, period=100),
              ],
          )

          # Evaluate on val set (held-out 15% from end, used for early stopping)
          final_preds_3c = final_model.predict(X_val_f)  # shape (N, 3)
          final_labels = np.argmax(final_preds_3c, axis=1)
          final_acc = accuracy_score(y_val_f, final_labels)
          final_prec_l = precision_score(y_val_f, final_labels, labels=[2], average='macro', zero_division=0)
          final_prec_s = precision_score(y_val_f, final_labels, labels=[0], average='macro', zero_division=0)
          final_mlogloss = log_loss(y_val_f, final_preds_3c, labels=[0, 1, 2])
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
          # Atomic save: write to temp then rename (prevents corrupt model on crash)
          _model_path = f'{DB_DIR}/model_{tf_name}.json'
          _model_tmp = _model_path + '.tmp'
          final_model.save_model(_model_tmp)
          os.replace(_model_tmp, _model_path)
          _feat_path = f'{DB_DIR}/features_{tf_name}_all.json'
          _feat_tmp = _feat_path + '.tmp'
          with open(_feat_tmp, 'w') as f:
              json.dump(_final_feature_cols, f, indent=2)
          os.replace(_feat_tmp, _feat_path)

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
