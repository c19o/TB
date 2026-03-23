#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
backtesting_audit.py — Comprehensive Backtesting Audit Tool
=============================================================
Simulates the optimized strategy on ALL BTC history and generates
detailed breakdowns by month, week, regime, trade type, and named periods.

Outputs:
  - audit_report.json   (machine-readable)
  - audit_report.txt    (human-readable text tables)
  - audit_heatmap.html  (standalone color-coded monthly P&L heatmap)

Usage:
    python backtesting_audit.py
    python backtesting_audit.py --tf 1h 4h 1d
"""

import sys, os, io, time, json, warnings, argparse, math
if os.name == 'nt':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import sqlite3
import numba
import scipy.sparse as sp_sparse
from datetime import datetime

# Import single-source-of-truth configs
try:
    from config import TF_PARENT_MAP, TRADE_TYPE_PARAMS, TRADE_THRESHOLDS as CFG_TRADE_THRESHOLDS
    _HAS_CONFIG = True
except ImportError:
    _HAS_CONFIG = False

# ---------------------------------------------------------------------------
# GPU backend: try CuPy, fall back to NumPy
# ---------------------------------------------------------------------------
try:
    import cupy as cp
    xp = cp
    GPU_ARRAY = True
    print(f"[GPU] CuPy + CUDA detected — GPU arrays enabled")
except ImportError:
    xp = np
    GPU_ARRAY = False
    print("[CPU] CuPy not available — using NumPy")

import xgboost as xgb

USE_GPU_XGB = False
try:
    _test = xgb.DMatrix(np.random.rand(10, 5), label=np.random.randint(0, 2, 10))
    xgb.train({'tree_method': 'gpu_hist', 'device': 'cuda', 'max_depth': 3},
              _test, num_boost_round=2)
    USE_GPU_XGB = True
    del _test
except Exception:
    pass
print(f"[XGB] GPU prediction: {'ENABLED' if USE_GPU_XGB else 'CPU only'}")

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
DB_DIR = os.environ.get('SAVAGE22_DB_DIR', os.path.dirname(os.path.abspath(__file__)))
START_TIME = time.time()
STARTING_BALANCE = 10000.0
FEE_RATE = 0.0018  # 0.18% round-trip

# Use config.py as single source of truth, fall back to local definition
if _HAS_CONFIG:
    TRADE_THRESHOLDS = CFG_TRADE_THRESHOLDS
else:
    TRADE_THRESHOLDS = {
        '5m':  {'scalp': 12,  'day': 72,  'swing': 288},
        '15m': {'scalp': 4,   'day': 48,  'swing': 192},
        '1h':  {'scalp': 1,   'day': 24,  'swing': 72},
        '4h':  {'scalp': 1,   'day': 6,   'swing': 84},
        '1d':  {'scalp': 0,   'day': 1,   'swing': 30},
        '1w':  {'scalp': 0,   'day': 0,   'swing': 4},
    }

# Confluence parent map fallback
if not _HAS_CONFIG:
    TF_PARENT_MAP = {
        '5m': '15m', '15m': '1h', '1h': '4h',
        '4h': '1d', '1d': None, '1w': None,
    }
    TRADE_TYPE_PARAMS = {
        'scalp':     {'max_correlation_positions': 1, 'sl_tightness': 1.0, 'tp_aggression': 1.5, 'partial_tp_pct': 0.75},
        'day_trade':  {'max_correlation_positions': 2, 'sl_tightness': 1.0, 'tp_aggression': 1.2, 'partial_tp_pct': 0.50},
        'swing':     {'max_correlation_positions': 3, 'sl_tightness': 0.8, 'tp_aggression': 1.0, 'partial_tp_pct': 0.25},
        'position':  {'max_correlation_positions': 2, 'sl_tightness': 0.6, 'tp_aggression': 0.8, 'partial_tp_pct': 0.0},
    }

NAMED_PERIODS = [
    ("COVID Crash",         "2020-03-01", "2020-03-31"),
    ("COVID Recovery",      "2020-04-01", "2020-07-31"),
    ("2020 Bull Run",       "2020-10-01", "2021-01-31"),
    ("May 2021 Crash",      "2021-05-01", "2021-06-30"),
    ("Nov 2021 ATH",        "2021-10-01", "2021-11-30"),
    ("Luna/3AC Collapse",   "2022-05-01", "2022-06-30"),
    ("FTX Crash",           "2022-11-01", "2022-11-30"),
    ("2023 Recovery",       "2023-01-01", "2023-06-30"),
    ("ETF Approval Rally",  "2024-01-01", "2024-03-31"),
    ("2024-2025 Bull",      "2024-10-01", "2025-03-31"),
]

# Regime multipliers matching live_trader.py
REGIME_MULT = {
    0: {'lev': 1.0,  'risk': 1.0,  'stop': 1.0,  'rr': 1.5,  'hold': 1.0},   # bull
    1: {'lev': 0.47, 'risk': 1.0,  'stop': 0.75, 'rr': 0.75, 'hold': 0.17},   # bear
    2: {'lev': 0.67, 'risk': 0.47, 'stop': 0.5,  'rr': 0.5,  'hold': 1.0},    # sideways
    3: {'lev': 0.2,  'risk': 0.25, 'stop': 0.5,  'rr': 0.5,  'hold': 0.1},    # crash
}
REGIME_NAMES = {0: 'bull', 1: 'bear', 2: 'sideways', 3: 'crash'}

# DB mapping (matches exhaustive_optimizer.py)
TF_DB_MAP = {
    '5m':  {'db': 'features_5m.db',       'table': 'features_5m',  'return_col': 'next_5m_return',  'cost_pct': 0.22},
    '15m': {'db': 'features_15m.db',      'table': 'features_15m', 'return_col': 'next_15m_return', 'cost_pct': 0.22},
    '1h':  {'db': 'features_1h.db',       'table': 'features_1h',  'return_col': 'next_1h_return',  'cost_pct': 0.23},
    '4h':  {'db': 'features_4h.db',       'table': 'features_4h',  'return_col': 'next_4h_return',  'cost_pct': 0.24},
    '1d':  {'db': 'features_1d.db',       'table': 'features_1d',  'return_col': 'next_1d_return',  'cost_pct': 0.0025},
    '1w':  {'db': 'features_1w.db',       'table': 'features_1w',  'return_col': 'next_1w_return',  'cost_pct': 0.0025},
}


def elapsed():
    return f"[{time.time() - START_TIME:.0f}s]"


# ---------------------------------------------------------------------------
# Data loader: full history + model predictions
# ---------------------------------------------------------------------------
def load_full_history(tf_name):
    """
    Load the ENTIRE feature DB + model for a TF.
    Predictions are restricted to OUT-OF-SAMPLE bars only:
      - If cpcv_oos_predictions_{tf}.pkl exists, uses CPCV OOS predictions directly.
      - Otherwise falls back to 80/20 temporal split (last 20% only) with a warning.
    Non-OOS bars get direction=0 (FLAT) so no trades are entered on training data.
    Returns dict with timestamps, directions, confidences, closes, highs, lows, atrs, n_bars
    or None on failure.
    """
    cfg = TF_DB_MAP[tf_name]
    db_path = os.path.join(DB_DIR, cfg['db'])
    model_path = os.path.join(DB_DIR, f'model_{tf_name}.json')
    features_all_path = os.path.join(DB_DIR, f'features_{tf_name}_all.json')
    features_pruned_path = os.path.join(DB_DIR, f'features_{tf_name}_pruned.json')

    if not os.path.exists(db_path):
        print(f"  SKIP {tf_name} -- {cfg['db']} not found", flush=True)
        return None
    if not os.path.exists(model_path):
        print(f"  SKIP {tf_name} -- model_{tf_name}.json not found", flush=True)
        return None

    # Load feature list
    saved_features = None
    if os.path.exists(features_all_path):
        with open(features_all_path, 'r') as f:
            saved_features = json.load(f)
        print(f"  Loaded {len(saved_features)} features from features_{tf_name}_all.json", flush=True)
    elif os.path.exists(features_pruned_path):
        with open(features_pruned_path, 'r') as f:
            saved_features = json.load(f)
        print(f"  Loaded {len(saved_features)} features from features_{tf_name}_pruned.json", flush=True)

    # Load data -- try V2 parquet naming first, then V1, then SQLite
    parquet_path = db_path.replace('.db', '.parquet')
    v2_parquet = os.path.join(DB_DIR, f'features_BTC_{tf_name}.parquet')
    if not os.path.exists(parquet_path) and os.path.exists(v2_parquet):
        parquet_path = v2_parquet
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        print(f"  Loaded from parquet: {parquet_path} ({len(df):,} rows)", flush=True)
    else:
        conn = sqlite3.connect(db_path)
        ext_check = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (cfg['table'] + '_ext',)
        ).fetchone()
        if ext_check:
            df_main = pd.read_sql_query(f"SELECT * FROM {cfg['table']}", conn)
            df_ext = pd.read_sql_query(f"SELECT * FROM {cfg['table']}_ext", conn)
            df_ext = df_ext.drop(columns=['timestamp'], errors='ignore')
            df = pd.concat([df_main, df_ext], axis=1)
        else:
            df = pd.read_sql_query(f"SELECT * FROM {cfg['table']}", conn)
        conn.close()
        print(f"  Loaded from SQLite: {db_path} ({len(df):,} rows)", flush=True)

    # Timestamps
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    elif 'date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['date'])

    timestamps = df['timestamp'].values

    # OHLC + ATR
    closes = pd.to_numeric(df['close'], errors='coerce').values.astype(np.float64)
    highs = pd.to_numeric(df['high'], errors='coerce').values.astype(np.float64) if 'high' in df.columns else closes.copy()
    lows = pd.to_numeric(df['low'], errors='coerce').values.astype(np.float64) if 'low' in df.columns else closes.copy()

    if 'atr_14' in df.columns:
        atrs = pd.to_numeric(df['atr_14'], errors='coerce').values.astype(np.float64)
    else:
        atrs = np.abs(np.diff(closes, prepend=closes[0])) * 1.5
    atrs = np.nan_to_num(atrs, nan=max(np.nanmean(closes) * 0.01, 1.0))
    atrs = np.maximum(atrs, 1e-8)

    # Feature columns
    meta_cols = {'timestamp', 'date', 'open', 'high', 'low', 'close', 'volume',
                 'quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote',
                 'open_time', 'date_norm'}
    target_like = {c for c in df.columns if 'next_' in c.lower() or 'target' in c.lower() or (c.lower().startswith('next_') and 'direction' in c.lower())}
    exclude_cols = meta_cols | target_like
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    if saved_features is not None:
        model_features = saved_features
    else:
        model_features = feature_cols
        print(f"  No feature file -- using all {len(model_features)} columns from DB", flush=True)

    # Build feature matrix
    X = np.empty((len(df), len(model_features)), dtype=np.float32)
    for i, feat in enumerate(model_features):
        if feat in feature_cols:
            X[:, i] = pd.to_numeric(df[feat], errors='coerce').values.astype(np.float32)
        else:
            X[:, i] = np.nan

    # Sanitize inf → NaN (matches ml_multi_tf.py)
    X = np.where(np.isinf(X), np.nan, X)

    # --- Load sparse cross features from .npz (matches ml_multi_tf.py lines 602-654) ---
    cross_matrix = None
    cross_cols = None
    npz_path = os.path.join(DB_DIR, f'v2_crosses_BTC_{tf_name}.npz')
    if os.path.exists(npz_path):
        try:
            print(f"  Loading sparse cross matrix: {npz_path}", flush=True)
            cross_matrix = sp_sparse.load_npz(npz_path).tocsr()
            # Load column names — try both naming conventions
            cols_path_v1 = npz_path.replace('.npz', '_columns.json')
            cols_path_v2 = os.path.join(DB_DIR, f'v2_cross_names_BTC_{tf_name}.json')
            if os.path.exists(cols_path_v1):
                with open(cols_path_v1) as f:
                    cross_cols = json.load(f)
            elif os.path.exists(cols_path_v2):
                with open(cols_path_v2) as f:
                    cross_cols = json.load(f)
            else:
                cross_cols = [f'cross_{i}' for i in range(cross_matrix.shape[1])]
            print(f"  Sparse crosses loaded: {cross_matrix.shape[0]} rows x {cross_matrix.shape[1]} cols "
                  f"({cross_matrix.nnz:,} non-zeros, "
                  f"{cross_matrix.nnz / max(1, cross_matrix.shape[0] * cross_matrix.shape[1]) * 100:.4f}% dense)", flush=True)
        except Exception as e:
            print(f"  WARNING: Failed to load sparse crosses: {e}", flush=True)
            cross_matrix = None
            cross_cols = None
    else:
        print(f"  WARNING: No sparse cross file found at {npz_path} — predicting with base features only", flush=True)

    # Combine base + crosses into X_all for prediction
    _X_all_is_sparse = False
    if cross_matrix is not None and cross_matrix.shape[0] == X.shape[0]:
        # Convert base to sparse, preserving NaN (XGBoost treats NaN as missing natively)
        X_base_sparse = sp_sparse.csr_matrix(X)
        X_all = sp_sparse.hstack([X_base_sparse, cross_matrix], format='csr')
        # DO NOT call eliminate_zeros() — it converts explicit 0.0 values into
        # structural zeros, which XGBoost treats as "missing." 0.0 means "the value
        # is zero" (a real signal), not "data is absent." NaN already encodes missing.
        model_features = model_features + cross_cols
        _X_all_is_sparse = True
        n_base = X.shape[1]
        print(f"  Combined sparse matrix: {X_all.shape[0]} rows x {X_all.shape[1]} cols "
              f"({n_base} base + {len(cross_cols)} crosses) "
              f"density={X_all.nnz / max(1, X_all.shape[0] * X_all.shape[1]) * 100:.4f}%", flush=True)
        del X_base_sparse, cross_matrix
        X = X_all  # replace X with combined sparse matrix
        del X_all
    elif cross_matrix is not None:
        print(f"  WARNING: Cross matrix row count ({cross_matrix.shape[0]}) != base ({X.shape[0]}), skipping crosses", flush=True)
        del cross_matrix

    # Load model and predict ALL bars
    model = xgb.Booster()
    model.load_model(model_path)
    if USE_GPU_XGB:
        model.set_param({'device': 'cuda'})

    dmat = xgb.DMatrix(X, feature_names=model_features, nthread=-1)
    raw_preds = model.predict(dmat)

    if raw_preds.ndim == 2 and raw_preds.shape[1] == 3:
        pred_class = np.argmax(raw_preds, axis=1)  # 0=SHORT, 1=FLAT, 2=LONG
        confidences = np.max(raw_preds, axis=1)
        directions = np.where(pred_class == 2, 1, np.where(pred_class == 0, -1, 0)).astype(np.int32)
        print(f"  3-class: LONG={np.sum(pred_class==2):,} FLAT={np.sum(pred_class==1):,} SHORT={np.sum(pred_class==0):,}", flush=True)
    else:
        raise ValueError(f"Model output is 1D ({raw_preds.shape}), expected 3-class softprob. Retrain with multi:softprob.")

    n_bars = len(df)
    print(f"  Confidence range: [{confidences.min():.3f}, {confidences.max():.3f}]", flush=True)

    # -----------------------------------------------------------------------
    # OOS filtering: only evaluate on out-of-sample bars
    # -----------------------------------------------------------------------
    oos_path = os.path.join(DB_DIR, f'cpcv_oos_predictions_{tf_name}.pkl')
    if os.path.exists(oos_path):
        import pickle
        with open(oos_path, 'rb') as f:
            oos_preds = pickle.load(f)
        # Build set of all OOS bar indices across CPCV paths
        oos_indices = set()
        for fold in oos_preds:
            oos_indices.update(fold['test_indices'].tolist() if hasattr(fold['test_indices'], 'tolist') else fold['test_indices'])
        # Use CPCV OOS predictions instead of model re-predictions where available
        oos_directions = np.zeros(n_bars, dtype=np.int32)  # FLAT for non-OOS
        oos_confidences = np.zeros(n_bars, dtype=np.float64)
        # Aggregate: last fold's prediction wins (CPCV paths may overlap)
        for fold in oos_preds:
            idxs = np.array(fold['test_indices'])
            valid = idxs < n_bars  # guard against stale pkl with different DB size
            idxs = idxs[valid]
            if 'y_pred_probs' in fold:
                probs = fold['y_pred_probs'][valid]
                if probs.ndim == 2 and probs.shape[1] == 3:
                    pc = np.argmax(probs, axis=1)
                    oos_directions[idxs] = np.where(pc == 2, 1, np.where(pc == 0, -1, 0)).astype(np.int32)
                    oos_confidences[idxs] = np.max(probs, axis=1)
                else:
                    oos_directions[idxs] = np.where(probs.ravel() > 0.5, 1, -1).astype(np.int32)
                    oos_confidences[idxs] = probs.ravel()
            else:
                # Fallback: use model predictions but only for OOS indices
                oos_directions[idxs] = directions[idxs]
                oos_confidences[idxs] = confidences[idxs]
        directions = oos_directions
        confidences = oos_confidences
        n_oos = int(np.sum(directions != 0))
        print(f"  OOS filter: loaded CPCV predictions, {len(oos_indices):,} OOS bars "
              f"({len(oos_indices)/n_bars*100:.1f}%), {n_oos:,} with signals", flush=True)
    else:
        # Fallback: 80/20 temporal split — only use last 20%
        split_idx = int(n_bars * 0.8)
        directions[:split_idx] = 0  # FLAT = no trades on training portion
        confidences[:split_idx] = 0.0
        n_oos = n_bars - split_idx
        print(f"  WARNING: cpcv_oos_predictions_{tf_name}.pkl not found — "
              f"using last 20% temporal split ({n_oos:,} bars). Results are approximate.", flush=True)

    return {
        'timestamps': timestamps,
        'directions': directions.astype(np.int32),
        'confidences': confidences.astype(np.float64),
        'closes': closes,
        'highs': highs,
        'lows': lows,
        'atrs': atrs,
        'n_bars': n_bars,
    }


# ---------------------------------------------------------------------------
# Load optimized config for a TF
# ---------------------------------------------------------------------------
def load_tf_config(tf_name):
    """Load optimized parameters from exhaustive_configs.json or per-TF file."""
    # Try combined file
    combined_path = os.path.join(DB_DIR, 'exhaustive_configs.json')
    if os.path.exists(combined_path):
        with open(combined_path, 'r') as f:
            all_cfg = json.load(f)
        if tf_name in all_cfg:
            tf_ec = all_cfg[tf_name]
            profile = tf_ec.get('dd15_best') or tf_ec.get('dd10_best') or tf_ec.get('dd15_sortino') or tf_ec.get('dd10_sortino')
            if profile:
                return profile

    # Try per-TF file
    per_tf_path = os.path.join(DB_DIR, f'exhaustive_configs_{tf_name}.json')
    if os.path.exists(per_tf_path):
        with open(per_tf_path, 'r') as f:
            tf_cfg = json.load(f)
        if tf_name in tf_cfg:
            tf_ec = tf_cfg[tf_name]
            profile = tf_ec.get('dd15_best') or tf_ec.get('dd10_best') or tf_ec.get('dd15_sortino') or tf_ec.get('dd10_sortino')
            if profile:
                return profile

    # Fall back to GA configs
    ga_path = os.path.join(DB_DIR, 'ml_multi_tf_configs.json')
    if os.path.exists(ga_path):
        with open(ga_path, 'r') as f:
            ml_configs = json.load(f)
        cfg = ml_configs.get(tf_name, {})
        for level in ['god_mode', 'aggressive', 'balanced']:
            c = cfg.get('configs', {}).get(level, {})
            if c and 'params' in c:
                p = c['params']
                return {
                    'leverage': p[0], 'risk_pct': p[1],
                    'stop_atr': p[2], 'rr': p[3],
                    'max_hold': int(p[4]), 'exit_type': int(p[5]) if len(p) > 5 else 0,
                    'conf_thresh': p[6] if len(p) > 6 else 0.6,
                }

    # Sensible defaults
    print(f"  WARNING: No config found for {tf_name}, using defaults", flush=True)
    return {
        'leverage': 5, 'risk_pct': 1.0, 'stop_atr': 0.5,
        'rr': 2.0, 'max_hold': 24, 'exit_type': 0, 'conf_thresh': 0.6,
    }


# ---------------------------------------------------------------------------
# Regime detection (vectorized)
# ---------------------------------------------------------------------------
def detect_regime_series(closes):
    """Vectorized regime detection. Returns int array: 0=bull, 1=bear, 2=sideways, 3=crash."""
    closes_s = pd.Series(closes)
    sma100 = closes_s.rolling(100, min_periods=1).mean().values
    slope = np.gradient(sma100) / np.maximum(np.abs(sma100), 1e-8)

    # Realized volatility (20-bar rolling std of log returns)
    log_ret = np.log(np.maximum(closes[1:] / np.maximum(closes[:-1], 1e-8), 1e-8))
    log_ret = np.concatenate([[0.0], log_ret])
    rvol_20 = pd.Series(log_ret).rolling(20, min_periods=1).std().values
    rvol_90_avg = pd.Series(rvol_20).rolling(90, min_periods=1).mean().values

    # 30-bar rolling high and drawdown from it
    rolling_high_30 = closes_s.rolling(30, min_periods=1).max().values
    dd_from_30h = (rolling_high_30 - closes) / np.maximum(rolling_high_30, 1e-8)

    regime = np.full(len(closes), 2, dtype=np.int32)  # default sideways
    above = closes > sma100
    bull_mask = above & (slope > 0.001)
    bear_mask = (~above) & (slope < -0.001)

    # Crash = volatility > 2x 90-day avg AND below SMA100 AND drawdown from 30-day high > 15%
    crash_mask = (rvol_20 > 2.0 * rvol_90_avg) & (~above) & (dd_from_30h > 0.15)

    regime[bull_mask] = 0
    regime[bear_mask] = 1
    regime[crash_mask] = 3  # crash overrides bear/sideways
    return regime


# ---------------------------------------------------------------------------
# Numba-compiled bar-by-bar simulation with trade recording
# ---------------------------------------------------------------------------
@numba.njit(cache=True)
def _simulate_with_trades(timestamps_i64, dirs, confs, closes, highs, lows, atrs, regimes,
                          leverage, risk_pct, stop_atr_mult, rr_ratio, max_hold, exit_type,
                          conf_thresh, fee_rate, starting_balance,
                          regime_lev, regime_risk, regime_stop, regime_rr, regime_hold,
                          slippage_rate=0.0002):
    """
    Numba-compiled bar-by-bar simulation recording every trade.

    regime_lev/risk/stop/rr/hold: arrays for bull=0, bear=1, sideways=2, crash=3 multipliers.

    Returns:
        equity_curve: (n_bars,) float64 array
        trades: (max_trades, 12) float64 array
        trade_count: int
    """
    n = len(closes)
    max_trades = n // 2 + 1
    equity = np.empty(n, dtype=np.float64)
    trades = np.empty((max_trades, 12), dtype=np.float64)
    trade_idx = 0

    balance = starting_balance
    peak_balance = starting_balance
    in_trade = False
    entry_bar = 0
    entry_price = 0.0
    stop_price = 0.0
    tp_price = 0.0
    trade_dir = 0
    bars_held = 0
    entry_conf = 0.0
    entry_lev = 0.0
    entry_risk = 0.0
    entry_regime = 0
    current_max_hold = 0.0

    for t in range(n):
        if balance <= 0.0:
            equity[t] = 0.0
            continue

        # --- Exit logic ---
        if in_trade:
            bars_held += 1
            hit_sl = False
            hit_tp = False
            exit_price = closes[t]

            if trade_dir == 1:  # LONG
                if lows[t] <= stop_price:
                    hit_sl = True
                    exit_price = stop_price
                elif highs[t] >= tp_price:
                    hit_tp = True
                    exit_price = tp_price
            else:  # SHORT
                if highs[t] >= stop_price:
                    hit_sl = True
                    exit_price = stop_price
                elif lows[t] <= tp_price:
                    hit_tp = True
                    exit_price = tp_price

            time_exit = bars_held >= current_max_hold

            if hit_sl or hit_tp or time_exit:
                # Apply slippage to exit price
                if trade_dir == 1:  # LONG exit: sell lower
                    exit_price *= (1 - slippage_rate)
                else:  # SHORT exit: buy higher
                    exit_price *= (1 + slippage_rate)

                # Compute PnL
                price_chg = (exit_price - entry_price) / max(entry_price, 1e-8) * trade_dir
                gross_pnl = price_chg * entry_lev
                fee_cost = fee_rate * entry_lev
                net_pnl_pct = gross_pnl - fee_cost

                # Partial TP scaling
                if hit_tp and exit_type > 0:
                    net_pnl_pct *= (exit_type / 100.0)

                # Drawdown scaling: reduce size in drawdown
                dd = (peak_balance - balance) / max(peak_balance, 1e-8)
                if dd < 0.15:
                    dd_scale = max(0.0, 1.0 - 2.0 * dd)
                else:
                    dd_scale = 0.0

                pnl_dollar = balance * (entry_risk / 100.0) * net_pnl_pct * dd_scale
                balance += pnl_dollar
                balance = max(balance, 0.0)
                peak_balance = max(peak_balance, balance)

                # Record trade
                if trade_idx < max_trades:
                    trades[trade_idx, 0] = entry_bar          # entry_bar
                    trades[trade_idx, 1] = t                   # exit_bar
                    trades[trade_idx, 2] = trade_dir           # direction
                    trades[trade_idx, 3] = entry_price         # entry_price
                    trades[trade_idx, 4] = exit_price          # exit_price
                    trades[trade_idx, 5] = bars_held           # bars_held
                    trades[trade_idx, 6] = pnl_dollar          # pnl
                    trades[trade_idx, 7] = net_pnl_pct * 100   # pnl_pct
                    trades[trade_idx, 8] = entry_conf          # confidence
                    trades[trade_idx, 9] = entry_lev           # leverage_used
                    trades[trade_idx, 10] = entry_risk         # risk_used
                    trades[trade_idx, 11] = entry_regime       # regime_at_entry
                    trade_idx += 1

                in_trade = False

        # --- Entry logic ---
        if not in_trade and balance > 0.0:
            if dirs[t] != 0 and confs[t] > conf_thresh:
                r = regimes[t]
                eff_lev = leverage * regime_lev[r]
                eff_risk = risk_pct * regime_risk[r]
                eff_stop = stop_atr_mult * regime_stop[r]
                eff_rr = rr_ratio * regime_rr[r]
                eff_hold = max_hold * regime_hold[r]

                # Kelly sizing: boost risk by up to 25% based on confidence
                kelly_f = (confs[t] - conf_thresh) / max(1.0 - conf_thresh, 1e-8)
                kelly_risk = eff_risk * (1.0 + 0.25 * kelly_f)
                kelly_risk = min(kelly_risk, eff_risk * 3.0)

                # Drawdown scaling
                dd = (peak_balance - balance) / max(peak_balance, 1e-8)
                if dd >= 0.15:
                    pass  # skip entry in deep drawdown
                else:
                    in_trade = True
                    entry_bar = t
                    trade_dir = dirs[t]
                    # Apply slippage to entry
                    if dirs[t] == 1:  # LONG: buy higher
                        entry_price = closes[t] * (1 + slippage_rate)
                    else:  # SHORT: sell lower
                        entry_price = closes[t] * (1 - slippage_rate)
                    entry_conf = confs[t]
                    entry_lev = eff_lev
                    entry_risk = kelly_risk
                    entry_regime = r
                    bars_held = 0
                    current_max_hold = max(1.0, eff_hold)

                    sl_dist = eff_stop * atrs[t]
                    tp_dist = sl_dist * eff_rr

                    if trade_dir == 1:
                        stop_price = entry_price - sl_dist
                        tp_price = entry_price + tp_dist
                    else:
                        stop_price = entry_price + sl_dist
                        tp_price = entry_price - tp_dist

        equity[t] = balance

    return equity, trades[:trade_idx], trade_idx


# ---------------------------------------------------------------------------
# Trade classification
# ---------------------------------------------------------------------------
def classify_trades(trades_df, tf_name):
    """Add trade_type column: scalp / day_trade / swing / position."""
    thresholds = TRADE_THRESHOLDS.get(tf_name, TRADE_THRESHOLDS['1h'])
    conditions = [
        trades_df['bars_held'] <= thresholds['scalp'],
        trades_df['bars_held'] <= thresholds['day'],
        trades_df['bars_held'] <= thresholds['swing'],
    ]
    choices = ['scalp', 'day_trade', 'swing']
    trades_df['trade_type'] = np.select(conditions, choices, default='position')
    return trades_df


# ---------------------------------------------------------------------------
# Report functions
# ---------------------------------------------------------------------------
def monthly_heatmap(trades_df):
    """Monthly P&L, win rate, and trade count as DataFrames (years x months)."""
    if trades_df.empty:
        return {'pnl': pd.DataFrame(), 'win_rate': pd.DataFrame(), 'count': pd.DataFrame()}

    trades_df = trades_df.copy()
    trades_df['year'] = trades_df['exit_time'].dt.year
    trades_df['month'] = trades_df['exit_time'].dt.month

    pnl = trades_df.groupby(['year', 'month'])['pnl'].sum().unstack(fill_value=0)
    count = trades_df.groupby(['year', 'month'])['pnl'].count().unstack(fill_value=0)
    wins = trades_df[trades_df['pnl'] > 0].groupby(['year', 'month'])['pnl'].count().unstack(fill_value=0)
    win_rate = (wins / count.replace(0, np.nan)).fillna(0)

    # Ensure all 12 months
    for m in range(1, 13):
        if m not in pnl.columns:
            pnl[m] = 0.0
            count[m] = 0
            win_rate[m] = 0.0
    pnl = pnl[sorted(pnl.columns)]
    count = count[sorted(count.columns)]
    win_rate = win_rate[sorted(win_rate.columns)]

    return {'pnl': pnl, 'win_rate': win_rate, 'count': count}


def weekly_performance(trades_df):
    """Group by ISO week number. Returns DataFrame."""
    if trades_df.empty:
        return pd.DataFrame()
    trades_df = trades_df.copy()
    trades_df['iso_week'] = trades_df['exit_time'].dt.isocalendar().week.astype(int)
    grouped = trades_df.groupby('iso_week').agg(
        count=('pnl', 'count'),
        pnl=('pnl', 'sum'),
        win_rate=('pnl', lambda x: (x > 0).mean()),
    ).reset_index()
    grouped['cumulative_pnl'] = grouped['pnl'].cumsum()
    return grouped


def regime_breakdown(trades_df):
    """Per-regime stats."""
    if trades_df.empty:
        return pd.DataFrame()
    trades_df = trades_df.copy()
    trades_df['regime_name'] = trades_df['regime_at_entry'].map(REGIME_NAMES)
    grouped = trades_df.groupby('regime_name').agg(
        count=('pnl', 'count'),
        win_rate=('pnl', lambda x: (x > 0).mean()),
        avg_pnl=('pnl', 'mean'),
        total_pnl=('pnl', 'sum'),
        avg_hold=('bars_held', 'mean'),
    ).reset_index()
    return grouped


def trade_type_breakdown(trades_df):
    """Per trade type stats."""
    if trades_df.empty:
        return pd.DataFrame()
    grouped = trades_df.groupby('trade_type').agg(
        count=('pnl', 'count'),
        win_rate=('pnl', lambda x: (x > 0).mean()),
        avg_pnl=('pnl', 'mean'),
        total_pnl=('pnl', 'sum'),
        avg_hold=('bars_held', 'mean'),
    ).reset_index()
    if 'tf' in trades_df.columns:
        tf_lists = trades_df.groupby('trade_type')['tf'].apply(lambda x: ', '.join(sorted(x.unique()))).reset_index()
        tf_lists.columns = ['trade_type', 'source_tfs']
        grouped = grouped.merge(tf_lists, on='trade_type', how='left')
    return grouped


def named_period_analysis(trades_df):
    """Compute stats for each named historical period."""
    results = []
    for name, start, end in NAMED_PERIODS:
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)
        mask = (trades_df['exit_time'] >= start_dt) & (trades_df['exit_time'] <= end_dt)
        period_trades = trades_df[mask]
        if period_trades.empty:
            results.append({
                'period': name, 'start': start, 'end': end,
                'trades': 0, 'pnl': 0, 'win_rate': 0, 'avg_pnl': 0,
            })
        else:
            results.append({
                'period': name, 'start': start, 'end': end,
                'trades': len(period_trades),
                'pnl': round(period_trades['pnl'].sum(), 2),
                'win_rate': round((period_trades['pnl'] > 0).mean(), 4),
                'avg_pnl': round(period_trades['pnl'].mean(), 2),
            })
    return pd.DataFrame(results)


def per_tf_summary(all_trades_df, tf_configs):
    """Per-TF summary: config, trades, ROI, max DD, best/worst month."""
    results = []
    for tf in sorted(all_trades_df['tf'].unique()):
        tf_trades = all_trades_df[all_trades_df['tf'] == tf]
        cfg = tf_configs.get(tf, {})
        total_pnl = tf_trades['pnl'].sum()
        roi = total_pnl / STARTING_BALANCE * 100

        # Max drawdown from cumulative P&L
        cum_pnl = tf_trades['pnl'].cumsum() + STARTING_BALANCE
        peak = cum_pnl.cummax()
        dd = (peak - cum_pnl) / peak
        max_dd = dd.max() * 100 if len(dd) > 0 else 0

        # Best/worst month
        tf_trades_c = tf_trades.copy()
        tf_trades_c['ym'] = tf_trades_c['exit_time'].dt.to_period('M')
        monthly = tf_trades_c.groupby('ym')['pnl'].sum()
        best_month = f"{monthly.idxmax()} (${monthly.max():,.0f})" if len(monthly) > 0 else "N/A"
        worst_month = f"{monthly.idxmin()} (${monthly.min():,.0f})" if len(monthly) > 0 else "N/A"

        results.append({
            'tf': tf,
            'leverage': cfg.get('leverage', 'N/A'),
            'risk_pct': cfg.get('risk_pct', 'N/A'),
            'stop_atr': cfg.get('stop_atr', 'N/A'),
            'rr': cfg.get('rr', 'N/A'),
            'trades': len(tf_trades),
            'win_rate': round((tf_trades['pnl'] > 0).mean(), 4) if len(tf_trades) > 0 else 0,
            'total_pnl': round(total_pnl, 2),
            'roi_pct': round(roi, 2),
            'max_dd_pct': round(max_dd, 2),
            'best_month': best_month,
            'worst_month': worst_month,
        })
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# HTML heatmap generator
# ---------------------------------------------------------------------------
def generate_heatmap_html(pnl_df, output_path):
    """Generate standalone HTML with color-coded monthly P&L heatmap."""
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    if pnl_df.empty:
        html = "<html><body><h2>No trade data available for heatmap</h2></body></html>"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        return

    max_abs = max(abs(pnl_df.values.max()), abs(pnl_df.values.min()), 1)

    rows_html = ""
    for year in sorted(pnl_df.index):
        row = f'<tr><td style="font-weight:bold;padding:6px 12px;border:1px solid #ddd;">{year}</td>'
        year_total = 0.0
        for m in range(1, 13):
            val = pnl_df.loc[year, m] if m in pnl_df.columns else 0.0
            year_total += val
            intensity = min(abs(val) / max_abs, 1.0)
            if val > 0:
                r, g, b = int(255 * (1 - intensity * 0.7)), 255, int(255 * (1 - intensity * 0.7))
            elif val < 0:
                r, g, b = 255, int(255 * (1 - intensity * 0.7)), int(255 * (1 - intensity * 0.7))
            else:
                r, g, b = 245, 245, 245
            color = f"rgb({r},{g},{b})"
            row += f'<td style="padding:6px 10px;text-align:right;border:1px solid #ddd;background:{color};">${val:,.0f}</td>'
        # Year total
        total_color = "#d4edda" if year_total > 0 else "#f8d7da" if year_total < 0 else "#f5f5f5"
        row += f'<td style="padding:6px 10px;text-align:right;border:1px solid #ddd;background:{total_color};font-weight:bold;">${year_total:,.0f}</td>'
        row += '</tr>'
        rows_html += row

    header = '<tr><th style="padding:6px 12px;border:1px solid #ddd;">Year</th>'
    for mn in month_names:
        header += f'<th style="padding:6px 10px;border:1px solid #ddd;">{mn}</th>'
    header += '<th style="padding:6px 10px;border:1px solid #ddd;">Total</th></tr>'

    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Backtesting Audit - Monthly P&amp;L Heatmap</title></head>
<body style="font-family:Consolas,monospace;margin:20px;background:#fafafa;">
<h2 style="color:#333;">Monthly P&amp;L Heatmap - Backtesting Audit</h2>
<p style="color:#666;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Starting Balance: ${STARTING_BALANCE:,.0f}</p>
<table style="border-collapse:collapse;font-size:13px;">
{header}
{rows_html}
</table>
<br>
<div style="display:flex;align-items:center;gap:20px;font-size:12px;color:#666;">
  <span>Legend:</span>
  <span style="display:inline-block;width:20px;height:14px;background:rgb(77,255,77);border:1px solid #ccc;"></span> Profit
  <span style="display:inline-block;width:20px;height:14px;background:rgb(255,77,77);border:1px solid #ccc;"></span> Loss
  <span style="display:inline-block;width:20px;height:14px;background:rgb(245,245,245);border:1px solid #ccc;"></span> Break-even
  <span>| Intensity proportional to magnitude (max: ${max_abs:,.0f})</span>
</div>
</body>
</html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)


# ---------------------------------------------------------------------------
# Text report writer
# ---------------------------------------------------------------------------
def write_text_report(output_path, tf_summary, regime_df, type_df, periods_df, weekly_df, monthly_data, all_trades_df, confluence_stats=None):
    """Write human-readable text tables."""
    lines = []
    lines.append("=" * 80)
    lines.append("  BACKTESTING AUDIT REPORT")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Starting Balance: ${STARTING_BALANCE:,.0f}")
    lines.append("=" * 80)

    # Overall summary
    if not all_trades_df.empty:
        total_pnl = all_trades_df['pnl'].sum()
        total_trades = len(all_trades_df)
        win_rate = (all_trades_df['pnl'] > 0).mean()
        lines.append(f"\n  OVERALL: {total_trades:,} trades | P&L: ${total_pnl:,.2f} | "
                     f"ROI: {total_pnl/STARTING_BALANCE*100:.1f}% | Win Rate: {win_rate:.1%}")

    # Per-TF summary
    lines.append(f"\n\n{'='*80}")
    lines.append("  PER-TIMEFRAME SUMMARY")
    lines.append(f"{'='*80}")
    if not tf_summary.empty:
        lines.append(f"  {'TF':<5s} {'Lev':>5s} {'Risk%':>6s} {'SL':>5s} {'RR':>5s} "
                     f"{'Trades':>7s} {'WR':>6s} {'P&L':>12s} {'ROI%':>8s} {'MaxDD%':>7s}")
        lines.append(f"  {'-'*76}")
        for _, row in tf_summary.iterrows():
            lines.append(f"  {row['tf']:<5s} {str(row['leverage']):>5s} {str(row['risk_pct']):>6s} "
                         f"{str(row['stop_atr']):>5s} {str(row['rr']):>5s} "
                         f"{row['trades']:>7d} {row['win_rate']:>5.1%} "
                         f"${row['total_pnl']:>10,.0f} {row['roi_pct']:>7.1f}% {row['max_dd_pct']:>6.1f}%")

    # Regime breakdown
    lines.append(f"\n\n{'='*80}")
    lines.append("  REGIME BREAKDOWN")
    lines.append(f"{'='*80}")
    if not regime_df.empty:
        lines.append(f"  {'Regime':<12s} {'Trades':>7s} {'WR':>6s} {'Avg P&L':>10s} {'Total P&L':>12s} {'Avg Hold':>9s}")
        lines.append(f"  {'-'*60}")
        for _, row in regime_df.iterrows():
            lines.append(f"  {row['regime_name']:<12s} {row['count']:>7d} {row['win_rate']:>5.1%} "
                         f"${row['avg_pnl']:>9,.2f} ${row['total_pnl']:>10,.0f} {row['avg_hold']:>8.1f}")

    # Trade type breakdown
    lines.append(f"\n\n{'='*80}")
    lines.append("  TRADE TYPE BREAKDOWN")
    lines.append(f"{'='*80}")
    if not type_df.empty:
        lines.append(f"  {'Type':<12s} {'Trades':>7s} {'WR':>6s} {'Avg P&L':>10s} {'Total P&L':>12s} {'Avg Hold':>9s}")
        lines.append(f"  {'-'*60}")
        for _, row in type_df.iterrows():
            lines.append(f"  {row['trade_type']:<12s} {row['count']:>7d} {row['win_rate']:>5.1%} "
                         f"${row['avg_pnl']:>9,.2f} ${row['total_pnl']:>10,.0f} {row['avg_hold']:>8.1f}")

    # Named periods
    lines.append(f"\n\n{'='*80}")
    lines.append("  NAMED PERIOD ANALYSIS")
    lines.append(f"{'='*80}")
    if not periods_df.empty:
        lines.append(f"  {'Period':<24s} {'Dates':>23s} {'Trades':>7s} {'P&L':>12s} {'WR':>6s} {'Avg P&L':>10s}")
        lines.append(f"  {'-'*86}")
        for _, row in periods_df.iterrows():
            date_range = f"{row['start']} - {row['end']}"
            lines.append(f"  {row['period']:<24s} {date_range:>23s} {row['trades']:>7d} "
                         f"${row['pnl']:>10,.0f} {row['win_rate']:>5.1%} ${row['avg_pnl']:>9,.2f}")

    # Confluence Filter Impact
    if confluence_stats is not None:
        lines.append(f"\n\n{'='*80}")
        lines.append("  CONFLUENCE FILTER IMPACT")
        lines.append(f"{'='*80}")
        total_blocks = confluence_stats.get('confluence_blocks', 0)
        lines.append(f"  Total trades blocked by parent TF opposition: {total_blocks:,}")
        blocked = confluence_stats.get('confluence_blocked_trades', [])
        if blocked:
            per_tf = _confluence_blocks_per_tf(confluence_stats)
            lines.append(f"  {'TF':<6s} {'Blocked':>8s} {'Parent TF':<10s}")
            lines.append(f"  {'-'*30}")
            for tf_key in ['5m', '15m', '1h', '4h']:
                count = per_tf.get(tf_key, 0)
                parent = TF_PARENT_MAP.get(tf_key, 'N/A')
                if count > 0:
                    lines.append(f"  {tf_key:<6s} {count:>8d} {parent:<10s}")
            long_blocks = sum(1 for b in blocked if b['direction'] == 1)
            short_blocks = sum(1 for b in blocked if b['direction'] == -1)
            lines.append(f"\n  LONG blocks: {long_blocks:,} | SHORT blocks: {short_blocks:,}")

    # Monthly P&L table
    lines.append(f"\n\n{'='*80}")
    lines.append("  MONTHLY P&L")
    lines.append(f"{'='*80}")
    pnl_df = monthly_data.get('pnl', pd.DataFrame())
    if not pnl_df.empty:
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        header = f"  {'Year':<6s}" + "".join(f"{m:>8s}" for m in month_labels) + f"{'Total':>10s}"
        lines.append(header)
        lines.append(f"  {'-'*(6 + 8*12 + 10)}")
        for year in sorted(pnl_df.index):
            vals = []
            total = 0.0
            for m in range(1, 13):
                v = pnl_df.loc[year, m] if m in pnl_df.columns else 0.0
                total += v
                vals.append(f"${v:>6,.0f}")
            lines.append(f"  {year:<6d}" + "".join(f"{v:>8s}" for v in vals) + f"${total:>8,.0f}")

    text = "\n".join(lines) + "\n"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Backtesting Audit Tool')
    parser.add_argument('--tf', nargs='+', default=['5m', '15m', '1h', '4h', '1d', '1w'],
                        help='Timeframes to audit')
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  BACKTESTING AUDIT")
    print(f"  GPU Array: {'CuPy (CUDA)' if GPU_ARRAY else 'NumPy (CPU)'}")
    print(f"  XGB GPU:   {'ENABLED' if USE_GPU_XGB else 'CPU'}")
    print(f"  Fee model: {FEE_RATE*100:.2f}% round-trip")
    print(f"  Starting:  ${STARTING_BALANCE:,.0f}")
    print(f"  TFs:       {args.tf}")
    print(f"  Started:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}", flush=True)

    # Build regime multiplier arrays for Numba (indexed by regime int)
    regime_lev = np.array([REGIME_MULT[i]['lev'] for i in range(4)], dtype=np.float64)
    regime_risk = np.array([REGIME_MULT[i]['risk'] for i in range(4)], dtype=np.float64)
    regime_stop = np.array([REGIME_MULT[i]['stop'] for i in range(4)], dtype=np.float64)
    regime_rr = np.array([REGIME_MULT[i]['rr'] for i in range(4)], dtype=np.float64)
    regime_hold = np.array([REGIME_MULT[i]['hold'] for i in range(4)], dtype=np.float64)

    all_trades_list = []
    all_equity = {}
    tf_configs = {}

    for tf in args.tf:
        print(f"\n{elapsed()} Loading {tf.upper()}...", flush=True)

        data = load_full_history(tf)
        if data is None:
            print(f"  Skipping {tf}", flush=True)
            continue

        config = load_tf_config(tf)
        tf_configs[tf] = config
        print(f"  Config: {config.get('leverage',0):.0f}x lev, {config.get('risk_pct',0):.2f}% risk, "
              f"{config.get('stop_atr',0):.2f}ATR, {config.get('rr',0):.1f}:1 RR, "
              f"{config.get('max_hold',0)} bar, conf>{config.get('conf_thresh',0):.2f}", flush=True)

        # Detect regimes
        regimes = detect_regime_series(data['closes'])
        regime_counts = {REGIME_NAMES[i]: int(np.sum(regimes == i)) for i in range(4)}
        print(f"  Regimes: {regime_counts}", flush=True)

        # Convert timestamps to int64 for Numba
        ts_i64 = data['timestamps'].astype('datetime64[ns]').astype(np.int64)

        # Run simulation
        print(f"  Simulating {data['n_bars']:,} bars...", flush=True)
        t0 = time.time()

        equity, trades_arr, n_trades = _simulate_with_trades(
            ts_i64, data['directions'], data['confidences'],
            data['closes'], data['highs'], data['lows'], data['atrs'], regimes,
            float(config['leverage']), float(config['risk_pct']),
            float(config['stop_atr']), float(config['rr']),
            int(config['max_hold']), int(config.get('exit_type', 0)),
            float(config['conf_thresh']), FEE_RATE, STARTING_BALANCE,
            regime_lev, regime_risk, regime_stop, regime_rr, regime_hold,
            float(TF_SLIPPAGE.get(tf, 0.0002)),
        )

        sim_time = time.time() - t0
        final_bal = equity[-1] if len(equity) > 0 else STARTING_BALANCE
        roi = (final_bal - STARTING_BALANCE) / STARTING_BALANCE * 100
        print(f"  Done in {sim_time:.1f}s | {n_trades:,} trades | Final: ${final_bal:,.2f} | ROI: {roi:.1f}%", flush=True)

        all_equity[tf] = equity

        # Convert trades array to DataFrame
        if n_trades > 0:
            trades_df = pd.DataFrame(trades_arr, columns=[
                'entry_bar', 'exit_bar', 'direction', 'entry_price', 'exit_price',
                'bars_held', 'pnl', 'pnl_pct', 'confidence', 'leverage_used',
                'risk_used', 'regime_at_entry',
            ])
            trades_df['tf'] = tf

            # Map bar indices to timestamps
            ts_series = pd.Series(data['timestamps'])
            trades_df['entry_time'] = ts_series.iloc[trades_df['entry_bar'].astype(int).values].values
            trades_df['exit_time'] = ts_series.iloc[trades_df['exit_bar'].astype(int).values].values

            trades_df = classify_trades(trades_df, tf)
            all_trades_list.append(trades_df)

    if not all_trades_list:
        print(f"\n{elapsed()} No trades generated across any TF. Nothing to report.", flush=True)
        return

    # Merge all trades
    all_trades_df = pd.concat(all_trades_list, ignore_index=True)
    all_trades_df = all_trades_df.sort_values('exit_time').reset_index(drop=True)
    print(f"\n{elapsed()} Total trades across all TFs: {len(all_trades_df):,}", flush=True)

    # Generate reports
    print(f"{elapsed()} Generating reports...", flush=True)

    monthly_data = monthly_heatmap(all_trades_df)
    weekly_df = weekly_performance(all_trades_df)
    regime_df = regime_breakdown(all_trades_df)
    type_df = trade_type_breakdown(all_trades_df)
    periods_df = named_period_analysis(all_trades_df)
    tf_summary = per_tf_summary(all_trades_df, tf_configs)

    # --- Output 1: JSON ---
    json_path = os.path.join(DB_DIR, 'audit_report.json')
    report_json = {
        'generated': datetime.now().isoformat(),
        'starting_balance': STARTING_BALANCE,
        'fee_rate': FEE_RATE,
        'timeframes': list(tf_configs.keys()),
        'configs': {tf: {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
                         for k, v in cfg.items()} for tf, cfg in tf_configs.items()},
        'overall': {
            'total_trades': len(all_trades_df),
            'total_pnl': round(float(all_trades_df['pnl'].sum()), 2),
            'win_rate': round(float((all_trades_df['pnl'] > 0).mean()), 4),
            'roi_pct': round(float(all_trades_df['pnl'].sum() / STARTING_BALANCE * 100), 2),
        },
        'per_tf': tf_summary.to_dict(orient='records') if not tf_summary.empty else [],
        'regime_breakdown': regime_df.to_dict(orient='records') if not regime_df.empty else [],
        'trade_type_breakdown': type_df.to_dict(orient='records') if not type_df.empty else [],
        'named_periods': periods_df.to_dict(orient='records') if not periods_df.empty else [],
        'weekly': weekly_df.to_dict(orient='records') if not weekly_df.empty else [],
        'monthly_pnl': monthly_data['pnl'].to_dict() if not monthly_data['pnl'].empty else {},
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report_json, f, indent=2, default=str)
    print(f"  Saved: {json_path}", flush=True)

    # --- Output 2: Text report ---
    txt_path = os.path.join(DB_DIR, 'audit_report.txt')
    write_text_report(txt_path, tf_summary, regime_df, type_df, periods_df, weekly_df, monthly_data, all_trades_df)
    print(f"  Saved: {txt_path}", flush=True)

    # --- Output 3: HTML heatmap ---
    html_path = os.path.join(DB_DIR, 'audit_heatmap.html')
    generate_heatmap_html(monthly_data['pnl'], html_path)
    print(f"  Saved: {html_path}", flush=True)

    # Print summary to console
    print(f"\n{'='*70}")
    print(f"  AUDIT COMPLETE")
    print(f"{'='*70}")
    total_pnl = all_trades_df['pnl'].sum()
    print(f"  Total Trades: {len(all_trades_df):,}")
    print(f"  Total P&L:    ${total_pnl:,.2f}")
    print(f"  ROI:          {total_pnl/STARTING_BALANCE*100:.1f}%")
    print(f"  Win Rate:     {(all_trades_df['pnl'] > 0).mean():.1%}")
    print(f"  TFs audited:  {list(tf_configs.keys())}")
    print(f"  Elapsed:      {time.time() - START_TIME:.1f}s")
    print(f"{'='*70}", flush=True)


# ---------------------------------------------------------------------------
# UNIFIED MULTI-TF SIMULTANEOUS BACKTEST
# ---------------------------------------------------------------------------

# Capital allocation — single source of truth in config.py
from config import TF_CAPITAL_ALLOC, TF_SLIPPAGE, load_tf_allocation

# Bar multiples relative to 5m (how many 5m bars per TF bar)
TF_BAR_MULTIPLES = {
    '5m':  1,
    '15m': 3,
    '1h':  12,
    '4h':  48,
    '1d':  288,
    '1w':  2016,
}


def _load_allocation():
    """Load capital allocation from optimal_allocation.json or config defaults."""
    alloc = load_tf_allocation()
    alloc_path = os.path.join(DB_DIR, 'optimal_allocation.json')
    if os.path.exists(alloc_path):
        print(f"  Loaded allocation from {alloc_path}", flush=True)
    else:
        print(f"  Using default allocation from config.py", flush=True)
    return alloc


def _classify_trade_type(bars_held, tf_name):
    """Classify a single trade by type based on bars held."""
    thresholds = TRADE_THRESHOLDS.get(tf_name, TRADE_THRESHOLDS['1h'])
    if bars_held <= thresholds['scalp']:
        return 'scalp'
    elif bars_held <= thresholds['day']:
        return 'day_trade'
    elif bars_held <= thresholds['swing']:
        return 'swing'
    else:
        return 'position'


def run_unified_backtest(tf_list=None):
    """
    Multi-TF simultaneous simulation.

    Loads all TF models and feature databases, then steps through time
    chronologically aligned to 5-minute bars. At each bar close, checks
    which TFs have a bar closing and processes predictions with portfolio-
    level risk management.

    Returns:
        all_trades: list of trade dicts
        equity_curve: dict of timestamp -> equity
        portfolio_stats: dict of summary statistics
    """
    if tf_list is None:
        tf_list = ['5m', '15m', '1h', '4h', '1d', '1w']

    print(f"\n{'='*70}")
    print(f"  UNIFIED MULTI-TF SIMULTANEOUS BACKTEST")
    print(f"  GPU Array: {'CuPy (CUDA)' if GPU_ARRAY else 'NumPy (CPU)'}")
    print(f"  XGB GPU:   {'ENABLED' if USE_GPU_XGB else 'CPU'}")
    print(f"  Fee model: {FEE_RATE*100:.2f}% round-trip")
    print(f"  Starting:  ${STARTING_BALANCE:,.0f}")
    print(f"  TFs:       {tf_list}")
    print(f"  Started:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}", flush=True)

    # --- Load allocation ---
    allocation = _load_allocation()

    # --- Load all TF data & configs ---
    tf_data = {}
    tf_configs = {}

    for tf in tf_list:
        print(f"\n{elapsed()} Loading {tf.upper()}...", flush=True)
        data = load_full_history(tf)
        if data is None:
            print(f"  Skipping {tf}", flush=True)
            continue

        config = load_tf_config(tf)
        tf_configs[tf] = config
        print(f"  Config: {config.get('leverage',0):.0f}x lev, {config.get('risk_pct',0):.2f}% risk, "
              f"{config.get('stop_atr',0):.2f}ATR, {config.get('rr',0):.1f}:1 RR, "
              f"{config.get('max_hold',0)} bar, conf>{config.get('conf_thresh',0):.2f}", flush=True)

        # Detect regimes
        regimes = detect_regime_series(data['closes'])
        data['regimes'] = regimes
        regime_counts = {REGIME_NAMES[i]: int(np.sum(regimes == i)) for i in range(4)}
        print(f"  Regimes: {regime_counts}", flush=True)

        # Convert timestamps to pandas for alignment
        data['ts_pd'] = pd.DatetimeIndex(data['timestamps'])
        tf_data[tf] = data

    if not tf_data:
        print(f"\n{elapsed()} No TF data loaded. Aborting.", flush=True)
        return [], {}, {}

    loaded_tfs = sorted(tf_data.keys(), key=lambda x: TF_BAR_MULTIPLES.get(x, 1))
    print(f"\n{elapsed()} Loaded TFs: {loaded_tfs}", flush=True)

    # --- Build unified 5m timeline ---
    # Find the overlapping time range across all loaded TFs
    all_starts = []
    all_ends = []
    for tf in loaded_tfs:
        ts = tf_data[tf]['ts_pd']
        all_starts.append(ts[0])
        all_ends.append(ts[-1])

    global_start = max(all_starts)
    global_end = min(all_ends)
    print(f"  Overlapping range: {global_start} -> {global_end}", flush=True)

    # Use the 5m timeline as base if available, otherwise the fastest loaded TF
    base_tf = '5m' if '5m' in tf_data else loaded_tfs[0]
    base_ts = tf_data[base_tf]['ts_pd']
    # Restrict to overlapping range
    base_mask = (base_ts >= global_start) & (base_ts <= global_end)
    timeline = base_ts[base_mask]
    timeline_indices = np.where(base_mask)[0]

    print(f"  Timeline bars: {len(timeline):,} (base TF: {base_tf})", flush=True)

    # For each TF, build a lookup: timestamp -> bar index in that TF's data
    tf_ts_to_idx = {}
    for tf in loaded_tfs:
        ts = tf_data[tf]['ts_pd']
        ts_set = {}
        for idx_i in range(len(ts)):
            ts_set[ts[idx_i]] = idx_i
        tf_ts_to_idx[tf] = ts_set

    # --- Portfolio state ---
    balance = STARTING_BALANCE
    peak_balance = STARTING_BALANCE
    per_tf_equity = {tf: STARTING_BALANCE * allocation.get(tf, 0.15) for tf in loaded_tfs}
    per_tf_peak = {tf: per_tf_equity[tf] for tf in loaded_tfs}
    per_tf_suspended = {tf: False for tf in loaded_tfs}
    portfolio_halted = False

    # Active positions: list of dicts
    active_positions = []
    all_trades = []
    equity_history = []

    # Cross-TF confluence filter state
    confluence_signals = {}  # {tf: int} where 1=long, -1=short, 0=flat
    confluence_blocks = 0
    confluence_blocked_trades = []  # hypothetical blocked trades for P&L analysis

    # Regime multiplier arrays
    regime_lev_arr = np.array([REGIME_MULT[i]['lev'] for i in range(4)], dtype=np.float64)
    regime_risk_arr = np.array([REGIME_MULT[i]['risk'] for i in range(4)], dtype=np.float64)
    regime_stop_arr = np.array([REGIME_MULT[i]['stop'] for i in range(4)], dtype=np.float64)
    regime_rr_arr = np.array([REGIME_MULT[i]['rr'] for i in range(4)], dtype=np.float64)
    regime_hold_arr = np.array([REGIME_MULT[i]['hold'] for i in range(4)], dtype=np.float64)

    print(f"\n{elapsed()} Running unified simulation...", flush=True)
    sim_start = time.time()
    last_pct = -1

    n_timeline = len(timeline)
    bar_counter = 0

    for t_idx in range(n_timeline):
        bar_counter += 1
        current_ts = timeline[t_idx]

        # Progress logging every 5%
        pct = int(t_idx / max(n_timeline - 1, 1) * 100)
        if pct >= last_pct + 5:
            last_pct = pct
            elapsed_s = time.time() - sim_start
            bars_per_sec = bar_counter / max(elapsed_s, 0.01)
            print(f"  {pct:3d}% | bar {t_idx:,}/{n_timeline:,} | "
                  f"balance=${balance:,.2f} | trades={len(all_trades)} | "
                  f"active={len(active_positions)} | {bars_per_sec:,.0f} bars/s", flush=True)

        # Get current price from the base TF (or fastest available)
        current_price = None
        for tf in loaded_tfs:
            if current_ts in tf_ts_to_idx[tf]:
                idx = tf_ts_to_idx[tf][current_ts]
                current_price = tf_data[tf]['closes'][idx]
                break
        if current_price is None or current_price <= 0:
            continue

        # --- Update existing positions ---
        still_active = []
        for pos in active_positions:
            pos['bars_held'] += 1
            tf = pos['tf']

            # Get current high/low/close for this TF if bar exists, else use base price
            p_high = current_price
            p_low = current_price
            p_close = current_price
            if current_ts in tf_ts_to_idx[tf]:
                pidx = tf_ts_to_idx[tf][current_ts]
                p_high = tf_data[tf]['highs'][pidx]
                p_low = tf_data[tf]['lows'][pidx]
                p_close = tf_data[tf]['closes'][pidx]

            # Check exits
            hit_sl = False
            hit_tp = False
            exit_price = p_close

            if pos['direction'] == 1:  # LONG
                if p_low <= pos['stop_price']:
                    hit_sl = True
                    exit_price = pos['stop_price']
                elif p_high >= pos['tp_price']:
                    hit_tp = True
                    exit_price = pos['tp_price']
            else:  # SHORT
                if p_high >= pos['stop_price']:
                    hit_sl = True
                    exit_price = pos['stop_price']
                elif p_low <= pos['tp_price']:
                    hit_tp = True
                    exit_price = pos['tp_price']

            time_exit = pos['bars_held'] >= pos['max_hold']

            if hit_sl or hit_tp or time_exit:
                # Apply per-TF slippage to exit price
                exit_slippage = TF_SLIPPAGE.get(tf, 0.0002)
                if pos['direction'] == 1:  # LONG exit: sell at worse (lower) price
                    exit_price *= (1 - exit_slippage)
                else:  # SHORT exit: buy at worse (higher) price
                    exit_price *= (1 + exit_slippage)

                # Compute PnL
                price_chg = (exit_price - pos['entry_price']) / max(pos['entry_price'], 1e-8) * pos['direction']
                gross_pnl = price_chg * pos['leverage']
                fee_cost = FEE_RATE * pos['leverage']
                net_pnl_pct = gross_pnl - fee_cost

                # Partial TP scaling: use trade type partial_tp_pct if available
                partial_tp = pos.get('partial_tp_pct', 0.0)
                if hit_tp and partial_tp > 0:
                    # Take partial_tp_pct at TP, remainder stays (simplified: scale PnL)
                    net_pnl_pct *= partial_tp
                elif hit_tp and pos.get('exit_type', 0) > 0:
                    net_pnl_pct *= (pos['exit_type'] / 100.0)

                # DD scaling
                dd = (peak_balance - balance) / max(peak_balance, 1e-8)
                dd_scale = max(0.0, 1.0 - 2.0 * dd) if dd < 0.15 else 0.0

                pnl_dollar = balance * (pos['risk_pct'] / 100.0) * net_pnl_pct * dd_scale
                balance += pnl_dollar
                balance = max(balance, 0.0)
                peak_balance = max(peak_balance, balance)

                # Update per-TF equity
                per_tf_equity[tf] += pnl_dollar
                per_tf_peak[tf] = max(per_tf_peak[tf], per_tf_equity[tf])

                # Check per-TF DD suspension (25% threshold)
                tf_dd = (per_tf_peak[tf] - per_tf_equity[tf]) / max(per_tf_peak[tf], 1e-8)
                if tf_dd > 0.25:
                    per_tf_suspended[tf] = True

                # Record trade
                close_reason = "SL" if hit_sl else ("TP" if hit_tp else "TIME")
                regime_val = int(pos['regime'])
                trade_rec = {
                    'tf': tf,
                    'direction': pos['direction'],
                    'direction_str': 'LONG' if pos['direction'] == 1 else 'SHORT',
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'entry_time': pos['entry_time'],
                    'exit_time': current_ts,
                    'bars_held': pos['bars_held'],
                    'pnl': pnl_dollar,
                    'pnl_pct': net_pnl_pct * 100,
                    'confidence': pos['confidence'],
                    'leverage_used': pos['leverage'],
                    'risk_used': pos['risk_pct'],
                    'regime_at_entry': regime_val,
                    'regime_name': REGIME_NAMES.get(regime_val, 'unknown'),
                    'trade_type': pos.get('trade_type', _classify_trade_type(pos['bars_held'], tf)),
                    'close_reason': close_reason,
                    'balance_after': balance,
                }
                all_trades.append(trade_rec)
            else:
                still_active.append(pos)

        active_positions = still_active

        # --- Check portfolio-level halt ---
        portfolio_dd = (peak_balance - balance) / max(peak_balance, 1e-8)
        if portfolio_dd > 0.15:
            portfolio_halted = True
        elif portfolio_dd < 0.10:
            # Reset halt when DD recovers below 10%
            portfolio_halted = False

        # --- Entry logic: check each TF with a closing bar ---
        if balance > 0 and not portfolio_halted:
            # Count same-direction active positions for correlation adjustment
            long_count = sum(1 for p in active_positions if p['direction'] == 1)
            short_count = sum(1 for p in active_positions if p['direction'] == -1)

            # Total current heat
            total_heat = sum(
                (balance * p['risk_pct'] / 100.0) for p in active_positions
            ) / max(balance, 1e-8)

            for tf in loaded_tfs:
                if per_tf_suspended[tf]:
                    continue

                # Check if this TF has a bar closing at current timestamp
                if current_ts not in tf_ts_to_idx[tf]:
                    continue

                bar_idx = tf_ts_to_idx[tf][current_ts]
                data = tf_data[tf]
                config = tf_configs[tf]
                conf_thresh = float(config['conf_thresh'])

                # Get prediction for this bar
                direction = int(data['directions'][bar_idx])
                confidence = float(data['confidences'][bar_idx])

                # Update confluence signal for this TF (always, even if no trade)
                if direction != 0 and confidence > conf_thresh:
                    confluence_signals[tf] = direction
                else:
                    confluence_signals[tf] = 0

                if direction == 0 or confidence <= conf_thresh:
                    continue

                # --- Confluence filter: check parent TF direction ---
                parent_tf = TF_PARENT_MAP.get(tf)
                confluence_scale = 1.0
                if parent_tf is not None:
                    parent_dir = confluence_signals.get(parent_tf, 0)
                    if parent_dir != 0 and parent_dir != direction:
                        # Parent opposite -> BLOCK, record hypothetical trade
                        confluence_blocks += 1
                        confluence_blocked_trades.append({
                            'tf': tf,
                            'direction': direction,
                            'confidence': confidence,
                            'entry_price': float(data['closes'][bar_idx]),
                            'entry_time': current_ts,
                            'parent_tf': parent_tf,
                            'parent_dir': parent_dir,
                        })
                        continue
                    elif parent_dir == 0:
                        # Parent flat -> half size
                        confluence_scale = 0.5

                # Check max concurrent per TF (max 2 per TF)
                tf_active = sum(1 for p in active_positions if p['tf'] == tf)
                if tf_active >= 2:
                    continue

                # Check total heat (3% max)
                new_risk_pct = float(config['risk_pct'])
                new_heat = new_risk_pct / 100.0
                if total_heat + new_heat > 0.03:
                    continue

                # Per-TF DD check (25% halts that TF)
                tf_dd = (per_tf_peak[tf] - per_tf_equity[tf]) / max(per_tf_peak[tf], 1e-8)
                if tf_dd > 0.25:
                    per_tf_suspended[tf] = True
                    continue

                # Regime adjustments
                r = int(data['regimes'][bar_idx])
                eff_lev = float(config['leverage']) * regime_lev_arr[r]
                eff_risk = new_risk_pct * regime_risk_arr[r]
                eff_stop = float(config['stop_atr']) * regime_stop_arr[r]
                eff_rr = float(config['rr']) * regime_rr_arr[r]
                eff_hold = float(config['max_hold']) * regime_hold_arr[r]

                # Classify expected trade type
                expected_max_hold = max(1, int(eff_hold))
                trade_type = _classify_trade_type(expected_max_hold, tf)
                tt_params = TRADE_TYPE_PARAMS.get(trade_type, TRADE_TYPE_PARAMS['day_trade'])

                # Apply trade type modifiers to SL and TP
                eff_stop /= tt_params['sl_tightness']    # wider for swing/position
                eff_rr /= tt_params['tp_aggression']     # adjusted RR

                # Kelly sizing
                kelly_f = (confidence - conf_thresh) / max(1.0 - conf_thresh, 1e-8)
                kelly_risk = eff_risk * (1.0 + 0.25 * kelly_f)
                kelly_risk = min(kelly_risk, eff_risk * 3.0)

                # Apply confluence scale (half size if parent flat)
                kelly_risk *= confluence_scale

                # Correlation adjustment: 3+ same direction -> reduce by 50%
                same_dir_count = long_count if direction == 1 else short_count

                # Use trade-type-aware max_correlation_positions
                max_corr_pos = tt_params['max_correlation_positions']
                if same_dir_count >= max_corr_pos + 2:
                    kelly_risk *= 0.5
                elif same_dir_count >= max_corr_pos:
                    kelly_risk *= 0.75

                # DD scaling on entry
                dd = (peak_balance - balance) / max(peak_balance, 1e-8)
                if dd >= 0.15:
                    continue

                # Compute stop/TP with per-TF slippage
                atr_val = float(data['atrs'][bar_idx])
                raw_entry_price = float(data['closes'][bar_idx])
                slippage = TF_SLIPPAGE.get(tf, 0.0002)

                # Apply slippage to entry price
                if direction == 1:  # LONG: buy at worse (higher) price
                    entry_price = raw_entry_price * (1 + slippage)
                else:  # SHORT: sell at worse (lower) price
                    entry_price = raw_entry_price * (1 - slippage)

                sl_dist = eff_stop * atr_val
                tp_dist = sl_dist * eff_rr

                if direction == 1:
                    stop_price = entry_price - sl_dist
                    tp_price = entry_price + tp_dist
                else:
                    stop_price = entry_price + sl_dist
                    tp_price = entry_price - tp_dist

                # Open position
                pos = {
                    'tf': tf,
                    'direction': direction,
                    'entry_price': entry_price,
                    'entry_time': current_ts,
                    'stop_price': stop_price,
                    'tp_price': tp_price,
                    'leverage': eff_lev,
                    'risk_pct': kelly_risk,
                    'max_hold': expected_max_hold,
                    'bars_held': 0,
                    'confidence': confidence,
                    'regime': r,
                    'exit_type': int(config.get('exit_type', 0)),
                    'trade_type': trade_type,
                    'partial_tp_pct': tt_params['partial_tp_pct'],
                }
                active_positions.append(pos)

                # Update heat tracking
                total_heat += kelly_risk / 100.0
                if direction == 1:
                    long_count += 1
                else:
                    short_count += 1

        # Record equity snapshot (sample every 100 bars to save memory)
        if t_idx % 100 == 0 or t_idx == n_timeline - 1:
            equity_history.append({
                'timestamp': current_ts,
                'balance': balance,
                'active_positions': len(active_positions),
            })

    sim_elapsed = time.time() - sim_start
    print(f"\n{elapsed()} Simulation complete in {sim_elapsed:.1f}s", flush=True)
    print(f"  Total trades: {len(all_trades):,}", flush=True)
    print(f"  Confluence blocks: {confluence_blocks:,}", flush=True)
    print(f"  Final balance: ${balance:,.2f}", flush=True)
    print(f"  ROI: {(balance - STARTING_BALANCE) / STARTING_BALANCE * 100:.1f}%", flush=True)
    print(f"  Peak: ${peak_balance:,.2f}", flush=True)
    print(f"  Portfolio DD: {(peak_balance - balance) / max(peak_balance, 1e-8) * 100:.1f}%", flush=True)

    # Per-TF stats
    print(f"\n  Per-TF equity:", flush=True)
    for tf in loaded_tfs:
        tf_dd = (per_tf_peak[tf] - per_tf_equity[tf]) / max(per_tf_peak[tf], 1e-8) * 100
        tf_trades = sum(1 for t in all_trades if t['tf'] == tf)
        tf_blocks = sum(1 for b in confluence_blocked_trades if b['tf'] == tf)
        status = "SUSPENDED" if per_tf_suspended[tf] else "active"
        print(f"    {tf:>4s}: ${per_tf_equity[tf]:>10,.2f} | DD={tf_dd:.1f}% | "
              f"trades={tf_trades} | blocked={tf_blocks} | [{status}]", flush=True)

    # Trade type stats
    if all_trades:
        tt_counts = {}
        for t in all_trades:
            tt = t.get('trade_type', 'unknown')
            tt_counts[tt] = tt_counts.get(tt, 0) + 1
        print(f"\n  Trade types: {tt_counts}", flush=True)

    # Build portfolio stats
    portfolio_stats = {
        'starting_balance': STARTING_BALANCE,
        'final_balance': balance,
        'peak_balance': peak_balance,
        'roi_pct': (balance - STARTING_BALANCE) / STARTING_BALANCE * 100,
        'total_trades': len(all_trades),
        'portfolio_dd_pct': (peak_balance - balance) / max(peak_balance, 1e-8) * 100,
        'per_tf_equity': {tf: per_tf_equity[tf] for tf in loaded_tfs},
        'per_tf_suspended': {tf: per_tf_suspended[tf] for tf in loaded_tfs},
        'simulation_time_s': sim_elapsed,
        'configs': tf_configs,
        'allocation': allocation,
        'confluence_blocks': confluence_blocks,
        'confluence_blocked_trades': confluence_blocked_trades,
    }

    return all_trades, equity_history, portfolio_stats


# ---------------------------------------------------------------------------
# Unified output: console print functions
# ---------------------------------------------------------------------------

def print_monthly_heatmap(trades):
    """Year x Month P&L heatmap with ANSI colors to console."""
    if not trades:
        print("\n  No trades to display in monthly heatmap.")
        return

    trades_df = pd.DataFrame(trades)
    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
    trades_df['year'] = trades_df['exit_time'].dt.year
    trades_df['month'] = trades_df['exit_time'].dt.month

    pnl = trades_df.groupby(['year', 'month'])['pnl'].sum().unstack(fill_value=0)
    count = trades_df.groupby(['year', 'month'])['pnl'].count().unstack(fill_value=0)

    # Ensure all 12 months
    for m in range(1, 13):
        if m not in pnl.columns:
            pnl[m] = 0.0
            count[m] = 0
    pnl = pnl[sorted(pnl.columns)]
    count = count[sorted(count.columns)]

    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # ANSI color codes
    GREEN = '\033[92m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    RESET = '\033[0m'
    DIM = '\033[2m'

    print(f"\n{'='*110}")
    print(f"  {BOLD}MONTHLY P&L HEATMAP (Unified Multi-TF){RESET}")
    print(f"{'='*110}")

    # Header
    header = f"  {'Year':<6s}"
    for mn in month_labels:
        header += f"{mn:>9s}"
    header += f"{'Total':>11s}  {'Trades':>7s}"
    print(header)
    print(f"  {'-'*108}")

    for year in sorted(pnl.index):
        row = f"  {year:<6d}"
        year_total = 0.0
        year_trades = 0
        for m in range(1, 13):
            val = pnl.loc[year, m] if m in pnl.columns else 0.0
            cnt = int(count.loc[year, m]) if m in count.columns else 0
            year_total += val
            year_trades += cnt
            if val > 0:
                row += f"{GREEN}${val:>7,.0f}{RESET}"
            elif val < 0:
                row += f"{RED}${val:>7,.0f}{RESET}"
            else:
                row += f"{DIM}${val:>7,.0f}{RESET}"
        # Year total
        if year_total > 0:
            row += f"  {GREEN}{BOLD}${year_total:>8,.0f}{RESET}"
        elif year_total < 0:
            row += f"  {RED}{BOLD}${year_total:>8,.0f}{RESET}"
        else:
            row += f"  {DIM}${year_total:>8,.0f}{RESET}"
        row += f"  {year_trades:>7d}"
        print(row)

    # Grand total
    grand_total = pnl.values.sum()
    grand_trades = int(count.values.sum())
    print(f"  {'-'*108}")
    color = GREEN if grand_total > 0 else RED
    print(f"  {'TOTAL':<6s}" + " " * 99 + f"{color}{BOLD}${grand_total:>8,.0f}{RESET}  {grand_trades:>7d}")
    print(f"{'='*110}\n")


def print_weekly_summary(trades):
    """ISO week x year summary table."""
    if not trades:
        print("\n  No trades for weekly summary.")
        return

    trades_df = pd.DataFrame(trades)
    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
    trades_df['year'] = trades_df['exit_time'].dt.year
    trades_df['iso_week'] = trades_df['exit_time'].dt.isocalendar().week.astype(int)
    trades_df['year_week'] = trades_df['year'].astype(str) + '-W' + trades_df['iso_week'].astype(str).str.zfill(2)

    weekly = trades_df.groupby('year_week').agg(
        count=('pnl', 'count'),
        pnl=('pnl', 'sum'),
        win_rate=('pnl', lambda x: (x > 0).mean()),
        avg_pnl=('pnl', 'mean'),
    ).reset_index()
    weekly['cumulative_pnl'] = weekly['pnl'].cumsum()
    weekly = weekly.sort_values('year_week')

    GREEN = '\033[92m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

    print(f"\n{'='*90}")
    print(f"  {BOLD}WEEKLY P&L SUMMARY (Unified Multi-TF){RESET}")
    print(f"{'='*90}")
    print(f"  {'Week':<10s} {'Trades':>7s} {'P&L':>12s} {'Cum P&L':>12s} {'WR':>7s} {'Avg P&L':>10s}")
    print(f"  {'-'*62}")

    # Show last 52 weeks + first/last summaries if too many
    if len(weekly) > 104:
        # Show first 26
        for _, row in weekly.head(26).iterrows():
            color = GREEN if row['pnl'] > 0 else RED
            print(f"  {row['year_week']:<10s} {row['count']:>7d} "
                  f"{color}${row['pnl']:>10,.0f}{RESET} ${row['cumulative_pnl']:>10,.0f} "
                  f"{row['win_rate']:>6.0%} ${row['avg_pnl']:>9,.2f}")
        print(f"  {'... ':>10s} ({len(weekly) - 52} weeks omitted)")
        # Show last 26
        for _, row in weekly.tail(26).iterrows():
            color = GREEN if row['pnl'] > 0 else RED
            print(f"  {row['year_week']:<10s} {row['count']:>7d} "
                  f"{color}${row['pnl']:>10,.0f}{RESET} ${row['cumulative_pnl']:>10,.0f} "
                  f"{row['win_rate']:>6.0%} ${row['avg_pnl']:>9,.2f}")
    else:
        for _, row in weekly.iterrows():
            color = GREEN if row['pnl'] > 0 else RED
            print(f"  {row['year_week']:<10s} {row['count']:>7d} "
                  f"{color}${row['pnl']:>10,.0f}{RESET} ${row['cumulative_pnl']:>10,.0f} "
                  f"{row['win_rate']:>6.0%} ${row['avg_pnl']:>9,.2f}")

    # Summary stats
    best_week = weekly.loc[weekly['pnl'].idxmax()]
    worst_week = weekly.loc[weekly['pnl'].idxmin()]
    avg_weekly = weekly['pnl'].mean()
    median_weekly = weekly['pnl'].median()
    positive_weeks = (weekly['pnl'] > 0).sum()
    total_weeks = len(weekly)

    print(f"\n  Summary: {positive_weeks}/{total_weeks} profitable weeks ({positive_weeks/max(total_weeks,1)*100:.0f}%)")
    print(f"  Best:    {best_week['year_week']} (${best_week['pnl']:,.0f})")
    print(f"  Worst:   {worst_week['year_week']} (${worst_week['pnl']:,.0f})")
    print(f"  Mean:    ${avg_weekly:,.2f} | Median: ${median_weekly:,.2f}")
    print(f"{'='*90}\n")


def print_trade_type_breakdown(trades):
    """Scalp/Day/Swing/Position counts and P&L per month."""
    if not trades:
        print("\n  No trades for trade type breakdown.")
        return

    trades_df = pd.DataFrame(trades)
    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])

    GREEN = '\033[92m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

    # Overall by type
    print(f"\n{'='*100}")
    print(f"  {BOLD}TRADE TYPE BREAKDOWN (Unified Multi-TF){RESET}")
    print(f"{'='*100}")

    type_stats = trades_df.groupby('trade_type').agg(
        count=('pnl', 'count'),
        win_rate=('pnl', lambda x: (x > 0).mean()),
        avg_pnl=('pnl', 'mean'),
        total_pnl=('pnl', 'sum'),
        avg_hold=('bars_held', 'mean'),
    ).reset_index()

    # Add TF sources
    tf_lists = trades_df.groupby('trade_type')['tf'].apply(lambda x: ', '.join(sorted(x.unique()))).reset_index()
    tf_lists.columns = ['trade_type', 'source_tfs']
    type_stats = type_stats.merge(tf_lists, on='trade_type', how='left')

    print(f"  {'Type':<12s} {'Trades':>7s} {'WR':>7s} {'Avg P&L':>10s} {'Total P&L':>12s} "
          f"{'Avg Hold':>9s} {'Source TFs'}")
    print(f"  {'-'*80}")
    for _, row in type_stats.iterrows():
        color = GREEN if row['total_pnl'] > 0 else RED
        print(f"  {row['trade_type']:<12s} {row['count']:>7d} {row['win_rate']:>6.1%} "
              f"${row['avg_pnl']:>9,.2f} {color}${row['total_pnl']:>10,.0f}{RESET} "
              f"{row['avg_hold']:>8.1f}  {row.get('source_tfs', '')}")

    # Monthly breakdown by type
    trades_df['year_month'] = trades_df['exit_time'].dt.to_period('M').astype(str)
    monthly_type = trades_df.groupby(['year_month', 'trade_type'])['pnl'].agg(['sum', 'count']).reset_index()
    monthly_type.columns = ['year_month', 'trade_type', 'pnl', 'count']
    pivot = monthly_type.pivot(index='year_month', columns='trade_type', values='pnl').fillna(0)

    print(f"\n  Monthly P&L by Trade Type (last 24 months):")
    print(f"  {'Month':<10s}", end="")
    for col in ['scalp', 'day_trade', 'swing', 'position']:
        if col in pivot.columns:
            print(f"{col:>12s}", end="")
    print(f"{'Total':>12s}")
    print(f"  {'-'*70}")

    months_to_show = sorted(pivot.index)[-24:]
    for ym in months_to_show:
        row_str = f"  {ym:<10s}"
        row_total = 0
        for col in ['scalp', 'day_trade', 'swing', 'position']:
            if col in pivot.columns:
                val = pivot.loc[ym, col]
                row_total += val
                c = GREEN if val > 0 else RED if val < 0 else RESET
                row_str += f"{c}${val:>10,.0f}{RESET}"
            else:
                row_str += f"{'$0':>12s}"
        c = GREEN if row_total > 0 else RED
        row_str += f"{c}${row_total:>10,.0f}{RESET}"
        print(row_str)
    print(f"{'='*100}\n")


def print_regime_breakdown(trades):
    """Per-regime ROI, DD, win rate analysis."""
    if not trades:
        print("\n  No trades for regime breakdown.")
        return

    trades_df = pd.DataFrame(trades)

    GREEN = '\033[92m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

    print(f"\n{'='*100}")
    print(f"  {BOLD}REGIME BREAKDOWN (Unified Multi-TF){RESET}")
    print(f"{'='*100}")

    for regime_name in ['bull', 'bear', 'sideways', 'crash']:
        regime_trades = trades_df[trades_df['regime_name'] == regime_name]
        if regime_trades.empty:
            print(f"\n  {regime_name.upper()}: No trades")
            continue

        total_pnl = regime_trades['pnl'].sum()
        win_rate = (regime_trades['pnl'] > 0).mean()
        roi = total_pnl / STARTING_BALANCE * 100
        avg_pnl = regime_trades['pnl'].mean()
        avg_hold = regime_trades['bars_held'].mean()

        # Max drawdown within regime
        cum_pnl = regime_trades['pnl'].cumsum() + STARTING_BALANCE
        peak = cum_pnl.cummax()
        dd = (peak - cum_pnl) / peak
        max_dd = dd.max() * 100 if len(dd) > 0 else 0

        # Per-TF within regime
        tf_breakdown = regime_trades.groupby('tf').agg(
            count=('pnl', 'count'),
            pnl=('pnl', 'sum'),
            wr=('pnl', lambda x: (x > 0).mean()),
        ).reset_index()

        color = GREEN if total_pnl > 0 else RED
        print(f"\n  {BOLD}{regime_name.upper()}{RESET}")
        print(f"    Trades: {len(regime_trades):,} | Win Rate: {win_rate:.1%} | "
              f"P&L: {color}${total_pnl:,.0f}{RESET} | ROI: {roi:.1f}% | "
              f"Max DD: {max_dd:.1f}% | Avg Hold: {avg_hold:.1f} bars")
        print(f"    {'TF':<6s} {'Trades':>7s} {'P&L':>10s} {'WR':>7s}")
        for _, row in tf_breakdown.iterrows():
            c = GREEN if row['pnl'] > 0 else RED
            print(f"    {row['tf']:<6s} {row['count']:>7d} {c}${row['pnl']:>8,.0f}{RESET} {row['wr']:>6.1%}")

    print(f"\n{'='*100}\n")


def print_named_periods(trades):
    """Performance during named market events: COVID, LUNA, FTX, ETF rally, etc."""
    if not trades:
        print("\n  No trades for named period analysis.")
        return

    trades_df = pd.DataFrame(trades)
    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])

    GREEN = '\033[92m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

    print(f"\n{'='*110}")
    print(f"  {BOLD}NAMED PERIOD ANALYSIS (Unified Multi-TF){RESET}")
    print(f"{'='*110}")
    print(f"  {'Period':<24s} {'Dates':>23s} {'Trades':>7s} {'P&L':>12s} {'WR':>7s} "
          f"{'Avg P&L':>10s} {'Best TF':<8s} {'Worst TF':<8s}")
    print(f"  {'-'*106}")

    for name, start, end in NAMED_PERIODS:
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)
        mask = (trades_df['exit_time'] >= start_dt) & (trades_df['exit_time'] <= end_dt)
        period_trades = trades_df[mask]

        if period_trades.empty:
            print(f"  {name:<24s} {start + ' - ' + end:>23s} {'0':>7s} {'$0':>12s} "
                  f"{'0%':>7s} {'$0.00':>10s} {'N/A':<8s} {'N/A':<8s}")
            continue

        total_pnl = period_trades['pnl'].sum()
        win_rate = (period_trades['pnl'] > 0).mean()
        avg_pnl = period_trades['pnl'].mean()

        # Best/worst TF
        tf_pnl = period_trades.groupby('tf')['pnl'].sum()
        best_tf = tf_pnl.idxmax() if len(tf_pnl) > 0 else 'N/A'
        worst_tf = tf_pnl.idxmin() if len(tf_pnl) > 0 else 'N/A'

        color = GREEN if total_pnl > 0 else RED
        date_range = f"{start} - {end}"
        print(f"  {name:<24s} {date_range:>23s} {len(period_trades):>7d} "
              f"{color}${total_pnl:>10,.0f}{RESET} {win_rate:>6.1%} "
              f"${avg_pnl:>9,.2f} {best_tf:<8s} {worst_tf:<8s}")

    print(f"{'='*110}\n")


def _confluence_blocks_per_tf(portfolio_stats):
    """Count confluence blocks per TF for reporting."""
    blocked = portfolio_stats.get('confluence_blocked_trades', [])
    per_tf = {}
    for b in blocked:
        tf = b['tf']
        per_tf[tf] = per_tf.get(tf, 0) + 1
    return per_tf


def print_confluence_impact(portfolio_stats):
    """Print Confluence Filter Impact section."""
    blocked = portfolio_stats.get('confluence_blocked_trades', [])
    total_blocks = portfolio_stats.get('confluence_blocks', 0)

    BOLD = '\033[1m'
    RESET = '\033[0m'
    GREEN = '\033[92m'
    RED = '\033[91m'

    print(f"\n{'='*100}")
    print(f"  {BOLD}CONFLUENCE FILTER IMPACT{RESET}")
    print(f"{'='*100}")
    print(f"  Total trades blocked by parent TF opposition: {total_blocks:,}")

    if not blocked:
        print(f"  No trades were blocked by the confluence filter.")
        print(f"{'='*100}\n")
        return

    # Per-TF breakdown
    per_tf = _confluence_blocks_per_tf(portfolio_stats)
    print(f"\n  Blocks per TF:")
    print(f"  {'TF':<6s} {'Blocked':>8s} {'Parent TF':<10s}")
    print(f"  {'-'*30}")
    for tf in ['5m', '15m', '1h', '4h']:
        count = per_tf.get(tf, 0)
        parent = TF_PARENT_MAP.get(tf, 'N/A')
        if count > 0:
            print(f"  {tf:<6s} {count:>8d} {parent:<10s}")

    # Direction breakdown
    long_blocks = sum(1 for b in blocked if b['direction'] == 1)
    short_blocks = sum(1 for b in blocked if b['direction'] == -1)
    print(f"\n  LONG blocks: {long_blocks:,} | SHORT blocks: {short_blocks:,}")

    # Avg confidence of blocked trades
    avg_conf = np.mean([b['confidence'] for b in blocked]) if blocked else 0
    print(f"  Avg confidence of blocked trades: {avg_conf:.3f}")

    print(f"\n  NOTE: Hypothetical P&L of blocked trades not computed (no exit simulation)")
    print(f"        The filter prevents counter-trend entries, expected net positive impact.")
    print(f"{'='*100}\n")


def unified_main():
    """Entry point for --unified mode."""
    parser = argparse.ArgumentParser(description='Backtesting Audit Tool (Unified)')
    parser.add_argument('--tf', nargs='+', default=['5m', '15m', '1h', '4h', '1d', '1w'],
                        help='Timeframes to include')
    args = parser.parse_args()

    all_trades, equity_history, portfolio_stats = run_unified_backtest(tf_list=args.tf)

    if not all_trades:
        print(f"\n{elapsed()} No trades generated. Nothing to report.", flush=True)
        return

    # --- Console output ---
    print_monthly_heatmap(all_trades)
    print_weekly_summary(all_trades)
    print_trade_type_breakdown(all_trades)
    print_regime_breakdown(all_trades)
    print_named_periods(all_trades)
    print_confluence_impact(portfolio_stats)

    # --- Also generate file outputs using existing functions ---
    trades_df = pd.DataFrame(all_trades)
    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])

    monthly_data = monthly_heatmap(trades_df)
    regime_df = regime_breakdown(trades_df)
    type_df = trade_type_breakdown(trades_df)
    periods_df = named_period_analysis(trades_df)
    weekly_df = weekly_performance(trades_df)
    tf_summary_df = per_tf_summary(trades_df, portfolio_stats.get('configs', {}))

    # JSON report
    json_path = os.path.join(DB_DIR, 'unified_audit_report.json')
    report_json = {
        'generated': datetime.now().isoformat(),
        'mode': 'unified_multi_tf',
        'starting_balance': STARTING_BALANCE,
        'final_balance': portfolio_stats.get('final_balance', 0),
        'peak_balance': portfolio_stats.get('peak_balance', 0),
        'roi_pct': portfolio_stats.get('roi_pct', 0),
        'fee_rate': FEE_RATE,
        'timeframes': list(portfolio_stats.get('configs', {}).keys()),
        'allocation': portfolio_stats.get('allocation', {}),
        'overall': {
            'total_trades': len(trades_df),
            'total_pnl': round(float(trades_df['pnl'].sum()), 2),
            'win_rate': round(float((trades_df['pnl'] > 0).mean()), 4),
            'roi_pct': round(float(trades_df['pnl'].sum() / STARTING_BALANCE * 100), 2),
        },
        'per_tf': tf_summary_df.to_dict(orient='records') if not tf_summary_df.empty else [],
        'regime_breakdown': regime_df.to_dict(orient='records') if not regime_df.empty else [],
        'trade_type_breakdown': type_df.to_dict(orient='records') if not type_df.empty else [],
        'named_periods': periods_df.to_dict(orient='records') if not periods_df.empty else [],
        'weekly': weekly_df.to_dict(orient='records') if not weekly_df.empty else [],
        'monthly_pnl': monthly_data['pnl'].to_dict() if not monthly_data['pnl'].empty else {},
        'per_tf_equity': portfolio_stats.get('per_tf_equity', {}),
        'per_tf_suspended': portfolio_stats.get('per_tf_suspended', {}),
        'confluence_filter': {
            'total_blocks': portfolio_stats.get('confluence_blocks', 0),
            'blocked_per_tf': _confluence_blocks_per_tf(portfolio_stats),
        },
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report_json, f, indent=2, default=str)
    print(f"  Saved: {json_path}", flush=True)

    # Text report
    txt_path = os.path.join(DB_DIR, 'unified_audit_report.txt')
    write_text_report(txt_path, tf_summary_df, regime_df, type_df, periods_df, weekly_df, monthly_data, trades_df, confluence_stats=portfolio_stats)
    print(f"  Saved: {txt_path}", flush=True)

    # HTML heatmap
    html_path = os.path.join(DB_DIR, 'unified_audit_heatmap.html')
    generate_heatmap_html(monthly_data['pnl'], html_path)
    print(f"  Saved: {html_path}", flush=True)

    # Final summary
    total_pnl = trades_df['pnl'].sum()
    print(f"\n{'='*70}")
    print(f"  UNIFIED AUDIT COMPLETE")
    print(f"{'='*70}")
    print(f"  Total Trades: {len(trades_df):,}")
    print(f"  Total P&L:    ${total_pnl:,.2f}")
    print(f"  Final Balance:${portfolio_stats.get('final_balance', 0):,.2f}")
    print(f"  ROI:          {total_pnl/STARTING_BALANCE*100:.1f}%")
    print(f"  Win Rate:     {(trades_df['pnl'] > 0).mean():.1%}")
    print(f"  Max DD:       {portfolio_stats.get('portfolio_dd_pct', 0):.1f}%")
    print(f"  TFs:          {list(portfolio_stats.get('configs', {}).keys())}")
    print(f"  Elapsed:      {time.time() - START_TIME:.1f}s")
    print(f"{'='*70}", flush=True)


if __name__ == '__main__':
    # Check for --unified flag
    if '--unified' in sys.argv:
        sys.argv.remove('--unified')
        unified_main()
    else:
        main()
