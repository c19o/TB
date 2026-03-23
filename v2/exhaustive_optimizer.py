#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
exhaustive_optimizer.py — GPU-Accelerated Exhaustive Grid Search Optimizer
==========================================================================
Replaces the 15K-combo NSGA-II GA with a full grid search testing ~384M
combinations per timeframe across 6 timeframes (5m, 15m, 1H, 4H, 1D, 1W).

Uses RTX 3090 GPU via CuPy (falls back to NumPy) for vectorized simulation.

Usage:
    python exhaustive_optimizer.py
"""

import sys, os, io, time, json, warnings, itertools, math
from concurrent.futures import ProcessPoolExecutor, as_completed
if os.name == 'nt':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime

# ---------------------------------------------------------------------------
# GPU backend: try CuPy, fall back to NumPy
# ---------------------------------------------------------------------------
try:
    import cupy as cp
    xp = cp
    GPU_ARRAY = True
    print(f"[GPU] CuPy + CUDA detected — RTX 3090 24GB — FULL GRID SEARCH")
except ImportError:
    xp = np
    GPU_ARRAY = False
    print("[CPU] CuPy not available — using NumPy (slower)")

import xgboost as xgb
from hardware_detect import detect_hardware

USE_GPU_XGB = False
try:
    _test = xgb.DMatrix(np.random.rand(10, 5), label=np.random.randint(0, 2, 10))
    xgb.train({'tree_method': 'gpu_hist', 'device': 'cuda', 'max_depth': 3},
              _test, num_boost_round=2)
    USE_GPU_XGB = True
    del _test
except Exception:
    pass
_HW = detect_hardware()
_N_GPUS = _HW['n_gpus'] or 1
print(f"[XGB] GPU prediction: {'ENABLED' if USE_GPU_XGB else 'CPU only'}")
print(f"[HW] {_N_GPUS} GPU(s) detected for parallel TF optimization")

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
DB_DIR = os.environ.get('SAVAGE22_DB_DIR', os.path.dirname(os.path.abspath(__file__)))
START_TIME = time.time()
TOTAL_COST_PER_TRADE = 0.0018  # 0.18% round-trip (Bitget taker + slippage)
STARTING_BALANCE = 10000.0

# Batch size: number of parameter combos to simulate at once on GPU
# RTX 3090 has 24 GB VRAM; each combo needs ~8 floats * n_bars * 4 bytes
# Conservative default; auto-tuned per TF based on bar count
GPU_BATCH_BASE = 500_000  # base batch for ~1000 bars

def elapsed():
    return f"[{time.time() - START_TIME:.0f}s]"

# ---------------------------------------------------------------------------
# Per-TF parameter grids
# ---------------------------------------------------------------------------
# REDUCED GRID — ~5M combos per TF × 6 TFs = ~30M total
# RTX 3090 24GB GPU — estimated ~1-2 hours overnight
# Still covers full parameter space, just coarser steps

TF_GRIDS = {
    '5m': {
        'lev': list(range(1, 126, 8)),                                        # 16 steps
        'risk': list(np.round(np.arange(0.01, 2.01, 0.20), 4)),              # 10 steps
        'stop_atr': list(np.round(np.arange(0.05, 1.01, 0.10), 4)),          # 10 steps
        'rr': list(np.round(np.arange(1.0, 4.1, 0.30), 4)),                  # 11 steps
        'hold': [1,2,4,8,12,20,30,48,60,72],                                  # 10 steps
        'exit_type': [0, 25, 50, 75, -2, -3],                                 # 6 steps
        'conf': list(np.round(np.arange(0.45, 0.91, 0.10), 4)),              # 5 steps
    },
    '15m': {
        'lev': list(range(1, 126, 8)),                                        # 16
        'risk': list(np.round(np.arange(0.01, 3.01, 0.30), 4)),              # 10
        'stop_atr': list(np.round(np.arange(0.1, 1.51, 0.14), 4)),           # 11
        'rr': list(np.round(np.arange(1.0, 5.1, 0.40), 4)),                  # 11
        'hold': [1,2,4,8,12,20,30,42,48,60],                                  # 10
        'exit_type': [0, 25, 50, 75, -2, -3],                                 # 6
        'conf': list(np.round(np.arange(0.45, 0.91, 0.10), 4)),              # 5
    },
    '1h': {
        'lev': list(range(1, 101, 5)),                                        # 20
        'risk': list(np.round(np.arange(0.05, 4.01, 0.40), 4)),              # 10
        'stop_atr': list(np.round(np.arange(0.2, 2.01, 0.18), 4)),           # 10
        'rr': list(np.round(np.arange(1.0, 6.1, 0.50), 4)),                  # 11
        'hold': [1,2,4,8,12,20,30,48,60,72],                                  # 10
        'exit_type': [0, 25, 50, 75, -2, -3],                                 # 6
        'conf': list(np.round(np.arange(0.45, 0.91, 0.10), 4)),              # 5
    },
    '4h': {
        'lev': list(range(1, 76, 5)),                                         # 15
        'risk': list(np.round(np.arange(0.1, 5.01, 0.50), 4)),               # 10
        'stop_atr': list(np.round(np.arange(0.3, 3.01, 0.27), 4)),           # 10
        'rr': list(np.round(np.arange(1.0, 8.1, 0.70), 4)),                  # 11
        'hold': [1,2,4,8,12,20,30,48,66,84],                                  # 10
        'exit_type': [0, 25, 50, 75, -2, -3],                                 # 6
        'conf': list(np.round(np.arange(0.45, 0.91, 0.10), 4)),              # 5
    },
    '1d': {
        'lev': list(range(1, 21, 2)),                                         # 10
        'risk': list(np.round(np.arange(0.1, 6.01, 0.60), 4)),               # 10
        'stop_atr': list(np.round(np.arange(0.5, 4.01, 0.35), 4)),           # 10
        'rr': list(np.round(np.arange(1.0, 8.1, 0.70), 4)),                  # 11
        'hold': [1,3,7,14,30,45,60,75,90,120],                                 # 10
        'exit_type': [0, 25, 50, 75, -2, -3],                                 # 6
        'conf': list(np.round(np.arange(0.45, 0.91, 0.10), 4)),              # 5
    },
    '1w': {
        'lev': list(range(1, 11, 1)),                                         # 10
        'risk': list(np.round(np.arange(0.1, 6.01, 0.60), 4)),               # 10
        'stop_atr': list(np.round(np.arange(1.0, 6.01, 0.50), 4)),           # 11
        'rr': list(np.round(np.arange(1.0, 10.1, 0.90), 4)),                 # 11
        'hold': [1,3,5,8,13,20,30,40,48,52],                                  # 10
        'exit_type': [0, 25, 50, 75, -2, -3],                                 # 6
        'conf': list(np.round(np.arange(0.45, 0.91, 0.10), 4)),              # 5
    },
}

# DB mapping  (matches ml_multi_tf.py TF_CONFIGS)
TF_DB_MAP = {
    '5m':  {'db': 'features_5m.db',       'table': 'features_5m',  'return_col': 'next_5m_return',  'cost_pct': 0.22,
            'rolling_window_bars': 105120},
    '15m': {'db': 'features_15m.db',      'table': 'features_15m', 'return_col': 'next_15m_return', 'cost_pct': 0.22,
            'rolling_window_bars': 35040},
    '1h':  {'db': 'features_1h.db',       'table': 'features_1h',  'return_col': 'next_1h_return',  'cost_pct': 0.23,
            'rolling_window_bars': 13140},
    '4h':  {'db': 'features_4h.db',       'table': 'features_4h',  'return_col': 'next_4h_return',  'cost_pct': 0.24,
            'rolling_window_bars': 8760},
    '1d':  {'db': 'features_1d.db',       'table': 'features_1d',  'return_col': 'next_1d_return',  'cost_pct': 0.0025,
            'rolling_window_bars': None},
    '1w':  {'db': 'features_1w.db',       'table': 'features_1w',  'return_col': 'next_1w_return',  'cost_pct': 0.0025,
            'rolling_window_bars': None},
}


def count_grid(grid: dict) -> int:
    """Count total combinations in a parameter grid."""
    total = 1
    for v in grid.values():
        total *= len(v)
    return total


def build_param_arrays(grid: dict):
    """
    Build arrays of every unique value per parameter.
    Returns dict of {param_name: np.array}.
    """
    return {k: np.array(v, dtype=np.float32) for k, v in grid.items()}


# ---------------------------------------------------------------------------
# Data loader: re-create walk-forward test window data + model predictions
# ---------------------------------------------------------------------------
def load_tf_data(tf_name: str):
    """
    Load feature DB, trained model, and feature list for a TF.
    Uses all features (not pruned). Tries features_{tf}_all.json first,
    falls back to _pruned.json, then derives from DB columns.
    Re-runs walk-forward split logic to get the last test window,
    then predicts confidences.

    Returns: (confidences, directions, returns, closes, atrs, highs, lows, n_bars) or None
    """
    cfg = TF_DB_MAP[tf_name]
    db_path = f"{DB_DIR}/{cfg['db']}"
    model_path = f"{DB_DIR}/model_{tf_name}.json"
    features_all_path = f"{DB_DIR}/features_{tf_name}_all.json"
    features_pruned_path = f"{DB_DIR}/features_{tf_name}_pruned.json"

    # V2 naming: features_BTC_{tf}.parquet
    v2_parquet = os.path.join(DB_DIR, f'features_BTC_{tf_name}.parquet')
    if not os.path.exists(db_path) and not os.path.exists(db_path.replace('.db', '.parquet')) and not os.path.exists(v2_parquet):
        print(f"  SKIP {tf_name} — {cfg['db']} not found")
        return None
    if not os.path.exists(model_path):
        print(f"  SKIP {tf_name} — model_{tf_name}.json not found")
        return None

    # Load feature list: prefer _all.json, fall back to _pruned.json, else derive from DB
    saved_features = None
    if os.path.exists(features_all_path):
        with open(features_all_path, 'r') as f:
            saved_features = json.load(f)
        print(f"  Loaded {len(saved_features)} features from features_{tf_name}_all.json")
    elif os.path.exists(features_pruned_path):
        with open(features_pruned_path, 'r') as f:
            saved_features = json.load(f)
        print(f"  Loaded {len(saved_features)} features from features_{tf_name}_pruned.json (legacy fallback)")

    # Load data — try parquet first (no column limit), fall back to SQLite
    parquet_path = db_path.replace('.db', '.parquet')
    if not os.path.exists(parquet_path) and os.path.exists(v2_parquet):
        parquet_path = v2_parquet
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        print(f"  Loaded from parquet: {parquet_path}")
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

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    elif 'date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['date'])

    # Return column
    return_col = cfg['return_col']
    if return_col not in df.columns:
        candidates = [c for c in df.columns if 'next' in c.lower() and 'return' in c.lower()]
        if candidates:
            return_col = candidates[0]
        else:
            print(f"  SKIP {tf_name} — no return column found")
            return None

    returns = pd.to_numeric(df[return_col], errors='coerce').values
    closes = pd.to_numeric(df['close'], errors='coerce').values
    highs = pd.to_numeric(df['high'], errors='coerce').values
    lows = pd.to_numeric(df['low'], errors='coerce').values

    # 3-class labels for filtering
    cost = cfg['cost_pct']
    y_3class = np.where(returns > cost, 1, np.where(returns < -cost, 0, -1))

    # Meta/target columns to exclude
    meta_cols = {'timestamp', 'date', 'open', 'high', 'low', 'close', 'volume',
                 'quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote',
                 'open_time', 'date_norm'}
    target_like = {c for c in df.columns if 'next_' in c.lower() or 'target' in c.lower() or 'direction' in c.lower()}
    exclude_cols = meta_cols | target_like
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Use saved feature list if available (ensures column order matches trained model),
    # otherwise use all feature columns derived from DB
    if saved_features is not None:
        # Verify all saved features exist in DB; use saved order for model compatibility
        missing = [f for f in saved_features if f not in feature_cols]
        if missing:
            print(f"  WARNING: {len(missing)} saved features missing from DB, filling with NaN")
        model_features = saved_features
    else:
        model_features = feature_cols
        print(f"  No feature file found — using all {len(model_features)} feature columns from DB")

    # Build feature matrix in model's expected column order
    # XGBoost handles NaN natively — do NOT replace with 0
    X_model = np.empty((len(df), len(model_features)), dtype=np.float32)
    for i, feat in enumerate(model_features):
        if feat in feature_cols:
            X_model[:, i] = pd.to_numeric(df[feat], errors='coerce').values.astype(np.float32)
        else:
            X_model[:, i] = np.nan  # missing feature — XGBoost treats NaN as missing

    # Recreate walk-forward splits (same logic as ml_multi_tf.py)
    n = len(df)
    rolling_window = cfg.get('rolling_window_bars')

    if rolling_window is None or rolling_window >= n:
        test_size = n // 4
        splits = [
            (0, n - 2 * test_size, n - 2 * test_size, n - test_size),
            (0, n - test_size, n - test_size, n),
        ]
    else:
        test_size = rolling_window // 4
        splits = []
        for wi in range(3):
            test_end = n - (2 - wi) * test_size
            test_start = test_end - test_size
            train_end = test_start
            train_start = max(0, train_end - rolling_window)
            if train_start < 0 or test_start < 0 or train_end - train_start < 500:
                continue
            splits.append((train_start, train_end, test_start, min(test_end, n)))

    if not splits:
        print(f"  SKIP {tf_name} — no valid walk-forward splits")
        return None

    # Use the LAST split's test window (same as GA in ml_multi_tf.py)
    ts, te, vs, ve = splits[-1]
    print(f"  Walk-forward last window: test [{vs}:{ve}] ({ve - vs} bars)")

    # Load trained model
    model = xgb.Booster()
    model.load_model(model_path)
    if USE_GPU_XGB:
        model.set_param({'device': 'cuda'})

    # Predict on full test window (3-class softprob: SHORT=0, FLAT=1, LONG=2)
    dtest_all = xgb.DMatrix(X_model[vs:ve], feature_names=model_features, nthread=-1)
    raw_preds = model.predict(dtest_all)

    # For 3-class: raw_preds shape is (N, 3) -> use max class prob as confidence
    if raw_preds.ndim == 2 and raw_preds.shape[1] == 3:
        pred_class = np.argmax(raw_preds, axis=1)  # 0=SHORT, 1=FLAT, 2=LONG
        confidences = np.max(raw_preds, axis=1)
        # Convert to direction: LONG=+1, SHORT=-1, FLAT=0
        directions = np.where(pred_class == 2, 1.0, np.where(pred_class == 0, -1.0, 0.0))
        print(f"  3-class preds: LONG={np.sum(pred_class==2)} FLAT={np.sum(pred_class==1)} SHORT={np.sum(pred_class==0)}")
        print(f"  Confidence range: [{confidences.min():.3f}, {confidences.max():.3f}]")
    else:
        # Legacy binary mode fallback
        confidences = raw_preds
        directions = np.where(raw_preds > 0.5, 1.0, -1.0)
        print(f"  Binary preds. Range: [{confidences.min():.3f}, {confidences.max():.3f}]")

    test_returns = returns[vs:ve]
    test_closes = closes[vs:ve]

    # ATR
    if 'atr_14' in df.columns:
        test_atrs = pd.to_numeric(df['atr_14'], errors='coerce').values[vs:ve]
    else:
        test_atrs = np.abs(test_returns) / 100 * test_closes
    test_atrs = np.nan_to_num(test_atrs, nan=max(test_closes.mean() * 0.01, 1.0))

    n_bars = ve - vs
    return confidences.astype(np.float32), directions.astype(np.float32), \
           test_returns.astype(np.float32), test_closes.astype(np.float32), \
           test_atrs.astype(np.float32), highs[vs:ve].astype(np.float32), \
           lows[vs:ve].astype(np.float32), n_bars


# ---------------------------------------------------------------------------
# Vectorized simulation engine
# ---------------------------------------------------------------------------
def simulate_batch(params_batch, confs, dirs, closes, atrs, highs, lows, regime, xp_lib):
    """
    Vectorized simulation of N parameter combos across T bars.

    params_batch: (N, 7) array — [lev, risk_pct, stop_atr, rr, max_hold, exit_type, conf_thresh]
    confs:  (T,) array of model confidences
    dirs:   (T,) array of model directions (+1=LONG, -1=SHORT, 0=FLAT)
    closes: (T,) array of close prices
    atrs:   (T,) array of ATR values
    highs:  (T,) array of high prices (for intrabar SL/TP checks)
    lows:   (T,) array of low prices (for intrabar SL/TP checks)
    regime: (T,) array of regime labels (0=bull, 1=bear, 2=sideways, 3=volatile)

    Returns: (N, 7) array — [final_balance, max_dd, win_rate, trade_count, roi_pct, sortino, total_trades]

    NOTE: Because trailing stops and sequential trade logic require bar-by-bar state,
    we iterate over bars but vectorize across all N combos simultaneously.
    This is the standard approach for GPU-accelerated backtesting.
    """
    N = params_batch.shape[0]
    T = len(confs)

    # Unpack params — each is (N,)
    lev       = params_batch[:, 0]
    risk_pct  = params_batch[:, 1] / 100.0   # convert to decimal
    stop_mult = params_batch[:, 2]
    rr        = params_batch[:, 3]
    max_hold  = params_batch[:, 4]
    exit_type = params_batch[:, 5]
    conf_th   = params_batch[:, 6]

    # State arrays (N,)
    balance    = xp_lib.full(N, STARTING_BALANCE, dtype=xp_lib.float32)
    peak       = xp_lib.full(N, STARTING_BALANCE, dtype=xp_lib.float32)
    max_dd     = xp_lib.zeros(N, dtype=xp_lib.float32)
    wins       = xp_lib.zeros(N, dtype=xp_lib.int32)
    losses     = xp_lib.zeros(N, dtype=xp_lib.int32)
    in_trade   = xp_lib.zeros(N, dtype=xp_lib.bool_)
    trade_bars = xp_lib.zeros(N, dtype=xp_lib.int32)
    trade_dir  = xp_lib.zeros(N, dtype=xp_lib.int32)   # 1=LONG, -1=SHORT
    entry_pr   = xp_lib.zeros(N, dtype=xp_lib.float32)
    stop_pr    = xp_lib.zeros(N, dtype=xp_lib.float32)
    tp_pr      = xp_lib.zeros(N, dtype=xp_lib.float32)
    best_pr    = xp_lib.zeros(N, dtype=xp_lib.float32)  # for trailing stop
    alive      = xp_lib.ones(N, dtype=xp_lib.bool_)     # balance > 0

    # For Sortino: track sum of log returns and sum of squared negative log returns
    sum_log_ret     = xp_lib.zeros(N, dtype=xp_lib.float64)
    sum_neg_sq      = xp_lib.zeros(N, dtype=xp_lib.float64)
    count_neg       = xp_lib.zeros(N, dtype=xp_lib.int32)
    total_trades    = xp_lib.zeros(N, dtype=xp_lib.int32)

    # Pre-compute: is this a trailing stop config?
    is_trail   = (exit_type < 0)                          # -2 or -3
    trail_mult = xp_lib.abs(exit_type) * is_trail.astype(xp_lib.float32)  # 2 or 3 (or 0)
    is_partial = (~is_trail)                               # 0, 25, 50, 75 partial TP

    fee_rate = TOTAL_COST_PER_TRADE

    # Regime multipliers (numpy arrays, indexed by scalar regime per bar)
    # Matches live_trader.py regime detection: [bull, bear, sideways, volatile]
    REGIME_LEV_MULT_np  = np.array([1.0, 0.47, 0.67, 0.07], dtype=np.float32)
    REGIME_RISK_MULT_np = np.array([1.0, 1.0, 0.47, 1.0], dtype=np.float32)
    REGIME_STOP_MULT_np = np.array([1.0, 0.75, 0.5, 1.5], dtype=np.float32)
    REGIME_RR_MULT_np   = np.array([1.0, 0.75, 0.5, 0.25], dtype=np.float32)
    REGIME_HOLD_MULT_np = np.array([1.0, 0.17, 1.0, 2.0], dtype=np.float32)

    for t in range(T):
        if not alive.any():
            break

        c_val = float(confs[t])
        dirs_t = float(dirs[t])  # +1=LONG, -1=SHORT, 0=FLAT
        p_val = float(closes[t]) if closes[t] > 0 else 1.0
        h_val = float(highs[t]) if highs[t] > 0 else p_val
        l_val = float(lows[t]) if lows[t] > 0 else p_val
        a_val = float(atrs[t]) if atrs[t] > 0 else p_val * 0.01

        # Regime multipliers for this bar (scalar, broadcast to N combos)
        r = int(regime[t])
        lev_m = float(REGIME_LEV_MULT_np[r])
        risk_m = float(REGIME_RISK_MULT_np[r])
        stop_m = float(REGIME_STOP_MULT_np[r])
        rr_m = float(REGIME_RR_MULT_np[r])
        hold_m = float(REGIME_HOLD_MULT_np[r])

        # --- Exit logic for those in trade ---
        active = in_trade & alive
        if active.any():
            trade_bars += active.astype(xp_lib.int32)

            # Update best price for trailing (use h_val for longs, l_val for shorts)
            long_active = active & (trade_dir == 1)
            short_active = active & (trade_dir == -1)
            if long_active.any():
                best_pr = xp_lib.where(long_active, xp_lib.maximum(best_pr, h_val), best_pr)
            if short_active.any():
                best_pr = xp_lib.where(short_active, xp_lib.minimum(best_pr, l_val), best_pr)

            # Trailing stop update: after reaching 1R profit, trail at trail_mult * ATR
            trail_active = active & is_trail
            if trail_active.any():
                # 1R profit level
                one_r_long  = entry_pr + stop_mult * a_val
                one_r_short = entry_pr - stop_mult * a_val
                past_1r_long  = trail_active & (trade_dir == 1) & (p_val >= one_r_long)
                past_1r_short = trail_active & (trade_dir == -1) & (p_val <= one_r_short)

                # Trail stop for longs: best_price - trail_mult * ATR
                new_trail_long = best_pr - trail_mult * a_val
                stop_pr = xp_lib.where(past_1r_long,
                                       xp_lib.maximum(stop_pr, new_trail_long), stop_pr)

                # Trail stop for shorts: best_price + trail_mult * ATR
                new_trail_short = best_pr + trail_mult * a_val
                stop_pr = xp_lib.where(past_1r_short,
                                       xp_lib.minimum(stop_pr, new_trail_short), stop_pr)

            # SL check (use lows for longs, highs for shorts — intrabar barrier)
            sl_long  = active & (trade_dir == 1)  & (l_val <= stop_pr)
            sl_short = active & (trade_dir == -1) & (h_val >= stop_pr)
            sl_hit = sl_long | sl_short

            # TP check (use highs for longs, lows for shorts — intrabar barrier)
            tp_long  = active & (trade_dir == 1)  & (h_val >= tp_pr)
            tp_short = active & (trade_dir == -1) & (l_val <= tp_pr)
            tp_hit = tp_long | tp_short

            # Max hold check (regime-adjusted)
            hold_exit = active & (trade_bars >= max_hold * hold_m)

            # Any exit
            exiting = sl_hit | tp_hit | hold_exit

            if exiting.any():
                # PnL calc — use barrier price for SL/TP exits, close for time exits
                # Priority: SL > TP > hold (conservative — if both barriers hit intrabar, assume SL)
                exit_price = xp_lib.where(
                    sl_hit, stop_pr,
                    xp_lib.where(tp_hit, tp_pr,
                                 xp_lib.full(N, p_val, dtype=xp_lib.float32)))
                price_chg = (exit_price - entry_pr) / xp_lib.maximum(entry_pr, 1e-8) * trade_dir.astype(xp_lib.float32)
                eff_lev = lev * lev_m
                gross_pnl = price_chg * eff_lev
                fee_cost  = fee_rate * eff_lev
                net_pnl   = gross_pnl - fee_cost

                # Partial TP: for exit_type 25/50/75, reduce the take-profit PnL
                # When TP is hit with partial, we get partial_pct of TP PnL and
                # the rest runs to SL or max_hold. Simplified: scale TP exits.
                # For simplicity in exhaustive search, partial TP means:
                # if TP hit, realize (exit_type/100) of the profit now, rest at entry (break-even)
                # Net effect: tp_pnl * (partial_pct + (1-partial_pct)*0) = tp_pnl * partial_pct
                # But exit_type=0 means NO partial TP (full TP hit = full profit)
                partial_pct_val = xp_lib.where(
                    is_partial & (exit_type > 0),
                    exit_type / 100.0,
                    xp_lib.ones(N, dtype=xp_lib.float32)
                )
                # For non-TP exits (SL or hold), full PnL applies
                # For TP exits with partial, scale down profit
                scale = xp_lib.where(tp_hit & is_partial & (exit_type > 0),
                                     partial_pct_val, xp_lib.ones(N, dtype=xp_lib.float32))

                eff_risk = risk_pct * risk_m
                pnl_dollar = balance * eff_risk * net_pnl * scale

                # Compute per-trade log return for Sortino
                new_bal = xp_lib.maximum(balance + pnl_dollar, 1.0)
                log_ret = xp_lib.log(new_bal / xp_lib.maximum(balance, 1.0))

                # Update Sortino accumulators
                sum_log_ret += xp_lib.where(exiting, log_ret.astype(xp_lib.float64), 0.0)
                neg_mask = exiting & (log_ret < 0)
                sum_neg_sq += xp_lib.where(neg_mask, (log_ret ** 2).astype(xp_lib.float64), 0.0)
                count_neg += neg_mask.astype(xp_lib.int32)
                total_trades += exiting.astype(xp_lib.int32)

                # Update balance
                balance = xp_lib.where(exiting, balance + pnl_dollar, balance)
                wins    += (exiting & (pnl_dollar > 0)).astype(xp_lib.int32)
                losses  += (exiting & (pnl_dollar <= 0)).astype(xp_lib.int32)

                # Reset trade state
                in_trade   = xp_lib.where(exiting, False, in_trade)
                trade_bars = xp_lib.where(exiting, 0, trade_bars)

                # Check alive
                alive = alive & (balance > 0)

        # --- Entry logic for those NOT in trade ---
        can_enter = (~in_trade) & alive
        if can_enter.any():
            # Use model direction (dirs_t): +1=LONG, -1=SHORT, 0=FLAT
            # Only enter when confidence exceeds threshold AND model has a direction
            go_long  = can_enter & (dirs_t == 1.0) & (c_val > conf_th)
            go_short = can_enter & (dirs_t == -1.0) & (c_val > conf_th)

            entering = go_long | go_short

            if entering.any():
                new_dir = xp_lib.where(go_long, 1, xp_lib.where(go_short, -1, 0))

                # Entry price
                entry_pr = xp_lib.where(entering, p_val, entry_pr)

                # Stop loss (regime-adjusted)
                sl_dist = stop_mult * stop_m * a_val
                stop_pr = xp_lib.where(
                    entering & (new_dir == 1),  p_val - sl_dist,
                    xp_lib.where(entering & (new_dir == -1), p_val + sl_dist, stop_pr)
                )

                # Take profit (regime-adjusted)
                tp_dist = stop_mult * stop_m * a_val * rr * rr_m
                tp_pr = xp_lib.where(
                    entering & (new_dir == 1),  p_val + tp_dist,
                    xp_lib.where(entering & (new_dir == -1), p_val - tp_dist, tp_pr)
                )

                # Best price init (for trailing)
                best_pr = xp_lib.where(entering, p_val, best_pr)

                trade_dir  = xp_lib.where(entering, new_dir, trade_dir)
                in_trade   = xp_lib.where(entering, True, in_trade)
                trade_bars = xp_lib.where(entering, 0, trade_bars)

        # Drawdown tracking (every bar)
        peak   = xp_lib.maximum(peak, balance)
        dd     = xp_lib.where(peak > 0, (peak - balance) / peak, 0.0)
        max_dd = xp_lib.maximum(max_dd, dd)

    # --- Compute final metrics ---
    total_tr = (wins + losses).astype(xp_lib.float32)
    win_rate = xp_lib.where(total_tr > 0, wins.astype(xp_lib.float32) / total_tr, 0.0)
    roi_pct  = (balance - STARTING_BALANCE) / STARTING_BALANCE * 100.0
    max_dd_pct = max_dd * 100.0

    # Sortino ratio
    mean_log = xp_lib.where(total_trades > 0,
                            sum_log_ret / total_trades.astype(xp_lib.float64), 0.0)
    downside_var = xp_lib.where(total_trades > 0,
                                sum_neg_sq / total_trades.astype(xp_lib.float64), 0.0)
    downside_std = xp_lib.sqrt(xp_lib.maximum(downside_var, 1e-12))
    sortino = xp_lib.where(downside_std > 1e-6,
                           mean_log / downside_std,
                           mean_log * 10.0).astype(xp_lib.float32)

    # Stack results: (N, 7)
    results = xp_lib.column_stack([
        balance,
        max_dd_pct,
        win_rate,
        total_tr,
        roi_pct,
        sortino,
        total_trades.astype(xp_lib.float32),
    ])
    return results


# ---------------------------------------------------------------------------
# Grid search driver for one TF
# ---------------------------------------------------------------------------
def run_grid_search(tf_name, confs, dirs, closes, atrs, highs, lows, n_bars, resume=False):
    """
    Run exhaustive grid search for one timeframe.
    Returns dict of best configs per profile.
    """
    grid = TF_GRIDS[tf_name]
    total_combos = count_grid(grid)
    print(f"\n{'='*70}")
    print(f"  {tf_name.upper()} EXHAUSTIVE GRID SEARCH")
    print(f"  Total combinations: {total_combos:,}")
    print(f"  Test window: {n_bars} bars")
    print(f"{'='*70}")

    # Checkpoint/resume support
    checkpoint_path = os.path.join(DB_DIR, f'optimizer_checkpoint_{tf_name}.pkl')
    start_batch = 0
    if resume and os.path.exists(checkpoint_path):
        import pickle
        with open(checkpoint_path, 'rb') as f:
            ckpt = pickle.load(f)
        best = ckpt['best']
        start_batch = ckpt['next_batch']
        print(f"  RESUME: loaded checkpoint at batch {start_batch}, {ckpt['total_processed']:,} combos done")

    # Determine batch size based on bar count and available memory
    # Each combo needs ~N_bars * ~10 floats of state per bar-step, but we
    # vectorize across combos (N dimension), iterating bars (T dimension).
    # Memory per combo ~ 20 float32 state vars = 80 bytes
    # RTX 3090 = 24 GB; leave 4 GB headroom = 20 GB usable
    # 20 GB / 80 bytes = 250M combos theoretically, but intermediate arrays
    # multiply this. Conservative: 80 bytes * 10 intermediates = 800 bytes/combo
    # 20 GB / 800 = 25M max batch. Use min of that and total.
    if GPU_ARRAY:
        try:
            import subprocess
            _nv = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                                 capture_output=True, text=True, timeout=10)
            free_vram_mb = max(int(x.strip()) for x in _nv.stdout.strip().split('\n') if x.strip())
        except (Exception, FileNotFoundError, OSError):
            free_vram_mb = 20000
        bytes_per_combo = 800
        usable_vram = int(free_vram_mb * 0.85 * 1024 * 1024)
        max_batch = max(500_000, usable_vram // bytes_per_combo)
        print(f"  VRAM: {free_vram_mb}MB free -> max batch {max_batch:,}")
    else:
        max_batch = min(2_000_000, total_combos)

    batch_size = max_batch
    print(f"  Batch size: {batch_size:,}")

    # Build parameter value arrays
    param_names = ['lev', 'risk', 'stop_atr', 'rr', 'hold', 'exit_type', 'conf']
    param_sizes = [len(grid[k]) for k in param_names]
    print(f"  Grid dimensions: {' x '.join(f'{k}={s}' for k, s in zip(param_names, param_sizes))}")

    # Compute regime per bar (once per TF, before simulation)
    sma100 = pd.Series(closes).rolling(100, min_periods=1).mean().values
    slope = np.gradient(sma100) / np.maximum(sma100, 1e-8)

    # Regime: 0=bull, 1=bear, 2=sideways, 3=crash (encode as int array)
    # Matches backtesting_audit.py and live_trader.py regime detection
    log_returns = np.diff(np.log(np.maximum(closes, 1e-8)), prepend=0.0)
    rvol_20 = pd.Series(np.abs(log_returns)).rolling(20, min_periods=1).std().values
    rvol_90_avg = pd.Series(rvol_20).rolling(90, min_periods=1).mean().values
    rolling_high_30 = pd.Series(closes).rolling(30, min_periods=1).max().values
    dd_from_30h = (rolling_high_30 - closes) / np.maximum(rolling_high_30, 1e-8)

    regime = np.full(len(closes), 2, dtype=np.int32)  # default sideways
    above_sma = closes > sma100
    for i in range(len(closes)):
        if above_sma[i] and slope[i] > 0.001:
            regime[i] = 0  # bull
        elif not above_sma[i] and slope[i] < -0.001:
            regime[i] = 1  # bear
        # Crash: high vol + below SMA100 + 15% drawdown from 30-bar high
        if (rvol_20[i] > 2.0 * max(rvol_90_avg[i], 1e-12)
                and not above_sma[i] and dd_from_30h[i] > 0.15):
            regime[i] = 3  # crash (overrides bear/sideways)

    regime_counts = {0: np.sum(regime==0), 1: np.sum(regime==1), 2: np.sum(regime==2), 3: np.sum(regime==3)}
    print(f"  Regime distribution: bull={regime_counts[0]} bear={regime_counts[1]} "
          f"sideways={regime_counts[2]} crash={regime_counts[3]}")

    # Transfer market data to GPU
    if GPU_ARRAY:
        g_confs  = cp.asarray(confs)
        g_dirs   = cp.asarray(dirs)
        g_closes = cp.asarray(closes)
        g_atrs   = cp.asarray(atrs)
        g_highs  = cp.asarray(highs)
        g_lows   = cp.asarray(lows)
        g_regime = cp.asarray(regime)
    else:
        g_confs  = confs
        g_dirs   = dirs
        g_closes = closes
        g_atrs   = atrs
        g_highs  = highs
        g_lows   = lows
        g_regime = regime

    # Best trackers (only init if not resuming from checkpoint)
    # We track: final_balance, max_dd_pct, win_rate, trade_count, roi_pct, sortino, total_trades
    # Indices:       0            1           2          3          4        5          6
    if start_batch == 0:
        best = {
            'dd10_best':    {'score': -np.inf, 'params': None, 'metrics': None},
            'dd10_sortino': {'score': -np.inf, 'params': None, 'metrics': None},
            'dd15_best':    {'score': -np.inf, 'params': None, 'metrics': None},
            'dd15_sortino': {'score': -np.inf, 'params': None, 'metrics': None},
        }

    # Generate combos in batches using itertools.product indices
    # Instead of materializing all combos, we iterate in chunks
    param_values = [np.array(grid[k], dtype=np.float32) for k in param_names]

    total_processed = start_batch * batch_size if start_batch > 0 else 0
    t_start = time.time()
    last_print = t_start

    # Use multi-index iteration: compute parameter combos from flat index
    # This avoids materializing all combos at once
    strides = []
    s = 1
    for ps in reversed(param_sizes):
        strides.insert(0, s)
        s *= ps

    n_batches = math.ceil(total_combos / batch_size)
    save_interval = max(1, n_batches // 20)  # checkpoint every ~5% of batches

    for batch_idx in range(start_batch, n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_combos)
        actual_batch = end_idx - start_idx

        # Build parameter array for this batch from flat indices
        flat_indices = np.arange(start_idx, end_idx, dtype=np.int64)
        params_np = np.empty((actual_batch, 7), dtype=np.float32)

        remaining = flat_indices.copy()
        for p, (stride, pv) in enumerate(zip(strides, param_values)):
            dim_idx = remaining // stride
            remaining = remaining % stride
            params_np[:, p] = pv[dim_idx.astype(np.int32)]

        # Transfer to GPU if needed
        if GPU_ARRAY:
            params_gpu = cp.asarray(params_np)
            results = simulate_batch(params_gpu, g_confs, g_dirs, g_closes, g_atrs, g_highs, g_lows, g_regime, cp)
            results_np = cp.asnumpy(results)
            del params_gpu, results
            cp.get_default_memory_pool().free_all_blocks()
        else:
            results_np = simulate_batch(params_np, g_confs, g_dirs, g_closes, g_atrs, g_highs, g_lows, g_regime, np)

        # results_np: (batch, 7) = [balance, max_dd_pct, win_rate, trade_count, roi_pct, sortino, total_trades]
        bal     = results_np[:, 0]
        dd_pct  = results_np[:, 1]
        wr      = results_np[:, 2]
        trades  = results_np[:, 3]
        roi     = results_np[:, 4]
        sortino = results_np[:, 5]

        # Filter: require at least 10 trades for validity
        valid = trades >= 10

        # DD <= 10% profiles
        dd10_mask = valid & (dd_pct <= 10.0)
        if dd10_mask.any():
            dd10_idx = np.where(dd10_mask)[0]
            # Best balance
            best_bal_local = dd10_idx[np.argmax(bal[dd10_idx])]
            if bal[best_bal_local] > best['dd10_best']['score']:
                best['dd10_best']['score'] = float(bal[best_bal_local])
                best['dd10_best']['params'] = params_np[best_bal_local].copy()
                best['dd10_best']['metrics'] = results_np[best_bal_local].copy()
            # Best sortino
            best_sort_local = dd10_idx[np.argmax(sortino[dd10_idx])]
            if sortino[best_sort_local] > best['dd10_sortino']['score']:
                best['dd10_sortino']['score'] = float(sortino[best_sort_local])
                best['dd10_sortino']['params'] = params_np[best_sort_local].copy()
                best['dd10_sortino']['metrics'] = results_np[best_sort_local].copy()

        # DD <= 15% profiles
        dd15_mask = valid & (dd_pct <= 15.0)
        if dd15_mask.any():
            dd15_idx = np.where(dd15_mask)[0]
            best_bal_local = dd15_idx[np.argmax(bal[dd15_idx])]
            if bal[best_bal_local] > best['dd15_best']['score']:
                best['dd15_best']['score'] = float(bal[best_bal_local])
                best['dd15_best']['params'] = params_np[best_bal_local].copy()
                best['dd15_best']['metrics'] = results_np[best_bal_local].copy()
            best_sort_local = dd15_idx[np.argmax(sortino[dd15_idx])]
            if sortino[best_sort_local] > best['dd15_sortino']['score']:
                best['dd15_sortino']['score'] = float(sortino[best_sort_local])
                best['dd15_sortino']['params'] = params_np[best_sort_local].copy()
                best['dd15_sortino']['metrics'] = results_np[best_sort_local].copy()

        total_processed += actual_batch
        now = time.time()

        # Progress every 5 seconds or last batch
        if now - last_print > 5.0 or batch_idx == n_batches - 1:
            pct = total_processed / total_combos * 100
            rate = total_processed / max(now - t_start, 0.01)
            remaining_combos = total_combos - total_processed
            eta_sec = remaining_combos / max(rate, 1)
            eta_min = eta_sec / 60

            # Show current best
            best_roi_str = ""
            if best['dd15_best']['metrics'] is not None:
                best_roi_str = f" | best dd15 ROI: {best['dd15_best']['metrics'][4]:+.1f}%"

            print(f"  [{pct:5.1f}%] {total_processed:>12,}/{total_combos:,} "
                  f"| {rate:,.0f} combos/s | ETA: {eta_min:.1f}min{best_roi_str}")
            last_print = now

        # Checkpoint save every ~5% of batches (or every save_interval batches)
        if (batch_idx + 1) % save_interval == 0:
            import pickle
            with open(checkpoint_path, 'wb') as f:
                pickle.dump({'best': best, 'next_batch': batch_idx + 1, 'total_processed': total_processed, 'tf_name': tf_name}, f)

    total_time = time.time() - t_start
    print(f"\n  Completed {total_combos:,} combos in {total_time:.1f}s "
          f"({total_combos / max(total_time, 0.01):,.0f} combos/sec)")

    # Clean up checkpoint on successful completion
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"  Checkpoint removed: {checkpoint_path}")

    return best


def format_exit_type(et):
    """Format exit_type value for display."""
    if et >= 0:
        return f"{int(et)}%TP"
    else:
        return f"Trail{int(abs(et))}x"


def params_to_dict(params, metrics):
    """Convert param array + metrics array to output dict."""
    if params is None or metrics is None:
        return None
    lev, risk, stop_atr, rr, hold, exit_type, conf = params
    bal, dd_pct, wr, trades, roi, sortino, total_trades = metrics
    return {
        'leverage': int(round(lev)),
        'risk_pct': round(float(risk), 4),
        'stop_atr': round(float(stop_atr), 4),
        'rr': round(float(rr), 4),
        'max_hold': int(round(hold)),
        'exit_type': int(round(exit_type)),
        'conf_thresh': round(float(conf), 4),
        'roi': round(float(roi), 2),
        'max_dd': round(float(dd_pct), 2),
        'win_rate': round(float(wr), 4),
        'trades': int(round(trades)),
        'sortino': round(float(sortino), 4),
        'final_balance': round(float(bal), 2),
    }


# ---------------------------------------------------------------------------
# PER-TF WORKER (for parallel optimization across GPUs)
# ---------------------------------------------------------------------------
def _optimize_single_tf(args_tuple):
    """
    Run grid search for a single TF in a subprocess.
    ALL GPUs visible — no CUDA_VISIBLE_DEVICES pinning.
    Each worker pins itself to a specific GPU via cp.cuda.Device(gpu_id).use().
    Returns: (tf_name, tf_config_dict, n_combos) or (tf_name, None, n_combos) on failure.
    """
    tf_name, resume, gpu_id = args_tuple
    import gc
    try:
        # Pin this worker to a specific GPU — all subsequent cp.asarray() calls go here
        if GPU_ARRAY:
            cp.cuda.Device(gpu_id).use()
            print(f"\n{elapsed()} Worker {tf_name.upper()} pinned to GPU {gpu_id}", flush=True)

        print(f"\n{elapsed()} Loading data for {tf_name.upper()} (worker, GPU {gpu_id})...", flush=True)
        data = load_tf_data(tf_name)
        if data is None:
            print(f"  Skipping {tf_name} — data not available", flush=True)
            return (tf_name, None, 0)

        confs, dirs, rets, closes, atrs, highs, lows, n_bars = data
        best = run_grid_search(tf_name, confs, dirs, closes, atrs, highs, lows, n_bars, resume=resume)

        tf_config = {}
        for profile_name, profile_data in best.items():
            result = params_to_dict(profile_data['params'], profile_data['metrics'])
            if result is not None:
                tf_config[profile_name] = result

        n_combos = count_grid(TF_GRIDS[tf_name])

        # Save per-TF config immediately (atomic)
        if tf_config:
            per_tf_path = f"{DB_DIR}/exhaustive_configs_{tf_name}.json"
            with open(per_tf_path, 'w') as f:
                json.dump({tf_name: tf_config}, f, indent=2)
            print(f"  {elapsed()} Saved per-TF config: {per_tf_path}", flush=True)

        gc.collect()
        return (tf_name, tf_config, n_combos)
    except Exception as e:
        print(f"  [FAILED] Optimizer for {tf_name}: {e}", flush=True)
        import traceback; traceback.print_exc()
        return (tf_name, None, 0)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main(resume=False):
    print(f"\n{'='*70}")
    print(f"  EXHAUSTIVE GRID SEARCH OPTIMIZER")
    print(f"  GPU Array: {'CuPy (CUDA)' if GPU_ARRAY else 'NumPy (CPU)'}")
    print(f"  XGB GPU:   {'ENABLED' if USE_GPU_XGB else 'CPU'}")
    print(f"  GPUs:      {_N_GPUS}")
    print(f"  Fee model: {TOTAL_COST_PER_TRADE*100:.2f}% round-trip")
    print(f"  Started:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    # Print grid sizes
    print(f"\nParameter grid sizes:")
    grand_total = 0
    for tf, grid in TF_GRIDS.items():
        n = count_grid(grid)
        grand_total += n
        print(f"  {tf:>4s}: {n:>15,} combinations")
    print(f"  {'TOTAL':>4s}: {grand_total:>15,} combinations across all TFs")

    # Process each TF — parallel across GPUs (each TF is independent)
    tf_order = [tf for tf in ['5m', '15m', '1h', '4h', '1d', '1w'] if tf in TF_GRIDS]
    all_configs = {}
    total_combos_tested = 0
    opt_workers = min(_N_GPUS, len(tf_order))

    if opt_workers > 1:
        # Parallel: one TF per GPU, round-robin GPU assignment
        print(f"\n  PARALLEL OPTIMIZATION: {opt_workers} workers across {_N_GPUS} GPUs")
        worker_args = [(tf_name, resume, i % _N_GPUS) for i, tf_name in enumerate(tf_order)]
        with ProcessPoolExecutor(max_workers=opt_workers) as pool:
            futures = {pool.submit(_optimize_single_tf, wa): wa[0] for wa in worker_args}
            for future in as_completed(futures):
                tf_name = futures[future]
                try:
                    tf_name_r, tf_config, n_combos = future.result()
                    if tf_config:
                        all_configs[tf_name_r] = tf_config
                    total_combos_tested += n_combos
                    print(f"  {elapsed()} Completed {tf_name_r}: {n_combos:,} combos", flush=True)
                except Exception as e:
                    print(f"  {elapsed()} FAILED {tf_name}: {e}", flush=True)
        import gc; gc.collect()
    else:
        # Sequential fallback (single GPU)
        if GPU_ARRAY:
            cp.cuda.Device(0).use()
        for tf_name in tf_order:
            print(f"\n{elapsed()} Loading data for {tf_name.upper()}...")
            data = load_tf_data(tf_name)
            if data is None:
                print(f"  Skipping {tf_name} — data not available")
                continue

            confs, dirs, rets, closes, atrs, highs, lows, n_bars = data
            best = run_grid_search(tf_name, confs, dirs, closes, atrs, highs, lows, n_bars, resume=resume)

            tf_config = {}
            for profile_name, profile_data in best.items():
                result = params_to_dict(profile_data['params'], profile_data['metrics'])
                if result is not None:
                    tf_config[profile_name] = result

            if tf_config:
                all_configs[tf_name] = tf_config

            total_combos_tested += count_grid(TF_GRIDS[tf_name])

    # Save results
    # Use TF-specific filename when running in parallel mode
    tf_suffix = '_'.join(sorted(all_configs.keys())) if all_configs else 'all'
    output_path = f"{DB_DIR}/exhaustive_configs_{tf_suffix}.json" if len(TF_GRIDS) < 6 else f"{DB_DIR}/exhaustive_configs.json"
    with open(output_path, 'w') as f:
        json.dump(all_configs, f, indent=2)
    print(f"\n{elapsed()} Results saved to: {output_path}")

    # ---------------------------------------------------------------------------
    # Print comprehensive results table
    # ---------------------------------------------------------------------------
    print(f"\n\n{'='*120}")
    print(f"  EXHAUSTIVE GRID SEARCH RESULTS")
    print(f"{'='*120}")

    profiles = ['dd10_best', 'dd10_sortino', 'dd15_best', 'dd15_sortino']
    profile_labels = {
        'dd10_best':    'DD<=10% MaxROI',
        'dd10_sortino': 'DD<=10% MaxSortino',
        'dd15_best':    'DD<=15% MaxROI',
        'dd15_sortino': 'DD<=15% MaxSortino',
    }

    for tf_name in tf_order:
        if tf_name not in all_configs:
            continue

        tf_cfg = all_configs[tf_name]
        print(f"\n  {tf_name.upper()}")
        print(f"  {'-'*116}")
        print(f"  {'Profile':<22s} {'Lev':>4s} {'Risk%':>6s} {'SL_ATR':>6s} {'RR':>5s} "
              f"{'Hold':>5s} {'Exit':>8s} {'Conf':>6s} {'ROI%':>10s} {'DD%':>6s} "
              f"{'WR':>6s} {'Trades':>7s} {'Sortino':>8s} {'Final$':>12s}")
        print(f"  {'-'*116}")

        for prof in profiles:
            if prof not in tf_cfg:
                print(f"  {profile_labels[prof]:<22s} {'--- no valid combo found ---'}")
                continue
            c = tf_cfg[prof]
            et_str = format_exit_type(c['exit_type'])
            print(f"  {profile_labels[prof]:<22s} {c['leverage']:>4d} {c['risk_pct']:>6.2f} "
                  f"{c['stop_atr']:>6.2f} {c['rr']:>5.2f} {c['max_hold']:>5d} {et_str:>8s} "
                  f"{c['conf_thresh']:>6.2f} {c['roi']:>+10.1f} {c['max_dd']:>6.1f} "
                  f"{c['win_rate']:>6.2f} {c['trades']:>7d} {c['sortino']:>8.2f} "
                  f"{c['final_balance']:>12,.2f}")

    print(f"\n{'='*120}")
    total_time = time.time() - START_TIME
    print(f"  Total combinations tested: {total_combos_tested:,}")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Average rate: {total_combos_tested / max(total_time, 0.01):,.0f} combos/sec")
    print(f"  Output: {output_path}")
    print(f"{'='*120}\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf', action='append', help='Only run specific timeframes (can repeat)')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    args = parser.parse_args()
    if args.tf:
        # Override tf_order in main() — monkey-patch before calling
        _original_main = main
        def _filtered_main():
            global TF_GRIDS
            valid_tfs = [t for t in args.tf if t in TF_GRIDS]
            if not valid_tfs:
                print("ERROR: No valid timeframes in %s" % args.tf)
                return
            # Filter TF_GRIDS to only requested TFs
            TF_GRIDS = {k: v for k, v in TF_GRIDS.items() if k in valid_tfs}
            _original_main(resume=args.resume)
        _filtered_main()
    else:
        main(resume=args.resume)
