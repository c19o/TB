#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
exhaustive_optimizer.py — Optuna TPE Optimizer (LightGBM)
==========================================================
Replaces exhaustive grid search with Optuna's TPE sampler for intelligent
Bayesian optimization across 6 timeframes (5m, 15m, 1H, 4H, 1D, 1W).

Uses RTX 3090 GPU via CuPy (falls back to NumPy) for vectorized simulation.
LightGBM replaces XGBoost for model inference.

Usage:
    python exhaustive_optimizer.py --n-trials 200
    python exhaustive_optimizer.py --tf 1h --tf 4h --n-trials 500
"""

import sys, os, io, time, json, warnings, pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
if os.name == 'nt':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime
import optuna

# Suppress Optuna's verbose trial logging (we print our own progress)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# GPU backend: try CuPy, fall back to NumPy
# ---------------------------------------------------------------------------
try:
    import cupy as cp
    xp = cp
    GPU_ARRAY = True
    print(f"[GPU] CuPy + CUDA detected — RTX 3090 24GB — OPTUNA OPTIMIZER")
except ImportError:
    xp = np
    GPU_ARRAY = False
    print("[CPU] CuPy not available — using NumPy (slower)")

import lightgbm as lgb
try:
    from hardware_detect import detect_hardware
except ImportError:
    def detect_hardware():
        import multiprocessing
        return {'cpu_count': multiprocessing.cpu_count() or 1, 'ram_gb': 64.0, 'gpu_count': 0, 'n_gpus': 0}
from config import (FEE_RATE as CONFIG_FEE_RATE, STARTING_BALANCE as CONFIG_STARTING_BALANCE,
                    REGIME_MULT as CONFIG_REGIME_MULT, REGIME_SLOPE_THRESHOLD,
                    REGIME_CRASH_VOL_MULT, REGIME_CRASH_DD_THRESHOLD,
                    OPTUNA_SEED, OPTUNA_N_STARTUP_TRIALS,
                    CONFIDENCE_SIZE_TIERS, TF_SLIPPAGE, DRAWDOWN_PROTOCOL)

_HW = detect_hardware()
_N_GPUS = _HW['n_gpus'] or 1
print(f"[LGB] LightGBM loaded for model inference")
print(f"[HW] {_N_GPUS} GPU(s) detected for parallel TF optimization")

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
DB_DIR = os.environ.get('SAVAGE22_DB_DIR', os.path.dirname(os.path.abspath(__file__)))
START_TIME = time.time()
TOTAL_COST_PER_TRADE = CONFIG_FEE_RATE  # from config.py (single source of truth)
STARTING_BALANCE = CONFIG_STARTING_BALANCE

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
    '15m': {
        'lev': list(range(1, 126, 8)),                                        # up to 125x
        'risk': list(np.round(np.arange(0.01, 3.01, 0.30), 4)),              # 10
        'stop_atr': list(np.round(np.arange(0.1, 1.51, 0.14), 4)),           # 11
        'rr': list(np.round(np.arange(1.0, 5.1, 0.40), 4)),                  # 11
        'hold': [1,2,4,8,12,20,30,42,48,60],                                  # 10
        'exit_type': [0, 25, 50, 75, -2, -3],                                 # 6
        'conf': list(np.round(np.arange(0.45, 0.91, 0.10), 4)),              # 5
    },
    '1h': {
        'lev': list(range(1, 126, 5)),                                        # up to 125x
        'risk': list(np.round(np.arange(0.05, 4.01, 0.40), 4)),              # 10
        'stop_atr': list(np.round(np.arange(0.2, 2.01, 0.18), 4)),           # 10
        'rr': list(np.round(np.arange(1.0, 6.1, 0.50), 4)),                  # 11
        'hold': [1,2,4,8,12,20,30,48,60,72],                                  # 10
        'exit_type': [0, 25, 50, 75, -2, -3],                                 # 6
        'conf': list(np.round(np.arange(0.45, 0.91, 0.10), 4)),              # 5
    },
    '4h': {
        'lev': list(range(1, 126, 5)),                                        # up to 125x
        'risk': list(np.round(np.arange(0.1, 5.01, 0.50), 4)),               # 10
        'stop_atr': list(np.round(np.arange(0.3, 3.01, 0.27), 4)),           # 10
        'rr': list(np.round(np.arange(1.0, 8.1, 0.70), 4)),                  # 11
        'hold': [1,2,4,8,12,20,30,48,66,84],                                  # 10
        'exit_type': [0, 25, 50, 75, -2, -3],                                 # 6
        'conf': list(np.round(np.arange(0.45, 0.91, 0.10), 4)),              # 5
    },
    '1d': {
        'lev': list(range(1, 21)),                                            # 1-20x — daily swing trade
        'risk': list(np.round(np.arange(0.5, 5.01, 0.50), 4)),               # 10
        'stop_atr': list(np.round(np.arange(1.0, 4.01, 0.30), 4)),           # wider stops for daily
        'rr': list(np.round(np.arange(1.5, 8.1, 0.60), 4)),                  # 11
        'hold': [3, 7, 14, 21, 30, 45, 60, 90],                               # days to months
        'exit_type': [0, 25, 50, 75, -2, -3],                                 # 6
        'conf': list(np.round(np.arange(0.45, 0.91, 0.10), 4)),              # 5
    },
    '1w': {
        'lev': [1, 2, 3],                                                     # 1-3x only — weekly = patient position trade
        'risk': list(np.round(np.arange(0.5, 5.01, 0.50), 4)),               # moderate risk per trade
        'stop_atr': list(np.round(np.arange(2.0, 8.01, 0.60), 4)),           # wide stops for weekly
        'rr': list(np.round(np.arange(1.5, 10.1, 0.80), 4)),                 # big R:R — let winners run
        'hold': [4, 8, 13, 20, 26, 39, 52],                                   # 1 month to 1 year
        'exit_type': [0, 25, -2, -3],                                         # trailing stops preferred
        'conf': list(np.round(np.arange(0.45, 0.91, 0.10), 4)),              # 5
    },
}

# DB mapping  (matches ml_multi_tf.py TF_CONFIGS)
TF_DB_MAP = {
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
    # LightGBM handles NaN natively — do NOT replace with 0
    X_model = np.empty((len(df), len(model_features)), dtype=np.float32)
    for i, feat in enumerate(model_features):
        if feat in feature_cols:
            X_model[:, i] = pd.to_numeric(df[feat], errors='coerce').values.astype(np.float32)
        else:
            X_model[:, i] = np.nan  # missing feature — LightGBM treats NaN as missing

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

    # Try to load saved CPCV OOS predictions (avoids data leakage from re-prediction)
    cpcv_oos_path = f"{DB_DIR}/cpcv_oos_{tf_name}.pkl"
    used_cpcv_oos = False
    if os.path.exists(cpcv_oos_path):
        try:
            with open(cpcv_oos_path, 'rb') as f:
                cpcv_oos = pickle.load(f)
            if isinstance(cpcv_oos, dict) and 'predictions' in cpcv_oos:
                oos_preds = np.array(cpcv_oos['predictions'], dtype=np.float32)
                oos_indices = cpcv_oos.get('indices', list(range(len(oos_preds))))
                # Check if OOS predictions cover the test window
                n_test = ve - vs
                if len(oos_preds) >= n_test:
                    # Use last n_test predictions (matching test window)
                    raw_preds = oos_preds[-n_test:]
                    print(f"  Using saved CPCV OOS predictions ({len(oos_preds)} samples, using last {n_test})")
                    used_cpcv_oos = True
                elif len(oos_preds) > 0:
                    print(f"  CPCV OOS has {len(oos_preds)} samples but need {n_test} — falling back to re-prediction")
                else:
                    print(f"  CPCV OOS file empty — falling back to re-prediction")
            else:
                print(f"  CPCV OOS file exists but unexpected format — falling back to re-prediction")
        except Exception as e:
            print(f"  CPCV OOS load failed ({e}) — falling back to re-prediction")

    if not used_cpcv_oos:
        # Load trained LightGBM model and re-predict (fallback when CPCV OOS not available)
        model = lgb.Booster(model_file=model_path)
        # Predict on full test window
        # LightGBM predict() takes raw numpy arrays directly (no DMatrix needed)
        raw_preds = model.predict(X_model[vs:ve])

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
def simulate_batch(params_batch, confs, dirs, closes, atrs, highs, lows, regime, xp_lib,
                    slippage=0.0):
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
    slippage: float — per-side slippage (applied to both entry and exit)

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
    entry_conf_mult = xp_lib.ones(N, dtype=xp_lib.float32)  # confidence-scaled sizing
    alive      = xp_lib.ones(N, dtype=xp_lib.bool_)     # balance > 0
    scaled_in  = xp_lib.zeros(N, dtype=xp_lib.bool_)    # 8C.3: track if trade was scaled into

    # For Sortino: track sum of log returns and sum of squared negative log returns
    sum_log_ret     = xp_lib.zeros(N, dtype=xp_lib.float64)
    sum_neg_sq      = xp_lib.zeros(N, dtype=xp_lib.float64)
    count_neg       = xp_lib.zeros(N, dtype=xp_lib.int32)
    total_trades    = xp_lib.zeros(N, dtype=xp_lib.int32)

    # Pre-compute: is this a trailing stop config?
    is_trail   = (exit_type < 0)                          # -2 or -3
    trail_mult = xp_lib.abs(exit_type) * is_trail.astype(xp_lib.float32)  # 2 or 3 (or 0)
    is_partial = (~is_trail)                               # 0, 25, 50, 75 partial TP

    # Pre-compute drawdown protocol thresholds (sorted descending for priority)
    dd_protocol_levels = sorted(DRAWDOWN_PROTOCOL.keys(), reverse=True)
    dd_protocol = []
    for dd_thresh in dd_protocol_levels:
        dd_cfg = DRAWDOWN_PROTOCOL[dd_thresh]
        dd_protocol.append({
            'threshold': dd_thresh,
            'risk_mult': dd_cfg['risk_multiplier'],
            'min_conf': dd_cfg.get('min_confidence'),
        })

    # Drawdown protocol state: current risk multiplier and min_confidence override
    dd_risk_mult = xp_lib.ones(N, dtype=xp_lib.float32)
    dd_min_conf  = xp_lib.zeros(N, dtype=xp_lib.float32)  # 0 = no override

    fee_rate = TOTAL_COST_PER_TRADE
    total_slippage = 2.0 * slippage  # applied to entry AND exit

    # Regime multipliers derived from config.py (single source of truth)
    REGIME_LEV_MULT_np  = np.array([CONFIG_REGIME_MULT[i]['lev']  for i in range(4)], dtype=np.float32)
    REGIME_RISK_MULT_np = np.array([CONFIG_REGIME_MULT[i]['risk'] for i in range(4)], dtype=np.float32)
    REGIME_STOP_MULT_np = np.array([CONFIG_REGIME_MULT[i]['stop'] for i in range(4)], dtype=np.float32)
    REGIME_RR_MULT_np   = np.array([CONFIG_REGIME_MULT[i]['rr']   for i in range(4)], dtype=np.float32)
    REGIME_HOLD_MULT_np = np.array([CONFIG_REGIME_MULT[i]['hold'] for i in range(4)], dtype=np.float32)

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

            # 8C.3: Scale-in on bar 2 if profitable and not yet scaled in
            bar2 = active & (trade_bars == 2) & (~scaled_in)
            if bar2.any():
                # Check profitability: current price vs entry
                profitable_long = bar2 & (trade_dir == 1) & (p_val > entry_pr)
                profitable_short = bar2 & (trade_dir == -1) & (p_val < entry_pr)
                scale_in_mask = profitable_long | profitable_short
                if scale_in_mask.any():
                    # Add 50% size: multiply entry_conf_mult by 1.5
                    entry_conf_mult = xp_lib.where(scale_in_mask,
                                                   entry_conf_mult * 1.5, entry_conf_mult)
                    # Update entry to weighted avg: (2/3 * old_entry + 1/3 * current)
                    # Because original=1.0 share, adding 0.5 share at current price
                    # total=1.5 shares, weights: 1.0/1.5=0.667, 0.5/1.5=0.333
                    new_entry = entry_pr * (2.0 / 3.0) + p_val * (1.0 / 3.0)
                    entry_pr = xp_lib.where(scale_in_mask, new_entry, entry_pr)
                    # Widen stop by 20% from new entry
                    old_stop_dist = xp_lib.abs(entry_pr - stop_pr)
                    new_stop_dist = old_stop_dist * 1.2
                    stop_pr = xp_lib.where(
                        scale_in_mask & (trade_dir == 1),
                        entry_pr - new_stop_dist, stop_pr)
                    stop_pr = xp_lib.where(
                        scale_in_mask & (trade_dir == -1),
                        entry_pr + new_stop_dist, stop_pr)
                    # Mark as scaled in
                    scaled_in = xp_lib.where(scale_in_mask, True, scaled_in)

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
                fee_cost  = (fee_rate + total_slippage) * eff_lev
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

                eff_risk = risk_pct * risk_m * entry_conf_mult * dd_risk_mult  # confidence-scaled + DD protocol
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
                entry_conf_mult = xp_lib.where(exiting, 1.0, entry_conf_mult)
                scaled_in  = xp_lib.where(exiting, False, scaled_in)  # 8C.3: reset scale-in flag

                # Check alive
                alive = alive & (balance > 0)

        # --- Entry logic for those NOT in trade ---
        can_enter = (~in_trade) & alive
        if can_enter.any():
            # Use model direction (dirs_t): +1=LONG, -1=SHORT, 0=FLAT
            # Only enter when confidence exceeds threshold AND model has a direction
            # Drawdown protocol: enforce min_confidence override when in DD
            eff_conf_th = xp_lib.maximum(conf_th, dd_min_conf)
            go_long  = can_enter & (dirs_t == 1.0) & (c_val > eff_conf_th) & (dd_risk_mult > 0)
            go_short = can_enter & (dirs_t == -1.0) & (c_val > eff_conf_th) & (dd_risk_mult > 0)

            entering = go_long | go_short

            if entering.any():
                new_dir = xp_lib.where(go_long, 1, xp_lib.where(go_short, -1, 0))

                # Confidence-scaled position sizing (matches live_trader.py)
                # Higher confidence = larger position. Tiers from config.py.
                _conf_mult = 0.25  # below all tiers
                for _ct, _cm in CONFIDENCE_SIZE_TIERS:
                    if c_val >= _ct:
                        _conf_mult = _cm
                        break
                # Store conf_mult for this entry (used in PnL calc)
                entry_conf_mult = xp_lib.where(entering, _conf_mult, entry_conf_mult)

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

        # Drawdown protocol: adjust risk/confidence based on current DD
        # Reset to defaults first, then apply highest matching threshold
        dd_risk_mult[:] = 1.0
        dd_min_conf[:]  = 0.0
        for dp in dd_protocol:
            dd_breach = (dd >= dp['threshold'])
            dd_risk_mult = xp_lib.where(dd_breach, dp['risk_mult'], dd_risk_mult)
            if dp['min_conf'] is not None:
                dd_min_conf = xp_lib.where(dd_breach, dp['min_conf'], dd_min_conf)

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
# Regime detection (shared by optimizer)
# ---------------------------------------------------------------------------
def detect_regime(closes):
    """
    Compute per-bar regime array from close prices.
    0=bull, 1=bear, 2=sideways, 3=crash (matches live_trader.py).
    Vectorized — no Python for-loops on arrays.
    """
    sma100 = pd.Series(closes).rolling(100, min_periods=1).mean().values
    slope = np.gradient(sma100) / np.maximum(sma100, 1e-8)

    log_returns = np.diff(np.log(np.maximum(closes, 1e-8)), prepend=0.0)
    rvol_20 = pd.Series(np.abs(log_returns)).rolling(20, min_periods=1).std().values
    rvol_90_avg = pd.Series(rvol_20).rolling(90, min_periods=1).mean().values
    rolling_high_30 = pd.Series(closes).rolling(30, min_periods=1).max().values
    dd_from_30h = (rolling_high_30 - closes) / np.maximum(rolling_high_30, 1e-8)

    above_sma = closes > sma100

    # Vectorized regime detection (replaces per-bar Python loop)
    regime = np.full(len(closes), 2, dtype=np.int32)  # default sideways
    regime[above_sma & (slope > REGIME_SLOPE_THRESHOLD)] = 0  # bull
    regime[~above_sma & (slope < -REGIME_SLOPE_THRESHOLD)] = 1  # bear
    # Crash overrides bear/sideways
    crash_mask = ((rvol_20 > REGIME_CRASH_VOL_MULT * np.maximum(rvol_90_avg, 1e-12))
                  & ~above_sma & (dd_from_30h > REGIME_CRASH_DD_THRESHOLD))
    regime[crash_mask] = 3

    return regime


# ---------------------------------------------------------------------------
# Optuna TPE optimizer for one TF
# ---------------------------------------------------------------------------
def run_optuna_search(tf_name, confs, dirs, closes, atrs, highs, lows, n_bars, n_trials=200):
    """
    Run Optuna TPE search for one timeframe.
    Returns dict of best configs per profile (dd10_best, dd10_sortino, dd15_best, dd15_sortino).
    """
    grid = TF_GRIDS[tf_name]
    print(f"\n{'='*70}")
    print(f"  {tf_name.upper()} OPTUNA TPE OPTIMIZER")
    print(f"  Trials: {n_trials}")
    print(f"  Test window: {n_bars} bars")
    print(f"{'='*70}")

    # Per-TF slippage from config.py (applied to entry AND exit)
    tf_slippage = TF_SLIPPAGE.get(tf_name, 0.0)
    print(f"  Slippage: {tf_slippage*100:.4f}% per side (from TF_SLIPPAGE)")

    # Compute regime
    regime = detect_regime(closes)
    regime_counts = {0: np.sum(regime==0), 1: np.sum(regime==1), 2: np.sum(regime==2), 3: np.sum(regime==3)}
    print(f"  Regime distribution: bull={regime_counts[0]} bear={regime_counts[1]} "
          f"sideways={regime_counts[2]} crash={regime_counts[3]}")

    # Transfer market data to GPU ONCE (shared across all trials)
    if GPU_ARRAY:
        g_confs  = cp.asarray(confs)
        g_dirs   = cp.asarray(dirs)
        g_closes = cp.asarray(closes)
        g_atrs   = cp.asarray(atrs)
        g_highs  = cp.asarray(highs)
        g_lows   = cp.asarray(lows)
        g_regime = cp.asarray(regime)
        xp_lib = cp
    else:
        g_confs  = confs
        g_dirs   = dirs
        g_closes = closes
        g_atrs   = atrs
        g_highs  = highs
        g_lows   = lows
        g_regime = regime
        xp_lib = np

    # Extract parameter ranges from TF_GRIDS
    lev_min, lev_max = int(grid['lev'][0]), int(grid['lev'][-1])
    risk_min, risk_max = float(grid['risk'][0]), float(grid['risk'][-1])
    stop_min, stop_max = float(grid['stop_atr'][0]), float(grid['stop_atr'][-1])
    rr_min, rr_max = float(grid['rr'][0]), float(grid['rr'][-1])
    hold_min, hold_max = int(grid['hold'][0]), int(grid['hold'][-1])
    exit_types = grid['exit_type']
    conf_min, conf_max = float(grid['conf'][0]), float(grid['conf'][-1])

    print(f"  Param ranges: lev=[{lev_min},{lev_max}] risk=[{risk_min},{risk_max}] "
          f"stop=[{stop_min},{stop_max}] rr=[{rr_min},{rr_max}] "
          f"hold=[{hold_min},{hold_max}] exit={exit_types} conf=[{conf_min},{conf_max}]")

    # Track ALL trial results for 4-profile extraction
    all_trial_results = []  # list of (params_array, metrics_array)
    t_start = time.time()
    trial_count = [0]  # mutable for closure

    def objective(trial):
        lev = trial.suggest_int('lev', lev_min, lev_max)
        risk = trial.suggest_float('risk', risk_min, risk_max)
        stop_atr = trial.suggest_float('stop_atr', stop_min, stop_max)
        rr = trial.suggest_float('rr', rr_min, rr_max)
        hold = trial.suggest_int('hold', hold_min, hold_max)
        exit_type = trial.suggest_categorical('exit_type', exit_types)
        conf = trial.suggest_float('conf', conf_min, conf_max)

        # Build (1, 7) param array for simulate_batch
        params_np = np.array([[lev, risk, stop_atr, rr, hold, exit_type, conf]], dtype=np.float32)

        if GPU_ARRAY:
            params_gpu = cp.asarray(params_np)
            results = simulate_batch(params_gpu, g_confs, g_dirs, g_closes, g_atrs,
                                     g_highs, g_lows, g_regime, cp,
                                     slippage=tf_slippage)
            results_np = cp.asnumpy(results)
            del params_gpu, results
        else:
            results_np = simulate_batch(params_np, g_confs, g_dirs, g_closes, g_atrs,
                                        g_highs, g_lows, g_regime, np,
                                        slippage=tf_slippage)

        # results_np: (1, 7) = [balance, max_dd_pct, win_rate, trade_count, roi_pct, sortino, total_trades]
        result = results_np[0]
        bal = float(result[0])
        max_dd = float(result[1])
        trades = float(result[3])
        roi = float(result[4])
        sortino_val = float(result[5])

        # Store for 4-profile extraction later
        all_trial_results.append((params_np[0].copy(), results_np[0].copy()))

        # Store metrics as user attributes for later filtering
        trial.set_user_attr('balance', bal)
        trial.set_user_attr('max_dd', max_dd)
        trial.set_user_attr('trades', trades)
        trial.set_user_attr('roi', roi)
        trial.set_user_attr('sortino', sortino_val)

        trial_count[0] += 1

        # Progress logging every 10 trials
        if trial_count[0] % 10 == 0 or trial_count[0] == n_trials:
            elapsed_s = time.time() - t_start
            rate = trial_count[0] / max(elapsed_s, 0.01)
            s_display = sortino_val if not np.isnan(sortino_val) else -999
            print(f"  Trial {trial_count[0]:>4d}/{n_trials} | "
                  f"sortino={s_display:+.3f} dd={max_dd:.1f}% roi={roi:+.1f}% trades={int(trades)} | "
                  f"{rate:.1f} trials/s", flush=True)

        # Penalize bad combos but let Optuna learn from them
        if trades < 10:
            return -999.0

        # Clean NaN sortino
        if np.isnan(sortino_val):
            sortino_val = -999.0

        # Primary objective: sortino with DD penalty
        if max_dd > 20:
            return sortino_val * 0.1
        elif max_dd > 15:
            return sortino_val * 0.5
        else:
            return sortino_val

    # Create Optuna study with TPE sampler — persisted to SQLite for resume
    storage = f"sqlite:///{DB_DIR}/optuna_optimizer_{tf_name}.db"
    study = optuna.create_study(
        study_name=f"optimizer_{tf_name}",
        storage=storage,
        load_if_exists=True,
        direction='maximize',
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=min(OPTUNA_N_STARTUP_TRIALS, n_trials // 4),
            multivariate=True,
            seed=OPTUNA_SEED,
        ),
    )

    print(f"\n  Starting Optuna optimization ({n_trials} trials)...")
    # n_jobs from hardware_detect, capped at 8 to avoid excessive memory with CuPy concurrent trials
    _opt_n_jobs = min(_HW['cpu_count'], 8)
    study.optimize(objective, n_trials=n_trials, n_jobs=_opt_n_jobs, show_progress_bar=False)

    total_time = time.time() - t_start
    print(f"\n  Completed {n_trials} trials in {total_time:.1f}s "
          f"({n_trials / max(total_time, 0.01):.1f} trials/sec)")
    print(f"  Best trial sortino (penalized): {study.best_value:.4f}")

    # --- Extract 4 profiles from ALL trial results ---
    best = {
        'dd10_best':    {'score': -np.inf, 'params': None, 'metrics': None},
        'dd10_sortino': {'score': -np.inf, 'params': None, 'metrics': None},
        'dd15_best':    {'score': -np.inf, 'params': None, 'metrics': None},
        'dd15_sortino': {'score': -np.inf, 'params': None, 'metrics': None},
    }

    for params_arr, metrics_arr in all_trial_results:
        bal = float(metrics_arr[0])
        dd_pct = float(metrics_arr[1])
        trades = float(metrics_arr[3])
        sortino_val = float(metrics_arr[5])

        # Require at least 10 trades
        if trades < 10:
            continue

        # DD <= 10% profiles
        if dd_pct <= 10.0:
            if bal > best['dd10_best']['score']:
                best['dd10_best']['score'] = bal
                best['dd10_best']['params'] = params_arr.copy()
                best['dd10_best']['metrics'] = metrics_arr.copy()
            if not np.isnan(sortino_val) and sortino_val > best['dd10_sortino']['score']:
                best['dd10_sortino']['score'] = sortino_val
                best['dd10_sortino']['params'] = params_arr.copy()
                best['dd10_sortino']['metrics'] = metrics_arr.copy()

        # DD <= 15% profiles
        if dd_pct <= 15.0:
            if bal > best['dd15_best']['score']:
                best['dd15_best']['score'] = bal
                best['dd15_best']['params'] = params_arr.copy()
                best['dd15_best']['metrics'] = metrics_arr.copy()
            if not np.isnan(sortino_val) and sortino_val > best['dd15_sortino']['score']:
                best['dd15_sortino']['score'] = sortino_val
                best['dd15_sortino']['params'] = params_arr.copy()
                best['dd15_sortino']['metrics'] = metrics_arr.copy()

    # Free GPU memory
    if GPU_ARRAY:
        del g_confs, g_dirs, g_closes, g_atrs, g_highs, g_lows, g_regime
        cp.get_default_memory_pool().free_all_blocks()

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
    Run Optuna search for a single TF in a subprocess.
    ALL GPUs visible — no CUDA_VISIBLE_DEVICES pinning.
    Each worker pins itself to a specific GPU via cp.cuda.Device(gpu_id).use().

    Returns: (tf_name, tf_config_dict, n_trials) or (tf_name, None, 0) on failure.
    """
    tf_name, n_trials, gpu_id = args_tuple
    import gc
    try:
        # Pin this worker to a specific GPU
        if GPU_ARRAY:
            cp.cuda.Device(gpu_id).use()
            print(f"\n{elapsed()} Worker {tf_name.upper()} pinned to GPU {gpu_id}", flush=True)

        print(f"\n{elapsed()} Loading data for {tf_name.upper()} (worker, GPU {gpu_id})...", flush=True)
        data = load_tf_data(tf_name)
        if data is None:
            print(f"  Skipping {tf_name} — data not available", flush=True)
            return (tf_name, None, 0)

        confs, dirs, rets, closes, atrs, highs, lows, n_bars = data
        best = run_optuna_search(tf_name, confs, dirs, closes, atrs, highs, lows, n_bars,
                                 n_trials=n_trials)

        tf_config = {}
        for profile_name, profile_data in best.items():
            result = params_to_dict(profile_data['params'], profile_data['metrics'])
            if result is not None:
                tf_config[profile_name] = result

        # Save per-TF config immediately (atomic) — same format as exhaustive_configs_{tf}.json
        if tf_config:
            per_tf_path = f"{DB_DIR}/optuna_configs_{tf_name}.json"
            with open(per_tf_path, 'w') as f:
                json.dump({tf_name: tf_config}, f, indent=2)
            print(f"  {elapsed()} Saved per-TF config: {per_tf_path}", flush=True)

        gc.collect()
        return (tf_name, tf_config, n_trials)
    except Exception as e:
        print(f"  [FAILED] Optimizer for {tf_name}: {e}", flush=True)
        import traceback; traceback.print_exc()
        return (tf_name, None, 0)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main(n_trials=200):
    print(f"\n{'='*70}")
    print(f"  OPTUNA TPE OPTIMIZER (LightGBM)")
    print(f"  GPU Array: {'CuPy (CUDA)' if GPU_ARRAY else 'NumPy (CPU)'}")
    print(f"  GPUs:      {_N_GPUS}")
    print(f"  Trials:    {n_trials} per TF")
    print(f"  Fee model: {TOTAL_COST_PER_TRADE*100:.2f}% round-trip")
    print(f"  Started:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    # Print grid ranges (for reference)
    print(f"\nParameter search ranges (from TF_GRIDS):")
    for tf, grid in TF_GRIDS.items():
        n = count_grid(grid)
        print(f"  {tf:>4s}: {n:>15,} exhaustive combos (Optuna samples {n_trials} intelligently)")

    # Process each TF — parallel across GPUs (each TF is independent)
    tf_order = [tf for tf in ['15m', '1h', '4h', '1d', '1w'] if tf in TF_GRIDS]
    all_configs = {}
    total_trials_run = 0
    opt_workers = min(_N_GPUS, len(tf_order))

    if opt_workers > 1:
        # Parallel: one TF per GPU, round-robin GPU assignment
        print(f"\n  PARALLEL OPTIMIZATION: {opt_workers} workers across {_N_GPUS} GPUs")
        worker_args = [(tf_name, n_trials, i % _N_GPUS) for i, tf_name in enumerate(tf_order)]
        with ProcessPoolExecutor(max_workers=opt_workers) as pool:
            futures = {pool.submit(_optimize_single_tf, wa): wa[0] for wa in worker_args}
            for future in as_completed(futures):
                tf_name = futures[future]
                try:
                    tf_name_r, tf_config, n_done = future.result()
                    if tf_config:
                        all_configs[tf_name_r] = tf_config
                    total_trials_run += n_done
                    print(f"  {elapsed()} Completed {tf_name_r}: {n_done} trials", flush=True)
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
            best = run_optuna_search(tf_name, confs, dirs, closes, atrs, highs, lows, n_bars,
                                     n_trials=n_trials)

            tf_config = {}
            for profile_name, profile_data in best.items():
                result = params_to_dict(profile_data['params'], profile_data['metrics'])
                if result is not None:
                    tf_config[profile_name] = result

            if tf_config:
                all_configs[tf_name] = tf_config

                # Save per-TF config immediately
                per_tf_path = f"{DB_DIR}/optuna_configs_{tf_name}.json"
                with open(per_tf_path, 'w') as f:
                    json.dump({tf_name: tf_config}, f, indent=2)
                print(f"  Saved: {per_tf_path}")

            total_trials_run += n_trials

    # Save combined results — same JSON format as exhaustive_configs.json
    tf_suffix = '_'.join(sorted(all_configs.keys())) if all_configs else 'all'
    output_path = f"{DB_DIR}/optuna_configs_{tf_suffix}.json" if len(TF_GRIDS) < 6 else f"{DB_DIR}/optuna_configs.json"
    with open(output_path, 'w') as f:
        json.dump(all_configs, f, indent=2)
    print(f"\n{elapsed()} Results saved to: {output_path}")

    # ---------------------------------------------------------------------------
    # Print comprehensive results table
    # ---------------------------------------------------------------------------
    print(f"\n\n{'='*120}")
    print(f"  OPTUNA TPE OPTIMIZATION RESULTS")
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
    print(f"  Total trials run: {total_trials_run:,} ({n_trials} per TF x {len(tf_order)} TFs)")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Output: {output_path}")
    print(f"{'='*120}\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Optuna TPE Optimizer for LightGBM trading strategy')
    parser.add_argument('--tf', action='append', help='Only run specific timeframes (can repeat)')
    parser.add_argument('--n-trials', type=int, default=200, help='Number of Optuna trials per TF (default: 200)')
    args = parser.parse_args()
    if args.tf:
        _original_main = main
        def _filtered_main():
            global TF_GRIDS
            valid_tfs = [t for t in args.tf if t in TF_GRIDS]
            if not valid_tfs:
                print("ERROR: No valid timeframes in %s" % args.tf)
                return
            TF_GRIDS = {k: v for k, v in TF_GRIDS.items() if k in valid_tfs}
            _original_main(n_trials=args.n_trials)
        _filtered_main()
    else:
        main(n_trials=args.n_trials)
