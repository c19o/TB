#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
exhaustive_optimizer.py — Optuna TPE Optimizer (LightGBM)
==========================================================
Replaces exhaustive grid search with Optuna's TPE sampler for intelligent
Bayesian optimization across 6 timeframes (5m, 15m, 1H, 4H, 1D, 1W).

Uses RTX 3090 GPU via CuPy (REQUIRED — no CPU fallback) for vectorized simulation.
LightGBM replaces XGBoost for model inference.

Usage:
    python exhaustive_optimizer.py --n-trials 200
    python exhaustive_optimizer.py --tf 1h --tf 4h --n-trials 500
"""

import sys, os, io, time, json, warnings, pickle, multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import qmc
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
os.environ.setdefault('CUPY_COMPILE_WITH_PTX', '1')  # Blackwell sm_120 compat
try:
    import cupy as cp
    # Verify GPU actually works (catches sm_120 / driver mismatch at import time)
    cp.array([1.0]) + cp.array([2.0])
    GPU_ARRAY = True
    print(f"[GPU] CuPy + CUDA detected — GPU verified — OPTUNA OPTIMIZER")
except Exception as e:
    raise RuntimeError(
        f"GPU REQUIRED: CuPy/CUDA failed ({e}). "
        f"Install CuPy with working CUDA. No CPU fallback."
    )

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
                    OPTUNA_SEED, OPTUNA_PHASE1_N_STARTUP,
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

# Batch size: auto-tuned per TF based on bar count
# RTX 3090 has 24 GB VRAM; state arrays are N-sized (not N*T), so batch can be large
# Reference: ~30 float32 state arrays * N * 4 bytes = ~120 bytes/combo
# At 500K combos = ~60 MB state (fits easily). Scale inversely with bar count
# to avoid excessive kernel launch time per batch.
GPU_BATCH_REFERENCE_BARS = 1000  # reference bar count for 500K batch
GPU_BATCH_REFERENCE_SIZE = 500_000  # batch size at reference bar count

# Optimizer mode: 'sobol' (default) = Sobol sweep + Bayesian refinement, 'optuna' = pure TPE
OPTIMIZER_MODE = os.environ.get('OPTIMIZER_MODE', 'sobol').lower()

# Sobol sweep defaults
SOBOL_N_CANDIDATES = int(os.environ.get('SOBOL_N_CANDIDATES', '131072'))  # 2^17 = 131072 (~100K, power-of-2)
SOBOL_TOP_K = int(os.environ.get('SOBOL_TOP_K', '256'))                   # top regions for Phase 2
SOBOL_PHASE2_TRIALS = int(os.environ.get('SOBOL_PHASE2_TRIALS', '200'))   # Bayesian refinement trials

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
        'conf': list(np.round(np.arange(0.34, 0.56, 0.04), 4)),              # 6 — lowered: 3-class model peaks at 0.55
    },
    '1h': {
        'lev': list(range(1, 126, 5)),                                        # up to 125x
        'risk': list(np.round(np.arange(0.05, 4.01, 0.40), 4)),              # 10
        'stop_atr': list(np.round(np.arange(0.2, 2.01, 0.18), 4)),           # 10
        'rr': list(np.round(np.arange(1.0, 6.1, 0.50), 4)),                  # 11
        'hold': [1,2,4,8,12,20,30,48,60,72],                                  # 10
        'exit_type': [0, 25, 50, 75, -2, -3],                                 # 6
        'conf': list(np.round(np.arange(0.34, 0.56, 0.04), 4)),              # 6 — lowered: 3-class model peaks at 0.55
    },
    '4h': {
        'lev': list(range(1, 126, 5)),                                        # up to 125x
        'risk': list(np.round(np.arange(0.1, 5.01, 0.50), 4)),               # 10
        'stop_atr': list(np.round(np.arange(0.3, 3.01, 0.27), 4)),           # 10
        'rr': list(np.round(np.arange(1.0, 8.1, 0.70), 4)),                  # 11
        'hold': [1,2,4,8,12,20,30,48,66,84],                                  # 10
        'exit_type': [0, 25, 50, 75, -2, -3],                                 # 6
        'conf': list(np.round(np.arange(0.34, 0.56, 0.04), 4)),              # 6 — lowered: 3-class model peaks at 0.55
    },
    '1d': {
        'lev': list(range(1, 21)),                                            # 1-20x — daily swing trade
        'risk': list(np.round(np.arange(0.5, 5.01, 0.50), 4)),               # 10
        'stop_atr': list(np.round(np.arange(1.0, 4.01, 0.30), 4)),           # wider stops for daily
        'rr': list(np.round(np.arange(1.5, 8.1, 0.60), 4)),                  # 11
        'hold': [3, 7, 14, 21, 30, 45, 60, 90],                               # days to months
        'exit_type': [0, 25, 50, 75, -2, -3],                                 # 6
        'conf': list(np.round(np.arange(0.34, 0.56, 0.04), 4)),              # 6 — lowered: 3-class model peaks at 0.55
    },
    '1w': {
        'lev': [1, 2, 3],                                                     # 1-3x only — weekly = patient position trade
        'risk': list(np.round(np.arange(0.5, 5.01, 0.50), 4)),               # moderate risk per trade
        'stop_atr': list(np.round(np.arange(2.0, 8.01, 0.60), 4)),           # wide stops for weekly
        'rr': list(np.round(np.arange(1.5, 10.1, 0.80), 4)),                 # big R:R — let winners run
        'hold': [4, 8, 13, 20, 26, 39, 52],                                   # 1 month to 1 year
        'exit_type': [0, 25, -2, -3],                                         # trailing stops preferred
        'conf': list(np.round(np.arange(0.34, 0.56, 0.04), 4)),              # 6 — lowered: 3-class model peaks at 0.55
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
# ---------------------------------------------------------------------------
# FIX #15: CuPy RawKernel — fuses entire bar-by-bar simulation into ONE kernel launch
# Each CUDA thread handles one parameter combo through all T bars.
# Eliminates ~30+ kernel launches per bar × T bars → 1 launch total.
# ---------------------------------------------------------------------------
_SIMULATE_KERNEL_CODE = r'''
extern "C" __global__
void simulate_kernel(
    const float* __restrict__ params,       // [N, 7] row-major
    const float* __restrict__ confs,        // [T]
    const float* __restrict__ dirs,         // [T]
    const float* __restrict__ closes,       // [T]
    const float* __restrict__ atrs,         // [T]
    const float* __restrict__ highs,        // [T]
    const float* __restrict__ lows,         // [T]
    const float* __restrict__ bar_lev_m,    // [T]
    const float* __restrict__ bar_risk_m,   // [T]
    const float* __restrict__ bar_stop_m,   // [T]
    const float* __restrict__ bar_rr_m,     // [T]
    const float* __restrict__ bar_hold_m,   // [T]
    const float* __restrict__ bar_conf_mult,// [T]
    const float* __restrict__ dd_thresh_vals,  // [n_dd]
    const float* __restrict__ dd_thresholds,   // [n_dd] risk multipliers
    const float* __restrict__ dd_min_confs_arr,// [n_dd]
    const int T,
    const int N,
    const int n_dd,
    const float starting_balance,
    const float fee_rate,
    const float total_slippage,
    float* __restrict__ results  // [N, 7]
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N) return;

    // Unpack params for this combo
    float lev = params[i * 7 + 0];
    float risk_pct = params[i * 7 + 1] / 100.0f;
    float stop_mult = params[i * 7 + 2];
    float rr = params[i * 7 + 3];
    float max_hold_base = params[i * 7 + 4];
    float exit_type_val = params[i * 7 + 5];
    float conf_th = params[i * 7 + 6];

    // Pre-compute trailing stop config
    bool is_trail = exit_type_val < 0.0f;
    float trail_mult = is_trail ? fabsf(exit_type_val) : 0.0f;
    bool is_partial = !is_trail;
    float partial_pct = (is_partial && exit_type_val > 0.0f) ? exit_type_val / 100.0f : 1.0f;

    // State
    float balance = starting_balance;
    float peak = starting_balance;
    float max_dd = 0.0f;
    int wins = 0, losses = 0;
    bool in_trade = false;
    int trade_bars = 0;
    int trade_dir_val = 0;
    float entry_pr = 0.0f, stop_pr_val = 0.0f, tp_pr_val = 0.0f, best_pr_val = 0.0f;
    float entry_conf_mult = 1.0f;
    bool alive = true;
    bool scaled_in = false;

    double sum_log_ret = 0.0;
    double sum_neg_sq = 0.0;
    int count_neg = 0;
    int total_trades = 0;

    float dd_risk_mult_val = 1.0f;
    float dd_min_conf_val = 0.0f;

    for (int t = 0; t < T; t++) {
        if (!alive) continue;

        float c_val = confs[t];
        float d_val = dirs[t];
        float p_val = closes[t];
        float h_val = highs[t];
        float l_val = lows[t];
        float a_val = atrs[t];
        float lev_m = bar_lev_m[t];
        float risk_m = bar_risk_m[t];
        float stop_m = bar_stop_m[t];
        float rr_m = bar_rr_m[t];
        float hold_m = bar_hold_m[t];
        float conf_mult_t = bar_conf_mult[t];

        if (in_trade) {
            trade_bars++;

            // Scale-in on bar 2
            if (trade_bars == 2 && !scaled_in) {
                bool profitable = (trade_dir_val == 1 && p_val > entry_pr) ||
                                  (trade_dir_val == -1 && p_val < entry_pr);
                if (profitable) {
                    entry_conf_mult *= 1.5f;
                    entry_pr = entry_pr * (2.0f / 3.0f) + p_val * (1.0f / 3.0f);
                    float old_stop_dist = fabsf(entry_pr - stop_pr_val) * 1.2f;
                    stop_pr_val = (trade_dir_val == 1) ? entry_pr - old_stop_dist : entry_pr + old_stop_dist;
                    scaled_in = true;
                }
            }

            // Update best price for trailing
            if (trade_dir_val == 1) best_pr_val = fmaxf(best_pr_val, h_val);
            else best_pr_val = fminf(best_pr_val, l_val);

            // Trailing stop update
            if (is_trail) {
                float one_r = stop_mult * a_val;
                if (trade_dir_val == 1 && p_val >= entry_pr + one_r) {
                    float new_trail = best_pr_val - trail_mult * a_val;
                    stop_pr_val = fmaxf(stop_pr_val, new_trail);
                }
                if (trade_dir_val == -1 && p_val <= entry_pr - one_r) {
                    float new_trail = best_pr_val + trail_mult * a_val;
                    stop_pr_val = fminf(stop_pr_val, new_trail);
                }
            }

            // SL check (intrabar)
            bool sl_hit = (trade_dir_val == 1 && l_val <= stop_pr_val) ||
                          (trade_dir_val == -1 && h_val >= stop_pr_val);

            // TP check (intrabar)
            bool tp_hit = (trade_dir_val == 1 && h_val >= tp_pr_val) ||
                          (trade_dir_val == -1 && l_val <= tp_pr_val);

            // Hold check (regime-adjusted)
            bool hold_exit = (float)trade_bars >= max_hold_base * hold_m;

            bool exiting = sl_hit || tp_hit || hold_exit;

            if (exiting) {
                // Priority: SL > TP > hold
                float exit_price = sl_hit ? stop_pr_val : (tp_hit ? tp_pr_val : p_val);

                float price_chg = (exit_price - entry_pr) / fmaxf(entry_pr, 1e-8f) * (float)trade_dir_val;
                float eff_lev = lev * lev_m;
                float gross_pnl = price_chg * eff_lev;
                float fee_cost = (fee_rate + total_slippage) * eff_lev;
                float net_pnl = gross_pnl - fee_cost;

                // Partial TP scale
                float scale = (tp_hit && is_partial && exit_type_val > 0.0f) ? partial_pct : 1.0f;

                float eff_risk = risk_pct * risk_m * entry_conf_mult * dd_risk_mult_val;
                float pnl_dollar = balance * eff_risk * net_pnl * scale;

                // Sortino accumulators
                float new_bal = fmaxf(balance + pnl_dollar, 1.0f);
                double log_ret_val = log((double)new_bal / fmax((double)balance, 1.0));
                sum_log_ret += log_ret_val;
                if (log_ret_val < 0.0) {
                    sum_neg_sq += log_ret_val * log_ret_val;
                    count_neg++;
                }
                total_trades++;

                balance = new_bal;
                if (pnl_dollar > 0.0f) wins++;
                else losses++;

                // Reset trade state
                in_trade = false;
                trade_bars = 0;
                entry_conf_mult = 1.0f;
                scaled_in = false;
            }
        }

        alive = balance > 0.0f;
        if (!alive) continue;

        // Entry logic
        if (!in_trade) {
            float eff_conf_th = fmaxf(conf_th, dd_min_conf_val);
            bool go_long = (d_val == 1.0f) && (c_val > eff_conf_th) && (dd_risk_mult_val > 0.0f);
            bool go_short = (d_val == -1.0f) && (c_val > eff_conf_th) && (dd_risk_mult_val > 0.0f);

            if (go_long || go_short) {
                trade_dir_val = go_long ? 1 : -1;
                entry_conf_mult = conf_mult_t;
                entry_pr = p_val;

                float sl_dist = stop_mult * stop_m * a_val;
                stop_pr_val = (trade_dir_val == 1) ? p_val - sl_dist : p_val + sl_dist;

                float tp_dist = sl_dist * rr * rr_m;
                tp_pr_val = (trade_dir_val == 1) ? p_val + tp_dist : p_val - tp_dist;

                best_pr_val = p_val;
                in_trade = true;
                trade_bars = 0;
                scaled_in = false;
            }
        }

        // Drawdown tracking
        peak = fmaxf(peak, balance);
        float dd = (peak > 0.0f) ? (peak - balance) / peak : 0.0f;
        max_dd = fmaxf(max_dd, dd);

        // Drawdown protocol
        dd_risk_mult_val = 1.0f;
        dd_min_conf_val = 0.0f;
        for (int di = 0; di < n_dd; di++) {
            if (dd >= dd_thresh_vals[di]) {
                dd_risk_mult_val = dd_thresholds[di];
                if (dd_min_confs_arr[di] > 0.0f) dd_min_conf_val = dd_min_confs_arr[di];
            }
        }
    }

    // Final metrics
    float total_tr = (float)(wins + losses);
    float win_rate = total_tr > 0.0f ? (float)wins / total_tr : 0.0f;
    float roi_pct = (balance - starting_balance) / starting_balance * 100.0f;
    float max_dd_pct = max_dd * 100.0f;

    // Sortino
    double mean_log = total_trades > 0 ? sum_log_ret / (double)total_trades : 0.0;
    double downside_var = count_neg > 0 ? sum_neg_sq / (double)count_neg : 0.0;
    double downside_std = sqrt(fmax(downside_var, 1e-12));
    float sortino = (downside_std > 1e-6) ? (float)(mean_log / downside_std) : (float)(mean_log * 10.0);

    results[i * 7 + 0] = balance;
    results[i * 7 + 1] = max_dd_pct;
    results[i * 7 + 2] = win_rate;
    results[i * 7 + 3] = total_tr;
    results[i * 7 + 4] = roi_pct;
    results[i * 7 + 5] = sortino;
    results[i * 7 + 6] = (float)total_trades;
}
'''

# Lazily compiled RawKernel (compiled once, reused)
_simulate_kernel = None

def _get_simulate_kernel():
    """Compile and cache the simulation RawKernel."""
    global _simulate_kernel
    if _simulate_kernel is None:
        _simulate_kernel = cp.RawKernel(_SIMULATE_KERNEL_CODE, 'simulate_kernel')
    return _simulate_kernel


def simulate_batch_rawkernel(params_batch, confs_cpu, dirs_cpu, closes_cpu, atrs_cpu,
                             highs_cpu, lows_cpu, bar_lev_m, bar_risk_m, bar_stop_m,
                             bar_rr_m, bar_hold_m, bar_conf_mult,
                             dd_thresh_vals, dd_thresholds, dd_min_confs,
                             starting_balance, fee_rate, total_slippage):
    """Run simulation via fused CUDA RawKernel — one launch for all combos × all bars."""
    N = params_batch.shape[0]
    T = len(confs_cpu)

    # Upload pre-computed per-bar arrays to GPU
    g_confs = cp.asarray(confs_cpu, dtype=cp.float32)
    g_dirs = cp.asarray(dirs_cpu, dtype=cp.float32)
    g_closes = cp.asarray(closes_cpu, dtype=cp.float32)
    g_atrs = cp.asarray(atrs_cpu, dtype=cp.float32)
    g_highs = cp.asarray(highs_cpu, dtype=cp.float32)
    g_lows = cp.asarray(lows_cpu, dtype=cp.float32)
    g_lev_m = cp.asarray(bar_lev_m, dtype=cp.float32)
    g_risk_m = cp.asarray(bar_risk_m, dtype=cp.float32)
    g_stop_m = cp.asarray(bar_stop_m, dtype=cp.float32)
    g_rr_m = cp.asarray(bar_rr_m, dtype=cp.float32)
    g_hold_m = cp.asarray(bar_hold_m, dtype=cp.float32)
    g_conf_mult = cp.asarray(bar_conf_mult, dtype=cp.float32)
    g_dd_thresh = cp.asarray(dd_thresh_vals, dtype=cp.float32)
    g_dd_mult = cp.asarray(dd_thresholds, dtype=cp.float32)
    g_dd_minc = cp.asarray(dd_min_confs, dtype=cp.float32)

    g_params = cp.asarray(params_batch, dtype=cp.float32)
    g_results = cp.empty((N, 7), dtype=cp.float32)

    n_dd = len(dd_thresh_vals)
    kernel = _get_simulate_kernel()
    block_size = 256
    grid_size = (N + block_size - 1) // block_size

    kernel((grid_size,), (block_size,), (
        g_params, g_confs, g_dirs, g_closes, g_atrs, g_highs, g_lows,
        g_lev_m, g_risk_m, g_stop_m, g_rr_m, g_hold_m, g_conf_mult,
        g_dd_thresh, g_dd_mult, g_dd_minc,
        np.int32(T), np.int32(N), np.int32(n_dd),
        np.float32(starting_balance), np.float32(fee_rate), np.float32(total_slippage),
        g_results
    ))

    return g_results


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
    Per-bar data is pre-computed on CPU to eliminate GPU→CPU sync overhead.
    All .any() guards removed to avoid GPU sync stalls (masked ops are cheap).
    """
    N = params_batch.shape[0]
    T = len(confs)

    # ---- PRE-COMPUTE: copy per-bar GPU data to CPU numpy (eliminates per-bar GPU→CPU syncs) ----
    confs_cpu  = cp.asnumpy(confs)
    dirs_cpu   = cp.asnumpy(dirs)
    closes_cpu = cp.asnumpy(closes)
    atrs_cpu   = cp.asnumpy(atrs)
    highs_cpu  = cp.asnumpy(highs)
    lows_cpu   = cp.asnumpy(lows)
    regime_cpu = cp.asnumpy(regime).astype(np.int32)

    # Clamp per-bar values (vectorized, CPU)
    closes_cpu = np.maximum(closes_cpu, 1.0).astype(np.float32)
    highs_cpu  = np.maximum(highs_cpu, closes_cpu).astype(np.float32)
    lows_cpu   = np.maximum(lows_cpu, 1e-4).astype(np.float32)
    atrs_cpu   = np.where(atrs_cpu > 0, atrs_cpu, closes_cpu * 0.01).astype(np.float32)

    # Pre-compute regime multiplier arrays (T,) — eliminates per-bar regime lookup
    REGIME_LEV_MULT_np  = np.array([CONFIG_REGIME_MULT[i]['lev']  for i in range(4)], dtype=np.float32)
    REGIME_RISK_MULT_np = np.array([CONFIG_REGIME_MULT[i]['risk'] for i in range(4)], dtype=np.float32)
    REGIME_STOP_MULT_np = np.array([CONFIG_REGIME_MULT[i]['stop'] for i in range(4)], dtype=np.float32)
    REGIME_RR_MULT_np   = np.array([CONFIG_REGIME_MULT[i]['rr']   for i in range(4)], dtype=np.float32)
    REGIME_HOLD_MULT_np = np.array([CONFIG_REGIME_MULT[i]['hold'] for i in range(4)], dtype=np.float32)

    bar_lev_m  = REGIME_LEV_MULT_np[regime_cpu]   # (T,) float32
    bar_risk_m = REGIME_RISK_MULT_np[regime_cpu]
    bar_stop_m = REGIME_STOP_MULT_np[regime_cpu]
    bar_rr_m   = REGIME_RR_MULT_np[regime_cpu]
    bar_hold_m = REGIME_HOLD_MULT_np[regime_cpu]

    # Pre-compute confidence tier multiplier for each bar (T,)
    # CONFIDENCE_SIZE_TIERS is sorted descending; first match wins.
    # Vectorize: iterate ascending (reverse), last overwrite = first match.
    bar_conf_mult = np.full(T, 0.25, dtype=np.float32)  # default: below all tiers
    for _ct, _cm in reversed(CONFIDENCE_SIZE_TIERS):
        bar_conf_mult[confs_cpu >= _ct] = _cm

    # Pre-compute drawdown protocol arrays for vectorized application
    dd_protocol_levels = sorted(DRAWDOWN_PROTOCOL.keys(), reverse=True)
    dd_thresholds = np.array([DRAWDOWN_PROTOCOL[k]['risk_multiplier'] for k in dd_protocol_levels], dtype=np.float32)
    dd_thresh_vals = np.array(dd_protocol_levels, dtype=np.float32)
    dd_min_confs = np.array([DRAWDOWN_PROTOCOL[k].get('min_confidence') or 0.0 for k in dd_protocol_levels], dtype=np.float32)

    # FIX #15: Try RawKernel path first — single kernel launch instead of ~30×T launches
    if xp_lib is cp:
        try:
            params_np = cp.asnumpy(params_batch) if isinstance(params_batch, cp.ndarray) else np.asarray(params_batch, dtype=np.float32)
            return simulate_batch_rawkernel(
                params_np, confs_cpu, dirs_cpu, closes_cpu, atrs_cpu,
                highs_cpu, lows_cpu, bar_lev_m, bar_risk_m, bar_stop_m,
                bar_rr_m, bar_hold_m, bar_conf_mult,
                dd_thresh_vals, dd_thresholds, dd_min_confs,
                STARTING_BALANCE, TOTAL_COST_PER_TRADE, 2.0 * slippage)
        except Exception as _rk_err:
            print(f"  [WARN] RawKernel failed ({_rk_err}), falling back to Python loop")

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

    # Pre-compute partial TP scale (constant per combo, not per bar)
    partial_pct_val = xp_lib.where(
        is_partial & (exit_type > 0),
        exit_type / 100.0,
        xp_lib.ones(N, dtype=xp_lib.float32)
    )

    # Drawdown protocol state: current risk multiplier and min_confidence override
    dd_risk_mult = xp_lib.ones(N, dtype=xp_lib.float32)
    dd_min_conf  = xp_lib.zeros(N, dtype=xp_lib.float32)  # 0 = no override

    fee_rate = TOTAL_COST_PER_TRADE
    total_slippage = 2.0 * slippage  # applied to entry AND exit

    # Pre-allocate constant GPU arrays used inside loop
    _ones_N  = xp_lib.ones(N, dtype=xp_lib.float32)
    _zeros_N = xp_lib.zeros(N, dtype=xp_lib.float32)
    _true_N  = xp_lib.ones(N, dtype=xp_lib.bool_)
    _false_N = xp_lib.zeros(N, dtype=xp_lib.bool_)

    for t in range(T):
        # Per-bar scalars (pre-computed on CPU — zero GPU sync)
        c_val  = float(confs_cpu[t])
        dirs_t = float(dirs_cpu[t])
        p_val  = float(closes_cpu[t])
        h_val  = float(highs_cpu[t])
        l_val  = float(lows_cpu[t])
        a_val  = float(atrs_cpu[t])
        lev_m  = float(bar_lev_m[t])
        risk_m = float(bar_risk_m[t])
        stop_m = float(bar_stop_m[t])
        rr_m   = float(bar_rr_m[t])
        hold_m = float(bar_hold_m[t])
        _conf_mult_t = float(bar_conf_mult[t])

        # --- Exit logic (no .any() guards — masked ops are cheap, GPU syncs are not) ---
        active = in_trade & alive
        trade_bars += active.astype(xp_lib.int32)

        # 8C.3: Scale-in on bar 2 if profitable and not yet scaled in
        bar2 = active & (trade_bars == 2) & (~scaled_in)
        profitable_long  = bar2 & (trade_dir == 1) & (p_val > entry_pr)
        profitable_short = bar2 & (trade_dir == -1) & (p_val < entry_pr)
        scale_in_mask = profitable_long | profitable_short
        entry_conf_mult = xp_lib.where(scale_in_mask, entry_conf_mult * 1.5, entry_conf_mult)
        new_entry = entry_pr * (2.0 / 3.0) + p_val * (1.0 / 3.0)
        entry_pr = xp_lib.where(scale_in_mask, new_entry, entry_pr)
        old_stop_dist = xp_lib.abs(entry_pr - stop_pr)
        new_stop_dist = old_stop_dist * 1.2
        stop_pr = xp_lib.where(scale_in_mask & (trade_dir == 1),
                               entry_pr - new_stop_dist, stop_pr)
        stop_pr = xp_lib.where(scale_in_mask & (trade_dir == -1),
                               entry_pr + new_stop_dist, stop_pr)
        scaled_in = xp_lib.where(scale_in_mask, True, scaled_in)

        # Update best price for trailing (longs track highs, shorts track lows)
        long_active  = active & (trade_dir == 1)
        short_active = active & (trade_dir == -1)
        best_pr = xp_lib.where(long_active, xp_lib.maximum(best_pr, h_val), best_pr)
        best_pr = xp_lib.where(short_active, xp_lib.minimum(best_pr, l_val), best_pr)

        # Trailing stop update: after reaching 1R profit, trail at trail_mult * ATR
        trail_active = active & is_trail
        one_r_long   = entry_pr + stop_mult * a_val
        one_r_short  = entry_pr - stop_mult * a_val
        past_1r_long  = trail_active & (trade_dir == 1) & (p_val >= one_r_long)
        past_1r_short = trail_active & (trade_dir == -1) & (p_val <= one_r_short)
        new_trail_long  = best_pr - trail_mult * a_val
        new_trail_short = best_pr + trail_mult * a_val
        stop_pr = xp_lib.where(past_1r_long, xp_lib.maximum(stop_pr, new_trail_long), stop_pr)
        stop_pr = xp_lib.where(past_1r_short, xp_lib.minimum(stop_pr, new_trail_short), stop_pr)

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

        # PnL calc — use barrier price for SL/TP exits, close for time exits
        # Priority: SL > TP > hold (conservative — if both barriers hit intrabar, assume SL)
        exit_price = xp_lib.where(sl_hit, stop_pr,
                     xp_lib.where(tp_hit, tp_pr, p_val))
        price_chg = (exit_price - entry_pr) / xp_lib.maximum(entry_pr, 1e-8) * trade_dir.astype(xp_lib.float32)
        eff_lev = lev * lev_m
        gross_pnl = price_chg * eff_lev
        fee_cost  = (fee_rate + total_slippage) * eff_lev
        net_pnl   = gross_pnl - fee_cost

        # Partial TP scale (pre-computed partial_pct_val)
        scale = xp_lib.where(tp_hit & is_partial & (exit_type > 0), partial_pct_val, _ones_N)

        eff_risk = risk_pct * risk_m * entry_conf_mult * dd_risk_mult
        pnl_dollar = balance * eff_risk * net_pnl * scale

        # Sortino accumulators (masked by exiting)
        new_bal = xp_lib.maximum(balance + pnl_dollar, 1.0)
        log_ret = xp_lib.log(new_bal / xp_lib.maximum(balance, 1.0))
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
        in_trade        = xp_lib.where(exiting, False, in_trade)
        trade_bars      = xp_lib.where(exiting, 0, trade_bars)
        entry_conf_mult = xp_lib.where(exiting, 1.0, entry_conf_mult)
        scaled_in       = xp_lib.where(exiting, False, scaled_in)

        # Check alive
        alive = alive & (balance > 0)

        # --- Entry logic for those NOT in trade ---
        can_enter = (~in_trade) & alive
        eff_conf_th = xp_lib.maximum(conf_th, dd_min_conf)
        go_long  = can_enter & (dirs_t == 1.0) & (c_val > eff_conf_th) & (dd_risk_mult > 0)
        go_short = can_enter & (dirs_t == -1.0) & (c_val > eff_conf_th) & (dd_risk_mult > 0)
        entering = go_long | go_short

        new_dir = xp_lib.where(go_long, 1, xp_lib.where(go_short, -1, 0))
        entry_conf_mult = xp_lib.where(entering, _conf_mult_t, entry_conf_mult)
        entry_pr = xp_lib.where(entering, p_val, entry_pr)

        # Stop loss (regime-adjusted)
        sl_dist = stop_mult * stop_m * a_val
        stop_pr = xp_lib.where(entering & (new_dir == 1), p_val - sl_dist,
                  xp_lib.where(entering & (new_dir == -1), p_val + sl_dist, stop_pr))

        # Take profit (regime-adjusted)
        tp_dist = stop_mult * stop_m * a_val * rr * rr_m
        tp_pr = xp_lib.where(entering & (new_dir == 1), p_val + tp_dist,
                xp_lib.where(entering & (new_dir == -1), p_val - tp_dist, tp_pr))

        # Best price init (for trailing)
        best_pr    = xp_lib.where(entering, p_val, best_pr)
        trade_dir  = xp_lib.where(entering, new_dir, trade_dir)
        in_trade   = xp_lib.where(entering, True, in_trade)
        trade_bars = xp_lib.where(entering, 0, trade_bars)

        # Drawdown tracking (every bar)
        peak   = xp_lib.maximum(peak, balance)
        dd     = xp_lib.where(peak > 0, (peak - balance) / peak, 0.0)
        max_dd = xp_lib.maximum(max_dd, dd)

        # Drawdown protocol: vectorized over protocol levels (no Python loop per combo)
        dd_risk_mult = _ones_N.copy()
        dd_min_conf  = _zeros_N.copy()
        for _di in range(len(dd_thresh_vals)):
            dd_breach = (dd >= float(dd_thresh_vals[_di]))
            dd_risk_mult = xp_lib.where(dd_breach, float(dd_thresholds[_di]), dd_risk_mult)
            if dd_min_confs[_di] > 0:
                dd_min_conf = xp_lib.where(dd_breach, float(dd_min_confs[_di]), dd_min_conf)

    # --- Compute final metrics ---
    total_tr = (wins + losses).astype(xp_lib.float32)
    win_rate = xp_lib.where(total_tr > 0, wins.astype(xp_lib.float32) / total_tr, 0.0)
    roi_pct  = (balance - STARTING_BALANCE) / STARTING_BALANCE * 100.0
    max_dd_pct = max_dd * 100.0

    # Sortino ratio
    mean_log = xp_lib.where(total_trades > 0,
                            sum_log_ret / total_trades.astype(xp_lib.float64), 0.0)
    downside_var = xp_lib.where(count_neg > 0,
                                sum_neg_sq / count_neg.astype(xp_lib.float64), 0.0)
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
# Sobol quasi-random candidate generator
# ---------------------------------------------------------------------------
def generate_sobol_candidates(grid, n_candidates, seed=OPTUNA_SEED):
    """
    Generate n_candidates parameter combos using scrambled Sobol sequences.
    Maps 7D unit hypercube to parameter ranges. Categorical exit_type is
    handled by quantizing the continuous Sobol dimension to discrete values.

    Returns: (n_candidates, 7) float32 array — [lev, risk, stop_atr, rr, hold, exit_type, conf]
    """
    # 7 dimensions: lev, risk, stop_atr, rr, hold, exit_type, conf
    sampler = qmc.Sobol(d=7, scramble=True, seed=seed)
    # Power-of-2 draw for balanced Sobol
    m = int(np.ceil(np.log2(max(n_candidates, 1))))
    n_draw = 2 ** m
    samples = sampler.random(n_draw)  # (n_draw, 7) in [0, 1)

    # Trim to requested count (if n_candidates wasn't already power-of-2)
    samples = samples[:n_candidates]

    # Map each dimension to parameter range
    lev_min, lev_max = int(grid['lev'][0]), int(grid['lev'][-1])
    risk_min, risk_max = float(grid['risk'][0]), float(grid['risk'][-1])
    stop_min, stop_max = float(grid['stop_atr'][0]), float(grid['stop_atr'][-1])
    rr_min, rr_max = float(grid['rr'][0]), float(grid['rr'][-1])
    hold_min, hold_max = int(grid['hold'][0]), int(grid['hold'][-1])
    exit_types = np.array(grid['exit_type'], dtype=np.float32)
    conf_min, conf_max = float(grid['conf'][0]), float(grid['conf'][-1])

    params = np.empty((n_candidates, 7), dtype=np.float32)
    params[:, 0] = np.round(samples[:, 0] * (lev_max - lev_min) + lev_min)    # lev (int)
    params[:, 1] = samples[:, 1] * (risk_max - risk_min) + risk_min            # risk
    params[:, 2] = samples[:, 2] * (stop_max - stop_min) + stop_min            # stop_atr
    params[:, 3] = samples[:, 3] * (rr_max - rr_min) + rr_min                  # rr
    params[:, 4] = np.round(samples[:, 4] * (hold_max - hold_min) + hold_min)  # hold (int)
    # exit_type: quantize continuous [0,1) to discrete categorical
    exit_idx = np.clip((samples[:, 5] * len(exit_types)).astype(int), 0, len(exit_types) - 1)
    params[:, 5] = exit_types[exit_idx]                                         # exit_type
    params[:, 6] = samples[:, 6] * (conf_max - conf_min) + conf_min            # conf

    return params


# ---------------------------------------------------------------------------
# Vectorized profile extraction (replaces Python-list scan)
# ---------------------------------------------------------------------------
def _extract_profiles_vectorized(all_params, all_results):
    """
    Extract 4 profiles (dd10_best, dd10_sortino, dd15_best, dd15_sortino)
    from (N, 7) params and (N, 7) results arrays using numpy indexing.
    Replaces O(N) Python-list iteration with vectorized masking.
    """
    # results columns: [balance, max_dd_pct, win_rate, trade_count, roi_pct, sortino, total_trades]
    bals     = all_results[:, 0]
    dd_pcts  = all_results[:, 1]
    trades   = all_results[:, 3]
    sortinos = all_results[:, 5]

    # Base mask: >= 10 trades
    valid = trades >= 10

    best = {
        'dd10_best':    {'score': -np.inf, 'params': None, 'metrics': None},
        'dd10_sortino': {'score': -np.inf, 'params': None, 'metrics': None},
        'dd15_best':    {'score': -np.inf, 'params': None, 'metrics': None},
        'dd15_sortino': {'score': -np.inf, 'params': None, 'metrics': None},
    }

    for dd_limit, prefix in [(10.0, 'dd10'), (15.0, 'dd15')]:
        mask = valid & (dd_pcts <= dd_limit)
        if not mask.any():
            continue

        # Best balance
        masked_bals = np.where(mask, bals, -np.inf)
        idx_best = np.argmax(masked_bals)
        if masked_bals[idx_best] > -np.inf:
            best[f'{prefix}_best'] = {
                'score': float(bals[idx_best]),
                'params': all_params[idx_best].copy(),
                'metrics': all_results[idx_best].copy(),
            }

        # Best sortino (must be finite)
        finite_sortino = mask & np.isfinite(sortinos)
        if finite_sortino.any():
            masked_sortinos = np.where(finite_sortino, sortinos, -np.inf)
            idx_sort = np.argmax(masked_sortinos)
            if masked_sortinos[idx_sort] > -np.inf:
                best[f'{prefix}_sortino'] = {
                    'score': float(sortinos[idx_sort]),
                    'params': all_params[idx_sort].copy(),
                    'metrics': all_results[idx_sort].copy(),
                }

    return best


# ---------------------------------------------------------------------------
# Sobol sweep + Bayesian refinement optimizer for one TF
# ---------------------------------------------------------------------------
def run_sobol_search(tf_name, confs, dirs, closes, atrs, highs, lows, n_bars,
                     n_candidates=SOBOL_N_CANDIDATES, top_k=SOBOL_TOP_K,
                     phase2_trials=SOBOL_PHASE2_TRIALS):
    """
    Two-phase optimizer:
      Phase 1: Sobol quasi-random sweep (n_candidates) — GPU-batched, online Sortino
      Phase 2: Bayesian refinement (Optuna TPE) seeded from top-K Sobol regions

    Returns dict of best configs per profile (dd10_best, dd10_sortino, dd15_best, dd15_sortino).
    """
    grid = TF_GRIDS[tf_name]
    print(f"\n{'='*70}")
    print(f"  {tf_name.upper()} SOBOL + BAYESIAN OPTIMIZER")
    print(f"  Phase 1: {n_candidates:,} Sobol candidates")
    print(f"  Phase 2: {phase2_trials} Bayesian trials (seeded from top {top_k})")
    print(f"  Test window: {n_bars} bars")
    print(f"{'='*70}")

    tf_slippage = TF_SLIPPAGE.get(tf_name, 0.0)
    print(f"  Slippage: {tf_slippage*100:.4f}% per side")

    # Compute regime
    regime = detect_regime(closes)
    regime_counts = {0: np.sum(regime==0), 1: np.sum(regime==1), 2: np.sum(regime==2), 3: np.sum(regime==3)}
    print(f"  Regime distribution: bull={regime_counts[0]} bear={regime_counts[1]} "
          f"sideways={regime_counts[2]} crash={regime_counts[3]}")

    # Transfer market data to GPU ONCE (CuPy required — no CPU fallback)
    g_confs  = cp.asarray(confs)
    g_dirs   = cp.asarray(dirs)
    g_closes = cp.asarray(closes)
    g_atrs   = cp.asarray(atrs)
    g_highs  = cp.asarray(highs)
    g_lows   = cp.asarray(lows)
    g_regime = cp.asarray(regime)

    # ===== PHASE 1: Sobol sweep =====
    print(f"\n  --- Phase 1: Sobol Sweep ({n_candidates:,} candidates) ---")
    t_phase1 = time.time()

    all_params = generate_sobol_candidates(grid, n_candidates)

    # Auto-tune batch size based on bar count
    # More bars = longer per-batch sim time, so keep batch size manageable
    # Inversely scale: 1w (819 bars) gets 500K, 15m (227K bars) gets ~2.2K
    adaptive_batch = int(GPU_BATCH_REFERENCE_SIZE * GPU_BATCH_REFERENCE_BARS / max(n_bars, 1))
    adaptive_batch = max(adaptive_batch, 1024)  # floor at 1K combos
    batch_size = min(adaptive_batch, n_candidates)
    # Round down to power-of-2 for GPU efficiency
    batch_size = 2 ** int(np.floor(np.log2(max(batch_size, 1))))
    print(f"  GPU batch size: {batch_size:,}")

    # Accumulate results
    all_results = np.empty((n_candidates, 7), dtype=np.float32)

    n_batches = (n_candidates + batch_size - 1) // batch_size
    for bi in range(n_batches):
        start = bi * batch_size
        end = min(start + batch_size, n_candidates)
        batch_params = all_params[start:end]

        params_gpu = cp.asarray(batch_params)
        results_gpu = simulate_batch(params_gpu, g_confs, g_dirs, g_closes, g_atrs,
                                     g_highs, g_lows, g_regime, cp,
                                     slippage=tf_slippage)
        all_results[start:end] = cp.asnumpy(results_gpu)
        del params_gpu, results_gpu

        if (bi + 1) % max(1, n_batches // 10) == 0 or bi == n_batches - 1:
            pct = (bi + 1) / n_batches * 100
            elapsed_s = time.time() - t_phase1
            rate = (end) / max(elapsed_s, 0.01)
            print(f"    Batch {bi+1}/{n_batches} ({pct:.0f}%) | "
                  f"{rate:,.0f} candidates/s | {elapsed_s:.1f}s", flush=True)

    phase1_time = time.time() - t_phase1
    print(f"  Phase 1 complete: {n_candidates:,} candidates in {phase1_time:.1f}s "
          f"({n_candidates / max(phase1_time, 0.01):,.0f} candidates/s)")

    # Extract sortino column (index 5) for ranking
    sortinos = all_results[:, 5]  # (N,)
    trades = all_results[:, 3]    # (N,)
    dd_pcts = all_results[:, 1]   # (N,)

    # Mask out invalid: <10 trades or NaN sortino
    valid_mask = (trades >= 10) & np.isfinite(sortinos)
    n_valid = np.sum(valid_mask)
    print(f"  Valid candidates (>=10 trades, finite sortino): {n_valid:,} / {n_candidates:,}")

    if n_valid == 0:
        print(f"  WARNING: No valid candidates found in Phase 1!")
        del g_confs, g_dirs, g_closes, g_atrs, g_highs, g_lows, g_regime
        cp.get_default_memory_pool().free_all_blocks()
        return {
            'dd10_best': {'score': -np.inf, 'params': None, 'metrics': None},
            'dd10_sortino': {'score': -np.inf, 'params': None, 'metrics': None},
            'dd15_best': {'score': -np.inf, 'params': None, 'metrics': None},
            'dd15_sortino': {'score': -np.inf, 'params': None, 'metrics': None},
        }

    # Use argpartition (O(N) vs O(N log N) for full sort) to find top-K by sortino
    # Apply DD penalty to sortino for ranking: >20% DD = 0.1x, >15% = 0.5x
    penalized_sortino = np.where(valid_mask, sortinos, -np.inf)
    penalized_sortino = np.where(dd_pcts > 20, penalized_sortino * 0.1, penalized_sortino)
    penalized_sortino = np.where((dd_pcts > 15) & (dd_pcts <= 20), penalized_sortino * 0.5, penalized_sortino)

    effective_top_k = min(top_k, n_valid)
    # argpartition: indices of top-K largest values
    top_k_indices = np.argpartition(penalized_sortino, -effective_top_k)[-effective_top_k:]
    # Sort the top-K for display
    top_k_indices = top_k_indices[np.argsort(penalized_sortino[top_k_indices])[::-1]]

    print(f"\n  Top {effective_top_k} candidates (by penalized Sortino):")
    for rank, idx in enumerate(top_k_indices[:5]):
        p = all_params[idx]
        m = all_results[idx]
        print(f"    #{rank+1}: sortino={m[5]:+.3f} dd={m[1]:.1f}% roi={m[4]:+.1f}% "
              f"trades={int(m[3])} lev={int(p[0])} risk={p[1]:.2f} rr={p[3]:.2f}")

    # ===== PHASE 2: Bayesian refinement around top-K regions =====
    print(f"\n  --- Phase 2: Bayesian Refinement ({phase2_trials} trials, seeded from top {effective_top_k}) ---")
    t_phase2 = time.time()

    # Extract parameter bounds from top-K to define refined search space
    top_params = all_params[top_k_indices]  # (top_k, 7)

    # Compute per-dimension min/max with some margin (±10% of range)
    grid_lev_min, grid_lev_max = int(grid['lev'][0]), int(grid['lev'][-1])
    grid_risk_min, grid_risk_max = float(grid['risk'][0]), float(grid['risk'][-1])
    grid_stop_min, grid_stop_max = float(grid['stop_atr'][0]), float(grid['stop_atr'][-1])
    grid_rr_min, grid_rr_max = float(grid['rr'][0]), float(grid['rr'][-1])
    grid_hold_min, grid_hold_max = int(grid['hold'][0]), int(grid['hold'][-1])
    exit_types = grid['exit_type']
    grid_conf_min, grid_conf_max = float(grid['conf'][0]), float(grid['conf'][-1])

    # Refined ranges: use top-K spread with 20% margin, clamped to original grid bounds
    def refined_range(col_idx, orig_min, orig_max):
        vals = top_params[:, col_idx]
        lo, hi = float(vals.min()), float(vals.max())
        margin = 0.2 * (hi - lo + 1e-8)
        return max(orig_min, lo - margin), min(orig_max, hi + margin)

    ref_lev_min, ref_lev_max = refined_range(0, grid_lev_min, grid_lev_max)
    ref_risk_min, ref_risk_max = refined_range(1, grid_risk_min, grid_risk_max)
    ref_stop_min, ref_stop_max = refined_range(2, grid_stop_min, grid_stop_max)
    ref_rr_min, ref_rr_max = refined_range(3, grid_rr_min, grid_rr_max)
    ref_hold_min, ref_hold_max = refined_range(4, grid_hold_min, grid_hold_max)
    ref_conf_min, ref_conf_max = refined_range(6, grid_conf_min, grid_conf_max)

    # Collect unique exit_types from top-K
    top_exit_types = sorted(set(int(round(e)) for e in top_params[:, 5]))
    if not top_exit_types:
        top_exit_types = exit_types

    print(f"  Refined ranges: lev=[{ref_lev_min:.0f},{ref_lev_max:.0f}] "
          f"risk=[{ref_risk_min:.2f},{ref_risk_max:.2f}] "
          f"stop=[{ref_stop_min:.2f},{ref_stop_max:.2f}] "
          f"rr=[{ref_rr_min:.2f},{ref_rr_max:.2f}] "
          f"hold=[{ref_hold_min:.0f},{ref_hold_max:.0f}] "
          f"exit={top_exit_types} "
          f"conf=[{ref_conf_min:.3f},{ref_conf_max:.3f}]")

    # Phase 2 trial results (Phase 1 already in all_params/all_results numpy arrays)
    p2_params_list = []
    p2_results_list = []

    trial_count = [0]
    t_start_p2 = time.time()

    def objective_phase2(trial):
        lev = trial.suggest_int('lev', int(ref_lev_min), int(ref_lev_max))
        risk = trial.suggest_float('risk', ref_risk_min, ref_risk_max)
        stop_atr = trial.suggest_float('stop_atr', ref_stop_min, ref_stop_max)
        rr = trial.suggest_float('rr', ref_rr_min, ref_rr_max)
        hold = trial.suggest_int('hold', int(ref_hold_min), int(ref_hold_max))
        exit_type = trial.suggest_categorical('exit_type', top_exit_types)
        conf = trial.suggest_float('conf', ref_conf_min, ref_conf_max)

        params_np = np.array([[lev, risk, stop_atr, rr, hold, exit_type, conf]], dtype=np.float32)
        params_gpu = cp.asarray(params_np)
        results = simulate_batch(params_gpu, g_confs, g_dirs, g_closes, g_atrs,
                                 g_highs, g_lows, g_regime, cp, slippage=tf_slippage)
        results_np = cp.asnumpy(results)
        del params_gpu, results

        result = results_np[0]
        bal = float(result[0])
        max_dd = float(result[1])
        n_trades = float(result[3])
        roi = float(result[4])
        sortino_val = float(result[5])

        p2_params_list.append(params_np[0].copy())
        p2_results_list.append(results_np[0].copy())

        trial.set_user_attr('balance', bal)
        trial.set_user_attr('max_dd', max_dd)
        trial.set_user_attr('trades', n_trades)
        trial.set_user_attr('roi', roi)
        trial.set_user_attr('sortino', sortino_val)

        trial_count[0] += 1
        if trial_count[0] % 10 == 0 or trial_count[0] == phase2_trials:
            elapsed_s = time.time() - t_start_p2
            rate = trial_count[0] / max(elapsed_s, 0.01)
            s_display = sortino_val if not np.isnan(sortino_val) else -999
            print(f"  P2 Trial {trial_count[0]:>4d}/{phase2_trials} | "
                  f"sortino={s_display:+.3f} dd={max_dd:.1f}% roi={roi:+.1f}% trades={int(n_trades)} | "
                  f"{rate:.1f} trials/s", flush=True)

        if n_trades < 10:
            return -999.0
        if np.isnan(sortino_val):
            return -999.0
        if max_dd > 20:
            return sortino_val * 0.1
        elif max_dd > 15:
            return sortino_val * 0.5
        return sortino_val

    # Seed Optuna with top-K Sobol results via enqueue_trial
    storage = f"sqlite:///{DB_DIR}/sobol_optimizer_{tf_name}.db"
    study = optuna.create_study(
        study_name=f"sobol_optimizer_{tf_name}",
        storage=storage,
        load_if_exists=True,
        direction='maximize',
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=min(OPTUNA_PHASE1_N_STARTUP, phase2_trials // 4),
            multivariate=True,
            seed=OPTUNA_SEED,
        ),
    )

    # Enqueue top-K as seed trials so TPE learns from the best Sobol regions
    for idx in top_k_indices[:min(effective_top_k, 50)]:  # cap seeded trials
        p = all_params[idx]
        study.enqueue_trial({
            'lev': int(round(p[0])),
            'risk': float(p[1]),
            'stop_atr': float(p[2]),
            'rr': float(p[3]),
            'hold': int(round(p[4])),
            'exit_type': int(round(p[5])),
            'conf': float(p[6]),
        })

    # n_jobs=1: GPU sims with N=1 per trial — threads fight over single CUDA context
    study.optimize(objective_phase2, n_trials=phase2_trials, n_jobs=1, show_progress_bar=False)

    phase2_time = time.time() - t_phase2
    total_time = time.time() - t_phase1
    print(f"\n  Phase 2 complete: {phase2_trials} trials in {phase2_time:.1f}s")
    print(f"  Total: {n_candidates + phase2_trials:,} evaluations in {total_time:.1f}s")
    if study.best_trial:
        print(f"  Best Phase 2 sortino (penalized): {study.best_value:.4f}")

    # --- Extract 4 profiles from ALL results (Phase 1 + Phase 2) using numpy vectorized ---
    # Merge Phase 1 + Phase 2 into single arrays
    if p2_params_list:
        combined_params = np.vstack([all_params, np.array(p2_params_list, dtype=np.float32)])
        combined_results = np.vstack([all_results, np.array(p2_results_list, dtype=np.float32)])
    else:
        combined_params = all_params
        combined_results = all_results

    best = _extract_profiles_vectorized(combined_params, combined_results)

    # Free GPU memory
    del g_confs, g_dirs, g_closes, g_atrs, g_highs, g_lows, g_regime
    cp.get_default_memory_pool().free_all_blocks()

    return best


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

    # Transfer market data to GPU ONCE (CuPy required — no CPU fallback)
    g_confs  = cp.asarray(confs)
    g_dirs   = cp.asarray(dirs)
    g_closes = cp.asarray(closes)
    g_atrs   = cp.asarray(atrs)
    g_highs  = cp.asarray(highs)
    g_lows   = cp.asarray(lows)
    g_regime = cp.asarray(regime)

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

    # Track ALL trial results for vectorized profile extraction
    _trial_params_list = []
    _trial_results_list = []
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
        params_gpu = cp.asarray(params_np)
        results = simulate_batch(params_gpu, g_confs, g_dirs, g_closes, g_atrs,
                                 g_highs, g_lows, g_regime, cp,
                                 slippage=tf_slippage)
        results_np = cp.asnumpy(results)
        del params_gpu, results

        # results_np: (1, 7) = [balance, max_dd_pct, win_rate, trade_count, roi_pct, sortino, total_trades]
        result = results_np[0]
        bal = float(result[0])
        max_dd = float(result[1])
        trades = float(result[3])
        roi = float(result[4])
        sortino_val = float(result[5])

        # Store for vectorized profile extraction
        _trial_params_list.append(params_np[0].copy())
        _trial_results_list.append(results_np[0].copy())

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
            n_startup_trials=min(OPTUNA_PHASE1_N_STARTUP, n_trials // 4),
            multivariate=True,
            seed=OPTUNA_SEED,
        ),
    )

    print(f"\n  Starting Optuna optimization ({n_trials} trials)...")
    # n_jobs=1: GPU sims with N=1 per trial — threads fight over single CUDA context
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)

    total_time = time.time() - t_start
    print(f"\n  Completed {n_trials} trials in {total_time:.1f}s "
          f"({n_trials / max(total_time, 0.01):.1f} trials/sec)")
    print(f"  Best trial sortino (penalized): {study.best_value:.4f}")

    # --- Extract 4 profiles using numpy vectorized indexing ---
    if _trial_params_list:
        combined_params = np.array(_trial_params_list, dtype=np.float32)
        combined_results = np.array(_trial_results_list, dtype=np.float32)
        best = _extract_profiles_vectorized(combined_params, combined_results)
    else:
        best = {
            'dd10_best':    {'score': -np.inf, 'params': None, 'metrics': None},
            'dd10_sortino': {'score': -np.inf, 'params': None, 'metrics': None},
            'dd15_best':    {'score': -np.inf, 'params': None, 'metrics': None},
            'dd15_sortino': {'score': -np.inf, 'params': None, 'metrics': None},
        }

    # Free GPU memory
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
        # Pin this worker to a specific GPU (CuPy always available)
        cp.cuda.Device(gpu_id).use()
        print(f"\n{elapsed()} Worker {tf_name.upper()} pinned to GPU {gpu_id}", flush=True)

        print(f"\n{elapsed()} Loading data for {tf_name.upper()} (worker, GPU {gpu_id})...", flush=True)
        data = load_tf_data(tf_name)
        if data is None:
            print(f"  Skipping {tf_name} — data not available", flush=True)
            return (tf_name, None, 0)

        confs, dirs, rets, closes, atrs, highs, lows, n_bars = data

        if OPTIMIZER_MODE == 'sobol':
            best = run_sobol_search(tf_name, confs, dirs, closes, atrs, highs, lows, n_bars,
                                    n_candidates=SOBOL_N_CANDIDATES, top_k=SOBOL_TOP_K,
                                    phase2_trials=SOBOL_PHASE2_TRIALS)
            config_prefix = 'sobol_configs'
            n_evals = SOBOL_N_CANDIDATES + SOBOL_PHASE2_TRIALS
        else:
            best = run_optuna_search(tf_name, confs, dirs, closes, atrs, highs, lows, n_bars,
                                     n_trials=n_trials)
            config_prefix = 'optuna_configs'
            n_evals = n_trials

        tf_config = {}
        for profile_name, profile_data in best.items():
            result = params_to_dict(profile_data['params'], profile_data['metrics'])
            if result is not None:
                tf_config[profile_name] = result

        # Save per-TF config immediately (atomic)
        if tf_config:
            per_tf_path = f"{DB_DIR}/{config_prefix}_{tf_name}.json"
            with open(per_tf_path, 'w') as f:
                json.dump({tf_name: tf_config}, f, indent=2)
            print(f"  {elapsed()} Saved per-TF config: {per_tf_path}", flush=True)

        gc.collect()
        return (tf_name, tf_config, n_evals)
    except Exception as e:
        print(f"  [FAILED] Optimizer for {tf_name}: {e}", flush=True)
        import traceback; traceback.print_exc()
        return (tf_name, None, 0)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main(n_trials=200):
    mode_label = 'SOBOL + BAYESIAN' if OPTIMIZER_MODE == 'sobol' else 'OPTUNA TPE'
    print(f"\n{'='*70}")
    print(f"  {mode_label} OPTIMIZER (LightGBM)")
    print(f"  GPU Array: CuPy (CUDA) — GPU REQUIRED")
    print(f"  GPUs:      {_N_GPUS}")
    if OPTIMIZER_MODE == 'sobol':
        print(f"  Phase 1:   {SOBOL_N_CANDIDATES:,} Sobol candidates per TF")
        print(f"  Phase 2:   {SOBOL_PHASE2_TRIALS} Bayesian refinement trials per TF")
    else:
        print(f"  Trials:    {n_trials} per TF")
    print(f"  Fee model: {TOTAL_COST_PER_TRADE*100:.2f}% round-trip")
    print(f"  Started:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    # Print grid ranges (for reference)
    print(f"\nParameter search ranges (from TF_GRIDS):")
    for tf, grid in TF_GRIDS.items():
        n = count_grid(grid)
        if OPTIMIZER_MODE == 'sobol':
            print(f"  {tf:>4s}: {n:>15,} exhaustive combos → Sobol samples {SOBOL_N_CANDIDATES:,} + {SOBOL_PHASE2_TRIALS} refinement")
        else:
            print(f"  {tf:>4s}: {n:>15,} exhaustive combos (Optuna samples {n_trials} intelligently)")

    # Process each TF — parallel across GPUs (each TF is independent)
    tf_order = [tf for tf in ['15m', '1h', '4h', '1d', '1w'] if tf in TF_GRIDS]
    all_configs = {}
    total_trials_run = 0
    opt_workers = min(_N_GPUS, len(tf_order))
    config_prefix = 'sobol_configs' if OPTIMIZER_MODE == 'sobol' else 'optuna_configs'

    if opt_workers > 1:
        # Parallel: one TF per GPU, round-robin GPU assignment
        print(f"\n  PARALLEL OPTIMIZATION: {opt_workers} workers across {_N_GPUS} GPUs")
        worker_args = [(tf_name, n_trials, i % _N_GPUS) for i, tf_name in enumerate(tf_order)]
        # CUDA is NOT fork-safe — must use spawn context to avoid corrupted GPU state
        spawn_ctx = mp.get_context('spawn')
        with ProcessPoolExecutor(max_workers=opt_workers, mp_context=spawn_ctx) as pool:
            futures = {pool.submit(_optimize_single_tf, wa): wa[0] for wa in worker_args}
            for future in as_completed(futures):
                tf_name = futures[future]
                try:
                    tf_name_r, tf_config, n_done = future.result()
                    if tf_config:
                        all_configs[tf_name_r] = tf_config
                    total_trials_run += n_done
                    print(f"  {elapsed()} Completed {tf_name_r}: {n_done:,} evaluations", flush=True)
                except Exception as e:
                    print(f"  {elapsed()} FAILED {tf_name}: {e}", flush=True)
        import gc; gc.collect()
    else:
        # Sequential fallback (single GPU)
        cp.cuda.Device(0).use()
        for tf_name in tf_order:
            print(f"\n{elapsed()} Loading data for {tf_name.upper()}...")
            data = load_tf_data(tf_name)
            if data is None:
                print(f"  Skipping {tf_name} — data not available")
                continue

            confs, dirs, rets, closes, atrs, highs, lows, n_bars = data

            if OPTIMIZER_MODE == 'sobol':
                best = run_sobol_search(tf_name, confs, dirs, closes, atrs, highs, lows, n_bars,
                                        n_candidates=SOBOL_N_CANDIDATES, top_k=SOBOL_TOP_K,
                                        phase2_trials=SOBOL_PHASE2_TRIALS)
                n_evals = SOBOL_N_CANDIDATES + SOBOL_PHASE2_TRIALS
            else:
                best = run_optuna_search(tf_name, confs, dirs, closes, atrs, highs, lows, n_bars,
                                         n_trials=n_trials)
                n_evals = n_trials

            tf_config = {}
            for profile_name, profile_data in best.items():
                result = params_to_dict(profile_data['params'], profile_data['metrics'])
                if result is not None:
                    tf_config[profile_name] = result

            if tf_config:
                all_configs[tf_name] = tf_config

                per_tf_path = f"{DB_DIR}/{config_prefix}_{tf_name}.json"
                with open(per_tf_path, 'w') as f:
                    json.dump({tf_name: tf_config}, f, indent=2)
                print(f"  Saved: {per_tf_path}")

            total_trials_run += n_evals

    # Save combined results — same JSON format as exhaustive_configs.json
    tf_suffix = '_'.join(sorted(all_configs.keys())) if all_configs else 'all'
    output_path = f"{DB_DIR}/{config_prefix}_{tf_suffix}.json" if len(TF_GRIDS) < 6 else f"{DB_DIR}/{config_prefix}.json"
    with open(output_path, 'w') as f:
        json.dump(all_configs, f, indent=2)
    print(f"\n{elapsed()} Results saved to: {output_path}")

    # ---------------------------------------------------------------------------
    # Print comprehensive results table
    # ---------------------------------------------------------------------------
    print(f"\n\n{'='*120}")
    print(f"  {mode_label} OPTIMIZATION RESULTS")
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
    print(f"  Total evaluations: {total_trials_run:,} across {len(tf_order)} TFs (mode={OPTIMIZER_MODE})")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Output: {output_path}")
    print(f"{'='*120}\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Trade Optimizer (Sobol + Bayesian / Optuna TPE) for LightGBM')
    parser.add_argument('--tf', action='append', help='Only run specific timeframes (can repeat)')
    parser.add_argument('--n-trials', type=int, default=200, help='Number of Optuna trials per TF (default: 200)')
    parser.add_argument('--mode', choices=['sobol', 'optuna'], default=None,
                        help='Optimizer mode (default: env OPTIMIZER_MODE or sobol)')
    parser.add_argument('--sobol-candidates', type=int, default=None,
                        help=f'Sobol Phase 1 candidates (default: {SOBOL_N_CANDIDATES})')
    parser.add_argument('--sobol-top-k', type=int, default=None,
                        help=f'Top-K from Phase 1 for Phase 2 seeding (default: {SOBOL_TOP_K})')
    parser.add_argument('--sobol-phase2', type=int, default=None,
                        help=f'Phase 2 Bayesian refinement trials (default: {SOBOL_PHASE2_TRIALS})')
    args = parser.parse_args()

    # CLI --mode overrides env var
    if args.mode:
        OPTIMIZER_MODE = args.mode
    if args.sobol_candidates:
        SOBOL_N_CANDIDATES = args.sobol_candidates
    if args.sobol_top_k:
        SOBOL_TOP_K = args.sobol_top_k
    if args.sobol_phase2:
        SOBOL_PHASE2_TRIALS = args.sobol_phase2

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
