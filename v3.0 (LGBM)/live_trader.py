#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
live_trader.py — Live Paper/Real Trading Engine
=================================================
Runs continuously. Every 15 minutes:
1. Fetch latest BTC candles from Bitget
2. Compute features (same code as backtest)
3. Run ML models (4H/1H/15m)
4. Execute trades via portfolio aggregator
5. Log everything to trades.db for dashboard

Usage: python live_trader.py [--mode paper|live]
"""

import sys, os, io, time, json, warnings, argparse, traceback
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import sqlite3
import lightgbm as lgb
from datetime import datetime, timedelta, timezone
import urllib.request
from knn_feature_engine import knn_features_from_ohlcv
from feature_library import build_all_features
from data_access import LiveDataLoader
from trade_journal import (log_trade_snapshot, log_price_path_bar,
                           log_trade_outcome, log_rejected_trade,
                           compute_post_trade_analysis)

# Institutional upgrades: meta-labeling + LSTM blending
try:
    from meta_labeling import predict_meta
    import pickle
    _HAS_META = True
except ImportError:
    _HAS_META = False

try:
    from lstm_sequence_model import LSTMFeatureExtractor, blend_predictions, apply_platt_calibration
    _HAS_LSTM = True
except ImportError:
    _HAS_LSTM = False

# HMM regime detection (matches ml_multi_tf.py training pipeline)
try:
    from hmmlearn.hmm import GaussianHMM
    _HAS_HMM = True
except ImportError:
    _HAS_HMM = False

DB_DIR = "C:/Users/C/Documents/Savage22 Server"
TRADES_DB = f"{DB_DIR}/trades.db"
FEE_RATE = 0.0018  # 0.12% fees + 0.06% conservative slippage = 0.18% round-trip
RISK_SCALE = 2.0

live_dal = LiveDataLoader(DB_DIR)
live_dal.initial_load()

WARMUP_BARS = {
    '5m': 600, '15m': 400, '1h': 300,
    '4h': 200, '1d': 100, '1w': 50,
}

# ============================================================
# INIT TRADES DATABASE
# ============================================================
def init_trades_db():
    conn = sqlite3.connect(TRADES_DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS account (
        id INTEGER PRIMARY KEY, mode TEXT, balance REAL, peak_balance REAL,
        total_trades INTEGER, wins INTEGER, losses INTEGER,
        max_dd REAL, updated_at TEXT
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tf TEXT, direction TEXT, confidence REAL,
        entry_price REAL, entry_time TEXT,
        exit_price REAL, exit_time TEXT,
        stop_price REAL, tp_price REAL,
        pnl REAL, pnl_pct REAL,
        bars_held INTEGER, exit_reason TEXT,
        regime TEXT, leverage REAL, risk_pct REAL,
        features_json TEXT, status TEXT DEFAULT 'open',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    # Prevent duplicate trades: same TF + same entry minute
    try:
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_trades_dedup ON trades(tf, substr(entry_time, 1, 16)) WHERE status='open'")
    except:
        pass  # partial index may not be supported, fallback dedup in code
    conn.execute("""CREATE TABLE IF NOT EXISTS equity_curve (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT, balance REAL, dd_pct REAL
    )""")

    # Init account if empty
    existing = conn.execute("SELECT COUNT(*) FROM account").fetchone()[0]
    if existing == 0:
        conn.execute("INSERT INTO account VALUES (1, 'paper', 100.0, 100.0, 0, 0, 0, 0.0, ?)",
                     (datetime.now(timezone.utc).isoformat(),))
    conn.commit()
    conn.close()

# ============================================================
# FETCH LIVE CANDLES FROM BITGET
# ============================================================
def fetch_bitget_candles(symbol='BTCUSDT', timeframe='15m', limit=300):
    """Fetch candles from Bitget API."""
    granularity_map = {
        '5m': '5m', '15m': '15m', '1h': '1H', '4h': '4H', '1d': '1D', '1w': '1W'
    }
    gran = granularity_map.get(timeframe, '15m')

    url = f"https://api.bitget.com/api/v2/mix/market/candles?productType=USDT-FUTURES&symbol={symbol}&granularity={gran}&limit={limit}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            if isinstance(data, dict) and 'data' in data:
                candles = data['data']
            elif isinstance(data, list):
                candles = data
            else:
                return pd.DataFrame()

            rows = []
            for c in candles:
                rows.append({
                    'timestamp': pd.to_datetime(int(c[0]), unit='ms', utc=True),
                    'open': float(c[1]),
                    'high': float(c[2]),
                    'low': float(c[3]),
                    'close': float(c[4]),
                    'volume': float(c[5]) if len(c) > 5 else 0,
                })
            df = pd.DataFrame(rows)
            df = df.sort_values('timestamp').reset_index(drop=True)
            return df
    except Exception as e:
        print(f"  Bitget API error: {e}")
        return pd.DataFrame()

# ============================================================
# COMPUTE FEATURES (shared library)
# ============================================================
def compute_features_live(tf_name, feat_names):
    """Compute features for current bar using shared feature library.
    Returns (feat_dict, df_features) — dict of last-bar values + full DataFrame for LSTM.
    """
    try:
        live_dal.refresh_caches()

        n_bars = WARMUP_BARS.get(tf_name, 300)
        ohlcv = live_dal.get_ohlcv_window(tf_name, n_bars)
        if ohlcv is None or len(ohlcv) < 50:
            return {}, None

        esoteric_frames = {
            'tweets': live_dal.get_tweets(),
            'news': live_dal.get_news(),
            'sports': live_dal.get_sports(),
            'onchain': live_dal.get_onchain(),
            'macro': live_dal.get_macro(),
        }

        htf_data = live_dal.get_htf_ohlcv(tf_name)
        astro_cache = live_dal.get_astro_cache()

        # Bug fix: pass space weather data to build_all_features
        space_weather_df = live_dal.get_space_weather()

        df_features = build_all_features(
            ohlcv=ohlcv,
            esoteric_frames=esoteric_frames,
            tf_name=tf_name,
            mode='live',
            htf_data=htf_data,
            astro_cache=astro_cache,
            space_weather_df=space_weather_df,
        )

        if df_features is None or len(df_features) == 0:
            return {}, None

        # FIX 6: Add GCP features for 1h timeframe (matches build_1h_features.py)
        if tf_name == '1h':
            try:
                from gcp_feature_builder import build_gcp_features
                gcp_feats = build_gcp_features(df_features)
                for col in gcp_feats.columns:
                    df_features[col] = gcp_feats[col]

                # GCP x trend crosses
                d_trend = pd.to_numeric(df_features.get('d_trend'), errors='coerce').fillna(0)
                bull = d_trend
                bear = 1 - d_trend
                for gcp_col in ['gcp_deviation_mean', 'gcp_deviation_max', 'gcp_rate_of_change', 'gcp_extreme']:
                    if gcp_col in df_features.columns:
                        sig = pd.to_numeric(df_features[gcp_col], errors='coerce').fillna(0)
                        df_features[f'tx_{gcp_col}_x_bull'] = sig * bull
                        df_features[f'tx_{gcp_col}_x_bear'] = sig * bear

                # GCP x fear/greed cross
                if 'gcp_deviation_mean' in df_features.columns:
                    gcp_dev = pd.to_numeric(df_features['gcp_deviation_mean'], errors='coerce').fillna(0)
                    fg_fear = pd.to_numeric(df_features.get('fg_extreme_fear'), errors='coerce').fillna(0)
                    fg_greed = pd.to_numeric(df_features.get('fg_extreme_greed'), errors='coerce').fillna(0)
                    df_features['gcp_x_fear'] = gcp_dev.abs() * fg_fear
                    df_features['gcp_x_greed'] = gcp_dev.abs() * fg_greed
                    # GCP x moon phase
                    moon = pd.to_numeric(df_features.get('west_moon_phase'), errors='coerce').fillna(0)
                    is_full = ((moon >= 13) & (moon <= 16)).astype(float)
                    is_new = ((moon < 2) | (moon > 27.5)).astype(float)
                    df_features['gcp_x_full_moon'] = gcp_dev.abs() * is_full
                    df_features['gcp_x_new_moon'] = gcp_dev.abs() * is_new
                    # GCP x Kp storm
                    kp = pd.to_numeric(df_features.get('sw_kp_is_storm'), errors='coerce').fillna(0)
                    df_features['gcp_x_kp_storm'] = gcp_dev.abs() * kp
            except Exception as e:
                print(f"  GCP features failed (non-fatal): {e}")

        last_row = df_features.iloc[-1]
        result = {}
        for fn in feat_names:
            val = last_row.get(fn, np.nan)
            if isinstance(val, (int, float)) and np.isinf(val):
                val = np.nan
            result[fn] = val
        return result, df_features
    except Exception as e:
        print(f"  compute_features_live error: {e}")
        import traceback
        traceback.print_exc()
        return {}, None

# ============================================================
# DETECT REGIME — 5-dimensional multiplier tables (matches v2/backtesting_audit.py)
# ============================================================
# Regime multipliers: each regime scales leverage, risk, stop, R:R, and hold independently
REGIME_MULT = {
    0: {'lev': 1.0,  'risk': 1.0,  'stop': 1.0,  'rr': 1.5,  'hold': 1.0},   # bull
    1: {'lev': 0.47, 'risk': 1.0,  'stop': 0.75, 'rr': 0.75, 'hold': 0.17},   # bear
    2: {'lev': 0.67, 'risk': 0.47, 'stop': 0.5,  'rr': 0.5,  'hold': 1.0},    # sideways
}

def detect_regime(close, sma100, sma100_slope):
    """Returns (regime_name, regime_int) for multiplier lookup."""
    if sma100 is None or sma100_slope is None:
        return 'sideways', 2
    above = close > sma100
    near = abs(close - sma100) / sma100 < 0.05
    if above and sma100_slope > 0.001:
        return 'bull', 0
    elif not above and sma100_slope < -0.001:
        return 'bear', 1
    else:
        return 'sideways', 2


# ============================================================
# HMM LIVE REGIME FEATURES (matches ml_multi_tf.py training)
# ============================================================
_hmm_model = None
_hmm_last_fit = None
_hmm_sorted_states = None

def fit_hmm_live():
    """Fit 3-state HMM on historical daily returns + volatility (no future leakage)."""
    global _hmm_model, _hmm_last_fit
    if not _HAS_HMM:
        return None

    try:
        import sqlite3
        conn = sqlite3.connect(f'{DB_DIR}/btc_prices.db')
        daily = pd.read_sql_query("""
            SELECT open_time, close FROM ohlcv
            WHERE timeframe='1d' AND symbol='BTC/USDT' ORDER BY open_time
        """, conn)
        conn.close()

        daily['close'] = pd.to_numeric(daily['close'], errors='coerce')
        daily = daily.dropna(subset=['close'])
        closes = daily['close'].values
        if len(closes) < 200:
            return None

        returns = np.log(closes[1:] / closes[:-1])
        abs_ret = np.abs(returns)
        vol10 = pd.Series(returns).rolling(10).std().values
        # Align: skip first 10 where vol10 is NaN
        valid = ~np.isnan(vol10)
        r = returns[valid]
        ar = abs_ret[valid]
        v = vol10[valid]
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
                continue

        _hmm_model = best_model
        _hmm_last_fit = datetime.now(timezone.utc)
        print(f"  HMM fitted: {len(X)} daily bars, score={best_score:.0f}")
        _compute_hmm_state_mapping()
        return best_model
    except Exception as e:
        print(f"  HMM fit failed: {e}")
        return None


def _compute_hmm_state_mapping():
    """Sort HMM states by mean return to match training (ml_multi_tf.py).
    States are labeled: sorted_states[0]=bear, [1]=neutral, [2]=bull."""
    global _hmm_sorted_states, _hmm_model
    if _hmm_model is None:
        _hmm_sorted_states = None
        return

    try:
        import sqlite3 as _sql
        conn = _sql.connect(f'{DB_DIR}/btc_prices.db')
        daily = pd.read_sql_query("""
            SELECT open_time, close FROM ohlcv
            WHERE timeframe='1d' AND symbol='BTC/USDT' ORDER BY open_time
        """, conn)
        conn.close()

        daily['close'] = pd.to_numeric(daily['close'], errors='coerce')
        daily = daily.dropna(subset=['close'])
        closes = daily['close'].values
        if len(closes) < 200:
            _hmm_sorted_states = None
            return

        returns = np.log(closes[1:] / closes[:-1])
        abs_ret = np.abs(returns)
        vol10 = pd.Series(returns).rolling(10).std().values
        valid = ~np.isnan(vol10)
        r = returns[valid]
        ar = abs_ret[valid]
        v = vol10[valid]
        X = np.column_stack([r, ar, v])

        states = _hmm_model.predict(X)
        state_means = {}
        for s in range(3):
            state_means[s] = r[states == s].mean() if (states == s).sum() > 0 else 0
        _hmm_sorted_states = sorted(state_means.keys(), key=lambda s: state_means[s])
        print(f"  HMM state mapping: bear={_hmm_sorted_states[0]}, neutral={_hmm_sorted_states[1]}, bull={_hmm_sorted_states[2]}")
    except Exception as e:
        print(f"  HMM state mapping failed: {e}")
        _hmm_sorted_states = None


def get_hmm_features(feat_dict):
    """Get HMM state probabilities for current bar. Returns dict of hmm_* features."""
    global _hmm_model, _hmm_last_fit
    if _hmm_model is None:
        return {}

    # Re-fit daily (at midnight)
    now = datetime.now(timezone.utc)
    if _hmm_last_fit and (now - _hmm_last_fit).total_seconds() > 86400:
        fit_hmm_live()

    try:
        # Use latest daily return + abs_ret + vol for prediction
        close = feat_dict.get('close', 0)
        prev_close = feat_dict.get('sma_5', close)  # approximate prev close
        if close <= 0 or prev_close <= 0:
            return {}
        ret = np.log(close / prev_close)
        abs_r = abs(ret)
        vol = feat_dict.get('volatility_short', abs_r)
        X = np.array([[ret, abs_r, vol]])

        probs = _hmm_model.predict_proba(X)[0]
        state_raw = int(np.argmax(probs))

        if _hmm_sorted_states is not None:
            sorted_states = _hmm_sorted_states
            return {
                'hmm_bull_prob': float(probs[sorted_states[2]]),
                'hmm_bear_prob': float(probs[sorted_states[0]]),
                'hmm_neutral_prob': float(probs[sorted_states[1]]),
                'hmm_state': sorted_states.index(state_raw),
            }
        else:
            # Fallback: raw indices (pre-mapping)
            return {
                'hmm_bull_prob': float(probs[0]),
                'hmm_bear_prob': float(probs[1]) if len(probs) > 1 else 0.0,
                'hmm_neutral_prob': float(probs[2]) if len(probs) > 2 else 0.0,
                'hmm_state': state_raw,
            }
    except Exception:
        return {}


# ============================================================
# BAR CLOSE DETECTION
# ============================================================
def is_bar_close(now, tf):
    """Check if a bar just closed for this timeframe."""
    if tf == '5m':
        return now.minute % 5 == 0
    elif tf == '15m':
        return now.minute % 15 == 0
    elif tf == '1h':
        return now.minute == 0
    elif tf == '4h':
        return now.minute == 0 and now.hour % 4 == 0
    elif tf == '1d':
        return now.minute == 0 and now.hour == 0  # midnight UTC
    elif tf == '1w':
        return now.minute == 0 and now.hour == 0 and now.weekday() == 0  # Monday midnight
    return False

# ============================================================
# MAIN TRADING LOOP
# ============================================================
def run_trading_loop(mode='paper'):
    print("=" * 60)
    print(f"  LIVE TRADER — Mode: {mode.upper()}")
    print(f"  Started: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    init_trades_db()

    # Capital allocation per TF
    TF_CAPITAL_ALLOC = {
        '5m': 0.10, '15m': 0.25, '1h': 0.25,
        '4h': 0.20, '1d': 0.10, '1w': 0.10,
    }

    # Per-TF pool tracking: balance, peak, drawdown
    conn_init = sqlite3.connect(TRADES_DB)
    total_balance = conn_init.execute("SELECT balance FROM account WHERE id=1").fetchone()[0]
    conn_init.close()
    tf_pools = {}
    for tf_name, alloc in TF_CAPITAL_ALLOC.items():
        pool_bal = total_balance * alloc
        tf_pools[tf_name] = {
            'balance': pool_bal,
            'peak': pool_bal,
            'dd': 0.0,
            'alloc': alloc,
            'halted': False,
        }
    MAX_PORTFOLIO_DD = 0.15  # 15% total portfolio DD halts all new entries
    MAX_TF_DD = 0.25  # 25% per-TF DD halts that TF

    # Load models
    ALL_TFS = ['5m', '15m', '1h', '4h', '1d', '1w']
    models = {}
    features_list = {}
    for tf in ALL_TFS:
        # LightGBM model: .json is the standard format
        model_path = f'{DB_DIR}/model_{tf}.json'
        # CHANGE 4: Try features_{tf}_all.json first, fall back to pruned
        feat_path_all = f'{DB_DIR}/features_{tf}_all.json'
        feat_path_pruned = f'{DB_DIR}/features_{tf}_pruned.json'
        if os.path.exists(feat_path_all):
            feat_path = feat_path_all
        elif os.path.exists(feat_path_pruned):
            feat_path = feat_path_pruned
        else:
            feat_path = None
        if os.path.exists(model_path) and feat_path:
            m = lgb.Booster(model_file=model_path)
            models[tf] = m
            with open(feat_path) as f:
                features_list[tf] = json.load(f)
            print(f"  Loaded {tf} model ({len(features_list[tf])} features) from {os.path.basename(feat_path)}")

    # Load meta-labeling models (institutional upgrade)
    meta_models = {}
    if _HAS_META:
        for tf in ALL_TFS:
            meta_path = f'{DB_DIR}/meta_model_{tf}.pkl'
            if os.path.exists(meta_path):
                with open(meta_path, 'rb') as f:
                    meta_models[tf] = pickle.load(f)
                print(f"  Loaded meta-model for {tf} (thresh={meta_models[tf].get('threshold', 0.5):.2f})")
    if not meta_models:
        print("  No meta-models found — using raw LightGBM predictions")

    # Load LSTM models (institutional upgrade)
    lstm_extractors = {}
    platt_models = {}
    if _HAS_LSTM:
        for tf in ALL_TFS:
            try:
                ext = LSTMFeatureExtractor(tf)
                if ext.model is not None:
                    lstm_extractors[tf] = ext
                    print(f"  Loaded LSTM model for {tf}")
            except Exception:
                pass
            platt_path = f'{DB_DIR}/platt_{tf}.pkl'
            if os.path.exists(platt_path):
                with open(platt_path, 'rb') as f:
                    platt_models[tf] = pickle.load(f)

    # Rolling performance tracking (live monitoring)
    trade_results = []  # list of recent trade P&Ls for Kelly re-estimation

    # Load GA configs
    with open(f'{DB_DIR}/ml_multi_tf_configs.json') as f:
        ml_configs = json.load(f)

    # Load optimizer configs (optuna_configs.json + per-TF files)
    optuna_configs = {}
    optuna_path = f'{DB_DIR}/optuna_configs.json'
    if os.path.exists(optuna_path):
        with open(optuna_path) as f:
            optuna_configs = json.load(f)
        print(f"  Loaded optuna_configs.json")
    # Also load per-TF optuna configs (e.g. optuna_configs_1h.json)
    import glob as _glob
    for per_tf_file in _glob.glob(f'{DB_DIR}/optuna_configs_*.json'):
        try:
            with open(per_tf_file) as f:
                per_tf_data = json.load(f)
            optuna_configs.update(per_tf_data)
            print(f"  Merged per-TF config: {os.path.basename(per_tf_file)}")
        except Exception:
            pass

    ga_params = {}
    for tf in ALL_TFS:
        # Try optuna configs first (format: {"15m": {"dd10_best": {...}, "dd15_best": {...}, ...}})
        if tf in optuna_configs:
            tf_ec = optuna_configs[tf]
            profile = tf_ec.get('dd15_best') or tf_ec.get('dd10_best') or tf_ec.get('dd15_sortino')
            if profile:
                ga_params[tf] = {
                    'leverage': profile['leverage'],
                    'risk_pct': profile['risk_pct'],
                    'stop_atr': profile['stop_atr'],
                    'rr': profile['rr'],
                    'max_hold': profile['max_hold'],
                    'exit_type': profile.get('exit_type', 0),
                    'conf_thresh': profile['conf_thresh'],
                }
                print(f"  {tf} config (exhaustive): {profile['leverage']:.0f}x lev, {profile['risk_pct']:.1f}% risk, {profile['stop_atr']:.1f}ATR, {profile['rr']:.1f}:1 RR, {profile['max_hold']}bar, conf>{profile['conf_thresh']:.2f}")
                continue

        # Fall back to GA configs
        cfg = ml_configs.get(tf, {})
        # Always use god_mode first (user approved 12.6% DD)
        for level in ['god_mode', 'aggressive', 'balanced']:
            c = cfg.get('configs', {}).get(level, {})
            if c and 'params' in c:
                p = c['params']
                ga_params[tf] = {
                    'leverage': p[0], 'risk_pct': p[1]/100 * RISK_SCALE,
                    'stop_atr': p[2], 'rr': p[3], 'max_hold': int(p[4]),
                    'conf_thresh': p[6]
                }
                print(f"  {tf} config ({level}): {p[0]:.0f}x lev, {p[1]*RISK_SCALE:.1f}% risk, {p[2]:.1f}ATR, {p[3]:.1f}:1 RR, {int(p[4])}bar, conf>{p[6]:.2f}")
                break

    print(f"\n  Models: {list(models.keys())}")
    print(f"  Configs: {list(ga_params.keys())}")
    pool_summary = {k: f"${v['balance']:.2f} ({v['alloc']*100:.0f}%)" for k, v in tf_pools.items()}
    print(f"  TF Pools: {pool_summary}")

    # Fit HMM on startup for live regime features
    if _HAS_HMM:
        fit_hmm_live()
    else:
        print("  HMM not available (pip install hmmlearn)")

    print(f"\n  Waiting for next bar close...\n")

    last_bar_time = {}

    while True:
        try:
            now = datetime.now(timezone.utc)

            # Wait for 15m bar close (at :00, :15, :30, :45 + 5 second buffer)
            minutes = now.minute
            seconds_to_next = ((15 - minutes % 15) * 60 - now.second + 5) % (15 * 60)
            if seconds_to_next > 10:
                time.sleep(min(seconds_to_next, 30))
                continue

            # Update portfolio-level DD
            conn_dd = sqlite3.connect(TRADES_DB)
            portfolio_balance = conn_dd.execute("SELECT balance FROM account WHERE id=1").fetchone()[0]
            portfolio_peak = conn_dd.execute("SELECT peak_balance FROM account WHERE id=1").fetchone()[0]
            conn_dd.close()
            portfolio_dd = (portfolio_peak - portfolio_balance) / portfolio_peak if portfolio_peak > 0 else 0
            portfolio_halted = portfolio_dd > MAX_PORTFOLIO_DD
            if portfolio_halted:
                print(f"  PORTFOLIO HALTED — DD {portfolio_dd*100:.1f}% exceeds {MAX_PORTFOLIO_DD*100:.0f}% limit")

            # Check which bars just closed
            for tf in ALL_TFS:
                tf_minutes = {'5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440, '1w': 10080}[tf]
                if not is_bar_close(now, tf):
                    continue
                if tf not in models:
                    continue

                # Check per-TF pool
                pool = tf_pools.get(tf, {})
                if pool.get('halted', False):
                    try:
                        log_rejected_trade(tf=tf, direction='UNKNOWN', timestamp=now.isoformat(),
                            price=None, confidence=None, reason='tf_halted',
                            meta_prob=None, confluence_info=None, feat_dict=None)
                    except Exception:
                        pass
                    continue

                bar_key = f"{tf}_{now.strftime('%Y%m%d%H%M')}"
                if bar_key in last_bar_time:
                    continue
                last_bar_time[bar_key] = True

                print(f"\n[{now.strftime('%H:%M:%S')}] {tf.upper()} bar closed")

                # Compute features via shared library (handles OHLCV + HTF + esoteric)
                feat_names = features_list[tf]
                feat_dict, feat_df = compute_features_live(tf, feat_names)
                if not feat_dict:
                    print(f"  No features returned for {tf}")
                    continue

                # Inject HMM features (Bug fix: were missing in live mode)
                hmm_feats = get_hmm_features(feat_dict)
                feat_dict.update(hmm_feats)

                price = feat_dict.get('close', 0)
                if price <= 0:
                    ohlcv_tmp = live_dal.get_ohlcv_window(tf, 2)
                    price = float(ohlcv_tmp['close'].iloc[-1]) if ohlcv_tmp is not None and len(ohlcv_tmp) > 0 else 0

                # Build feature vector
                X = np.array([[feat_dict.get(fn, np.nan) for fn in feat_names]], dtype=np.float32)
                X = np.where(np.isinf(X), np.nan, X)

                # Predict (3-class softprob: SHORT=0, FLAT=1, LONG=2)
                # LightGBM predicts directly on numpy arrays (no DMatrix needed)
                raw_pred = models[tf].predict(X)
                if raw_pred.ndim == 2:
                    probs_3c = raw_pred[0]  # shape (3,)
                elif len(raw_pred) >= 3:
                    probs_3c = raw_pred[:3]  # flattened
                else:
                    # Legacy binary model fallback
                    probs_3c = np.array([1 - raw_pred[0], 0.0, raw_pred[0]])

                p_short, p_flat, p_long = float(probs_3c[0]), float(probs_3c[1]), float(probs_3c[2])
                pred_class = int(np.argmax(probs_3c))
                confidence = float(np.max(probs_3c))
                atr = feat_dict.get('atr_14', price * 0.01)
                params = ga_params.get(tf, {})
                conf_thresh = params.get('conf_thresh', 0.80)

                # Detect regime (returns name + int for multiplier lookup)
                sma100 = feat_dict.get('sma_100', price)
                ema50_slope_val = feat_dict.get('ema50_slope', 0)
                sma100_slope = ema50_slope_val / 100.0 if ema50_slope_val else 0
                regime, regime_idx = detect_regime(price, sma100, sma100_slope)
                r_mult = REGIME_MULT[regime_idx]

                direction = None
                if pred_class == 2 and confidence > conf_thresh:
                    direction = 'LONG'
                    prob = p_long
                elif pred_class == 0 and confidence > conf_thresh:
                    direction = 'SHORT'
                    prob = p_short
                else:
                    prob = confidence

                print(f"  Price: ${price:,.0f} | Pred: {['SHORT','FLAT','LONG'][pred_class]} "
                      f"({confidence:.1%}) | P(L/F/S)={p_long:.2f}/{p_flat:.2f}/{p_short:.2f} | Regime: {regime} (lev={r_mult['lev']:.2f}x risk={r_mult['risk']:.2f}x)")

                if direction and portfolio_halted:
                    print(f"  SKIP — portfolio DD halt active ({portfolio_dd*100:.1f}%)")
                    try:
                        log_rejected_trade(tf=tf, direction=direction, timestamp=now.isoformat(),
                            price=price, confidence=confidence, reason='dd_halt',
                            meta_prob=None, confluence_info=None, feat_dict=feat_dict)
                    except Exception:
                        pass
                    direction = None

                if direction:
                    # DEDUP: skip if already have an open trade for this TF
                    # Use single connection for atomic check+insert
                    conn = sqlite3.connect(TRADES_DB, timeout=10)
                    conn.execute("BEGIN IMMEDIATE")
                    existing_open = conn.execute(
                        "SELECT COUNT(*) FROM trades WHERE tf=? AND status='open'", (tf,)
                    ).fetchone()[0]
                    if existing_open > 0:
                        print(f"  SKIP — already have {existing_open} open {tf} trade(s)")
                        try:
                            log_rejected_trade(tf=tf, direction=direction, timestamp=now.isoformat(),
                                price=price, confidence=confidence, reason='duplicate',
                                meta_prob=None, confluence_info=None, feat_dict=feat_dict)
                        except Exception:
                            pass
                        conn.rollback()
                        conn.close()
                        continue

                    # Also check for duplicate entry within same minute (race protection)
                    entry_minute = now.strftime('%Y-%m-%dT%H:%M')
                    recent_dup = conn.execute(
                        "SELECT COUNT(*) FROM trades WHERE tf=? AND entry_time LIKE ?",
                        (tf, entry_minute + '%')
                    ).fetchone()[0]
                    if recent_dup > 0:
                        print(f"  SKIP — duplicate entry for {tf} at {entry_minute}")
                        try:
                            log_rejected_trade(tf=tf, direction=direction, timestamp=now.isoformat(),
                                price=price, confidence=confidence, reason='duplicate',
                                meta_prob=None, confluence_info=None, feat_dict=feat_dict)
                        except Exception:
                            pass
                        conn.rollback()
                        conn.close()
                        continue

                    # Extra dedup: check both open AND closed trades in same minute
                    any_dup = conn.execute(
                        "SELECT COUNT(*) FROM trades WHERE tf=? AND substr(entry_time,1,16)=?",
                        (tf, entry_minute)
                    ).fetchone()[0]
                    if any_dup > 0:
                        print(f"  SKIP — exact duplicate for {tf} at {entry_minute}")
                        try:
                            log_rejected_trade(tf=tf, direction=direction, timestamp=now.isoformat(),
                                price=price, confidence=confidence, reason='duplicate',
                                meta_prob=None, confluence_info=None, feat_dict=feat_dict)
                        except Exception:
                            pass
                        conn.rollback()
                        conn.close()
                        continue

                    # === SINGLE PROBABILITY PIPELINE ===
                    # 1. Base LightGBM prob (already computed above as 'prob')
                    # prob is P(predicted_class): p_long for LONG, p_short for SHORT

                    # 2. LSTM blending (if model exists and feat_df available)
                    if _HAS_LSTM and tf in lstm_extractors and feat_df is not None:
                        try:
                            lstm_feats = lstm_extractors[tf].extract(feat_df)
                            lstm_p = lstm_feats['lstm_prob'].iloc[-1]
                            if not np.isnan(lstm_p):
                                if tf in platt_models:
                                    lstm_p = apply_platt_calibration(np.array([lstm_p]), platt_models[tf])[0]
                                # Blend: 80% LightGBM + 20% LSTM
                                prob = 0.8 * prob + 0.2 * lstm_p
                                print(f"  LSTM blend: lgbm={prob:.3f} lstm={lstm_p:.3f} → blended={prob:.3f}")
                        except Exception as e:
                            print(f"  LSTM blend failed for {tf}: {e}")

                    # 3. Meta-labeling gate (if model exists)
                    if _HAS_META and tf in meta_models:
                        try:
                            # Build 3-class prob array for meta input
                            _lgbm_3c = np.array([[p_short, p_flat, p_long]])
                            meta_probs, take = predict_meta(meta_models[tf], _lgbm_3c)
                            if not take[0]:
                                print(f"  META GATE: {tf} trade rejected (meta_prob={meta_probs[0]:.3f})")
                                try:
                                    log_rejected_trade(tf=tf, direction=direction, timestamp=now.isoformat(),
                                        price=price, confidence=prob, reason='meta_gate',
                                        meta_prob=float(meta_probs[0]), confluence_info=None, feat_dict=feat_dict)
                                except Exception:
                                    pass
                                conn.rollback()
                                conn.close()
                                continue
                        except Exception:
                            pass  # meta failed, allow trade through

                    # 4. Kelly uses blended+gated probability
                    confidence = prob  # already set to p_long for LONG, p_short for SHORT

                    # Base params from optimizer
                    base_lev = params.get('leverage', 10)
                    base_stop = params.get('stop_atr', 1.0)
                    base_rr = params.get('rr', 2.0)
                    base_hold = params.get('max_hold', 4)

                    # Apply 5-dimensional regime multipliers (matches backtest)
                    lev = base_lev * r_mult['lev']
                    stop_mult = base_stop * r_mult['stop']
                    rr = base_rr * r_mult['rr']
                    max_hold = int(base_hold * r_mult['hold'])

                    # Kelly-based bet sizing (fractional, with safety)
                    base_risk = params.get('risk_pct', 0.01) * r_mult['risk']
                    kelly_frac = 0.25  # safety factor: use 25% of full Kelly
                    p_win = confidence  # P(correct direction) — blended + meta-gated
                    b_ratio = rr  # avg_win / avg_loss ≈ reward:risk ratio
                    kelly_f = (p_win * b_ratio - (1 - p_win)) / max(b_ratio, 0.01)
                    kelly_f = max(kelly_f, 0)  # never negative (don't bet against yourself)
                    kelly_risk = base_risk * (1 + kelly_frac * kelly_f)  # scale base risk by Kelly
                    kelly_risk = min(kelly_risk, base_risk * 3)  # cap at 3x base risk

                    # Drawdown scaling: reduce as DD deepens
                    dd_scale = max(0.0, 1.0 - 2.0 * portfolio_dd) if portfolio_dd < 0.15 else 0.0
                    risk = kelly_risk * dd_scale

                    d = 1 if direction == 'LONG' else -1
                    stop_price = price - d * stop_mult * atr
                    tp_price = price + d * stop_mult * atr * rr
                    # Get account balance
                    balance = conn.execute("SELECT balance FROM account WHERE id=1").fetchone()[0]

                    # Save ALL pruned features for full trade reasoning
                    key_feats = {}
                    for fn in feat_names:
                        val = feat_dict.get(fn, np.nan)
                        key_feats[fn] = round(val, 4) if not (isinstance(val, float) and np.isnan(val)) else None

                    cur = conn.execute("""INSERT INTO trades
                        (tf, direction, confidence, entry_price, entry_time, stop_price, tp_price,
                         regime, leverage, risk_pct, features_json, status)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')""",
                        (tf, direction, confidence, price, now.isoformat(),
                         stop_price, tp_price, regime, lev, risk * 100,
                         json.dumps(key_feats)))
                    trade_id = cur.lastrowid
                    conn.commit()
                    conn.close()

                    # --- Journal: log trade snapshot ---
                    try:
                        log_trade_snapshot(
                            trade_id=trade_id, tf=tf, direction=direction,
                            entry_time=now.isoformat(), entry_price=price,
                            xgb_probs={'long': p_long, 'flat': p_flat, 'short': p_short},  # kwarg name kept for trade_journal compat (data is LightGBM)
                            lstm_prob=locals().get('lstm_p'),
                            meta_prob=locals().get('meta_probs', [None])[0] if locals().get('meta_probs') is not None else None,
                            blended_conf=confidence,
                            regime_info={
                                'regime': regime, 'regime_idx': regime_idx,
                                'hmm_bull': hmm_feats.get('hmm_bull_prob'),
                                'hmm_bear': hmm_feats.get('hmm_bear_prob'),
                                'hmm_neutral': hmm_feats.get('hmm_neutral_prob'),
                                'hmm_state': hmm_feats.get('hmm_state'),
                            },
                            sizing_info={
                                'kelly_fraction': kelly_f, 'base_risk_pct': base_risk,
                                'final_risk_pct': risk, 'leverage': lev,
                                'dd_scale': dd_scale, 'portfolio_dd': portfolio_dd,
                                'tf_pool_dd': pool.get('dd', 0),
                            },
                            trade_params={
                                'stop_atr_mult': stop_mult, 'rr_ratio': rr,
                                'max_hold': max_hold, 'atr_14': atr,
                            },
                            feat_dict=key_feats,
                        )
                    except Exception as e:
                        print(f"  [journal] snapshot error: {e}")

                    # Write prediction cache
                    prediction = {
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'timeframe': tf,
                        'direction': direction,
                        'confidence': float(confidence),
                        'entry_price': float(price),
                        'stop_loss': float(stop_price),
                        'take_profit': float(tp_price),
                    }
                    try:
                        with open(f'{DB_DIR}/prediction_cache.json', 'w') as f:
                            json.dump(prediction, f, indent=2)
                    except Exception:
                        pass

                    print(f"  >>> {direction} {tf.upper()} @ ${price:,.0f} | Conf={confidence:.1%} "
                          f"| SL=${stop_price:,.0f} TP=${tp_price:,.0f} | Lev={lev:.0f}x Risk={risk*100:.1f}%")
                else:
                    print(f"  No signal (conf {prob:.3f} below threshold {conf_thresh:.2f})")
                    # --- Journal: log rejected trade (below threshold / flat) ---
                    try:
                        _rej_dir = ['SHORT', 'FLAT', 'LONG'][pred_class]
                        _rej_reason = 'below_threshold' if confidence <= conf_thresh else 'flat_prediction'
                        log_rejected_trade(tf=tf, direction=_rej_dir, timestamp=now.isoformat(),
                            price=price, confidence=confidence, reason=_rej_reason,
                            meta_prob=None, confluence_info=None, feat_dict=feat_dict)
                    except Exception:
                        pass

                # Check open positions for exits
                conn = sqlite3.connect(TRADES_DB)
                open_trades = conn.execute(
                    "SELECT id, tf, direction, entry_price, stop_price, tp_price, entry_time, leverage, risk_pct FROM trades WHERE status='open'"
                ).fetchall()

                for trade in open_trades:
                    tid, ttf, tdir, entry, sl, tp, etime, tlev, trisk = trade
                    entry_dt = datetime.fromisoformat(etime)
                    bars_held = int((now - entry_dt).total_seconds() / (tf_minutes * 60))
                    max_h = ga_params.get(ttf, {}).get('max_hold', 4)

                    sl_hit = (tdir == 'LONG' and price <= sl) or (tdir == 'SHORT' and price >= sl)
                    tp_hit = (tdir == 'LONG' and price >= tp) or (tdir == 'SHORT' and price <= tp)
                    time_exit = bars_held >= max_h

                    if sl_hit or tp_hit or time_exit:
                        d = 1 if tdir == 'LONG' else -1
                        pchange = (price - entry) / entry * d
                        gross = pchange * tlev
                        fee = FEE_RATE * tlev
                        net = gross - fee
                        balance = conn.execute("SELECT balance FROM account WHERE id=1").fetchone()[0]
                        pnl = balance * (trisk / 100) * net
                        pnl_pct = pnl / balance * 100
                        reason = 'SL' if sl_hit else ('TP' if tp_hit else 'TIME')

                        new_balance = balance + pnl
                        peak = conn.execute("SELECT peak_balance FROM account WHERE id=1").fetchone()[0]
                        new_peak = max(peak, new_balance)
                        dd = (new_peak - new_balance) / new_peak if new_peak > 0 else 0
                        wins = conn.execute("SELECT wins FROM account WHERE id=1").fetchone()[0]
                        losses = conn.execute("SELECT losses FROM account WHERE id=1").fetchone()[0]
                        total = conn.execute("SELECT total_trades FROM account WHERE id=1").fetchone()[0]

                        conn.execute("""UPDATE trades SET exit_price=?, exit_time=?, pnl=?, pnl_pct=?,
                            bars_held=?, exit_reason=?, status='closed' WHERE id=?""",
                            (price, now.isoformat(), pnl, pnl_pct, bars_held, reason, tid))

                        conn.execute("""UPDATE account SET balance=?, peak_balance=?,
                            total_trades=?, wins=?, losses=?, max_dd=?, updated_at=? WHERE id=1""",
                            (new_balance, new_peak, total + 1,
                             wins + (1 if pnl > 0 else 0),
                             losses + (1 if pnl <= 0 else 0),
                             max(dd, conn.execute("SELECT max_dd FROM account WHERE id=1").fetchone()[0]),
                             now.isoformat()))

                        conn.execute("INSERT INTO equity_curve (timestamp, balance, dd_pct) VALUES (?, ?, ?)",
                                     (now.isoformat(), new_balance, dd * 100))

                        print(f"  <<< CLOSED {tdir} {ttf.upper()} @ ${price:,.0f} | PnL=${pnl:+.2f} ({pnl_pct:+.1f}%) [{reason}]")

                        # --- Journal: log trade outcome ---
                        try:
                            log_trade_outcome(
                                trade_id=tid, tf=ttf, direction=tdir,
                                entry_price=entry, exit_price=price,
                                pnl=pnl, pnl_pct=pnl_pct, exit_reason=reason,
                                bars_held=bars_held,
                                feat_dict_exit=feat_dict if ttf == tf else None,
                                predicted_dir=tdir,
                                confidence=None,
                            )
                        except Exception as e:
                            print(f"  [journal] outcome error: {e}")

                        # Update per-TF pool tracking
                        if ttf in tf_pools:
                            tf_pools[ttf]['balance'] += pnl
                            tf_pools[ttf]['peak'] = max(tf_pools[ttf]['peak'], tf_pools[ttf]['balance'])
                            if tf_pools[ttf]['peak'] > 0:
                                tf_pools[ttf]['dd'] = (tf_pools[ttf]['peak'] - tf_pools[ttf]['balance']) / tf_pools[ttf]['peak']
                            if tf_pools[ttf]['dd'] > MAX_TF_DD:
                                tf_pools[ttf]['halted'] = True
                                print(f"  !!! {ttf.upper()} POOL HALTED — DD {tf_pools[ttf]['dd']*100:.1f}% exceeds {MAX_TF_DD*100:.0f}% limit")

                conn.commit()
                conn.close()

                # --- Journal: monitor open trades for MAE/MFE tracking ---
                try:
                    conn_j = sqlite3.connect(TRADES_DB)
                    open_monitor = conn_j.execute(
                        "SELECT id, tf, direction, entry_price, stop_price, tp_price, entry_time "
                        "FROM trades WHERE status='open'"
                    ).fetchall()
                    conn_j.close()
                    for m_tid, m_tf, m_dir, m_entry, m_sl, m_tp, m_etime in open_monitor:
                        ohlcv_mon = live_dal.get_ohlcv_window(m_tf, 2)
                        if ohlcv_mon is not None and len(ohlcv_mon) > 0:
                            bar = ohlcv_mon.iloc[-1]
                            m_entry_dt = datetime.fromisoformat(m_etime)
                            m_tf_minutes = {'5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440, '1w': 10080}.get(m_tf, 60)
                            bar_num = max(0, int((now - m_entry_dt).total_seconds() / (m_tf_minutes * 60)))
                            log_price_path_bar(
                                trade_id=m_tid, bar_num=bar_num,
                                timestamp=now.isoformat(),
                                ohlcv={'open': float(bar.get('open', 0)), 'high': float(bar.get('high', 0)),
                                       'low': float(bar.get('low', 0)), 'close': float(bar.get('close', 0)),
                                       'volume': float(bar.get('volume', 0))},
                                entry_price=m_entry, direction=m_dir,
                                sl=m_sl if m_sl else m_entry,
                                tp=m_tp if m_tp else m_entry,
                            )
                except Exception as e:
                    print(f"  [journal] monitor error: {e}")

            time.sleep(5)

        except KeyboardInterrupt:
            print("\n  Shutting down...")
            break
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            time.sleep(30)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='paper', choices=['paper', 'live'])
    args = parser.parse_args()
    run_trading_loop(args.mode)
