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
from scipy.sparse import csr_matrix as _csr_matrix
from datetime import datetime, timedelta, timezone
import urllib.request
from feature_library import build_all_features
from data_access import LiveDataLoader
from trade_journal import (log_trade_snapshot, log_price_path_bar,
                           log_trade_outcome, log_rejected_trade,
                           compute_post_trade_analysis)
from inference_crosses import InferenceCrossComputer
from config import (RISK_LIMITS, DRAWDOWN_PROTOCOL, CIRCUIT_BREAKERS,
                    MODEL_VERSION_SCHEMA, TF_CAPITAL_ALLOC as CONFIG_TF_ALLOC,
                    FEE_RATE, TF_SLIPPAGE, REGIME_MULT, REGIME_SLOPE_THRESHOLD,
                    REGIME_NEAR_SMA_PCT, RISK_SCALE, KELLY_SAFETY_FRACTION,
                    KELLY_MAX_RISK_MULT, DD_HALT_THRESHOLD, DD_SCALE_STEEPNESS,
                    MAX_PORTFOLIO_DD, MAX_TF_DD, WARMUP_BARS, TRADES_DB,
                    STARTING_BALANCE as BACKTEST_STARTING_BALANCE,
                    LIVE_STARTING_BALANCE, PROJECT_DIR,
                    LIVE_CONF_THRESH_FALLBACK,
                    REGIME_CRASH_VOL_MULT, REGIME_CRASH_DD_THRESHOLD,
                    TRADE_THRESHOLDS, TRADE_TYPE_PARAMS)

# Institutional upgrades: meta-labeling + LSTM blending
try:
    from meta_labeling import predict_meta
    import pickle
    _HAS_META = True
except ImportError:
    _HAS_META = False

try:
    from lstm_sequence_model import LSTMFeatureExtractor, apply_platt_calibration
    _HAS_LSTM = True
except ImportError:
    _HAS_LSTM = False

# lleaves compiled model support (5.4x faster inference)
try:
    import lleaves
    _HAS_LLEAVES = True
except ImportError:
    _HAS_LLEAVES = False

# HMM regime detection (matches ml_multi_tf.py training pipeline)
try:
    from hmmlearn.hmm import GaussianHMM
    _HAS_HMM = True
except ImportError:
    _HAS_HMM = False

DB_DIR = os.environ.get('SAVAGE22_DB_DIR', PROJECT_DIR)

live_dal = LiveDataLoader(DB_DIR)
live_dal.initial_load()

# ── Kill Switch ──
KILL_SWITCH_FILE = os.path.join(PROJECT_DIR, 'KILL_SWITCH')

def check_kill_switch():
    """Check if kill switch file exists. Create this file to immediately halt all trading.
    Usage: touch KILL_SWITCH  (or: echo HALT > KILL_SWITCH)
    Remove the file to resume: rm KILL_SWITCH
    """
    return os.path.exists(KILL_SWITCH_FILE)

# ── Circuit Breaker State ──
_order_timestamps = []  # recent order times for rate limiting
_recent_pnls = []       # recent trade PnLs for sanity check

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
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        scaled_in INTEGER DEFAULT 0,
        original_size REAL DEFAULT 0,
        scaled_out INTEGER DEFAULT 0,
        partial_pnl REAL DEFAULT 0
    )""")
    # Add scale columns to existing DBs (idempotent)
    for _col, _type, _default in [
        ('scaled_in', 'INTEGER', '0'), ('original_size', 'REAL', '0'),
        ('scaled_out', 'INTEGER', '0'), ('partial_pnl', 'REAL', '0'),
    ]:
        try:
            conn.execute(f"ALTER TABLE trades ADD COLUMN {_col} {_type} DEFAULT {_default}")
        except:
            pass  # column already exists
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
        conn.execute("INSERT INTO account VALUES (1, 'paper', ?, ?, 0, 0, 0, 0.0, ?)",
                     (LIVE_STARTING_BALANCE, LIVE_STARTING_BALANCE,
                      datetime.now(timezone.utc).isoformat()))
    conn.commit()
    conn.close()

# ============================================================
# FETCH LIVE CANDLES FROM BITGET
# ============================================================
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

        # Daily-resolution features for 1w (AlphaNumetrix approach)
        ltf_data = None
        if tf_name == '1w':
            _daily_ohlcv = live_dal.get_ohlcv_window('1d', 500)
            if _daily_ohlcv is not None and len(_daily_ohlcv) > 10:
                ltf_data = {'1d': _daily_ohlcv}

        df_features = build_all_features(
            ohlcv=ohlcv,
            esoteric_frames=esoteric_frames,
            tf_name=tf_name,
            mode='live',
            htf_data=htf_data,
            ltf_data=ltf_data,
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
                # NaN-preserving crosses: NaN * anything = NaN (correct missing semantics)
                d_trend = pd.to_numeric(df_features.get('d_trend'), errors='coerce')
                bull = d_trend
                bear = 1 - d_trend
                for gcp_col in ['gcp_deviation_mean', 'gcp_deviation_max', 'gcp_rate_of_change', 'gcp_extreme']:
                    if gcp_col in df_features.columns:
                        sig = pd.to_numeric(df_features[gcp_col], errors='coerce')
                        df_features[f'tx_{gcp_col}_x_bull'] = sig * bull
                        df_features[f'tx_{gcp_col}_x_bear'] = sig * bear

                # GCP x fear/greed cross
                if 'gcp_deviation_mean' in df_features.columns:
                    gcp_dev = pd.to_numeric(df_features['gcp_deviation_mean'], errors='coerce')
                    fg_fear = pd.to_numeric(df_features.get('fg_extreme_fear'), errors='coerce')
                    fg_greed = pd.to_numeric(df_features.get('fg_extreme_greed'), errors='coerce')
                    df_features['gcp_x_fear'] = gcp_dev.abs() * fg_fear
                    df_features['gcp_x_greed'] = gcp_dev.abs() * fg_greed
                    # GCP x moon phase
                    moon = pd.to_numeric(df_features.get('west_moon_phase'), errors='coerce')
                    is_full = ((moon >= 13) & (moon <= 16)).astype(float)
                    is_new = ((moon < 2) | (moon > 27.5)).astype(float)
                    df_features['gcp_x_full_moon'] = gcp_dev.abs() * is_full
                    df_features['gcp_x_new_moon'] = gcp_dev.abs() * is_new
                    # GCP x Kp storm
                    kp = pd.to_numeric(df_features.get('sw_kp_is_storm'), errors='coerce')
                    df_features['gcp_x_kp_storm'] = gcp_dev.abs() * kp
            except ImportError:
                pass  # gcp_feature_builder not installed — GCP features stay NaN (correct)
            except Exception as e:
                import traceback
                print(f"  GCP features FAILED: {e}")
                traceback.print_exc()
                # GCP columns stay NaN — model handles missing natively

        last_row = df_features.iloc[-1]
        result = {}
        for fn in feat_names:
            val = last_row.get(fn, np.nan)
            if isinstance(val, (int, float)) and np.isinf(val):
                val = np.nan
            result[fn] = val
        return result, df_features
    except Exception as e:
        # PHILOSOPHY: crash > silent degradation. Log full traceback and RE-RAISE.
        # If features can't be computed, the bar should be SKIPPED (caller handles empty dict),
        # but the error must be visible — not silently masked.
        import traceback
        print(f"  CRITICAL: compute_features_live FAILED: {e}")
        traceback.print_exc()
        raise  # Let caller handle — do NOT return empty dict silently

# ============================================================
# DETECT REGIME — 5-dimensional multiplier tables (matches v2/backtesting_audit.py)
# ============================================================
# Regime multipliers imported from config.py (single source of truth)
# REGIME_MULT is imported at top from config

def detect_regime(close, sma100, sma100_slope, feat_dict=None):
    """Returns (regime_name, regime_int) for multiplier lookup.
    Matches backtesting_audit.py and exhaustive_optimizer.py:
    0=bull, 1=bear, 2=sideways, 3=crash.
    Crash: below SMA100 + high volatility + deep drawdown from recent high.
    """
    from config import (REGIME_NAMES, REGIME_CRASH_VOL_MULT,
                        REGIME_CRASH_DD_THRESHOLD)
    if sma100 is None or sma100_slope is None:
        return 'sideways', 2
    above = close > sma100

    # Base regime: bull / bear / sideways
    if above and sma100_slope > REGIME_SLOPE_THRESHOLD:
        regime_name, regime_idx = 'bull', 0
    elif not above and sma100_slope < -REGIME_SLOPE_THRESHOLD:
        regime_name, regime_idx = 'bear', 1
    else:
        regime_name, regime_idx = 'sideways', 2

    # Crash override: below SMA100 + volatility spike + drawdown from 30-bar high.
    # Use atr_14_pct as rvol proxy if available; dd estimated from price vs sma_30_high.
    if feat_dict and not above:
        atr_pct = feat_dict.get('atr_14_pct', None)
        atr_pct_sma = feat_dict.get('atr_14_pct_sma', atr_pct)  # long-term avg
        high_30 = feat_dict.get('high_30', None)  # 30-bar rolling high
        if high_30 is None:
            # Approximate from available features
            high_30 = feat_dict.get('sma_30_high', feat_dict.get('bb_upper_20', None))
        if (atr_pct is not None and atr_pct_sma is not None and atr_pct_sma > 0
                and high_30 is not None and high_30 > 0):
            vol_spike = atr_pct > REGIME_CRASH_VOL_MULT * atr_pct_sma
            dd_from_high = (high_30 - close) / high_30
            if vol_spike and dd_from_high > REGIME_CRASH_DD_THRESHOLD:
                regime_name, regime_idx = 'crash', 3

    return regime_name, regime_idx


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
        # Use latest daily return + abs_ret + vol for prediction.
        # HMM was trained on actual daily log returns: log(close_t / close_{t-1}).
        # sma_5 is the average of 5 bars, NOT the previous close — using it produces
        # log(close / avg5) which is a different signal and biases HMM state probabilities.
        close = feat_dict.get('close', 0)
        _ohlcv_hmm = live_dal.get_ohlcv_window('1d', 2)
        if _ohlcv_hmm is not None and len(_ohlcv_hmm) >= 2:
            prev_close = float(_ohlcv_hmm['close'].iloc[-2])
        else:
            prev_close = close  # fallback: log return = 0.0 (neutral, avoids wrong sma_5)
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
    except Exception as _hmm_err:
        import traceback
        print(f"  HMM features FAILED: {_hmm_err}")
        traceback.print_exc()
        # Return NaN for all HMM features — model handles missing natively
        return {
            'hmm_bull_prob': np.nan,
            'hmm_bear_prob': np.nan,
            'hmm_neutral_prob': np.nan,
            'hmm_state': np.nan,
        }


# ============================================================
# BAR CLOSE DETECTION
# ============================================================
def is_bar_close(now, tf):
    """Check if a bar just closed for this timeframe."""
    if tf == '15m':
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

def check_circuit_breakers(now, portfolio_balance, portfolio_peak):
    """Check all circuit breakers. Returns (ok, reason) tuple."""
    # Kill switch: immediate halt if file exists
    if check_kill_switch():
        return False, f"KILL_SWITCH file detected at {KILL_SWITCH_FILE}"

    # Rate limit: max orders per minute
    global _order_timestamps
    cutoff = now - timedelta(minutes=1)
    _order_timestamps = [t for t in _order_timestamps if t > cutoff]
    if len(_order_timestamps) >= CIRCUIT_BREAKERS['max_orders_per_minute']:
        return False, f"rate_limit ({len(_order_timestamps)} orders in last minute)"

    # PnL sanity check: if last trade PnL is >5σ from recent distribution
    if len(_recent_pnls) >= CIRCUIT_BREAKERS['pnl_lookback_trades']:
        recent = np.array(_recent_pnls[-CIRCUIT_BREAKERS['pnl_lookback_trades']:])
        mu, sigma = recent.mean(), recent.std()
        if sigma > 0 and len(_recent_pnls) > 0:
            last_z = abs(_recent_pnls[-1] - mu) / sigma
            if last_z > CIRCUIT_BREAKERS['pnl_sanity_sigma']:
                return False, f"pnl_anomaly (last trade {last_z:.1f}σ, threshold {CIRCUIT_BREAKERS['pnl_sanity_sigma']}σ)"

    # Concurrent position limit
    try:
        _cb_conn = sqlite3.connect(TRADES_DB, timeout=5)
        total_open = _cb_conn.execute("SELECT COUNT(*) FROM trades WHERE status='open'").fetchone()[0]
        if total_open >= RISK_LIMITS['max_concurrent_positions']:
            _cb_conn.close()
            return False, f"max_concurrent_positions ({total_open} >= {RISK_LIMITS['max_concurrent_positions']})"

        # Daily loss limit: sum of today's closed trade PnL
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        daily_pnl = _cb_conn.execute(
            "SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE status='closed' AND exit_time >= ?",
            (today_start,)
        ).fetchone()[0]
        if portfolio_balance > 0 and abs(daily_pnl) / portfolio_balance > RISK_LIMITS['max_daily_loss_pct'] and daily_pnl < 0:
            _cb_conn.close()
            return False, f"max_daily_loss ({abs(daily_pnl)/portfolio_balance*100:.1f}% > {RISK_LIMITS['max_daily_loss_pct']*100:.0f}%)"

        # Weekly loss limit: sum of this week's closed trade PnL
        days_since_monday = now.weekday()
        week_start = (now - timedelta(days=days_since_monday)).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        weekly_pnl = _cb_conn.execute(
            "SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE status='closed' AND exit_time >= ?",
            (week_start,)
        ).fetchone()[0]
        if portfolio_balance > 0 and abs(weekly_pnl) / portfolio_balance > RISK_LIMITS['max_weekly_loss_pct'] and weekly_pnl < 0:
            _cb_conn.close()
            return False, f"max_weekly_loss ({abs(weekly_pnl)/portfolio_balance*100:.1f}% > {RISK_LIMITS['max_weekly_loss_pct']*100:.0f}%)"

        # Gross exposure check: sum of (risk_pct/100 * leverage) for all open trades
        # This approximates notional exposure as fraction of equity
        exposure_rows = _cb_conn.execute(
            "SELECT COALESCE(SUM(risk_pct / 100.0 * leverage), 0) FROM trades WHERE status='open'"
        ).fetchone()[0]
        if portfolio_balance > 0 and exposure_rows > RISK_LIMITS['max_exposure_pct']:
            _cb_conn.close()
            return False, f"max_exposure ({exposure_rows*100:.0f}% > {RISK_LIMITS['max_exposure_pct']*100:.0f}%)"

        _cb_conn.close()
    except Exception:
        pass  # DB access failure should not block — other checks still apply

    return True, "ok"


def check_stale_data(tf_name, ohlcv):
    """Check if OHLCV data is stale (last bar too old). Returns (ok, reason) tuple.
    Uses CIRCUIT_BREAKERS['stale_data_max_bars'] — halt if data is older than N bars.
    """
    if ohlcv is None or len(ohlcv) == 0:
        return False, "no_data"

    tf_seconds = {'15m': 900, '1h': 3600, '4h': 14400, '1d': 86400, '1w': 604800}
    bar_sec = tf_seconds.get(tf_name, 3600)
    max_stale_bars = CIRCUIT_BREAKERS['stale_data_max_bars']

    last_bar_ts = ohlcv.index[-1]
    # 8B.10: Always use timezone-aware UTC (deprecated utcnow removed)
    now = datetime.now(timezone.utc)
    last_bar_dt = pd.Timestamp(last_bar_ts)
    if last_bar_dt.tzinfo is None:
        last_bar_dt = last_bar_dt.tz_localize('UTC')
    now_ts = pd.Timestamp(now)

    age_seconds = (now_ts - last_bar_dt).total_seconds()
    age_bars = age_seconds / bar_sec

    if age_bars > max_stale_bars:
        return False, f"stale_data ({tf_name}: last bar {age_bars:.1f} bars old, max={max_stale_bars})"
    return True, "ok"


def get_drawdown_adjustments(portfolio_dd):
    """Apply drawdown protocol. Returns (risk_mult, min_conf, halted, description)."""
    risk_mult = 1.0
    min_conf = None
    halted = False
    desc = "normal"

    for dd_threshold in sorted(DRAWDOWN_PROTOCOL.keys()):
        if portfolio_dd >= dd_threshold:
            protocol = DRAWDOWN_PROTOCOL[dd_threshold]
            risk_mult = protocol['risk_multiplier']
            min_conf = protocol.get('min_confidence')
            halted = protocol['action'] == 'sim_only'
            desc = protocol['description']

    return risk_mult, min_conf, halted, desc


def check_max_notional(risk_pct, balance, leverage=1.0):
    """Ensure single order notional (risk_pct * leverage) doesn't exceed max notional limit.
    max_notional_per_order is fraction of equity — caps gross exposure per trade.
    Returns capped risk_pct.
    """
    max_notional_pct = CIRCUIT_BREAKERS['max_notional_per_order']
    # Notional = risk_pct * leverage. Cap so notional <= max_notional_pct of equity.
    if leverage > 0 and risk_pct * leverage > max_notional_pct:
        capped = max_notional_pct / leverage
        return capped
    return risk_pct


def log_model_version():
    """Log current model version info for auditability."""
    import hashlib, glob as _g
    version_info = {}
    try:
        # Code hash: hash of all .py files in project dir
        py_files = sorted(_g.glob(os.path.join(os.path.dirname(__file__), '*.py')))
        hasher = hashlib.sha256()
        for pf in py_files:
            with open(pf, 'rb') as f:
                hasher.update(f.read())
        version_info['code_hash'] = hasher.hexdigest()[:12]
        version_info['timestamp'] = datetime.now(timezone.utc).isoformat()
        version_info['config_hash'] = hashlib.sha256(
            json.dumps(RISK_LIMITS, sort_keys=True).encode()
        ).hexdigest()[:12]

        version_path = MODEL_VERSION_SCHEMA['version_file']
        # Append to version log
        history = []
        if os.path.exists(version_path):
            with open(version_path) as f:
                history = json.load(f) if os.path.getsize(version_path) > 0 else []
        history.append(version_info)
        # Keep last 100 entries
        history = history[-100:]
        with open(version_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"  Model version logged: code={version_info['code_hash']} config={version_info['config_hash']}")
    except Exception as e:
        print(f"  Model version logging failed: {e}")


# ============================================================
# MAIN TRADING LOOP
# ============================================================
def run_trading_loop(mode='paper'):
    print("=" * 60)
    print(f"  LIVE TRADER — Mode: {mode.upper()}")
    print(f"  Started: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    init_trades_db()

    # ── Configuration Validation (Tier 1: catch typos/corruption before any trading) ──
    _config_errors = []
    if RISK_LIMITS['max_leverage'] <= 0 or RISK_LIMITS['max_leverage'] > 200:
        _config_errors.append(f"max_leverage={RISK_LIMITS['max_leverage']} — must be in (0, 200]")
    if RISK_LIMITS['max_daily_loss_pct'] <= 0 or RISK_LIMITS['max_daily_loss_pct'] > 1.0:
        _config_errors.append(f"max_daily_loss_pct={RISK_LIMITS['max_daily_loss_pct']} — must be in (0, 1]")
    if RISK_LIMITS['max_weekly_loss_pct'] <= 0 or RISK_LIMITS['max_weekly_loss_pct'] > 1.0:
        _config_errors.append(f"max_weekly_loss_pct={RISK_LIMITS['max_weekly_loss_pct']} — must be in (0, 1]")
    if RISK_LIMITS['max_open_risk_pct'] <= 0 or RISK_LIMITS['max_open_risk_pct'] > 1.0:
        _config_errors.append(f"max_open_risk_pct={RISK_LIMITS['max_open_risk_pct']} — must be in (0, 1]")
    if RISK_LIMITS['max_concurrent_positions'] <= 0 or RISK_LIMITS['max_concurrent_positions'] > 50:
        _config_errors.append(f"max_concurrent_positions={RISK_LIMITS['max_concurrent_positions']} — must be in (0, 50]")
    if RISK_LIMITS['max_exposure_pct'] <= 0 or RISK_LIMITS['max_exposure_pct'] > 10.0:
        _config_errors.append(f"max_exposure_pct={RISK_LIMITS['max_exposure_pct']} — must be in (0, 10]")
    if FEE_RATE < 0 or FEE_RATE > 0.05:
        _config_errors.append(f"FEE_RATE={FEE_RATE} — must be in [0, 0.05]")
    if LIVE_STARTING_BALANCE <= 0:
        _config_errors.append(f"LIVE_STARTING_BALANCE={LIVE_STARTING_BALANCE} — must be > 0")
    for tf, slip in TF_SLIPPAGE.items():
        if slip < 0 or slip > 0.01:
            _config_errors.append(f"TF_SLIPPAGE['{tf}']={slip} — must be in [0, 0.01]")
    for dd_thresh in DRAWDOWN_PROTOCOL:
        if dd_thresh <= 0 or dd_thresh > 1.0:
            _config_errors.append(f"DRAWDOWN_PROTOCOL threshold {dd_thresh} — must be in (0, 1]")
    if CIRCUIT_BREAKERS['max_orders_per_minute'] <= 0:
        _config_errors.append(f"max_orders_per_minute={CIRCUIT_BREAKERS['max_orders_per_minute']} — must be > 0")
    if CIRCUIT_BREAKERS['stale_data_max_bars'] <= 0:
        _config_errors.append(f"stale_data_max_bars={CIRCUIT_BREAKERS['stale_data_max_bars']} — must be > 0")
    if _config_errors:
        print("\n  *** CONFIGURATION VALIDATION FAILED ***")
        for err in _config_errors:
            print(f"    - {err}")
        print("  Fix config.py and restart. Refusing to trade with invalid config.")
        sys.exit(1)
    print("  Config validation: PASSED")

    log_model_version()

    # Capital allocation per TF — use optimized allocation if available, else defaults
    from config import load_tf_allocation
    TF_CAPITAL_ALLOC = load_tf_allocation()

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
    # MAX_PORTFOLIO_DD and MAX_TF_DD imported from config.py

    # Load models
    ALL_TFS = ['15m', '1h', '4h', '1d', '1w']
    models = {}
    features_list = {}
    _lleaves_tfs = set()  # track which TFs use compiled lleaves models
    for tf in ALL_TFS:
        # Model priority: lleaves compiled > pruned lgb > full lgb
        # Compiled models give 5.4x prediction speedup
        _ext = '.dll' if sys.platform == 'win32' else ('.dylib' if sys.platform == 'darwin' else '.so')
        compiled_path = f'{DB_DIR}/model_{tf}_compiled{_ext}'
        pruned_model_path = f'{DB_DIR}/model_{tf}_pruned.json'
        model_path = f'{DB_DIR}/model_{tf}.json'

        # Feature list priority: pruned (matches pruned/compiled model) > all
        feat_path_pruned = f'{DB_DIR}/features_{tf}_pruned.json'
        feat_path_all = f'{DB_DIR}/features_{tf}_all.json'

        # Select model: compiled > pruned > full
        m = None
        used_model = None
        if _HAS_LLEAVES and os.path.exists(compiled_path):
            try:
                m = lleaves.Model(cache=compiled_path)
                used_model = f'lleaves compiled ({os.path.basename(compiled_path)})'
                _lleaves_tfs.add(tf)
            except Exception as e:
                print(f"  WARNING: lleaves load failed for {tf}: {e} — falling back to LightGBM")
                m = None
        if m is None and os.path.exists(pruned_model_path):
            m = lgb.Booster(model_file=pruned_model_path)
            used_model = f'pruned LightGBM ({os.path.basename(pruned_model_path)})'
        if m is None and os.path.exists(model_path):
            m = lgb.Booster(model_file=model_path)
            used_model = f'full LightGBM ({os.path.basename(model_path)})'

        # Select feature list: pruned if using pruned/compiled model, else all
        if os.path.exists(feat_path_pruned) and (
                os.path.exists(pruned_model_path) or os.path.exists(compiled_path)):
            feat_path = feat_path_pruned
        elif os.path.exists(feat_path_all):
            feat_path = feat_path_all
        elif os.path.exists(feat_path_pruned):
            feat_path = feat_path_pruned
        else:
            feat_path = None

        if m is not None and feat_path:
            models[tf] = m
            with open(feat_path) as f:
                features_list[tf] = json.load(f)
            # Model staleness detection: warn if model file is >30 days old
            # Use whichever model file exists for age check
            _age_path = next((p for p in [compiled_path, pruned_model_path, model_path]
                              if os.path.exists(p)), None)
            model_age_days = (time.time() - os.path.getmtime(_age_path)) / 86400 if _age_path else 0
            age_warning = ""
            if model_age_days > 90:
                age_warning = " *** CRITICAL: MODEL >90 DAYS OLD — RETRAIN IMMEDIATELY ***"
            elif model_age_days > 30:
                age_warning = f" ** WARNING: model {model_age_days:.0f} days old — consider retraining **"
            # 8B.3: Feature order validation — ensure features_list matches model's expected order
            # lleaves compiled models don't expose feature_name() — skip validation (trust pruned features file)
            model_feature_names = m.feature_name() if hasattr(m, 'feature_name') else None
            if model_feature_names and features_list[tf] != model_feature_names:
                model_set = set(model_feature_names)
                file_set = set(features_list[tf])
                only_in_model = model_set - file_set
                only_in_file = file_set - model_set
                print(f"\n  {'='*60}")
                print(f"  *** CRITICAL: FEATURE ORDER MISMATCH for {tf} ***")
                print(f"  Model expects {len(model_feature_names)} features")
                print(f"  Features file has {len(features_list[tf])} features")
                if only_in_model:
                    print(f"  In model but NOT in file: {len(only_in_model)} features")
                if only_in_file:
                    print(f"  In file but NOT in model: {len(only_in_file)} features")
                if not only_in_model and not only_in_file:
                    print(f"  Same features but DIFFERENT ORDER — using model's order")
                print(f"  CORRECTED: Using model's feature order (authoritative)")
                print(f"  {'='*60}\n")
                # Write full mismatch details to log file for investigation
                try:
                    mismatch_log = os.path.join(DB_DIR, f'feature_mismatch_{tf}.log')
                    with open(mismatch_log, 'w') as f:
                        f.write(f"Feature order mismatch for {tf}\n")
                        f.write(f"Model features: {len(model_feature_names)}\n")
                        f.write(f"File features: {len(features_list[tf])}\n")
                        if only_in_model:
                            f.write(f"\nIn model but NOT in file ({len(only_in_model)}):\n")
                            for feat in sorted(only_in_model):
                                f.write(f"  {feat}\n")
                        if only_in_file:
                            f.write(f"\nIn file but NOT in model ({len(only_in_file)}):\n")
                            for feat in sorted(only_in_file):
                                f.write(f"  {feat}\n")
                except Exception:
                    pass
                features_list[tf] = model_feature_names

            print(f"  Loaded {tf} model via {used_model} ({len(features_list[tf])} features, {model_age_days:.0f}d old){age_warning}")

    # 8B.1: Initialize InferenceCrossComputer for each TF with a trained model
    cross_computers = {}
    for tf in models:
        try:
            xc = InferenceCrossComputer(tf)
            cross_computers[tf] = xc
            print(f"  Loaded InferenceCrossComputer for {tf}")
        except Exception as e:
            print(f"  WARNING: InferenceCrossComputer for {tf} failed: {e}")

    # S-3: Build O(1) lookup structures for sparse inference — built once at startup.
    # Eliminates 2.9M dict iterations per bar (line 979 for-loop + line 1013 list comp).
    # feat_name_to_idx[tf]: {feature_name: col_index} for O(1) positional lookup.
    # cross_feat_positions[tf]: int32 array mapping cross_values[i] → col_index in feat_names.
    feat_name_to_idx = {}
    cross_feat_positions = {}
    for tf in models:
        idx_map = {name: i for i, name in enumerate(features_list[tf])}
        feat_name_to_idx[tf] = idx_map
        if tf in cross_computers:
            try:
                cnames = cross_computers[tf].get_cross_feature_names()
                cross_feat_positions[tf] = np.array(
                    [idx_map.get(cn, -1) for cn in cnames], dtype=np.int32)
            except Exception as e:
                print(f"  WARNING: Could not build cross positions for {tf}: {e}")
                cross_feat_positions[tf] = np.array([], dtype=np.int32)

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
    configs_path = os.path.join(DB_DIR, 'ml_multi_tf_configs.json')
    if os.path.exists(configs_path):
        with open(configs_path) as f:
            ml_configs = json.load(f)
    else:
        ml_configs = {}  # Will be populated by optuna_configs below

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

    # 8B.4: Trailing stop tracking — best price per trade for trail stop computation
    _trade_best_prices = {}  # trade_id -> best price seen since entry

    # 8B.5: Partial TP tracking — which trades have already scaled out
    _trade_scaled_out = set()  # set of trade_ids that have taken partial profit

    while True:
        try:
            now = datetime.now(timezone.utc)

            # ── Kill Switch Check (top of every loop iteration) ──
            if check_kill_switch():
                print(f"\n  *** KILL SWITCH ACTIVE *** — File: {KILL_SWITCH_FILE}")
                print(f"  All trading halted. Remove the file to resume.")
                time.sleep(10)
                continue

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
            # Drawdown protocol (Tier 1)
            dd_risk_mult, dd_min_conf, dd_sim_only, dd_desc = get_drawdown_adjustments(portfolio_dd)
            if dd_desc != "normal":
                print(f"  DRAWDOWN PROTOCOL: {dd_desc} (DD={portfolio_dd*100:.1f}%)")
            portfolio_halted = dd_sim_only or portfolio_dd > MAX_PORTFOLIO_DD
            if portfolio_halted:
                print(f"  PORTFOLIO HALTED — DD {portfolio_dd*100:.1f}% exceeds {MAX_PORTFOLIO_DD*100:.0f}% limit")

            # Check which bars just closed
            for tf in ALL_TFS:
                tf_minutes = {'15m': 15, '1h': 60, '4h': 240, '1d': 1440, '1w': 10080}[tf]
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

                # ── Stale Data Check (before feature computation) ──
                _stale_ohlcv = live_dal.get_ohlcv_window(tf, 5)
                _stale_ok, _stale_reason = check_stale_data(tf, _stale_ohlcv)
                if not _stale_ok:
                    print(f"  STALE DATA HALT: {_stale_reason}")
                    try:
                        log_rejected_trade(tf=tf, direction='UNKNOWN', timestamp=now.isoformat(),
                            price=None, confidence=None, reason=f'stale_data:{_stale_reason}',
                            meta_prob=None, confluence_info=None, feat_dict=None)
                    except Exception:
                        pass
                    continue

                # Compute features via shared library (handles OHLCV + HTF + esoteric)
                feat_names = features_list[tf]
                try:
                    feat_dict, feat_df = compute_features_live(tf, feat_names)
                except Exception as _feat_err:
                    # Feature computation crashed — log full error and skip this bar.
                    # NOT silently degraded — error is printed with traceback by compute_features_live.
                    print(f"  SKIPPING {tf} bar — feature computation failed: {_feat_err}")
                    continue
                if not feat_dict:
                    print(f"  No features returned for {tf}")
                    continue

                # Inject HMM features (Bug fix: were missing in live mode)
                hmm_feats = get_hmm_features(feat_dict)
                feat_dict.update(hmm_feats)

                # 8B.0: Compute regime BEFORE cross features.
                # Regime-aware DOY crosses (dw_N_B, dw_N_R, dw_N_S context tags) require the
                # regime label at compute() time. feat_dict is fully populated by this point.
                _price_rc = feat_dict.get('close', 0)
                _sma100_rc = feat_dict.get('sma_100', _price_rc)
                _ema50_slope_rc = feat_dict.get('ema50_slope', 0)
                _sma100_slope_rc = _ema50_slope_rc / 100.0 if _ema50_slope_rc else 0
                _regime_for_cross, _regime_idx_early = detect_regime(
                    _price_rc, _sma100_rc, _sma100_slope_rc, feat_dict=feat_dict)

                # 8B.1: Compute cross features via InferenceCrossComputer
                cross_values = None  # initialize — remains None if computation fails or TF has no computer
                if tf in cross_computers:
                    try:
                        # Extract day_of_year and regime for DOY crosses
                        _doy = now.timetuple().tm_yday
                        cross_values, cross_ms = cross_computers[tf].compute(feat_dict, day_of_year=_doy, regime=_regime_for_cross)
                        print(f"  Cross features: {len(cross_values)} computed in {cross_ms:.1f}ms (regime={_regime_for_cross})")
                    except Exception as e:
                        print(f"  WARNING: Cross feature computation failed for {tf}: {e}")

                price = feat_dict.get('close', 0)
                if price <= 0:
                    ohlcv_tmp = live_dal.get_ohlcv_window(tf, 2)
                    price = float(ohlcv_tmp['close'].iloc[-1]) if ohlcv_tmp is not None and len(ohlcv_tmp) > 0 else 0

                # ── Price Sanity Check ──
                # If price is still 0/negative/unreasonable after fallback, skip entirely
                if price <= 0:
                    print(f"  SKIP {tf} — price is {price} (zero/negative after fallback)")
                    try:
                        log_rejected_trade(tf=tf, direction='UNKNOWN', timestamp=now.isoformat(),
                            price=price, confidence=None, reason='invalid_price_zero',
                            meta_prob=None, confluence_info=None, feat_dict=feat_dict)
                    except Exception:
                        pass
                    continue
                # Sanity: BTC price should be between $100 and $10M (catches API garbage)
                if price < 100 or price > 10_000_000:
                    print(f"  SKIP {tf} — price ${price:,.2f} outside sanity range [$100, $10M]")
                    try:
                        log_rejected_trade(tf=tf, direction='UNKNOWN', timestamp=now.isoformat(),
                            price=price, confidence=None, reason=f'invalid_price_range:{price}',
                            meta_prob=None, confluence_info=None, feat_dict=feat_dict)
                    except Exception:
                        pass
                    continue

                # Build sparse CSR feature vector — skips 2.9M dict iterations.
                # Base features from feat_dict (~300 entries): stored as explicit values.
                #   NaN stored explicitly → LightGBM treats as missing (correct).
                #   0.0 NOT stored → structural zero = 0.0 (correct: value IS zero).
                # Cross features from cross_values (numpy array): only non-zeros stored.
                #   Structural zero in CSR = 0.0 (correct: cross didn't fire).
                # LightGBM Booster.predict() accepts scipy.sparse.csr_matrix directly.
                _idx_map = feat_name_to_idx.get(tf, {})
                _n_total_feats = len(feat_names)
                _sp_data = []
                _sp_cols = []
                _n_base_feats = 0
                _n_base_nan = 0

                # 1. Base features: iterate feat_dict (~300 items, fast)
                _cross_prefixes = ('dx_', 'ax_', 'ex2_', 'ax2_', 'ta2_', 'hod_', 'mx_', 'vx_', 'cross_')
                for fn, val in feat_dict.items():
                    cidx = _idx_map.get(fn)
                    if cidx is None:
                        continue
                    fval = float(val) if val is not None else np.nan
                    if np.isinf(fval):
                        fval = np.nan
                    if not fn.startswith(_cross_prefixes):
                        _n_base_feats += 1
                        if np.isnan(fval):
                            _n_base_nan += 1
                    # Store explicit only for non-zero or NaN (NaN = missing signal, must be explicit)
                    if fval != 0.0 or np.isnan(fval):
                        _sp_data.append(fval)
                        _sp_cols.append(cidx)

                # 2. Cross features: vectorized non-zero extraction from numpy array
                _n_cross_fired = 0
                _cv = cross_values if (cross_values is not None
                                       and tf in cross_feat_positions
                                       and len(cross_feat_positions[tf]) > 0) else None
                if _cv is not None:
                    _cpos = cross_feat_positions[tf]
                    _nz_mask = (_cv != 0) & (_cpos >= 0)
                    _nz_idx = np.nonzero(_nz_mask)[0]
                    if len(_nz_idx) > 0:
                        _n_cross_fired = len(_nz_idx)
                        _sp_data.extend(_cv[_nz_idx].astype(np.float32).tolist())
                        _sp_cols.extend(_cpos[_nz_idx].tolist())

                _sp_data_arr = np.array(_sp_data, dtype=np.float32)
                _sp_cols_arr = np.array(_sp_cols, dtype=np.int32)
                X = _csr_matrix(
                    (_sp_data_arr, _sp_cols_arr, np.array([0, len(_sp_data_arr)], dtype=np.int32)),
                    shape=(1, _n_total_feats)
                )

                # ── Data Quality Check: all-NaN feature vector ──
                # No explicit values at all = feat_dict empty and no cross features fired.
                if len(_sp_data_arr) == 0 or int(np.isnan(_sp_data_arr).sum()) == len(_sp_data_arr):
                    print(f"  SKIP {tf} — ALL {_n_total_feats} features are NaN (data quality failure)")
                    try:
                        log_rejected_trade(tf=tf, direction='UNKNOWN', timestamp=now.isoformat(),
                            price=price, confidence=None, reason='all_features_nan',
                            meta_prob=None, confluence_info=None, feat_dict=feat_dict)
                    except Exception:
                        pass
                    continue
                if _n_base_feats > 0 and _n_base_nan / _n_base_feats > 0.80:
                    print(f"  WARNING: {_n_base_nan}/{_n_base_feats} base features are NaN ({_n_base_nan/_n_base_feats*100:.0f}%)")

                # PHILOSOPHY GATE: Detect if cross feature computation failed at inference.
                # cross_values is None only when InferenceCrossComputer raised an exception.
                # All-zero cross_values is valid (no esoteric signal active this bar — not a failure).
                _n_cross_total = len(cross_feat_positions.get(tf, []))
                if _n_cross_total > 0 and cross_values is None:
                    print(f"  HALTED: Cross feature computation failed — cannot predict without the matrix")
                    print(f"  Cross features are MANDATORY. Skipping {tf} prediction this bar.")
                    continue  # Skip this TF — do NOT predict on base-only features

                # Predict (3-class softprob: SHORT=0, FLAT=1, LONG=2)
                # lleaves requires dense numpy; lgb.Booster accepts sparse directly.
                # Single-row .toarray() is cheap (~1ms even for 6M features).
                if tf in _lleaves_tfs:
                    raw_pred = models[tf].predict(X.toarray())
                else:
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
                conf_thresh = params.get('conf_thresh', LIVE_CONF_THRESH_FALLBACK)

                # Regime already computed before cross features (8B.0) — reuse result.
                regime, regime_idx = _regime_for_cross, _regime_idx_early
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
                    # === SCALE-IN CHECK (before dedup) ===
                    # If existing open trade in SAME direction AND profitable -> scale in
                    conn = sqlite3.connect(TRADES_DB, timeout=10)
                    conn.execute("BEGIN IMMEDIATE")
                    _si_row = conn.execute(
                        "SELECT id, direction, entry_price, stop_price, tp_price, "
                        "risk_pct, leverage, original_size, scaled_in "
                        "FROM trades WHERE tf=? AND status='open' LIMIT 1", (tf,)
                    ).fetchone()
                    if _si_row is not None:
                        _si_id, _si_dir, _si_entry, _si_sl, _si_tp = _si_row[:5]
                        _si_risk, _si_lev, _si_orig, _si_done = _si_row[5:]
                        _si_profitable = (_si_dir == 'LONG' and price > _si_entry) or \
                                         (_si_dir == 'SHORT' and price < _si_entry)
                        if _si_dir == direction and _si_profitable and not _si_done:
                            # Scale in: add 50% of original size at current price
                            _orig_risk = _si_orig if _si_orig and _si_orig > 0 else _si_risk
                            _add_risk = _orig_risk * 0.5
                            _new_risk = _si_risk + _add_risk
                            # Weighted average entry price
                            _w_old = _si_risk / _new_risk
                            _w_new = _add_risk / _new_risk
                            _new_entry = _si_entry * _w_old + price * _w_new
                            # Widen stop by 20% from new avg entry
                            _d_si = 1 if direction == 'LONG' else -1
                            _stop_dist = abs(_new_entry - _si_sl) * 1.2
                            _new_sl = _new_entry - _d_si * _stop_dist
                            # Preserve original RR for TP recalc
                            _old_rr = abs(_si_tp - _si_entry) / max(abs(_si_entry - _si_sl), 1e-8)
                            _new_tp = _new_entry + _d_si * _stop_dist * _old_rr
                            conn.execute(
                                "UPDATE trades SET entry_price=?, stop_price=?, tp_price=?, "
                                "risk_pct=?, scaled_in=1, original_size=? WHERE id=?",
                                (_new_entry, _new_sl, _new_tp, _new_risk, _orig_risk, _si_id)
                            )
                            conn.commit()
                            conn.close()
                            print(f"  >>> SCALE-IN {direction} {tf.upper()} @ ${price:,.0f} | "
                                  f"Avg entry=${_new_entry:,.0f} SL=${_new_sl:,.0f} TP=${_new_tp:,.0f} | "
                                  f"Risk {_si_risk:.1f}%->>{_new_risk:.1f}%")
                            continue  # updated existing trade, skip new entry
                        else:
                            # Existing trade not eligible for scale-in: dedup reject
                            _reason = 'duplicate'
                            if _si_dir != direction:
                                _reason = 'duplicate_opposite'
                            elif not _si_profitable:
                                _reason = 'duplicate_unprofitable'
                            elif _si_done:
                                _reason = 'duplicate_already_scaled'
                            print(f"  SKIP -- open {tf} trade ({_reason})")
                            try:
                                log_rejected_trade(tf=tf, direction=direction, timestamp=now.isoformat(),
                                    price=price, confidence=confidence, reason=_reason,
                                    meta_prob=None, confluence_info=None, feat_dict=feat_dict)
                            except Exception:
                                pass
                            conn.rollback()
                            conn.close()
                            continue

                    # No open trade for this TF -- proceed with new entry checks
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
                    p_win = confidence  # P(correct direction) — blended + meta-gated
                    b_ratio = rr  # avg_win / avg_loss ≈ reward:risk ratio
                    kelly_f = (p_win * b_ratio - (1 - p_win)) / max(b_ratio, 0.01)
                    kelly_f = max(kelly_f, 0)  # never negative (don't bet against yourself)
                    kelly_risk = base_risk * (1 + KELLY_SAFETY_FRACTION * kelly_f)
                    kelly_risk = min(kelly_risk, base_risk * KELLY_MAX_RISK_MULT)

                    # Confidence-scaled sizing: model conviction drives capital
                    from config import get_confidence_multiplier
                    conf_mult = get_confidence_multiplier(confidence)
                    kelly_risk = kelly_risk * conf_mult

                    # Drawdown scaling: reduce as DD deepens
                    dd_scale = max(0.0, 1.0 - DD_SCALE_STEEPNESS * portfolio_dd) if portfolio_dd < DD_HALT_THRESHOLD else 0.0
                    risk = kelly_risk * dd_scale

                    d = 1 if direction == 'LONG' else -1
                    # Apply per-TF slippage to entry price (matches backtesting_audit.py)
                    entry_slippage = TF_SLIPPAGE.get(tf, 0.0002)
                    if direction == 'LONG':     # LONG entry = buy higher
                        entry_price = price * (1 + entry_slippage)
                    else:                       # SHORT entry = sell lower
                        entry_price = price * (1 - entry_slippage)
                    stop_price = entry_price - d * stop_mult * atr
                    tp_price = entry_price + d * stop_mult * atr * rr
                    # Get account balance
                    balance = conn.execute("SELECT balance FROM account WHERE id=1").fetchone()[0]

                    # ── Circuit Breaker Checks (Tier 1) ──
                    cb_ok, cb_reason = check_circuit_breakers(now, portfolio_balance, portfolio_peak)
                    if not cb_ok:
                        print(f"  CIRCUIT BREAKER: {cb_reason}")
                        log_rejected_trade(tf=tf, direction=direction, timestamp=now.isoformat(),
                            price=price, confidence=confidence, reason=f'circuit_breaker:{cb_reason}',
                            meta_prob=None, confluence_info=None, feat_dict=feat_dict)
                        conn.rollback()
                        conn.close()
                        continue

                    # Max notional check (accounts for leverage)
                    risk = check_max_notional(risk, balance, leverage=lev)

                    # Drawdown protocol risk adjustment
                    risk = risk * dd_risk_mult
                    if dd_min_conf and confidence < dd_min_conf:
                        print(f"  DD PROTOCOL: conf {confidence:.3f} below DD threshold {dd_min_conf}")
                        log_rejected_trade(tf=tf, direction=direction, timestamp=now.isoformat(),
                            price=price, confidence=confidence, reason='dd_protocol_conf',
                            meta_prob=None, confluence_info=None, feat_dict=feat_dict)
                        conn.rollback()
                        conn.close()
                        continue

                    # Enforce hard leverage cap
                    lev = min(lev, RISK_LIMITS['max_leverage'])

                    # ── Max Open Risk Check (RISK_LIMITS enforcement) ──
                    # Sum risk_pct of all open trades + this new one. If exceeds max_open_risk_pct, reject.
                    try:
                        open_risks = conn.execute(
                            "SELECT COALESCE(SUM(risk_pct), 0) FROM trades WHERE status='open'"
                        ).fetchone()[0] / 100.0  # stored as percentage, convert to fraction
                        total_risk_if_added = open_risks + risk
                        if total_risk_if_added > RISK_LIMITS['max_open_risk_pct']:
                            print(f"  MAX OPEN RISK: {total_risk_if_added*100:.1f}% would exceed {RISK_LIMITS['max_open_risk_pct']*100:.0f}% limit")
                            log_rejected_trade(tf=tf, direction=direction, timestamp=now.isoformat(),
                                price=price, confidence=confidence, reason='max_open_risk',
                                meta_prob=None, confluence_info=None, feat_dict=feat_dict)
                            conn.rollback()
                            conn.close()
                            continue
                    except Exception as _risk_err:
                        print(f"  Warning: open risk check failed: {_risk_err}")

                    # Track order for rate limiting
                    _order_timestamps.append(now)

                    # Save ALL pruned features for full trade reasoning
                    key_feats = {}
                    for fn in feat_names:
                        val = feat_dict.get(fn, np.nan)
                        key_feats[fn] = round(val, 4) if not (isinstance(val, float) and np.isnan(val)) else None

                    cur = conn.execute("""INSERT INTO trades
                        (tf, direction, confidence, entry_price, entry_time, stop_price, tp_price,
                         regime, leverage, risk_pct, features_json, status, original_size)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)""",
                        (tf, direction, confidence, entry_price, now.isoformat(),
                         stop_price, tp_price, regime, lev, risk * 100,
                         json.dumps(key_feats), risk * 100))
                    trade_id = cur.lastrowid
                    conn.commit()
                    conn.close()

                    # --- Journal: log trade snapshot ---
                    try:
                        log_trade_snapshot(
                            trade_id=trade_id, tf=tf, direction=direction,
                            entry_time=now.isoformat(), entry_price=entry_price,
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
                                'confidence_multiplier': conf_mult,
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
                        'entry_price': float(entry_price),
                        'raw_price': float(price),
                        'slippage_pct': float(entry_slippage * 100),
                        'stop_loss': float(stop_price),
                        'take_profit': float(tp_price),
                    }
                    try:
                        with open(f'{DB_DIR}/prediction_cache.json', 'w') as f:
                            json.dump(prediction, f, indent=2)
                    except Exception:
                        pass

                    print(f"  >>> {direction} {tf.upper()} @ ${entry_price:,.0f} (slip={entry_slippage*100:.3f}%) | Conf={confidence:.1%} (size={conf_mult:.1f}x) "
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
                    # 8B.2 + 8B.9: Use trade's own TF for bars_held (not outer loop's tf_minutes)
                    trade_tf_minutes = {'15m': 15, '1h': 60, '4h': 240, '1d': 1440, '1w': 10080}[ttf]
                    bars_held = int((now - entry_dt).total_seconds() / (trade_tf_minutes * 60))
                    max_h = ga_params.get(ttf, {}).get('max_hold', 4)

                    # 8B.4: Update best price tracking for trailing stop
                    if tid not in _trade_best_prices:
                        _trade_best_prices[tid] = entry
                    if tdir == 'LONG':
                        _trade_best_prices[tid] = max(_trade_best_prices[tid], price)
                    else:
                        _trade_best_prices[tid] = min(_trade_best_prices[tid], price)

                    # 8B.6: Classify trade type based on bars_held
                    tf_thresholds = TRADE_THRESHOLDS.get(ttf, {'scalp': 1, 'day': 24, 'swing': 72})
                    if bars_held <= tf_thresholds['scalp']:
                        trade_type = 'scalp'
                    elif bars_held <= tf_thresholds['day']:
                        trade_type = 'day_trade'
                    elif bars_held <= tf_thresholds['swing']:
                        trade_type = 'swing'
                    else:
                        trade_type = 'position'
                    trade_type_params = TRADE_TYPE_PARAMS.get(trade_type, TRADE_TYPE_PARAMS['day_trade'])

                    # 8B.4: Trailing stop check — uses trade_type_params trail_mult
                    trail_sl_hit = False
                    if _trade_best_prices.get(tid) is not None:
                        best = _trade_best_prices[tid]
                        atr_exit = ga_params.get(ttf, {}).get('stop_atr', 1.0)
                        atr_val = price * 0.01  # fallback 1% ATR estimate
                        trail_dist = atr_exit * atr_val * trade_type_params.get('trail_mult', 2.0)
                        if tdir == 'LONG' and best > entry and price < best - trail_dist:
                            trail_sl_hit = True
                        elif tdir == 'SHORT' and best < entry and price > best + trail_dist:
                            trail_sl_hit = True

                    sl_hit = (tdir == 'LONG' and price <= sl) or (tdir == 'SHORT' and price >= sl)
                    tp_hit = (tdir == 'LONG' and price >= tp) or (tdir == 'SHORT' and price <= tp)
                    time_exit = bars_held >= max_h

                    # 8B.5 + 8C.2: Scale-out / Partial TP — first TP hit closes 50%,
                    # moves SL to breakeven, lets remainder trail with ATR
                    if tp_hit and tid not in _trade_scaled_out:
                        partial_pct = trade_type_params.get('partial_tp_pct', 0.5)
                        if partial_pct > 0 and partial_pct < 1.0:
                            # Take partial profit — reduce risk_pct, move SL to breakeven
                            remaining_risk = trisk * (1 - partial_pct)
                            d_partial = 1 if tdir == 'LONG' else -1
                            partial_pnl_per_unit = (price - entry) / entry * d_partial * tlev - FEE_RATE * tlev
                            balance = conn.execute("SELECT balance FROM account WHERE id=1").fetchone()[0]
                            partial_pnl_val = balance * (trisk / 100) * partial_pct * partial_pnl_per_unit
                            # Credit partial profit to balance
                            conn.execute("UPDATE account SET balance=balance+? WHERE id=1", (partial_pnl_val,))
                            # Update trade: reduce risk_pct, move SL to breakeven, record scale-out
                            conn.execute(
                                "UPDATE trades SET risk_pct=?, stop_price=?, scaled_out=1, partial_pnl=? WHERE id=?",
                                (remaining_risk, entry, partial_pnl_val, tid))
                            trisk = remaining_risk  # update local var for rest of exit logic
                            sl = entry  # SL now at breakeven
                            _trade_scaled_out.add(tid)
                            print(f"  SCALE-OUT ({partial_pct*100:.0f}%): {tdir} {ttf.upper()} | "
                                  f"Partial PnL=${partial_pnl_val:+.2f} | SL->breakeven | remainder trails ATR")
                            tp_hit = False  # Don't fully close — let remaining ride
                            continue  # Skip to next trade, remaining position still open

                    if sl_hit or tp_hit or time_exit or trail_sl_hit:
                        d = 1 if tdir == 'LONG' else -1
                        # Apply per-TF slippage to exit price (matches backtesting_audit.py)
                        exit_slippage = TF_SLIPPAGE.get(ttf, 0.0002)
                        if tdir == 'LONG':    # LONG exit = sell lower
                            slipped_exit = price * (1 - exit_slippage)
                        else:                 # SHORT exit = buy higher
                            slipped_exit = price * (1 + exit_slippage)
                        pchange = (slipped_exit - entry) / entry * d
                        gross = pchange * tlev
                        fee = FEE_RATE * tlev
                        net = gross - fee
                        balance = conn.execute("SELECT balance FROM account WHERE id=1").fetchone()[0]
                        pnl = balance * (trisk / 100) * net
                        # 8C.2: If this trade was scaled out, add partial PnL to total trade PnL
                        _prior_partial = conn.execute(
                            "SELECT COALESCE(partial_pnl, 0) FROM trades WHERE id=?", (tid,)
                        ).fetchone()[0]
                        total_trade_pnl = pnl + _prior_partial
                        pnl_pct = total_trade_pnl / balance * 100 if balance > 0 else 0
                        # Track PnL for circuit breaker sanity check
                        _recent_pnls.append(pnl_pct)
                        reason = 'SL' if sl_hit else ('TRAIL' if trail_sl_hit else ('TP' if tp_hit else 'TIME'))
                        if _prior_partial != 0:
                            reason = reason + '_REMAINDER'

                        # 8B.4: Clean up trailing stop tracking
                        _trade_best_prices.pop(tid, None)
                        _trade_scaled_out.discard(tid)

                        new_balance = balance + pnl
                        peak = conn.execute("SELECT peak_balance FROM account WHERE id=1").fetchone()[0]
                        new_peak = max(peak, new_balance)
                        dd = (new_peak - new_balance) / new_peak if new_peak > 0 else 0
                        wins = conn.execute("SELECT wins FROM account WHERE id=1").fetchone()[0]
                        losses = conn.execute("SELECT losses FROM account WHERE id=1").fetchone()[0]
                        total = conn.execute("SELECT total_trades FROM account WHERE id=1").fetchone()[0]

                        conn.execute("""UPDATE trades SET exit_price=?, exit_time=?, pnl=?, pnl_pct=?,
                            bars_held=?, exit_reason=?, status='closed' WHERE id=?""",
                            (slipped_exit, now.isoformat(), total_trade_pnl, pnl_pct, bars_held, reason, tid))

                        conn.execute("""UPDATE account SET balance=?, peak_balance=?,
                            total_trades=?, wins=?, losses=?, max_dd=?, updated_at=? WHERE id=1""",
                            (new_balance, new_peak, total + 1,
                             wins + (1 if total_trade_pnl > 0 else 0),
                             losses + (1 if total_trade_pnl <= 0 else 0),
                             max(dd, conn.execute("SELECT max_dd FROM account WHERE id=1").fetchone()[0]),
                             now.isoformat()))

                        conn.execute("INSERT INTO equity_curve (timestamp, balance, dd_pct) VALUES (?, ?, ?)",
                                     (now.isoformat(), new_balance, dd * 100))

                        # ── Balance Reconciliation ──
                        # Cross-check: account balance should equal starting + sum of all closed PnL
                        try:
                            _sum_pnl = conn.execute("SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE status='closed'").fetchone()[0]
                            _expected_balance = LIVE_STARTING_BALANCE + _sum_pnl
                            _actual_balance = new_balance
                            _drift = abs(_actual_balance - _expected_balance)
                            if _drift > 0.01:  # more than 1 cent drift = rounding accumulation
                                print(f"  RECONCILIATION DRIFT: actual=${_actual_balance:.4f} expected=${_expected_balance:.4f} drift=${_drift:.4f}")
                                if _drift > 1.0:  # >$1 drift is suspicious
                                    print(f"  *** WARNING: significant balance drift detected — investigate ***")
                        except Exception as _recon_err:
                            print(f"  [reconciliation] check failed: {_recon_err}")

                        # 8B.6 + 8C.2: Log trade type + scale-out info
                        _so_note = f" (incl partial=${_prior_partial:+.2f})" if _prior_partial != 0 else ""
                        print(f"  <<< CLOSED {tdir} {ttf.upper()} @ ${price:,.0f} | PnL=${total_trade_pnl:+.2f} ({pnl_pct:+.1f}%) [{reason}]{_so_note} type={trade_type} ({bars_held} bars)")

                        # --- Journal: log trade outcome ---
                        try:
                            log_trade_outcome(
                                trade_id=tid, tf=ttf, direction=tdir,
                                entry_price=entry, exit_price=slipped_exit,
                                pnl=total_trade_pnl, pnl_pct=pnl_pct, exit_reason=reason,
                                bars_held=bars_held,
                                feat_dict_exit=feat_dict if ttf == tf else None,
                                predicted_dir=tdir,
                                confidence=None,
                            )
                        except Exception as e:
                            print(f"  [journal] outcome error: {e}")

                        # Update per-TF pool tracking (remainder PnL only; partial already credited)
                        if ttf in tf_pools:
                            tf_pools[ttf]['balance'] += pnl  # remainder portion only
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
                            m_tf_minutes = {'15m': 15, '1h': 60, '4h': 240, '1d': 1440, '1w': 10080}.get(m_tf, 60)
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
