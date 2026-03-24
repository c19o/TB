#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
trade_journal.py — Trade Journal Database & Logging
=====================================================
Core logging module for the Savage22 self-learning system.
Creates/manages trade_journal.db with 4 tables:
  - trade_snapshots  (full state at entry)
  - price_path       (bar-by-bar MAE/MFE tracking)
  - trade_outcomes   (post-close analysis)
  - rejected_trades  (signals we did NOT take)

Called by live_trader.py at entry, exit, bar-monitor, and rejection points.
All timestamps UTC. All DB ops use parameterized queries.
"""

import os
import json
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from contextlib import contextmanager

from config import PROJECT_DIR, TRADES_DB

DB_DIR = PROJECT_DIR
TRADE_JOURNAL_DB = os.path.join(DB_DIR, "trade_journal.db")


# ============================================================
# DATABASE INIT
# ============================================================

def _init_journal_db(conn):
    """Create all journal tables if they don't exist."""

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS trade_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id INTEGER NOT NULL,
            tf TEXT NOT NULL,
            direction TEXT NOT NULL,
            entry_time TEXT NOT NULL,
            entry_price REAL NOT NULL,

            -- Model outputs (BEFORE blending/gating)
            xgb_prob_long REAL,
            xgb_prob_flat REAL,
            xgb_prob_short REAL,
            lstm_prob REAL,
            meta_prob REAL,
            blended_confidence REAL,

            -- Regime state
            regime TEXT,
            regime_idx INTEGER,
            hmm_bull_prob REAL,
            hmm_bear_prob REAL,
            hmm_neutral_prob REAL,
            hmm_state INTEGER,

            -- Position sizing inputs
            kelly_fraction REAL,
            base_risk_pct REAL,
            final_risk_pct REAL,
            leverage_used REAL,
            dd_scale REAL,
            portfolio_dd_pct REAL,
            tf_pool_dd_pct REAL,

            -- Trade params
            stop_atr_mult REAL,
            rr_ratio REAL,
            max_hold_bars INTEGER,
            atr_14 REAL,
            confluence_parent_tf TEXT,
            confluence_parent_dir INTEGER,
            confluence_scale REAL,

            -- Full feature vector (compressed JSON)
            features_json TEXT NOT NULL,

            -- Key signal states at entry (extracted for fast querying)
            moon_phase REAL,
            mercury_retro INTEGER,
            kp_index REAL,
            gematria_sum_price INTEGER,
            numerology_date_sum INTEGER,
            astro_dominant_sign TEXT,

            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_snap_trade ON trade_snapshots(trade_id);
        CREATE INDEX IF NOT EXISTS idx_snap_tf ON trade_snapshots(tf);
        CREATE INDEX IF NOT EXISTS idx_snap_time ON trade_snapshots(entry_time);

        CREATE TABLE IF NOT EXISTS price_path (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id INTEGER NOT NULL,
            bar_num INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL,
            volume REAL,

            -- Running metrics
            unrealized_pnl_pct REAL,
            max_adverse_excursion REAL,
            max_favorable_excursion REAL,
            distance_to_stop_pct REAL,
            distance_to_tp_pct REAL,

            UNIQUE(trade_id, bar_num)
        );

        CREATE INDEX IF NOT EXISTS idx_path_trade ON price_path(trade_id);

        CREATE TABLE IF NOT EXISTS trade_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id INTEGER NOT NULL UNIQUE,
            tf TEXT NOT NULL,
            direction TEXT NOT NULL,

            -- Outcome
            pnl REAL,
            pnl_pct REAL,
            exit_reason TEXT,
            bars_held INTEGER,

            -- Excursion analysis
            max_adverse_excursion REAL,
            max_favorable_excursion REAL,
            mae_bar INTEGER,
            mfe_bar INTEGER,
            mfe_before_mae INTEGER,

            -- Timing analysis
            optimal_exit_bar INTEGER,
            optimal_exit_pnl REAL,
            timing_efficiency REAL,
            entry_timing_error_bars INTEGER,

            -- Features at exit time
            exit_features_json TEXT,

            -- Prediction correctness
            predicted_direction TEXT,
            actual_direction TEXT,
            prediction_correct INTEGER,
            predicted_confidence REAL,

            -- Signal breakdown
            active_signals_json TEXT,
            correct_signals_json TEXT,
            incorrect_signals_json TEXT,

            -- Post-trade price action
            price_1bar_after REAL,
            price_5bars_after REAL,
            price_10bars_after REAL,
            continued_in_direction INTEGER,

            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_outcome_tf ON trade_outcomes(tf);
        CREATE INDEX IF NOT EXISTS idx_outcome_time ON trade_outcomes(created_at);

        CREATE TABLE IF NOT EXISTS rejected_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tf TEXT NOT NULL,
            direction TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            price REAL,
            confidence REAL,

            -- Why it was rejected
            rejection_reason TEXT,
            meta_prob REAL,
            confluence_parent TEXT,
            confluence_parent_dir INTEGER,

            -- What would have happened (filled in post-hoc)
            hypothetical_exit_price REAL,
            hypothetical_pnl_pct REAL,
            hypothetical_exit_reason TEXT,
            was_correct_rejection INTEGER,

            features_json TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_rejected_tf ON rejected_trades(tf);
        CREATE INDEX IF NOT EXISTS idx_rejected_reason ON rejected_trades(rejection_reason);
    """)


@contextmanager
def _journal_conn(readonly=False):
    """Context manager for journal DB connections with auto-init."""
    conn = sqlite3.connect(TRADE_JOURNAL_DB, timeout=15)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    _init_journal_db(conn)
    try:
        yield conn
        if not readonly:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ============================================================
# SIGNAL CATEGORY EXTRACTION
# ============================================================

SIGNAL_CATEGORIES = {
    'gematria': ['gem_', 'gematria_', 'ordinal_', 'cipher_', 'hebrew_', 'english_ext_'],
    'numerology': ['num_', 'angel_', 'master_', 'pythagorean_', 'chaldean_'],
    'astrology': ['astro_', 'moon_', 'mercury_', 'planet_', 'zodiac_', 'lunar_',
                  'eclipse_', 'retrograde_'],
    'space_weather': ['kp_', 'solar_', 'schumann_', 'geomag_', 'sw_', 'dst_',
                      'solar_wind_', 'sunspot_'],
    'ta': ['rsi', 'macd', 'sma_', 'ema_', 'atr_', 'bollinger_', 'volume_',
           'adx_', 'obv_', 'vwap_', 'stoch_', 'cci_', 'mfi_', 'williams_'],
    'sports': ['sport_', 'nba_', 'nfl_', 'mlb_', 'nhl_', 'soccer_'],
    'sentiment': ['sent_', 'sentiment_', 'fear_', 'greed_', 'news_', 'tweet_',
                  'headline_', 'social_'],
}


def _categorize_feature(feature_name):
    """Return the signal category for a feature name."""
    fname_lower = feature_name.lower()
    for category, prefixes in SIGNAL_CATEGORIES.items():
        if any(fname_lower.startswith(p) for p in prefixes):
            return category
    return 'other'


def _extract_active_signals(feat_dict):
    """Extract which signals are active (non-zero, non-NaN) grouped by category."""
    active = {}
    for fname, val in feat_dict.items():
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        if val == 0:
            continue
        cat = _categorize_feature(fname)
        if cat not in active:
            active[cat] = []
        active[cat].append(fname)
    return active


def _extract_key_signals(feat_dict):
    """Extract key esoteric signal values from feature dict for fast querying."""
    def _get(prefix_list, default=None):
        for prefix in prefix_list:
            for k, v in feat_dict.items():
                if k.lower().startswith(prefix.lower()) and v is not None:
                    if isinstance(v, float) and np.isnan(v):
                        continue
                    return v
        return default

    return {
        'moon_phase': _get(['moon_phase', 'lunar_phase']),
        'mercury_retro': _get(['mercury_retro', 'retrograde_mercury']),
        'kp_index': _get(['kp_', 'kp_index']),
        'gematria_sum_price': _get(['gematria_sum', 'gem_sum']),
        'numerology_date_sum': _get(['num_date', 'numerology_date']),
        'astro_dominant_sign': _get(['astro_dominant', 'zodiac_sign']),
    }


# ============================================================
# LOGGING FUNCTIONS (called by live_trader.py)
# ============================================================

def log_trade_snapshot(trade_id, tf, direction, entry_time, entry_price,
                       xgb_probs=None, lstm_prob=None, meta_prob=None,
                       blended_conf=None, regime_info=None, sizing_info=None,
                       trade_params=None, feat_dict=None):
    """
    Log full state at trade entry time.

    Parameters
    ----------
    trade_id : int
        FK to trades.db trades.id
    tf : str
        Timeframe (e.g. '1h', '4h')
    direction : str
        'LONG' or 'SHORT'
    entry_time : datetime or str
        UTC entry timestamp
    entry_price : float
        Entry price
    xgb_probs : dict, optional
        {'long': float, 'flat': float, 'short': float}
    lstm_prob : float, optional
        LSTM probability (before blending)
    meta_prob : float, optional
        Meta-labeling probability
    blended_conf : float, optional
        Final blended confidence
    regime_info : dict, optional
        {'regime': str, 'regime_idx': int, 'hmm_bull': float,
         'hmm_bear': float, 'hmm_neutral': float, 'hmm_state': int}
    sizing_info : dict, optional
        {'kelly_fraction': float, 'base_risk_pct': float,
         'final_risk_pct': float, 'leverage': float,
         'dd_scale': float, 'portfolio_dd': float, 'tf_pool_dd': float}
    trade_params : dict, optional
        {'stop_atr_mult': float, 'rr_ratio': float, 'max_hold': int,
         'atr_14': float, 'confluence_parent_tf': str,
         'confluence_parent_dir': int, 'confluence_scale': float}
    feat_dict : dict, optional
        Full feature vector at entry time
    """
    if isinstance(entry_time, datetime):
        entry_time = entry_time.isoformat()

    xgb_probs = xgb_probs or {}
    regime_info = regime_info or {}
    sizing_info = sizing_info or {}
    trade_params = trade_params or {}
    feat_dict = feat_dict or {}

    # Extract key signals for fast querying
    key_sigs = _extract_key_signals(feat_dict)

    # Serialize feature dict
    features_json = json.dumps(feat_dict, default=str)

    with _journal_conn() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("""
            INSERT INTO trade_snapshots (
                trade_id, tf, direction, entry_time, entry_price,
                xgb_prob_long, xgb_prob_flat, xgb_prob_short,
                lstm_prob, meta_prob, blended_confidence,
                regime, regime_idx, hmm_bull_prob, hmm_bear_prob,
                hmm_neutral_prob, hmm_state,
                kelly_fraction, base_risk_pct, final_risk_pct,
                leverage_used, dd_scale, portfolio_dd_pct, tf_pool_dd_pct,
                stop_atr_mult, rr_ratio, max_hold_bars, atr_14,
                confluence_parent_tf, confluence_parent_dir, confluence_scale,
                features_json,
                moon_phase, mercury_retro, kp_index,
                gematria_sum_price, numerology_date_sum, astro_dominant_sign
            ) VALUES (
                ?, ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?,
                ?,
                ?, ?, ?,
                ?, ?, ?
            )
        """, (
            trade_id, tf, direction, entry_time, entry_price,
            xgb_probs.get('long'), xgb_probs.get('flat'), xgb_probs.get('short'),
            lstm_prob, meta_prob, blended_conf,
            regime_info.get('regime'), regime_info.get('regime_idx'),
            regime_info.get('hmm_bull'), regime_info.get('hmm_bear'),
            regime_info.get('hmm_neutral'), regime_info.get('hmm_state'),
            sizing_info.get('kelly_fraction'), sizing_info.get('base_risk_pct'),
            sizing_info.get('final_risk_pct'), sizing_info.get('leverage'),
            sizing_info.get('dd_scale'), sizing_info.get('portfolio_dd'),
            sizing_info.get('tf_pool_dd'),
            trade_params.get('stop_atr_mult'), trade_params.get('rr_ratio'),
            trade_params.get('max_hold'), trade_params.get('atr_14'),
            trade_params.get('confluence_parent_tf'),
            trade_params.get('confluence_parent_dir'),
            trade_params.get('confluence_scale'),
            features_json,
            key_sigs.get('moon_phase'), key_sigs.get('mercury_retro'),
            key_sigs.get('kp_index'), key_sigs.get('gematria_sum_price'),
            key_sigs.get('numerology_date_sum'), key_sigs.get('astro_dominant_sign'),
        ))


def log_price_path_bar(trade_id, bar_num, timestamp, ohlcv,
                        entry_price, direction, sl, tp):
    """
    Log a single bar of price path for an open trade.

    Parameters
    ----------
    trade_id : int
        FK to trades.db trades.id
    bar_num : int
        0 = entry bar, 1 = next bar, etc.
    timestamp : datetime or str
        Bar timestamp (UTC)
    ohlcv : dict
        {'open': float, 'high': float, 'low': float, 'close': float, 'volume': float}
    entry_price : float
        Trade entry price
    direction : str
        'LONG' or 'SHORT'
    sl : float
        Stop-loss price
    tp : float
        Take-profit price
    """
    if isinstance(timestamp, datetime):
        timestamp = timestamp.isoformat()

    d = 1 if direction == 'LONG' else -1
    close = ohlcv.get('close', entry_price)
    high = ohlcv.get('high', close)
    low = ohlcv.get('low', close)

    # Compute unrealized PnL based on direction
    unrealized_pnl = (close - entry_price) / entry_price * d

    # Compute worst/best within this bar
    bar_worst = (low - entry_price) / entry_price * d if d == 1 else \
                (high - entry_price) / entry_price * d
    bar_best = (high - entry_price) / entry_price * d if d == 1 else \
               (low - entry_price) / entry_price * d

    with _journal_conn() as conn:
        # Get running MAE/MFE from previous bars
        prev = conn.execute(
            "SELECT MIN(max_adverse_excursion), MAX(max_favorable_excursion) "
            "FROM price_path WHERE trade_id = ?", (trade_id,)
        ).fetchone()

        prev_mae = prev[0] if prev[0] is not None else 0.0
        prev_mfe = prev[1] if prev[1] is not None else 0.0

        mae = min(prev_mae, bar_worst, unrealized_pnl)
        mfe = max(prev_mfe, bar_best, unrealized_pnl)

        # Distance to stop/TP as percentage (0 = at level, 1 = at entry)
        stop_range = abs(entry_price - sl) if sl else 1.0
        tp_range = abs(tp - entry_price) if tp else 1.0

        if d == 1:  # LONG
            dist_to_stop = (close - sl) / stop_range if stop_range > 0 else 1.0
            dist_to_tp = (close - entry_price) / tp_range if tp_range > 0 else 0.0
        else:  # SHORT
            dist_to_stop = (sl - close) / stop_range if stop_range > 0 else 1.0
            dist_to_tp = (entry_price - close) / tp_range if tp_range > 0 else 0.0

        conn.execute("""
            INSERT OR REPLACE INTO price_path (
                trade_id, bar_num, timestamp, open, high, low, close, volume,
                unrealized_pnl_pct, max_adverse_excursion, max_favorable_excursion,
                distance_to_stop_pct, distance_to_tp_pct
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade_id, bar_num, timestamp,
            ohlcv.get('open', 0.0), high, low, close, ohlcv.get('volume', 0.0),
            unrealized_pnl, mae, mfe,
            dist_to_stop, dist_to_tp,
        ))


def log_trade_outcome(trade_id, tf, direction, entry_price, exit_price,
                       pnl, pnl_pct, exit_reason, bars_held,
                       feat_dict_exit=None, predicted_dir=None,
                       confidence=None, signals_analysis=None):
    """
    Log trade outcome after exit. Computes excursion analysis from price_path.

    Parameters
    ----------
    trade_id : int
        FK to trades.db trades.id
    tf : str
        Timeframe
    direction : str
        'LONG' or 'SHORT'
    entry_price : float
    exit_price : float
    pnl : float
        Dollar PnL
    pnl_pct : float
        Percentage PnL
    exit_reason : str
        'SL', 'TP', 'TIME', 'MANUAL'
    bars_held : int
    feat_dict_exit : dict, optional
        Feature vector at exit time
    predicted_dir : str, optional
        Direction we predicted ('LONG' or 'SHORT')
    confidence : float, optional
        Confidence at entry
    signals_analysis : dict, optional
        Pre-computed signal analysis (active/correct/incorrect signals)
    """
    d = 1 if direction == 'LONG' else -1
    actual_move = (exit_price - entry_price) * d
    actual_direction = 'LONG' if exit_price > entry_price else 'SHORT'
    prediction_correct = 1 if direction == actual_direction else 0

    # Serialize exit features
    exit_features_json = json.dumps(feat_dict_exit, default=str) if feat_dict_exit else None

    # Get signal analysis from snapshot's features if not provided
    if signals_analysis is None:
        signals_analysis = {}

    active_signals = signals_analysis.get('active_signals', {})
    correct_signals = signals_analysis.get('correct_signals', {})
    incorrect_signals = signals_analysis.get('incorrect_signals', {})

    with _journal_conn() as conn:
        # Compute excursion analysis from price_path
        excursion = compute_post_trade_analysis(trade_id, conn=conn)

        conn.execute("""
            INSERT OR REPLACE INTO trade_outcomes (
                trade_id, tf, direction,
                pnl, pnl_pct, exit_reason, bars_held,
                max_adverse_excursion, max_favorable_excursion,
                mae_bar, mfe_bar, mfe_before_mae,
                optimal_exit_bar, optimal_exit_pnl, timing_efficiency,
                entry_timing_error_bars,
                exit_features_json,
                predicted_direction, actual_direction, prediction_correct,
                predicted_confidence,
                active_signals_json, correct_signals_json, incorrect_signals_json,
                price_1bar_after, price_5bars_after, price_10bars_after,
                continued_in_direction
            ) VALUES (
                ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?,
                ?,
                ?, ?, ?,
                ?,
                ?, ?, ?,
                ?, ?, ?,
                ?
            )
        """, (
            trade_id, tf, direction,
            pnl, pnl_pct, exit_reason, bars_held,
            excursion.get('mae'), excursion.get('mfe'),
            excursion.get('mae_bar'), excursion.get('mfe_bar'),
            excursion.get('mfe_before_mae'),
            excursion.get('optimal_exit_bar'), excursion.get('optimal_exit_pnl'),
            excursion.get('timing_efficiency'),
            excursion.get('entry_timing_error_bars'),
            exit_features_json,
            predicted_dir or direction, actual_direction, prediction_correct,
            confidence,
            json.dumps(active_signals, default=str),
            json.dumps(correct_signals, default=str),
            json.dumps(incorrect_signals, default=str),
            None, None, None,  # post-exit prices filled later
            None,  # continued_in_direction filled later
        ))


def log_rejected_trade(tf, direction, timestamp, price, confidence,
                        reason, meta_prob=None, confluence_info=None,
                        feat_dict=None):
    """
    Log a trade signal that was blocked/rejected.

    Parameters
    ----------
    tf : str
        Timeframe
    direction : str
        'LONG' or 'SHORT'
    timestamp : datetime or str
        UTC timestamp of rejection
    price : float
        Current price when rejected
    confidence : float
        Model confidence at rejection
    reason : str
        Rejection reason: 'meta_gate', 'confluence_block', 'dd_halt',
        'below_threshold', 'tf_halted', 'duplicate'
    meta_prob : float, optional
        Meta-label probability (if meta_gate rejection)
    confluence_info : dict, optional
        {'parent_tf': str, 'parent_dir': int}
    feat_dict : dict, optional
        Feature vector at rejection time
    """
    if isinstance(timestamp, datetime):
        timestamp = timestamp.isoformat()

    confluence_info = confluence_info or {}
    features_json = json.dumps(feat_dict, default=str) if feat_dict else None

    with _journal_conn() as conn:
        conn.execute("""
            INSERT INTO rejected_trades (
                tf, direction, timestamp, price, confidence,
                rejection_reason, meta_prob,
                confluence_parent, confluence_parent_dir,
                features_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            tf, direction, timestamp, price, confidence,
            reason, meta_prob,
            confluence_info.get('parent_tf'),
            confluence_info.get('parent_dir'),
            features_json,
        ))


# ============================================================
# POST-TRADE ANALYSIS
# ============================================================

def compute_post_trade_analysis(trade_id, conn=None):
    """
    Compute MAE/MFE/timing efficiency from price_path after trade closes.

    Parameters
    ----------
    trade_id : int
        The trade to analyze
    conn : sqlite3.Connection, optional
        Reuse existing connection if provided

    Returns
    -------
    dict
        Excursion and timing analysis results
    """
    close_conn = False
    if conn is None:
        conn = sqlite3.connect(TRADE_JOURNAL_DB, timeout=15)
        close_conn = True

    try:
        rows = conn.execute("""
            SELECT bar_num, unrealized_pnl_pct, max_adverse_excursion,
                   max_favorable_excursion
            FROM price_path WHERE trade_id = ?
            ORDER BY bar_num
        """, (trade_id,)).fetchall()

        if not rows:
            return {
                'mae': None, 'mfe': None,
                'mae_bar': None, 'mfe_bar': None,
                'mfe_before_mae': None,
                'optimal_exit_bar': None, 'optimal_exit_pnl': None,
                'timing_efficiency': None,
                'entry_timing_error_bars': None,
            }

        bar_nums = [r[0] for r in rows]
        pnls = [r[1] for r in rows]
        maes = [r[2] for r in rows]
        mfes = [r[3] for r in rows]

        # Overall MAE/MFE
        overall_mae = min(maes) if maes else None
        overall_mfe = max(mfes) if mfes else None

        # Which bar had worst/best
        mae_bar = bar_nums[maes.index(min(maes))] if maes else None
        mfe_bar = bar_nums[mfes.index(max(mfes))] if mfes else None

        # Did best come before worst? (indicates reversal after being right)
        mfe_before_mae = 1 if (mfe_bar is not None and mae_bar is not None
                               and mfe_bar < mae_bar) else 0

        # Optimal exit: bar with highest unrealized PnL
        if pnls:
            best_pnl = max(pnls)
            optimal_bar = bar_nums[pnls.index(best_pnl)]
            actual_pnl = pnls[-1] if pnls else 0

            # Timing efficiency: actual / optimal (capped at 1.0+)
            timing_eff = actual_pnl / best_pnl if best_pnl > 0 else (
                1.0 if actual_pnl >= 0 else 0.0
            )
        else:
            best_pnl = None
            optimal_bar = None
            timing_eff = None

        # Entry timing error: bars until price first moves in our direction
        entry_error = 0
        for i, p in enumerate(pnls):
            if p > 0:
                entry_error = i
                break
        else:
            entry_error = len(pnls)  # never went positive

        return {
            'mae': overall_mae,
            'mfe': overall_mfe,
            'mae_bar': mae_bar,
            'mfe_bar': mfe_bar,
            'mfe_before_mae': mfe_before_mae,
            'optimal_exit_bar': optimal_bar,
            'optimal_exit_pnl': best_pnl,
            'timing_efficiency': timing_eff,
            'entry_timing_error_bars': entry_error,
        }
    finally:
        if close_conn:
            conn.close()


def fill_hypothetical_outcomes(hours_lookback=24):
    """
    For rejected trades within lookback window, compute what would have happened.
    Uses historical price data from btc_prices.db.

    This is called by hypothetical_tracker.py.

    Parameters
    ----------
    hours_lookback : int
        Only process rejected trades from the last N hours

    Returns
    -------
    int
        Number of rejected trades updated
    """
    from data_access import OfflineDataLoader

    cutoff = datetime.now(timezone.utc).isoformat()
    dal = OfflineDataLoader(DB_DIR)

    updated = 0

    with _journal_conn() as conn:
        # Get unfilled rejected trades
        rejects = conn.execute("""
            SELECT id, tf, direction, timestamp, price, confidence, features_json
            FROM rejected_trades
            WHERE hypothetical_exit_price IS NULL
              AND datetime(timestamp) >= datetime(?, '-' || ? || ' hours')
            ORDER BY timestamp
        """, (cutoff, hours_lookback)).fetchall()

        if not rejects:
            return 0

        for rej in rejects:
            rej_id, tf, direction, ts, price, conf, feat_json = rej

            try:
                # Load default trade params from optimizer config
                config_path = os.path.join(DB_DIR, 'optimal_config.json')
                if os.path.exists(config_path):
                    with open(config_path) as f:
                        config = json.load(f)
                    tf_config = config.get(tf, config.get('default', {}))
                else:
                    tf_config = {}

                stop_atr_mult = tf_config.get('stop_atr', 1.0)
                rr = tf_config.get('rr', 2.0)
                max_hold = tf_config.get('max_hold', 4)

                # Load price data after rejection time
                ohlcv = dal.load_ohlcv(tf)
                if ohlcv is None or ohlcv.empty:
                    continue

                rej_dt = pd.to_datetime(ts, utc=True)
                future_bars = ohlcv[ohlcv.index > rej_dt].head(max_hold + 5)

                if future_bars.empty:
                    continue

                # Compute ATR for SL/TP calculation
                recent = ohlcv[ohlcv.index <= rej_dt].tail(14)
                if len(recent) < 2:
                    continue

                tr = pd.concat([
                    recent['high'] - recent['low'],
                    abs(recent['high'] - recent['close'].shift(1)),
                    abs(recent['low'] - recent['close'].shift(1)),
                ], axis=1).max(axis=1)
                atr = tr.mean()

                d = 1 if direction == 'LONG' else -1
                sl = price - d * stop_atr_mult * atr
                tp = price + d * stop_atr_mult * atr * rr

                # Simulate trade
                hyp_exit_price = None
                hyp_exit_reason = None

                for bar_i, (_, bar) in enumerate(future_bars.iterrows()):
                    sl_hit = (d == 1 and bar['low'] <= sl) or (d == -1 and bar['high'] >= sl)
                    tp_hit = (d == 1 and bar['high'] >= tp) or (d == -1 and bar['low'] <= tp)
                    time_exit = bar_i >= max_hold

                    if sl_hit:
                        hyp_exit_price = sl
                        hyp_exit_reason = 'SL'
                        break
                    elif tp_hit:
                        hyp_exit_price = tp
                        hyp_exit_reason = 'TP'
                        break
                    elif time_exit:
                        hyp_exit_price = bar['close']
                        hyp_exit_reason = 'TIME'
                        break

                if hyp_exit_price is None:
                    # Not enough future bars yet — skip
                    continue

                hyp_pnl_pct = (hyp_exit_price - price) / price * d * 100
                was_correct = 1 if hyp_pnl_pct <= 0 else 0  # rejection was correct if trade would have lost

                conn.execute("""
                    UPDATE rejected_trades SET
                        hypothetical_exit_price = ?,
                        hypothetical_pnl_pct = ?,
                        hypothetical_exit_reason = ?,
                        was_correct_rejection = ?
                    WHERE id = ?
                """, (hyp_exit_price, hyp_pnl_pct, hyp_exit_reason, was_correct, rej_id))

                updated += 1

            except Exception:
                continue

    return updated


# ============================================================
# UTILITY QUERIES
# ============================================================

def get_recent_snapshots(n=100, tf=None):
    """Get recent trade snapshots as DataFrame."""
    with _journal_conn(readonly=True) as conn:
        query = "SELECT * FROM trade_snapshots ORDER BY entry_time DESC LIMIT ?"
        params = [n]
        if tf:
            query = "SELECT * FROM trade_snapshots WHERE tf = ? ORDER BY entry_time DESC LIMIT ?"
            params = [tf, n]
        return pd.read_sql_query(query, conn, params=params)


def get_recent_outcomes(n=100, tf=None):
    """Get recent trade outcomes as DataFrame."""
    with _journal_conn(readonly=True) as conn:
        query = "SELECT * FROM trade_outcomes ORDER BY created_at DESC LIMIT ?"
        params = [n]
        if tf:
            query = "SELECT * FROM trade_outcomes WHERE tf = ? ORDER BY created_at DESC LIMIT ?"
            params = [tf, n]
        return pd.read_sql_query(query, conn, params=params)


def get_rejected_trades(n=100, reason=None):
    """Get recent rejected trades as DataFrame."""
    with _journal_conn(readonly=True) as conn:
        if reason:
            query = "SELECT * FROM rejected_trades WHERE rejection_reason = ? ORDER BY timestamp DESC LIMIT ?"
            params = [reason, n]
        else:
            query = "SELECT * FROM rejected_trades ORDER BY timestamp DESC LIMIT ?"
            params = [n]
        return pd.read_sql_query(query, conn, params=params)


def get_price_path(trade_id):
    """Get full price path for a trade."""
    with _journal_conn(readonly=True) as conn:
        return pd.read_sql_query(
            "SELECT * FROM price_path WHERE trade_id = ? ORDER BY bar_num",
            conn, params=[trade_id]
        )


def get_journal_stats():
    """Get summary stats from the journal."""
    with _journal_conn(readonly=True) as conn:
        stats = {}
        stats['total_snapshots'] = conn.execute(
            "SELECT COUNT(*) FROM trade_snapshots").fetchone()[0]
        stats['total_outcomes'] = conn.execute(
            "SELECT COUNT(*) FROM trade_outcomes").fetchone()[0]
        stats['total_rejected'] = conn.execute(
            "SELECT COUNT(*) FROM rejected_trades").fetchone()[0]
        stats['total_price_path_bars'] = conn.execute(
            "SELECT COUNT(*) FROM price_path").fetchone()[0]

        # Outcomes with analysis
        stats['analyzed_outcomes'] = conn.execute(
            "SELECT COUNT(*) FROM trade_outcomes WHERE max_adverse_excursion IS NOT NULL"
        ).fetchone()[0]

        # Rejected with hypotheticals
        stats['hypotheticals_filled'] = conn.execute(
            "SELECT COUNT(*) FROM rejected_trades WHERE hypothetical_exit_price IS NOT NULL"
        ).fetchone()[0]

        return stats


# ============================================================
# INIT ON IMPORT
# ============================================================

# Ensure DB and tables exist on first import
try:
    with _journal_conn() as conn:
        pass
except Exception as e:
    print(f"[trade_journal] Warning: Could not init journal DB: {e}")
