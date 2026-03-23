#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
live_trader.py — Live Paper/Real Trading Engine
=================================================
Runs continuously. Every 15 minutes:
1. Fetch latest BTC candles from Bitget
2. Compute features (same code as backtest)
3. Run ML models (4H/1H/15m) — supports V1 (base only) and V2 (base + sparse crosses)
4. Execute trades via portfolio aggregator
5. Log everything to trades.db for dashboard

V2 SPARSE CROSS FIX: Models trained on 4-6M features (base + sparse crosses) now
get the same crosses computed at inference time. Without this, every live prediction
from a V2 model was wrong because it only received ~1000 base features.

Usage: python live_trader.py [--mode paper|live]
"""

import sys, os, io, time, json, warnings, argparse, traceback, signal
if os.name == 'nt':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
except ImportError:
    pass

import numpy as np
import pandas as pd
import sqlite3
import xgboost as xgb
from scipy import sparse
from datetime import datetime, timedelta, timezone
import urllib.request
import requests
from knn_feature_engine import knn_features_from_ohlcv
from feature_library import build_all_features
from data_access import LiveDataLoader
from config import (TF_CAPITAL_ALLOC, TF_SLIPPAGE, load_tf_allocation,
                    TF_PARENT_MAP, TRADE_TYPE_PARAMS, TRADE_THRESHOLDS)

# V2 cross generation — MANDATORY (no fallback, no degradation)
V2_DIR = os.path.dirname(os.path.abspath(__file__))
from v2_cross_generator import (
    binarize_contexts, extract_signal_groups, create_doy_windows,
    create_regime_doy, create_multi_signal_combos,
)

# V2 feature layers — MANDATORY (trained with these, must have at inference)
from v2_feature_layers import add_all_v2_layers

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

# Self-learning trade journal (in parent directory)
_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)
from trade_journal import (log_trade_snapshot, log_price_path_bar,
                           log_trade_outcome, log_rejected_trade,
                           compute_post_trade_analysis)

DB_DIR = os.environ.get("SAVAGE22_V1_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
# V2 SPARSE CROSS FEATURES AT INFERENCE TIME
# ============================================================

# Startup caches — loaded once, reused every bar
_v2_cross_names_cache = {}   # {tf: [list of cross feature names]}


def load_v2_cross_names(symbol, tf):
    """
    Load pre-computed cross feature names from v2_cross_names_{symbol}_{tf}.json.
    Returns list of cross names in the EXACT order training saw them, or None.
    Cached at startup so we never re-read the file on every bar.
    """
    cache_key = f"{symbol}_{tf}"
    if cache_key in _v2_cross_names_cache:
        return _v2_cross_names_cache[cache_key]

    names_path = os.path.join(V2_DIR, f'v2_cross_names_{symbol}_{tf}.json')
    if not os.path.exists(names_path):
        _v2_cross_names_cache[cache_key] = None
        return None
    try:
        with open(names_path, 'r') as f:
            names = json.load(f)
        _v2_cross_names_cache[cache_key] = names
        print(f"  Loaded {len(names):,} cross names from {os.path.basename(names_path)}")
        return names
    except Exception as e:
        print(f"  [WARN] Failed to load cross names from {names_path}: {e}")
        _v2_cross_names_cache[cache_key] = None
        return None


def _cross_pair_last_row(result_dict, left_names, left_arrays, right_names, right_arrays,
                          prefix, last_idx):
    """
    Generate cross products for ONE row (last_idx) from two signal groups.
    Populates result_dict with {cross_name: float_value} for all non-zero crosses.
    Uses the same naming convention as v2_cross_generator.py:
        fname = f'{prefix}_{left_name[:20]}_{right_name[:20]}'
    """
    for ln, la in zip(left_names, left_arrays):
        lv = float(la[last_idx])
        if lv == 0.0:
            continue
        ln_trunc = ln[:20]
        for rn, ra in zip(right_names, right_arrays):
            rv = float(ra[last_idx])
            if rv == 0.0:
                continue
            fname = f'{prefix}_{ln_trunc}_{rn[:20]}'
            result_dict[fname] = lv * rv


def compute_live_crosses(df_features, cross_names, symbol, tf):
    """
    Compute sparse cross features for ONE bar (the last row of df_features)
    using the EXACT same logic as v2_cross_generator.py's generate_all_crosses().

    The binarization uses the full warmup window for percentile computation
    (same relative approach as training — percentiles are window-relative).

    Args:
        df_features: Full feature DataFrame from build_all_features() (all warmup bars).
                     Binarization uses ALL rows for percentile thresholds.
        cross_names: List of cross feature names from v2_cross_names_{symbol}_{tf}.json.
                     Defines the EXACT column ordering the model expects.
        symbol: Asset symbol (e.g., 'BTC').
        tf: Timeframe (e.g., '1d').

    Returns:
        scipy.sparse.csr_matrix of shape (1, len(cross_names)) — the cross feature
        vector for the last bar, ready for hstack with base features.
        Returns None on failure.
    """
    if cross_names is None or len(cross_names) == 0:
        raise ValueError(f"No cross names loaded for {symbol} {tf} — cannot compute V2 crosses")

    t0 = time.time()
    n_crosses = len(cross_names)
    last_idx = len(df_features) - 1

    try:
        # Step 1: Binarize ALL numeric columns using 4-tier (same as offline)
        ctx_names, ctx_arrays = binarize_contexts(df_features, four_tier=True)
        if not ctx_names:
            raise ValueError(f"No binarized contexts for {symbol} {tf} — V2 cross generation requires contexts")

        # Step 2: Extract signal groups (same grouping logic as offline)
        groups = extract_signal_groups(df_features, ctx_names, ctx_arrays)

        # Step 3: DOY windows
        doy_names, doy_arrays = create_doy_windows(df_features)

        # Step 4: Regime-aware DOY
        reg_names, reg_arrays = create_regime_doy(doy_names, doy_arrays, df_features)

        # Step 5: Multi-signal combos (same limits as offline)
        ax2_names, ax2_arrays = [], []
        if len(groups['astro']) >= 2:
            ax2_names, ax2_arrays = create_multi_signal_combos(
                groups['astro'], 'a2', max_pairs=50
            )

        ta2_names, ta2_arrays = [], []
        if len(groups['ta']) >= 2:
            ta2_names, ta2_arrays = create_multi_signal_combos(
                groups['ta'][:60], 'ta2', max_pairs=30
            )

        # Step 6: Extract group names/arrays
        def _grp(key):
            sigs = groups.get(key, [])
            return [s[0] for s in sigs], [s[1] for s in sigs]

        astro_n, astro_a = _grp('astro')
        ta_n, ta_a = _grp('ta')
        eso_n, eso_a = _grp('esoteric')
        sw_n, sw_a = _grp('space_weather')
        hod_n, hod_a = _grp('session')
        mx_n, mx_a = _grp('macro')
        asp_n, asp_a = _grp('aspect')
        pn_n, pn_a = _grp('price_num')
        mn_n, mn_a = _grp('moon')
        vx_n, vx_a = _grp('volatility')

        # Step 7: Generate all crosses for the last row ONLY
        # Must follow EXACT same order as generate_all_crosses() in v2_cross_generator.py
        live_cross_values = {}

        # Cross 1: DOY x ALL contexts
        _cross_pair_last_row(live_cross_values,
                              doy_names, doy_arrays, ctx_names, ctx_arrays,
                              'dx', last_idx)

        # Cross 2: Astro x TA
        if astro_n and ta_n:
            _cross_pair_last_row(live_cross_values,
                                  astro_n, astro_a, ta_n, ta_a,
                                  'ax', last_idx)

        # Cross 3: Multi-astro combos x TA
        if ax2_names and ta_n:
            _cross_pair_last_row(live_cross_values,
                                  ax2_names, ax2_arrays, ta_n, ta_a,
                                  'ax2', last_idx)

        # Cross 4: Multi-TA combos x (DOY + astro)
        if ta2_names:
            combined_n = doy_names + astro_n
            combined_a = doy_arrays + astro_a
            _cross_pair_last_row(live_cross_values,
                                  ta2_names, ta2_arrays, combined_n, combined_a,
                                  'ta2', last_idx)

        # Cross 5: Esoteric x TA
        if eso_n and ta_n:
            _cross_pair_last_row(live_cross_values,
                                  eso_n, eso_a, ta_n, ta_a,
                                  'ex2', last_idx)

        # Cross 6: Space weather x ALL
        if sw_n:
            _cross_pair_last_row(live_cross_values,
                                  sw_n, sw_a, ctx_names, ctx_arrays,
                                  'sw', last_idx)

        # Cross 7: Session x ALL
        if hod_n:
            _cross_pair_last_row(live_cross_values,
                                  hod_n, hod_a, ctx_names, ctx_arrays,
                                  'hod', last_idx)

        # Cross 8: Macro x ALL
        if mx_n:
            _cross_pair_last_row(live_cross_values,
                                  mx_n, mx_a, ctx_names, ctx_arrays,
                                  'mx', last_idx)

        # Cross 9: Volatility x ALL
        if vx_n:
            _cross_pair_last_row(live_cross_values,
                                  vx_n, vx_a, ctx_names, ctx_arrays,
                                  'vx', last_idx)

        # Cross 10: Aspects x ALL
        if asp_n:
            _cross_pair_last_row(live_cross_values,
                                  asp_n, asp_a, ctx_names, ctx_arrays,
                                  'asp', last_idx)

        # Cross 11: Price numerology x ALL
        if pn_n:
            _cross_pair_last_row(live_cross_values,
                                  pn_n, pn_a, ctx_names, ctx_arrays,
                                  'pn', last_idx)

        # Cross 12: Moon x ALL
        if mn_n:
            _cross_pair_last_row(live_cross_values,
                                  mn_n, mn_a, ctx_names, ctx_arrays,
                                  'mn', last_idx)

        # Cross 13: Regime DOY x ALL
        if reg_names:
            _cross_pair_last_row(live_cross_values,
                                  reg_names, reg_arrays, ctx_names, ctx_arrays,
                                  'rdx', last_idx)

        # Step 8: Build sparse row vector aligned to the saved cross_names ordering.
        # The model expects columns in EXACTLY this order.
        col_indices = []
        values = []
        for i, cname in enumerate(cross_names):
            val = live_cross_values.get(cname, 0.0)
            if val != 0.0:
                col_indices.append(i)
                values.append(val)

        if col_indices:
            row_indices = np.zeros(len(col_indices), dtype=np.int32)
            col_arr = np.array(col_indices, dtype=np.int32)
            val_arr = np.array(values, dtype=np.float32)
            cross_sparse = sparse.csr_matrix(
                (val_arr, (row_indices, col_arr)),
                shape=(1, n_crosses),
                dtype=np.float32
            )
        else:
            cross_sparse = sparse.csr_matrix((1, n_crosses), dtype=np.float32)

        elapsed_ms = (time.time() - t0) * 1000
        n_nonzero = len(col_indices)
        n_generated = len(live_cross_values)
        print(f"  Crosses: {n_nonzero:,} non-zero / {n_crosses:,} total "
              f"({n_generated:,} generated, {elapsed_ms:.0f}ms)")

        return cross_sparse

    except Exception as e:
        print(f"  [FATAL] compute_live_crosses failed: {e}")
        traceback.print_exc()
        # NO FALLBACK — V2 model requires crosses. Fix the bug, don't degrade.
        raise RuntimeError(f"V2 cross generation failed for {symbol} {tf}. "
                          f"Cannot trade without crosses. Fix and restart.") from e


def compute_features_live_v2(tf_name, base_feat_names, symbol='BTC'):
    """
    V2-aware feature computation. Returns the full feature DataFrame (for cross
    generation) AND the base feature dict (for the base feature vector).

    Unlike compute_features_live() which only returns a dict of {name: value},
    this returns:
        (feat_dict, df_features_full)
    where df_features_full is the complete DataFrame needed for binarization.
    """
    try:
        live_dal.refresh_caches()

        n_bars = WARMUP_BARS.get(tf_name, 300)
        ohlcv = live_dal.get_ohlcv_window(tf_name, n_bars)
        if ohlcv is None or len(ohlcv) < 50:
            raise ValueError(f"Not enough OHLCV bars for {tf_name}: got {len(ohlcv) if ohlcv is not None else 0}, need 50+")

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
            raise ValueError(f"Feature pipeline returned empty DataFrame for {tf_name}")

        # Add V2 feature layers — MANDATORY, no silent degradation
        df_features = add_all_v2_layers(
            df_features, symbol=symbol, tf=tf_name,
            astro_cache=astro_cache,
        )

        last_row = df_features.iloc[-1]
        result = {}
        for fn in base_feat_names:
            val = last_row.get(fn, np.nan)
            if isinstance(val, (int, float)) and np.isinf(val):
                val = np.nan
            result[fn] = val

        return result, df_features

    except Exception as e:
        print(f"  [FATAL] compute_features_live_v2 FAILED: {e}")
        traceback.print_exc()
        # NO FALLBACK — V2 features are mandatory. Crash and fix.
        raise RuntimeError(f"V2 feature computation failed for {tf_name}: {e}") from e


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
    except Exception as e:
        print(f"  [FATAL] HMM feature extraction FAILED: {e}")
        traceback.print_exc()
        raise RuntimeError(f"HMM features failed — fix root cause: {e}") from e


# ============================================================
# CROSS-TF CONFLUENCE FILTER
# ============================================================
class TFConfluenceFilter:
    """
    Maintains latest signal direction per TF and filters lower-TF entries
    using the parent TF direction.

    Directions: 1=LONG, -1=SHORT, 0=flat (no signal / below threshold)

    Rules:
    - Parent same direction  -> ALLOW (full size, scale=1.0)
    - Parent opposite        -> BLOCK (scale=0.0)
    - Parent flat / unknown  -> ALLOW half size (scale=0.5)
    - 1w and 1d have no parent filter (they ARE the trend)
    """

    def __init__(self):
        # {tf: int} where 1=long, -1=short, 0=flat
        self.signals = {}

    def update(self, tf, direction_int):
        """Update the latest signal for a TF. direction_int: 1, -1, or 0."""
        self.signals[tf] = direction_int

    def check(self, tf, direction_int):
        """
        Check if a trade on `tf` in `direction_int` is allowed.

        Returns:
            (allowed: bool, size_scale: float, reason: str)
        """
        parent_tf = TF_PARENT_MAP.get(tf)

        # No parent -> always allowed (1w, 1d are the trend)
        if parent_tf is None:
            return True, 1.0, "no parent filter"

        parent_dir = self.signals.get(parent_tf, None)

        # Parent has no signal yet (first bars, no history)
        if parent_dir is None or parent_dir == 0:
            return True, 0.5, f"parent {parent_tf} flat/unknown — half size"

        # Parent same direction -> full size
        if parent_dir == direction_int:
            return True, 1.0, f"parent {parent_tf} confirms {'LONG' if direction_int == 1 else 'SHORT'}"

        # Parent opposite direction -> BLOCK
        dir_name = 'LONG' if direction_int == 1 else 'SHORT'
        parent_dir_name = 'LONG' if parent_dir == 1 else 'SHORT'
        return False, 0.0, f"Confluence BLOCKED: {tf} {dir_name} rejected ({parent_tf} is {parent_dir_name})"


# ============================================================
# TRADE TYPE CLASSIFICATION
# ============================================================
def classify_expected_trade_type(tf, max_hold):
    """
    Classify the expected trade type based on TF and max_hold bars.
    Uses TRADE_THRESHOLDS from config.py (single source of truth).

    Returns one of: 'scalp', 'day_trade', 'swing', 'position'
    """
    thresholds = TRADE_THRESHOLDS.get(tf, TRADE_THRESHOLDS['1h'])
    if max_hold <= thresholds['scalp']:
        return 'scalp'
    elif max_hold <= thresholds['day']:
        return 'day_trade'
    elif max_hold <= thresholds['swing']:
        return 'swing'
    else:
        return 'position'


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
# REFRESH OHLCV — fetch latest candles into btc_prices.db
# ============================================================
_TF_GRANULARITY_MAP = {
    '5m': '5m', '15m': '15m', '1h': '1H', '4h': '4H', '1d': '1D', '1w': '1W'
}
_TF_INTERVAL_SECONDS = {
    '5m': 300, '15m': 900, '1h': 3600, '4h': 14400, '1d': 86400, '1w': 604800
}
PRICES_DB = f"{DB_DIR}/btc_prices.db"


def refresh_ohlcv(tfs, symbol='BTCUSDT', db_symbol='BTC/USDT', limit=100):
    """
    Fetch latest candles from Bitget REST API for each timeframe
    and INSERT OR REPLACE into btc_prices.db so OHLCV data is always fresh.

    Args:
        tfs: list of timeframe strings (e.g. ['15m', '1h', '4h'])
        symbol: Bitget API symbol (e.g. 'BTCUSDT')
        db_symbol: symbol format stored in the DB (e.g. 'BTC/USDT')
        limit: number of candles to fetch per TF
    """
    conn = sqlite3.connect(PRICES_DB)
    total_inserted = 0

    for tf in tfs:
        gran = _TF_GRANULARITY_MAP.get(tf)
        if not gran:
            print(f"  [OHLCV] Unknown TF '{tf}', skipping")
            continue

        url = (f"https://api.bitget.com/api/v2/mix/market/candles"
               f"?productType=USDT-FUTURES&symbol={symbol}"
               f"&granularity={gran}&limit={limit}")
        try:
            resp = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
            resp.raise_for_status()
            data = resp.json()

            if isinstance(data, dict) and 'data' in data:
                candles = data['data']
            elif isinstance(data, list):
                candles = data
            else:
                print(f"  [OHLCV] {tf}: unexpected response format")
                continue

            if not candles:
                print(f"  [OHLCV] {tf}: no candles returned")
                continue

            rows = []
            for c in candles:
                rows.append((
                    db_symbol,          # symbol
                    tf,                 # timeframe
                    int(c[0]),          # open_time (ms epoch)
                    float(c[1]),        # open
                    float(c[2]),        # high
                    float(c[3]),        # low
                    float(c[4]),        # close
                    float(c[5]) if len(c) > 5 else 0.0,  # volume
                    float(c[6]) if len(c) > 6 else 0.0,  # quote_volume
                    0,                  # trades (not in Bitget response)
                    0.0,                # taker_buy_volume
                    0.0,                # taker_buy_quote
                    None,               # close_time
                ))

            conn.executemany(
                """INSERT OR REPLACE INTO ohlcv
                   (symbol, timeframe, open_time, open, high, low, close,
                    volume, quote_volume, trades, taker_buy_volume,
                    taker_buy_quote, close_time)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                rows
            )
            conn.commit()
            total_inserted += len(rows)

            # Freshness check: warn if latest candle is stale
            latest_ts_ms = max(int(c[0]) for c in candles)
            now_ms = int(time.time() * 1000)
            age_s = (now_ms - latest_ts_ms) / 1000.0
            interval_s = _TF_INTERVAL_SECONDS.get(tf, 900)
            if age_s > 2 * interval_s:
                age_min = age_s / 60
                print(f"  [OHLCV] WARNING: {tf} latest candle is {age_min:.0f}m old "
                      f"(>{2*interval_s/60:.0f}m threshold)")
            else:
                print(f"  [OHLCV] {tf}: {len(rows)} candles refreshed (latest {age_s/60:.1f}m ago)")

        except requests.exceptions.RequestException as e:
            print(f"  [OHLCV] {tf}: API error — {e}")
        except Exception as e:
            print(f"  [OHLCV] {tf}: unexpected error — {e}")

    conn.close()
    if total_inserted > 0:
        print(f"  [OHLCV] Total: {total_inserted} rows upserted into btc_prices.db")


# ============================================================
# COMPUTE FEATURES (shared library) — V1 fallback
# ============================================================
def compute_features_live(tf_name, feat_names):
    """Compute features for current bar using shared feature library.
    V1 path: returns only feat_dict (no DataFrame for crosses)."""
    try:
        live_dal.refresh_caches()

        n_bars = WARMUP_BARS.get(tf_name, 300)
        ohlcv = live_dal.get_ohlcv_window(tf_name, n_bars)
        if ohlcv is None or len(ohlcv) < 50:
            raise RuntimeError(f"compute_features_live({tf_name}): OHLCV data insufficient — got {0 if ohlcv is None else len(ohlcv)} bars, need 50+")

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
            raise RuntimeError(f"compute_features_live({tf_name}): build_all_features returned empty — pipeline broken")

        last_row = df_features.iloc[-1]
        result = {}
        for fn in feat_names:
            val = last_row.get(fn, np.nan)
            if isinstance(val, (int, float)) and np.isinf(val):
                val = np.nan
            result[fn] = val
        return result
    except Exception as e:
        print(f"  [FATAL] compute_features_live FAILED: {e}")
        traceback.print_exc()
        raise RuntimeError(f"Feature computation failed: {e}") from e

# ============================================================
# DETECT REGIME
# ============================================================
# 5D regime multipliers matching exhaustive_optimizer.py
# Index: [bull=0, bear=1, sideways=2, crash=3]
REGIME_MULT = {
    'bull':       {'lev': 1.0,  'risk': 1.0,  'stop': 1.0,  'rr': 1.0,  'hold': 1.0},
    'bear':       {'lev': 0.47, 'risk': 1.0,  'stop': 0.75, 'rr': 0.75, 'hold': 0.17},
    'sideways':   {'lev': 0.67, 'risk': 0.47, 'stop': 0.5,  'rr': 0.5,  'hold': 1.0},
    'crash':      {'lev': 0.07, 'risk': 1.0,  'stop': 1.5,  'rr': 0.25, 'hold': 2.0},
    'transition': {'lev': 0.8,  'risk': 0.8,  'stop': 0.8,  'rr': 0.8,  'hold': 0.8},
    'unknown':    {'lev': 1.0,  'risk': 1.0,  'stop': 1.0,  'rr': 1.0,  'hold': 1.0},
}

def detect_regime(close, sma100, sma100_slope, rvol_20=None, rvol_90_avg=None, dd_from_30h=None):
    if sma100 is None or sma100_slope is None:
        return 'unknown', REGIME_MULT['unknown']
    # Crash detection: high vol + below SMA100 + deep drawdown from recent high
    if (rvol_20 is not None and rvol_90_avg is not None and dd_from_30h is not None
            and rvol_20 > rvol_90_avg * 2.0 and close < sma100 and dd_from_30h > 0.15):
        return 'crash', REGIME_MULT['crash']
    above = close > sma100
    near = abs(close - sma100) / sma100 < 0.05
    if above and sma100_slope > 0.001:
        return 'bull', REGIME_MULT['bull']
    elif not above and sma100_slope < -0.001:
        return 'bear', REGIME_MULT['bear']
    elif near or (sma100_slope >= -0.001 and sma100_slope <= 0.001):
        return 'sideways', REGIME_MULT['sideways']
    return 'transition', REGIME_MULT['transition']

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
# GRACEFUL SHUTDOWN
# ============================================================
_shutdown = False

def _handle_signal(sig, frame):
    global _shutdown
    _shutdown = True
    print(f"\n[SHUTDOWN] Signal {sig} received, finishing current cycle...")

signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

# ============================================================
# MAIN TRADING LOOP
# ============================================================
def run_trading_loop(mode='paper'):
    print("=" * 60)
    print(f"  LIVE TRADER — Mode: {mode.upper()}")
    print(f"  Started: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    init_trades_db()

    # Capital allocation per TF — single source of truth in config.py
    # Supports override from optimal_allocation.json
    tf_alloc = load_tf_allocation()

    # Per-TF pool tracking: balance, peak, drawdown
    conn_init = sqlite3.connect(TRADES_DB)
    total_balance = conn_init.execute("SELECT balance FROM account WHERE id=1").fetchone()[0]
    conn_init.close()
    tf_pools = {}
    for tf_name, alloc in tf_alloc.items():
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

    # Cross-TF confluence filter (persists between bars)
    confluence_filter = TFConfluenceFilter()
    print(f"  Confluence filter: ACTIVE (hierarchy: 1w > 1d > 4h > 1h > 15m > 5m)")

    # ── Load models — V2 first (with crosses), then V1 fallback ──
    ALL_TFS = ['5m', '15m', '1h', '4h', '1d', '1w']
    models = {}
    features_list = {}       # {tf: [base feature names]}
    cross_names_by_tf = {}   # {tf: [cross feature names] or None}
    model_type = {}          # {tf: 'v1' or 'v2'}

    for tf in ALL_TFS:
        loaded = False

        # --- Try V2 models first (production > unified > per-asset) ---
        v2_model_candidates = [
            (f'model_v2_prod_BTC_{tf}.json', f'features_v2_production_BTC_{tf}.json'),
            (f'model_v2_unified_{tf}.json', f'features_v2_unified_{tf}.json'),
            (f'model_v2_BTC_{tf}.json', f'features_v2_per-asset_BTC_{tf}.json'),
        ]
        for model_file, feat_file in v2_model_candidates:
            model_path = os.path.join(V2_DIR, model_file)
            feat_path = os.path.join(V2_DIR, feat_file)
            if os.path.exists(model_path) and os.path.exists(feat_path):
                m = xgb.Booster()
                m.load_model(model_path)
                models[tf] = m
                with open(feat_path) as f:
                    all_feat_names = json.load(f)

                # Split feature list into base features and cross features.
                # Cross features have prefixes from v2_cross_generator.
                cross_prefixes = ('dx_', 'ax_', 'ax2_', 'ta2_', 'ex2_', 'sw_',
                                  'hod_', 'mx_', 'vx_', 'asp_', 'pn_', 'mn_', 'rdx_')
                base_feats = [fn for fn in all_feat_names if not fn.startswith(cross_prefixes)]
                cross_feats = [fn for fn in all_feat_names if fn.startswith(cross_prefixes)]

                features_list[tf] = base_feats
                model_type[tf] = 'v2'

                # Load cross names from the JSON saved during build (authoritative ordering)
                cross_names_loaded = load_v2_cross_names('BTC', tf)
                if cross_names_loaded is not None:
                    cross_names_by_tf[tf] = cross_names_loaded
                elif cross_feats:
                    # NO FALLBACK — cross names JSON is authoritative. Model feature list
                    # may have different ordering. Require the JSON file.
                    raise FileNotFoundError(
                        f"v2_cross_names JSON missing for {tf}. "
                        f"Found {len(cross_feats):,} cross features in model but ordering "
                        f"may differ from training. Re-run build_features_v2.py to generate JSON.")
                else:
                    cross_names_by_tf[tf] = None

                n_cross = len(cross_names_by_tf.get(tf) or [])
                print(f"  Loaded V2 {tf} model ({len(base_feats)} base + {n_cross:,} crosses) "
                      f"from {model_file}")
                loaded = True
                break

        if loaded:
            continue

        # --- Fallback: V1 models ---
        model_path = f'{DB_DIR}/model_{tf}.json'
        feat_path_all = f'{DB_DIR}/features_{tf}_all.json'
        feat_path_pruned = f'{DB_DIR}/features_{tf}_pruned.json'
        if os.path.exists(feat_path_all):
            feat_path = feat_path_all
        elif os.path.exists(feat_path_pruned):
            feat_path = feat_path_pruned
        else:
            feat_path = None
        if os.path.exists(model_path) and feat_path:
            m = xgb.Booster()
            m.load_model(model_path)
            models[tf] = m
            with open(feat_path) as f:
                features_list[tf] = json.load(f)
            model_type[tf] = 'v1'
            cross_names_by_tf[tf] = None
            print(f"  Loaded V1 {tf} model ({len(features_list[tf])} features) "
                  f"from {os.path.basename(feat_path)}")

    # Load meta-labeling models (institutional upgrade)
    meta_models = {}
    if _HAS_META:
        for tf in ALL_TFS:
            meta_path = os.path.join(V2_DIR, f'meta_model_{tf}.pkl')
            if not os.path.exists(meta_path):
                meta_path = os.path.join(DB_DIR, f'meta_model_{tf}.pkl')
            if os.path.exists(meta_path):
                with open(meta_path, 'rb') as f:
                    meta_models[tf] = pickle.load(f)
                print(f"  Loaded meta-model for {tf} (thresh={meta_models[tf].get('threshold', 0.5):.2f})")
    if not meta_models:
        print("  No meta-models found — using raw XGBoost predictions")

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
            platt_path = os.path.join(V2_DIR, f'platt_lstm_{tf}.pkl')
            if not os.path.exists(platt_path):
                platt_path = os.path.join(DB_DIR, f'platt_lstm_{tf}.pkl')
            if os.path.exists(platt_path):
                with open(platt_path, 'rb') as f:
                    platt_models[tf] = pickle.load(f)

    # Rolling performance tracking (live monitoring)
    trade_results = []  # list of recent trade P&Ls for Kelly re-estimation

    # Trailing stop tracking: {trade_id: best_price}
    # Tracks the best price (highest for longs, lowest for shorts) since entry
    best_prices = {}

    # Partial TP tracking: {trade_id: True} — positions that already took partial TP
    partial_tp_taken = {}

    # Load GA configs
    try:
        with open(f'{DB_DIR}/ml_multi_tf_configs.json') as f:
            ml_configs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        ml_configs = {}

    # Load exhaustive_configs.json if available (overrides GA configs)
    # Try unified file first, then merge per-TF files
    exhaustive_configs = {}
    exhaustive_path = os.path.join(V2_DIR, 'exhaustive_configs.json')
    if not os.path.exists(exhaustive_path):
        exhaustive_path = f'{DB_DIR}/exhaustive_configs.json'
    if os.path.exists(exhaustive_path):
        with open(exhaustive_path) as f:
            exhaustive_configs = json.load(f)
        print(f"  Loaded exhaustive_configs.json")
    # Also load per-TF files (e.g. exhaustive_configs_1h.json) — cloud pipeline saves these
    import glob as _glob
    per_tf_files = _glob.glob(os.path.join(V2_DIR, 'exhaustive_configs_*.json'))
    if not per_tf_files:
        per_tf_files = _glob.glob(f'{DB_DIR}/exhaustive_configs_*.json')
    for per_tf_file in per_tf_files:
        try:
            with open(per_tf_file) as f:
                per_tf_data = json.load(f)
            exhaustive_configs.update(per_tf_data)
            print(f"  Loaded {os.path.basename(per_tf_file)}")
        except Exception:
            pass

    ga_params = {}
    for tf in ALL_TFS:
        # Try exhaustive configs first (format: {"15m": {"dd10_best": {...}, "dd15_best": {...}, ...}})
        if tf in exhaustive_configs:
            tf_ec = exhaustive_configs[tf]
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

    # ── Startup summary ──
    v1_models = [tf for tf in models if model_type.get(tf) == 'v1']
    v2_models = [tf for tf in models if model_type.get(tf) == 'v2']
    print(f"\n  V1 Models: {v1_models if v1_models else 'none'}")
    print(f"  V2 Models (with crosses): {v2_models if v2_models else 'none'}")
    if v2_models:
        for tf in v2_models:
            cn = cross_names_by_tf.get(tf)
            if cn:
                print(f"    {tf}: {len(cn):,} cross features loaded")
            else:
                raise RuntimeError(f"V2 model {tf} has NO crosses available — cannot run degraded. Fix cross loading or retrain as V1.")
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
        if _shutdown:
            break
        try:
            now = datetime.now(timezone.utc)

            # Wait for 15m bar close (at :00, :15, :30, :45 + 5 second buffer)
            minutes = now.minute
            seconds_to_next = ((15 - minutes % 15) * 60 - now.second + 5) % (15 * 60)
            if seconds_to_next > 10:
                time.sleep(min(seconds_to_next, 30))
                continue

            # ── Refresh OHLCV from Bitget BEFORE any feature computation ──
            active_tfs = [tf for tf in ALL_TFS if tf in models]
            if active_tfs:
                print(f"\n[{now.strftime('%H:%M:%S')}] Refreshing OHLCV for {active_tfs}...")
                refresh_ohlcv(active_tfs)

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
                        log_rejected_trade(
                            tf=tf, direction='FLAT', timestamp=now.isoformat(),
                            price=0, confidence=0,
                            reason='tf_dd_halt',
                        )
                    except Exception:
                        pass
                    continue

                bar_key = f"{tf}_{now.strftime('%Y%m%d%H%M')}"
                if bar_key in last_bar_time:
                    continue
                last_bar_time[bar_key] = True

                print(f"\n[{now.strftime('%H:%M:%S')}] {tf.upper()} bar closed")

                # ── Compute features ──
                feat_names = features_list[tf]
                is_v2 = model_type.get(tf) == 'v2'
                cross_names = cross_names_by_tf.get(tf) if is_v2 else None

                # V2 path: need full DataFrame for cross generation
                if is_v2 and cross_names:
                    feat_dict, df_features_full = compute_features_live_v2(
                        tf, feat_names, symbol='BTC'
                    )
                else:
                    # V1 path: just get the feature dict
                    feat_dict = compute_features_live(tf, feat_names)
                    df_features_full = None

                if not feat_dict:
                    print(f"  No features returned for {tf}")
                    continue

                # Inject HMM features (Bug fix: were missing in V2 live mode)
                hmm_feats = get_hmm_features(feat_dict)
                feat_dict.update(hmm_feats)

                price = feat_dict.get('close', 0)
                if price <= 0:
                    ohlcv_tmp = live_dal.get_ohlcv_window(tf, 2)
                    price = float(ohlcv_tmp['close'].iloc[-1]) if ohlcv_tmp is not None and len(ohlcv_tmp) > 0 else 0

                # ── Build feature vector and predict ──
                X_base = np.array([[feat_dict.get(fn, np.nan) for fn in feat_names]], dtype=np.float32)
                X_base = np.where(np.isinf(X_base), np.nan, X_base)

                if is_v2 and cross_names and df_features_full is not None:
                    # V2: compute sparse crosses and combine with base
                    X_cross_sparse = compute_live_crosses(
                        df_features_full, cross_names, 'BTC', tf
                    )

                    if X_cross_sparse is not None:
                        # Combine: hstack([base_sparse, cross_sparse])
                        # Matches trainer: sparse.hstack([X_base_all, X_sparse_all])
                        X_base_sparse = sparse.csr_matrix(X_base)
                        X_combined = sparse.hstack([X_base_sparse, X_cross_sparse], format='csr')

                        # Build combined feature names for DMatrix
                        all_feat_names = feat_names + cross_names

                        # Predict with sparse DMatrix (3-class softprob safe)
                        dmat = xgb.DMatrix(X_combined, feature_names=all_feat_names, nthread=-1)
                        raw_pred = models[tf].predict(dmat)
                        if raw_pred.ndim == 2 and raw_pred.shape[1] == 3:
                            p_short, p_flat, p_long = float(raw_pred[0][0]), float(raw_pred[0][1]), float(raw_pred[0][2])
                            pred_class = int(np.argmax(raw_pred[0]))
                            confidence = float(np.max(raw_pred[0]))
                            direction_int = 1 if pred_class == 2 else (-1 if pred_class == 0 else 0)
                        else:
                            prob = float(raw_pred[0]) if raw_pred.ndim == 1 else float(raw_pred[0][0])
                            confidence = prob
                            direction_int = 1 if prob > 0.5 else -1
                            p_short, p_flat, p_long = (1 - prob, 0.0, prob) if prob > 0.5 else (prob, 0.0, 1 - prob)
                            pred_class = 2 if prob > 0.5 else 0
                        print(f"  V2 prediction: {X_combined.shape[1]:,} features "
                              f"({len(feat_names)} base + {len(cross_names):,} crosses)")
                    else:
                        # NO FALLBACK — V2 model requires crosses. Fix the bug.
                        raise RuntimeError(
                            f"V2 cross generation returned None for {tf}. "
                            f"Cannot predict with base-only features on a V2 model. Fix and restart.")
                else:
                    # V1 model or no crosses available — base features only
                    dmat = xgb.DMatrix(X_base, feature_names=feat_names, nthread=-1)
                    raw_pred = models[tf].predict(dmat)
                    if raw_pred.ndim == 2 and raw_pred.shape[1] == 3:
                        p_short, p_flat, p_long = float(raw_pred[0][0]), float(raw_pred[0][1]), float(raw_pred[0][2])
                        pred_class = int(np.argmax(raw_pred[0]))
                        confidence = float(np.max(raw_pred[0]))
                        direction_int = 1 if pred_class == 2 else (-1 if pred_class == 0 else 0)
                    else:
                        prob = float(raw_pred[0]) if raw_pred.ndim == 1 else float(raw_pred[0][0])
                        confidence = prob
                        direction_int = 1 if prob > 0.5 else -1
                        p_short, p_flat, p_long = (1 - prob, 0.0, prob) if prob > 0.5 else (prob, 0.0, 1 - prob)
                        pred_class = 2 if prob > 0.5 else 0

                atr = feat_dict.get('atr_14', price * 0.01)
                params = ga_params.get(tf, {})
                conf_thresh = params.get('conf_thresh', 0.80)

                # Detect regime
                sma100 = feat_dict.get('sma_100', price)
                ema50_slope_val = feat_dict.get('ema50_slope', 0)
                sma100_slope = ema50_slope_val / 100.0 if ema50_slope_val else 0
                # Crash detection features
                rvol_20 = feat_dict.get('rvol_20', None)
                rvol_90_avg = feat_dict.get('rvol_90_avg', None)
                dd_from_30h = feat_dict.get('dd_from_30d_high', None)
                regime, regime_mult = detect_regime(price, sma100, sma100_slope,
                                                    rvol_20=rvol_20, rvol_90_avg=rvol_90_avg,
                                                    dd_from_30h=dd_from_30h)

                direction = None
                if pred_class == 2 and confidence > conf_thresh:
                    direction = 'LONG'
                    prob = p_long
                elif pred_class == 0 and confidence > conf_thresh:
                    direction = 'SHORT'
                    prob = p_short
                else:
                    prob = confidence

                # Update confluence filter with this TF's signal (always, even if no trade)
                dir_int = 1 if direction == 'LONG' else (-1 if direction == 'SHORT' else 0)
                confluence_filter.update(tf, dir_int)

                print(f"  Price: ${price:,.0f} | Pred: {['SHORT','FLAT','LONG'][pred_class]} "
                      f"({confidence:.1%}) | P(L/F/S)={p_long:.2f}/{p_flat:.2f}/{p_short:.2f} | Regime: {regime} (lev={regime_mult['lev']:.2f})")

                if direction and portfolio_halted:
                    print(f"  SKIP — portfolio DD halt active ({portfolio_dd*100:.1f}%)")
                    try:
                        log_rejected_trade(
                            tf=tf, direction=direction, timestamp=now.isoformat(),
                            price=price, confidence=confidence,
                            reason='portfolio_dd_halt',
                        )
                    except Exception as e:
                        print(f"  [journal] rejected error: {e}")
                    direction = None

                # Confluence filter: check parent TF direction before entry
                conf_scale = 1.0
                if direction:
                    allowed, conf_scale, conf_reason = confluence_filter.check(tf, dir_int)
                    if not allowed:
                        print(f"  {conf_reason}")
                        try:
                            _parent_tf_rej = TF_PARENT_MAP.get(tf)
                            log_rejected_trade(
                                tf=tf, direction=direction, timestamp=now.isoformat(),
                                price=price, confidence=confidence,
                                reason='confluence_block',
                                confluence_info={
                                    'parent_tf': _parent_tf_rej,
                                    'parent_dir': confluence_filter.signals.get(_parent_tf_rej),
                                },
                            )
                        except Exception as e:
                            print(f"  [journal] rejected error: {e}")
                        direction = None
                    elif conf_scale < 1.0:
                        print(f"  Confluence: {conf_reason}")

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
                            log_rejected_trade(
                                tf=tf, direction=direction, timestamp=now.isoformat(),
                                price=price, confidence=confidence,
                                reason='already_in_position',
                            )
                        except Exception as e:
                            print(f"  [journal] rejected error: {e}")
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
                            log_rejected_trade(
                                tf=tf, direction=direction, timestamp=now.isoformat(),
                                price=price, confidence=confidence,
                                reason='duplicate',
                            )
                        except Exception as e:
                            print(f"  [journal] rejected error: {e}")
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
                            log_rejected_trade(
                                tf=tf, direction=direction, timestamp=now.isoformat(),
                                price=price, confidence=confidence,
                                reason='duplicate',
                            )
                        except Exception as e:
                            print(f"  [journal] rejected error: {e}")
                        conn.rollback()
                        conn.close()
                        continue

                    # === SINGLE PROBABILITY PIPELINE ===
                    # 1. Base XGBoost prob (already computed above as 'prob')
                    # prob is P(long) from XGBoost softprob

                    # 2. LSTM blending (if model exists)
                    if _HAS_LSTM and tf in lstm_extractors and df_features_full is not None:
                        try:
                            lstm_feats = lstm_extractors[tf].extract(df_features_full)
                            lstm_p = lstm_feats['lstm_prob'].iloc[-1]
                            if not np.isnan(lstm_p):
                                if tf in platt_models:
                                    lstm_p = apply_platt_calibration(np.array([lstm_p]), platt_models[tf])[0]
                                # Load alpha from blend_config_{tf}.json (default 0.2)
                                _blend_alpha = 0.2
                                _blend_cfg_path = os.path.join(V2_DIR, f'blend_config_{tf}.json')
                                if not os.path.exists(_blend_cfg_path):
                                    _blend_cfg_path = os.path.join(DB_DIR, f'blend_config_{tf}.json')
                                try:
                                    if os.path.exists(_blend_cfg_path):
                                        with open(_blend_cfg_path) as _bcf:
                                            _blend_cfg = json.load(_bcf)
                                        _blend_alpha = float(_blend_cfg.get('alpha', 0.2))
                                except Exception:
                                    pass  # default 0.2
                                # Blend: (1-alpha) XGBoost + alpha LSTM
                                prob = (1 - _blend_alpha) * prob + _blend_alpha * lstm_p
                        except Exception as e:
                            print(f"  [FATAL] LSTM inference failed for {tf}: {e}")
                            traceback.print_exc()
                            raise

                    # 3. Meta-labeling gate (if model exists)
                    if _HAS_META and tf in meta_models:
                        try:
                            # Build 3-class prob array for meta input [short, flat, long]
                            # Flat gets the remainder so it's never exactly 0
                            if prob > 0.5:
                                _xgb_3c = np.array([[0.0, 1 - prob, prob]])  # long-leaning: flat=remainder
                            else:
                                _xgb_3c = np.array([[1 - prob, prob, 0.0]])  # short-leaning: flat=remainder
                            meta_probs, take = predict_meta(meta_models[tf], _xgb_3c)
                            if not take[0]:
                                print(f"  META GATE: {tf} trade rejected (meta_prob={meta_probs[0]:.3f})")
                                try:
                                    log_rejected_trade(
                                        tf=tf, direction=direction, timestamp=now.isoformat(),
                                        price=price, confidence=confidence,
                                        reason='meta_gate',
                                        meta_prob=float(meta_probs[0]),
                                    )
                                except Exception as e:
                                    print(f"  [journal] rejected error: {e}")
                                conn.rollback()
                                conn.close()
                                continue
                        except Exception as e:
                            print(f"  [FATAL] Meta-labeling failed for {tf}: {e}")
                            traceback.print_exc()
                            raise

                    # 4. Kelly uses blended+gated probability
                    confidence = prob if direction == 'LONG' else 1 - prob
                    lev = params.get('leverage', 10) * regime_mult['lev']
                    stop_mult = params.get('stop_atr', 1.0) * regime_mult['stop']
                    rr = params.get('rr', 2.0) * regime_mult['rr']
                    max_hold = int(params.get('max_hold', 4) * regime_mult['hold'])

                    # 4a. Trade type classification + parameter modifiers
                    trade_type = classify_expected_trade_type(tf, max_hold)
                    tt_params = TRADE_TYPE_PARAMS.get(trade_type, {})
                    sl_tightness = tt_params.get('sl_tightness', 1.0)
                    tp_aggression = tt_params.get('tp_aggression', 1.0)
                    stop_mult = stop_mult / sl_tightness if sl_tightness else stop_mult
                    rr = rr / tp_aggression if tp_aggression else rr

                    # 4b. Apply confluence size scale
                    # conf_scale is 1.0 (parent confirms), 0.5 (parent flat), already checked above

                    # Kelly-based bet sizing (fractional, with safety)
                    base_risk = params.get('risk_pct', 0.01)
                    kelly_frac = 0.25  # safety factor: use 25% of full Kelly
                    p_win = confidence  # P(correct direction) — blended + meta-gated
                    b_ratio = rr  # avg_win / avg_loss ≈ reward:risk ratio
                    kelly_f = (p_win * b_ratio - (1 - p_win)) / max(b_ratio, 0.01)
                    kelly_f = max(kelly_f, 0)  # never negative (don't bet against yourself)
                    kelly_risk = base_risk * (1 + kelly_frac * kelly_f)  # scale base risk by Kelly
                    kelly_risk = min(kelly_risk, base_risk * 3)  # cap at 3x base risk

                    # Drawdown scaling: reduce as DD deepens
                    dd_scale = max(0.0, 1.0 - 2.0 * portfolio_dd) if portfolio_dd < 0.15 else 0.0
                    risk = kelly_risk * regime_mult['risk'] * dd_scale * conf_scale

                    d = 1 if direction == 'LONG' else -1
                    # Apply per-TF slippage to expected fill price
                    slippage = TF_SLIPPAGE.get(tf, 0.0002)
                    if direction == 'LONG':
                        price *= (1 + slippage)   # buy at worse (higher) price
                    else:
                        price *= (1 - slippage)   # sell at worse (lower) price
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
                        _regime_idx_map = {'bull': 2, 'bear': 0, 'sideways': 1,
                                           'transition': 1, 'crash': 0, 'unknown': 1}
                        _parent_tf = TF_PARENT_MAP.get(tf)
                        _parent_dir = confluence_filter.signals.get(_parent_tf) if _parent_tf else None
                        log_trade_snapshot(
                            trade_id=trade_id, tf=tf, direction=direction,
                            entry_time=now.isoformat(), entry_price=price,
                            xgb_probs={'long': p_long, 'flat': p_flat, 'short': p_short},
                            lstm_prob=locals().get('lstm_p'),
                            meta_prob=(locals().get('meta_probs', [None])[0]
                                       if locals().get('meta_probs') is not None else None),
                            blended_conf=confidence,
                            regime_info={
                                'regime': regime,
                                'regime_idx': _regime_idx_map.get(regime, 1),
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
                                'confluence_parent_tf': _parent_tf,
                                'confluence_parent_dir': _parent_dir,
                                'confluence_scale': conf_scale,
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

                    print(f"  >>> {trade_type.upper()} {direction} {tf.upper()} @ ${price:,.0f} | Conf={confidence:.1%} "
                          f"| SL=${stop_price:,.0f} TP=${tp_price:,.0f} | Lev={lev:.0f}x Risk={risk*100:.1f}%")
                else:
                    print(f"  No signal (conf {prob:.3f} below threshold {conf_thresh:.2f})")
                    try:
                        _rej_dir = 'LONG' if pred_class == 2 else ('SHORT' if pred_class == 0 else 'FLAT')
                        log_rejected_trade(
                            tf=tf, direction=_rej_dir, timestamp=now.isoformat(),
                            price=price, confidence=float(prob),
                            reason='below_threshold',
                        )
                    except Exception as e:
                        print(f"  [journal] rejected error: {e}")

                # Check open positions for exits
                conn = sqlite3.connect(TRADES_DB)
                open_trades = conn.execute(
                    "SELECT id, tf, direction, entry_price, stop_price, tp_price, entry_time, leverage, risk_pct FROM trades WHERE status='open'"
                ).fetchall()

                # Fetch current bar's high and low for accurate SL/TP checking
                bar_high, bar_low = price, price  # fallback to close
                try:
                    ohlcv_bar = live_dal.get_ohlcv_window(tf, 2)
                    if ohlcv_bar is not None and len(ohlcv_bar) > 0:
                        bar_high = float(ohlcv_bar['high'].iloc[-1])
                        bar_low = float(ohlcv_bar['low'].iloc[-1])
                except Exception:
                    pass  # fallback to close price if OHLCV fetch fails

                for trade in open_trades:
                    tid, ttf, tdir, entry, sl, tp, etime, tlev, trisk = trade
                    entry_dt = datetime.fromisoformat(etime)
                    ttf_minutes = {'5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440, '1w': 10080}.get(ttf, tf_minutes)
                    bars_held = int((now - entry_dt).total_seconds() / (ttf_minutes * 60))
                    max_h = ga_params.get(ttf, {}).get('max_hold', 4)

                    # Fetch high/low for the trade's own TF if different from current TF
                    t_high, t_low = bar_high, bar_low
                    t_open, t_close, t_vol = price, price, 0.0
                    if ttf != tf:
                        try:
                            ohlcv_ttf = live_dal.get_ohlcv_window(ttf, 2)
                            if ohlcv_ttf is not None and len(ohlcv_ttf) > 0:
                                t_open = float(ohlcv_ttf['open'].iloc[-1])
                                t_high = float(ohlcv_ttf['high'].iloc[-1])
                                t_low = float(ohlcv_ttf['low'].iloc[-1])
                                t_close = float(ohlcv_ttf['close'].iloc[-1])
                                t_vol = float(ohlcv_ttf['volume'].iloc[-1]) if 'volume' in ohlcv_ttf.columns else 0.0
                        except Exception:
                            t_high, t_low = price, price
                    else:
                        try:
                            ohlcv_bar_ref = live_dal.get_ohlcv_window(tf, 2)
                            if ohlcv_bar_ref is not None and len(ohlcv_bar_ref) > 0:
                                t_open = float(ohlcv_bar_ref['open'].iloc[-1])
                                t_close = float(ohlcv_bar_ref['close'].iloc[-1])
                                t_vol = float(ohlcv_bar_ref['volume'].iloc[-1]) if 'volume' in ohlcv_bar_ref.columns else 0.0
                        except Exception:
                            pass

                    # --- Journal: log price path bar for MAE/MFE tracking ---
                    try:
                        log_price_path_bar(
                            trade_id=tid, bar_num=bars_held,
                            timestamp=now.isoformat(),
                            ohlcv={'open': t_open, 'high': t_high, 'low': t_low,
                                   'close': t_close, 'volume': t_vol},
                            entry_price=entry, direction=tdir,
                            sl=sl, tp=tp,
                        )
                    except Exception as e:
                        print(f"  [journal] price_path error: {e}")

                    # ── Fix 10a: Check SL/TP against bar high/low, not just close ──
                    if tdir == 'LONG':
                        sl_hit = t_low <= sl
                        tp_hit = t_high >= tp
                    else:  # SHORT
                        sl_hit = t_high >= sl
                        tp_hit = t_low <= tp
                    time_exit = bars_held >= max_h

                    # ── Fix 10b: Trailing stop logic ──
                    # Get ATR for trailing computation
                    t_atr = feat_dict.get('atr_14', price * 0.01)
                    stop_dist = abs(entry - sl)  # original stop distance = 1R

                    # Classify trade type for trail_mult and partial_tp_pct
                    t_trade_type = classify_expected_trade_type(ttf, max_h)
                    t_tt_params = TRADE_TYPE_PARAMS.get(t_trade_type, {})
                    t_trail_mult = t_tt_params.get('trail_mult', 2.0)

                    # Update best_price tracking
                    if tid not in best_prices:
                        best_prices[tid] = entry  # initialize to entry price
                    if tdir == 'LONG':
                        best_prices[tid] = max(best_prices[tid], t_high)
                    else:
                        best_prices[tid] = min(best_prices[tid], t_low)

                    # Check if position has reached 1R profit (price moved stop_distance in right direction)
                    trail_hit = False
                    trail_stop = sl  # default to original SL
                    if tdir == 'LONG':
                        in_profit_1r = best_prices[tid] >= entry + stop_dist
                    else:
                        in_profit_1r = best_prices[tid] <= entry - stop_dist

                    if in_profit_1r:
                        # Compute trailing stop
                        if tdir == 'LONG':
                            trail_stop = best_prices[tid] - t_trail_mult * t_atr
                            # Only tighten: trailing stop must be above original SL
                            trail_stop = max(trail_stop, sl)
                            trail_hit = t_low <= trail_stop
                        else:
                            trail_stop = best_prices[tid] + t_trail_mult * t_atr
                            trail_stop = min(trail_stop, sl)
                            trail_hit = t_high >= trail_stop

                    # ── Fix 10c: Partial take profit ──
                    partial_tp_pct = t_tt_params.get('partial_tp_pct', 0.0)
                    if tp_hit and partial_tp_pct > 0 and tid not in partial_tp_taken:
                        # Realize partial profit, move stop to breakeven, keep position open
                        d = 1 if tdir == 'LONG' else -1
                        exit_slippage = TF_SLIPPAGE.get(ttf, 0.0002)
                        # Exit price at the TP barrier price (not close)
                        partial_exit_price = tp
                        if tdir == 'SHORT':
                            partial_exit_price *= (1 + exit_slippage)
                        else:
                            partial_exit_price *= (1 - exit_slippage)
                        pchange = (partial_exit_price - entry) / entry * d
                        gross = pchange * tlev
                        fee = FEE_RATE * tlev
                        net = gross - fee
                        balance = conn.execute("SELECT balance FROM account WHERE id=1").fetchone()[0]
                        partial_pnl = balance * (trisk / 100) * net * partial_tp_pct
                        partial_pnl_pct = partial_pnl / balance * 100

                        # Update balance for partial realization
                        new_balance = balance + partial_pnl
                        peak = conn.execute("SELECT peak_balance FROM account WHERE id=1").fetchone()[0]
                        new_peak = max(peak, new_balance)
                        dd = (new_peak - new_balance) / new_peak if new_peak > 0 else 0

                        conn.execute("""UPDATE account SET balance=?, peak_balance=?,
                            max_dd=?, updated_at=? WHERE id=1""",
                            (new_balance, new_peak,
                             max(dd, conn.execute("SELECT max_dd FROM account WHERE id=1").fetchone()[0]),
                             now.isoformat()))

                        conn.execute("INSERT INTO equity_curve (timestamp, balance, dd_pct) VALUES (?, ?, ?)",
                                     (now.isoformat(), new_balance, dd * 100))

                        # Move stop to breakeven
                        conn.execute("UPDATE trades SET stop_price=? WHERE id=?", (entry, tid))
                        sl = entry  # update local var too

                        partial_tp_taken[tid] = True
                        print(f"  ~~~ PARTIAL TP {tdir} {ttf.upper()} | {partial_tp_pct*100:.0f}% taken "
                              f"| PnL=${partial_pnl:+.2f} ({partial_pnl_pct:+.1f}%) | SL moved to BE ${entry:,.0f}")

                        # Position stays open — do NOT close, skip to trailing/time exit check
                        tp_hit = False  # prevent full close on TP this bar

                        # Update per-TF pool for partial realization
                        if ttf in tf_pools:
                            tf_pools[ttf]['balance'] += partial_pnl
                            tf_pools[ttf]['peak'] = max(tf_pools[ttf]['peak'], tf_pools[ttf]['balance'])

                    # ── Determine exit ──
                    if sl_hit or tp_hit or trail_hit or time_exit:
                        d = 1 if tdir == 'LONG' else -1
                        exit_slippage = TF_SLIPPAGE.get(ttf, 0.0002)

                        # Fix 10a: exit price = barrier price for SL/TP/TRAIL hits, close for TIME
                        if sl_hit:
                            exit_price = sl  # fill at stop price
                        elif trail_hit:
                            exit_price = trail_stop  # fill at trailing stop
                        elif tp_hit:
                            exit_price = tp  # fill at TP price
                        else:
                            exit_price = price  # time exit at current close

                        # Apply slippage to exit price
                        if tdir == 'SHORT':
                            exit_price *= (1 + exit_slippage)
                        else:
                            exit_price *= (1 - exit_slippage)

                        pchange = (exit_price - entry) / entry * d
                        gross = pchange * tlev
                        fee = FEE_RATE * tlev
                        net = gross - fee
                        balance = conn.execute("SELECT balance FROM account WHERE id=1").fetchone()[0]

                        # If partial TP was already taken, remaining risk is reduced
                        effective_risk = trisk
                        if tid in partial_tp_taken:
                            effective_risk = trisk * (1 - partial_tp_pct)  # remaining fraction

                        pnl = balance * (effective_risk / 100) * net
                        pnl_pct = pnl / balance * 100
                        reason = 'SL' if sl_hit else ('TRAIL' if trail_hit else ('TP' if tp_hit else 'TIME'))

                        new_balance = balance + pnl
                        peak = conn.execute("SELECT peak_balance FROM account WHERE id=1").fetchone()[0]
                        new_peak = max(peak, new_balance)
                        dd = (new_peak - new_balance) / new_peak if new_peak > 0 else 0
                        wins = conn.execute("SELECT wins FROM account WHERE id=1").fetchone()[0]
                        losses = conn.execute("SELECT losses FROM account WHERE id=1").fetchone()[0]
                        total = conn.execute("SELECT total_trades FROM account WHERE id=1").fetchone()[0]

                        conn.execute("""UPDATE trades SET exit_price=?, exit_time=?, pnl=?, pnl_pct=?,
                            bars_held=?, exit_reason=?, status='closed' WHERE id=?""",
                            (exit_price, now.isoformat(), pnl, pnl_pct, bars_held, reason, tid))

                        conn.execute("""UPDATE account SET balance=?, peak_balance=?,
                            total_trades=?, wins=?, losses=?, max_dd=?, updated_at=? WHERE id=1""",
                            (new_balance, new_peak, total + 1,
                             wins + (1 if pnl > 0 else 0),
                             losses + (1 if pnl <= 0 else 0),
                             max(dd, conn.execute("SELECT max_dd FROM account WHERE id=1").fetchone()[0]),
                             now.isoformat()))

                        conn.execute("INSERT INTO equity_curve (timestamp, balance, dd_pct) VALUES (?, ?, ?)",
                                     (now.isoformat(), new_balance, dd * 100))

                        partial_note = " (after partial TP)" if tid in partial_tp_taken else ""
                        print(f"  <<< CLOSED {tdir} {ttf.upper()} @ ${exit_price:,.0f} | PnL=${pnl:+.2f} ({pnl_pct:+.1f}%) [{reason}]{partial_note}")

                        # --- Journal: log trade outcome ---
                        try:
                            log_trade_outcome(
                                trade_id=tid, tf=ttf, direction=tdir,
                                entry_price=entry, exit_price=exit_price,
                                pnl=pnl, pnl_pct=pnl_pct,
                                exit_reason=reason, bars_held=bars_held,
                                predicted_dir=tdir, confidence=None,
                            )
                        except Exception as e:
                            print(f"  [journal] outcome error: {e}")

                        # Clean up tracking dicts
                        best_prices.pop(tid, None)
                        partial_tp_taken.pop(tid, None)

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

            time.sleep(5)

        except KeyboardInterrupt:
            _shutdown = True
            break
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            time.sleep(30)

    # ── Shutdown: log final status of open trades ──
    print("\n[SHUTDOWN] Logging final status of open trades...")
    try:
        conn = sqlite3.connect(TRADES_DB)
        open_trades = conn.execute(
            "SELECT id, tf, direction, entry_price, stop_price, tp_price, entry_time FROM trades WHERE status='open'"
        ).fetchall()
        if open_trades:
            for t in open_trades:
                print(f"  OPEN trade #{t[0]}: {t[2]} {t[1].upper()} entry=${t[3]:,.0f} SL=${t[4]:,.0f} TP=${t[5]:,.0f} since {t[6]}")
        else:
            print("  No open trades.")
        conn.close()
    except Exception as e:
        print(f"  [SHUTDOWN] Could not log open trades: {e}")
    print("[SHUTDOWN] Graceful shutdown complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='paper', choices=['paper', 'live'])
    args = parser.parse_args()
    run_trading_loop(args.mode)
