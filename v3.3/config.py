"""
Savage22 Crypto Matrix V3.3 — Configuration
=============================================
Multi-asset universal matrix. 2.9M+ sparse features. BTC 2010-2026.
V3.3: LightGBM + Optuna, no 5m, institutional risk framework, pip + SCP deploy.
"""
import os

# Sparse CSR NNZ overflow threshold — LightGBM PR #1719 fixed int64 indptr support
INT32_MAX = 2_147_483_647

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
V1_DIR = os.environ.get("SAVAGE22_V1_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Shared Data Directory ──
# v3.1 reads feature parquets/npz from v3.0 to avoid duplicating ~30GB of data.
# Training outputs (models, configs, predictions) are written to PROJECT_DIR.
V30_DATA_DIR = os.environ.get("V30_DATA_DIR",
    os.path.join(os.path.dirname(PROJECT_DIR), "v3.0 (LGBM)"))

# Pipeline manifest for checkpoint/resume
PIPELINE_MANIFEST = os.path.join(PROJECT_DIR, "pipeline_manifest.json")

# ── Databases ──
SAVAGE_DB = os.path.join(PROJECT_DIR, "savage22.db")
BTC_DB = os.path.join(PROJECT_DIR, "btc_prices.db")
MULTI_ASSET_DB = os.path.join(PROJECT_DIR, "multi_asset_prices.db")
V2_SIGNALS_DB = os.path.join(PROJECT_DIR, "v2_signals.db")
CHROMA_DIR = os.path.join(PROJECT_DIR, "chroma_db")

# V1 databases (shared — tweets, news, sports, astro, etc.)
V1_TWEETS_DB = os.path.join(V1_DIR, "tweets.db")
V1_NEWS_DB = os.path.join(V1_DIR, "news_articles.db")
V1_ASTRO_DB = os.path.join(V1_DIR, "astrology_full.db")
V1_EPHEMERIS_DB = os.path.join(V1_DIR, "ephemeris_cache.db")
V1_FEAR_GREED_DB = os.path.join(V1_DIR, "fear_greed.db")
V1_SPORTS_DB = os.path.join(V1_DIR, "sports_results.db")
V1_SPACE_WEATHER_DB = os.path.join(V1_DIR, "space_weather.db")
V1_MACRO_DB = os.path.join(V1_DIR, "macro_data.db")
V1_ONCHAIN_DB = os.path.join(V1_DIR, "onchain_data.db")
V1_FUNDING_DB = os.path.join(V1_DIR, "funding_rates.db")
V1_OI_DB = os.path.join(V1_DIR, "open_interest.db")
V1_GOOGLE_TRENDS_DB = os.path.join(V1_DIR, "google_trends.db")
V1_LLM_CACHE_DB = os.path.join(V1_DIR, "llm_cache.db")

# Esoteric KB for cross-referencing
ESOTERIC_KB_DIR = os.environ.get("ESOTERIC_KB_DIR",
    r"C:\Users\C\Desktop\MY GOOGLE DRIVE\Orgonite master" if os.name == 'nt'
    else os.path.expanduser("~/Orgonite_master"))

# ── Asset Definitions ──
TRAINING_CRYPTO = ['BTC', 'ETH', 'XRP', 'LTC', 'SOL', 'DOGE', 'ADA', 'BNB',
                   'AVAX', 'LINK', 'DOT', 'MATIC', 'UNI', 'AAVE']

TRAINING_STOCKS = ['SPY', 'QQQ', 'TSLA', 'MSTR', 'ARKK', 'GLD', 'SLV', 'USO',
                   'UNG', 'WEAT', 'CORN', 'DBA', 'EWJ', 'FXI', 'EWG', 'XLE', 'XLF']

INVERSE_SIGNALS = ['UUP', 'TLT', 'FXY']  # Features only, not training targets

ALL_TRAINING = TRAINING_CRYPTO + TRAINING_STOCKS  # 31 total

# Crypto symbols to detect in messages (expanded for V2)
SYMBOLS = ["BTC", "XRP", "ETH", "XDC", "XLM", "QNT", "HBAR", "ADA", "SOL",
           "DOGE", "LINK", "DOT", "MATIC", "AVAX", "UNI", "ALGO", "VET",
           "BNB", "AAVE", "LTC"]

# ── BTC Genesis (for natal chart transits) ──
BTC_GENESIS_DATE = "2009-01-03"
BTC_GENESIS_TIME = "18:15:00"  # UTC, genesis block timestamp

# ── Timeframes ──
# Multi-asset at daily+: all 31 tickers
# Multi-asset at intraday: crypto only (stocks trade 6.5 hrs/day)
TIMEFRAMES_ALL_ASSETS = ['1d', '1w']
TIMEFRAMES_CRYPTO_ONLY = ['15m', '1h', '4h']  # 5m dropped in v3.1 (esoteric signals meaningless at 5m)

# Signal type keywords for auto-tagging
SIGNAL_KEYWORDS = {
    "gematria": ["gematria", "gematrix", "ordinal", "reverse ordinal", "cipher",
                 "english extended", "jewish", "sumerian", "isopsephy"],
    "numerology": ["numerology", "master number", "angel number", "vibration",
                   "11:11", "22", "33", "44", "55", "66", "77", "88", "99",
                   "111", "222", "333", "444", "555", "666", "777", "888", "999",
                   "dr matrix", "magic number", "pythagorean", "chaldean"],
    "astrology": ["astrology", "zodiac", "mercury retrograde", "full moon",
                  "new moon", "eclipse", "equinox", "solstice", "saturn",
                  "jupiter", "mars", "venus", "conjunction", "opposition",
                  "natal", "transit", "horoscope"],
    "technical": ["support", "resistance", "breakout", "breakdown", "fibonacci",
                  "fib", "ema", "sma", "rsi", "macd", "bollinger", "volume",
                  "divergence", "bull flag", "bear flag", "head and shoulders",
                  "cup and handle", "wedge", "triangle", "channel",
                  "long", "short", "buy", "sell", "entry", "target", "stop loss",
                  "take profit", "tp", "sl"],
    "decode": ["decode", "decoded", "decoding", "hidden message", "symbolism",
               "riddle", "riddler", "hint", "clue", "signal", "tweet",
               "elon", "musk", "schwartz", "dorsey"],
}

# Side detection patterns
LONG_PATTERNS = ["long", "buy", "bull", "pump", "moon", "breakout", "upside",
                 "accumulate", "load up", "going up", "higher", "green"]
SHORT_PATTERNS = ["short", "sell", "bear", "dump", "crash", "breakdown",
                  "downside", "drop", "lower", "red", "flush"]

# ── Capital Allocation per TF (single source of truth) ──
# Overridden by optimal_allocation.json if Optuna has optimized it
TF_CAPITAL_ALLOC = {
    '15m': 0.20,
    '1h':  0.25,
    '4h':  0.25,
    '1d':  0.20,
    '1w':  0.10,
}

# ── Per-TF Slippage Model ──
# Applied to entry AND exit prices (both directions)
TF_SLIPPAGE = {
    '15m': 0.0003,   # 0.03%
    '1h':  0.0002,   # 0.02%
    '4h':  0.0001,   # 0.01%
    '1d':  0.00005,  # 0.005%
    '1w':  0.00005,  # 0.005%
}


def load_tf_allocation():
    """Load capital allocation from optimal_allocation.json or use defaults."""
    import json as _json
    alloc_path = os.path.join(PROJECT_DIR, 'optimal_allocation.json')
    if os.path.exists(alloc_path):
        with open(alloc_path, 'r') as f:
            return _json.load(f)
    return TF_CAPITAL_ALLOC.copy()


# ── Cross-TF Confluence Hierarchy ──
# Higher TFs filter lower TF entries. Parent = one level above.
TF_HIERARCHY = ['1w', '1d', '4h', '1h', '15m']  # highest to lowest
TF_PARENT_MAP = {
    '15m': '1h',
    '1h': '4h',
    '4h': '1d',
    '1d': None,   # no parent filter (IS the trend)
    '1w': None,   # no parent filter (IS the trend)
}

# ── Trade Type Parameters ──
# Applied as modifiers to SL width, TP target, partial TP based on trade classification.
TRADE_TYPE_PARAMS = {
    'scalp': {
        'max_correlation_positions': 1,  # only 1 scalp at a time
        'sl_tightness': 1.0,             # standard
        'tp_aggression': 1.5,            # quick profit taking
        'partial_tp_pct': 0.75,          # take 75% at first TP
        'trail_mult': 1.5,              # tight trailing for scalps
    },
    'day_trade': {
        'max_correlation_positions': 2,
        'sl_tightness': 1.0,
        'tp_aggression': 1.2,
        'partial_tp_pct': 0.50,
        'trail_mult': 2.0,              # standard trailing
    },
    'swing': {
        'max_correlation_positions': 3,
        'sl_tightness': 0.8,             # wider stops
        'tp_aggression': 1.0,
        'partial_tp_pct': 0.25,          # let it run
        'trail_mult': 2.5,              # wider trailing for swings
    },
    'position': {
        'max_correlation_positions': 2,
        'sl_tightness': 0.6,             # widest stops
        'tp_aggression': 0.8,            # patience
        'partial_tp_pct': 0.0,           # no partial, ride the trend
        'trail_mult': 3.0,              # widest trailing for positions
    },
}

# Trade type classification thresholds (bars held) per TF
# Single source of truth — imported by backtesting_audit.py and live_trader.py
TRADE_THRESHOLDS = {
    '15m': {'scalp': 4,   'day': 48,  'swing': 192},
    '1h':  {'scalp': 1,   'day': 24,  'swing': 72},
    '4h':  {'scalp': 1,   'day': 6,   'swing': 84},
    '1d':  {'scalp': 0,   'day': 1,   'swing': 30},
    '1w':  {'scalp': 0,   'day': 0,   'swing': 4},
}

# ── Protected Feature Prefixes (NEVER prune/filter/remove) ──
# Core philosophy: esoteric signals ARE the edge. Any pruning that removes
# features matching these prefixes is a philosophy violation.
PROTECTED_FEATURE_PREFIXES = [
    'gem_', 'dr_', 'gematria', 'digital_root',
    'tweet', 'caution', 'pump', 'misdirection',
    'moon_', 'nakshatra', 'eclipse', 'equinox',
    'vedic_', 'bazi_', 'tzolkin', 'arabic_lot',
    'hebrew_', 'shmita',
    'sw_',
    'sport', 'horse', 'upset',
    'onchain', 'funding', 'oi_', 'liq_', 'whale',
    'master_', 'angel', 'palindrome',
    'doy_', 'dx_', 'ax_', 'ex2_', 'asp_', 'pn_', 'mn_',
    # Cross-generator prefixes
    'ax2_', 'ta2_', 'hod_', 'mx_', 'vx_',
    # Esoteric prefixes matching actual features
    'price_dr', 'price_angel', 'price_master', 'price_near_round', 'price_repeating',
    'date_dr', 'date_palindrome',
    'vortex_', 'sephirah', 'chinese_new_year', 'diwali_', 'ramadan_',
    'news_gem', 'news_sentiment', 'news_bull', 'news_bear', 'headline_gem',
    'cross_', 'aspect_count',
    # Esoteric cycle and astro prefixes
    'schumann_', 'chakra_', 'jupiter_', 'mercury_', 'planetary_',
    'saros_', 'metonic_', 'news_astro_', 'game_astro_',
]


def validate_no_protected_removed(original_cols, final_cols):
    """Verify no protected features were removed. Raises ValueError if violated."""
    final_set = set(final_cols)
    missing = [c for c in original_cols if any(c.startswith(p) for p in PROTECTED_FEATURE_PREFIXES) and c not in final_set]
    if missing:
        raise ValueError(f"PHILOSOPHY VIOLATION: {len(missing)} protected features removed: {missing[:10]}")


# ── Per-TF Feature Filter ──
# MATRIX PHILOSOPHY: NO FILTERING — the model decides via tree splits, not us.
# apply_tf_feature_filter() is a no-op for ALL timeframes.
# TF_1W_PROVEN_FEATURES kept as documentation only (historical reference).
# 1w PROVEN feature list — features with gain > 0 from model training,
# plus broad astro cycles that may need more data to prove themselves.
# 141 active features out of 621 tested. Kept as reference for SHAP auditing.
TF_1W_PROVEN_FEATURES = [
    # ON-CHAIN (highest avg gain — cycle detection)
    'puell_multiple', 'puell_multiple_high', 'onchain_miners_revenue',
    'onchain_difficulty', 'onchain_hash_rate', 'onchain_n_transactions',
    'hash_ribbon_60_128', 'hash_ribbon_compression',
    # MA/BB (35 active — backbone of weekly analysis)
    'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_100', 'sma_200',
    'sma_200_slope', 'sma_50_slope', 'sma_20_slope',
    'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_200',
    'close_vs_sma_5', 'close_vs_sma_10', 'close_vs_sma_20',
    'close_vs_sma_50', 'close_vs_sma_100', 'close_vs_sma_200',
    'close_vs_ema_5', 'close_vs_ema_10', 'close_vs_ema_20',
    'close_vs_ema_100', 'close_vs_ema_200',
    'bb_lower_20', 'bb_upper_20', 'bb_pctb_20', 'bb_width_20',
    'bb_pctb_20_lag4', 'bb_pctb_20_lag8',
    'donchian_lower', 'donchian_upper', 'obv_sma_20',
    # MOON (17 active — lunar cycles proven on weekly)
    'return_since_full_moon', 'moon_phase_cos', 'moon_phase_sin',
    'lunar_node_cycle_cos', 'lunar_node_cycle_sin', 'lunar_node_sign_idx',
    'bars_since_full_moon', 'moon_approach_return_3d', 'moon_approach_return_5d',
    'moon_modality', 'moon_x_trend', 'full_moon_decay_fast', 'full_moon_decay_slow',
    'tweet_astro_moon_phase_day', 'cross_moon_x_sport_upset',
    'cross_new_moon_x_bear', 'cross_sport_upset_x_full_moon',
    # SPACE WEATHER (5 active — Schumann resonance cycles)
    'schumann_783d_sin', 'schumann_783d_cos',
    'schumann_133d_sin', 'schumann_133d_cos', 'schumann_143d_sin',
    # ASTRO (10 active — specific aspects + arabic lots)
    'arabic_lot_catastrophe', 'arabic_lot_treachery',
    'asp_saturn_uranus_square', 'asp_sun_jupiter_trine', 'asp_sun_mars_square',
    'eclipse_decay_fast', 'eclipse_proximity_days', 'equinox_proximity',
    'west_soft_aspects', 'west_zodiac_sign_idx',
    # MACRO (32 active)
    'macro_coin', 'macro_coin_roc20d', 'macro_coin_roc5d',
    'macro_dxy', 'macro_gold_roc20d',
    'macro_hyg', 'macro_hyg_roc20d', 'macro_hyg_roc5d',
    'macro_mstr_roc20d', 'macro_mstr_roc5d',
    'macro_nasdaq_roc20d', 'macro_oil', 'macro_oil_roc20d', 'macro_oil_roc5d',
    'macro_russell', 'macro_russell_roc20d',
    'macro_silver', 'macro_silver_roc20d',
    'macro_spx', 'macro_spx_roc20d',
    'macro_tlt', 'macro_tlt_roc5d',
    'macro_us10y', 'macro_us10y_roc20d', 'macro_vix_roc5d',
    'btc_dxy_corr', 'btc_dxy_corr_13w_chg',
    'btc_spx_corr', 'btc_spx_corr_13w_chg', 'btc_spx_corr_26w_chg',
    'btc_tlt_corr_30d', 'btc_vix_corr',
    # CYCLE
    'halving_proximity', 'fib_13_from_high', 'fib_21_from_low',
    'frac_diff_close_0.2', 'frac_diff_close_0.4', 'frac_diff_close_0.6',
    'frac_diff_log_close_0.4', 'frac_diff_obv_0.3',
    # REGIME / KNN
    'knn_best_match_dist', 'knn_pattern_std',
    # OTHER (proven useful)
    'wyckoff_sc_bars_ago', 'avwap_from_swing_high', 'avwap_from_swing_low',
    'avwap_position', 'keltner_lower',
    'high_greed_decay_fast', 'high_fear_decay_fast', 'high_fear_decay_slow',
    'bars_since_high_fear', 'bars_since_high_greed',
    'chinese_year_animal', 'chinese_year_element',
    'cross_schumann_peak_x_funding', 'cross_vol_score_x_trend',
    'volume_sma_20', 'volume_ratio', 'volume_x_atr',
    'adx_14', 'month_cos', 'year_dr', 'yield_curve_proxy',
    'return_4bar', 'return_12bar', 'max_return_streak_10', 'streak_green',
    # BROAD ASTRO TO TEST (not yet proven but theoretically relevant for weekly)
    'saros_cycle', 'metonic_cycle',  # 18/19-year eclipse cycles
    'asp_jupiter_saturn_conjunction',  # 20-year great conjunction cycle
    'price_200w_ratio', 'price_200w_ratio_zscore',  # cycle position
]


def apply_tf_feature_filter(df, tf_name):
    """No-op. Matrix philosophy: the model decides via tree splits, not us.
    Whitelist removed 2026-03-29: was the last human filter, violated NO FILTERING rule.
    With min_data_in_leaf=8 and 818 weekly rows, DOY/gematria/binarized features
    appear ~8x — viable for LightGBM splits. EFB handles dimensionality.
    """
    return df


# ── TF-Aware Zero-Variance Feature Trimming ──
# These features are STRUCTURALLY constant (zero variance) at the given TF resolution.
# NOT signal filtering — these literally produce one value for every row.
# Removing them also eliminates thousands of useless cross features downstream.
#
# 1w (weekly candles close same day/hour every week):
#   hour_sin, hour_cos  — same UTC hour every bar (constant)
#   dow_sin, dow_cos    — same day-of-week every bar (constant)
#   day_of_week         — same integer every bar (constant)
#   is_monday           — always 0 (weekly bars close Sunday/Monday depending on exchange)
#   is_friday           — always 0 (same reason)
#   is_weekend          — always 0 or always 1 (same reason)
#   day_of_month        — weekly bars land on 7 possible days, but exchange weekly close
#                         is always the same weekday, so day_of_month varies BUT
#                         is perfectly predicted by week_of_year (redundant, not constant).
#                         Included here per audit: empirically zero variance in training data.
#   is_month_end        — weekly bar rarely lands exactly on month end (near-zero variance)
#
# 1d (daily candles always close at same hour):
#   hour_sin, hour_cos  — same UTC hour every bar (constant)
SKIP_FEATURES_1W = frozenset([
    'hour_sin', 'hour_cos',
    'dow_sin', 'dow_cos',
    'day_of_week',
    'is_monday', 'is_friday', 'is_weekend',
    'day_of_month', 'is_month_end',
])

SKIP_FEATURES_1D = frozenset([
    'hour_sin', 'hour_cos',
])

# Combined lookup: tf_name -> set of features to drop
SKIP_FEATURES_BY_TF = {
    '1w': SKIP_FEATURES_1W,
    '1d': SKIP_FEATURES_1D,
}


# ── Lean 1W Mode (AlphaNumetrix-inspired) ──
# PROBLEM: 1w has ~3500 features but only 819 rows — terrible signal-to-noise.
# SOLUTION: Keep ONLY core TA backbone (SAR, EMA, RSI) + ALL esoteric + numerology-on-TA hybrids.
# Drops redundant TA (Ichimoku, Bollinger, MACD, Stochastic, etc.) that add noise on weekly.
# AlphaNumetrix proves: EMAs + RSI + SAR backbone + numerology interpretation = edge.
# Result: ~200-300 features (from ~3500) — much better for 819 rows.
LEAN_1W_MODE = True
BINARY_1W_MODE = True  # DEPRECATED — use BINARY_TF_MODE instead
BINARY_TF_MODE = {'1w': True, '1d': True, '4h': False, '1h': False, '15m': False}  # Per-TF binary mode toggle

# TA features to KEEP in lean 1w mode (everything else gets dropped).
# These are the AlphaNumetrix backbone: SAR + EMAs + RSI + price-relative measures.
LEAN_1W_TA_KEEPLIST = frozenset([
    # --- Parabolic SAR (core backbone) ---
    'sar_value', 'sar_bullish', 'sar_flip',
    # --- EMAs (21≈20, 50, 200 — the AlphaNumetrix trio) ---
    'ema_20', 'ema_50', 'ema_200',
    'close_vs_ema_20', 'close_vs_ema_50', 'close_vs_ema_200',
    'sma_200', 'sma_200_slope', 'sma_50_slope',
    'above_sma200', 'golden_cross', 'death_cross',
    # --- RSI (the momentum backbone) ---
    'rsi_14', 'rsi_14_ob', 'rsi_14_os',
    # --- Price action essentials (keep minimal) ---
    'return_1bar', 'return_4bar', 'return_12bar',
    'atr_14',
    'volume', 'volume_ratio', 'volume_sma_20', 'volume_x_atr',
    # --- Regime (needed for position sizing) ---
    'adx_14',
    # --- Streaks (proven in 1w SHAP) ---
    'streak_green', 'max_return_streak_10',
    # --- SAR-numerology hybrids (NEW — computed in feature_library) ---
    'sar_digit_sum', 'sar_angel_match', 'sar_digital_root',
    'rsi_digit_sum', 'rsi_digital_root', 'rsi_numerology_zone',
    'ema200_digit_sum', 'ema200_digital_root',
    'price_sar_digit_diff', 'price_sar_dr_diff',
    'price_level_gematria',
    'sar_close_ratio', 'sar_close_ratio_dr',
    'ema_cross_21_50', 'ema_cross_50_200', 'price_vs_sar_flip_dr',
    # 52-week features (multi-month patterns for 9-158 bar trades)
    'return_26bar', 'return_52bar', 'rsi_26',
    'price_vs_52w_high', 'price_vs_52w_low',
    # Additional proven TA from v5 top features
    'close_vs_sma_100', 'close_vs_sma_200',
    'avwap_from_swing_high', 'avwap_from_swing_low', 'avwap_position',
    'supertrend_bullish', 'supertrend_direction',
])

# Prefixes for features that are ALWAYS kept in lean 1w mode regardless of TA keeplist.
# All esoteric, numerology, astrology, space weather, cross features, etc.
LEAN_1W_ALWAYS_KEEP_PREFIXES = tuple(PROTECTED_FEATURE_PREFIXES) + (
    # Additional esoteric/numerology prefixes not in PROTECTED list
    'date_dr', 'date_palindrome', 'digital_root', 'price_dr', 'price_contains',
    'price_is_master', 'price_near_round', 'price_repeating',
    'volume_dr', 'pump_date', 'vortex', 'sephirah', 'shemitah', 'jubilee',
    'lo_shu', 'pythagorean', 'haramein', 'angel_prox', 'base_tension',
    'year_dr', 'week_digital_root', 'month_digital_root',
    'candle_body_dr', 'candle_range_dr', 'body_dr', 'range_dr',
    'is_doji', 'is_hammer', 'is_engulfing', 'bull_engulfing', 'bear_engulfing',
    'shooting_star', 'morning_star', 'evening_star',
    # Time/calendar features (proven on weekly)
    'month_cos', 'month_sin', 'quarter',
    'halving', 'fib_',
    # Macro (proven in 1w SHAP)
    'macro_', 'btc_dxy', 'btc_spx', 'btc_tlt', 'btc_vix', 'yield_curve',
    # On-chain (highest gain in 1w)
    'puell', 'hash_ribbon', 'onchain_',
    # Fear/greed (proven)
    'high_greed', 'high_fear', 'bars_since_high',
    # KNN (proven)
    'knn_',
    # Frac diff (proven)
    'frac_diff',
    # Cycle features
    'chinese_', 'diwali', 'ramadan',
    # Regime features (needed for trading)
    'regime_', 'hmm_',
    # Decay features
    'decay_',
    # Higher-TF features
    'htf_',
    # Wyckoff/AVWAP (proven in 1w SHAP)
    'wyckoff_', 'avwap_',
    # DOY flags (binary, handled by EFB)
    'doy_', 'is_',
    # SAR-numerology hybrids
    'sar_digit', 'sar_angel', 'sar_digital', 'sar_close_ratio',
    'rsi_digit', 'rsi_digital', 'rsi_numerology',
    'ema200_digit', 'ema200_digital',
    'price_sar_d', 'price_level_gematria', 'price_vs_sar_flip',
    'ema_cross_',
    # Prime number features (sacred geometry — indivisible market energy)
    'prime_', 'price_is_prime', 'sar_is_prime', 'rsi_is_prime',
    'week_is_prime', 'doy_is_prime', 'price_prime_digit_count',
    # TA x TA and esoteric x TA crosses
    'tt_', 'et_',
    # Tweet engagement features
    'tweet_', 'engagement_',
    # Google Trends derived
    'gtrends_',
    # Target columns (must keep!)
    'target_', 'label_',
    # OHLCV (needed downstream)
    'open', 'high', 'low', 'close',
)


def apply_lean_1w_filter(df, tf_name):
    """For 1w in lean mode: keep only SAR/EMA/RSI TA + all esoteric + numerology-on-TA hybrids.
    Drops redundant TA (Ichimoku, Bollinger, MACD, Stochastic, MFI, Williams, etc.)
    Returns filtered DataFrame. No-op for non-1w timeframes or when LEAN_1W_MODE=False.
    """
    if not LEAN_1W_MODE or tf_name != '1w':
        return df

    keep_cols = []
    drop_cols = []
    for col in df.columns:
        # Always keep if in explicit TA keeplist
        if col in LEAN_1W_TA_KEEPLIST:
            keep_cols.append(col)
            continue
        # Always keep if matches an esoteric/always-keep prefix
        if any(col.startswith(p) for p in LEAN_1W_ALWAYS_KEEP_PREFIXES):
            keep_cols.append(col)
            continue
        # Everything else is redundant TA — drop it
        drop_cols.append(col)

    if drop_cols:
        print(f"    [LEAN 1W] Keeping {len(keep_cols)} features, dropping {len(drop_cols)} redundant TA features")
        print(f"    [LEAN 1W] Sample drops: {drop_cols[:15]}")
        df = df[keep_cols]

    # Validate no protected features removed
    validate_no_protected_removed(df.columns.tolist() + drop_cols, df.columns.tolist())

    return df


# ── LightGBM Training Parameters ──
# V3.0: LightGBM replaces XGBoost. CPU-only (GPU doesn't support sparse).
# Rare signal friendly: min_data_in_leaf=3 + min_gain_to_split=2.0
V3_LGBM_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "device": "cpu",
    "force_col_wise": True,
    "max_bin": 7,                      # binary features need 2 bins, 4-tier binarization needs ~5. 7 = safe ceiling. 36x less memory than 255.
    "max_depth": -1,                  # -1 = no limit; Optuna searches [4, 12]
    "num_threads": 0,                 # 0 = auto-detect via OpenMP (not -1 which is undocumented)
    # deterministic: REMOVED — kills GPU + multi-thread perf. Reproducibility via CPCV seed instead.
    "feature_pre_filter": False,      # CRITICAL: True silently kills rare esoteric features at Dataset construction
    "is_enable_sparse": True,
    "min_data_in_bin": 1,              # allow bins with 1 sample (rare esoteric signals)
    "bin_construct_sample_cnt": 5000,  # default 200K is 40x overkill for binary features → 10-30% construction speedup
    "path_smooth": 0.5,                # regularization: smooths leaf toward parent — 0.5 = 3.2% dampening at n=15 (was 2.0 = 21%)
    "min_data_in_leaf": 3,
    "min_sum_hessian_in_leaf": 1.5,    # adaptive guard — shrinks as model overfits
    "min_gain_to_split": 2.0,
    "lambda_l1": 0.5,
    "lambda_l2": 3.0,
    "feature_fraction": 0.9,
    "feature_fraction_bynode": 0.8,
    "bagging_fraction": 0.95,          # rare signal protection: P(10-fire in bag) = 0.95^10 = 59.9% (was 0.8 = 10.7%)
    "bagging_freq": 1,
    "num_leaves": 63,
    "learning_rate": 0.03,
    "verbosity": -1,
}

# Per-TF min_data_in_leaf overrides
# CRITICAL: must be <= rare signal frequency (10-20 firings). Higher values make rare
# esoteric signals INVISIBLE to LightGBM (can't create leaf with fewer samples than this).
# Compensating regularization: min_gain_to_split=2.0, lambda_l1/l2, path_smooth=0.5.
# Perplexity-confirmed 2026-03-29: old values (30/50) were same bug class as feature_fraction=0.005.
TF_MIN_DATA_IN_LEAF = {
    '1w': 2,   # 819 rows — must catch signals firing 5-10x. Floor=2 lets Optuna search [2,8].
    '1d': 8,   # 5.7K rows — lowered to 8: P(8-fire in bag@bf=0.95)>>P(10-fire) (was 10)
    '4h': 8,   # 23K rows — lowered to 8: more headroom for signals firing 8-10x (was 10)
    '1h': 8,   # 75K rows — lowered to 8: matches 1w (was 10)
    '15m': 8,  # 294K rows — lowered to 8: consistent floor across all TFs (was 10)
}

# Per-TF max for min_data_in_leaf Optuna range (default=10)
# 1w: cap at 8 (not 10) — 819 rows means even 10 is aggressive for rare signals
TF_MIN_DATA_IN_LEAF_MAX = {
    '1w': 8,
}

# TFs that use force_row_wise instead of force_col_wise during training.
# 15m: 294K rows / 23K EFB bundles = 12.8 — row-wise is faster for high rows/bundles ratio.
# All others: force_col_wise (better for high feature count, low rows/bundles ratio).
TF_FORCE_ROW_WISE = frozenset(['15m'])

# ── EFB Pre-Bundle Config ──
# External pre-bundling bypasses LightGBM's O(F^2) conflict graph.
# Binary features packed 127/bundle → 79K bundles instead of 10M histograms.
# enable_bundle=False in LightGBM when pre-bundled (already done externally).
# Per-TF toggle: True = use pre-bundled matrix, False = raw sparse + LightGBM EFB.
EFB_PREBUNDLE_ENABLED = {
    '1w': True,    # 1158 rows, ~600 crosses — fast either way, but consistent
    '1d': True,    # 5.7K rows, ~23K crosses — prebundle saves build time
    '4h': True,    # 23K rows, ~2.9M crosses — major speedup
    '1h': True,    # 75K rows, ~5M crosses — critical for training speed
    '15m': True,   # 294K rows, ~10M crosses — BIGGEST win, 128x histogram reduction
}

# Per-TF class_weight — reweights by inverse class frequency
# LightGBM: applied via is_unbalance=True or class_weight param
# 1w: explicit SHORT upweight (3x) because model never predicts SHORT without it
TF_CLASS_WEIGHT = {
    '1d': {0: 3.0, 1: 1.0, 2: 1.0},  # SHORT=3x — force directional SHORT learning
    '1w': {0: 2.0, 1: 1.0, 2: 1.0},  # SHORT=2x — reduced from 3x (model was ONLY predicting SHORT)
    '4h': {0: 2.0, 1: 1.0, 2: 1.0},  # SHORT=2x — insurance (v3.2 4h had SHORT accuracy issues)
}

# Per-TF CPCV group settings (n_groups, n_test_groups) — FINAL evaluation (K=2)
# Splits = C(N,K), unique paths phi = (K/N)*C(N,K), train fraction = (N-K)/N
# Expert ML: 8 groups for 1w (75% train, 28 paths — max data with K=2 PBO), 10 for data-rich TFs
TF_CPCV_GROUPS = {
    '1w': (8, 2),    # C(8,2)=28 paths, 75% train — more data per fold for 819 rows + 50-bar purge
    '1d': (5, 2),    # C(5,2)=10 paths, 60% train — optimal for 3-20 signal fires
    '4h': (10, 2),   # C(10,2)=45 paths, 80% train — sample 30 paths
    '1h': (10, 2),   # C(10,2)=45 paths, 80% train — sample 30 paths
    '15m': (10, 2),  # C(10,2)=45 paths, 80% train — sample 30 paths
}

# ── CPCV Path Sampling (Expert ML: sample paths instead of exhaustive) ──
# When total paths > CPCV_SAMPLE_PATHS, deterministically sample a subset.
# Guarantees: every group appears in at least one test fold (full row coverage).
CPCV_N_GROUPS = 10                # default for new TFs
CPCV_SAMPLE_PATHS = 30            # training: sample 30 from C(10,2)=45 (None = exhaustive)
CPCV_OPTUNA_SAMPLE_PATHS = 15     # Optuna search: fewer paths for speed
CPCV_SAMPLE_SEED = 42             # deterministic sampling for reproducibility

# Per-TF num_leaves caps (scaled with data size — larger TFs support deeper trees)
TF_NUM_LEAVES = {
    '1w': 15,    # 819 rows — raised from 7. Perplexity: 15-31 optimal for tiny data. 15 = conservative start.
    '1d': 15,    # 5.7K rows — moderate (EFB ratio 0.28:1)
    '4h': 31,    # 23K rows — standard (EFB ratio 0.92:1)
    '1h': 63,    # 75K rows — can handle complexity (EFB ratio 3.8:1)
    '15m': 127,  # 294K rows — deep trees viable (EFB ratio 6.9:1)
}

# Per-TF learning_rate overrides (global default = 0.03 in V3_LGBM_PARAMS)
# 1w: 0.03 → 0.1 — tiny data (819 rows) needs faster learning. Each tree must make a meaningful step.
# Perplexity-confirmed: LR=0.03 on 819 rows → validation metric never improves → ES kills at tree 1.
TF_LEARNING_RATE = {
    '1w': 0.1,     # 819 rows — fast learning, compensated by fewer boost rounds
    # Other TFs use global 0.03 (omitted = use default)
}

# Per-TF num_boost_round overrides (global default = 800 via --boost-rounds CLI)
# 1w: 800 → 300 — tiny data doesn't need 800 rounds. With LR=0.1, model converges much faster.
TF_NUM_BOOST_ROUND = {
    '1w': 300,     # 819 rows — fewer rounds needed at higher LR
    # Other TFs use global 800 (omitted = use CLI default)
}

# Per-TF early stopping patience overrides
# Default formula: max(50, int(100 * (0.1 / lr))). At LR=0.03 → 333, at LR=0.1 → 100.
# 1w: override to 50 — on 819 rows, genuine improvements happen early or not at all.
TF_ES_PATIENCE = {
    '1w': 50,      # 819 rows — don't waste 100 rounds when model stops improving
    # Other TFs use dynamic formula (omitted = use formula)
}

# SHAP analysis config
SHAP_N_SAMPLES = 10000        # High-confidence samples for SHAP analysis
SHAP_TOP_N = 1000             # Top N features by |SHAP| to report
SHAP_CROSS_PREFIXES = ('dx_', 'ax_', 'ax2_', 'ta2_', 'ex2_', 'sw_', 'hod_', 'mx_', 'vx_', 'asp_', 'mn_', 'pn_')

# ── Optuna v2: Phase 1 (rapid search) + Validation Gate ──
# Phase 1: 2 seeded + 8 random + 15 TPE = 25 trials, 2-fold CPCV, fast LR
# Validation Gate: top-3 re-evaluated with 4-fold CPCV, longer rounds
# Final retrain: unchanged (full CPCV K=2, 800 rounds, LR=0.03)
OPTUNA_SEED = 42

OPTUNA_PHASE1_TRIALS = 25              # 2 seeded + 8 random + 15 TPE
OPTUNA_PHASE1_CPCV_GROUPS = 2          # 2-fold for speed
OPTUNA_PHASE1_ROUNDS = 60              # max rounds (ES fires at ~30)
OPTUNA_PHASE1_LR = 0.15                # 5x final LR for fast convergence
OPTUNA_PHASE1_ES_PATIENCE = 15         # aggressive ES during search
OPTUNA_PHASE1_N_STARTUP = 8            # random trials before TPE

OPTUNA_VALIDATION_TOP_K = 3            # validate top-K from Phase 1
OPTUNA_VALIDATION_CPCV_GROUPS = 4      # 4-fold validation
OPTUNA_VALIDATION_ROUNDS = 200         # longer eval
OPTUNA_VALIDATION_LR = 0.08            # closer to final LR
OPTUNA_VALIDATION_ES_PATIENCE = 50     # patient for rare signals

# Warm-start (downstream TFs)
OPTUNA_WARMSTART_PHASE1_TRIALS = 15    # fewer trials needed
OPTUNA_WARMSTART_VALIDATION_TOP_K = 2  # fewer validation runs

# Row subsampling for search (final model always uses ALL rows)
OPTUNA_TF_ROW_SUBSAMPLE = {
    '1w': 1.0,    # 1158 rows — can't subsample
    '1d': 1.0,    # 5733 rows — use all
    '4h': 1.0,    # 23K rows — sparse fix eliminates OOM, use all data
    '1h': 0.50,   # 75K → ~38K rows (sparse OK, subsample for speed)
    '15m': 0.25,  # 294K → ~74K rows (sparse OK, subsample for speed)
}

# Per-TF trial count overrides
OPTUNA_TF_PHASE1_TRIALS = {
    '1w': 15,   # was 20 — smaller search space (fewer leaves/depth combos), 15 is sufficient
    '1d': 25, '4h': 25, '1h': 25, '15m': 30,
}

# Per-TF Phase 1 LR overrides (default = OPTUNA_PHASE1_LR = 0.15)
# 1w: higher LR for tiny data — needs stronger signal per round to converge in fewer rounds
OPTUNA_TF_PHASE1_LR = {
    '1w': 0.20,  # 819 rows — higher LR for faster convergence on tiny data
}

# Per-TF Phase 1 max rounds overrides (default = OPTUNA_PHASE1_ROUNDS = 60)
OPTUNA_TF_PHASE1_ROUNDS = {
    '1w': 40,    # 819 rows — fewer rounds needed at higher LR, prevents overfitting
}

# Per-TF max_depth Optuna search range (default = [2, 8])
# 1w: floor at 3 to prevent trivially shallow trees on tiny data
OPTUNA_TF_MAX_DEPTH_RANGE = {
    '1w': (3, 8),    # tiny data — floor=3 prevents 2-deep stumps, cap=8 prevents memorization
    '1d': (3, 8),    # moderate data — same logic
}

# Per-TF learning_rate Optuna search range (default = None, meaning fixed LR from phase config)
# When set, learning_rate becomes a searchable hyperparameter instead of fixed
OPTUNA_TF_LR_SEARCH_RANGE = {
    '1w': (0.05, 0.3),  # tiny data — wider range, model needs to find optimal LR
}

# Keep final retrain unchanged (per-TF overrides below)
OPTUNA_FINAL_LR = 0.03
OPTUNA_FINAL_ROUNDS = 800

# Per-TF final round caps (default = OPTUNA_FINAL_ROUNDS = 800)
# 1w: 300 cap — 819 rows overfits well before 800 rounds at LR=0.03
OPTUNA_TF_FINAL_ROUNDS = {
    '1w': 300,   # 819 rows — overfits by round ~200 at LR=0.03. 300 with ES is safe cap.
    '1d': 600,   # 5.7K rows — moderate cap
}

# n_jobs: env var override or auto
OPTUNA_N_JOBS = int(os.environ.get('OPTUNA_N_JOBS', 0))  # 0 = auto (total_cores // 8). 13900K=3, 128c=16.

# ── Multi-GPU Fold-Parallel CPCV ──
# Number of GPUs to use for parallel CPCV fold training.
# 0 = auto-detect via nvidia-smi. Each fold trains on a separate GPU.
# Set CPCV_PARALLEL_GPUS env var to override (e.g., for testing on fewer GPUs).
CPCV_PARALLEL_GPUS = int(os.environ.get('CPCV_PARALLEL_GPUS', 0))  # 0 = auto-detect

# ── Fee & Cost Model (single source of truth) ──
# All files must use these — never hardcode fee/slippage separately.
FEE_RATE = 0.0018            # 0.18% round-trip (Bitget taker + conservative slippage)
PORTFOLIO_FEE_RATE = 0.0018  # same for portfolio aggregator (was 0.0012 — now consistent)

# ── Regime Multipliers (single source of truth) ──
# Every file that does regime-adjusted sizing MUST import this dict.
# Keys: 0=bull, 1=bear, 2=sideways, 3=crash
REGIME_MULT = {
    0: {'lev': 1.0,  'risk': 1.0,  'stop': 1.0,  'rr': 1.5,  'hold': 1.0},   # bull
    1: {'lev': 0.47, 'risk': 1.0,  'stop': 0.75, 'rr': 0.75, 'hold': 0.17},   # bear
    2: {'lev': 0.67, 'risk': 0.47, 'stop': 0.5,  'rr': 0.5,  'hold': 1.0},    # sideways
    3: {'lev': 0.2,  'risk': 0.25, 'stop': 0.5,  'rr': 0.5,  'hold': 0.1},    # crash
}
REGIME_NAMES = {0: 'bull', 1: 'bear', 2: 'sideways', 3: 'crash'}

# ── Regime Detection Thresholds ──
REGIME_SLOPE_THRESHOLD = 0.001   # SMA100 slope threshold for bull/bear classification
REGIME_NEAR_SMA_PCT = 0.05      # within 5% of SMA100 = "near"
REGIME_CRASH_VOL_MULT = 2.0     # rvol_20 > 2x rvol_90_avg
REGIME_CRASH_DD_THRESHOLD = 0.15 # dd_from_30h > 15%

# ── Backtest / Optimizer Defaults ──
STARTING_BALANCE = 10000.0       # backtest/optimizer starting capital
LIVE_STARTING_BALANCE = 100.0    # paper trader initial balance ($100)

# ── Live Trading Tunable Constants ──
RISK_SCALE = 2.0                 # multiplier for GA config risk_pct
KELLY_SAFETY_FRACTION = 0.25    # use 25% of full Kelly (conservative)
KELLY_MAX_RISK_MULT = 3.0       # cap Kelly-adjusted risk at 3x base
DD_HALT_THRESHOLD = 0.15        # portfolio DD at which dd_scale goes to 0
DD_SCALE_STEEPNESS = 2.0        # dd_scale = max(0, 1 - steepness * dd)
MAX_PORTFOLIO_DD = 0.15          # portfolio DD halts all new entries
MAX_TF_DD = 0.25                 # per-TF DD halts that TF
DEFAULT_CONF_THRESH = 0.60       # default confidence threshold (optimizer overrides)

# ── Confidence-Scaled Position Sizing ──
# Higher confidence = larger position. Multiplied with Kelly output.
# The model's conviction drives capital allocation.
CONFIDENCE_SIZE_TIERS = [
    (0.90, 2.5),   # 90%+ confidence → 2.5x base size
    (0.80, 2.0),   # 80-90% → 2.0x
    (0.70, 1.5),   # 70-80% → 1.5x
    (0.65, 1.0),   # 65-70% → 1.0x (baseline)
    (0.60, 0.5),   # 60-65% → 0.5x (low conviction = small size)
]  # sorted high→low, first match wins


def get_confidence_multiplier(confidence):
    """Return position size multiplier based on model confidence."""
    for threshold, mult in CONFIDENCE_SIZE_TIERS:
        if confidence >= threshold:
            return mult
    return 0.25  # below all tiers = minimum size
LIVE_CONF_THRESH_FALLBACK = 0.80 # live trader fallback if no optimizer config (more conservative)

# ── Warmup Bars (for live feature computation) ──
WARMUP_BARS = {
    '15m': 400, '1h': 300,
    '4h': 200, '1d': 100, '1w': 50,
}

# ── Database Paths ──
TRADES_DB = os.path.join(PROJECT_DIR, "trades.db")

# ══════════════════════════════════════════════════════════════════════
# TIER 1 — INSTITUTIONAL RISK FRAMEWORK (v3.1)
# ══════════════════════════════════════════════════════════════════════

# ── Hard Risk Limits ──
# Breaches auto-halt trading. No override except manual restart.
RISK_LIMITS = {
    'max_daily_loss_pct': 0.05,       # 5% of equity — halt all new entries for the day
    'max_weekly_loss_pct': 0.10,      # 10% of equity — halt all new entries for the week
    'max_open_risk_pct': 0.08,        # 8% — sum of (stop_distance × position_size) / equity
    'max_leverage': 125,              # absolute cap — regime multipliers can't push above this
    'max_concurrent_positions': 5,    # across all TFs combined
    'max_exposure_pct': 1.0,          # 100% of equity max gross notional (with leverage)
}

# ── Drawdown Protocol ──
# Pre-defined, deterministic response to drawdowns. Not reactive — pre-coded.
DRAWDOWN_PROTOCOL = {
    0.10: {  # at -10% from peak
        'action': 'reduce_risk',
        'risk_multiplier': 0.50,            # halve position sizes
        'min_confidence': None,             # no change to confidence threshold
        'description': 'Halve risk at 10% DD',
    },
    0.20: {  # at -20% from peak
        'action': 'top_confidence_only',
        'risk_multiplier': 0.25,            # quarter position sizes
        'min_confidence': 0.90,             # only take 90%+ confidence signals
        'description': 'Top-confidence only at 20% DD',
    },
    0.30: {  # at -30% from peak
        'action': 'sim_only',
        'risk_multiplier': 0.0,             # no real trades
        'min_confidence': None,
        'description': 'Simulation only at 30% DD — manual review required',
    },
}

# ── Circuit Breakers (Fat-Finger / Bug Safeguards) ──
CIRCUIT_BREAKERS = {
    'max_orders_per_minute': 3,       # prevent rapid-fire bugs
    'max_notional_per_order': 0.25,   # 25% of equity per single order
    'pnl_sanity_sigma': 5.0,          # if PnL moves >5σ vs recent distribution, kill switch
    'pnl_lookback_trades': 20,        # look at last 20 trades for σ calculation
    'stale_data_max_bars': 3,         # halt if features are >3 bars old
}

# ── Model Versioning ──
# Every live run is tagged. Reconstruct any day's decisions ex-post.
MODEL_VERSION_SCHEMA = {
    'fields': ['code_hash', 'model_id', 'feature_schema_version',
               'config_hash', 'training_date', 'cpcv_oos_sharpe',
               'optuna_best_trial', 'n_features', 'n_training_rows'],
    'version_file': os.path.join(PROJECT_DIR, 'model_version.json'),
}

# ── Promotion Pipeline ──
# Research → Paper → Limited → Full. Defined sign-off criteria.
PROMOTION_PIPELINE = {
    'paper_trade': {
        'min_trades': 200,
        'min_days': 30,
        'min_sharpe': 0.5,            # annualized, after fees
        'max_dd': 0.20,               # max drawdown during paper phase
    },
    'limited_capital': {
        'capital_pct': 0.25,          # 25% of total capital
        'min_trades': 100,
        'min_days': 14,
        'min_sharpe': 0.3,
        'max_dd': 0.15,
    },
    'full_capital': {
        'capital_pct': 1.0,
        'requires': 'limited_capital_passed',
    },
}
