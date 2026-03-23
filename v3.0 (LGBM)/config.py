"""
Savage22 Crypto Matrix V2 — Configuration
==========================================
Multi-asset universal matrix. 31 training assets. 2-3M sparse features.
"""
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
V1_DIR = os.environ.get("SAVAGE22_V1_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
TIMEFRAMES_CRYPTO_ONLY = ['5m', '15m', '1h', '4h']

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
# Overridden by optimal_allocation.json if it exists
TF_CAPITAL_ALLOC = {
    '5m':  0.10,
    '15m': 0.15,
    '1h':  0.20,
    '4h':  0.25,
    '1d':  0.20,
    '1w':  0.10,
}

# ── Per-TF Slippage Model ──
# Applied to entry AND exit prices (both directions)
TF_SLIPPAGE = {
    '5m':  0.0005,   # 0.05% - lower liquidity on short TFs
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
TF_HIERARCHY = ['1w', '1d', '4h', '1h', '15m', '5m']  # highest to lowest
TF_PARENT_MAP = {
    '5m': '15m',
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
    '5m':  {'scalp': 12,  'day': 72,  'swing': 288},
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
    'rdx_', 'ax2_', 'ta2_', 'hod_', 'mx_', 'vx_',
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
    "max_bin": 15,
    "num_threads": -1,
    "is_enable_sparse": True,
    "min_data_in_leaf": 3,
    "min_gain_to_split": 2.0,
    "lambda_l1": 0.5,
    "lambda_l2": 3.0,
    "feature_fraction": 0.10,
    "feature_fraction_bynode": 0.5,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "num_leaves": 63,
    "learning_rate": 0.03,
    "verbosity": -1,
}

# Per-TF min_data_in_leaf overrides (rare astro conjunctions fire 10-20x on daily)
TF_MIN_DATA_IN_LEAF = {
    '1w': 3,
    '1d': 3,
    '4h': 5,
    '1h': 8,
    '15m': 15,
    '5m': 15,
}

# ── Optuna Optimizer Config ──
OPTUNA_N_TRIALS = 200
OPTUNA_N_STARTUP_TRIALS = 25
OPTUNA_SEED = 42
