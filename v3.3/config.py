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
# 1w has too few rows (1158) for thousands of features. Drop short-period TA noise.
# The matrix thesis scales with DATA — more rows = more features add value.
# Per-TF feature filtering strategy:
# 1w: WHITELIST approach — too few rows (1158) for thousands of features.
#     Only keep MAs, BBands, HMM/regime, macro, broad astro, month gematria.
#     Drop 4-tier binarized variants (_HIGH/_LOW/_EXTREME_*) and DOY.
# All other TFs: no filtering (full feature set + crosses).
# 1w PROVEN feature list — only features with gain > 0 from model training,
# plus broad astro cycles that may need more data to prove themselves.
# 141 active features out of 621 tested. Dead features are pure noise on 1w.
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

# 1w: Run 2 config was best (380 features, 22.6% SHORT precision).
# Pattern whitelist + drop 4-tier bins + drop DOY. NO cycle features (they diluted signal).
TF_FEATURE_WHITELIST = {
    '1w': [
        'open', 'high', 'low', 'close', 'volume', 'quote_volume',
        'sma_', 'ema_', 'close_vs_sma', 'close_vs_ema', 'golden_cross', 'death_cross',
        'bb_', 'hmm', 'regime', 'vol_regime',
        'return_1bar', 'return_4bar', 'return_12bar', 'return_streak', 'max_return',
        'retrograde', 'eclipse', 'equinox', 'solstice', 'zodiac_sign',
        'macro_', 'btc_spx', 'btc_dxy', 'btc_vix',
        'fear_greed', 'funding', 'oi_', 'onchain_',
        'month_num', 'month_gematria', 'caution_gematria', 'quarter',
        'trend_strength', 'adx_14', 'halving', 'knn_',
        'volume_', 'obv_', 'frac_diff_',
        'px_', 'tx_', 'cross_',
        'fib_', 'avwap', 'donchian', 'wyckoff',
        'is_quarter', 'is_month', 'opex', 'tax',
    ],
}

TF_DROP_SUFFIXES = {
    '1w': ['_HIGH', '_LOW', '_EXTREME_HIGH', '_EXTREME_LOW'],
}
TF_DROP_PREFIXES = {
    '1w': ['doy_'],
}


def apply_tf_feature_filter(df, tf_name):
    """Filter features per-TF. Pattern whitelist for 1w, no-op for others."""
    whitelist = TF_FEATURE_WHITELIST.get(tf_name)
    if whitelist is None:
        return df  # No filtering

    keep = []
    for col in df.columns:
        cl = col.lower()
        for pat in whitelist:
            if pat.lower() in cl:
                keep.append(col)
                break

    drop_suffixes = TF_DROP_SUFFIXES.get(tf_name, [])
    if drop_suffixes:
        keep = [c for c in keep if not any(c.endswith(s) for s in drop_suffixes)]

    drop_prefixes = TF_DROP_PREFIXES.get(tf_name, [])
    if drop_prefixes:
        keep = [c for c in keep if not any(c.lower().startswith(p.lower()) for p in drop_prefixes)]

    print(f"[config] TF feature filter ({tf_name}): {len(df.columns)} -> {len(keep)} features", flush=True)
    return df[keep]


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
    "max_bin": 255,                    # max EFB bundle size (254/bundle → ~23K bundles for 6M features). Binary features still get 2 bins.
    "max_depth": -1,                  # -1 = no limit; Optuna searches [4, 12]
    "num_threads": 0,                 # 0 = auto-detect via OpenMP (not -1 which is undocumented)
    "deterministic": True,            # Perplexity: required for reproducible sparse training
    "feature_pre_filter": False,      # CRITICAL: True silently kills rare esoteric features at Dataset construction
    "is_enable_sparse": True,
    "min_data_in_bin": 1,              # allow bins with 1 sample (rare esoteric signals)
    "path_smooth": 2.0,                # regularization: smooths extreme leaf predictions toward parent (was 0.1)
    "min_data_in_leaf": 3,
    "min_sum_hessian_in_leaf": 1.5,    # adaptive guard — shrinks as model overfits
    "min_gain_to_split": 2.0,
    "lambda_l1": 0.5,
    "lambda_l2": 3.0,
    "feature_fraction": 0.1,
    "feature_fraction_bynode": 0.5,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "num_leaves": 63,
    "learning_rate": 0.03,
    "verbosity": -1,
}

# Per-TF min_data_in_leaf overrides (rare astro conjunctions fire 10-20x on daily)
TF_MIN_DATA_IN_LEAF = {
    '1w': 30,   # 1158 rows — high for stability
    '1d': 50,   # 5.7K rows — research says n/100 to n/20 = 57-285. Prevents noise memorization.
    '4h': 20,   # 23K rows — moderate
    '1h': 15,   # 75K rows — can be lower (more data per leaf)
    '15m': 15,  # 294K rows — standard
}

# Per-TF class_weight — reweights by inverse class frequency
# LightGBM: applied via is_unbalance=True or class_weight param
# 1w: explicit SHORT upweight (3x) because model never predicts SHORT without it
TF_CLASS_WEIGHT = {
    '1d': {0: 3.0, 1: 1.0, 2: 1.0},  # SHORT=3x — force directional SHORT learning
    '1w': {0: 3.0, 1: 1.0, 2: 1.0},  # SHORT=3x — worked great on 1w final run
    '4h': {0: 2.0, 1: 1.0, 2: 1.0},  # SHORT=2x — insurance (v3.2 4h had SHORT accuracy issues)
}

# Per-TF CPCV group settings (n_groups, n_test_groups) — FINAL evaluation (K=2)
# Splits = C(N,K), unique paths phi = (K/N)*C(N,K), train fraction = (N-K)/N
TF_CPCV_GROUPS = {
    '1w': (5, 2),   # C(5,2)=10 splits, 4 unique paths, 60% train
    '1d': (5, 2),   # C(5,2)=10 splits, 4 unique paths, 60% train
    '4h': (6, 2),   # C(6,2)=15 splits, 5 unique paths, 67% train
    '1h': (6, 2),   # C(6,2)=15 splits, 5 unique paths, 67% train
    '15m': (6, 2),  # C(6,2)=15 splits, 5 unique paths, 67% train
}

# Per-TF num_leaves caps (scaled with data size — larger TFs support deeper trees)
TF_NUM_LEAVES = {
    '1w': 7,     # 1158 rows — v3.2 best was 7. Max 7 leaves for weekly.
    '1d': 15,    # 5.7K rows — moderate (EFB ratio 0.28:1)
    '4h': 31,    # 23K rows — standard (EFB ratio 0.92:1)
    '1h': 63,    # 75K rows — can handle complexity (EFB ratio 3.8:1)
    '15m': 127,  # 294K rows — deep trees viable (EFB ratio 6.9:1)
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

# Per-TF overrides
OPTUNA_TF_PHASE1_TRIALS = {
    '1w': 20, '1d': 25, '4h': 25, '1h': 25, '15m': 30,
}

# Keep final retrain unchanged
OPTUNA_FINAL_LR = 0.03
OPTUNA_FINAL_ROUNDS = 800

# n_jobs: env var override or auto
OPTUNA_N_JOBS = int(os.environ.get('OPTUNA_N_JOBS', 0))  # 0 = auto (total_cores // 8)

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
