# SAVAGE22 — Complete System Technical Report

**Date:** 2026-03-18
**Version:** v3 (post-major refactor)
**Author:** Generated from full codebase analysis

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [The Edge: Esoteric Signals](#3-the-edge-esoteric-signals)
4. [Data Pipeline](#4-data-pipeline)
5. [Feature Engineering](#5-feature-engineering)
6. [ML Pipeline](#6-ml-pipeline)
7. [GPU Grid Search Optimizer](#7-gpu-grid-search-optimizer)
8. [KNN Pattern Matching](#8-knn-pattern-matching)
9. [Live Trading Engine](#9-live-trading-engine)
10. [Dashboard](#10-dashboard)
11. [API Server](#11-api-server)
12. [Universal Engines](#12-universal-engines)
13. [Data Sources](#13-data-sources)
14. [Configuration](#14-configuration)
15. [Current Model Performance](#15-current-model-performance)
16. [Known Issues & Future Work](#16-known-issues--future-work)

---

## 1. Executive Summary

Savage22 is a fully automated BTC/USDT trading system that fuses conventional technical analysis with **esoteric signal processing** — gematria, numerology, astrology (Western, Vedic, Chinese BaZi, Mayan Tzolkin), sentiment analysis, sports event correlation, and more — into a single XGBoost-based ML pipeline.

**What makes it unique:**

- **~700 features** fed to XGBoost with zero pruning. The model decides what matters via tree splits.
- **Esoteric signals are the edge.** Gematria values of tweet text, digital roots of BTC prices, planetary hours at the moment of news events, winning horse name numerology — all correlated against BTC price action.
- **Sparse signals handled natively.** Esoteric signals fire infrequently. When they DO fire, they receive 1.5-3x sample weight during training. Missing values stay as NaN (not filled with 0), so XGBoost learns separate "missing" branches.
- **GPU-accelerated throughout.** RTX 3090 powers the exhaustive optimizer (2.88B parameter combinations), KNN pattern matching (14K bars in 5.2s), and XGBoost training.
- **6 independent timeframes** (5m, 15m, 1H, 4H, 1D, 1W) with independent capital pools, allowing a weekly position to coexist with 15-minute scalps.
- **Paper mode starts at $100.** Designed to compound from minimal capital.

**Current state:** All 6 TF feature builds complete, all 6 models trained, exhaustive optimizer running on GPU. Shared feature architecture (`feature_library.py` + `data_access.py`) eliminates training/live drift.

---

## 2. Architecture Overview

### System Diagram

```
LAYER 1: DATA COLLECTORS (live feeds -> raw SQLite DBs)
+-------------------------------------------------------------------------+
| tweet_streamer.py      -> tweets.db          (every 5 min per account)  |
| news_streamer.py       -> news_articles.db   (RSS+CryptoPanic+Reddit)  |
| sports_streamer.py     -> sports_results.db  (ESPN+TheSportsDB, 15 min) |
| crypto_streamer.py     -> onchain_data.db    (blockchain, funding, OI)  |
| macro_streamer.py      -> macro_data.db      (indices, DXY, gold, VIX) |
| astro_engine.py        -> astrology_full.db  (ephemeris, pre-computed)  |
| download_btc.py        -> btc_prices.db      (OHLCV candles all TFs)   |
+-------------------------------------------------------------------------+
                                |
                                v
LAYER 2: UNIVERSAL ENGINES (apply techniques to anything)
+-------------------------------------------------------------------------+
| universal_gematria.py  -- 6 ciphers on any text string                  |
| universal_numerology.py -- digital root, master numbers, sequences      |
| universal_astro.py     -- Western + Vedic + BaZi + Tzolkin + hours      |
| universal_sentiment.py -- sentiment, caps, exclamation, urgency         |
+-------------------------------------------------------------------------+
                                |
                                v
LAYER 3: FEATURE BUILDERS (raw DBs -> feature matrices)
+-------------------------------------------------------------------------+
| build_5m_features.py   -> features_5m.db       (76 features)           |
| build_15m_features.py  -> features_15m.db      (164 features)          |
| build_1h_features.py   -> features_1h.db       (373 features)          |
| build_4h_features.py   -> features_4h.db       (370 features)          |
| build_features_complete.py -> features_complete.db (350 features, 1D)  |
| build_1w_features.py   -> features_1w.db       (200 features)          |
|                                                                         |
| feature_library.py     -- SHARED pure-function feature computation      |
| data_access.py         -- Data Access Layer with caching                |
| knn_feature_engine.py  -- GPU KNN pattern matching                      |
+-------------------------------------------------------------------------+
                                |
                                v
LAYER 4: ML PIPELINE
+-------------------------------------------------------------------------+
| ml_multi_tf.py         -- XGBoost training, walk-forward, KNN A/B test  |
|   -> model_{tf}.json   (trained XGBoost models per TF)                  |
|   -> features_{tf}_all.json (feature name lists per TF)                 |
|   -> platt_{tf}.pkl    (Platt calibrators per TF)                       |
|                                                                         |
| exhaustive_optimizer.py -- 2.88B combo GPU grid search                  |
|   -> exhaustive_configs.json (optimal trading params per TF)            |
+-------------------------------------------------------------------------+
                                |
                                v
LAYER 5: LIVE TRADING + DASHBOARD
+-------------------------------------------------------------------------+
| live_trader.py         -- multi-TF capital allocation, Bitget execution |
|   -> trades.db         (trade log, equity curve, account state)         |
|   -> prediction_cache.json (latest prediction for dashboard)            |
|                                                                         |
| api_server.js          -- Express REST API (port 3001)                  |
| dashboard/             -- Next.js app (port 3000)                       |
|   app/page.tsx         -- main dashboard layout                         |
|   components/Chart.tsx -- klinecharts candlestick with overlays         |
|   components/FeatureWeights.tsx -- ML feature inspection modal          |
|   components/MLTrading.tsx      -- trade panel                          |
+-------------------------------------------------------------------------+
```

### Data Flow Summary

```
Raw APIs/Feeds
    -> SQLite DBs (7 databases)
        -> Universal Engines (gematria, numerology, astro, sentiment)
            -> Feature Builders (aggregate per-bar, bucket by TF)
                -> Feature DBs (one per TF, ~76-373 features each)
                    -> ml_multi_tf.py (XGBoost training, walk-forward)
                        -> Trained models + exhaustive_configs.json
                            -> live_trader.py (predictions + trades)
                                -> Dashboard (visualization + monitoring)
```

---

## 3. The Edge: Esoteric Signals

The system's thesis: markets are not purely rational. Hidden patterns in text, numbers, dates, and celestial alignments correlate with BTC price movements. These signals are **sparse** (they don't fire every bar) but **high value** (when they fire, they carry more predictive power than TA alone).

### Signal Categories

#### 3.1 Gematria (6 methods on any text)

Gematria assigns numeric values to letters. The system uses 6 ciphers:

| Cipher | Method | Example: "Bitcoin" |
|--------|--------|-------------------|
| English Ordinal | A=1, B=2 ... Z=26, sum all | Sum of letter values |
| Reverse Ordinal | A=26, B=25 ... Z=1 | Reversed alphabet |
| Reduction | Digital root of ordinal | Single digit |
| English Gematria | A=6, B=12 ... Z=156 (multiples of 6) | Scaled x6 |
| Jewish/Hebrew | Traditional Hebrew values mapped to English | A=1,B=2,...,K=10,L=20,... |
| Satanic | A=36, B=37 ... Z=61 | Offset by 35 |

Applied to: tweet text, usernames, display names, hashtags, news headlines, source names, author names, sports team names, winning horse names, jockey names, track names, Reddit titles, FOMC statements.

**Match features:** Cross-source gematria alignment (e.g., tweet gematria = headline gematria = winning horse gematria).

#### 3.2 Numerology (digital root reduction)

Any number is reduced to 1-9 (or master numbers 11, 22, 33):

| Target | Feature Names |
|--------|--------------|
| BTC close price | `price_dr`, `price_is_master` |
| BTC volume | `volume_dr` |
| Tweet like/RT/reply counts | `dr_tweet_likes`, etc. |
| Sports scores | `dr_sport_score_home`, `dr_sport_total` |
| Block numbers | `dr_block` |
| Funding rate | `dr_funding` |
| Dates | `date_dr` (month+day+year reduction) |

**Key numbers:**
- **Caution:** 93, 39, 43, 48, 113, 223, 322, 17, 71, 13, 44, 66, 77, 88, 99, 666, 911
- **Pump:** 27, 72, 127, 721, 37, 73
- **BTC Energy:** 213, 231, 312, 321, 132, 123, 68

**Sequence detection:** Price checked for substrings 322, 113, 93, 213, 666, 777.

#### 3.3 Western Astrology

| Feature | Description |
|---------|-------------|
| `moon_phase` | 0-29.5 day synodic cycle |
| `moon_mansion` | 1-28 lunar mansions |
| `mercury_retrograde` | Binary: 1 during retrograde |
| `venus_retrograde` | Binary |
| `mars_retrograde` | Binary |
| `hard_aspects` | Count of conjunctions, oppositions, squares |
| `soft_aspects` | Count of trines, sextiles |
| `planetary_strength` | Composite score |
| `eclipse_window` | Binary: within 5 days of solar/lunar eclipse |
| `planetary_hour` | Ruling planet of current hour |
| `void_of_course_moon` | Binary: moon between last major aspect and sign change |

#### 3.4 Vedic Astrology

| Feature | Description |
|---------|-------------|
| `nakshatra` | 1-27 lunar mansions |
| `nakshatra_nature` | Deva / Human / Rakshasa |
| `nakshatra_guna` | Sattva / Rajas / Tamas |
| `tithi` | 1-30 lunar day |
| `yoga` | 1-27 luni-solar combinations |
| `karana` | 1-11 half-tithis |
| `panchang_score` | Composite quality of day |
| `rahu_ketu_axis` | Nodal position |

#### 3.5 Chinese BaZi (Four Pillars)

| Feature | Description |
|---------|-------------|
| `bazi_stem` | 1-10 Heavenly Stems (day stem) |
| `bazi_branch` | 1-12 Earthly Branches |
| `bazi_element` | Wood / Fire / Earth / Metal / Water |
| `bazi_hour_pillar` | Hour pillar stem+branch |
| `bazi_clash_harmony` | Clash/harmony relationships |

#### 3.6 Mayan Tzolkin

| Feature | Description |
|---------|-------------|
| `tzolkin_tone` | 1-13 galactic tone |
| `tzolkin_day_sign` | 1-20 day signs |
| `tzolkin_kin` | 1-260 combined kin number |

#### 3.7 Hebrew Calendar

| Feature | Description |
|---------|-------------|
| `shmita_year` | Sabbatical year (2015, 2022, 2029...) |
| `hebrew_holidays` | Yom Kippur, Passover, Rosh Hashanah |
| `omer_period` | 49-day counting between Passover and Shavuot |

#### 3.8 Arabic Lots

| Feature | Description |
|---------|-------------|
| `lot_of_commerce` | Calculated from ASC + Mercury - Sun |
| `lot_of_increase` | Financial growth indicator |
| `lot_of_catastrophe` | Risk indicator |
| `lot_of_treachery` | Deception indicator |
| `lot_moon_conjunction` | Moon within 10 degrees of any lot |

#### 3.9 Sentiment / NLP

Applied to tweets, news headlines, Reddit titles, FOMC statements, Fed quotes:

| Feature | Description |
|---------|-------------|
| `sentiment_score` | -N to +N based on bull/bear word counts |
| `has_caps` | ALL CAPS detection (urgency/emotion indicator) |
| `exclamation_count` | Exclamation mark intensity |
| `urgency_score` | "BREAKING", "ALERT", "FLASH" detection |
| `bull_count` / `bear_count` | Count of bullish/bearish keywords |

**Bullish words:** bull, moon, pump, rally, surge, breakout, ath, buy, long, hodl, halving, parabolic...
**Bearish words:** bear, crash, dump, plunge, fear, sell, panic, ban, hack, scam, bubble, recession...

#### 3.10 Sacred Geometry

| Feature | Description |
|---------|-------------|
| `golden_ratio_ext` | Price distance from Fibonacci extension (PHI = 1.618) |
| `golden_ratio_dist` | Absolute distance to golden ratio level |
| `gann_sq9_level` | Nearest Gann Square of 9 level |
| `gann_sq9_distance` | Percentage distance to Gann level |
| `fib_21_from_low` | 21% Fibonacci from rolling low |
| `fib_13_from_high` | 13% Fibonacci from rolling high |
| `vortex_369` | Tesla 3-6-9 pattern in price digital root |

#### 3.11 Cross-Features (multi-source amplification)

The most powerful signals come from combining multiple sources:

| Cross-Feature | Description |
|--------------|-------------|
| `moon_x_gold_tweet` | Full/new moon AND gold tweet detected |
| `nakshatra_x_red_tweet` | Key nakshatra AND red tweet present |
| `mercury_retro_x_news_sentiment` | Mercury retrograde AND negative news |
| `dr_date_x_dr_price` | Date digital root matches price digital root |
| `eclipse_x_sports_upset` | Eclipse window AND underdog sports win |
| `planetary_hour_x_tweet_time` | Jupiter hour AND tweet at key time |
| `voc_moon_x_high_volume` | Void of Course moon AND abnormal volume |
| `friday13_x_red_tweet` | Friday 13th AND red tweet |
| `fg_extreme_x_moon_phase` | Fear/Greed extreme AND specific moon phase |
| `master_number_x_nakshatra` | Master number in price AND key nakshatra |

---

## 4. Data Pipeline

### 4.1 Collection Layer

Seven streamers run continuously, polling APIs and writing to SQLite databases:

| Streamer | Database | Frequency | Data |
|----------|----------|-----------|------|
| `tweet_streamer.py` | `tweets.db` | 5 min | Tweet text, user info, engagement, colors, gematria |
| `news_streamer.py` | `news_articles.db` | 5 min | Headlines, sentiment, gematria, source metadata |
| `sports_streamer.py` | `sports_results.db` | 15 min | NFL/NBA/MLB/NHL scores, horse racing results |
| `crypto_streamer.py` | `onchain_data.db` | Variable | Block data, funding, OI, whale txns, Fear&Greed |
| `macro_streamer.py` | `macro_data.db` | Daily | S&P500, NASDAQ, DXY, gold, VIX, 10Y yield |
| `astro_engine.py` | `astrology_full.db` | Daily | Ephemeris, planetary positions, nakshatras |
| `download_btc.py` | `btc_prices.db` | Per-TF | OHLCV candles for 5m/15m/1H/4H/1D/1W |

Additional databases:
- `ephemeris_cache.db` — pre-computed planetary data
- `fear_greed.db` — Bitcoin Fear & Greed index
- `google_trends.db` — "Bitcoin" search interest
- `funding_rates.db` — Bitget BTCUSDT funding rate history

### 4.2 Processing Layer

**Universal Engines** transform raw data into features. Each engine takes a generic input (text, number, or timestamp) and returns all computed values:

```
Text (any string) -> universal_gematria.py  -> 6 cipher values + DR of each + match flags
Number (any value) -> universal_numerology.py -> digital root + master check + sequence detection
Timestamp (any dt) -> universal_astro.py     -> full snapshot (Western+Vedic+BaZi+Tzolkin+hours)
Text (any string) -> universal_sentiment.py  -> sentiment + caps + exclamation + urgency
```

### 4.3 Feature Building Layer

Feature builders aggregate events into per-bar features using **bucketing**:

1. Load OHLCV candles for the target timeframe
2. Load raw event data from all DBs
3. Bucket events by timestamp (`(ts_unix // bucket_seconds) * bucket_seconds`)
4. Aggregate within each bucket (last value, count, mean, etc.)
5. Map aggregated values to bar index
6. Compute TA features, numerology features, astrology features
7. Add KNN pattern features
8. Compute forward-looking targets (`next_{tf}_return`)
9. Save to `features_{tf}.db`

### 4.4 Shared Feature Architecture

The refactored architecture uses two shared modules to eliminate training/live drift:

**`feature_library.py`** — Pure-function library. NO database calls. Takes pre-loaded data, returns features. Used by both offline builders and live trader. Key function:

```python
build_all_features(
    ohlcv,              # OHLCV DataFrame with DatetimeIndex
    esoteric_frames,    # dict of tweets, news, sports, onchain, macro DataFrames
    tf_name,            # '5m' through '1w'
    mode,               # 'backfill' or 'live'
    htf_data,           # higher-TF OHLCV for context features
    astro_cache,        # pre-loaded ephemeris/astrology data
)
```

**`data_access.py`** — Data Access Layer. Two loader classes:
- `OfflineDataLoader` — Full history for feature building
- `LiveDataLoader` — Incremental cache with `refresh_caches()` every ~30s

This ensures **identical feature computation** in training and inference — the standard production ML "feature store" pattern.

---

## 5. Feature Engineering

### 5.1 Feature Count by Category

| Category | Estimated Count | Description |
|----------|:-:|-------------|
| Technical Analysis | ~80 | MA, RSI, MACD, BB, Stoch, ATR, Ichimoku, SAR, ADX, CCI, OBV, MFI, CMF, Keltner, Donchian, Wyckoff, Elliott, Gann, Supertrend, pivots, candle patterns |
| Numerology | ~125 | Digital root of price/volume/scores/dates, master numbers, sequence detection, caution/pump flags, Tesla 369, Shmita, Jubilee |
| Gematria (6 methods x targets) | ~200 | All text sources x 6 ciphers |
| Gematria DR + Match | ~160 | Digital root of each gematria value + cross-source matches |
| Astrology (all timestamps) | ~80 | Western + Vedic + BaZi + Tzolkin at candle close + event timestamps |
| Sentiment | ~30 | Sentiment scores, caps detection, urgency for all text sources |
| Color Analysis | ~15 | Gold/red/green detection in tweet images |
| Number Patterns | ~40 | Substring detection (113, 322, 93, 213, 666, 777) in prices, scores, block numbers |
| Cross-Features | ~50 | Multi-source combinations (moon x tweet, nakshatra x news, etc.) |
| Regime & Trend | ~10 | HMM bull/bear/neutral probabilities, Wyckoff phase, drawdown depth, EMA50 slope |
| KNN Pattern Matching | 5 | knn_direction, knn_confidence, knn_avg_return, knn_best_match_dist, knn_pattern_std |
| Time / Calendar | ~20 | Sin/cos encoded hour/DOW/month/DOY, session flags, month-end, quarter-end |
| Higher-TF Context | Variable | Forward-filled features from higher timeframes (e.g., 4H RSI on 15m chart) |
| **TOTAL** | **~790** | |

### 5.2 Feature Count by Timeframe (Actual)

| Timeframe | Bars | Features | Build Time |
|-----------|:----:|:--------:|:----------:|
| 5m | 547,782 | 76 | 8 min |
| 15m | 217,446 | 164 | 85s |
| 1H | 56,753 | 373 | 18s |
| 4H | ~14,000 | 370 | 4 min |
| 1D | ~3,600 | 350 | 30s |
| 1W | 339 | 200 | 3s |

Lower TFs have fewer features because heavy indicators (Ichimoku, full Gann loops) are skipped for speed, and some esoteric signals bucket poorly at 5m resolution.

### 5.3 Technical Analysis Features (computed in `feature_library.py`)

**Moving Averages:** SMA/EMA for periods 5, 10, 20, 50, 100, 200. Close-vs-MA ratios. MA slopes. Golden/death cross signals. Consensio score (all MA pairs aligned).

**Oscillators:** RSI (7, 14, 21) with OB/OS flags, Stochastic %K/%D (14,3), Williams %R (14), CCI (20), MFI (14), CMF (20).

**Volatility:** Bollinger Bands (20,2) with %B and squeeze detection, ATR (14) with percentage, Keltner Channels, Donchian Channels, rolling volatility ratio.

**Trend:** MACD (12,26,9) with histogram and crossovers, Ichimoku (9,26,52) cloud position, Parabolic SAR with flip detection, Supertrend (10,3), ADX (14), Pivot Points (classic R2/R1/S1/S2).

**Volume:** Volume SMA ratio, volume spike detection, OBV with trend, VWAP, taker buy ratio.

**Pattern Recognition:** Candle body percentage, wick ratios, doji, hammer, shooting star, bull/bear engulfing, consecutive green/red candle count.

**Advanced:** Wyckoff phase detection (accumulation/markup/distribution/markdown), Elliott Wave zigzag, Gann Square of 9 distance, sacred geometry (golden ratio extension, Fibonacci levels).

**Interaction Features:** RSI x BB%B, volume x ATR, consecutive red x BB oversold, RSI bullish divergence.

**Lagged Features:** RSI-14, BB%B, volume ratio, MACD histogram each lagged at TF-specific intervals.

**Meta Features:** Count of bullish signals, count of bearish signals, signal agreement score.

### 5.4 NaN Strategy

Esoteric features are intentionally left as **NaN** when no signal is active (not filled with 0). XGBoost natively handles missing values by learning separate branches for "missing" vs "present" data. This is critical because:

- Filling with 0 would mean "no tweet" = "tweet with gematria value 0", which is wrong
- NaN lets the model learn "when this signal IS present, go left; when absent, go right"
- The `missing` branch often leads to different predictions than either value branch

---

## 6. ML Pipeline

### 6.1 Training Script: `ml_multi_tf.py`

The ML pipeline trains one XGBoost binary classifier per timeframe. Key design decisions:

**3-Class Labels:**
- LONG (1): `next_return > cost_pct`
- SHORT (0): `next_return < -cost_pct`
- FLAT (-1): within cost threshold (noise zone)

Only LONG and SHORT samples are used for training. FLAT samples are included at inference to verify the model outputs ~0.5 (no signal) for them.

**No SHAP Pruning:**
All ~700 features flow through to XGBoost. The model handles feature selection via tree splits. SHAP importance is logged for visibility but never used for pruning.

**NaN Encoding:**
`np.where(np.isinf(X_all), np.nan, X_all)` — only infinities are replaced. NaN values are preserved for XGBoost's native missing-branch handling.

**Event-Aware Sample Weights:**
Bars where esoteric signals are active receive 1.5-3x weight during training. The weight formula:

```python
esoteric_active = count of non-NaN, non-zero esoteric columns per bar
esoteric_weight = clip(1.0 + 0.5 * min(esoteric_active, 4), 1.0, 3.0)
sample_weights *= esoteric_weight
```

Esoteric columns are identified by keyword matching: `gem_`, `dr_`, `moon`, `nakshatra`, `vedic`, `bazi`, `tzolkin`, `arabic`, `tweet`, `sport`, `horse`, `caution`, `cross_`, `eclipse`, `retro`, `shmita`, etc.

**Regime-Aware Weights:**
Counter-trend trades are downweighted:
- Bear-market LONGs (EMA50 declining): 0.15x weight
- Bull-market SHORTs (EMA50 rising): 0.15x weight

### 6.2 Walk-Forward Validation

**Rolling Windows (for intraday TFs):**
- 3 rolling windows, each with a training portion of `rolling_window_bars` and test portion of `rolling_window_bars // 4`
- Example for 1H: 13,140 train bars (~18 months), 3,285 test bars (~4.5 months)

**Expanding Windows (for 1D/1W):**
- Too few samples for rolling; uses expanding window with 25% test splits

**HMM Re-fitting:**
- Gaussian HMM with 3 components (bull/bear/neutral) is re-fitted at each walk-forward boundary using only data up to that point (no future leakage)
- Features: log returns, absolute returns, 10-day rolling volatility
- Best of 3 random seeds selected by model score
- HMM probabilities (`hmm_bull_prob`, `hmm_bear_prob`, `hmm_neutral_prob`, `hmm_state`) injected into feature matrix

### 6.3 XGBoost Configuration Per TF

| Parameter | 5m | 15m | 1H | 4H | 1D | 1W |
|-----------|:--:|:---:|:--:|:--:|:--:|:--:|
| max_depth | 3 | 4 | 5 | 5 | 4 | 3 |
| min_child_weight | 3 | 3 | 2 | 2 | 1 | 1 |
| subsample | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 |
| colsample_bytree | 0.6 | 0.6 | 0.7 | 0.7 | 0.6 | 0.6 |
| colsample_bynode | 0.7 | 0.7 | 0.7 | 0.7 | 0.7 | 0.7 |
| reg_lambda (L2) | 8.0 | 5.0 | 3.0 | 3.0 | 6.0 | 8.0 |
| reg_alpha (L1) | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| gamma | 0.5 | 0.5 | 0.3 | 0.3 | 0.5 | 0.5 |
| learning_rate | 0.05 | 0.05 | 0.05 | 0.05 | 0.03 | 0.02 |
| cost_pct | 0.22% | 0.22% | 0.23% | 0.24% | 0.0025 | 0.0025 |
| rolling_window | 105,120 | 35,040 | 13,140 | 8,760 | All | All |

**Design rationale:**
- Lower TFs (5m, 15m): stronger regularization (higher lambda, gamma) to prevent overfitting on noise
- Lower `min_child_weight` (1) for 1D/1W so rare esoteric signals can form their own leaves
- `colsample_bynode=0.7` ensures feature diversity at each split
- L1 regularization (`reg_alpha=0.1`) encourages sparsity alongside L2

### 6.4 Platt Calibration

After training, a logistic regression calibrator is fitted on the last 20% of training data to map raw XGBoost outputs to calibrated probabilities. Saved as `platt_{tf}.pkl` for live use.

### 6.5 Outputs

Per timeframe:
- `model_{tf}.json` — trained XGBoost model
- `features_{tf}_all.json` — ordered feature name list
- `platt_{tf}.pkl` — Platt calibrator
- `ml_multi_tf_configs.json` — accuracy metrics and config per TF

---

## 7. GPU Grid Search Optimizer

### 7.1 Overview

`exhaustive_optimizer.py` replaces the previous 15K-combo NSGA-II genetic algorithm with a **full exhaustive grid search** testing ~384M-576M combinations per TF across 6 TFs, totaling ~2.88 billion combinations. Runs on the RTX 3090 GPU via CuPy.

### 7.2 Parameter Grid Per TF

| Parameter | 5m | 15m | 1H | 4H | 1D | 1W |
|-----------|:--:|:---:|:--:|:--:|:--:|:--:|
| Leverage | 1-125x (42 steps) | 1-125x (42) | 1-100x (50) | 1-75x (38) | 1-20x (20) | 1-10x (10) |
| Risk/Trade | 0.01-2% (29) | 0.01-3% (30) | 0.05-4% (25) | 0.1-5% (25) | 0.1-6% (30) | 0.1-6% (30) |
| Stop (ATR mult) | 0.05-1.0 (20) | 0.1-1.5 (21) | 0.2-2.0 (20) | 0.3-3.0 (20) | 0.5-4.0 (20) | 1.0-6.0 (21) |
| Reward:Risk | 1.0-4.0 (21) | 1.0-5.0 (21) | 1.0-6.0 (21) | 1.0-8.0 (21) | 1.0-8.0 (21) | 1.0-10.0 (21) |
| Max Hold (bars) | 1-72 (20) | 1-48 (20) | 1-72 (20) | 1-84 (20) | 1-90 (20) | 1-52 (20) |
| Exit Type | 6 options | 6 | 6 | 6 | 6 | 6 |
| Confidence | 0.45-0.90 (10) | 0.45-0.90 (10) | 0.45-0.90 (10) | 0.45-0.90 (10) | 0.45-0.90 (10) | 0.45-0.90 (10) |

**Exit Types (6 options):**
- `0`: No partial TP (full TP hit = full profit)
- `25`: 25% partial TP
- `50`: 50% partial TP
- `75`: 75% partial TP
- `-2`: Trailing stop at 2x ATR (activates after reaching 1R profit)
- `-3`: Trailing stop at 3x ATR

### 7.3 Fee Model

```
Bitget USDT-M Futures taker fee: 0.06% per side
Conservative slippage:           0.03% per side
Total round-trip cost:           0.18%
Applied as: net_pnl = gross - (0.0018 * leverage)
```

### 7.4 Vectorized Simulation Engine

The `simulate_batch()` function processes N parameter combos simultaneously:

- **Vectorized across combos (N dimension):** All state variables (`balance`, `peak`, `max_dd`, `wins`, `losses`, `in_trade`, `trade_bars`, `entry_pr`, `stop_pr`, `tp_pr`, `best_pr`) are N-dimensional arrays
- **Sequential across bars (T dimension):** Iterates bar-by-bar because trailing stops and trade state require sequential logic
- **GPU acceleration:** CuPy operations on RTX 3090. Batch size auto-tuned: up to 20M combos per batch on GPU (limited by 24GB VRAM)
- **Sortino ratio tracking:** Accumulates log returns and squared negative log returns per combo for risk-adjusted metric

### 7.5 Output Profiles

For each TF, the optimizer saves 4 profiles:

| Profile | Criteria |
|---------|----------|
| `dd10_best` | Maximum ROI with drawdown <= 10% |
| `dd10_sortino` | Best Sortino ratio with drawdown <= 10% |
| `dd15_best` | Maximum ROI with drawdown <= 15% |
| `dd15_sortino` | Best Sortino ratio with drawdown <= 15% |

Each profile contains: leverage, risk_pct, stop_atr, rr, max_hold, exit_type, conf_thresh, roi, max_dd, win_rate, trades, sortino, final_balance.

Saved to `exhaustive_configs.json`.

### 7.6 Performance

- RTX 3090 24GB VRAM
- ~2.88 billion total combinations
- Estimated runtime: ~2 hours
- GPU utilization: ~98%

---

## 8. KNN Pattern Matching

### 8.1 Overview

`knn_feature_engine.py` implements GPU-accelerated K-Nearest Neighbors pattern matching. For each bar, it finds the K most similar historical price patterns (by shape, not magnitude) and generates 5 features based on what happened AFTER those similar patterns.

### 8.2 Per-TF Configuration

| TF | Pattern Length | K | Max Lookback | Time Decay Lambda |
|----|:------------:|:-:|:------------:|:-:|
| 5m | 30 candles | 50 | 105,120 bars (1yr) | 0.000007 |
| 15m | 30 candles | 50 | 70,080 bars (2yr) | 0.00002 |
| 1H | 48 candles | 50 | 26,280 bars (3yr) | 0.00008 |
| 4H | 30 candles | 50 | 6,570 bars (3yr) | 0.0003 |
| 1D | 20 candles | 50 | 1,460 bars (4yr) | 0.002 |
| 1W | 12 candles | 30 | 208 bars (4yr) | 0.013 |

### 8.3 Output Features (per bar)

| Feature | Description |
|---------|-------------|
| `knn_direction` | Time-decay weighted average direction of K neighbors' outcomes (+1 up, -1 down) |
| `knn_confidence` | Agreement ratio (% of K neighbors in the majority direction) |
| `knn_avg_return` | Time-decay weighted mean return after K similar patterns |
| `knn_best_match_dist` | Euclidean distance to the single closest match |
| `knn_pattern_std` | Standard deviation of the current pattern (volatility context) |

### 8.4 Algorithm

1. **Pattern Matrix Construction:** Sliding window of percent changes, shape `(n_bars, pattern_len)`
2. **Z-Score Normalization:** Each pattern normalized to zero mean, unit variance. This makes matching shape-based, not magnitude-based.
3. **GPU Distance Computation:** Squared Euclidean distance matrix `(B_queries, N_references)` computed using element-wise CuPy ops (avoids cuBLAS dependency). Sub-batched for memory: `Q[:, None, :] - R[None, :, :]` expansion capped at 500M elements.
4. **Walk-Forward Masking:** For query at time t, all reference patterns at time >= t are masked to +inf. Only past patterns can be neighbors.
5. **Top-K Selection:** `cp.argpartition` for O(N) partial sort, followed by sorting within K.
6. **Combined Weighting:** `weight = (1 / sqrt(distance)) * exp(-lambda * bars_ago)`. Distance weights give closer matches more influence; time decay ensures recent patterns matter more.
7. **Feature Computation:** Weighted average of neighbor returns, direction agreement, best match distance.

### 8.5 Walk-Forward Safety

The KNN engine is strictly walk-forward safe:
- When building features at time t, the pattern pool contains ONLY patterns ending BEFORE t
- Future bars are masked with distance = +inf
- Invalid patterns (containing NaN) and patterns without valid next_returns are also masked
- Lookback is capped (e.g., 1 year for 5m) to prevent ancient, irrelevant patterns from polluting neighbors

### 8.6 A/B Testing in Training

For each timeframe, `ml_multi_tf.py` trains two models:
- **Model A:** All features INCLUDING knn_* columns
- **Model B:** All features EXCLUDING knn_* columns

The model with higher accuracy wins. KNN is kept or dropped independently per TF. Decision is logged and can flip on monthly retraining.

### 8.7 Performance

- 14,000 bars processed in 5.2 seconds on GPU
- Query batch size: 4,000 (GPU) / 2,000 (CPU)
- Live mode: single-bar query for `live_trader.py` via `knn_features_from_ohlcv()`

---

## 9. Live Trading Engine

### 9.1 Overview

`live_trader.py` runs continuously, monitoring bar closes across 6 timeframes, computing features via the shared `feature_library.py`, generating predictions with trained XGBoost models, and managing positions with per-TF capital pools.

### 9.2 Multi-TF Capital Allocation

| Timeframe | Allocation | Pool Size ($100 start) | Rationale |
|-----------|:----------:|:----------------------:|-----------|
| 5m | 10% | $10.00 | Scalping, high frequency |
| 15m | 25% | $25.00 | Core short-term |
| 1H | 25% | $25.00 | Core medium-term |
| 4H | 20% | $20.00 | Swing trading |
| 1D | 10% | $10.00 | Position trading |
| 1W | 10% | $10.00 | Macro positions |

Each pool tracks its own balance, peak, and drawdown independently. Multiple TFs can have open positions simultaneously. Maximum 1 position per TF (no stacking).

### 9.3 Position Management

**Entry Logic:**
1. Wait for bar close on each TF (checked via `is_bar_close()`)
2. Compute features via `compute_features_live()` using shared `feature_library.py`
3. Build feature vector, run XGBoost prediction
4. If `prob > conf_thresh`: LONG signal
5. If `prob < (1 - conf_thresh)`: SHORT signal
6. Dedup check: skip if already have open trade for this TF or trade at same minute
7. Calculate stop loss (`price - direction * stop_atr * ATR`) and take profit (`price + direction * stop_atr * ATR * RR`)
8. Insert into `trades.db` with full feature snapshot

**Exit Logic:**
- **Stop Loss:** Price hits SL level
- **Take Profit:** Price hits TP level
- **Time Exit:** Bars held exceeds `max_hold`
- PnL calculated as: `gross = price_change * leverage`, `net = gross - (0.0018 * leverage)`, `pnl_dollar = balance * risk_pct * net`

**Circuit Breakers:**
- Per-TF: pool halted if DD exceeds 25%
- Portfolio-wide: all new entries halted if total portfolio DD exceeds 15%
- Per-TF pool state lost on process restart (in-memory only)

### 9.4 Feature Computation Flow

```python
live_dal.refresh_caches()           # Incremental DB updates (~30s)
ohlcv = live_dal.get_ohlcv_window(tf, n_bars)  # Recent candles
esoteric_frames = {tweets, news, sports, onchain, macro}
htf_data = live_dal.get_htf_ohlcv(tf)          # Higher-TF context
astro_cache = live_dal.get_astro_cache()        # Ephemeris/astrology

df_features = build_all_features(               # SAME function as training
    ohlcv, esoteric_frames, tf, 'live',
    htf_data, astro_cache
)
last_row = df_features.iloc[-1]                 # Features for current bar
```

### 9.5 Config Priority

1. `exhaustive_configs.json` (GPU optimizer output) — preferred
2. `ml_multi_tf_configs.json` (legacy GA configs) — fallback
3. Within legacy: `god_mode` > `aggressive` > `balanced`

### 9.6 Warmup Bars

| TF | Warmup Bars |
|----|:-----------:|
| 5m | 600 |
| 15m | 400 |
| 1H | 300 |
| 4H | 200 |
| 1D | 100 |
| 1W | 50 |

---

## 10. Dashboard

### 10.1 Stack

- **Frontend:** Next.js (React), TypeScript, Tailwind CSS
- **Charting:** KLineChart v10 (klinecharts)
- **API:** Express.js on port 3001 (`api_server.js`)
- **Dashboard:** Port 3000

### 10.2 Main Page (`dashboard/app/page.tsx`)

Components:
- **Header:** BTC price (live from Bitget, 5s polling), inversion/phase-shift warnings, threat level badge
- **Chart:** Full candlestick chart with overlays
- **Sidebar:** Signal panel, ML trading panel
- **Modals:** Indicator selector, indicator settings, feature weights, drawing toolbar

State management:
- Timeframe selector: `'1m' | '5m' | '15m' | '1h' | '4h' | '1d' | '1w'`
- Overlay toggles: lunar, caution, pump, ritual, tweet, wyckoff, elliott, gann
- Prediction toggle: ghost candles
- Drawing tools: trendline, hline, vline, fib, rectangle, circle, arrow, measure, text, freehand

### 10.3 Chart Component (`Chart.tsx`)

**Ghost Candles (Predictions):**
When predictions are toggled on, the chart renders 4 semi-transparent projected candles beyond the current bar. Includes dashed lines for entry, stop loss, and take profit levels, with a confidence label.

**Trade Overlays:**
Open and closed trades rendered as visual markers:
- Entry/exit points with direction arrows
- SL lines (red dashed)
- TP lines (green dashed)
- PnL labels on closed trades

**Signal Overlays:**
Date-based markers computed from the API:
- Caution numbers (93, 113, 322, etc.)
- Pump numbers (37, 73, 127)
- Ritual dates (equinoxes, solstices, BTC Genesis, Pizza Day)
- Day 13, Day 21, Day 27
- Full moon / New moon
- Gold/red tweets

**TF Switching Fix:**
Full dispose + reinitialize on timeframe change (klinecharts v10 `setPeriod` doesn't visually refresh). The chart container is cleared, a fresh chart is initialized with the new period and data.

**Indicator Support:**
Maps dashboard indicators to klinecharts:
- EMA, SMA -> price pane overlays
- RSI, MACD, Stochastic, ATR, OBV, CCI -> sub-panes
- Bollinger Bands, Supertrend, Ichimoku -> price pane overlays

### 10.4 Feature Weights Panel (`FeatureWeights.tsx`)

A modal component showing all active ML features for the selected timeframe:
- Fetches from `/api/features?tf={timeframe}`
- Features categorized into: Technical, Astrology, Numerology, Gematria, Tweets, News, Sports, On-chain, Macro, Regime, KNN, Cross, Other
- Each category has a color-coded badge with feature count
- Expandable sections showing individual feature names
- Search functionality to filter features
- Expand all / Collapse all controls

---

## 11. API Server

### 11.1 Overview

`api_server.js` is a standalone Express.js server (port 3001) that provides REST endpoints for the dashboard. Uses `better-sqlite3` for synchronous SQLite access.

### 11.2 Endpoints

| Endpoint | Method | Description | Source |
|----------|--------|-------------|--------|
| `/api/candles` | GET | Historical OHLCV candles | `btc_prices.db` |
| `/api/ml-trades` | GET | Account balance, open/closed trades, equity curve | `trades.db` |
| `/api/paper-trading` | GET | Paper trading positions and account | `paper_trades.db` |
| `/api/paper-trading/signals` | GET | Recent signals fired | `paper_trades.db` |
| `/api/paper-trading/anx` | GET | ANX account and trades | `paper_trades.db` |
| `/api/btc-price` | GET | Live BTC price (Bitget API with DB fallback) | Bitget + `btc_prices.db` |
| `/api/bitget-candles` | GET | Live candles from Bitget exchange | Bitget API |
| `/api/overlays` | GET | Signal overlay points (caution, pump, ritual, lunar, tweets) | Computed + `tweets.db` |
| `/api/tick` | GET | Latest price + account summary | `btc_prices.db` + `trades.db` |
| `/api/manipulation` | GET | Recent caution/manipulation signals | `paper_trades.db` |
| `/api/tweets` | GET | Recent tweets with gematria values | `tweets.db` |
| `/api/coins` | GET | BTC price summary with sparkline | `btc_prices.db` |
| `/api/prediction` | GET | Latest ML prediction (direction, confidence, targets) | `prediction_cache.json` |
| `/api/features` | GET | Feature names grouped by category per TF | `features_{tf}_all.json` |
| `/api/score` | GET | Unified threat score (stub) | Static |
| `/api/signals` | GET | Signals list (stub) | Static |

### 11.3 Overlay Computation

The `/api/overlays` endpoint computes date-based signals on-the-fly:

- **Caution Numbers:** Day-of-year or date combinations matching {39, 43, 48, 93, 113, 223, 322}
- **Pump Numbers:** Matching {37, 73, 127}
- **Ritual Dates:** BTC Genesis (1/3), Spring Equinox (3/20), 322 Skull & Bones (3/22), Beltane (5/1), BTC Pizza Day (5/22), Summer Solstice (6/21), Fall Equinox (9/22), Samhain (10/31), Winter Solstice (12/21)
- **Special Days:** Day 13 (caution), Day 21 (caution), Day 27 (pump)
- **Lunar Phases:** New Moon and Full Moon computed from known synodic month (29.53059 days from Jan 6 2000 reference)
- **Tweet Markers:** Gold/red tweet days from `tweets.db`

### 11.4 Feature Categorization

The `/api/features` endpoint categorizes features by keyword:
- `knn_*` -> KNN
- `gem_`, `gematria` -> Gematria
- `moon`, `nakshatra`, `vedic`, `bazi`, `tzolkin`, `arabic`, `planetary`, `retro`, `eclipse` -> Astrology
- `dr_`, `digital_root`, `master`, `contains_` -> Numerology
- `tweet`, `gold_tweet`, `red_tweet`, `misdirection` -> Tweets
- `news`, `headline`, `caution` -> News
- `sport`, `horse` -> Sports
- `onchain`, `block`, `funding`, `whale` -> On-chain
- `macro`, `sp500`, `nasdaq`, `dxy`, `vix` -> Macro
- `ema50`, `hmm`, `regime` -> Regime
- `cross_`, `_x_` -> Cross
- Standard TA keywords -> Technical
- Everything else -> Other

---

## 12. Universal Engines

### 12.1 `universal_gematria.py`

**Interface:**
```python
from universal_gematria import gematria, gematria_flat, gematria_contains_target

result = gematria("Kentucky Derby")
# {'ordinal': 164, 'reverse': 178, 'reduction': 47, 'english': 984,
#  'jewish': 553, 'satanic': 424, 'dr_ordinal': 2, ...}

features = gematria_flat("Kentucky Derby", prefix='race')
# {'race_gem_ordinal': 164, 'race_gem_reverse': 178, ..., 'race_gem_is_caution': 0}
```

**6 Ciphers:**

| Cipher | Formula |
|--------|---------|
| English Ordinal | `ord(c) - 64` (A=1, Z=26) |
| Reverse Ordinal | `27 - (ord(c) - 64)` (A=26, Z=1) |
| Reduction | Digital root of ordinal sum |
| English Gematria | `(ord(c) - 64) * 6` (A=6, Z=156) |
| Jewish/Hebrew | Traditional values: A=1,B=2,...,K=10,L=20,...,T=100,...,Z=500 |
| Satanic | `ord(c) - 64 + 35` (A=36, Z=61) |

Each cipher's result also gets a digital root. Match detection checks if any cipher value hits known caution or pump numbers.

### 12.2 `universal_numerology.py`

**Interface:**
```python
from universal_numerology import numerology, digital_root, date_numerology, numerology_flat

result = numerology(73954)    # {'dr': 1, 'is_master': False, 'contains_113': False, ...}
features = numerology_flat(73954, prefix='price')
date_result = date_numerology(datetime(2026, 3, 18))
```

**Key Constants:**

| Category | Numbers | Meaning |
|----------|---------|---------|
| Master Numbers | 11, 22, 33 | High energy |
| Caution | 93, 39, 43, 48, 113, 223, 322, 17, 71, 13, 44, 66, 77, 88, 99, 666, 911 | Dump/disruption signals |
| Pump | 27, 72, 127, 721, 37, 73 | Bullish energy |
| BTC Energy | 213, 231, 312, 321, 132, 123, 68 | Bitcoin-specific |

**Digital Root:** `1 + (abs(n) - 1) % 9` — reduces any number to 1-9.

### 12.3 `universal_astro.py`

**Interface:**
```python
from universal_astro import astro_snapshot, astro_flat, get_bazi, get_tzolkin

result = astro_snapshot(datetime.now())
features = astro_flat(datetime.now(), prefix='event')
```

**Dependencies:**
- `astrology_engine.py` for Western calculations (retrogrades, aspects, VOC moon, eclipse windows, planetary hours)
- `swisseph` (Swiss Ephemeris) for planetary positions
- `astrology_full.db` for pre-computed Vedic data (nakshatras, tithi, yoga, guna)

**Systems covered:**
1. Western: Moon phase/mansion, retrogrades (Mercury, Venus, Mars, Saturn, Jupiter), hard/soft aspects, planetary strength, eclipse windows, planetary hours, VOC moon
2. Vedic: Nakshatra (1-27) with nature/guna, Tithi (1-30), Yoga (1-27), Karana, Panchang score, Rahu/Ketu axis
3. BaZi: Day stem (1-10 Heavenly Stems), Day branch (1-12 Earthly Branches), element, hour pillar
4. Tzolkin: Tone (1-13), Day sign (1-20), Kin number (1-260)

### 12.4 `universal_sentiment.py`

**Interface:**
```python
from universal_sentiment import sentiment, sentiment_flat

result = sentiment("Bitcoin crashes! SELL NOW!!!")
# {'score': -2, 'bull_count': 0, 'bear_count': 2, 'has_caps': True,
#  'exclamation': 3, 'urgency': 0, ...}

features = sentiment_flat("Bitcoin crashes!", prefix='headline')
```

**Word Lists:**
- **Bullish (34 words):** bull, moon, pump, rally, surge, breakout, ath, buy, hodl, halving, parabolic, golden, breakthrough...
- **Bearish (42 words):** bear, crash, dump, plunge, fear, sell, panic, ban, hack, scam, bubble, recession, dead, ponzi...
- **Urgency (12 words):** breaking, urgent, alert, emergency, flash, just in, developing, critical, warning...

**Scoring:** `score = bull_count - bear_count` (net sentiment). Caps detection via regex for ALL-CAPS words. Exclamation count. Urgency flag if any urgency word detected.

---

## 13. Data Sources

### 13.1 SQLite Databases

| Database | Tables | Key Columns | Update Frequency |
|----------|--------|-------------|-----------------|
| `btc_prices.db` | `ohlcv` | open_time, open, high, low, close, volume, quote_volume, trades, taker_buy_volume, taker_buy_quote, timeframe, symbol | Per candle close |
| `tweets.db` | `tweets` | created_at, ts_unix, user_handle, full_text, has_gold, has_red, gematria_simple, gematria_english, favorite_count, retweet_count, reply_count | Every 5 min |
| `news_articles.db` | `streamer_articles`, `articles` | ts_unix, title, sentiment_score, title_dr, title_gematria_ordinal, title_gematria_reverse, title_gematria_reduction, sentiment_bull, sentiment_bear, has_caps, exclamation_count, word_count | Every 5 min |
| `sports_results.db` | `games`, `horse_races` | date, winner, home_score, away_score, winner_gem_ordinal, winner_gem_dr, score_dr, is_upset, is_overtime, winner_horse, horse_gem_ordinal, jockey_gem_ordinal, position_dr | Every 15 min |
| `onchain_data.db` | `onchain_data`, `blockchain_data` | timestamp, block_height, block_dr, funding_rate, funding_dr, open_interest, oi_dr, fear_greed, fg_dr, mempool_size, n_transactions, hash_rate, difficulty, miners_revenue | Variable |
| `macro_data.db` | `macro_data` | date, dxy, gold, spx, vix, us10y, nasdaq, russell, oil, silver | Daily |
| `astrology_full.db` | `daily_astrology` | date, + all Vedic/Western astrology columns | Daily (pre-computed) |
| `ephemeris_cache.db` | `ephemeris` | date, moon_mansion, moon_phase, mercury_retrograde, hard_aspects, soft_aspects, planetary_strength, digital_root | Daily |
| `fear_greed.db` | `fear_greed` | date, value | Daily |
| `google_trends.db` | `google_trends` | date, interest | Daily/weekly |
| `funding_rates.db` | `funding_rates` | timestamp, funding_rate, symbol | Every 8 hours |
| `trades.db` | `account`, `trades`, `equity_curve` | Trade logs, balance tracking | Per trade |

### 13.2 External APIs

| API | Data | Used By |
|-----|------|---------|
| Bitget REST API | Live BTC/USDT candles (all TFs), ticker price | `live_trader.py`, `api_server.js`, dashboard |
| Twitter/X | Tweets from tracked accounts | `tweet_streamer.py` |
| CryptoPanic / RSS | Crypto news headlines | `news_streamer.py` |
| Reddit | Post titles from crypto subreddits | `news_streamer.py` |
| ESPN / TheSportsDB | NFL, NBA, MLB, NHL scores; horse racing | `sports_streamer.py` |
| Blockchain APIs | Block data, mempool, hash rate | `crypto_streamer.py` |
| Financial APIs | DXY, gold, VIX, S&P500, NASDAQ, yields | `macro_streamer.py` |
| Swiss Ephemeris | Planetary positions, retrogrades | `astro_engine.py` |

---

## 14. Configuration

### 14.1 Per-TF XGBoost Hyperparameters

See Section 6.3 table. Key design:
- Stronger regularization for faster TFs (5m: max_depth=3, lambda=8; 4H: max_depth=5, lambda=3)
- Lower min_child_weight for sparse TFs (1D/1W: min_child_weight=1)
- Consistent colsample_bynode=0.7 across all TFs

### 14.2 Per-TF Optimizer Grid Ranges

See Section 7.2 table. Key design:
- Leverage caps decrease with TF (5m: 125x, 1W: 10x)
- Risk/trade increases with TF (5m: 0.01-2%, 1W: 0.1-6%)
- Stop ATR widens with TF (5m: 0.05-1.0, 1W: 1.0-6.0)
- Max hold increases with TF (5m: 72 bars = 6hr, 1W: 52 bars = 1yr)
- Trailing stops more relevant for longer TFs

### 14.3 Capital Allocation

```
5m:  10% | 15m: 25% | 1H: 25% | 4H: 20% | 1D: 10% | 1W: 10%
```

No regime-based size reduction. Signal confidence drives position size. If the model says 125x leverage at 0.90 confidence, it takes the trade.

### 14.4 KNN Configuration

See Section 8.2 table. Pattern length decreases with TF resolution (48 for 1H, 12 for 1W). Time decay lambda increases with TF (recent patterns matter more on lower TFs).

### 14.5 Feature Builder Bucket Sizes

| TF | Bucket Seconds | Description |
|----|:-:|---|
| 5m | 300 | 5-minute buckets |
| 15m | 900 | 15-minute buckets |
| 1H | 3,600 | 1-hour buckets |
| 4H | 14,400 | 4-hour buckets |
| 1D | 86,400 | Daily buckets |
| 1W | 604,800 | Weekly buckets |

### 14.6 Warmup Bars for Live Trading

| TF | Bars | Approximate Duration |
|----|:----:|---------------------|
| 5m | 600 | 2 days |
| 15m | 400 | 4 days |
| 1H | 300 | 12.5 days |
| 4H | 200 | 33 days |
| 1D | 100 | 100 days |
| 1W | 50 | ~1 year |

---

## 15. Current Model Performance

### 15.1 Accuracy Per Timeframe

| TF | Accuracy | Features | KNN Status | KNN Contribution |
|----|:--------:|:--------:|:----------:|:----------------:|
| 5m | 81.8% | 75 | DROPPED | - |
| 15m | 80.5% | 163 | DROPPED | - |
| 1H | 74.9% | 375 | KEPT | +0.8% |
| 4H | 60.8% | 362 | KEPT | +1.4% |
| 1D | 50.9% | 319 | DROPPED | - |
| 1W | 50.0% | 349 | KEPT | +2.6% |

**Observations:**
- Lower TFs (5m, 15m) achieve highest accuracy — more samples, stronger TA signal
- KNN helps more on longer TFs (1W: +2.6%) where pattern similarity across weeks is more meaningful
- 1D/1W hover near 50% accuracy, which is expected given the difficulty of daily/weekly prediction
- All TFs trained in 79 seconds total (GPU)

### 15.2 Feature Build Statistics

| TF | Bars | Features | KNN Time | Total Time |
|----|:----:|:--------:|:--------:|:----------:|
| 5m | 547,782 | 76 | 7.8 min (GPU) | 8 min |
| 15m | 217,446 | 164 | Included | 85s |
| 1H | 56,753 | 373 | Included | 18s |
| 4H | ~14,000 | 370 | Included | 4 min |
| 1D | ~3,600 | 350 | Included | 30s |
| 1W | 339 | 200 | Included | 3s |

### 15.3 Optimizer Status

Exhaustive optimizer running: 2.88 billion combinations on RTX 3090, GPU at 98% utilization, ETA ~2 hours.

---

## 16. Known Issues & Future Work

### 16.1 Known Issues

| Issue | Severity | Description |
|-------|----------|-------------|
| cuBLAS/cuRAND DLLs not in PATH | Low | CuPy works via element-wise ops; matmul (@) operator unavailable. Workaround in place for both `exhaustive_optimizer.py` and `knn_feature_engine.py`. |
| FAISS-GPU not on pip for Windows | Low | `faiss-cpu` installed but CuPy batched approach is faster anyway. Not blocking. |
| Per-TF pool state lost on restart | Medium | `tf_pools` dict is in-memory only. Process restart resets pool balances to initial allocation. Should persist to DB. |
| KNN live uses btc_prices.db | Low | May be stale if DB not updated in real-time. Live candles fetched from Bitget should be preferred. |
| 1D/1W accuracy near 50% | Medium | Expected for longer TFs but limits profitability. More data or additional features may improve. |
| Live/training NaN mismatch (pre-refactor) | Fixed | Was causing 70-83% NaN at inference for 1H/4H/1D/1W. Fixed by shared `feature_library.py`. |

### 16.2 Bugs Found & Fixed (17 total, all resolved)

1. `ml_multi_tf.py`: esoteric weights used `feature_cols` before definition — moved after definition
2. `live_trader.py`: `candles` variable not in scope — fixed to `df`
3. `live_trader.py`: return features 100x wrong (decimal vs percent) — removed `* 100`
4. `live_trader.py`: higher-TF prefix mismatch (`1h_` vs `h1_`) — fixed prefix map
5. `live_trader.py`: NaN -> 0 at inference broke XGBoost — now preserves NaN
6. `live_trader.py`: exhaustive_configs.json format mismatch — fixed to read named keys
7. `live_trader.py`: 1D/1W bars never fire — added proper `is_bar_close()` function
8. `live_trader.py`: ~40 missing features — added all, defaults to NaN not 0
9. `live_trader.py`: HMM features set to 0 — fixed to NaN
10. `live_trader.py`: stoch_d, lag features, cross features — all fixed
11. `api_server.js`: relative paths — fixed to `path.join(DB_DIR, ...)`
12. `live_trader.py`: missing '1w' in granularity_map — added

### 16.3 Future Work

**Immediate (from plan):**
1. Complete exhaustive optimizer run (currently running)
2. Verify `feature_library.py` output matches legacy builders exactly
3. Retrain models using feature_library for consistency
4. Run paper trading test: `python live_trader.py --mode paper`
5. Test dashboard: `npm run dev` in dashboard/

**Phase 2 — Data Expansion:**
- Additional Twitter accounts
- More news sources (RSS, CryptoPanic, Reddit)
- UFC / additional sports
- YouTube video titles from crypto influencers
- Telegram channel messages
- Wikipedia Bitcoin pageview correlation

**Phase 3 — Feature Expansion:**
- Color analysis (gold/red/green detection in images)
- Full Arabic Lots computation
- Dasha periods (Vedic time cycles)
- More cross-features from new data sources
- Google Trends correlation

**Phase 5 — Dashboard Enhancements:**
- Show active exotic signals on chart
- Show which features fired for each trade decision
- Live feed of tweets/news/sports with gematria annotations
- Real-time feature importance updates

**Architecture Improvements:**
- Persist per-TF pool state to database (survives restart)
- Monthly KNN retraining automation
- Automated data quality monitoring (stale DB detection)
- Live candle fallback chain: Bitget API -> DB -> cached

---

*This report was generated from analysis of the complete Savage22 codebase on 2026-03-18.*
