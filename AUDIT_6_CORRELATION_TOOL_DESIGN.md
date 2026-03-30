# Audit Agent 6: Manual Correlation / Time Window Investigation Tool

## Design Document — `investigate.py`

---

## 1. EXECUTIVE SUMMARY

A CLI tool (`investigate.py`) that lets you post-mortem any trade or time period by pulling together every data source in the system into a single correlated view. Three modes: Trade Investigation, Signal Correlation, and Training Verification. Runs locally on the 13900K+3090, reads directly from existing SQLite DBs and parquet files, outputs to terminal with optional HTML report export.

---

## 2. EXISTING DATA INVENTORY

### 2.1 Trade Databases
| DB | Table | Key Columns | Rows |
|---|---|---|---|
| `trades.db` | `trades` | id, tf, direction, confidence, entry_price, entry_time, exit_price, exit_time, stop_price, tp_price, pnl, pnl_pct, bars_held, exit_reason, regime, leverage, risk_pct, **features_json**, status | 0 (ML trader, not yet active) |
| `trades.db` | `equity_curve` | timestamp, balance, dd_pct | 0 |
| `trades.db` | `account` | balance, peak_balance, total_trades, wins, losses, max_dd | 1 |
| `paper_trades.db` | `positions` | signal_name, direction, entry_price, stop_loss, take_profit, leverage, risk_amount, position_size, timeframe, category, status, exit_price, pnl, result, opened_at, closed_at | 3 |
| `paper_trades.db` | `signals_fired` | timestamp, candle_time, timeframe, signal_name, direction, strength, category, action, btc_price | 3 |
| `paper_trades.db` | `trade_log` | timestamp, event, signal_name, direction, details, balance | 6 |

### 2.2 Feature Databases (Pre-Computed)
| DB | Table | Rows | Columns |
|---|---|---|---|
| `features_1h.db` | `features_1h` | 56,753 | 1,990 |
| `features_1h.db` | `features_1h_ext` | 56,753 | 1,127 |
| `features_4h.db` | `features_4h` | 14,194 | 1,990 |
| `features_4h.db` | `features_4h_ext` | 14,194 | 1,416 |
| `features_1d.db` | (daily features) | varies | ~1,990 |
| `features_5m.db` | (5-minute) | varies | ~1,990 |
| `features_15m.db` | (15-minute) | varies | ~1,990 |
| `features_1w.db` | (weekly) | varies | ~1,990 |
| `features_complete.db` | `features` | 2,368 | 1,075 |

### 2.3 Raw Data Sources
| DB | Table | Key Data | Rows |
|---|---|---|---|
| `btc_prices.db` | `ohlcv` | OHLCV candles (all TFs) | large |
| `tweets.db` | `tweets` | Elon/influencer tweets with text, timestamps | 10,981 |
| `news_articles.db` | articles | crypto headlines + sentiment | varies |
| `sports_data.db` | `daily_sports` | NFL/NBA/horse racing gematria | 3,363 |
| `sports_data.db` | `games` | individual game results | 8,604 |
| `sports_data.db` | `horse_racing` | triple crown etc | 27 |
| `space_weather.db` | `space_weather` | Kp, solar wind, Bz | 1 (live) |
| `space_weather.db` | `solar_flares` | flare events | 63 |
| `onchain_data.db` | `blockchain_data` | hash rate, mempool, difficulty | 6,162 |
| `funding_rates.db` | `funding_rates` | perp funding | 9,111 |
| `funding_rates.db` | `open_interest` | OI snapshots | 12,040 |
| `astrology_full.db` | daily astro data | planetary positions | varies |
| `ephemeris_cache.db` | ephemeris | planet coordinates | varies |
| `fear_greed.db` | fear/greed index | daily FGI | varies |
| `google_trends.db` | search trends | "bitcoin" popularity | varies |
| `macro_data.db` | macro indicators | DXY, rates, SPX etc | varies |

### 2.4 Model Artifacts
| File | Purpose |
|---|---|
| `model_{tf}.json` | XGBoost model per timeframe |
| `features_{tf}_all.json` | Feature name list (all features) |
| `features_{tf}_pruned.json` | Feature name list (SHAP-pruned) |
| `meta_model_{tf}.pkl` | Meta-labeling gate model |
| `platt_{tf}.pkl` | Platt calibration for LSTM |
| `cpcv_oos_predictions_{tf}.pkl` | CPCV out-of-sample predictions |
| `ml_multi_tf_configs.json` | GA-optimized trading params |
| `exhaustive_configs.json` | Exhaustive-search params |

---

## 3. MODE A: TRADE INVESTIGATION

### 3.1 CLI Interface
```bash
# By trade ID (from trades.db or paper_trades.db)
python investigate.py trade --id 3

# By trade ID from paper_trades
python investigate.py trade --paper-id 1

# By time range (shows all trades in range)
python investigate.py trade --from "2026-03-17T04:00" --to "2026-03-18T16:00"
```

### 3.2 Output Sections

#### Section 1: Trade Summary
```
===============================================================================
  TRADE INVESTIGATION: #1 — 4H_17dates_LONG
===============================================================================
  Signal:      4H_17dates_LONG (category: numerology, strength: 5)
  Direction:   LONG
  Timeframe:   4H
  Entry:       $73,954.55 @ 2026-03-17T05:51:07
  Exit:        $71,365.06 @ 2026-03-18T15:38:53  (SL hit)
  Stop Loss:   $73,281.27    Take Profit: $75,301.11
  Leverage:    16.7x         Risk Amount: $30.00
  PnL:         $-30.00       Result: LOSS (SL)
  Bars Held:   ~8 bars (4H)
  Price Move:  -3.50% (entry to exit)
```

#### Section 2: Feature Snapshot at Entry
Groups all ~1,990 features into categories at entry candle time:

```
--- TECHNICAL ANALYSIS (Entry Bar) ---
  close:           73,954.55    sma_5:         73,800    sma_20:        74,200
  rsi_14:          42.3         macd_line:     -120      macd_signal:   -85
  bb_upper:        76,100       bb_lower:      71,800    bb_pctb:       0.55
  atr_14:          673.28       volume:        1,234     ema50_slope:   -2.1%
  ichimoku_cloud:  above        sar_direction: short     wyckoff_phase: distribution
  fvg_nearest:     0.012        ob_distance:   0.008

--- NUMEROLOGY ---
  pump_date:       0            caution_date:  1 (17 = caution!)
  price_dr:        6            date_reduction: 8
  btc_energy_date: 0            date_gem_73:   1         date_gem_48:   0
  bars_since_pump: 12

--- GEMATRIA (Tweets) ---
  tweet_gem_pump:  1            tweet_gem_caution: 0
  tweet_count:     3            tweet_sentiment:   0.42
  news_gem_pump:   0            news_gem_caution:  1

--- ASTROLOGY ---
  west_mercury_rx: 0            west_moon_phase:   18 (waning gibbous)
  voc_moon:        0            planetary_hour:    Saturn
  sun_sign:        Pisces       lunar_node:        south

--- SPACE WEATHER ---
  kp_index:        3.7          solar_wind_speed:  420
  solar_wind_bz:   -2.1         sw_kp_roc_3d:      +0.8

--- SPORTS ---
  sport_upset_count: 2          game_count:    4
  horse_race_dr:   N/A

--- ON-CHAIN ---
  hash_rate_roc:   +1.2%        mempool_size:  45,000
  funding_rate:    0.0012       open_interest: $24.5B

--- SENTIMENT ---
  fear_greed:      35 (fear)    google_trends: 62
  headline_sentiment: -0.15

--- REGIME ---
  ema50_rising:    0            ema50_declining: 1
  range_position:  0.45         hmm_state:     1 (bear)
  hmm_bull_prob:   0.15         hmm_bear_prob: 0.65

--- MATRIX CONVERGENCE ---
  matrix_bull_count:    2/8     matrix_bear_count:   4/8
  matrix_net_score:     -3.5    matrix_convergence:  0.62
  matrix_direction:     BEARISH
```

#### Section 3: Model Decision Anatomy
```
--- MODEL PREDICTION ---
  XGBoost 4H:     P(LONG)=0.82  P(FLAT)=0.10  P(SHORT)=0.08
  Predicted:      LONG @ 82% confidence
  Threshold:      0.80 → PASSED

--- META-LABELING GATE ---
  Meta prob:       0.71          Threshold: 0.50 → PASSED (would have taken trade)

--- LSTM BLEND (if available) ---
  LSTM prob:       N/A (no LSTM model for 4H)
  Blended:         0.82 (XGBoost only)

--- KELLY SIZING ---
  Base risk:       3.0%          P(win): 0.82       R:R ratio: 2.0
  Full Kelly:      0.73          Quarter Kelly: 0.18
  Kelly-scaled:    3.5%          DD scale: 1.0 (no drawdown)
  Final risk:      3.0% ($30.00 of $1,000)

--- REGIME ADJUSTMENT ---
  Regime:          bear (idx=1)
  Lev mult:        0.47          Risk mult: 1.0
  Stop mult:       0.75          R:R mult:  0.75
  Hold mult:       0.17
  Adjusted lev:    ~16.7x        Adjusted stop: 0.75 ATR
```

#### Section 4: Price Action During Trade
```
--- PRICE ACTION: Entry → Exit ---
  Time          Price      Change    Event
  ─────────────────────────────────────────────────
  03-17 04:00   $73,955    ──        Entry bar close
  03-17 08:00   $73,400    -0.75%    ▼ first pullback
  03-17 12:00   $73,100    -1.15%    ▼ approaching SL
  03-17 16:00   $72,800    -1.56%    ▼ SL zone
  03-17 20:00   $72,200    -2.37%    ▼ below SL (SL breached)
  03-18 00:00   $71,900    -2.78%    ▼ continued decline
  03-18 04:00   $71,600    -3.18%    ▼
  03-18 08:00   $71,500    -3.32%    trough
  03-18 12:00   $71,365    -3.50%    exit recorded
  03-18 16:00   $71,800    -2.91%    post-exit bounce

  Peak price:    $73,955 (entry)     Time to peak: 0 bars
  Trough price:  $71,365 (exit)      Time to trough: 8 bars
  Max Adverse Excursion (MAE): -3.50%
  Max Favorable Excursion (MFE): +0.00%
```

#### Section 5: Other Signals in Window
```
--- ALL SIGNALS ACTIVE DURING TRADE (03-17 04:00 → 03-18 16:00) ---
  Signal                    Time          Dir    Strength  Category       Outcome
  ──────────────────────────────────────────────────────────────────────────────
  4H_17dates_LONG          03-17 04:00    LONG   5         numerology     LOSS (traded)
  4H_BuyPressure_LONG      03-17 04:00    LONG   1         orderflow      LOSS (traded)
  4H_TweetGematria_LONG    03-17 04:00    LONG   2         manipulation   LOSS (traded)
  [no other signals in window]

--- SIGNALS THAT SHOULD HAVE WARNED ---
  matrix_bear_count=4/8 (bear majority)
  ema50_declining=1 (downtrend)
  hmm_state=1 (HMM says bear regime)
  fear_greed=35 (fear territory)
  caution_date=1 (date 17 is numerological caution)
```

#### Section 6: Post-Mortem Verdict
```
--- VERDICT ---
  The model predicted LONG with 82% confidence but:
  1. Matrix convergence was BEARISH (4 bear / 2 bull domains)
  2. HMM regime was bear (65% bear prob)
  3. EMA50 was declining (downtrend)
  4. Date 17 is a CAUTION number in numerology (contradicts LONG signal)
  5. Fear & Greed at 35 = fear territory

  ROOT CAUSE: The 4H_17dates signal fired because date=17, but 17 is ALSO
  a caution number. The model weighted numerology pump signal over the
  caution signal. The matrix convergence correctly warned (bearish majority)
  but the model's 82% confidence overrode this information.

  RECOMMENDATION: Check if matrix_convergence features have sufficient
  weight in training. The meta-label gate passed at 0.71 — consider
  raising the meta threshold for regimes where matrix disagrees with model.
```

---

## 4. MODE B: SIGNAL CORRELATION

### 4.1 CLI Interface
```bash
# Full signal correlation for a time range
python investigate.py signals --from "2026-03-15" --to "2026-03-20" --tf 4h

# Specific category deep-dive
python investigate.py signals --from "2026-03-15" --to "2026-03-20" --category numerology

# Matrix convergence timeline
python investigate.py signals --from "2026-03-15" --to "2026-03-20" --matrix-only
```

### 4.2 Output: Domain State Timeline
```
===============================================================================
  SIGNAL CORRELATION: 2026-03-15 → 2026-03-20 (4H bars)
===============================================================================

  Time         Price    TA     Num   Gem   Astro  Space  Sports  OnCh  Sent  Matrix   Direction
  ─────────────────────────────────────────────────────────────────────────────────────────────
  03-15 00:00  $75,100  BULL   --    --    --     CALM   --      BULL  NEUT   +2/8     FLAT
  03-15 04:00  $74,800  BEAR   --    PUMP  --     CALM   --      BULL  FEAR   +1/8     BEAR
  03-15 08:00  $74,500  BEAR   --    --    VOC    CALM   UPSET   BULL  FEAR   -1/8     BEAR
  03-15 12:00  $74,200  BEAR   CAUT  --    VOC    STORM  UPSET   BULL  FEAR   -3/8     BEAR >>>
  03-15 16:00  $73,800  BEAR   CAUT  --    --     STORM  --      BEAR  FEAR   -4/8     BEAR >>>
  03-15 20:00  $73,500  BEAR   --    --    --     CALM   --      BEAR  FEAR   -3/8     BEAR
  03-16 00:00  $73,200  BEAR   --    --    --     CALM   --      BEAR  FEAR   -2/8     BEAR
  ...

  CORRELATION MATRIX (signal vs 4H return):
    Category       Pearson r    Hit Rate    Predictive?
    ──────────────────────────────────────────────────
    TA regime       +0.42        67.3%       YES
    Numerology      +0.15        54.2%       WEAK
    Gematria        +0.08        51.1%       NO (this window)
    Astrology       +0.22        58.7%       MODERATE
    Space weather   +0.31        62.1%       YES
    Sports          -0.05        49.3%       NO
    On-chain        +0.28        60.5%       YES
    Sentiment       +0.35        64.8%       YES

  MISSED SIGNALS (signals that were predictive but model ignored):
    - sw_kp_storm at 03-15 12:00 (Kp=5.3) preceded 2.5% drop
    - fear_greed dropped below 35 at 03-15 08:00, stayed low
    - matrix_bear_count reached 4/8 BEFORE the drop started
```

---

## 5. MODE C: TRAINING VERIFICATION

### 5.1 CLI Interface
```bash
# Verify training labels for a period
python investigate.py training --from "2026-03-15" --to "2026-03-20" --tf 4h

# Check if a specific bar was in train or test
python investigate.py training --bar "2026-03-17T04:00" --tf 4h

# Feature availability audit (forward-looking check)
python investigate.py training --leakage-check --tf 4h
```

### 5.2 Output: Label Verification
```
===============================================================================
  TRAINING VERIFICATION: 4H | 2026-03-15 → 2026-03-20
===============================================================================

--- TRIPLE-BARRIER LABELS ---
  Time          Close     ATR(14)   TP Level   SL Level   Label    Actual
  ─────────────────────────────────────────────────────────────────────────
  03-15 00:00   $75,100   $680      $76,460    $73,740    FLAT     price stayed in range (correct)
  03-15 04:00   $74,800   $675      $76,150    $73,450    SHORT    hit SL-side barrier (correct)
  03-15 08:00   $74,500   $670      $75,840    $73,160    SHORT    continued drop (correct)
  03-15 12:00   $74,200   $665      $75,530    $72,870    SHORT    strong drop (correct)
  03-15 16:00   $73,800   $660      $75,120    $72,480    FLAT     bounced within range
  03-16 00:00   $73,200   $655      $74,510    $71,890    LONG     bounced to TP (correct)
  03-17 04:00   $73,955   $673      $75,301    $73,282    SHORT    hit SL then dropped (MODEL SAID LONG!)
  ...

  LABEL ACCURACY IN WINDOW:
    Labels correct (price confirmed barrier): 14/20 = 70.0%
    Labels wrong (price reversed after label): 2/20 = 10.0%
    Labels ambiguous (tight range): 4/20 = 20.0%

--- CPCV FOLD ASSIGNMENT ---
  Source: cpcv_oos_predictions_4h.pkl
  Total CPCV samples: 14,194
  This window (30 bars): bars 13,800 - 13,830

  Bar 03-17 04:00 (idx 13,815):
    CPCV Group:       6 of 6
    In TRAIN set:     Paths 1,2,3,5,7,8,10,12,13,14 (10 of 15)
    In TEST set:      Paths 4,6,9,11,15 (5 of 15)
    OOS prediction:   P(L)=0.78, P(F)=0.12, P(S)=0.10
    Actual label:     SHORT (0)
    MODEL WAS WRONG in OOS — predicted LONG, actual was SHORT

--- FORWARD-LOOKING DATA AUDIT ---
  Checking features at bar 03-17 04:00 for leakage...

  Feature                    Value     Source Time     Lag      Status
  ─────────────────────────────────────────────────────────────────────
  close                      73,954    03-17 04:00     0h       OK (current bar)
  sma_20                     74,200    03-17 04:00     0h       OK (lookback only)
  rsi_14                     42.3      03-17 04:00     0h       OK (lookback only)
  fear_greed                 35        03-16           -28h     OK (previous day)
  tweet_gem_pump             1         03-16 22:00     -6h      OK (past tweet)
  kp_index                   3.7       03-17 03:00     -1h      OK (recent measurement)
  next_4h_return             -3.50%    FORWARD!        +32h     LEAK! (only in targets)
  triple_barrier_label       0         FORWARD!        +32h     LEAK! (only in targets)

  LEAKAGE CHECK: PASSED (forward-looking data only in target columns, not features)
```

---

## 6. IMPLEMENTATION PLAN

### 6.1 Architecture

```
investigate.py                    # Main CLI entry point (~800 lines)
  |
  +-- investigator/
  |     +-- __init__.py
  |     +-- trade_mode.py         # Mode A: Trade investigation (~400 lines)
  |     +-- signal_mode.py        # Mode B: Signal correlation (~350 lines)
  |     +-- training_mode.py      # Mode C: Training verification (~300 lines)
  |     +-- data_loader.py        # Unified data access (wraps data_access.py) (~200 lines)
  |     +-- feature_categorizer.py # Maps ~1,990 features to 9 categories (~150 lines)
  |     +-- report_renderer.py    # Terminal + HTML output (~250 lines)
  |     +-- price_analyzer.py     # MAE/MFE/price action analysis (~150 lines)
```

### 6.2 Format: Python CLI with `argparse`

Rationale:
- CLI is fastest to iterate on
- Can pipe output to files, grep through results
- No server/browser dependency
- Can later be wrapped as a dashboard page if wanted
- Uses existing `data_access.py` and `feature_library.py` directly

### 6.3 Feature Categorization Map

The tool needs to map ~1,990 feature columns to human-readable categories. This is derived from the feature prefixes discovered in the codebase:

```python
FEATURE_CATEGORIES = {
    'ta': {
        'name': 'Technical Analysis',
        'prefixes': ['sma_', 'ema_', 'rsi_', 'macd_', 'bb_', 'atr_',
                     'volume_', 'close_vs_', 'ichimoku_', 'sar_',
                     'wyckoff_', 'stoch_', 'adx_', 'obv_', 'vwap_',
                     'supertrend_', 'keltner_', 'donchian_', 'elliott_',
                     'gann_', 'fvg_', 'ob_', 'bos_', 'return_',
                     'volatility_', 'frac_diff_', 'range_position',
                     'knn_'],
        'exact': ['open', 'high', 'low', 'close', 'volume',
                  'quote_volume', 'trades', 'taker_buy_volume'],
    },
    'numerology': {
        'name': 'Numerology',
        'prefixes': ['pump_', 'caution_', 'price_dr', 'date_reduction',
                     'btc_energy', 'is_master_', 'is_angel_',
                     'day_of_year_dr', 'gem_', 'date_gem_',
                     'sequence_', 'arabic_'],
    },
    'gematria': {
        'name': 'Gematria (Tweets/News)',
        'prefixes': ['tweet_gem_', 'tweet_', 'news_gem_', 'news_',
                     'headline_'],
    },
    'astrology': {
        'name': 'Astrology',
        'prefixes': ['west_', 'vedic_', 'bazi_', 'tzolkin_',
                     'planetary_hour', 'lunar_', 'voc_moon',
                     'eclipse_', 'equinox_'],
    },
    'space_weather': {
        'name': 'Space Weather',
        'prefixes': ['sw_', 'kp_', 'solar_', 'geomag_'],
    },
    'sports': {
        'name': 'Sports',
        'prefixes': ['sport_', 'game_', 'horse_', 'nfl_', 'nba_',
                     'mlb_'],
    },
    'onchain': {
        'name': 'On-Chain',
        'prefixes': ['onchain_', 'hash_rate', 'mempool_', 'funding_',
                     'oi_', 'open_interest'],
    },
    'sentiment': {
        'name': 'Sentiment & Macro',
        'prefixes': ['fear_greed', 'google_trends', 'macro_',
                     'btc_dominance', 'dxy_'],
    },
    'regime': {
        'name': 'Regime & Matrix',
        'prefixes': ['ema50_', 'hmm_', 'regime_', 'matrix_',
                     'h4_trend', 'd_trend', 'w_trend'],
    },
    'cross': {
        'name': 'Cross Features',
        'prefixes': ['cross_', 'tx_', 'px_', 'ex_'],
    },
    'time': {
        'name': 'Time/Calendar',
        'prefixes': ['hour_', 'day_', 'month_', 'is_weekend',
                     'hebrew_', 'mkt_', 'fomc_', 'opex_',
                     'halving_', 'tax_'],
    },
    'htf': {
        'name': 'Higher Timeframe',
        'prefixes': ['h4_', 'd_', 'w_'],
    },
    'cycle': {
        'name': 'Cycles',
        'prefixes': ['cycle_', 'schumann_', 'chakra_', 'jupiter_',
                     'mercury_cycle'],
    },
    'decay': {
        'name': 'Decay Features',
        'prefixes': ['decay_'],
    },
}
```

### 6.4 Data That Needs to Be Stored (Currently Missing)

| Data Gap | What's Missing | Fix |
|---|---|---|
| **Model input vector** | `features_json` in `trades.db` stores feature values, but only for ML trader trades. V2 signal-based paper trades (paper_trades.db) do NOT store the feature snapshot | Add `features_json TEXT` to `positions` table in paper_trades.db. Populate at trade entry |
| **Model probabilities** | `confidence` stores the max prob but not the full 3-class vector P(L/F/S) | Add `prob_long REAL, prob_flat REAL, prob_short REAL` columns to trades table |
| **Meta-label decision** | Whether meta-model passed/rejected + its probability | Add `meta_prob REAL, meta_passed INTEGER` columns |
| **LSTM blend values** | The LSTM probability and final blended probability | Add `lstm_prob REAL, blended_prob REAL` columns |
| **Kelly calculation** | The Kelly fraction used and base/scaled risk | Add `kelly_fraction REAL, base_risk REAL, dd_scale REAL` columns |
| **Matrix convergence at entry** | Stored in features_json (if populated) but should also be top-level | Add `matrix_bull_count INT, matrix_bear_count INT, matrix_net_score REAL` columns |
| **Regime details** | Only stores regime name, not multiplier values | Add `regime_idx INT` column (can derive multipliers from code) |
| **Signals that fired but were NOT traded** | Only traded signals are logged. Signals that fired but were rejected (by confidence threshold, meta gate, dedup, DD halt) are silently dropped | **NEW TABLE**: `signals_evaluated` -- log every signal that fires, with rejection reason |
| **Historical space weather** | Only 1 row in space_weather.db (latest). No history | Accumulate historical readings. Already have kp_7day.json, kp_noaa_daily.json etc. Need a `kp_history` table |
| **CPCV fold membership** | Only stored in pkl file, not easily queryable | On training, save fold assignments to `cpcv_fold_map_{tf}.json` mapping bar_index -> list of fold IDs |

### 6.5 Schema Migrations

```sql
-- trades.db: Add investigation columns
ALTER TABLE trades ADD COLUMN prob_long REAL;
ALTER TABLE trades ADD COLUMN prob_flat REAL;
ALTER TABLE trades ADD COLUMN prob_short REAL;
ALTER TABLE trades ADD COLUMN meta_prob REAL;
ALTER TABLE trades ADD COLUMN meta_passed INTEGER;
ALTER TABLE trades ADD COLUMN lstm_prob REAL;
ALTER TABLE trades ADD COLUMN blended_prob REAL;
ALTER TABLE trades ADD COLUMN kelly_fraction REAL;
ALTER TABLE trades ADD COLUMN base_risk REAL;
ALTER TABLE trades ADD COLUMN dd_scale REAL;
ALTER TABLE trades ADD COLUMN matrix_bull_count INTEGER;
ALTER TABLE trades ADD COLUMN matrix_bear_count INTEGER;
ALTER TABLE trades ADD COLUMN matrix_net_score REAL;
ALTER TABLE trades ADD COLUMN regime_idx INTEGER;

-- trades.db: New table for ALL evaluated signals
CREATE TABLE IF NOT EXISTS signals_evaluated (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    tf TEXT NOT NULL,
    direction TEXT,
    confidence REAL,
    prob_long REAL,
    prob_flat REAL,
    prob_short REAL,
    pred_class INTEGER,
    conf_threshold REAL,
    meta_prob REAL,
    meta_passed INTEGER,
    regime TEXT,
    regime_idx INTEGER,
    matrix_bull_count INTEGER,
    matrix_bear_count INTEGER,
    matrix_net_score REAL,
    price REAL,
    action TEXT NOT NULL,  -- 'traded', 'below_threshold', 'meta_rejected', 'dedup', 'dd_halt', 'pool_halted'
    trade_id INTEGER,      -- FK to trades.id if action='traded'
    features_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- paper_trades.db: Add features to positions
ALTER TABLE positions ADD COLUMN features_json TEXT;

-- space_weather.db: Historical Kp accumulation
CREATE TABLE IF NOT EXISTS kp_history (
    date TEXT PRIMARY KEY,
    kp_index REAL,
    solar_wind_speed REAL,
    solar_wind_bz REAL,
    sunspot_number REAL,
    solar_flux_f107 REAL,
    source TEXT
);
```

### 6.6 Changes to `live_trader.py`

The main trading loop (line ~530-700) needs to log every signal evaluation, not just trades. Here is what changes:

1. **After prediction** (after line ~560): Log the raw 3-class probabilities to `signals_evaluated` regardless of whether a trade is taken.

2. **After meta-gate** (after line ~655): Log the meta_prob and pass/fail.

3. **After Kelly sizing** (after line ~680): Store kelly_fraction, base_risk, dd_scale in the trade record.

4. **On trade insert** (line ~699): Add all the new columns to the INSERT statement.

5. **On signal rejection**: When a signal is below threshold, meta-rejected, deduped, or DD-halted, insert a row into `signals_evaluated` with `action='below_threshold'` etc.

### 6.7 Performance

- **Feature lookup**: Direct SQLite query on `features_{tf}.db` by timestamp. Index exists on timestamp. Sub-100ms for any single bar.
- **Price action**: Pull OHLCV from `btc_prices.db` for the trade window. Instant.
- **Feature categorization**: In-memory dict lookup. Instant.
- **CPCV fold check**: Load pickle once (~2-5 seconds for large files), then index lookup.
- **Full investigation**: < 3 seconds total for any single trade.
- **Signal correlation (wide window)**: Reading 100 bars of 1,990 features = ~200K cells. < 2 seconds.
- **No GPU needed**: This is a read-only investigation tool. All data is pre-computed.

### 6.8 Optional HTML Report

```bash
python investigate.py trade --id 3 --html report_trade3.html
```

Generates a self-contained HTML file with:
- All terminal output in styled `<pre>` blocks
- A small inline price chart (matplotlib -> base64 PNG embedded in HTML)
- Collapsible sections for each feature category
- Color-coded signal states (green=bullish, red=bearish, grey=neutral)

---

## 7. IMPLEMENTATION PRIORITY

### Phase 1 (Day 1): Core Investigation Tool
1. Create `investigate.py` CLI skeleton with argparse
2. Implement `data_loader.py` - unified reader for all DBs
3. Implement `feature_categorizer.py` - map columns to categories
4. Implement `trade_mode.py` - Trade Investigation (reads from both trades.db and paper_trades.db)
5. Implement `price_analyzer.py` - MAE/MFE computation from btc_prices.db

### Phase 2 (Day 2): Signal Correlation + Training
6. Implement `signal_mode.py` - Signal Correlation over time ranges
7. Implement `training_mode.py` - Label verification + CPCV fold lookup
8. Implement `report_renderer.py` - terminal formatter + HTML export

### Phase 3 (Day 3): Instrumentation
9. Apply schema migrations to trades.db
10. Modify `live_trader.py` to log all signal evaluations (signals_evaluated table)
11. Modify `live_trader.py` to store full probability vectors + Kelly details
12. Backfill kp_history from existing JSON files

### Phase 4 (Optional): Dashboard Integration
13. Add `/investigate` page to Next.js dashboard
14. API route that calls `investigate.py` subprocess and returns JSON
15. Interactive time-range selector with plotly chart

---

## 8. KEY DESIGN DECISIONS

| Decision | Choice | Rationale |
|---|---|---|
| Format | Python CLI | Fastest to build, no dependencies, can pipe/grep, later wrappable |
| Storage | SQLite (existing) | Already have all DBs, no new infrastructure |
| Feature DB access | Direct SQL query by timestamp | Pre-computed features exist for all TFs in `features_{tf}.db` |
| Categorization | Prefix-based mapping | Feature names are consistently prefixed (tweet_, west_, sw_, etc.) |
| Price data | From btc_prices.db OHLCV | Already has all timeframes stored |
| CPCV verification | Load pkl file | `cpcv_oos_predictions_{tf}.pkl` already stores fold indices |
| Verdict generation | Rule-based (not AI) | Deterministic: checks matrix vs model direction, regime conflicts, caution overlaps |
| HTML reports | Optional flag | Not everyone needs it; terminal is the primary interface |

---

## 9. SAMPLE COMMAND CHEATSHEET

```bash
# Investigate a specific ML trade
python investigate.py trade --id 5

# Investigate a paper trade
python investigate.py trade --paper-id 1

# See all trades in a time window
python investigate.py trade --from "2026-03-17" --to "2026-03-18"

# Signal correlation for a week on 4H
python investigate.py signals --from "2026-03-10" --to "2026-03-17" --tf 4h

# Just matrix convergence timeline
python investigate.py signals --from "2026-03-15" --to "2026-03-20" --matrix-only

# Deep-dive on numerology signals
python investigate.py signals --from "2026-03-15" --to "2026-03-20" --category numerology

# Verify training labels
python investigate.py training --from "2026-03-15" --to "2026-03-20" --tf 4h

# Check a specific bar's fold assignment
python investigate.py training --bar "2026-03-17T04:00" --tf 4h

# Forward-looking data leakage check
python investigate.py training --leakage-check --tf 4h

# Export HTML report
python investigate.py trade --id 5 --html trade5_report.html
```

---

## 10. DATA FLOW DIAGRAM

```
                    investigate.py
                         |
          +--------------+--------------+
          |              |              |
     trade_mode     signal_mode    training_mode
          |              |              |
          +--------------+--------------+
                         |
                   data_loader.py
                         |
    +----------+---------+--------+---------+
    |          |         |        |         |
trades.db  paper_     features  btc_     cpcv_oos_
           trades.db  _{tf}.db  prices   predictions
                                .db      _{tf}.pkl
    +----------+---------+--------+---------+
    |          |         |        |         |
space_   sports_   tweets  onchain  funding
weather  data.db   .db    _data.db  _rates.db
.db
    +----------+---------+
    |          |         |
astrology  fear_   google_
_full.db   greed  trends.db
           .db
```

---

## 11. FILES TO CREATE/MODIFY

### New Files
| File | Lines (est.) | Purpose |
|---|---|---|
| `investigate.py` | ~150 | CLI entry point, argparse, mode dispatch |
| `investigator/__init__.py` | ~5 | Package init |
| `investigator/data_loader.py` | ~250 | Unified data access across all DBs |
| `investigator/feature_categorizer.py` | ~150 | Map 1,990 features to 9 categories |
| `investigator/trade_mode.py` | ~450 | Mode A implementation |
| `investigator/signal_mode.py` | ~350 | Mode B implementation |
| `investigator/training_mode.py` | ~300 | Mode C implementation |
| `investigator/price_analyzer.py` | ~150 | MAE/MFE/price trajectory |
| `investigator/report_renderer.py` | ~250 | Terminal + HTML output |

### Modified Files
| File | Change |
|---|---|
| `live_trader.py` | Add `signals_evaluated` logging, store full prob vector + Kelly details |

### Total: ~2,050 lines new code + ~50 lines modified

---

## APPENDIX A: FEATURE COUNT BY CATEGORY (from features_4h_all.json, 1,083 features)

| Category | Count | Top Prefixes |
|---|---|---|
| Cross/Trend (tx_, px_, ex_) | 364 | tx_eclipse_x_, px_73_x_, tx_knn_x_ |
| Gematria (tweet_, news_) | 86 | tweet_gem_, news_gem_, tweet_count |
| Cross-domain (cross_) | 49 | cross_moon_x_, cross_fg_x_ |
| Macro | 43 | macro_dxy_, macro_sp500 |
| Sports | 53 | sport_upset_, game_count, horse_ |
| Space weather (sw_) | 17 | sw_kp_, sw_solar_wind_ |
| Technical (sma/ema/rsi/bb/...) | ~200 | sma_, ema_, rsi_, bb_, macd_, atr_ |
| Numerology | ~40 | pump_, caution_, price_dr, date_gem_ |
| Astrology (west_) | ~30 | west_mercury_, west_moon_, voc_ |
| On-chain | ~20 | onchain_, hash_rate, funding_ |
| Regime | ~15 | ema50_, hmm_, range_position |
| Time/Calendar | ~25 | hour_, day_, hebrew_, mkt_ |
| Cycle | ~10 | cycle_schumann_, cycle_eclipse_ |
| KNN | ~10 | knn_direction, knn_strength |
| Other | ~120 | Various cross features |
