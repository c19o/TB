# Savage22 ML Training Guide

Complete pipeline reference — current system + institutional upgrades (Lopez de Prado, Two Sigma methodology).

---

## Prerequisites

### Hardware
| Component | Local | Cloud (RunPod) |
|-----------|-------|----------------|
| CPU | 13900K | H200 SXM (weak CPU) |
| GPU | RTX 3090 24GB | H200 80GB or 4x RTX 5090 |
| RAM | 64GB | 80GB+ |
| Best for | LSTM, exhaustive optimizer | XGBoost bulk training |

**Rule**: Use local 13900K+3090 for LSTM and optimizer (CPU-bound). H200 for XGBoost if speed needed.

### Required Python Packages
```
xgboost, numpy, pandas, scikit-learn, scipy, torch, optuna, pyarrow, hmmlearn, cupy-cuda12x
```

### Required Data Files
```
btc_prices.db          — OHLCV candles (all TFs)
ephemeris_cache.db     — Western astrology (ephemeris)
astrology_full.db      — Vedic + Chinese + Mayan astrology
fear_greed.db          — Fear & Greed index
tweets.db              — Scraped tweets + gematria
news_articles.db       — News headlines + sentiment
sports_results.db      — Sports outcomes
onchain_data.db        — Blockchain metrics
macro_data.db          — SPX, DXY, Gold, etc.
space_weather.db       — Live Kp, solar wind
kp_history.txt         — Historical Kp index (1932-present)
heartbeat_data/results/gcp_hourly_cache.json  — GCP consciousness data
```

---

## Pipeline Overview

```
STEP 0.5        STEP 1          STEP 2              STEP 3           STEP 4              STEP 5           STEP 6
Cross Test      Build           Train XGBoost       Train            Optimize            Train            Verify
(pre-screen) -> Feature DBs --> w/ Purged CV    --> Meta-Labeler --> Trade Params    --> LSTM         --> Backtest + PBO
                                + Embargo           (bet/no-bet)     (ROI/Sharpe/DD)     (Optional)

systematic_     build_*_        ml_multi_tf.py      ml_multi_tf.py   exhaustive_         run_optuna_      full_backtest.py
cross_tester.py features.py     (purged CV mode)    (meta step)      optimizer.py        local.py         + pbo_analysis.py
~5 min          ~60 min         ~2-3 hrs            ~10 min          ~30-60 min          ~2-4 hrs         ~10 min
```

**Outputs at each step:**

| Step | Output Files | Used By |
|------|-------------|---------|
| 0.5 | systematic_cross_results_{tf}.csv | Step 1 (determines which px_ crosses to build) |
| 1 | features_{tf}.db/.parquet | Steps 2-5 |
| 2 | model_{tf}.json, features_{tf}_all.json, platt_{tf}.pkl | Steps 3-4, live_trader |
| 3 | meta_model_{tf}.json | live_trader (bet/no-bet filter) |
| 4 | exhaustive_configs.json | live_trader |
| 5 | lstm_{tf}.pt | (future: live_trader stacking) |
| 6 | pbo_report.json, backtest results | Validation only |

---

## Current vs Institutional Gap Analysis

| Component | Our System Now | Institutional Standard | Status |
|-----------|---------------|----------------------|--------|
| Labels | Triple-barrier (ATR-scaled) | Triple-barrier (vol-scaled) | DONE |
| Walk-forward | 3 rolling windows (naive) | Purged k-fold + CPCV + embargo | UPGRADE NEEDED |
| Sample weighting | Regime 0.15x + esoteric 3x | + Uniqueness weighting (overlap-aware) | UPGRADE NEEDED |
| Feature stationarity | Raw returns/levels | + Fractional differentiation | UPGRADE NEEDED |
| Meta-labeling | None | Bet/no-bet secondary classifier | UPGRADE NEEDED |
| Overfitting control | None measured | PBO + Deflated Sharpe Ratio | UPGRADE NEEDED |
| Feature importance | XGBoost gain (final model) | MDI + MDA + SFI per CV fold | UPGRADE NEEDED |
| Bet sizing | Fixed risk% from optimizer | Probability-scaled (sigmoid/Kelly) | UPGRADE NEEDED |
| Ensemble | XGBoost only | Stacked XGBoost + LSTM meta-model | UPGRADE NEEDED |
| Platt calibration | Per-class multinomial LR | Same | DONE |
| HMM regime | Refitted per window | Same | DONE |
| KNN A/B test | Auto keep/drop | Same | DONE |

---

## Step 1: Build Feature Databases

### Build Order (Staggered)
Small TFs parallel, 15m and 5m solo due to memory.

```bash
# 1. Small TFs (can run in parallel)
python -u build_1w_features.py 2>&1 | tee build_1w.log    # ~10s, 339 rows
python -u build_4h_features.py 2>&1 | tee build_4h.log    # ~1 min, 14K rows

# 2. Medium TFs
python -u build_1h_features.py 2>&1 | tee build_1h.log    # ~3 min, 57K rows

# 3. Large TFs (run solo)
python -u build_15m_features.py 2>&1 | tee build_15m.log  # ~5 min, 217K rows

# 4. Largest (run solo)
python -u build_5m_features.py 2>&1 | tee build_5m.log    # ~10 min, 548K rows
```

### Feature Groups Computed (24 groups, in order)
1. Technical Analysis (RSI, MACD, BB, ATR, Ichimoku, SAR, EMAs, volume profile)
2. Time features (hour/day/month sin/cos, sessions, weekend flags)
3. Numerology (digital roots, gematria, master numbers, planetary day x DR combos)
4. Astrology (moon phase, retrograde, nakshatras, BaZi, Mayan Tzolkin, Arabic lots)
5. Esoteric (tweets, news, sports, onchain, macro correlations)
6. Event astrology (astro state at each tweet/news timestamp)
7. Higher TF context (4h/1d/1w indicators mapped to lower TFs)
8. Regime (EMA50 slope, range position 0-1)
9. Space weather (Kp, solar flux, sunspot)
10. Cycles (Schumann, Jupiter, Mercury, eclipses, equinoxes)
11. Hebrew/cultural calendar (Shemitah, holidays)
12. Market calendar (FOMC, OpEx, halving, tax windows)
13. Composite (vol x direction interactions)
14. Cross-features (eclipse x moon, fear x kp_storm, etc.)
15. Decay features (exponential decay since events)
16. Trend crosses (all signals x bull/bear, x VWAP, x range position)
17. Esoteric x TA crosses (signals x RSI/SAR/BB/MACD/Ichimoku/Wyckoff)
18. KNN pattern features (GPU-accelerated)
19. GCP consciousness features (1h only)
20. Power crosses (px_) -- #73 x TA oversold, #93 x orderbook, numerology x volume, etc.
21. Mirror/inverse numbers (is_72 anti-pump, is_71 recovery, DR mirror match, price mirrors)
22. **DOY flags (doy_1 through doy_365)** -- day-of-year binary flags
23. **Systematic power crosses** -- DOY x all contexts, esoteric x all contexts (from systematic_cross_tester.py)
24. **Confidence-weighted rare signals** -- rare but extreme directional crosses with Bayesian weight

### Step 0.5: Pre-Screen Crosses (NEW — before feature builds)
```bash
python -u systematic_cross_tester.py --tf 1h 4h 2>&1 | tee cross_test.log
# Also run on 5m/15m once those parquets are rebuilt
python -u systematic_cross_tester.py --tf 5m 15m 2>&1 | tee cross_test_intraday.log
```

**What it does:**
- Tests ~588,000 crosses per TF (571 signals x 1,030 contexts)
- Signals include DOY 1-365, all numerology, astrology, TA binary signals
- Contexts include ALL 739 base features, binarized (continuous → top/bottom 20th percentile)
- Dual survival criteria:
  - **Standard:** n≥30, |t|>1.96, |edge|>0.02% (statistically significant)
  - **Rare-but-extreme:** n≥5, hit rate >80% or <20% (directionally consistent)
- Confidence weight: `n / (n + 30)` — Bayesian shrinkage
- Output: `systematic_cross_results_{tf}.csv`

**Results (current):**
- 1H: 4,048 standard + 3,531 rare = 7,579 survivors, 359/365 DOYs covered
- 4H: 3,903 standard + 23,730 rare = 27,633 survivors, 365/365 DOYs covered
- Multi-TF (both 1H+4H): 1,761 survivors
- 5M/15M: pending parquet rebuild

**Statistical power by TF (observations per DOY):**
- 5M: ~5,184 bars/DOY — full context expansion viable, most crosses reach n≥30
- 15M: ~1,728 bars/DOY — most contexts work
- 1H: ~432 bars/DOY — common contexts pass standard, rare contexts need extreme hit rate
- 4H: ~108 bars/DOY — mostly rare signals, need multi-TF confirmation
- 1D: ~6-7 per DOY — rare-but-extreme only (hit rate >80%)
- 1W: ~1 per DOY — DOY crosses not viable

### Confidence-Weighted Rare Signals
Some crosses have very few observations (n=5-29) but extreme directional consistency (>80% hit rate).
These are valid signals but need reduced influence in the model.

**Formula:** `confidence = n / (n + 30)`
| n | confidence | Interpretation |
|---|-----------|---------------|
| 5 | 0.14 | Very low — signal exists but barely trusted |
| 15 | 0.33 | Low-medium — directional hint |
| 30 | 0.50 | Medium — minimum for standard survival |
| 100 | 0.77 | High — reliable |
| 500 | 0.94 | Very high — strong confidence |

**Integration:** Each rare px_ cross gets a companion `_conf` feature:
- `px_doy72_x_kp_severe` = binary cross (0/1)
- `px_doy72_x_kp_severe_conf` = confidence weight (0-1)
- XGBoost can split on the cross for direction and use confidence to size the bet

### Expected Feature Counts (Post-Expansion — "Everything x Everything")
| TF | Rows | Features (v2) | Features (v3 est) | Notes |
|----|------|---------------|-------------------|-------|
| 1W | ~339 | ~1,012 | ~150,000+ | DOY x all 370 contexts |
| 1D | ~2,368 | ~1,049 | ~150,000+ | DOY x all 370 contexts |
| 4H | ~14,194 | ~3,404 | ~150,000+ | DOY x all 370 contexts |
| 1H | ~56,753 | ~3,116 | ~150,000+ | DOY x all 370 contexts + px_sys_ |
| 15M | ~217,446 | ~1,106 | ~150,000+ | DOY x all 370 contexts |
| 5M | ~547,782 | ~1,106 | ~150,000+ | DOY x all 370 contexts |

**Note:** feature_library.py now dynamically discovers ALL base columns as context
dimensions (binary used directly, continuous binarized at 80th/20th percentile).
No hardcoded context list — every feature crosses with every DOY automatically.

### Training Parameters (Post-Expansion)
With 8K-12K features, regularization must increase:

| Parameter | v2 (3K features) | v3 (8K+ features) | Why |
|-----------|-----------------|-------------------|-----|
| colsample_bytree | 0.7 | 0.25-0.35 | Each tree sees fewer features, forces diversity |
| colsample_bynode | 0.7 | 0.40-0.45 | Combined keeps effective features ~3K/tree |
| reg_lambda | 3.0 | 5.0-8.0 | More L2 to suppress spurious splits |
| reg_alpha | 0.1 | 0.3-0.5 | L1 zeroes noise features |
| gamma | 0.3 | 0.5-1.0 | Higher bar for splits |
| min_child_weight | 2 | 3-5 | DOY flags fire ~155 times, need mass per leaf |
| rolling_window_bars (1H) | 13,140 | ~25,000 | Healthy sample:feature ratio |

### GPU Requirements (Post-Expansion)
| TF | DMatrix Size | VRAM Needed | Notes |
|----|-------------|-------------|-------|
| 5M (548K x 10K) | ~21 GB | ~28-32 GB | Needs A100 40GB or RTX 5090 32GB |
| 15M (217K x 10K) | ~8 GB | ~12-15 GB | RTX 3090 24GB works |
| 1H (57K x 10K) | ~2 GB | ~5-7 GB | Any GPU |
| 4H/1D/1W | <1 GB | ~2-3 GB | Any GPU |

### Storage: Parquet Primary, SQLite Legacy
With 3,000+ features, SQLite's 2000 column limit is exceeded. All build scripts now save:
- **Parquet** (primary): `features_{tf}.parquet` -- no column limit, faster reads, smaller files
- **SQLite** (legacy fallback): split into `features_{tf}` + `features_{tf}_ext` if >2000 cols
- Training scripts (ml_multi_tf.py, exhaustive_optimizer.py, lstm_sequence_model.py) try parquet first

**Status of parquet migration:**
- build_1h_features.py: DONE
- build_4h_features.py: DONE
- build_15m_features.py: NEEDS UPDATE
- build_5m_features.py: NEEDS UPDATE
- build_1w_features.py: NEEDS UPDATE (may still fit in SQLite)

### UPGRADE: Fractional Differentiation
Add `fd_close` and `fd_volume` features using fixed-width fractional differencing:
- Find minimum d* where ADF test passes (typically d~0.3-0.5 for BTC)
- d=0 = raw price (non-stationary, full memory), d=1 = returns (stationary, no memory)
- d* = sweet spot: stationary AND retains long-term memory
- Improves longer-horizon predictions (4H, 1D) where price memory matters

### Verification
```bash
python -c "
import pandas as pd
for tf in ['1w', '1d', '4h', '1h', '15m', '5m']:
    try:
        df = pd.read_parquet(f'features_{tf}.parquet')
        print(f'{tf}: {len(df)} rows x {len(df.columns)} cols')
    except:
        import sqlite3
        conn = sqlite3.connect(f'features_{tf}.db')
        df = pd.read_sql('SELECT * FROM features_{tf} LIMIT 1', conn)
        conn.close()
        print(f'{tf}: check DB — {len(df.columns)} cols')
"
```

---

## Step 2: Train XGBoost Models

```bash
python -u ml_multi_tf.py 2>&1 | tee ml_training.log
```

**Duration**: ~17-25 min local, ~4-8 min RunPod H200

### What It Does (Per TF)
1. Loads feature DB (parquet preferred)
2. Uses pre-computed triple-barrier labels (LONG=2, FLAT=1, SHORT=0)
3. Computes sample uniqueness weights (Lopez de Prado)
4. Applies regime-aware weights (counter-trend = 0.15x)
5. Applies esoteric event boost (active esoteric signals up to 3.0x)
6. Final weight = uniqueness x regime_weight x esoteric_weight
7. Runs **purged walk-forward** validation with embargo
8. Refits HMM per window (no future leakage)
9. Trains XGBoost multi:softprob (3-class classifier)
10. KNN A/B test — keeps KNN only if they help
11. Platt calibration (multinomial LR on held-out set)
12. Feature importance stability check (MDI across folds)
13. Outer fold validation at confidence thresholds [0.45-0.70]
14. Saves model artifacts

### Triple-Barrier Config
| TF | TP | SL | Max Hold |
|----|----|----|----------|
| 5M | 1.5x ATR | 1.5x ATR | 48 bars |
| 15M | 2.0x ATR | 2.0x ATR | 32 bars |
| 1H | 2.0x ATR | 2.0x ATR | 24 bars |
| 4H | 2.5x ATR | 2.5x ATR | 16 bars |
| 1D | 3.0x ATR | 3.0x ATR | 10 bars |
| 1W | 3.0x ATR | 3.0x ATR | 6 bars |

### XGBoost Hyperparameters
| TF | max_depth | min_child_weight | subsample | colsample_bytree | reg_lambda | reg_alpha | gamma | learning_rate | trees (max) |
|----|-----------|------------------|-----------|------------------|-----------|-----------|-------|---------------|-------------|
| 1W | 3 | 1 | 0.8 | 0.6 | 8.0 | 0.1 | 0.5 | 0.02 | 800 |
| 1D | 4 | 1 | 0.8 | 0.6 | 6.0 | 0.1 | 0.5 | 0.03 | 800 |
| 4H | 5 | 2 | 0.8 | 0.7 | 3.0 | 0.1 | 0.3 | 0.05 | 800 |
| 1H | 5 | 2 | 0.8 | 0.7 | 3.0 | 0.1 | 0.3 | 0.05 | 800 |
| 15M | 4 | 3 | 0.8 | 0.6 | 5.0 | 0.1 | 0.5 | 0.05 | 800 |
| 5M | 3 | 3 | 0.8 | 0.6 | 8.0 | 0.1 | 0.5 | 0.05 | 800 |

**Design**: Lower TFs get MORE regularization (lower depth, higher lambda). Early stopping at 50 rounds. Trees typically 23-715.

### UPGRADE: Purged Cross-Validation with Embargo

**Problem**: Naive rolling windows don't purge overlapping label horizons. With a 24-bar hold period, the last 24 bars of training leak into test.

**Solution**: For each CV fold:
1. **Purge**: Remove training samples whose label window (entry to barrier touch/expiry) overlaps ANY test sample's label window
2. **Embargo**: Add buffer after each test block (one max_hold_bars period) and remove from training

Implementation: `purged_cv.py` → `PurgedKFold` splitter replacing naive rolling windows.

**Expected Impact**: Accuracy drops 1-3% but becomes REAL. Models generalize better live.

### UPGRADE: Combinatorial Purged CV (CPCV)

**Problem**: 3 walk-forward paths aren't enough for statistical significance.

**Solution**: Split data into N=10 monthly blocks. Test all C(10,2)=45 combinations with purge+embargo. Gives distribution of metrics instead of point estimates.

Reports: median accuracy, 25th percentile, std across all 45 paths.

### UPGRADE: Sample Uniqueness Weighting (Lopez de Prado)

**Problem**: Events with overlapping label windows share information — they're not independent samples.

**Solution**:
1. Build indicator matrix: (n_bars x n_events), 1 if event i active at bar t
2. Concurrency at bar t = sum of active events at t
3. Uniqueness of event i = mean(1/concurrency[t]) over its active window
4. Final weight = uniqueness x regime_weight x esoteric_weight

**Why this helps our esoteric signals**: Esoteric events are SPARSE (many NaN, fire rarely). Sparse events have HIGH uniqueness by definition. Lopez de Prado's framework mathematically validates upweighting rare alternative data signals — exactly what we're doing with the esoteric boost, but now with theoretical justification.

### UPGRADE: Feature Importance Stability (MDI/MDA/SFI)

**Problem**: We only check feature gain from final model. A feature ranking #1 in one fold and #500 in another is noise.

**Solution**: For each CPCV fold:
- **MDI**: Mean Decrease Impurity (fast, from tree splits)
- **MDA**: Mean Decrease Accuracy (permutation, unbiased)
- **SFI**: Single Feature Importance (one feature at a time, most robust)

Compare rankings across folds. Low rank variance = real signal. High rank variance = noise.

### Walk-Forward Config
| TF | Type | Windows | Train Size | Test Size | Purge | Embargo |
|----|------|---------|-----------|-----------|-------|---------|
| 1W | Expanding | 2 | 50-75% | 12.5% | 6 bars | 6 bars |
| 1D | Expanding | 2 | 50-75% | 12.5% | 10 bars | 10 bars |
| 4H | Rolling | 3 | 8,760 (~18mo) | 2,190 | 16 bars | 16 bars |
| 1H | Rolling | 3 | 13,140 (~18mo) | 3,285 | 24 bars | 24 bars |
| 15M | Rolling | 3 | 35,040 (~3mo) | 8,760 | 32 bars | 32 bars |
| 5M | Rolling | 3 | 105,120 (~3mo) | 26,280 | 48 bars | 48 bars |

### Sample Weighting (3 layers, multiplied)
1. **Uniqueness** (Lopez de Prado): overlap-aware, 0-1 range
2. **Counter-trend penalty**: LONG in bear = 0.15x, SHORT in bull = 0.15x
3. **Esoteric boost**: clip(1.0 + 0.5 * min(active_count, 4), 1.0, 3.0)

### Output Files
```
model_{tf}.json              — XGBoost model (native JSON)
features_{tf}_all.json       — Feature column names in model order
platt_{tf}.pkl               — Platt calibration (sklearn LogisticRegression)
ml_multi_tf_results.txt      — Human-readable training log
ml_multi_tf_configs.json     — Structured results (accuracy, features, window metrics)
feature_stability_{tf}.json  — MDI/MDA stability rankings per fold
```

### What "Good" Looks Like (with purged CV — expect lower than naive)
| TF | WF Accuracy | Conf>0.70 Acc | Notes |
|----|-------------|---------------|-------|
| 5M | >68% | >73% | Highest accuracy TF |
| 15M | >57% | >65% | Good trade volume |
| 1H | >55% | >60% | Core TF, most features |
| 4H | >51% | >55% | Lower frequency |
| 1D | >55% | >58% | Small sample, high variance |
| 1W | >65% | N/A | Too few for conf filter |

---

## Step 3: Meta-Labeling (Bet/No-Bet Filter)

```bash
# Runs as part of ml_multi_tf.py with --meta flag, or as separate step
python -u ml_multi_tf.py --meta 2>&1 | tee meta_training.log
```

**Duration**: ~10 min local

### What It Does
1. Takes base model from Step 2 (direction predictor)
2. Generates OOS predictions across walk-forward windows
3. For each LONG/SHORT prediction, checks if triple-barrier outcome matched:
   - Base predicted LONG and barrier hit was LONG → meta_label = 1 (profitable)
   - Base predicted LONG but barrier hit SHORT/FLAT → meta_label = 0 (unprofitable)
4. Trains secondary XGBoost on same features + base_confidence as input
5. Meta-model learns WHEN the base model is right vs wrong

### Live Trading Integration
- Base model predicts direction + confidence
- Meta-model predicts probability that direction is correct
- Only take trade if BOTH: base_conf > threshold AND meta_prob > 0.55
- Meta-probability drives bet sizing (higher meta-prob = bigger position)

### Output Files
```
meta_model_{tf}.json         — Meta-labeling XGBoost model
meta_features_{tf}.json      — Meta-model feature list
```

### Why This Matters
This is the #1 technique that separates institutional ML from retail. Without it, you take every signal above threshold. With it, you learn which setups actually work — including WHEN esoteric signals add value (e.g., moon phase matters during high vol, not during consolidation).

---

## Step 4: Exhaustive Trade Parameter Optimizer

```bash
python -u exhaustive_optimizer.py 2>&1 | tee optimizer.log
```

**Duration**: ~30-60 min local (RTX 3090), ~15 min cloud (4x RTX 5090)

### What It Does
This produces **ROI, Sharpe, drawdown** numbers. Step 2 only trains classifiers.

### Parameters Searched (per TF)
| Parameter | Range | Steps | Description |
|-----------|-------|-------|-------------|
| leverage | 1-125x | 10-20 | Position leverage |
| risk_pct | 0.01-2.0% | 10 | Capital risked per trade |
| stop_atr | 0.05-1.0x | 10-11 | Stop loss as ATR multiple |
| rr | 1.0-4.0 | 11 | Risk:reward ratio |
| max_hold | varies | 10 | Max bars before forced exit |
| exit_type | 6 types | 6 | Partial TP %, trailing stop mult |
| conf_thresh | 0.45-0.90 | 5 | Min model confidence to trade |

**Total**: ~5-12M combos per TF, ~41M across all TFs

### UPGRADE: Probability-Scaled Bet Sizing

**Replace fixed risk%** with probability-scaled sizing:
```
bet_size = b_max * (2 / (1 + exp(-k * (p - 0.5))) - 1)
```
- p = Platt-calibrated probability (or meta-model probability)
- k = slope (steeper = more aggressive at high confidence)
- b_max = cap from optimizer's max risk%

Or Kelly criterion: `f* = (p * rr - (1-p)) / rr`

More capital on high-conviction, less on uncertain = better Sharpe.

### 4 Optimization Profiles
| Profile | Constraint | Objective |
|---------|-----------|-----------|
| dd10_best | Max DD <= 10% | Maximize final balance |
| dd10_sortino | Max DD <= 10% | Maximize Sortino ratio |
| dd15_best | Max DD <= 15% | Maximize final balance |
| dd15_sortino | Max DD <= 15% | Maximize Sortino ratio |

All require minimum 10 trades.

### UPGRADE: Deflated Sharpe Ratio
After finding best config, compute DSR = Sharpe adjusted for:
- Number of trials tested (41M combos = massive inflation)
- Return skewness and kurtosis
- Non-normal distribution correction

DSR > 2.0 with 41M trials = genuinely strong. DSR < 1.0 = likely overfit.

### Output
```
exhaustive_configs.json      — Best params per TF per profile
```

live_trader.py picks profile priority: dd15_best > dd10_best > dd15_sortino.

---

## Step 5: LSTM Training (Optional)

```bash
python -u run_optuna_local.py 2>&1 | tee lstm_training.log
```

**Duration**: ~2-4 hours on RTX 3090

### What It Does
1. Optuna hyperparameter search per TF:
   - window: 8-80, hidden: 64/128/256/512, layers: 1/2/3
   - dropout: 0.1-0.6, lr: 1e-4 to 3e-3, batch: 32/64/128
2. 100 trials (1w/1d), 200 trials (4h/1h/15m/5m), 20 min timeout
3. Train 5-seed ensemble with best config
4. Each seed: 80 epochs, early stopping patience=20

### Architecture
```
Input: (batch, window, n_features) sliding window
  → LSTM (hidden_size, num_layers, dropout)
  → FC (hidden -> hidden/2) → ReLU
  → FC (hidden/2 -> 1) → Sigmoid
Output: P(next bar goes UP)
```

### UPGRADE: Stacking with XGBoost
Future integration:
- LSTM outputs hidden state features per bar
- These become additional features for XGBoost meta-model
- Purged CV ensures no leakage between LSTM training and XGBoost training
- Simple LogisticRegression meta-model combines XGBoost + LSTM OOS predictions

### Output
```
lstm_{tf}.pt                 — 5-seed ensemble + config + normalization params
```

Status: NOT yet wired into live_trader.py.

---

## Step 6: Verification + Overfitting Analysis

### Check Model Artifacts
```bash
python -c "
import os
required = []
for tf in ['5m', '15m', '1h', '4h', '1d', '1w']:
    required.extend([f'model_{tf}.json', f'features_{tf}_all.json', f'platt_{tf}.pkl'])
required += ['exhaustive_configs.json', 'ml_multi_tf_configs.json']
for f in required:
    exists = os.path.exists(f)
    size = os.path.getsize(f) if exists else 0
    print(f'  {\"OK\" if exists else \"!!\"} {f:40s} {size/1024:.0f}K' if exists else f'  !! {f:40s} MISSING')
"
```

### Verify live_trader Can Load
```bash
python -c "
import xgboost as xgb, json, pickle
for tf in ['5m', '15m', '1h', '4h', '1d', '1w']:
    m = xgb.Booster(); m.load_model(f'model_{tf}.json')
    feats = json.load(open(f'features_{tf}_all.json'))
    platt = pickle.load(open(f'platt_{tf}.pkl', 'rb'))
    print(f'{tf}: {m.num_boosted_rounds()} trees, {len(feats)} features, platt OK')
"
```

### UPGRADE: Backtest Overfitting Probability (PBO)

**Problem**: Exhaustive optimizer tests 41M combos. Best one's Sharpe is inflated.

**Solution** (Bailey & Lopez de Prado):
1. Run CPCV across top 100 configs from optimizer
2. For each config: get IS and OOS Sharpe across all 45 CPCV paths
3. For each path: rank configs by IS Sharpe, check where IS-best lands in OOS ranking
4. PBO = fraction of paths where IS-best is below OOS median
5. **PBO < 0.3 = real signal. PBO > 0.5 = overfit. Reject.**

```bash
python -u pbo_analysis.py 2>&1 | tee pbo.log
```

Output: `pbo_report.json` with PBO per TF and deflated Sharpe ratios.

---

## Quick Reference: Command Cheat Sheet

### Full Institutional Pipeline (Local, ~4-6 hours)
```bash
cd "C:\Users\C\Documents\Savage22 Server"

# Step 1: Build features (staggered)
python -u build_1w_features.py 2>&1 | tee build_1w.log
python -u build_4h_features.py 2>&1 | tee build_4h.log
python -u build_1h_features.py 2>&1 | tee build_1h.log
python -u build_15m_features.py 2>&1 | tee build_15m.log
python -u build_5m_features.py 2>&1 | tee build_5m.log

# Step 2: Train XGBoost (purged CV + embargo + uniqueness weights)
python -u ml_multi_tf.py 2>&1 | tee ml_training.log

# Step 3: Meta-labeling
python -u ml_multi_tf.py --meta 2>&1 | tee meta_training.log

# Step 4: Optimize trade params (GPU)
python -u exhaustive_optimizer.py 2>&1 | tee optimizer.log

# Step 5: Train LSTM (optional, longest step)
python -u run_optuna_local.py 2>&1 | tee lstm_training.log

# Step 6: Verify + PBO analysis
python -u pbo_analysis.py 2>&1 | tee pbo.log
```

### Just Retrain 1H (Quick, ~5 min — no institutional upgrades)
```bash
python -u build_1h_features.py 2>&1 | tee build_1h.log
python -u ml_multi_tf.py 2>&1 | tee ml_training.log
```

---

## Appendix A: Model Artifacts

| File | Format | Required? | Purpose |
|------|--------|-----------|---------|
| model_{tf}.json | XGBoost JSON | YES | 3-class classifier |
| features_{tf}_all.json | JSON array | YES | Feature names in model order |
| platt_{tf}.pkl | Pickle (sklearn LR) | Optional | Calibrates probabilities |
| meta_model_{tf}.json | XGBoost JSON | Recommended | Bet/no-bet filter |
| exhaustive_configs.json | JSON | Optional | Optimal trade params |
| ml_multi_tf_configs.json | JSON | Optional | Training metadata |
| lstm_{tf}.pt | PyTorch checkpoint | No | Not yet in live trader |
| pbo_report.json | JSON | Validation | Overfitting probability per TF |
| feature_stability_{tf}.json | JSON | Validation | MDI/MDA rankings per fold |

### Data Flow
```
btc_prices.db + auxiliary DBs
        |
        v
feature_library.build_all_features()      <-- single source of truth
        |
        v
features_{tf}.db / .parquet
        |
        v
ml_multi_tf.py (purged CV + embargo)  --> model_{tf}.json + platt_{tf}.pkl
        |
        v
ml_multi_tf.py --meta                 --> meta_model_{tf}.json
        |
        v
exhaustive_optimizer.py                --> exhaustive_configs.json
        |
        v
pbo_analysis.py                        --> pbo_report.json (overfitting check)
        |
        v
live_trader.py (loads all artifacts, trades on Bitget)
        |
        +--> base model predicts direction + confidence
        +--> meta model filters: take trade or skip?
        +--> bet size = f(meta_probability, Kelly, Platt-calibrated confidence)
        +--> execute with optimizer's leverage/stop/TP/hold params
```

---

## Appendix B: Why Institutional Methods Validate Our Esoteric Signals

Institutions call them "alternative data." The framework doesn't care about the source — it cares about statistical rigor.

1. **Sample uniqueness** naturally upweights sparse esoteric signals. They fire rarely, don't overlap with each other, and have HIGH uniqueness scores. Lopez de Prado says these are the most informative samples.

2. **Meta-labeling** learns WHEN esoteric signals add value. Moon phase during high vol = signal. Moon phase during consolidation = noise. The meta-model captures this automatically.

3. **Feature stability (MDI/MDA/SFI)** proves which esoteric features have consistent predictive power across time. Spurious correlations get flagged. Real signals survive.

4. **CPCV + PBO** gives statistical proof across 45 independent paths. If esoteric-enhanced configs have low PBO and positive 25th-percentile Sharpe, the alpha is real.

5. **Orthogonality test**: Train model with and without esoteric features. If removing them degrades CPCV metrics consistently across folds/regimes, they're adding real information. If degradation is inconsistent or concentrated in one path, it's overfitting.

The signals don't need defending. They need the same validation every institutional signal gets. If they survive purged CV, CPCV, PBO, and stability analysis — they're alpha.

---

## Appendix C: Implementation Priority

| # | Component | Effort | Impact | File to Modify |
|---|-----------|--------|--------|----------------|
| 1 | Purged CV + Embargo | 1 day | CRITICAL | New: purged_cv.py, modify ml_multi_tf.py |
| 2 | CPCV | 1 day | CRITICAL | New: cpcv.py, modify ml_multi_tf.py |
| 3 | Sample Uniqueness | 0.5 day | HIGH | ml_multi_tf.py (weighting section) |
| 4 | Meta-Labeling | 1 day | HIGH | ml_multi_tf.py (new step after training) |
| 5 | Fractional Diff | 0.5 day | MODERATE | feature_library.py (new feature group) |
| 6 | PBO + Deflated Sharpe | 1 day | CRITICAL | New: pbo_analysis.py |
| 7 | Bet Sizing | 0.5 day | HIGH | live_trader.py |
| 8 | Feature Stability | 0.5 day | MODERATE | ml_multi_tf.py (after CPCV) |

**Total: ~6 days for full institutional upgrade**

---

## References

- Lopez de Prado, "Advances in Financial Machine Learning" (2018) — Chapters 3-12
- Bailey & Lopez de Prado, "The Probability of Backtest Overfitting" (2014) — SSRN
- Two Sigma, "A Machine Learning Approach to Regime Modeling" (public blog)
- Lopez de Prado, "The 10 Reasons Most Machine Learning Funds Fail" (2018) — JPM
- Wikipedia, "Purged cross-validation" — implementation reference
