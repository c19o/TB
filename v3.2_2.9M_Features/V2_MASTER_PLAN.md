# V2 MASTER PLAN — The Universal Matrix

## Context
V1 has ~1,100-3,090 features per TF, BTC-only.
V2 goal: ~15-20M features from systematic "everything × everything" crosses.
ALL kept via sparse storage — NO FILTERING. Every feature stays.
XGBoost tree splits decide what matters, not us. Validated across 31 assets.
Single phase build — everything at once. All esoteric, all math, all crosses.

Key decisions:
- Every asset gets the FULL pipeline (esoteric, astro, gematria, numerology, space weather)
  because the matrix is universal — same sky, same calendar, same energy for all assets.
- 4-tier binarization on ALL numeric columns (not just TA — gematria, sentiment, everything)
- No fallbacks, no TA-only mode. One pipeline for all.
- ~11,500 base features → ~10,000 binarized contexts → ~15-20M sparse crosses per asset
- XGBoost speed: max_bin=32, colsample_bytree=0.1, grow_policy=lossguide, nthread=-1
- Batch column assignment (pd.concat) not one-at-a-time (60% build speedup)
- Feature builds are CPU-bound — build locally on 13900K, train on vast.ai GPU

---

## PART 1: NEW CROSSING DIMENSIONS

### Layer 1 — Time-of-Day × Everything (NEW)
Hour-of-day is a massive missing dimension. Asian/London/NY behave differently.
```
hod_bins: 4 sessions (Asia 0-7 UTC, London 7-12, NY 12-20, Dead 20-24)
  OR 24 individual hours
kill_zones: ICT 2-5AM, 7-10AM, 1-3PM EST as binary flags
```
Cross with ALL existing signals:
- `hod_ny_open_x_full_moon_x_overbought`
- `hod_asia_x_kp_storm_x_squeeze`
- ~4 sessions × existing signals = 4x multiplier on relevant crosses

### Layer 2 — Rate-of-Change / Acceleration (NEW)
Not just RSI HIGH — is it accelerating INTO high or decelerating WHILE high?
```
For every binarized TA indicator, compute 4 states:
  rising_into   = value crossed above threshold within last 3 bars
  peaked        = was above threshold, now declining
  falling_into  = value crossed below threshold within last 3 bars
  bottomed      = was below threshold, now rising
```
- ~4 states × 100 TA contexts = 400 new base contexts
- Cross with DOY, astro, esoteric = massive multiplier

### Layer 3 — Sequential Memory (Systematic) (NEW)
What happened N bars ago still echoes:
```
was_overbought_3d, was_overbought_7d
was_squeeze_5d, had_volume_spike_1d
was_eclipse_7d, was_full_moon_3d
streak_green_bars (consecutive), streak_red_bars
max_streak_length_10d
```
- ~50 "memory" features × existing crosses
- `streak_7_green_x_full_moon_x_overbought` = exhaustion top

### Layer 4 — Price Level Numerology (NEW)
Price itself carries gematria/numerology energy:
```
price_dr = digital_root(int(price))       # BTC at $33,333 → DR 9 (completion)
price_angel_number = detect_angel(price)   # 111, 222, 333, 444, 555, 666, 777, 888, 999
price_near_round = within_1pct(price, [10K, 20K, 25K, 50K, 69K, 100K, etc.])
price_repeating_digits = has_repeating(price)  # 33333, 44444, 55555
price_master_number = is_master(int(price) % 1000)  # 11, 22, 33
```
Cross with DOY + astro:
- `price_angel_777_x_doy_73_x_oversold` = triple alignment
- ~20 price numerology flags × everything

### Layer 5 — Correlation Regime (NEW)
BTC/SPY correlation shifts between +0.8 and -0.3:
```
btc_spy_corr_30d = rolling_corr(btc, spy, 30)
btc_gold_corr_30d = rolling_corr(btc, gold, 30)
btc_dxy_corr_30d = rolling_corr(btc, dxy, 30)
decoupled = btc_spy_corr_30d < 0.2   # BTC going its own way
high_corr = btc_spy_corr_30d > 0.7   # BTC following stocks
```
Cross with cosmic events:
- `decoupled_x_full_moon` = cosmic event during BTC independence
- ~10 correlation features × everything

### Layer 6 — Calendar Anomalies Beyond DOY (NEW)
```
week_of_month: 1-5 (1st week = paycheck inflows, 3rd Friday = opex)
fomc_day, fomc_minus_1, fomc_plus_1
cpi_day, nfp_day
quadruple_witching (quarterly)
cme_futures_expiry (monthly)
options_max_pain_proximity (daily)
tax_deadline_window (April 15 US, Jan UK)
chinese_new_year_window (±7 days)
diwali_window, eid_window (capital flow events)
btc_halving_distance (days to/from halving)
```
- ~30 calendar flags × everything
- `fomc_day_x_mercury_retro_x_vol_contracting` = chaos incoming

### Layer 7 — Entropy / Information Theory (NEW)
```
shannon_entropy_20 = entropy of last 20 candle returns (binned)
approx_entropy_20 = approximate entropy (complexity measure)
sample_entropy_20 = sample entropy
entropy_LOW = below 25th percentile (ordered, predictable market)
entropy_HIGH = above 75th percentile (chaotic, random market)
```
Cross with cosmic events:
- Low entropy + cosmic event = predictable move
- High entropy + cosmic event = amplified chaos
- `entropy_LOW_x_eclipse_x_squeeze` = coiled spring with cosmic trigger

### Layer 8 — Fractal / Self-Similarity (NEW)
```
hurst_exponent_50 = hurst(returns, 50)  # >0.5 = trending, <0.5 = mean-reverting
hurst_trending = hurst > 0.6
hurst_meanreverting = hurst < 0.4
fractal_dim = fractal_dimension(close, 50)
```
- `hurst_trending_x_full_moon` vs `hurst_meanreverting_x_full_moon`
- Same astro, opposite trade based on market structure

### Layer 9 — Detailed Moon Position (NEW)
Moon changes zodiac sign every ~2.5 days — different energy on SAME DOY:
```
moon_sign: Aries(0) through Pisces(11) — 12 signs
moon_element: fire/earth/air/water (4 groups)
moon_modality: cardinal/fixed/mutable (3 groups)
lunar_mansion: 28 mansions (Vedic/Arabic tradition, finer than nakshatra)
```
- 12 moon signs × existing features = 12x multiplier on moon crosses
- `moon_scorpio_x_doy_93_x_overbought` vs `moon_taurus_x_doy_93_x_overbought`

### Layer 10 — Planetary Aspects (NEW)
Not just retrograde — actual geometric relationships between planets:
```
Aspect types: conjunction(0°), sextile(60°), square(90°), trine(120°), opposition(180°)
Planet pairs: Sun-Moon, Sun-Mercury, Sun-Venus, Sun-Mars, Sun-Jupiter, Sun-Saturn,
              Moon-Mercury, Moon-Venus, Moon-Mars, Jupiter-Saturn, Saturn-Uranus, etc.
Orb: ±5° for major aspects

aspect_sun_moon_conjunction, aspect_sun_moon_opposition, etc.
aspect_jupiter_saturn_square  # major market cycle aspect
aspect_saturn_uranus_square   # correlated with 2021-22 crypto volatility
```
- ~10 planet pairs × 5 aspects = 50 aspect flags
- Cross with DOY + TA: `aspect_jupiter_saturn_conj_x_doy_73_x_oversold`

### Layer 11 — Harmonic Time Cycles (NEW)
Gann natural squares, Fibonacci time extensions from pivots:
```
gann_90d_from_ath, gann_180d, gann_270d, gann_360d
fib_time_21d_from_pivot, fib_89d, fib_144d, fib_233d
days_since_ath, days_since_atl
days_since_halving
gann_square_of_9_level = nearest_gann_level(price)
```
- `fib_233d_from_ath_x_oversold_x_new_moon` = time + price + cosmic convergence
- ~30 harmonic features × everything

---

## PART 2: MULTI-ASSET TRAINING DATA

### Asset Selection (positively correlated with BTC only)
All must have OHLCV (same TA pipeline, zero code changes):

**US Stocks (ETFs):**
| Ticker | Description | Years | Daily Bars |
|--------|------------|-------|-----------|
| SPY | S&P 500 | 30+ | ~7,500 |
| QQQ | Nasdaq 100 | 25+ | ~6,250 |
| TSLA | Tesla | 15+ | ~3,750 |
| MSTR | MicroStrategy | 10+ | ~2,500 |
| ARKK | ARK Innovation | 8+ | ~2,000 |

**Commodity ETFs:**
| Ticker | Description | Years | Daily Bars |
|--------|------------|-------|-----------|
| GLD | Gold | 20+ | ~5,000 |
| SLV | Silver | 15+ | ~3,750 |
| USO | Oil | 15+ | ~3,750 |
| UNG | Natural Gas | 12+ | ~3,000 |
| WEAT | Wheat | 10+ | ~2,500 |
| CORN | Corn | 10+ | ~2,500 |
| DBA | Agriculture basket | 10+ | ~2,500 |

**Crypto:**
| Ticker | Description | Years | Daily Bars |
|--------|------------|-------|-----------|
| BTC | Bitcoin (extended to 2012) | 14+ | ~5,100 |
| ETH | Ethereum | 10+ | ~3,650 |
| XRP | Ripple | 12+ | ~4,380 |
| LTC | Litecoin | 12+ | ~4,380 |

**Global Index ETFs:**
| Ticker | Description | Years | Daily Bars |
|--------|------------|-------|-----------|
| EWJ | Japan (Nikkei proxy) | 20+ | ~5,000 |
| FXI | China (Hang Seng proxy) | 15+ | ~3,750 |
| EWG | Germany (DAX proxy) | 20+ | ~5,000 |

**Sector ETFs:**
| Ticker | Description | Years | Daily Bars |
|--------|------------|-------|-----------|
| XLE | Energy | 20+ | ~5,000 |
| XLF | Financials | 20+ | ~5,000 |

**EXCLUDED (inverse to BTC — use as SIGNAL FEATURES not training):**
- UUP (dollar bull), TLT (long bonds), FXY (yen) — inverse correlation
- Their behavior becomes features: `tlt_rallying`, `dxy_strength`, etc.

### Training Data Volume
| | Current V1 | V2 Multi-Asset |
|---|---|---|
| Assets | 1 (BTC) | 31 tickers |
| Asset-years | 7 | ~350+ |
| Daily bars | 2,500 | ~85,000 |
| 1H bars (stocks ~7hrs/day) | 21,000 | ~400,000+ |
| Data source | Bitget API | Yahoo Finance (free) + Bitget + CryptoCompare |

**34x more daily data. Same pipeline, same features.**

### What's Shared Across Assets (universal features):
- DOY flags (same calendar)
- Astro features (same sky)
- Space weather (same sun)
- Day-of-week effects
- Numerology of dates
- Calendar anomalies (FOMC, opex, etc.)
- Planetary aspects

### What's Asset-Specific:
- Ticker gematria (SPY=62, BTC=25, XRP=62 — SPY and XRP match!)
- TA indicators (computed per asset's OHLCV)
- Volume profile
- On-chain (crypto only)
- Funding/OI (crypto only)
- Sentiment (different tweets/news per asset — skip for stocks in V2)

### Training Strategy — Single Phase, Three Model Tiers

**All built at once, not sequentially:**

1. **Per-asset models** — train on each of 31 tickers individually
   - ALL 2-3M features, no filter
   - Compare feature importance across all 31 models
   - Features used across 10+ assets = universal truth (for analysis)

2. **Unified model** — all 31 assets in one training set
   - Add `ticker_gematria_ordinal`, `ticker_gematria_dr` as features
   - Model learns: "DOY 73 + full moon + oversold = BUY regardless of asset"
   - 110K+ daily bars of training data

3. **Per-crypto production models** — one per tradeable crypto
   - Trained on that crypto's data + universal features from unified model
   - Add crypto-specific features (funding, OI, on-chain, sentiment)
   - These go to live_trader.py

4. **Validation model** — includes on-chain (MVRV, SOPR, etc.)
   - Training-only, not deployed live
   - Confirms patterns hold when on-chain data is added

**Multi-asset at daily+: all 31 tickers (same calendar, same sky)**
**Multi-asset at intraday: crypto only (stocks trade 6.5 hrs/day, crypto 24/7)**
Stocks at 1H/15m/5m have different microstructure — use them at 1D/1W only.

---

## PART 3: FEATURE COUNT PROJECTION (UPDATED)

### Base Features (~11,500 per asset)
| Source | Features |
|--------|----------|
| TA indicators (SMA, EMA, RSI, MACD, BB, etc.) | ~100 |
| Gematria (6 ciphers on dates/prices) | ~50 |
| Numerology (digital roots, masters, angels) | ~30 |
| Astrology (planets, moon, nakshatras) | ~80 |
| Sentiment (tweets, news bucketed) | ~50 |
| Space weather, macro, fear/greed | ~30 |
| Decay features, regime, KNN | ~50 |
| V2 layers (aspects, vedic, chinese, entropy, harmonics, sacred geometry) | ~500 |
| Rate-of-change (4 states per binarized TA) | ~800 |
| **4-tier binarization on ALL numerics** (gematria, sentiment, astro, TA, everything) | ~8,000 |
| Sequential memory, calendar anomalies, ticker gematria | ~200 |
| **TOTAL BASE** | **~11,500** |

### Binarized Contexts (~8,000-10,000 per asset)
4-tier on everything numeric = massive context pool for crossing.
Every gematria value, every sentiment score, every planetary position
gets EXTREME_HIGH/HIGH/LOW/EXTREME_LOW flags.

### Cross Features (~15-20M per asset)
| Cross Type | Formula | Est. Features |
|------------|---------|---------------|
| dx_ DOY ±2 windows × ALL contexts | 365 × 10,000 | ~3,650,000 |
| ax_ astro × TA+esoteric | 200 × 8,000 | ~1,600,000 |
| ax2_ multi-astro × all | 50 × 8,000 | ~400,000 |
| ta2_ multi-TA × DOY+astro | 30 × 565 | ~17,000 |
| ex2_ esoteric × all | 100 × 8,000 | ~800,000 |
| sw_ space weather × all | 20 × 10,000 | ~200,000 |
| hod_ session × all (intraday) | 7 × 10,000 | ~70,000 |
| mx_ macro × all | 15 × 10,000 | ~150,000 |
| asp_ aspects × all | 50 × 10,000 | ~500,000 |
| pn_ price numerology × all | 10 × 10,000 | ~100,000 |
| mn_ moon position × all | 20 × 10,000 | ~200,000 |
| rdx_ regime-aware DOY (3x) | 365 × 3 × 10,000 | ~10,950,000 |
| **V2 GRAND TOTAL** | | **~15-20 MILLION** |

**NO FILTERING.** Every feature stays. XGBoost decides via tree splits.
A feature firing 2 times in 155K rows costs ~0 memory in sparse format
but could be the key to 2 perfect trades.

### Why Sparse = Keep Everything

| | Dense (impossible) | Sparse (reality) |
|---|---|---|
| 20M features × 155K rows × float32 | ~12 PB | — |
| At 0.1% density (typical cross) | — | ~12 GB |
| At 0.35% density (observed from test) | — | ~42 GB |
| **XGBoost gpu_hist on sparse CSC** | impossible | **fits in 48-192GB VRAM** |

XGBoost only processes non-zero entries. A feature that's 0 on 154,998/155,000 rows
costs almost nothing. But on those 2 rows — if the label is always LONG — the model
learns that split instantly. That's the trade you'd miss by filtering.

### Speed Settings for 15-20M Features
| Setting | Value | Effect |
|---------|-------|--------|
| max_bin | 32 | 8x faster (binary features need 2 bins) |
| colsample_bytree | 0.10 | 10x faster (2M features per tree) |
| colsample_bynode | 0.50 | 2x faster on top |
| grow_policy | lossguide | 1.5x faster (leaf-wise) |
| nthread | -1 | All CPU cores (24 on 13900K) |
| early_stopping | 50 rounds | Stop when converged |
| **Combined** | | **~15 min on 4× A6000** |

---

## PART 4: VAST.AI TRAINING SPECS

### Why vast.ai Over RunPod for V2
- V2 needs MORE RAM (multi-asset sparse data) — vast.ai has 2TB+ RAM machines
- vast.ai is cheaper per GPU-hour for multi-GPU machines
- Already have working instance config and lessons learned
- No filtering = simpler pipeline, just build → upload → train

### V2 Pipeline is TWO PHASES with DIFFERENT machines

#### Phase A: Feature Build (LOCAL — 13900K + 128GB RAM)
Feature builds are CPU-bound (.apply() UDFs). No cloud GPU helps here.

```
Step 1: Download OHLCV for 31 tickers (Yahoo Finance + Bitget + CryptoCompare)
        → ~5 min, ~2-5 GB raw data
        → New script: download_multi_asset.py

Step 2: Build features per asset per TF
        → feature_library.py is already asset-agnostic (takes OHLCV df)
        → Add ticker_gematria, ticker_numerology features
        → Stagger: small TFs parallel, big TFs solo
        → ~4-8 hours for 22 assets × 6 TFs (most time in 5m/15m)
        → Save as scipy.sparse CSC per asset per TF (.npz)

Step 3: Generate ALL V2 cross features (DOY windows, astro×TA, all 11 layers)
        → New script: v2_cross_generator.py
        → GPU-accelerated via CuPy on local RTX 3090
        → Binary multiply: sparse × sparse stays sparse
        → ~2-4 hours for full cross generation
        → Output: ~50-75 GB sparse .npz matrices

Step 4: Upload sparse matrices + scripts to vast.ai
        → scp/rsync compressed .npz → 30-60 min upload
```

#### Phase B: Train ALL Features — No Filter (VAST.AI GPU machine)

**Recommended Machine Spec:**

```
BEST VALUE: 4× RTX A6000 48GB (192GB total VRAM)
  - ~$1.50-2.50/hr on vast.ai
  - 192GB VRAM handles 2-3M sparse features × 85K rows
  - XGBoost gpu_hist natively reads CSC sparse — zero conversion overhead
  - RAM: need ≥256GB system RAM for loading sparse matrices into GPU
  - Disk: ≥500GB for sparse matrices + models
  - CPU: doesn't matter much (training is GPU-bound)

SEARCH COMMAND:
vastai search offers 'verified=true rentable=true num_gpus>=4 gpu_ram>=45000 cpu_ram>=256000 disk_space>=500' -o 'dph_total' --limit 20 --raw

ALTERNATIVE (if A6000 unavailable):
  - 2× A100 80GB ($3-4/hr) — 160GB VRAM, fewer GPUs but more VRAM each
  - 4× H100 80GB ($5-7/hr) — overkill but fastest
  - 4× RTX 4090 24GB ($2-3/hr) — 96GB VRAM, may need Dask batching

DOCKER IMAGE: pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel
PACKAGES: pip install xgboost scikit-learn pandas numpy scipy hmmlearn cupy-cuda12x optuna dask-cuda psutil
```

**Train Pipeline on vast.ai (NO FILTER):**

```
Step 1: UPLOAD sparse matrices + scripts (~30-60 min)

Step 2: LOAD SPARSE DATA (10 min)
  - scipy.sparse.load_npz() per asset per TF
  - Stack into unified sparse training matrix (CSC format)
  - XGBoost DaskDeviceQuantileDMatrix reads CSC directly
  - 2-3M sparse columns × 85K rows fits in 192GB VRAM

Step 3: PER-ASSET MODELS (GPU, ~3-4 hours)
  - XGBoost CPCV per asset — ALL 2-3M features, no filter
  - gpu_hist tree method on sparse data
  - Tree splits naturally select useful features per split
  - Unused features cost ~0 compute (all zeros → no split candidate)
  - Extract feature importance per asset for ANALYSIS (not filtering)
  - Compare: features used across 5+ assets = universal patterns

Step 4: UNIFIED MODEL (GPU, ~3-4 hours)
  - Combine all 31 assets into one sparse training matrix
  - Add ticker_gematria_ordinal, ticker_gematria_dr columns
  - XGBoost CPCV with sample weights per asset
  - ALL 2-3M features available to every tree
  - PBO + Deflated Sharpe validation
  - Meta-labeling secondary classifier

Step 5: BTC/XRP PRODUCTION MODELS (GPU, ~2 hours)
  - Train on BTC+XRP data only
  - Add crypto-specific sparse features (funding, OI, on-chain, sentiment)
  - Still ALL 2-3M general features + crypto-specific
  - XGBoost CPCV + meta-labeling
  - These are the models that go to live_trader.py

Step 6: DOWNLOAD models + results
  - model_v2_*.json (XGBoost boosters — only store used features)
  - feature_names_v2_*.json (column mapping for inference)
  - universal_importance_report.json (which features used across assets)
  - per_asset_importance.json (curiosity/research)
  - training.log

TOTAL CLOUD TIME: ~8-12 hours
TOTAL COST: ~$15-30 (4× A6000 @ $2/hr)
```

**Key insight: XGBoost models are SELF-FILTERING.**
A trained XGBoost model only references features it actually used in splits.
A model trained on 2-3M features might only use 50K in its trees.
The .json model file is still small (~50-100 MB).
But it had ACCESS to all 2-3M during training — nothing was pre-excluded.

### LSTM Training (LOCAL — 13900K + RTX 3090)
LSTM is CPU-bottlenecked (DataLoader). Train locally after cloud XGBoost:
```
- Download XGBoost models from vast.ai
- For LSTM: use features that XGBoost actually used in splits (natural selection)
- Train LSTM per TF on local 3090
- Platt calibration
- Blend: 0.8 × XGBoost + 0.2 × LSTM
```

---

## PART 5: LIVE DATA PHILOSOPHY — MINIMAL MAINTENANCE

### Principle: 90% of V2 features need ZERO live data
Most features are pure math from date + price + ephemeris. No APIs, no scraping.
Only add live data sources that are dead-simple (one REST call, free, reliable).

### Live Data Tiers

**TIER 0 — COMPUTED (zero live data, infinite features)**
All of these come from OHLCV + date + ephemeris math:
- Time-of-day sessions, rate-of-change, sequential memory
- Price numerology (DR, angel numbers, sacred geometry)
- Calendar anomalies (FOMC/opex dates hardcoded years ahead)
- Entropy, fractal dimension, Hurst exponent
- All moon positions, planetary aspects, transits
- Vedic astrology (tithi, karana, yogas)
- Chinese astrology (year animal/element)
- Harmonic cycles (Gann/Fib from price pivots)
- BTC natal chart transits
- Calendar harmonics (sin/cos encoding)
- Autoregressive (model's own previous predictions)
- Cross-crypto correlation regime
- 4-tier binarization, DOY windows, regime-aware DOY
- ALL cross features (multiplying existing features)

**TIER 1 — ALREADY RUNNING (no new work)**
| Source | Streamer | Frequency |
|--------|----------|-----------|
| BTC/XRP OHLCV | Bitget API | Every bar |
| Fear & Greed | alternative.me | Daily |
| Space weather (Kp, solar) | NOAA API | Daily |
| Macro (VIX, DXY, yields) | macro_streamer.py | Daily |
| Tweets | scrape_tw.py | Continuous |
| News | news_streamer.py | Continuous |
| Sports | sports_streamer.py | Daily |
| Google Trends | google_trends.db | Daily |

**TIER 2 — EASY TO ADD (one GET call each, free, no auth hassles)**
| Source | API | Auth | Frequency | Lines of Code |
|--------|-----|------|-----------|---------------|
| DeFi TVL | DefiLlama `/protocols` | None | Daily | ~10 lines |
| BTC dominance | CoinGecko `/global` | None | Daily | ~10 lines |
| Mining hash rate + difficulty | Blockchain.com `/stats` | None | Daily | ~10 lines |
| COT positioning | CFTC weekly CSV download | None | Weekly | ~20 lines |
| Multi-crypto OHLCV (SOL, DOGE, etc.) | CoinGecko or CryptoCompare | Free key | Daily | ~15 lines |

**5 new endpoints. ~65 lines of code total. Add to existing streamer cron.**

**TIER 3 — TRAINING-ONLY (download historical once, never maintain live)**
These have great historical data but painful live APIs. Use for training validation only.
NOT in the production model — trained in a separate validation model (Tier 2 model).
| Source | Historical Source | Purpose |
|--------|------------------|---------|
| On-chain (MVRV, SOPR, NVT) | CoinMetrics community CSV (free) | Validate patterns |
| Stablecoin flows | CoinMetrics community CSV | Validate patterns |
| Exchange flows | CoinMetrics community CSV | Validate patterns |

**Tier 2 validation model approach:**
- Download historical on-chain CSVs once (CoinMetrics community = free)
- Train a SEPARATE model that includes these features
- Compare feature importance between Tier 1 model and Tier 2 model
- If both models agree on which patterns matter → extra confidence
- Only Tier 1 model goes to live_trader.py (no on-chain dependency)

**DROP ENTIRELY:**
- Mempool data (needs Bitcoin node, fragile)
- Real-time exchange flows (Glassnode = $40/mo, not worth it)
- Any source requiring paid subscriptions or scraping

### Trading Any Crypto
The unified model learns universal patterns from 31 assets.
At inference time for ANY crypto (SOL, DOGE, ADA, whatever):
1. Fetch that crypto's OHLCV from Bitget/CoinGecko (one API call)
2. Compute TA features from OHLCV (same pipeline)
3. Compute ticker gematria (pure math from ticker name)
4. Calendar/astro/space weather = identical (universal)
5. Feed to unified model → prediction
**No new live data needed. Just OHLCV for the target crypto.**

---

## PART 6: NEW SCRIPTS TO CREATE

### 1. `download_multi_asset.py` (NEW)
Downloads OHLCV for all 31 tickers from free APIs.
```python
TICKERS_STOCK = ['SPY','QQQ','TSLA','MSTR','ARKK','GLD','SLV','USO','UNG',
                 'WEAT','CORN','DBA','EWJ','FXI','EWG','XLE','XLF']
TICKERS_CRYPTO = ['BTC','ETH','XRP','LTC','SOL','DOGE','ADA','BNB',
                  'AVAX','LINK','DOT','MATIC','UNI','AAVE']
INVERSE_SIGNALS = ['UUP','TLT','FXY']  # features only, not training assets
```
- Yahoo Finance for stocks (yfinance library, free)
- CryptoCompare for extended crypto history (free API key)
- Bitget for recent crypto (already have key)
- FRED for macro (VIX, DXY, yields — free)
- Saves to `multi_asset_prices.db`

### 2. `v2_cross_generator.py` (NEW)
Generates ALL V2 cross features in sparse format:
- All 11 new crossing layers (time-of-day, acceleration, memory, price numerology,
  correlation, calendar anomalies, entropy, moon position, aspects, harmonics,
  sacred geometry)
- Plus: astro×TA, multi-astro, multi-TA, esoteric×TA, space weather×all
- DOY ±2 windows, 4-tier binarization, regime-aware DOY (3x)
- Confluence/resonance scores
- BTC natal chart transits, Vedic deep, Chinese cycles
- Calendar harmonics (sin/cos continuous)
- Autoregressive features (previous model predictions)
- Uses CuPy GPU for sparse matrix multiplication
- Saves as scipy.sparse CSC .npz per asset per TF

### 3. `v2_multi_asset_trainer.py` (NEW)
Multi-asset training orchestrator — NO FILTERING:
- Load all sparse matrices (2-3M features)
- Per-asset XGBoost CPCV on ALL features
- Unified model on ALL features (all 31 assets combined)
- Per-crypto production models on ALL features + crypto-specific
- Cross-asset feature importance comparison (for ANALYSIS, not filtering)
- Meta-labeling secondary classifier

### 4. `v2_cloud_runner.py` (NEW)
Updated cloud orchestration for vast.ai:
- Upload sparse matrices
- Run full training pipeline (no filter step)
- Download models + importance reports
- Cost tracking

### 5. `v2_easy_streamers.py` (NEW)
Single file with 5 easy live data additions:
```python
def fetch_defi_tvl():       # DefiLlama, 1 GET, no auth
def fetch_btc_dominance():  # CoinGecko, 1 GET, no auth
def fetch_mining_stats():   # Blockchain.com, 1 GET, no auth
def fetch_cot_report():     # CFTC CSV, weekly download
def fetch_multi_crypto():   # CoinGecko, batch OHLCV for all crypto
```
~65 lines total. Called daily by existing cron/scheduler.

### 6. `build_*_features_v2.py` (MODIFY existing)
Update build scripts to:
- Accept any ticker (not just BTC)
- Add ticker_gematria, ticker_numerology features
- Generate ALL V2 crosses
- Save sparse .npz format

### 7. `feature_library_v2.py` (EXPAND from V1)
- All 11 new crossing layers
- 4-tier binarization option
- DOY ±2 windows
- Planetary aspects (50 flags from ephemeris)
- BTC natal chart transits
- Vedic deep (tithi, karana, yogas)
- Chinese astrology (year animal/element, flying stars)
- Sacred geometry (golden ratio, sqrt levels)
- Entropy/fractal (Shannon, Hurst, fractal dimension)
- Harmonic time cycles (Gann squares, Fib extensions)
- Calendar harmonics (sin/cos continuous encoding)
- Autoregressive features
- Keep backward compatible with V1

### 8. `data_access_v2.py` (EXPAND from V1)
- Multi-asset loading from `multi_asset_prices.db`
- Inverse signal loading (UUP, TLT, FXY as features)
- Extended crypto history (14 cryptos)
- Easy streamer data (DeFi TVL, BTC.D, mining, COT)
- Historical on-chain CSVs (CoinMetrics, training-only)

---

## PART 7: IMPLEMENTATION ORDER — SINGLE PHASE

Everything built at once. No waves, no tiers, no cutting.

### Step 1: Data Download
- `download_multi_asset.py` — all 31 tickers + 3 inverse signals
- `data_access_v2.py` — multi-asset loading layer
- `v2_easy_streamers.py` — 5 easy live data endpoints (DeFi TVL, BTC.D, mining, COT, multi-crypto)
- Historical on-chain CSVs from CoinMetrics (training validation)
- Verify data quality across all assets

### Step 2: Feature Library Expansion
- `feature_library_v2.py` — ALL new layers in one build:
  - Time-of-day sessions + kill zones
  - Rate-of-change / acceleration (4 states per TA)
  - Sequential memory (was_X_Nd, streaks)
  - Price numerology (DR, angel numbers, round numbers, master numbers)
  - Correlation regime (BTC/SPY, BTC/GLD rolling corr)
  - Calendar anomalies (FOMC, opex, CPI, halving, Chinese NY, Diwali, Eid)
  - Entropy / fractal (Shannon, Hurst, fractal dimension)
  - Detailed moon position (12 signs, element, modality, 28 mansions)
  - Planetary aspects (50 flags: conjunctions, squares, trines, oppositions)
  - BTC natal chart transits (Jan 3 2009 18:15 UTC natal)
  - Vedic deep (tithi, karana, yogas, dasha periods, Rahu-Ketu axis)
  - Chinese astrology (year animal/element, flying star period, Tong Shu)
  - Sacred geometry (golden ratio extensions, sqrt levels, phi harmonics)
  - Harmonic time cycles (Gann squares, Fib time extensions from pivots)
  - Calendar harmonics (sin/cos continuous encoding for all cycles)
  - Autoregressive (previous model predictions, win/loss streaks)
  - DeFi metrics (TVL trend, DEX volume)
  - Mining data (hash rate trend, difficulty adjustment window)
  - BTC dominance trend
  - COT positioning (large spec, commercial, retail extremes)
  - On-chain valuation (MVRV, SOPR, NVT — training validation model)
  - 4-tier binarization (EXTREME_HIGH/HIGH/LOW/EXTREME_LOW)
  - DOY ±2 day windows (5x more data per DOY)
  - Regime-aware DOY (3x: bull/bear/sideways)
  - Confluence + resonance scores

### Step 3: Cross Generator
- `v2_cross_generator.py` — systematic "everything × everything"
  - Every base feature × every other base feature (sparse multiply)
  - Multi-astro combos (2+ astro firing simultaneously)
  - Multi-TA combos (overbought + squeeze as one feature)
  - Astro × TA, esoteric × TA, space weather × all
  - All new layers × all existing layers
  - CuPy GPU sparse matrix generation
  - Output: scipy.sparse CSC .npz per asset per TF

### Step 4: Build All Assets
- `build_features_v2.py` — unified multi-asset builder
- Build features for all 31 assets × 6 TFs locally on 13900K
- Stagger: 1w+1d+4h parallel, then 1h, then 15m, then 5m
- Multi-asset at daily TF: all 31 tickers
- Multi-asset at intraday: crypto only (stocks have different market hours)
- Save ALL as sparse .npz matrices

### Step 5: Cloud Training
- Rent vast.ai (4× A6000 48GB, ~$2/hr)
- Upload sparse matrices
- `v2_multi_asset_trainer.py`:
  - Per-asset models on ALL 2-3M features (no filter)
  - Unified model on ALL 31 assets combined
  - Per-crypto production models (BTC, XRP, SOL, etc.)
  - Validation model with on-chain features (training-only)
  - Meta-labeling + CPCV + PBO
- Download models + importance reports

### Step 6: Integration
- Update `live_trader.py` for V2 models + multi-crypto
- V2 feature computation at inference (90% math, 5 easy API calls)
- Paper trading on BTC + SOL + DOGE + ADA

---

## PART 8: VAST.AI COST ESTIMATE

| Phase | Machine | Hours | Cost/hr | Total |
|-------|---------|-------|---------|-------|
| Feature build | LOCAL (13900K) | 10 | $0 | $0 |
| Upload to vast.ai | — | 1 | — | $0 |
| Train (no filter, all 2-3M) | 4× A6000 48GB | 12 | ~$2.00 | ~$24 |
| LSTM (local) | LOCAL (3090) | 4 | $0 | $0 |
| **TOTAL** | | **~27 hrs** | | **~$24** |

Compare to V1: $6.50 per training run on 4× H100.
V2 is ~4x the cost but trains ALL 2-3M features across 31 assets with 44x more data.
Can trade ANY crypto with no additional live data beyond OHLCV.

---

## PART 9: EXPECTED OUTCOMES

| Metric | V1 | V2 Target |
|--------|-----|-----------|
| Features available to model | ~3,000 | ~2-3M (ALL, no filter) |
| Features model actually uses | ~3,000 | XGBoost self-selects via splits |
| Training assets | 1 (BTC) | 31 tickers |
| Training bars (daily) | 2,500 | ~110,000 |
| Training bars (1H) | 21,000 | ~500,000+ |
| Tradeable cryptos | BTC + XRP | ANY crypto with OHLCV |
| New live data sources | 0 | 5 (all one-liner REST calls) |
| Live data maintenance burden | Tweets + news + sports | Same + 5 trivial GETs |
| Universal pattern validation | None | Cross-31-asset importance analysis |
| Win rate | 53-56% | 60-66% |
| Sharpe ratio | 0.8 | 1.5-2.5 |
| Overfitting risk | High (small data) | Low (massive multi-asset data) |
| Missed rare signals | Unknown | Zero (nothing filtered out) |

### The Key Edge
The model sees ALL 2-3M features. It decides what matters via tree splits.
A 3-way cross that fires 2 times in BTC might fire 40 times across 31 assets —
the unified model has the data to learn it. Nothing is pre-excluded.
If DOY 73 + full moon + oversold = BUY across SPY (30 years), GLD (20 years),
and BTC (14 years) — **that's not noise, that's the matrix.**
And if eclipse_x_angel_777_x_doy_93 only fires twice but both are 10% winners —
the model sees it. No filter threw it away.

**90% of features are pure math from date + price. No live data needed.**
**Trade ANY crypto by just fetching its OHLCV. The matrix is universal.**
