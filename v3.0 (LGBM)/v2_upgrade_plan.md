# V2 Upgrade Plan — Everything × Everything with Multi-Way Interactions

## Current System (V1) — What We Have
- DOY × single context = 2-way interaction (dx_ features, ~135K)
- ex_ = astro × regime (curated, ~200-400)
- tx_ = events × bull/bear/VWAP/range (~3K)
- px_ = curated numerology × TA (~601)
- cross_ = moon × sentiment, price DR × tweet DR (~35)
- XGBoost max_depth=4 learns up to 4-way implicitly via tree splits
- 80/20 binarization = coarse

## What's Missing
- No systematic astro × TA crosses (only curated ex_ combos)
- No multi-astro combinations (full moon + mercury retro as ONE signal)
- No multi-TA combinations (overbought + squeeze + divergence as ONE signal)
- No 3-way or 4-way pre-computed features
- DOY fires 1 day/year = only 7 samples per DOY
- No regime-aware DOY crosses

---

## V2 UPGRADES

### 1. Systematic Astro × TA Crosses (EVERYTHING × EVERYTHING)
Cross EVERY astro binary with EVERY binarized TA indicator:

**Astro signals (~20):**
- is_full_moon, is_new_moon, moon_phase_HIGH/LOW
- mercury_retrograde, venus_retrograde
- eclipse_window, eclipse_solar, eclipse_lunar
- key_nakshatra, nakshatra_nature (favorable/unfavorable)
- void_of_course_moon
- planetary_hour_match, planetary_day_resonance
- bazi_day_clash, tzolkin_portal_day
- equinox_window, solstice_window

**TA contexts (~100 binarized):**
- rsi_14_HIGH/LOW, rsi_14_EXTREME_HIGH/EXTREME_LOW
- macd_line_HIGH/LOW, macd_histogram_HIGH/LOW
- bb_pctb_20_HIGH/LOW, bb_squeeze_20
- atr_14_pct_HIGH/LOW
- stoch_k_14_HIGH/LOW
- volume_ratio_HIGH/LOW, volume_spike
- ema50_rising, ema50_declining
- obv_HIGH/LOW, cvd_slope_HIGH/LOW
- funding_rate_HIGH/LOW
- fear_greed_HIGH/LOW
- ichimoku_above_cloud, ichimoku_below_cloud
- supertrend_bull, supertrend_bear
- vwap_above, vwap_below

**Result: ~20 × ~100 = ~2,000 astro×TA features**
```
ax_full_moon_x_rsi_HIGH        ← full moon + overbought
ax_full_moon_x_rsi_LOW         ← full moon + oversold (opposite signal!)
ax_mercury_retro_x_bb_squeeze  ← retro + squeeze = breakout coming
ax_eclipse_x_volume_spike      ← eclipse + volume explosion
ax_new_moon_x_fear_greed_LOW   ← new cycle + extreme fear = buy
ax_key_nakshatra_x_ema50_rising ← favorable star + uptrend
```

### 2. Multi-Astro Combinations (astro stacking)
Pre-compute combinations of 2+ astro events firing simultaneously:

```
astro_2x_moon_retro = is_full_moon * mercury_retrograde
astro_2x_eclipse_nakshatra = eclipse_window * key_nakshatra
astro_2x_moon_void = is_full_moon * void_of_course_moon
astro_2x_retro_eclipse = mercury_retrograde * eclipse_window
astro_2x_newmoon_equinox = is_new_moon * equinox_window
```

Then cross THESE with TA:
```
ax2_moon_retro_x_rsi_HIGH     ← full moon AND mercury retro AND overbought
ax2_eclipse_nakshatra_x_bear  ← eclipse AND key nakshatra AND bear regime
```

**~50 astro pairs × ~100 TA = ~5,000 features**

### 3. Multi-TA Combinations (TA confluence stacking)
Pre-compute combinations of 2+ TA conditions:

```
ta_2x_overbought_squeeze = rsi_HIGH * bb_squeeze
ta_2x_oversold_volume = rsi_LOW * volume_spike
ta_2x_trend_momentum = ema50_rising * macd_line_HIGH
ta_2x_bearish_divergence = rsi_HIGH * cvd_slope_LOW  ← price up but selling
ta_2x_capitulation = fear_greed_LOW * volume_spike * atr_HIGH
ta_2x_euphoria = fear_greed_HIGH * funding_rate_HIGH * rsi_HIGH
```

Then cross with DOY and astro:
```
dx_93_x_ta_capitulation        ← destruction day + capitulation setup
ax_full_moon_x_ta_euphoria     ← full moon + euphoria = top signal
```

**~30 curated TA combos × (365 DOY + 20 astro) = ~11,500 features**

### 4. Esoteric × TA Crosses (gematria/numerology × market state)
Cross gematria/numerology signals with TA:

```
ex2_gem_caution_x_rsi_HIGH     ← caution gematria + overbought = sell
ex2_gem_caution_x_rsi_LOW      ← caution gematria + oversold = already priced in
ex2_gem_pump_x_ema50_rising    ← pump gematria + uptrend = add to long
ex2_price_dr_9_x_atr_HIGH     ← completion number + high vol = trend ending
ex2_date_dr_match_x_volume_spike ← date/price DR match + volume = significant bar
ex2_master_number_x_bb_squeeze  ← master number day + squeeze = explosive move
```

**~15 esoteric flags × ~100 TA = ~1,500 features**

### 5. Space Weather × Everything
Cross space weather with TA, astro, and DOY:

```
sw_kp_storm_x_rsi_HIGH        ← geomagnetic storm + overbought
sw_kp_storm_x_full_moon       ← storm + full moon (double cosmic energy)
sw_solar_flux_HIGH_x_atr_HIGH ← high solar activity + high volatility
dx_93_x_kp_storm              ← destruction day + geomagnetic storm
sw_kp_storm_x_gem_caution     ← storm + caution gematria
```

**~10 space weather × ~120 (TA+astro) = ~1,200 features**

### 6. Sentiment × Astro × TA (triple cross)
Pre-compute sentiment state then cross with astro and TA:

```
sx3_bulltweet_x_fullmoon_x_oversold   ← bullish tweets + full moon + oversold = strong buy
sx3_cautionnews_x_retro_x_overbought  ← caution headlines + retro + overbought = strong sell
sx3_beartweet_x_newmoon_x_squeeze     ← bearish tweets + new cycle + squeeze = reversal up
```

**~100 curated triple combos**

### 7. DOY Windows (±2 days) — Data Multiplier
Replace exact DOY flags with windows:
```
doy_73_w2 = 1 if 71 <= current_doy <= 75 else 0
```

Increases samples from 7 to 35 per window (7 years × 5 days).
Cross EVERYTHING above with DOY windows instead of exact DOY.
Same features, 5x more training data per feature.

### 8. Finer Binarization: 4-Tier Contexts
Current: HIGH (>80th) and LOW (<20th)
Upgrade to 4 tiers:
```
EXTREME_HIGH (>95th)  ← rare, strongest signal
HIGH (>75th)          ← common overbought
LOW (<25th)           ← common oversold
EXTREME_LOW (<5th)    ← rare, strongest signal
```

Doubles all context counts. "Mildly overbought on pump day" vs "extremely overbought on pump day"
are different signals with different directional implications.

### 9. Astro Confluence Score (compressed multi-astro)
Single feature counting how many astro signals fire simultaneously:
```
astro_confluence = is_full_moon + mercury_retrograde + eclipse_window +
                   key_nakshatra + planetary_hour_match + void_of_course +
                   equinox_window + bazi_clash
```

Range 0-8. Binarized: confluence_HIGH (≥3), confluence_EXTREME (≥5)
Cross with DOY and TA:
```
dx_73_x_astro_confluence_HIGH   ← wisdom day + 3+ astro events = major day
ax_confluence_EXTREME_x_rsi_LOW ← 5+ astro events + oversold = cosmic bottom
```

### 10. Esoteric Resonance Score (compressed multi-esoteric)
```
resonance = tweet_gem_caution + news_gem_caution + price_dr_match +
            date_dr_9 + master_number_active + btc_energy_date +
            shemitah_year + fibonacci_day + palindrome_date
```

Range 0-9. Cross with everything.

### 11. Regime-Aware DOY Crosses (3x feature explosion)
Separate DOY crosses per HMM regime:
```
dx_73_BULL_x_rsi_HIGH    ← day 73 in bull regime + overbought
dx_73_BEAR_x_rsi_HIGH    ← day 73 in bear regime + overbought
dx_73_SIDEWAYS_x_rsi_HIGH ← day 73 in chop + overbought
```

Same DOY, same TA, but regime changes the meaning entirely.
3x current dx_ feature count.

### 12. Temporal Context Windows (what happened recently)
```
dx_73_x_was_full_moon_3d       ← day 73 + full moon within last 3 days
dx_73_x_kp_storm_7d            ← day 73 + geomagnetic storm in last week
dx_73_x_eclipse_14d            ← day 73 + eclipse within 2 weeks
dx_73_x_tweet_caution_streak_3 ← day 73 + 3 consecutive caution tweets recently
ax_retro_x_was_squeeze_5d      ← mercury retro + BB was squeezing recently
```

Captures "energy echoes" — events that happened recently still affecting price.

### 13. Cross-Timeframe Conditional Signals
Feed higher-TF model predictions as features to lower-TF:
```
htf_1d_prob_long = 0.72
htf_4h_prob_long = 0.65
dx_73_x_htf_aligned_bull    ← day 73 + both daily and 4H models say long
ax_full_moon_x_htf_conflict ← full moon + daily says long but 4H says short
```

Multi-TF confluence or divergence as explicit features.

### 14. Market State Embeddings (autoencoder)
Train autoencoder on 50 TA features → 4-8 latent dimensions.
Each dimension = learned market regime (not hand-crafted).
Cross with DOY and astro:
```
dx_73_x_state_cluster_3  ← day 73 + market in learned state 3
ax_retro_x_state_cluster_1 ← retro + market in state 1
```

### 15. Interaction Feature Networks (deep feature synthesis)
Small MLP: [DOY, astro, TA, esoteric, sentiment, space_weather] → 16 interaction embeddings.
Neural net learns which 3-4-5 way combos matter. Feed embeddings to XGBoost.
Best of both worlds: neural net discovers interactions, XGBoost makes decisions.

### 16. Sports/Horse Racing × Market State
```
sx_sports_upset_x_overbought   ← sports chaos energy + overbought = sell
sx_horse_gem_match_x_pump_date ← horse name gematria match + pump day
sx_upset_streak_x_fear_HIGH    ← multiple upsets + elevated fear = volatility
```

### 17. On-Chain × Astro (whale behavior during cosmic events)
```
oc_whale_x_full_moon     ← whale activity during full moon
oc_funding_x_eclipse     ← funding rates during eclipse windows
oc_liquidation_x_retro   ← liquidation cascades during mercury retro
```

### 18. News Source Gematria × Price DR
```
nx_source_dr_match_price ← news publisher gematria DR matches price DR
nx_headline_caution_x_oversold ← caution headline + oversold = buy the fear
nx_headline_pump_x_overbought  ← pump headline + overbought = distribution
```

---

## Feature Count Summary

| Cross Type | Formula | Est. Features |
|------------|---------|---------------|
| dx_ DOY × context (current) | 365 × 370 | ~135,000 |
| ax_ astro × TA (NEW) | 20 × 100 | ~2,000 |
| ax2_ multi-astro × TA (NEW) | 50 × 100 | ~5,000 |
| ta2_ multi-TA × DOY+astro (NEW) | 30 × 385 | ~11,500 |
| ex2_ esoteric × TA (NEW) | 15 × 100 | ~1,500 |
| sw_ space weather × all (NEW) | 10 × 120 | ~1,200 |
| sx3_ sentiment triple (NEW) | curated | ~100 |
| Regime-aware DOY (NEW) | 3 × 135K | ~405,000 |
| DOY window (replaces exact) | ±2 days | same count, 5x data |
| 4-tier binarization | 2x contexts | doubles all crosses |
| Confluence scores | compressed | ~50 |
| Temporal windows | curated | ~500 |
| Cross-TF signals | 6 TF × probs | ~50 |
| Sports/horse × market | curated | ~200 |
| On-chain × astro | curated | ~300 |
| News × price DR | curated | ~200 |
| **V2 Total (without regime 3x)** | | **~157,600** |
| **V2 Total (with regime 3x DOY)** | | **~562,600** |

## Priority Order

### Tier 1 — Highest Impact, Easiest (do first)
1. **DOY windows ±2 days** — 5x more data per DOY, biggest single improvement
2. **Systematic astro × TA crosses** — 2K features, fills the biggest gap
3. **Astro + esoteric confluence scores** — compress multi-signal into clean feature
4. **Multi-astro combinations** — full moon + retro as explicit feature

### Tier 2 — High Impact, Moderate Effort
5. **Finer 4-tier binarization** — more resolution, doubles contexts
6. **Multi-TA combinations** — overbought+squeeze as one feature
7. **Esoteric × TA crosses** — gematria × market state
8. **Space weather × everything** — Kp storms × TA × astro
9. **Regime-aware DOY crosses** — 3x features but regime-specific

### Tier 3 — Advanced, Requires New Architecture
10. **Temporal context windows** — what happened recently
11. **Sentiment triple crosses** — sentiment × astro × TA
12. **Cross-TF conditional signals** — multi-TF model confluence
13. **Sports/horse × market state**
14. **On-chain × astro**
15. **News × price DR**

### Tier 4 — Research Level
16. **Market state embeddings** — autoencoder
17. **Interaction feature networks** — MLP + XGBoost stacking

## Implementation Estimate
- Tier 1 (items 1-4): ~3-4 hours code, same pipeline
- Tier 2 (items 5-9): ~6-8 hours code, 2-3x feature count
- Tier 3 (items 10-15): ~1-2 days code
- Tier 4 (items 16-17): ~2-3 days, new model architectures

## Expected Impact
- V1 (current): ~53-56% win rate, Sharpe ~0.8
- V2 Tier 1: ~56-59% win rate, Sharpe ~1.0-1.3
- V2 Tier 1+2: ~58-62% win rate, Sharpe ~1.3-1.7
- V2 Tier 1+2+3: ~60-64% win rate, Sharpe ~1.5-2.0
- V2 Full: ~62-66% win rate, Sharpe ~1.8-2.5

### 19. Macro Environment × Everything (market conditions layer)
The macro environment changes how EVERY signal behaves.

**Macro conditions (~15 binarized):**
- vix_HIGH / vix_LOW (fear vs complacency)
- dxy_rising / dxy_declining (dollar strength)
- spx_above_sma200 / spx_below_sma200 (risk-on vs risk-off)
- yield_curve_inverted (recession signal)
- btc_dominance_HIGH / btc_dominance_LOW (alt season vs BTC season)
- funding_rate_HIGH / funding_rate_LOW (leverage)
- open_interest_HIGH / open_interest_LOW (positioning)
- crypto_fear_greed_EXTREME_FEAR / EXTREME_GREED

**Cross with everything:**
```
mx_vix_HIGH_x_full_moon_x_oversold    ← fear environment + full moon + oversold = capitulation buy
mx_vix_LOW_x_full_moon_x_overbought   ← complacent + full moon + overbought = blow-off top
mx_dxy_rising_x_gem_caution           ← strong dollar + caution gematria = crypto dump
mx_risk_off_x_kp_storm               ← risk-off macro + geomagnetic storm = accelerated sell
dx_73_x_vix_HIGH                      ← wisdom day + high fear environment
dx_93_x_funding_HIGH_x_overbought    ← destruction day + overleveraged + overbought = liquidation cascade
ax_mercury_retro_x_yield_inverted    ← communication chaos + recession signal = confusion
```

**Macro × regime × DOY triple:**
```
mx3_vix_HIGH_x_BEAR_x_doy_93    ← fear + bear market + destruction day = capitulation bottom?
mx3_vix_LOW_x_BULL_x_doy_27     ← complacent + bull + pump day = blow-off continuation
mx3_risk_off_x_eclipse_x_doy_322 ← risk-off + eclipse + S&B day = major event
```

**~15 macro × ~120 (TA+astro) = ~1,800 base features**
**+ curated macro triples = ~500 features**

### 20. Volatility Regime × Everything
Not just HIGH/LOW — the CHANGE in volatility matters:

```
vol_expanding = atr increasing over 10 bars
vol_contracting = atr decreasing over 10 bars (squeeze forming)
vol_shock = atr jumped >2x in 3 bars
vol_dead = atr at 30-day low
```

Cross with signals:
```
vx_expanding_x_full_moon          ← vol expanding + full moon = climax
vx_contracting_x_doy_73          ← squeeze + wisdom day = explosive breakout coming
vx_shock_x_gem_caution           ← vol shock + caution gematria = panic
vx_dead_x_new_moon               ← dead vol + new cycle = accumulation
dx_93_x_vol_shock_x_bear         ← destruction + vol shock + bear = crash
```

**~4 vol states × ~385 (DOY+astro+TA) = ~1,540 features**

### 21. Order Flow × Cosmic Events
Combine microstructure with esoteric timing:

```
of_taker_buy_x_new_moon          ← aggressive buying + new cycle
of_taker_sell_x_full_moon        ← aggressive selling + culmination
of_liquidation_cascade_x_eclipse ← forced selling + eclipse window
of_whale_accumulation_x_doy_27   ← whale buying + pump day
of_funding_flip_x_mercury_retro  ← funding rate flipped + communication chaos
of_oi_spike_x_astro_confluence   ← positioning + multiple astro events
```

**~10 order flow × ~20 cosmic = ~200 features**

## Updated Feature Count Summary

| Cross Type | Est. Features |
|------------|---------------|
| dx_ DOY × context (V1) | ~135,000 |
| ax_ astro × TA | ~2,000 |
| ax2_ multi-astro × TA | ~5,000 |
| ta2_ multi-TA × DOY+astro | ~11,500 |
| ex2_ esoteric × TA | ~1,500 |
| sw_ space weather × all | ~1,200 |
| sx3_ sentiment triples | ~100 |
| mx_ macro × everything (NEW) | ~1,800 |
| mx3_ macro triples (NEW) | ~500 |
| vx_ volatility regime × all (NEW) | ~1,540 |
| of_ order flow × cosmic (NEW) | ~200 |
| Regime-aware DOY (3x) | ~405,000 |
| 4-tier binarization (2x) | doubles crosses |
| Confluence scores | ~50 |
| Temporal windows | ~500 |
| Cross-TF signals | ~50 |
| Sports/horse × market | ~200 |
| On-chain × astro | ~300 |
| News × price DR | ~200 |
| **V2 TOTAL** | **~566,000+** |

### 22. Multi-Asset Training Data (The Universal Matrix)

The matrix isn't BTC-specific — DOY, astro, gematria, space weather affect ALL markets.
Use stock + older crypto data as TRAINING DATA to teach the model universal calendar/energy patterns.
Only TRADE BTC and XRP. Stocks are the textbook, crypto is the exam.

**Why this works:**
- 7 years of BTC is thin. SPY has 50+ years. AAPL has 40+.
- If DOY 73 + full moon is bullish for SPY AND BTC AND gold across decades — that's REAL
- If it only shows up in BTC 7 years — that might be noise
- More data = XGBoost learns TRUE patterns vs coincidences
- Gematria validation: test which numbers actually predict across 1000s of independent price series

**Assets to add (6 stocks + older crypto):**

Stocks (long history, high quality data):
- **SPY** (S&P 500 ETF) — 30+ years daily, 20+ years intraday. The benchmark.
- **QQQ** (Nasdaq 100) — 25+ years. Tech-heavy, correlates with crypto sentiment.
- **GLD** (Gold ETF) — 20+ years. Alternative store of value, same cosmic cycles.
- **TSLA** — 15+ years. Volatile, meme-driven, Elon gematria connection to crypto.
- **MSTR** — 10+ years (BTC proxy since 2020). Direct BTC correlation.
- **JPM** (JP Morgan) — 30+ years. Banking/institutional money, inverse crypto signal.

Crypto (fill in missing years):
- **BTC** — extend back to 2010-2012 from Bitstamp/Mt.Gox archives (currently 2017+)
- **XRP** — 2013+ from Bitstamp, Poloniex archives
- **ETH** — 2015+ from early exchanges
- **LTC** — 2013+ (oldest alt, longest history)

**Data sources (free):**
- Yahoo Finance API: stocks daily back to 1990s, intraday 5m/15m back ~2 years
- Alpha Vantage: stocks daily/intraday, free tier 5 calls/min
- CryptoCompare: BTC/ETH/XRP/LTC daily back to 2010
- Bitstamp historical: BTC back to 2011
- Binance historical: all crypto pairs back to 2017
- FRED: macro data (VIX, DXY, yields) back to 1990s

**How training works:**

Option A — Unified model:
- Combine all assets into one training set
- Add `ticker_gematria_ordinal`, `ticker_gematria_dr` as features
- DOY × astro × TA features are the same across assets
- Model learns: "DOY 73 + full moon + oversold = BUY regardless of asset"
- Then predict BTC/XRP using the universal patterns

Option B — Transfer learning:
- Pre-train on stocks (50 years × 6 tickers = 300 ticker-years)
- Fine-tune on BTC (7 years) and XRP (10 years)
- Stock patterns transfer to crypto where they're universal (calendar, astro)
- Crypto-specific patterns (funding rate, on-chain) learned in fine-tuning

Option C — Feature validation:
- Train separate models per asset
- Compare feature importance across all models
- Features that rank high across SPY + GLD + BTC + XRP = universal truth
- Features that only rank high for BTC = BTC-specific (still keep, but lower confidence)
- This VALIDATES which DOY × astro × TA patterns are real vs noise

**Recommended: Option C first (validation), then Option A (unified).**

**Training data volume:**
| Asset | Years | Daily bars | 1H bars (est) |
|-------|-------|-----------|---------------|
| SPY | 30 | ~7,500 | ~52,500 |
| QQQ | 25 | ~6,250 | ~43,750 |
| GLD | 20 | ~5,000 | ~35,000 |
| TSLA | 15 | ~3,750 | ~26,250 |
| MSTR | 10 | ~2,500 | ~17,500 |
| JPM | 30 | ~7,500 | ~52,500 |
| BTC (extended) | 15 | ~5,475 | ~56,000 |
| XRP | 12 | ~4,380 | ~43,800 |
| ETH | 10 | ~3,650 | ~36,500 |
| LTC | 12 | ~4,380 | ~43,800 |
| **TOTAL** | | **~50,385** | **~407,600** |

50K daily bars across 10 assets = massive validation dataset for calendar/energy patterns.
Compare: current BTC-only has 2,368 daily bars. This is **21x more data**.

**What stays the same across assets:**
- DOY flags (same calendar)
- Astro features (same sky)
- Space weather (same sun)
- Day-of-week effects (same week)
- Numerology of dates (same dates)
- Macro environment (same VIX, same DXY)

**What changes per asset:**
- Ticker name gematria (SPY=62 ordinal, BTC=25, XRP=62 — SPY and XRP match!)
- TA indicators (computed per asset)
- Volume profile (different per asset)
- Sentiment (different tweets/news per asset)
- On-chain (crypto only)
- Funding/OI (crypto only)

**Implementation:**
1. Download historical data for all 10 assets
2. Compute features using same feature_library.py (it's asset-agnostic)
3. Train separate models per asset → compare feature importance
4. Identify universal calendar/energy patterns
5. Train unified model on all assets
6. Fine-tune for BTC and XRP specifically
7. Deploy on BTC/XRP only

## Key Principle
The edge is NOT any single signal. The edge is the MATRIX — when multiple independent
signal systems (astro, numerology, TA, sentiment, space weather, on-chain) all agree
on direction for a specific calendar day, that's the highest conviction trade.
V2 makes this explicit instead of relying on XGBoost to discover it implicitly.

The multi-asset training data VALIDATES which patterns are universal truth vs BTC-specific
noise. 50 years of SPY data proving that DOY 73 + full moon = bullish is worth more than
7 years of BTC alone. Train on the universe, trade the crypto.
