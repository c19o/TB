# FEATURE GAP ANALYSIS -- DATAPOINT_MATRIX vs Reality
## Generated 2026-03-18

---

## Section 1: BROKEN PIPELINES (in plan, in DB schema, but data is NULL/zero)

These features exist as columns in the feature DBs but contain NO DATA.
The root cause for most: the 1H and 4H feature builders fail to join
source data that IS available. The 1D and 1W builders DO populate
these same features correctly.

### 1A. ASTROLOGY -- ALL NULL in 1H/4H (30 features x 2 DBs = 60 broken slots)

The entire Western + Vedic + BaZi astrology stack is dead in 1H and 4H.
Source: astrology_full.db has 6,285 rows (2009-01-01 to 2026-03-17) -- DATA EXISTS.
1D/1W populate these correctly. 1H/4H builders have a broken date join.

| Feature | DBs Broken | Source | Likely Root Cause | Priority |
|---------|-----------|--------|-------------------|----------|
| west_moon_phase | 1H, 4H | astrology_full.db | Date join mismatch (daily->hourly) | HIGH |
| west_moon_mansion | 1H, 4H | astrology_full.db | Same | HIGH |
| west_mercury_retrograde | 1H, 4H | astrology_full.db | Same (also all-zero in 1D/1W) | MEDIUM |
| west_hard_aspects | 1H, 4H | astrology_full.db | Same | HIGH |
| west_soft_aspects | 1H, 4H | astrology_full.db | Same | HIGH |
| west_planetary_strength | 1H, 4H | astrology_full.db | Same | HIGH |
| west_digital_root | 1H, 4H | astrology_full.db | Same | MEDIUM |
| lunar_phase_sin | 1H, 4H | astrology_full.db | Same | HIGH |
| lunar_phase_cos | 1H, 4H | astrology_full.db | Same | HIGH |
| vedic_nakshatra | 1H, 4H | astrology_full.db | Same | HIGH |
| vedic_tithi | 1H, 4H | astrology_full.db | Same | HIGH |
| vedic_yoga | 1H, 4H | astrology_full.db | Same | MEDIUM |
| vedic_nature_encoded | 1H, 4H | astrology_full.db | Same | MEDIUM |
| vedic_guna_encoded | 1H, 4H | astrology_full.db | Same | MEDIUM |
| vedic_key_nakshatra | 1H, 4H | astrology_full.db | All-zero (binary flag never set) | MEDIUM |
| bazi_day_stem | 1H, 4H | astrology_full.db | Same | MEDIUM |
| bazi_day_branch | 1H, 4H | astrology_full.db | Same | MEDIUM |
| bazi_day_clash_branch | 1H, 4H | astrology_full.db | Same | LOW |
| bazi_day_element_idx | 1H, 4H | astrology_full.db | Same | LOW |
| bazi_btc_friendly | ALL 4 DBs | astrology_full.db | All-zero everywhere, logic bug | LOW |
| mayan_tone | 1H, 4H | computed from date | Same date join issue | MEDIUM |
| mayan_sign_idx | 1H, 4H | computed from date | Same | MEDIUM |
| mayan_tone_1/9/13 | 1H, 4H | computed from date | All-zero (derived from null tone) | MEDIUM |

**FIX:** The 1H/4H builders need their astrology join fixed. They should
extract the DATE portion of each hourly/4h timestamp and join to
astrology_full.db on date. The 1D builder does this correctly -- copy
that pattern.

### 1B. MACRO DATA -- ALL NULL in 1H/4H (48 features x 2 DBs = 96 broken slots)

macro_data.db has 1,816 rows (2019-01-02 to 2026-03-17) -- DATA EXISTS.
1D/1W populate all 48 macro features (100% populated). 1H/4H have the
columns but every single one is NULL.

| Feature Group | Count | DBs Broken | Source | Root Cause | Priority |
|--------------|-------|-----------|--------|------------|----------|
| macro_spx + roc5d/roc20d | 3 | 1H, 4H | macro_data.db | Date join broken | HIGH |
| macro_nasdaq + rocs | 3 | 1H, 4H | macro_data.db | Same | HIGH |
| macro_dxy + rocs | 3 | 1H, 4H | macro_data.db | Same | HIGH |
| macro_vix + rocs | 3 | 1H, 4H | macro_data.db | Same | HIGH |
| macro_gold + rocs | 3 | 1H, 4H | macro_data.db | Same | MEDIUM |
| macro_oil + rocs | 3 | 1H, 4H | macro_data.db | Same | MEDIUM |
| macro_us10y + rocs | 3 | 1H, 4H | macro_data.db | Same | MEDIUM |
| macro_tlt + rocs | 3 | 1H, 4H | macro_data.db | Same | MEDIUM |
| macro_silver + rocs | 3 | 1H, 4H | macro_data.db | Same | LOW |
| macro_russell + rocs | 3 | 1H, 4H | macro_data.db | Same | LOW |
| macro_hyg + rocs | 3 | 1H, 4H | macro_data.db | Same | LOW |
| macro_coin + rocs | 3 | 1H, 4H | macro_data.db | Same | LOW |
| macro_ibit + rocs | 3 | 1H, 4H | macro_data.db | Same | LOW |
| macro_mstr + rocs | 3 | 1H, 4H | macro_data.db | Same | LOW |
| btc_dxy_corr | 1 | 1H, 4H | macro_data.db | Same | HIGH |
| btc_spx_corr | 1 | 1H, 4H | macro_data.db | Same | HIGH |
| btc_gold_corr | 1 | 1H, 4H | macro_data.db | Same | MEDIUM |
| btc_vix_corr | 1 | 1H, 4H | macro_data.db | Same | MEDIUM |

**FIX:** Same root cause as astrology. The 1H/4H builders need to
forward-fill daily macro data onto hourly bars (each hour gets the
most recent daily macro value). 1D builder does this correctly.

### 1C. FEAR & GREED -- ALL NULL in 1H/4H (9 features x 2 DBs = 18 broken slots)

fear_greed.db has 2,963 rows (2018-02-01 to 2026-03-17) -- DATA EXISTS.
1D has 99.96% populated. 1H/4H have all NULLs.

| Feature | DBs Broken | Priority |
|---------|-----------|----------|
| fear_greed | 1H, 4H | HIGH |
| fear_greed_lag* (4 lags) | 1H, 4H | HIGH |
| fg_roc | 1H, 4H | HIGH |
| fg_vs_price_div | 1H, 4H | HIGH |
| fg_extreme_fear | 1H, 4H | HIGH (all-zero, derived from null) |
| fg_extreme_greed | 1H, 4H | HIGH (all-zero, derived from null) |
| fg_x_moon_phase | 1H, 4H | HIGH (cross-feature, both inputs null) |

**FIX:** Forward-fill daily F&G onto hourly bars.

### 1D. ON-CHAIN DATA -- ALL NULL in 1H/4H (11 features x 2 DBs = 22 broken slots)

onchain_data.db: blockchain_data has 6,162 rows (2009-01-03 to 2026-03-16).
1D populates hash_rate, difficulty, n_transactions etc at 99%+.
1H/4H have columns but all NULL or all-zero.

| Feature | DBs Broken | Priority |
|---------|-----------|----------|
| onchain_hash_rate | 1H, 4H | HIGH |
| onchain_hash_rate_roc | 1H, 4H | HIGH |
| onchain_hash_rate_capitulation | 1H, 4H | HIGH (all-zero, derived from null) |
| onchain_difficulty | 1H, 4H | MEDIUM |
| onchain_n_transactions | 1H, 4H | MEDIUM |
| onchain_miners_revenue | 1H, 4H | MEDIUM |
| onchain_mempool_size | 1H, 4H | LOW |
| onchain_mempool | 1H, 4H | LOW (all-zero) |
| onchain_block_dr | 1H, 4H | LOW (all-zero) |
| onchain_fg / onchain_fg_dr | 1H, 4H | MEDIUM (all-zero, never computed) |

**FIX:** Forward-fill daily blockchain_data onto hourly bars.

### 1E. FUNDING RATE -- ALL NULL in 1H/4H

onchain_data.db live table has only 24 rows (same day only).
1D has 74.5% populated (since funding rate started ~2020).
1H/4H have the column but it is NULL everywhere.

| Feature | DBs Broken | Priority |
|---------|-----------|----------|
| funding_rate | 1H, 4H | HIGH |
| funding_rate_high | 1H, 4H | HIGH (all-zero, derived from null) |
| funding_rate_neg | 1H, 4H | HIGH (all-zero, derived from null) |

**FIX:** Need historical funding rate data source. crypto_streamer.py
should download it. Currently only 24 rows in onchain_data table.

### 1F. GOOGLE TRENDS -- ALL NULL in 1H/4H

1D has gtrends 99.7% populated. 1H/4H have NULL.

| Feature | DBs Broken | Priority |
|---------|-----------|----------|
| gtrends_interest | 1H, 4H | MEDIUM |
| gtrends_interest_high | 1H, 4H | MEDIUM (all-zero, derived from null) |

**FIX:** Forward-fill daily trends onto hourly bars.

### 1G. ARABIC LOTS -- ALL NULL in 1H/4H/1D (8 features)

arabic_lot_commerce/increase/catastrophe/treachery: all NULL.
The moon conjunction flags are all-zero (derived from null lots).
These are computed from planetary positions but the computation
is apparently never running.

| Feature | DBs Broken | Priority |
|---------|-----------|----------|
| arabic_lot_commerce | 1H, 4H | MEDIUM |
| arabic_lot_increase | 1H, 4H | MEDIUM |
| arabic_lot_catastrophe | 1H, 4H | MEDIUM |
| arabic_lot_treachery | 1H, 4H | MEDIUM |
| arabic_lot_*_moon_conj (x4) | 1H, 4H | MEDIUM (all-zero) |

**FIX:** Requires ephemeris computation (swisseph). The astro_engine
needs to compute these from planetary longitudes. Not just a join fix.

### 1H. MARKET MICROSTRUCTURE -- ALL NULL in 1H (5 features)

| Feature | DBs Broken | Priority |
|---------|-----------|----------|
| trades | 1H | MEDIUM |
| quote_volume | 1H | MEDIUM |
| taker_buy_volume | 1H | MEDIUM |
| taker_buy_quote | 1H | MEDIUM |
| taker_buy_ratio | 1H | MEDIUM |

These come from Binance OHLCV data. btc_prices.db has 3.8M rows but
the 1H builder may not be pulling all columns from the source candles.

**FIX:** Check if btc_prices.db ohlcv table has these columns and if
the 1H builder's SELECT query includes them.

### 1I. OTHER BROKEN FEATURES

| Feature | DBs Broken | Root Cause | Priority |
|---------|-----------|------------|----------|
| psi | 1H, 4H | Never implemented (unknown metric) | LOW |
| digital_root_genesis | 1H, 4H | Computation not implemented | LOW |
| news_count_today | 1H, 4H | All-zero, aggregation broken | MEDIUM |
| news_sentiment_today | 1H, 4H | All-zero, aggregation broken | MEDIUM |
| tweets_today | 1H, 4H | All-zero, aggregation broken | MEDIUM |
| gold_tweet_today | 1H, 4H | All-zero, aggregation broken | LOW |
| red_tweet_today | 1H, 4H | All-zero, aggregation broken | LOW |
| misdirection | 1H, 4H | All-zero, never fires | LOW |
| cross_moon_x_news_caution | 1H, 4H | All-zero (moon is null) | MEDIUM |
| cross_moon_x_sport_upset | 1H, 4H | All-zero (moon is null) | LOW |

### BROKEN PIPELINE SUMMARY

| Category | Broken Features | DBs Affected | Source Data Exists? |
|----------|----------------|-------------|---------------------|
| Astrology (Western+Vedic+BaZi+Mayan) | 23 | 1H, 4H | YES (6,285 rows) |
| Macro Data | 48 | 1H, 4H | YES (1,816 rows) |
| Fear & Greed | 9 | 1H, 4H | YES (2,963 rows) |
| On-Chain | 11 | 1H, 4H | YES (6,162 rows) |
| Funding Rate | 3 | 1H, 4H | PARTIAL (24 live rows only) |
| Google Trends | 2 | 1H, 4H | YES (via 1D) |
| Arabic Lots | 8 | 1H, 4H | NO (computation needed) |
| Market Microstructure | 5 | 1H | MAYBE (check btc_prices.db) |
| Other | 10 | 1H, 4H | MIXED |
| **TOTAL** | **~119 unique** | | |

**THE ROOT CAUSE IS SINGULAR:** The 1H and 4H feature builders do
not forward-fill daily source data onto sub-daily bars. The 1D and
1W builders work correctly because they operate at daily+ resolution
where the join is trivial. Fixing the date-to-hourly forward-fill
in build_1h_features.py and build_4h_features.py would instantly
populate ~100 of these 119 broken features.

---

## Section 2: NOT YET BUILT (in DATAPOINT_MATRIX plan, NOT in any DB)

These features are described in the DATAPOINT_MATRIX.md plan but have
no corresponding column in any feature database.

### 2A. GEMATRIA EXPANSION (~150 missing features)

The plan calls for 6 gematria methods on every text source. Currently
only a handful are implemented:

**What EXISTS:**
- headline_gem_ord_mean, headline_gem_dr_mode (1H/4H)
- caution_gematria (1D/1H/4H)
- w_news_gemgem_* and w_tw_gemgem_* (11 each, 1W only)
- horse_winner_gem_dr_mode, sport_winner_gem_dr_mode (all DBs, but all-zero)

**What is MISSING (from plan):**
- gem_ord_tweet, gem_rev_tweet, gem_red_tweet, gem_eng_tweet, gem_jew_tweet, gem_sat_tweet
- gem_ord_user, gem_rev_user, etc. (tweet username x 6 methods)
- gem_ord_displayname x 6
- gem_ord_hashtags x 6
- gem_ord_headline x 6 (only ordinal mean exists)
- gem_ord_source x 6 (news source name)
- gem_ord_author x 6 (news author)
- gem_ord_team_home/away/winner/loser x 6 (sports teams)
- gem_ord_venue x 6 (stadium names)
- gem_ord_mvp x 6
- gem_ord_horse_winner/jockey/trainer/race/track x 6 each
- gem_ord_reddit x 6
- gem_ord_fomc/fed_quote/cpi_title x 6
- DR of each gematria value (~130 features)
- Gematria MATCH features (~30 features: gem_match_tweet_headline, etc.)

**Estimate:** ~150-200 features not yet built.
**Effort:** MEDIUM. Universal gematria engine exists (feature_library.py).
Need to apply it to all text sources in the builders.
**Blocker:** tweets.db only has data to Sept 2021 (10,967 rows). Need
tweet_streamer to be running continuously. Sports and horse racing
source data is nearly empty (4 games, 1 horse race).

### 2B. SENTIMENT EXPANSION (~15 missing features)

**What EXISTS:**
- news_sentiment_1h/4h/today, headline_sentiment_mean
- tweet-related sentiment is minimal

**What is MISSING:**
- tweet_sentiment, tweet_caps, tweet_exclamation, tweet_urgency
- reddit_sentiment, reddit_caps
- fomc_sentiment, fed_sentiment
- headline_urgency (only caps exist)
- LLM-based sentiment (llm_sentiment, llm_sarcastic, llm_urgent, llm_context_score)

**Estimate:** ~15 features.
**Effort:** LOW for basic sentiment (regex), MEDIUM for LLM (SDK integration).

### 2C. COLOR ANALYSIS (~10 missing features)

**What EXISTS:**
- gold_tweet_this_1h/4h, gold_tweet_today, red_tweet_this_1h/4h, red_tweet_today

**What is MISSING:**
- tweet_green (green detection in images)
- tweet_dominant_color
- news_gold, news_red (news article images)

**Estimate:** ~6 features.
**Effort:** MEDIUM. Need image download + color analysis pipeline.

### 2D. ASTROLOGY AT EVENT TIMESTAMPS (~25 missing features)

The plan calls for astrology at the MOMENT of each event, not just current candle:

**What is MISSING:**
- tweet_moon_phase, tweet_nakshatra, tweet_planetary_hour, tweet_mercury_retro
- news_moon_phase, news_planetary_hour, news_nakshatra
- game_moon_phase, game_planetary_hour, game_nakshatra
- race_moon_phase, race_planetary_hour, race_nakshatra
- block_planetary_hour, block_nakshatra

**Estimate:** ~25 features.
**Effort:** HIGH. Need to compute astrology for each event timestamp,
not just forward-fill daily values.

### 2E. HORSE RACING (~30 missing features)

sports_results.db has only 1 horse race (2014-10-10). The entire horse
racing feature set is dead:

**What is MISSING:**
- gem_ord_horse_winner x 6, gem_ord_jockey x 6, gem_ord_trainer x 6
- gem_ord_race x 6, gem_ord_track x 6
- dr_horse_position, dr_race_time, dr_horse_odds
- All numerology features for horse data

**Estimate:** ~30 features.
**Effort:** HIGH. Need horse racing data source (sports_streamer.py).
**Blocker:** No historical horse racing data pipeline exists.

### 2F. SPORTS EXPANSION (~15 missing features)

sports_results.db has only 4 games (all from 2026-03-18). Nearly empty.

**What is MISSING:**
- dr_sport_score_home, dr_sport_score_away
- dr_sport_total, dr_sport_diff
- gem_ord_venue x 6
- gem_ord_mvp x 6
- sport championship/playoff game flags
- jersey_numbers_of_scorers numerology

**Estimate:** ~15 features beyond what the columns already have.
**Effort:** HIGH. Need continuous sports data collection first.

### 2G. CROSS-FEATURES (~35 missing features)

The plan lists ~50 cross-features. Only a handful exist:

**What EXISTS:**
- consec_red_x_bb_os, cross_date_price_dr_match
- cross_moon_x_news_caution, cross_moon_x_sport_upset (but all-zero)
- fg_x_moon_phase (null, since both inputs broken in 1H/4H)
- rsi_x_bbpctb

**What is MISSING:**
- moon_x_gold_tweet
- nakshatra_x_red_tweet
- mercury_retro_x_news_sentiment
- eclipse_x_sports_upset
- planetary_hour_x_tweet_time
- voc_moon_x_high_volume
- consec_green_x_caps_tweet
- shmita_x_bear_regime
- day13_x_full_moon
- arabic_lot_conj_x_tweet
- friday13_x_red_tweet
- master_number_x_nakshatra
- fg_extreme_x_moon_phase
- many more combinations

**Estimate:** ~35 features.
**Effort:** LOW once input features are fixed (mostly multiplication/AND logic).

### 2H. REGIME FEATURES (~4 missing features)

**What EXISTS:**
- ema50_declining, ema50_rising, ema50_slope
- wyckoff_phase, current_dd_depth

**What is MISSING:**
- hmm_bull_prob, hmm_bear_prob, hmm_neutral_prob (HMM regime)
- Void of Course Moon (west_voc_moon) -- in 1W schema but not 1H/4H

**Estimate:** ~4 features.
**Effort:** MEDIUM (HMM requires hmmlearn or similar).

### 2I. SOCIAL/CULTURAL (~8 missing features)

**What is MISSING:**
- Wikipedia Bitcoin pageviews
- Reddit post titles + gematria (no reddit data source)
- YouTube video titles gematria
- Telegram channel messages gematria

**Estimate:** ~8 features.
**Effort:** HIGH. Need new data sources/scrapers.

### 2J. HEBREW/CALENDAR EXPANSION (~8 missing features)

**What EXISTS:**
- shemitah_year, jubilee_proximity, sephirah

**What is MISSING:**
- Hebrew date conversion
- Major holiday flags (Yom Kippur, Passover, Rosh Hashanah, etc.)
- Omer counting period
- Date palindromes (partially exists in 1W)
- Ramadan start/end
- Diwali
- Chinese New Year

**Estimate:** ~8 features.
**Effort:** LOW (date-based, no external data needed).

### NOT YET BUILT SUMMARY

| Category | Missing Features | Effort | Blocker |
|----------|-----------------|--------|---------|
| Gematria expansion | ~150-200 | MEDIUM | Text data sources empty |
| Sentiment expansion | ~15 | LOW-MEDIUM | LLM SDK needed |
| Color analysis | ~6 | MEDIUM | Image pipeline needed |
| Astrology at event timestamps | ~25 | HIGH | Per-event computation |
| Horse racing | ~30 | HIGH | No data source |
| Sports expansion | ~15 | HIGH | Nearly empty source DB |
| Cross-features | ~35 | LOW | Fix input features first |
| Regime (HMM) | ~4 | MEDIUM | hmmlearn library |
| Social/cultural | ~8 | HIGH | New scrapers needed |
| Hebrew/calendar | ~8 | LOW | Date computation only |
| **TOTAL** | **~300-350** | | |

---

## Section 3: NEW FROM THIS SESSION (not in original plan, should be added)

### 3A. SPACE WEATHER FEATURES (~25-30 new features)

Confirmed strong volatility signal (r=-0.40, p<0.0001).
space_weather_streamer.py written. space_weather.db created (1 row + 63 flares).

| Feature | Type | Status | Signal |
|---------|------|--------|--------|
| kp_index | Raw | Streamer built, historical not yet downloaded | Vol r=-0.08 |
| sunspot_number | Raw | In kp_history.txt (1932-2026) | Vol r=-0.40 |
| solar_flux_f107 | Raw | In kp_history.txt | Vol r=-0.40 |
| solar_wind_speed | Raw | Streamer built | Vol signal confirmed |
| solar_wind_bz | Raw | Streamer built | Vol signal |
| dst_index | Raw | Not yet in streamer | Vol signal |
| kp_is_storm (Kp>=5) | Flag | Derived | Lower vol on storm days |
| kp_is_severe (Kp>=7) | Flag | Derived | -0.49% mean return |
| solar_flare_x_class | Flag | Streamer built | Vol signal |
| cme_earth_directed | Flag | Streamer built | Vol signal |
| kp_dr, sunspot_dr, solar_flux_dr | Numerology | Not built | NO signal (tested) |
| kp_delta_3d | Rate of change | Not built | Vol p<0.01 |
| solar_flare_decay | Decay | Not built | Planned in Phase 2 |
| geomag_storm_decay | Decay | Not built | Planned in Phase 2 |
| bars_since_flare | Decay | Not built | Planned |
| bars_since_storm | Decay | Not built | Planned |
| noaa_r_scale | NOAA scale | Streamer built | Not tested |
| noaa_s_scale | NOAA scale | Streamer built | Not tested |
| noaa_g_scale | NOAA scale | Streamer built | Not tested |
| kp_x_moon_phase | Cross | Not built | Not tested |
| solar_flare_x_mercury_retro | Cross | Not built | Not tested |
| geomag_storm_x_full_moon | Cross | Not built | Not tested |
| solar_wind_bz_x_nakshatra | Cross | Not built | Not tested |

### 3B. CONFIRMED CYCLE FEATURES (~8 new features)

From book_correlations_*.py analysis. All survive Bonferroni correction.

| Feature | Period | Signal | Status |
|---------|--------|--------|--------|
| schumann_133d_sin | 133 day | Vol r=-0.19, p~1e-20 | Not built |
| schumann_143d_sin | 143 day | Vol r=+0.13, p~9e-11 | Not built |
| schumann_783d_sin | 783 day | Vol r=+0.14, p~3e-12 | Not built |
| chakra_solar_133d | 133 day | Vol r=-0.19, p~1e-20 | Not built |
| chakra_heart_161d | 161 day (golden ratio) | Vol r=+0.09, p~5e-6 | Not built |
| jupiter_365d_cos | 365 day | Vol r=-0.28, p<0.001 | Not built |
| mercury_1216d_sin | 1216 day | Dir r=0.057, p=0.006 | Not built |
| equinox_proximity | Quarterly | Regime shift, p~1e-14 | Not built |

### 3C. COMPOSITE VOL-TO-DIRECTION FEATURES (10 new features)

From DIRECTIONAL_SIGNAL_ANALYSIS.md. These bridge the gap between
volatility prediction and directional trading.

| Feature | Formula Concept | Status |
|---------|----------------|--------|
| esoteric_vol_score | Weighted sum of all vol predictors | Not built |
| vol_regime_transition | esoteric_vol_score - realized_vol_zscore | Not built |
| leverage_x_volatility | sign(funding) * vol_score * abs(funding_zscore) | Not built |
| eclipse_funding_cross | eclipse_window * sign(funding) * abs(funding) | Not built |
| storm_recovery_momentum | exp(-0.5 * days_since_storm) * ema_slope | Not built |
| cycle_confluence_score | Product of normalized cycle values | Not built |
| vol_directional_asymmetry | vol_score * (down_vol/up_vol - 1) | Not built |
| kp_oi_squeeze_indicator | kp_delta * oi_zscore * sign(funding) | Not built |
| seasonal_vol_direction | quarter + vol_score interaction | Not built |
| vol_breakout_direction | sign(close - ema50) * max(0, vol_score - 0.5) | Not built |

### 3D. ADDITIONAL SESSION DISCOVERIES

| Feature | Signal | Status |
|---------|--------|--------|
| post_storm_7d_bullish | +2.2% mean return after Kp>=7 storms (p=0.046) | Not built |
| eclipse_window_vol | Higher vol during eclipse +/-7d (p=0.004) | Not built |
| equinox_solstice_regime | Quarterly regime shifts (p~1e-14) | Not built |

### NEW FEATURES SUMMARY: ~53 new features from this session

---

## Section 4: WORKING AND CONFIRMED (in plan, in DB, actually has data)

### 4A. BTC TECHNICALS (fully populated, all DBs)

| Feature Group | Count | Signal Type |
|--------------|-------|-------------|
| Moving averages (SMA/EMA 5/10/20/50/100/200) | ~24 | Direction |
| close_vs_ema/sma (distance features) | ~12 | Direction (best: r=0.295 for h4) |
| RSI (7/14/21 + lags) | ~12 | Direction (mean reversion) |
| MACD (line/signal/histogram + lags) | ~10 | Direction |
| Bollinger Bands (pctb, width, squeeze) | ~8 | Both |
| Ichimoku (cloud, tenkan, kijun, crosses) | ~8 | Direction |
| Stochastic, Williams %R, ADX, CCI, MFI | ~6 | Direction |
| Supertrend + SAR | ~6 | Direction |
| OBV, CMF, volume features | ~10 | Direction |
| ATR, volatility ratios | ~6 | Volatility |
| Candlestick patterns | ~6 | Direction |
| Return features (1/4/8/24 bar) | ~5 | Direction |
| Donchian, Keltner channels | ~4 | Direction |
| Consensio, signal agreement | ~4 | Direction |

**Total: ~121 features, all working.**

### 4B. MULTI-TF FEATURES (populated in 1H, referring to higher TFs)

| Feature | Count | Signal |
|---------|-------|--------|
| h4_return, h4_rsi14, h4_macd, h4_bb_pctb, h4_ema50_dist, h4_trend, h4_vol_ratio, h4_volatility, h4_atr_pct | 9 | Direction (h4_return is #1, r=0.295) |
| d_return, d_rsi14, d_macd, d_bb_pctb, d_ema50_dist, d_trend, d_volatility | 7 | Direction |
| w_bb_pctb, w_rsi14, w_trend | 3 | Direction/regime |

**Total: 19 features, all working.** These are THE strongest directional features.

### 4C. KNN PATTERN FEATURES (populated where KNN was KEPT)

| Feature | Signal |
|---------|--------|
| knn_direction | Dir r=0.017, p=3.4e-5 |
| knn_avg_return | Dir r=0.009, p=0.024 |
| knn_confidence | Weak |
| knn_best_match_dist | Volatility r=0.060 |
| knn_pattern_std | Volatility r=0.340 |

**Total: 5 features, working in 1H/4H/1W (where KNN was kept).**

### 4D. DATE/TIME FEATURES (all working)

| Feature Group | Count | Signal |
|--------------|-------|--------|
| Hour sin/cos, day sin/cos, month sin/cos, doy sin/cos | 8 | Volatility |
| Session flags (asia, london, ny, overlap) | 5 | Direction (NY r=0.018) |
| Day flags (monday, friday, weekend, month_end, quarter_end) | 6 | Mixed |
| Day of month/week/year | 3 | Volatility |

**Total: ~22 features, all working.**

### 4E. NUMEROLOGY/SACRED (partially working)

| Feature | Signal |
|---------|--------|
| price_dr, digital_root_price, sephirah | Weak (r=0.004) |
| price_dr_6, price_is_master | Weak |
| price_contains_113/213/322/93 | Weak |
| vortex_369 | Weak dir (r=-0.012, p=0.003) |
| date_dr | Weak |
| shemitah_year | Dir r=-0.011, p=0.006 |
| jubilee_proximity | Vol r=-0.106 |
| pump_date | Weak |
| day_13 | Dir r=0.009, p=0.030 |
| fib_13_from_high, fib_21_from_low | Weak dir, strong vol |
| golden_ratio_dist | Dir r=0.016, VOL r=0.288 (strong!) |
| gann_sq9_distance/level | Weak |
| near_fib_13, near_fib_21 | Vol r=0.092 |

**Total: ~20 features working. golden_ratio_dist is the standout (r=0.288 vol).**

### 4F. NEWS FEATURES (partially working)

| Feature | Status |
|---------|--------|
| news_count_1h/4h | Working (sparse) |
| news_sentiment_1h/4h | Working (sparse) |
| headline_sentiment_mean | Working (sparse) |
| headline_caps_any | Working (very sparse) |
| headline_caution_any | Working (very sparse) |
| headline_gem_ord_mean | Working (sparse) |
| headline_gem_dr_mode | Working (sparse) |
| news_article_count | Working (sparse) |
| caution_gematria_1h/4h | Working |
| news_count_today | BROKEN (all-zero in 1H/4H) |

**Total: ~9 working, ~2 broken.**

### 4G. TWEET FEATURES (partially working)

| Feature | Status |
|---------|--------|
| tweets_this_1h/4h | Working (very sparse -- tweets.db ends Sep 2021) |
| gold_tweet_this_1h/4h | Working (very sparse) |
| red_tweet_this_1h/4h | Working (very sparse) |

**Total: ~3 working but extremely sparse.** tweets.db has 10,967 rows
ending September 2021. No recent tweet data.

### 4H. MACRO/ONCHAIN (working in 1D and 1W only)

| Feature Group | Count in 1D | Status in 1H/4H |
|--------------|-------------|------------------|
| macro_* (14 instruments x 3 each) | 42 | ALL BROKEN |
| btc_*_corr (4 correlations) | 4 | ALL BROKEN |
| fear_greed + lags + derived | 9 | ALL BROKEN |
| funding_rate + derived | 3 | ALL BROKEN |
| gtrends_interest + derived | 2 | ALL BROKEN |
| onchain_* (8 features) | 8 | ALL BROKEN |

**Total: ~68 features working in 1D/1W, all broken in 1H/4H.**

### 4I. ASTROLOGY (working in 1D and 1W only)

In 1D (features_complete.db), these are populated:
- west_moon_phase, west_hard/soft_aspects, west_planetary_strength
- vedic_nakshatra, vedic_tithi, vedic_yoga, vedic_nakshatra_idx, vedic_tithi_idx, vedic_yoga_idx
- lunar_phase_sin/cos
- bazi_day_stem, bazi_day_branch

In 1W, the full w_astro_* prefix set (28 features) is populated.

**Total: ~25 features working in 1D, ~28 in 1W, ALL BROKEN in 1H/4H.**

### WORKING FEATURES SUMMARY

| Category | Working in 1H/4H | Working in 1D/1W only |
|----------|------------------|----------------------|
| BTC Technicals | ~121 | -- |
| Multi-TF | 19 | -- |
| KNN | 5 | -- |
| Date/Time | ~22 | -- |
| Numerology/Sacred | ~20 | -- |
| News | ~9 | -- |
| Tweets | ~3 | -- |
| Macro/On-chain | 0 | ~68 |
| Fear & Greed | 0 | ~9 |
| Astrology | 0 | ~25-28 |
| **TOTAL** | **~199** | **~102-105** |

---

## Section 5: UNIFIED FEATURE COUNT

| Metric | Count |
|--------|-------|
| Original DATAPOINT_MATRIX plan | ~790 |
| Columns in 1H DB (largest) | 388 |
| Columns in 4H DB | 374 |
| Columns in 1D DB | 335 |
| Columns in 1W DB | 359 |
| | |
| **Actually populated in 1H/4H** | **~250 (of 377 features)** |
| **Broken/empty in 1H/4H** | **~127** |
| **Actually populated in 1D** | **~311 (of 325 features)** |
| **Broken/empty in 1D** | **~14** |
| **Actually populated in 1W** | **~290 (of 350 features)** |
| **Broken/empty in 1W** | **~60** |
| | |
| Unique features across all DBs (working) | ~310 |
| Broken pipelines (Section 1) | ~119 unique features |
| Not yet built (Section 2) | ~300-350 features |
| New from this session (Section 3) | ~53 features |
| | |
| **Total when complete** | **~790 + 53 = ~843** |
| **Currently working** | **~310 (~37% of target)** |
| **Gap to close** | **~533 features** |

---

## Section 6: PRIORITY FIX ORDER

### TIER 1: FIX THE 1H/4H BUILDER JOIN (1 fix, ~100 features unlocked)

**Priority: CRITICAL**
**Effort: LOW (copy pattern from 1D builder)**
**Impact: Unlocks ~100 features that already have source data**

This single fix addresses:
- All 48 macro features (macro_data.db, 1,816 rows)
- All 9 fear & greed features (fear_greed.db, 2,963 rows)
- All 23 astrology features (astrology_full.db, 6,285 rows)
- All 11 on-chain features (blockchain_data, 6,162 rows)
- 2 Google Trends features
- Cross-features that depend on these inputs (~5-10 more)

The fix: forward-fill daily data onto sub-daily bars. Extract date from
timestamp, join to daily source, ffill. The 1D builder (build_features_complete.py)
already does this correctly.

After fix: 1H goes from ~250 to ~350 populated features.
4H goes from ~237 to ~337 populated features.

### TIER 2: BUILD SPACE WEATHER + CYCLE FEATURES (~33 features)

**Priority: HIGH**
**Effort: MEDIUM**
**Signal: Strongest esoteric signal confirmed (r=-0.40)**

Build order:
1. Download historical Kp/sunspot/solar flux from kp_history.txt into space_weather.db
2. Build kp_index, sunspot_number, solar_flux features in feature_library.py
3. Build cycle features: schumann_133d_sin, chakra_solar_133d, jupiter_365d_cos, etc.
4. Build equinox_proximity, eclipse_window features
5. Add kp_delta_3d rate of change
6. Add to all 6 builders

Data: kp_history.txt already downloaded (1932-2026, daily).

### TIER 3: BUILD COMPOSITE VOL-TO-DIRECTION FEATURES (10 features)

**Priority: HIGH**
**Effort: LOW (once Tier 2 done)**
**Signal: Bridges volatility prediction to directional trading**

Build order:
1. esoteric_vol_score (combine all vol predictors)
2. vol_breakout_direction (simplest: trend x vol_score)
3. vol_regime_transition (predicted vs realized vol gap)
4. leverage_x_volatility (vol x funding direction -- needs funding rate)
5. storm_recovery_momentum (post-storm bullish decay)
6-10. Remaining composite features

### TIER 4: BUILD DECAY FEATURES (~25 features)

**Priority: MEDIUM-HIGH**
**Effort: MEDIUM**
**Signal: Carries sparse signals forward in time**

Add exp(-lambda * bars_since_event) for:
- tweets (gold, red, any, caps)
- news (any, caution)
- sports (upset, game)
- on-chain (whale tx)
- space weather (solar flare, geomag storm, CME)
- astrology (eclipse, retrograde start)

### TIER 5: FIX FUNDING RATE PIPELINE

**Priority: MEDIUM-HIGH**
**Effort: MEDIUM**
**Signal: Critical for composite features (leverage_x_volatility etc.)**

onchain_data.db live table has only 24 rows.
Need: crypto_streamer.py to download historical funding rate data
(Binance API, available since ~2020). This is a blocker for 3
composite features in Tier 3.

### TIER 6: GEMATRIA EXPANSION (~150 features)

**Priority: MEDIUM**
**Effort: MEDIUM (engine exists, need to apply)**
**Signal: Low standalone signal but contributes to ensemble**

Apply universal gematria to all text sources. However, this is
partially blocked by empty source data:
- tweets.db: ends September 2021 (stale)
- sports_results.db: 4 games total
- horse_races: 1 race total

For news: 56,051 articles available -- CAN do full gematria expansion now.

### TIER 7: CROSS-FEATURES (~35 features)

**Priority: MEDIUM**
**Effort: LOW (multiplication/AND logic)**
**Depends on: Tiers 1-3 being complete**

Build all planned cross-features from DATAPOINT_MATRIX.md.
Most are simple products or AND conditions of features that
need to be working first.

### TIER 8: HEBREW CALENDAR + DATE FEATURES (~8 features)

**Priority: LOW-MEDIUM**
**Effort: LOW (pure date computation)**
**Signal: Untested but low cost to add**

Holiday flags, Omer counting, Ramadan, Diwali, Chinese New Year.
No external data needed.

### TIER 9: SENTIMENT + LLM INTEGRATION (~15 features)

**Priority: MEDIUM (per Phase 2 plan)**
**Effort: MEDIUM-HIGH (SDK integration)**
**Signal: Research shows LLMs beat regex/dictionary**

Anthropic SDK integration for Haiku sentiment + Sonnet risk manager.
This is Phase 2 Step 4 in the current plan.

### TIER 10: HMM REGIME + ARABIC LOTS + SOCIAL (~20 features)

**Priority: LOW**
**Effort: HIGH**

- HMM regime: needs hmmlearn, moderate complexity
- Arabic Lots: needs ephemeris computation from scratch
- Social/cultural (Reddit, YouTube, Telegram, Wikipedia): needs new scrapers

---

## Source Database Health Summary

| Database | Rows | Date Range | Status |
|----------|------|------------|--------|
| btc_prices.db | 3,856,799 | Full history | HEALTHY |
| astrology_full.db | 6,285 | 2009-01-01 to 2026-03-17 | HEALTHY |
| macro_data.db | 1,816 | 2019-01-02 to 2026-03-17 | HEALTHY |
| fear_greed.db | 2,963 | 2018-02-01 to 2026-03-17 | HEALTHY |
| onchain_data.db (blockchain_data) | 6,162 | 2009-01-03 to 2026-03-16 | HEALTHY |
| onchain_data.db (onchain_data) | 24 | 2026-03-18 only | LIVE ONLY, no history |
| news_articles.db (articles) | 56,051 | 2015-01-14 to 2026-03-17 | HEALTHY |
| news_articles.db (streamer_articles) | 517 | Recent only | LIVE FEED |
| tweets.db | 10,967 | ??? to 2021-09-29 | STALE (4.5 years old) |
| sports_results.db (games) | 4 | 2026-03-18 only | NEARLY EMPTY |
| sports_results.db (horse_races) | 1 | 2014-10-10 | DEAD |
| space_weather.db | 1 + 63 flares | 2026-03-18 only | JUST STARTED |
| kp_history.txt | ~34,000 days | 1932-2026 | DOWNLOADED, not in DB |

---

## Key Takeaways

1. **One fix unlocks 100 features.** The 1H/4H builders' broken daily-data
   join is the single biggest problem. Fix it and go from 37% to ~55% of target.

2. **Source data is there.** Macro, F&G, astrology, on-chain -- all have
   years of data sitting in their DBs. The feature builders just can't
   reach it at sub-daily resolution.

3. **Space weather is the strongest new signal** (r=-0.40) and is trivial
   to add since kp_history.txt is already downloaded. Just needs parsing
   into features.

4. **Tweets and sports are dead sources.** tweets.db ends Sep 2021.
   sports_results.db has 4 games. Horse racing has 1 race. These entire
   feature categories (~60+ planned features) are blocked until live
   streamers run for months.

5. **The composite vol-to-direction features** are the strategic priority
   after fixing the pipeline. They bridge the confirmed volatility signal
   to actionable directional trades.

6. **310 of 843 target features work today (37%).** With Tier 1 fix:
   ~410 (49%). With Tiers 1-3: ~453 (54%). Full build requires Tiers 1-10.
