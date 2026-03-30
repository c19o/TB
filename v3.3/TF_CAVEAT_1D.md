# 1D (Daily) Timeframe Caveats & Configuration Guide

## Trade Characteristics
- **Rows**: 5,733 daily bars (~2009-2025)
- **Trade duration**: 6-90 bars (6 days to 3 months)
- **max_hold_bars**: 90 (current config)
- **Triple barrier**: tp=2.0×ATR, sl=2.0×ATR
- **Label distribution**: ~56% LONG, ~43% SHORT, ~1% FLAT
- **Max possible trades**: 5733/6 = ~955 (shortest hold) to 5733/90 = ~64 (longest hold)
- **Realistic trade count**: 100-300 over full history

## Why 1D is Different from 1W

### More Data = More Feature Capacity
- 1w: 819 rows / 2587 features = **0.32 samples per feature** (severe curse of dimensionality)
- 1d: 5,733 rows / 3500+ features = **1.6 samples per feature** (still tight but workable)
- 1d with crosses: 5,733 rows / 500K features = **0.01 samples per feature** (rely on EFB bundling + feature_fraction)

**Implication**: 1d can handle the FULL feature set. No lean mode needed. Cross gen should run. The all-vs-all crosses that were terrible for 1w (819 rows) start to make sense at 5,733 rows — but still need aggressive feature_fraction (0.7+) and EFB bundling to manage 500K features.

### Why All-vs-All Crosses Were Bad for 1W
The cross generator creates binary AND pairs from ~2300 binarized columns → 2.9M cross features. On 1w:
- 819 rows × 2.9M features = **0.00028 samples per feature**
- Each cross feature fires ~5-10 times in 819 rows
- min_data_in_leaf=5 means a leaf needs 5 samples. A cross that fires 5 times can ONLY go into 1 leaf.
- The model memorizes individual bars, not patterns → Jaccard=0.175 (unstable)

On 1d with 5,733 rows:
- Each cross fires ~35-70 times → enough for 7-14 leaf assignments → real pattern learning
- But 2.9M features is still extreme. EFB bundles 127 binary features per bundle → ~23K bundles
- feature_fraction=0.7 sees 16K bundles per tree → manageable

**Recommendation**: Enable cross gen for 1d but monitor feature importance. If esoteric crosses have zero gain after training, cross gen is adding noise.

## Feature Adjustments for 1D

### What to KEEP from 1W (proven to work)
1. **SAR-numerology hybrids** — price_sar_dr_diff, rsi_digit_sum, sar_digit_sum all had gain on 1w. More data on 1d should amplify these.
2. **Jupiter-Saturn regime** — gain=2.6 on 1w. 5,733 daily bars span same period but with 7x more resolution.
3. **Calendar seasonality (doy_sin/cos)** — TOP esoteric on 1w (gain=7.8). Even more relevant on 1d because daily resolution captures exact calendar effects.
4. **Binary classification** — Test both. 1d has enough data for 3-class, but binary may still win because FLAT is noise on daily too.
5. **max_bin=7** — Binary features dominate all TFs. Global setting.
6. **deterministic=False** — Critical for all TFs. Multi-threaded histograms.

### What to ADD for 1D (not needed on 1w)

#### Time Features
- **day_of_week** — Monday/Friday effects are REAL on daily crypto. Weekend volume differs. KEEP (was removed for 1w because always same day).
- **hour features** — REMOVE (daily bars always close at same hour, just like 1w).
- **month_sin/cos** — Already computed. Critical for daily seasonal patterns.
- **month_digital_root** — Already computed. Numerological monthly cycles.
- **week_of_year_sin/cos** — Already computed. 52-week seasonal cycle.
- **quarter_sin/cos** — Already computed. Q4 rally, Q1 tax selling.

#### Numerology/Gematria for 1D
- **Day-of-year numerology** — KEEP all doy_ features. On 1d, each doy fires ~15-16 times (5733/365). Enough for pattern detection.
- **Week digital root** — Already computed. 7 values cycling through the year.
- **Month digital root** — Already computed. 9 values (months 1-12 reduced to 1-9).
- **Date string gematria** — NOT implemented yet. Opportunity: compute gematria of "YYYY-MM-DD" string → maps each date to a numerical value. Could capture hidden date patterns.
- **Price-level numerology** — All sar_digit_sum, price_dr, angel_number features work on any TF. Already computed.

**Key insight from 1w**: Month-of-year is more important than day-of-year for WEEKLY because each month appears 4x in weekly data (enough signal) while individual days appear only 0-1x. On DAILY, day-of-year IS useful because each day appears ~15-16x across 16 years.

#### Why We Adjusted Month-of-Year vs Day-of-Year on 1W
On 1w, we didn't remove doy features — they were actually the TOP esoteric signal (gain=7.8). What we DID remove were:
- **day_of_week** (always Monday for weekly bars → zero variance)
- **hour_sin/cos** (always same hour → zero variance)
- **is_monday/is_friday/is_weekend** (constant for weekly)

For 1d: day_of_week IS relevant (Monday effect, Friday positioning). hour_sin/cos still constant (remove). The rest stays.

#### Return Bars for 1D
Current: `[1, 3, 7, 14, 30]` — 1 day to 1 month returns.

**Problem**: Trades last 6-90 bars but longest return feature is 30 bars (1 month). Missing:
- `return_60bar` — 2-month return (captures medium-term momentum)
- `return_90bar` — 3-month return (captures the full trade horizon)

**Fix**: Add to config `'return_bars': [1, 3, 7, 14, 30, 60, 90]`

#### Multi-Week Features for 1D
On daily bars, "52 weeks" = 365 bars. These features ARE useful on 1d:
- `price_vs_365d_high` — distance from yearly high
- `price_vs_365d_low` — distance from yearly low
- `rsi_90` — 90-day RSI (quarterly momentum)
- These work because 5,733 bars >> 365 rolling window (unlike 1w where 52-bar window ate 6% of data)

### What to CHANGE for 1D vs 1W

| Setting | 1W (proven) | 1D (recommended) | Why |
|---------|------------|------------------|-----|
| LEAN_1W_MODE | True | **False** | 5,733 rows handles full TA |
| BINARY_1W_MODE | True | **Test both** | More data might support 3-class |
| Cross gen | SKIP | **ENABLE** | 5,733 rows can learn cross patterns |
| CPCV | (8,2) 28 paths | **(5,2) 10 paths** | Standard. Enough data per fold. |
| num_leaves | 5 (Optuna chose) | **15-31** (let Optuna decide) | More data supports complexity |
| learning_rate | 0.234 | **0.1-0.2** (let Optuna decide) | More data → slower LR OK |
| ES patience | 50 | **100** | More rounds useful with more data |
| num_boost_round | 300 | **500-800** | Can afford more rounds |
| max_hold_bars | 78 | **90** (current, fine for 6-90 bar trades) | Covers full trade range |
| min_data_in_leaf | 5 | **5-8** (let Optuna decide) | More data but rare signals still need low min |
| feature_fraction | 0.7+ | **0.7+** (non-negotiable) | Matrix thesis |
| SKIP_FEATURES | hour_sin/cos, dow, etc. | **hour_sin/cos ONLY** | 1d has day-of-week variation |

## Expected 1D Accuracy
- 1w achieved 55% binary CPCV with 819 rows
- 1d has 7x more rows → expect **58-65% CPCV**
- Cross features (esoteric x TA) should fire enough to contribute → +3-5%
- Calendar seasonality should be even stronger on daily → doy_ features amplified
- More trades (100-300) → statistically significant results
- **Target: 60-65% CPCV, 65-70% at high confidence**

## Trade Duration Insight from 1W (Apply to 1D)
1w showed the edge concentrates in **16-50 bar trades** (59.2% accuracy). Shorter trades were below random.

For 1d with 6-90 bar trades:
- Expect the edge to concentrate in **20-60 bar trades** (3 weeks to 2 months)
- Very short trades (6-10 bars) may be noise — consider minimum hold period
- Very long trades (70-90 bars) may dilute signal — max_hold=90 is appropriate
- Implement a minimum hold filter: don't exit before 15-20 bars unless SL hit

## Esoteric Signals on 1D — What to Expect
On 1w, esoteric contributed ~15% of model gain. On 1d:
- **Mercury retrograde** (3-4 per year × 16 years = 48-64 events × 21 days each = ~1000-1344 daily bars) — enough data to learn from!
- **Lunar cycles** (29.5 day cycle = 12.4 cycles/year = ~200 complete cycles) — VERY learnable on daily
- **Eclipse windows** (4-5 per year × 16 years = ~64-80 events × ~7 days each = ~450-560 bars) — learnable
- **Jupiter-Saturn** — same 20-year cycle. More resolution helps.
- **Hebrew calendar** — Shemitah (7-year), holidays (annual) — all get daily resolution
- **Space weather** — Kp index changes daily. Solar flares are daily events. Finally enough resolution!

**Prediction**: Esoteric features will contribute 25-40% of model gain on 1d (vs 15% on 1w), because daily resolution gives enough data points for each esoteric event type.

## Databases to Load (existing, no new APIs)
All available and partially unused:
1. **open_interest.db** — 12,040 rows UNUSED. Load for 1d.
2. **funding_rates.db** — has open_interest table (12,040 rows) ignored. Load.
3. **onchain_data.db** — market_cap column unused. Enables NVT ratio.
4. **google_trends.db** — only 2 of many columns used. Add gtrends_roc, gtrends_zscore.
5. **tweets.db** — engagement columns (retweet_count, favorite_count) unused. Add influence-weighted sentiment.
6. **fear_greed.db** — classification column unused. Add extreme fear/greed flags.

## Bugs to Fix Before 1D Training
1. **Audit step KeyError 'timestamp'** — backtesting_audit.py:203. Non-critical but fix it.
2. **Sobol optimizer skip** — skipped on 1w, verify it works on 1d.
3. **TA x TA crosses binarization** — column names must match AFTER binarization step.
