# 1W Results & 1D Training Plan — 2026-03-30

## What Worked for 1W (apply to 1d)

### Accuracy Journey
```
v1: 37.8% (3-class, broken Optuna, untuned params, 3714 features)
v3: 46.3% (3-class, tuned params, lean mode 2587 features)
v4: 54.5% (BINARY mode — biggest single improvement +8.2%)
v5: 55.0% (+ TA crosses + 52w features + max_hold=78) ← BEST CPCV
v6: 53.6% (+ more features — too many for 819 rows, slight overfit)
```

### What Made the Biggest Impact (ranked)

#### 1. BINARY CLASSIFICATION (+8.2%)
- Dropped FLAT class, converted SHORT→DOWN(0), LONG→UP(1)
- 1049 rows → 954 rows (dropped 95 FLAT)
- Doubled effective samples per class
- `config.py: BINARY_1W_MODE = True`
- **1d consideration**: 1d has 5733 rows — 3-class may work better with more data. Test both.

#### 2. PARAMETER TUNING (+8.5% total)
- `deterministic=False` (was True — killed all multi-threading)
- `max_bin=7` (was 255 — 36x less memory for binary features)
- `learning_rate=0.1→0.234` (Optuna found higher LR better for tiny data)
- `num_leaves=5` (Optuna chose simpler trees, not our initial 15)
- `early_stopping=50` (was 333 — don't waste rounds)
- `extra_trees=True` (randomized splits reduce variance)
- `CPCV (8,2)` = 28 paths, 75% train (was (5,2) = 10 paths, 60% train)
- **1d consideration**: Let Optuna tune for 1d. Higher LR range [0.05, 0.3]. More leaves OK (5733 rows).

#### 3. LEAN MODE (+1.3%)
- Dropped redundant TA (Ichimoku, Bollinger, MACD, Stochastic, MFI, Williams)
- Kept: SAR, EMA 20/50/200, RSI 14, volume, ATR, ADX, AVWAP
- Full esoteric suite preserved
- `config.py: LEAN_1W_MODE = True`
- **1d consideration**: 1d has 7x more rows — may NOT need lean mode. Keep full TA.

#### 4. SAR-NUMEROLOGY HYBRIDS (new signal)
- `price_sar_dr_diff` (gain=2.1) — SAR-numerology works!
- `rsi_digit_sum` (gain=0.9) — RSI numerology works
- `sar_digit_sum` (gain=1.2 in v4) — validates AlphaNumetrix thesis
- **1d consideration**: Keep all SAR-numerology hybrids. More data = more signal.

#### 5. max_hold_bars 50→78
- Captures full BTC half-cycles (bull 52-78 weeks, bear 52-65 weeks)
- **1d consideration**: Check max_hold_bars for 1d. Should cover 42-144 bar trades.

### What Didn't Work for 1W

1. **TA x TA crosses** — Added noise, slight accuracy drop. 819 rows can't support interaction features.
   - **1d**: With 5733 rows + cross gen, these interactions SHOULD work.

2. **52w features** — return_26, return_52, price_vs_52w_high/low. Not enough data for 52-week rolling windows on 819 bars (first 52 bars are NaN).
   - **1d**: With 5733 daily bars, 52-day features are fine.

3. **Going beyond 2587 features** — v6 added 43 features to 2630, accuracy dropped. 819 rows is the hard limit.
   - **1d**: 5733 rows can handle more features. Cross gen (500K+) will help.

4. **Esoteric features alone** — Only 156/2587 features had any gain. Esoteric contributes ~15% of total gain, TA contributes ~75%.
   - **1d**: More rows = esoteric features fire more often. Expect higher esoteric contribution.

### Top Features by Category (for 1d reference)

**TA (core — keep for all TFs):**
- sma_200_slope, close_vs_sma_200, close_vs_ema_200 (trend)
- volume_sma_20 (regime detection)
- sar_value (SAR level)
- ema_200, ema_50, ema_20 (multi-timeframe trend)
- adx_14 (trend strength)
- avwap_from_swing_high/low (institutional levels)

**Esoteric (proven at weekly scale):**
- doy_sin/cos (calendar seasonality — #1 esoteric, gain=7.8)
- jupiter_saturn_regime (astrology — gain=2.6)
- price_sar_dr_diff (SAR-numerology — gain=2.1)
- tweet_bull/bear_count (social sentiment — gain=1.3)
- cross_new_moon_x_bear (lunar — gain=0.8)
- rsi_digit_sum (numerology — gain=0.9)
- is_fibonacci_day (sacred geometry — gain=0.7)

**Unused but available in DBs (add for 1d):**
- open_interest.db — 12,040 rows UNUSED
- market_cap in onchain_data.db — enables NVT ratio
- tweet engagement (retweet_count, favorite_count) — influence-weighted sentiment
- google_trends momentum features — only 2 of many columns used
- funding_rates extreme values — overleveraged long/short detection

### Confidence Sweet Spot
- **>= 80% confidence: 58.1% accuracy on 1,589 trades** (statistically significant)
- Model predicts UP slightly better (PrecL=67.9%) than DOWN (PrecS=36.9%)
- At high confidence, DOWN predictions improve (more selective)

---

## 1D Training Plan

### Machine
- Same: 1x RTX 5090, 128c EPYC, 258GB RAM ($0.54/hr)
- ETA: ~1.3 hours

### Key Differences from 1W
| Setting | 1W | 1D |
|---------|----|----|
| Rows | 819 (binary ~954) | 5,733 |
| Features | ~2600 (lean) | ~3500+ (full TA) |
| Cross gen | SKIP | YES (500K+ crosses) |
| Binary mode | YES | TBD (test both) |
| Lean mode | YES | NO (enough rows for full TA) |
| CPCV | (8,2) 28 paths | (5,2) 10 paths |
| max_hold_bars | 78 | 90 |

### What to Enable for 1D
1. Cross gen → esoteric x TA crosses (the real matrix signal)
2. Full TA suite (no lean filter — 5733 rows can handle it)
3. Test 3-class AND binary — compare
4. Load open_interest, market_cap, tweet engagement features
5. SAR-numerology hybrids (proven on 1w)
6. TA x TA crosses (failed on 1w due to size, should work on 1d)
7. Let Optuna tune aggressively (25 trials, higher LR range)

### Expected 1D Accuracy
- 1w binary: 55% CPCV
- 1d has 7x more data → expect 58-65% CPCV
- Cross features add esoteric x TA interactions → +3-5%
- More esoteric features should fire → higher esoteric contribution
- Target: **60-65% CPCV binary, 65-70% at high confidence**

### Bugs to Fix Before 1D
1. Audit step KeyError on 'timestamp' — non-critical but annoying
2. Sobol optimizer skip on 1w — check if it works on 1d
3. LSTM sm_120 — verify torch 2.7.0+cu128 is still installed
