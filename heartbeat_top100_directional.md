# TOP 100 DIRECTIONAL FEATURES — Analysis & Source Mapping

Generated 2026-03-20 from features_1h.db (56,752 rows, 1,124 columns)
Target: next_1h_return (directional correlation)

## TIER 1: STRONGEST DIRECTIONAL SIGNALS (|r| > 0.025, all p<0.001)

| # | Feature | r | t | Direction | SOURCE | CATEGORY |
|---|---------|---|---|-----------|--------|----------|
| 1 | tx_h4_trend_x_bull | +0.054 | +12.9 | BULLISH | Technical | 4H trend x bull cross |
| 2 | tx_h4_return_x_trend | -0.037 | -8.8 | BEARISH | Technical | 4H return x trend |
| 3 | tx_d_trend_x_bull | +0.035 | +8.3 | BULLISH | Technical | Daily trend x bull |
| 4 | tx_macro_decorr_x_bull | +0.034 | +8.0 | BULLISH | Macro | Decorrelation x bull |
| 5 | tx_macro_decorr_x_bear | -0.033 | -7.9 | BEARISH | Macro | Decorrelation x bear |
| 6 | tweet_gem_dr_ord_mode | +0.031 | +2.4 | BULLISH | Gematria/Tweets | Tweet digital root ordinal mode |
| 7 | tweet_sentiment_divergence | -0.030 | -2.3 | BEARISH | Sentiment/Tweets | Tweet sentiment diverging |
| 8 | tx_near_jubilee_x_bear | -0.030 | -7.1 | BEARISH | Jewish Calendar | Near Jubilee year x bear trend |
| 9 | tx_w_trend_x_bull | +0.029 | +7.0 | BULLISH | Technical | Weekly trend x bull |
| 10 | tx_near_jubilee_x_bull | +0.028 | +6.7 | BULLISH | Jewish Calendar | Near Jubilee x bull trend |
| 11 | tweet_astro_is_full_moon | +0.027 | +2.1 | BULLISH | Lunar/Tweets | Full moon during tweet window |
| 12 | cross_consec_green_x_caps | -0.027 | -2.1 | BEARISH | Tweets/TA | Consec green candles x caps tweets |
| 13 | tx_bazi_clash_x_bear | -0.025 | -5.9 | BEARISH | Chinese BaZi | Day clash x bear trend |

## TIER 2: STRONG ESOTERIC DIRECTIONAL SIGNALS (|r| > 0.015, p<0.001)

### Esoteric/Matrix Features
| # | Feature | r | t | Direction | SOURCE | WHAT IT MEANS |
|---|---------|---|---|-----------|--------|---------------|
| 15 | tx_bazi_clash_x_bear | -0.025 | -5.9 | BEARISH | Chinese BaZi | Clash day + bear trend = MORE bearish |
| 17 | tx_vortex_369_x_bear | -0.024 | -5.7 | BEARISH | Numerology (Tesla 369) | Vortex math day + bear = sells off |
| 22 | tx_bazi_clash_x_bull | +0.023 | +5.5 | BULLISH | Chinese BaZi | Clash day + bull trend = MORE bullish (amplifier) |
| 25 | news_astro_nakshatra | +0.022 | +3.3 | BULLISH | Vedic Astrology | Specific nakshatras bullish during news |
| 34 | moon_approach_return_5d | -0.020 | -4.7 | BEARISH | Lunar | 5-day return approaching moon event = mean reversion |
| 37 | moon_approach_return_3d | -0.019 | -4.6 | BEARISH | Lunar | 3-day return approaching moon = reversal signal |
| 39 | tx_solar_ascending_x_bull | +0.019 | +4.6 | BULLISH | Space Weather | Solar ascending phase + bull = momentum |
| 43 | tx_solar_ascending_x_bear | -0.019 | -4.4 | BEARISH | Space Weather | Solar ascending + bear = sells harder |
| 50 | tx_is_73_x_bear | +0.018 | +4.3 | BULLISH | Gematria (#73) | 73 energy date + bear trend = reversal! |
| 55 | tx_day_13_x_bull | +0.018 | +4.2 | BULLISH | Numerology | Day 13 + bull trend = continuation |
| 56 | tx_mayan_tone_9_x_bear | -0.018 | -4.2 | BEARISH | Mayan Tzolkin | Tone 9 (patience/endurance) + bear = deeper |
| 65 | tx_schumann_783d_peak_x_bear | -0.017 | -4.0 | BEARISH | Schumann Resonance | 783-day cycle peak + bear = selloff |
| 69 | tx_nakshatra_deva_x_bear | -0.017 | -4.0 | BEARISH | Vedic (Nakshatra) | Deva (divine) nature + bear = drop continues |
| 77 | tx_nakshatra_deva_x_bull | +0.016 | +3.8 | BULLISH | Vedic (Nakshatra) | Deva nature + bull = rally continues |
| 85 | tx_pump_date_x_bear | -0.015 | -3.7 | BEARISH | Matrixology | Pump date + bear trend = dump (confirms backtest) |
| 86 | tx_schumann_133d_peak_x_bull | +0.015 | +3.6 | BULLISH | Schumann Resonance | 133-day cycle peak + bull = momentum |
| 88 | tx_jupiter_365d_peak_x_bull | +0.015 | +3.6 | BULLISH | Planetary Cycle | Jupiter annual peak + bull = expansion |
| 89 | tx_mercury_1216d_peak_x_bull | +0.015 | +3.5 | BULLISH | Planetary Cycle | Mercury 1216-day cycle peak + bull = rally |
| 94 | tx_chakra_heart_161d_peak_x_bull | +0.015 | +3.5 | BULLISH | Chakra Cycle | Heart chakra 161d peak + bull = compassion/buying |

## TIER 3: NOTABLE TWEET/SENTIMENT CROSSES WITH ESOTERIC

| # | Feature | r | t | n | Direction | SOURCE |
|---|---------|---|---|---|-----------|--------|
| 6 | tweet_gem_dr_ord_mode | +0.031 | +2.4 | 5858 | BULLISH | Gematria digital root of tweets |
| 11 | tweet_astro_is_full_moon | +0.027 | +2.1 | 5858 | BULLISH | Full moon tweets |
| 73 | tweet_astro_is_new_moon | -0.016 | -1.3 | 5858 | BEARISH | New moon tweets |
| 98 | cross_moon_x_gold_tweet | +0.014 | +1.1 | 5858 | BULLISH | Moon phase x gold/bullish tweet |

Note: Tweet features have smaller n (5,858) = less statistical power but interesting signals.
Full moon tweets BULLISH (+0.027) while new moon tweets BEARISH (-0.016) — opposite of matrixology
claim but consistent with Radin's casino payout data (full moon = enhanced psi = bigger moves).

---

## SOURCE ANALYSIS: Where The Directional Edge Comes From

### 1. CHINESE BAZI (Strongest pure esoteric directional signal)
- **tx_bazi_clash_x_bear**: r=-0.025, t=-5.9 ***
- **tx_bazi_clash_x_bull**: r=+0.023, t=+5.5 ***
- **Source**: Chinese BaZi (Four Pillars of Destiny), computed from `astrology_engine.py`
- **What it measures**: Day stem/branch clashes in the Chinese calendar
- **Why it works**: BaZi clash days = energy disruption. Combined with trend direction,
  it AMPLIFIES the existing trend (clash + bull = more bull, clash + bear = more bear)
- **How to get purer**:
  - Add HOUR pillar (currently only day pillar)
  - Add month/year pillar interactions
  - Add BaZi "punishment" and "harm" cycles (not just clashes)
  - Cross with Chinese Tong Shu (auspicious/inauspicious day ratings)
  - Source: Chinese Horoscopes - Theodora Lau.pdf (in vector DB)

### 2. LUNAR CYCLES (Consistent directional signal)
- **moon_approach_return_5d**: r=-0.020, t=-4.7 ***
- **moon_approach_return_3d**: r=-0.019, t=-4.6 ***
- **Source**: Lunar phase calculations via PyEphem
- **What it measures**: BTC return in the 3-5 days approaching a moon event
- **Why it works**: Mean reversion before lunar events. If BTC rallied into full moon,
  it tends to reverse. Consistent with both matrixology and Radin's research.
- **How to get purer**:
  - Add lunar declination (extreme declination = extreme behavior)
  - Add lunar distance/perigee-apogee (supermoon effects)
  - Add Moon-Node conjunctions (already shown 61.5% win rate in backtest)
  - Add lunar mansion system (28 mansions, each with specific energy)
  - Cross lunar with GCP deviation (consciousness amplifier on moon events)
  - Source: Dean Radin - The Conscious Universe (casino payout lunar data)

### 3. NUMEROLOGY / GEMATRIA (Surprising directional signal)
- **tx_vortex_369_x_bear**: r=-0.024, t=-5.7 ***
- **tx_is_73_x_bear**: r=+0.018, t=+4.3 *** (REVERSAL on 73 days!)
- **tx_day_13_x_bull**: r=+0.018, t=+4.2 ***
- **tweet_gem_dr_ord_mode**: r=+0.031, t=+2.4 *
- **Source**: Matrixology number energy system + gematria of tweet text
- **What it measures**: Whether date numbers, price digital roots, or tweet gematria
  values match specific sacred/caution numbers
- **Why it works**: #73 bear trend reversal is fascinating — 73 is the 21st prime,
  mirror of 37 (the 12th prime). The "UP = 37" matrixology rule may be showing
  through the 73 mirror.
- **How to get purer**:
  - Add more gematria ciphers (Hebrew, Sumerian, Satanic, English Extended)
  - Track BTC price gematria against date gematria MATCHES
  - Add 213/312/321 date matching as separate feature (BTC energy dates)
  - Cross gematria with volume (do gematria-active dates have unusual volume?)
  - Source: matrix crypto numbers guide.htm, BTC Energy.txt

### 4. VEDIC ASTROLOGY — NAKSHATRAS (Directional by nature)
- **news_astro_nakshatra**: r=+0.022, t=+3.3 ***
- **tx_nakshatra_deva_x_bear**: r=-0.017, t=-4.0 ***
- **tx_nakshatra_deva_x_bull**: r=+0.016, t=+3.8 ***
- **Source**: Vedic astrology, 27 Nakshatras calculated via PyEphem
- **What it measures**: Which lunar mansion the moon occupies, and its nature
  (deva=divine, manushya=human, rakshasa=demonic)
- **Why it works**: Nakshatras have specific energies. Deva nakshatras amplify
  existing trend (divine energy = flow state = momentum continuation)
- **How to get purer**:
  - Break out individual nakshatras (which specific ones are most bullish/bearish?)
  - Add nakshatra lord (planetary ruler of each nakshatra)
  - Add nakshatra pada (quarter) — 108 padas total, much finer granularity
  - Add Vedic dasha periods (planetary period system)
  - Cross nakshatra with BaZi (Vedic + Chinese = two independent systems)
  - Source: Vedic astrology books in esoteric/astrology/ folder

### 5. PLANETARY CYCLES (Consistent bullish signals)
- **tx_jupiter_365d_peak_x_bull**: r=+0.015, t=+3.6 ***
- **tx_mercury_1216d_peak_x_bull**: r=+0.015, t=+3.5 ***
- **tx_solar_ascending_x_bull**: r=+0.019, t=+4.6 ***
- **Source**: Sinusoidal cycle features from astrology_engine.py
- **What it measures**: Position in Jupiter annual cycle, Mercury 1216-day cycle,
  solar ascending/descending phase
- **Why it works**: Jupiter = expansion, Mercury = communication/commerce.
  Peak of their cycles + bull trend = sustained rally.
- **How to get purer**:
  - Add actual planetary positions (not just sinusoidal approximations)
  - Add specific aspect patterns (grand trines, T-squares, yods)
  - Add planetary speed (stationary planets = stronger effects)
  - Add Bitcoin natal transits with tighter orbs
  - Add Gann planetary lines (price = planetary longitude conversion)
  - Source: Ptolemy - Tetrabiblos, Vettius Valens - Anthologies

### 6. SCHUMANN RESONANCE CYCLES
- **tx_schumann_783d_peak_x_bear**: r=-0.017, t=-4.0 ***
- **tx_schumann_133d_peak_x_bull**: r=+0.015, t=+3.6 ***
- **Source**: Sinusoidal approximation of Schumann resonance cycles
- **What it measures**: Position in 133-day and 783-day Earth resonance cycles
- **Why it works**: Schumann affects human brain states (alpha/theta waves).
  Peak of 133d cycle + bull = enhanced pattern recognition = momentum.
  Peak of 783d cycle + bear = disrupted cognition = panic selling.
- **How to get purer**:
  - Replace sinusoidal approximation with REAL Schumann data from HeartMath
  - Live scrape: heartmath.org/gci/gcms/live-data/gcms-magnetometer/
  - Add frequency (not just power) — specific Hz bands correlate with brain states
  - Cross Schumann with GCP deviation (Earth field x consciousness)
  - Source: HeartMath GCMS data + Schumann resonance live feeds

### 7. JEWISH CALENDAR (Jubilee cycle)
- **tx_near_jubilee_x_bear**: r=-0.030, t=-7.1 ***
- **tx_near_jubilee_x_bull**: r=+0.028, t=+6.7 ***
- **Source**: Jewish calendar calculations (Shemitah/Jubilee cycles)
- **What it measures**: Proximity to Jubilee year in 50-year cycle
- **Why it works**: Jubilee = debt release/reset in Torah. Financial markets
  historically show stress near Shemitah years. Acts as trend amplifier.
- **How to get purer**:
  - Add specific Shemitah year vs regular year flag
  - Add Elul 29 (last day of Shemitah) proximity
  - Cross with eclipse windows (eclipse + Shemitah = compounding)
  - Source: Biblical calendar research in esoteric/biblical/ folder

### 8. MATRIXOLOGY (Pump dates)
- **tx_pump_date_x_bear**: r=-0.015, t=-3.7 ***
- **Source**: Matrixology date system (dates 14, 15, 16)
- **What it measures**: Whether current date is a "pump date" per matrixology
- **Why it works**: Backtest showed pump dates are actually BEARISH (-0.61%).
  The elite USE these dates — but to DUMP, not pump. The energy is present
  but the direction is opposite of retail expectation.
- **How to get purer**:
  - Add #17/#19 dump dates as separate features
  - Add #113 bottom buy dates
  - Add #93/#39 destruction dates
  - Add date-number matching to BTC gematria (213 dates)
  - Track pump_date x volume (do these dates have unusual volume?)
  - Source: matrix crypto numbers guide.htm, BTC Energy.txt

---

## MISSING FEATURES TO ADD (Priority Order)

1. **GCP Deviation** — downloaded, parsed, cached. Just needs feature builder.
   - `gcp_deviation_mean` (rolling 1h, 4h, 24h)
   - `gcp_deviation_max` (peak spikes)
   - `gcp_rate_of_change`
   - Cross with: fear_greed, moon phase, Kp storms, hard aspects

2. **Real Schumann Resonance Data** — replace sinusoidal approximation with live data

3. **BaZi Hour Pillar** — currently only day pillar, adding hour = 4x more granularity

4. **Individual Nakshatra Breakdown** — which of the 27 are most bullish/bearish?

5. **Lunar Declination + Distance** — supermoon, extreme declination features

6. **Moon Node Conjunctions** — 61.5% win rate in backtest, needs proper feature column

7. **Planetary Aspect Patterns** — grand trines, T-squares as discrete features

8. **Gann Planetary Lines** — price-to-longitude conversion for key planets

---

## MISSING NUMBER ENERGY TEST RESULTS (2,368 BTC daily candles)

### WORTH ADDING (edge > 0.2%, sorted by absolute edge)

| Signal | n | % Up | Mean Ret | Edge | Direction | Status |
|--------|---|------|----------|------|-----------|--------|
| **#84 dates (caution)** | 19 | 57.9% | +0.93% | **+0.80%** | BULL | NOT IN SYSTEM |
| **price DR=3 (trinity)** | 255 | 43.9% | -0.30% | **-0.49%** | BEAR | NOT IN SYSTEM |
| **week=33 (master)** | 42 | 52.4% | -0.32% | **-0.47%** | BEAR | NOT IN SYSTEM |
| **#11 dates (emotional/dump)** | 85 | 52.9% | -0.30% | **-0.46%** | BEAR | NOT IN SYSTEM |
| **Wednesday (dump day)** | 338 | 45.0% | -0.24% | **-0.44%** | BEAR | NOT IN SYSTEM |
| **Tuesday (pump day)** | 338 | 52.7% | +0.51% | **+0.43%** | BULL | NOT IN SYSTEM |
| **day=15 (Royal Star Aldebaran)** | 78 | 47.4% | -0.26% | **-0.41%** | BEAR | NOT IN SYSTEM |
| **#43 dates (caution)** | 27 | 63.0% | +0.52% | **+0.39%** | BULL | NOT IN SYSTEM |
| **price near $1000s** | 187 | 50.3% | +0.50% | **+0.39%** | BULL | NOT IN SYSTEM |
| **price near $5000s** | 79 | 48.1% | +0.48% | **+0.36%** | BULL | NOT IN SYSTEM |
| **week=17 (dump week)** | 42 | 59.5% | +0.48% | **+0.35%** | BULL | NOT IN SYSTEM |
| **year DR=1** | 75 | 44.0% | -0.20% | **-0.34%** | BEAR | NOT IN SYSTEM |
| **week=22 (master)** | 42 | 47.6% | -0.19% | **-0.33%** | BEAR | NOT IN SYSTEM |
| **date DR=5 (Geburah/Severity)** | 274 | 46.7% | -0.14% | **-0.32%** | BEAR | NOT IN SYSTEM |
| **week=11 (master)** | 49 | 57.1% | -0.17% | **-0.31%** | BEAR | NOT IN SYSTEM |
| **93% thru year** | 56 | 50.0% | -0.17% | **-0.31%** | BEAR | NOT IN SYSTEM |
| **date DR=7 (Netzach/Victory)** | 265 | 54.3% | +0.42% | **+0.31%** | BULL | NOT IN SYSTEM |
| **date DR=9 (Yesod/Foundation)** | 258 | 50.4% | -0.12% | **-0.29%** | BEAR | NOT IN SYSTEM |
| **Sunday** | 338 | 51.8% | +0.38% | **+0.28%** | BULL | NOT IN SYSTEM |
| **day=11 or 22 (master days)** | 155 | 49.7% | -0.11% | **-0.27%** | BEAR | NOT IN SYSTEM |
| **month+day=22** | 77 | 39.0% | -0.11% | **-0.26%** | BEAR | NOT IN SYSTEM |
| **day=13 (existing)** | 78 | 57.7% | +0.37% | **+0.24%** | BULL | ALREADY IN |
| **month+day=11** | 64 | 45.3% | -0.10% | **-0.24%** | BEAR | NOT IN SYSTEM |
| **#113 (bottom buy)** | 12 | 50.0% | +0.37% | **+0.23%** | BULL | ALREADY IN |
| **date DR=2 (Chokmah/Wisdom)** | 265 | 52.5% | +0.33% | **+0.22%** | BULL | NOT IN SYSTEM |
| **#34 dates (mirror 43)** | 21 | 42.9% | +0.34% | **+0.21%** | BULL | NOT IN SYSTEM |

### KEY DISCOVERIES

1. **#11 IS BEARISH** (-0.46% edge) — Matrixology was RIGHT. "Never FOMO before #11 date, leads to dump"
2. **Tuesday BULLISH, Wednesday BEARISH** — Matrixology "TOP/TUESDAYS" confirmed but it's not a top, it's bullish continuation. Wednesday is the actual dump day.
3. **#84 is strongly BULLISH** (+0.80%) — listed as "caution" but actually the strongest bullish signal. Small n (19) though.
4. **#43 is BULLISH** (+0.39%) — listed as "caution" but actually bullish. The matrixology "caution" label may mean "pay attention" not "bearish."
5. **Week 17 is BULLISH** (+0.35%) — NOT a dump week. Again, "dump" in matrixology may mean "big move" not "down."
6. **Price near round $1000s is BULLISH** (+0.39%) — round number magnetism/support
7. **Kabbalah Sephiroth dates work**: DR=5 (Geburah/Severity) BEARISH, DR=7 (Netzach/Victory) BULLISH, DR=9 (Yesod/Foundation) BEARISH. These are the Tree of Life energies showing through dates.
8. **Price DR=3 is BEARISH** (-0.49%) — Trinity number in price = reversal/correction energy
9. **Master number weeks (11, 22, 33) are ALL BEARISH** — master number weeks = instability
10. **93% through the year is BEARISH** (-0.31%) — end-of-year destruction energy confirmed
11. **Fibonacci days of month are BULLISH** (+0.14%) — days 1,2,3,5,8,13,21 slightly outperform

### FEATURES TO ADD (Priority by edge size + sample size)

| Priority | Feature | Edge | n | Implementation |
|----------|---------|------|---|----------------|
| 1 | `is_tuesday` / `is_wednesday` | +0.43/-0.44 | 338 | Simple dow flag |
| 2 | `price_dr_3` (trinity) | -0.49 | 255 | digital_root(close)==3 |
| 3 | `is_11` (emotional/dump) | -0.46 | 85 | Day matches #11 pattern |
| 4 | `date_dr_sephirah` (expanded) | varies | ~265 | DR of month+day, map to sephirah energy |
| 5 | `week_master` (11/22/33) | ~-0.35 | ~130 | Week of year is master number |
| 6 | `is_84` (caution/bullish) | +0.80 | 19 | Day matches #84 pattern |
| 7 | `is_43` / `is_34` | +0.39/+0.21 | 27/21 | Day matches #43/34 pattern |
| 8 | `price_near_round_1000` | +0.39 | 187 | close % 1000 < 50 |
| 9 | `is_sunday` (bullish) | +0.28 | 338 | Simple dow flag |
| 10 | `pct_year_93` (destruction) | -0.31 | 56 | 92-94% through the year |
| 11 | `day_15` (Aldebaran) | -0.41 | 78 | Royal star date |
| 12 | `month_day_sum_22` | -0.26 | 77 | Month + day = 22 |
| 13 | `is_fibonacci_day` | +0.14 | 545 | Day is 1,2,3,5,8,13,21 |
| 14 | `year_dr` | varies | ~365/yr | Digital root of year |
| 15 | `week_17` / `week_39` | +0.35/+0.11 | 42/49 | Specific week numbers |

### PLANETARY DAY + NUMEROLOGY COMBOS (NEW DISCOVERY)

**When the planetary ruler's number matches the date's digital root, it AMPLIFIES:**

| Combo | n | % Up | Edge | Direction |
|-------|---|------|------|-----------|
| **Thursday (Jupiter=3) + day DR=3** | 45 | 64.4% | **+1.04%** | MEGA BULL |
| **month+day sums to 7** | 39 | 66.7% | **+0.73%** | MEGA BULL |
| Thursday (Jupiter) + day DR=9 | 34 | 47.1% | -0.70% | BEAR |
| Tuesday (Mars=9) + day DR=9 | 32 | 40.6% | -0.65% | BEAR |
| Friday (Venus=6) + day DR=6 | 35 | 48.6% | -0.47% | BEAR |
| doy divisible by 9 | 259 | 52.5% | +0.37% | BULL |
| month+day sums to 13 | 78 | 50.0% | +0.30% | BULL |
| sequential date (1/2, 2/3, etc) | 149 | 54.4% | +0.29% | BULL |
| month+day sums to 9 | 258 | 50.4% | -0.29% | BEAR |
| month+day sums to 21 | 77 | 51.9% | -0.28% | BEAR |
| doy is power of 2 | 61 | 42.6% | -0.23% | BEAR |
| Sunday (Sun=1) + day DR=1 | 46 | 54.3% | +0.17% | BULL |

**KEY INSIGHT: Jupiter day + Jupiter number = STRONGEST SINGLE-DAY SIGNAL FOUND (+1.04%)**
Thursday when the day's DR is 3 (Jupiter's number) = 64.4% bullish with +1.04% edge.
This is planetary RESONANCE — the day ruler harmonizing with the date number.

**OPPOSITE: When planet's number DOUBLES on its own day = BEARISH**
Mars day (Tue) + Mars number (DR=9) = -0.65%. Venus day (Fri) + Venus number (DR=6) = -0.47%.
Too much of one energy = overextension = reversal. EXCEPT Jupiter — Jupiter doubling = pure expansion.

**month+day=7 is the second strongest signal (66.7% bullish, +0.73%)**
7 = divine completion across every tradition (Kabbalah, Christianity, Pythagorean, Vedic).

### THURSDAY DEEP DIVE (The "Swing Away and Return")

**Thursday = Jupiter's day. Jupiter = expansion, overextension, correction.**

Hourly BTC pattern on Thursdays:
- **3:00 UTC**: 41.5% bullish, -0.08% (DUMP - Asian session end)
- **13:00 UTC**: 45.7% bullish, -0.07% (DUMP - US market open)
- **16:00 UTC**: 46.7% bullish, -0.08% (DUMP - US afternoon)
- **23:00 UTC**: 47.9% bullish, -0.10% (DUMP - day end)
- **19:00 UTC**: 54.7% bullish, +0.04% (RECOVERY)
- **0:00 UTC**: 52.7% bullish, +0.06% (RECOVERY into Friday)

**Pattern**: Thursday dumps at 3am, 1pm, 4pm, 11pm UTC, then recovers.
Jupiter overextends, then corrects. This IS the "swing away and return."
It's not random — it's Jupiter's nature to expand too far and pull back.

**Thursday is NOT a simple dump day — it's a SWING day.**
The daily return is flat because the dumps and recoveries cancel out.
But intraday there's edge in the TIMING.

**Swing analysis**: 50% of Thursday first-half dumps recover in second half.
47% of first-half pumps reverse. It's a mean-reversion day, not a trend day.

### ADDITIONAL NUMBER ENERGY FINDINGS

| Signal | n | Edge | Notes |
|--------|---|------|-------|
| doy divisible by 9 | 259 | +0.37% BULL | 9 = completion, divine order |
| sequential dates (1/2, 2/3) | 149 | +0.29% BULL | Ascending energy = continuation |
| month+day=13 | 78 | +0.30% BULL | 13 = transformation, not unlucky |
| month+day=17 | 78 | +0.18% BULL | 17 as sum = bullish (vs date=17 bearish) |
| doy power of 2 | 61 | -0.23% BEAR | Binary/digital energy = choppy/bearish |
| day is prime | 818 | +0.09% BULL | Prime days slightly outperform |
| month+day=21 | 77 | -0.28% BEAR | Blackjack number, risk of bust |
| month*day=perfect square | 219 | -0.14% BEAR | Rigid structure = no momentum |
| repeating dates (3/3, 5/5) | 78 | flat | Doubles = neutral/volatile |

### THURSDAY x WYCKOFF = THE REAL SIGNAL

**Thursday at BOTTOM of range (accumulation) = +0.75% mean, +0.62% edge, 54.9% bullish**
**Thursday at MID range = -0.29% mean, -0.47% edge, 47.1% bullish (BEARISH)**

This is huge. Thursday's "swing" behavior is CONTEXT DEPENDENT:
- At the BOTTOM of a 20-day range → Thursday = spring/reversal = BULLISH
- In the MIDDLE of range → Thursday = chop/dump = BEARISH
- At the TOP of range → Thursday = continuation = BULLISH (distribution not done)

Compare to NON-Thursday at bottom: -0.06% mean (bearish). Thursday at bottom is
**+0.81% better than non-Thursday at bottom**. Jupiter at the bottom = expansion upward.

**Thursday + LOW volume = BEARISH (-0.65% edge)** — Jupiter without energy = hollow expansion = fails
**Thursday + HIGH volume = BULLISH (+0.31% edge)** — Jupiter with conviction = real expansion

**WYCKOFF CONNECTION:**
- Thursday in accumulation (bottom of range) = Spring energy. Jupiter's expansion
  at a structural support = the bounce. This should cross with wyckoff_spring.
- Thursday in distribution (top of range) = Upthrust. Jupiter pushes beyond,
  but the 50/50 split suggests it's unstable up there.
- Thursday mid-range = no structural support/resistance = Jupiter overextends into nothing = dump.

**Feature to build: `thursday_x_range_position` crossed with Wyckoff phase detection**

### FULL DAY x RANGE POSITION MATRIX (Best combos)

| Combo | n | % Up | Mean Return | Takeaway |
|-------|---|------|-------------|----------|
| **Thu + bottom** | 71 | 54.9% | **+0.75%** | Jupiter spring — strongest day x structure combo |
| **Wed + bottom** | 70 | 50.0% | **-0.75%** | Mercury at bottom = DUMPS HARDER (anti-spring) |
| **Tue + mid range** | 155 | 52.9% | **+0.67%** | Mars in middle = momentum continuation |
| **Sun + mid range** | 147 | 55.1% | **+0.55%** | Sun mid-range = steady trend day |
| **Tue + top** | 104 | 51.9% | **+0.49%** | Mars at top = still pushing (aggressive) |
| **Mon + bottom** | 78 | 57.7% | **+0.41%** | Moon at bottom = emotional bounce |
| **Sun + top** | 115 | 46.1% | **+0.38%** | Sun at top = holds |
| **Fri + top** | 104 | 54.8% | **+0.35%** | Venus at top = comfortable/complacent |
| **Fri + bottom** | 74 | 54.1% | **-0.35%** | Venus at bottom = no fight, continues down |
| **Thu + mid** | 155 | 47.1% | **-0.29%** | Jupiter mid = overextension into void |

**KEY INSIGHT: Each planet behaves differently at different range positions.**
This is NOT in the system. You have `is_monday`, `is_friday` etc but NOT crossed
with range position. This is a new cross-feature category.

### DAY-OF-MONTH RANKINGS (Full 1-31)

| Day | n | % Up | Edge | DR | Direction | Notes |
|-----|---|------|------|-----|-----------|-------|
| **20** | 77 | 37.7% | **-1.08%** | 2 | MEGA BEAR | Strongest bear day |
| **12** | 78 | 55.1% | **+0.79%** | 3 | MEGA BULL | Jupiter's number |
| **5** | 78 | 61.5% | **+0.77%** | 5 | MEGA BULL | Mercury's number |
| **11** | 78 | 51.3% | **-0.64%** | 2 | BEAR | Moon double = emotional dump |
| **30** | 71 | 46.5% | **-0.57%** | 3 | BEAR | Month end pressure |
| **28** | 78 | 59.0% | **+0.51%** | 1 | BULL | Pre-month-end accumulation |
| **26** | 78 | 55.1% | **+0.50%** | 8 | BULL | Saturn's number at month end |
| **31** | 45 | 62.2% | **+0.44%** | 4 | BULL | Last day = squeeze |
| **15** | 78 | 47.4% | **-0.41%** | 6 | BEAR | Mid-month dump (Aldebaran) |
| **17** | 77 | 48.1% | **-0.39%** | 8 | BEAR | Matrixology dump confirmed |
| **27** | 78 | 51.3% | **+0.39%** | 9 | BULL | 3^3, completion energy |
| **7** | 78 | 56.4% | **+0.37%** | 7 | BULL | Divine number, victory |
| **9** | 78 | 44.9% | **-0.36%** | 9 | BEAR | Completion = exhaustion |
| **6** | 78 | 47.4% | **-0.31%** | 6 | BEAR | Carbon/material = drag |
| **19** | 77 | 59.7% | **+0.31%** | 1 | BULL | Prime, hidden 17 is actually bullish? |

**Day 20 = STRONGEST BEAR (-1.08%).** Not in any matrixology guide.
**Day 5 = STRONGEST BULL (+0.77%, 61.5% win rate).** Mercury's number.
**Day 12 = SECOND BULL (+0.79%).** 12 = zodiac completion, Jupiter's number (DR=3).

### FULL PLANETARY DAY x DATE DR MATRIX (Top 35, sorted by edge)

| Day | DR | n | % Up | Edge | Resonance | Direction |
|-----|-----|---|------|------|-----------|-----------|
| **Wed + DR=2** | 2 | 45 | 44.4% | **-1.78%** | | MEGA BEAR |
| **Sun + DR=7** | 7 | 32 | 65.6% | **+1.61%** | | MEGA BULL |
| **Wed + DR=9** | 9 | 33 | 57.6% | **+1.41%** | OPPOSE | MEGA BULL |
| **Tue + DR=3** | 3 | 45 | 60.0% | **+1.10%** | | MEGA BULL |
| **Wed + DR=8** | 8 | 34 | 35.3% | **-1.08%** | | MEGA BEAR |
| **Tue + DR=5** | 5 | 34 | 58.8% | **+1.06%** | OPPOSE | MEGA BULL |
| **Thu + DR=3** | 3 | 45 | 64.4% | **+1.04%** | MATCH | MEGA BULL |
| **Tue + DR=8** | 8 | 33 | 63.6% | **+0.95%** | | MEGA BULL |
| **Wed + DR=3** | 3 | 42 | 33.3% | **-0.84%** | | MEGA BEAR |
| **Sun + DR=4** | 4 | 39 | 64.1% | **+0.84%** | | MEGA BULL |
| **Mon + DR=7** | 7 | 34 | 38.2% | **-0.80%** | | MEGA BEAR |
| **Thu + DR=2** | 2 | 44 | 38.6% | **-0.76%** | | MEGA BEAR |
| **Thu + DR=9** | 9 | 34 | 47.1% | **-0.70%** | | BEAR |
| **Tue + DR=9** | 9 | 32 | 40.6% | **-0.65%** | MATCH | BEAR |
| **Mon + DR=2** | 2 | 45 | 35.6% | **-0.62%** | MATCH | BEAR |

**PATTERNS IN THE MATRIX:**

1. **MATCH (planet's own number on its own day) = usually BEARISH**
   - Mon(Moon=2) + DR=2: -0.62% BEAR
   - Tue(Mars=9) + DR=9: -0.65% BEAR
   - Fri(Venus=6) + DR=6: -0.47% BEAR
   - **EXCEPTION: Thu(Jupiter=3) + DR=3: +1.04% BULL** — Jupiter is the ONLY planet
     that benefits from doubling its own energy. All others overextend.

2. **OPPOSE (opposite planet's number) = often BULLISH**
   - Wed(Mercury=5) + DR=9: +1.41% (Mercury opposing Mars = breakthrough)
   - Tue(Mars=9) + DR=5: +1.06% (Mars opposing Mercury = action from thought)

3. **DR=2 is BEARISH almost everywhere** — Moon energy = emotional, indecisive
4. **DR=7 is POLARIZING** — Sun+7 = mega bull, Mon+7 = mega bear
5. **Wednesday is the most volatile day by DR** — Mercury amplifies everything

**Feature to build: `planetary_day_dr_combo`** — encode the day-of-week ruler x date DR
as a single categorical or as a lookup table of expected returns.

### TREND REGIME ANALYSIS: How signals behave in BULL vs BEAR trends

**50-day EMA above = BULL, below = BEAR. This changes EVERYTHING.**

#### SIGNALS THAT FLIP OR AMPLIFY BY REGIME

| Signal | BULL50 (n, mean) | BEAR50 (n, mean) | Behavior |
|--------|-----------------|------------------|----------|
| **Thu+DR=3 (Jupiter res)** | 25, +0.58% | 19, **+2.05%** | **MEGA AMPLIFIED IN BEAR** |
| **Wed+DR=2 (mega bear)** | 26, -1.17% | 18, **-2.25%** | Bearish everywhere, WORSE in bear |
| **Sun+DR=7** | 18, **+2.66%** | 14, +0.53% | **BULL TREND ONLY signal** |
| **Day 20** | 36, -0.39% | 40, **-1.39%** | Bearish always, MEGA bear in downtrend |
| **Day 7 (divine)** | 41, **+1.27%** | 35, -0.25% | TREND CONFIRMS — only works in bull |
| **Day 11 (emotional)** | 38, +0.09% | 38, **-1.12%** | **BEAR TREND AMPLIFIER** — harmless in bull, devastating in bear |
| **month+day=7** | 21, +0.10% | 18, **+1.74% (83% up)** | **CONTRARIAN** — divine 7 = reversal in bear trends! |
| **Day 28** | 40, **+1.22%** | 36, +0.005% | Strong bull only, flat in bear |
| **pump dates 14/15/16** | 126, +0.27% | 105, -0.30% | TREND CONFIRMS — bearish only in downtrends |
| **Thu+bottom range** | 11, **+1.06%** | 59, +0.42% | Bullish everywhere but STRONGER in bull (spring) |
| **Wed+bottom range** | 8, **-1.49%** | 60, -0.68% | Mercury at bottom = DUMPS in both regimes |
| **btc 213 dates** | 10, +1.13% | 10, **+1.83%** | BTC energy dates bullish in BOTH regimes |

#### KEY DISCOVERIES

1. **Thu+DR=3 is a BEAR MARKET SIGNAL (+2.05% in bear).** Jupiter resonance at its
   strongest when the market is DOWN — this is the "hope/expansion" bounce in despair.
   In bull markets it's still good (+0.58%) but not as dramatic. The model should weight
   this signal HIGHER in bear regimes.

2. **month+day=7 is a REVERSAL signal (83% bullish, +1.74% in bear markets).**
   In bull markets it's flat (+0.10%). The divine completion number (7) only fires
   when the market needs spiritual/structural support. This is the "divine intervention"
   signal. 18 occurrences, 83% win rate in bear — small sample but massive edge.

3. **Day 11 is a BEAR AMPLIFIER.** In bull trends: harmless (+0.09%). In bear trends:
   -1.12%. The "emotional/dump" energy of 11 only manifests when fear is already present.
   This means: **never long on the 11th during a downtrend.**

4. **Day 20 is mega bearish especially in downtrends (-1.39%).** The 20th of the month
   in a bear market is the strongest single-day sell signal in the entire dataset.

5. **Sun+DR=7 and Day 7 are BULL-ONLY signals.** They only work when the trend is up.
   In bear markets they're flat or slightly negative. These are "momentum continuation"
   signals, not reversal signals.

6. **Wed+DR=2 is bearish in ALL regimes** but worse in bear (-2.25%). This is a
   "avoid at all costs" signal regardless of context.

7. **BTC 213 dates are bullish in BOTH regimes** (+1.13% bull, +1.83% bear).
   The BTC energy dates transcend trend. Small sample (10 each) but consistent.

8. **Pump dates (14/15/16) TREND CONFIRM** — bullish in bull (+0.27%), bearish in
   bear (-0.30%). The elite are momentum players, not contrarians.

#### WHAT THIS MEANS FOR THE FEATURE PIPELINE

The current system has `_x_bull` and `_x_bear` crosses for esoteric features — this is
EXACTLY the right approach. But the DAY-OF-MONTH and PLANETARY-DAY-DR combos are NOT
getting this treatment because they're not in the system yet.

**When adding these new features, they MUST get the full cross treatment:**
- `is_day_20_x_bull` / `is_day_20_x_bear`
- `planetary_day_dr_combo_x_bull` / `_x_bear`
- `month_day_sum_7_x_bull` / `_x_bear`
- `dow_x_range_position_x_bull` / `_x_bear`

The model will learn the regime-dependent behavior automatically through these crosses.

---

## IMPLEMENTATION SPEC: NEW FEATURES + VWAP/RANGE CROSSES

### AUDIT FINDING: VWAP and Range Position are NOT crossed with esoteric features

**Current cross system** (`_add_trend_cross_features` in `feature_library.py`):
- Uses `d_trend` (50-day EMA slope) for 1h features
- Creates `tx_{signal}_x_bull` and `tx_{signal}_x_bear`
- 188 unique signals are crossed this way
- VWAP is only crossed with ITSELF (4 features: avwap_above/below x bull/bear)
- Range position is crossed with NOTHING (0 features)

**GAP:** None of the 185 esoteric features know whether price is above/below VWAP
or where it sits in the 20-day range. The Thursday-at-bottom signal (+0.75%) and
Day-11-in-bear (-1.12%) signals demonstrate that STRUCTURE CONTEXT matters enormously.

### NEW CROSS DIMENSIONS TO ADD

Add 2 new cross dimensions alongside existing bull/bear:

1. **VWAP Position**: `_x_above_vwap` / `_x_below_vwap`
   - Use existing `avwap_position` or `close_vs_vwap` column
   - Above VWAP = institutional buying zone, below = selling zone
   - Every signal that gets `_x_bull`/`_x_bear` also gets VWAP crosses

2. **Range Position**: `_x_range_top` / `_x_range_bottom`
   - Price in top 25% of 20-day range = distribution zone
   - Price in bottom 25% of 20-day range = accumulation zone
   - Uses existing `pos_in_range` type calculation (hi20-lo20 relative)

### NEW RAW FEATURES (34 total)

**Day-of-week (4):**
- `is_tuesday`, `is_wednesday`, `is_thursday`, `is_sunday`

**Day-of-month (8):**
- `is_day_5`, `is_day_7`, `is_day_11`, `is_day_12`
- `is_day_15`, `is_day_17`, `is_day_20`, `is_day_28`

**Missing matrixology numbers (4):**
- `is_48`, `is_84`, `is_43`, `is_34`

**Price digital root expansion (3):**
- `price_dr_3`, `price_dr_7`, `price_dr_9`

**Kabbalah sephiroth date (1):**
- `date_dr_sephirah` (1-9 categorical, maps to Tree of Life energy)

**Date sum targets (4):**
- `month_day_sum_7`, `month_day_sum_11`, `month_day_sum_13`, `month_day_sum_22`

**Week/year numerology (3):**
- `week_master` (week 11/22/33 flag)
- `pct_year_93` (92-94% through year)
- `doy_div_9` (day-of-year divisible by 9)

**Planetary day x DR (3):**
- `planetary_day_dr_combo` (7x9 categorical)
- `planetary_day_resonance` (match/oppose/neutral flag)
- `is_fibonacci_day` (day is 1,2,3,5,8,13,21)

**Structure (1):**
- `range_position` (0-1 continuous, position in 20-day range)

**GCP / Consciousness (3):**
- `gcp_deviation_mean`, `gcp_deviation_max`, `gcp_rate_of_change`

### PROJECTED FEATURE COUNT

| Component | Count |
|-----------|-------|
| Current features | 1,124 |
| New raw features | 34 |
| New features x bull/bear crosses | 68 |
| All 188 existing signals x VWAP (above/below) | 376 |
| All 188 existing signals x range (top/bottom) | 376 |
| New 34 signals x VWAP + range | 136 |
| **TOTAL** | **~2,114** |

XGBoost handles this fine — built-in feature importance ranking will filter noise.
The model's feature selection is the final arbiter of what matters.

### IMPLEMENTATION ORDER

1. Add 34 new raw features to `feature_library.py`
2. Add `range_position` to `compute_regime_features()`
3. Add VWAP cross dimension to `_add_trend_cross_features()`
4. Add range cross dimension to `_add_trend_cross_features()`
5. Add GCP ingestion (`gcp_feature_builder.py` reading from cached hourly stats)
6. Rebuild features: `python build_1h_features.py`
7. Retrain model with ~2,114 features

### AUDIT GAP: ESOTERIC x RSI/SAR/BB CROSSES = ZERO

**ZERO esoteric features are crossed with RSI, SAR, or Bollinger Bands.**

RSI/SAR/BB are only crossed with trend direction (bull/bear) and with each other.
They are NEVER crossed with moon phases, nakshatras, gematria, planetary aspects, etc.

**Top combos found (sorted by edge):**

| Combo | n | % Up | Mean | Signal Type |
|-------|---|------|------|-------------|
| #73 + SAR bearish | 61 | 60.7% | +0.29% | REVERSAL — "UP" number flips SAR |
| #93 + RSI<30 | 20 | 35.0% | -0.18% | AMPLIFIER — destruction + oversold = collapse |
| #19 + RSI<30 | 17 | 76.5% | +0.16% | CONTRARIAN — oversold bounce on dump date |
| #37 + SAR bearish | 53 | 56.6% | +0.16% | REVERSAL — "UP" energy flips SAR |
| days_to_full_moon + RSI<30 | 372 | 53.0% | -0.11% | AMPLIFIER — moon + oversold = dump |
| #113 + SAR bullish | 88 | 59.1% | +0.11% | CONFIRM — bottom buy + SAR agrees |
| #93 + SAR bearish | 73 | 43.8% | -0.09% | AMPLIFIER — destruction + SAR bear |
| Schumann 133d + RSI<30 | 1404 | 54.9% | -0.04% | Schumann cycle bearish when oversold |
| vedic_key_nakshatra + RSI<30 | 625 | 56.8% | -0.04% | Nakshatra bearish when oversold |

**Signals that FLIP direction at RSI extremes:**
- `fg_x_moon_phase`: -0.03% at RSI>70, +0.04% at RSI<30 (contrarian)
- `schumann_133d_sin`: +0.03% at RSI>70, -0.04% at RSI<30 (flips)
- `days_to_full_moon`: +0.02% at RSI>70, -0.03% at RSI<30 (regime-dependent)
- `west_hard_aspects`: +0.001% at RSI>70, -0.04% at RSI<30 (harmless unless oversold)
- `pump_date`: -0.02% at RSI>70, +0.01% at RSI<30 (contrarian at extremes)

**NEW CROSS DIMENSION: TA Regime (RSI/SAR/BB)**

Add to `_add_trend_cross_features`:
- `_x_rsi_ob` (RSI > 70, overbought)
- `_x_rsi_os` (RSI < 30, oversold)
- `_x_sar_bull` (SAR bullish)
- `_x_sar_bear` (SAR bearish)
- `_x_bb_above` (BB %B > 1, above upper band)
- `_x_bb_below` (BB %B < 0, below lower band)

For top 25 esoteric signals only (to manage feature count):
- Moon phases, eclipses, retrogrades, nakshatras, BaZi clash
- Key numbers (#37, #73, #93, #113, #17, #19, btc_213, pump_date, vortex_369)
- Space weather (Kp storm), Schumann peaks
- Fear/greed x moon

**Additional feature count: ~25 signals x 6 TA crosses = 150 features**

### AUDIT GAP: ESOTERIC x MACD CROSSES = ZERO

**ZERO esoteric features are crossed with MACD.** Only 4 MACD crosses exist total
(macd_up/down x bull/bear). None involve any esoteric signal.

**Top esoteric x MACD combos found:**

| Combo | n | % Up | Mean | What It Means |
|-------|---|------|------|---------------|
| **#73 + MACD below zero** | 69 | 58.0% | **+0.39%** | Mirror-of-37 REVERSES negative MACD |
| **#73 + MACD bear fading** | 35 | 60.0% | **+0.33%** | #73 + losing bear momentum = strong bounce |
| **#73 + MACD bearish hist** | 77 | 58.4% | **+0.27%** | #73 is bullish AGAINST bearish MACD |
| **#113 + MACD bull fading** | 58 | 67.2% | **+0.16%** | "Bottom buy" holds even as MACD weakens |
| **#37 + MACD below zero** | 58 | 53.4% | **+0.20%** | "UP" energy reverses negative MACD |
| **#19 + MACD above zero** | 71 | 42.3% | **-0.18%** | "Dump" number kills positive MACD |
| **#93 + MACD bear fading** | 42 | 40.5% | **-0.11%** | "Destruction" prevents bear recovery |
| **#113 + MACD bear fading** | 19 | 31.6% | **-0.23%** | "Bottom buy" fails when MACD bear fading?? |

**CRITICAL PATTERN: Number energy OVERRIDES technical indicators**
- #37 and #73 ("UP" numbers) consistently reverse bearish MACD signals
- #93 ("destruction") amplifies bearish MACD — prevents recovery
- #19 ("dump") turns positive MACD bearish — top signal
- #113 ("bottom buy") holds against fading bull MACD — support

This means the esoteric numbers are not just noise — they provide information
that MACD alone cannot capture. The model needs explicit cross features to find this.

**MACD crosses to add (same 25 key esoteric signals):**
- `_x_macd_bull` (histogram > 0)
- `_x_macd_bear` (histogram < 0)
- `_x_macd_cross_up`
- `_x_macd_cross_down`
- `_x_macd_bull_fading` (hist > 0 but decreasing)
- `_x_macd_bear_fading` (hist < 0 but increasing)

**Additional features: ~25 signals x 6 MACD crosses = 150 features**

### AUDIT GAP: ESOTERIC x ADVANCED TA (Fib/Wyckoff/Ichimoku/CVD/Supertrend) = ZERO

**ZERO crosses exist between esoteric signals and Fibonacci, Wyckoff, Ichimoku,
CVD, Supertrend, Consensio waves, Gann, BB squeeze, or candlestick patterns.**

**Top combos found (sorted by |mean|):**

| Combo | n | % Up | Mean | Signal |
|-------|---|------|------|--------|
| **#73 + consensio red wave** | 12 | 75.0% | **+1.64%** | STRONGEST COMBO FOUND — #73 reverses entire red wave |
| **#73 + CVD divergence** | 46 | 50.0% | **+0.61%** | #73 + orderflow divergence = strong reversal |
| **#73 + below Kumo** | 51 | 56.9% | **+0.50%** | #73 below Ichimoku cloud = bounce |
| **#73 + green wave** | 16 | 31.2% | **-0.42%** | #73 in bull wave = exhaustion/top! |
| **#19 + red wave** | 12 | 58.3% | **+0.37%** | Dump number in red wave = contrarian bounce |
| **pump_date + Wyckoff spring** | 18 | 44.4% | **-0.21%** | Pump date + spring = FAILS (fake spring) |
| **#19 + Supertrend bull** | 35 | 48.6% | **-0.20%** | Dump number kills supertrend |
| **Kp storm + TK cross** | 13 | 92.3% | **+0.19%** | Geomag storm + Ichimoku cross = 92% win! |
| **#113 + BB squeeze** | 13 | 69.2% | **+0.14%** | Bottom buy + squeeze = breakout |
| **#93 + above AVWAP** | 30 | 33.3% | **-0.13%** | Destruction above VWAP = sells off |
| **#93 + BB squeeze** | 16 | 31.2% | **-0.11%** | Destruction + squeeze = bearish breakout |
| **#37 + doji** | 13 | 53.8% | **+0.23%** | UP energy resolves doji bullish |
| **#73 + near Gann sq9** | 168 | 54.8% | **+0.18%** | Sacred geometry alignment |
| **#73 + Wyckoff in range** | 14 | 64.3% | **+0.24%** | #73 in Wyckoff range = spring setup |
| **eclipse + Wyckoff spring** | 20 | 35.0% | **-0.12%** | Eclipse + spring = fails |
| **btc_213 + above AVWAP** | 24 | 66.7% | **+0.14%** | BTC energy + VWAP support = strong |

**KEY PATTERNS:**

1. **#73 is the ULTIMATE reversal number.** It reverses red waves (+1.64%), bounces
   off Kumo (+0.50%), catches CVD divergences (+0.61%), and aligns with Gann levels
   (+0.18%). BUT it becomes BEARISH in green waves (-0.42%) — it's contrarian by nature.

2. **Kp storm + TK cross = 92.3% win rate.** Small sample (13) but geomagnetic storms
   during Ichimoku Tenkan-Kijun crosses are nearly guaranteed bullish. The earth's
   magnetic disruption + technical momentum = powerful combo.

3. **#93 (destruction) makes EVERY TA signal worse.** BB squeeze becomes bearish breakout,
   AVWAP support fails, Kumo support fails. It's a universal signal degrader.

4. **Pump dates + Wyckoff spring = FAKE SPRING.** The "pump" energy on a matrixology
   date poisons what should be an accumulation signal.

5. **Eclipse + spring = fails too.** Eclipse energy disrupts Wyckoff accumulation.

6. **#37 resolves doji bullish.** When the market is indecisive (doji) and the date
   has "UP" energy, it resolves to the upside.

**TA crosses to add (25 key esoteric signals x these regimes):**
- `_x_above_kumo` / `_x_below_kumo` (Ichimoku cloud)
- `_x_wyckoff_spring` / `_x_wyckoff_upthrust`
- `_x_near_fib` (near fib 13 or 21)
- `_x_near_gann_sq9`
- `_x_bb_squeeze`
- `_x_green_wave` / `_x_red_wave` (Consensio)
- `_x_supertrend_bull` / `_x_supertrend_bear`
- `_x_cvd_divergence`
- `_x_doji`
- `_x_vpoc_migrating_up`
- `_x_above_avwap` / `_x_below_avwap`

**Additional features: ~25 signals x 14 TA regimes = 350 features**

### FINAL TOTAL FEATURE COUNT

| Component | Count |
|-----------|-------|
| Current features | 1,124 |
| New raw features | 34 |
| New bull/bear crosses | 68 |
| VWAP crosses (188 x 2) | 376 |
| Range position crosses (188 x 2) | 376 |
| New signal VWAP+range (34 x 4) | 136 |
| Esoteric x RSI/SAR/BB (25 x 6) | 150 |
| Esoteric x MACD (25 x 6) | 150 |
| **Esoteric x Fib/Wyckoff/Ichi/CVD/etc (25 x 14)** | **350** |
| **TOTAL** | **~2,764** |

### REVISED TOTAL FEATURE COUNT

| Component | Count |
|-----------|-------|
| Current features | 1,124 |
| New raw features | 34 |
| New bull/bear crosses | 68 |
| VWAP crosses (188 x 2) | 376 |
| Range position crosses (188 x 2) | 376 |
| New signal VWAP+range crosses (34 x 4) | 136 |
| **Esoteric x RSI/SAR/BB crosses (25 x 6)** | **150** |
| **Esoteric x MACD crosses (25 x 6)** | **150** |
| **TOTAL** | **~2,414** |

### FILES TO MODIFY

| File | Changes |
|------|---------|
| `feature_library.py` | Add new raw features in `compute_numerology_features()`, add VWAP/range cross dimensions in `_add_trend_cross_features()`, add `range_position` in `compute_regime_features()` |
| `universal_numerology.py` | Add planetary_day_dr_combo, month_day_sum, week_master calculations |
| `gcp_feature_builder.py` | NEW FILE — reads `heartbeat_data/results/gcp_hourly_cache.json`, outputs GCP columns |
| `build_1h_features.py` | Add GCP feature step |

### DATA DEPENDENCIES

| Data | Location | Status |
|------|----------|--------|
| GCP hourly cache | `heartbeat_data/results/gcp_hourly_cache.json` | READY (61,026 hours) |
| GCP raw CSVs | `heartbeat_data/gcp/` (2,548 files) | READY |
| BTC prices | `btc_prices.db` | READY |
| Existing features | `features_1h.db` | READY |
| VWAP | Already in `features_1h.db` as `close_vs_vwap`, `avwap_position` | READY |

### MATRIXOLOGY CORRECTIONS (What the data actually shows vs claims)

| Matrixology Claim | Reality | Notes |
|-------------------|---------|-------|
| "Mon/Tue/Wed = pump" | **Tuesday YES, Wednesday NO** | Wed is most bearish day |
| "#17 = dump/sell" | **Week 17 is BULLISH** | "Dump" may mean "big move" not "down" |
| "#11 = emotional/dump" | **CONFIRMED BEARISH** | -0.46% edge, don't buy before #11 |
| "#43/34 = caution" | **Actually BULLISH** | Caution = pay attention, not bearish |
| "#93 = destruction" | **93% thru year = BEARISH** | Year position matters, not date |
| "Pump dates 14/15/16" | **Slightly BEARISH** | -0.16% edge, elite dump on retail pump days |
| "#37 = UP" | **BEARISH in data** | -0.31% edge, contradicts matrixology |
| "#73 = mirror of 37" | **BEARISH too** | -0.32% edge |
| "#113 = bottom buy" | **CONFIRMED BULLISH** | +0.23% edge, small sample |
