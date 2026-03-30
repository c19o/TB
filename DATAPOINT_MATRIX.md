# SAVAGE22 — Complete Datapoint Matrix & Build Plan

Every technique applied to every target. Each intersection = a feature for the ML model.
**Target: ~600+ features. All correlated with BTC. All backtested. All used live.**

---

## ARCHITECTURE

```
LAYER 1: DATA COLLECTORS (live feeds → raw DBs)
├── tweet_streamer.py      → tweets.db          (every 5 min per account)
├── news_streamer.py       → news_articles.db   (RSS + CryptoPanic + Reddit, every 5 min)
├── sports_streamer.py     → sports_results.db  (ESPN + TheSportsDB, every 15 min)
├── crypto_streamer.py     → onchain_data.db    (blockchain, funding, OI, whale txns)
├── macro_streamer.py      → macro_data.db      (indices, DXY, gold, VIX, yields)
├── astro_engine.py        → astrology_full.db  (ephemeris, pre-computed daily)
└── download_btc.py        → btc_prices.db      (candles, already exists)

LAYER 2: UNIVERSAL ENGINES (apply techniques to anything)
├── numerology_engine.py   — digital_root(any_number), master_check, sequence_detect
├── gematria_engine.py     — all 6 methods on any text string
├── astro_engine.py        — planetary_hour(timestamp), nakshatra(timestamp), moon_phase(ts)
├── sentiment_engine.py    — sentiment(text), caps_detect, exclamation_score
└── color_engine.py        — detect_gold/red/green in image URL

LAYER 3: FEATURE BUILDERS (raw DBs → feature matrices)
├── build_15m_features.py  → features_15m.db    (all 600+ features at 15m resolution)
├── build_1h_features.py   → features_1h.db
├── build_4h_features.py   → features_4h.db
└── build_1d_features.py   → features_complete.db

LAYER 4: ML PIPELINE
├── ml_multi_tf.py         — train XGBoost with ALL features, protected list, sample weights
└── live_trader.py         — compute features live from all DBs, predict, trade
```

---

## TECHNIQUES (13 categories)

### 1. NUMEROLOGY — Digital Root Reduction
Reduce any number to 1-9 (or master numbers 11, 22, 33).

### 2. GEMATRIA (6 methods on any text)
- **English Ordinal**: A=1, B=2 ... Z=26, sum all letters
- **Reverse Ordinal**: A=26, B=25 ... Z=1
- **Reduction**: Digital root of ordinal
- **English Gematria**: A=6, B=12 ... Z=156 (multiples of 6)
- **Simple/Jewish**: Traditional Hebrew letter values
- **Satanic**: A=36, B=37 ... Z=61

### 3. WESTERN ASTROLOGY
- Moon phase (0-29.5 day cycle)
- Moon mansion (1-28)
- Mercury retrograde (binary)
- Venus retrograde
- Mars retrograde
- Saturn station/retrograde
- Jupiter station/retrograde
- Planetary strength (composite score)
- Hard aspects (conjunction, opposition, square)
- Soft aspects (trine, sextile)
- Eclipse windows (solar/lunar ± 5 days)
- Planetary hours (ruling planet of current hour)
- Void of Course Moon

### 4. VEDIC ASTROLOGY
- Nakshatra (1-27 lunar mansions)
- Nakshatra nature (deva/human/rakshasa)
- Nakshatra guna (sattva/rajas/tamas)
- Key nakshatras (Purva Ashadha, Shravana, etc.)
- Tithi (1-30 lunar day)
- Yoga (1-27 luni-solar combinations)
- Karana (1-11 half-tithis)
- Panchang score (composite quality of day)
- Rahu/Ketu axis
- Dasha periods

### 5. CHINESE BAZI / FOUR PILLARS
- Day stem (1-10 Heavenly Stems)
- Day branch (1-12 Earthly Branches)
- Day element (wood/fire/earth/metal/water)
- Hour pillar
- Clash/harmony relationships

### 6. MAYAN TZOLKIN
- Tone (1-13)
- Day sign (1-20)
- Combined kin number (1-260)

### 7. HEBREW CALENDAR
- Hebrew date
- Shmita (sabbatical) year
- Major holidays (Yom Kippur, Passover, Rosh Hashanah, etc.)
- Omer counting period

### 8. ARABIC LOTS
- Lot of Commerce
- Lot of Increase
- Lot of Catastrophe
- Lot of Treachery
- Moon conjunctions to each lot (within 10°)

### 9. SACRED GEOMETRY
- Golden ratio distance (price vs Fibonacci levels)
- Fibonacci retracement levels
- Gann Square of 9 distance
- Tesla 369 patterns

### 10. NUMBER PATTERNS
- Master numbers in value (11, 22, 33)
- Sequence detection (113, 322, 93, 213, 666, 777, etc.)
- Consecutive patterns (3+ green/red days)
- Day 13, Day 21, Day 27 of month

### 11. SENTIMENT / NLP
- Headline sentiment (positive/negative/neutral)
- ALL CAPS detection
- Exclamation intensity
- Keyword extraction
- Urgency score

### 12. COLOR ANALYSIS
- Gold detection (in images)
- Red detection (in images)
- Green detection
- Dominant color extraction

### 13. TIMING PATTERNS
- Hour of day (sin/cos encoded)
- Day of week
- Day of month
- Day of year
- Days remaining in year
- Specific times (noon CST, 3pm CST, midnight UTC)

---

## TARGETS (114 items across 9 categories)

### A. BTC PRICE DATA (17 targets)
1. Current price (last trade)
2. Candle open
3. Candle high
4. Candle low
5. Candle close
6. Daily close
7. Weekly close
8. Monthly close
9. Price change (dollar)
10. Price change (percent)
11. Volume
12. VWAP
13. ATR value
14. Bollinger Band values (upper, mid, lower)
15. Support/resistance levels
16. All-time high distance
17. 52-week high/low distance

### B. TWITTER/X POSTS (13 targets)
18. Tweet full text → gematria, sentiment, keywords
19. Tweet username → gematria
20. Tweet display name → gematria
21. Tweet timestamp → numerology, astrology of moment
22. Retweet count → numerology
23. Like count → numerology
24. Reply count → numerology
25. Image colors → gold/red/green detection
26. Tweet word count → numerology
27. Hashtags → gematria per hashtag
28. @mentions → gematria per mention
29. Time between tweets → pattern detection
30. Tweet during specific planetary hours

### C. NEWS HEADLINES (9 targets)
31. Headline text → gematria (all 6 methods)
32. Headline text → sentiment
33. Headline text → keyword extraction
34. Publication timestamp → numerology, astrology
35. Source name → gematria
36. Author name → gematria
37. Article word count → numerology
38. Breaking news frequency → count per 4H bucket
39. Headline digital root → caution (DR=9) detection

### D. SPORTS — NFL / NBA / MLB / NHL (16 targets)
40. Home team name → gematria (all methods)
41. Away team name → gematria
42. Winning team name → gematria
43. Losing team name → gematria
44. Final score → numerology (each score + combined)
45. Score differential → numerology
46. Game timestamp → numerology, astrology
47. Game day of year → numerology
48. Stadium/venue name → gematria
49. MVP/star player name → gematria
50. Jersey numbers of scorers → numerology
51. Total points → numerology
52. Overtime (binary) → pattern detection
53. Upset (underdog wins) → correlation with BTC
54. Championship/playoff game → correlation
55. Super Bowl / Finals / World Series date → numerology

### E. HORSE RACING (13 targets)
56. Race name → gematria (all methods)
57. Winning horse name → gematria
58. Losing horses → gematria
59. Jockey name → gematria
60. Trainer name → gematria
61. Horse number/post position → numerology
62. Race distance → numerology
63. Winning time → numerology
64. Track name → gematria
65. Race timestamp → numerology, astrology
66. Odds → numerology
67. Triple Crown races → specific correlation
68. Derby/Preakness/Belmont dates → numerology

### F. CRYPTO ON-CHAIN (13 targets)
69. Block number → numerology
70. Block timestamp → astrology of mining moment
71. Transaction hash → gematria of hex
72. Gas price → numerology
73. Mempool size → numerology
74. Hash rate → numerology
75. Difficulty adjustment → numerology
76. Halving countdown → days remaining DR
77. Whale transaction amounts → numerology
78. Exchange inflow/outflow → numerology
79. Funding rate → numerology
80. Open interest → numerology
81. Liquidation amounts → numerology

### G. MACRO / TRADITIONAL FINANCE (12 targets)
82. S&P 500 close → numerology
83. NASDAQ close → numerology
84. DXY (Dollar Index) → numerology
85. Gold price → numerology
86. Oil price → numerology
87. VIX → numerology
88. 10Y Treasury yield → numerology
89. Fed Funds Rate → numerology
90. CPI number → numerology, gematria of report title
91. FOMC statement → gematria, sentiment
92. Fed chair quotes → gematria
93. Jobs report numbers → numerology

### H. SOCIAL / CULTURAL (7 targets)
94. Google Trends interest → correlation
95. Wikipedia Bitcoin pageviews → correlation
96. Reddit post titles → gematria, sentiment
97. Reddit comment count → numerology
98. Fear & Greed Index → numerology, cross with moon phase
99. YouTube video titles (crypto influencers) → gematria
100. Telegram channel messages → gematria, sentiment

### I. DATES & CALENDAR (14 targets)
101. Current date → all numerology (month+day+year reduction)
102. Day of year (1-366) → DR, master number check
103. Days remaining in year → DR
104. Date palindromes (e.g., 02/02/2020)
105. Repeating dates (11/11, 12/12)
106. Full moon / new moon dates
107. Equinox / solstice dates
108. Eclipse dates
109. Retrograde start/end dates
110. Friday the 13th
111. Shmita year transitions
112. Chinese New Year
113. Diwali (Vedic)
114. Ramadan start/end

---

## THE FULL FEATURE EXPANSION

### Numerology × Everything (~125 features)
| Target | Features |
|--------|----------|
| BTC price close | `dr_close`, `master_close` (is 11/22/33) |
| BTC daily close | `dr_daily_close`, `master_daily_close` |
| BTC weekly close | `dr_weekly_close`, `master_weekly_close` |
| BTC monthly close | `dr_monthly_close` |
| BTC volume | `dr_volume` |
| BTC ATR | `dr_atr` |
| BTC price change % | `dr_pct_change` |
| Tweet like count | `dr_tweet_likes` |
| Tweet RT count | `dr_tweet_rts` |
| Tweet reply count | `dr_tweet_replies` |
| Tweet word count | `dr_tweet_wordcount` |
| Tweet timestamp hour | `dr_tweet_hour` |
| News article count (4H) | `dr_news_count_4h` |
| NFL/NBA final score (home) | `dr_sport_score_home` |
| NFL/NBA final score (away) | `dr_sport_score_away` |
| NFL/NBA combined score | `dr_sport_total` |
| NFL/NBA score differential | `dr_sport_diff` |
| Horse post position (winner) | `dr_horse_position` |
| Horse race winning time | `dr_race_time` |
| Horse odds (winner) | `dr_horse_odds` |
| Block number | `dr_block` |
| Whale tx amount | `dr_whale_amount` |
| Funding rate | `dr_funding` |
| Open interest | `dr_open_interest` |
| S&P 500 close | `dr_sp500` |
| NASDAQ close | `dr_nasdaq` |
| DXY | `dr_dxy` |
| Gold price | `dr_gold` |
| VIX | `dr_vix` |
| 10Y yield | `dr_10y` |
| CPI number | `dr_cpi` |
| Fear & Greed | `dr_fear_greed` |
| Reddit comment count | `dr_reddit_comments` |
| Date itself | `dr_date` (month+day+year) |
| Day of year | `dr_doy` |
| Days remaining | `dr_days_remaining` |

### Gematria × Everything (~200 features)
Each text target × 6 gematria methods:

| Target | Feature Names (×6 methods) |
|--------|---------------------------|
| Tweet full text | `gem_ord_tweet`, `gem_rev_tweet`, `gem_red_tweet`, `gem_eng_tweet`, `gem_jew_tweet`, `gem_sat_tweet` |
| Tweet username | `gem_ord_user`, `gem_rev_user`, `gem_red_user`, `gem_eng_user`, `gem_jew_user`, `gem_sat_user` |
| Tweet display name | `gem_ord_displayname`, ... ×6 |
| Tweet hashtags (combined) | `gem_ord_hashtags`, ... ×6 |
| News headline | `gem_ord_headline`, `gem_rev_headline`, `gem_red_headline`, `gem_eng_headline`, `gem_jew_headline`, `gem_sat_headline` |
| News source name | `gem_ord_source`, ... ×6 |
| News author name | `gem_ord_author`, ... ×6 |
| NFL/NBA home team | `gem_ord_team_home`, ... ×6 |
| NFL/NBA away team | `gem_ord_team_away`, ... ×6 |
| NFL/NBA winning team | `gem_ord_team_winner`, ... ×6 |
| NFL/NBA losing team | `gem_ord_team_loser`, ... ×6 |
| Stadium/venue name | `gem_ord_venue`, ... ×6 |
| MVP/star player | `gem_ord_mvp`, ... ×6 |
| Winning horse name | `gem_ord_horse_winner`, ... ×6 |
| Jockey name | `gem_ord_jockey`, ... ×6 |
| Trainer name | `gem_ord_trainer`, ... ×6 |
| Race name | `gem_ord_race`, ... ×6 |
| Track name | `gem_ord_track`, ... ×6 |
| Reddit post title | `gem_ord_reddit`, ... ×6 |
| FOMC statement (summary) | `gem_ord_fomc`, ... ×6 |
| Fed chair quote | `gem_ord_fed_quote`, ... ×6 |
| CPI report title | `gem_ord_cpi_title`, ... ×6 |

**Plus digital root of each gematria value** → another ~130 features:
| | |
|---|---|
| `dr_gem_ord_tweet` | DR of the ordinal gematria of the tweet text |
| `dr_gem_ord_headline` | DR of the ordinal gematria of the headline |
| etc. | Every gematria value gets a digital root |

**Plus gematria MATCH features** (~30 features):
| | |
|---|---|
| `gem_match_tweet_headline` | Does tweet gematria = headline gematria? |
| `gem_match_tweet_horse` | Does tweet gematria = winning horse gematria? |
| `gem_match_headline_team` | Does headline gematria = winning team gematria? |
| `gem_match_price_tweet` | Does price DR = tweet DR? |
| `gem_match_date_horse` | Does date DR = horse winner DR? |

### Astrology at Every Timestamp (~80 features)
Apply astrology not just to "now" but to the MOMENT of each event:

| Event Timestamp | Astrology Features |
|----------------|-------------------|
| Current candle close | `moon_phase`, `nakshatra`, `planetary_hour`, `mercury_retro`, `voc_moon`, `hard_aspects`, `soft_aspects`, `planetary_strength`, `vedic_tithi`, `vedic_yoga`, `bazi_stem`, `tzolkin_tone` |
| Tweet posted time | `tweet_moon_phase`, `tweet_nakshatra`, `tweet_planetary_hour`, `tweet_mercury_retro` |
| News published time | `news_moon_phase`, `news_planetary_hour`, `news_nakshatra` |
| Sports game time | `game_moon_phase`, `game_planetary_hour`, `game_nakshatra` |
| Horse race time | `race_moon_phase`, `race_planetary_hour`, `race_nakshatra` |
| Block mined time | `block_planetary_hour`, `block_nakshatra` |

### Sentiment × All Text (~30 features)
| Target | Features |
|--------|----------|
| Tweet text | `tweet_sentiment`, `tweet_caps`, `tweet_exclamation`, `tweet_urgency` |
| News headline | `headline_sentiment`, `headline_caps`, `headline_urgency` |
| Reddit title | `reddit_sentiment`, `reddit_caps` |
| FOMC statement | `fomc_sentiment` |
| Fed quote | `fed_sentiment` |

### Color × All Images (~15 features)
| Target | Features |
|--------|----------|
| Tweet images | `tweet_gold`, `tweet_red`, `tweet_green`, `tweet_dominant_color` |
| News article images | `news_gold`, `news_red` |

### Number Patterns × Everything (~40 features)
| Target | Features |
|--------|----------|
| BTC price | `price_contains_113`, `price_contains_322`, `price_contains_93`, `price_contains_213`, `price_contains_666`, `price_contains_777` |
| Tweet text | `tweet_contains_113`, `tweet_contains_322`, `tweet_mentions_numbers` |
| Sports score | `score_contains_33`, `score_is_master` |
| Block number | `block_contains_113`, `block_contains_322` |

### Cross-Features (most powerful — ~50 features)
Combinations of signals from different sources that amplify each other:

| Cross-Feature | Description |
|--------------|-------------|
| `moon_x_gold_tweet` | Moon phase × gold tweet present |
| `nakshatra_x_red_tweet` | Key nakshatra × red tweet |
| `mercury_retro_x_news_sentiment` | Mercury retro × negative news |
| `dr_date_x_dr_price` | Date DR matches price DR |
| `gem_match_cross_sources` | Gematria alignment across tweet + headline + horse |
| `eclipse_x_sports_upset` | Eclipse window × underdog win |
| `planetary_hour_x_tweet_time` | Jupiter hour × tweet at 3pm CST |
| `voc_moon_x_high_volume` | VOC moon × abnormal volume |
| `consec_green_x_caps_tweet` | 3+ green days × ALL CAPS tweet |
| `shmita_x_bear_regime` | Shmita year × EMA declining |
| `day13_x_full_moon` | Day 13 of month × full moon |
| `arabic_lot_conj_x_tweet` | Arabic lot conjunction × tweet gematria match |
| `friday13_x_red_tweet` | Friday 13th × red tweet |
| `master_number_x_nakshatra` | Master number in price × key nakshatra |
| `fg_extreme_x_moon_phase` | Fear/Greed extreme × moon phase |

### Regime & Trend (~10 features)
| Feature | Description |
|---------|-------------|
| `ema50_declining` | EMA50 slope < -3% over 20 periods |
| `ema50_rising` | EMA50 slope > 3% |
| `ema50_slope` | Continuous slope value |
| `hmm_bull_prob` | HMM bull regime probability |
| `hmm_bear_prob` | HMM bear regime probability |
| `hmm_neutral_prob` | HMM neutral probability |
| `wyckoff_phase` | Wyckoff accumulation/distribution |
| `current_dd_depth` | Current drawdown from peak |

---

## ESTIMATED TOTAL: ~600-700 features

| Category | Count |
|----------|-------|
| Numerology × all targets | ~125 |
| Gematria × all text (6 methods) | ~200 |
| Gematria DR + match features | ~160 |
| Astrology at all timestamps | ~80 |
| Sentiment × all text | ~30 |
| Color × all images | ~15 |
| Number patterns × all values | ~40 |
| Cross-features | ~50 |
| Regime & trend | ~10 |
| BTC technicals (existing) | ~80 |
| **TOTAL** | **~790** |

---

## BUILD ORDER (priority)

### Phase 1: Universal Engines (foundation)
- [ ] `universal_gematria.py` — takes any string, returns all 6 values + DR of each
- [ ] `universal_numerology.py` — takes any number, returns DR + master check + sequence detection
- [ ] `universal_astro.py` — takes any timestamp, returns full astrology snapshot (Western + Vedic + BaZi + Tzolkin + planetary hour)
- [ ] `universal_sentiment.py` — takes any text, returns sentiment + caps + exclamation + urgency

### Phase 2: Live Data Collectors
- [x] `tweet_streamer.py` — Twitter/X (CREATED)
- [ ] `news_streamer.py` — RSS feeds + CryptoPanic + Reddit (based on news_collector.py)
- [ ] `sports_streamer.py` — ESPN API + TheSportsDB horse racing (based on build_sports_features.py)
- [ ] `crypto_streamer.py` — blockchain data, funding, OI, whale txns
- [ ] `macro_streamer.py` — market indices, DXY, gold, VIX, yields

### Phase 3: Feature Builders (apply engines to all data)
- [ ] Update `build_4h_features.py` — apply universal engines to tweets, news, sports, horses
- [ ] Update `build_1h_features.py` — same
- [ ] Update `build_15m_features.py` — same (subset relevant at 15m resolution)
- [ ] All gematria × text features
- [ ] All numerology × number features
- [ ] All astrology × timestamp features
- [ ] All cross-features

### Phase 4: Training & Live
- [ ] Retrain with ALL ~700 features (increase shap_top_k, expand protected list)
- [ ] Update `live_trader.py` to compute ALL features live from all DBs
- [ ] Backtest full feature set across all BTC history
- [ ] Verify exotic features have measurable SHAP importance

### Phase 5: Dashboard
- [ ] Show active exotic signals on chart
- [ ] Show which features fired for each trade decision
- [ ] Live feed of tweets/news/sports with gematria annotations
