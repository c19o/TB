# HEARTBEAT — Collective Consciousness Field Reader for BTC Trading

## Concept

Use existing global sensor networks that measure collective consciousness / field anomalies
as a data feed into the trading system. This is essentially digital radionics — hardware RNG
networks and magnetometers reading the "ether" and outputting numbers we can ingest.

No hardware purchase needed. Multiple organizations have been running these sensors for decades
and publish live data.

---

## Data Sources

### 1. Global Consciousness Project (GCP) — Princeton RNG Network

- **What**: 60-70 hardware random number generators distributed worldwide, running since 1998
- **Theory**: When mass consciousness shifts (fear, euphoria, collective attention), the RNG outputs
  deviate from expected randomness. The deviation IS the signal.
- **Data format**: CSV — one record per second per egg. 200-bit trials, bit count sums.
- **CSV spec**: https://noosphere.princeton.edu/basket_CSV_v2.html
  - Record type 13 = actual data rows
  - Field 1: type code (13)
  - Field 2: Unix timestamp
  - Field 3: civil time (yyyy-mm-dd hh:mm:ss), optional
  - Fields 4+: sample values (bit counts from each egg's trial)
  - Missing data = empty fields (NOT zero)
- **Historical data extract**: https://noosphere.princeton.edu/extract.html
  - Custom date range pulls, CSV download
  - Database spans Aug 4, 1998 to present
- **Real-time endpoint (scrapable)**:
  ```
  http://gcpdot.com/gcpindex.php?small=1
  ```
  Returns decimal values every minute. Higher values = more deviation from randomness.
- **Real-time viewer**: https://gcpdot.com/realtime/
  - Has CSV export button for raw trial data + per-second/per-minute computations
  - Normally runs with 10-20 min delay from current time
- **Advanced/bulk data**: https://noosphere.princeton.edu/wget.html
  - Multi-day bulk downloads, may need to contact director
- **External analysis tools**:
  - Eggshell (Fourmilab): https://www.fourmilab.ch/eggtools/eggshell/
  - EggAnalysis (Windows app): https://www.treurniet.ca/GCP/

#### GCP-to-VIX Correlation (Published Research)

- Paper: "A novel market sentiment measure: assessing the link between VIX and the Global
  Consciousness Project's data" — Journal of Economic Studies, Vol 51 Issue 7, 2024
- Published by Emerald Publishing (peer reviewed)
- Finding: **Significant covariation between GCP Max[Z] values and the VIX** (S&P 500 volatility/fear index)
- Max[Z] = the largest daily composite GCP data value
- URL: https://www.emerald.com/insight/content/doi/10.1108/jes-11-2023-0663/full/html

#### Node.js API Wrapper (Proven Scrapable)

```javascript
// From: https://gist.github.com/quartzjer/6a3270e0252a572236ad
var request = require('request');
var express = require('express');
var app = express();

app.get('/', function(req, res){
  request.get("http://gcpdot.com/gcpindex.php?small=1", function(err, r, body){
    if(err || r.statusCode != 200 || typeof body != "string") return res.send("000000");
    var high = 0;
    body.match(/(0\.\d+)/g).forEach(function(gcp){ if(gcp > high) high = gcp });
    // Returns hex color based on deviation level
    // Values closer to 0.5 = more random (normal)
    // Values far from 0.5 = deviation (signal)
    res.send(high.toString());
  });
});
```

### 2. HeartMath Global Coherence Monitoring System (GCMS)

- **What**: Worldwide network of magnetometers measuring Earth's magnetic field
- **Measures**: Schumann resonances (0.32 to 36 Hz), geomagnetic field strength
- **Live data**: https://www.heartmath.org/gci/gcms/live-data/gcms-magnetometer/
- **Update frequency**: Hourly spectrograms, 24-hour moving averages
- **Also hosts GCP 2.0**: https://www.heartmath.org/gci/gcms/live-data/global-consciousness-project/
- **Live GCP map (new network)**: https://gcp2.net/data-results/live-data

### 3. Schumann Resonance Live Feeds

- https://schumannresonancelive.com/ — 24/7 live graph
- https://schumannresonancedata.com/ — real-time analytics
- https://geocenter.info/en/monitoring/schumann — Russian monitoring station
- https://meteoagent.com/schumann-resonance-forecast — forecast data

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│           FIELD SENSORS (already exist)          │
│                                                  │
│  GCP RNG Network ──→ deviation from randomness   │
│  HeartMath GCMS  ──→ Schumann/geomagnetic data   │
│  Schumann feeds  ──→ Earth resonance frequency    │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│           DATA INGESTION LAYER                   │
│                                                  │
│  gcp_scraper.py                                  │
│    - Hits gcpdot.com/gcpindex.php?small=1        │
│      every 60 seconds                            │
│    - Parses deviation values                     │
│    - Stores in feature DB                        │
│                                                  │
│  heartmath_scraper.py                            │
│    - Scrapes GCMS magnetometer page hourly       │
│    - Extracts Schumann power levels              │
│    - Stores in feature DB                        │
│                                                  │
│  gcp_historical_loader.py                        │
│    - Bulk loads 1998-present CSV data            │
│    - For backtesting against BTC price history   │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│        EXISTING FEATURE MATRIX                   │
│                                                  │
│  Space weather (Kp, solar flux, etc.)            │
│  Lunar cycles (already in system)                │
│  Astrology (already in system)                   │
│  Sentiment (already in system)                   │
│  On-chain metrics                                │
│  Technical indicators                            │
│  + NEW: GCP deviation signal                     │
│  + NEW: Schumann resonance power                 │
│  + NEW: HeartMath coherence index                │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│           ML TRADER (live_trader.py)             │
│                                                  │
│  All features fed into model ensemble            │
│  GCP deviation = collective fear/euphoria proxy  │
│  Schumann = Earth field energy state             │
│  Combined = "field reading" signal               │
└─────────────────────────────────────────────────┘
```

---

## What The GCP Signal Means For Trading

- **High deviation from randomness** = collective consciousness is "activated"
  - Could mean mass fear (crash incoming) or mass euphoria (pump incoming)
  - Cross-correlate with sentiment to determine direction
  - The GCP tells you INTENSITY, sentiment tells you DIRECTION

- **Normal randomness** = business as usual, no major collective energy shift
  - Low-volatility regime, mean reversion strategies

- **The VIX correlation**: GCP Max[Z] covaries with VIX. VIX = fear. When collective
  consciousness spikes AND fear sentiment is present, expect volatility expansion.

---

## Radionics Theory (From Vector DB Library)

Key books indexed in the Orgonite master vector DB:

### Directly Relevant
- **Magic Power of Radionics Psionics and Orgone.pdf** (247 chunks)
  - Radionic rate = targeting system that locks onto a target
  - Witness well = introduces interference into the circuit
  - Dial tuning = achieving resonance with the target
  - The machine amplifies the operator's intent
- **combined_text.txt** (8944 chunks)
  - Contains circuit schematics, DAC interfaces, printer port hookups
  - Someone built a radionics system with Visual Studio 6 + scalar wave generators
  - Goal: "get a sample of a radionic data pattern into the computer"
  - 10-bit DAC with software-controlled square/sine wave output
- **Jon Logan - ZPE Orgone** — radionic circuit board schematics, function generators

### Financial Occult
- **Book of Shadows Complete.pdf** — Blue magic (wealth belief modification), money spells,
  servitor creation for financial tasks, talisman charging
- **Crowley - Magick in Theory and Practice.pdf** — Money as talisman
- **Picatrix** — Astrological talismans for wealth
- **Goetia / Key of Solomon** — Spirits associated with treasure/wealth finding
- **Wattles - Science of Getting Rich** — Manifestation for wealth
- **Ellen Brown - Web of Debt** — Banking system mechanics
- **David Astle - Babylonian Woe** — Temple banking, ancient monetary control

### Mind Reading / Reading People & Groups
- **Complete Hypnotism Mesmerism Mind-Reading.txt** — Practical telepathy techniques
- **Ascension Glossary** — Telepathy, collective consciousness, hive mind mechanics,
  "field influence can easily take over individual and group behavior"
- **Neville Goddard - Feeling is the Secret** — Subconscious programming, state assumption
- **Neville Goddard - Law and the Promise** — Mental revision technique
- **Vadim Zeland - Reality Transurfing** — Pendulums (group energy structures), directly
  applicable to market crowd psychology
- **CIA Gateway Process Report** — Consciousness exploration, Monroe Institute protocols
- **Robert Monroe - Journeys Out of the Body** — Non-objective signals influencing people

### Matrixology / Number Energy (Already In System)
- **BTC Energy.txt** — Bitcoin = 213, BTC = 223, dates matching number energy
  correlate with major BTC moves. Located at:
  `C:\Users\C\Desktop\MY GOOGLE DRIVE\Orgonite master\drop_here\financial\BTC Energy.txt`
- **matrixology_notes.txt** (indexed in vector DB) — Bullish within 5 days of new moon,
  bearish within 5 days of full moon, 6am UTC = short term reversal point

### Financial Astrology
- **Ptolemy - Tetrabiblos** (p.195) — Sepharial's "Law of Values" — planetary causes
  of stock/share fluctuations
- **Vettius Valens - Anthologies** — Chronocrator timing techniques for market cycles
- **William Lilly - Christian Astrology (3 vols)** — Horary = yes/no trade questions

---

## Vector DB Location

```
C:\Users\C\Desktop\MY GOOGLE DRIVE\Orgonite master\
├── chroma_db/          — ChromaDB vector embeddings (all-MiniLM-L6-v2)
├── orgone_kb.db        — SQLite FTS5 full-text search (BM25 ranking)
├── search_engine.py    — BM25 search interface
├── vector_db.py        — ChromaDB semantic search interface
├── config.py           — Topic keywords, source dirs, chunk settings
├── database.py         — Core DB operations
├── kb.py               — Knowledge base CLI
├── drop_here/          — Source PDFs organized by topic
│   ├── financial/      — Money/banking/trading books
│   ├── magick/         — Grimoires, ceremonial magic
│   ├── mind_power/     — Telepathy, manifestation, consciousness
│   ├── esoteric/       — Hermetics, alchemy, Kabbalah, astrology
│   ├── freemasonry/    — Secret societies, power structures
│   └── matrix/         — Reality/simulation, consciousness research
```

- **342 documents indexed**, **149,914 text chunks**, **2.73 GB** total
- Topics auto-tagged by keyword matching (see config.py TOPIC_KEYWORDS)
- Vector search: `from vector_db import vector_search; vector_search("query", limit=10)`
- FTS search: `from search_engine import do_search; do_search("query", limit=10)`

---

## CORRELATION RESULTS (Completed 2026-03-20)

### Dataset
- **61,026 GCP hourly data points** (2019-01-01 to 2025-12-14)
- **54,955 BTC 1h candles** from local `btc_prices.db`
- **54,684 paired hours** with both GCP + BTC data
- GCP parsed data cached at: `heartbeat_data/results/gcp_hourly_cache.json`
- Full paired CSV at: `heartbeat_data/results/gcp_btc_correlation.csv`
- Full paired JSON at: `heartbeat_data/results/gcp_btc_correlation.json`

### Pearson Correlation
| Test | r | t-stat | Significant? |
|------|---|--------|-------------|
| GCP deviation vs BTC direction | -0.001 | -0.24 | NO |
| GCP deviation vs BTC volatility | +0.008 | +1.85 | Almost |
| GCP deviation vs BTC volume | +0.002 | +0.47 | NO |
| GCP max_var vs BTC volatility | **-0.107** | **-25.12** | **YES p<0.001** |
| GCP max_var vs BTC volume | **-0.188** | **-44.65** | **YES p<0.001** |
| GCP max_var vs BTC range | **-0.073** | **-17.13** | **YES p<0.001** |

### Quintile Analysis (GCP deviation buckets -> BTC behavior)
| Bucket | GCP dev | BTC |chg| | BTC range | Volume | Directional bias |
|--------|---------|-----------|-----------|--------|-----------------|
| Q1 LOWEST | -0.0093 | 0.386% | 0.782% | 17 | +0.009% |
| Q2 LOW | -0.0033 | 0.399% | 0.821% | 22 | -0.000% |
| Q3 MIDDLE | +0.0000 | 0.409% | 0.841% | 22 | +0.007% |
| Q4 HIGH | +0.0033 | 0.412% | 0.839% | 23 | -0.003% |
| Q5 HIGHEST | +0.0102 | 0.381% | 0.811% | 18 | -0.002% |

### Directional Analysis — DOES NOT PREDICT DIRECTION ALONE
- High vs Low deviation: 50.1% vs 50.6% bullish — **no edge**
- Rising vs Falling field: 50.3% vs 50.5% — **no edge**
- All lags (1h through 24h): 50.0-50.7% — **no edge at any lag**

### Extreme Events (Top 1% deviation) — SLIGHT BULLISH BIAS
- **539 extreme GCP events** found
- BTC next 24h: **289 up / 250 down (53.6% bullish)**
- Average 24h cumulative move: **+0.15%**
- Notable: New Year's Eve/Day 2020 clustered extreme events

### KEY FINDING
**GCP is a VOLATILITY signal, not a directional signal.**
- Triple-star significance on volatility/volume correlation
- When collective consciousness spikes (max_var), BTC vol/volume DECREASES
- When field is "calm", BTC gets wild
- Use as REGIME DETECTOR: "something big is about to happen"
- Must combine with directional signals (sentiment, on-chain) for trade direction

### NEXT STEP: Find the DIRECTIONAL esoteric signal

The GCP tells you WHEN (volatility regime), but NOT which way. Deep research completed
across Perplexity (academic), vector DB (342 books), and existing feature DBs.

#### EXISTING DIRECTIONAL FEATURES ALREADY IN THE SYSTEM (~300 esoteric features in features_1h.db)
The system ALREADY HAS these directional esoteric features built:
- Lunar phase sin/cos, days to full/new moon, moon decay features
- Mercury/Venus/Mars retrograde flags
- Hard/soft aspect counts, planetary strength index (PSI)
- Vedic nakshatra, tithi, yoga, guna
- Chinese BaZi day stem/branch/element/clash
- Mayan Tzolkin tone/sign
- Arabic Lots (commerce, increase, catastrophe, treachery)
- Matrixology: pump dates (14/15/16), #17/#19 dump dates, #113 bottom buy, #93/39 destruction
- BTC 213 energy dates, date palindromes
- Space weather: Kp, solar wind, solar flux, storm flags
- Schumann resonance cycles (133d, 143d, 783d)
- Eclipse windows, VOC moon
- Jewish calendar windows (Shemitah, Yom Kippur, etc.)
- 150+ cross-features (moon x sentiment, nakshatra x tweets, etc.)

#### BACKTEST RESULTS ALREADY FOUND (astrology_backtest_results.txt)
- **Moon South Node conjunction**: +0.68% edge, 61.5% win rate (n=39) <-- BEST SIGNAL
- **PSI strongly negative**: +0.27% edge, 55.1% win rate (n=156)
- **Mars retrograde**: +0.31% edge (n=72)
- **Saturn station**: +0.27% edge (n=24)
- **Tuesday**: +0.32% edge (n=112)
- **Venus retrograde**: -0.42% edge (BEARISH, n=57)
- **Pump dates 14/15/16**: -0.61% edge (BEARISH, n=78) <-- opposite of expected!
- **VOC Moon filter**: avoiding VOC trades improves returns +0.05%

#### DIRECTIONAL SIGNALS FROM VECTOR DB (books)
1. **Matrixology directional rules** (matrixology_notes.txt):
   - New moon = bullish (+5 day window), Full moon = bearish (+5 day window)
   - Mon-Wed = pump days (Tue = top), Thu-Sat = dump days (Sat = bottom)
   - 6am UTC = bottom, 12pm UTC = top, midnight = reversal
   - #17, #19 = dumps; #11 = emotional/dump; #113 = bottom buy
2. **Radin (Conscious Universe)**: Full moon = higher casino payouts (enhanced psi),
   geomagnetic storms = NEGATIVE correlation with luck/judgment (bearish mood)
3. **Planetary aspect direction** (Perplexity + astrology books):
   - Jupiter/Venus/trine/sextile = bullish energy
   - Saturn/Mars/square/opposition = bearish energy
   - Saturn hard aspects to BTC natal chart = bearish (documented in BTC astrology)
   - Sun-Jupiter conjunction/sextile/trine = bullish (especially if followed by soft outer planet aspect)
4. **Kybalion pendulum law**: Extreme moves predict opposite swing of equal magnitude
5. **Geomagnetic direction**: High Kp = bearish mood/poor judgment (Radin + space weather data)

#### GCP + EXISTING FEATURES = THE COMBO SIGNAL
The key insight: GCP alone doesn't predict direction, but it AMPLIFIES.
When GCP deviation is extreme AND:
- Sentiment is fearful -> BEARISH (high confidence)
- Sentiment is greedy -> BULLISH (high confidence)
- Moon South Node conjunction -> strong directional bias
- High Kp storm -> bearish amplification
- Jupiter soft aspects -> bullish amplification

**ACTION FOR NEXT SESSION:**
1. Run GCP deviation as a NEW feature into feature pipeline
2. Create GCP x sentiment cross-features (the Holmberg paper showed GCP + VIX interaction
   predicted S&P 500 direction with R^2 improvement from 0.341 to 0.352)
3. Create GCP x existing esoteric cross-features
4. Retrain model with GCP features included
5. Focus on the Moon South Node conjunction signal (61.5% win rate) — best directional edge found

#### REFERENCE: Holmberg 2023 Paper
"Market sentiment and the Global Consciousness Project's data" by Ulf Holmberg
- GCP Max[Z] + VIX interaction = directional S&P 500 predictor
- Out-of-sample: 5.1-13.9% outperformance over 1 year
- Key: GCP is NOT standalone — it works through INTERACTION with sentiment
- PDF: https://ulfholmberg.info/onewebmedia/Market%20sentiment%20and%20the%20Global%20Consciousness%20Projects%20data.pdf

---

## Implementation Steps

### Phase 1: GCP Integration Into Trading System (READY NOW)
- GCP data: 2,548 daily CSV files in `heartbeat_data/gcp/`
- Parsed hourly stats cached in `heartbeat_data/results/gcp_hourly_cache.json`
- BTC correlation data in `heartbeat_data/results/gcp_btc_correlation.csv`
- Script: `heartbeat_download.py` — downloads, parses, correlates
- **To add as feature**: write `gcp_feature_builder.py` that:
  1. Reads `gcp_hourly_cache.json`
  2. Computes rolling deviation stats (1h, 4h, 24h windows)
  3. Creates `gcp_deviation_mean`, `gcp_deviation_max`, `gcp_rate_of_change` columns
  4. Inserts into feature DB alongside existing features
  5. For live trading: scrape `gcpdot.com/gcpindex.php?small=1` every 60s

### Phase 2: Real-Time GCP Scraper (For Live Trading)
- Endpoint: `http://gcpdot.com/gcpindex.php?small=1`
- Returns decimal values every minute
- Write `gcp_live_scraper.py` that runs alongside live_trader.py
- Store in feature DB same schema as historical

### Phase 3: HeartMath / Schumann Integration
- Scrape `heartmath.org/gci/gcms/live-data/gcms-magnetometer/` hourly
- Add Schumann resonance power as feature
- Cross-correlate with existing Kp/solar features

### Phase 4: Directional Signal Discovery
- Run correlation analysis on all esoteric features vs BTC direction
- Test matrixology dates, planetary aspects, lunar+GCP combos
- Find the directional signal to pair with GCP volatility signal

### Phase 5: Claude as Interpreter (Future)
- Feed full field state to Claude
- Vector DB (342+ books) as "rate book" for interpretation
- Qualitative "field reading" supplements ML signal

---

## Files Created In This Session

| File | Location | Purpose |
|------|----------|---------|
| `heartbeat.md` | Savage22 Server root | This document — full project reference |
| `heartbeat_download.py` | Savage22 Server root | GCP downloader + parser + BTC correlator |
| `heartbeat_data/gcp/` | 2,548 CSV files | Raw GCP RNG data (2019-2025) |
| `heartbeat_data/results/gcp_hourly_cache.json` | ~61K entries | Parsed hourly deviation stats (CACHED) |
| `heartbeat_data/results/gcp_btc_correlation.json` | ~55K entries | Paired GCP+BTC hourly data |
| `heartbeat_data/results/gcp_btc_correlation.csv` | ~55K rows | Same as above, CSV format |

## Existing Data Already In Savage22 Server

| Database | Contents |
|----------|----------|
| `btc_prices.db` | 3.8M OHLCV rows, all timeframes (used for this analysis) |
| `features_1h.db` | Hourly feature matrix |
| `features_4h.db` | 4-hour feature matrix |
| `features_1d.db` | Daily feature matrix |
| `astrology_full.db` | Astrological features |
| `fear_greed.db` | Fear & greed index |
| `onchain_data.db` | On-chain metrics |
| `funding_rates.db` | Funding rate data |
| `google_trends.db` | Google trends data |
| `macro_data.db` | Macro economic data |

## Vector DB (Esoteric Research Library)

Location: `C:\Users\C\Desktop\MY GOOGLE DRIVE\Orgonite master\`
- **342+ documents**, **149,914+ text chunks**, **2.73 GB**
- ChromaDB (semantic) + SQLite FTS5 (keyword)
- Key new additions: Dean Radin books (Conscious Universe, Entangled Minds) — NOW INDEXED
- Radin documents PEAR lab proof that consciousness affects RNG output
- This is the scientific foundation for why GCP data correlates with market activity

---

## Key URLs

- GCP Real-time scrape: http://gcpdot.com/gcpindex.php?small=1
- GCP Historical CSV: https://noosphere.princeton.edu/extract.html
- GCP CSV format spec: https://noosphere.princeton.edu/basket_CSV_v2.html
- GCP Bulk download: https://noosphere.princeton.edu/wget.html
- GCP Live viewer: https://gcpdot.com/realtime/
- GCP 2.0 Live map: https://gcp2.net/data-results/live-data
- HeartMath Live: https://www.heartmath.org/gci/gcms/live-data/
- HeartMath Schumann: https://www.heartmath.org/gci/gcms/live-data/gcms-magnetometer/
- Schumann Live: https://schumannresonancelive.com/
- VIX-GCP Paper: https://www.emerald.com/insight/content/doi/10.1108/jes-11-2023-0663/full/html
- Node.js wrapper: https://gist.github.com/quartzjer/6a3270e0252a572236ad
