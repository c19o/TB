# How to Add New Data Points to Savage22

This guide explains how to add any new data source or signal into the ML trading system.

---

## The 4-Step Process

Every new data point follows the same pattern:

```
1. COLLECT → Get the raw data (API, scraper, database)
2. COMPUTE → Apply universal engines (gematria, numerology, astrology, sentiment)
3. BUILD   → Add features to the feature builders (historical training data)
4. LIVE    → Add features to live_trader.py (real-time predictions)
```

---

## Step 1: COLLECT — Get the Raw Data

### Option A: Add to an existing streamer

If your data source fits an existing category, add it to the relevant streamer:

| Data Type | File | Database |
|-----------|------|----------|
| Twitter/X accounts | `tweet_streamer.py` | `tweets.db` |
| News/RSS/Reddit | `news_streamer.py` | `news_articles.db` |
| Sports/Horse Racing | `sports_streamer.py` | `sports_results.db` |
| Crypto on-chain | `crypto_streamer.py` | `onchain_data.db` |
| Macro/indices | `macro_streamer.py` | `macro_data.db` |

**Example: Add a new Twitter account**
```python
# In tweet_streamer.py (or scrape_twitter.py), add to ACCOUNTS list:
ACCOUNTS = [
    "elonmusk",
    "NEW_ACCOUNT_HERE",  # ← just add the handle
    ...
]
```

### Option B: Create a new streamer

For a completely new data source:

1. Create `new_source_streamer.py` following the pattern of existing streamers
2. Create a new SQLite database (e.g., `new_source.db`)
3. Include columns for: timestamp, raw values, and pre-computed gematria/numerology

**Template:**
```python
#!/usr/bin/env python
import sqlite3, time, logging, os, requests
from universal_gematria import gematria, digital_root
from universal_numerology import numerology

DB_DIR = os.path.dirname(os.path.abspath(__file__))
log = logging.getLogger(__name__)

def init_db():
    conn = sqlite3.connect(f'{DB_DIR}/new_source.db')
    conn.execute("""CREATE TABLE IF NOT EXISTS data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT, ts_unix INTEGER,
        raw_value TEXT, raw_number REAL,
        gem_ordinal INTEGER, gem_dr INTEGER,
        value_dr INTEGER,
        inserted_at TEXT DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.commit()
    return conn

def fetch_data():
    # Your API call here
    resp = requests.get("https://api.example.com/data")
    return resp.json()

def run():
    conn = init_db()
    while True:
        try:
            data = fetch_data()
            # Process and insert
            for item in data:
                g = gematria(item['text'])
                n = numerology(item['value'])
                conn.execute("INSERT INTO data (...) VALUES (...)", (...))
            conn.commit()
        except Exception as e:
            log.error(f"Error: {e}")
        time.sleep(300)  # 5 minutes

if __name__ == "__main__":
    run()
```

---

## Step 2: COMPUTE — Apply Universal Engines

The universal engines take ANY input and return all computed values:

### Gematria (any text)
```python
from universal_gematria import gematria, gematria_flat, gematria_contains_target

# Full analysis
g = gematria("Kentucky Derby")
# Returns: {'ordinal': 164, 'reverse': 178, 'reduction': 47, 'english': 984,
#           'jewish': 553, 'satanic': 424, 'dr_ordinal': 2, ...}

# Flat dict for ML features (with prefix)
features = gematria_flat("Kentucky Derby", prefix='race')
# Returns: {'race_gem_ordinal': 164, 'race_gem_reverse': 178, ..., 'race_gem_is_caution': 0}

# Check for target numbers
targets = gematria_contains_target("Kentucky Derby")
# Returns: [('reduction', 47, 'caution')] if matches
```

### Numerology (any number)
```python
from universal_numerology import numerology, digital_root, date_numerology, numerology_flat

# Full analysis
n = numerology(73954)
# Returns: {'dr': 1, 'is_master': False, 'contains_113': False, ...}

# Flat dict for ML
features = numerology_flat(73954, prefix='price')
# Returns: {'price_dr': 1, 'price_is_master': 0, 'price_contains_113': 0, ...}

# Date analysis
d = date_numerology(datetime(2026, 3, 18))
# Returns: {'day_of_year': 77, 'dr_date': 4, 'is_caution_date': True, ...}
```

### Astrology (any timestamp)
```python
from universal_astro import astro_snapshot, astro_flat

# Full analysis
a = astro_snapshot(datetime(2026, 3, 18, 15, 30))
# Returns: {'moon_phase': 'waning_crescent', 'nakshatra': 22, 'planetary_hour': 'Saturn',
#           'bazi_stem': 7, 'tzolkin_tone': 3, ...}

# Flat dict for ML
features = astro_flat(datetime.now(), prefix='event')
# Returns: {'event_moon_phase_day': 29.51, 'event_nakshatra': 22, ...}
```

### Sentiment (any text)
```python
from universal_sentiment import sentiment, sentiment_flat

# Full analysis
s = sentiment("Bitcoin crashes! SELL NOW!!!")
# Returns: {'score': -2, 'bull_count': 0, 'bear_count': 2, 'has_caps': True, ...}

# Flat dict for ML
features = sentiment_flat("Bitcoin crashes!", prefix='headline')
# Returns: {'headline_sentiment': -2, 'headline_has_caps': 0, ...}
```

---

## Step 3: BUILD — Add to Feature Builders

Add your new features to the feature builder files so they're included in historical training data.

### Where to add code

Each feature builder has a section labeled `UNIVERSAL ENGINE FEATURES`. Add your new data source query there.

| Timeframe | File | Bucket Size |
|-----------|------|-------------|
| 15m | `build_15m_features.py` | 86400 (daily) |
| 1h | `build_1h_features.py` | 3600 (1 hour) |
| 4h | `build_4h_features.py` | 14400 (4 hours) |

### Pattern for adding a new data source

```python
# --- Your new data source (from new_source.db) ---
try:
    conn = sqlite3.connect(f'{DB_DIR}/new_source.db', timeout=5)
    data = pd.read_sql_query("SELECT ts_unix, raw_value, gem_ordinal, gem_dr, value_dr FROM data ORDER BY ts_unix", conn)
    conn.close()

    if len(data) > 0:
        data['ts_unix'] = pd.to_numeric(data['ts_unix'], errors='coerce')
        data['bucket'] = (data['ts_unix'] // BUCKET_SIZE) * BUCKET_SIZE

        agg = data.groupby('bucket').agg(
            new_source_gem_dr=('gem_dr', 'last'),
            new_source_value_dr=('value_dr', 'last'),
            new_source_count=('raw_value', 'count'),
        ).reset_index()

        bucket_map = agg.set_index('bucket')
        for col in agg.columns:
            if col == 'bucket': continue
            mapped = df_buckets.map(bucket_map[col].to_dict())
            df[col] = mapped.fillna(0)

        print(f"    New source features: {len(agg.columns)-1} columns mapped")
except Exception as e:
    print(f"    New source features skipped: {e}")
```

---

## Step 4: LIVE — Add to live_trader.py

Add the same feature computation to `live_trader.py` so the model gets these features in real-time.

### Where to add code

In the `compute_features` function (around line 130-450), add after the existing feature blocks:

```python
    # --- Your new data source (live) ---
    try:
        conn = sqlite3.connect(f'{DB_DIR}/new_source.db', timeout=5)
        row = conn.execute(
            "SELECT gem_dr, value_dr FROM data ORDER BY ts_unix DESC LIMIT 1"
        ).fetchone()
        if row:
            features['new_source_gem_dr'] = row[0] or 0
            features['new_source_value_dr'] = row[1] or 0
        conn.close()
    except Exception:
        features['new_source_gem_dr'] = 0
        features['new_source_value_dr'] = 0
```

---

## Step 5: PROTECT — Add to Protected Features List

In `ml_multi_tf.py`, add your new feature names to the `PROTECTED_FEATURES` list so they survive SHAP pruning:

```python
PROTECTED_FEATURES = [
    ...
    # Your new source
    'new_source_gem_dr', 'new_source_value_dr', 'new_source_count',
    ...
]
```

---

## Step 6: RETRAIN

After adding new features:

```bash
# 1. Rebuild features (includes new data)
python build_15m_features.py
python build_1h_features.py
python build_4h_features.py

# 2. Retrain models
python ml_multi_tf.py

# 3. Verify new features are in the model
python -c "import json; f=json.load(open('features_4h_pruned.json')); print([x for x in f if 'new_source' in x])"

# 4. Restart trader
python live_trader.py --mode paper
```

---

## Quick Reference: What Technique to Apply to What

| Data Type | Apply Gematria? | Apply Numerology? | Apply Astrology? | Apply Sentiment? |
|-----------|----------------|-------------------|-----------------|-----------------|
| Text (tweets, headlines, names) | YES (all 6 methods) | On word count | At timestamp | YES |
| Numbers (prices, scores, counts) | No | YES (DR, sequences, master) | No | No |
| Timestamps (when events happen) | No | YES (date numerology) | YES (full snapshot) | No |
| Images | No (use color detection) | No | No | No |
| Binary events (retro, eclipse) | No | No | Already astro | No |

---

## Example: Adding UFC Fight Results

1. **COLLECT**: Create `ufc_streamer.py` that polls UFC API for fight results
2. **COMPUTE**: `gematria("Conor McGregor")` → ordinal, DR, caution check
3. **BUILD**: Query `ufc.db` in feature builders, bucket by 4H, aggregate winner gematria DR
4. **LIVE**: Query latest fight result in `live_trader.py`
5. **PROTECT**: Add `ufc_winner_gem_dr`, `ufc_fights_today` to PROTECTED_FEATURES
6. **RETRAIN**: Rebuild features, retrain models

The model will learn if UFC fight winner gematria correlates with BTC price movement. If it does → it gets high SHAP importance → it influences predictions. If it doesn't → low SHAP → it's still included (protected) but has minimal weight.
