# AUDIT FIX PLAN -- Savage22 Trading System
## Generated 2026-03-21 from 10-Agent Audit

---

## FIX 1: 5m/15m SQL Missing Columns
**Priority: P0 (blocks training -- features built without quote_volume/trades/taker_buy_quote)**

### Files
- `C:\Users\C\Documents\Savage22 Server\build_5m_features.py` line 47
- `C:\Users\C\Documents\Savage22 Server\build_15m_features.py` line 47

### Problem
The SQL SELECT loads only `open_time, open, high, low, close, volume, taker_buy_volume` but 1H correctly loads `quote_volume, trades, taker_buy_quote` as well. The meta_cols list at line 356 references these columns, confirming they are expected to exist. Without them, any feature_library.py code that uses these columns silently gets NaN.

### Old Code (both files, line 46-50)
```python
btc = pd.read_sql_query("""
    SELECT open_time, open, high, low, close, volume, taker_buy_volume
    FROM ohlcv WHERE timeframe='5m' AND symbol='BTC/USDT'
    ORDER BY open_time
""", conn)
```

### New Code (both files, line 46-50)
```python
btc = pd.read_sql_query("""
    SELECT open_time, open, high, low, close, volume, quote_volume, trades,
           taker_buy_volume, taker_buy_quote
    FROM ohlcv WHERE timeframe='5m' AND symbol='BTC/USDT'
    ORDER BY open_time
""", conn)
```
(Change `'5m'` to `'15m'` for the 15m file.)

### Also update the numeric coercion loop (line 54 in both files)
**Old:**
```python
for col in ['open', 'high', 'low', 'close', 'volume', 'taker_buy_volume']:
```
**New:**
```python
for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote']:
```

### Verification
After fix, run:
```python
import sqlite3, pandas as pd
conn = sqlite3.connect('btc_prices.db')
df = pd.read_sql_query("SELECT open_time, open, high, low, close, volume, quote_volume, trades, taker_buy_volume, taker_buy_quote FROM ohlcv WHERE timeframe='5m' LIMIT 5", conn)
print(df.columns.tolist())  # Must show all 10 columns
conn.close()
```
Then rebuild features_5m and features_15m and confirm `quote_volume`, `trades`, `taker_buy_quote` columns are NOT all NaN.

---

## FIX 2: Timestamped Onchain Data Not Used in Builds
**Priority: P0 (blocks training -- whale_volume, liquidations, coinbase_premium all missing)**

### Files (all 7 build scripts)
- `build_5m_features.py` lines 142-147
- `build_15m_features.py` lines 142-147
- `build_1h_features.py` lines 145-150
- `build_4h_features.py` lines 166-169 (similar)
- `build_1d_features.py` lines 126-129
- `build_1w_features.py` lines 126-129

### Problem
All build scripts only query `blockchain_data` table (daily: n_transactions, hash_rate, difficulty, mempool_size, miners_revenue). The `onchain_data` table (timestamped: block_height, block_dr, funding_rate, funding_dr, open_interest, oi_dr, fear_greed, fg_dr, whale_volume_btc, liq_long_count, liq_short_count, liq_long_vol, liq_short_vol, coinbase_premium) is never loaded.

`data_access.py` OfflineDataLoader.load_onchain() (line 192) returns BOTH as `{'daily': ..., 'timestamped': ...}`. The live path already loads both (lines 424-468). Build scripts only load daily.

### Old Code (example from build_5m_features.py lines 141-147)
```python
# On-chain
conn = sqlite3.connect(f'{DB_DIR}/onchain_data.db')
onchain_df = pd.read_sql_query("SELECT * FROM blockchain_data ORDER BY date", conn)
conn.close()
onchain_df['date'] = pd.to_datetime(onchain_df['date'])
onchain_df = onchain_df.drop_duplicates(subset='date', keep='last').set_index('date')
print(f"  On-chain: {len(onchain_df)}")
```

### New Code (for all 7 build scripts)
```python
# On-chain (daily blockchain + timestamped onchain)
conn = sqlite3.connect(f'{DB_DIR}/onchain_data.db')
onchain_df = pd.read_sql_query("SELECT * FROM blockchain_data ORDER BY date", conn)
onchain_ts_df = pd.read_sql_query("""
    SELECT timestamp, block_height, block_dr, funding_rate,
           funding_dr, open_interest, oi_dr, fear_greed, fg_dr,
           mempool_size, whale_volume_btc,
           liq_long_count, liq_short_count, liq_long_vol, liq_short_vol,
           coinbase_premium
    FROM onchain_data ORDER BY timestamp
""", conn)
conn.close()
onchain_df['date'] = pd.to_datetime(onchain_df['date'])
onchain_df = onchain_df.drop_duplicates(subset='date', keep='last').set_index('date')
print(f"  On-chain daily: {len(onchain_df)}")
print(f"  On-chain timestamped: {len(onchain_ts_df)}")
```

### Also update the esoteric_frames dict (around line 290 in each file)
**Old:**
```python
esoteric_frames = {
    ...
    'onchain': onchain_df,
    ...
}
```
**New:**
```python
esoteric_frames = {
    ...
    'onchain': {'daily': onchain_df, 'timestamped': onchain_ts_df},
    ...
}
```
This matches the dict format returned by `OfflineDataLoader.load_onchain()` and expected by `LiveDataLoader.get_onchain()` (line 625-632 in data_access.py).

### Impact on feature_library.py
Need to verify that `build_all_features()` handles `esoteric_frames['onchain']` as a dict. Search for how it unpacks onchain data. If it currently expects a DataFrame, it will need updating too. The live path (data_access.py line 625-632) already returns a dict.

### Verification
After fix, rebuild any TF and grep the output parquet for columns containing `whale`, `liq_`, `coinbase`, `oi_dr` -- they should have non-NaN values.

---

## FIX 3: News Schema Inversion
**Priority: P1 (data still loads, but loads wrong/incomplete table first)**

### Files (5 of 7 build scripts -- 4h is different, see below)
- `build_5m_features.py` lines 107-112
- `build_15m_features.py` lines 107-112
- `build_1h_features.py` lines 111-115
- `build_1d_features.py` lines 93-95
- `build_1w_features.py` lines 93-95

### Problem
Build scripts try `articles` first, fall back to `streamer_articles`. But `data_access.py` (line 137-154) correctly tries `streamer_articles` first (richer schema with gematria_ordinal, gematria_reverse, gematria_reduction, sentiment_bull, sentiment_bear, has_caps, exclamation_count, word_count) and falls back to `articles`. The build scripts get the poorer schema when the richer one exists.

### Old Code (lines 107-112 in 5m/15m)
```python
conn = sqlite3.connect(f'{DB_DIR}/news_articles.db')
try:
    news_df = pd.read_sql_query("SELECT timestamp, ts_unix, title, title_gematria, title_dr, sentiment_score, date_doy FROM articles ORDER BY timestamp", conn)
except:
    news_df = pd.read_sql_query("SELECT timestamp, ts_unix, title FROM streamer_articles ORDER BY timestamp", conn)
conn.close()
```

### New Code
```python
conn = sqlite3.connect(f'{DB_DIR}/news_articles.db')
try:
    news_df = pd.read_sql_query("""
        SELECT timestamp, ts_unix, title, sentiment_score, title_dr,
               title_gematria_ordinal, title_gematria_reverse,
               title_gematria_reduction, sentiment_bull, sentiment_bear,
               has_caps, exclamation_count, word_count
        FROM streamer_articles ORDER BY timestamp
    """, conn)
except:
    news_df = pd.read_sql_query("SELECT timestamp, ts_unix, title, title_gematria, title_dr, sentiment_score, date_doy FROM articles ORDER BY timestamp", conn)
conn.close()
```

### Special case: build_4h_features.py (lines 124-129)
This file only loads from `articles` with no try/except fallback. Apply the same fix: try `streamer_articles` first.

**Old:**
```python
conn = sqlite3.connect(f'{DB_DIR}/news_articles.db')
news_df = pd.read_sql_query("""
    SELECT timestamp, ts_unix, title, title_gematria, title_dr, sentiment_score, date_doy
    FROM articles ORDER BY timestamp
""", conn)
conn.close()
```

**New:**
```python
conn = sqlite3.connect(f'{DB_DIR}/news_articles.db')
try:
    news_df = pd.read_sql_query("""
        SELECT timestamp, ts_unix, title, sentiment_score, title_dr,
               title_gematria_ordinal, title_gematria_reverse,
               title_gematria_reduction, sentiment_bull, sentiment_bear,
               has_caps, exclamation_count, word_count
        FROM streamer_articles ORDER BY timestamp
    """, conn)
except:
    news_df = pd.read_sql_query("""
        SELECT timestamp, ts_unix, title, title_gematria, title_dr, sentiment_score, date_doy
        FROM articles ORDER BY timestamp
    """, conn)
conn.close()
```

### Verification
After fix, print `news_df.columns.tolist()` and confirm it includes `sentiment_bull`, `sentiment_bear`, `has_caps`, etc. (the streamer_articles schema).

---

## FIX 4: Space Weather Format Mismatch in Live
**Priority: P1 (space weather features all NaN in live trading)**

### File
- `C:\Users\C\Documents\Savage22 Server\data_access.py` -- `LiveDataLoader._refresh_space_weather()` (lines 535-567) and `get_space_weather()` (line 638-640)

### Problem
`_refresh_space_weather()` stores the raw SQL result DataFrame in `self._caches['space_weather']` without converting the index to DatetimeIndex. The raw DataFrame has a RangeIndex (0, 1, 2...) with a `timestamp` column.

`feature_library.py` `compute_space_weather_features()` (line 3952-3953) does:
```python
sw = sw.reindex(_df_idx_sw, method='ffill')
```
This requires `sw` to have a DatetimeIndex. With RangeIndex, the reindex produces all NaN silently.

### Old Code (lines 558-567)
```python
if df.empty:
    return

if 'space_weather' in self._caches and len(self._caches['space_weather']) > 0:
    self._caches['space_weather'] = pd.concat([self._caches['space_weather'], df], ignore_index=True)
else:
    self._caches['space_weather'] = df

if 'timestamp' in df.columns and len(df) > 0:
    self._last_seen['space_weather'] = str(df['timestamp'].max())
```

### New Code (lines 558-580)
```python
if df.empty:
    return

# Convert timestamp to DatetimeIndex for feature_library compatibility
if 'timestamp' in df.columns:
    df['date'] = pd.to_datetime(df['timestamp'], unit='s', utc=True, errors='coerce')
    if df['date'].isna().all():
        df['date'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
    df = df.dropna(subset=['date'])
    if not df.empty:
        df = df.drop_duplicates(subset='date', keep='last').set_index('date').sort_index()

if df.empty:
    return

if 'space_weather' in self._caches and len(self._caches['space_weather']) > 0:
    combined = pd.concat([self._caches['space_weather'], df])
    self._caches['space_weather'] = combined[~combined.index.duplicated(keep='last')].sort_index()
else:
    self._caches['space_weather'] = df

self._last_seen['space_weather'] = str(df.index.max())
```

### Also fix `initial_load()` (lines 748+)
Space weather is not loaded during `initial_load()` at all -- the initial_load method (line 678) doesn't call `_refresh_space_weather()`. Add it:

After line 787 (`self._last_seen['onchain'] = ...`), add:
```python
# Load space weather
self._refresh_space_weather()
print(f"  space_weather: {len(self._caches.get('space_weather', []))} rows")
```

### Verification
After fix:
```python
sw = live_dal.get_space_weather()
print(type(sw.index))  # Must be DatetimeIndex, NOT RangeIndex
print(sw.head())       # Must show date-indexed rows with kp_index etc.
```

---

## FIX 5: HMM State Mapping Inverted in Live
**Priority: P1 (live regime labels don't match training -- bull/bear/neutral probabilities are scrambled)**

### Files
- `C:\Users\C\Documents\Savage22 Server\live_trader.py` lines 305-311 (get_hmm_features)
- Reference: `C:\Users\C\Documents\Savage22 Server\ml_multi_tf.py` lines 377-388 (training)

### Problem
Training sorts HMM states by mean return to label them:
```python
# ml_multi_tf.py lines 377-388
state_means = {}
for s in range(3):
    state_means[s] = r[states == s].mean() if (states == s).sum() > 0 else 0
sorted_states = sorted(state_means.keys(), key=lambda s: state_means[s])
# sorted_states[0] = bear (lowest mean return)
# sorted_states[1] = neutral
# sorted_states[2] = bull (highest mean return)
hmm_df = pd.DataFrame({
    'hmm_bull_prob': probs[:, sorted_states[2]],
    'hmm_bear_prob': probs[:, sorted_states[0]],
    'hmm_neutral_prob': probs[:, sorted_states[1]],
    'hmm_state': [sorted_states.index(s) for s in states],
})
```

But live_trader.py uses raw indices without sorting:
```python
# live_trader.py lines 305-311
probs = _hmm_model.predict_proba(X)[0]
state = int(np.argmax(probs))
return {
    'hmm_bull_prob': float(probs[0]),      # BUG: assumes state 0 = bull
    'hmm_bear_prob': float(probs[1]),      # BUG: assumes state 1 = bear
    'hmm_neutral_prob': float(probs[2]),   # BUG: assumes state 2 = neutral
    'hmm_state': state,
}
```

### Old Code (live_trader.py lines 283-314, the entire `get_hmm_features` function)
```python
def get_hmm_features(feat_dict):
    """Get HMM state probabilities for current bar. Returns dict of hmm_* features."""
    global _hmm_model, _hmm_last_fit
    if _hmm_model is None:
        return {}

    # Re-fit daily (at midnight)
    now = datetime.now(timezone.utc)
    if _hmm_last_fit and (now - _hmm_last_fit).total_seconds() > 86400:
        fit_hmm_live()

    try:
        # Use latest daily return + abs_ret + vol for prediction
        close = feat_dict.get('close', 0)
        prev_close = feat_dict.get('sma_5', close)  # approximate prev close
        if close <= 0 or prev_close <= 0:
            return {}
        ret = np.log(close / prev_close)
        abs_r = abs(ret)
        vol = feat_dict.get('volatility_short', abs_r)
        X = np.array([[ret, abs_r, vol]])

        probs = _hmm_model.predict_proba(X)[0]
        state = int(np.argmax(probs))
        return {
            'hmm_bull_prob': float(probs[0]),
            'hmm_bear_prob': float(probs[1]) if len(probs) > 1 else 0.0,
            'hmm_neutral_prob': float(probs[2]) if len(probs) > 2 else 0.0,
            'hmm_state': state,
        }
    except Exception:
        return {}
```

### New Code
```python
_hmm_sorted_states = None  # Module-level cache for sorted state mapping

def get_hmm_features(feat_dict):
    """Get HMM state probabilities for current bar. Returns dict of hmm_* features."""
    global _hmm_model, _hmm_last_fit, _hmm_sorted_states
    if _hmm_model is None:
        return {}

    # Re-fit daily (at midnight)
    now = datetime.now(timezone.utc)
    if _hmm_last_fit and (now - _hmm_last_fit).total_seconds() > 86400:
        fit_hmm_live()

    try:
        close = feat_dict.get('close', 0)
        prev_close = feat_dict.get('sma_5', close)
        if close <= 0 or prev_close <= 0:
            return {}
        ret = np.log(close / prev_close)
        abs_r = abs(ret)
        vol = feat_dict.get('volatility_short', abs_r)
        X = np.array([[ret, abs_r, vol]])

        probs = _hmm_model.predict_proba(X)[0]

        # Sort states by mean return (matches ml_multi_tf.py training)
        if _hmm_sorted_states is None:
            _compute_hmm_state_mapping()

        if _hmm_sorted_states is not None:
            sorted_states = _hmm_sorted_states
            state_raw = int(np.argmax(probs))
            return {
                'hmm_bull_prob': float(probs[sorted_states[2]]),
                'hmm_bear_prob': float(probs[sorted_states[0]]),
                'hmm_neutral_prob': float(probs[sorted_states[1]]),
                'hmm_state': sorted_states.index(state_raw),
            }
        else:
            # Fallback: raw indices (should not happen)
            state = int(np.argmax(probs))
            return {
                'hmm_bull_prob': float(probs[0]),
                'hmm_bear_prob': float(probs[1]) if len(probs) > 1 else 0.0,
                'hmm_neutral_prob': float(probs[2]) if len(probs) > 2 else 0.0,
                'hmm_state': state,
            }
    except Exception:
        return {}


def _compute_hmm_state_mapping():
    """Compute sorted state mapping from historical data (matches training)."""
    global _hmm_sorted_states, _hmm_model
    try:
        conn = sqlite3.connect(f'{DB_DIR}/btc_prices.db')
        daily = pd.read_sql_query("""
            SELECT open_time, close FROM ohlcv
            WHERE timeframe='1d' AND symbol='BTC/USDT' ORDER BY open_time
        """, conn)
        conn.close()
        daily['close'] = pd.to_numeric(daily['close'], errors='coerce')
        daily = daily.dropna(subset=['close'])
        closes = daily['close'].values
        returns = np.log(closes[1:] / closes[:-1])
        abs_ret = np.abs(returns)
        vol10 = pd.Series(returns).rolling(10).std().values
        valid = ~np.isnan(vol10)
        r = returns[valid]
        ar = abs_ret[valid]
        v = vol10[valid]
        X_hist = np.column_stack([r, ar, v])
        states = _hmm_model.predict(X_hist)
        state_means = {}
        for s in range(3):
            state_means[s] = r[states == s].mean() if (states == s).sum() > 0 else 0
        _hmm_sorted_states = sorted(state_means.keys(), key=lambda s: state_means[s])
        print(f"  HMM state mapping: bear={_hmm_sorted_states[0]}, neutral={_hmm_sorted_states[1]}, bull={_hmm_sorted_states[2]}")
    except Exception as e:
        print(f"  HMM state mapping failed: {e}")
        _hmm_sorted_states = None
```

Also add `_hmm_sorted_states = None` at module level (after line 228) and add a call to `_compute_hmm_state_mapping()` at the end of `fit_hmm_live()` (after line 276, before the return).

### Verification
After fix, print the sorted_states mapping and confirm it matches training: state with lowest mean return = bear, highest = bull.

---

## FIX 6: GCP Features Missing in Live (1h)
**Priority: P1 (1h models trained with GCP features get NaN at inference)**

### Files
- `C:\Users\C\Documents\Savage22 Server\live_trader.py` -- `compute_features_live()` function (lines 147-198)
- Reference: `C:\Users\C\Documents\Savage22 Server\build_1h_features.py` lines 332-374

### Problem
`build_1h_features.py` adds GCP features AFTER `build_all_features()` returns (lines 332-374). This includes `gcp_deviation_mean`, `gcp_deviation_max`, `gcp_rate_of_change`, `gcp_extreme`, plus cross features `tx_gcp_*_x_bull`, `tx_gcp_*_x_bear`, `gcp_x_fear`, `gcp_x_greed`, `gcp_x_full_moon`, `gcp_x_new_moon`, `gcp_x_kp_storm`. If the 1h model was trained on these, they are all NaN in live.

### Fix
Add GCP computation after `build_all_features()` in `compute_features_live()`, gated to only run for 1h (matching training):

After line 181 (`space_weather_df=space_weather_df,`), after the `if df_features is None` check (line 183), add:

```python
        # GCP features (1h only -- matches build_1h_features.py post-processing)
        if tf_name == '1h':
            try:
                from gcp_feature_builder import build_gcp_features
                gcp_feats = build_gcp_features(df_features)
                for col in gcp_feats.columns:
                    df_features[col] = gcp_feats[col]

                # GCP x trend crosses
                d_trend = pd.to_numeric(df_features.get('d_trend'), errors='coerce').fillna(0)
                bull = d_trend
                bear = 1 - d_trend
                for gcp_col in ['gcp_deviation_mean', 'gcp_deviation_max', 'gcp_rate_of_change', 'gcp_extreme']:
                    if gcp_col in df_features.columns:
                        sig = pd.to_numeric(df_features[gcp_col], errors='coerce').fillna(0)
                        df_features[f'tx_{gcp_col}_x_bull'] = sig * bull
                        df_features[f'tx_{gcp_col}_x_bear'] = sig * bear

                # GCP x fear/greed, moon, Kp
                if 'gcp_deviation_mean' in df_features.columns:
                    gcp_dev = pd.to_numeric(df_features['gcp_deviation_mean'], errors='coerce').fillna(0)
                    fg_fear = pd.to_numeric(df_features.get('fg_extreme_fear'), errors='coerce').fillna(0)
                    fg_greed = pd.to_numeric(df_features.get('fg_extreme_greed'), errors='coerce').fillna(0)
                    df_features['gcp_x_fear'] = gcp_dev.abs() * fg_fear
                    df_features['gcp_x_greed'] = gcp_dev.abs() * fg_greed
                    moon = pd.to_numeric(df_features.get('west_moon_phase'), errors='coerce').fillna(0)
                    is_full = ((moon >= 13) & (moon <= 16)).astype(float)
                    is_new = ((moon < 2) | (moon > 27.5)).astype(float)
                    df_features['gcp_x_full_moon'] = gcp_dev.abs() * is_full
                    df_features['gcp_x_new_moon'] = gcp_dev.abs() * is_new
                    kp = pd.to_numeric(df_features.get('sw_kp_is_storm'), errors='coerce').fillna(0)
                    df_features['gcp_x_kp_storm'] = gcp_dev.abs() * kp
            except Exception as e:
                pass  # GCP not available -- features will be NaN (XGBoost handles)
```

### Verification
Run live_trader and confirm 1h predictions now include GCP features in the feat_dict. Check `prediction_cache.json` after a 1h bar.

---

## FIX 7: Sports Columns Missing in Live
**Priority: P2 (missing columns are gematria-enriched -- affects esoteric features)**

### File
- `C:\Users\C\Documents\Savage22 Server\data_access.py`
- `_refresh_sports()` lines 505-511
- `initial_load()` lines 730-736

### Problem
The `OfflineDataLoader.load_sports()` (lines 173-190) loads the full schema:
```sql
SELECT date, winner, home_team, away_team, home_score, away_score,
       venue, winner_gem_ordinal, winner_gem_dr,
       home_gem_ordinal, home_gem_dr, away_gem_ordinal, away_gem_dr,
       score_total, score_diff, score_dr, score_total_dr,
       is_upset, is_overtime
FROM games
```

But `LiveDataLoader._refresh_sports()` (lines 505-511) and `initial_load()` (lines 730-736) load a subset:
```sql
SELECT date, winner, home_score, away_score,
       winner_gem_ordinal, winner_gem_dr, score_dr,
       score_total_dr, is_upset, is_overtime
FROM games
```

Missing: `home_team, away_team, venue, home_gem_ordinal, home_gem_dr, away_gem_ordinal, away_gem_dr, score_total, score_diff`

Similarly for horse_races, missing: `winner_trainer, race_gem_ordinal, odds_dr`

### Fix for `_refresh_sports()` (lines 505-518)
**Old:**
```python
games = _safe_read_sql(conn, f"""
    SELECT date, winner, home_score, away_score,
           winner_gem_ordinal, winner_gem_dr, score_dr,
           score_total_dr, is_upset, is_overtime
    FROM games WHERE date > '{last}'
    ORDER BY date
""")

horses = _safe_read_sql(conn, f"""
    SELECT date, winner_horse, horse_gem_ordinal, horse_gem_dr,
           jockey_gem_ordinal, position_dr
    FROM horse_races WHERE date > '{last}'
    ORDER BY date
""")
```

**New:**
```python
games = _safe_read_sql(conn, f"""
    SELECT date, winner, home_team, away_team, home_score, away_score,
           venue, winner_gem_ordinal, winner_gem_dr,
           home_gem_ordinal, home_gem_dr, away_gem_ordinal, away_gem_dr,
           score_total, score_diff, score_dr, score_total_dr,
           is_upset, is_overtime
    FROM games WHERE date > '{last}'
    ORDER BY date
""")

horses = _safe_read_sql(conn, f"""
    SELECT date, winner_horse, winner_jockey, winner_trainer,
           horse_gem_ordinal, horse_gem_dr,
           jockey_gem_ordinal, race_gem_ordinal, position_dr, odds_dr
    FROM horse_races WHERE date > '{last}'
    ORDER BY date
""")
```

### Fix for `initial_load()` (lines 730-742)
Apply the same column expansion to the initial_load sports queries.

**Old:**
```python
games = _safe_read_sql(conn, f"""
    SELECT date, winner, home_score, away_score,
           winner_gem_ordinal, winner_gem_dr, score_dr,
           score_total_dr, is_upset, is_overtime
    FROM games WHERE date > '{cutoff_date}'
    ORDER BY date
""")
horses = _safe_read_sql(conn, f"""
    SELECT date, winner_horse, horse_gem_ordinal, horse_gem_dr,
           jockey_gem_ordinal, position_dr
    FROM horse_races WHERE date > '{cutoff_date}'
    ORDER BY date
""")
```

**New:**
```python
games = _safe_read_sql(conn, f"""
    SELECT date, winner, home_team, away_team, home_score, away_score,
           venue, winner_gem_ordinal, winner_gem_dr,
           home_gem_ordinal, home_gem_dr, away_gem_ordinal, away_gem_dr,
           score_total, score_diff, score_dr, score_total_dr,
           is_upset, is_overtime
    FROM games WHERE date > '{cutoff_date}'
    ORDER BY date
""")
horses = _safe_read_sql(conn, f"""
    SELECT date, winner_horse, winner_jockey, winner_trainer,
           horse_gem_ordinal, horse_gem_dr,
           jockey_gem_ordinal, race_gem_ordinal, position_dr, odds_dr
    FROM horse_races WHERE date > '{cutoff_date}'
    ORDER BY date
""")
```

### Verification
```python
sports = live_dal.get_sports()
print(sports['games'].columns.tolist())  # Must include home_team, away_team, venue, etc.
print(sports['horse_races'].columns.tolist())  # Must include winner_trainer, race_gem_ordinal, odds_dr
```

---

## FIX 8: V2 live_trader.py 3-Class Prediction Crash
**Priority: P1 (V2 live trading crashes on every prediction)**

### File
- `C:\Users\C\Documents\Savage22 Server\v2\live_trader.py` line 1053

### Problem
Line 1053: `prob = float(models[tf].predict(dmat)[0])`

With `multi:softprob` 3-class models, `predict()` returns shape `(1, 3)` -- an array of 3 probabilities. `[0]` gets the first row (array of 3 floats). `float()` on a 3-element array crashes with `TypeError: only length-1 arrays can be converted to Python scalars`.

Root V1 (live_trader.py lines 549-560) handles this correctly:
```python
raw_pred = models[tf].predict(dmat)
if raw_pred.ndim == 2:
    probs_3c = raw_pred[0]
elif len(raw_pred) >= 3:
    probs_3c = raw_pred[:3]
else:
    probs_3c = np.array([1 - raw_pred[0], 0.0, raw_pred[0]])
p_short, p_flat, p_long = float(probs_3c[0]), float(probs_3c[1]), float(probs_3c[2])
pred_class = int(np.argmax(probs_3c))
confidence = float(np.max(probs_3c))
```

### Old Code (v2/live_trader.py lines 1051-1064, approximate)
```python
                    dmat = xgb.DMatrix(X_combined, feature_names=all_feat_names)
                    prob = float(models[tf].predict(dmat)[0])
                    ...
                else:
                    ...
                    dmat = xgb.DMatrix(X_base, feature_names=feat_names)
                    prob = float(models[tf].predict(dmat)[0])
```

### New Code (replace all `prob = float(models[tf].predict(dmat)[0])` occurrences)
```python
                    dmat = xgb.DMatrix(X_combined, feature_names=all_feat_names)
                    raw_pred = models[tf].predict(dmat)
                    if raw_pred.ndim == 2:
                        probs_3c = raw_pred[0]  # shape (3,)
                    elif len(raw_pred) >= 3:
                        probs_3c = raw_pred[:3]  # flattened
                    else:
                        probs_3c = np.array([1 - raw_pred[0], 0.0, raw_pred[0]])
                    p_short, p_flat, p_long = float(probs_3c[0]), float(probs_3c[1]), float(probs_3c[2])
                    pred_class = int(np.argmax(probs_3c))
                    confidence = float(np.max(probs_3c))
                    prob = confidence
```

Apply this replacement to ALL 3 locations where `prob = float(models[tf].predict(dmat)[0])` appears (lines ~1053, ~1060, ~1064).

Also update the direction logic downstream (currently uses `prob > conf_thresh` for LONG and `prob < (1-conf_thresh)` for SHORT). Replace with:
```python
                direction = None
                if pred_class == 2 and confidence > conf_thresh:
                    direction = 'LONG'
                    prob = p_long
                elif pred_class == 0 and confidence > conf_thresh:
                    direction = 'SHORT'
                    prob = p_short
```

### Verification
Load a V2 3-class model and run a prediction manually:
```python
raw = model.predict(dmat)
print(raw.shape, raw.ndim)  # Should be (1, 3) or (3,)
```

---

## FIX 9: V2 Missing space_weather_df in build_all_features Call
**Priority: P1 (all space weather features NaN in V2 live)**

### File
- `C:\Users\C\Documents\Savage22 Server\v2\live_trader.py` lines 359-366 and lines 679-686

### Problem
Two functions (`compute_features_live_v2` at line 359 and `compute_features_live` at line 679) call `build_all_features()` without `space_weather_df` parameter. Root V1 correctly passes it (line 180).

### Old Code (v2/live_trader.py line 359-366)
```python
        df_features = build_all_features(
            ohlcv=ohlcv,
            esoteric_frames=esoteric_frames,
            tf_name=tf_name,
            mode='live',
            htf_data=htf_data,
            astro_cache=astro_cache,
        )
```

### New Code (v2/live_trader.py line 359-367)
```python
        space_weather_df = live_dal.get_space_weather()

        df_features = build_all_features(
            ohlcv=ohlcv,
            esoteric_frames=esoteric_frames,
            tf_name=tf_name,
            mode='live',
            htf_data=htf_data,
            astro_cache=astro_cache,
            space_weather_df=space_weather_df,
        )
```

Apply the same fix to the second occurrence at lines 679-686.

### Verification
After fix, check that `sw_kp_index`, `sw_sunspot_number` etc. are non-NaN in live predictions.

---

## FIX 10: V2 Missing HMM Features
**Priority: P1 (V2 live predictions missing HMM regime features entirely)**

### File
- `C:\Users\C\Documents\Savage22 Server\v2\live_trader.py`

### Problem
V2 live_trader.py has no HMM import, no HMM fitting, no HMM feature injection. Root V1 has:
- Import at lines 46-50
- `fit_hmm_live()` at lines 230-280
- `get_hmm_features()` at lines 283-314
- Injection at lines 534-536

If V2 models were trained with HMM features (hmm_bull_prob, hmm_bear_prob, hmm_neutral_prob, hmm_state), they will all be NaN in V2 live.

### Fix
Add to the top of v2/live_trader.py (after line 35, the existing imports):

```python
# HMM regime detection (matches ml_multi_tf.py training pipeline)
try:
    from hmmlearn.hmm import GaussianHMM
    _HAS_HMM = True
except ImportError:
    _HAS_HMM = False
```

Then copy (or import from root) the `fit_hmm_live()`, `_compute_hmm_state_mapping()`, and `get_hmm_features()` functions from root live_trader.py (after Fix 5 is applied).

In the trading loop, add HMM fitting at startup (matching V1 line 477-478):
```python
if _HAS_HMM:
    fit_hmm_live()
```

And inject HMM features after `compute_features_live_v2()` returns (matching V1 line 534-536):
```python
hmm_feats = get_hmm_features(feat_dict)
feat_dict.update(hmm_feats)
```

### Verification
After fix, check `feat_dict` contains `hmm_bull_prob`, `hmm_bear_prob`, `hmm_neutral_prob`, `hmm_state` with non-NaN values.

---

## FIX 11: compute_sample_uniqueness @njit
**Priority: P2 (performance -- 200K+ iterations in raw Python loops)**

### File
- `C:\Users\C\Documents\Savage22 Server\v2\v2_multi_asset_trainer.py` lines 359-387

### Problem
Two raw Python for-loops over `n_samples` (200K+). The first loop (line 374) increments a concurrent counter. The second loop (line 380) computes uniqueness. Both are O(n * max_hold_bars) in Python -- extremely slow for 200K samples.

### Old Code (lines 359-387)
```python
def compute_sample_uniqueness(n_samples, max_hold_bars):
    t0_arr = np.arange(n_samples)
    t1_arr = np.minimum(t0_arr + max_hold_bars, n_samples - 1)

    concurrent = np.zeros(n_samples, dtype=np.int32)
    for i in range(n_samples):
        s, e = int(t0_arr[i]), min(int(t1_arr[i]) + 1, n_samples)
        concurrent[s:e] += 1

    uniqueness = np.ones(n_samples, dtype=np.float64)
    for i in range(n_samples):
        s, e = int(t0_arr[i]), min(int(t1_arr[i]) + 1, n_samples)
        if e > s:
            conc_slice = concurrent[s:e].astype(np.float64)
            conc_slice = np.where(conc_slice > 0, conc_slice, 1)
            uniqueness[i] = np.mean(1.0 / conc_slice)

    return uniqueness
```

### New Code
```python
try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


if _HAS_NUMBA:
    @njit(cache=True)
    def _compute_uniqueness_numba(n_samples, max_hold_bars):
        t0_arr = np.arange(n_samples)
        t1_arr = np.minimum(t0_arr + max_hold_bars, n_samples - 1)

        concurrent = np.zeros(n_samples, dtype=np.int32)
        for i in range(n_samples):
            s = t0_arr[i]
            e = min(t1_arr[i] + 1, n_samples)
            for j in range(s, e):
                concurrent[j] += 1

        uniqueness = np.ones(n_samples, dtype=np.float64)
        for i in range(n_samples):
            s = t0_arr[i]
            e = min(t1_arr[i] + 1, n_samples)
            if e > s:
                total = 0.0
                for j in range(s, e):
                    c = concurrent[j]
                    if c > 0:
                        total += 1.0 / c
                    else:
                        total += 1.0
                uniqueness[i] = total / (e - s)

        return uniqueness


def compute_sample_uniqueness(n_samples, max_hold_bars):
    """Compute average uniqueness per sample (Lopez de Prado method)."""
    if _HAS_NUMBA:
        return _compute_uniqueness_numba(n_samples, max_hold_bars)

    # Fallback: vectorized NumPy (no per-element loops)
    t0_arr = np.arange(n_samples)
    t1_arr = np.minimum(t0_arr + max_hold_bars, n_samples - 1)

    concurrent = np.zeros(n_samples, dtype=np.int32)
    for i in range(n_samples):
        s, e = int(t0_arr[i]), min(int(t1_arr[i]) + 1, n_samples)
        concurrent[s:e] += 1

    uniqueness = np.ones(n_samples, dtype=np.float64)
    for i in range(n_samples):
        s, e = int(t0_arr[i]), min(int(t1_arr[i]) + 1, n_samples)
        if e > s:
            conc_slice = concurrent[s:e].astype(np.float64)
            conc_slice = np.where(conc_slice > 0, conc_slice, 1)
            uniqueness[i] = np.mean(1.0 / conc_slice)

    return uniqueness
```

### Verification
```python
import time
t0 = time.time()
u = compute_sample_uniqueness(200_000, 24)
print(f"  {time.time()-t0:.2f}s, shape={u.shape}, range=[{u.min():.4f}, {u.max():.4f}]")
```
Should complete in <1s with Numba vs 30+ seconds without.

---

## FIX 12: V1 exhaustive_optimizer Direction Bug
**Priority: P1 (optimizer ignores model direction -- treats confidence as direction signal)**

### File
- `C:\Users\C\Documents\Savage22 Server\exhaustive_optimizer.py`

### Problem
`load_tf_data()` (line 335) returns `directions` (LONG=+1, SHORT=-1, FLAT=0). But `run_grid_search()` (line 578) signature is `run_grid_search(tf_name, confs, closes, atrs, n_bars)` -- it does NOT take `dirs`. Line 811 unpacks: `confs, dirs, rets, closes, atrs, n_bars = data` but only passes `confs, closes, atrs` to `run_grid_search()`.

Then `simulate_batch()` (line 343) signature is `simulate_batch(params_batch, confs, closes, atrs, xp_lib)` -- also no `dirs`. Instead, it invents direction from confidence at lines 507-510:
```python
go_long  = can_enter & (c_val > conf_th)
go_short = can_enter & (c_val < (1.0 - conf_th))
```
This is WRONG for 3-class softprob. Confidence is always 0.33-1.0 (max of 3 probs). The model says LONG or SHORT via `argmax`, not via confidence threshold.

V2 exhaustive_optimizer.py (line 353) correctly takes `dirs`:
```python
def simulate_batch(params_batch, confs, dirs, closes, atrs, highs, lows, regime, xp_lib):
```
And uses `dirs_t` at line 548-549:
```python
go_long  = can_enter & (dirs_t == 1.0) & (c_val > conf_th)
go_short = can_enter & (dirs_t == -1.0) & (c_val > conf_th)
```

### Fix

1. **Update `run_grid_search` signature** (line 578):
**Old:** `def run_grid_search(tf_name, confs, closes, atrs, n_bars):`
**New:** `def run_grid_search(tf_name, confs, dirs, closes, atrs, n_bars):`

2. **Update `simulate_batch` signature** (line 343):
**Old:** `def simulate_batch(params_batch, confs, closes, atrs, xp_lib):`
**New:** `def simulate_batch(params_batch, confs, dirs, closes, atrs, xp_lib):`

3. **Add `dirs` extraction in simulate_batch** at line 403-404:
Add: `dirs_t = float(dirs[t])`

4. **Fix entry logic** (lines 507-510):
**Old:**
```python
go_long  = can_enter & (c_val > conf_th)
go_short = can_enter & (c_val < (1.0 - conf_th))
```
**New:**
```python
go_long  = can_enter & (dirs_t == 1.0) & (c_val > conf_th)
go_short = can_enter & (dirs_t == -1.0) & (c_val > conf_th)
```

5. **Update call site** (line 811):
**Old:** `best = run_grid_search(tf_name, confs, closes, atrs, n_bars)`
**New:** `best = run_grid_search(tf_name, confs, dirs, closes, atrs, n_bars)`

6. **Update GPU transfer** in run_grid_search (add after line 617):
```python
if GPU_ARRAY:
    g_dirs = cp.asarray(dirs)
else:
    g_dirs = dirs
```

7. **Update simulate_batch calls** (lines 669, 674):
**Old:** `results = simulate_batch(params_gpu, g_confs, g_closes, g_atrs, cp)`
**New:** `results = simulate_batch(params_gpu, g_confs, g_dirs, g_closes, g_atrs, cp)`

### Verification
Run optimizer on a small TF. Check that SHORT trades now appear when the model predicts SHORT (pred_class=0), not when confidence < (1-threshold).

---

## FIX 13: V1 Optimizer Close-Only SL/TP
**Priority: P1 (SL/TP checked against close prices only, misses intra-bar touches)**

### File
- `C:\Users\C\Documents\Savage22 Server\exhaustive_optimizer.py` lines 439-447

### Problem
SL and TP checks use `p_val = float(closes[t])` (line 404). In reality, SL should check against lows (for longs) and highs (for shorts), and TP should check against highs (for longs) and lows (for shorts).

V2 optimizer correctly uses `highs` and `lows`:
```python
def simulate_batch(params_batch, confs, dirs, closes, atrs, highs, lows, regime, xp_lib):
```

### Fix
This is a larger refactor -- need to:

1. **Add `highs` and `lows` to `load_tf_data()` return** (after line 325):
```python
test_highs = pd.to_numeric(df['high'], errors='coerce').values[vs:ve]
test_lows = pd.to_numeric(df['low'], errors='coerce').values[vs:ve]
# Fill NaN with close
test_highs = np.where(np.isnan(test_highs), test_closes, test_highs)
test_lows = np.where(np.isnan(test_lows), test_closes, test_lows)
```
Update return (line 335):
```python
return confidences, directions, test_returns, test_closes, test_atrs, test_highs, test_lows, n_bars
```

2. **Update simulate_batch** to use `h_val` for long TP and short SL, `l_val` for long SL and short TP:

At line 404, add:
```python
h_val = float(highs[t]) if highs[t] > 0 else p_val
l_val = float(lows[t]) if lows[t] > 0 else p_val
```

Lines 439-447, change:
**Old:**
```python
sl_long  = active & (trade_dir == 1)  & (p_val <= stop_pr)
sl_short = active & (trade_dir == -1) & (p_val >= stop_pr)
sl_hit = sl_long | sl_short

tp_long  = active & (trade_dir == 1)  & (p_val >= tp_pr)
tp_short = active & (trade_dir == -1) & (p_val <= tp_pr)
tp_hit = tp_long | tp_short
```
**New:**
```python
sl_long  = active & (trade_dir == 1)  & (l_val <= stop_pr)
sl_short = active & (trade_dir == -1) & (h_val >= stop_pr)
sl_hit = sl_long | sl_short

tp_long  = active & (trade_dir == 1)  & (h_val >= tp_pr)
tp_short = active & (trade_dir == -1) & (l_val <= tp_pr)
tp_hit = tp_long | tp_short
```

3. **Update all call sites** to pass `highs, lows` through.

### Verification
Compare optimizer results before/after on same data. Expect more SL/TP triggers (intra-bar wicks now caught). Trade count should increase and max DD may change.

---

## FIX 14: Prediction Cache Timezone
**Priority: P2 (cosmetic -- timestamp in prediction_cache.json is local time, not UTC)**

### Files
- `C:\Users\C\Documents\Savage22 Server\live_trader.py` line 711
- `C:\Users\C\Documents\Savage22 Server\v2\live_trader.py` line 1252

### Old Code (both files)
```python
'timestamp': datetime.now().isoformat(),
```

### New Code (both files)
```python
'timestamp': datetime.now(timezone.utc).isoformat(),
```

### Verification
After fix, check `prediction_cache.json` -- timestamp should end with `+00:00` (UTC).

---

## EXECUTION ORDER

### Phase 1: Must Fix Before Training (P0)
1. **Fix 1** -- 5m/15m SQL columns (5 min)
2. **Fix 2** -- Onchain timestamped data in builds (30 min -- touches all 7 build files + possibly feature_library.py)
3. **Fix 3** -- News schema inversion (15 min -- all build files)

### Phase 2: Must Fix Before Live Trading (P1)
4. **Fix 5** -- HMM state mapping (20 min)
5. **Fix 4** -- Space weather DatetimeIndex (15 min)
6. **Fix 8** -- V2 3-class prediction crash (10 min)
7. **Fix 9** -- V2 space weather param (5 min)
8. **Fix 10** -- V2 HMM features (20 min)
9. **Fix 12** -- Optimizer direction bug (20 min)
10. **Fix 13** -- Optimizer close-only SL/TP (20 min)
11. **Fix 6** -- GCP features in live (15 min)
12. **Fix 7** -- Sports columns in live (10 min)

### Phase 3: Nice to Have (P2)
13. **Fix 11** -- Numba @njit for uniqueness (10 min)
14. **Fix 14** -- Prediction cache timezone (2 min)

**Total estimated time: ~3 hours**
