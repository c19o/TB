#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, subprocess
_skip_gpu = os.environ.get('V2_SKIP_GPU') == '1'
if not _skip_gpu:
    try:
        _nv = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=5)
        if int(_nv.stdout.strip().split('.')[0]) >= 580:
            _skip_gpu = True
    except Exception:
        print("[GPU] nvidia-smi check failed, assuming CUDA 12 compatible", flush=True)
if not _skip_gpu:
    try:
        import cudf.pandas
        cudf.pandas.install()
        print("[GPU] cudf.pandas ENABLED")
    except (ImportError, Exception) as e:
        if os.environ.get('ALLOW_CPU', '0') == '1':
            print(f"[WARNING] GPU not available ({e}), running on CPU (ALLOW_CPU=1)", flush=True)
        else:
            raise RuntimeError(f"GPU REQUIRED: cuDF.pandas import failed: {e}. Set ALLOW_CPU=1 to force CPU mode.") from e
"""
build_1d_features.py
=====================
Build daily (1D) feature matrix from BTC/USDT daily candles using the unified
feature_library.build_all_features() pipeline.

Loads ALL source DBs and passes them to the single shared function so that
every feature group (TA, time, numerology, astrology, esoteric, space weather,
cycles, composites, gematria, cross-features, decay, event astro, calendar,
arabic lots, KNN) is computed consistently.

Saves to features_1d.db, table features_1d.
"""

import sys, os, time, warnings
from datetime import datetime
# Unbuffered output for progress logging
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import sqlite3

DB_DIR = os.path.dirname(os.path.abspath(__file__))
START_TIME = time.time()

def elapsed():
    return f"[{time.time()-START_TIME:.0f}s]"

# ============================================================
# LOAD ALL DATA SOURCES
# ============================================================
print("=" * 70)
print("BUILD 1D FEATURE MATRIX (unified pipeline)")
print("=" * 70)

# --- 1D BTC candles (primary OHLCV) ---
print(f"\n{elapsed()} Loading 1D BTC/USDT candles...")
conn = sqlite3.connect(f'{DB_DIR}/btc_prices.db')
btc = pd.read_sql_query("""
    SELECT open_time, open, high, low, close, volume, quote_volume, trades,
           taker_buy_volume, taker_buy_quote
    FROM ohlcv WHERE timeframe='1d' AND symbol='BTC/USDT'
    ORDER BY open_time
""", conn)
conn.close()
btc['timestamp'] = pd.to_datetime(btc['open_time'], unit='ms', utc=True)
btc = btc.drop_duplicates(subset='open_time', keep='last').set_index('timestamp').sort_index()
for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote']:
    if col in btc.columns:
        btc[col] = pd.to_numeric(btc[col], errors='coerce')
first_good = btc[btc['volume'] > 0].index[0]
btc = btc.loc[first_good:]
print(f"  1D candles: {len(btc)} ({btc.index.min()} to {btc.index.max()})")

# --- Auxiliary data ---
print(f"{elapsed()} Loading auxiliary data...")

# Ephemeris
conn = sqlite3.connect(f'{DB_DIR}/ephemeris_cache.db')
ephem_df = pd.read_sql_query("SELECT * FROM ephemeris ORDER BY date", conn)
conn.close()
ephem_df['date'] = pd.to_datetime(ephem_df['date'])
ephem_df = ephem_df.drop_duplicates(subset='date', keep='last').set_index('date')
print(f"  Ephemeris: {len(ephem_df)}")

# Astrology Full
conn = sqlite3.connect(f'{DB_DIR}/astrology_full.db')
astro_df = pd.read_sql_query("SELECT * FROM daily_astrology ORDER BY date", conn)
conn.close()
astro_df['date'] = pd.to_datetime(astro_df['date'])
astro_df = astro_df.drop_duplicates(subset='date', keep='last').set_index('date')
print(f"  Astrology: {len(astro_df)}")

# Fear & Greed
conn = sqlite3.connect(f'{DB_DIR}/fear_greed.db')
fg_df = pd.read_sql_query("SELECT * FROM fear_greed ORDER BY date", conn)
conn.close()
fg_df['date'] = pd.to_datetime(fg_df['date'])
fg_df = fg_df.drop_duplicates(subset='date', keep='last').set_index('date')
fg_df['value'] = pd.to_numeric(fg_df['value'], errors='coerce')
print(f"  Fear/Greed: {len(fg_df)}")

# News
conn = sqlite3.connect(f'{DB_DIR}/news_articles.db')
_table_exists = conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND name=?", ('articles',)
).fetchone()
if _table_exists:
    news_df = pd.read_sql_query("SELECT timestamp, ts_unix, title, title_gematria, title_dr, sentiment_score, date_doy FROM articles ORDER BY timestamp", conn)
else:
    _fallback_exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", ('streamer_articles',)
    ).fetchone()
    if _fallback_exists:
        news_df = pd.read_sql_query("SELECT timestamp, ts_unix, title FROM streamer_articles ORDER BY timestamp", conn)
        print("  Using fallback table 'streamer_articles'", flush=True)
    else:
        print("  WARNING: Neither 'articles' nor 'streamer_articles' table found", flush=True)
        news_df = pd.DataFrame(columns=['timestamp', 'ts_unix', 'title'])
conn.close()
news_df['dt'] = pd.to_datetime(news_df['timestamp'], errors='coerce', utc=True)
news_df = news_df.dropna(subset=['dt'])
news_df['date'] = pd.to_datetime(news_df['dt'].dt.date)
news_df['ts_unix_num'] = pd.to_numeric(news_df.get('ts_unix', pd.Series(dtype=float)), errors='coerce')
print(f"  News: {len(news_df)}")

# Tweets
conn = sqlite3.connect(f'{DB_DIR}/tweets.db')
tweets_df = pd.read_sql_query("""
    SELECT created_at, ts_unix, user_handle, full_text, has_gold, has_red, has_green, dominant_colors,
           gematria_simple, gematria_english, favorite_count, retweet_count, reply_count,
           sentiment_bull, sentiment_bear
    FROM tweets ORDER BY created_at
""", conn)
conn.close()
tweets_df['dt'] = pd.to_datetime(tweets_df['created_at'], errors='coerce', utc=True)
tweets_df = tweets_df.dropna(subset=['dt'])
tweets_df['date'] = pd.to_datetime(tweets_df['dt'].dt.date)
tweets_df['ts_unix_num'] = pd.to_numeric(tweets_df['ts_unix'], errors='coerce')
print(f"  Tweets: {len(tweets_df)}")

# Macro
conn = sqlite3.connect(f'{DB_DIR}/macro_data.db')
macro_df = pd.read_sql_query("SELECT * FROM macro_data ORDER BY date", conn)
conn.close()
macro_df['date'] = pd.to_datetime(macro_df['date'])
macro_df = macro_df.drop_duplicates(subset='date', keep='last').set_index('date')
print(f"  Macro: {len(macro_df)}")

# On-chain
conn = sqlite3.connect(f'{DB_DIR}/onchain_data.db')
onchain_df = pd.read_sql_query("SELECT * FROM blockchain_data ORDER BY date", conn)
conn.close()
onchain_df['date'] = pd.to_datetime(onchain_df['date'])
onchain_df = onchain_df.drop_duplicates(subset='date', keep='last').set_index('date')
print(f"  On-chain: {len(onchain_df)}")

# Google Trends
conn = sqlite3.connect(f'{DB_DIR}/google_trends.db')
gtrends_df = pd.read_sql_query("SELECT * FROM google_trends ORDER BY date", conn)
conn.close()
gtrends_df['date'] = pd.to_datetime(gtrends_df['date'])
gtrends_df = gtrends_df.drop_duplicates(subset='date', keep='last').set_index('date')
print(f"  Google Trends: {len(gtrends_df)}")

# Funding rates
conn = sqlite3.connect(f'{DB_DIR}/funding_rates.db')
funding_df = pd.read_sql_query("""
    SELECT timestamp, funding_rate FROM funding_rates
    WHERE symbol='BTCUSDT' ORDER BY timestamp
""", conn)
conn.close()
funding_df['date'] = pd.to_datetime(funding_df['timestamp'], errors='coerce').dt.date
funding_df = funding_df.dropna(subset=['date'])
funding_df['date'] = pd.to_datetime(funding_df['date'])
funding_daily = funding_df.groupby('date')['funding_rate'].mean().to_frame('avg_funding_rate')
print(f"  Funding: {len(funding_daily)}")

# Open Interest
oi_daily = pd.DataFrame()
oi_db_path = f'{DB_DIR}/open_interest.db'
if os.path.exists(oi_db_path):
    try:
        conn = sqlite3.connect(oi_db_path)
        oi_raw = pd.read_sql_query("SELECT timestamp, oi_contracts, oi_usd FROM open_interest ORDER BY timestamp", conn)
        conn.close()
        if len(oi_raw) > 0:
            oi_raw['date'] = pd.to_datetime(oi_raw['timestamp'], errors='coerce').dt.normalize()
            oi_raw = oi_raw.dropna(subset=['date'])
            oi_raw['oi_usd'] = pd.to_numeric(oi_raw['oi_usd'], errors='coerce')
            oi_raw['oi_contracts'] = pd.to_numeric(oi_raw['oi_contracts'], errors='coerce')
            oi_daily = oi_raw.groupby('date').agg(oi_usd=('oi_usd', 'last'), oi_contracts=('oi_contracts', 'last'))
            print(f"  Open Interest: {len(oi_daily)}")
    except Exception as e:
        print(f"  Open Interest load failed: {e}")
else:
    print(f"  open_interest.db not found at {oi_db_path}")

# Sports
print(f"{elapsed()} Loading sports data...")
sp_games = pd.DataFrame()
sp_horses = pd.DataFrame()
try:
    sp_conn = sqlite3.connect(f'{DB_DIR}/sports_results.db', timeout=5)
    sp_games = pd.read_sql_query("SELECT * FROM games ORDER BY date", sp_conn)
    try:
        sp_horses = pd.read_sql_query("SELECT * FROM horse_races ORDER BY date", sp_conn)
    except Exception:
        print("  WARNING: 'horse_races' table not found in sports_results.db", flush=True)
        sp_horses = pd.DataFrame()
    sp_conn.close()
    print(f"  Games: {len(sp_games)}, Horse races: {len(sp_horses)}")
except Exception as e:
    print(f"  Sports data: {e}")

# --- Space weather: kp_history_gfz.txt + space_weather.db ---
print(f"{elapsed()} Loading space weather data...")
sw_rows = []
kp_file = f'{DB_DIR}/kp_history_gfz.txt'
if os.path.exists(kp_file):
    with open(kp_file, 'r', encoding='utf-8', errors='replace') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 26:
                continue
            try:
                yr, mo, dy = int(parts[0]), int(parts[1]), int(parts[2])
                kp_vals = [float(parts[i]) for i in range(7, 15)]
                kp_valid = [k for k in kp_vals if k >= 0]
                kp_mean = np.mean(kp_valid) if kp_valid else np.nan
                sn = float(parts[24])
                if sn < 0:
                    sn = np.nan
                f107 = float(parts[25])
                if f107 < 0:
                    f107 = np.nan
                dt = datetime(yr, mo, dy)
                sw_rows.append({
                    'date': dt,
                    'kp_index': kp_mean,
                    'sunspot_number': sn,
                    'solar_flux_f107': f107,
                })
            except (ValueError, IndexError):
                continue
    print(f"  kp_history_gfz.txt: {len(sw_rows)} days loaded")
else:
    print(f"  kp_history_gfz.txt not found at {kp_file}")

if sw_rows:
    sw_hist = pd.DataFrame(sw_rows)
    sw_hist['date'] = pd.to_datetime(sw_hist['date'])
    sw_hist = sw_hist.drop_duplicates(subset='date', keep='last').set_index('date').sort_index()
else:
    sw_hist = pd.DataFrame(columns=['kp_index', 'sunspot_number', 'solar_flux_f107'])
    sw_hist.index.name = 'date'

sw_db_path = f'{DB_DIR}/space_weather.db'
if os.path.exists(sw_db_path):
    try:
        conn_sw = sqlite3.connect(sw_db_path)
        sw_live = pd.read_sql_query("SELECT * FROM space_weather ORDER BY timestamp", conn_sw)
        conn_sw.close()
        if len(sw_live) > 0:
            sw_live['date'] = pd.to_datetime(sw_live['timestamp'], unit='s', utc=True)
            sw_live['date'] = sw_live['date'].dt.tz_localize(None).dt.normalize()
            sw_live = sw_live.drop_duplicates(subset='date', keep='last').set_index('date')
            live_mapped = pd.DataFrame(index=sw_live.index)
            live_mapped['kp_index'] = pd.to_numeric(sw_live.get('kp_index', pd.Series(dtype=float)), errors='coerce')
            live_mapped['sunspot_number'] = np.nan
            live_mapped['solar_flux_f107'] = np.nan
            if 'solar_wind_speed' in sw_live.columns:
                live_mapped['solar_wind_speed'] = pd.to_numeric(sw_live['solar_wind_speed'], errors='coerce')
            if 'solar_wind_bz' in sw_live.columns:
                live_mapped['solar_wind_bz'] = pd.to_numeric(sw_live['solar_wind_bz'], errors='coerce')
            for sc in ['r_scale', 's_scale', 'g_scale']:
                if sc in sw_live.columns:
                    live_mapped[sc] = pd.to_numeric(sw_live[sc], errors='coerce')
            new_dates = live_mapped.index.difference(sw_hist.index)
            if len(new_dates) > 0:
                sw_hist = pd.concat([sw_hist, live_mapped.loc[new_dates]])
                sw_hist = sw_hist.sort_index()
            for col in ['solar_wind_speed', 'solar_wind_bz', 'r_scale', 's_scale', 'g_scale']:
                if col in live_mapped.columns:
                    if col not in sw_hist.columns:
                        sw_hist[col] = np.nan
                    overlap = live_mapped.index.intersection(sw_hist.index)
                    if len(overlap) > 0:
                        sw_hist.loc[overlap, col] = live_mapped.loc[overlap, col]
            print(f"  space_weather.db: {len(sw_live)} live rows merged")
    except Exception as e:
        print(f"  space_weather.db merge failed: {e}")
else:
    print(f"  space_weather.db not found")

sw_hist.index = pd.to_datetime(sw_hist.index, utc=True)
space_weather_df = sw_hist
print(f"  Combined space weather: {len(space_weather_df)} days ({space_weather_df.index.min()} to {space_weather_df.index.max()})")

print(f"\n{elapsed()} All data loaded.\n")

# ============================================================
# BUILD FEATURES USING UNIFIED PIPELINE
# ============================================================
print("=" * 70)
print("Computing 1D features via feature_library.build_all_features()...")
print("=" * 70)

from feature_library import build_all_features

ohlcv = btc.copy()

esoteric_frames = {
    'tweets': tweets_df,
    'news': news_df,
    'sports': {'games': sp_games, 'horse_races': sp_horses},
    'onchain': onchain_df,
    'macro': macro_df,
}

astro_cache = {
    'ephemeris': ephem_df,
    'astrology': astro_df,
    'fear_greed': fg_df,
    'google_trends': gtrends_df,
    'funding_daily': funding_daily,
    'open_interest': oi_daily,
}

# Load 1W as higher TF context (1D is NOT the highest — 1W is)
htf_data = {}
try:
    conn_1w = sqlite3.connect(f'{DB_DIR}/btc_prices.db')
    df_1w = pd.read_sql_query("""
        SELECT open_time, open, high, low, close, volume
        FROM ohlcv WHERE timeframe='1w' AND symbol='BTC/USDT'
        ORDER BY open_time
    """, conn_1w)
    conn_1w.close()
    if len(df_1w) > 0:
        df_1w['timestamp'] = pd.to_datetime(df_1w['open_time'], unit='ms', utc=True)
        df_1w = df_1w.drop_duplicates(subset='open_time', keep='last').set_index('timestamp').sort_index()
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df_1w[col] = pd.to_numeric(df_1w[col], errors='coerce')
        htf_data['1w'] = df_1w
        print(f"  1W HTF context: {len(df_1w)} candles")
except Exception as e:
    print(f"  1W HTF context failed: {e}")

print(f"{elapsed()} Calling build_all_features(tf_name='1d', mode='backfill')...")
print(f"  OHLCV: {len(ohlcv)} rows")
print(f"  Space weather: {len(space_weather_df) if space_weather_df is not None else 0} rows")

df = build_all_features(
    ohlcv=ohlcv,
    esoteric_frames=esoteric_frames,
    tf_name='1d',
    mode='backfill',
    htf_data=htf_data,
    astro_cache=astro_cache,
    space_weather_df=space_weather_df,
    include_targets=True,
    include_knn=True,
)

print(f"\n{elapsed()} Feature computation complete.")
print(f"  Result shape: {df.shape}")

# ============================================================
# SAVE TO features_1d.db
# ============================================================
print(f"\n{elapsed()} Saving to features_1d.db...")

df_save = df.copy()
df_save['timestamp'] = df_save.index.strftime('%Y-%m-%d %H:%M:%S')
df_save = df_save.reset_index(drop=True)
cols = ['timestamp'] + [c2 for c2 in df_save.columns if c2 != 'timestamp']
df_save = df_save[cols]

# Save parquet FIRST (no column limit)
parquet_path = f'{DB_DIR}/features_1d.parquet'
df_save.to_parquet(parquet_path, index=False)
print(f"  Saved: {parquet_path} ({len(df_save)} rows x {len(df_save.columns)} cols)", flush=True)

# SQLite backup (may fail if >2000 cols — non-fatal)
try:
    out_db = f'{DB_DIR}/features_1d.db'
    conn_out = sqlite3.connect(out_db)
    df_save.to_sql('features_1d', conn_out, if_exists='replace', index=False)
    conn_out.execute("CREATE INDEX IF NOT EXISTS idx_1d_ts ON features_1d(timestamp)")
    conn_out.close()
except Exception as e:
    print(f"  WARNING: SQLite save failed ({e}) — parquet is primary", flush=True)

target_cols = [c2 for c2 in df_save.columns if c2.startswith('target_') or c2.startswith('next_')]
meta_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'taker_buy_volume',
             'quote_volume', 'trades', 'taker_buy_quote']
feature_cols = [c2 for c2 in df_save.columns if c2 not in target_cols + meta_cols]

import gc
del df
gc.collect()

print(f"\n{'='*70}")
print(f"1D FEATURE MATRIX COMPLETE")
print(f"{'='*70}")
print(f"  Total samples:  {len(df_save)}")
print(f"  Total features: {len(feature_cols)}")
print(f"  Target cols:    {target_cols}")
print(f"  Date range:     {df_save['timestamp'].iloc[0]} to {df_save['timestamp'].iloc[-1]}")
print(f"  Output: {out_db}")
print(f"  Time: {elapsed()}")
