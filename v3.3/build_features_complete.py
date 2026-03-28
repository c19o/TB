#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_features_complete.py
===========================
Build daily (1D) feature matrix from BTC/USDT daily candles using the unified
feature_library.build_all_features() pipeline.

Step 1: Download/update 11 macro tickers from yfinance into macro_data.db
Step 2: Load ALL source DBs and call build_all_features() for consistent
        feature computation across all timeframes.

Saves to features_complete.db.
"""

import sys, os, io, time, warnings
from datetime import datetime
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
except:
    pass
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import sqlite3

DB_DIR = os.path.dirname(os.path.abspath(__file__))
START_TIME = time.time()

def elapsed():
    return f"[{time.time()-START_TIME:.0f}s]"

# ============================================================
# PART 1: DOWNLOAD 11 MACRO TICKERS FROM YFINANCE
# ============================================================
print("=" * 70)
print("PART 1: Downloading macro tickers from yfinance...")
print("=" * 70)

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    print("  [WARN] yfinance not installed — skipping macro download")

if HAS_YFINANCE:
    conn_macro = sqlite3.connect(f'{DB_DIR}/macro_data.db')

    tickers = {
        '^VIX': 'vix',
        '^TNX': 'us10y',
        '^IXIC': 'nasdaq',
        '^RUT': 'russell',
        'CL=F': 'oil',
        'SI=F': 'silver',
        'MSTR': 'mstr',
        'COIN': 'coin',
        'HYG': 'hyg',
        'TLT': 'tlt',
        'IBIT': 'ibit',
    }

    for col in tickers.values():
        try:
            conn_macro.execute(f'ALTER TABLE macro_data ADD COLUMN {col}_close REAL')
        except:
            pass

    for ticker, name in tickers.items():
        try:
            data = yf.download(ticker, start='2019-01-01', progress=False)
            if data.empty:
                print(f'  {ticker} ({name}): no data')
                continue
            count = 0
            for date_idx, row in data.iterrows():
                d = date_idx.strftime('%Y-%m-%d')
                try:
                    val = float(row['Close'].iloc[0]) if hasattr(row['Close'], 'iloc') else float(row['Close'])
                except:
                    val = float(row['Close'])
                conn_macro.execute(f'UPDATE macro_data SET {name} = ? WHERE date = ?', (val, d))
                count += 1
            conn_macro.commit()
            print(f'  {ticker} ({name}): {count} rows updated')
        except Exception as e:
            print(f'  {ticker}: {e}')

    conn_macro.close()
    print("Macro download complete.\n")
else:
    print("  Skipping PART 1 (no yfinance). Using existing macro_data.db if available.\n")

# ============================================================
# PART 2: LOAD ALL DATA SOURCES
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
    btc[col] = pd.to_numeric(btc[col], errors='coerce')
first_good = btc[btc['volume'] > 0].index[0]
btc = btc.loc[first_good:]
print(f"  1D candles: {len(btc)} ({btc.index.min()} to {btc.index.max()})")

# --- Higher TF candles ---
print(f"{elapsed()} Loading higher TF candles...")
conn = sqlite3.connect(f'{DB_DIR}/btc_prices.db')
btc_1w = pd.read_sql_query("SELECT open_time, open, high, low, close, volume FROM ohlcv WHERE timeframe='1w' AND symbol='BTC/USDT' ORDER BY open_time", conn)
conn.close()

btc_1w['timestamp'] = pd.to_datetime(btc_1w['open_time'], unit='ms', utc=True)
btc_1w.drop_duplicates(subset='open_time', keep='last', inplace=True)
btc_1w.set_index('timestamp', inplace=True)
btc_1w.sort_index(inplace=True)
for col in ['open', 'high', 'low', 'close', 'volume']:
    btc_1w[col] = pd.to_numeric(btc_1w[col], errors='coerce')
print(f"  1W: {len(btc_1w)} candles")

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
try:
    news_df = pd.read_sql_query("SELECT timestamp, ts_unix, title, title_gematria, title_dr, sentiment_score, date_doy FROM articles ORDER BY timestamp", conn)
except:
    news_df = pd.read_sql_query("SELECT timestamp, ts_unix, title FROM streamer_articles ORDER BY timestamp", conn)
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
           gematria_simple, gematria_english, favorite_count, retweet_count, reply_count
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

# Sports
print(f"{elapsed()} Loading sports data...")
sp_games = pd.DataFrame()
sp_horses = pd.DataFrame()
try:
    sp_conn = sqlite3.connect(f'{DB_DIR}/sports_results.db', timeout=5)
    sp_games = pd.read_sql_query("SELECT * FROM games ORDER BY date", sp_conn)
    try:
        sp_horses = pd.read_sql_query("SELECT * FROM horse_races ORDER BY date", sp_conn)
    except:
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
}

htf_data = {
    '1w': btc_1w,
}

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
# SAVE TO features_complete.db
# ============================================================
print(f"\n{elapsed()} Saving to features_complete.db...")

df_save = df.copy()
df_save['timestamp'] = df_save.index.strftime('%Y-%m-%d %H:%M:%S')
df_save = df_save.reset_index(drop=True)
cols = ['timestamp'] + [c2 for c2 in df_save.columns if c2 != 'timestamp']
df_save = df_save[cols]

out_db = f'{DB_DIR}/features_complete.db'
conn_out = sqlite3.connect(out_db)
df_save.to_sql('features', conn_out, if_exists='replace', index=False)
conn_out.execute("CREATE INDEX IF NOT EXISTS idx_complete_ts ON features(timestamp)")
conn_out.close()

target_cols = [c2 for c2 in df_save.columns if c2.startswith('target_') or c2.startswith('next_')]
meta_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'taker_buy_volume',
             'quote_volume', 'trades', 'taker_buy_quote']
feature_cols = [c2 for c2 in df_save.columns if c2 not in target_cols + meta_cols]

print(f"\n{'='*70}")
print(f"1D FEATURE MATRIX COMPLETE")
print(f"{'='*70}")
print(f"  Total samples:  {len(df_save)}")
print(f"  Total features: {len(feature_cols)}")
print(f"  Target cols:    {target_cols}")
print(f"  Date range:     {df_save['timestamp'].iloc[0]} to {df_save['timestamp'].iloc[-1]}")
print(f"  Output: {out_db}")
print(f"  Time: {elapsed()}")
