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
build_4h_features.py
=====================
Build 4H feature matrix (~14K samples x 400+ features) from BTC/USDT 4H candles,
then train ML (LightGBM, RF, LASSO) on walk-forward windows,
then run GA strategy optimization.

This should dramatically improve over the daily model (2,367 samples -> 14,194 samples).
"""

import sys, os, time, math, random, warnings
# Unbuffered output for progress logging
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from knn_feature_engine import knn_features_from_ohlcv

DB_DIR = os.path.dirname(os.path.abspath(__file__))
START_TIME = time.time()

def elapsed():
    return f"[{time.time()-START_TIME:.0f}s]"

# ============================================================
# STEP 1: BUILD 4H FEATURE MATRIX
# ============================================================
print("=" * 70)
print("STEP 1: BUILD 4H FEATURE MATRIX")
print("=" * 70)

# --- Load 4H BTC candles ---
print(f"\n{elapsed()} Loading 4H BTC/USDT candles...")
conn = sqlite3.connect(f'{DB_DIR}/btc_prices.db')
btc_4h = pd.read_sql_query("""
    SELECT open_time, open, high, low, close, volume, quote_volume, trades,
           taker_buy_volume, taker_buy_quote
    FROM ohlcv WHERE timeframe='4h' AND symbol='BTC/USDT'
    ORDER BY open_time
""", conn)
conn.close()

btc_4h['timestamp'] = pd.to_datetime(btc_4h['open_time'], unit='ms', utc=True)
btc_4h = btc_4h.drop_duplicates(subset='open_time', keep='last')
btc_4h = btc_4h.set_index('timestamp')
btc_4h = btc_4h.sort_index()

# Convert to float
for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote']:
    btc_4h[col] = pd.to_numeric(btc_4h[col], errors='coerce')

# Filter out rows with zero volume at the start
first_good = btc_4h[btc_4h['volume'] > 0].index[0]
btc_4h = btc_4h.loc[first_good:]

print(f"  4H candles: {len(btc_4h)} ({btc_4h.index.min()} to {btc_4h.index.max()})")

# --- Load 1D candles for higher TF features ---
print(f"{elapsed()} Loading 1D candles...")
conn = sqlite3.connect(f'{DB_DIR}/btc_prices.db')
btc_1d = pd.read_sql_query("""
    SELECT open_time, open, high, low, close, volume
    FROM ohlcv WHERE timeframe='1d' AND symbol='BTC/USDT'
    ORDER BY open_time
""", conn)
conn.close()
btc_1d['timestamp'] = pd.to_datetime(btc_1d['open_time'], unit='ms', utc=True)
btc_1d = btc_1d.drop_duplicates(subset='open_time', keep='last').set_index('timestamp').sort_index()
for col in ['open', 'high', 'low', 'close', 'volume']:
    btc_1d[col] = pd.to_numeric(btc_1d[col], errors='coerce')
print(f"  1D candles: {len(btc_1d)}")

# --- Load 1W candles ---
print(f"{elapsed()} Loading 1W candles...")
conn = sqlite3.connect(f'{DB_DIR}/btc_prices.db')
btc_1w = pd.read_sql_query("""
    SELECT open_time, open, high, low, close, volume
    FROM ohlcv WHERE timeframe='1w' AND symbol='BTC/USDT'
    ORDER BY open_time
""", conn)
conn.close()
btc_1w['timestamp'] = pd.to_datetime(btc_1w['open_time'], unit='ms', utc=True)
btc_1w = btc_1w.drop_duplicates(subset='open_time', keep='last').set_index('timestamp').sort_index()
for col in ['open', 'high', 'low', 'close', 'volume']:
    btc_1w[col] = pd.to_numeric(btc_1w[col], errors='coerce')
print(f"  1W candles: {len(btc_1w)}")

# --- Load auxiliary data ---
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
    news_df = pd.read_sql_query("""
        SELECT timestamp, ts_unix, title, title_gematria, title_dr, sentiment_score, date_doy
        FROM articles ORDER BY timestamp
    """, conn)
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
news_df['date'] = news_df['dt'].dt.date
news_df['date'] = pd.to_datetime(news_df['date'])
news_df['hour'] = news_df['dt'].dt.hour
# Bucket into 4H windows
news_df['ts_unix_num'] = pd.to_numeric(news_df['ts_unix'], errors='coerce')
news_df['bucket_4h'] = (news_df['ts_unix_num'] // 14400) * 14400
print(f"  News: {len(news_df)}")

# Tweets
conn = sqlite3.connect(f'{DB_DIR}/tweets.db')
tweets_df = pd.read_sql_query("""
    SELECT created_at, ts_unix, user_handle, full_text, has_gold, has_red, has_green, dominant_colors,
           gematria_simple, gematria_english
    FROM tweets ORDER BY created_at
""", conn)
conn.close()
tweets_df['dt'] = pd.to_datetime(tweets_df['created_at'], errors='coerce', utc=True)
tweets_df = tweets_df.dropna(subset=['dt'])
tweets_df['date'] = tweets_df['dt'].dt.date
tweets_df['date'] = pd.to_datetime(tweets_df['date'])
tweets_df['ts_unix_num'] = pd.to_numeric(tweets_df['ts_unix'], errors='coerce')
tweets_df['bucket_4h'] = (tweets_df['ts_unix_num'] // 14400) * 14400
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

print(f"\n{elapsed()} All data loaded.\n")

# --- Load space weather data ---
print(f"{elapsed()} Loading space weather data...")
space_weather_df = None
try:
    # 1. Load kp_history_gfz.txt (GFZ Potsdam format)
    kp_rows = []
    kp_path = f'{DB_DIR}/kp_history_gfz.txt'
    if os.path.exists(kp_path):
        with open(kp_path, 'r', encoding='utf-8', errors='replace') as fh:
            for line in fh:
                if line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 26:
                    continue
                try:
                    yyyy = int(parts[0])
                    mm = int(parts[1])
                    dd = int(parts[2])
                    kp_vals = [float(parts[i]) for i in range(7, 15)]
                    kp_mean = np.mean([k for k in kp_vals if k >= 0])
                    ap_val = float(parts[23])
                    sn_val = float(parts[24])
                    f107_val = float(parts[25])
                    dt = datetime(yyyy, mm, dd)
                    kp_rows.append({
                        'date': dt,
                        'kp_mean': kp_mean if not np.isnan(kp_mean) else np.nan,
                        'sunspot_number': sn_val if sn_val >= 0 else np.nan,
                        'solar_flux_f107': f107_val if f107_val >= 0 else np.nan,
                    })
                except (ValueError, IndexError):
                    continue
        print(f"  kp_history_gfz.txt: {len(kp_rows)} daily rows")
    else:
        print(f"  kp_history_gfz.txt not found")

    kp_df = pd.DataFrame(kp_rows) if kp_rows else pd.DataFrame()

    # 2. Load space_weather.db for recent real-time data
    sw_db_path = f'{DB_DIR}/space_weather.db'
    sw_recent = pd.DataFrame()
    if os.path.exists(sw_db_path):
        try:
            sw_conn = sqlite3.connect(sw_db_path)
            sw_recent = pd.read_sql_query("SELECT * FROM space_weather ORDER BY timestamp", sw_conn)
            sw_conn.close()
            if len(sw_recent) > 0:
                sw_recent['date'] = pd.to_datetime(sw_recent['timestamp'], unit='s', utc=True).dt.normalize()
                sw_recent = sw_recent.groupby('date').agg({
                    'kp_index': 'mean',
                    'solar_wind_speed': 'mean',
                    'solar_wind_bz': 'mean',
                    'r_scale': 'max',
                    's_scale': 'max',
                    'g_scale': 'max',
                }).reset_index()
                sw_recent['date'] = pd.to_datetime(sw_recent['date']).dt.tz_localize(None)
                print(f"  space_weather.db: {len(sw_recent)} daily rows")
        except Exception as e:
            print(f"  space_weather.db error: {e}")

    # 3. Combine kp_history_gfz + space_weather.db
    if len(kp_df) > 0:
        space_weather_df = kp_df.copy()
        space_weather_df['date'] = pd.to_datetime(space_weather_df['date'])
        # Merge in real-time fields from space_weather.db
        if len(sw_recent) > 0:
            space_weather_df = space_weather_df.merge(sw_recent, on='date', how='outer')
            space_weather_df = space_weather_df.sort_values('date').reset_index(drop=True)
            # Fill kp_mean from kp_index where missing
            if 'kp_index' in space_weather_df.columns:
                space_weather_df['kp_mean'] = space_weather_df['kp_mean'].fillna(space_weather_df['kp_index'])
        space_weather_df = space_weather_df.set_index('date')
        space_weather_df = space_weather_df.sort_index()
        # Rename kp_mean -> kp_index for feature_library compatibility
        if 'kp_mean' in space_weather_df.columns and 'kp_index' not in space_weather_df.columns:
            space_weather_df = space_weather_df.rename(columns={'kp_mean': 'kp_index'})
        elif 'kp_mean' in space_weather_df.columns and 'kp_index' in space_weather_df.columns:
            space_weather_df['kp_index'] = space_weather_df['kp_index'].fillna(space_weather_df['kp_mean'])
            space_weather_df = space_weather_df.drop(columns=['kp_mean'])
        print(f"  Combined space weather: {len(space_weather_df)} rows, cols={list(space_weather_df.columns)}")
    elif len(sw_recent) > 0:
        space_weather_df = sw_recent.set_index('date').sort_index()
        print(f"  Space weather (db only): {len(space_weather_df)} rows")
    else:
        print(f"  No space weather data found")

    # Localize to UTC to match OHLCV index
    if space_weather_df is not None and space_weather_df.index.tz is None:
        space_weather_df.index = space_weather_df.index.tz_localize('UTC')

except Exception as e:
    print(f"  Space weather loading failed: {e}")
    space_weather_df = None

# --- Load sports data ---
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

print(f"\n{elapsed()} All data loaded.\n")

# ============================================================
# BUILD FEATURES USING FEATURE LIBRARY
# ============================================================
print("=" * 70)
print("Computing 4H features via feature_library...")
print("=" * 70)

from feature_library import build_all_features

# Prepare OHLCV for feature_library
ohlcv = btc_4h.copy()

# Build esoteric_frames dict
esoteric_frames = {
    'tweets': tweets_df,
    'news': news_df,
    'sports': {'games': sp_games, 'horse_races': sp_horses},
    'onchain': onchain_df,
    'macro': macro_df,
}

# Build astro_cache dict
astro_cache = {
    'ephemeris': ephem_df,
    'astrology': astro_df,
    'fear_greed': fg_df,
    'google_trends': gtrends_df,
    'funding_daily': funding_daily,
}

# Build htf_data dict
htf_data = {
    '1d': btc_1d,
    '1w': btc_1w,
}

print(f"{elapsed()} Calling build_all_features(tf_name='4h', mode='backfill')...")
print(f"  OHLCV: {len(ohlcv)} rows")
print(f"  Space weather: {len(space_weather_df) if space_weather_df is not None else 0} rows")

df = build_all_features(
    ohlcv=ohlcv,
    esoteric_frames=esoteric_frames,
    tf_name='4h',
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
# SAVE TO features_4h.db
# ============================================================
print(f"\n{elapsed()} Saving to features_4h.db...")

df_save = df.copy()
import gc
del df
gc.collect()
df_save['timestamp'] = df_save.index.strftime('%Y-%m-%d %H:%M:%S')
df_save = df_save.reset_index(drop=True)

# Move timestamp to front
cols = ['timestamp'] + [c2 for c2 in df_save.columns if c2 != 'timestamp']
df_save = df_save[cols]

out_db = f'{DB_DIR}/features_4h.db'
parquet_path = f'{DB_DIR}/features_4h.parquet'

# Save parquet (primary — no column limit, fast reads)
df_save.to_parquet(parquet_path, index=False)
print(f"  Saved parquet: {parquet_path} ({len(df_save.columns)} cols)")

# Save SQLite (split if needed for legacy compatibility)
MAX_COLS = 1990
conn_out = sqlite3.connect(out_db)
all_cols = list(df_save.columns)
if len(all_cols) <= MAX_COLS:
    df_save.to_sql('features_4h', conn_out, if_exists='replace', index=False)
else:
    chunk1 = all_cols[:MAX_COLS]
    chunk2 = ['timestamp'] + all_cols[MAX_COLS:]
    df_save[chunk1].to_sql('features_4h', conn_out, if_exists='replace', index=False)
    df_save[chunk2].to_sql('features_4h_ext', conn_out, if_exists='replace', index=False)
    print(f"  SQLite split: features_4h ({len(chunk1)} cols) + features_4h_ext ({len(chunk2)} cols)")
conn_out.execute("CREATE INDEX IF NOT EXISTS idx_4h_ts ON features_4h(timestamp)")
conn_out.close()

target_cols = ['next_4h_return', 'next_4h_direction', 'next_3bar_return', 'next_6bar_return',
               'target_return_1bar', 'target_return_6bar', 'target_return_12bar', 'target_return_42bar',
               'target_direction_1bar', 'target_direction_6bar', 'target_direction_12bar', 'target_direction_42bar']
target_cols = [t for t in target_cols if t in df_save.columns]
meta_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades',
             'taker_buy_volume', 'taker_buy_quote']
feature_cols = [c2 for c2 in df_save.columns if c2 not in target_cols + meta_cols]

print(f"\n{'='*70}")
print(f"FEATURE MATRIX COMPLETE")
print(f"{'='*70}")
print(f"  Total samples:  {len(df_save)}")
print(f"  Total features: {len(feature_cols)}")
print(f"  Target cols:    {target_cols}")
print(f"  Date range:     {df_save['timestamp'].iloc[0]} to {df_save['timestamp'].iloc[-1]}")
print(f"  Output: {out_db}")
# ============================================================
# STEP 2: TRAIN ML ON 14K SAMPLES
# ============================================================
print(f"\n\n{'='*70}")
print("STEP 2: TRAIN ML ON 4H DATA (~14K samples)")
print(f"{'='*70}")

# Re-establish feature/target arrays
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

X = df_save[feature_cols].values.astype(np.float32)  # NaN preserved — LightGBM handles missing natively
# Drop rows where label is NaN (can't train on unknown direction)
valid_label_mask = df_save['next_4h_direction'].notna().values
X = X[valid_label_mask]
y = df_save.loc[valid_label_mask, 'next_4h_direction'].values.astype(int)
timestamps = df_save.loc[valid_label_mask, 'timestamp'].values
closes_arr = df_save.loc[valid_label_mask, 'close'].values.astype(float)
returns_4h = df_save.loc[valid_label_mask, 'next_4h_return'].values.astype(float)  # NaN preserved

if 'atr_14' in df_save.columns:
    atr_values = df_save.loc[valid_label_mask, 'atr_14'].values.astype(float)  # NaN preserved — "not enough bars for ATR" != ATR is zero
else:
    atr_values = (df_save.loc[valid_label_mask, 'high'] - df_save.loc[valid_label_mask, 'low']).values.astype(float)

print(f"{elapsed()} Data: {X.shape[0]} rows x {X.shape[1]} features")
print(f"  Target: {np.sum(y==1)} up / {np.sum(y==0)} down")

# LightGBM CUDA does NOT support sparse — always use device='cpu' with force_col_wise=True
USE_GPU = False
print(f"{elapsed()} LightGBM: device='cpu', force_col_wise=True (sparse-compatible)")

# Walk-forward windows (using timestamps)
windows = [
    ('W1', '2020-04-01', '2023-12-31', '2024-01-01', '2024-12-31'),
    ('W2', '2020-04-01', '2024-12-31', '2025-01-01', '2025-12-31'),
    ('W3', '2020-04-01', '2025-12-31', '2026-01-01', '2026-03-31'),
]

all_results = {}
lgb_importances_all = np.zeros(len(feature_cols))
lasso_weights_all = np.zeros(len(feature_cols))
lasso_count = 0

# Store for GA
all_test_indices = []
all_test_probs = []
all_test_y = []
all_test_returns = []
all_test_closes = []
all_test_atrs = []

for wname, tr_start, tr_end, te_start, te_end in windows:
    print(f"\n--- {wname}: Train {tr_start[:7]} to {tr_end[:7]}, Test {te_start[:7]} to {te_end[:7]} ---")

    train_mask = (timestamps >= tr_start) & (timestamps <= tr_end + ' 23:59:59')
    test_mask = (timestamps >= te_start) & (timestamps <= te_end + ' 23:59:59')

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    test_returns = returns_4h[test_mask]
    test_closes_w = closes_arr[test_mask]
    test_atrs_w = atr_values[test_mask]
    test_idx = np.where(test_mask)[0]

    print(f"  Train: {X_train.shape[0]} rows, Test: {X_test.shape[0]} rows")

    if X_test.shape[0] == 0:
        print(f"  SKIP - no test data")
        continue

    results = {}

    # --- LightGBM ---
    t0 = time.time()
    lgb_model = lgb.LGBMClassifier(
        max_depth=6, learning_rate=0.05, n_estimators=500,
        colsample_bytree=0.8, min_data_in_leaf=5,  # 4H per config: rare astro signals
        objective='binary', random_state=42, verbose=-1,
        device='cpu', force_col_wise=True,
    )
    lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

    lgb_pred = lgb_model.predict(X_test)
    lgb_prob = lgb_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, lgb_pred)
    prec = precision_score(y_test, lgb_pred, zero_division=0)
    rec = recall_score(y_test, lgb_pred, zero_division=0)
    results['LightGBM'] = {'acc': acc, 'prec': prec, 'rec': rec}
    print(f"  LightGBM: Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  ({time.time()-t0:.1f}s)")
    lgb_importances_all += lgb_model.feature_importances_

    all_test_indices.extend(test_idx.tolist())
    all_test_probs.extend(lgb_prob.tolist())
    all_test_y.extend(y_test.tolist())
    all_test_returns.extend(test_returns.tolist())
    all_test_closes.extend(test_closes_w.tolist())
    all_test_atrs.extend(test_atrs_w.tolist())

    # --- Random Forest ---
    t0 = time.time()
    rf_model = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=10,
        max_features='sqrt', random_state=42, n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, rf_pred)
    prec = precision_score(y_test, rf_pred, zero_division=0)
    rec = recall_score(y_test, rf_pred, zero_division=0)
    results['RandomForest'] = {'acc': acc, 'prec': prec, 'rec': rec}
    print(f"  RF:       Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  ({time.time()-t0:.1f}s)")

    # --- LASSO ---
    t0 = time.time()
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    # sklearn LinearModel (LASSO) cannot handle NaN — nan_to_num required here only.
    # LightGBM paths above preserve NaN natively; this is sklearn-specific.
    X_train_sc = np.nan_to_num(X_train_sc, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_sc = np.nan_to_num(X_test_sc, nan=0.0, posinf=0.0, neginf=0.0)

    lasso_model = LogisticRegression(
        penalty='l1', C=0.1, solver='saga', max_iter=5000, random_state=42, n_jobs=-1,
    )
    lasso_model.fit(X_train_sc, y_train)
    lasso_pred = lasso_model.predict(X_test_sc)
    acc = accuracy_score(y_test, lasso_pred)
    prec = precision_score(y_test, lasso_pred, zero_division=0)
    rec = recall_score(y_test, lasso_pred, zero_division=0)
    results['LASSO'] = {'acc': acc, 'prec': prec, 'rec': rec}
    print(f"  LASSO:    Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  ({time.time()-t0:.1f}s)")

    lasso_weights_all += np.abs(lasso_model.coef_[0])
    lasso_count += 1

    all_results[wname] = results

# Average importances
n_windows = max(len([w for w in windows if np.sum((timestamps >= w[3]) & (timestamps <= w[4] + ' 23:59:59')) > 0]), 1)
lgb_importances_avg = lgb_importances_all / n_windows
if lasso_count > 0:
    lasso_weights_avg = lasso_weights_all / lasso_count
else:
    lasso_weights_avg = lasso_weights_all

# TOP 40 LightGBM features
lgb_top_idx = np.argsort(lgb_importances_avg)[::-1][:40]
print(f"\n{'='*70}")
print("TOP 40 FEATURES BY LIGHTGBM IMPORTANCE (4H)")
print(f"{'='*70}")
for rank, idx in enumerate(lgb_top_idx):
    fname = feature_cols[idx]
    imp = lgb_importances_avg[idx]
    print(f"  {rank+1:2d}. {fname:45s} {imp:.6f}")

# LASSO non-zero weights
lasso_nonzero_idx = np.where(lasso_weights_avg > 1e-6)[0]
lasso_sorted = sorted(lasso_nonzero_idx, key=lambda i: lasso_weights_avg[i], reverse=True)
print(f"\n{'='*70}")
print(f"LASSO NON-ZERO WEIGHTS ({len(lasso_sorted)} features)")
print(f"{'='*70}")
for idx in lasso_sorted[:40]:
    fname = feature_cols[idx]
    w = lasso_weights_avg[idx]
    print(f"  {fname:45s} {w:.6f}")

# Comparison with daily
print(f"\n{'='*70}")
print("COMPARISON: 4H vs DAILY MODEL")
print(f"{'='*70}")
print(f"  Daily: 2,367 samples, 477 features, ~62.9% accuracy")
print(f"  4H:    {len(df_save)} samples, {len(feature_cols)} features")
for wname, res in all_results.items():
    for model, metrics in res.items():
        print(f"    {wname} {model:12s}: Acc={metrics['acc']:.4f}  Prec={metrics['prec']:.4f}")

# ============================================================
# STEP 3: GA STRATEGY OPTIMIZATION ON 4H
# ============================================================
print(f"\n\n{'='*70}")
print("STEP 3: GA STRATEGY OPTIMIZATION ON 4H DATA")
print(f"{'='*70}")

# Convert pooled test data
ga_probs = np.array(all_test_probs)
ga_y = np.array(all_test_y)
ga_returns = np.array(all_test_returns)
ga_closes = np.array(all_test_closes)
ga_atrs = np.array(all_test_atrs)
ga_indices = np.array(all_test_indices)

sort_order = np.argsort(ga_indices)
ga_probs = ga_probs[sort_order]
ga_y = ga_y[sort_order]
ga_returns = ga_returns[sort_order]
ga_closes = ga_closes[sort_order]
ga_atrs = ga_atrs[sort_order]

print(f"{elapsed()} Pooled test data: {len(ga_probs)} rows")

# GA parameters
LEVERAGE_VALS = [1, 2, 3, 5, 7, 10, 15, 20, 25]
RISK_VALS = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15]
STOP_ATR_VALS = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
RR_VALS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
TRAIL_ATR_VALS = [0.5, 0.8, 1.0, 1.5, 2.0]
MAX_HOLD_VALS_4H = [1, 2, 4, 6, 8, 12, 24, 48, 96]  # in 4H candles (4h to 16 days)
PARTIAL_TP_VALS = [0, 0.25, 0.50, 0.75]
CONF_THRESH_VALS = [0.50, 0.55, 0.60, 0.65, 0.70]

GENOME_LEN = 30 + 4 + 3 + 3 + 3 + 3 + 4 + 2 + 3  # = 55
from config import FEE_RATE as CONFIG_FEE_RATE
FEE = CONFIG_FEE_RATE  # single source of truth from config.py
BALANCE_CAP = 100000.0
POP_SIZE = 150
N_GENS = 200

# Consensus top 30 for GA
lgb_rank = {feature_cols[idx]: rank for rank, idx in enumerate(np.argsort(lgb_importances_avg)[::-1])}
consensus = {}
for f in feature_cols:
    consensus[f] = lgb_rank.get(f, len(feature_cols))
top30_features = sorted(consensus, key=consensus.get)[:30]

def decode_genome(genome):
    pos = 0
    feature_mask = genome[pos:pos+30]; pos += 30
    lev_idx = int(''.join(map(str, genome[pos:pos+4])), 2) % len(LEVERAGE_VALS); pos += 4
    risk_idx = int(''.join(map(str, genome[pos:pos+3])), 2) % len(RISK_VALS); pos += 3
    stop_idx = int(''.join(map(str, genome[pos:pos+3])), 2) % len(STOP_ATR_VALS); pos += 3
    rr_idx = int(''.join(map(str, genome[pos:pos+3])), 2) % len(RR_VALS); pos += 3
    trail_idx = int(''.join(map(str, genome[pos:pos+3])), 2) % len(TRAIL_ATR_VALS); pos += 3
    hold_idx = int(''.join(map(str, genome[pos:pos+4])), 2) % len(MAX_HOLD_VALS_4H); pos += 4
    ptp_idx = int(''.join(map(str, genome[pos:pos+2])), 2) % len(PARTIAL_TP_VALS); pos += 2
    conf_idx = int(''.join(map(str, genome[pos:pos+3])), 2) % len(CONF_THRESH_VALS); pos += 3

    return {
        'feature_mask': feature_mask,
        'leverage': LEVERAGE_VALS[lev_idx],
        'risk_pct': RISK_VALS[risk_idx],
        'stop_atr': STOP_ATR_VALS[stop_idx],
        'rr': RR_VALS[rr_idx],
        'trail_atr': TRAIL_ATR_VALS[trail_idx],
        'max_hold': MAX_HOLD_VALS_4H[hold_idx],
        'partial_tp': PARTIAL_TP_VALS[ptp_idx],
        'conf_thresh': CONF_THRESH_VALS[conf_idx],
    }


def simulate_strategy(params, probs, returns, closes, atrs):
    balance = 10000.0
    peak_balance = balance
    max_dd = 0.0
    trades_count = 0
    wins = 0
    total_pnl = 0.0

    leverage = params['leverage']
    base_risk = params['risk_pct']
    stop_atr_mult = params['stop_atr']
    rr = params['rr']
    trail_atr_mult = params['trail_atr']
    max_hold = params['max_hold']
    partial_tp = params['partial_tp']
    conf_thresh = params['conf_thresh']

    i = 0
    n = len(probs)

    while i < n:
        if balance <= 0:
            break

        prob = probs[i]
        if prob >= conf_thresh:
            direction = 1
        elif prob <= (1 - conf_thresh):
            direction = -1
        else:
            i += 1
            continue

        risk_amount = balance * base_risk
        entry_price = closes[i]
        atr = atrs[i] if atrs[i] > 0 else entry_price * 0.005  # 0.5% for 4H
        stop_dist = atr * stop_atr_mult

        if stop_dist <= 0 or entry_price <= 0:
            i += 1
            continue

        position_size = (risk_amount / stop_dist) * entry_price
        position_size = min(position_size, balance * leverage)
        entry_fee = position_size * FEE

        tp_dist = stop_dist * rr
        trail_dist = atr * trail_atr_mult
        best_price = entry_price
        exit_price = entry_price
        hold_bars = 0
        partial_taken = False
        partial_pnl = 0.0

        for j in range(1, min(max_hold + 1, n - i)):
            idx = i + j
            if idx >= n:
                break
            current_price = closes[idx]
            hold_bars = j
            price_move = (current_price - entry_price) * direction

            if direction == 1:
                best_price = max(best_price, current_price)
                trail_stop = best_price - trail_dist
                hit_stop = current_price <= entry_price - stop_dist or current_price <= trail_stop
            else:
                if current_price < best_price or best_price == entry_price:
                    best_price = current_price
                trail_stop = best_price + trail_dist
                hit_stop = current_price >= entry_price + stop_dist or current_price >= trail_stop

            if not partial_taken and partial_tp > 0 and price_move >= tp_dist * 0.5:
                partial_pnl = (position_size * partial_tp) * (price_move / entry_price) - (position_size * partial_tp * FEE)
                partial_taken = True

            if price_move >= tp_dist:
                exit_price = entry_price + (tp_dist * direction)
                break

            if hit_stop:
                exit_price = current_price
                break

            exit_price = current_price

        remaining_size = position_size * (1 - partial_tp) if partial_taken else position_size
        move_pct = (exit_price - entry_price) / entry_price * direction
        trade_pnl = remaining_size * move_pct + partial_pnl
        exit_fee = remaining_size * FEE
        trade_pnl -= (entry_fee + exit_fee)

        balance += trade_pnl
        balance = min(balance, BALANCE_CAP)
        total_pnl += trade_pnl
        trades_count += 1

        if trade_pnl > 0:
            wins += 1

        if balance > peak_balance:
            peak_balance = balance
        dd = (peak_balance - balance) / peak_balance * 100 if peak_balance > 0 else 0
        max_dd = max(max_dd, dd)

        i += max(hold_bars, 1)

    roi = (balance - 10000) / 10000 * 100
    wr = (wins / trades_count * 100) if trades_count > 0 else 0

    return {
        'balance': balance,
        'roi': roi,
        'max_dd': max_dd,
        'wr': wr,
        'trades': trades_count,
        'wins': wins,
    }


def ga_fitness(params, probs, returns, closes, atrs):
    res = simulate_strategy(params, probs, returns, closes, atrs)
    roi = res['roi']
    dd = res['max_dd']
    wr = res['wr']
    trades_count = res['trades']
    if trades_count < 5:
        return -9999
    return roi * (1 - dd / 100) * max(wr / 50, 0.5)


def create_random_genome():
    return [random.randint(0, 1) for _ in range(GENOME_LEN)]

def crossover(p1, p2):
    a, b = sorted(random.sample(range(GENOME_LEN), 2))
    return p1[:a] + p2[a:b] + p1[b:]

def mutate(genome, rate=0.05):
    return [g if random.random() > rate else (1 - g) for g in genome]


print(f"{elapsed()} Running GA: pop={POP_SIZE}, gens={N_GENS}")
print(f"  Test data points: {len(ga_probs)}")

population = [create_random_genome() for _ in range(POP_SIZE)]
best_fitness_ever = -9999
best_genome_ever = None

for gen in range(N_GENS):
    fitnesses = []
    for genome in population:
        params = decode_genome(genome)
        f = ga_fitness(params, ga_probs, ga_returns, ga_closes, ga_atrs)
        fitnesses.append(f)

    fitnesses_arr = np.array(fitnesses)
    best_idx = np.argmax(fitnesses_arr)
    gen_best = fitnesses_arr[best_idx]
    gen_avg = np.mean(fitnesses_arr[fitnesses_arr > -9000])

    if gen_best > best_fitness_ever:
        best_fitness_ever = gen_best
        best_genome_ever = population[best_idx].copy()

    if (gen + 1) % 25 == 0 or gen == 0:
        best_params = decode_genome(best_genome_ever)
        best_res = simulate_strategy(best_params, ga_probs, ga_returns, ga_closes, ga_atrs)
        print(f"  {elapsed()} Gen {gen+1:3d}: Fit={best_fitness_ever:10.2f}  "
              f"ROI={best_res['roi']:8.2f}%  DD={best_res['max_dd']:5.2f}%  "
              f"WR={best_res['wr']:5.1f}%  Trades={best_res['trades']}")

    # Selection + reproduction
    new_pop = [best_genome_ever.copy()]
    while len(new_pop) < POP_SIZE:
        t1 = random.sample(range(POP_SIZE), 5)
        t2 = random.sample(range(POP_SIZE), 5)
        p1 = population[max(t1, key=lambda x: fitnesses_arr[x])]
        p2 = population[max(t2, key=lambda x: fitnesses_arr[x])]
        child = crossover(p1, p2)
        child = mutate(child, rate=0.03 + 0.02 * (gen / N_GENS))
        new_pop.append(child)
    population = new_pop

# Final results
best_params = decode_genome(best_genome_ever)
best_result = simulate_strategy(best_params, ga_probs, ga_returns, ga_closes, ga_atrs)

print(f"\n{elapsed()} GA COMPLETE")
print(f"\n{'='*70}")
print("GA BEST STRATEGY (4H)")
print(f"{'='*70}")
print(f"  Leverage:     {best_params['leverage']}x")
print(f"  Risk %:       {best_params['risk_pct']*100:.1f}%")
print(f"  Stop ATR:     {best_params['stop_atr']}")
print(f"  R:R Ratio:    {best_params['rr']}")
print(f"  Trail ATR:    {best_params['trail_atr']}")
print(f"  Max Hold:     {best_params['max_hold']} bars ({best_params['max_hold']*4}h)")
print(f"  Partial TP:   {best_params['partial_tp']*100:.0f}%")
print(f"  Conf Thresh:  {best_params['conf_thresh']}")

active_features = [top30_features[i] for i, bit in enumerate(best_params['feature_mask']) if bit == 1]
print(f"  Active Features ({len(active_features)}):")
for f in active_features:
    print(f"    - {f}")

print(f"\n{'='*70}")
print("GA BEST PERFORMANCE (4H)")
print(f"{'='*70}")
print(f"  ROI:          {best_result['roi']:+.2f}%")
print(f"  Max DD:       {best_result['max_dd']:.2f}%")
print(f"  Win Rate:     {best_result['wr']:.1f}%")
print(f"  Trades:       {best_result['trades']}")
print(f"  Final Balance:${best_result['balance']:,.2f}")

# ============================================================
# SAVE RESULTS
# ============================================================
results_file = f'{DB_DIR}/ml_4h_results.txt'
with open(results_file, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("4H ML + GA RESULTS\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"FEATURE MATRIX\n")
    f.write(f"  Samples:  {len(df_save)}\n")
    f.write(f"  Features: {len(feature_cols)}\n")
    f.write(f"  Range:    {df_save['timestamp'].iloc[0]} to {df_save['timestamp'].iloc[-1]}\n\n")

    f.write("WALK-FORWARD ML RESULTS\n")
    f.write("-" * 50 + "\n")
    for wname, res in all_results.items():
        f.write(f"\n  {wname}:\n")
        for model, metrics in res.items():
            f.write(f"    {model:12s}: Acc={metrics['acc']:.4f}  Prec={metrics['prec']:.4f}  Rec={metrics['rec']:.4f}\n")

    f.write(f"\nCOMPARISON\n")
    f.write(f"  Daily: 2,367 samples, 477 features, ~62.9% accuracy\n")
    f.write(f"  4H:    {len(df_save)} samples, {len(feature_cols)} features\n\n")

    f.write("TOP 40 FEATURES (LightGBM)\n")
    f.write("-" * 50 + "\n")
    for rank, idx in enumerate(lgb_top_idx):
        fname = feature_cols[idx]
        imp = lgb_importances_avg[idx]
        f.write(f"  {rank+1:2d}. {fname:45s} {imp:.6f}\n")

    f.write(f"\nLASSO NON-ZERO ({len(lasso_sorted)} features)\n")
    f.write("-" * 50 + "\n")
    for idx in lasso_sorted[:40]:
        fname = feature_cols[idx]
        w = lasso_weights_avg[idx]
        f.write(f"  {fname:45s} {w:.6f}\n")

    f.write(f"\nGA BEST STRATEGY\n")
    f.write("-" * 50 + "\n")
    f.write(f"  Leverage:     {best_params['leverage']}x\n")
    f.write(f"  Risk %:       {best_params['risk_pct']*100:.1f}%\n")
    f.write(f"  Stop ATR:     {best_params['stop_atr']}\n")
    f.write(f"  R:R Ratio:    {best_params['rr']}\n")
    f.write(f"  Trail ATR:    {best_params['trail_atr']}\n")
    f.write(f"  Max Hold:     {best_params['max_hold']} bars ({best_params['max_hold']*4}h)\n")
    f.write(f"  Partial TP:   {best_params['partial_tp']*100:.0f}%\n")
    f.write(f"  Conf Thresh:  {best_params['conf_thresh']}\n")
    f.write(f"  Active Features: {active_features}\n\n")

    f.write(f"GA PERFORMANCE\n")
    f.write("-" * 50 + "\n")
    f.write(f"  ROI:          {best_result['roi']:+.2f}%\n")
    f.write(f"  Max DD:       {best_result['max_dd']:.2f}%\n")
    f.write(f"  Win Rate:     {best_result['wr']:.1f}%\n")
    f.write(f"  Trades:       {best_result['trades']}\n")
    f.write(f"  Final Balance:${best_result['balance']:,.2f}\n")

print(f"\n{elapsed()} Results saved to: {results_file}")
print(f"\nDONE.")
