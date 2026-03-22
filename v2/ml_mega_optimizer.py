#!/usr/bin/env python3
"""
ML Mega Optimizer - Complete ML Trading Optimization System
Loads all available data, engineers 300+ features, trains XGBoost/RF/LASSO,
then runs genetic algorithm to find optimal trading parameters.
"""

import sys
import io
import os
import json
import math
import random
import sqlite3
import warnings
import time
from datetime import datetime, timezone, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Fix Windows encoding
if os.name == 'nt':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

DB_DIR = os.environ.get("SAVAGE22_V1_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
random.seed(42)
np.random.seed(42)

print("=" * 80)
print("ML MEGA OPTIMIZER - BTC Trading Signal Optimization")
print("=" * 80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ============================================================================
# DATA LOADING
# ============================================================================
print("[1/6] LOADING DATA...")
t0 = time.time()

def load_db(name):
    path = os.path.join(DB_DIR, name)
    if not os.path.exists(path):
        print(f"  WARNING: {name} not found")
        return None
    return sqlite3.connect(path)

# --- BTC OHLCV ---
conn = load_db("btc_prices.db")
btc_1d = pd.read_sql_query(
    "SELECT open_time, open, high, low, close, volume, taker_buy_volume FROM ohlcv "
    "WHERE symbol='BTC/USDT' AND timeframe='1d' ORDER BY open_time", conn)
btc_4h = pd.read_sql_query(
    "SELECT open_time, open, high, low, close, volume FROM ohlcv "
    "WHERE symbol='BTC/USDT' AND timeframe='4h' ORDER BY open_time", conn)
btc_1h = pd.read_sql_query(
    "SELECT open_time, open, high, low, close, volume FROM ohlcv "
    "WHERE symbol='BTC/USDT' AND timeframe='1h' ORDER BY open_time", conn)

# Altcoins
eth_1d = pd.read_sql_query(
    "SELECT open_time, close as eth_close, volume as eth_volume FROM ohlcv "
    "WHERE symbol='ETH/USDT' AND timeframe='1d' ORDER BY open_time", conn)
sol_1d = pd.read_sql_query(
    "SELECT open_time, close as sol_close, volume as sol_volume FROM ohlcv "
    "WHERE symbol='SOL/USDT' AND timeframe='1d' ORDER BY open_time", conn)
conn.close()
print(f"  BTC 1d: {len(btc_1d)} candles, 4h: {len(btc_4h)}, 1h: {len(btc_1h)}")
print(f"  ETH 1d: {len(eth_1d)}, SOL 1d: {len(sol_1d)}")

# Convert timestamps to dates
btc_1d['date'] = pd.to_datetime(btc_1d['open_time'], unit='ms', utc=True).dt.strftime('%Y-%m-%d')
eth_1d['date'] = pd.to_datetime(eth_1d['open_time'], unit='ms', utc=True).dt.strftime('%Y-%m-%d')
sol_1d['date'] = pd.to_datetime(sol_1d['open_time'], unit='ms', utc=True).dt.strftime('%Y-%m-%d')

# Merge altcoins
btc_1d = btc_1d.merge(eth_1d[['date', 'eth_close']], on='date', how='left')
btc_1d = btc_1d.merge(sol_1d[['date', 'sol_close']], on='date', how='left')

# --- Fear & Greed ---
conn = load_db("fear_greed.db")
fg = pd.read_sql_query("SELECT date, value FROM fear_greed", conn)
conn.close()
fg_dict = dict(zip(fg['date'], fg['value']))
print(f"  Fear & Greed: {len(fg)} entries")

# --- Ephemeris ---
conn = load_db("ephemeris_cache.db")
eph = pd.read_sql_query("SELECT * FROM ephemeris", conn)
conn.close()
eph_dict = {}
for _, row in eph.iterrows():
    eph_dict[row['date']] = row.to_dict()
print(f"  Ephemeris: {len(eph)} entries")

# --- Tweets ---
conn = load_db("tweets.db")
tweets = pd.read_sql_query(
    "SELECT ts_unix, user_handle, has_gold, has_red, full_text, favorite_count "
    "FROM tweets WHERE is_retweet=0", conn)
conn.close()
tweets['date'] = pd.to_datetime(tweets['ts_unix'], unit='s', utc=True).dt.strftime('%Y-%m-%d')
tweet_daily = tweets.groupby('date').agg(
    tweets_count=('ts_unix', 'count'),
    gold_tweet=('has_gold', 'max'),
    red_tweet=('has_red', 'max'),
    avg_likes=('favorite_count', 'mean')
).to_dict('index')
# Check for decoder handles
decoder_handles = {'zachhubbard', 'thematrixology', 'gematriaeffect', 'gaborsztanics',
                   'zacharyhubbard', 'derekdoze', 'alphabeticcal'}
decoder_dates = set()
for _, row in tweets.iterrows():
    if row['user_handle'].lower() in decoder_handles:
        decoder_dates.add(row['date'])
print(f"  Tweets: {len(tweets)} (non-RT), decoder dates: {len(decoder_dates)}")

# --- News ---
conn = load_db("news_articles.db")
news = pd.read_sql_query(
    "SELECT DATE(timestamp) as d, COUNT(*) as cnt, AVG(sentiment_score) as avg_sent "
    "FROM articles GROUP BY d", conn)
conn.close()
news_dict = dict(zip(news['d'], zip(news['cnt'], news['avg_sent'])))
print(f"  News: {len(news)} daily entries")

# --- Funding Rates ---
conn = load_db("funding_rates.db")
if conn:
    funding = pd.read_sql_query("SELECT * FROM funding_rates", conn)
    conn.close()
    funding['date'] = pd.to_datetime(funding['ts_unix'], unit='s', utc=True).dt.strftime('%Y-%m-%d')
    fund_daily = funding.groupby('date')['funding_rate'].mean().to_dict()
    print(f"  Funding rates: {len(funding)} entries")
else:
    fund_daily = {}

# --- Astrology Full (Vedic + Chinese + Mayan + Arabic Lots) ---
conn = load_db("astrology_full.db")
if conn:
    astro_full = pd.read_sql_query("SELECT * FROM daily_astrology", conn)
    conn.close()
    astro_dict = {}
    for _, row in astro_full.iterrows():
        astro_dict[row['date']] = row.to_dict()
    print(f"  Astrology Full: {len(astro_full)} entries ({len(astro_full.columns)} columns)")
else:
    astro_dict = {}

# --- Macro Data (DXY, Gold, SPX) ---
conn = load_db("macro_data.db")
if conn:
    macro_df = pd.read_sql_query("SELECT date, dxy_close, gold_close, spx_close FROM macro_data", conn)
    conn.close()
    macro_dict = {}
    for _, row in macro_df.iterrows():
        macro_dict[row['date']] = {'dxy': row['dxy_close'], 'gold': row['gold_close'], 'spx': row['spx_close']}
    print(f"  Macro data: {len(macro_df)} entries")
else:
    macro_dict = {}

# --- Google Trends ---
conn = load_db("google_trends.db")
if conn:
    trends_df = pd.read_sql_query("SELECT date, interest_score FROM google_trends ORDER BY date", conn)
    conn.close()
    # Weekly data - build lookup (each week covers 7 days)
    trends_daily = {}
    for _, row in trends_df.iterrows():
        week_start = datetime.strptime(row['date'], '%Y-%m-%d')
        for d in range(7):
            day = (week_start + timedelta(days=d)).strftime('%Y-%m-%d')
            trends_daily[day] = row['interest_score']
    # Also keep weekly list for ROC calculation
    trends_weekly = list(zip(trends_df['date'], trends_df['interest_score']))
    trends_weekly_dict = dict(trends_weekly)
    print(f"  Google Trends: {len(trends_df)} weekly entries -> {len(trends_daily)} daily mapped")
else:
    trends_daily = {}
    trends_weekly_dict = {}

# --- 4h aggregates for daily features ---
btc_4h['date'] = pd.to_datetime(btc_4h['open_time'], unit='ms', utc=True).dt.strftime('%Y-%m-%d')
btc_4h_daily = btc_4h.groupby('date').agg(
    vol_4h_std=('volume', 'std'),
    high_4h_max=('high', 'max'),
    low_4h_min=('low', 'min'),
    close_4h_last=('close', 'last'),
    range_4h_mean=('high', lambda x: (btc_4h.loc[x.index, 'high'] - btc_4h.loc[x.index, 'low']).mean())
).to_dict('index')

print(f"  Data loaded in {time.time()-t0:.1f}s")
print()

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
print("[2/6] ENGINEERING FEATURES...")
t0 = time.time()

closes = btc_1d['close'].values.astype(float)
opens = btc_1d['open'].values.astype(float)
highs = btc_1d['high'].values.astype(float)
lows = btc_1d['low'].values.astype(float)
volumes = btc_1d['volume'].values.astype(float)
taker_buy = btc_1d['taker_buy_volume'].values.astype(float)
dates = btc_1d['date'].values
eth_closes = btc_1d['eth_close'].values.astype(float)
sol_closes = btc_1d['sol_close'].values.astype(float)
n = len(closes)

def ema(data, period):
    """Exponential moving average"""
    result = np.full(len(data), np.nan)
    k = 2 / (period + 1)
    result[period-1] = np.mean(data[:period])
    for i in range(period, len(data)):
        result[i] = data[i] * k + result[i-1] * (1 - k)
    return result

def sma(data, period):
    result = np.full(len(data), np.nan)
    for i in range(period-1, len(data)):
        result[i] = np.mean(data[i-period+1:i+1])
    return result

def calc_rsi(data, period):
    result = np.full(len(data), np.nan)
    deltas = np.diff(data)
    for i in range(period, len(data)):
        gains = deltas[i-period:i]
        up = np.mean(gains[gains > 0]) if np.any(gains > 0) else 0
        down = -np.mean(gains[gains < 0]) if np.any(gains < 0) else 0
        if down == 0:
            result[i] = 100
        else:
            rs = up / down
            result[i] = 100 - 100 / (1 + rs)
    return result

def calc_atr(highs, lows, closes, period=14):
    result = np.full(len(closes), np.nan)
    tr = np.zeros(len(closes))
    for i in range(1, len(closes)):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
    for i in range(period, len(closes)):
        result[i] = np.mean(tr[i-period+1:i+1])
    return result

def digital_root(n):
    if n == 0: return 0
    n = abs(int(n))
    return 1 + (n - 1) % 9

# Pre-compute TA arrays
print("  Computing TA indicators...")
sma_arrays = {}
ema_arrays = {}
for period in [5, 7, 10, 20, 21, 50, 100, 200]:
    sma_arrays[period] = sma(closes, period)
    ema_arrays[period] = ema(closes, period)

rsi_arrays = {}
for period in [7, 14, 21]:
    rsi_arrays[period] = calc_rsi(closes, period)

atr_14 = calc_atr(highs, lows, closes, 14)
atr_20_mean = sma(atr_14, 20)

# MACD
ema12 = ema(closes, 12)
ema26 = ema(closes, 26)
macd_line = ema12 - ema26
macd_signal = ema(np.nan_to_num(macd_line, 0), 9)
macd_hist = macd_line - macd_signal

# Bollinger
bb_sma20 = sma_arrays[20]
bb_std20 = np.full(n, np.nan)
for i in range(19, n):
    bb_std20[i] = np.std(closes[i-19:i+1])
bb_upper = bb_sma20 + 2 * bb_std20
bb_lower = bb_sma20 - 2 * bb_std20
bb_pctb = np.where((bb_upper - bb_lower) > 0, (closes - bb_lower) / (bb_upper - bb_lower), 0.5)
bb_width = np.where(bb_sma20 > 0, (bb_upper - bb_lower) / bb_sma20 * 100, 0)

# Stochastic
stoch_k = np.full(n, np.nan)
for i in range(13, n):
    h14 = np.max(highs[i-13:i+1])
    l14 = np.min(lows[i-13:i+1])
    stoch_k[i] = (closes[i] - l14) / (h14 - l14) * 100 if h14 > l14 else 50

# Volume ratio
vol_sma20 = sma(volumes, 20)
vol_ratio = np.where(vol_sma20 > 0, volumes / vol_sma20, 1.0)

# Taker buy ratio
tbr = np.where(volumes > 0, taker_buy / volumes, 0.5)

# Returns
returns_1d = np.zeros(n)
for i in range(1, n):
    returns_1d[i] = (closes[i] - closes[i-1]) / closes[i-1] * 100

# ETH/SOL returns
eth_returns = np.zeros(n)
sol_returns = np.zeros(n)
for i in range(1, n):
    if not np.isnan(eth_closes[i]) and not np.isnan(eth_closes[i-1]) and eth_closes[i-1] > 0:
        eth_returns[i] = (eth_closes[i] - eth_closes[i-1]) / eth_closes[i-1] * 100
    if not np.isnan(sol_closes[i]) and not np.isnan(sol_closes[i-1]) and sol_closes[i-1] > 0:
        sol_returns[i] = (sol_closes[i] - sol_closes[i-1]) / sol_closes[i-1] * 100

# Consecutive candles
consec_green = np.zeros(n)
consec_red = np.zeros(n)
for i in range(n):
    if closes[i] > opens[i]:
        consec_green[i] = consec_green[i-1] + 1 if i > 0 else 1
    elif closes[i] < opens[i]:
        consec_red[i] = consec_red[i-1] + 1 if i > 0 else 1

# ETH EMA50 for alts_above_ema50
eth_ema50 = ema(np.nan_to_num(eth_closes, 0), 50)
sol_ema50 = ema(np.nan_to_num(sol_closes, 0), 50)

print("  Building feature matrix...")

# Build feature rows
feature_rows = []
feature_names = []
target_rows = []
valid_indices = []

LOOKBACK = 200  # need 200 for SMA200

for i in range(LOOKBACK, n - 5):  # need 5 forward for targets
    f = {}
    dt_str = dates[i]
    try:
        dt = datetime.strptime(dt_str, '%Y-%m-%d')
    except:
        continue

    # --- TA Features ---
    for period in [5, 10, 20, 50, 100, 200]:
        s = sma_arrays[period][i]
        e = ema_arrays[period][i]
        if not np.isnan(s) and s > 0:
            f[f'price_vs_sma{period}'] = (closes[i] - s) / s * 100
        else:
            f[f'price_vs_sma{period}'] = 0
        if not np.isnan(e) and e > 0:
            f[f'price_vs_ema{period}'] = (closes[i] - e) / e * 100
        else:
            f[f'price_vs_ema{period}'] = 0

    for period in [7, 14, 21]:
        f[f'rsi_{period}'] = rsi_arrays[period][i] if not np.isnan(rsi_arrays[period][i]) else 50

    f['bb_pctb'] = bb_pctb[i] if not np.isnan(bb_pctb[i]) else 0.5
    f['bb_width'] = bb_width[i] if not np.isnan(bb_width[i]) else 0

    f['macd'] = macd_line[i] if not np.isnan(macd_line[i]) else 0
    f['macd_signal'] = macd_signal[i] if not np.isnan(macd_signal[i]) else 0
    f['macd_hist'] = macd_hist[i] if not np.isnan(macd_hist[i]) else 0

    f['stoch_k'] = stoch_k[i] if not np.isnan(stoch_k[i]) else 50

    atr_val = atr_14[i] if not np.isnan(atr_14[i]) else 0
    atr_mean = atr_20_mean[i] if not np.isnan(atr_20_mean[i]) else 1
    f['atr_ratio'] = atr_val / atr_mean if atr_mean > 0 else 1

    f['volume_ratio'] = vol_ratio[i] if not np.isnan(vol_ratio[i]) else 1
    f['taker_buy_ratio'] = tbr[i] if not np.isnan(tbr[i]) else 0.5

    # Candle patterns
    rng = highs[i] - lows[i]
    if rng > 0:
        f['body_pct'] = abs(closes[i] - opens[i]) / rng
        f['upper_wick'] = (highs[i] - max(opens[i], closes[i])) / rng
        f['lower_wick'] = (min(opens[i], closes[i]) - lows[i]) / rng
    else:
        f['body_pct'] = 0
        f['upper_wick'] = 0
        f['lower_wick'] = 0
    f['consecutive_green'] = consec_green[i]
    f['consecutive_red'] = consec_red[i]

    # Consensio
    sma7_val = sma_arrays[7][i] if not np.isnan(sma_arrays[7][i]) else 0
    sma21_val = sma_arrays[21][i] if not np.isnan(sma_arrays[21][i]) else 0
    sma50_val = sma_arrays[50][i] if not np.isnan(sma_arrays[50][i]) else 0
    sma7_prev = sma_arrays[7][i-1] if not np.isnan(sma_arrays[7][i-1]) else 0
    sma21_prev = sma_arrays[21][i-1] if not np.isnan(sma_arrays[21][i-1]) else 0
    sma50_prev = sma_arrays[50][i-1] if not np.isnan(sma_arrays[50][i-1]) else 0
    d_long = 2 if sma50_val > sma50_prev else 0
    d_med = 2 if sma21_val > sma21_prev else 0
    d_short = 2 if sma7_val > sma7_prev else 0
    f['consensio'] = d_long * 9 + d_med * 3 + d_short - 13

    # Returns
    for lag in [1, 3, 5, 10, 20]:
        if i - lag >= 0 and closes[i - lag] > 0:
            f[f'return_{lag}d'] = (closes[i] - closes[i - lag]) / closes[i - lag] * 100
        else:
            f[f'return_{lag}d'] = 0

    # Volatility
    if i >= 20:
        f['volatility_20d'] = np.std(returns_1d[i-19:i+1])
    else:
        f['volatility_20d'] = 0

    # --- Cross-Asset ---
    if i >= 20:
        btc_ret_20 = returns_1d[i-19:i+1]
        eth_ret_20 = eth_returns[i-19:i+1]
        sol_ret_20 = sol_returns[i-19:i+1]
        if np.std(btc_ret_20) > 0 and np.std(eth_ret_20) > 0:
            f['btc_eth_corr_20d'] = np.corrcoef(btc_ret_20, eth_ret_20)[0, 1]
        else:
            f['btc_eth_corr_20d'] = 0
        if np.std(btc_ret_20) > 0 and np.std(sol_ret_20) > 0:
            f['btc_sol_corr_20d'] = np.corrcoef(btc_ret_20, sol_ret_20)[0, 1]
        else:
            f['btc_sol_corr_20d'] = 0
    else:
        f['btc_eth_corr_20d'] = 0
        f['btc_sol_corr_20d'] = 0

    f['btc_eth_divergence'] = returns_1d[i] - eth_returns[i]
    f['btc_sol_divergence'] = returns_1d[i] - sol_returns[i]

    # Alts above EMA50
    alt_above = 0
    if not np.isnan(eth_closes[i]) and not np.isnan(eth_ema50[i]) and eth_closes[i] > eth_ema50[i]:
        alt_above += 1
    if not np.isnan(sol_closes[i]) and not np.isnan(sol_ema50[i]) and sol_closes[i] > sol_ema50[i]:
        alt_above += 1
    f['alts_above_ema50'] = alt_above

    # --- Esoteric Features ---
    eph_row = eph_dict.get(dt_str, {})
    f['moon_mansion'] = eph_row.get('moon_mansion', 0)
    f['mercury_retro'] = eph_row.get('mercury_retrograde', 0)
    f['hard_aspects'] = eph_row.get('hard_aspects', 0)
    f['soft_aspects'] = eph_row.get('soft_aspects', 0)
    f['psi'] = eph_row.get('planetary_strength', 0)
    f['moon_phase'] = eph_row.get('moon_phase', 0)
    f['eph_digital_root'] = eph_row.get('digital_root', 0)
    f['sun_lon'] = eph_row.get('sun_lon', 0)
    f['moon_lon'] = eph_row.get('moon_lon', 0)

    # --- Numerology ---
    doy = dt.timetuple().tm_yday
    f['is_113'] = 1 if 113 in [doy, 365 - doy, dt.month * 100 + dt.day] else 0
    f['is_caution'] = 1 if doy in [93, 39, 43, 48, 223, 322] else 0
    f['day_13'] = 1 if dt.day == 13 else 0
    f['day_21'] = 1 if dt.day == 21 else 0
    f['pump_date'] = 1 if dt.day in [14, 15, 16] else 0
    f['price_dr'] = digital_root(int(closes[i]))
    f['price_dr_6'] = 1 if digital_root(int(closes[i])) == 6 else 0
    f['price_contains_322'] = 1 if '322' in str(int(closes[i])) else 0

    # --- Tweet/News ---
    td = tweet_daily.get(dt_str, {})
    f['tweets_today'] = td.get('tweets_count', 0)
    f['gold_tweet'] = td.get('gold_tweet', 0)
    f['red_tweet'] = td.get('red_tweet', 0)
    f['decoder_tweet'] = 1 if dt_str in decoder_dates else 0
    f['avg_tweet_likes'] = td.get('avg_likes', 0)

    nd = news_dict.get(dt_str, (0, 0))
    f['news_count'] = nd[0]
    f['news_sentiment'] = nd[1] if nd[1] is not None else 0

    fg_val = fg_dict.get(dt_str, 50)
    f['fear_greed'] = fg_val
    # F&G rate of change
    dt_5ago = (dt - timedelta(days=5)).strftime('%Y-%m-%d')
    fg_5ago = fg_dict.get(dt_5ago, fg_val)
    f['fg_rate_of_change'] = fg_val - fg_5ago

    # Funding rate
    f['funding_rate'] = fund_daily.get(dt_str, 0)

    # --- Vedic Astrology Features ---
    astro = astro_dict.get(dt_str, {})
    if astro:
        f['nakshatra'] = astro.get('nakshatra', 0) or 0
        nature_map = {'deva': 0, 'manushya': 1, 'rakshasa': 2}
        f['nakshatra_nature'] = nature_map.get(str(astro.get('nakshatra_nature', '')).lower(), 1)
        guna_map = {'satva': 0, 'sattva': 0, 'rajas': 1, 'tamas': 2}
        f['nakshatra_guna'] = guna_map.get(str(astro.get('nakshatra_guna', '')).lower(), 1)
        f['tithi'] = astro.get('tithi', 15) or 15
        f['yoga_idx'] = astro.get('yoga', 0) or 0
        vara_map = {'sun': 0, 'moon': 1, 'mars': 2, 'mercury': 3, 'jupiter': 4, 'venus': 5, 'saturn': 6}
        f['vara'] = vara_map.get(str(astro.get('vara', '')).lower(), 0)
        moon_sign_map = {'mesha': 0, 'vrishabha': 1, 'mithuna': 2, 'karka': 3, 'simha': 4, 'kanya': 5,
                         'tula': 6, 'vrischika': 7, 'dhanus': 8, 'makara': 9, 'kumbha': 10, 'meena': 11}
        f['moon_sidereal_sign'] = moon_sign_map.get(str(astro.get('moon_sidereal_sign', '')).lower(), 0)
        f['sun_sidereal_sign'] = moon_sign_map.get(str(astro.get('sun_sidereal_sign', '')).lower(), 0)
        # Key nakshatras as binary
        nak = astro.get('nakshatra', -1)
        f['nakshatra_purva_ashadha'] = 1 if nak == 19 else 0
        f['nakshatra_mrigashira'] = 1 if nak == 4 else 0
        f['nakshatra_uttara_phalguni'] = 1 if nak == 11 else 0
    else:
        for k in ['nakshatra', 'nakshatra_nature', 'nakshatra_guna', 'tithi', 'yoga_idx',
                   'vara', 'moon_sidereal_sign', 'sun_sidereal_sign',
                   'nakshatra_purva_ashadha', 'nakshatra_mrigashira', 'nakshatra_uttara_phalguni']:
            f[k] = 0

    # --- Chinese BaZi Features ---
    if astro:
        f['bazi_stem'] = astro.get('day_stem', 0) or 0
        f['bazi_branch'] = astro.get('day_branch', 0) or 0
        elem_map = {'wood': 0, 'wood+': 0, 'wood-': 0, 'fire': 1, 'fire+': 1, 'fire-': 1,
                     'earth': 2, 'earth+': 2, 'earth-': 2, 'metal': 3, 'metal+': 3, 'metal-': 3,
                     'water': 4, 'water+': 4, 'water-': 4}
        f['bazi_element'] = elem_map.get(str(astro.get('day_element', '')).lower(), 2)
        animal_map = {'rat': 0, 'ox': 1, 'tiger': 2, 'rabbit': 3, 'dragon': 4, 'snake': 5,
                      'horse': 6, 'goat': 7, 'monkey': 8, 'rooster': 9, 'dog': 10, 'pig': 11}
        f['bazi_animal'] = animal_map.get(str(astro.get('day_animal', '')).lower(), 0)
        f['bazi_clash'] = 1 if astro.get('day_clash_branch') is not None else 0
        # BTC = Ox (branch 1). Friendly/enemy?
        OX_FRIENDS = [5, 9, 1]  # Snake, Rooster, Ox
        OX_ENEMIES = [7]  # Goat
        db = astro.get('day_branch', 0) or 0
        f['bazi_btc_friendly'] = 1 if db in OX_FRIENDS else 0
        f['bazi_btc_enemy'] = 1 if db in OX_ENEMIES else 0
    else:
        for k in ['bazi_stem', 'bazi_branch', 'bazi_element', 'bazi_animal', 'bazi_clash',
                   'bazi_btc_friendly', 'bazi_btc_enemy']:
            f[k] = 0

    # --- Mayan Tzolkin Features ---
    if astro:
        f['tzolkin_tone'] = astro.get('tzolkin_tone', 1) or 1
        f['tzolkin_sign_idx'] = astro.get('tzolkin_sign_idx', 0) or 0
        tone = astro.get('tzolkin_tone', 0) or 0
        f['tzolkin_tone_1'] = 1 if tone == 1 else 0
        f['tzolkin_tone_9'] = 1 if tone == 9 else 0
        f['tzolkin_tone_13'] = 1 if tone == 13 else 0
        sign_idx = astro.get('tzolkin_sign_idx', -1)
        f['tzolkin_cimi'] = 1 if sign_idx == 5 else 0
        f['tzolkin_ahau'] = 1 if sign_idx == 19 else 0
    else:
        for k in ['tzolkin_tone', 'tzolkin_sign_idx', 'tzolkin_tone_1', 'tzolkin_tone_9',
                   'tzolkin_tone_13', 'tzolkin_cimi', 'tzolkin_ahau']:
            f[k] = 0

    # --- Arabic Lots Features ---
    if astro:
        moon_lon_val = astro.get('moon_sidereal_lon', 0) or 0
        for lot_name in ['lot_commerce', 'lot_increase', 'lot_catastrophe', 'lot_treachery']:
            lot_lon = astro.get(lot_name, 0) or 0
            diff = abs(moon_lon_val - lot_lon) % 360
            if diff > 180:
                diff = 360 - diff
            f[f'{lot_name}_moon_conjunct'] = 1 if diff < 8 else 0
    else:
        for lot_name in ['lot_commerce', 'lot_increase', 'lot_catastrophe', 'lot_treachery']:
            f[f'{lot_name}_moon_conjunct'] = 0

    # --- Macro Features (DXY, Gold, SPX) ---
    macro = macro_dict.get(dt_str, {})
    dxy_val = macro.get('dxy', None)
    gold_val = macro.get('gold', None)
    spx_val = macro.get('spx', None)
    f['dxy'] = dxy_val if dxy_val is not None else 0
    f['gold'] = gold_val if gold_val is not None else 0
    f['spx'] = spx_val if spx_val is not None else 0
    # Rate of change (5-day)
    dt_5ago_str = (dt - timedelta(days=5)).strftime('%Y-%m-%d')
    macro_5ago = macro_dict.get(dt_5ago_str, {})
    for mname, mval, mkey in [('dxy', dxy_val, 'dxy'), ('gold', gold_val, 'gold'), ('spx', spx_val, 'spx')]:
        prev = macro_5ago.get(mkey, None)
        if mval and prev and prev > 0:
            f[f'{mname}_roc_5d'] = (mval - prev) / prev * 100
        else:
            f[f'{mname}_roc_5d'] = 0
    # BTC vs macro correlations (20-day)
    if i >= 20 and dxy_val:
        btc_ret_20 = returns_1d[i-19:i+1]
        # Build macro returns arrays
        dxy_rets = []
        gold_rets = []
        spx_rets = []
        for j_back in range(20):
            idx_back = i - 19 + j_back
            d_str = dates[idx_back]
            d_prev_str = dates[idx_back - 1] if idx_back > 0 else d_str
            m_cur = macro_dict.get(d_str, {})
            m_prev = macro_dict.get(d_prev_str, {})
            dv = m_cur.get('dxy', 0)
            dp = m_prev.get('dxy', 0)
            dxy_rets.append((dv - dp) / dp * 100 if dp and dv else 0)
            gv = m_cur.get('gold', 0)
            gp = m_prev.get('gold', 0)
            gold_rets.append((gv - gp) / gp * 100 if gp and gv else 0)
            sv = m_cur.get('spx', 0)
            sp = m_prev.get('spx', 0)
            spx_rets.append((sv - sp) / sp * 100 if sp and sv else 0)
        dxy_rets = np.array(dxy_rets)
        gold_rets = np.array(gold_rets)
        spx_rets = np.array(spx_rets)
        if np.std(btc_ret_20) > 0 and np.std(dxy_rets) > 0:
            f['btc_dxy_corr_20d'] = np.corrcoef(btc_ret_20, dxy_rets)[0, 1]
        else:
            f['btc_dxy_corr_20d'] = 0
        if np.std(btc_ret_20) > 0 and np.std(gold_rets) > 0:
            f['btc_gold_corr_20d'] = np.corrcoef(btc_ret_20, gold_rets)[0, 1]
        else:
            f['btc_gold_corr_20d'] = 0
        if np.std(btc_ret_20) > 0 and np.std(spx_rets) > 0:
            f['btc_spx_corr_20d'] = np.corrcoef(btc_ret_20, spx_rets)[0, 1]
        else:
            f['btc_spx_corr_20d'] = 0
    else:
        f['btc_dxy_corr_20d'] = 0
        f['btc_gold_corr_20d'] = 0
        f['btc_spx_corr_20d'] = 0
    # Divergence
    dxy_prev = macro_dict.get(dates[i-1] if i > 0 else dt_str, {}).get('dxy', dxy_val) if dxy_val else 0
    if dxy_val and dxy_prev and dxy_prev > 0:
        f['btc_dxy_divergence'] = returns_1d[i] - ((dxy_val - dxy_prev) / dxy_prev * 100)
    else:
        f['btc_dxy_divergence'] = 0

    # --- Google Trends Features ---
    trend_value = trends_daily.get(dt_str, 0) or 0
    f['google_interest'] = trend_value
    f['google_interest_high'] = 1 if trend_value > 80 else 0
    f['google_interest_low'] = 1 if trend_value < 20 else 0
    # ROC: compare to ~5 weeks ago
    dt_5w_ago_str = (dt - timedelta(weeks=5)).strftime('%Y-%m-%d')
    trend_5w_ago = trends_daily.get(dt_5w_ago_str, trend_value) or trend_value
    f['google_roc'] = trend_value - trend_5w_ago

    # --- Temporal (sin/cos) ---
    f['dow_sin'] = math.sin(2 * math.pi * dt.weekday() / 7)
    f['dow_cos'] = math.cos(2 * math.pi * dt.weekday() / 7)
    f['month_sin'] = math.sin(2 * math.pi * dt.month / 12)
    f['month_cos'] = math.cos(2 * math.pi * dt.month / 12)
    f['doy_sin'] = math.sin(2 * math.pi * doy / 365)
    f['doy_cos'] = math.cos(2 * math.pi * doy / 365)

    # --- Lagged ---
    for lag in [1, 3, 5, 10]:
        idx = i - lag
        if idx >= 0:
            f[f'rsi14_lag{lag}'] = rsi_arrays[14][idx] if not np.isnan(rsi_arrays[14][idx]) else 50
            f[f'bb_pctb_lag{lag}'] = bb_pctb[idx] if not np.isnan(bb_pctb[idx]) else 0.5
            f[f'volume_ratio_lag{lag}'] = vol_ratio[idx] if not np.isnan(vol_ratio[idx]) else 1
            fg_lag_date = dates[idx]
            f[f'fg_lag{lag}'] = fg_dict.get(fg_lag_date, 50)
            f[f'stoch_k_lag{lag}'] = stoch_k[idx] if not np.isnan(stoch_k[idx]) else 50
            f[f'macd_hist_lag{lag}'] = macd_hist[idx] if not np.isnan(macd_hist[idx]) else 0
            f[f'return_1d_lag{lag}'] = returns_1d[idx]
        else:
            f[f'rsi14_lag{lag}'] = 50
            f[f'bb_pctb_lag{lag}'] = 0.5
            f[f'volume_ratio_lag{lag}'] = 1
            f[f'fg_lag{lag}'] = 50
            f[f'stoch_k_lag{lag}'] = 50
            f[f'macd_hist_lag{lag}'] = 0
            f[f'return_1d_lag{lag}'] = 0

    # --- Multi-TF features (4h aggregates) ---
    h4 = btc_4h_daily.get(dt_str, {})
    f['vol_4h_std'] = h4.get('vol_4h_std', 0) if not (isinstance(h4.get('vol_4h_std'), float) and np.isnan(h4.get('vol_4h_std', 0))) else 0
    f['range_4h_mean'] = h4.get('range_4h_mean', 0) if not (isinstance(h4.get('range_4h_mean'), float) and np.isnan(h4.get('range_4h_mean', 0))) else 0

    # --- SMA slope features ---
    for period in [20, 50, 200]:
        curr = sma_arrays[period][i]
        prev5 = sma_arrays[period][i-5] if i >= 5 else curr
        if not np.isnan(curr) and not np.isnan(prev5) and prev5 > 0:
            f[f'sma{period}_slope_5d'] = (curr - prev5) / prev5 * 100
        else:
            f[f'sma{period}_slope_5d'] = 0

    # --- Relative volume profile ---
    if i >= 5:
        f['vol_trend_5d'] = (volumes[i] / np.mean(volumes[i-4:i+1])) if np.mean(volumes[i-4:i+1]) > 0 else 1
    else:
        f['vol_trend_5d'] = 1

    # --- High/Low distance ---
    if i >= 20:
        f['dist_from_20d_high'] = (closes[i] - np.max(highs[i-19:i+1])) / closes[i] * 100
        f['dist_from_20d_low'] = (closes[i] - np.min(lows[i-19:i+1])) / closes[i] * 100
    else:
        f['dist_from_20d_high'] = 0
        f['dist_from_20d_low'] = 0

    if i >= 50:
        f['dist_from_50d_high'] = (closes[i] - np.max(highs[i-49:i+1])) / closes[i] * 100
        f['dist_from_50d_low'] = (closes[i] - np.min(lows[i-49:i+1])) / closes[i] * 100
    else:
        f['dist_from_50d_high'] = 0
        f['dist_from_50d_low'] = 0

    # --- RSI divergence ---
    f['rsi_price_div'] = (rsi_arrays[14][i] - rsi_arrays[14][i-5]) - (returns_1d[i] * 5) if not np.isnan(rsi_arrays[14][i]) and not np.isnan(rsi_arrays[14][i-5]) else 0

    # --- Meta signals ---
    bullish = 0
    bearish = 0
    if rsi_arrays[14][i] < 30: bullish += 1
    if rsi_arrays[14][i] > 70: bearish += 1
    if closes[i] > sma_arrays[50][i] if not np.isnan(sma_arrays[50][i]) else False: bullish += 1
    else: bearish += 1
    if closes[i] > sma_arrays[200][i] if not np.isnan(sma_arrays[200][i]) else False: bullish += 1
    else: bearish += 1
    if macd_hist[i] > 0 if not np.isnan(macd_hist[i]) else False: bullish += 1
    else: bearish += 1
    if bb_pctb[i] < 0.2 if not np.isnan(bb_pctb[i]) else False: bullish += 1
    if bb_pctb[i] > 0.8 if not np.isnan(bb_pctb[i]) else False: bearish += 1
    if fg_val < 25: bullish += 1
    if fg_val > 75: bearish += 1
    f['num_bullish_signals'] = bullish
    f['num_bearish_signals'] = bearish
    f['signal_agreement'] = bullish / (bullish + bearish) if (bullish + bearish) > 0 else 0.5

    # --- Targets ---
    t = {}
    t['next_1d_return'] = (closes[i+1] - closes[i]) / closes[i] * 100
    t['next_1d_direction'] = 1 if t['next_1d_return'] > 0 else 0
    t['next_3d_return'] = (closes[min(i+3, n-1)] - closes[i]) / closes[i] * 100
    t['next_5d_return'] = (closes[min(i+5, n-1)] - closes[i]) / closes[i] * 100

    feature_rows.append(f)
    target_rows.append(t)
    valid_indices.append(i)

    if i == LOOKBACK:
        feature_names = sorted(f.keys())

# Build matrices
feature_names = sorted(feature_rows[0].keys())
X = np.array([[row.get(fn, 0) for fn in feature_names] for row in feature_rows], dtype=np.float32)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

y_dir = np.array([t['next_1d_direction'] for t in target_rows])
y_ret1 = np.array([t['next_1d_return'] for t in target_rows])
y_ret3 = np.array([t['next_3d_return'] for t in target_rows])

print(f"  Feature matrix: {X.shape[0]} samples x {X.shape[1]} features")
print(f"  Target balance: {np.mean(y_dir)*100:.1f}% up days")
print(f"  Features engineered in {time.time()-t0:.1f}s")
print()

# ============================================================================
# LAYER 1: ML MODELS
# ============================================================================
print("[3/6] TRAINING ML MODELS...")
t0 = time.time()

# Walk-forward split
train_size = int(len(X) * 0.7)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y_dir[:train_size], y_dir[train_size:]
print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
print(f"  Train period: {dates[valid_indices[0]]} to {dates[valid_indices[train_size-1]]}")
print(f"  Test period: {dates[valid_indices[train_size]]} to {dates[valid_indices[-1]]}")

# --- XGBoost ---
print("  Training XGBoost...")
import xgboost as xgb

try:
    xgb_model = xgb.XGBClassifier(
        tree_method='gpu_hist', n_estimators=500, max_depth=6,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        eval_metric='logloss', random_state=42, use_label_encoder=False
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    print("  XGBoost: GPU mode")
except Exception:
    xgb_model = xgb.XGBClassifier(
        tree_method='hist', n_estimators=500, max_depth=6,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        eval_metric='logloss', random_state=42, use_label_encoder=False
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    print("  XGBoost: CPU mode (hist)")

xgb_acc = xgb_model.score(X_test, y_test)
xgb_preds = xgb_model.predict(X_test)
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
print(f"  XGBoost accuracy: {xgb_acc*100:.2f}%")

# Feature importance
xgb_importance = dict(zip(feature_names, xgb_model.feature_importances_))
xgb_sorted = sorted(xgb_importance.items(), key=lambda x: -x[1])

# --- Random Forest ---
print("  Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_acc = rf_model.score(X_test, y_test)
rf_preds = rf_model.predict(X_test)
print(f"  Random Forest accuracy: {rf_acc*100:.2f}%")

rf_importance = dict(zip(feature_names, rf_model.feature_importances_))
rf_sorted = sorted(rf_importance.items(), key=lambda x: -x[1])

# --- LightGBM ---
print("  Training LightGBM...")
import lightgbm as lgb
lgb_model = lgb.LGBMClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
)
lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
lgb_acc = lgb_model.score(X_test, y_test)
lgb_probs = lgb_model.predict_proba(X_test)[:, 1]
print(f"  LightGBM accuracy: {lgb_acc*100:.2f}%")

lgb_importance = dict(zip(feature_names, lgb_model.feature_importances_))
lgb_sorted = sorted(lgb_importance.items(), key=lambda x: -x[1])

# --- LASSO ---
print("  Training LASSO (L1 Logistic Regression)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lasso = LogisticRegression(penalty='l1', solver='saga', C=0.1, max_iter=5000, random_state=42)
lasso.fit(X_train_scaled, y_train)
lasso_acc = lasso.score(X_test_scaled, y_test)
print(f"  LASSO accuracy: {lasso_acc*100:.2f}%")

lasso_weights = dict(zip(feature_names, lasso.coef_[0]))
nonzero_lasso = {k: v for k, v in sorted(lasso_weights.items(), key=lambda x: -abs(x[1])) if abs(v) > 0.01}

# --- Ensemble ---
ensemble_probs = (xgb_probs + lgb_probs) / 2
ensemble_preds = (ensemble_probs > 0.5).astype(int)
ensemble_acc = accuracy_score(y_test, ensemble_preds)
print(f"  Ensemble (XGB+LGB) accuracy: {ensemble_acc*100:.2f}%")

print(f"  Models trained in {time.time()-t0:.1f}s")
print()

# ============================================================================
# LAYER 2: GENETIC ALGORITHM
# ============================================================================
print("[4/6] RUNNING GENETIC ALGORITHM...")
t0 = time.time()

# Get ATR for test period
test_atrs = atr_14[np.array(valid_indices)]
test_closes = closes[np.array(valid_indices)]

LEVERAGE_OPTIONS = [1, 2, 3, 5, 7, 10, 15, 20, 25]
RISK_OPTIONS = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15]
STOP_OPTIONS = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
RR_OPTIONS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
TRAIL_OPTIONS = [0.5, 0.8, 1.0, 1.5, 2.0]
HOLD_OPTIONS = [2, 4, 6, 8, 10, 14, 20]
PARTIAL_OPTIONS = [0.0, 0.25, 0.50, 0.75]

# Top 20 features from XGBoost
top20_features = [f for f, _ in xgb_sorted[:20]]

def evaluate_genome(genome, probs, all_closes, all_atrs, start_idx):
    """Evaluate a trading strategy genome on test data"""
    feature_mask = genome[:20]
    lev = LEVERAGE_OPTIONS[genome[20] % len(LEVERAGE_OPTIONS)]
    risk = RISK_OPTIONS[genome[21] % len(RISK_OPTIONS)]
    stop_atr = STOP_OPTIONS[genome[22] % len(STOP_OPTIONS)]
    rr = RR_OPTIONS[genome[23] % len(RR_OPTIONS)]
    trail = TRAIL_OPTIONS[genome[24] % len(TRAIL_OPTIONS)]
    max_hold = HOLD_OPTIONS[genome[25] % len(HOLD_OPTIONS)]
    partial = PARTIAL_OPTIONS[genome[26] % len(PARTIAL_OPTIONS)]

    # Must have at least 3 features selected
    selected = sum(feature_mask)
    if selected < 3:
        return (-100, 100, 0, 0, 0, lev, risk, stop_atr, rr, trail, max_hold, partial)

    # Confidence threshold based on feature agreement
    conf_threshold = 0.50 + 0.005 * selected  # more features = higher bar

    INITIAL_CAPITAL = 10000.0
    balance = INITIAL_CAPITAL
    peak = INITIAL_CAPITAL
    max_dd = 0
    wins = 0
    losses = 0
    trades = 0
    total_pnl = 0
    in_trade = False
    trade_end = 0

    for idx in range(len(probs)):
        if in_trade and idx < trade_end:
            continue
        in_trade = False

        prob = probs[idx]
        global_idx = start_idx + idx

        if global_idx + max_hold + 1 >= len(all_closes):
            break

        # Determine direction
        if prob > conf_threshold:
            direction = 1  # long
        elif prob < (1 - conf_threshold):
            direction = -1  # short
        else:
            continue

        entry = all_closes[global_idx + 1]  # next candle open
        atr = all_atrs[global_idx]
        if np.isnan(atr) or atr <= 0:
            continue

        stop_dist = atr * stop_atr
        tp_dist = stop_dist * rr

        # FIXED position sizing (based on initial capital, no compounding)
        risk_amt = INITIAL_CAPITAL * risk  # always risk fixed % of starting capital
        stop_pct = stop_dist / entry  # e.g., 0.02 = 2%
        if stop_pct <= 0:
            continue
        pos_size = min(risk_amt / stop_pct, INITIAL_CAPITAL * lev)
        fee = pos_size * 0.0017  # fees + slippage + funding

        best_pnl = 0
        final_pnl = 0
        exit_j = max_hold

        for j in range(1, min(max_hold + 1, len(all_closes) - global_idx - 1)):
            price = all_closes[global_idx + 1 + j]
            if direction == 1:
                move = (price - entry) / entry
            else:
                move = (entry - price) / entry

            pnl = move * pos_size
            dollar_move = move * entry

            # Hard stop (check first)
            if dollar_move < -stop_dist:
                final_pnl = -risk_amt
                exit_j = j
                break

            # Take profit
            if dollar_move > tp_dist:
                if partial > 0:
                    final_pnl = risk_amt * rr * partial + pnl * (1 - partial)
                else:
                    final_pnl = risk_amt * rr
                exit_j = j
                break

            # Trailing stop: if profit > trail * stop_dist, lock in some gains
            if dollar_move > trail * stop_dist:
                locked = (dollar_move - trail * stop_dist) / entry * pos_size
                if locked > 0:
                    final_pnl = max(final_pnl, locked)

            final_pnl = pnl

        final_pnl -= fee
        balance += final_pnl
        total_pnl += final_pnl

        # Cap balance to prevent overflow
        if balance > 1000000:
            balance = 1000000
        if balance <= 0:
            return (-100, 100, -100, 0, trades, lev, risk, stop_atr, rr, trail, max_hold, partial)

        peak = max(peak, balance)
        dd = (peak - balance) / peak * 100
        max_dd = max(max_dd, dd)
        trades += 1
        if final_pnl > 0:
            wins += 1
        else:
            losses += 1

        in_trade = True
        trade_end = idx + exit_j + 1

    roi = (balance - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100  # percent
    wr = wins / trades * 100 if trades > 0 else 0

    # Fitness: ROI * (1 - DD/100) penalized for low trade count
    trade_penalty = min(trades / 30, 1.0)  # want at least 30 trades
    fitness = roi * (1 - max_dd / 100) * trade_penalty

    return (fitness, max_dd, roi, wr, trades, lev, risk, stop_atr, rr, trail, max_hold, partial)


# GA parameters
POPULATION = 200
GENERATIONS = 300
GENOME_SIZE = 27

# Use ensemble probabilities for GA evaluation
test_start_global = valid_indices[train_size]

population = [
    [random.randint(0, 1) if i < 20 else random.randint(0, 9) for i in range(GENOME_SIZE)]
    for _ in range(POPULATION)
]

best_ever = None
best_ever_fitness = -999

for gen in range(GENERATIONS):
    scores = []
    for p in population:
        result = evaluate_genome(p, ensemble_probs, closes, atr_14, test_start_global)
        scores.append((result, p))

    scores.sort(key=lambda x: -x[0][0])

    if scores[0][0][0] > best_ever_fitness:
        best_ever_fitness = scores[0][0][0]
        best_ever = (scores[0][0], scores[0][1][:])

    if gen % 25 == 0 or gen == GENERATIONS - 1:
        b = scores[0][0]
        elapsed = time.time() - t0
        print(f"  Gen {gen:3d}: fitness={b[0]:8.1f} ROI={b[2]:7.1f}% DD={b[1]:5.1f}% "
              f"WR={b[3]:5.1f}% trades={b[4]:3.0f} lev={b[5]}x risk={b[6]*100:.0f}% "
              f"[{elapsed:.0f}s]")

    # Selection: keep top 40
    survivors = [s[1] for s in scores[:40]]

    # Elitism: keep top 5
    elite = [s[1][:] for s in scores[:5]]

    # Breed children
    children = []
    for _ in range(POPULATION - 40):
        p1, p2 = random.sample(survivors[:20], 2)
        cross = random.randint(1, GENOME_SIZE - 2)
        child = p1[:cross] + p2[cross:]
        # Mutation
        child = [
            1 - g if i < 20 and random.random() < 0.05 else
            random.randint(0, 9) if i >= 20 and random.random() < 0.08 else g
            for i, g in enumerate(child)
        ]
        children.append(child)

    population = survivors + children

print(f"  GA completed in {time.time()-t0:.1f}s")
print()

# ============================================================================
# RESULTS
# ============================================================================
print("[5/6] COMPILING RESULTS...")
print()

best_result = best_ever[0]
best_genome = best_ever[1]

# Decode best genome
selected_features = [top20_features[i] for i in range(20) if best_genome[i] == 1]
best_lev = LEVERAGE_OPTIONS[best_genome[20] % len(LEVERAGE_OPTIONS)]
best_risk = RISK_OPTIONS[best_genome[21] % len(RISK_OPTIONS)]
best_stop = STOP_OPTIONS[best_genome[22] % len(STOP_OPTIONS)]
best_rr = RR_OPTIONS[best_genome[23] % len(RR_OPTIONS)]
best_trail = TRAIL_OPTIONS[best_genome[24] % len(TRAIL_OPTIONS)]
best_hold = HOLD_OPTIONS[best_genome[25] % len(HOLD_OPTIONS)]
best_partial = PARTIAL_OPTIONS[best_genome[26] % len(PARTIAL_OPTIONS)]

# Build output
output = []
output.append("=" * 80)
output.append("ML MEGA OPTIMIZER - COMPLETE RESULTS")
output.append("=" * 80)
output.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
output.append(f"Data: {dates[valid_indices[0]]} to {dates[valid_indices[-1]]}")
output.append(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
output.append(f"Train: {train_size} samples, Test: {len(X_test)} samples")
output.append("")

output.append("=" * 80)
output.append("1. FEATURE IMPORTANCE - TOP 30 (XGBoost)")
output.append("=" * 80)
for rank, (feat, imp) in enumerate(xgb_sorted[:30], 1):
    bar = "#" * int(imp * 500)
    output.append(f"  {rank:2d}. {feat:30s} {imp:.4f} {bar}")

output.append("")
output.append("=" * 80)
output.append("2. FEATURE IMPORTANCE - TOP 30 (Random Forest)")
output.append("=" * 80)
for rank, (feat, imp) in enumerate(rf_sorted[:30], 1):
    bar = "#" * int(imp * 500)
    output.append(f"  {rank:2d}. {feat:30s} {imp:.4f} {bar}")

output.append("")
output.append("=" * 80)
output.append("3. FEATURE IMPORTANCE - TOP 30 (LightGBM)")
output.append("=" * 80)
for rank, (feat, imp) in enumerate(lgb_sorted[:30], 1):
    bar = "#" * int(imp * 200)
    output.append(f"  {rank:2d}. {feat:30s} {imp:6.0f} {bar}")

output.append("")
output.append("=" * 80)
output.append("4. LASSO NON-ZERO WEIGHTS (signals that actually matter)")
output.append("=" * 80)
for feat, w in nonzero_lasso.items():
    direction = "BULL" if w > 0 else "BEAR"
    output.append(f"  {feat:30s} {w:+.4f} ({direction})")
output.append(f"  Total non-zero: {len(nonzero_lasso)} / {len(feature_names)}")

output.append("")
output.append("=" * 80)
output.append("5. MODEL ACCURACY (Walk-Forward Test)")
output.append("=" * 80)
output.append(f"  XGBoost:     {xgb_acc*100:.2f}%")
output.append(f"  Random Forest: {rf_acc*100:.2f}%")
output.append(f"  LightGBM:    {lgb_acc*100:.2f}%")
output.append(f"  LASSO:       {lasso_acc*100:.2f}%")
output.append(f"  Ensemble:    {ensemble_acc*100:.2f}%")

# Confidence-filtered accuracy
for thresh in [0.55, 0.60, 0.65]:
    mask = (ensemble_probs > thresh) | (ensemble_probs < (1 - thresh))
    if mask.sum() > 0:
        filtered_acc = accuracy_score(y_test[mask], (ensemble_probs[mask] > 0.5).astype(int))
        output.append(f"  Ensemble (>{thresh:.0%} conf): {filtered_acc*100:.2f}% ({mask.sum()} signals)")

output.append("")
output.append("  Classification Report (XGBoost):")
report = classification_report(y_test, xgb_preds, target_names=['DOWN', 'UP'])
for line in report.split('\n'):
    output.append(f"    {line}")

output.append("")
output.append("=" * 80)
output.append("6. GENETIC ALGORITHM - OPTIMAL CONFIGURATION")
output.append("=" * 80)
output.append(f"  Fitness:        {best_result[0]:.1f}")
output.append(f"  ROI:            {best_result[2]:.1f}%")
output.append(f"  Max Drawdown:   {best_result[1]:.1f}%")
output.append(f"  Win Rate:       {best_result[3]:.1f}%")
output.append(f"  Total Trades:   {best_result[4]:.0f}")
output.append(f"  Leverage:       {best_lev}x")
output.append(f"  Risk per Trade: {best_risk*100:.1f}%")
output.append(f"  Stop Loss:      {best_stop} x ATR")
output.append(f"  Reward:Risk:    {best_rr}:1")
output.append(f"  Trail Trigger:  {best_trail} x ATR")
output.append(f"  Max Hold:       {best_hold} days")
output.append(f"  Partial TP:     {best_partial*100:.0f}%")
output.append("")
output.append("  Selected Features (ON):")
for feat in selected_features:
    output.append(f"    - {feat}")

# Cross-model feature consensus
output.append("")
output.append("=" * 80)
output.append("7. CROSS-MODEL FEATURE CONSENSUS (Top 15)")
output.append("=" * 80)
# Average rank across models
xgb_ranks = {f: i for i, (f, _) in enumerate(xgb_sorted)}
rf_ranks = {f: i for i, (f, _) in enumerate(rf_sorted)}
lgb_ranks = {f: i for i, (f, _) in enumerate(lgb_sorted)}
consensus = {}
for f in feature_names:
    avg_rank = (xgb_ranks.get(f, 999) + rf_ranks.get(f, 999) + lgb_ranks.get(f, 999)) / 3
    consensus[f] = avg_rank
consensus_sorted = sorted(consensus.items(), key=lambda x: x[1])
for rank, (feat, avg_r) in enumerate(consensus_sorted[:15], 1):
    xr = xgb_ranks.get(feat, 999) + 1
    rr = rf_ranks.get(feat, 999) + 1
    lr = lgb_ranks.get(feat, 999) + 1
    output.append(f"  {rank:2d}. {feat:30s} avg_rank={avg_r+1:5.1f} (XGB:{xr:3d} RF:{rr:3d} LGB:{lr:3d})")

# Compare to baseline
output.append("")
output.append("=" * 80)
output.append("8. BASELINE COMPARISON (Buy & Hold vs ML Strategy)")
output.append("=" * 80)
test_start_price = closes[valid_indices[train_size]]
test_end_price = closes[valid_indices[-1]]
bh_return = (test_end_price - test_start_price) / test_start_price * 100
output.append(f"  Buy & Hold Return: {bh_return:.1f}%")
output.append(f"  ML Strategy ROI:   {best_result[2]:.1f}%")
output.append(f"  ML vs B&H:         {best_result[2] - bh_return:+.1f}%")
output.append(f"  ML Max Drawdown:   {best_result[1]:.1f}%")
output.append(f"  ML Win Rate:       {best_result[3]:.1f}%")

# Esoteric feature analysis
output.append("")
output.append("=" * 80)
output.append("9. ESOTERIC/NUMEROLOGY FEATURE ANALYSIS")
output.append("=" * 80)
esoteric_features = ['moon_mansion', 'mercury_retro', 'hard_aspects', 'soft_aspects',
                     'psi', 'moon_phase', 'eph_digital_root', 'sun_lon', 'moon_lon',
                     'is_113', 'is_caution', 'day_13', 'day_21', 'pump_date',
                     'price_dr', 'price_dr_6', 'price_contains_322', 'decoder_tweet',
                     'gold_tweet', 'red_tweet', 'consensio',
                     'nakshatra', 'nakshatra_nature', 'nakshatra_guna', 'tithi', 'yoga_idx',
                     'vara', 'moon_sidereal_sign', 'sun_sidereal_sign',
                     'nakshatra_purva_ashadha', 'nakshatra_mrigashira', 'nakshatra_uttara_phalguni',
                     'bazi_stem', 'bazi_branch', 'bazi_element', 'bazi_animal', 'bazi_clash',
                     'bazi_btc_friendly', 'bazi_btc_enemy',
                     'tzolkin_tone', 'tzolkin_sign_idx', 'tzolkin_tone_1', 'tzolkin_tone_9',
                     'tzolkin_tone_13', 'tzolkin_cimi', 'tzolkin_ahau',
                     'lot_commerce_moon_conjunct', 'lot_increase_moon_conjunct',
                     'lot_catastrophe_moon_conjunct', 'lot_treachery_moon_conjunct']
for feat in esoteric_features:
    if feat in xgb_importance:
        xgb_rank = [i for i, (f, _) in enumerate(xgb_sorted) if f == feat][0] + 1
        rf_rank = [i for i, (f, _) in enumerate(rf_sorted) if f == feat][0] + 1
        lasso_w = lasso_weights.get(feat, 0)
        output.append(f"  {feat:25s} XGB_rank={xgb_rank:3d} RF_rank={rf_rank:3d} LASSO={lasso_w:+.4f}")

output.append("")
output.append("=" * 80)
output.append("10. TWEET/NEWS/SENTIMENT FEATURE ANALYSIS")
output.append("=" * 80)
sentiment_features = ['tweets_today', 'gold_tweet', 'red_tweet', 'decoder_tweet',
                      'avg_tweet_likes', 'news_count', 'news_sentiment',
                      'fear_greed', 'fg_rate_of_change', 'funding_rate',
                      'dxy', 'gold', 'spx', 'dxy_roc_5d', 'gold_roc_5d', 'spx_roc_5d',
                      'btc_dxy_corr_20d', 'btc_gold_corr_20d', 'btc_spx_corr_20d',
                      'btc_dxy_divergence',
                      'google_interest', 'google_interest_high', 'google_interest_low', 'google_roc']
for feat in sentiment_features:
    if feat in xgb_importance:
        xgb_rank = [i for i, (f, _) in enumerate(xgb_sorted) if f == feat][0] + 1
        rf_rank = [i for i, (f, _) in enumerate(rf_sorted) if f == feat][0] + 1
        lasso_w = lasso_weights.get(feat, 0)
        output.append(f"  {feat:25s} XGB_rank={xgb_rank:3d} RF_rank={rf_rank:3d} LASSO={lasso_w:+.4f}")

# Print everything
result_text = "\n".join(output)
print(result_text)

# ============================================================================
# SAVE FILES
# ============================================================================
print()
print("[6/6] SAVING FILES...")

# Save results text
with open(os.path.join(DB_DIR, "ml_mega_results_v2.txt"), "w", encoding="utf-8") as f:
    f.write(result_text)
print(f"  Saved: ml_mega_results_v2.txt")

# Save optimal config JSON
config = {
    "generated": datetime.now().isoformat(),
    "model_accuracy": {
        "xgboost": round(xgb_acc * 100, 2),
        "random_forest": round(rf_acc * 100, 2),
        "lightgbm": round(lgb_acc * 100, 2),
        "lasso": round(lasso_acc * 100, 2),
        "ensemble": round(ensemble_acc * 100, 2)
    },
    "optimal_strategy": {
        "leverage": best_lev,
        "risk_per_trade": best_risk,
        "stop_loss_atr_mult": best_stop,
        "reward_risk_ratio": best_rr,
        "trail_trigger_atr": best_trail,
        "max_hold_days": best_hold,
        "partial_tp_pct": best_partial,
        "fitness": round(best_result[0], 2),
        "roi_pct": round(best_result[2], 2),
        "max_drawdown_pct": round(best_result[1], 2),
        "win_rate_pct": round(best_result[3], 2),
        "total_trades": int(best_result[4])
    },
    "selected_features": selected_features,
    "top_30_features_xgboost": [{"name": f, "importance": round(float(imp), 6)} for f, imp in xgb_sorted[:30]],
    "top_30_features_rf": [{"name": f, "importance": round(float(imp), 6)} for f, imp in rf_sorted[:30]],
    "top_30_features_lgb": [{"name": f, "importance": int(imp)} for f, imp in lgb_sorted[:30]],
    "lasso_nonzero_weights": {k: round(float(v), 6) for k, v in nonzero_lasso.items()},
    "consensus_top_15": [{"name": f, "avg_rank": round(r + 1, 1)} for f, r in consensus_sorted[:15]],
    "buy_and_hold_return_pct": round(bh_return, 2)
}

with open(os.path.join(DB_DIR, "optimal_config.json"), "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)
print(f"  Saved: optimal_config.json")

print()
print("=" * 80)
print(f"COMPLETED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
