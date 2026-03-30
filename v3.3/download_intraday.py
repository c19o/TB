#!/usr/bin/env python
"""
Download missing intraday OHLCV data + resample 4h/1w from existing data.

Sources:
  - 15m, 5m: Binance Spot /api/v3/klines (free, no API key, 1000 candles/request)
  - 4h: Resampled from existing 1h data in multi_asset_prices.db
  - 1w: Resampled from existing 1d data in multi_asset_prices.db

Saves to multi_asset_prices.db with same schema.
Resume support — skips symbols that already have data for a timeframe.
"""

import os
import sys
import time
import json
import sqlite3
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone

os.environ['PYTHONUNBUFFERED'] = '1'

V2_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(V2_DIR, "multi_asset_prices.db")

CRYPTO_SYMBOLS = {
    'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'XRP': 'XRPUSDT', 'LTC': 'LTCUSDT',
    'SOL': 'SOLUSDT', 'DOGE': 'DOGEUSDT', 'ADA': 'ADAUSDT', 'BNB': 'BNBUSDT',
    'AVAX': 'AVAXUSDT', 'LINK': 'LINKUSDT', 'DOT': 'DOTUSDT',
    'MATIC': 'MATICUSDT', 'UNI': 'UNIUSDT', 'AAVE': 'AAVEUSDT',
}

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def get_row_count(conn, symbol, tf):
    cur = conn.execute("SELECT COUNT(*) FROM ohlcv WHERE symbol=? AND timeframe=?", (symbol, tf))
    return cur.fetchone()[0]


def get_last_ts(conn, symbol, tf):
    cur = conn.execute("SELECT MAX(open_time) FROM ohlcv WHERE symbol=? AND timeframe=?", (symbol, tf))
    row = cur.fetchone()
    return row[0] if row and row[0] else None


def download_binance_klines(conn, symbol, binance_pair, tf, start_date='2020-01-01'):
    """Download klines from Binance Spot API. Free, no API key."""
    existing = get_row_count(conn, symbol, tf)
    if existing > 1000:
        log(f"  {symbol} {tf}: already have {existing} rows, checking for new...")

    last_ts = get_last_ts(conn, symbol, tf)
    if last_ts:
        start_ms = last_ts + 1
    else:
        start_ms = int(datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp() * 1000)

    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    total_new = 0
    request_count = 0

    while start_ms < end_ms:
        url = (f"https://api.binance.us/api/v3/klines"
               f"?symbol={binance_pair}&interval={tf}"
               f"&startTime={start_ms}&endTime={end_ms}&limit=1000")

        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                log(f"  Rate limited, sleeping 60s...")
                time.sleep(60)
                continue
            raise
        except Exception as e:
            log(f"  Error: {e}, retrying in 5s...")
            time.sleep(5)
            continue

        if not data:
            break

        rows = []
        for k in data:
            open_time_ms = int(k[0])
            rows.append((
                symbol, tf, open_time_ms,
                float(k[1]), float(k[2]), float(k[3]), float(k[4]),  # O H L C
                float(k[5]),  # volume
                float(k[7]),  # quote_volume
            ))

        conn.executemany("""
            INSERT OR IGNORE INTO ohlcv
            (symbol, timeframe, open_time, open, high, low, close, volume, quote_volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
        conn.commit()

        total_new += len(rows)
        request_count += 1
        start_ms = int(data[-1][0]) + 1  # next candle after last

        if request_count % 10 == 0:
            log(f"  {symbol} {tf}: {total_new:,} candles downloaded ({request_count} requests)")

        # Binance rate limit: 1200 req/min, be conservative
        time.sleep(0.1)

    log(f"  {symbol} {tf}: {total_new:,} new candles ({request_count} requests)")
    return total_new


def resample_tf(conn, symbol, source_tf, target_tf, bars_per_candle):
    """Resample higher TF from lower TF data."""
    existing = get_row_count(conn, symbol, target_tf)
    if existing > 100:
        log(f"  {symbol} {target_tf}: already have {existing} rows, skipping resample")
        return 0

    cur = conn.execute("""
        SELECT open_time, open, high, low, close, volume,
               COALESCE(quote_volume, 0) as quote_volume
        FROM ohlcv WHERE symbol=? AND timeframe=?
        ORDER BY open_time
    """, (symbol, source_tf))
    rows = cur.fetchall()

    if not rows:
        log(f"  {symbol} {target_tf}: no {source_tf} data to resample from")
        return 0

    # Group into candles
    resampled = []
    for i in range(0, len(rows) - bars_per_candle + 1, bars_per_candle):
        chunk = rows[i:i + bars_per_candle]
        if len(chunk) < bars_per_candle:
            break
        open_time = chunk[0][0]
        o = chunk[0][1]
        h = max(r[2] for r in chunk)
        l = min(r[3] for r in chunk)
        c = chunk[-1][4]
        v = sum(r[5] for r in chunk)
        qv = sum(r[6] for r in chunk)
        resampled.append((symbol, target_tf, open_time, o, h, l, c, v, qv))

    conn.executemany("""
        INSERT OR IGNORE INTO ohlcv
        (symbol, timeframe, open_time, open, high, low, close, volume, quote_volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, resampled)
    conn.commit()

    log(f"  {symbol} {target_tf}: {len(resampled):,} candles resampled from {source_tf}")
    return len(resampled)


def main():
    log("=" * 60)
    log("INTRADAY DATA DOWNLOADER")
    log("=" * 60)

    conn = get_conn()

    # ── Step 1: Resample 1w from 1d (all assets) ──
    log("\n--- Step 1: Resample 1w from 1d ---")
    cur = conn.execute("SELECT DISTINCT symbol FROM ohlcv WHERE timeframe='1d'")
    daily_symbols = [r[0] for r in cur.fetchall()]
    for sym in daily_symbols:
        resample_tf(conn, sym, '1d', '1w', 7)

    # ── Step 2: Resample 4h from 1h (crypto only) ──
    log("\n--- Step 2: Resample 4h from 1h ---")
    cur = conn.execute("SELECT DISTINCT symbol FROM ohlcv WHERE timeframe='1h'")
    hourly_symbols = [r[0] for r in cur.fetchall()]
    for sym in hourly_symbols:
        resample_tf(conn, sym, '1h', '4h', 4)

    # ── Step 3: Download 15m from Binance ──
    log("\n--- Step 3: Download 15m from Binance ---")
    for sym, pair in CRYPTO_SYMBOLS.items():
        start = '2020-01-01'  # Most altcoins listed after this
        if sym == 'BTC':
            start = '2017-01-01'
        elif sym in ('ETH', 'LTC', 'XRP'):
            start = '2018-01-01'
        download_binance_klines(conn, sym, pair, '15m', start)

    # ── Step 4: Download 5m from Binance ──
    log("\n--- Step 4: Download 5m from Binance ---")
    for sym, pair in CRYPTO_SYMBOLS.items():
        start = '2020-01-01'
        if sym == 'BTC':
            start = '2017-01-01'
        elif sym in ('ETH', 'LTC', 'XRP'):
            start = '2018-01-01'
        download_binance_klines(conn, sym, pair, '5m', start)

    # ── Summary ──
    log("\n" + "=" * 60)
    log("DOWNLOAD COMPLETE — Summary:")
    log("=" * 60)
    cur = conn.execute("""
        SELECT symbol, timeframe, COUNT(*), MIN(open_time), MAX(open_time)
        FROM ohlcv GROUP BY symbol, timeframe ORDER BY symbol, timeframe
    """)
    for sym, tf, cnt, min_t, max_t in cur.fetchall():
        min_d = datetime.fromtimestamp(min_t / 1000, tz=timezone.utc).strftime('%Y-%m-%d') if min_t else '?'
        max_d = datetime.fromtimestamp(max_t / 1000, tz=timezone.utc).strftime('%Y-%m-%d') if max_t else '?'
        log(f"  {sym:<8} {tf:<5} {cnt:>8,} rows  ({min_d} to {max_d})")

    conn.close()
    log("\nDone!")


if __name__ == '__main__':
    main()
