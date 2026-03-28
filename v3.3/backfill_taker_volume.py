#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
backfill_taker_volume.py — Backfill taker_buy_volume from Binance.US
=====================================================================
Binance.US klines return 12 fields including taker_buy_base_asset_volume.
This script fetches klines and updates the existing btc_prices.db rows.

Usage:
    python backfill_taker_volume.py              # all timeframes
    python backfill_taker_volume.py --tf 1h      # single timeframe
    python backfill_taker_volume.py --tf 1h 4h   # multiple timeframes
"""
import os
import sys
import time
import sqlite3
import argparse
import requests
from datetime import datetime, timezone

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT_DIR, 'btc_prices.db')

BINANCE_US_KLINES = "https://api.binance.us/api/v3/klines"
SYMBOL = "BTCUSDT"
MAX_PER_REQUEST = 1000

# Map our TF names to Binance interval strings
TF_MAP = {
    '5m': '5m',
    '15m': '15m',
    '1h': '1h',
    '4h': '4h',
    '1d': '1d',
    '1w': '1w',
}


def backfill_tf(conn, tf, start_ms=None):
    """Backfill taker_buy_volume for a single timeframe."""
    interval = TF_MAP.get(tf)
    if not interval:
        print(f"  Unknown timeframe: {tf}")
        return 0

    # Find rows that need updating (NULL or 0 taker_buy_volume)
    row = conn.execute(
        "SELECT MIN(open_time), MAX(open_time), COUNT(*) FROM ohlcv "
        "WHERE symbol='BTC/USDT' AND timeframe=? AND "
        "(taker_buy_volume IS NULL OR taker_buy_volume = '' OR CAST(taker_buy_volume AS REAL) = 0)",
        (tf,)
    ).fetchone()

    if row[2] == 0:
        print(f"  {tf}: all rows already have taker_buy_volume")
        return 0

    total_need = row[2]
    first_ts = row[0]
    last_ts = row[1]
    print(f"  {tf}: {total_need} rows need taker_buy_volume")
    print(f"    Range: {datetime.fromtimestamp(first_ts/1000, tz=timezone.utc).strftime('%Y-%m-%d')} "
          f"to {datetime.fromtimestamp(last_ts/1000, tz=timezone.utc).strftime('%Y-%m-%d')}")

    # Track progress with resume support
    if start_ms is None:
        start_ms = first_ts

    updated = 0
    current_ms = start_ms
    end_ms = last_ts + 1
    batch_count = 0

    while current_ms < end_ms:
        try:
            params = {
                'symbol': SYMBOL,
                'interval': interval,
                'startTime': str(current_ms),
                'limit': str(MAX_PER_REQUEST),
            }
            r = requests.get(BINANCE_US_KLINES, params=params, timeout=30)

            if r.status_code == 429:
                retry_after = int(r.headers.get('Retry-After', 60))
                print(f"    Rate limited, waiting {retry_after}s...")
                time.sleep(retry_after)
                continue

            if r.status_code != 200:
                print(f"    HTTP {r.status_code}: {r.text[:200]}")
                time.sleep(5)
                continue

            data = r.json()
            if not data:
                break

            # Binance kline format:
            # [0] open_time, [1] open, [2] high, [3] low, [4] close,
            # [5] volume, [6] close_time, [7] quote_volume, [8] trades,
            # [9] taker_buy_base, [10] taker_buy_quote, [11] ignore
            batch_updated = 0
            for candle in data:
                open_time = candle[0]
                taker_buy_vol = float(candle[9])
                taker_buy_quote = float(candle[10])
                trades = int(candle[8])
                quote_volume = float(candle[7])

                # Update existing row
                cur = conn.execute(
                    "UPDATE ohlcv SET taker_buy_volume=?, taker_buy_quote=?, "
                    "trades=?, quote_volume=? "
                    "WHERE symbol='BTC/USDT' AND timeframe=? AND open_time=?",
                    (taker_buy_vol, taker_buy_quote, trades, quote_volume, tf, open_time)
                )
                if cur.rowcount > 0:
                    batch_updated += 1

            conn.commit()
            updated += batch_updated
            batch_count += 1

            # Move to next batch
            current_ms = data[-1][0] + 1

            if batch_count % 20 == 0:
                dt = datetime.fromtimestamp(data[-1][0] / 1000, tz=timezone.utc)
                pct = min(100, updated / max(1, total_need) * 100)
                print(f"    {updated}/{total_need} updated ({pct:.0f}%) ... {dt.strftime('%Y-%m-%d %H:%M')}")

            # Rate limiting: Binance.US allows 1200 req/min
            time.sleep(0.1)

        except requests.exceptions.Timeout:
            print("    Timeout, retrying in 10s...")
            time.sleep(10)
        except requests.exceptions.ConnectionError:
            print("    Connection error, retrying in 30s...")
            time.sleep(30)
        except Exception as e:
            print(f"    Error: {e}, retrying in 5s...")
            time.sleep(5)

    print(f"  {tf}: Done — {updated} rows updated")
    return updated


def main():
    parser = argparse.ArgumentParser(description='Backfill taker_buy_volume from Binance.US')
    parser.add_argument('--tf', nargs='*', default=None,
                        help='Timeframes to backfill (default: all)')
    args = parser.parse_args()

    timeframes = args.tf if args.tf else ['1h', '4h', '15m', '5m', '1w']

    print("\n=== Backfill taker_buy_volume from Binance.US ===\n")
    print(f"Timeframes: {timeframes}")
    print(f"Source: {BINANCE_US_KLINES}")
    print(f"Note: Binance.US volume is lower than Binance main,")
    print(f"      but taker_buy_ratio (buy/total) is still valid.\n")

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    total = 0
    for tf in timeframes:
        print(f"\n--- {tf} ---")
        total += backfill_tf(conn, tf)

    conn.close()
    print(f"\n=== Total: {total} rows updated ===")


if __name__ == '__main__':
    main()
