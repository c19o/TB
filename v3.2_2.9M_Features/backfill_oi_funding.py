#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
backfill_oi_funding.py — Backfill OI + funding from Bitget history
===================================================================
Bitget provides:
  - Historical funding rates (paginated, goes back years)
  - Current OI snapshot (polls and stores)

Also adds Coinbase Premium Index by comparing Coinbase vs Bitget spot.

Usage:
    python backfill_oi_funding.py                # backfill everything
    python backfill_oi_funding.py --funding      # only funding rates
    python backfill_oi_funding.py --oi           # only OI snapshots
"""
import os
import sys
import time
import sqlite3
import argparse
import requests
from datetime import datetime, timezone, timedelta

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OI_DB = os.path.join(PROJECT_DIR, 'open_interest.db')
FUNDING_DB = os.path.join(PROJECT_DIR, 'funding_rates.db')
ONCHAIN_DB = os.path.join(PROJECT_DIR, 'onchain_data.db')

# Bitget API endpoints
BITGET_FUNDING_HISTORY = "https://api.bitget.com/api/v2/mix/market/history-fund-rate"
BITGET_OI = "https://api.bitget.com/api/v2/mix/market/open-interest"
BITGET_TICKER = "https://api.bitget.com/api/v2/spot/market/tickers"
COINBASE_SPOT = "https://api.coinbase.com/v2/prices/BTC-USD/spot"


def init_oi_db():
    conn = sqlite3.connect(OI_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS open_interest (
            timestamp TEXT NOT NULL,
            ts_unix INTEGER,
            oi_contracts REAL,
            oi_usd REAL,
            source TEXT DEFAULT 'bitget',
            PRIMARY KEY (timestamp, source)
        )
    """)
    conn.commit()
    return conn


def backfill_funding():
    """Backfill historical funding rates from Bitget (paginated)."""
    conn = sqlite3.connect(FUNDING_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS funding_rates (
            timestamp TEXT NOT NULL,
            ts_unix INTEGER,
            symbol TEXT DEFAULT 'BTCUSDT',
            funding_rate REAL,
            PRIMARY KEY (timestamp, symbol)
        )
    """)
    conn.commit()

    # Find latest existing timestamp
    row = conn.execute("SELECT MAX(ts_unix) FROM funding_rates WHERE symbol='BTCUSDT'").fetchone()
    latest_ts = row[0] if row[0] else 0
    print(f"  Latest existing funding rate: {datetime.fromtimestamp(latest_ts/1000, tz=timezone.utc) if latest_ts else 'none'}")

    # Bitget funding rates: paginated, each page has up to 100 entries
    # We need to page backwards from now
    total_new = 0
    page_end_time = None  # None = start from most recent
    empty_pages = 0

    while True:
        params = {
            'symbol': 'BTCUSDT',
            'productType': 'USDT-FUTURES',
            'pageSize': '100',
        }
        if page_end_time:
            params['endTime'] = str(page_end_time)

        try:
            r = requests.get(BITGET_FUNDING_HISTORY, params=params, timeout=30)
            if r.status_code != 200:
                print(f"  HTTP {r.status_code}, retrying...")
                time.sleep(5)
                continue

            data = r.json().get('data', [])
            if not data:
                empty_pages += 1
                if empty_pages >= 3:
                    break
                if page_end_time:
                    page_end_time -= 86400000 * 30  # jump back 30 days
                continue

            empty_pages = 0
            batch_new = 0
            oldest_ts = None

            for entry in data:
                ts_ms = int(entry['fundingTime'])
                rate = float(entry['fundingRate'])
                dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                ts_str = dt.strftime('%Y-%m-%d %H:%M:%S')

                if ts_ms <= latest_ts:
                    # We've reached data we already have
                    if batch_new == 0:
                        print(f"  Reached existing data at {ts_str}")
                        conn.close()
                        print(f"  Total new funding rates: {total_new}")
                        return total_new
                    continue

                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO funding_rates (timestamp, ts_unix, symbol, funding_rate) "
                        "VALUES (?, ?, 'BTCUSDT', ?)",
                        (ts_str, ts_ms, rate)
                    )
                    batch_new += 1
                except sqlite3.IntegrityError:
                    pass

                oldest_ts = ts_ms

            conn.commit()
            total_new += batch_new

            if oldest_ts:
                page_end_time = oldest_ts - 1
                dt = datetime.fromtimestamp(oldest_ts / 1000, tz=timezone.utc)
                print(f"    {total_new} new rates... back to {dt.strftime('%Y-%m-%d')}")

            time.sleep(0.2)

        except Exception as e:
            print(f"  Error: {e}, retrying in 5s...")
            time.sleep(5)

    conn.close()
    print(f"  Total new funding rates: {total_new}")
    return total_new


def snapshot_oi():
    """Take a current OI snapshot and store it."""
    conn = init_oi_db()
    try:
        r = requests.get(BITGET_OI,
            params={'symbol': 'BTCUSDT', 'productType': 'USDT-FUTURES'}, timeout=10)
        data = r.json()
        if data.get('code') == '00000':
            oi_list = data['data']['openInterestList']
            ts_ms = int(data['data']['ts'])
            if oi_list:
                oi_contracts = float(oi_list[0]['size'])
                # Estimate USD value (need current price)
                try:
                    r2 = requests.get(BITGET_TICKER,
                        params={'symbol': 'BTCUSDT'}, timeout=10)
                    price_data = r2.json().get('data', [])
                    price = float(price_data[0]['lastPr']) if price_data else 0
                    oi_usd = oi_contracts * price
                except Exception:
                    oi_usd = 0

                dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                ts_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                conn.execute(
                    "INSERT OR IGNORE INTO open_interest (timestamp, ts_unix, oi_contracts, oi_usd, source) "
                    "VALUES (?, ?, ?, ?, 'bitget')",
                    (ts_str, ts_ms, oi_contracts, oi_usd)
                )
                conn.commit()
                print(f"  OI snapshot: {oi_contracts:.2f} BTC (${oi_usd:,.0f}) at {ts_str}")
    except Exception as e:
        print(f"  OI snapshot error: {e}")
    conn.close()


def get_coinbase_premium():
    """Compute Coinbase Premium = (Coinbase price - Bitget price) / Bitget price."""
    try:
        # Coinbase spot
        r1 = requests.get(COINBASE_SPOT, timeout=10)
        cb_price = float(r1.json()['data']['amount'])

        # Bitget spot
        r2 = requests.get(BITGET_TICKER, params={'symbol': 'BTCUSDT'}, timeout=10)
        bg_price = float(r2.json()['data'][0]['lastPr'])

        premium = (cb_price - bg_price) / bg_price
        return premium, cb_price, bg_price
    except Exception as e:
        print(f"  Coinbase premium error: {e}")
        return None, None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--funding', action='store_true', help='Backfill funding rates only')
    parser.add_argument('--oi', action='store_true', help='Snapshot OI only')
    parser.add_argument('--premium', action='store_true', help='Check Coinbase premium only')
    args = parser.parse_args()

    do_all = not (args.funding or args.oi or args.premium)

    print("\n=== OI + Funding Backfill ===\n")

    if do_all or args.funding:
        print("--- Funding Rate Backfill ---")
        backfill_funding()

    if do_all or args.oi:
        print("\n--- OI Snapshot ---")
        snapshot_oi()

    if do_all or args.premium:
        print("\n--- Coinbase Premium ---")
        premium, cb, bg = get_coinbase_premium()
        if premium is not None:
            print(f"  Coinbase: ${cb:,.2f}  Bitget: ${bg:,.2f}  Premium: {premium*100:.4f}%")

    print("\nDone.")


if __name__ == '__main__':
    main()
