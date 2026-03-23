#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
crypto_streamer.py -- Continuous crypto on-chain data poller
============================================================
Polls every 5 minutes:
  - Blockchain.info stats API: block height, hash rate, mempool size, est BTC sent
  - Bitget API: funding rate, open interest (BTCUSDT)
  - Alternative.me: Fear & Greed Index
  - Blockchain.info: whale volume (24h estimated BTC sent)
  - OKX: liquidation orders (BTCUSDT)

Stores in onchain_data.db with digital roots via universal_numerology.

Run as:
  python crypto_streamer.py          # continuous loop
  python crypto_streamer.py --once   # single poll then exit
"""
import os
import sys
import io
import time
import sqlite3
import logging
import json
import argparse
from datetime import datetime, timezone

try:
    import requests
except ImportError:
    import urllib.request
    import urllib.error
    requests = None

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

from universal_numerology import digital_root, numerology

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_DIR, 'crypto_streamer.log'), encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

# Poll interval (seconds)
POLL_INTERVAL = 300  # 5 minutes

DB_PATH = os.path.join(PROJECT_DIR, 'onchain_data.db')

# API endpoints
# blockchain.info/q/hashrate is dead (404), use stats API instead
BLOCKCHAIN_STATS_URL = "https://api.blockchain.info/stats"
BLOCKCHAIN_BLOCK_HEIGHT = "https://blockchain.info/q/getblockcount"
BLOCKCHAIN_MEMPOOL = "https://blockchain.info/q/unconfirmedcount"

# Binance futures is geo-blocked (451), use Bitget instead
BITGET_FUNDING_RATE = "https://api.bitget.com/api/v2/mix/market/history-fund-rate"
BITGET_OPEN_INTEREST = "https://api.bitget.com/api/v2/mix/market/open-interest"

FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1"

# New data sources
# Whale volume: blockchain.info stats -> estimated_btc_sent (24h, in satoshi)
# Liquidations: OKX public API (Binance is geo-blocked)
OKX_LIQUIDATIONS = "https://www.okx.com/api/v5/public/liquidation-orders"

# Coinbase Premium: compare Coinbase spot vs Bitget spot
COINBASE_SPOT = "https://api.coinbase.com/v2/prices/BTC-USD/spot"
BITGET_TICKER = "https://api.bitget.com/api/v2/spot/market/tickers"


# ============================================================
# HTTP helpers
# ============================================================

def http_get(url, json_response=False, timeout=15, params=None):
    """Fetch URL using requests if available, else urllib."""
    try:
        if requests:
            resp = requests.get(url, timeout=timeout, params=params)
            resp.raise_for_status()
            return resp.json() if json_response else resp.text.strip()
        else:
            if params:
                query = "&".join(f"{k}={v}" for k, v in params.items())
                url = f"{url}?{query}"
            req = urllib.request.Request(url, headers={"User-Agent": "crypto_streamer/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8").strip()
                return json.loads(body) if json_response else body
    except Exception as e:
        log.warning(f"  HTTP GET failed for {url}: {e}")
        return None


# ============================================================
# Database
# ============================================================

def init_db():
    """Create onchain_data.db and table if not exists."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS onchain_data (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            block_height INTEGER,
            hash_rate   REAL,
            mempool_size INTEGER,
            funding_rate REAL,
            open_interest REAL,
            fear_greed  INTEGER,
            block_dr    INTEGER,
            funding_dr  INTEGER,
            oi_dr       INTEGER,
            fg_dr       INTEGER,
            whale_volume_btc REAL,
            liq_long_count INTEGER,
            liq_short_count INTEGER,
            liq_long_vol REAL,
            liq_short_vol REAL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_onchain_ts ON onchain_data(timestamp)
    """)
    # Add new columns if they dont exist (for existing DBs)
    for col_def in [
        ("whale_volume_btc", "REAL"),
        ("liq_long_count", "INTEGER"),
        ("liq_short_count", "INTEGER"),
        ("liq_long_vol", "REAL"),
        ("liq_short_vol", "REAL"),
        ("coinbase_premium", "REAL"),
    ]:
        try:
            conn.execute(f"ALTER TABLE onchain_data ADD COLUMN {col_def[0]} {col_def[1]}")
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.commit()
    return conn


def get_last_timestamp(conn):
    """Get the most recent timestamp in the DB."""
    row = conn.execute(
        "SELECT MAX(timestamp) FROM onchain_data"
    ).fetchone()
    return row[0] if row and row[0] else None


# ============================================================
# Data fetchers
# ============================================================

def fetch_blockchain_info():
    """Fetch block height, hash rate, mempool size, whale volume from blockchain.info."""
    block_height = None
    hash_rate = None
    mempool_size = None
    whale_volume_btc = None

    # Use stats API for hash rate and whale volume (24h estimated BTC sent)
    stats = http_get(BLOCKCHAIN_STATS_URL, json_response=True)
    if stats and isinstance(stats, dict):
        try:
            hash_rate = float(stats.get("hash_rate", 0))
        except (ValueError, TypeError):
            log.warning("  Bad hash_rate in stats response")
        try:
            # estimated_btc_sent is in satoshi, convert to BTC
            sat = stats.get("estimated_btc_sent", 0)
            if sat:
                whale_volume_btc = float(sat) / 1e8
        except (ValueError, TypeError):
            log.warning("  Bad estimated_btc_sent in stats response")

    # Block height from dedicated endpoint (still works)
    val = http_get(BLOCKCHAIN_BLOCK_HEIGHT)
    if val is not None:
        try:
            block_height = int(val)
        except (ValueError, TypeError):
            log.warning(f"  Bad block height response: {val}")

    # Mempool unconfirmed tx count (still works)
    val = http_get(BLOCKCHAIN_MEMPOOL)
    if val is not None:
        try:
            mempool_size = int(val)
        except (ValueError, TypeError):
            log.warning(f"  Bad mempool response: {val}")

    return block_height, hash_rate, mempool_size, whale_volume_btc


def fetch_funding_and_oi():
    """Fetch funding rate and open interest from Bitget (Binance is geo-blocked)."""
    funding_rate = None
    open_interest = None

    # Funding rate from Bitget
    data = http_get(BITGET_FUNDING_RATE, json_response=True, params={
        "symbol": "BTCUSDT",
        "productType": "USDT-FUTURES",
        "pageSize": "1",
        "pageNo": "1",
    })
    if data and isinstance(data, dict) and data.get("code") == "00000":
        records = data.get("data", [])
        if records and len(records) > 0:
            try:
                funding_rate = float(records[0].get("fundingRate", 0))
            except (ValueError, TypeError, KeyError):
                log.warning(f"  Bad Bitget funding rate response: {data}")

    # Open interest from Bitget
    data = http_get(BITGET_OPEN_INTEREST, json_response=True, params={
        "symbol": "BTCUSDT",
        "productType": "USDT-FUTURES",
    })
    if data and isinstance(data, dict) and data.get("code") == "00000":
        oi_data = data.get("data", {})
        if isinstance(oi_data, dict):
            oi_list = oi_data.get("openInterestList", [])
            if oi_list and len(oi_list) > 0:
                try:
                    open_interest = float(oi_list[0].get("size", 0))
                except (ValueError, TypeError):
                    log.warning(f"  Bad Bitget OI response: {data}")

    return funding_rate, open_interest


def fetch_fear_greed():
    """Fetch Fear & Greed Index from alternative.me."""
    data = http_get(FEAR_GREED_URL, json_response=True)
    if data and isinstance(data, dict):
        try:
            fg_data = data.get("data", [])
            if fg_data and len(fg_data) > 0:
                return int(fg_data[0].get("value", 0))
        except (ValueError, TypeError, KeyError):
            log.warning(f"  Bad fear/greed response: {data}")
    return None


def fetch_liquidations():
    """Fetch recent liquidation orders from OKX (Binance geo-blocked).
    Returns (long_count, short_count, long_vol_usd, short_vol_usd)."""
    long_count = 0
    short_count = 0
    long_vol = 0.0
    short_vol = 0.0

    data = http_get(OKX_LIQUIDATIONS, json_response=True, params={
        "instType": "SWAP",
        "uly": "BTC-USDT",
        "state": "filled",
        "limit": "100",
    })
    if data and isinstance(data, dict) and data.get("code") == "0":
        entries = data.get("data", [])
        for entry in entries:
            details = entry.get("details", [])
            for d in details:
                side = d.get("side", "")
                try:
                    sz = float(d.get("sz", 0))
                except (ValueError, TypeError):
                    sz = 0.0
                # side=sell means long got liquidated, side=buy means short got liquidated
                if side == "sell":
                    long_count += 1
                    long_vol += sz
                elif side == "buy":
                    short_count += 1
                    short_vol += sz

    if long_count == 0 and short_count == 0:
        return None, None, None, None

    return long_count, short_count, long_vol, short_vol


# ============================================================
# Digital root helpers
# ============================================================

def safe_dr(value):
    """Compute digital root, returning None if value is None."""
    if value is None:
        return None
    try:
        # For floats like funding rate, use significant digits
        if isinstance(value, float):
            # Use fixed-point format to avoid scientific notation (e.g. 7.8e-05)
            digits_str = f"{abs(value):.12f}".replace('.', '').lstrip('0').rstrip('0')
            if not digits_str:
                return 0
            return digital_root(int(digits_str))
        return digital_root(int(value))
    except (ValueError, TypeError):
        return None


def fetch_coinbase_premium():
    """Compute Coinbase Premium = (Coinbase - Bitget) / Bitget.
    Positive = US institutional buying. Negative = US selling."""
    try:
        cb_data = http_get(COINBASE_SPOT, json_response=True)
        if not cb_data or 'data' not in cb_data:
            return None
        cb_price = float(cb_data['data']['amount'])

        bg_data = http_get(BITGET_TICKER, json_response=True, params={'symbol': 'BTCUSDT'})
        if not bg_data or not bg_data.get('data'):
            return None
        bg_price = float(bg_data['data'][0]['lastPr'])

        if bg_price > 0:
            return (cb_price - bg_price) / bg_price
    except Exception as e:
        log.warning(f"  Coinbase premium error: {e}")
    return None


# ============================================================
# Main poll
# ============================================================

def poll_once(conn):
    """Fetch all data sources and insert a row."""
    now_utc = datetime.now(timezone.utc)
    ts_str = now_utc.strftime("%Y-%m-%d %H:%M:%S")

    # Check if we already have data for this minute
    last_ts = get_last_timestamp(conn)
    if last_ts:
        # Only insert if at least 4 minutes have passed
        try:
            last_dt = datetime.strptime(last_ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            elapsed = (now_utc - last_dt).total_seconds()
            if elapsed < 240:  # 4 minutes minimum gap
                log.debug(f"  Skipping -- only {elapsed:.0f}s since last insert")
                return False
        except ValueError:
            pass

    log.info(f"  Polling on-chain data at {ts_str}...")

    # Fetch all sources
    block_height, hash_rate, mempool_size, whale_volume_btc = fetch_blockchain_info()
    funding_rate, open_interest = fetch_funding_and_oi()
    fear_greed = fetch_fear_greed()
    liq_long_count, liq_short_count, liq_long_vol, liq_short_vol = fetch_liquidations()
    coinbase_premium = fetch_coinbase_premium()

    # Compute digital roots
    block_dr = safe_dr(block_height)
    funding_dr = safe_dr(funding_rate)
    oi_dr = safe_dr(open_interest)
    fg_dr = safe_dr(fear_greed)

    # Log summary
    log.info(f"    Block: {block_height} (DR={block_dr})  HashRate: {hash_rate}")
    log.info(f"    Mempool: {mempool_size}  Funding: {funding_rate} (DR={funding_dr})")
    log.info(f"    OI: {open_interest} (DR={oi_dr})  F&G: {fear_greed} (DR={fg_dr})")
    log.info(f"    WhaleVol: {whale_volume_btc} BTC")
    log.info(f"    Liqs: long={liq_long_count}/{liq_long_vol} short={liq_short_count}/{liq_short_vol}")
    log.info(f"    Coinbase Premium: {coinbase_premium*100:.4f}%" if coinbase_premium is not None else "    Coinbase Premium: N/A")

    # Insert
    conn.execute("""
        INSERT INTO onchain_data
        (timestamp, block_height, hash_rate, mempool_size, funding_rate,
         open_interest, fear_greed, block_dr, funding_dr, oi_dr, fg_dr,
         whale_volume_btc, liq_long_count, liq_short_count, liq_long_vol, liq_short_vol,
         coinbase_premium)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        ts_str, block_height, hash_rate, mempool_size, funding_rate,
        open_interest, fear_greed, block_dr, funding_dr, oi_dr, fg_dr,
        whale_volume_btc, liq_long_count, liq_short_count, liq_long_vol, liq_short_vol,
        coinbase_premium
    ))
    conn.commit()
    log.info(f"    Inserted row at {ts_str}")
    return True


# ============================================================
# Streamer loop
# ============================================================

def run_streamer(once=False):
    log.info("=" * 60)
    log.info("  CRYPTO STREAMER -- On-Chain Data Poller")
    log.info(f"  DB: {DB_PATH}")
    log.info(f"  Poll interval: {POLL_INTERVAL}s")
    if once:
        log.info("  Mode: single poll (--once)")
    log.info("=" * 60)

    conn = init_db()

    # Show current state
    last_ts = get_last_timestamp(conn)
    row_count = conn.execute("SELECT COUNT(*) FROM onchain_data").fetchone()[0]
    log.info(f"  Existing rows: {row_count}")
    if last_ts:
        log.info(f"  Last data point: {last_ts}")
    else:
        log.info(f"  Database is empty -- first poll starting now")

    if once:
        log.info("  Running single poll...")
        poll_once(conn)
        new_count = conn.execute("SELECT COUNT(*) FROM onchain_data").fetchone()[0]
        log.info(f"  Rows after poll: {new_count}")
        conn.close()
        return

    log.info(f"\n  Starting continuous poll loop...\n")

    while True:
        try:
            poll_once(conn)
            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            log.info("\n  Shutting down crypto streamer...")
            break
        except Exception as e:
            log.error(f"  Streamer error: {e}")
            time.sleep(30)

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crypto on-chain data streamer")
    parser.add_argument("--once", action="store_true", help="Single poll then exit")
    args = parser.parse_args()
    run_streamer(once=args.once)
