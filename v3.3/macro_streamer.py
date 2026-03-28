#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
macro_streamer.py — Continuous macro data poller
==================================================
Polls every 15 minutes for all 14 macro tickers:
  SPX, NASDAQ, DXY, Gold, VIX, US10Y, Russell, Oil,
  Silver, MSTR, COIN, HYG, TLT, IBIT

Uses yfinance if available, else stores NULL (missing).
Stores in macro_data.db matching the existing schema (date column, 14 tickers).

Run as: python macro_streamer.py
"""
import os
import sys
import io
import time
import sqlite3
import logging
from datetime import datetime, timezone

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

from universal_numerology import digital_root

# Try to import yfinance
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_DIR, 'macro_streamer.log'), encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

# Poll interval (seconds)
POLL_INTERVAL = 900  # 15 minutes

DB_PATH = os.path.join(PROJECT_DIR, 'macro_data.db')

# Tickers to track — keys match macro_data.db column names
TICKERS = {
    "spx":     "^GSPC",
    "nasdaq":  "^IXIC",
    "dxy":     "DX-Y.NYB",
    "gold":    "GC=F",
    "vix":     "^VIX",
    "us10y":   "^TNX",
    "russell":  "^RUT",
    "oil":     "CL=F",
    "silver":  "SI=F",
    "mstr":    "MSTR",
    "coin":    "COIN",
    "hyg":     "HYG",
    "tlt":     "TLT",
    "ibit":    "IBIT",
}


# ============================================================
# Database
# ============================================================

def init_db():
    """Open macro_data.db, create table if not exists (matching existing schema)."""
    conn = sqlite3.connect(DB_PATH)
    # Build column defs dynamically from TICKERS
    ticker_cols = ", ".join(f"{name} REAL" for name in TICKERS)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS macro_data (
            date TEXT NOT NULL,
            {ticker_cols}
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_macro_date ON macro_data(date)
    """)
    conn.commit()
    return conn


def get_last_timestamp(conn):
    """Get the most recent date in the DB."""
    row = conn.execute(
        "SELECT MAX(date) FROM macro_data"
    ).fetchone()
    return row[0] if row and row[0] else None


# ============================================================
# Data fetchers
# ============================================================

def fetch_yfinance_prices():
    """Fetch current prices using yfinance."""
    prices = {}
    for name, ticker in TICKERS.items():
        try:
            tk = yf.Ticker(ticker)
            # fast_info or history for latest price
            try:
                price = tk.fast_info.get("lastPrice", None)
                if price is None:
                    price = tk.fast_info.get("last_price", None)
            except Exception:
                price = None

            if price is None:
                # Fallback: get last close from 1d history
                hist = tk.history(period="1d")
                if not hist.empty:
                    price = float(hist["Close"].iloc[-1])

            if price is not None:
                prices[name] = float(price)
                log.info(f"    {name} ({ticker}): {prices[name]:.2f}")
            else:
                log.warning(f"    {name} ({ticker}): no price available — skipping (NaN=missing, not 0)")
        except Exception as e:
            log.warning(f"    {name} ({ticker}): fetch failed — {e} — skipping")

    return prices


def fetch_fallback_prices():
    """Fallback when yfinance is not available — returns None (missing, not zero)."""
    log.warning("  yfinance not available — storing None for all tickers (NaN = missing)")
    return {name: None for name in TICKERS}


def fetch_prices():
    """Fetch prices using best available method."""
    if HAS_YFINANCE:
        return fetch_yfinance_prices()
    else:
        return fetch_fallback_prices()


# ============================================================
# Digital root helpers
# ============================================================

def price_dr(value):
    """Compute digital root of a price value.
    Removes decimal, takes integer part for DR.
    """
    if value is None or value == 0:
        return 0
    try:
        # Use the integer part of the price for digital root
        int_val = int(abs(value))
        if int_val == 0:
            # For small values like VIX, use digits without decimal
            digits_str = str(abs(value)).replace('.', '').replace('-', '').lstrip('0')
            if not digits_str:
                return 0
            return digital_root(int(digits_str))
        return digital_root(int_val)
    except (ValueError, TypeError):
        return 0


# ============================================================
# Main poll
# ============================================================

def poll_once(conn):
    """Fetch all 14 macro tickers and insert/update today's row."""
    now_utc = datetime.now(timezone.utc)
    date_str = now_utc.strftime("%Y-%m-%d")

    # Check if we already have recent data (within 14 min)
    last_ts = get_last_timestamp(conn)
    if last_ts:
        try:
            # Handle both date-only and datetime formats
            fmt = "%Y-%m-%d %H:%M:%S" if " " in last_ts else "%Y-%m-%d"
            last_dt = datetime.strptime(last_ts, fmt).replace(tzinfo=timezone.utc)
            elapsed = (now_utc - last_dt).total_seconds()
            if elapsed < 840:  # 14 minutes minimum gap
                log.debug(f"  Skipping — only {elapsed:.0f}s since last insert")
                return False
        except ValueError:
            pass

    log.info(f"  Polling macro data at {date_str}...")

    # Fetch prices for all 14 tickers
    prices = fetch_prices()

    # Log summary
    for name in TICKERS:
        val = prices.get(name, None)
        if val is not None:
            log.info(f"    {name:>8s}: {val:.2f}")
        else:
            log.info(f"    {name:>8s}: NULL (missing)")

    # Build dynamic INSERT — use date column, all 14 ticker columns
    col_names = ", ".join(TICKERS.keys())
    placeholders = ", ".join(["?"] * (1 + len(TICKERS)))
    values = [date_str] + [prices.get(name, None) for name in TICKERS]

    # Check if today's row already exists — UPDATE instead of INSERT
    existing = conn.execute(
        "SELECT rowid FROM macro_data WHERE date = ?", (date_str,)
    ).fetchone()

    if existing:
        # Update existing row with fresh prices
        set_clause = ", ".join(f"{name} = ?" for name in TICKERS)
        update_vals = [prices.get(name, None) for name in TICKERS] + [date_str]
        conn.execute(
            f"UPDATE macro_data SET {set_clause} WHERE date = ?",
            update_vals
        )
        log.info(f"    Updated existing row for {date_str}")
    else:
        conn.execute(
            f"INSERT INTO macro_data (date, {col_names}) VALUES ({placeholders})",
            values
        )
        log.info(f"    Inserted new row for {date_str}")

    conn.commit()
    return True


# ============================================================
# Streamer loop
# ============================================================

def run_streamer():
    log.info("=" * 60)
    log.info("  MACRO STREAMER — Market Data Poller")
    log.info(f"  DB: {DB_PATH}")
    log.info(f"  Poll interval: {POLL_INTERVAL}s")
    log.info(f"  yfinance available: {HAS_YFINANCE}")
    log.info("=" * 60)

    conn = init_db()

    # Show current state
    last_ts = get_last_timestamp(conn)
    row_count = conn.execute("SELECT COUNT(*) FROM macro_data").fetchone()[0]
    log.info(f"  Existing rows: {row_count}")
    if last_ts:
        log.info(f"  Last data point: {last_ts}")
    else:
        log.info(f"  Database is empty — first poll starting now")

    log.info(f"\n  Starting continuous poll loop...\n")

    while True:
        try:
            poll_once(conn)
            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            log.info("\n  Shutting down macro streamer...")
            break
        except Exception as e:
            log.error(f"  Streamer error: {e}")
            time.sleep(30)

    conn.close()


if __name__ == "__main__":
    run_streamer()
