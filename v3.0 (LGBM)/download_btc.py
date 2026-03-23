"""
Step 3: Download BTC historical OHLCV data from Binance.
1-minute to weekly candles with full volume data.
Resume support — continues from where it left off.
"""
import os
import sys
import io
import time
import sqlite3
from datetime import datetime, timedelta

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import BTC_DB

import ccxt


def init_db():
    conn = sqlite3.connect(BTC_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv (
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            open_time INTEGER NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL,
            quote_volume REAL,
            trades INTEGER,
            taker_buy_volume REAL,
            taker_buy_quote REAL,
            close_time INTEGER,
            PRIMARY KEY (symbol, timeframe, open_time)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_time ON ohlcv(symbol, timeframe, open_time)")
    conn.commit()
    return conn


def get_last_timestamp(conn, symbol, timeframe):
    """Get the last candle timestamp for resume support."""
    row = conn.execute(
        "SELECT MAX(open_time) FROM ohlcv WHERE symbol = ? AND timeframe = ?",
        (symbol, timeframe)
    ).fetchone()
    return row[0] if row[0] else None


def download_candles(symbol="BTC/USDT", timeframe="1d", since_str="2017-08-17", limit_per_req=1000):
    """Download candles from Binance with resume support."""
    conn = init_db()
    # Use Binance.US for US-based access, fallback to other exchanges
    try:
        exchange = ccxt.binanceus({"enableRateLimit": True})
        exchange.load_markets()
    except Exception:
        try:
            exchange = ccxt.kraken({"enableRateLimit": True})
            exchange.load_markets()
            if symbol == "BTC/USDT":
                symbol = "BTC/USD"  # Kraken uses USD
        except Exception:
            exchange = ccxt.coinbasepro({"enableRateLimit": True})
            exchange.load_markets()
            if symbol == "BTC/USDT":
                symbol = "BTC/USD"

    # Check for resume
    last_ts = get_last_timestamp(conn, symbol, timeframe)
    if last_ts:
        since_ms = last_ts + 1  # Start after last candle
        print(f"  Resuming {symbol} {timeframe} from {datetime.utcfromtimestamp(last_ts/1000).isoformat()}")
    else:
        since_ms = exchange.parse8601(since_str + "T00:00:00Z")
        print(f"  Starting {symbol} {timeframe} from {since_str}")

    now_ms = int(time.time() * 1000)
    total_candles = 0
    batch = 0

    while since_ms < now_ms:
        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit_per_req)

            if not candles:
                break

            # Insert candles
            for c in candles:
                # Binance returns: [timestamp, open, high, low, close, volume]
                # For detailed data we need to use the raw API
                conn.execute("""
                    INSERT OR IGNORE INTO ohlcv
                    (symbol, timeframe, open_time, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (symbol, timeframe, c[0], c[1], c[2], c[3], c[4], c[5]))

            conn.commit()
            total_candles += len(candles)
            batch += 1

            # Move to next batch
            since_ms = candles[-1][0] + 1

            if batch % 10 == 0:
                dt = datetime.utcfromtimestamp(candles[-1][0] / 1000)
                print(f"    {total_candles:>8} candles... {dt.strftime('%Y-%m-%d %H:%M')}")

            # Rate limiting
            time.sleep(0.1)

        except ccxt.RateLimitExceeded:
            print("    Rate limited, waiting 30s...")
            time.sleep(30)
        except Exception as e:
            print(f"    Error: {e}, retrying in 5s...")
            time.sleep(5)

    print(f"  Done: {total_candles} candles for {symbol} {timeframe}")
    conn.close()
    return total_candles


def download_all():
    """Download all timeframes."""
    symbol = "BTC/USDT"
    print("\n=== BTC Price Data Download ===\n")

    # Start with higher timeframes (faster), then work down
    timeframes = [
        ("1w", "2017-08-17"),
        ("1d", "2017-08-17"),
        ("4h", "2017-08-17"),
        ("1h", "2017-08-17"),
        ("15m", "2020-01-01"),   # 15m from 2020 to save time
        ("5m", "2021-01-01"),    # 5m from 2021
        ("1m", "2021-06-01"),    # 1m from mid-2021 (Discord server period)
    ]

    for tf, start in timeframes:
        print(f"\n--- {tf} candles ---")
        download_candles(symbol, tf, start)

    # Print final stats
    conn = sqlite3.connect(BTC_DB)
    for tf, _ in timeframes:
        count = conn.execute(
            "SELECT COUNT(*) FROM ohlcv WHERE symbol = ? AND timeframe = ?",
            (symbol, tf)
        ).fetchone()[0]
        first = conn.execute(
            "SELECT MIN(open_time) FROM ohlcv WHERE symbol = ? AND timeframe = ?",
            (symbol, tf)
        ).fetchone()[0]
        last = conn.execute(
            "SELECT MAX(open_time) FROM ohlcv WHERE symbol = ? AND timeframe = ?",
            (symbol, tf)
        ).fetchone()[0]
        if first and last:
            first_dt = datetime.utcfromtimestamp(first / 1000).strftime("%Y-%m-%d")
            last_dt = datetime.utcfromtimestamp(last / 1000).strftime("%Y-%m-%d")
            print(f"  {tf:>4}: {count:>10,} candles  ({first_dt} to {last_dt})")
    conn.close()


if __name__ == "__main__":
    download_all()
