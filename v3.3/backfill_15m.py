"""Backfill BTC/USDT 15m data from 2017-11-01 to 2019-09-23 (the gap)."""
import os, sys, io, time, sqlite3
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import BTC_DB
import ccxt

def main():
    conn = sqlite3.connect(BTC_DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS ohlcv (
        symbol TEXT NOT NULL, timeframe TEXT NOT NULL, open_time INTEGER NOT NULL,
        open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL,
        volume REAL NOT NULL, UNIQUE(symbol, timeframe, open_time))""")

    exchange = ccxt.binance({"enableRateLimit": True})
    exchange.load_markets()
    print("Connected to Binance global")

    symbol = "BTC/USDT"
    tf = "15m"
    start = "2017-11-01T00:00:00Z"
    end = "2019-09-23T08:30:00Z"  # Where existing data starts

    since_ms = exchange.parse8601(start)
    end_ms = exchange.parse8601(end)
    total = 0
    batch = 0

    print(f"Backfilling {symbol} {tf}: 2017-11-01 to 2019-09-23")
    while since_ms < end_ms:
        try:
            candles = exchange.fetch_ohlcv(symbol, tf, since=since_ms, limit=1000)
            if not candles:
                break
            for c in candles:
                if c[0] >= end_ms:
                    break
                conn.execute("""INSERT OR IGNORE INTO ohlcv
                    (symbol, timeframe, open_time, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (symbol, tf, c[0], c[1], c[2], c[3], c[4], c[5]))
            conn.commit()
            total += len(candles)
            since_ms = candles[-1][0] + 1
            batch += 1
            if batch % 10 == 0:
                from datetime import datetime, timezone
                dt = datetime.fromtimestamp(since_ms/1000, tz=timezone.utc)
                print(f"  {total:,d} candles | at {dt.strftime('%Y-%m-%d %H:%M')}")
        except Exception as e:
            print(f"  Error: {e}, retrying in 5s...")
            time.sleep(5)

    conn.close()
    print(f"\nDone: {total:,d} candles backfilled for {symbol} {tf}")

    # Verify
    conn = sqlite3.connect(BTC_DB)
    row = conn.execute("""SELECT COUNT(*), MIN(datetime(open_time/1000, 'unixepoch')),
        MAX(datetime(open_time/1000, 'unixepoch')) FROM ohlcv
        WHERE symbol=? AND timeframe=?""", (symbol, tf)).fetchone()
    print(f"Total 15m: {row[0]:,d} rows | {row[1]} to {row[2]}")
    conn.close()

if __name__ == "__main__":
    main()
