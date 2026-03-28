"""Backfill crypto 1h data from 2017-08-17 to 2024-03-21 for all 14 crypto assets.
Then derive 4h from 1h. Uses multi_asset_prices.db."""
import os, sys, io, time, sqlite3
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ccxt
from datetime import datetime, timezone

MULTI_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "multi_asset_prices.db")

CRYPTO_ASSETS = [
    "ETH/USDT", "XRP/USDT", "LTC/USDT", "SOL/USDT", "DOGE/USDT",
    "ADA/USDT", "BNB/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT",
    "MATIC/USDT", "UNI/USDT", "AAVE/USDT"
]

# Per-asset earliest Binance listing dates (approximate)
ASSET_START = {
    "ETH/USDT": "2017-08-17", "XRP/USDT": "2018-01-01", "LTC/USDT": "2017-12-01",
    "SOL/USDT": "2020-08-11", "DOGE/USDT": "2019-07-05", "ADA/USDT": "2018-04-17",
    "BNB/USDT": "2017-11-06", "AVAX/USDT": "2020-09-22", "LINK/USDT": "2019-01-16",
    "DOT/USDT": "2020-08-19", "MATIC/USDT": "2019-04-26", "UNI/USDT": "2020-09-17",
    "AAVE/USDT": "2020-10-13",
}

def main():
    conn = sqlite3.connect(MULTI_DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS ohlcv (
        symbol TEXT NOT NULL, timeframe TEXT NOT NULL, open_time INTEGER NOT NULL,
        open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL,
        volume REAL NOT NULL, UNIQUE(symbol, timeframe, open_time))""")

    exchange = ccxt.binance({"enableRateLimit": True})
    exchange.load_markets()
    print("Connected to Binance global")

    end_str = "2024-03-21T00:00:00Z"
    end_ms = exchange.parse8601(end_str)

    for symbol in CRYPTO_ASSETS:
        start_date = ASSET_START.get(symbol, "2018-01-01")
        since_ms = exchange.parse8601(f"{start_date}T00:00:00Z")

        # Check existing data
        row = conn.execute("SELECT MIN(open_time) FROM ohlcv WHERE symbol=? AND timeframe='1h'",
                           (symbol,)).fetchone()
        if row[0] and row[0] <= since_ms:
            print(f"{symbol}: already has 1h from {datetime.fromtimestamp(row[0]/1000, tz=timezone.utc).strftime('%Y-%m-%d')} -- skipping")
            continue

        # If we have data starting at some point, only backfill up to that point
        actual_end = end_ms
        if row[0]:
            actual_end = row[0]
            print(f"{symbol}: backfilling 1h from {start_date} to {datetime.fromtimestamp(actual_end/1000, tz=timezone.utc).strftime('%Y-%m-%d')}")
        else:
            print(f"{symbol}: downloading 1h from {start_date} to 2024-03-21")

        total = 0
        batch = 0
        cur_ms = since_ms
        while cur_ms < actual_end:
            try:
                candles = exchange.fetch_ohlcv(symbol, "1h", since=cur_ms, limit=1000)
                if not candles:
                    break
                for c in candles:
                    if c[0] >= actual_end:
                        break
                    conn.execute("""INSERT OR IGNORE INTO ohlcv
                        (symbol, timeframe, open_time, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (symbol, "1h", c[0], c[1], c[2], c[3], c[4], c[5]))
                conn.commit()
                total += len(candles)
                cur_ms = candles[-1][0] + 1
                batch += 1
                if batch % 20 == 0:
                    dt = datetime.fromtimestamp(cur_ms/1000, tz=timezone.utc)
                    print(f"  {symbol}: {total:,d} candles | at {dt.strftime('%Y-%m-%d %H:%M')}")
            except Exception as e:
                print(f"  Error {symbol}: {e}, retrying in 5s...")
                time.sleep(5)

        print(f"  {symbol}: {total:,d} 1h candles backfilled")

    # Now derive 4h from 1h for all crypto assets
    print("\n=== Deriving 4h from 1h ===")
    for symbol in CRYPTO_ASSETS:
        rows = conn.execute("""SELECT open_time, open, high, low, close, volume
            FROM ohlcv WHERE symbol=? AND timeframe='1h' ORDER BY open_time""",
            (symbol,)).fetchall()
        if not rows:
            continue

        count_4h = 0
        i = 0
        while i < len(rows):
            ts = rows[i][0]
            # Floor to 4h boundary (0,4,8,12,16,20 UTC)
            dt = datetime.fromtimestamp(ts/1000, tz=timezone.utc)
            hour_block = (dt.hour // 4) * 4
            block_start_hour = hour_block

            # Collect up to 4 bars in this block
            block = []
            while i < len(rows):
                t = rows[i][0]
                d = datetime.fromtimestamp(t/1000, tz=timezone.utc)
                if (d.hour // 4) * 4 == block_start_hour and d.date() == dt.date():
                    block.append(rows[i])
                    i += 1
                else:
                    break

            if len(block) == 4:  # Only complete 4h bars
                o = block[0][1]
                h = max(b[2] for b in block)
                l = min(b[3] for b in block)
                c = block[-1][4]
                v = sum(b[5] for b in block)
                bar_ts = block[0][0]
                conn.execute("""INSERT OR IGNORE INTO ohlcv
                    (symbol, timeframe, open_time, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (symbol, "4h", bar_ts, o, h, l, c, v))
                count_4h += 1

        conn.commit()
        print(f"  {symbol}: {count_4h:,d} 4h bars derived")

    conn.close()
    print("\nAll multi-asset backfill complete!")

if __name__ == "__main__":
    main()
