"""
Download historical BTC/USDT open interest from Binance data.binance.vision.
Daily CSV files with 5-minute data. We keep only every 4-hour snapshot.
Data available from ~2020-09 to present.
Stores in funding_rates.db -> open_interest table.
"""
import os
import sqlite3
import io
import zipfile
import requests
import time
from datetime import datetime, timezone, timedelta

_V1 = os.environ.get("SAVAGE22_V1_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(_V1, "funding_rates.db")
BASE_URL = "https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT"

conn = sqlite3.connect(DB_PATH)
conn.execute("""
    CREATE TABLE IF NOT EXISTS open_interest (
        timestamp TEXT,
        ts_unix INTEGER PRIMARY KEY,
        symbol TEXT DEFAULT 'BTCUSDT',
        sum_open_interest REAL,
        sum_open_interest_value REAL
    )
""")
conn.commit()

before = conn.execute("SELECT COUNT(*) FROM open_interest").fetchone()[0]
print(f"OI rows before: {before}")

# Download from 2020-09-15 to yesterday
start_date = datetime(2020, 9, 15, tzinfo=timezone.utc)
end_date = datetime.now(timezone.utc) - timedelta(days=1)

current = start_date
total_inserted = 0
total_days = 0
errors = 0
consecutive_errors = 0

while current <= end_date:
    date_str = current.strftime("%Y-%m-%d")
    url = f"{BASE_URL}/BTCUSDT-metrics-{date_str}.zip"

    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 404:
            current += timedelta(days=1)
            consecutive_errors = 0
            continue
        resp.raise_for_status()
        consecutive_errors = 0
    except Exception as e:
        errors += 1
        consecutive_errors += 1
        if consecutive_errors > 10:
            print(f"  Too many consecutive errors at {date_str}, skipping 30 days")
            current += timedelta(days=30)
            consecutive_errors = 0
            continue
        current += timedelta(days=1)
        continue

    try:
        z = zipfile.ZipFile(io.BytesIO(resp.content))
        with z.open(z.namelist()[0]) as f:
            content = f.read().decode("utf-8")
    except Exception:
        current += timedelta(days=1)
        continue

    lines = content.strip().split("\n")
    # Skip header
    start_idx = 1 if lines and not lines[0][0].isdigit() else 0

    # Keep only every 4 hours (every 48th row at 5-min intervals)
    # Or more precisely, keep rows at 00:00, 04:00, 08:00, 12:00, 16:00, 20:00
    inserted = 0
    for line in lines[start_idx:]:
        parts = line.strip().split(",")
        if len(parts) < 4:
            continue
        try:
            # Format: 2024-01-15 00:00:00
            dt = parts[0]
            # Only keep on-the-hour entries at 4h intervals
            if len(dt) >= 19:
                hour = int(dt[11:13])
                minute = int(dt[14:16])
                if minute != 0 or hour % 4 != 0:
                    continue

            ts_unix = int(datetime.strptime(dt, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())
            oi_val = float(parts[2]) if parts[2] else 0
            oi_val_usd = float(parts[3]) if parts[3] else 0

            conn.execute(
                "INSERT OR REPLACE INTO open_interest (timestamp, ts_unix, symbol, sum_open_interest, sum_open_interest_value) VALUES (?,?,?,?,?)",
                (dt, ts_unix, "BTCUSDT", oi_val, oi_val_usd)
            )
            inserted += 1
        except (ValueError, IndexError):
            continue

    conn.commit()
    total_inserted += inserted
    total_days += 1

    if total_days % 100 == 0:
        print(f"  {date_str}: {total_days} days processed, {total_inserted} OI records total")

    current += timedelta(days=1)
    # Gentle rate limiting
    if total_days % 5 == 0:
        time.sleep(0.1)

print(f"\nDownload complete.")
print(f"  Days processed: {total_days}")
print(f"  OI records inserted: {total_inserted}")
print(f"  Errors: {errors}")

row = conn.execute("SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM open_interest").fetchone()
print(f"  DB totals: {row[0]} records | {row[1]} -> {row[2]}")

conn.close()
print(f"\nData saved to: {DB_PATH}")
