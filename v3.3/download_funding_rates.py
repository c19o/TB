"""
Download ALL historical BTC/USDT funding rates.
Sources:
  1. Binance data.binance.vision (monthly CSVs) - 2020-01 to latest complete month
  2. Bitget API (paginated) - fills in remaining data
Also downloads open interest from Bitget.
Saves to funding_rates.db with deduplication on ts_unix.
"""
import os
import sqlite3
import time
import io
import zipfile
import requests
from datetime import datetime, timezone

_V1 = os.environ.get("SAVAGE22_V1_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(_V1, "funding_rates.db")
BINANCE_DATA_URL = "https://data.binance.vision/data/futures/um/monthly/fundingRate/BTCUSDT"
BITGET_FR_URL = "https://api.bitget.com/api/v2/mix/market/history-fund-rate"

# ---- DB Setup ----
conn = sqlite3.connect(DB_PATH)
conn.execute("""
    CREATE TABLE IF NOT EXISTS funding_rates (
        timestamp TEXT,
        ts_unix INTEGER PRIMARY KEY,
        symbol TEXT DEFAULT 'BTCUSDT',
        funding_rate REAL
    )
""")
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

before_count = conn.execute("SELECT COUNT(*) FROM funding_rates").fetchone()[0]
print(f"Funding rates in DB before download: {before_count}")

# ============================================================
# PART 1: Download from Binance data.binance.vision (monthly CSVs)
# ============================================================
print("\n" + "=" * 60)
print("PART 1: Binance historical data (data.binance.vision)")
print("=" * 60)

total_binance = 0
# Data available from 2020-01 onward. Go through current year+month.
now = datetime.now(timezone.utc)

for year in range(2020, now.year + 1):
    for month in range(1, 13):
        if year == now.year and month >= now.month:
            break  # Current month not complete yet

        url = f"{BINANCE_DATA_URL}/BTCUSDT-fundingRate-{year}-{month:02d}.zip"
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
        except Exception as e:
            print(f"  {year}-{month:02d}: error - {e}")
            continue

        try:
            z = zipfile.ZipFile(io.BytesIO(resp.content))
            with z.open(z.namelist()[0]) as f:
                content = f.read().decode("utf-8")
        except Exception as e:
            print(f"  {year}-{month:02d}: zip error - {e}")
            continue

        lines = content.strip().split("\n")
        # Skip header if present
        start_idx = 0
        if lines and not lines[0][0].isdigit():
            start_idx = 1

        inserted = 0
        for line in lines[start_idx:]:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            ts_ms = int(parts[0])
            ts_unix = ts_ms // 1000
            rate = float(parts[2])
            dt_str = datetime.fromtimestamp(ts_unix, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO funding_rates (timestamp, ts_unix, symbol, funding_rate) VALUES (?,?,?,?)",
                    (dt_str, ts_unix, "BTCUSDT", rate)
                )
                inserted += 1
            except Exception:
                pass

        conn.commit()
        total_binance += inserted
        if month in (1, 6, 12) or year >= 2025:
            print(f"  {year}-{month:02d}: {inserted} records")

print(f"Binance total: {total_binance} records processed")

# ============================================================
# PART 2: Download from Bitget API (fills gaps + recent data)
# ============================================================
print("\n" + "=" * 60)
print("PART 2: Bitget API (all available history)")
print("=" * 60)

total_bitget = 0
page = 1
max_pages = 60  # Bitget has ~53 pages

while page <= max_pages:
    try:
        resp = requests.get(BITGET_FR_URL, params={
            "symbol": "BTCUSDT",
            "productType": "USDT-FUTURES",
            "pageSize": "100",
            "pageNo": str(page),
        }, timeout=15)
        data = resp.json()
    except Exception as e:
        print(f"  Page {page} error: {e}")
        break

    if data.get("code") != "00000":
        print(f"  API error on page {page}: {data.get('msg')}")
        break

    records = data.get("data", [])
    if not records:
        print(f"  No more data at page {page}.")
        break

    inserted = 0
    for rec in records:
        ts_ms = int(rec["fundingTime"])
        ts_unix = ts_ms // 1000
        rate = float(rec["fundingRate"])
        dt_str = datetime.fromtimestamp(ts_unix, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        try:
            conn.execute(
                "INSERT OR IGNORE INTO funding_rates (timestamp, ts_unix, symbol, funding_rate) VALUES (?,?,?,?)",
                (dt_str, ts_unix, "BTCUSDT", rate)
            )
            inserted += 1
        except Exception:
            pass

    conn.commit()
    total_bitget += inserted

    if page % 10 == 1:
        oldest = min(int(r["fundingTime"]) for r in records)
        oldest_dt = datetime.fromtimestamp(oldest // 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        print(f"  Page {page:>3}: {len(records)} records, {inserted} new | oldest: {oldest_dt}")

    if len(records) < 100:
        print(f"  Reached last page at {page}.")
        break

    page += 1
    time.sleep(0.3)

print(f"Bitget total new: {total_bitget} records")

# ============================================================
# PART 3: Verify and report
# ============================================================
print("\n" + "=" * 60)
print("FUNDING RATES SUMMARY")
print("=" * 60)
row = conn.execute("SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM funding_rates").fetchone()
print(f"Total records: {row[0]}")
print(f"Date range: {row[1]} -> {row[2]}")

# Check coverage gaps
print("\nYearly coverage:")
for year in range(2019, now.year + 1):
    start_ts = int(datetime(year, 1, 1, tzinfo=timezone.utc).timestamp())
    end_ts = int(datetime(year + 1, 1, 1, tzinfo=timezone.utc).timestamp())
    cnt = conn.execute("SELECT COUNT(*) FROM funding_rates WHERE ts_unix >= ? AND ts_unix < ?",
                       (start_ts, end_ts)).fetchone()[0]
    # Expected: 3 per day * 365 = ~1095
    print(f"  {year}: {cnt} records")

# ============================================================
# PART 4: Open Interest from Bitget
# ============================================================
print("\n" + "=" * 60)
print("PART 4: Open Interest from Bitget")
print("=" * 60)

BITGET_OI_URL = "https://api.bitget.com/api/v2/mix/market/open-interest"

oi_before = conn.execute("SELECT COUNT(*) FROM open_interest").fetchone()[0]
print(f"Open interest rows before: {oi_before}")

# Bitget only provides current open interest, not historical
# Let's get current OI and store it
try:
    resp = requests.get(BITGET_OI_URL, params={
        "symbol": "BTCUSDT",
        "productType": "USDT-FUTURES",
    }, timeout=15)
    data = resp.json()
    if data.get("code") == "00000" and data.get("data"):
        rec = data["data"]
        ts_unix = int(time.time())
        oi_val = float(rec.get("amount", 0))
        oi_val_usd = float(rec.get("usdtAmount", 0))
        dt_str = datetime.fromtimestamp(ts_unix, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        conn.execute(
            "INSERT OR REPLACE INTO open_interest (timestamp, ts_unix, symbol, sum_open_interest, sum_open_interest_value) VALUES (?,?,?,?,?)",
            (dt_str, ts_unix, "BTCUSDT", oi_val, oi_val_usd)
        )
        conn.commit()
        print(f"  Current OI: {oi_val:.2f} BTC / ${oi_val_usd:.2f}")
    else:
        print(f"  OI API response: {data.get('msg', 'unknown')}")
except Exception as e:
    print(f"  OI error: {e}")

# Try to get historical OI from Binance data.binance.vision
print("\nAttempting historical OI from Binance data.binance.vision...")
BINANCE_OI_DATA_URL = "https://data.binance.vision/data/futures/um/monthly/metrics/BTCUSDT"

oi_total = 0
for year in range(2020, now.year + 1):
    for month in range(1, 13):
        if year == now.year and month >= now.month:
            break
        url = f"{BINANCE_OI_DATA_URL}/BTCUSDT-metrics-{year}-{month:02d}.zip"
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
        except Exception:
            continue

        try:
            z = zipfile.ZipFile(io.BytesIO(resp.content))
            with z.open(z.namelist()[0]) as f:
                content = f.read().decode("utf-8")
        except Exception:
            continue

        lines = content.strip().split("\n")
        start_idx = 0
        if lines and not lines[0][0].isdigit():
            start_idx = 1
            header = lines[0].split(",")
            # Find column indices
            # Expected: create_time,symbol,sum_open_interest,sum_open_interest_value,count_toptrader_long_short_ratio,...

        inserted = 0
        for line in lines[start_idx:]:
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            try:
                ts_ms = int(parts[0])
                ts_unix = ts_ms // 1000
                oi_val = float(parts[2]) if parts[2] else 0
                oi_val_usd = float(parts[3]) if parts[3] else 0
                dt_str = datetime.fromtimestamp(ts_unix, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                conn.execute(
                    "INSERT OR REPLACE INTO open_interest (timestamp, ts_unix, symbol, sum_open_interest, sum_open_interest_value) VALUES (?,?,?,?,?)",
                    (dt_str, ts_unix, "BTCUSDT", oi_val, oi_val_usd)
                )
                inserted += 1
            except (ValueError, IndexError):
                continue

        conn.commit()
        oi_total += inserted
        if month in (1, 6, 12) or year >= 2025:
            print(f"  {year}-{month:02d}: {inserted} OI records")

print(f"Historical OI total: {oi_total} records")

oi_row = conn.execute("SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM open_interest").fetchone()
print(f"OI in DB: {oi_row[0]} records | {oi_row[1]} -> {oi_row[2]}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
fr_row = conn.execute("SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM funding_rates").fetchone()
oi_row = conn.execute("SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM open_interest").fetchone()
print(f"Funding Rates: {fr_row[0]} rows | {fr_row[1]} -> {fr_row[2]}")
print(f"Open Interest: {oi_row[0]} rows | {oi_row[1]} -> {oi_row[2]}")

conn.close()
print(f"\nData saved to: {DB_PATH}")
