"""
Download all free data sources:
1. Fear & Greed Index -> fear_greed.db
2. BTC candles (full columns) -> btc_prices.db (update NULLs, symbol='BTC/USDT', timeframe='1d')
3. Ephemeris cache (2009-2026) -> ephemeris_cache.db
4. Hebrew calendar dates -> hebrew_calendar.json
"""

import sqlite3
import json
import time
import math
import urllib.request
import urllib.error
import os
import sys
from datetime import datetime, date, timedelta, timezone

BASE_DIR = os.environ.get("SAVAGE22_V1_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. FEAR & GREED INDEX
# ─────────────────────────────────────────────────────────────────────────────

def download_fear_greed():
    log("=== FEAR & GREED INDEX ===")
    url = "https://api.alternative.me/fng/?limit=0&format=json"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        log(f"  ERROR fetching Fear & Greed: {e}")
        return 0

    records = data.get("data", [])
    log(f"  Received {len(records)} records")

    db_path = os.path.join(BASE_DIR, "fear_greed.db")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS fear_greed (
            date TEXT PRIMARY KEY,
            value INTEGER,
            classification TEXT
        )
    """)

    inserted = 0
    for r in records:
        ts = int(r["timestamp"])
        dt = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        val = int(r["value"])
        cls = r["value_classification"]
        c.execute("INSERT OR REPLACE INTO fear_greed VALUES (?,?,?)", (dt, val, cls))
        inserted += 1

    conn.commit()
    conn.close()
    log(f"  Saved {inserted} rows to fear_greed.db")
    return inserted


# ─────────────────────────────────────────────────────────────────────────────
# 2. BTC CANDLES – fill NULLs in ohlcv table
# ─────────────────────────────────────────────────────────────────────────────

def fetch_binance_klines(symbol="BTCUSDT", interval="1d", start_ms=None, limit=1000):
    """Fetch klines from Binance.US public API."""
    base = "https://api.binance.us/api/v3/klines"
    url = f"{base}?symbol={symbol}&interval={interval}&limit={limit}"
    if start_ms:
        url += f"&startTime={start_ms}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())

def download_btc_candles():
    log("=== BTC CANDLES (Binance.US) ===")
    db_path = os.path.join(BASE_DIR, "btc_prices.db")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Verify table exists with correct schema
    c.execute("PRAGMA table_info(ohlcv)")
    cols = {row[1] for row in c.fetchall()}
    log(f"  Table: ohlcv, columns: {sorted(cols)}")

    # Check current state
    c.execute("SELECT MIN(open_time), MAX(open_time), COUNT(*) FROM ohlcv WHERE symbol='BTC/USDT' AND timeframe='1d'")
    row = c.fetchone()
    log(f"  DB range: open_time {row[0]} to {row[1]}, {row[2]} rows")

    c.execute("SELECT COUNT(*) FROM ohlcv WHERE symbol='BTC/USDT' AND timeframe='1d' AND (trades IS NULL OR taker_buy_volume IS NULL)")
    null_count = c.fetchone()[0]
    log(f"  Rows with NULL trades or taker_buy_volume: {null_count}")
    conn.close()

    # Download ALL history in chunks of 1000 days from Binance.US
    # Binance.US launched Sep 2019; try from 2017-08-17 (Binance exchange launch)
    start_dt = datetime(2017, 8, 17, tzinfo=timezone.utc)
    end_dt   = datetime.now(timezone.utc)

    all_candles = []
    current = start_dt
    batch = 0
    while current < end_dt:
        start_ms = int(current.timestamp() * 1000)
        try:
            klines = fetch_binance_klines("BTCUSDT", "1d", start_ms=start_ms, limit=1000)
            if not klines:
                break
            all_candles.extend(klines)
            last_close_ms = klines[-1][6]
            current = datetime.fromtimestamp(last_close_ms / 1000, tz=timezone.utc) + timedelta(milliseconds=1)
            batch += 1
            log(f"  Batch {batch}: {len(klines)} candles, up to {current.strftime('%Y-%m-%d')}")
            if len(klines) < 1000:
                break
            time.sleep(0.2)
        except Exception as e:
            log(f"  ERROR at batch {batch}: {e}")
            break

    log(f"  Total candles downloaded: {len(all_candles)}")
    if not all_candles:
        return 0

    # The ohlcv table uses open_time (ms integer) as key, symbol='BTC/USDT', timeframe='1d'
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Build lookup: open_time -> candle
    candle_map = {}
    for k in all_candles:
        open_time_ms = int(k[0])
        candle_map[open_time_ms] = k

    # Get all existing open_times for BTC/USDT 1d
    c.execute("SELECT open_time FROM ohlcv WHERE symbol='BTC/USDT' AND timeframe='1d'")
    existing_times = {row[0] for row in c.fetchall()}
    log(f"  Existing open_times in DB: {len(existing_times)}")

    updated = 0
    inserted = 0

    for open_time_ms, k in candle_map.items():
        o       = float(k[1])
        h       = float(k[2])
        l       = float(k[3])
        cl      = float(k[4])
        vol     = float(k[5])
        close_time = int(k[6])
        quote_vol  = float(k[7])
        trades     = int(k[8])
        tbbase     = float(k[9])   # taker_buy_volume (base asset)
        tbquote    = float(k[10])  # taker_buy_quote

        if open_time_ms in existing_times:
            c.execute("""
                UPDATE ohlcv SET
                    open=?, high=?, low=?, close=?, volume=?,
                    quote_volume=?, trades=?, taker_buy_volume=?,
                    taker_buy_quote=?, close_time=?
                WHERE symbol='BTC/USDT' AND timeframe='1d' AND open_time=?
            """, (o, h, l, cl, vol, quote_vol, trades, tbbase, tbquote, close_time, open_time_ms))
            updated += 1
        else:
            c.execute("""
                INSERT INTO ohlcv
                    (symbol, timeframe, open_time, open, high, low, close, volume,
                     quote_volume, trades, taker_buy_volume, taker_buy_quote, close_time)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, ('BTC/USDT', '1d', open_time_ms, o, h, l, cl, vol,
                  quote_vol, trades, tbbase, tbquote, close_time))
            inserted += 1

    conn.commit()

    # Verify
    c.execute("SELECT COUNT(*) FROM ohlcv WHERE symbol='BTC/USDT' AND timeframe='1d' AND (trades IS NULL OR taker_buy_volume IS NULL)")
    remaining_nulls = c.fetchone()[0]
    c.execute("SELECT MIN(open_time), MAX(open_time), COUNT(*) FROM ohlcv WHERE symbol='BTC/USDT' AND timeframe='1d'")
    final = c.fetchone()
    conn.close()

    log(f"  Updated {updated} rows, inserted {inserted} new rows")
    log(f"  Remaining NULLs: {remaining_nulls}")
    log(f"  Final range: {final[0]} to {final[1]}, {final[2]} total rows")
    return updated + inserted


# ─────────────────────────────────────────────────────────────────────────────
# 3. EPHEMERIS CACHE (2009-2026)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ephemeris():
    log("=== EPHEMERIS CACHE (2009–2026) ===")
    try:
        import ephem
    except ImportError:
        log("  ERROR: ephem not installed in this Python. Try: pip install ephem")
        return 0

    db_path = os.path.join(BASE_DIR, "ephemeris_cache.db")
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS ephemeris (
            date TEXT PRIMARY KEY,
            sun_lon REAL,
            moon_lon REAL,
            mercury_lon REAL,
            venus_lon REAL,
            mars_lon REAL,
            jupiter_lon REAL,
            saturn_lon REAL,
            uranus_lon REAL,
            neptune_lon REAL,
            pluto_lon REAL,
            moon_phase REAL,
            moon_mansion INTEGER,
            mercury_retrograde INTEGER,
            hard_aspects INTEGER,
            soft_aspects INTEGER,
            planetary_strength REAL,
            day_count INTEGER,
            digital_root INTEGER
        )
    """)
    conn.commit()

    c.execute("SELECT COUNT(*) FROM ephemeris")
    existing_count = c.fetchone()[0]
    log(f"  Existing cache rows: {existing_count}")

    # Collect all dates already done
    c.execute("SELECT date FROM ephemeris")
    cached_dates = {row[0] for row in c.fetchall()}

    HARD_ANGLES = [90, 180, 270]
    SOFT_ANGLES = [60, 120]
    ORBS = 8

    def angle_diff(a, b):
        d = abs(a - b) % 360
        return min(d, 360 - d)

    def count_aspects(lons):
        hard = soft = 0
        n = len(lons)
        for i in range(n):
            for j in range(i + 1, n):
                diff = angle_diff(lons[i], lons[j])
                matched_hard = any(abs(diff - ang) <= ORBS for ang in HARD_ANGLES)
                matched_soft = any(abs(diff - ang) <= ORBS for ang in SOFT_ANGLES)
                if matched_hard:
                    hard += 1
                elif matched_soft:
                    soft += 1
        return hard, soft

    def digital_root(n):
        if n <= 0:
            return 0
        return 1 + (n - 1) % 9

    def planetary_strength_index(hard, soft, moon_phase):
        phase_factor = abs(math.cos(math.radians(moon_phase))) + 1
        return round((soft * 2 + hard * 0.5) * phase_factor, 4)

    genesis = date(2009, 1, 3)
    start   = date(2009, 1, 1)
    end     = date(2026, 3, 17)

    total_days = (end - start).days + 1
    done = 0
    rows_buffer = []
    batch_size = 500

    current = start
    while current <= end:
        dt_str = current.strftime("%Y-%m-%d")

        if dt_str not in cached_dates:
            ephem_date = ephem.Date(current.strftime("%Y/%m/%d"))

            planet_objects = [
                ephem.Sun(), ephem.Moon(), ephem.Mercury(),
                ephem.Venus(), ephem.Mars(), ephem.Jupiter(),
                ephem.Saturn(), ephem.Uranus(), ephem.Neptune(), ephem.Pluto(),
            ]
            lons = []
            for body in planet_objects:
                body.compute(ephem_date)
                lon = math.degrees(body.hlong) % 360
                lons.append(lon)

            # Mercury retrograde: compare longitude to next day
            merc_today = ephem.Mercury()
            merc_today.compute(ephem_date)
            merc_next = ephem.Mercury()
            merc_next.compute(ephem.Date(ephem_date + 1))
            lon_today = math.degrees(merc_today.hlong) % 360
            lon_next  = math.degrees(merc_next.hlong) % 360
            # Retrograde if longitude decreases (accounting for 359->0 wrap)
            delta = (lon_next - lon_today + 360) % 360
            merc_retro = 1 if delta > 180 else 0  # >180 means it went backwards

            # Moon phase angle (distance from Sun in ecliptic longitude)
            moon_phase_angle = (lons[1] - lons[0] + 360) % 360

            # Moon mansion (27 nakshatras)
            moon_mansion = int(lons[1] / (360.0 / 27))

            hard, soft = count_aspects(lons)
            strength = planetary_strength_index(hard, soft, moon_phase_angle)
            day_count = (current - genesis).days
            dr = digital_root(max(day_count, 1))

            rows_buffer.append((
                dt_str,
                round(lons[0], 4),   # sun
                round(lons[1], 4),   # moon
                round(lons[2], 4),   # mercury
                round(lons[3], 4),   # venus
                round(lons[4], 4),   # mars
                round(lons[5], 4),   # jupiter
                round(lons[6], 4),   # saturn
                round(lons[7], 4),   # uranus
                round(lons[8], 4),   # neptune
                round(lons[9], 4),   # pluto
                round(moon_phase_angle, 4),
                moon_mansion,
                merc_retro,
                hard,
                soft,
                strength,
                day_count,
                dr
            ))

        done += 1
        if done % 500 == 0:
            log(f"  Computed {done}/{total_days} days ({current})")

        if len(rows_buffer) >= batch_size:
            c.executemany("""
                INSERT OR REPLACE INTO ephemeris VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, rows_buffer)
            conn.commit()
            rows_buffer = []

        current += timedelta(days=1)

    if rows_buffer:
        c.executemany("""
            INSERT OR REPLACE INTO ephemeris VALUES
            (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, rows_buffer)
        conn.commit()

    c.execute("SELECT COUNT(*) FROM ephemeris")
    total_rows = c.fetchone()[0]

    # Show sample
    c.execute("SELECT date, sun_lon, moon_lon, mercury_retrograde, moon_mansion, hard_aspects, digital_root FROM ephemeris ORDER BY date DESC LIMIT 3")
    log("  Sample (recent):")
    for row in c.fetchall():
        log(f"    {row}")

    conn.close()
    log(f"  Ephemeris cache complete: {total_rows} rows")
    return total_rows


# ─────────────────────────────────────────────────────────────────────────────
# 4. HEBREW CALENDAR DATES
# ─────────────────────────────────────────────────────────────────────────────

def compute_hebrew_calendar():
    log("=== HEBREW CALENDAR DATES (2019–2026) ===")

    hebrew_holidays = {
        2019: {
            "year_name": "5779-5780",
            "rosh_hashanah": "2019-09-29",
            "yom_kippur": "2019-10-08",
            "sukkot_start": "2019-10-13",
            "hanukkah_start": "2019-12-22",
            "passover_start": "2020-04-08",
            "shavuot": "2020-05-28",
            "shmita": False,
        },
        2020: {
            "year_name": "5780-5781",
            "rosh_hashanah": "2020-09-18",
            "yom_kippur": "2020-09-27",
            "sukkot_start": "2020-10-02",
            "hanukkah_start": "2020-12-10",
            "passover_start": "2021-03-27",
            "shavuot": "2021-05-16",
            "shmita": False,
        },
        2021: {
            "year_name": "5781-5782",
            "rosh_hashanah": "2021-09-06",
            "yom_kippur": "2021-09-15",
            "sukkot_start": "2021-09-20",
            "hanukkah_start": "2021-11-28",
            "passover_start": "2022-04-15",
            "shavuot": "2022-06-04",
            "shmita": False,
            "shmita_begins": "2021-09-06",
        },
        2022: {
            "year_name": "5782-5783",
            "rosh_hashanah": "2022-09-25",
            "yom_kippur": "2022-10-04",
            "sukkot_start": "2022-10-09",
            "hanukkah_start": "2022-12-18",
            "passover_start": "2023-04-05",
            "shavuot": "2023-05-25",
            "shmita": True,
            "shmita_year": "Tishrei 5782 - Elul 5782 (Sep 6 2021 - Sep 25 2022)",
        },
        2023: {
            "year_name": "5783-5784",
            "rosh_hashanah": "2023-09-15",
            "yom_kippur": "2023-09-24",
            "sukkot_start": "2023-09-29",
            "hanukkah_start": "2023-12-07",
            "passover_start": "2024-04-22",
            "shavuot": "2024-06-11",
            "shmita": False,
        },
        2024: {
            "year_name": "5784-5785",
            "rosh_hashanah": "2024-10-02",
            "yom_kippur": "2024-10-11",
            "sukkot_start": "2024-10-16",
            "hanukkah_start": "2024-12-25",
            "passover_start": "2025-04-12",
            "shavuot": "2025-06-01",
            "shmita": False,
        },
        2025: {
            "year_name": "5785-5786",
            "rosh_hashanah": "2025-09-22",
            "yom_kippur": "2025-10-01",
            "sukkot_start": "2025-10-06",
            "hanukkah_start": "2025-12-14",
            "passover_start": "2026-04-01",
            "shavuot": "2026-05-21",
            "shmita": False,
        },
        2026: {
            "year_name": "5786-5787",
            "rosh_hashanah": "2026-09-11",
            "yom_kippur": "2026-09-20",
            "sukkot_start": "2026-09-25",
            "hanukkah_start": "2026-12-04",
            "passover_start": "2027-04-21",
            "shavuot": "2027-06-10",
            "shmita": False,
        },
    }

    shmita_info = {
        "description": "Shmita (Sabbatical Year) every 7 Hebrew years",
        "recent_shmita_years": [
            {
                "hebrew_year": "5782",
                "gregorian": "Sep 6 2021 (Rosh Hashanah 5782) to Sep 25 2022 (Erev Rosh Hashanah 5783)",
                "significance": "7th year land sabbatical; debt release; financial reset themes in numerology"
            },
            {
                "hebrew_year": "5789",
                "gregorian": "Starts ~Sep 2028",
                "significance": "Next Shmita year"
            }
        ],
        "jubilee_note": "50-year Jubilee follows 7 Shmita cycles. Last notable jubilee ~1917 or 1967."
    }

    all_dates = []
    for year, h in hebrew_holidays.items():
        for holiday in ["rosh_hashanah", "yom_kippur", "sukkot_start",
                        "hanukkah_start", "passover_start", "shavuot"]:
            dt = h.get(holiday)
            if dt:
                all_dates.append({
                    "date": dt,
                    "holiday": holiday.replace("_start", "").replace("_", " ").title(),
                    "gregorian_year": year,
                    "hebrew_year": h["year_name"],
                    "shmita_year": h.get("shmita", False)
                })

    all_dates.sort(key=lambda x: x["date"])

    output = {
        "generated": datetime.now().strftime("%Y-%m-%d"),
        "description": "Hebrew calendar key dates for crypto/numerology analysis",
        "shmita": shmita_info,
        "holidays_by_year": hebrew_holidays,
        "all_key_dates_sorted": all_dates,
    }

    out_path = os.path.join(BASE_DIR, "hebrew_calendar.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    log(f"  Saved {len(all_dates)} holiday dates to hebrew_calendar.json")
    return len(all_dates)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = {}

    log("Starting all free data downloads...")
    log("=" * 60)

    try:
        n = download_fear_greed()
        results["fear_greed"] = {"status": "OK", "rows": n}
    except Exception as e:
        log(f"Fear & Greed FAILED: {e}")
        results["fear_greed"] = {"status": "FAILED", "error": str(e)}

    try:
        n = download_btc_candles()
        results["btc_candles"] = {"status": "OK", "rows": n}
    except Exception as e:
        import traceback; traceback.print_exc()
        log(f"BTC Candles FAILED: {e}")
        results["btc_candles"] = {"status": "FAILED", "error": str(e)}

    try:
        n = compute_ephemeris()
        results["ephemeris"] = {"status": "OK", "rows": n}
    except Exception as e:
        import traceback; traceback.print_exc()
        log(f"Ephemeris FAILED: {e}")
        results["ephemeris"] = {"status": "FAILED", "error": str(e)}

    try:
        n = compute_hebrew_calendar()
        results["hebrew_calendar"] = {"status": "OK", "entries": n}
    except Exception as e:
        log(f"Hebrew Calendar FAILED: {e}")
        results["hebrew_calendar"] = {"status": "FAILED", "error": str(e)}

    log("=" * 60)
    log("FINAL RESULTS:")
    for k, v in results.items():
        log(f"  {k}: {v}")

    summary_path = os.path.join(BASE_DIR, "data_download_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"run_time": datetime.now().isoformat(), "results": results}, f, indent=2)
    log(f"Summary saved to data_download_summary.json")
