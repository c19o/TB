#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
space_weather_streamer.py -- Continuous space weather data poller
=================================================================
Polls every 15 minutes from NOAA SWPC APIs:
  - Kp index (1-min resolution)
  - Solar wind plasma (speed, density, temperature)
  - Solar wind magnetics (Bz, Bt)
  - NOAA scales (R, S, G)
  - Solar flares (class B/C/M/X)
  - 7-day Kp index

Stores in space_weather.db (SQLite).

Run as: python space_weather_streamer.py
        python space_weather_streamer.py --once
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
try:
    from config import DB_DIR as _DB_DIR
except ImportError:
    _DB_DIR = PROJECT_DIR

# Setup logging — logs go to script dir, DB goes to canonical DB_DIR
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_DIR, 'space_weather_streamer.log'), encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

# Poll interval (seconds)
POLL_INTERVAL = 900  # 15 minutes

DB_PATH = os.path.join(_DB_DIR, 'space_weather.db')

# NOAA SWPC API endpoints
KP_1MIN_URL = "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"
SOLAR_WIND_PLASMA_URL = "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json"
SOLAR_WIND_MAG_URL = "https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json"
XRAY_FLARES_URL = "https://services.swpc.noaa.gov/json/goes/primary/xray-flares-7-day.json"
NOAA_SCALES_URL = "https://services.swpc.noaa.gov/products/noaa-scales.json"
KP_7DAY_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
F107_URL = "https://services.swpc.noaa.gov/json/f107_cm_flux.json"
SUNSPOT_URL = "https://www.sidc.be/SILSO/DATA/EISN/EISN_current.csv"


# ============================================================
# HTTP helpers
# ============================================================

def http_get(url, json_response=True, timeout=15):
    """Fetch URL using requests if available, else urllib."""
    try:
        if requests:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.json() if json_response else resp.text.strip()
        else:
            req = urllib.request.Request(url, headers={"User-Agent": "space_weather_streamer/1.0"})
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
    """Create space_weather.db and tables if not exists."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS space_weather (
            timestamp INTEGER PRIMARY KEY,
            kp_index REAL,
            estimated_kp REAL,
            solar_wind_speed REAL,
            solar_wind_density REAL,
            solar_wind_temp REAL,
            solar_wind_bz REAL,
            solar_wind_bt REAL,
            r_scale INTEGER,
            s_scale INTEGER,
            g_scale INTEGER,
            sunspot_number REAL,
            solar_flux_f107 REAL,
            inserted_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS solar_flares (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            begin_time TEXT,
            max_time TEXT,
            end_time TEXT,
            max_class TEXT,
            max_xrlong REAL,
            ts_unix INTEGER,
            inserted_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(begin_time, max_class)
        )
    """)
    conn.commit()
    return conn


def ensure_new_columns(conn):
    """Add sunspot_number and solar_flux_f107 columns to existing tables."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(space_weather)").fetchall()}
    if 'sunspot_number' not in existing:
        conn.execute("ALTER TABLE space_weather ADD COLUMN sunspot_number REAL")
        log.info("  Added sunspot_number column to space_weather table")
    if 'solar_flux_f107' not in existing:
        conn.execute("ALTER TABLE space_weather ADD COLUMN solar_flux_f107 REAL")
        log.info("  Added solar_flux_f107 column to space_weather table")
    conn.commit()


def get_last_timestamp(conn):
    """Get the most recent timestamp in the space_weather table."""
    row = conn.execute(
        "SELECT MAX(timestamp) FROM space_weather"
    ).fetchone()
    return row[0] if row and row[0] else None


# ============================================================
# Parsing helpers
# ============================================================

def parse_noaa_time(time_tag):
    """Parse a NOAA time_tag string to unix timestamp (seconds)."""
    if not time_tag:
        return None
    try:
        # Handle both formats: "2024-01-15 12:30:00.000" and "2024-01-15T12:30:00Z"
        time_tag = time_tag.replace("T", " ").replace("Z", "").strip()
        # Strip fractional seconds if present
        if "." in time_tag:
            time_tag = time_tag[:time_tag.index(".")]
        dt = datetime.strptime(time_tag, "%Y-%m-%d %H:%M:%S")
        dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except (ValueError, TypeError):
        return None


def safe_float(val):
    """Convert a value to float, returning None on failure."""
    if val is None or val == "" or val == "null":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def safe_int(val):
    """Convert a value to int, returning None on failure."""
    if val is None or val == "" or val == "null":
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def parse_tabular_json(data):
    """Parse NOAA tabular JSON where first row is column headers.
    Returns list of dicts.
    """
    if not data or not isinstance(data, list) or len(data) < 2:
        return []
    headers = [str(h).strip() for h in data[0]]
    rows = []
    for row in data[1:]:
        if len(row) >= len(headers):
            rows.append(dict(zip(headers, row)))
    return rows


# ============================================================
# Data fetchers
# ============================================================

def fetch_kp_index():
    """Fetch latest Kp index from 1-min endpoint.
    Returns (kp_index, estimated_kp, ts_unix) for latest entry.
    """
    data = http_get(KP_1MIN_URL)
    if not data or not isinstance(data, list) or len(data) == 0:
        return None, None, None
    # Get the latest entry
    latest = data[-1]
    ts = parse_noaa_time(latest.get("time_tag"))
    kp = safe_float(latest.get("kp_index"))
    est_kp = safe_float(latest.get("estimated_kp"))
    # Some responses use "kp" instead of "kp_index"
    if kp is None:
        kp = safe_float(latest.get("kp"))
    return kp, est_kp, ts


def fetch_solar_wind_plasma():
    """Fetch latest solar wind plasma data (speed, density, temp).
    Returns (speed, density, temp, ts_unix).
    """
    data = http_get(SOLAR_WIND_PLASMA_URL)
    rows = parse_tabular_json(data)
    if not rows:
        return None, None, None, None
    # Get the latest row with valid data
    for row in reversed(rows):
        speed = safe_float(row.get("speed"))
        density = safe_float(row.get("density"))
        temp = safe_float(row.get("temperature"))
        ts = parse_noaa_time(row.get("time_tag"))
        if speed is not None or density is not None:
            return speed, density, temp, ts
    return None, None, None, None


def fetch_solar_wind_mag():
    """Fetch latest solar wind magnetic field data (Bz, Bt).
    Returns (bz, bt, ts_unix).
    """
    data = http_get(SOLAR_WIND_MAG_URL)
    rows = parse_tabular_json(data)
    if not rows:
        return None, None, None
    # Get the latest row with valid data
    for row in reversed(rows):
        bz = safe_float(row.get("bz_gsm"))
        bt = safe_float(row.get("bt"))
        ts = parse_noaa_time(row.get("time_tag"))
        if bz is not None or bt is not None:
            return bz, bt, ts
    return None, None, None


def fetch_noaa_scales():
    """Fetch current NOAA space weather scales (R, S, G).
    Returns (r_scale, s_scale, g_scale).
    """
    data = http_get(NOAA_SCALES_URL)
    if not data or not isinstance(data, dict):
        return None, None, None
    try:
        # Current conditions are in the "0" key
        current = data.get("0", {})
        r_val = safe_int(current.get("R", {}).get("Scale"))
        s_val = safe_int(current.get("S", {}).get("Scale"))
        g_val = safe_int(current.get("G", {}).get("Scale"))
        return r_val, s_val, g_val
    except (AttributeError, TypeError):
        return None, None, None


def fetch_kp_7day():
    """Fetch 7-day Kp index as fallback.
    Returns (kp, ts_unix) for latest entry.
    """
    data = http_get(KP_7DAY_URL)
    rows = parse_tabular_json(data)
    if not rows:
        return None, None
    latest = rows[-1]
    kp = safe_float(latest.get("Kp"))
    ts = parse_noaa_time(latest.get("time_tag"))
    return kp, ts


def fetch_solar_flares():
    """Fetch solar flares from 7-day endpoint.
    Returns list of dicts with flare info.
    """
    data = http_get(XRAY_FLARES_URL)
    if not data or not isinstance(data, list):
        return []
    flares = []
    for entry in data:
        max_class = entry.get("max_class")
        if not max_class:
            continue
        begin_time = entry.get("begin_time") or entry.get("time_tag")
        max_time = entry.get("max_time") or entry.get("time_tag")
        end_time = entry.get("end_time")
        max_xrlong = safe_float(entry.get("max_xrlong"))
        ts_unix = parse_noaa_time(begin_time)
        flares.append({
            "begin_time": begin_time,
            "max_time": max_time,
            "end_time": end_time,
            "max_class": str(max_class).strip(),
            "max_xrlong": max_xrlong,
            "ts_unix": ts_unix,
        })
    return flares


def fetch_solar_flux_f107():
    """Fetch latest F10.7 solar flux from NOAA."""
    data = http_get(F107_URL)
    if not data or not isinstance(data, list) or len(data) == 0:
        return None
    latest = data[-1]
    return safe_float(latest.get("flux"))


def fetch_sunspot_number():
    """Fetch latest daily sunspot number from SILSO (Royal Observatory of Belgium)."""
    try:
        text = http_get(SUNSPOT_URL, json_response=False)
        if not text:
            return None

        lines = text.strip().split('\n')
        if not lines:
            return None
        # Last line has latest data
        last_line = lines[-1].strip()
        parts = [p.strip() for p in last_line.split(',')]
        if len(parts) >= 5:
            return safe_float(parts[4])  # estimated_ssn
        return None
    except Exception as e:
        log.warning(f"  Sunspot fetch failed: {e}")
        return None


# ============================================================
# Main poll
# ============================================================

def poll_once(conn):
    """Fetch all space weather data and insert into DB."""
    now_utc = datetime.now(timezone.utc)
    now_unix = int(now_utc.timestamp())
    ts_str = now_utc.strftime("%Y-%m-%d %H:%M:%S")

    log.info(f"  Polling space weather data at {ts_str}...")

    # Fetch Kp index
    kp, est_kp, kp_ts = fetch_kp_index()
    if kp is None:
        # Fallback to 7-day Kp
        kp_7d, kp_7d_ts = fetch_kp_7day()
        if kp_7d is not None:
            kp = kp_7d
            log.info("    Used 7-day Kp fallback")

    # Fetch solar wind plasma
    sw_speed, sw_density, sw_temp, plasma_ts = fetch_solar_wind_plasma()

    # Fetch solar wind magnetics
    sw_bz, sw_bt, mag_ts = fetch_solar_wind_mag()

    # Fetch NOAA scales
    r_scale, s_scale, g_scale = fetch_noaa_scales()

    # Fetch solar flares
    flares = fetch_solar_flares()

    # Fetch sunspot number
    sunspot = fetch_sunspot_number()

    # Fetch solar flux F10.7
    f107 = fetch_solar_flux_f107()

    # Log summary
    log.info(f"    Kp: {kp}  Est Kp: {est_kp}")
    log.info(f"    Solar wind: speed={sw_speed} density={sw_density} temp={sw_temp}")
    log.info(f"    Magnetics: Bz={sw_bz} Bt={sw_bt}")
    log.info(f"    NOAA scales: R={r_scale} S={s_scale} G={g_scale}")
    log.info(f"    Sunspot: {sunspot}  F10.7: {f107}")
    log.info(f"    Flares found: {len(flares)}")

    # Insert space weather row (dedup by timestamp)
    new_sw = 0
    try:
        conn.execute("""
            INSERT OR IGNORE INTO space_weather
            (timestamp, kp_index, estimated_kp, solar_wind_speed,
             solar_wind_density, solar_wind_temp, solar_wind_bz,
             solar_wind_bt, r_scale, s_scale, g_scale,
             sunspot_number, solar_flux_f107)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            now_unix, kp, est_kp, sw_speed,
            sw_density, sw_temp, sw_bz,
            sw_bt, r_scale, s_scale, g_scale,
            sunspot, f107
        ))
        new_sw = conn.total_changes
        conn.commit()
    except Exception as e:
        log.error(f"    DB insert error (space_weather): {e}")

    # Insert flares (dedup by begin_time + max_class via UNIQUE constraint)
    new_flares = 0
    for flare in flares:
        try:
            conn.execute("""
                INSERT OR IGNORE INTO solar_flares
                (begin_time, max_time, end_time, max_class, max_xrlong, ts_unix)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                flare["begin_time"],
                flare["max_time"],
                flare["end_time"],
                flare["max_class"],
                flare["max_xrlong"],
                flare["ts_unix"],
            ))
            if conn.total_changes > 0:
                new_flares += 1
        except Exception as e:
            log.error(f"    DB insert error (solar_flares): {e}")
    conn.commit()

    # Count totals
    sw_count = conn.execute("SELECT COUNT(*) FROM space_weather").fetchone()[0]
    flare_count = conn.execute("SELECT COUNT(*) FROM solar_flares").fetchone()[0]

    log.info(f"    Inserted 1 space_weather row (total: {sw_count})")
    log.info(f"    New flares this poll: {new_flares} (total: {flare_count})")

    return True


# ============================================================
# Streamer loop
# ============================================================

def run_streamer(once=False):
    log.info("=" * 60)
    log.info("  SPACE WEATHER STREAMER -- NOAA SWPC Poller")
    log.info(f"  DB: {DB_PATH}")
    log.info(f"  Poll interval: {POLL_INTERVAL}s")
    log.info("=" * 60)

    conn = init_db()
    ensure_new_columns(conn)

    # Show current state
    last_ts = get_last_timestamp(conn)
    sw_count = conn.execute("SELECT COUNT(*) FROM space_weather").fetchone()[0]
    flare_count = conn.execute("SELECT COUNT(*) FROM solar_flares").fetchone()[0]
    log.info(f"  Existing space_weather rows: {sw_count}")
    log.info(f"  Existing solar_flares rows: {flare_count}")
    if last_ts:
        log.info(f"  Last data point: {last_ts} (unix)")
    else:
        log.info(f"  Database is empty -- first poll starting now")

    if once:
        log.info(f"\n  Running single poll (--once mode)...\n")
        try:
            poll_once(conn)
        except Exception as e:
            log.error(f"  Poll error: {e}")
        conn.close()
        return

    log.info(f"\n  Starting continuous poll loop...\n")

    while True:
        try:
            poll_once(conn)
            time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            log.info("\n  Shutting down space weather streamer...")
            break
        except Exception as e:
            log.error(f"  Streamer error: {e}")
            time.sleep(30)

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Space Weather Streamer -- NOAA SWPC Poller")
    parser.add_argument("--once", action="store_true", help="Single poll, no loop")
    args = parser.parse_args()
    run_streamer(once=args.once)
