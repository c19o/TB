"""
V2 Multi-Asset Data Downloader
================================
Downloads OHLCV for all 31 training assets + 3 inverse signal assets.
Uses FREE APIs only: Yahoo Finance (stocks/ETFs), CryptoCompare (crypto).
Resume support — continues from where it left off.

Saves to multi_asset_prices.db with same schema as btc_prices.db.
"""

import os
import sys
import time
import json
import sqlite3
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone

V2_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────────────────────────────────────
# ASSET DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

TICKERS_STOCK = {
    # US Stocks & ETFs — Yahoo Finance
    'SPY':  {'name': 'S&P 500 ETF',        'start': '1993-01-29', 'source': 'yahoo'},
    'QQQ':  {'name': 'Nasdaq 100 ETF',      'start': '1999-03-10', 'source': 'yahoo'},
    'TSLA': {'name': 'Tesla',               'start': '2010-06-29', 'source': 'yahoo'},
    'MSTR': {'name': 'MicroStrategy',       'start': '2000-06-07', 'source': 'yahoo'},
    'ARKK': {'name': 'ARK Innovation',      'start': '2014-10-31', 'source': 'yahoo'},
    # Commodity ETFs
    'GLD':  {'name': 'Gold ETF',            'start': '2004-11-18', 'source': 'yahoo'},
    'SLV':  {'name': 'Silver ETF',          'start': '2006-04-28', 'source': 'yahoo'},
    'USO':  {'name': 'Oil ETF',             'start': '2006-04-10', 'source': 'yahoo'},
    'UNG':  {'name': 'Natural Gas ETF',     'start': '2007-04-18', 'source': 'yahoo'},
    'WEAT': {'name': 'Wheat ETF',           'start': '2011-09-19', 'source': 'yahoo'},
    'CORN': {'name': 'Corn ETF',            'start': '2010-06-09', 'source': 'yahoo'},
    'DBA':  {'name': 'Agriculture ETF',     'start': '2007-01-05', 'source': 'yahoo'},
    # Global Index ETFs
    'EWJ':  {'name': 'Japan ETF',           'start': '1996-03-12', 'source': 'yahoo'},
    'FXI':  {'name': 'China ETF',           'start': '2004-10-05', 'source': 'yahoo'},
    'EWG':  {'name': 'Germany ETF',         'start': '1996-03-12', 'source': 'yahoo'},
    # Sector ETFs
    'XLE':  {'name': 'Energy Sector',       'start': '1998-12-16', 'source': 'yahoo'},
    'XLF':  {'name': 'Financials Sector',   'start': '1998-12-16', 'source': 'yahoo'},
}

TICKERS_CRYPTO = {
    # Crypto — CryptoCompare (free, daily back to genesis)
    'BTC':  {'name': 'Bitcoin',     'start': '2010-07-17', 'source': 'cryptocompare', 'fsym': 'BTC'},
    'ETH':  {'name': 'Ethereum',    'start': '2015-08-07', 'source': 'cryptocompare', 'fsym': 'ETH'},
    'XRP':  {'name': 'Ripple',      'start': '2013-08-04', 'source': 'cryptocompare', 'fsym': 'XRP'},
    'LTC':  {'name': 'Litecoin',    'start': '2013-04-28', 'source': 'cryptocompare', 'fsym': 'LTC'},
    'SOL':  {'name': 'Solana',      'start': '2020-04-10', 'source': 'cryptocompare', 'fsym': 'SOL'},
    'DOGE': {'name': 'Dogecoin',    'start': '2013-12-15', 'source': 'cryptocompare', 'fsym': 'DOGE'},
    'ADA':  {'name': 'Cardano',     'start': '2017-10-01', 'source': 'cryptocompare', 'fsym': 'ADA'},
    'BNB':  {'name': 'Binance Coin','start': '2017-07-25', 'source': 'cryptocompare', 'fsym': 'BNB'},
    'AVAX': {'name': 'Avalanche',   'start': '2020-09-22', 'source': 'cryptocompare', 'fsym': 'AVAX'},
    'LINK': {'name': 'Chainlink',   'start': '2017-09-20', 'source': 'cryptocompare', 'fsym': 'LINK'},
    'DOT':  {'name': 'Polkadot',    'start': '2020-08-19', 'source': 'cryptocompare', 'fsym': 'DOT'},
    'MATIC':{'name': 'Polygon',     'start': '2019-04-26', 'source': 'cryptocompare', 'fsym': 'MATIC'},
    'UNI':  {'name': 'Uniswap',     'start': '2020-09-17', 'source': 'cryptocompare', 'fsym': 'UNI'},
    'AAVE': {'name': 'Aave',        'start': '2020-10-02', 'source': 'cryptocompare', 'fsym': 'AAVE'},
}

# Inverse assets — features only, not training targets
TICKERS_INVERSE = {
    'UUP':  {'name': 'Dollar Bull ETF',    'start': '2007-02-20', 'source': 'yahoo'},
    'TLT':  {'name': 'Long Bond ETF',      'start': '2002-07-26', 'source': 'yahoo'},
    'FXY':  {'name': 'Yen ETF',            'start': '2007-02-12', 'source': 'yahoo'},
}

ALL_TICKERS = {**TICKERS_STOCK, **TICKERS_CRYPTO, **TICKERS_INVERSE}

DB_PATH = os.path.join(V2_DIR, "multi_asset_prices.db")


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────────────────────────────

def init_db():
    """Create multi-asset DB with same schema as btc_prices.db."""
    conn = sqlite3.connect(DB_PATH)
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
    conn.execute("CREATE INDEX IF NOT EXISTS idx_multi_ohlcv ON ohlcv(symbol, timeframe, open_time)")

    # Asset metadata table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS assets (
            symbol TEXT PRIMARY KEY,
            name TEXT,
            asset_class TEXT,
            source TEXT,
            start_date TEXT,
            last_updated TEXT,
            total_bars INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    return conn


def get_last_date(conn, symbol, timeframe='1d'):
    """Get last downloaded date for resume support."""
    row = conn.execute(
        "SELECT MAX(open_time) FROM ohlcv WHERE symbol=? AND timeframe=?",
        (symbol, timeframe)
    ).fetchone()
    if row[0]:
        return datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# YAHOO FINANCE DOWNLOADER (stocks/ETFs)
# ─────────────────────────────────────────────────────────────────────────────

def download_yahoo(conn, symbol, info, timeframe='1d'):
    """
    Download daily OHLCV from Yahoo Finance v8 chart API.
    Free, no auth, no library needed. Rate limit ~2000/hr.
    """
    last_date = get_last_date(conn, symbol, timeframe)
    if last_date:
        # Resume from day after last
        start_ts = int((last_date + timedelta(days=1)).timestamp())
        log(f"  {symbol}: resuming from {last_date.strftime('%Y-%m-%d')}")
    else:
        start_dt = datetime.strptime(info['start'], '%Y-%m-%d').replace(tzinfo=timezone.utc)
        start_ts = int(start_dt.timestamp())
        log(f"  {symbol}: downloading from {info['start']}")

    end_ts = int(datetime.now(timezone.utc).timestamp())

    if start_ts >= end_ts:
        log(f"  {symbol}: already up to date")
        return 0

    # Yahoo Finance v8 chart API — free, no auth
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?period1={start_ts}&period2={end_ts}&interval=1d"
        f"&includeAdjustedClose=true"
    )

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        # Yahoo sometimes returns 404 for valid tickers, retry with v7
        log(f"  {symbol}: Yahoo v8 HTTP {e.code}, trying v7...")
        url_v7 = (
            f"https://query2.finance.yahoo.com/v7/finance/chart/{symbol}"
            f"?period1={start_ts}&period2={end_ts}&interval=1d"
        )
        try:
            req = urllib.request.Request(url_v7, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e2:
            log(f"  {symbol}: FAILED — {e2}")
            return 0
    except Exception as e:
        log(f"  {symbol}: FAILED — {e}")
        return 0

    # Parse Yahoo response
    try:
        result = data['chart']['result'][0]
        timestamps = result['timestamp']
        quote = result['indicators']['quote'][0]
        opens = quote['open']
        highs = quote['high']
        lows = quote['low']
        closes = quote['close']
        volumes = quote['volume']
    except (KeyError, IndexError, TypeError) as e:
        log(f"  {symbol}: parse error — {e}")
        return 0

    inserted = 0
    for i in range(len(timestamps)):
        if any(v is None for v in [opens[i], highs[i], lows[i], closes[i]]):
            continue
        open_time_ms = timestamps[i] * 1000
        try:
            conn.execute("""
                INSERT OR IGNORE INTO ohlcv
                (symbol, timeframe, open_time, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (symbol, timeframe, open_time_ms,
                  opens[i], highs[i], lows[i], closes[i],
                  volumes[i] if volumes[i] else 0))
            inserted += 1
        except Exception:
            continue

    conn.commit()

    # Update asset metadata
    conn.execute("""
        INSERT OR REPLACE INTO assets (symbol, name, asset_class, source, start_date, last_updated, total_bars)
        VALUES (?, ?, ?, 'yahoo', ?, ?, ?)
    """, (symbol, info['name'],
          'stock' if symbol in TICKERS_STOCK else 'inverse',
          info['start'], datetime.now().strftime('%Y-%m-%d %H:%M'), inserted))
    conn.commit()

    log(f"  {symbol}: {inserted} bars inserted")
    return inserted


# ─────────────────────────────────────────────────────────────────────────────
# CRYPTOCOMPARE DOWNLOADER (crypto)
# ─────────────────────────────────────────────────────────────────────────────

def download_cryptocompare(conn, symbol, info, timeframe='1d'):
    """
    Download daily OHLCV from CryptoCompare.
    Free tier: 100K calls/month, no auth needed for daily.
    Data goes back to coin genesis.
    """
    fsym = info['fsym']
    last_date = get_last_date(conn, symbol, timeframe)

    if last_date:
        start_dt = last_date + timedelta(days=1)
        log(f"  {symbol}: resuming from {last_date.strftime('%Y-%m-%d')}")
    else:
        start_dt = datetime.strptime(info['start'], '%Y-%m-%d').replace(tzinfo=timezone.utc)
        log(f"  {symbol}: downloading from {info['start']}")

    now = datetime.now(timezone.utc)
    if start_dt >= now:
        log(f"  {symbol}: already up to date")
        return 0

    total_inserted = 0

    # CryptoCompare returns max 2000 daily bars per request
    # Walk forward from start_date
    current_ts = int(start_dt.timestamp())
    end_ts = int(now.timestamp())

    while current_ts < end_ts:
        # toTs = end of window, limit = bars to fetch
        # We request 2000 bars ending at current position + 2000 days
        window_end = min(current_ts + (2000 * 86400), end_ts)

        url = (
            f"https://min-api.cryptocompare.com/data/v2/histoday"
            f"?fsym={fsym}&tsym=USD&limit=2000&toTs={window_end}"
        )

        headers = {'User-Agent': 'Mozilla/5.0'}

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            log(f"  {symbol}: API error — {e}, retrying in 5s...")
            time.sleep(5)
            continue

        if data.get('Response') != 'Success':
            log(f"  {symbol}: API returned {data.get('Message', 'unknown error')}")
            break

        bars = data.get('Data', {}).get('Data', [])
        if not bars:
            break

        batch_inserted = 0
        max_ts = current_ts

        for bar in bars:
            ts = bar['time']
            if ts < int(start_dt.timestamp()):
                continue

            # Skip bars with zero volume AND zero close (empty/pre-listing)
            if bar['close'] == 0 and bar['volumeto'] == 0:
                continue

            open_time_ms = ts * 1000

            try:
                conn.execute("""
                    INSERT OR IGNORE INTO ohlcv
                    (symbol, timeframe, open_time, open, high, low, close, volume, quote_volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (symbol, timeframe, open_time_ms,
                      bar['open'], bar['high'], bar['low'], bar['close'],
                      bar['volumefrom'], bar['volumeto']))
                batch_inserted += 1
            except Exception:
                continue

            if ts > max_ts:
                max_ts = ts

        conn.commit()
        total_inserted += batch_inserted

        # Move window forward
        if max_ts <= current_ts:
            # No progress, jump forward
            current_ts = window_end
        else:
            current_ts = max_ts + 86400  # Next day

        # Rate limit: free tier is generous but be polite
        time.sleep(0.3)

        if batch_inserted > 0 and total_inserted % 2000 == 0:
            dt = datetime.fromtimestamp(max_ts, tz=timezone.utc)
            log(f"    {symbol}: {total_inserted} bars... {dt.strftime('%Y-%m-%d')}")

    # Update asset metadata
    conn.execute("""
        INSERT OR REPLACE INTO assets (symbol, name, asset_class, source, start_date, last_updated, total_bars)
        VALUES (?, ?, 'crypto', 'cryptocompare', ?, ?, ?)
    """, (symbol, info['name'], info['start'],
          datetime.now().strftime('%Y-%m-%d %H:%M'), total_inserted))
    conn.commit()

    log(f"  {symbol}: {total_inserted} bars inserted")
    return total_inserted


# ─────────────────────────────────────────────────────────────────────────────
# CRYPTO HOURLY DATA (for intraday TFs — crypto only)
# ─────────────────────────────────────────────────────────────────────────────

def download_crypto_hourly(conn, symbol, info):
    """
    Download hourly OHLCV for crypto from CryptoCompare.
    Used for 1H/4H TF training. Max 2000 bars per request.
    Downloads from coin's earliest available date (no 730-day limit).
    """
    fsym = info['fsym']
    timeframe = '1h'

    last_date = get_last_date(conn, symbol, timeframe)
    if last_date:
        start_dt = last_date + timedelta(hours=1)
        log(f"  {symbol} 1H: resuming from {last_date.strftime('%Y-%m-%d %H:%M')}")
    else:
        # Use actual coin start date — no artificial 730-day cap
        # Floor at 2017-08-17 (Binance-era, CryptoCompare hourly availability)
        coin_start = datetime.strptime(info['start'], '%Y-%m-%d').replace(tzinfo=timezone.utc)
        hourly_floor = datetime(2017, 8, 17, tzinfo=timezone.utc)
        start_dt = max(coin_start, hourly_floor)
        log(f"  {symbol} 1H: downloading from {start_dt.strftime('%Y-%m-%d')}")

    now = datetime.now(timezone.utc)
    if start_dt >= now:
        log(f"  {symbol} 1H: already up to date")
        return 0

    total_inserted = 0
    current_ts = int(start_dt.timestamp())
    end_ts = int(now.timestamp())

    while current_ts < end_ts:
        window_end = min(current_ts + (2000 * 3600), end_ts)

        url = (
            f"https://min-api.cryptocompare.com/data/v2/histohour"
            f"?fsym={fsym}&tsym=USD&limit=2000&toTs={window_end}"
        )

        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            log(f"  {symbol} 1H: API error — {e}, retrying...")
            time.sleep(5)
            continue

        if data.get('Response') != 'Success':
            break

        bars = data.get('Data', {}).get('Data', [])
        if not bars:
            break

        batch_inserted = 0
        max_ts = current_ts

        for bar in bars:
            ts = bar['time']
            if ts < int(start_dt.timestamp()) or (bar['close'] == 0 and bar['volumeto'] == 0):
                continue

            try:
                conn.execute("""
                    INSERT OR IGNORE INTO ohlcv
                    (symbol, timeframe, open_time, open, high, low, close, volume, quote_volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (symbol, timeframe, ts * 1000,
                      bar['open'], bar['high'], bar['low'], bar['close'],
                      bar['volumefrom'], bar['volumeto']))
                batch_inserted += 1
            except Exception:
                continue

            if ts > max_ts:
                max_ts = ts

        conn.commit()
        total_inserted += batch_inserted

        if max_ts <= current_ts:
            current_ts = window_end
        else:
            current_ts = max_ts + 3600

        time.sleep(0.3)

    log(f"  {symbol} 1H: {total_inserted} bars inserted")
    return total_inserted


def download_crypto_4h(conn, symbol, info):
    """
    Derive 4H candles from 1H data already in the database.
    Aggregates 4 consecutive 1H bars into 1 4H bar (00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC).
    """
    timeframe = '4h'

    # Check if we already have 4h data
    existing = conn.execute(
        "SELECT COUNT(*) FROM ohlcv WHERE symbol=? AND timeframe=?",
        (symbol, timeframe)
    ).fetchone()[0]

    # Get all 1h data
    rows_1h = conn.execute(
        "SELECT open_time, open, high, low, close, volume, quote_volume "
        "FROM ohlcv WHERE symbol=? AND timeframe='1h' ORDER BY open_time",
        (symbol,)
    ).fetchall()

    if not rows_1h:
        log(f"  {symbol} 4H: no 1H data to aggregate")
        return 0

    # Group into 4-hour windows (0, 4, 8, 12, 16, 20 UTC)
    from collections import defaultdict
    windows = defaultdict(list)
    for row in rows_1h:
        ts_ms = row[0]
        ts_s = ts_ms / 1000
        dt = datetime.fromtimestamp(ts_s, tz=timezone.utc)
        # Floor to nearest 4-hour boundary
        hour_4 = (dt.hour // 4) * 4
        window_dt = dt.replace(hour=hour_4, minute=0, second=0, microsecond=0)
        window_ms = int(window_dt.timestamp() * 1000)
        windows[window_ms].append(row)

    inserted = 0
    for window_ms, bars in sorted(windows.items()):
        if len(bars) < 4:
            continue  # Incomplete 4h window, skip
        o = bars[0][1]   # open of first bar
        h = max(b[2] for b in bars)  # highest high
        l = min(b[3] for b in bars)  # lowest low
        c = bars[-1][4]  # close of last bar
        v = sum(b[5] or 0 for b in bars)  # sum volume
        qv = sum(b[6] or 0 for b in bars)  # sum quote volume

        try:
            conn.execute("""
                INSERT OR IGNORE INTO ohlcv
                (symbol, timeframe, open_time, open, high, low, close, volume, quote_volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (symbol, timeframe, window_ms, o, h, l, c, v, qv))
            inserted += 1
        except Exception:
            continue

    conn.commit()
    log(f"  {symbol} 4H: {inserted} bars aggregated from 1H (had {existing} existing)")
    return inserted


# ─────────────────────────────────────────────────────────────────────────────
# MACRO DATA (FRED — VIX, DXY, yields, etc.)
# ─────────────────────────────────────────────────────────────────────────────

def download_fred_series(conn, series_id, description):
    """
    Download a FRED series. Free, no auth needed for CSV format.
    """
    log(f"  FRED {series_id}: {description}")

    url = (
        f"https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series_id}&cosd=1990-01-01"
    )

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            csv_data = resp.read().decode()
    except Exception as e:
        log(f"  FRED {series_id}: FAILED — {e}")
        return 0

    conn.execute("""
        CREATE TABLE IF NOT EXISTS macro (
            series TEXT NOT NULL,
            date TEXT NOT NULL,
            value REAL,
            PRIMARY KEY (series, date)
        )
    """)

    inserted = 0
    for line in csv_data.strip().split('\n')[1:]:  # Skip header
        parts = line.strip().split(',')
        if len(parts) != 2:
            continue
        date_str, val_str = parts
        if val_str == '.' or val_str == '':
            continue
        try:
            val = float(val_str)
            conn.execute("INSERT OR REPLACE INTO macro VALUES (?, ?, ?)",
                         (series_id, date_str, val))
            inserted += 1
        except (ValueError, sqlite3.Error):
            continue

    conn.commit()
    log(f"  FRED {series_id}: {inserted} data points")
    return inserted


def download_all_macro(conn):
    """Download key macro indicators from FRED."""
    log("\n=== MACRO DATA (FRED) ===")
    series = {
        'VIXCLS':    'VIX (CBOE Volatility Index)',
        'DTWEXBGS':  'Trade-Weighted USD Index (Broad)',
        'DGS10':     'US 10-Year Treasury Yield',
        'DGS2':      'US 2-Year Treasury Yield',
        'T10Y2Y':    'Yield Curve (10Y-2Y Spread)',
        'BAMLH0A0HYM2': 'High Yield Spread (risk appetite)',
        'DCOILWTICO': 'WTI Crude Oil Price',
        'GOLDAMGBD228NLBM': 'Gold Price (London Fix)',
    }
    total = 0
    for sid, desc in series.items():
        total += download_fred_series(conn, sid, desc)
        time.sleep(0.5)
    return total


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def download_all():
    """Download everything."""
    conn = init_db()

    log("=" * 60)
    log("V2 MULTI-ASSET DATA DOWNLOAD")
    log(f"Database: {DB_PATH}")
    log(f"Tickers: {len(TICKERS_STOCK)} stocks + {len(TICKERS_CRYPTO)} crypto + {len(TICKERS_INVERSE)} inverse")
    log("=" * 60)

    # ── Stocks & ETFs (Yahoo Finance) ──
    log("\n=== STOCKS & ETFS (Yahoo Finance — daily) ===")
    stock_total = 0
    for symbol, info in TICKERS_STOCK.items():
        stock_total += download_yahoo(conn, symbol, info)
        time.sleep(1)  # Rate limit: 1 req/sec

    # ── Inverse signals (Yahoo Finance) ──
    log("\n=== INVERSE SIGNALS (Yahoo Finance — daily) ===")
    for symbol, info in TICKERS_INVERSE.items():
        stock_total += download_yahoo(conn, symbol, info)
        time.sleep(1)

    # ── Crypto daily (CryptoCompare) ──
    log("\n=== CRYPTO (CryptoCompare — daily) ===")
    crypto_total = 0
    for symbol, info in TICKERS_CRYPTO.items():
        crypto_total += download_cryptocompare(conn, symbol, info)

    # ── Crypto hourly (CryptoCompare — full history) ──
    log("\n=== CRYPTO HOURLY (CryptoCompare — 1H, full history) ===")
    crypto_hourly = 0
    for symbol, info in TICKERS_CRYPTO.items():
        crypto_hourly += download_crypto_hourly(conn, symbol, info)

    # ── Crypto 4H (aggregated from 1H) ──
    log("\n=== CRYPTO 4H (aggregated from 1H data) ===")
    crypto_4h = 0
    for symbol, info in TICKERS_CRYPTO.items():
        crypto_4h += download_crypto_4h(conn, symbol, info)

    # ── Macro data (FRED) ──
    macro_total = download_all_macro(conn)

    # ── Summary ──
    log("\n" + "=" * 60)
    log("DOWNLOAD COMPLETE")
    log("=" * 60)

    # Print asset summary
    rows = conn.execute("""
        SELECT symbol, COUNT(*), MIN(open_time), MAX(open_time)
        FROM ohlcv WHERE timeframe='1d'
        GROUP BY symbol ORDER BY symbol
    """).fetchall()

    log(f"\n{'Symbol':<8} {'Bars':>8} {'From':>12} {'To':>12}")
    log("-" * 44)
    total_bars = 0
    for r in rows:
        sym = r[0]
        count = r[1]
        start = datetime.fromtimestamp(r[2]/1000, tz=timezone.utc).strftime('%Y-%m-%d')
        end = datetime.fromtimestamp(r[3]/1000, tz=timezone.utc).strftime('%Y-%m-%d')
        log(f"{sym:<8} {count:>8} {start:>12} {end:>12}")
        total_bars += count

    log("-" * 44)
    log(f"{'TOTAL':<8} {total_bars:>8} daily bars across {len(rows)} assets")

    # Hourly summary
    hourly_rows = conn.execute("""
        SELECT symbol, COUNT(*) FROM ohlcv WHERE timeframe='1h' GROUP BY symbol
    """).fetchall()
    hourly_total = sum(r[1] for r in hourly_rows)
    log(f"{'HOURLY':<8} {hourly_total:>8} bars across {len(hourly_rows)} crypto assets")

    # 4H summary
    fourh_rows = conn.execute("""
        SELECT symbol, COUNT(*) FROM ohlcv WHERE timeframe='4h' GROUP BY symbol
    """).fetchall()
    fourh_total = sum(r[1] for r in fourh_rows)
    log(f"{'4H':<8} {fourh_total:>8} bars across {len(fourh_rows)} crypto assets")

    # Macro summary
    macro_rows = conn.execute("SELECT series, COUNT(*) FROM macro GROUP BY series").fetchall()
    macro_total = sum(r[1] for r in macro_rows)
    log(f"{'MACRO':<8} {macro_total:>8} data points across {len(macro_rows)} series")

    conn.close()
    log("\nDone.")


if __name__ == '__main__':
    download_all()
