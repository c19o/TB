"""
V2 Easy Streamers — 5 simple REST endpoints, no auth, no scraping.
Each is one GET call. Run daily via cron or scheduler.

1. DefiLlama — Total DeFi TVL
2. CoinGecko — BTC dominance + global market data
3. Blockchain.com — Mining stats (hash rate, difficulty)
4. CFTC — COT report (weekly CSV)
5. CoinGecko — Multi-crypto OHLCV for live inference
"""

import os
import json
import time
import signal
import sqlite3
import urllib.request
from datetime import datetime, timezone

from universal_numerology import digital_root
from universal_gematria import gematria as _gematria

V2_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(V2_DIR, "v2_signals.db")


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _get_json(url, timeout=30):
    """Simple GET returning parsed JSON."""
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _dr(v):
    """Digital root of a numeric value. Returns None if value is None or zero."""
    if v is None:
        return None
    iv = int(abs(v))
    if iv == 0:
        return 0
    return digital_root(iv)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS defi_tvl (
            date TEXT PRIMARY KEY,
            total_tvl REAL,
            total_tvl_change_1d REAL,
            tvl_dr INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS btc_dominance (
            date TEXT PRIMARY KEY,
            btc_dominance REAL,
            eth_dominance REAL,
            total_market_cap REAL,
            total_volume REAL,
            active_cryptos INTEGER,
            btc_dom_dr INTEGER,
            eth_dom_dr INTEGER,
            mcap_dr INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mining_stats (
            date TEXT PRIMARY KEY,
            hash_rate REAL,
            difficulty REAL,
            blocks_mined INTEGER,
            miners_revenue REAL,
            avg_block_size REAL,
            hash_rate_dr INTEGER,
            difficulty_dr INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cot_report (
            date TEXT NOT NULL,
            asset TEXT NOT NULL,
            large_spec_long INTEGER,
            large_spec_short INTEGER,
            commercial_long INTEGER,
            commercial_short INTEGER,
            small_spec_long INTEGER,
            small_spec_short INTEGER,
            ls_long_dr INTEGER,
            ls_short_dr INTEGER,
            asset_gem_ordinal INTEGER,
            asset_gem_dr INTEGER,
            PRIMARY KEY (date, asset)
        )
    """)
    _ensure_esoteric_cols(conn)
    conn.commit()
    return conn


_DEFI_COLS     = {"tvl_dr": "INTEGER"}
_DOM_COLS      = {"btc_dom_dr": "INTEGER", "eth_dom_dr": "INTEGER", "mcap_dr": "INTEGER"}
_MINING_COLS   = {"hash_rate_dr": "INTEGER", "difficulty_dr": "INTEGER"}
_COT_COLS      = {"ls_long_dr": "INTEGER", "ls_short_dr": "INTEGER",
                  "asset_gem_ordinal": "INTEGER", "asset_gem_dr": "INTEGER"}


def _ensure_esoteric_cols(conn):
    """Add esoteric DR/gematria columns to existing tables if not present."""
    for table, cols in [
        ("defi_tvl", _DEFI_COLS),
        ("btc_dominance", _DOM_COLS),
        ("mining_stats", _MINING_COLS),
        ("cot_report", _COT_COLS),
    ]:
        existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        for col, td in cols.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {td}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. DEFI TVL (DefiLlama — free, no auth, no rate limit)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_defi_tvl(conn):
    """Fetch total DeFi TVL from DefiLlama. Historical daily data."""
    log("Fetching DeFi TVL (DefiLlama)...")
    try:
        data = _get_json("https://api.llama.fi/v2/historicalChainTvl")
    except Exception as e:
        log(f"  FAILED: {e}")
        return 0

    inserted = 0
    prev_tvl = None
    for point in data:
        ts = point.get('date', 0)
        tvl = point.get('tvl', 0)
        if ts == 0 or tvl == 0:
            continue

        date_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d')
        change_1d = ((tvl / prev_tvl) - 1) * 100 if prev_tvl and prev_tvl > 0 else 0
        prev_tvl = tvl

        conn.execute(
            "INSERT OR REPLACE INTO defi_tvl VALUES (?, ?, ?, ?)",
            (date_str, tvl, change_1d, _dr(tvl)),
        )
        inserted += 1

    conn.commit()
    log(f"  DeFi TVL: {inserted} days")
    return inserted


# ─────────────────────────────────────────────────────────────────────────────
# 2. BTC DOMINANCE (CoinGecko — free, no auth)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_btc_dominance(conn):
    """Fetch BTC dominance + global market data from CoinGecko."""
    log("Fetching BTC dominance (CoinGecko)...")
    try:
        data = _get_json("https://api.coingecko.com/api/v3/global")
    except Exception as e:
        log(f"  FAILED: {e}")
        return 0

    gd = data.get('data', {})
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')

    btc_dom = gd.get('market_cap_percentage', {}).get('btc', 0)
    eth_dom = gd.get('market_cap_percentage', {}).get('eth', 0)
    mcap    = gd.get('total_market_cap', {}).get('usd', 0)
    conn.execute(
        "INSERT OR REPLACE INTO btc_dominance VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (today, btc_dom, eth_dom, mcap,
         gd.get('total_volume', {}).get('usd', 0),
         gd.get('active_cryptocurrencies', 0),
         _dr(btc_dom), _dr(eth_dom), _dr(mcap)),
    )
    conn.commit()
    log(f"  BTC dominance: {gd.get('market_cap_percentage', {}).get('btc', 0):.1f}%")
    return 1


# ─────────────────────────────────────────────────────────────────────────────
# 3. MINING STATS (Blockchain.com — free, no auth)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_mining_stats(conn):
    """Fetch BTC mining stats from Blockchain.com."""
    log("Fetching mining stats (Blockchain.com)...")
    try:
        data = _get_json("https://api.blockchain.info/stats")
    except Exception as e:
        log(f"  FAILED: {e}")
        return 0

    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')

    hr   = data.get('hash_rate', 0)
    diff = data.get('difficulty', 0)
    conn.execute(
        "INSERT OR REPLACE INTO mining_stats VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (today, hr, diff,
         data.get('n_blocks_mined', 0),
         data.get('miners_revenue_btc', 0),
         data.get('blocks_size', 0),
         _dr(hr), _dr(diff)),
    )
    conn.commit()
    log(f"  Hash rate: {data.get('hash_rate', 0):.2e}, Difficulty: {data.get('difficulty', 0):.2e}")
    return 1


# ─────────────────────────────────────────────────────────────────────────────
# 4. COT REPORT (CFTC — weekly CSV, free)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_cot_report(conn):
    """
    Fetch CME Bitcoin futures COT data from CFTC.
    Published every Friday for the prior Tuesday.
    Uses the Quandl-style CFTC bulk CSV.
    """
    log("Fetching COT report (CFTC)...")

    # CFTC publishes combined futures report as CSV
    # Bitcoin futures contract code: 133741
    url = "https://www.cftc.gov/dea/newcot/deafut.txt"

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60) as resp:
            text = resp.read().decode('latin-1')
    except Exception as e:
        log(f"  COT FAILED: {e}")
        return 0

    # Parse the fixed-width CFTC report for Bitcoin futures
    inserted = 0
    for line in text.split('\n'):
        if 'BITCOIN' not in line.upper():
            continue

        # CFTC format is comma-separated with many fields
        fields = line.split(',')
        if len(fields) < 20:
            continue

        try:
            date_str = fields[2].strip()  # As-of date
            # Parse CFTC date format
            if len(date_str) >= 6:
                # Try to normalize date
                for fmt in ['%Y%m%d', '%y%m%d', '%m/%d/%Y', '%m/%d/%y']:
                    try:
                        dt = datetime.strptime(date_str, fmt)
                        date_str = dt.strftime('%Y-%m-%d')
                        break
                    except ValueError:
                        continue

            ls_long  = int(fields[7].strip())  if fields[7].strip().isdigit()  else 0
            ls_short = int(fields[8].strip())  if fields[8].strip().isdigit()  else 0
            asset_name = 'BTC_FUTURES'
            _gem = _gematria(asset_name)
            conn.execute("""
                INSERT OR REPLACE INTO cot_report
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (date_str, asset_name,
                  ls_long, ls_short,
                  int(fields[11].strip()) if fields[11].strip().isdigit() else 0,  # commercial long
                  int(fields[12].strip()) if fields[12].strip().isdigit() else 0,  # commercial short
                  int(fields[15].strip()) if fields[15].strip().isdigit() else 0,  # small spec long
                  int(fields[16].strip()) if fields[16].strip().isdigit() else 0,  # small spec short
                  _dr(ls_long), _dr(ls_short),
                  _gem["ordinal"], _gem["dr_ordinal"]))
            inserted += 1
        except (ValueError, IndexError):
            continue

    conn.commit()
    log(f"  COT: {inserted} BTC futures reports")
    return inserted


# ─────────────────────────────────────────────────────────────────────────────
# 5. MULTI-CRYPTO LIVE OHLCV (CoinGecko — for live inference on any crypto)
# ─────────────────────────────────────────────────────────────────────────────

LIVE_CRYPTO_IDS = {
    'BTC': 'bitcoin', 'ETH': 'ethereum', 'XRP': 'ripple', 'SOL': 'solana',
    'DOGE': 'dogecoin', 'ADA': 'cardano', 'AVAX': 'avalanche-2',
    'LINK': 'chainlink', 'DOT': 'polkadot', 'MATIC': 'matic-network',
    'BNB': 'binancecoin', 'UNI': 'uniswap', 'LTC': 'litecoin', 'AAVE': 'aave',
}

def fetch_multi_crypto_prices():
    """
    Fetch current prices for all tradeable cryptos.
    Returns dict of {symbol: {price, volume_24h, market_cap, change_24h}}.
    Used by live_trader.py for multi-crypto inference.
    """
    ids = ','.join(LIVE_CRYPTO_IDS.values())
    url = (
        f"https://api.coingecko.com/api/v3/simple/price"
        f"?ids={ids}&vs_currencies=usd"
        f"&include_24hr_vol=true&include_24hr_change=true&include_market_cap=true"
    )

    try:
        data = _get_json(url)
    except Exception as e:
        log(f"  Multi-crypto prices FAILED: {e}")
        return {}

    result = {}
    for sym, cg_id in LIVE_CRYPTO_IDS.items():
        if cg_id in data:
            d = data[cg_id]
            result[sym] = {
                'price': d.get('usd', 0),
                'volume_24h': d.get('usd_24h_vol', 0),
                'market_cap': d.get('usd_market_cap', 0),
                'change_24h': d.get('usd_24h_change', 0),
            }

    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_all():
    """Run all easy streamers. Call daily."""
    conn = init_db()

    log("=" * 50)
    log("V2 EASY STREAMERS")
    log("=" * 50)

    fetch_defi_tvl(conn)
    time.sleep(1)

    fetch_btc_dominance(conn)
    time.sleep(1)

    fetch_mining_stats(conn)
    time.sleep(1)

    fetch_cot_report(conn)

    prices = fetch_multi_crypto_prices()
    if prices:
        log(f"\nLive crypto prices: {len(prices)} coins")
        for sym, d in sorted(prices.items()):
            log(f"  {sym}: ${d['price']:,.2f} ({d['change_24h']:+.1f}%)")

    conn.close()
    log("\nDone.")


# Precomputed coin name gematria (static reference — logged at startup)
_COIN_GEM = {sym: _gematria(sym) for sym in LIVE_CRYPTO_IDS}


def _log_coin_gematria():
    """Log gematria reference values for all tracked coins at startup."""
    log("=== Coin Name Gematria Reference ===")
    for sym, g in _COIN_GEM.items():
        log(f"  {sym}: ordinal={g['ordinal']} dr={g['dr_ordinal']} "
            f"is_caution={g['ordinal'] in {93,39,43,48,223,322,17,113,11,19,13,22,33,44,66,77,88,99}} "
            f"is_btc_energy={g['ordinal'] in {213,231,312,321,132,123,68}}")


_SHUTDOWN = False


def _handle_signal(sig, frame):
    global _SHUTDOWN
    _SHUTDOWN = True
    log("Shutdown signal received — finishing current cycle...")


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    _log_coin_gematria()

    log("Starting V2 Easy Streamers daemon (daily cycle)...")
    while not _SHUTDOWN:
        try:
            run_all()
        except Exception as e:
            log(f"ERROR in run_all: {e}")
        if _SHUTDOWN:
            break
        # Sleep 24h in 1s ticks so signal handling is responsive
        for _ in range(86400):
            if _SHUTDOWN:
                break
            time.sleep(1)

    log("V2 Easy Streamers daemon stopped.")
