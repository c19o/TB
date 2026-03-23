#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
data_access_v2.py — V2 Multi-Asset Data Access Layer
=====================================================
Extends V1 data_access.py with:
  - Multi-asset OHLCV loading from multi_asset_prices.db
  - V2 signal data (DeFi TVL, BTC dominance, mining, COT)
  - Inverse signal loading (UUP, TLT, FXY as features)
  - Macro data from FRED
  - All V1 esoteric data (tweets, news, astro, sports, etc.) via V1 DBs

Used by:
  - build_features_v2.py  →  V2OfflineDataLoader (full history, any asset)
  - live_trader_v2.py     →  V2LiveDataLoader (incremental cache)
"""

import os
import time
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# V2 project directory
V2_DIR = os.path.dirname(os.path.abspath(__file__))

# V1 directory (shared esoteric DBs)
V1_DIR = os.environ.get("SAVAGE22_V1_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# V2 databases
MULTI_ASSET_DB = os.path.join(V2_DIR, "multi_asset_prices.db")
V2_SIGNALS_DB = os.path.join(V2_DIR, "v2_signals.db")

# V1 databases (shared — never duplicate, always read from V1)
V1_DBS = {
    'btc_prices':    os.path.join(V1_DIR, "btc_prices.db"),
    'tweets':        os.path.join(V1_DIR, "tweets.db"),
    'news':          os.path.join(V1_DIR, "news_articles.db"),
    'astrology':     os.path.join(V1_DIR, "astrology_full.db"),
    'ephemeris':     os.path.join(V1_DIR, "ephemeris_cache.db"),
    'fear_greed':    os.path.join(V1_DIR, "fear_greed.db"),
    'sports':        os.path.join(V1_DIR, "sports_results.db"),
    'space_weather': os.path.join(V1_DIR, "space_weather.db"),
    'macro':         os.path.join(V1_DIR, "macro_data.db"),
    'onchain':       os.path.join(V1_DIR, "onchain_data.db"),
    'funding':       os.path.join(V1_DIR, "funding_rates.db"),
    'oi':            os.path.join(V1_DIR, "open_interest.db"),
    'google_trends': os.path.join(V1_DIR, "google_trends.db"),
    'llm_cache':     os.path.join(V1_DIR, "llm_cache.db"),
}


# ============================================================
# SHARED DB HELPERS
# ============================================================

def _connect(db_path):
    """Open a read-only SQLite connection."""
    if not os.path.exists(db_path):
        print(f"WARNING: DB missing: {db_path}", flush=True)
        return None
    return sqlite3.connect(db_path, timeout=10)


def _safe_read_sql(conn, query, params=None):
    """Read SQL with error handling."""
    try:
        return pd.read_sql_query(query, conn, params=params)
    except Exception:
        return pd.DataFrame()


# Module-level flag: only validate DBs once per process
_validated = False


def validate_required_dbs():
    """
    Check all V1 + V2 databases and print a summary of what's available vs missing.
    Called once on the first load_all_data_for_asset() call so the user knows
    immediately if the model will train on partial data.
    """
    global _validated
    if _validated:
        return
    _validated = True

    print("\n" + "=" * 60, flush=True)
    print("DATABASE AVAILABILITY CHECK", flush=True)
    print("=" * 60, flush=True)

    missing = []
    present = []

    # V1 databases
    for name, path in V1_DBS.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            present.append((f"V1/{name}", path, size_mb))
        else:
            missing.append((f"V1/{name}", path))

    # V2 databases
    for name, path in [("V2/multi_asset_prices", MULTI_ASSET_DB),
                        ("V2/v2_signals", V2_SIGNALS_DB)]:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            present.append((name, path, size_mb))
        else:
            missing.append((name, path))

    # Print present
    for name, path, size_mb in present:
        print(f"  OK   {name:30s} ({size_mb:,.1f} MB)", flush=True)

    # Print missing with loud warnings
    if missing:
        print(flush=True)
        print("!" * 60, flush=True)
        print(f"WARNING: {len(missing)} DATABASE(S) MISSING — model trains WITHOUT these signals!", flush=True)
        print("!" * 60, flush=True)
        for name, path in missing:
            print(f"  MISSING  {name:30s}  ({path})", flush=True)
        print("!" * 60, flush=True)
    else:
        print(f"\n  All {len(present)} databases present.", flush=True)

    print("=" * 60 + "\n", flush=True)


# ============================================================
# V2 OFFLINE DATA LOADER (full history, multi-asset)
# ============================================================

class V2OfflineDataLoader:
    """
    Loads full history for feature building. Handles any asset.
    Reads OHLCV from multi_asset_prices.db, esoteric from V1 DBs.
    """

    def __init__(self):
        pass

    # ── OHLCV ──

    def load_ohlcv(self, symbol: str, tf: str = '1d') -> pd.DataFrame:
        """
        Load OHLCV for any asset. Checks multi_asset_prices.db first,
        falls back to V1 btc_prices.db for BTC intraday.

        Returns DatetimeIndex DataFrame: open, high, low, close, volume, etc.
        """
        df = self._load_from_multi_asset(symbol, tf)

        # Fallback to V1 btc_prices.db for BTC intraday
        if df.empty and symbol in ('BTC', 'BTC/USDT'):
            df = self._load_from_v1_btc(tf)

        return df

    def _load_from_multi_asset(self, symbol, tf):
        conn = _connect(MULTI_ASSET_DB)
        if conn is None:
            return pd.DataFrame()

        df = _safe_read_sql(conn, """
            SELECT open_time, open, high, low, close, volume, quote_volume
            FROM ohlcv WHERE symbol=? AND timeframe=?
            ORDER BY open_time
        """, (symbol, tf))
        conn.close()

        if df.empty:
            return df

        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
        df = df.drop_duplicates(subset='open_time', keep='last')
        df = df.set_index('timestamp').sort_index()
        df.index = df.index.tz_localize(None)  # Strip TZ for cuDF compat

        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop zero-volume leading bars
        if 'volume' in df.columns and len(df) > 0:
            first_good = df[df['volume'] > 0].index
            if len(first_good) > 0:
                df = df.loc[first_good[0]:]

        return df

    def _load_from_v1_btc(self, tf):
        """Load BTC from V1 btc_prices.db (has all intraday TFs)."""
        conn = _connect(V1_DBS['btc_prices'])
        if conn is None:
            return pd.DataFrame()

        df = _safe_read_sql(conn, """
            SELECT open_time, open, high, low, close, volume,
                   quote_volume, trades, taker_buy_volume, taker_buy_quote
            FROM ohlcv WHERE timeframe=? AND symbol='BTC/USDT'
            ORDER BY open_time
        """, (tf,))
        conn.close()

        if df.empty:
            return df

        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
        df = df.drop_duplicates(subset='open_time', keep='last')
        df = df.set_index('timestamp').sort_index()
        df.index = df.index.tz_localize(None)

        for col in ['open', 'high', 'low', 'close', 'volume',
                     'quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'volume' in df.columns and len(df) > 0:
            first_good = df[df['volume'] > 0].index
            if len(first_good) > 0:
                df = df.loc[first_good[0]:]

        return df

    def load_all_assets_daily(self) -> dict:
        """
        Load daily OHLCV for ALL 31 training assets.
        Returns dict: {symbol: DataFrame}
        """
        from config import ALL_TRAINING
        result = {}
        for symbol in ALL_TRAINING:
            df = self.load_ohlcv(symbol, '1d')
            if not df.empty:
                result[symbol] = df
        return result

    def load_inverse_signals(self) -> dict:
        """Load daily OHLCV for inverse signal assets (UUP, TLT, FXY)."""
        from config import INVERSE_SIGNALS
        result = {}
        for symbol in INVERSE_SIGNALS:
            df = self.load_ohlcv(symbol, '1d')
            if not df.empty:
                result[symbol] = df
        return result

    # ── V2 SIGNAL DATA ──

    def load_defi_tvl(self) -> pd.DataFrame:
        """Load DeFi TVL history from v2_signals.db."""
        conn = _connect(V2_SIGNALS_DB)
        if conn is None:
            return pd.DataFrame()

        df = _safe_read_sql(conn, "SELECT * FROM defi_tvl ORDER BY date")
        conn.close()

        if df.empty:
            return df

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.drop_duplicates(subset='date', keep='last').set_index('date')
        return df

    def load_btc_dominance(self) -> pd.DataFrame:
        """Load BTC dominance history from v2_signals.db."""
        conn = _connect(V2_SIGNALS_DB)
        if conn is None:
            return pd.DataFrame()

        df = _safe_read_sql(conn, "SELECT * FROM btc_dominance ORDER BY date")
        conn.close()

        if df.empty:
            return df

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.drop_duplicates(subset='date', keep='last').set_index('date')
        return df

    def load_mining_stats(self) -> pd.DataFrame:
        """Load mining stats history from v2_signals.db."""
        conn = _connect(V2_SIGNALS_DB)
        if conn is None:
            return pd.DataFrame()

        df = _safe_read_sql(conn, "SELECT * FROM mining_stats ORDER BY date")
        conn.close()

        if df.empty:
            return df

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.drop_duplicates(subset='date', keep='last').set_index('date')
        return df

    def load_cot_report(self) -> pd.DataFrame:
        """Load COT positioning data from v2_signals.db."""
        conn = _connect(V2_SIGNALS_DB)
        if conn is None:
            return pd.DataFrame()

        df = _safe_read_sql(conn, "SELECT * FROM cot_report ORDER BY date")
        conn.close()

        if df.empty:
            return df

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.drop_duplicates(subset=['date', 'asset'], keep='last')
        return df

    def load_macro_fred(self) -> pd.DataFrame:
        """Load FRED macro data from multi_asset_prices.db."""
        conn = _connect(MULTI_ASSET_DB)
        if conn is None:
            return pd.DataFrame()

        df = _safe_read_sql(conn, "SELECT * FROM macro ORDER BY date")
        conn.close()

        if df.empty:
            return df

        # Pivot: each series becomes a column
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        pivot = df.pivot_table(index='date', columns='series', values='value', aggfunc='last')
        pivot = pivot.sort_index()
        return pivot

    # ── V1 ESOTERIC DATA (shared, read from V1 directory) ──

    def load_tweets(self, start_ts: int = None) -> pd.DataFrame:
        conn = _connect(V1_DBS['tweets'])
        if conn is None:
            return pd.DataFrame()

        query = """
            SELECT created_at, ts_unix, user_handle, full_text,
                   has_gold, has_red, has_green, dominant_colors,
                   gematria_simple, gematria_english,
                   favorite_count, retweet_count, reply_count
            FROM tweets
        """
        if start_ts:
            query += f" WHERE ts_unix >= {start_ts}"
        query += " ORDER BY ts_unix"

        df = _safe_read_sql(conn, query)
        conn.close()

        if not df.empty:
            df['ts_unix'] = pd.to_numeric(df['ts_unix'], errors='coerce')
            df = df.dropna(subset=['ts_unix'])
        return df

    def load_news(self, start_ts: int = None) -> pd.DataFrame:
        conn = _connect(V1_DBS['news'])
        if conn is None:
            return pd.DataFrame()

        where = f" WHERE ts_unix >= {start_ts}" if start_ts else ""
        df = _safe_read_sql(conn, f"""
            SELECT ts_unix, title, sentiment_score, title_dr,
                   title_gematria_ordinal, title_gematria_reverse,
                   title_gematria_reduction, sentiment_bull, sentiment_bear,
                   has_caps, exclamation_count, word_count
            FROM streamer_articles {where} ORDER BY ts_unix
        """)

        if df.empty:
            df = _safe_read_sql(conn, f"""
                SELECT ts_unix, title, sentiment_score, title_dr,
                       title_gematria, body_word_count, date_doy
                FROM articles {where} ORDER BY ts_unix
            """)

        conn.close()

        if not df.empty:
            df['ts_unix'] = pd.to_numeric(df['ts_unix'], errors='coerce')
            df = df.dropna(subset=['ts_unix'])
        return df

    def load_sports(self) -> dict:
        conn = _connect(V1_DBS['sports'])
        if conn is None:
            return {'games': pd.DataFrame(), 'horse_races': pd.DataFrame()}

        games = _safe_read_sql(conn, """
            SELECT date, winner, home_team, away_team, home_score, away_score,
                   venue, winner_gem_ordinal, winner_gem_dr,
                   home_gem_ordinal, home_gem_dr, away_gem_ordinal, away_gem_dr,
                   score_total, score_diff, score_dr, score_total_dr,
                   is_upset, is_overtime
            FROM games ORDER BY date
        """)

        horses = _safe_read_sql(conn, """
            SELECT date, winner_horse, winner_jockey, winner_trainer,
                   horse_gem_ordinal, horse_gem_dr,
                   jockey_gem_ordinal, race_gem_ordinal, position_dr, odds_dr
            FROM horse_races ORDER BY date
        """)

        conn.close()
        return {'games': games, 'horse_races': horses}

    def load_onchain(self) -> dict:
        conn = _connect(V1_DBS['onchain'])
        if conn is None:
            return {'daily': pd.DataFrame(), 'timestamped': pd.DataFrame()}

        blockchain = _safe_read_sql(conn, """
            SELECT date, n_transactions, hash_rate, difficulty,
                   mempool_size, miners_revenue
            FROM blockchain_data ORDER BY date
        """)

        onchain = _safe_read_sql(conn, """
            SELECT timestamp, block_height, block_dr, funding_rate,
                   funding_dr, open_interest, oi_dr, fear_greed, fg_dr,
                   mempool_size, whale_volume_btc,
                   liq_long_count, liq_short_count, liq_long_vol, liq_short_vol,
                   coinbase_premium
            FROM onchain_data ORDER BY timestamp
        """)

        conn.close()

        if not blockchain.empty:
            blockchain['date'] = pd.to_datetime(blockchain['date'], errors='coerce')
            blockchain = blockchain.drop_duplicates(subset='date', keep='last')
            blockchain = blockchain.set_index('date').sort_index()

        return {'daily': blockchain, 'timestamped': onchain}

    def load_macro_v1(self) -> pd.DataFrame:
        conn = _connect(V1_DBS['macro'])
        if conn is None:
            return pd.DataFrame()

        df = _safe_read_sql(conn, "SELECT * FROM macro_data ORDER BY date")
        conn.close()

        if df.empty:
            return df

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.drop_duplicates(subset='date', keep='last')
        df = df.set_index('date').sort_index()
        return df

    def load_astro_cache(self) -> dict:
        """Preload all astrology/ephemeris/auxiliary daily data from V1."""
        cache = {}

        # Ephemeris
        conn = _connect(V1_DBS['ephemeris'])
        if conn:
            df = _safe_read_sql(conn, "SELECT * FROM ephemeris ORDER BY date")
            conn.close()
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.drop_duplicates(subset='date', keep='last').set_index('date')
                cache['ephemeris'] = df

        # Astrology Full
        conn = _connect(V1_DBS['astrology'])
        if conn:
            df = _safe_read_sql(conn, "SELECT * FROM daily_astrology ORDER BY date")
            conn.close()
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.drop_duplicates(subset='date', keep='last').set_index('date')
                cache['astrology'] = df

        # Fear & Greed
        conn = _connect(V1_DBS['fear_greed'])
        if conn:
            df = _safe_read_sql(conn, "SELECT * FROM fear_greed ORDER BY date")
            conn.close()
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.drop_duplicates(subset='date', keep='last').set_index('date')
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                cache['fear_greed'] = df

        # Space weather
        conn = _connect(V1_DBS['space_weather'])
        if conn:
            df = _safe_read_sql(conn, "SELECT * FROM space_weather ORDER BY date")
            conn.close()
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.drop_duplicates(subset='date', keep='last').set_index('date')
                cache['space_weather'] = df

        # Also load kp_history.txt (from V1 or V2)
        for kp_dir in [V2_DIR, V1_DIR]:
            kp_path = os.path.join(kp_dir, 'kp_history.txt')
            if os.path.exists(kp_path):
                cache['kp_history_path'] = kp_path
                break

        # Google Trends
        conn = _connect(V1_DBS['google_trends'])
        if conn:
            df = _safe_read_sql(conn, "SELECT * FROM google_trends ORDER BY date")
            conn.close()
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.drop_duplicates(subset='date', keep='last').set_index('date')
                cache['google_trends'] = df

        # Funding rates
        conn = _connect(V1_DBS['funding'])
        if conn:
            df = _safe_read_sql(conn, """
                SELECT timestamp, funding_rate FROM funding_rates
                WHERE symbol='BTCUSDT' ORDER BY timestamp
            """)
            conn.close()
            if not df.empty:
                df['date'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.date
                df = df.dropna(subset=['date'])
                df['date'] = pd.to_datetime(df['date'])
                daily = df.groupby('date')['funding_rate'].mean().to_frame('avg_funding_rate')
                cache['funding_daily'] = daily

        # V2 signals
        defi = self.load_defi_tvl()
        if not defi.empty:
            cache['defi_tvl'] = defi

        dominance = self.load_btc_dominance()
        if not dominance.empty:
            cache['btc_dominance'] = dominance

        mining = self.load_mining_stats()
        if not mining.empty:
            cache['mining_stats'] = mining

        cot = self.load_cot_report()
        if not cot.empty:
            cache['cot'] = cot

        # FRED macro
        fred = self.load_macro_fred()
        if not fred.empty:
            cache['fred'] = fred

        # Inverse signals for correlation features
        inverse = self.load_inverse_signals()
        if inverse:
            cache['inverse_signals'] = inverse

        return cache

    def load_all_data_for_asset(self, symbol: str, tf: str) -> dict:
        """
        Load ALL data needed to build features for a given asset + timeframe.
        Returns dict ready for feature_library_v2.build_all_features().
        """
        validate_required_dbs()

        data = {
            'ohlcv': self.load_ohlcv(symbol, tf),
            'symbol': symbol,
            'tf': tf,
            'astro_cache': self.load_astro_cache(),
        }

        # ALL assets get ALL esoteric data — the matrix is universal.
        # Same sky, same tweets, same energy = same features for every asset.
        # SPY under the same full moon with the same caution tweet = same signal.
        # Only on-chain/funding is crypto-specific (NaN for stocks, LightGBM handles it).
        data['tweets'] = self.load_tweets()
        data['news'] = self.load_news()
        data['sports'] = self.load_sports()
        data['onchain'] = self.load_onchain()
        data['macro'] = self.load_macro_v1()

        # HTF data for lower TFs
        if tf in ('5m', '15m', '1h', '4h'):
            htf_map = {'5m': ['15m','1h','4h','1d','1w'],
                       '15m': ['1h','4h','1d','1w'],
                       '1h': ['4h','1d','1w'],
                       '4h': ['1d','1w']}
            data['htf_data'] = {}
            for htf in htf_map.get(tf, []):
                htf_df = self.load_ohlcv(symbol, htf)
                if not htf_df.empty:
                    data['htf_data'][htf] = htf_df

        return data


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == '__main__':
    loader = V2OfflineDataLoader()

    print("=== V2 Data Access Test ===\n")

    # Test multi-asset
    for sym in ['SPY', 'BTC', 'GLD', 'SOL']:
        df = loader.load_ohlcv(sym, '1d')
        if not df.empty:
            print(f"{sym}: {len(df)} daily bars, {df.index[0].date()} to {df.index[-1].date()}")
        else:
            print(f"{sym}: no data")

    # Test V2 signals
    defi = loader.load_defi_tvl()
    print(f"\nDeFi TVL: {len(defi)} days" if not defi.empty else "\nDeFi TVL: no data")

    mining = loader.load_mining_stats()
    print(f"Mining stats: {len(mining)} days" if not mining.empty else "Mining stats: no data")

    # Test FRED
    fred = loader.load_macro_fred()
    if not fred.empty:
        print(f"FRED macro: {len(fred)} days, series: {list(fred.columns)}")

    # Test astro cache
    cache = loader.load_astro_cache()
    print(f"\nAstro cache keys: {list(cache.keys())}")

    # Test full data load
    data = loader.load_all_data_for_asset('SPY', '1d')
    print(f"\nSPY full load: OHLCV={len(data['ohlcv'])} bars, "
          f"astro_cache keys={list(data['astro_cache'].keys())}")
