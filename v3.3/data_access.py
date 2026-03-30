#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
data_access.py — Data Access Layer
====================================
Loads data from SQLite DBs for both offline (backfill) and live modes.
Returns DataFrames ready for feature_library.py — NO feature computation here.

Used by:
  - build_*_features.py  →  OfflineDataLoader (full history)
  - live_trader.py        →  LiveDataLoader (incremental cache)
"""

import os
import time
import sqlite3
import pandas as pd
from datetime import datetime, timedelta

DB_DIR = os.environ.get("SAVAGE22_DB_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# SHARED DB HELPERS
# ============================================================

def _connect(db_name, db_dir=None):
    """Open a read-only SQLite connection."""
    path = os.path.join(db_dir or DB_DIR, db_name)
    if not os.path.exists(path):
        return None
    return sqlite3.connect(path, timeout=10)


def _safe_read_sql(conn, query, params=None):
    """Read SQL with error handling. Returns empty DataFrame on failure."""
    try:
        return pd.read_sql_query(query, conn, params=params)
    except Exception:
        return pd.DataFrame()


# ============================================================
# OFFLINE DATA LOADER (full history for feature building)
# ============================================================

class OfflineDataLoader:
    """
    Loads full history from SQLite DBs for feature building.
    All methods return DataFrames ready for feature_library.py.
    """

    def __init__(self, db_dir: str = None):
        self.db_dir = db_dir or DB_DIR

    def load_ohlcv(self, tf: str, symbol: str = 'BTC/USDT') -> pd.DataFrame:
        """
        Load OHLCV candles for a given timeframe.

        Returns DataFrame with DatetimeIndex and columns:
            open_time, open, high, low, close, volume,
            quote_volume, trades, taker_buy_volume, taker_buy_quote
        """
        conn = _connect('btc_prices.db', self.db_dir)
        if conn is None:
            return pd.DataFrame()

        df = _safe_read_sql(conn, """
            SELECT open_time, open, high, low, close, volume,
                   quote_volume, trades, taker_buy_volume, taker_buy_quote
            FROM ohlcv WHERE timeframe=? AND symbol=?
            ORDER BY open_time
        """, (tf, symbol))
        conn.close()

        if df.empty:
            return df

        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
        df = df.drop_duplicates(subset='open_time', keep='last')
        df = df.set_index('timestamp').sort_index()

        for col in ['open', 'high', 'low', 'close', 'volume',
                     'quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop zero-volume leading bars
        if 'volume' in df.columns and len(df) > 0:
            first_good = df[df['volume'] > 0].index
            if len(first_good) > 0:
                df = df.loc[first_good[0]:]

        return df

    def load_tweets(self, start_ts: int = None) -> pd.DataFrame:
        """
        Load tweets. Returns DataFrame with columns:
            ts_unix, full_text, user_handle, favorite_count, retweet_count,
            reply_count, has_gold, has_red, gematria_simple, gematria_english
        """
        conn = _connect('tweets.db', self.db_dir)
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

        if df.empty:
            return df

        df['ts_unix'] = pd.to_numeric(df['ts_unix'], errors='coerce')
        df = df.dropna(subset=['ts_unix'])
        return df

    def load_news(self, start_ts: int = None) -> pd.DataFrame:
        """
        Load news articles. Tries streamer_articles first, falls back to articles.
        Returns DataFrame with: ts_unix, title, sentiment_score, title_dr, etc.
        """
        conn = _connect('news_articles.db', self.db_dir)
        if conn is None:
            return pd.DataFrame()

        # Try streamer_articles first (richer schema)
        where = f" WHERE ts_unix >= {start_ts}" if start_ts else ""
        df = _safe_read_sql(conn, f"""
            SELECT ts_unix, title, sentiment_score, title_dr,
                   title_gematria_ordinal, title_gematria_reverse,
                   title_gematria_reduction, sentiment_bull, sentiment_bear,
                   has_caps, exclamation_count, word_count
            FROM streamer_articles {where}
            ORDER BY ts_unix
        """)

        if df.empty:
            df = _safe_read_sql(conn, f"""
                SELECT ts_unix, title, sentiment_score, title_dr,
                       title_gematria, body_word_count, date_doy
                FROM articles {where}
                ORDER BY ts_unix
            """)

        conn.close()

        if df.empty:
            return df

        df['ts_unix'] = pd.to_numeric(df['ts_unix'], errors='coerce')
        df = df.dropna(subset=['ts_unix'])
        return df

    def load_sports(self, start_ts: int = None) -> dict:
        """
        Load sports results. Returns dict with 'games' and 'horse_races' DataFrames.
        """
        conn = _connect('sports_results.db', self.db_dir)
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

    def load_onchain(self, start_ts: int = None) -> dict:
        """
        Load on-chain data. Returns dict with 'daily' and 'timestamped' DataFrames.

        Returns:
            {
                'daily': DataFrame (date-indexed) from blockchain_data,
                'timestamped': DataFrame from onchain_data (has timestamp col),
            }
        """
        conn = _connect('onchain_data.db', self.db_dir)
        if conn is None:
            return {'daily': pd.DataFrame(), 'timestamped': pd.DataFrame()}

        # Load daily blockchain data
        blockchain = _safe_read_sql(conn, """
            SELECT date, n_transactions, hash_rate, difficulty,
                   mempool_size, miners_revenue
            FROM blockchain_data ORDER BY date
        """)

        # Load timestamped on-chain data (including whale/liq/premium cols)
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

    def load_macro(self, start_ts: int = None) -> pd.DataFrame:
        """
        Load macro data. Returns date-indexed DataFrame with:
            dxy, gold, spx, vix, us10y, nasdaq, russell, oil, silver, etc.
        """
        conn = _connect('macro_data.db', self.db_dir)
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
        """
        Preload all astrology/ephemeris/auxiliary daily data.
        Returns dict suitable for feature_library astro_cache parameter.
        """
        cache = {}

        # Ephemeris
        conn = _connect('ephemeris_cache.db', self.db_dir)
        if conn:
            df = _safe_read_sql(conn, "SELECT * FROM ephemeris ORDER BY date")
            conn.close()
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.drop_duplicates(subset='date', keep='last').set_index('date')
                cache['ephemeris'] = df

        # Astrology Full
        conn = _connect('astrology_full.db', self.db_dir)
        if conn:
            df = _safe_read_sql(conn, "SELECT * FROM daily_astrology ORDER BY date")
            conn.close()
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.drop_duplicates(subset='date', keep='last').set_index('date')
                cache['astrology'] = df

        # Fear & Greed
        conn = _connect('fear_greed.db', self.db_dir)
        if conn:
            df = _safe_read_sql(conn, "SELECT * FROM fear_greed ORDER BY date")
            conn.close()
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.drop_duplicates(subset='date', keep='last').set_index('date')
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                cache['fear_greed'] = df

        # Google Trends
        conn = _connect('google_trends.db', self.db_dir)
        if conn:
            df = _safe_read_sql(conn, "SELECT * FROM google_trends ORDER BY date")
            conn.close()
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.drop_duplicates(subset='date', keep='last').set_index('date')
                cache['google_trends'] = df

        # Funding rates (aggregate to daily)
        conn = _connect('funding_rates.db', self.db_dir)
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

        return cache


# ============================================================
# LIVE DATA LOADER (incremental cache for live trading)
# ============================================================

class LiveDataLoader:
    """
    Maintains in-memory caches, queries DBs incrementally.
    Call refresh_caches() every ~30s to pick up new data.
    """

    def __init__(self, db_dir: str = None):
        self.db_dir = db_dir or DB_DIR
        # V1 databases (tweets, news, sports, astro, etc.) live in the V1 parent dir,
        # not in the v3.1 project dir. Use config.py V1_DIR for those lookups.
        try:
            from config import V1_DIR
            self.v1_dir = V1_DIR
        except ImportError:
            self.v1_dir = DB_DIR  # fallback: module-level DB_DIR already points to V1 parent
        self._caches = {}       # per-source DataFrames
        self._last_seen = {}    # per-source last timestamp
        self._astro_cache = {}  # daily data (refreshed less often)
        self._astro_last_refresh = 0

    def _v1_connect(self, db_name):
        """Connect to a V1 database (tweets, news, sports, etc.).
        Tries v1_dir first, falls back to db_dir for backward compat."""
        conn = _connect(db_name, self.v1_dir)
        if conn is None and self.v1_dir != self.db_dir:
            conn = _connect(db_name, self.db_dir)
        return conn

    def refresh_caches(self):
        """
        Called every ~30s by live_trader.
        Queries each DB for new data since last_seen.
        """
        self._refresh_tweets()
        self._refresh_news()
        self._refresh_onchain()
        self._refresh_macro()
        self._refresh_sports()
        self._refresh_space_weather()

        # Refresh astro cache less often (daily data, check every 5 min)
        if time.time() - self._astro_last_refresh > 300:
            self._refresh_astro_cache()
            self._astro_last_refresh = time.time()

    def _refresh_tweets(self):
        """Incrementally load new tweets."""
        last = self._last_seen.get('tweets', 0)
        conn = self._v1_connect('tweets.db')
        if conn is None:
            return

        df = _safe_read_sql(conn, f"""
            SELECT created_at, ts_unix, user_handle, full_text,
                   has_gold, has_red, has_green, dominant_colors,
                   gematria_simple, gematria_english,
                   favorite_count, retweet_count, reply_count
            FROM tweets WHERE ts_unix > {last}
            ORDER BY ts_unix
        """)
        conn.close()

        if df.empty:
            return

        df['ts_unix'] = pd.to_numeric(df['ts_unix'], errors='coerce')
        df = df.dropna(subset=['ts_unix'])

        if 'tweets' in self._caches and len(self._caches['tweets']) > 0:
            self._caches['tweets'] = pd.concat([self._caches['tweets'], df], ignore_index=True)
        else:
            self._caches['tweets'] = df

        if len(df) > 0:
            self._last_seen['tweets'] = int(df['ts_unix'].max())

    def _refresh_news(self):
        """Incrementally load new news."""
        last = self._last_seen.get('news', 0)
        conn = self._v1_connect('news_articles.db')
        if conn is None:
            return

        # Try streamer_articles first
        df = _safe_read_sql(conn, f"""
            SELECT ts_unix, title, sentiment_score, title_dr,
                   title_gematria_ordinal, title_gematria_reverse,
                   title_gematria_reduction, sentiment_bull, sentiment_bear,
                   has_caps, exclamation_count, word_count
            FROM streamer_articles WHERE ts_unix > {last}
            ORDER BY ts_unix
        """)

        if df.empty:
            df = _safe_read_sql(conn, f"""
                SELECT ts_unix, title, sentiment_score, title_dr,
                       title_gematria, body_word_count
                FROM articles WHERE ts_unix > {last}
                ORDER BY ts_unix
            """)

        conn.close()

        if df.empty:
            return

        df['ts_unix'] = pd.to_numeric(df['ts_unix'], errors='coerce')
        df = df.dropna(subset=['ts_unix'])

        if 'news' in self._caches and len(self._caches['news']) > 0:
            self._caches['news'] = pd.concat([self._caches['news'], df], ignore_index=True)
        else:
            self._caches['news'] = df

        if len(df) > 0:
            self._last_seen['news'] = int(df['ts_unix'].max())

    def _refresh_onchain(self):
        """Incrementally load new on-chain data."""
        conn = self._v1_connect('onchain_data.db')
        if conn is None:
            return

        # Timestamped onchain data
        last = self._last_seen.get('onchain', '2000-01-01')
        df = _safe_read_sql(conn, f"""
            SELECT timestamp, block_height, block_dr, funding_rate,
                   funding_dr, open_interest, oi_dr, fear_greed, fg_dr,
                   mempool_size, whale_volume_btc,
                   liq_long_count, liq_short_count, liq_long_vol, liq_short_vol,
                   coinbase_premium
            FROM onchain_data WHERE timestamp > '{last}'
            ORDER BY timestamp
        """)

        # Daily blockchain data
        last_daily = self._last_seen.get('onchain_daily', '2000-01-01')
        blockchain = _safe_read_sql(conn, f"""
            SELECT date, n_transactions, hash_rate, difficulty,
                   mempool_size, miners_revenue
            FROM blockchain_data WHERE date > '{last_daily}'
            ORDER BY date
        """)

        conn.close()

        if not df.empty:
            if 'onchain' in self._caches and len(self._caches['onchain']) > 0:
                self._caches['onchain'] = pd.concat([self._caches['onchain'], df], ignore_index=True)
            else:
                self._caches['onchain'] = df
            self._last_seen['onchain'] = str(df['timestamp'].max())

        if not blockchain.empty:
            blockchain['date'] = pd.to_datetime(blockchain['date'], errors='coerce')
            blockchain = blockchain.drop_duplicates(subset='date', keep='last').set_index('date')
            if 'onchain_daily' in self._caches and len(self._caches['onchain_daily']) > 0:
                combined = pd.concat([self._caches['onchain_daily'], blockchain])
                self._caches['onchain_daily'] = combined[~combined.index.duplicated(keep='last')]
            else:
                self._caches['onchain_daily'] = blockchain
            self._last_seen['onchain_daily'] = str(blockchain.index.max())

    def _refresh_macro(self):
        """Incrementally load new macro data."""
        last = self._last_seen.get('macro', '2000-01-01')
        conn = self._v1_connect('macro_data.db')
        if conn is None:
            return

        df = _safe_read_sql(conn, f"""
            SELECT * FROM macro_data WHERE date > '{last}'
            ORDER BY date
        """)
        conn.close()

        if df.empty:
            return

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.drop_duplicates(subset='date', keep='last').set_index('date')

        if 'macro' in self._caches and len(self._caches['macro']) > 0:
            combined = pd.concat([self._caches['macro'], df])
            self._caches['macro'] = combined[~combined.index.duplicated(keep='last')]
        else:
            self._caches['macro'] = df

        if len(df) > 0:
            self._last_seen['macro'] = str(df.index.max().date())

    def _refresh_sports(self):
        """Incrementally load new sports data."""
        conn = self._v1_connect('sports_results.db')
        if conn is None:
            return

        last = self._last_seen.get('sports', '2000-01-01')
        games = _safe_read_sql(conn, f"""
            SELECT date, winner, home_score, away_score,
                   winner_gem_ordinal, winner_gem_dr, score_dr,
                   score_total_dr, is_upset, is_overtime
            FROM games WHERE date > '{last}'
            ORDER BY date
        """)

        horses = _safe_read_sql(conn, f"""
            SELECT date, winner_horse, horse_gem_ordinal, horse_gem_dr,
                   jockey_gem_ordinal, position_dr
            FROM horse_races WHERE date > '{last}'
            ORDER BY date
        """)

        conn.close()

        if not games.empty:
            if 'games' in self._caches and len(self._caches['games']) > 0:
                self._caches['games'] = pd.concat([self._caches['games'], games], ignore_index=True)
            else:
                self._caches['games'] = games
            self._last_seen['sports'] = str(games['date'].max())

        if not horses.empty:
            if 'horse_races' in self._caches and len(self._caches['horse_races']) > 0:
                self._caches['horse_races'] = pd.concat([self._caches['horse_races'], horses], ignore_index=True)
            else:
                self._caches['horse_races'] = horses

    def _refresh_space_weather(self):
        """Incrementally load new space weather data."""
        last = self._last_seen.get('space_weather', '2000-01-01')
        conn = self._v1_connect('space_weather.db')
        if conn is None:
            return

        df = _safe_read_sql(conn, f"""
            SELECT timestamp, kp_index, sunspot_number, solar_flux_f107,
                   solar_wind_speed, solar_wind_bz, r_scale, s_scale, g_scale
            FROM space_weather WHERE timestamp > '{last}'
            ORDER BY timestamp
        """)

        # Fallback: try simpler schema
        if df.empty:
            df = _safe_read_sql(conn, f"""
                SELECT * FROM space_weather WHERE timestamp > '{last}'
                ORDER BY timestamp
            """)

        conn.close()

        if df.empty:
            return

        if 'timestamp' in df.columns and len(df) > 0:
            self._last_seen['space_weather'] = str(df['timestamp'].max())

        # Convert timestamp column to DatetimeIndex for feature_library compatibility
        # (feature_library uses sw.reindex(df.index, method='ffill') which needs DatetimeIndex)
        if 'timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp'], unit='s', utc=True, errors='coerce')
            if df['date'].isna().all():
                df['date'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
            df = df.dropna(subset=['date'])
            if not df.empty:
                df['date'] = df['date'].dt.tz_localize(None)  # strip TZ for cuDF compat
                df = df.drop_duplicates(subset='date', keep='last').set_index('date').sort_index()

        if df.empty:
            return

        if 'space_weather' in self._caches and len(self._caches['space_weather']) > 0:
            combined = pd.concat([self._caches['space_weather'], df])
            self._caches['space_weather'] = combined[~combined.index.duplicated(keep='last')].sort_index()
        else:
            self._caches['space_weather'] = df

    def _refresh_astro_cache(self):
        """Refresh daily astro/ephemeris data (called less often)."""
        loader = OfflineDataLoader(self.v1_dir)
        self._astro_cache = loader.load_astro_cache()

    # ==========================================================
    # PUBLIC GETTERS
    # ==========================================================

    def get_ohlcv_window(self, tf: str, n_bars: int,
                         symbol: str = 'BTC/USDT') -> pd.DataFrame:
        """
        Get the most recent n_bars of OHLCV data.
        Queries DB directly (not cached) since candle data updates frequently.
        """
        conn = self._v1_connect('btc_prices.db')
        if conn is None:
            return pd.DataFrame()

        df = _safe_read_sql(conn, f"""
            SELECT open_time, open, high, low, close, volume,
                   quote_volume, trades, taker_buy_volume, taker_buy_quote
            FROM ohlcv WHERE timeframe=? AND symbol=?
            ORDER BY open_time DESC LIMIT ?
        """, (tf, symbol, n_bars))
        conn.close()

        if df.empty:
            return df

        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
        df = df.drop_duplicates(subset='open_time', keep='last')
        df = df.set_index('timestamp').sort_index()

        for col in ['open', 'high', 'low', 'close', 'volume',
                     'quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def get_tweets(self) -> pd.DataFrame:
        """Returns cached tweets DataFrame."""
        return self._caches.get('tweets', pd.DataFrame())

    def get_news(self) -> pd.DataFrame:
        """Returns cached news DataFrame."""
        return self._caches.get('news', pd.DataFrame())

    def get_sports(self) -> dict:
        """Returns dict with 'games' and 'horse_races' DataFrames."""
        return {
            'games': self._caches.get('games', pd.DataFrame()),
            'horse_races': self._caches.get('horse_races', pd.DataFrame()),
        }

    def get_onchain(self) -> dict:
        """
        Returns on-chain data as dict with 'daily' and 'timestamped' DataFrames.
        """
        return {
            'daily': self._caches.get('onchain_daily', pd.DataFrame()),
            'timestamped': self._caches.get('onchain', pd.DataFrame()),
        }

    def get_macro(self) -> pd.DataFrame:
        """Returns cached macro data (date-indexed)."""
        return self._caches.get('macro', pd.DataFrame())

    def get_space_weather(self) -> pd.DataFrame:
        """Returns cached space weather DataFrame (Kp index, solar wind, etc.)."""
        return self._caches.get('space_weather', pd.DataFrame())

    def get_astro_cache(self) -> dict:
        """Returns cached astro/ephemeris data."""
        if not self._astro_cache:
            self._refresh_astro_cache()
        return self._astro_cache

    def get_htf_ohlcv(self, base_tf: str, n_bars_map: dict = None) -> dict:
        """
        Get higher-TF OHLCV data for context features.

        Args:
            base_tf: the base timeframe ('5m', '15m', '1h', '4h')
            n_bars_map: override bars per TF, e.g. {'4h': 500, '1d': 200, '1w': 100}

        Returns:
            dict like {'4h': df, '1d': df, '1w': df}
        """
        defaults = {
            '5m':  {'15m': 400, '1h': 300, '4h': 200, '1d': 100, '1w': 50},
            '15m': {'1h': 300, '4h': 200, '1d': 100, '1w': 50},
            '1h':  {'4h': 200, '1d': 100, '1w': 50},
            '4h':  {'1d': 200, '1w': 50},
            '1d':  {'1w': 100},
            '1w':  {},
        }

        bars = n_bars_map or defaults.get(base_tf, {})
        result = {}

        for tf, n in bars.items():
            df = self.get_ohlcv_window(tf, n)
            if not df.empty:
                result[tf] = df

        return result

    def initial_load(self):
        """
        Perform full initial load of all caches on startup.
        Called once when live_trader starts.
        """
        # Load recent tweets (last 30 days)
        cutoff = int((datetime.utcnow() - timedelta(days=30)).timestamp())
        conn = self._v1_connect('tweets.db')
        if conn:
            df = _safe_read_sql(conn, f"""
                SELECT created_at, ts_unix, user_handle, full_text,
                       has_gold, has_red, has_green, dominant_colors,
                       gematria_simple, gematria_english,
                       favorite_count, retweet_count, reply_count
                FROM tweets WHERE ts_unix > {cutoff}
                ORDER BY ts_unix
            """)
            conn.close()
            if not df.empty:
                df['ts_unix'] = pd.to_numeric(df['ts_unix'], errors='coerce')
                df = df.dropna(subset=['ts_unix'])
                self._caches['tweets'] = df
                self._last_seen['tweets'] = int(df['ts_unix'].max())

        # Load recent news
        conn = self._v1_connect('news_articles.db')
        if conn:
            df = _safe_read_sql(conn, f"""
                SELECT ts_unix, title, sentiment_score, title_dr,
                       title_gematria_ordinal, title_gematria_reverse,
                       title_gematria_reduction, sentiment_bull, sentiment_bear,
                       has_caps, exclamation_count, word_count
                FROM streamer_articles WHERE ts_unix > {cutoff}
                ORDER BY ts_unix
            """)
            if df.empty:
                df = _safe_read_sql(conn, f"""
                    SELECT ts_unix, title, sentiment_score, title_dr
                    FROM articles WHERE ts_unix > {cutoff}
                    ORDER BY ts_unix
                """)
            conn.close()
            if not df.empty:
                df['ts_unix'] = pd.to_numeric(df['ts_unix'], errors='coerce')
                df = df.dropna(subset=['ts_unix'])
                self._caches['news'] = df
                self._last_seen['news'] = int(df['ts_unix'].max())

        # Load recent sports (last 30 days)
        cutoff_date = (datetime.utcnow() - timedelta(days=30)).strftime('%Y-%m-%d')
        conn = self._v1_connect('sports_results.db')
        if conn:
            games = _safe_read_sql(conn, f"""
                SELECT date, winner, home_score, away_score,
                       winner_gem_ordinal, winner_gem_dr, score_dr,
                       score_total_dr, is_upset, is_overtime
                FROM games WHERE date > '{cutoff_date}'
                ORDER BY date
            """)
            horses = _safe_read_sql(conn, f"""
                SELECT date, winner_horse, horse_gem_ordinal, horse_gem_dr,
                       jockey_gem_ordinal, position_dr
                FROM horse_races WHERE date > '{cutoff_date}'
                ORDER BY date
            """)
            conn.close()
            if not games.empty:
                self._caches['games'] = games
            if not horses.empty:
                self._caches['horse_races'] = horses
            self._last_seen['sports'] = cutoff_date

        # Load macro (full history — small dataset)
        conn = self._v1_connect('macro_data.db')
        if conn:
            df = _safe_read_sql(conn, "SELECT * FROM macro_data ORDER BY date")
            conn.close()
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.drop_duplicates(subset='date', keep='last').set_index('date')
                self._caches['macro'] = df
                self._last_seen['macro'] = str(df.index.max().date())

        # Load onchain
        conn = self._v1_connect('onchain_data.db')
        if conn:
            blockchain = _safe_read_sql(conn, """
                SELECT date, n_transactions, hash_rate, difficulty,
                       mempool_size, miners_revenue
                FROM blockchain_data ORDER BY date
            """)
            onchain = _safe_read_sql(conn, f"""
                SELECT timestamp, block_height, block_dr, funding_rate,
                       funding_dr, open_interest, oi_dr, fear_greed, fg_dr,
                       mempool_size, whale_volume_btc,
                       liq_long_count, liq_short_count, liq_long_vol, liq_short_vol,
                       coinbase_premium
                FROM onchain_data WHERE timestamp > '{cutoff_date}'
                ORDER BY timestamp
            """)
            conn.close()

            if not blockchain.empty:
                blockchain['date'] = pd.to_datetime(blockchain['date'], errors='coerce')
                blockchain = blockchain.drop_duplicates(subset='date', keep='last').set_index('date')
                self._caches['onchain_daily'] = blockchain

            if not onchain.empty:
                self._caches['onchain'] = onchain
                self._last_seen['onchain'] = str(onchain['timestamp'].max())

        # Load space weather (full history — small dataset)
        self._refresh_space_weather()

        # Load astro cache
        self._refresh_astro_cache()

        print(f"[LiveDataLoader] Initial load complete:")
        print(f"  tweets:  {len(self._caches.get('tweets', []))} rows")
        print(f"  news:    {len(self._caches.get('news', []))} rows")
        print(f"  games:   {len(self._caches.get('games', []))} rows")
        print(f"  horses:  {len(self._caches.get('horse_races', []))} rows")
        print(f"  macro:   {len(self._caches.get('macro', []))} rows")
        print(f"  onchain: {len(self._caches.get('onchain', []))} rows")
        print(f"  space_w: {len(self._caches.get('space_weather', []))} rows")
        print(f"  astro:   {len(self._astro_cache)} sub-caches")


if __name__ == '__main__':
    print("data_access.py — Data Access Layer")
    print(f"  DB_DIR: {DB_DIR}")
    print()

    # Quick test: offline loader
    loader = OfflineDataLoader()
    ohlcv = loader.load_ohlcv('1h')
    print(f"  1H candles: {len(ohlcv)} rows")
    tweets = loader.load_tweets()
    print(f"  Tweets: {len(tweets)} rows")
    news = loader.load_news()
    print(f"  News: {len(news)} rows")
    sports = loader.load_sports()
    print(f"  Games: {len(sports['games'])} rows")
    print(f"  Horse races: {len(sports['horse_races'])} rows")
    macro = loader.load_macro()
    print(f"  Macro: {len(macro)} rows")
    astro = loader.load_astro_cache()
    print(f"  Astro cache: {list(astro.keys())}")
