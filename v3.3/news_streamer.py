#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
news_streamer.py -- Continuous live news poller
================================================
Polls RSS feeds, CryptoPanic, and Reddit for new articles every few minutes.
Computes gematria, sentiment, and date numerology on each headline using
the universal engines. Stores everything in news_articles.db with dedup.

Reuses data sources from news_collector.py.

Run as: python news_streamer.py
"""

import io
import os
import re
import sys
import time
import sqlite3
import hashlib
import logging
from datetime import datetime, timezone

if os.name == 'nt' and sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT_DIR, "news_articles.db")

# ---------------------------------------------------------------------------
# Imports from news_collector (feeds, API endpoints, headers)
# news_collector.py lives in root dir (parent of v2/)
# ---------------------------------------------------------------------------
import sys as _sys
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in _sys.path:
    _sys.path.insert(0, _parent)
from news_collector import (
    RSS_FEEDS,
    REDDIT_SUBS,
    HEADERS,
)

import requests
import feedparser

# ---------------------------------------------------------------------------
# Universal engines
# ---------------------------------------------------------------------------
from universal_gematria import gematria, digital_root
from universal_sentiment import sentiment
from universal_numerology import date_numerology

# ---------------------------------------------------------------------------
# Logging (mirrors tweet_streamer.py style)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_DIR, "news_streamer.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Poll intervals (seconds)
# ---------------------------------------------------------------------------
RSS_INTERVAL = 300        # 5 minutes
CRYPTOPANIC_INTERVAL = 300  # 5 minutes
REDDIT_INTERVAL = 600     # 10 minutes

# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS streamer_articles (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    source                  TEXT,
    title                   TEXT,
    url                     TEXT,
    title_hash              TEXT UNIQUE,
    published_at            TEXT,
    ts_unix                 INTEGER,
    title_gematria_ordinal  INTEGER,
    title_gematria_reverse  INTEGER,
    title_gematria_reduction INTEGER,
    title_dr                INTEGER,
    sentiment_score         INTEGER,
    sentiment_bull          INTEGER,
    sentiment_bear          INTEGER,
    has_caps                INTEGER,
    exclamation_count       INTEGER,
    word_count              INTEGER,
    inserted_at             TEXT DEFAULT (datetime('now'))
)
"""


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(CREATE_TABLE_SQL)
    conn.commit()
    return conn


def title_hash(title: str) -> str:
    """SHA-256 of lowered/stripped title for dedup."""
    return hashlib.sha256(title.strip().lower().encode("utf-8")).hexdigest()


def already_exists(conn, t_hash: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM streamer_articles WHERE title_hash = ?", (t_hash,)
    ).fetchone()
    return row is not None


# ---------------------------------------------------------------------------
# Enrichment: compute gematria + sentiment + date DR for one article
# ---------------------------------------------------------------------------

def enrich(title: str, published_at: str | None, ts_unix: int | None) -> dict:
    """Return computed columns for a single headline."""
    gem = gematria(title)
    sent = sentiment(title)

    # Date DR from the article's publish time (fall back to now)
    if ts_unix:
        dt = datetime.fromtimestamp(ts_unix, tz=timezone.utc)
    elif published_at:
        try:
            dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        except Exception:
            dt = datetime.now(timezone.utc)
    else:
        dt = datetime.now(timezone.utc)

    return {
        "title_gematria_ordinal": gem["ordinal"],
        "title_gematria_reverse": gem["reverse"],
        "title_gematria_reduction": gem["reduction"],
        "title_dr": digital_root(gem["ordinal"]),
        "sentiment_score": sent["score"],
        "sentiment_bull": sent["bull_count"],
        "sentiment_bear": sent["bear_count"],
        "has_caps": int(sent["has_caps"]),
        "exclamation_count": sent["exclamation"],
        "word_count": sent["word_count"],
    }


def insert_article(conn, source, title, url, published_at, ts_unix, enriched):
    """Insert one article if not already present. Returns True if inserted."""
    t_hash = title_hash(title)
    if already_exists(conn, t_hash):
        return False
    try:
        conn.execute("""
            INSERT OR IGNORE INTO streamer_articles
            (source, title, url, title_hash, published_at, ts_unix,
             title_gematria_ordinal, title_gematria_reverse, title_gematria_reduction,
             title_dr, sentiment_score, sentiment_bull, sentiment_bear,
             has_caps, exclamation_count, word_count)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            source, title, url, t_hash, published_at, ts_unix,
            enriched["title_gematria_ordinal"],
            enriched["title_gematria_reverse"],
            enriched["title_gematria_reduction"],
            enriched["title_dr"],
            enriched["sentiment_score"],
            enriched["sentiment_bull"],
            enriched["sentiment_bear"],
            enriched["has_caps"],
            enriched["exclamation_count"],
            enriched["word_count"],
        ))
        return True
    except sqlite3.IntegrityError:
        return False


# ---------------------------------------------------------------------------
# Source pollers
# ---------------------------------------------------------------------------

def poll_rss(conn) -> int:
    """Fetch all RSS feeds and insert new articles. Returns count of new."""
    total_new = 0
    for name, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url, agent=HEADERS["User-Agent"])
            new_count = 0
            for entry in feed.entries:
                title = entry.get("title", "").strip()
                if not title:
                    continue

                # Parse timestamp
                ts_unix = None
                published_at = None
                for attr in ("published_parsed", "updated_parsed"):
                    parsed = getattr(entry, attr, None)
                    if parsed:
                        dt = datetime(*parsed[:6], tzinfo=timezone.utc)
                        published_at = dt.isoformat()
                        ts_unix = int(dt.timestamp())
                        break

                link = entry.get("link", "")
                enriched = enrich(title, published_at, ts_unix)
                if insert_article(conn, f"rss_{name}", title, link, published_at, ts_unix, enriched):
                    new_count += 1
                    log.info(f"  NEW [{name}] {title[:90]}")

            if new_count:
                log.info(f"  rss_{name}: {new_count} new article(s)")
            total_new += new_count
        except Exception as e:
            log.error(f"  rss_{name}: {e}")
    conn.commit()
    return total_new


def poll_cryptopanic(conn) -> int:
    """Fetch CryptoPanic public API. Returns count of new articles."""
    total_new = 0
    try:
        url = "https://cryptopanic.com/api/free/v1/posts/?public=true&filter=hot"
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            for post in data.get("results", []):
                title = post.get("title", "").strip()
                if not title:
                    continue
                pub = post.get("published_at", "")
                ts_unix = None
                if pub:
                    try:
                        dt = datetime.fromisoformat(pub.replace("Z", "+00:00"))
                        ts_unix = int(dt.timestamp())
                    except Exception:
                        pass
                link = post.get("url", "")
                enriched = enrich(title, pub, ts_unix)
                if insert_article(conn, "cryptopanic", title, link, pub, ts_unix, enriched):
                    total_new += 1
                    log.info(f"  NEW [cryptopanic] {title[:90]}")
        else:
            log.warning(f"  cryptopanic: HTTP {resp.status_code}")
    except Exception as e:
        log.error(f"  cryptopanic: {e}")
    conn.commit()
    return total_new


def poll_reddit(conn) -> int:
    """Fetch Reddit hot posts. Returns count of new articles."""
    total_new = 0
    for sub in REDDIT_SUBS:
        try:
            url = f"https://www.reddit.com/r/{sub}/hot.json?limit=100"
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                new_count = 0
                for child in data.get("data", {}).get("children", []):
                    post = child.get("data", {})
                    title = post.get("title", "").strip()
                    if not title:
                        continue
                    created = post.get("created_utc", 0)
                    ts_unix = int(created) if created else None
                    published_at = (
                        datetime.fromtimestamp(created, tz=timezone.utc).isoformat()
                        if created else None
                    )
                    link = f"https://reddit.com{post.get('permalink', '')}"
                    enriched = enrich(title, published_at, ts_unix)
                    if insert_article(conn, f"reddit_r/{sub}", title, link, published_at, ts_unix, enriched):
                        new_count += 1
                        log.info(f"  NEW [r/{sub}] {title[:90]}")
                if new_count:
                    log.info(f"  reddit r/{sub}: {new_count} new post(s)")
                total_new += new_count
            elif resp.status_code == 429:
                log.warning(f"  reddit r/{sub}: rate limited, will retry next cycle")
            else:
                log.warning(f"  reddit r/{sub}: HTTP {resp.status_code}")
        except Exception as e:
            log.error(f"  reddit r/{sub}: {e}")
        time.sleep(2)  # polite delay between subreddits
    conn.commit()
    return total_new


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_streamer():
    log.info("=" * 60)
    log.info("  NEWS STREAMER -- Continuous Live Poller")
    log.info(f"  RSS interval:        {RSS_INTERVAL}s")
    log.info(f"  CryptoPanic interval: {CRYPTOPANIC_INTERVAL}s")
    log.info(f"  Reddit interval:     {REDDIT_INTERVAL}s")
    log.info(f"  DB: {DB_PATH}")
    log.info("=" * 60)

    conn = init_db()

    # Show current row count
    row_count = conn.execute("SELECT COUNT(*) FROM streamer_articles").fetchone()[0]
    log.info(f"  Existing articles in DB: {row_count}")
    log.info("  Starting continuous poll loop...\n")

    last_rss = 0.0
    last_cryptopanic = 0.0
    last_reddit = 0.0

    while True:
        try:
            now = time.time()

            # --- RSS feeds (every 5 min) ---
            if now - last_rss >= RSS_INTERVAL:
                log.info("[RSS] Polling feeds...")
                new = poll_rss(conn)
                log.info(f"[RSS] Done -- {new} new article(s)")
                last_rss = time.time()

            # --- CryptoPanic (every 5 min) ---
            if now - last_cryptopanic >= CRYPTOPANIC_INTERVAL:
                log.info("[CryptoPanic] Polling...")
                new = poll_cryptopanic(conn)
                log.info(f"[CryptoPanic] Done -- {new} new article(s)")
                last_cryptopanic = time.time()

            # --- Reddit (every 10 min) ---
            if now - last_reddit >= REDDIT_INTERVAL:
                log.info("[Reddit] Polling subreddits...")
                new = poll_reddit(conn)
                log.info(f"[Reddit] Done -- {new} new post(s)")
                last_reddit = time.time()

            # Sleep before next check (30s tick)
            time.sleep(30)

        except KeyboardInterrupt:
            log.info("\n  Shutting down news streamer...")
            break
        except Exception as e:
            log.error(f"  Streamer error: {e}", exc_info=True)
            time.sleep(30)

    conn.close()
    log.info("  News streamer stopped.")


if __name__ == "__main__":
    run_streamer()
