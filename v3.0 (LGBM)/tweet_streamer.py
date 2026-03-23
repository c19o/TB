#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tweet_streamer.py -- Continuous live tweet poller
=================================================
Polls target Twitter/X accounts every 5 minutes for new tweets.
Inserts into tweets.db with gematria, sentiment, color detection, engagement metrics.
Reuses auth cookies and parsing from scrape_twitter.py.

Run as:
  python tweet_streamer.py           # continuous polling
  python tweet_streamer.py --once    # single poll cycle, then exit
  python tweet_streamer.py --backfill  # fill missing gematria/sentiment on existing rows
"""
import os
import sys
import io
import time
import sqlite3
import logging
import argparse
from datetime import datetime, timezone

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Import shared functions from scrape_twitter
from scrape_twitter import (
    ACCOUNTS, COOKIES, HEADERS, TWEET_FEATURES,
    init_db, get_user_id, get_user_tweets, parse_tweet_data,
)

# Import gematria and sentiment engines
from universal_gematria import gematria, ordinal, english
from universal_sentiment import sentiment

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_DIR, 'tweet_streamer.log'), encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

# Poll interval per account (seconds)
POLL_INTERVAL = 300  # 5 minutes

# Cache user IDs to avoid re-fetching
user_id_cache = {}


def ensure_sentiment_columns(conn):
    """Add sentiment columns to tweets table if they don't exist."""
    existing = set()
    for row in conn.execute("PRAGMA table_info(tweets)").fetchall():
        existing.add(row[1])

    new_cols = {
        'sentiment_score': 'INTEGER DEFAULT 0',
        'sentiment_bull': 'INTEGER DEFAULT 0',
        'sentiment_bear': 'INTEGER DEFAULT 0',
        'sentiment_caps': 'INTEGER DEFAULT 0',
        'sentiment_urgency': 'INTEGER DEFAULT 0',
    }
    for col, typedef in new_cols.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE tweets ADD COLUMN {col} {typedef}")
            log.info(f"  Added column: {col}")
    conn.commit()


def compute_gematria_sentiment(text):
    """Compute gematria and sentiment for a tweet text. Returns flat dict."""
    # Strip non-ASCII to avoid unicode edge cases in gematria ciphers
    ascii_text = ''.join(c for c in (text or '') if ord(c) < 128)
    gem_simple = ordinal(ascii_text) if ascii_text else 0
    gem_eng = english(ascii_text) if ascii_text else 0

    s = sentiment(text) if text else {'score': 0, 'bull_count': 0, 'bear_count': 0, 'caps_words': 0, 'urgency': 0}

    return {
        'gematria_simple': gem_simple,
        'gematria_english': gem_eng,
        'sentiment_score': s.get('score', 0),
        'sentiment_bull': s.get('bull_count', 0),
        'sentiment_bear': s.get('bear_count', 0),
        'sentiment_caps': s.get('caps_words', 0),
        'sentiment_urgency': s.get('urgency', 0),
    }


def get_cached_user_id(handle):
    if handle not in user_id_cache:
        uid = get_user_id(handle)
        if uid:
            user_id_cache[handle] = uid
    return user_id_cache.get(handle)


def get_latest_tweet_ts(conn, handle):
    """Get the latest tweet timestamp for an account."""
    row = conn.execute(
        "SELECT MAX(ts_unix) FROM tweets WHERE user_handle = ?", (handle,)
    ).fetchone()
    return row[0] if row and row[0] else 0


def poll_account(handle, conn):
    """Check for new tweets from an account. Returns count of new tweets."""
    uid = get_cached_user_id(handle)
    if not uid:
        log.warning(f"  Could not resolve @{handle}")
        return 0

    latest_ts = get_latest_tweet_ts(conn, handle)

    try:
        resp = get_user_tweets(uid, handle=handle, count=20)

        if resp.status_code == 429:
            log.warning(f"  @{handle}: Rate limited, will retry next cycle")
            return -1  # Signal rate limit

        if resp.status_code != 200:
            log.warning(f"  @{handle}: HTTP {resp.status_code}")
            return 0

        data = resp.json()
        instructions = (data.get("data", {})
                       .get("user", {})
                       .get("result", {})
                       .get("timeline", {})
                       .get("timeline", {})
                       .get("instructions", []))

        new_count = 0
        for instruction in instructions:
            entries = instruction.get("entries", [])
            if not entries and "entry" in instruction:
                entries = [instruction["entry"]]

            for entry in entries:
                if entry.get("entryId", "").startswith("cursor-"):
                    continue

                content = entry.get("content", {})
                tweet_result = (content.get("itemContent", {})
                               .get("tweet_results", {})
                               .get("result", {}))

                if not tweet_result or tweet_result.get("__typename") == "TweetTombstone":
                    continue

                tweet_data = parse_tweet_data(tweet_result)
                if not tweet_data or not tweet_data["tweet_id"]:
                    continue

                if not tweet_data["user_handle"]:
                    tweet_data["user_handle"] = handle

                # Only insert if newer than latest
                if tweet_data["ts_unix"] <= latest_ts:
                    continue

                # Compute gematria + sentiment before storing
                gs = compute_gematria_sentiment(tweet_data["full_text"])

                try:
                    conn.execute("""
                        INSERT OR IGNORE INTO tweets
                        (tweet_id, user_handle, user_name, created_at, ts_unix,
                         full_text, retweet_count, favorite_count, reply_count,
                         media_urls, is_retweet, is_reply, reply_to_user,
                         reply_to_tweet, lang, day_of_year, date_gematria,
                         gematria_simple, gematria_english,
                         sentiment_score, sentiment_bull, sentiment_bear,
                         sentiment_caps, sentiment_urgency)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """, (
                        tweet_data["tweet_id"], tweet_data["user_handle"],
                        tweet_data["user_name"], tweet_data["created_at"],
                        tweet_data["ts_unix"], tweet_data["full_text"],
                        tweet_data["retweet_count"], tweet_data["favorite_count"],
                        tweet_data["reply_count"], tweet_data["media_urls"],
                        tweet_data["is_retweet"], tweet_data["is_reply"],
                        tweet_data["reply_to_user"], tweet_data["reply_to_tweet"],
                        tweet_data["lang"], tweet_data["day_of_year"],
                        tweet_data["date_gematria"],
                        gs["gematria_simple"], gs["gematria_english"],
                        gs["sentiment_score"], gs["sentiment_bull"],
                        gs["sentiment_bear"], gs["sentiment_caps"],
                        gs["sentiment_urgency"],
                    ))
                    new_count += 1
                    log.info(f"  NEW @{handle}: gem={gs['gematria_simple']} sent={gs['sentiment_score']} | {tweet_data['full_text'][:70]}...")
                except Exception:
                    pass

        conn.commit()
        return new_count

    except Exception as e:
        log.error(f"  @{handle}: Error: {e}")
        return 0


def backfill_gematria_sentiment(conn):
    """Fill gematria and sentiment for all existing tweets that are missing them."""
    log.info("  Backfilling gematria + sentiment on existing tweets...")

    rows = conn.execute("""
        SELECT tweet_id, full_text FROM tweets
        WHERE (gematria_simple IS NULL OR gematria_simple = 0
               OR sentiment_score IS NULL)
        AND full_text IS NOT NULL AND full_text != ''
    """).fetchall()

    log.info(f"  {len(rows)} tweets need backfill")

    updated = 0
    for tweet_id, text in rows:
        gs = compute_gematria_sentiment(text)
        conn.execute("""
            UPDATE tweets SET
                gematria_simple = ?,
                gematria_english = ?,
                sentiment_score = ?,
                sentiment_bull = ?,
                sentiment_bear = ?,
                sentiment_caps = ?,
                sentiment_urgency = ?
            WHERE tweet_id = ?
        """, (
            gs["gematria_simple"], gs["gematria_english"],
            gs["sentiment_score"], gs["sentiment_bull"],
            gs["sentiment_bear"], gs["sentiment_caps"],
            gs["sentiment_urgency"],
            tweet_id,
        ))
        updated += 1
        if updated % 1000 == 0:
            conn.commit()
            log.info(f"  Backfilled {updated}/{len(rows)}...")

    conn.commit()
    log.info(f"  Backfill complete: {updated} tweets updated")
    return updated


def run_once(conn):
    """Single poll cycle across all accounts, then exit."""
    log.info("=" * 60)
    log.info("  TWEET STREAMER -- Single Poll (--once)")
    log.info(f"  Accounts: {len(ACCOUNTS)}")
    log.info("=" * 60)

    total_new = 0
    for handle in ACCOUNTS:
        new = poll_account(handle, conn)
        if new > 0:
            total_new += new
            log.info(f"  @{handle}: {new} new tweet(s)")
        elif new == -1:
            log.warning(f"  @{handle}: rate limited")
        else:
            log.info(f"  @{handle}: no new tweets")
        time.sleep(2)

    log.info(f"  Single poll complete: {total_new} new tweets total")
    return total_new


def run_streamer(conn):
    log.info("=" * 60)
    log.info("  TWEET STREAMER -- Live Poller")
    log.info(f"  Accounts: {len(ACCOUNTS)}")
    log.info(f"  Poll interval: {POLL_INTERVAL}s per account")
    log.info("=" * 60)

    # Show current state
    for handle in ACCOUNTS:
        latest = get_latest_tweet_ts(conn, handle)
        if latest:
            dt = datetime.fromtimestamp(latest, tz=timezone.utc)
            log.info(f"  @{handle}: latest tweet {dt.strftime('%Y-%m-%d %H:%M')}")
        else:
            log.info(f"  @{handle}: no tweets in DB")

    log.info(f"\n  Starting continuous poll loop...\n")

    account_idx = 0
    rate_limited_until = {}

    while True:
        try:
            handle = ACCOUNTS[account_idx % len(ACCOUNTS)]
            now = time.time()

            # Skip if rate limited recently
            if handle in rate_limited_until and now < rate_limited_until[handle]:
                account_idx += 1
                time.sleep(1)
                continue

            new = poll_account(handle, conn)
            if new == -1:
                # Rate limited -- back off 60s for this account
                rate_limited_until[handle] = now + 60
            elif new > 0:
                log.info(f"  @{handle}: {new} new tweet(s)")

            account_idx += 1

            # After cycling through all accounts, wait for next interval
            if account_idx % len(ACCOUNTS) == 0:
                # Stagger: sleep = total interval / num accounts
                per_account_delay = POLL_INTERVAL / len(ACCOUNTS)
                time.sleep(per_account_delay)
            else:
                # Small delay between accounts to avoid hammering
                time.sleep(3)

        except KeyboardInterrupt:
            log.info("\n  Shutting down tweet streamer...")
            break
        except Exception as e:
            log.error(f"  Streamer error: {e}")
            time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tweet Streamer")
    parser.add_argument("--once", action="store_true", help="Single poll cycle then exit")
    parser.add_argument("--backfill", action="store_true", help="Backfill gematria/sentiment on existing rows")
    args = parser.parse_args()

    conn = init_db()
    ensure_sentiment_columns(conn)

    if args.backfill:
        backfill_gematria_sentiment(conn)
    elif args.once:
        run_once(conn)
    else:
        run_streamer(conn)

    conn.close()
