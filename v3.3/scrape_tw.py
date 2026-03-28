"""
Twitter scraper using twscrape library.
Scrapes all target accounts in parallel.
"""
import asyncio
import sys
import io
import os
import json
import sqlite3
from datetime import datetime

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from twscrape import API, gather
from twscrape.logger import set_log_level

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

ACCOUNTS = ['elonmusk', 'JoelKatz', 'jack', 'IAmSteveHarvey', 'tyler', 'cameron', 'IOHK_Charles']

# Load cookies from twitter_cookies.json or environment
_cookies_path = os.path.join(PROJECT_DIR, 'twitter_cookies.json')
if os.path.exists(_cookies_path):
    with open(_cookies_path) as _f:
        COOKIES = json.load(_f)
else:
    COOKIES = {
        "auth_token": os.environ.get("TWITTER_AUTH_TOKEN", ""),
        "ct0": os.environ.get("TWITTER_CT0", ""),
    }
    if not COOKIES["auth_token"]:
        print("WARNING: No twitter_cookies.json found and no TWITTER_AUTH_TOKEN env var set.")


def init_db():
    db_path = os.path.join(PROJECT_DIR, "tweets.db")
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tweets (
            tweet_id TEXT PRIMARY KEY,
            user_handle TEXT,
            user_name TEXT,
            created_at TEXT,
            ts_unix INTEGER,
            full_text TEXT,
            retweet_count INTEGER,
            favorite_count INTEGER,
            reply_count INTEGER,
            media_urls TEXT,
            is_retweet INTEGER DEFAULT 0,
            is_reply INTEGER DEFAULT 0,
            day_of_year INTEGER,
            date_gematria TEXT
        )
    """)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS tweets_fts USING fts5(
            full_text, content='tweets', content_rowid='rowid',
            tokenize='porter unicode61'
        )
    """)
    conn.execute("""
        CREATE TRIGGER IF NOT EXISTS tweets_ai AFTER INSERT ON tweets BEGIN
            INSERT INTO tweets_fts(rowid, full_text) VALUES (new.rowid, new.full_text);
        END
    """)
    conn.commit()
    return conn


def calc_numerology(dt):
    doy = dt.timetuple().tm_yday
    rem = 366 - doy if dt.year % 4 == 0 else 365 - doy
    s = dt.month + dt.day + sum(int(d) for d in str(dt.year))
    while s > 9 and s not in (11, 22, 33): s = sum(int(d) for d in str(s))
    return json.dumps({"day_of_year": doy, "days_remaining": rem, "reduction": s})


async def main():
    set_log_level("WARNING")
    api = API()

    # Add account with cookies as string format
    cookie_str = "; ".join(f"{k}={v}" for k, v in COOKIES.items())
    await api.pool.add_account(
        username="scraper_session",
        password="unused",
        email="unused@unused.com",
        email_password="unused",
        cookies=cookie_str,
    )
    await api.pool.set_active("scraper_session", True)

    conn = init_db()
    print("=== Twitter Scraper (twscrape) ===\n")

    for handle in ACCOUNTS:
        print(f"  Scraping @{handle}...", flush=True)
        try:
            user = await api.user_by_login(handle)
            if not user:
                print(f"    User not found")
                continue

            count = 0
            async for tweet in api.user_tweets(user.id, limit=3200):
                dt = tweet.date
                text = tweet.rawContent or ""
                media = [m.url for m in (tweet.media or {}).get("photos", [])] if tweet.media else []

                conn.execute("""
                    INSERT OR IGNORE INTO tweets
                    (tweet_id, user_handle, user_name, created_at, ts_unix, full_text,
                     retweet_count, favorite_count, reply_count, media_urls,
                     is_retweet, is_reply, day_of_year, date_gematria)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    str(tweet.id), handle, user.displayname,
                    dt.isoformat() if dt else "",
                    int(dt.timestamp()) if dt else 0,
                    text,
                    tweet.retweetCount or 0, tweet.likeCount or 0, tweet.replyCount or 0,
                    json.dumps(media) if media else None,
                    1 if text.startswith("RT @") else 0,
                    1 if tweet.inReplyToTweetId else 0,
                    dt.timetuple().tm_yday if dt else 0,
                    calc_numerology(dt) if dt else None,
                ))
                count += 1
                if count % 100 == 0:
                    conn.commit()
                    print(f"    {count} tweets...", flush=True)

            conn.commit()
            print(f"    Done: {count} tweets from @{handle}")

        except Exception as e:
            print(f"    Error @{handle}: {e}")

    # Stats
    print(f"\n=== Complete ===")
    total = conn.execute("SELECT COUNT(*) FROM tweets").fetchone()[0]
    print(f"Total: {total} tweets")
    for row in conn.execute("SELECT user_handle, COUNT(*) FROM tweets GROUP BY user_handle ORDER BY COUNT(*) DESC"):
        print(f"  @{row[0]:<20} {row[1]:>5}")
    conn.close()


if __name__ == "__main__":
    asyncio.run(main())
