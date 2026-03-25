"""
Twitter/X scraper using auth cookies.
Pulls full tweet history with timestamps for decode accounts.
"""
import os
import sys
import io
import json
import re
import time
import sqlite3
import requests
from datetime import datetime

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    from config import DB_DIR as _DB_DIR
except ImportError:
    _DB_DIR = PROJECT_DIR

# Full cookie jar from Chrome
COOKIES = {
    'auth_token': 'e2bc3e408900712dd67f24b9ca891a73b90d4528',
    'ct0': 'ea1139f64a08220ab4f4a35105e21cc0b2f2350b635681d8b6aa7d8c8aece75cce374a644d0c12d6ad3cbfc76661383a3dbb45ed6230bb0a321f18228c5e8805819f2ad86d46892726b6944fd547bbb4',
    'twid': 'u%3A1975391583835938817',
    'guest_id': 'v1%3A175980493553275703',
    'dnt': '1',
    'kdt': 'cP8wOf9TEjl7NhLRI64X0JbSuafQBXPvj50dySpv',
    'd_prefs': 'MjoxLGNvbnNlbnRfdmVyc2lvbjoyLHRleHRfdmVyc2lvbjoxMDAw',
    'personalization_id': 'v1_fNyLjVqhEEGuG2K1yIi/mg==',
    'lang': 'en',
}

# Target accounts to scrape
ACCOUNTS = [
    "elonmusk",
    "JoelKatz",        # David Schwartz / Ripple CTO
    "jack",            # Jack Dorsey
    "IAmSteveHarvey",  # Steve Harvey
    "tyler",           # Tyler Winklevoss
    "cameron",         # Cameron Winklevoss
    "IOHK_Charles",    # Charles Hoskinson
    "paabortrader",    # Tabor Trader (XRP)
]

HEADERS = {
    "authorization": "Bearer AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs%3D1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA",
    "x-csrf-token": COOKIES['ct0'],
    "x-twitter-active-user": "yes",
    "x-twitter-auth-type": "OAuth2Session",
    "x-twitter-client-language": "en",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "accept": "*/*",
    "accept-language": "en-US,en;q=0.9",
    "origin": "https://x.com",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
}


# ============================================================
# COLOR DETECTION
# ============================================================

# Emoji sets per color
_GOLD_EMOJIS = set('🟡💛⭐🌟✨🏆🥇👑💰🔑')
_RED_EMOJIS  = set('🔴❤️🟥❌🚨⛔🔻📉')
_GREEN_EMOJIS = set('🟢💚✅🟩📈🌿')
_BLUE_EMOJIS = set('🔵💙🟦')

# Whole-word text patterns per color (compiled once)
_GOLD_RE  = re.compile(r'\b(gold|golden)\b', re.IGNORECASE)
_RED_RE   = re.compile(r'\b(red|danger|warning|blood)\b', re.IGNORECASE)
_GREEN_RE = re.compile(r'\b(green|bullish)\b', re.IGNORECASE)
_BLUE_RE  = re.compile(r'\b(blue)\b', re.IGNORECASE)


def detect_tweet_colors(full_text, media_urls_json=None):
    """
    Detect color signals from tweet text (emojis + keywords).
    Returns dict with has_gold, has_red, has_green, dominant_colors.
    """
    text = full_text or ''

    # Check emojis (iterate characters; multi-codepoint emojis handled by set membership)
    chars = set(text)
    has_gold  = 1 if (chars & _GOLD_EMOJIS) or _GOLD_RE.search(text) else 0
    has_red   = 1 if (chars & _RED_EMOJIS) or _RED_RE.search(text) else 0
    has_green = 1 if (chars & _GREEN_EMOJIS) or _GREEN_RE.search(text) else 0
    has_blue  = 1 if (chars & _BLUE_EMOJIS) or _BLUE_RE.search(text) else 0

    # Build dominant_colors string
    colors = []
    if has_gold:  colors.append('gold')
    if has_red:   colors.append('red')
    if has_green: colors.append('green')
    if has_blue:  colors.append('blue')

    return {
        'has_gold': has_gold,
        'has_red': has_red,
        'has_green': has_green,
        'dominant_colors': ','.join(colors) if colors else None,
    }


def ensure_color_columns(conn):
    """Add color columns to tweets table if they don't exist."""
    existing = set()
    for row in conn.execute("PRAGMA table_info(tweets)").fetchall():
        existing.add(row[1])

    new_cols = {
        'has_gold': 'INTEGER DEFAULT 0',
        'has_red': 'INTEGER DEFAULT 0',
        'has_green': 'INTEGER DEFAULT 0',
        'dominant_colors': 'TEXT',
    }
    for col, typedef in new_cols.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE tweets ADD COLUMN {col} {typedef}")
    conn.commit()


def init_db():
    db_path = os.path.join(_DB_DIR, "tweets.db")
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
            reply_to_user TEXT,
            reply_to_tweet TEXT,
            quote_tweet_id TEXT,
            lang TEXT,
            day_of_year INTEGER,
            date_gematria TEXT,
            has_gold INTEGER DEFAULT 0,
            has_red INTEGER DEFAULT 0,
            has_green INTEGER DEFAULT 0,
            dominant_colors TEXT
        )
    """)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS tweets_fts USING fts5(
            full_text,
            content='tweets',
            content_rowid='rowid',
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


def get_user_id(handle):
    """Get Twitter user ID from handle."""
    vars_str = json.dumps({"screen_name": handle, "withGrokTranslatedBio": True})
    feats_str = json.dumps({
        "hidden_profile_subscriptions_enabled": True,
        "profile_label_improvements_pcf_label_in_post_enabled": True,
        "responsive_web_profile_redirect_enabled": False,
        "rweb_tipjar_consumption_enabled": False,
        "verified_phone_label_enabled": False,
        "subscriptions_verification_info_is_identity_verified_enabled": True,
        "responsive_web_graphql_skip_user_profile_image_extensions_enabled": False,
        "responsive_web_graphql_timeline_navigation_enabled": True,
    })
    url = f"https://x.com/i/api/graphql/pLsOiyHJ1eFwPJlNmLp4Bg/UserByScreenName?variables={requests.utils.quote(vars_str)}&features={requests.utils.quote(feats_str)}"
    hdrs = dict(HEADERS)
    hdrs["referer"] = f"https://x.com/{handle}"
    resp = requests.get(url, headers=hdrs, cookies=COOKIES)
    if resp.status_code == 200:
        try:
            return resp.json()["data"]["user"]["result"]["rest_id"]
        except (KeyError, TypeError):
            return None
    return None


TWEET_FEATURES = {"rweb_video_screen_enabled":False,"profile_label_improvements_pcf_label_in_post_enabled":True,"responsive_web_profile_redirect_enabled":False,"rweb_tipjar_consumption_enabled":False,"verified_phone_label_enabled":False,"creator_subscriptions_tweet_preview_api_enabled":True,"responsive_web_graphql_timeline_navigation_enabled":True,"responsive_web_graphql_skip_user_profile_image_extensions_enabled":False,"premium_content_api_read_enabled":False,"communities_web_enable_tweet_community_results_fetch":True,"c9s_tweet_anatomy_moderator_badge_enabled":True,"responsive_web_grok_analyze_button_fetch_trends_enabled":False,"responsive_web_grok_analyze_post_followups_enabled":True,"responsive_web_jetfuel_frame":True,"responsive_web_grok_share_attachment_enabled":True,"responsive_web_grok_annotations_enabled":True,"articles_preview_enabled":True,"responsive_web_edit_tweet_api_enabled":True,"graphql_is_translatable_rweb_tweet_is_translatable_enabled":True,"view_counts_everywhere_api_enabled":True,"longform_notetweets_consumption_enabled":True,"responsive_web_twitter_article_tweet_consumption_enabled":True,"tweet_awards_web_tipping_enabled":False,"content_disclosure_indicator_enabled":True,"content_disclosure_ai_generated_indicator_enabled":True,"responsive_web_grok_show_grok_translated_post":True,"responsive_web_grok_analysis_button_from_backend":True,"post_ctas_fetch_enabled":True,"freedom_of_speech_not_reach_fetch_enabled":True,"standardized_nudges_misinfo":True,"tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled":True,"longform_notetweets_rich_text_read_enabled":True,"longform_notetweets_inline_media_enabled":False,"responsive_web_grok_image_annotation_enabled":True,"responsive_web_grok_imagine_annotation_enabled":True,"responsive_web_grok_community_note_auto_translation_is_enabled":False,"responsive_web_enhance_cards_enabled":False}


def get_user_tweets(user_id, handle="", cursor=None, count=20):
    """Fetch tweets from a user's timeline."""
    variables = {
        "userId": user_id,
        "count": count,
        "includePromotedContent": False,
        "withQuickPromoteEligibilityTweetFields": True,
        "withVoice": True,
    }
    if cursor:
        variables["cursor"] = cursor

    url = f"https://x.com/i/api/graphql/Y59DTUMfcKmUAATiT2SlTw/UserTweets?variables={requests.utils.quote(json.dumps(variables))}&features={requests.utils.quote(json.dumps(TWEET_FEATURES))}&fieldToggles={requests.utils.quote(json.dumps({'withArticlePlainText':False}))}"
    hdrs = dict(HEADERS)
    hdrs["referer"] = f"https://x.com/{handle}"
    resp = requests.get(url, headers=hdrs, cookies=COOKIES)
    return resp


def calculate_date_numerology(dt):
    """Calculate date numerology values."""
    day_of_year = dt.timetuple().tm_yday
    days_remaining = 365 - day_of_year + (1 if dt.year % 4 == 0 else 0)

    # Date reduction
    date_sum = dt.month + dt.day + sum(int(d) for d in str(dt.year))
    while date_sum > 9 and date_sum not in (11, 22, 33):
        date_sum = sum(int(d) for d in str(date_sum))

    return {
        "day_of_year": day_of_year,
        "days_remaining": days_remaining,
        "date_reduction": date_sum,
        "month_day": f"{dt.month}/{dt.day}",
    }


def parse_tweet_data(tweet_result):
    """Extract tweet data from GraphQL response."""
    try:
        tweet = tweet_result.get("tweet", tweet_result)
        legacy = tweet.get("legacy", {})
        core = tweet.get("core", {}).get("user_results", {}).get("result", {})
        user_legacy = core.get("legacy", {})

        created_at = legacy.get("created_at", "")
        if created_at:
            dt = datetime.strptime(created_at, "%a %b %d %H:%M:%S +0000 %Y")
            ts_unix = int(dt.timestamp())
            numerology = calculate_date_numerology(dt)
        else:
            dt = None
            ts_unix = 0
            numerology = {}

        # Extract media URLs
        media_urls = []
        for media in legacy.get("entities", {}).get("media", []):
            media_urls.append(media.get("media_url_https", ""))

        full_text = legacy.get("full_text", "")
        media_urls_json = json.dumps(media_urls) if media_urls else None

        # Detect color signals from text + emojis
        colors = detect_tweet_colors(full_text, media_urls_json)

        return {
            "tweet_id": legacy.get("id_str", tweet.get("rest_id", "")),
            "user_handle": user_legacy.get("screen_name", "") or core.get("legacy", {}).get("screen_name", ""),
            "user_name": user_legacy.get("name", "") or core.get("legacy", {}).get("name", ""),
            "created_at": created_at,
            "ts_unix": ts_unix,
            "full_text": full_text,
            "retweet_count": legacy.get("retweet_count", 0),
            "favorite_count": legacy.get("favorite_count", 0),
            "reply_count": legacy.get("reply_count", 0),
            "media_urls": media_urls_json,
            "is_retweet": 1 if full_text.startswith("RT @") else 0,
            "is_reply": 1 if legacy.get("in_reply_to_status_id_str") else 0,
            "reply_to_user": legacy.get("in_reply_to_screen_name"),
            "reply_to_tweet": legacy.get("in_reply_to_status_id_str"),
            "lang": legacy.get("lang", ""),
            "day_of_year": numerology.get("day_of_year", 0),
            "date_gematria": json.dumps(numerology) if numerology else None,
            "has_gold": colors["has_gold"],
            "has_red": colors["has_red"],
            "has_green": colors["has_green"],
            "dominant_colors": colors["dominant_colors"],
        }
    except Exception as e:
        return None


def scrape_account(handle, conn, max_tweets=5000):
    """Scrape all tweets from an account."""
    print(f"\n  Scraping @{handle}...")

    user_id = get_user_id(handle)
    if not user_id:
        print(f"    Could not find user ID for @{handle}")
        return 0

    print(f"    User ID: {user_id}")

    cursor = None
    total = 0
    consecutive_empty = 0

    while total < max_tweets:
        try:
            resp = get_user_tweets(user_id, handle=handle, cursor=cursor)

            if resp.status_code == 429:
                import random
                wait = random.uniform(30, 45)
                print(f"    Rate limited, waiting {wait:.0f}s... ({total} tweets so far)")
                time.sleep(wait)
                continue

            if resp.status_code != 200:
                print(f"    HTTP {resp.status_code}, waiting 10s...")
                time.sleep(10)
                consecutive_empty += 1
                if consecutive_empty > 3:
                    break
                continue

            data = resp.json()

            # Navigate the GraphQL response (timeline, not timeline_v2)
            instructions = (data.get("data", {})
                          .get("user", {})
                          .get("result", {})
                          .get("timeline", {})
                          .get("timeline", {})
                          .get("instructions", []))

            tweets_found = 0
            next_cursor = None

            for instruction in instructions:
                entries = instruction.get("entries", [])
                if not entries and "entry" in instruction:
                    entries = [instruction["entry"]]
                if not entries:
                    entries = instruction.get("moduleItems", [])

                for entry in entries:
                    # Check for cursor
                    if entry.get("entryId", "").startswith("cursor-bottom"):
                        next_cursor = entry.get("content", {}).get("value")
                        continue

                    # Extract tweet
                    content = entry.get("content", {})
                    tweet_result = (content.get("itemContent", {})
                                  .get("tweet_results", {})
                                  .get("result", {}))

                    if not tweet_result or tweet_result.get("__typename") == "TweetTombstone":
                        continue

                    tweet_data = parse_tweet_data(tweet_result)
                    if tweet_data and tweet_data["tweet_id"]:
                        # Fallback handle to the account we're scraping
                        if not tweet_data["user_handle"]:
                            tweet_data["user_handle"] = handle
                        try:
                            conn.execute("""
                                INSERT OR IGNORE INTO tweets
                                (tweet_id, user_handle, user_name, created_at, ts_unix,
                                 full_text, retweet_count, favorite_count, reply_count,
                                 media_urls, is_retweet, is_reply, reply_to_user,
                                 reply_to_tweet, lang, day_of_year, date_gematria,
                                 has_gold, has_red, has_green, dominant_colors)
                                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
                                tweet_data["has_gold"], tweet_data["has_red"],
                                tweet_data["has_green"], tweet_data["dominant_colors"],
                            ))
                            tweets_found += 1
                        except Exception:
                            pass

            conn.commit()
            total += tweets_found

            if tweets_found > 0 and total % 100 < 20:
                print(f"    {total} tweets...")

            if not next_cursor or tweets_found == 0:
                consecutive_empty += 1
                if consecutive_empty > 2:
                    break
            else:
                consecutive_empty = 0
                cursor = next_cursor

            import random
            delay = random.uniform(2, 4)
            time.sleep(delay)

        except Exception as e:
            print(f"    Error: {e}")
            time.sleep(5)
            consecutive_empty += 1
            if consecutive_empty > 3:
                break

    print(f"    Done: {total} tweets from @{handle}")
    return total


def scrape_all():
    conn = init_db()
    ensure_color_columns(conn)
    print("=== Twitter/X Scraper — SEQUENTIAL (shared cookies) ===\n", flush=True)

    for handle in ACCOUNTS:
        count = scrape_account(handle, conn)
        print(f"  === @{handle}: {count} tweets ===\n", flush=True)

    # Final stats
    print(f"\n=== Scrape Complete ===", flush=True)
    total = conn.execute("SELECT COUNT(*) FROM tweets").fetchone()[0]
    print(f"Total tweets: {total}", flush=True)
    rows = conn.execute("""
        SELECT user_handle, COUNT(*), MIN(created_at), MAX(created_at)
        FROM tweets GROUP BY user_handle ORDER BY COUNT(*) DESC
    """).fetchall()
    for handle, count, first, last in rows:
        print(f"  @{handle:<20} {count:>6} tweets  ({first[:10] if first else '?'} to {last[:10] if last else '?'})", flush=True)
    conn.close()


if __name__ == "__main__":
    scrape_all()
