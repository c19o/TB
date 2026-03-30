# Twitter/X Scraper — Technical Guide

## What Works (as of March 2026)

**Method: Python `requests` library with full Chrome cookie jar + correct GraphQL endpoint hashes.**

Everything else failed:
- `twscrape` — broken, can't parse X's current JS
- `snscrape` — deprecated, unreliable on modern X
- Playwright headless — Cloudflare blocks it
- Playwright headed — cookies not accepted in fresh browser context
- Python `requests` with only 2 cookies — Cloudflare blocks it
- Chrome extension JS — times out on long-running scripts

## Why This Method Works

Twitter's web app makes GraphQL API calls from the browser. When you replicate:
1. **ALL cookies** from Chrome (not just auth_token + ct0)
2. **The correct GraphQL operation hashes** (these change when Twitter deploys new code)
3. **Browser-like headers** (origin, referer, sec-fetch-*, user-agent)

...then Twitter's API treats the request like a normal browser tab.

## How to Get Cookies

1. Install "Cookie-Editor" Chrome extension
2. Go to x.com while logged in
3. Export all cookies
4. Extract these values into the script:
   - `auth_token` (required — your session)
   - `ct0` (required — CSRF token)
   - `twid` (required — your user ID)
   - `guest_id` (helps avoid detection)
   - `dnt` (do not track flag)
   - `kdt` (session key)
   - `d_prefs` (preferences)
   - `personalization_id` (tracking ID)
   - `lang` (language)

**Cookies expire!** The `auth_token` lasts ~1 year but `ct0` rotates. If scraper stops working, re-export cookies from Chrome.

## How to Get GraphQL Hashes

These change when Twitter deploys new code. To find current hashes:

1. Open Chrome DevTools (F12) → Network tab
2. Go to any Twitter profile
3. Filter by "graphql" in the network filter
4. Find these requests:
   - `UserByScreenName` — hash is in the URL path
   - `UserTweets` — hash is in the URL path

Current hashes (March 2026):
- UserByScreenName: `pLsOiyHJ1eFwPJlNmLp4Bg`
- UserTweets: `Y59DTUMfcKmUAATiT2SlTw`

Also copy the `features` JSON from the UserTweets request — Twitter validates these.

## The Headers That Matter

```python
HEADERS = {
    "authorization": "Bearer AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs%3D...",
    "x-csrf-token": COOKIES['ct0'],
    "x-twitter-active-user": "yes",
    "x-twitter-auth-type": "OAuth2Session",
    "x-twitter-client-language": "en",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36...",
    "accept": "*/*",
    "origin": "https://x.com",
    "referer": "https://x.com/{handle}",  # MUST match the account you're scraping
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
}
```

The `authorization` Bearer token is Twitter's public web app token — same for all users, rarely changes.

## Response Structure

Twitter's GraphQL response for UserTweets:
```
data.user.result.timeline.timeline.instructions[]
  → entries[] or entry
    → content.itemContent.tweet_results.result
      → legacy (tweet data: full_text, created_at, entities, etc.)
      → core.user_results.result.legacy (user data: screen_name, name)
```

**NOT** `timeline_v2` — they changed this. If your scraper returns 0 tweets, check this path first.

## Cursor Pagination

Each response includes a `cursor-bottom` entry:
```
entry.entryId starts with "cursor-bottom"
entry.content.value = "the_cursor_string"
```

Pass this as `variables.cursor` in the next request to get older tweets.
Stop when: no cursor returned, no new tweets, or 3 consecutive empty pages.

## Rate Limits

- ~2-4 second delay between requests is safe for a single account
- 15-30 second delay if being extra cautious
- On 429 (rate limit): wait 30-45 seconds
- Max ~3,200 tweets accessible per account (Twitter's limit)
- Running 7+ accounts in parallel threads works fine with same cookies

## Live Scraping (Future — Interval-Based)

To scrape new tweets on a schedule:

```python
# live_scraper.py — run as a cron job or scheduled task
import schedule
import time

def check_new_tweets():
    """Check each account for tweets newer than our last stored tweet."""
    conn = sqlite3.connect('tweets.db')
    for handle in ACCOUNTS:
        # Get our most recent tweet timestamp for this account
        last_ts = conn.execute(
            "SELECT MAX(ts_unix) FROM tweets WHERE user_handle = ?", (handle,)
        ).fetchone()[0]

        # Fetch first page of tweets (most recent)
        user_id = get_user_id(handle)
        resp = get_user_tweets(user_id, handle=handle)
        # Parse and insert only tweets newer than last_ts
        # ... (same parsing logic as main scraper)

    conn.close()

# Run every 5 minutes
schedule.every(5).minutes.do(check_new_tweets)
while True:
    schedule.run_pending()
    time.sleep(1)
```

### Setting Up as Windows Scheduled Task:
```
schtasks /create /tn "TwitterScraper" /tr "python C:\Users\C\Documents\Savage22 Server\live_scraper.py" /sc minute /mo 5
```

### Or as a background service:
```bash
# Run in background
nohup python live_scraper.py &

# Or use screen/tmux on Linux
screen -dmS scraper python live_scraper.py
```

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| 403 Cloudflare page | Missing cookies or headers | Re-export ALL cookies from Chrome, check sec-fetch headers |
| 404 on GraphQL | Hash changed | Capture new hash from Chrome DevTools Network tab |
| 429 rate limit | Too many requests | Increase delay between requests |
| 0 tweets returned | Wrong response path | Check `timeline` vs `timeline_v2` in response structure |
| Empty user_handle | Parse bug | Check `core.user_results.result.legacy.screen_name` path |
| Cookies expired | Session timeout | Re-login to Twitter in Chrome, re-export cookies |

## File Locations

```
C:\Users\C\Documents\Savage22 Server\
  scrape_twitter.py      — main scraper (parallel, all accounts)
  tweets.db              — SQLite database of scraped tweets
  TWITTER_SCRAPER_GUIDE.md — this file
```

## Database Schema

```sql
tweets (
    tweet_id TEXT PRIMARY KEY,
    user_handle TEXT,
    user_name TEXT,
    created_at TEXT,        -- "Mon Mar 16 19:14:38 +0000 2026"
    ts_unix INTEGER,        -- Unix timestamp
    full_text TEXT,          -- Full tweet text
    retweet_count INTEGER,
    favorite_count INTEGER,
    reply_count INTEGER,
    media_urls TEXT,         -- JSON array of image URLs
    is_retweet INTEGER,
    is_reply INTEGER,
    reply_to_user TEXT,
    reply_to_tweet TEXT,
    quote_tweet_id TEXT,
    lang TEXT,
    day_of_year INTEGER,    -- For numerology calculations
    date_gematria TEXT      -- JSON with day_of_year, days_remaining, reduction
)

tweets_fts — FTS5 full-text search index on full_text
```
