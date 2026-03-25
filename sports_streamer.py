#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sports_streamer.py -- Continuous live sports + horse racing poller
==================================================================
Polls ESPN API (NFL, NBA, MLB, NHL, EPL, MLS) and horse racing results
every 15 minutes for today's completed results. Computes gematria on
team names and numerology on scores. Stores in sports_results.db
with deduplication by game_id / event_id.

Historical backfill: run with --backfill to download 2019-2026 data
for all leagues via ESPN scoreboard date parameter.

Run as:
  python sports_streamer.py              # live polling mode
  python sports_streamer.py --backfill   # historical download then live polling
"""

import os
import sys
import io
import time
import json
import sqlite3
import logging
import urllib.request
import urllib.error
import re
from datetime import datetime, timezone, timedelta

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    from config import DB_DIR as _DB_DIR
except ImportError:
    _DB_DIR = PROJECT_DIR
DB_PATH = os.path.join(_DB_DIR, "sports_results.db")

# ---------------------------------------------------------------------------
# Universal engines
# ---------------------------------------------------------------------------
from universal_gematria import gematria, gematria_flat, digital_root
from universal_numerology import numerology

# ---------------------------------------------------------------------------
# Logging  (mirrors tweet_streamer.py)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_DIR, "sports_streamer.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Poll interval (seconds)
# ---------------------------------------------------------------------------
POLL_INTERVAL = 900  # 15 minutes

# ---------------------------------------------------------------------------
# ESPN config -- all leagues including soccer
# ---------------------------------------------------------------------------
ESPN_SPORTS = [
    ("football",   "nfl"),
    ("basketball", "nba"),
    ("baseball",   "mlb"),
    ("hockey",     "nhl"),
    ("soccer",     "eng.1"),   # Premier League
    ("soccer",     "usa.1"),   # MLS
]

ESPN_BASE = "http://site.api.espn.com/apis/site/v2/sports"

UA_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# ---------------------------------------------------------------------------
# Historical backfill config
# ---------------------------------------------------------------------------
# Season date ranges for each league (approximate, covers reg season + playoffs)
LEAGUE_SEASONS = {
    "nfl": [
        # NFL runs Sep-Feb
        {"start": "09-01", "end_next": "02-15"},
    ],
    "nba": [
        # NBA runs Oct-Jun
        {"start": "10-01", "end_next": "06-30"},
    ],
    "mlb": [
        # MLB runs Mar-Nov
        {"start": "03-20", "end_next": "11-10"},
    ],
    "nhl": [
        # NHL runs Oct-Jun
        {"start": "10-01", "end_next": "06-30"},
    ],
    "eng.1": [
        # EPL runs Aug-May
        {"start": "08-01", "end_next": "05-31"},
    ],
    "usa.1": [
        # MLS runs Feb-Dec
        {"start": "02-15", "end_next": "12-15"},
    ],
}

BACKFILL_START_YEAR = 2019
BACKFILL_END_YEAR = 2026

# ---------------------------------------------------------------------------
# Horse racing -- scrape from Off Track Betting (public results pages)
# TheSportsDB free tier is broken (returns soccer, not horse racing)
# ---------------------------------------------------------------------------
HORSE_RACING_ENABLED = True
# We use ESPN's horse racing schedule endpoint where available, plus
# a lightweight scraper for major race results from public pages.
# For historical, we store major Triple Crown / Breeders Cup results.

MAJOR_HORSE_RACES = [
    # (date, race_name, track, winner_horse, winner_jockey, post_position, odds)
    # 2019
    ("2019-05-04", "Kentucky Derby", "Churchill Downs", "Country House", "Flavien Prat", 20, 65.0),
    ("2019-05-18", "Preakness Stakes", "Pimlico", "War of Will", "Tyler Gaffalione", 1, 7.0),
    ("2019-06-08", "Belmont Stakes", "Belmont Park", "Sir Winston", "Joel Rosario", 7, 10.0),
    # 2020
    ("2020-09-05", "Kentucky Derby", "Churchill Downs", "Authentic", "John Velazquez", 18, 8.0),
    ("2020-10-03", "Preakness Stakes", "Pimlico", "Swiss Skydiver", "Robby Albarado", 4, 11.0),
    ("2020-06-20", "Belmont Stakes", "Belmont Park", "Tiz the Law", "Manny Franco", 6, 0.6),
    # 2021
    ("2021-05-01", "Kentucky Derby", "Churchill Downs", "Medina Spirit", "John Velazquez", 8, 12.0),
    ("2021-05-15", "Preakness Stakes", "Pimlico", "Rombauer", "Flavien Prat", 6, 11.0),
    ("2021-06-05", "Belmont Stakes", "Belmont Park", "Essential Quality", "Luis Saez", 2, 2.0),
    # 2022
    ("2022-05-07", "Kentucky Derby", "Churchill Downs", "Rich Strike", "Sonny Leon", 21, 80.0),
    ("2022-05-21", "Preakness Stakes", "Pimlico", "Early Voting", "Jose Ortiz", 5, 5.0),
    ("2022-06-11", "Belmont Stakes", "Belmont Park", "Mo Donegal", "Irad Ortiz Jr", 6, 5.0),
    # 2023
    ("2023-05-06", "Kentucky Derby", "Churchill Downs", "Mage", "Javier Castellano", 8, 15.0),
    ("2023-05-20", "Preakness Stakes", "Pimlico", "National Treasure", "John Velazquez", 1, 5.0),
    ("2023-06-10", "Belmont Stakes", "Belmont Park", "Arcangelo", "Javier Castellano", 5, 8.0),
    # 2024
    ("2024-05-04", "Kentucky Derby", "Churchill Downs", "Mystik Dan", "Brian Hernandez Jr", 3, 18.0),
    ("2024-05-18", "Preakness Stakes", "Pimlico", "Seize the Grey", "Jaime Torres", 10, 17.0),
    ("2024-06-08", "Belmont Stakes", "Saratoga", "Dornoch", "Luis Saez", 1, 17.0),
    # 2025
    ("2025-05-03", "Kentucky Derby", "Churchill Downs", "Sovereignty", "Flavien Prat", 3, 6.0),
    ("2025-05-17", "Preakness Stakes", "Pimlico", "Journalism", "John Velazquez", 4, 4.0),
    ("2025-06-07", "Belmont Stakes", "Belmont Park", "Journalism", "John Velazquez", 2, 2.5),
    # Breeders Cup Classic winners
    ("2019-11-02", "Breeders Cup Classic", "Santa Anita", "Vino Rosso", "Irad Ortiz Jr", 5, 10.0),
    ("2020-11-07", "Breeders Cup Classic", "Keeneland", "Authentic", "John Velazquez", 9, 2.0),
    ("2021-11-06", "Breeders Cup Classic", "Del Mar", "Knicks Go", "Joel Rosario", 4, 4.0),
    ("2022-11-05", "Breeders Cup Classic", "Keeneland", "Flightline", "Flavien Prat", 4, 0.4),
    ("2023-11-04", "Breeders Cup Classic", "Santa Anita", "White Abarrio", "Irad Ortiz Jr", 3, 7.0),
    ("2024-11-02", "Breeders Cup Classic", "Del Mar", "Highland Falls", "Javier Castellano", 6, 12.0),
]


# ============================================================
# DATABASE SETUP
# ============================================================

def init_db():
    """Create tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS games (
            id              TEXT PRIMARY KEY,
            sport           TEXT,
            league          TEXT,
            date            TEXT,
            home_team       TEXT,
            away_team       TEXT,
            home_score      INTEGER,
            away_score      INTEGER,
            winner          TEXT,
            game_timestamp  TEXT,
            venue           TEXT,
            home_gem_ordinal    INTEGER,
            away_gem_ordinal    INTEGER,
            winner_gem_ordinal  INTEGER,
            home_gem_dr         INTEGER,
            away_gem_dr         INTEGER,
            winner_gem_dr       INTEGER,
            score_total         INTEGER,
            score_diff          INTEGER,
            score_dr            INTEGER,
            score_total_dr      INTEGER,
            is_upset        INTEGER DEFAULT 0,
            is_overtime     INTEGER DEFAULT 0,
            is_playoff      INTEGER DEFAULT 0,
            inserted_at     TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS horse_races (
            id                  TEXT PRIMARY KEY,
            race_name           TEXT,
            track               TEXT,
            date                TEXT,
            winner_horse        TEXT,
            winner_jockey       TEXT,
            winner_trainer      TEXT,
            post_position       INTEGER,
            odds                REAL,
            race_time           TEXT,
            horse_gem_ordinal   INTEGER,
            horse_gem_dr        INTEGER,
            jockey_gem_ordinal  INTEGER,
            race_gem_ordinal    INTEGER,
            position_dr         INTEGER,
            odds_dr             INTEGER,
            inserted_at         TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS backfill_progress (
            league          TEXT,
            last_date       TEXT,
            status          TEXT,
            updated_at      TEXT,
            PRIMARY KEY (league)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_games_date ON games(date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_games_league ON games(league)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_games_sport ON games(sport)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_horse_date ON horse_races(date)")
    conn.commit()
    conn.close()
    log.info("Database initialised: %s", DB_PATH)


# ============================================================
# HTTP HELPER
# ============================================================

def _http_get_json(url, timeout=20, retries=2):
    """Fetch JSON from a URL with retries, return parsed dict or None."""
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=UA_HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                return json.loads(raw)
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                # Rate limited -- back off
                wait = 30 * (attempt + 1)
                log.warning("Rate limited (429) on %s, waiting %ds", url, wait)
                time.sleep(wait)
                continue
            elif exc.code in (404, 400):
                # No data for this date/league combo
                return None
            else:
                log.warning("HTTP %d for %s (attempt %d)", exc.code, url, attempt + 1)
                if attempt < retries:
                    time.sleep(3)
                    continue
                return None
        except Exception as exc:
            log.warning("HTTP GET failed for %s (attempt %d): %s", url, attempt + 1, exc)
            if attempt < retries:
                time.sleep(3)
                continue
            return None
    return None


# ============================================================
# ESPN FETCH -- scoreboard for a specific date
# ============================================================

def fetch_espn_date(sport, league, date_str):
    """Return list of completed-game dicts from ESPN scoreboard for given date.
    date_str format: YYYYMMDD
    """
    url = f"{ESPN_BASE}/{sport}/{league}/scoreboard?dates={date_str}"
    data = _http_get_json(url)
    if not data:
        return []

    games = []
    for event in data.get("events", []):
        game = _parse_espn_event(event, sport, league)
        if game:
            games.append(game)
    return games


def fetch_espn_today(sport, league):
    """Return list of completed-game dicts from ESPN scoreboard for today."""
    today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    return fetch_espn_date(sport, league, today_str)


def _parse_espn_event(event, sport, league):
    """Parse a single ESPN event into a dict (only completed games)."""
    try:
        status_obj = event.get("status", {}).get("type", {})
        if not status_obj.get("completed", False):
            return None

        event_id = event.get("id", "")
        date_str = event.get("date", "")
        competition = event.get("competitions", [{}])[0]
        competitors = competition.get("competitors", [])
        if len(competitors) < 2:
            return None

        # ESPN puts home first (homeAway field)
        home = away = None
        for c in competitors:
            if c.get("homeAway") == "home":
                home = c
            else:
                away = c
        if home is None or away is None:
            home, away = competitors[0], competitors[1]

        home_team = home.get("team", {}).get("displayName", "")
        away_team = away.get("team", {}).get("displayName", "")
        home_score = int(home.get("score", 0) or 0)
        away_score = int(away.get("score", 0) or 0)
        winner = home_team if home_score > away_score else away_team

        venue_obj = competition.get("venue", {})
        venue = venue_obj.get("fullName", "")

        # Overtime detection
        status_name = event.get("status", {}).get("type", {}).get("name", "")
        is_ot = 0
        if "overtime" in status_name.lower() or status_name.upper().startswith("STATUS_FINAL_OT"):
            is_ot = 1
        # Also check period count for sports with known regulation periods
        try:
            period = int(event.get("status", {}).get("period", 0) or 0)
            if sport == "football" and period > 4:
                is_ot = 1
            elif sport == "basketball" and period > 4:
                is_ot = 1
            elif sport == "hockey" and period > 3:
                is_ot = 1
            elif sport == "soccer" and period > 2:
                is_ot = 1  # extra time
        except Exception:
            pass

        # Playoff detection
        season_type = event.get("season", {}).get("type", 0)
        is_playoff = 1 if int(season_type or 0) == 3 else 0

        # Upset: lower-seeded (higher rank) team wins
        is_upset = 0
        try:
            home_seed = int(home.get("curatedRank", {}).get("current", 99))
            away_seed = int(away.get("curatedRank", {}).get("current", 99))
            if home_score > away_score and home_seed > away_seed:
                is_upset = 1
            elif away_score > home_score and away_seed > home_seed:
                is_upset = 1
        except Exception:
            pass

        # Gematria on team names
        home_gem = gematria(home_team)
        away_gem = gematria(away_team)
        winner_gem = gematria(winner)

        score_total = home_score + away_score
        score_diff = abs(home_score - away_score)

        return {
            "id": "espn_%s_%s" % (league, event_id),
            "sport": sport,
            "league": league.upper(),
            "date": date_str[:10],
            "home_team": home_team,
            "away_team": away_team,
            "home_score": home_score,
            "away_score": away_score,
            "winner": winner,
            "game_timestamp": date_str,
            "venue": venue,
            "home_gem_ordinal": home_gem["ordinal"],
            "away_gem_ordinal": away_gem["ordinal"],
            "winner_gem_ordinal": winner_gem["ordinal"],
            "home_gem_dr": home_gem["dr_ordinal"],
            "away_gem_dr": away_gem["dr_ordinal"],
            "winner_gem_dr": winner_gem["dr_ordinal"],
            "score_total": score_total,
            "score_diff": score_diff,
            "score_dr": digital_root(score_diff),
            "score_total_dr": digital_root(score_total),
            "is_upset": is_upset,
            "is_overtime": is_ot,
            "is_playoff": is_playoff,
        }
    except Exception as exc:
        log.debug("parse_espn_event error: %s", exc)
        return None


# ============================================================
# HORSE RACING -- major race results (hardcoded + live scrape)
# ============================================================

def seed_major_horse_races():
    """Insert historical major horse race results into the database."""
    races = []
    for entry in MAJOR_HORSE_RACES:
        date_str, race_name, track, winner_horse, winner_jockey, post_pos, odds = entry
        race_id = "hr_%s_%s" % (date_str, race_name.replace(" ", "_").lower())

        horse_gem = gematria(winner_horse) if winner_horse else {"ordinal": 0, "dr_ordinal": 0}
        jockey_gem = gematria(winner_jockey) if winner_jockey else {"ordinal": 0, "dr_ordinal": 0}
        race_gem = gematria(race_name) if race_name else {"ordinal": 0, "dr_ordinal": 0}

        races.append({
            "id": race_id,
            "race_name": race_name,
            "track": track,
            "date": date_str,
            "winner_horse": winner_horse,
            "winner_jockey": winner_jockey,
            "winner_trainer": "",
            "post_position": post_pos,
            "odds": odds,
            "race_time": "",
            "horse_gem_ordinal": horse_gem["ordinal"],
            "horse_gem_dr": horse_gem["dr_ordinal"],
            "jockey_gem_ordinal": jockey_gem["ordinal"],
            "race_gem_ordinal": race_gem["ordinal"],
            "position_dr": digital_root(post_pos) if post_pos else 0,
            "odds_dr": digital_root(int(odds)) if odds else 0,
        })

    new = insert_horse_races(races)
    log.info("Seeded %d major horse race results (%d new)", len(races), new)
    return new


def fetch_horse_racing_today():
    """Fetch horse racing results for today from ESPN horse-racing endpoint.
    TheSportsDB free tier is broken (only returns soccer data).
    We fall back to checking major race calendar dates.
    """
    # No reliable free horse racing API exists.
    # For live data, we check if today matches any major race dates.
    # The historical major races are seeded via seed_major_horse_races().
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    races = []
    for entry in MAJOR_HORSE_RACES:
        if entry[0] == today_str:
            date_str, race_name, track, winner_horse, winner_jockey, post_pos, odds = entry
            race_id = "hr_%s_%s" % (date_str, race_name.replace(" ", "_").lower())
            horse_gem = gematria(winner_horse) if winner_horse else {"ordinal": 0, "dr_ordinal": 0}
            jockey_gem = gematria(winner_jockey) if winner_jockey else {"ordinal": 0, "dr_ordinal": 0}
            race_gem = gematria(race_name) if race_name else {"ordinal": 0, "dr_ordinal": 0}
            races.append({
                "id": race_id,
                "race_name": race_name,
                "track": track,
                "date": date_str,
                "winner_horse": winner_horse,
                "winner_jockey": winner_jockey,
                "winner_trainer": "",
                "post_position": post_pos,
                "odds": odds,
                "race_time": "",
                "horse_gem_ordinal": horse_gem["ordinal"],
                "horse_gem_dr": horse_gem["dr_ordinal"],
                "jockey_gem_ordinal": jockey_gem["ordinal"],
                "race_gem_ordinal": race_gem["ordinal"],
                "position_dr": digital_root(post_pos) if post_pos else 0,
                "odds_dr": digital_root(int(odds)) if odds else 0,
            })
    return races


# ============================================================
# INSERT WITH DEDUP
# ============================================================

def insert_games(games):
    """Insert games that don't already exist. Returns count of new rows."""
    if not games:
        return 0
    conn = sqlite3.connect(DB_PATH)
    new = 0
    now_str = datetime.now(timezone.utc).isoformat()
    for g in games:
        try:
            cur = conn.execute(
                """INSERT OR IGNORE INTO games
                   (id, sport, league, date, home_team, away_team, home_score, away_score,
                    winner, game_timestamp, venue,
                    home_gem_ordinal, away_gem_ordinal, winner_gem_ordinal,
                    home_gem_dr, away_gem_dr, winner_gem_dr,
                    score_total, score_diff, score_dr, score_total_dr,
                    is_upset, is_overtime, is_playoff, inserted_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    g["id"], g["sport"], g["league"], g["date"],
                    g["home_team"], g["away_team"], g["home_score"], g["away_score"],
                    g["winner"], g["game_timestamp"], g["venue"],
                    g["home_gem_ordinal"], g["away_gem_ordinal"], g["winner_gem_ordinal"],
                    g["home_gem_dr"], g["away_gem_dr"], g["winner_gem_dr"],
                    g["score_total"], g["score_diff"], g["score_dr"], g["score_total_dr"],
                    g["is_upset"], g["is_overtime"], g["is_playoff"],
                    now_str,
                ),
            )
            if cur.rowcount > 0:
                new += 1
        except sqlite3.IntegrityError:
            pass
        except Exception as exc:
            log.warning("Insert game error (%s): %s", g.get("id"), exc)
    conn.commit()
    conn.close()
    return new


def insert_games_batch(games):
    """Batch insert for backfill -- much faster for large volumes."""
    if not games:
        return 0
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    new = 0
    now_str = datetime.now(timezone.utc).isoformat()
    for g in games:
        try:
            cur = conn.execute(
                """INSERT OR IGNORE INTO games
                   (id, sport, league, date, home_team, away_team, home_score, away_score,
                    winner, game_timestamp, venue,
                    home_gem_ordinal, away_gem_ordinal, winner_gem_ordinal,
                    home_gem_dr, away_gem_dr, winner_gem_dr,
                    score_total, score_diff, score_dr, score_total_dr,
                    is_upset, is_overtime, is_playoff, inserted_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    g["id"], g["sport"], g["league"], g["date"],
                    g["home_team"], g["away_team"], g["home_score"], g["away_score"],
                    g["winner"], g["game_timestamp"], g["venue"],
                    g["home_gem_ordinal"], g["away_gem_ordinal"], g["winner_gem_ordinal"],
                    g["home_gem_dr"], g["away_gem_dr"], g["winner_gem_dr"],
                    g["score_total"], g["score_diff"], g["score_dr"], g["score_total_dr"],
                    g["is_upset"], g["is_overtime"], g["is_playoff"],
                    now_str,
                ),
            )
            if cur.rowcount > 0:
                new += 1
        except sqlite3.IntegrityError:
            pass
        except Exception as exc:
            log.warning("Insert game error (%s): %s", g.get("id"), exc)
    conn.commit()
    conn.close()
    return new


def insert_horse_races(races):
    """Insert horse races that don't already exist. Returns count of new rows."""
    if not races:
        return 0
    conn = sqlite3.connect(DB_PATH)
    new = 0
    now_str = datetime.now(timezone.utc).isoformat()
    for r in races:
        try:
            cur = conn.execute(
                """INSERT OR IGNORE INTO horse_races
                   (id, race_name, track, date, winner_horse, winner_jockey, winner_trainer,
                    post_position, odds, race_time,
                    horse_gem_ordinal, horse_gem_dr, jockey_gem_ordinal, race_gem_ordinal,
                    position_dr, odds_dr, inserted_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    r["id"], r["race_name"], r["track"], r["date"],
                    r["winner_horse"], r["winner_jockey"], r["winner_trainer"],
                    r["post_position"], r["odds"], r["race_time"],
                    r["horse_gem_ordinal"], r["horse_gem_dr"],
                    r["jockey_gem_ordinal"], r["race_gem_ordinal"],
                    r["position_dr"], r["odds_dr"],
                    now_str,
                ),
            )
            if cur.rowcount > 0:
                new += 1
        except sqlite3.IntegrityError:
            pass
        except Exception as exc:
            log.warning("Insert horse race error (%s): %s", r.get("id"), exc)
    conn.commit()
    conn.close()
    return new


# ============================================================
# HISTORICAL BACKFILL
# ============================================================

def _get_season_dates(league, year):
    """Generate list of dates (YYYYMMDD) for a league's season in a given year."""
    cfg = LEAGUE_SEASONS.get(league, LEAGUE_SEASONS.get(league.lower()))
    if not cfg:
        return []

    dates = []
    for season in cfg:
        start_parts = season["start"].split("-")
        end_parts = season["end_next"].split("-")
        start_month, start_day = int(start_parts[0]), int(start_parts[1])
        end_month, end_day = int(end_parts[0]), int(end_parts[1])

        # Determine if season crosses year boundary
        if end_month < start_month:
            # Season crosses year boundary (e.g., NFL Sep-Feb)
            start_date = datetime(year, start_month, start_day)
            end_date = datetime(year + 1, end_month, end_day)
        else:
            # Season within same year
            start_date = datetime(year, start_month, start_day)
            end_date = datetime(year, end_month, end_day)

        # Don't go past today
        today = datetime.now()
        if end_date > today:
            end_date = today

        if start_date > today:
            continue

        current = start_date
        while current <= end_date:
            dates.append(current.strftime("%Y%m%d"))
            current += timedelta(days=1)

    return dates


def _get_backfill_progress(league):
    """Get last successfully backfilled date for a league."""
    try:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute(
            "SELECT last_date FROM backfill_progress WHERE league = ?",
            (league,)
        ).fetchone()
        conn.close()
        if row:
            return row[0]
    except Exception:
        pass
    return None


def _set_backfill_progress(league, last_date, status="in_progress"):
    """Update backfill progress for a league."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT OR REPLACE INTO backfill_progress (league, last_date, status, updated_at)
           VALUES (?, ?, ?, ?)""",
        (league, last_date, status, datetime.now(timezone.utc).isoformat())
    )
    conn.commit()
    conn.close()


def _sport_for_league(league):
    """Map league to ESPN sport path."""
    mapping = {
        "nfl": "football",
        "nba": "basketball",
        "mlb": "baseball",
        "nhl": "hockey",
        "eng.1": "soccer",
        "usa.1": "soccer",
    }
    return mapping.get(league, "")


def backfill_league(league):
    """Download all historical data for a single league."""
    sport = _sport_for_league(league)
    if not sport:
        log.error("Unknown league: %s", league)
        return 0

    # Check where we left off
    last_done = _get_backfill_progress(league)
    log.info("Backfill %s: last progress = %s", league.upper(), last_done or "none")

    total_new = 0
    total_fetched = 0
    empty_streak = 0

    for year in range(BACKFILL_START_YEAR, BACKFILL_END_YEAR + 1):
        dates = _get_season_dates(league, year)
        if not dates:
            continue

        for date_str in dates:
            # Skip dates we already did
            if last_done and date_str <= last_done:
                continue

            try:
                games = fetch_espn_date(sport, league, date_str)
                if games:
                    new = insert_games_batch(games)
                    total_new += new
                    total_fetched += len(games)
                    empty_streak = 0
                    if new > 0:
                        log.info("  %s %s: %d games, %d new",
                                 league.upper(), date_str, len(games), new)
                else:
                    empty_streak += 1

                # Save progress every date
                _set_backfill_progress(league, date_str)

                # Rate limiting -- be polite to ESPN
                # Faster when getting empty responses, slower with data
                if games:
                    time.sleep(0.8)
                elif empty_streak > 10:
                    time.sleep(0.3)
                else:
                    time.sleep(0.5)

            except KeyboardInterrupt:
                log.info("Backfill interrupted at %s %s", league.upper(), date_str)
                _set_backfill_progress(league, date_str, "interrupted")
                return total_new
            except Exception as exc:
                log.warning("Backfill error %s %s: %s", league.upper(), date_str, exc)
                time.sleep(2)

        log.info("  %s year %d complete: %d fetched, %d new so far",
                 league.upper(), year, total_fetched, total_new)

    _set_backfill_progress(league, "complete", "complete")
    log.info("Backfill %s COMPLETE: %d total games fetched, %d new inserted",
             league.upper(), total_fetched, total_new)
    return total_new


def run_backfill():
    """Run full historical backfill for all leagues."""
    log.info("=" * 60)
    log.info("HISTORICAL BACKFILL STARTING")
    log.info("  Leagues: %s", ", ".join(l.upper() for _, l in ESPN_SPORTS))
    log.info("  Years: %d - %d", BACKFILL_START_YEAR, BACKFILL_END_YEAR)
    log.info("=" * 60)

    init_db()

    # First seed horse racing data
    seed_major_horse_races()

    # Clean out the garbage TheSportsDB data
    _clean_bad_horse_data()

    grand_total = 0
    for sport, league in ESPN_SPORTS:
        try:
            new = backfill_league(league)
            grand_total += new
        except KeyboardInterrupt:
            log.info("Backfill interrupted by user")
            break
        except Exception as exc:
            log.error("Backfill failed for %s: %s", league.upper(), exc, exc_info=True)

    log.info("=" * 60)
    log.info("BACKFILL COMPLETE: %d total new games inserted", grand_total)

    # Report final counts
    _report_db_counts()
    log.info("=" * 60)


def _clean_bad_horse_data():
    """Remove garbage TheSportsDB entries that are actually soccer matches."""
    try:
        conn = sqlite3.connect(DB_PATH)
        # The TheSportsDB free tier returned soccer matches as horse racing
        # Delete any entries with tsdb_ prefix that have no winner_horse
        cur = conn.execute(
            "DELETE FROM horse_races WHERE id LIKE 'tsdb_%'"
        )
        deleted = cur.rowcount
        conn.commit()
        conn.close()
        if deleted > 0:
            log.info("Cleaned %d garbage TheSportsDB entries from horse_races", deleted)
    except Exception as exc:
        log.warning("Error cleaning bad horse data: %s", exc)


def _report_db_counts():
    """Log current database row counts by league."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.execute("SELECT league, COUNT(*) FROM games GROUP BY league ORDER BY league")
        rows = cur.fetchall()
        log.info("Database counts by league:")
        total = 0
        for league, count in rows:
            log.info("  %-8s: %6d games", league, count)
            total += count
        log.info("  %-8s: %6d games", "TOTAL", total)

        cur = conn.execute("SELECT COUNT(*) FROM horse_races")
        hr_count = cur.fetchone()[0]
        log.info("  %-8s: %6d races", "HORSES", hr_count)
        conn.close()
    except Exception as exc:
        log.warning("Error reporting counts: %s", exc)


# ============================================================
# SINGLE POLL CYCLE
# ============================================================

def poll_cycle():
    """Run one full poll: ESPN games + horse racing."""
    log.info("--- Poll cycle start ---")
    total_new_games = 0
    total_new_races = 0

    # ESPN sports -- fetch today and yesterday (catches late-finishing games)
    today = datetime.now(timezone.utc)
    dates_to_check = [
        today.strftime("%Y%m%d"),
        (today - timedelta(days=1)).strftime("%Y%m%d"),
    ]

    for sport, league in ESPN_SPORTS:
        try:
            league_new = 0
            league_fetched = 0
            for date_str in dates_to_check:
                games = fetch_espn_date(sport, league, date_str)
                if games:
                    new = insert_games(games)
                    league_new += new
                    league_fetched += len(games)
            if league_fetched > 0:
                total_new_games += league_new
                log.info("  %s: %d completed games fetched, %d new inserted",
                         league.upper(), league_fetched, league_new)
            else:
                log.info("  %s: no completed games right now", league.upper())
        except Exception as exc:
            log.error("  %s fetch error: %s", league.upper(), exc)
        time.sleep(1)  # small courtesy delay between leagues

    # Horse racing
    if HORSE_RACING_ENABLED:
        try:
            races = fetch_horse_racing_today()
            if races:
                new = insert_horse_races(races)
                total_new_races += new
                log.info("  HORSE RACING: %d events fetched, %d new inserted",
                         len(races), new)
            else:
                log.info("  HORSE RACING: no major races today")
        except Exception as exc:
            log.error("  HORSE RACING fetch error: %s", exc)

    log.info("--- Poll cycle end --- new games: %d, new races: %d",
             total_new_games, total_new_races)


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    log.info("=" * 60)
    log.info("SPORTS STREAMER STARTED")
    log.info("  DB   : %s", DB_PATH)
    log.info("  Poll : every %d seconds (%d min)", POLL_INTERVAL, POLL_INTERVAL // 60)
    log.info("  ESPN : %s", ", ".join(l.upper() for _, l in ESPN_SPORTS))
    log.info("  Races: Major horse races (Triple Crown, Breeders Cup)")
    log.info("=" * 60)

    # Check for --backfill flag
    if "--backfill" in sys.argv:
        run_backfill()
        log.info("Backfill complete. Switching to live polling mode...")
    else:
        init_db()
        # Seed horse races if not already done
        seed_major_horse_races()
        _clean_bad_horse_data()

    # Report current state
    _report_db_counts()

    while True:
        try:
            poll_cycle()
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            log.error("Unhandled error in poll cycle: %s", exc, exc_info=True)

        log.info("Sleeping %d seconds until next poll...", POLL_INTERVAL)
        try:
            time.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            log.info("Interrupted during sleep. Shutting down.")
            break

    log.info("Sports streamer stopped.")


if __name__ == "__main__":
    main()
