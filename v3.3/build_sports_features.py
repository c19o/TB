#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_sports_features.py
========================
Download historical sports data (NFL, NBA, MLB, NHL) from ESPN API + horse racing
from TheSportsDB. Compute gematria on team names, timestamps, scores.
Add as daily features for ML correlation with BTC.
"""

import sys, os, io, time, json, warnings, math
if os.name == 'nt':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import urllib.request
import urllib.error

DB_DIR = os.environ.get("SAVAGE22_V1_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
START = time.time()

def elapsed():
    return f"[{time.time()-START:.0f}s]"

# ============================================================
# GEMATRIA FUNCTIONS
# ============================================================
def ordinal_gematria(text):
    """A=1, B=2, ... Z=26. Sum of all letters."""
    return sum(ord(c.upper()) - 64 for c in text if c.isalpha())

def reverse_ordinal(text):
    """A=26, B=25, ... Z=1."""
    return sum(27 - (ord(c.upper()) - 64) for c in text if c.isalpha())

def reduction_gematria(text):
    """Digital root of ordinal."""
    val = ordinal_gematria(text)
    if val == 0: return 0
    return 1 + (val - 1) % 9

def english_gematria(text):
    """A=6, B=12, ... Z=156 (multiples of 6)."""
    return sum((ord(c.upper()) - 64) * 6 for c in text if c.isalpha())

def digital_root(n):
    if n == 0: return 0
    return 1 + (abs(int(n)) - 1) % 9

def compute_all_gematria(text):
    return {
        'ordinal': ordinal_gematria(text),
        'reverse': reverse_ordinal(text),
        'reduction': reduction_gematria(text),
        'english': english_gematria(text),
        'dr': digital_root(ordinal_gematria(text)),
    }

# ============================================================
# ESPN API — Free, no auth needed
# ============================================================
def fetch_espn_scores(sport, league, season, max_pages=50):
    """Fetch game scores from ESPN API for a given sport/league/season."""
    games = []
    # ESPN API endpoint for scoreboards by date range
    # We'll iterate by week/date ranges
    base_url = f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/scoreboard"

    if sport == 'football':
        # NFL: iterate by week (17-18 weeks per season)
        for week in range(1, 23):
            url = f"{base_url}?dates={season}&seasontype=2&week={week}"
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read())
                    for event in data.get('events', []):
                        game = parse_espn_event(event, sport, league)
                        if game:
                            games.append(game)
            except Exception as e:
                pass
            time.sleep(0.3)
    else:
        # NBA/MLB/NHL: iterate by date
        start_dates = {
            'basketball': (f'{season}-10-15', f'{int(season)+1}-06-30'),
            'baseball': (f'{season}-03-20', f'{season}-11-05'),
            'hockey': (f'{season}-10-01', f'{int(season)+1}-06-30'),
        }
        if sport not in start_dates:
            return games

        start_str, end_str = start_dates[sport]
        try:
            start = datetime.strptime(start_str, '%Y-%m-%d')
            end = datetime.strptime(end_str, '%Y-%m-%d')
        except Exception:
            return games

        current = start
        while current <= end:
            date_str = current.strftime('%Y%m%d')
            url = f"{base_url}?dates={date_str}"
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read())
                    for event in data.get('events', []):
                        game = parse_espn_event(event, sport, league)
                        if game:
                            games.append(game)
            except Exception:
                pass
            current += timedelta(days=7)  # sample weekly for speed
            time.sleep(0.2)

    return games

def parse_espn_event(event, sport, league):
    """Parse an ESPN event into a dict."""
    try:
        date_str = event.get('date', '')
        name = event.get('name', '')
        competitors = event.get('competitions', [{}])[0].get('competitors', [])
        if len(competitors) < 2:
            return None

        home = competitors[0]
        away = competitors[1]
        home_team = home.get('team', {}).get('displayName', '')
        away_team = away.get('team', {}).get('displayName', '')
        home_score = int(home.get('score', 0) or 0)
        away_score = int(away.get('score', 0) or 0)
        status = event.get('status', {}).get('type', {}).get('completed', False)

        if not status:
            return None

        return {
            'date': date_str[:10],
            'datetime': date_str,
            'sport': sport,
            'league': league.upper(),
            'home_team': home_team,
            'away_team': away_team,
            'home_score': home_score,
            'away_score': away_score,
            'total_score': home_score + away_score,
            'winner': home_team if home_score > away_score else away_team,
        }
    except Exception:
        return None

# ============================================================
# MAIN: FETCH SPORTS DATA
# ============================================================
print("=" * 70)
print("FETCHING SPORTS DATA + HORSE RACING FOR MATRIX CORRELATION")
print("=" * 70)

all_games = []

# ESPN sports to fetch
sports_config = [
    ('football', 'nfl', range(2017, 2026)),
    ('basketball', 'nba', range(2017, 2026)),
    ('baseball', 'mlb', range(2017, 2026)),
    ('hockey', 'nhl', range(2017, 2026)),
]

for sport, league, seasons in sports_config:
    print(f"\n{elapsed()} Fetching {league.upper()} data...")
    league_games = []
    for season in seasons:
        games = fetch_espn_scores(sport, league, str(season))
        league_games.extend(games)
        if games:
            print(f"  {season}: {len(games)} games")
    all_games.extend(league_games)
    print(f"  Total {league.upper()}: {len(league_games)} games")

print(f"\n{elapsed()} Total games fetched: {len(all_games)}")

# If ESPN fails or is slow, create synthetic sports features from known schedules
if len(all_games) < 100:
    print(f"\n{elapsed()} ESPN fetch was limited. Building features from known patterns instead...")
    # Generate sports season indicators as features
    # NFL: Sep-Feb, NBA: Oct-Jun, MLB: Apr-Oct, NHL: Oct-Jun
    # These are still useful as the matrix operates on energy cycles

# ============================================================
# COMPUTE GEMATRIA ON ALL TEAMS + SCORES
# ============================================================
print(f"\n{elapsed()} Computing gematria on teams and scores...")

if all_games:
    games_df = pd.DataFrame(all_games)
    games_df['date'] = pd.to_datetime(games_df['date'])

    # Gematria on team names (vectorized via list comprehension — no .apply())
    for side in ['home_team', 'away_team', 'winner']:
        games_df[f'{side}_ordinal'] = [ordinal_gematria(x) for x in games_df[side].values]
        games_df[f'{side}_dr'] = [digital_root(ordinal_gematria(x)) for x in games_df[side].values]
        games_df[f'{side}_reverse'] = [reverse_ordinal(x) for x in games_df[side].values]

    # Score gematria (vectorized)
    games_df['total_score_dr'] = np.where(
        games_df['total_score'] == 0, 0,
        1 + (np.abs(games_df['total_score'].astype(int)) - 1) % 9)
    games_df['score_diff'] = (games_df['home_score'] - games_df['away_score']).abs()
    games_df['score_diff_dr'] = np.where(
        games_df['score_diff'] == 0, 0,
        1 + (games_df['score_diff'].astype(int) - 1) % 9)

    # Score contains special numbers (vectorized string check)
    total_str = games_df['total_score'].astype(str)
    for num in ['33', '22', '11', '13', '27', '37', '93', '39']:
        games_df[f'score_contains_{num}'] = total_str.str.contains(num, regex=False).astype(int)

    # Winner gematria patterns
    games_df['winner_caution'] = games_df['winner_dr'].isin([9, 6]).astype(int)
    games_df['winner_pump'] = games_df['winner_dr'].isin([3, 7]).astype(int)

    print(f"  Games with gematria: {len(games_df)}")
    print(f"  Date range: {games_df['date'].min()} to {games_df['date'].max()}")
    print(f"  Leagues: {games_df['league'].value_counts().to_dict()}")
else:
    games_df = pd.DataFrame()

# ============================================================
# AGGREGATE TO DAILY FEATURES
# ============================================================
print(f"\n{elapsed()} Aggregating to daily features...")

# Create date range matching our BTC data
date_range = pd.date_range('2017-01-01', '2026-03-17', freq='D')
daily = pd.DataFrame({'date': date_range})

if len(games_df) > 0:
    # Per-league daily game counts
    for league in ['NFL', 'NBA', 'MLB', 'NHL']:
        league_games = games_df[games_df['league'] == league]
        if len(league_games) > 0:
            daily_counts = league_games.groupby('date').size().rename(f'{league.lower()}_games')
            daily = daily.merge(daily_counts.reset_index(), on='date', how='left')
            daily[f'{league.lower()}_games'] = daily[f'{league.lower()}_games'].fillna(0).astype(int)

            # Average winner gematria DR per day
            daily_winner_dr = league_games.groupby('date')['winner_dr'].mean().rename(f'{league.lower()}_winner_dr')
            daily = daily.merge(daily_winner_dr.reset_index(), on='date', how='left')

            # Total score DR per day
            daily_score_dr = league_games.groupby('date')['total_score_dr'].mean().rename(f'{league.lower()}_score_dr')
            daily = daily.merge(daily_score_dr.reset_index(), on='date', how='left')

            # Caution/pump signals from winner names
            daily_caution = league_games.groupby('date')['winner_caution'].sum().rename(f'{league.lower()}_caution_winners')
            daily = daily.merge(daily_caution.reset_index(), on='date', how='left')

            daily_pump = league_games.groupby('date')['winner_pump'].sum().rename(f'{league.lower()}_pump_winners')
            daily = daily.merge(daily_pump.reset_index(), on='date', how='left')

    # Total sports activity
    game_cols = [c for c in daily.columns if c.endswith('_games')]
    daily['total_games_today'] = daily[game_cols].sum(axis=1) if game_cols else 0

    # Any caution winner today
    caution_cols = [c for c in daily.columns if c.endswith('_caution_winners')]
    daily['any_caution_winner'] = (daily[caution_cols].sum(axis=1) > 0).astype(int) if caution_cols else 0

    # Any pump winner today
    pump_cols = [c for c in daily.columns if c.endswith('_pump_winners')]
    daily['any_pump_winner'] = (daily[pump_cols].sum(axis=1) > 0).astype(int) if pump_cols else 0

# Sports season flags (always available even without API data)
doy = daily['date'].dt.dayofyear
month = daily['date'].dt.month
dow = daily['date'].dt.dayofweek

daily['nfl_season'] = ((month >= 9) | (month <= 2)).astype(int)
daily['nba_season'] = ((month >= 10) | (month <= 6)).astype(int)
daily['mlb_season'] = ((month >= 4) & (month <= 10)).astype(int)
daily['nhl_season'] = ((month >= 10) | (month <= 6)).astype(int)

# Super Bowl week (usually early February)
daily['super_bowl_week'] = ((month == 2) & (daily['date'].dt.day <= 14)).astype(int)
# World Series week (usually late October)
daily['world_series_week'] = ((month == 10) & (daily['date'].dt.day >= 20)).astype(int)
# NBA Finals (usually June)
daily['nba_finals_month'] = (month == 6).astype(int)
# March Madness
daily['march_madness'] = (month == 3).astype(int)

# Sunday = NFL day
daily['is_nfl_sunday'] = ((dow == 6) & (daily['nfl_season'] == 1)).astype(int)
# Monday Night Football
daily['is_mnf'] = ((dow == 0) & (daily['nfl_season'] == 1)).astype(int)
# Thursday Night Football
daily['is_tnf'] = ((dow == 3) & (daily['nfl_season'] == 1)).astype(int)

# Major horse racing events (Triple Crown dates - approximate)
# Kentucky Derby: 1st Saturday in May
daily['kentucky_derby_week'] = ((month == 5) & (daily['date'].dt.day <= 7) & (dow == 5)).astype(int)
# Preakness: 3rd Saturday in May
daily['preakness_week'] = ((month == 5) & (daily['date'].dt.day >= 14) & (daily['date'].dt.day <= 21) & (dow == 5)).astype(int)
# Belmont Stakes: ~early June
daily['belmont_week'] = ((month == 6) & (daily['date'].dt.day <= 14) & (dow == 5)).astype(int)
daily['triple_crown_season'] = ((month >= 5) & (month <= 6)).astype(int)

# Gematria of major sports events
events_gematria = {
    'Super Bowl': compute_all_gematria('Super Bowl'),
    'World Series': compute_all_gematria('World Series'),
    'NBA Finals': compute_all_gematria('NBA Finals'),
    'Kentucky Derby': compute_all_gematria('Kentucky Derby'),
    'March Madness': compute_all_gematria('March Madness'),
    'Stanley Cup': compute_all_gematria('Stanley Cup'),
}
print(f"\n  Event Gematria:")
for event, gem in events_gematria.items():
    print(f"    {event}: ordinal={gem['ordinal']}, dr={gem['dr']}, reverse={gem['reverse']}")

# ============================================================
# FAMOUS TEAM GEMATRIA (pre-computed for correlation)
# ============================================================
print(f"\n{elapsed()} Pre-computing team gematria...")

nfl_teams = [
    'Arizona Cardinals', 'Atlanta Falcons', 'Baltimore Ravens', 'Buffalo Bills',
    'Carolina Panthers', 'Chicago Bears', 'Cincinnati Bengals', 'Cleveland Browns',
    'Dallas Cowboys', 'Denver Broncos', 'Detroit Lions', 'Green Bay Packers',
    'Houston Texans', 'Indianapolis Colts', 'Jacksonville Jaguars', 'Kansas City Chiefs',
    'Las Vegas Raiders', 'Los Angeles Chargers', 'Los Angeles Rams', 'Miami Dolphins',
    'Minnesota Vikings', 'New England Patriots', 'New Orleans Saints', 'New York Giants',
    'New York Jets', 'Philadelphia Eagles', 'Pittsburgh Steelers', 'San Francisco 49ers',
    'Seattle Seahawks', 'Tampa Bay Buccaneers', 'Tennessee Titans', 'Washington Commanders',
]

team_gematria = {}
for team in nfl_teams:
    gem = compute_all_gematria(team)
    team_gematria[team] = gem

# Find teams with caution numbers (DR 9, 6) and pump numbers (DR 3, 7)
caution_teams = [t for t, g in team_gematria.items() if g['dr'] in [9, 6]]
pump_teams = [t for t, g in team_gematria.items() if g['dr'] in [3, 7]]
master_teams = [t for t, g in team_gematria.items() if g['ordinal'] in [22, 33, 44, 55, 66, 77, 88, 99, 111, 222]]

print(f"  Caution DR teams (9,6): {caution_teams[:5]}...")
print(f"  Pump DR teams (3,7): {pump_teams[:5]}...")
print(f"  Master number teams: {master_teams[:3]}...")

# Famous horse names gematria
horses = ['Secretariat', 'Seabiscuit', 'American Pharoah', 'Justify', 'Affirmed',
          'Seattle Slew', 'Citation', 'War Admiral', 'Gallant Fox', 'Omaha',
          'Count Fleet', 'Assault', 'Whirlaway', 'Sir Barton', 'Nyquist',
          'California Chrome', 'I ll Have Another', 'Mine That Bird', 'Big Brown',
          'Smarty Jones', 'Funny Cide', 'Monarchos', 'Fusaichi Pegasus']

print(f"\n  Horse Name Gematria (sample):")
for horse in horses[:8]:
    gem = compute_all_gematria(horse)
    print(f"    {horse}: ordinal={gem['ordinal']}, dr={gem['dr']}")

# ============================================================
# ESOTERIC CROSS-REFERENCE: Astrology + Numerology on game dates
# ============================================================
print(f"\n{elapsed()} Cross-referencing with astrology systems...")

# Load all astrology data
try:
    conn_astro = sqlite3.connect(f'{DB_DIR}/astrology_full.db')
    astro_df = pd.read_sql_query("SELECT * FROM daily_astrology ORDER BY date", conn_astro)
    conn_astro.close()
    astro_df['date'] = pd.to_datetime(astro_df['date'])
    astro_df = astro_df.drop_duplicates(subset='date', keep='last').set_index('date')
    print(f"  Astrology loaded: {len(astro_df)} days")
except Exception as e:
    print(f"  WARNING: astrology load failed: {e}")
    astro_df = pd.DataFrame()
    print(f"  Astrology not available")

try:
    conn_ephem = sqlite3.connect(f'{DB_DIR}/ephemeris_cache.db')
    ephem_df = pd.read_sql_query("SELECT * FROM ephemeris ORDER BY date", conn_ephem)
    conn_ephem.close()
    ephem_df['date'] = pd.to_datetime(ephem_df['date'])
    ephem_df = ephem_df.drop_duplicates(subset='date', keep='last').set_index('date')
    print(f"  Ephemeris loaded: {len(ephem_df)} days")
except Exception as e:
    print(f"  WARNING: ephemeris load failed: {e}")
    ephem_df = pd.DataFrame()
    print(f"  Ephemeris not available")

daily['date_dt'] = pd.to_datetime(daily['date'])

# Map astrology to game dates
if len(astro_df) > 0:
    astro_df.index = astro_df.index.normalize()
    daily_norm = daily['date_dt'].dt.normalize()

    # Vedic nakshatra on game day
    if 'nakshatra' in astro_df.columns:
        s = pd.to_numeric(astro_df['nakshatra'], errors='coerce')
        mapped = s.reindex(daily_norm.values).ffill()
        daily['game_day_nakshatra'] = mapped.values

    # Vedic tithi
    if 'tithi' in astro_df.columns:
        s = pd.to_numeric(astro_df['tithi'], errors='coerce')
        mapped = s.reindex(daily_norm.values).ffill()
        daily['game_day_tithi'] = mapped.values

    # Chinese BaZi day stem
    if 'day_stem' in astro_df.columns:
        s = pd.to_numeric(astro_df['day_stem'], errors='coerce')
        mapped = s.reindex(daily_norm.values).ffill()
        daily['game_day_bazi_stem'] = mapped.values

    # Mayan Tzolkin tone
    if 'tzolkin_tone' in astro_df.columns:
        s = pd.to_numeric(astro_df['tzolkin_tone'], errors='coerce')
        mapped = s.reindex(daily_norm.values).ffill()
        daily['game_day_mayan_tone'] = mapped.values

    # Arabic lots
    for lot in ['lot_commerce', 'lot_catastrophe']:
        if lot in astro_df.columns:
            s = pd.to_numeric(astro_df[lot], errors='coerce')
            mapped = s.reindex(daily_norm.values).ffill()
            daily[f'game_day_{lot}'] = mapped.values

if len(ephem_df) > 0:
    ephem_df.index = ephem_df.index.normalize()

    # Western moon mansion
    if 'moon_mansion' in ephem_df.columns:
        s = pd.to_numeric(ephem_df['moon_mansion'], errors='coerce')
        mapped = s.reindex(daily_norm.values).ffill()
        daily['game_day_moon_mansion'] = mapped.values

    # Mercury retrograde
    if 'mercury_retrograde' in ephem_df.columns:
        s = pd.to_numeric(ephem_df['mercury_retrograde'], errors='coerce')
        mapped = s.reindex(daily_norm.values).ffill()
        daily['game_day_mercury_retro'] = mapped.values

    # Planetary strength index
    if 'planetary_strength' in ephem_df.columns:
        s = pd.to_numeric(ephem_df['planetary_strength'], errors='coerce')
        mapped = s.reindex(daily_norm.values).ffill()
        daily['game_day_psi'] = mapped.values

    # Moon phase
    if 'moon_phase' in ephem_df.columns:
        s = pd.to_numeric(ephem_df['moon_phase'], errors='coerce')
        mapped = s.reindex(daily_norm.values).ffill()
        daily['game_day_moon_phase'] = mapped.values

# Numerology on game dates
print(f"  {elapsed()} Date numerology...")
# Vectorized date digital root (no .apply)
_date_digits_sum = np.array([sum(int(x) for x in d.strftime('%Y%m%d')) for d in daily['date_dt']])
daily['game_date_dr'] = np.where(_date_digits_sum == 0, 0, 1 + (_date_digits_sum - 1) % 9)
daily['game_day_13'] = (daily['date_dt'].dt.day == 13).astype(int)
daily['game_day_22'] = (daily['date_dt'].dt.day == 22).astype(int)
daily['game_day_27'] = (daily['date_dt'].dt.day == 27).astype(int)
daily['game_doy'] = daily['date_dt'].dt.dayofyear

# Special DOY numbers
for doy_num in [33, 66, 93, 113, 223, 322]:
    daily[f'game_doy_{doy_num}'] = (daily['game_doy'] == doy_num).astype(int)

# Interactions: games on caution astrology days
if 'game_day_nakshatra' in daily.columns and 'total_games_today' in daily.columns:
    daily['games_on_caution_nakshatra'] = daily['total_games_today'] * (daily['game_day_nakshatra'].isin([0, 6, 9, 17]).astype(int))
if 'game_day_mercury_retro' in daily.columns and 'total_games_today' in daily.columns:
    daily['games_during_mercury_retro'] = daily['total_games_today'] * daily['game_day_mercury_retro']
if 'game_date_dr' in daily.columns and 'total_games_today' in daily.columns:
    daily['games_on_dr9_day'] = daily['total_games_today'] * (daily['game_date_dr'] == 9).astype(int)

# Score + Astrology interactions (if we have game data)
if len(games_df) > 0 and 'game_day_nakshatra' in daily.columns:
    # Average score on specific nakshatras
    games_with_astro = games_df.merge(
        daily[['date_dt', 'game_day_nakshatra', 'game_day_moon_phase']].rename(columns={'date_dt': 'date'}),
        on='date', how='left'
    )
    # Winner DR by nakshatra (does astrology predict which team name wins?)
    nak_winner = games_with_astro.groupby('game_day_nakshatra')['winner_dr'].mean()
    print(f"\n  Winner DR by Nakshatra (sample):")
    for nak, dr in list(nak_winner.items())[:5]:
        print(f"    Nakshatra {int(nak)}: avg winner DR = {dr:.2f}")

daily = daily.drop(columns=['date_dt', 'game_doy'], errors='ignore')
print(f"\n  Total daily features after esoteric cross-ref: {len([c for c in daily.columns if c != 'date'])}")

# ============================================================
# SAVE TO DATABASE
# ============================================================
print(f"\n{elapsed()} Saving to sports_data.db...")

# Fill only count-type columns with 0 (no game = 0 games, not NaN)
# Leave DR values, caution/pump signals, and esoteric features as NaN (missing data)
count_cols = [c for c in daily.columns if c.endswith('_games') or c == 'total_games_today']
for col in count_cols:
    daily[col] = daily[col].fillna(0)

# Ensure all columns are numeric except date (coerce non-numeric strings, preserve NaN)
for col in daily.columns:
    if col != 'date':
        daily[col] = pd.to_numeric(daily[col], errors='coerce')

conn = sqlite3.connect(f'{DB_DIR}/sports_data.db')

# Save daily features
daily_save = daily.copy()
daily_save['date'] = daily_save['date'].dt.strftime('%Y-%m-%d')
daily_save.to_sql('daily_sports', conn, if_exists='replace', index=False)
conn.execute("CREATE INDEX IF NOT EXISTS idx_sports_date ON daily_sports(date)")

# Save raw games
if len(games_df) > 0:
    games_save = games_df.copy()
    games_save['date'] = games_save['date'].dt.strftime('%Y-%m-%d')
    games_save.to_sql('games', conn, if_exists='replace', index=False)

# Save team gematria lookup
team_rows = []
for team, gem in team_gematria.items():
    team_rows.append({'team': team, **gem})
pd.DataFrame(team_rows).to_sql('team_gematria', conn, if_exists='replace', index=False)

conn.close()

feature_cols = [c for c in daily.columns if c != 'date']
print(f"\n{'='*70}")
print(f"SPORTS DATA COMPLETE")
print(f"{'='*70}")
print(f"  Games fetched: {len(all_games)}")
print(f"  Daily features: {len(feature_cols)}")
print(f"  Feature list: {feature_cols}")
print(f"  Date range: {daily['date'].min()} to {daily['date'].max()}")
print(f"  Output: {DB_DIR}/sports_data.db")
print(f"  Time: {elapsed()}")
