#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
universal_astro.py — Universal Astrology Engine
=================================================
Single source of truth for ALL astrology calculations.
Takes any timestamp, returns full snapshot across all systems.

Consolidates from:
  - astrology_engine.py (Western: planets, retrogrades, aspects, eclipses, VOC)
  - numerology_engine.py (moon phase, planetary hours, zodiac)
  - astrology_full.db (Vedic: nakshatras, tithi, yoga, guna)

Usage:
  from universal_astro import astro_snapshot, astro_flat
  result = astro_snapshot(datetime.now())
"""

import os
import sys
import math
import sqlite3
from datetime import datetime, timezone, timedelta

# Ensure parent directory (root) is on sys.path so astrology_engine.py is importable
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

DB_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Try importing from existing engines (they have the heavy lifting)
# ============================================================
_HAS_ASTRO_ENGINE = False
_HAS_SWISSEPH = False

try:
    from astrology_engine import (
        is_mercury_retrograde, is_planet_retrograde,
        count_hard_aspects, count_soft_aspects,
        is_voc_moon_classical as is_void_of_course_moon, is_eclipse_window,
        get_planetary_hour as _astro_planetary_hour,
        get_astrology_signals,
    )
    _HAS_ASTRO_ENGINE = True
except ImportError:
    pass

try:
    import swisseph as swe
    _HAS_SWISSEPH = True
except ImportError:
    pass


# ============================================================
# MOON PHASE (standalone — no external deps)
# ============================================================
SYNODIC_MONTH = 29.53059
KNOWN_NEW_MOON = datetime(2000, 1, 6, 18, 14)

def get_moon_phase(dt):
    """Calculate moon phase from any datetime. No external deps."""
    if dt.tzinfo:
        dt = dt.replace(tzinfo=None)
    diff = (dt - KNOWN_NEW_MOON).total_seconds() / 86400
    phase_day = diff % SYNODIC_MONTH

    if phase_day < 1.85: phase = "new_moon"
    elif phase_day < 7.38: phase = "waxing_crescent"
    elif phase_day < 9.23: phase = "first_quarter"
    elif phase_day < 13.84: phase = "waxing_gibbous"
    elif phase_day < 16.61: phase = "full_moon"
    elif phase_day < 22.15: phase = "waning_gibbous"
    elif phase_day < 23.99: phase = "last_quarter"
    else: phase = "waning_crescent"

    lunar_sin = math.sin(2 * math.pi * phase_day / SYNODIC_MONTH)
    lunar_cos = math.cos(2 * math.pi * phase_day / SYNODIC_MONTH)

    return {
        'phase': phase,
        'phase_day': round(phase_day, 2),
        'lunar_sin': round(lunar_sin, 4),
        'lunar_cos': round(lunar_cos, 4),
        'is_new_moon': phase == "new_moon",
        'is_full_moon': phase == "full_moon",
    }


# ============================================================
# PLANETARY HOURS (standalone)
# ============================================================
PLANET_ORDER = ["Saturn", "Jupiter", "Mars", "Sun", "Venus", "Mercury", "Moon"]
DAY_START_PLANET = {
    "Saturday": 0, "Thursday": 1, "Tuesday": 2, "Sunday": 3,
    "Friday": 4, "Wednesday": 5, "Monday": 6,
}
PLANETARY_RULERS = {
    "Monday": "Moon", "Tuesday": "Mars", "Wednesday": "Mercury",
    "Thursday": "Jupiter", "Friday": "Venus", "Saturday": "Saturn", "Sunday": "Sun",
}

def get_planetary_hour(dt):
    """Classical planetary hour calculation."""
    day_name = dt.strftime("%A")
    hour_since_sunrise = (dt.hour - 6) % 24
    start_idx = DAY_START_PLANET.get(day_name, 0)
    planet_idx = (start_idx + hour_since_sunrise) % 7
    planet = PLANET_ORDER[planet_idx]
    day_ruler = PLANETARY_RULERS[day_name]

    return {
        'planet': planet,
        'day_ruler': day_ruler,
        'double_power': planet == day_ruler,
        'is_jupiter': planet == "Jupiter",
        'is_saturn': planet == "Saturn",
        'is_mars': planet == "Mars",
    }


# ============================================================
# ZODIAC (standalone)
# ============================================================
ZODIAC_SIGNS = [
    ("Capricorn", 1, 19), ("Aquarius", 1, 20), ("Aquarius", 2, 18),
    ("Pisces", 2, 19), ("Pisces", 3, 20), ("Aries", 3, 21), ("Aries", 4, 19),
    ("Taurus", 4, 20), ("Taurus", 5, 20), ("Gemini", 5, 21), ("Gemini", 6, 20),
    ("Cancer", 6, 21), ("Cancer", 7, 22), ("Leo", 7, 23), ("Leo", 8, 22),
    ("Virgo", 8, 23), ("Virgo", 9, 22), ("Libra", 9, 23), ("Libra", 10, 22),
    ("Scorpio", 10, 23), ("Scorpio", 11, 21), ("Sagittarius", 11, 22),
    ("Sagittarius", 12, 21), ("Capricorn", 12, 22),
]

def get_zodiac(dt):
    """Get zodiac sign for a date."""
    m, d = dt.month, dt.day
    signs = {
        (1, 20): "Aquarius", (2, 19): "Pisces", (3, 21): "Aries", (4, 20): "Taurus",
        (5, 21): "Gemini", (6, 21): "Cancer", (7, 23): "Leo", (8, 23): "Virgo",
        (9, 23): "Libra", (10, 23): "Scorpio", (11, 22): "Sagittarius", (12, 22): "Capricorn",
    }
    for (start_m, start_d), sign in sorted(signs.items()):
        if m < start_m or (m == start_m and d < start_d):
            break
        result = sign
    else:
        result = "Capricorn"
    return result


# ============================================================
# VEDIC ASTROLOGY (from astrology_full.db)
# ============================================================

def get_vedic(dt):
    """Query Vedic astrology data from astrology_full.db for nearest date."""
    result = {
        'nakshatra': 0, 'nakshatra_nature': 0, 'nakshatra_guna': 0,
        'key_nakshatra': 0, 'tithi': 0, 'yoga': 0, 'karana': 0,
    }

    db_path = os.path.join(DB_DIR, 'astrology_full.db')
    if not os.path.exists(db_path):
        return result

    try:
        conn = sqlite3.connect(db_path, timeout=5)
        conn.row_factory = sqlite3.Row
        date_str = dt.strftime('%Y-%m-%d')

        row = conn.execute(
            "SELECT * FROM daily_astrology WHERE date <= ? ORDER BY date DESC LIMIT 1",
            (date_str,)
        ).fetchone()

        if row:
            cols = row.keys()
            for col in cols:
                val = row[col]
                if col == 'nakshatra' and val is not None:
                    result['nakshatra'] = val
                elif col == 'nakshatra_nature' and val is not None:
                    # Encode: deva=1, human=0, rakshasa=-1
                    nature_map = {'deva': 1, 'human': 0, 'rakshasa': -1}
                    result['nakshatra_nature'] = nature_map.get(str(val).lower(), 0)
                elif col == 'nakshatra_guna' and val is not None:
                    guna_map = {'sattva': 1, 'rajas': 0, 'tamas': -1}
                    result['nakshatra_guna'] = guna_map.get(str(val).lower(), 0)
                elif col == 'tithi' and val is not None:
                    result['tithi'] = val
                elif col == 'yoga' and val is not None:
                    result['yoga'] = val
                elif col == 'karana' and val is not None:
                    result['karana'] = val

            # Key nakshatras (backtested as significant)
            key_nakshatras = {19, 21, 23, 12}  # Purva Ashadha, Shravana, Shatabhisha, Hasta
            try:
                nk = int(result['nakshatra'])
                result['key_nakshatra'] = 1 if nk in key_nakshatras else 0
            except (ValueError, TypeError):
                pass

        conn.close()
    except Exception:
        pass

    return result


# ============================================================
# CHINESE BAZI (simplified)
# ============================================================

HEAVENLY_STEMS = ["Jia", "Yi", "Bing", "Ding", "Wu", "Ji", "Geng", "Xin", "Ren", "Gui"]
EARTHLY_BRANCHES = ["Zi", "Chou", "Yin", "Mao", "Chen", "Si", "Wu_b", "Wei", "Shen", "You", "Xu", "Hai"]
ELEMENTS = ["Wood", "Wood", "Fire", "Fire", "Earth", "Earth", "Metal", "Metal", "Water", "Water"]

def get_bazi(dt):
    """Simplified BaZi day pillar calculation."""
    # Day stem cycles every 60 days from a known reference
    ref = datetime(2000, 1, 7)  # Known: Jia-Zi day
    if dt.tzinfo:
        dt = dt.replace(tzinfo=None)
    days = (dt - ref).days
    stem_idx = days % 10
    branch_idx = days % 12
    element_idx = stem_idx  # Stems map directly to elements

    return {
        'stem': HEAVENLY_STEMS[stem_idx],
        'stem_idx': stem_idx,
        'branch': EARTHLY_BRANCHES[branch_idx],
        'branch_idx': branch_idx,
        'element': ELEMENTS[element_idx],
        'element_idx': ELEMENTS.index(ELEMENTS[element_idx]) // 2,  # 0-4
    }


# ============================================================
# MAYAN TZOLKIN
# ============================================================

TZOLKIN_SIGNS = [
    "Imix", "Ik", "Akbal", "Kan", "Chicchan", "Cimi", "Manik", "Lamat",
    "Muluc", "Oc", "Chuen", "Eb", "Ben", "Ix", "Men", "Cib",
    "Caban", "Etznab", "Cauac", "Ahau"
]

def get_tzolkin(dt):
    """Calculate Mayan Tzolkin date."""
    # Reference: Dec 21, 2012 = 4 Ahau (tone=4, sign=20)
    ref = datetime(2012, 12, 21)
    if dt.tzinfo:
        dt = dt.replace(tzinfo=None)
    days = (dt - ref).days

    tone = ((days + 3) % 13) + 1  # 1-13 (ref was tone 4)
    sign_idx = (days + 19) % 20   # 0-19 (ref was Ahau=19)
    kin = ((days + 159) % 260) + 1  # 1-260 (ref: Dec 21, 2012 = kin 160)

    return {
        'tone': tone,
        'sign': TZOLKIN_SIGNS[sign_idx],
        'sign_idx': sign_idx,
        'kin': kin,
    }


# ============================================================
# WESTERN ASTROLOGY (from astrology_engine.py if available)
# ============================================================

def get_western(dt):
    """Get Western astrology data. Uses astrology_engine.py if available."""
    result = {
        'mercury_retrograde': 0,
        'venus_retrograde': 0,
        'mars_retrograde': 0,
        'hard_aspects': 0,
        'soft_aspects': 0,
        'eclipse_window': 0,
        'voc_moon': 0,
        'planetary_strength': 0,
    }

    if _HAS_ASTRO_ENGINE:
        try:
            mr = is_mercury_retrograde(dt)
            result['mercury_retrograde'] = 1 if mr[0] else 0
        except Exception:
            pass
        try:
            result['venus_retrograde'] = 1 if is_planet_retrograde("Venus", dt) else 0
        except Exception:
            pass
        try:
            result['mars_retrograde'] = 1 if is_planet_retrograde("Mars", dt) else 0
        except Exception:
            pass
        try:
            ha = count_hard_aspects(dt)
            result['hard_aspects'] = ha[0] if isinstance(ha, tuple) else ha
        except Exception:
            pass
        try:
            sa = count_soft_aspects(dt)
            result['soft_aspects'] = sa[0] if isinstance(sa, tuple) else sa
        except Exception:
            pass
        try:
            ew = is_eclipse_window(dt)
            result['eclipse_window'] = 1 if ew[0] else 0
        except Exception:
            pass
        try:
            voc = is_void_of_course_moon(dt)
            result['voc_moon'] = 1 if voc[0] else 0
        except Exception:
            pass
        try:
            # Compute planetary strength as soft - hard ratio
            result['planetary_strength'] = result['soft_aspects'] - result['hard_aspects']
        except Exception:
            pass
    else:
        # Fallback: query astrology_full.db
        db_path = os.path.join(DB_DIR, 'astrology_full.db')
        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path, timeout=5)
                date_str = dt.strftime('%Y-%m-%d')
                row = conn.execute(
                    "SELECT * FROM daily_astrology WHERE date <= ? ORDER BY date DESC LIMIT 1",
                    (date_str,)
                ).fetchone()
                if row:
                    cols = [d[0] for d in conn.execute("PRAGMA table_info(daily_astrology)").fetchall()]
                    row_dict = dict(zip([c[1] for c in conn.execute("PRAGMA table_info(daily_astrology)").fetchall()], row))
                    for key in result:
                        if key in row_dict and row_dict[key] is not None:
                            try:
                                result[key] = int(float(row_dict[key]))
                            except (ValueError, TypeError):
                                pass
                conn.close()
            except Exception:
                pass

    return result


# ============================================================
# MAIN: ASTRO SNAPSHOT
# ============================================================

def astro_snapshot(dt):
    """
    Returns FULL astrology snapshot for any timestamp.
    Combines Western, Vedic, Chinese BaZi, Mayan Tzolkin, and Lunar data.
    """
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)

    moon = get_moon_phase(dt)
    planetary_hour = get_planetary_hour(dt)
    zodiac = get_zodiac(dt)
    vedic = get_vedic(dt)
    bazi = get_bazi(dt)
    tzolkin = get_tzolkin(dt)
    western = get_western(dt)

    return {
        # Lunar
        'moon_phase': moon['phase'],
        'moon_phase_day': moon['phase_day'],
        'lunar_phase_sin': moon['lunar_sin'],
        'lunar_phase_cos': moon['lunar_cos'],
        'is_new_moon': moon['is_new_moon'],
        'is_full_moon': moon['is_full_moon'],
        # Planetary hours
        'planetary_hour': planetary_hour['planet'],
        'day_ruler': planetary_hour['day_ruler'],
        'double_power': planetary_hour['double_power'],
        'is_jupiter_hour': planetary_hour['is_jupiter'],
        'is_saturn_hour': planetary_hour['is_saturn'],
        'is_mars_hour': planetary_hour['is_mars'],
        # Zodiac
        'zodiac_sign': zodiac,
        # Western
        'mercury_retrograde': western['mercury_retrograde'],
        'venus_retrograde': western['venus_retrograde'],
        'mars_retrograde': western['mars_retrograde'],
        'hard_aspects': western['hard_aspects'],
        'soft_aspects': western['soft_aspects'],
        'eclipse_window': western['eclipse_window'],
        'voc_moon': western['voc_moon'],
        'planetary_strength': western['planetary_strength'],
        # Vedic
        'nakshatra': vedic['nakshatra'],
        'nakshatra_nature': vedic['nakshatra_nature'],
        'nakshatra_guna': vedic['nakshatra_guna'],
        'key_nakshatra': vedic['key_nakshatra'],
        'tithi': vedic['tithi'],
        'yoga': vedic['yoga'],
        # Chinese BaZi
        'bazi_stem': bazi['stem_idx'],
        'bazi_branch': bazi['branch_idx'],
        'bazi_element': bazi['element_idx'],
        # Mayan Tzolkin
        'tzolkin_tone': tzolkin['tone'],
        'tzolkin_sign': tzolkin['sign_idx'],
        'tzolkin_kin': tzolkin['kin'],
    }


def astro_flat(dt, prefix=''):
    """Returns flat dict suitable for ML features."""
    snap = astro_snapshot(dt)
    p = f"{prefix}_" if prefix else ""

    # Convert non-numeric values
    result = {}
    for k, v in snap.items():
        if k in ('moon_phase', 'planetary_hour', 'day_ruler', 'zodiac_sign'):
            continue  # Skip string values for ML
        if isinstance(v, bool):
            result[f'{p}{k}'] = int(v)
        elif isinstance(v, (int, float)):
            result[f'{p}{k}'] = v
        else:
            try:
                result[f'{p}{k}'] = float(v)
            except (ValueError, TypeError):
                result[f'{p}{k}'] = 0

    return result


if __name__ == "__main__":
    now = datetime(2026, 3, 18, 15, 30)
    snap = astro_snapshot(now)
    print(f"=== Astro Snapshot for {now} ===")
    for k, v in snap.items():
        print(f"  {k}: {v}")
    print(f"\n  Total fields: {len(snap)}")
