"""
Astrology Engine — Real planetary calculations for trading signals
==================================================================
Uses PyEphem for precise astronomical computations:
  - Mercury retrograde periods
  - Eclipse windows (solar + lunar)
  - Planetary aspects (conjunction, opposition, square, trine, sextile)
  - Zodiac sign transitions (sun ingress)
  - Void of course moon
  - Planetary retrograde periods (all planets)
  - Bitcoin natal chart transits
  - Great conjunctions (Saturn-Jupiter)

All functions accept a datetime and return signal data.
"""
import ephem
import logging
import math
import numpy as np
from datetime import datetime, timedelta
from functools import lru_cache

# ===========================================================================
# CONSTANTS
# ===========================================================================

# Bitcoin's birth: Jan 3, 2009, 18:15:05 UTC (genesis block timestamp)
BTC_BIRTH = datetime(2009, 1, 3, 18, 15, 5)

# Zodiac signs
ZODIAC_SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]

# Zodiac sign boundaries (ecliptic longitude in degrees)
ZODIAC_BOUNDS = [(i * 30, (i + 1) * 30) for i in range(12)]

# Aspect definitions: name, angle, orb (tolerance in degrees)
ASPECTS = {
    "conjunction": (0, 8),
    "opposition": (180, 8),
    "square": (90, 6),
    "trine": (120, 6),
    "sextile": (60, 4),
}

# Planet objects
PLANETS = {
    "Mercury": ephem.Mercury,
    "Venus": ephem.Venus,
    "Mars": ephem.Mars,
    "Jupiter": ephem.Jupiter,
    "Saturn": ephem.Saturn,
    "Uranus": ephem.Uranus,
    "Neptune": ephem.Neptune,
    "Pluto": ephem.Pluto,
}

# Trading energy by planet
PLANET_ENERGY = {
    "Mercury": "communication/speed",
    "Venus": "value/money",
    "Mars": "aggression/volatility",
    "Jupiter": "expansion/growth",
    "Saturn": "restriction/contraction",
    "Uranus": "disruption/innovation",
    "Neptune": "illusion/deception",
    "Pluto": "transformation/power",
}


# ===========================================================================
# CORE PLANETARY FUNCTIONS
# ===========================================================================

def _to_ephem_date(dt):
    """Convert datetime to ephem.Date."""
    return ephem.Date(dt)


def get_planet_position(planet_name, dt):
    """Get GEOCENTRIC ecliptic longitude of a planet in degrees.
    Uses ephem.Ecliptic() for proper geocentric coordinates (needed for
    retrograde detection, aspects, dignities, etc.)."""
    planet = PLANETS[planet_name]()
    planet.compute(_to_ephem_date(dt))
    ecl = ephem.Ecliptic(planet)
    lon_deg = math.degrees(float(ecl.lon))
    return lon_deg % 360


def get_planet_zodiac(planet_name, dt):
    """Get which zodiac sign a planet is in."""
    lon = get_planet_position(planet_name, dt)
    sign_idx = int(lon / 30) % 12
    degree_in_sign = lon % 30
    return ZODIAC_SIGNS[sign_idx], degree_in_sign


def get_sun_zodiac(dt):
    """Get current zodiac sign of the Sun."""
    sun = ephem.Sun()
    sun.compute(_to_ephem_date(dt))
    lon = math.degrees(float(ephem.Ecliptic(sun).lon))
    sign_idx = int(lon / 30) % 12
    return ZODIAC_SIGNS[sign_idx]


# ===========================================================================
# MERCURY RETROGRADE
# ===========================================================================

def is_mercury_retrograde(dt):
    """Check if Mercury is in retrograde motion.
    Mercury retrogrades ~3-4 times per year for ~3 weeks each.
    Returns: (is_retrograde: bool, days_in_retrograde: int or None)
    """
    m = ephem.Mercury()
    m.compute(_to_ephem_date(dt))

    # Check if Mercury's geocentric longitude is decreasing (retrograde)
    # Compare position yesterday vs today using geocentric ecliptic coords
    m_today = ephem.Mercury()
    m_today.compute(_to_ephem_date(dt))

    m_yesterday = ephem.Mercury()
    m_yesterday.compute(_to_ephem_date(dt - timedelta(days=1)))

    lon_today = math.degrees(float(ephem.Ecliptic(m_today).lon)) % 360
    lon_yesterday = math.degrees(float(ephem.Ecliptic(m_yesterday).lon)) % 360

    # Handle wrap-around at 360/0
    diff = lon_today - lon_yesterday
    if diff > 180:
        diff -= 360
    elif diff < -180:
        diff += 360

    is_retro = diff < 0  # Moving backward = retrograde

    if is_retro:
        # Count days into retrograde
        days = 0
        check_dt = dt
        while days < 30:
            m_check = ephem.Mercury()
            m_check.compute(_to_ephem_date(check_dt))
            m_prev = ephem.Mercury()
            m_prev.compute(_to_ephem_date(check_dt - timedelta(days=1)))
            lon_c = math.degrees(float(ephem.Ecliptic(m_check).lon)) % 360
            lon_p = math.degrees(float(ephem.Ecliptic(m_prev).lon)) % 360
            d = lon_c - lon_p
            if d > 180:
                d -= 360
            elif d < -180:
                d += 360
            if d >= 0:
                break
            days += 1
            check_dt -= timedelta(days=1)
        return True, days
    return False, None


def is_planet_retrograde(planet_name, dt):
    """Check if a planet is retrograde (geocentric motion going backward)."""
    p_today = PLANETS[planet_name]()
    p_today.compute(_to_ephem_date(dt))
    p_yesterday = PLANETS[planet_name]()
    p_yesterday.compute(_to_ephem_date(dt - timedelta(days=1)))

    lon_today = math.degrees(float(ephem.Ecliptic(p_today).lon)) % 360
    lon_yesterday = math.degrees(float(ephem.Ecliptic(p_yesterday).lon)) % 360

    diff = lon_today - lon_yesterday
    if diff > 180:
        diff -= 360
    elif diff < -180:
        diff += 360

    return diff < 0


# ===========================================================================
# ECLIPSES
# ===========================================================================

def _is_near_eclipse(dt):
    """Check if date is near a solar or lunar eclipse by checking:
    - Sun-Moon conjunction (new moon) or opposition (full moon)
    - Near the lunar nodes (within ~18 degrees for eclipses to occur)
    Returns: (is_eclipse: bool, eclipse_type: str or None)
    """
    try:
        sun_lon = get_sun_ecliptic_lon(dt)
        moon_lon = get_moon_ecliptic_lon(dt)

        # Get North Node longitude
        T = (ephem.Date(dt) - ephem.Date('2000/1/1.5')) / 36525.0
        north_node_lon = (125.0445479 - 1934.1362891 * T + 0.0020754 * T * T) % 360

        # Check Sun-Moon angle
        sun_moon_angle = abs(sun_lon - moon_lon) % 360
        if sun_moon_angle > 180:
            sun_moon_angle = 360 - sun_moon_angle

        # Check Sun proximity to nodes
        sun_node_angle = abs(sun_lon - north_node_lon) % 360
        if sun_node_angle > 180:
            sun_node_angle = 360 - sun_node_angle
        # Also check south node
        sun_snode_angle = abs(sun_lon - (north_node_lon + 180) % 360) % 360
        if sun_snode_angle > 180:
            sun_snode_angle = 360 - sun_snode_angle
        near_node = min(sun_node_angle, sun_snode_angle) <= 30  # wider orb for eclipse season

        if near_node:
            if sun_moon_angle <= 12:  # New moon near node = solar eclipse
                return True, "solar"
            elif abs(sun_moon_angle - 180) <= 12:  # Full moon near node = lunar eclipse
                return True, "lunar"

        return False, None
    except (ephem.AlwaysUpError, ephem.NeverUpError, ephem.CircumpolarError) as e:
        logging.debug(f"Circumpolar condition: {e}")
        return np.nan, None
    except Exception as e:
        logging.warning(f"Astrology calc failed: {e}")
        return np.nan, None


def is_eclipse_window(dt, window_days=7):
    """Check if within window_days of any eclipse.
    Scans nearby days for eclipse conditions.
    Returns: (in_window: bool, eclipse_type: str or None, days_to_eclipse: int or None)
    """
    # Check each day in window range
    for d in range(0, window_days + 1):
        # Forward
        is_ecl, e_type = _is_near_eclipse(dt + timedelta(days=d))
        if is_ecl:
            return True, e_type, d
        # Backward
        if d > 0:
            is_ecl, e_type = _is_near_eclipse(dt - timedelta(days=d))
            if is_ecl:
                return True, e_type, -d

    return False, None, None


# ===========================================================================
# PLANETARY ASPECTS
# ===========================================================================

def get_aspect(planet1, planet2, dt):
    """Check if two planets are in a major aspect.
    Returns: (aspect_name, exactness_degrees) or (None, None)
    """
    lon1 = get_planet_position(planet1, dt)
    lon2 = get_planet_position(planet2, dt)

    angle = abs(lon1 - lon2) % 360
    if angle > 180:
        angle = 360 - angle

    for aspect_name, (target_angle, orb) in ASPECTS.items():
        if abs(angle - target_angle) <= orb:
            exactness = abs(angle - target_angle)
            return aspect_name, exactness

    return None, None


def get_active_aspects(dt, min_planets=None):
    """Get all active planetary aspects for a date.
    Returns: list of (planet1, planet2, aspect_name, exactness)
    """
    planets = list(PLANETS.keys()) if min_planets is None else min_planets
    aspects = []

    for i in range(len(planets)):
        for j in range(i + 1, len(planets)):
            aspect, exactness = get_aspect(planets[i], planets[j], dt)
            if aspect:
                aspects.append((planets[i], planets[j], aspect, exactness))

    return aspects


def count_hard_aspects(dt):
    """Count 'hard' aspects (conjunction, opposition, square) — volatility indicator."""
    aspects = get_active_aspects(dt)
    hard = [a for a in aspects if a[2] in ("conjunction", "opposition", "square")]
    return len(hard), hard


def count_soft_aspects(dt):
    """Count 'soft' aspects (trine, sextile) — harmony/stability indicator."""
    aspects = get_active_aspects(dt)
    soft = [a for a in aspects if a[2] in ("trine", "sextile")]
    return len(soft), soft


def get_regime_astrology(dt):
    """Get astrological regime indicators for market regime detection."""
    try:
        n_hard, _ = count_hard_aspects(dt)
        n_soft, _ = count_soft_aspects(dt)
        ratio = n_hard / max(n_soft, 1)
        return {"hard": n_hard, "soft": n_soft, "ratio": ratio}
    except:
        return {"hard": 3, "soft": 3, "ratio": 1.0}


# ===========================================================================
# ZODIAC INGRESS (Sun entering new sign)
# ===========================================================================

def is_zodiac_ingress(dt, window_days=2):
    """Check if Sun is transitioning between zodiac signs.
    Returns: (is_ingress, from_sign, to_sign, days_to_transition)
    """
    current_sign = get_sun_zodiac(dt)
    for d in range(1, window_days + 1):
        future_sign = get_sun_zodiac(dt + timedelta(days=d))
        if future_sign != current_sign:
            return True, current_sign, future_sign, d
        past_sign = get_sun_zodiac(dt - timedelta(days=d))
        if past_sign != current_sign:
            return True, past_sign, current_sign, -d
    return False, current_sign, None, None


# ===========================================================================
# VOID OF COURSE MOON
# ===========================================================================

def is_void_of_course_moon(dt):
    """Simplified void of course detection.
    VOC = Moon has made its last major aspect in current sign before changing signs.
    Returns: (is_voc: bool, hours_until_sign_change: float or None)
    """
    moon = ephem.Moon()
    moon.compute(_to_ephem_date(dt))
    moon_lon = math.degrees(float(ephem.Ecliptic(moon).lon)) % 360
    current_sign_idx = int(moon_lon / 30)
    degree_in_sign = moon_lon % 30

    # Moon moves ~13 degrees/day = ~0.54 deg/hour
    # If in last 5 degrees of sign, likely VOC
    degrees_remaining = 30 - degree_in_sign
    hours_to_change = degrees_remaining / 0.54

    if degree_in_sign > 25:  # Last 5 degrees = likely VOC
        # Check if any major aspect is still ahead
        has_upcoming_aspect = False
        for hours_ahead in range(1, int(hours_to_change) + 1):
            future_dt = dt + timedelta(hours=hours_ahead)
            for planet_name in ["Mercury", "Venus", "Mars", "Jupiter", "Saturn"]:
                aspect, _ = get_aspect("Moon_approx", planet_name, future_dt)
                # Skip Moon aspect check (would need special handling)
                break
        return True, hours_to_change
    return False, None


# ===========================================================================
# BITCOIN NATAL CHART TRANSITS
# ===========================================================================

def get_btc_natal_positions():
    """Get planetary positions at Bitcoin's birth (genesis block)."""
    positions = {}
    for name in PLANETS:
        positions[name] = get_planet_position(name, BTC_BIRTH)
    return positions


def get_btc_transits(dt):
    """Check current planetary transits to Bitcoin's natal chart.
    Returns list of (transiting_planet, natal_planet, aspect, exactness)
    """
    natal = get_btc_natal_positions()
    transits = []

    for t_name in PLANETS:
        t_lon = get_planet_position(t_name, dt)
        for n_name, n_lon in natal.items():
            angle = abs(t_lon - n_lon) % 360
            if angle > 180:
                angle = 360 - angle
            for aspect_name, (target, orb) in ASPECTS.items():
                if abs(angle - target) <= orb:
                    transits.append((t_name, n_name, aspect_name, abs(angle - target)))

    return transits


def get_btc_transit_score(dt):
    """Score BTC transits: hard aspects = bearish, soft = bullish.
    Returns: (score: float, detail: str)
    Score: negative = bearish transits dominate, positive = bullish
    """
    transits = get_btc_transits(dt)
    score = 0.0
    details = []

    for t_planet, n_planet, aspect, exactness in transits:
        # Weight by exactness (tighter = stronger)
        strength = 1.0 - (exactness / 8.0)

        if aspect in ("square", "opposition"):
            # Hard aspects to BTC natal = bearish pressure
            weight = -strength
            # Mars/Saturn hard aspects are worse
            if t_planet in ("Mars", "Saturn", "Pluto"):
                weight *= 1.5
        elif aspect == "conjunction":
            # Conjunctions depend on planet
            if t_planet in ("Jupiter", "Venus"):
                weight = strength  # Benefic conjunction = bullish
            elif t_planet in ("Saturn", "Mars", "Pluto"):
                weight = -strength  # Malefic conjunction = bearish
            else:
                weight = strength * 0.5
        elif aspect in ("trine", "sextile"):
            # Soft aspects = bullish flow
            weight = strength * 0.7
            if t_planet in ("Jupiter", "Venus"):
                weight *= 1.3
        else:
            weight = 0

        score += weight
        if abs(weight) > 0.3:
            details.append(f"{t_planet} {aspect} natal {n_planet} ({exactness:.1f}deg)")

    return score, "; ".join(details[:5])


# ===========================================================================
# GREAT CONJUNCTIONS (Saturn-Jupiter)
# ===========================================================================

# Known great conjunctions near our data range
GREAT_CONJUNCTIONS = [
    datetime(2000, 5, 28),   # In Taurus
    datetime(2020, 12, 21),  # In Aquarius (pandemic era)
    datetime(2040, 10, 31),  # In Libra (future)
]


def days_from_great_conjunction(dt, window_days=30):
    """Check proximity to Saturn-Jupiter conjunction.
    Discord noted: 66w6d (468 days) pattern between great conjunctions and events.
    Returns: (near_conjunction: bool, days_since_last: int, pattern_468d: bool)
    """
    for gc in GREAT_CONJUNCTIONS:
        days_diff = abs((dt - gc).days)
        if days_diff <= window_days:
            return True, (dt - gc).days, False
        # Check 468-day pattern (66w6d from Discord)
        if abs(days_diff - 468) <= 7:
            return False, (dt - gc).days, True
        if abs(days_diff - 468 * 2) <= 7:
            return False, (dt - gc).days, True

    # Return days since last
    last_gc = max(gc for gc in GREAT_CONJUNCTIONS if gc <= dt) if any(gc <= dt for gc in GREAT_CONJUNCTIONS) else None
    return False, (dt - last_gc).days if last_gc else 9999, False


# ===========================================================================
# COMPREHENSIVE ASTROLOGY SIGNAL GENERATOR
# ===========================================================================

def get_astrology_signals(dt):
    """Generate all astrology trading signals for a given datetime.
    Returns: list of (signal_name, direction, strength, detail)
    """
    signals = []

    # 1. Mercury retrograde
    is_retro, retro_days = is_mercury_retrograde(dt)
    if is_retro:
        # Mercury retrograde = caution/reversal energy
        # First few days = most volatile
        strength = 7 if retro_days and retro_days <= 3 else 5
        signals.append(("MERCURY_RETRO", "short", strength,
                        f"Mercury retrograde day {retro_days}"))

    # 2. Eclipse window
    in_eclipse, e_type, e_days = is_eclipse_window(dt, window_days=5)
    if in_eclipse:
        # Eclipses = turning points, increased volatility
        strength = 6 if abs(e_days or 0) <= 2 else 4
        signals.append(("ECLIPSE_WINDOW", "volatile", strength,
                        f"{e_type} eclipse {e_days}d away"))

    # 3. Hard aspects (volatility/bearish)
    n_hard, hard_list = count_hard_aspects(dt)
    if n_hard >= 3:
        # Multiple hard aspects = high tension
        signals.append(("HARD_ASPECTS", "short", min(n_hard, 8),
                        f"{n_hard} hard aspects active"))
    elif n_hard == 0:
        n_soft, _ = count_soft_aspects(dt)
        if n_soft >= 3:
            signals.append(("SOFT_ASPECTS", "long", min(n_soft, 6),
                            f"{n_soft} soft aspects, no hard"))

    # 4. BTC natal transits
    btc_score, btc_detail = get_btc_transit_score(dt)
    if abs(btc_score) > 1.5:
        direction = "long" if btc_score > 0 else "short"
        strength = min(int(abs(btc_score) * 2), 8)
        signals.append(("BTC_TRANSIT", direction, strength, btc_detail))

    # 5. Zodiac ingress
    is_ingress, from_s, to_s, days_to = is_zodiac_ingress(dt)
    if is_ingress:
        signals.append(("ZODIAC_INGRESS", "volatile", 3,
                        f"Sun {from_s}->{to_s} in {days_to}d"))

    # 6. Great conjunction proximity
    near_gc, days_since, pattern_468 = days_from_great_conjunction(dt)
    if near_gc:
        signals.append(("GREAT_CONJUNCTION", "volatile", 7,
                        f"{days_since}d from great conjunction"))
    elif pattern_468:
        signals.append(("GC_468_PATTERN", "short", 5,
                        f"468d pattern from great conjunction"))

    # 7. Multiple planet retrogrades
    retro_planets = [p for p in PLANETS if is_planet_retrograde(p, dt)]
    if len(retro_planets) >= 4:
        signals.append(("MULTI_RETROGRADE", "short", 6,
                        f"{len(retro_planets)} planets retrograde: {', '.join(retro_planets[:4])}"))

    # 8. Mars aspects (aggression/volatility)
    for aspect_info in get_active_aspects(dt, ["Mars"]):
        p1, p2, aspect, exactness = aspect_info
        other = p2 if p1 == "Mars" else p1
        if aspect in ("square", "opposition") and other in ("Saturn", "Uranus", "Pluto"):
            if exactness < 3:  # Tight aspect
                signals.append(("MARS_TENSION", "short", 6,
                                f"Mars {aspect} {other} ({exactness:.1f}deg)"))

    return signals


# ===========================================================================
# MATRIXOLOGY TIME-BASED SIGNALS
# ===========================================================================

def is_6am_utc_window(dt):
    """6am UTC bottom detector — is current hour in the 5-7am UTC window?
    For 1H/4H candles this marks potential bottom zone.
    Returns: bool
    """
    return 5 <= dt.hour <= 7


def is_midnight_utc_reversal(dt):
    """Midnight UTC reversal detector — within 1 hour of midnight UTC.
    Returns: bool
    """
    return dt.hour == 0 or dt.hour == 23


def get_6h_cycle_phase(dt):
    """Divide day into 4 blocks of 6 hours each:
    Block 0: 00-06 UTC (Asia close)
    Block 1: 06-12 UTC (Europe open)
    Block 2: 12-18 UTC (US open)
    Block 3: 18-24 UTC (US close/Asia open)
    Returns: (block_number: int, block_name: str)
    """
    block = dt.hour // 6
    names = {0: "Asia_close", 1: "Europe_open", 2: "US_open", 3: "US_close_Asia_open"}
    return block, names[block]


def is_5day_peak_crash_lag(dt, price_history):
    """Check if BTC made a local high 4-6 days ago (peak-to-crash lag).
    price_history: list of (datetime, high_price) tuples sorted by date descending.
    Returns: bool
    """
    if len(price_history) < 10:
        return False
    # Look at prices 4-6 days ago and compare to surrounding days
    for lag in range(4, 7):
        if lag >= len(price_history):
            continue
        peak_price = price_history[lag][1]
        # Check if it's higher than 3 days before and after it
        is_local_high = True
        for offset in range(1, 4):
            before_idx = lag + offset
            after_idx = lag - offset
            if before_idx < len(price_history) and price_history[before_idx][1] >= peak_price:
                is_local_high = False
                break
            if 0 <= after_idx < len(price_history) and after_idx != lag and price_history[after_idx][1] >= peak_price:
                is_local_high = False
                break
        if is_local_high:
            return True
    return False


def is_pump_date(dt):
    """14th/15th/16th of month = pump dates.
    Returns: bool
    """
    return dt.day in (14, 15, 16)


def is_tuesday_after_monday_pump(dt, monday_return=None):
    """Tuesday = top of pump. Check if it's Tuesday AND Monday was a pump day.
    dt: current datetime
    monday_return: float, Monday's return (close/open - 1). If None, just check if Tuesday.
    Returns: (is_tuesday: bool, is_tuesday_after_pump: bool)
    """
    is_tue = dt.weekday() == 1  # 0=Mon, 1=Tue
    if monday_return is not None and is_tue:
        return is_tue, monday_return > 0.005  # Monday pumped > 0.5%
    return is_tue, False


# ===========================================================================
# IMPROVED VOID OF COURSE MOON (Classical Method)
# ===========================================================================

CLASSICAL_PLANETS = ["Mercury", "Venus", "Mars", "Jupiter", "Saturn"]
# Sun is also used for aspects but computed differently

MAJOR_ASPECT_ANGLES = [0, 60, 90, 120, 180]  # conjunction, sextile, square, trine, opposition
ASPECT_ORB_MOON = 6  # degrees orb for Moon aspects


def get_moon_ecliptic_lon(dt):
    """Get Moon's ecliptic longitude in degrees."""
    moon = ephem.Moon()
    moon.compute(_to_ephem_date(dt))
    return math.degrees(float(ephem.Ecliptic(moon).lon)) % 360


def get_sun_ecliptic_lon(dt):
    """Get Sun's ecliptic longitude in degrees."""
    sun = ephem.Sun()
    sun.compute(_to_ephem_date(dt))
    return math.degrees(float(ephem.Ecliptic(sun).lon)) % 360


def is_voc_moon_classical(dt):
    """Improved Void of Course Moon using classical method.
    Check if Moon makes any major aspect to classical planets before leaving sign.
    If no aspects ahead -> VOC = True = DO NOT TRADE.
    Returns: (is_voc: bool, hours_until_sign_change: float)
    """
    try:
        moon_lon = get_moon_ecliptic_lon(dt)
        current_sign_idx = int(moon_lon / 30)
        degree_in_sign = moon_lon % 30
        degrees_remaining = 30 - degree_in_sign
        hours_to_change = degrees_remaining / 0.55  # Moon speed ~13.2 deg/day

        # Check if Moon makes any major aspect before leaving current sign
        # Step through time in 2-hour increments until sign change
        steps = max(1, int(hours_to_change / 2))
        for step in range(1, steps + 1):
            future_dt = dt + timedelta(hours=step * 2)
            future_moon_lon = get_moon_ecliptic_lon(future_dt)
            future_sign_idx = int(future_moon_lon / 30)

            # Stop if Moon has changed signs
            if future_sign_idx != current_sign_idx:
                break

            # Check aspects to Sun
            sun_lon = get_sun_ecliptic_lon(future_dt)
            angle = abs(future_moon_lon - sun_lon) % 360
            if angle > 180:
                angle = 360 - angle
            for asp_angle in MAJOR_ASPECT_ANGLES:
                if abs(angle - asp_angle) <= ASPECT_ORB_MOON:
                    return False, hours_to_change  # Aspect found, NOT VOC

            # Check aspects to classical planets
            for pname in CLASSICAL_PLANETS:
                p_lon = get_planet_position(pname, future_dt)
                angle = abs(future_moon_lon - p_lon) % 360
                if angle > 180:
                    angle = 360 - angle
                for asp_angle in MAJOR_ASPECT_ANGLES:
                    if abs(angle - asp_angle) <= ASPECT_ORB_MOON:
                        return False, hours_to_change  # Aspect found, NOT VOC

        return True, hours_to_change  # No aspects found -> VOC
    except (ephem.AlwaysUpError, ephem.NeverUpError, ephem.CircumpolarError) as e:
        logging.debug(f"Circumpolar condition: {e}")
        return np.nan, None
    except Exception as e:
        logging.warning(f"Astrology calc failed: {e}")
        return np.nan, None


# ===========================================================================
# MOON NODES (North Node / South Node)
# ===========================================================================

def get_moon_node_signal(dt):
    """Moon conjunct North Node = positive; Moon conjunct South Node = negative.
    Uses ephem to get node positions.
    Returns: (signal: str or None, orb: float)
      signal: 'north_conjunction', 'south_conjunction', or None
    """
    try:
        moon_lon = get_moon_ecliptic_lon(dt)

        # Get Moon's mean ascending node (North Node)
        # ephem doesn't have a direct node object, compute from Moon
        moon = ephem.Moon()
        moon.compute(_to_ephem_date(dt))
        # Moon's ascending node (north node) - approximate via ephem
        # We can get it from the Moon's node attribute
        # Actually use the formula: compute ecliptic lat crossings
        # Simpler: use the Moon's ascending node longitude
        # ephem provides this as moon.hlat crossings
        # Best approach: check multiple days to find node
        # Alternative: use the formula from Meeus
        # For speed, use J2000 epoch calculation
        T = (ephem.Date(dt) - ephem.Date('2000/1/1.5')) / 36525.0
        # Mean longitude of ascending node (degrees)
        north_node_lon = (125.0445479 - 1934.1362891 * T
                          + 0.0020754 * T * T) % 360
        south_node_lon = (north_node_lon + 180) % 360

        # Check Moon conjunction with North Node
        angle_north = abs(moon_lon - north_node_lon) % 360
        if angle_north > 180:
            angle_north = 360 - angle_north
        if angle_north <= 8:
            return 'north_conjunction', angle_north

        # Check Moon conjunction with South Node
        angle_south = abs(moon_lon - south_node_lon) % 360
        if angle_south > 180:
            angle_south = 360 - angle_south
        if angle_south <= 8:
            return 'south_conjunction', angle_south

        return None, None
    except (ephem.AlwaysUpError, ephem.NeverUpError, ephem.CircumpolarError) as e:
        logging.debug(f"Circumpolar condition: {e}")
        return np.nan, None
    except Exception as e:
        logging.warning(f"Astrology calc failed: {e}")
        return np.nan, None


# ===========================================================================
# PLANETARY DIGNITY SCORING
# ===========================================================================

# Domicile (rulership) assignments: planet -> list of signs ruled
DOMICILE = {
    "Sun": ["Leo"],
    "Moon_node": ["Cancer"],  # placeholder - won't be used directly
    "Mercury": ["Gemini", "Virgo"],
    "Venus": ["Taurus", "Libra"],
    "Mars": ["Aries", "Scorpio"],
    "Jupiter": ["Sagittarius", "Pisces"],
    "Saturn": ["Capricorn", "Aquarius"],
    "Uranus": ["Aquarius"],
    "Neptune": ["Pisces"],
    "Pluto": ["Scorpio"],
}

EXALTATION = {
    "Sun": "Aries",
    "Mercury": "Virgo",
    "Venus": "Pisces",
    "Mars": "Capricorn",
    "Jupiter": "Cancer",
    "Saturn": "Libra",
    "Uranus": "Scorpio",
    "Neptune": "Cancer",
    "Pluto": "Leo",
}

DETRIMENT = {
    "Sun": ["Aquarius"],
    "Mercury": ["Sagittarius", "Pisces"],
    "Venus": ["Aries", "Scorpio"],
    "Mars": ["Taurus", "Libra"],
    "Jupiter": ["Gemini", "Virgo"],
    "Saturn": ["Cancer", "Leo"],
    "Uranus": ["Leo"],
    "Neptune": ["Virgo"],
    "Pluto": ["Taurus"],
}

FALL = {
    "Sun": "Libra",
    "Mercury": "Pisces",
    "Venus": "Virgo",
    "Mars": "Cancer",
    "Jupiter": "Capricorn",
    "Saturn": "Aries",
    "Uranus": "Taurus",
    "Neptune": "Capricorn",
    "Pluto": "Aquarius",
}

# Triplicity rulers (day chart): element -> planet
TRIPLICITY_DAY = {
    "Fire": "Sun",     # Aries, Leo, Sagittarius
    "Earth": "Venus",  # Taurus, Virgo, Capricorn
    "Air": "Saturn",   # Gemini, Libra, Aquarius
    "Water": "Mars",   # Cancer, Scorpio, Pisces
}

SIGN_ELEMENT = {
    "Aries": "Fire", "Leo": "Fire", "Sagittarius": "Fire",
    "Taurus": "Earth", "Virgo": "Earth", "Capricorn": "Earth",
    "Gemini": "Air", "Libra": "Air", "Aquarius": "Air",
    "Cancer": "Water", "Scorpio": "Water", "Pisces": "Water",
}


def get_planet_dignity_score(planet_name, dt):
    """Score a planet's dignity: domicile=+5, exaltation=+4, triplicity=+3,
    detriment=-5, fall=-4. Returns int score."""
    try:
        sign, deg = get_planet_zodiac(planet_name, dt)
        score = 0

        # Domicile
        if planet_name in DOMICILE and sign in DOMICILE[planet_name]:
            score += 5

        # Exaltation
        if planet_name in EXALTATION and sign == EXALTATION[planet_name]:
            score += 4

        # Triplicity (using day chart ruler)
        if sign in SIGN_ELEMENT:
            element = SIGN_ELEMENT[sign]
            if element in TRIPLICITY_DAY and TRIPLICITY_DAY[element] == planet_name:
                score += 3

        # Detriment
        if planet_name in DETRIMENT and sign in DETRIMENT[planet_name]:
            score -= 5

        # Fall
        if planet_name in FALL and sign == FALL[planet_name]:
            score -= 4

        # Term (+2) and Face (+1) - simplified: based on degree position
        # Face: divide each sign into 3 faces of 10 degrees each
        face_idx = int(deg / 10)
        # Simplified face rulers (Chaldean order repeating)
        chaldean = ["Saturn", "Jupiter", "Mars", "Sun_skip", "Venus", "Mercury", "Moon_skip"]
        sign_idx = ZODIAC_SIGNS.index(sign)
        face_ruler_idx = (sign_idx * 3 + face_idx) % 7
        if face_ruler_idx < len(chaldean) and chaldean[face_ruler_idx] == planet_name:
            score += 1

        return score
    except (ephem.AlwaysUpError, ephem.NeverUpError, ephem.CircumpolarError) as e:
        logging.debug(f"Circumpolar condition: {e}")
        return np.nan
    except Exception as e:
        logging.warning(f"Astrology calc failed: {e}")
        return np.nan


def get_planetary_strength_index(dt):
    """Sum all planet dignity scores for a daily 'planetary strength index'.
    Positive = strong/dignified sky, negative = debilitated sky.
    Returns: (total_score: int, details: dict)
    """
    total = 0
    details = {}
    for pname in PLANETS:
        s = get_planet_dignity_score(pname, dt)
        details[pname] = s
        total += s
    return total, details


# ===========================================================================
# SATURN / MARS / VENUS RETROGRADE STATIONS & SPECIAL SIGNALS
# ===========================================================================

def is_saturn_station(dt, window_days=3):
    """Saturn retrograde STATION — within window_days of Saturn going retrograde or direct.
    This is a stress event.
    Returns: (is_station: bool, station_type: str or None)
    """
    try:
        retro_today = is_planet_retrograde("Saturn", dt)
        for d in range(1, window_days + 1):
            retro_past = is_planet_retrograde("Saturn", dt - timedelta(days=d))
            retro_future = is_planet_retrograde("Saturn", dt + timedelta(days=d))
            if retro_past != retro_today:
                return True, "station_retrograde" if retro_today else "station_direct"
            if retro_future != retro_today:
                return True, "station_retrograde" if retro_future else "station_direct"
        return False, None
    except (ephem.AlwaysUpError, ephem.NeverUpError, ephem.CircumpolarError) as e:
        logging.debug(f"Circumpolar condition: {e}")
        return np.nan, None
    except Exception as e:
        logging.warning(f"Astrology calc failed: {e}")
        return np.nan, None


def is_mars_retrograde_signal(dt):
    """Mars retrograde = false breakouts signal.
    Returns: bool
    """
    return is_planet_retrograde("Mars", dt)


def is_venus_retrograde_signal(dt):
    """Venus retrograde = value reassessment.
    Returns: bool
    """
    return is_planet_retrograde("Venus", dt)


# ===========================================================================
# INGRESS CHART BIAS (Quarterly Shifts)
# ===========================================================================

CARDINAL_SIGNS = ["Aries", "Cancer", "Libra", "Capricorn"]


def get_ingress_chart_bias(dt, window_days=7):
    """When Sun enters Aries/Cancer/Libra/Capricorn = quarterly shift.
    Check aspects at ingress moment for bias.
    Returns: (near_ingress: bool, sign: str or None, aspect_score: float)
    """
    try:
        current_sign = get_sun_zodiac(dt)

        # Check if Sun recently entered or is about to enter a cardinal sign
        for d in range(-window_days, window_days + 1):
            check_dt = dt + timedelta(days=d)
            sign_at = get_sun_zodiac(check_dt)
            if sign_at in CARDINAL_SIGNS:
                # Check if this is near the actual ingress (sign change)
                prev_sign = get_sun_zodiac(check_dt - timedelta(days=1))
                if prev_sign != sign_at:
                    # This is the ingress day
                    # Score aspects at ingress moment
                    aspects = get_active_aspects(check_dt)
                    score = 0.0
                    for p1, p2, asp, ex in aspects:
                        strength = 1.0 - (ex / 8.0)
                        if asp in ("trine", "sextile"):
                            score += strength
                        elif asp in ("square", "opposition"):
                            score -= strength
                        elif asp == "conjunction":
                            if p1 in ("Jupiter", "Venus") or p2 in ("Jupiter", "Venus"):
                                score += strength
                            elif p1 in ("Saturn", "Mars") or p2 in ("Saturn", "Mars"):
                                score -= strength
                    return True, sign_at, score

        return False, None, 0.0
    except (ephem.AlwaysUpError, ephem.NeverUpError, ephem.CircumpolarError) as e:
        logging.debug(f"Circumpolar condition: {e}")
        return np.nan, None, np.nan
    except Exception as e:
        logging.warning(f"Astrology calc failed: {e}")
        return np.nan, None, np.nan


# ===========================================================================
# ZODIACAL RELEASING FROM LOT OF SPIRIT (Bitcoin natal chart)
# ===========================================================================

# Bitcoin birth: Jan 3, 2009, 18:15:05 UTC
# For Zodiacal Releasing we need the Lot of Spirit
# Lot of Spirit = Asc + Moon - Sun (day chart) or Asc + Sun - Moon (night chart)

# Bitcoin natal positions (pre-computed for genesis block)
# We need Ascendant which requires location — Bitcoin is "born" everywhere,
# but convention uses 0 Aries rising for mundane charts.
# Alternative: use London (0 long) or just compute from ephem

# Minor periods for each sign (years)
ZR_MINOR_PERIODS = {
    "Aries": 15, "Taurus": 8, "Gemini": 20, "Cancer": 25,
    "Leo": 19, "Virgo": 20, "Libra": 8, "Scorpio": 15,
    "Sagittarius": 12, "Capricorn": 27, "Aquarius": 27, "Pisces": 12
}


def _get_btc_natal_asc_lon():
    """Compute Bitcoin's Ascendant. Using 0 Aries rising convention (0 degrees).
    Some astrologers use the IC of the chart; we'll use a computed value.
    Bitcoin was born Jan 3, 2009, 18:15:05 UTC.
    For a mundane/event chart, we compute local sidereal time.
    Using 0E 0N (null island) as Bitcoin has no physical location.
    """
    try:
        obs = ephem.Observer()
        obs.lon = '0'  # Greenwich
        obs.lat = '0'
        obs.date = _to_ephem_date(BTC_BIRTH)
        # Local sidereal time gives us the Ascendant approximately
        lst = float(obs.sidereal_time())  # radians
        asc_lon = math.degrees(lst) % 360
        return asc_lon
    except (ephem.AlwaysUpError, ephem.NeverUpError, ephem.CircumpolarError) as e:
        logging.debug(f"Circumpolar condition: {e}")
        return np.nan
    except Exception as e:
        logging.warning(f"Astrology calc failed: {e}")
        return np.nan


def get_btc_lot_of_spirit():
    """Calculate Bitcoin's Lot of Spirit.
    Day chart: Asc + Sun - Moon
    Night chart: Asc + Moon - Sun
    Bitcoin born 18:15 UTC -> Sun below horizon at 0N/0E in January -> night chart.
    Returns: lot_lon in degrees
    """
    try:
        asc = _get_btc_natal_asc_lon()
        sun_lon = get_sun_ecliptic_lon(BTC_BIRTH)
        moon_lon = get_moon_ecliptic_lon(BTC_BIRTH)

        # Night chart Spirit: Asc + Moon - Sun
        lot = (asc + moon_lon - sun_lon) % 360
        return lot
    except (ephem.AlwaysUpError, ephem.NeverUpError, ephem.CircumpolarError) as e:
        logging.debug(f"Circumpolar condition: {e}")
        return np.nan
    except Exception as e:
        logging.warning(f"Astrology calc failed: {e}")
        return np.nan


def get_btc_lot_of_fortune():
    """Calculate Bitcoin's Lot of Fortune.
    Day: Asc + Moon - Sun; Night: Asc + Sun - Moon (reversed from Spirit)
    Actually: Fortune day = Asc + Moon - Sun; Fortune night = Asc + Sun - Moon
    Spirit day = Asc + Sun - Moon; Spirit night = Asc + Moon - Sun
    Bitcoin = night chart, so Fortune = Asc + Sun - Moon, Spirit = Asc + Moon - Sun
    Wait, let's be precise:
    Fortune: Day = Asc + Moon - Sun, Night = Asc + Sun - Moon
    Spirit:  Day = Asc + Sun - Moon, Night = Asc + Moon - Sun
    """
    try:
        asc = _get_btc_natal_asc_lon()
        sun_lon = get_sun_ecliptic_lon(BTC_BIRTH)
        moon_lon = get_moon_ecliptic_lon(BTC_BIRTH)
        # Night chart Fortune = Asc + Sun - Moon
        lot = (asc + sun_lon - moon_lon) % 360
        return lot
    except (ephem.AlwaysUpError, ephem.NeverUpError, ephem.CircumpolarError) as e:
        logging.debug(f"Circumpolar condition: {e}")
        return np.nan
    except Exception as e:
        logging.warning(f"Astrology calc failed: {e}")
        return np.nan


def get_zodiacal_releasing_sign(dt):
    """Calculate current Zodiacal Releasing sign from Lot of Spirit for Bitcoin.
    Walk through the minor periods from birth until we reach current date.
    Returns: (current_zr_sign: str, is_peak_period: bool)
    Peak = when releasing reaches 10th sign from Lot of Fortune.
    """
    try:
        lot_spirit = get_btc_lot_of_spirit()
        lot_fortune = get_btc_lot_of_fortune()

        lot_spirit_sign_idx = int(lot_spirit / 30) % 12
        lot_fortune_sign_idx = int(lot_fortune / 30) % 12
        peak_sign_idx = (lot_fortune_sign_idx + 9) % 12  # 10th sign = +9 index

        # Walk through signs starting from Lot of Spirit sign
        total_days = (dt - BTC_BIRTH).days
        current_sign_idx = lot_spirit_sign_idx
        days_elapsed = 0

        # Safety: limit iterations
        for _ in range(200):
            sign_name = ZODIAC_SIGNS[current_sign_idx]
            period_days = ZR_MINOR_PERIODS[sign_name] * 365.25
            if days_elapsed + period_days > total_days:
                # We're in this sign
                is_peak = (current_sign_idx == peak_sign_idx)
                return sign_name, is_peak
            days_elapsed += period_days
            current_sign_idx = (current_sign_idx + 1) % 12

        return ZODIAC_SIGNS[lot_spirit_sign_idx], False
    except (ephem.AlwaysUpError, ephem.NeverUpError, ephem.CircumpolarError) as e:
        logging.debug(f"Circumpolar condition: {e}")
        return np.nan, np.nan
    except Exception as e:
        logging.warning(f"Astrology calc failed: {e}")
        return np.nan, np.nan


# ===========================================================================
# EXTENDED ECLIPSE WINDOW (Ptolemy — 30 days)
# ===========================================================================

def is_eclipse_window_extended(dt, window_days=30):
    """Extended eclipse effect window per Ptolemy (30 days, not 5-7).
    Returns: (in_window: bool, eclipse_type: str or None, days_to_eclipse: int or None)
    """
    return is_eclipse_window(dt, window_days=window_days)


# ===========================================================================
# COMPREHENSIVE NEW SIGNALS GENERATOR
# ===========================================================================

def get_new_astrology_signals(dt, price_history=None, monday_return=None):
    """Generate ALL new astrology signals for backtesting.
    Returns: dict of signal_name -> value (bool, float, or tuple)
    """
    signals = {}

    # --- Matrixology time-based ---
    signals["6AM_UTC_BOTTOM"] = is_6am_utc_window(dt)
    signals["MIDNIGHT_REVERSAL"] = is_midnight_utc_reversal(dt)
    block, block_name = get_6h_cycle_phase(dt)
    signals["CYCLE_PHASE"] = block
    signals["CYCLE_PHASE_NAME"] = block_name
    signals["PUMP_DATE_14_15_16"] = is_pump_date(dt)
    is_tue, tue_after_pump = is_tuesday_after_monday_pump(dt, monday_return)
    signals["IS_TUESDAY"] = is_tue
    signals["TUESDAY_AFTER_PUMP"] = tue_after_pump

    if price_history:
        signals["5DAY_PEAK_CRASH_LAG"] = is_5day_peak_crash_lag(dt, price_history)
    else:
        signals["5DAY_PEAK_CRASH_LAG"] = False

    # --- Classical astrology ---
    voc, voc_hours = is_voc_moon_classical(dt)
    signals["VOC_MOON"] = voc
    signals["VOC_MOON_HOURS"] = voc_hours

    node_signal, node_orb = get_moon_node_signal(dt)
    signals["MOON_NORTH_NODE"] = (node_signal == 'north_conjunction')
    signals["MOON_SOUTH_NODE"] = (node_signal == 'south_conjunction')
    signals["MOON_NODE_ORB"] = node_orb

    psi, psi_details = get_planetary_strength_index(dt)
    signals["PLANETARY_STRENGTH_INDEX"] = psi

    sat_station, sat_type = is_saturn_station(dt)
    signals["SATURN_STATION"] = sat_station
    signals["SATURN_STATION_TYPE"] = sat_type

    signals["MARS_RETROGRADE"] = is_mars_retrograde_signal(dt)
    signals["VENUS_RETROGRADE"] = is_venus_retrograde_signal(dt)

    near_ingress, ingress_sign, ingress_score = get_ingress_chart_bias(dt)
    signals["CARDINAL_INGRESS"] = near_ingress
    signals["INGRESS_SIGN"] = ingress_sign
    signals["INGRESS_SCORE"] = ingress_score

    zr_sign, zr_peak = get_zodiacal_releasing_sign(dt)
    signals["ZR_SIGN"] = zr_sign
    signals["ZR_PEAK_PERIOD"] = zr_peak

    ecl_ext, ecl_ext_type, ecl_ext_days = is_eclipse_window_extended(dt, window_days=30)
    signals["ECLIPSE_30D_WINDOW"] = ecl_ext
    signals["ECLIPSE_30D_TYPE"] = ecl_ext_type
    signals["ECLIPSE_30D_DAYS"] = ecl_ext_days

    return signals


# ===========================================================================
# PLANETARY EXPANSION — SPEED, COMBUSTION, DIGNITY, SYNODIC, DECAN, STARS
# ===========================================================================

# ---------------------------------------------------------------------------
# 1. Planetary Speed (degrees/day) — heliocentric daily motion
# ---------------------------------------------------------------------------

def get_planetary_speeds(dt):
    """Compute geocentric daily motion for 5 visible planets.

    Uses geocentric ecliptic longitude (via ephem.Ecliptic) so that
    stations and retrogrades are properly detected. Near-zero geocentric
    speed = station (planet appearing to stop before changing direction).
    Stations are moments of maximum planetary power in traditional astrology.

    Station thresholds are planet-specific (fraction of mean daily motion):
        Mercury: <0.3, Venus: <0.15, Mars: <0.05,
        Jupiter: <0.02, Saturn: <0.01 deg/day.

    Returns dict:
        {name}_speed      : float degrees/day (negative = retrograde)
        {name}_is_stationary : 1.0 if |speed| < threshold, else 0.0
    """
    # Planet-specific station thresholds (approx 20% of mean geocentric speed)
    _STATION_THRESH = {
        'mercury': 0.30,  # mean ~1.4 deg/day
        'venus': 0.15,    # mean ~1.0 deg/day
        'mars': 0.05,     # mean ~0.52 deg/day
        'jupiter': 0.02,  # mean ~0.08 deg/day
        'saturn': 0.01,   # mean ~0.03 deg/day
    }
    d1 = _to_ephem_date(dt)
    d0 = _to_ephem_date(dt - timedelta(days=1))
    speeds = {}
    for name, body_cls in [('mercury', ephem.Mercury), ('venus', ephem.Venus),
                            ('mars', ephem.Mars), ('jupiter', ephem.Jupiter),
                            ('saturn', ephem.Saturn)]:
        b1 = body_cls()
        b1.compute(d1)
        b0 = body_cls()
        b0.compute(d0)
        # Geocentric ecliptic longitude — proper for station detection
        lon1 = math.degrees(float(ephem.Ecliptic(b1).lon)) % 360
        lon0 = math.degrees(float(ephem.Ecliptic(b0).lon)) % 360
        speed = lon1 - lon0
        if speed > 180:
            speed -= 360
        if speed < -180:
            speed += 360
        speeds[f'{name}_speed'] = float(speed)
        thresh = _STATION_THRESH[name]
        speeds[f'{name}_is_stationary'] = 1.0 if abs(speed) < thresh else 0.0
    return speeds


# ---------------------------------------------------------------------------
# 2. Combustion and Cazimi
# ---------------------------------------------------------------------------

def get_combustion_cazimi(dt):
    """Check if planets are combust (<8.5 deg from Sun) or cazimi (<0.28 deg).

    Combust = planet hidden by Sun's light, weakened.
    Cazimi  = planet in the heart of the Sun, maximally empowered.

    Returns dict:
        {name}_combust  : 1.0 / 0.0
        {name}_cazimi   : 1.0 / 0.0
        n_combust       : count of combust planets
        n_cazimi        : count of cazimi planets
    """
    sun = ephem.Sun()
    sun.compute(_to_ephem_date(dt))
    sun_lon = math.degrees(float(ephem.Ecliptic(sun).lon)) % 360

    result = {'n_combust': 0.0, 'n_cazimi': 0.0}
    for name, body_cls in [('mercury', ephem.Mercury), ('venus', ephem.Venus),
                            ('mars', ephem.Mars), ('jupiter', ephem.Jupiter),
                            ('saturn', ephem.Saturn)]:
        body = body_cls()
        body.compute(_to_ephem_date(dt))
        p_lon = math.degrees(float(ephem.Ecliptic(body).lon)) % 360
        sep = abs(p_lon - sun_lon) % 360
        if sep > 180:
            sep = 360.0 - sep
        is_cazimi = (sep <= 0.28)
        is_combust = (sep < 8.5) and (not is_cazimi)
        result[f'{name}_combust'] = 1.0 if is_combust else 0.0
        result[f'{name}_cazimi'] = 1.0 if is_cazimi else 0.0
        result['n_combust'] += (1.0 if is_combust else 0.0)
        result['n_cazimi'] += (1.0 if is_cazimi else 0.0)
    return result


# ---------------------------------------------------------------------------
# 3. Essential Dignity Score (per Lilly — full table)
# ---------------------------------------------------------------------------

def get_essential_dignity_scores(dt):
    """Score EACH planet's essential dignity (Lilly system) and return
    individual scores plus total.

    Scoring:  domicile +5, exaltation +4, triplicity +3,
              face +1, detriment -5, fall -4.

    Re-uses existing DOMICILE / EXALTATION / DETRIMENT / FALL / TRIPLICITY_DAY
    tables and get_planet_zodiac() already defined above.

    Returns dict:
        {name}_dignity   : float (individual score)
        total_dignity    : float (sum of all)
    """
    result = {}
    total = 0.0
    for pname in ['Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn',
                  'Uranus', 'Neptune', 'Pluto']:
        s = float(get_planet_dignity_score(pname, dt))
        result[f'{pname.lower()}_dignity'] = s
        total += s
    # Also score Sun via sign check (Sun uses same tables)
    try:
        sun_sign = get_sun_zodiac(dt)
        sun_score = 0.0
        if sun_sign in DOMICILE.get("Sun", []):
            sun_score += 5.0
        if EXALTATION.get("Sun") == sun_sign:
            sun_score += 4.0
        if sun_sign in DETRIMENT.get("Sun", []):
            sun_score -= 5.0
        if FALL.get("Sun") == sun_sign:
            sun_score -= 4.0
        result['sun_dignity'] = sun_score
        total += sun_score
    except (ephem.AlwaysUpError, ephem.NeverUpError, ephem.CircumpolarError) as e:
        logging.debug(f"Circumpolar condition: {e}")
        result['sun_dignity'] = np.nan
    except Exception as e:
        logging.warning(f"Astrology calc failed: {e}")
        result['sun_dignity'] = np.nan
    result['total_dignity'] = total
    return result


# ---------------------------------------------------------------------------
# 4. Synodic Cycle Phases (sin/cos encoding)
# ---------------------------------------------------------------------------

# Mean synodic periods in days (planet-Sun as seen from Earth)
_SYNODIC_PERIODS = {
    'mercury_sun': 115.88,
    'venus_sun': 583.9,
    'mars_sun': 779.9,
    'jupiter_saturn': 7253.0,
}

def get_synodic_phases(dt):
    """Encode position within major synodic cycles as sin/cos.

    For planet-Sun pairs, the angular separation tells us where we are
    in the synodic cycle (conjunction=0, opposition=180).
    For Jupiter-Saturn, their mutual separation encodes the ~20-year cycle.

    Returns dict:
        synodic_{pair}_phase_sin : float (-1..1)
        synodic_{pair}_phase_cos : float (-1..1)
    """
    result = {}
    sun = ephem.Sun()
    sun.compute(_to_ephem_date(dt))
    sun_lon = math.degrees(float(ephem.Ecliptic(sun).lon)) % 360

    planet_lons = {}
    for name, body_cls in [('mercury', ephem.Mercury), ('venus', ephem.Venus),
                            ('mars', ephem.Mars), ('jupiter', ephem.Jupiter),
                            ('saturn', ephem.Saturn)]:
        body = body_cls()
        body.compute(_to_ephem_date(dt))
        planet_lons[name] = math.degrees(float(ephem.Ecliptic(body).lon)) % 360

    # Planet-Sun pairs: phase = angular separation / 360
    for pair, period in _SYNODIC_PERIODS.items():
        parts = pair.split('_')
        if parts[1] == 'sun':
            # Angular separation from Sun
            p_lon = planet_lons.get(parts[0])
            if p_lon is None:
                continue
            sep = (p_lon - sun_lon) % 360.0  # 0..360
        else:
            # Planet-planet pair (jupiter_saturn)
            p1_lon = planet_lons.get(parts[0])
            p2_lon = planet_lons.get(parts[1])
            if p1_lon is None or p2_lon is None:
                continue
            sep = (p1_lon - p2_lon) % 360.0
        phase = sep / 360.0  # 0..1
        result[f'synodic_{pair}_phase_sin'] = math.sin(2.0 * math.pi * phase)
        result[f'synodic_{pair}_phase_cos'] = math.cos(2.0 * math.pi * phase)
    return result


# ---------------------------------------------------------------------------
# 5. Decan Index (Sun's 10-degree segment of ecliptic)
# ---------------------------------------------------------------------------

def get_decan(dt):
    """Sun's decan (0-35). Each decan = 10 degrees of ecliptic.

    Decans are sub-divisions of zodiac signs used in Hellenistic and
    Egyptian astrology, each with its own planetary ruler.

    Returns dict:
        sun_decan : int 0-35
    """
    sun_lon = get_sun_ecliptic_lon(dt)
    return {'sun_decan': float(int(sun_lon / 10.0) % 36)}


# ---------------------------------------------------------------------------
# 6. Fixed Star Conjunctions (7 Behenian Stars)
# ---------------------------------------------------------------------------

# Behenian star ecliptic longitudes (epoch ~2000). Precession ~1 deg / 72 yr.
# These stars are considered the most powerful fixed stars in medieval astrology.
_BEHENIAN_EPOCH_2000 = {
    'algol':     56.0,
    'aldebaran': 69.0,
    'sirius':    104.0,
    'regulus':   150.0,
    'spica':     204.0,
    'antares':   249.0,
    'fomalhaut': 334.0,
}


def _behenian_lon(star_name, dt):
    """Return precession-corrected ecliptic longitude of a Behenian star."""
    base = _BEHENIAN_EPOCH_2000[star_name]
    years_since_2000 = (dt - datetime(2000, 1, 1)).days / 365.25
    return (base + years_since_2000 / 72.0) % 360.0


def count_star_conjunctions(dt, orb=2.0):
    """Count planets conjunct (within orb degrees) Behenian fixed stars.

    Returns dict:
        n_star_conjunctions       : total count
        behenian_{star}_activated : 1.0 if any planet is conjunct this star
    """
    # Gather planet longitudes
    planet_lons = []
    for pname in ['Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn']:
        planet_lons.append(get_planet_position(pname, dt))
    # Also Sun and Moon
    planet_lons.append(get_sun_ecliptic_lon(dt))
    planet_lons.append(get_moon_ecliptic_lon(dt))

    result = {'n_star_conjunctions': 0.0}
    for star_name in _BEHENIAN_EPOCH_2000:
        star_lon = _behenian_lon(star_name, dt)
        activated = 0.0
        for p_lon in planet_lons:
            sep = abs(p_lon - star_lon) % 360.0
            if sep > 180.0:
                sep = 360.0 - sep
            if sep <= orb:
                activated = 1.0
                break
        result[f'behenian_{star_name}_activated'] = activated
        result['n_star_conjunctions'] += activated
    return result


# ---------------------------------------------------------------------------
# 7. Batch compute ALL planetary expansion features for a single datetime
# ---------------------------------------------------------------------------

def get_planetary_expansion(dt):
    """Compute all planetary expansion features for a single datetime.

    Returns a flat dict with ~30 float values.
    """
    out = {}
    out.update(get_planetary_speeds(dt))
    out.update(get_combustion_cazimi(dt))
    out.update(get_essential_dignity_scores(dt))
    out.update(get_synodic_phases(dt))
    out.update(get_decan(dt))
    out.update(count_star_conjunctions(dt))
    return out


# ===========================================================================
# MOON DISTANCE (Apogee / Perigee)
# ===========================================================================

def get_moon_distance(dt):
    """Get Earth-Moon distance. Perigee ~356500km, Apogee ~406700km.
    Returns dict with distance in km, normalized 0-1 (perigee-apogee),
    and binary flags for perigee/apogee windows.
    """
    moon = ephem.Moon()
    moon.compute(_to_ephem_date(dt))
    # earth_distance is in AU, convert to km
    dist_km = float(moon.earth_distance) * 149597870.7
    return {
        'moon_distance_km': dist_km,
        'moon_distance_norm': (dist_km - 356500.0) / (406700.0 - 356500.0),  # 0=perigee, 1=apogee
        'moon_at_perigee': 1.0 if dist_km < 362000.0 else 0.0,
        'moon_at_apogee': 1.0 if dist_km > 404000.0 else 0.0,
    }


# ===========================================================================
# LUNAR NODE SIGN
# ===========================================================================

def get_lunar_node_sign(dt):
    """Get North Node zodiac sign index (changes every ~18 months).
    Returns sign index 0-11 and flag for near sign change.
    """
    T = (ephem.Date(dt) - ephem.Date('2000/1/1.5')) / 36525.0
    north_node_lon = (125.0445 - 1934.136 * T) % 360
    sign_idx = int(north_node_lon / 30) % 12
    degree_in_sign = north_node_lon % 30
    near_sign_change = 1.0 if (degree_in_sign < 2.0 or degree_in_sign > 28.0) else 0.0
    return {
        'lunar_node_sign_idx': float(sign_idx),
        'lunar_node_near_change': near_sign_change,
    }


# ===========================================================================
# SOLUNAR PERIODS
# ===========================================================================

def get_solunar_period(dt):
    """Classic solunar theory: major periods (moon transit/antitransit),
    minor (rise/set). Approximated from moon hour angle.
    Uses NYC as reference longitude for consistency.
    """
    obs = ephem.Observer()
    obs.lat, obs.lon = '40.7128', '-74.0060'  # NYC as reference
    obs.date = ephem.Date(dt)
    moon = ephem.Moon()
    moon.compute(obs)

    # Hour angle approximation (0-360)
    ha = float(obs.sidereal_time() - moon.ra) * 180.0 / math.pi
    ha = ha % 360.0

    # Major period: moon near meridian (0 or 180) = overhead or underfoot
    major_proximity = min(ha % 180.0, 180.0 - ha % 180.0)
    is_major = 1.0 if major_proximity < 15.0 else 0.0  # within ~1 hour

    # Minor period: moon near horizon (90 or 270) = rising or setting
    minor_proximity = min(abs(ha - 90.0), abs(ha - 270.0))
    is_minor = 1.0 if minor_proximity < 15.0 else 0.0

    return {
        'solunar_major': is_major,
        'solunar_minor': is_minor,
        'solunar_ha': ha / 360.0,  # normalized hour angle
    }


# ===========================================================================
# TIDAL FORCE PROXY
# ===========================================================================

def get_tidal_force(dt):
    """Combined Sun+Moon tidal force proxy.
    Tidal force proportional to M/d^3. Returns normalized value.
    """
    moon = ephem.Moon()
    moon.compute(ephem.Date(dt))
    sun = ephem.Sun()
    sun.compute(ephem.Date(dt))

    # Tidal force proportional to M/d^3
    # Moon: mass=7.342e22kg, Sun: mass=1.989e30kg
    d_moon = float(moon.earth_distance) * 149597870700.0  # AU to meters
    d_sun = float(sun.earth_distance) * 149597870700.0

    f_moon = 7.342e22 / (d_moon ** 3)
    f_sun = 1.989e30 / (d_sun ** 3)

    # Normalize so typical values are ~1.0-3.0 range
    f_total = (f_moon + f_sun) / 1e-3
    return {'tidal_force': f_total}


# ===========================================================================
# QUICK TEST
# ===========================================================================
if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print("=" * 80)
    print("  ASTROLOGY ENGINE TEST")
    print("=" * 80)

    # Test current date
    test_dates = [
        datetime(2022, 1, 14),   # Mercury retrograde started
        datetime(2022, 6, 18),   # BTC bottom
        datetime(2020, 12, 21),  # Great conjunction
        datetime(2024, 4, 8),    # Solar eclipse
        datetime(2025, 3, 16),   # Recent
        datetime(2026, 3, 16),   # Current
    ]

    for dt in test_dates:
        print(f"\n  --- {dt.strftime('%Y-%m-%d')} ---")

        # Mercury retrograde
        is_retro, days = is_mercury_retrograde(dt)
        print(f"  Mercury retrograde: {'YES (day ' + str(days) + ')' if is_retro else 'no'}")

        # Sun zodiac
        print(f"  Sun in: {get_sun_zodiac(dt)}")

        # Planet retrogrades
        retros = [p for p in PLANETS if is_planet_retrograde(p, dt)]
        print(f"  Retrograde planets: {', '.join(retros) if retros else 'none'}")

        # Hard aspects
        n_hard, hard = count_hard_aspects(dt)
        print(f"  Hard aspects: {n_hard}")
        for p1, p2, asp, ex in hard[:3]:
            print(f"    {p1} {asp} {p2} ({ex:.1f}deg)")

        # BTC transits
        score, detail = get_btc_transit_score(dt)
        print(f"  BTC transit score: {score:+.2f} {'BULLISH' if score > 0 else 'BEARISH'}")
        if detail:
            print(f"    {detail}")

        # Eclipse
        in_ecl, e_type, e_days = is_eclipse_window(dt)
        if in_ecl:
            print(f"  ECLIPSE: {e_type} {e_days}d away")

        # All signals
        sigs = get_astrology_signals(dt)
        if sigs:
            print(f"  SIGNALS:")
            for name, direction, strength, det in sigs:
                print(f"    {name}: {direction} str={strength} | {det}")
