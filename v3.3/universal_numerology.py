#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
universal_numerology.py — Universal Numerology Engine
======================================================
Single source of truth for ALL numerology calculations.
Takes any number or date, returns all computed values.

Usage:
  from universal_numerology import numerology, date_numerology, sequence_detect
  result = numerology(73954)           # any number
  result = date_numerology(datetime)   # any date
"""

import os
from datetime import datetime, timedelta


# ============================================================
# CONSTANTS
# ============================================================

MASTER_NUMBERS = {11, 22, 33}

CAUTION_NUMBERS = {
    93: "destruction/thelema", 39: "mirror of 93",
    43: "civil unrest", 48: "Illuminati/hoax",
    223: "Skull & Bones", 322: "Skull & Bones reverse",
    17: "Freemasonry/kill", 71: "mirror of 17",
    113: "dump signal/dishonest", 311: "mirror of 113",
    11: "emotional/chaotic", 19: "surrender",
    13: "rebellion/death", 47: "authority/government",
    22: "master builder (can crash)", 33: "master teacher",
    44: "kill/execution", 66: "woman/Venus",
    77: "Christ/power", 88: "Trump/poison",
    99: "completion/ending", 666: "beast/Mark",
    911: "emergency/Thelema", 322: "S&B",
}

PUMP_NUMBERS = {
    27: "pump signal", 72: "pump reverse",
    127: "strong pump", 721: "pump mirror",
    37: "UP energy", 73: "UP reverse/wisdom",
}

BTC_ENERGY_NUMBERS = {
    213: "BTC energy", 231: "BTC mirror", 312: "BTC cycle",
    321: "BTC countdown", 132: "BTC cycle", 123: "ascending",
    68: "Bitcoin ordinal gematria",
}

# Sequences to detect in any number
KEY_SEQUENCES = [113, 322, 93, 213, 666, 777, 911, 369, 147, 258, 33, 11, 22]


# ============================================================
# LO SHU MAGIC SQUARE
# ============================================================

# Lo Shu grid positions (digit -> (row, col)):
LO_SHU = {
    4:(0,0), 9:(0,1), 2:(0,2),
    3:(1,0), 5:(1,1), 7:(1,2),
    8:(2,0), 1:(2,1), 6:(2,2),
}
# Lo Shu lines (each sums to 15):
LO_SHU_LINES = [
    frozenset({4,9,2}), frozenset({3,5,7}), frozenset({8,1,6}),  # rows
    frozenset({4,3,8}), frozenset({9,5,1}), frozenset({2,7,6}),  # cols
    frozenset({4,5,6}), frozenset({2,5,8}),                       # diagonals
]

LO_SHU_ROW = {4:0, 9:0, 2:0, 3:1, 5:1, 7:1, 8:2, 1:2, 6:2}
LO_SHU_COL = {4:0, 9:1, 2:2, 3:0, 5:1, 7:2, 8:0, 1:1, 6:2}


def loshu_position(dr):
    """Map digital root (1-9) to Lo Shu grid position (row, col)."""
    return LO_SHU.get(dr, (None, None))


def loshu_grid_type(dr):
    """Center(5), corner(2,4,6,8), or edge(1,3,7,9)."""
    if dr == 5: return 'center'
    if dr in (2,4,6,8): return 'corner'
    if dr in (1,3,7,9): return 'edge'
    return None


def loshu_line_completion(dr_seq):
    """Check if 3 consecutive DRs form a Lo Shu line (row/col/diagonal)."""
    s = frozenset(dr_seq[-3:]) if len(dr_seq) >= 3 else frozenset()
    return any(s == line for line in LO_SHU_LINES)


# ============================================================
# PYTHAGOREAN CHALLENGE NUMBERS
# ============================================================

def pythagorean_challenges(dt):
    """Compute Pythagorean challenge numbers from date components.

    Challenge numbers reveal obstacles/lessons encoded in a date.
    Each challenge is the absolute difference between reduced date components.
    A result of 0 maps to 9 (completion energy).
    """
    m_dr = digital_root(dt.month)
    d_dr = digital_root(dt.day)
    y_dr = digital_root(sum(int(c) for c in str(dt.year)))

    challenge_1 = abs(m_dr - d_dr)
    challenge_2 = abs(d_dr - y_dr)
    challenge_3 = abs(challenge_1 - challenge_2)

    return {
        'challenge_1': challenge_1 or 9,  # 0 maps to 9
        'challenge_2': challenge_2 or 9,
        'challenge_3': challenge_3 or 9,
    }


# ============================================================
# CORE FUNCTIONS
# ============================================================

def digital_root(n):
    """Reduce any number to 1-9."""
    n = abs(int(n))
    if n == 0:
        return 0
    return 1 + (n - 1) % 9


def reduce_keep_master(n):
    """Reduce number but keep master numbers 11, 22, 33."""
    n = abs(int(n))
    while n > 9 and n not in MASTER_NUMBERS:
        n = sum(int(d) for d in str(n))
    return n


def is_master(n):
    """Check if number reduces to a master number."""
    return reduce_keep_master(abs(int(n))) in MASTER_NUMBERS


def master_value(n):
    """Returns master number (11/22/33) or None."""
    r = reduce_keep_master(abs(int(n)))
    return r if r in MASTER_NUMBERS else None


def sequence_detect(n, patterns=None):
    """
    Check if a number contains key sequences.
    Returns dict of {pattern: True/False}.
    """
    if patterns is None:
        patterns = KEY_SEQUENCES
    s = str(abs(int(n)))
    return {p: str(p) in s for p in patterns}


def is_caution(n):
    """Check if number matches any caution/bearish number."""
    n = abs(int(n))
    return n in CAUTION_NUMBERS


def is_pump(n):
    """Check if number matches any pump/bullish number."""
    n = abs(int(n))
    return n in PUMP_NUMBERS


def is_btc_energy(n):
    """Check if number matches BTC energy numbers."""
    n = abs(int(n))
    return n in BTC_ENERGY_NUMBERS


def numerology(n):
    """
    Returns ALL numerology values for any number.

    >>> numerology(73954)
    {'value': 73954, 'dr': 1, 'reduced': 1, 'is_master': False, 'master': None,
     'is_caution': False, 'is_pump': False, 'is_btc_energy': False,
     'contains_113': True, 'contains_322': False, ...}
    """
    n_abs = abs(int(n))
    dr = digital_root(n_abs)
    seqs = sequence_detect(n_abs)

    return {
        'value': n_abs,
        'dr': dr,
        'reduced': reduce_keep_master(n_abs),
        'is_master': is_master(n_abs),
        'master': master_value(n_abs),
        'is_caution': is_caution(n_abs) or is_caution(dr),
        'is_pump': is_pump(n_abs) or is_pump(dr),
        'is_btc_energy': is_btc_energy(n_abs),
        # Sequence detection
        'contains_113': seqs.get(113, False),
        'contains_322': seqs.get(322, False),
        'contains_93': seqs.get(93, False),
        'contains_213': seqs.get(213, False),
        'contains_666': seqs.get(666, False),
        'contains_777': seqs.get(777, False),
        'contains_911': seqs.get(911, False),
        'contains_369': seqs.get(369, False),
        'contains_33': seqs.get(33, False),
        'contains_11': seqs.get(11, False),
        'contains_22': seqs.get(22, False),
        # Mirror check
        'mirror_value': int(str(n_abs)[::-1]) if n_abs > 9 else n_abs,
        'is_palindrome': str(n_abs) == str(n_abs)[::-1],
    }


# ============================================================
# DATE NUMEROLOGY
# ============================================================

def date_numerology(dt):
    """
    Returns ALL numerological values for a date.

    >>> date_numerology(datetime(2026, 3, 18))
    {'day_of_year': 77, 'days_remaining': 288, 'week_number': 12, ...}
    """
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)

    day_of_year = dt.timetuple().tm_yday
    is_leap = dt.year % 4 == 0 and (dt.year % 100 != 0 or dt.year % 400 == 0)
    total_days = 366 if is_leap else 365
    days_remaining = total_days - day_of_year
    week_number = dt.isocalendar()[1]
    pct_of_year = round(day_of_year / total_days * 100, 1)

    # Date sum reduction
    date_sum = dt.month + dt.day + sum(int(d) for d in str(dt.year))
    date_reduction = reduce_keep_master(date_sum)

    # Month/day as number
    month_day = dt.month * 100 + dt.day  # e.g., 318 for March 18
    day_month = dt.day * 100 + dt.month  # e.g., 183 for 18th of March

    # Day of week (0=Mon, 6=Sun)
    dow = dt.weekday()
    dow_name = dt.strftime("%A")

    # Digital roots of everything
    dr_doy = digital_root(day_of_year)
    dr_remaining = digital_root(days_remaining)
    dr_week = digital_root(week_number)
    dr_month_day = digital_root(month_day)

    # Caution/pump checks on all date components
    date_components = [day_of_year, days_remaining, date_sum, date_reduction,
                       month_day, day_month, week_number]
    any_caution = any(c in CAUTION_NUMBERS for c in date_components)
    any_pump = any(c in PUMP_NUMBERS for c in date_components)

    # Mirror check on date components
    mirrors_caution = False
    for num in date_components:
        if num > 9:
            mirrored = int(str(num)[::-1])
            if mirrored in CAUTION_NUMBERS:
                mirrors_caution = True
                break

    # Special dates
    is_friday_13 = dow == 4 and dt.day == 13
    is_day_13 = dt.day == 13
    is_day_21 = dt.day == 21
    is_day_27 = dt.day == 27
    is_repeating = dt.month == dt.day  # 1/1, 2/2, 3/3, etc.
    is_palindrome = str(month_day) == str(month_day)[::-1]

    # BTC energy dates (month, day) pairs
    btc_energy_dates = {
        (1, 3), (2, 13), (3, 12), (3, 21), (2, 1),
        (12, 3), (1, 23), (3, 1),
    }
    is_btc_213 = (dt.month, dt.day) in btc_energy_dates

    return {
        'day_of_year': day_of_year,
        'days_remaining': days_remaining,
        'week_number': week_number,
        'pct_of_year': pct_of_year,
        'date_sum': date_sum,
        'date_reduction': date_reduction,
        'month_day': month_day,
        'day_month': day_month,
        'day_of_week': dow,
        'day_of_week_name': dow_name,
        # Digital roots
        'dr_doy': dr_doy,
        'dr_remaining': dr_remaining,
        'dr_week': dr_week,
        'dr_month_day': dr_month_day,
        'dr_date': digital_root(date_sum),
        # Flags
        'is_caution_date': any_caution,
        'is_pump_date': any_pump,
        'mirrors_caution': mirrors_caution,
        'is_friday_13': is_friday_13,
        'is_day_13': is_day_13,
        'is_day_21': is_day_21,
        'is_day_27': is_day_27,
        'is_repeating_date': is_repeating,
        'is_palindrome_date': is_palindrome,
        'is_btc_213_date': is_btc_213,
        'is_master_date': is_master(date_sum),
    }


def numerology_flat(n, prefix=''):
    """
    Returns flat dict suitable for ML features.
    If prefix='price_', returns {'price_dr': 5, 'price_is_master': 0, ...}
    """
    r = numerology(n)
    p = f"{prefix}_" if prefix else ""
    return {
        f'{p}dr': r['dr'],
        f'{p}is_master': int(r['is_master']),
        f'{p}is_caution': int(r['is_caution']),
        f'{p}is_pump': int(r['is_pump']),
        f'{p}contains_113': int(r['contains_113']),
        f'{p}contains_322': int(r['contains_322']),
        f'{p}contains_93': int(r['contains_93']),
        f'{p}contains_213': int(r['contains_213']),
        f'{p}contains_666': int(r['contains_666']),
        f'{p}contains_777': int(r['contains_777']),
    }


def date_numerology_flat(dt, prefix='date'):
    """Returns flat dict of date numerology for ML features."""
    r = date_numerology(dt)
    p = f"{prefix}_" if prefix else ""
    return {
        f'{p}dr': r['dr_date'],
        f'{p}doy': r['day_of_year'],
        f'{p}remaining': r['days_remaining'],
        f'{p}is_caution': int(r['is_caution_date']),
        f'{p}is_pump': int(r['is_pump_date']),
        f'{p}is_friday_13': int(r['is_friday_13']),
        f'{p}is_day_13': int(r['is_day_13']),
        f'{p}is_day_21': int(r['is_day_21']),
        f'{p}is_day_27': int(r['is_day_27']),
        f'{p}is_btc_213': int(r['is_btc_213_date']),
        f'{p}is_master': int(r['is_master_date']),
    }


# ============================================================
# GPU / VECTORIZED NUMEROLOGY (zero .apply())
# ============================================================

import numpy as np

def digital_root_vec(arr):
    """Vectorized digital root on numpy or cupy array. 0→0, else 1+((n-1)%9)."""
    if os.environ.get('V2_SKIP_GPU') != '1':
        try:
            import cupy as cp
            if isinstance(arr, cp.ndarray):
                x = cp.abs(arr).astype(cp.int64)
                return cp.where(x == 0, cp.int32(0), (1 + (x - 1) % 9).astype(cp.int32))
        except ImportError:
            if os.environ.get('ALLOW_CPU', '0') != '1':
                raise RuntimeError("GPU REQUIRED: CuPy unavailable in digital_root_vec. Set ALLOW_CPU=1 for CPU mode.")
    x = np.abs(np.asarray(arr, dtype=np.int64))
    return np.where(x == 0, 0, 1 + (x - 1) % 9).astype(np.int32)


def is_in_set_vec(arr, target_set):
    """Vectorized set membership check. Returns int32 0/1 array."""
    if os.environ.get('V2_SKIP_GPU') != '1':
        try:
            import cupy as cp
            if isinstance(arr, cp.ndarray):
                targets_gpu = cp.asarray(sorted(target_set), dtype=cp.int64)
                return cp.isin(arr.astype(cp.int64), targets_gpu).astype(cp.int32)
        except ImportError:
            if os.environ.get('ALLOW_CPU', '0') != '1':
                raise RuntimeError("GPU REQUIRED: CuPy unavailable in is_in_set_vec. Set ALLOW_CPU=1 for CPU mode.")
    targets = np.array(sorted(target_set), dtype=np.int64)
    return np.isin(np.asarray(arr, dtype=np.int64), targets).astype(np.int32)


def price_contains_pattern_vec(prices, pattern_str):
    """Vectorized check if price (as int string) contains a digit pattern."""
    if os.environ.get('V2_SKIP_GPU') != '1':
        try:
            import cudf
            if hasattr(prices, 'str'):
                # cuDF or pandas Series with .str accessor
                return prices.astype(str).str.contains(pattern_str, regex=False).astype('int32')
        except ImportError:
            if os.environ.get('ALLOW_CPU', '0') != '1':
                raise RuntimeError("GPU REQUIRED: cuDF unavailable in price_contains_pattern_vec. Set ALLOW_CPU=1 for CPU mode.")
    # Numpy fallback
    import pandas as pd
    s = pd.Series(prices).astype(str)
    return s.str.contains(pattern_str, regex=False).astype(np.int32).values


if __name__ == "__main__":
    # Quick test
    print("=== Number Numerology ===")
    for n in [73954, 71365, 113, 322, 33, 27]:
        r = numerology(n)
        print(f"  {n}: DR={r['dr']} master={r['master']} caution={r['is_caution']} pump={r['is_pump']} "
              f"113={r['contains_113']} 322={r['contains_322']}")

    print("\n=== Date Numerology ===")
    for d in [datetime(2026, 3, 18), datetime(2026, 3, 13), datetime(2026, 1, 3)]:
        r = date_numerology(d)
        print(f"  {d.strftime('%Y-%m-%d')}: DOY={r['day_of_year']} DR={r['dr_date']} "
              f"caution={r['is_caution_date']} pump={r['is_pump_date']} "
              f"fri13={r['is_friday_13']} btc213={r['is_btc_213_date']}")
