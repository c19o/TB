#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2_feature_layers.py — V2 Feature Computation Layers
=====================================================
New feature layers added on top of V1's existing ~1000-3000 features.
Imported by feature_library_v2.py.

ALL functions are VECTORIZED — no .apply() with Python UDFs.
No lookahead bias. Uses .shift(), .expanding(), rolling with min_periods.
Binary features are 0/1 integers, not booleans.
"""

import numpy as np
import pandas as pd
from datetime import datetime

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    def njit(*args, **kwargs):
        def _wrap(f): return f
        if args and callable(args[0]): return args[0]
        return _wrap

# ============================================================
# HELPERS
# ============================================================

def _digital_root_vec(arr):
    """Vectorized digital root: 1 + (n-1) % 9, with 0 -> 0."""
    arr = np.asarray(arr, dtype=np.int64)
    result = np.where(arr == 0, 0, 1 + (np.abs(arr) - 1) % 9)
    return result


def _safe_col(df, col):
    """Return column as float Series if it exists, else None."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors='coerce')
    return None


def _map_ephemeris_to_bars(ephemeris_df, df_index, col_name):
    """Map a daily ephemeris column to bar-level index via date alignment + ffill."""
    if ephemeris_df is None or col_name not in ephemeris_df.columns:
        return pd.Series(np.nan, index=df_index)
    s = pd.to_numeric(ephemeris_df[col_name], errors='coerce')
    s.index = s.index.normalize()
    if hasattr(s.index, 'tz') and s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    df_date_idx = df_index.normalize()
    if hasattr(df_date_idx, 'tz') and df_date_idx.tz is not None:
        df_date_idx = df_date_idx.tz_localize(None)
    mapped = s.reindex(df_date_idx)
    mapped.index = df_index
    return mapped.ffill()


@njit(cache=True)
def _consecutive_count(arr):
    """Count consecutive True values in a boolean array. Numba-compiled."""
    n = len(arr)
    result = np.zeros(n, dtype=np.int32)
    for i in range(n):
        if arr[i]:
            result[i] = result[i - 1] + 1 if i > 0 else 1
        else:
            result[i] = 0
    return result


# ============================================================
# 1. TIME OF DAY FEATURES
# ============================================================

def add_time_of_day_features(df, tf='1h'):
    """Add session bins and ICT kill zone flags for intraday timeframes."""
    if tf not in ('5m', '15m', '1h', '4h'):
        return df

    if not isinstance(df.index, pd.DatetimeIndex):
        return df

    hour_utc = df.index.hour

    new_cols = {}
    # Session bins (UTC)
    new_cols['session_asia'] = ((hour_utc >= 0) & (hour_utc < 8)).astype(np.int32)
    new_cols['session_london'] = ((hour_utc >= 7) & (hour_utc < 12)).astype(np.int32)
    new_cols['session_ny'] = ((hour_utc >= 12) & (hour_utc < 20)).astype(np.int32)
    new_cols['session_dead'] = ((hour_utc >= 20) | (hour_utc < 0)).astype(np.int32)

    # ICT kill zones (EST = UTC-5, so convert)
    # 2-5AM EST = 7-10 UTC
    # 7-10AM EST = 12-15 UTC
    # 1-3PM EST = 18-20 UTC
    new_cols['ict_killzone_asian'] = ((hour_utc >= 7) & (hour_utc < 10)).astype(np.int32)
    new_cols['ict_killzone_london'] = ((hour_utc >= 12) & (hour_utc < 15)).astype(np.int32)
    new_cols['ict_killzone_ny_pm'] = ((hour_utc >= 18) & (hour_utc < 20)).astype(np.int32)

    # Batch assign
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df


# ============================================================
# 2. RATE OF CHANGE FEATURES
# ============================================================

def add_rate_of_change_features(df):
    """Add rising_into/peaked/falling_into/bottomed for binarized TA indicators."""
    # Find all columns ending with _HIGH or _LOW (binarized indicators)
    high_cols = [c for c in df.columns if c.endswith('_HIGH')]
    low_cols = [c for c in df.columns if c.endswith('_LOW')]

    new_cols = {}
    for col in high_cols:
        base = col  # e.g. rsi_14_HIGH
        s = pd.to_numeric(df[col], errors='coerce')
        s_diff = s.diff()
        # rising_into: crossed above (was 0, now 1) within last 3 bars
        crossed_above = ((s == 1) & (s.shift(1) == 0)).astype(np.float64)
        new_cols[f'{base}_rising_into'] = crossed_above.rolling(3, min_periods=1).max().astype(np.int32)
        # peaked: was above (1), now declining (0)
        new_cols[f'{base}_peaked'] = ((s == 0) & (s.shift(1) == 1)).astype(np.int32)

    for col in low_cols:
        base = col  # e.g. rsi_14_LOW
        s = pd.to_numeric(df[col], errors='coerce')
        # falling_into: crossed below (was 0, now 1) within last 3 bars
        crossed_below = ((s == 1) & (s.shift(1) == 0)).astype(np.float64)
        new_cols[f'{base}_falling_into'] = crossed_below.rolling(3, min_periods=1).max().astype(np.int32)
        # bottomed: was below (1), now rising (0)
        new_cols[f'{base}_bottomed'] = ((s == 0) & (s.shift(1) == 1)).astype(np.int32)

    # Batch assign
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df


# ============================================================
# 3. SEQUENTIAL MEMORY FEATURES
# ============================================================

def add_sequential_memory_features(df):
    """Add shifted versions of key binary signals at 1, 3, 5, 7 bars."""
    new_cols = {}

    # RSI overbought/oversold memory
    if 'rsi_14_ob' in df.columns:
        for lag in [3, 7]:
            new_cols[f'was_overbought_{lag}'] = df['rsi_14_ob'].shift(lag).astype(np.float32)
    if 'rsi_14_os' in df.columns:
        for lag in [3, 7]:
            new_cols[f'was_oversold_{lag}'] = df['rsi_14_os'].shift(lag).astype(np.float32)

    # Volume spike memory
    if 'volume_spike' in df.columns:
        for lag in [1, 3]:
            new_cols[f'had_volume_spike_{lag}'] = df['volume_spike'].shift(lag).astype(np.float32)

    # Streak features (consecutive green/red candles)
    if 'close' in df.columns and 'open' in df.columns:
        c = df['close'].astype(float).values
        o = df['open'].astype(float).values
        is_green = (c > o).astype(np.int32)
        is_red = (c < o).astype(np.int32)
        new_cols['streak_green'] = _consecutive_count(is_green)
        new_cols['streak_red'] = _consecutive_count(is_red)

    # General: shift key binary signals
    key_binary_cols = [c for c in df.columns if c in (
        'golden_cross', 'death_cross', 'macd_cross_up', 'macd_cross_down',
        'bb_squeeze_20', 'sar_flip', 'supertrend_flip',
        'wyckoff_spring', 'wyckoff_upthrust', 'bull_engulfing', 'bear_engulfing',
    )]
    for col in key_binary_cols:
        for lag in [1, 3, 5, 7]:
            new_cols[f'{col}_lag{lag}'] = df[col].shift(lag).astype(np.float32)

    # Batch assign
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df


# ============================================================
# 4. PRICE NUMEROLOGY FEATURES
# ============================================================

def add_price_numerology_features(df):
    """Add digital root, angel numbers, round numbers, repeating digits, master numbers from price."""
    if 'close' not in df.columns:
        return df

    c = df['close'].astype(float).values
    c_int = np.where(np.isnan(c), 0, np.abs(c).astype(np.int64))

    new_cols = {}

    # Digital root of price
    new_cols['price_dr'] = np.where(np.isnan(c), np.nan, _digital_root_vec(c_int)).astype(np.float32)

    # Angel numbers (111, 222, ..., 999) in price string
    c_str = pd.Series(c_int, index=df.index).astype(str)
    angel_mask = np.zeros(len(df), dtype=np.int32)
    for triple in ['111', '222', '333', '444', '555', '666', '777', '888', '999']:
        angel_mask |= c_str.str.contains(triple, regex=False).values.astype(np.int32)
    new_cols['price_angel'] = np.where(np.isnan(c), np.nan, angel_mask).astype(np.float32)

    # Near round numbers (within 1%)
    round_levels = np.array([10000, 20000, 25000, 30000, 40000, 50000,
                             60000, 69000, 75000, 80000, 90000, 100000,
                             125000, 150000, 200000], dtype=np.float64)
    c_col = np.where(np.isnan(c), 0, c)
    # For each price, check distance to nearest round number
    # Vectorized: compute min percentage distance across all levels
    c_2d = c_col[:, np.newaxis]  # (N, 1)
    pct_dist = np.abs(c_2d - round_levels[np.newaxis, :]) / round_levels[np.newaxis, :]
    min_pct_dist = np.min(pct_dist, axis=1)
    new_cols['price_near_round'] = np.where(np.isnan(c), np.nan,
                                       (min_pct_dist < 0.01).astype(np.int32)).astype(np.float32)

    # Repeating digits detection — vectorized via pandas str ops
    # A price has repeating digits if any digit appears 3+ times
    _repeating_mask = np.zeros(len(c_str), dtype=np.int32)
    for d in '0123456789':
        _repeating_mask |= (c_str.str.count(d) >= 3).values.astype(np.int32)
    new_cols['price_repeating'] = np.where(np.isnan(c), np.nan,
                                      _repeating_mask).astype(np.float32)

    # Master numbers in price mod 1000
    c_mod1000 = c_int % 1000
    new_cols['price_master_11'] = np.where(np.isnan(c), np.nan,
                                      (c_mod1000 % 100 == 11).astype(np.int32)).astype(np.float32)
    new_cols['price_master_22'] = np.where(np.isnan(c), np.nan,
                                      (c_mod1000 % 100 == 22).astype(np.int32)).astype(np.float32)
    new_cols['price_master_33'] = np.where(np.isnan(c), np.nan,
                                      (c_mod1000 % 100 == 33).astype(np.int32)).astype(np.float32)

    # Batch assign
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df


# ============================================================
# 5. CALENDAR ANOMALY FEATURES
# ============================================================

# FOMC meeting dates (final day of each meeting, 2017-2026)
_FOMC_DATES = set()
for y, dates in {
    2017: [(2,1),(3,15),(5,3),(6,14),(7,26),(9,20),(11,1),(12,13)],
    2018: [(1,31),(3,21),(5,2),(6,13),(8,1),(9,26),(11,8),(12,19)],
    2019: [(1,30),(3,20),(5,1),(6,19),(7,31),(9,18),(10,30),(12,11)],
    2020: [(1,29),(3,3),(3,15),(4,29),(6,10),(7,29),(9,16),(11,5),(12,16)],
    2021: [(1,27),(3,17),(4,28),(6,16),(7,28),(9,22),(11,3),(12,15)],
    2022: [(1,26),(3,16),(5,4),(6,15),(7,27),(9,21),(11,2),(12,14)],
    2023: [(2,1),(3,22),(5,3),(6,14),(7,26),(9,20),(11,1),(12,13)],
    2024: [(1,31),(3,20),(5,1),(6,12),(7,31),(9,18),(11,7),(12,18)],
    2025: [(1,29),(3,19),(5,7),(6,18),(7,30),(9,17),(11,5),(12,17)],
    2026: [(1,28),(3,18),(5,6),(6,17),(7,29),(9,16),(10,28),(12,16)],
}.items():
    for m, d in dates:
        try:
            _FOMC_DATES.add(datetime(y, m, d).date())
        except Exception:
            pass

# BTC halving dates
_BTC_HALVINGS = [
    datetime(2012, 11, 28),
    datetime(2016, 7, 9),
    datetime(2020, 5, 11),
    datetime(2024, 4, 19),
]
_NEXT_HALVING = datetime(2028, 4, 15)  # estimated

# Chinese New Year dates 2017-2027
_CHINESE_NY = {
    2017: (1, 28), 2018: (2, 16), 2019: (2, 5), 2020: (1, 25),
    2021: (2, 12), 2022: (2, 1), 2023: (1, 22), 2024: (2, 10),
    2025: (1, 29), 2026: (2, 17), 2027: (2, 6),
}


def add_calendar_anomaly_features(df):
    """Add calendar event features: FOMC, OpEx, quad witching, CPI, halvings, etc."""
    if not isinstance(df.index, pd.DatetimeIndex):
        return df

    idx = df.index
    dates = idx.date
    years = idx.year
    months = idx.month
    days = idx.day
    dow = idx.dayofweek  # 0=Mon

    new_cols = {}

    # Week of month (1-5)
    new_cols['week_of_month'] = ((days - 1) // 7 + 1).astype(np.int32)

    # Month boundaries
    new_cols['is_month_end'] = idx.is_month_end.astype(np.int32)
    new_cols['is_month_start'] = idx.is_month_start.astype(np.int32)
    new_cols['is_quarter_end'] = idx.is_quarter_end.astype(np.int32)

    # FOMC days
    fomc_arr = np.isin(dates, np.array(list(_FOMC_DATES))).astype(np.int32)
    new_cols['is_fomc_day'] = fomc_arr

    # OpEx: 3rd Friday of each month
    # 3rd Friday = first Friday + 14 days. Day of week for 1st = dow.
    # Friday = 4. If 1st is dow=d, first Friday = 1 + (4 - d) % 7, third Friday = first + 14
    first_of_month_dow = pd.Timestamp(2000, 1, 1).dayofweek  # placeholder
    # Vectorized: compute 3rd Friday for each month
    first_day_dow = pd.to_datetime(
        pd.DataFrame({'year': years, 'month': months, 'day': np.ones(len(idx), dtype=int)})
    ).dt.dayofweek.values
    first_friday_day = 1 + (4 - first_day_dow) % 7
    third_friday_day = first_friday_day + 14
    is_opex = (days == third_friday_day).astype(np.int32)
    new_cols['is_opex'] = is_opex

    # Quadruple witching: 3rd Friday of Mar, Jun, Sep, Dec
    is_quad_month = np.isin(months, [3, 6, 9, 12])
    new_cols['is_quadruple_witching'] = (is_opex & is_quad_month.astype(np.int32)).astype(np.int32)

    # CPI day: treat 13th (or 12th/14th if 13th is weekend) as CPI
    cpi_day_est = np.where(
        (days == 13) & (dow < 5), 1,
        np.where((days == 12) & (dow == 4), 1,  # Friday before weekend 13th
                 np.where((days == 14) & (dow == 0), 1, 0))  # Monday after weekend 13th
    )
    new_cols['is_cpi_day'] = cpi_day_est.astype(np.int32)

    # BTC halving proximity
    halving_timestamps = np.array([h.timestamp() for h in _BTC_HALVINGS])
    next_halving_ts = _NEXT_HALVING.timestamp()
    bar_timestamps = idx.astype(np.int64) // 10**9  # unix seconds

    # Days since the MOST RECENT halving (largest timestamp that's still past)
    days_since = np.full(len(idx), np.nan)
    for h_ts in sorted(halving_timestamps):
        diff_days = (bar_timestamps - h_ts) / 86400.0
        mask = diff_days >= 0
        days_since = np.where(mask, diff_days, days_since)
    new_cols['btc_halving_days_since'] = days_since.astype(np.float32)

    # Days until next halving
    diff_next = (next_halving_ts - bar_timestamps) / 86400.0
    new_cols['btc_halving_days_until'] = np.maximum(diff_next, 0).astype(np.float32)

    # Chinese New Year window (+-7 days)
    cny_arr = np.zeros(len(idx), dtype=np.int32)
    for yr_val in np.unique(years):
        if yr_val in _CHINESE_NY:
            m, d = _CHINESE_NY[yr_val]
            try:
                cny_dt = datetime(yr_val, m, d)
                cny_ts = cny_dt.timestamp()
                diff_days = np.abs((bar_timestamps - cny_ts) / 86400.0)
                cny_arr = np.where((years == yr_val) & (diff_days <= 7), 1, cny_arr)
            except Exception:
                pass
    new_cols['is_chinese_new_year_window'] = cny_arr

    # Tax deadline window (April 1-15)
    new_cols['tax_deadline_window'] = ((months == 4) & (days >= 1) & (days <= 15)).astype(np.int32)

    # Batch assign
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df


# ============================================================
# 6. ENTROPY & FRACTAL FEATURES
# ============================================================

def add_entropy_fractal_features(df, window=20):
    """Add Shannon entropy, approximate entropy, and simplified Hurst exponent."""
    if 'close' not in df.columns:
        return df

    c = df['close'].astype(float)
    returns = c.pct_change()

    new_cols = {}

    # Better Shannon entropy: vectorized with sliding_window_view
    from numpy.lib.stride_tricks import sliding_window_view
    # fillna(0) required here: sliding_window_view + searchsorted can't handle NaN
    # Shannon entropy input must be dense; NaN returns treated as zero-return bars
    ret_vals = returns.fillna(0).values.astype(np.float64)
    n = len(ret_vals)
    entropy_arr = np.full(n, np.nan, dtype=np.float32)
    if n > window:
        win = sliding_window_view(ret_vals, window)  # shape (n-window+1, window)
        # Fixed bin edges from global range
        _rmin, _rmax = np.nanmin(ret_vals), np.nanmax(ret_vals)
        if _rmin < _rmax:
            _bins = np.linspace(_rmin, _rmax, 11)  # 10 bins
            idx = np.clip(np.searchsorted(_bins, win, side='right') - 1, 0, 9)
            # Batch histogram: count per bin across all windows
            counts = np.zeros((win.shape[0], 10), dtype=np.float64)
            for b in range(10):  # 10 iterations, not n iterations
                counts[:, b] = np.sum(idx == b, axis=1)
            probs = counts / window
            log_probs = np.zeros_like(probs)
            np.log2(probs, out=log_probs, where=probs > 0)
            ent = -np.sum(probs * log_probs, axis=1)
            entropy_arr[window:] = ent.astype(np.float32)
    new_cols['shannon_entropy'] = entropy_arr

    # Approximate entropy: rolling std of first differences of returns
    ret_diff = returns.diff()
    new_cols['approx_entropy'] = ret_diff.rolling(window, min_periods=window // 2).std().astype(np.float32)

    # Simplified Hurst exponent: vectorized R/S with sliding_window_view
    hurst_arr = np.full(n, np.nan, dtype=np.float32)
    if n > window:
        win = sliding_window_view(ret_vals, window)  # shape (n-window+1, window)
        mean_w = win.mean(axis=1, keepdims=True)
        dev = win - mean_w
        z = np.cumsum(dev, axis=1)
        R = z.max(axis=1) - z.min(axis=1)
        S = win.std(axis=1, ddof=1)
        valid = (S > 0) & (R > 0)
        rs = np.zeros(win.shape[0], dtype=np.float64)
        rs[valid] = np.log(R[valid] / S[valid]) / np.log(window)
        rs[~valid] = np.nan
        hurst_arr[window:] = rs.astype(np.float32)
    new_cols['hurst_exponent'] = hurst_arr

    # Binarized flags
    ent = pd.Series(entropy_arr, index=df.index)
    ent_q75 = ent.rolling(200, min_periods=50).quantile(0.75)
    ent_q25 = ent.rolling(200, min_periods=50).quantile(0.25)
    new_cols['entropy_HIGH'] = (ent > ent_q75).astype(np.int32)
    new_cols['entropy_LOW'] = (ent < ent_q25).astype(np.int32)

    hurst = pd.Series(hurst_arr, index=df.index)
    new_cols['hurst_trending'] = (hurst > 0.6).astype(np.int32)
    new_cols['hurst_meanreverting'] = (hurst < 0.4).astype(np.int32)

    # Batch assign
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df


# ============================================================
# 7. MOON POSITION FEATURES
# ============================================================

_ZODIAC_SIGNS = ['aries', 'taurus', 'gemini', 'cancer', 'leo', 'virgo',
                 'libra', 'scorpio', 'sagittarius', 'capricorn', 'aquarius', 'pisces']
_ELEMENTS = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]  # fire, earth, air, water
_MODALITIES = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]  # cardinal, fixed, mutable


def add_moon_position_features(df, ephemeris_df=None):
    """Add moon sign, element, modality flags from ephemeris longitude."""
    if ephemeris_df is None:
        return df

    moon_lon = _map_ephemeris_to_bars(ephemeris_df, df.index, 'moon_tropical_lon')
    if moon_lon.isna().all():
        # Try alternate column names
        for alt in ['moon_lon', 'moon_longitude']:
            moon_lon = _map_ephemeris_to_bars(ephemeris_df, df.index, alt)
            if not moon_lon.isna().all():
                break

    if moon_lon.isna().all():
        return df

    # Moon sign (0-11): each sign is 30 degrees
    moon_sign = (moon_lon.values % 360) // 30
    moon_sign = np.where(np.isnan(moon_lon.values), np.nan, moon_sign)

    new_cols = {}
    new_cols['moon_sign'] = moon_sign.astype(np.float32)

    # Element and modality
    sign_int = np.where(np.isnan(moon_sign), 0, moon_sign).astype(int) % 12
    elements_arr = np.array(_ELEMENTS)
    modalities_arr = np.array(_MODALITIES)

    new_cols['moon_element'] = np.where(np.isnan(moon_sign), np.nan,
                                   elements_arr[sign_int]).astype(np.float32)
    new_cols['moon_modality'] = np.where(np.isnan(moon_sign), np.nan,
                                    modalities_arr[sign_int]).astype(np.float32)

    # Binary flags for each sign
    for i, name in enumerate(_ZODIAC_SIGNS):
        new_cols[f'moon_in_{name}'] = np.where(np.isnan(moon_sign), 0,
                                          (moon_sign == i).astype(np.int32))

    # Element flags
    new_cols['moon_fire'] = np.where(np.isnan(moon_sign), 0,
                                np.isin(sign_int, [0, 4, 8]).astype(np.int32))
    new_cols['moon_earth'] = np.where(np.isnan(moon_sign), 0,
                                 np.isin(sign_int, [1, 5, 9]).astype(np.int32))
    new_cols['moon_air'] = np.where(np.isnan(moon_sign), 0,
                               np.isin(sign_int, [2, 6, 10]).astype(np.int32))
    new_cols['moon_water'] = np.where(np.isnan(moon_sign), 0,
                                 np.isin(sign_int, [3, 7, 11]).astype(np.int32))

    # Batch assign
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df


# ============================================================
# 8. PLANETARY ASPECT FEATURES
# ============================================================

_PLANET_PAIRS = [
    ('sun', 'moon'), ('sun', 'mercury'), ('sun', 'venus'), ('sun', 'mars'),
    ('sun', 'jupiter'), ('sun', 'saturn'), ('moon', 'mars'),
    ('jupiter', 'saturn'), ('saturn', 'uranus'), ('saturn', 'pluto'),
]

_ASPECTS = {
    'conjunction': 0,
    'sextile': 60,
    'square': 90,
    'trine': 120,
    'opposition': 180,
}

_ASPECT_ORB = 8.0  # degrees


def add_planetary_aspect_features(df, ephemeris_df=None):
    """Add binary flags for planetary aspects (conjunction, sextile, square, trine, opposition)."""
    if ephemeris_df is None:
        return df

    # Try to load planet longitudes
    planet_lons = {}
    for planet in ['sun', 'moon', 'mercury', 'venus', 'mars', 'jupiter', 'saturn', 'uranus', 'pluto']:
        for col_pattern in [f'{planet}_tropical_lon', f'{planet}_lon', f'{planet}_longitude']:
            s = _map_ephemeris_to_bars(ephemeris_df, df.index, col_pattern)
            if not s.isna().all():
                planet_lons[planet] = s.values
                break

    if len(planet_lons) < 2:
        return df

    new_cols = {}
    aspect_count = np.zeros(len(df), dtype=np.int32)

    for p1, p2 in _PLANET_PAIRS:
        if p1 not in planet_lons or p2 not in planet_lons:
            continue

        lon1 = planet_lons[p1]
        lon2 = planet_lons[p2]

        # Angular difference (0-180)
        diff = np.abs(lon1 - lon2) % 360
        diff = np.minimum(diff, 360 - diff)

        for asp_name, asp_angle in _ASPECTS.items():
            # Within orb?
            within_orb = (np.abs(diff - asp_angle) <= _ASPECT_ORB)
            col_name = f'asp_{p1}_{p2}_{asp_name}'
            col_vals = np.where(
                np.isnan(lon1) | np.isnan(lon2), 0,
                within_orb.astype(np.int32)
            )
            new_cols[col_name] = col_vals
            aspect_count += col_vals.astype(np.int32)

    new_cols['aspect_count'] = aspect_count

    # Batch assign
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df


# ============================================================
# 9. HARMONIC CYCLE FEATURES
# ============================================================

_FIB_DAYS = [21, 34, 55, 89, 144, 233, 377]
_GANN_DAYS = [90, 180, 270, 360]


def add_harmonic_cycle_features(df):
    """Add Gann/Fibonacci time cycle flags from ATH/ATL pivots (no lookahead)."""
    if 'close' not in df.columns:
        return df

    c = df['close'].astype(float)

    # Expanding max/min (no lookahead)
    ath = c.expanding(min_periods=1).max()
    atl = c.expanding(min_periods=1).min()

    # Days since ATH: where close == expanding max, reset counter
    is_ath = (c == ath).values
    is_atl = (c == atl).values

    n = len(df)
    days_since_ath = np.full(n, np.nan, dtype=np.float32)
    days_since_atl = np.full(n, np.nan, dtype=np.float32)

    last_ath_idx = -1
    last_atl_idx = -1
    for i in range(n):
        if is_ath[i]:
            last_ath_idx = i
        if is_atl[i]:
            last_atl_idx = i
        if last_ath_idx >= 0:
            days_since_ath[i] = i - last_ath_idx
        if last_atl_idx >= 0:
            days_since_atl[i] = i - last_atl_idx

    new_cols = {}
    new_cols['days_since_ath'] = days_since_ath
    new_cols['days_since_atl'] = days_since_atl

    # Fibonacci time flags (+-2 bar window from ATH)
    for fib_d in _FIB_DAYS:
        new_cols[f'fib_{fib_d}d_ath'] = (np.abs(days_since_ath - fib_d) <= 2).astype(np.int32)
        new_cols[f'fib_{fib_d}d_atl'] = (np.abs(days_since_atl - fib_d) <= 2).astype(np.int32)

    # Gann time flags (+-3 bar window from ATH)
    for gann_d in _GANN_DAYS:
        new_cols[f'gann_{gann_d}d_ath'] = (np.abs(days_since_ath - gann_d) <= 3).astype(np.int32)
        new_cols[f'gann_{gann_d}d_atl'] = (np.abs(days_since_atl - gann_d) <= 3).astype(np.int32)

    # Batch assign
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df


# ============================================================
# 10. SACRED GEOMETRY FEATURES
# ============================================================

_PHI = 1.6180339887


def add_sacred_geometry_features(df):
    """Add golden ratio extensions/retracements, sqrt price levels, phi ratios."""
    if 'close' not in df.columns or 'high' not in df.columns or 'low' not in df.columns:
        return df

    c = df['close'].astype(float).values
    h = df['high'].astype(float).values
    l = df['low'].astype(float).values

    # 20-bar swing high/low (rolling max/min)
    swing_high = pd.Series(h, index=df.index).rolling(20, min_periods=5).max().values
    swing_low = pd.Series(l, index=df.index).rolling(20, min_periods=5).min().values

    swing_range = swing_high - swing_low
    swing_range_safe = np.where(swing_range == 0, np.nan, swing_range)

    new_cols = {}

    # Golden extension 1.618 from swing low
    golden_ext = swing_low + _PHI * swing_range
    new_cols['golden_ext_1618'] = (np.abs(c - golden_ext) / np.where(golden_ext == 0, np.nan, golden_ext) < 0.01).astype(np.int32)

    # Golden retracement 0.618 from swing high
    golden_ret_618 = swing_high - 0.618 * swing_range
    new_cols['golden_ret_618'] = (np.abs(c - golden_ret_618) / np.where(swing_high == 0, np.nan, swing_high) < 0.01).astype(np.int32)

    # Golden retracement 0.382 from swing high
    golden_ret_382 = swing_high - 0.382 * swing_range
    new_cols['golden_ret_382'] = (np.abs(c - golden_ret_382) / np.where(swing_high == 0, np.nan, swing_high) < 0.01).astype(np.int32)

    # Sqrt price level: |sqrt(close) - round(sqrt(close))| < 0.05
    sqrt_c = np.sqrt(np.where(np.isnan(c), np.nan, np.abs(c)))
    new_cols['sqrt_price_level'] = (np.abs(sqrt_c - np.round(sqrt_c)) < 0.05).astype(np.int32)

    # Phi ratio of consecutive swing distances
    # Compute ratio of current swing range to previous swing range
    prev_swing_range = pd.Series(swing_range, index=df.index).shift(20).values
    prev_safe = np.where((prev_swing_range == 0) | np.isnan(prev_swing_range), np.nan, prev_swing_range)
    ratio = swing_range / prev_safe
    new_cols['price_phi_ratio'] = (np.abs(ratio - _PHI) < 0.1).astype(np.int32)

    # Batch assign
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df


# ============================================================
# 11. CALENDAR HARMONICS FEATURES
# ============================================================

def add_calendar_harmonics_features(df):
    """Add continuous sin/cos encoding for DOY, moon phase, week, month, hour."""
    if not isinstance(df.index, pd.DatetimeIndex):
        return df

    idx = df.index
    doy = idx.dayofyear.values.astype(np.float64)
    dow = idx.dayofweek.values.astype(np.float64)
    dom = idx.day.values.astype(np.float64)
    hour = idx.hour.values.astype(np.float64)

    new_cols = {}

    # DOY sin/cos (365.25 for leap year averaging)
    new_cols['doy_sin'] = np.sin(2 * np.pi * doy / 365.25).astype(np.float32)
    new_cols['doy_cos'] = np.cos(2 * np.pi * doy / 365.25).astype(np.float32)

    # Moon phase sin/cos (synodic month = 29.53 days)
    # Approximate moon phase from a known new moon reference
    # Reference new moon: 2000-01-06 18:14 UTC
    ref_new_moon_ts = 947181240.0  # unix timestamp
    bar_ts = idx.astype(np.int64).values / 10**9
    days_since_new_moon = (bar_ts - ref_new_moon_ts) / 86400.0
    moon_phase_frac = (days_since_new_moon % 29.53) / 29.53
    new_cols['moon_phase_sin'] = np.sin(2 * np.pi * moon_phase_frac).astype(np.float32)
    new_cols['moon_phase_cos'] = np.cos(2 * np.pi * moon_phase_frac).astype(np.float32)

    # Week sin/cos
    new_cols['week_sin'] = np.sin(2 * np.pi * dow / 7).astype(np.float32)
    new_cols['week_cos'] = np.cos(2 * np.pi * dow / 7).astype(np.float32)

    # Month sin/cos (30.44 avg days per month)
    new_cols['month_sin'] = np.sin(2 * np.pi * dom / 30.44).astype(np.float32)
    new_cols['month_cos'] = np.cos(2 * np.pi * dom / 30.44).astype(np.float32)

    # Hour sin/cos (intraday only — always compute, let caller filter)
    new_cols['hour_sin'] = np.sin(2 * np.pi * hour / 24).astype(np.float32)
    new_cols['hour_cos'] = np.cos(2 * np.pi * hour / 24).astype(np.float32)

    # Batch assign
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df


# ============================================================
# 12. AUTOREGRESSIVE FEATURES
# ============================================================

def add_autoregressive_features(df, predictions_col=None):
    """Add model's own prediction history + return streak features."""
    if 'close' not in df.columns:
        return df

    new_cols = {}

    # Model prediction history
    if predictions_col is not None and predictions_col in df.columns:
        pred = pd.to_numeric(df[predictions_col], errors='coerce')
        new_cols['prev_prediction_1'] = pred.shift(1).astype(np.float32)
        new_cols['prev_prediction_3'] = pred.shift(3).astype(np.float32)

        # Prediction streak: consecutive same-sign predictions
        pred_sign = np.sign(pred.values)
        new_cols['prediction_streak'] = _consecutive_count(
            np.concatenate([[False], pred_sign[1:] == pred_sign[:-1]])
        ).astype(np.int32)

    # Return streak: consecutive positive/negative returns
    c = df['close'].astype(float)
    ret = c.pct_change().values
    pos_ret = (ret > 0).astype(np.int32)
    neg_ret = (ret < 0).astype(np.int32)
    streak_pos = _consecutive_count(pos_ret).astype(np.int32)
    streak_neg = _consecutive_count(neg_ret).astype(np.int32)
    new_cols['return_streak_pos'] = streak_pos
    new_cols['return_streak_neg'] = streak_neg
    return_streak = np.where(pos_ret, streak_pos, -streak_neg).astype(np.int32)
    new_cols['return_streak'] = return_streak

    # Max streak in last 10 bars
    streak_abs = np.abs(return_streak).astype(np.float64)
    new_cols['max_return_streak_10'] = pd.Series(streak_abs, index=df.index).rolling(
        10, min_periods=1).max().astype(np.int32)

    # Batch assign
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df


# ============================================================
# 13. VEDIC ASTROLOGY FEATURES
# ============================================================

def add_vedic_astro_features(df, ephemeris_df=None):
    """Add Vedic tithi, karana, tithi_auspicious, rahu/ketu sign from ephemeris."""
    if ephemeris_df is None:
        return df

    sun_lon = _map_ephemeris_to_bars(ephemeris_df, df.index, 'sun_tropical_lon')
    moon_lon = _map_ephemeris_to_bars(ephemeris_df, df.index, 'moon_tropical_lon')

    # Try alternate column names
    if sun_lon.isna().all():
        for alt in ['sun_lon', 'sun_longitude']:
            sun_lon = _map_ephemeris_to_bars(ephemeris_df, df.index, alt)
            if not sun_lon.isna().all():
                break
    if moon_lon.isna().all():
        for alt in ['moon_lon', 'moon_longitude']:
            moon_lon = _map_ephemeris_to_bars(ephemeris_df, df.index, alt)
            if not moon_lon.isna().all():
                break

    if sun_lon.isna().all() or moon_lon.isna().all():
        return df

    sun_v = sun_lon.values
    moon_v = moon_lon.values

    new_cols = {}

    # Tithi: lunar day (1-30) from Sun-Moon elongation
    # elongation = (moon_lon - sun_lon) % 360
    # tithi = floor(elongation / 12) + 1
    elongation = (moon_v - sun_v) % 360
    tithi = np.floor(elongation / 12).astype(np.float32) + 1
    tithi = np.where(np.isnan(sun_v) | np.isnan(moon_v), np.nan, tithi)
    new_cols['tithi'] = tithi

    # Karana: half-tithi (1-60)
    karana = np.floor(elongation / 6).astype(np.float32) + 1
    karana = np.where(np.isnan(sun_v) | np.isnan(moon_v), np.nan, karana)
    new_cols['karana'] = karana

    # Auspicious tithis: 2 (Dwitiya), 3 (Tritiya), 5 (Panchami), 7 (Saptami),
    # 10 (Dashami), 11 (Ekadashi), 13 (Trayodashi), 15 (Purnima/Amavasya)
    auspicious = np.array([2, 3, 5, 7, 10, 11, 13, 15], dtype=np.float32)
    tithi_clean = np.where(np.isnan(tithi), -1, tithi)
    new_cols['tithi_auspicious'] = np.isin(tithi_clean, auspicious).astype(np.int32)

    # Rahu/Ketu signs (if available)
    for node, col_candidates in [
        ('rahu', ['rahu_tropical_lon', 'north_node_lon', 'rahu_lon']),
        ('ketu', ['ketu_tropical_lon', 'south_node_lon', 'ketu_lon']),
    ]:
        node_lon = pd.Series(np.nan, index=df.index)
        for col in col_candidates:
            node_lon = _map_ephemeris_to_bars(ephemeris_df, df.index, col)
            if not node_lon.isna().all():
                break
        if not node_lon.isna().all():
            new_cols[f'{node}_sign'] = np.where(np.isnan(node_lon.values), np.nan,
                                           (node_lon.values % 360 // 30).astype(np.float32))

    # Batch assign
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df


# ============================================================
# 14. CHINESE ASTROLOGY FEATURES
# ============================================================

def add_chinese_astro_features(df):
    """Add Chinese zodiac year animal, element, flying star period, 60-year cycle."""
    if not isinstance(df.index, pd.DatetimeIndex):
        return df

    years = df.index.year.values.astype(np.int64)

    new_cols = {}

    # Chinese year animal: (year - 4) % 12
    # 0=Rat, 1=Ox, 2=Tiger, 3=Rabbit, 4=Dragon, 5=Snake,
    # 6=Horse, 7=Goat, 8=Monkey, 9=Rooster, 10=Dog, 11=Pig
    new_cols['chinese_year_animal'] = ((years - 4) % 12).astype(np.int32)

    # Chinese year element: ((year - 4) % 10) // 2
    # 0=Wood, 1=Fire, 2=Earth, 3=Metal, 4=Water
    new_cols['chinese_year_element'] = (((years - 4) % 10) // 2).astype(np.int32)

    # Flying star period: Period 9 started Feb 4, 2024
    # Period 8: Feb 4, 2004 - Feb 3, 2024
    # Period 9: Feb 4, 2024 - Feb 3, 2044
    # Simplified: use year boundary
    bar_ts = df.index.astype(np.int64).values // 10**9
    period9_start = datetime(2024, 2, 4).timestamp()
    period8_start = datetime(2004, 2, 4).timestamp()
    new_cols['flying_star_period'] = np.where(bar_ts >= period9_start, 9,
                                np.where(bar_ts >= period8_start, 8, 7)).astype(np.int32)

    # Combined 60-year cycle: animal_element (0-59)
    new_cols['chinese_60yr_cycle'] = ((years - 4) % 60).astype(np.int32)

    # Batch assign
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df


# ============================================================
# 15. FOUR-TIER BINARIZATION
# ============================================================

@njit(cache=True)
def _rolling_percentile_vec(arr, window=200, min_periods=50, q=95):
    """Numba JIT-compiled rolling percentile using np.partition.
    O(n) per window, ~50-100x faster than interpreted Python loop.
    Called 4x per column × 4000-8000 columns = 91.6M iterations without JIT."""
    n = len(arr)
    result = np.full(n, np.nan, dtype=np.float32)
    q_frac = q / 100.0
    for i in range(min_periods, n):
        start = max(0, i - window)
        w = arr[start:i]
        # Count valid (non-NaN) entries
        count = 0
        for j in range(len(w)):
            if not np.isnan(w[j]):
                count += 1
        if count < min_periods:
            continue
        # Extract valid values into contiguous array for partition
        valid = np.empty(count, dtype=np.float32)
        vi = 0
        for j in range(len(w)):
            if not np.isnan(w[j]):
                valid[vi] = w[j]
                vi += 1
        k = min(int(count * q_frac), count - 1)
        result[i] = np.partition(valid, k)[k]
    return result


@njit(cache=True)
def _rolling_percentiles_4(arr, window=200, min_periods=50):
    """Compute q5, q20, q80, q95 in ONE pass — 4x fewer valid-value extractions."""
    n = len(arr)
    q05 = np.full(n, np.nan, dtype=np.float32)
    q20 = np.full(n, np.nan, dtype=np.float32)
    q80 = np.full(n, np.nan, dtype=np.float32)
    q95 = np.full(n, np.nan, dtype=np.float32)
    for i in range(min_periods, n):
        start = max(0, i - window)
        w = arr[start:i]
        # Extract valid values ONCE
        count = 0
        for j in range(len(w)):
            if not np.isnan(w[j]):
                count += 1
        if count < min_periods:
            continue
        valid = np.empty(count, dtype=np.float32)
        vi = 0
        for j in range(len(w)):
            if not np.isnan(w[j]):
                valid[vi] = w[j]
                vi += 1
        # Sort once, read all 4 quantiles (faster than 4 separate partitions)
        valid.sort()
        q05[i] = valid[min(int(count * 0.05), count - 1)]
        q20[i] = valid[min(int(count * 0.20), count - 1)]
        q80[i] = valid[min(int(count * 0.80), count - 1)]
        q95[i] = valid[min(int(count * 0.95), count - 1)]
    return q05, q20, q80, q95


def add_four_tier_binarization(df):
    """
    Add 4-tier binarization on ALL numeric columns — not just TA.
    Gematria, numerology, astro positions, space weather, sentiment,
    fear/greed, funding rates, everything numeric gets tiered.

    EXTREME_HIGH (>95th), HIGH (>80th), LOW (<20th), EXTREME_LOW (<5th)
    Uses rolling 200-bar window (no lookahead).
    GPU-compatible: uses numpy vectorized percentiles instead of pandas rolling.quantile.
    """
    # Skip columns that are already binary, metadata, targets, or crosses
    skip_prefixes = ('doy_', 'dx_', 'ax_', 'ax2_', 'ta2_', 'ex2_', 'sw_', 'hod_',
                     'mx_', 'vx_', 'asp_', 'pn_', 'mn_', 'rdx_', 'tx_', 'px_',
                     'cross_', 'next_', 'target_', 'is_', 'has_', 'session_',
                     'kill_zone_', 'moon_in_', 'moon_fire', 'moon_earth',
                     'moon_air', 'moon_water')
    skip_exact = {'timestamp', 'date', 'open', 'high', 'low', 'close', 'volume',
                  'quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote',
                  'open_time', 'triple_barrier_label', 'day_of_year', 'day_of_week',
                  'month', 'hour', 'minute'}

    new_cols = {}
    for col in df.columns:
        if col in skip_exact or col.startswith(skip_prefixes):
            continue
        # Skip columns that end with _HIGH, _LOW, _EXTREME_HIGH, _EXTREME_LOW (already binarized)
        if col.endswith(('_HIGH', '_LOW', '_EXTREME_HIGH', '_EXTREME_LOW', '_XH', '_XL', '_H', '_L')):
            continue

        s = pd.to_numeric(df[col], errors='coerce')
        n_unique = s.dropna().nunique()

        # Skip binary columns (0/1 only) and constant columns
        if n_unique <= 3:
            continue

        # Rolling percentiles via numpy (no pandas rolling.quantile — not supported by cuDF)
        vals = s.values.astype(np.float32)
        q05, q20, q80, q95 = _rolling_percentiles_4(vals, 200, 50)

        xh = (vals > q95).astype(np.int32)
        xl = (vals < q05).astype(np.int32)

        # Only add if they actually fire sometimes
        if xh.sum() > 3:
            new_cols[f'{col}_EXTREME_HIGH'] = xh
        if xl.sum() > 3:
            new_cols[f'{col}_EXTREME_LOW'] = xl
        if f'{col}_HIGH' not in df.columns:
            h = (vals > q80).astype(np.int32)
            if h.sum() > 3:
                new_cols[f'{col}_HIGH'] = h
        if f'{col}_LOW' not in df.columns:
            lo = (vals < q20).astype(np.int32)
            if lo.sum() > 3:
                new_cols[f'{col}_LOW'] = lo

    # Batch assign
    if new_cols:
        new_df = pd.DataFrame(new_cols, index=df.index)
        df = pd.concat([df, new_df], axis=1)

    return df


# ============================================================
# 16. DOY WINDOW FEATURES
# ============================================================

def add_doy_window_features(df):
    """Replace exact DOY flags with +-2 day window flags for more training samples."""
    if not isinstance(df.index, pd.DatetimeIndex):
        return df

    doy = df.index.dayofyear.values

    new_cols = {}
    for d in range(1, 366):
        col = f'doy_{d}'
        if col not in df.columns:
            continue
        # Create window: doy within [d-2, d+2], wrapping around year boundary
        in_window = np.zeros(len(df), dtype=np.int32)
        for offset in range(-2, 3):
            target = d + offset
            if target <= 0:
                target += 365
            elif target > 365:
                target -= 365
            in_window |= (doy == target).astype(np.int32)
        new_cols[f'doy_{d}_w2'] = in_window

    # Batch assign
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df


# ============================================================
# 17. TICKER GEMATRIA FEATURES
# ============================================================

def add_ticker_gematria_features(df, symbol='BTCUSDT'):
    """Add gematria of ticker name and resonance with price digital root."""
    # Ordinal gematria: A=1, B=2, ..., Z=26
    ticker_clean = ''.join(c for c in symbol.upper() if c.isalpha())
    ord_values = np.array([ord(c) - ord('A') + 1 for c in ticker_clean], dtype=np.int64)
    ticker_gem_ordinal = int(ord_values.sum()) if len(ord_values) > 0 else 0

    # Digital root of ordinal
    ticker_gem_dr = int(_digital_root_vec(np.array([ticker_gem_ordinal]))[0])

    # Reduction gematria: sum digits until single digit (same as digital root)
    ticker_gem_reduction = ticker_gem_dr

    new_cols = {}
    new_cols['ticker_gem_ordinal'] = np.int32(ticker_gem_ordinal)
    new_cols['ticker_gem_dr'] = np.int32(ticker_gem_dr)
    new_cols['ticker_gem_reduction'] = np.int32(ticker_gem_reduction)

    # Resonance: price DR matches ticker DR
    if 'price_dr' in df.columns:
        price_dr = pd.to_numeric(df['price_dr'], errors='coerce').values
        new_cols['price_dr_matches_ticker_dr'] = np.where(
            np.isnan(price_dr), 0,
            (price_dr == ticker_gem_dr).astype(np.int32)
        )
    elif 'digital_root_price' in df.columns:
        price_dr = pd.to_numeric(df['digital_root_price'], errors='coerce').values
        new_cols['price_dr_matches_ticker_dr'] = np.where(
            np.isnan(price_dr), 0,
            (price_dr == ticker_gem_dr).astype(np.int32)
        )
    else:
        # Compute price DR on the fly
        if 'close' in df.columns:
            c = df['close'].astype(float).values
            c_int = np.where(np.isnan(c), 0, np.abs(c).astype(np.int64))
            price_dr = _digital_root_vec(c_int).astype(np.float32)
            new_cols['price_dr_matches_ticker_dr'] = np.where(
                np.isnan(c), 0,
                (price_dr == ticker_gem_dr).astype(np.int32)
            )

    # Batch assign
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df


# ============================================================
# 18. CORRELATION REGIME FEATURES
# ============================================================

def add_correlation_regime_features(df, inverse_signals=None):
    """Add rolling correlation features between BTC and other assets (UUP, TLT, FXY)."""
    if inverse_signals is None or 'close' not in df.columns:
        return df

    btc_returns = df['close'].astype(float).pct_change()

    new_cols = {}
    for asset_name, asset_df in inverse_signals.items():
        if asset_df is None or 'close' not in asset_df.columns:
            continue

        asset_close = asset_df['close'].astype(float)
        asset_returns = asset_close.pct_change()

        # Align to BTC index
        asset_returns_aligned = asset_returns.reindex(df.index, method='ffill')

        # 30-day rolling correlation
        corr = btc_returns.rolling(30, min_periods=15).corr(asset_returns_aligned)
        clean_name = asset_name.lower().replace(' ', '_')
        new_cols[f'btc_{clean_name}_corr_30d'] = corr.astype(np.float32)

        # Regime flags
        new_cols[f'btc_{clean_name}_decoupled'] = (corr.abs() < 0.2).astype(np.int32)
        new_cols[f'btc_{clean_name}_high_corr'] = (corr.abs() > 0.7).astype(np.int32)

    # Batch assign
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df


# ============================================================
# 19. V2 SIGNAL FEATURES
# ============================================================

def add_v2_signal_features(df, astro_cache=None):
    """Add V2 streamer data: DeFi TVL, BTC dominance, hash rate, COT positioning."""
    if astro_cache is None:
        return df

    if not isinstance(df.index, pd.DatetimeIndex):
        return df

    df_date_idx = df.index.normalize()
    if hasattr(df_date_idx, 'tz') and df_date_idx.tz is not None:
        df_date_idx = df_date_idx.tz_localize(None)

    new_cols = {}

    # DeFi TVL trend
    defi_df = astro_cache.get('defi_tvl', None)
    if defi_df is not None and 'tvl' in defi_df.columns:
        tvl = pd.to_numeric(defi_df['tvl'], errors='coerce')
        tvl.index = tvl.index.normalize()
        if hasattr(tvl.index, 'tz') and tvl.index.tz is not None:
            tvl.index = tvl.index.tz_localize(None)
        tvl_mapped = tvl.reindex(df_date_idx).ffill()
        tvl_mapped.index = df.index
        tvl_sma = tvl_mapped.rolling(7, min_periods=1).mean()
        new_cols['defi_tvl_rising'] = (tvl_mapped > tvl_sma).astype(np.int32)
        new_cols['defi_tvl_declining'] = (tvl_mapped < tvl_sma).astype(np.int32)

    # BTC dominance
    dom_df = astro_cache.get('btc_dominance', None)
    if dom_df is not None and 'dominance' in dom_df.columns:
        dom = pd.to_numeric(dom_df['dominance'], errors='coerce')
        dom.index = dom.index.normalize()
        if hasattr(dom.index, 'tz') and dom.index.tz is not None:
            dom.index = dom.index.tz_localize(None)
        dom_mapped = dom.reindex(df_date_idx).ffill()
        dom_mapped.index = df.index
        dom_sma = dom_mapped.rolling(7, min_periods=1).mean()
        new_cols['btc_dom_high'] = (dom_mapped > 60).astype(np.int32)
        new_cols['btc_dom_low'] = (dom_mapped < 40).astype(np.int32)
        new_cols['btc_dom_rising'] = (dom_mapped > dom_sma).astype(np.int32)
        new_cols['btc_dom_declining'] = (dom_mapped < dom_sma).astype(np.int32)

    # Hash rate trend
    hash_df = astro_cache.get('hash_rate', None)
    if hash_df is not None and 'hash_rate' in hash_df.columns:
        hr = pd.to_numeric(hash_df['hash_rate'], errors='coerce')
        hr.index = hr.index.normalize()
        if hasattr(hr.index, 'tz') and hr.index.tz is not None:
            hr.index = hr.index.tz_localize(None)
        hr_mapped = hr.reindex(df_date_idx).ffill()
        hr_mapped.index = df.index
        hr_sma = hr_mapped.rolling(7, min_periods=1).mean()
        new_cols['hash_rate_rising'] = (hr_mapped > hr_sma).astype(np.int32)
        new_cols['hash_rate_declining'] = (hr_mapped < hr_sma).astype(np.int32)

    # COT positioning
    cot_df = astro_cache.get('cot', None)
    if cot_df is not None and 'net_position' in cot_df.columns:
        cot = pd.to_numeric(cot_df['net_position'], errors='coerce')
        cot.index = cot.index.normalize()
        if hasattr(cot.index, 'tz') and cot.index.tz is not None:
            cot.index = cot.index.tz_localize(None)
        cot_mapped = cot.reindex(df_date_idx).ffill()
        cot_mapped.index = df.index
        cot_q80 = cot_mapped.rolling(200, min_periods=20).quantile(0.8)
        cot_q20 = cot_mapped.rolling(200, min_periods=20).quantile(0.2)
        new_cols['cot_extreme_long'] = (cot_mapped > cot_q80).astype(np.int32)
        new_cols['cot_extreme_short'] = (cot_mapped < cot_q20).astype(np.int32)

    # Batch assign
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    return df


# ============================================================
# 20. MASTER FUNCTION
# ============================================================

def add_all_v2_layers(df, symbol='BTCUSDT', tf='1h', astro_cache=None,
                      inverse_signals=None, predictions_col=None):
    """
    Master function: calls all V2 feature layers in order.
    Handles missing data gracefully (returns NaN for missing inputs).

    Args:
        df: DataFrame with DatetimeIndex, OHLCV columns + existing V1 features
        symbol: Ticker symbol (e.g. 'BTCUSDT')
        tf: Timeframe string (e.g. '5m', '15m', '1h', '4h', '1d', '1w')
        astro_cache: dict with ephemeris, defi_tvl, btc_dominance, hash_rate, cot data
        inverse_signals: dict of asset DataFrames for correlation features
        predictions_col: column name of model predictions (for autoregressive features)

    Returns:
        DataFrame with all V2 feature columns added.
    """
    # V2 layers use index/lookup/arithmetic ops that don't benefit from cuDF
    # Convert to pandas for compatibility, results are assigned back
    _was_gpu = False
    try:
        from feature_library import _is_gpu, _to_cpu
        if _is_gpu(df):
            _was_gpu = True
            df = _to_cpu(df)
    except ImportError:
        pass

    ephemeris_df = None
    if astro_cache is not None:
        ephemeris_df = astro_cache.get('ephemeris', None)
        if ephemeris_df is None:
            ephemeris_df = astro_cache.get('astrology', None)

    # 1. Time of day
    df = add_time_of_day_features(df, tf=tf)

    # 2. Rate of change (needs _HIGH/_LOW columns from V1 binarization)
    df = add_rate_of_change_features(df)

    # 3. Sequential memory
    df = add_sequential_memory_features(df)

    # 4. Price numerology
    df = add_price_numerology_features(df)

    # 5. Calendar anomalies
    df = add_calendar_anomaly_features(df)

    # 6. Entropy & fractal
    df = add_entropy_fractal_features(df)

    # 7. Moon position
    df = add_moon_position_features(df, ephemeris_df=ephemeris_df)

    # 8. Planetary aspects
    df = add_planetary_aspect_features(df, ephemeris_df=ephemeris_df)

    # 9. Harmonic cycles
    df = add_harmonic_cycle_features(df)

    # 10. Sacred geometry
    df = add_sacred_geometry_features(df)

    # 11. Calendar harmonics
    df = add_calendar_harmonics_features(df)

    # 12. Autoregressive
    df = add_autoregressive_features(df, predictions_col=predictions_col)

    # 13. Vedic astrology
    df = add_vedic_astro_features(df, ephemeris_df=ephemeris_df)

    # 14. Chinese astrology
    df = add_chinese_astro_features(df)

    # 15. Four-tier binarization
    df = add_four_tier_binarization(df)

    # 16. DOY window features
    df = add_doy_window_features(df)

    # 17. Ticker gematria
    df = add_ticker_gematria_features(df, symbol=symbol)

    # 18. Correlation regime
    df = add_correlation_regime_features(df, inverse_signals=inverse_signals)

    # 19. V2 signal features
    df = add_v2_signal_features(df, astro_cache=astro_cache)

    return df
