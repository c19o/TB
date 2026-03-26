#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
feature_library.py — Shared Feature Computation Library
========================================================
Pure-function feature library. NO database calls.
Takes pre-loaded data, returns features.

Used by BOTH:
  - build_*_features.py (offline/backfill)
  - live_trader.py (live/incremental)

This eliminates training/live drift by ensuring identical feature
computation in both modes.
"""

import math
import os
import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings('ignore')

# ── CUDA version detection (BEFORE any GPU library imports) ──
# RAPIDS 25.02 (cuDF, cuML, CuPy-cuda12x) are compiled for CUDA 12.x.
# On CUDA 13.0+ (driver 580+), ALL RAPIDS GPU operations SEGFAULT.
# Detect driver version FIRST and skip ALL GPU imports on CUDA 13+.
_CUDA_MAJOR = 0
try:
    import subprocess as _sp
    _nv = _sp.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                   capture_output=True, text=True, timeout=5)
    _drv = int(_nv.stdout.strip().split('.')[0])
    _CUDA_MAJOR = 13 if _drv >= 580 else 12
except Exception:
    _CUDA_MAJOR = 12  # assume CUDA 12 compatible

# GPU acceleration — skip entirely on CUDA 13+ (RAPIDS binary incompat)
if _CUDA_MAJOR >= 13:
    cp = None
    _HAS_GPU = False
    _N_GPUS = 0
    os.environ['V2_SKIP_GPU'] = '1'  # Signal to sub-modules (knn_feature_engine, etc.)
    print(f"[feature_library] GPU DISABLED — CUDA {_CUDA_MAJOR}.x driver (580+). RAPIDS 25.02 needs CUDA 12.x. Using CPU mode.")
else:
    try:
        import cupy as cp
        _N_GPUS = cp.cuda.runtime.getDeviceCount()
        _HAS_GPU = _N_GPUS > 0
        if _HAS_GPU:
            _gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
            print(f"[feature_library] CuPy GPU: {_gpu_name} x{_N_GPUS}")
    except (ImportError, Exception):
        cp = None
        _HAS_GPU = False
        _N_GPUS = 0

try:
    if _CUDA_MAJOR >= 13:
        raise ImportError("CUDA 13+ — all RAPIDS disabled")
    import cudf
    _HAS_CUDF = _HAS_GPU
    if _HAS_CUDF:
        print(f"[feature_library] cuDF available — GPU DataFrames enabled")
except ImportError:
    cudf = None
    _HAS_CUDF = False

try:
    from numba import njit, prange
    _HAS_NUMBA = True
    print(f"[feature_library] Numba available — JIT-compiled loops enabled")
except ImportError:
    _HAS_NUMBA = False
    # Fallback: identity decorator so @_njit doesn't break without numba
    def njit(*args, **kwargs):
        def _wrap(f):
            return f
        if args and callable(args[0]):
            return args[0]
        return _wrap
    def prange(*a): return range(*a)

try:
    if _CUDA_MAJOR >= 13:
        raise ImportError("CUDA 13+ driver — cuML disabled (same RAPIDS 25.02 binary compat issue)")
    from cuml.neighbors import KNeighborsClassifier as cuml_KNN
    _HAS_CUML = True
    print(f"[feature_library] cuML KNN available — GPU-accelerated KNN")
except ImportError:
    _HAS_CUML = False

try:
    if _CUDA_MAJOR >= 13:
        raise ImportError("CUDA 13+ — CuPy kernels segfault at runtime")
    from cupyx.scipy.signal import convolve as cp_convolve
    _HAS_CP_CONV = True
    print(f"[feature_library] CuPy convolve available — GPU FFD")
except ImportError:
    _HAS_CP_CONV = False


def _strip_tz(idx):
    """Strip timezone from a DatetimeIndex to avoid datetime64[ns,UTC] vs datetime64[ns] mismatch."""
    if hasattr(idx, 'tz') and idx.tz is not None:
        return idx.tz_localize(None)
    return idx


def _to_gpu(df):
    """Convert pandas DataFrame to cuDF for GPU processing"""
    if _is_gpu(df):
        return df
    if _HAS_CUDF:
        try:
            # Strip timezone info before converting (cuDF timezone support is limited)
            for col in df.columns:
                if hasattr(df[col], 'dt') and hasattr(df[col].dt, 'tz') and df[col].dt.tz is not None:
                    df[col] = df[col].dt.tz_localize(None)
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return cudf.from_pandas(df)
        except Exception as e:
            import logging
            logging.getLogger('feature_library').warning(f"GPU FALLBACK: cuDF conversion failed ({e}), falling back to CPU pandas")
            return df
    return df


def _to_cpu(df):
    """Convert cuDF DataFrame back to pandas"""
    if _HAS_CUDF and hasattr(df, 'to_pandas'):
        return df.to_pandas()
    return df


def _is_gpu(df):
    """Check if DataFrame is on GPU"""
    return _HAS_CUDF and (hasattr(df, '__cuda_array_interface__') or (hasattr(df, 'to_pandas') and type(df).__module__.startswith('cudf')))


def _np(x):
    """Extract numpy array — universal driver compat (CUDA 12.8 + 13.0)."""
    if hasattr(x, 'to_pandas'):
        try:
            return x.to_pandas().values
        except Exception:
            pass
    if hasattr(x, 'to_numpy'):
        try:
            return x.to_numpy()
        except (ValueError, Exception):
            return x.to_numpy(dtype='float64', na_value=np.nan)
    if cp is not None and hasattr(x, 'get'):
        return x.get()
    return np.asarray(x)


def _concat(objs, **kwargs):
    """Module-aware concat (cudf.concat or pd.concat)."""
    if objs and _HAS_CUDF and _is_gpu(objs[0]):
        return cudf.concat(objs, **kwargs)
    return pd.concat(objs, **kwargs)


# ============================================================
# NUMBA JIT-COMPILED STATEFUL LOOPS
# ============================================================

@njit(cache=True)
def _sar_loop(h_vals, l_vals, af_start, af_step, af_max):
    """Parabolic SAR — stateful sequential loop, ~50-100x faster with Numba."""
    n = len(h_vals)
    sar_arr = np.empty(n)
    sar_bull = np.ones(n, dtype=np.int8)
    sar_val = l_vals[0]
    af = af_start
    bull = True
    ep = h_vals[0]
    sar_arr[0] = sar_val
    for i in range(1, n):
        prev_sar = sar_val
        sar_val = prev_sar + af * (ep - prev_sar)
        if bull:
            if l_vals[i] < sar_val:
                bull = False
                sar_val = ep
                ep = l_vals[i]
                af = af_start
            else:
                if h_vals[i] > ep:
                    ep = h_vals[i]
                    af = min(af + af_step, af_max)
        else:
            if h_vals[i] > sar_val:
                bull = True
                sar_val = ep
                ep = h_vals[i]
                af = af_start
            else:
                if l_vals[i] < ep:
                    ep = l_vals[i]
                    af = min(af + af_step, af_max)
        sar_arr[i] = sar_val
        sar_bull[i] = 1 if bull else 0
    return sar_arr, sar_bull


@njit(cache=True)
def _supertrend_loop(c_vals, ub_vals, lb_vals):
    """Supertrend — stateful direction tracking, compiled."""
    n = len(c_vals)
    st_dir = np.ones(n, dtype=np.int64)
    st_vals = np.empty(n)
    st_vals[0] = np.nan
    for i in range(1, n):
        if c_vals[i] > ub_vals[i - 1]:
            st_dir[i] = 1
        elif c_vals[i] < lb_vals[i - 1]:
            st_dir[i] = -1
        else:
            st_dir[i] = st_dir[i - 1]
        st_vals[i] = lb_vals[i] if st_dir[i] == 1 else ub_vals[i]
    return st_vals, st_dir


@njit(cache=True)
def _consec_count_loop(is_true):
    """Consecutive True count — stateful sequential, compiled."""
    n = len(is_true)
    out = np.zeros(n, dtype=np.int64)
    for i in range(1, n):
        if is_true[i]:
            out[i] = out[i - 1] + 1
    return out


@njit(cache=True)
def _elliott_zigzag_loop(c_vals, pct_threshold):
    """Elliott zigzag — stateful direction tracking, compiled."""
    n = len(c_vals)
    zigzag = np.zeros(n, dtype=np.int64)
    direction = 1
    last_pivot_val = c_vals[0]
    for i in range(1, n):
        if direction == 1:
            if c_vals[i] < last_pivot_val * (1.0 - pct_threshold):
                direction = -1
                last_pivot_val = c_vals[i]
                zigzag[i] = -1
            elif c_vals[i] > last_pivot_val:
                last_pivot_val = c_vals[i]
        else:
            if c_vals[i] > last_pivot_val * (1.0 + pct_threshold):
                direction = 1
                last_pivot_val = c_vals[i]
                zigzag[i] = 1
            elif c_vals[i] < last_pivot_val:
                last_pivot_val = c_vals[i]
    return zigzag, direction


@njit(cache=True)
def _gann_sq9_vec(c_vals):
    """Gann Square of 9 — vectorized per-price, compiled."""
    n = len(c_vals)
    gann_level = np.empty(n)
    gann_dist = np.empty(n)
    for i in range(n):
        price = c_vals[i]
        if price <= 0.0 or np.isnan(price):
            gann_level[i] = np.nan
            gann_dist[i] = np.nan
            continue
        sqrt_p = math.sqrt(price)
        lower = math.floor(sqrt_p) ** 2
        upper = math.ceil(sqrt_p) ** 2
        if upper == lower:
            upper = (int(math.ceil(sqrt_p)) + 1) ** 2
        dist_lower = abs(price - lower) / price
        dist_upper = abs(price - upper) / price
        gann_level[i] = lower if dist_lower < dist_upper else upper
        gann_dist[i] = min(dist_lower, dist_upper)
    return gann_level, gann_dist


@njit(cache=True)
def _wyckoff_loop(c_arr, h_arr, l_arr, o_arr, v_arr, vol_ma_arr, atr_arr,
                  roll_low_30, bar_spread, n):
    """Wyckoff state machine — full 127-line loop, compiled to native code."""
    wk_phase = np.zeros(n, dtype=np.int8)
    wk_in_range = np.zeros(n, dtype=np.int8)
    wk_range_pos = np.full(n, np.nan)
    wk_sc_bars_ago = np.full(n, np.nan)
    wk_spring = np.zeros(n, dtype=np.int8)
    wk_upthrust = np.zeros(n, dtype=np.int8)
    wk_sos = np.zeros(n, dtype=np.int8)
    wk_sow = np.zeros(n, dtype=np.int8)
    wk_lps_count = np.zeros(n, dtype=np.int32)
    wk_lpsy_count = np.zeros(n, dtype=np.int32)
    wk_dir_score = np.full(n, np.nan)
    wk_range_width = np.full(n, np.nan)
    wk_bars_in_range = np.zeros(n, dtype=np.int32)

    sc_idx = -1
    sc_low = np.nan
    sc_close = np.nan
    sc_vol = np.nan
    ar_idx = -1
    ar_high = np.nan
    st_idx = -1
    range_support = np.nan
    range_resistance = np.nan
    range_start_idx = -1
    range_defined = False
    spring_detected = False
    upthrust_detected = False
    sos_detected = False
    sow_detected = False
    lps_count = 0
    lpsy_count = 0
    prev_low_in_range = np.nan
    prev_high_in_range = np.nan
    phase = 0

    for i in range(30, n):
        cur_atr = atr_arr[i]
        if np.isnan(cur_atr) or cur_atr == 0.0:
            cur_atr = bar_spread[i] if bar_spread[i] > 0 else 1e-10

        # SC detection
        if (not np.isnan(vol_ma_arr[i]) and vol_ma_arr[i] > 0
                and v_arr[i] > 3.0 * vol_ma_arr[i]
                and bar_spread[i] > 1.5 * cur_atr
                and not np.isnan(roll_low_30[i])
                and l_arr[i] <= roll_low_30[i]
                and bar_spread[i] > 0
                and (c_arr[i] - l_arr[i]) / bar_spread[i] > 0.25):
            sc_idx = i
            sc_low = l_arr[i]
            sc_close = c_arr[i]
            sc_vol = v_arr[i]
            ar_idx = -1
            ar_high = np.nan
            st_idx = -1
            range_defined = False
            spring_detected = False
            upthrust_detected = False
            sos_detected = False
            sow_detected = False
            lps_count = 0
            lpsy_count = 0
            prev_low_in_range = np.nan
            prev_high_in_range = np.nan
            phase = 1
            range_start_idx = i

        # AR detection
        if sc_idx >= 0 and ar_idx < 0 and 3 <= (i - sc_idx) <= 15:
            window_start = max(0, i - 2)
            window_end = min(n - 1, i + 2)
            local_max = np.max(h_arr[window_start:window_end + 1])
            if (h_arr[i] >= local_max
                    and c_arr[i] > sc_close + 1.5 * cur_atr):
                ar_idx = i
                ar_high = h_arr[i]

        # ST detection
        if sc_idx >= 0 and ar_idx >= 0 and st_idx < 0 and i > ar_idx:
            if (l_arr[i] <= sc_low * 1.05
                    and v_arr[i] < 0.6 * sc_vol):
                st_idx = i
                range_support = sc_low
                range_resistance = ar_high
                range_defined = True
                phase = 2

        if sc_idx >= 0:
            wk_sc_bars_ago[i] = float(i - sc_idx)

        if range_defined and range_resistance > range_support:
            rng_width = range_resistance - range_support
            wk_in_range[i] = 1 if range_support <= c_arr[i] <= range_resistance else 0
            wk_range_pos[i] = (c_arr[i] - range_support) / rng_width
            wk_range_width[i] = rng_width / c_arr[i] if c_arr[i] > 0 else np.nan
            wk_bars_in_range[i] = i - range_start_idx
            mid = (range_support + range_resistance) / 2.0

            if (l_arr[i] < range_support
                    and c_arr[i] >= range_support
                    and not np.isnan(vol_ma_arr[i])
                    and v_arr[i] < 1.0 * vol_ma_arr[i]):
                wk_spring[i] = 1
                spring_detected = True
                if phase < 3:
                    phase = 3

            if (h_arr[i] > range_resistance
                    and c_arr[i] <= range_resistance):
                wk_upthrust[i] = 1
                upthrust_detected = True
                if phase < 3:
                    phase = 3

            if (c_arr[i] > mid
                    and c_arr[i] > c_arr[i - 1]
                    and not np.isnan(vol_ma_arr[i])
                    and v_arr[i] > 1.5 * vol_ma_arr[i]):
                wk_sos[i] = 1
                sos_detected = True
                if spring_detected and phase < 4:
                    phase = 4

            if (c_arr[i] < mid
                    and c_arr[i] < c_arr[i - 1]
                    and not np.isnan(vol_ma_arr[i])
                    and v_arr[i] > 1.5 * vol_ma_arr[i]):
                wk_sow[i] = 1
                sow_detected = True
                if upthrust_detected and phase < 4:
                    phase = 4

            if range_support <= l_arr[i] <= range_resistance:
                if not np.isnan(prev_low_in_range) and l_arr[i] > prev_low_in_range:
                    lps_count += 1
                prev_low_in_range = l_arr[i]
            wk_lps_count[i] = lps_count

            if range_support <= h_arr[i] <= range_resistance:
                if not np.isnan(prev_high_in_range) and h_arr[i] < prev_high_in_range:
                    lpsy_count += 1
                prev_high_in_range = h_arr[i]
            wk_lpsy_count[i] = lpsy_count

            if phase >= 4:
                if c_arr[i] > range_resistance + 0.5 * rng_width:
                    phase = 5
                elif c_arr[i] < range_support - 0.5 * rng_width:
                    phase = 5

        wk_phase[i] = phase

    return (wk_phase, wk_in_range, wk_range_pos, wk_sc_bars_ago,
            wk_spring, wk_upthrust, wk_sos, wk_sow,
            wk_lps_count, wk_lpsy_count, wk_dir_score,
            wk_range_width, wk_bars_in_range)


@njit(cache=True)
def _avwap_loop(h_arr, l_arr, c_arr, v_arr, typical_price, n, swing_lookback):
    """Anchored VWAP from swing low/high — compiled loop. ~20-50x faster than Python."""
    avwap_low_arr = np.full(n, np.nan)
    avwap_high_arr = np.full(n, np.nan)
    avwap_pos_arr = np.full(n, np.nan)
    for i in range(swing_lookback, n):
        lookback_start = i - swing_lookback if i >= swing_lookback else 0
        # Find swing low/high index in lookback window
        min_val = l_arr[lookback_start]
        min_idx = lookback_start
        max_val = h_arr[lookback_start]
        max_idx = lookback_start
        for k in range(lookback_start + 1, i + 1):
            if l_arr[k] < min_val:
                min_val = l_arr[k]
                min_idx = k
            if h_arr[k] > max_val:
                max_val = h_arr[k]
                max_idx = k
        # AVWAP from swing low
        cum_tpv = 0.0
        cum_v = 0.0
        for k in range(min_idx, i + 1):
            cum_tpv += typical_price[k] * v_arr[k]
            cum_v += v_arr[k]
        if cum_v > 0:
            avwap_low_arr[i] = cum_tpv / cum_v
        # AVWAP from swing high
        cum_tpv = 0.0
        cum_v = 0.0
        for k in range(max_idx, i + 1):
            cum_tpv += typical_price[k] * v_arr[k]
            cum_v += v_arr[k]
        if cum_v > 0:
            avwap_high_arr[i] = cum_tpv / cum_v
        # Position
        if not np.isnan(avwap_low_arr[i]) and not np.isnan(avwap_high_arr[i]):
            if c_arr[i] > avwap_low_arr[i] and c_arr[i] > avwap_high_arr[i]:
                avwap_pos_arr[i] = 1.0
            elif c_arr[i] < avwap_low_arr[i] and c_arr[i] < avwap_high_arr[i]:
                avwap_pos_arr[i] = -1.0
            else:
                avwap_pos_arr[i] = 0.0
    return avwap_low_arr, avwap_high_arr, avwap_pos_arr


@njit(cache=True)
def _ict_smc_loop(h_arr, l_arr, c_arr, o_arr, fvg_bull, fvg_bear, n):
    """ICT/SMC features: FVG nearest distance, BOS, liquidity sweep, order blocks. All compiled."""
    BOS_N = 20
    MAX_FVG = 50

    fvg_nearest_dist = np.full(n, np.nan)
    bos_dir = np.zeros(n, dtype=np.float64)
    liq_sweep = np.zeros(n, dtype=np.float64)
    ob_dist = np.full(n, np.nan)

    # Fixed-size arrays for active FVGs (replace Python lists)
    bull_fvg_lo = np.empty(MAX_FVG, dtype=np.float64)
    bull_fvg_hi = np.empty(MAX_FVG, dtype=np.float64)
    bear_fvg_lo = np.empty(MAX_FVG, dtype=np.float64)
    bear_fvg_hi = np.empty(MAX_FVG, dtype=np.float64)
    n_bull_fvg = 0
    n_bear_fvg = 0

    last_bull_ob = np.nan
    last_bear_ob = np.nan

    for i in range(2, n):
        # --- FVG tracking ---
        if fvg_bull[i]:
            if n_bull_fvg < MAX_FVG:
                bull_fvg_lo[n_bull_fvg] = h_arr[i - 2]
                bull_fvg_hi[n_bull_fvg] = l_arr[i]
                n_bull_fvg += 1
            else:
                # Shift left (drop oldest)
                for k in range(MAX_FVG - 1):
                    bull_fvg_lo[k] = bull_fvg_lo[k + 1]
                    bull_fvg_hi[k] = bull_fvg_hi[k + 1]
                bull_fvg_lo[MAX_FVG - 1] = h_arr[i - 2]
                bull_fvg_hi[MAX_FVG - 1] = l_arr[i]

        if fvg_bear[i]:
            if n_bear_fvg < MAX_FVG:
                bear_fvg_lo[n_bear_fvg] = h_arr[i]
                bear_fvg_hi[n_bear_fvg] = l_arr[i - 2]
                n_bear_fvg += 1
            else:
                for k in range(MAX_FVG - 1):
                    bear_fvg_lo[k] = bear_fvg_lo[k + 1]
                    bear_fvg_hi[k] = bear_fvg_hi[k + 1]
                bear_fvg_lo[MAX_FVG - 1] = h_arr[i]
                bear_fvg_hi[MAX_FVG - 1] = l_arr[i - 2]

        # Remove filled bull FVGs
        write = 0
        for k in range(n_bull_fvg):
            if l_arr[i] > bull_fvg_lo[k]:
                bull_fvg_lo[write] = bull_fvg_lo[k]
                bull_fvg_hi[write] = bull_fvg_hi[k]
                write += 1
        n_bull_fvg = write

        # Remove filled bear FVGs
        write = 0
        for k in range(n_bear_fvg):
            if h_arr[i] < bear_fvg_hi[k]:
                bear_fvg_lo[write] = bear_fvg_lo[k]
                bear_fvg_hi[write] = bear_fvg_hi[k]
                write += 1
        n_bear_fvg = write

        # Nearest FVG distance
        min_dist = np.inf
        for k in range(n_bull_fvg):
            mid = (bull_fvg_lo[k] + bull_fvg_hi[k]) / 2.0
            d = abs(c_arr[i] - mid)
            if d < min_dist:
                min_dist = d
        for k in range(n_bear_fvg):
            mid = (bear_fvg_lo[k] + bear_fvg_hi[k]) / 2.0
            d = abs(c_arr[i] - mid)
            if d < min_dist:
                min_dist = d
        if min_dist < np.inf and c_arr[i] > 0:
            fvg_nearest_dist[i] = min_dist / c_arr[i]

        # --- BOS ---
        if i >= BOS_N:
            highest_high = h_arr[i - BOS_N]
            lowest_low = l_arr[i - BOS_N]
            for k in range(i - BOS_N + 1, i):
                if h_arr[k] > highest_high:
                    highest_high = h_arr[k]
                if l_arr[k] < lowest_low:
                    lowest_low = l_arr[k]
            if c_arr[i] > highest_high:
                bos_dir[i] = 1.0
            elif c_arr[i] < lowest_low:
                bos_dir[i] = -1.0

        # --- Liquidity Sweep ---
        if i >= BOS_N + 1:
            swing_high = h_arr[i - BOS_N]
            swing_low = l_arr[i - BOS_N]
            for k in range(i - BOS_N + 1, i):
                if h_arr[k] > swing_high:
                    swing_high = h_arr[k]
                if l_arr[k] < swing_low:
                    swing_low = l_arr[k]
            if l_arr[i] < swing_low and c_arr[i] > swing_low:
                liq_sweep[i] = 1.0
            elif h_arr[i] > swing_high and c_arr[i] < swing_high:
                liq_sweep[i] = -1.0

        # --- Order Block ---
        if i >= BOS_N + 1:
            if bos_dir[i] == 1.0:
                for j in range(i - 1, max(i - BOS_N - 1, -1), -1):
                    if c_arr[j] < o_arr[j]:
                        last_bull_ob = (o_arr[j] + c_arr[j]) / 2.0
                        break
            elif bos_dir[i] == -1.0:
                for j in range(i - 1, max(i - BOS_N - 1, -1), -1):
                    if c_arr[j] > o_arr[j]:
                        last_bear_ob = (o_arr[j] + c_arr[j]) / 2.0
                        break

            min_ob = np.inf
            if not np.isnan(last_bull_ob) and c_arr[i] > 0:
                d_ob = abs(c_arr[i] - last_bull_ob) / c_arr[i]
                if d_ob < min_ob:
                    min_ob = d_ob
            if not np.isnan(last_bear_ob) and c_arr[i] > 0:
                d_ob = abs(c_arr[i] - last_bear_ob) / c_arr[i]
                if d_ob < min_ob:
                    min_ob = d_ob
            if min_ob < np.inf:
                ob_dist[i] = min_ob

    return fvg_nearest_dist, bos_dir, liq_sweep, ob_dist


@njit(cache=True)
def _volume_profile_loop(h_arr, l_arr, c_arr, v_arr, n, vp_window, vp_bins):
    """Volume Profile / POC Migration — nested loop, compiled."""
    vpoc_arr = np.full(n, np.nan)
    vpoc_dist_arr = np.full(n, np.nan)
    vpoc_mig_arr = np.full(n, np.nan)
    va_pos_arr = np.full(n, np.nan)
    hvn_dist_arr = np.full(n, np.nan)
    lvn_dist_arr = np.full(n, np.nan)

    for i in range(vp_window, n):
        ws = i - vp_window
        w_h = h_arr[ws:i + 1]
        w_l = l_arr[ws:i + 1]
        w_c = c_arr[ws:i + 1]
        w_v = v_arr[ws:i + 1]

        rng_high = np.nanmax(w_h)
        rng_low = np.nanmin(w_l)
        if rng_high <= rng_low or np.isnan(rng_high):
            continue

        bin_edges = np.linspace(rng_low, rng_high, vp_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        bin_vol = np.zeros(vp_bins)

        for j in range(len(w_c)):
            bar_lo = w_l[j]
            bar_hi = w_h[j]
            if np.isnan(bar_lo) or np.isnan(bar_hi) or bar_hi <= bar_lo:
                continue
            lo_bin = np.searchsorted(bin_edges, bar_lo) - 1
            hi_bin = np.searchsorted(bin_edges, bar_hi) - 1
            if lo_bin < 0:
                lo_bin = 0
            if lo_bin >= vp_bins:
                lo_bin = vp_bins - 1
            if hi_bin < 0:
                hi_bin = 0
            if hi_bin >= vp_bins:
                hi_bin = vp_bins - 1
            n_bins_touched = hi_bin - lo_bin + 1
            if n_bins_touched > 0 and not np.isnan(w_v[j]):
                share = w_v[j] / n_bins_touched
                for b in range(lo_bin, hi_bin + 1):
                    bin_vol[b] += share

        poc_bin = np.argmax(bin_vol)
        vpoc_arr[i] = bin_centers[poc_bin]
        if c_arr[i] > 0:
            vpoc_dist_arr[i] = (c_arr[i] - bin_centers[poc_bin]) / c_arr[i]

        if i > vp_window and not np.isnan(vpoc_arr[i - 1]):
            diff = vpoc_arr[i] - vpoc_arr[i - 1]
            if diff > 0:
                vpoc_mig_arr[i] = 1.0
            elif diff < 0:
                vpoc_mig_arr[i] = -1.0
            else:
                vpoc_mig_arr[i] = 0.0

        total_vol = np.sum(bin_vol)
        if total_vol > 0:
            va_target = 0.70 * total_vol
            va_vol = bin_vol[poc_bin]
            va_lo_idx = poc_bin
            va_hi_idx = poc_bin
            while va_vol < va_target and (va_lo_idx > 0 or va_hi_idx < vp_bins - 1):
                add_lo = bin_vol[va_lo_idx - 1] if va_lo_idx > 0 else 0.0
                add_hi = bin_vol[va_hi_idx + 1] if va_hi_idx < vp_bins - 1 else 0.0
                if add_lo >= add_hi and va_lo_idx > 0:
                    va_lo_idx -= 1
                    va_vol += bin_vol[va_lo_idx]
                elif va_hi_idx < vp_bins - 1:
                    va_hi_idx += 1
                    va_vol += bin_vol[va_hi_idx]
                else:
                    va_lo_idx -= 1
                    va_vol += bin_vol[va_lo_idx]

            va_low = bin_edges[va_lo_idx]
            va_high = bin_edges[va_hi_idx + 1]
            if c_arr[i] < va_low:
                va_pos_arr[i] = -1.0
            elif c_arr[i] > va_high:
                va_pos_arr[i] = 1.0
            else:
                va_pos_arr[i] = 0.0

        # HVN below price
        best_hvn_vol = -1.0
        best_hvn_center = np.nan
        for b in range(vp_bins):
            if bin_centers[b] < c_arr[i] and bin_vol[b] > best_hvn_vol:
                best_hvn_vol = bin_vol[b]
                best_hvn_center = bin_centers[b]
        if best_hvn_vol >= 0 and c_arr[i] > 0:
            hvn_dist_arr[i] = (c_arr[i] - best_hvn_center) / c_arr[i]

        # LVN above price
        best_lvn_vol = np.inf
        best_lvn_center = np.nan
        for b in range(vp_bins):
            if bin_centers[b] > c_arr[i] and bin_vol[b] > 0 and bin_vol[b] < best_lvn_vol:
                best_lvn_vol = bin_vol[b]
                best_lvn_center = bin_centers[b]
        if best_lvn_vol < np.inf and c_arr[i] > 0:
            lvn_dist_arr[i] = (best_lvn_center - c_arr[i]) / c_arr[i]

    return vpoc_arr, vpoc_dist_arr, vpoc_mig_arr, va_pos_arr, hvn_dist_arr, lvn_dist_arr


# ============================================================
# UNIVERSAL ENGINES
# ============================================================
from universal_gematria import (
    gematria, digital_root as gem_dr, gematria_flat,
    CAUTION_TARGETS as GEM_CAUTION, PUMP_TARGETS as GEM_PUMP,
    BTC_ENERGY_TARGETS as GEM_BTC_ENERGY,
    gematria_gpu_batch, digital_root_gpu,
)
from universal_numerology import (
    numerology as num_calc, digital_root as num_dr,
    numerology_flat, date_numerology, date_numerology_flat,
    is_caution as num_is_caution, is_pump as num_is_pump,
    sequence_detect, CAUTION_NUMBERS, PUMP_NUMBERS, BTC_ENERGY_NUMBERS,
    digital_root_vec, is_in_set_vec, price_contains_pattern_vec,
)
from universal_astro import (
    astro_flat, get_bazi, get_tzolkin, get_planetary_hour,
    get_moon_phase, get_western, get_vedic, get_zodiac,
)
from universal_sentiment import sentiment as sent_calc, sentiment_flat, sentiment_gpu_batch
from knn_feature_engine import knn_features_from_ohlcv

# LLM features (lazy import -- only used if enabled)
# Set SKIP_LLM=1 env var to disable (e.g. when API credits are exhausted)
if os.environ.get('SKIP_LLM', '0') == '1':
    _HAS_LLM = False
else:
    try:
        from llm_features import compute_llm_features as _compute_llm_features
        _HAS_LLM = True
    except ImportError:
        _HAS_LLM = False


# ============================================================
# TF-SPECIFIC CONFIGURATION
# ============================================================
TF_CONFIG = {
    '15m': {
        'bucket_seconds': 900,
        'return_bars': [1, 4, 16, 96],        # 15m, 1h, 4h, 1d
        'lag_bars': [1, 4, 16, 48, 96],
        'vol_short': 10,
        'vol_long': 96,
        'vwap_window': 96,
        'fg_lag_bars': [96, 288, 480, 960],
        'macro_roc_short': 96 * 5,
        'macro_roc_long': 96 * 20,
        'corr_window': 96 * 20,
        'hash_roc_bars': 96 * 7,
        'fg_roc_bars': 96 * 5,
        'ema50_slope_bars': 80,
    },
    '1h': {
        'bucket_seconds': 3600,
        'return_bars': [1, 4, 8, 24, 48],     # 1h, 4h, 8h, 1d, 2d
        'lag_bars': [1, 4, 8, 24, 48],
        'vol_short': 10,
        'vol_long': 24,
        'vwap_window': 24,
        'fg_lag_bars': [24, 72, 120, 240],
        'macro_roc_short': 120,                # 5 days
        'macro_roc_long': 480,                 # 20 days
        'corr_window': 480,
        'hash_roc_bars': 168,                  # 7 days
        'fg_roc_bars': 120,
        'ema50_slope_bars': 20,
    },
    '4h': {
        'bucket_seconds': 14400,
        'return_bars': [1, 6, 12, 42],         # 4h, 1d, 2d, 1w
        'lag_bars': [1, 6, 12, 24, 42],
        'vol_short': 6,
        'vol_long': 42,
        'vwap_window': 6,
        'fg_lag_bars': [6, 18, 30, 60],
        'macro_roc_short': 30,
        'macro_roc_long': 120,
        'corr_window': 120,
        'hash_roc_bars': 42,
        'fg_roc_bars': 30,
        'ema50_slope_bars': 10,
    },
    '1d': {
        'bucket_seconds': 86400,
        'return_bars': [1, 3, 7, 14, 30],
        'lag_bars': [1, 3, 7, 14, 30],
        'vol_short': 5,
        'vol_long': 20,
        'vwap_window': 5,
        'fg_lag_bars': [1, 3, 5, 10],
        'macro_roc_short': 5,
        'macro_roc_long': 20,
        'corr_window': 20,
        'hash_roc_bars': 7,
        'fg_roc_bars': 5,
        'ema50_slope_bars': 5,
    },
    '1w': {
        'bucket_seconds': 604800,
        'return_bars': [1, 4, 12],
        'lag_bars': [1, 4, 8],
        'vol_short': 4,
        'vol_long': 12,
        'vwap_window': 4,
        'fg_lag_bars': [1, 2, 4],
        'macro_roc_short': 4,
        'macro_roc_long': 12,
        'corr_window': 12,
        'hash_roc_bars': 4,
        'fg_roc_bars': 4,
        'ema50_slope_bars': 3,
    },
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def compute_rsi(series, period):
    """Wilder RSI calculation."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    # cuDF ewm doesn't support min_periods — use try/except for compat
    _cudf_fallback = False
    try:
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    except (NotImplementedError, TypeError):
        avg_gain = gain.ewm(alpha=1 / period).mean()
        avg_loss = loss.ewm(alpha=1 / period).mean()
        _cudf_fallback = True
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    # When min_periods was dropped (cuDF path), manually NaN-out the first
    # `period` values so output matches pandas semantics.  Works for both
    # pandas Series and cuDF Series (.iloc is supported by both).
    if _cudf_fallback:
        rsi.iloc[:period] = float("nan")
    return rsi


def _digital_root(n):
    """Local digital root for vectorized use."""
    n = abs(int(n))
    if n == 0:
        return 0
    return 1 + (n - 1) % 9


def _rolling_slope_vectorized(y, window=5):
    """Closed-form rolling linear regression slope. No .apply(), no polyfit.
    Uses convolution for O(N) computation instead of O(N*W) polyfit per window."""
    y = np.asarray(y, dtype=np.float64)
    W = window
    x = np.arange(W, dtype=np.float64)
    sum_x = x.sum()
    sum_x2 = (x ** 2).sum()
    denom = W * sum_x2 - sum_x ** 2
    if denom == 0:
        return pd.Series(np.full(len(y), np.nan))
    # Convolve to get rolling sums
    kernel_ones = np.ones(W)
    kernel_x = x[::-1]  # reversed for convolution
    # Replace NaNs with 0 for convolution, track valid
    y_clean = np.where(np.isnan(y), 0, y)
    valid = (~np.isnan(y)).astype(np.float64)
    sum_y = np.convolve(y_clean, kernel_ones, mode='valid')
    sum_xy = np.convolve(y_clean, kernel_x, mode='valid')
    n_valid = np.convolve(valid, kernel_ones, mode='valid')
    slopes = np.where(n_valid >= 2, (W * sum_xy - sum_x * sum_y) / denom, np.nan)
    # Pad front with NaN
    result = np.full(len(y), np.nan)
    result[W - 1:] = slopes
    return pd.Series(result)


def _gem_is_caution(val):
    """Check if a gematria value matches any caution target."""
    return 1 if val in GEM_CAUTION else 0


def _gem_is_pump(val):
    """Check if a gematria value matches any pump target."""
    return 1 if val in GEM_PUMP else 0


def _gem_is_btc_energy(val):
    """Check if a gematria value matches any BTC energy target."""
    return 1 if val in GEM_BTC_ENERGY else 0


def _mode_or_nan(x):
    """Mode aggregation that returns NaN instead of 0 for empty."""
    m = x.mode()
    if len(m) > 0:
        return m.iloc[0]
    return np.nan




# ============================================================
# COMPUTE TA FEATURES
# ============================================================

def compute_ta_features(df: pd.DataFrame, tf_name: str = '1h') -> pd.DataFrame:
    """
    Compute all technical analysis features from OHLCV data.

    Args:
        df: DataFrame with columns: open, high, low, close, volume.
            Optional: quote_volume, trades, taker_buy_volume, taker_buy_quote.
            Must have DatetimeIndex.
        tf_name: Timeframe name for TF-specific configs.

    Returns:
        DataFrame with all TA feature columns added.
    """
    cfg = TF_CONFIG.get(tf_name, TF_CONFIG['1h'])
    _gpu = _is_gpu(df)
    _m = cudf if (_HAS_CUDF and _gpu) else pd
    out = _m.DataFrame(index=df.index)

    c = df['close'].astype(float)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    o = df['open'].astype(float)
    v = df['volume'].astype(float)
    c_vals = _np(c)
    h_vals = _np(h).astype(float)
    l_vals = _np(l).astype(float)

    # --- Moving Averages ---
    for w in [5, 10, 20, 50, 100, 200]:
        out[f'sma_{w}'] = c.rolling(w).mean()
        out[f'ema_{w}'] = c.ewm(span=w, adjust=False).mean()
        out[f'close_vs_sma_{w}'] = (c - out[f'sma_{w}']) / out[f'sma_{w}']
        out[f'close_vs_ema_{w}'] = (c - out[f'ema_{w}']) / out[f'ema_{w}']

    for w in [20, 50, 200]:
        out[f'sma_{w}_slope'] = out[f'sma_{w}'].pct_change(5)

    out['golden_cross'] = ((out['sma_50'] > out['sma_200']) &
                           (out['sma_50'].shift(1) <= out['sma_200'].shift(1))).astype(int)
    out['death_cross'] = ((out['sma_50'] < out['sma_200']) &
                          (out['sma_50'].shift(1) >= out['sma_200'].shift(1))).astype(int)
    out['above_sma200'] = (c > out['sma_200']).astype(int)

    # --- RSI ---
    for p in [7, 14, 21]:
        out[f'rsi_{p}'] = compute_rsi(c, p)
    out['rsi_14_ob'] = (out['rsi_14'] > 70).astype(int)
    out['rsi_14_os'] = (out['rsi_14'] < 30).astype(int)

    # --- Bollinger Bands ---
    mid = c.rolling(20).mean()
    std = c.rolling(20).std()
    out['bb_upper_20'] = mid + 2 * std
    out['bb_lower_20'] = mid - 2 * std
    out['bb_width_20'] = (out['bb_upper_20'] - out['bb_lower_20']) / mid
    out['bb_pctb_20'] = (c - out['bb_lower_20']) / (out['bb_upper_20'] - out['bb_lower_20'])
    # cuDF rolling doesn't support .quantile() — compute on CPU
    _bbw = _np(out['bb_width_20'])
    _bbw_s = pd.Series(_bbw).rolling(500).quantile(0.1).values
    out['bb_squeeze_20'] = (_np(out['bb_width_20']) < _bbw_s).astype(np.int32)

    # --- MACD ---
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    out['macd_line'] = ema12 - ema26
    out['macd_signal'] = out['macd_line'].ewm(span=9, adjust=False).mean()
    out['macd_histogram'] = out['macd_line'] - out['macd_signal']
    out['macd_cross_up'] = ((out['macd_line'] > out['macd_signal']) &
                            (out['macd_line'].shift(1) <= out['macd_signal'].shift(1))).astype(int)
    out['macd_cross_down'] = ((out['macd_line'] < out['macd_signal']) &
                              (out['macd_line'].shift(1) >= out['macd_signal'].shift(1))).astype(int)

    # --- Stochastic ---
    low_min = l.rolling(14).min()
    high_max = h.rolling(14).max()
    out['stoch_k_14'] = 100 * (c - low_min) / (high_max - low_min)
    out['stoch_d_14'] = out['stoch_k_14'].rolling(3).mean()

    # --- ATR ---
    tr = _concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    out['atr_14'] = tr.rolling(14).mean()
    out['atr_14_pct'] = out['atr_14'] / c

    # --- Volume features ---
    out['volume_sma_20'] = v.rolling(20).mean()
    out['volume_ratio'] = v / out['volume_sma_20']
    out['volume_spike'] = (out['volume_ratio'] > 2.0).astype(int)
    if 'taker_buy_volume' in df.columns:
        out['taker_buy_ratio'] = df['taker_buy_volume'].astype(float) / v.replace(0, np.nan)

    # --- Returns (TF-specific) ---
    for p in cfg['return_bars']:
        out[f'return_{p}bar'] = c.pct_change(p)

    # --- Volatility ---
    out[f'volatility_{cfg["vol_short"]}bar'] = out['return_1bar'].rolling(cfg['vol_short']).std() if 'return_1bar' in out.columns else c.pct_change().rolling(cfg['vol_short']).std()
    out[f'volatility_{cfg["vol_long"]}bar'] = out['return_1bar'].rolling(cfg['vol_long']).std() if 'return_1bar' in out.columns else c.pct_change().rolling(cfg['vol_long']).std()
    out['volatility_ratio'] = out[f'volatility_{cfg["vol_short"]}bar'] / out[f'volatility_{cfg["vol_long"]}bar']

    # --- Ichimoku ---
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    out['ichimoku_tenkan'] = tenkan
    out['ichimoku_kijun'] = kijun
    out['ichimoku_senkou_a'] = senkou_a
    out['ichimoku_senkou_b'] = senkou_b
    out['ichimoku_above_cloud'] = ((c > senkou_a) & (c > senkou_b)).astype(int)
    out['ichimoku_below_cloud'] = ((c < senkou_a) & (c < senkou_b)).astype(int)
    out['ichimoku_in_cloud'] = (~out['ichimoku_above_cloud'].astype(bool) &
                                ~out['ichimoku_below_cloud'].astype(bool)).astype(int)
    out['ichimoku_tk_cross'] = ((tenkan > kijun) & (tenkan.shift(1) <= kijun.shift(1))).astype(int)

    # --- Parabolic SAR (Numba JIT) ---
    sar_arr, sar_bull_arr = _sar_loop(h_vals, l_vals, 0.02, 0.02, 0.2)
    out['sar_value'] = sar_arr
    out['sar_bullish'] = sar_bull_arr.astype(int)
    out['sar_flip'] = (out['sar_bullish'] != out['sar_bullish'].shift(1)).astype(int)

    # --- Supertrend (Numba JIT) ---
    atr_st = tr.rolling(10).mean()
    hl2 = (h + l) / 2
    upper_band = hl2 + 3 * atr_st
    lower_band = hl2 - 3 * atr_st
    ub_vals = _np(upper_band).astype(float)
    lb_vals = _np(lower_band).astype(float)
    st_vals, st_dir = _supertrend_loop(c_vals.astype(float), ub_vals, lb_vals)
    out['supertrend'] = st_vals
    out['supertrend_bullish'] = (st_dir == 1).astype(int)
    out['supertrend_flip'] = np.concatenate([[0], np.diff(st_dir) != 0]).astype(int)

    # --- Pivot Points ---
    prev_h = h.shift(1)
    prev_l = l.shift(1)
    prev_c = c.shift(1)
    out['pivot'] = (prev_h + prev_l + prev_c) / 3
    out['pivot_r1'] = 2 * out['pivot'] - prev_l
    out['pivot_s1'] = 2 * out['pivot'] - prev_h
    out['pivot_r2'] = out['pivot'] + (prev_h - prev_l)
    out['pivot_s2'] = out['pivot'] - (prev_h - prev_l)
    out['close_vs_pivot'] = (c - out['pivot']) / out['pivot']

    # --- OBV ---
    obv_dir = np.where(c_vals > np.roll(c_vals, 1), 1,
                       np.where(c_vals < np.roll(c_vals, 1), -1, 0))
    obv_dir[0] = 0
    out['obv'] = (_np(v) * obv_dir).cumsum()
    out['obv_sma_20'] = out['obv'].rolling(20).mean()
    out['obv_trend'] = (out['obv'] > out['obv_sma_20']).astype(int)

    # --- VWAP ---
    vwap_w = cfg['vwap_window']
    cum_vol = v.rolling(vwap_w).sum()
    cum_pv = (c * v).rolling(vwap_w).sum()
    out[f'vwap_{vwap_w}'] = cum_pv / cum_vol
    out['close_vs_vwap'] = (c - out[f'vwap_{vwap_w}']) / out[f'vwap_{vwap_w}']

    # --- Candle Patterns ---
    body = (c - o).abs()
    full_range = (h - l).replace(0, np.nan)
    out['body_pct'] = body / full_range
    out['upper_wick'] = (h - _concat([c, o], axis=1).max(axis=1)) / full_range
    out['lower_wick'] = (_concat([c, o], axis=1).min(axis=1) - l) / full_range
    out['doji'] = (out['body_pct'] < 0.1).astype(int)
    out['hammer'] = ((out['lower_wick'] > 0.6) & (out['body_pct'] < 0.3) & (c > o)).astype(int)
    out['shooting_star'] = ((out['upper_wick'] > 0.6) & (out['body_pct'] < 0.3) & (c < o)).astype(int)
    bull_engulf = (c > o) & (c.shift(1) < o.shift(1)) & (c > o.shift(1)) & (o < c.shift(1))
    bear_engulf = (c < o) & (c.shift(1) > o.shift(1)) & (o > c.shift(1)) & (c < o.shift(1))
    out['bull_engulfing'] = bull_engulf.astype(int)
    out['bear_engulfing'] = bear_engulf.astype(int)

    # --- Consecutive red/green ---
    # --- Consecutive red/green (Numba JIT) ---
    is_green = _np((c > o).astype(int)).astype(np.int64)
    is_red = _np((c < o).astype(int)).astype(np.int64)
    out['consec_green'] = _consec_count_loop(is_green)
    out['consec_red'] = _consec_count_loop(is_red)

    # --- ADX ---
    _h_diff = _np(h.diff())
    _l_diff = _np(l.diff())
    plus_dm = np.where((_h_diff > 0) & (_h_diff > -_l_diff), _h_diff, 0)
    minus_dm = np.where((-_l_diff > 0) & (-_l_diff > _h_diff), -_l_diff, 0)
    plus_di = 100 * _m.Series(plus_dm, index=df.index).ewm(span=14).mean() / out['atr_14']
    minus_di = 100 * _m.Series(minus_dm, index=df.index).ewm(span=14).mean() / out['atr_14']
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    out['adx_14'] = dx.ewm(span=14).mean()

    # --- CCI ---
    tp = (h + l + c) / 3
    out['cci_20'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())

    # --- Williams %R ---
    out['williams_r_14'] = -100 * (h.rolling(14).max() - c) / (h.rolling(14).max() - l.rolling(14).min())

    # --- MFI ---
    tp_series = (h + l + c) / 3
    rmf = tp_series * v
    pos_mf = rmf.where(tp_series > tp_series.shift(1), 0).rolling(14).sum()
    neg_mf = rmf.where(tp_series < tp_series.shift(1), 0).rolling(14).sum()
    out['mfi_14'] = 100 - (100 / (1 + pos_mf / neg_mf.replace(0, np.nan)))

    # --- CMF ---
    clv = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
    out['cmf_20'] = (clv * v).rolling(20).sum() / v.rolling(20).sum()

    # --- Keltner Channels ---
    kelt_mid = c.ewm(span=20).mean()
    kelt_atr = tr.rolling(10).mean()
    out['keltner_upper'] = kelt_mid + 2 * kelt_atr
    out['keltner_lower'] = kelt_mid - 2 * kelt_atr

    # --- Donchian Channels ---
    out['donchian_upper'] = h.rolling(20).max()
    out['donchian_lower'] = l.rolling(20).min()

    # --- Wyckoff Full State Machine ---
    vol_ma = v.rolling(20).mean()
    atr_arr = _np(out['atr_14'])
    n = len(df)
    c_arr = c_vals.astype(float)
    h_arr = h_vals.astype(float)
    l_arr = l_vals.astype(float)
    o_arr = _np(o).astype(float)
    v_arr = _np(v).astype(float)
    vol_ma_arr = _np(vol_ma).astype(float)

    # Pre-compute rolling lowest low (30-bar) for swing low detection
    roll_low_30 = _np(l.rolling(30, min_periods=30).min())
    bar_spread = h_arr - l_arr

    # Wyckoff state machine (Numba JIT)
    (wk_phase, wk_in_range, wk_range_pos, wk_sc_bars_ago,
     wk_spring, wk_upthrust, wk_sos, wk_sow,
     wk_lps_count, wk_lpsy_count, wk_dir_score,
     wk_range_width, wk_bars_in_range) = _wyckoff_loop(
        c_arr, h_arr, l_arr, o_arr, v_arr, vol_ma_arr, atr_arr,
        roll_low_30, bar_spread, n)

    # --- Effort vs Result (vectorized rolling 10-bar) ---
    up_bar = (c_arr > o_arr).astype(np.float64)
    wide_spread = (bar_spread > np.nanmedian(bar_spread[bar_spread > 0])).astype(np.float64)
    high_vol = np.zeros(n)
    valid_vol = ~np.isnan(vol_ma_arr) & (vol_ma_arr > 0)
    high_vol[valid_vol] = (v_arr[valid_vol] > vol_ma_arr[valid_vol]).astype(np.float64)
    low_vol = np.zeros(n)
    low_vol[valid_vol] = (v_arr[valid_vol] < vol_ma_arr[valid_vol]).astype(np.float64)
    # +1 for bullish effort (up bar + high vol + wide spread), -1 for bearish (down bar + low vol + narrow)
    evr_signal = (up_bar * high_vol * wide_spread) - ((1 - up_bar) * low_vol * (1 - wide_spread))
    evr_sum = _m.Series(evr_signal, index=df.index).rolling(10, min_periods=5).sum()
    evr_max = evr_sum.abs().rolling(50, min_periods=10).max().replace(0, 1)
    wk_evr = _np((evr_sum / evr_max).clip(-1, 1))

    # --- Volume trend: declining vol on dips = +1, declining on rallies = -1 ---
    price_down = (c_arr < np.roll(c_arr, 1)).astype(np.float64)
    price_down[0] = 0
    price_up = (c_arr > np.roll(c_arr, 1)).astype(np.float64)
    price_up[0] = 0
    vol_declining = np.zeros(n)
    vol_declining[1:] = (v_arr[1:] < v_arr[:-1]).astype(np.float64)
    vol_trend_raw = (price_down * vol_declining) - (price_up * vol_declining)
    wk_vol_trend_s = _m.Series(vol_trend_raw, index=df.index).rolling(10, min_periods=5).mean()
    wk_vol_trend_arr = np.sign(_np(wk_vol_trend_s))

    # --- Directional score (0-100) — vectorized ---
    wk_dir_score = np.full(n, 50.0)
    wk_dir_score += wk_spring.astype(np.float64) * 20
    wk_dir_score -= wk_upthrust.astype(np.float64) * 20
    wk_dir_score += wk_sos.astype(np.float64) * 15
    wk_dir_score -= wk_sow.astype(np.float64) * 15
    wk_dir_score += np.minimum(wk_lps_count, 4).astype(np.float64) * 5
    wk_dir_score -= np.minimum(wk_lpsy_count, 4).astype(np.float64) * 5
    _evr_valid = ~np.isnan(wk_evr)
    wk_dir_score[_evr_valid] += wk_evr[_evr_valid] * 10
    wk_dir_score = np.clip(wk_dir_score, 0, 100)

    out['wyckoff_phase'] = wk_phase
    out['wyckoff_in_range'] = wk_in_range
    out['wyckoff_range_position'] = wk_range_pos
    out['wyckoff_sc_bars_ago'] = wk_sc_bars_ago
    out['wyckoff_spring'] = wk_spring
    out['wyckoff_upthrust'] = wk_upthrust
    out['wyckoff_sos'] = wk_sos
    out['wyckoff_sow'] = wk_sow
    out['wyckoff_lps_count'] = wk_lps_count
    out['wyckoff_lpsy_count'] = wk_lpsy_count
    out['wyckoff_effort_vs_result'] = wk_evr
    out['wyckoff_directional_score'] = wk_dir_score
    out['wyckoff_volume_trend'] = wk_vol_trend_arr
    out['wyckoff_range_width'] = wk_range_width
    out['wyckoff_bars_in_range'] = wk_bars_in_range

    # --- Volume Profile / POC Migration (Numba JIT) ---
    vp_window = 50 if cfg['bucket_seconds'] < 3600 else 20
    vp_bins = 20
    vpoc_arr, vpoc_dist_arr, vpoc_mig_arr, va_pos_arr, hvn_dist_arr, lvn_dist_arr = \
        _volume_profile_loop(h_arr, l_arr, c_arr, v_arr, n, vp_window, vp_bins)
    out['vpoc'] = vpoc_arr
    out['vpoc_distance'] = vpoc_dist_arr
    out['vpoc_migration'] = vpoc_mig_arr
    out['value_area_position'] = va_pos_arr
    out['hvn_distance'] = hvn_dist_arr
    out['lvn_distance'] = lvn_dist_arr

    # --- ICT / Smart Money Concepts ---
    # Fair Value Gaps (FVG) — vectorized (no for-loop)
    fvg_bull = np.zeros(n, dtype=np.int8)
    fvg_bear = np.zeros(n, dtype=np.int8)
    if n > 2:
        fvg_bull[2:] = (h_arr[:-2] < l_arr[2:]).astype(np.int8)
        fvg_bear[2:] = (l_arr[:-2] > h_arr[2:]).astype(np.int8)
    out['fvg_bullish'] = fvg_bull
    out['fvg_bearish'] = fvg_bear

    # FVG nearest distance + BOS + liquidity sweep + order block — all @njit compiled
    fvg_nearest_dist, bos_dir, liq_sweep, ob_dist = _ict_smc_loop(
        h_arr, l_arr, c_arr, o_arr, fvg_bull, fvg_bear, n)
    out['fvg_nearest_distance'] = fvg_nearest_dist
    out['bos_direction'] = bos_dir
    out['liquidity_sweep'] = liq_sweep
    out['order_block_distance'] = ob_dist

    # --- Anchored VWAP (Numba JIT) ---
    typical_price = (h_arr + l_arr + c_arr) / 3.0
    avwap_low_arr, avwap_high_arr, avwap_pos_arr = _avwap_loop(
        h_arr, l_arr, c_arr, v_arr, typical_price, n, 50)
    out['avwap_from_swing_low'] = avwap_low_arr
    out['avwap_from_swing_high'] = avwap_high_arr
    out['avwap_position'] = avwap_pos_arr

    # --- Elliott Zigzag (Numba JIT) ---
    zigzag, direction = _elliott_zigzag_loop(c_vals.astype(np.float64), 0.03)
    out['elliott_zigzag'] = zigzag
    out['elliott_wave_dir'] = direction

    # --- Gann Square of 9 (Numba JIT) ---
    gann_level, gann_dist = _gann_sq9_vec(c_vals.astype(np.float64))
    out['gann_sq9_level'] = gann_level
    out['gann_sq9_distance'] = gann_dist

    # --- Consensio ---
    ma_list = [5, 10, 20, 50, 100, 200]
    consensio_score = _m.Series(0, index=df.index, dtype=float)
    for i_m in range(len(ma_list)):
        for j_m in range(i_m + 1, len(ma_list)):
            consensio_score += (out[f'sma_{ma_list[i_m]}'] > out[f'sma_{ma_list[j_m]}']).astype(int)
    max_pairs = 15
    out['consensio_score'] = (2 * consensio_score - max_pairs)
    out['consensio_green_wave'] = (out['consensio_score'] == max_pairs).astype(int)
    out['consensio_red_wave'] = (out['consensio_score'] == -max_pairs).astype(int)

    # --- Sacred Geometry ---
    PHI = 1.618033988749895
    rolling_low = l.rolling(200).min()
    rolling_high = h.rolling(200).max()
    out['golden_ratio_ext'] = rolling_low + (rolling_high - rolling_low) * PHI
    out['golden_ratio_dist'] = (c - out['golden_ratio_ext']).abs() / c
    out['fib_21_from_low'] = rolling_low * 1.21
    out['fib_13_from_high'] = rolling_high * 0.87
    out['near_fib_21'] = ((c - out['fib_21_from_low']).abs() / c < 0.02).astype(int)
    out['near_fib_13'] = ((c - out['fib_13_from_high']).abs() / c < 0.02).astype(int)

    # --- Interaction features ---
    def _col_or(df, col, default=0):
        """Like .get() but works with both pandas and cuDF."""
        return df[col] if col in df.columns else default
    out['rsi_x_bbpctb'] = _col_or(out, 'rsi_14', 0) * _col_or(out, 'bb_pctb_20', 0)
    out['volume_x_atr'] = _col_or(out, 'volume_ratio', 0) * _col_or(out, 'atr_14_pct', 0)
    out['consec_red_x_bb_os'] = out['consec_red'] * (_col_or(out, 'bb_pctb_20', 0.5) < 0).astype(int)

    price_low_w = max(cfg['vol_long'] * 20, 480)
    price_low = (c == c.rolling(price_low_w).min()).astype(int)
    rsi_low = (out['rsi_14'] == out['rsi_14'].rolling(price_low_w).min()).astype(int)
    out['rsi_bullish_div'] = (price_low & ~rsi_low.astype(bool)).astype(int)

    # --- Drawdown ---
    rolling_max_price = c.cummax()  # cuDF-compatible (no .expanding())
    out['current_dd_depth'] = (c - rolling_max_price) / rolling_max_price
    out['dd_from_ath'] = out['current_dd_depth']

    # --- Lagged features (TF-specific) ---
    for feat in ['rsi_14', 'bb_pctb_20', 'volume_ratio', 'macd_histogram']:
        if feat in out.columns:
            for lag in cfg['lag_bars']:
                out[f'{feat}_lag{lag}'] = out[feat].shift(lag)

    # --- Meta features ---
    bullish_cols = ['ichimoku_above_cloud', 'supertrend_bullish', 'sar_bullish',
                    'above_sma200', 'obv_trend', 'golden_cross', 'macd_cross_up',
                    'bull_engulfing', 'hammer', 'consensio_green_wave']
    bearish_cols = ['ichimoku_below_cloud', 'death_cross', 'macd_cross_down',
                    'bear_engulfing', 'shooting_star', 'consensio_red_wave', 'rsi_14_ob']
    existing_bull = [c2 for c2 in bullish_cols if c2 in out.columns]
    existing_bear = [c2 for c2 in bearish_cols if c2 in out.columns]
    out['num_bullish_signals'] = out[existing_bull].sum(axis=1)
    out['num_bearish_signals'] = out[existing_bear].sum(axis=1)
    out['signal_agreement'] = out['num_bullish_signals'] - out['num_bearish_signals']

    return out.to_pandas() if hasattr(out, 'to_pandas') else out


# ============================================================
# COMPUTE TIME FEATURES
# ============================================================

def compute_time_features(df: pd.DataFrame, tf_name: str = '1h') -> pd.DataFrame:
    """
    Compute time-based features from the DatetimeIndex.
    """
    _gpu = _is_gpu(df)
    _m = cudf if (_HAS_CUDF and _gpu) else pd
    # cuDF DatetimeIndex doesn't support .isin() or .is_month_end etc. — extract CPU index only
    if _gpu:
        _idx_cpu = pd.DatetimeIndex(_np(df.index))
    else:
        _idx_cpu = df.index
    out = pd.DataFrame(index=_idx_cpu)
    idx = _idx_cpu

    hour_utc = idx.hour
    doy = idx.dayofyear
    dow = idx.dayofweek
    month = idx.month
    day_of_month = idx.day

    # Cyclical encoding
    out['hour_sin'] = np.sin(2 * np.pi * hour_utc / 24)
    out['hour_cos'] = np.cos(2 * np.pi * hour_utc / 24)
    out['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    out['dow_cos'] = np.cos(2 * np.pi * dow / 7)
    out['month_sin'] = np.sin(2 * np.pi * month / 12)
    out['month_cos'] = np.cos(2 * np.pi * month / 12)
    out['doy_sin'] = np.sin(2 * np.pi * doy / 365)
    out['doy_cos'] = np.cos(2 * np.pi * doy / 365)

    # Key hours for crypto (only meaningful for sub-daily TFs)
    if tf_name in ('15m', '1h', '4h'):
        for hr in [0, 4, 6, 8, 12, 16, 20]:
            out[f'is_hour_{hr}'] = (hour_utc == hr).astype(int)

        # Session flags
        out['is_asia_session'] = hour_utc.isin([0, 1, 2, 3, 4, 5, 6, 7]).astype(int)
        out['is_london_session'] = hour_utc.isin([8, 9, 10, 11, 12, 13, 14, 15]).astype(int)
        out['is_ny_session'] = hour_utc.isin([14, 15, 16, 17, 18, 19, 20, 21]).astype(int)
        out['is_london_ny_overlap'] = hour_utc.isin([14, 15, 16]).astype(int)

    # Calendar features
    out['day_of_year'] = doy
    out['day_of_month'] = day_of_month
    out['month'] = month
    out['day_of_week'] = dow

    out['is_month_end'] = idx.is_month_end.astype(int)
    out['is_quarter_end'] = idx.is_quarter_end.astype(int)
    out['is_monday'] = (dow == 0).astype(int)
    out['is_friday'] = (dow == 4).astype(int)
    out['is_weekend'] = (dow >= 5).astype(int)

    return out.to_pandas() if hasattr(out, 'to_pandas') else out


# ============================================================
# COMPUTE NUMEROLOGY FEATURES
# ============================================================

def compute_numerology_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute numerology features from OHLCV data + DatetimeIndex.
    """
    _gpu = _is_gpu(df)
    # Numerology uses .str ops, .isin(), .map() — extract CPU index + values, avoid full _to_cpu()
    if _gpu:
        idx = pd.DatetimeIndex(_np(df.index))
    else:
        idx = df.index
    out = pd.DataFrame(index=idx)
    c_vals = _np(df['close']).astype(np.float64)
    c = pd.Series(c_vals, index=idx)
    v_vals = _np(df['volume']).astype(np.float64)
    v = pd.Series(v_vals, index=idx)

    doy = idx.dayofyear
    day_of_month = idx.day

    # Day flags
    out['day_13'] = (day_of_month == 13).astype(int)
    out['day_21'] = (day_of_month == 21).astype(int)
    out['day_6'] = (day_of_month == 6).astype(int)
    out['day_27'] = (day_of_month == 27).astype(int)

    # DOY flags
    out['is_113'] = (doy == 113).astype(int)
    out['is_93'] = (doy == 93).astype(int)
    out['is_39'] = (doy == 39).astype(int)
    out['is_223'] = (doy == 223).astype(int)
    out['is_322'] = (doy == 322).astype(int)
    out['is_37'] = (doy == 37).astype(int)
    out['is_73'] = (doy == 73).astype(int)
    out['is_17'] = (doy == 17).astype(int)
    out['is_19'] = (doy == 19).astype(int)
    out['btc_213'] = (doy == 213).astype(int)

    # Mirror DOY numbers (inverse energy of key numbers)
    out['is_72'] = (doy == 72).astype(int)    # mirror of 27 (pump) = ANTI-PUMP
    out['is_71'] = (doy == 71).astype(int)    # mirror of 17 (kill) = recovery
    out['is_91'] = (doy == 91).astype(int)    # mirror of 19 (surrender) = resistance
    out['is_311'] = (doy == 311).astype(int)  # mirror of 113 (bottom buy) = confirmation
    out['is_312'] = (doy == 312).astype(int)  # mirror of 213 (BTC energy) = cycle
    out['is_132'] = (doy == 132).astype(int)  # mirror of 231 (BTC mirror) = cycle
    # DOY 1-365 flags (for systematic cross expansion)
    for d in range(1, 366):
        out[f'doy_{d}'] = (doy == d).astype(np.int8)

    # Date DR — vectorized (no list comp)
    _date_digits = (idx.year * 10000 + idx.month * 100 + idx.day).values.astype(np.int64)
    # Sum digits of YYYYMMDD, then digital root
    _digit_sums = np.zeros(len(idx), dtype=np.int64)
    _temp = _date_digits.copy()
    while np.any(_temp > 0):
        _digit_sums += _temp % 10
        _temp //= 10
    out['date_dr'] = digital_root_vec(_digit_sums)

    # Price numerology — vectorized (no list comp)
    _c_int = np.where(np.isnan(c_vals), 0, np.abs(c_vals).astype(np.int64))
    out['digital_root_price'] = np.where(np.isnan(c_vals), np.nan, digital_root_vec(_c_int))
    out['price_dr'] = out['digital_root_price']

    # Price contains patterns — vectorized via string ops
    _c_not_nan = ~np.isnan(c_vals)
    _c_str = pd.Series(np.where(_c_not_nan, np.abs(c_vals).astype(np.int64), 0), index=idx).astype(int).astype(str)
    out['price_contains_322'] = np.where(_c_not_nan, _c_str.str.contains('322', regex=False).astype(np.int32), np.nan)
    out['price_contains_113'] = np.where(_c_not_nan, _c_str.str.contains('113', regex=False).astype(np.int32), np.nan)
    out['price_contains_93'] = np.where(_c_not_nan, _c_str.str.contains('93', regex=False).astype(np.int32), np.nan)
    out['price_contains_213'] = np.where(_c_not_nan, _c_str.str.contains('213', regex=False).astype(np.int32), np.nan)

    # Price master number — vectorized (check last 2 digits)
    _c_mod100 = np.where(np.isnan(c_vals), -1, np.abs(c_vals).astype(np.int64) % 100)
    out['price_is_master'] = np.isin(_c_mod100, [11, 22, 33]).astype(np.int32)

    # Volume DR — vectorized
    _v_int = np.where(np.isnan(v.values) | (v.values <= 0), 0, np.abs(v.values).astype(np.int64))
    out['volume_dr'] = np.where(_v_int == 0, 0, digital_root_vec(_v_int))

    # Special dates
    special_doys = {22, 33, 44, 55, 66, 77, 88, 99, 111, 222, 333}
    out['pump_date'] = ((out['date_dr'] == 9) | (day_of_month == 22) |
                        pd.Series(doy, index=idx).isin(special_doys)).astype(int)

    out['price_dr_6'] = (out['digital_root_price'] == 6).astype(int)
    out['vortex_369'] = out['digital_root_price'].isin([3, 6, 9]).astype(int)

    # Kabbalah / Shemitah
    out['sephirah'] = out['digital_root_price']
    # Shemitah spans from ~Sep of start year to ~Sep of end year
    # 2014-09 to 2015-09, 2021-09 to 2022-09, 2028-09 to 2029-09
    _yr = idx.year
    _mo = idx.month
    out['shemitah_year'] = (
        ((_yr == 2014) & (_mo >= 9)) | ((_yr == 2015) & (_mo < 10)) |
        ((_yr == 2021) & (_mo >= 9)) | ((_yr == 2022) & (_mo < 10)) |
        ((_yr == 2028) & (_mo >= 9)) | ((_yr == 2029) & (_mo < 10))
    ).astype(float)
    jubilee_ref = 2017
    _jub_raw = ((idx.year - jubilee_ref) % 50).astype(float).values
    out['jubilee_proximity'] = np.minimum(_jub_raw, 50 - _jub_raw) / 25

    # ================================================================
    # NEW FEATURES FROM HEARTBEAT DIRECTIONAL ANALYSIS
    # ================================================================

    dow = idx.dayofweek  # 0=Mon ... 6=Sun

    # --- Day-of-week flags (4) ---
    out['is_tuesday'] = (dow == 1).astype(int)
    out['is_wednesday'] = (dow == 2).astype(int)
    out['is_thursday'] = (dow == 3).astype(int)
    out['is_sunday'] = (dow == 6).astype(int)

    # --- Day-of-month flags (8) ---
    out['is_day_5'] = (day_of_month == 5).astype(int)
    out['is_day_7'] = (day_of_month == 7).astype(int)
    out['is_day_11'] = (day_of_month == 11).astype(int)
    out['is_day_12'] = (day_of_month == 12).astype(int)
    out['is_day_15'] = (day_of_month == 15).astype(int)
    out['is_day_17'] = (day_of_month == 17).astype(int)
    out['is_day_20'] = (day_of_month == 20).astype(int)
    out['is_day_28'] = (day_of_month == 28).astype(int)

    # --- Missing matrixology numbers (4) ---
    out['is_48'] = (doy == 48).astype(int)
    out['is_84'] = (doy == 84).astype(int)
    out['is_43'] = (doy == 43).astype(int)
    out['is_34'] = (doy == 34).astype(int)
    # Also is_11 (day-of-month 11 is covered above; DOY 11 for completeness)
    out['is_11_doy'] = (doy == 11).astype(int)

    # --- Price digital root expansion (3) ---
    out['price_dr_3'] = (out['digital_root_price'] == 3).astype(int)
    out['price_dr_7'] = (out['digital_root_price'] == 7).astype(int)
    out['price_dr_9'] = (out['digital_root_price'] == 9).astype(int)

    # --- Kabbalah sephiroth date energy (1 categorical, maps DR to tree of life) ---
    # DR 1-9 each maps to a sephirah: 1=Kether, 2=Chokmah, 3=Binah, 4=Chesed,
    # 5=Geburah(severity), 6=Tiphareth, 7=Netzach(victory), 8=Hod, 9=Yesod
    month_day_sum = pd.Series(idx.month + idx.day, index=idx)
    out['date_dr_sephirah'] = digital_root_vec(month_day_sum.values)

    # --- Date sum targets (4) ---
    out['month_day_sum_7'] = (month_day_sum == 7).astype(int)
    out['month_day_sum_11'] = (month_day_sum == 11).astype(int)
    out['month_day_sum_13'] = (month_day_sum == 13).astype(int)
    out['month_day_sum_22'] = (month_day_sum == 22).astype(int)

    # --- Week/year numerology (3) ---
    week_of_year = pd.Series(idx.isocalendar().week.values, index=idx, dtype=int)
    out['week_master'] = week_of_year.isin([11, 22, 33]).astype(int)
    _years = idx.year.values
    _is_leap = ((_years % 4 == 0) & ((_years % 100 != 0) | (_years % 400 == 0)))
    total_days_in_year = pd.Series(np.where(_is_leap, 366, 365).astype(float), index=idx)
    pct_year = pd.Series(doy, index=idx).astype(float) / total_days_in_year * 100
    out['pct_year_93'] = ((pct_year >= 92) & (pct_year <= 94)).astype(int)
    out['doy_div_9'] = (pd.Series(doy, index=idx) % 9 == 0).astype(int)

    # --- Planetary day x DR combos (3) ---
    # Planetary rulers: Mon=Moon(2), Tue=Mars(9), Wed=Mercury(5), Thu=Jupiter(3),
    #                   Fri=Venus(6), Sat=Saturn(8), Sun=Sun(1)
    _planetary_lut = np.array([2, 9, 5, 3, 6, 8, 1], dtype=np.int32)  # indexed by dow 0-6
    planet_num = pd.Series(_planetary_lut[dow.values], index=idx)
    date_dr_series = out['date_dr'].astype(int)

    # Categorical combo: dow * 10 + date_dr (encodes 7x9=63 combos)
    out['planetary_day_dr_combo'] = dow * 10 + date_dr_series

    # Resonance flag: 1=match (planet's number == date DR), -1=oppose, 0=neutral
    # Oppose lookup: planet_num values 1-9 → opposed number (0 unused, -1 default)
    _oppose_lut = np.array([-1, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=np.int32)  # idx 0-9
    oppose_num = pd.Series(_oppose_lut[planet_num.values], index=idx)
    resonance = pd.Series(np.zeros(len(idx)), index=idx)
    resonance[planet_num.values == date_dr_series.values] = 1
    resonance[oppose_num.values == date_dr_series.values] = -1
    out['planetary_day_resonance'] = resonance.astype(int)

    # --- Fibonacci day (1) ---
    fib_days = {1, 2, 3, 5, 8, 13, 21}
    out['is_fibonacci_day'] = pd.Series(day_of_month, index=idx).isin(fib_days).astype(int)

    # --- Price near round $1000s (1) ---
    out['price_near_round_1000'] = (c % 1000 < 50).astype(int) | ((1000 - c % 1000) < 50).astype(int)

    # --- Sequential date flag (1) ---
    month_s = pd.Series(idx.month, index=idx)
    day_s = pd.Series(day_of_month, index=idx)
    out['is_sequential_date'] = (day_s == month_s + 1).astype(int)

    # --- Year DR (1) ---
    out['year_dr'] = pd.Series(digital_root_vec(idx.year.values.astype(np.int64)), index=idx)

    # --- DOY power of 2 (1) ---
    powers_of_2 = {1, 2, 4, 8, 16, 32, 64, 128, 256}
    out['doy_power_of_2'] = pd.Series(doy, index=idx).isin(powers_of_2).astype(int)

    # --- Day is prime (1) ---
    prime_days = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31}
    out['day_is_prime'] = pd.Series(day_of_month, index=idx).isin(prime_days).astype(int)

    # --- DR Mirror features ---
    # Mirror match: date DR mirrors price DR (1<->8, 2<->7, 3<->6, 4<->5, 9<->9)
    _dr_mirror = {1:8, 2:7, 3:6, 4:5, 5:4, 6:3, 7:2, 8:1, 9:9}
    date_dr_s = out['date_dr'].astype(int)
    price_dr_s = out['digital_root_price']
    mirror_price_dr = price_dr_s.map(_dr_mirror)
    out['dr_mirror_match'] = (date_dr_s == mirror_price_dr).astype(int)
    out['dr_completion_9'] = ((date_dr_s + price_dr_s) == 9).astype(int)

    # --- Price mirror contains ---
    # Price mirror contains — vectorized string ops (no list comp)
    out['price_contains_72'] = _c_str.str.contains('72', regex=False).astype(np.int32)
    out['price_contains_71'] = _c_str.str.contains('71', regex=False).astype(np.int32)
    out['price_contains_91'] = _c_str.str.contains('91', regex=False).astype(np.int32)
    out['price_contains_39'] = _c_str.str.contains('39', regex=False).astype(np.int32)
    out['price_contains_37'] = _c_str.str.contains('37', regex=False).astype(np.int32)
    out['price_contains_73'] = _c_str.str.contains('73', regex=False).astype(np.int32)

    # --- Angel numbers (repeating digit patterns in price) ---
    # Uses _c_str (integer price as string), respects NaN via _c_not_nan mask
    _angel_patterns = ['111', '222', '333', '444', '555', '666', '777', '888', '999']
    _angel_cols = {}
    for pat in _angel_patterns:
        _angel_cols[f'price_has_{pat}'] = np.where(
            _c_not_nan, _c_str.str.contains(pat, regex=False).astype(np.int32), np.nan)
    # Angel count = sum of all pattern hits per bar
    _angel_sum = np.zeros(len(idx), dtype=np.float64)
    for col_name, col_vals in _angel_cols.items():
        out[col_name] = col_vals
        _angel_sum = np.where(np.isnan(col_vals), np.nan,
                              np.where(np.isnan(_angel_sum), np.nan, _angel_sum + col_vals))
    out['price_angel_count'] = _angel_sum

    # --- Price palindrome detection ---
    _price_reversed = _c_str.str[::-1]
    out['price_palindrome'] = np.where(
        _c_not_nan, (_c_str == _price_reversed).astype(np.int32), np.nan)

    return out.to_pandas() if hasattr(out, 'to_pandas') else out


# ============================================================
# NUMEROLOGY EXPANSION — Lo Shu, Angel Numbers, Haramein, Pythagorean Challenges
# ============================================================

def compute_numerology_expansion_features(df: pd.DataFrame, tf_name: str = '1d') -> pd.DataFrame:
    """
    Numerology expansion features:
      1. Lo Shu Magic Square — grid position, type, 3-bar line completion
      2. Angel number proximity — 777/888 harmonics
      3. Base-10 vs Base-12 tension
      4. Haramein 64-grid harmonic (Vector Equilibrium)
      5. Pythagorean challenge numbers (vectorized)
    """
    from universal_numerology import LO_SHU_LINES, LO_SHU_ROW, LO_SHU_COL

    _gpu = _is_gpu(df)
    if _gpu:
        idx = pd.DatetimeIndex(_np(df.index))
    else:
        idx = df.index
    out = pd.DataFrame(index=idx)
    c_vals = _np(df['close']).astype(np.float64)
    c_not_nan = ~np.isnan(c_vals)

    # Price digital root (recompute vectorized)
    _c_int = np.where(c_not_nan, (np.abs(c_vals) * 100).astype(np.int64), 0)
    _c_int = np.clip(_c_int, 1, None)
    price_dr = np.where(c_not_nan, (((_c_int - 1) % 9) + 1).astype(np.int32), 0)

    # 1. Lo Shu grid position
    _dr_series = pd.Series(price_dr, index=idx)
    out['loshu_row'] = np.where(c_not_nan, _dr_series.map(LO_SHU_ROW).values.astype(np.float64), np.nan)
    out['loshu_col'] = np.where(c_not_nan, _dr_series.map(LO_SHU_COL).values.astype(np.float64), np.nan)
    out['loshu_is_center'] = np.where(c_not_nan, (price_dr == 5).astype(np.float64), np.nan)
    out['loshu_is_corner'] = np.where(c_not_nan, np.isin(price_dr, [2, 4, 6, 8]).astype(np.float64), np.nan)
    out['loshu_is_edge'] = np.where(c_not_nan, np.isin(price_dr, [1, 3, 7, 9]).astype(np.float64), np.nan)

    # 2. Lo Shu line completion (rolling 3-bar DR)
    dr_float = pd.Series(np.where(c_not_nan, price_dr.astype(np.float64), np.nan), index=idx)

    def _check_loshu_line(w):
        if np.isnan(w).any():
            return np.nan
        s = frozenset(w.astype(int))
        return 1.0 if s in LO_SHU_LINES else 0.0

    out['loshu_line_complete'] = dr_float.rolling(3, min_periods=3).apply(
        _check_loshu_line, raw=True
    )

    # 3. Angel number proximity (777, 888 harmonics)
    close_int = np.where(c_not_nan, np.abs(c_vals).astype(np.int64), 0)
    mod_777 = np.where(c_not_nan, (close_int % 777).astype(np.float64), np.nan)
    mod_888 = np.where(c_not_nan, (close_int % 888).astype(np.float64), np.nan)
    out['price_mod_777'] = mod_777
    out['price_near_777'] = np.where(c_not_nan,
        ((close_int % 777 < 5) | (777 - close_int % 777 < 5)).astype(np.float64), np.nan)
    out['price_mod_888'] = mod_888
    out['price_near_888'] = np.where(c_not_nan,
        ((close_int % 888 < 5) | (888 - close_int % 888 < 5)).astype(np.float64), np.nan)

    # 4. Base-10 vs Base-12 tension
    doy = idx.dayofyear.values
    b12_price = np.where(c_not_nan, (close_int % 12 == 0).astype(np.float64), np.nan)
    b12_doy = (doy % 12 == 0).astype(np.float64)
    b10_price = np.where(c_not_nan, (close_int % 10 == 0).astype(np.float64), np.nan)
    b10_doy = (doy % 10 == 0).astype(np.float64)
    out['base12_resonance'] = np.where(c_not_nan, np.maximum(b12_price, b12_doy), b12_doy)
    out['base10_resonance'] = np.where(c_not_nan, np.maximum(b10_price, b10_doy), b10_doy)
    _b12_bool = np.where(c_not_nan, b12_price, 0).astype(bool) | b12_doy.astype(bool)
    _b10_bool = np.where(c_not_nan, b10_price, 0).astype(bool) | b10_doy.astype(bool)
    out['base_tension'] = (_b12_bool ^ _b10_bool).astype(np.float64)

    # 5. Haramein 64-grid harmonic (Vector Equilibrium)
    out['haramein_64_phase'] = (doy % 64).astype(np.float64)
    out['haramein_64_node'] = ((doy % 64 == 0) | (doy % 32 == 0)).astype(np.float64)

    # 6. Pythagorean challenge numbers (vectorized)
    m = idx.month.values.astype(np.int64)
    d = idx.day.values.astype(np.int64)
    _y_raw = idx.year.values.astype(np.int64)
    _y_temp = _y_raw.copy()
    _y_digit_sums = np.zeros(len(idx), dtype=np.int64)
    while np.any(_y_temp > 0):
        _y_digit_sums += _y_temp % 10
        _y_temp //= 10
    m_dr = ((m - 1) % 9 + 1).astype(np.float64)
    d_dr = ((d - 1) % 9 + 1).astype(np.float64)
    y_dr = ((_y_digit_sums - 1) % 9 + 1).astype(np.float64)
    ch1 = np.abs(m_dr - d_dr)
    ch2 = np.abs(d_dr - y_dr)
    ch3 = np.abs(ch1 - ch2)
    out['pyth_challenge_1'] = np.where(ch1 == 0, 9.0, ch1)
    out['pyth_challenge_2'] = np.where(ch2 == 0, 9.0, ch2)
    out['pyth_challenge_3'] = np.where(ch3 == 0, 9.0, ch3)

    return out


# ============================================================
# VORTEX MATH (RODIN/TESLA) & SACRED GEOMETRY FEATURES
# ============================================================

def compute_vortex_sacred_geometry_features(df: pd.DataFrame, tf_name: str = '1d') -> pd.DataFrame:
    """
    Vortex math (Rodin/Tesla) and sacred geometry features.

    A. Rodin Vortex Math — family number groups, doubling circuit, 3-6-9 candle body/range
    B. Sacred Geometry Ratios — proximity to sqrt(2), sqrt(3), phi, sqrt(5)
    C. Platonic Solid Cycles — 4,6,8,12,20 face phase + Torus/Metatron/Pentagonal
    D. Tesla 3-6-9 Temporal — bar-index and DOY modular cycles
    E. Kabbalah Pillars — Tree of Life pillar mapping (distinct from Rodin groups)

    All features are float-typed, vectorized, NaN-propagating.
    """
    _gpu = _is_gpu(df)
    if _gpu:
        idx = pd.DatetimeIndex(_np(df.index))
    else:
        idx = df.index

    cols = {}  # accumulate all columns, batch-assign at the end

    close_vals = _np(df['close']).astype(np.float64)
    open_vals = _np(df['open']).astype(np.float64)
    high_vals = _np(df['high']).astype(np.float64)
    low_vals = _np(df['low']).astype(np.float64)
    vol_vals = _np(df['volume']).astype(np.float64)

    # ── Digital roots (reuse universal_numerology.digital_root_vec) ──────
    _ci = np.where(np.isnan(close_vals), 0, (np.abs(close_vals) * 100).astype(np.int64))
    price_dr = np.where(_ci > 0, digital_root_vec(_ci), np.nan)

    _vi = np.where(np.isnan(vol_vals) | (vol_vals <= 0), 0,
                   np.abs(vol_vals).astype(np.int64))
    volume_dr = np.where(_vi > 0, digital_root_vec(_vi), np.nan)

    # Date DR (digit-sum then DR)
    _date_digits = (idx.year * 10000 + idx.month * 100 + idx.day).values.astype(np.int64)
    _digit_sums = np.zeros(len(idx), dtype=np.int64)
    _temp = _date_digits.copy()
    while np.any(_temp > 0):
        _digit_sums += _temp % 10
        _temp //= 10
    date_dr = digital_root_vec(_digit_sums).astype(np.float64)

    # ── A. Rodin Vortex Math ────────────────────────────────────────────
    # Family Number Groups (Rodin coil)
    #   Group 1: {1,4,7} — Physical/Magnetic
    #   Group 2: {2,5,8} — Electric
    #   Group 3: {3,6,9} — Flux/Control (Tesla axis)
    cols['vortex_family_group'] = np.where(
        np.isnan(price_dr), np.nan,
        np.where(np.isin(price_dr, [1, 4, 7]), 1.0,
        np.where(np.isin(price_dr, [2, 5, 8]), 2.0,
        np.where(np.isin(price_dr, [3, 6, 9]), 3.0, np.nan))))

    cols['vortex_date_family_group'] = np.where(
        np.isin(date_dr, [1, 4, 7]), 1.0,
        np.where(np.isin(date_dr, [2, 5, 8]), 2.0,
        np.where(np.isin(date_dr, [3, 6, 9]), 3.0, np.nan)))

    cols['vortex_vol_family_group'] = np.where(
        np.isnan(volume_dr), np.nan,
        np.where(np.isin(volume_dr, [1, 4, 7]), 1.0,
        np.where(np.isin(volume_dr, [2, 5, 8]), 2.0,
        np.where(np.isin(volume_dr, [3, 6, 9]), 3.0, np.nan))))

    # Family group alignment: price vs date (same group = resonance)
    cols['vortex_family_align'] = np.where(
        np.isnan(price_dr) | np.isnan(date_dr), np.nan,
        (cols['vortex_family_group'] == cols['vortex_date_family_group']).astype(np.float64))

    # Doubling Circuit position: {1,2,4,8,7,5} → positions 0-5
    # DR 3,6,9 are NOT on the doubling circuit → NaN
    _doubling_lut = np.full(10, np.nan)  # index 0-9
    _doubling_lut[1] = 0.0
    _doubling_lut[2] = 1.0
    _doubling_lut[4] = 2.0
    _doubling_lut[8] = 3.0
    _doubling_lut[7] = 4.0
    _doubling_lut[5] = 5.0
    _safe_dr = np.where(np.isnan(price_dr), 0, price_dr).astype(np.int64)
    cols['vortex_doubling_pos'] = np.where(
        np.isnan(price_dr), np.nan, _doubling_lut[_safe_dr])

    # 3-6-9 on candle body and range
    body = np.abs(close_vals - open_vals)
    _body_nan = np.isnan(body) | (body == 0)
    body_int = np.where(_body_nan, 1, (body * 100).astype(np.int64)).clip(min=1)
    body_dr = digital_root_vec(body_int).astype(np.float64)
    cols['candle_body_dr'] = np.where(_body_nan, np.nan, body_dr)
    cols['candle_body_is_369'] = np.where(_body_nan, np.nan,
                                          np.isin(body_dr, [3, 6, 9]).astype(np.float64))

    range_val = high_vals - low_vals
    _range_nan = np.isnan(range_val) | (range_val == 0)
    range_int = np.where(_range_nan, 1, (range_val * 100).astype(np.int64)).clip(min=1)
    range_dr = digital_root_vec(range_int).astype(np.float64)
    cols['candle_range_dr'] = np.where(_range_nan, np.nan, range_dr)
    cols['candle_range_is_369'] = np.where(_range_nan, np.nan,
                                            np.isin(range_dr, [3, 6, 9]).astype(np.float64))

    # ── B. Sacred Geometry Ratios (Vesica Piscis) ───────────────────────
    SACRED_ROOTS = np.array([
        np.sqrt(2),              # 1.41421 — diagonal of unit square
        np.sqrt(3),              # 1.73205 — vesica piscis height ratio
        (1.0 + np.sqrt(5)) / 2, # 1.61803 — phi / golden ratio
        np.sqrt(5),              # 2.23607 — diagonal of 1x2 rectangle
    ])
    # Reciprocals of sacred roots (also sacred)
    SACRED_RECIP = 1.0 / SACRED_ROOTS

    close_series = pd.Series(close_vals, index=idx)

    for lag in [1, 2, 3, 5, 8, 13]:
        shifted = close_series.shift(lag).values
        ratio = np.where((shifted > 0) & ~np.isnan(shifted) & ~np.isnan(close_vals),
                         close_vals / shifted, np.nan)
        # Minimum distance to any sacred root or its reciprocal
        min_dist = np.full(len(ratio), np.inf)
        for root, recip in zip(SACRED_ROOTS, SACRED_RECIP):
            dist_root = np.abs(ratio - root)
            dist_recip = np.abs(ratio - recip)
            min_dist = np.minimum(min_dist, np.minimum(dist_root, dist_recip))
        cols[f'sacred_ratio_dist_{lag}'] = np.where(np.isnan(ratio), np.nan, min_dist)

    # Vesica piscis: sqrt(3) retracement level within candle
    hl_range = high_vals - low_vals
    _hl_valid = (hl_range > 0) & ~np.isnan(hl_range) & ~np.isnan(close_vals) & ~np.isnan(low_vals)
    retrace_pct = np.where(_hl_valid, (close_vals - low_vals) / hl_range, np.nan)
    cols['vesica_sqrt3_retrace'] = np.where(_hl_valid, retrace_pct - (1.0 / np.sqrt(3)), np.nan)

    # ── C. Platonic Solid Cycles ────────────────────────────────────────
    bar_idx = np.arange(len(df), dtype=np.float64)

    # 5 Platonic solids: tetrahedron(4), cube(6), octahedron(8), dodecahedron(12), icosahedron(20)
    for n_faces in [4, 6, 8, 12, 20]:
        cols[f'platonic_{n_faces}_phase'] = (bar_idx % n_faces)

    # Torus 7-phase (seven-color map theorem on torus)
    cols['torus_7_phase'] = (bar_idx % 7)

    # Metatron's Cube: 13-sphere cycle (13 circles of Fruit of Life)
    cols['metatron_13_phase'] = (bar_idx % 13)

    # Cosmic Cube / 144 (12x12 grid, New Jerusalem 144,000)
    cols['metatron_144_phase'] = (bar_idx % 144)

    # 72-degree pentagonal rotation (DNA double helix, 360/5=72)
    cols['pentagonal_72_phase'] = (bar_idx % 72)

    # ── D. Tesla 3-6-9 Temporal ─────────────────────────────────────────
    cols['tesla_3_phase'] = (bar_idx % 3)
    cols['tesla_6_phase'] = (bar_idx % 6)
    cols['tesla_9_phase'] = (bar_idx % 9)
    cols['tesla_36_phase'] = (bar_idx % 36)

    # DOY mod 3 and 6 (doy_div_9 already exists in numerology, skip)
    doy = idx.dayofyear.values.astype(np.float64)
    cols['doy_mod_3'] = (doy % 3)
    cols['doy_mod_6'] = (doy % 6)

    # ── E. Kabbalah Tree of Life Pillars ────────────────────────────────
    # Note: these groupings differ from Rodin vortex groups!
    #   Pillar of Mercy   = {2,4,7}  (Chokmah, Chesed, Netzach)
    #   Pillar of Severity = {3,5,8}  (Binah, Geburah, Hod)
    #   Middle Pillar      = {1,6,9}  (Kether, Tiphareth, Yesod)
    cols['kabbalah_pillar_mercy'] = np.where(
        np.isnan(price_dr), np.nan, np.isin(price_dr, [2, 4, 7]).astype(np.float64))
    cols['kabbalah_pillar_severity'] = np.where(
        np.isnan(price_dr), np.nan, np.isin(price_dr, [3, 5, 8]).astype(np.float64))
    cols['kabbalah_pillar_balance'] = np.where(
        np.isnan(price_dr), np.nan, np.isin(price_dr, [1, 6, 9]).astype(np.float64))

    # Date-based pillar (same groupings on date DR)
    cols['kabbalah_date_pillar_mercy'] = np.isin(date_dr, [2, 4, 7]).astype(np.float64)
    cols['kabbalah_date_pillar_severity'] = np.isin(date_dr, [3, 5, 8]).astype(np.float64)
    cols['kabbalah_date_pillar_balance'] = np.isin(date_dr, [1, 6, 9]).astype(np.float64)

    # ── Batch assign all columns ────────────────────────────────────────
    out = pd.DataFrame(cols, index=idx)
    return out


# ============================================================
# COMPUTE ASTROLOGY FEATURES (from pre-loaded caches)
# ============================================================

def compute_astrology_features(df: pd.DataFrame, astro_cache: dict) -> pd.DataFrame:
    """
    Compute astrology features from pre-loaded ephemeris/astrology DataFrames.

    Args:
        df: OHLCV DataFrame with DatetimeIndex
        astro_cache: dict with keys:
            'ephemeris': DataFrame (date-indexed) from ephemeris_cache.db
            'astrology': DataFrame (date-indexed) from astrology_full.db
            'fear_greed': DataFrame (date-indexed) from fear_greed.db
            'google_trends': DataFrame (date-indexed) from google_trends.db
            'funding_daily': DataFrame (date-indexed) avg daily funding rate
    """
    _gpu = _is_gpu(df)
    # Astrology merges with external pandas DataFrames — extract CPU index/values, avoid full _to_cpu()
    if _gpu:
        _cpu_idx = pd.DatetimeIndex(_np(df.index))
    else:
        _cpu_idx = df.index
    out = pd.DataFrame(index=_cpu_idx)
    c = pd.Series(_np(df['close']).astype(np.float64), index=_cpu_idx)
    # Normalize dates and strip timezone for alignment with tz-naive daily data
    df_date_idx = _cpu_idx.normalize()
    if df_date_idx.tz is not None:
        df_date_idx = df_date_idx.tz_localize(None)

    ephem_df = astro_cache.get('ephemeris', pd.DataFrame())
    astro_df = astro_cache.get('astrology', pd.DataFrame())

    # --- Western Astrology (from ephemeris) ---
    for col in ['moon_mansion', 'moon_phase', 'mercury_retrograde', 'hard_aspects',
                'soft_aspects', 'planetary_strength', 'digital_root']:
        if col in ephem_df.columns:
            s = pd.to_numeric(ephem_df[col], errors='coerce')
            s.index = s.index.normalize()
            mapped = s.reindex(df_date_idx)
            mapped.index = _cpu_idx
            out[f'west_{col}'] = mapped.ffill()

    if 'planetary_strength' in ephem_df.columns:
        s = pd.to_numeric(ephem_df['planetary_strength'], errors='coerce')
        s.index = s.index.normalize()
        mapped = s.reindex(df_date_idx)
        mapped.index = _cpu_idx
        out['psi'] = mapped.ffill()

    if 'digital_root' in ephem_df.columns:
        s = pd.to_numeric(ephem_df['digital_root'], errors='coerce')
        s.index = s.index.normalize()
        mapped = s.reindex(df_date_idx)
        mapped.index = _cpu_idx
        out['digital_root_genesis'] = mapped.ffill()

    if 'west_moon_phase' in out.columns:
        moon_phase = pd.to_numeric(out['west_moon_phase'], errors='coerce')
        out['lunar_phase_sin'] = np.sin(2 * np.pi * moon_phase / 30)
        out['lunar_phase_cos'] = np.cos(2 * np.pi * moon_phase / 30)

    # --- Vedic Astrology (from astrology_full) ---
    for col in ['nakshatra', 'tithi', 'yoga', 'karana']:
        if col in astro_df.columns:
            s = pd.to_numeric(astro_df[col], errors='coerce')
            s.index = s.index.normalize()
            mapped = s.reindex(df_date_idx)
            mapped.index = _cpu_idx
            out[f'vedic_{col}'] = mapped.ffill()

    if 'nakshatra_nature' in astro_df.columns:
        nature_map = {'Deva': 1, 'Manushya': 0, 'Rakshasa': -1,
                      'deva': 1, 'manushya': 0, 'rakshasa': -1}
        s = astro_df['nakshatra_nature'].map(nature_map)
        s.index = s.index.normalize()
        mapped = s.reindex(df_date_idx)
        mapped.index = _cpu_idx
        out['vedic_nature_encoded'] = mapped.ffill()

    if 'nakshatra_guna' in astro_df.columns:
        guna_map = {'Sattva': 1, 'Rajas': 0, 'Tamas': -1,
                    'satva': 1, 'sattva': 1, 'rajas': 0, 'tamas': -1}
        s = astro_df['nakshatra_guna'].map(guna_map)
        s.index = s.index.normalize()
        mapped = s.reindex(df_date_idx)
        mapped.index = _cpu_idx
        out['vedic_guna_encoded'] = mapped.ffill()

    if 'vedic_nakshatra' in out.columns:
        key_nakshatras = [0, 6, 9, 13, 17, 22, 26]
        out['vedic_key_nakshatra'] = out['vedic_nakshatra'].isin(key_nakshatras).astype(int)

    # --- Chinese BaZi (from astrology_full) ---
    for col in ['day_stem', 'day_branch', 'day_clash_branch']:
        if col in astro_df.columns:
            s = pd.to_numeric(astro_df[col], errors='coerce')
            s.index = s.index.normalize()
            mapped = s.reindex(df_date_idx)
            mapped.index = _cpu_idx
            out[f'bazi_{col}'] = mapped.ffill()

    if 'day_element' in astro_df.columns:
        s = astro_df['day_element']
        elem_map = {'Wood': 0, 'Fire': 1, 'Earth': 2, 'Metal': 3, 'Water': 4,
                    'Fire+': 1, 'Fire-': 1, 'Wood+': 0, 'Wood-': 0,
                    'Earth+': 2, 'Earth-': 2, 'Metal+': 3, 'Metal-': 3,
                    'Water+': 4, 'Water-': 4}
        s_num = s.map(elem_map)
        if s_num.isna().all():
            s_num = pd.to_numeric(s, errors='coerce')
        s_num.index = s_num.index.normalize()
        mapped = s_num.reindex(df_date_idx)
        mapped.index = _cpu_idx
        out['bazi_day_element_idx'] = mapped.ffill()
        friendly_map = {0: 0, 1: 1, 2: 0, 3: 1, 4: -1}
        out['bazi_btc_friendly'] = out['bazi_day_element_idx'].map(friendly_map)

    # --- Mayan Tzolkin (from astrology_full) ---
    for col in ['tzolkin_tone', 'tzolkin_sign_idx']:
        if col in astro_df.columns:
            s = pd.to_numeric(astro_df[col], errors='coerce')
            s.index = s.index.normalize()
            mapped = s.reindex(df_date_idx)
            mapped.index = _cpu_idx
            short_name = 'mayan_tone' if 'tone' in col else 'mayan_sign_idx'
            out[short_name] = mapped.ffill()

    if 'mayan_tone' in out.columns:
        out['mayan_tone_1'] = (out['mayan_tone'] == 1).astype(int)
        out['mayan_tone_9'] = (out['mayan_tone'] == 9).astype(int)
        out['mayan_tone_13'] = (out['mayan_tone'] == 13).astype(int)

    # --- Arabic Lots (from astrology_full) ---
    for lot_col in ['lot_commerce', 'lot_increase', 'lot_catastrophe', 'lot_treachery']:
        if lot_col in astro_df.columns:
            s = pd.to_numeric(astro_df[lot_col], errors='coerce')
            s.index = s.index.normalize()
            mapped = s.reindex(df_date_idx)
            mapped.index = _cpu_idx
            out[f'arabic_{lot_col}'] = mapped.ffill()

    if 'moon_tropical_lon' in astro_df.columns:
        moon_lon = pd.to_numeric(astro_df['moon_tropical_lon'], errors='coerce')
        moon_lon.index = moon_lon.index.normalize()
        moon_mapped = moon_lon.reindex(df_date_idx)
        moon_mapped.index = _cpu_idx
        moon_mapped = moon_mapped.ffill()
        for lot_col in ['lot_commerce', 'lot_increase', 'lot_catastrophe', 'lot_treachery']:
            if f'arabic_{lot_col}' in out.columns:
                lot_vals = out[f'arabic_{lot_col}']
                diff = (moon_mapped - lot_vals).abs()
                diff = diff.where(diff < 180, 360 - diff)
                out[f'arabic_{lot_col}_moon_conj'] = (diff < 10).astype(int)

    # --- BaZi / Tzolkin from universal_astro (fallback compute) ---
    # If astro_df didn't have them, compute from universal_astro
    if 'bazi_day_stem' not in out.columns and 'bazi_stem_idx' not in out.columns:
        unique_dates = pd.Series(_cpu_idx.date).unique()
        bazi_map = {}
        tzolkin_map = {}
        for d in unique_dates:
            try:
                dt = datetime.combine(d, datetime.min.time())
                bz = get_bazi(dt)
                tz = get_tzolkin(dt)
                bazi_map[d] = (bz['stem_idx'], bz['branch_idx'], bz['element_idx'])
                tzolkin_map[d] = (tz['tone'], tz['sign_idx'], tz['kin'])
            except Exception:
                bazi_map[d] = (0, 0, 0)
                tzolkin_map[d] = (0, 0, 0)

        df_dates_for_astro = pd.Series(_cpu_idx.date, index=_cpu_idx)
        out['bazi_stem_idx'] = df_dates_for_astro.map(lambda d: bazi_map.get(d, (0, 0, 0))[0])
        out['bazi_branch_idx'] = df_dates_for_astro.map(lambda d: bazi_map.get(d, (0, 0, 0))[1])
        out['bazi_element_idx'] = df_dates_for_astro.map(lambda d: bazi_map.get(d, (0, 0, 0))[2])
        out['tzolkin_tone'] = df_dates_for_astro.map(lambda d: tzolkin_map.get(d, (0, 0, 0))[0])
        out['tzolkin_sign_idx'] = df_dates_for_astro.map(lambda d: tzolkin_map.get(d, (0, 0, 0))[1])
        out['tzolkin_kin'] = df_dates_for_astro.map(lambda d: tzolkin_map.get(d, (0, 0, 0))[2])

    # --- Venus/Mars retrograde + Void-of-Course Moon (from get_western) ---
    # These are NOT in astro_df (DB doesn't store them), so compute per unique date
    unique_dates_w = pd.Series(_cpu_idx.date).unique()
    west_map = {}
    for d in unique_dates_w:
        try:
            dt = datetime.combine(d, datetime.min.time())
            w = get_western(dt)
            west_map[d] = (w['venus_retrograde'], w['mars_retrograde'], w['voc_moon'])
        except Exception:
            west_map[d] = (np.nan, np.nan, np.nan)
    df_dates_w = pd.Series(_cpu_idx.date, index=_cpu_idx)
    out['west_venus_retrograde'] = df_dates_w.map(lambda d: west_map.get(d, (np.nan,))[0])
    out['west_mars_retrograde'] = df_dates_w.map(lambda d: west_map.get(d, (np.nan,))[1])
    out['west_voc_moon'] = df_dates_w.map(lambda d: west_map.get(d, (np.nan,))[2])

    # --- Zodiac sign (sun sign 0-11) from get_zodiac ---
    _zodiac_name_to_idx = {
        'Aries': 0, 'Taurus': 1, 'Gemini': 2, 'Cancer': 3, 'Leo': 4, 'Virgo': 5,
        'Libra': 6, 'Scorpio': 7, 'Sagittarius': 8, 'Capricorn': 9, 'Aquarius': 10, 'Pisces': 11,
    }
    zodiac_map = {}
    for d in unique_dates_w:
        try:
            dt = datetime.combine(d, datetime.min.time())
            sign = get_zodiac(dt)
            zodiac_map[d] = _zodiac_name_to_idx.get(sign, np.nan)
        except Exception:
            zodiac_map[d] = np.nan
    out['west_zodiac_sign_idx'] = df_dates_w.map(zodiac_map)

    # --- Planetary hour (standalone bar feature) ---
    # Uses hour from each bar's timestamp; only meaningful for intraday, but
    # for daily bars the noon (12:00) approximation still yields a valid planetary hour.
    _ph_day_start = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int32)  # indexed by dow (Mon=0)
    _ph_hours = _cpu_idx.hour.values
    _ph_dows = _cpu_idx.dayofweek.values  # Mon=0
    _ph_offset = np.where(_ph_hours >= 6, _ph_hours - 6, _ph_hours + 18)
    _ph_planet_idx = (_ph_day_start[_ph_dows] + _ph_offset) % 7
    out['planetary_hour_idx'] = _ph_planet_idx
    # Jupiter=4 (prosperity), Saturn=6 (restriction)
    out['planetary_hour_is_jupiter'] = (_ph_planet_idx == 4).astype(np.int32)
    out['planetary_hour_is_saturn'] = (_ph_planet_idx == 6).astype(np.int32)
    # Double power: day ruler == hour ruler
    # Day rulers: Mon=Moon(1), Tue=Mars(2), Wed=Mercury(3), Thu=Jupiter(4),
    #             Fri=Venus(5), Sat=Saturn(6), Sun=Sun(0)
    _day_ruler_idx = _ph_day_start[_ph_dows]
    out['planetary_double_power'] = (_ph_planet_idx == _day_ruler_idx).astype(np.int32)

    return out


# ============================================================
# COMPUTE PLANETARY EXPANSION FEATURES (~30 cols)
# ============================================================

def compute_planetary_expansion_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ~30 planetary expansion features via PyEphem.

    Features: planetary speeds (10), combustion/cazimi (12), essential dignity (10),
    synodic cycle phases (8), sun decan (1), Behenian star conjunctions (8).

    Computation is per unique *date* (PyEphem is date-resolution for most of these).
    For intraday TFs, values are forward-filled to bar frequency.
    NaN propagation: if ephem fails for a date, that date's features stay NaN.

    Args:
        df: OHLCV DataFrame with DatetimeIndex (pandas or cuDF).

    Returns:
        DataFrame aligned to df.index with ~30 float columns.
    """
    from astrology_engine import get_planetary_expansion

    _gpu = _is_gpu(df)
    if _gpu:
        _cpu_idx = pd.DatetimeIndex(_np(df.index))
    else:
        _cpu_idx = df.index

    # Get unique dates (PyEphem is daily resolution)
    dates_series = _cpu_idx.normalize()
    if hasattr(dates_series, 'tz') and dates_series.tz is not None:
        dates_series = dates_series.tz_localize(None)
    unique_dates = dates_series.unique()

    # Compute per unique date
    date_features = {}
    for d in unique_dates:
        dt = d.to_pydatetime()
        try:
            feats = get_planetary_expansion(dt)
            date_features[d] = feats
        except Exception:
            # Leave this date as NaN (will propagate naturally)
            pass

    if not date_features:
        # No dates computed successfully — return empty frame with NaN
        return pd.DataFrame(index=_cpu_idx)

    # Build DataFrame from date->features dict, then map back to bars
    feat_df = pd.DataFrame.from_dict(date_features, orient='index')
    feat_df.index = pd.DatetimeIndex(feat_df.index)

    # Map to bar-level index via date alignment + ffill for intraday
    mapped = feat_df.reindex(dates_series)
    mapped.index = _cpu_idx
    mapped = mapped.ffill()

    # Ensure float dtype (no ints, no objects)
    for col in mapped.columns:
        mapped[col] = pd.to_numeric(mapped[col], errors='coerce').astype(np.float32)

    return mapped


# ============================================================
# COMPUTE ESOTERIC FEATURES
# ============================================================

def compute_esoteric_features(df: pd.DataFrame, tweets_df: pd.DataFrame,
                              news_df: pd.DataFrame, sports_df: pd.DataFrame,
                              onchain_df: pd.DataFrame, macro_df: pd.DataFrame,
                              astro_cache: dict, bucket_seconds: int) -> pd.DataFrame:
    """
    Compute ALL esoteric features from pre-loaded event data.

    Each event df has a 'ts_unix' column (numeric epoch seconds).
    Features are bucketed by bucket_seconds.
    Missing data = NaN (never 0, never ffill).

    Uses universal engines: gematria, numerology, astro, sentiment.

    Args:
        df: OHLCV DataFrame with DatetimeIndex and 'open_time' column (ms epoch)
        tweets_df: from tweets.db (ts_unix, full_text, user_handle, has_gold, has_red, etc.)
        news_df: from news_articles.db (ts_unix, title, sentiment_score, title_dr, etc.)
        sports_df: dict with 'games' and 'horse_races' DataFrames
        onchain_df: from onchain_data.db (timestamp, block_dr, funding_dr, etc.)
        macro_df: from macro_data.db (date-indexed, dxy, gold, spx, vix, etc.)
        astro_cache: dict with 'ephemeris', 'astrology', 'fear_greed', etc.
        bucket_seconds: bucket size in seconds matching the timeframe
    """
    _gpu = _is_gpu(df)
    # Esoteric merges with external pandas DataFrames — extract CPU index/values, avoid full _to_cpu()
    if _gpu:
        _cpu_idx = pd.DatetimeIndex(_np(df.index))
    else:
        _cpu_idx = df.index
    out = pd.DataFrame(index=_cpu_idx)
    c = pd.Series(_np(df['close']).astype(np.float64), index=_cpu_idx)
    o = pd.Series(_np(df['open']).astype(np.float64), index=_cpu_idx)
    v = pd.Series(_np(df['volume']).astype(np.float64), index=_cpu_idx)
    df_date_idx = _cpu_idx.normalize()
    if df_date_idx.tz is not None:
        df_date_idx = df_date_idx.tz_localize(None)

    # Open time in seconds for bucketing
    if 'open_time' in (df.columns if not _gpu else list(df.columns)):
        ts_sec = pd.to_numeric(pd.Series(_np(df['open_time']), index=_cpu_idx), errors='coerce') / 1000
    else:
        ts_sec = pd.Series(_cpu_idx.astype(np.int64) // 10**9, index=_cpu_idx)
    df_buckets = (ts_sec // bucket_seconds * bucket_seconds).astype(int)

    # ===========================================================
    # TWEET FEATURES
    # ===========================================================
    if tweets_df is not None and len(tweets_df) > 0:
        tw = tweets_df.copy()
        tw['ts_unix'] = pd.to_numeric(tw['ts_unix'], errors='coerce')
        tw['bucket'] = (tw['ts_unix'] // bucket_seconds) * bucket_seconds

        # Ensure has_green column exists (may be absent in older data)
        if 'has_green' not in tw.columns:
            tw['has_green'] = 0
        if 'dominant_colors' not in tw.columns:
            tw['dominant_colors'] = None

        # --- Color helper columns for aggregation ---
        tw['_has_gold_num'] = pd.to_numeric(tw['has_gold'], errors='coerce').fillna(0)
        tw['_has_red_num'] = pd.to_numeric(tw['has_red'], errors='coerce').fillna(0)
        tw['_has_green_num'] = pd.to_numeric(tw['has_green'], errors='coerce').fillna(0)
        tw['_has_any_color'] = ((tw['_has_gold_num'] > 0) |
                                (tw['_has_red_num'] > 0) |
                                (tw['_has_green_num'] > 0)).astype(int)
        # Color sentiment per tweet: gold=+1, green=+1, red=-1, none=0
        tw['_color_sent'] = tw['_has_gold_num'] + tw['_has_green_num'] - tw['_has_red_num']

        # Dominant color per tweet: 0=none, 1=gold, 2=red, 3=green, 4=other — vectorized
        _g = tw['_has_gold_num'].values
        _r = tw['_has_red_num'].values
        _gr = tw['_has_green_num'].values
        _no_color = (_g == 0) & (_r == 0) & (_gr == 0)
        # Check dominant_colors column for "other" presence
        _has_other = np.zeros(len(tw), dtype=bool)
        if 'dominant_colors' in tw.columns:
            _dc = tw['dominant_colors'].fillna('').astype(str)
            _has_other = ~_dc.isin(['', 'None', 'null', '[]'])
        # Default: pick strongest (ties: gold > red > green)
        _dom = np.where(_g >= _r, np.where(_g >= _gr, 1, 3), np.where(_r >= _gr, 2, 3))
        # Override: no color → check other, else 0
        _dom = np.where(_no_color, np.where(_has_other, 4, 0), _dom)
        tw['_dominant_code'] = _dom

        # Basic bucketed counts
        tw_basic = tw.groupby('bucket').agg(
            tweets_count=('full_text', 'count'),
            gold_tweet=('has_gold', lambda x: int(pd.to_numeric(x, errors='coerce').sum() > 0)),
            red_tweet=('has_red', lambda x: int(pd.to_numeric(x, errors='coerce').sum() > 0)),
            green_tweet=('has_green', lambda x: int(pd.to_numeric(x, errors='coerce').sum() > 0)),
            tweet_dominant_color=('_dominant_code', lambda x: int(x.mode().iloc[0]) if len(x) > 0 else 0),
            tweet_color_count=('_has_any_color', 'sum'),
            color_sentiment=('_color_sent', 'mean'),
        )

        bucket_vals = df_buckets.values
        # Vectorized merge instead of per-row dict lookup
        _bucket_df = pd.DataFrame({'_bk': bucket_vals})
        tw_basic_reset = tw_basic.reset_index().rename(columns={'bucket': '_bk'})
        _merged = _bucket_df.merge(tw_basic_reset, on='_bk', how='left')
        tweets_count = _merged['tweets_count'].values.astype(np.float64)
        tweet_gold = _merged['gold_tweet'].values.astype(np.float64)
        tweet_red = _merged['red_tweet'].values.astype(np.float64)
        tweet_green = _merged['green_tweet'].values.astype(np.float64)
        tweet_dom_color = _merged['tweet_dominant_color'].values.astype(np.float64)
        tweet_color_cnt = _merged['tweet_color_count'].values.astype(np.float64)
        tweet_color_sent = _merged['color_sentiment'].values.astype(np.float64)
        del _bucket_df, tw_basic_reset, _merged

        tf_label = _tf_label(bucket_seconds)
        bucket_col_name = f'tweets_this_{tf_label}'
        out[bucket_col_name] = tweets_count
        out[f'gold_tweet_this_{tf_label}'] = tweet_gold
        out[f'red_tweet_this_{tf_label}'] = tweet_red
        out[f'green_tweet_this_{tf_label}'] = tweet_green
        out[f'tweet_dominant_color_this_{tf_label}'] = tweet_dom_color
        out[f'tweet_color_count_this_{tf_label}'] = tweet_color_cnt
        out[f'color_sentiment_this_{tf_label}'] = tweet_color_sent

        # Daily totals
        tw['dt'] = pd.to_datetime(tw['ts_unix'], unit='s', errors='coerce', utc=True)
        tw['date'] = tw['dt'].dt.normalize().dt.tz_localize(None)
        tw_daily = tw.groupby('date').agg(
            tweets_today=('full_text', 'count'),
            gold_tweet_today=('has_gold', lambda x: int(pd.to_numeric(x, errors='coerce').sum() > 0)),
            red_tweet_today=('has_red', lambda x: int(pd.to_numeric(x, errors='coerce').sum() > 0)),
            green_tweet_today=('has_green', lambda x: int(pd.to_numeric(x, errors='coerce').sum() > 0)),
            tweet_dominant_color_today=('_dominant_code', lambda x: int(x.mode().iloc[0]) if len(x) > 0 else 0),
            tweet_color_count_today=('_has_any_color', 'sum'),
            color_sentiment_today=('_color_sent', 'mean'),
        )
        for col in tw_daily.columns:
            s = tw_daily[col]
            s.index = s.index.normalize()
            mapped = s.reindex(df_date_idx)
            mapped.index = _cpu_idx
            out[col] = mapped  # NaN if no tweets that day

        # Misdirection (red tweet yesterday + green candle today)
        if 'red_tweet_today' in out.columns:
            bars_per_day = max(1, 86400 // bucket_seconds)
            out['misdirection'] = ((out['red_tweet_today'].shift(bars_per_day) == 1) &
                                   (c > o)).astype(int)

        # Gematria (all 6 ciphers) + sentiment on tweets — GPU BATCH (zero .apply())
        try:
            # All 6 gematria ciphers + DRs + flags on tweet text — single GPU batch call
            _tw_gem = gematria_gpu_batch(tw['full_text'], prefix='tweet_gem')
            tw['tweet_gem_ord'] = _tw_gem['tweet_gem_ordinal'].values
            tw['tweet_gem_rev'] = _tw_gem['tweet_gem_reverse'].values
            tw['tweet_gem_red'] = _tw_gem['tweet_gem_reduction'].values
            tw['tweet_gem_eng'] = _tw_gem['tweet_gem_english'].values
            tw['tweet_gem_jew'] = _tw_gem['tweet_gem_jewish'].values
            tw['tweet_gem_sat'] = _tw_gem['tweet_gem_satanic'].values
            tw['tweet_gem_chal'] = _tw_gem['tweet_gem_chaldean'].values
            tw['tweet_gem_alb'] = _tw_gem['tweet_gem_albam'].values
            tw['tweet_gem_dr_ord'] = _tw_gem['tweet_gem_dr_ordinal'].values
            tw['tweet_gem_dr_rev'] = _tw_gem['tweet_gem_dr_reverse'].values
            tw['tweet_gem_dr_eng'] = digital_root_vec(_tw_gem['tweet_gem_english'].values)
            tw['tweet_gem_dr_jew'] = digital_root_vec(_tw_gem['tweet_gem_jewish'].values)
            tw['tweet_gem_dr_sat'] = digital_root_vec(_tw_gem['tweet_gem_satanic'].values)
            tw['tweet_gem_dr_chal'] = _tw_gem['tweet_gem_dr_chaldean'].values
            tw['tweet_gem_dr_alb'] = _tw_gem['tweet_gem_dr_albam'].values
            tw['tweet_is_caution'] = _tw_gem['tweet_gem_is_caution'].values
            tw['tweet_is_pump'] = _tw_gem['tweet_gem_is_pump'].values
            tw['tweet_is_btc_energy'] = _tw_gem['tweet_gem_is_btc_energy'].values

            # User handle gematria — single GPU batch call
            _uh_gem = gematria_gpu_batch(tw['user_handle'], prefix='user_gem')
            tw['user_gem_ord'] = _uh_gem['user_gem_ordinal'].values
            tw['user_gem_rev'] = _uh_gem['user_gem_reverse'].values
            tw['user_gem_red'] = _uh_gem['user_gem_reduction'].values
            tw['user_gem_eng'] = _uh_gem['user_gem_english'].values
            tw['user_gem_jew'] = _uh_gem['user_gem_jewish'].values
            tw['user_gem_sat'] = _uh_gem['user_gem_satanic'].values
            tw['user_gem_chal'] = _uh_gem['user_gem_chaldean'].values
            tw['user_gem_alb'] = _uh_gem['user_gem_albam'].values
            tw['user_gem_dr'] = _uh_gem['user_gem_dr_ordinal'].values

            # Sentiment — single GPU batch call
            _tw_sent = sentiment_gpu_batch(tw['full_text'], prefix='tw_sent')
            tw['tweet_sentiment'] = _tw_sent['tw_sent_score'].values
            tw['tweet_caps'] = _tw_sent['tw_sent_has_caps'].values
            tw['tweet_excl'] = _tw_sent['tw_sent_exclamation'].values
            tw['tweet_urgency'] = _tw_sent['tw_sent_urgency'].values
            tw['tweet_is_bull'] = (tw['tweet_sentiment'] > 0).astype(int)
            tw['tweet_is_bear'] = (tw['tweet_sentiment'] < 0).astype(int)

            # Engagement DR — vectorized
            for eng_col in ['favorite_count', 'retweet_count', 'reply_count']:
                if eng_col in tw.columns:
                    dr_col = eng_col.replace('_count', '').replace('favorite', 'likes') + '_dr'
                    _eng_vals = pd.to_numeric(tw[eng_col], errors='coerce').fillna(0).values
                    _eng_int = np.where(_eng_vals > 0, np.abs(_eng_vals).astype(np.int64), 0)
                    tw[dr_col] = np.where(_eng_int > 0, digital_root_vec(_eng_int), 0)

            # Build aggregation dict -- all 6 ciphers mean + DR modes
            agg_dict_tw = {
                # 6 cipher means
                'tweet_gem_ordinal_mean': ('tweet_gem_ord', 'mean'),
                'tweet_gem_reverse_mean': ('tweet_gem_rev', 'mean'),
                'tweet_gem_reduction_mean': ('tweet_gem_red', 'mean'),
                'tweet_gem_english_mean': ('tweet_gem_eng', 'mean'),
                'tweet_gem_jewish_mean': ('tweet_gem_jew', 'mean'),
                'tweet_gem_satanic_mean': ('tweet_gem_sat', 'mean'),
                'tweet_gem_chaldean_mean': ('tweet_gem_chal', 'mean'),
                'tweet_gem_albam_mean': ('tweet_gem_alb', 'mean'),
                # DR modes for each cipher
                'tweet_gem_dr_ord_mode': ('tweet_gem_dr_ord', _mode_or_nan),
                'tweet_gem_dr_rev_mode': ('tweet_gem_dr_rev', _mode_or_nan),
                'tweet_gem_dr_eng_mode': ('tweet_gem_dr_eng', _mode_or_nan),
                'tweet_gem_dr_jew_mode': ('tweet_gem_dr_jew', _mode_or_nan),
                'tweet_gem_dr_sat_mode': ('tweet_gem_dr_sat', _mode_or_nan),
                'tweet_gem_dr_chal_mode': ('tweet_gem_dr_chal', _mode_or_nan),
                'tweet_gem_dr_alb_mode': ('tweet_gem_dr_alb', _mode_or_nan),
                # User handle gematria means
                'tweet_user_gem_ordinal_mean': ('user_gem_ord', 'mean'),
                'tweet_user_gem_reverse_mean': ('user_gem_rev', 'mean'),
                'tweet_user_gem_reduction_mean': ('user_gem_red', 'mean'),
                'tweet_user_gem_english_mean': ('user_gem_eng', 'mean'),
                'tweet_user_gem_jewish_mean': ('user_gem_jew', 'mean'),
                'tweet_user_gem_satanic_mean': ('user_gem_sat', 'mean'),
                'tweet_user_gem_chaldean_mean': ('user_gem_chal', 'mean'),
                'tweet_user_gem_albam_mean': ('user_gem_alb', 'mean'),
                'tweet_user_gem_dr_mode': ('user_gem_dr', _mode_or_nan),
                # Caution/pump/energy flags (any tweet in bucket)
                'tweet_gem_caution': ('tweet_is_caution', 'max'),
                'tweet_gem_pump': ('tweet_is_pump', 'max'),
                'tweet_gem_btc_energy': ('tweet_is_btc_energy', 'max'),
                # Sentiment (expanded)
                'tweet_sentiment_mean': ('tweet_sentiment', 'mean'),
                'tweet_sentiment_min': ('tweet_sentiment', 'min'),
                'tweet_sentiment_max': ('tweet_sentiment', 'max'),
                'tweet_caps_any': ('tweet_caps', 'max'),
                'tweet_caps_pct': ('tweet_caps', 'mean'),
                'tweet_excl_max': ('tweet_excl', 'max'),
                'tweet_excl_mean': ('tweet_excl', 'mean'),
                'tweet_urgency_mean': ('tweet_urgency', 'mean'),
                'tweet_bull_count': ('tweet_is_bull', 'sum'),
                'tweet_bear_count': ('tweet_is_bear', 'sum'),
            }
            # Legacy aliases for backward compatibility
            agg_dict_tw['tweet_gem_ord_mean'] = ('tweet_gem_ord', 'mean')
            agg_dict_tw['tweet_gem_dr_mode'] = ('tweet_gem_dr_ord', _mode_or_nan)
            agg_dict_tw['user_gem_dr_mode'] = ('user_gem_dr', _mode_or_nan)

            if 'likes_dr' in tw.columns:
                agg_dict_tw['likes_dr_mode'] = ('likes_dr', _mode_or_nan)

            tw_agg = tw.groupby('bucket').agg(**agg_dict_tw).reset_index()

            bucket_map_tw = tw_agg.set_index('bucket')
            for col in tw_agg.columns:
                if col == 'bucket':
                    continue
                mapped = df_buckets.map(bucket_map_tw[col].to_dict())
                out[col] = mapped  # NaN if no tweets in bucket

            # Derived: divergence = bull_count - bear_count
            if 'tweet_bull_count' in out.columns and 'tweet_bear_count' in out.columns:
                out['tweet_sentiment_divergence'] = out['tweet_bull_count'] - out['tweet_bear_count']
        except Exception:
            pass

    # ===========================================================
    # NEWS FEATURES
    # ===========================================================
    if news_df is not None and len(news_df) > 0:
        nw = news_df.copy()
        nw['ts_unix'] = pd.to_numeric(nw['ts_unix'], errors='coerce')
        nw['bucket'] = (nw['ts_unix'] // bucket_seconds) * bucket_seconds

        # Basic bucketed news
        # Try to use pre-computed columns if they exist
        agg_dict = {'news_count': ('title', 'count')}

        if 'sentiment_score' in nw.columns:
            nw['sentiment_score'] = pd.to_numeric(nw['sentiment_score'], errors='coerce')
            agg_dict['news_sentiment'] = ('sentiment_score', 'mean')

        if 'title_dr' in nw.columns:
            nw['title_dr'] = pd.to_numeric(nw['title_dr'], errors='coerce')
            agg_dict['caution_gematria'] = ('title_dr', lambda x: int((x == 9).sum() > 0))

        nw_basic = nw.groupby('bucket').agg(**agg_dict)
        # Vectorized merge instead of per-row dict lookup
        _bucket_nw_df = pd.DataFrame({'_bk': bucket_vals})
        nw_basic_reset = nw_basic.reset_index().rename(columns={'bucket': '_bk'})
        _nw_merged = _bucket_nw_df.merge(nw_basic_reset, on='_bk', how='left')
        news_count = _nw_merged['news_count'].values.astype(np.float64) if 'news_count' in _nw_merged else np.full(len(df), np.nan)
        news_sent = _nw_merged['news_sentiment'].values.astype(np.float64) if 'news_sentiment' in _nw_merged else np.full(len(df), np.nan)
        news_caution = _nw_merged['caution_gematria'].values.astype(np.float64) if 'caution_gematria' in _nw_merged else np.full(len(df), np.nan)
        del _bucket_nw_df, nw_basic_reset, _nw_merged

        out[f'news_count_{_tf_label(bucket_seconds)}'] = news_count
        out[f'news_sentiment_{_tf_label(bucket_seconds)}'] = news_sent
        out[f'caution_gematria_{_tf_label(bucket_seconds)}'] = news_caution

        # Daily news totals
        nw['dt'] = pd.to_datetime(nw['ts_unix'], unit='s', errors='coerce', utc=True)
        nw['date'] = nw['dt'].dt.normalize().dt.tz_localize(None)
        nw_daily_agg = {'news_count_today': ('title', 'count')}
        if 'sentiment_score' in nw.columns:
            nw_daily_agg['news_sentiment_today'] = ('sentiment_score', 'mean')
        nw_daily = nw.groupby('date').agg(**nw_daily_agg)
        for col in nw_daily.columns:
            s = nw_daily[col]
            s.index = s.index.normalize()
            mapped = s.reindex(df_date_idx)
            mapped.index = _cpu_idx
            out[col] = mapped  # NaN if no news

        # Gematria (all 6 ciphers) + sentiment on headlines — GPU BATCH (zero .apply())
        try:
            # All 6 gematria ciphers + DRs + flags on headline text — single GPU batch call
            # Use pre-computed columns if available, but still need all 6 for flags
            _nw_gem = gematria_gpu_batch(nw['title'], prefix='headline_gem')
            nw['headline_gem_ord'] = _nw_gem['headline_gem_ordinal'].values
            nw['headline_gem_rev'] = _nw_gem['headline_gem_reverse'].values
            nw['headline_gem_red'] = _nw_gem['headline_gem_reduction'].values
            nw['headline_gem_eng'] = _nw_gem['headline_gem_english'].values
            nw['headline_gem_jew'] = _nw_gem['headline_gem_jewish'].values
            nw['headline_gem_sat'] = _nw_gem['headline_gem_satanic'].values
            nw['headline_gem_chal'] = _nw_gem['headline_gem_chaldean'].values
            nw['headline_gem_alb'] = _nw_gem['headline_gem_albam'].values

            # Override with pre-computed if available (trust stored values)
            if 'title_gematria_ordinal' in nw.columns:
                _precomp = pd.to_numeric(nw['title_gematria_ordinal'], errors='coerce')
                _mask = _precomp.notna()
                nw.loc[_mask, 'headline_gem_ord'] = _precomp[_mask].values
            if 'title_gematria_reverse' in nw.columns:
                _precomp = pd.to_numeric(nw['title_gematria_reverse'], errors='coerce')
                _mask = _precomp.notna()
                nw.loc[_mask, 'headline_gem_rev'] = _precomp[_mask].values
            if 'title_gematria_reduction' in nw.columns:
                _precomp = pd.to_numeric(nw['title_gematria_reduction'], errors='coerce')
                _mask = _precomp.notna()
                nw.loc[_mask, 'headline_gem_red'] = _precomp[_mask].values

            # Digital roots — vectorized
            for _gem_sfx, _gem_col in [('ord', 'headline_gem_ord'), ('rev', 'headline_gem_rev'),
                                       ('eng', 'headline_gem_eng'), ('jew', 'headline_gem_jew'),
                                       ('sat', 'headline_gem_sat'),
                                       ('chal', 'headline_gem_chal'), ('alb', 'headline_gem_alb')]:
                _gem_vals = nw[_gem_col].values
                _gem_mask = pd.notna(_gem_vals)
                _gem_safe = np.where(_gem_mask, np.nan_to_num(_gem_vals, nan=0), 0).astype(np.int64)
                _gem_dr = digital_root_vec(_gem_safe)
                nw[f'headline_gem_dr_{_gem_sfx}'] = np.where(_gem_mask, _gem_dr, np.nan)

            # Flags — already computed by GPU batch
            nw['headline_is_caution'] = _nw_gem['headline_gem_is_caution'].values
            nw['headline_is_pump'] = _nw_gem['headline_gem_is_pump'].values
            nw['headline_is_btc_energy'] = _nw_gem['headline_gem_is_btc_energy'].values

            # Sentiment — single GPU batch call
            _nw_sent = sentiment_gpu_batch(nw['title'], prefix='nw_sent')
            nw['headline_sentiment'] = _nw_sent['nw_sent_score'].values
            nw['headline_caps'] = _nw_sent['nw_sent_has_caps'].values
            nw['headline_urgency'] = _nw_sent['nw_sent_urgency'].values
            nw['headline_is_bull'] = (nw['headline_sentiment'] > 0).astype(int)
            nw['headline_is_bear'] = (nw['headline_sentiment'] < 0).astype(int)

            # News source gematria — GPU batch (if source column exists)
            has_source = False
            for src_col in ['source', 'source_name', 'publisher']:
                if src_col in nw.columns:
                    _src_gem = gematria_gpu_batch(nw[src_col], prefix='news_source_gem')
                    nw['news_source_gem_ord'] = _src_gem['news_source_gem_ordinal'].values
                    nw['news_source_gem_rev'] = _src_gem['news_source_gem_reverse'].values
                    nw['news_source_gem_red'] = _src_gem['news_source_gem_reduction'].values
                    nw['news_source_gem_eng'] = _src_gem['news_source_gem_english'].values
                    nw['news_source_gem_jew'] = _src_gem['news_source_gem_jewish'].values
                    nw['news_source_gem_sat'] = _src_gem['news_source_gem_satanic'].values
                    nw['news_source_gem_chal'] = _src_gem['news_source_gem_chaldean'].values
                    nw['news_source_gem_alb'] = _src_gem['news_source_gem_albam'].values
                    nw['news_source_gem_dr'] = _src_gem['news_source_gem_dr_ordinal'].values
                    has_source = True
                    break

            # Aggregation dict -- all 8 cipher means + DR modes
            nw_gem_agg = {
                # 6 cipher means
                'news_gem_ordinal_mean': ('headline_gem_ord', 'mean'),
                'news_gem_reverse_mean': ('headline_gem_rev', 'mean'),
                'news_gem_reduction_mean': ('headline_gem_red', 'mean'),
                'news_gem_english_mean': ('headline_gem_eng', 'mean'),
                'news_gem_jewish_mean': ('headline_gem_jew', 'mean'),
                'news_gem_satanic_mean': ('headline_gem_sat', 'mean'),
                'news_gem_chaldean_mean': ('headline_gem_chal', 'mean'),
                'news_gem_albam_mean': ('headline_gem_alb', 'mean'),
                # DR modes
                'news_gem_dr_ord_mode': ('headline_gem_dr_ord', _mode_or_nan),
                'news_gem_dr_rev_mode': ('headline_gem_dr_rev', _mode_or_nan),
                'news_gem_dr_eng_mode': ('headline_gem_dr_eng', _mode_or_nan),
                'news_gem_dr_jew_mode': ('headline_gem_dr_jew', _mode_or_nan),
                'news_gem_dr_sat_mode': ('headline_gem_dr_sat', _mode_or_nan),
                'news_gem_dr_chal_mode': ('headline_gem_dr_chal', _mode_or_nan),
                'news_gem_dr_alb_mode': ('headline_gem_dr_alb', _mode_or_nan),
                # Flags
                'news_gem_caution': ('headline_is_caution', 'max'),
                'news_gem_pump': ('headline_is_pump', 'max'),
                'news_gem_btc_energy': ('headline_is_btc_energy', 'max'),
                # Sentiment (expanded)
                'headline_sentiment_mean': ('headline_sentiment', 'mean'),
                'news_sentiment_min': ('headline_sentiment', 'min'),
                'news_sentiment_max': ('headline_sentiment', 'max'),
                'headline_caps_any': ('headline_caps', 'max'),
                'news_caps_count': ('headline_caps', 'sum'),
                'news_urgency_mean': ('headline_urgency', 'mean'),
                'news_bull_count': ('headline_is_bull', 'sum'),
                'news_bear_count': ('headline_is_bear', 'sum'),
                'news_article_count': ('title', 'count'),
            }
            # Legacy aliases
            nw_gem_agg['headline_gem_ord_mean'] = ('headline_gem_ord', 'mean')
            nw_gem_agg['headline_gem_dr_mode'] = ('headline_gem_dr_ord', _mode_or_nan)
            nw_gem_agg['headline_caution_any'] = ('headline_is_caution', 'max')

            # Source gematria aggregations
            if has_source:
                nw_gem_agg['news_source_gem_ordinal_mean'] = ('news_source_gem_ord', 'mean')
                nw_gem_agg['news_source_gem_reverse_mean'] = ('news_source_gem_rev', 'mean')
                nw_gem_agg['news_source_gem_reduction_mean'] = ('news_source_gem_red', 'mean')
                nw_gem_agg['news_source_gem_english_mean'] = ('news_source_gem_eng', 'mean')
                nw_gem_agg['news_source_gem_jewish_mean'] = ('news_source_gem_jew', 'mean')
                nw_gem_agg['news_source_gem_satanic_mean'] = ('news_source_gem_sat', 'mean')
                nw_gem_agg['news_source_gem_chaldean_mean'] = ('news_source_gem_chal', 'mean')
                nw_gem_agg['news_source_gem_albam_mean'] = ('news_source_gem_alb', 'mean')
                nw_gem_agg['news_source_gem_dr_mode'] = ('news_source_gem_dr', _mode_or_nan)

            nw_agg = nw.groupby('bucket').agg(**nw_gem_agg).reset_index()

            bucket_map_nw = nw_agg.set_index('bucket')
            for col in nw_agg.columns:
                if col == 'bucket':
                    continue
                mapped = df_buckets.map(bucket_map_nw[col].to_dict())
                out[col] = mapped  # NaN if no news in bucket

            # Derived: divergence = bull_count - bear_count
            if 'news_bull_count' in out.columns and 'news_bear_count' in out.columns:
                out['news_sentiment_divergence'] = out['news_bull_count'] - out['news_bear_count']
        except Exception:
            pass

    # ===========================================================
    # CROSS-SOURCE SENTIMENT (tweet vs news alignment/conflict)
    # ===========================================================
    tw_mean = out.get('tweet_sentiment_mean')
    nw_mean = out.get('headline_sentiment_mean')
    if tw_mean is not None and nw_mean is not None:
        both_valid = tw_mean.notna() & nw_mean.notna()
        agree = ((tw_mean > 0) & (nw_mean > 0)) | ((tw_mean < 0) & (nw_mean < 0))
        disagree = ((tw_mean > 0) & (nw_mean < 0)) | ((tw_mean < 0) & (nw_mean > 0))
        out['sentiment_alignment'] = np.where(both_valid, agree.astype(int), np.nan)
        out['sentiment_conflict'] = np.where(both_valid, disagree.astype(int), np.nan)
    else:
        out['sentiment_alignment'] = np.nan
        out['sentiment_conflict'] = np.nan

    # ===========================================================
    # SPORTS FEATURES
    # ===========================================================
    if sports_df is not None:
        games_df = sports_df.get('games', pd.DataFrame()) if isinstance(sports_df, dict) else sports_df
        horses_df = sports_df.get('horse_races', pd.DataFrame()) if isinstance(sports_df, dict) else pd.DataFrame()

        if len(games_df) > 0:
            gm = games_df.copy()
            gm['date'] = pd.to_datetime(gm['date'], errors='coerce')

            # Compute all 6 ciphers on winner name — GPU batch
            _w_gem = gematria_gpu_batch(gm['winner'], prefix='winner_gem')
            gm['winner_gem_ordinal'] = _w_gem['winner_gem_ordinal'].values
            gm['winner_gem_reverse'] = _w_gem['winner_gem_reverse'].values
            gm['winner_gem_reduction'] = _w_gem['winner_gem_reduction'].values
            gm['winner_gem_english'] = _w_gem['winner_gem_english'].values
            gm['winner_gem_jewish'] = _w_gem['winner_gem_jewish'].values
            gm['winner_gem_satanic'] = _w_gem['winner_gem_satanic'].values
            gm['winner_gem_chaldean'] = _w_gem['winner_gem_chaldean'].values
            gm['winner_gem_albam'] = _w_gem['winner_gem_albam'].values
            gm['winner_gem_dr'] = _w_gem['winner_gem_dr_ordinal'].values

            # Loser gematria (derive loser from home/away) — GPU batch
            if 'home_team' in gm.columns and 'away_team' in gm.columns:
                gm['loser'] = np.where(gm['winner'].values == gm['home_team'].values,
                                        gm['away_team'].values, gm['home_team'].values)
                _l_gem = gematria_gpu_batch(gm['loser'], prefix='loser_gem')
                gm['loser_gem_ordinal'] = _l_gem['loser_gem_ordinal'].values
                gm['loser_gem_reverse'] = _l_gem['loser_gem_reverse'].values
                gm['loser_gem_reduction'] = _l_gem['loser_gem_reduction'].values
                gm['loser_gem_english'] = _l_gem['loser_gem_english'].values
                gm['loser_gem_jewish'] = _l_gem['loser_gem_jewish'].values
                gm['loser_gem_satanic'] = _l_gem['loser_gem_satanic'].values
                gm['loser_gem_chaldean'] = _l_gem['loser_gem_chaldean'].values
                gm['loser_gem_albam'] = _l_gem['loser_gem_albam'].values
                gm['loser_gem_dr'] = _l_gem['loser_gem_dr_ordinal'].values

            # Venue gematria — GPU batch
            if 'venue' in gm.columns:
                _v_gem = gematria_gpu_batch(gm['venue'], prefix='venue_gem')
                gm['venue_gem_ordinal'] = _v_gem['venue_gem_ordinal'].values
                gm['venue_gem_reverse'] = _v_gem['venue_gem_reverse'].values
                gm['venue_gem_reduction'] = _v_gem['venue_gem_reduction'].values
                gm['venue_gem_english'] = _v_gem['venue_gem_english'].values
                gm['venue_gem_jewish'] = _v_gem['venue_gem_jewish'].values
                gm['venue_gem_satanic'] = _v_gem['venue_gem_satanic'].values
                gm['venue_gem_chaldean'] = _v_gem['venue_gem_chaldean'].values
                gm['venue_gem_albam'] = _v_gem['venue_gem_albam'].values
                gm['venue_gem_dr'] = _v_gem['venue_gem_dr_ordinal'].values

            # Score DRs — vectorized
            for sc_col in ['score_total', 'score_diff']:
                if sc_col in gm.columns:
                    gm[sc_col] = pd.to_numeric(gm[sc_col], errors='coerce')
                    dr_col = f'{sc_col}_dr'
                    if dr_col not in gm.columns:
                        _sc_vals = gm[sc_col].values
                        _sc_notna = pd.notna(_sc_vals)
                        _sc_int = np.where(_sc_notna & (_sc_vals > 0), np.nan_to_num(_sc_vals, nan=0).astype(np.int64), 0)
                        _sc_dr = np.where(_sc_int > 0, digital_root_vec(_sc_int), 0)
                        gm[dr_col] = np.where(_sc_notna, _sc_dr, np.nan)
                    else:
                        gm[dr_col] = pd.to_numeric(gm[dr_col], errors='coerce')

            # Build aggregation dict
            sp_agg = {
                # Winner 6 cipher means
                'sport_winner_gem_ordinal_mean': ('winner_gem_ordinal', 'mean'),
                'sport_winner_gem_reverse_mean': ('winner_gem_reverse', 'mean'),
                'sport_winner_gem_reduction_mean': ('winner_gem_reduction', 'mean'),
                'sport_winner_gem_english_mean': ('winner_gem_english', 'mean'),
                'sport_winner_gem_jewish_mean': ('winner_gem_jewish', 'mean'),
                'sport_winner_gem_satanic_mean': ('winner_gem_satanic', 'mean'),
                'sport_winner_gem_chaldean_mean': ('winner_gem_chaldean', 'mean'),
                'sport_winner_gem_albam_mean': ('winner_gem_albam', 'mean'),
                'sport_winner_gem_dr_mode': ('winner_gem_dr', _mode_or_nan),
                # Score DRs
                'sport_score_dr_mode': ('score_dr', _mode_or_nan) if 'score_dr' in gm.columns else ('winner_gem_dr', _mode_or_nan),
                # Game counts + flags
                'sport_upset_count': ('is_upset', 'sum'),
                'sport_overtime_count': ('is_overtime', 'sum'),
                'sport_games_today': ('winner', 'count'),
            }

            # Loser aggregations
            if 'loser_gem_ordinal' in gm.columns:
                sp_agg['sport_loser_gem_ordinal_mean'] = ('loser_gem_ordinal', 'mean')
                sp_agg['sport_loser_gem_reverse_mean'] = ('loser_gem_reverse', 'mean')
                sp_agg['sport_loser_gem_reduction_mean'] = ('loser_gem_reduction', 'mean')
                sp_agg['sport_loser_gem_english_mean'] = ('loser_gem_english', 'mean')
                sp_agg['sport_loser_gem_jewish_mean'] = ('loser_gem_jewish', 'mean')
                sp_agg['sport_loser_gem_satanic_mean'] = ('loser_gem_satanic', 'mean')
                sp_agg['sport_loser_gem_chaldean_mean'] = ('loser_gem_chaldean', 'mean')
                sp_agg['sport_loser_gem_albam_mean'] = ('loser_gem_albam', 'mean')
                sp_agg['sport_loser_gem_dr_mode'] = ('loser_gem_dr', _mode_or_nan)

            # Venue aggregations
            if 'venue_gem_ordinal' in gm.columns:
                sp_agg['sport_venue_gem_ordinal_mean'] = ('venue_gem_ordinal', 'mean')
                sp_agg['sport_venue_gem_reverse_mean'] = ('venue_gem_reverse', 'mean')
                sp_agg['sport_venue_gem_reduction_mean'] = ('venue_gem_reduction', 'mean')
                sp_agg['sport_venue_gem_english_mean'] = ('venue_gem_english', 'mean')
                sp_agg['sport_venue_gem_jewish_mean'] = ('venue_gem_jewish', 'mean')
                sp_agg['sport_venue_gem_satanic_mean'] = ('venue_gem_satanic', 'mean')
                sp_agg['sport_venue_gem_chaldean_mean'] = ('venue_gem_chaldean', 'mean')
                sp_agg['sport_venue_gem_albam_mean'] = ('venue_gem_albam', 'mean')
                sp_agg['sport_venue_gem_dr_mode'] = ('venue_gem_dr', _mode_or_nan)

            # Score total/diff DR aggregations
            if 'score_total_dr' in gm.columns:
                sp_agg['sport_score_total_dr_mode'] = ('score_total_dr', _mode_or_nan)
            if 'score_diff_dr' in gm.columns:
                sp_agg['sport_score_diff_dr_mode'] = ('score_diff_dr', _mode_or_nan)

            # Legacy alias
            sp_agg['sport_winner_gem_dr_mode_legacy'] = ('winner_gem_dr', _mode_or_nan)

            sp_daily = gm.groupby(gm['date'].dt.date).agg(**sp_agg)

            # Rename legacy back
            if 'sport_winner_gem_dr_mode_legacy' in sp_daily.columns:
                sp_daily.drop(columns=['sport_winner_gem_dr_mode_legacy'], inplace=True)

            df_dates = _cpu_idx.date
            for col in sp_daily.columns:
                mapped = pd.Series(df_dates).map(sp_daily[col].to_dict())
                mapped.index = _cpu_idx
                out[col] = mapped  # NaN if no games

        if len(horses_df) > 0:
            hr = horses_df.copy()
            hr['date'] = pd.to_datetime(hr['date'], errors='coerce')

            # Winner horse gematria -- GPU batch
            _h_gem = gematria_gpu_batch(hr['winner_horse'], prefix='horse_gem')
            hr['horse_gem_ordinal'] = _h_gem['horse_gem_ordinal'].values
            hr['horse_gem_reverse'] = _h_gem['horse_gem_reverse'].values
            hr['horse_gem_reduction'] = _h_gem['horse_gem_reduction'].values
            hr['horse_gem_english'] = _h_gem['horse_gem_english'].values
            hr['horse_gem_jewish'] = _h_gem['horse_gem_jewish'].values
            hr['horse_gem_satanic'] = _h_gem['horse_gem_satanic'].values
            hr['horse_gem_chaldean'] = _h_gem['horse_gem_chaldean'].values
            hr['horse_gem_albam'] = _h_gem['horse_gem_albam'].values
            hr['horse_gem_dr'] = _h_gem['horse_gem_dr_ordinal'].values

            # Jockey gematria -- GPU batch
            if 'winner_jockey' in hr.columns:
                _j_gem = gematria_gpu_batch(hr['winner_jockey'], prefix='jockey_gem')
                hr['jockey_gem_ordinal_c'] = _j_gem['jockey_gem_ordinal'].values
                hr['jockey_gem_reverse'] = _j_gem['jockey_gem_reverse'].values
                hr['jockey_gem_reduction'] = _j_gem['jockey_gem_reduction'].values
                hr['jockey_gem_english'] = _j_gem['jockey_gem_english'].values
                hr['jockey_gem_jewish'] = _j_gem['jockey_gem_jewish'].values
                hr['jockey_gem_satanic'] = _j_gem['jockey_gem_satanic'].values
                hr['jockey_gem_chaldean'] = _j_gem['jockey_gem_chaldean'].values
                hr['jockey_gem_albam'] = _j_gem['jockey_gem_albam'].values
                hr['jockey_gem_dr'] = _j_gem['jockey_gem_dr_ordinal'].values

            # Trainer gematria -- GPU batch
            if 'winner_trainer' in hr.columns:
                _t_gem = gematria_gpu_batch(hr['winner_trainer'], prefix='trainer_gem')
                hr['trainer_gem_ordinal'] = _t_gem['trainer_gem_ordinal'].values
                hr['trainer_gem_reverse'] = _t_gem['trainer_gem_reverse'].values
                hr['trainer_gem_reduction'] = _t_gem['trainer_gem_reduction'].values
                hr['trainer_gem_english'] = _t_gem['trainer_gem_english'].values
                hr['trainer_gem_jewish'] = _t_gem['trainer_gem_jewish'].values
                hr['trainer_gem_satanic'] = _t_gem['trainer_gem_satanic'].values
                hr['trainer_gem_chaldean'] = _t_gem['trainer_gem_chaldean'].values
                hr['trainer_gem_albam'] = _t_gem['trainer_gem_albam'].values
                hr['trainer_gem_dr'] = _t_gem['trainer_gem_dr_ordinal'].values

            # Build horse aggregation dict
            hr_agg = {
                # Winner horse 6 ciphers
                'horse_winner_gem_ordinal_mean': ('horse_gem_ordinal', 'mean'),
                'horse_winner_gem_reverse_mean': ('horse_gem_reverse', 'mean'),
                'horse_winner_gem_reduction_mean': ('horse_gem_reduction', 'mean'),
                'horse_winner_gem_english_mean': ('horse_gem_english', 'mean'),
                'horse_winner_gem_jewish_mean': ('horse_gem_jewish', 'mean'),
                'horse_winner_gem_satanic_mean': ('horse_gem_satanic', 'mean'),
                'horse_winner_gem_chaldean_mean': ('horse_gem_chaldean', 'mean'),
                'horse_winner_gem_albam_mean': ('horse_gem_albam', 'mean'),
                'horse_winner_gem_dr_mode': ('horse_gem_dr', _mode_or_nan),
                'horse_races_today': ('winner_horse', 'count'),
            }

            # Jockey aggregations
            if 'jockey_gem_ordinal_c' in hr.columns:
                hr_agg['horse_jockey_gem_ordinal_mean'] = ('jockey_gem_ordinal_c', 'mean')
                hr_agg['horse_jockey_gem_reverse_mean'] = ('jockey_gem_reverse', 'mean')
                hr_agg['horse_jockey_gem_reduction_mean'] = ('jockey_gem_reduction', 'mean')
                hr_agg['horse_jockey_gem_english_mean'] = ('jockey_gem_english', 'mean')
                hr_agg['horse_jockey_gem_jewish_mean'] = ('jockey_gem_jewish', 'mean')
                hr_agg['horse_jockey_gem_satanic_mean'] = ('jockey_gem_satanic', 'mean')
                hr_agg['horse_jockey_gem_chaldean_mean'] = ('jockey_gem_chaldean', 'mean')
                hr_agg['horse_jockey_gem_albam_mean'] = ('jockey_gem_albam', 'mean')
                hr_agg['horse_jockey_gem_dr_mode'] = ('jockey_gem_dr', _mode_or_nan)

            # Trainer aggregations
            if 'trainer_gem_ordinal' in hr.columns:
                hr_agg['horse_trainer_gem_ordinal_mean'] = ('trainer_gem_ordinal', 'mean')
                hr_agg['horse_trainer_gem_reverse_mean'] = ('trainer_gem_reverse', 'mean')
                hr_agg['horse_trainer_gem_reduction_mean'] = ('trainer_gem_reduction', 'mean')
                hr_agg['horse_trainer_gem_english_mean'] = ('trainer_gem_english', 'mean')
                hr_agg['horse_trainer_gem_jewish_mean'] = ('trainer_gem_jewish', 'mean')
                hr_agg['horse_trainer_gem_satanic_mean'] = ('trainer_gem_satanic', 'mean')
                hr_agg['horse_trainer_gem_chaldean_mean'] = ('trainer_gem_chaldean', 'mean')
                hr_agg['horse_trainer_gem_albam_mean'] = ('trainer_gem_albam', 'mean')
                hr_agg['horse_trainer_gem_dr_mode'] = ('trainer_gem_dr', _mode_or_nan)

            # Position/odds DR
            if 'position_dr' in hr.columns:
                hr['position_dr'] = pd.to_numeric(hr['position_dr'], errors='coerce')
                hr_agg['horse_position_dr_mode'] = ('position_dr', _mode_or_nan)
            if 'odds_dr' in hr.columns:
                hr['odds_dr'] = pd.to_numeric(hr['odds_dr'], errors='coerce')
                hr_agg['horse_odds_dr_mode'] = ('odds_dr', _mode_or_nan)

            hr_daily = hr.groupby(hr['date'].dt.date).agg(**hr_agg)

            df_dates = _cpu_idx.date
            for col in hr_daily.columns:
                mapped = pd.Series(df_dates).map(hr_daily[col].to_dict())
                mapped.index = _cpu_idx
                out[col] = mapped  # NaN if no races

    # ===========================================================
    # ON-CHAIN FEATURES
    # ===========================================================
    if onchain_df is not None:
        # Support both dict format {'daily': df, 'timestamped': df} and plain DataFrame
        if isinstance(onchain_df, dict):
            oc_daily_df = onchain_df.get('daily', pd.DataFrame())
            oc_ts_df = onchain_df.get('timestamped', pd.DataFrame())
        else:
            # Plain DataFrame — try to detect format
            oc_daily_df = onchain_df if ('hash_rate' in onchain_df.columns or
                                          'date' in onchain_df.columns) else pd.DataFrame()
            oc_ts_df = onchain_df if 'timestamp' in onchain_df.columns else pd.DataFrame()

        # Daily on-chain data (blockchain_data style)
        if len(oc_daily_df) > 0:
            oc_daily = oc_daily_df.copy()
            if not isinstance(oc_daily.index, pd.DatetimeIndex):
                if 'date' in oc_daily.columns:
                    oc_daily['date'] = pd.to_datetime(oc_daily['date'], errors='coerce')
                    oc_daily = oc_daily.set_index('date')
            oc_daily.index = oc_daily.index.normalize()
            if oc_daily.index.tz is not None:
                oc_daily.index = oc_daily.index.tz_localize(None)

            for col in ['hash_rate', 'n_transactions', 'difficulty', 'mempool_size', 'miners_revenue']:
                if col in oc_daily.columns:
                    vals = pd.to_numeric(oc_daily[col], errors='coerce').ffill()
                    mapped = vals.reindex(df_date_idx)
                    mapped.index = _cpu_idx
                    out[f'onchain_{col}'] = mapped.ffill()

            if 'onchain_hash_rate' in out.columns:
                bars_per_week = max(1, 7 * 86400 // bucket_seconds)
                out['onchain_hash_rate_roc'] = out['onchain_hash_rate'].pct_change(bars_per_week)
                out['onchain_hash_rate_capitulation'] = (out['onchain_hash_rate_roc'] < -0.10).astype(int)

        # Bucketed on-chain (onchain_data style with timestamps)
        if len(oc_ts_df) > 0 and 'timestamp' in oc_ts_df.columns:
            oc_ts = oc_ts_df.copy()
            oc_ts['ts'] = pd.to_datetime(oc_ts['timestamp'], errors='coerce')
            oc_ts['bucket'] = (oc_ts['ts'].astype(np.int64) // 10**9 // bucket_seconds) * bucket_seconds

            # Base aggregation dict
            agg_dict_oc = {
                'onchain_block_dr': ('block_dr', 'last'),
                'onchain_funding_dr': ('funding_dr', 'last'),
                'onchain_oi_dr': ('oi_dr', 'last'),
                'onchain_fg': ('fear_greed', 'last'),
                'onchain_fg_dr': ('fg_dr', 'last'),
                'onchain_mempool': ('mempool_size', 'last'),
            }
            # Add whale/liquidation/OI/funding columns if available
            if 'whale_volume_btc' in oc_ts.columns:
                agg_dict_oc['onchain_whale_vol'] = ('whale_volume_btc', 'sum')
            if 'liq_long_vol' in oc_ts.columns:
                agg_dict_oc['onchain_liq_long_vol'] = ('liq_long_vol', 'sum')
            if 'liq_short_vol' in oc_ts.columns:
                agg_dict_oc['onchain_liq_short_vol'] = ('liq_short_vol', 'sum')
            if 'liq_long_count' in oc_ts.columns:
                agg_dict_oc['onchain_liq_long_count'] = ('liq_long_count', 'sum')
            if 'liq_short_count' in oc_ts.columns:
                agg_dict_oc['onchain_liq_short_count'] = ('liq_short_count', 'sum')
            if 'open_interest' in oc_ts.columns:
                agg_dict_oc['onchain_oi'] = ('open_interest', 'last')
            if 'funding_rate' in oc_ts.columns:
                agg_dict_oc['onchain_funding_raw'] = ('funding_rate', 'last')
            if 'coinbase_premium' in oc_ts.columns:
                agg_dict_oc['coinbase_premium'] = ('coinbase_premium', 'last')
            # Filter agg_dict to only include columns that exist
            agg_dict_oc = {k: v for k, v in agg_dict_oc.items() if v[0] in oc_ts.columns}
            oc_agg = oc_ts.groupby('bucket').agg(**agg_dict_oc).reset_index()

            bucket_map_oc = oc_agg.set_index('bucket')
            for col in oc_agg.columns:
                if col == 'bucket':
                    continue
                mapped = df_buckets.map(bucket_map_oc[col].to_dict())
                out[col] = mapped  # NaN if no data

    # ===========================================================
    # MACRO FEATURES
    # ===========================================================
    cfg = TF_CONFIG.get(_tf_from_bucket(bucket_seconds), TF_CONFIG['1h'])

    if macro_df is not None and len(macro_df) > 0:
        mc = macro_df.copy()
        if not isinstance(mc.index, pd.DatetimeIndex):
            mc['date'] = pd.to_datetime(mc['date'], errors='coerce')
            mc = mc.set_index('date')
        mc.index = mc.index.normalize()

        macro_tickers = ['dxy', 'gold', 'spx', 'vix', 'us10y', 'nasdaq', 'russell',
                         'oil', 'silver', 'mstr', 'coin', 'hyg', 'tlt', 'ibit']

        for ticker in macro_tickers:
            if ticker in mc.columns:
                vals = pd.to_numeric(mc[ticker], errors='coerce').ffill()
                mapped = vals.reindex(df_date_idx)
                mapped.index = _cpu_idx
                mapped = mapped.ffill()
                out[f'macro_{ticker}'] = mapped
                out[f'macro_{ticker}_roc5d'] = mapped.pct_change(cfg['macro_roc_short'])
                out[f'macro_{ticker}_roc20d'] = mapped.pct_change(cfg['macro_roc_long'])

        for corr_ticker in ['dxy', 'gold', 'spx', 'vix']:
            if f'macro_{corr_ticker}' in out.columns:
                btc_ret = c.pct_change()
                macro_ret = out[f'macro_{corr_ticker}'].pct_change()
                out[f'btc_{corr_ticker}_corr'] = btc_ret.rolling(cfg['corr_window']).corr(macro_ret)

        # Macro DR features (if available in db with pre-computed DRs)
        if 'timestamp' in macro_df.columns:
            # Has bucketed macro with DR columns
            mc_ts = macro_df.copy()
            mc_ts['ts'] = pd.to_datetime(mc_ts['timestamp'], errors='coerce')
            mc_ts['bucket'] = (mc_ts['ts'].astype(np.int64) // 10**9 // bucket_seconds) * bucket_seconds

            dr_cols = [c for c in mc_ts.columns if c.endswith('_dr')]
            if dr_cols:
                agg_dict_mc = {f'macro_{c}': (c, 'last') for c in dr_cols}
                mc_agg = mc_ts.groupby('bucket').agg(**agg_dict_mc).reset_index()
                bucket_map_mc = mc_agg.set_index('bucket')
                for col in mc_agg.columns:
                    if col == 'bucket':
                        continue
                    mapped = df_buckets.map(bucket_map_mc[col].to_dict())
                    out[col] = mapped  # NaN if no data

    # ===========================================================
    # FEAR & GREED
    # ===========================================================
    fg_df = astro_cache.get('fear_greed', pd.DataFrame()) if astro_cache else pd.DataFrame()
    if len(fg_df) > 0:
        fg_vals = pd.to_numeric(fg_df['value'], errors='coerce') if 'value' in fg_df.columns else pd.Series(dtype=float)
        if len(fg_vals) > 0:
            fg_vals.index = fg_vals.index.normalize()
            fg_mapped = fg_vals.reindex(df_date_idx)
            fg_mapped.index = _cpu_idx
            out['fear_greed'] = fg_mapped.ffill()
            out['fg_roc'] = out['fear_greed'].pct_change(cfg['fg_roc_bars'])
            out['fg_extreme_fear'] = (out['fear_greed'] < 20).astype(int)
            out['fg_extreme_greed'] = (out['fear_greed'] > 80).astype(int)

            fg_roll = max(cfg['corr_window'], 30)
            fg_norm = (out['fear_greed'] - out['fear_greed'].rolling(fg_roll).mean()) / out['fear_greed'].rolling(fg_roll).std()
            price_norm = (c - c.rolling(fg_roll).mean()) / c.rolling(fg_roll).std()
            out['fg_vs_price_div'] = fg_norm - price_norm

            # Fear/greed lagged
            for lag in cfg['fg_lag_bars']:
                out[f'fear_greed_lag{lag}'] = out['fear_greed'].shift(lag)

    # ===========================================================
    # GOOGLE TRENDS
    # ===========================================================
    gt_df = astro_cache.get('google_trends', pd.DataFrame()) if astro_cache else pd.DataFrame()
    if len(gt_df) > 0 and 'interest_score' in gt_df.columns:
        gt_vals = pd.to_numeric(gt_df['interest_score'], errors='coerce')
        gt_vals.index = gt_vals.index.normalize()
        gt_mapped = gt_vals.reindex(df_date_idx)
        gt_mapped.index = _cpu_idx
        gt_mapped = gt_mapped.ffill()
        out['gtrends_interest'] = gt_mapped
        out['gtrends_interest_high'] = (gt_mapped > 75).astype(int)

    # ===========================================================
    # FUNDING RATES
    # ===========================================================
    fr_df = astro_cache.get('funding_daily', pd.DataFrame()) if astro_cache else pd.DataFrame()
    if len(fr_df) > 0:
        fr_col = 'avg_funding_rate' if 'avg_funding_rate' in fr_df.columns else fr_df.columns[0]
        fr = pd.to_numeric(fr_df[fr_col], errors='coerce')
        fr.index = fr.index.normalize()
        fr_mapped = fr.reindex(df_date_idx)
        fr_mapped.index = _cpu_idx
        out['funding_rate'] = fr_mapped.ffill()
        out['funding_rate_high'] = (out['funding_rate'] > 0.001).astype(int)
        out['funding_rate_neg'] = (out['funding_rate'] < 0).astype(int)

        # --- Funding Rate Z-Score Regime ---
        fr_series = out['funding_rate']
        bars_per_30d = max(1, 30 * 86400 // bucket_seconds)
        bars_per_90d = max(1, 90 * 86400 // bucket_seconds)
        fr_mean_30 = fr_series.rolling(bars_per_30d, min_periods=max(1, bars_per_30d // 4)).mean()
        fr_std_30 = fr_series.rolling(bars_per_30d, min_periods=max(1, bars_per_30d // 4)).std()
        fr_mean_90 = fr_series.rolling(bars_per_90d, min_periods=max(1, bars_per_90d // 4)).mean()
        fr_std_90 = fr_series.rolling(bars_per_90d, min_periods=max(1, bars_per_90d // 4)).std()
        out['funding_zscore_30d'] = ((fr_series - fr_mean_30) / fr_std_30.replace(0, np.nan))
        out['funding_zscore_90d'] = ((fr_series - fr_mean_90) / fr_std_90.replace(0, np.nan))
        # Funding regime: -2=extreme neg, -1=neg, 0=neutral, 1=pos, 2=extreme pos
        zscore_30 = out['funding_zscore_30d']
        out['funding_regime'] = 0
        out.loc[zscore_30 > 2, 'funding_regime'] = 2
        out.loc[(zscore_30 > 0.5) & (zscore_30 <= 2), 'funding_regime'] = 1
        out.loc[(zscore_30 < -0.5) & (zscore_30 >= -2), 'funding_regime'] = -1
        out.loc[zscore_30 < -2, 'funding_regime'] = -2

    # ===========================================================
    # INSTITUTIONAL DIRECTIONAL SIGNALS (Tier 1)
    # ===========================================================

    # --- Liquidation Cascade Signals ---
    if 'onchain_liq_long_vol' in out.columns and 'onchain_liq_short_vol' in out.columns:
        liq_l = out['onchain_liq_long_vol']
        liq_s = out['onchain_liq_short_vol']
        liq_total = liq_l + liq_s
        # Ratio: >1 means longs getting flushed (bearish), <1 means shorts squeezed (bullish)
        out['liq_long_short_ratio'] = (liq_l / liq_s.replace(0, np.nan))
        # Spike detection: z-score of total liquidation volume
        liq_mean = liq_total.rolling(20, min_periods=5).mean()
        liq_std = liq_total.rolling(20, min_periods=5).std()
        out['liq_total_spike'] = ((liq_total - liq_mean) / liq_std.replace(0, np.nan))
        # Asymmetry: one-sided liquidation = forced directional move
        out['liq_asymmetry'] = ((liq_l - liq_s).abs() / liq_total.replace(0, np.nan))
        # Post-liquidation direction: after spike, mark next 5 bars
        is_spike = out['liq_total_spike'] > 2.0
        spike_decay = is_spike.astype(float)
        for shift_n in range(1, 6):
            spike_decay = spike_decay + is_spike.shift(shift_n).fillna(0).astype(float) * (1 - shift_n / 6)
        out['post_liq_flag'] = (spike_decay > 0).astype(int)

    # --- OI + Price Divergence ---
    if 'onchain_oi' in out.columns:
        oi_series = out['onchain_oi']
        oi_pct = oi_series.pct_change(5)
        out['oi_change_pct'] = oi_pct
        # OI-price divergence: price up + OI down = weak rally
        price_dir = c.pct_change(5)
        out['oi_price_divergence'] = np.where(
            (price_dir > 0) & (oi_pct < 0), -1,   # weak rally
            np.where(
                (price_dir < 0) & (oi_pct < 0), 1,  # capitulation (bullish)
                np.where(
                    (price_dir > 0) & (oi_pct > 0), 0.5,  # healthy trend
                    np.where(
                        (price_dir < 0) & (oi_pct > 0), -0.5,  # shorts building
                        0
                    )
                )
            )
        )
        # OI x Funding combo: 4 regimes
        if 'funding_rate' in out.columns:
            fund_pos = (out['funding_rate'] > 0).astype(int)
            oi_rising = (oi_pct > 0).astype(int)
            # 0=unwind-short, 1=build-long, 2=build-short, 3=unwind-long
            out['oi_funding_combo'] = fund_pos * 2 + oi_rising - 1
            out['funding_x_oi_trend'] = out['funding_rate'] * oi_pct

    # --- Whale Flow Signals ---
    if 'onchain_whale_vol' in out.columns:
        whale = out['onchain_whale_vol']
        whale_mean = whale.rolling(30, min_periods=5).mean()
        whale_std = whale.rolling(30, min_periods=5).std()
        out['whale_vol_zscore'] = ((whale - whale_mean) / whale_std.replace(0, np.nan))
        # Whale spike + price direction = accumulation vs distribution
        whale_spike = (out['whale_vol_zscore'] > 1.5).astype(int)
        price_5bar = c.pct_change(5)
        out['whale_vol_x_price_dir'] = whale_spike * np.sign(price_5bar)
        # Whale volume trend (rising = institutional activity) — closed-form OLS slope
        out['whale_vol_trend'] = _rolling_slope_vectorized(whale.values, window=5)

    # --- Coinbase Premium Index ---
    if 'coinbase_premium' in out.columns:
        cb_prem = out['coinbase_premium']
        out['coinbase_premium_positive'] = (cb_prem > 0).astype(int)
        out['coinbase_premium_zscore'] = (
            (cb_prem - cb_prem.rolling(20, min_periods=5).mean()) /
            cb_prem.rolling(20, min_periods=5).std().replace(0, np.nan)
        )

    # --- Macro Correlation Regime ---
    bars_per_30d_macro = max(1, 30 * 86400 // bucket_seconds)
    bars_per_90d_macro = max(1, 90 * 86400 // bucket_seconds)
    btc_ret = c.pct_change()

    corr_regime_cols = []
    for ticker in ['spx', 'dxy', 'vix']:
        col_name = f'macro_{ticker}'
        if col_name in out.columns:
            macro_ret = out[col_name].pct_change()
            min_p_30 = min(bars_per_30d_macro, max(3, bars_per_30d_macro // 4))
            min_p_90 = min(bars_per_90d_macro, max(3, bars_per_90d_macro // 4))
            corr_30 = btc_ret.rolling(bars_per_30d_macro, min_periods=min_p_30).corr(macro_ret)
            corr_90 = btc_ret.rolling(bars_per_90d_macro, min_periods=min_p_90).corr(macro_ret)
            out[f'btc_{ticker}_corr_30d'] = corr_30
            corr_regime_cols.append(f'btc_{ticker}_corr_30d')
            # Regime shift: correlation flipped sign vs 90d baseline
            out[f'btc_{ticker}_corr_regime_shift'] = (
                (np.sign(corr_30) != np.sign(corr_90)) & corr_90.notna()
            ).astype(int)

    # Macro decorrelation: BTC decorrelating from ALL macro simultaneously
    if len(corr_regime_cols) >= 2:
        abs_corrs = out[corr_regime_cols].abs()
        out['macro_decorrelation'] = (abs_corrs.mean(axis=1) < 0.15).astype(int)

    # --- CVD / Cumulative Delta (from OHLCV taker_buy_volume) ---
    _has_tbv = 'taker_buy_volume' in (list(df.columns) if _gpu else df.columns)
    if _has_tbv:
        tbv = pd.to_numeric(pd.Series(_np(df['taker_buy_volume']), index=_cpu_idx), errors='coerce')
        if tbv.notna().sum() > 100:  # only compute if we have enough data
            taker_sell = v - tbv
            delta_bar = tbv - taker_sell
            out['delta_bar'] = delta_bar
            out['delta_ratio'] = tbv / v.replace(0, np.nan)
            out['cvd'] = delta_bar.cumsum()
            # CVD slope (regression over N bars)
            cvd_window = min(20, max(5, len(df) // 100))
            out['cvd_slope'] = _rolling_slope_vectorized(out['cvd'].values, window=cvd_window)
            # CVD zero cross
            cvd_sign = np.sign(out['cvd'])
            out['cvd_zero_cross'] = (cvd_sign != cvd_sign.shift(1)).astype(int)
            # CVD-price divergence: price new high but CVD not, or vice versa
            price_high_20 = c.rolling(20, min_periods=10).max()
            cvd_high_20 = out['cvd'].rolling(20, min_periods=10).max()
            price_at_high = (c >= price_high_20 * 0.999)
            cvd_at_high = (out['cvd'] >= cvd_high_20 * 0.999)
            # Bearish divergence: price at high, CVD not
            out['cvd_price_divergence'] = np.where(
                price_at_high & ~cvd_at_high, -1,
                np.where(~price_at_high & cvd_at_high, 1, 0)
            )

    # ===========================================================
    # CROSS-FEATURES (gematria match across sources)
    # ===========================================================
    try:
        # Compute date DR and price DR for cross-matching — vectorized
        _cm_date_digits = (_cpu_idx.year * 10000 + _cpu_idx.month * 100 + _cpu_idx.day).values.astype(np.int64)
        _cm_ds = np.zeros(len(_cpu_idx), dtype=np.int64)
        _cm_tmp = _cm_date_digits.copy()
        while np.any(_cm_tmp > 0):
            _cm_ds += _cm_tmp % 10
            _cm_tmp //= 10
        date_dr = pd.Series(digital_root_vec(_cm_ds), index=_cpu_idx)
        _cm_c = np.where(np.isnan(c.values), 0, np.abs(c.values).astype(np.int64))
        price_dr = pd.Series(np.where(np.isnan(c.values), np.nan, digital_root_vec(_cm_c)), index=_cpu_idx)

        # Tweet DR x News DR match (ordinal DR)
        if 'tweet_gem_dr_ord_mode' in out.columns and 'news_gem_dr_ord_mode' in out.columns:
            out['gem_match_tweet_news'] = (
                out['tweet_gem_dr_ord_mode'] == out['news_gem_dr_ord_mode']).astype(int)
        # Legacy alias
        if 'tweet_gem_dr_mode' in out.columns and 'headline_gem_dr_mode' in out.columns:
            out['cross_tweet_news_gem_match'] = (
                out['tweet_gem_dr_mode'] == out['headline_gem_dr_mode']).astype(int)

        # Price DR x Tweet DR match
        if 'tweet_gem_dr_ord_mode' in out.columns:
            out['gem_match_price_tweet'] = (price_dr == out['tweet_gem_dr_ord_mode']).astype(int)
            # Legacy alias
            out['cross_price_tweet_dr_match'] = out['gem_match_price_tweet']

        # Date DR x Sport winner DR match
        if 'sport_winner_gem_dr_mode' in out.columns:
            out['gem_match_date_sport'] = (date_dr == out['sport_winner_gem_dr_mode']).astype(int)

        # Tweet DR x Sport winner DR match
        if 'tweet_gem_dr_ord_mode' in out.columns and 'sport_winner_gem_dr_mode' in out.columns:
            out['gem_match_tweet_sport'] = (
                out['tweet_gem_dr_ord_mode'] == out['sport_winner_gem_dr_mode']).astype(int)

        # Price DR x News DR match
        if 'news_gem_dr_ord_mode' in out.columns:
            out['gem_match_price_news'] = (price_dr == out['news_gem_dr_ord_mode']).astype(int)

        # Date DR x News DR match
        if 'news_gem_dr_ord_mode' in out.columns:
            out['gem_match_date_news'] = (date_dr == out['news_gem_dr_ord_mode']).astype(int)

        # Date DR x Tweet DR match
        if 'tweet_gem_dr_ord_mode' in out.columns:
            out['gem_match_date_tweet'] = (date_dr == out['tweet_gem_dr_ord_mode']).astype(int)

        # Sport winner DR x News DR match
        if 'sport_winner_gem_dr_mode' in out.columns and 'news_gem_dr_ord_mode' in out.columns:
            out['gem_match_sport_news'] = (
                out['sport_winner_gem_dr_mode'] == out['news_gem_dr_ord_mode']).astype(int)

        # Horse winner DR x Tweet DR match
        if 'horse_winner_gem_dr_mode' in out.columns and 'tweet_gem_dr_ord_mode' in out.columns:
            out['gem_match_horse_tweet'] = (
                out['horse_winner_gem_dr_mode'] == out['tweet_gem_dr_ord_mode']).astype(int)

        # Price DR x Sport winner DR match
        if 'sport_winner_gem_dr_mode' in out.columns:
            out['gem_match_price_sport'] = (price_dr == out['sport_winner_gem_dr_mode']).astype(int)

        # Triple match: price DR = tweet DR = news DR
        if 'tweet_gem_dr_ord_mode' in out.columns and 'news_gem_dr_ord_mode' in out.columns:
            out['gem_triple_match'] = (
                (price_dr == out['tweet_gem_dr_ord_mode']) &
                (price_dr == out['news_gem_dr_ord_mode'])).astype(int)

        # Date x price DR match (legacy)
        out['cross_date_price_dr_match'] = (date_dr == price_dr).astype(int)

        # Caution convergence: multiple sources flagging caution simultaneously
        caution_cols = [c for c in ['tweet_gem_caution', 'news_gem_caution'] if c in out.columns]
        if len(caution_cols) >= 2:
            out['gem_caution_convergence'] = out[caution_cols].sum(axis=1)

        # Pump convergence
        pump_cols = [c for c in ['tweet_gem_pump', 'news_gem_pump'] if c in out.columns]
        if len(pump_cols) >= 2:
            out['gem_pump_convergence'] = out[pump_cols].sum(axis=1)

    except Exception:
        pass

    return out





# ============================================================
# COMPUTE EVENT-TIMESTAMP ASTROLOGY FEATURES (~25 features)
# ============================================================
# Astrology computed at the MOMENT each event occurred
# (tweet posted, news published, game played), NOT the candle time.
# Optimized: daily astro cached per date, only planetary_hour per-event.

def _event_astro_daily_cache(dates_array):
    """
    Pre-compute daily astro lookups for a set of unique dates.
    Moon phase, nakshatra, mercury retro change at most daily.
    Returns dict: date -> {moon_phase_day, lunar_sin, lunar_cos,
                           is_full_moon, is_new_moon, nakshatra,
                           nakshatra_nature, nakshatra_guna, key_nakshatra,
                           mercury_retrograde,
                           bazi_stem, bazi_element,
                           tzolkin_tone, tzolkin_sign_idx}
    """
    cache = {}
    for d in dates_array:
        if d in cache or pd.isna(d):
            continue
        try:
            dt = pd.Timestamp(d).to_pydatetime().replace(hour=12)
        except Exception:
            continue
        moon = get_moon_phase(dt)
        vedic = get_vedic(dt)
        western = get_western(dt)
        try:
            bazi = get_bazi(dt)
            bazi_stem = bazi['stem_idx']
            bazi_element = bazi['element_idx']
        except Exception:
            bazi_stem = np.nan
            bazi_element = np.nan
        try:
            tzolkin = get_tzolkin(dt)
            tzolkin_tone = tzolkin['tone']
            tzolkin_sign_idx = tzolkin['sign_idx']
        except Exception:
            tzolkin_tone = np.nan
            tzolkin_sign_idx = np.nan
        cache[d] = {
            'moon_phase_day': moon['phase_day'],
            'lunar_sin': moon['lunar_sin'],
            'lunar_cos': moon['lunar_cos'],
            'is_full_moon': int(moon['is_full_moon']),
            'is_new_moon': int(moon['is_new_moon']),
            'nakshatra': vedic['nakshatra'],
            'nakshatra_nature': vedic['nakshatra_nature'],
            'nakshatra_guna': vedic['nakshatra_guna'],
            'key_nakshatra': vedic['key_nakshatra'],
            'mercury_retrograde': western['mercury_retrograde'],
            'bazi_stem': bazi_stem,
            'bazi_element': bazi_element,
            'tzolkin_tone': tzolkin_tone,
            'tzolkin_sign_idx': tzolkin_sign_idx,
        }
    return cache


def compute_event_astrology(df, tweets_df, news_df, sports_df,
                            bucket_seconds):
    """
    Compute astrology-at-event-timestamp features.

    For each event type (tweets, news, sports games), computes astro
    at the MOMENT the event occurred, then aggregates per candle bucket
    using LAST value.

    Produces ~37 features total across all event types:
      tweet_astro_*  (15 features: moon_phase_day, lunar_sin, lunar_cos,
                      is_full_moon, is_new_moon, nakshatra, nakshatra_nature,
                      nakshatra_guna, key_nakshatra, bazi_stem, bazi_element,
                      tzolkin_tone, tzolkin_sign_idx, mercury_retrograde,
                      planetary_hour_idx)
      news_astro_*   (14 features: same minus mercury_retrograde)
      game_astro_*   (14 features: same minus mercury_retrograde)

    Handles cuDF input by converting to CPU internally.

    Args:
        df: OHLCV DataFrame with DatetimeIndex and optional 'open_time'
        tweets_df: DataFrame with ts_unix column (or None)
        news_df: DataFrame with ts_unix column (or None)
        sports_df: dict with 'games' key -> DataFrame with date col (or None)
        bucket_seconds: bucket size matching timeframe

    Returns:
        DataFrame with event-astrology features, NaN where no events.
    """
    _gpu = _is_gpu(df)
    # Event astrology uses datetime loops + external pandas DataFrames — extract CPU index, avoid full _to_cpu()
    if _gpu:
        _cpu_idx = pd.DatetimeIndex(_np(df.index))
    else:
        _cpu_idx = df.index
    out = pd.DataFrame(index=_cpu_idx)

    # Build candle bucket keys
    if 'open_time' in (list(df.columns) if _gpu else df.columns):
        ts_sec = pd.to_numeric(pd.Series(_np(df['open_time']), index=_cpu_idx), errors='coerce') / 1000
    else:
        ts_sec = pd.Series(_cpu_idx.astype(np.int64) // 10**9, index=_cpu_idx)
    df_buckets = (ts_sec // bucket_seconds * bucket_seconds).astype(int)

    # ------ Helper: process one event source ------
    def _process_events(ev_df, ts_col, prefix, has_mercury=False):
        """
        Compute astro at each event timestamp, aggregate per bucket.
        ts_col: name of unix-timestamp column in ev_df.
        prefix: feature name prefix (e.g. 'tweet_astro', 'news_astro').
        has_mercury: whether to include mercury_retrograde feature.
        """
        ev = ev_df.copy()
        ev['_ts'] = pd.to_numeric(ev[ts_col], errors='coerce')
        ev = ev.dropna(subset=['_ts'])
        if len(ev) == 0:
            return

        ev['_bucket'] = (ev['_ts'] // bucket_seconds) * bucket_seconds
        ev['_dt'] = pd.to_datetime(ev['_ts'], unit='s', errors='coerce', utc=True)
        ev['_date'] = ev['_dt'].dt.date

        # Build daily cache for all unique event dates
        unique_dates = ev['_date'].dropna().unique()
        daily_cache = _event_astro_daily_cache(unique_dates)

        # Map daily astro to each event row
        daily_keys = [
            'moon_phase_day', 'lunar_sin', 'lunar_cos',
            'is_full_moon', 'is_new_moon',
            'nakshatra', 'nakshatra_nature', 'nakshatra_guna', 'key_nakshatra',
            'bazi_stem', 'bazi_element', 'tzolkin_tone', 'tzolkin_sign_idx',
        ]
        if has_mercury:
            daily_keys.append('mercury_retrograde')

        for k in daily_keys:
            ev[f'_astro_{k}'] = ev['_date'].map(
                {d: v.get(k, np.nan) for d, v in daily_cache.items()})

        # Planetary hour per-event — vectorized (hour + dow based)
        # Planetary hour is determined by day-of-week and hour-of-day
        # Each day starts at sunrise (~6am), each planetary hour = 1 clock hour
        # Rulers cycle: Sun(0), Moon(1), Mars(2), Mercury(3), Jupiter(4), Venus(5), Saturn(6)
        # Day rulers: Sun=0, Mon=1, Tue=2, Wed=3, Thu=4, Fri=5, Sat=6
        _day_start_planet = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int32)  # indexed by dow (Mon=0)
        _ts_vals = pd.to_numeric(ev['_ts'], errors='coerce').values
        _ts_valid = ~np.isnan(_ts_vals)
        _dt_series = pd.to_datetime(_ts_vals[_ts_valid], unit='s', utc=True, errors='coerce')
        _ph_result = np.full(len(ev), np.nan)
        if len(_dt_series) > 0:
            _hours = _dt_series.hour.values
            _dows = _dt_series.dayofweek.values  # Mon=0
            # Approximate: planetary hour = (day_start_planet + hour_since_sunrise) % 7
            _hour_offset = np.where(_hours >= 6, _hours - 6, _hours + 18)  # hours since ~6am
            _planet_idx = (_day_start_planet[_dows] + _hour_offset) % 7
            _ph_result[_ts_valid] = _planet_idx
        ev['_astro_planetary_hour_idx'] = _ph_result

        # Aggregate per bucket: use LAST event in each bucket
        astro_cols = [f'_astro_{k}' for k in daily_keys] + ['_astro_planetary_hour_idx']
        agg_dict = {c: 'last' for c in astro_cols}
        ev_agg = ev.groupby('_bucket').agg(agg_dict)

        # Map to candle bars
        feature_names = daily_keys + ['planetary_hour_idx']
        for astro_col, feat_name in zip(astro_cols, feature_names):
            col_name = f'{prefix}_{feat_name}'
            mapped = df_buckets.map(ev_agg[astro_col].to_dict())
            out[col_name] = mapped  # NaN if no events in bucket

    # ======================================================
    # TWEET EVENT ASTROLOGY (with mercury_retrograde)
    # ======================================================
    if tweets_df is not None and len(tweets_df) > 0 and 'ts_unix' in tweets_df.columns:
        try:
            _process_events(tweets_df, 'ts_unix', 'tweet_astro', has_mercury=True)
        except Exception:
            pass

    # ======================================================
    # NEWS EVENT ASTROLOGY
    # ======================================================
    if news_df is not None and len(news_df) > 0 and 'ts_unix' in news_df.columns:
        try:
            _process_events(news_df, 'ts_unix', 'news_astro', has_mercury=False)
        except Exception:
            pass

    # ======================================================
    # SPORTS GAME ASTROLOGY
    # ======================================================
    if sports_df is not None:
        games_df = sports_df.get('games', pd.DataFrame()) if isinstance(sports_df, dict) else sports_df
        if games_df is not None and len(games_df) > 0:
            try:
                gm = games_df.copy()
                # Sports may have ts_unix or game_time; fall back to date
                if 'ts_unix' in gm.columns:
                    _process_events(gm, 'ts_unix', 'game_astro', has_mercury=False)
                elif 'game_time' in gm.columns:
                    _process_events(gm, 'game_time', 'game_astro', has_mercury=False)
                elif 'date' in gm.columns:
                    # Convert date to approximate ts_unix (noon UTC)
                    gm['_ts_approx'] = pd.to_datetime(
                        gm['date'], errors='coerce'
                    ).astype(np.int64) // 10**9 + 43200  # +12h for noon
                    _process_events(gm, '_ts_approx', 'game_astro', has_mercury=False)
            except Exception:
                pass

    return out


# ============================================================
# COMPUTE HIGHER TIMEFRAME FEATURES
# ============================================================

def compute_higher_tf_features(df: pd.DataFrame, htf_data: dict) -> pd.DataFrame:
    """
    Add higher-TF context features.

    Args:
        df: base TF DataFrame with DatetimeIndex
        htf_data: dict like {'4h': df_4h, '1d': df_1d, '1w': df_1w}
            Each htf df must have: open, high, low, close, volume columns and DatetimeIndex

    Returns:
        DataFrame with higher-TF features (forward-filled to base TF)
    """
    _gpu = _is_gpu(df)
    # HTF features merge with external pandas DataFrames — extract CPU index, avoid full _to_cpu()
    if _gpu:
        _cpu_idx = pd.DatetimeIndex(_np(df.index))
    else:
        _cpu_idx = df.index
    out = pd.DataFrame(index=_cpu_idx)
    df_date_idx = _cpu_idx.normalize()
    if df_date_idx.tz is not None:
        df_date_idx = df_date_idx.tz_localize(None)

    prefix_map = {'15m': 'm15', '4h': 'h4', '1d': 'd', '1w': 'w', '1h': 'h1'}

    for tf_name, htf_df in (htf_data or {}).items():
        if htf_df is None or len(htf_df) == 0:
            continue

        prefix = prefix_map.get(tf_name, tf_name)
        htf_c = htf_df['close'].astype(float)
        htf_h = htf_df['high'].astype(float)
        htf_l = htf_df['low'].astype(float)
        htf_v = htf_df['volume'].astype(float) if 'volume' in htf_df.columns else pd.Series(0, index=htf_df.index)

        # Compute HTF indicators
        htf_ema50 = htf_c.ewm(span=50, adjust=False).mean()
        htf_rsi14 = compute_rsi(htf_c, 14)
        htf_mid = htf_c.rolling(20).mean()
        htf_std = htf_c.rolling(20).std()
        htf_bb_pctb = (htf_c - (htf_mid - 2 * htf_std)) / ((htf_mid + 2 * htf_std) - (htf_mid - 2 * htf_std))
        htf_ema12 = htf_c.ewm(span=12, adjust=False).mean()
        htf_ema26 = htf_c.ewm(span=26, adjust=False).mean()
        htf_macd = htf_ema12 - htf_ema26
        htf_tr = pd.concat([htf_h - htf_l, (htf_h - htf_c.shift(1)).abs(),
                            (htf_l - htf_c.shift(1)).abs()], axis=1).max(axis=1)
        htf_atr = htf_tr.rolling(14).mean()
        htf_vol_ratio = htf_v / htf_v.rolling(20).mean()

        features = {
            f'{prefix}_trend': (htf_c > htf_ema50).astype(int),
            f'{prefix}_ema50_dist': (htf_c - htf_ema50) / htf_ema50,
            f'{prefix}_rsi14': htf_rsi14,
            f'{prefix}_bb_pctb': htf_bb_pctb,
            f'{prefix}_macd': htf_macd,
            f'{prefix}_return': htf_c.pct_change(),
            f'{prefix}_volatility': htf_c.pct_change().rolling(20).std(),
            f'{prefix}_atr_pct': htf_atr / htf_c,
            f'{prefix}_vol_ratio': htf_vol_ratio,
        }

        htf_features = pd.DataFrame(features, index=htf_df.index)

        # Forward-fill to base TF
        if tf_name in ('1d', '1w'):
            # Daily/weekly: match by normalized date, strip tz for alignment
            htf_norm = htf_features.index.normalize()
            if htf_norm.tz is not None:
                htf_norm = htf_norm.tz_localize(None)
            htf_features.index = htf_norm
            for col in htf_features.columns:
                mapped = htf_features[col].reindex(df_date_idx)
                mapped.index = _cpu_idx
                out[col] = mapped.ffill()
        else:
            # Sub-daily: reindex with ffill
            # Strip timezone from both indices to avoid datetime64[ns,UTC] vs datetime64[ns] mismatch
            _htf_idx = htf_features.index
            _df_idx = _cpu_idx
            if hasattr(_htf_idx, 'tz') and _htf_idx.tz is not None:
                htf_features = htf_features.copy()
                htf_features.index = _htf_idx.tz_localize(None)
            if hasattr(_df_idx, 'tz') and _df_idx.tz is not None:
                _df_idx = _df_idx.tz_localize(None)
            for col in htf_features.columns:
                s = htf_features[col].reindex(_df_idx, method='ffill')
                s.index = _cpu_idx
                out[col] = s

    return out.to_pandas() if hasattr(out, 'to_pandas') else out


# ============================================================
# COMPUTE REGIME FEATURES
# ============================================================

def compute_regime_features(df: pd.DataFrame, tf_name: str = '1h') -> pd.DataFrame:
    """
    Compute EMA50 slope/regime features.
    HMM features are NaN for live (require full history).
    """
    _gpu = _is_gpu(df)
    _m = cudf if (_HAS_CUDF and _gpu) else pd
    cfg = TF_CONFIG.get(tf_name, TF_CONFIG['1h'])
    out = _m.DataFrame(index=df.index)
    c = df['close'].astype(float)

    ema50 = c.ewm(span=50).mean()
    slope_bars = cfg['ema50_slope_bars']
    ema50_slope = (ema50 - ema50.shift(slope_bars)) / ema50.shift(slope_bars) * 100

    out['ema50_declining'] = (ema50_slope < -3.0).astype(int)
    out['ema50_rising'] = (ema50_slope > 3.0).astype(int)
    out['ema50_slope'] = ema50_slope

    # Range position: where price sits in the 20-day high-low range (0=bottom, 1=top)
    bucket_seconds = cfg['bucket_seconds']
    bars_per_day = max(1, int(86400 / bucket_seconds))
    window_20d = max(2, 20 * bars_per_day)
    hi20 = df['high'].astype(float).rolling(window_20d, min_periods=min(10, window_20d)).max()
    lo20 = df['low'].astype(float).rolling(window_20d, min_periods=min(10, window_20d)).min()
    rng = hi20 - lo20
    out['range_position'] = ((c - lo20) / rng.replace(0, np.nan)).clip(0, 1)

    return out.to_pandas() if hasattr(out, 'to_pandas') else out


# ============================================================
# CONFIRMED CYCLE FEATURES
# ============================================================

# Cycles confirmed vs BTC volatility (all survive Bonferroni correction)
CONFIRMED_CYCLES = {
    'schumann_133d': 133,       # r=-0.19, p~1e-20
    'schumann_143d': 143,       # r=+0.13, p~9e-11
    'schumann_783d': 783,       # r=+0.14, p~3e-12
    'chakra_heart_161d': 161,   # r=+0.09, p~5e-6 (golden ratio)
    'jupiter_365d': 365,        # r=-0.28, p<0.001
    'mercury_1216d': 1216,      # r=0.057 DIRECTIONAL, p=0.006
}

# Known solar/lunar eclipses 2017-2027
_ECLIPSE_DATES = [
    (2017, 2, 11), (2017, 2, 26), (2017, 8, 7), (2017, 8, 21),
    (2018, 1, 31), (2018, 2, 15), (2018, 7, 13), (2018, 7, 27), (2018, 8, 11),
    (2019, 1, 6), (2019, 1, 21), (2019, 7, 2), (2019, 7, 16), (2019, 12, 26),
    (2020, 1, 10), (2020, 6, 5), (2020, 6, 21), (2020, 7, 5), (2020, 11, 30),
    (2020, 12, 14),
    (2021, 5, 26), (2021, 6, 10), (2021, 11, 19), (2021, 12, 4),
    (2022, 4, 30), (2022, 5, 16), (2022, 10, 25), (2022, 11, 8),
    (2023, 4, 20), (2023, 5, 5), (2023, 10, 14), (2023, 10, 28),
    (2024, 3, 25), (2024, 4, 8), (2024, 9, 18), (2024, 10, 2),
    (2025, 3, 14), (2025, 3, 29), (2025, 9, 7), (2025, 9, 21),
    (2026, 2, 17), (2026, 3, 3), (2026, 8, 12), (2026, 8, 28),
    (2027, 2, 6), (2027, 2, 20), (2027, 7, 18), (2027, 8, 2), (2027, 8, 17),
]

# Equinox/solstice dates (month, day)
_EQUINOX_SOLSTICE = [(3, 20), (6, 21), (9, 22), (12, 21)]


# ============================================================
# FRACTIONAL DIFFERENTIATION (FFD — Fixed-Width Window)
# ============================================================

def _ffd_weights(d, threshold=1e-3, max_k=100):
    """Compute FFD weights for fractional differentiation.
    w_k = -w_{k-1} * (d - k + 1) / k, starting with w_0 = 1.
    Truncate when |w_k| < threshold.
    Uses threshold=1e-3 and max_k=100 to balance stationarity vs data loss."""
    weights = [1.0]
    for k in range(1, max_k):
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)
    return np.array(weights, dtype=np.float64)


def _frac_diff_ffd(series, d, threshold=1e-5):
    """Apply Fixed-Width Window Fractional Differentiation to a series.
    Uses CuPy convolution on GPU when available, numpy fallback on CPU."""
    weights = _ffd_weights(d, threshold)
    W = len(weights)
    y = np.asarray(series, dtype=np.float64)

    # Replace NaN with 0 for convolution, track validity
    y_clean = np.where(np.isnan(y), 0, y)
    valid = (~np.isnan(y)).astype(np.float64)

    # Use CuPy convolve on GPU when available
    if _HAS_CP_CONV and _HAS_GPU:
        try:
            y_gpu = cp.asarray(y_clean)
            w_gpu = cp.asarray(weights)
            v_gpu = cp.asarray(valid)
            ones_gpu = cp.ones(W)
            result = cp.asnumpy(cp_convolve(y_gpu, w_gpu, mode='full'))[:len(y)]
            valid_count = cp.asnumpy(cp_convolve(v_gpu, ones_gpu, mode='full'))[:len(y)]
            del y_gpu, w_gpu, v_gpu, ones_gpu
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            # Fallback to numpy
            result = np.convolve(y_clean, weights, mode='full')[:len(y)]
            valid_count = np.convolve(valid, np.ones(W), mode='full')[:len(y)]
    else:
        # Convolve: each output[i] = sum(weights[k] * y[i-k] for k in range(W))
        result = np.convolve(y_clean, weights, mode='full')[:len(y)]
        valid_count = np.convolve(valid, np.ones(W), mode='full')[:len(y)]

    # Mark positions where we don't have enough history as NaN
    result = np.where(valid_count >= W * 0.5, result, np.nan)
    # First W-1 values are unreliable
    result[:W - 1] = np.nan
    return result


def compute_frac_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute fractionally differentiated features.

    Uses FFD (Fixed-Width Window) method from Lopez de Prado.
    d values are fixed per feature cluster (not optimized during model selection):
      - Price cluster (close): d=0.4 (preserves memory while achieving stationarity)
      - Volume cluster (volume, OBV): d=0.3 (volume is less persistent)

    d parameter strategy: estimated ONCE on fixed historical period, kept FIXED.
    Do NOT re-optimize d during model selection — that's another hyperparameter search.
    """
    _gpu = _is_gpu(df)
    # FFD uses convolution — CuPy on GPU when available, numpy fallback
    if _gpu:
        _cpu_idx = pd.DatetimeIndex(_np(df.index))
    else:
        _cpu_idx = df.index
    out = pd.DataFrame(index=_cpu_idx)
    c_vals = _np(df['close']).astype(np.float64)
    v_vals = _np(df['volume']).astype(np.float64)

    # Price cluster — d=0.4
    d_price = 0.4
    out[f'frac_diff_close_{d_price}'] = _frac_diff_ffd(c_vals, d=d_price)
    # Also compute log-close frac diff (more standard)
    log_close = np.log(np.where(c_vals > 0, c_vals, np.nan))
    out[f'frac_diff_log_close_{d_price}'] = _frac_diff_ffd(log_close, d=d_price)

    # Volume cluster — d=0.3
    d_vol = 0.3
    out[f'frac_diff_volume_{d_vol}'] = _frac_diff_ffd(v_vals, d=d_vol)

    # OBV (On-Balance Volume) frac diff
    _has_obv = 'obv' in (list(df.columns) if _gpu else df.columns)
    if _has_obv:
        obv = _np(df['obv']).astype(np.float64)
    else:
        # Compute OBV inline
        price_change = np.sign(np.diff(c_vals, prepend=c_vals[0]))
        obv = np.cumsum(v_vals * price_change)
    out[f'frac_diff_obv_{d_vol}'] = _frac_diff_ffd(obv, d=d_vol)

    # Additional d values for experimentation (fixed, not optimized)
    for d_extra in [0.2, 0.6]:
        out[f'frac_diff_close_{d_extra}'] = _frac_diff_ffd(c_vals, d=d_extra)

    return out


def compute_cycle_features(df: pd.DataFrame, tf_name: str = '1h') -> pd.DataFrame:
    """
    Compute confirmed cycle sin/cos features and astronomical calendar signals.
    Cycles are from book correlation analysis (all survive Bonferroni correction).
    """
    _gpu = _is_gpu(df)
    # Cycle features use Python datetime loops — extract CPU index, avoid full _to_cpu()
    if _gpu:
        _cpu_idx = pd.DatetimeIndex(_np(df.index))
    else:
        _cpu_idx = df.index
    cfg = TF_CONFIG.get(tf_name, TF_CONFIG['1h'])
    bucket_seconds = cfg['bucket_seconds']
    bars_per_day = max(1.0, 86400.0 / bucket_seconds)
    out = pd.DataFrame(index=_cpu_idx)
    n = len(_cpu_idx)
    bar_idx = np.arange(n, dtype=np.float64)

    # Sin/cos encoding for each confirmed cycle
    for cycle_name, period_days in CONFIRMED_CYCLES.items():
        period_bars = period_days * bars_per_day
        if period_bars < 2:
            continue
        phase = 2.0 * np.pi * bar_idx / period_bars
        out[f'{cycle_name}_sin'] = np.sin(phase)
        out[f'{cycle_name}_cos'] = np.cos(phase)

    dates = _cpu_idx

    # Strip timezone for naive datetime comparisons
    def _to_naive(dt):
        if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
            return dt.replace(tzinfo=None)
        return dt

    # --- equinox_proximity: days to nearest equinox/solstice ---
    def _days_to_nearest_equinox(dt):
        dt = _to_naive(dt)
        year = dt.year
        best = 366
        for y in [year - 1, year, year + 1]:
            for m, d in _EQUINOX_SOLSTICE:
                try:
                    eq = datetime(y, m, d)
                    diff = abs((dt - eq).days)
                    if diff < best:
                        best = diff
                except Exception:
                    pass
        return best

    eq_prox = np.array([_days_to_nearest_equinox(d.to_pydatetime()) for d in dates],
                       dtype=np.float64)
    out['equinox_proximity'] = eq_prox

    # --- eclipse_window: 1 if within 7 days of known eclipse ---
    eclipse_dt_list = []
    for yr, mo, dy in _ECLIPSE_DATES:
        try:
            eclipse_dt_list.append(datetime(yr, mo, dy))
        except Exception:
            pass

    def _in_eclipse_window(dt, window=7):
        dt = _to_naive(dt)
        for edt in eclipse_dt_list:
            if abs((dt - edt).days) <= window:
                return 1
        return 0

    out['eclipse_window'] = np.array(
        [_in_eclipse_window(d.to_pydatetime()) for d in dates], dtype=np.int32)

    # --- eclipse_proximity_days: continuous distance to nearest eclipse (0 = eclipse day) ---
    # Vectorized: convert bar dates and eclipse dates to ordinals, then broadcast min-abs-diff
    _bar_ordinals = np.array([d.to_pydatetime().toordinal() for d in dates], dtype=np.int64)
    _eclipse_ordinals = np.array([edt.toordinal() for edt in eclipse_dt_list], dtype=np.int64)
    if len(_eclipse_ordinals) > 0:
        # shape: (n_bars, n_eclipses) — efficient for <60 eclipses
        _abs_diffs = np.abs(_bar_ordinals[:, None] - _eclipse_ordinals[None, :])
        out['eclipse_proximity_days'] = _abs_diffs.min(axis=1).astype(np.float64)
    else:
        out['eclipse_proximity_days'] = np.full(len(dates), np.nan)

    # --- equinox_pre_post: +1 within 30d before, -1 within 30d after ---
    def _equinox_pre_post(dt):
        dt = _to_naive(dt)
        year = dt.year
        for y in [year - 1, year, year + 1]:
            for m, d in _EQUINOX_SOLSTICE:
                try:
                    eq = datetime(y, m, d)
                    diff = (dt - eq).days  # negative = before equinox
                    if -30 <= diff < 0:
                        return 1
                    if 0 <= diff <= 30:
                        return -1
                except Exception:
                    pass
        return 0

    out['equinox_pre_post'] = np.array(
        [_equinox_pre_post(d.to_pydatetime()) for d in dates], dtype=np.int32)

    return out


# ============================================================
# COMPOSITE VOL-TO-DIRECTION FEATURES
# ============================================================

def compute_composite_features(df: pd.DataFrame, tf_name: str = '1h') -> pd.DataFrame:
    """
    Bridge volatility prediction to directional trading.
    Only builds composites from columns that actually exist in df.
    """
    _gpu = _is_gpu(df)
    _m = cudf if (_HAS_CUDF and _gpu) else pd
    cfg = TF_CONFIG.get(tf_name, TF_CONFIG['1h'])
    out = _m.DataFrame(index=df.index)

    # GPU-safe numeric coerce: cuDF columns are already numeric, pd needs to_numeric
    def _numeric(series):
        if _gpu:
            try:
                return series.astype('float64')
            except Exception:
                return _m.Series(np.nan, index=df.index)
        return pd.to_numeric(series, errors='coerce')

    # --- 1. esoteric_vol_score: weighted sum of confirmed vol predictors ---
    vol_weights = {
        'jupiter_365d_sin': -0.28,
        'jupiter_365d_cos': -0.28,
        'schumann_133d_sin': -0.19,
        'schumann_133d_cos': -0.19,
        'schumann_783d_sin': 0.14,
        'schumann_783d_cos': 0.14,
        'schumann_143d_sin': 0.13,
        'schumann_143d_cos': 0.13,
        'chakra_heart_161d_sin': 0.09,
        'chakra_heart_161d_cos': 0.09,
        'mercury_1216d_sin': 0.057,
        'mercury_1216d_cos': 0.057,
    }
    weight_sum = 0.0
    score = _m.Series(0.0, index=df.index)
    for col, w in vol_weights.items():
        if col in df.columns:
            score = score + w * _numeric(df[col])
            weight_sum += abs(w)
    if weight_sum > 0:
        score = score / weight_sum
    out['esoteric_vol_score'] = score

    # --- 2. vol_regime_transition ---
    c = df['close'].astype(float) if 'close' in df.columns else None
    if c is not None:
        log_ret = np.log(c / c.shift(1))
        vol_short = cfg.get('vol_short', 10)
        realized_vol = log_ret.rolling(vol_short).std()
        rv_mean = realized_vol.rolling(vol_short * 5).mean()
        rv_std = realized_vol.rolling(vol_short * 5).std()
        realized_vol_zscore = (realized_vol - rv_mean) / rv_std.replace(0, np.nan)
        out['vol_regime_transition'] = out['esoteric_vol_score'] - realized_vol_zscore
    else:
        out['vol_regime_transition'] = np.nan

    # --- 3. vol_breakout_direction ---
    if c is not None:
        ema50 = c.ewm(span=50).mean()
        direction_sign = np.sign(c - ema50)
        vol_above = np.maximum(out['esoteric_vol_score'] - 0.5, 0)
        out['vol_breakout_direction'] = direction_sign * vol_above
    else:
        out['vol_breakout_direction'] = np.nan

    # --- 4. vol_directional_asymmetry ---
    if c is not None:
        returns = c.pct_change()
        vol_short = cfg.get('vol_short', 10)
        window = max(vol_short * 3, 20)
        neg_ret = returns.where(returns < 0, 0.0)
        pos_ret = returns.where(returns > 0, 0.0)
        down_vol = neg_ret.rolling(window).std()
        up_vol = pos_ret.rolling(window).std()
        out['vol_directional_asymmetry'] = down_vol / up_vol.replace(0, np.nan)
    else:
        out['vol_directional_asymmetry'] = np.nan

    # --- 5. cycle_confluence_score ---
    cycle_cols = [col for col in df.columns
                  if any(col.startswith(cn) for cn in CONFIRMED_CYCLES.keys())
                  and (col.endswith('_sin') or col.endswith('_cos'))]
    if cycle_cols:
        # Normalize each to [0, 1] then take product
        normalized = _m.DataFrame(index=df.index)
        for col in cycle_cols:
            vals = _numeric(df[col])
            normalized[col] = (vals + 1.0) / 2.0  # sin/cos range [-1,1] -> [0,1]
        out['cycle_confluence_score'] = normalized.prod(axis=1)
    else:
        out['cycle_confluence_score'] = np.nan

    # --- 6. seasonal_vol_direction ---
    _df_cpu_idx = pd.DatetimeIndex(_np(df.index)) if _gpu else df.index
    quarter = _df_cpu_idx.quarter.astype(np.float64)
    out['seasonal_vol_direction'] = (quarter / 4.0) * _np(out['esoteric_vol_score'])

    return out.to_pandas() if hasattr(out, 'to_pandas') else out


# ============================================================
# HEBREW / CULTURAL CALENDAR FEATURES
# ============================================================

def compute_hebrew_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Approximate cultural/religious calendar features from dates.
    No external data required -- uses fixed approximations.
    """
    _gpu = _is_gpu(df)
    # Calendar features use Python datetime loops — extract CPU index, avoid full _to_cpu()
    if _gpu:
        _cpu_idx = pd.DatetimeIndex(_np(df.index))
    else:
        _cpu_idx = df.index
    out = pd.DataFrame(index=_cpu_idx)

    dates = _cpu_idx

    # Yom Kippur (10 Tishrei)
    _YOM_KIPPUR = {
        2017: (9, 30), 2018: (9, 19), 2019: (10, 9), 2020: (9, 28),
        2021: (9, 16), 2022: (10, 5), 2023: (9, 25), 2024: (10, 12),
        2025: (10, 2), 2026: (9, 22), 2027: (10, 11),
        2028: (9, 30), 2029: (9, 19), 2030: (10, 7), 2031: (9, 27),
        2032: (9, 15), 2033: (10, 3), 2034: (9, 23), 2035: (9, 12),
    }
    # Passover (15 Nisan)
    _PASSOVER = {
        2017: (4, 11), 2018: (3, 31), 2019: (4, 20), 2020: (4, 9),
        2021: (3, 28), 2022: (4, 16), 2023: (4, 6), 2024: (4, 23),
        2025: (4, 13), 2026: (4, 2), 2027: (4, 22),
        2028: (4, 11), 2029: (3, 31), 2030: (4, 18), 2031: (4, 8),
        2032: (3, 27), 2033: (4, 14), 2034: (4, 4), 2035: (4, 23),
    }
    # Rosh Hashanah (1 Tishrei)
    _ROSH_HASHANAH = {
        2017: (9, 21), 2018: (9, 10), 2019: (9, 30), 2020: (9, 19),
        2021: (9, 7), 2022: (9, 26), 2023: (9, 16), 2024: (10, 3),
        2025: (9, 23), 2026: (9, 12), 2027: (10, 2),
        2028: (9, 21), 2029: (9, 10), 2030: (9, 28), 2031: (9, 18),
        2032: (9, 6), 2033: (9, 24), 2034: (9, 14), 2035: (9, 3),
    }
    # Sukkot (15 Tishrei)
    _SUKKOT = {
        2017: (10, 5), 2018: (9, 24), 2019: (10, 14), 2020: (10, 3),
        2021: (9, 21), 2022: (10, 10), 2023: (9, 30), 2024: (10, 17),
        2025: (10, 7), 2026: (9, 27), 2027: (10, 16),
        2028: (10, 5), 2029: (9, 24), 2030: (10, 12), 2031: (10, 2),
        2032: (9, 20), 2033: (10, 8), 2034: (9, 28), 2035: (9, 17),
    }
    # Chinese New Year
    _CHINESE_NY = {
        2017: (1, 28), 2018: (2, 16), 2019: (2, 5), 2020: (1, 25),
        2021: (2, 12), 2022: (2, 1), 2023: (1, 22), 2024: (2, 10),
        2025: (1, 29), 2026: (2, 17), 2027: (2, 6),
        2028: (1, 26), 2029: (2, 13), 2030: (2, 3), 2031: (1, 23),
        2032: (2, 11), 2033: (1, 31), 2034: (2, 19), 2035: (2, 8),
    }
    # Diwali
    _DIWALI = {
        2017: (10, 19), 2018: (11, 7), 2019: (10, 27), 2020: (11, 14),
        2021: (11, 4), 2022: (10, 24), 2023: (11, 12), 2024: (11, 1),
        2025: (10, 20), 2026: (11, 8), 2027: (10, 29),
        2028: (10, 26), 2029: (11, 14), 2030: (11, 4), 2031: (10, 24),
        2032: (11, 11), 2033: (11, 1), 2034: (10, 21), 2035: (11, 8),
    }
    # Ramadan start (approximate)
    _RAMADAN_START = {
        2017: (5, 27), 2018: (5, 16), 2019: (5, 6), 2020: (4, 24),
        2021: (4, 13), 2022: (4, 2), 2023: (3, 23), 2024: (3, 11),
        2025: (3, 1), 2026: (2, 18), 2027: (2, 8),
        2028: (12, 4), 2029: (11, 24), 2030: (11, 13), 2031: (11, 2),
        2032: (10, 22), 2033: (10, 12), 2034: (10, 1), 2035: (9, 20),
    }
    # Eid al-Fitr (end of Ramadan, 1st of Shawwal)
    _EID_AL_FITR = {
        2026: (3, 30), 2027: (3, 20), 2028: (1, 3), 2029: (12, 23),
        2030: (12, 13), 2031: (12, 2), 2032: (11, 20), 2033: (11, 10),
        2034: (10, 30), 2035: (10, 20),
    }
    # Eid al-Adha (10th of Dhul Hijjah)
    _EID_AL_ADHA = {
        2026: (6, 7), 2027: (5, 27), 2028: (5, 15), 2029: (5, 5),
        2030: (4, 24), 2031: (4, 13), 2032: (4, 2), 2033: (3, 22),
        2034: (3, 12), 2035: (3, 1),
    }
    # Navratri (Sharad Navratri, 9 nights before Dussehra)
    _NAVRATRI = {
        2026: (9, 22), 2027: (10, 12), 2028: (9, 30), 2029: (9, 19),
        2030: (10, 8), 2031: (9, 28), 2032: (9, 16), 2033: (10, 5),
        2034: (9, 24), 2035: (9, 14),
    }
    # Holi (full moon of Phalguna)
    _HOLI = {
        2026: (3, 10), 2027: (2, 28), 2028: (3, 17), 2029: (3, 7),
        2030: (2, 24), 2031: (3, 15), 2032: (3, 4), 2033: (3, 22),
        2034: (3, 11), 2035: (3, 1),
    }
    # Chuseok Korea (15th of 8th lunar month)
    _CHUSEOK = {
        2026: (9, 25), 2027: (10, 15), 2028: (10, 3), 2029: (9, 22),
        2030: (10, 11), 2031: (10, 1), 2032: (9, 19), 2033: (10, 7),
        2034: (9, 27), 2035: (9, 16),
    }

    def _holiday_window(dt, holiday_dict, window=3):
        """Return 1 if within window days of holiday."""
        yr = dt.year
        if yr in holiday_dict:
            m, d = holiday_dict[yr]
            try:
                hdt = datetime(yr, m, d)
                if abs((dt - hdt).days) <= window:
                    return 1
            except Exception:
                pass
        return 0

    def _ramadan_window(dt, window=30):
        """Return 1 if within Ramadan (~30 day window from start)."""
        yr = dt.year
        if yr in _RAMADAN_START:
            m, d = _RAMADAN_START[yr]
            try:
                start = datetime(yr, m, d)
                diff = (dt - start).days
                if 0 <= diff <= window:
                    return 1
            except Exception:
                pass
        return 0

    def _is_date_palindrome(dt):
        """Check if date is palindrome in YYYYMMDD or MMDDYYYY format."""
        fmt1 = dt.strftime('%Y%m%d')
        fmt2 = dt.strftime('%m%d%Y')
        if fmt1 == fmt1[::-1]:
            return 1
        if fmt2 == fmt2[::-1]:
            return 1
        return 0

    pydates = [d.to_pydatetime() for d in dates]

    out['yom_kippur_window'] = np.array(
        [_holiday_window(d, _YOM_KIPPUR) for d in pydates], dtype=np.int32)
    out['passover_window'] = np.array(
        [_holiday_window(d, _PASSOVER) for d in pydates], dtype=np.int32)
    out['rosh_hashanah_window'] = np.array(
        [_holiday_window(d, _ROSH_HASHANAH) for d in pydates], dtype=np.int32)
    out['sukkot_window'] = np.array(
        [_holiday_window(d, _SUKKOT) for d in pydates], dtype=np.int32)
    out['chinese_new_year_window'] = np.array(
        [_holiday_window(d, _CHINESE_NY) for d in pydates], dtype=np.int32)
    out['diwali_window'] = np.array(
        [_holiday_window(d, _DIWALI) for d in pydates], dtype=np.int32)
    out['ramadan_window'] = np.array(
        [_ramadan_window(d) for d in pydates], dtype=np.int32)
    out['eid_al_fitr_window'] = np.array(
        [_holiday_window(d, _EID_AL_FITR) for d in pydates], dtype=np.int32)
    out['eid_al_adha_window'] = np.array(
        [_holiday_window(d, _EID_AL_ADHA) for d in pydates], dtype=np.int32)
    out['navratri_window'] = np.array(
        [_holiday_window(d, _NAVRATRI) for d in pydates], dtype=np.int32)
    out['holi_window'] = np.array(
        [_holiday_window(d, _HOLI) for d in pydates], dtype=np.int32)
    out['chuseok_window'] = np.array(
        [_holiday_window(d, _CHUSEOK) for d in pydates], dtype=np.int32)
    # Golden Week Japan (Apr 29 - May 5, fixed Gregorian — pure calendar logic)
    out['golden_week_window'] = np.array(
        [1 if (d.month == 4 and d.day >= 29) or (d.month == 5 and d.day <= 5) else 0
         for d in pydates], dtype=np.int32)
    out['date_palindrome'] = np.array(
        [_is_date_palindrome(d) for d in pydates], dtype=np.int32)

    return out


# ============================================================
# MARKET SIGNAL FEATURES (DeFi TVL, BTC dominance, mining stats)
# ============================================================

def compute_market_signal_features(df: pd.DataFrame,
                                    market_signals: dict,
                                    tf_name: str = '1d') -> pd.DataFrame:
    """Compute features from DeFi TVL, BTC dominance, and mining stats."""
    cfg = TF_CONFIG.get(tf_name, TF_CONFIG['1d'])
    bars_per_day = max(1, 86400 // cfg['bucket_seconds'])
    out = pd.DataFrame(index=df.index)

    if market_signals is None:
        return out

    def _align_daily(daily_df, col_map):
        if daily_df is None or daily_df.empty:
            return
        aligned = daily_df.copy()
        if hasattr(aligned.index, 'tz') and aligned.index.tz is not None:
            aligned.index = aligned.index.tz_localize(None)
        _idx = _strip_tz(df.index)
        aligned = aligned.reindex(_idx, method='ffill')
        aligned.index = df.index
        for src, dst in col_map.items():
            if src in aligned.columns:
                out[dst] = pd.to_numeric(aligned[src], errors='coerce')

    _align_daily(market_signals.get('defi_tvl', pd.DataFrame()), {
        'total_tvl': 'mkt_defi_tvl',
        'total_tvl_change_1d': 'mkt_defi_tvl_change_1d',
    })
    _align_daily(market_signals.get('btc_dominance', pd.DataFrame()), {
        'btc_dominance': 'mkt_btc_dominance',
        'eth_dominance': 'mkt_eth_dominance',
        'total_market_cap': 'mkt_total_market_cap',
        'total_volume': 'mkt_total_volume',
        'active_cryptos': 'mkt_active_cryptos',
    })
    _align_daily(market_signals.get('mining_stats', pd.DataFrame()), {
        'hash_rate': 'mkt_hash_rate',
        'difficulty': 'mkt_difficulty',
        'blocks_mined': 'mkt_blocks_mined',
        'miners_revenue': 'mkt_miners_revenue',
        'avg_block_size': 'mkt_avg_block_size',
    })

    # Derived features
    tvl = out.get('mkt_defi_tvl')
    if tvl is not None:
        shift_7d = 7 * bars_per_day
        if shift_7d > 0 and len(tvl) > shift_7d:
            out['mkt_defi_tvl_pct_7d'] = tvl.pct_change(shift_7d)

    btc_dom = out.get('mkt_btc_dominance')
    if btc_dom is not None:
        shift_7d = 7 * bars_per_day
        if shift_7d > 0 and len(btc_dom) > shift_7d:
            out['mkt_btc_dom_delta_7d'] = btc_dom - btc_dom.shift(shift_7d)
        out['mkt_alt_season'] = (btc_dom < 40).astype(float)
        out.loc[btc_dom.isna(), 'mkt_alt_season'] = np.nan

    hr = out.get('mkt_hash_rate')
    if hr is not None:
        shift_30d = 30 * bars_per_day
        if shift_30d > 0 and len(hr) > shift_30d:
            out['mkt_hash_rate_pct_30d'] = hr.pct_change(shift_30d)

    return out


# ============================================================
# MARKET CALENDAR FEATURES (real-world events that move markets)
# ============================================================

# FOMC meeting dates (announcement day, 2019-2028)
_FOMC_DATES = [
    # 2019
    (2019,1,30),(2019,3,20),(2019,5,1),(2019,6,19),(2019,7,31),(2019,9,18),(2019,10,30),(2019,12,11),
    # 2020
    (2020,1,29),(2020,3,3),(2020,3,15),(2020,4,29),(2020,6,10),(2020,7,29),(2020,9,16),(2020,11,5),(2020,12,16),
    # 2021
    (2021,1,27),(2021,3,17),(2021,4,28),(2021,6,16),(2021,7,28),(2021,9,22),(2021,11,3),(2021,12,15),
    # 2022
    (2022,1,26),(2022,3,16),(2022,5,4),(2022,6,15),(2022,7,27),(2022,9,21),(2022,11,2),(2022,12,14),
    # 2023
    (2023,2,1),(2023,3,22),(2023,5,3),(2023,6,14),(2023,7,26),(2023,9,20),(2023,11,1),(2023,12,13),
    # 2024
    (2024,1,31),(2024,3,20),(2024,5,1),(2024,6,12),(2024,7,31),(2024,9,18),(2024,11,7),(2024,12,18),
    # 2025
    (2025,1,29),(2025,3,19),(2025,5,7),(2025,6,18),(2025,7,30),(2025,9,17),(2025,10,29),(2025,12,17),
    # 2026
    (2026,1,28),(2026,3,18),(2026,4,29),(2026,6,17),(2026,7,29),(2026,9,16),(2026,11,4),(2026,12,16),
    # 2027
    (2027,1,27),(2027,3,17),(2027,4,28),(2027,6,16),(2027,7,28),(2027,9,22),(2027,11,3),(2027,12,15),
    # 2028 (projected — Fed publishes ~1 year ahead)
    (2028,1,26),(2028,3,15),(2028,4,26),(2028,6,14),(2028,7,26),(2028,9,20),(2028,11,1),(2028,12,13),
]

# Bitcoin halving dates
_HALVING_DATES = [
    (2012, 11, 28),  # Block 210,000
    (2016, 7, 9),    # Block 420,000
    (2020, 5, 11),   # Block 630,000
    (2024, 4, 19),   # Block 840,000
    (2028, 4, 15),   # Block 1,050,000 (estimated)
]

# Super Bowl dates (Sunday)
_SUPER_BOWL_DATES = [
    (2019,2,3),(2020,2,2),(2021,2,7),(2022,2,13),(2023,2,12),
    (2024,2,11),(2025,2,9),(2026,2,8),(2027,2,7),
]

# US Presidential election day (first Tuesday after first Monday in November)
_US_ELECTION_DATES = [
    (2020, 11, 3), (2024, 11, 5), (2028, 11, 7),
]
_US_MIDTERM_DATES = [
    (2018, 11, 6), (2022, 11, 8), (2026, 11, 3),
]


def compute_market_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Real-world market events that drive buying/selling behavior.
    All date-based — no external data feeds needed.

    Categories:
      A) Economic calendar: FOMC, CPI, NFP
      B) Crypto market structure: options expiry, halving
      C) Tax & payroll cycles
      D) Seasonal behavior
      E) Political cycles
      F) Bitcoin-specific
    """
    _gpu = _is_gpu(df)
    # Market calendar uses Python datetime loops — extract CPU index, avoid full _to_cpu()
    if _gpu:
        _cpu_idx = pd.DatetimeIndex(_np(df.index))
    else:
        _cpu_idx = df.index
    out = pd.DataFrame(index=_cpu_idx)

    dates = _cpu_idx
    pydates = [d.to_pydatetime() for d in dates]

    def _to_naive(dt):
        if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
            return dt.replace(tzinfo=None)
        return dt

    def _in_window(dt, event_dates, window=2):
        """Return 1 if within ±window days of any event date."""
        dt = _to_naive(dt)
        for yr, mo, dy in event_dates:
            try:
                evt = datetime(yr, mo, dy)
                if abs((dt - evt).days) <= window:
                    return 1
            except Exception:
                pass
        return 0

    def _is_event_day(dt, event_dates):
        """Return 1 if this IS the event day."""
        dt = _to_naive(dt)
        for yr, mo, dy in event_dates:
            try:
                evt = datetime(yr, mo, dy)
                if (dt - evt).days == 0:
                    return 1
            except Exception:
                pass
        return 0

    # ================================================================
    # A) ECONOMIC CALENDAR
    # ================================================================

    # FOMC meeting day + window (±2 days)
    out['fomc_day'] = np.array([_is_event_day(d, _FOMC_DATES) for d in pydates], dtype=np.int32)
    out['fomc_window'] = np.array([_in_window(d, _FOMC_DATES, 2) for d in pydates], dtype=np.int32)

    # CPI release: typically 2nd or 3rd Wednesday of month, ~8:30am ET
    # Approximate: 12th-15th of each month
    day_of_month = dates.day
    out['cpi_window'] = ((day_of_month >= 10) & (day_of_month <= 15)).astype(int)

    # NFP: first Friday of month
    dow = dates.dayofweek  # 0=Monday, 4=Friday
    out['nfp_window'] = ((dow == 4) & (day_of_month <= 7)).astype(int)

    # ================================================================
    # B) CRYPTO MARKET STRUCTURE
    # ================================================================

    # Monthly options expiry: last Friday of month
    # Check if next Friday is in the next month
    next_day = dates + pd.Timedelta(days=7)
    is_friday = (dow == 4).astype(int)
    last_friday = is_friday & (next_day.month != dates.month)
    out['monthly_opex'] = last_friday.astype(int)
    # Window ±2 days around last Friday
    opex_window = np.zeros(len(_cpu_idx), dtype=np.int32)
    lf_indices = np.where(last_friday)[0]
    for idx in lf_indices:
        for offset in range(-2, 3):
            j = idx + offset
            if 0 <= j < len(_cpu_idx):
                opex_window[j] = 1
    out['monthly_opex_window'] = opex_window

    # Quarterly options expiry (Mar, Jun, Sep, Dec)
    month = dates.month
    out['quarterly_opex'] = (last_friday & month.isin([3, 6, 9, 12])).astype(int)
    out['quarterly_opex_window'] = (out['monthly_opex_window'] &
                                     month.isin([3, 6, 9, 12]).astype(int)).astype(int)

    # Bitcoin halving proximity (days to nearest halving, ±30 day window)
    def _halving_proximity(dt):
        dt = _to_naive(dt)
        best = 9999
        for yr, mo, dy in _HALVING_DATES:
            try:
                hdt = datetime(yr, mo, dy)
                diff = abs((dt - hdt).days)
                if diff < best:
                    best = diff
            except Exception:
                pass
        return best

    halving_prox = np.array([_halving_proximity(d) for d in pydates], dtype=np.float64)
    out['halving_proximity'] = halving_prox
    out['halving_window'] = (halving_prox <= 30).astype(int)

    # ================================================================
    # C) TAX & PAYROLL CYCLES
    # ================================================================

    # US tax deadline window: March 15 - April 20 (selling pressure zone)
    out['us_tax_window'] = ((month == 3) & (day_of_month >= 15) |
                             (month == 4) & (day_of_month <= 20)).astype(int)

    # Quarterly estimated tax deadlines: Apr 15, Jun 15, Sep 15, Jan 15
    _QTAX = [(1, 15), (4, 15), (6, 15), (9, 15)]
    out['quarterly_tax_window'] = np.array(
        [1 if any(abs((d - datetime(d.year, m, dy)).days) <= 3
                  for m, dy in _QTAX) else 0
         for d in [_to_naive(d) for d in pydates]], dtype=np.int32)

    # Payday proximity: ±2 days of 1st and 15th (buying pressure)
    out['payday_window'] = ((day_of_month <= 3) | (day_of_month >= 29) |
                             ((day_of_month >= 13) & (day_of_month <= 17))).astype(int)

    # Year-end tax loss harvesting: Dec 15-31
    out['tax_loss_harvest'] = ((month == 12) & (day_of_month >= 15)).astype(int)

    # January effect: fresh allocations Jan 1-15
    out['january_effect'] = ((month == 1) & (day_of_month <= 15)).astype(int)

    # ================================================================
    # D) SEASONAL BEHAVIOR
    # ================================================================

    # Black Friday window: Nov 22-30 (Thanksgiving week + Cyber Monday)
    out['black_friday_window'] = ((month == 11) & (day_of_month >= 22) &
                                   (day_of_month <= 30)).astype(int)

    # Super Bowl window: ±2 days
    out['super_bowl_window'] = np.array(
        [_in_window(d, _SUPER_BOWL_DATES, 2) for d in pydates], dtype=np.int32)

    # Summer doldrums: Jun 15 - Aug 31 (low volume season)
    out['summer_doldrums'] = (((month == 6) & (day_of_month >= 15)) |
                               (month == 7) | (month == 8)).astype(int)

    # Year-end bonus / new allocation season: Dec 20 - Jan 15
    out['bonus_season'] = (((month == 12) & (day_of_month >= 20)) |
                            ((month == 1) & (day_of_month <= 15))).astype(int)

    # ================================================================
    # E) POLITICAL CYCLES
    # ================================================================

    # Election year (presidential)
    year = dates.year
    out['election_year'] = (year % 4 == 0).astype(int)

    # Election window: ±30 days of election day
    out['election_window'] = np.array(
        [_in_window(d, _US_ELECTION_DATES, 30) for d in pydates], dtype=np.int32)

    # Midterm window
    out['midterm_window'] = np.array(
        [_in_window(d, _US_MIDTERM_DATES, 30) for d in pydates], dtype=np.int32)

    # ================================================================
    # F) BITCOIN-SPECIFIC
    # ================================================================

    # BTC birthday: Jan 3 (genesis block)
    out['btc_birthday'] = ((month == 1) & (day_of_month == 3)).astype(int)

    # BTC Pizza Day: May 22 (first real-world BTC transaction, culturally significant)
    out['btc_pizza_day'] = ((month == 5) & (day_of_month == 22)).astype(int)

    # Block reward era (which halving epoch are we in)
    def _block_reward_era(dt):
        dt = _to_naive(dt)
        era = 1
        for yr, mo, dy in _HALVING_DATES:
            try:
                if dt >= datetime(yr, mo, dy):
                    era += 1
            except Exception:
                pass
        return era
    out['block_reward_era'] = np.array(
        [_block_reward_era(d) for d in pydates], dtype=np.int32)

    # Pi Cycle indicator: 350-day MA * 2 vs 111-day MA crossover
    c = pd.Series(_np(df['close']).astype(np.float64), index=_cpu_idx) if 'close' in (list(df.columns) if _gpu else df.columns) else None
    if c is not None:
        c = pd.to_numeric(c, errors='coerce')
        ma_350x2 = c.rolling(350, min_periods=200).mean() * 2
        ma_111 = c.rolling(111, min_periods=60).mean()
        out['pi_cycle_cross'] = ((ma_111 > ma_350x2) &
                                  (ma_111.shift(1) <= ma_350x2.shift(1))).astype(int)
        out['pi_cycle_ratio'] = ma_111 / ma_350x2

    return out


# ============================================================
# COMPUTE KNN FEATURES
# ============================================================

def compute_knn_features(df: pd.DataFrame, tf_name: str = '1h') -> pd.DataFrame:
    """
    Compute KNN pattern similarity features.
    KNN engine already uses CuPy internally — just extract numpy arrays.
    """
    _gpu = _is_gpu(df)
    # Extract CPU index + numpy arrays directly — no full _to_cpu() needed
    if _gpu:
        _cpu_idx = pd.DatetimeIndex(_np(df.index))
    else:
        _cpu_idx = df.index
    out = pd.DataFrame(index=_cpu_idx)
    try:
        closes_arr = _np(df['close']).astype(np.float64)
        opens_arr = _np(df['open']).astype(np.float64)
        knn_feats = knn_features_from_ohlcv(opens_arr, closes_arr, tf_name, walkforward=True)
        for col, vals in knn_feats.items():
            out[col] = vals
    except Exception:
        pass
    return out.to_pandas() if hasattr(out, 'to_pandas') else out


# ============================================================
# SPACE WEATHER FEATURES
# ============================================================

def compute_space_weather_features(df: pd.DataFrame,
                                   space_weather_df: pd.DataFrame,
                                   tf_name: str = '1h') -> pd.DataFrame:
    """
    Compute space weather features from pre-loaded space_weather DataFrame.

    Pure function -- NO database calls. The caller must load data from
    space_weather.db and/or kp_history.txt and forward-fill to match bar
    frequency before calling this.

    Args:
        df: OHLCV DataFrame with DatetimeIndex
        space_weather_df: DataFrame with DatetimeIndex containing:
            kp_index, sunspot_number, solar_flux_f107,
            solar_wind_speed, solar_wind_bz,
            r_scale, s_scale, g_scale
            (already forward-filled to bar frequency)
        tf_name: timeframe name for TF_CONFIG lookup

    Returns:
        DataFrame with space weather features. NaN for missing data.
    """
    _gpu = _is_gpu(df)
    # Merges with external pandas df + _bars_since_event loops — extract CPU index, avoid full _to_cpu()
    if _gpu:
        _cpu_idx = pd.DatetimeIndex(_np(df.index))
    else:
        _cpu_idx = df.index
    cfg = TF_CONFIG.get(tf_name, TF_CONFIG['1h'])
    bucket_seconds = cfg['bucket_seconds']
    bars_per_day = max(1, 86400 // bucket_seconds)

    out = pd.DataFrame(index=_cpu_idx)

    if space_weather_df is None or space_weather_df.empty:
        return out

    sw = space_weather_df.copy()

    # Align space weather data to OHLCV index via reindex + ffill
    # Strip timezone to avoid datetime64[ns,UTC] vs datetime64[ns] mismatch
    if hasattr(sw.index, 'tz') and sw.index.tz is not None:
        sw.index = sw.index.tz_localize(None)
    _df_idx_sw = _strip_tz(_cpu_idx)
    if not sw.index.equals(_df_idx_sw):
        sw = sw.reindex(_df_idx_sw, method='ffill')
        sw.index = _cpu_idx

    # ------------------------------------------------------------------
    # RAW FEATURES
    # ------------------------------------------------------------------
    raw_cols = {
        'kp_index': 'sw_kp_index',
        'sunspot_number': 'sw_sunspot_number',
        'solar_flux_f107': 'sw_solar_flux_f107',
        'solar_wind_speed': 'sw_solar_wind_speed',
        'solar_wind_bz': 'sw_solar_wind_bz',
        'r_scale': 'sw_noaa_r_scale',
        's_scale': 'sw_noaa_s_scale',
        'g_scale': 'sw_noaa_g_scale',
    }
    for src_col, dst_col in raw_cols.items():
        if src_col in sw.columns:
            out[dst_col] = pd.to_numeric(sw[src_col], errors='coerce')

    # ------------------------------------------------------------------
    # DERIVED FEATURES
    # ------------------------------------------------------------------
    kp = out.get('sw_kp_index')
    if kp is not None:
        out['sw_kp_is_storm'] = (kp >= 5).astype(float)
        out['sw_kp_is_severe'] = (kp >= 7).astype(float)
        # NaN propagation: where kp is NaN, flags should be NaN
        out.loc[kp.isna(), 'sw_kp_is_storm'] = np.nan
        out.loc[kp.isna(), 'sw_kp_is_severe'] = np.nan

        # Kp rate of change over 3 days
        shift_3d = 3 * bars_per_day
        if shift_3d > 0:
            out['sw_kp_delta_3d'] = kp - kp.shift(shift_3d)

    # Solar cycle phase from sunspot 365d rolling mean
    sn = out.get('sw_sunspot_number')
    if sn is not None:
        rolling_365d = bars_per_day * 365
        if rolling_365d >= 2:
            sn_smooth = sn.rolling(window=min(rolling_365d, len(sn)),
                                   min_periods=max(1, bars_per_day * 30)).mean()
            sn_diff = sn_smooth.diff()
            # ascending=1, peak=2, descending=3 (encode as numeric)
            phase = pd.Series(np.nan, index=_cpu_idx)
            phase[sn_diff > 0] = 1.0    # ascending
            phase[sn_diff < 0] = 3.0    # descending
            # peak: ascending->descending transition zone (diff near zero, high value)
            sn_median = sn_smooth.median()
            phase[(sn_diff.abs() < 0.5) & (sn_smooth > sn_median)] = 2.0  # peak
            out['sw_solar_cycle_phase'] = phase

    # ------------------------------------------------------------------
    # DECAY FEATURES
    # ------------------------------------------------------------------
    if kp is not None:
        # bars_since_storm (kp >= 5)
        storm_mask = (kp >= 5).astype(float)
        storm_mask[kp.isna()] = np.nan
        bss = _bars_since_event(storm_mask)
        out['sw_bars_since_storm'] = bss
        out['sw_storm_decay'] = np.exp(-0.1 * bss)

        # bars_since_severe (kp >= 7)
        severe_mask = (kp >= 7).astype(float)
        severe_mask[kp.isna()] = np.nan
        bsv = _bars_since_event(severe_mask)
        out['sw_bars_since_severe'] = bsv
        out['sw_severe_decay'] = np.exp(-0.1 * bsv)

        # post_storm_7d: 1 if within 7 days after kp >= 7 (bullish signal)
        bars_7d = 7 * bars_per_day
        out['sw_post_storm_7d'] = (bsv <= bars_7d).astype(float)
        out.loc[bsv.isna(), 'sw_post_storm_7d'] = np.nan

    # ------------------------------------------------------------------
    # CROSS-FEATURES (with astrology columns if present)
    # ------------------------------------------------------------------
    _df_cols = list(df.columns) if _gpu else df.columns
    if kp is not None:
        if 'west_moon_phase' in _df_cols:
            moon = pd.to_numeric(pd.Series(_np(df['west_moon_phase']), index=_cpu_idx), errors='coerce')
            out['sw_kp_x_moon_phase'] = kp * moon

        # Full moon ~ phase day 14-16 out of 29.5 day cycle (phase value ~14)
        # Check for is_full_moon or derive from west_moon_phase
        if 'west_moon_phase' in _df_cols:
            moon = pd.to_numeric(pd.Series(_np(df['west_moon_phase']), index=_cpu_idx), errors='coerce')
            is_full = ((moon >= 13) & (moon <= 16)).astype(float)
            is_full[moon.isna()] = np.nan
            storm_flag = out.get('sw_kp_is_storm')
            if storm_flag is not None:
                out['sw_storm_x_full_moon'] = storm_flag * is_full

    return out


# ============================================================
# LUNAR / ELECTROMAGNETIC FEATURES
# ============================================================

def compute_lunar_electromagnetic_features(df: pd.DataFrame,
                                           tf_name: str = '1d') -> pd.DataFrame:
    """
    Lunar distance, solunar periods, biorhythm, tidal force, synodic cycles.

    Pure function -- NO database calls. Uses PyEphem for per-date lunar
    computations and numpy broadcasting for cycle features.

    Args:
        df: OHLCV DataFrame with DatetimeIndex
        tf_name: timeframe name for TF_CONFIG lookup

    Returns:
        DataFrame with lunar/electromagnetic features. NaN for missing data.
    """
    from astrology_engine import get_moon_distance, get_lunar_node_sign, get_solunar_period, get_tidal_force

    _gpu = _is_gpu(df)
    _cpu_idx = pd.DatetimeIndex(_np(df.index)) if _gpu else df.index
    n = len(df)

    # Accumulate all features in dict, then build DataFrame once
    feat_dict = {}

    # ------------------------------------------------------------------
    # BTC genesis biorhythm cycles (vectorized numpy)
    # ------------------------------------------------------------------
    GENESIS = pd.Timestamp('2009-01-03')
    if hasattr(_cpu_idx, 'tz') and _cpu_idx.tz is not None:
        days_since = (_cpu_idx - GENESIS.tz_localize(_cpu_idx.tz)).total_seconds() / 86400.0
    else:
        days_since = (_cpu_idx - GENESIS).total_seconds() / 86400.0
    days_arr = days_since.values.astype(np.float64)

    # Core biorhythm cycles
    feat_dict['biorhythm_physical'] = np.sin(2.0 * np.pi * days_arr / 23.0)
    feat_dict['biorhythm_emotional'] = np.sin(2.0 * np.pi * days_arr / 28.0)
    feat_dict['biorhythm_intellectual'] = np.sin(2.0 * np.pi * days_arr / 33.0)

    # Critical days (zero crossings) -- near-zero = transitional energy
    feat_dict['biorhythm_physical_critical'] = (np.abs(feat_dict['biorhythm_physical']) < 0.05).astype(np.float64)
    feat_dict['biorhythm_emotional_critical'] = (np.abs(feat_dict['biorhythm_emotional']) < 0.05).astype(np.float64)
    feat_dict['biorhythm_intellectual_critical'] = (np.abs(feat_dict['biorhythm_intellectual']) < 0.05).astype(np.float64)

    # ------------------------------------------------------------------
    # Planetary synodic cycles (vectorized numpy sin/cos encoding)
    # ------------------------------------------------------------------
    # Venus synodic: 583.9 days
    feat_dict['venus_synodic_sin'] = np.sin(2.0 * np.pi * days_arr / 583.9)
    feat_dict['venus_synodic_cos'] = np.cos(2.0 * np.pi * days_arr / 583.9)
    # Mars synodic: 779.9 days
    feat_dict['mars_synodic_sin'] = np.sin(2.0 * np.pi * days_arr / 779.9)
    feat_dict['mars_synodic_cos'] = np.cos(2.0 * np.pi * days_arr / 779.9)
    # Saturn cycle: ~10759 days (29.46 years)
    feat_dict['saturn_cycle_sin'] = np.sin(2.0 * np.pi * days_arr / 10759.0)
    feat_dict['saturn_cycle_cos'] = np.cos(2.0 * np.pi * days_arr / 10759.0)
    # Lunar node cycle: 6798 days (18.6 years / Saros)
    feat_dict['lunar_node_cycle_sin'] = np.sin(2.0 * np.pi * days_arr / 6798.0)
    feat_dict['lunar_node_cycle_cos'] = np.cos(2.0 * np.pi * days_arr / 6798.0)
    # Metonic cycle: 6940 days (19 years)
    feat_dict['metonic_cycle_sin'] = np.sin(2.0 * np.pi * days_arr / 6940.0)
    feat_dict['metonic_cycle_cos'] = np.cos(2.0 * np.pi * days_arr / 6940.0)

    # ------------------------------------------------------------------
    # Moon distance, solunar, lunar node sign, tidal force — PyEphem per unique date
    # (slow per call, so deduplicate to unique dates then map back)
    # ------------------------------------------------------------------
    df_dates = _cpu_idx.date
    unique_dates = pd.Series(df_dates).unique()

    moon_records = {}
    for d in unique_dates:
        dt = datetime(d.year, d.month, d.day)
        md = get_moon_distance(dt)
        sl = get_solunar_period(dt)
        ln = get_lunar_node_sign(dt)
        tf = get_tidal_force(dt)
        moon_records[d] = {**md, **sl, **ln, **tf}

    moon_df = pd.DataFrame.from_dict(moon_records, orient='index')
    date_series = pd.Series(df_dates)
    for col in moon_df.columns:
        mapping = moon_df[col].to_dict()
        feat_dict[col] = date_series.map(mapping).values.astype(np.float64)

    # Build output DataFrame in one shot
    out = pd.DataFrame(feat_dict, index=_cpu_idx)
    return out


@njit(cache=True)
def _bars_since_event_kernel(event_arr):
    """Numba-compiled bars-since-event kernel. ~50-100x faster than Python loop."""
    n = len(event_arr)
    out = np.full(n, np.nan)
    last = -1
    for i in range(n):
        if np.isnan(event_arr[i]):
            out[i] = np.nan
            continue
        if event_arr[i] >= 1.0:
            last = i
            out[i] = 0.0
        elif last >= 0:
            out[i] = float(i - last)
    return out


def _bars_since_event(event_series: pd.Series) -> pd.Series:
    """
    Compute bars since last True event in a boolean-like series.

    Args:
        event_series: Series with 1.0 for events, 0.0 for non-events, NaN for missing

    Returns:
        Series with bar count since last event. NaN before first event.
    """
    vals = _np(event_series).astype(np.float64)
    result = _bars_since_event_kernel(vals)
    return pd.Series(result, index=event_series.index)


# ============================================================
# COMPUTE TARGETS
# ============================================================

TRIPLE_BARRIER_CONFIG = {
    '15m': {'tp_atr_mult': 2.0, 'sl_atr_mult': 2.0, 'max_hold_bars': 32},
    '1h':  {'tp_atr_mult': 2.0, 'sl_atr_mult': 2.0, 'max_hold_bars': 24},
    '4h':  {'tp_atr_mult': 2.5, 'sl_atr_mult': 2.5, 'max_hold_bars': 16},
    '1d':  {'tp_atr_mult': 3.0, 'sl_atr_mult': 3.0, 'max_hold_bars': 10},
    '1w':  {'tp_atr_mult': 3.0, 'sl_atr_mult': 3.0, 'max_hold_bars': 6},
}


@njit(cache=True)
def _triple_barrier_atr_kernel(h, l, c, n):
    """Numba-compiled ATR(14) via true range + 14-bar SMA."""
    tr = np.empty(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        hl = h[i] - l[i]
        hc = abs(h[i] - c[i - 1])
        lc = abs(l[i] - c[i - 1])
        tr[i] = hl
        if hc > tr[i]:
            tr[i] = hc
        if lc > tr[i]:
            tr[i] = lc
    atr = np.full(n, np.nan)
    for i in range(13, n):
        s = 0.0
        for k in range(i - 13, i + 1):
            s += tr[k]
        atr[i] = s / 14.0
    return atr


@njit(cache=True)
def _triple_barrier_label_kernel(c, h, l, atr, tp_mult, sl_mult, max_hold):
    """Numba-compiled triple-barrier labeling loop. ~50-100x faster than Python."""
    n = len(c)
    labels = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(atr[i]) or np.isnan(c[i]):
            continue
        tp_price = c[i] + tp_mult * atr[i]
        sl_price = c[i] - sl_mult * atr[i]
        end_bar = i + max_hold
        if end_bar > n - 1:
            end_bar = n - 1
        if i + 1 > end_bar:
            continue
        hit = False
        for j in range(i + 1, end_bar + 1):
            if h[j] >= tp_price:
                labels[i] = 2.0  # LONG
                hit = True
                break
            if l[j] <= sl_price:
                labels[i] = 0.0  # SHORT
                hit = True
                break
        if not hit:
            labels[i] = 1.0  # FLAT -- time expiry
    return labels


def compute_triple_barrier_labels(df, tf_name):
    """
    Compute triple-barrier labels for each bar.

    For each bar i, look forward up to max_hold_bars and check which
    barrier is hit FIRST:
      - Price rises by tp_atr_mult * ATR(14) -> LONG  (2)
      - Price falls by sl_atr_mult * ATR(14) -> SHORT (0)
      - Neither hit within max_hold_bars      -> FLAT  (1)

    Args:
        df: DataFrame with 'high', 'low', 'close' columns.
        tf_name: timeframe name for config lookup.

    Returns:
        numpy array of labels (0=SHORT, 1=FLAT, 2=LONG), length = len(df).
        Bars near the end without enough forward data get NaN.
    """
    tb_cfg = TRIPLE_BARRIER_CONFIG.get(tf_name, TRIPLE_BARRIER_CONFIG['1h'])
    tp_mult = tb_cfg['tp_atr_mult']
    sl_mult = tb_cfg['sl_atr_mult']
    max_hold = tb_cfg['max_hold_bars']

    c = df['close'].astype(float).values
    h = df['high'].astype(float).values
    l = df['low'].astype(float).values
    n = len(c)

    # ATR(14) -- true range then 14-bar SMA (Numba-compiled)
    atr = _triple_barrier_atr_kernel(h, l, c, n)

    # Triple-barrier labeling (Numba-compiled)
    labels = _triple_barrier_label_kernel(c, h, l, atr, tp_mult, sl_mult, max_hold)

    return labels


def compute_targets(df: pd.DataFrame, tf_name: str = '1h') -> pd.DataFrame:
    """
    Compute prediction targets (future returns + triple-barrier labels).
    """
    _gpu = _is_gpu(df)
    _m = cudf if (_HAS_CUDF and _gpu) else pd
    cfg = TF_CONFIG.get(tf_name, TF_CONFIG['1h'])
    out = _m.DataFrame(index=df.index)
    c = df['close'].astype(float)

    # Next-bar return and direction (existing binary targets -- kept for backward compat)
    out[f'next_{_tf_label_from_name(tf_name)}_return'] = (c.shift(-1) - c) / c * 100
    out[f'next_{_tf_label_from_name(tf_name)}_direction'] = (out[f'next_{_tf_label_from_name(tf_name)}_return'] > 0).astype(int)

    # Multi-bar returns
    for bars in cfg['return_bars'][1:]:  # skip 1-bar (already have next return)
        out[f'next_{bars}bar_return'] = (c.shift(-bars) - c) / c * 100

    # Triple-barrier label (NEW -- 0=SHORT, 1=FLAT, 2=LONG)
    # compute_triple_barrier_labels uses Numba @njit loops — extract values directly
    # Build a lightweight pandas df with just the columns needed (close, high, low)
    _tb_df = pd.DataFrame({
        'close': _np(df['close']).astype(np.float64),
        'high': _np(df['high']).astype(np.float64),
        'low': _np(df['low']).astype(np.float64),
    })
    out['triple_barrier_label'] = compute_triple_barrier_labels(_tb_df, tf_name)

    return out.to_pandas() if hasattr(out, 'to_pandas') else out


# ============================================================
# ============================================================
# MAIN ENTRY POINT
# ============================================================

def build_all_features(ohlcv: pd.DataFrame, esoteric_frames: dict,
                       tf_name: str, mode: str = 'backfill',
                       htf_data: dict = None,
                       astro_cache: dict = None,
                       space_weather_df: pd.DataFrame = None,
                       market_signals: dict = None,
                       include_targets: bool = True,
                       include_knn: bool = True) -> pd.DataFrame:
    """
    Main entry point. Computes ALL features for a timeframe.

    Args:
        ohlcv: OHLCV candle data with DatetimeIndex.
            Required columns: open, high, low, close, volume.
            Optional: quote_volume, trades, taker_buy_volume, taker_buy_quote, open_time.
        esoteric_frames: {
            'tweets': df,    # from tweets.db
            'news': df,      # from news_articles.db
            'sports': {'games': df, 'horse_races': df},
            'onchain': df,   # from onchain_data.db
            'macro': df,     # from macro_data.db (date-indexed)
        }
        tf_name: '5m', '15m', '1h', '4h', '1d', '1w'
        mode: 'backfill' (full history) or 'live' (small window)
        htf_data: {'4h': df_4h, '1d': df_1d, '1w': df_1w}
        astro_cache: {
            'ephemeris': df,       # from ephemeris_cache.db (date-indexed)
            'astrology': df,       # from astrology_full.db (date-indexed)
            'fear_greed': df,      # from fear_greed.db (date-indexed)
            'google_trends': df,   # from google_trends.db (date-indexed)
            'funding_daily': df,   # daily avg funding (date-indexed)
        }
        space_weather_df: DataFrame (DatetimeIndex) from space_weather.db
            and/or kp_history.txt, forward-filled to bar frequency.
            Columns: kp_index, sunspot_number, solar_flux_f107,
            solar_wind_speed, solar_wind_bz, r_scale, s_scale, g_scale
        market_signals: {
            'defi_tvl': df,        # from DeFi TVL streamer
            'btc_dominance': df,   # from CoinGecko/CMC
            'mining_stats': df,    # from blockchain.info
        }
        include_targets: whether to compute future-looking targets
        include_knn: whether to compute KNN features (slow)

    Returns:
        DataFrame with all features. NaN for missing esoteric data.
    """
    cfg = TF_CONFIG.get(tf_name, TF_CONFIG['1h'])
    bucket_seconds = cfg['bucket_seconds']

    if astro_cache is None:
        astro_cache = {}
    if esoteric_frames is None:
        esoteric_frames = {}

    # Preserve OHLCV columns
    df = ohlcv.copy()

    # Strip timezone from index globally — prevents datetime64[ns,UTC] vs datetime64[ns]
    # mismatch errors in cudf.pandas fallback paths (reindex, merge, etc.)
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Also strip TZ from space_weather_df if provided
    if space_weather_df is not None and hasattr(space_weather_df.index, 'tz') and space_weather_df.index.tz is not None:
        space_weather_df = space_weather_df.copy()
        space_weather_df.index = space_weather_df.index.tz_localize(None)

    # Strip TZ from htf_data indices
    if htf_data:
        for htf_key in htf_data:
            if isinstance(htf_data[htf_key], pd.DataFrame):
                if hasattr(htf_data[htf_key].index, 'tz') and htf_data[htf_key].index.tz is not None:
                    htf_data[htf_key] = htf_data[htf_key].copy()
                    htf_data[htf_key].index = htf_data[htf_key].index.tz_localize(None)

    # Convert to GPU DataFrame if available
    _gpu_mode = _HAS_CUDF
    if _gpu_mode:
        try:
            df = _to_gpu(df)
            print(f"  [GPU MODE] DataFrame on GPU ({len(df)} rows)")
        except Exception as e:
            print(f"  [CPU MODE] GPU conversion failed: {e}")
            _gpu_mode = False

    # result stays pandas for safe column assignment from compute functions.
    # Functions that benefit from GPU rolling (composite) get a cuDF copy on-the-fly.
    if _gpu_mode:
        _result_idx = pd.DatetimeIndex(_np(df.index))
    else:
        _result_idx = df.index
    result = pd.DataFrame(index=_result_idx)
    result['open'] = _np(df['open'])
    result['high'] = _np(df['high'])
    result['low'] = _np(df['low'])
    result['close'] = _np(df['close'])
    result['volume'] = _np(df['volume'])
    _df_cols_init = list(df.columns) if _gpu_mode else df.columns
    for extra in ['quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote']:
        if extra in _df_cols_init:
            result[extra] = _np(df[extra])

    # 1. TA features — run on GPU (cuDF rolling/ewm/shift accelerated)
    t0 = time.time()
    ta_feats = compute_ta_features(df, tf_name)  # pass cuDF directly
    print(f"    TA features: {time.time()-t0:.1f}s ({len(ta_feats.columns)} cols)")
    for col in ta_feats.columns:
        result[col] = ta_feats[col]

    # ALL compute functions handle GPU/CPU internally — pass df (cuDF) directly
    # Each function detects _is_gpu(df) and converts to CPU as needed

    # 1b. Fractional differentiation features (after TA, needs close/volume)
    t0 = time.time()
    frac_feats = compute_frac_diff_features(df)
    print(f"    Frac diff features: {time.time()-t0:.1f}s ({len(frac_feats.columns)} cols)")
    for col in frac_feats.columns:
        result[col] = frac_feats[col]

    # 2. Time features
    t0 = time.time()
    time_feats = compute_time_features(df, tf_name)
    print(f"    Time features: {time.time()-t0:.1f}s ({len(time_feats.columns)} cols)")
    for col in time_feats.columns:
        result[col] = time_feats[col]

    # 3. Numerology features
    t0 = time.time()
    num_feats = compute_numerology_features(df)
    print(f"    Numerology features: {time.time()-t0:.1f}s ({len(num_feats.columns)} cols)")
    for col in num_feats.columns:
        result[col] = num_feats[col]

    # 3b. Numerology expansion (Lo Shu, angel numbers, Haramein, Pythagorean challenges)
    t0 = time.time()
    numx_feats = compute_numerology_expansion_features(df, tf_name)
    print(f"    Numerology expansion features: {time.time()-t0:.1f}s ({len(numx_feats.columns)} cols)")
    for col in numx_feats.columns:
        result[col] = numx_feats[col]

    # 3c. Vortex math & sacred geometry features
    t0 = time.time()
    vortex_feats = compute_vortex_sacred_geometry_features(df, tf_name)
    print(f"    Vortex/Sacred Geometry features: {time.time()-t0:.1f}s ({len(vortex_feats.columns)} cols)")
    for col in vortex_feats.columns:
        result[col] = vortex_feats[col]

    # 4. Astrology features (from cached daily data)
    t0 = time.time()
    astro_feats = compute_astrology_features(df, astro_cache)
    print(f"    Astrology features: {time.time()-t0:.1f}s ({len(astro_feats.columns)} cols)")
    for col in astro_feats.columns:
        result[col] = astro_feats[col]

    # 4b. Planetary expansion features (speeds, combustion, dignity, synodic, decan, stars)
    t0 = time.time()
    planet_exp_feats = compute_planetary_expansion_features(df)
    print(f"    Planetary expansion features: {time.time()-t0:.1f}s ({len(planet_exp_feats.columns)} cols)")
    for col in planet_exp_feats.columns:
        result[col] = planet_exp_feats[col]

    # 5. Esoteric features — GPU BATCH gematria/sentiment (zero .apply())
    t0 = time.time()
    eso_feats = compute_esoteric_features(
        df,
        tweets_df=esoteric_frames.get('tweets'),
        news_df=esoteric_frames.get('news'),
        sports_df=esoteric_frames.get('sports'),
        onchain_df=esoteric_frames.get('onchain'),
        macro_df=esoteric_frames.get('macro'),
        astro_cache=astro_cache,
        bucket_seconds=bucket_seconds,
    )
    print(f"    Esoteric features: {time.time()-t0:.1f}s ({len(eso_feats.columns)} cols)")
    for col in eso_feats.columns:
        result[col] = eso_feats[col]


    # 5.5 Event-timestamp astrology features — vectorized (no .apply())
    t0 = time.time()
    evt_astro_feats = compute_event_astrology(
        df,
        tweets_df=esoteric_frames.get('tweets'),
        news_df=esoteric_frames.get('news'),
        sports_df=esoteric_frames.get('sports'),
        bucket_seconds=bucket_seconds,
    )
    print(f"    Event astrology features: {time.time()-t0:.1f}s ({len(evt_astro_feats.columns)} cols)")
    for col in evt_astro_feats.columns:
        result[col] = evt_astro_feats[col]

    # 6. Higher TF features
    if htf_data:
        t0 = time.time()
        htf_feats = compute_higher_tf_features(df, htf_data)
        print(f"    Higher TF features: {time.time()-t0:.1f}s ({len(htf_feats.columns)} cols)")
        for col in htf_feats.columns:
            result[col] = htf_feats[col]

    # 7. Regime features
    t0 = time.time()
    regime_feats = compute_regime_features(df, tf_name)
    print(f"    Regime features: {time.time()-t0:.1f}s ({len(regime_feats.columns)} cols)")
    for col in regime_feats.columns:
        result[col] = regime_feats[col]

    # 8. Space weather features
    if space_weather_df is not None:
        t0 = time.time()
        sw_feats = compute_space_weather_features(df, space_weather_df, tf_name)
        print(f"    Space weather features: {time.time()-t0:.1f}s ({len(sw_feats.columns)} cols)")
        for col in sw_feats.columns:
            result[col] = sw_feats[col]

    # 8b. Lunar / electromagnetic features
    t0 = time.time()
    lunar_feats = compute_lunar_electromagnetic_features(df, tf_name)
    print(f"    Lunar/EM features: {time.time()-t0:.1f}s ({len(lunar_feats.columns)} cols)")
    for col in lunar_feats.columns:
        result[col] = lunar_feats[col]

    # 9. Confirmed cycle features
    t0 = time.time()
    cycle_feats = compute_cycle_features(df, tf_name)
    print(f"    Cycle features: {time.time()-t0:.1f}s ({len(cycle_feats.columns)} cols)")
    for col in cycle_feats.columns:
        result[col] = cycle_feats[col]

    # 10. Hebrew / cultural calendar features
    t0 = time.time()
    hebrew_feats = compute_hebrew_calendar_features(df)
    print(f"    Hebrew calendar features: {time.time()-t0:.1f}s ({len(hebrew_feats.columns)} cols)")
    for col in hebrew_feats.columns:
        result[col] = hebrew_feats[col]

    # 10b. Market calendar features (FOMC, options expiry, halving, tax, etc.)
    t0 = time.time()
    mkt_feats = compute_market_calendar_features(result)
    print(f"    Market calendar features: {time.time()-t0:.1f}s ({len(mkt_feats.columns)} cols)")
    for col in mkt_feats.columns:
        result[col] = mkt_feats[col]

    # 10c. Market signal features (DeFi TVL, BTC dominance, mining stats)
    if market_signals is not None:
        t0 = time.time()
        mkt_sig_feats = compute_market_signal_features(result, market_signals, tf_name)
        print(f"    Market signal features: {time.time()-t0:.1f}s ({len(mkt_sig_feats.columns)} cols)")
        for col in mkt_sig_feats.columns:
            result[col] = mkt_sig_feats[col]

    # 11. Composite vol-to-direction features (after cycles are in result)
    # Pass cuDF copy to get GPU-accelerated rolling/ewm ops inside composite
    t0 = time.time()
    _comp_input = _to_gpu(result) if _gpu_mode else result
    composite_feats = compute_composite_features(_comp_input, tf_name)
    del _comp_input
    print(f"    Composite features: {time.time()-t0:.1f}s ({len(composite_feats.columns)} cols)")
    for col in composite_feats.columns:
        result[col] = composite_feats[col]

    # 12. Cross-features that span multiple feature groups
    # Must run on CPU (uses pd.to_numeric, complex logic)
    t0 = time.time()
    _add_cross_features(result)
    print(f"    Cross features: {time.time()-t0:.1f}s")

    # 12b. Decay features (exponential decay since esoteric events)
    t0 = time.time()
    decay_feats = compute_decay_features(result, esoteric_frames, tf_name)
    print(f"    Decay features: {time.time()-t0:.1f}s ({len(decay_feats.columns)} cols)")
    for col in decay_feats.columns:
        result[col] = decay_feats[col]

    # 12c. Trend-context cross features — SKIPPED
    # These (tx_, ex_, dx_, vwap, range, ex_adv) are ALL generated by
    # v2_cross_generator.py as GPU sparse matrices. Generating them here
    # as DataFrame columns is a CPU bottleneck (239K+ pd.DataFrame(dict) calls)
    # and the cross generator explicitly skips these prefixes anyway.
    print(f"    Trend cross features: SKIPPED (handled by v2_cross_generator)")

    # 12d. LLM sentiment features (Haiku -- cached, cheap)
    if _HAS_LLM:
        try:
            t0 = time.time()
            # LLM features is an external module — pass result (already pandas)
            llm_feats = _compute_llm_features(
                result,
                tweets_df=esoteric_frames.get('tweets'),
                news_df=esoteric_frames.get('news'),
                bucket_seconds=bucket_seconds,
            )
            print(f"    LLM features: {time.time()-t0:.1f}s ({len(llm_feats.columns)} cols)")
            for col in llm_feats.columns:
                result[col] = llm_feats[col]
        except Exception as _llm_err:
            import logging as _lg
            _lg.getLogger('feature_library').warning(
                'LLM features failed (non-fatal): %s', _llm_err
            )

    # 13. KNN features (optional, slow)
    if include_knn and mode == 'backfill':
        t0 = time.time()
        knn_feats = compute_knn_features(df, tf_name)
        print(f"    KNN features: {time.time()-t0:.1f}s ({len(knn_feats.columns)} cols)")
        for col in knn_feats.columns:
            result[col] = knn_feats[col]

    # 13b. KNN x trend cross (HTF-aware regime)
    knn_dir = result.get('knn_direction')
    if knn_dir is not None:
        _knn = pd.to_numeric(knn_dir, errors='coerce')
        _knn_bull_sig = (_knn > 0).astype(float)
        _knn_bear_sig = (_knn < 0).astype(float)
        # Use same HTF regime logic
        _b = None
        if tf_name == '15m':
            _htf = result.get('h4_trend')
            if _htf is not None:
                _b = pd.to_numeric(_htf, errors='coerce')
                _br = (1 - _b)
        elif tf_name == '1h':
            _htf = result.get('d_trend')
            if _htf is not None:
                _b = pd.to_numeric(_htf, errors='coerce')
                _br = (1 - _b)
        elif tf_name == '4h':
            _htf = result.get('w_trend')
            if _htf is not None:
                _b = pd.to_numeric(_htf, errors='coerce')
                _br = (1 - _b)
        if _b is None:
            _b = pd.to_numeric(result.get('ema50_rising', np.nan), errors='coerce')
            _br = pd.to_numeric(result.get('ema50_declining', np.nan), errors='coerce')
        result['tx_knn_bull_x_bull'] = _knn_bull_sig * _b
        result['tx_knn_bull_x_bear'] = _knn_bull_sig * _br
        result['tx_knn_bear_x_bull'] = _knn_bear_sig * _b
        result['tx_knn_bear_x_bear'] = _knn_bear_sig * _br

    # 14. Targets (only for backfill)
    if include_targets and mode == 'backfill':
        t0 = time.time()
        target_feats = compute_targets(df, tf_name)
        print(f"    Target features: {time.time()-t0:.1f}s ({len(target_feats.columns)} cols)")
        for col in target_feats.columns:
            result[col] = target_feats[col]

    # Clean up string columns
    str_cols = [col for col in result.columns if result[col].dtype == 'object']
    if str_cols:
        result = result.drop(columns=str_cols, errors='ignore')

    # Ensure all numeric
    # Deduplicate columns (multiple compute functions may produce the same column name)
    if result.columns.duplicated().any():
        n_dupes = result.columns.duplicated().sum()
        print(f"    Deduplicating {n_dupes} duplicate columns...", flush=True)
        result = result.loc[:, ~result.columns.duplicated()]

    for col in result.columns:
        if result[col].dtype not in ['float64', 'int64', 'float32', 'int32', 'bool']:
            result[col] = pd.to_numeric(result[col], errors='coerce')

    # Downcast float64 → float32: LightGBM uses float32 histograms internally,
    # so float64 wastes 2x RAM with zero benefit. On 217K rows × 3000 cols,
    # this saves ~2.5 GB RAM and halves parquet file size.
    _f64_cols = result.select_dtypes(include=['float64']).columns
    if len(_f64_cols) > 0:
        result[_f64_cols] = result[_f64_cols].astype(np.float32)

    return result


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _add_cross_features(df: pd.DataFrame):
    """Add cross-domain interaction features in-place (~35+ features).

    Each cross-feature checks that BOTH input columns exist before computing.
    NaN propagates naturally through multiplication.
    """
    try:
        # Helper: safe column fetch as float
        def _col(name):
            if name in df.columns:
                return pd.to_numeric(df[name], errors='coerce')
            return None

        moon = _col('west_moon_phase')
        price_dr = _col('price_dr')

        # --- Original cross features (preserved) ---
        if moon is not None:
            moon_fill = moon
            tc = _col('tweet_gem_caution')
            if tc is not None:
                df['cross_moon_x_tweet_caution'] = moon_fill * tc
            ngc = _col('news_gem_caution')
            hca = _col('headline_caution_any')
            if ngc is not None:
                df['cross_moon_x_news_caution'] = moon_fill * ngc
            elif hca is not None:
                df['cross_moon_x_news_caution'] = moon_fill * hca
            suc = _col('sport_upset_count')
            if suc is not None:
                df['cross_moon_x_sport_upset'] = moon_fill * suc

        fg = _col('fear_greed')
        if fg is not None and moon is not None:
            df['fg_x_moon_phase'] = fg * moon

        # Price DR x Tweet gematria DR
        if price_dr is not None:
            tgdm_ord = _col('tweet_gem_dr_ord_mode')
            tgdm = _col('tweet_gem_dr_mode')
            if tgdm_ord is not None:
                df['cross_price_tweet_dr_match'] = (
                    price_dr == tgdm_ord).astype(int)
            elif tgdm is not None:
                df['cross_price_tweet_dr_match'] = (
                    price_dr == tgdm).astype(int)
            date_dr = _col('date_dr')
            if date_dr is not None:
                df['cross_date_price_dr_match'] = (
                    date_dr == price_dr).astype(int)

        # Tweet DR x News DR
        tgdm_ord2 = _col('tweet_gem_dr_ord_mode')
        ngdm_ord = _col('news_gem_dr_ord_mode')
        tgdm2 = _col('tweet_gem_dr_mode')
        hgdm = _col('headline_gem_dr_mode')
        if tgdm_ord2 is not None and ngdm_ord is not None:
            df['cross_tweet_news_gem_match'] = (
                tgdm_ord2 == ngdm_ord).astype(int)
        elif tgdm2 is not None and hgdm is not None:
            df['cross_tweet_news_gem_match'] = (
                tgdm2 == hgdm).astype(int)

        # ------------------------------------------------------------------
        # NEW CROSS-FEATURES (~35)
        # ------------------------------------------------------------------

        # Detect tf-specific tweet columns dynamically
        gold_tweet = None
        red_tweet = None
        green_tweet = None
        color_sent = None
        tweets_bucket = None
        news_sent = None
        for c_name in df.columns:
            if c_name.startswith('gold_tweet_this_') and gold_tweet is None:
                gold_tweet = _col(c_name)
            if c_name.startswith('red_tweet_this_') and red_tweet is None:
                red_tweet = _col(c_name)
            if c_name.startswith('green_tweet_this_') and green_tweet is None:
                green_tweet = _col(c_name)
            if c_name.startswith('color_sentiment_this_') and color_sent is None:
                color_sent = _col(c_name)
            if c_name.startswith('tweets_this_') and tweets_bucket is None:
                tweets_bucket = _col(c_name)
            if (c_name.startswith('news_sentiment_')
                    and c_name != 'news_sentiment_today'
                    and news_sent is None):
                news_sent = _col(c_name)
        # Fallback to daily columns
        if gold_tweet is None:
            gold_tweet = _col('gold_tweet_today')
        if red_tweet is None:
            red_tweet = _col('red_tweet_today')
        if green_tweet is None:
            green_tweet = _col('green_tweet_today')
        if color_sent is None:
            color_sent = _col('color_sentiment_today')
        if news_sent is None:
            news_sent = _col('news_sentiment_today')

        nakshatra = _col('vedic_nakshatra')
        key_nak = _col('vedic_key_nakshatra')
        merc_retro = _col('west_mercury_retrograde')
        eclipse = _col('eclipse_window')
        sport_upset = _col('sport_upset_count')
        consec_green = _col('consec_green')
        tweet_caps = _col('tweet_caps_any')
        shemitah = _col('shemitah_year')
        ema50_dec = _col('ema50_declining')
        day_13 = _col('day_13')
        fg_fear = _col('fg_extreme_fear')
        fg_greed = _col('fg_extreme_greed')
        kp_storm = _col('sw_kp_is_storm')
        funding = _col('funding_rate')
        headline_caution = _col('headline_caution_any')
        eso_vol = _col('esoteric_vol_score')
        ema50_slope = _col('ema50_slope')
        tweet_gem_dr = _col('tweet_gem_dr_mode')
        if tweet_gem_dr is None:
            tweet_gem_dr = _col('tweet_gem_dr_ord_mode')
        voc_moon = _col('voc_moon')

        # Compute volume zscore on the fly
        vol_raw = _col('volume')
        vol_zscore = None
        if vol_raw is not None:
            vol_mean = vol_raw.rolling(100, min_periods=20).mean()
            vol_std = vol_raw.rolling(100, min_periods=20).std().replace(
                0, np.nan)
            vol_zscore = (vol_raw - vol_mean) / vol_std

        # Compute open_interest zscore on the fly
        oi_raw = _col('open_interest')
        oi_zscore = None
        if oi_raw is not None:
            oi_mean = oi_raw.rolling(100, min_periods=20).mean()
            oi_std = oi_raw.rolling(100, min_periods=20).std().replace(
                0, np.nan)
            oi_zscore = (oi_raw - oi_mean) / oi_std

        # Derive is_full_moon and is_new_moon from west_moon_phase
        is_full_moon = None
        is_new_moon = None
        if moon is not None:
            is_full_moon = ((moon >= 13) & (moon <= 16)).astype(float)
            is_full_moon[moon.isna()] = np.nan
            is_new_moon = ((moon < 2) | (moon > 27.5)).astype(float)
            is_new_moon[moon.isna()] = np.nan

            # --- Moon-Trend Cross Features ---
            # Full moons amplify existing trend, new moons cause consolidation/reversal
            ema50_r = _col('ema50_rising')
            ema50_d = _col('ema50_declining')

            if ema50_r is not None:
                # Full moon in bull trend = trend amplifier (buy)
                df['cross_full_moon_x_bull'] = is_full_moon * ema50_r
                # New moon in bear trend = bounce/reversal signal
                if ema50_d is not None:
                    df['cross_new_moon_x_bear'] = is_new_moon * ema50_d
                # Full moon in bear trend = trend continues (short)
                if ema50_d is not None:
                    df['cross_full_moon_x_bear'] = is_full_moon * ema50_d
                # New moon in bull trend = consolidation/pause
                df['cross_new_moon_x_bull'] = is_new_moon * ema50_r

            # Days until next full/new moon (anticipation features)
            SYNODIC = 29.53059
            df['days_to_full_moon'] = np.where(
                moon <= 14.765, 14.765 - moon,
                SYNODIC - moon + 14.765
            )
            df['days_to_new_moon'] = np.where(
                moon <= 0, -moon,
                SYNODIC - moon
            )
            # Moon phase momentum: sin encoding that peaks at full moon
            df['moon_full_proximity'] = np.cos(2 * np.pi * (moon - 14.765) / SYNODIC)
            # Moon-trend interaction: continuous (positive = full moon + bull, negative = new moon + bear)
            ema50_slope_col = _col('ema50_slope')
            if ema50_slope_col is not None:
                df['moon_x_trend'] = df['moon_full_proximity'] * ema50_slope_col

            # --- Moon Window Features: what price did around recent moon events ---
            # These are BACKWARD-LOOKING only (no future data leak)
            # "What happened in the 5 days after the LAST full/new moon?"
            close = _col('close')
            if close is not None:
                # Returns over various lookback windows
                ret_1d = close.pct_change(1)
                ret_3d = close.pct_change(3)
                ret_5d = close.pct_change(5)

                # Price action leading INTO the current moon phase
                # (how price behaved in last 5 bars — context for the LSTM)
                df['moon_approach_return_3d'] = np.where(
                    (is_full_moon == 1) | (is_new_moon == 1), ret_3d, np.nan
                )
                df['moon_approach_return_5d'] = np.where(
                    (is_full_moon == 1) | (is_new_moon == 1), ret_5d, np.nan
                )
                # Forward-fill so bars AFTER the event know what happened at the event
                df['moon_approach_return_3d'] = df['moon_approach_return_3d'].ffill()
                df['moon_approach_return_5d'] = df['moon_approach_return_5d'].ffill()

                # Bars since last full/new moon (decay features)
                full_moon_bars = is_full_moon.copy()
                full_moon_bars[full_moon_bars != 1] = np.nan
                df['bars_since_full_moon'] = full_moon_bars.groupby(
                    full_moon_bars.notna().cumsum()).cumcount()
                new_moon_bars = is_new_moon.copy()
                new_moon_bars[new_moon_bars != 1] = np.nan
                df['bars_since_new_moon'] = new_moon_bars.groupby(
                    new_moon_bars.notna().cumsum()).cumcount()

                # Return since last full/new moon (how price moved AFTER the event)
                # This is the key feature: "price went up X% since last full moon"
                last_full_price = close.where(is_full_moon == 1).ffill()
                last_new_price = close.where(is_new_moon == 1).ffill()
                df['return_since_full_moon'] = (close - last_full_price) / last_full_price
                df['return_since_new_moon'] = (close - last_new_price) / last_new_price

        # Derive friday_13
        friday_13 = None
        if day_13 is not None:
            dow = pd.Series(df.index.dayofweek, index=df.index)
            friday_13 = ((dow == 4) & (day_13 == 1)).astype(float)

        # Derive master_number from price_dr
        master_number = None
        if price_dr is not None:
            master_number = price_dr.isin([11, 22, 33]).astype(float)
            master_number[price_dr.isna()] = np.nan

        # Schumann peak flag
        schumann_peak = None
        sch_sin = _col('schumann_133d_sin')
        if sch_sin is not None:
            schumann_peak = (sch_sin > 0.8).astype(float)
            schumann_peak[sch_sin.isna()] = np.nan

        # 1. moon_x_gold_tweet
        if moon is not None and gold_tweet is not None:
            df['cross_moon_x_gold_tweet'] = moon * gold_tweet

        # 2. nakshatra_x_red_tweet
        if nakshatra is not None and red_tweet is not None:
            df['cross_nakshatra_x_red_tweet'] = nakshatra * red_tweet

        # 3. mercury_retro_x_news_sentiment
        if merc_retro is not None and news_sent is not None:
            df['cross_mercury_retro_x_news_sent'] = merc_retro * news_sent

        # 4. eclipse_x_sport_upset
        if eclipse is not None and sport_upset is not None:
            df['cross_eclipse_x_sport_upset'] = eclipse * sport_upset

        # 5. planetary_hour_x_tweet_count (use bazi_stem_idx as proxy)
        if tweets_bucket is not None:
            ph_idx = _col('bazi_stem_idx')
            if ph_idx is not None:
                df['cross_astro_hour_x_tweet_count'] = ph_idx * tweets_bucket

        # 6. voc_moon_x_volume
        if voc_moon is not None and vol_zscore is not None:
            df['cross_voc_moon_x_volume'] = voc_moon * vol_zscore

        # 7. consec_green_x_caps_tweet
        if consec_green is not None and tweet_caps is not None:
            df['cross_consec_green_x_caps'] = consec_green * tweet_caps

        # 8. shmita_x_bear_regime
        if shemitah is not None and ema50_dec is not None:
            df['cross_shmita_x_bear'] = shemitah * ema50_dec

        # 9. day13_x_full_moon
        if day_13 is not None and is_full_moon is not None:
            df['cross_day13_x_full_moon'] = day_13 * is_full_moon

        # 10. friday13_x_red_tweet
        if friday_13 is not None and red_tweet is not None:
            df['cross_friday13_x_red_tweet'] = friday_13 * red_tweet

        # 11. master_number_x_nakshatra
        if master_number is not None and key_nak is not None:
            df['cross_master_num_x_nakshatra'] = master_number * key_nak

        # 12a. fg_extreme_fear_x_moon
        if fg_fear is not None and moon is not None:
            df['cross_fg_fear_x_moon'] = fg_fear * moon

        # 12b. fg_extreme_greed_x_moon
        if fg_greed is not None and moon is not None:
            df['cross_fg_greed_x_moon'] = fg_greed * moon

        # 13. kp_storm_x_funding
        if kp_storm is not None and funding is not None:
            df['cross_kp_storm_x_funding'] = kp_storm * funding

        # 14. eclipse_x_funding
        if eclipse is not None and funding is not None:
            df['cross_eclipse_x_funding'] = eclipse * funding

        # 15. storm_x_oi
        if kp_storm is not None and oi_zscore is not None:
            df['cross_kp_storm_x_oi'] = kp_storm * oi_zscore

        # 16. tweet_gem_x_price_dr (match = 1)
        if tweet_gem_dr is not None and price_dr is not None:
            df['cross_tweet_gem_price_dr_match'] = (
                tweet_gem_dr == price_dr).astype(float)

        # 17. news_caution_x_moon
        if headline_caution is not None and moon is not None:
            df['cross_news_caution_x_moon'] = headline_caution * moon

        # 18. sport_upset_x_mercury
        if sport_upset is not None and merc_retro is not None:
            df['cross_sport_upset_x_mercury'] = sport_upset * merc_retro

        # 19. schumann_peak_x_funding
        if schumann_peak is not None and funding is not None:
            df['cross_schumann_peak_x_funding'] = schumann_peak * funding

        # 20. vol_score_x_trend
        if eso_vol is not None and ema50_slope is not None:
            df['cross_vol_score_x_trend'] = eso_vol * ema50_slope

        # --- Additional multiplicative crosses for broader coverage ---

        # 21. eclipse_x_moon
        if eclipse is not None and moon is not None:
            df['cross_eclipse_x_moon'] = eclipse * moon

        # 22. kp_storm_x_moon
        if kp_storm is not None and moon is not None:
            df['cross_kp_storm_x_moon'] = kp_storm * moon

        # 23. gold_tweet_x_consec_green
        if gold_tweet is not None and consec_green is not None:
            df['cross_gold_tweet_x_consec_green'] = gold_tweet * consec_green

        # 24. red_tweet_x_ema50_declining
        if red_tweet is not None and ema50_dec is not None:
            df['cross_red_tweet_x_bear'] = red_tweet * ema50_dec

        # 25. fg_fear_x_kp_storm
        if fg_fear is not None and kp_storm is not None:
            df['cross_fg_fear_x_kp_storm'] = fg_fear * kp_storm

        # 26. fg_greed_x_funding
        if fg_greed is not None and funding is not None:
            df['cross_fg_greed_x_funding'] = fg_greed * funding

        # 27. nakshatra_x_moon
        if nakshatra is not None and moon is not None:
            df['cross_nakshatra_x_moon'] = nakshatra * moon

        # 28. eclipse_x_consec_green
        if eclipse is not None and consec_green is not None:
            df['cross_eclipse_x_consec_green'] = eclipse * consec_green

        # 29. shmita_x_kp_storm
        if shemitah is not None and kp_storm is not None:
            df['cross_shmita_x_kp_storm'] = shemitah * kp_storm

        # 30. day13_x_red_tweet
        if day_13 is not None and red_tweet is not None:
            df['cross_day13_x_red_tweet'] = day_13 * red_tweet

        # 31. gold_tweet_x_funding
        if gold_tweet is not None and funding is not None:
            df['cross_gold_tweet_x_funding'] = gold_tweet * funding

        # 32. caps_tweet_x_vol_zscore
        if tweet_caps is not None and vol_zscore is not None:
            df['cross_caps_tweet_x_vol'] = tweet_caps * vol_zscore

        # 33. fg_fear_x_eclipse
        if fg_fear is not None and eclipse is not None:
            df['cross_fg_fear_x_eclipse'] = fg_fear * eclipse

        # 34. sport_upset_x_full_moon
        if sport_upset is not None and is_full_moon is not None:
            df['cross_sport_upset_x_full_moon'] = sport_upset * is_full_moon

        # 35. master_number_x_eclipse
        if master_number is not None and eclipse is not None:
            df['cross_master_num_x_eclipse'] = master_number * eclipse

        # 36. schumann_peak_x_kp_storm
        if schumann_peak is not None and kp_storm is not None:
            df['cross_schumann_peak_x_kp_storm'] = schumann_peak * kp_storm

        # --- Color analysis cross features ---

        # 37. green_tweet_x_consec_green (green tweet + green candles = momentum)
        if green_tweet is not None and consec_green is not None:
            df['cross_green_tweet_x_consec_green'] = green_tweet * consec_green

        # 38. green_tweet_x_funding (green tweet + funding rate alignment)
        if green_tweet is not None and funding is not None:
            df['cross_green_tweet_x_funding'] = green_tweet * funding

        # 39. color_sentiment_x_moon (color sentiment modulated by lunar phase)
        if color_sent is not None and moon is not None:
            df['cross_color_sent_x_moon'] = color_sent * moon

        # 40. color_sentiment_x_fg (color sentiment vs fear/greed alignment)
        if color_sent is not None and fg_fear is not None:
            df['cross_color_sent_x_fg_fear'] = color_sent * fg_fear
        if color_sent is not None and fg_greed is not None:
            df['cross_color_sent_x_fg_greed'] = color_sent * fg_greed

        # ==============================================================
        # POWER CROSSES — from cross_test.py correlation analysis
        # These are the strongest untapped crosses found by testing
        # 5,784 combinations against next_1h_return
        # ==============================================================

        # --- Helper: safe binary/continuous fetch ---
        def _bin(name):
            """Get binary column, return None if missing."""
            s = _col(name)
            return s if s is not None else None

        def _cont_high(name, q=0.8):
            """Binarize continuous column at top quintile."""
            s = _col(name)
            if s is None: return None
            return (s > s.quantile(q)).astype(float)

        def _cont_low(name, q=0.2):
            """Binarize continuous column at bottom quintile."""
            s = _col(name)
            if s is None: return None
            return (s < s.quantile(q)).astype(float)

        def _cross(name, a, b):
            """Create cross feature in-place."""
            if a is not None and b is not None:
                df[name] = a * b

        # --- #73 (UP/reversal) x TA extremes (strongest single-number crosses) ---
        n73 = _bin('is_73')
        _cross('px_73_x_gann_high', n73, _cont_high('gann_sq9_distance'))
        _cross('px_73_x_bb_low', n73, _cont_low('bb_pctb_20'))
        _cross('px_73_x_cci_low', n73, _cont_low('cci_20'))
        _cross('px_73_x_stoch_low', n73, _cont_low('stoch_k_14'))
        _cross('px_73_x_williams_low', n73, _cont_low('williams_r_14'))
        _cross('px_73_x_ema50_slope_low', n73, _cont_low('ema50_slope'))
        _cross('px_73_x_va_low', n73, _cont_low('value_area_position'))
        _cross('px_73_x_volume_spike', n73, _bin('volume_spike'))
        _cross('px_73_x_fvg_bearish', n73, _bin('fvg_bearish'))
        _cross('px_73_x_below_cloud', n73, _bin('ichimoku_below_cloud'))
        _cross('px_73_x_green_wave', n73, _bin('consensio_green_wave'))
        _cross('px_73_x_sar_flip', n73, _bin('sar_flip'))
        _cross('px_73_x_atr_high', n73, _cont_high('atr_14_pct'))
        _cross('px_73_x_cvd_slope_high', n73, _cont_high('cvd_slope'))

        # --- #73 x Orderbook (number energy validates orderflow) ---
        _cross('px_73_x_delta_high', n73, _cont_high('delta_ratio'))
        _cross('px_73_x_taker_high', n73, _cont_high('taker_buy_ratio'))
        _cross('px_73_x_cvd_div', n73, _bin('cvd_price_divergence'))
        _cross('px_73_x_delta_bar_high', n73, _cont_high('delta_bar'))
        _cross('px_73_x_vol_ratio_high', n73, _cont_high('volume_ratio'))

        # --- #37 (UP energy) x TA/Orderbook ---
        n37 = _bin('is_37')
        _cross('px_37_x_range_low', n37, _cont_low('range_position'))
        _cross('px_37_x_rsi_low', n37, _cont_low('rsi_14'))
        _cross('px_37_x_ema50_slope_low', n37, _cont_low('ema50_slope'))
        _cross('px_37_x_volume_spike', n37, _bin('volume_spike'))
        _cross('px_37_x_vol_ratio_high', n37, _cont_high('volume_ratio'))
        _cross('px_37_x_vol_ratio_low', n37, _cont_low('volume_ratio'))

        # --- #39 (mirror of 93) x TA/Orderbook ---
        n39 = _bin('is_39')
        _cross('px_39_x_atr_high', n39, _cont_high('atr_14_pct'))
        _cross('px_39_x_macd_high', n39, _cont_high('macd_histogram'))
        _cross('px_39_x_stoch_high', n39, _cont_high('stoch_k_14'))
        _cross('px_39_x_wyckoff_evr_high', n39, _cont_high('wyckoff_effort_vs_result'))
        _cross('px_39_x_doji', n39, _bin('doji'))
        _cross('px_39_x_cvd_div', n39, _bin('cvd_price_divergence'))
        _cross('px_39_x_vol_ratio_low', n39, _cont_low('volume_ratio'))

        # --- #93 (destruction) x TA/Orderbook (amplifies/invalidates signals) ---
        n93 = _bin('is_93')
        _cross('px_93_x_macd_high', n93, _cont_high('macd_histogram'))
        _cross('px_93_x_fvg_bearish', n93, _bin('fvg_bearish'))
        _cross('px_93_x_delta_bar_high', n93, _cont_high('delta_bar'))
        _cross('px_93_x_vol_ratio_high', n93, _cont_high('volume_ratio'))
        _cross('px_93_x_vol_ratio_low', n93, _cont_low('volume_ratio'))
        _cross('px_93_x_taker_high', n93, _cont_high('taker_buy_ratio'))

        # --- #19 (dump/surrender) x TA/Orderbook ---
        n19 = _bin('is_19')
        _cross('px_19_x_fvg_bearish', n19, _bin('fvg_bearish'))
        _cross('px_19_x_ema50_slope_high', n19, _cont_high('ema50_slope'))
        _cross('px_19_x_macd_high', n19, _cont_high('macd_histogram'))
        _cross('px_19_x_cvd_slope_high', n19, _cont_high('cvd_slope'))
        _cross('px_19_x_taker_high', n19, _cont_high('taker_buy_ratio'))

        # --- #17 (Freemasonry/kill) x TA ---
        n17 = _bin('is_17')
        _cross('px_17_x_fvg_bearish', n17, _bin('fvg_bearish'))

        # --- Day-of-month x TA ---
        d12 = _bin('is_day_12')
        _cross('px_day12_x_rsi_os', d12, _bin('rsi_14_os'))
        _cross('px_day12_x_red_wave', d12, _bin('consensio_red_wave'))
        d5 = _bin('is_day_5')
        _cross('px_day5_x_wyckoff_upthrust', d5, _bin('wyckoff_upthrust'))
        d20 = _bin('is_day_20')
        _cross('px_day20_x_rsi_os', d20, _bin('rsi_14_os'))
        _cross('px_day20_x_red_wave', d20, _bin('consensio_red_wave'))

        # --- Day-of-week x TA ---
        thu = _bin('is_thursday')
        _cross('px_thu_x_supertrend', thu, _bin('supertrend_flip'))
        _cross('px_thu_x_range_low', thu, _cont_low('range_position'))
        sun = _bin('is_sunday')
        _cross('px_sun_x_death_cross', sun, _bin('death_cross'))

        # --- Price DR x TA ---
        pdr3 = _bin('price_dr_3')
        _cross('px_pdr3_x_supertrend', pdr3, _bin('supertrend_flip'))
        pdr6_ = _bin('price_dr_6')
        _cross('px_pdr6_x_supertrend', pdr6_, _bin('supertrend_flip'))

        # --- Date sum x TA ---
        mds7 = _bin('month_day_sum_7')
        _cross('px_mds7_x_rsi_os', mds7, _bin('rsi_14_os'))
        _cross('px_mds7_x_ichi_tk', mds7, _bin('ichimoku_tk_cross'))
        mds13 = _bin('month_day_sum_13')
        _cross('px_mds13_x_ichi_tk', mds13, _bin('ichimoku_tk_cross'))

        # --- Other esoteric x TA ---
        prime = _bin('day_is_prime')
        _cross('px_prime_x_wyckoff_spring', prime, _bin('wyckoff_spring'))
        fib_day = _bin('is_fibonacci_day')
        _cross('px_fibday_x_near_fib21', fib_day, _bin('near_fib_21'))
        d13 = _bin('day_13')
        _cross('px_day13_x_near_fib21', d13, _bin('near_fib_21'))
        n84 = _bin('is_84')
        _cross('px_84_x_sar_flip', n84, _bin('sar_flip'))
        n48 = _bin('is_48')
        _cross('px_48_x_obv_low', n48, _cont_low('obv'))
        _cross('px_48_x_macd_high', n48, _cont_high('macd_histogram'))
        pc322 = _bin('price_contains_322')
        _cross('px_p322_x_macd_high', pc322, _cont_high('macd_histogram'))
        _cross('px_p322_x_cvd_slope_low', pc322, _cont_low('cvd_slope'))
        pc113 = _bin('price_contains_113')
        _cross('px_p113_x_rsi_os', pc113, _bin('rsi_14_os'))
        btc213 = _bin('btc_213')
        _cross('px_213_x_wyckoff_evr_high', btc213, _cont_high('wyckoff_effort_vs_result'))
        _cross('px_213_x_vol_ratio_high', btc213, _cont_high('volume_ratio'))

        # --- GCP x TA ---
        gcp_ext = _bin('gcp_extreme')
        _cross('px_gcp_x_near_fib13', gcp_ext, _bin('near_fib_13'))
        _cross('px_gcp_x_near_fib21', gcp_ext, _bin('near_fib_21'))
        _cross('px_gcp_x_red_wave', gcp_ext, _bin('consensio_red_wave'))
        _cross('px_gcp_x_rsi_os', gcp_ext, _bin('rsi_14_os'))
        _cross('px_gcp_x_rsi_ob', gcp_ext, _bin('rsi_14_ob'))
        _cross('px_gcp_x_macd_cross_up', gcp_ext, _bin('macd_cross_up'))
        _cross('px_gcp_x_bull_engulf', gcp_ext, _bin('bull_engulfing'))
        _cross('px_gcp_x_shooting_star', gcp_ext, _bin('shooting_star'))
        _cross('px_gcp_x_bb_squeeze', gcp_ext, _bin('bb_squeeze_20'))
        _cross('px_gcp_x_vol_spike', gcp_ext, _bin('volume_spike'))
        _cross('px_gcp_x_sar_flip', gcp_ext, _bin('sar_flip'))
        _cross('px_gcp_x_fvg_bearish', gcp_ext, _bin('fvg_bearish'))
        _cross('px_gcp_x_atr_high', gcp_ext, _cont_high('atr_14_pct'))
        _cross('px_gcp_x_atr_low', gcp_ext, _cont_low('atr_14_pct'))
        _cross('px_gcp_x_range_low', gcp_ext, _cont_low('range_position'))
        _cross('px_gcp_x_va_high', gcp_ext, _cont_high('value_area_position'))
        _cross('px_gcp_x_obv_low', gcp_ext, _cont_low('obv'))
        _cross('px_gcp_x_obv_high', gcp_ext, _cont_high('obv'))

        # --- GCP x Orderbook ---
        _cross('px_gcp_x_delta_low', gcp_ext, _cont_low('delta_bar'))
        _cross('px_gcp_x_vol_ratio_high', gcp_ext, _cont_high('volume_ratio'))
        _cross('px_gcp_x_taker_low', gcp_ext, _cont_low('taker_buy_ratio'))
        _cross('px_gcp_x_cvd_div', gcp_ext, _bin('cvd_price_divergence'))
        _cross('px_gcp_x_funding', gcp_ext, _col('funding_regime'))

        # --- GCP x Esoteric (consciousness + number energy) ---
        gcp_dev = _col('gcp_deviation_mean')
        if gcp_dev is not None:
            gcp_abs = gcp_dev.abs()
            _cross('px_gcp_abs_x_73', gcp_abs, n73)
            _cross('px_gcp_abs_x_39', gcp_abs, n39)
            _cross('px_gcp_abs_x_93', gcp_abs, n93)
            _cross('px_gcp_abs_x_pdr9', gcp_abs, _bin('price_dr_9'))
            _cross('px_gcp_abs_x_pdr7', gcp_abs, _bin('price_dr_7'))
            _cross('px_gcp_abs_x_vortex', gcp_abs, _bin('vortex_369'))
            _cross('px_gcp_abs_x_mds11', gcp_abs, _bin('month_day_sum_11'))
        _cross('px_gcp_x_mayan9', gcp_ext, _bin('mayan_tone_9'))
        _cross('px_gcp_x_eclipse', gcp_ext, _bin('eclipse_window'))
        _cross('px_gcp_x_kp_storm', gcp_ext, _bin('sw_kp_is_storm'))

        # --- Esoteric x Esoteric (untapped combos) ---
        _cross('px_39_x_fg_greed', n39, fg_greed)
        _cross('px_39_x_pdr9', n39, _bin('price_dr_9'))
        _cross('px_39_x_pdr6', n39, pdr6_)
        _cross('px_73_x_pdr3', n73, pdr3)
        _cross('px_73_x_fib_day', n73, fib_day)
        _cross('px_73_x_prime_day', n73, prime)
        _cross('px_73_x_fg_fear', n73, fg_fear)
        _cross('px_37_x_pump', n37, _bin('pump_date'))
        _cross('px_37_x_fg_fear', n37, fg_fear)
        _cross('px_d12_x_doy_div9', d12, _bin('doy_div_9'))
        _cross('px_d12_x_fg_fear', d12, fg_fear)
        _cross('px_d12_x_mayan9', d12, _bin('mayan_tone_9'))
        _cross('px_wk_master_x_mayan9', _bin('week_master'), _bin('mayan_tone_9'))
        _cross('px_113_x_pdr7', _bin('is_113'), _bin('price_dr_7'))
        _cross('px_48_x_fg_greed', n48, fg_greed)
        _cross('px_48_x_mayan13', n48, _bin('mayan_tone_13'))
        _cross('px_thu_x_48', thu, n48)
        _cross('px_p213_x_pump', _bin('price_contains_213'), _bin('pump_date'))
        _cross('px_p113_x_wk_master', pc113, _bin('week_master'))

        # --- MIRROR NUMBER POWER CROSSES (from mirror_test.py) ---

        # DOY 72 (anti-pump, mirror of 27) — MEGA BEAR signal
        n72 = _bin('is_72')
        _cross('px_72_x_cvd_div', n72, _bin('cvd_price_divergence'))
        _cross('px_72_x_ema50_slope_low', n72, _cont_low('ema50_slope'))
        _cross('px_72_x_range_low', n72, _cont_low('range_position'))
        _cross('px_72_x_cvd_slope_high', n72, _cont_high('cvd_slope'))
        _cross('px_72_x_rsi_low', n72, _cont_low('rsi_14'))
        _cross('px_72_x_fg_fear', n72, fg_fear)
        _cross('px_72_x_stoch_low', n72, _cont_low('stoch_k_14'))
        _cross('px_72_x_delta_high', n72, _cont_high('delta_bar'))
        _cross('px_72_x_nakshatra', n72, _bin('vedic_key_nakshatra'))
        _cross('px_72_x_vol_ratio_high', n72, _cont_high('volume_ratio'))
        _cross('px_72_x_atr_high', n72, _cont_high('atr_14_pct'))
        _cross('px_72_x_bazi_clash', n72, _bin('bazi_day_clash_branch'))
        _cross('px_72_x_vol_spike', n72, _bin('volume_spike'))

        # DOY 71 (recovery, mirror of 17 kill) — BULL signal
        n71 = _bin('is_71')
        _cross('px_71_x_vol_spike', n71, _bin('volume_spike'))
        _cross('px_71_x_vol_ratio_high', n71, _cont_high('volume_ratio'))
        _cross('px_71_x_delta_high', n71, _cont_high('delta_bar'))
        _cross('px_71_x_cvd_slope_low', n71, _cont_low('cvd_slope'))
        _cross('px_71_x_fg_greed', n71, fg_greed)
        _cross('px_71_x_bb_low', n71, _cont_low('bb_pctb_20'))
        _cross('px_71_x_macd_low', n71, _cont_low('macd_histogram'))
        _cross('px_71_x_stoch_low', n71, _cont_low('stoch_k_14'))

        # DOY 311 (mirror of 113 bottom buy) — BULL confirmation
        n311 = _bin('is_311')
        _cross('px_311_x_taker_high', n311, _cont_high('taker_buy_ratio'))
        _cross('px_311_x_delta_high', n311, _cont_high('delta_bar'))
        _cross('px_311_x_macd_low', n311, _cont_low('macd_histogram'))
        _cross('px_311_x_cvd_slope_high', n311, _cont_high('cvd_slope'))
        _cross('px_311_x_obv_low', n311, _cont_low('obv'))

        # DOY 91 (mirror of 19 surrender) — resistance/fight back
        n91 = _bin('is_91')
        _cross('px_91_x_macd_high', n91, _cont_high('macd_histogram'))
        _cross('px_91_x_obv_low', n91, _cont_low('obv'))

        # DR mirror/completion crosses
        dr_mirror = _bin('dr_mirror_match')
        dr_comp = _bin('dr_completion_9')
        _cross('px_dr_mirror_x_fg_fear', dr_mirror, fg_fear)
        _cross('px_dr_mirror_x_fg_greed', dr_mirror, fg_greed)
        _cross('px_dr_mirror_x_vol_spike', dr_mirror, _bin('volume_spike'))
        _cross('px_dr_comp9_x_fg_fear', dr_comp, fg_fear)
        _cross('px_dr_comp9_x_fg_greed', dr_comp, fg_greed)
        _cross('px_dr_comp9_x_eclipse', dr_comp, _bin('eclipse_window'))

        # Price mirror contains crosses
        pc72 = _bin('price_contains_72')
        _cross('px_pc72_x_rsi_os', pc72, _bin('rsi_14_os'))
        _cross('px_pc72_x_macd_high', pc72, _cont_high('macd_histogram'))
        pc37 = _bin('price_contains_37')
        _cross('px_pc37_x_rsi_os', pc37, _bin('rsi_14_os'))
        pc73_p = _bin('price_contains_73')
        _cross('px_pc73_x_bb_low', pc73_p, _cont_low('bb_pctb_20'))

        # ════════════════════════════════════════════════════════════
        # SYSTEMATIC CROSS EXPANSION — Dynamic from CSV survivors
        # ════════════════════════════════════════════════════════════
        _integrate_systematic_crosses(df, _bin, _cont_high, _cont_low, _cross)

    except Exception:
        pass


def _integrate_systematic_crosses(df, _bin, _cont_high, _cont_low, _cross):
    """
    Dynamically generate cross features from systematic_cross_tester.py survivors.
    Reads systematic_cross_results_{tf}.csv and creates px_sys_ prefixed features.

    Context binarization rules:
      - Column ending '_HIGH' → top 20th percentile of base column
      - Column ending '_LOW'  → bottom 20th percentile of base column
      - No suffix → treat as binary (use directly)
    """
    import os, re

    # Find CSV files in same directory as the parquet data
    csv_candidates = []
    for f in os.listdir('.'):
        if f.startswith('systematic_cross_results_') and f.endswith('.csv') and f != 'systematic_cross_results_all.csv':
            csv_candidates.append(f)

    if not csv_candidates:
        # Try parent/project directory
        proj_dir = os.path.dirname(os.path.abspath(__file__))
        for f in os.listdir(proj_dir):
            if f.startswith('systematic_cross_results_') and f.endswith('.csv') and f != 'systematic_cross_results_all.csv':
                csv_candidates.append(os.path.join(proj_dir, f))

    if not csv_candidates:
        return

    # Load and deduplicate survivors across TFs
    all_survivors = []
    for csv_path in csv_candidates:
        try:
            survivors = pd.read_csv(csv_path)
            all_survivors.append(survivors)
        except Exception:
            continue

    if not all_survivors:
        return

    combined = pd.concat(all_survivors, ignore_index=True)

    # Deduplicate: keep the cross with best confidence across TFs
    combined = combined.sort_values('confidence', ascending=False).drop_duplicates(
        subset=['signal', 'context'], keep='first'
    )

    created = 0
    skipped = 0

    for _, row in combined.iterrows():
        sig_name = row['signal']
        ctx_name = row['context']
        confidence = row.get('confidence', 0.5)
        is_rare = row.get('rare_signal', False)

        # Build feature name
        # Truncate long names to keep column names manageable
        sig_short = sig_name[:20]
        ctx_short = ctx_name[:25]
        feat_name = f'px_sys_{sig_short}_x_{ctx_short}'

        # Skip if already exists
        if feat_name in df.columns:
            skipped += 1
            continue

        # Get signal array
        sig_arr = _bin(sig_name)
        if sig_arr is None:
            skipped += 1
            continue

        # Get context array — handle _HIGH/_LOW suffix
        if ctx_name.endswith('_HIGH'):
            base_col = ctx_name[:-5]
            ctx_arr = _cont_high(base_col)
        elif ctx_name.endswith('_LOW'):
            base_col = ctx_name[:-4]
            ctx_arr = _cont_low(base_col)
        else:
            ctx_arr = _bin(ctx_name)

        if ctx_arr is None:
            skipped += 1
            continue

        # Create cross
        _cross(feat_name, sig_arr, ctx_arr)

        created += 1

    if created > 0:
        print(f'    [SYS_CROSS] Created {created} systematic crosses ({skipped} skipped)')


def compute_decay_features(df: pd.DataFrame, esoteric_frames: dict,
                           tf_name: str) -> pd.DataFrame:
    """
    Compute exponential decay features for esoteric events (~25-35 features).

    For each event type, computes:
      - bars_since_EVENT: count of bars since last occurrence
      - EVENT_decay_fast: exp(-0.1 * bars_since)
      - EVENT_decay_slow: exp(-0.05 * bars_since) (for important events)

    Args:
        df: result DataFrame with all previously computed features
        esoteric_frames: dict (unused here, events detected from df columns)
        tf_name: timeframe name

    Returns:
        DataFrame with decay features. NaN for missing.
    """
    _gpu = _is_gpu(df)
    # Decay uses _bars_since_event (now @njit compiled) — extract CPU index, avoid full _to_cpu()
    if _gpu:
        _cpu_idx = pd.DatetimeIndex(_np(df.index))
        _df_cols = list(df.columns)
    else:
        _cpu_idx = df.index
        _df_cols = df.columns
    LAMBDA_FAST = 0.1
    LAMBDA_SLOW = 0.05

    out = pd.DataFrame(index=_cpu_idx)

    def _col(name):
        if name in _df_cols:
            return pd.to_numeric(pd.Series(_np(df[name]), index=_cpu_idx), errors='coerce')
        return None

    moon = _col('west_moon_phase')

    # Detect tf-specific tweet columns dynamically
    gold_tweet = None
    red_tweet = None
    green_tweet = None
    for c_name in _df_cols:
        if c_name.startswith('gold_tweet_this_') and gold_tweet is None:
            gold_tweet = _col(c_name)
        if c_name.startswith('red_tweet_this_') and red_tweet is None:
            red_tweet = _col(c_name)
        if c_name.startswith('green_tweet_this_') and green_tweet is None:
            green_tweet = _col(c_name)
    if gold_tweet is None:
        gold_tweet = _col('gold_tweet_today')
    if red_tweet is None:
        red_tweet = _col('red_tweet_today')
    if green_tweet is None:
        green_tweet = _col('green_tweet_today')

    # ------------------------------------------------------------------
    # Define events: (name, event_series, both_decays)
    # both_decays=True -> create fast AND slow decay
    # ------------------------------------------------------------------
    events = []

    # 1. Eclipse: transition from 0 to 1
    eclipse = _col('eclipse_window')
    if eclipse is not None:
        eclipse_onset = (
            (eclipse == 1) & (eclipse.shift(1).fillna(0) == 0)
        ).astype(float)
        eclipse_onset[eclipse.isna()] = np.nan
        events.append(('eclipse', eclipse_onset, True))

    # 2. Full moon: west_moon_phase near 14-15
    if moon is not None:
        is_full = ((moon >= 13) & (moon <= 16)).astype(float)
        is_full[moon.isna()] = np.nan
        events.append(('full_moon', is_full, True))

    # 3. New moon: west_moon_phase near 0 or 29
    if moon is not None:
        is_new = ((moon <= 1) | (moon >= 28)).astype(float)
        is_new[moon.isna()] = np.nan
        events.append(('new_moon', is_new, True))

    # 4. Gold tweet
    if gold_tweet is not None:
        gt_event = (gold_tweet >= 1).astype(float)
        gt_event[gold_tweet.isna()] = np.nan
        events.append(('gold_tweet', gt_event, True))

    # 5. Red tweet
    if red_tweet is not None:
        rt_event = (red_tweet >= 1).astype(float)
        rt_event[red_tweet.isna()] = np.nan
        events.append(('red_tweet', rt_event, True))

    # 5b. Green tweet
    if green_tweet is not None:
        grt_event = (green_tweet >= 1).astype(float)
        grt_event[green_tweet.isna()] = np.nan
        events.append(('green_tweet', grt_event, True))

    # 6. Caps tweet
    caps = _col('tweet_caps_any')
    if caps is not None:
        caps_event = (caps >= 1).astype(float)
        caps_event[caps.isna()] = np.nan
        events.append(('caps_tweet', caps_event, False))

    # 7. Sport upset
    sport = _col('sport_upset_count')
    if sport is not None:
        sport_event = (sport > 0).astype(float)
        sport_event[sport.isna()] = np.nan
        events.append(('sport_upset', sport_event, True))

    # 8. News caution
    news_caut = _col('headline_caution_any')
    if news_caut is not None:
        nc_event = (news_caut > 0).astype(float)
        nc_event[news_caut.isna()] = np.nan
        events.append(('news_caution', nc_event, False))

    # 9. Gematria match: any gem_match/gem_caution/gem_pump column fires
    gem_cols = [c for c in _df_cols
                if 'gem_match' in c or 'gem_caution' in c or 'gem_pump' in c]
    if gem_cols:
        gem_any = pd.Series(0.0, index=_cpu_idx)
        for gc in gem_cols:
            gv = _col(gc)
            if gv is not None:
                gem_any = gem_any.where(gv.isna() | (gv == 0), 1.0)
        events.append(('gem_match', gem_any, False))

    # 10. High fear
    fg_fear = _col('fg_extreme_fear')
    if fg_fear is not None:
        fear_event = (fg_fear >= 1).astype(float)
        fear_event[fg_fear.isna()] = np.nan
        events.append(('high_fear', fear_event, True))

    # 11. High greed
    fg_greed = _col('fg_extreme_greed')
    if fg_greed is not None:
        greed_event = (fg_greed >= 1).astype(float)
        greed_event[fg_greed.isna()] = np.nan
        events.append(('high_greed', greed_event, True))

    # 12. Kp storm
    kp_storm = _col('sw_kp_is_storm')
    if kp_storm is not None:
        storm_event = (kp_storm >= 1).astype(float)
        storm_event[kp_storm.isna()] = np.nan
        events.append(('kp_storm', storm_event, False))

    # ------------------------------------------------------------------
    # Compute bars_since + decay for each event
    # ------------------------------------------------------------------
    for name, event_series, both in events:
        try:
            bse = _bars_since_event(event_series)
            out[f'bars_since_{name}'] = bse

            # Fast decay (lambda=0.1)
            out[f'{name}_decay_fast'] = np.exp(-LAMBDA_FAST * bse)

            # Slow decay (lambda=0.05) for important events
            if both:
                out[f'{name}_decay_slow'] = np.exp(-LAMBDA_SLOW * bse)
        except Exception:
            pass

    return out


def _tf_label(bucket_seconds):
    """Convert bucket seconds to human-readable TF label."""
    labels = {300: '5m', 900: '15m', 3600: '1h', 14400: '4h', 86400: '1d', 604800: '1w'}
    return labels.get(bucket_seconds, f'{bucket_seconds}s')


def _tf_from_bucket(bucket_seconds):
    """Convert bucket seconds to TF name."""
    return _tf_label(bucket_seconds)


def _tf_label_from_name(tf_name):
    """Get label for target columns."""
    return tf_name


if __name__ == '__main__':
    print("feature_library.py — Shared Feature Computation Library")
    print(f"  TF configs: {list(TF_CONFIG.keys())}")
    print(f"  Functions: compute_ta_features, compute_esoteric_features, "
          f"compute_higher_tf_features, compute_regime_features, "
          f"compute_space_weather_features, compute_lunar_electromagnetic_features, "
          f"compute_decay_features, compute_llm_features, build_all_features")
    print(f"  LLM features available: {_HAS_LLM}")
