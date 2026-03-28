#!/usr/bin/env python
"""
v2_cross_generator.py — V2 Everything × Everything Cross Feature Generator
============================================================================
Generates ALL V2 cross features using sparse batch operations.
MEMORY-OPTIMIZED: Streams results to sparse chunks instead of accumulating
dense arrays in a dict. Peak RAM stays under ~4GB regardless of feature count.

Cross types generated:
  dx_  = DOY window (±2 days) × context
  ax_  = astro × TA
  ax2_ = multi-astro × TA
  ta2_ = multi-TA × DOY + astro
  ex2_ = esoteric × TA
  sw_  = space weather × all
  hod_ = hour-of-day × all
  mx_  = macro × all
  vx_  = volatility regime × all
  asp_ = planetary aspects × all
  pn_  = price numerology × all
  3x regime-aware DOY crosses

Usage:
  python v2_cross_generator.py --tf 1d
  python v2_cross_generator.py --tf 1d --gpu 0
  python v2_cross_generator.py --tf 1h --save-sparse
"""

import os, time, argparse, warnings, gc, glob
import ctypes
try:
    _libc = ctypes.cdll.LoadLibrary("libc.so.6")
    def _malloc_trim():
        _libc.malloc_trim(0)
except Exception:
    def _malloc_trim():
        pass
import numpy as np
import pandas as pd
from scipy import sparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from numba import njit, prange

warnings.filterwarnings('ignore')

V2_DIR = os.path.dirname(os.path.abspath(__file__))

# ── CUDA version detection (BEFORE any GPU library imports) ──
# CuPy-cuda12x is compiled for CUDA 12.x.
# On CUDA 13.0+ (driver 580+), CuPy runtime operations SEGFAULT.
# Detect driver version FIRST and skip CuPy import on CUDA 13+.
_CUDA_MAJOR = 12
try:
    import subprocess as _sp
    _nv = _sp.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                   capture_output=True, text=True, timeout=5)
    _drv = int(_nv.stdout.strip().split('.')[0])
    _CUDA_MAJOR = 13 if _drv >= 580 else 12
except Exception:
    pass

# ── GPU setup (guarded by CUDA version) ──
if _CUDA_MAJOR >= 13:
    if os.environ.get('ALLOW_CPU', '0') != '1':
        raise RuntimeError("GPU REQUIRED: CUDA 13+ driver (580+) — CuPy would SEGFAULT. Set ALLOW_CPU=1 for CPU mode.")
    cp = None
    cusp = None
    GPU = False
    print(f"[v2_cross_generator] ALLOW_CPU=1 — CUDA {_CUDA_MAJOR}.x driver (580+). CuPy would SEGFAULT. Using CPU mode.")
else:
    try:
        import cupy as cp
        import cupyx.scipy.sparse as cusp
        GPU = True
    except ImportError:
        if os.environ.get('ALLOW_CPU', '0') != '1':
            raise RuntimeError("GPU REQUIRED: CuPy not installed. Install CuPy or set ALLOW_CPU=1 for CPU mode.")
        cp = None
        cusp = None
        GPU = False
        print("[v2_cross_generator] ALLOW_CPU=1 — CuPy not available, using CPU mode.")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _get_available_ram_gb():
    """Get available RAM in GB (cgroup v1+v2 aware). Delegates to shared hardware_detect."""
    try:
        from hardware_detect import get_available_ram_gb
        return get_available_ram_gb()
    except ImportError:
        pass
    # Inline fallback if hardware_detect not available
    try:
        import psutil
        return psutil.virtual_memory().available / (1024**3)
    except (ImportError, Exception):
        pass
    return 64.0  # conservative fallback


# ── Adaptive batch sizing based on available RAM ──
def _get_right_chunk():
    """Auto-detect RAM and set chunk size accordingly."""
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
    except (ImportError, Exception):
        try:
            with open('/proc/meminfo') as f:
                for line in f:
                    if 'MemTotal' in line:
                        ram_gb = int(line.split()[1]) / (1024**2)
                        break
                else:
                    ram_gb = 64.0
        except (FileNotFoundError, Exception):
            ram_gb = 64.0  # conservative default

    if ram_gb >= 512:
        RIGHT_CHUNK = 2000   # cloud: 512GB+ RAM, process 2000 contexts at a time
    elif ram_gb >= 256:
        RIGHT_CHUNK = 1000   # cloud: 256GB RAM
    elif ram_gb >= 128:
        RIGHT_CHUNK = 500    # mid-range
    elif ram_gb >= 64:
        RIGHT_CHUNK = 200    # local 64GB
    else:
        RIGHT_CHUNK = 100    # low RAM

    # Scale down for high-row-count TFs to prevent memory accumulation
    # n_rows is not available here, so we check via environment variable or default
    _env_nrows = os.environ.get('V2_NROWS')
    if _env_nrows:
        _nrows = int(_env_nrows)
        RIGHT_CHUNK = max(200, int(RIGHT_CHUNK * min(1.0, 50000 / _nrows)))

    return RIGHT_CHUNK


def _get_cross_batch_size(n_rows):
    """Auto-size batch for vectorized cross computation. Targets 25% of available RAM."""
    try:
        import psutil
        avail = psutil.virtual_memory().available
    except (ImportError, Exception):
        try:
            with open('/proc/meminfo') as f:
                for line in f:
                    if 'MemAvailable' in line:
                        avail = int(line.split()[1]) * 1024
                        break
                else:
                    avail = 16 * 1024**3
        except Exception:
            avail = 16 * 1024**3
    target = int(avail * 0.25)
    # Each pair: 2 column reads (N×4B) + 1 result (N×4B) = 12N bytes
    bytes_per_pair = n_rows * 12
    batch = max(500, min(50000, target // max(1, bytes_per_pair)))
    _env_batch = os.environ.get("V2_BATCH_MAX")
    if _env_batch:
        batch = min(batch, int(_env_batch))
    return batch


# Env var override for orchestrator OOM retry
_env_chunk = os.environ.get('V2_RIGHT_CHUNK')
RIGHT_CHUNK = int(_env_chunk) if _env_chunk else _get_right_chunk()
log(f"Adaptive RIGHT_CHUNK = {RIGHT_CHUNK} (detected RAM{', env override' if _env_chunk else ''})")

# Minimum co-occurrence threshold for cross features.
# Crosses firing fewer than this many times cannot reliably appear in both
# CPCV train AND validation splits (98.3% coverage at n=8 with 5-fold CPCV).
# This is a math constraint on the validation pipeline, not a signal filter.
_env_co_occur = os.environ.get('V2_MIN_CO_OCCURRENCE')
MIN_CO_OCCURRENCE = int(_env_co_occur) if _env_co_occur else 3  # Lowered from 8: matches min_data_in_leaf=3, preserves rare esoteric crosses


# ── GPU VRAM-adaptive batch sizing ──
def _get_gpu_vram_gb(gpu_id=0):
    """Detect GPU VRAM in GB. Returns 0 on CUDA 13+ (GPU disabled)."""
    if _CUDA_MAJOR >= 13 or cp is None:
        return 0.0
    try:
        mem = cp.cuda.Device(gpu_id).mem_info  # returns (free, total)
        return mem[1] / (1024**3)  # total VRAM in GB
    except Exception:
        return 12.0  # default: assume 3090


def _get_optimal_batch(n_bars, n_right, gpu_vram_gb=None, gpu_id=0):
    """Auto-size GPU batch based on available VRAM. Fully dynamic — no hard cap.
    Scales from BATCH=5 on 12GB 3090 to BATCH=200+ on 96GB H100."""
    if gpu_vram_gb is None:
        gpu_vram_gb = _get_gpu_vram_gb(gpu_id=gpu_id)
    # Leave 30% headroom for kernel overhead + other GPU operations
    available_bytes = gpu_vram_gb * 0.7 * (1024**3)
    # Shape is (n_bars, BATCH, n_right) in float32 = 4 bytes
    bytes_per_batch_elem = n_bars * n_right * 4
    if bytes_per_batch_elem <= 0:
        return 10  # safe minimum
    max_batch = int(available_bytes / bytes_per_batch_elem)
    return max(1, max_batch)


# ============================================================
# BINARIZATION
# ============================================================

@njit(parallel=True, cache=True)
def _binarize_batch_4tier(values, n_cols):
    """Compute 4-tier binarization (XH/H/L/XL) for all columns in parallel.

    Parameters
    ----------
    values : float32 array (n_rows, n_cols) — contiguous, NaN-preserving
    n_cols : int

    Returns
    -------
    masks : float32 array (n_cols * 4, n_rows) — row-major per mask
        Layout: [col0_XH, col0_H, col0_L, col0_XL, col1_XH, ...]
    thresholds : float32 array (n_cols, 4) — [q95, q75, q25, q5]
    mask_sums : float32 array (n_cols * 4) — nansum of each mask (for filtering)
    """
    n_rows = values.shape[0]
    masks = np.zeros((n_cols * 4, n_rows), dtype=np.float32)
    thresholds = np.zeros((n_cols, 4), dtype=np.float32)
    mask_sums = np.zeros(n_cols * 4, dtype=np.float32)

    for col_idx in prange(n_cols):
        col = values[:, col_idx]

        # Gather non-NaN values
        valid_count = 0
        for i in range(n_rows):
            if not np.isnan(col[i]):
                valid_count += 1
        if valid_count < 10:
            # Near-empty — masks stay zero, thresholds stay zero
            continue

        valid = np.empty(valid_count, dtype=np.float32)
        vi = 0
        for i in range(n_rows):
            if not np.isnan(col[i]):
                valid[vi] = col[i]
                vi += 1

        # Count non-zero valid entries
        nz_count = 0
        for i in range(valid_count):
            if valid[i] != 0.0:
                nz_count += 1

        # Use non-zero subset if >100 non-zero values, else use all valid
        if nz_count > 100:
            nz = np.empty(nz_count, dtype=np.float32)
            ni = 0
            for i in range(valid_count):
                if valid[i] != 0.0:
                    nz[ni] = valid[i]
                    ni += 1
        else:
            nz = valid

        # Sort for percentile computation
        sorted_vals = np.sort(nz)
        n = len(sorted_vals)
        # Compute percentiles via index (matches numpy nearest-rank behavior)
        q95 = sorted_vals[min(int(n * 0.95), n - 1)]
        q75 = sorted_vals[min(int(n * 0.75), n - 1)]
        q25 = sorted_vals[min(int(n * 0.25), n - 1)]
        q5 = sorted_vals[min(int(n * 0.05), n - 1)]
        thresholds[col_idx, 0] = q95
        thresholds[col_idx, 1] = q75
        thresholds[col_idx, 2] = q25
        thresholds[col_idx, 3] = q5

        base = col_idx * 4
        s_xh = np.float32(0.0)
        s_h = np.float32(0.0)
        s_l = np.float32(0.0)
        s_xl = np.float32(0.0)
        for i in range(n_rows):
            v = col[i]
            # NaN comparisons yield False — masks stay 0.0 for NaN rows
            if v > q95:
                masks[base, i] = 1.0
                s_xh += 1.0
            if v > q75:
                masks[base + 1, i] = 1.0
                s_h += 1.0
            if v < q25:
                masks[base + 2, i] = 1.0
                s_l += 1.0
            if v < q5:
                masks[base + 3, i] = 1.0
                s_xl += 1.0
        mask_sums[base] = s_xh
        mask_sums[base + 1] = s_h
        mask_sums[base + 2] = s_l
        mask_sums[base + 3] = s_xl

    return masks, thresholds, mask_sums


@njit(parallel=True, cache=True)
def _binarize_batch_2tier(values, n_cols):
    """Compute 2-tier binarization (H/L) for all columns in parallel.

    Same structure as 4-tier but with q80/q20 thresholds.
    """
    n_rows = values.shape[0]
    masks = np.zeros((n_cols * 2, n_rows), dtype=np.float32)
    thresholds = np.zeros((n_cols, 2), dtype=np.float32)
    mask_sums = np.zeros(n_cols * 2, dtype=np.float32)

    for col_idx in prange(n_cols):
        col = values[:, col_idx]

        valid_count = 0
        for i in range(n_rows):
            if not np.isnan(col[i]):
                valid_count += 1
        if valid_count < 10:
            continue

        valid = np.empty(valid_count, dtype=np.float32)
        vi = 0
        for i in range(n_rows):
            if not np.isnan(col[i]):
                valid[vi] = col[i]
                vi += 1

        nz_count = 0
        for i in range(valid_count):
            if valid[i] != 0.0:
                nz_count += 1

        if nz_count > 100:
            nz = np.empty(nz_count, dtype=np.float32)
            ni = 0
            for i in range(valid_count):
                if valid[i] != 0.0:
                    nz[ni] = valid[i]
                    ni += 1
        else:
            nz = valid

        sorted_vals = np.sort(nz)
        n = len(sorted_vals)
        q80 = sorted_vals[min(int(n * 0.80), n - 1)]
        q20 = sorted_vals[min(int(n * 0.20), n - 1)]
        thresholds[col_idx, 0] = q80
        thresholds[col_idx, 1] = q20

        base = col_idx * 2
        s_h = np.float32(0.0)
        s_l = np.float32(0.0)
        for i in range(n_rows):
            v = col[i]
            if v > q80:
                masks[base, i] = 1.0
                s_h += 1.0
            if v < q20:
                masks[base + 1, i] = 1.0
                s_l += 1.0
        mask_sums[base] = s_h
        mask_sums[base + 1] = s_l

    return masks, thresholds, mask_sums


def binarize_contexts(df, four_tier=True):
    """
    Binarize all suitable columns into binary context arrays.
    Returns: (ctx_names, ctx_arrays) where ctx_arrays is list of float32 arrays.

    With four_tier=True:
      EXTREME_HIGH (>95th), HIGH (>75th), LOW (<25th), EXTREME_LOW (<5th)

    Uses Numba @njit(parallel=True) to compute all percentiles + masks in one
    parallel kernel instead of 3000+ sequential np.percentile() calls.
    """
    skip_pre = ('tx_', 'px_', 'ex_', 'dx_', 'cross_', 'next_', 'target_',
                'doy_', 'ax_', 'ax2_', 'ta2_', 'ex2_', 'sw_', 'hod_',
                'mx_', 'vx_', 'pn_', 'seq_', 'roc_', 'harm_')
    skip_ex = {'timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume',
               'trades', 'taker_buy_volume', 'taker_buy_quote', 'triple_barrier_label',
               'open_time', 'open_time_ms'}

    N = len(df)
    ctx_names = []
    ctx_arrays = []

    # ── Pass 1: classify columns (Python — handles strings, dedup, filtering) ──
    binary_cols = []     # (col_name, vals_array) for <=3 unique values
    multi_cols = []      # (col_name, col_index_in_batch) for >3 unique values
    multi_vals_list = [] # corresponding float32 arrays to stack

    seen_cols = set()
    for col in df.columns:
        if col in seen_cols:
            continue
        seen_cols.add(col)
        if col.startswith(skip_pre) or col in skip_ex:
            continue

        raw = df[col]
        if isinstance(raw, pd.DataFrame):
            raw = raw.iloc[:, 0]
        vals = pd.to_numeric(raw, errors='coerce').values.astype(np.float32)
        uniq = np.unique(vals[~np.isnan(vals)])
        if len(uniq) <= 1:
            continue

        if len(uniq) <= 3:
            # Binary/ternary — handle directly (no percentile needed)
            b = (vals > 0).astype(np.float32)
            if 5 < np.nansum(b) < N * 0.98:
                binary_cols.append((col, b))
        else:
            multi_cols.append((col, len(multi_vals_list)))
            multi_vals_list.append(vals)

    # ── Add binary columns directly ──
    for col_name, b in binary_cols:
        ctx_names.append(col_name)
        ctx_arrays.append(b)

    # ── Pass 2: Numba parallel kernel for all multi-valued columns ──
    if multi_vals_list:
        # Stack into contiguous (n_rows, n_multi_cols) array
        batch = np.column_stack(multi_vals_list)  # (N, n_multi_cols) float32
        n_multi = len(multi_vals_list)

        if four_tier:
            masks, _thresholds, mask_sums = _binarize_batch_4tier(batch, n_multi)
            tags = ('XH', 'H', 'L', 'XL')
            n_tiers = 4
        else:
            masks, _thresholds, mask_sums = _binarize_batch_2tier(batch, n_multi)
            tags = ('H', 'L')
            n_tiers = 2

        # ── Pass 3: construct output from kernel results (Python — string ops) ──
        for col_name, batch_idx in multi_cols:
            base = batch_idx * n_tiers
            for t, tag in enumerate(tags):
                if mask_sums[base + t] > 5:
                    ctx_names.append(f'{col_name}_{tag}')
                    ctx_arrays.append(masks[base + t].copy())

        del batch, masks, mask_sums
        gc.collect()

    return ctx_names, ctx_arrays


# ============================================================
# SIGNAL GROUP EXTRACTION
# ============================================================

def extract_signal_groups(df, ctx_names, ctx_arrays):
    """
    Extract named groups of binary signals for targeted crossing.
    Returns dict of {group_name: [(name, array), ...]}.
    """
    groups = {
        'astro': [], 'ta': [], 'esoteric': [], 'space_weather': [],
        'macro': [], 'regime': [], 'session': [], 'aspect': [],
        'price_num': [], 'moon': [], 'volatility': [],
    }

    astro_keys = ('moon_phase', 'moon_illumination', 'moon_distance',
                  'new_moon', 'full_moon',
                  'retro', 'eclipse', 'nakshatra', 'void_of_course',
                  'planetary_hour', 'planetary_day', 'bazi', 'tzolkin',
                  'equinox', 'solstice', 'tithi', 'karana')
    ta_keys = ('rsi', 'macd', 'stoch', 'volume_', 'ema', 'sma',
               'obv', 'ichimoku', 'supertrend',
               'vwap', 'adx', 'cci', 'mfi', 'entropy', 'hurst')
    esoteric_keys = (
        # gematria / digital root
        'gem_', 'dr_', 'gematria', 'digital_root',
        # tweet / sentiment / caution
        'tweet', 'caution', 'pump', 'misdirection', 'sentiment',
        # on-chain
        'onchain', 'oi_', 'funding', 'liq_', 'whale', 'mempool',
        'hash_rate', 'block_height', 'coinbase_premium', 'cvd',
        # hebrew calendar / shmita
        'hebrew', 'shmita',
        # esoteric time cycles
        'fibonacci_time', 'gann_time', 'tesla',
        # numerology
        'master_', 'angel', 'contains_', 'palindrome', 'friday_13',
        # text features
        'caps_', 'excl_', 'word_count',
        # fear & greed
        'fear_greed', 'f_g_',
    )
    volatility_keys = ('atr', 'bb_width', 'realized_vol', 'vol_', 'keltner',
                        'donchian', 'true_range', 'natr', 'bb_')
    space_keys = ('kp_', 'solar', 'sunspot')
    macro_keys = ('vix', 'dxy', 'yield', 'spx', 'dominance', 'tvl', 'cot_')
    regime_keys = ('regime', 'bull', 'bear', 'sideways')
    session_keys = ('session_', 'kill_zone', 'hod_')
    aspect_keys = ('asp_',)
    price_num_keys = ('price_dr', 'price_angel', 'price_near', 'price_master')
    moon_keys = ('moon_in_', 'moon_fire', 'moon_earth', 'moon_air', 'moon_water',
                 'moon_sign', 'moon_cardinal', 'moon_fixed', 'moon_mutable')

    for name, arr in zip(ctx_names, ctx_arrays):
        nl = name.lower()
        if any(k in nl for k in astro_keys):
            groups['astro'].append((name, arr))
        elif any(k in nl for k in esoteric_keys):
            groups['esoteric'].append((name, arr))
        elif any(k in nl for k in volatility_keys):
            groups['volatility'].append((name, arr))
        elif any(k in nl for k in ta_keys):
            groups['ta'].append((name, arr))
        elif any(k in nl for k in space_keys):
            groups['space_weather'].append((name, arr))
        elif any(k in nl for k in macro_keys):
            groups['macro'].append((name, arr))
        elif any(k in nl for k in regime_keys):
            groups['regime'].append((name, arr))
        elif any(k in nl for k in session_keys):
            groups['session'].append((name, arr))
        elif any(k in nl for k in aspect_keys):
            groups['aspect'].append((name, arr))
        elif any(k in nl for k in price_num_keys):
            groups['price_num'].append((name, arr))
        elif any(k in nl for k in moon_keys):
            groups['moon'].append((name, arr))
        else:
            groups['ta'].append((name, arr))  # default to TA

    return groups


# ============================================================
# HELPERS: dict -> COO triplet conversion
# ============================================================

    # _dict_to_coo_triplets removed — GPU/CPU chunks now return COO directly


# ============================================================
# BATCH GPU CROSS MULTIPLICATION (CHUNKED RIGHT-SIDE)
# ============================================================

def gpu_batch_cross(left_names, left_arrays, right_names, right_arrays, prefix,
                    gpu_id=0, min_nonzero=None, max_features=None, col_offset=0):
    """
    Cross every left signal with every right signal using GPU batch multiply.
    Returns (feature_names_list, rows_list, cols_list, data_list, n_new_cols)
    as COO triplet arrays for efficient assembly.

    Right-side is processed in chunks of RIGHT_CHUNK to cap memory.
    col_offset: starting column index for global COO assembly.

    NOTE: The RIGHT_CHUNK loop CANNOT be parallelized across GPUs.
    Each chunk allocates a tensor of shape (n_bars, BATCH, n_right) that
    consumes most of GPU VRAM. Running multiple chunks simultaneously would
    OOM. The chunking exists precisely because the full cross product exceeds
    VRAM. Sequential chunking is the correct approach here — the bottleneck
    is VRAM capacity, not GPU compute utilization.
    """
    if min_nonzero is None:
        min_nonzero = MIN_CO_OCCURRENCE

    if not left_arrays or not right_arrays:
        return [], [], [], [], 0

    N = len(left_arrays[0])
    left_mat = np.column_stack(left_arrays)  # (N, n_left) — usually small

    all_names = []
    all_rows = []
    all_cols = []
    all_data = []
    _local_csr_chunks = []  # CSR chunks flushed per RIGHT_CHUNK
    total_feats = 0
    current_offset = col_offset

    # Process right-side in chunks of RIGHT_CHUNK
    for rc_start in range(0, len(right_names), RIGHT_CHUNK):
        rc_end = min(rc_start + RIGHT_CHUNK, len(right_names))
        r_names_chunk = right_names[rc_start:rc_end]
        r_arrays_chunk = right_arrays[rc_start:rc_end]
        right_mat_chunk = np.column_stack(r_arrays_chunk)  # (N, <=RIGHT_CHUNK)

        # GPU now uses sparse matmul pre-filter — no 3D tensor, works for ANY row count.
        # Skip if: CUDA 13+ detected, env var override, or CuPy unavailable.
        _skip_gpu = _CUDA_MAJOR >= 13 or os.environ.get('V2_SKIP_GPU') == '1'
        if GPU and not _skip_gpu:
            c_names, c_rows, c_cols, c_data, c_ncols = _gpu_cross_chunk(
                left_names, left_mat, r_names_chunk, right_mat_chunk,
                prefix, gpu_id, min_nonzero, max_features, total_feats,
                col_offset=current_offset
            )
        else:
            c_names, c_rows, c_cols, c_data, c_ncols = _cpu_cross_chunk(
                left_names, left_mat, r_names_chunk, right_mat_chunk,
                prefix, min_nonzero, max_features, total_feats,
                col_offset=current_offset
            )

        if c_names:
            # FLUSH per RIGHT_CHUNK: convert COO to CSR immediately
            # Memory-safe: build CSR in sub-batches to avoid giant concatenation OOM.
            # Each entry in c_rows[i] is one column's nonzero row indices.
            all_names.extend(c_names)
            if c_rows:
                # Estimate total NNZ to pick safe sub-batch size
                # Target: sub-batch concat peak under 500MB (~125M int32 entries)
                _total_nnz = sum(len(r) for r in c_rows)
                _avg_nnz = _total_nnz / max(1, len(c_rows))
                # Max entries per sub-batch concat = 125M (500MB / 4 bytes)
                _max_entries = 125_000_000
                SUB_FLUSH = max(100, min(5000, int(_max_entries / max(1, _avg_nnz))))

                n_cols_total = len(c_rows)
                for sf_start in range(0, n_cols_total, SUB_FLUSH):
                    sf_end = min(sf_start + SUB_FLUSH, n_cols_total)
                    sf_n_cols = sf_end - sf_start
                    # Build local COO with column indices 0..sf_n_cols-1
                    sf_r_parts = []
                    sf_c_parts = []
                    sf_d_parts = []
                    for local_col, global_col in enumerate(range(sf_start, sf_end)):
                        r_arr = c_rows[global_col]
                        if len(r_arr) > 0:
                            sf_r_parts.append(r_arr)
                            sf_c_parts.append(np.full(len(r_arr), local_col, dtype=np.int32))
                            sf_d_parts.append(c_data[global_col])
                    if sf_r_parts:
                        _sf_r = np.concatenate(sf_r_parts)
                        _sf_c = np.concatenate(sf_c_parts)
                        _sf_d = np.concatenate(sf_d_parts)
                        if cp is not None and cusp is not None:
                            try:
                                _sf_d_gpu = cp.asarray(_sf_d)
                                _sf_r_gpu = cp.asarray(_sf_r)
                                _sf_c_gpu = cp.asarray(_sf_c)
                                _sf_csr = cusp.coo_matrix((_sf_d_gpu, (_sf_r_gpu, _sf_c_gpu)), shape=(N, sf_n_cols)).tocsr()
                                _sf_csr = _sf_csr.get()  # Transfer back to scipy CSR on CPU
                                del _sf_d_gpu, _sf_r_gpu, _sf_c_gpu
                                cp.get_default_memory_pool().free_all_blocks()
                            except Exception:
                                _sf_csr = sparse.coo_matrix((_sf_d, (_sf_r, _sf_c)), shape=(N, sf_n_cols)).tocsr()
                        else:
                            _sf_csr = sparse.coo_matrix((_sf_d, (_sf_r, _sf_c)), shape=(N, sf_n_cols)).tocsr()
                        _local_csr_chunks.append(_sf_csr)
                        del _sf_r, _sf_c, _sf_d, _sf_csr
                    del sf_r_parts, sf_c_parts, sf_d_parts
                    gc.collect()
            current_offset += c_ncols
            total_feats += len(c_names)
            del c_rows, c_cols, c_data, c_names

        del right_mat_chunk, r_arrays_chunk
        gc.collect()
        _malloc_trim()

        if max_features and total_feats >= max_features:
            break

    del left_mat
    n_total_cols = current_offset - col_offset
    # Return CSR chunks if we flushed (memory-safe path)
    if _local_csr_chunks:
        return all_names, _local_csr_chunks, None, None, n_total_cols
    return all_names, all_rows, all_cols, all_data, n_total_cols


def _gpu_cross_chunk(left_names, left_mat, right_names, right_mat, prefix,
                     gpu_id, min_nonzero, max_features, feats_so_far,
                     col_offset=0):
    """
    GPU cross multiply with sparse matmul pre-filter.
    Step 1: Sparse matmul on CPU for co-occurrence (instant, tiny output).
    Step 2: GPU element-wise multiply ONLY for valid pairs.
    No 3D tensor — works for ANY row count including 217K+ (15m).
    """
    names = []
    rows_list = []
    cols_list = []
    data_list = []
    col_idx = 0

    # ── Step 1: Sparse matmul pre-filter (cuSPARSE on GPU, scipy fallback) ──
    if cp is not None and cusp is not None:
        try:
            left_gpu_sp = cusp.csc_matrix(cp.asarray(left_mat.astype(np.float32)))
            right_gpu_sp = cusp.csc_matrix(cp.asarray(right_mat.astype(np.float32)))
            co_occur = cp.asnumpy((left_gpu_sp.T @ right_gpu_sp).toarray())
            del left_gpu_sp, right_gpu_sp
            cp.get_default_memory_pool().free_all_blocks()
        except Exception as e:
            log(f"  cuSPARSE SpGEMM failed ({e}), falling back to scipy")
            left_sp = sparse.csc_matrix(left_mat)
            if left_sp.indices.dtype != np.int64:
                left_sp.indices = left_sp.indices.astype(np.int64)
                left_sp.indptr = left_sp.indptr.astype(np.int64)
            right_sp = sparse.csc_matrix(right_mat)
            if right_sp.indices.dtype != np.int64:
                right_sp.indices = right_sp.indices.astype(np.int64)
                right_sp.indptr = right_sp.indptr.astype(np.int64)
            co_occur = (left_sp.T @ right_sp).toarray()
    else:
        left_sp = sparse.csc_matrix(left_mat)
        if left_sp.indices.dtype != np.int64:
            left_sp.indices = left_sp.indices.astype(np.int64)
            left_sp.indptr = left_sp.indptr.astype(np.int64)
        right_sp = sparse.csc_matrix(right_mat)
        if right_sp.indices.dtype != np.int64:
            right_sp.indices = right_sp.indices.astype(np.int64)
            right_sp.indptr = right_sp.indptr.astype(np.int64)
        co_occur = (left_sp.T @ right_sp).toarray()  # (n_left, n_right) — small
    valid_pairs = np.argwhere(co_occur >= min_nonzero)
    n_valid = len(valid_pairs)

    if n_valid == 0:
        return names, rows_list, cols_list, data_list, col_idx

    gpu_vram_gb = _get_gpu_vram_gb(gpu_id=gpu_id)
    log(f"  GPU VRAM: {gpu_vram_gb:.1f} GB | {n_valid} valid pairs (pre-filtered from {left_mat.shape[1]}×{right_mat.shape[1]})")

    # Pre-build all feature names (preserves exact order)
    all_names = [f'{prefix}_{left_names[int(p[0])][:40]}_{right_names[int(p[1])][:40]}'
                 for p in valid_pairs]

    # ── Step 2: Vectorized GPU batch multiply ──
    _dev = 0 if os.environ.get('CUDA_VISIBLE_DEVICES') else gpu_id
    cp.cuda.Device(_dev).use()
    left_gpu = cp.asarray(np.ascontiguousarray(left_mat))
    right_gpu = cp.asarray(np.ascontiguousarray(right_mat))

    N = left_mat.shape[0]
    avail_vram = int(gpu_vram_gb * 0.5 * 1024**3)
    bytes_per_pair = N * 12
    BATCH = max(100, min(50000, avail_vram // max(1, bytes_per_pair)))

    for b_start in range(0, n_valid, BATCH):
        b_end = min(b_start + BATCH, n_valid)
        chunk = valid_pairs[b_start:b_end]

        # Vectorized batch multiply on GPU
        left_cols = left_gpu[:, chunk[:, 0]]
        right_cols = right_gpu[:, chunk[:, 1]]
        crosses_gpu = left_cols * right_cols
        del left_cols, right_cols

        # GPU nonzero extraction (Phase 1B — avoids full CPU scan)
        nz_rows_gpu, nz_cols_gpu = cp.nonzero(crosses_gpu)
        nz_rows_all = cp.asnumpy(nz_rows_gpu)
        nz_cols_all = cp.asnumpy(nz_cols_gpu)
        del nz_rows_gpu, nz_cols_gpu

        # Transfer crosses to CPU for data value extraction
        crosses = cp.asnumpy(crosses_gpu)
        del crosses_gpu
        cp.get_default_memory_pool().free_all_blocks()

        if len(nz_rows_all) > 0:
            unique_cols, col_starts = np.unique(nz_cols_all, return_index=True)
            col_ends = np.append(col_starts[1:], len(nz_cols_all))

            for k in range(len(unique_cols)):
                c = int(unique_cols[k])
                s, e = int(col_starts[k]), int(col_ends[k])
                nz = nz_rows_all[s:e]
                names.append(all_names[b_start + c])
                rows_list.append(nz)
                cols_list.append(np.full(len(nz), col_offset + col_idx, dtype=np.int64))
                data_list.append(crosses[nz, c].astype(np.float32))
                col_idx += 1

        if max_features and (feats_so_far + col_idx) >= max_features:
            break

    del left_gpu, right_gpu
    cp.get_default_memory_pool().free_all_blocks()
    return names, rows_list, cols_list, data_list, col_idx


@njit(parallel=True, cache=True)
def _parallel_cross_multiply(left, right, out):
    n_rows, n_pairs = left.shape
    for j in prange(n_pairs):
        for i in range(n_rows):
            out[i, j] = left[i, j] * right[i, j]


def _process_cross_block(left_mat, right_mat, valid_pairs, all_names,
                         b_start, b_end, col_offset_base):
    """
    Process a block of cross pairs. Thread-safe — numpy element-wise ops release GIL.
    Returns (names, rows_list, cols_list, data_list) for this block.
    """
    chunk = valid_pairs[b_start:b_end]
    left_cols = np.ascontiguousarray(left_mat[:, chunk[:, 0]])     # (N, chunk_size)
    right_cols = np.ascontiguousarray(right_mat[:, chunk[:, 1]])   # (N, chunk_size)
    crosses = np.empty_like(left_cols)
    _parallel_cross_multiply(left_cols, right_cols, crosses)       # Numba parallel prange

    nz_rows_all, nz_cols_all = np.nonzero(crosses)

    names = []
    rows_list = []
    cols_list = []
    data_list = []

    if len(nz_rows_all) > 0:
        unique_cols, col_starts = np.unique(nz_cols_all, return_index=True)
        col_ends = np.append(col_starts[1:], len(nz_cols_all))

        for k in range(len(unique_cols)):
            c = int(unique_cols[k])
            s, e = int(col_starts[k]), int(col_ends[k])
            nz = nz_rows_all[s:e]
            names.append(all_names[b_start + c])
            rows_list.append(nz)
            # col indices assigned later during merge to ensure correct global ordering
            data_list.append(crosses[nz, c].astype(np.float32))

    return names, rows_list, data_list


def _cpu_cross_chunk(left_names, left_mat, right_names, right_mat, prefix,
                     min_nonzero, max_features, feats_so_far, col_offset=0):
    """
    CPU cross multiply with sparse matmul pre-filter + MULTI-THREADED execution.
    Step 1: Compute ALL co-occurrence counts via sparse matmul (instant).
    Step 2: Only compute actual crosses for valid pairs (skip ~92% of work).
    Step 3: Parallel thread execution — numpy element-wise ops release GIL.
    Returns COO triplets directly, never stores dense columns.
    """
    names = []
    rows_list = []
    cols_list = []
    data_list = []
    col_idx = 0

    # ── Sparse matmul pre-filter: compute ALL co-occurrence counts at once ──
    # GPU path (cuSPARSE SpGEMM) with CPU fallback
    if cp is not None and cusp is not None:
        try:
            left_gpu_sp = cusp.csc_matrix(cp.asarray(left_mat.astype(np.float32)))
            right_gpu_sp = cusp.csc_matrix(cp.asarray(right_mat.astype(np.float32)))
            co_occur = cp.asnumpy((left_gpu_sp.T @ right_gpu_sp).toarray())
            del left_gpu_sp, right_gpu_sp
            cp.get_default_memory_pool().free_all_blocks()
        except Exception as e:
            left_sp = sparse.csc_matrix(left_mat)
            if left_sp.indices.dtype != np.int64:
                left_sp.indices = left_sp.indices.astype(np.int64)
                left_sp.indptr = left_sp.indptr.astype(np.int64)
            right_sp = sparse.csc_matrix(right_mat)
            if right_sp.indices.dtype != np.int64:
                right_sp.indices = right_sp.indices.astype(np.int64)
                right_sp.indptr = right_sp.indptr.astype(np.int64)
            co_occur = (left_sp.T @ right_sp).toarray()
    else:
        left_sp = sparse.csc_matrix(left_mat)
        if left_sp.indices.dtype != np.int64:
            left_sp.indices = left_sp.indices.astype(np.int64)
            left_sp.indptr = left_sp.indptr.astype(np.int64)
        right_sp = sparse.csc_matrix(right_mat)
        if right_sp.indices.dtype != np.int64:
            right_sp.indices = right_sp.indices.astype(np.int64)
            right_sp.indptr = right_sp.indptr.astype(np.int64)
        co_occur = (left_sp.T @ right_sp).toarray()  # small: (n_left, n_right)

    valid_pairs = np.argwhere(co_occur >= min_nonzero)

    if len(valid_pairs) == 0:
        return names, rows_list, cols_list, data_list, col_idx

    all_names = [f'{prefix}_{left_names[int(p[0])][:40]}_{right_names[int(p[1])][:40]}'
                 for p in valid_pairs]

    # ── Multi-threaded batch cross multiply ──
    # numpy element-wise * releases the GIL, so threads give true parallelism.
    # Cap at 64 threads — memory bandwidth saturates before that on most machines.
    N = left_mat.shape[0]
    n_valid = len(valid_pairs)

    # Determine thread count first, then size batches to fill threads
    try:
        from hardware_detect import get_cpu_count
        n_cpus = get_cpu_count()
    except ImportError:
        try:
            n_cpus = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            n_cpus = os.cpu_count() or 1
    # Size batches based on available RAM
    BATCH_MAX = _get_cross_batch_size(N)

    # RAM-aware thread cap: each worker holds dense arrays of (N × BATCH × 4 bytes)
    # With OMP_NUM_THREADS set by cloud_run_tf.py, total threads = n_threads × 4 (Numba prange inside each)
    _ram_gb = _get_available_ram_gb()
    _ram_per_worker_gb = max(0.1, N * BATCH_MAX * 8 * 3 / 1e9)  # float64 (8 bytes) × 3 arrays (left+right+result)
    _ram_limited = max(4, int(_ram_gb * 0.4 / _ram_per_worker_gb))
    n_threads = min(_ram_limited, n_cpus, max(64, int(_ram_gb * 0.3 / _ram_per_worker_gb)))  # RAM-aware cap

    # Size batches so we get at least n_threads batches (saturate all threads)
    # But don't make batches too small (< 500 pairs) — overhead dominates
    BATCH = min(BATCH_MAX, max(500, n_valid // n_threads))
    n_batches = (n_valid + BATCH - 1) // BATCH
    n_threads = min(n_threads, n_batches)

    if n_threads <= 1 or n_batches <= 1:
        # Single-threaded fast path (small cross, no overhead)
        for b_start in range(0, n_valid, BATCH):
            b_end = min(b_start + BATCH, n_valid)
            blk_names, blk_rows, blk_data = _process_cross_block(
                left_mat, right_mat, valid_pairs, all_names, b_start, b_end, 0)
            for i in range(len(blk_names)):
                names.append(blk_names[i])
                rows_list.append(blk_rows[i])
                cols_list.append(np.full(len(blk_rows[i]), col_offset + col_idx, dtype=np.int64))
                data_list.append(blk_data[i])
                col_idx += 1
            if max_features and (feats_so_far + col_idx) >= max_features:
                break
        return names, rows_list, cols_list, data_list, col_idx

    # ── Multi-threaded execution ──
    batch_ranges = []
    for b_start in range(0, n_valid, BATCH):
        b_end = min(b_start + BATCH, n_valid)
        batch_ranges.append((b_start, b_end))

    log(f"    Parallel cross: {n_valid} pairs, {len(batch_ranges)} batches, {n_threads} threads")

    # Windowed execution: process batches in groups of 2*n_threads
    # to prevent dense intermediates from accumulating in memory.
    # Results consumed per window, then freed before next window.
    WINDOW = 2 * n_threads  # max inflight futures
    results = [None] * len(batch_ranges)
    for win_start in range(0, len(batch_ranges), WINDOW):
        win_end = min(win_start + WINDOW, len(batch_ranges))
        win_ranges = batch_ranges[win_start:win_end]
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = {}
            for idx_offset, (b_start, b_end) in enumerate(win_ranges):
                f = executor.submit(_process_cross_block,
                                    left_mat, right_mat, valid_pairs, all_names,
                                    b_start, b_end, 0)
                futures[f] = win_start + idx_offset
            for f in as_completed(futures):
                results[futures[f]] = f.result()
        # Free dense intermediates between windows
        gc.collect()
        _malloc_trim()

    # Merge results in order (preserves deterministic feature ordering)
    for blk_names, blk_rows, blk_data in results:
        for i in range(len(blk_names)):
            names.append(blk_names[i])
            rows_list.append(blk_rows[i])
            cols_list.append(np.full(len(blk_rows[i]), col_offset + col_idx, dtype=np.int64))
            data_list.append(blk_data[i])
            col_idx += 1
        if max_features and (feats_so_far + col_idx) >= max_features:
            break

    return names, rows_list, cols_list, data_list, col_idx


# ============================================================
# DOY WINDOW GENERATION (±2 days)
# ============================================================

def create_doy_windows(df, window=2):
    """
    Create DOY ±window day flags. Each window covers 2*window+1 days.
    Returns (doy_names, doy_arrays).
    """
    if hasattr(df.index, 'dayofyear'):
        doy_vals = df.index.dayofyear.values
    elif 'day_of_year' in df.columns:
        doy_vals = pd.to_numeric(df['day_of_year'], errors='coerce').values.astype(int)
    else:
        log("  [WARN] No DOY source, skipping DOY windows")
        return [], []

    names = []
    arrays = []
    for d in range(1, 366):
        # Window wraps around year boundary — vectorized with np.isin
        targets = [(d + offset - 1) % 365 + 1 for offset in range(-window, window + 1)]
        mask = np.isin(doy_vals, targets).astype(np.float32)

        if mask.sum() > 0:
            names.append(f'dw_{d}')
            arrays.append(mask)

    return names, arrays


# ============================================================
# REGIME-AWARE DOY (3x)
# ============================================================

def create_regime_doy(doy_names, doy_arrays, df):
    """
    Split DOY windows by regime: bull, bear, sideways.
    Returns additional (names, arrays) — triples the DOY features.
    """
    # Find regime column
    regime_col = None
    for col in ('regime', 'hmm_regime', 'regime_label'):
        if col in df.columns:
            regime_col = col
            break

    # If no regime column, try ema50_rising as proxy
    if regime_col is None:
        if 'ema50_rising' in df.columns:
            vals = pd.to_numeric(df['ema50_rising'], errors='coerce').values
            # NaN preserved — LightGBM treats missing as unknown split direction
            bull = (vals > 0).astype(np.float32)
            bear = (vals == 0).astype(np.float32)
            sideways = np.zeros(len(df), dtype=np.float32)  # No sideways without HMM
        else:
            return [], []
    else:
        regime_vals = df[regime_col].values
        bull = (regime_vals == 'bull').astype(np.float32) if regime_vals.dtype == object else \
               (regime_vals == 0).astype(np.float32)
        bear = (regime_vals == 'bear').astype(np.float32) if regime_vals.dtype == object else \
               (regime_vals == 1).astype(np.float32)
        sideways = (regime_vals == 'sideways').astype(np.float32) if regime_vals.dtype == object else \
                   (regime_vals == 2).astype(np.float32)

    names = []
    arrays = []
    for dn, da in zip(doy_names, doy_arrays):
        for tag, regime_mask in [('B', bull), ('R', bear), ('S', sideways)]:
            cross = da * regime_mask
            if cross.sum() > 0:
                names.append(f'{dn}_{tag}')
                arrays.append(cross)

    return names, arrays


# ============================================================
# MULTI-SIGNAL COMBINATIONS
# ============================================================

def create_multi_signal_combos(signals, prefix, max_pairs=100):
    """
    Create 2-way combinations of signals firing simultaneously.
    Returns (names, arrays) for the combos.
    """
    names_list = [s[0] for s in signals]
    arrays_list = [s[1] for s in signals]
    n = len(signals)

    combo_names = []
    combo_arrays = []
    count = 0

    # Batched inner loop: for each i, multiply against all j>i at once
    for i in range(n):
        remaining = n - (i + 1)
        if remaining <= 0:
            break

        # Stack all j>i arrays into a matrix (N, remaining)
        right_mat = np.column_stack(arrays_list[i + 1:])  # (N, remaining)

        # Broadcast multiply: (N, 1) * (N, remaining) = (N, remaining)
        combos_batch = arrays_list[i][:, None] * right_mat  # vectorized

        # Compute sums along axis=0 to filter
        sums = combos_batch.sum(axis=0)  # (remaining,)

        # Extract valid combos
        valid_mask = sums >= MIN_CO_OCCURRENCE
        valid_indices = np.nonzero(valid_mask)[0]

        for idx in valid_indices:
            j = i + 1 + idx
            combo_names.append(f'{prefix}_{names_list[i][:15]}_{names_list[j][:15]}')
            combo_arrays.append(combos_batch[:, idx].copy())
            count += 1
            if count >= max_pairs:
                return combo_names, combo_arrays

        del right_mat, combos_batch, sums
        if count >= max_pairs:
            break

    return combo_names, combo_arrays


# ============================================================
# MAIN CROSS GENERATION
# ============================================================

def generate_all_crosses(df, tf='1d', gpu_id=0, save_sparse=False, output_dir=None, max_crosses=None):
    """
    Generate ALL V2 cross features for a single asset's feature DataFrame.

    MEMORY-OPTIMIZED: Each cross type streams directly to sparse chunks.
    We never hold more than one batch of dense arrays at a time.
    Peak RAM stays under ~4GB regardless of total feature count.

    Returns base DataFrame (crosses are in sparse .npz file, not in df).
    """
    t0 = time.time()
    N = len(df)
    out_dir = output_dir or V2_DIR

    # Pin to GPU — CUDA_VISIBLE_DEVICES remaps, so always use Device(0) when set
    if GPU:
        if os.environ.get('CUDA_VISIBLE_DEVICES'):
            cp.cuda.Device(0).use()
        else:
            cp.cuda.Device(gpu_id).use()

    log(f"V2 Cross Generator — {tf.upper()}")
    log(f"  Input: {N:,} rows × {len(df.columns):,} cols")

    # ── Step 1: Binarize all contexts (4-tier) ──
    log("  Step 1: Binarizing contexts (4-tier)...")
    ctx_names, ctx_arrays = binarize_contexts(df, four_tier=True)
    log(f"    {len(ctx_names)} context signals")

    # ── Step 2: Extract signal groups ──
    groups = extract_signal_groups(df, ctx_names, ctx_arrays)
    for g, sigs in groups.items():
        if sigs:
            log(f"    {g}: {len(sigs)} signals")

    # ── Step 3: DOY windows ──
    log("  Step 2: Creating DOY ±2 windows...")
    doy_names, doy_arrays = create_doy_windows(df)
    log(f"    {len(doy_names)} DOY windows")

    # ── Pre-extract TA and astro arrays for targeted crossing ──
    # Crosses 6-12 use targeted right-sides (TA, TA+astro, TA+DOY) instead of ALL contexts.
    # Extract once here so they're available to all cross types.
    ta_n = [s[0] for s in groups['ta']] if groups['ta'] else []
    ta_a = [s[1] for s in groups['ta']] if groups['ta'] else []
    astro_n = [s[0] for s in groups['astro']] if groups['astro'] else []
    astro_a = [s[1] for s in groups['astro']] if groups['astro'] else []

    # ── Accumulate sparse CSR chunks + column names across all cross types ──
    # PERF: Convert each cross type to CSR immediately instead of accumulating
    # one massive COO. hstack of small CSR matrices is much faster than
    # converting a single giant COO→CSR (scipy COO→CSR is single-threaded).
    _csr_chunks = []    # list of CSR matrices, one per cross type
    all_cross_names = []    # flat list of feature names
    col_offset = 0      # running column offset (for gpu_batch_cross)

    _total_collected = 0

    # ── Per-cross-type checkpoint/resume ──
    # Each completed cross type is saved as an individual NPZ + JSON checkpoint.
    # On OOM at cross 12, types 1-11 are recoverable from checkpoint files.
    import json as _json_mod
    _ckpt_dir = out_dir
    _ckpt_pattern = os.path.join(_ckpt_dir, f'_cross_checkpoint_{tf}_*.npz')
    _completed_prefixes = set()
    _checkpoint_files = sorted(glob.glob(_ckpt_pattern))
    if _checkpoint_files:
        log(f"  Found {len(_checkpoint_files)} checkpoint(s), resuming...")
        for _cf in _checkpoint_files:
            _cf_base = os.path.basename(_cf)
            # Pattern: _cross_checkpoint_{tf}_{prefix}.npz
            _cf_prefix = _cf_base.replace(f'_cross_checkpoint_{tf}_', '').replace('.npz', '')
            # Skip names json files that might match glob
            if '_names' in _cf_prefix:
                continue
            _completed_prefixes.add(_cf_prefix)
            # Load the CSR chunk
            _resume_csr = sparse.load_npz(_cf)
            _csr_chunks.append(_resume_csr)
            # Load matching names
            _cf_names_path = os.path.join(_ckpt_dir, f'_cross_checkpoint_{tf}_{_cf_prefix}_names.json')
            if os.path.exists(_cf_names_path):
                with open(_cf_names_path, 'r') as _f:
                    _resume_names = _json_mod.load(_f)
                all_cross_names.extend(_resume_names)
                _total_collected += len(_resume_names)
                col_offset += _resume_csr.shape[1]
                log(f"    Restored checkpoint: {_cf_prefix} ({_resume_csr.shape[1]:,} features, {len(_resume_names):,} names)")
            else:
                log(f"    WARNING: checkpoint {_cf_prefix} missing names file, skipping")
                _csr_chunks.pop()
                _completed_prefixes.discard(_cf_prefix)
            del _resume_csr
        gc.collect()
        _malloc_trim()
        log(f"  Resumed {len(_completed_prefixes)} cross type(s): {sorted(_completed_prefixes)}")
        log(f"  Accumulated: {_total_collected:,} features so far")

    def _save_checkpoint(cross_prefix):
        """Save the last completed cross type's CSR chunk + names as a checkpoint file."""
        if not _csr_chunks:
            return
        _last_chunk = _csr_chunks[-1]
        _n_cols_this = _last_chunk.shape[1]
        _these_names = all_cross_names[-_n_cols_this:]
        _ckpt_npz = os.path.join(_ckpt_dir, f'_cross_checkpoint_{tf}_{cross_prefix}.npz')
        _ckpt_names = os.path.join(_ckpt_dir, f'_cross_checkpoint_{tf}_{cross_prefix}_names.json')
        sparse.save_npz(_ckpt_npz, _last_chunk)
        with open(_ckpt_names, 'w') as _f:
            _json_mod.dump([str(n) for n in _these_names], _f)
        log(f"    Checkpoint saved: {cross_prefix} ({_n_cols_this:,} features)")

    def _cleanup_checkpoints():
        """Delete all checkpoint files after successful final save."""
        _removed = 0
        for _f in glob.glob(os.path.join(_ckpt_dir, f'_cross_checkpoint_{tf}_*.npz')):
            os.remove(_f)
            _removed += 1
        for _f in glob.glob(os.path.join(_ckpt_dir, f'_cross_checkpoint_{tf}_*_names.json')):
            os.remove(_f)
            _removed += 1
        if _removed:
            log(f"  Cleaned up {_removed} checkpoint files")

    def _collect_cross(label, names, rows_list, cols_list, data_list, n_new_cols):
        """Convert cross type results to CSR. Handles both CSR-flushed and legacy COO paths."""
        nonlocal col_offset, _total_collected
        count = len(names)
        if count > 0:
            all_cross_names.extend(names)
            # Check if rows_list contains pre-flushed CSR chunks
            if rows_list is not None and isinstance(rows_list, list) and len(rows_list) > 0 and hasattr(rows_list[0], "indptr"):
                _csr_chunks.extend(rows_list)
                col_offset += n_new_cols
                _total_collected += count
                gc.collect()
                _malloc_trim()
            elif rows_list is not None and cols_list is not None and isinstance(rows_list, list) and len(rows_list) > 0:
                _r = np.concatenate(rows_list)
                _c = np.concatenate(cols_list)
                _d = np.concatenate(data_list)
                _c_local = _c - col_offset
                if cp is not None and cusp is not None:
                    try:
                        _d_gpu = cp.asarray(_d)
                        _r_gpu = cp.asarray(_r)
                        _c_local_gpu = cp.asarray(_c_local)
                        chunk = cusp.coo_matrix((_d_gpu, (_r_gpu, _c_local_gpu)), shape=(N, n_new_cols)).tocsr()
                        chunk = chunk.get()  # Transfer back to scipy CSR on CPU
                        del _d_gpu, _r_gpu, _c_local_gpu
                        cp.get_default_memory_pool().free_all_blocks()
                    except Exception:
                        chunk = sparse.coo_matrix((_d, (_r, _c_local)), shape=(N, n_new_cols)).tocsr()
                else:
                    chunk = sparse.coo_matrix((_d, (_r, _c_local)), shape=(N, n_new_cols)).tocsr()
                _csr_chunks.append(chunk)
                del _r, _c, _c_local, _d, chunk
                gc.collect()
                _malloc_trim()
                col_offset += n_new_cols
                _total_collected += count
            else:
                col_offset += n_new_cols
                _total_collected += count
            log(f"    {label} crosses: {count:,} (total: {_total_collected:,})")
        return count

    def _remaining():
        """How many more crosses can we generate before hitting max_crosses."""
        if max_crosses is None:
            return None
        return max(0, max_crosses - _total_collected)

    def _at_limit():
        """True if we've hit the max_crosses cap."""
        return max_crosses is not None and _total_collected >= max_crosses

    # ── Cross 1: DOY × ALL contexts ──
    if 'dx' in _completed_prefixes:
        log("  Cross 1 (dx): SKIPPED (checkpoint)")
    else:
        log("  Cross 1: DOY windows × ALL contexts...")
        names, r, c, d, nc = gpu_batch_cross(doy_names, doy_arrays, ctx_names, ctx_arrays,
                                              'dx', gpu_id=gpu_id, col_offset=col_offset,
                                              max_features=_remaining())
        _collect_cross('dx_', names, r, c, d, nc)
        _save_checkpoint('dx')
        del names, r, c, d
        gc.collect()
        _malloc_trim()

    # ── Cross 2: Astro × TA ──
    if 'ax' in _completed_prefixes:
        log("  Cross 2 (ax): SKIPPED (checkpoint)")
    elif not _at_limit() and astro_n and ta_n:
        log("  Cross 2: Astro × TA...")
        names, r, c, d, nc = gpu_batch_cross(astro_n, astro_a, ta_n, ta_a, 'ax',
                                              gpu_id=gpu_id, col_offset=col_offset,
                                              max_features=_remaining())
        _collect_cross('ax_', names, r, c, d, nc)
        _save_checkpoint('ax')
        del names, r, c, d
        gc.collect()
        _malloc_trim()

    # ── Cross 3: Multi-astro combos × TA ──
    if 'ax2' in _completed_prefixes:
        log("  Cross 3 (ax2): SKIPPED (checkpoint)")
    elif not _at_limit() and len(groups['astro']) >= 2 and ta_n:
        log("  Cross 3: Multi-astro combos × TA...")
        ax2_names, ax2_arrays = create_multi_signal_combos(groups['astro'], 'a2', max_pairs=50)
        if ax2_names:
            names, r, c, d, nc = gpu_batch_cross(ax2_names, ax2_arrays, ta_n, ta_a, 'ax2',
                                                  gpu_id=gpu_id, col_offset=col_offset,
                                                  max_features=_remaining())
            _collect_cross('ax2_', names, r, c, d, nc)
            _save_checkpoint('ax2')
            del names, r, c, d
        del ax2_names, ax2_arrays
        gc.collect()
        _malloc_trim()

    # ── Cross 4: Multi-TA combos × DOY + astro ──
    if 'ta2' in _completed_prefixes:
        log("  Cross 4 (ta2): SKIPPED (checkpoint)")
    elif not _at_limit() and len(groups['ta']) >= 2:
        log("  Cross 4: Multi-TA combos × DOY + astro...")
        ta2_names, ta2_arrays = create_multi_signal_combos(groups['ta'][:60], 'ta2', max_pairs=30)
        if ta2_names:
            combined_n = doy_names + astro_n
            combined_a = doy_arrays + astro_a
            names, r, c, d, nc = gpu_batch_cross(ta2_names, ta2_arrays, combined_n, combined_a,
                                                  'ta2', gpu_id=gpu_id, col_offset=col_offset,
                                                  max_features=_remaining())
            _collect_cross('ta2_', names, r, c, d, nc)
            _save_checkpoint('ta2')
            del names, r, c, d, combined_n, combined_a
        del ta2_names, ta2_arrays
        gc.collect()
        _malloc_trim()

    # ── Cross 5: Esoteric × TA ──
    if 'ex2' in _completed_prefixes:
        log("  Cross 5 (ex2): SKIPPED (checkpoint)")
    elif not _at_limit() and groups['esoteric'] and ta_n:
        log("  Cross 5: Esoteric × TA...")
        eso_n = [s[0] for s in groups['esoteric']]
        eso_a = [s[1] for s in groups['esoteric']]
        names, r, c, d, nc = gpu_batch_cross(eso_n, eso_a, ta_n, ta_a, 'ex2',
                                              gpu_id=gpu_id, col_offset=col_offset,
                                              max_features=_remaining())
        _collect_cross('ex2_', names, r, c, d, nc)
        _save_checkpoint('ex2')
        del names, r, c, d
        gc.collect()
        _malloc_trim()

    # ── Cross 6: Space weather × TA (targeted — was ALL) ──
    if 'sw' in _completed_prefixes:
        log("  Cross 6 (sw): SKIPPED (checkpoint)")
    elif not _at_limit() and groups['space_weather'] and ta_n:
        log("  Cross 6: Space weather × TA...")
        sw_n = [s[0] for s in groups['space_weather']]
        sw_a = [s[1] for s in groups['space_weather']]
        names, r, c, d, nc = gpu_batch_cross(sw_n, sw_a, ta_n, ta_a, 'sw',
                                              gpu_id=gpu_id, col_offset=col_offset,
                                              max_features=_remaining())
        _collect_cross('sw_', names, r, c, d, nc)
        _save_checkpoint('sw')
        del names, r, c, d
        gc.collect()
        _malloc_trim()

    # ── Cross 7: Session/HOD × TA+astro (targeted — was ALL) ──
    if 'hod' in _completed_prefixes:
        log("  Cross 7 (hod): SKIPPED (checkpoint)")
    elif not _at_limit() and groups['session'] and (ta_n or astro_n):
        log("  Cross 7: Session × TA+astro...")
        hod_n = [s[0] for s in groups['session']]
        hod_a = [s[1] for s in groups['session']]
        hod_right_n = ta_n + astro_n
        hod_right_a = ta_a + astro_a
        names, r, c, d, nc = gpu_batch_cross(hod_n, hod_a, hod_right_n, hod_right_a, 'hod',
                                              gpu_id=gpu_id, col_offset=col_offset,
                                              max_features=_remaining())
        _collect_cross('hod_', names, r, c, d, nc)
        _save_checkpoint('hod')
        del names, r, c, d, hod_right_n, hod_right_a
        gc.collect()
        _malloc_trim()

    # ── Cross 8: Macro × TA (targeted — was ALL) ──
    if 'mx' in _completed_prefixes:
        log("  Cross 8 (mx): SKIPPED (checkpoint)")
    elif not _at_limit() and groups['macro'] and ta_n:
        log("  Cross 8: Macro × TA...")
        mx_n = [s[0] for s in groups['macro']]
        mx_a = [s[1] for s in groups['macro']]
        names, r, c, d, nc = gpu_batch_cross(mx_n, mx_a, ta_n, ta_a, 'mx',
                                              gpu_id=gpu_id, col_offset=col_offset,
                                              max_features=_remaining())
        _collect_cross('mx_', names, r, c, d, nc)
        _save_checkpoint('mx')
        del names, r, c, d
        gc.collect()
        _malloc_trim()

    # ── Cross 9: Volatility regime × TA+DOY (targeted — was ALL) ──
    if 'vx' in _completed_prefixes:
        log("  Cross 9 (vx): SKIPPED (checkpoint)")
    elif not _at_limit() and groups['volatility'] and (ta_n or doy_names):
        log("  Cross 9: Volatility × TA+DOY...")
        vx_n = [s[0] for s in groups['volatility']]
        vx_a = [s[1] for s in groups['volatility']]
        vx_right_n = ta_n + list(doy_names)
        vx_right_a = ta_a + list(doy_arrays)
        names, r, c, d, nc = gpu_batch_cross(vx_n, vx_a, vx_right_n, vx_right_a, 'vx',
                                              gpu_id=gpu_id, col_offset=col_offset,
                                              max_features=_remaining())
        _collect_cross('vx_', names, r, c, d, nc)
        _save_checkpoint('vx')
        del names, r, c, d, vx_right_n, vx_right_a
        gc.collect()
        _malloc_trim()

    # ── Cross 10: Planetary aspects × TA (targeted — was ALL) ──
    if 'asp' in _completed_prefixes:
        log("  Cross 10 (asp): SKIPPED (checkpoint)")
    elif not _at_limit() and groups['aspect'] and ta_n:
        log("  Cross 10: Aspects × TA...")
        asp_n = [s[0] for s in groups['aspect']]
        asp_a = [s[1] for s in groups['aspect']]
        names, r, c, d, nc = gpu_batch_cross(asp_n, asp_a, ta_n, ta_a, 'asp',
                                              gpu_id=gpu_id, col_offset=col_offset,
                                              max_features=_remaining())
        _collect_cross('asp_', names, r, c, d, nc)
        _save_checkpoint('asp')
        del names, r, c, d
        gc.collect()
        _malloc_trim()

    # ── Cross 11: Price numerology × TA (targeted — was ALL) ──
    if 'pn' in _completed_prefixes:
        log("  Cross 11 (pn): SKIPPED (checkpoint)")
    elif not _at_limit() and groups['price_num'] and ta_n:
        log("  Cross 11: Price numerology × TA...")
        pn_n = [s[0] for s in groups['price_num']]
        pn_a = [s[1] for s in groups['price_num']]
        names, r, c, d, nc = gpu_batch_cross(pn_n, pn_a, ta_n, ta_a, 'pn',
                                              gpu_id=gpu_id, col_offset=col_offset,
                                              max_features=_remaining())
        _collect_cross('pn_', names, r, c, d, nc)
        _save_checkpoint('pn')
        del names, r, c, d
        gc.collect()
        _malloc_trim()

    # ── Cross 12: Moon position × TA (targeted — was ALL) ──
    if 'mn' in _completed_prefixes:
        log("  Cross 12 (mn): SKIPPED (checkpoint)")
    elif not _at_limit() and groups['moon'] and ta_n:
        log("  Cross 12: Moon × TA...")
        mn_n = [s[0] for s in groups['moon']]
        mn_a = [s[1] for s in groups['moon']]
        names, r, c, d, nc = gpu_batch_cross(mn_n, mn_a, ta_n, ta_a, 'mn',
                                              gpu_id=gpu_id, col_offset=col_offset,
                                              max_features=_remaining())
        _collect_cross('mn_', names, r, c, d, nc)
        _save_checkpoint('mn')
        del names, r, c, d
        gc.collect()
        _malloc_trim()

    # ── Cross 13: REMOVED — Regime-aware DOY was redundant with dx_ (Cross 1) ──
    # Regime DOY crossed ALL contexts, but dx_ already crosses DOY × ALL.
    # The regime segmentation (bull/bear/neutral) added noise without unique signal.
    # Kept create_regime_doy() function intact in case needed for future experiments.

    # ── Save inference artifacts (for live cross computation) ──
    if len(all_cross_names) > 0:
        try:
            from inference_crosses import save_inference_artifacts
            # Combine all context names: binarized + DOY windows + regime DOY
            # The cross generator uses ctx_names (binarized) + doy_names as left/right
            # for different cross types. Merge them all for inference.
            all_ctx_names = list(ctx_names) + list(doy_names)
            all_ctx_arrays = list(ctx_arrays) + list(doy_arrays)
            log("  Saving inference artifacts...")
            save_inference_artifacts(
                all_ctx_names, all_ctx_arrays, all_cross_names, df, tf,
                output_dir=out_dir,
            )
        except Exception as e:
            log(f"  WARNING: Failed to save inference artifacts: {e}")
            import traceback
            traceback.print_exc()

    # ── Free context arrays — no longer needed ──
    del ctx_names, ctx_arrays, doy_names, doy_arrays, groups, ta_n, ta_a, astro_n, astro_a
    gc.collect()
    _malloc_trim()

    # ── SAVE CROSSES AS SPARSE (memory efficient) ──
    total_crosses = len(all_cross_names)
    log(f"\n  TOTAL NEW FEATURES: {total_crosses:,}")

    t_assign = time.time()

    if total_crosses > 0 and _csr_chunks:
        cross_names = all_cross_names

        # Memory-safe hstack via CSC splicing (no COO intermediate)
        log(f"  Memory-safe CSC splice of {len(_csr_chunks)} chunks...")
        csc_chunks = [c.tocsc() for c in _csr_chunks]
        del _csr_chunks
        gc.collect()
        _malloc_trim()
        _all_data = np.concatenate([c.data for c in csc_chunks])
        _all_indices = np.concatenate([c.indices for c in csc_chunks])
        _cumulative = 0
        _indptr_parts = []
        for c in csc_chunks:
            _indptr_parts.append(c.indptr[:-1] + _cumulative)
            _cumulative += c.nnz
        _indptr_parts.append(np.array([_cumulative], dtype=np.int64))
        _new_indptr = np.concatenate(_indptr_parts)
        n_cols = sum(c.shape[1] for c in csc_chunks)
        sparse_mat = sparse.csc_matrix((_all_data, _all_indices, _new_indptr), shape=(N, n_cols)).tocsr()
        del csc_chunks, _all_data, _all_indices, _indptr_parts, _new_indptr
        gc.collect()
        _malloc_trim()

        # Force int64 to prevent silent overflow at >2B NNZ
        if sparse_mat.nnz > 2**30:
            sparse_mat.indices = sparse_mat.indices.astype(np.int64)
            sparse_mat.indptr = sparse_mat.indptr.astype(np.int64)
            log(f"  Upgraded to int64 indices (NNZ={sparse_mat.nnz:,})")


        log(f"  Sparse matrix: {sparse_mat.shape}, {sparse_mat.nnz:,} non-zeros, "
            f"density={sparse_mat.nnz / (sparse_mat.shape[0] * sparse_mat.shape[1]) * 100:.3f}%")

        # Save sparse matrix + column names
        import json
        symbol_tag = ''
        if hasattr(df, 'attrs') and 'symbol' in df.attrs:
            symbol_tag = f"_{df.attrs['symbol']}"

        # Per-symbol naming for modular builds
        symbol_tag = getattr(df, '_v2_symbol', None) or ''
        if symbol_tag:
            npz_path = os.path.join(out_dir, f'v2_crosses_{symbol_tag}_{tf}.npz')
            names_path = os.path.join(out_dir, f'v2_cross_names_{symbol_tag}_{tf}.json')
        else:
            npz_path = os.path.join(out_dir, f'v2_crosses_{tf}.npz')
            names_path = os.path.join(out_dir, f'v2_cross_names_{tf}.json')

        from atomic_io import atomic_save_npz, atomic_save_json
        # Ensure cross names are native Python strings (not np.str_) for JSON
        # Dedup names from truncation collisions — keep first occurrence, drop duplicates
        # CRITICAL: dedup BEFORE saving NPZ so matrix cols match name count
        seen = set()
        dedup_indices = []
        for i, n in enumerate(cross_names):
            s = str(n)
            if s in seen:
                continue
            seen.add(s)
            dedup_indices.append(i)
        if len(dedup_indices) < len(cross_names):
            n_dups = len(cross_names) - len(dedup_indices)
            log(f"  Deduplicating {n_dups} cross names from truncation collisions")
            cross_names = [str(cross_names[i]) for i in dedup_indices]
            # Also remove duplicate columns from sparse matrix
            sparse_mat = sparse_mat[:, dedup_indices]
        else:
            cross_names = [str(n) for n in cross_names]
        atomic_save_npz(sparse_mat, npz_path)
        # FIX 26: Verify NPZ integrity immediately after save
        try:
            _verify = sparse.load_npz(npz_path)
            assert _verify.shape == sparse_mat.shape, f"NPZ verify failed: shape {_verify.shape} != {sparse_mat.shape}"
            del _verify
            log(f"  NPZ verified: {npz_path} ({os.path.getsize(npz_path)/1e6:.1f} MB)")
        except Exception as e:
            log(f"  NPZ CORRUPT: {e} — deleting and retrying")
            os.remove(npz_path)
            raise
        atomic_save_json(cross_names, names_path)

        # Final save succeeded — clean up per-cross-type checkpoints
        _cleanup_checkpoints()

        size_mb = os.path.getsize(npz_path) / 1e6
        log(f"  Saved: {npz_path} ({size_mb:.1f} MB)")

        # Also save base features (non-cross) as parquet
        base_path = os.path.join(out_dir, f'v2_base_{tf}.parquet')
        df.to_parquet(base_path)
        log(f"  Saved base: {base_path}")

        del sparse_mat
        gc.collect()
        _malloc_trim()

    log(f"  Sparse build: {time.time()-t_assign:.1f}s")

    elapsed = time.time() - t0
    log(f"\n  DONE: {N:,} rows, {len(df.columns):,} base cols + {total_crosses:,} cross cols ({elapsed:.0f}s)")

    # Final cleanup
    del all_cross_names
    gc.collect()
    _malloc_trim()

    # Return base df (crosses are in sparse file, not in df — too big for dense)
    return df


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='V2 Cross Feature Generator')
    parser.add_argument('--tf', nargs='+', default=['1d'], help='Timeframes to process')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--save-sparse', action='store_true', help='Save sparse .npz files')
    parser.add_argument('--input', help='Input parquet path (overrides auto-detect)')
    parser.add_argument('--symbol', default='BTC', help='Asset symbol')
    args = parser.parse_args()

    for tf in args.tf:
        # Look for input parquet
        if args.input:
            path = args.input
        else:
            # Try V2 path first, then V1
            candidates = [
                os.path.join(V2_DIR, f'features_{args.symbol}_{tf}.parquet'),
                os.path.join(V2_DIR, f'features_{tf}.parquet'),
            ]
            path = None
            for c in candidates:
                if os.path.exists(c):
                    path = c
                    break

            if path is None:
                log(f"[SKIP] No parquet found for {args.symbol} {tf}")
                continue

        log(f"Loading {path}...")
        df = pd.read_parquet(path)
        # Set symbol attribute so output files include symbol in name
        # (v2_crosses_BTC_{tf}.npz, not v2_crosses_{tf}.npz)
        # pd.read_parquet doesn't preserve custom attributes
        df._v2_symbol = args.symbol
        df = generate_all_crosses(df, tf=tf, gpu_id=args.gpu,
                                   save_sparse=args.save_sparse,
                                   output_dir=V2_DIR)

        # Save expanded parquet
        out_path = os.path.join(V2_DIR, f'features_{args.symbol}_{tf}_v2.parquet')
        df.to_parquet(out_path)
        log(f"Saved: {out_path}")
