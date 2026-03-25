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

import os, time, argparse, warnings, gc
import numpy as np
import pandas as pd
from scipy import sparse

warnings.filterwarnings('ignore')

V2_DIR = os.path.dirname(os.path.abspath(__file__))

# ── GPU setup ──
try:
    import cupy as cp
    GPU = True
except ImportError:
    GPU = False


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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
        return 2000   # cloud: 512GB+ RAM, process 2000 contexts at a time
    elif ram_gb >= 256:
        return 1000   # cloud: 256GB RAM
    elif ram_gb >= 128:
        return 500    # mid-range
    elif ram_gb >= 64:
        return 200    # local 64GB
    else:
        return 100    # low RAM

# Env var override for orchestrator OOM retry
_env_chunk = os.environ.get('V2_RIGHT_CHUNK')
RIGHT_CHUNK = int(_env_chunk) if _env_chunk else _get_right_chunk()
log(f"Adaptive RIGHT_CHUNK = {RIGHT_CHUNK} (detected RAM{', env override' if _env_chunk else ''})")

# Minimum co-occurrence threshold for cross features.
# Crosses firing fewer than this many times cannot reliably appear in both
# CPCV train AND validation splits (98.3% coverage at n=8 with 5-fold CPCV).
# This is a math constraint on the validation pipeline, not a signal filter.
_env_co_occur = os.environ.get('V2_MIN_CO_OCCURRENCE')
MIN_CO_OCCURRENCE = int(_env_co_occur) if _env_co_occur else 8


# ── GPU VRAM-adaptive batch sizing ──
def _get_gpu_vram_gb(gpu_id=0):
    """Detect GPU VRAM in GB."""
    try:
        import cupy as cp
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

def binarize_contexts(df, four_tier=True):
    """
    Binarize all suitable columns into binary context arrays.
    Returns: (ctx_names, ctx_arrays) where ctx_arrays is list of float32 arrays.

    With four_tier=True:
      EXTREME_HIGH (>95th), HIGH (>75th), LOW (<25th), EXTREME_LOW (<5th)
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

    seen_cols = set()
    for col in df.columns:
        if col in seen_cols:
            continue  # skip duplicate column names
        seen_cols.add(col)
        if col.startswith(skip_pre) or col in skip_ex:
            continue

        raw = df[col]
        if isinstance(raw, pd.DataFrame):
            raw = raw.iloc[:, 0]  # handle duplicate column names
        vals = pd.to_numeric(raw, errors='coerce').values.astype(np.float32)
        # NaN preserved — LightGBM treats missing as unknown split direction
        uniq = np.unique(vals[~np.isnan(vals)])
        if len(uniq) <= 1:
            continue

        if len(uniq) <= 3:
            # Already binary — use directly
            b = (vals > 0).astype(np.float32)
            if 5 < np.nansum(b) < N * 0.98:
                ctx_names.append(col)
                ctx_arrays.append(b)
        else:
            try:
                nz = vals[vals != 0] if np.nansum(vals != 0) > 100 else vals
                nz = nz[~np.isnan(nz)]  # percentile cannot handle NaN
                if four_tier:
                    # 4-tier: EXTREME_HIGH, HIGH, LOW, EXTREME_LOW
                    q95, q75, q25, q5 = np.percentile(nz, [95, 75, 25, 5])
                    for tag, mask in [('XH', vals > q95), ('H', vals > q75),
                                      ('L', vals < q25), ('XL', vals < q5)]:
                        m = mask.astype(np.float32)
                        if np.nansum(m) > 5:
                            ctx_names.append(f'{col}_{tag}')
                            ctx_arrays.append(m)
                else:
                    q80, q20 = np.percentile(nz, [80, 20])
                    for tag, mask in [('H', vals > q80), ('L', vals < q20)]:
                        m = mask.astype(np.float32)
                        if np.nansum(m) > 5:
                            ctx_names.append(f'{col}_{tag}')
                            ctx_arrays.append(m)
            except:
                pass

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
    total_feats = 0
    current_offset = col_offset

    # Process right-side in chunks of RIGHT_CHUNK
    for rc_start in range(0, len(right_names), RIGHT_CHUNK):
        rc_end = min(rc_start + RIGHT_CHUNK, len(right_names))
        r_names_chunk = right_names[rc_start:rc_end]
        r_arrays_chunk = right_arrays[rc_start:rc_end]
        right_mat_chunk = np.column_stack(r_arrays_chunk)  # (N, <=RIGHT_CHUNK)

        # Skip GPU for large row counts (>100K) — guaranteed OOM on any GPU.
        # Also skip if V2_SKIP_GPU=1 env var is set.
        _skip_gpu = os.environ.get('V2_SKIP_GPU') == '1' or N > 100000
        if GPU and not _skip_gpu:
            try:
                c_names, c_rows, c_cols, c_data, c_ncols = _gpu_cross_chunk(
                    left_names, left_mat, r_names_chunk, right_mat_chunk,
                    prefix, gpu_id, min_nonzero, max_features, total_feats,
                    col_offset=current_offset
                )
            except Exception as e:
                log(f"  GPU failed ({e}), falling back to CPU for chunk")
                c_names, c_rows, c_cols, c_data, c_ncols = _cpu_cross_chunk(
                    left_names, left_mat, r_names_chunk, right_mat_chunk,
                    prefix, min_nonzero, max_features, total_feats,
                    col_offset=current_offset
                )
        else:
            c_names, c_rows, c_cols, c_data, c_ncols = _cpu_cross_chunk(
                left_names, left_mat, r_names_chunk, right_mat_chunk,
                prefix, min_nonzero, max_features, total_feats,
                col_offset=current_offset
            )

        if c_names:
            all_names.extend(c_names)
            all_rows.extend(c_rows)
            all_cols.extend(c_cols)
            all_data.extend(c_data)
            current_offset += c_ncols
            total_feats += len(c_names)

        del right_mat_chunk, r_arrays_chunk

        if max_features and total_feats >= max_features:
            break

    del left_mat
    n_total_cols = current_offset - col_offset
    return all_names, all_rows, all_cols, all_data, n_total_cols


def _gpu_cross_chunk(left_names, left_mat, right_names, right_mat, prefix,
                     gpu_id, min_nonzero, max_features, feats_so_far,
                     col_offset=0):
    """
    GPU cross multiply — returns COO triplets directly, never stores dense columns.
    Peak memory = tensor (N × BATCH × n_right) + COO triplets (sparse, tiny).
    """
    names = []
    rows_list = []
    cols_list = []
    data_list = []
    n_right = right_mat.shape[1]
    n_bars = left_mat.shape[0]
    col_idx = 0

    gpu_vram_gb = _get_gpu_vram_gb(gpu_id=gpu_id)
    BATCH = _get_optimal_batch(n_bars, n_right, gpu_vram_gb, gpu_id=gpu_id)
    log(f"  GPU VRAM: {gpu_vram_gb:.1f} GB | n_bars={n_bars} n_right={n_right} -> BATCH={BATCH}")

    _dev = 0 if os.environ.get('CUDA_VISIBLE_DEVICES') else gpu_id
    cp.cuda.Device(_dev).use()
    right_gpu = cp.asarray(np.ascontiguousarray(right_mat))

    for b_start in range(0, left_mat.shape[1], BATCH):
        b_end = min(b_start + BATCH, left_mat.shape[1])
        left_batch = cp.asarray(np.ascontiguousarray(left_mat[:, b_start:b_end]))

        # GPU outer product: (N, batch, 1) * (N, 1, n_right) = (N, batch, n_right)
        crosses = left_batch[:, :, None] * right_gpu[:, None, :]

        # Sum along rows on GPU to check co-occurrence threshold
        sums = cp.sum(crosses, axis=0)  # (batch, n_right)

        # Find valid (i, j) pairs on GPU — avoids transferring entire tensor
        valid_ij = cp.argwhere(sums >= min_nonzero)  # (K, 2)

        if len(valid_ij) > 0:
            valid_ij_cpu = cp.asnumpy(valid_ij)
            for idx in range(len(valid_ij_cpu)):
                i, j = int(valid_ij_cpu[idx, 0]), int(valid_ij_cpu[idx, 1])
                ln = left_names[b_start + i]
                rn = right_names[j]
                fname = f'{prefix}_{ln[:40]}_{rn[:40]}'

                # Extract single column from GPU, get nonzeros directly
                col_gpu = crosses[:, i, j]
                nz_mask = col_gpu != 0
                nz_rows = cp.asnumpy(cp.flatnonzero(nz_mask))
                nz_vals = cp.asnumpy(col_gpu[nz_mask]).astype(np.float32)

                if len(nz_rows) > 0:
                    names.append(fname)
                    rows_list.append(nz_rows)
                    cols_list.append(np.full(len(nz_rows), col_offset + col_idx, dtype=np.int64))
                    data_list.append(nz_vals)
                    col_idx += 1

                if max_features and (feats_so_far + col_idx) >= max_features:
                    break

        del crosses, left_batch, sums, valid_ij
        cp.get_default_memory_pool().free_all_blocks()

        if max_features and (feats_so_far + col_idx) >= max_features:
            break

    del right_gpu
    cp.get_default_memory_pool().free_all_blocks()
    return names, rows_list, cols_list, data_list, col_idx


def _cpu_cross_chunk(left_names, left_mat, right_names, right_mat, prefix,
                     min_nonzero, max_features, feats_so_far, col_offset=0):
    """
    CPU cross multiply — returns COO triplets directly, never stores dense columns.
    """
    names = []
    rows_list = []
    cols_list = []
    data_list = []
    col_idx = 0

    for i in range(left_mat.shape[1]):
        la = left_mat[:, i]
        ln = left_names[i]
        for j in range(right_mat.shape[1]):
            ra = right_mat[:, j]
            cross = la * ra
            if cross.sum() >= min_nonzero:
                rn = right_names[j]
                fname = f'{prefix}_{ln[:40]}_{rn[:40]}'
                nz = np.nonzero(cross)[0]
                if len(nz) > 0:
                    names.append(fname)
                    rows_list.append(nz)
                    cols_list.append(np.full(len(nz), col_offset + col_idx, dtype=np.int64))
                    data_list.append(cross[nz].astype(np.float32))
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

    # ── Accumulate sparse CSR chunks + column names across all cross types ──
    # PERF: Convert each cross type to CSR immediately instead of accumulating
    # one massive COO. hstack of small CSR matrices is much faster than
    # converting a single giant COO→CSR (scipy COO→CSR is single-threaded).
    _csr_chunks = []    # list of CSR matrices, one per cross type
    all_cross_names = []    # flat list of feature names
    col_offset = 0      # running column offset (for gpu_batch_cross)

    _total_collected = 0

    def _collect_cross(label, names, rows_list, cols_list, data_list, n_new_cols):
        """Convert one cross type's COO triplets to CSR immediately, then free COO."""
        nonlocal col_offset, _total_collected
        count = len(names)
        if count > 0 and rows_list:
            all_cross_names.extend(names)
            _r = np.concatenate(rows_list)
            _c = np.concatenate(cols_list)
            _d = np.concatenate(data_list)
            # gpu_batch_cross returns columns offset by col_offset (global indices).
            # Shift to 0-based local indices for this chunk's standalone CSR.
            _c_local = _c - col_offset
            chunk = sparse.coo_matrix((_d, (_r, _c_local)), shape=(N, n_new_cols)).tocsr()
            _csr_chunks.append(chunk)
            del _r, _c, _c_local, _d, chunk
            col_offset += n_new_cols
            _total_collected += count
        elif count > 0:
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
    log("  Cross 1: DOY windows × ALL contexts...")
    names, r, c, d, nc = gpu_batch_cross(doy_names, doy_arrays, ctx_names, ctx_arrays,
                                          'dx', gpu_id=gpu_id, col_offset=col_offset,
                                          max_features=_remaining())
    _collect_cross('dx_', names, r, c, d, nc)
    del names, r, c, d
    gc.collect()

    # ── Cross 2: Astro × TA ──
    if not _at_limit() and groups['astro'] and groups['ta']:
        log("  Cross 2: Astro × TA...")
        astro_n = [s[0] for s in groups['astro']]
        astro_a = [s[1] for s in groups['astro']]
        ta_n = [s[0] for s in groups['ta']]
        ta_a = [s[1] for s in groups['ta']]
        names, r, c, d, nc = gpu_batch_cross(astro_n, astro_a, ta_n, ta_a, 'ax',
                                              gpu_id=gpu_id, col_offset=col_offset,
                                              max_features=_remaining())
        _collect_cross('ax_', names, r, c, d, nc)
        del names, r, c, d
        gc.collect()

    # ── Cross 3: Multi-astro combos × TA ──
    if not _at_limit() and len(groups['astro']) >= 2 and groups['ta']:
        log("  Cross 3: Multi-astro combos × TA...")
        ax2_names, ax2_arrays = create_multi_signal_combos(groups['astro'], 'a2', max_pairs=50)
        if ax2_names:
            ta_n = [s[0] for s in groups['ta']]
            ta_a = [s[1] for s in groups['ta']]
            names, r, c, d, nc = gpu_batch_cross(ax2_names, ax2_arrays, ta_n, ta_a, 'ax2',
                                                  gpu_id=gpu_id, col_offset=col_offset,
                                                  max_features=_remaining())
            _collect_cross('ax2_', names, r, c, d, nc)
            del names, r, c, d
        del ax2_names, ax2_arrays
        gc.collect()

    # ── Cross 4: Multi-TA combos × DOY + astro ──
    if not _at_limit() and len(groups['ta']) >= 2:
        log("  Cross 4: Multi-TA combos × DOY + astro...")
        ta2_names, ta2_arrays = create_multi_signal_combos(groups['ta'][:60], 'ta2', max_pairs=30)
        if ta2_names:
            combined_n = doy_names + [s[0] for s in groups['astro']]
            combined_a = doy_arrays + [s[1] for s in groups['astro']]
            names, r, c, d, nc = gpu_batch_cross(ta2_names, ta2_arrays, combined_n, combined_a,
                                                  'ta2', gpu_id=gpu_id, col_offset=col_offset,
                                                  max_features=_remaining())
            _collect_cross('ta2_', names, r, c, d, nc)
            del names, r, c, d, combined_n, combined_a
        del ta2_names, ta2_arrays
        gc.collect()

    # ── Cross 5: Esoteric × TA ──
    if not _at_limit() and groups['esoteric'] and groups['ta']:
        log("  Cross 5: Esoteric × TA...")
        eso_n = [s[0] for s in groups['esoteric']]
        eso_a = [s[1] for s in groups['esoteric']]
        ta_n = [s[0] for s in groups['ta']]
        ta_a = [s[1] for s in groups['ta']]
        names, r, c, d, nc = gpu_batch_cross(eso_n, eso_a, ta_n, ta_a, 'ex2',
                                              gpu_id=gpu_id, col_offset=col_offset,
                                              max_features=_remaining())
        _collect_cross('ex2_', names, r, c, d, nc)
        del names, r, c, d
        gc.collect()

    # ── Cross 6: Space weather × ALL ──
    if not _at_limit() and groups['space_weather']:
        log("  Cross 6: Space weather × ALL contexts...")
        sw_n = [s[0] for s in groups['space_weather']]
        sw_a = [s[1] for s in groups['space_weather']]
        names, r, c, d, nc = gpu_batch_cross(sw_n, sw_a, ctx_names, ctx_arrays, 'sw',
                                              gpu_id=gpu_id, col_offset=col_offset,
                                              max_features=_remaining())
        _collect_cross('sw_', names, r, c, d, nc)
        del names, r, c, d
        gc.collect()

    # ── Cross 7: Session/HOD × ALL (intraday only) ──
    if not _at_limit() and groups['session']:
        log("  Cross 7: Session × ALL contexts...")
        hod_n = [s[0] for s in groups['session']]
        hod_a = [s[1] for s in groups['session']]
        names, r, c, d, nc = gpu_batch_cross(hod_n, hod_a, ctx_names, ctx_arrays, 'hod',
                                              gpu_id=gpu_id, col_offset=col_offset,
                                              max_features=_remaining())
        _collect_cross('hod_', names, r, c, d, nc)
        del names, r, c, d
        gc.collect()

    # ── Cross 8: Macro × ALL ──
    if not _at_limit() and groups['macro']:
        log("  Cross 8: Macro × ALL contexts...")
        mx_n = [s[0] for s in groups['macro']]
        mx_a = [s[1] for s in groups['macro']]
        names, r, c, d, nc = gpu_batch_cross(mx_n, mx_a, ctx_names, ctx_arrays, 'mx',
                                              gpu_id=gpu_id, col_offset=col_offset,
                                              max_features=_remaining())
        _collect_cross('mx_', names, r, c, d, nc)
        del names, r, c, d
        gc.collect()

    # ── Cross 9: Volatility regime × ALL ──
    if not _at_limit() and groups['volatility']:
        log("  Cross 9: Volatility × ALL contexts...")
        vx_n = [s[0] for s in groups['volatility']]
        vx_a = [s[1] for s in groups['volatility']]
        names, r, c, d, nc = gpu_batch_cross(vx_n, vx_a, ctx_names, ctx_arrays, 'vx',
                                              gpu_id=gpu_id, col_offset=col_offset,
                                              max_features=_remaining())
        _collect_cross('vx_', names, r, c, d, nc)
        del names, r, c, d
        gc.collect()

    # ── Cross 10: Planetary aspects × ALL ──
    if not _at_limit() and groups['aspect']:
        log("  Cross 10: Aspects × ALL contexts...")
        asp_n = [s[0] for s in groups['aspect']]
        asp_a = [s[1] for s in groups['aspect']]
        names, r, c, d, nc = gpu_batch_cross(asp_n, asp_a, ctx_names, ctx_arrays, 'asp',
                                              gpu_id=gpu_id, col_offset=col_offset,
                                              max_features=_remaining())
        _collect_cross('asp_', names, r, c, d, nc)
        del names, r, c, d
        gc.collect()

    # ── Cross 11: Price numerology × ALL ──
    if not _at_limit() and groups['price_num']:
        log("  Cross 11: Price numerology × ALL contexts...")
        pn_n = [s[0] for s in groups['price_num']]
        pn_a = [s[1] for s in groups['price_num']]
        names, r, c, d, nc = gpu_batch_cross(pn_n, pn_a, ctx_names, ctx_arrays, 'pn',
                                              gpu_id=gpu_id, col_offset=col_offset,
                                              max_features=_remaining())
        _collect_cross('pn_', names, r, c, d, nc)
        del names, r, c, d
        gc.collect()

    # ── Cross 12: Moon position × ALL ──
    if not _at_limit() and groups['moon']:
        log("  Cross 12: Moon × ALL contexts...")
        mn_n = [s[0] for s in groups['moon']]
        mn_a = [s[1] for s in groups['moon']]
        names, r, c, d, nc = gpu_batch_cross(mn_n, mn_a, ctx_names, ctx_arrays, 'mn',
                                              gpu_id=gpu_id, col_offset=col_offset,
                                              max_features=_remaining())
        _collect_cross('mn_', names, r, c, d, nc)
        del names, r, c, d
        gc.collect()

    # ── Cross 13: Regime-aware DOY (3x) ──
    if _at_limit():
        log("  Cross 13: SKIPPED (max_crosses reached)")
    else:
        log("  Cross 13: Regime-aware DOY (3x)...")
        reg_names, reg_arrays = create_regime_doy(doy_names, doy_arrays, df)
        if reg_names:
            names, r, c, d, nc = gpu_batch_cross(reg_names, reg_arrays, ctx_names, ctx_arrays,
                                                  'rdx', gpu_id=gpu_id, col_offset=col_offset,
                                                  max_features=_remaining())
            _collect_cross('rdx_', names, r, c, d, nc)
            del names, r, c, d
        del reg_names, reg_arrays
        gc.collect()

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
    del ctx_names, ctx_arrays, doy_names, doy_arrays, groups
    gc.collect()

    # ── SAVE CROSSES AS SPARSE (memory efficient) ──
    total_crosses = len(all_cross_names)
    log(f"\n  TOTAL NEW FEATURES: {total_crosses:,}")

    t_assign = time.time()

    if total_crosses > 0 and _csr_chunks:
        cross_names = all_cross_names

        # hstack pre-built CSR chunks (much faster than one giant COO→CSR)
        log(f"  hstack {len(_csr_chunks)} CSR chunks into ({N:,} x {col_offset:,})...")
        sparse_mat = sparse.hstack(_csr_chunks, format='csr')
        del _csr_chunks
        gc.collect()

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
        atomic_save_npz(sparse_mat, npz_path)
        # Ensure cross names are native Python strings (not np.str_) for JSON
        # Dedup names from truncation collisions — keep first occurrence, drop duplicates
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
        atomic_save_json(cross_names, names_path)

        size_mb = os.path.getsize(npz_path) / 1e6
        log(f"  Saved: {npz_path} ({size_mb:.1f} MB)")

        # Also save base features (non-cross) as parquet
        base_path = os.path.join(out_dir, f'v2_base_{tf}.parquet')
        df.to_parquet(base_path)
        log(f"  Saved base: {base_path}")

        del sparse_mat
        gc.collect()

    log(f"  Sparse build: {time.time()-t_assign:.1f}s")

    elapsed = time.time() - t0
    log(f"\n  DONE: {N:,} rows, {len(df.columns):,} base cols + {total_crosses:,} cross cols ({elapsed:.0f}s)")

    # Final cleanup
    del all_cross_names
    gc.collect()

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
