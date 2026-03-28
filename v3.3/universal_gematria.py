#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
universal_gematria.py — Universal Gematria Engine
===================================================
Single source of truth for ALL gematria calculations.
Takes any text string, returns all cipher values.

8 ciphers:
  1. English Ordinal: A=1, B=2 ... Z=26
  2. Reverse Ordinal: A=26, B=25 ... Z=1
  3. Reduction: each letter reduced to 1-9, then summed
  4. English Gematria: A=6, B=12 ... Z=156 (multiples of 6)
  5. Jewish/Hebrew: traditional Hebrew values mapped to English
  6. Satanic: A=36, B=37 ... Z=61
  7. Chaldean: ancient Babylonian cipher (9 is never assigned to any letter)
  8. AlBam: Kabbalistic rotation cipher (A<->N, B<->O, rotate by 13)

GPU batch mode:
  from universal_gematria import gematria_gpu_batch
  df = gematria_gpu_batch(cudf_string_series, prefix='tweet_gem')
  # Returns pandas DataFrame with all 6 cipher values + DRs + flags per row

Usage:
  from universal_gematria import gematria, gematria_match, gematria_contains_target
  result = gematria("Bitcoin")  # returns dict with all values
"""

import os
import numpy as np


# ============================================================
# CIPHER TABLES
# ============================================================

# English Ordinal: A=1..Z=26
def _ordinal(c):
    return ord(c.upper()) - 64

# Reverse Ordinal: A=26..Z=1
def _reverse(c):
    return 27 - (ord(c.upper()) - 64)

# English Gematria: A=6..Z=156 (multiples of 6)
def _english(c):
    return (ord(c.upper()) - 64) * 6

# Jewish/Hebrew mapped to English alphabet
# Standard Hebrew values: A=1,B=2,C=3,D=4,E=5,F=6,G=7,H=8,I=9,J=600,K=10,L=20,M=30,
# N=40,O=50,P=60,Q=70,R=80,S=90,T=100,U=200,V=700,W=900,X=300,Y=400,Z=500
_JEWISH = {
    'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9,
    'J': 600, 'K': 10, 'L': 20, 'M': 30, 'N': 40, 'O': 50, 'P': 60, 'Q': 70,
    'R': 80, 'S': 90, 'T': 100, 'U': 200, 'V': 700, 'W': 900, 'X': 300, 'Y': 400, 'Z': 500,
}

def _jewish(c):
    return _JEWISH.get(c.upper(), 0)

# Satanic: A=36..Z=61
def _satanic(c):
    return ord(c.upper()) - 64 + 35


# Chaldean: ancient Babylonian cipher (9 is NEVER assigned to any letter — 9 is sacred)
_CHALDEAN = {
    'a':1,'i':1,'j':1,'q':1,'y':1,
    'b':2,'k':2,'r':2,
    'c':3,'g':3,'l':3,'s':3,
    'd':4,'m':4,'t':4,
    'e':5,'h':5,'n':5,'x':5,
    'u':6,'v':6,'w':6,
    'o':7,'z':7,
    'f':8,'p':8,
}

def _chaldean(c):
    return _CHALDEAN.get(c.lower(), 0)


# AlBam: Kabbalistic rotation cipher (A<->N, B<->O, ..., rotate by 13)
def _albam(c):
    if c.isalpha():
        idx = ord(c.lower()) - ord('a')  # 0-25
        rotated = (idx + 13) % 26        # rotate by 13
        return rotated + 1               # 1-26
    return 0


# ============================================================
# DIGITAL ROOT
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
    while n > 9 and n not in (11, 22, 33):
        n = sum(int(d) for d in str(n))
    return n


# ============================================================
# CORE FUNCTIONS
# ============================================================

def ordinal(text):
    """English Ordinal gematria. A=1..Z=26."""
    return sum(_ordinal(c) for c in text if c.isalpha())

def reverse(text):
    """Reverse Ordinal gematria. A=26..Z=1."""
    return sum(_reverse(c) for c in text if c.isalpha())

def reduction(text):
    """Reduction gematria. Each letter reduced to 1-9, then summed."""
    total = 0
    for c in text:
        if c.isalpha():
            val = _ordinal(c)
            while val > 9:
                val = sum(int(d) for d in str(val))
            total += val
    return total

def english(text):
    """English Gematria. A=6..Z=156."""
    return sum(_english(c) for c in text if c.isalpha())

def jewish(text):
    """Jewish/Hebrew gematria mapped to English letters."""
    return sum(_jewish(c) for c in text if c.isalpha())

def satanic(text):
    """Satanic gematria. A=36..Z=61."""
    return sum(_satanic(c) for c in text if c.isalpha())


def chaldean(text):
    """Chaldean numerology cipher. 9 is never assigned to any letter (9 is sacred)."""
    return sum(_CHALDEAN.get(c, 0) for c in str(text).lower())


def albam(text):
    """AlBam cipher: Kabbalistic rotation (A->N, B->O, ..., N->A, O->B, ...)."""
    total = 0
    for c in str(text):
        if c.isalpha():
            idx = ord(c.lower()) - ord('a')  # 0-25
            rotated = (idx + 13) % 26         # rotate by 13
            total += rotated + 1              # 1-26
    return total


def gematria(text):
    """
    Returns ALL gematria values for any text string.

    >>> gematria("Bitcoin")
    {'ordinal': 72, 'reverse': 117, 'reduction': 36, 'english': 432,
     'jewish': 213, 'satanic': 317, 'dr_ordinal': 9, 'dr_reverse': 9,
     'dr_english': 9, 'dr_jewish': 6, 'dr_satanic': 2,
     'master_ordinal': None, 'master_reverse': None}
    """
    ord_val = ordinal(text)
    rev_val = reverse(text)
    red_val = reduction(text)
    eng_val = english(text)
    jew_val = jewish(text)
    sat_val = satanic(text)
    chal_val = chaldean(text)
    alb_val = albam(text)

    return {
        # Raw values
        'ordinal': ord_val,
        'reverse': rev_val,
        'reduction': red_val,
        'english': eng_val,
        'jewish': jew_val,
        'satanic': sat_val,
        'chaldean': chal_val,
        'albam': alb_val,
        # Digital roots of each
        'dr_ordinal': digital_root(ord_val),
        'dr_reverse': digital_root(rev_val),
        'dr_english': digital_root(eng_val),
        'dr_jewish': digital_root(jew_val),
        'dr_satanic': digital_root(sat_val),
        'dr_chaldean': digital_root(chal_val),
        'dr_albam': digital_root(alb_val),
        # Master number checks
        'master_ordinal': reduce_keep_master(ord_val) if reduce_keep_master(ord_val) in (11, 22, 33) else None,
        'master_reverse': reduce_keep_master(rev_val) if reduce_keep_master(rev_val) in (11, 22, 33) else None,
    }


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

# Key target numbers for caution/pump detection
CAUTION_TARGETS = {93, 39, 43, 48, 223, 322, 17, 113, 11, 19, 13, 22, 33, 44, 66, 77, 88, 99, 666, 911, 47}
PUMP_TARGETS = {27, 72, 127, 721, 37, 73}
BTC_ENERGY_TARGETS = {213, 231, 312, 321, 132, 123, 68}


def gematria_match(text1, text2):
    """
    Check if two texts share any gematria values.
    Returns dict of matching ciphers, or empty dict if no match.
    """
    g1 = gematria(text1)
    g2 = gematria(text2)
    matches = {}
    for cipher in ('ordinal', 'reverse', 'reduction', 'english', 'jewish', 'satanic', 'chaldean', 'albam'):
        if g1[cipher] == g2[cipher] and g1[cipher] > 0:
            matches[cipher] = g1[cipher]
    return matches


def gematria_contains_target(text, targets=None):
    """
    Check if any gematria value of text matches target numbers.
    Returns list of (cipher, value, target_set) matches.
    """
    if targets is None:
        targets = CAUTION_TARGETS | PUMP_TARGETS | BTC_ENERGY_TARGETS

    g = gematria(text)
    matches = []
    for cipher in ('ordinal', 'reverse', 'reduction', 'english', 'jewish', 'satanic', 'chaldean', 'albam'):
        val = g[cipher]
        if val in targets:
            if val in CAUTION_TARGETS:
                matches.append((cipher, val, 'caution'))
            elif val in PUMP_TARGETS:
                matches.append((cipher, val, 'pump'))
            elif val in BTC_ENERGY_TARGETS:
                matches.append((cipher, val, 'btc_energy'))
    return matches


def gematria_flat(text, prefix=''):
    """
    Returns flat dict suitable for ML features.
    If prefix='tweet_', returns {'tweet_gem_ordinal': 68, 'tweet_gem_reverse': 112, ...}
    """
    g = gematria(text)
    p = f"{prefix}gem_" if prefix else "gem_"
    return {
        f'{p}ordinal': g['ordinal'],
        f'{p}reverse': g['reverse'],
        f'{p}reduction': g['reduction'],
        f'{p}english': g['english'],
        f'{p}jewish': g['jewish'],
        f'{p}satanic': g['satanic'],
        f'{p}chaldean': g['chaldean'],
        f'{p}albam': g['albam'],
        f'{p}dr_ordinal': g['dr_ordinal'],
        f'{p}dr_reverse': g['dr_reverse'],
        f'{p}dr_chaldean': g['dr_chaldean'],
        f'{p}dr_albam': g['dr_albam'],
        f'{p}is_caution': 1 if any(g[c] in CAUTION_TARGETS for c in ('ordinal', 'reverse', 'english', 'jewish', 'satanic', 'chaldean', 'albam')) else 0,
        f'{p}is_pump': 1 if any(g[c] in PUMP_TARGETS for c in ('ordinal', 'reverse', 'english', 'jewish', 'satanic', 'chaldean', 'albam')) else 0,
        f'{p}is_btc_energy': 1 if any(g[c] in BTC_ENERGY_TARGETS for c in ('ordinal', 'reverse', 'english', 'jewish', 'satanic', 'chaldean', 'albam')) else 0,
    }


# ============================================================
# REFERENCE VALUES
# ============================================================

GEMATRIA_REFERENCE = {
    "Bitcoin": gematria("Bitcoin"),
    "BTC": gematria("BTC"),
    "crash": gematria("crash"),
    "pump": gematria("pump"),
    "dump": gematria("dump"),
    "gold": gematria("gold"),
    "death": gematria("death"),
    "moon": gematria("moon"),
    "bear": gematria("bear"),
    "bull": gematria("bull"),
}


# ============================================================
# GPU BATCH GEMATRIA (cuDF + CuPy — zero .apply())
# ============================================================

# Build 128-entry lookup tables (indexed by lowercase ASCII code 0-127)
# Non-alpha chars map to 0. Lowercase a=97..z=122.
def _build_cipher_lut():
    """Build lookup tables for all 8 ciphers as numpy arrays (128 entries each)."""
    ordinal_lut = np.zeros(128, dtype=np.int32)
    reverse_lut = np.zeros(128, dtype=np.int32)
    reduction_lut = np.zeros(128, dtype=np.int32)
    english_lut = np.zeros(128, dtype=np.int32)
    jewish_lut = np.zeros(128, dtype=np.int32)
    satanic_lut = np.zeros(128, dtype=np.int32)
    chaldean_lut = np.zeros(128, dtype=np.int32)
    albam_lut = np.zeros(128, dtype=np.int32)

    jewish_vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 600, 10, 20, 30,
                   40, 50, 60, 70, 80, 90, 100, 200, 700, 900, 300, 400, 500]

    # Chaldean values per letter a-z (9 is never assigned)
    chaldean_vals = [1, 2, 3, 4, 5, 8, 3, 5, 1, 1, 2, 3, 4,
                     5, 7, 8, 1, 2, 3, 4, 6, 6, 6, 5, 1, 7]

    for i in range(26):
        code = 97 + i  # lowercase a-z
        pos = i + 1    # 1-26
        ordinal_lut[code] = pos
        reverse_lut[code] = 27 - pos
        # Reduction: digital root of ordinal value
        val = pos
        while val > 9:
            val = sum(int(d) for d in str(val))
        reduction_lut[code] = val
        english_lut[code] = pos * 6
        jewish_lut[code] = jewish_vals[i]
        satanic_lut[code] = pos + 35
        chaldean_lut[code] = chaldean_vals[i]
        # AlBam: rotate by 13 positions, value = rotated position (1-26)
        albam_lut[code] = (i + 13) % 26 + 1

    return (ordinal_lut, reverse_lut, reduction_lut, english_lut,
            jewish_lut, satanic_lut, chaldean_lut, albam_lut)

(_ORDINAL_LUT, _REVERSE_LUT, _REDUCTION_LUT, _ENGLISH_LUT,
 _JEWISH_LUT, _SATANIC_LUT, _CHALDEAN_LUT, _ALBAM_LUT) = _build_cipher_lut()


def _digital_root_vec(arr):
    """Vectorized digital root on numpy array. 0→0, else 1+((n-1)%9)."""
    result = np.where(arr == 0, 0, 1 + (np.abs(arr) - 1) % 9)
    return result.astype(np.int32)


def gematria_gpu_batch(text_series, prefix='gem'):
    """
    GPU-accelerated batch gematria on a cuDF or pandas string Series.

    Returns a pandas DataFrame with columns:
        {prefix}_ordinal, {prefix}_reverse, {prefix}_reduction,
        {prefix}_english, {prefix}_jewish, {prefix}_satanic,
        {prefix}_chaldean, {prefix}_albam,
        {prefix}_dr_ordinal, {prefix}_dr_reverse,
        {prefix}_dr_chaldean, {prefix}_dr_albam,
        {prefix}_is_caution, {prefix}_is_pump, {prefix}_is_btc_energy

    Falls back to CPU vectorized path if cuDF is unavailable.
    """
    if os.environ.get('V2_SKIP_GPU') == '1':
        return _gematria_cpu_vectorized(text_series, prefix)
    try:
        import cudf
        import cupy as cp
        return _gematria_gpu_cudf(text_series, prefix)
    except (ImportError, Exception):
        return _gematria_cpu_vectorized(text_series, prefix)


def _gematria_gpu_cudf(text_series, prefix):
    """GPU path using cuDF string methods + CuPy lookups."""
    import cudf
    import cupy as cp

    # Ensure cuDF Series
    if not isinstance(text_series, cudf.Series):
        text_series = cudf.Series(text_series)

    n = len(text_series)

    # Fill nulls with empty string, lowercase
    s = text_series.fillna('').str.lower()

    # Get code points as a list column of ints
    code_lists = s.str.code_points()

    # Upload cipher LUTs to GPU
    ord_lut_gpu = cp.asarray(_ORDINAL_LUT)
    rev_lut_gpu = cp.asarray(_REVERSE_LUT)
    red_lut_gpu = cp.asarray(_REDUCTION_LUT)
    eng_lut_gpu = cp.asarray(_ENGLISH_LUT)
    jew_lut_gpu = cp.asarray(_JEWISH_LUT)
    sat_lut_gpu = cp.asarray(_SATANIC_LUT)
    chal_lut_gpu = cp.asarray(_CHALDEAN_LUT)
    alb_lut_gpu = cp.asarray(_ALBAM_LUT)

    # Flatten code_lists to a single 1D array + offsets for per-string aggregation
    # code_lists is a ListColumn — extract offsets and flat values
    flat_codes = code_lists.list.leaves.values  # CuPy array of all codepoints
    offsets = code_lists.list.offsets.values     # CuPy array of offsets

    # Clamp codes to valid LUT range (0-127), anything >= 128 maps to 0
    flat_codes_clamped = cp.where(flat_codes < 128, flat_codes, cp.int32(0))

    # Apply all 8 cipher LUTs via fancy indexing (GPU vectorized)
    ord_vals = ord_lut_gpu[flat_codes_clamped]
    rev_vals = rev_lut_gpu[flat_codes_clamped]
    red_vals = red_lut_gpu[flat_codes_clamped]
    eng_vals = eng_lut_gpu[flat_codes_clamped]
    jew_vals = jew_lut_gpu[flat_codes_clamped]
    sat_vals = sat_lut_gpu[flat_codes_clamped]
    chal_vals = chal_lut_gpu[flat_codes_clamped]
    alb_vals = alb_lut_gpu[flat_codes_clamped]

    # Sum per string using segment reduction via offsets
    def _segment_sum(values, offsets, n):
        """Sum values between consecutive offsets. Returns array of length n."""
        # cumsum approach: cumsum all values, then diff at offset boundaries
        cumsum = cp.cumsum(values)
        # Prepend 0 for the subtraction
        starts = offsets[:-1]
        ends = offsets[1:]
        # result[i] = cumsum[ends[i]-1] - cumsum[starts[i]-1]  (with starts[i]-1 = -1 → 0)
        result = cp.zeros(n, dtype=cp.int64)
        mask = ends > starts  # non-empty strings
        end_vals = cp.where(mask, cumsum[ends - 1], cp.int64(0))
        start_vals = cp.where(mask & (starts > 0), cumsum[starts - 1], cp.int64(0))
        result = end_vals - start_vals
        return result.astype(cp.int32)

    ord_sums = _segment_sum(ord_vals, offsets, n)
    rev_sums = _segment_sum(rev_vals, offsets, n)
    red_sums = _segment_sum(red_vals, offsets, n)
    eng_sums = _segment_sum(eng_vals, offsets, n)
    jew_sums = _segment_sum(jew_vals, offsets, n)
    sat_sums = _segment_sum(sat_vals, offsets, n)
    chal_sums = _segment_sum(chal_vals, offsets, n)
    alb_sums = _segment_sum(alb_vals, offsets, n)

    # Digital roots (GPU vectorized)
    def _dr_gpu(arr):
        return cp.where(arr == 0, cp.int32(0), 1 + (cp.abs(arr) - 1) % 9).astype(cp.int32)

    dr_ord = _dr_gpu(ord_sums)
    dr_rev = _dr_gpu(rev_sums)
    dr_chal = _dr_gpu(chal_sums)
    dr_alb = _dr_gpu(alb_sums)

    # Caution/pump/energy flags — check if ANY of 5 cipher values is in target set
    # (reduction excluded from target check, matching original gematria_flat behavior)
    caution_set = cp.asarray(sorted(CAUTION_TARGETS), dtype=cp.int32)
    pump_set = cp.asarray(sorted(PUMP_TARGETS), dtype=cp.int32)
    energy_set = cp.asarray(sorted(BTC_ENERGY_TARGETS), dtype=cp.int32)

    def _any_in_set(cipher_arrays, target_set):
        """Check if any cipher value for each row is in the target set."""
        result = cp.zeros(n, dtype=cp.int32)
        for arr in cipher_arrays:
            result |= cp.isin(arr, target_set).astype(cp.int32)
        return result

    check_ciphers = [ord_sums, rev_sums, eng_sums, jew_sums, sat_sums, chal_sums, alb_sums]
    is_caution = _any_in_set(check_ciphers, caution_set)
    is_pump = _any_in_set(check_ciphers, pump_set)
    is_energy = _any_in_set(check_ciphers, energy_set)

    # Build result as pandas DataFrame (transfer from GPU)
    import pandas as pd
    p = f'{prefix}_' if prefix else ''
    result = pd.DataFrame({
        f'{p}ordinal': cp.asnumpy(ord_sums),
        f'{p}reverse': cp.asnumpy(rev_sums),
        f'{p}reduction': cp.asnumpy(red_sums),
        f'{p}english': cp.asnumpy(eng_sums),
        f'{p}jewish': cp.asnumpy(jew_sums),
        f'{p}satanic': cp.asnumpy(sat_sums),
        f'{p}chaldean': cp.asnumpy(chal_sums),
        f'{p}albam': cp.asnumpy(alb_sums),
        f'{p}dr_ordinal': cp.asnumpy(dr_ord),
        f'{p}dr_reverse': cp.asnumpy(dr_rev),
        f'{p}dr_chaldean': cp.asnumpy(dr_chal),
        f'{p}dr_albam': cp.asnumpy(dr_alb),
        f'{p}is_caution': cp.asnumpy(is_caution),
        f'{p}is_pump': cp.asnumpy(is_pump),
        f'{p}is_btc_energy': cp.asnumpy(is_energy),
    }, index=text_series.index if hasattr(text_series, 'index') else None)

    # Free GPU memory
    cp.get_default_memory_pool().free_all_blocks()

    return result


def _gematria_cpu_vectorized(text_series, prefix):
    """CPU fallback using numpy vectorized lookups (still no .apply())."""
    import pandas as pd

    texts = text_series.fillna('').astype(str).values
    n = len(texts)

    ord_sums = np.zeros(n, dtype=np.int32)
    rev_sums = np.zeros(n, dtype=np.int32)
    red_sums = np.zeros(n, dtype=np.int32)
    eng_sums = np.zeros(n, dtype=np.int32)
    jew_sums = np.zeros(n, dtype=np.int32)
    sat_sums = np.zeros(n, dtype=np.int32)
    chal_sums = np.zeros(n, dtype=np.int32)
    alb_sums = np.zeros(n, dtype=np.int32)

    for i, text in enumerate(texts):
        codes = np.frombuffer(text.lower().encode('ascii', 'ignore'), dtype=np.uint8)
        codes = codes[codes < 128]
        if len(codes) > 0:
            ord_sums[i] = _ORDINAL_LUT[codes].sum()
            rev_sums[i] = _REVERSE_LUT[codes].sum()
            red_sums[i] = _REDUCTION_LUT[codes].sum()
            eng_sums[i] = _ENGLISH_LUT[codes].sum()
            jew_sums[i] = _JEWISH_LUT[codes].sum()
            sat_sums[i] = _SATANIC_LUT[codes].sum()
            chal_sums[i] = _CHALDEAN_LUT[codes].sum()
            alb_sums[i] = _ALBAM_LUT[codes].sum()

    dr_ord = _digital_root_vec(ord_sums)
    dr_rev = _digital_root_vec(rev_sums)
    dr_chal = _digital_root_vec(chal_sums)
    dr_alb = _digital_root_vec(alb_sums)

    check = np.column_stack([ord_sums, rev_sums, eng_sums, jew_sums, sat_sums, chal_sums, alb_sums])
    caution_arr = np.array(sorted(CAUTION_TARGETS))
    pump_arr = np.array(sorted(PUMP_TARGETS))
    energy_arr = np.array(sorted(BTC_ENERGY_TARGETS))

    is_caution = np.any(np.isin(check, caution_arr), axis=1).astype(np.int32)
    is_pump = np.any(np.isin(check, pump_arr), axis=1).astype(np.int32)
    is_energy = np.any(np.isin(check, energy_arr), axis=1).astype(np.int32)

    p = f'{prefix}_' if prefix else ''
    return pd.DataFrame({
        f'{p}ordinal': ord_sums,
        f'{p}reverse': rev_sums,
        f'{p}reduction': red_sums,
        f'{p}english': eng_sums,
        f'{p}jewish': jew_sums,
        f'{p}satanic': sat_sums,
        f'{p}chaldean': chal_sums,
        f'{p}albam': alb_sums,
        f'{p}dr_ordinal': dr_ord,
        f'{p}dr_reverse': dr_rev,
        f'{p}dr_chaldean': dr_chal,
        f'{p}dr_albam': dr_alb,
        f'{p}is_caution': is_caution,
        f'{p}is_pump': is_pump,
        f'{p}is_btc_energy': is_energy,
    }, index=text_series.index if hasattr(text_series, 'index') else None)


def digital_root_gpu(arr):
    """Vectorized digital root. Works with numpy or cupy arrays."""
    if os.environ.get('V2_SKIP_GPU') != '1':
        try:
            import cupy as cp
            if isinstance(arr, cp.ndarray):
                x = cp.abs(arr).astype(cp.int64)
                return cp.where(x == 0, cp.int32(0), (1 + (x - 1) % 9).astype(cp.int32))
        except ImportError:
            pass
    x = np.abs(np.asarray(arr)).astype(np.int64)
    return np.where(x == 0, 0, 1 + (x - 1) % 9).astype(np.int32)


if __name__ == "__main__":
    # Quick test — CPU path
    tests = ["Bitcoin", "BTC", "Elon Musk", "Steve Harvey", "crash", "pump"]
    for t in tests:
        g = gematria(t)
        targets = gematria_contains_target(t)
        print(f'"{t}": ord={g["ordinal"]} rev={g["reverse"]} red={g["reduction"]} '
              f'eng={g["english"]} jew={g["jewish"]} sat={g["satanic"]} '
              f'chal={g["chaldean"]} alb={g["albam"]} '
              f'dr={g["dr_ordinal"]}' + (f' TARGETS: {targets}' if targets else ''))

    # Test GPU batch (falls back to CPU vectorized if no GPU)
    print("\n=== GPU Batch Test ===")
    import pandas as pd
    test_series = pd.Series(tests)
    result = gematria_gpu_batch(test_series, prefix='test_gem')
    print(result.to_string())
