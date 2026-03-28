#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
universal_sentiment.py — Universal Sentiment Engine
=====================================================
Single source of truth for ALL text sentiment analysis.
Takes any text string, returns sentiment score + metadata.

Usage:
  from universal_sentiment import sentiment, sentiment_flat
  result = sentiment("Bitcoin crashes to new low!")
"""

import os
import re


# ============================================================
# WORD LISTS (consolidated from news_collector.py + others)
# ============================================================

BULLISH_WORDS = {
    'bull', 'bullish', 'moon', 'pump', 'rally', 'surge', 'soar', 'gain',
    'rise', 'rising', 'breakout', 'ath', 'record', 'high', 'buy', 'long',
    'boom', 'explode', 'skyrocket', 'green', 'recover', 'recovery', 'up',
    'upgrade', 'adoption', 'approve', 'approved', 'approval', 'etf',
    'institutional', 'accumulate', 'hodl', 'hold', 'support', 'uptrend',
    'optimistic', 'positive', 'strong', 'strength', 'outperform', 'profit',
    'inflow', 'demand', 'milestone', 'parabolic', 'launch', 'partnership',
    'integration', 'mainstream', 'halving', 'scarcity', 'golden', 'breakthrough',
}

BEARISH_WORDS = {
    'bear', 'bearish', 'crash', 'dump', 'plunge', 'collapse', 'fear', 'sell',
    'selloff', 'short', 'drop', 'decline', 'fall', 'falling', 'tank', 'red',
    'panic', 'capitulate', 'liquidat', 'ban', 'hack', 'scam', 'fraud',
    'regulate', 'crackdown', 'warning', 'risk', 'bubble', 'overvalued',
    'reject', 'rejection', 'resistance', 'downtrend', 'weak', 'loss',
    'correction', 'negative', 'trouble', 'bankrupt', 'insolvency', 'default',
    'recession', 'inflation', 'outflow', 'sec', 'lawsuit', 'investigation',
    'exploit', 'vulnerability', 'death', 'dead', 'worthless', 'ponzi',
}

URGENCY_WORDS = {
    'breaking', 'urgent', 'alert', 'emergency', 'flash', 'just in',
    'developing', 'live', 'now', 'imminent', 'critical', 'warning',
}


# ============================================================
# CORE FUNCTIONS
# ============================================================

def sentiment(text):
    """
    Returns ALL sentiment analysis for any text.

    >>> sentiment("Bitcoin crashes to new low! SELL NOW!!!")
    {'score': -2, 'bull_count': 0, 'bear_count': 2, 'has_caps': True,
     'caps_words': 2, 'exclamation': 3, 'urgency': 1, 'word_count': 7, ...}
    """
    if not text:
        return _empty_sentiment()

    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)

    # Count bull/bear words
    bull_count = 0
    bear_count = 0
    bull_found = []
    bear_found = []

    for word in words:
        if word in BULLISH_WORDS:
            bull_count += 1
            bull_found.append(word)
        # Use startswith for partial matches (e.g., "liquidat" matches "liquidation")
        for bw in BEARISH_WORDS:
            if word.startswith(bw) or word == bw:
                bear_count += 1
                bear_found.append(word)
                break

    # Caps detection
    caps_words = len([w for w in text.split() if w.isupper() and len(w) > 1])
    has_caps = caps_words > 0

    # Exclamation
    exclamation = text.count('!')
    question = text.count('?')

    # Urgency
    urgency_count = sum(1 for uw in URGENCY_WORDS if uw in text_lower)

    # Word count
    word_count = len(words)

    # Score
    score = bull_count - bear_count

    return {
        'score': score,
        'bull_count': bull_count,
        'bear_count': bear_count,
        'bull_words': bull_found,
        'bear_words': bear_found,
        'has_caps': has_caps,
        'caps_words': caps_words,
        'exclamation': exclamation,
        'question': question,
        'urgency': urgency_count,
        'word_count': word_count,
        'is_bullish': score > 0,
        'is_bearish': score < 0,
        'is_neutral': score == 0,
    }


def _empty_sentiment():
    return {
        'score': 0, 'bull_count': 0, 'bear_count': 0,
        'bull_words': [], 'bear_words': [],
        'has_caps': False, 'caps_words': 0,
        'exclamation': 0, 'question': 0, 'urgency': 0,
        'word_count': 0, 'is_bullish': False, 'is_bearish': False, 'is_neutral': True,
    }


def sentiment_flat(text, prefix=''):
    """
    Returns flat dict suitable for ML features.
    If prefix='tweet_', returns {'tweet_sentiment': -2, 'tweet_has_caps': 1, ...}
    """
    s = sentiment(text)
    p = f"{prefix}_" if prefix else ""
    return {
        f'{p}sentiment': s['score'],
        f'{p}bull_words': s['bull_count'],
        f'{p}bear_words': s['bear_count'],
        f'{p}has_caps': int(s['has_caps']),
        f'{p}caps_words': s['caps_words'],
        f'{p}exclamation': s['exclamation'],
        f'{p}urgency': s['urgency'],
        f'{p}word_count': s['word_count'],
    }


# ============================================================
# GPU BATCH SENTIMENT (cuDF string ops — zero .apply())
# ============================================================

# Build expanded bearish lexicon for exact matching (prefix expansion)
# Original uses startswith("liquidat") to match liquidation/liquidated/etc.
# For GPU exact join, we expand prefixes to common variants.
_BEARISH_EXPANDED = set(BEARISH_WORDS)
_BEARISH_PREFIXES = {'liquidat'}  # words that use prefix matching
_BEARISH_SUFFIX_VARIANTS = ['', 'e', 'ed', 'es', 'ing', 'ion', 'ions', 'or', 'ors']
for prefix in _BEARISH_PREFIXES:
    for suffix in _BEARISH_SUFFIX_VARIANTS:
        _BEARISH_EXPANDED.add(prefix + suffix)


def sentiment_gpu_batch(text_series, prefix='sent'):
    """
    GPU-accelerated batch sentiment on a cuDF or pandas string Series.

    Returns a pandas DataFrame with columns:
        {prefix}_score, {prefix}_bull_count, {prefix}_bear_count,
        {prefix}_has_caps, {prefix}_caps_words,
        {prefix}_exclamation, {prefix}_urgency, {prefix}_word_count

    Falls back to CPU vectorized path if cuDF unavailable.
    """
    if os.environ.get('V2_SKIP_GPU') == '1':
        if os.environ.get('ALLOW_CPU', '0') != '1':
            raise RuntimeError("GPU REQUIRED: V2_SKIP_GPU=1 but ALLOW_CPU not set in universal_sentiment. Set ALLOW_CPU=1 for CPU mode.")
        return _sentiment_cpu_vectorized(text_series, prefix)
    try:
        import cudf
        import cupy as cp
        return _sentiment_gpu_cudf(text_series, prefix)
    except (ImportError, Exception):
        if os.environ.get('ALLOW_CPU', '0') != '1':
            raise RuntimeError("GPU REQUIRED: cuDF/CuPy unavailable in universal_sentiment. Set ALLOW_CPU=1 for CPU mode.")
        return _sentiment_cpu_vectorized(text_series, prefix)


def _sentiment_gpu_cudf(text_series, prefix):
    """GPU path using cuDF string methods."""
    import cudf
    import cupy as cp
    import pandas as pd

    if not isinstance(text_series, cudf.Series):
        text_series = cudf.Series(text_series)

    n = len(text_series)
    s = text_series.fillna('')

    # --- Caps detection (on original text, before lowering) ---
    # Count uppercase words: split, check each token for all-upper + len > 1
    # cuDF approach: count uppercase chars vs total chars
    upper_count = s.str.count(r'[A-Z]')
    total_len = s.str.len()
    # Caps words: approximate by counting fully uppercase tokens
    # Split into words, check which are all caps with len > 1
    # Simpler GPU approach: count occurrences of pattern \b[A-Z]{2,}\b
    caps_words = s.str.count(r'\b[A-Z]{2,}\b')
    has_caps = (caps_words > 0).astype('int32')

    # --- Exclamation and question marks ---
    exclamation = s.str.count(r'!')
    question = s.str.count(r'\?')

    # --- Lowercase for word matching ---
    s_lower = s.str.lower()

    # --- Word count ---
    word_count = s_lower.str.count(r'\b\w+\b')

    # --- Urgency: count how many urgency phrases appear ---
    urgency = cudf.Series(cp.zeros(n, dtype=cp.int32))
    for uw in URGENCY_WORDS:
        urgency = urgency + s_lower.str.contains(uw, regex=False).astype('int32')

    # --- Bull/bear word counting via tokenize + join ---
    # Tokenize: split into words, explode
    tokens = s_lower.str.findall(r'\b\w+\b')
    # Add row index before exploding
    token_df = cudf.DataFrame({'_idx': cudf.Series(cp.arange(n, dtype=cp.int32)), '_tokens': tokens})
    exploded = token_df.explode('_tokens').rename(columns={'_tokens': '_word'})
    exploded = exploded.dropna(subset=['_word'])

    if len(exploded) > 0:
        # Build bull/bear lexicon DataFrames
        bull_lex = cudf.DataFrame({'_word': cudf.Series(list(BULLISH_WORDS)), '_bull': cudf.Series([1] * len(BULLISH_WORDS), dtype='int32')})
        bear_lex = cudf.DataFrame({'_word': cudf.Series(list(_BEARISH_EXPANDED)), '_bear': cudf.Series([1] * len(_BEARISH_EXPANDED), dtype='int32')})

        # Join tokens with lexicons (exact match)
        merged_bull = exploded.merge(bull_lex, on='_word', how='left')
        merged_bear = exploded.merge(bear_lex, on='_word', how='left')

        # Aggregate per original row
        bull_counts = merged_bull.groupby('_idx')['_bull'].sum().reset_index()
        bear_counts = merged_bear.groupby('_idx')['_bear'].sum().reset_index()

        # Map back to full index
        bull_result = cp.zeros(n, dtype=cp.int32)
        bear_result = cp.zeros(n, dtype=cp.int32)

        if len(bull_counts) > 0:
            bc_idx = bull_counts['_idx'].values
            bc_vals = bull_counts['_bull'].values.astype(cp.int32)
            bull_result[bc_idx] = bc_vals

        if len(bear_counts) > 0:
            brc_idx = bear_counts['_idx'].values
            brc_vals = bear_counts['_bear'].values.astype(cp.int32)
            bear_result[brc_idx] = brc_vals
    else:
        bull_result = cp.zeros(n, dtype=cp.int32)
        bear_result = cp.zeros(n, dtype=cp.int32)

    score = bull_result - bear_result

    # Build result
    p = f'{prefix}_' if prefix else ''
    result = pd.DataFrame({
        f'{p}score': cp.asnumpy(score),
        f'{p}bull_count': cp.asnumpy(bull_result),
        f'{p}bear_count': cp.asnumpy(bear_result),
        f'{p}has_caps': cp.asnumpy(has_caps.values),
        f'{p}caps_words': cp.asnumpy(caps_words.values),
        f'{p}exclamation': cp.asnumpy(exclamation.values),
        f'{p}urgency': cp.asnumpy(urgency.values),
        f'{p}word_count': cp.asnumpy(word_count.values),
    }, index=text_series.index if hasattr(text_series, 'index') else None)

    cp.get_default_memory_pool().free_all_blocks()
    return result


def _sentiment_cpu_vectorized(text_series, prefix):
    """CPU fallback — still faster than .apply() since it avoids per-row function calls."""
    import pandas as pd
    import numpy as np

    texts = text_series.fillna('').astype(str).values
    n = len(texts)

    scores = np.zeros(n, dtype=np.int32)
    bulls = np.zeros(n, dtype=np.int32)
    bears = np.zeros(n, dtype=np.int32)
    has_caps_arr = np.zeros(n, dtype=np.int32)
    caps_words_arr = np.zeros(n, dtype=np.int32)
    excl_arr = np.zeros(n, dtype=np.int32)
    urgency_arr = np.zeros(n, dtype=np.int32)
    word_count_arr = np.zeros(n, dtype=np.int32)

    for i, text in enumerate(texts):
        s = sentiment(text)
        scores[i] = s['score']
        bulls[i] = s['bull_count']
        bears[i] = s['bear_count']
        has_caps_arr[i] = int(s['has_caps'])
        caps_words_arr[i] = s['caps_words']
        excl_arr[i] = s['exclamation']
        urgency_arr[i] = s['urgency']
        word_count_arr[i] = s['word_count']

    p = f'{prefix}_' if prefix else ''
    return pd.DataFrame({
        f'{p}score': scores,
        f'{p}bull_count': bulls,
        f'{p}bear_count': bears,
        f'{p}has_caps': has_caps_arr,
        f'{p}caps_words': caps_words_arr,
        f'{p}exclamation': excl_arr,
        f'{p}urgency': urgency_arr,
        f'{p}word_count': word_count_arr,
    }, index=text_series.index if hasattr(text_series, 'index') else None)


if __name__ == "__main__":
    tests = [
        "Bitcoin crashes to new low! SELL NOW!!!",
        "BTC hits new ATH! Moon mission confirmed!",
        "Federal Reserve announces rate decision",
        "BREAKING: SEC approves Bitcoin ETF application",
        "Market looks weak, bearish rejection at resistance",
    ]
    for t in tests:
        s = sentiment(t)
        print(f'"{t[:60]}..."')
        print(f'  Score={s["score"]} bull={s["bull_count"]} bear={s["bear_count"]} '
              f'caps={s["caps_words"]} !!={s["exclamation"]} urgency={s["urgency"]}')

    # Test GPU batch (falls back to CPU vectorized)
    print("\n=== GPU Batch Test ===")
    import pandas as pd
    test_series = pd.Series(tests)
    result = sentiment_gpu_batch(test_series, prefix='sent')
    print(result.to_string())
