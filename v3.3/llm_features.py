#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
llm_features.py -- LLM-Powered Sentiment & Risk Features (Anthropic SDK)
==========================================================================
Uses Claude Haiku for bulk sentiment analysis (cheap, fast) and
Claude Sonnet for pre-trade risk gating (smarter, slower).

Integrates with feature_library.py via compute_llm_features().
All results cached in SQLite to avoid redundant API calls.
Full error handling, exponential backoff, cost tracking.

Models:
  - claude-haiku-4-5-20251001  (sentiment / feature building)
  - claude-sonnet-4-6          (risk management / pre-trade gate)
"""

import hashlib
import json
import logging
import os
import re
import sqlite3
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger('llm_features')
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    ))
    logger.addHandler(_handler)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HAIKU_MODEL = 'claude-haiku-4-5-20251001'
SONNET_MODEL = 'claude-sonnet-4-6'

MAX_RETRIES = 3
BACKOFF_BASE = 2.0  # seconds
BACKOFF_MAX = 30.0

# Cost per 1M tokens (USD) -- approximate as of 2026-03
COST_PER_1M = {
    HAIKU_MODEL:  {'input': 0.80, 'output': 4.00},
    SONNET_MODEL: {'input': 3.00, 'output': 15.00},
}

RISK_CACHE_TTL_SECONDS = 300  # 5 min TTL for risk check cache

# Batch size for multi-text sentiment calls
SENTIMENT_BATCH_SIZE = 10

# ---------------------------------------------------------------------------
# SDK initialisation (lazy)
# ---------------------------------------------------------------------------
_client = None


def _get_client():
    """Lazily initialise the Anthropic client. Checks .env then env vars."""
    global _client
    if _client is not None:
        return _client

    try:
        import anthropic
    except ImportError:
        logger.error('anthropic package not installed. Run: pip install anthropic')
        raise RuntimeError('anthropic package not installed')

    api_key = os.environ.get('ANTHROPIC_API_KEY', '')

    # Try loading from .env if not in env
    if not api_key:
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
        if os.path.exists(env_path):
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('ANTHROPIC_API_KEY='):
                        api_key = line.split('=', 1)[1].strip().strip('"').strip("'")
                        break

    if not api_key:
        logger.warning(
            'No ANTHROPIC_API_KEY found. LLM features will use regex fallback.'
        )
        return None

    _client = anthropic.Anthropic(api_key=api_key)
    logger.info('Anthropic client initialised (key ending ...%s)', api_key[-4:])
    return _client


# ============================================================
# SQLITE CACHE
# ============================================================

def _init_cache_db(cache_db):
    """Create cache tables if they don't exist."""
    db_path = cache_db
    if not os.path.isabs(db_path):
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), cache_db)

    conn = sqlite3.connect(db_path, timeout=10)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS sentiment_cache (
            text_hash TEXT PRIMARY KEY,
            text_preview TEXT,
            result_json TEXT,
            model TEXT,
            input_tokens INTEGER,
            output_tokens INTEGER,
            created_at TEXT
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS risk_cache (
            context_hash TEXT PRIMARY KEY,
            result_json TEXT,
            model TEXT,
            input_tokens INTEGER,
            output_tokens INTEGER,
            created_at TEXT,
            expires_at TEXT
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS cost_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            model TEXT,
            call_type TEXT,
            input_tokens INTEGER,
            output_tokens INTEGER,
            cost_usd REAL
        )
    ''')
    conn.commit()
    return conn


def _text_hash(text):
    """SHA-256 hash of normalised text."""
    normalised = text.strip().lower()
    return hashlib.sha256(normalised.encode('utf-8', errors='replace')).hexdigest()


def _context_hash(context_dict):
    """Hash of trade context for risk cache."""
    # Deterministic JSON serialisation
    serialised = json.dumps(context_dict, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode('utf-8')).hexdigest()


def _get_cached_sentiment(conn, text_hash):
    """Fetch cached sentiment result or None."""
    row = conn.execute(
        'SELECT result_json FROM sentiment_cache WHERE text_hash = ?',
        (text_hash,)
    ).fetchone()
    if row:
        return json.loads(row[0])
    return None


def _set_cached_sentiment(conn, text_hash, text, result, model, inp_tok, out_tok):
    """Store sentiment result in cache."""
    preview = text[:200] if text else ''
    conn.execute(
        '''INSERT OR REPLACE INTO sentiment_cache
           (text_hash, text_preview, result_json, model, input_tokens, output_tokens, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)''',
        (text_hash, preview, json.dumps(result), model, inp_tok, out_tok,
         datetime.utcnow().isoformat())
    )
    conn.commit()


def _get_cached_risk(conn, ctx_hash):
    """Fetch cached risk result if not expired."""
    row = conn.execute(
        'SELECT result_json, expires_at FROM risk_cache WHERE context_hash = ?',
        (ctx_hash,)
    ).fetchone()
    if row:
        expires = datetime.fromisoformat(row[1])
        if datetime.utcnow() < expires:
            return json.loads(row[0])
    return None


def _set_cached_risk(conn, ctx_hash, result, model, inp_tok, out_tok):
    """Store risk result in cache with TTL."""
    now = datetime.utcnow()
    expires = now + timedelta(seconds=RISK_CACHE_TTL_SECONDS)
    conn.execute(
        '''INSERT OR REPLACE INTO risk_cache
           (context_hash, result_json, model, input_tokens, output_tokens, created_at, expires_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)''',
        (ctx_hash, json.dumps(result), model, inp_tok, out_tok,
         now.isoformat(), expires.isoformat())
    )
    conn.commit()


def _log_cost(conn, model, call_type, inp_tok, out_tok):
    """Log API call cost for tracking."""
    rates = COST_PER_1M.get(model, {'input': 1.0, 'output': 5.0})
    cost = (inp_tok * rates['input'] + out_tok * rates['output']) / 1_000_000
    conn.execute(
        '''INSERT INTO cost_log (timestamp, model, call_type, input_tokens, output_tokens, cost_usd)
           VALUES (?, ?, ?, ?, ?, ?)''',
        (datetime.utcnow().isoformat(), model, call_type, inp_tok, out_tok, cost)
    )
    conn.commit()
    logger.debug(
        'API cost: model=%s type=%s in=%d out=%d cost=$%.6f',
        model, call_type, inp_tok, out_tok, cost,
    )


# ============================================================
# COST REPORTING
# ============================================================

def get_cost_summary(cache_db='llm_cache.db', days=None):
    """
    Get cost summary from the cost log.

    Args:
        cache_db: path to cache database
        days: if set, only show last N days. None = all time.

    Returns:
        dict with total_cost, by_model, by_type, daily breakdown
    """
    conn = _init_cache_db(cache_db)
    try:
        where = ''
        params = ()
        if days is not None:
            cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
            where = ' WHERE timestamp >= ?'
            params = (cutoff,)

        rows = conn.execute(
            f'''SELECT model, call_type,
                       SUM(input_tokens) as total_in,
                       SUM(output_tokens) as total_out,
                       SUM(cost_usd) as total_cost,
                       COUNT(*) as calls
                FROM cost_log{where}
                GROUP BY model, call_type''',
            params
        ).fetchall()

        summary = {
            'total_cost': 0.0,
            'total_calls': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'by_model': {},
            'by_type': {},
        }

        for model, call_type, total_in, total_out, total_cost, calls in rows:
            summary['total_cost'] += total_cost
            summary['total_calls'] += calls
            summary['total_input_tokens'] += total_in
            summary['total_output_tokens'] += total_out

            if model not in summary['by_model']:
                summary['by_model'][model] = {'cost': 0.0, 'calls': 0}
            summary['by_model'][model]['cost'] += total_cost
            summary['by_model'][model]['calls'] += calls

            if call_type not in summary['by_type']:
                summary['by_type'][call_type] = {'cost': 0.0, 'calls': 0}
            summary['by_type'][call_type]['cost'] += total_cost
            summary['by_type'][call_type]['calls'] += calls

        # Daily breakdown (last 7 days)
        daily_rows = conn.execute(
            '''SELECT DATE(timestamp) as day,
                      SUM(cost_usd) as cost,
                      COUNT(*) as calls
               FROM cost_log
               WHERE timestamp >= ?
               GROUP BY DATE(timestamp)
               ORDER BY day DESC
               LIMIT 7''',
            ((datetime.utcnow() - timedelta(days=7)).isoformat(),)
        ).fetchall()

        summary['daily'] = [
            {'date': r[0], 'cost': r[1], 'calls': r[2]}
            for r in daily_rows
        ]

        return summary
    finally:
        conn.close()


# ============================================================
# REGEX FALLBACK SENTIMENT (no API needed)
# ============================================================

# Reuse word lists from universal_sentiment
_BULL_WORDS = {
    'bull', 'bullish', 'moon', 'pump', 'rally', 'surge', 'soar', 'gain',
    'rise', 'rising', 'breakout', 'ath', 'record', 'high', 'buy', 'long',
    'boom', 'explode', 'skyrocket', 'green', 'recover', 'recovery',
    'upgrade', 'adoption', 'approve', 'approved', 'approval', 'etf',
    'institutional', 'accumulate', 'hodl', 'support', 'uptrend',
    'optimistic', 'positive', 'strong', 'outperform', 'profit',
    'inflow', 'demand', 'milestone', 'parabolic', 'launch', 'partnership',
}

_BEAR_WORDS = {
    'bear', 'bearish', 'crash', 'dump', 'plunge', 'collapse', 'fear', 'sell',
    'selloff', 'short', 'drop', 'decline', 'fall', 'falling', 'tank', 'red',
    'panic', 'capitulate', 'liquidat', 'ban', 'hack', 'scam', 'fraud',
    'regulate', 'crackdown', 'warning', 'risk', 'bubble', 'overvalued',
    'reject', 'rejection', 'resistance', 'downtrend', 'weak', 'loss',
    'correction', 'negative', 'bankrupt', 'insolvency', 'default',
    'recession', 'inflation', 'outflow', 'sec', 'lawsuit', 'investigation',
    'exploit', 'vulnerability', 'death', 'dead', 'worthless', 'ponzi',
}

_SARCASM_MARKERS = {
    'lol', 'lmao', 'sure', 'totally', 'definitely', 'obviously',
    '/s', 'wow', 'amazing', 'great job', 'nice one',
}

_BTC_KEYWORDS = {
    'bitcoin', 'btc', 'crypto', 'blockchain', 'satoshi', 'halving',
    'mining', 'hash', 'mempool', 'lightning', 'layer 2', 'defi',
    'exchange', 'binance', 'coinbase', 'bitget', 'kraken',
}


def _regex_fallback_sentiment(text):
    """Fallback sentiment using regex word matching. No API needed."""
    if not text:
        return {
            'llm_sentiment': 0.0,
            'llm_sarcastic': False,
            'llm_urgent': False,
            'llm_context_score': 0.0,
            'llm_market_impact': 0.0,
        }

    text_lower = text.lower()
    words = set(re.findall(r'\b\w+\b', text_lower))

    bull = sum(1 for w in words if w in _BULL_WORDS)
    bear = sum(1 for w in words if w in _BEAR_WORDS)
    total = bull + bear

    if total > 0:
        sentiment_score = (bull - bear) / max(total, 1)
    else:
        sentiment_score = 0.0
    sentiment_score = max(-1.0, min(1.0, sentiment_score))

    sarcastic = any(m in text_lower for m in _SARCASM_MARKERS)
    urgent = any(u in text_lower for u in [
        'breaking', 'urgent', 'alert', 'emergency', 'flash',
        'just in', 'developing', 'critical',
    ])

    btc_hits = sum(1 for k in _BTC_KEYWORDS if k in text_lower)
    context_score = min(1.0, btc_hits / 3.0)

    market_impact = sentiment_score * (0.5 + 0.5 * context_score)

    return {
        'llm_sentiment': round(sentiment_score, 4),
        'llm_sarcastic': sarcastic,
        'llm_urgent': urgent,
        'llm_context_score': round(context_score, 4),
        'llm_market_impact': round(market_impact, 4),
    }


# ============================================================
# LLM SENTIMENT (Haiku -- for feature building)
# ============================================================

_SENTIMENT_SYSTEM_PROMPT = (
    'You are a financial sentiment analyzer specializing in Bitcoin and cryptocurrency markets. '
    'Analyze the given text(s) and return a JSON object for EACH text with these exact fields:\n'
    '- sentiment: float from -1.0 (extremely bearish) to 1.0 (extremely bullish), 0.0 = neutral\n'
    '- sarcastic: boolean, true if the text uses sarcasm or irony\n'
    '- urgent: boolean, true if the text conveys urgency or breaking news\n'
    '- context_score: float from 0.0 to 1.0, how relevant this text is to Bitcoin/crypto markets\n'
    '- market_impact: float from -1.0 to 1.0, expected short-term market impact direction\n\n'
    'Return ONLY valid JSON. For a single text, return one object. '
    'For multiple texts, return a JSON array of objects in the same order as the input texts.\n'
    'Do NOT include any explanation or commentary. JSON only.'
)


def _parse_llm_sentiment_response(response_text, count=1):
    """Parse the JSON response from Claude. Returns list of dicts."""
    # Strip markdown code fences if present
    cleaned = response_text.strip()
    if cleaned.startswith('```'):
        lines = cleaned.split('\n')
        # Remove first and last lines (fences)
        lines = [l for l in lines if not l.strip().startswith('```')]
        cleaned = '\n'.join(lines).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning('Failed to parse LLM sentiment JSON, using regex fallback')
        return None

    if isinstance(parsed, dict):
        parsed = [parsed]

    results = []
    for item in parsed:
        results.append({
            'llm_sentiment': float(item.get('sentiment', 0.0)),
            'llm_sarcastic': bool(item.get('sarcastic', False)),
            'llm_urgent': bool(item.get('urgent', False)),
            'llm_context_score': float(item.get('context_score', 0.0)),
            'llm_market_impact': float(item.get('market_impact', 0.0)),
        })

    return results if len(results) >= count else None


def llm_sentiment(text, cache_db='llm_cache.db'):
    """
    Get LLM-analyzed sentiment from Claude Haiku.

    Returns:
        dict with keys:
            llm_sentiment: float (-1 to 1)
            llm_sarcastic: bool
            llm_urgent: bool
            llm_context_score: float (0-1, how relevant to BTC)
            llm_market_impact: float (-1 to 1)

    Caches results in SQLite to avoid re-calling for same text.
    Falls back to regex sentiment if API unavailable.
    """
    if not text or not text.strip():
        return _regex_fallback_sentiment('')

    conn = _init_cache_db(cache_db)
    try:
        t_hash = _text_hash(text)

        # Check cache first
        cached = _get_cached_sentiment(conn, t_hash)
        if cached is not None:
            logger.debug('Sentiment cache hit: %s...', text[:50])
            return cached

        # Try API
        client = _get_client()
        if client is None:
            result = _regex_fallback_sentiment(text)
            _set_cached_sentiment(conn, t_hash, text, result, 'regex', 0, 0)
            return result

        result = _call_sentiment_api(client, [text], conn)
        if result and len(result) > 0:
            _set_cached_sentiment(
                conn, t_hash, text, result[0],
                HAIKU_MODEL, result[0].get('_inp_tok', 0), result[0].get('_out_tok', 0),
            )
            # Remove internal token tracking keys
            result[0].pop('_inp_tok', None)
            result[0].pop('_out_tok', None)
            return result[0]

        # Fallback
        fallback = _regex_fallback_sentiment(text)
        _set_cached_sentiment(conn, t_hash, text, fallback, 'regex_fallback', 0, 0)
        return fallback
    finally:
        conn.close()


def llm_sentiment_batch(texts, cache_db='llm_cache.db'):
    """
    Batch sentiment analysis. Checks cache for each text first,
    then sends uncached texts to API in batches.

    Args:
        texts: list of strings

    Returns:
        list of sentiment dicts (same order as input)
    """
    if not texts:
        return []

    conn = _init_cache_db(cache_db)
    try:
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        # Check cache for each text
        for i, text in enumerate(texts):
            if not text or not text.strip():
                results[i] = _regex_fallback_sentiment('')
                continue

            t_hash = _text_hash(text)
            cached = _get_cached_sentiment(conn, t_hash)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if not uncached_texts:
            logger.info('Batch sentiment: all %d texts cached', len(texts))
            return results

        logger.info(
            'Batch sentiment: %d cached, %d to process',
            len(texts) - len(uncached_texts), len(uncached_texts),
        )

        # Process uncached in batches
        client = _get_client()

        for batch_start in range(0, len(uncached_texts), SENTIMENT_BATCH_SIZE):
            batch_end = min(batch_start + SENTIMENT_BATCH_SIZE, len(uncached_texts))
            batch_texts = uncached_texts[batch_start:batch_end]
            batch_indices = uncached_indices[batch_start:batch_end]

            if client is not None:
                batch_results = _call_sentiment_api(client, batch_texts, conn)
            else:
                batch_results = None

            for j, idx in enumerate(batch_indices):
                text = batch_texts[j]
                t_hash = _text_hash(text)

                if batch_results and j < len(batch_results):
                    r = batch_results[j]
                    r.pop('_inp_tok', None)
                    r.pop('_out_tok', None)
                    results[idx] = r
                    _set_cached_sentiment(conn, t_hash, text, r, HAIKU_MODEL, 0, 0)
                else:
                    fallback = _regex_fallback_sentiment(text)
                    results[idx] = fallback
                    _set_cached_sentiment(conn, t_hash, text, fallback, 'regex_fallback', 0, 0)

        return results
    finally:
        conn.close()


def _call_sentiment_api(client, texts, conn):
    """
    Call Claude Haiku API for sentiment analysis with retries.

    Args:
        client: Anthropic client
        texts: list of text strings
        conn: SQLite connection for cost logging

    Returns:
        list of sentiment dicts, or None on failure
    """
    if len(texts) == 1:
        user_msg = texts[0]
    else:
        # Number each text for batch processing
        numbered = []
        for i, t in enumerate(texts, 1):
            # Truncate very long texts
            truncated = t[:1000] if len(t) > 1000 else t
            numbered.append(f'[{i}] {truncated}')
        user_msg = '\n\n'.join(numbered)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=HAIKU_MODEL,
                max_tokens=1024,
                system=_SENTIMENT_SYSTEM_PROMPT,
                messages=[{'role': 'user', 'content': user_msg}],
            )

            inp_tok = response.usage.input_tokens
            out_tok = response.usage.output_tokens
            _log_cost(conn, HAIKU_MODEL, 'sentiment', inp_tok, out_tok)

            response_text = response.content[0].text
            parsed = _parse_llm_sentiment_response(response_text, count=len(texts))

            if parsed:
                # Attach token counts for cache logging
                for p in parsed:
                    p['_inp_tok'] = inp_tok // len(parsed)
                    p['_out_tok'] = out_tok // len(parsed)
                return parsed

            logger.warning(
                'Attempt %d: failed to parse sentiment response', attempt + 1
            )

        except Exception as e:
            err_name = type(e).__name__
            logger.warning(
                'Attempt %d: sentiment API error (%s): %s',
                attempt + 1, err_name, str(e)[:200],
            )

            # Check for rate limit
            if 'rate' in str(e).lower() or '429' in str(e):
                wait = min(BACKOFF_MAX, BACKOFF_BASE ** (attempt + 2))
                logger.info('Rate limited, waiting %.1fs', wait)
                time.sleep(wait)
            elif attempt < MAX_RETRIES - 1:
                wait = min(BACKOFF_MAX, BACKOFF_BASE ** (attempt + 1))
                time.sleep(wait)

    logger.error('All %d sentiment API attempts failed, using regex fallback', MAX_RETRIES)
    return None


# ============================================================
# LLM RISK MANAGER (Sonnet -- for live trading)
# ============================================================

_RISK_SYSTEM_PROMPT = (
    'You are a Bitcoin trade risk manager. Given the current market context, '
    'you must decide whether a proposed trade should proceed.\n\n'
    'Respond with EXACTLY one JSON object containing:\n'
    '- decision: one of "OK", "CAUTION", or "BLOCK"\n'
    '- reason: brief explanation (1-2 sentences)\n'
    '- risk_score: float 0.0 (no risk) to 1.0 (extreme risk)\n\n'
    'Decision criteria:\n'
    '- BLOCK: flash crashes (>5% move in <1h), exchange outages, major regulatory news '
    '(country bans, SEC actions), extreme liquidation cascades (>$500M), '
    'confirmed hacks/exploits on major exchanges\n'
    '- CAUTION: elevated volatility (>3% move in <1h), mixed/conflicting signals, '
    'weekend/holiday low liquidity, moderate liquidations ($100-500M), '
    'uncertain regulatory developments\n'
    '- OK: normal market conditions, signals align, adequate liquidity\n\n'
    'Return ONLY valid JSON. No explanation outside the JSON.'
)


def llm_risk_check(trade_context, cache_db='llm_cache.db'):
    """
    Pre-trade risk gate using Claude Sonnet.

    Args:
        trade_context: dict with keys:
            price: current BTC price
            prediction: model prediction (LONG/SHORT/FLAT)
            confidence: prediction confidence (0-1)
            recent_news: list of recent headline strings
            recent_tweets: list of recent tweet strings
            price_change_1h: % price change last hour
            price_change_24h: % price change last 24h
            volume_ratio: current vs avg volume ratio
            funding_rate: current funding rate
            open_interest_change: OI change %

    Returns:
        dict with:
            decision: 'OK' | 'CAUTION' | 'BLOCK'
            reason: str
            risk_score: float (0-1)
    """
    default_ok = {'decision': 'OK', 'reason': 'API unavailable, defaulting to OK', 'risk_score': 0.0}

    conn = _init_cache_db(cache_db)
    try:
        ctx_hash = _context_hash(trade_context)

        # Check cache (5 min TTL)
        cached = _get_cached_risk(conn, ctx_hash)
        if cached is not None:
            logger.debug('Risk cache hit')
            return cached

        client = _get_client()
        if client is None:
            logger.warning('No API client for risk check, defaulting to OK')
            return default_ok

        # Build context message
        ctx_parts = []
        ctx_parts.append('Current BTC price: ${:,.2f}'.format(trade_context.get('price', 0)))
        ctx_parts.append('Prediction: {} (confidence: {:.2%})'.format(
            trade_context.get('prediction', 'UNKNOWN'),
            trade_context.get('confidence', 0),
        ))
        ctx_parts.append('Price change 1h: {:.2%}'.format(
            trade_context.get('price_change_1h', 0)
        ))
        ctx_parts.append('Price change 24h: {:.2%}'.format(
            trade_context.get('price_change_24h', 0)
        ))
        ctx_parts.append('Volume ratio (vs avg): {:.2f}'.format(
            trade_context.get('volume_ratio', 1.0)
        ))
        ctx_parts.append('Funding rate: {:.4%}'.format(
            trade_context.get('funding_rate', 0)
        ))
        ctx_parts.append('OI change: {:.2%}'.format(
            trade_context.get('open_interest_change', 0)
        ))

        news = trade_context.get('recent_news', [])
        if news:
            ctx_parts.append('\nRecent headlines:')
            for h in news[:10]:
                ctx_parts.append(f'  - {h[:200]}')

        tweets = trade_context.get('recent_tweets', [])
        if tweets:
            ctx_parts.append('\nRecent tweets:')
            for t in tweets[:10]:
                ctx_parts.append(f'  - {t[:200]}')

        user_msg = '\n'.join(ctx_parts)

        result = _call_risk_api(client, user_msg, conn)
        if result:
            _set_cached_risk(
                conn, ctx_hash, result, SONNET_MODEL,
                result.pop('_inp_tok', 0), result.pop('_out_tok', 0),
            )
            return result

        logger.error('Risk API failed, defaulting to CAUTION')
        return {'decision': 'CAUTION', 'reason': 'API failed, defaulting to caution', 'risk_score': 0.5}
    finally:
        conn.close()


def _call_risk_api(client, user_msg, conn):
    """Call Claude Sonnet for risk assessment with retries."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model=SONNET_MODEL,
                max_tokens=512,
                system=_RISK_SYSTEM_PROMPT,
                messages=[{'role': 'user', 'content': user_msg}],
            )

            inp_tok = response.usage.input_tokens
            out_tok = response.usage.output_tokens
            _log_cost(conn, SONNET_MODEL, 'risk_check', inp_tok, out_tok)

            response_text = response.content[0].text.strip()

            # Strip markdown fences
            cleaned = response_text
            if cleaned.startswith('```'):
                lines = cleaned.split('\n')
                lines = [l for l in lines if not l.strip().startswith('```')]
                cleaned = '\n'.join(lines).strip()

            parsed = json.loads(cleaned)

            decision = parsed.get('decision', 'OK').upper()
            if decision not in ('OK', 'CAUTION', 'BLOCK'):
                decision = 'CAUTION'

            result = {
                'decision': decision,
                'reason': str(parsed.get('reason', 'No reason provided')),
                'risk_score': float(parsed.get('risk_score', 0.0)),
                '_inp_tok': inp_tok,
                '_out_tok': out_tok,
            }

            logger.info(
                'Risk check: %s (score=%.2f) -- %s',
                result['decision'], result['risk_score'], result['reason'],
            )
            return result

        except Exception as e:
            err_name = type(e).__name__
            logger.warning(
                'Attempt %d: risk API error (%s): %s',
                attempt + 1, err_name, str(e)[:200],
            )
            if 'rate' in str(e).lower() or '429' in str(e):
                wait = min(BACKOFF_MAX, BACKOFF_BASE ** (attempt + 2))
                time.sleep(wait)
            elif attempt < MAX_RETRIES - 1:
                wait = min(BACKOFF_MAX, BACKOFF_BASE ** (attempt + 1))
                time.sleep(wait)

    return None


# ============================================================
# FEATURE INTEGRATION (for feature_library.py)
# ============================================================

def compute_llm_features(df, tweets_df=None, news_df=None,
                         bucket_seconds=3600, cache_db='llm_cache.db'):
    """
    Compute LLM sentiment features for each OHLCV bucket.

    For each time bucket, finds the most impactful tweet/headline and
    gets LLM sentiment on it. Uses cache heavily -- most historical
    text is already analyzed.

    Args:
        df: OHLCV DataFrame with DatetimeIndex
        tweets_df: DataFrame with 'created_at'/'timestamp' and 'text' columns
        news_df: DataFrame with 'published_at'/'timestamp' and 'title'/'headline' columns
        bucket_seconds: timeframe bucket size in seconds
        cache_db: path to cache database

    Returns:
        DataFrame with columns:
            llm_sentiment, llm_sarcastic, llm_urgent,
            llm_context_score, llm_market_impact
    """
    n = len(df)
    out = pd.DataFrame(index=df.index)

    # Initialise with NaN (not 0 -- LightGBM handles NaN natively)
    for col in ['llm_sentiment', 'llm_sarcastic', 'llm_urgent',
                'llm_context_score', 'llm_market_impact']:
        out[col] = np.nan

    if (tweets_df is None or tweets_df.empty) and (news_df is None or news_df.empty):
        logger.debug('No tweets or news data for LLM features')
        return out

    # ---- Prepare text sources with timestamps ----
    all_texts = []  # list of (timestamp, text, source)

    if tweets_df is not None and not tweets_df.empty:
        ts_col = None
        for c in ['created_at', 'timestamp', 'date']:
            if c in tweets_df.columns:
                ts_col = c
                break
        text_col = None
        for c in ['text', 'content', 'tweet']:
            if c in tweets_df.columns:
                text_col = c
                break

        if ts_col and text_col:
            for _, row in tweets_df.iterrows():
                ts = pd.Timestamp(row[ts_col])
                text = str(row[text_col]) if pd.notna(row[text_col]) else ''
                if text.strip():
                    all_texts.append((ts, text, 'tweet'))

    if news_df is not None and not news_df.empty:
        ts_col = None
        for c in ['published_at', 'timestamp', 'date', 'created_at']:
            if c in news_df.columns:
                ts_col = c
                break
        text_col = None
        for c in ['title', 'headline', 'text']:
            if c in news_df.columns:
                text_col = c
                break

        if ts_col and text_col:
            for _, row in news_df.iterrows():
                ts = pd.Timestamp(row[ts_col])
                text = str(row[text_col]) if pd.notna(row[text_col]) else ''
                if text.strip():
                    all_texts.append((ts, text, 'news'))

    if not all_texts:
        logger.debug('No valid text entries for LLM features')
        return out

    # ---- Assign texts to buckets ----
    # For each OHLCV bar, find the "most impactful" text in its time window
    # "Most impactful" = longest text (proxy for most detailed/informative)
    bucket_td = pd.Timedelta(seconds=bucket_seconds)

    # Make df index timezone-naive for matching
    bar_times = df.index
    if bar_times.tz is not None:
        bar_times_naive = bar_times.tz_localize(None)
    else:
        bar_times_naive = bar_times

    # Sort texts by timestamp
    all_texts.sort(key=lambda x: x[0])

    # Map: bar_index -> best text for that bar
    bar_texts = {}
    for ts, text, source in all_texts:
        # Make timestamp naive for comparison
        if hasattr(ts, 'tz') and ts.tz is not None:
            ts_naive = ts.tz_localize(None)
        else:
            ts_naive = ts

        # Find which bar this text belongs to
        # Binary search for efficiency
        idx = bar_times_naive.searchsorted(ts_naive, side='right') - 1
        if idx < 0 or idx >= n:
            continue

        bar_start = bar_times_naive[idx]
        bar_end = bar_start + bucket_td
        if ts_naive < bar_start or ts_naive >= bar_end:
            continue

        # Keep the longest text per bar (proxy for most detailed)
        if idx not in bar_texts or len(text) > len(bar_texts[idx]):
            bar_texts[idx] = text

    if not bar_texts:
        logger.debug('No texts mapped to OHLCV bars')
        return out

    logger.info(
        'LLM features: %d bars have text (out of %d total bars)',
        len(bar_texts), n,
    )

    # ---- Batch process all texts ----
    sorted_indices = sorted(bar_texts.keys())
    texts_to_process = [bar_texts[i] for i in sorted_indices]

    results = llm_sentiment_batch(texts_to_process, cache_db=cache_db)

    # ---- Map results back to output DataFrame ----
    for j, bar_idx in enumerate(sorted_indices):
        if j < len(results) and results[j] is not None:
            r = results[j]
            iloc_pos = bar_idx  # Direct integer position
            out.iloc[iloc_pos, out.columns.get_loc('llm_sentiment')] = r.get('llm_sentiment', np.nan)
            out.iloc[iloc_pos, out.columns.get_loc('llm_sarcastic')] = float(r.get('llm_sarcastic', False))
            out.iloc[iloc_pos, out.columns.get_loc('llm_urgent')] = float(r.get('llm_urgent', False))
            out.iloc[iloc_pos, out.columns.get_loc('llm_context_score')] = r.get('llm_context_score', np.nan)
            out.iloc[iloc_pos, out.columns.get_loc('llm_market_impact')] = r.get('llm_market_impact', np.nan)

    # Convert bool columns to float for LightGBM
    out['llm_sarcastic'] = pd.to_numeric(out['llm_sarcastic'], errors='coerce')
    out['llm_urgent'] = pd.to_numeric(out['llm_urgent'], errors='coerce')

    return out


# ============================================================
# SELF-TEST
# ============================================================

if __name__ == '__main__':
    print('llm_features.py -- LLM Sentiment & Risk Features')
    print('=' * 60)

    # Test regex fallback
    test_texts = [
        'Bitcoin surges past $100K! Bullish momentum unstoppable',
        'BREAKING: Major exchange hacked, funds stolen. Sell everything!',
        'The market is totally fine, nothing to see here /s lol',
        'Fed announces rate decision tomorrow, markets wait',
        '',
    ]

    print('\n--- Regex Fallback Test ---')
    for t in test_texts:
        r = _regex_fallback_sentiment(t)
        print(f'  Text: {t[:60]}...')
        print(f'    sentiment={r["llm_sentiment"]:.2f}  sarcastic={r["llm_sarcastic"]}  '
              f'urgent={r["llm_urgent"]}  context={r["llm_context_score"]:.2f}  '
              f'impact={r["llm_market_impact"]:.2f}')

    # Test API if key available
    print('\n--- API Test ---')
    try:
        client = _get_client()
        if client:
            print('API key found. Testing single sentiment...')
            result = llm_sentiment(test_texts[0])
            print(f'  Result: {result}')

            print('\nTesting batch sentiment...')
            batch = llm_sentiment_batch(test_texts[:3])
            for i, r in enumerate(batch):
                print(f'  [{i}] {r}')

            print('\nTesting risk check...')
            ctx = {
                'price': 98500.0,
                'prediction': 'LONG',
                'confidence': 0.72,
                'price_change_1h': 0.015,
                'price_change_24h': 0.032,
                'volume_ratio': 1.2,
                'funding_rate': 0.0001,
                'open_interest_change': 0.05,
                'recent_news': ['Bitcoin ETF sees record inflows'],
                'recent_tweets': ['BTC looking strong above 98K'],
            }
            risk = llm_risk_check(ctx)
            print(f'  Risk: {risk}')

            print('\nCost summary:')
            costs = get_cost_summary()
            print(f'  Total cost: ${costs["total_cost"]:.6f}')
            print(f'  Total calls: {costs["total_calls"]}')
        else:
            print('No API key. Set ANTHROPIC_API_KEY in .env to enable LLM features.')
    except Exception as e:
        print(f'API test error: {e}')

    print('\nDone.')
