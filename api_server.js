#!/usr/bin/env node
/**
 * Standalone API server for Savage22 Dashboard
 * Handles: SQLite queries, Bitget live data, trade data
 * Runs independently from Next.js — no webpack compilation issues
 */

const express = require('express');
const cors = require('cors');
const Database = require('better-sqlite3');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = 3001;
const DB_DIR = process.env.SAVAGE22_DB_DIR || 'C:/Users/C/Documents/Savage22 Server/v3.1';

// Enable CORS for Next.js frontend
app.use(cors({ origin: ['http://localhost:3000', 'http://127.0.0.1:3000'] }));
app.use(express.json());

// ============ DB CONNECTIONS (singletons) ============
let pricesDb, tradesDb;

function getPricesDb() {
  if (!pricesDb) {
    pricesDb = new Database(path.join(DB_DIR, 'btc_prices.db'), { readonly: true });
    console.log('[DB] Opened btc_prices.db');
  }
  return pricesDb;
}

function getTradesDb() {
  if (!tradesDb) {
    tradesDb = new Database(path.join(DB_DIR, 'trades.db'), { readonly: true });
    console.log('[DB] Opened trades.db');
  }
  return tradesDb;
}

// ============ CANDLES ============
app.get('/api/candles', (req, res) => {
  try {
    const symbol = req.query.symbol || 'BTC';
    const timeframe = req.query.timeframe || '1d';
    const from = req.query.from;
    const to = req.query.to;
    const limit = parseInt(req.query.limit || '0');  // 0 = no limit

    const db = getPricesDb();
    let query = 'SELECT open_time as time, open, high, low, close, volume FROM ohlcv WHERE symbol = ? AND timeframe = ?';
    const params = [symbol, timeframe];

    if (from) { query += ' AND open_time >= ?'; params.push(parseInt(from) * 1000); }
    if (to) { query += ' AND open_time <= ?'; params.push(parseInt(to) * 1000); }

    query += ' ORDER BY open_time DESC';
    if (limit > 0) { query += ' LIMIT ?'; params.push(limit); }

    const rows = db.prepare(query).all(...params);
    rows.reverse();

    const candles = rows.map(row => ({
      time: Math.floor(row.time / 1000),
      open: row.open, high: row.high, low: row.low, close: row.close,
      volume: row.volume || 0,
    }));

    res.json({ candles, count: candles.length });
  } catch (err) {
    console.error('Candles error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

// ============ ML TRADES ============
app.get('/api/ml-trades', (req, res) => {
  try {
    const limit = parseInt(req.query.limit || '100');
    const db = getTradesDb();

    const account = db.prepare('SELECT * FROM account WHERE id = 1').get();
    const openTrades = db.prepare('SELECT * FROM trades WHERE status = ? ORDER BY created_at DESC').all('open');
    const closedTrades = db.prepare('SELECT * FROM trades WHERE status = ? ORDER BY exit_time DESC LIMIT ?').all('closed', limit);
    const allTrades = db.prepare('SELECT * FROM trades ORDER BY created_at DESC LIMIT ?').all(500);
    const equity = db.prepare('SELECT timestamp, balance, dd_pct FROM equity_curve ORDER BY timestamp DESC LIMIT 500').all();

    const today = new Date().toISOString().split('T')[0];
    const todayTrades = db.prepare("SELECT COUNT(*) as count, SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins FROM trades WHERE date(exit_time) = ? AND status = 'closed'").get(today);

    res.json({
      account: account ? {
        balance: account.balance, peak_balance: account.peak_balance,
        total_trades: account.total_trades, wins: account.wins, losses: account.losses,
        max_dd: account.max_dd,
        roi: account.balance ? ((account.balance - 100) / 100 * 100) : 0,
        win_rate: account.total_trades > 0 ? (account.wins / account.total_trades * 100) : 0,
        mode: account.mode, updated_at: account.updated_at,
      } : null,
      open_trades: openTrades,
      closed_trades: closedTrades,
      all_trades: allTrades,
      equity_curve: equity.reverse(),
      today: { trades: todayTrades?.count || 0, wins: todayTrades?.wins || 0 },
    });
  } catch (err) {
    res.json({
      account: { balance: 100, peak_balance: 100, total_trades: 0, wins: 0, losses: 0, max_dd: 0, roi: 0, win_rate: 0, mode: 'paper', updated_at: '' },
      open_trades: [], closed_trades: [], all_trades: [], equity_curve: [], today: { trades: 0, wins: 0 },
    });
  }
});

// ============ PAPER TRADING (v3.1 — reads trades.db) ============
app.get('/api/paper-trading', (req, res) => {
  try {
    const db = getTradesDb();
    if (!db) return res.json({ error: 'trades.db not found', account: null, open_positions: [], recent_trades: [], equity_curve: [] });
    const account = db.prepare('SELECT * FROM account WHERE id = 1').get();
    const pnl = (account.balance || 1000) - 1000;
    const roi = (pnl / 1000) * 100;
    const winRate = account.total_trades > 0 ? (account.wins / account.total_trades) * 100 : 0;
    const pf = account.losses > 0 ? account.wins / account.losses : account.wins > 0 ? 999 : 0;
    const enriched = { ...account, pnl, roi, win_rate: winRate, profit_factor: pf, current_dd: account.max_dd || 0 };
    const open_positions = db.prepare("SELECT trades.id, tf as signal_name, direction, entry_price, stop_price as stop_loss, tp_price as take_profit, confidence, leverage, risk_pct, entry_time, NULL as trailing_stop, (risk_pct/100.0 * 1000) as risk_amount, (leverage * risk_pct/100.0 * 1000) as position_size FROM trades WHERE trades.status = 'open' ORDER BY created_at DESC").all();
    const recent_trades = db.prepare("SELECT id, tf as signal_name, direction, entry_price, exit_price, stop_price as stop_loss, tp_price as take_profit, pnl, pnl_pct as pct_return, bars_held, exit_reason, entry_time, exit_time, confidence, regime, leverage FROM trades WHERE status = 'closed' ORDER BY exit_time DESC LIMIT 50").all();
    const equity_curve = db.prepare("SELECT timestamp as time, balance as value, dd_pct FROM equity_curve ORDER BY timestamp").all();
    res.json({ account: enriched, open_positions, recent_trades, equity_curve });
  } catch (err) {
    res.json({ error: err.message, account: null, open_positions: [], recent_trades: [], equity_curve: [] });
  }
});

app.get('/api/paper-trading/signals', (req, res) => {
  try {
    const db = getPaperTradesDb();
    if (!db) return res.json({ signals: [] });
    const hours = parseInt(req.query.hours || '24');
    // Build signals from trades (executed) + rejected trades
    const traded = db.prepare("SELECT id, tf, direction, confidence, entry_price as price, entry_time as timestamp, regime, 'traded' as action FROM trades ORDER BY created_at DESC LIMIT 50").all();
    const rejected = db.prepare("SELECT id, tf, direction, confidence, price, timestamp, reason as action FROM rejected_trades ORDER BY timestamp DESC LIMIT 50").all().catch?.(() => []) || [];
    const signals = [...traded, ...rejected].sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()).slice(0, 100);
    const summary = { traded: traded.length, filtered: Array.isArray(rejected) ? rejected.length : 0 };
    res.json({ signals, summary });
  } catch (err) { res.json({ signals: [], summary: { traded: 0, filtered: 0 } }); }
});

app.get('/api/paper-trading/anx', (req, res) => {
  res.json({ account: null, trades: [] });
});

// ============ BTC PRICE (Bitget) ============
app.get('/api/btc-price', async (req, res) => {
  try {
    const resp = await fetch(
      'https://api.bitget.com/api/v2/mix/market/ticker?productType=USDT-FUTURES&symbol=BTCUSDT',
      { signal: AbortSignal.timeout(5000) }
    );
    const data = await resp.json();
    const ticker = data.data?.[0] || data.data || {};
    const price = parseFloat(ticker.lastPr || ticker.last || '0');

    if (price > 0) {
      const db = getPricesDb();
      const prevRow = db.prepare('SELECT close FROM ohlcv WHERE symbol = ? AND timeframe = ? ORDER BY open_time DESC LIMIT 1').get('BTC', '1d');
      const change24h = prevRow?.close > 0 ? ((price - prevRow.close) / prevRow.close) * 100 : 0;
      return res.json({ price, change24h: Math.round(change24h * 100) / 100, source: 'live' });
    }
  } catch {}

  // Fallback to DB
  try {
    const db = getPricesDb();
    const rows = db.prepare('SELECT close FROM ohlcv WHERE symbol = ? AND timeframe = ? ORDER BY open_time DESC LIMIT 2').all('BTC', '1d');
    const price = rows[0]?.close || null;
    const change24h = rows.length >= 2 && rows[1].close > 0 ? ((rows[0].close - rows[1].close) / rows[1].close) * 100 : 0;
    res.json({ price, change24h: Math.round(change24h * 100) / 100, source: 'db' });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ============ BITGET CANDLES (for live/backfill) ============
const BITGET_GRAN = { '1m': '1m', '5m': '5m', '15m': '15m', '1h': '1H', '4h': '4H', '1d': '1D', '1w': '1W' };

app.get('/api/bitget-candles', async (req, res) => {
  try {
    const tf = req.query.timeframe || '1d';
    const limit = Math.min(parseInt(req.query.limit || '200'), 200);
    const gran = BITGET_GRAN[tf] || '1D';

    const resp = await fetch(
      `https://api.bitget.com/api/v2/mix/market/candles?productType=USDT-FUTURES&symbol=BTCUSDT&granularity=${gran}&limit=${limit}`,
      { signal: AbortSignal.timeout(5000) }
    );
    const data = await resp.json();
    const raw = data?.data || data;
    if (!Array.isArray(raw)) return res.json({ candles: [] });

    const candles = raw.map(c => ({
      timestamp: parseInt(c[0]),
      open: parseFloat(c[1]), high: parseFloat(c[2]),
      low: parseFloat(c[3]), close: parseFloat(c[4]),
      volume: parseFloat(c[5]) || 0,
    })).sort((a, b) => a.timestamp - b.timestamp);

    res.json({ candles, count: candles.length });
  } catch (err) {
    res.status(500).json({ error: err.message, candles: [] });
  }
});

// ============ OVERLAYS ============
app.get('/api/overlays', (req, res) => {
  try {
    const from = parseInt(req.query.from || '0');
    const to = parseInt(req.query.to || String(Math.floor(Date.now() / 1000)));
    const points = [];

    const CAUTION_NUMBERS = new Set([39, 43, 48, 93, 113, 223, 322]);
    const PUMP_NUMBERS = new Set([37, 73, 127]);
    const RITUAL_DATES = {
      '1/3': 'BTC Genesis', '3/20': 'Spring Equinox', '3/22': '322 Skull & Bones',
      '5/1': 'Beltane', '5/22': 'BTC Pizza Day', '6/21': 'Summer Solstice',
      '9/22': 'Fall Equinox', '10/31': 'Samhain', '12/21': 'Winter Solstice',
    };

    const current = new Date(from * 1000);
    current.setUTCHours(0, 0, 0, 0);
    const endDate = new Date(to * 1000);
    const dayTypeKeys = new Set();

    while (current <= endDate) {
      const ts = Math.floor(current.getTime() / 1000);
      const dayKey = current.toISOString().split('T')[0];
      const dayOfYear = Math.floor((current - new Date(Date.UTC(current.getUTCFullYear(), 0, 0))) / 86400000);
      const month = current.getUTCMonth() + 1;
      const day = current.getUTCDate();

      // Caution/pump numbers
      const vals = [dayOfYear, month * 100 + day, day * 100 + month];
      for (const v of vals) {
        if (CAUTION_NUMBERS.has(v) && !dayTypeKeys.has(`${dayKey}-caution`)) {
          dayTypeKeys.add(`${dayKey}-caution`);
          points.push({ time: ts, type: 'caution', label: `⚠ ${v}`, color: '#ef4444', direction: 'bearish' });
        }
        if (PUMP_NUMBERS.has(v) && !dayTypeKeys.has(`${dayKey}-pump`)) {
          dayTypeKeys.add(`${dayKey}-pump`);
          points.push({ time: ts, type: 'pump', label: `📈 ${v}`, color: '#22c55e', direction: 'bullish' });
        }
      }

      // Ritual dates
      const md = `${month}/${day}`;
      if (RITUAL_DATES[md] && !dayTypeKeys.has(`${dayKey}-ritual`)) {
        dayTypeKeys.add(`${dayKey}-ritual`);
        points.push({ time: ts, type: 'ritual', label: `🔮 ${RITUAL_DATES[md]}`, color: '#8b5cf6', direction: 'neutral' });
      }

      // Day 21
      if (day === 21 && !dayTypeKeys.has(`${dayKey}-day21`)) {
        dayTypeKeys.add(`${dayKey}-day21`);
        points.push({ time: ts, type: 'caution', label: '💀 Day 21', color: '#dc2626', direction: 'bearish' });
      }

      // Day 13
      if (day === 13 && !dayTypeKeys.has(`${dayKey}-day13`)) {
        dayTypeKeys.add(`${dayKey}-day13`);
        points.push({ time: ts, type: 'caution', label: '⚠ Day 13', color: '#f59e0b', direction: 'bearish' });
      }

      // Day 27
      if (day === 27 && !dayTypeKeys.has(`${dayKey}-day27`)) {
        dayTypeKeys.add(`${dayKey}-day27`);
        points.push({ time: ts, type: 'pump', label: '🔼 Day 27', color: '#10b981', direction: 'bullish' });
      }

      // Lunar: full moon (~every 29.53 days from known new moon Jan 6 2000)
      const SYNODIC = 29.53059;
      const KNOWN_NEW = Date.UTC(2000, 0, 6, 18, 14);
      const phaseDays = ((current.getTime() - KNOWN_NEW) / 86400000) % SYNODIC;
      if (phaseDays < 1.0 && !dayTypeKeys.has(`${dayKey}-lunar`)) {
        dayTypeKeys.add(`${dayKey}-lunar`);
        points.push({ time: ts, type: 'lunar', label: '🌑 New Moon', color: '#6366f1', direction: 'bullish' });
      }
      if (Math.abs(phaseDays - SYNODIC / 2) < 1.0 && !dayTypeKeys.has(`${dayKey}-lunar`)) {
        dayTypeKeys.add(`${dayKey}-lunar`);
        points.push({ time: ts, type: 'lunar', label: '🌕 Full Moon', color: '#f59e0b', direction: 'bearish' });
      }

      current.setUTCDate(current.getUTCDate() + 1);
    }

    // Tweet overlays: gold/red tweet days from tweets.db
    try {
      const tweetsDb = new Database(path.join(DB_DIR, 'tweets.db'), { readonly: true });
      const tweetDays = tweetsDb.prepare(`
        SELECT DISTINCT substr(created_at, 5, 11) as date_str, ts_unix, has_gold, has_red
        FROM tweets WHERE ts_unix >= ? AND ts_unix <= ? AND (has_gold = 1 OR has_red = 1)
        ORDER BY ts_unix
      `).all(from, to);
      tweetsDb.close();
      for (const tw of tweetDays) {
        const tts = tw.ts_unix;
        const dayTs = tts - (tts % 86400);
        const dk = `${dayTs}-tweet`;
        if (!dayTypeKeys.has(dk)) {
          dayTypeKeys.add(dk);
          if (tw.has_gold) {
            points.push({ time: dayTs, type: 'tweet', label: '🟡 Gold Tweet', color: '#eab308', direction: 'bullish' });
          }
          if (tw.has_red) {
            points.push({ time: dayTs, type: 'tweet', label: '🔴 Red Tweet', color: '#ef4444', direction: 'bearish' });
          }
        }
      }
    } catch (e) { /* tweets.db not available */ }

    res.json({ points, count: points.length });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ============ TICK (live price + account) ============
app.get('/api/tick', (req, res) => {
  try {
    const db = getPricesDb();
    const candle = db.prepare("SELECT close, open_time FROM ohlcv WHERE symbol='BTC' AND timeframe='15m' ORDER BY open_time DESC LIMIT 1").get();
    const tdb = getTradesDb();
    const account = tdb.prepare('SELECT * FROM account WHERE id = 1').get();
    const openTrades = tdb.prepare("SELECT * FROM trades WHERE status = 'open'").all();
    res.json({
      price: candle?.close || 0, timestamp: candle?.open_time || 0,
      account: account ? { balance: account.balance, peak_balance: account.peak_balance, mode: account.mode, total_trades: account.total_trades, wins: account.wins, losses: account.losses } : null,
      open_trades: openTrades.length,
    });
  } catch (err) { res.json({ price: 0, timestamp: 0, account: null, open_trades: 0 }); }
});

// ============ MANIPULATION SIGNALS ============
app.get('/api/manipulation', (req, res) => {
  try {
    // Check for recent caution signals from paper_trades.db signals_fired table
    const pdb = getPaperTradesDb();
    if (pdb) {
      const signals = pdb.prepare("SELECT * FROM signals_fired WHERE action='traded' ORDER BY timestamp DESC LIMIT 10").all();
      const cautionCount = signals.filter(s => s.category === 'manipulation' || s.category === 'caution').length;
      res.json({
        signals: signals.map(s => ({ name: s.signal_name, direction: s.direction, strength: s.strength, category: s.category, time: s.timestamp })),
        score: Math.min(cautionCount * 20, 100),
        level: cautionCount >= 3 ? 'HIGH' : cautionCount >= 1 ? 'MEDIUM' : 'LOW',
      });
    } else {
      res.json({ signals: [], score: 0, level: 'neutral' });
    }
  } catch (err) { res.json({ signals: [], score: 0, level: 'neutral' }); }
});

// ============ TWEETS (from tweets.db) ============
app.get('/api/tweets', (req, res) => {
  try {
    const limit = parseInt(req.query.limit || '50');
    const tweetsDb = new Database(path.join(DB_DIR, 'tweets.db'), { readonly: true });
    const tweets = tweetsDb.prepare("SELECT user_handle, user_name, full_text, created_at, ts_unix, favorite_count, retweet_count, reply_count, has_gold, has_red, gematria_simple, gematria_english FROM tweets ORDER BY ts_unix DESC LIMIT ?").all(limit);
    tweetsDb.close();
    res.json({ tweets });
  } catch (err) { res.json({ tweets: [] }); }
});

// ============ COINS ============
app.get('/api/coins', (req, res) => {
  try {
    const db = getPricesDb();
    const recent = db.prepare("SELECT close FROM ohlcv WHERE symbol='BTC' AND timeframe='1d' ORDER BY open_time DESC LIMIT 8").all();
    const price = recent[0]?.close || 0;
    const prevPrice = recent[1]?.close || price;
    const change24h = prevPrice > 0 ? ((price - prevPrice) / prevPrice * 100) : 0;
    const sparkline = recent.map(r => r.close).reverse();
    res.json({ coins: [{
      symbol: 'BTC', name: 'Bitcoin', shortName: 'BTC',
      price, change24h: Math.round(change24h * 100) / 100, sparkline,
    }] });
  } catch (err) {
    res.json({ coins: [{ symbol: 'BTC', name: 'Bitcoin', shortName: 'BTC', price: 0, change24h: 0, sparkline: [] }] });
  }
});

// ============ PREDICTION (from prediction_cache.json) ============
app.get('/api/prediction', (req, res) => {
  try {
    const fs = require('fs');
    const data = JSON.parse(fs.readFileSync(path.join(DB_DIR, 'prediction_cache.json'), 'utf8'));
    res.json(data);
  } catch (e) {
    res.json({ error: 'No prediction available' });
  }
});

// ============ FEATURES (grouped by category) ============
app.get('/api/features', (req, res) => {
  const tf = req.query.tf || '15m';
  try {
    const fs = require('fs');
    // Load feature list
    let features;
    try {
      features = JSON.parse(fs.readFileSync(path.join(DB_DIR, `features_${tf}_all.json`), 'utf8'));
    } catch {
      features = JSON.parse(fs.readFileSync(path.join(DB_DIR, `features_${tf}_pruned.json`), 'utf8'));
    }

    // Categorize features
    const categories = {
      'Technical': [], 'Astrology': [], 'Numerology': [], 'Gematria': [],
      'Tweets': [], 'News': [], 'Sports': [], 'On-chain': [],
      'Macro': [], 'Regime': [], 'KNN': [], 'Cross': [], 'Other': []
    };

    for (const f of features) {
      if (f.startsWith('knn_')) categories['KNN'].push(f);
      else if (f.includes('gem_') || f.includes('gematria')) categories['Gematria'].push(f);
      else if (f.includes('moon') || f.includes('nakshatra') || f.includes('vedic') || f.includes('bazi') || f.includes('tzolkin') || f.includes('arabic') || f.includes('planetary') || f.includes('retro') || f.includes('eclipse')) categories['Astrology'].push(f);
      else if (f.includes('dr_') || f.includes('digital_root') || f.includes('master') || f.includes('contains_')) categories['Numerology'].push(f);
      else if (f.includes('tweet') || f.includes('gold_tweet') || f.includes('red_tweet') || f.includes('misdirection')) categories['Tweets'].push(f);
      else if (f.includes('news') || f.includes('headline') || f.includes('caution')) categories['News'].push(f);
      else if (f.includes('sport') || f.includes('horse')) categories['Sports'].push(f);
      else if (f.includes('onchain') || f.includes('block') || f.includes('funding') || f.includes('whale')) categories['On-chain'].push(f);
      else if (f.includes('macro') || f.includes('sp500') || f.includes('nasdaq') || f.includes('dxy') || f.includes('vix')) categories['Macro'].push(f);
      else if (f.includes('ema50') || f.includes('hmm') || f.includes('regime')) categories['Regime'].push(f);
      else if (f.includes('cross_') || f.includes('_x_')) categories['Cross'].push(f);
      else if (['rsi', 'ema', 'atr', 'bb_', 'macd', 'volume', 'obv', 'adx', 'cci', 'stoch', 'willr', 'mfi'].some(ta => f.includes(ta))) categories['Technical'].push(f);
      else categories['Other'].push(f);
    }

    // Remove empty categories
    Object.keys(categories).forEach(k => { if (categories[k].length === 0) delete categories[k]; });

    res.json({ tf, total: features.length, categories });
  } catch (e) {
    res.json({ error: e.message });
  }
});

// ============ SCORE ============
app.get('/api/score', (req, res) => {
  res.json({ score: 50, threat_level: 'LOW', inversion_warning: false, phase_shift: false });
});

// ============ SIGNALS ============
app.get('/api/signals', (req, res) => {
  res.json({ signals: [] });
});

// ============ FUTURE SIGNALS ============
app.get('/api/future-signals', (req, res) => {
  try {
    const days = parseInt(req.query.days || '30');
    const category = req.query.category || null;
    const calendarPath = path.join(DB_DIR, 'signal_calendar.json');

    if (!fs.existsSync(calendarPath)) {
      return res.status(404).json({ error: 'signal_calendar.json not found. Run: python generate_signal_calendar.py' });
    }

    const raw = fs.readFileSync(calendarPath, 'utf8');
    const calendar = JSON.parse(raw);

    // Filter to requested date range
    const today = new Date().toISOString().split('T')[0];
    const cutoff = new Date(Date.now() + days * 86400000).toISOString().split('T')[0];

    let events = calendar.events.filter(e => e.date >= today && e.date <= cutoff);

    // Filter by category if specified
    if (category) {
      events = events.filter(e => e.category === category);
    }

    // Group by date
    const grouped = {};
    for (const e of events) {
      if (!grouped[e.date]) grouped[e.date] = [];
      grouped[e.date].push(e);
    }

    res.json({
      generated: calendar.generated,
      days_requested: days,
      total_events: events.length,
      dates_with_events: Object.keys(grouped).length,
      events,
      grouped,
    });
  } catch (err) {
    console.error('[API] future-signals error:', err.message);
    res.status(500).json({ error: err.message });
  }
});

// ============ START ============
app.listen(PORT, () => {
  console.log(`\n  SAVAGE22 API Server running on http://localhost:${PORT}`);
  console.log(`  DB: ${DB_DIR}`);
  // Pre-warm DB connections
  getPricesDb();
  getTradesDb();
  console.log(`  Ready!\n`);
});
