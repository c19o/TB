# SAVAGE22 Trading Terminal

Crypto trading analysis dashboard combining numerology, astrology, technical analysis, and tweet decoding signals overlaid on TradingView-style candlestick charts.

## Setup

```bash
cd dashboard
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Requirements

- Node.js 18+
- The parent directory must contain the SQLite databases:
  - `btc_prices.db` - OHLCV candle data
  - `tweets.db` - Scraped decoded tweets
  - `savage22.db` - Discord signals

## API Routes

| Route | Params | Description |
|-------|--------|-------------|
| `/api/candles` | `symbol`, `timeframe`, `from`, `to`, `limit` | OHLCV candle data |
| `/api/signals` | `date`, `from`, `to` | Numerology/astrology signals |
| `/api/tweets` | `limit`, `handle`, `from`, `to` | Decoded tweets with gematria |
| `/api/score` | `date` | Unified convergence score |
| `/api/overlays` | `type`, `from`, `to` | Chart overlay marker data |

## Stack

- Next.js 14 (App Router)
- TradingView lightweight-charts
- better-sqlite3
- Tailwind CSS
- TypeScript
