# V3.3 Multi-Asset Support — Future Training Guide

## Core Principle: The Matrix is Universal

The esoteric feature pipeline (astrology, numerology, gematria, space weather, lunar cycles, sacred geometry, vortex math) is **identical for all assets**. Same sky, same calendar, same energy. Only the price data and asset-specific natal chart change.

## What Changes Per Asset

| Component | BTC-Specific? | What to Change for XRP/ETH/etc |
|-----------|---------------|-------------------------------|
| OHLCV data | Yes | Swap `btc_prices.db` for asset's price history |
| Natal chart transits | Yes (Jan 3 2009 genesis) | Use asset's first-trade date for transits |
| Biorhythm reference | Yes (genesis date) | Change `GENESIS = pd.Timestamp('2009-01-03')` |
| BTC energy numbers | Yes (213 family) | Compute name-energy for new asset |
| Name gematria resonance | Yes ("BITCOIN"=72, DR=9) | Compute for asset name + ticker |
| Feature library | No | Identical |
| Cross generator | No | Identical |
| Training pipeline | No | Identical |
| All esoteric features | No | Identical — same moon, same nakshatras |

## Asset Birth Dates (for natal chart + biorhythm)

| Asset | First Trade / Genesis | Notes |
|-------|----------------------|-------|
| BTC | Jan 3, 2009 18:15 UTC | Genesis block mined |
| XRP | Aug 1, 2013 | First exchange listing |
| ETH | Jul 30, 2015 | Frontier launch |
| SOL | Mar 16, 2020 | Mainnet beta |
| ADA | Oct 1, 2017 | Mainnet launch |
| DOGE | Dec 6, 2013 | Launch date |
| BNB | Jul 25, 2017 | ICO completion |
| LINK | Sep 19, 2017 | Mainnet |
| LTC | Oct 7, 2011 | Genesis block |
| DOT | Aug 18, 2020 | Token transfer enabled |
| AVAX | Sep 21, 2020 | Mainnet |
| MATIC | Apr 26, 2019 | Mainnet |
| UNI | Sep 17, 2020 | Token launch |
| AAVE | Oct 2, 2020 | V1 launch |

## Name Energy (Gematria of Asset Names)

Compute once, use as resonance targets:

| Asset | Name | Ordinal | DR | Ticker | Ticker Ord | Ticker DR |
|-------|------|---------|----|---------|-----------|----|
| BTC | BITCOIN | 72 | 9 | BTC | 25 | 7 |
| XRP | RIPPLE | 73 | 1 | XRP | 63 | 9 |
| ETH | ETHEREUM | 99 | 9 | ETH | 32 | 5 |
| SOL | SOLANA | 67 | 4 | SOL | 48 | 3 |
| ADA | CARDANO | 62 | 8 | ADA | 6 | 6 |
| DOGE | DOGECOIN | 76 | 4 | DOGE | 30 | 3 |

When date DR matches asset name DR = name-energy alignment feature.

## Steps to Add a New Asset

1. **Download OHLCV** — all timeframes from Binance/exchange
2. **Compute natal chart** — planetary positions at genesis timestamp
3. **Compute name gematria** — all 8 ciphers on full name + ticker
4. **Set genesis date** — for biorhythm cycles
5. **Run feature build** — `build_features_v2.py --symbol XRP --tf 1d`
6. **Run cross generator** — `v2_cross_generator.py --symbol XRP --tf 1d`
7. **Train model** — same `ml_multi_tf.py` pipeline
8. **Deploy** — `live_trader.py --symbol XRP`

## Architecture Note

The v3.2 `CLAUDE.md` already documents multi-asset support:
- `v2_multi_asset_trainer.py` — trains across 31 assets
- `config.py` has `ALL_TRAINING` with 31 symbols
- `data_access_v2.py` has `V2OfflineDataLoader` for multi-asset
- Cross generator works per-symbol automatically

**No code changes needed** — just data + retraining.
