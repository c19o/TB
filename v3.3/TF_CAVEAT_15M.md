# 15M (15-Minute) Timeframe Caveats

## Trade Characteristics
- **Rows**: ~227,000
- **Trade duration**: 7-40 bars (1.75 hours to 10 hours)
- **max_hold_bars**: 24
- **Cross features**: ~2.9M (full cross gen, memmap streaming, force_row_wise)
- **ETA on 1x RTX 5090**: ~23-25 hours
- **Peak RAM**: ~210 GB (TIGHT on 258GB machine — may OOM)
- **ETA on 8x RTX 5090**: ~3.5-5.5 hours

## Key Differences — THE MONSTER TF
- Most data, most features, longest training time
- 227K rows: statistical power to detect ANYTHING
- 2.9M cross features all viable (0.08 samples/feature BUT EFB bundles to 23K)
- force_row_wise=True (high row/feature ratio after bundling)
- This TF is where the full matrix thesis reaches maximum expression

## Feature Adjustments
- ALL features KEEP — 227K rows supports everything
- minute_sin/cos: KEEP (15m resolution matters)
- session flags: KEEP (critical for intraday)
- return_bars: [1, 4, 16, 96] — 15m to 1d returns
- Full cross gen with memmap streaming (3TB+ before compression)
- CPCV: (10,2) = 45 paths, sample 30
- num_leaves: 127-255
- LR: 0.01-0.05
- 3-class optimal

## RAM Warning
210GB peak on 258GB machine = only 48GB headroom. Mitigations:
- V2_RIGHT_CHUNK=200 (reduce cross gen memory)
- Monitor RSS during first CPCV fold
- If OOM, need 512GB+ machine

## Prime Numbers on 15M
- 227K bars: primes are a RICH signal
- Every 15m bar can check: minute_is_prime, hour_is_prime, price_is_prime
- Prime confluence fires thousands of times → stable signal
- Prime × session × trend triple crosses become viable
- Distance-to-nearest-prime on price → support/resistance levels

## Esoteric Expectations
- This is the matrix's FINAL FORM
- 227K rows: every esoteric feature fires 10,000+ times
- Cross features (esoteric × TA): 2.9M combinations
- Expect 40-60% of model gain from esoteric on 15m
- Space weather changes happen on 15m timescale
- Lunar phase precision is meaningful (0.5° per 15m bar)
- Planetary hour changes happen within the trading day
