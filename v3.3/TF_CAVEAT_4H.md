# 4H (4-Hour) Timeframe Caveats

## Trade Characteristics
- **Rows**: 8,794
- **Trade duration**: 10-140 bars (40 hours to 23 days)
- **max_hold_bars**: 72
- **Cross features**: ~1.5M (cross gen ENABLED)
- **ETA on 1x RTX 5090**: ~3.5 hours

## Key Differences from 1W/1D
- First TF where cross gen produces SUBSTANTIAL feature count
- 8,794 rows / 1.5M features with EFB bundling → manageable
- Full TA suite — no lean mode needed
- 3-class likely better than binary (enough rows)
- Esoteric features fire 6x more than 1w (6 bars per day × 365 × 16 years)
- Lunar cycles: ~30 days = ~180 bars per cycle → well-captured
- Mercury retrograde: ~21 days = ~126 bars per event → very learnable

## Feature Adjustments
- hour_sin/cos: KEEP (4h bars have different hours)
- day_of_week: KEEP (weekday effects real)
- return_bars: [1, 6, 12, 42] → add [84] for 2-week return
- All numerology/gematria: KEEP
- Prime features: ADD — enough bars for primes to fire reliably
- CPCV: (10,2) = 45 paths, sample 30
- num_leaves: 31-63
- LR: 0.05-0.15

## Prime Numbers on 4H
- 8,794 bars: price_is_prime fires ~1000x (BTC price is prime ~11% of integers)
- week_is_prime fires ~15 weeks × 42 bars × 16 years = ~10K bars
- doy_is_prime fires ~54 prime DOY values × 6 bars × 16 = ~5K bars
- Prime confluence: meaningful signal at this row count
