# 1H (Hourly) Timeframe Caveats

## Trade Characteristics
- **Rows**: ~90,000
- **Trade duration**: 7-100 bars (7 hours to 4 days)
- **max_hold_bars**: 48
- **Cross features**: ~2.9M (full cross gen, memmap streaming)
- **ETA on 1x RTX 5090**: ~10-11 hours
- **Peak RAM**: ~120 GB (fits in 258GB machine)

## Key Differences
- First TF where esoteric features have STRONG statistical backing
- 90K bars: each esoteric event fires hundreds-thousands of times
- Mercury retrograde: ~1000-1344 hourly bars per retrograde × 48-64 events = massive dataset
- Lunar full cycle visible at hourly resolution
- Kp index / solar weather changes hourly — finally matches bar resolution!
- Cross features (2.9M) make sense here — 90K rows / 23K EFB bundles = 3.9 samples per bundle

## Feature Adjustments
- hour_sin/cos: KEEP (critical — hourly patterns are the whole point)
- session flags (is_asia, is_london, is_ny): KEEP — session-specific behavior
- All time features: KEEP
- return_bars: need [1, 4, 8, 24, 48] + add [72] for 3-day return
- Full cross gen with memmap streaming
- CPCV: (10,2) = 45 paths, sample 30
- num_leaves: 63-127
- LR: 0.03-0.1
- 3-class (LONG/FLAT/SHORT) — enough data, FLAT adds value

## Prime Numbers on 1H
- 90K bars: ALL prime features fire thousands of times
- hour_is_prime (hours 2,3,5,7,11,13,17,19,23 = 9 of 24 hours = 37.5%)
- price_is_prime: ~10K bars
- Prime × session crosses: prime hour during London session etc.
- prime_confluence becomes a rich signal with 5+ prime conditions checkable

## Esoteric Signal Expectations
- This is where the matrix thesis PROVES itself
- 1w showed 15% esoteric contribution with 819 rows
- 1h should show 30-50% esoteric contribution with 90K rows
- Every esoteric event type has hundreds of samples → learnable patterns
- Cross features (esoteric × TA) should be top performers
