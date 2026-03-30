# 1W (Weekly) Timeframe Caveats

## Trade Characteristics
- **Rows**: 819 weekly bars (~2009-2025)
- **Trade duration**: 9-130 bars (2 months to 2.5 years)
- **max_hold_bars**: 78 (adjusted from 50)
- **Actual trades**: 11 over full backtest (optimizer converged on this)
- **Win rate**: 63.6% (7W/4L)
- **Best accuracy zone**: 16-50 bar trades (59.2%)

## What Worked
- Binary mode (+8.2%) — too few rows for 3-class
- LR=0.234, leaves=5 — Optuna found aggressive simple model
- Lean mode (2587 features) — dropped redundant TA
- SAR-numerology hybrids — AlphaNumetrix approach validated
- Calendar doy_sin = TOP esoteric (gain=7.8)
- Jupiter-Saturn regime = #10 overall feature
- CPCV (8,2) = 28 paths, 75% train

## Hard Limits
- 819 rows cannot support >2600 features
- Cross gen adds noise (too few rows per cross)
- Short trades (2-15 bars) worse than random
- Only 11 tradable signals in 16 years

## Implemented Features
- month_sin/cos, month_digital_root ✅
- week_of_year_sin/cos, week_digital_root ✅
- quarter_sin/cos ✅
- halving_era, year_in_halving_cycle, weeks_since_halving ✅
- SAR-numerology (9 features) ✅
- Planetary week ruler ✅
- TA x TA crosses (13 features — need binarization fix) ⚠️
- Prime features — MISSING, only day_is_prime exists ❌
