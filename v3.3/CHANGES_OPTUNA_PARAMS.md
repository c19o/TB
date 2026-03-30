# Optuna Parameter Range Fixes — Rare Signal Protection

**Date:** 2026-03-30
**Source:** EXPERT_RARE_SIGNALS.md, EXPERT_DATASET_BUILD.md findings

## Changes

### 1. `feature_fraction_bynode` floor: 0.5 → 0.7
- **File:** `run_optuna_local.py`, `validate.py`
- **Why:** With `feature_fraction=0.7` (floor), effective rate was `0.5 * 0.7 = 0.35` — only 35% of features visible per node. Devastating for rare esoteric signals that fire 3-15 times.
- **Fix:** Floor raised to 0.7. Worst case now `0.7 * 0.7 = 0.49` — still aggressive but survivable.

### 2. `bagging_fraction` floor: 0.5 → 0.7
- **File:** `run_optuna_local.py`, `validate.py`
- **Why:** 50% row dropout on a 3-fire feature means P(included) = `0.5^3 = 12.5%` — feature invisible 87.5% of rounds. At 0.7: `0.7^3 = 34.3%` — still risky but 2.7x better.
- **Fix:** Floor raised to 0.7. Config default remains 0.95 (optimal for rare signals).

### 3. `min_gain_to_split` range: [0.5, 10.0] → [0.0, 5.0]
- **File:** `run_optuna_local.py`, `validate.py`
- **Why:** Floor of 0.5 blocks marginal rare splits where gain is small but real. Ceiling of 10.0 is unnecessarily high — gains > 5.0 are extremely rare and the model is better off deciding its own threshold. Lowering to 0.0 lets Optuna explore splits the old range would have silently blocked.
- **Fix:** Range changed to [0.0, 5.0]. Config default stays at 2.0.

### 4. `bin_construct_sample_cnt` = 5000
- **File:** `config.py` (V3_LGBM_PARAMS)
- **Why:** LightGBM default is 200,000 samples for bin construction. With binary/sparse features (only 2 distinct values), this is 40x overkill. Setting to 5,000 gives 10-30% Dataset construction speedup with zero accuracy impact.
- **Fix:** Added `bin_construct_sample_cnt: 5000` to V3_LGBM_PARAMS.

### 5. `lambda_l2` — noted but unchanged
- Config default is 3.0. This is 5x penalty on 3-fire features, but L2 shrinks proportionally (doesn't zero like L1). Optuna range [1e-4, 10.0] with log-scale concentrates mass near zero. No change needed — Optuna can find low values already.

## Validation
- `validate.py` constraints updated to match new floors/ceilings
- `feature_fraction_bynode >= 0.7` (was 0.5)
- `bagging_fraction >= 0.7` (was 0.5)
- `min_gain_to_split >= 0.0, <= 5.0` (was >= 0.5, no upper check)
