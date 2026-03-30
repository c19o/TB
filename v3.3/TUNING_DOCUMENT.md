# V3.3 Tuning Document — All Timeframes

## Triple Barrier Configuration

The barrier config determines label balance. HOLD-dominated labels = model learns "do nothing." Balanced labels = model learns directional signals.

**v3.2 1h proved this**: 2.0x ATR / 2.0x ATR / hold=24 → 44% LONG / 43% SHORT / 13% HOLD → **53.9% accuracy**
**v3.3 1d failed this**: 3.5x ATR / 1.5x ATR / hold=8 → 34% LONG / 15% SHORT / 51% HOLD → **37% directional accuracy**

### Per-TF Feature Strategy (LOCKED 2026-03-29)

The matrix thesis scales with DATA. More rows = more features add value. Fewer rows = features become noise.

| TF | Rows | Feature Strategy | Target Features | Cross Type |
|----|------|-----------------|----------------|-----------|
| **1w** | 1,158 | **Base features ONLY** | ~380 | **NONE** (skip crosses) |
| **1d** | 5,733 | Base + **month** crosses | ~3.4M | Month bins (12 bins, not 365 DOY) |
| **4h** | 23,237 | Base + **DOY** crosses | ~5.56M | Full DOY (64 samples/bin) |
| **1h** | 75,406 | Base + DOY crosses | ~6M+ | Full DOY |
| **15m** | 293,980 | Base + DOY crosses | ~10M+ | Full DOY |

**Why month crosses for 1d**: DOY crossing gives 996K features per cross group but only 15-16 samples/DOY bin. Month gives 43K features with 478 samples/bin. mlogloss jumped from 0.998 (DOY) to 0.753 (month).

**Why base-only for 1w**: 1158 rows / 380 features = 3:1 ratio. Any crosses would push ratio below 1:1 (noise dominates signal). Puell multiple was #1 feature — macro/cycle features carry 1w.

**1w feature selection**: Only keep features relevant to weekly/monthly timeframe:
- Macro correlations (DXY, SPY, GLD, TLT, rates, BTC dominance)
- Big astro (lunar cycles, eclipse seasons, retrogrades, equinox/solstice)
- Long-period MAs (SMA50, SMA100, SMA200, golden/death cross)
- Space weather (Kp index, solar cycles)
- Numerology/sacred geometry (fibonacci days, halving cycles)
- Regime/HMM features
- **DROP**: All short-period TA (RSI_14, stoch, MACD, BBands, CCI, williams_r, ADX)
- **DROP**: All short-period MAs (SMA5, SMA10, SMA20, EMA5-20)

### Rule: hold_bars Must Match Actual Trade Duration

| TF | Candle Duration | Trade Duration | hold_bars Range |
|----|----------------|----------------|-----------------|
| 1w | 1 week | 2.5 months to 1.8 years | **11-94 bars** |
| 1d | 1 day | 6 weeks to 5 months | **42-144 bars** |
| 4h | 4 hours | 40 hrs to 3.5 weeks (some to 23 days) | **10-140 bars** |
| 1h | 1 hour | 7 hrs to 4 days (most ~30 hrs) | **7-100 bars** |
| 15m | 15 min | 1.75 hrs to 10 hrs | **7-40 bars** |

### LOCKED Config (Tested on Real Data, 2026-03-29)

All barrier configs tested on real BTC data and label distributions verified before training.

```python
TRIPLE_BARRIER_CONFIG = {
    '15m': {'tp_atr_mult': 1.0, 'sl_atr_mult': 1.0, 'max_hold_bars': 24},   # Tight barriers, ~6 hrs hold
    '1h':  {'tp_atr_mult': 1.2, 'sl_atr_mult': 1.2, 'max_hold_bars': 48},   # ~2 days hold
    '4h':  {'tp_atr_mult': 1.5, 'sl_atr_mult': 1.5, 'max_hold_bars': 72},   # ~12 days hold, 51/49 LONG/SHORT balance
    '1d':  {'tp_atr_mult': 2.0, 'sl_atr_mult': 2.0, 'max_hold_bars': 90},   # ~3 months hold
    '1w':  {'tp_atr_mult': 2.5, 'sl_atr_mult': 2.5, 'max_hold_bars': 50},   # ~1 year hold
}
```

### Class Weights (LOCKED)
- **1w**: SHORT = 3x weight
- **1d**: SHORT = 3x weight
- **4h**: SHORT = 2x weight (naturally balanced labels)
- **1h/15m**: TBD (pending training)

---

## Optuna Hyperparameter Ranges

### Per-TF num_leaves Scaling (LOCKED 2026-03-29)

| TF | num_leaves | min_data_in_leaf | Rationale |
|----|-----------|-----------------|-----------|
| 1w | **7** | 30 | v3.2 best was 7. Only 1158 rows. |
| 1d | **15** | 50 | 5733 rows, 3.4M features. Heavy regularization needed. |
| 4h | **31** | 20 | 23K rows, 5.56M features. Standard complexity. |
| 1h | **63** | 15 | 75K rows. Moderate complexity. |
| 15m | **127** | 15 | 294K rows. Full complexity. |

### CRITICAL: feature_fraction Range (CORRECTED 2026-03-29)

**feature_fraction was silently killing the matrix edge.**

| Param | OLD (broken) | NEW (corrected) | Why |
|-------|-------------|-----------------|-----|
| feature_fraction | 0.005 – 0.05 | **0.7 – 1.0** | Low = only 150 EFB bundles/tree = rare signals excluded |
| feature_fraction_bynode | 0.05 – 0.3 | **0.5 – 1.0** | Low = <10 bundles per node split = only TA survives |
| V3_LGBM_PARAMS default | 0.1 | **0.9** | Seed trials must see nearly all features |

**What happened**: Optuna optimized mlogloss by finding that IGNORING rare esoteric features gives better loss. With feature_fraction=0.01 and EFB bundling 3.4M→15K bundles, each tree only saw 150 bundles. The 1d "best" trial (mlogloss=0.753) and 4h warm-start (0.657) were achieved WITHOUT the matrix edge — just common TA. This is a **silent philosophy violation**.

**After EFB**: feature_fraction operates on **bundles, not raw features**. 0.9 × 23K bundles = 20,700 bundles per tree. This preserves rare signals while still providing de-correlation.

**Perplexity confirmed**: For millions of sparse binary features where rare signals ARE the edge, feature_fraction must be >= 0.7. With EFB bundling to 23K, feature_fraction=1.0 fits easily in 32GB GPU VRAM (~200-500MB per trial).

**RULE**: NEVER allow feature_fraction below 0.7 in the Optuna search range. If Optuna wants regularization, use lambda_l1/l2 and min_gain_to_split instead — those don't systematically exclude features.

### Previous v3.3 1d (produced 51.2% accuracy, PBO 0.20 — OBSOLETE, feature_fraction was broken)

Winner params: num_leaves=36, min_data_in_leaf=3, feature_fraction=0.023, lambda_l2=13.5, max_depth=11

### v3.2 vs v3.3 Comparison (WHY v3.2 was better)

| Param | v3.2 (7 leaves, 73.5%) | v3.3 (36 leaves, 51.2%) |
|-------|----------------------|----------------------|
| num_leaves | **7** | 36 (5x more complex) |
| min_data_in_leaf | **13** | 3 (memorization risk) |
| lambda_l1 | **0.71** | 0.048 (15x weaker) |
| bagging_fraction | **0.95** | 0.56 |

**v3.2 used a SIMPLE model. v3.3 went complex and overfitted.** The matrix signals are subtle — shallow trees with strong regularization capture the real patterns without memorizing noise.

### Recommended Changes

| Param | Current Range | New Range | Reason |
|-------|-------------|-----------|--------|
| num_leaves | [15, 127] | **[4, 63]** | v3.2 best was 7. Shallower trees = less overfitting |
| min_data_in_leaf | [3, 50] | **[8, 50]** | v3.2 used 13. Prevents memorizing rare noise |
| lambda_l1 | [0.01, 5] | **[0.1, 10]** | v3.2 used 0.71. Stronger L1 sparsity |
| lambda_l2 | [0.01, 20] | **[1.0, 100]** | Stronger regularization for millions of features |
| max_depth | [6, 15] | **[3, 12]** | Cap depth to prevent memorization |
| feature_fraction | [0.01, 0.3] | **[0.005, 0.1]** | Less features per tree = more diversity |
| bagging_fraction | [0.5, 1.0] | **[0.7, 1.0]** | v3.2 used 0.95, more row coverage |
| extra_trees | not searched | **[True, False]** | Random split thresholds reduce variance |

---

## Critical Code Fixes

### 1. Remove Dense Conversion (DONE — Unblocked 4h/1h)
- `run_optuna_local.py`: removed `.toarray()` — trains on sparse CSR directly
- `ml_multi_tf.py`: dense conversion removed
- LightGBM accepts scipy sparse CSR natively

### 2. Add force_col_wise=True (Speed)
- `config.py` V3_LGBM_PARAMS: add `'force_col_wise': True`
- Bypasses PushDataToMultiValBin bug, linear thread scaling

### 3. Replace run_tee (DONE)
- `cloud_run_tf.py`: Python Popen drain loop replaces shell tee
- Prevents false FAIL reports on long runs

### 4. Lower Confidence Thresholds (Trade Optimizer)
- `exhaustive_optimizer.py` all TF grids: `[0.45, 0.90]` → `[0.34, 0.55]`
- Add probability margin dimension `[0.02, 0.20]`

### 5. Class Weighting
- Ensure `class_weight='balanced'` is applied in BOTH Optuna path AND ml_multi_tf.py training
- Forces model to learn directional signals instead of defaulting to HOLD

---

## New Cycle Detection Features (2026-03-29)

Built from EXISTING data — no new downloads needed:

| Feature | Source | What It Detects |
|---------|--------|----------------|
| `hash_ribbon_30_60` | hash_rate MAs | Miner capitulation (entry signal) |
| `hash_ribbon_buy` | 30MA crossing above 60MA | Hash ribbon buy signal |
| `hash_ribbon_capitulation` | 30MA < 60MA | Active miner stress |
| `puell_multiple` | miners_revenue / 365d MA | Revenue vs trend (>4 = top, <0.5 = bottom) |
| `whale_vol_zscore_26w` | whale_volume 26-week z-score | Accumulation/distribution extremes |
| `vix_stress` / `vix_complacency` | VIX > 30 / < 15 | Macro risk regime |
| `vix_zscore_52w` | VIX 52-week z-score | VIX relative to yearly context |
| `yield_curve_proxy` | US10Y - TLT trend | Rate environment |
| `macro_risk_on_score` | SPX + DXY + VIX composite | Risk-on/off regime (0-3) |
| `price_200w_ratio` | close / SMA200 | Cycle position (>2.5 = overheated) |
| `btc_spx_corr_13w_chg` | Correlation trend | Decoupling/recoupling signals |

**Why these matter for SHORT**: The model had zero SHORT signal because existing features encode uptrend naturally (MAs, macro correlations). These new features explicitly encode cycle TOPS and STRESS — giving the model data points that historically precede drawdowns.

**Still missing (need data download)**: MVRV ratio, NUPL, SOPR, exchange reserves, stock-to-flow. These are the #1 cycle-top detectors in academic research. Priority for next session.

## Ongoing Tuning Strategy (Apply to ALL TFs)

### Step 1: Label Balance
- Test barrier configs on real data BEFORE training
- Target: <20% HOLD, balanced LONG/SHORT
- Hold bars must match actual trade duration for the TF

### Step 2: Feature Appropriateness
- 1w: Base features only (~380, no crosses). Puell multiple = #1 feature.
- 1d: Base + month crosses (~3.4M features). DOY too sparse (15 samples/bin).
- 4h+: Base + DOY crosses (5.56M+ features). DOY meaningful at 64+ samples/bin.
- Per-TF feature count should be < 3x row count for base features

### Step 3: Model Complexity (Scale with EFB-Adjusted Row:Feature Ratio)

| TF | Rows | EFB Bundles | Effective Ratio | num_leaves | Complexity |
|----|------|------------|-----------------|------------|------------|
| 1w | 1,158 | 400 (no EFB) | 2.9:1 | **3-7** | Minimal |
| 1d | 5,733 | ~20,740 | 0.28:1 | **7-15** | Heavy regularization |
| 4h | 23,237 | ~25,193 | 0.92:1 | **15-31** | Standard |
| 1h | 75,406 | ~19,717 | 3.8:1 | **31-63** | Moderate |
| 15m | 293,980 | ~42,870 | 6.9:1 | **63-127** | Full complexity |

- v3.2 proved: simpler models (num_leaves=7) outperform complex ones on small data
- 1d is weakest ratio (0.28:1) — most at risk of overfitting, needs feature_fraction=0.005
- 15m has best ratio (6.9:1) — can handle deepest trees
- EFB compresses 5M binary crosses into ~20K bundles (254 per bundle at max_bin=255)

### Step 4: Class Weighting
- If model predicts 0 of a class, add class_weight (SHORT=3.0)
- Check raw probabilities before and after — if max prob for missing class < 0.20, features are the issue not weights

### Step 5: Confidence Calibration
- Always check accuracy vs confidence after training
- High-confidence HOLD ≠ good model (it means the model learned to dodge)
- Target: directional accuracy > 50% at conf >= 0.45

## Training Order (Updated 2026-03-29)

1. ~~Fix barriers + sparse + force_col_wise + run_tee~~ **DONE**
2. ~~1w: base features only, 380 features~~ **DONE** — 57.5% CPCV, 80.3% LONG@0.80
3. **1d: Optuna Phase 1 running** (Norway m:29579, 17/25 trials, best mlogloss 0.753)
4. **4h: Optuna Phase 1 running** (NJ m:32893, 15 warm-start trials, 5.56M features)
5. 1h: after 4h verified (need 768GB+ machine)
6. 15m: after 1h verified (need 1TB+ machine)

---

## Key Metrics to Track Per TF

- **Label balance**: LONG% / SHORT% / HOLD% — target <25% HOLD
- **Directional accuracy**: accuracy on LONG+SHORT predictions only — target >45%
- **Overall CPCV accuracy**: target >50%
- **PBO**: target <0.15
- **Confidence calibration**: accuracy at conf >= 0.50, >= 0.60
- **Trade count**: optimizer must find >50 trades in 200 trials
