# EXPERT: Financial ML & CPCV Specialist

**Date:** 2026-03-30
**Context:** BTC trading, LightGBM, 2-10M sparse binary cross-features, CPCV validation, triple-barrier labels, rare signals (fire 3-20 times)

---

## 1. CPCV Speed Optimization: Can We Reduce 15 Folds to 10?

### Answer: YES, but with conditions

Reducing from 15 to 10 chronological groups is safe **if each test segment still contains enough labeled events**. The critical constraint is not fold count -- it is **positives per test path**.

**Key findings:**
- With signals firing only 3-20 times, many CPCV test paths may contain 0-2 positive events regardless of fold count. This makes the validation distribution dominated by sampling noise, not model skill.
- The practical rule: choose the number of chronological groups so each held-out test segment contains **multiple labeled events** (minimum ~5 positives per path).
- Fewer groups = larger test blocks = more events per block = lower variance per path estimate.

**Recommendation for our pipeline:**
- **1w/1d (fewer rows, rarer signals):** 3-5 groups, sampled paths (not exhaustive)
- **4h (moderate rows):** 5-8 groups
- **1h/15m (more rows, more events):** 8-10 groups maximum
- Never use more groups than warranted by event density

**Speed gains from 15 -> 10 folds:**
- Exhaustive CPCV paths: C(15,2) = 105 vs C(10,2) = 45 -- a **57% reduction** in paths
- With sampled paths (recommended): sample 20-30 representative paths regardless of group count

### Faster Alternatives to Full CPCV

**Two-stage approach (recommended):**
1. **Stage 1 -- Fast prescreening:** Single purged walk-forward or 10-15 sampled CPCV paths. Eliminates 90% of hyperparameter candidates cheaply.
2. **Stage 2 -- Full CPCV:** Run only on the shortlisted configs (top 3-5 from Stage 1).

**Additional speed wins (higher ROI than fold reduction):**
- Cache purge masks, embargo masks, event spans, and fold membership once -- reuse across all Optuna trials
- Precompute triple-barrier event end-times and overlap matrices once
- Parallelize over CPCV paths (they are independent)
- Cut effective feature count before CV via EFB / frequency thresholds / near-duplicate removal
- LightGBM EFB already handles this for sparse binary features -- keep it enabled

**Source:** [Combinatorial Purged Cross Validation for Optimization](https://www.quantbeckman.com/p/with-code-combinatorial-purged-cross), [Purged cross-validation - Wikipedia](https://en.wikipedia.org/wiki/Purged_cross-validation)

---

## 2. Walk-Forward vs CPCV: Which to Use?

### Answer: CPCV for research, Walk-Forward for deployment validation

| Layer | Purpose | When |
|---|---|---|
| **CPCV** | Hyperparameter tuning, feature pruning, threshold selection, model comparison | Research phase |
| **Walk-Forward** | Final frozen-process validation before paper/live | Post-selection |
| **Holdout/Paper** | Untouched approval gate | Before live capital |

**Why CPCV wins for model selection:**
- Produces a **distribution** of OOS outcomes, not one fragile estimate
- Reduces path dependence (one walk-forward path can be dominated by one regime)
- Explicitly handles triple-barrier label leakage via purging and embargoing
- Better at false-discovery control under non-stationarity

**Why walk-forward still matters:**
- Best approximation of live operations (train, freeze, trade, roll)
- Tests retraining schedule and operational process
- Industry standard for realistic trading simulation

**For our pipeline:**
- Use CPCV inside Optuna for hyperparameter search
- After final model selection, run single untouched walk-forward with exact retrain cadence, execution lag, fees, funding, slippage
- Never skip walk-forward -- CPCV proves the model family is real; walk-forward proves the process is livable

**Critical note:** With only 3-20 signal firings per window, walk-forward alone has catastrophically low statistical power. The 2025 walk-forward framework paper reported only 12% power with 34 folds in a much more active setup.

**Source:** [Walk-forward validation discussion](https://www.reddit.com/r/algotrading/comments/1rsqzj8/is_walkforward_validation_actually_worth_the/), [Backtest overfitting comparison](https://www.sciencedirect.com/science/article/abs/pii/S0950705124011110)

---

## 3. CPCV Fold Reduction Impact on Calibration

### Answer: Calibration degrades, but we should use scores as ranks anyway

**Core finding:** Reducing folds makes confidence calibration worse because:
- Fewer independent OOS score-label pairs for calibration estimation
- Fewer positives available for any post-hoc calibrator
- Calibration errors (Brier, log-loss, ECE) become dominated by tiny positive counts

**With our signal density (3-20 fires):**
- Many CPCV folds will contain 0-2 positives regardless
- Empirical mapping from LightGBM score to realized hit rate becomes jagged and non-identifiable
- Post-hoc calibration methods (Platt, isotonic) can actually **worsen** proper scoring performance

**Practical guidance:**
- **Do NOT treat raw `predict_proba` as trustworthy probabilities** -- treat them as monotonic rank scores
- If calibration is needed for bet sizing: use **Beta calibration or Venn-Abers** (outperformed Platt/isotonic in 2026 large-scale tabular study)
- Evaluate on **pooled CPCV OOS predictions** (concatenate all path predictions), not per-fold
- If any fold has fewer than ~5 positive events, calibration estimates from that fold are pure noise

**For our pipeline:**
- Primary use of model output: **trade filter/ranker** (take/skip decision)
- Secondary use: position sizing via calibrated score buckets (only after pooled OOS calibration validation)
- Metric hierarchy: PR-AUC > precision@k > recall at fixed FP budget > calibration error

**Source:** [Backtest overfitting comparison](https://www.sciencedirect.com/science/article/abs/pii/S0950705124011110), [Post-hoc calibration study](https://arxiv.org/html/2601.19944v1)

---

## 4. Meta-Labeling (M2) Architecture for Our Signal Density

### Answer: Small, heavily regularized LightGBM -- NOT deep/wide

**Core principle:** Size the model off **positives per training fold**, not total features or rows. With 3-20 signal fires in some windows, many CPCV folds live in the "extremely data-starved" bucket.

### Recommended Architecture

| Component | Value | Rationale |
|---|---|---|
| Model family | LightGBM `gbdt` on CPU sparse matrix | Handles CSR natively, EFB for sparse |
| `num_leaves` | 4-16 | Tiny trees prevent memorization |
| `max_depth` | 2-4 | Matches leaf constraint |
| `learning_rate` | 0.02-0.05 | Slow learning with early stopping |
| `n_estimators` | High cap (2000+) | Rely on early stopping |
| `min_data_in_leaf` | 100-1000+ | Far above default 20; tune by fold-positive count |
| `min_sum_hessian_in_leaf` | >> 1e-3 | Above default |
| `feature_fraction` | **0.7-1.0** (per our constraint) | CRITICAL: >= 0.7 to preserve rare cross signals |
| `lambda_l1`, `lambda_l2` | Nonzero | L1/L2 regularization |
| `min_gain_to_split` | > 0 | Suppress marginal memorization |
| `path_smooth` | > 0 if leaves noisy | Smoothing |
| `scale_pos_weight` | Set per fold prevalence | Better control than `is_unbalance` |
| Early stopping | Yes, `first_metric_only=true` | On PR-AUC or precision |
| Sparse handling | EFB enabled, sparse optimization on | Core LightGBM advantage |

### Sizing Ladder by Fold Positives

| Fold Positives | num_leaves | max_depth | min_data_in_leaf |
|---|---|---|---|
| < 50 | 4-8 | 2-3 | 500+ |
| 50-150 | 8-16 | 3-4 | 200-500 |
| > 150 | 16-31 | 4-6 | 100-200 |

### Metric Stack for Meta-Labeling

1. **Primary:** PR-AUC / average precision (focuses on minority class)
2. **Secondary:** Precision@k, recall at fixed false-positive budget
3. **Tertiary:** Calibration error (only if mapping probabilities to size)
4. **Never use:** Accuracy, ROC-AUC alone (inflated by true negatives)

### Imbalance Handling

- Use `scale_pos_weight` (not `is_unbalance`) -- gives tighter control across CPCV folds with varying prevalence
- WARNING: imbalance weighting distorts raw probabilities. Train for ranking/filtering first, calibrate separately if needed for sizing.

**Note on feature_fraction:** Perplexity research suggested 0.05-0.3 for huge cross spaces. However, per our established constraint (`feature_fraction >= 0.7`), we keep this floor to avoid killing rare esoteric cross signals that fire infrequently. The regularization burden shifts to other parameters (leaf constraints, L1/L2, min_gain_to_split).

**Source:** [Meta Labeling for Algorithmic Trading](https://www.reddit.com/r/algotrading/comments/1lnm48w/meta_labeling_for_algorithmic_trading_how_to/), [LightGBM Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html)

---

## 5. Optimal Purge/Embargo Per Timeframe

### Answer: Tie to triple-barrier holding horizon, not arbitrary constants

**Core principle from de Prado:** Purge length should equal the **maximum triple-barrier label horizon** (the furthest t1 endpoint from any training observation that overlaps with the test period). Embargo should cover **post-event autocorrelation decay**.

### Recommended Values Per Timeframe

| Timeframe | Typical Barrier Horizon | Purge (bars) | Embargo (bars) | Embargo (time) |
|---|---|---|---|---|
| **1w** | 4-8 weeks | 8 bars | 2-3 bars | 2-3 weeks |
| **1d** | 5-20 days | 20 bars | 3-5 bars | 3-5 days |
| **4h** | 2-10 days | 60 bars (~10d) | 6-12 bars | 1-2 days |
| **1h** | 1-5 days | 120 bars (~5d) | 12-24 bars | 12-24 hours |
| **15m** | 4-48 hours | 192 bars (~2d) | 24-48 bars | 6-12 hours |

### Purge Sizing Logic

- Purge = `max(t1) - t_test_start` for all training observations whose triple-barrier label endpoint `t1` extends into the test period
- In practice: set purge to the **maximum holding period** of your triple-barrier configuration for that timeframe
- Event-based purging (keyed to each label's end time `t1`) is superior to naive bar-based purging

### Embargo Sizing Logic

- Embargo covers the **post-event information decay** period
- For BTC: autocorrelation in returns decays within 1-3 bars for most timeframes, but feature autocorrelation (especially momentum/trend features) can persist longer
- Conservative rule: embargo = 0.5x to 1x the purge window
- Aggressive rule: embargo = 2-5% of total dataset length (de Prado's original suggestion)

### Implementation Notes

- Purge/embargo must be **event-based**, not row-index-based, because triple-barrier labels have variable-length holding periods
- Each label's `t1` (barrier hit time) determines its purge boundary, not the label's start time
- Pre-compute all `t1` values once and cache the overlap matrix for all CPCV folds
- For our pipeline: `validate.py` should enforce purge >= max_barrier_horizon per TF

**Source:** [Purging and Embargo](https://abouttrading.substack.com/p/purging-and-embargo-two-tricks-that), [Cross Validation in Finance](https://blog.quantinsti.com/cross-validation-embargo-purging-combinatorial/), [The Hidden Flaw in Financial ML -- Label Concurrency](https://www.mql5.com/en/articles/19850)

---

## Summary: Actionable Changes for v3.3

### Immediate (no code risk)
1. **Reduce CPCV groups** per TF based on event density table above
2. **Sample CPCV paths** (20-30) instead of exhaustive enumeration
3. **Cache purge/embargo masks** once per Optuna study, not per trial
4. **Switch primary metric** to PR-AUC for Optuna optimization

### Short-term (requires validation)
5. **Implement two-stage tuning:** fast walk-forward prescreening -> CPCV on shortlist
6. **Set purge/embargo per TF** using the holding-horizon-based table above
7. **Add pooled OOS prediction collection** across CPCV paths for calibration assessment

### M2 Meta-Labeling (future)
8. Use small LightGBM (num_leaves 4-16, max_depth 2-4)
9. Size by fold-positive count, not global dataset
10. Train as trade filter/ranker first, calibrate for sizing separately
11. Use Beta calibration or Venn-Abers, not Platt/isotonic

### Constraints to Preserve
- `feature_fraction >= 0.7` (non-negotiable: protects rare cross signals)
- Never row-partition or subsample (kills rare signals)
- EFB enabled (core LightGBM sparse advantage)
- Sparse CSR format preserved (int64 indptr for NNZ > 2^31)
