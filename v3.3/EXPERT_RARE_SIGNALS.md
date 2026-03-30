# EXPERT: Rare Signal Preservation in LightGBM

**Context:** 2-10M sparse binary features, 90%+ fire only 3-20 times. These rare cross-signals are the alpha. No filtering allowed.

**Sources:** LightGBM 4.6 official docs (Parameters, Parameters-Tuning, Features, Advanced-Topics), NeurIPS LightGBM paper, GitHub issues #2258/#3816/#5200/#5205/#5492/#622, Perplexity deep research 2026-03-30.

---

## 1. THE CORE PROBLEM: ADMISSION vs AMPLITUDE

Rare signals face two distinct threats:

### A. Admission Gates (can the feature even split?)
These **structurally block** rare features from ever appearing in a tree:

| Gate | How it kills rare signals |
|------|--------------------------|
| `feature_pre_filter=True` (default) | Filters "unsplittable" features at Dataset construction. A 5-fire feature with `min_data_in_leaf=10` is permanently removed before training starts. |
| `min_data_in_leaf` | A split isolating 3-20 rows fails if this exceeds the positive count. The #1 rare signal killer. |
| `min_sum_hessian_in_leaf` | Second gate: even if row count passes, low Hessian mass blocks the split. For binary classification, Hessian = p*(1-p) per row, so 3 rows near 0.5 give ~0.75 total. |
| `min_gain_to_split` | Any positive threshold makes marginal rare splits less likely. Rare features produce small absolute gain. |

### B. Amplitude Controls (how much does the split contribute?)
These **shrink** the leaf output after a valid split:

| Control | Effect on rare leaves |
|---------|----------------------|
| `lambda_l1` | L1 drives small leaf outputs to zero. Worst for rare signals -- acts like a kill switch on weak leaves. |
| `lambda_l2` | L2 shrinks continuously. Less destructive than L1 but still disproportionately affects tiny-support leaves. |
| `path_smooth` | Pulls small-node weights toward parent. Explicitly designed for few-sample leaves. |
| `learning_rate` | Uniformly scales all updates. Not biased against rare signals. |

**Key insight:** Fix admission first. Amplitude tuning is pointless if the feature can never split.

---

## 2. KEY QUESTIONS ANSWERED

### Q: Minimum fires needed for LightGBM to learn a signal?

**Answer: Theoretically 1, practically 3.**

- With `min_data_in_leaf=1`, a feature firing once can create a valid split (1 row in one child, N-1 in the other).
- With `min_data_in_leaf=3`, you need at least 3 fires on the same side of any prior split in the tree path.
- The real constraint is statistical: a 3-fire feature needs concentrated gradient signal to win against millions of competitors. With binary classification, 3 rows contribute ~0.75 Hessian mass -- enough for `min_sum_hessian_in_leaf=1e-3` but not for higher values.
- LightGBM confirmed (issue #2258): `min_data_in_leaf=10` makes a 5-occurrence feature unsplittable.

### Q: Optimal feature_fraction for 2M features where 90% fire <20 times?

**Answer: Start at 1.0. Only reduce if training cost is unbearable.**

The math: `feature_fraction=f` gives each feature probability `f` of being considered per tree. Probability of appearing in at least 1 of T trees = `1-(1-f)^T`.

| feature_fraction | Trees=1000 | Trees=5000 | Expected appearances/5000 |
|-----------------|------------|------------|---------------------------|
| 1.0 | 100% | 100% | 5000 |
| 0.9 | ~100% | ~100% | 4500 |
| 0.7 | ~100% | ~100% | 3500 |
| 0.1 | ~100% | ~100% | 500 |
| 0.01 | 99.996% | ~100% | 50 |
| 0.001 | 63.2% | 99.3% | 5 |

**Critical:** `feature_fraction` is uniform random -- it does NOT penalize rare features more than common ones. But each missed tree is a missed opportunity for a feature that already struggles to win splits. With 5000 rounds at f=0.7, a rare feature gets 3500 chances instead of 5000 -- that's 1500 fewer opportunities for residuals to align with its 3-20 fires.

**NEVER use `feature_fraction_bynode` below 0.8.** It compounds with tree-level sampling: effective per-node fraction = `feature_fraction * feature_fraction_bynode`. At 0.7 * 0.5 = 0.35 effective, rare features lose 65% of split opportunities at every node.

**Current v3.3:** `feature_fraction=0.9`, `feature_fraction_bynode=0.8` (effective 0.72 per node). Floor at 0.7 enforced. This is acceptable but conservative. Research suggests 1.0/1.0 for maximum rare signal preservation.

### Q: How many rounds to address rare signal residuals?

**Answer: Far more than typical. Rare signals appear LATE.**

LightGBM grows leaf-wise, picking the leaf with maximum delta loss. This means:
1. **Early rounds (1-500):** Common patterns dominate. High-frequency features model the bulk of residuals.
2. **Mid rounds (500-2000):** Intermediate features. Common residual structure removed.
3. **Late rounds (2000-5000+):** Residuals become "pure" enough that a 7-fire feature can finally win a split against millions of competitors.

**Mental model:** Boosting is a queue. Common patterns go first. Rare alpha waits until residuals on those 3-20 rows become concentrated enough to produce competitive gain.

**Implications:**
- Early stopping patience must be LARGE (500-1000 rounds, not 50-100).
- Validation metric may plateau on common-signal performance while rare-signal learning hasn't started.
- With `learning_rate=0.03` and 5000 rounds, rare features realistically get addressed in rounds 2000-5000.
- Row/column subsampling makes the queue LONGER because rare rows/features are missing from many trees.

### Q: Better loss function that upweights rare feature splits?

**Answer: No native LightGBM loss function does this. But there are workarounds.**

LightGBM's split criterion is gain = `(sum_gradient)^2 / (sum_hessian + lambda)`. This is sample-count-agnostic in principle but sample-count-dependent in practice because sum_gradient and sum_hessian scale with row count.

**Options:**
1. **Custom objective with amplified gradients on rare-pattern rows:** If you can identify rows where rare features fire, amplify their gradient/hessian contribution. Complex and risky.
2. **`extra_trees=True`:** Randomizes split thresholds instead of optimizing them. Can help rare binary features because the threshold is random anyway (0/1 only has one split point). Reduces overfitting without blocking rare splits.
3. **`forcedsplits_filename`:** JSON file that forces specific features to split at specific tree depths. If you know which rare features matter, you can guarantee they get used early. Useful for prior knowledge.
4. **CEGB penalties:** `cegb_penalty_split`, `cegb_penalty_feature_lazy`, `cegb_penalty_feature_coupled` -- these are the OPPOSITE of what you want. They penalize feature usage. Keep at 0.

---

## 3. CURRENT V3.3 PARAMETER AUDIT

### config.py V3_LGBM_PARAMS (current values)

| Parameter | Current | Assessment | Recommendation |
|-----------|---------|------------|----------------|
| `feature_pre_filter` | `False` | CORRECT. Prevents silent elimination. | Keep. |
| `min_data_in_leaf` | 3 | GOOD. Allows 3-fire features. Could be 1-2 for max preservation. | Consider 1-2 for 1w/1d (few rows). Keep 3 for 4h/1h/15m. |
| `min_gain_to_split` | 2.0 | BORDERLINE HIGH. Any positive value blocks marginal rare splits. | Reduce to 0.5-1.0 or let Optuna explore [0.0, 5.0]. |
| `lambda_l1` | 0.5 | SLIGHTLY HIGH for rare signals. L1 drives weak leaf outputs to zero. | Reduce to 0.1 or use log-scale [1e-4, 1.0]. |
| `lambda_l2` | 3.0 | MODERATE. Less destructive than L1 but still shrinks tiny-support leaves. | Acceptable. Let Optuna explore [0.1, 10.0]. |
| `feature_fraction` | 0.9 | ACCEPTABLE. 90% per-tree inclusion. | Could be 1.0 for max preservation. |
| `feature_fraction_bynode` | 0.8 | CAUTION. Effective = 0.9*0.8=0.72 per node. | Raise to 0.9+ or 1.0. |
| `path_smooth` | 0.5 | GOOD. Gentle regularization for small leaves. 3.2% dampening at n=15. | Keep. Sweet spot for rare signals. |
| `bagging_fraction` | 0.8 | CONCERNING. 20% row dropout means rare 3-fire features lose rows. | Raise to 0.9-1.0. |
| `bagging_freq` | 1 | Active with above. | Set to 0 if bagging_fraction=1.0. |
| `learning_rate` | 0.03 | GOOD. Low rate + many rounds = rare signals get their turn. | Keep. |
| `num_iterations` | 5000 | GOOD. Enough rounds for late rare signal learning. | Keep or increase to 8000 with early stopping. |

### run_optuna_local.py Search Space (current ranges)

| Parameter | Current Range | Assessment | Recommendation |
|-----------|--------------|------------|----------------|
| `min_data_in_leaf` | [max(3, tf), 15] | GOOD floor, ceiling acceptable. | Keep. |
| `feature_fraction` | [0.7, 1.0] | GOOD floor at 0.7. | Keep. |
| `feature_fraction_bynode` | [0.5, 1.0] | FLOOR TOO LOW. 0.5 * 0.7 = 0.35 effective. | Raise floor to 0.7. |
| `bagging_fraction` | [0.5, 1.0] | FLOOR TOO LOW. Drops 50% of rows. Rare features devastated. | Raise floor to 0.7. |
| `lambda_l1` | [1e-4, 4.0] log | CEILING OK. Log-scale keeps mass near zero. | Keep or reduce ceiling to 2.0. |
| `lambda_l2` | [1e-4, 10.0] log | ACCEPTABLE. | Keep. |
| `min_gain_to_split` | [0.5, 10.0] | FLOOR TOO HIGH. Should include 0.0. CEILING TOO HIGH. 10.0 kills rare signals. | Change to [0.0, 5.0]. |
| `max_depth` | [2, 8] | GOOD. Deeper trees can isolate rare subpopulations. | Keep. |

---

## 4. THE INFORMATION GAIN PROBLEM

For a binary feature with k=10 fires out of N=10000 rows, the maximum possible gain is:

```
gain = (sum_gradients_left)^2 / (sum_hessians_left + lambda)
     + (sum_gradients_right)^2 / (sum_hessians_right + lambda)
     - (sum_gradients_parent)^2 / (sum_hessians_parent + lambda)
```

With binary classification (logloss), the Hessian per row is p*(1-p). At initialization (p~0.5), each row contributes ~0.25 Hessian.

- **10-fire feature:** Left child has ~2.5 Hessian mass. With `lambda_l2=3.0`, denominator = 5.5. Gain is modest.
- **3-fire feature:** Left child has ~0.75 Hessian mass. With `lambda_l2=3.0`, denominator = 3.75. The lambda DOMINATES the Hessian. Gain is heavily suppressed.

**This is why `lambda_l2=3.0` disproportionately hurts rare features:** when Hessian mass is 0.75, adding 3.0 to the denominator multiplies the effective regularization by 5x compared to a 1000-fire feature (Hessian=250, lambda adds only 1.2%).

The same logic applies to `min_gain_to_split=2.0`: a 3-fire feature might produce gain of 0.5-1.5, which gets blocked entirely.

---

## 5. HISTOGRAM BINNING FOR BINARY FEATURES

Binary features (0/1) need only 2 histogram bins. LightGBM handles this efficiently:
- `max_bin=255` (default) works fine -- binary features use 2 bins regardless.
- `is_enable_sparse=True` skips zero-valued bins during histogram construction: O(2 * #non_zero) instead of O(#data).
- `enable_bundle=True` (EFB) bundles mutually exclusive sparse features. Critical for 2M+ features.
- `bin_construct_sample_cnt`: Default 200000. For very sparse data, increase to 1000000+ to avoid poor bin boundaries on the non-binary features.

**Important:** `zero_as_missing=False` (default). Your zeros mean "feature absent," not "missing." Changing this would corrupt the entire feature space semantics.

---

## 6. RECOMMENDED PARAMETER CHANGES

### Priority 1: Immediate Optuna range fixes

```python
# run_optuna_local.py -- raise floors on sampling, lower ceilings on regularization
feature_fraction_bynode = trial.suggest_float('feature_fraction_bynode', 0.7, 1.0)  # was 0.5
bagging_fraction = trial.suggest_float('bagging_fraction', 0.7, 1.0)                # was 0.5
min_gain_to_split = trial.suggest_float('min_gain_to_split', 0.0, 5.0)              # was [0.5, 10.0]
```

### Priority 2: config.py default adjustments

```python
# These changes preserve more rare signals while keeping overfitting controls
"min_gain_to_split": 0.5,          # was 2.0 -- let Optuna find the right value from a lower starting point
"lambda_l1": 0.1,                  # was 0.5 -- L1 is the rare signal killer; keep near zero
"bagging_fraction": 0.9,           # was 0.8 -- preserve more rows for rare features
"bagging_freq": 1,                 # keep active but with higher fraction
"feature_fraction_bynode": 0.9,    # was 0.8 -- effective per-node = 0.9*0.9 = 0.81
```

### Priority 3: Experimental (validate before adopting)

```python
# Maximum rare signal preservation mode -- test on 1w first
"min_data_in_leaf": 1,             # absolute minimum for 1w (818 rows, features fire 3-5 times)
"lambda_l1": 0.0,                  # zero L1; rely on path_smooth + lambda_l2 only
"feature_fraction": 1.0,           # no column sampling
"feature_fraction_bynode": 1.0,    # no per-node sampling
"bagging_fraction": 1.0,           # no row sampling
"path_smooth": 1.0,                # slightly stronger smoothing to compensate
"extra_trees": True,               # random thresholds -- helps binary features
```

---

## 7. REGULARIZATION STRATEGY FOR RARE SIGNALS

**Correct order of regularization knobs (least to most destructive for rare signals):**

1. `max_depth` / `num_leaves` -- Limits tree complexity. Does NOT block rare splits, just limits how deep they appear. **Safe.**
2. `learning_rate` + early stopping -- Uniformly scales all updates. Not biased against rare features. **Safe.**
3. `path_smooth` -- Smooths small-node outputs toward parent. Gentle, explicitly designed for few-sample leaves. **Safe at 0.5-2.0.**
4. `lambda_l2` -- Continuous shrinkage. Disproportionately affects tiny-support leaves but doesn't zero them out. **Moderate risk.**
5. `bagging_fraction` -- Row dropout. Each dropped row is catastrophic for a 3-fire feature. **High risk below 0.9.**
6. `feature_fraction` / `feature_fraction_bynode` -- Column dropout. Linearly reduces split opportunities. **High risk below 0.7.**
7. `lambda_l1` -- Hard sparsity on leaf outputs. Can zero out weak rare-signal leaves entirely. **Dangerous above 1.0.**
8. `min_gain_to_split` -- Hard threshold. Rare features produce small absolute gain. **Dangerous above 2.0.**
9. `min_data_in_leaf` -- Structural blocker. Features with fewer fires than this value CANNOT SPLIT. **Most dangerous.**
10. `feature_pre_filter=True` -- Permanent elimination at Dataset construction. **Catastrophic. Must always be False.**

**Rule: Never regularize rare signals at the admission layer (9-10). Always prefer amplitude controls (1-4).**

---

## 8. LATE LEARNING DETECTION

### How to tell if rare signals are being learned:

1. **Feature importance by iteration range:**
   ```python
   # Train with callbacks that log feature importance every 500 rounds
   # Compare importance of known rare features at round 500 vs 2000 vs 5000
   # Rare features should appear/grow in later rounds
   ```

2. **Split count analysis:**
   ```python
   model.trees_to_dataframe()  # Get all splits
   # Count how many times each feature was used
   # Features with 3-20 fires should appear at least once in 5000 trees
   # If they never appear, admission gates are too tight
   ```

3. **Leaf size distribution:**
   ```python
   # Check minimum leaf sizes in trained model
   # If no leaves have <10 samples, min_data_in_leaf is too high
   ```

4. **Early stopping patience test:**
   ```python
   # Train with patience=100 vs patience=1000
   # If accuracy improves with more patience, rare signals are learning late
   ```

---

## 9. SUMMARY TABLE

| Question | Answer |
|----------|--------|
| Minimum fires to learn? | 3 with current settings (min_data_in_leaf=3). Theoretically 1. |
| Optimal feature_fraction? | 1.0 for max preservation. 0.7 floor for Optuna. |
| Rounds needed? | 3000-5000+ for rare signals. Early stopping patience >= 500. |
| Best loss function? | Standard logloss. No native rare-aware loss. Use extra_trees for binary features. |
| Most dangerous param? | min_data_in_leaf > fire count (structural block). |
| Second most dangerous? | lambda_l1 > 1.0 (zeros out weak leaves). |
| Safest regularization? | max_depth cap + learning_rate + path_smooth. |
| Current v3.3 assessment? | Mostly good. Three fixes needed: raise bagging/bynode floors, lower min_gain_to_split floor. |
