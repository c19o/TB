# EXPERT: Feature Engineering & Cross-Feature Theory

**Context:** BTC trading system with 2-10M sparse binary cross-features generated from esoteric signals (astrology, numerology, space weather, gematria) x technical analysis. LightGBM with EFB on sparse CSR matrices. Rare signals are the edge.

---

## 1. The Matrix Thesis in Information-Theoretic Terms

The core thesis -- more diverse signals produce stronger predictions -- has a clean information-theoretic justification. When two feature families (e.g., astrology and RSI structure) are **conditionally independent given the target**, their conjunction carries strictly more mutual information than either alone. The relevant quantity is **synergy**:

```
synergy(x_i, x_j; y) = I(x_i, x_j; y) - I(x_i; y) - I(x_j; y)
```

Positive synergy means the joint state carries more label information than the sum of the marginals. This is exactly the "esoteric + TA only works together" behavior the matrix is designed to capture. An astrology bit alone may be noise, and an RSI bit alone mediocre, but when both fire during a specific volatility regime, the next-bar move has directional skew that neither signal predicts independently.

Cross-features make these interactions **explicit** rather than relying on tree depth to discover them. This is critical because:
- Rare conjunctions may never appear in enough leaf nodes for trees to isolate them
- Explicit binary crosses let LightGBM's leaf-wise growth find them in a single split
- EFB bundles mutually exclusive sparse crosses efficiently, so the compute cost of carrying millions of crosses is manageable

**Key principle:** The value of the matrix is NOT in any single feature. It is in the combinatorial explosion of regime-specific conjunctions across diverse signal families.

---

## 2. Four-Tier Binarization Theory

### Current Implementation (v2_cross_generator.py)

The system uses **4-tier binarization** with percentile thresholds on the non-zero distribution:

| Tier | Suffix | Threshold | Meaning | Expected Sparsity |
|------|--------|-----------|---------|-------------------|
| XH | `_xh` | > q95 | Extreme high | ~5% ones |
| H | `_h` | > q75 | High | ~25% ones |
| L | `_l` | < q25 | Low | ~25% ones |
| XL | `_xl` | < q5 | Extreme low | ~5% ones |

Plus a 2-tier fallback (q80/q20) for certain feature families.

### Why These Thresholds Are Correct

**q95/q5 (extreme tiers):** These capture tail events -- the exact regime where esoteric signals are expected to have predictive power. With ~5% activation rate on 818 weekly bars, XH fires ~41 times. After AND-crossing with another XH feature (~5%), the conjunction fires ~2 times (0.25%). This is at the absolute floor of statistical detectability but is where the edge lives.

**q75/q25 (standard tiers):** These provide the "context" layer. A feature being "high" (top quartile) is common enough to co-occur with rare signals from other families. Crossing H (25%) with XL (5%) gives ~1.25% activation -- roughly 10 bars on weekly data, enough for LightGBM's `min_data_in_leaf=3` to make a split.

**Non-zero distribution basis:** The implementation correctly computes percentiles on non-zero values when there are >100 non-zero entries. This prevents structural zeros (e.g., "no lunar eclipse this bar") from distorting the threshold computation. Binary/boolean source features naturally collapse to fewer meaningful tiers.

### Information-Theoretic Justification

Fixed-percentile binarization is a form of **equal-frequency discretization**. For gradient boosting, this is often superior to equal-width binning because:

1. **Histogram bin efficiency:** LightGBM uses 2 bins for binary features (max_bin=2). Each tier becomes its own feature, so the model gets 4 independent binary splits per continuous source -- more expressive than a single 256-bin histogram
2. **Cross-feature compatibility:** Binary inputs enable sparse AND-crossing via matrix multiplication. Continuous inputs would require binning anyway or produce dense crosses
3. **Tail sensitivity:** Extreme tiers (q95/q5) concentrate information about rare events. Information gain from splitting at the 95th percentile is highest when the target distribution differs most between tail and body -- exactly the regime-change detection the matrix targets

### What the Literature Says

Quantile-based discretization preserves rank ordering and adapts to the feature's distribution shape. The Binarsity penalization framework (Bray et al., JMLR 2018) shows that grouped one-hot encodings of binned continuous features with total-variation penalization can select the minimal number of relevant cut-points within each group. The 4-tier system is a manual but well-motivated version of this -- pre-selecting cut-points at distribution landmarks rather than learning them.

For gradient boosting specifically, explicit binarization at multiple thresholds is most justified when:
- The informative regime is rare enough that tree growth may never isolate it early enough
- Cross-features need binary inputs for sparse matmul generation
- The same feature has different predictive meaning at different quantile levels (e.g., RSI > 95th vs RSI > 75th may predict different things)

---

## 3. Co-Occurrence Threshold Theory

### Current Setting: MIN_CO_OCCURRENCE = 3

This was lowered from 8 to match `min_data_in_leaf=3`. The rationale: if LightGBM can make a split with 3 samples in a leaf, then a cross-feature firing 3 times is the minimum needed to potentially appear in a leaf.

### The Tension: Noise vs. Rare Edge

The fundamental tension in setting the co-occurrence floor:

| Floor | Pros | Cons |
|-------|------|------|
| 1 | Keeps ALL possible interactions | Pure noise; single-event features can't validate in CPCV |
| 2-3 | Preserves ultra-rare regime conjunctions | High variance; many spurious features survive |
| 5-8 | Features can appear in multiple CPCV folds | Kills the rarest esoteric crosses |
| 10-20 | Statistically stable | Eliminates exactly the rare events the matrix targets |
| 50+ | Classical association-rule territory | Destroys the entire thesis |

### Why 3 Is Defensible (But Aggressive)

With 5-fold CPCV on 818 weekly bars (~164 bars per fold), a feature firing 3 times has:
- **98.3% probability** of appearing in at least one training fold (binomial calculation)
- **~40% probability** of appearing in ANY validation fold
- Only 3 co-occurring bars across ~16 years of data

This is at the statistical boundary. The feature is ONLY useful if:
1. LightGBM's regularization (`min_data_in_leaf`, `lambda_l1`, `lambda_l2`) prevents overfitting to 3 samples
2. The feature survives `feature_fraction >= 0.7` subsampling (70% chance per round)
3. The conjunction represents a real regime, not a data artifact

### Recommended Tiered Approach

Rather than a single global threshold, the literature on rare-event prediction recommends tiered support:

```
Tier 1 (floor=2):  Hypothesis features -- hand-curated esoteric x TA families
                    where you have prior belief in the conjunction
Tier 2 (floor=3):  Cross-family pairs (different signal families)
                    e.g., astro x TA, space_weather x volatility
Tier 3 (floor=5):  Same-family pairs (within signal family)
                    e.g., astro x astro, TA x TA
```

The current system uses a flat floor=3, which is acceptable because:
- LightGBM's regularization is the real filter, not the co-occurrence threshold
- `feature_pre_filter=False` ensures ALL features are considered
- EFB bundles near-exclusive features, so carrying extras is cheap
- The co-occurrence filter is a **math constraint on validation**, not a signal filter

### What Information Theory Says

The co-occurrence floor is NOT an information-theoretic quantity. It is a **statistical reliability floor**. Mutual information of a binary pair with support=2 has enormous variance -- the 95% confidence interval on I(z; y) covers nearly the entire [0, 1] range. Empirical-Bayes shrinkage or pseudocount smoothing is needed for any MI-based scoring with support < ~30.

For the matrix thesis, the correct approach is:
1. Set the co-occurrence floor **low** (2-3) to preserve rare conjunctions
2. Let LightGBM's split regularization decide what's useful
3. Use **walk-forward PR-AUC** and trading expectancy to evaluate, not in-sample MI
4. Never raise the floor based on classical association-rule intuition -- those thresholds are designed for market-basket analysis, not rare-event prediction

---

## 4. Cross-Feature Interaction Orders

### 2nd-Order (Pairwise) Crosses -- The Sweet Spot

The current system generates **2nd-order crosses** (A AND B). This is the correct choice because:

- **Tree depth equivalence:** A tree of depth `h` can capture interactions of order `h`. With `num_leaves=31` (default), LightGBM trees have effective depth ~5, so they CAN discover 2nd-order interactions -- but only if both features appear in the same tree path. Explicit 2nd-order crosses guarantee the conjunction is available in a single split
- **Sparsity scaling:** If feature A fires 5% of the time and feature B fires 5%, then A AND B fires ~0.25%. At 2M source features, pairwise crosses produce ~2 trillion candidates before co-occurrence filtering. The co-occurrence filter reduces this to 2-10M actual features -- manageable for LightGBM with EFB
- **Interpretability:** "Mercury retrograde AND RSI oversold" is interpretable. 3rd-order "Mercury retrograde AND RSI oversold AND lunar eclipse" is harder to reason about

### 3rd-Order Crosses -- Usually Not Worth It

**Do NOT generate explicit 3rd-order crosses.** Reasons:

1. **Combinatorial explosion:** With N base features, 3rd-order produces O(N^3) candidates. Even with aggressive filtering, this can exceed memory and training time
2. **Extreme sparsity:** 5% x 5% x 5% = 0.0125% activation. On 818 weekly bars, that's ~0.1 bars. Almost no 3rd-order cross will pass MIN_CO_OCCURRENCE=3
3. **LightGBM handles it:** With explicit 2nd-order crosses already in the feature set, a tree of depth 2+ can combine a cross-feature with another base feature, effectively discovering 3rd-order interactions WITHOUT materializing them
4. **EFB cost:** 3rd-order crosses that are near-exclusive with their parent 2nd-order crosses get bundled by EFB anyway, adding no information

**Exception:** If you have a specific hypothesis (e.g., "planetary aspect + volatility regime + DOY window"), hand-craft that specific 3rd-order cross. Don't enumerate all possible triples.

### The Interaction Detection Question

How to know which 2nd-order pairs are worth crossing? The literature offers several approaches:

**Friedman's H-statistic:** Measures the strength of interaction between variables in an existing model. H varies 0-1 (no interaction to pure interaction). Compute H for the top-importance features from a baseline model, then generate crosses only for pairs with H > 0.1. Problem: expensive to compute for millions of features.

**Conditional Mutual Information (CMI):** For candidate pair (x_i, x_j):
```
CMI(x_i, x_j; y) = I(x_i; y | x_j) - I(x_i; y)
```
Positive CMI means x_i gains predictive power when conditioned on x_j. This directly identifies synergistic pairs. Problem: requires reliable probability estimates, which are noisy at low support.

**Synergy scoring (recommended for this system):**
```
score_ij = shrink(I(x_i AND x_j; y))
         + lambda_1 * shrink(synergy_ij)
         - lambda_2 * redundancy
         - lambda_3 * instability
```
Where `shrink()` applies empirical-Bayes shrinkage for low-support pairs, and instability is measured as score variance across rolling train folds.

### What the Matrix Actually Does

The current v2_cross_generator.py uses **structured family crossing**, not exhaustive enumeration:

| Prefix | Left Family | Right Family | Rationale |
|--------|-------------|--------------|-----------|
| `dx_` | DOY window | All contexts | Calendar seasonality x everything |
| `ax_` | Astro single | TA indicators | Planetary state x market structure |
| `ax2_` | Multi-astro | TA indicators | Complex astro x market structure |
| `ta2_` | Multi-TA | DOY + astro | TA confluence x calendar/planets |
| `ex2_` | Esoteric misc | TA indicators | Numerology/gematria x market |
| `sw_` | Space weather | All | Solar/geomagnetic x everything |
| `hod_` | Hour-of-day | All | Intraday timing x everything |
| `mx_` | Macro signals | All | Economic regime x everything |
| `vx_` | Vol regime | All | Volatility state x everything |
| `asp_` | Planetary aspects | All | Aspect geometry x everything |
| `pn_` | Price numerology | All | Numeric harmony x everything |

This is **superior to blind enumeration** because:
- It enforces cross-family diversity (esoteric x TA, not TA x TA)
- It embodies the thesis: diverse signals interacting produce edge
- It avoids the combinatorial explosion of same-family crosses that add redundancy, not information

---

## 5. Optimal Pair Selection for Sparse Binary Features

### The Screening Pipeline

With 2-10M sparse binary features, you cannot score all pairs exhaustively. The correct pipeline is:

**Stage 1: Candidate Generation (current system)**
- Use structured family crossing (NOT all-pairs)
- Co-occurrence filter via sparse matmul: `left.T @ right >= MIN_CO_OCCURRENCE`
- This is O(nnz) in sparse format -- fast even for millions of features

**Stage 2: LightGBM as the Filter**
- `feature_pre_filter=False` -- all features enter training
- EFB bundles near-exclusive features (most sparse crosses qualify)
- Leaf-wise growth with `min_data_in_leaf` acts as a natural regularization
- `feature_fraction >= 0.7` provides stochastic regularization without killing rare signals
- CPCV validation prevents overfitting to specific folds

**Stage 3: Post-Training Analysis (optional, for future versions)**
- SHAP values identify which crosses actually contribute
- Ablation testing: remove feature families, measure accuracy drop
- Stability analysis: do the same features appear across CPCV folds?

### Why NOT to Pre-Filter with MI/Synergy

For this system, MI-based pre-filtering is **counterproductive** because:

1. **Support is too low:** Most crosses fire 3-50 times. MI estimates at this support level are unreliable (huge confidence intervals)
2. **LightGBM IS the filter:** Tree splits are a natural form of conditional MI maximization. The model already selects the most informative splits
3. **EFB makes it cheap:** Carrying 10M sparse binary features costs ~the same as 500K dense features after bundling. There is no computational reason to pre-filter
4. **False negatives are fatal:** Filtering out a rare cross that would have been the edge is worse than carrying 100 useless crosses. The matrix thesis demands keeping everything

### When Pre-Filtering Makes Sense

Pre-filtering would be justified if:
- Feature count exceeds what EFB can bundle (~50M+)
- Training time becomes prohibitive (days per fold)
- Memory prevents loading the full sparse matrix

None of these apply at 2-10M features with the current hardware.

---

## 6. Rare-Event Prediction Theory

### Why Rare Signals Can Be Real Edge

Classical statistics says "rare = unreliable." But in trading:

1. **Rare events move markets most:** Tail events produce the largest returns. A signal that fires 5x/year but captures 3 major moves is more valuable than one that fires daily with 51% accuracy
2. **Rare combinations are hard to arbitrage:** If a signal requires Mercury retrograde + RSI divergence + high Kp index, few other traders are looking at that conjunction. Scarcity of attention = persistence of edge
3. **LightGBM handles rare splits well:** Unlike linear models, tree-based models can assign extreme predictions to tiny leaf nodes. A leaf with 3 samples and 100% accuracy contributes to the ensemble even if it rarely activates

### Regularization for Rare Features

The correct regularization stack for rare binary crosses:

| Control | Purpose | Recommended Setting |
|---------|---------|-------------------|
| `min_data_in_leaf` | Minimum samples in a leaf | 3 (weekly), 5 (4h), 8 (1h), 15 (15m) |
| `feature_fraction` | Random feature subset per tree | >= 0.7 (protects rare features) |
| `lambda_l1` | L1 penalty on leaf values | Low (0-0.1) -- aggressive L1 kills rare features |
| `lambda_l2` | L2 penalty on leaf values | Moderate (1-10) -- shrinks extreme predictions |
| `feature_pre_filter` | Remove features that can't split | ALWAYS False |
| `min_gain_to_split` | Minimum gain for a split | 0 -- let the model decide |
| `max_bin` | Bins per feature | 2 for binary features (automatic) |
| CPCV | Cross-validation strategy | 5-fold, combinatorial purged |

### The PR-AUC Principle

For rare-event prediction, **Precision-Recall AUC** is the correct evaluation metric, not ROC-AUC. ROC-AUC can look excellent (0.95+) even when the model never correctly predicts rare events, because it rewards true negatives. PR-AUC directly measures whether the model's positive predictions (trade signals) are accurate.

Walk-forward PR-AUC with cost adjustment (spread + fees) is the gold standard for evaluating whether rare cross-features are adding real trading value.

---

## 7. LightGBM + EFB + Sparse CSR: Architecture Fit

### Why LightGBM Is Architecturally Correct

| Property | LightGBM Behavior | Matrix Benefit |
|----------|-------------------|----------------|
| **EFB** | Bundles mutually exclusive features | Sparse binary crosses are near-exclusive by construction |
| **max_bin=2** | Binary features use 2 bins | No histogram waste on binary crosses |
| **Sparse CSR input** | Native sparse support | 2-10M features fit in memory as CSR |
| **Leaf-wise growth** | Grows deepest on most informative splits | Finds rare-but-strong conjunctions first |
| **force_col_wise** | Column-parallel histogram building | Scales to millions of features |
| **feature_fraction** | Random feature subsetting | Stochastic regularization for huge feature sets |
| **bin_construct_sample_cnt** | Samples for bin construction | Increase for very sparse data (default 200000) |

### EFB and Cross-Features

EFB (Exclusive Feature Bundling) identifies features that rarely co-occur (conflict ratio < threshold) and bundles them into a single feature with offset bins. For sparse binary crosses:

- Two crosses from different families (e.g., `ax_mercury_retro_XH AND rsi_div_XL` vs `sw_kp_high AND macd_cross_XH`) almost never fire simultaneously
- EFB bundles them into one feature with 3 bins: {neither, first, second}
- This effectively compresses 10M sparse binary features into ~500K-2M bundles
- Training cost scales with bundles, not raw feature count

**Critical setting:** `enable_bundle=True` (always). Disabling EFB removes the entire architecture advantage.

### Sparse CSR Constraints

- `indptr` MUST be int64 when NNZ > 2^31 (2.1 billion non-zeros). With 2M features x 800 rows x ~5% density = 80M NNZ, int32 is sufficient for weekly. For 15m (~227K rows), NNZ can exceed int32
- `indices` can stay int32 (row indices fit in 32 bits for any practical dataset)
- NEVER convert to dense for training. Dense 2M features x 800 rows = 6.4GB float32 vs ~320MB sparse

---

## 8. Key Research References

### Feature Cross Generation & Selection
- **mRMR (Peng et al., 2005):** Max-Relevance Min-Redundancy. Rank features by MI with target minus MI with selected set. Foundation for cross-feature selection
- **Conditional MI (Fleuret, 2004):** Fast binary feature selection via conditional mutual information. Directly applicable to binary cross-feature ranking
- **Synergy/Interaction Information (Timme et al., 2018):** Detecting pairwise interactive effects via I(x_i, x_j; y) - I(x_i; y) - I(x_j; y). Key metric for identifying conjunctions that are more than the sum of parts

### Rare Event Prediction
- **Comprehensive Survey on Rare Event Prediction (arXiv 2309.11356):** Central obstacles: skewed class distribution, lack of minority samples, temporal dependence. Recommends against aggressive pre-filtering of rare features
- **PR-AUC for Rare Outcomes (PMC8561661):** Standard ROC metrics mislead under severe class imbalance. PR-AUC is the correct evaluation for rare-event models

### LightGBM Architecture
- **LightGBM Paper (Ke et al., NeurIPS 2017):** EFB, GOSS, histogram-based splitting. EFB specifically designed for high-dimensional sparse features
- **LightGBM Parameters Documentation:** `feature_pre_filter`, `enable_bundle`, `force_col_wise`, `bin_construct_sample_cnt` -- all critical for sparse binary feature sets

### Binarization & Discretization
- **Binarsity (Bray et al., JMLR 2018):** Grouped one-hot encodings with total-variation penalization for selecting optimal cut-points. Theoretical foundation for multi-tier binarization
- **Optimal Classification Trees (AAAI 2025):** Dynamic programming for optimal threshold selection on continuous features. Confirms that percentile-based thresholds are near-optimal for balanced bin populations

---

## 9. Summary: Non-Negotiable Principles

1. **Never pre-filter features before LightGBM.** EFB + tree regularization IS the filter
2. **Cross-family diversity is the edge.** Esoteric x TA crosses are worth more than TA x TA crosses because synergy is highest between independent signal families
3. **4-tier binarization at q95/q75/q25/q5 is correct.** It captures both tail events (XH/XL) and context (H/L)
4. **MIN_CO_OCCURRENCE=3 is the right floor.** Lower loses all validation power; higher kills the rare-event edge
5. **2nd-order crosses only.** 3rd-order is redundant (trees combine 2nd-order crosses with base features) and combinatorially explosive
6. **Evaluate with walk-forward PR-AUC and trading expectancy, not MI.** MI is unreliable at low support; model performance is the only truth
7. **Sparse CSR through the entire pipeline.** Dense conversion destroys the architecture advantage
8. **feature_fraction >= 0.7.** Lower values kill rare esoteric crosses via stochastic exclusion
9. **feature_pre_filter=False, always.** True silently drops features that can't split on small samples -- exactly the rare features that matter
10. **The model decides.** Not us. No hand-pruning. No importance-based pre-selection. Feed everything, let LightGBM sort it out
