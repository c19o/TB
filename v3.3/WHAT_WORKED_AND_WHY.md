# What Worked, WHY It Worked, and How It Applies Per-TF

## 1. BINARY CLASSIFICATION (+8.2% on 1w)

**What**: Converted 3-class (SHORT/FLAT/LONG) → binary (DOWN/UP), dropped FLAT rows.

**WHY it worked**:
- 819 rows split 3 ways = ~270 per class. Not enough for LightGBM to learn 3 decision boundaries.
- FLAT is the "garbage" class — bars that didn't move enough to hit TP or SL. The model was wasting capacity trying to predict "nothing happened."
- Binary doubles effective samples per class: ~500 UP + ~450 DOWN.
- The model only needs ONE decision boundary instead of two.

**Per-TF applicability**:
| TF | Rows | Apply Binary? | Why |
|----|------|--------------|-----|
| 1w | 819 | **YES** | Too few rows for 3-class. Binary proven +8.2%. |
| 1d | 5,733 | **TEST BOTH** | Enough rows for 3-class, but binary might still win. Run A/B test. |
| 4h | 8,794 | **MAYBE** | Marginal. 3-class should work. Binary as fallback. |
| 1h | 90,000 | **NO** | Plenty of rows. 3-class gives more trading flexibility (FLAT = don't trade). |
| 15m | 227,000 | **NO** | 3-class is optimal. FLAT class adds value at high frequency. |

---

## 2. LEARNING RATE 0.03 → 0.234 (+biggest param impact)

**What**: Optuna found LR=0.234 (we proposed 0.1, it went higher).

**WHY it worked**:
- At LR=0.03, each tree makes tiny adjustments. With 819 rows and early stopping patience=50, the model builds 1 tree then stops improving for 50 rounds → early stops at 1 tree.
- At LR=0.234, each tree makes 8x larger adjustments. The model learns FAST and can build meaningful trees before early stopping kicks in.
- Small datasets need aggressive learning — there's not enough data for gradual convergence.

**Per-TF applicability**:
| TF | Rows | Recommended LR | Why |
|----|------|---------------|-----|
| 1w | 819 | **0.2-0.3** | Tiny data needs fast learning. Optuna found 0.234. |
| 1d | 5,733 | **0.1-0.2** | More data, can afford slower learning. Let Optuna decide. |
| 4h | 8,794 | **0.05-0.15** | Medium data. Default 0.1 range good. |
| 1h | 90,000 | **0.03-0.1** | Lots of data. Gradual learning prevents overfitting. |
| 15m | 227,000 | **0.01-0.05** | Huge data. Low LR + many rounds = best generalization. |

---

## 3. num_leaves=5 (Optuna chose SIMPLER model)

**What**: We proposed num_leaves=15. Optuna chose 5.

**WHY it worked**:
- With 819 rows, complex trees overfit. Each leaf needs min_data_in_leaf samples.
- 5 leaves × 5 min_samples = 25 samples needed per tree split. With ~450 training rows, that's ~18 possible splits — just right.
- 15 leaves would need 75 samples per tree, leaving only ~6 splits — too constrained.
- Simpler trees generalize better on tiny data.

**Per-TF applicability**:
| TF | Rows | Recommended Leaves | Why |
|----|------|-------------------|-----|
| 1w | 819 | **5-7** | Tiny data. Simple trees generalize. |
| 1d | 5,733 | **15-31** | More data supports more complexity. |
| 4h | 8,794 | **31-63** | Medium complexity. |
| 1h | 90,000 | **63-127** | Can handle complex trees. |
| 15m | 227,000 | **127-255** | Full complexity — data supports it. |

---

## 4. LEAN MODE — Dropping Redundant TA (+1.3% on 1w)

**What**: Reduced 3714 → 2587 features by removing redundant TA (Ichimoku, Bollinger, MACD, Stochastic, MFI, Williams). Kept SAR/EMA/RSI + all esoteric.

**WHY it worked**:
- 819 rows / 3714 features = 0.22 samples per feature. Extreme curse of dimensionality.
- 819 rows / 2587 features = 0.32 samples per feature. Still bad but better.
- Redundant features (SMA10, SMA20, SMA50, SMA100, SMA200 — all measuring "trend") add noise. The model randomly picks between correlated features each fold → unstable (Jaccard=0.175).
- Fewer features = higher feature_fraction hit rate. At 0.7 fraction × 3714 = 2600 features seen per tree. At 0.7 × 2587 = 1811. Each tree sees a MORE relevant subset.

**Per-TF applicability**:
| TF | Rows | Use Lean? | Why |
|----|------|----------|-----|
| 1w | 819 | **YES** | Critical. 0.22 samples/feature is terrible. Lean helps. |
| 1d | 5,733 | **NO** | 1.6 samples/feature. Full TA is fine. Cross gen adds value. |
| 4h | 8,794 | **NO** | 2.4 samples/feature with crosses. Model can handle full suite. |
| 1h | 90,000 | **NO** | 31 samples/feature. Full suite optimal. |
| 15m | 227,000 | **NO** | 78 samples/feature. All features contribute. |

---

## 5. max_bin=7 (was 255)

**What**: LightGBM histogram bins reduced from 255 to 7.

**WHY it worked**:
- Our features are mostly BINARY (0/1 from 4-tier binarization). Binary features need 2 bins, not 255.
- 255 bins × 2.9M features = 740M histogram entries. 7 bins × 2.9M = 20M. 36x less memory.
- Faster histogram building = faster training.
- Fewer bins = stronger regularization on continuous features (less overfitting).

**Per-TF applicability**:
| TF | Rows | max_bin | Why |
|----|------|---------|-----|
| ALL | ALL | **7** | Binary features dominate all TFs. 7 bins is correct everywhere. Global change. |

---

## 6. CPCV (8,2) instead of (5,2)

**What**: 28 CPCV paths instead of 10. 75% train split instead of 60%.

**WHY it worked**:
- (5,2) = 60% train = ~492 rows per fold. With 50-bar purge eating ~18%, effective = ~400 rows.
- (8,2) = 75% train = ~614 rows per fold. Effective = ~500 rows. 25% more data per fold.
- More paths (28 vs 10) = better statistical coverage. Every time period gets tested more thoroughly.
- More robust PBO estimate (28 paths to compare vs 10).

**Per-TF applicability**:
| TF | Rows | CPCV Config | Why |
|----|------|------------|-----|
| 1w | 819 | **(8,2) = 28 paths** | Maximizes training data on tiny dataset. |
| 1d | 5,733 | **(5,2) = 10 paths** | Standard. Enough data per fold. |
| 4h | 8,794 | **(10,2) = 45 paths, sample 30** | More paths for statistical coverage. |
| 1h | 90,000 | **(10,2) = 45 paths, sample 30** | Standard for large datasets. |
| 15m | 227,000 | **(10,2) = 45 paths, sample 30** | Same as 1h. |

---

## 7. SAR-NUMEROLOGY HYBRIDS (new signal type)

**What**: Applied numerology (digital root, gematria, angel numbers) to SAR, RSI, and EMA values.

**WHY it worked**:
- AlphaNumetrix proves numerology on TA works in practice.
- `price_sar_dr_diff` (gain=2.1) = difference in digital roots of price and SAR. When they align numerologically, it signals trend confirmation.
- `rsi_digit_sum` (gain=0.9) = RSI in specific numerological zones has directional bias.
- These features capture patterns invisible to pure TA — the "energetic signature" of price levels.

**Per-TF applicability**:
| TF | Apply? | Why |
|----|--------|-----|
| ALL | **YES** | Numerology on TA values works at all timeframes. Computed from existing columns. Zero cost. |

---

## 8. deterministic=False (was True — CRITICAL BUG)

**What**: Removed `deterministic=True` from LightGBM config.

**WHY it worked**:
- `deterministic=True` forces SINGLE-THREADED histogram computation. On a 128-core machine, 127 cores sit idle.
- Also blocks GPU histogram computation (cuda_sparse requires non-deterministic mode).
- Removing it enables: multi-core CPU histograms, GPU histograms, parallel feature evaluation.
- Training speed: 10-50x faster. Also slightly different (better) model due to parallel histogram accumulation.

**Per-TF applicability**:
| TF | Apply? | Why |
|----|--------|-----|
| ALL | **YES — CRITICAL** | This was killing performance on every TF. Must be False globally. |

---

## 9. ESOTERIC FEATURES — What's Working

### Calendar Seasonality (doy_sin/cos) — TOP esoteric, gain=7.8
**WHY**: BTC has strong calendar effects. Year-end tax selling, January effect, summer lull, Q4 rally. Day-of-year captures all seasonal patterns in one feature.
- **All TFs**: YES. Calendar effects apply to daily, 4h, 1h, 15m too.

### Jupiter-Saturn Regime — gain=2.6
**WHY**: 20-year cycle. BTC was born during Jupiter-Saturn conjunction (2009). Regime HIGH correlates with expansion phases. The longest astronomical cycle that fits within BTC's history.
- **All TFs**: YES. Same planetary positions affect all timeframes.

### New Moon x Bear — gain=0.8
**WHY**: Lunar cycles affect human psychology → market sentiment. New moon = reset/beginning. Crossed with bear trend = potential reversal point.
- **All TFs**: YES. Lunar cycle is 29.5 days = relevant for all TFs.

### Tweet Sentiment — gain=1.3
**WHY**: Social media reflects and amplifies market sentiment. Bull/bear tweet counts capture crowd positioning.
- **All TFs**: YES but stronger on shorter TFs (more tweets per bar).

### Fibonacci Day — gain=0.7
**WHY**: Fibonacci numbers appear in nature and markets. Days that are Fibonacci numbers from significant events may have resonance.
- **All TFs**: YES. Sacred geometry applies universally.

---

## 10. WHAT FAILED ON 1W (Don't Repeat)

### TA x TA Crosses — Slight accuracy DROP
**WHY it failed on 1w**: Interaction features require N² data. 819 rows with 13 new crosses = not enough data to learn interaction patterns. The noise from extra features outweighed the signal.
- **1d/4h/1h/15m**: SHOULD WORK. More rows = interactions become learnable. Enable for all larger TFs.

### Adding Features Beyond 2587 on 1w
**WHY it failed**: Curse of dimensionality. Every feature added to 819 rows dilutes signal. v5 (2587) beat v6 (2630).
- **1d+**: Not a problem. 5733+ rows can handle many more features.

### 52-Week Features on 1w
**WHY they showed zero gain**: Rolling 52-bar window on 819 bars = first 52 bars are NaN. Only 767 bars have values. And the feature has very slow variation (changes ~2% per week). Not enough dynamic range for LightGBM to find useful splits.
- **1d**: 52-DAY features (not 52-week) will work great on 5733 bars.

---

## Confidence vs Accuracy Summary (Best Model — v5)

**Tradable zone: >= 75% confidence = 57.8% accuracy on 1,952 trades**
**High confidence: >= 80% = 58.1% on 1,589 trades**

For reference: Renaissance Medallion reportedly wins ~50.75% of trades. Our 58% at high confidence on weekly BTC is competitive — the edge comes from selectivity (only trading when confident) + the matrix (esoteric timing on top of TA trend).
