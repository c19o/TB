# EXPERT: Inference Optimization for Live Trading

## Context
LightGBM models trained on ~6M sparse features (the matrix) typically use only 5K-15K features in actual tree splits. Live BTC trading inference must be fast and deterministic. This document covers how to exploit that sparsity gap for minimum-latency prediction.

---

## 1. The Core Insight: 99.7% of Features Are Never Split On

LightGBM tree inference cost is driven by **tree traversal** (number of trees x splits checked), not by the raw feature count in the training schema. If a feature never appears in any split, it contributes zero to tree traversal. However, the live pipeline still wastes time:
- Computing 6M features when only 5K-15K matter
- Serializing/deserializing the full feature vector
- Cache pressure from a massive input array
- Memory allocation for unused columns

**The biggest latency win is outside the model itself**: stop generating unused features.

## 2. Extract Used Features from Trained Model

### Method: `feature_importance(importance_type="split")`

```python
import lightgbm as lgb
import numpy as np

model = lgb.Booster(model_file="model.txt")

# Get split counts per feature
importance = model.feature_importance(importance_type="split")
feature_names = model.feature_name()

# Features actually used in any tree split
used_mask = importance > 0
used_features = [f for f, used in zip(feature_names, used_mask) if used]
used_indices = np.where(used_mask)[0]

print(f"Total features: {len(feature_names)}")
print(f"Used in splits: {len(used_features)}")
print(f"Reduction: {100 * (1 - len(used_features)/len(feature_names)):.1f}%")
```

### Method: `trees_to_dataframe()` (most precise)

```python
df_trees = model.trees_to_dataframe()
split_features = df_trees["split_feature"].dropna().unique().tolist()
# This gives the exact set of features referenced in any node
```

**Use `split` importance, not `gain`**. A low-gain feature can still be required in some branches. Removing it changes predictions. Split count > 0 = feature is referenced in prediction paths.

## 3. Two Deployment Strategies

### Strategy A: Keep Original Model, Prune Feature Pipeline Only (SAFEST)

- Extract used features from the trained model
- Modify the live feature pipeline to compute ONLY those 5K-15K features
- Pass a full-width vector to the model (unused columns = 0 or NaN)
- **Predictions are bit-for-bit identical** to the original model
- Latency win comes from feature engineering time, not model scoring

**Pros**: Zero risk of prediction drift, no retraining needed
**Cons**: Still allocating/passing a 6M-wide vector to the scorer

### Strategy B: Retrain on Reduced Feature Set (RECOMMENDED)

- Extract used features from the best model
- Retrain with identical hyperparameters on only those features
- Validate: compare predictions, accuracy, SHAP distributions, and trading metrics on OOS data
- Deploy the smaller model

**Pros**: Entire pipeline is smaller (feature build, model file, inference input)
**Cons**: Requires retraining and validation; results may differ slightly due to EFB re-binning

### WARNING: Do NOT Hand-Edit Model Text Files

LightGBM stores `split_feature` using **internal feature indices** (via `InnerFeatureIndex`), which can differ from your DataFrame column order. Hand-remapping indices in the text dump is fragile and can silently produce wrong predictions. Always retrain instead.

## 4. lleaves: LLVM-Compiled Inference (5-10x Faster)

[lleaves](https://github.com/siboehm/lleaves) compiles LightGBM model text files into native machine code via LLVM IR.

### Published Benchmarks (NYC-taxi dataset)

| Batch Size | LightGBM (us) | lleaves (us) | Speedup |
|-----------|---------------|-------------|---------|
| 1 (single row) | 52.31 | 9.61 | **5.4x** |
| 100 | 441.15 | 31.88 | **13.8x** |

### Key Details
- Repository: `siboehm/lleaves`, latest release v1.3.0 (Dec 2024)
- Loads `model.txt`, converts to LLVM IR, compiles to native code
- Removes model-interpretation overhead, enables CPU branch prediction optimization
- **Single-row tuning**: set `fblocksize=Model.num_trees()` to disable cache blocking (cache blocking adds overhead for batch=1)

### Usage

```python
from lleaves import Model

model = Model(model_file="model.txt")
model.compile(cache="compiled_model.bin")  # One-time compilation

# Inference
predictions = model.predict(X)  # Drop-in replacement for lgb.Booster.predict
```

### Applicability to Our System
- Our models have moderate tree counts (Optuna-tuned), so single-row speedup should be in the 5-8x range
- Compilation is one-time cost at model load
- **Combine with Strategy B** (reduced features) for compounding gains

## 5. C API Fast Single-Row Path (Lowest Possible Latency)

LightGBM's C API exposes a dedicated fast single-row prediction path that removes per-call setup overhead.

### The Problem with Python predict()
- Rebuilds `Config` object on every call
- Reparses prediction parameters
- ~1/3 of single-row inference time is setup, not tree traversal
- DataFrame conversion adds more overhead

### C API Fast Path

```c
// One-time init (per model, per thread)
LGBM_BoosterPredictForMatSingleRowFastInit(
    booster_handle, predict_type, start_iteration, num_iteration,
    data_type, ncol, parameter, &fast_config_handle
);

// Per-tick scoring (microseconds)
LGBM_BoosterPredictForMatSingleRowFast(
    fast_config_handle, data, &out_len, out_result
);
```

### Real-World Benchmark (530 features, 10-230 trees, depth 8)

| Method | Single-Row Latency |
|--------|-------------------|
| C API Standard | 7.99 us |
| C API Fast | **3.89 us** |
| Python Booster.predict (NumPy) | ~50-100 us |
| Python with pandas | ~200-500 us |

Source: r/quant production trading benchmark, 2024

### Python Fast-Path Tips (If Staying in Python)
- Use `Booster.predict()` with **NumPy arrays**, never pandas DataFrames in the hot path
- Set `num_threads=1` for single-row prediction (multi-thread overhead > tree traversal time)
- Keep the Booster object resident in memory (never reload per tick)
- Use `predict(data, num_iteration=best_iteration)` to avoid scoring extra trees

## 6. Recommended Stack for Live BTC Trading

### Latency Budget Analysis

| Component | Current (est.) | Optimized |
|-----------|---------------|-----------|
| Feature engineering (6M cols) | 500-2000 ms | 10-50 ms (5K-15K cols) |
| Model scoring (Python, 6M input) | 100-500 ms | 5-50 us (lleaves/C API) |
| Total per-tick | 600-2500 ms | 10-50 ms |

**Feature engineering dominates**. Optimizing the model scorer alone is insufficient. The 40-100x win comes from pruning the feature pipeline.

### Implementation Priority

1. **Extract used features** from trained model (immediate, zero risk)
2. **Prune live feature pipeline** to compute only used features (biggest win)
3. **Retrain on reduced feature set** for cleaner deployment (Strategy B)
4. **Switch to lleaves** for compiled inference (5-8x scorer speedup)
5. **C API fast path** only if sub-millisecond scoring is required (unlikely -- feature eng dominates)

### Architecture

```
Live Candle Data
    |
    v
Minimal Feature Pipeline (5K-15K features only)
    |
    v
lleaves compiled model (or LightGBM Booster with NumPy)
    |
    v
Prediction -> Trade Signal
```

## 7. What NOT to Do

| Anti-Pattern | Why |
|-------------|-----|
| Hand-edit model.txt to remove features | Internal index remapping breaks predictions silently |
| Use pandas in prediction hot path | 10-50x slower than NumPy |
| Leave num_threads=default for single-row | Thread management overhead > tree traversal |
| Score all boosting rounds when early-stopped | `predict(num_iteration=best_iteration)` is faster |
| Build 6M features then select 5K post-hoc | Wasted compute; build only what's needed |
| Use gain importance to prune features | Low-gain features can still be required in branches |

## 8. Validation Checklist (Before Deploying Pruned Model)

- [ ] Prediction parity: pruned model vs original on 1000+ OOS samples (max absolute diff < 1e-6)
- [ ] Accuracy metrics: same or better on validation set
- [ ] SHAP distribution: top-20 features match between original and pruned
- [ ] Trading metrics: PnL, Sharpe, max drawdown on paper trades match within 5%
- [ ] Confidence calibration: predicted probabilities have same distribution shape
- [ ] Feature pipeline timing: measure end-to-end latency reduction, not just scorer

## Sources

- [LightGBM Advanced Topics](https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html)
- [LightGBM Parameters Tuning](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html)
- [LightGBM C API docs](https://lightgbm.readthedocs.io/en/v4.4.0/C-API.html)
- [lleaves GitHub](https://github.com/siboehm/lleaves) -- LLVM compiler for LightGBM, v1.3.0 (Dec 2024)
- [lleaves: Compiling Decision Trees for Fast Prediction](https://siboehm.com/articles/21/lleaves)
- [LightGBM Issue #2935: Single row prediction speedup](https://github.com/microsoft/LightGBM/issues/2935)
- [LightGBM Issue #5879: How to improve prediction speed](https://github.com/microsoft/LightGBM/issues/5879)
- [r/quant: LightGBM for trading decisions in production](https://www.reddit.com/r/quant/comments/1s4ow43/anyone_using_lightgbm_for_trading_decision_in/)
- [End-to-End Decision Forest Inference Pipelines comparison (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12406228/)
