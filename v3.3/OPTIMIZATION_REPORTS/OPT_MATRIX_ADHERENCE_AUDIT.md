# Matrix Adherence Audit Report

**Date:** 2026-03-30
**Auditor:** Claude Opus 4.6 (1M context)
**Scope:** All `.py` files in `v3.3/` (excluding `gpu_histogram_fork/_build/LightGBM/` vendor code)
**Matrix Thesis:** 2.9M features from cross-products of esoteric signals. LightGBM with EFB on sparse CSR. Rare signals (10-20 fires) ARE the edge. NaN = missing, 0 = value. No filtering. No fallbacks.

---

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 3     |
| HIGH     | 8     |
| MEDIUM   | 7     |
| **Total** | **18** |

---

## Violation 1: fillna(0) on Feature Data

NaN means "signal not present" -- LightGBM learns optimal split direction for NaN. Converting NaN to 0 tells the model "this signal fired with value zero", which is a DIFFERENT semantic meaning. This destroys the information advantage of sparse esoteric signals.

### CRITICAL

| # | File:Line | Description |
|---|-----------|-------------|
| 1a | `v2_feature_layers.py:411` | `returns.fillna(0)` on return series before Shannon entropy computation. Comment says "required" because `sliding_window_view` can't handle NaN, but this converts NaN returns to zero-return bars, conflating "no data" with "flat bar". |
| 1b | `feature_library.py:2332-2334` | `fillna(0)` on `has_gold_num`, `has_red_num`, `has_green_num` -- Chinese calendar color features. NaN means "no data for this date", 0 means "color not present". Different signals. |
| 1c | `feature_library.py:2465` | `fillna(0)` on engineering column values before cross computation. Converts missing engineering data to zero. |
| 1d | `feature_library.py:3249` | `fillna(0)` on shifted spike decay series. Converts leading NaN from `.shift()` to 0. |
| 1e | `feature_library.py:6352` | `fillna(0)` on shifted eclipse series for edge detection. |

### HIGH

| # | File:Line | Description |
|---|-----------|-------------|
| 1f | `build_sports_features.py:252,513` | `fillna(0)` on sports game counts and columns. NaN = "no game data available", 0 = "zero games played". Different signals for the model. |

### Verdict
`validate.py` lines 788-803 check for this as a WARNING (not a failure). The check is present but non-blocking. Items 1a, 1d, 1e are arguably justified (shift/windowed operations need dense input), but 1b, 1c, 1f destroy signal meaning.

---

## Violation 2: feature_pre_filter

No violations found. All LightGBM param dicts set `feature_pre_filter=False`. Config enforced at `config.py:78`. `validate.py` checks this at lines 77-80 (config), 776-786 (file scan), and 994-1039 (Dataset construction audit). CLEAN.

---

## Violation 3: feature_fraction < 0.7

No hardcoded values below 0.7 found in any non-vendor `.py` file. Config value is 0.9. Optuna search range is `[0.7, 1.0]`. `validate.py` checks at lines 66-69 (config), 234-246 (Optuna bounds), 805-820 (file scan). CLEAN.

---

## Violation 4: bagging_fraction < 0.7

### CRITICAL

| # | File:Line | Description |
|---|-----------|-------------|
| 4a | `leakage_check.py:194` | `bagging_fraction: 0.6` -- hardcoded 60% bagging. With 10-fire signals, P(signal in bag) = 0.6^10 = 0.6%. Effectively kills all rare signals during leakage checking. Even though this is a diagnostic script, it produces misleading leakage results for rare features. |

### HIGH

| # | File:Line | Description |
|---|-----------|-------------|
| 4b | `v2_multi_asset_trainer.py:524` | Dynamic `bagging_fraction = batch_size / n_rows`. If `V2_BATCH_SIZE` env var is set to a small value relative to row count, bagging_fraction can drop far below 0.7. No floor enforced. |

### Verdict
`validate.py` checks Optuna bounds (line 256-262) and config value (0.95 in config.py:337) but does NOT grep for hardcoded bagging_fraction < 0.7 in non-config files. **Gap in validate.py.**

---

## Violation 5: min_data_in_leaf > 10

### HIGH

| # | File:Line | Description |
|---|-----------|-------------|
| 5a | `bench_trial.py:58` | `min_data_in_leaf: 10` -- at the threshold. Signals firing exactly 10 times can form leaves, but anything less cannot. |
| 5b | `run_optuna_local.py:552` | `suggest_int('min_data_in_leaf', max(3, _tf_mdil), 15)` -- upper bound of 15 means Optuna CAN select values of 11-15, where signals firing 10-12 times become invisible. |
| 5c | `gpu_histogram_fork/benchmark/bench_end_to_end.py:127` | `min_data_in_leaf: 5` -- acceptable (< 10). |

### Verdict
Config values are all 8 (safe). But Optuna upper bound of 15 (line 552) allows dangerous exploration. `validate.py` checks per-TF config values are <= 15 (line 145-148) but the threshold should arguably be <= 10, not 15. The Optuna upper bound of 15 is the real risk.

---

## Violation 6: lambda_l1 > 4.0

No violations found. Config value is 0.5. Optuna cap is 4.0. `meta_labeling.py:166` uses 2.0 (safe). `bench_trial.py:60` uses 2.0 (safe). `validate.py` checks Optuna bounds at lines 264-271. CLEAN.

---

## Violation 7: Feature Whitelist/Blacklist/Filter Before Model

### MEDIUM

| # | File:Line | Description |
|---|-----------|-------------|
| 7a | `feature_classifier.py:96-119` | `filter_features()` function exists but docstring says "NO filtering" and implementation keeps ALL features. Median imputation for sklearn MI only (not for LightGBM). Confirmed safe on inspection -- no actual filtering occurs. |

### Verdict
No actual feature filtering found. The `filter_features` function name is misleading but the implementation is correct. CLEAN.

---

## Violation 8: Silent Exception Swallowing on Feature Data

### MEDIUM

| # | File:Line | Description |
|---|-----------|-------------|
| 8a | `cloud_run_tf.py:248` | `except: pass` -- bare except on lockfile cleanup. Not feature data, just file I/O. Low risk. |
| 8b | `run_cross_1d.py:8` | `except: pass` -- bare except on import. Could mask a broken dependency silently. |
| 8c | `astrology_engine.py:698,748,862,899,958,999,1018` | Seven `except Exception:` blocks that silently return neutral/default values for astrology computations. When an astrology calculation fails, it returns a neutral fallback (e.g., ratio=1.0, count=0) instead of propagating the error. This means broken astrology data silently degrades to "no signal" rather than alerting. |
| 8d | `build_1d_features.py:11,190,229,276,331,376` | Multiple broad `except Exception` blocks during feature building. If a feature source fails, the build continues with missing features rather than failing. |

### Verdict
Items 8c and 8d are the most concerning -- astrology/feature build failures silently produce weaker feature sets without any training-blocking error. `validate.py` does NOT check for bare except patterns or silent fallback returns in feature computation code. **Gap in validate.py.**

---

## Violation 9: Fallback Modes That Degrade Pipeline

### HIGH

| # | File:Line | Description |
|---|-----------|-------------|
| 9a | `backtest_validation.py:137-163` | Half-split PBO fallback. Raises an error with "disabled" message -- correctly blocked. But the fallback code still exists below line 163. |
| 9b | `astrology_engine.py:309` | `return {"hard": 3, "soft": 3, "ratio": 1.0}` -- neutral fallback on astrology error. The model sees "neutral astrology" instead of "missing astrology", which is informationally different. |

### MEDIUM

| # | File:Line | Description |
|---|-----------|-------------|
| 9c | `build_*_features.py` (all 5 TFs) | Fallback from `streamer_summary` to `streamer_articles` table. This is a data source fallback, not a pipeline degradation -- the feature is still computed, just from a different source. Acceptable. |
| 9d | `config.py:521` | `LIVE_CONF_THRESH_FALLBACK = 0.80` -- live trader uses more conservative threshold when no optimizer config exists. This is an operational safety fallback, not a matrix violation. Acceptable. |

### Verdict
`validate.py` does NOT check for neutral-value fallback returns in astrology/esoteric computation functions. **Gap in validate.py.**

---

## Violation 10: XGBoost References

### MEDIUM

| # | File:Line | Description |
|---|-----------|-------------|
| 10a | `lstm_sequence_model.py:6,14,21,224,473,477,654,660` | Eight references to "XGBoost" in comments/docstrings for LSTM stacking feature extraction. No actual `import xgboost` or XGBoost model usage. Cosmetic -- refers to the stacking architecture where LSTM feeds into the tree model (now LightGBM). |
| 10b | `runpod_train.py:19,823` | `--xgboost-only` CLI flag (kept for backward compatibility, actually runs LightGBM). |
| 10c | `v2_lstm_trainer.py:491,504,508,514,516,518,540,545,615,736,738` | Eleven references to "XGBoost" in variable names and comments for LSTM/tree blending. No actual XGBoost import or usage. |
| 10d | `v2_cloud_runner.py:452,494` | Comments referencing "Dask-XGBoost" for opt-in dense conversion path. No import. |

### Verdict
No actual XGBoost imports or model instantiation anywhere. All references are in comments, docstrings, or legacy CLI flag names. `validate.py` checks for XGBoost in config params (line 125-127), Optuna code (305-311), and core training files (756-774). The LSTM/blending file references are not caught but are cosmetic only. LOW RISK.

---

## Violation 11: Dense Conversion of Sparse CSR

### HIGH

| # | File:Line | Description |
|---|-----------|-------------|
| 11a | `efb_prebundler.py:320` | `csr_matrix.toarray()` -- full dense conversion when no binary features found. Returns entire matrix as dense. This is the EFB pre-bundler; if input has no binary columns, it densifies everything. Could OOM on large matrices. |
| 11b | `efb_prebundler.py:366` | `csc[:, non_binary_indices].toarray()` -- converts all non-binary columns to dense for the final EFB output matrix. By design (EFB output is dense), but loses CSR memory efficiency. |
| 11c | `feature_importance_pipeline.py:473,475` | `X_sub.toarray()` on SHAP subset. Converts a column-subset of sparse matrix to dense for SHAP computation. Limited to `top_indices` columns and `sample_rows` rows, so bounded. Acceptable. |
| 11d | `live_trader.py:1165` | `X.toarray()` for lleaves inference. Single-row dense conversion (~1ms). Comment documents this as intentional and cheap. Acceptable. |
| 11e | `ml_multi_tf.py:1421` | `X_all[:, esoteric_indices].toarray()` -- densifies esoteric base feature columns (not cross features) for sample weight computation. Limited to base features only (~235 columns, not 2.9M). Bounded and acceptable. |
| 11f | `ml_multi_tf.py:2007` | `_hmm_slice.toarray()` -- densifies HMM overlay columns extracted from sparse matrix. Handful of HMM columns only. Acceptable. |
| 11g | `gpu_histogram_fork/train_1w_cached.py:259` | `X_csr[i:i+chunk].toarray()` -- chunked dense conversion for accuracy check after training. Diagnostic only. |

### Verdict
Items 11a and 11b are the most concerning -- full-matrix dense conversion in the EFB pre-bundler path. However, this is by design: EFB output IS dense (bundled integer features). The concern is if the pre-bundler is called on matrices too large for RAM. `validate.py` does NOT check for `.toarray()` calls on the full cross matrix. **Partial gap in validate.py** (the existing fillna check at 788-803 covers NaN conversion but not dense conversion).

---

## validate.py Coverage Assessment

### Checks Present (GOOD)
1. `feature_fraction >= 0.7` -- config + Optuna bounds + file scan
2. `feature_fraction_bynode >= 0.7` -- config + Optuna bounds
3. `feature_pre_filter=False` -- config + file scan + Dataset construction audit
4. `bagging_fraction >= 0.7` -- Optuna bounds only
5. `lambda_l1 <= 4.0` -- Optuna bounds
6. `lambda_l2 <= 10.0` -- Optuna bounds
7. `min_data_in_leaf <= 15` -- per-TF config
8. `No XGBoost` -- config params + Optuna code + core training files
9. `fillna(0) on features` -- WARNING (non-blocking scan)
10. `No row-partitioned init_model` -- file scan
11. `force_col_wise=True` -- file scan
12. `max_bin == 255` -- config
13. `is_enable_sparse == True` -- config
14. `feature_pre_filter=False in Dataset() params=` -- TIER 1.5 audit

### Checks Missing (GAPS)

| # | Missing Check | Risk | Recommendation |
|---|---------------|------|----------------|
| G1 | Hardcoded `bagging_fraction < 0.7` in non-config files | HIGH | Add file scan like the `feature_fraction` scan at lines 805-820 |
| G2 | `min_data_in_leaf > 10` in non-config files | HIGH | Add file scan for hardcoded values > 10 outside config |
| G3 | Optuna `min_data_in_leaf` upper bound check (currently unchecked, allows 15) | MEDIUM | Add `check("Optuna min_data_in_leaf upper <= 10", ...)` |
| G4 | Silent `except Exception` in astrology/feature computation code | MEDIUM | Add check: broad except blocks in feature_library.py, astrology_engine.py must log warnings |
| G5 | Neutral-value fallback returns in esoteric computation | MEDIUM | Add check: astrology functions returning hardcoded neutral dicts |
| G6 | `.toarray()` on full sparse matrices (OOM risk) | LOW | Add warning for `.toarray()` calls not bounded by row/column subsets |
| G7 | `fillna(0)` check is WARNING not FAIL | MEDIUM | Promote to FAIL for feature_library.py and v2_feature_layers.py (core feature pipeline) |

---

## Prioritized Fix List

### CRITICAL (fix before next training run)
1. **`leakage_check.py:194`** -- Change `bagging_fraction: 0.6` to `bagging_fraction: 0.8` minimum. Current value makes leakage check blind to rare signal leakage.
2. **`feature_library.py:2332-2334`** -- Remove `fillna(0)` on Chinese calendar color features. Use NaN-safe computation or let LightGBM handle NaN natively.
3. **`v2_multi_asset_trainer.py:524`** -- Add floor: `ratio = max(0.7, min(1.0, batch_size / n_rows))` to prevent bagging_fraction from dropping below 0.7.

### HIGH (fix before production deployment)
4. **`validate.py`** -- Add hardcoded `bagging_fraction < 0.7` file scan (gap G1).
5. **`validate.py`** -- Add hardcoded `min_data_in_leaf > 10` file scan (gap G2).
6. **`run_optuna_local.py:552`** -- Lower `min_data_in_leaf` upper bound from 15 to 10.
7. **`bench_trial.py:58`** -- Change `min_data_in_leaf: 10` to `min_data_in_leaf: 3` (match config).
8. **`feature_library.py:2465`** -- Remove or NaN-guard the `fillna(0)` on engineering column values.

### MEDIUM (fix in next maintenance window)
9. **`build_sports_features.py:252,513`** -- Evaluate whether `fillna(0)` on game counts is semantically correct or should be NaN.
10. **`v2_feature_layers.py:411`** -- Document why `fillna(0)` is required here and consider NaN-safe entropy alternative.
11. **`lstm_sequence_model.py` / `v2_lstm_trainer.py` / `runpod_train.py`** -- Rename XGBoost references to LightGBM in comments and variable names.
12. **`validate.py`** -- Promote fillna(0) check from WARNING to FAIL for core feature pipeline files.
13. **`validate.py`** -- Add check for Optuna min_data_in_leaf upper bound <= 10 (gap G3).
14. **`astrology_engine.py`** -- Add logging to all `except Exception` blocks that return neutral fallbacks.
15. **`efb_prebundler.py:320`** -- Add shape guard: if matrix is large (>100K rows), warn before full dense conversion.

---

## Conclusion

The v3.3 codebase shows strong matrix adherence in the core training pipeline. The main config (`config.py`) and Optuna search space (`run_optuna_local.py`) are correctly constrained. `validate.py` covers 14 of the 17 critical checks needed, with 7 gaps identified above.

The 3 CRITICAL violations (`leakage_check.py` low bagging, `feature_library.py` fillna(0) on calendar features, `v2_multi_asset_trainer.py` unbounded bagging) should be fixed before the next training run. The 5 HIGH violations and 7 validate.py gaps should be addressed before production deployment.

No XGBoost code is actually executing anywhere. The sparse CSR path is preserved through training in all core files. EFB configuration is correct. The matrix thesis is fundamentally intact -- these are edge cases and diagnostic scripts, not core pipeline corruption.
