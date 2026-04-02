# OPT_LIVE_INFERENCE_EXPERT — Live Inference Audit Report

**Date:** 2026-03-30
**File:** `v3.3/live_trader.py` (1737 lines)
**Auditor:** Live Inference/Deployment Expert
**Scope:** Training/inference parity for 2.9M-feature BTC trading pipeline

---

## VERDICT: 7/8 PASS | 1 FINDING (MEDIUM)

---

## 1. Regime Computed BEFORE Crosses in Inference

**STATUS: PASS**

Lines 1037-1045: Regime is computed from `feat_dict` (which contains all base features + HMM) BEFORE `cross_computers[tf].compute()` is called at line 1053. The regime string is passed directly to the cross computer:

```python
# 8B.0: Compute regime BEFORE cross features.
_regime_for_cross, _regime_idx_early = detect_regime(
    _price_rc, _sma100_rc, _sma100_slope_rc, feat_dict=feat_dict)
...
cross_values, cross_ms = cross_computers[tf].compute(
    feat_dict, day_of_year=_doy, regime=_regime_for_cross)
```

The `InferenceCrossComputer.compute()` uses regime for regime-aware DOY crosses (`dw_N_B`, `dw_N_R`, `dw_N_S` at inference_crosses.py line 329-336). Ordering is correct and matches training.

---

## 2. HMM Uses Actual prev_close Not sma_5

**STATUS: PASS**

Lines 418-426: HMM fetches the actual previous daily close from the database:

```python
_ohlcv_hmm = live_dal.get_ohlcv_window('1d', 2)
if _ohlcv_hmm is not None and len(_ohlcv_hmm) >= 2:
    prev_close = float(_ohlcv_hmm['close'].iloc[-2])
else:
    prev_close = close  # fallback: log return = 0.0 (neutral)
ret = np.log(close / prev_close)
```

This correctly computes `log(close_t / close_{t-1})` which matches training in `ml_multi_tf.py`. The comment at line 417 explicitly documents the sma_5 bug that was fixed. The fallback (when only 1 bar available) returns log_return=0 which maps to neutral HMM state -- safe behavior.

---

## 3. Combo Context Formulas Persisted and Loaded

**STATUS: PASS**

**Persistence (inference_crosses.py lines 70-73):** `save_inference_artifacts()` accepts `combo_formulas` parameter and saves to `inference_{tf}_combo_formulas.json`.

**Loading (inference_crosses.py lines 215-220):** `InferenceCrossComputer.__init__()` loads combo formulas from disk. Falls back to empty dict for pre-fix artifacts.

**Resolution (inference_crosses.py lines 284-288, 353-369):** Pass 1a computes base contexts, pass 1b resolves combos as `ctx_binary[idx_a] & ctx_binary[idx_b]`. NaN propagation through combos is correct -- if either parent is NaN, combo is NaN (line 366-368).

---

## 4. NaN Propagation -- No Silent fillna(0) on Feature Data

**STATUS: PASS**

No `fillna(0)` or `fill_value=0` found anywhere in `live_trader.py`. NaN handling is correct throughout:

- **Base features (line 1105-1115):** `val = np.nan` for None/inf. NaN values are stored explicitly in the CSR matrix (line 1113: `fval != 0.0 or np.isnan(fval)`), ensuring LightGBM sees them as missing, not zero.
- **Cross features (inference_crosses.py line 377-379):** NaN propagates through crosses via `nan_cross_mask`. If either input context is NaN, the cross output is NaN.
- **HMM features (line 454-460):** On failure, returns NaN for all HMM features (not 0).
- **GCP features (line 228-233):** On failure, GCP columns stay NaN -- explicit comment confirms this.
- **Inf handling (line 1106-1107):** Inf values converted to NaN (correct -- inf is not a valid signal).

**Philosophy compliance: CONFIRMED.** NaN = "missing" (model learns split direction), 0 = "value is zero" (different signal). No conflation.

---

## 5. Sparse CSR Predict Path -- No Dense Conversion

**STATUS: PASS (with acceptable exception)**

Lines 1162-1167:
```python
if tf in _lleaves_tfs:
    raw_pred = models[tf].predict(X.toarray())  # lleaves requires dense
else:
    raw_pred = models[tf].predict(X)  # lgb.Booster accepts sparse directly
```

- **LightGBM Booster:** Receives `scipy.sparse.csr_matrix` directly. No `.toarray()` or `.todense()` call. CORRECT.
- **lleaves compiled models:** Uses `.toarray()` which is necessary -- lleaves (the compiled LLVM model) does not accept sparse input. Comment documents this: "Single-row .toarray() is cheap (~1ms even for 6M features)." This is an acceptable tradeoff for the 5.4x prediction speedup from lleaves.

The CSR construction itself (lines 1092-1136) correctly builds a single-row sparse matrix with:
- Non-zero base features stored explicitly
- NaN stored explicitly (so LightGBM treats as missing)
- Structural zeros = 0.0 (cross didn't fire) -- implicit in CSR
- Cross features: only non-zeros stored via `_nz_mask`

---

## 6. V2_CROSS + V2_LAYERS Mandatory Imports

**STATUS: FINDING (MEDIUM) -- Indirect Import via inference_crosses.py**

`live_trader.py` does NOT directly import `v2_cross_generator` or `v2_feature_layers`. Instead, it uses:
- `from inference_crosses import InferenceCrossComputer` (line 33) -- handles cross features
- `from feature_library import build_all_features` (line 28) -- handles base features

`feature_library.py` itself does NOT import `v2_cross_generator` or `v2_feature_layers` at the top level. At line 5395-5398 it explicitly SKIPS trend cross features, stating they are "handled by v2_cross_generator" during training.

The build pipeline (`build_features_v2.py` lines 58-59) imports both:
```python
from v2_feature_layers import add_all_v2_layers
from v2_cross_generator import generate_all_crosses
```

**Assessment:** This is architecturally correct for the inference path. At inference time:
1. `feature_library.build_all_features()` computes ~300 base features
2. `InferenceCrossComputer.compute()` reconstructs 2.9M cross features from persisted artifacts (thresholds, index pairs, combo formulas)

The cross features are NOT recomputed from scratch at inference -- they're reconstructed from training artifacts. This is the designed architecture. V2_CROSS and V2_LAYERS are training-time dependencies, not inference-time dependencies.

**However:** If the inference artifacts are missing or stale, `InferenceCrossComputer.__init__()` will raise `FileNotFoundError` and the cross computer won't load. Line 815 catches this and prints a WARNING but does NOT halt. Then at line 1156-1159, if `cross_values is None` and cross features exist, the bar is correctly HALTED ("cannot predict without the matrix"). This is safe.

**Risk:** If `v2_feature_layers.add_all_v2_layers()` adds base features during training that `feature_library.build_all_features()` does NOT produce at inference, those features will be NaN at inference. This would be a parity bug but is outside the scope of `live_trader.py` alone -- it depends on whether `feature_library.py` and `v2_feature_layers.py` produce the same base feature set.

---

## 7. No Blanket try/except Masking Failures

**STATUS: PASS (with minor notes)**

Analyzed all 47 try/except blocks in `live_trader.py`:

**Correctly handled:**
- `compute_features_live()` (line 243-250): Catches, prints full traceback, RE-RAISES. Caller at line 1022-1028 handles the exception by skipping the bar with a visible error message.
- Cross feature failure (line 1156-1159): If cross_values is None AND cross features should exist, the bar is HALTED with a printed message. Does NOT fall through to prediction.
- Main loop (line 1727-1730): Catches Exception, prints traceback, sleeps 30s, continues. This is correct for a long-running daemon -- crash recovery with visible error.

**Acceptable silent catches:**
- Lines 126, 131: Bare `except:` for ALTER TABLE (schema migration -- column already exists is expected)
- Lines 996, 1016, 1071, 1081, 1146, 1206, 1266, 1285, 1302, 1518: `except Exception:` around `log_rejected_trade()` calls. These are logging-only -- failure to LOG a rejection should not prevent the rejection itself.
- Line 1504: Writing prediction_cache.json -- non-critical.
- Line 541: Circuit breaker DB access -- comment explains "DB access failure should not block."

**One concern (MINOR):**
- **Line 1343:** `except Exception: pass` on meta-labeling. Comment says "meta failed, allow trade through." This means a crash in `predict_meta()` silently degrades to trading without the meta gate. This is a design choice (availability over safety) but should at minimum print the error:

```python
except Exception as _meta_err:
    print(f"  WARNING: Meta-labeling failed for {tf}: {_meta_err} — proceeding without meta gate")
```

---

## 8. Feature Column Order Matches Training Exactly

**STATUS: PASS (with runtime validation)**

Lines 769-804: At startup, for each TF with a non-lleaves model, the code:
1. Calls `m.feature_name()` to get the model's expected feature order
2. Compares against `features_list[tf]` loaded from JSON
3. If mismatch: **CORRECTS** by using model's order (authoritative), prints CRITICAL warning, writes mismatch log

```python
if model_feature_names and features_list[tf] != model_feature_names:
    ...
    features_list[tf] = model_feature_names  # Use model's order
```

The sparse CSR construction (lines 1092-1136) uses `feat_name_to_idx[tf]` which is built from `features_list[tf]` at startup (line 825). Since `features_list[tf]` is corrected to match the model if there's a mismatch, the column indices in the CSR matrix will match the model's expectations.

For lleaves compiled models: `feature_name()` is not available (`hasattr(m, 'feature_name')` returns False), so validation is skipped. The comment says "trust pruned features file." This is a minor risk if the pruned features file gets out of sync, but acceptable since lleaves models are compiled from the same LightGBM model.

---

## SUMMARY TABLE

| # | Check | Status | Notes |
|---|-------|--------|-------|
| 1 | Regime before crosses | PASS | Lines 1037-1045 before line 1053 |
| 2 | HMM uses prev_close | PASS | Fetches actual close from DB, not sma_5 |
| 3 | Combo formulas persisted | PASS | JSON save/load/resolve with NaN propagation |
| 4 | No silent fillna(0) | PASS | NaN preserved throughout pipeline |
| 5 | Sparse CSR predict | PASS | lgb.Booster gets CSR directly; lleaves needs dense (acceptable) |
| 6 | V2_CROSS + V2_LAYERS | MEDIUM | Indirect via inference_crosses artifacts -- architecturally correct but feature parity with v2_feature_layers unverified |
| 7 | No blanket try/except | PASS | Meta-labeling except should print error (minor) |
| 8 | Column order match | PASS | Runtime validation + auto-correction at startup |

---

## RECOMMENDED FIXES

### FIX 1 (MINOR): Log meta-labeling failures instead of silent pass

**File:** `v3.3/live_trader.py` line 1343

Current:
```python
except Exception:
    pass  # meta failed, allow trade through
```

Recommended:
```python
except Exception as _meta_err:
    print(f"  WARNING: Meta-labeling failed for {tf}: {_meta_err} -- proceeding without meta gate")
```

### FIX 2 (RECOMMENDED): Verify v2_feature_layers parity with feature_library

Run a one-time check: build features with `feature_library.build_all_features()` AND `v2_feature_layers.add_all_v2_layers()` on the same OHLCV data. Diff the column sets. Any columns in v2_feature_layers but NOT in feature_library means the model was trained on features that are always NaN at inference.

---

## ARCHITECTURE NOTES

The inference pipeline is well-designed:
1. **Base features** via `feature_library.build_all_features()` (~300 features)
2. **HMM features** via `get_hmm_features()` (~4 features)
3. **Regime** computed from base features (before crosses)
4. **Cross features** via `InferenceCrossComputer.compute()` (~2.9M features, ~20ms)
5. **Sparse CSR assembly** with O(1) positional lookup via pre-built index maps
6. **Prediction** directly on sparse matrix (LightGBM) or dense (lleaves)

The mandatory cross feature gate at line 1156-1159 correctly enforces the matrix thesis: no prediction without the full 2.9M-feature matrix.
