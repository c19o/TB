# Validation Warnings Log
**Last Run:** 2026-04-01  
**Validator:** QA Lead  
**Status:** 95/95 passed ✓, 2 warnings

---

## BLOCKERS (Must Fix Before Training)

### SAV-39: LightGBM DLL Missing ✓ RESOLVED
**Status:** ✓ Fixed  
**Issue:** `FileNotFoundError: Could not find module 'lib_lightgbm.dll'`  
**Resolution:** Reinstalled LightGBM 4.6.0 via pip  
**Verified:** `python -c "import lightgbm"` succeeds

---

## WARNINGS (Non-Blocking, Review Recommended)

### 1. Potential NaN→0 Conversions
**Severity:** Medium  
**Impact:** May alter model behavior (NaN = missing, 0 = value)

**Files flagged:**
- `backtesting_audit.py:250`
- `backtesting_audit.py:755`
- `backtesting_audit.py:1982`
- `build_4h_features.py:597`
- `build_4h_features.py:598`

**Note:** LightGBM treats NaN as "missing" (learns split direction), but explicit 0 means "value is zero". These are different signals. The v3.3 pipeline philosophy is to preserve NaN for feature data.

**Action:** Review each location. If it's feature data going to LightGBM, keep NaN. If it's LSTM preprocessing or display logic, fillna(0) may be intentional.

### 2. cloud_run_tf.py Missing ALLOW_CPU=1 Documentation
**Severity:** Low  
**Impact:** Deployment on CUDA 13+ may fail without env var set

**Context:** CUDA 13+ drops cuDF support. LightGBM falls back to pandas, which requires `ALLOW_CPU=1` environment variable.

**Current state:** cloud_run_tf.py does not mention this requirement in comments or setup.

**Action:** Add to cloud_run_tf.py header:
```python
# CUDA 13+ requirement: export ALLOW_CPU=1 before running
# (cuDF dropped, pandas fallback needs explicit CPU allowance)
```

### 3. 1w Cloud Runtime Monitoring Contract Added
**Severity:** Low  
**Impact:** Runtime observability for cloud 1w now writes heartbeat + checkpoint integrity signals; not yet reflected in acceptance summary workflows.

**Files updated:** `cloud_run_tf.py`  
**Observed behavior:** `cloud_run_1w_heartbeat.json` is now emitted when running `cloud_run_tf.py --tf 1w`.
- Progress step transitions + stall protection
- Resource snapshots (RAM/CPU/disk and GPU if present)
- Runtime contract values
- Monitoring of 1w cross-checkpoint pair integrity in `_cross_checkpoint_1w_*.npz` families

**Action:** Keep monitoring alerts and heartbeat artifacts in production runbook for alerting thresholds and retention policy.

---

## Validation Summary

**Total checks:** 95  
**Passed:** 95 ✓  
**Failed:** 0  
**Warnings:** 2 (NaN conversions, ALLOW_CPU doc)

**Status:** ✓ TRAINING APPROVED - All critical checks pass

**Recommendation:** Warnings are non-blocking. Can be addressed in subsequent cleanup pass.

---

**Last validation:** 2026-04-01 (after SAV-39 fix)  
**Full report:** Run `python validate.py` from v3.3/
