# OPT_VALIDATE_EXPERT -- validate.py Audit Report

**Date:** 2026-03-30
**File:** `v3.3/validate.py` (1152 lines, 96 checks across 5 categories)
**Cross-referenced:** `v3.3/config.py`, `v3.3/CLAUDE.md`

---

## VERIFICATION OF 10 REQUIRED CHECKS

| # | Check | Status | Location | Notes |
|---|-------|--------|----------|-------|
| 1 | `feature_fraction >= 0.7` | PRESENT | L66-69 (config), L234-246 (Optuna), L805-820 (hardcoded scan) | Triple coverage: default, search space, and grep across all .py files |
| 2 | `bagging_fraction >= 0.7` | **PARTIAL** | L256-262 (Optuna only) | **MISSING: no check on config.py default value (currently 0.95, safe but unguarded)** |
| 3 | `feature_pre_filter=False` | PRESENT | L77-80 (config), L776-786 (training files), L994-1039 (Dataset construction) | Excellent: checks config, training files, AND per-Dataset-call verification |
| 4 | `min_data_in_leaf` caps per TF | PRESENT | L143-148 | Checks all TFs against <= 15 ceiling |
| 5 | `lambda_l1 <= 4.0` | **PARTIAL** | L264-271 (Optuna only) | **MISSING: no check on config.py default value (currently 0.5, safe but unguarded)** |
| 6 | `min_data_in_bin=1` | PRESENT | L115-118 | Direct config check |
| 7 | `is_enable_sparse=True` | PRESENT | L92-95 | Direct config check |
| 8 | LightGBM only (no XGBoost) | PRESENT | L125-127 (params), L305-311 (Optuna), L756-774 (all training files) | Triple coverage across config, Optuna, and all .py files |
| 9 | No 5m timeframe | PRESENT | L163-165 | Checks both TIMEFRAMES_ALL_ASSETS and TIMEFRAMES_CRYPTO_ONLY |
| 10 | `CPCV_PARALLEL_GPUS` config | **MISSING** | N/A | No validation check exists for this newly added config |

---

## MISSING CHECKS (must be added)

### M-1: `bagging_fraction >= 0.7` on config default (SIGNAL-KILLING risk)
- **Current state:** Only checked in Optuna search space bounds (L256-262). The actual `V3_LGBM_PARAMS['bagging_fraction']` default is NOT validated.
- **Risk:** If someone sets `bagging_fraction=0.5` in config.py, validation passes. 50% row dropout gives P(10-fire in bag) = 0.001 -- rare signals vanish.
- **Fix:** Add to `check_config_params()`:
```python
check("bagging_fraction >= 0.7",
      p.get('bagging_fraction', 0) >= 0.7,
      f"bagging_fraction={p.get('bagging_fraction')} -- must be >= 0.7. "
      f"50% row dropout destroys rare esoteric signals (P(10-fire)=0.001 at bf=0.5). "
      f"FIX: config.py V3_LGBM_PARAMS")
```

### M-2: `lambda_l1 <= 4.0` on config default (SIGNAL-KILLING risk)
- **Current state:** Only checked in Optuna search space upper bound (L264-271). The actual `V3_LGBM_PARAMS['lambda_l1']` default is NOT validated.
- **Risk:** If someone sets `lambda_l1=10.0` in config.py, validation passes. L1 > 4 zeros leaf weights for signals firing < 15 times.
- **Fix:** Add to `check_config_params()`:
```python
check("lambda_l1 <= 4.0",
      p.get('lambda_l1', 999) <= 4.0,
      f"lambda_l1={p.get('lambda_l1')} -- must be <= 4.0. "
      f"L1 > 4 zeros leaf weights for signals firing < 15 times. "
      f"FIX: config.py V3_LGBM_PARAMS")
```

### M-3: `lambda_l2 <= 10.0` on config default
- **Current state:** Only checked in Optuna search space (L273-279). Config default (3.0) is safe but unguarded.
- **Fix:** Add to `check_config_params()`:
```python
check("lambda_l2 <= 10.0",
      p.get('lambda_l2', 999) <= 10.0,
      f"lambda_l2={p.get('lambda_l2')} -- must be <= 10.0. "
      f"FIX: config.py V3_LGBM_PARAMS")
```

### M-4: `CPCV_PARALLEL_GPUS` validation
- **Current state:** Config exists at `config.py:465` but validate.py has zero checks on it.
- **Risk:** Invalid values (negative, or exceeding actual GPU count on cloud) could cause silent failures or resource contention.
- **Fix:** Add to `check_config_params()`:
```python
check("CPCV_PARALLEL_GPUS >= 0",
      cfg.CPCV_PARALLEL_GPUS >= 0,
      f"CPCV_PARALLEL_GPUS={cfg.CPCV_PARALLEL_GPUS} -- must be >= 0 (0=auto-detect). "
      f"FIX: config.py or CPCV_PARALLEL_GPUS env var")
```
- Add to `check_environment()` (cloud mode):
```python
if cfg.CPCV_PARALLEL_GPUS > 0:
    # Verify requested GPUs <= available GPUs
    try:
        result = sp.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                        capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            n_gpus_available = len(result.stdout.strip().split('\n'))
            check(f"CPCV_PARALLEL_GPUS <= available GPUs ({cfg.CPCV_PARALLEL_GPUS} <= {n_gpus_available})",
                  cfg.CPCV_PARALLEL_GPUS <= n_gpus_available,
                  f"Requested {cfg.CPCV_PARALLEL_GPUS} GPUs but only {n_gpus_available} available. "
                  f"FIX: set CPCV_PARALLEL_GPUS={n_gpus_available} or 0 for auto-detect")
    except Exception:
        pass
```

### M-5: `bagging_freq` must be > 0 when `bagging_fraction < 1.0`
- **Current state:** No check. LightGBM silently ignores `bagging_fraction` if `bagging_freq=0`.
- **Risk:** Setting `bagging_freq=0` disables bagging entirely, making `bagging_fraction` a no-op. No error, just silent behavior change.
- **Fix:**
```python
check("bagging_freq > 0 (enables bagging_fraction)",
      p.get('bagging_freq', 0) > 0,
      f"bagging_freq={p.get('bagging_freq')} -- must be > 0 for bagging_fraction to take effect. "
      f"FIX: config.py V3_LGBM_PARAMS, set bagging_freq=1")
```

### M-6: `feature_fraction_bynode` hardcoded scan (like feature_fraction)
- **Current state:** `feature_fraction` has a hardcoded-value grep across all .py files (L805-820). `feature_fraction_bynode` does NOT.
- **Risk:** A training script could hardcode `feature_fraction_bynode=0.3` and validation would miss it.
- **Fix:** Add companion scan in `check_training_consistency()`:
```python
ffbn_pattern = re.compile(r"['\"]?feature_fraction_bynode['\"]?\s*[:=]\s*(0\.\d+)")
low_ffbn_files = []
for fname, content in all_py_contents.items():
    if fname == 'validate.py':
        continue
    lines = content.split('\n')
    for i, line in enumerate(lines):
        code_part = line.split('#')[0]
        for match in ffbn_pattern.finditer(code_part):
            val = float(match.group(1))
            if val < 0.7:
                low_ffbn_files.append(f"{fname}:{i+1} (val={val})")
check("no hardcoded feature_fraction_bynode < 0.7",
      len(low_ffbn_files) == 0,
      f"Low feature_fraction_bynode in: {', '.join(low_ffbn_files[:5])}. Must be >= 0.7.")
```

### M-7: `path_smooth` cap (minor but defensive)
- **Current state:** No check. Config default is 0.5 (good). But `path_smooth=10.0` would over-regularize rare leaf predictions toward parent, killing esoteric signal edges.
- **Fix:**
```python
check("path_smooth <= 2.0",
      p.get('path_smooth', 0) <= 2.0,
      f"path_smooth={p.get('path_smooth')} -- must be <= 2.0. "
      f"Higher values over-regularize rare leaf predictions. FIX: config.py V3_LGBM_PARAMS")
```

---

## EXISTING CHECKS -- STRENGTH ASSESSMENT

### Excellent Coverage (no action needed)
- **feature_pre_filter=False**: Triple-layered (config default + training files + per-Dataset-call AST scan). Best check in the file.
- **feature_fraction >= 0.7**: Triple coverage (config + Optuna + hardcoded grep). Model protection.
- **No XGBoost**: Checked in config params, Optuna file, and all core training files.
- **No 5m**: Checks both timeframe lists.
- **Per-TF num_leaves caps**: All 5 TFs checked with appropriate ceilings.
- **Class weight alignment (np.pad bug)**: Catches the exact historical bug pattern.
- **HMM lookahead in parallel CPCV (T-2)**: Checks 3 distinct code patterns.
- **CPCV purge / max_hold_bars**: Checks for hardcoded values AND verifies TRIPLE_BARRIER_CONFIG import.

### Good Coverage (minor gaps)
- **min_data_in_leaf**: Checks all TFs against <= 15 but does not verify minimum (e.g., min_data_in_leaf=0 would pass -- LightGBM defaults to 20 if 0, so not a real risk).
- **num_threads=0**: Correct check. -1 is undocumented.

---

## SUMMARY

| Metric | Value |
|--------|-------|
| Required checks present | 7/10 fully present |
| Partially present | 2/10 (bagging_fraction, lambda_l1 -- Optuna only, no config default check) |
| Missing | 1/10 (CPCV_PARALLEL_GPUS) |
| Additional missing checks found | 4 (lambda_l2 default, bagging_freq, feature_fraction_bynode scan, path_smooth cap) |
| **Total checks to add** | **7** |
| Signal-killing risk if not added | **HIGH for M-1 and M-2** (bagging_fraction and lambda_l1 config defaults unguarded) |

**Priority order for implementation:**
1. M-1: bagging_fraction config default (signal-killing)
2. M-2: lambda_l1 config default (signal-killing)
3. M-5: bagging_freq > 0 (silent no-op)
4. M-6: feature_fraction_bynode hardcoded scan (consistency with feature_fraction)
5. M-4: CPCV_PARALLEL_GPUS (newly added, unvalidated)
6. M-3: lambda_l2 config default (defensive)
7. M-7: path_smooth cap (defensive)
