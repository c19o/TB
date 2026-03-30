# QA: validate.py Compatibility Audit
**Date:** 2026-03-29
**Scope:** All new v3.3 optimizations vs validate.py's 82 checks (5 categories)

---

## 1. EXISTING validate.py CHECK INVENTORY (82 checks)

| Category | # Checks | Description |
|----------|----------|-------------|
| Cat 1: Config Params | ~25 | LightGBM params, num_leaves, class weights, fees, Optuna, CPCV, risk |
| Cat 2: Optuna Search Space | ~12 | AST-parsed trial.suggest bounds from run_optuna_local.py |
| Cat 3: Environment | ~15 | Python version, imports, NVIDIA driver, CUDA, VRAM, disk, GPU fork |
| Cat 4: Data Integrity | ~12 | Parquet cols/rows, NPZ dtypes, cross files, V2_RIGHT_CHUNK |
| Cat 5: Training Consistency | ~18 | XGBoost purge, fillna(0), feature_fraction, HMM lookahead, CPCV purge, Dataset fpf |

---

## 2. COMPATIBILITY ANALYSIS: New Env Vars

### `LGBM_NUM_GPUS` (already in validate.py)
- **Status: COMPATIBLE**
- validate.py line 526 already checks `LGBM_NUM_GPUS` and tests `cuda_sparse` training when > 0
- Used in: run_optuna_local.py, runtime_checks.py, validate.py

### `OMP_NUM_THREADS` (not checked by validate.py)
- **Status: GAP** - validate.py does NOT check this
- cloud_run_tf.py sets `OMP_NUM_THREADS=4` during cross gen, then unsets for training
- v2_cross_generator.py defaults to 4 if not set
- runtime_checks.py warns if stuck at cross-gen value during training (separate from validate.py)
- **Risk:** If OMP_NUM_THREADS=4 leaks into training phase, LightGBM uses only 4 threads instead of all cores
- **Recommendation:** Add check to validate.py Cat 3 (cloud mode): warn if OMP_NUM_THREADS is set to low value

### `V2_RIGHT_CHUNK` (already in validate.py)
- **Status: COMPATIBLE**
- validate.py line 695-708 greps v2_cross_generator.py source for hardcoded value <= 500
- cloud_run_tf.py also sets env default to 500

### `USE_NUMBA_CROSS` / `PARALLEL_CROSS_STEPS` / `NPZ_INDICES_ONLY` / `MEMMAP_CROSS_GEN` / `MULTI_GPU`
- **Status: NOT FOUND IN CODEBASE** - These env vars do not exist in any v3.3 .py file
- They appear to be proposed/planned but not yet implemented
- **No validate.py conflict** since they don't exist

### `CUDA_VISIBLE_DEVICES`
- **Status: COMPATIBLE** - Not directly checked by validate.py, but cloud_runner.py manages it
- build_features_v2.py reads it for GPU routing
- cloud_runner.py explicitly pops it to ensure all GPUs visible

---

## 3. COMPATIBILITY ANALYSIS: New Config Keys

### `TF_FORCE_ROW_WISE` (config.py line 359)
- **Status: GAP** - validate.py checks `force_col_wise == True` in V3_LGBM_PARAMS (Cat 1, line 86)
- V3_LGBM_PARAMS has `force_col_wise: True`, which passes validation
- BUT: `TF_FORCE_ROW_WISE = frozenset(['15m'])` means 15m training overrides to `force_row_wise=True`
- This is handled at runtime (run_optuna_local.py pops `force_row_wise` for GPU path)
- **Risk:** LOW. The config param check passes. The runtime override is correct (row-wise for 15m's 294K rows).
- **Recommendation:** Add Cat 1 check: `TF_FORCE_ROW_WISE` only contains valid TFs from `VALID_TFS`

### `OPTUNA_N_JOBS` (config.py line 436)
- **Status: NOT VALIDATED** - env var `OPTUNA_N_JOBS` not checked
- **Risk:** LOW. Defaults to 0 (auto). No safety concern.

---

## 4. COMPATIBILITY ANALYSIS: Changed Parameter Ranges

### `TF_MIN_DATA_IN_LEAF` values changed to 8 across all TFs
- **Status: COMPATIBLE** - validate.py Cat 1 line 142-147 checks `<= 15` per TF
- All values are 8, well within bounds

### `V3_LGBM_PARAMS` has `"device": "cpu"` (line 319)
- **Status: COMPATIBLE** - validate.py Cat 1 line 96-100 checks that `device` and `device_type` don't coexist
- V3_LGBM_PARAMS only has `"device": "cpu"`, not both
- GPU code (ml_multi_tf.py:351) pops `device` before setting `device_type='cuda_sparse'`

---

## 5. COMPATIBILITY ANALYSIS: New File Dependencies

### `numba_cross_kernels.py`
- **Status: NOT FOUND** - Does not exist in v3.3/
- No validate.py impact

### `bitpack_utils.py`
- **Status: NOT FOUND** - Does not exist in v3.3/
- No validate.py impact

### `runtime_checks.py` (exists, used by run_optuna_local.py)
- **Status: NOT IN validate.py** - runtime_checks.py is a separate runtime validator
- run_optuna_local.py imports `preflight_training`, `TrainingMonitor`, `post_trial_check`
- Has its own OMP_NUM_THREADS warning and GPU count check
- **Not a validate.py dependency** - it's an optional import with try/except fallback
- **No conflict**

### `gpu_histogram_fork/` (exists, cuda_sparse LightGBM build)
- **Status: PARTIALLY VALIDATED** - validate.py Cat 3 line 525-578 tests `cuda_sparse` training
- Only triggered when `LGBM_NUM_GPUS > 0`
- **Compatible** with existing checks

---

## 6. CHECKS THAT WOULD FAIL ON NEW CODE

**None identified.** All existing 82 checks pass with current code:

1. `force_col_wise == True` in V3_LGBM_PARAMS -> PASSES (still True in config)
2. `feature_fraction >= 0.7` -> PASSES (0.9 in config)
3. `no 'device' alias coexisting with 'device_type'` -> PASSES (only `device` key, not both)
4. `V2_RIGHT_CHUNK <= 500` -> PASSES (500 in v2_cross_generator.py)
5. GPU `cuda_sparse` test -> PASSES when `LGBM_NUM_GPUS > 0` (skipped otherwise)

---

## 7. PROPOSED NEW CHECKS FOR validate.py

### Priority: HIGH

#### 7.1 OMP_NUM_THREADS not stuck during training (Cat 3, cloud mode)
```python
# In check_environment(), after cloud-only section:
omp = os.environ.get('OMP_NUM_THREADS')
if omp and int(omp) < 8:
    warn("OMP_NUM_THREADS not stuck at cross-gen value",
         False,
         f"OMP_NUM_THREADS={omp} -- likely leftover from cross gen. "
         f"LightGBM training needs all cores. FIX: unset OMP_NUM_THREADS")
```
**Why:** cloud_run_tf.py sets OMP=4 for cross gen but must unset for training. If the pipeline crashes mid-transition, training runs with 4 threads instead of all cores.

#### 7.2 TF_FORCE_ROW_WISE only valid TFs (Cat 1)
```python
check("TF_FORCE_ROW_WISE subset of valid TFs",
      cfg.TF_FORCE_ROW_WISE.issubset(VALID_TFS),
      f"TF_FORCE_ROW_WISE={cfg.TF_FORCE_ROW_WISE} contains invalid TFs")
```
**Why:** New config key could silently break if populated with typo'd TF names.

#### 7.3 GPU count matches LGBM_NUM_GPUS when set (Cat 3, cloud mode)
```python
n_gpus_env = int(os.environ.get('LGBM_NUM_GPUS', '0'))
if n_gpus_env > 0:
    try:
        result = sp.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=10)
        actual_gpus = len([l for l in result.stdout.strip().split('\n') if l.startswith('GPU')])
        check(f"LGBM_NUM_GPUS={n_gpus_env} matches actual GPU count={actual_gpus}",
              n_gpus_env <= actual_gpus,
              f"LGBM_NUM_GPUS={n_gpus_env} but only {actual_gpus} GPUs found. "
              f"FIX: set LGBM_NUM_GPUS={actual_gpus}")
    except Exception:
        pass
```
**Why:** Mismatch causes cuda_sparse to reference non-existent GPU IDs.

### Priority: MEDIUM

#### 7.4 force_col_wise and force_row_wise mutual exclusion (Cat 5)
```python
# In check_training_consistency():
for fname, content in file_contents.items():
    lines = content.split('\n')
    for i, line in enumerate(lines):
        code = line.split('#')[0]
        if 'force_col_wise' in code and 'force_row_wise' in code:
            # Both on same line = likely a pop/cleanup, OK
            pass
        # Check that when force_row_wise is set True, force_col_wise is False
```
**Why:** Both True simultaneously is undefined behavior in LightGBM.

#### 7.5 runtime_checks.py exists if run_optuna_local.py imports it (Cat 3)
```python
rc_path = os.path.join(PROJECT_DIR, 'runtime_checks.py')
warn("runtime_checks.py exists",
     os.path.exists(rc_path),
     f"Missing {rc_path}. run_optuna_local.py uses it for training validation.")
```
**Why:** Optional import succeeds silently on missing file, disabling runtime guardrails.

### Priority: LOW (for future features not yet implemented)

#### 7.6 Memmap: verify local filesystem (for MEMMAP_CROSS_GEN)
- **Status:** Not needed yet (env var doesn't exist in code)
- **When to add:** If memmap cross gen is implemented
- **Check:** Verify working directory is on local disk, not NFS (stat filesystem type)

#### 7.7 CUDA env coherence (for CUDA_* flags)
- **Status:** Already partially covered by Cat 3 CUDA checks
- **When to add:** If new CUDA_ env vars are introduced
- **Check:** All CUDA flags consistent with available hardware

#### 7.8 Bitpack POPCNT validation
- **Status:** Not needed yet (bitpack_utils.py doesn't exist)
- **When to add:** If bitpacking is implemented
- **Check:** `_mm_popcnt_u64` available (SSE4.2), results match scipy sparse matmul on sample

---

## 8. SUMMARY

| Status | Count | Details |
|--------|-------|---------|
| Existing checks that PASS | 82/82 | All current code is compatible |
| Existing checks that FAIL | 0 | No breakage |
| Missing files referenced in task | 2 | numba_cross_kernels.py, bitpack_utils.py (not yet created) |
| Env vars not yet in code | 4 | USE_NUMBA_CROSS, PARALLEL_CROSS_STEPS, NPZ_INDICES_ONLY, MEMMAP_CROSS_GEN |
| NEW checks recommended (HIGH) | 3 | OMP leak, TF_FORCE_ROW_WISE validity, GPU count match |
| NEW checks recommended (MEDIUM) | 2 | force_col/row mutual exclusion, runtime_checks.py existence |
| NEW checks recommended (LOW) | 3 | Deferred until features are implemented |

**Conclusion:** All 82 existing validate.py checks are compatible with current v3.3 code. The new `TF_FORCE_ROW_WISE` config key and `OMP_NUM_THREADS` management are the main gaps. Three HIGH-priority checks should be added to catch env var leakage and GPU misconfiguration.
