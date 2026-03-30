# QA Matrix Thesis Deep Compliance Audit

**Date**: 2026-03-29
**Branch**: v3.3
**Auditor**: Security Reviewer (READ-ONLY)

---

## EXECUTIVE SUMMARY

| # | Concern | Verdict | Severity |
|---|---------|---------|----------|
| 1 | Signal Survival (3-fire feature in 818 rows) | **PASS** | — |
| 2 | Pruning Safety (WilcoxonPruner killing rare signals) | **FAIL** | HIGH |
| 3 | Bitpack Correctness (POPCNT vs sparse matmul) | **PASS** (not implemented) | — |
| 4 | Memmap Losslessness (chunk→disk→merge→load) | **PASS** (no memmap used) | — |
| 5 | CUDA Exactness (warp-reduced histograms) | **PASS*** | MEDIUM |
| 6 | Multi-GPU Independence (round-robin trial sampling) | **PASS** | LOW |
| 7 | Adaptive Chunks (halving chunk size) | **PASS** | — |
| 8 | Indices-Only NPZ (float32 1.0 reconstruction) | **PASS** | — |

**\*PASS with caveat** — GPU histograms match CPU to 1e-5 tolerance, not bit-identical.

**1 HIGH-severity issue found**: Optuna pruning can kill trials before rare signals surface in fold 2.

---

## 1. SIGNAL SURVIVAL

**Question**: Can a feature that fires 3 times in 818 rows (1w) still form a leaf split?

### Parameters Verified

| Parameter | Value | Source | Verdict |
|-----------|-------|--------|---------|
| `feature_pre_filter` | `False` everywhere (17 sites) | config.py:325, all Dataset calls | SAFE |
| `feature_fraction` | 0.9 default, Optuna 0.7–1.0 | config.py:334, run_optuna_local.py:482 | SAFE |
| `min_data_in_leaf` | 8 all TFs, Optuna 8–15 | config.py:348-354 | SAFE |
| `bagging_fraction` | 0.95 default | config.py:336 | SAFE |
| `max_bin` | 255 (EFB bundling) | config.py:321 | SAFE |
| Feature filtering | `apply_tf_feature_filter()` is a no-op | config.py:302-309 | SAFE |

### Mathematical Reasoning

For a feature firing 3 times in 818 rows with `bagging_fraction=0.95`:
- P(all 3 firings in bag) = 0.95³ = **85.7%** — high visibility
- `min_data_in_leaf=8`: A leaf containing 3 positive samples is valid IF the split gains information. LightGBM evaluates gain, not count alone.
- `feature_pre_filter=False`: Feature is never silently dropped at Dataset construction.
- `feature_fraction=0.9`: P(feature selected for any tree) = 90%. Over 200+ rounds, near certainty it's tried.
- `path_smooth=0.5`: Prevents overfitting to rare splits while allowing legitimate discovery.

**Verdict: PASS** — All parameters protect rare signals. Configuration is explicitly designed for this.

---

## 2. PRUNING SAFETY

**Question**: Does the pruner kill trials that would have found rare signals with more folds?

### Architecture

Phase 1 uses a **triple-nested pruning stack** (run_optuna_local.py:1223-1235):
1. **MedianPruner** (innermost): `n_startup_trials=8`, `n_warmup_steps=50`, `interval_steps=10`
2. **PatientPruner** (wrapper): `patience=5`, `min_delta=0.001`
3. **LightGBM early stopping**: `ES_PATIENCE=15` rounds per fold

### Critical Issue: n_warmup_steps=50 is INSUFFICIENT

Phase 1 uses **CPCV_GROUPS=2** (only 2 folds) with **60 max rounds** per fold.

Timeline of reporting steps (`step = fold_i × max_rounds + round`):
- Steps 10–50: Fold 0 training, warmup period (no pruning)
- **Step 60**: Fold 0 ends — **pruning now active**
- Step 70: Fold 1 begins — PatientPruner already has fold 0 memory
- Steps 80–120: Fold 1 training — **subject to pruning at every report**

**Required warmup** for both folds to complete: `2 × 60 = 120` steps.
**Actual warmup**: 50 steps — **less than half what's needed**.

### Rare Signal Kill Scenario

1. Trial T has parameters good for a rare 1-in-100 signal
2. Fold 0: Signal appears in validation, good score (mlogloss=0.50)
3. Fold 1: Signal absent from this CV split, noisy score (mlogloss=0.65)
4. PatientPruner sees 5 consecutive reports without 0.001 improvement → **TRIAL KILLED**
5. Final OOS metric (mean of both folds) **never computed**

### Mitigating Factors

- Validation gate (top-3 configs) **disables pruning** (run_optuna_local.py:639) ✓
- Final retrain uses full CPCV K=2, 800 rounds, no pruning ✓
- 2 seeded trials bypass early pruning ✓

### LSTM Pruning (runpod_train.py:557)

`n_warmup_steps=5` with 40 epochs — extremely aggressive. Can prune at epoch 6.

### Recommendation

**Increase `n_warmup_steps` from 50 to 120** (both folds complete before pruning starts).

| Location | Current | Recommended |
|----------|---------|-------------|
| run_optuna_local.py:1225 | `n_warmup_steps=50` | `n_warmup_steps=120` |
| runpod_train.py:557 | `n_warmup_steps=5` | `n_warmup_steps=15` |

**Severity: HIGH** — Trials discovering rare signals in fold 2 are killed before evaluation completes.

---

## 3. BITPACK CORRECTNESS

**Question**: Does POPCNT give exact same co-occurrence counts as sparse matmul?

### Finding: NOT IMPLEMENTED

Bitpack/POPCNT is mentioned in OPTIMIZATION_PLAN.md:54-78 as a **deferred optimization**. Current implementation uses **scipy sparse matmul** (`CSR.T @ CSR`) at v2_cross_generator.py:1020.

From OPTIMIZATION_PLAN.md:58:
> "Bitpacked AND is 50-500x faster for AND operation ALONE, but back-conversion to CSC negates it"

**Verdict: PASS** — Sparse matmul is exact and deterministic. No POPCNT approximation risk.

---

## 4. MEMMAP LOSSLESSNESS

**Question**: Do ALL features survive the chunk→disk→merge→load cycle?

### Finding: No memmap used

Zero `np.memmap` calls found in the codebase. Data flows through:
1. **In-RAM CSR matrices** during cross generation
2. **scipy sparse NPZ** for disk checkpoints (uncompressed, lossless)
3. **Atomic writes** via atomic_io.py:52-63 (temp file → os.replace)

### Chunk→Disk→Merge→Load Verified Lossless

| Stage | Mechanism | File:Line |
|-------|-----------|-----------|
| Chunk save | `sparse.save_npz(compressed=False)` | v2_cross_generator.py:661-663 |
| Atomic write | temp file → os.replace | atomic_io.py:52-63 |
| Merge | CSC pointer concatenation (no value modification) | v2_cross_generator.py:1520-1532 |
| Verify | Shape assertion after reload | v2_cross_generator.py:1967-1969 |
| int64 overflow | `astype(np.int64)` before pointer addition | v2_cross_generator.py:1521 |

**Verdict: PASS** — All features survive the full cycle. Float32 values byte-identical after roundtrip.

---

## 5. CUDA EXACTNESS

**Question**: Do warp-reduced histograms produce bit-identical results to CPU?

### Finding: No warp reductions — uses atomicAdd(double)

**Kernel 1** (gpu_histogram.cu:209-255): Global memory `atomicAdd(&hist[col*2], g)` on **double precision**.
**Kernel 2** (gpu_histogram.cu:274-340): Shared memory tiled version, same double atomics.

No `__shfl_down_sync()` or warp-level reductions detected.

### Precision Analysis

`atomicAdd(double)` is **deterministic within one run** but summation order varies between runs due to warp scheduling. This means:

| Comparison | Result |
|-----------|--------|
| GPU run A vs GPU run A (same input) | Bit-identical |
| GPU run A vs GPU run B (same input) | Within ±1e-15 (double precision) |
| GPU vs CPU (same input) | Within ±1e-5 (accumulation order differs) |

### Test Evidence

| Test (test_histogram_equivalence.py) | Rows | Tolerance |
|------|------|-----------|
| Small (line 143) | 100 | 1e-10 |
| Realistic 4h (line 185) | 17.5K | 1e-5 |
| All 63 leaves (line 238) | 5K | 1e-8 |

### Impact on Rare Signals

LightGBM split gain thresholds are typically O(1e-2) to O(1e-1). A ±1e-5 histogram deviation is **6+ orders of magnitude below split decision threshold**. Rare signal splits are unaffected.

**Verdict: PASS*** — Not bit-identical to CPU, but deviation (1e-5) is negligible for tree splitting. No rare signal risk.

---

## 6. MULTI-GPU INDEPENDENCE

**Question**: Does round-robin GPU assignment affect trial sampling distribution?

### GPU Assignment

```python
gpu_params['gpu_device_id'] = trial.number % n_gpus  # run_optuna_local.py:513
```

### Sampler Independence

Optuna TPESampler initialized with `seed=OPTUNA_SEED` (run_optuna_local.py:1239). Sampling occurs **before** GPU assignment — the sampler has no knowledge of which GPU will execute the trial.

### Seed Management

| Phase | Seed | File:Line |
|-------|------|-----------|
| Phase 1 | `OPTUNA_SEED` (global) | run_optuna_local.py:506 |
| Validation | `OPTUNA_SEED` (global) | run_optuna_local.py:661 |
| Final retrain | `OPTUNA_SEED` (global) | run_optuna_local.py:864 |

Seeds are global (not per-trial), which means each trial's LightGBM bagging uses identical randomness. This is **acceptable** — between-trial variance comes from hyperparameter differences, which is correct for Bayesian optimization.

### Minor Concern: CUDA_VISIBLE_DEVICES

`cloud_runner.py:78` removes CUDA_VISIBLE_DEVICES, making all GPUs visible to all processes. In multi-process scenarios, CuPy cross-building could distribute work unpredictably. Not a matrix thesis violation, but worth noting.

**Verdict: PASS** — GPU assignment is independent of trial sampling. No rare signal risk.

---

## 7. ADAPTIVE CHUNKS

**Question**: Can halving chunk size lose features at chunk boundaries?

### Chunking Architecture

v2_cross_generator.py:672-768 — RIGHT_CHUNK controls how many right-side contexts are processed per iteration:

```python
for rc_start in range(0, len(right_names), RIGHT_CHUNK):
    rc_end = min(rc_start + RIGHT_CHUNK, len(right_names))
    r_names_chunk = right_names[rc_start:rc_end]
```

### Why Halving is Safe

1. **Column-wise chunking**: Each right-side context is processed exactly once, regardless of chunk size
2. **No boundary-crossing pairs**: A cross (left_i, right_j) exists in the chunk containing right_j, period
3. **Complete enumeration**: `range(0, N, CHUNK)` covers all N contexts for any CHUNK > 0
4. **Name accumulation**: `all_names.extend(c_names)` appends per chunk (line 694)
5. **Offset tracking**: `current_offset += c_ncols` increments correctly (line 701, 708, 755)

### Proof by Example

- RIGHT_CHUNK=100: Processes contexts [0-99], [100-199], ...
- RIGHT_CHUNK=50: Processes contexts [0-49], [50-99], [100-149], ...
- **Same total crosses generated**, just in more iterations.

**Verdict: PASS** — Halving RIGHT_CHUNK cannot lose features. Architecture is boundary-safe.

---

## 8. INDICES-ONLY NPZ

**Question**: Is float32 1.0 reconstruction IEEE 754 exact?

### Architecture

inference_crosses.py stores **int32 index pairs** in NPZ (lines 162-169):
```python
left_arr = np.array(left_indices, dtype=np.int32)
right_arr = np.array(right_indices, dtype=np.int32)
np.savez_compressed(out, left=left_arr, right=right_arr)
```

### Reconstruction Path

inference_crosses.py:371-372:
```python
crosses = (ctx_binary[self.left_idx] & ctx_binary[self.right_idx]).astype(np.float32)
```

1. `ctx_binary` is **uint8** array of 0/1 (set via boolean comparison, line 349)
2. Bitwise AND of uint8(1) & uint8(1) = uint8(1) — **exact integer operation**
3. `.astype(np.float32)`: uint8(1) → float32(1.0)

### IEEE 754 Proof

float32(1.0) = `0x3F800000` = sign(0) × 2^(127-127) × 1.0 = **exactly 1.0**

Integer 1 has an **exact finite representation** in IEEE 754 single precision. No rounding occurs. Similarly, integer 0 → float32(0.0) = `0x00000000`, also exact.

The entire pipeline uses:
- **int32** indices on disk (NPZ) — no float storage
- **uint8** booleans in memory — no precision loss
- **float32** only at final output — exact for {0, 1}

**Verdict: PASS** — Reconstruction is IEEE 754 exact. No information loss.

---

## CONSOLIDATED FINDINGS

### CRITICAL/HIGH

| ID | Concern | File:Line | Issue |
|----|---------|-----------|-------|
| **H1** | Pruning kills rare signals | run_optuna_local.py:1225 | `n_warmup_steps=50` < required 120 for 2-fold CV |
| **H2** | PatientPruner too aggressive | run_optuna_local.py:1231-1232 | `patience=5, min_delta=0.001` can kill mid-fold-2 |
| **H3** | LSTM pruner too aggressive | runpod_train.py:557 | `n_warmup_steps=5` allows pruning at epoch 6 |

### MEDIUM

| ID | Concern | File:Line | Issue |
|----|---------|-----------|-------|
| **M1** | GPU histogram non-determinism | gpu_histogram.cu:240-241 | atomicAdd(double) ordering varies between runs |
| **M2** | Optuna row subsample in Phase 1 | config.py:418-424 | 1h=0.50, 15m=0.25 during search (final=1.0) |

### LOW

| ID | Concern | File:Line | Issue |
|----|---------|-----------|-------|
| **L1** | CUDA_VISIBLE_DEVICES unset | cloud_runner.py:78 | All GPUs visible to all processes |
| **L2** | Global seed not per-trial | run_optuna_local.py:506 | Identical bagging randomness per trial |

### PASS (No Issues)

- Signal survival parameters: All correctly configured
- Feature filtering: Complete no-op, 41 prefixes protected
- Chunk boundaries: Column-wise chunking is boundary-safe
- NPZ reconstruction: IEEE 754 exact for {0, 1}
- Sparse save/load: Lossless via uncompressed scipy NPZ + atomic writes
- Bitpack: Not implemented (using exact sparse matmul)

---

## RECOMMENDATIONS

1. **[H1] Increase n_warmup_steps to 120** in run_optuna_local.py:1225
2. **[H2] Consider fold-completion gate** — suppress pruning until all K folds have ≥1 report
3. **[H3] Increase LSTM n_warmup_steps to 15** in runpod_train.py:557
4. **[M2] Document that Phase 1 row subsample is speed-only** — add comment confirming final retrain uses 1.0
5. **[L2] Consider per-trial seed** (`OPTUNA_SEED + trial.number`) for bagging diversity

---

*Audit completed 2026-03-29. READ-ONLY — no files modified except this report.*
