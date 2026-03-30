# DOUBLE AUDIT #2: Correctness — Game-Changers #1-5

**Date:** 2026-03-30
**Auditor:** Error Checker Agent (READ-ONLY)
**Scope:** 5 major game-changer implementations audited for correctness bugs

---

## EXECUTIVE SUMMARY

| Game-Changer | Verdict | Critical Bugs | High Bugs |
|---|---|---|---|
| #1 EFB Pre-Bundler | **NOT IMPLEMENTED** | 1 | 0 |
| #2 CPCV Fold Reduction | **FAIL** | 1 | 0 |
| #3 Sobol Trade Optimizer | **FAIL** | 2 | 2 |
| #4 lleaves Inference | **PASS** | 0 | 1 |
| #5 Optuna Params | **FAIL** | 2 | 0 |

**Total: 6 CRITICAL, 3 HIGH bugs across 5 game-changers.**

---

## #1 EFB Pre-Bundler

### Status: NOT IMPLEMENTED

`efb_prebundler.py` **does not exist**. Only a design document exists at `EXPERT_LIGHTGBM_EFB.md`.

| # | Finding | Severity | Reference |
|---|---------|----------|-----------|
| 1.1 | `efb_prebundler.py` file missing — no implementation exists | **CRITICAL** | N/A |
| 1.2 | Collision detection algorithm (row set intersection) — unimplemented | INFO | EXPERT_LIGHTGBM_EFB.md:96 |
| 1.3 | Offset encoding bounds checking (value ∈ [0,254]) — unimplemented | INFO | EXPERT_LIGHTGBM_EFB.md:56 |
| 1.4 | 127 features per bundle enforcement — unimplemented | INFO | EXPERT_LIGHTGBM_EFB.md:57 |
| 1.5 | Feature→bundle mapping reversibility — unimplemented | INFO | EXPERT_LIGHTGBM_EFB.md:213 |
| 1.6 | Non-binary feature handling — unimplemented | INFO | EXPERT_LIGHTGBM_EFB.md:54 |
| 1.7 | EFB config contradiction: CLAUDE.md says "always True" but EXPERT doc says "disable for >1M features" | HIGH | EXPERT_LIGHTGBM_EFB.md:18 |

**Conclusion:** Cannot audit what doesn't exist. The design doc is comprehensive but no code was written.

---

## #2 CPCV Fold Reduction

### Status: FAIL — Temporal Leakage Bug

#### CRITICAL: Temporal Leakage with Non-Contiguous Test Groups
**Files:** `ml_multi_tf.py:244-255`, `run_optuna_local.py:196-207`

The purge logic uses distance-based filtering from group boundaries. When test groups are non-contiguous (e.g., groups 0 and 2), rows between the test groups are NOT purged from training.

**Example (n_groups=10, group_size=2500, max_hold=48):**
- test_groups=(0, 2) → test=[0-2499] ∪ [5000-7499]
- group[1]=[2500-4999] sits between test groups
- Purge only removes rows within 48 of boundaries
- **LEAKAGE:** Rows 2501-4998 (~2250 rows) remain in training despite being temporally inside the test range

**Affected configurations:**

| Config | n_groups | Safe Paths | Leaking Paths |
|--------|----------|------------|---------------|
| OPTUNA_PHASE1 | 2 | 1/1 | 0 |
| OPTUNA_VALIDATION | 4 | 1/6 | **5/6** |
| TF_CPCV 1w/1d | 5 | ~1/10 | **~9/10** |
| TF_CPCV 4h/1h/15m | 6 | ~1/15 | **~14/15** |

**Impact:** Most production models trained after Phase 1 contain temporal leakage. Validation metrics are biased optimistically.

#### MEDIUM: Embargo Zone Gaps
**Files:** `ml_multi_tf.py:257-264`, `run_optuna_local.py:209-216`

Embargo is applied independently after each test group. The gap between non-contiguous groups is unprotected.

#### Passing Checks
- Path sampling: **PASS** — deterministic via `itertools.combinations` (no RNG)
- Row coverage: **PASS** — mathematical guarantee, every row in ≥1 test fold

---

## #3 Sobol Trade Optimizer

### Status: FAIL — Sobol Not Implemented + Sortino Bug

#### CRITICAL #1: Sobol Not Implemented
**File:** `exhaustive_optimizer.py:4, 877`

The file uses **Optuna TPE sampler**, not Sobol sequences. No `scipy.stats.qmc.Sobol` import exists anywhere.
- Line 4: `"exhaustive_optimizer.py — Optuna TPE Optimizer (LightGBM)"`
- Line 877: `sampler=optuna.samplers.TPESampler(...)`
- EXPERT_TRADE_OPTIMIZER.md Line 64: `"Verdict: Use Scrambled Sobol"`

The 7 parameter dimensions are correctly mapped (line 379: `[lev, risk_pct, stop_atr, rr, max_hold, exit_type, conf_thresh]`).

#### CRITICAL #2: Sortino Denominator Wrong
**File:** `exhaustive_optimizer.py:684-691`

```python
# Line 687 — WRONG: divides by total_trades
downside_var = sum_neg_sq / total_trades   # ← Should be count_neg

# Correct:
downside_var = sum_neg_sq / count_neg
```

**Impact:** If 20% of trades are losses, Sortino is **inflated by 5x**. The optimizer selects parameter sets that appear strong but are actually suboptimal.

**Example:**
- 100 trades, 20 losses, sum_neg_sq = 0.08
- Current (wrong): downside_std = sqrt(0.08/100) = 0.028 → Sortino = 1.77
- Correct: downside_std = sqrt(0.08/20) = 0.063 → Sortino = 0.79

#### HIGH #1: Batch Size Not Power-of-2
**File:** `exhaustive_optimizer.py:80`

`GPU_BATCH_BASE = 500_000` — not a power of 2. EXPERT_TRADE_OPTIMIZER.md requires power-of-2 for GPU optimization. Nearest: 524,288 (2^19).

#### HIGH #2: No Phase 2 Refinement
**File:** `exhaustive_optimizer.py:741-939`

Single call to `study.optimize()` with no refinement phase. EXPERT_TRADE_OPTIMIZER.md specifies a two-phase approach: broad search → narrowed Sobol around top-K clusters.

#### MEDIUM: Startup Trials Too Low
**File:** `exhaustive_optimizer.py:878`

`n_startup_trials=8` for a 7D space with 43M-99M combinations. Only 4% exploration before TPE exploitation begins.

---

## #4 lleaves Inference

### Status: PASS (with one note)

#### PASS: Feature Extraction Uses Split Count (Not Gain)
**File:** `cloud_run_tf.py:660-665`

```python
split_scores = dict(zip(all_features, model.feature_importance(importance_type='split')))
active_features = [f for f, v in split_importance.items() if v > 0]
```

Correctly implements Strategy A from EXPERT_INFERENCE.md §2.

#### PASS: Training Uses ALL Features
**File:** `ml_multi_tf.py:6, 491, 1929`

- Philosophy: "ALL features used -- no SHAP pruning"
- `feature_pre_filter=False` enforced at all training points
- Saves as `features_{tf}_all.json`

#### HIGH (minor): Training Logs Gain-Based Importance
**File:** `ml_multi_tf.py:1896`

```python
importance = dict(zip(final_model.feature_name(), model.feature_importance(importance_type='gain')))
```

Logs `gain` importance for visibility. Not a bug (no pruning occurs), but inconsistent with the `split`-based extraction used in inference. Could mislead future optimization.

#### PASS: live_trader.py Fallback Chain
**File:** `live_trader.py:707-764, 1112-1119`

1. Prefers `features_{tf}_all.json` → falls back to `features_{tf}_pruned.json`
2. Model feature order is authoritative (line 764)
3. Cross features are MANDATORY — halts prediction if missing (line 1119)

Correctly implements "NO FALLBACKS" philosophy.

---

## #5 Optuna Params

### Status: FAIL — Floors Too Low

#### CRITICAL #1: feature_fraction_bynode Floor = 0.5 (Should Be 0.7)
**File:** `run_optuna_local.py:483`

```python
feature_fraction_bynode = trial.suggest_float('feature_fraction_bynode', 0.5, 1.0)  # ← 0.5 not 0.7
```

Validated by `validate.py:247-252` which also only checks `>= 0.5`.

**Impact:** Optuna can sample values as low as 0.5, killing rare esoteric cross signals at the per-node level.

#### CRITICAL #2: bagging_fraction Floor = 0.5 (Should Be 0.7)
**File:** `run_optuna_local.py:484`

```python
bagging_fraction = trial.suggest_float('bagging_fraction', 0.5, 1.0)  # ← 0.5 not 0.7
```

**Impact:** At bagging_fraction=0.5, probability of a rare 10-fire signal surviving in a single bag = 0.5^10 = **0.1%**. Config.py:336 comment explicitly justifies 0.95 for rare signal protection: "P(10-fire in bag) = 0.95^10 = 59.9%".

#### MEDIUM: bin_construct_sample_cnt Not Set
**File:** Not found in any v3.3 production code

`bin_construct_sample_cnt=5000` is not configured anywhere. Falls back to LightGBM default of 200,000 (40x higher). Only appears in GPU fork test fixtures.

#### Baseline Config (PASS)
**File:** `config.py:320-341`

- `feature_fraction: 0.9` ≥ 0.7 ✓
- `feature_fraction_bynode: 0.8` ≥ 0.7 ✓
- `bagging_fraction: 0.95` ≥ 0.7 ✓

Baseline is correct — only the Optuna search ranges are too wide.

---

## PRIORITY ACTION LIST

### BLOCK DEPLOYMENT (Fix Before Any Training)

1. **Sortino denominator** (`exhaustive_optimizer.py:687`): Change `total_trades` → `count_neg`
2. **CPCV temporal leakage** (`ml_multi_tf.py:244-255`, `run_optuna_local.py:196-207`): Purge all rows between min(test) and max(test), not just boundary-adjacent
3. **feature_fraction_bynode floor** (`run_optuna_local.py:483`): Change 0.5 → 0.7
4. **bagging_fraction floor** (`run_optuna_local.py:484`): Change 0.5 → 0.7

### HIGH Priority

5. **Implement EFB Pre-Bundler** or remove from game-changer list
6. **Add bin_construct_sample_cnt=5000** to V3_LGBM_PARAMS if intended
7. **Implement Sobol sequences** or update documentation to reflect TPE
8. **Add Phase 2 refinement** to optimizer

### MEDIUM Priority

9. Update `validate.py` floors to match 0.7 requirements
10. Add temporal overlap validation to validate.py
11. Fix GPU batch size to power-of-2 (524,288)
12. Add split-importance logging alongside gain in training

---

*Audit complete. 6 CRITICAL + 3 HIGH bugs identified. Items 1-4 must be fixed before any training run.*
