# DOUBLE AUDIT #1: Matrix Thesis — Every Branch, Every File

**Auditor:** security-reviewer agent
**Date:** 2026-03-30
**Base branch:** v3.3

---

## Summary

| # | Check | Result |
|---|-------|--------|
| 1 | feature_pre_filter = False | **PASS** (all branches) |
| 2 | feature_fraction >= 0.7 | **PASS** (all branches) |
| 3 | feature_fraction_bynode >= 0.7 | **FAIL** — `ceo/backend-dev-e443f758` has floor 0.5 |
| 4 | bagging_fraction >= 0.7 | **FAIL** — `ceo/backend-dev-e443f758` has floor 0.5 |
| 5 | min_data_in_leaf <= 10-15 per TF | **PASS** (all branches) |
| 6 | No fillna(0) / NaN→0 on features | **PASS** (all branches) |
| 7 | No drop/filter on features | **PASS** (all branches) |
| 8 | Row subsample >= 0.95 or 1.0 | **PASS** (all branches) |
| 9 | EFB pre-bundler: zero dropped | **PASS** |
| 10 | Feature pruning: inference only | **PASS** |

**Overall: 2 FAILURES found in `ceo/backend-dev-e443f758`**

---

## Branch-by-Branch Detail

### ceo/backend-dev-106b1d97 — Multi-GPU Optuna
- **feature_pre_filter:** PASS — `False` documented in CHANGES_TRAINING_MULTIGPU.md
- **feature_fraction:** PASS — `>= 0.7` documented
- **feature_fraction_bynode:** PASS — not modified (inherits base)
- **bagging_fraction:** PASS — not modified
- **min_data_in_leaf:** PASS — not modified
- **fillna/NaN→0:** PASS — none
- **drop/filter:** PASS — none
- **row_subsample:** PASS — default `1.0` in function signature
- **EFB:** N/A
- **Feature pruning:** PASS — Optuna MedianPruner (trial pruning, not feature pruning)

**RESULT: PASS**

---

### ceo/backend-dev-22d6ed1e — Numba/CUDA Cross Kernels
- All 10 checks: **PASS** — pure compute kernels, no parameter changes

**RESULT: PASS**

---

### ceo/backend-dev-605acb8a — EFB Pre-Bundler ⭐
- **EFB zero-drop:** PASS — docstring: "ZERO features dropped — every feature lands in a bundle"
- Code verified: no `drop`, `discard`, `remove`, `filter`, or `skip` on features
- All other checks: N/A or PASS (no LightGBM params modified)

**RESULT: PASS**

---

### ceo/backend-dev-8dedf3ff — Optuna Params ⭐
- **feature_pre_filter:** PASS — `False` in config.py, validated in validate.py
- **feature_fraction:** PASS — `suggest_float('feature_fraction', 0.7, 1.0)`
- **feature_fraction_bynode:** PASS — **RAISED from 0.5 → 0.7** — `suggest_float('feature_fraction_bynode', 0.7, 1.0)`
- **bagging_fraction:** PASS — **RAISED from 0.5 → 0.7** — `suggest_float('bagging_fraction', 0.7, 1.0)`
- **min_data_in_leaf:** PASS — range `max(3, _tf_mdil)` to `15`
- **validate.py:** PASS — all floor checks updated to 0.7 with correct error messages
- **fillna/NaN→0:** PASS — none
- **drop/filter:** PASS — none

**RESULT: PASS** — This is the fix branch for the 0.5 → 0.7 floor issue.

---

### ceo/backend-dev-b161bc3a — Cross-Gen Parallel
- All 10 checks: **PASS** — pure parallelism refactor, no parameter changes

**RESULT: PASS**

---

### ceo/backend-dev-cf941cf8 — CPCV, Sobol, lleaves, Config+System ⭐
- **feature_pre_filter:** PASS — explicitly `False` at two locations:
  - `run_optuna_local.py`: `train_params['feature_pre_filter'] = False  # NEVER filter rare features`
  - `ml_multi_tf.py`: `params={'feature_pre_filter': False, ...}`
- **Feature pruning:** PASS — `inference_pruner.py` is INFERENCE ONLY:
  - Docstring: "Feature Pruning for Deployment Models"
  - Uses split_count > 0 from already-trained model
  - live_trader.py loads pruned model only for inference
  - Training always uses full feature set
- All other checks: PASS

**RESULT: PASS**

---

### ceo/backend-dev-da8e680c — Atomic IO
- All 10 checks: **PASS** — pure IO improvements, no parameter changes

**RESULT: PASS**

---

### ceo/backend-dev-d1bd48cc — Multi-GPU (duplicate of 106b1d97)
- Same multi-GPU feature, same checks
- **feature_pre_filter:** PASS — `False` documented
- **row_subsample:** PASS — default `1.0`

**RESULT: PASS**

---

### ceo/backend-dev-4181eede — (2 commits, no matrix params)
- All checks: **PASS** — clean, no matrix-relevant changes

**RESULT: PASS**

---

### ceo/backend-dev-702e95d2 — Co-occurrence threshold
- `MIN_CO_OCCURRENCE = 3` matches `min_data_in_leaf=3` — PASS
- No other matrix params touched

**RESULT: PASS**

---

### ceo/backend-dev-a8040695 — (1 commit, no matrix params)
- All checks: **PASS** — clean

**RESULT: PASS**

---

### ceo/backend-dev-e443f758 — WilcoxonPruner (3 commits) ⚠️ FAILURES

- **feature_fraction:** PASS — `suggest_float('feature_fraction', 0.7, 1.0)` at line 503
- **feature_fraction_bynode:** **FAIL** — `suggest_float('feature_fraction_bynode', 0.5, 1.0)` at `run_optuna_local.py:504`
  - Floor is 0.5, must be >= 0.7
  - Effective rate: 0.5 × 0.7 = 0.35 — only 35% of features visible per node
  - **Devastating for rare esoteric signals that fire 3-15 times**
- **bagging_fraction:** **FAIL** — `suggest_float('bagging_fraction', 0.5, 1.0)` at `run_optuna_local.py:505`
  - Floor is 0.5, must be >= 0.7
  - 50% row dropout on a 3-fire feature: P(included) = 0.5³ = 12.5% — feature invisible 87.5% of rounds
- **min_data_in_leaf:** PASS — range 3-15
- All other checks: PASS

**RESULT: FAIL**

**Root cause:** This branch was forked BEFORE `ceo/backend-dev-8dedf3ff` applied the 0.5 → 0.7 floor fix. The WilcoxonPruner changes are fine, but the param floors are stale.

**Fix required:** Merge `ceo/backend-dev-8dedf3ff` INTO `ceo/backend-dev-e443f758`, OR ensure `8dedf3ff` is merged AFTER `e443f758` during final integration (so 0.7 floors override 0.5).

---

## Merge Order Dependency

**CRITICAL**: If `ceo/backend-dev-e443f758` is merged AFTER `ceo/backend-dev-8dedf3ff`, it will **revert** the 0.7 floors back to 0.5 on `run_optuna_local.py` lines 504-505. The safe merge order is:

1. Merge `e443f758` first (WilcoxonPruner, with stale 0.5 floors)
2. Merge `8dedf3ff` second (overrides floors to 0.7)

Or: rebase `e443f758` onto `8dedf3ff` before merging.

---

## Branches with No Commits (verified clean)
- ceo/backend-dev-3fcac80c — 0 commits, no changes
- ceo/backend-dev-afb08116 — 0 commits, no changes
- ceo/backend-dev-b3bc55dc — 0 commits, no changes
- ceo/backend-dev-b94865a8 — 0 commits, no changes
- ceo/backend-dev-dcd1cfc5 — 0 commits, no changes
- ceo/backend-dev-e92b0f1b — 0 commits, no changes
- ceo/backend-dev-f0229fbb — 0 commits, no changes
