# V3.3 Session Resume — 2026-03-30 (Post-Training Run)

## INSTRUCTION TO NEW SESSION
Read this file completely. Then read v3.3/CLAUDE.md. The 1w training run completed on cloud — artifacts are still on the machine. Evaluate results before destroying.

## ACTIVE MACHINE
- **Instance 33840373** — Vietnam, 1x RTX 3060 Ti, EPYC 7B12 64c, 516GB RAM, $0.402/hr
- **SSH**: `ssh -p 10372 root@ssh9.vast.ai`
- **Status**: 1w pipeline completed (8.4 min total). Artifacts on disk at /workspace/
- **DOWNLOAD ARTIFACTS BEFORE DESTROYING** — model_1w.json, cpcv_oos_predictions_1w.pkl, sobol_configs_1w.json, lstm_1w.pt, 1w_training.log, ml_multi_tf_configs.json, validation_report_1w.json

## 1W TRAINING RESULTS (2026-03-30)

### Model Performance
- **CPCV OOS Average**: Acc=41.0%, PrecL=37.6%, PrecS=37.0%, mlogloss=1.10
- **Final model accuracy**: 59.2% (retrained on all data)
- **Active features**: 86 / 3552 (2.4% — NEEDS INVESTIGATION)
- **LSTM**: test_acc=55.9%, 150 epochs, massive overfitting (train=100%, test=55.9%)
- **PBO**: 0.200 (INVESTIGATE — borderline)
- **Optimizer best**: Sortino +8.36, ROI +29.4%, DD 0.7%, 16 trades

### CPCV Fold Variance (CONCERNING)
| Path | Acc | PrecL | PrecS | Trees |
|------|-----|-------|-------|-------|
| 1 | 68.7% | 70.9% | 48.4% | 25 |
| 2 | 34.5% | 66.7% | 28.7% | 22 |
| 3 | 26.5% | 0.0% | 26.5% | 3 |
| 4 | 26.6% | 0.0% | 26.6% | 10 |
| 5 | 52.8% | 54.0% | 46.2% | 21 |
| 6 | 35.6% | 0.0% | 35.6% | 1 |
| 7 | 35.6% | 0.0% | 35.6% | 1 |
| 8 | 46.9% | 89.3% | 43.0% | 22 |
| 9 | 40.1% | 0.0% | 40.1% | 27 |
| 10 | 42.6% | 94.7% | 39.4% | 118 |

5 of 10 paths have PrecL=0.0% (predicts only SHORT). Huge variance. Only 819 rows — expected to be noisy but this level of instability is concerning.

### Pipeline Timing (on 64c EPYC + RTX 3060 Ti)
| Step | Time | Notes |
|------|------|-------|
| Validation | 8s | 106/106 passed |
| Feature Build | ~20s | 3552 features (DENSE, no cross gen) |
| Optuna Search | 7s | Skipped (used config defaults) |
| CPCV Training | ~60s | 10 paths, sequential (dense forced) |
| Final Retrain | ~20s | 59.2% accuracy |
| Feature Importance | <1s | 86/3552 active features |
| Optimizer | 247s | 131K Sobol + 200 Bayesian, GPU |
| PBO | 1s | Score=0.200 (INVESTIGATE) |
| Meta-labeling | 7s | OK |
| LSTM | 104s | 150 epochs, GPU said cuda but 0% util |
| Audit | 2s | FAILED (non-critical) |
| **TOTAL** | **503s (8.4 min)** | |

### Bottlenecks Found
1. **LSTM runs on CPU despite claiming CUDA** — 0% GPU utilization, 115% single-core CPU
2. **Sequential CPCV forced for dense data** — parallel workers crash on numpy arrays (no .nnz)
3. **No Optuna search ran** — used config.py defaults (no optuna_configs_1w.json)
4. **Only 2.4% of features active** — 86/3552 used by model. NEEDS INVESTIGATION.

## BUGS FIXED THIS SESSION (9 total)

### Pre-existing bugs found by 10-agent optimization company:
1. `v2_cross_generator.py:1045,1326` — NameError: co_occur not in scope (CRITICAL)
2. `leakage_check.py:194` — bagging_fraction=0.6 killing rare signals
3. `v2_multi_asset_trainer.py:461,628` — missing feature_pre_filter=False in Dataset()
4. `ml_multi_tf.py:2092` — missing min_data_in_bin=1 in parent Dataset fallback
5. `feature_library.py:2332,2465,3249,6352` — fillna(0) on 4 feature columns
6. `run_optuna_local.py:552` — min_data_in_leaf upper bound 15→10
7. `astrology_engine.py` — 11 silent except blocks returning neutral instead of NaN

### Deployment bugs found during cloud run:
8. `ml_multi_tf.py` — parallel CPCV crashes on dense data (no .nnz on numpy array). Fix: force sequential for dense.
9. `cloud_run_tf.py` — numactl --interleave=all fails in containers without SYS_NICE. Fix: test before use.
10. `ml_multi_tf.py` — worker count not capped by row count (10 workers × 24 threads on 443 rows). Fix: row-aware cap.
11. `ml_multi_tf.py` — training failure swallowed, exits 0. Fix: sys.exit(1) on exception.
12. `validate.py` — 1w row threshold 1000 but only 819 weekly candles exist. Fix: lowered to 500.
13. 8 new validate.py checks added (bagging_fraction, lambda_l1/l2, bagging_freq, path_smooth, CPCV_PARALLEL_GPUS, file scans)

## NEW CODE ADDED THIS SESSION
- **Fold-parallel GPU CPCV**: _detect_gpu_count(), _gpu_fold_worker(), run_cpcv_gpu_parallel() in ml_multi_tf.py
- **CPCV_PARALLEL_GPUS** config in config.py (env var override, 0=auto)
- **_detect_n_gpus()** in run_optuna_local.py
- **9 expert reports** in v3.3/OPTIMIZATION_REPORTS/
- **eta_calculator recommendations** in OPT_PER_TF_SPECIALIST.md and OPT_VASTAI_COST_PERF.md

## WHAT NEEDS TO BE DONE NEXT SESSION

### Priority 1: Evaluate 1w model quality
- Download artifacts from cloud machine (still running at $0.40/hr!)
- Launch evaluation company: is 41% CPCV accuracy tradable? Is 86/3552 feature usage acceptable?
- Compare with old v3.2 results (57.9% was INVALID due to signal-killing params)
- Check if esoteric features are among the 86 active ones
- Institutional standards assessment

### Priority 2: Fix LSTM GPU utilization
- LSTM claims CUDA but runs on CPU (0% GPU, 115% single-core)
- Launch company to investigate pytorch CUDA path

### Priority 3: Run Optuna search
- 1w ran with config.py defaults (no Optuna search)
- Need to run `run_optuna_local.py --tf 1w` for proper HPO

### Priority 4: Investigate 2.4% feature utilization
- Only 86/3552 features used by model
- Is EFB bundling too aggressive? Are esoteric features in the active set?
- Are cross features being generated for 1w? (currently DENSE, no cross gen)

### Priority 5: Push fixes to git
- 12 bug fixes not yet committed/pushed to TB-3.3
- Need to commit all changes from this session

## GIT STATE
- **Branch**: v3.3-clean (local), pushed to github.com/c19o/TB-3.3 (main)
- **Uncommitted changes**: ml_multi_tf.py, cloud_run_tf.py, validate.py, feature_library.py, astrology_engine.py, run_optuna_local.py, leakage_check.py, v2_cross_generator.py, v2_multi_asset_trainer.py
- **Remote**: TB-3.3 repo on GitHub (source code only, no artifacts)

## BEHAVIORAL RULES (REINFORCED THIS SESSION)
1. **NEVER solo debug** — every bug requires agent company with Perplexity + matrix thesis
2. **No git worktrees for CEO agents** — each copies 10GB+, kills disk
3. **CEO agents need non-bare mode** — bare mode strips auth, agents fail with "Not logged in"
4. **All files live in v3.3/** — not project root
5. **ALLOW_CPU=1** needed for feature builds without RAPIDS/cuDF
