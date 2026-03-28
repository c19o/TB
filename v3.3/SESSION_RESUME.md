# V3.3 Session Resume — 2026-03-28

## INSTRUCTION TO NEW SESSION: Read this file completely. Then read v3.3/OPTIMIZATION_PLAN.md, v3.3/CLAUDE.md. These files are the complete context. Ask the user what to do next.

---

## STATUS: GPU FORK PHASE 3 COMPLETE → CPU TRAINING VERIFIED → BUILD REMAINING TFs

GPU histogram fork hit architectural wall (EFB mismatch). CPU training already works well (73.9% holdout). Decision pending on GPU path. Meanwhile, proceed with building + training remaining timeframes locally.

### TWO SEPARATE OPTIMIZERS IN OUR PIPELINE
1. **LightGBM HPO** (`run_optuna_local.py`) — optimizes 10 LightGBM hyperparams (num_leaves, feature_fraction, learning_rate, lambdas, etc.). Objective: minimize mlogloss via CPCV. THIS is the 90% bottleneck (200 trials × 4-15 folds × 150 min/fold).
2. **Trade Strategy Optimizer** (`exhaustive_optimizer.py`) — optimizes 7 trading params (leverage, risk%, stop-loss, R:R, hold, exit, confidence). Objective: maximize Sortino. Runs on pre-computed predictions, GPU-vectorized on 3090. Fast (minutes).

---

## CURRENT STATE (2026-03-28)

### GPU Histogram Fork
- **Phase 1 (standalone benchmark)**: COMPLETE — 71x speedup on real 1w data (RTX 3090)
- **Phase 2 (LightGBM integration)**: COMPLETE — Fork builds (42/42), `device_type="cuda_sparse"` accepted, GPU detected
- **Phase 3 (CSR bridge)**: COMPLETE — CSR bridge fix applied, dangling pointer fixed, DLL rebuilt
- **Architectural issue discovered**: EFB histogram mismatch — GPU produces per-feature sums, CPU expects per-EFB-bundle bins. This is a fundamental mismatch between the sparse histogram kernel output format and LightGBM's internal EFB-bundled histogram format.
- **Decision pending**: Fix EFB mapping in GPU kernel (non-trivial) vs accept CPU training (which already works well at 73.9%)
- **Location**: `v3.3/gpu_histogram_fork/` (isolated from main pipeline)
- **Commit**: `0a94b4e` (latest)

### Training Status
| TF | Status | Accuracy | Features | Notes |
|----|--------|----------|----------|-------|
| **1w** | DONE | CPCV 71.9% mean (73.3%, 66.7%, 73.0%, 74.6%), holdout **73.9%** | 2.2M | CPU training, real labels |
| **1d** | NEEDS BUILD | — | — | Feature build + cross gen + training locally |
| **4h** | NEEDS BUILD | — | — | Feature build + cross gen + training locally |
| **1h** | NEEDS CLOUD | — | — | After local TFs verified |
| **15m** | NEEDS CLOUD | — | — | Separate cloud machine, after local verified |

### Data Status
| Asset | Status | Notes |
|-------|--------|-------|
| **1w NPZ** | COMPLETE | 2.2M features, parquet, cross names |
| **1w model** | COMPLETE | model_1w.json (113MB), CPCV backup |
| **1d cross gen** | OLD/STALE | v3.0 NPZ exists (6M features) but uses min_nonzero=8, needs rebuild with min_nonzero=3 |
| **4h/1h/15m** | NOT BUILT | Need feature build + cross gen |
| **16 DBs** | LOCATED | Being copied to v3.3 |

### Config & Code Changes (ALL APPLIED)
- max_bin=255, min_data_in_bin=1, row subsample=1.0
- HMM weights removed (MC-1), early stopping scales with LR (MC-4)
- SIGTERM checkpoint (MC-5), co-occurrence=3 (MC-6)
- enable_bundle=False for 1h/15m
- All TFs CPCV (4,1) = 4 folds

### All Machines Destroyed
No active cloud machines.

---

## NEXT STEPS (ordered)

1. **Verify DBs aren't stale** — check all 16 .db files have current data
2. **Copy DBs to v3.3** — ensure all databases are in v3.3 directory
3. **Build 1d features locally** — feature_library.py + cross gen with min_nonzero=3
4. **Train 1d with CPCV** — 4 folds, fixed params (no Optuna initially)
5. **Build 4h features locally** — feature build + cross gen
6. **Train 4h with CPCV** — 4 folds, fixed params
7. **Verify all local models** — ensure 1w/1d/4h good for trading
8. **Rent cloud: 1h** — one machine, build + train
9. **Rent cloud: 15m** — separate machine (high RAM), build + train
10. **Push to git** — all models + artifacts

### GPU Fork Decision (deferred)
- Option A: Fix EFB mapping in GPU kernel (map per-feature histograms to per-bundle bins)
- Option B: Accept CPU training — 73.9% accuracy is strong, CPU path fully working
- Can revisit after all 5 TFs trained and live trading validated

---

## TRAINING PLAN & COSTS

### Per-TF Training Times (with optimizations, per pipeline step)

| Step | 1w | 1d | 4h | 1h | 15m |
|------|-----|-----|-----|------|------|
| Feature build | 5m | 15m | 30m | 1hr | 2hr |
| Cross gen | 15s | 10m | 36m | 2.3hr | 5.8hr |
| save_binary | 30s | 5m | 15m | 30m | 45m |
| CPCV per fold | 3m | 30m | 50m | 104m | 150m |
| CPCV total (4 folds all TFs) | 12m | 2hr | 3.3hr | 7hr | 10hr |
| **Total no Optuna** | **25m** | **2.5hr** | **5-10hr** | **9-15hr** | **13-19hr** |

### Machine Assignments

| TF | Where | Machine | Notes |
|----|-------|---------|-------|
| 1w | LOCAL | 13900K + 3090 | DONE |
| 1d | LOCAL | 13900K + 3090 | Next |
| 4h | LOCAL | 13900K + 3090 | After 1d |
| 1h | CLOUD | 128c+ / 768GB+ | After local verified |
| 15m | CLOUD | 128c+ / 1TB+ | Separate machine |

---

## KEY RESEARCH FINDINGS

1. **v3.2 NEVER completed training** — crashed on feature_name mismatch. No baseline exists.
2. **max_bin=255** — controls EFB bundle size. max_bin=15 created 18x more bundles than needed.
3. **GPU histogram fork proved 71x speedup** but hit EFB mismatch wall. CPU training at 73.9% is strong enough.
4. **GOSS/CEGB/quantized_grad** — all kill rare esoteric signals. REJECTED.
5. **HMM regime weights 0.15 is a thesis violation** — crushes counter-trend esoteric signals by 85%.
6. **Co-occurrence filter 8→3** — matches min_data_in_leaf=3 for 1d/1w.
7. **enable_bundle=False for 1h/15m** — EFB conflict graph intractable at 10M features.
8. **Cross features are pure 0/1** — structural zeros = feature OFF, not missing. No NaN after binarization.
9. **We are in UNCHARTED TERRITORY** — no published work combines esoteric × TA at 2-10M sparse binary features.

---

## REJECTED GPU ALTERNATIVES (researched, all blocked)
| Library | Blocker | Details |
|---------|---------|---------|
| LightGBM CUDA | Sparse CSR not supported | Dense = 2.94TB, impossible |
| ThunderGBM | Dead (2019), int32 overflow (5.88B NNZ > 2.1B max) | No EFB, no Python API, no callbacks |
| ScalaGBM (KDD 2025) | int32 overflow, no EFB, no Python API | 5 GitHub stars, research prototype |
| XGBoost GPU | No EFB, 12% accuracy drop | Already tested and rejected in v3.2 |
| Custom GPU fork | EFB histogram mismatch | GPU per-feature sums vs CPU per-bundle bins |

---

## ARTIFACTS IN v3.3/ DIRECTORY

### Models & Training Results
- model_1w.json (113MB, 2.2M features, 71.9% CPCV / 73.9% holdout)
- model_1w_cpcv_backup.json

### Cross Feature Data
- v2_cross_names_BTC_1w.json
- 1d artifacts: OLD/STALE (v3.0 min_nonzero=8, needs rebuild)

### Documentation (AUTHORITATIVE)
- SESSION_RESUME.md (this file)
- OPTIMIZATION_PLAN.md — comprehensive optimization plan
- CLAUDE.md — project rules + lessons learned
- TRAINING_1W.md through TRAINING_15M.md — per-TF training guides

### GPU Fork
- v3.3/gpu_histogram_fork/ — full LightGBM fork with CUDA sparse histogram kernel
- Phase 1 benchmark: 71x speedup proven
- Phase 2-3: integrated but hit EFB mismatch

## GIT STATUS
Branch: v3.3
Latest commit: `0a94b4e` — v3.3: GPU session resume, cached training, _parent_ds bugfix
Uncommitted: Various modified files (see git status)
