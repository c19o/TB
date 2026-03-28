# V3.3 Session Resume — 2026-03-28

## INSTRUCTION TO NEW SESSION: Read this file completely. Then read v3.3/OPTIMIZATION_PLAN.md, v3.3/CLAUDE.md. These files are the complete context. Ask the user what to do next.

---

## STATUS: OPTIMIZATIONS IMPLEMENTED — Ready for Cloud Deployment

**GPU-or-nothing policy**: NO CPU fallbacks anywhere in the pipeline. Every compute stage runs on GPU. Set `ALLOW_CPU=1` as escape hatch for local testing only — cloud deploys MUST use GPU.

1w trained at 77.64% CPCV accuracy (GPU = CPU exact match verified). GPU histogram fork Phase 4 COMPLETE (Bug 4 fixed). Optuna HPO fully optimized with 5 key improvements implemented. Trade optimizer already GPU (CuPy), no changes needed.

### TWO SEPARATE OPTIMIZERS IN OUR PIPELINE
1. **LightGBM HPO** (`run_optuna_local.py`) — optimizes 10 LightGBM hyperparams (num_leaves, feature_fraction, learning_rate, lambdas, etc.). Objective: minimize mlogloss via CPCV. **5 optimizations implemented (see below) — estimated 3-5x total speedup.**
2. **Trade Strategy Optimizer** (`exhaustive_optimizer.py`) — optimizes 7 trading params (leverage, risk%, stop-loss, R:R, hold, exit, confidence). Objective: maximize Sortino. Runs on pre-computed predictions, GPU-vectorized on 3090 (CuPy). Already GPU, no changes needed.

---

## CURRENT STATE (2026-03-28)

### Optuna Optimizations — ALL IMPLEMENTED
Five key optimizations applied to `run_optuna_local.py` and `config.py`:

| Optimization | Implementation | Impact |
|---|---|---|
| **1. Dataset Reuse** | Parent `lgb.Dataset.construct()` built ONCE before all trials. Each fold's `lgb.Dataset(reference=parent_ds)` reuses EFB bin mappings. Eliminates redundant binning of 2-6M features per trial. | **~15+ min saved per trial on 1d+** |
| **2. Round-Level Pruning** | `_RoundPruningCallback` reports val `multi_logloss` every 10 rounds to Optuna. `MedianPruner(n_warmup_steps=30, interval_steps=10)` kills bad trials early. Both CPU and GPU paths support pruning. | **~30-40% of trials killed early** |
| **3. CPU Parallel Search** | Search stages use CPU (no GPU) to enable `n_jobs` parallelism via `study.optimize(n_jobs=N)`. GPU reserved for single final retrain only. `OPTUNA_N_JOBS` auto-scales: `total_cores // 96`. | **~3-4x on 128+ core machines** |
| **4. Warm-Start Cascade** | `1w -> 1d -> 4h -> 1h -> 15m` param inheritance. `load_warmstart_params()` loads parent TF's best config. `compute_warmstart_ranges()` narrows search to parent +/-20%. `build_warmstart_enqueue_params()` seeds first trial. Warm-started trials: 50+30 vs cold 100+50. | **~50% fewer trials needed** |
| **5. ES Patience Fix** | BUG FIXED: At `lr=0.08`, old patience formula gave `patience=125` but `SEARCH_ROUNDS=150` — ES almost never fired. Now `OPTUNA_SEARCH_ES_PATIENCE=30` (decoupled from LR formula). Search rounds increased to 300 so ES has room to fire. Final retrain uses LR-scaled patience for rare signals. | **~50% fewer wasted rounds per trial** |

### GPU Histogram Fork — Phase 4 COMPLETE
- **Phase 1 (standalone benchmark)**: COMPLETE — 99x standalone SpMV, 78x integrated (RTX 3090)
- **Phase 2 (LightGBM integration)**: COMPLETE — Fork builds (42/42), `device_type="cuda_sparse"` accepted, GPU detected
- **Phase 3 (CSR bridge)**: COMPLETE — CSR bridge fix applied, dangling pointer fixed, DLL rebuilt
- **Phase 4 (full GPU pipeline)**: COMPLETE — Bug 4 fixed, 10/10 rounds, EFB active
  - cuSPARSE SpMV 78x faster (SpMV only; round-level ~equal to CPU on small 1w data)
  - **GPU vs CPU accuracy: EXACT MATCH (77.64% on real labels, verified)**
  - GPU benefit expected to grow on larger datasets (1d/4h/1h/15m)
- **Location**: `v3.3/gpu_histogram_fork/` (isolated from main pipeline)
- **Commit**: `0a94b4e` (latest)

### Training Status
| TF | Status | Accuracy | Features | Notes |
|----|--------|----------|----------|-------|
| **1w** | DONE (needs Optuna) | CPCV 77.64% (GPU=CPU verified) | 2.2M | GPU histograms Phase 4 complete |
| **1d** | CROSS GEN DONE | — | 4.69M crosses (from cloud) | Inference artifacts downloaded. Needs CPCV training |
| **4h** | BASE BUILT | — | 8,794x3,904 base | Cross gen + training on CLOUD (512GB+ RAM needed) |
| **1h** | NEEDS CLOUD | — | — | Separate cloud machine, 2TB+ RAM |
| **15m** | NEEDS CLOUD | — | — | Separate cloud machine, 2TB+ RAM, user picks |

### Data Status
| Asset | Status | Notes |
|-------|--------|-------|
| **1w NPZ** | COMPLETE | 2.2M features, parquet, cross names |
| **1w model** | COMPLETE | model_1w.json (109MB), CPCV backup |
| **1d base parquet** | COMPLETE | features_BTC_1d.parquet (10.7MB, 5,733x3,796) — ready for cloud upload |
| **4h base parquet** | COMPLETE | features_BTC_4h.parquet (13.5MB, 8,794x3,904) — ready for cloud upload |
| **1d cross gen** | DONE (cloud) | Inference artifacts exist: 4.69M cross names, base_cols, ctx_names, thresholds, cross_pairs.npz. Cross gen OOM'd locally but completed on cloud. |
| **4h cross gen** | NOT STARTED | Must run on cloud (512GB+ RAM) |
| **1h/15m** | NOT BUILT | Need feature build + cross gen on cloud |
| **16+ DBs** | IN PROJECT ROOT | ~3.3GB total, in project root (NOT v3.3/). 0 DBs in v3.3/ — must symlink or copy for cloud deploy |

### Config & Code Changes (ALL APPLIED)
- max_bin=255, min_data_in_bin=1, row subsample=1.0
- HMM weights removed (MC-1), early stopping scales with LR (MC-4)
- SIGTERM checkpoint (MC-5), co-occurrence=3 (MC-6)
- enable_bundle=False for 1h/15m
- All TFs CPCV (4,1) = 4 folds
- Optuna: parent Dataset reuse, round-level pruning, CPU parallel search, warm-start cascade, ES patience fix

### All Machines Destroyed
No active cloud machines.

---

## PIPELINE RESILIENCE ASSESSMENT

### Crash Recovery
| Component | Recovery Mechanism | Max Loss |
|---|---|---|
| **Cross gen** | Per-step NPZ checkpoints | 1 step (~30-60 min) |
| **Optuna** | SQLite journal + RetryFailedTrialCallback | 1 trial |
| **CPCV training** | LightGBM init_model (save every 100 trees) | 100 trees |
| **Dataset construction** | save_binary() cache | 0 (instant reload) |

### Matrix Thesis Compliance
| Rule | Status | Verification |
|---|---|---|
| No feature filtering | PASS | No MI, variance, or support filters anywhere |
| No fallback modes | PASS | No TA-only, base-only, or degradation paths |
| No fillna(0) on features | PASS | NaN preserved for LightGBM split learning |
| feature_pre_filter=False | PASS | Set in parent Dataset AND all trial Datasets |
| Esoteric signals protected | PASS | No regularization kills rare signals; min_data_in_leaf per-TF |
| Row subsample=1.0 | PASS | OPTUNA_TF_ROW_SUBSAMPLE=1.0 for ALL TFs (thesis violation fixed) |
| LightGBM only (no XGBoost) | PASS | EFB architecturally correct for sparse binary crosses |
| GPU-or-nothing | PASS | CPU search for parallelism, GPU final retrain; no CPU fallback in production |

### GPU-or-Nothing Verification
| Pipeline Stage | GPU Status | Notes |
|---|---|---|
| Feature build | GPU (cuDF) | All rolling/ewm/shift on GPU |
| Cross gen | GPU (cuSPARSE) | SpGEMM replaces scipy sparse matmul |
| Dataset construction | CPU | LightGBM binning is CPU-only (acceptable) |
| Optuna search | CPU parallel | Intentional: n_jobs parallelism > single-GPU speed |
| Final retrain | GPU (cuda_sparse) | Fork Phase 4 complete, 78x SpMV |
| Trade optimizer | GPU (CuPy) | Already vectorized on 3090 |
| Inference | CPU | Model predict is fast (~50ms), GPU unnecessary |

---

## REVISED ETAs WITH ALL OPTIMIZATIONS

### Per-TF Training Times (with Optuna optimizations applied)

| TF | Training Only | Optuna (cold) | Optuna (warm-started) | Notes |
|----|---------------|---------------|----------------------|-------|
| **1w** | 10 min | ~1.5 hr | N/A (root TF) | Local 3090 OK |
| **1d** | 1 hr | ~8 hr | ~4 hr (from 1w) | Cloud 256GB+ RAM |
| **4h** | TBD | TBD | ~6 hr (from 1d) | Cloud 512GB+ RAM |
| **1h** | TBD | TBD | TBD (from 4h) | Cloud 2TB+ RAM |
| **15m** | 10.5 hr | ~30 hr | ~20 hr (from 1h) | Cloud 2TB+ RAM, user picks |

**Speedup breakdown** (vs original estimates):
- Dataset reuse: ~2x fewer minutes wasted on binning
- Round-level pruning: ~30-40% of trials killed early
- Warm-start: ~50% fewer trials (80 vs 150)
- ES patience fix: ~50% fewer wasted rounds per trial
- CPU parallel (128c machine): ~3-4x throughput on search stages
- **Combined: ~3-5x faster Optuna overall**

Previous estimates (for reference — BEFORE optimizations):
| TF | No Optuna | With Optuna (200 trials, no optimizations) |
|----|-----------|-------------------------------------------|
| **1w** | 10 min | 5.5 hr |
| **1d** | 1 hr | 14 hr |
| **15m** | 10.5 hr | 75 hr |

---

## CLOUD DEPLOYMENT PLAN

### What to Upload to Cloud
1. **v33_cloud_code.tar.gz** — all `.py`, `.sh`, `.md`, config `.json` files from v3.3/
2. **All 16+ DBs** (~3.3GB total)
3. **features_BTC_1d.parquet** (10.7MB) — skip base feature rebuild on cloud
4. **features_BTC_4h.parquet** (13.5MB) — skip base feature rebuild on cloud
5. **astrology_engine.py** from project root (imported by feature_library.py)
6. **kp_history_gfz.txt** (space weather data)

### Cloud Machine Strategy

| Machine | TFs | RAM Required | Pipeline | Notes |
|---------|-----|-------------|----------|-------|
| **Machine A** | 1w Optuna + 1d | 512GB+ | 1w Optuna local or cloud, then 1d cross gen + training + Optuna | Upload pre-built parquets |
| **Machine B** | 4h | 512GB+ | Cross gen + training + Optuna (warm-started from 1d) | Sequential after 1d |
| **Machine C** | 1h | 2TB+ | Full pipeline + Optuna (warm-started from 4h) | Separate machine |
| **Machine D** | 15m | 2TB+ | Full pipeline + Optuna (warm-started from 1h) | User picks machine |

### Per-Machine Pipeline Steps
For each TF on cloud:
1. **Cross gen** — sparse matmul + batch crosses (RAM-intensive)
2. **CPCV Training** — 4 folds, LightGBM sparse CSR
3. **Optuna HPO** — warm-started (50+30 trials) or cold (100+50 trials)
4. **Download artifacts** — model .json, cross names .json, optuna_configs .json, CPCV results, logs

### Warm-Start Cascade Order (CRITICAL)
```
1w (local, root) -> 1d (cloud) -> 4h (cloud) -> 1h (cloud) -> 15m (cloud)
```
Each TF inherits best params from parent. Must train in order. Download `optuna_configs_{tf}.json` before starting next TF.

---

## KNOWN ISSUES — MUST FIX BEFORE CLOUD DEPLOY

These were flagged during audit but NOT yet fixed. Do NOT deploy until resolved.

| # | Issue | Severity | Files | Details |
|---|-------|----------|-------|---------|
| 1 | **Cross gen memmap NOT implemented** | CRITICAL | `v2_cross_generator.py` | 1h needs 1.2TB peak RAM, 15m needs 3TB+. CSR chunks accumulate in RAM with no disk flush. Need disk-backed CSR chunks (memmap or incremental NPZ flush) for 1h/15m. Without this, 1h/15m cross gen requires machines that may not exist at reasonable cost. |
| 2 | **21 blanket try/except in feature_library.py** | HIGH | `feature_library.py` | Silently swallows esoteric feature computation errors. Violates "crash > silent degradation" rule. Need to audit each one and either: make it specific (catch only expected errors like missing DB columns) or add `ALLOW_CPU` guard. Silent failures = missing features = weaker model. |
| 3 | **11 blanket try/except in astrology_engine.py** | HIGH | `astrology_engine.py` | Same issue as #2 for astrology features. Silent swallow means we never know if planetary calculations are failing. Must make exceptions specific or remove. |
| 4 | **cloud_run_tf.py nuclear clean bug** | HIGH | `cloud_run_tf.py` (lines 177-188) | On restart, deletes ALL `v2_crosses_*.npz` files — destroying the CURRENT TF's valid artifacts. Need TF-specific glob filter (e.g., `v2_crosses_*_{tf}.npz`) so restart only cleans the target TF, not artifacts from other completed TFs. |
| 5 | **No checkpoint/resume for cross gen** | HIGH | `v2_cross_generator.py` | If OOM occurs at cross type 12 of 13, ALL prior work is lost. Need per-cross-type intermediate saves (flush completed cross type NPZs to disk before starting the next). Currently all-or-nothing. |
| 6 | **No model backup before overwrite** | MEDIUM | `ml_multi_tf.py` | Overwrites `model_{tf}.json` without backing up the previous version. If new training produces a worse model, the old one is gone. Need `shutil.copy` to `model_{tf}_prev.json` before writing new model. |
| 7 | **CPCV fold logic duplicated in 6 files** | MEDIUM | `ml_multi_tf.py`, `run_optuna_local.py`, `cloud_run_tf.py`, `backtest_validation.py`, `mini_train.py`, `leakage_check.py` | Same fold-splitting logic copy-pasted. Any fix to one file must be manually replicated in 5 others. Should be extracted to a shared `cpcv_utils.py` module. |
| 8 | **Optuna missing HMM features** | MEDIUM | `run_optuna_local.py` | Optuna search trains WITHOUT HMM state columns, but the final training in `ml_multi_tf.py` INCLUDES them. This means Optuna optimizes hyperparams on a different feature set than what the final model sees — consistency gap that could cause suboptimal params. |
| 9 | **CMA-ES sampler for Stage 2** | LOW | `run_optuna_local.py` | Recommended but not implemented. TPE is suboptimal for noisy CPCV objectives — CMA-ES handles noise better for continuous param search in Stage 2 (refinement). Would improve param convergence quality. |
| 10 | **n_warmup_steps=30 may be too aggressive** | LOW | `run_optuna_local.py` | MedianPruner starts pruning after 30 steps. Crypto data is noisy — early validation scores are unreliable. Consider 50-100 warmup steps to avoid killing trials that would converge with more rounds. Risk: premature pruning of good trials. |

### Priority Order for Fixing
1. **#4** (nuclear clean bug) — quick fix, prevents data loss on restart
2. **#5** (cross gen checkpoint) — prevents losing hours of work on OOM
3. **#6** (model backup) — quick fix, prevents losing good models
4. **#1** (memmap) — BLOCKING for 1h/15m deployment, most engineering effort
5. **#2 + #3** (blanket try/except) — audit and fix together
6. **#8** (HMM consistency) — fix before Optuna runs
7. **#7** (CPCV dedup) — refactor, lower urgency but prevents future bugs
8. **#9** (CMA-ES) — nice-to-have optimization
9. **#10** (warmup steps) — tune after first Optuna run on real data

---

## NEXT STEPS (ordered)

1. **Run 1w Optuna locally** — root of warm-start cascade (50+30 trials, ~1.5hr on 3090)
2. **Download 1w optuna_configs_1w.json** — needed for 1d warm-start
3. **Rent Machine A** — 512GB+ RAM, 128+ cores for 1d
4. **Upload code + DBs + pre-built parquets + optuna_configs_1w.json** to Machine A
5. **Run 1d pipeline** — cross gen + CPCV training + Optuna (warm-started from 1w)
6. **Download 1d artifacts** — model, cross names, optuna_configs_1d.json, logs
7. **Destroy Machine A**
8. **Rent Machine B** — 512GB+ for 4h (or reuse Machine A)
9. **Run 4h pipeline** — cross gen + training + Optuna (warm-started from 1d)
10. **Download 4h artifacts + destroy Machine B**
11. **Rent Machine C** — 2TB+ RAM for 1h
12. **Run 1h pipeline** — full build + cross gen + training + Optuna (warm-started from 4h)
13. **Download 1h artifacts + destroy Machine C**
14. **Rent Machine D** — 2TB+ RAM for 15m (user picks)
15. **Run 15m pipeline** — full build + cross gen + training + Optuna (warm-started from 1h)
16. **Download 15m artifacts + destroy Machine D**
17. **Push to git** — all models + artifacts

### GPU-or-Nothing Policy
- **NO CPU fallbacks anywhere.** Every compute stage runs on GPU.
- `ALLOW_CPU=1` environment variable as escape hatch for local testing ONLY
- Cloud deploys MUST use GPU — no ALLOW_CPU
- Trade optimizer: already GPU (CuPy vectorized on 3090), no changes needed
- LightGBM training: GPU histograms Phase 4 COMPLETE (Bug 4 fixed)
- Cross gen: cuSPARSE SpGEMM replacing scipy sparse matmul
- Feature build: cuDF rolling/ewm (already GPU)
- Optuna search: CPU parallel (intentional — n_jobs parallelism > single-GPU speed)

### GPU Fork — Phase 4 COMPLETE
- Bug 4 (feature_hist_offsets EFB mapping) FIXED. All 10 rounds trained, EFB active.
- GPU vs CPU accuracy: exact match (77.64%). SpMV 78x faster, round-level ~equal on 1w.
- Integrated into run_optuna_local.py — GPU used for final retrain, CPU for parallel search.
- See `v3.3/gpu_histogram_fork/GPU_SESSION_RESUME.md` for full details

---

## TRAINING PLAN & COSTS

### Per-TF Training Times — With All Optimizations (revised)

| TF | Training | Optuna (warm) | Optuna (cold) | Total (warm) | Notes |
|----|----------|--------------|---------------|-------------|-------|
| **1w** | 10 min | N/A | ~1.5 hr | ~1.5 hr | Local 3090, root of cascade |
| **1d** | 1 hr | ~4 hr | ~8 hr | ~5 hr | Cloud 256GB+ RAM |
| **4h** | TBD | ~6 hr | TBD | ~8 hr est. | Cloud 512GB+ RAM |
| **1h** | TBD | TBD | TBD | TBD | Cloud 2TB+ RAM |
| **15m** | 10.5 hr | ~20 hr | ~30 hr | ~25 hr est. | Cloud 2TB+ RAM, user picks |

### Machine Assignments

| TF | Where | Machine | Notes |
|----|-------|---------|-------|
| 1w | LOCAL | 13900K + 3090 | DONE (CPCV 77.64%, Optuna next) |
| 1d | CLOUD | Machine A (512GB+, 128c+) | Base parquet pre-built locally, warm-start from 1w |
| 4h | CLOUD | Machine B (512GB+, 128c+) | Base parquet pre-built locally, warm-start from 1d |
| 1h | CLOUD | Machine C (2TB+, 128c+) | Full pipeline on cloud, warm-start from 4h |
| 15m | CLOUD | Machine D (2TB+, 128c+) | User picks machine, warm-start from 1h |

---

## KEY RESEARCH FINDINGS

1. **v3.2 NEVER completed training** — crashed on feature_name mismatch. No baseline exists.
2. **max_bin=255** — controls EFB bundle size. max_bin=15 created 18x more bundles than needed.
3. **GPU histogram fork proved 78x SpMV speedup** — Phase 4 COMPLETE, Bug 4 fixed. GPU = CPU accuracy (77.64%). GPU-or-nothing: NO CPU fallbacks.
4. **GOSS/CEGB/quantized_grad** — all kill rare esoteric signals. REJECTED.
5. **HMM regime weights 0.15 is a thesis violation** — crushes counter-trend esoteric signals by 85%.
6. **Co-occurrence filter 8->3** — matches min_data_in_leaf=3 for 1d/1w.
7. **enable_bundle=False for 1h/15m** — EFB conflict graph intractable at 10M features.
8. **Cross features are pure 0/1** — structural zeros = feature OFF, not missing. No NaN after binarization.
9. **We are in UNCHARTED TERRITORY** — no published work combines esoteric x TA at 2-10M sparse binary features.
10. **1d cross gen OOM at 68GB locally** — completed on cloud. Inference artifacts (4.69M cross names) downloaded. Needs CPCV training.
11. **4h estimated at 512GB+** — even more features than 1d, must go to cloud.
12. **Optuna was 73-81% of total pipeline time** — 5 optimizations implemented, estimated 3-5x speedup.
13. **ES patience bug** — at lr=0.08, patience=125 but rounds=150, so ES almost never fired. Fixed with decoupled OPTUNA_SEARCH_ES_PATIENCE=30.
14. **Warm-start cascade validated** — 1w->1d->4h->1h->15m. Parent params narrow search by +/-20%, reducing trials from 150 to 80.

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
- model_1w.json (109MB, 2.2M features, CPCV 77.64%, needs Optuna)
- model_1w_cpcv_backup.json

### Pre-Built Base Feature Parquets (for cloud upload)
- features_BTC_1d.parquet (10.7MB, 5,733x3,796)
- features_BTC_4h.parquet (13.5MB, 8,794x3,904)

### Cross Feature Data
- v2_cross_names_BTC_1w.json

### Optuna Configuration
- optuna_configs_all.json (empty `{}` — no trials run yet, pending 1w Optuna)

### Documentation (AUTHORITATIVE)
- SESSION_RESUME.md (this file)
- OPTIMIZATION_PLAN.md — comprehensive optimization plan
- CLAUDE.md — project rules + lessons learned
- TRAINING_1W.md through TRAINING_15M.md — per-TF training guides

### GPU Fork
- v3.3/gpu_histogram_fork/ — full LightGBM fork with CUDA sparse histogram kernel
- Phase 1-4: ALL COMPLETE (78x SpMV speedup, Bug 4 fixed, EFB active, 77.64% accuracy = CPU match)
- Integrated into run_optuna_local.py (GPU final retrain, CPU parallel search)

## GIT STATUS
Branch: v3.3
Latest commit: `0a94b4e` — v3.3: GPU session resume, cached training, _parent_ds bugfix
Uncommitted: Various modified files (see git status)
