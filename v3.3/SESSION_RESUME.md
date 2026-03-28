# V3.3 Session Resume — 2026-03-27 23:30 UTC

## INSTRUCTION TO NEW SESSION: Read this file completely. Then read v3.3/OPTIMIZATION_PLAN.md, v3.3/CLAUDE.md, and v3.3/CLOUD_TRAINING_PROTOCOL.md. These four files are the complete context. Ask the user what to do next.

---

## STATUS: CONFIG + MC FIXES IMPLEMENTED → READY FOR CROSS-GEN OPTIMIZATION + TRAINING

Code changes from OPTIMIZATION_PLAN.md Steps 1-2 are DONE. Next: implement cross-gen optimization (Phase 1), training optimization (Phase 2), then train all 5 TFs.

### TWO SEPARATE OPTIMIZERS IN OUR PIPELINE
1. **LightGBM HPO** (`run_optuna_local.py`) — optimizes 10 LightGBM hyperparams (num_leaves, feature_fraction, learning_rate, lambdas, etc.). Objective: minimize mlogloss via CPCV. THIS is the 90% bottleneck (200 trials × 4-15 folds × 150 min/fold).
2. **Trade Strategy Optimizer** (`exhaustive_optimizer.py`) — optimizes 7 trading params (leverage, risk%, stop-loss, R:R, hold, exit, confidence). Objective: maximize Sortino. Runs on pre-computed predictions, GPU-vectorized on 3090. Fast (minutes).

### GPU ACCELERATION VERDICT (researched thoroughly)
- **LightGBM CUDA does NOT support sparse CSR** — confirmed from source. Dense = 2.94TB for 15m. Impossible.
- **ThunderGBM**: Dead project (last release 2019), int32 overflow at our 5.88B NNZ, no EFB, no Python API. NOT VIABLE.
- **ScalaGBM (KDD 2025)**: Same int32 limit, no EFB, no Python API, 5 GitHub stars. Research prototype only. NOT VIABLE TODAY.
- **B200 GPU**: Would sit idle. Our workload is 100% CPU-bound. GCP c3d-360 ($3.40/hr) with 360 CPU cores is 24x better value.
- **Correct architecture**: LightGBM + EFB + sparse CSR + high-core-count CPU. EFB is the irreplaceable advantage.

---

## CURRENT STATE

### Completed
- **1w model**: OLD model (71.9%) used stale 658K-feature NPZ. NEW training with correct 2.2M features IN PROGRESS on vast.ai (instance 33197100, fold 1/4). See bugs fixed below.
- **1d cross gen**: OLD NPZ in v3.3/ is stale (v3.0, min_nonzero=8). Must rebuild with min_nonzero=3 on cloud. 1d machine (instance 33686947) is STOPPED with code+DBs uploaded.
- **Optimization research**: 39 agent rounds. Consolidated in OPTIMIZATION_PLAN.md.
- **All docs updated**: CLAUDE.md, SESSION_RESUME.md, OPTIMIZATION_PLAN.md, all 5 TRAINING_*.md files.
- **Config changes (Step 1)**: ALL DONE — max_bin=255, max_conflict_rate removed, min_data_in_bin=1, row subsample=1.0, enable_bundle=False for 1h/15m, 15m CPCV (4,1)
- **MC fixes (Step 2)**: ALL DONE — MC-1 (HMM weights removed), MC-2 (row subsample), MC-4 (early stopping scaled), MC-5 (SIGTERM checkpoint), MC-6 (co-occurrence 3), MC-8 (atomic NPZ already existed)
- **Cascade fixes**: ALL DONE — max_bin=255 in feature_importance_pipeline.py (3 locations), smoke_test_v3.py, run_optuna_local.py, v2_multi_asset_trainer.py. Stale comments updated.

### User Decisions Made (2026-03-27)
1. **MC-1**: Remove HMM regime weighting entirely, add HMM state as input feature (Option A — fully matrix-aligned)
2. **MC-3**: Accept save_binary speed — zero leakage for binary crosses (bin boundary always 0.5)
3. **ALL TFs CPCV**: (4,1)=4 folds for ALL timeframes. Production model accuracy identical regardless of fold count (trains on ALL data). Multi-machine distribution researched + audited but scrapped (diminishing returns). See FOLD_STRATEGY.md.
4. **No Optuna for initial training**: Fixed params first. Train all 5 TFs, evaluate, then optionally tune 2-3 params.

### GPU Histogram Fork Status
- **Phase 1 (standalone benchmark)**: COMPLETE
  - 71x speedup measured on real 1w data (RTX 3090)
  - 473x on synthetic 1d data
  - cuSPARSE SpMV approach proven
  - 111 tests written
- **Phase 2 (LightGBM integration)**: IN PROGRESS
  - Fork builds successfully (42/42 compiled)
  - `device_type="cuda_sparse"` accepted
  - GPU detected: RTX 3090, 82 SMs, 24GB VRAM
  - SetExternalCSR C API needed (in progress)
  - Config.cpp patched for cuda_sparse validation
- **Commit**: `1f7db7c` (57 files, 25,537 lines)
- **Location**: `v3.3/gpu_histogram_fork/` (isolated from main pipeline)

### Not Started
- Cross-gen optimization (Phase 1 from OPTIMIZATION_PLAN.md — Numba kernel, parallel steps, adaptive chunks)
- Training optimization (Phase 2 — parent Dataset + .subset(), save_binary(), parallel Optuna)
- Training for 1d, 4h, 1h, 15m
- Optuna hyperparameter optimization

### All Machines Destroyed
No active cloud machines. vast.ai balance: ~$25.

---

## WHAT TO IMPLEMENT (ordered by priority)

### STEP 1: Config changes ✓ DONE
File: `v3.3/config.py`
- [x] Change max_bin from 15 to 255 in V3_LGBM_PARAMS
- [x] Remove max_conflict_rate from V3_LGBM_PARAMS (replaced with min_data_in_bin=1)
- [x] Add min_data_in_bin=1 to V3_LGBM_PARAMS
- [x] Set OPTUNA_TF_ROW_SUBSAMPLE to 1.0 for ALL TFs
- [x] Add TF_ENABLE_BUNDLE dict + enable_bundle=False for 1h/15m TFs
- [x] Change 15m TF_CPCV_GROUPS from (6,2) to (4,1)

### STEP 2: Matrix compliance fixes ✓ DONE
- [x] MC-1: HMM regime weights REMOVED (Option A). HMM state used as input feature. ml_multi_tf.py + run_optuna_local.py.
- [x] MC-2: OPTUNA_TF_ROW_SUBSAMPLE = 1.0 for ALL TFs
- [x] MC-3: ACCEPTED — save_binary leakage is zero for binary crosses (bin boundary always 0.5). USER DECISION.
- [x] MC-4: Early stopping scales with LR: `ES = max(50, int(100 * (0.1 / lr)))`. All 5 locations in ml_multi_tf.py + run_optuna_local.py.
- [x] MC-5: SIGTERM checkpoint callback added — saves model every 100 rounds + on SIGTERM. Sequential + final training paths.
- [x] MC-6: smoke_test_v3.py MIN_CO_OCCURRENCE changed from 8 to 3
- [x] MC-8: Atomic NPZ writes — already implemented via atomic_io.py (temp+os.replace)
- [ ] MC-7: Used-features-only inference — deferred to post-training Phase 6

### Cascade fixes ✓ DONE
- [x] max_bin=255 in feature_importance_pipeline.py (3 locations)
- [x] max_bin=255 in smoke_test_v3.py fallback params
- [x] max_bin=255 locked in run_optuna_local.py (was 15)
- [x] max_bin search narrowing removed (locked at 255)
- [x] max_conflict_rate removed from run_optuna_local.py final_retrain (replaced with min_data_in_bin=1)
- [x] enable_bundle wired into ml_multi_tf.py + run_optuna_local.py from TF_ENABLE_BUNDLE config
- [x] Docstrings updated: ml_multi_tf.py, v2_multi_asset_trainer.py

### STEP 3: Cross gen optimization (2-4 hours)
File: `v3.3/v2_cross_generator.py`
- [ ] Numba sorted-index intersection kernel (replaces dense materialization in _cpu_cross_chunk)
- [ ] Bitpacked POPCNT for co-occurrence pre-filter (replaces sparse matmul)
- [ ] LLVM intrinsics for POPCNT/CTZ via numba.extending.intrinsic (no C extensions)
- [ ] All 13 cross steps parallel via ThreadPoolExecutor (confirmed independent)
- [ ] Adaptive RIGHT_CHUNK controller (rolling RSS-based sizing)
- [ ] Per-step NPZ checkpointing with atomic temp+rename
- [ ] Memmap CSC streaming for 1h/15m (Phase 1E — required, 1h peaks at 1.8TB without it)
- [ ] Sort pairs by left index for L2 cache reuse (Phase 1F — 2-5x on 15m)

### STEP 4: Training optimization (1-2 hours)
Files: `v3.3/ml_multi_tf.py`, `v3.3/run_optuna_local.py`
- [ ] Parent Dataset + .subset() for CPCV folds (reconstruct per fold, not shared across folds)
- [ ] save_binary() for Dataset caching (skip construction on Optuna trials)
- [ ] Parallel Optuna n_jobs=4 with constant_liar=True TPESampler
- [ ] Successive halving for Stage 2 (30 low → 8 mid → 3 high instead of 100 full trials)
- [ ] Cross-TF warm-start: enqueue best params from cheaper TFs

NOTE: CPCV must stay SEQUENTIAL for all TFs with >1M features (1d/4h/1h/15m). Dense+parallel CPCV = 400GB pickle bottleneck. Do NOT attempt to parallelize CPCV within a process.

### STEP 5: OS tuning (add to setup.sh)
- [ ] tcmalloc via LD_PRELOAD (libtcmalloc_minimal.so)
- [ ] THP always + defrag=defer+madvise
- [ ] vm.swappiness=1
- [ ] NUMA binding for multi-socket machines

### STEP 6: Test locally on 1w, then deploy
- [ ] Run 1w locally to verify all changes work
- [ ] Deploy 1w + 1d to vast.ai ($4 total, CPCV only)
- [ ] Evaluate accuracy, compare 1w optimized vs 1w baseline (71.9%)
- [ ] If good, scale to 4h/1h/15m

---

## TRAINING PLAN & COSTS

### Recommended approach: Fixed params first, Optuna later
Google/Facebook/hedge funds find good params once, fix stable ones, retune 2-3 adaptive params per cycle. 200 Optuna trials is overkill. TPE converges at 50-150 trials for 10D.

### Per-TF Training Times (with optimizations, per pipeline step)

| Step | 1w | 1d | 4h | 1h | 15m |
|------|-----|-----|-----|------|------|
| Feature build | 5m | 15m | 30m | 1hr | 2hr |
| Cross gen | 15s | 10m | 36m | 2.3hr | 5.8hr |
| save_binary | 30s | 5m | 15m | 30m | 45m |
| CPCV per fold | 3m | 30m | 50m | 104m | 150m |
| CPCV total (4 folds all TFs) | 12m | 2hr | 3.3hr | 7hr | 10hr |
| **Total no Optuna** | **25m** | **2.5hr** | **5-10hr** | **9-15hr** | **13-19hr** |

**NOTE**: 15m currently uses TF_CPCV_GROUPS=(6,2) = 15 fold-combos. The 10hr estimate assumes changing to (4,1) = 4 folds. With current (6,2), CPCV would be ~37.5 hrs. This change requires user decision.

### Machine Assignments

| TF | Provider | Machine | Spot $/hr | RAM |
|----|----------|---------|-----------|-----|
| 1w | vast.ai | 64c | ~$0.30 | 256GB |
| 1d | vast.ai | 128c | ~$1.75 | 512GB |
| 4h | GCP | c3d-highmem-180 | ~$1.75 | 1440GB |
| 1h | GCP | c3d-highmem-360 | ~$3.40 | 2880GB |
| 15m | GCP | c3d-highmem-360 | ~$3.40 | 2880GB |

**CRITICAL: GCP free trial limits to 8 vCPUs. Must upgrade each account to PAID (keeps $300 credits). Then request quota increase for C3D CPUs in us-central1 to 400.**

### Budget Options

| Strategy | Cost | Wall Time | Notes |
|----------|------|-----------|-------|
| Fixed params, no Optuna, all 5 TFs | **$115** | **19 hrs** | Best value. ~1-3% accuracy loss vs optimized. |
| + 30 warm-start trials on 1w/1d | **$125** | **19 hrs** | Cheap Optuna on smallest TFs |
| + 30 warm trials all TFs | **$350** | **44 hrs** | Full coverage |
| Full 200 trials all TFs | **$615** | **94 hrs** | Overkill per Google/FB research |

### 15m Speed Options

| Architecture | Wall Time | Cost | Notes |
|-------------|-----------|------|-------|
| Single machine, no Optuna | 19 hrs | $65 | Baseline |
| Single machine, 30 warm trials | 44 hrs | $180 | Acceptable |
| Parallel CPCV folds (4 machines) | 5.5 hrs + 5.8 cross gen | $90 | Fast but complex |
| 50 Optuna workers + fold parallelism | ~6 hrs | ~$9,700 | Not realistic |

**Irreducible bottleneck**: single CPCV fold on 15m = 120-150 min. Nothing can make this faster except reducing rows (violates thesis) or faster hardware (diminishing returns past 128 cores).

**Key insight**: 15m uses TF_CPCV_GROUPS=(6,2) = 15 fold-combos. Changing to (4,1) = 4 folds (like 1w/1d) saves 73% of CPCV time with plenty of data at 294K rows.

---

## OPTUNA BOTTLENECK — RESOLVED STRATEGY

Optuna was 90-95% of pipeline time. Research found:
- Google Vizier: 15-80 trials, not 200
- Facebook: small 1-D sweeps, not joint 10-D search
- Hedge funds: find params once, fix stable ones, retune 2-3 per cycle
- TPE converges at 50-150 trials for 10D
- Robust defaults are within 1-3% of optimized

**New strategy**:
1. Run full Optuna on 1w/1d (cheap — $7 total)
2. Identify 5-6 stable params across TFs → fix permanently
3. For 4h/1h/15m: warm-start + 20-30 narrow trials with successive halving
4. Total: ~50-80 full-trial-equivalents across ALL TFs (vs current 1,000)

---

## KEY RESEARCH FINDINGS (from 39 agent rounds)

1. **v3.2 NEVER completed training** — crashed on feature_name mismatch. No baseline exists.
2. **max_bin=255** — controls EFB bundle size. max_bin=15 created 18x more bundles than needed.
3. **save_binary() + .subset()** — skip Dataset rebuilding per fold. ~0.9-1.2GB file for 1d.
4. **GOSS/CEGB/quantized_grad** — all kill rare esoteric signals. REJECTED.
5. **DART** — weight dilution risk. Keep gbdt as default, DART as Optuna option with skip_drop=0.8.
6. **Bitpacked AND** — 50-500x for computation only. CSC conversion negates for large TFs. Use sorted-index intersection as primary approach.
7. **GCP c3d-highmem-360** ($3.40/hr spot) — only reliable 2TB+ RAM option for 1h/15m.
8. **NaN semantics verified** — cross features are pure 0/1, structural zeros = feature OFF, not missing.
9. **HMM regime weights 0.15 is a thesis violation** — crushes counter-trend esoteric signals by 85%.
10. **Co-occurrence filter 8→3** in production code (smoke test still says 8).
11. **enable_bundle=False for 1h/15m** — EFB conflict graph intractable at 10M features.
12. **Optuna row subsampling is a thesis violation** — kills rare signals below min_data_in_leaf.
13. **Early stopping(50) may kill esoteric signals** — scale with LR: lr=0.05→ES(200).
14. **Used-features-only inference** — 5K of 6M needed, ~50-100ms per bar. Must use full 6M-wide sparse.
15. **We are in UNCHARTED TERRITORY** — no published work combines esoteric × TA at 2-10M sparse binary features.

---

## MATRIX THESIS COMPLIANCE: 90/100

### Fixed violations
- Row subsampling → 1.0 for all TFs
- HMM weights → flagged for fix
- save_binary leakage → USER DECISION pending (accept for speed or reconstruct per fold)
- Structural zero semantics → corrected everywhere
- Co-occurrence filter 8→3

### Remaining tensions (acceptable)
- Early stopping may clip esoteric signal learning (mitigated by scaling with LR)
- 3-class FLAT dominance (60%) drowns directional signals
- CPCV purge=10 bars may be short for eclipse effects (consider 14)

---

## ARTIFACTS IN v3.3/ DIRECTORY

### Models & Training Results
- model_1w.json (113MB, 2.2M features, 71.9% accuracy)
- model_1w_cpcv_backup.json

### Cross Feature Data (downloaded from cloud)
- v2_crosses_BTC_1d.npz (757MB, 6M features)
- v2_cross_names_BTC_1d.json (276MB)
- v2_cross_names_BTC_1w.json
- inference_1d_*.json/npz (5 files)

### Documentation (AUTHORITATIVE)
- OPTIMIZATION_PLAN.md — comprehensive optimization plan (3x reviewed, 3x matrix audited)
- CLAUDE.md — project rules + lessons learned (fully updated)
- TRAINING_1W.md through TRAINING_15M.md — per-TF training guides (all corrected with observed values)
- AUDIT_RESULTS.md, AUDIT_HANDOFF_PROMPT.md

### Config & Pipeline
- config.py — LightGBM params, Optuna config, TF settings (NEEDS max_bin + max_conflict_rate changes)
- ml_multi_tf.py — CPCV training loop (NEEDS MC fixes)
- run_optuna_local.py — Optuna HPO (NEEDS parallel + warm-start changes)
- v2_cross_generator.py — cross feature generation (NEEDS Numba kernel)
- cloud_run_tf.py — cloud pipeline orchestrator
- live_trader.py — live/paper trading

---

## REJECTED GPU ALTERNATIVES (researched, all blocked)
| Library | Blocker | Details |
|---------|---------|---------|
| LightGBM CUDA | Sparse CSR not supported | Dense = 2.94TB, impossible |
| ThunderGBM | Dead (2019), int32 overflow (5.88B NNZ > 2.1B max) | No EFB, no Python API, no callbacks |
| ScalaGBM (KDD 2025) | int32 overflow, no EFB, no Python API | 5 GitHub stars, research prototype |
| XGBoost GPU | No EFB, 12% accuracy drop | Already tested and rejected in v3.2 |
| CatBoost GPU | No EFB equivalent | int32 limits, OOM on high-dim |
| B200/H100/A100 | All blocked by above | GPU sits idle, CPU is the bottleneck |

**GPU path now being built**: Custom LightGBM fork with cuSPARSE sparse histogram kernel. Phase 1 benchmark proved 71x speedup on real data. Phase 2 LightGBM integration in progress — see GPU Histogram Fork Status above.

**CPU fallback path**: more CPU cores + EFB compression + parallel CPCV folds via save_binary()+spawn. See OPTIMIZATION_PLAN.md.

## GIT STATUS
Branch: v3.3
Latest commit: `3f3a06e` — v3.3: update SESSION_RESUME with full training status
Uncommitted: OPTIMIZATION_PLAN.md, updated CLAUDE.md, updated TRAINING_*.md, updated SESSION_RESUME.md
