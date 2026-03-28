# V3.3 Session Resume — 2026-03-28

## INSTRUCTION TO NEW SESSION: Read this file completely. Then read v3.3/OPTIMIZATION_PLAN.md, v3.3/CLAUDE.md. These files are the complete context. Ask the user what to do next.

---

## STATUS: 1W DONE → 1D/4H CLOUD REQUIRED (OOM locally) → BUILD REMAINING TFs

1w training complete locally (CPCV 67.7%, needs Optuna). 1d cross gen crashed locally at 68GB RAM — needs 256GB+. Both 1d and 4h must train on cloud. GPU histogram fork hit architectural wall (EFB mismatch). CPU training already works well.

### TWO SEPARATE OPTIMIZERS IN OUR PIPELINE
1. **LightGBM HPO** (`run_optuna_local.py`) — optimizes 10 LightGBM hyperparams (num_leaves, feature_fraction, learning_rate, lambdas, etc.). Objective: minimize mlogloss via CPCV. THIS is the 90% bottleneck (200 trials × 4-15 folds × 150 min/fold).
2. **Trade Strategy Optimizer** (`exhaustive_optimizer.py`) — optimizes 7 trading params (leverage, risk%, stop-loss, R:R, hold, exit, confidence). Objective: maximize Sortino. Runs on pre-computed predictions, GPU-vectorized on 3090. Fast (minutes).

---

## CURRENT STATE (2026-03-28)

### GPU Histogram Fork — Phase 4: Full GPU Pipeline Acceleration
- **Phase 1 (standalone benchmark)**: COMPLETE — 71x speedup on real 1w data (RTX 3090)
- **Phase 2 (LightGBM integration)**: COMPLETE — Fork builds (42/42), `device_type="cuda_sparse"` accepted, GPU detected
- **Phase 3 (CSR bridge)**: COMPLETE — CSR bridge fix applied, dangling pointer fixed, DLL rebuilt
- **Phase 4 (full GPU pipeline acceleration)**: IN PROGRESS — 20 parallel agents implementing 4 components:
  1. **cuSPARSE SpGEMM** replacing scipy sparse matmul in cross gen (15-40x speedup) — scipy `left_sp.T @ right_sp` is single-threaded CPU, cuSPARSE does sparse-sparse matmul on GPU
  2. **GPU nonzero** replacing CPU `np.nonzero` (3-5x speedup) — CuPy/CUDA kernel extracts nonzero indices directly on GPU
  3. **binarize_contexts vectorized with Numba prange** (10-20x speedup) — saturates multi-core CPUs (64-512 cores on cloud)
  4. **LightGBM feature_hist_offsets mapping fix** — extracts cumulative bin offset table from `share_state_`, uploads to GPU, remaps SpMV output indices in CUDA kernel. Re-enables EFB bundling with GPU histograms (was BLOCKED in Phase 3)
- **Expected 15m pipeline**: 90-174h (CPU) → 15-25h (GPU-accelerated)
- **Location**: `v3.3/gpu_histogram_fork/` (isolated from main pipeline)
- **Commit**: `0a94b4e` (latest)

### Training Status
| TF | Status | Accuracy | Features | Notes |
|----|--------|----------|----------|-------|
| **1w** | DONE | CPCV 67.7% (needs Optuna) | 2.2M | CPU training, real labels |
| **1d** | BASE BUILT | — | 5,733×3,796 base | Cross gen OOM at 68GB locally. Needs 256GB+ RAM → CLOUD |
| **4h** | BASE BUILT | — | 8,794×3,904 base | Cross gen + training → CLOUD (512GB+ RAM needed) |
| **1h** | NEEDS CLOUD | — | — | Separate cloud machine, 2TB+ RAM |
| **15m** | NEEDS CLOUD | — | — | Separate cloud machine, 2TB+ RAM, user picks |

### Data Status
| Asset | Status | Notes |
|-------|--------|-------|
| **1w NPZ** | COMPLETE | 2.2M features, parquet, cross names |
| **1w model** | COMPLETE | model_1w.json (113MB), CPCV backup |
| **1d base parquet** | COMPLETE | features_BTC_1d.parquet (10.7MB, 5,733×3,796) — ready for cloud upload |
| **4h base parquet** | COMPLETE | features_BTC_4h.parquet (13.5MB, 8,794×3,904) — ready for cloud upload |
| **1d cross gen** | FAILED LOCALLY | OOM at 68GB. Must run on cloud (256GB+ RAM) |
| **4h cross gen** | NOT STARTED | Must run on cloud (512GB+ RAM) |
| **1h/15m** | NOT BUILT | Need feature build + cross gen on cloud |
| **16 DBs** | READY | ~3.3GB total, all in v3.3 directory |

### Config & Code Changes (ALL APPLIED)
- max_bin=255, min_data_in_bin=1, row subsample=1.0
- HMM weights removed (MC-1), early stopping scales with LR (MC-4)
- SIGTERM checkpoint (MC-5), co-occurrence=3 (MC-6)
- enable_bundle=False for 1h/15m
- All TFs CPCV (4,1) = 4 folds

### All Machines Destroyed
No active cloud machines.

---

## CLOUD DEPLOYMENT PLAN

### What to Upload to Cloud
1. **v33_cloud_code.tar.gz** — all `.py`, `.sh`, `.md`, config `.json` files from v3.3/
2. **All 16 DBs** (~3.3GB total)
3. **features_BTC_1d.parquet** (10.7MB) — skip base feature rebuild on cloud
4. **features_BTC_4h.parquet** (13.5MB) — skip base feature rebuild on cloud
5. **astrology_engine.py** from project root (imported by feature_library.py)
6. **kp_history_gfz.txt** (space weather data)

### Cloud Machine Strategy

| Machine | TFs | RAM Required | Pipeline | Notes |
|---------|-----|-------------|----------|-------|
| **Machine A** | 1d + 4h | 512GB+ | Sequential: 1d first, then 4h | Upload pre-built parquets to skip base build |
| **Machine B** | 1h | 2TB+ | Full pipeline | Separate machine — high RAM for cross gen |
| **Machine C** | 15m | 2TB+ | Full pipeline | User picks machine personally |

### Per-Machine Pipeline Steps
For each TF on cloud:
1. **Cross gen** — sparse matmul + batch crosses (RAM-intensive)
2. **CPCV Training** — 4 folds, LightGBM sparse CSR
3. **Optuna HPO** — 200 trials (optional, can do after initial training)
4. **Download artifacts** — model .json, cross names .json, CPCV results, logs

### Machine A Details (1d + 4h)
- Upload: code tar + DBs + features_BTC_1d.parquet + features_BTC_4h.parquet
- Run 1d: `python -u cloud_run_tf.py --symbol BTC --tf 1d` (skips base build, starts at cross gen)
- Download 1d artifacts
- Run 4h: `python -u cloud_run_tf.py --symbol BTC --tf 4h` (skips base build, starts at cross gen)
- Download 4h artifacts
- **Estimated time**: 1d ~2.5hr + 4h ~5-10hr = ~8-13hr total

---

## NEXT STEPS (ordered)

1. **Rent Machine A** — 512GB+ RAM, 64+ cores for 1d + 4h
2. **Upload code + DBs + pre-built parquets** to Machine A
3. **Run 1d pipeline** — cross gen + CPCV training
4. **Download 1d artifacts** — model, cross names, logs
5. **Run 4h pipeline** — cross gen + CPCV training (same machine)
6. **Download 4h artifacts** — model, cross names, logs
7. **Destroy Machine A**
8. **Rent Machine B** — 2TB+ RAM for 1h
9. **Run 1h pipeline** — full build + cross gen + training
10. **Download 1h artifacts + destroy Machine B**
11. **Rent Machine C** — 2TB+ RAM for 15m (user picks)
12. **Run 15m pipeline** — full build + cross gen + training
13. **Download 15m artifacts + destroy Machine C**
14. **Run Optuna** on all models (can be local for 1w, cloud for others)
15. **Push to git** — all models + artifacts

### GPU Fork — Phase 4 Active (no longer deferred)
- Phase 4 implements full GPU pipeline acceleration via 20 parallel agents
- 4 components: cuSPARSE SpGEMM (cross gen), GPU nonzero, Numba prange binarize, feature_hist_offsets mapping fix (LightGBM EFB)
- Target: 15m pipeline 90-174h → 15-25h
- See `v3.3/gpu_histogram_fork/GPU_SESSION_RESUME.md` for full details

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

### Machine Assignments (UPDATED — local OOM forced cloud)

| TF | Where | Machine | Notes |
|----|-------|---------|-------|
| 1w | LOCAL | 13900K + 3090 | DONE (CPCV 67.7%, needs Optuna) |
| 1d | CLOUD | Machine A (512GB+, 64c+) | Base parquet pre-built locally, upload to skip rebuild |
| 4h | CLOUD | Machine A (same, sequential) | Base parquet pre-built locally, upload to skip rebuild |
| 1h | CLOUD | Machine B (2TB+, 128c+) | Full pipeline on cloud |
| 15m | CLOUD | Machine C (2TB+, 128c+) | User picks machine |

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
10. **1d cross gen OOM at 68GB locally** — needs 256GB+ RAM. Base features build fine (5,733×3,796).
11. **4h estimated at 512GB+** — even more features than 1d, must go to cloud.

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
- model_1w.json (113MB, 2.2M features, CPCV 67.7%, needs Optuna)
- model_1w_cpcv_backup.json

### Pre-Built Base Feature Parquets (for cloud upload)
- features_BTC_1d.parquet (10.7MB, 5,733×3,796)
- features_BTC_4h.parquet (13.5MB, 8,794×3,904)

### Cross Feature Data
- v2_cross_names_BTC_1w.json

### Documentation (AUTHORITATIVE)
- SESSION_RESUME.md (this file)
- OPTIMIZATION_PLAN.md — comprehensive optimization plan
- CLAUDE.md — project rules + lessons learned
- TRAINING_1W.md through TRAINING_15M.md — per-TF training guides

### GPU Fork
- v3.3/gpu_histogram_fork/ — full LightGBM fork with CUDA sparse histogram kernel
- Phase 1 benchmark: 71x speedup proven
- Phase 2-3: integrated, CSR bridge fixed, EFB mismatch identified
- Phase 4: full GPU pipeline acceleration in progress (20 parallel agents)
  - cuSPARSE SpGEMM (15-40x), GPU nonzero (3-5x), Numba prange binarize (10-20x), feature_hist_offsets fix (EFB re-enabled)
  - Target: 15m pipeline 90-174h → 15-25h

## GIT STATUS
Branch: v3.3
Latest commit: `0a94b4e` — v3.3: GPU session resume, cached training, _parent_ds bugfix
Uncommitted: Various modified files (see git status)
