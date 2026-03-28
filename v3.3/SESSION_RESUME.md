# V3.3 Session Resume — 2026-03-28

## INSTRUCTION TO NEW SESSION: Read this file completely. Then read v3.3/OPTIMIZATION_PLAN.md, v3.3/CLAUDE.md. These files are the complete context. Ask the user what to do next.

---

## STATUS: GPU-OR-NOTHING — LightGBM GPU Histograms + Optuna Re-enabled

**GPU-or-nothing policy**: NO CPU fallbacks anywhere in the pipeline. Every compute stage runs on GPU. Set `ALLOW_CPU=1` as escape hatch for local testing only — cloud deploys MUST use GPU.

1w trained at 77.64% CPCV accuracy (GPU = CPU exact match verified). GPU histogram fork Phase 4 COMPLETE (Bug 4 fixed). Optuna HPO code is wired but never executed (empty optuna_configs_all.json) — re-enabling is critical for pushing accuracy higher. Trade optimizer already GPU (CuPy), no changes needed.

### TWO SEPARATE OPTIMIZERS IN OUR PIPELINE
1. **LightGBM HPO** (`run_optuna_local.py`) — optimizes 10 LightGBM hyperparams (num_leaves, feature_fraction, learning_rate, lambdas, etc.). Objective: minimize mlogloss via CPCV. THIS is the 90% bottleneck (200 trials × 4-15 folds). **Being re-enabled — was the cause of 67.7% vs 71.9%.**
2. **Trade Strategy Optimizer** (`exhaustive_optimizer.py`) — optimizes 7 trading params (leverage, risk%, stop-loss, R:R, hold, exit, confidence). Objective: maximize Sortino. Runs on pre-computed predictions, GPU-vectorized on 3090 (CuPy). Already GPU, no changes needed.

---

## CURRENT STATE (2026-03-28)

### GPU Histogram Fork — Bug 4 Debug (20 Parallel Agents)
- **Phase 1 (standalone benchmark)**: COMPLETE — 99x standalone SpMV, 78x integrated (RTX 3090)
- **Phase 2 (LightGBM integration)**: COMPLETE — Fork builds (42/42), `device_type="cuda_sparse"` accepted, GPU detected
- **Phase 3 (CSR bridge)**: COMPLETE — CSR bridge fix applied, dangling pointer fixed, DLL rebuilt
- **Phase 4 (full GPU pipeline)**: COMPLETE — Bug 4 fixed, 10/10 rounds, EFB active
  - cuSPARSE SpMV 78x faster (SpMV only; round-level ~equal to CPU on small 1w data)
  - **GPU vs CPU accuracy: EXACT MATCH (77.64% on real labels, verified)**
  - GPU benefit expected to grow on larger datasets (1d/4h/1h/15m)
- **Location**: `v3.3/gpu_histogram_fork/` (isolated from main pipeline)
- **Commit**: `0a94b4e` (latest)

### LightGBM Optuna — Code Wired But Never Executed
- Optuna HPO was the difference between 67.7% (no Optuna) and 71.9% (with Optuna)
- Code is wired in run_optuna_local.py + cloud_run_optuna.py, but **optuna_configs_all.json is empty `{}`** — no trials have been run
- Re-enabling across all TFs is critical for target accuracy
- 200 TPE trials, CPCV folds, mlogloss objective

### Training Status
| TF | Status | Accuracy | Features | Notes |
|----|--------|----------|----------|-------|
| **1w** | DONE (needs Optuna) | CPCV 77.64% (GPU=CPU verified) | 2.2M | GPU histograms Phase 4 complete |
| **1d** | CROSS GEN DONE | — | 4.69M crosses (from cloud) | Inference artifacts downloaded. Needs CPCV training |
| **4h** | BASE BUILT | — | 8,794×3,904 base | Cross gen + training → CLOUD (512GB+ RAM needed) |
| **1h** | NEEDS CLOUD | — | — | Separate cloud machine, 2TB+ RAM |
| **15m** | NEEDS CLOUD | — | — | Separate cloud machine, 2TB+ RAM, user picks |

### Data Status
| Asset | Status | Notes |
|-------|--------|-------|
| **1w NPZ** | COMPLETE | 2.2M features, parquet, cross names |
| **1w model** | COMPLETE | model_1w.json (109MB), CPCV backup |
| **1d base parquet** | COMPLETE | features_BTC_1d.parquet (10.7MB, 5,733×3,796) — ready for cloud upload |
| **4h base parquet** | COMPLETE | features_BTC_4h.parquet (13.5MB, 8,794×3,904) — ready for cloud upload |
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

### All Machines Destroyed
No active cloud machines.

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

### GPU-or-Nothing Policy
- **NO CPU fallbacks anywhere.** Every compute stage runs on GPU.
- `ALLOW_CPU=1` environment variable as escape hatch for local testing ONLY
- Cloud deploys MUST use GPU — no ALLOW_CPU
- Trade optimizer: already GPU (CuPy vectorized on 3090), no changes needed
- LightGBM training: GPU histograms Phase 4 COMPLETE (Bug 4 fixed)
- Cross gen: cuSPARSE SpGEMM replacing scipy sparse matmul
- Feature build: cuDF rolling/ewm (already GPU)

### GPU Fork — Phase 4 COMPLETE
- Bug 4 (feature_hist_offsets EFB mapping) FIXED. All 10 rounds trained, EFB active.
- GPU vs CPU accuracy: exact match (77.64%). SpMV 78x faster, round-level ~equal on 1w.
- Next: integrate into ml_multi_tf.py for CPCV, deploy to cloud for larger TFs.
- See `v3.3/gpu_histogram_fork/GPU_SESSION_RESUME.md` for full details

### Revised ETAs with GPU Histograms
| TF | No Optuna | With Optuna (200 trials) |
|----|-----------|--------------------------|
| **1w** | 10 min | 2 hr |
| **1d** | 1 hr | 14 hr |
| **4h** | — | — |
| **1h** | — | — |
| **15m** | 10.5 hr | 50 hr |

---

## TRAINING PLAN & COSTS

### Per-TF Training Times — GPU Histograms (revised)

| TF | No Optuna | With Optuna (200 trials) | Notes |
|----|-----------|--------------------------|-------|
| **1w** | **10 min** | **2 hr** | Local 3090 OK |
| **1d** | **1 hr** | **14 hr** | Cloud 256GB+ RAM |
| **4h** | TBD | TBD | Cloud 512GB+ RAM |
| **1h** | TBD | TBD | Cloud 2TB+ RAM |
| **15m** | **10.5 hr** | **50 hr** | Cloud 2TB+ RAM, user picks |

Previous CPU estimates (for reference):
| Step | 1w | 1d | 4h | 1h | 15m |
|------|-----|-----|-----|------|------|
| Feature build | 5m | 15m | 30m | 1hr | 2hr |
| Cross gen | 15s | 10m | 36m | 2.3hr | 5.8hr |
| CPCV total (4 folds, CPU) | 12m | 2hr | 3.3hr | 7hr | 10hr |
| **Total no Optuna (CPU)** | **25m** | **2.5hr** | **5-10hr** | **9-15hr** | **13-19hr** |

### Machine Assignments (UPDATED — local OOM forced cloud)

| TF | Where | Machine | Notes |
|----|-------|---------|-------|
| 1w | LOCAL | 13900K + 3090 | DONE (CPCV 77.64%, needs Optuna) |
| 1d | CLOUD | Machine A (512GB+, 64c+) | Base parquet pre-built locally, upload to skip rebuild |
| 4h | CLOUD | Machine A (same, sequential) | Base parquet pre-built locally, upload to skip rebuild |
| 1h | CLOUD | Machine B (2TB+, 128c+) | Full pipeline on cloud |
| 15m | CLOUD | Machine C (2TB+, 128c+) | User picks machine |

---

## KEY RESEARCH FINDINGS

1. **v3.2 NEVER completed training** — crashed on feature_name mismatch. No baseline exists.
2. **max_bin=255** — controls EFB bundle size. max_bin=15 created 18x more bundles than needed.
3. **GPU histogram fork proved 78x SpMV speedup** — Phase 4 COMPLETE, Bug 4 fixed. GPU = CPU accuracy (77.64%). GPU-or-nothing: NO CPU fallbacks.
4. **GOSS/CEGB/quantized_grad** — all kill rare esoteric signals. REJECTED.
5. **HMM regime weights 0.15 is a thesis violation** — crushes counter-trend esoteric signals by 85%.
6. **Co-occurrence filter 8→3** — matches min_data_in_leaf=3 for 1d/1w.
7. **enable_bundle=False for 1h/15m** — EFB conflict graph intractable at 10M features.
8. **Cross features are pure 0/1** — structural zeros = feature OFF, not missing. No NaN after binarization.
9. **We are in UNCHARTED TERRITORY** — no published work combines esoteric × TA at 2-10M sparse binary features.
10. **1d cross gen OOM at 68GB locally** — completed on cloud. Inference artifacts (4.69M cross names) downloaded. Needs CPCV training.
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
- model_1w.json (109MB, 2.2M features, CPCV 77.64%, needs Optuna)
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
- Phase 1-4: ALL COMPLETE (78x SpMV speedup, Bug 4 fixed, EFB active, 77.64% accuracy = CPU match)
- Next: integrate GPU path into ml_multi_tf.py for CPCV, deploy to cloud for larger TFs
- Revised ETAs: 1w 10min, 1d 1hr, 15m 10.5hr (no Optuna); with Optuna: 1w 2hr, 1d 14hr, 15m 50hr

## GIT STATUS
Branch: v3.3
Latest commit: `0a94b4e` — v3.3: GPU session resume, cached training, _parent_ds bugfix
Uncommitted: Various modified files (see git status)
