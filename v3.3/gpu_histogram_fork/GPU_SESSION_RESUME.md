# V3.3 GPU Histogram Fork — Session Resume

## INSTRUCTION TO NEW SESSION
Read this file completely, then read `v3.3/MASTER_FIX_PLAN.md` and `docs/BUG_REGISTRY.md`.
The pipeline has been through a MASSIVE audit and fix session (44 agents, $46, 30+ bugs fixed).
ALL previous training results are INVALID — must retrain everything with fixed pipeline.

## ACTIVE MACHINES (2026-03-30 ~04:00 UTC)

**NO ACTIVE MACHINES**

**DESTROYED**: Norway, NJ, Belgium (m:33799816), Poland (m:33802014)
**Total cost: $0/hr**

## TRAINING STATUS — ALL RESULTS SUSPECT/INVALID

| TF | Status | Accuracy | Notes |
|----|--------|----------|-------|
| 1w | **NEEDS RETRAIN** | OLD: 57.9% CPCV (INVALID — purge=6 leak, whitelist filter, lambda=100 killing signals) | Now has 621+ features (whitelist removed), purge=50, all params fixed |
| 1d | **NEEDS RETRAIN** | — | Artifacts ready (NPZ, parquet). Never successfully trained with correct params. |
| 4h | **NEEDS RETRAIN** | — | Artifacts ready (NPZ 1.67GB, parquet). cuda_sparse .so loading needs fix for cloud. |
| 1h | **NEEDS CROSS GEN + TRAIN** | — | Need machine with 128GB+ RAM for cross gen + training |
| 15m | **NEEDS CROSS GEN + TRAIN** | — | Need machine with 128GB+ RAM. Cloud only (100GB training RAM). |

## WHY ALL RESULTS ARE INVALID

The 2026-03-30 audit session found **12+ mechanisms** that were silently killing rare esoteric signals:

1. **CPCV purge=6** when max_hold_bars=50 → label leakage inflated all OOS scores
2. **feature_fraction=0.05** in GPU fork → 95% of EFB bundles skipped per tree
3. **min_data_in_leaf=30-50** → rare signals (10-20 fires) could NEVER form a leaf
4. **lambda_l1 up to 100** → zeroed leaf weights for ANY signal firing ≤33 times
5. **Class weight np.pad misalignment** → SHORT 3x upweighting on wrong rows
6. **bagging_fraction=0.8** → 10-fire signals only in 11% of trees (P=0.8^10)
7. **path_smooth=2.0** → dampened rare leaf magnitudes 21-35%
8. **HMM global fit in parallel CPCV** → future data leaking into early folds
9. **is_unbalance=True in Optuna** vs explicit 3x weights in final → gradient mismatch
10. **TF_FEATURE_WHITELIST for 1w** → dropped DOY, gematria flags, _EXTREME bins (human override)
11. **Live inference: regime=None** → tens of thousands of DOY crosses always zero
12. **Live inference: HMM using sma_5** → wrong distribution vs training

**Combined effect**: The model was training fast because it was learning NOTHING from the matrix.

## ALL FIXES APPLIED (3 commits on v3.3 branch)

### Commit 15bdb43 — Wave 1 (17 bugs)
- CPCV purge = max_hold_bars per TF
- Class weight alignment (fold-level, no np.pad)
- Uniqueness +1 in run_optuna_local.py
- feature_pre_filter=False on ALL lgb.Dataset() calls
- feature_fraction=0.9 (config + GPU fork + all hardcoded instances)
- min_data_in_leaf capped at 8-10 per TF
- device='cpu' leak fixed in GPU path
- is_enable_sparse=True always
- model_to_string removed → best_iter tracking
- n_jobs: cores//96 → cores//16 (6-30x Optuna speedup)
- OMP/NUMBA threads for binarization
- Astrology 11 bare excepts → NaN + logging
- subsample=0.8 removed, colsample=0.01 → 0.9
- Lambda L1 capped [1e-4, 4.0], L2 [1e-4, 10.0]
- HMM per-fold fits in parallel CPCV
- is_unbalance removed from Optuna (explicit weights)
- validate.py expanded to 82 checks

### Commit 1f039f8 — Wave 2+3
- sparse-dot-mkl for cross gen matmul (20-50x multi-core)
- Sparse CSR predict in live_trader (eliminated 2.9M for-loops)
- Regime computed before crosses in inference
- HMM uses actual prev_close not sma_5
- Combo context formulas persisted for inference
- NaN propagation in inference binarization
- bagging_fraction raised to 0.95
- path_smooth reduced to 0.5
- TF_FEATURE_WHITELIST REMOVED (1w now gets all 621+ features)
- force_col_wise removed for 15m (20-30% speed gain)
- NPZ compressed=False
- CPCV per-fold checkpoints (O(1) vs O(n²))
- Numba thread formula fix for local
- os.cpu_count → cgroup-aware
- MKL thread cap lifted
- numactl --interleave=all for 4+ NUMA
- SharedMemory for CPCV IPC
- All 7 streamers: WAL mode + DR/gematria enrichment
- date_numerology in news_streamer
- Space weather DR + flare gematria
- v2_easy_streamers daemon loop
- Macro sub-1.0 DR fix + ticker gematria
- Tweet gematria: consistent (no ASCII strip)
- docs/ folder created, BUG_REGISTRY.md, MODEL_STATUS.md
- 73GB cleanup (deleted v3.0, v3.1, v3.2, v2, heartbeat_data, discord archives)

## SPECIALIST FINDINGS (Not Yet Implemented — Future Work)

### CUDA GPU Kernel Optimizations (3-5x per-fold speedup available)
- 63,000 H2D transfers/run → batch to 1,000 (per-leaf gradient upload)
- 10-40% warp efficiency → warp-cooperative atomic kernel
- Python for-loop launches 48 GPU kernels → vectorize to 2
- CSR+CSR.T dual storage → CUSPARSE_OPERATION_TRANSPOSE for 15m
- All matrix-safe. Proprietary fork changes needed.

### LightGBM Internals Confirmed
- EFB is SAFE — bin offsets preserve rare signal identity
- Sparse histogram explicitly visits all non-zeros (O(2×NNZ))
- 15-fire signal produces gain ≈ 20 (well above threshold 2.0)
- With fixed params, rare signals CAN be learned

## PIPELINE TIMING ESTIMATES (Post-Fix, 8x RTX 5090 + EPYC 384c)

Pipeline execution order per TF:
1. Data Load (parquet + NPZ)
2. Cross Gen (if NPZ not cached — sparse-dot-mkl)
3. EFB Dataset Construction
4. Phase 1 Optuna Search (25-30 trials, 8 parallel GPU workers)
5. Phase 2 Optuna Validation (top 3, 4-fold CPCV)
6. Final Retrain (all data, 800 rounds)
7. CPCV Full (15 folds, confidence calibration)
8. Meta-labeling
9. Exhaustive Optimizer (trade strategy params)
10. PBO + Audit

| TF | Cross Gen | EFB Build | Optuna (P1+P2) | Final Retrain | CPCV Full | Optimizer | **TOTAL** |
|----|-----------|-----------|---------------|---------------|-----------|-----------|-----------|
| 1w | cached | 18s | 18min | 40min | 1.3hr | 30min | **~3hr** |
| 1d | cached | 30s | 1hr | 2hr | 4.1hr | 1hr | **~8.5hr** |
| 4h | cached | 4min | 2.6hr | 5.3hr | 10.7hr | 2hr | **~21hr** |
| 1h | 15min | 9min | 3.7hr | 12.4hr | 24.9hr | 3hr | **~45hr** |
| 15m | 15min | 31min | 6.4hr | 25hr | 50hr | 5hr | **~87hr** |
| **ALL 5 sequential** | | | | | | | **~165hr (6.9 days)** |

With CUDA kernel optimizations (not yet implemented): 15m drops to ~25-35hr.

## LOCAL TRAINING (13900K + RTX 3090 + 64GB RAM)

| TF | Can Train? | RAM | VRAM | Time |
|----|-----------|-----|------|------|
| 1w | YES | ~7.1GB | ~0.2GB | ~30-40min (Optuna) |
| 1d | YES | ~1.6GB | ~0.85GB | ~8hr |
| 4h | YES | ~10.8GB | ~4.5GB | ~21hr (GPU) |
| 1h | YES (tight) | ~38GB | ~15.3GB | Very long (single GPU) |
| 15m | NO | ~100GB | N/A | Cloud only |

## GPU FORK STATUS
- cuda_sparse fork BUILT locally (RTX 3090, sm_86)
- .so tested on A40 (sm_89) and RTX 5090 (sm_120 via PTX)
- **CLOUD LOADING BUG**: Belgium crashed because .so wasn't swapped into site-packages correctly
- Fix needed: reliable .so loading mechanism for cloud deploys

## NEXT STEPS (Priority Order)
1. **Retrain 1w locally** — 30-40min, test the full fixed pipeline end-to-end
2. **Compare results** — new 1w (621+ features, purge=50, all fixes) vs old (141 features, purge=6, signal-killing params)
3. **If 1w looks good**: retrain 1d locally (~8hr)
4. **Rent cloud machine** for 4h (needs GPU for reasonable speed)
5. **Fix cuda_sparse .so loading** for cloud before deploying GPU training
6. **Implement CUDA kernel optimizations** (3-5x speedup) for 1h/15m viability
7. **1h + 15m on cloud** — need 128GB+ RAM machine

## KEY FILES
- `v3.3/MASTER_FIX_PLAN.md` — full fix plan with Perplexity validations
- `docs/BUG_REGISTRY.md` — all 30+ bugs catalogued
- `docs/ACTIVE/MODEL_STATUS.md` — per-TF status tracker
- `docs/ACTIVE/DEPLOY_CHECKLIST.md` — cloud deployment protocol
- `v3.3/validate.py` — 82 pre-flight checks (run before ANY training)
- `v3.3/CLAUDE.md` — project rules + audit pipeline
