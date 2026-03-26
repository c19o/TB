# V3.3 Session Resume — 2026-03-26 (Latest)

## STATUS: 1w training on vast.ai (512c Michigan, ID 33555611)

### Current Machine
- **ID**: 33555611 (vast.ai)
- **Specs**: 512 vCPUs, 515 GB RAM, 8x RTX 5090, Michigan
- **Image**: pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime (Docker-free, deps via pip)
- **SSH**: `ssh -i ~/.ssh/vast_key -p 35610 root@ssh6.vast.ai`
- **Cost**: $3.22/hr

### Pipeline Progress (1w)
- [x] Base features: 1,374 cols (8s)
- [x] V2 feature layers: 3,331 cols
- [x] Cross generation: 1,134,120 crosses in **~2 minutes** (was 75-90 min single-threaded!)
- [x] NPZ + Parquet saved
- [ ] XGBoost CPCV training (4 paths, 818 rows x 1.1M features) — IN PROGRESS
- [ ] Feature stability
- [ ] Optuna trade optimizer
- [ ] Meta-labeling
- [ ] LSTM
- [ ] SHAP analysis
- [ ] Download artifacts

## WHAT CHANGED THIS SESSION (MASSIVE)

### 1. Docker-Free Cloud Deployment
- **Ditched Docker entirely.** No more 12+ min image pulls.
- Use any lightweight base image (pytorch, ubuntu, etc.) — usually cached on vast.ai
- pip install all deps (~30s): `pip install xgboost lightgbm scikit-learn scipy ephem astropy pytz joblib pandas numpy pyarrow optuna hmmlearn numba tqdm pyyaml`
- SCP code tar (~11MB) + btc_prices.db (1.3GB) + other DBs
- Total setup: **2-3 min** vs 12+ min Docker pull
- Works on ANY provider, ANY machine, no CUDA/driver lock-in

### 2. Multi-Threaded Cross Generation (40x speedup!)
- **Root cause found**: `v2_cross_generator.py` element-wise multiply (`left_cols * right_cols`) was single-threaded. Comment said "multi-core via BLAS" but element-wise `*` does NOT use BLAS.
- **Fix**: ThreadPoolExecutor — numpy ufuncs release the GIL, so threads give true parallelism
- **Dynamic batch sizing**: `BATCH = min(MAX_BATCH, max(500, n_pairs // n_threads))` — ensures enough batches to saturate all threads
- **Result**: Astro × TA (131K pairs) went from 12+ min → 4 seconds. Total cross gen from 75-90 min → 2 min.
- **RAM reduction**: 375GB → 62GB (threaded COO assembly vs dense array accumulation)

### 3. Missing Module Fix
- `astrology_engine.py` was in project root, not v3.2_2.9M_Features/. Now copied in.

### 4. Pip Dependency Checklist
- Missing `pandas`, `pyarrow`, `numba` caused 3 separate crashes before caught
- Full dep list now documented. MANDATORY import test before running.

## FILES MODIFIED
- `v3.2_2.9M_Features/v2_cross_generator.py` — ThreadPoolExecutor parallel cross gen + dynamic batch sizing
- `v3.2_2.9M_Features/gpu_cross_builder.py` — ThreadPoolExecutor for DOY CPU fallback
- `v3.2_2.9M_Features/astrology_engine.py` — copied from root (was missing)
- `v3.2_2.9M_Features/CLAUDE.md` — updated with all new lessons, deployment protocol, batch sizing rule
- `v3.3/CLOUD_TRAINING_PROTOCOL.md` — Docker-free deployment, parallelism docs
- `v3.3/setup.sh` — new (lightweight setup script for Docker-free deploy)

## EXPECTED TIMES (REVISED with parallel cross gen)

| TF | Cross Gen | Training | Total | Cost (@$3.22/hr) |
|----|-----------|----------|-------|-------------------|
| 1w | ~2 min | ~5-10 min | ~15 min | $0.80 |
| 1d | ~5 min | ~15 min | ~25 min | $1.34 |
| 4h | ~4 min | ~10 min | ~20 min | $1.07 |
| 1h | ~15 min | ~30 min | ~50 min | $2.68 |
| 15m | ~30 min | ~2 hrs | ~3 hrs | $9.66 |

## NEXT STEPS
1. Wait for 1w training to complete
2. Download all 1w artifacts
3. Run 1d on same machine (or bigger if needed)
4. Commit all changes to git (branch v3.3)
5. After all TFs: paper trading validation

## KEY LESSONS (NEW)
1. Docker is unnecessary for CPU-bound pipelines. pip + SCP is 5x faster setup.
2. numpy element-wise `*` is NOT BLAS — single-threaded. Use ThreadPoolExecutor for parallel column blocks.
3. Batch size must match thread count, not just RAM. `n_pairs / 50000 = 3 batches` wastes 509 of 512 cores.
4. Always test ALL pip imports before launching pipeline. One missing module = wasted run.
5. `kill -9` on vast.ai containers kills the init process — use `kill PID` not `killall`.

## GIT: Branch v3.3, uncommitted changes — commit after 1w completes
