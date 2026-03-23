# Session Resume — V3.0 (LGBM) — 2026-03-23

## STATUS: DEPLOYING TO CLOUD
Code complete, smoke test 27/27 passed. Deploying to Minnesota.

## ACTIVE CLOUD INSTANCE
- **vast.ai ID: 33366593** — Minnesota, US
- 8x Tesla V100-SXM2-32GB (256GB total VRAM)
- 80 CPU cores, 755GB RAM
- $1.58/hr
- Docker: pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel
- SSH: `ssh -p 16592 root@ssh8.vast.ai`
- Need to pip install: cudf-cu12, lightgbm, optuna, scipy, scikit-learn, hmmlearn, psutil, numba

## WHAT TO DO NEXT SESSION
1. **If instance still running**: SSH in, check progress
2. **If instance dead**: Re-rent (search 700GB+ RAM, avoid Belgium machines 49722/48486/30443)
3. **After training completes**: Download artifacts, compare V2 vs V3
4. **Paper trade**: Start live_trader.py in paper mode with V3 models

## GIT
- Branch: `v3-lgbm` (10 commits, all pushed)
- Remote: `https://github.com/c19o/TB.git`

## V3 CHANGES FROM V2
1. **XGBoost → LightGBM**: EFB for sparse, leaf-wise growth, CPU-only (CUDA doesn't support sparse), `force_col_wise=True`, `max_bin=15`
2. **30M Exhaustive → Optuna 200 TPE trials**: Sortino objective, `n_jobs=4`
3. **min_data_in_leaf=3** (1d/1w) + **min_gain_to_split=2.0**: Rare signals now usable
4. **Co-occurrence filter MIN_CO_OCCURRENCE=8**: Math constraint on CPCV splits
5. **Direct COO extraction from GPU tensor**: Never accumulates dense columns (prevents OOM on 15m/5m)
6. **Incremental CSR per cross type**: hstack small chunks instead of giant COO→CSR
7. **Eliminated tolil()/tocsr() in CPCV fold loop**: HMM overlay as separate dense array
8. **CPCV workers capped at min(cores/2, 15)**: Prevents thread oversubscription
9. **num_threads per worker = cores/workers**: Fair share
10. **LSTM DataLoader workers raised to 32**, Optuna n_jobs=4

## PIPELINE PHASES
| Phase | Time Est | Notes |
|-------|----------|-------|
| Feature builds (1d/1w/4h) | ~1.5 hrs | 31 daily + 14 crypto, parallel on 8 GPUs |
| Feature builds (1h) | ~1 hr | 14 crypto, parallel on 8 GPUs |
| Feature builds (15m BTC) | ~2 hrs | Single asset, V100 batch ~80 (32GB VRAM) |
| Feature builds (5m BTC) | ~4 hrs | Single asset, biggest bottleneck |
| LightGBM training | ~30 min | 80 cores, force_col_wise |
| Optuna (200 trials × 6 TFs) | ~10 min | CuPy GPU, n_jobs=4 |
| Validation + Meta | ~5 min | CPU |
| LSTM (3 rounds on 8 GPUs) | ~15 min | DataParallel per TF |
| Audit | ~2 min | Fast |
| **Total** | **~9-10 hrs** | **~$15** |

## PHILOSOPHY (NON-NEGOTIABLE)
- NO FILTERING. LightGBM decides via tree splits, not us
- NO FALLBACKS. One pipeline for all. If it breaks, fix it
- Esoteric signals ARE the edge. Never regularize them away
- NaN = missing. NEVER convert to 0
- Sparse = the edge, not noise
- Cross features justified: tree models can't find 3-way interactions at colsample=0.15

## ARTIFACTS TO DOWNLOAD
- `*_model_*.json` — LightGBM models
- `oos_predictions_*.pkl` — OOS predictions with IS metrics
- `optuna_configs_*.json` — optimizer configs
- `meta_model_*.pkl` — meta-labeling models
- `lstm_*.pt` + `blend_config_*.json` — LSTM + blend weights
- `validation_report_*.json` — PBO/DSR reports

## LESSONS LEARNED (vast.ai)
- Belgium machines (IDs 49722, 48486, 30443, 49822) have broken SSH authorized_keys permissions
- Australia machines had port forwarding failures
- Texas machine went offline mid-rent
- RAPIDS Docker image `rapidsai/base:25.12-cuda12-py3.11` is ~15GB, slow to pull
- PyTorch image `pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel` is smaller, faster pull
- Always use `cuda12` not `cuda12.5` in RAPIDS tags (major version only)
- Need to pip install cudf-cu12 + lightgbm + optuna on PyTorch image
