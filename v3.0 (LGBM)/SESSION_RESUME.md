# Session Resume — V3.0 (LGBM) — 2026-03-22

## STATUS: READY FOR SMOKE TEST → CLOUD DEPLOY
All code converted. 7 commits on `v3-lgbm` branch, pushed to GitHub.
Need: smoke test locally → fix loop until clean → deploy to Belgium.

## WHAT TO DO NEXT SESSION
1. **Run smoke test**: `cd "v3.0 (LGBM)" && python smoke_test_v3.py --quick` (~3-5 min on 13900K)
2. **Fix any failures**, re-run until clean
3. **Rent Belgium machine**: `vastai create instance 30818925 --image rapidsai/base:25.02-cuda12.5-py3.12 --disk 200`
   - 4x A40 (180GB VRAM), EPYC 7662 (256 cores), 2TB RAM, $1.15/hr
4. **Deploy**: rsync v3.0 scripts + databases (~2.9GB)
5. **Run full pipeline**: `PYTHONUNBUFFERED=1 python v2_cloud_runner.py --engine lightgbm --dph 1.15 2>&1 | tee train.log`
6. **Monitor**: `tail -f train.log` or `bash monitor_cloud.sh`
7. **Download artifacts** when done (~$1.50 total)
8. **Compare V2 vs V3**: OOS accuracy, PBO, DSR, Sharpe, max DD

## GIT
- Branch: `v3-lgbm` (7 commits, all pushed)
- Remote: `https://github.com/c19o/TB.git`

## CLOUD TARGET
- **vast.ai ID: 30818925** — Belgium
- 4x A40 45GB (180GB total VRAM)
- AMD EPYC 7662 64-Core (256 effective cores)
- 2TB RAM (5m builds need 250GB — massive headroom)
- $1.15/hr → ~$1.50 for full pipeline
- Docker: `rapidsai/base:25.02-cuda12.5-py3.12`

## V3 CHANGES FROM V2 (7 commits)

### Core Changes
1. **XGBoost → LightGBM**: EFB for sparse, leaf-wise growth, ~40% faster. CPU-only (CUDA doesn't support sparse). `force_col_wise=True`, `max_bin=15`, `num_threads=-1`
2. **30M Exhaustive → Optuna 200 TPE trials**: 30M = noise per PBO paper. Sortino objective, DD penalty, `n_jobs=4`
3. **min_data_in_leaf=3** (1d/1w) + **min_gain_to_split=2.0**: Rare astro crosses now usable (were killed by min_child_weight=10-50)
4. **Co-occurrence filter MIN_CO_OCCURRENCE=8**: Drops crosses firing <8 times (can't appear in both CPCV splits). Math constraint, not signal filter.

### Performance Fixes
5. **Incremental CSR per cross type**: Was single-threaded giant COO→CSR (2-6 hrs on 15m/5m). Now hstack of ~13 small CSR chunks.
6. **Eliminated tolil()/tocsr() in CPCV fold loop**: HMM overlay as separate dense (N,4) array. Saves 150-450s per TF.
7. **CPCV workers capped at min(cores/2, 15)**: Prevents 73K thread oversubscription on 384-core machine. `V3_CPCV_WORKERS` env var for override.
8. **num_threads per worker = cores/workers**: Fair share, no contention.
9. **LSTM DataLoader workers raised to 32**: Was capped at 8.
10. **Optuna n_jobs=4**: Parallel trial evaluation.

### Conversion (all XGBoost removed)
- v2_multi_asset_trainer.py, ml_multi_tf.py, exhaustive_optimizer.py, meta_labeling.py, live_trader.py, backtesting_audit.py, portfolio_aggregator.py, smoke_test_pipeline.py, feature_classifier.py, leakage_check.py, build_4h_features.py, ml_mega_optimizer.py, cloud_runner.py, runpod_train.py, runpod_optimize.py, v2_cloud_runner.py, config.py, CLAUDE.md
- All `exhaustive_configs` → `optuna_configs`
- All model saves as `.json` (LightGBM native)
- 3-class prediction handling fixed in portfolio_aggregator.py
- live_trader.py: optuna_configs fallback, missing config guard

## PIPELINE ON BELGIUM (skipping feature builds = use v2's)
| Phase | Time Est | How |
|-------|----------|-----|
| Feature builds (all TFs) | ~60 min | 4 GPUs cuDF, 256 cores, 2TB RAM |
| LightGBM training | ~10 min | 256 cores, force_col_wise, CPU |
| Optuna (200 trials × 6 TFs) | ~5 min | CuPy GPU + n_jobs=4 |
| Validation + Meta-label | ~2 min | CPU |
| LSTM (6 TFs, 2 rounds) | ~20 min | 4x A40, DataParallel |
| Audit | ~1 min | CPU |
| **Total** | **~100 min** | **~$1.90** |

## LightGBM Config
```python
V3_LGBM_PARAMS = {
    "objective": "multiclass", "num_class": 3,
    "metric": "multi_logloss", "boosting_type": "gbdt",
    "device": "cpu", "force_col_wise": True,
    "max_bin": 15, "num_threads": -1,
    "is_enable_sparse": True,
    "min_data_in_leaf": 3,  # per-TF: 1d/1w=3, 4h=5, 1h=8, 15m/5m=15
    "min_gain_to_split": 2.0,
    "lambda_l1": 0.5, "lambda_l2": 3.0,
    "feature_fraction": 0.10, "feature_fraction_bynode": 0.5,
    "bagging_fraction": 0.8, "bagging_freq": 1,
    "num_leaves": 63, "learning_rate": 0.03,
    "verbosity": -1,
}
```

## PHILOSOPHY (NON-NEGOTIABLE)
- The matrix is UNIVERSAL — same sky, same calendar, same energy for ALL assets
- NO FILTERING. LightGBM decides via tree splits, not us
- NO FALLBACKS. One pipeline for all. If it breaks, fix it
- Esoteric signals ARE the edge. Never regularize them away
- More diverse signals = stronger predictions. The edge is the matrix
- Sparse = the edge, not noise. Co-occurrence filter of 8 is a MATH constraint, not signal filtering
- NaN = missing (LightGBM handles natively). NEVER convert to 0.
- Cross features justified: tree models can't find 3-way interactions at colsample=0.15 (0.34% chance per tree)

## KEY RESEARCH FINDINGS (2026-03-22 Perplexity)
- CPCV+PBO+DSR+meta-labeling = Lopez de Prado's intended system (NOT over-engineered)
- LightGBM CUDA doesn't support sparse → CPU with force_col_wise wins
- 30M grid search = guaranteed noise per PBO paper → Optuna 200 trials
- min_data_in_leaf=3 + min_gain_to_split=2.0 = rare signals usable (compensating guard)
- n=8 co-occurrence threshold gives 98.3% CPCV coverage (timeframe-independent)
- Esoteric signals don't decay like conventional alpha (nobody arbitraging moon phases)

## ARTIFACTS TO DOWNLOAD
- `*_model_*.json` — LightGBM models
- `oos_predictions_*.pkl` — OOS predictions with IS metrics
- `optuna_configs_*.json` — optimizer configs
- `meta_model_*.pkl` — meta-labeling models
- `lstm_*.pt` + `blend_config_*.json` — LSTM + blend weights
- `validation_report_*.json` — PBO/DSR reports

## V2 COMPARISON
- V2 running on Croatia 4x4090 (ID 33350833) — XGBoost, exhaustive optimizer
- V3 will run on Belgium 4xA40 (ID 30818925) — LightGBM, Optuna
- Compare side-by-side after both complete
