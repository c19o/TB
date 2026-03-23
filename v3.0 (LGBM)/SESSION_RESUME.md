# Session Resume — V3.0 (LGBM) — 2026-03-22

## STATUS: V3 CODE CONVERSION IN PROGRESS
All v2 code copied to `v3.0 (LGBM)/`. XGBoost → LightGBM conversion ~90% complete.
3 audit passes required before deploying to cloud. Smoke test written, not yet run.

## WHAT CHANGED FROM V2 (Backed by 12+ Perplexity Research Queries)

### 1. XGBoost → LightGBM
- **Why:** LightGBM has Exclusive Feature Bundling (auto-bundles sparse binary crosses), leaf-wise growth naturally seeks rare high-gain splits, ~40% faster on sparse data
- **Critical:** LightGBM CUDA does NOT support sparse features. CPU with `force_col_wise=True` is faster than GPU
- **Config:** `device="cpu"`, `force_col_wise=True`, `max_bin=15`, `num_threads=-1` (auto-detect)

### 2. 30M Exhaustive Grid → Optuna 200 TPE Trials
- **Why:** 30M combinations = guaranteed noise memorization per Lopez de Prado's own PBO paper. Optuna TPE finds same parameter plateaus with 200 intelligent trials
- **Config:** `TPESampler(n_startup_trials=25, multivariate=True)`, Sortino objective, DD penalty
- **Output:** `optuna_configs_{tf}.json` (same format as old exhaustive_configs)

### 3. Rare Signal Friendly Hyperparameters
- **Why:** Old `min_child_weight=10-50` silently killed rare astro crosses (fire 10-20x on daily). XGBoost literally refused to create those leaves
- **Fix:** `min_data_in_leaf=3` (1d/1w) with `min_gain_to_split=2.0` as compensating guard. Blocks low-gain noise splits while letting high-gain rare signals through
- **Per-TF:** 1d/1w=3, 4h=5, 1h=8, 15m/5m=15

### 4. Cross Co-occurrence Filter (MIN_CO_OCCURRENCE=8)
- **Why:** Crosses firing <8 times can't reliably appear in both CPCV train AND validation splits (98.3% coverage at n=8 with 5-fold CPCV). Dead weight in the model.
- **This is a MATH constraint, not a signal filter.** XGBoost/LightGBM couldn't use them anyway.
- **Expected:** 2M crosses → ~200K-500K per asset. Still massive, but all usable.

### 5. Features UNCHANGED
- Same feature_library.py, same build scripts, same cross generator logic
- Parquets + .npz from v2 are 100% compatible with v3 training
- No rebuild needed — just swap training code

## WHAT WAS VALIDATED AS NOT OVER-ENGINEERED
- CPCV + PBO + DSR + meta-labeling (Lopez de Prado's intended integrated system)
- No feature filtering before model (academically validated)
- 150K base features with sparse CSR (manageable with colsample cascade)
- Cross features for conditional context (tree models can't find 3-way interactions natively)
- Esoteric signals as base features (the edge, let the model decide)
- LSTM blend (genuine architectural diversity)

## FILES CONVERTED (v3.0 directory)
| File | Status | What Changed |
|------|--------|-------------|
| v2_multi_asset_trainer.py | DONE | XGBoost→LightGBM, Dask removed, CPU-only |
| ml_multi_tf.py | DONE | All DMatrix→Dataset, callbacks, 168 lines removed |
| exhaustive_optimizer.py | DONE | 30M grid→Optuna 200 TPE trials, Sortino objective |
| meta_labeling.py | DONE | xgb_shallow→lgbm_shallow |
| v2_cloud_runner.py | DONE | Updated Phase 2/3 calls, optuna_configs refs |
| v2_cross_generator.py | DONE | MIN_CO_OCCURRENCE=8 filter added |
| config.py | DONE | V3_LGBM_PARAMS, TF_MIN_DATA_IN_LEAF, Optuna config |
| live_trader.py | DONE | LightGBM Booster, optuna_configs fallback |
| backtesting_audit.py | IN PROGRESS | Agent converting |
| portfolio_aggregator.py | IN PROGRESS | Agent converting |
| smoke_test_pipeline.py | IN PROGRESS | Agent converting |
| feature_classifier.py | DONE | LightGBM conversion |
| leakage_check.py | DONE | LightGBM conversion |
| smoke_test_v3.py | NEW | 6-step plumbing test (~3-5 min) |
| CLAUDE.md | DONE | Updated rules for LightGBM |
| feature_library.py | DONE | Comment updates only |
| data_access_v2.py | DONE | Comment updates only |
| llm_features.py | DONE | Comment updates only |

## CLOUD DEPLOYMENT PLAN

### Target Machine
- **vast.ai ID: 30555152**
- 8x RTX 5090 (256GB total VRAM)
- AMD EPYC 9654 96-Core (384 effective cores)
- 503GB RAM, $2.99/hr
- Docker: `rapidsai/base:25.02-cuda12.5-py3.12`

### Pipeline on Beast (full build + train)
| Phase | Time Est | How |
|-------|----------|-----|
| Upload v3 scripts + DBs | ~5 min | rsync ~2.9GB |
| Feature builds (all TFs, all assets) | ~35 min | 8 GPUs cuDF, 384 cores, 503GB RAM |
| LightGBM training (all TFs) | ~8 min | 384 cores, force_col_wise, CPU mode |
| Optuna (200 trials per TF) | ~5 min | 384 cores parallel |
| Validation + Meta-label | ~2 min | Trivial |
| LSTM (all TFs) | ~10 min | 8x RTX 5090 DataParallel |
| Audit | ~1 min | Fast |
| **Total** | **~70 min** | **~$3.50** |

## BEFORE DEPLOYING — CHECKLIST
- [ ] Audit pass 1: No XGBoost references remain (DONE — fixing remaining files)
- [ ] Audit pass 2: Cross-file consistency (artifact names, model extensions, config keys)
- [ ] Audit pass 3: Smoke test passes locally (`python smoke_test_v3.py --quick`)
- [ ] 3 consecutive clean audit passes
- [ ] Rent beast (ID 30555152)
- [ ] Deploy via rsync
- [ ] Monitor via `monitor_cloud.sh`

## ARTIFACTS TO DOWNLOAD AFTER CLOUD RUN
- `*_model_*.txt` — trained LightGBM models
- `oos_predictions_*.pkl` — OOS predictions with IS metrics
- `optuna_configs_*.json` — optimizer configs
- `meta_model_*.pkl` — meta-labeling models
- `lstm_*.pt` + `blend_config_*.json` — LSTM models + blend weights
- `validation_report_*.json` — PBO/DSR reports
- `backtesting_audit_report.json` — final audit

## KEY RESEARCH FINDINGS (2026-03-22 Perplexity Session)

### Cross Features Are Justified
- Tree models CANNOT find 3-way interactions at colsample=0.15 (only 0.34% chance per tree)
- Pre-computed crosses collapse discovery from depth-3 tree path to single split
- BUT: crosses firing <8 times are killed by min_data_in_leaf anyway → co-occurrence filter

### LightGBM > XGBoost for This Pipeline
- EFB auto-bundles mutually exclusive sparse crosses (free compression)
- Leaf-wise growth actively seeks high-gain rare splits
- ~40% faster training on high-dimensional sparse data
- GPU useless for sparse — CPU with many cores wins

### 30M Grid Search = Overfitting
- Lopez de Prado's own PBO paper: exhaustive search on random data produces Sharpe >1.5 in-sample
- Optuna TPE with 200 trials finds same plateaus without noise memorization

### Esoteric Signals Don't Decay Like Conventional Alpha
- Planetary cycles, gematria, numerological dates are eternal — nobody is arbitraging them
- Decay risk is regime shift (market structure change), not signal crowding
- Retraining scheduler + self-learner handle regime adaptation

## V2 COMPARISON
- V2 pipeline running on Croatia 4x4090 (ID 33350833) — XGBoost, exhaustive optimizer
- V3 will run on beast (ID 30555152) — LightGBM, Optuna
- Compare: OOS accuracy, PBO lambda, DSR p-value, Sharpe, max DD
- If V3 matches or beats V2 → V3 is the production pipeline
