# Finishing v3.2 2.9M Feature Training — Plan

## CURRENT STATE (2026-03-24 ~07:00 CST)

### Completed & Downloaded
| TF | Model | LSTM | Meta | PBO | NPZ | Optuna | Dir |
|---|---|---|---|---|---|---|---|
| 1w | ✅ 656K SPARSE | ✅ 55.3% | ✅ AUC=0.670 | REJECT (818 rows) | ❌ rebuild 30s | ❌ deferred | 1w_norway/ |
| 1d | ✅ 2.87M SPARSE | ✅ 7MB | ✅ | **DEPLOY (0.00)** | ✅ 292MB | ❌ deferred | 1d_final/ |
| 4h | ✅ 3.2M SPARSE | ✅ 17MB 53.5% | ✅ AUC=0.616 | REJECT (4380 rows) | ✅ 125MB | ❌ deferred | 4h_final/ |
| 1h | ✅ 4.25M SPARSE | ✅ 27MB 53.9% | ✅ AUC=0.648 | **DEPLOY (0.14)** | ✅ 1.6GB | ❌ deferred | 1h_final/ |
| 15m | ⏳ training on Texas | ❌ | ❌ | ❌ | on Texas | ❌ | — |

### Texas Machine (33446502)
- 256c, 3.7GHz, 1TB RAM, $1.33/hr
- SSH: ssh -p 16502 root@ssh6.vast.ai
- 15m cross gen DONE (2,282,328 crosses, took 2+ hours)
- 15m training: SPARSE single-threaded (int32 safe per Perplexity — LightGBM v4.6 uses int64)
- DON'T KILL — cross gen is not salvageable if lost

## WHAT'S LEFT

### Phase 1: Let Texas finish 15m (ETA ~3-4 more hours)
- Training (sparse, single-threaded) → ~2-3 hrs
- Meta-labeling → ~1 min
- LSTM → ~5 min
- PBO + Audit → ~2 min
- Download ALL artifacts including NPZ before killing

### Phase 2: Run Optuna on all 5 TFs (separate step)
Per Perplexity: Optuna is fully stateless and decoupled. Needs only parquet + NPZ.

**Option A: Run Optuna locally** (free, slow)
```bash
cd v3.2_2.9M_Features
# Copy NPZ from cloud results
cp ../v32_cloud_results/1d_final/v2_crosses_BTC_1d.npz .
cp "../v3.0 (LGBM)/features_BTC_1d.parquet" .
cp ../btc_prices.db .
python -u run_optuna_local.py --tf 1d
```
- 1w: ~15 min (small)
- 1d/4h: ~2-4 hrs each (single-threaded sparse or dense if 64GB+ RAM)
- 1h: ~4-6 hrs
- 15m: ~8-12 hrs

**Option B: Run Optuna on cloud** (fast, costs $)
- Rent 5 machines, upload parquet + NPZ + code (~200MB-1.7GB per TF)
- Each machine runs ONLY Optuna (no cross gen, no training)
- Uses dense conversion for multi-core (except 15m which stays sparse)
- ETA: 1-2 hrs per TF on 384c machines

**Option C: Skip Optuna for now, run later**
- The models are already trained with well-tuned defaults
- 1d and 1h already pass PBO — production ready
- Optuna can improve but isn't blocking

### Phase 3: Rerun any TFs that need improvement
- 1w PBO=0.50 (REJECT) — expected, only 818 rows
- 4h PBO=0.40 (REJECT) — might improve with Optuna regularization tuning
- 15m — verify after training completes

## MACHINE SPECS FOR PARALLEL FRESH RUNS

If doing a complete fresh training run (all steps including Optuna):

| TF | Training | Dense RAM | Optuna Time (384c) | Best Machine |
|---|---|---|---|---|
| 1w | Dense (2GB) | 32GB+ | ~15 min | Any 128c+ |
| 1d | Dense (66GB) | 128GB+ | ~1.5 hrs | 256c, 252GB |
| 4h | Dense (56GB) | 128GB+ | ~1.5 hrs | 256c, 252GB |
| 1h | Dense (300GB) | 512GB+ | ~3 hrs | 384c, 755GB |
| 15m | SPARSE only | 64GB+ | ~6 hrs (single-thread) | 384c, 503GB+ |

**15m cannot go dense** — 227K × 2.3M × 4 = 1.89 TB. No machine has that.

### 15m MULTI-CORE SOLUTION: Feature Chunking (per Perplexity)

Split 2.3M features into 4 chunks of ~570K each. Train 4 separate LightGBM models, ensemble predictions.

**Why this works:**
- Each chunk: 227K × 570K × 4 = 518GB dense → fits in 755GB+ machine
- Each chunk NNZ ≈ 970M → safely under int32 limit
- Each chunk trains multi-core with dense conversion
- Ensemble of 4 models often BETTER than 1 model (less overfit per chunk)
- Preserves ALL rows AND ALL features — no data loss, no filtering

**Implementation:**
```python
chunk_size = 570_000
for start in range(0, n_features, chunk_size):
    X_chunk = X_all[:, start:start+chunk_size].toarray()  # Dense per chunk
    model = lgb.train(params, lgb.Dataset(X_chunk, label=y))
    models.append(model)
# Ensemble: average 3-class probabilities
```

**Additional speedups (stack these):**
- `max_bin=63` (not 255) — 4x faster histogram construction
- `data_sample_strategy="goss"` — focuses on high-gradient samples
- `num_leaves=63, max_depth=7` — prevents overfit on sparse structure
- `enable_bundle=true` (default) — EFB compresses mutually exclusive features

## KEY PERPLEXITY FINDINGS

1. **LightGBM v4.6 int64 sparse is SAFE** — scipy auto-promotes to int64, LightGBM respects it. No silent corruption. GPU prediction has a separate bug (#7101) — CPU only.

2. **Optuna is fully decoupled** — parquet + NPZ is all it needs. Run anywhere, anytime. Persist to SQLite for pause/resume.

3. **Sparse vs dense = mathematically identical results** — `zero_as_missing=false` (default) treats sparse zeros as value 0. Only difference is speed (dense = multi-core, sparse = single-core).

4. **Critical Optuna params for 2.9M features:** feature_fraction (0.05-0.2), min_child_samples, lambda_l1/l2, num_leaves, min_gain_to_split. These prevent overfitting with extreme feature:row ratios.

## CRITICAL FIXES IN LOCAL CODE (all applied)

1. --symbol BTC (not --asset)
2. df._v2_symbol from CLI args (NPZ naming)
3. btc_prices.db symbol UPDATE (BTC→BTC/USDT)
4. meta_labeling.py + backtest_validation.py --tf CLI
5. V30_DATA_DIR + SAVAGE22_DB_DIR env vars
6. LSTM parquet symlink (features_{tf}.parquet → features_BTC_{tf}.parquet)
7. killall self-kill → targeted pgrep
8. tee pipefail (bash -c)
9. LSTM NaN imputation (nan_to_num after z-score)
10. Optuna sparse→dense conversion
11. ml_multi_tf.py sparse→dense with RAM check (skips if > 70% avail RAM)
12. nnz guard (hasattr)
13. .toarray() guards on esoteric + HMM lines
14. NPZ skip logic (skip cross gen if NPZ exists)
15. MIN_BASE_FEATURES = 1000 (15m has 1,284 — correct for intraday)
16. backtesting_audit.py USE_GPU_XGB → fixed
17. Optuna/optimizer SKIPPED in cloud pipeline (run separately)

## COST SO FAR
- ~$15 of $32 budget spent
- Texas burning $1.33/hr for 15m
- Estimated remaining: ~$5-8 (Texas 3-4 hrs + any Optuna runs)
