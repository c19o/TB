# Session Resume — V3.2 Training — 2026-03-24

## STATUS: 4/5 TFs COMPLETE. 15m TRAINING ON TEXAS.

## VERIFIED RESULTS

| TF | Features | Model | LSTM | Meta AUC | PBO | NNZ | Status |
|---|---|---|---|---|---|---|---|
| 1w | 656,825 SPARSE | ✅ 142KB | ✅ 55.3% | 0.670 | 0.50 REJECT | 20.8M ✅ | Downloaded (1w_norway/) |
| 1d | ~2.87M SPARSE | ✅ 598KB | ✅ 7MB | ❓ | **0.00 DEPLOY** | 215M ✅ | Downloaded (1d_final/) |
| 4h | 3,196,691 SPARSE | ✅ 893KB | ✅ 53.5% | 0.616 | 0.40 REJECT | 288M ✅ | Downloaded (4h_final/) |
| 1h | 4,249,865 SPARSE | ✅ 2.7MB | ✅ 53.9% | 0.648 | **0.14 DEPLOY** | 1.19B ✅ | Downloaded (1h_final/) |
| 15m | ~2.3M SPARSE | ⏳ training | ❌ | ❌ | ❌ | **3.88B ⚠️** | Texas running |

**1d PBO=0.00 and 1h PBO=0.14 = PRODUCTION READY**

## ACTIVE MACHINES

| Machine | TF | Cores | RAM | $/hr | SSH |
|---|---|---|---|---|---|
| Texas (33446502) | 15m | 256c | 1TB | $1.33 | ssh -p 16502 root@ssh6.vast.ai |

Norway (33430087) is done — can kill to save $0.94/hr.

## DOWNLOADED ARTIFACTS (v32_cloud_results/)

```
1w_norway/   23 files — model, LSTM, meta, PBO, cross names (NO NPZ)
1d_final/    12 files — model, LSTM, NPZ (292MB), predictions
4h_final/    24 files — model, LSTM (17MB), NPZ (125MB), meta, PBO, cross names
1h_final/    20 files — model, LSTM (27MB), NPZ (1.6GB), meta, PBO, cross names
```

## OPTUNA — DEFERRED

Optuna was skipped in the pipeline (runs later). Per Perplexity: Optuna is fully stateless — just needs parquet + NPZ. Can run on any machine anytime. Critical for regularization tuning with 2.9M features.

To run Optuna later:
```bash
cd v3.2_2.9M_Features
cp ../v32_cloud_results/1d_final/v2_crosses_BTC_1d.npz .
cp "../v3.0 (LGBM)/features_BTC_1d.parquet" .
python -u run_optuna_local.py --tf 1d
```

## 15m INT32 NNZ RISK

15m cross matrix has 3.88B non-zero entries, exceeding LightGBM's int32 limit (2.15B). Model may silently produce garbage. Verify predictions after training. If bad, subsample to 100K rows.

## ALL BUGS FIXED IN LOCAL CODE

1. --symbol BTC (not --asset)
2. df._v2_symbol set from CLI args
3. btc_prices.db symbol UPDATE (BTC→BTC/USDT)
4. meta_labeling.py + backtest_validation.py --tf CLI
5. V30_DATA_DIR + SAVAGE22_DB_DIR env vars
6. LSTM parquet symlink
7. killall self-kill → targeted pgrep
8. tee pipefail
9. LSTM NaN imputation (nan_to_num after z-score)
10. Optuna sparse→dense conversion
11. ml_multi_tf.py sparse→dense with RAM check
12. nnz guard (hasattr)
13. .toarray() guards on esoteric + HMM lines
14. NPZ skip logic in cloud_run_tf.py
15. MIN_BASE_FEATURES = 1000 (15m has 1,284)
16. backtesting_audit.py USE_GPU_XGB undefined
17. Optuna + optimizer SKIPPED (run later)

## LESSONS LEARNED (saved to memory)

- LightGBM sparse = single-threaded (must convert to dense)
- LightGBM int32 NNZ limit > 2^31 = silent garbage
- Audit ALL scripts for same bug pattern before deploying
- vast.ai machines die without warning — download partial results frequently
- LSTM crashes on NaN — must impute after z-score
- Run 1 TF at a time, smallest first
- Never kill running cross gen — add NPZ skip logic
- Optuna is fully decoupled — run later with parquet + NPZ

## COST TRACKING

- Norway: ~$0.94/hr × ~7 hrs = ~$6.58
- Texas: ~$1.33/hr × ongoing
- Previous 5 machines: ~$3.32/hr × ~1.5 hrs = ~$4.98
- 1w rerun (California, killed): ~$0.39 × 0.1 = ~$0.04
- Total spent: ~$12-15 of $32 budget
