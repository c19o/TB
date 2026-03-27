# V3.3 Session Resume — 2026-03-27 11:30 UTC

## STATUS: TRAINING IN PROGRESS (1d + 4h active, 1w DONE)

## ACTIVE MACHINES (vast.ai)
| TF | Instance ID | Host | Port | $/hr | Status |
|----|------------|------|------|------|--------|
| **1d** | 33637153 | ssh7.vast.ai | 37152 | $1.35 | CPCV training (sparse+sequential, 6M features) |
| **4h** | 33638560 | ssh8.vast.ai | 38560 | $1.16 | Cross gen step 5/13 (Esoteric×TA) |

**Burn rate: $2.50/hr. Balance: ~$30.**

SSH: `ssh -o StrictHostKeyChecking=no -o IdentityFile=~/.ssh/vast_key -o IdentitiesOnly=yes -p PORT root@HOST`

## COMPLETED
- **1w**: DONE. 71.9% accuracy (4 CPCV folds). 2.2M features. Model 113MB. All artifacts downloaded to v3.3/. Machine destroyed.

## CANCELLED (too expensive at current speeds)
- **1h**: Cancelled. Only parquet downloaded (no crosses completed). Cross gen RC=500 peaked 1871G/2003G near OOM. Would need RC=300 + sparse+sequential fix.
- **15m**: Cancelled. Cross gen RC=500 OOM'd at 1892G. RC=200 stable but total pipeline estimated 120-300h ($146-366). Not viable without code optimization.

## CODE FIXES APPLIED THIS SESSION (all pushed to git, branch v3.3)

### Fix 1: Int64 indptr for NNZ > 2^31 (Perplexity-validated)
- `_ensure_lgbm_sparse_dtypes()`: indptr=int64, indices=int32 on ALL TFs
- Replaces ValueError with logging + sequential CPCV force
- `_predict_chunked()` for IS predictions on large train sets
- Row-partitioned boosting **REJECTED** (Perplexity: kills rare signals)
- Files: ml_multi_tf.py, run_optuna_local.py, config.py

### Fix 2: RIGHT_CHUNK OOM prevention
- Auto RIGHT_CHUNK=2000 OOMs on ALL TFs except 1w
- Per-TF settings: 1w=auto, 1d=200, 4h=500, 1h=300, 15m=200(optimal 300)
- Set via `export V2_RIGHT_CHUNK=N` before launch

### Fix 3: Sparse+Sequential CPCV for 1M+ features (Perplexity-validated, 6-12x speedup)
- Skip dense conversion when features > 1M (pickle serialization bottleneck)
- Force sequential CPCV (no ProcessPoolExecutor) for 1M+ features
- Cap num_threads to 32 for <10K rows
- Add deterministic=True to V3_LGBM_PARAMS
- **Root cause**: dense+parallel pickles ~400GB to workers = 8h+ stuck at load 1.04
- **Fix**: sparse sequential = 2-3h for 1d (vs 12-30h before)
- Files: ml_multi_tf.py, config.py

### Fix 4: 15m RAM threshold 2048→1500 (cgroup reports less than host)
- File: cloud_run_tf.py

## OBSERVED RAM PEAKS (cross gen, per TF)
| TF | Rows | RC | Peak RAM | Machine | Result |
|----|------|----|----------|---------|--------|
| 1w | 818 | auto(2000) | 11G | 377GB | OK |
| 1d | 5,727 | 200 | 313G | 944GB | OK (OOM'd at 377GB RC=2000, 503GB RC=500) |
| 4h | 17,520 | 500 | 1213G | 2TB | OK (OOM'd at 1007GB RC=2000, 1007GB RC=500) |
| 1h | 75,405 | 300 | 1871G | 2TB | Near-OOM (RC=500 peaked 1871G/2003G) |
| 15m | 293,980 | 200 | 574G | 2TB | OK (OOM'd at 1892G RC=500) |

## CROSS FEATURE COUNTS (observed)
| TF | Base | Crosses | Total | NNZ | Density | Training Mode |
|----|------|---------|-------|-----|---------|---------------|
| 1w | 3,331 | 2,195,129 | **2.2M** | 46M | 2.6% | Dense (7GB) |
| 1d | 3,796 | 6,039,797 | **6.0M** | 498M | 1.4% | Sparse (fix applied) |
| 4h | TBD | TBD | TBD | TBD | TBD | Sparse (fix pre-deployed) |

## 1w RESULTS (first-ever completed model)
- **CPCV Accuracy**: 71.9% mean (73.3%, 66.7%, 73.0%, 74.6%)
- **PrecL**: 92.0%, 61.9%, 62.5%, 0.0% (varies by fold)
- **PrecS**: 0.0% all folds (only 56 shorts in 818 rows)
- **Trees**: 96, 385, 446, 52 (final model: 1041 trees, 347 per class)
- **Model**: 113MB, 2,198,427 features
- **Meta-labeling**: AUC=0.250, acc=0.643 (tiny test set: 28 trades)
- **PBO**: FAILED (is_metrics not passed — code bug, non-fatal)
- **No v3.2 baseline exists** — 1w never completed training before

## WHAT TO DO NEXT
1. **Monitor 1d**: CPCV training with sparse+sequential fix. Should complete in ~3-4h. Download artifacts + evaluate before destroying.
2. **Monitor 4h**: Still in cross gen step 5. Fix already uploaded. When CPCV starts, it will use sparse+sequential. ~8-12h total.
3. **After 1d/4h complete**: Evaluate results, compare accuracy, download everything.
4. **1h/15m**: Need code optimization before attempting again. Key issues:
   - Cross gen too slow (single-threaded element-wise multiply)
   - 15m CPCV would take 120-300h with current code
   - Need parallel cross gen fix (Numba prange or ThreadPoolExecutor)
   - Or bigger/faster machines
5. **PBO bug**: Fix is_metrics passing in ml_multi_tf.py → backtesting_audit.py

## KEY PERPLEXITY FINDINGS (this session)
- Row-partitioned boosting KILLS rare signals (features firing 15 times → ~1 per chunk → below min_data_in_leaf)
- LightGBM PR #1719 (2018) fixed int64 sparse support — no need for row partitioning
- Dense+parallel CPCV is SLOWER than sparse+sequential for 1M+ features (pickle bottleneck)
- Sparse histogram O(2×NNZ) is faster than dense O(rows×features) at <5% density
- "Sparse CSR serializes OpenMP" claim is OUTDATED — not true in LightGBM 4.x
- EFB works identically in sparse and dense modes
- Chunked cross gen (column chunks + hstack) preserves matrix exactly — lossless

## MACHINES DESTROYED THIS SESSION
- 33634628 (1w RTX 5060 Ti) — done, artifacts downloaded
- 33634629 (1d RTX 4060 Ti) — OOM'd at 377GB
- 33635568 (1d RTX 5080 504GB) — OOM'd at 503GB
- 33634631 (4h RTX 5080 1TB) — OOM'd at 1007GB
- 33634634 (15m A40 2TB) — SSH key issue, re-rented
- 33634957 (15m A40 2TB) — cancelled, too expensive
- 33634633 (1h A40 2TB) — cancelled, too expensive

## GIT STATUS (branch v3.3, all pushed)
Latest commit: `59f5ff2` — sparse+sequential CPCV fix
All training docs updated with RIGHT_CHUNK per TF and OOM history.

## MATRIX THESIS (include in ALL agent contexts)
This is NOT a conventional trading bot. The edge is 2.9 MILLION sparse binary cross features generated by cross-multiplying esoteric signals (gematria, astrology, numerology, space weather, tweet decodes) with technical analysis indicators. LightGBM's EFB (Exclusive Feature Bundling) bundles mutually exclusive binary features — architecturally why LightGBM is used, never XGBoost. NaN = "missing signal" (LightGBM learns optimal split direction). 0 = "value is zero" (DIFFERENT meaning). NO feature can EVER be filtered, subsampled, or removed. The model decides via tree splits. feature_pre_filter=False is CRITICAL. Esoteric features ARE the edge.
