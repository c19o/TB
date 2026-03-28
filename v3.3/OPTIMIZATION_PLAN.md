# V3.3 Optimization Plan — 2026-03-27
# AUTHORITATIVE. 3 rounds of 30-agent Perplexity research. 3x reviewed. 3x matrix compliance audited.

## Matrix Thesis (NON-NEGOTIABLE)
- ALL features preserved. No filtering, no subsampling, no feature selection.
- NaN = missing signal (LightGBM learns split direction). 0 = "value is zero" (different).
- Esoteric features ARE the edge. More diverse signals = stronger predictions.
- feature_pre_filter=False always. The model decides via tree splits, not us.
- LightGBM + EFB is architecturally correct for sparse binary cross features.

## WHY V3.3 IS SLOWER THAN V3.2
1. **3-10x more features**: min_nonzero=3 (was 8) → 2.2M-6M features (was 658K)
2. **More BTC data**: 2010-2026 (was 2019-2026) → 5733 rows on 1d (was ~3500)
3. **Sequential CPCV**: >1M features forces sequential (parallel = 400GB pickle bottleneck)
4. **feature_pre_filter=False**: correct but adds Dataset construction overhead
5. **v3.2 never completed**: both v3.1 and v3.2 CRASHED (feature_name mismatch). No baseline exists.

---

## PHASE 1: CROSS-GEN OVERHAUL

### 1A. Numba prange sorted-index intersection (PRIMARY approach)
- **What**: Convert L/R to CSC, do two-pointer sorted intersection on `indptr`/`indices` for each pair.
- **Why**: Review pass 1 found bitpack+CSC conversion is SLOWER than scipy for 1h/15m. Direct CSC intersection avoids format conversion entirely.
- **How**: Two-pass Numba kernel. Pass 1 = count intersection sizes per pair. Pass 2 = fill pre-allocated indices array.
- **LLVM intrinsics**: Use `numba.extending.intrinsic` for `llvm.ctpop.i64` (co-occurrence counts) and `llvm.cttz.i64` (bit extraction). No C extensions.
- **Expected speedup**: 3-8x for cross gen (eliminates dense allocation + np.nonzero overhead). NOT 50-500x — that was exaggerated for the AND only, ignoring CSC conversion cost.
- **REVIEW CORRECTION**: Bitpacked AND is 50-500x faster for the AND operation ALONE, but back-conversion to CSC negates it for 1h/15m. Use sorted-index intersection as the primary approach. Keep bitpack for co-occurrence pre-filter (POPCNT counting) only.
- **Safety**: SAFE. Binary features have no NaN. bit=0 = feature OFF. Lossless.
- **File**: `v3.3/v2_cross_generator.py` — replace `_cpu_cross_chunk` inner loop.

### 1B. All 13 cross steps parallel
- **What**: Confirmed all 13 steps are independent (no step uses previous cross outputs as input).
- **How**: ThreadPoolExecutor for top-level steps. Numba threads for inner prange.
- **Memory-aware scheduling**: Steps sorted by estimated RAM, launched until 70% RAM ceiling.
- **Safety**: SAFE. Each step reads from same immutable base matrix.

### 1C. Adaptive RIGHT_CHUNK controller
- **What**: Rolling RSS-based sizing instead of static per-TF values.
- **How**: Pilot with RC=16 for first 2 chunks, measure bytes/col, size subsequent chunks from worst of last 3.
- **MemoryError**: Catch, halve, retry same index. V2_RIGHT_CHUNK env var becomes max cap.
- **Safety**: SAFE. Element-wise multiply is order-independent.

### 1D. Bitpacked POPCNT co-occurrence pre-filter
- **What**: Use bitpacked AND+POPCNT for co-occurrence counting. Keep sparse matmul as backup.
- **REVIEW CORRECTION**: Dense BLAS is SLOWER than sparse matmul at 5-15% density. Do NOT replace sparse matmul with dense BLAS.
- **Bitpacked POPCNT**: 8-21ms for ALL 7.1M pairs — faster than both sparse matmul AND dense BLAS.
- **Bloom/CMS**: NOT NEEDED. Exact bitpacked counting is trivially fast.
- **Safety**: SAFE. Mathematically identical results.

### 1E. Memmap CSC streaming (1h/15m only)
- **What**: Write cross gen chunks to NVMe as CSC, two-pass merge into memmaps.
- **Why**: 1h peaks at 1.8TB RAM without streaming.
- **How**: CSC memmap for final output. Producer-consumer pipeline.
- **REVIEW CORRECTION**: "Reduces RAM to a few GB" is MISLEADING. True for cross-gen only. LightGBM Dataset.construct() still needs 80-150GB for 1h (10M features). Budget 100-200GB for full pipeline.
- **Safety**: SAFE on local filesystem. Never use mmap over network storage.

### 1F. Sort pairs by left index for L2 cache reuse
- **What**: Group pairs so same left column stays hot in L2 across its right-side partners.
- **Why**: Each left column participates in ~100 pairs on average. Without tiling, reloaded 100x.
- **Expected**: 2-5x speedup on 15m from tiling alone.

### 1G. Per-step NPZ checkpointing
- **What**: Save intermediate NPZ + names JSON after EACH cross step completes.
- **Why**: Step 5 (Esoteric×TA) takes 9+ hours. Machine death = full restart.
- **Max loss**: One cross step (~30-60 min) instead of full cross gen (~hours).
- **REVIEW CORRECTION**: scipy.sparse.save_npz is NOT atomic. Must use temp file + os.replace() pattern to prevent partial writes from corrupting checkpoints.

### 1H. Intra-step flush (1h/15m only)
- **What**: Every 20 minutes, flush accumulated CSR chunks to partial NPZ on disk.
- **Why**: Single steps on 1h/15m take hours.
- **Max loss**: ~20 minutes instead of full step.

---

## PHASE 2: TRAINING SPEEDUP

### 2A. Parent Dataset + .subset() for CPCV folds
- **What**: Build ONE lgb.Dataset.construct() before CPCV loop. Each fold uses .subset(indices).
- **Why**: Dataset construction (binning 6M features from CSR) takes 15+ min. Currently done per fold.
- **How**: .subset() reuses parent bin mappers + EFB bundles. Near-instant.
- **Leakage**: NONE for binary features (only 2 bins, boundary always at 0.5).
- **HMM overlay**: Pre-compute ALL possible HMM columns, include in parent.
- **Safety**: SAFE. Confirmed from LightGBM C++ source.
- **Files**: `ml_multi_tf.py`, `run_optuna_local.py`.

### 2B. save_binary() for Dataset caching
- **What**: After constructing parent Dataset, save_binary("parent.bin"). Reload for Optuna trials.
- **Why**: Skips ALL CSR-to-bin conversion on subsequent runs.
- **Size**: ~3GB for 1w (6M features), ~214GB for 15m (NOT feasible for 15m).
- **Crash recovery**: Parent.bin survives machine death. Reload + .subset() = instant resume.
- **Safety**: SAFE. Pin LightGBM version (binary format not guaranteed across versions).

### 2C. Parallel Optuna (n_jobs=4, constant_liar=True)
- **What**: Run 4 Optuna trials concurrently via ThreadPoolExecutor.
- **Constraint**: Search space MUST exclude max_bin, min_data_in_bin (trigger Dataset reconstruction = race condition).
- **Sampler**: TPESampler(constant_liar=True, n_startup_trials=40, multivariate=True, group=True).
- **Safety**: SAFE if search space is constrained to tree/boosting params only.

### 2D. WilcoxonPruner (inter-fold pruning)
- **What**: After each CPCV fold, test if trial is statistically worse than best.
- **Config**: WilcoxonPruner(p_threshold=0.1, n_startup_steps=2).
- **CANNOT combine with LightGBMPruningCallback** (step namespace collision).
- **Use early_stopping(50) for intra-fold** (native LightGBM, no trial.report interference).
- **Safety**: RISKY for rare signals. Set n_startup_steps >= 2, p_threshold conservative.

### 2E. extra_trees=True
- **What**: Random threshold selection per feature.
- **For binary features**: Only 1 possible threshold → extra_trees is a no-op. Completely safe.
- **Benefit**: Injects randomness in feature selection, reduces tree correlation.
- **Safety**: SAFE. Binary features = 1 bin boundary = deterministic regardless.

### 2F. DART boosting in Optuna search space
- **What**: Add boosting_type=['gbdt', 'dart'] to Optuna.
- **DART**: Drops trees, not rows. All 5733 rows participate every iteration.
- **REVIEW CORRECTION**: DART has progressive weight dilution risk. After ~50 drops with |D|=5, tree weight = (5/6)^50 ≈ 0.01%. Genuine rare signals get smeared across weakly-weighted trees. Mitigation: skip_drop=0.8, max_drop=1. Or just default to gbdt (safer for rare signals).
- **GOSS**: EXCLUDED. Drops rows where rare esoteric signals fire during "easy" periods. With 5733 rows, 73% chance all 3 rare-signal rows discarded per tree. Zero speed benefit at this row count.
- **RF**: EXCLUDED. No sequential error correction.
- **Safety**: RISKY. Include DART as option but expect gbdt to win most trials.

---

## PHASE 3: LightGBM PARAMETER FIXES

### 3A. min_data_in_bin — NOT critical (review corrected)
- **REVIEW CORRECTION**: min_data_in_bin=3 does NOT kill binary features. LightGBM's GreedyFindBin always adds the rightmost bin unconditionally. A binary feature with 3 ones still gets both bins.
- **Real killer**: feature_pre_filter=True (default) + min_data_in_leaf=20 (default). NeedFilter() checks if any split can satisfy min_data_in_leaf on both sides. We already have feature_pre_filter=False and per-TF min_data_in_leaf (3-15).
- **Action**: Set min_data_in_bin=1 anyway (no harm, minimal benefit). The real protection is feature_pre_filter=False which we already have.

### 3B. Remove max_conflict_rate from config
- **Why**: Parameter was REMOVED from LightGBM 4.x (PR #2699). Generates warnings, does nothing.
- **EFB behavior**: Hard-coded to rows/10000 = 0 for our datasets. Cannot be controlled.

### 3C. max_bin=255 (CHANGE from 15)
- **REVIEW PASS 2 CORRECTION**: max_bin DIRECTLY controls max EFB bundle size. EFB offset encoding needs K+1 bins for K binary features per bundle. max_bin=15 → max 14 features/bundle → ~428K bundles. max_bin=255 → max 254/bundle → ~23K bundles = 18x fewer histograms.
- **Binary features still get num_bin=2 individually** — max_bin is a ceiling, not a floor.
- **The "4x slowdown at max_bin=63" in CLAUDE.md was likely a different issue** (different feature counts or params). Larger max_bin = fewer bundles = less total work.
- **EXCEPTION**: 1h/15m may need enable_bundle=False (EFB conflict graph intractable at 10M features). In that case, max_bin=255 still helps continuous base features.
- **Action**: Change config.py V3_LGBM_PARAMS max_bin from 15 to 255.

### 3D. Optuna search space updates
- **ADD**: min_sum_hessian_in_leaf (log 0.001-0.5), path_smooth (0.0-50.0)
- **ADD**: extra_trees as searchable categorical [True, False] (NOT default True — degrades continuous base features)
- **REMOVE**: max_conflict_rate (dead parameter in LightGBM 4.x)
- **REMOVE**: DART from boosting_type search (REVIEW PASS 2: weight dilution risk, skip_drop=0.8 = noisy gbdt. Use gbdt only.)
- **CONSTRAIN**: max_depth to 3-7 (was 3-12, binary features overfit with deep trees)
- **FIX**: early_stopping patience scale inversely with learning_rate
- **FIX THESIS VIOLATION**: Set OPTUNA_TF_ROW_SUBSAMPLE to 1.0 for ALL TFs. Row subsampling in Stage 1 kills rare esoteric signals below min_data_in_leaf. Configs found on subsampled data systematically miss rare-signal hyperparams.
- **ADD for 1h/15m**: enable_bundle=False (EFB conflict graph is O(rows×k²), intractable at 10M features)
- **ADD**: max_bin=255 in search-frozen params (was 15, limits EFB to 14/bundle)

### 3E. NaN semantics verified CORRECT
- **Cross features**: Pure 0/1 after binarization. No NaN survives. Structural zeros = feature OFF (value=0).
- **Base features**: NaN preserved as explicit CSR entries. LightGBM treats as missing.
- **Bitpacking**: Unambiguous. bit=1 = ON, bit=0 = OFF. Lossless.
- **CLAUDE.md correction needed**: "structural zero = missing" is wrong for cross features. Should be "structural zero = 0 (feature OFF)".

---

## PHASE 4: OS / SYSTEM TUNING (20-40% combined)

### 4A. tcmalloc via LD_PRELOAD
- **What**: `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4`
- **Why**: Per-thread malloc caches, lower lock contention. 5-15% speedup.
- **Safety**: SAFE. Use `_minimal` only (no profiler/signal handlers). Compatible with scipy + Numba.

### 4B. Transparent Huge Pages (THP)
- **What**: `echo always > /sys/kernel/mm/transparent_hugepage/enabled`, `defrag=defer+madvise`
- **Why**: Reduces dTLB misses from 5-15% to <2%. 5-20% speedup.
- **Safety**: SAFE with defer+madvise (no synchronous stalls).

### 4C. vm.swappiness=1
- **What**: `sysctl vm.swappiness=1`
- **Why**: Prevents kernel from swapping hot model state to disk.

### 4D. NUMA binding (multi-socket machines)
- **What**: `numactl --cpunodebind=0 --membind=0` per training process.
- **Why**: Eliminates 1.7-2.5x remote DRAM penalty. 10-30% speedup on 2+ socket machines.
- **For parallel Optuna**: One worker per NUMA node.

### 4E. GC disable during training
- **What**: `gc.disable()` during CPCV + Optuna loops. `gc.enable(); gc.collect()` after.
- **Why**: LightGBM C++ does the heavy lifting. Python GC churn is pure overhead.

---

## PHASE 5: I/O OPTIMIZATION

### 5A. Indices-only NPZ storage
- **What**: Store only `indptr` + `indices` arrays. Reconstruct `data = np.ones()` on load.
- **Why**: All values are 1.0 (binary). Skip 50% of I/O.
- **Safety**: SAFE. `np.ones(n, dtype=np.float32)` produces exact IEEE 754 1.0.
- **Verify**: Assert `(original.data == 1.0).all()` before switching.

### 5B. CSC format for LightGBM
- **What**: Store final cross matrix as CSC (not CSR). LightGBM iterates column-wise internally.
- **Why**: Avoids internal CSR→column transpose.

---

## PHASE 6: INFERENCE OPTIMIZATION

### 6A. Used-features-only inference
- **What**: Extract features with non-zero split importance from trained model. Only compute those crosses at inference.
- **Why**: Model uses ~5K-15K features out of 6M. 99.75% of crosses are never split on.
- **How**: `importances = model.feature_importance('split'); used = importances > 0`
- **Live inference**: ~50-100ms per bar (dominated by pandas feature computation, not crosses).

### 6B. Pruned model for live trading
- **What**: Remap model's feature indices from 6M-space to 5K-space. Smaller model, faster predict.
- **How**: Parse model_to_string(), remap split_feature indices, reload.

### 6C. lleaves compilation (optional)
- **What**: Compile LightGBM model to LLVM native code. 10-30x faster prediction.
- **When**: Only needed for sub-minute timeframes.

---

## PHASE 7: CLOUD DEPLOYMENT

### 7A. Provider strategy
- 1w/1d: vast.ai (cheap, 64-128 cores, 256-512GB RAM)
- 4h: GCP c3-highmem-176 (176 vCPUs, 1408GB RAM, $1.70/hr spot)
- 1h/15m: GCP c3d-highmem-360 (360 vCPUs, 2880GB RAM, ~$3.40/hr spot)
- GCP $300 free credits for new accounts

### 7B. Crash recovery
- Per-step NPZ checkpoints (cross gen)
- save_binary() for Dataset cache
- Optuna SQLite with heartbeat + RetryFailedTrialCallback
- Local sync watchdog (rsync every 5 min)
- LightGBM init_model for mid-fold resume (save every 100 trees)
- Max loss target: <30 minutes of work per machine death

### 7C. Auto-tune per machine
- Detect cores, RAM, NUMA topology, disk speed at deploy time
- Set RIGHT_CHUNK, OMP_NUM_THREADS, NUMBA_NUM_THREADS, LightGBM num_threads
- NUMA-aware Optuna worker placement

---

## PHASE 8: POST-TRAINING

### 8A. Feature importance pipeline
- Gain importance (not split count — biased for binary)
- Fold-wise stability (which features recur across CPCV splits)
- Sign consistency (bullish in all folds or flips?)
- Composite: rank_gain × stability × sign_consistency
- SHAP only on top 200 features (27 seconds)
- Permutation importance on top 1K (1 minute)

### 8B. GBDT+LR calibration layer
- Shallow GBDT (50 trees, 8 leaves) → leaf features → elastic net LR
- Produces calibrated probabilities for Kelly sizing
- OOF leaf extraction (inner 3-fold within each CPCV fold) — no leakage
- NOT the Facebook architecture (they had billions of rows)

### 8C. Custom PnL-aligned objective (optional)
- Asymmetric cost: wrong direction = 3x missed opportunity
- Regime-weighted: crash bars get 3x cost scaling
- Complementary to meta-labeling (M1 maximizes recall, M2 handles precision)
- Guard: hessian floored at 1e-6, no exact hessian (stability risk)

### 8D. Retraining schedule
- 1w: monthly. 1d/4h: weekly. 1h/15m: weekly + daily incremental.
- Expanding window (not sliding) — esoteric signals have multi-year cycles.
- Pre-compute esoteric crosses for future weeks (60% of work, timestamp-only).
- PSI monitoring: technical features (0.05 watch/0.10 alert), esoteric (0.10/0.20).
- Performance gates: rolling Sharpe vs deflated OOS baseline.

---

---

## MATRIX COMPLIANCE FIXES (must apply before training)

### MC-1. HMM regime weights: VIOLATION
- **Current**: bear LONGs get 0.15 weight, bull SHORTs get 0.15 weight
- **Problem**: Pre-decides counter-trend signals are noise BEFORE the model evaluates them. Esoteric signals may specifically predict counter-trend moves (e.g., full moon at bull-to-bear transition).
- **Fix**: Either (a) remove regime weighting entirely, add HMM state as input feature column (fully thesis-aligned), or (b) raise floor from 0.15 to 0.5 (compromise).
- **Files**: ml_multi_tf.py:585-586, run_optuna_local.py:285-286

### MC-2. Optuna Stage 1 row subsampling: VIOLATION
- **Current**: OPTUNA_TF_ROW_SUBSAMPLE = {1w:1.0, 1d:0.5, 4h:0.5, 1h:0.3, 15m:0.3}
- **Problem**: Subsampling rows kills rare signals below min_data_in_leaf. Stage 1 configs miss rare-signal hyperparams, bias Stage 2 search.
- **Fix**: Set ALL values to 1.0.
- **File**: config.py:313-316

### MC-3. save_binary() shared across CPCV folds: VIOLATION
- **Current**: Plan proposed building ONE parent Dataset and using .subset() for all folds.
- **Problem**: EFB conflict graph computed on ALL rows including test fold = temporal structure leakage.
- **Fix**: Reconstruct lgb.Dataset per fold using train-only rows. Use .subset() WITHIN each fold (train/val split), not ACROSS folds.
- **Alternative**: For binary features, leakage is minimal (bin boundaries fixed at 0.5). Accept this for speed, document the tradeoff.

### MC-4. Early stopping: scale inversely with learning rate
- **Current**: early_stopping(50) fixed.
- **Problem**: Trees 1-200 learn TA signals. Esoteric signals need 800+ trees. ES(50) kills training before esoteric residuals are addressed.
- **Fix**: ES = max(50, int(100 * (0.1 / learning_rate))). At lr=0.05→ES(200), lr=0.01→ES(1000).
- **Files**: ml_multi_tf.py, run_optuna_local.py

### MC-5. SIGTERM checkpoint: add LightGBM callback
- **Current**: SIGTERM handler only cleans lockfile. All training progress lost on preemption.
- **Fix**: Add LightGBM callback that saves model every 100 rounds. SIGTERM sets atomic flag, callback checks flag and saves at next safe boundary.
- **Files**: ml_multi_tf.py, cloud_run_tf.py

### MC-6. Co-occurrence filter: fix smoke test
- **Current**: Production uses MIN_CO_OCCURRENCE=3, smoke_test_v3.py uses 8.
- **Fix**: Align smoke_test_v3.py to 3.
- **File**: smoke_test_v3.py:176

### MC-7. Used-features-only inference: full-dimension sparse
- **Current**: Plan says "5K features at positions 0-4999."
- **Problem**: LightGBM maps by column index. 5K columns → positions 0-4999 ≠ actual feature indices → WRONG predictions.
- **Fix**: Build full 6M-wide sparse vector, populate only 5K used positions. Or: remap model indices AND verify mapping with model.feature_name().

### MC-8. Atomic NPZ writes
- **Current**: scipy.sparse.save_npz is NOT atomic.
- **Fix**: Use tempfile + os.replace() pattern. Already noted in Phase 1G.

---

## REJECTED OPTIMIZATIONS (with reasons)

| Optimization | Reason for Rejection |
|-------------|---------------------|
| **Manual pre-bundling (6M→430K)** | Interaction loss between features in same bundle is permanent. Safety check: RISKY. |
| **GOSS boosting** | Discards "easy" rows where rare esoteric signals validate. Unsafe for matrix thesis. |
| **CEGB penalty** | Flat penalty kills rare feature splits (gain=0.003 vs penalty=0.004 = invisible). |
| **use_quantized_grad** | Gradient rounding to 4 bins kills rare signal detection. |
| **Roaring Bitmaps** | Container dispatch overhead at our row counts. No LightGBM integration. |
| **XGBoost** | No EFB. Catastrophically slow with sparse. Caused 12% accuracy drop in v3.2. |
| **CatBoost** | No EFB equivalent. |
| **TabNet/NODE** | Can't handle 6M features. |
| **Row-partitioned boosting** | Kills rare signals (Perplexity-confirmed). |
| **Dense for 1M+ features** | Pickle bottleneck in parallel CPCV. |
| **AVX-512 manual intrinsics** | Memory-bandwidth bound. LLVM auto-vectorizes. |
| **Dask/Ray for Optuna** | Overhead not worth it for single machine. |
| **Feature filtering/subsampling** | Violates matrix thesis. |
| **Training on partial crosses** | Violates matrix thesis (all features must be seen). |
| **SVD latent factors** | 5733 rows too few. Unsupervised SVD captures noise, not alpha. |
| **Regime-conditional models** | 1433 rows/regime = statistical disaster at 6M features. Violates "matrix is universal". |
| **Bloom/CMS pre-filter** | Bitpacked POPCNT makes exact counting faster than probabilistic. |

---

## TRAINING TIME ESTIMATES (per TF, per stage, with optimizations)

### 1W — vast.ai 64c, $0.30/hr
| Stage | Time |
|-------|------|
| Feature build | 5 min |
| Cross gen (bitpacked) | 15 sec |
| save_binary | 30 sec |
| CPCV (4 folds, dense) | 13 min |
| Optuna (200 trials, n_jobs=4) | 5.5 hrs |
| Meta + PBO + SHAP | 8 min |
| **TOTAL** | **6.5 hrs ($2)** |
| **Without Optuna** | **45 min ($0.25)** |

### 1D — vast.ai 128c, ~$1.75/hr
| Stage | Time |
|-------|------|
| Feature build | 15 min |
| Cross gen (Numba intersection) | 10 min |
| save_binary | 5 min (~0.9-1.2GB file) |
| CPCV (4 folds, sparse seq) | 2 hrs (~30 min/fold, REVIEW CORRECTED from 25 min) |
| Optuna (200 trials, n_jobs=4) | 13.5 hrs (~120 effective fold-units × 30 min / 4) |
| Meta + PBO + SHAP | 15 min |
| **TOTAL** | **16 hrs ($28)** |
| **Without Optuna** | **2.5 hrs ($4)** |

### 4H — GCP c3-highmem-176, ~$1.70/hr spot
| Stage | Time |
|-------|------|
| Feature build | 30 min |
| Cross gen (Numba intersection) | 36 min |
| save_binary | 15 min |
| CPCV (4 folds, sparse seq) | 3.3 hrs (~50 min/fold, REVIEW CORRECTED from 43 min) |
| Optuna (200 trials, n_jobs=4) | 25 hrs (~120 × 50 min / 4) |
| Meta + PBO + SHAP | 15 min |
| **TOTAL** | **30 hrs ($51)** |
| **Without Optuna** | **5 hrs ($9)** |

### 1H — GCP c3d-highmem-360, ~$3.40/hr spot (REVIEW CORRECTED from $6.37)
| Stage | Time |
|-------|------|
| Feature build | 1 hr |
| Cross gen (Numba + streaming) | 2.3 hrs |
| save_binary | 30 min |
| CPCV (4 folds, sparse seq) | 7 hrs (~104 min/fold) |
| Optuna (200 trials, n_jobs=4) | 52 hrs (~120 × 104 min / 4) |
| Meta + PBO + SHAP | 20 min |
| **TOTAL** | **63 hrs ($214)** |
| **Without Optuna** | **11 hrs ($37)** |

### 15M — GCP c3d-highmem-360, ~$3.40/hr spot (REVIEW CORRECTED from $6.37)
| Stage | Time |
|-------|------|
| Feature build | 2 hrs |
| Cross gen (Numba + streaming) | 5.8 hrs |
| save_binary | 45 min |
| CPCV (4 folds, sparse seq) | 10 hrs (~150 min/fold, REVIEW CORRECTED from 174) |
| Optuna (200 trials, n_jobs=4) | 75 hrs (~120 × 150 min / 4) |
| Meta + PBO + SHAP | 30 min |
| **TOTAL** | **94 hrs ($320)** |
| **Without Optuna** | **19 hrs ($65)** |

### BUDGET OPTIONS (REVIEW CORRECTED — GCP spot ~$3.40/hr not $6.37)
| Strategy | Total Cost | Total Time (parallel) |
|----------|-----------|----------------------|
| All 5 TFs, NO Optuna | **$115** | **19 hrs** |
| + Optuna 1w/1d only (200 trials) | **$147** | **19 hrs** (Optuna runs after) |
| + Optuna 4h (200 trials) | **$198** | **30 hrs** |
| + Optuna 1h (200 trials) | **$412** | **63 hrs** |
| + Optuna 15m (200 trials) | **$732** | **94 hrs** |
| All 5 TFs, full Optuna | **$615** | **94 hrs** |

---

## IMPLEMENTATION ORDER

### Weekend 1: Core optimizations + CPCV training ($115)
1. Implement Numba sorted-index intersection (Phase 1A)
2. Implement adaptive RIGHT_CHUNK (Phase 1C)
3. Implement parent Dataset + .subset() (Phase 2A)
4. Apply LightGBM param fixes (Phase 3A-3E)
5. Apply OS tuning to setup.sh (Phase 4)
6. Train ALL 5 TFs with CPCV only (no Optuna)

### Weekend 2: Optuna for cheap TFs ($7)
7. Implement parallel Optuna + WilcoxonPruner (Phase 2C-2D)
8. Run Optuna on 1w (200 trials, 6.5 hrs, $2)
9. Run Optuna on 1d (200 trials, 13 hrs, $5)
10. Warm-start 4h/1h/15m Optuna with best params from 1w/1d

### Weekend 3+: Optuna for expensive TFs
11. Run Optuna on 4h (50-200 trials, $45)
12. Run Optuna on 1h (50-100 trials, warm-started, $100-376)
13. Run Optuna on 15m (50 trials, warm-started, $160-637)

### Post-training:
14. Feature importance pipeline (Phase 8A)
15. GBDT+LR calibration (Phase 8B)
16. Used-features-only inference (Phase 6A)
17. Pruned model deployment (Phase 6B)
18. Retraining schedule setup (Phase 8D)
