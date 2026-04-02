# Per-Timeframe ML Specialist Report — V3.3

**Generated**: 2026-03-30
**Pipeline**: LightGBM sparse CSR + EFB pre-bundling + Optuna Phase 1/2 + CPCV
**Hardware baseline**: Local 13900K (24c/32t, 5.8 GHz) + RTX 3090 (24GB) + 64GB DDR5

---

## Executive Summary

The pipeline trains 5 timeframes with wildly different statistical profiles. 1w has a feature-to-row ratio of 2,500:1 (extreme overfitting territory), while 15m has a healthy 10:1 ratio. Each TF needs distinct regularization, CPCV grouping, and hardware strategy. The dominant bottleneck shifts from I/O (1w) to histogram construction (15m).

---

## 1. Per-TF Statistical Profile

| TF | Rows | Base Features | Cross Features | Total Features | EFB Bundles (est.) | Feature:Row Ratio | Sparsity (%) |
|----|------|---------------|----------------|----------------|--------------------|--------------------|--------------|
| 1w | 1,158 | ~621 | ~600K | ~600K | ~4,700 | 518:1 | ~99.9% |
| 1d | 5,733 | ~621 | ~1M+ | ~1M+ | ~8,000 | 175:1 | ~99.8% |
| 4h | 8,794 | ~621 | ~2.9M | ~2.9M | ~23,000 | 330:1 | ~99.7% |
| 1h | 90,000 | ~621 | ~5M | ~5M | ~40,000 | 56:1 | ~99.5% |
| 15m | 227,000 | ~621 | ~10M | ~10M | ~79,000 | 44:1 | ~99.3% |

**Note**: Row counts from config.py comments. 1w shows "818" (BTC only) and "1158" (multi-asset) in different places; using 1,158 as the training count.

---

## 2. Per-TF Deep Analysis

### 2.1 — 1w (Weekly)

**Rows**: 1,158 | **Features**: ~600K | **Ratio**: 518:1

#### Overfitting Risk: EXTREME
- 518 features per row. Even with EFB bundling down to ~4,700, that is 4.1 bundles per row.
- With num_leaves=7 and min_data_in_leaf=8, each leaf needs 8 rows minimum. A 7-leaf tree partitions 1,158 rows into ~165 rows/leaf on average. This is survivable.
- The real guard is EFB: 600K binary crosses collapse to ~4,700 bundles via exclusive feature bundling, making the effective dimensionality manageable.
- CPCV (5,2) gives 60% train = ~695 rows for training. At 8 min_data_in_leaf, max leaf count is 695/8 = 86. num_leaves=7 is well below this ceiling.
- **Verdict**: Overfitting contained by EFB + shallow trees + aggressive early stopping.

#### CPU vs GPU Crossover
- **CPU wins decisively**. 1,158 rows x 4,700 EFB bundles = 5.4M histogram cells per round. A single CPU core fills this in <1ms. GPU kernel launch overhead (~50us/kernel x 48 kernels/round) dominates.
- GPU histogram only makes sense when rows x bundles > ~100M (GPU amortizes launch overhead).
- **Verdict**: CPU only. `device=cpu` is correct. GPU adds latency, not speed.

#### CPCV Groups: (5, 2)
- C(5,2) = 10 paths. 60% train fraction = 695 rows.
- With 10-20 rare signal firings per feature, 60% train means ~6-12 firings in training set. At min_data_in_leaf=8, a signal firing 10x will appear 6x in training — barely enough for a single leaf.
- groups=3 would give 67% train = 775 rows, only marginally better.
- groups=10 would give 80% train = 926 rows, but C(10,2) = 45 paths is wasteful for 1,158 rows (each fold = 116 rows, too few for reliable OOS scoring).
- **Verdict**: (5, 2) is correct. Optimal for rare signal preservation vs fold size.

#### Optimal Parameters
| Parameter | Current | Optimal | Rationale |
|-----------|---------|---------|-----------|
| min_data_in_leaf | 8 | **8** | Cannot go lower without fitting noise. Cannot go higher without killing 10-fire signals. |
| num_leaves | 7 | **7** | 1,158 / 7 = 165 rows/leaf. Trees deeper than 7 would create leaves with <50 rows. |
| learning_rate | 0.03 final | **0.03** | Correct for 800 rounds. Lower (0.01) would need 2,400 rounds with no gain on 1,158 rows. |
| num_rounds | 800 | **200-400** | Early stopping likely fires at 150-250 on this data size. 800 is the ceiling, not the target. |
| feature_fraction | 0.9 | **0.9** | After EFB bundling, 0.9 x 4,700 = 4,230 bundles/tree. Adequate coverage of rare signals. |

#### Training Time Bottleneck
- **I/O and Dataset construction** dominate. Loading 600K-column NPZ + EFB pre-bundling takes longer than actual boosting.
- Histogram construction: 1,158 rows x 4,700 bundles x 200 rounds = 1.1B operations. Single-threaded CPU: <5 seconds.
- Split finding: negligible (4,700 candidates x 255 bins x 7 leaves = 8.4M comparisons/round).
- **Bottleneck**: Optuna trial overhead (dataset reconstruction per trial) > boosting time.

---

### 2.2 — 1d (Daily)

**Rows**: 5,733 | **Features**: ~1M+ | **Ratio**: 175:1

#### Overfitting Risk: HIGH
- 175:1 ratio before EFB. After EFB (~8,000 bundles), effective ratio is 0.72:1 (rows > bundles). This is healthy territory.
- num_leaves=15 with 5,733 rows = 382 rows/leaf average. Plenty of data per leaf.
- **Verdict**: Moderate risk. EFB makes this tractable. Current params are appropriate.

#### CPU vs GPU Crossover
- 5,733 x 8,000 = 45.9M histogram cells/round. CPU handles this in ~3-5ms/round (32 threads). GPU would need ~2ms (kernel launch + compute). Near the crossover point.
- For Optuna (25 trials x ~60 rounds), CPU is faster (no GPU context switching between trials).
- For final retrain (800 rounds), GPU is marginally faster if the cuda_sparse fork is loaded.
- **Verdict**: CPU for Optuna, GPU marginal benefit for final train. CPU recommended for simplicity.

#### CPCV Groups: (5, 2)
- 60% train = 3,440 rows. 10-fire signals get ~6 in training. Tight but viable.
- (10, 2) would give 80% train = 4,586 rows, but each fold = 573 rows — adequate for OOS.
- **Verdict**: (5, 2) is correct for consistency with 1w. Could upgrade to (7, 2) for more train data while keeping folds at 819 rows.

#### Optimal Parameters
| Parameter | Current | Optimal | Rationale |
|-----------|---------|---------|-----------|
| min_data_in_leaf | 8 | **8** | Correct. 10-fire signals at 60% train = 6 in train. 8 is the minimum leaf, signal needs co-occurrence with another split. |
| num_leaves | 15 | **15-31** | 5,733 / 15 = 382 rows/leaf. Could push to 31 (185 rows/leaf) for more expressiveness. |
| Optuna subsample | 1.0 | **1.0** | Correct. 5,733 rows is small enough to use all. |

#### Training Time Bottleneck
- **EFB construction** (30s) + Optuna search (25 trials x 60 rounds x 2-fold CPCV = 3,000 boosting iterations total).
- Per iteration: 5,733 x 8,000 x 32 threads = ~1-2ms. Total Optuna: ~6 seconds of pure boosting. Overhead (dataset creation, eval, logging) dominates at 5-10x actual compute.
- **Bottleneck**: CPCV fold dataset construction (Python overhead, not numeric).

---

### 2.3 — 4h (4-Hour)

**Rows**: 8,794 | **Features**: ~2.9M | **Ratio**: 330:1

#### Overfitting Risk: HIGH
- 330:1 raw ratio, but EFB (~23,000 bundles) brings effective ratio to 0.38:1 (2.6 rows per bundle). Marginal.
- num_leaves=31 with 8,794 rows = 284 rows/leaf. Viable, but deeper trees risk overfitting.
- **Verdict**: Moderate-high risk. EFB is critical. Without pre-bundling this TF would be untrainable.

#### CPU vs GPU Crossover
- 8,794 x 23,000 = 202M histogram cells/round. This is the **GPU crossover point**.
- CPU (32t): ~15-20ms/round. GPU (RTX 3090 with cuda_sparse): ~5-8ms/round.
- GPU wins 2-3x on per-round time, but CPCV fold switching has overhead.
- For Optuna: CPU (parallel trials across cores) vs GPU (single trial on GPU). CPU likely wins for Phase 1 (parallel trials on CPU faster than serial on GPU).
- **Verdict**: GPU for final CPCV training. CPU for Optuna search.

#### CPCV Groups: (10, 2)
- 80% train = 7,035 rows. Each fold = 879 rows.
- 10-fire signals get ~8 in training. Matches min_data_in_leaf=8 exactly. Tight.
- (5, 2) would give 60% train = 5,276. Better for signal preservation but 45% fewer training rows.
- **Verdict**: (10, 2) is correct. 879-row folds are large enough for reliable OOS estimation.

#### Optimal Parameters
| Parameter | Current | Optimal | Rationale |
|-----------|---------|---------|-----------|
| min_data_in_leaf | 8 | **8** | At the boundary. 10-fire at 80% train = 8 in train. Matches exactly. |
| num_leaves | 31 | **31** | 8,794 / 31 = 284 rows/leaf. Appropriate for this data size. |
| force_col_wise | True | **True** | 8,794 / 23,000 = 0.38 rows per bundle. Col-wise is correct (high feature count, moderate rows). |

#### Training Time Bottleneck
- **Histogram construction** starts to dominate. 202M cells/round x 800 rounds = 162B operations.
- CPU: 800 x 20ms = 16 seconds of pure histogram time per model.
- CPCV 45 paths (sampled to 30) x 16s = 480s = 8 minutes of histogram time.
- GPU (cuda_sparse): 800 x 8ms = 6.4s/model. 30 paths = 192s = 3.2 minutes.
- **Bottleneck**: Histogram construction on CPU, I/O overhead on GPU (H2D transfers for 63K per-leaf gradients).

---

### 2.4 — 1h (Hourly)

**Rows**: 90,000 | **Features**: ~5M | **Ratio**: 56:1

#### Overfitting Risk: MODERATE
- After EFB (~40,000 bundles), effective ratio = 2.25:1 (rows > bundles). Healthy territory.
- num_leaves=63 with 90,000 rows = 1,429 rows/leaf. Excellent data density.
- With Optuna 50% subsample (45,000 rows), ratio drops to 1.13:1. Still viable.
- **Verdict**: Low risk after EFB. This is the sweet spot for the matrix approach.

#### CPU vs GPU Crossover
- 90,000 x 40,000 = 3.6B histogram cells/round. **GPU strongly favored**.
- CPU (32t): ~200-300ms/round. GPU (RTX 3090): ~30-50ms/round. 5-8x speedup.
- Multi-GPU (8x A100): ~5-8ms/round (fold-parallel CPCV, one GPU per fold).
- **Verdict**: GPU mandatory for reasonable training time. Multi-GPU ideal.

#### CPCV Groups: (10, 2)
- 80% train = 72,000 rows. Each fold = 9,000 rows. Excellent fold sizes.
- 10-fire signals: 8 in training. Still tight at min_data_in_leaf=8.
- **Verdict**: (10, 2) is optimal. Folds are large enough, train fraction sufficient.

#### Optimal Parameters
| Parameter | Current | Optimal | Rationale |
|-----------|---------|---------|-----------|
| min_data_in_leaf | 8 | **8** | With 90K rows and 10-fire rare signals, this is the correct floor. |
| num_leaves | 63 | **63-127** | 90,000 / 63 = 1,429 rows/leaf. Could push to 127 (709 rows/leaf) if Optuna finds benefit. |
| Optuna subsample | 0.50 | **0.50** | 45,000 rows is sufficient for hyperparameter search. |
| force_col_wise | True | **True** | 90,000 / 40,000 = 2.25. Borderline. Col-wise still better (high bundle count). |

#### Training Time Bottleneck
- **Histogram construction dominates entirely**. 3.6B cells/round.
- CPU: 800 rounds x 300ms = 240 seconds per model.
- CPCV 30 paths x 240s = 7,200s = 2 hours (CPU).
- GPU (3090): 800 x 50ms = 40s/model. 30 paths = 1,200s = 20 minutes (single GPU).
- Multi-GPU (8x): 30 paths / 8 GPUs = ~4 batches x 40s = 160s = 2.7 minutes.
- **Bottleneck**: GPU histogram + H2D transfer overhead. CUDA kernel optimization (batch gradient uploads) would cut this by 3-5x.

---

### 2.5 — 15m (15-Minute)

**Rows**: 227,000 | **Features**: ~10M | **Ratio**: 44:1

#### Overfitting Risk: LOW
- After EFB (~79,000 bundles), effective ratio = 2.87:1. Healthy.
- num_leaves=127 with 227,000 rows = 1,787 rows/leaf. Excellent density.
- With Optuna 25% subsample (56,750 rows), ratio = 0.72:1. Tight but EFB preserves signal structure.
- **Verdict**: Lowest overfitting risk of all TFs. The matrix shines here.

#### CPU vs GPU Crossover
- 227,000 x 79,000 = 17.9B histogram cells/round. **GPU absolutely mandatory**.
- CPU (32t): ~1-2s/round. GPU (3090): ~100-200ms/round. 8-15x speedup.
- Multi-GPU (8x A100-80GB): ~15-25ms/round.
- **Verdict**: Multi-GPU required. Single GPU (3090) feasible but slow. 8x A100 optimal.

#### CPCV Groups: (10, 2)
- 80% train = 181,600 rows. Each fold = 22,700 rows. Excellent statistics.
- 10-fire signals: 8 in training. Still the limiting factor.
- **Verdict**: (10, 2) optimal. Could consider (15, 2) for even more train data (87% = 197K rows), but diminishing returns.

#### Optimal Parameters
| Parameter | Current | Optimal | Rationale |
|-----------|---------|---------|-----------|
| min_data_in_leaf | 8 | **8-10** | With 227K rows, could relax to 10 for slightly stronger regularization. But 8 preserves rare signals. |
| num_leaves | 127 | **127-255** | 227K / 127 = 1,787 rows/leaf. Could push to 255 (890 rows/leaf) for more expressiveness. Optuna should decide. |
| Optuna subsample | 0.25 | **0.25** | 56,750 rows is sufficient. Lower subsample saves massive time in search. |
| force_row_wise | True | **True** | 227K / 79K = 2.87. Row-wise is correct (rows > bundles). |

#### Training Time Bottleneck
- **Histogram construction overwhelmingly dominant**. 17.9B cells/round.
- CPU (32t): single model = 800 x 2s = 1,600s = 26.7 minutes.
- CPCV 30 paths x 26.7min = 800 minutes = 13.3 hours (CPU only).
- GPU (3090): 800 x 200ms = 160s/model. 30 paths = 4,800s = 80 minutes.
- Multi-GPU (8x A100): 800 x 25ms = 20s/model. 30 paths / 8 = ~4 batches x 20s = 80s = 1.3 minutes.
- **Bottleneck**: GPU memory bandwidth for sparse histogram accumulation. CUDA kernel optimizations (warp-cooperative atomics, batched H2D) critical.
- **RAM bottleneck**: ~100GB for full sparse matrix + training. Cloud only.

---

## 3. Cross-TF Comparative Analysis

### 3.1 Feature-to-Row Ratio After EFB

| TF | Raw Ratio | EFB Ratio | Risk Level | Mitigation |
|----|-----------|-----------|------------|------------|
| 1w | 518:1 | 4.1:1 | Extreme | Shallow trees (7 leaves), CPCV (5,2), aggressive ES |
| 1d | 175:1 | 1.4:1 | High | Moderate depth (15 leaves), CPCV (5,2) |
| 4h | 330:1 | 2.6:1 | High | Standard depth (31 leaves), CPCV (10,2) |
| 1h | 56:1 | 0.44:1 | Low | Deep trees viable (63 leaves), CPCV (10,2) |
| 15m | 44:1 | 0.35:1 | Lowest | Deepest trees (127 leaves), most data |

### 3.2 Where CPU vs GPU Crosses Over

| TF | Histogram Cells/Round | CPU ms/round | GPU ms/round | Winner | Speedup |
|----|----------------------|-------------|-------------|--------|---------|
| 1w | 5.4M | <1 | ~2 (overhead) | **CPU** | N/A |
| 1d | 45.9M | 3-5 | ~3 (near parity) | **CPU** | 1x |
| 4h | 202M | 15-20 | 5-8 | **GPU** | 2-3x |
| 1h | 3.6B | 200-300 | 30-50 | **GPU** | 5-8x |
| 15m | 17.9B | 1,000-2,000 | 100-200 | **GPU** | 8-15x |

**Crossover**: ~100-200M histogram cells/round (4h territory).

### 3.3 CPCV Grouping Appropriateness

| TF | Groups | Paths | Train % | Train Rows | Rows/Fold | 10-Fire in Train | Assessment |
|----|--------|-------|---------|------------|-----------|-----------------|------------|
| 1w | (5,2) | 10 | 60% | 695 | 232 | ~6 | Correct. More groups = too few rows/fold |
| 1d | (5,2) | 10 | 60% | 3,440 | 1,147 | ~6 | Correct. Could push to (7,2) |
| 4h | (10,2) | 45->30 | 80% | 7,035 | 879 | ~8 | Correct. Good balance |
| 1h | (10,2) | 45->30 | 80% | 72,000 | 9,000 | ~8 | Correct. Excellent fold sizes |
| 15m | (10,2) | 45->30 | 80% | 181,600 | 22,700 | ~8 | Correct. Could push to (15,2) |

**Key insight**: Rare signal fires (10-20x total) remain the binding constraint across ALL TFs. Even at 15m with 227K rows, a 10-fire signal only appears ~8x in 80% of data.

---

## 4. ETA Table — Full Pipeline Per Step

### Reference: Pipeline Steps
1. **Data Load** — Load parquet + NPZ from disk
2. **Cross Gen** — Sparse matmul (sparse-dot-mkl) if NPZ not cached
3. **EFB Build** — Pre-bundle binary features into EFB groups
4. **Optuna P1** — 25-30 trials, 2-fold CPCV, fast LR (0.15), 60 rounds max
5. **Optuna P2** — Top-3 validated with 4-fold CPCV, 200 rounds, LR 0.08
6. **Final Train** — Full CPCV (K=2), 800 rounds, LR 0.03, all data
7. **CPCV Full** — 30 sampled paths, confidence calibration
8. **Meta-Label** — Meta-labeling model on CPCV predictions
9. **Optimizer** — Exhaustive trade strategy parameter search
10. **PBO + Audit** — Probability of Backtest Overfitting + backtesting audit

### 4.1 — Local: 13900K + RTX 3090 + 64GB RAM

| Step | 1w | 1d | 4h | 1h | 15m |
|------|----|----|----|----|-----|
| 1. Data Load | 2s | 5s | 15s | 45s | N/A (OOM) |
| 2. Cross Gen | cached | cached | cached | 8-12min | N/A |
| 3. EFB Build | 5s | 15s | 2min | 5min | N/A |
| 4. Optuna P1 (25t x 2-fold x 60r) | 3min | 12min | 45min | 3.5hr | N/A |
| 5. Optuna P2 (3 x 4-fold x 200r) | 5min | 20min | 1.5hr | 8hr | N/A |
| 6. Final Train (800r x all data) | 8min | 30min | 2hr | 10hr | N/A |
| 7. CPCV Full (30 paths) | 15min | 1.5hr | 5hr | 20hr | N/A |
| 8. Meta-Label | 2min | 5min | 15min | 30min | N/A |
| 9. Optimizer | 10min | 30min | 1hr | 2hr | N/A |
| 10. PBO + Audit | 3min | 10min | 20min | 40min | N/A |
| **TOTAL** | **~50min** | **~3.5hr** | **~11hr** | **~45hr** | **N/A** |

**15m cannot train locally** — requires ~100GB RAM for sparse matrix + training overhead.
**1h is tight** — 38GB RAM with 64GB system. Swap may be needed. GPU (3090 24GB) can handle 15.3GB VRAM.

### 4.2 — Cloud: 8x RTX 4090 (24GB each) + EPYC 128c + 256GB RAM

| Step | 1w | 1d | 4h | 1h | 15m |
|------|----|----|----|----|-----|
| 1. Data Load | 1s | 3s | 10s | 30s | 2min |
| 2. Cross Gen | cached | cached | cached | 10min | 12min |
| 3. EFB Build | 3s | 10s | 1.5min | 4min | 15min |
| 4. Optuna P1 | 2min | 8min | 25min | 1.5hr | 3hr |
| 5. Optuna P2 | 3min | 12min | 45min | 3hr | 5hr |
| 6. Final Train | 5min | 18min | 1hr | 4hr | 10hr |
| 7. CPCV Full (30 paths, 8 GPU parallel) | 3min | 15min | 40min | 2.5hr | 6hr |
| 8. Meta-Label | 1min | 3min | 8min | 15min | 30min |
| 9. Optimizer | 5min | 15min | 30min | 1hr | 2hr |
| 10. PBO + Audit | 2min | 5min | 10min | 20min | 40min |
| **TOTAL** | **~25min** | **~1.5hr** | **~4.5hr** | **~13hr** | **~27hr** |

### 4.3 — Cloud: 8x A100-80GB + EPYC 128c + 512GB RAM

| Step | 1w | 1d | 4h | 1h | 15m |
|------|----|----|----|----|-----|
| 1. Data Load | 1s | 3s | 8s | 25s | 1.5min |
| 2. Cross Gen | cached | cached | cached | 8min | 10min |
| 3. EFB Build | 3s | 8s | 1min | 3min | 12min |
| 4. Optuna P1 | 1.5min | 6min | 18min | 1hr | 2hr |
| 5. Optuna P2 | 2min | 8min | 30min | 2hr | 3.5hr |
| 6. Final Train | 3min | 12min | 40min | 2.5hr | 6hr |
| 7. CPCV Full (30 paths, 8 GPU parallel) | 2min | 10min | 25min | 1.5hr | 3.5hr |
| 8. Meta-Label | 1min | 2min | 5min | 10min | 20min |
| 9. Optimizer | 4min | 10min | 20min | 40min | 1.5hr |
| 10. PBO + Audit | 1min | 4min | 8min | 15min | 30min |
| **TOTAL** | **~18min** | **~1hr** | **~3hr** | **~8.5hr** | **~18hr** |

### 4.4 — Cloud: 8x H100-80GB + EPYC 192c + 1TB RAM

| Step | 1w | 1d | 4h | 1h | 15m |
|------|----|----|----|----|-----|
| 1. Data Load | <1s | 2s | 5s | 15s | 1min |
| 2. Cross Gen | cached | cached | cached | 5min | 8min |
| 3. EFB Build | 2s | 5s | 45s | 2min | 8min |
| 4. Optuna P1 | 1min | 4min | 12min | 40min | 1.5hr |
| 5. Optuna P2 | 1.5min | 6min | 20min | 1.5hr | 2.5hr |
| 6. Final Train | 2min | 8min | 25min | 1.5hr | 4hr |
| 7. CPCV Full (30 paths, 8 GPU parallel) | 1.5min | 6min | 15min | 50min | 2hr |
| 8. Meta-Label | <1min | 1min | 3min | 8min | 15min |
| 9. Optimizer | 3min | 8min | 15min | 30min | 1hr |
| 10. PBO + Audit | 1min | 3min | 5min | 10min | 20min |
| **TOTAL** | **~12min** | **~40min** | **~2hr** | **~5.5hr** | **~12hr** |

---

## 5. All-TF Sequential Totals

| Hardware | 1w | 1d | 4h | 1h | 15m | **ALL 5 Sequential** |
|----------|----|----|----|----|-----|---------------------|
| Local (13900K+3090) | 50min | 3.5hr | 11hr | 45hr | N/A | **~60hr (2.5 days)** excl. 15m |
| 8x RTX 4090 | 25min | 1.5hr | 4.5hr | 13hr | 27hr | **~46hr (1.9 days)** |
| 8x A100-80GB | 18min | 1hr | 3hr | 8.5hr | 18hr | **~31hr (1.3 days)** |
| 8x H100-80GB | 12min | 40min | 2hr | 5.5hr | 12hr | **~20hr (0.8 days)** |

---

## 6. Best Cloud Machine Recommendation Per TF

| TF | Recommended | Why | Est. Cost |
|----|-------------|-----|-----------|
| 1w | **Local 13900K+3090** | 50 minutes. No cloud needed. Saves money. | $0 |
| 1d | **Local 13900K+3090** | 3.5 hours. Reasonable overnight run. | $0 |
| 4h | **1x A100-80GB + 64c** | GPU crossover. Single A100 sufficient. 11hr down to ~5hr. | ~$10 |
| 1h | **4x A100-80GB + 128c + 256GB** | Fold-parallel CPCV. 45hr local to ~12hr cloud. | ~$30 |
| 15m | **8x A100-80GB + 128c + 512GB** | Full multi-GPU. 512GB RAM mandatory. | ~$50 |

### Cost-Optimal Strategy
1. Train 1w + 1d **locally** (free, ~4.5hr combined)
2. Rent **1x A100** for 4h (~$10, ~5hr)
3. Rent **8x A100** for 1h + 15m sequentially (~$80, ~27hr combined)
4. **Total cloud spend**: ~$90 for a full retrain of all 5 TFs

### Speed-Optimal Strategy
1. Train 1w + 1d locally while renting cloud machine
2. Rent **8x H100** for 4h + 1h + 15m (~$160, ~20hr)
3. All 5 TFs complete in ~24hr (local runs in parallel with cloud setup time)

---

## 7. Unwired Optimizations (Not Yet Implemented)

These changes would dramatically cut ETAs, especially for 1h and 15m:

### 7.1 CUDA Kernel Optimizations (3-5x per-fold speedup)
- **Batch gradient uploads**: 63,000 H2D transfers/run to 1,000 (per-leaf batching)
- **Warp-cooperative atomics**: 10-40% warp efficiency to 60-80%
- **Vectorized kernel launches**: Python for-loop (48 kernels/round) to 2 kernel launches
- **CSR+CSR.T dual storage**: CUSPARSE_OPERATION_TRANSPOSE for 15m (eliminates scatter-gather)
- **Impact**: 15m 8x H100 drops from ~12hr to ~4hr

### 7.2 Optuna Phase 1 Parallelism
- Currently serial trials on GPU. Could run N_GPU parallel trials (1 per GPU).
- With 8 GPUs: 25 trials / 8 = 4 serial rounds instead of 25.
- **Impact**: Optuna P1 time / 8 for GPU-bound TFs.

### 7.3 SharedMemory CPCV (Already Implemented)
- Fold data shared via mmap between workers. No serialization overhead.
- Already coded but needs validation on cloud with large matrices.

### 7.4 Dense Conversion for Sub-79K-Bundle TFs
- For 1w (4,700 bundles) and 1d (8,000 bundles), dense training is viable and faster (no sparse overhead).
- Convert EFB output to dense numpy array, train with `device=cpu`.
- **Impact**: 1w/1d train time -20-30%.

---

## 8. Risk Matrix

| TF | Overfitting | RAM OOM | GPU OOM | Label Leakage | Signal Starvation |
|----|-------------|---------|---------|---------------|-------------------|
| 1w | HIGH (518:1 raw) | None (7GB) | None (CPU) | Fixed (purge=50) | HIGH (6 fires in train) |
| 1d | MODERATE (175:1) | None (1.6GB) | None (CPU) | Fixed (purge=90) | MODERATE (6 fires) |
| 4h | MODERATE (330:1) | Low (10.8GB) | Low (4.5GB) | Fixed (purge=72) | MODERATE (8 fires) |
| 1h | LOW (56:1) | Tight (38GB) | Moderate (15GB) | Fixed (purge=48) | LOW (8 fires) |
| 15m | LOWEST (44:1) | HIGH (100GB+) | Moderate (15GB) | Fixed (purge=24) | LOW (8 fires) |

**Critical finding**: Signal starvation (rare features with 10-20 total fires getting only 6-8 in training splits) is the universal constraint across ALL timeframes. This is inherent to CPCV with K=2. The current min_data_in_leaf=8 is the correct balance point -- lower invites noise, higher kills rare signals.

---

## 9. Recommendations Summary

1. **Train 1w and 1d locally first** -- validate the fixed pipeline end-to-end before spending on cloud
2. **4h is the GPU crossover** -- rent a single A100 when ready
3. **1h and 15m are cloud-only workloads** -- 8x A100 or 8x H100
4. **CPCV groupings are correct** -- (5,2) for 1w/1d, (10,2) for 4h/1h/15m
5. **min_data_in_leaf=8 is the universal floor** -- do not change
6. **EFB pre-bundling is the single most important optimization** -- without it, 2.9M features would require 200x more histogram computation
7. **CUDA kernel optimizations are the next multiplier** -- 3-5x speedup for 1h/15m would cut cloud costs from ~$90 to ~$30
8. **Optuna row subsampling (0.5x for 1h, 0.25x for 15m) is critical** -- saves 4-8x on Phase 1 search time with minimal quality loss
