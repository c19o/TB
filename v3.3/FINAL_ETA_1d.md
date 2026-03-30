# FINAL ETA: 1d Timeframe — All Optimizations + Game-Changers + Bug Fixes

**Date:** 2026-03-30
**Machine:** 8x RTX 5090 (32GB each) + EPYC 128c + 512GB-1TB RAM + NVMe
**Dataset:** 5,733 rows × 6M features
**Cross Gen:** CACHED (skip)
**Cost:** ~$3.50/hr assumed

---

## 1d Characteristics

| Property | Value | Impact |
|---|---|---|
| Rows | 5,733 | GPU ~50-60% saturated (not small enough to be pure CPU, not large enough to max GPU) |
| Features (raw) | ~6M | Standard 1d feature set |
| EFB bundles | ~47K | 127x compression from 6M raw features |
| Cross gen | CACHED | Skip entirely — no time penalty |
| CPCV config | 10 groups × C(10,2)=45 paths, sample 30 | 67% of exhaustive — balanced coverage vs speed |
| Multi-GPU | Marginal for training | Effective for 8-way Optuna trial parallelism (CPU workers) |

**Key insight:** 5,733 rows is the sweet spot where GPU histogram kernels are partially saturated. CUDA 2-2.5x speedup (not full 2.5x). Optuna benefits strongly from 8-way CPU trial parallelism since each trial takes ~30-45s on 16 cores — 8 concurrent trials compress P1+P2 dramatically.

---

## Optimization Stack Applied

### Cross Gen Stage (CACHED — N/A)
All Numba prange, parallel 13-step, L2 pair sorting, bitpack POPCNT, memmap CSC, adaptive chunk, NPZ checkpointing optimizations are irrelevant — cross gen result already on disk.

### EFB Dataset Construction
- **6M → ~47K bundles:** 127x feature count reduction for histogram bins
- **EFB build time:** ~25-30s wall-clock (linear in NNZ of CSR, not feature count)
- Note: FINAL_ETAS.md used ~23K bundle estimate; 47K bundles adds ~10-15% to per-round training time vs that estimate. Accounted for below.

### Optuna Phase 1 + Phase 2
- **Multi-GPU Optuna (8 CPU workers, 16c/trial):** Each P1 trial ~35s on CPU → 8 parallel = ~4.4x compression
- **WilcoxonPruner:** Kills bad trials early (~1.25x pruning speedup)
- **GC disable + NUMA interleave:** ~1.2x combined
- **Extra_trees sampler:** Better hyperparameter space coverage in fewer trials
- **Sobol optimizer:** Quasi-random initial sampling → faster convergence
- **Combined P1+P2 multiplier:** ~5x vs baseline

### Final Retrain
- **CUDA batch H2D + vectorized 3-stream:** 2-2.5x at 5.7K rows (partially saturated)
- **47K bundles vs 23K:** ~1.12x slower per round vs FINAL_ETAS estimate (more histogram slots)
- **GC disable + NUMA:** 1.2x
- **CSC format:** 1.05x
- **Net combined multiplier:** ~2.6x vs baseline (was 3.2x in FINAL_ETAS before 47K bundle adjustment)

### CPCV Full (10 groups, 30 sampled paths = 30 sequential folds)
- **Note:** FINAL_ETAS used 15 folds. 30 sampled paths = 2x more folds vs baseline 15-fold config.
- **30 folds (67% of C(10,2)=45) vs 15-fold baseline:** ~2x more folds to run
- **CUDA 2-2.5x per fold:** Partially offsets the 2x fold count increase
- **Net vs FINAL_ETAS CPCV:** ~1.2-1.4x longer (more folds, partially offset by CUDA)
- **GC disable + NUMA:** 1.2x savings still apply

### Meta-labeling + Trade Optimizer + PBO
- **Sortino fixed:** Correct risk-adjusted metric (no functional speedup, correctness fix)
- **CPCV purge fixed:** Correct purge = max_hold_bars (no speed change)
- **Sobol optimizer:** ~1.3x faster convergence in trade param search
- **NUMA + GC:** 1.15x

---

## Stage-by-Stage Breakdown

| Stage | Baseline | Optimistic | **Expected** | Pessimistic | Key Driver |
|---|---|---|---|---|---|
| Data Load | 15s | 10s | **12s** | 18s | NVMe sequential read |
| Cross Gen | cached | cached | **cached** | cached | Already on disk |
| EFB Build | 30s | 20s | **28s** | 35s | 47K bundles (slight overhead vs 23K) |
| Optuna P1+P2 | 60 min | 14 min | **20 min** | 32 min | 8-way CPU parallel × WilcoxonPruner |
| Final Retrain | 2 hr | 38 min | **55 min** | 1.3 hr | CUDA 2.3x × 47K bundle adj |
| CPCV Full | 4.1 hr | 1.4 hr | **2.1 hr** | 3.2 hr | 30 folds × CUDA 2.3x (net ~1.35x vs baseline 15-fold) |
| Meta+Opt+PBO | 1 hr | 28 min | **40 min** | 55 min | Sobol optimizer + NUMA |
| **TOTAL** | **~8.5 hr** | **~2.7 hr** | **~4.1 hr** | **~5.9 hr** |  |

### Adjustment vs FINAL_ETAS.md (3.6 hr expected)
FINAL_ETAS assumed:
- ~23K EFB bundles (faster per-round training)
- 15 CPCV folds

This file uses:
- ~47K EFB bundles → +10-15% training/CPCV time
- 30 CPCV paths (67% of exhaustive) → more folds, partially offset by CUDA

**Net adjustment: +0.5 hr expected** (4.1 hr vs 3.6 hr)

---

## CPCV Detail: 10 Groups × 30 Sampled Paths

| Config | Paths | Folds Run | Coverage | Time vs 15-fold |
|---|---|---|---|---|
| Baseline (5 groups, C(5,2)=10 × 2) | 20 | 20 | ~44% of C(10,2) exhaustive | 1.0x |
| FINAL_ETAS assumption | 15 | 15 | — | — |
| **This spec (10 groups, 30 of 45)** | **30** | **30** | **67% of exhaustive** | **~2x fold count** |
| Exhaustive (all 45 paths) | 45 | 45 | 100% | ~3x fold count |

**At 5,733 rows:** Each CPCV fold trains on ~5,160 rows (90%), validates on ~573 rows (10%). Per fold: ~4.2 min expected (CUDA-accelerated). 30 folds = ~2.1 hr total CPCV.

---

## GPU Utilization at 5,733 Rows

| Stage | GPU Util | Notes |
|---|---|---|
| Optuna P1+P2 | ~15-25% | CPU-bound (8 trials on 16c each) — GPU mostly idle |
| Final Retrain | ~50-60% | Histogram kernel partially saturated at 5.7K rows |
| CPCV folds | ~50-60% | Same as retrain — small batch per fold |
| Meta/Opt | ~5-10% | CPU-bound optimizer, occasional GPU inference |

**Multi-GPU note:** Only 1 GPU used for training (single LightGBM instance). Other 7 GPUs handle concurrent Optuna trials (CPU-side, GPU not used per trial). Multi-GPU DOES NOT accelerate per-fold training for 1d — dataset too small to benefit from model parallelism.

---

## Bug Status Impact on 1d

| Bug | Status | 1d Impact |
|---|---|---|
| BUG-C1 (Warp Shuffle OFF) | Present | CUDA is 2-2.5x instead of 3-4x. Costs ~25 min on expected |
| BUG-C2 (Multi-GPU CPU fallback) | Present | Optuna trials run on CPU. At 1d scale (fast trials), limited impact. Costs ~10 min |
| CPCV purge fixed | FIXED | Correct purge = max_hold_bars (correctness, not speed) |
| feature_fraction=0.9 | FIXED | All bundles eligible per tree |
| All 12 signal-killing bugs | FIXED | Correctness gain. No speed impact |

**If BUG-C1 fixed:** CUDA 3x → saves ~18 min on Final Retrain + ~28 min on CPCV = ~46 min savings → expected drops to **~3.2 hr**
**If both C1+C2 fixed:** Additional ~10 min Optuna savings → expected **~3.1 hr**

---

## Summary

| Scenario | Total | vs Baseline |
|---|---|---|
| Baseline (no opts, 15-fold CPCV) | 8.5 hr | 1.0x |
| Optimistic (all opts, ideal conditions) | **2.7 hr** | 3.1x faster |
| **Expected (realistic, 30-fold CPCV)** | **4.1 hr** | **2.1x faster** |
| Pessimistic (contention, memory pressure) | 5.9 hr | 1.4x faster |
| If BUG-C1+C2 fixed | ~3.1 hr | 2.7x faster |

**Expected wall-clock: 4.1 hours**
**Expected cost: ~$14 @ $3.50/hr**
**Bottleneck: CPCV Full (51% of total time, 30 sequential folds)**

---

## Execution Notes

1. **Validate first:** `python validate.py` — must pass 82 checks before starting
2. **EFB build is fast (~28s):** 47K bundles constructed once, reused across all Optuna trials and CPCV folds (Dataset reuse pattern)
3. **Optuna P1:** 25 trials, 8 concurrent CPU workers, WilcoxonPruner active — expect ~10-12 min
4. **Optuna P2:** Top 3 configs, 4-fold quick CPCV each — expect ~8 min
5. **CPCV Full:** 30 paths, single GPU, sequential — this is the wall. No way to accelerate without fold-parallel multi-GPU (not committed)
6. **Download checkpoint:** After Final Retrain (model artifact), after CPCV Full (confidence calibration), after Optimizer (trade params)
7. **1d can run locally** (13900K + RTX 3090 + 64GB RAM) — ~1.6GB RAM, ~0.85GB VRAM. Cloud machine overkill but correct if batching all 5 TFs.
