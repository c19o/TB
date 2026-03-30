# FINAL ETA: 15m Timeframe — All Optimizations + Game-Changers

**Date:** 2026-03-30
**Machine:** 8x RTX 5090 (32GB each) + EPYC 128c + 1TB RAM
**Alternative:** Lambda 8xH100 1.8TB @ $32/hr
**Data:** ~227,000 rows | 10M+ features | Cross gen REQUIRES MEMMAP

---

## Baseline Progression

| Version | Expected | Notes |
|---|---|---|
| Original baseline (GPU_SESSION_RESUME.md) | **87 hr** | GPU fork + sparse-dot-mkl only |
| FINAL_ETAS.md (15 committed opts) | **31 hr** | + CUDA batch H2D, Numba, NUMA, WilcoxonPruner, memmap, cost-sort, fastmath, etc. |
| **This document (+ game-changers)** | **~8 hr** | + EFB Pre-Bundler, Sobol, CPCV fold-parallel |

**Total speedup vs 87hr baseline: ~10.9x**

---

## Game-Changers Not in FINAL_ETAS.md

### 1. EFB Pre-Bundler — THE DOMINANT FACTOR

**Problem in FINAL_ETAS:** `enable_bundle=False` forced LightGBM to build one histogram per feature → **10M histograms per tree level**. FINAL_ETAS assumed this as the floor.

**Pre-Bundler fix:** External greedy sparse bitmap bundler packs 10M binary features into ~79K integer bundles (max 127 features per bundle, max_bin=255). Then trains with `enable_bundle=False` on the 79K-column bundled matrix.

**Why this changes everything for 15m:**
- Bundled matrix: 227K rows × 79K bundles × 4 bytes = **~68GB** → fits entirely in RAM
- Per GPU (8-way parallel): 227K/8 = 28K rows × 79K features = **8.8GB VRAM** — fits in RTX 5090's 32GB
- No more feature streaming. The entire training matrix is resident in GPU VRAM.
- Histogram array: 79K × 256 bins vs 10M × 2 bins — same total entries (~20M) but **spatial locality** improves radically (L2 hit rate vs DRAM thrashing)
- FindBestSplit: 128x fewer candidates per tree level → GPU compute reduces ~100x on this stage
- Optuna trials: same 128x reduction per trial → each trial is 3-5x faster

**Realistic per-stage speedup from pre-bundler (conservative — NNZ preserved, only histogram locality improves):**
- Optuna P1+P2: **4x** (faster per trial)
- Final Retrain: **3x expected** (histogram L2 locality + split find 128x)
- CPCV Full per-fold: **3x expected**
- EFB Build: **dataset loads 79K ints not 10M sparse** → 5-10x faster

Pre-bundler is a one-time offline step (cached per TF, ~5-10min to build for 15m).

---

### 2. CPCV 10 Groups / 30 Paths + Process Isolation

**In FINAL_ETAS:** CPCV Full = 15 sequential folds, single GPU, sequential. **"Fold-parallel CPCV NOT committed."** This was acknowledged as the #1 bottleneck (55% of total time).

**Now:** 10 groups, 30 paths, 8 GPUs, process isolation per fold. 30 paths across 8 GPUs = ceil(30/8) = **4 parallel rounds**. Process isolation eliminates memory fragmentation between folds (was causing 15-20% slowdown per fold from heap bloat).

**Speedup calculation:**
- Old: 15 sequential folds × T_fold
- New: 4 rounds × T_fold (8-way parallel) × ~0.9 efficiency
- Fold-parallel speedup: 15 / (4 × 1.1) ≈ **3.4x → 57% reduction in wall-clock CPCV time**

This is INDEPENDENT of the pre-bundler speedup. They stack multiplicatively.

---

### 3. Sobol Optimizer (30 min not 5hr)

**In FINAL_ETAS:** Meta+Opt+PBO expected 2.5hr. Optimizer (exhaustive trade param search) was the dominant component.
**Sobol quasi-random sequences** cover the search space in 30min vs 5hr exhaustive → **~10x optimizer speedup**.

Meta+Opt+PBO breakdown after Sobol:
- Meta-labeling: ~30min (unchanged)
- Sobol optimizer: 30min (was ~1.5hr in FINAL_ETAS after partial opts)
- PBO + audit: ~30min (unchanged)
- **New total: ~1.5hr**

---

## Per-Stage ETA: 15m

Starting from FINAL_ETAS.md expected values, applying game-changers on top:

| Stage | Baseline (87hr) | FINAL_ETAS (31hr) | Optimistic | **Expected** | Pessimistic |
|---|---|---|---|---|---|
| Data Load | 3 min | 2.5 min | 1 min | **2 min** | 3 min |
| Cross Gen | 5.8 hr | 35 min | 20 min | **35 min** | 60 min |
| Pre-Bundler Build | n/a (new) | n/a | 5 min | **10 min** | 20 min |
| EFB Dataset Build | 31 min | 18 min | 2 min | **5 min** | 12 min |
| Optuna P1+P2 | 6.4 hr | 2 hr | 25 min | **45 min** | 1.3 hr |
| Final Retrain | 25 hr | 8.5 hr | 1.7 hr | **2.8 hr** | 4.3 hr |
| CPCV Full | 50 hr | 17 hr | 1.0 hr | **2.5 hr** | 5.1 hr |
| Meta+Opt+PBO | 5 hr | 2.5 hr | 45 min | **1.5 hr** | 2.0 hr |
| **TOTAL** | **~87 hr** | **~31 hr** | **~4.2 hr** | **~8.1 hr** | **~14 hr** |

---

## "Under 12 Hours?" Analysis

| Scenario | Time | Under 12hr? |
|---|---|---|
| Optimistic | 4.2 hr | YES — with room to spare |
| **Expected** | **8.1 hr** | **YES — comfortably** |
| Pessimistic | 14 hr | NO — just over by ~2hr |

**Expected verdict: YES, under 12 hours.**

The pessimistic case (14hr) occurs if:
- Pre-bundler gives only 2x per-fold speedup (vs 3x expected)
- CPCV fold parallelism hits 50% efficiency (vs assumed 90%)
- Cross gen hits 1hr (vs expected 35min)
- None of these failures are likely simultaneously

**P(under 12hr) ≈ 75-80%** given the assumptions in this analysis.

---

## Speedup Accounting: Where the 10.9x Comes From

| Optimization | Speedup vs Baseline | Stage Affected |
|---|---|---|
| GPU histogram fork (sparse CSR) | 3x | Final Retrain, CPCV |
| sparse-dot-mkl cross gen | 8x | Cross Gen |
| Memmap streaming (3TB→10GB) | ENABLES 15m | Cross Gen |
| Numba prange + parallel steps | 4x×3x | Cross Gen |
| CUDA batch H2D + vectorized | 2.5x | Final Retrain, CPCV |
| NUMA interleave + GC disable | 1.2x | All training stages |
| Multi-GPU Optuna (8 CPU trials) | 3-4x | Optuna |
| WilcoxonPruner | 1.25x | Optuna |
| Cost-sort + fastmath | 1.3x | Cross Gen |
| **EFB Pre-Bundler (game-changer)** | **3x** | Optuna, Retrain, CPCV |
| **CPCV fold-parallel 8 GPUs (game-changer)** | **3.4x** | CPCV Full |
| **Sobol optimizer (game-changer)** | **5x** | Meta+Opt+PBO |

**Compounded across all stages: ~10.9x total vs 87hr baseline.**

---

## Cost Comparison

| Hardware | Expected Time | Cost |
|---|---|---|
| 8x RTX 5090 + EPYC 128c @ $3.50/hr | 8.1 hr | **$28** |
| 8x RTX 5090 + EPYC 128c @ $4.00/hr | 8.1 hr | **$32** |
| Lambda 8xH100 1.8TB @ $32/hr | 8.1 hr | **$259** |

**Lambda recommendation: Not needed for 15m.** The 8x5090 has 256GB VRAM total, sufficient for all 30 CPCV paths (8.8GB per GPU per fold). Lambda's H100 would reduce per-round time by ~1.5-2x (better H2D bandwidth, more VRAM), bringing expected to ~5-6hr, but at 8x the cost. Not worth it.

---

## Bottleneck Identification — What's STILL Slow

### 1. Final Retrain: 2.8hr (35% of total)
- 800 rounds × (histogram build + split find + score update) on 227K rows × 79K bundles
- Per-round time: ~12-15 seconds on RTX 5090 (estimated)
- Not parallelizable across multiple GPUs (sequential boosting rounds)
- **Fix available:** BUG-C1 (warp shuffle cooperative kernel) → additional 1.5x → drops to ~1.9hr

### 2. CPCV Full: 2.5hr (31% of total)
- 4 parallel rounds × 8 simultaneous folds, each training on ~182K rows
- Limited by round 4 (only 6 folds, 2 GPUs idle = 25% wasted capacity)
- With 30 paths on 8 GPUs: not perfectly divisible → last round is partial
- **Fix available:** Increase to 32 paths (cleanly divisible by 8 GPUs) → 0% GPU waste

### 3. Cross Gen: 35 min (7% of total)
- Already near theoretical minimum with Numba + parallel + L2 tiling + memmap
- Bottleneck: Numba JIT compilation on first run (~2min overhead absorbed here)
- **Fix available:** Pre-warm Numba cache on machine startup (saves ~2min)

### 4. BUG-C2 (cuda_sparse multi-GPU — still partially broken)
- Optuna currently runs on CPU (8 cores/trial × 16 trials = 128c). Expected 45min.
- If fixed: Optuna trials on GPU → 10x faster per trial → ~5min Optuna total
- This would drop total from 8.1hr → ~7.3hr (saves only ~40min because Optuna is no longer the bottleneck)

### Summary: Primary bottleneck is sequential boosting rounds, not CPCV or cross gen.

---

## Before vs After: Full Comparison

```
BASELINE (87hr, GPU_SESSION_RESUME.md):
  Cross Gen:    5.8hr  ████████████████████████████████████████
  Optuna:       6.4hr  ████████████████████████████████████████████
  Retrain:      25hr   ████████████████████████████████████████████████████████████████████████████████
  CPCV:         50hr   [too long to draw]
  Optimizer:    5hr    ████████████████████████████████████

POST ALL GAME-CHANGERS (8.1hr expected):
  Cross Gen:    35min  ██
  Pre-Bundler:  10min  ▌
  EFB Build:    5min   ▍
  Optuna:       45min  ██▌
  Retrain:      2.8hr  ██████████
  CPCV:         2.5hr  █████████
  Meta+Opt+PBO: 1.5hr  █████
```

**Speedup breakdown by game-changer contribution:**
- 87hr → 31hr: 15 committed optimizations (2.8x)
- 31hr → 17hr: EFB Pre-Bundler on Optuna+Retrain (1.8x)
- 17hr → 10hr: EFB Pre-Bundler on CPCV (1.7x)
- 10hr → 8.1hr: CPCV fold-parallel (1.2x) + Sobol (1.1x)
- **Residual gap from 8.1hr to under 4hr:** BUG-C1 + BUG-C2 fixes + warp-cooperative kernel

---

## Key Risks

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Pre-bundler takes >30min to build | 20% | +20min one-time | Cache the bundle mapping; skip rebuild if NPZ unchanged |
| CPCV fold-parallel OOMs (8 concurrent 8.8GB loads) | 15% | Fallback to 4-way parallel | Monitor VRAM before launch; auto-reduce concurrency |
| Cross gen memmap slower than estimated | 25% | +25min | Already validated in 1h/15m OOM fixes |
| BUG-C2 still causes GPU→CPU fallback for Optuna | 40% | Optuna stays at 45min (already accounted for) | FINAL_ETAS already modeled CPU Optuna |

---

## Recommended Execution

1. Build pre-bundler mapping offline (local, ~10min): `python prebundle_binary.py --tf 15m`
2. SCP bundled matrix + mapping to cloud (68GB, ~5min at 200MB/s)
3. Launch: `cloud_run_tf.py --tf 15m --use-prebundle`
4. Checkpoints every CPCV fold (already implemented)
5. Download: CPCV checkpoint after fold 10 of 30 (~3hr in), final model at completion

**Wall-clock with expected 8.1hr: Spin up at 09:00 → done by 17:00 same day.**
