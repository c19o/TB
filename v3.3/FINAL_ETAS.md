# FINAL ETA: All 5 Timeframes — 8x RTX 5090 + EPYC 128c

**Date:** 2026-03-30
**Machine:** 8x RTX 5090 (32GB each) + EPYC 128c + 512GB-1TB RAM + NVMe
**Cost:** ~$3.50/hr assumed

---

## Methodology

**Baseline** = GPU_SESSION_RESUME.md timings (already includes GPU histogram fork, sparse-dot-mkl cross gen, dataset reuse, round-level pruning, warm-start cascade, ES patience fix).

**New** = all 15 committed optimizations applied on top, WITH the following realistic adjustments:

### Bug Deductions (Non-Negotiable)
- **BUG-C1 (Warp Shuffle OFF):** CUDA kernel speedup reduced from 3-5x to **2-3x** (only batch H2D + vectorized 3-stream pipeline active)
- **BUG-C2 (Multi-GPU = CPU fallback):** Optuna multi-GPU trials run on CPU, not GPU. Still get 8-way trial parallelism via 128 cores (16c/trial), but per-trial speed is CPU-bound
- **Fold-parallel CPCV NOT committed:** Multi-GPU opt #12 is Optuna trial-level only. CPCV Full remains sequential single-GPU folds

### Per-Stage Multipliers Used

| Optimization | Cross Gen | Optuna P1+P2 | Final Retrain | CPCV Full | Other |
|---|---|---|---|---|---|
| Numba prange (3-8x) | **4x** | - | - | - | - |
| Parallel 13 steps | **3x** | - | - | - | - |
| L2 pair sorting | **2x** (15m), 1.5x (1h), 1x (others) | - | - | - | - |
| Bitpack POPCNT | **1.3x** | - | - | - | - |
| Memmap CSC | enables 1h/15m | - | - | - | - |
| Adaptive chunk | OOM prevention | - | - | - | - |
| NPZ checkpointing | crash recovery | - | - | - | - |
| Indices-only NPZ | **1.2x** I/O | - | - | - | - |
| CSC format | - | 1.05x | 1.05x | 1.05x | - |
| WilcoxonPruner | - | **1.25x** (kills losers) | - | - | - |
| extra_trees | - | better params | - | - | - |
| Multi-GPU Optuna | - | **3-4x** (8 CPU trials) | - | - | - |
| GC disable | - | 1.05x | 1.05x | 1.05x | - |
| NUMA interleave | - | 1.15x | 1.15x | 1.15x | 1.1x |
| CUDA kernels (2-3x) | - | N/A (CPU trials) | **2.5x** | **2.5x** | - |

**Cross gen combined:** Numba 4x * Parallel 3x * L2 * Bitpack 1.3x * I/O 1.2x = **~15-19x** theoretical, capped at **~10x expected** (memory/scheduling overhead)

**Optuna P1+P2 combined:** 8 CPU trials (3-4x) * pruner 1.25x * GC/NUMA 1.2x = **~4.5-6x**

**Final Retrain combined:** CUDA 2.5x * GC 1.05x * NUMA 1.15x * CSC 1.05x = **~3.2x**

**CPCV Full combined:** CUDA 2.5x * GC 1.05x * NUMA 1.15x * CSC 1.05x = **~3.2x** (still sequential folds)

---

## 1w — 818 rows, 2.2M features, cross gen CACHED

| Stage | Baseline | Optimistic | Expected | Pessimistic |
|---|---|---|---|---|
| Data Load | 10s | 8s | 10s | 12s |
| Cross Gen | cached | cached | cached | cached |
| EFB Build | 18s | 12s | 15s | 18s |
| Optuna P1+P2 | 18 min | 5 min | 8 min | 14 min |
| Final Retrain | 40 min | 15 min | 22 min | 32 min |
| CPCV Full | 78 min | 28 min | 38 min | 55 min |
| Meta+Opt+PBO | 30 min | 16 min | 22 min | 28 min |
| **TOTAL** | **~3.0 hr** | **~1.1 hr** | **~1.5 hr** | **~2.2 hr** |

**Notes:** 818 rows = GPU barely saturated. CUDA kernel speedup is only ~1.5-2x (not full 2.5x). Optuna trials are already fast (seconds each) so parallelism has diminishing returns. Main savings from NUMA + GC + modest CUDA.

---

## 1d — 5,733 rows, 6M features, cross gen CACHED

| Stage | Baseline | Optimistic | Expected | Pessimistic |
|---|---|---|---|---|
| Data Load | 15s | 10s | 12s | 18s |
| Cross Gen | cached | cached | cached | cached |
| EFB Build | 30s | 20s | 25s | 30s |
| Optuna P1+P2 | 60 min | 15 min | 22 min | 35 min |
| Final Retrain | 2 hr | 35 min | 50 min | 1.2 hr |
| CPCV Full | 4.1 hr | 1.2 hr | 1.7 hr | 2.5 hr |
| Meta+Opt+PBO | 1 hr | 28 min | 38 min | 50 min |
| **TOTAL** | **~8.5 hr** | **~2.6 hr** | **~3.6 hr** | **~5.0 hr** |

**Notes:** 5.7K rows gives decent GPU utilization. CUDA 2-2.5x realistic. 8-way Optuna parallelism is very effective here (each trial = ~30s on CPU with 16 cores). EFB with 6M features into ~23K bundles works well.

---

## 4h — 8,794 rows, 6-8M features, cross gen NEEDS TO RUN

| Stage | Baseline | Optimistic | Expected | Pessimistic |
|---|---|---|---|---|
| Data Load | 30s | 20s | 25s | 35s |
| Cross Gen | 36 min | 3 min | 5 min | 10 min |
| EFB Build | 4 min | 2 min | 3 min | 4 min |
| Optuna P1+P2 | 2.6 hr | 30 min | 45 min | 1.2 hr |
| Final Retrain | 5.3 hr | 1.3 hr | 1.8 hr | 3 hr |
| CPCV Full | 10.7 hr | 2.8 hr | 3.8 hr | 6 hr |
| Meta+Opt+PBO | 2 hr | 45 min | 1 hr | 1.5 hr |
| **TOTAL** | **~21.6 hr** | **~5.5 hr** | **~7.5 hr** | **~12 hr** |

**Notes:** Cross gen drops dramatically with Numba prange + parallel steps (8.8K rows is manageable in RAM). 128 cores fully utilized for cross gen. GPU well-saturated at ~9K rows. This is the sweet spot where ALL optimizations compound nicely.

---

## 1h — ~90K rows, 10M+ features, cross gen NEEDS MEMMAP

| Stage | Baseline | Optimistic | Expected | Pessimistic |
|---|---|---|---|---|
| Data Load | 1.5 min | 1 min | 1.2 min | 2 min |
| Cross Gen | 2.3 hr | 12 min | 18 min | 30 min |
| EFB Build | 9 min | 4 min | 6 min | 9 min |
| Optuna P1+P2 | 3.7 hr | 45 min | 1.1 hr | 2 hr |
| Final Retrain | 12.4 hr | 3 hr | 4.2 hr | 7 hr |
| CPCV Full | 24.9 hr | 6 hr | 8.5 hr | 14 hr |
| Meta+Opt+PBO | 3 hr | 1 hr | 1.5 hr | 2.5 hr |
| **TOTAL** | **~47 hr** | **~11 hr** | **~15.8 hr** | **~26 hr** |

**Notes:** Memmap CSC streaming is the ENABLER (baseline 1.8TB peak RAM reduced to ~5GB cross gen + ~100GB training). Cross gen with Numba + parallel + L2 tiling (1.5x for 1h) is dramatic. CPCV Full is still the bottleneck at 8.5hr expected — sequential 15 folds on single GPU. CUDA 2.5x well-utilized at 90K rows.

**RAM requirement:** ~100-150GB for EFB Dataset construction (10M features). 512GB machine handles this.

---

## 15m — ~227K rows, 10M+ features, cross gen NEEDS MEMMAP

| Stage | Baseline | Optimistic | Expected | Pessimistic |
|---|---|---|---|---|
| Data Load | 3 min | 2 min | 2.5 min | 4 min |
| Cross Gen | 5.8 hr | 20 min | 35 min | 1.2 hr |
| EFB Build | 31 min | 12 min | 18 min | 28 min |
| Optuna P1+P2 | 6.4 hr | 1.2 hr | 2 hr | 3.5 hr |
| Final Retrain | 25 hr | 6 hr | 8.5 hr | 14 hr |
| CPCV Full | 50 hr | 12 hr | 17 hr | 28 hr |
| Meta+Opt+PBO | 5 hr | 1.5 hr | 2.5 hr | 4 hr |
| **TOTAL** | **~93 hr** | **~21 hr** | **~31 hr** | **~51 hr** |

**Notes:** CPCV Full dominates EVERYTHING (55% of expected time). 15 sequential folds on single GPU, each training on ~215K rows with 10M features. CUDA 2.5x helps but can't overcome the sequential fold bottleneck. Cross gen drops from 5.8hr to ~35min thanks to Numba + parallel + L2 tiling (2x for 15m) + memmap streaming.

**RAM requirement:** ~150-200GB for EFB Dataset construction. 1TB machine recommended.

**If BUG-C2 is fixed** (cuda_sparse on multi-GPU): Optuna drops to ~1hr, CPCV could go fold-parallel → total drops to ~18-22hr expected.

---

## Summary Table

| TF | Baseline | Optimistic | Expected | Pessimistic | Speedup | Cost (Expected) |
|---|---|---|---|---|---|---|
| 1w | 3.0 hr | 1.1 hr | **1.5 hr** | 2.2 hr | 2.0x | $5 |
| 1d | 8.5 hr | 2.6 hr | **3.6 hr** | 5.0 hr | 2.4x | $13 |
| 4h | 21.6 hr | 5.5 hr | **7.5 hr** | 12 hr | 2.9x | $26 |
| 1h | 47 hr | 11 hr | **15.8 hr** | 26 hr | 3.0x | $55 |
| 15m | 93 hr | 21 hr | **31 hr** | 51 hr | 3.0x | $109 |

---

## All 5 Sequential

| Scenario | Total Hours | Total Days | Total Cost |
|---|---|---|---|
| Baseline (no new opts) | **173 hr** | **7.2 days** | $606 |
| Optimistic | **41 hr** | **1.7 days** | $144 |
| **Expected** | **59 hr** | **~2.5 days** | **$208** |
| Pessimistic | **96 hr** | **4.0 days** | $337 |

---

## Where the Time Goes (Expected, All 5)

| Stage | Hours | % of Total |
|---|---|---|
| CPCV Full | 31.0 hr | **53%** |
| Final Retrain | 15.3 hr | **26%** |
| Optuna P1+P2 | 4.9 hr | 8% |
| Cross Gen | 1.0 hr | 2% |
| Meta+Opt+PBO | 5.9 hr | 10% |
| Other (load, EFB) | 0.5 hr | 1% |

**CPCV Full is 53% of total pipeline time.** The single biggest improvement available is fold-parallel CPCV across 8 GPUs (NOT one of the 15 committed optimizations, but described in EXPERT_MULTI_GPU.md). This alone would cut CPCV by 4-8x, bringing total from 59hr to ~35-40hr.

---

## Critical Path Analysis

### What actually speeds things up (ranked by impact):
1. **CUDA batch H2D + vectorized launches** (2-3x on training) — saves ~25hr across all 5 TFs
2. **Numba prange cross gen** (4x) — saves ~8hr but only on 4h/1h/15m
3. **Parallel cross steps** (3x) — compounds with Numba, saves ~5hr
4. **Multi-GPU Optuna trials** (3-4x on Optuna stage) — saves ~8hr total
5. **NUMA interleave + GC disable** (1.2x combined) — saves ~12hr across all stages

### What's blocked by bugs:
- **BUG-C1 fix (warp shuffle):** Would add ~1.5x on top of current CUDA → saves ~15hr more
- **BUG-C2 fix (cuda_sparse multi-GPU):** Would make multi-GPU run on actual GPUs → Optuna 2x faster + enables fold-parallel CPCV → saves ~20-25hr

### If BOTH bugs are fixed:
| Scenario | Expected Total |
|---|---|
| Current (bugs present) | **59 hr** |
| BUG-C2 fixed only | **~45 hr** |
| Both C1 + C2 fixed | **~35 hr (1.5 days)** |
| Both fixed + fold-parallel CPCV | **~22 hr (~1 day)** |

---

## Recommended Execution Order

1. **1w** (1.5hr) — validate full pipeline end-to-end
2. **1d** (3.6hr) — compare warm-start vs cold
3. **4h** (7.5hr) — first TF with live cross gen
4. **1h** (15.8hr) — memmap stress test
5. **15m** (31hr) — longest, run overnight

**Parallel opportunity:** 1w + 1d can run on different GPUs simultaneously (different VRAM requirements). Save ~1.5hr.

**Total wall-clock with 1w||1d:** ~57.5 hr expected, ~2.4 days.

---

## Cost Comparison

| Approach | Hours | Cost @ $3.50/hr |
|---|---|---|
| No optimizations (old baseline) | 165 hr | $578 |
| With 15 optimizations (expected) | 59 hr | $208 |
| With bug fixes + fold-parallel | 22 hr | $77 |

**The 15 optimizations save ~$370 vs baseline.** Fixing BUG-C2 + adding fold-parallel CPCV would save another ~$130 on top.
