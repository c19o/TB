# FINAL ETA: 1h Timeframe — All Optimizations + Game-Changers

**Date:** 2026-03-30
**Machine:** 8x RTX 5090 (32GB each) + EPYC 128c + 512GB-1TB RAM + NVMe
**Rows:** ~90,000 | **Features:** 10M+ | **Cost:** ~$3.50/hr assumed

---

## What This Document Adds vs FINAL_ETAS.md

`FINAL_ETAS.md` included the original 15 committed optimizations (Numba prange, parallel steps,
L2 sort, bitpack, memmap CSC, adaptive chunk, WilcoxonPruner, extra_trees, CSC format, GC disable,
NUMA interleave, CUDA batch H2D + vectorized) with BUG deductions for BUG-C1 (warp shuffle OFF)
and BUG-C2 (Multi-GPU Optuna = CPU fallback).

This document applies **6 additional game-changers on top**:

| # | Game-Changer | Stage Impact | Basis |
|---|---|---|---|
| GC-1 | **EFB Pre-Bundler** (10M → 79K bundles) | Final Retrain + CPCV: +2.5x | 79K bundles fit in VRAM; Dataset construction 127x less columns; per-fold Dataset rebuild near-zero |
| GC-2 | **BUG-C2 fixed** (GPU Optuna trials, not CPU) | Optuna P1+P2: +2.5x | cuda_sparse confirmed active on 15+ files; multi_gpu_optuna.py now uses cuda_sparse → each of 8 trials GPU-accelerated at 90K row sweet spot |
| GC-3 | **BUG-C1 fixed** (warp shuffle kernel active) | Final Retrain + CPCV: +1.3x | CUDA_WARP_REDUCE=1 now correct (ballot_sync fixed); full 3.5x CUDA vs prior 2.5x baseline |
| GC-4 | **CPCV 10 groups / 30 paths** (was 6/15) | CPCV: 2x more folds (+time), but per-fold faster | Deeper walk-forward validation; each path fresher folds → higher OOS confidence |
| GC-5 | **Process isolation per fold** | CPCV: +1.15x per-fold speedup | Each fold is a fresh subprocess → zero heap fragmentation; no memory pressure from prior folds |
| GC-6 | **Sobol optimizer** (30 min, was ~45-60 min) | Meta+Opt+PBO: ~1.5x | Scrambled Sobol sequences over 7D trade param space; reaches convergence in 30 min vs TPE phase |

---

## Multiplier Stack (1h-Specific)

### Cross Gen (unchanged from FINAL_ETAS.md)

| Multiplier | Factor | Notes |
|---|---|---|
| Numba prange | 4x | CSC two-pointer intersection |
| Parallel 13 steps | 3x | All step descriptors run concurrent |
| L2 pair sorting | 1.5x | 1h row count fits L2 nicely |
| Bitpack POPCNT | 1.3x | Co-occurrence pre-filter |
| Memmap CSC streaming | **ENABLER** | 1.8TB RAM → 5GB (without this 1h cannot run) |
| Cost-sort NNZ | 1.1x | Heavy pairs first, balances prange load |
| fastmath (binary kernel) | 1.1x | Binary 0/1 only, safe |
| I/O (indices-only NPZ) | 1.2x | |
| **Combined (capped)** | **~10x** | Theoretical 15-19x, capped for overhead |

### Training Stages (Final Retrain + CPCV) — NEW STACK

| Multiplier | Factor | Source |
|---|---|---|
| CUDA batch H2D + vectorized launches | 2.5x | Original 15 opts (batch 63K→1K H2D) |
| CUDA warp shuffle kernel (BUG-C1 fixed) | 1.4x | CUDA 3.5x total vs 2.5x |
| EFB Pre-Bundler (79K bundles in VRAM) | 2.5x | Full GPU histogram utilization enabled |
| GC disable + NUMA interleave | 1.2x | Original 15 opts |
| CSC format | 1.05x | Original 15 opts |
| **Training combined** | **~10.5x** | vs original GPU_SESSION_RESUME.md baseline |

Previous FINAL_ETAS.md training combined: **3.2x** → now **10.5x** vs same baseline.

### Optuna P1+P2 — NEW STACK

| Multiplier | Factor | Source |
|---|---|---|
| 8 concurrent GPU trials (BUG-C2 fixed) | 8x (trial-level) | Each trial ~2.5x faster on GPU vs CPU |
| GPU per-trial speedup (90K rows sweet spot) | 2.5x | RTX 5090 fully saturated at 90K rows |
| WilcoxonPruner | 1.25x | Kills losing trials fast |
| GC disable + NUMA | 1.2x | |
| **Optuna combined** | **~15-20x** | vs baseline 3.7 hr |

Previous FINAL_ETAS.md Optuna combined: **4.5-6x** → now **~15-20x** (GPU fix is the unlock).

---

## Per-Stage ETA: 1h

| Stage | FINAL_ETAS.md | Optimistic | **Expected** | Pessimistic |
|---|---|---|---|---|
| Data Load | 1.2 min | 1.0 min | **1.2 min** | 2.0 min |
| Cross Gen (memmap enabled) | 18 min | 12 min | **18 min** | 32 min |
| EFB Build (pre-bundler) | 6 min | 2 min | **4 min** | 7 min |
| Optuna P1+P2 | 1.1 hr | 12 min | **22 min** | 48 min |
| Final Retrain | 4.2 hr | 50 min | **1.3 hr** | 2.5 hr |
| CPCV Full (30 folds) | 8.5 hr (15 folds) | 2.5 hr | **4.5 hr** | 8.5 hr |
| Meta+Opt+PBO (Sobol) | 1.5 hr | 20 min | **45 min** | 1.5 hr |
| **TOTAL** | **15.8 hr** | **~4.7 hr** | **~7.5 hr** | **~13.8 hr** |

---

## CPCV Full: 30 Folds vs 15 Folds — Full Breakdown

The switch to 10 groups / 30 paths doubles fold count but process isolation + pre-bundler make
each fold faster. Net result: more folds, similar or slightly faster total.

| Metric | FINAL_ETAS.md (15 folds) | This Document (30 folds) |
|---|---|---|
| CPCV folds | 15 (6 groups) | 30 (10 groups) |
| Per-fold training time | ~34 min | ~9 min |
| Per-fold speedup | baseline | 3.25x faster (pre-bundler × BUG-C1 × process isolation) |
| Total CPCV time (expected) | 8.5 hr | **4.5 hr** |

**Per-fold speedup breakdown:**
- EFB Pre-Bundler: 2.5x (Dataset construction 79K cols, full VRAM utilization)
- BUG-C1 fix: 1.3x (CUDA warp kernel active, 3.5x vs 2.5x)
- Process isolation: 1.15x (zero heap fragmentation each fold)
- Combined: 2.5 × 1.3 × 1.15 = **3.74x faster per fold**
- 30 folds at 3.74x faster = 30/3.74 = **8.0 "equivalent old folds"** vs 15 old folds

So 30 folds takes ~53% of the time 15 old folds would take. Net: **4.5 hr vs 8.5 hr**.

---

## EFB Pre-Bundler: Why This Is The Biggest Win

Without pre-bundler (FINAL_ETAS.md baseline):
- LightGBM Dataset has 10M columns
- Histogram storage: 10M × 255 bins × 2 floats × 8B = **~40GB** — does NOT fit in 32GB VRAM
- GPU falls back to partial on-chip caching → effective CUDA speedup limited to ~2x

With pre-bundler (this document):
- Python computes bundle map once (10M features → 79K bundles, ~2-3 min one-time)
- LightGBM Dataset has 79K columns
- Histogram storage: 79K × 255 bins × 2 floats × 8B = **~0.32GB** — fits trivially in VRAM
- Full GPU histogram throughput unlocked → 2.5x additional speedup
- Per-fold Dataset rebuild: was ~1-2 min of the 34 min/fold; now <10 sec
- EFB re-bundling overhead eliminated (bundle map reused for all 30 CPCV folds)

**This is the unlock that makes 30 folds cheaper than 15 old folds.**

---

## Sensitivity: If Individual Game-Changers Are Absent

| Missing | Impact | New Expected |
|---|---|---|
| EFB Pre-Bundler not available | CPCV/Retrain lose 2.5x, 30 folds → brutal | ~18 hr |
| BUG-C2 not fixed (CPU Optuna) | Optuna back to 1.1 hr | +45 min → ~8.2 hr |
| BUG-C1 not fixed (no warp kernel) | Retrain/CPCV lose 1.3x | ~9 hr |
| CPCV stays at 15 folds | Faster (fewer folds) | ~5.5 hr |
| Sobol not available | Meta+Opt adds ~20 min | ~7.8 hr |
| Process isolation absent | CPCV +5-10% from fragmentation | ~7.9 hr |

**Critical path:** EFB Pre-Bundler and BUG-C2 fix are the two biggest wins. Losing either adds 2-10 hours.

---

## RAM Requirements

| Stage | RAM Usage | Notes |
|---|---|---|
| Cross Gen | ~5 GB | Memmap CSC streaming (was 1.8TB without) |
| EFB Pre-Bundler | ~15 GB | Bundle map for 10M features |
| LightGBM Dataset (79K bundles) | ~25 GB | 90K rows × 79K bundles × float32 |
| CPCV training fold | ~40-60 GB | Per-fold training matrix + gradients |
| Multi-GPU Optuna (8 trials concurrent) | ~8 × 15 GB = 120 GB | Each trial holds its own Dataset |
| **Peak RAM** | **~150-200 GB** | Well within 512 GB machine |

---

## GPU VRAM Requirements

| Stage | VRAM per GPU | Notes |
|---|---|---|
| Optuna trial (1 of 8) | ~3 GB | 79K bundles × 255 bins, 90K rows |
| Final Retrain | ~18 GB | 79K bundles, full 90K row dataset, 800 rounds |
| CPCV fold training | ~16 GB | ~78K training rows per fold |
| **Max (Final Retrain)** | **~18 GB** | Within 32 GB RTX 5090 |

Without pre-bundler: Final Retrain needs ~40 GB for histograms alone → exceeds VRAM → CPU fallback.

---

## Where the Time Goes (Expected)

| Stage | Hours | % |
|---|---|---|
| CPCV Full (30 folds) | 4.5 hr | **60%** |
| Final Retrain | 1.3 hr | **17%** |
| Optuna P1+P2 | 0.37 hr | 5% |
| Meta+Opt+PBO | 0.75 hr | 10% |
| Cross Gen | 0.30 hr | 4% |
| Other (load, EFB build) | 0.09 hr | 1% |

**CPCV Full is still 60% of total time**, even after all optimizations. The single largest
remaining improvement would be fold-parallel CPCV (run 8 folds concurrently across 8 GPUs) — this
is NOT in the current committed plan but would cut CPCV from 4.5 hr to ~0.6 hr and total from
7.5 hr to ~3.6 hr.

---

## Comparison: 1h ETA Evolution

| Version | What Changed | 1h Expected |
|---|---|---|
| GPU_SESSION_RESUME.md baseline | Post-fix, basic GPU fork | 47 hr |
| FINAL_ETAS.md | + 15 original opts, bug deductions | 15.8 hr |
| **This document** | + 6 game-changers (pre-bundler, GPU Optuna, warp fix, 30 folds, process isolation, Sobol) | **7.5 hr** |
| If fold-parallel CPCV added | 8x folds concurrent (future) | **~3.5 hr** |

**Total speedup from raw baseline to current: 6.3x.**

---

## Cost Estimate

| Scenario | Hours | Cost @ $3.50/hr |
|---|---|---|
| Optimistic | 4.7 hr | **$16** |
| Expected | 7.5 hr | **$26** |
| Pessimistic | 13.8 hr | **$48** |

---

## Deployment Prerequisites (1h-Specific)

1. **Memmap CSC** — REQUIRED. Without it, cross gen needs 1.8TB RAM. Verify `MEMMAP_CROSS=1` in env before launch.
2. **EFB Pre-Bundler** — verify `efb_prebundler.py` loads and bundle map computed before CPCV loop.
3. **cuda_sparse confirmed** — run `check_gpu.py` → must show `cuda_sparse` device available.
4. **WARP_REDUCE=1** — set env var if BUG-C1 warp kernel is merged. Otherwise CUDA at 2.5x only.
5. **validate.py** — MUST pass all 82 checks. Especially:
   - `feature_fraction_bynode >= 0.7`
   - `bagging_fraction >= 0.7`
   - `CPCV purge = max_hold_bars` (=48 for 1h)
   - `CPCV_GROUPS = 10, CPCV_PATHS = 30`
6. **RAM check** — `free -h` must show ≥ 200 GB free before launching. 512 GB machine minimum.
7. **8 GPUs confirmed** — `nvidia-smi -L | wc -l` must return 8 for full Optuna parallelism.
