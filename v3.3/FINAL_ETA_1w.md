# FINAL ETA: 1w Timeframe — All Optimizations + Game-Changers

**Date:** 2026-03-30
**Machine:** 8x RTX 5090 (32GB each) + EPYC 128c + 512GB-1TB RAM
**Data:** 818 rows, 2.2M features, cross gen CACHED
**Rate:** ~$3.50/hr

---

## Per-Stage Breakdown

| Stage | Baseline¹ | Post-15-opts² | Optimistic | **Expected** | Pessimistic |
|---|---|---|---|---|---|
| Data Load | 10s | 10s | 8s | **10s** | 15s |
| Cross Gen | cached | cached | cached | **cached** | cached |
| EFB Build | 18s | 15s | 2s | **4s** | 8s |
| Optuna P1+P2 | 18 min | 8 min | 3 min | **5 min** | 9 min |
| Final Retrain | 40 min | 22 min | 7 min | **13 min** | 20 min |
| CPCV Full | 78 min | 38 min | 10 min | **17 min** | 26 min |
| Meta+Opt+PBO | 30 min | 22 min | 4 min | **11 min** | 17 min |
| **TOTAL** | **~3.0 hr** | **~1.5 hr** | **~25 min** | **~47 min** | **~73 min** |

¹ GPU_SESSION_RESUME.md baseline (GPU fork only, no wave optimizations)
² FINAL_ETAS.md expected (15 committed optimizations applied)

---

## Game-Changer Deltas Applied on Top of Post-15-opts

### 1. EFB Pre-Bundler: 2.2M → ~17K bundles (128x reduction)

**Impact on 1w:** LARGEST single game-changer for this TF.

At 818 rows, histogram computation dominates training time:
- O(rounds × nodes × rows × features_sampled) = 800 × 31 × 818 × 2.2M × 0.9 ≈ 40T ops
- With 17K bundles: 800 × 31 × 818 × 17K × 0.9 ≈ 310B ops — **128x reduction**

Dataset construction in each CPCV fold rebuilds from scratch. 128x fewer features = 128x less bin construction work. At 818 rows the compute is trivial; it's the memory bandwidth across 2.2M feature columns that was the bottleneck.

| Stage | Post-15-opts | After EFB Pre-Bundler |
|---|---|---|
| EFB Build | 15s | ~4s (pre-bundler does upfront work, Dataset sees 17K not 2.2M) |
| Optuna P1+P2 | 8 min | ~6 min (faster per-trial Dataset construction) |
| Final Retrain | 22 min | ~14 min (~1.6x at 818 rows — compute trivial, overhead dominant) |
| CPCV Full | 38 min | ~26 min (~1.5x — Dataset rebuild per fold is faster) |

### 2. CPCV: 10 groups, 30 sampled paths (was 15 groups, 105 exhaustive)

**Impact:** Fewer model trainings in CPCV.

- Old: 15 groups → 15 sequential model trainings (fold holdouts), 105 combinatorial equity curves
- New: 10 groups → 10 sequential model trainings, 30 sampled backtest paths

Speedup = 10/15 = **1.5x** on CPCV training (the expensive part).

After EFB bundler (26 min) → after CPCV restructure: 26 × (10/15) = **~17 min**

### 3. Sobol Trade Optimizer (was exhaustive grid)

**Impact on Meta+Opt+PBO:** The exhaustive grid portion (~10 min of the 22 min) is replaced by quasi-random Sobol sampling. Sobol with 200 points vs grid with 8,000+ evaluations = **40x fewer evaluations**.

- Optimizer sub-stage: 10 min → ~1 min
- Meta-labeling: ~8 min (unchanged)
- PBO: ~4 min (unchanged, already fast)
- EFB speedup on inference (minor): saves ~1 min
- **New Meta+Opt+PBO total: ~11 min**

### 4. Optuna Params Fixed (bynode=0.7, bagging=0.7, bin_construct=5000)

- `bin_construct=5000`: higher-resolution bins, minor training cost increase absorbed by EFB bundler
- `bagging=0.7`, `bynode=0.7`: negligible speed effect at 818 rows
- Net effect: ~0 on timing, positive on accuracy

### 5. WilcoxonPruner (warmup=120, patience=10)

Already captured in FINAL_ETAS.md (1.25x pruner speedup in Optuna).

### 6. Multi-GPU: cuda_sparse fixed (requires 2+ GPUs)

**NOT USED for 1w.** 818 rows = GPU launch overhead > compute. Single RTX 5090 is the right choice. Multi-GPU adds synchronization overhead with negligible throughput gain at this row count.

The other 7 GPUs are idle during 1w training — consider running 1d simultaneously on a separate GPU.

### 7. CUDA: Batch H2D + Vectorized Launches

Already in post-15-opts baseline. At 818 rows this gave ~1.5x (not full 2.5x — GPU barely saturated).

### 8. THP=madvise, Cost-Sorted Pairs, Fastmath

Minor (1.05-1.1x). Already in post-15-opts baseline.

---

## Summary of Multipliers Applied

| Game-Changer | Stage Affected | Multiplier |
|---|---|---|
| EFB Pre-Bundler | EFB Build | ~4x faster |
| EFB Pre-Bundler | Final Retrain | ~1.6x faster |
| EFB Pre-Bundler | CPCV Full | ~1.5x faster |
| EFB Pre-Bundler | Optuna P1+P2 | ~1.3x faster |
| CPCV 10g/30p | CPCV Full | 1.5x faster (10 vs 15 models) |
| Sobol Optimizer | Meta+Opt+PBO | ~2.5x faster (optimizer sub-stage ×40, others unchanged) |
| Multi-GPU | — | No benefit at 818 rows |

**Combined CPCV speedup from baseline:** 78 min → 38 min (15-opts) → 26 min (EFB) → 17 min (CPCV restructure) = **4.6x total**

---

## Cost

| Scenario | Time | Cost @ $3.50/hr |
|---|---|---|
| Optimistic | 25 min | **$1.46** |
| **Expected** | **47 min** | **$2.74** |
| Pessimistic | 73 min | $4.26 |

---

## Bottleneck Analysis

| Stage | Expected Time | % of Total |
|---|---|---|
| CPCV Full | 17 min | **36%** |
| Final Retrain | 13 min | **28%** |
| Meta+Opt+PBO | 11 min | 23% |
| Optuna P1+P2 | 5 min | 11% |
| Load + EFB | ~14s | <1% |

**Primary bottleneck:** CPCV Full (sequential fold training, 10 models).

**Why GPU barely helps at 818 rows:**
- At 818 rows, gradient histograms fill in microseconds — GPU kernel launch overhead (~10-50μs) is comparable to compute time per kernel
- Memory bandwidth (2.2M → 17K bundles) was the real constraint, fixed by EFB pre-bundler
- GPU helps most at 5K+ rows; 1w should almost certainly train faster on CPU (EPYC 128c) vs GPU for Optuna, and single GPU for final retrain/CPCV is marginal

**Recommendation:** Run Optuna on CPU (128c / 8 cores per trial = 16 parallel trials). GPU only for Final Retrain + CPCV. If single GPU is slower than CPU at 818 rows, consider CPU-only for all stages — test both on trial 1.

---

## Confidence Assessment

| Scenario | Confidence | Key Assumption |
|---|---|---|
| Optimistic (25 min) | 20% | All multipliers stack, no overhead, GPU runs efficiently |
| **Expected (47 min)** | **60%** | EFB pre-bundler delivers ~1.5x, Sobol delivers savings, no surprises |
| Pessimistic (73 min) | 20% | EFB overhead adds setup time, GPU slower than CPU at 818 rows |

**Known unknowns:**
- EFB pre-bundler is a NEW implementation — first real-world timing
- bin_construct=5000 may slow Dataset construction (more bins = more memory)
- 10-group CPCV with 30 sampled paths: implementation overhead unknown

---

## Compared to Other Timeframes

| TF | Expected (all opts+GC) | CPCV % |
|---|---|---|
| **1w** | **~47 min** | 36% |
| 1d | ~3.0 hr (est. with GC) | ~50% |
| 4h | ~6.0 hr (est. with GC) | ~55% |
| 1h | ~13 hr (est. with GC) | ~60% |
| 15m | ~26 hr (est. with GC) | ~65% |

Note: 1d-15m estimates in this table are rough (GC = game-changers). FINAL_ETAS.md post-15-opts numbers for those TFs need similar game-changer deltas applied.

---

## Recommended Execution

1. Run 1w first — validates full fixed pipeline end-to-end in <1 hour
2. While 1w runs on 1 GPU, start 1d on a second GPU simultaneously (different VRAM footprints)
3. If 1w takes <30 min → optimistic range confirmed → accelerate remaining TF estimates
4. If 1w takes >60 min → investigate bottleneck before committing to longer TF runs
