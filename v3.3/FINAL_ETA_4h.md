# FINAL ETA: 4h Timeframe — All Optimizations + Game-Changers + Bug Fixes

**Date:** 2026-03-30
**Machine:** 8x RTX 5090 (32GB each) + EPYC 128c + 512GB–1TB RAM + NVMe
**Cost rate:** ~$3.50/hr
**Data:** 8,794 rows × 3,904 base features → 6–8M cross features (~7.6M pairwise)

---

## What's New vs FINAL_ETAS.md

FINAL_ETAS.md showed 4h at **7.5hr expected**. This document adds the game-changers:

| Game-Changer | Delta | Direction |
|---|---|---|
| CPCV 10 groups / 30 paths (vs 15 sequential folds) | +2.7hr | Slower (quality upgrade: 30 paths vs 15) |
| EFB Pre-Bundler (8M → ~63K bundles) | −1 min | Faster (minor; pre-computes bundle assignments) |
| Warm-start from 1d Optuna results | −10 min | Faster (20% fewer Optuna trials needed) |
| Sobol sampler in Optuna | −5 min | Faster (better initial coverage than TPE cold start) |
| fastmath flag in GPU fork | −12 min | Faster (~1.1x on retrain + CPCV) |
| THP=madvise (confirmed) | −8 min | Faster (large sparse allocs, already in deploy) |
| cost-sort cross pairs by NNZ descending | −3 min | Faster (GPU H2D coherence, minor at 8.8K rows) |
| Sortino fixed in optimizer | 0hr | Quality fix only (no timing delta) |
| CPCV purge fixed (purge = max_hold_bars) | 0hr | Quality fix only |

**Net change: +2.3hr expected vs FINAL_ETAS.md → ~10hr expected total**

The CPCV(10, 30 paths) upgrade is the dominant delta. It is NOT a speed improvement — it provides 30 out of C(10,2)=45 possible train-test path configurations for better PBO coverage, combinatorial uniqueness scores, and confidence calibration. Each of the 30 paths trains on 8/10 = 80% of data, vs 15 folds each training on 93% of data.

- **Old:** 15 sequential folds × 0.933 = 14.0 fold-equivalents
- **New:** 30 paths × 0.80 = 24.0 fold-equivalents
- **Ratio:** 1.71× more CPCV compute (better statistics, not slower machine)

---

## Why 4h Is the GPU Sweet Spot

- **818 rows (1w):** GPU barely saturated, CUDA speedup only ~1.5–2×
- **8,794 rows (4h):** GPU histogram builds are efficient, CUDA speedup ~2.3–2.5×
- **90K rows (1h):** GPU fully saturated, CUDA speedup ~2.5× — but CPCV becomes 8.5hr+
- At 4h: ALL optimizations compound cleanly with no memmap overhead, good GPU util, and EFB bundles fit in GPU memory

---

## Multipliers Applied (4h Specific)

| Stage | Key Multipliers | Combined | Notes |
|---|---|---|---|
| Cross Gen | Numba prange 4× · parallel steps 3× · bitpack 1.3× · indices NPZ 1.2× | **~10×** expected (18× theoretical, capped for overhead) | No memmap: direct RAM, no disk streaming |
| EFB Build | Pre-Bundler ~2× | **~2×** | Skips internal bundle computation |
| Optuna P1+P2 | 8 CPU trials 3–4× · WilcoxonPruner 1.25× · NUMA 1.15× · GC 1.05× · warm-start 1.2× · Sobol 1.1× | **~6–7×** | Warm-start: 1d params used as seed, 20% fewer trials |
| Final Retrain | CUDA 2.3× · fastmath 1.1× · THP 1.07× · NUMA 1.15× · GC 1.05× · CSC 1.05× | **~3.5×** | 8.8K rows = good GPU saturation |
| CPCV Full | CUDA 2.3× · fastmath 1.1× · NUMA 1.15× · GC 1.05× · CSC 1.05× | **~3.2×** per path; but 30 paths vs 15 folds = 1.71× more total | Sequential single-GPU paths (fold-parallel not implemented) |
| Meta+Opt+PBO | Sobol optimizer vs grid · 30-path PBO | **~1.8×** | Sobol finds optimum in fewer evaluations |

---

## Stage-by-Stage Breakdown

| Stage | Baseline | Optimistic | **Expected** | Pessimistic | Key Bottleneck |
|---|---|---|---|---|---|
| Data Load (parquet + NPZ) | 30s | 20s | **25s** | 35s | NVMe I/O |
| Cross Gen | 36 min | 2 min | **4 min** | 8 min | Numba+parallel ≈10× |
| EFB Pre-Build (8M→63K) | 4 min | 1.5 min | **2 min** | 3 min | Pre-bundler 2× |
| Optuna P1+P2 | 2.6 hr | 22 min | **35 min** | 55 min | 8 CPU trial parallelism |
| Final Retrain (800 rounds) | 5.3 hr | 1.1 hr | **1.6 hr** | 2.5 hr | CUDA histogram |
| CPCV Full (10 groups/30 paths) | 10.7 hr† | 4.5 hr | **6.5 hr** | 10 hr | Sequential folds × 1.71 more |
| Meta + Sobol Optimizer + PBO | 2 hr | 40 min | **1 hr** | 1.5 hr | Sobol convergence |
| **TOTAL** | **~21.6 hr** | **~6.5 hr** | **~9.8 hr** | **~15 hr** | |

†Baseline CPCV assumes 15 folds for comparison. New config is 30 paths, so "true" baseline for new config would be 10.7 × 1.71 ÷ (old CPCV speedup ratio) — shown for reference only.

---

## Cost Projection

| Scenario | Hours | Cost @ $3.50/hr |
|---|---|---|
| Baseline (no opts) | 21.6 hr | $76 |
| Optimistic | 6.5 hr | $23 |
| **Expected** | **9.8 hr** | **$34** |
| Pessimistic | 15.0 hr | $53 |

---

## Where Time Goes (Expected)

| Stage | Hours | % of Total |
|---|---|---|
| CPCV Full | 6.5 hr | **66%** |
| Final Retrain | 1.6 hr | 16% |
| Optuna P1+P2 | 0.6 hr | 6% |
| Meta+Opt+PBO | 1.0 hr | 10% |
| Cross Gen + EFB + Load | 0.1 hr | 1% |

**CPCV is 66% of wall time.** The fix: fold-parallel CPCV across 8 GPUs. Not yet implemented (BUG-C2 must be fixed first for cuda_sparse multi-GPU). If BUG-C2 is fixed and fold-parallel enabled:
- CPCV drops from 6.5hr → ~1.1hr (8-GPU parallel, ~6× speedup with 30 paths)
- Total drops from **9.8hr → ~4.6hr expected** — under 5 hours

---

## Blocked Improvements (What Would Actually Get to 4–5hr)

| Fix | Impact on 4h Total | Dependency |
|---|---|---|
| BUG-C2 fix (cuda_sparse multi-GPU) | −5.4hr (CPCV fold-parallel 8 GPUs) | cuda_sparse .so loading bug |
| BUG-C1 fix (warp shuffle ON) | −0.5hr (1.5× more on retrain/CPCV) | CUDA kernel modification |
| Fold-parallel CPCV (30 paths → 8 per GPU) | Requires BUG-C2 | Not yet committed |

**With both bugs fixed + fold-parallel:** Expected **~4.0hr** (optimistic **~3.0hr**)

---

## Cross Gen Details (4h Specifics)

**Input:** 8,794 rows × 3,904 base features
**Output:** ~7.6M pairwise binary cross features (3904 × 3903 ÷ 2)
**RAM:** ~7.6M × 8,794 × 1-bit ≈ **8.4GB sparse** → fits trivially in 512GB, NO memmap

**Optimization stack applied:**
1. **Numba prange:** Inner loop parallelized across 128 EPYC cores → 4× speedup
2. **Parallel 13 steps:** All cross-gen pipeline stages run concurrently → 3× speedup
3. **Bitpack POPCNT:** Binary feature pairs packed to uint64, POPCNT for popcount → 1.3× memory bandwidth reduction
4. **L2-sorted pairs (1.0× for 4h):** Pair ordering by NNZ (cost-sort commit). 4h gets minimal benefit (no L2 tiling needed at this row count — that kicks in at 90K+ rows)
5. **Indices-only NPZ:** Store only non-zero indices, not dense float arrays → 1.2× I/O speedup
6. **Adaptive chunks:** Chunk size auto-tuned to EPYC L3 cache, prevents NUMA cross-socket thrashing

**Net: 36 min → ~4 min expected (9× actual speedup)**

---

## EFB Pre-Bundler Details

**Standard flow:** lgb.Dataset(8M features) → LightGBM internal EFB bundling during construction
**New flow:** pre_bundle(8M features) → 63K bundle assignments → lgb.Dataset(63K bundles, bundle_map)

- 8,000,000 features → ~63,000 bundles = **127:1 compression**
- Rare signal identity PRESERVED: bin offsets within each bundle distinguish original features
- Pre-bundler runs once, result reused for all CPCV folds (not recomputed per fold)
- EFB Build time: 4 min → ~2 min (pre-bundler computes in parallel, LightGBM skips internal EFB step)
- Training time per tree: evaluates 63K bundle histograms instead of building for 8M features → **same** (LightGBM EFB already gives this; pre-bundler just moves the computation earlier and makes it reusable)

---

## Warm-Start from 1d Optuna

1d Optuna results provide a prior distribution for 4h hyperparameters:
- `num_leaves`, `learning_rate`, `lambda_l1`, `lambda_l2` ranges narrowed from 1d posterior
- Sobol sampler seeds initial 25 trials from this narrowed space
- WilcoxonPruner kills underperforming trials at round 100 (before they reach 800 rounds)
- Expected: **~20% fewer trials** complete to convergence → ~10 min savings on Optuna

---

## RAM Usage (512GB Machine)

| Stage | Peak RAM |
|---|---|
| Cross Gen (sparse CSR in RAM) | ~12–20GB |
| EFB Dataset (63K bundles × 8,794 rows) | ~5–8GB |
| Training (gradient arrays + histograms) | ~15–25GB (VRAM primary) |
| CPCV (30 paths, sequential) | ~15–25GB (reused per path) |
| **Peak total** | **~40–60GB** |

**No memmap required.** 512GB is 8–12× headroom. All arrays in-RAM with numactl --interleave=all.

---

## Recommended Execution

```
# Validate first (non-negotiable)
python validate.py

# Run 4h
python -u cloud_run_tf.py --symbol BTC --tf 4h 2>&1 | tee logs/4h_run.log

# Monitor: cross gen should finish in <8min, Optuna in <1hr, retrain in <2.5hr
# Download checkpoint after cross gen NPZ saved
# Download checkpoint after Optuna completes
# Download final model + CPCV report + PBO table
```

**Expected checkpoint times:**
- Cross gen done: T+4 min → download NPZ immediately
- EFB built: T+6 min
- Optuna done: T+41 min → download best_params.json
- Final retrain done: T+2.3 hr → download model.txt
- CPCV done: T+8.8 hr → download cpcv_report.json + pbo.json
- **Full pipeline done: T+9.8 hr**

---

## vs Existing FINAL_ETAS.md Comparison

| Metric | FINAL_ETAS.md | This Document | Delta |
|---|---|---|---|
| Expected total | 7.5 hr | **9.8 hr** | +2.3 hr |
| Cross gen | 5 min | 4 min | −1 min |
| Optuna | 45 min | 35 min | −10 min |
| Final retrain | 1.8 hr | 1.6 hr | −12 min |
| CPCV Full | 3.8 hr | 6.5 hr | +2.7 hr |
| Meta+Opt | 1.0 hr | 1.0 hr | 0 |
| Cost (expected) | $26 | $34 | +$8 |

**The +2.3hr is entirely from CPCV(10,30 paths) vs 15 folds.** Every other game-changer is net-negative (faster). The PBO calculation from 30 paths is statistically much stronger than 15-fold CV — this is the correct trade-off for live deployment confidence.
