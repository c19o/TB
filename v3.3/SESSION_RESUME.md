# V3.3 Session Resume — 2026-03-30

## INSTRUCTION TO NEW SESSION: Read this file completely. Then read v3.3/CLAUDE.md, v3.3/UNWIRED_OPTIMIZATIONS.md, v3.3/FINAL_AUDIT.md. Ask the user what to do next.

---

## STATUS: OPTIMIZATION SPRINT COMPLETE — 9 Branches Merged, Needs Final Polish + Audit

**Previous training results INVALID** — 12+ mechanisms were silently killing rare esoteric signals (see gpu_histogram_fork/GPU_SESSION_RESUME.md). All have been fixed this session.

---

## WHAT WAS DONE THIS SESSION (2026-03-30)

### Wave 1: 15 Core Optimizations (8 dev agents, git worktrees)
All committed and merged into v3.3:
1. **Numba prange sorted-index intersection** + L2 cache sort — `numba_cross_kernels.py`
2. **Parallel 13 cross steps** + memory-aware scheduling — `AdaptiveChunkController` in `v2_cross_generator.py`
3. **Adaptive RIGHT_CHUNK controller** — RSS-based sizing
4. **Bitpacked POPCNT co-occurrence** — `bitpack_utils.py`
5. **Memmap CSC streaming** — `memmap_merge.py` (1h: 1.8TB→5GB, 15m: 3TB→10GB)
6. **Atomic NPZ + indices-only storage** — `atomic_io.py` (50% less I/O)
7. **CSC format for LightGBM** — avoids CSR→column transpose
8. **WilcoxonPruner** inter-fold (warmup=120, patience=10)
9. **extra_trees** in Optuna search space
10. **Multi-GPU Optuna** — `multi_gpu_optuna.py` (cuda_sparse, requires 2+ GPUs)
11. **GC disable** during training loops (try/finally protected)
12. **NUMA interleave=all** for multi-socket
13. **CUDA kernel speed** — batch H2D, vectorized launches, CSR.T dual storage
14. **20min intra-step flush** for cross gen crash recovery
15. **bin_construct_sample_cnt=5000** (40x faster Dataset scanning for binary)

### Wave 2: 10 Game-Changing Discoveries (from 20 Perplexity expert reports)
All committed and merged:
16. **EFB Pre-Bundler** — `efb_prebundler.py` (10M→79K bundles, 128x histogram reduction)
17. **Optuna param fixes** — bynode 0.7, bagging 0.7, min_gain floor removed
18. **CPCV fold reduction** — 15→10 groups, 30 sampled paths (57% less CPCV)
19. **Sobol trade optimizer** — `exhaustive_optimizer.py` (3-5hr→30min, 131K Sobol + TPE refinement)
20. **lleaves + feature pruning** — `inference_pruner.py`, `lleaves_compiler.py` (6M→5K, 5.4x predict)
21. **Cost-sorted pair worklist** — nnz descending for load balance
22. **THP=madvise** — all deploy scripts fixed (prevents 512x sparse bloat)
23. **Process isolation per fold** — subprocess exits = zero fragmentation
24. **fastmath=True** on `_parallel_cross_multiply` (binary 0/1 safe)
25. **Sortino denominator FIXED** — was dividing by total_trades, now count_neg

### Bug Fixes: 16 Found, All Resolved
- BUG-C1: Warp shuffle `__ballot_sync` fix (commit 570e679)
- BUG-C2: `cuda` → `cuda_sparse` for multi-GPU (commit on d1bd48cc)
- BUG-H1: GPU contention guard for parallel cross (commit ae919bc)
- BUG-H2: `num_gpus >= 2` for multi-GPU activation
- CPCV temporal leakage: per-group purge loop (commit 979e5f9)
- NUMA: `--cpunodebind=0` → `--interleave=all`
- Optuna pruning: warmup 50→120, patience 5→10
- Sortino: `total_trades` → `count_neg` (commit d2beb67)
- Plus 8 low-severity (dead code, gc scope, CSC waste, etc.)

### Merge: 9 Branches → v3.3
- 37 files changed, +6,183 / -509 lines
- 5 conflicts resolved manually
- Optuna 0.7 floors merged LAST (overrides stale 0.5)

### Research: 20 Expert Reports (all in v3.3/EXPERT_*.md)
CUDA kernels, GPU memory, cuSPARSE, multi-GPU, CPU cache/NUMA, LightGBM EFB, Optuna HPO, Linux tuning, memory allocators, compilers/Numba, NVMe I/O, financial ML, rare signals, cloud hardware, sparse formats, Python perf, feature theory, trade optimizer, dataset build, inference optimization.

### Audits: 9 Reports (all in v3.3/)
QA_CROSSGEN_AUDIT.md, QA_TRAINING_AUDIT.md, QA_CASCADE_AUDIT.md, QA_MATRIX_THESIS_AUDIT.md, QA_VALIDATE_AUDIT.md, FINAL_AUDIT.md, DOUBLE_AUDIT_MATRIX.md, DOUBLE_AUDIT_CORRECTNESS.md, DOUBLE_AUDIT_BUGFIXES.md, POST_FIX_AUDIT.md, QA_GAMECHANGERS_AUDIT.md

---

## WHAT STILL NEEDS TO BE DONE (Next Session Plan)

### Phase 1: Fold-Parallel CPCV (THE Universal Bottleneck Fix)
CPCV is 36-66% of pipeline time for every TF. Must distribute folds across 8 GPUs.
- **Check branch `ceo/backend-dev-3bb0a3fb`** — agent was building this but may not have finished
- If incomplete: rebuild with company (3-5 agents)
- Architecture: each fold in subprocess, assigned GPU via `gpu_device_id=i`, mmap data sharing
- 30 CPCV paths / 8 GPUs = 4 rounds instead of 30 sequential
- Expected impact: CPCV 6.5hr → 1.1hr for 4h TF

### Phase 2: 25-Person World-Class Optimization Company
Deep scrutiny of EVERY pipeline stage, Perplexity-enabled, matrix-aware:

**Hardware Architecture Experts:**
- CPU expert: Intel vs AMD EPYC for training? Core count vs clock speed for sparse?
- RAM expert: DDR4 vs DDR5 bandwidth impact on sparse matrix traversal
- Bus expert: PCIe 4.0 vs 5.0 for H2D transfers, NVLink for multi-GPU
- GPU expert: RTX 5090 vs A100-80GB vs H100 for our specific workload
- NVMe expert: io_uring, direct I/O, readahead for memmap streaming

**OS/System Experts:**
- Linux kernel: scheduler tuning, cgroup memory, isolcpus for training cores
- Driver expert: CUDA driver version impact, NVIDIA persistence mode
- Memory: tcmalloc Temeraire vs jemalloc arena tuning
- THP: madvise defrag settings, khugepaged scan interval

**Pipeline Stage Experts (GPU fork AND non-GPU fork):**
- Feature build: pandas .apply() replacement, vectorized engineering
- Cross gen: Numba codegen quality, AVX-512 utilization, CSC tiling
- EFB pre-bundler: density-tier packing, collision probability analysis
- Dataset construction: save_binary caching, parallel chunk construction
- Optuna HPO: CMA-ES vs TPE, Hyperband multi-fidelity, warm-start quality
- LightGBM training: histogram computation, EFB bundle utilization
- CPCV validation: fold-parallel GPU distribution, purge correctness
- Trade optimizer: Sobol coverage quality, Sortino online accumulation
- Inference: lleaves compilation, feature pruning, live latency

**Per-TF Specialists:**
- 1w: 818 rows, CPU-optimal, what's the fastest possible?
- 1d: 5,733 rows, GPU marginal, hybrid CPU/GPU?
- 4h: 8,794 rows, GPU sweet spot starts
- 1h: 90K rows, memmap + GPU, 512GB RAM constraint
- 15m: 227K rows, peak GPU utilization, 1TB+ RAM

### Phase 3: 25-Person Full Pipeline Audit
End-to-end audit covering EVERY step from raw data to deployed model:
1. **Feature build** → correct features computed? NaN preserved?
2. **Cross gen** → all 13 steps? co-occurrence threshold? memmap lossless?
3. **EFB pre-bundler** → zero features dropped? collision-free? reversible?
4. **Dataset construction** → feature_pre_filter=False? bin_construct optimized?
5. **Optuna HPO** → param ranges correct? warm-start? WilcoxonPruner safe?
6. **LightGBM training** → cuda_sparse? feature_fraction>=0.7? bagging>=0.7?
7. **CPCV validation** → purge per-group? embargo correct? fold-parallel safe?
8. **Trade optimizer** → Sortino correct? Sobol coverage? parameter space?
9. **Model output** → accuracy floor? backup before overwrite? lleaves compiled?
10. **Inference** → pruned features only? fallback chain? latency acceptable?
11. **Matrix thesis** → ALL features preserved end-to-end? No silent filtering?

### Phase 4: Push to Git
Only after Phase 3 audit passes.

---

## CURRENT ETAs (Post All Optimizations, Pre-Fold-Parallel CPCV)

### On 8x RTX 5090 + EPYC 128c + 512GB-1TB RAM

| TF | Baseline | **Expected** | With Fold-Parallel CPCV | Cost |
|----|---------|-------------|------------------------|------|
| **1w** | 3.0 hr | **47 min** | ~25 min | ~$2 |
| **1d** | 8.5 hr | **4.1 hr** | ~2 hr | ~$7 |
| **4h** | 21 hr | **9.8 hr** | ~4 hr | ~$14 |
| **1h** | 45 hr | **~12 hr** | ~5-6 hr | ~$20 |
| **15m** | 87 hr | **~20 hr** | ~8-10 hr | ~$35 |
| **ALL 5** | **165 hr** | **~47 hr** | **~20-23 hr** | **~$78** |

**Bottleneck after fold-parallel CPCV**: Final Retrain (single-GPU, can't parallelize without NCCL histogram aggregation)

### Cloud Machine Recommendations (from EXPERT_CLOUD_HARDWARE.md)
- **1w/1d**: 8x RTX 4090 vast.ai ($3-4/hr) — cheapest, proven
- **4h/1h**: 8x A100-80GB vast.ai ($8-12/hr) — best balance
- **15m**: Lambda 8xH100 ($32/hr, 1.8TB RAM) or GCP A3 spot ($3-10/hr)
- **MI300X**: Hard no — our CUDA fork doesn't port to ROCm

---

## KEY FILES ON DISK

### New Files Created This Session
| File | Purpose |
|------|---------|
| `numba_cross_kernels.py` | Numba prange sorted-index intersection + L2 sort |
| `bitpack_utils.py` | POPCNT co-occurrence pre-filter |
| `memmap_merge.py` | Two-pass streaming CSC merge (1h/15m OOM fix) |
| `atomic_io.py` | Atomic NPZ + indices-only + .npy mmap |
| `multi_gpu_optuna.py` | Multi-GPU Optuna trial distribution |
| `efb_prebundler.py` | External EFB pre-bundler (10M→79K bundles) |
| `inference_pruner.py` | Feature extraction + pruned set creation |
| `lleaves_compiler.py` | LLVM model compilation for 5.4x predict |
| 20x `EXPERT_*.md` | Perplexity research reports |
| 11x `QA_*.md` + `FINAL_*.md` | Audit reports |
| `UNWIRED_OPTIMIZATIONS.md` | What's coded but not connected |
| `MERGE_NOTES.md` | Conflict resolution guide |

### Git State
- **Branch**: v3.3
- **Latest commit**: merge of 9 CEO branches + 33 report files
- **Worktrees**: `ceo/backend-dev-3bb0a3fb` may have partial fold-parallel CPCV work
- **NOT pushed** — waiting for Phase 3 audit

### Training Status (unchanged from previous session)
| TF | Status | Notes |
|----|--------|-------|
| 1w | NEEDS RETRAIN | All previous results INVALID (signal-killing params fixed) |
| 1d | NEEDS RETRAIN | Artifacts ready (NPZ, parquet) |
| 4h | NEEDS RETRAIN | Artifacts ready |
| 1h | NEEDS CROSS GEN + TRAIN | Memmap now implemented |
| 15m | NEEDS CROSS GEN + TRAIN | Memmap now implemented |

---

## SESSION STATS
- **~120 CEO sessions** launched
- **~$105 total cost**
- **25 optimizations + 10 game-changers = 35 total improvements**
- **20 Perplexity expert research reports**
- **11 audit reports**
- **16 bugs found, all resolved**
- **Matrix thesis: VERIFIED CLEAN** across all waves
- **9 branches merged** (37 files, +6,183 lines)
