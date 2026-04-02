# OPT_CPU_EXPERT.md -- CPU Architecture Analysis for Sparse CSR LightGBM
## Generated 2026-03-30 | Perplexity-validated (4 queries, Claude model)

---

## 1. Executive Summary

The pipeline's CPU bottleneck is **memory bandwidth and L3 cache capacity**, not raw FLOPS. LightGBM sparse histogram construction performs O(2 x NNZ) operations per feature with highly scattered memory access patterns (CSR indirect indexing into gradient/hessian arrays). For the 2.9M-feature matrix with ~0.1% density, the working set after EFB (~23K bundles) produces ~112 MB of histogram data per tree node -- 3x larger than the 13900K's 36MB L3 but well within EPYC 9004's 384MB L3.

**Key finding**: Clock speed dominates for small TFs (1w/1d), core count + cache + bandwidth dominate for large TFs (1h/15m). The optimal CPU changes per timeframe.

---

## 2. LightGBM Sparse Histogram Internals

### How It Works
- **Column-wise (force_col_wise=True)**: OpenMP threads partition the feature/bundle axis. Each thread processes a slice of the 23K EFB bundles independently. Per-bundle histogram is ~5KB (255 bins x 20 bytes) -- fits in L1 on any CPU.
- **Row-wise (force_row_wise=True, 15m only)**: Threads partition the data axis, each building private histogram copies, then merge. Merge is serial and grows with thread count.
- **Histogram subtraction trick**: Only the smaller child needs explicit construction; the other = parent - sibling. Requires caching parent histograms (~112MB each).

### Per-Feature Non-Zero Counts at 0.1% Density

| TF | Rows | Avg NNZ/Feature | Implication |
|----|------|-----------------|-------------|
| 1w | 818 | ~0.82 | 99%+ features empty per split. Thread overhead > work |
| 1d | 5,733 | ~5.7 | Very low. Most histogram loops execute 0-5 iterations |
| 4h | 23,000 | ~23 | Moderate. Parallelism starts amortizing overhead |
| 1h | 75,000 | ~75 | Decent work per thread. Memory bandwidth becomes limiter |
| 15m | 227,000 | ~227 | Substantial work. Row-wise merge overhead caps scaling |

---

## 3. Intel 13900K vs AMD EPYC 9004 Analysis

### Architecture Comparison

| Factor | Intel 13900K | AMD EPYC 9654 (96C) |
|--------|-------------|---------------------|
| Cores/Threads | 8P+16E / 32T | 96C / 192T |
| Effective cores for LGBM | ~8 P-cores (E-cores ~40% as capable) | 96 physical |
| L3 Cache | 36 MB (monolithic) | 384 MB (distributed, ~32MB/CCD) |
| Memory Bandwidth | ~64 GB/s (2ch DDR5) | ~460 GB/s (8ch DDR5) |
| Memory Latency (L3 hit) | ~4 ns (flat) | ~10 ns local, ~30-40 ns cross-CCD |
| NUMA Topology | Single die, flat | 4-8 NUMA nodes |
| Boost Clock | 5.8 GHz | ~3.7 GHz |

### Where 13900K Wins
- **Small datasets (1w, 1d)**: At <6K rows, per-feature work is near-zero. Thread scheduling overhead dominates. 5.8 GHz single-core speed + flat memory topology = lower latency per tree node.
- **Live inference**: Single-pass prediction latency. Flat NUMA = predictable tail latency for trading decisions.
- **Simplicity**: No NUMA configuration needed. Just works.

### Where EPYC 9004 Wins
- **Large datasets (4h, 1h, 15m)**: Memory bandwidth 7x advantage is decisive when histogram working set (112MB) spills L3. Gradient/hessian random access at 75K+ rows saturates 13900K's 64 GB/s.
- **Histogram subtraction pool**: Each cached parent node = ~112MB. EPYC holds ~3 nodes in L3; 13900K holds zero. Every subtraction on 13900K hits DRAM.
- **Parallel CPCV**: 96 cores can run multiple folds simultaneously (subprocess isolation). 13900K limited to sequential folds.

### Critical EPYC Caveat: NUMA
The 384MB L3 is distributed across 12 CCDs. Each CCD has ~32MB local L3 -- nearly identical to the 13900K. Without NUMA pinning, cross-CCD memory access doubles latency and halves bandwidth. EPYC can **underperform** 13900K if NUMA is misconfigured.

Required NUMA configuration:
```bash
# Single-TF training: bind to one NUMA node
numactl --cpunodebind=0 --membind=0 python -u cloud_run_tf.py --symbol BTC --tf 1w

# Multi-TF parallel: one TF per NUMA node
numactl --cpunodebind=0 --membind=0 python train_1w.py &
numactl --cpunodebind=1 --membind=1 python train_1d.py &

# Full interleave for single large TF (already in codebase)
numactl --interleave=all python -u cloud_run_tf.py --symbol BTC --tf 15m
```

---

## 4. Thread Scaling for Sparse LightGBM

### Scaling Profile (Perplexity-validated)

| Thread Range | Speedup Factor | Bottleneck |
|-------------|---------------|------------|
| 1 -> 4 | ~3.2-3.6x | Near-linear (L1/L2 resident) |
| 4 -> 8 | ~1.5-1.8x | Memory BW contention begins |
| 8 -> 16 | ~1.2-1.4x | Bandwidth plateau |
| 16 -> 32 | ~1.05-1.15x | Mostly noise |
| Beyond 32 | Often regression | OpenMP barriers + NUMA penalties |

**Key insight**: Sparse data at 0.1% density means nearly every non-zero access is a cache miss (fetched cache lines are ~99.9% zero). This makes the workload memory-bandwidth bound, not compute bound. Adding cores beyond the bandwidth saturation point yields zero benefit.

### EFB Mitigation
EFB bundling 2.9M sparse features into ~23K denser bundles significantly improves cache locality. Each bundle packs non-zeros from multiple exclusive features, reducing the cache-miss-per-unit-work ratio. Post-EFB, scaling improves to approximately:
- 1 -> 8 cores: ~5-6x
- 8 -> 16 cores: ~1.3-1.5x
- Beyond 16: diminishing returns

### Row-Wise (15m) Specific Ceiling
For force_row_wise, histogram merge is serial: O(threads x features x bins). Above ~16-20 threads, merge time grows faster than build time shrinks. Hard Amdahl ceiling at ~16-20 physical cores.

---

## 5. L3 Cache Impact on EFB Bundle Traversal

### Working Set Sizes

| Component | Size | 36MB L3 | 384MB L3 |
|-----------|------|---------|----------|
| Single bundle histogram | ~5 KB | L1 resident | L1 resident |
| Full node histogram (23K bundles) | ~112 MB | 3x overflow | Fits (71% headroom) |
| Gradient/hessian @ 818 rows | 13 KB | Fits | Fits |
| Gradient/hessian @ 5.7K rows | 91 KB | Fits | Fits |
| Gradient/hessian @ 23K rows | 368 KB | Fits | Fits |
| Gradient/hessian @ 75K rows | 1.2 MB | Fits | Fits |
| Gradient/hessian @ 227K rows | 3.6 MB | Fits | Fits |

**Column-wise mode (most TFs)**: Per-bundle histogram is always L1-resident on both CPUs. The differentiator is gradient/hessian random access during sparse index traversal -- both CPUs handle this fine for the row counts in this pipeline.

**The real gap**: Histogram subtraction pool. Each parent node's cached histogram = ~112MB. The 13900K cannot hold even one parent histogram in L3. Every subtraction pass re-reads from DRAM. EPYC holds ~3 complete parent histograms. This is likely the **largest real-world performance gap** between the two platforms.

---

## 6. Per-TF CPU Recommendations

### 1w (818 rows, ~621 base + crosses)

| Parameter | Recommendation | Rationale |
|-----------|---------------|-----------|
| Best CPU | **13900K** (local) | Near-zero work per feature. Clock speed wins |
| num_threads | 8 (P-cores only) | E-cores add overhead without proportional work |
| force_col_wise | True | Standard for high feature count |
| Expected time | 30-40 min Optuna | Already validated locally |
| NUMA | N/A (single die) | -- |

### 1d (5,733 rows, ~23K crosses)

| Parameter | Recommendation | Rationale |
|-----------|---------------|-----------|
| Best CPU | **13900K** (local) or EPYC w/ NUMA pinning | Still low NNZ/feature (~5.7). Clock speed competitive |
| num_threads | 8 (13900K P-cores) or 12-16 (EPYC single node) | Thread overhead still significant |
| force_col_wise | True | -- |
| Expected time | ~8hr local | Fits in 64GB RAM |
| NUMA | cpunodebind=0 if EPYC | Mandatory |

### 4h (23,000 rows, ~2.9M crosses)

| Parameter | Recommendation | Rationale |
|-----------|---------------|-----------|
| Best CPU | **EPYC 9004** (cloud) | 2.9M features = 112MB histogram. L3 + bandwidth advantage kicks in |
| num_threads | 16-24 physical cores | Sweet spot before bandwidth saturation |
| force_col_wise | True | 23K rows / 23K bundles = 1:1 ratio, col-wise optimal |
| Expected time | ~21hr (GPU), faster with EPYC histogram caching | -- |
| NUMA | cpunodebind=0 --membind=0 | Single NUMA node sufficient |

### 1h (75,000 rows, ~5M crosses)

| Parameter | Recommendation | Rationale |
|-----------|---------------|-----------|
| Best CPU | **EPYC 9004** (cloud, 128GB+ RAM) | 75K rows x 5M features = massive working set. 460 GB/s bandwidth critical |
| num_threads | 24-32 physical cores | Bandwidth saturation ~32 cores |
| force_col_wise | True | rows:bundles ratio = 3.8:1, col-wise still optimal |
| Expected time | ~45hr estimated | Histogram subtraction pool fits EPYC L3 |
| NUMA | interleave=all for >4 NUMA nodes | Spread memory pages across all nodes |

### 15m (227,000 rows, ~10M crosses)

| Parameter | Recommendation | Rationale |
|-----------|---------------|-----------|
| Best CPU | **EPYC 9004** (cloud, 128GB+ RAM) | Only viable option (100GB+ training RAM). Bandwidth-dominated |
| num_threads | 16 (row-wise merge ceiling) | Merge overhead caps at 16-20 threads |
| force_row_wise | True (already configured) | rows:bundles = 6.9:1, row-wise faster |
| Expected time | ~87hr estimated | Cloud only |
| NUMA | interleave=all | Mandatory for large allocations |

---

## 7. Specific Optimization Recommendations

### A. 13900K Local Optimizations (1w, 1d)

1. **Disable E-cores for LightGBM**: Set process affinity to P-cores only. E-cores at ~4.3 GHz are ~40% as capable and cause OpenMP load imbalance.
   ```bash
   # Windows: set affinity to first 8 cores (P-cores) via taskset equivalent
   # Or set num_threads=8 in LightGBM params (already capped in ml_multi_tf.py for <10K rows)
   ```

2. **Disable hyperthreading for training**: LightGBM docs explicitly state physical cores only. HT yields 10-30% at best, can degrade memory-bound sparse work.

3. **Current config is mostly correct**: `num_threads=0` (auto-detect) + the <10K row cap in ml_multi_tf.py. Consider hardcoding `num_threads=8` for 1w/1d.

### B. EPYC Cloud Optimizations (4h, 1h, 15m)

1. **NUMA pinning is NON-NEGOTIABLE**: Without it, EPYC's effective per-CCD L3 is ~32MB (same as 13900K) and cross-CCD latency adds 20-30ns per access.

2. **Cap threads per TF**:
   - 4h: `num_threads=16`
   - 1h: `num_threads=24`
   - 15m: `num_threads=16` (row-wise merge ceiling)

3. **OpenMP environment**:
   ```bash
   export OMP_PROC_BIND=close        # Keep threads on nearby cores (single NUMA)
   export OMP_SCHEDULE=static         # Post-EFB bundles are well-balanced
   export GOMP_SPINCOUNT=100000000    # Reduce wake-up latency between rounds
   ```

4. **Parallel CPCV across NUMA nodes**: Run different CPCV folds on different NUMA nodes (subprocess isolation already in codebase). Each fold gets its own 32MB L3 partition.

### C. AMD 3D V-Cache Consideration

AMD Ryzen 9 7950X3D has 128MB L3 (3D V-Cache) at 5.7 GHz boost, 16 cores. This is a potential sweet spot for 4h training:
- 3x the L3 of 13900K (histogram subtraction pool partially fits)
- Near-identical clock speed
- Better memory latency than EPYC (2 CCD vs 12)
- 16 cores is at the scaling plateau for sparse LightGBM

**Not recommended** for 1h/15m (insufficient RAM), but could be optimal for 1w/1d/4h if renting locally.

### D. Code Changes Recommended

1. **ml_multi_tf.py line 1563-1566**: The <10K row thread cap is good but uses `min(cores, max(4, cores//4))`. For 13900K, this gives `min(24, max(4, 6))` = 6. Should be `min(8, ...)` to target P-cores specifically.

2. **config.py**: Add per-TF `num_threads` overrides (like TF_MIN_DATA_IN_LEAF):
   ```python
   TF_NUM_THREADS = {
       '1w': 8,    # P-cores only, thread overhead > work
       '1d': 8,    # P-cores only
       '4h': 16,   # Cloud: bandwidth plateau
       '1h': 24,   # Cloud: bandwidth plateau
       '15m': 16,  # Cloud: row-wise merge ceiling
   }
   ```

3. **Cloud deploy template**: Add NUMA + OMP environment to deployment checklist:
   ```bash
   export OMP_PROC_BIND=close
   export OMP_SCHEDULE=static
   export GOMP_SPINCOUNT=100000000
   # Then numactl per TF
   ```

---

## 8. Hardware Recommendation Summary

| Scenario | CPU | Cores Used | Why |
|----------|-----|-----------|-----|
| 1w + 1d local training | 13900K (existing) | 8 P-cores | Clock speed dominates. Already owned. 30-40min + 8hr |
| 4h cloud training | EPYC 9354 (32C, 128MB L3) or 7950X3D | 16 | L3 fits histogram subtraction. Bandwidth sufficient |
| 1h cloud training | EPYC 9654 (96C, 384MB L3) | 24 | 128GB+ RAM required. L3 + bandwidth critical |
| 15m cloud training | EPYC 9654 (96C, 384MB L3) | 16 | 128GB+ RAM required. Row-wise merge caps threads |
| Live inference | 13900K (existing) | 8 P-cores | Latency-critical. Flat NUMA = predictable |

**Cost-optimal cloud path**: Rent EPYC 9654 (96C/384MB L3/128GB+ RAM) once for 4h+1h+15m sequentially. Total ~150hr. At $1-2/hr on vast.ai = $150-300.

---

## 9. Matrix Adherence Statement

All recommendations in this report preserve the matrix thesis:

- **NO feature filtering**: All 2.9M features preserved through training. EFB bundles features but preserves signal identity (bin offsets). Thread count and cache optimizations do not alter the feature set.
- **NO NaN conversion**: NaN = missing, structural zeros = feature OFF. CPU architecture has no impact on NaN handling.
- **feature_fraction >= 0.7**: Thread/cache recommendations are independent of feature_fraction. The 0.9 setting in config.py is correct and untouched.
- **Rare signals protected**: min_data_in_leaf=8, feature_pre_filter=False, bagging_fraction=0.95 -- all CPU-independent. A 10-fire signal in the CSR matrix reaches LightGBM identically regardless of CPU architecture.
- **Esoteric signals ARE the edge**: CPU optimizations affect only speed, never signal selection. The astrology/gematria/numerology/space weather cross-products traverse the same histogram code path as any other feature.

**No parameter in this report changes what the model learns. Only how fast it learns it.**

---

## 10. Sources

All findings validated via Perplexity (Claude model, 2026-03-30) with full matrix thesis context:
1. LightGBM documentation: Parameters, Features, GPU Tuning Guide (lightgbm.readthedocs.io)
2. LightGBM GitHub issues: #5205 (sparse loading), #5039 (thread deadlock), #1723 (num_threads)
3. AMD EPYC 9004 performance briefs (amd.com)
4. NeurIPS 2017: "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"
5. Technical City: EPYC 9654 vs Core i9-13900K benchmarks
6. Stack Overflow: OpenMP histogram reduction, NUMA effects
7. Reddit r/LocalLLaMA: DDR5 bandwidth measurements across platforms
