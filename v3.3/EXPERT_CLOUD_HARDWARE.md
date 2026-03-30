# EXPERT: Cloud Hardware Selection for LightGBM Sparse Training

**Scope**: Selecting optimal cloud hardware for LightGBM training with 2-10M sparse binary features, 512GB-1TB RAM requirement, high CPU core count.
**Workload Profile**: CSR matrices, EFB enabled, max_bin=2, feature_fraction>=0.7, 5 timeframes (1w/1d/4h/1h/15m), CPCV folds, cuda_sparse fork.
**Date**: 2026-03-30

---

## Table of Contents

1. [Workload Characteristics](#1-workload-characteristics)
2. [GPU Comparison Matrix](#2-gpu-comparison-matrix)
3. [Does 8x RTX 5090 Exist on Cloud?](#3-does-8x-rtx-5090-exist-on-cloud)
4. [Bang-for-Buck: 8x4090 vs 4xA100 vs 8xH100](#4-bang-for-buck-8x4090-vs-4xa100-vs-8xh100)
5. [Provider Comparison](#5-provider-comparison)
6. [High-Memory Options for 15m (512GB+ RAM)](#6-high-memory-options-for-15m-512gb-ram)
7. [AMD MI300X: Worth Considering?](#7-amd-mi300x-worth-considering)
8. [Thermal Throttling on Sustained Runs](#8-thermal-throttling-on-sustained-runs)
9. [Concrete Recommendations](#9-concrete-recommendations)

---

## 1. Workload Characteristics

What makes this workload unusual and drives hardware selection:

| Factor | Value | Hardware Implication |
|--------|-------|---------------------|
| Feature count | 2.9M-10M sparse binary | EFB bundles to ~23K post-bundle; GPU histogram building is the payoff |
| Matrix format | CSR, int64 indptr, int32 indices | ~60GB sparse for large TFs; must fit in GPU VRAM or use cuda_sparse fork |
| Training rows | 50K (1w) to 294K (15m) | Row count is modest; feature width is the bottleneck |
| RAM for 15m | 512GB-1TB needed | Cross gen + fold prep + metadata must all be resident |
| CPU dependency | DMatrix construction, fold prep, feature engineering | CPU score (cores x GHz) determines non-GPU phases |
| GPU dependency | Histogram building (cuda_sparse fork), EFB bundling | Memory bandwidth > raw FLOPS for this workload |
| Training duration | 6-15 hours per TF depending on machine | Must sustain full load without throttling |

**Critical constraint**: LightGBM GPU acceleration speeds up histogram construction. With post-EFB 23K features and 294K rows, the GPU histogram phase is memory-bandwidth-bound, not compute-bound. This means **memory bandwidth matters more than TFLOPS**.

---

## 2. GPU Comparison Matrix

| GPU | VRAM | Mem BW (TB/s) | NVLink | Price/GPU/hr (vast.ai) | Best For |
|-----|------|---------------|--------|------------------------|----------|
| **RTX 4090** | 24 GB GDDR6X | 1.01 | No | ~$0.30-0.50 | Budget single-GPU jobs, small TFs (1w/1d) |
| **RTX 5090** | 32 GB GDDR7 | 1.79 | No | ~$0.37 median | Better VRAM headroom, good price/perf |
| **A100 80GB** | 80 GB HBM2e | 2.04 | Yes (600 GB/s) | ~$1.00-1.50 | Best balance of VRAM, bandwidth, availability |
| **H100 SXM** | 80 GB HBM3 | 3.35 | Yes (900 GB/s) | ~$2.10-4.00 | Fastest histogram building, best for 15m |
| **H200** | 141 GB HBM3e | 4.80 | Yes | ~$2.50+ | Overkill VRAM but insane bandwidth |
| **MI300X** | 192 GB HBM3 | 5.30 | No (IF instead) | Limited avail | Memory king but ROCm ecosystem weak for LightGBM |

**Key insight**: For our cuda_sparse fork doing SpMV-based histogram building, the ranking by memory bandwidth is:
MI300X (5.3) > H200 (4.8) > H100 (3.35) > A100 (2.04) > RTX 5090 (1.79) > RTX 4090 (1.01)

But memory bandwidth alone does not determine training speed. CPU phases, DMatrix construction, fold handling, and cross gen are equally important.

---

## 3. Does 8x RTX 5090 Exist on Cloud?

**Status (March 2026)**: Partially.

- **Vast.ai**: RTX 5090 availability listed as "High (120+)" on the marketplace, median ~$0.37/GPU/hr. However, **8-GPU 5090 nodes with 512GB+ RAM are extremely rare**. Most hosts have 1-2 GPUs with consumer-level RAM (32-128GB).
- **Lambda Labs**: Shipping RTX 5090 workstations (1-2 GPUs) but **no public cloud instances** with 5090 yet. One third-party source claims ~$1.59/hr but unconfirmed.
- **RunPod**: Some 5090 availability but again mostly 1-2 GPU configurations.
- **GCP/Azure/AWS**: No RTX 5090 cloud instances. These providers only offer data-center GPUs (A100, H100, H200, L40S).

**Verdict**: An 8x RTX 5090 node with 512GB+ RAM is currently a unicorn on cloud marketplaces. You might find 4x5090 with 256GB RAM, but the full 8-GPU high-RAM config is not reliably available.

**Alternative**: 8x RTX 4090 nodes are much more common on vast.ai (many hosts built mining rigs that converted). These have proven reliability for our workload.

---

## 4. Bang-for-Buck: 8x4090 vs 4xA100 vs 8xH100

### Cost Comparison (15-hour training run)

| Config | Hourly Cost | 15hr Cost | GPU Mem BW (total) | System RAM (typical) | CPU Cores (typical) |
|--------|-------------|-----------|--------------------|-----------------------|---------------------|
| **8x RTX 4090** (vast.ai) | ~$3-4/hr | $45-60 | 8.08 TB/s | 128-256 GB | 32-64 |
| **4x A100 80GB** (vast.ai) | ~$5-6/hr | $75-90 | 8.16 TB/s | 256-512 GB | 48-96 |
| **8x A100 80GB** (vast.ai) | ~$8-12/hr | $120-180 | 16.3 TB/s | 512-1008 GB | 96-128 |
| **8x H100 SXM** (Lambda) | ~$32/hr | $480 | 26.8 TB/s | 1800 GB | 208 vCPU |
| **8x H100 SXM** (GCP spot) | ~$3-10/hr | $45-150 | 26.8 TB/s | 1872 GB | 208 vCPU |
| **8x H100 SXM** (vast.ai) | ~$17-24/hr | $255-360 | 26.8 TB/s | varies | varies |

### Analysis

**8x RTX 4090** ($45-60 per run):
- Cheapest option. Proven working with our cuda_sparse fork.
- **Problem**: Only 24GB VRAM per GPU, no NVLink, and hosts rarely have >256GB system RAM. **Cannot handle 15m timeframe** (needs 512GB+).
- **Best for**: 1w, 1d, 4h timeframes where RAM < 256GB is sufficient.

**4x A100 80GB** ($75-90 per run):
- Solid middle ground. 80GB VRAM eliminates GPU OOM risk.
- NVLink enables multi-GPU communication if needed.
- **Problem**: Only 4 GPUs means fewer parallel folds. System RAM may still be <512GB.
- **Best for**: Medium TFs if combined with high-RAM host.

**8x A100 80GB** ($120-180 per run):
- Best value for the full pipeline. Often comes with 512-1008GB RAM on vast.ai.
- NVLink, 80GB VRAM, plenty of CPU cores.
- **Problem**: A100 nodes are in high demand; availability varies.
- **Best for**: All timeframes including 15m. The recommended default.

**8x H100 SXM** ($45-480 per run depending on provider):
- Fastest possible. 3.35 TB/s per GPU = ~1.6x A100 bandwidth.
- Lambda nodes come with 1800GB RAM and 208 vCPU — solves 15m RAM problem completely.
- GCP spot pricing can be competitive ($3-10/hr) but preemption risk.
- **Problem**: Expensive at list price. Lambda at $32/hr = $480 per 15-hour run.
- **Best for**: 15m timeframe where speed and RAM are both critical.

### Winner by Scenario

| Scenario | Best Pick | Why |
|----------|-----------|-----|
| Budget-constrained, small TFs | 8x RTX 4090 on vast.ai | $3-4/hr, proven working |
| All TFs except 15m | 8x A100 80GB on vast.ai | Best balance of cost, VRAM, RAM |
| 15m timeframe specifically | 8x H100 on GCP spot or Lambda | 1.8TB RAM, 208 vCPU, massive bandwidth |
| Fastest possible wall-clock | 8x H100 SXM on Lambda | Premium but guaranteed resources |

---

## 5. Provider Comparison

| Provider | Best Config | Price Range | RAM | Reliability | Preemption Risk |
|----------|-------------|-------------|-----|-------------|-----------------|
| **Vast.ai** | 8x A100 80GB | $8-12/hr | Up to 1008 GB | Variable (host-dependent) | Low (reserved) to High (interruptible) |
| **Lambda Labs** | 8x H100 SXM | ~$32/hr | 1800 GB | High (data center) | None (dedicated) |
| **GCP (A3 spot)** | 8x H100 | ~$3-10/hr | 1872 GB | High (Google infra) | Medium (spot preemption) |
| **Azure (ND96isr)** | 8x H100 | ~$70-98/hr | ~1900 GB | High | Low (on-demand) |
| **RunPod** | 8x A100/H100 | $8-16/hr | Varies | Medium | Low (community) |

### Provider Recommendations

**For most training runs (1w/1d/4h/1h)**: Vast.ai with 8x A100 80GB or 8x 4090.
- Filter for: verified host, 256GB+ RAM, 64+ CPU cores, high reliability score.
- Use reserved (not interruptible) for 15-hour runs.

**For 15m timeframe**: Lambda Labs 8x H100 SXM or GCP A3 spot.
- Lambda: $32/hr but guaranteed 1.8TB RAM, 208 vCPU, no surprises.
- GCP spot: Much cheaper but must have robust checkpointing for preemption.

**Avoid for this workload**:
- Azure: Too expensive ($70-98/hr) for training that doesn't need enterprise compliance.
- Single-GPU instances: Our CPCV pipeline benefits from parallel fold training across GPUs.

---

## 6. High-Memory Options for 15m (512GB+ RAM)

The 15m timeframe is the bottleneck: 294K rows x 10M features, cross gen needs 512GB-1TB RAM.

### Options Ranked

| Option | RAM | CPU | GPU | Cost/hr | Notes |
|--------|-----|-----|-----|---------|-------|
| **Lambda 8x H100** | 1800 GB | 208 vCPU | 8x H100 SXM | ~$32 | Gold standard. Everything fits. |
| **GCP A3 8x H100** | 1872 GB | 208 vCPU | 8x H100 SXM | ~$3-10 (spot) | Best value if you handle preemption |
| **Vast.ai 8x A100** | 512-1008 GB | 96-128 cores | 8x A100 80GB | ~$8-12 | Hunt for high-RAM hosts |
| **AWS High-Memory** | Up to 24 TB | 448 vCPU | None (CPU-only) | ~$10-27 | For cross gen only, no GPU training |
| **GCP M3-megamem** | 1-2 TB | 128 vCPU | None | ~$15-25 | CPU-only high-RAM option |

### Strategy for 15m

The practical approach given our modular pipeline:

1. **Cross gen on high-RAM CPU machine**: Use vast.ai high-RAM node or GCP M3 for cross gen phase (CPU-bound anyway).
2. **Training on GPU machine**: Transfer the generated cross features to an 8x A100/H100 GPU node.
3. **Or**: Find a single Lambda/GCP node with 1.8TB RAM + 8 GPUs and do everything there.

Option 3 is simplest operationally but most expensive. Options 1+2 save money but add transfer time and complexity.

---

## 7. AMD MI300X: Worth Considering?

### Specs That Look Amazing

- 192 GB HBM3 per GPU (2.4x H100)
- 5.3 TB/s memory bandwidth (1.6x H100)
- Could theoretically hold entire post-EFB dataset per GPU

### Why NOT for This Workload

1. **ROCm ecosystem**: LightGBM's CUDA backend does not support ROCm. Our entire cuda_sparse fork is CUDA/cuSPARSE. Porting to HIP/rocSPARSE would be a massive effort.
2. **No NVLink equivalent**: AMD uses Infinity Fabric, which has different scaling characteristics.
3. **Software maturity**: SemiAnalysis benchmarks (late 2024) showed MI300X achieving ~620 TFLOP/s BF16 vs H100's ~720 TFLOP/s despite better paper specs. Software stack matters.
4. **Cloud availability**: MI300X cloud instances are rare and mostly offered by specialized providers (CoreWeave has some).
5. **Training benchmarks lagging**: Real-world training throughput on MI300X consistently underperforms its paper specs due to ROCm kernel maturity.

**Verdict**: Hard no. Our cuda_sparse fork, cuSPARSE dependency, and CUDA-specific optimizations make MI300X a non-starter without a full rewrite.

---

## 8. Thermal Throttling on Sustained Runs

### Throttling Thresholds

| GPU | Throttle Temp | Typical Sustained Temp | Risk Level |
|-----|---------------|------------------------|------------|
| **RTX 4090** | 82-83C (die) | 75-85C in datacenter | **Medium-High** — GDDR6X memory can throttle separately |
| **RTX 5090** | ~83C (estimated) | 78-85C | **Medium** — GDDR7 runs cooler than GDDR6X |
| **A100 SXM** | 85C | 65-75C in datacenter | **Low** — HBM2e does not have thermal throttling issues |
| **H100 SXM** | 80C+ triggers slowdown | 65-78C in datacenter | **Low** — datacenter cooling designed for 700W TDP |
| **H200** | Similar to H100 | 65-78C | **Low** — liquid cooling standard |

### Key Findings

1. **H100/A100 in datacenters**: Designed for 24/7 sustained loads. Liquid or high-CFM air cooling keeps temps well under throttle thresholds. A 15-hour training run is routine.

2. **RTX 4090 on vast.ai**: **This is where throttling risk lives.** Consumer-grade cooling in heterogeneous host environments. Some hosts have proper rack cooling; others are repurposed mining rigs in garages. An H100 at 88C runs 15% slower than at 75C. RTX 4090 GDDR6X can hit thermal limits during sustained training.

3. **Mitigation on vast.ai**:
   - Check host reliability score and DLPerf rating before renting
   - Monitor `nvidia-smi` temps in first 30 minutes
   - If any GPU exceeds 80C sustained, the host has inadequate cooling — migrate immediately
   - Prefer hosts with "datacenter" or "verified" tags

4. **HBM vs GDDR6X**: HBM memory (A100/H100/H200) does NOT experience thermal throttling that impacts GDDR6X during sustained loads. This is a structural advantage of datacenter GPUs for long training runs.

**Bottom line**: For 15-hour sustained runs, datacenter GPUs (A100/H100) are significantly more reliable than consumer GPUs (4090/5090) due to cooling architecture.

---

## 9. Concrete Recommendations

### Tier 1: Default for Most TFs (1w, 1d, 4h, 1h)

**8x RTX 4090 on vast.ai** — if RAM sufficient (256GB+)
- Cost: ~$3-4/hr ($45-60 per 15hr run)
- Proven with cuda_sparse fork
- Filter: verified host, 256GB+ RAM, 64+ cores, reliability >95%

**8x A100 80GB on vast.ai** — if 4090 RAM insufficient
- Cost: ~$8-12/hr ($120-180 per 15hr run)
- 80GB VRAM eliminates GPU OOM, NVLink for multi-GPU
- Better thermal profile for sustained runs

### Tier 2: 15m Timeframe (512GB+ RAM Required)

**GCP A3 spot 8x H100** — best value
- Cost: ~$3-10/hr if spot available ($45-150 per run)
- 1872 GB RAM, 208 vCPU, built for this
- **Requires**: Robust checkpointing for spot preemption

**Lambda 8x H100** — most reliable
- Cost: ~$32/hr ($480 per 15hr run)
- 1800 GB RAM, 208 vCPU, dedicated, no preemption
- **Use when**: GCP spot unavailable or checkpointing not robust enough

### Tier 3: Experimental / Future

**8x RTX 5090 on vast.ai** — when 8-GPU high-RAM hosts appear
- Cost: ~$3/hr projected ($45 per 15hr run)
- 32GB VRAM (better than 4090), 1.79 TB/s bandwidth
- **Wait for**: Hosts with 512GB+ RAM and 8 GPUs to appear on marketplace

**H200 instances** — when pricing drops
- 141 GB VRAM, 4.8 TB/s bandwidth
- Currently $2.50+/GPU/hr (~$300+ per run for 8x)
- Overkill on VRAM but the bandwidth would accelerate histogram building significantly

### Decision Flowchart

```
Is this 15m timeframe?
├── YES → Need 512GB+ RAM
│   ├── Budget OK ($32/hr)? → Lambda 8x H100
│   └── Budget tight? → GCP A3 spot 8x H100
└── NO → 1w/1d/4h/1h
    ├── Post-EFB fits in 24GB VRAM? → 8x RTX 4090 (vast.ai, cheapest)
    └── Needs >24GB VRAM? → 8x A100 80GB (vast.ai)
```

### Machine Selection Checklist (vast.ai)

When filtering machines on vast.ai marketplace:

- [ ] GPU count: 8x (minimum 4x)
- [ ] System RAM: 256GB+ (512GB+ for 15m)
- [ ] CPU cores: 64+ (128+ preferred)
- [ ] CPU clock: 2.5+ GHz (higher = faster DMatrix construction)
- [ ] CPU Score: cores x GHz >= 200
- [ ] Disk: 500GB+ NVMe (for checkpoints and cross features)
- [ ] Reliability: >95%
- [ ] DLPerf: Check benchmark score
- [ ] Verified host preferred
- [ ] Internet speed: 500+ Mbps (for SCP transfers)
- [ ] CUDA version: 12.x+ (for cuda_sparse fork compatibility)
- [ ] NVIDIA driver: 535+ (universal compatibility)

---

## Sources

- [Vast.ai GPU Pricing](https://vast.ai/pricing) — Live marketplace rates
- [Lambda Labs Pricing](https://lambda.ai/pricing) — Dedicated 8x H100 instances
- [GCP A3 Instance Specs](https://cloudprice.net/gcp/compute/instances/a3-highgpu-8g) — 8x H100 spot pricing
- [Azure ND H100 v5 Pricing](https://cyfuture.cloud/kb/gpu/azure-nd-h100-v5-pricing-updated-cloud-gpu-costs-2025)
- [SemiAnalysis MI300X vs H100 Benchmarks](https://newsletter.semianalysis.com/p/mi300x-vs-h100-vs-h200-benchmark-part-1-training)
- [LightGBM GPU Performance Guide](https://lightgbm.readthedocs.io/en/latest/GPU-Performance.html)
- [Cloud GPU Thermal Throttling](https://matt.sh/cloud-gpu-thermal-throttling)
- [Safe GPU Temperatures for AI](https://www.whaleflux.com/blog/safe-gpu-temperatures-a-guide-for-ai-teams/)
- [RTX 4090 vs A100 vs H100 Comparison](https://www.synpixcloud.com/blog/rtx-4090-vs-a100-vs-h100-comparison)
- [GPU Cloud Pricing Comparison 2026](https://www.spheron.network/blog/gpu-cloud-pricing-comparison-2026/)
