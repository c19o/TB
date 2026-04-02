# V3.3 Cloud GPU Cost-Performance Analysis
## vast.ai / Lambda Labs / GCP Spot -- March 2026

**Generated**: 2026-03-30
**Pipeline**: LightGBM + EFB on 2.9M sparse CSR binary features, 15-fold CPCV
**Goal**: Rent 5 machines in parallel (one per TF), minimize wall-clock time, maximize cost efficiency

---

## 1. KEY FINDING: RTX 4090 IS THE SWEET SPOT FOR LIGHTGBM

Perplexity research confirms: LightGBM histogram-based training is **memory-bandwidth bound**, not FLOP-bound. RTX 4090 (~1 TB/s GDDR6X) matches or beats A100 (~2 TB/s HBM2e) per dollar because:
- H100/A100 tensor cores provide zero benefit for tree-based models
- EFB compresses 2.9M features into dense bundles, keeping VRAM usage low (~4-15GB per fold)
- H100 offers only 10-30% speedup over 4090 at 5-10x the price
- **Exception**: 1h/15m need 128GB+ system RAM, which limits machine choices

---

## 2. PIPELINE TIMING REFERENCE (from GPU_SESSION_RESUME.md)

| TF | Cross Gen | Optuna P1+P2 | Final Retrain | CPCV Full | Optimizer | **TOTAL** | RAM Needed |
|----|-----------|-------------|---------------|-----------|-----------|-----------|------------|
| 1w | cached | 18min | 40min | 1.3hr | 30min | **~3hr** | ~7 GB |
| 1d | cached | 1hr | 2hr | 4.1hr | 1hr | **~8.5hr** | ~1.6 GB |
| 4h | cached | 2.6hr | 5.3hr | 10.7hr | 2hr | **~21hr** | ~11 GB |
| 1h | 15min | 3.7hr | 12.4hr | 24.9hr | 3hr | **~45hr** | ~38 GB |
| 15m | 15min | 6.4hr | 25hr | 50hr | 5hr | **~87hr** | ~100 GB |

*Timings based on 8x RTX 5090 + EPYC 384c. Scale accordingly for actual rented hardware.*

---

## 3. SCORING FORMULA

### Base Metrics

| Metric | Symbol | How to get |
|--------|--------|------------|
| CPU Score | `C` | `num_cores x avg_GHz` |
| GPU Bandwidth | `G` | GB/s (effective memory bandwidth) |
| System RAM | `R` | GB |
| GPU VRAM | `V` | GB |
| Price per hour | `P` | $/hr |

### Per-TF Weighted Score

```
Score(TF) = (w_cpu * C_norm + w_gpu * G_norm + w_ram * R_norm + w_vram * V_norm) / P
```

Where `*_norm` = value / max_value_across_candidates (0-1 normalized).

### Per-TF Weights

| TF | w_cpu | w_gpu | w_ram | w_vram | Rationale |
|----|-------|-------|-------|--------|-----------|
| **1w** | **0.55** | 0.20 | 0.15 | 0.10 | Few rows (2-3K). CPU-bound Optuna trials. GPU barely used. |
| **1d** | **0.45** | 0.25 | 0.15 | 0.15 | ~10K rows. CPU still dominant. GPU helps with histogram. |
| **4h** | 0.30 | **0.35** | 0.20 | 0.15 | ~40K rows. Balanced. GPU histogram matters. |
| **1h** | 0.15 | 0.25 | **0.35** | **0.25** | ~120K rows. 38GB RAM needed. GPU VRAM matters for large histograms. |
| **15m** | 0.10 | 0.20 | **0.40** | **0.30** | ~227K rows. 100GB+ RAM mandatory. GPU VRAM critical for batch size. |

### Interpretation
- Higher score = better value for money
- Score > 100 = excellent deal
- Score 50-100 = acceptable
- Score < 50 = overpaying for capability you don't need

---

## 4. PROVIDER PRICING COMPARISON (March 2026)

### vast.ai (marketplace -- prices fluctuate)

| GPU | $/hr (on-demand) | $/hr (interruptible) | VRAM | Typical Host RAM | Typical CPU |
|-----|-------------------|---------------------|------|-----------------|-------------|
| RTX 4090 (1x) | $0.29-0.39 | $0.20-0.25 | 24GB | 64-256GB | 16-64c |
| RTX 4090 (4x) | $1.20-1.60 | $0.80-1.00 | 96GB | 128-256GB | 32-128c |
| A100 80GB (1x) | $2.00-3.50 | $1.00-1.80 | 80GB | 128-512GB | 32-128c |
| H100 80GB (1x) | $1.65-2.50 | $1.00-1.50 | 80GB | 128-512GB | 32-128c |
| A100 80GB (4x) | $8-14 total | $4-7 total | 320GB | 256-1024GB | 64-256c |

### Lambda Labs (on-demand only -- no spot)

| GPU | $/hr | VRAM | RAM | vCPUs |
|-----|------|------|-----|-------|
| A100 80GB SXM | $2.49-2.79 | 80GB | 220-2900GB | 120-240 |
| H100 80GB PCIe | $3.29 | 80GB | 225GB | 26 |
| H100 80GB SXM | $3.99-4.29 | 80GB | 720-2900GB | 120-240 |
| H100 reserved (1yr) | $1.89-2.76 | 80GB | varies | varies |

### GCP (spot/preemptible)

| Instance | GPU | $/hr on-demand | $/hr spot | RAM | vCPUs |
|----------|-----|----------------|-----------|-----|-------|
| A2 Standard (A100 40GB) | A100 40GB | ~$3.67 | ~$1.15 | 85-680GB | 12-96 |
| A2 Ultra (A100 80GB) | A100 80GB | ~$5-6 | ~$2.5-3.5 | 170-1360GB | 24-192 |
| A3 High (H100 SXM) | H100 80GB | ~$3-4/GPU | ~$2.25/GPU | 240-1920GB | 52-416 |

---

## 5. PER-TF MACHINE RECOMMENDATIONS

### 1w -- CPU-Heavy, Tiny Dataset (~3hr runtime)

**TRAIN LOCALLY on 13900K + RTX 3090 (FREE)**

- Only 7GB RAM needed, ~30-40min Optuna
- No reason to rent cloud for this
- Local GPU is more than sufficient

If cloud needed: vast.ai cheapest 4090 ($0.25/hr) = $0.75 total

### 1d -- CPU-Dominant, Small Dataset (~8.5hr runtime)

**TRAIN LOCALLY on 13900K + RTX 3090 (FREE)**

- Only 1.6GB RAM needed, ~8hr total
- Local machine handles this easily overnight

If cloud needed: vast.ai cheapest 4090 ($0.25/hr) = $2.13 total

### 4h -- Balanced CPU/GPU (~21hr runtime)

**Recommended: vast.ai 1x RTX 4090 + 64c + 128GB RAM**

vast.ai search command:
```bash
vastai search offers 'gpu_name=RTX_4090 num_gpus=1 cpu_cores>=32 cpu_ram>=128 reliability>0.95 dph<=0.50' -o 'score-'
```

| Metric | Target | Score Contribution |
|--------|--------|--------------------|
| CPU | 32-64c @ 3.0+ GHz = Score 96-192 | 0.30 weight |
| GPU BW | ~1 TB/s (4090) | 0.35 weight |
| RAM | 128GB (plenty for 11GB need) | 0.20 weight |
| VRAM | 24GB (plenty for 4.5GB need) | 0.15 weight |
| Price | $0.30-0.50/hr | |
| **Est. Total Cost** | **$6.30-10.50** | 21hr x $0.30-0.50 |

Scaling factor vs 8x5090+384c reference: ~3-4x slower, so budget ~60-80hr actual.
**Realistic cost: $18-40** at $0.30-0.50/hr.

### 1h -- GPU+RAM Heavy (~45hr runtime on reference)

**Recommended: vast.ai 1x A100 80GB + 128c + 256GB RAM**

vast.ai search command:
```bash
vastai search offers 'gpu_name=A100_SXM4 num_gpus=1 cpu_cores>=64 cpu_ram>=256 reliability>0.95 dph<=2.00' -o 'score-'
```

Alternative (cheaper):
```bash
vastai search offers 'gpu_name=RTX_4090 num_gpus=4 cpu_cores>=64 cpu_ram>=256 reliability>0.95 dph<=2.00' -o 'score-'
```

| Metric | A100 80GB Option | 4x 4090 Option |
|--------|------------------|----------------|
| CPU | 64-128c | 64-128c |
| GPU BW | 2 TB/s | 4 TB/s (aggregate) |
| RAM | 256GB (38GB needed) | 256GB |
| VRAM | 80GB (15.3GB needed) | 96GB (4x24GB) |
| Price | $1.00-1.80/hr | $1.00-1.60/hr |
| **Est. Total** | **$45-80** | **$45-72** |

**Pick 4x RTX 4090** if available with 256GB+ RAM -- cheaper per GPU-hr and LightGBM is single-GPU anyway (the extra GPUs run parallel Optuna/CPCV folds).

### 15m -- Maximum RAM + GPU (~87hr runtime on reference)

**Recommended: vast.ai 4x RTX 4090 + 128c + 256GB RAM (or A100 80GB + 256GB)**

vast.ai search commands:
```bash
# Option A: 4x 4090 (cheaper, if 256GB+ RAM available)
vastai search offers 'gpu_name=RTX_4090 num_gpus>=4 cpu_cores>=96 cpu_ram>=256 reliability>0.95 dph<=2.50' -o 'score-'

# Option B: A100 80GB (guaranteed HBM bandwidth)
vastai search offers 'gpu_name=A100_SXM4 num_gpus>=1 cpu_cores>=96 cpu_ram>=256 reliability>0.95 dph<=3.00' -o 'score-'

# Option C: H100 (if price is right)
vastai search offers 'gpu_name=H100_SXM5 num_gpus>=1 cpu_cores>=96 cpu_ram>=256 reliability>0.95 dph<=3.00' -o 'score-'
```

| Metric | 4x 4090 | 1x A100 80GB | 1x H100 80GB |
|--------|---------|--------------|--------------|
| CPU | 96-128c | 64-128c | 64-128c |
| GPU BW | 4 TB/s agg. | 2 TB/s | 3.3 TB/s |
| RAM | 256GB+ (100GB needed) | 256GB+ | 256GB+ |
| VRAM | 96GB (4x24) | 80GB | 80GB |
| Price | $1.20-2.00/hr | $1.00-1.80/hr | $1.00-2.50/hr |
| **Est. Total** | **$100-175** | **$87-157** | **$87-218** |

**WARNING**: 15m is the longest job. Download checkpoints after every CPCV fold. Machine death = lost progress without checkpoints.

---

## 6. TOTAL BUDGET ESTIMATE (ALL 5 TFs IN PARALLEL)

### Scenario A: Local 1w+1d, Cloud 4h+1h+15m (RECOMMENDED)

| TF | Machine | $/hr | Est. Hours | Cost |
|----|---------|------|------------|------|
| 1w | LOCAL (13900K+3090) | $0 | 0.5-1hr | **$0** |
| 1d | LOCAL (13900K+3090) | $0 | 8hr | **$0** |
| 4h | vast.ai 1x 4090 128GB | $0.35 | 60-80hr | **$21-28** |
| 1h | vast.ai 4x 4090 256GB | $1.40 | 45-60hr | **$63-84** |
| 15m | vast.ai 4x 4090 256GB | $1.60 | 87-120hr | **$139-192** |
| | | | | **TOTAL: $223-304** |

Wall-clock time: ~87-120hr (limited by 15m TF) = **3.6-5 days**

### Scenario B: All Cloud (fastest wall-clock, everything parallel)

| TF | Machine | $/hr | Est. Hours | Cost |
|----|---------|------|------------|------|
| 1w | vast.ai 1x 4090 64GB | $0.25 | 3-5hr | **$0.75-1.25** |
| 1d | vast.ai 1x 4090 64GB | $0.25 | 8-12hr | **$2-3** |
| 4h | vast.ai 1x 4090 128GB | $0.35 | 60-80hr | **$21-28** |
| 1h | vast.ai 4x 4090 256GB | $1.40 | 45-60hr | **$63-84** |
| 15m | vast.ai 4x 4090 256GB | $1.60 | 87-120hr | **$139-192** |
| | | | | **TOTAL: $226-308** |

Wall-clock time: same (15m dominates)

### Scenario C: Premium Speed (H100s for 1h+15m)

| TF | Machine | $/hr | Est. Hours | Cost |
|----|---------|------|------------|------|
| 1w | LOCAL | $0 | 1hr | **$0** |
| 1d | LOCAL | $0 | 8hr | **$0** |
| 4h | vast.ai 1x 4090 128GB | $0.35 | 60-80hr | **$21-28** |
| 1h | vast.ai 1x H100 256GB | $1.50 | 35-50hr | **$53-75** |
| 15m | vast.ai 1x H100 256GB | $2.00 | 65-90hr | **$130-180** |
| | | | | **TOTAL: $204-283** |

Wall-clock: ~65-90hr = **2.7-3.8 days** (H100 saves ~25% on big TFs)

---

## 7. PROVIDER COMPARISON SUMMARY

| Provider | Best For | Pros | Cons |
|----------|----------|------|------|
| **vast.ai** | ALL TFs | Cheapest (4090 $0.25-0.39), marketplace variety, 4090 multi-GPU nodes common | Machines die, unstable hosts, consumer hardware |
| **Lambda Labs** | None for this workload | Stable, managed, massive RAM | No spot, 2-4x more expensive, overkill for tree models |
| **GCP Spot** | Fallback for 15m if vast.ai unavailable | Reliable, massive RAM, preemptible pricing | Still 2-3x more than vast.ai, complex setup |

### Verdict: vast.ai wins for all 5 TFs

Lambda Labs and GCP spot are **2-5x more expensive** for equivalent capability. The only reason to use them:
- Lambda: if you need guaranteed uptime for a 5-day 15m run (worth the premium to avoid restart)
- GCP: if vast.ai has no 256GB+ RAM machines available

---

## 8. VAST.AI SEARCH COMMANDS (COPY-PASTE READY)

```bash
# 4h TF: Cheap 4090 with good CPU
vastai search offers 'gpu_name=RTX_4090 num_gpus=1 cpu_cores>=32 cpu_ram>=128 reliability>0.95 dph<=0.50' -o 'score-'

# 1h TF: Multi-4090 or A100 with high RAM
vastai search offers 'gpu_name=RTX_4090 num_gpus>=4 cpu_cores>=64 cpu_ram>=256 reliability>0.95 dph<=2.00' -o 'score-'

# 15m TF: Maximum RAM, strong CPU+GPU
vastai search offers 'gpu_name=RTX_4090 num_gpus>=4 cpu_cores>=96 cpu_ram>=256 reliability>0.95 dph<=2.50' -o 'score-'

# 15m Alternative: H100 (check if price-competitive)
vastai search offers 'gpu_name=H100_SXM5 num_gpus>=1 cpu_cores>=64 cpu_ram>=256 reliability>0.95 dph<=3.00' -o 'score-'

# General: Show all cheap high-RAM options
vastai search offers 'cpu_cores>=64 cpu_ram>=256 reliability>0.95 dph<=3.00' -o 'dph+'
```

---

## 9. CRITICAL DEPLOYMENT REMINDERS

1. **Download artifacts at EVERY checkpoint** -- vast.ai machines die without warning
2. **1w and 1d train locally** -- renting cloud for these is wasting money
3. **RTX 4090 beats H100 per dollar** for LightGBM tree training (confirmed by benchmarks)
4. **128GB+ RAM is the hard constraint** for 1h/15m, not GPU power
5. **Never destroy machines before evaluating results** (accuracy, SHAP, confidence)
6. **Run validate.py first** on every cloud machine before training
7. **Verify all 16+ .db files** are uploaded before starting any run
8. **RIGHT_CHUNK=500** for cross gen on 1h/15m (prevents OOM)
9. **OMP_NUM_THREADS=4** for cross gen thread exhaustion prevention

---

## 10. SCORING FORMULA WORKED EXAMPLE

**Example: Scoring a vast.ai RTX 4090 node for 4h TF**

Machine: 1x RTX 4090, 48 cores @ 3.5 GHz, 128GB RAM, 24GB VRAM, $0.35/hr

```
C = 48 * 3.5 = 168 (CPU Score)
G = 1008 (GB/s bandwidth, RTX 4090)
R = 128 (GB RAM)
V = 24 (GB VRAM)

Normalize (assume max candidates: C_max=384, G_max=3350, R_max=512, V_max=80):
C_norm = 168/384 = 0.4375
G_norm = 1008/3350 = 0.3009
R_norm = 128/512 = 0.2500
V_norm = 24/80 = 0.3000

4h weights: w_cpu=0.30, w_gpu=0.35, w_ram=0.20, w_vram=0.15

Weighted = 0.30*0.4375 + 0.35*0.3009 + 0.20*0.2500 + 0.15*0.3000
         = 0.1313 + 0.1053 + 0.0500 + 0.0450
         = 0.3316

Score = 0.3316 / 0.35 = 0.947 (normalized per-dollar value)
```

Compare this to an H100 at $2.00/hr:
```
C = 128 * 2.5 = 320; G = 3350; R = 256; V = 80
C_norm=0.833, G_norm=1.0, R_norm=0.5, V_norm=1.0

Weighted = 0.30*0.833 + 0.35*1.0 + 0.20*0.5 + 0.15*1.0
         = 0.250 + 0.350 + 0.100 + 0.150
         = 0.850

Score = 0.850 / 2.00 = 0.425
```

**4090 Score: 0.947 vs H100 Score: 0.425** -- 4090 is 2.2x better value for 4h TF.

For 15m TF (RAM-heavy weights):
```
4090: 0.10*0.4375 + 0.20*0.3009 + 0.40*0.2500 + 0.30*0.3000 = 0.2539 / 0.35 = 0.725
H100: 0.10*0.833 + 0.20*1.0 + 0.40*0.5 + 0.30*1.0 = 0.783 / 2.00 = 0.392
```

**Still 4090 wins** -- only H100 at $0.70/hr or less would beat a $0.35 4090 on value.

---

*Report generated by cost-performance analysis. All prices are March 2026 estimates from Perplexity research. Verify live pricing before renting.*
