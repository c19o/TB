# Savage22 Training Specs: GPU Setup & Cost/Speed Comparison for vast.ai

**Date:** 2026-03-21
**Budget available:** ~$12 on vast.ai
**Previous machine:** 8x RTX 4090, 774GB RAM, $2.88/hr (Sichuan, China)
**Current stopped instance:** 4x H100 SXM 80GB, 2.2TB RAM, India, $6.53/hr (ID: 33243562)

---

## WORKLOAD BREAKDOWN (What Needs to Run)

### Phase 1: Feature Builds (6 Timeframes)
- **Nature:** 70% CPU-bound (.apply() Python UDFs), 30% GPU (CuPy DOY crosses, cuDF rolling ops)
- **RAM requirements:** 1w=2GB, 1d=5GB, 4h=25GB, 1h=80GB, 15m=250GB, 5m=450GB
- **VRAM requirements (shared across all GPUs):** 1w=1GB, 1d=2GB, 4h=10GB, 1h=35GB, 15m=50GB, 5m=80GB
- **Strategy:** 1w+1d+4h parallel, then 1h, then 15m, then 5m sequential
- **Bottleneck:** CPU clock speed (not GPU count)
- **Est. time:** ~45-60 min regardless of GPU type

### Phase 2: XGBoost Training (CPCV, 6 TFs)
- **Nature:** GPU-bound, device='cuda', histogram method
- **VRAM per split:** Depends on feature matrix density; 24GB sufficient for most TFs, 5m may need 48GB+
- **Parallel CPCV splits across GPUs** = more GPUs = faster
- **Est. time:** 30-60 min depending on GPU count/speed

### Phase 3: LSTM Training (nn.DataParallel)
- **Nature:** GPU compute + CPU DataLoader bottleneck
- **Requirement:** Strong CPU (3.5GHz+) to keep GPUs fed
- **VRAM:** 24GB per GPU sufficient
- **Est. time:** 30-45 min with good CPU, 90+ min with weak datacenter CPU

### Phase 4: Exhaustive Optimizer
- **Nature:** CuPy GPU simulation, ~30M combos per TF
- **Distributes TFs across GPUs** (6 TFs, so 4+ GPUs ideal)
- **VRAM:** Moderate (24GB per GPU OK)
- **Est. time:** 20-40 min

### Phase 5: V2 Multi-Asset (31 assets)
- **Nature:** Same pipeline but 31x daily + BTC intraday
- **RAM:** Moderate (daily TFs are small)
- **Est. time:** 15-30 min

**TOTAL PIPELINE: ~2.5-4.0 hours end-to-end**

---

## GPU PRICING TABLE (vast.ai, March 2026)

### Per-GPU Starting Rates (marketplace floor prices)

| GPU | VRAM | Arch | Mem BW | FP32 TFLOPS | $/hr (1x) |
|-----|------|------|--------|-------------|-----------|
| RTX 3090 | 24 GB GDDR6X | Ampere | 936 GB/s | 35.6 | $0.13 |
| RTX 4090 | 24 GB GDDR6X | Ada | 1,008 GB/s | 82.6 | $0.28 |
| A6000 | 48 GB GDDR6 | Ampere | 768 GB/s | 38.7 | $0.37 |
| L40S | 48 GB GDDR6 | Ada | 864 GB/s | 91.6 | $0.47 |
| A100 SXM4 | 80 GB HBM2e | Ampere | 2,039 GB/s | 19.5* | $0.73 |
| H100 SXM | 80 GB HBM3 | Hopper | 3,350 GB/s | 67.0* | $1.51 |

*A100/H100 FP32 is lower but they excel at mixed precision and have vastly superior memory bandwidth.

---

## MULTI-GPU COMPARISON TABLE

| GPU | Count | VRAM Total | Typical RAM | CPU Type | Est. $/hr | Est. Time | Est. Cost | Grade |
|-----|-------|------------|-------------|----------|-----------|-----------|-----------|-------|
| **RTX 3090** | 4x | 96 GB | 128-256 GB | Consumer/Server | $0.52 | 5-6 hr | $2.60-3.12 | BUDGET |
| **RTX 3090** | 8x | 192 GB | 256-512 GB | Server | $1.04 | 4-5 hr | $4.16-5.20 | - |
| **RTX 4090** | 4x | 96 GB | 256-512 GB | Mixed | $1.12 | 3.5-4.5 hr | $3.92-5.04 | VALUE |
| **RTX 4090** | 8x | 192 GB | 512-774 GB | Server | $2.24-2.88 | 2.5-3.5 hr | $5.60-10.08 | SWEET SPOT |
| **A6000** | 4x | 192 GB | 256-512 GB | Server | $1.48 | 4-5 hr | $5.92-7.40 | - |
| **A6000** | 8x | 384 GB | 512-1024 GB | Server | $2.96 | 3-4 hr | $8.88-11.84 | - |
| **L40S** | 4x | 192 GB | 256-512 GB | Server (Xeon) | $1.88 | 3-4 hr | $5.64-7.52 | RUNNER-UP |
| **L40S** | 8x | 384 GB | 512-1024 GB | Server (Xeon) | $3.76 | 2.5-3.5 hr | $9.40-13.16 | - |
| **A100 SXM4** | 4x | 320 GB | 512-2048 GB | EPYC/Xeon | $2.92 | 2.5-3.5 hr | $7.30-10.22 | PROVEN |
| **A100 SXM4** | 8x | 640 GB | 1024-2048 GB | EPYC/Xeon | $5.84 | 2-3 hr | $11.68-17.52 | - |
| **H100 SXM** | 4x | 320 GB | 1024-2048 GB | EPYC (weak!) | $6.04 | 2-2.5 hr | $12.08-15.10 | FASTEST |
| **H100 SXM** | 8x | 640 GB | 2048+ GB | EPYC (weak!) | $12.08 | 1.5-2 hr | $18.12-24.16 | OVERKILL |

---

## TOP 3 RECOMMENDATIONS (Ranked by Cost-Effectiveness)

### #1 BEST VALUE: 8x RTX 4090, 512GB+ RAM
- **Price:** ~$2.24-2.88/hr (your last machine was $2.88)
- **Total VRAM:** 192 GB (24 GB x 8)
- **Why it wins:**
  - Best FP32 performance per dollar of ANY GPU on the list
  - 8 GPUs = 8 parallel CPCV splits, 6+ TFs distributed in optimizer
  - Consumer-grade Ada architecture has excellent cuDF/CuPy perf
  - 192 GB total VRAM handles all workloads including DOY crosses
  - Often paired with decent CPUs (AMD EPYC) in 8x configs
  - $2.88/hr x 3 hours = ~$8.64 total (fits $12 budget)
- **Risk:** LSTM DataLoader bottleneck if CPU is weak (<3 GHz)
- **Mitigation:** Filter for machines with CPU GHz >= 3.0

### #2 RUNNER-UP: 4x L40S, 512GB+ RAM
- **Price:** ~$1.88/hr
- **Total VRAM:** 192 GB (48 GB x 4)
- **Why:**
  - 48 GB per GPU = comfortable headroom for large feature matrices
  - Ada architecture (same gen as 4090) with better enterprise stability
  - Datacenter CPUs (usually Xeon) tend to be more reliable
  - Fewer GPUs but each has 2x the VRAM of a 4090
  - $1.88/hr x 3.5 hours = ~$6.58 total (excellent budget fit)
- **Risk:** Only 4 GPUs limits parallelism for CPCV/optimizer
- **Good for:** If you want stability over raw speed

### #3 PROVEN PERFORMER: 4x A100 SXM4 80GB, 512GB+ RAM
- **Price:** ~$2.92/hr (your stopped instance is $6.53 for 4x H100)
- **Total VRAM:** 320 GB (80 GB x 4)
- **Why:**
  - This is your PROVEN setup (instance 33243562 is 4x H100, similar)
  - 80 GB per GPU = no OOM risk even on 5m DOY crosses
  - HBM2e memory bandwidth (2 TB/s) excels at XGBoost histogram ops
  - Massive system RAM typically available (512GB-2TB)
  - $2.92/hr x 3 hours = ~$8.76 total
- **Risk:** Higher per-hour cost, EPYC CPUs can be slow for LSTM
- **Good for:** If you need guaranteed success (you know it works)

---

## MINIMUM VIABLE SETUP (Cheapest That Works)

**4x RTX 4090, 512GB RAM** @ ~$1.12/hr
- Total: ~$4.48-5.60 for full pipeline
- 96 GB VRAM handles all builds except 5m needs staggering
- May need to run 5m DOY crosses with reduced batch size
- LSTM will be fine, XGBoost fine for most TFs
- Optimizer distributes 6 TFs across 4 GPUs (some queuing)

**Warning:** 96 GB VRAM means 5m build (needs 80GB shared) has to run alone with all 4 GPUs visible. Works but tight.

## FASTEST POSSIBLE SETUP

**8x H100 SXM, 2TB+ RAM** @ ~$12.08/hr
- Total: ~$18-24 for full pipeline (~1.5-2 hours)
- 640 GB VRAM = everything fits with room to spare
- HBM3 bandwidth dominates XGBoost training
- **BUT:** weak EPYC CPUs will bottleneck LSTM DataLoader
- **AND:** exceeds your $12 budget for a single run
- Only worth it if you need to iterate very fast and are adding funds

---

## YOUR PREVIOUS MACHINE vs. NOW

**Previous:** 8x RTX 4090, 774GB RAM, $2.88/hr (Sichuan, China)
- This was an excellent pick. 774 GB RAM handles even 5m builds.
- $2.88/hr is slightly above the $2.24 floor but reasonable for a high-RAM config.
- **Verdict:** Hard to beat. Look for the same config or similar.

**Stopped instance:** 4x H100 SXM, 2.2TB RAM, $6.53/hr (India)
- More expensive per hour but proven. 2.2TB RAM is overkill.
- Only restart if you need guaranteed results fast.
- **Verdict:** Use as fallback if 8x 4090 not available.

---

## VAST.AI CLI SEARCH COMMANDS

### Find 8x RTX 4090 with 512GB+ RAM (RECOMMENDED)
```bash
vastai search offers 'gpu_name=RTX_4090 num_gpus=8 cpu_ram>=512 reliability>=0.95' -o 'dph_total' --limit 20
```

### Find 4x RTX 4090 with 512GB+ RAM (BUDGET)
```bash
vastai search offers 'gpu_name=RTX_4090 num_gpus=4 cpu_ram>=512 reliability>=0.95' -o 'dph_total' --limit 20
```

### Find 4x L40S with 512GB+ RAM (RUNNER-UP)
```bash
vastai search offers 'gpu_name=L40S num_gpus>=4 cpu_ram>=512 reliability>=0.95' -o 'dph_total' --limit 20
```

### Find 4x A100 SXM4 80GB with 512GB+ RAM (PROVEN)
```bash
vastai search offers 'gpu_name=A100_SXM4 num_gpus>=4 cpu_ram>=512 reliability>=0.95' -o 'dph_total' --limit 20
```

### Find ANY 8-GPU machine with 512GB+ RAM, sorted cheapest first
```bash
vastai search offers 'num_gpus=8 cpu_ram>=512 gpu_ram>=24 reliability>=0.95' -o 'dph_total' --limit 30
```

### Find ANY 4-GPU machine with 48GB+ VRAM each (A6000/L40S/A100/H100)
```bash
vastai search offers 'num_gpus>=4 cpu_ram>=512 gpu_ram>=48 reliability>=0.95' -o 'dph_total' --limit 30
```

---

## BUDGET ANALYSIS ($12 Available)

| Setup | $/hr | Max Hours | Fits Full Pipeline? |
|-------|------|-----------|---------------------|
| 8x RTX 4090 @ $2.88 | $2.88 | 4.2 hr | YES (est. 2.5-3.5 hr) |
| 4x RTX 4090 @ $1.12 | $1.12 | 10.7 hr | YES (est. 3.5-4.5 hr) |
| 4x L40S @ $1.88 | $1.88 | 6.4 hr | YES (est. 3-4 hr) |
| 4x A100 @ $2.92 | $2.92 | 4.1 hr | YES (est. 2.5-3.5 hr) |
| 4x H100 @ $6.04 | $6.04 | 2.0 hr | RISKY (est. 2-2.5 hr) |
| 8x H100 @ $12.08 | $12.08 | 1.0 hr | NO (need 1.5-2 hr) |

**Clear winner for $12 budget: 8x RTX 4090 at ~$2.88/hr**
- Gives you ~4 hours of runway
- Pipeline completes in ~3 hours
- Leaves ~$3.50 buffer for debugging/reruns

---

## CRITICAL REMINDERS

1. **Docker image:** `rapidsai/base:25.02-cuda12.5-py3.12`
2. **DO NOT pin CUDA_VISIBLE_DEVICES** — let all GPUs be visible to all processes
3. **Strip timezones** before any cuDF operations
4. **CuPy arrays must be contiguous** — use `np.ascontiguousarray()` before `cp.asarray()`
5. **Save parquet FIRST**, SQLite as non-fatal backup (2000 column limit)
6. **PYTHONUNBUFFERED=1** always set for progress visibility
7. **Feature builds staggering:** small TFs parallel, then 1h, then 15m, then 5m solo
8. **Upload SSH key to vast.ai account BEFORE renting**

---

## FINAL RECOMMENDATION

**Search for 8x RTX 4090 with 512-774GB RAM first.** Your last Sichuan machine at $2.88/hr was excellent. If that exact config isn't available, the 4x L40S at ~$1.88/hr is the safest fallback — 48GB per GPU eliminates VRAM pressure and datacenter CPUs are more predictable. The 4x A100 is proven (you have instance 33243562 as backup) but costs more.

Run this search first:
```bash
vastai search offers 'gpu_name=RTX_4090 num_gpus=8 cpu_ram>=512 reliability>=0.95' -o 'dph_total' --limit 20
```

If nothing good shows up, try:
```bash
vastai search offers 'gpu_name=L40S num_gpus>=4 cpu_ram>=512 reliability>=0.95' -o 'dph_total' --limit 20
```
