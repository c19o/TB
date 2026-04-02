# 2026-04-02 Datacenter GPU Feasibility

## Question

Would Savage22 materially benefit from commercial datacenter GPUs such as NVIDIA B200 or older enterprise/data-center cards, compared with the RTX-class machines already being certified?

## Short Answer

- `1w`: no meaningful benefit
- `1d`: limited benefit right now
- `4h`: moderate benefit if retrain becomes reliably hybrid/GPU
- `1h`: real benefit
- `15m`: strongest benefit

B200/H100/H200-class hardware is only economically justified once the lower-timeframe retrain path is truly GPU-native. Today, the practical datacenter sweet spot is older enterprise cards with large VRAM and stable server thermals, such as `A40`, `L40S`, `RTX 6000 Ada`, and `A100/H100` when available at sane cost.

## Why Datacenter GPUs Help

Datacenter GPUs buy four things that matter to this project:

1. Much larger VRAM and memory bandwidth
- `A40`: `48GB` GDDR6, about `696-798 GB/s`, PCIe Gen4, optional NVLink, passive data-center form factor
- `RTX 6000 Ada`: `48GB`, around `960 GB/s`
- `A100 80GB PCIe`: `80GB` HBM2e, about `1.9 TB/s`
- `H100 NVL`: `94GB`, about `3.9 TB/s`
- `B200`: `180GB` HBM3e per GPU in HGX/DGX configurations, with very high aggregate NVLink/NVSwitch bandwidth

2. Better memory safety for large sparse workloads
- ECC/HBM and data-center thermals matter when the same machine runs long training jobs repeatedly.

3. Better multi-GPU scaling infrastructure
- NVLink / NVSwitch are relevant once retrain and lower-TF folds are genuinely GPU-dispatched.

4. Better server fit
- passive or data-center-validated boards are better suited for long rented-server duty than consumer cards in mixed chassis conditions.

## Why They Do Not Automatically Fix The Pipeline

The current bottleneck is not “insufficient FLOPS everywhere.”

Current reality:
- `1w` is trimmed and CPU-first
- `1d` is still largely CPU-first in the maintained path
- the pipeline still contains CPU orchestration, sparse bookkeeping, checkpointing, validation, and file movement
- `step5_retrain` is not yet a fully certified GPU-native backend on all lower TFs

That means a `B200` does not automatically turn the whole pipeline into a fast GPU shop. It only pays off once the hot path is actually on GPU.

## Practical Fit By GPU Class

### B200 / HGX B200

Best use:
- future `1h` / `15m`
- large same-machine GPU lanes
- extremely memory-heavy multi-GPU retrain once the backend is certified

Current feasibility:
- technically powerful, but not the best next purchase/rental target for Savage22 today
- too expensive and too specialized relative to the current code path
- the current pipeline would underutilize it on `1w` and `1d`

Verdict:
- not the next practical step
- worthwhile only after GPU-native retrain is certified for lower TFs

### H100 / H200 / A100

Best use:
- lower TFs where multi-GPU fold training and large VRAM matter
- high-memory, high-throughput retrain lanes

Current feasibility:
- more realistic than B200 for this project
- still only fully justified for `1h` / `15m`, and maybe later `4h`

Verdict:
- credible upper-tier target for a later certified GPU lane
- still overkill for `1w`
- only partially helpful for `1d` until the retrain backend changes

### L40S / A40 / RTX 6000 Ada

Best use:
- practical enterprise path for this repo right now
- `4h`, `1h`, and `15m` where larger VRAM and more stable server behavior help
- same-machine lanes that need more VRAM without jumping to hyperscaler-class hardware

Current feasibility:
- good
- these are the most relevant datacenter/commercial GPU classes for Savage22 in the near term

Verdict:
- best enterprise-style fit before B200-class hardware

## Recommendation

Near-term private-shop path:

1. Keep `1w` and `1d` on CPU-first / high-RAM lanes unless a certified GPU retrain backend lands.
2. Certify `4h` as the first serious hybrid lane.
3. For enterprise/datacenter GPU experimentation, prefer:
   - `A40`
   - `L40S`
   - `RTX 6000 Ada`
   - `A100/H100` if rental economics are good
4. Revisit `B200` only after:
   - `1h` and `15m` retrain are truly GPU-native
   - same-machine lower-TF resume is certified
   - the pipeline can actually exploit NVLink/NVSwitch-class scaling

## Sources

Official NVIDIA sources used:
- NVIDIA A40: https://www.nvidia.com/en-us/data-center/a40/
- NVIDIA RTX 6000 Ada: https://www.nvidia.com/en-us/products/workstations/rtx-6000/
- NVIDIA A100: https://www.nvidia.com/en-us/data-center/a100/
- NVIDIA H100: https://www.nvidia.com/en-us/data-center/h100/
- NVIDIA DGX B200 / HGX B200 references:
  - https://www.nvidia.com/en-us/data-center/dgx-b200/
  - https://www.nvidia.com/en-us/data-center/hgx
  - https://docs.nvidia.com/cuda/archive/12.8.0/blackwell-tuning-guide/index.html

Local research sources used:
- `LightGBM_GPU_Acceleration_Paper.pdf`
- `GBDT_Benchmarking_2023.pdf`
- `CUDA_C_Programming_Guide.pdf`
- `Systems.Performance.Enterprise.and.the.Cloud.2nd.Edition.2020.12.pdf`
