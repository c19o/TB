# V3.3 GPU Session Resume — 2026-03-30 (Updated: Post-Optimization Company)

## INSTRUCTION TO NEW SESSION
Read this file AND v3.3/SESSION_RESUME.md. The main session resume has the full picture. This file covers GPU-specific findings.

## ACTIVE MACHINE (STILL RUNNING — $0.54/hr)
- **Instance 33852303** — Ontario CA, 1x RTX 5090 32GB, EPYC 7B12 128c, 258GB RAM
- **SSH**: `ssh -p 12302 root@ssh7.vast.ai`
- **Previous machine (DESTROYED)**: Instance 33840373, Vietnam, RTX 3060 Ti

## GPU FORK STATUS
- GPU dead-code block REMOVED (was preventing GPU from ever being used)
- __ballot_sync race condition fixed (CUDA_WARP_REDUCE=0 default)
- Error enum aligned between C header and Python wrapper
- SM_120 (RTX 5090/Blackwell) support added to cuda_compat.py
- cudaGetLastError added after kernel launches
- cuSPARSE CSR cache keyed correctly

## CRITICAL DISCOVERY: deterministic=True
This single config flag was forcing ALL LightGBM histogram computation to single-threaded mode. With it removed:
- cuda_sparse device type works (GPU training enabled)
- Multi-core CPU histograms work (128 threads instead of 1)
- Expected 10-50x training speedup

## CUDA/GPU ISSUES ON RTX 5090
1. **LSTM fails** — PyTorch installed doesn't support SM 120. Need torch 2.6+ with CUDA 12.8+
2. **cuDF unavailable** — CUDA 13+ not supported by RAPIDS cuDF. Use ALLOW_CPU=1 for feature builds.
3. **GPU fork compilation** — needs testing on SM 120. PTX fallback should work but slower than native SASS.

## max_bin=7 DISCOVERY
Binary features only need 2-3 bins, not 255. Changing max_bin from 255 to 7:
- 36x less histogram memory per leaf
- Enables 2.9M features to fit in 32GB VRAM (via EFB bundling to ~23K)
- Raw 2.9M at max_bin=255 = 660GB binned (impossible for GPU)
- EFB bundled 23K at max_bin=7 = ~8-10GB VRAM (fits easily in RTX 5090)

## MULTI-GPU FEATURES IMPLEMENTED
- Fold-parallel CPCV: 30 paths / N GPUs (subprocess isolation, CUDA_VISIBLE_DEVICES)
- Multi-GPU Sobol optimizer: Split 131K candidates across N-1 GPUs
- Parallel LSTM + optimizer: LSTM on GPU 0, optimizer on GPUs 1-N simultaneously
- torch.compile + AMP for LSTM (Tensor Core acceleration)

## VRAM REQUIREMENTS PER TF (with EFB, max_bin=7)
| TF | VRAM per GPU |
|----|-------------|
| 1w | ~4 GB |
| 1d | ~8 GB |
| 4h | ~10 GB |
| 1h | ~10 GB |
| 15m | ~10 GB |

All fit in single RTX 5090 (32GB). 8x 5090 = 256GB total.
