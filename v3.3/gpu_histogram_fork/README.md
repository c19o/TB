# LightGBM GPU Histogram Co-Processor Fork

## Project: Savage22 GPU-Accelerated Training

### Problem
LightGBM training on 3-6M sparse binary cross features takes 50-150 min per CPCV fold on 128-256 CPU cores. The bottleneck is histogram building from sparse CSR — iterating nonzero indices and accumulating gradients/hessians into per-feature-bin histograms. GPU sits completely idle.

### Solution
Fork LightGBM to offload sparse CSR histogram building to GPU while keeping all tree logic on CPU. The GPU builds histograms 3-10x faster using massively parallel sparse operations, then returns results to CPU for split finding.

### Why This Works for Our Workload
- **Binary features**: All 3-6M cross features are 0/1 — simplifies GPU kernel (no value loads)
- **EFB bundles**: 3-6M features compress to ~23K effective bundles — histogram fits in GPU shared memory
- **Sparse CSR on GPU**: 2-40GB sparse matrix uploaded once, stays resident in GPU HBM
- **Small gradients**: Only ~7MB transferred per boosting round (294K rows x 3 classes x 2 x 4 bytes)
- **Universal GPU support**: Must work on ANY CUDA GPU (RTX 3090, A100, H100, B200, etc.)

### Target GPU Compatibility
| GPU | VRAM | Bandwidth | CUDA CC | Notes |
|-----|------|-----------|---------|-------|
| RTX 3090 | 24 GB | 936 GB/s | 8.6 | Local dev/test. 4h sparse fits (~4GB) |
| A100 80GB | 80 GB | 2 TB/s | 8.0 | Cloud workhorse. All TFs fit |
| H100 80GB | 80 GB | 3.35 TB/s | 9.0 | Fast cloud option. All TFs fit |
| B200 192GB | 192 GB | 8 TB/s | 10.0+ | Ultimate speed. Multi-GPU overkill |
| A40 48GB | 48 GB | 696 GB/s | 8.6 | Budget cloud. 4h/1d fit, 15m tight |

**Design principle**: Auto-detect GPU VRAM at init. If sparse CSR fits in GPU memory, use GPU histogram path. If not, fall back to CPU (current behavior). No hardcoded GPU assumptions.

### Architecture
```
CPU (LightGBM tree learner)          GPU (histogram co-processor)
  |                                    |
  | 1. Upload CSR matrix once -------> | [CSR resident in HBM]
  |                                    |
  | For each boosting round:           |
  | 2. Send gradients (~7MB) --------> |
  |                                    |
  |   For each tree node:              |
  |   3. Send row mask / indices ----> |
  |                                    | 4. Build histograms (parallel)
  |   5. <---- Return histogram (~47MB)|
  |   6. Find best split (CPU)        |
  |   7. Update node partitions       |
  |                                    |
  | After all rounds:                  |
  | 8. Final model on CPU             |
```

### Expected Speedup
| TF | CPU (128c) | GPU (1x B200 est.) | Speedup |
|----|-----------|-------------------|---------|
| 4h | 50 min/fold | 8-25 min/fold | 2-6x |
| 1h | 104 min/fold | 17-52 min/fold | 2-6x |
| 15m | 150 min/fold | 25-75 min/fold | 2-6x |

### Matrix Thesis Compliance
- ALL features preserved (GPU processes identical sparse CSR)
- ALL rows used (no subsampling)
- NaN semantics preserved (structural zeros = feature OFF)
- EFB bundling identical (computed once on CPU, used by GPU)
- Histogram results bit-for-bit equivalent (same accumulation, FP ordering may differ slightly)
- feature_pre_filter=False enforced
- Production model accuracy IDENTICAL

### Project Structure
```
gpu_histogram_fork/
  README.md              # This file
  ARCHITECTURE.md        # Detailed GPU integration design
  CUDA_KERNEL_DESIGN.md  # CUDA kernel specifications
  IMPLEMENTATION_PLAN.md # Phased roadmap with milestones
  RESEARCH.md            # Collected research findings
  src/                   # Source code (when implementation begins)
    gpu_histogram.cu     # CUDA kernels
    gpu_histogram.h      # C++ interface
    integration.cpp      # LightGBM integration shim
```

### References
- LightGBM source: github.com/microsoft/LightGBM
- ThunderGBM: github.com/Xtra-Computing/thundergbm (GPU GBDT with sparse CSR support)
- Integration point: SerialTreeLearner::ConstructHistograms() in src/treelearner/serial_tree_learner.cpp
- EFB: src/io/bin.cpp, src/io/dataset.cpp

### Status
**PLANNING** — Architecture design and feasibility research in progress.
