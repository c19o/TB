# vast.ai Deployment Guide -- GPU Histogram Fork

Complete guide for deploying the LightGBM CUDA sparse histogram fork on vast.ai machines. This fork offloads histogram building from CPU to GPU, delivering 71-473x speedup on the histogram pass for sparse binary cross features.

---

## 1. Machine Selection

### Minimum Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| GPU | 1x 24GB+ VRAM | 1x A100 80GB |
| CUDA CC | 8.0+ (Ampere) | 8.0+ |
| NVIDIA Driver | 525+ | 535+ |
| RAM | 64GB (1w), 256GB (1d) | 512GB+ |
| CPU Cores | 32+ | 64+ |
| Disk | 50GB+ | 100GB+ |

### GPU Tiers

| GPU | VRAM | Bandwidth | CC | Price (~) | Handles |
|-----|------|-----------|-----|-----------|---------|
| RTX 4090 | 24 GB | 1008 GB/s | 8.9 | $0.30-0.50/hr | 1w, 1d, 4h |
| RTX 3090 | 24 GB | 936 GB/s | 8.6 | $0.20-0.35/hr | 1w, 1d, 4h |
| A40 | 48 GB | 696 GB/s | 8.6 | $0.30-0.50/hr | 1w, 1d, 4h, 1h (tight) |
| A100 80GB | 80 GB | 2 TB/s | 8.0 | $1.00-1.80/hr | ALL (1w-15m) |
| H100 80GB | 80 GB | 3.35 TB/s | 9.0 | $2.00-3.50/hr | ALL (1w-15m) |
| H200 141GB | 141 GB | 4.8 TB/s | 9.0 | $3.00-5.00/hr | ALL + headroom |

**Single GPU is sufficient.** The histogram fork uses one GPU for histogram building while CPU handles tree logic. Multi-GPU is not needed.

**A100 80GB is the recommended workhorse** -- handles all timeframes including 15m, has 2 TB/s bandwidth for fast SpMV, and is widely available on vast.ai at reasonable prices.

### vast.ai Search Filters

Go to https://cloud.vast.ai/search and apply:

```
GPU RAM:  >= 24 GB          (minimum for 1w/1d/4h)
         >= 80 GB          (if running 1h or 15m)
CPU RAM:  >= 64 GB          (1w), >= 256 GB (1d), >= 512 GB (4h+)
CPU Cores: >= 32
CUDA Version: >= 12.0
Driver Version: >= 525
Disk Space: >= 50 GB
GPU Model: A100, H100, H200, A40, RTX 4090, RTX 3090 (any of these)
Docker Image: pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
```

For the vast.ai CLI:
```bash
# Search for A100 80GB machines with 256GB+ RAM
vastai search offers \
    'gpu_ram >= 80 cpu_ram >= 256 num_gpus >= 1 cuda_vers >= 12.0 \
     driver_version >= 525 disk_space >= 50 reliability > 0.95' \
    --order 'dph_total' \
    --type on-demand

# Budget option: RTX 4090/3090 for 1w/1d/4h only
vastai search offers \
    'gpu_ram >= 24 cpu_ram >= 64 num_gpus >= 1 cuda_vers >= 12.0 \
     driver_version >= 525 disk_space >= 50 reliability > 0.95 \
     gpu_name in [RTX_4090, RTX_3090, A40]' \
    --order 'dph_total' \
    --type on-demand
```

### Per-TF Machine Requirements

| TF | Min GPU VRAM | Min System RAM | Min CPU Cores | Recommended GPU |
|----|-------------|----------------|---------------|-----------------|
| 1w | 24 GB | 64 GB | 32 | Any 24GB+ |
| 1d | 24 GB | 256 GB | 64 | Any 24GB+ |
| 4h | 24 GB | 512 GB | 64 | Any 24GB+ |
| 1h | 80 GB | 768 GB | 128 | A100 80GB |
| 15m | 80 GB | 1024 GB | 128 | A100 80GB |

---

## 2. Quick Deploy (Pre-Built Wheel)

Use this path when you have a pre-built `lightgbm_savage*.whl` file. This is the fastest deployment -- no CUDA toolkit or cmake needed on the target machine.

### Prerequisites on local machine

Build the wheel first (requires CUDA toolkit):
```bash
cd v3.3/gpu_histogram_fork
bash build_wheel.sh
# Output: dist/lightgbm_savage-4.6.0+cuda_sparse-cp312-cp312-linux_x86_64.whl
```

### Step-by-step

```bash
# ── 1. Rent machine on vast.ai ──
# Use the search filters above, then:
vastai create instance INSTANCE_ID \
    --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime \
    --disk 80 \
    --onstart-cmd 'pip install -q scipy scikit-learn ephem astropy pytz joblib pyarrow optuna hmmlearn numba tqdm pyyaml psutil'

# ── 2. Wait for machine to be ready, get SSH info ──
vastai show instances --raw | grep -E 'ssh_host|ssh_port'
# Note HOST and PORT from output

# ── 3. Upload wheel + code + DBs ──
# From your local machine:
PORT=XXXXX  # from vast.ai
HOST=ssh.vast.ai  # or the specific host

# Upload the pre-built wheel
scp -P $PORT v3.3/gpu_histogram_fork/dist/lightgbm_savage*.whl root@$HOST:/workspace/

# Upload v3.3 code tarball (all .py + .json files)
cd "Savage22 Server"
tar czf /tmp/v33_code.tar.gz v3.3/*.py v3.3/*.json v3.3/gpu_histogram_fork/*.py \
    v3.3/gpu_histogram_fork/src/*.py v3.3/gpu_histogram_fork/check_gpu.py \
    v3.3/astrology_engine.py
scp -P $PORT /tmp/v33_code.tar.gz root@$HOST:/workspace/

# Upload databases (ALL of them -- missing DB = weaker model)
scp -P $PORT btc_prices.db tweets.db news_articles.db sports_results.db \
    space_weather.db onchain_data.db macro_data.db astrology_full.db \
    ephemeris_cache.db fear_greed.db funding_rates.db google_trends.db \
    open_interest.db multi_asset_prices.db llm_cache.db v2_signals.db \
    kp_history_gfz.txt root@$HOST:/workspace/

# ── 4. SSH in and set up ──
ssh -p $PORT root@$HOST

# Install the wheel (replaces stock LightGBM)
pip install /workspace/lightgbm_savage*.whl --force-reinstall --no-deps

# Extract code
cd /workspace
tar xzf v33_code.tar.gz

# Symlink DBs into v3.3 directory
ln -sf /workspace/*.db /workspace/v3.3/
ln -sf /workspace/btc_prices.db /workspace/
ln -sf /workspace/kp_history_gfz.txt /workspace/v3.3/

# ── 5. Verify GPU + fork ──
cd /workspace/v3.3
python gpu_histogram_fork/check_gpu.py

# Expected output should show:
#   - GPU detected with correct VRAM
#   - LightGBM cuda_sparse: SUPPORTED
#   - Per-TF fit analysis showing which TFs fit in VRAM

# ── 6. Verify DB count ──
ls /workspace/*.db | wc -l
# Must be >= 16. If not, STOP and upload missing DBs.

# ── 7. Run training ──
export V30_DATA_DIR=/workspace
export PYTHONUNBUFFERED=1

# Train with GPU histogram acceleration
python -u cloud_run_tf.py --symbol BTC --tf 1w 2>&1 | tee /workspace/train_1w.log
```

---

## 3. Full Build Deploy (From Source)

Use this path when you do not have a pre-built wheel, or the target machine has a different Python version / CUDA architecture than the wheel was built for.

### Requirements on target machine

- CUDA toolkit 11.8+ (nvcc on PATH)
- cmake 3.18+
- g++ / gcc
- git
- Python 3.10+

Most vast.ai images with `cuda` in the name have these pre-installed.

### Step-by-step

```bash
# ── 1. Rent machine (same search filters as above) ──
# Use an image that includes CUDA toolkit:
vastai create instance INSTANCE_ID \
    --image nvidia/cuda:12.4.1-devel-ubuntu22.04 \
    --disk 100 \
    --onstart-cmd 'apt-get update && apt-get install -y cmake git python3-pip'

# ── 2. SSH in ──
ssh -p $PORT root@$HOST

# ── 3. Upload code + DBs (same as Quick Deploy steps 3) ──
# ... (from local machine, same scp commands as above)

# ── 4. Build from source ──
cd /workspace/v3.3/gpu_histogram_fork

# Option A: Use the full build+test script (~10-15 min)
bash build_and_test.sh --clone-dir /tmp/lightgbm-fork

# Option B: Build wheel only (~8-12 min)
bash build_wheel.sh

# Option C: Use the v3.3 setup script (includes system optimizations)
cd /workspace/v3.3
bash setup.sh
# Then build the fork on top:
cd gpu_histogram_fork
bash build_wheel.sh
pip install dist/lightgbm_savage*.whl --force-reinstall --no-deps

# ── 5. Verify ──
python check_gpu.py --bench
# The --bench flag runs a cuSPARSE SpMV micro-benchmark to confirm GPU acceleration works.
# Expect:
#   SciPy baseline: ~X ms/iter
#   CuPy cuSPARSE:  ~Y ms/iter
#   Speedup: 50-500x depending on matrix size

# ── 6. Train (same as Quick Deploy step 7) ──
cd /workspace/v3.3
export V30_DATA_DIR=/workspace
export PYTHONUNBUFFERED=1
python -u cloud_run_tf.py --symbol BTC --tf 1w 2>&1 | tee /workspace/train_1w.log
```

### Build Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_ARCHS` | `80;86;89;90` | Semicolon-separated SM targets for fat binary |
| `NPROC` | `$(nproc)` | Parallel compile jobs |
| `PYTHON` | `python3` | Python interpreter path |
| `CLONE_DIR` | `/tmp/lightgbm-fork` | Where to clone LightGBM source |

The build script automatically adds `sm_100` (B200/B100) if CUDA toolkit >= 12.8 is detected.

---

## 4. Per-TF GPU Compatibility

### VRAM Requirements

| TF | Rows | Features | CSR Size | GPU Needed | Est. Histogram Speedup |
|----|------|----------|----------|------------|----------------------|
| 1w | 818 | 2.2M | ~165 MB | Any 24GB+ | 71x (measured) |
| 1d | 5,727 | 6M | ~2.6 GB | Any 24GB+ | 473x (measured) |
| 4h | 22,908 | ~6M | ~4 GB | Any 24GB+ | ~350x (estimated) |
| 1h | 91,632 | ~10M | ~28 GB | A100 80GB | ~200x (estimated) |
| 15m | 227,000 | ~10M | ~50 GB | A100 80GB | ~150x (estimated) |

**Speedup column** refers to the histogram building step only (cuSPARSE SpMV vs scipy CSR scan). The end-to-end training speedup depends on what fraction of total training time is histogram building vs. split finding, tree construction, gradient computation, etc.

### GPU VRAM Fit Matrix

| GPU | VRAM | 1w | 1d | 4h | 1h | 15m |
|-----|------|-----|-----|-----|-----|------|
| RTX 3090 | 24 GB | YES | YES | YES | NO | NO |
| RTX 4090 | 24 GB | YES | YES | YES | NO | NO |
| A40 | 48 GB | YES | YES | YES | TIGHT | NO |
| A100 | 80 GB | YES | YES | YES | YES | YES |
| H100 | 80 GB | YES | YES | YES | YES | YES |
| H200 | 141 GB | YES | YES | YES | YES | YES |
| B200 | 192 GB | YES | YES | YES | YES | YES |

VRAM budget uses 85% safety factor (never allocate more than 85% of total VRAM). "TIGHT" means it fits within the 85% margin but leaves little headroom.

### Fallback Behavior

If the CSR matrix does not fit in GPU VRAM, the fork automatically falls back to CPU histogram building (standard LightGBM behavior). No code changes needed. The `check_gpu.py` script reports which TFs will use GPU vs CPU before training starts.

---

## 5. Troubleshooting

### "Unknown device type cuda_sparse"

**Cause:** The LightGBM fork is not installed. Stock LightGBM does not recognize `cuda_sparse`.

**Fix:**
```bash
# Check which LightGBM is installed
python -c "import lightgbm; print(lightgbm.__version__)"
# If it does NOT contain "cuda_sparse" in the version string:

# Rebuild from source
cd /workspace/v3.3/gpu_histogram_fork
bash build_wheel.sh --clean
pip install dist/lightgbm_savage*.whl --force-reinstall --no-deps

# Verify
python check_gpu.py
```

### "No external CSR set" or "CSR not uploaded to GPU"

**Cause:** The training pipeline is not calling `SetExternalCSR()` before `lgb.train()`. The GPU histogram path needs the sparse CSR matrix uploaded to GPU memory before training begins.

**Fix:** Ensure the integration layer is active. In `ml_multi_tf.py`, the `get_training_params()` function from `gpu_histogram_fork.src.train_pipeline` should be called before training:
```python
from gpu_histogram_fork.src.train_pipeline import get_training_params
params = get_training_params(base_params, X_all, tf_name=tf_name)
```

### CUDA Out of Memory (OOM)

**Cause:** The CSR matrix for this timeframe is too large for the GPU's VRAM.

**Fix:**
```bash
# Check what fits
python gpu_histogram_fork/check_gpu.py

# If the TF doesn't fit, the fork falls back to CPU automatically.
# To force CPU mode for debugging:
export GPU_HISTOGRAM_FORCE_CPU=1
python -u cloud_run_tf.py --symbol BTC --tf 1h
```

For 1h/15m on 24GB GPUs, the only real fix is using a GPU with more VRAM (A100 80GB or higher).

### Driver Too Old

**Cause:** NVIDIA driver < 525 does not support the CUDA features used by the fork.

**Fix:**
```bash
# Check driver version
nvidia-smi | head -3

# If driver < 525, find a different machine on vast.ai
# Use the search filter: driver_version >= 535
```

### Build Fails: "nvcc not found"

**Cause:** CUDA toolkit is not installed or not on PATH.

**Fix:**
```bash
# Check if CUDA toolkit exists
ls /usr/local/cuda*/bin/nvcc

# Add to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# If no CUDA toolkit at all, use the Quick Deploy path (pre-built wheel)
# or rent a machine with a devel image (nvidia/cuda:12.4.1-devel-ubuntu22.04)
```

### Build Fails: cmake errors

**Fix:**
```bash
# Ensure cmake >= 3.18
cmake --version

# If too old:
pip install cmake --upgrade
# or
apt-get install -y cmake

# Clean rebuild
cd /workspace/v3.3/gpu_histogram_fork
bash build_wheel.sh --clean
```

### cuSPARSE SpMV benchmark shows no speedup

**Cause:** CuPy not installed, or wrong CUDA version.

**Fix:**
```bash
# Install CuPy matching your CUDA version
pip install cupy-cuda12x  # for CUDA 12.x
# or
pip install cupy-cuda11x  # for CUDA 11.x

# Verify
python -c "import cupy; print(cupy.__version__); print(cupy.cuda.runtime.runtimeGetVersion())"
```

### Training runs but GPU utilization is 0%

**Cause:** The fork fell back to CPU, or `device_type` was not set to `cuda_sparse`.

**Check:**
```bash
# Monitor GPU during training
watch -n 1 nvidia-smi

# Check training logs for:
#   "Using GPU histogram path (cuda_sparse)" = GPU active
#   "Falling back to CPU histogram path"     = GPU not used
#   "GPU VRAM insufficient"                  = CSR too large
```

---

## 6. Integration with cloud_run_tf.py

To use GPU histograms in the existing training pipeline, modify `ml_multi_tf.py` at the point where LightGBM params are constructed.

### Minimal Integration (3 lines)

In `ml_multi_tf.py`, before the CPCV loop where `lgb.train()` is called:

```python
# ── At top of file, add import ──
try:
    from gpu_histogram_fork.src.train_pipeline import get_training_params
    _GPU_HIST_AVAILABLE = True
except ImportError:
    _GPU_HIST_AVAILABLE = False

# ── Before lgb.train() calls, modify params ──
# Replace:
#   params = {... your existing params ...}
# With:
if _GPU_HIST_AVAILABLE:
    params = get_training_params(params, X_all, tf_name=tf_name)
# This auto-detects GPU, checks VRAM, sets device_type='cuda_sparse' if possible,
# and falls back to CPU if not. No other code changes needed.
```

### What get_training_params() Does

1. Detects CUDA devices via CuPy
2. Estimates VRAM needed for the CSR matrix based on its shape and NNZ
3. If GPU has enough VRAM (with 85% safety margin), sets `device_type='cuda_sparse'`
4. If GPU insufficient or not available, leaves params unchanged (CPU path)
5. Logs the decision: GPU path chosen, fallback reason, VRAM numbers

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GPU_HISTOGRAM_FORCE_CPU` | unset | Set to `1` to force CPU histogram path (debugging) |
| `CUDA_VISIBLE_DEVICES` | all | Restrict which GPU is used (e.g., `0` for first GPU) |

### cloud_run_tf.py Changes

No changes needed to `cloud_run_tf.py` itself. The integration lives entirely in `ml_multi_tf.py` via the `get_training_params()` call. `cloud_run_tf.py` orchestrates the pipeline (feature building, cross gen, training) and calls `ml_multi_tf.py` for the training step -- the GPU histogram logic is transparent to it.

---

## 7. Workflow Summary

### For 1w / 1d / 4h (fits on any 24GB+ GPU)

```
1. Rent RTX 4090 or RTX 3090 on vast.ai ($0.20-0.50/hr)
2. Quick Deploy with pre-built wheel
3. python check_gpu.py  (verify all 3 TFs show "YES")
4. Train each TF:
     python -u cloud_run_tf.py --symbol BTC --tf 1w
     python -u cloud_run_tf.py --symbol BTC --tf 1d
     python -u cloud_run_tf.py --symbol BTC --tf 4h
5. Download artifacts after EACH TF completes
6. Destroy instance
```

### For 1h / 15m (requires 80GB+ VRAM)

```
1. Rent A100 80GB on vast.ai ($1.00-1.80/hr)
2. Quick Deploy with pre-built wheel
3. python check_gpu.py  (verify 1h and 15m show "YES")
4. Train:
     python -u cloud_run_tf.py --symbol BTC --tf 1h
     python -u cloud_run_tf.py --symbol BTC --tf 15m
5. Download artifacts after EACH TF completes
6. Destroy instance
```

### Download Artifacts (MANDATORY before destroying machine)

```bash
# From local machine:
scp -P $PORT root@$HOST:/workspace/v3.3/model_*.json .
scp -P $PORT root@$HOST:/workspace/v3.3/inference_*.json .
scp -P $PORT root@$HOST:/workspace/v3.3/v2_cross_names_*.json .
scp -P $PORT root@$HOST:/workspace/v3.3/optuna_configs_*.json .
scp -P $PORT root@$HOST:/workspace/v3.3/feature_importance_*.json .
scp -P $PORT root@$HOST:/workspace/v3.3/ml_multi_tf_results.txt .
scp -P $PORT "root@$HOST:/workspace/v3.3/*.log" .
```

---

## 8. Performance Expectations

### Histogram Speedup (GPU vs CPU)

These numbers are from actual benchmarks on RTX 3090 (24GB, 936 GB/s):

| Matrix Shape | CPU (scipy) | GPU (cuSPARSE) | Speedup | BW Utilization |
|-------------|-------------|----------------|---------|---------------|
| 1w: 818 x 2.2M | 38 ms | 0.24 ms | 160x | 45% |
| 1d: 5727 x 3M | 359 ms | 0.98 ms | 367x | 73% |
| 1d: 5727 x 6M | 901 ms | 1.90 ms | 473x | 75% |

On A100 80GB (2 TB/s bandwidth), expect ~2x faster than RTX 3090 due to higher memory bandwidth.

On H100 80GB (3.35 TB/s bandwidth), expect ~3.5x faster than RTX 3090.

### End-to-End Training Speedup

Histogram building is one part of the training loop. The overall speedup depends on how much time is spent in histograms vs. other operations (split finding, gradient computation, tree construction, EFB bundling, etc.).

Expected end-to-end per-fold speedup:

| TF | CPU-only (128c) | With GPU Histograms | Est. Speedup |
|----|----------------|-------------------|-------------|
| 1w | ~5-10 min/fold | ~3-5 min/fold | 1.5-2x |
| 1d | ~15-30 min/fold | ~5-10 min/fold | 2-3x |
| 4h | ~50 min/fold | ~8-25 min/fold | 2-6x |
| 1h | ~104 min/fold | ~17-52 min/fold | 2-6x |
| 15m | ~150 min/fold | ~25-75 min/fold | 2-6x |

The larger the CSR matrix relative to other training overhead, the bigger the speedup. 1d and 4h benefit the most because histogram building dominates their training time.

---

## 9. Cost Optimization

### Cost per full training run (all 5 TFs)

| Strategy | Machine | Est. Time | Est. Cost |
|----------|---------|-----------|-----------|
| Budget | RTX 4090 for 1w/1d/4h, then A100 for 1h/15m | ~8-12 hr total | $8-15 |
| Fast | A100 80GB for all TFs | ~6-10 hr total | $8-18 |
| Maximum speed | H100 80GB for all TFs | ~4-7 hr total | $12-25 |

### Tips

- **Rent on-demand, not interruptible** -- training is stateful, interruptions waste progress
- **Download artifacts after each TF** -- vast.ai machines can die without warning
- **Start with 1w** to verify the pipeline works (smallest, cheapest, fastest)
- **Use `--onstart-cmd`** to pre-install dependencies while the machine boots
- **Do not over-provision GPU count** -- the fork uses exactly 1 GPU, extra GPUs are wasted money
- **CPU core count still matters** -- split finding, EFB bundling, gradient computation, and cross generation all run on CPU. Pick machines with high CPU Score (cores x base GHz)
