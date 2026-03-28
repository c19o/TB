#!/usr/bin/env bash
# =============================================================================
# deploy_vastai_quick.sh — Quick deployment using pre-built wheel
# =============================================================================
#
# Minimal script for when you already have a pre-built
# lightgbm_savage-4.6.0+cuda_sparse wheel. Skips the entire build-from-source
# process (saves 10-15 minutes).
#
# Usage:
#   # Upload wheel + script:
#   scp -P PORT lightgbm_savage*.whl deploy_vastai_quick.sh root@HOST:/workspace/
#   ssh -p PORT root@HOST 'bash /workspace/deploy_vastai_quick.sh'
#
#   # Then upload code + data and train as normal.
#
# The wheel has CUDA runtime statically linked, so it works on ANY driver 525+
# without needing a CUDA toolkit on the target machine.
# =============================================================================

set -euo pipefail

LOGFILE="/workspace/deploy_quick.log"
exec > >(tee -a "$LOGFILE") 2>&1
START_TS=$(date +%s)

info()  { echo -e "\033[1;32m[$(( $(date +%s) - START_TS ))s]\033[0m $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m $*"; }
error() { echo -e "\033[1;31m[FATAL]\033[0m $*"; exit 1; }

echo "============================================================"
echo "  Quick Deploy — Pre-built LightGBM CUDA Sparse Wheel"
echo "============================================================"

# ── Step 1: GPU Detection (quick) ──
if ! command -v nvidia-smi &>/dev/null; then
    error "nvidia-smi not found. Not a GPU machine."
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | xargs)
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d '[:space:]')
VRAM_GB=$(( VRAM_MB / 1024 ))
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d '[:space:]')
DRIVER_MAJOR=$(echo "$DRIVER" | cut -d. -f1)

info "GPU: $GPU_NAME x$GPU_COUNT (${VRAM_GB}GB) | Driver: $DRIVER"

if [ "$DRIVER_MAJOR" -lt 525 ]; then
    error "Driver $DRIVER too old (need 525+)."
fi

# ── Step 2: Find wheel ──
WHEEL=""
for candidate in \
    /workspace/lightgbm_savage*.whl \
    /workspace/v3.3/gpu_histogram_fork/dist/lightgbm_savage*.whl \
    "$(dirname "$0")/dist/lightgbm_savage*.whl"; do
    # Use ls to resolve glob
    found=$(ls $candidate 2>/dev/null | head -1 || true)
    if [ -n "$found" ] && [ -f "$found" ]; then
        WHEEL="$found"
        break
    fi
done

if [ -z "$WHEEL" ]; then
    error "No lightgbm_savage*.whl found. Upload the wheel first, or use deploy_vastai.sh to build from source."
fi

info "Wheel: $WHEEL ($(du -h "$WHEEL" | cut -f1))"

# ── Step 3: Install system deps (minimal) ──
info "Installing system packages..."
apt-get update -qq 2>/dev/null || true
apt-get install -y -qq google-perftools numactl 2>/dev/null || true

# ── Step 4: Install Python deps ──
info "Installing Python packages..."
pip install -q \
    scikit-learn scipy ephem astropy pytz joblib \
    pandas numpy pyarrow optuna hmmlearn numba tqdm pyyaml psutil 2>&1 | tail -5

# CuPy
CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+' || echo "12")
CUPY_PKG="cupy-cuda${CUDA_VER}x"
pip install -q "$CUPY_PKG" 2>&1 | tail -3 || {
    pip install -q cupy-cuda12x 2>&1 | tail -3 || warn "CuPy install failed"
}

# ── Step 5: Install the wheel ──
info "Installing LightGBM wheel..."
pip uninstall -y lightgbm 2>/dev/null || true
pip install --no-deps "$WHEEL" 2>&1 | tail -5

# ── Step 6: Verify ──
info "Verifying..."
python3 -c "
import lightgbm as lgb
print(f'  LightGBM: {lgb.__version__}')
import numpy as np
X = np.random.rand(100, 10)
y = np.random.randint(0, 3, 100)
ds = lgb.Dataset(X, label=y, free_raw_data=False)
for dev in ['cuda_sparse', 'cuda', 'cpu']:
    try:
        m = lgb.train({'objective':'multiclass','num_class':3,'device_type':dev,'num_leaves':4,'verbosity':-1}, ds, num_boost_round=3)
        print(f'  device_type={dev}: OK')
        break
    except Exception as e:
        print(f'  device_type={dev}: {str(e)[:60]}')
import pandas, numpy, scipy, sklearn, ephem, astropy, pyarrow, optuna, numba, hmmlearn, yaml, tqdm, psutil
print('  All imports: OK')
" || error "Verification failed."

# ── Step 7: Performance setup ──
# tcmalloc
TCMALLOC_LIB=""
for candidate in \
    /usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 \
    /usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so \
    /usr/lib/libtcmalloc_minimal.so.4; do
    [ -f "$candidate" ] && TCMALLOC_LIB="$candidate" && break
done
if [ -z "$TCMALLOC_LIB" ]; then
    TCMALLOC_LIB=$(ldconfig -p 2>/dev/null | grep libtcmalloc_minimal | head -1 | awk '{print $NF}' || true)
fi

# THP
if [ -w /sys/kernel/mm/transparent_hugepage/enabled ]; then
    echo always > /sys/kernel/mm/transparent_hugepage/enabled
    echo defer+madvise > /sys/kernel/mm/transparent_hugepage/defrag
fi

# Kernel tuning
sysctl -w vm.swappiness=1 2>/dev/null || true
sysctl -w vm.overcommit_memory=1 2>/dev/null || true

# Launch wrapper
cat > /usr/local/bin/lgbm-run << WRAPPER_EOF
#!/bin/bash
export LD_PRELOAD="${TCMALLOC_LIB:-}"
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=10737418240
export PYTHONMALLOC=malloc
export PYTHONUNBUFFERED=1
NUMA_NODES=\$(numactl --hardware 2>/dev/null | awk '/available:/{print \$2}' || echo 1)
if [ "\$NUMA_NODES" -gt 1 ] 2>/dev/null; then
    exec numactl --interleave=all "\$@"
else
    exec "\$@"
fi
WRAPPER_EOF
chmod +x /usr/local/bin/lgbm-run

cat > /workspace/lgbm_env.sh << ENV_EOF
export LD_PRELOAD="${TCMALLOC_LIB:-}"
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=10737418240
export PYTHONMALLOC=malloc
export PYTHONUNBUFFERED=1
ENV_EOF

# ── Summary ──
ELAPSED=$(( $(date +%s) - START_TS ))
NPROC_VAL=$(nproc)
RAM_GB_SYS=$(free -g | awk '/Mem/{print $2}')

echo ""
echo "============================================================"
echo "  QUICK DEPLOY COMPLETE (${ELAPSED}s)"
echo "============================================================"
echo ""
echo "  GPU: $GPU_NAME x$GPU_COUNT (${VRAM_GB}GB) | ${NPROC_VAL}c / ${RAM_GB_SYS}GB"
echo "  LightGBM: $(python3 -c 'import lightgbm; print(lightgbm.__version__)' 2>/dev/null)"
echo "  tcmalloc: ${TCMALLOC_LIB:-not found}"
echo ""
echo "  Train: cd /workspace/v3.3 && lgbm-run python -u cloud_run_tf.py --symbol BTC --tf 1w"
echo ""
