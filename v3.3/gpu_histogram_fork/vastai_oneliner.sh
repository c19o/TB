#!/usr/bin/env bash
# Savage22 GPU Histogram — one-liner cloud setup
# curl -sL https://raw.githubusercontent.com/c19o/TB/v3.3/v3.3/gpu_histogram_fork/vastai_oneliner.sh | bash
set -euo pipefail

G='\033[1;32m'; Y='\033[1;33m'; R='\033[1;31m'; C='\033[1;36m'; N='\033[0m'
ok()   { echo -e "  ${G}OK${N} $*"; }
fail() { echo -e "  ${R}FAIL${N} $*"; exit 1; }
info() { echo -e "  ${C}..${N} $*"; }

echo -e "\n${G}=== Savage22 GPU Histogram Setup ===${N}"

# --- 1. Detect GPU ---
command -v nvidia-smi &>/dev/null || fail "nvidia-smi not found — no NVIDIA GPU"
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | xargs)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | xargs)
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | xargs)
echo -e "GPU:    ${C}${GPU_NAME}${N} (${GPU_MEM} MiB)"
echo -e "Driver: ${DRIVER}"

# --- 2. Detect CUDA ---
if command -v nvcc &>/dev/null; then
    CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
else
    CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo "")
fi
[ -z "$CUDA_VER" ] && fail "Cannot detect CUDA version"
CUDA_MAJOR=${CUDA_VER%%.*}
echo -e "CUDA:   ${CUDA_VER}"

# --- 3. Detect Python ---
PY=$(command -v python3 || command -v python) || fail "Python not found"
PY_VER=$($PY --version 2>&1 | grep -oP '[0-9]+\.[0-9]+')
PY_MAJOR=${PY_VER%%.*}; PY_MINOR=${PY_VER##*.}
[ "$PY_MAJOR" -lt 3 ] || [ "$PY_MINOR" -lt 10 ] && fail "Python 3.10+ required (got ${PY_VER})"
echo -e "Python: ${PY_VER} (${PY})\n"

# --- 4. Install minimal deps ---
info "Installing numpy, scipy, cupy-cuda${CUDA_MAJOR}x..."
$PY -m pip install -q --no-deps numpy scipy "cupy-cuda${CUDA_MAJOR}x" 2>&1 | tail -3
ok "Core deps installed"

# --- 5. Install LightGBM CUDA sparse fork ---
WHEEL_TAG="cp3${PY_MINOR}-cp3${PY_MINOR}-linux_x86_64"
WHEEL_NAME="lightgbm_savage-4.6.0+cuda_sparse-${WHEEL_TAG}.whl"
WHEEL_URL="https://github.com/c19o/TB/releases/download/v3.3-wheels/${WHEEL_NAME}"

info "Trying pre-built wheel: ${WHEEL_NAME}"
if $PY -m pip install -q --no-deps "$WHEEL_URL" 2>/dev/null; then
    ok "LightGBM installed from pre-built wheel"
else
    echo -e "  ${Y}WARN${N} Pre-built wheel not available — building from source (~10 min)"
    # Ensure build deps
    apt-get update -qq && apt-get install -y -qq cmake g++ git >/dev/null 2>&1 || true
    $PY -m pip install -q scikit-build-core 2>&1 | tail -1

    BUILD_DIR="/tmp/lgbm-savage-build"
    rm -rf "$BUILD_DIR"; mkdir -p "$BUILD_DIR"; cd "$BUILD_DIR"

    info "Cloning LightGBM v4.6.0..."
    git clone --depth 1 --branch v4.6.0 --recurse-submodules \
        https://github.com/microsoft/LightGBM.git lgbm 2>&1 | tail -2

    # Detect SM arch from GPU
    SM=$($PY -c "
import subprocess, re
out = subprocess.check_output(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader']).decode()
print(re.sub(r'\.', '', out.strip().split('\n')[0]))
" 2>/dev/null || echo "80")

    info "Building with CUDA (sm_${SM})..."
    cd lgbm && mkdir -p build && cd build
    cmake .. -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$SM" \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_FLAGS="-O3 --use_fast_math" \
        >/dev/null 2>&1
    cmake --build . --config Release -j "$(nproc)" 2>&1 | tail -5

    info "Installing Python package..."
    cd ../python-package
    $PY -m pip install -q --no-build-isolation --no-deps . 2>&1 | tail -2
    cd /tmp && rm -rf "$BUILD_DIR"
    ok "LightGBM built and installed from source"
fi

# --- 6. Verify ---
echo ""
info "Verifying imports..."
$PY -c "
import lightgbm as lgb; import cupy as cp; import numpy as np; import scipy.sparse as sp
print(f'  LightGBM {lgb.__version__}')
print(f'  CuPy {cp.__version__} (CUDA {cp.cuda.runtime.runtimeGetVersion()})')

# Quick GPU benchmark: sparse matmul
rows, cols = 5000, 100000
density = 0.01
X = sp.random(rows, cols, density=density, format='csr', dtype=np.float32)
X_gpu = cp.sparse.csr_matrix(X)
grad = cp.random.randn(rows, dtype=cp.float32)

import time
# CPU
t0 = time.perf_counter()
for _ in range(10): _ = X.T @ X.T.toarray()[:, 0]  # dummy
cpu_t = (time.perf_counter() - t0) / 10

# GPU
cp.cuda.Stream.null.synchronize()
t0 = time.perf_counter()
for _ in range(10): _ = X_gpu.T @ grad
cp.cuda.Stream.null.synchronize()
gpu_t = (time.perf_counter() - t0) / 10

speedup = cpu_t / max(gpu_t, 1e-9)
print(f'  GPU SpMV benchmark: {speedup:.0f}x speedup')
" || fail "Import verification failed"
ok "All imports verified"

# --- 7. Print TF capabilities ---
RAM_GB=$($PY -c "import os; print(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') // (1024**3))" 2>/dev/null || echo 0)
GPU_GB=$((GPU_MEM / 1024))
echo -e "\n${G}=== Ready ===${N}"
echo -e "RAM: ${RAM_GB} GB | VRAM: ${GPU_GB} GB\n"
echo "Timeframe support:"
for tf in 1w 1d 4h 1h 15m; do
    case $tf in
        1w)  need=64;;   1d)  need=192;;
        4h)  need=768;;  1h)  need=1024;; 15m) need=1500;;
    esac
    if [ "$RAM_GB" -ge "$need" ]; then
        echo -e "  ${G}OK${N}  $tf  (needs ${need}GB RAM)"
    else
        echo -e "  ${R}--${N}  $tf  (needs ${need}GB, have ${RAM_GB}GB)"
    fi
done

echo -e "\nRun:"
echo "  python -u cloud_run_tf.py --symbol BTC --tf 1w"
echo "  python -u cloud_run_tf.py --symbol BTC --tf 1d"
