#!/usr/bin/env bash
# =============================================================================
# deploy_vastai.sh — Universal GPU Histogram Fork Deployment for vast.ai
# =============================================================================
#
# Builds LightGBM with CUDA sparse histogram support from source on ANY
# vast.ai GPU machine. Handles CUDA 11.8 through 13.x, single/multi-GPU,
# any NVIDIA driver 525+, and pytorch/pytorch or nvidia/cuda base images.
#
# Usage:
#   scp -P PORT deploy_vastai.sh root@HOST:/workspace/
#   ssh -p PORT root@HOST 'bash /workspace/deploy_vastai.sh'
#
#   # Then upload code + data:
#   scp -P PORT code.tar.gz dbs.tar.gz root@HOST:/workspace/
#   ssh -p PORT root@HOST 'cd /workspace && tar xzf code.tar.gz && tar xzf dbs.tar.gz'
#
#   # Train:
#   ssh -p PORT root@HOST 'cd /workspace/v3.3 && lgbm-run python -u cloud_run_tf.py --symbol BTC --tf 1w'
#
# Tested on: pytorch/pytorch:2.5.1-cuda12.4, nvidia/cuda:12.4.1-devel, bare metal
# =============================================================================

set -euo pipefail

# ── Configuration ──
LGBM_VERSION="4.6.0"
LGBM_TAG="v${LGBM_VERSION}"
LGBM_REPO="https://github.com/microsoft/LightGBM.git"
BUILD_ROOT="/workspace/gpu_histogram_build"
LGBM_DIR="${BUILD_ROOT}/LightGBM"
FORK_DIR=""  # Set after code upload detection
LOGFILE="/workspace/deploy_vastai.log"

# CUDA compute capabilities — fat binary for all production GPUs
# sm_70=V100 sm_75=T4 sm_80=A100 sm_86=3090/A40 sm_89=4090/L40S sm_90=H100/H200
CUDA_ARCHS="70;75;80;86;89;90"

# ── Logging ──
exec > >(tee -a "$LOGFILE") 2>&1
START_TS=$(date +%s)

info()  { echo -e "\033[1;32m[$(( $(date +%s) - START_TS ))s]\033[0m $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m $*"; }
error() { echo -e "\033[1;31m[FATAL]\033[0m $*"; exit 1; }

banner() {
    echo ""
    echo "============================================================"
    echo "  $*"
    echo "============================================================"
}

# =============================================================================
# STEP 1: Detect GPU Hardware
# =============================================================================
banner "STEP 1: GPU Hardware Detection"

if ! command -v nvidia-smi &>/dev/null; then
    error "nvidia-smi not found. This is not an NVIDIA GPU machine."
fi

# Parse nvidia-smi
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d '[:space:]')
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | xargs)
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d '[:space:]')
VRAM_GB=$(( VRAM_MB / 1024 ))
DRIVER_MAJOR=$(echo "$DRIVER_VERSION" | cut -d. -f1)

info "GPU:     $GPU_NAME"
info "Count:   $GPU_COUNT"
info "VRAM:    ${VRAM_GB}GB per GPU"
info "Driver:  $DRIVER_VERSION"

# Driver version check
if [ "$DRIVER_MAJOR" -lt 525 ]; then
    error "Driver $DRIVER_VERSION too old. Need 525+ for CUDA 12.x support."
fi

# CUDA version from nvidia-smi (driver-supported max)
DRIVER_CUDA=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
DRIVER_CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo "unknown")
info "Driver CUDA: $DRIVER_CUDA_VER"

# Multi-GPU guidance
if [ "$GPU_COUNT" -gt 1 ]; then
    info ""
    info "MULTI-GPU DETECTED ($GPU_COUNT GPUs)"
    info "  Training uses 1 GPU for histograms. Extra GPUs are unused by LightGBM."
    info "  For parallel Optuna: assign 1 GPU per worker via CUDA_VISIBLE_DEVICES."
    info "  Example (2 workers, 2 GPUs):"
    info "    CUDA_VISIBLE_DEVICES=0 python -u cloud_run_tf.py --tf 1w &"
    info "    CUDA_VISIBLE_DEVICES=1 python -u cloud_run_tf.py --tf 1d &"
fi

# VRAM fitness per TF
echo ""
info "VRAM Fitness (per GPU):"
declare -A TF_VRAM=([1w]=3 [1d]=6 [4h]=13 [1h]=26 [15m]=41)
for TF in 1w 1d 4h 1h 15m; do
    NEED=${TF_VRAM[$TF]}
    if [ "$VRAM_GB" -ge "$NEED" ]; then
        info "  $TF: ${NEED}GB needed / ${VRAM_GB}GB available -- OK"
    else
        warn "  $TF: ${NEED}GB needed / ${VRAM_GB}GB available -- WON'T FIT (CPU fallback)"
    fi
done

# =============================================================================
# STEP 2: Detect / Install CUDA Toolkit
# =============================================================================
banner "STEP 2: CUDA Toolkit Detection"

CUDA_VERSION=""
CUDA_MAJOR=""
CUDA_MINOR=""
NVCC_PATH=""

# Try nvcc first
if command -v nvcc &>/dev/null; then
    NVCC_PATH=$(which nvcc)
    CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
    info "nvcc found: $NVCC_PATH (CUDA $CUDA_VERSION)"
else
    info "nvcc not found in PATH. Searching..."

    # Common locations on vast.ai / cloud images
    for CANDIDATE in \
        /usr/local/cuda/bin/nvcc \
        /usr/local/cuda-12.4/bin/nvcc \
        /usr/local/cuda-12.6/bin/nvcc \
        /usr/local/cuda-12.2/bin/nvcc \
        /usr/local/cuda-12.1/bin/nvcc \
        /usr/local/cuda-12.0/bin/nvcc \
        /usr/local/cuda-11.8/bin/nvcc \
        /usr/local/cuda-13.0/bin/nvcc; do
        if [ -x "$CANDIDATE" ]; then
            NVCC_PATH="$CANDIDATE"
            export PATH="$(dirname "$CANDIDATE"):$PATH"
            CUDA_VERSION=$("$CANDIDATE" --version | grep -oP 'release \K[0-9]+\.[0-9]+')
            info "Found nvcc: $CANDIDATE (CUDA $CUDA_VERSION)"
            break
        fi
    done
fi

# If still no nvcc, install cuda-toolkit
if [ -z "$NVCC_PATH" ]; then
    warn "No nvcc found anywhere. Installing CUDA toolkit..."

    # Determine which CUDA to install based on driver
    if [ "$DRIVER_MAJOR" -ge 580 ]; then
        INSTALL_CUDA="13.0"
    elif [ "$DRIVER_MAJOR" -ge 550 ]; then
        INSTALL_CUDA="12.6"
    elif [ "$DRIVER_MAJOR" -ge 535 ]; then
        INSTALL_CUDA="12.2"
    elif [ "$DRIVER_MAJOR" -ge 525 ]; then
        INSTALL_CUDA="12.0"
    else
        INSTALL_CUDA="11.8"
    fi

    info "Installing CUDA toolkit $INSTALL_CUDA for driver $DRIVER_VERSION..."

    # Try conda first (fastest, works in pytorch containers)
    if command -v conda &>/dev/null; then
        CUDA_CONDA_MAJOR=$(echo "$INSTALL_CUDA" | cut -d. -f1)
        info "Using conda to install cuda-toolkit..."
        conda install -y -c nvidia/label/cuda-${INSTALL_CUDA}.0 cuda-toolkit 2>&1 | tail -5 || {
            warn "conda install failed, trying apt..."
        }
    fi

    # Try apt if conda didn't work
    if ! command -v nvcc &>/dev/null; then
        apt-get update -qq 2>/dev/null || true
        # nvidia-cuda-toolkit is the easiest apt package
        apt-get install -y -qq nvidia-cuda-toolkit 2>/dev/null || {
            # Fallback: install from NVIDIA repo
            CUDA_APT_MAJOR=$(echo "$INSTALL_CUDA" | tr -d '.')
            apt-get install -y -qq cuda-toolkit-${CUDA_APT_MAJOR} 2>/dev/null || {
                warn "apt install failed. Trying pip cmake + header-only approach..."
            }
        }
    fi

    # Re-detect
    for CANDIDATE in /usr/local/cuda/bin/nvcc /usr/bin/nvcc $(find /usr/local -name nvcc -type f 2>/dev/null | head -1); do
        if [ -x "$CANDIDATE" ]; then
            NVCC_PATH="$CANDIDATE"
            export PATH="$(dirname "$CANDIDATE"):$PATH"
            CUDA_VERSION=$("$CANDIDATE" --version | grep -oP 'release \K[0-9]+\.[0-9]+')
            break
        fi
    done

    if [ -z "$NVCC_PATH" ]; then
        error "Cannot find or install nvcc. Manual CUDA toolkit install required."
    fi
    info "Installed nvcc: $NVCC_PATH (CUDA $CUDA_VERSION)"
fi

CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)
info "CUDA: $CUDA_VERSION (major=$CUDA_MAJOR, minor=$CUDA_MINOR)"

# Set CUDA_HOME for cmake
CUDA_HOME=$(dirname "$(dirname "$NVCC_PATH")")
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
info "CUDA_HOME: $CUDA_HOME"

# Add sm_100 (B200/B100) if CUDA >= 12.8
if [ "$CUDA_MAJOR" -gt 12 ] || { [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 8 ]; }; then
    CUDA_ARCHS="${CUDA_ARCHS};100"
    info "CUDA >= 12.8 — including sm_100 (Blackwell)"
fi

# Check for libcudart and libcusparse (needed for build)
CUDART_PATH=$(find "$CUDA_HOME" /usr/lib /usr/local -name 'libcudart*' -type f 2>/dev/null | head -1 || true)
CUSPARSE_PATH=$(find "$CUDA_HOME" /usr/lib /usr/local -name 'libcusparse*' -type f 2>/dev/null | head -1 || true)
[ -n "$CUDART_PATH" ] && info "libcudart: $CUDART_PATH" || warn "libcudart not found (static link will be used)"
[ -n "$CUSPARSE_PATH" ] && info "libcusparse: $CUSPARSE_PATH" || warn "libcusparse not found (may need cuda-toolkit-dev)"

# =============================================================================
# STEP 3: Install System Dependencies
# =============================================================================
banner "STEP 3: System Dependencies"

# Track what we install
INSTALLED=""

install_if_missing() {
    local CMD="$1"
    local PKG="$2"
    if command -v "$CMD" &>/dev/null; then
        info "  $CMD: already installed"
    else
        info "  Installing $PKG..."
        apt-get install -y -qq "$PKG" 2>/dev/null || {
            warn "  apt-get install $PKG failed. Trying alternatives..."
            # pip fallback for cmake
            if [ "$CMD" = "cmake" ]; then
                pip install cmake 2>&1 | tail -3
            elif [ "$CMD" = "ninja" ]; then
                pip install ninja 2>&1 | tail -3
            fi
        }
        INSTALLED="$INSTALLED $PKG"
    fi
}

apt-get update -qq 2>/dev/null || true

install_if_missing cmake cmake
install_if_missing ninja ninja-build
install_if_missing git git
install_if_missing make build-essential

# tcmalloc for memory optimization (from setup.sh)
apt-get install -y -qq google-perftools libgoogle-perftools-dev numactl 2>/dev/null || true

# Verify cmake version >= 3.18
CMAKE_VER=$(cmake --version | head -1 | grep -oP '[0-9]+\.[0-9]+' | head -1)
CMAKE_MAJOR=$(echo "$CMAKE_VER" | cut -d. -f1)
CMAKE_MINOR=$(echo "$CMAKE_VER" | cut -d. -f2)
if [ "$CMAKE_MAJOR" -lt 3 ] || { [ "$CMAKE_MAJOR" -eq 3 ] && [ "$CMAKE_MINOR" -lt 18 ]; }; then
    warn "cmake $CMAKE_VER too old (need 3.18+). Installing via pip..."
    pip install --upgrade cmake 2>&1 | tail -3
fi

info "cmake: $(cmake --version | head -1)"
info "ninja: $(ninja --version 2>/dev/null || echo 'not found — will use make')"
info "git:   $(git --version)"

# =============================================================================
# STEP 4: Install Python Dependencies
# =============================================================================
banner "STEP 4: Python Dependencies"

NPROC_VAL=$(nproc)
RAM_GB_SYS=$(free -g | awk '/Mem/{print $2}')
info "System: ${NPROC_VAL} cores, ${RAM_GB_SYS}GB RAM"

# Core ML deps
info "Installing core Python packages..."
pip install -q --no-deps \
    scikit-learn scipy ephem astropy pytz joblib \
    pandas numpy pyarrow optuna hmmlearn numba tqdm pyyaml psutil 2>&1 | tail -5

# Reinstall with deps for anything that needs them
pip install -q \
    scikit-learn scipy ephem astropy pytz joblib \
    pandas numpy pyarrow optuna hmmlearn numba tqdm pyyaml psutil 2>&1 | tail -5

# CuPy — match CUDA version
info "Installing CuPy for CUDA $CUDA_MAJOR..."
CUPY_PKG="cupy-cuda${CUDA_MAJOR}x"
if [ "$CUDA_MAJOR" -eq 11 ]; then
    CUPY_PKG="cupy-cuda11x"
elif [ "$CUDA_MAJOR" -ge 12 ]; then
    CUPY_PKG="cupy-cuda12x"
fi

pip install -q "$CUPY_PKG" 2>&1 | tail -3 || {
    warn "cupy package install failed. Trying cupy from source..."
    pip install cupy 2>&1 | tail -5 || warn "CuPy install failed — GPU histogram Python fallback won't work"
}

# Verify all imports
info "Verifying Python imports..."
python3 -c "
import pandas, numpy, scipy, sklearn, ephem, astropy, pyarrow, optuna, numba, hmmlearn, yaml, tqdm, psutil
try:
    import cupy
    print(f'  CuPy: {cupy.__version__} (CUDA runtime {cupy.cuda.runtime.runtimeGetVersion()})')
except Exception as e:
    print(f'  CuPy: FAILED ({e})')
print('ALL CORE IMPORTS OK')
" || error "Core Python imports failed. Fix before proceeding."

# =============================================================================
# STEP 5: Build LightGBM Fork from Source
# =============================================================================
banner "STEP 5: Build LightGBM with CUDA Sparse Histograms"

# Detect fork source directory (uploaded code)
for CANDIDATE_FORK in \
    /workspace/v3.3/gpu_histogram_fork \
    /workspace/gpu_histogram_fork \
    "$(dirname "$(readlink -f "$0")")" ; do
    if [ -d "$CANDIDATE_FORK/src" ] && [ -f "$CANDIDATE_FORK/build_wheel.sh" ]; then
        FORK_DIR="$CANDIDATE_FORK"
        break
    fi
done

if [ -z "$FORK_DIR" ]; then
    warn "GPU histogram fork source not found. Building stock LightGBM with CUDA..."
    warn "Upload code.tar.gz first for the full fork build."
    warn "Falling back to stock LightGBM (pip install)..."
    pip install lightgbm 2>&1 | tail -5
else
    info "Fork source: $FORK_DIR"

    # 5a: Clone LightGBM
    mkdir -p "$BUILD_ROOT"
    if [ -d "$LGBM_DIR/.git" ]; then
        info "LightGBM already cloned. Verifying tag..."
        cd "$LGBM_DIR"
        CURRENT_TAG=$(git describe --tags --exact-match 2>/dev/null || echo 'none')
        if [ "$CURRENT_TAG" != "$LGBM_TAG" ]; then
            info "Wrong tag ($CURRENT_TAG). Re-cloning..."
            cd /workspace
            rm -rf "$LGBM_DIR"
            git clone --depth 1 --branch "$LGBM_TAG" --recurse-submodules "$LGBM_REPO" "$LGBM_DIR"
        fi
    else
        info "Cloning LightGBM $LGBM_TAG..."
        git clone --depth 1 --branch "$LGBM_TAG" --recurse-submodules "$LGBM_REPO" "$LGBM_DIR"
    fi

    # 5b: Copy fork source files
    info "Copying fork source files..."

    # Our custom tree learner
    if [ -d "$FORK_DIR/src/treelearner" ]; then
        cp -f "$FORK_DIR/src/treelearner/"* "$LGBM_DIR/src/treelearner/" 2>/dev/null || true
        info "  Copied treelearner files"
    fi

    # GPU kernel files
    for F in gpu_histogram.cu gpu_histogram.h histogram_output_mapper.h; do
        if [ -f "$FORK_DIR/src/$F" ]; then
            cp -f "$FORK_DIR/src/$F" "$LGBM_DIR/src/treelearner/"
            info "  Copied $F"
        fi
    done

    # 5c: Patch config.h — add cuda_sparse device_type
    CONFIG_H="$LGBM_DIR/include/LightGBM/config.h"
    if [ -f "$CONFIG_H" ] && ! grep -q "cuda_sparse" "$CONFIG_H"; then
        info "Patching config.h..."
        sed -i.bak 's/\("device_type".*options.*\)\(cuda"\)/\1\2, cuda_sparse/' "$CONFIG_H"
        if grep -q "kCUDA" "$CONFIG_H"; then
            sed -i.bak '/kCUDA/!b;n;/kCUDASparse/!a\  static const int kCUDASparse = 4;' "$CONFIG_H"
        fi
        rm -f "${CONFIG_H}.bak"
        info "  config.h patched (cuda_sparse device type added)"
    else
        info "  config.h already patched or not found"
    fi

    # 5d: Patch tree_learner.cpp — factory dispatch
    TREE_LEARNER_CPP="$LGBM_DIR/src/treelearner/tree_learner.cpp"
    if [ -f "$TREE_LEARNER_CPP" ] && ! grep -q "cuda_sparse" "$TREE_LEARNER_CPP"; then
        info "Patching tree_learner.cpp..."
        if ! grep -q "cuda_sparse_hist_tree_learner.h" "$TREE_LEARNER_CPP"; then
            sed -i.bak '/#include.*serial_tree_learner.h/a #include "cuda_sparse_hist_tree_learner.h"' \
                "$TREE_LEARNER_CPP"
        fi
        sed -i.bak '/device_type.*==.*"cuda"/!b;n;/cuda_sparse/!{
            :loop
            n
            /^[[:space:]]*}/!bloop
            a\  } else if (config->device_type == std::string("cuda_sparse")) {\
    return new CUDASparseHistTreeLearner(config);
        }' "$TREE_LEARNER_CPP"
        rm -f "${TREE_LEARNER_CPP}.bak"
        info "  tree_learner.cpp patched (cuda_sparse factory dispatch)"
    fi

    # 5e: Patch CMakeLists.txt — USE_CUDA_SPARSE
    LGBM_CMAKE="$LGBM_DIR/CMakeLists.txt"
    if [ -f "$LGBM_CMAKE" ] && ! grep -q "USE_CUDA_SPARSE" "$LGBM_CMAKE"; then
        info "Patching CMakeLists.txt..."
        sed -i.bak '/option(USE_CUDA /a option(USE_CUDA_SPARSE "Build with CUDA sparse histogram support" OFF)' \
            "$LGBM_CMAKE"
        cat >> "$LGBM_CMAKE" << 'PATCH_EOF'

# --- CUDA Sparse Histogram Fork (Savage22) ---
if(USE_CUDA_SPARSE)
    message(STATUS "USE_CUDA_SPARSE=ON -- building with sparse histogram GPU kernel")
    add_definitions(-DUSE_CUDA_SPARSE)
    list(APPEND SOURCES
        src/treelearner/cuda_sparse_hist_tree_learner.cu
        src/treelearner/gpu_histogram.cu
    )
endif()
PATCH_EOF
        rm -f "${LGBM_CMAKE}.bak"
        info "  CMakeLists.txt patched (USE_CUDA_SPARSE option)"
    fi

    # 5f: Patch python-package version
    LGBM_PY_VERSION="$LGBM_DIR/python-package/lightgbm/VERSION.txt"
    if [ -f "$LGBM_PY_VERSION" ]; then
        echo "${LGBM_VERSION}+cuda_sparse" > "$LGBM_PY_VERSION"
        info "  Version set to ${LGBM_VERSION}+cuda_sparse"
    fi

    # 5g: Build with CMake
    info "Building LightGBM with CUDA sparse support..."
    BUILD_DIR="$LGBM_DIR/build"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    # Compiler compatibility flags
    EXTRA_CUDA_FLAGS="-O3 --use_fast_math --expt-relaxed-constexpr"

    # --allow-unsupported-compiler for newer GCC + older CUDA combos
    GCC_MAJOR=$(gcc -dumpversion 2>/dev/null | cut -d. -f1 || echo 0)
    if [ "$CUDA_MAJOR" -le 11 ] && [ "$GCC_MAJOR" -ge 12 ]; then
        EXTRA_CUDA_FLAGS="$EXTRA_CUDA_FLAGS --allow-unsupported-compiler"
        info "  Adding --allow-unsupported-compiler (GCC $GCC_MAJOR + CUDA $CUDA_VERSION)"
    elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -le 2 ] && [ "$GCC_MAJOR" -ge 13 ]; then
        EXTRA_CUDA_FLAGS="$EXTRA_CUDA_FLAGS --allow-unsupported-compiler"
        info "  Adding --allow-unsupported-compiler (GCC $GCC_MAJOR + CUDA $CUDA_VERSION)"
    fi

    # Static link CUDA runtime — wheel is self-contained (target only needs driver)
    EXTRA_CUDA_FLAGS="$EXTRA_CUDA_FLAGS -cudart static"

    CMAKE_ARGS=(
        -DCMAKE_BUILD_TYPE=Release
        -DUSE_CUDA=ON
        -DUSE_CUDA_SPARSE=ON
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS"
        -DCMAKE_CUDA_FLAGS="$EXTRA_CUDA_FLAGS"
        -DCMAKE_CXX_FLAGS="-O3"
        -DCMAKE_C_FLAGS="-O3"
        -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME"
        -DCMAKE_SHARED_LINKER_FLAGS="-Wl,--no-as-needed -ldl -lpthread"
    )

    # Use ninja if available (faster than make)
    if command -v ninja &>/dev/null; then
        CMAKE_ARGS+=(-G Ninja)
        BUILD_CMD="ninja -j $NPROC_VAL"
    else
        BUILD_CMD="make -j $NPROC_VAL"
    fi

    info "CMake configure..."
    cmake "$LGBM_DIR" "${CMAKE_ARGS[@]}" 2>&1 | tail -20

    info "Building with $NPROC_VAL parallel jobs..."
    $BUILD_CMD 2>&1 | tail -30

    # Verify built library
    LIB_FILE=$(find "$BUILD_DIR" -name 'lib_lightgbm.so' -type f | head -1)
    if [ -z "$LIB_FILE" ]; then
        error "lib_lightgbm.so not found after build. Check build output in $LOGFILE"
    fi
    info "Built: $LIB_FILE ($(du -h "$LIB_FILE" | cut -f1))"

    # Check for CUDA sparse symbols
    if nm -D "$LIB_FILE" 2>/dev/null | grep -qi "sparse_hist\|CUDASparseHist"; then
        info "  CUDA sparse histogram symbols verified in lib_lightgbm.so"
    else
        warn "  CUDA sparse histogram symbols not found (may be internal linkage — not fatal)"
    fi

    # 5h: Install Python package
    info "Installing LightGBM Python package..."
    cd "$LGBM_DIR/python-package"

    # Copy built lib into Python package
    LGBM_PY_LIB_DIR="$LGBM_DIR/python-package/lightgbm/lib"
    mkdir -p "$LGBM_PY_LIB_DIR"
    cp -f "$LIB_FILE" "$LGBM_PY_LIB_DIR/lib_lightgbm.so"

    # Uninstall stock lightgbm first
    pip uninstall -y lightgbm 2>/dev/null || true

    # Install our build
    pip install --no-build-isolation --no-deps . 2>&1 | tail -10

    cd /workspace
fi

# =============================================================================
# STEP 6: Verify Installation
# =============================================================================
banner "STEP 6: Verification"

python3 << 'VERIFY_EOF'
import sys

def check(label, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    msg = f"  [{status}] {label}"
    if detail:
        msg += f" -- {detail}"
    print(msg)
    return passed

all_ok = True

# 1. Import
try:
    import lightgbm as lgb
    ok = check("Import lightgbm", True, f"version={lgb.__version__}")
except ImportError as e:
    check("Import lightgbm", False, str(e))
    sys.exit(1)

# 2. Version
version = lgb.__version__
ok = check("Version contains cuda_sparse",
           "cuda_sparse" in version,
           f"version={version}")
if not ok:
    print("  (Stock LightGBM installed — GPU histogram fork NOT active)")

# 3. Lib file
import os
lib_dir = os.path.join(os.path.dirname(lgb.__file__), "lib")
lib_path = os.path.join(lib_dir, "lib_lightgbm.so")
ok = check("Compiled library exists",
           os.path.isfile(lib_path),
           lib_path if os.path.isfile(lib_path) else "NOT FOUND")

# 4. Size check
if os.path.isfile(lib_path):
    size_mb = os.path.getsize(lib_path) / (1024 * 1024)
    check("Library > 10MB (includes CUDA)",
          size_mb > 10,
          f"{size_mb:.1f} MB")

# 5. Quick training smoke test
import numpy as np
X = np.random.rand(200, 20).astype(np.float64)
y = np.random.randint(0, 3, 200)
ds = lgb.Dataset(X, label=y, free_raw_data=False)

# Try cuda_sparse first, fallback to cpu
for device in ["cuda_sparse", "cuda", "cpu"]:
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "device_type": device,
        "num_leaves": 4,
        "verbosity": -1,
    }
    try:
        model = lgb.train(params, ds, num_boost_round=5)
        check(f"Train with device_type={device}", True, "5 rounds OK")
        break
    except Exception as e:
        check(f"Train with device_type={device}", False, str(e)[:80])

# 6. CuPy + GPU check
try:
    import cupy as cp
    mem = cp.cuda.Device(0).mem_info
    check("CuPy GPU access", True,
          f"free={mem[0]/1e9:.1f}GB total={mem[1]/1e9:.1f}GB")
except Exception as e:
    check("CuPy GPU access", False, str(e)[:80])

print()
print("Verification complete.")
VERIFY_EOF

# =============================================================================
# STEP 7: Memory / Performance Optimizations (from setup.sh)
# =============================================================================
banner "STEP 7: Memory & Performance Optimizations"

# tcmalloc
TCMALLOC_LIB=""
for candidate in \
    /usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 \
    /usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so \
    /usr/lib/libtcmalloc_minimal.so.4 \
    /usr/lib64/libtcmalloc_minimal.so.4; do
    if [ -f "$candidate" ]; then
        TCMALLOC_LIB="$candidate"
        break
    fi
done
if [ -z "$TCMALLOC_LIB" ]; then
    TCMALLOC_LIB=$(ldconfig -p 2>/dev/null | grep libtcmalloc_minimal | head -1 | awk '{print $NF}' || true)
fi
[ -n "$TCMALLOC_LIB" ] && info "tcmalloc: $TCMALLOC_LIB" || warn "tcmalloc not found"

# THP
if [ -w /sys/kernel/mm/transparent_hugepage/enabled ]; then
    echo madvise > /sys/kernel/mm/transparent_hugepage/enabled
    echo defer+madvise > /sys/kernel/mm/transparent_hugepage/defrag
    info "THP: enabled=madvise, defrag=defer+madvise"
else
    info "THP: container restricted (skipped)"
fi

# Kernel tuning
if command -v sysctl &>/dev/null; then
    sysctl -w vm.swappiness=1 2>/dev/null && info "vm.swappiness=1" || true
    sysctl -w vm.overcommit_memory=1 2>/dev/null && info "vm.overcommit_memory=1" || true
fi

# NUMA detection
NUMA_NODES=1
if command -v numactl &>/dev/null; then
    NUMA_NODES=$(numactl --hardware 2>/dev/null | awk '/available:/{print $2}' || echo 1)
    info "NUMA nodes: $NUMA_NODES"
fi

# Create launch wrapper
cat > /usr/local/bin/lgbm-run << WRAPPER_EOF
#!/bin/bash
# Optimized LightGBM launch wrapper
export LD_PRELOAD="${TCMALLOC_LIB:-}"
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=10737418240
export PYTHONMALLOC=malloc
export PYTHONUNBUFFERED=1
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:\$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:\${LD_LIBRARY_PATH:-}"

echo "[lgbm-run] LD_PRELOAD=\${LD_PRELOAD:-none}"
echo "[lgbm-run] CUDA_HOME=\$CUDA_HOME"
echo "[lgbm-run] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

NUMA_NODES=\$(numactl --hardware 2>/dev/null | awk '/available:/{print \$2}' || echo 1)
if [ "\$NUMA_NODES" -gt 1 ] 2>/dev/null; then
    echo "[lgbm-run] Multi-socket: numactl --interleave=all"
    exec numactl --interleave=all "\$@"
else
    exec "\$@"
fi
WRAPPER_EOF
chmod +x /usr/local/bin/lgbm-run

# Environment file
cat > /workspace/lgbm_env.sh << ENV_EOF
export LD_PRELOAD="${TCMALLOC_LIB:-}"
export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=10737418240
export PYTHONMALLOC=malloc
export PYTHONUNBUFFERED=1
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:\$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:\${LD_LIBRARY_PATH:-}"
ENV_EOF

info "Created /usr/local/bin/lgbm-run and /workspace/lgbm_env.sh"

# =============================================================================
# STEP 8: Summary
# =============================================================================
ELAPSED=$(( $(date +%s) - START_TS ))

banner "DEPLOYMENT COMPLETE (${ELAPSED}s)"

echo ""
echo "  Hardware:"
echo "    GPU:    $GPU_NAME x$GPU_COUNT (${VRAM_GB}GB each)"
echo "    CPU:    ${NPROC_VAL} cores, ${RAM_GB_SYS}GB RAM"
echo "    Driver: $DRIVER_VERSION (CUDA $CUDA_VERSION)"
echo "    NUMA:   $NUMA_NODES node(s)"
echo ""
echo "  Software:"
echo "    LightGBM: $(python3 -c 'import lightgbm; print(lightgbm.__version__)' 2>/dev/null || echo 'NOT INSTALLED')"
echo "    CuPy:     $(python3 -c 'import cupy; print(cupy.__version__)' 2>/dev/null || echo 'NOT INSTALLED')"
echo "    Python:   $(python3 --version 2>&1)"
echo "    tcmalloc: ${TCMALLOC_LIB:-not found}"
echo ""
echo "  Next steps:"
echo "    1. Upload code + data:"
echo "       scp -P PORT code.tar.gz dbs.tar.gz root@HOST:/workspace/"
echo "       ssh -p PORT root@HOST 'cd /workspace && tar xzf code.tar.gz && tar xzf dbs.tar.gz'"
echo "    2. Symlink DBs:"
echo "       ln -sf /workspace/*.db /workspace/v3.3/"
echo "       ln -sf /workspace/v3.3/btc_prices.db /workspace/"
echo "    3. Train:"
echo "       cd /workspace/v3.3 && lgbm-run python -u cloud_run_tf.py --symbol BTC --tf 1w"
echo ""
echo "  GPU check script:"
echo "    python3 /workspace/v3.3/gpu_histogram_fork/check_gpu.py"
echo ""
echo "  Full log: $LOGFILE"
echo ""
