#!/bin/bash
# ============================================================
# Build LightGBM CUDA Sparse Histogram Fork on Linux
# ============================================================
# Requires: CUDA toolkit 12.x, GCC 10+, CMake 3.28+, Ninja
#
# Usage:
#   bash build_linux.sh              # Standard build
#   bash build_linux.sh --clean      # Clean rebuild
#   bash build_linux.sh --install    # Build + install into site-packages
#
# The fork adds USE_CUDA_SPARSE to LightGBM's CMake, which compiles
# cuda_sparse_hist_tree_learner.cu and links cuSPARSE + CUDA runtime.
#
# NOTE: The upstream CMakeLists.txt has Windows-specific flags for
# USE_CUDA_SPARSE (MSVC /utf-8, /FORCE:MULTIPLE). This script patches
# them to Linux equivalents before building.
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/_build/LightGBM"
LINUX_BUILD="$BUILD_DIR/build_linux"

# Parse args
CLEAN=0
INSTALL=0
for arg in "$@"; do
    case "$arg" in
        --clean)   CLEAN=1 ;;
        --install) INSTALL=1 ;;
    esac
done

echo "============================================================"
echo "  LightGBM CUDA Sparse Histogram Fork — Linux Build"
echo "============================================================"
echo "Source:    $BUILD_DIR"
echo "Build:    $LINUX_BUILD"
echo ""

# ============================================================
# 1. Check prerequisites
# ============================================================
echo ">>> [1/5] Checking prerequisites..."

# CUDA
if ! command -v nvcc &>/dev/null; then
    echo "ERROR: nvcc not found. Install CUDA toolkit 12.x."
    echo "  Ubuntu: apt install nvidia-cuda-toolkit"
    echo "  Or add to PATH: export PATH=/usr/local/cuda/bin:\$PATH"
    exit 1
fi
NVCC_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
echo "  nvcc:  $NVCC_VERSION"

# GCC
if ! command -v gcc &>/dev/null; then
    echo "ERROR: gcc not found. Install: apt install build-essential"
    exit 1
fi
GCC_VERSION=$(gcc -dumpversion)
echo "  gcc:   $GCC_VERSION"

# CMake
if ! command -v cmake &>/dev/null; then
    echo "ERROR: cmake not found. Install: pip install cmake"
    exit 1
fi
CMAKE_VERSION=$(cmake --version | head -1 | grep -oP '[0-9]+\.[0-9]+\.[0-9]+')
echo "  cmake: $CMAKE_VERSION"

# Ninja (preferred) or Make
if command -v ninja &>/dev/null; then
    GENERATOR="Ninja"
    BUILD_CMD="ninja -j$(nproc) _lightgbm"
    echo "  ninja: $(ninja --version)"
else
    echo "  ninja: not found, falling back to make"
    GENERATOR="Unix Makefiles"
    BUILD_CMD="make -j$(nproc) _lightgbm"
fi

# Clone LightGBM if not present, then apply fork patches
if [ ! -f "$BUILD_DIR/CMakeLists.txt" ]; then
    echo "  LightGBM source not found — cloning v4.6.0..."
    git clone --depth 1 -b v4.6.0 https://github.com/microsoft/LightGBM.git "$BUILD_DIR"
    cd "$BUILD_DIR" && git submodule update --init --recursive && cd "$SCRIPT_DIR"
    echo "  Cloned + submodules initialized"
fi

# Apply ALL fork patches (12 modified files)
PATCH_TAR="$SCRIPT_DIR/lgbm_fork_patches.tar.gz"
if [ -f "$PATCH_TAR" ]; then
    echo "  Applying fork patches from lgbm_fork_patches.tar.gz..."
    tar xzf "$PATCH_TAR" -C "$BUILD_DIR"
    echo "  Applied $(tar tzf "$PATCH_TAR" | wc -l) patched files"
else
    echo "  WARN: lgbm_fork_patches.tar.gz not found — using source tree as-is"
fi

# ============================================================
# 2. Patch CMakeLists.txt for Linux (idempotent)
# ============================================================
echo ""
echo ">>> [2/5] Patching CMakeLists.txt for Linux..."

CMAKELISTS="$BUILD_DIR/CMakeLists.txt"

# --- Patch 1: CUDA compiler flags (MSVC -> GCC) ---
# Windows:  -Xcompiler=/utf-8
# Linux:    -Xcompiler=-fPIC (position-independent code, required for .so)
if grep -q 'Xcompiler=/utf-8' "$CMAKELISTS"; then
    sed -i 's|-Xcompiler=/utf-8|-Xcompiler=-fPIC|g' "$CMAKELISTS"
    echo "  Patched: -Xcompiler=/utf-8 -> -Xcompiler=-fPIC"
else
    echo "  OK: CUDA flags already patched (no /utf-8 found)"
fi

# --- Patch 2: Debug flags (MSVC -> GCC) ---
# Windows:  -Xcompiler=-MDd -Xcompiler=-Zi
# Linux:    -g (debug symbols)
if grep -q 'Xcompiler=-MDd' "$CMAKELISTS"; then
    sed -i 's|-Xcompiler=-fPIC -Xcompiler=-MDd -Xcompiler=-Zi|-g|' "$CMAKELISTS"
    echo "  Patched: Debug flags MSVC -> GCC"
else
    echo "  OK: Debug flags already patched"
fi

# --- Patch 3: Release flags (MSVC -> GCC) ---
# Windows:  -Xcompiler=-fPIC -Xcompiler=-MD -O3 --use_fast_math
# Linux:    -Xcompiler=-fPIC -O3 --use_fast_math
if grep -q 'Xcompiler=-MD ' "$CMAKELISTS"; then
    sed -i 's|-Xcompiler=-fPIC -Xcompiler=-MD -O3 --use_fast_math|-Xcompiler=-fPIC -O3 --use_fast_math|' "$CMAKELISTS"
    echo "  Patched: Release flags MSVC -> GCC"
else
    echo "  OK: Release flags already patched"
fi

# --- Patch 4: Linker flags (MSVC -> GCC) ---
# Windows:  LINK_FLAGS "/FORCE:MULTIPLE"
# Linux:    LINK_FLAGS "-Wl,--allow-multiple-definition"
# (Needed for OMP_NUM_THREADS symbol conflict between openmp_wrapper and CUDA)
if grep -q '/FORCE:MULTIPLE' "$CMAKELISTS"; then
    sed -i 's|/FORCE:MULTIPLE|-Wl,--allow-multiple-definition|g' "$CMAKELISTS"
    echo "  Patched: LINK_FLAGS /FORCE:MULTIPLE -> -Wl,--allow-multiple-definition"
else
    echo "  OK: Linker flags already patched"
fi

# Verify patches
echo ""
echo "  Verification (USE_CUDA_SPARSE blocks):"
grep -n 'USE_CUDA_SPARSE\|Xcompiler\|FORCE:MULTIPLE\|allow-multiple' "$CMAKELISTS" | head -15
echo ""

# ============================================================
# 3. Copy fork source files into LightGBM tree
# ============================================================
echo ">>> [3/5] Syncing fork source files..."

# The fork .cu and .h live in gpu_histogram_fork/src/
# They need to be in the LightGBM source tree for CMake
FORK_SRC="$SCRIPT_DIR/src/treelearner"
LGBM_SRC="$BUILD_DIR/src/treelearner"

if [ -f "$FORK_SRC/cuda_sparse_hist_tree_learner.cu" ]; then
    cp -v "$FORK_SRC/cuda_sparse_hist_tree_learner.cu" "$LGBM_SRC/"
    echo "  Copied .cu file"
else
    echo "  WARN: Fork .cu not found at $FORK_SRC/"
fi

if [ -f "$FORK_SRC/cuda_sparse_hist_tree_learner.h" ]; then
    cp -v "$FORK_SRC/cuda_sparse_hist_tree_learner.h" "$LGBM_SRC/"
    echo "  Copied .h file"
else
    echo "  WARN: Fork .h not found at $FORK_SRC/"
fi

# ============================================================
# 4. Build
# ============================================================
echo ""
echo ">>> [4/5] Building LightGBM with CUDA Sparse..."

if [ "$CLEAN" -eq 1 ] && [ -d "$LINUX_BUILD" ]; then
    echo "  Cleaning previous build..."
    rm -rf "$LINUX_BUILD"
fi

mkdir -p "$LINUX_BUILD"
cd "$LINUX_BUILD"

# Detect CUDA compiler path
NVCC_PATH=$(which nvcc)
GCC_PATH=$(which gcc)

echo "  NVCC: $NVCC_PATH"
echo "  GCC:  $GCC_PATH"
echo "  Generator: $GENERATOR"
echo "  Cores: $(nproc)"
echo ""

cmake "$BUILD_DIR" \
    -G "$GENERATOR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_CUDA_SPARSE=ON \
    -DUSE_OPENMP=ON \
    -DCMAKE_CUDA_COMPILER="$NVCC_PATH" \
    -DCMAKE_CUDA_HOST_COMPILER="$GCC_PATH" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DFMT_UNICODE=0 \
    2>&1

echo ""
echo "=== COMPILING ($(nproc) cores) ==="
$BUILD_CMD 2>&1

# Find the built library
SO_FILE=""
for candidate in \
    "$LINUX_BUILD/lib_lightgbm.so" \
    "$LINUX_BUILD/lib/lib_lightgbm.so" \
    "$LINUX_BUILD/liblightgbm.so" \
    "$LINUX_BUILD/lib/liblightgbm.so"; do
    if [ -f "$candidate" ]; then
        SO_FILE="$candidate"
        break
    fi
done

if [ -z "$SO_FILE" ]; then
    echo "ERROR: Built .so file not found. Searching..."
    find "$LINUX_BUILD" -name "*lightgbm*.so" -o -name "*lightgbm*.so.*" 2>/dev/null || true
    echo "Build may have failed. Check output above."
    exit 1
fi

echo ""
echo "  Built: $SO_FILE ($(du -h "$SO_FILE" | cut -f1))"

# ============================================================
# 5. Install
# ============================================================
echo ""
echo ">>> [5/5] Installing..."

# Always copy to fork dir for manual use
cp "$SO_FILE" "$SCRIPT_DIR/lib_lightgbm.so"
echo "  Copied to: $SCRIPT_DIR/lib_lightgbm.so"

if [ "$INSTALL" -eq 1 ]; then
    # Find lightgbm site-packages location
    SITE_PACKAGES=$(python3 -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null || \
                    python -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)

    if [ -z "$SITE_PACKAGES" ]; then
        echo "  WARN: Could not determine site-packages. Manual install needed."
        echo "  Copy $SCRIPT_DIR/lib_lightgbm.so to your lightgbm package lib/ directory."
    else
        # Try both possible locations (varies by lightgbm version)
        INSTALLED=0
        for target in \
            "$SITE_PACKAGES/lightgbm/lib/lib_lightgbm.so" \
            "$SITE_PACKAGES/lightgbm/lib_lightgbm.so"; do
            target_dir=$(dirname "$target")
            if [ -d "$target_dir" ]; then
                # Backup original
                if [ -f "$target" ]; then
                    cp "$target" "${target}.bak"
                    echo "  Backed up original: ${target}.bak"
                fi
                cp "$SO_FILE" "$target"
                echo "  Installed: $target"
                INSTALLED=1
            fi
        done

        if [ "$INSTALLED" -eq 0 ]; then
            echo "  WARN: lightgbm lib directory not found in $SITE_PACKAGES"
            echo "  Trying: pip show lightgbm..."
            LGBM_LOC=$(pip show lightgbm 2>/dev/null | grep Location | awk '{print $2}')
            if [ -n "$LGBM_LOC" ]; then
                for target in \
                    "$LGBM_LOC/lightgbm/lib/lib_lightgbm.so" \
                    "$LGBM_LOC/lightgbm/lib_lightgbm.so"; do
                    target_dir=$(dirname "$target")
                    if [ -d "$target_dir" ]; then
                        cp "$SO_FILE" "$target"
                        echo "  Installed: $target"
                        INSTALLED=1
                    fi
                done
            fi
        fi
    fi
fi

# ============================================================
# Verify
# ============================================================
echo ""
echo "============================================================"
echo "  BUILD COMPLETE"
echo "============================================================"
echo ""
echo "  Library: $SCRIPT_DIR/lib_lightgbm.so"
echo "  Size:    $(du -h "$SCRIPT_DIR/lib_lightgbm.so" | cut -f1)"
echo ""

# Quick verification
PYTHON=$(command -v python3 || command -v python)
if [ -n "$PYTHON" ]; then
    echo "  Verifying LightGBM import..."
    $PYTHON -c "
import lightgbm as lgb
print(f'  LightGBM {lgb.__version__} loaded OK')
# Check if CUDA sparse is available
try:
    from ctypes import cdll
    lib = cdll.LoadLibrary('$SCRIPT_DIR/lib_lightgbm.so')
    print('  Fork .so loads OK')
except Exception as e:
    print(f'  Fork .so load test: {e}')
" 2>&1 || echo "  WARN: Verification failed (lightgbm may not be pip-installed)"
fi

echo ""
echo "  To install into lightgbm package:"
echo "    bash build_linux.sh --install"
echo ""
echo "  To use without installing (LD_PRELOAD):"
echo "    export LD_PRELOAD=$SCRIPT_DIR/lib_lightgbm.so"
echo "    python -u train.py"
echo ""
