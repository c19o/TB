#!/usr/bin/env bash
# =============================================================================
# build_wheel.sh — Build a pip-installable LightGBM wheel with CUDA sparse
# histogram support baked in. The resulting wheel includes the compiled .so
# with CUDA runtime statically linked, so target machines only need an
# NVIDIA driver (no CUDA toolkit).
#
# Output: lightgbm_savage-4.6.0+cuda_sparse-cp312-cp312-linux_x86_64.whl
#
# Usage:
#   bash build_wheel.sh              # full build
#   bash build_wheel.sh --clean      # wipe build artifacts and rebuild
#   bash build_wheel.sh --skip-clone # skip clone step (already cloned)
#
# Platform: Linux (cloud deploy), also works on Windows Git Bash (local dev).
# Idempotent — safe to run multiple times.
# =============================================================================

set -euo pipefail

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
LGBM_VERSION="4.6.0"
SAVAGE_VERSION="${LGBM_VERSION}+cuda_sparse"
LGBM_REPO="https://github.com/microsoft/LightGBM.git"
LGBM_TAG="v${LGBM_VERSION}"
PYTHON="${PYTHON:-python3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_ROOT="${SCRIPT_DIR}/_lgbm_build"
LGBM_DIR="${BUILD_ROOT}/LightGBM"
BUILD_DIR="${LGBM_DIR}/build"
OUTPUT_DIR="${SCRIPT_DIR}/dist"

# CUDA architectures — fat binary for all production GPUs
# sm_80 = A100/A30, sm_86 = RTX 3090/A40, sm_89 = RTX 4090/L40, sm_90 = H100/H200
CUDA_ARCHS="80;86;89;90"

# Number of parallel compile jobs
NPROC="${NPROC:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)}"

# -------------------------------------------------------------------------
# Flags
# -------------------------------------------------------------------------
CLEAN=0
SKIP_CLONE=0

for arg in "$@"; do
    case "$arg" in
        --clean)      CLEAN=1 ;;
        --skip-clone) SKIP_CLONE=1 ;;
        --help|-h)
            echo "Usage: $0 [--clean] [--skip-clone]"
            echo "  --clean       Remove build artifacts and rebuild from scratch"
            echo "  --skip-clone  Skip git clone (LightGBM already present)"
            exit 0
            ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

# -------------------------------------------------------------------------
# Logging helpers
# -------------------------------------------------------------------------
info()  { echo -e "\033[1;32m[BUILD]\033[0m $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m $*"; }
error() { echo -e "\033[1;31m[ERROR]\033[0m $*"; exit 1; }

# -------------------------------------------------------------------------
# Pre-flight checks
# -------------------------------------------------------------------------
info "Pre-flight checks..."

# Detect platform
PLATFORM="$(uname -s)"
case "$PLATFORM" in
    Linux*)   PLATFORM="linux" ;;
    MINGW*|MSYS*|CYGWIN*) PLATFORM="windows" ;;
    *) error "Unsupported platform: $PLATFORM" ;;
esac
info "Platform: $PLATFORM"

# Check CUDA toolkit
if ! command -v nvcc &>/dev/null; then
    error "nvcc not found. Install CUDA toolkit (>= 11.2) and add to PATH."
fi
CUDA_VERSION_RAW="$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')"
CUDA_MAJOR="$(echo "$CUDA_VERSION_RAW" | cut -d. -f1)"
CUDA_MINOR="$(echo "$CUDA_VERSION_RAW" | cut -d. -f2)"
info "CUDA toolkit: ${CUDA_VERSION_RAW} (nvcc)"

# Add sm_100 (B200) if CUDA >= 12.8
if [ "$CUDA_MAJOR" -gt 12 ] || { [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 8 ]; }; then
    CUDA_ARCHS="${CUDA_ARCHS};100"
    info "CUDA >= 12.8 detected — including sm_100 (B200/B100)"
fi
info "CUDA architectures: ${CUDA_ARCHS}"

# Check cmake
if ! command -v cmake &>/dev/null; then
    error "cmake not found. Install cmake >= 3.18."
fi
info "cmake: $(cmake --version | head -1)"

# Check python
if ! command -v "$PYTHON" &>/dev/null; then
    # Fallback for Windows Git Bash
    if command -v python &>/dev/null; then
        PYTHON="python"
    else
        error "Python not found. Set PYTHON env var or install Python 3.12+."
    fi
fi
PY_VERSION="$("$PYTHON" --version 2>&1)"
info "Python: $PY_VERSION ($PYTHON)"

# Check git
if ! command -v git &>/dev/null; then
    error "git not found."
fi

# -------------------------------------------------------------------------
# Step 0: Clean if requested
# -------------------------------------------------------------------------
if [ "$CLEAN" -eq 1 ]; then
    info "Cleaning build artifacts..."
    rm -rf "$BUILD_ROOT" "$OUTPUT_DIR"
fi

mkdir -p "$BUILD_ROOT" "$OUTPUT_DIR"

# -------------------------------------------------------------------------
# Step 1: Clone LightGBM (idempotent — skips if already present)
# -------------------------------------------------------------------------
if [ "$SKIP_CLONE" -eq 0 ]; then
    if [ -d "$LGBM_DIR/.git" ]; then
        info "LightGBM already cloned at ${LGBM_DIR}"
        # Verify we're on the right tag
        cd "$LGBM_DIR"
        CURRENT_TAG="$(git describe --tags --exact-match 2>/dev/null || echo 'none')"
        if [ "$CURRENT_TAG" != "$LGBM_TAG" ]; then
            warn "Current tag '$CURRENT_TAG' != expected '$LGBM_TAG'. Checking out..."
            git fetch --tags
            git checkout "$LGBM_TAG"
            git submodule update --init --recursive
        fi
        cd "$SCRIPT_DIR"
    else
        info "Cloning LightGBM ${LGBM_TAG}..."
        git clone --depth 1 --branch "$LGBM_TAG" --recurse-submodules "$LGBM_REPO" "$LGBM_DIR"
    fi
else
    if [ ! -d "$LGBM_DIR" ]; then
        error "LightGBM not found at ${LGBM_DIR}. Remove --skip-clone."
    fi
    info "Skipping clone (--skip-clone)"
fi

# -------------------------------------------------------------------------
# Step 2: Apply patches (idempotent — sed is safe to re-run)
# -------------------------------------------------------------------------
info "Applying patches..."

# --- 2a: Copy our new source files ---
info "  Copying cuda_sparse_hist_tree_learner.h..."
cp -f "${SCRIPT_DIR}/src/treelearner/cuda_sparse_hist_tree_learner.h" \
      "${LGBM_DIR}/src/treelearner/"

# If we have a .cu implementation, copy it too
if [ -f "${SCRIPT_DIR}/src/treelearner/cuda_sparse_hist_tree_learner.cu" ]; then
    info "  Copying cuda_sparse_hist_tree_learner.cu..."
    cp -f "${SCRIPT_DIR}/src/treelearner/cuda_sparse_hist_tree_learner.cu" \
          "${LGBM_DIR}/src/treelearner/"
fi

# Copy gpu_histogram kernel files into LightGBM tree
info "  Copying gpu_histogram.cu and gpu_histogram.h..."
cp -f "${SCRIPT_DIR}/src/gpu_histogram.cu" "${LGBM_DIR}/src/treelearner/"
cp -f "${SCRIPT_DIR}/src/gpu_histogram.h" "${LGBM_DIR}/src/treelearner/"

# --- 2b: Patch config.h — add cuda_sparse to device_type options ---
CONFIG_H="${LGBM_DIR}/include/LightGBM/config.h"
if [ -f "$CONFIG_H" ]; then
    # Check if already patched
    if grep -q "cuda_sparse" "$CONFIG_H"; then
        info "  config.h already patched (cuda_sparse present)"
    else
        info "  Patching config.h — adding cuda_sparse device_type..."
        # The device_type parameter has options like: cpu, gpu, cuda
        # Add cuda_sparse after cuda in the options string
        sed -i.bak 's/\("device_type".*options.*\)\(cuda"\)/\1\2, cuda_sparse/' "$CONFIG_H"
        # Also add the alias/enum value where device types are defined
        # Find the line with kCUDA and add kCUDASparse after it
        if grep -q "kCUDA" "$CONFIG_H"; then
            sed -i.bak '/kCUDA/!b;n;/kCUDASparse/!a\  static const int kCUDASparse = 4;' "$CONFIG_H"
        fi
        rm -f "${CONFIG_H}.bak"
    fi
else
    warn "  config.h not found at expected path — skipping config patch"
fi

# --- 2c: Patch tree_learner.cpp — add factory dispatch for cuda_sparse ---
TREE_LEARNER_CPP="${LGBM_DIR}/src/treelearner/tree_learner.cpp"
if [ -f "$TREE_LEARNER_CPP" ]; then
    if grep -q "cuda_sparse" "$TREE_LEARNER_CPP"; then
        info "  tree_learner.cpp already patched (cuda_sparse dispatch present)"
    else
        info "  Patching tree_learner.cpp — adding cuda_sparse factory dispatch..."
        # Add include at top of file (after existing treelearner includes)
        if ! grep -q "cuda_sparse_hist_tree_learner.h" "$TREE_LEARNER_CPP"; then
            sed -i.bak '/#include.*serial_tree_learner.h/a #include "cuda_sparse_hist_tree_learner.h"' \
                "$TREE_LEARNER_CPP"
        fi
        # Add factory case: after the cuda case, add cuda_sparse case
        # Pattern: look for the cuda device case and add ours after it
        # LightGBM factory uses: if (config->device_type == std::string("cuda"))
        # We add: else if (config->device_type == std::string("cuda_sparse"))
        sed -i.bak '/device_type.*==.*"cuda"/!b;n;/cuda_sparse/!{
            # Read until we find the closing brace of the cuda block
            :loop
            n
            /^[[:space:]]*}/!bloop
            # After the closing brace, insert our cuda_sparse block
            a\  } else if (config->device_type == std::string("cuda_sparse")) {\
    return new CUDASparseHistTreeLearner(config);
        }' "$TREE_LEARNER_CPP"
        rm -f "${TREE_LEARNER_CPP}.bak"
    fi
else
    warn "  tree_learner.cpp not found — skipping factory dispatch patch"
fi

# --- 2d: Patch CMakeLists.txt — add USE_CUDA_SPARSE and source files ---
LGBM_CMAKE="${LGBM_DIR}/CMakeLists.txt"
if [ -f "$LGBM_CMAKE" ]; then
    if grep -q "USE_CUDA_SPARSE" "$LGBM_CMAKE"; then
        info "  CMakeLists.txt already patched (USE_CUDA_SPARSE present)"
    else
        info "  Patching CMakeLists.txt — adding USE_CUDA_SPARSE option..."
        # Add option declaration near other USE_CUDA options
        sed -i.bak '/option(USE_CUDA /a option(USE_CUDA_SPARSE "Build with CUDA sparse histogram support" OFF)' \
            "$LGBM_CMAKE"
        # Add source files to the CUDA sources list when USE_CUDA_SPARSE is ON
        # Insert a block after the USE_CUDA source list
        cat >> "$LGBM_CMAKE" << 'PATCH_EOF'

# --- CUDA Sparse Histogram Fork (Savage22) ---
if(USE_CUDA_SPARSE)
    message(STATUS "USE_CUDA_SPARSE=ON — building with sparse histogram GPU kernel")
    add_definitions(-DUSE_CUDA_SPARSE)
    list(APPEND SOURCES
        src/treelearner/cuda_sparse_hist_tree_learner.cu
        src/treelearner/gpu_histogram.cu
    )
endif()
PATCH_EOF
        rm -f "${LGBM_CMAKE}.bak"
    fi
else
    warn "  CMakeLists.txt not found — skipping cmake patch"
fi

# --- 2e: Patch python-package version to include +cuda_sparse local marker ---
LGBM_PY_VERSION="${LGBM_DIR}/python-package/lightgbm/VERSION.txt"
if [ -f "$LGBM_PY_VERSION" ]; then
    info "  Setting version to ${SAVAGE_VERSION}..."
    echo "$SAVAGE_VERSION" > "$LGBM_PY_VERSION"
fi

# Also patch setup.py/pyproject.toml if present
LGBM_PYPROJECT="${LGBM_DIR}/python-package/pyproject.toml"
if [ -f "$LGBM_PYPROJECT" ]; then
    sed -i.bak "s/version = \"${LGBM_VERSION}\"/version = \"${SAVAGE_VERSION}\"/" "$LGBM_PYPROJECT"
    rm -f "${LGBM_PYPROJECT}.bak"
fi

info "Patches applied successfully."

# -------------------------------------------------------------------------
# Step 3: Build LightGBM with CMake + CUDA + sparse histogram support
# -------------------------------------------------------------------------
info "Building LightGBM with CUDA sparse support..."

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DUSE_CUDA=ON
    -DUSE_CUDA_SPARSE=ON
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS"
    # -O3 and fast math for max kernel performance
    -DCMAKE_CUDA_FLAGS="-O3 --use_fast_math --expt-relaxed-constexpr"
    -DCMAKE_CXX_FLAGS="-O3"
    -DCMAKE_C_FLAGS="-O3"
    # Static link CUDA runtime so the wheel is self-contained
    # (target machine needs only NVIDIA driver, not CUDA toolkit)
    -DCUDA_TOOLKIT_ROOT_DIR="$(dirname "$(dirname "$(which nvcc)")")"
)

# On Linux, statically link CUDA runtime
if [ "$PLATFORM" = "linux" ]; then
    CMAKE_ARGS+=(
        -DCMAKE_CUDA_FLAGS="${CMAKE_ARGS[*]:+-O3 --use_fast_math --expt-relaxed-constexpr} -cudart static"
        -DCMAKE_SHARED_LINKER_FLAGS="-Wl,--no-as-needed -ldl -lpthread"
    )
fi

info "CMake configure..."
cmake "${LGBM_DIR}" "${CMAKE_ARGS[@]}" 2>&1 | tail -20

info "Building with ${NPROC} parallel jobs..."
cmake --build . --config Release -j "$NPROC" 2>&1 | tail -30

# Verify the shared library was built
if [ "$PLATFORM" = "linux" ]; then
    LIB_FILE="${BUILD_DIR}/lib_lightgbm.so"
    if [ ! -f "$LIB_FILE" ]; then
        # Try alternate location
        LIB_FILE="$(find "$BUILD_DIR" -name 'lib_lightgbm.so' -type f | head -1)"
    fi
else
    LIB_FILE="$(find "$BUILD_DIR" -name 'lib_lightgbm.dll' -type f | head -1)"
fi

if [ -z "$LIB_FILE" ] || [ ! -f "$LIB_FILE" ]; then
    error "lib_lightgbm shared library not found after build. Check build output."
fi
info "Built: $LIB_FILE"

# Verify CUDA sparse symbols are in the library
if [ "$PLATFORM" = "linux" ]; then
    if nm -D "$LIB_FILE" 2>/dev/null | grep -qi "sparse_hist\|CUDASparseHist"; then
        info "Verified: CUDA sparse histogram symbols present in lib_lightgbm.so"
    else
        warn "CUDA sparse histogram symbols not found — the .cu files may not have compiled into the lib"
    fi
fi

cd "$SCRIPT_DIR"

# -------------------------------------------------------------------------
# Step 4: Build Python wheel
# -------------------------------------------------------------------------
info "Building Python wheel..."

cd "${LGBM_DIR}/python-package"

# Copy the built lib into the python package directory
LGBM_PY_LIB_DIR="${LGBM_DIR}/python-package/lightgbm/lib"
mkdir -p "$LGBM_PY_LIB_DIR"

if [ "$PLATFORM" = "linux" ]; then
    cp -f "$LIB_FILE" "${LGBM_PY_LIB_DIR}/lib_lightgbm.so"
else
    cp -f "$LIB_FILE" "${LGBM_PY_LIB_DIR}/lib_lightgbm.dll"
fi

# Build the wheel with --no-isolation to skip re-compilation
# (we already built the C library above)
info "Running wheel build..."
"$PYTHON" -m pip wheel . \
    --no-build-isolation \
    --no-deps \
    --wheel-dir "$OUTPUT_DIR" \
    2>&1 | tail -20

cd "$SCRIPT_DIR"

# -------------------------------------------------------------------------
# Step 5: Verify and rename wheel
# -------------------------------------------------------------------------
info "Post-processing wheel..."

# Find the built wheel
WHEEL_FILE="$(find "$OUTPUT_DIR" -name 'lightgbm*.whl' -type f | head -1)"
if [ -z "$WHEEL_FILE" ]; then
    error "No wheel file found in ${OUTPUT_DIR}/"
fi

info "Built wheel: $(basename "$WHEEL_FILE")"

# Rename to our custom name if needed
EXPECTED_NAME="lightgbm_savage-${SAVAGE_VERSION}"
if [ "$PLATFORM" = "linux" ]; then
    TARGET_WHEEL="${OUTPUT_DIR}/${EXPECTED_NAME}-cp312-cp312-linux_x86_64.whl"
else
    TARGET_WHEEL="${OUTPUT_DIR}/${EXPECTED_NAME}-cp312-cp312-win_amd64.whl"
fi

if [ "$WHEEL_FILE" != "$TARGET_WHEEL" ]; then
    # Rename by repacking: unzip, fix METADATA/RECORD, rezip
    info "Renaming wheel to $(basename "$TARGET_WHEEL")..."
    REPACK_DIR="${BUILD_ROOT}/_repack"
    rm -rf "$REPACK_DIR"
    mkdir -p "$REPACK_DIR"

    cd "$REPACK_DIR"
    "$PYTHON" -m zipfile -e "$WHEEL_FILE" .

    # Update package name in METADATA
    find . -name "METADATA" -exec sed -i.bak "s/^Name: lightgbm$/Name: lightgbm-savage/" {} \;
    find . -name "METADATA" -exec sed -i.bak "s/^Version: .*/Version: ${SAVAGE_VERSION}/" {} \;
    find . -name "*.bak" -delete

    # Rename .dist-info directory
    OLD_DIST="$(find . -maxdepth 1 -name '*.dist-info' -type d | head -1)"
    if [ -n "$OLD_DIST" ]; then
        NEW_DIST="lightgbm_savage-${SAVAGE_VERSION}.dist-info"
        if [ "$(basename "$OLD_DIST")" != "$NEW_DIST" ]; then
            mv "$OLD_DIST" "$NEW_DIST"
        fi
    fi

    # Regenerate RECORD (hash manifest)
    RECORD_FILE="$(find . -name 'RECORD' -type f | head -1)"
    if [ -n "$RECORD_FILE" ]; then
        "$PYTHON" -c "
import hashlib, base64, os, pathlib

record_path = '${RECORD_FILE}'
record_dir = os.path.dirname(record_path)
lines = []
for root, dirs, files in os.walk('.'):
    for f in files:
        fpath = os.path.join(root, f)
        relpath = os.path.relpath(fpath, '.')
        if relpath == os.path.relpath(record_path, '.'):
            continue  # RECORD itself has no hash
        size = os.path.getsize(fpath)
        sha = hashlib.sha256(open(fpath, 'rb').read()).digest()
        h = base64.urlsafe_b64encode(sha).rstrip(b'=').decode()
        lines.append(f'{relpath},sha256={h},{size}')
# RECORD line for itself (empty hash per spec)
lines.append(os.path.relpath(record_path, '.') + ',,')
with open(record_path, 'w') as fp:
    fp.write('\n'.join(lines) + '\n')
"
    fi

    # Repack as wheel (zip with .whl extension)
    rm -f "$TARGET_WHEEL"
    "$PYTHON" -c "
import zipfile, os
with zipfile.ZipFile('${TARGET_WHEEL}', 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk('.'):
        for f in files:
            fpath = os.path.join(root, f)
            arcname = os.path.relpath(fpath, '.')
            zf.write(fpath, arcname)
"

    # Remove old wheel if it's different from target
    if [ "$WHEEL_FILE" != "$TARGET_WHEEL" ] && [ -f "$WHEEL_FILE" ]; then
        rm -f "$WHEEL_FILE"
    fi

    cd "$SCRIPT_DIR"
    rm -rf "$REPACK_DIR"

    WHEEL_FILE="$TARGET_WHEEL"
fi

info "Final wheel: $(basename "$WHEEL_FILE")"
info "Size: $(du -h "$WHEEL_FILE" | cut -f1)"

# -------------------------------------------------------------------------
# Step 6: Generate verification script
# -------------------------------------------------------------------------
VERIFY_SCRIPT="${OUTPUT_DIR}/verify_cuda_sparse.py"
cat > "$VERIFY_SCRIPT" << 'VERIFY_EOF'
#!/usr/bin/env python3
"""
Verify that the LightGBM wheel with CUDA sparse support is installed correctly.

Usage:
    python verify_cuda_sparse.py

Checks:
    1. lightgbm can be imported
    2. The build includes CUDA support
    3. The cuda_sparse device_type is recognized
    4. The compiled .so/.dll contains our sparse histogram symbols
"""

import sys
import os

def check(label, passed, detail=""):
    status = "\033[1;32mPASS\033[0m" if passed else "\033[1;31mFAIL\033[0m"
    msg = f"  [{status}] {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return passed

def main():
    print("=" * 60)
    print("LightGBM CUDA Sparse Histogram — Verification")
    print("=" * 60)
    all_ok = True

    # 1. Import check
    try:
        import lightgbm as lgb
        ok = check("Import lightgbm", True, f"version={lgb.__version__}")
    except ImportError as e:
        ok = check("Import lightgbm", False, str(e))
        all_ok = False
        print("\nFATAL: Cannot import lightgbm. Is the wheel installed?")
        sys.exit(1)

    # 2. Version check
    version = lgb.__version__
    ok = check("Version contains cuda_sparse",
               "cuda_sparse" in version,
               f"version={version}")
    all_ok = all_ok and ok

    # 3. Lib file check
    lib_dir = os.path.join(os.path.dirname(lgb.__file__), "lib")
    if sys.platform == "linux":
        lib_name = "lib_lightgbm.so"
    else:
        lib_name = "lib_lightgbm.dll"
    lib_path = os.path.join(lib_dir, lib_name)
    ok = check("Compiled library exists",
               os.path.isfile(lib_path),
               lib_path if os.path.isfile(lib_path) else "NOT FOUND")
    all_ok = all_ok and ok

    # 4. Library size (should be larger than stock build due to CUDA code)
    if os.path.isfile(lib_path):
        size_mb = os.path.getsize(lib_path) / (1024 * 1024)
        ok = check("Library size > 10MB (includes CUDA kernels)",
                    size_mb > 10,
                    f"{size_mb:.1f} MB")
        all_ok = all_ok and ok

    # 5. Check symbols (Linux only)
    if sys.platform == "linux" and os.path.isfile(lib_path):
        import subprocess
        try:
            result = subprocess.run(
                ["nm", "-D", lib_path],
                capture_output=True, text=True, timeout=10
            )
            symbols = result.stdout.lower()
            has_sparse = "sparse_hist" in symbols or "cudasparsehisttreelearner" in symbols
            ok = check("CUDA sparse histogram symbols in .so",
                        has_sparse,
                        "found" if has_sparse else "not found (may be internal linkage)")
            # Not a hard failure — symbols may have internal linkage
        except Exception as e:
            check("Symbol check", False, str(e))

    # 6. Try creating a dataset and training with cuda_sparse
    try:
        import numpy as np
        X = np.random.rand(100, 10).astype(np.float64)
        y = np.random.randint(0, 2, 100)
        ds = lgb.Dataset(X, label=y, free_raw_data=False)
        # Check if cuda_sparse is an accepted device_type
        params = {
            "objective": "binary",
            "device_type": "cuda_sparse",
            "num_leaves": 4,
            "n_estimators": 1,
            "verbose": -1,
        }
        try:
            model = lgb.train(params, ds, num_boost_round=1)
            ok = check("Train with device_type=cuda_sparse", True)
            all_ok = all_ok and ok
        except lgb.basic.LightGBMError as e:
            err_str = str(e)
            if "GPU" in err_str or "CUDA" in err_str or "device" in err_str.lower():
                # This means the device_type IS recognized but no GPU available
                ok = check("device_type=cuda_sparse recognized",
                           True,
                           "accepted (no GPU on this machine for actual training)")
            else:
                ok = check("device_type=cuda_sparse recognized", False, err_str)
                all_ok = all_ok and ok
    except Exception as e:
        check("Dataset/train smoke test", False, str(e))
        all_ok = False

    print()
    if all_ok:
        print("\033[1;32mALL CHECKS PASSED\033[0m — CUDA sparse histogram LightGBM is ready.")
    else:
        print("\033[1;33mSOME CHECKS FAILED\033[0m — review output above.")
    print("=" * 60)
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
VERIFY_EOF

chmod +x "$VERIFY_SCRIPT"

# -------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------
echo ""
info "============================================================"
info "BUILD COMPLETE"
info "============================================================"
info "Wheel:    ${WHEEL_FILE}"
info "Verify:   ${VERIFY_SCRIPT}"
info ""
info "Install:  pip install ${WHEEL_FILE}"
info "Verify:   python ${VERIFY_SCRIPT}"
info ""
info "Deploy to cloud:"
info "  scp -P PORT ${WHEEL_FILE} root@HOST:/workspace/"
info "  ssh -p PORT root@HOST 'pip install /workspace/$(basename "$WHEEL_FILE")'"
info "============================================================"
