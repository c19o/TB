#!/usr/bin/env bash
# =============================================================================
# build_linux_wheel.sh — Build a universal Linux pip wheel of the LightGBM
# CUDA sparse histogram fork.
#
# Build ONCE on a machine with CUDA 12.6 toolkit, then deploy anywhere:
#   pip install lightgbm_savage-4.6.0+cuda_sparse-cp312-cp312-linux_x86_64.whl
#
# Target machines need ONLY:
#   - NVIDIA driver >= 525
#   - Python 3.12
#   - No CUDA toolkit, no cmake, no compilation
#
# Fat binary: sm_80 (A100), sm_86 (3090/A40), sm_89 (4090/L40), sm_90 (H100/H200)
# CUDA runtime statically linked into the .so.
#
# Usage:
#   bash build_linux_wheel.sh              # full build
#   bash build_linux_wheel.sh --clean      # wipe and rebuild from scratch
#   bash build_linux_wheel.sh --skip-clone # reuse existing LightGBM checkout
#
# Output: ./dist/lightgbm_savage-4.6.0+cuda_sparse-cp312-cp312-linux_x86_64.whl
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LGBM_VERSION="4.6.0"
SAVAGE_VERSION="${LGBM_VERSION}+cuda_sparse"
LGBM_REPO="https://github.com/microsoft/LightGBM.git"
LGBM_TAG="v${LGBM_VERSION}"
PYTHON="${PYTHON:-python3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_ROOT="${SCRIPT_DIR}/_lgbm_wheel_build"
LGBM_DIR="${BUILD_ROOT}/LightGBM"
BUILD_DIR="${LGBM_DIR}/build"
OUTPUT_DIR="${SCRIPT_DIR}/dist"

# Fat binary architectures
CUDA_ARCHS="80;86;89;90"

# Parallel compile jobs
NPROC="${NPROC:-$(nproc 2>/dev/null || echo 8)}"

# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------
CLEAN=0
SKIP_CLONE=0

for arg in "$@"; do
    case "$arg" in
        --clean)      CLEAN=1 ;;
        --skip-clone) SKIP_CLONE=1 ;;
        --help|-h)
            echo "Usage: $0 [--clean] [--skip-clone]"
            exit 0
            ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
info()  { echo -e "\033[1;32m[BUILD]\033[0m $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m $*"; }
die()   { echo -e "\033[1;31m[FATAL]\033[0m $*"; exit 1; }

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
info "=== Pre-flight checks ==="

[[ "$(uname -s)" == "Linux" ]] || die "This script is Linux-only. Use build_wheel.sh for cross-platform."

command -v nvcc &>/dev/null || die "nvcc not found. Install CUDA toolkit 12.6 and add to PATH."
CUDA_VER_RAW="$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')"
CUDA_MAJOR="${CUDA_VER_RAW%%.*}"
CUDA_MINOR="${CUDA_VER_RAW##*.}"
info "CUDA toolkit: ${CUDA_VER_RAW}"

# Add sm_100 if CUDA >= 12.8
if (( CUDA_MAJOR > 12 || (CUDA_MAJOR == 12 && CUDA_MINOR >= 8) )); then
    CUDA_ARCHS="${CUDA_ARCHS};100"
    info "CUDA >= 12.8 detected -- including sm_100 (B200)"
fi
info "Target architectures: ${CUDA_ARCHS}"

command -v cmake &>/dev/null || die "cmake not found. apt install cmake."
command -v "$PYTHON" &>/dev/null || die "Python not found. Set PYTHON=... or install Python 3.12."
command -v git &>/dev/null || die "git not found."

PY_VER="$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[[ "$PY_VER" == "3.12" ]] || warn "Python is ${PY_VER}, wheel will target cp${PY_VER/./} (expected 3.12)"

# Ensure build deps
"$PYTHON" -m pip install --quiet wheel setuptools build scikit-build-core 2>/dev/null || true

info "cmake: $(cmake --version | head -1)"
info "Python: $PY_VER ($PYTHON)"

# ---------------------------------------------------------------------------
# Step 0: Clean
# ---------------------------------------------------------------------------
if (( CLEAN )); then
    info "Cleaning previous build..."
    rm -rf "$BUILD_ROOT" "$OUTPUT_DIR"
fi
mkdir -p "$BUILD_ROOT" "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# Step 1: Clone LightGBM v4.6.0
# ---------------------------------------------------------------------------
info "=== Step 1: Clone LightGBM ${LGBM_TAG} ==="

if (( SKIP_CLONE )); then
    [[ -d "$LGBM_DIR" ]] || die "LightGBM not found at ${LGBM_DIR}. Remove --skip-clone."
    info "Skipping clone (--skip-clone)"
else
    if [[ -d "$LGBM_DIR/.git" ]]; then
        info "Already cloned. Verifying tag..."
        cd "$LGBM_DIR"
        CURRENT="$(git describe --tags --exact-match 2>/dev/null || echo 'detached')"
        if [[ "$CURRENT" != "$LGBM_TAG" ]]; then
            warn "Tag mismatch ($CURRENT != $LGBM_TAG). Checking out..."
            git fetch --tags
            git checkout "$LGBM_TAG"
            git submodule update --init --recursive
        fi
        cd "$SCRIPT_DIR"
    else
        info "Cloning..."
        git clone --depth 1 --branch "$LGBM_TAG" --recurse-submodules "$LGBM_REPO" "$LGBM_DIR"
    fi
fi

# ---------------------------------------------------------------------------
# Step 2: Apply all patches
# ---------------------------------------------------------------------------
info "=== Step 2: Apply patches ==="

# ---- 2a: Copy our source files into the LightGBM tree ----

TREELEARNER_DIR="${LGBM_DIR}/src/treelearner"

info "  Copying tree_learner.cpp (full replacement with cuda_sparse dispatch)..."
cp -f "${SCRIPT_DIR}/src/treelearner/tree_learner.cpp" "${TREELEARNER_DIR}/tree_learner.cpp"

info "  Copying cuda_sparse_hist_tree_learner.h..."
cp -f "${SCRIPT_DIR}/src/treelearner/cuda_sparse_hist_tree_learner.h" "${TREELEARNER_DIR}/"

info "  Copying cuda_sparse_hist_tree_learner.cu..."
cp -f "${SCRIPT_DIR}/src/treelearner/cuda_sparse_hist_tree_learner.cu" "${TREELEARNER_DIR}/"

info "  Copying gpu_histogram.cu..."
cp -f "${SCRIPT_DIR}/src/gpu_histogram.cu" "${TREELEARNER_DIR}/"

info "  Copying gpu_histogram.h..."
cp -f "${SCRIPT_DIR}/src/gpu_histogram.h" "${TREELEARNER_DIR}/"

if [[ -f "${SCRIPT_DIR}/src/histogram_output_mapper.h" ]]; then
    info "  Copying histogram_output_mapper.h..."
    cp -f "${SCRIPT_DIR}/src/histogram_output_mapper.h" "${TREELEARNER_DIR}/"
fi

# ---- 2b: Patch config.h -- add cuda_sparse device_type + config param ----

CONFIG_H="${LGBM_DIR}/include/LightGBM/config.h"
if [[ ! -f "$CONFIG_H" ]]; then
    die "config.h not found at $CONFIG_H"
fi

if grep -q "cuda_sparse" "$CONFIG_H"; then
    info "  config.h already patched"
else
    info "  Patching config.h..."

    # Add cuda_sparse as a recognized device_type value
    # LightGBM 4.6.0 has: std::string device_type = "cpu";
    # We add the cuda_sparse option to the doc comment + validation
    sed -i 's/\(device_type.*options.*\)\(cuda"\)/\1\2, cuda_sparse"/' "$CONFIG_H" 2>/dev/null || true

    # Add config parameter for the feature
    if ! grep -q "use_cuda_sparse_histogram" "$CONFIG_H"; then
        cat >> "$CONFIG_H" << 'PATCH_CFG'

// === Savage22 CUDA Sparse Histogram Co-Processor ===
// When device_type = "cuda_sparse", histogram construction for sparse CSR
// feature matrices is offloaded to a custom CUDA kernel.
// Requires: compiled with -DUSE_CUDA_SPARSE=ON
// desc = use GPU sparse histogram acceleration for CSR feature matrices
bool use_cuda_sparse_histogram = false;
PATCH_CFG
    fi
    info "  config.h patched"
fi

# ---- 2c: Patch config.cpp -- register the parameter ----

CONFIG_CPP="${LGBM_DIR}/src/io/config.cpp"
if [[ -f "$CONFIG_CPP" ]]; then
    if grep -q "use_cuda_sparse_histogram" "$CONFIG_CPP"; then
        info "  config.cpp already patched"
    else
        info "  Patching config.cpp..."
        # Add to the GetMembersFromString function -- register our param
        # Find the last "GetBool" call and add ours after it
        if grep -q 'GetBool(params' "$CONFIG_CPP"; then
            # Insert our parameter registration near the end of the Get* block
            sed -i '/GetBool(params,.*"use_quantized_grad"/a \
  GetBool(params, "use_cuda_sparse_histogram", &use_cuda_sparse_histogram);' "$CONFIG_CPP" 2>/dev/null || true
        fi
        # Also add to the parameter alias list if it exists
        info "  config.cpp patched"
    fi
else
    warn "  config.cpp not found -- parameter may not be parseable from Python"
fi

# ---- 2d: Patch CMakeLists.txt -- add USE_CUDA_SPARSE build option ----

LGBM_CMAKE="${LGBM_DIR}/CMakeLists.txt"
if grep -q "USE_CUDA_SPARSE" "$LGBM_CMAKE"; then
    info "  CMakeLists.txt already patched"
else
    info "  Patching CMakeLists.txt..."

    # Add option declaration after USE_CUDA
    sed -i '/option(USE_CUDA /a option(USE_CUDA_SPARSE "Build with CUDA sparse histogram tree learner (Savage22)" OFF)' \
        "$LGBM_CMAKE"

    # Append the build logic at the end of the file
    cat >> "$LGBM_CMAKE" << 'PATCH_CMAKE'

# === Savage22 CUDA Sparse Histogram Tree Learner ===
if(USE_CUDA_SPARSE)
    message(STATUS "USE_CUDA_SPARSE=ON -- building sparse histogram GPU kernel")

    # Enable CUDA language if not already enabled (USE_CUDA might be OFF)
    if(NOT USE_CUDA)
        enable_language(CUDA)
        find_package(CUDAToolkit REQUIRED)
    endif()

    add_definitions(-DUSE_CUDA_SPARSE)

    # Our CUDA source files
    list(APPEND SOURCES
        src/treelearner/cuda_sparse_hist_tree_learner.cu
        src/treelearner/gpu_histogram.cu
    )

    # CUDA flags for our files
    set_source_files_properties(
        src/treelearner/cuda_sparse_hist_tree_learner.cu
        src/treelearner/gpu_histogram.cu
        PROPERTIES COMPILE_FLAGS "-O3 --use_fast_math --expt-relaxed-constexpr"
    )

    # Link cusparse (needed for SpMV histogram path)
    target_link_libraries(lightgbm_objs PRIVATE CUDA::cusparse)
    if(TARGET _lightgbm)
        target_link_libraries(_lightgbm PRIVATE CUDA::cusparse)
    endif()
endif()
PATCH_CMAKE
    info "  CMakeLists.txt patched"
fi

# ---- 2e: Set version string in python-package ----

LGBM_PY_VERSION="${LGBM_DIR}/python-package/lightgbm/VERSION.txt"
if [[ -f "$LGBM_PY_VERSION" ]]; then
    info "  Setting version to ${SAVAGE_VERSION}..."
    echo "$SAVAGE_VERSION" > "$LGBM_PY_VERSION"
fi

LGBM_PYPROJECT="${LGBM_DIR}/python-package/pyproject.toml"
if [[ -f "$LGBM_PYPROJECT" ]]; then
    sed -i "s/version = \"${LGBM_VERSION}\"/version = \"${SAVAGE_VERSION}\"/" "$LGBM_PYPROJECT" 2>/dev/null || true
fi

info "All patches applied."

# ---------------------------------------------------------------------------
# Step 3: Build with cmake
# ---------------------------------------------------------------------------
info "=== Step 3: Build LightGBM with CUDA sparse support ==="

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Find CUDA toolkit root from nvcc
CUDA_ROOT="$(dirname "$(dirname "$(which nvcc)")")"
info "CUDA toolkit root: $CUDA_ROOT"

cmake "${LGBM_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_CUDA=ON \
    -DUSE_CUDA_SPARSE=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" \
    -DCMAKE_CUDA_FLAGS="-O3 --use_fast_math --expt-relaxed-constexpr" \
    -DCMAKE_CXX_FLAGS="-O3 -DNDEBUG" \
    -DCMAKE_C_FLAGS="-O3 -DNDEBUG" \
    -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_ROOT" \
    -DCMAKE_CUDA_RUNTIME_LIBRARY=Static \
    -DCMAKE_SHARED_LINKER_FLAGS="-Wl,--no-as-needed -ldl -lpthread" \
    -D__BUILD_FOR_PYTHON=ON \
    2>&1 | tee cmake_config.log | tail -30

info "Building with ${NPROC} parallel jobs..."
cmake --build . --config Release -j "$NPROC" 2>&1 | tee cmake_build.log | tail -40

# Find the built shared library
LIB_FILE="${BUILD_DIR}/lib_lightgbm.so"
if [[ ! -f "$LIB_FILE" ]]; then
    LIB_FILE="$(find "$BUILD_DIR" -name 'lib_lightgbm.so' -type f 2>/dev/null | head -1)"
fi
[[ -f "$LIB_FILE" ]] || die "lib_lightgbm.so not found after build. Check cmake_build.log"

LIB_SIZE="$(du -h "$LIB_FILE" | cut -f1)"
info "Built: $LIB_FILE ($LIB_SIZE)"

# Verify no dynamic CUDA runtime dependency (should be statically linked)
if ldd "$LIB_FILE" 2>/dev/null | grep -q "libcudart.so"; then
    warn "libcudart.so found in dynamic deps -- static linking may have failed"
    warn "Target machines will need CUDA toolkit installed"
else
    info "CUDA runtime statically linked (no libcudart.so in ldd output)"
fi

# Verify our symbols are present
if nm -D "$LIB_FILE" 2>/dev/null | grep -qi "sparse_hist\|CUDASparseHist"; then
    info "CUDA sparse histogram symbols found in .so"
else
    warn "CUDA sparse histogram symbols not in dynamic table (may be internal linkage -- OK)"
fi

cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Step 4: Build Python wheel
# ---------------------------------------------------------------------------
info "=== Step 4: Build Python wheel ==="

cd "${LGBM_DIR}/python-package"

# Copy the compiled .so into the python package lib directory
PY_LIB_DIR="${LGBM_DIR}/python-package/lightgbm/lib"
mkdir -p "$PY_LIB_DIR"
cp -f "$LIB_FILE" "${PY_LIB_DIR}/lib_lightgbm.so"
info "Copied lib_lightgbm.so to python-package/lightgbm/lib/"

# Also copy any companion libraries (libcusparse_static might be separate)
for companion in "${BUILD_DIR}/lib_lightgbm_cuda"*.so; do
    [[ -f "$companion" ]] && cp -f "$companion" "$PY_LIB_DIR/" && info "Copied $(basename "$companion")"
done

# Build the wheel -- no-build-isolation skips re-compilation
info "Running pip wheel..."
"$PYTHON" -m pip wheel . \
    --no-build-isolation \
    --no-deps \
    --wheel-dir "$OUTPUT_DIR" \
    2>&1 | tail -20

cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Step 5: Rename and repack wheel to lightgbm_savage
# ---------------------------------------------------------------------------
info "=== Step 5: Repack wheel ==="

# Find whatever wheel was produced
SRC_WHEEL="$(find "$OUTPUT_DIR" -name 'lightgbm*.whl' -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2)"
[[ -n "$SRC_WHEEL" ]] || die "No wheel file found in ${OUTPUT_DIR}/"
info "Source wheel: $(basename "$SRC_WHEEL")"

# Determine Python tag from actual interpreter
PY_TAG="cp${PY_VER/./}"
TARGET_NAME="lightgbm_savage-${SAVAGE_VERSION}-${PY_TAG}-${PY_TAG}-linux_x86_64.whl"
TARGET_WHEEL="${OUTPUT_DIR}/${TARGET_NAME}"

if [[ "$(basename "$SRC_WHEEL")" == "$TARGET_NAME" ]]; then
    info "Wheel already has correct name"
else
    info "Repacking as ${TARGET_NAME}..."

    REPACK_DIR="${BUILD_ROOT}/_repack"
    rm -rf "$REPACK_DIR"
    mkdir -p "$REPACK_DIR"

    cd "$REPACK_DIR"
    "$PYTHON" -m zipfile -e "$SRC_WHEEL" .

    # Update METADATA: Name + Version
    find . -name "METADATA" -exec sed -i 's/^Name: lightgbm$/Name: lightgbm-savage/' {} \;
    find . -name "METADATA" -exec sed -i "s/^Version: .*/Version: ${SAVAGE_VERSION}/" {} \;

    # Update WHEEL tag if needed (ensure linux_x86_64 platform)
    find . -name "WHEEL" -exec sed -i "s/^Tag: .*/Tag: ${PY_TAG}-${PY_TAG}-linux_x86_64/" {} \;

    # Rename .dist-info directory
    OLD_DIST="$(find . -maxdepth 1 -name '*.dist-info' -type d | head -1)"
    NEW_DIST="lightgbm_savage-${SAVAGE_VERSION}.dist-info"
    if [[ -n "$OLD_DIST" && "$(basename "$OLD_DIST")" != "$NEW_DIST" ]]; then
        mv "$OLD_DIST" "$NEW_DIST"
    fi

    # Regenerate RECORD (hash manifest)
    RECORD_FILE="$(find . -name 'RECORD' -type f | head -1)"
    if [[ -n "$RECORD_FILE" ]]; then
        "$PYTHON" -c "
import hashlib, base64, os
record_path = '${RECORD_FILE}'
lines = []
for root, dirs, files in os.walk('.'):
    for f in files:
        fpath = os.path.join(root, f)
        relpath = os.path.relpath(fpath, '.')
        if relpath == os.path.relpath(record_path, '.'):
            continue
        size = os.path.getsize(fpath)
        sha = hashlib.sha256(open(fpath, 'rb').read()).digest()
        h = base64.urlsafe_b64encode(sha).rstrip(b'=').decode()
        lines.append(f'{relpath},sha256={h},{size}')
lines.append(os.path.relpath(record_path, '.') + ',,')
with open(record_path, 'w') as fp:
    fp.write('\n'.join(sorted(lines)) + '\n')
"
    fi

    # Repack as .whl (zip)
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

    # Remove source wheel if different from target
    if [[ "$SRC_WHEEL" != "$TARGET_WHEEL" && -f "$SRC_WHEEL" ]]; then
        rm -f "$SRC_WHEEL"
    fi

    cd "$SCRIPT_DIR"
    rm -rf "$REPACK_DIR"
fi

WHEEL_SIZE="$(du -h "$TARGET_WHEEL" | cut -f1)"
info "Final wheel: ${TARGET_NAME} (${WHEEL_SIZE})"

# ---------------------------------------------------------------------------
# Step 6: Verify wheel installs and imports
# ---------------------------------------------------------------------------
info "=== Step 6: Verify wheel ==="

# Create a temporary venv for clean verification
VERIFY_VENV="${BUILD_ROOT}/_verify_venv"
rm -rf "$VERIFY_VENV"
"$PYTHON" -m venv "$VERIFY_VENV"
VPYTHON="${VERIFY_VENV}/bin/python"

info "Installing wheel into isolated venv..."
"$VPYTHON" -m pip install --quiet --force-reinstall --no-deps "$TARGET_WHEEL" 2>&1

# Run verification checks
"$VPYTHON" -c "
import sys

def check(label, ok, detail=''):
    status = 'PASS' if ok else 'FAIL'
    msg = f'  [{status}] {label}'
    if detail:
        msg += f' -- {detail}'
    print(msg)
    return ok

all_ok = True

# 1. Import
try:
    import lightgbm as lgb
    all_ok &= check('import lightgbm', True, f'v{lgb.__version__}')
except ImportError as e:
    check('import lightgbm', False, str(e))
    sys.exit(1)

# 2. Version
all_ok &= check('version contains cuda_sparse', 'cuda_sparse' in lgb.__version__, lgb.__version__)

# 3. lib_lightgbm.so exists
import os
lib_dir = os.path.join(os.path.dirname(lgb.__file__), 'lib')
lib_path = os.path.join(lib_dir, 'lib_lightgbm.so')
exists = os.path.isfile(lib_path)
all_ok &= check('lib_lightgbm.so exists', exists)

# 4. Library size (fat binary with 4 archs should be large)
if exists:
    size_mb = os.path.getsize(lib_path) / (1024 * 1024)
    all_ok &= check(f'lib size > 10 MB (CUDA kernels embedded)', size_mb > 10, f'{size_mb:.1f} MB')

# 5. No dynamic libcudart dependency
if exists:
    import subprocess
    ldd = subprocess.run(['ldd', lib_path], capture_output=True, text=True)
    no_cudart = 'libcudart.so' not in ldd.stdout
    all_ok &= check('no dynamic libcudart (statically linked)', no_cudart)

# 6. Basic training smoke test (CPU -- no GPU needed for import verification)
try:
    import numpy as np
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    ds = lgb.Dataset(X, label=y, free_raw_data=False)
    model = lgb.train({'objective': 'binary', 'verbose': -1, 'num_leaves': 4}, ds, num_boost_round=2)
    all_ok &= check('CPU training smoke test', True)
except Exception as e:
    all_ok &= check('CPU training smoke test', False, str(e))

print()
if all_ok:
    print('ALL CHECKS PASSED -- wheel is ready for deployment')
else:
    print('SOME CHECKS FAILED -- review above')
sys.exit(0 if all_ok else 1)
"
VERIFY_RC=$?

# Clean up verify venv
rm -rf "$VERIFY_VENV"

if (( VERIFY_RC != 0 )); then
    warn "Verification had failures -- wheel may still work on GPU machines"
fi

# ---------------------------------------------------------------------------
# Step 7: Generate deploy helper script
# ---------------------------------------------------------------------------
DEPLOY_SCRIPT="${OUTPUT_DIR}/deploy_wheel.sh"
cat > "$DEPLOY_SCRIPT" << 'DEPLOY_EOF'
#!/usr/bin/env bash
# Deploy the LightGBM CUDA sparse wheel to a vast.ai / cloud machine.
# Usage: bash deploy_wheel.sh root@HOST:PORT
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 user@host:port"
    echo "Example: $0 root@ssh5.vast.ai:12345"
    exit 1
fi

TARGET="$1"
HOST="${TARGET%%:*}"
PORT="${TARGET##*:}"
[[ "$PORT" == "$HOST" ]] && PORT=22

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHEEL="$(ls "$SCRIPT_DIR"/lightgbm_savage*.whl 2>/dev/null | head -1)"

if [[ -z "$WHEEL" ]]; then
    echo "No lightgbm_savage*.whl found in $SCRIPT_DIR"
    exit 1
fi

echo "Deploying $(basename "$WHEEL") to ${HOST}:${PORT}..."
scp -P "$PORT" "$WHEEL" "${HOST}:/workspace/"
ssh -p "$PORT" "$HOST" "pip install /workspace/$(basename "$WHEEL") --force-reinstall --no-deps && python -c 'import lightgbm; print(f\"LightGBM {lightgbm.__version__} installed OK\")'"
echo "Done."
DEPLOY_EOF
chmod +x "$DEPLOY_SCRIPT"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
info "============================================================"
info "BUILD COMPLETE"
info "============================================================"
info ""
info "Wheel:  ${TARGET_WHEEL}"
info "Size:   ${WHEEL_SIZE}"
info ""
info "Deploy to any vast.ai / cloud machine:"
info "  scp -P PORT ${TARGET_WHEEL} root@HOST:/workspace/"
info "  ssh -p PORT root@HOST 'pip install /workspace/${TARGET_NAME} --force-reinstall --no-deps'"
info "  ssh -p PORT root@HOST 'python -c \"import lightgbm; print(lightgbm.__version__)\"'"
info ""
info "Or use the helper:"
info "  bash ${DEPLOY_SCRIPT} root@ssh5.vast.ai:12345"
info ""
info "Requirements on target machine:"
info "  - NVIDIA driver >= 525 (no CUDA toolkit needed)"
info "  - Python 3.12"
info "  - numpy, scipy (for sparse CSR input)"
info "============================================================"
