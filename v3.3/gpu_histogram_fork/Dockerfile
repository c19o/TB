# =============================================================================
# Dockerfile — Savage22 GPU Histogram Fork
# =============================================================================
# Builds LightGBM from source with USE_CUDA_SPARSE=ON, installs all Python
# dependencies for the v3.3 training pipeline.
#
# Base: nvidia/cuda:12.6.0-devel-ubuntu22.04 (includes nvcc for compilation)
#
# Multi-arch fat binary: sm_80 (A100), sm_86 (3090/A40), sm_89 (4090/L40),
#   sm_90 (H100/H200) + PTX fallback for future GPUs.
#
# Build:
#   docker build -t savage22-gpu .
#
# Run:
#   docker run --gpus all -v /workspace:/workspace savage22-gpu \
#       python -u /workspace/v3.3/cloud_run_tf.py --symbol BTC --tf 1w
#
# Or use as base image on vast.ai / RunPod.
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Build LightGBM with CUDA sparse histogram support
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.6.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# System build deps — minimal set for LightGBM + Python wheel
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        cmake ninja-build git \
        software-properties-common \
        build-essential \
        ca-certificates \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update -qq \
    && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv python3.12-distutils \
    && rm -rf /var/lib/apt/lists/*

# Set python3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Bootstrap pip for 3.12
RUN python3.12 -m ensurepip --upgrade && \
    python3.12 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Clone LightGBM at pinned tag
ARG LGBM_VERSION=4.6.0
ARG LGBM_TAG=v${LGBM_VERSION}

RUN git clone --depth 1 --branch ${LGBM_TAG} --recurse-submodules \
    https://github.com/microsoft/LightGBM.git /opt/LightGBM

# Copy our fork source files into the LightGBM tree
COPY src/treelearner/cuda_sparse_hist_tree_learner.h \
     /opt/LightGBM/src/treelearner/
COPY src/treelearner/cuda_sparse_hist_tree_learner.cu \
     /opt/LightGBM/src/treelearner/
COPY src/gpu_histogram.cu /opt/LightGBM/src/treelearner/
COPY src/gpu_histogram.h  /opt/LightGBM/src/treelearner/

# Apply patches to LightGBM source
# 2a: config.h — add cuda_sparse device_type
RUN CONFIG_H="/opt/LightGBM/include/LightGBM/config.h" && \
    if ! grep -q "cuda_sparse" "$CONFIG_H"; then \
        sed -i 's/\("device_type".*options.*\)\(cuda"\)/\1\2, cuda_sparse/' "$CONFIG_H" && \
        sed -i '/kCUDA/!b;n;/kCUDASparse/!a\  static const int kCUDASparse = 4;' "$CONFIG_H"; \
    fi

# 2b: tree_learner.cpp — add factory dispatch for cuda_sparse
RUN TL="/opt/LightGBM/src/treelearner/tree_learner.cpp" && \
    if ! grep -q "cuda_sparse" "$TL"; then \
        sed -i '/#include.*serial_tree_learner.h/a #include "cuda_sparse_hist_tree_learner.h"' "$TL" && \
        sed -i '/device_type.*==.*"cuda"/!b;n;/cuda_sparse/!{:loop;n;/^[[:space:]]*}/!bloop;a\  } else if (config->device_type == std::string("cuda_sparse")) {\n    return new CUDASparseHistTreeLearner(config);\n' "$TL"; \
    fi

# 2c: CMakeLists.txt — add USE_CUDA_SPARSE option and source files
RUN CMAKE_F="/opt/LightGBM/CMakeLists.txt" && \
    if ! grep -q "USE_CUDA_SPARSE" "$CMAKE_F"; then \
        sed -i '/option(USE_CUDA /a option(USE_CUDA_SPARSE "Build with CUDA sparse histogram support" OFF)' "$CMAKE_F" && \
        printf '\n# --- CUDA Sparse Histogram Fork (Savage22) ---\nif(USE_CUDA_SPARSE)\n    message(STATUS "USE_CUDA_SPARSE=ON")\n    add_definitions(-DUSE_CUDA_SPARSE)\n    list(APPEND SOURCES\n        src/treelearner/cuda_sparse_hist_tree_learner.cu\n        src/treelearner/gpu_histogram.cu\n    )\nendif()\n' >> "$CMAKE_F"; \
    fi

# 2d: Set version tag in python package
RUN echo "${LGBM_VERSION}+cuda_sparse" > /opt/LightGBM/python-package/lightgbm/VERSION.txt

# Build with CMake
# Fat binary: sm_80, sm_86, sm_89, sm_90 + PTX for forward compat
RUN mkdir -p /opt/LightGBM/build && cd /opt/LightGBM/build && \
    cmake /opt/LightGBM \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DUSE_CUDA=ON \
        -DUSE_CUDA_SPARSE=ON \
        -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90" \
        -DCMAKE_CUDA_FLAGS="-O3 --use_fast_math --expt-relaxed-constexpr -cudart static" \
        -DCMAKE_CXX_FLAGS="-O3" \
        -DCMAKE_C_FLAGS="-O3" \
        -DCMAKE_SHARED_LINKER_FLAGS="-Wl,--no-as-needed -ldl -lpthread" \
    && ninja -j$(nproc)

# Build Python wheel
RUN mkdir -p /opt/LightGBM/python-package/lightgbm/lib && \
    cp /opt/LightGBM/build/lib_lightgbm.so \
       /opt/LightGBM/python-package/lightgbm/lib/ && \
    cd /opt/LightGBM/python-package && \
    python3.12 -m pip wheel . --no-build-isolation --no-deps --wheel-dir /opt/wheels

# ---------------------------------------------------------------------------
# Stage 2: Runtime image — no build tools, just CUDA runtime + Python
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# System runtime deps
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        google-perftools \
        numactl \
        ca-certificates \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update -qq \
    && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-distutils \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set python3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Bootstrap pip
RUN python3.12 -m ensurepip --upgrade && \
    python3.12 -m pip install --no-cache-dir --upgrade pip

# Install our custom LightGBM wheel from the builder stage
COPY --from=builder /opt/wheels/*.whl /tmp/wheels/
RUN pip install --no-cache-dir /tmp/wheels/*.whl && rm -rf /tmp/wheels

# Install Python dependencies (matches setup.sh + pipeline requirements)
RUN pip install --no-cache-dir \
        numpy scipy pandas scikit-learn pyarrow \
        cupy-cuda12x \
        ephem astropy pytz \
        optuna hmmlearn numba tqdm \
        joblib pyyaml psutil

# Copy our gpu_histogram_fork package
COPY . /opt/savage22-gpu-histogram/
RUN pip install --no-cache-dir -e /opt/savage22-gpu-histogram/

# tcmalloc preload for LightGBM memory performance (setup.sh equivalent)
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 \
    TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=10737418240 \
    PYTHONMALLOC=malloc

WORKDIR /workspace

# Healthcheck: verify CUDA runtime + LightGBM import + our fork symbols
HEALTHCHECK --interval=60s --timeout=30s --retries=3 \
    CMD python -c "\
import lightgbm as lgb; \
assert 'cuda_sparse' in lgb.__version__, f'Wrong build: {lgb.__version__}'; \
import numpy as np; \
import scipy.sparse; \
import cupy; \
print(f'LightGBM {lgb.__version__} | CuPy CUDA {cupy.cuda.runtime.runtimeGetVersion()}'); \
" || exit 1

# Default: drop into bash (override with your training command)
CMD ["bash"]
