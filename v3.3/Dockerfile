# V3.3 Training Image — RAPIDS + ML deps, CUDA-safe
# Eliminates all pip install at runtime = zero CUDA breakage
#
# Build: docker build -t ghcr.io/c19o/savage22-train:v3.3 -f v3.3/Dockerfile .
# Push:  docker push ghcr.io/c19o/savage22-train:v3.3
# Use:   vastai create instance <ID> --image ghcr.io/c19o/savage22-train:v3.3 --disk 50 --ssh

FROM rapidsai/base:25.02-cuda12.8-py3.12

# Install ML deps WITHOUT touching CUDA packages (--no-deps prevents cuda-bindings upgrade)
# scipy, numpy, pandas, pyarrow already in RAPIDS container
# torch installed --no-deps (its deps numpy/typing-extensions already present)
RUN pip install --no-cache-dir --no-deps lightgbm optuna ephem hmmlearn torch && \
    pip install --no-cache-dir alembic cmaes colorlog sqlalchemy tqdm PyYAML joblib threadpoolctl

# Disable numba_cuda external package (causes CUDA init errors in RAPIDS 25.02)
# The package tries CUDA context init on import → fails on some vast.ai machines
# Disabling it lets cudf/cupy use their own CUDA paths without numba interference
RUN SITE=$(python -c 'import site; print(site.getsitepackages()[0])') && \
    cd $SITE && \
    mv numba_cuda numba_cuda_DISABLED 2>/dev/null || true && \
    mv _numba_cuda_redirector.py _numba_cuda_redirector.py.DISABLED 2>/dev/null || true && \
    for f in *.pth; do grep -q numba_cuda "$f" 2>/dev/null && mv "$f" "${f}.DISABLED"; done || true

# Verify non-GPU imports work (cudf needs GPU at runtime, skip in build)
RUN python -c "\
import lightgbm; print(f'lightgbm {lightgbm.__version__} OK'); \
import optuna; print(f'optuna {optuna.__version__} OK'); \
import ephem; print(f'ephem OK'); \
from scipy.sparse import csr_matrix; print('scipy.sparse OK'); \
print('=== BUILD DEPS VERIFIED (cudf verified at runtime) ==='); \
"

# Workspace created at runtime by vast.ai (or launch script)
WORKDIR /root

# Default env vars for training
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONUNBUFFERED=1
