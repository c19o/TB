"""
Pytest fixtures and markers for gpu_histogram_fork tests.

Markers:
    gpu  - skip if no CUDA device available
    slow - skip when running with --fast flag
"""

import pytest
import numpy as np
from scipy.sparse import random as sparse_random, csr_matrix


# ---------------------------------------------------------------------------
# CLI option: --fast
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption(
        "--fast", action="store_true", default=False,
        help="Skip tests marked @pytest.mark.slow",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires CUDA GPU")
    config.addinivalue_line("markers", "slow: slow test, skipped with --fast")


def pytest_collection_modifyitems(config, items):
    # --fast: skip slow tests
    if config.getoption("--fast"):
        skip_slow = pytest.mark.skip(reason="skipped with --fast")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    # auto-skip gpu tests when no CUDA
    try:
        import cupy
        cupy.cuda.Device(0).compute_capability
        has_gpu = True
    except Exception:
        has_gpu = False

    if not has_gpu:
        skip_gpu = pytest.mark.skip(reason="no CUDA device available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_csr():
    """Small 200x50 sparse CSR matrix for unit tests (density ~5%)."""
    rng = np.random.default_rng(42)
    mat = sparse_random(200, 50, density=0.05, format='csr', random_state=rng)
    return mat


@pytest.fixture
def profile_4h_csr():
    """
    4h-profile test data: 2190 rows x 500K features, density ~0.3%.
    Mimics the 4h timeframe sparse cross matrix shape.
    """
    rng = np.random.default_rng(22)
    n_rows = 2_190
    n_cols = 500_000
    density = 0.003
    mat = sparse_random(n_rows, n_cols, density=density, format='csr',
                        dtype=np.float32, random_state=rng)
    return mat


@pytest.fixture
def gradients_hessians(small_csr):
    """Gradient and hessian vectors matching small_csr row count."""
    rng = np.random.default_rng(42)
    n = small_csr.shape[0]
    grad = rng.standard_normal(n).astype(np.float32)
    hess = np.abs(rng.standard_normal(n)).astype(np.float32) + 1e-6
    return grad, hess
