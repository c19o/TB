#!/usr/bin/env python3
"""
1W GPU Fork Training with Dataset Caching.

Builds the LightGBM Dataset (EFB DISABLED) ONCE and saves to binary.
Subsequent runs load from binary (~1 second vs ~5-8 minutes).

EFB is disabled because GPU SpMV produces per-feature gradient sums.
With EFB, 2.2M features bundle into ~23K bins — histogram buffer overflow.

Usage:
  python train_1w_cached.py              # First run: build + save + train
  python train_1w_cached.py --from-cache # Skip EFB, load from binary
  python train_1w_cached.py --build-only # Build + save binary, don't train
"""

import os, sys, time, argparse
import numpy as np
import scipy.sparse as sp

# Windows CUDA DLL path
if sys.platform == 'win32':
    for cuda_path in [
        'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin',
        'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin',
    ]:
        if os.path.isdir(cuda_path):
            os.add_dll_directory(cuda_path)
            break

# Workaround: numpy 2.4 + scipy 1.17 on Windows = `from numpy import *`
# hangs (triggers lazy load of numpy.testing which deadlocks).
# lightgbm/__init__.py imports sklearn which imports scipy.stats which hangs.
# Fix: load lightgbm.basic directly, bypassing __init__.py and sklearn.
import types as _types
import importlib.util as _imputil
import ctypes as _ctypes

# 1. Create lightgbm package module (prevents __init__.py execution)
_lgbm_pkg = _types.ModuleType('lightgbm')
_lgbm_dir = os.path.join(os.path.dirname(_imputil.find_spec('lightgbm').origin))
_lgbm_pkg.__path__ = [_lgbm_dir]
_lgbm_pkg.__package__ = 'lightgbm'
sys.modules['lightgbm'] = _lgbm_pkg

# 2. Load DLL and register libpath
_dll_path = os.path.join(_lgbm_dir, 'lib', 'lib_lightgbm.dll')
_LIB = _ctypes.cdll.LoadLibrary(_dll_path)
_libpath = _types.ModuleType('lightgbm.libpath')
_libpath._LIB = _LIB
_libpath._find_lib_path = lambda: [_dll_path]
_libpath.__package__ = 'lightgbm'
sys.modules['lightgbm.libpath'] = _libpath

# 3. Register compat with minimal stubs (no sklearn/pandas needed for training)
_compat = _types.ModuleType('lightgbm.compat')
_compat.__package__ = 'lightgbm'
for _a in ['SKLEARN_INSTALLED', 'PANDAS_INSTALLED', 'PYARROW_INSTALLED',
           'CFFI_INSTALLED', 'MATPLOTLIB_INSTALLED', 'GRAPHVIZ_INSTALLED',
           'DATATABLE_INSTALLED', 'DASK_INSTALLED']:
    setattr(_compat, _a, False)
# Dummy types for isinstance() checks in basic.py
class _Unreachable:
    """Type that nothing is an instance of."""
    pass
class _FakeCffi:
    CData = type(None)
class _FakeCompute:
    all = None; equal = None
for _c in ['pd_DataFrame', 'pd_Series', 'pd_CategoricalDtype', 'pa_Array',
           'pa_ChunkedArray', 'pa_Table', 'dt_DataTable']:
    setattr(_compat, _c, _Unreachable)
_compat.pa_compute = _FakeCompute
_compat.pa_chunked_array = None
_compat.arrow_is_boolean = None
_compat.arrow_is_integer = None
_compat.arrow_is_floating = None
_compat.concat = None
_compat.arrow_cffi = _FakeCffi()
_compat._LGBMCpuCount = lambda only_physical_cores=True: os.cpu_count() or 1
sys.modules['lightgbm.compat'] = _compat

# 4. Load lightgbm.basic directly
_spec = _imputil.spec_from_file_location(
    'lightgbm.basic', os.path.join(_lgbm_dir, 'basic.py'))
_basic = _imputil.module_from_spec(_spec)
_basic.__package__ = 'lightgbm'
sys.modules['lightgbm.basic'] = _basic
_spec.loader.exec_module(_basic)

# 5. Wire up the package
_lgbm_pkg.basic = _basic
_lgbm_pkg.Dataset = _basic.Dataset
_lgbm_pkg.Booster = _basic.Booster
_lgbm_pkg.__version__ = getattr(_basic, '__version__', 'unknown')

# Alias for compatibility
import lightgbm as lgb

# Paths
V33_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ_PATH = os.path.join(V33_DIR, 'v2_crosses_BTC_1w.npz')
BINARY_PATH = os.path.join(V33_DIR, 'lgbm_dataset_1w.bin')
MODEL_PATH = os.path.join(V33_DIR, 'model_1w_gpu.json')

def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def build_and_save_binary(X_csr, y, binary_path):
    """Build LightGBM Dataset with EFB DISABLED and save to binary.

    EFB disabled because GPU SpMV produces n_features gradient sums (one per
    raw feature).  With EFB, LightGBM bundles 2.2M features into ~23K bins —
    writing 2.2M values into a 23K-bin histogram buffer overflows and corrupts
    results.  enable_bundle=False keeps total_hist_bins == n_features * 2
    (2 bins per binary feature), matching the SpMV output exactly.
    """
    log(f'Building Dataset from {X_csr.shape[1]:,} features (EFB DISABLED for GPU SpMV)...')
    t0 = time.time()

    # Feed sparse CSR directly — LightGBM accepts scipy sparse
    X_f32 = X_csr.astype(np.float32) if X_csr.dtype != np.float32 else X_csr

    ds = lgb.Dataset(
        X_f32, label=y,
        params={
            'feature_pre_filter': False,
            'max_bin': 255,
            'enable_bundle': False,
        },
        free_raw_data=False,
    )
    ds.construct()
    log(f'  Dataset constructed ({time.time()-t0:.1f}s)')

    ds.save_binary(binary_path)
    log(f'  Saved binary: {binary_path} ({os.path.getsize(binary_path)/1e6:.0f}MB)')
    log(f'  Total EFB time: {time.time()-t0:.1f}s')
    log(f'  Next run: use --from-cache to skip EFB (~1 second load)')

    return ds


def load_from_binary(binary_path):
    """Load pre-built Dataset from binary (skips EFB completely)."""
    log(f'Loading from binary cache: {binary_path}')
    t0 = time.time()
    ds = lgb.Dataset(binary_path)
    ds.construct()
    log(f'  Loaded in {time.time()-t0:.1f}s (EFB SKIPPED)')
    return ds


def train_gpu(ds, X_csr, num_rounds=200):
    """Train with GPU fork (cuda_sparse device type)."""
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'device_type': 'cuda_sparse',
        'num_leaves': 63,
        'learning_rate': 0.03,
        'max_bin': 255,
        'min_data_in_leaf': 3,
        'feature_pre_filter': False,
        'verbose': 1,
        # Limit CPU histogram pool to ~512 MB.  With enable_bundle=False
        # each leaf cache entry is ~96 MB (4M bins * 24B) + 2.2M
        # FeatureHistogram objects (~200 MB).  Default (pool_size<=0)
        # allocates num_leaves=63 cache entries = 18+ GB → OOM on 64 GB.
        'histogram_pool_size': 512,
    }

    log('Creating Booster (cuda_sparse)...')
    t0 = time.time()
    booster = lgb.Booster(params, ds)
    log(f'  Booster created ({time.time()-t0:.1f}s)')

    # Set external CSR for GPU histogram building
    log(f'Setting external CSR ({X_csr.nnz:,} NNZ)...')
    booster.set_external_csr(X_csr)
    log(f'  CSR set ({time.time()-t0:.1f}s)')

    # Train
    log(f'Training {num_rounds} rounds on GPU...')
    t_train = time.time()
    for i in range(num_rounds):
        booster.update()
        if i == 0:
            log(f'  Round 1: {time.time()-t_train:.2f}s')
        if (i + 1) % 50 == 0:
            log(f'  Round {i+1}/{num_rounds}: {time.time()-t_train:.1f}s')

    total = time.time() - t_train
    log(f'  Training done: {total:.1f}s ({total/num_rounds:.3f}s/round)')

    return booster


def train_cpu(ds, num_rounds=200):
    """Train with CPU (baseline comparison)."""
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'device_type': 'cpu',
        'num_leaves': 63,
        'learning_rate': 0.03,
        'max_bin': 255,
        'min_data_in_leaf': 3,
        'feature_pre_filter': False,
        'force_col_wise': True,
        'verbose': 0,
    }

    log('Training CPU baseline...')
    t0 = time.time()
    model = lgb.train(params, ds, num_boost_round=num_rounds)
    log(f'  CPU training: {time.time()-t0:.1f}s')
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--from-cache', action='store_true', help='Load from saved binary (skip EFB)')
    parser.add_argument('--build-only', action='store_true', help='Build + save binary, no training')
    parser.add_argument('--rounds', type=int, default=200)
    parser.add_argument('--cpu-baseline', action='store_true', help='Also run CPU for comparison')
    args = parser.parse_args()

    log('=== 1W GPU FORK TRAINING ===')

    # Load sparse crosses
    if not os.path.exists(NPZ_PATH):
        log(f'ERROR: {NPZ_PATH} not found. Run cross gen first.')
        sys.exit(1)

    X_csr = sp.load_npz(NPZ_PATH)
    log(f'Crosses: {X_csr.shape[0]} rows x {X_csr.shape[1]:,} features, {X_csr.nnz:,} NNZ')

    # Labels (random for testing — real training uses triple-barrier from feature_library)
    y = np.random.randint(0, 3, X_csr.shape[0])

    # Build or load Dataset
    if args.from_cache and os.path.exists(BINARY_PATH):
        ds = load_from_binary(BINARY_PATH)
    else:
        ds = build_and_save_binary(X_csr, y, BINARY_PATH)

    if args.build_only:
        log('Build-only mode. Binary saved. Done.')
        return

    # GPU training — NO CPU FALLBACK. Fix the GPU issue, don't hide it.
    booster = train_gpu(ds, X_csr, args.rounds)
    booster.save_model(MODEL_PATH)
    log(f'Model saved: {MODEL_PATH}')

    # Quick accuracy check — predict in row chunks to avoid dense OOM
    chunk = 100
    all_preds = []
    for i in range(0, X_csr.shape[0], chunk):
        block = X_csr[i:i+chunk].toarray().astype(np.float32)
        all_preds.append(booster.predict(block))
    preds = np.vstack(all_preds)
    acc = (np.argmax(preds, axis=1) == y).mean()
    log(f'Train accuracy: {acc:.3f}')

    # Optional CPU baseline
    if args.cpu_baseline:
        model_cpu = train_cpu(ds, args.rounds)
        log('CPU baseline complete')

    log('=== DONE ===')


if __name__ == '__main__':
    main()
