#!/usr/bin/env python3
"""
test_gpu_accuracy.py — Verify GPU histogram training matches CPU accuracy
=========================================================================

Loads real 1w data (features_BTC_1w.parquet + v2_crosses_BTC_1w.npz),
computes triple-barrier labels from OHLC, does 80/20 chronological split,
trains CPU model (200 rounds, early stopping) and GPU model (cuda_sparse,
200 rounds), then compares holdout accuracy.

PASS criteria: GPU accuracy within 1-2% of CPU accuracy.
"""

import sys, os

# --- Import deadlock fix ---
class _F:
    def find_module(self, n, p=None):
        try: os.write(1, b'.')
        except: pass
        return None
sys.meta_path.insert(0, _F())

# --- CUDA DLL directory ---
if sys.platform == 'win32':
    os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin')

import time
import json
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp

warnings.filterwarnings("ignore", category=FutureWarning)

# --- LightGBM import (bypass __init__.py deadlock) ---
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

# 3. Register compat with minimal stubs
_compat = _types.ModuleType('lightgbm.compat')
_compat.__package__ = 'lightgbm'
for _a in ['SKLEARN_INSTALLED', 'PANDAS_INSTALLED', 'PYARROW_INSTALLED',
           'CFFI_INSTALLED', 'MATPLOTLIB_INSTALLED', 'GRAPHVIZ_INSTALLED',
           'DATATABLE_INSTALLED', 'DASK_INSTALLED']:
    setattr(_compat, _a, False)
class _Unreachable:
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

import lightgbm as lgb

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_V33_DIR = (os.path.dirname(_THIS_DIR)
            if os.path.basename(_THIS_DIR) == 'gpu_histogram_fork'
            else _THIS_DIR)
_PROJECT_ROOT = os.path.dirname(_V33_DIR)

V30_DATA_DIR = os.environ.get('V30_DATA_DIR',
                              os.path.join(_PROJECT_ROOT, 'v3.0 (LGBM)'))
V32_DATA_DIR = os.path.join(_PROJECT_ROOT, 'v3.2_2.9M_Features')
DB_DIR = os.environ.get('SAVAGE22_DB_DIR', _V33_DIR)


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Triple-barrier label computation
# ═══════════════════════════════════════════════════════════════════════════
def compute_labels_from_ohlc(df):
    """Triple-barrier: 0=SHORT, 1=FLAT, 2=LONG. ATR-based TP/SL."""
    c = df['close'].astype(float).values
    h = df['high'].astype(float).values
    lo = df['low'].astype(float).values
    n = len(c)

    # ATR(14)
    tr = np.empty(n)
    tr[0] = h[0] - lo[0]
    for i in range(1, n):
        tr[i] = max(h[i] - lo[i], abs(h[i] - c[i-1]), abs(lo[i] - c[i-1]))
    atr = np.full(n, np.nan)
    for i in range(13, n):
        atr[i] = np.mean(tr[i-13:i+1])

    # Triple-barrier: tp=3xATR, sl=3xATR, max_hold=6
    labels = np.full(n, np.nan)
    tp_mult, sl_mult, max_hold = 3.0, 3.0, 6
    for i in range(n):
        if np.isnan(atr[i]) or np.isnan(c[i]):
            continue
        tp_price = c[i] + tp_mult * atr[i]
        sl_price = c[i] - sl_mult * atr[i]
        end_bar = min(i + max_hold, n - 1)
        if i + 1 > end_bar:
            continue
        hit = False
        for j in range(i + 1, end_bar + 1):
            if h[j] >= tp_price:
                labels[i] = 2.0
                hit = True
                break
            if lo[j] <= sl_price:
                labels[i] = 0.0
                hit = True
                break
        if not hit:
            labels[i] = 1.0
    return labels


# ═══════════════════════════════════════════════════════════════════════════
# 2. Data loading
# ═══════════════════════════════════════════════════════════════════════════
def find_file(candidates, label):
    for p in candidates:
        if os.path.isfile(p):
            return p
    log(f'ERROR: {label} not found. Searched:')
    for p in candidates:
        log(f'  {p}')
    return None


def load_data():
    """Load 1w parquet + crosses NPZ. Returns (X_csr, y) or None."""
    # Parquet
    parquet_candidates = [
        os.path.join(DB_DIR, 'features_BTC_1w.parquet'),
        os.path.join(V32_DATA_DIR, 'features_BTC_1w.parquet'),
        os.path.join(V30_DATA_DIR, 'features_BTC_1w.parquet'),
    ]
    parquet_path = find_file(parquet_candidates, '1w parquet')
    if parquet_path is None:
        return None

    log(f'Loading parquet: {parquet_path}')
    df = pd.read_parquet(parquet_path)
    log(f'  {len(df)} rows x {len(df.columns)} columns')

    # Labels
    if 'triple_barrier_label' in df.columns:
        y = pd.to_numeric(df['triple_barrier_label'], errors='coerce').values
        log('  Using pre-computed triple_barrier_label')
    elif 'close' in df.columns and 'high' in df.columns and 'low' in df.columns:
        log('  Computing triple-barrier labels from OHLC...')
        y = compute_labels_from_ohlc(df)
    else:
        log('ERROR: No labels and no OHLC data')
        return None

    valid = ~np.isnan(y)
    log(f'  Labels: LONG={int((y==2).sum())}, SHORT={int((y==0).sum())}, '
        f'FLAT={int((y==1).sum())}, NaN={int((~valid).sum())}')

    # Feature columns (exclude meta + target-like)
    meta_cols = {
        'timestamp', 'date', 'open', 'high', 'low', 'close', 'volume',
        'quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote',
        'open_time', 'date_norm',
    }
    target_like = {c for c in df.columns
                   if 'next_' in c.lower() or c == 'triple_barrier_label'}
    exclude = meta_cols | target_like
    feature_cols = [c for c in df.columns if c not in exclude]
    log(f'  Base feature columns: {len(feature_cols)}')

    X_base = df[feature_cols].values.astype(np.float32)
    X_base = np.where(np.isinf(X_base), np.nan, X_base)

    # Cross features NPZ
    npz_candidates = [
        os.path.join(DB_DIR, 'v2_crosses_BTC_1w.npz'),
        os.path.join(V32_DATA_DIR, 'v2_crosses_BTC_1w.npz'),
        os.path.join(V30_DATA_DIR, 'v2_crosses_BTC_1w.npz'),
    ]
    npz_path = find_file(npz_candidates, 'v2_crosses_BTC_1w.npz')

    if npz_path is not None:
        log(f'Loading crosses: {npz_path}')
        cross_matrix = sp.load_npz(npz_path).tocsr()

        if cross_matrix.indices.dtype != np.int32:
            cross_matrix.indices = cross_matrix.indices.astype(np.int32)
        if cross_matrix.indptr.dtype != np.int64:
            cross_matrix.indptr = cross_matrix.indptr.astype(np.int64)
        if cross_matrix.data.dtype != np.float64:
            cross_matrix.data = cross_matrix.data.astype(np.float64)

        log(f'  Crosses: {cross_matrix.shape[0]} x {cross_matrix.shape[1]:,} '
            f'({cross_matrix.nnz:,} nnz)')

        if cross_matrix.shape[0] != X_base.shape[0]:
            log(f'  WARNING: row mismatch {cross_matrix.shape[0]} vs {X_base.shape[0]}, '
                f'skipping crosses')
            cross_matrix = None
    else:
        cross_matrix = None

    # Combine base + crosses
    X_base_sp = sp.csr_matrix(X_base)
    if cross_matrix is not None:
        X_combined = sp.hstack([X_base_sp, cross_matrix], format='csr')
        log(f'  Combined: {X_combined.shape[0]} x {X_combined.shape[1]:,}')
    else:
        X_combined = X_base_sp
        log(f'  Base only: {X_combined.shape[0]} x {X_combined.shape[1]:,}')

    return X_combined, y


# ═══════════════════════════════════════════════════════════════════════════
# 3. Train/test split
# ═══════════════════════════════════════════════════════════════════════════
def split_data(X, y, holdout_frac=0.2):
    """Chronological 80/20 split on valid-label rows."""
    valid_idx = np.where(~np.isnan(y))[0]
    n_train = int(len(valid_idx) * (1 - holdout_frac))
    train_idx = valid_idx[:n_train]
    test_idx = valid_idx[n_train:]
    return (X[train_idx], X[test_idx],
            y[train_idx].astype(np.int32), y[test_idx].astype(np.int32))


# ═══════════════════════════════════════════════════════════════════════════
# 4. CPU training
# ═══════════════════════════════════════════════════════════════════════════
def train_cpu(X_train, y_train, X_test, y_test, num_rounds=200):
    """Train on CPU via Booster API (no lgb.train needed). Returns (accuracy, time, rounds)."""
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'device_type': 'cpu',
        'max_bin': 255,
        'num_leaves': 63,
        'learning_rate': 0.03,
        'feature_pre_filter': False,
        'force_col_wise': True,
        'is_enable_sparse': True,
        'min_data_in_leaf': 3,
        'min_gain_to_split': 2.0,
        'feature_fraction': 0.05,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'deterministic': True,
        'num_threads': 0,
        'verbosity': 0,
        'seed': 42,
    }

    log('=' * 60)
    log('CPU TRAINING (stock LightGBM, Booster API)')
    log('=' * 60)
    log(f'  X_train: {X_train.shape[0]} x {X_train.shape[1]:,}')
    log(f'  X_test:  {X_test.shape[0]} x {X_test.shape[1]:,}')

    ds_params = {'feature_pre_filter': False}

    t0 = time.perf_counter()
    dtrain = lgb.Dataset(X_train, label=y_train, params=ds_params, free_raw_data=False)
    dtrain.construct()
    ds_time = time.perf_counter() - t0
    log(f'  Dataset construction: {ds_time:.1f}s')

    # Use Booster API (same as GPU path for fair comparison)
    booster = lgb.Booster(params, dtrain)

    t_train = time.perf_counter()
    for i in range(num_rounds):
        booster.update()
        if i == 0:
            log(f'    Round 1: {time.perf_counter()-t_train:.2f}s')
        if (i + 1) % 50 == 0:
            log(f'    Round {i+1}/{num_rounds}: {time.perf_counter()-t_train:.1f}s')

    train_time = time.perf_counter() - t_train

    # Predict on holdout in chunks (avoid dense OOM)
    chunk = 100
    all_probs = []
    for i in range(0, X_test.shape[0], chunk):
        block = X_test[i:i+chunk]
        if sp.issparse(block):
            block = block.toarray().astype(np.float32)
        all_probs.append(booster.predict(block))
    probs = np.vstack(all_probs)
    preds = np.argmax(probs, axis=1)
    accuracy = np.mean(preds == y_test) * 100

    log(f'  Training time:    {train_time:.1f}s')
    log(f'  Holdout accuracy: {accuracy:.2f}%')
    log(f'  Preds: SHORT={int((preds==0).sum())}, FLAT={int((preds==1).sum())}, '
        f'LONG={int((preds==2).sum())}')

    return accuracy, train_time, num_rounds


# ═══════════════════════════════════════════════════════════════════════════
# 5. GPU training (cuda_sparse fork)
# ═══════════════════════════════════════════════════════════════════════════
def train_gpu(X_train, y_train, X_test, y_test, num_rounds=200):
    """Train on GPU using cuda_sparse device type + set_external_csr.
    Returns (accuracy, time, rounds_completed)."""
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'device_type': 'cuda_sparse',
        'max_bin': 255,
        'num_leaves': 63,
        'learning_rate': 0.03,
        'feature_pre_filter': False,
        'is_enable_sparse': True,
        'min_data_in_leaf': 3,
        'min_gain_to_split': 2.0,
        'feature_fraction': 0.05,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'deterministic': True,
        'num_threads': 0,
        'verbosity': 1,
        'seed': 42,
        'histogram_pool_size': 512,
    }

    log('=' * 60)
    log('GPU TRAINING (cuda_sparse fork)')
    log('=' * 60)
    log(f'  X_train: {X_train.shape[0]} x {X_train.shape[1]:,}')
    log(f'  X_test:  {X_test.shape[0]} x {X_test.shape[1]:,}')

    # Ensure CSR with correct dtypes for the GPU fork
    X_train_csr = X_train.tocsr() if sp.issparse(X_train) else sp.csr_matrix(X_train)
    if X_train_csr.indices.dtype != np.int32:
        X_train_csr.indices = X_train_csr.indices.astype(np.int32)
    if X_train_csr.indptr.dtype != np.int64:
        X_train_csr.indptr = X_train_csr.indptr.astype(np.int64)
    if X_train_csr.data.dtype != np.float64:
        X_train_csr.data = X_train_csr.data.astype(np.float64)

    log(f'  CSR NNZ: {X_train_csr.nnz:,}')

    ds_params = {'feature_pre_filter': False}

    t0 = time.perf_counter()
    ds_train = lgb.Dataset(X_train, label=y_train, params=ds_params, free_raw_data=False)
    ds_train.construct()
    ds_time = time.perf_counter() - t0
    log(f'  Dataset construction: {ds_time:.1f}s')

    # Create Booster and set external CSR for GPU histogram building
    log('  Creating Booster (cuda_sparse)...')
    booster = lgb.Booster(params, ds_train)
    log(f'  Setting external CSR ({X_train_csr.nnz:,} NNZ)...')
    booster.set_external_csr(X_train_csr)

    # Train round-by-round
    log(f'  Training {num_rounds} rounds on GPU...')
    t_train = time.perf_counter()
    for i in range(num_rounds):
        booster.update()
        if i == 0:
            log(f'    Round 1: {time.perf_counter()-t_train:.2f}s')
        if (i + 1) % 50 == 0:
            log(f'    Round {i+1}/{num_rounds}: {time.perf_counter()-t_train:.1f}s')

    train_time = time.perf_counter() - t_train
    log(f'  Training done: {train_time:.1f}s ({train_time/num_rounds:.3f}s/round)')

    # Predict on holdout in chunks
    chunk = 100
    all_probs = []
    for i in range(0, X_test.shape[0], chunk):
        block = X_test[i:i+chunk]
        if sp.issparse(block):
            block = block.toarray().astype(np.float32)
        all_probs.append(booster.predict(block))
    probs = np.vstack(all_probs)
    preds = np.argmax(probs, axis=1)
    accuracy = np.mean(preds == y_test) * 100

    log(f'  Holdout accuracy: {accuracy:.2f}%')
    log(f'  Preds: SHORT={int((preds==0).sum())}, FLAT={int((preds==1).sum())}, '
        f'LONG={int((preds==2).sum())}')

    return accuracy, train_time, num_rounds


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    log('=' * 60)
    log('GPU ACCURACY VERIFICATION TEST')
    log('=' * 60)
    log(f'DLL: {_dll_path}')
    log(f'V33 dir: {_V33_DIR}')
    log('')

    # 1. Load data
    result = load_data()
    if result is None:
        log('FATAL: Could not load data')
        sys.exit(1)
    X, y = result

    # 2. Split 80/20
    X_train, X_test, y_train, y_test = split_data(X, y)
    log(f'Split: {X_train.shape[0]} train, {X_test.shape[0]} test')
    log('')

    # 3. CPU training
    cpu_acc, cpu_time, cpu_best = train_cpu(X_train, y_train, X_test, y_test, 200)
    log('')

    # 4. GPU training
    gpu_acc, gpu_time, gpu_rounds = train_gpu(X_train, y_train, X_test, y_test, 200)
    log('')

    # 5. Compare
    diff = abs(cpu_acc - gpu_acc)
    log('=' * 60)
    log('RESULTS COMPARISON')
    log('=' * 60)
    log(f'  CPU accuracy:   {cpu_acc:.2f}%  ({cpu_time:.1f}s, {cpu_best} rounds)')
    log(f'  GPU accuracy:   {gpu_acc:.2f}%  ({gpu_time:.1f}s, {gpu_rounds} rounds)')
    log(f'  Difference:     {diff:.2f}%')
    log(f'  Speedup:        {cpu_time/gpu_time:.2f}x' if gpu_time > 0 else '  Speedup: N/A')
    log('')

    if diff <= 2.0:
        log('PASS: GPU accuracy within 2% of CPU')
    elif diff <= 5.0:
        log('WARN: GPU accuracy differs by {:.2f}% (>2% threshold)'.format(diff))
    else:
        log('FAIL: GPU accuracy differs by {:.2f}% (>5%)'.format(diff))

    log('=' * 60)
    log('DONE')


if __name__ == '__main__':
    main()
