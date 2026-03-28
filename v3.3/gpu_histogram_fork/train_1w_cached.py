#!/usr/bin/env python3
"""
1W GPU Fork Training with Dataset Caching.

Builds the LightGBM Dataset (EFB bundling) ONCE and saves to binary.
Subsequent runs load from binary (~1 second vs ~5-8 minutes).

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

import lightgbm as lgb

# Paths
V33_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ_PATH = os.path.join(V33_DIR, 'v2_crosses_BTC_1w.npz')
BINARY_PATH = os.path.join(V33_DIR, 'lgbm_dataset_1w.bin')
MODEL_PATH = os.path.join(V33_DIR, 'model_1w_gpu.json')

def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def build_and_save_binary(X_csr, y, binary_path):
    """Build LightGBM Dataset with full EFB bundling and save to binary."""
    log(f'Building Dataset from {X_csr.shape[1]:,} features (EFB bundling)...')
    t0 = time.time()

    # LightGBM needs dense for Dataset construction with EFB
    X_dense = X_csr.toarray().astype(np.float32)
    log(f'  Dense: {X_dense.nbytes/1e9:.1f}GB ({time.time()-t0:.1f}s)')

    ds = lgb.Dataset(
        X_dense, label=y,
        params={'feature_pre_filter': False, 'max_bin': 255},
        free_raw_data=False,
    )
    ds.construct()
    log(f'  Dataset constructed ({time.time()-t0:.1f}s)')

    ds.save_binary(binary_path)
    log(f'  Saved binary: {binary_path} ({os.path.getsize(binary_path)/1e6:.0f}MB)')
    log(f'  Total EFB time: {time.time()-t0:.1f}s')
    log(f'  Next run: use --from-cache to skip EFB (~1 second load)')

    del X_dense
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
        'verbose': 0,
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

    # GPU training
    try:
        booster = train_gpu(ds, X_csr, args.rounds)
        booster.save_model(MODEL_PATH)
        log(f'Model saved: {MODEL_PATH}')

        # Quick accuracy check
        X_dense = X_csr.toarray().astype(np.float32)
        preds = booster.predict(X_dense)
        acc = (np.argmax(preds, axis=1) == y).mean()
        log(f'Train accuracy: {acc:.3f}')
    except Exception as e:
        log(f'GPU training failed: {e}')
        log('Falling back to CPU...')
        model = train_cpu(ds, args.rounds)
        model.save_model(MODEL_PATH)
        log(f'CPU model saved: {MODEL_PATH}')

    # Optional CPU baseline
    if args.cpu_baseline:
        model_cpu = train_cpu(ds, args.rounds)
        log('CPU baseline complete')

    log('=== DONE ===')


if __name__ == '__main__':
    main()
