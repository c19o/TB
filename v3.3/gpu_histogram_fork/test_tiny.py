#!/usr/bin/env python3
"""Tiny GPU test - 100 rows, 500 features."""
import os, sys
if sys.platform == 'win32':
    for p in ['C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin',
              'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4/bin']:
        if os.path.isdir(p):
            os.add_dll_directory(p)
            break

import numpy as np
import scipy.sparse as sp
import lightgbm as lgb

print('imports done', flush=True)

np.random.seed(42)
X = sp.random(100, 500, density=0.1, format='csr', dtype=np.float32)
X.data[:] = 1.0
y = np.random.randint(0, 3, 100)
print(f'X: {X.shape}, nnz: {X.nnz}', flush=True)

ds = lgb.Dataset(X, label=y,
                 params={'feature_pre_filter': False, 'max_bin': 255},
                 free_raw_data=False)
ds.construct()
print('Dataset constructed', flush=True)

params = {
    'objective': 'multiclass', 'num_class': 3,
    'device_type': 'cuda_sparse',
    'num_leaves': 31, 'learning_rate': 0.1,
    'max_bin': 255, 'min_data_in_leaf': 3,
    'feature_pre_filter': False, 'verbose': 1,
}
print('Creating booster...', flush=True)
b = lgb.Booster(params, ds)
print('Booster created', flush=True)

b.set_external_csr(X)
print('CSR set', flush=True)

b.update()
print('Round 1 done!', flush=True)

b.update()
print('Round 2 done!', flush=True)

print('SUCCESS', flush=True)
