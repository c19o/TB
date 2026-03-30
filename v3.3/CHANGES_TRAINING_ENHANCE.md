# Training Pipeline Enhancements (2026-03-29)

## Optimizations Implemented

### OPT-9: CSC Format for LightGBM
- **Files**: `run_optuna_local.py`, `ml_multi_tf.py`
- After building the combined sparse matrix (base + crosses), convert CSR to CSC via `.tocsc()`
- LightGBM with `force_col_wise=True` iterates features column-wise; CSC stores columns contiguously in memory, improving cache locality during histogram building
- No impact on row-slicing for CPCV (scipy handles CSC slicing correctly)

### OPT-10: WilcoxonPruner for Inter-Fold Pruning
- **Files**: `run_optuna_local.py`
- Replaced MedianPruner/PatientPruner with `WilcoxonPruner(p_threshold=0.1, n_startup_steps=2)`
- After each CPCV fold completes, reports fold mlogloss to Optuna via `trial.report(mlogloss, step=fold_idx)`
- Calls `trial.should_prune()` to statistically test if the trial is worse than the best
- Removed `_RoundPruningCallback` from training callbacks (step namespace collision with fold-level pruning)
- Intra-fold early stopping preserved via native `lgb.early_stopping()`

### OPT-11: extra_trees in Optuna Search Space
- **Files**: `run_optuna_local.py`, `ml_multi_tf.py`
- Added `extra_trees = trial.suggest_categorical('extra_trees', [True, False])` to the Phase 1 objective
- Propagated through validation gate, final retrain, and ml_multi_tf.py Optuna param overlay
- For binary features (0/1), extra_trees is a no-op (only 1 possible threshold) -- safe diversity injection

### OPT-13: GC Disable During Training
- **Files**: `run_optuna_local.py`, `ml_multi_tf.py`
- `gc.disable()` before CPCV/Optuna training loops, `gc.enable(); gc.collect()` after
- LightGBM C++ does the heavy lifting; Python GC cycle detection during training is pure overhead

### OPT-14: NUMA Binding for Multi-Socket Cloud Machines
- **Files**: `cloud_run_tf.py`
- Installs `numactl` during setup
- Detects NUMA topology via `numactl --hardware`
- For multi-node systems: wraps Optuna search and training commands with `numactl --cpunodebind=0 --membind=0`
- Single-node systems: no binding (no overhead)
- Graceful fallback if numactl unavailable

## Wave 3 Re-Applied Fixes

### force_row_wise for 15m Timeframe
- **Files**: `ml_multi_tf.py`, `run_optuna_local.py`
- 15m has 294K rows / ~23K EFB bundles (ratio 12.8) -- row-wise iteration is faster
- Checks `TF_FORCE_ROW_WISE` from config.py; sets `force_row_wise=True` and removes `force_col_wise` for 15m
- Applied in both CPCV params and final model params

### cgroup-Aware CPU Count + threadpoolctl
- **Files**: `run_optuna_local.py`, `ml_multi_tf.py`
- Replaced `os.cpu_count()` with `get_cpu_count()` (from `hardware_detect.py`) which reads `/sys/fs/cgroup` on cloud
- Added `threadpoolctl.threadpool_limits()` at import time to lift MKL/OpenBLAS thread caps

### SharedMemory for CPCV IPC
- **Files**: `ml_multi_tf.py`
- Parallel CPCV workers now receive CSR matrix data via `multiprocessing.shared_memory.SharedMemory`
- Eliminates pickle serialization bottleneck for large matrices (15m: ~2GB+ CSR data)
- Workers attach to named shared memory blocks (zero-copy), reconstruct CSR matrix in-process
- Graceful fallback to pickle if SharedMemory unavailable (e.g., Python < 3.8 edge case)
- Shared memory blocks cleaned up (close + unlink) after all workers complete
