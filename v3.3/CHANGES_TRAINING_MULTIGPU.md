# Training-MultiGPU — Optimization #12

## Summary
Multi-GPU trial-level parallelism for Optuna HPO. Each Optuna trial runs on a separate GPU via round-robin assignment. 4 GPUs = 4 concurrent trials.

## Architecture
- **Trial-level parallelism** (NOT data-parallel training)
- Each trial sees ALL features and ALL rows — no partitioning
- `feature_pre_filter=False` on every trial (matrix thesis protected)
- Round-robin GPU assignment: `gpu_id = trial.number % num_gpus`
- `TPESampler(constant_liar=True)` for thread-safe parallel sampling
- SQLite storage for thread-safe study persistence

## Files Changed

### New: `v3.3/multi_gpu_optuna.py`
Self-contained module with:
- `detect_gpus()` — PyTorch → nvidia-smi fallback GPU detection
- `MultiGPUConfig` — config dataclass (num_gpus, device_type, n_jobs, threads_per_trial)
- `get_multi_gpu_config()` — builds config from env vars + auto-detection
- `apply_gpu_params()` — sets device_type, gpu_device_id, histogram_pool_size per trial
- `create_gpu_safe_sampler()` — TPESampler with constant_liar=True
- `gpu_oom_handler` — decorator that catches CUDA OOM, returns inf (failed trial)
- `get_gpu_trial_summary()` — GPU distribution logging

### Modified: `v3.3/run_optuna_local.py`
- Imports `multi_gpu_optuna` module
- `run_search_for_tf()`: uses `get_multi_gpu_config()` for GPU detection, overrides n_jobs to GPU count
- `build_phase1_objective()`: accepts `gpu_cfg` param, uses `apply_gpu_params()` for per-trial GPU assignment
- Objective wrapped with `gpu_oom_handler` when multi-GPU active
- Sampler uses `constant_liar=True` when >1 GPU detected
- Phase 1 completion logs GPU trial distribution summary
- `main()`: auto-detects GPU count for n_jobs default

## Environment Variables

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `MULTI_GPU` | `1` / `0` | auto-detect | Force enable/disable multi-GPU |
| `LGBM_NUM_GPUS` | integer | 0 (detect) | Override GPU count (skips detection) |

## Hardware Targets
- **4x A100** (vast.ai ~$4/hr): `MULTI_GPU=1` → 4 concurrent trials
- **8x H100** (Lambda ~$20/hr): `MULTI_GPU=1` → 8 concurrent trials
- **Single GPU** (local 3090): auto-detected → n_jobs=1, single GPU training
- **CPU-only**: `MULTI_GPU=0` or no GPUs → falls back to CPU parallel trials

## Usage
```bash
# Auto-detect GPUs (recommended)
python run_optuna_local.py --tf 1d

# Force multi-GPU with explicit count
LGBM_NUM_GPUS=4 python run_optuna_local.py --tf 1d

# Force disable (CPU only)
MULTI_GPU=0 python run_optuna_local.py --tf 1d

# Override n_jobs manually (ignores GPU count)
python run_optuna_local.py --tf 1d --n-jobs 2
```

## OOM Handling
- GPU OOM errors are caught per-trial (not fatal)
- Failed trial returns `inf` — Optuna marks it as failed and avoids that region
- GPU memory is freed via `torch.cuda.empty_cache()` + gc
- Other trials continue on their assigned GPUs

## Matrix Thesis Compliance
- Each trial uses ALL features (no feature partitioning)
- Each trial uses ALL rows (no data partitioning)
- `feature_pre_filter=False` always set
- `feature_fraction >= 0.7` range unchanged
- Sparse CSR preserved — no dense conversion
