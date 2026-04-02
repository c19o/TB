# Pre-1W Issue Board

Last updated: 2026-04-01

This is the local visible board for deciding what must be fixed before a fresh `1w` retrain.

## Fixed In Local Wave

### `SAV-54` Harden `ALLOW_CPU=1` validation gate
- Status: fixed enough for retrain gating
- Result:
  - [validate.py](/C:/Users/C/Documents/Savage22%20Server/v3.3/validate.py) enforces the cuDF-less `ALLOW_CPU=1` contract
  - [cloud_run_tf.py](/C:/Users/C/Documents/Savage22%20Server/v3.3/cloud_run_tf.py) already sets `ALLOW_CPU=1` for cloud/runtime fallback
- Verification:
  - `python v3.3/validate.py` -> `97/97 PASSED, TRAINING APPROVED`

### `SAV-51` Fix `validate.py` bagging floor check to `0.95`
- Status: fixed
- Result:
  - config-level validation now requires `bagging_fraction >= 0.95`
  - hardcoded scan now blocks any `bagging_fraction < 0.95`
  - stale `0.8` leftovers in bench/smoke/classifier files were raised to `0.95`
- Verification:
  - `python v3.3/validate.py` -> `97/97 PASSED, TRAINING APPROVED`

### `SAV-19` Reopened kill switch / SIGTERM safety gap
- Status: fixed locally
- Result:
  - [live_trader.py](/C:/Users/C/Documents/Savage22%20Server/v3.3/live_trader.py) now has explicit `SIGINT` / `SIGTERM` handling with stop-event driven sleeps so shutdown is prompt instead of waiting on long sleeps
- Verification:
  - `python -m py_compile v3.3/live_trader.py`

### Observability / watchdog gap from `SAV-40`
- Status: partially fixed locally
- Result:
  - [cloud_run_tf.py](/C:/Users/C/Documents/Savage22%20Server/v3.3/cloud_run_tf.py) now writes a heartbeat file, tracks current step, refreshes progress during tee-streaming steps, and emits watchdog warnings on prolonged no-progress
- Verification:
  - `python -m py_compile v3.3/cloud_run_tf.py`

## Remaining Before 1W Production Retrain

### Review the new watchdog/heartbeat path for owner approval
- Reason: it changes runtime observability behavior, though not core training logic
- Evidence:
  - [cloud_run_tf.py](/C:/Users/C/Documents/Savage22%20Server/v3.3/cloud_run_tf.py)

### Run a fresh `1w` smoke/retrain with artifact capture
- Reason: validation is green, but production readiness still needs a real run with visible checkpoints and artifact verification
- Evidence:
  - [PRODUCTION_READINESS.md](/C:/Users/C/Documents/Savage22%20Server/v3.3/PRODUCTION_READINESS.md)

## Can Wait Until After 1W

### `SAV-4`, `SAV-44`, `SAV-59`, `SAV-60`, `SAV-63`, `SAV-64`, `SAV-68`, `SAV-69`
- Reason: these are critical for `1d+` and the lower-TF ladder, but not blockers for a fresh `1w` retrain if `1w` does not depend on the daemon reload path
- Evidence:
  - [SESSION_RESUME.md](/C:/Users/C/Documents/Savage22%20Server/v3.3/SESSION_RESUME.md): `1w` complete; `1d+` blocked on daemon reload path
- Caveat: still high-priority for the broader project; just not necessarily a pre-`1w` gate

### `SAV-3`, `SAV-5`, `SAV-6`, `SAV-7`
- Reason: these are lower-TF training execution tasks (1D/4H/1H/15M), so they do not gate a fresh `1w` training run
- Evidence:
  - `SAV-3`: depends on daemon stability already tracked by pre-1w blockers and must follow a stable 1w+daemon baseline
  - `SAV-5`/`SAV-6`/`SAV-7`: execution lane and machine rental runbook items for cascade-down TFs only

### `SAV-16`
- Reason: this is a modeling enhancement in `ml_multi_tf.py` (`HMM CPCV`, decomposed targets, LSTM stacker) and should be scheduled after baseline pre-1w readiness is stable

### `SAV-29`
- Reason: this is rollback + diagnosis hardening (`deploy_model.py`/`model_diagnosis.py`) and should be done after the first clean `1w` retrain/repro path exists

### `SAV-48` deployment / rollback re-audit
- Reason: useful for production maturity, but not a hard gate if we are only deciding whether the next `1w` retrain can run cleanly

### `SAV-57` stale "74 checks" references
- Reason: documentation correctness, not a training blocker

### `SAV-67` Perplexity-first research drift audit
- Reason: governance-important, but not a direct training blocker once current prompt fixes are in place

## Owner Approval Required Before Production Use

These are not necessarily bugs. They are changes that affect speed / throughput / runtime behavior and therefore require owner approval before production use.

### `SAV-53` `OMP_NUM_THREADS=4` cross-gen tuning
- Reason: explicitly speed-affecting runtime knob
- Evidence:
  - [PRODUCTION_READINESS.md](/C:/Users/C/Documents/Savage22%20Server/v3.3/PRODUCTION_READINESS.md)

### Recent runtime-affecting changes in:
- [cloud_run_tf.py](/C:/Users/C/Documents/Savage22%20Server/v3.3/cloud_run_tf.py)
- [gpu_daemon.py](/C:/Users/C/Documents/Savage22%20Server/v3.3/gpu_daemon.py)
- [gpu_histogram_fork/train_1w_cached.py](/C:/Users/C/Documents/Savage22%20Server/v3.3/gpu_histogram_fork/train_1w_cached.py)
- [run_optuna_local.py](/C:/Users/C/Documents/Savage22%20Server/v3.3/run_optuna_local.py)
- [ml_multi_tf.py](/C:/Users/C/Documents/Savage22%20Server/v3.3/ml_multi_tf.py)

### Any change to:
- `RIGHT_CHUNK`
- fold parallelism
- thread counts / OMP settings
- dense vs sparse path
- GPU vs CPU path
- retry / scheduling behavior
- machine choice / throughput profile

## Current Local Call

If the goal is a fresh `1w` retrain before anything else, the local gate is now:
1. owner review of the new watchdog/heartbeat behavior
2. run a fresh visible `1w` smoke/retrain with artifact capture

The daemon incident remains critical for the project as a whole, but looks more like a `1d+` gate than a strict `1w` retrain gate.
