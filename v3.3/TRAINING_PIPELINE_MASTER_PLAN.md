# Training Pipeline Master Plan

Last updated: 2026-04-01

## What Is True Today (Cross-Lane Consolidation)
- `1w` training artifacts are present and valid at the repo level (`v3.3/1w_cloud_artifacts_v3/`).
- `python validate.py` and `python convention_gate.py check` pass in governance (`97/97`, conventions all clear).
- Local/lanes all agree the remaining runtime dependency is a **fresh proof-run for `1d` step 2 daemon path** (`RELOAD -> READY -> RESULT`) after RELOAD/int64 fixes.
- Non-daemon blockers still recurring: local CUDA13 smoke defaults to require `ALLOW_CPU=1` in the current environment unless cuDF path is aligned; this is a deployment caveat, not an `1w` model-quality blocker.
- `1d` lower timeframes (`1d`, `4h`, `1h`, `15m`) are queued behind successful step-2 rerun evidence and should not proceed on assumed-green status.

## Before 1w (Owner-Approved Launch Preconditions)
1. [P0] Confirm owner authorization and timing for approved Washington launch target (`33923286`, Washington, US, 2x RTX 4090, CUDA 12.8). Do not auto-rent.
2. [P0] If a fresh end-to-end `1w` run on current code has not been executed in this cycle, run explicit 1w sequence from `CLOUD_1W_LAUNCH_CONTRACT.md` and capture all stage artifacts.
3. [P1] Re-run and record `validate.py`, `convention_gate.py check`, and baseline `smoke_test_pipeline.py --tf 1w` in the selected launch context; keep `ALLOW_CPU=1` fallback evidence for local CUDA13 environments.
4. [P1] Gate `1w` completion only if post-run checkpoints include: features parquet, sparse crosses + names, `model_1w*.json`, `optuna_configs_1w.json`, `meta_model_1w.pkl`, `cpcv_oos_1w.pkl`, and confidence/SHAP/feature artifacts.

## After 1w
1. [P0] Execute and archive `1d` step 2 rerun through daemon contract path; collect logs for each cycle: RELOAD count, READY acknowledgements, batch dispatch, nnz/feature growth, memory deltas.
2. [P0] On successful rerun evidence, continue `1d` pipeline steps 3-7 and record real wall-clock/timing and artifact deltas versus expected profile.
3. [P1] Resume ladder in order: `4h -> 1h -> 15m` using existing dependency chain and per-timeframe checkpoints.
4. [P1] Update `QA_VERIFICATION_2026-04-01.md`, `DAEMON_RELOAD_AUDIT.md`, `PRODUCTION_READINESS.md`, and `ISSUE_STATUS_AUDIT.md` after each gating transition.

## Owner Approval Required
1. [P1] Speed/performance policy changes still under owner control (`OMP_NUM_THREADS`, NUMA strategy, warp-reduce/CUDA kernel mode, any non-default daemon parallelism toggles).
2. [P2] Any launch-timing/rental decisions beyond the approved `1w` machine and any manual cloud-machine handoff.
3. [P2] Governance/process fixes that require policy changes (e.g., stricter KB-evidence enforcement, mirror failover policy, workflow-state synchronization).
