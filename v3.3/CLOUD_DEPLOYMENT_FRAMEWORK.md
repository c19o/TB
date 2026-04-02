# Cloud Deployment Framework

Date: 2026-04-02

This is the maintained deployment authority for cloud training runs.

Authoritative files:
- `deploy_tf.py`
- `deploy_1w.sh`
- `deploy_1d.sh`
- `deploy_4h.sh`
- `deploy_1h.sh`
- `deploy_15m.sh`
- `contracts/pipeline_contract.json`
- `contracts/deploy_profiles.json`
- `contracts/private_shop_controls.json`
- `path_contract.py`
- `deploy_verify.py`
- `validate.py`
- `PRIVATE_SHOP_OPERATING_MODEL.md`

Reference-only docs:
- `TRAINING_*.md`
- `TF_CAVEAT_*.md`
- `FINAL_ETA_*.md`
- provider/region-specific launch notes such as `deploy_washington_1w.sh`

## Runtime Home

Local non-source data now belongs under `SAVAGE22_RUNTIME_HOME`.

Default Windows location:
- `C:\Users\C\Documents\Savage22 Runtime`

Expected subdirectories:
- `shared_db/`
- `runs/`
- `artifacts/`
- `logs/`
- `archives/`
- `downloads/`
- `cache/`

The maintained source tree should stay source-only. Use `python v3.3/audit_runtime_home.py` to inventory clutter and `python v3.3/audit_runtime_home.py --migrate` to move recognized runtime files out of the repo.

## Four-Root Contract

Every maintained run uses the same root model:
- `CODE_ROOT`: immutable release under `/workspace/releases/v3.3_<run_id>`
- `SHARED_DB_ROOT`: shared DB/text seed root
- `ARTIFACT_ROOT`: run-produced artifacts only
- `RUN_ROOT`: heartbeat, logs, manifest, checkpoints, quarantine

Run-produced artifacts must not be validated from `/workspace`, repo root, or release root fallbacks.

## Maintained Entry Points

Generic engine:

```bash
python -X utf8 v3.3/deploy_tf.py --tf 1d --ssh-host <host> --ssh-port <port>
```

Thin wrappers:

```bash
./v3.3/deploy_1w.sh --ssh-host <host> --ssh-port <port>
./v3.3/deploy_1d.sh --ssh-host <host> --ssh-port <port>
./v3.3/deploy_4h.sh --ssh-host <host> --ssh-port <port>
./v3.3/deploy_1h.sh --ssh-host <host> --ssh-port <port>
./v3.3/deploy_15m.sh --ssh-host <host> --ssh-port <port>
```

The engine is machine-agnostic. Provider wrappers may exist, but they are adapters only.

## Timeframe Policy

- `1w`: CPU-first, trimmed weekly path, no crosses.
- `1d`: CPU-first, crosses required, warm-start from `1w`.
- `4h`: hybrid transition lane, crosses required, warm-start from `1d`.
- `1h`: GPU-preferred to effectively GPU-required, warm-start from `4h`.
- `15m`: GPU-required, same-machine required, warm-start from `1h`.

The hard floor and preferred machine profile for each TF live in `contracts/deploy_profiles.json`.

## Private-Shop Controls

The private-shop operating layer is explicit, not implicit:
- `PRIVATE_SHOP_OPERATING_MODEL.md` defines the operating model
- `contracts/private_shop_controls.json` defines:
  - per-TF certified retrain backend intent
  - rare-feature health artifact requirements
  - training/inference parity artifact requirements
  - model governance state rules

These controls complement the deploy and pipeline contracts. They do not replace them.

## Timeframe Lanes

Per-timeframe lane folders live under repo-root `lanes/` as git worktrees:
- `lanes/1w`
- `lanes/1d`
- `lanes/4h`
- `lanes/1h`
- `lanes/15m`

Shared architecture changes land on branch `private-shop-core` first.
Per-TF branches are:
- `lane/1w`
- `lane/1d`
- `lane/4h`
- `lane/1h`
- `lane/15m`

Use:
- `scripts/create_lane_worktrees.ps1`
- `scripts/remove_lane_worktrees.ps1`

for idempotent lane creation and cleanup.

## Release Rules

- Release bundles are source-only and manifest-hashed.
- Verification happens against staging before promotion.
- Shared DB bootstrap is manifest-driven via `gcs_shared_seed.py`.
- `contracts/pipeline_contract.json` defines phase order and required artifacts.
- `contracts/deploy_profiles.json` defines machine policy, warm-start parent, env defaults, and resume policy.

## Full Maintained Phase Model

Maintained runs target this full phase sequence:
- `step0_preflight`
- `step1_features`
- `step2_crosses`
- `step3_baseline`
- `step4_optuna`
- `step5_retrain`
- `step6_optimizer`
- `step7_meta`
- `step8_lstm`
- `step9_pbo`
- `step10_audit`
- `complete`

`1w` is the trimmed weekly lane: it keeps `step2_crosses` explicitly skipped and its maintained completion target stops after `step6_optimizer`.

## Notes

- `deploy_washington_1w.sh` remains as a compatibility script for the proven weekly path, but it is no longer the architecture name for maintained deploys.
- If docs disagree with the contracts or deploy engine, the code/contracts win.
