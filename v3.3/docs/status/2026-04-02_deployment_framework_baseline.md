# 2026-04-02 Deployment Framework Baseline

## What Changed

- Added a maintained generic deploy engine: `deploy_tf.py`
- Added maintained thin wrappers:
  - `deploy_1w.sh`
  - `deploy_1d.sh`
  - `deploy_4h.sh`
  - `deploy_1h.sh`
  - `deploy_15m.sh`
- Added deploy-profile contract: `contracts/deploy_profiles.json`
- Extended the unified pipeline contract to the full maintained phase model through `step10_audit`
- Added local runtime-home tooling:
  - `runtime_home.py`
  - `audit_runtime_home.py`
- Upgraded `deploy_manifest.py` to build recursive source-only manifests
- Started aligning runner/validators with the unified contract and artifact-root policy
- Corrected maintained `1w` back to the trimmed weekly lane:
  - `step2_crosses` is a validated skip
  - `step7_meta` through `step10_audit` are validated skips
  - maintained `1w` completion ends after `step6_optimizer`

## What Is Authoritative Now

- `deploy_tf.py`
- `contracts/pipeline_contract.json`
- `contracts/deploy_profiles.json`
- `path_contract.py`
- `deploy_verify.py`
- `validate.py`
- `CLOUD_DEPLOYMENT_FRAMEWORK.md`
- `CLOUD_*_PROFILE.md`

## What Remains Legacy Or Compatibility-Only

- `deploy_washington_1w.sh`
- `TRAINING_*.md`
- `TF_CAVEAT_*.md`
- `FINAL_ETA_*.md`
- provider/region-specific notes that are not thin wrappers over `deploy_tf.py`

## Timeframes Covered

- `1w`
- `1d`
- `4h`
- `1h`
- `15m`

## Current Supported Machine Policy

- `1w`: CPU-first, trimmed weekly path, no crosses
- `1d`: CPU-first, crosses required
- `4h`: hybrid transition lane
- `1h`: GPU-preferred to effectively GPU-required
- `15m`: GPU-required and same-machine required

## Known Gaps / Deferred Work

- `deploy_tf.py` is the maintained engine, but the older weekly-specific shell path still exists and should eventually be demoted to a pure compatibility shim.
- Lower-TF GPU/fork deploy paths still depend on the broader `gpu_histogram_fork` runtime assumptions and need continued runtime validation on real boxes.
- Some older docs still contain ETA-style or aspirational guidance and should be treated as reference only.

## Defining Files

- `contracts/pipeline_contract.json`
- `contracts/deploy_profiles.json`
- `deploy_tf.py`
- `runtime_home.py`
- `audit_runtime_home.py`
- `deploy_manifest.py`
