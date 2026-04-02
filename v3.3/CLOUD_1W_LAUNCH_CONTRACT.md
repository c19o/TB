# Cloud 1W Launch Contract

Last updated: 2026-04-01

This is the declared launch path for the approved Washington cloud machine.

## Approved Machine

- Type: `33923286`
- Region: `Washington, US`
- GPUs: `2x RTX 4090`
- CPU: `AMD EPYC 7B13 64-Core Processor`
- RAM: `516 GB`
- Max CUDA: `12.8`
- Price: `$0.832/hr` plus bandwidth

## Why This Contract Exists

The generic cloud wrapper has cloud-oriented behavior that is not the source of truth for a fresh `1w` retrain:

- it may conditionally set `ALLOW_CPU=1` for fallback environments
- it contains 1w-specific cloud shortcuts
- it is optimized for one-shot orchestration, not transparent operator review

For the first Washington `1w` run, use the explicit stepwise training flow so behavior is auditable.

## Launch Rules

1. Do not rent automatically.
2. Once the owner confirms launch timing, rent the exact approved machine.
3. On the machine, prefer the explicit `TRAINING_1W.md` sequence over a blind `cloud_run_tf.py` one-shot.
4. Do not force `ALLOW_CPU=1` on CUDA 12.8 if cuDF is available.
5. Download artifacts at every checkpoint.
6. Audit `1w` results before any lower timeframe run.

## Deployment Protocol

Use the enforced 4-root contract for maintained runs. Do not run directly from a mutable `/workspace/v3.3` tree.

- Code root: `/workspace/releases/v3.3_<run_id>`
- Shared DB root: `/workspace`
- Run root: `/workspace/runs/<run_id>`
- Artifact root: `/workspace/artifacts/<run_id>`
- Current symlink: `/workspace/current_v3.3`
- Run logs: `/workspace/runs/<run_id>/logs`
- Heartbeat: `/workspace/runs/<run_id>/cloud_run_1w_heartbeat.json`
- Shared cache: `/workspace/cache`

Required rules:

1. Upload into a staging release directory, verify it, then atomically promote it.
2. Never delete or overwrite the active release in place during a run.
3. Every run gets its own run root, artifact root, and heartbeat path.
4. Preflight must hard-fail if code root, artifact contract, run-root write test, or artifact-root write test is missing.
5. Run-produced artifacts belong only in the artifact root. Do not accept run-produced artifacts from `/workspace`, `/workspace/v3.3`, or the immutable code root.
6. Shared temp/output directories such as `_idx` must be cleared only inside the active artifact root, never by broad `/workspace` cleanup.

## Canonical Operator States

The weekly wrapper must expose only these heartbeat states:

- `running`: phase is actively executing
- `validated`: phase command returned and required artifacts were verified
- `failed`: phase aborted or artifact validation failed
- `complete`: entire weekly contract finished and final artifacts were verified

Do not treat "command launched" as equivalent to "phase complete".

## Canonical Weekly Phases

The operator-visible phase names are:

1. `step0_preflight`
2. `step1_features`
3. `step2_crosses`
4. `step3_baseline`
5. `step4_optuna`
6. `step5_retrain`
7. `step6_optimizer`
8. `complete`

The heartbeat payload should remain truthful enough that a stale process cannot impersonate a healthy run:

- `run_id`
- `session_name`
- `owner`
- `instance_id`
- `code_root`
- `shared_db_root`
- `run_root`
- `artifact_root`
- `heartbeat_path`
- `phase`
- `phase_seq`
- `status`
- `detail`
- `expected_artifacts`
- `artifact_contract`
- `release_manifest`

## Final Retrain Policy Knobs

The final retrain path now makes its scheduling decision explicit and uses a machine-aware `auto` default.

- `FINAL_RETRAIN_PARALLEL_POLICY`
  - `auto` (default)
  - `parallel`
  - `sequential`
- `FINAL_RETRAIN_PARALLEL_MIN_ROWS`
  - default: `512`
  - used only in `auto` mode
- `OPTUNA_FINAL_RETRAIN_MAX_PARALLEL_FOLDS`
  - default: `0` (auto)
  - optional cap on concurrent final retrain folds

Use these knobs under the 2026-04-01 owner override only when the evidence says the change improves speed without weakening Matrix Thesis, rare-signal retention, calibration, or OOS behavior.

## Declared 1W Run Path

Working directory:

```bash
cd /workspace/releases/v3.3_<run_id>
```

Environment baseline:

```bash
export PYTHONUNBUFFERED=1
export V30_DATA_DIR=/workspace/artifacts/<run_id>
export SAVAGE22_ARTIFACT_DIR=/workspace/artifacts/<run_id>
export SAVAGE22_RUN_DIR=/workspace/runs/<run_id>
export SAVAGE22_DB_DIR=/workspace
export SAVAGE22_V1_DIR=/workspace
export OPTUNA_SKIP_FINAL_RETRAIN=1
export V3_HOT_PATH_TRAINING=1
export V3_RUN_FI_STABILITY=0
export V3_RUN_ADVANCED_FI=0
export V3_CHECKPOINT_PERIOD=200
```

Conditional fallback only if cuDF is unavailable:

```bash
python - <<'PY'
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("cudf") else 1)
PY

# only if the command above fails:
export ALLOW_CPU=1
```

Execution sequence:

```bash
python -u build_features_v2.py --symbol BTC --tf 1w
python -u v2_cross_generator.py --tf 1w --symbol BTC --save-sparse
python -u ml_multi_tf.py --tf 1w --boost-rounds 800
python -u run_optuna_local.py --tf 1w --search-only
python -u ml_multi_tf.py --tf 1w --boost-rounds 800
python -u exhaustive_optimizer.py --tf 1w --n-trials 200
```

## Checkpoint Artifacts To Download

After each major phase, download at minimum:

- `features_BTC_1w.parquet`
- `v2_crosses_BTC_1w.npz`
- `v2_cross_names_BTC_1w.json`
- `model_1w.json`
- `model_1w_cpcv_backup.json`
- `optuna_configs_1w.json`
- `lgbm_dataset_1w.bin`
- `meta_model_1w.pkl`
- `shap_analysis_1w.json`
- `cpcv_oos_predictions_1w.pkl`

The machine-side source of truth for required phase outputs is:

- [`WEEKLY_1W_ARTIFACT_CONTRACT.json`](/C:/Users/C/Documents/Savage22%20Server/v3.3/WEEKLY_1W_ARTIFACT_CONTRACT.json)

The operator-side source of truth for heartbeat and logs is the per-run directory under `/workspace/runs/<run_id>`, not a shared root-level file.
The validator and launcher must bind phase completion to the declared run root plus artifact root, not to legacy `/workspace` or `/workspace/v3.3` fallbacks.

## Post-Run Audit

Before moving on from `1w`, review:

1. CPCV accuracy and class behavior
2. confidence-bucket quality
3. SHORT behavior and calibration
4. feature/cross usage sanity
5. artifact completeness

Only after that should the daemon-gated lower TF ladder resume.
