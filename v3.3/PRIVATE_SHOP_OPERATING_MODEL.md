# Private-Shop Operating Model

Date: 2026-04-02

This file encodes the private-shop mindset for Savage22 as an explicit project authority, not an implicit terminal habit.

## What Private-Shop Means Here

- one maintained execution backend per heavy phase
- source-only repo, runtime data outside the repo
- machine-agnostic deployment with strict run provenance
- per-timeframe lanes with separate terminals and isolated branches
- no hidden fallback from a certified parallel path to a slower path
- promotion based on evidence, not on “training finished”

## Shared Core And Timeframe Lanes

Shared architecture branch:
- `private-shop-core`

Timeframe branches:
- `lane/1w`
- `lane/1d`
- `lane/4h`
- `lane/1h`
- `lane/15m`

Visible worktree folders:
- `lanes/1w`
- `lanes/1d`
- `lanes/4h`
- `lanes/1h`
- `lanes/15m`

Rule:
- shared infra lands on `private-shop-core`
- timeframe-specific divergence lands in the matching lane branch

## Runtime Discipline

All non-source data belongs under `SAVAGE22_RUNTIME_HOME`.

Default Windows runtime home:
- `C:\Users\C\Documents\Savage22 Runtime`

Canonical subdirectories:
- `shared_db/`
- `runs/`
- `artifacts/`
- `logs/`
- `archives/`
- `downloads/`
- `cache/`

The maintained repo should stay source-only. Local defaults now point to runtime-home roots instead of the code tree.

## Backend Certification

The maintained backend intent lives in `contracts/private_shop_controls.json`.

Current target posture:
- `1w step5`: `cpu_parallel`
- `1d step5`: `cpu_parallel_memmap`
- `4h step5`: `hybrid_parallel`
- `1h step5`: `gpu_parallel`
- `15m step5`: `gpu_parallel_same_machine`

Rules:
- execution mode is chosen up front
- degradation policy is `fail_fast`
- no silent fallback to sequential when the contract says parallel

## Rare-Feature Health

Rare features are the edge, not garbage.

Maintained runs should emit rare-feature health artifacts per TF:
- `rare_feature_health_1w.json`
- `rare_feature_health_1d.json`
- `rare_feature_health_4h.json`
- `rare_feature_health_1h.json`
- `rare_feature_health_15m.json`

Minimum health dimensions:
- protected-prefix coverage
- split fire counts
- active-bar coverage
- event overlap where applicable
- rare-feature drift between train and inference contexts

## Training / Inference Parity

Training is not production-ready until the saved bundle is compatible with inference.

Maintained runs should produce per-TF parity artifacts:
- `training_inference_parity_<tf>.json`

These bind the parity-critical bundle:
- model
- saved feature list
- calibrator
- optuna/optimizer configs
- any TF-specific inference dependencies

## Model Governance

Every trained model should have an explicit governance state:
- `research`
- `candidate`
- `shadow`
- `live`
- `retired`

Maintained runs should emit:
- `model_governance_<tf>.json`

Promotion gates should include:
- backend certification
- rare-feature health
- training/inference parity
- validation report
- audit outputs

## Machine Certification

Per-TF machine policy lives in `contracts/deploy_profiles.json`.
Current operational split:
- `1w`: CPU-first trimmed lane
- `1d`: CPU-first high-RAM lane
- `4h`: hybrid lane
- `1h`: GPU lane
- `15m`: same-machine GPU lane

Live under-`$6/hr` candidate research belongs in dated status docs, not in stale ETA notes.

## Operational Rule

If behavior, docs, or ad hoc terminal lore disagree:
1. contracts
2. maintained code
3. dated status docs
4. historical notes

That order wins.
