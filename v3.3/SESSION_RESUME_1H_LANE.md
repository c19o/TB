# Session Resume - 1H Lane

Date: 2026-04-02
Branch: `lane/1h`
Folder: `lanes/1h`

## Read This First

If this lane is being resumed after a context reset:

1. Read this file fully.
2. Read `v3.3/LANE_SESSION_START.md`.
3. Read `v3.3/CLOUD_1H_PROFILE.md`.
4. Read the latest dated status docs under `v3.3/docs/status/`.

## Lane Purpose

- `1h` is the first maintained GPU lane.
- This lane should make the GPU path explicit, measurable, and machine-aware.
- `1h` is where stronger multi-GPU boxes start making obvious economic sense.

## Current Truth

- `1h` should not be treated as just a bigger `4h`.
- The intended maintained posture is GPU-forward, with private-shop-grade backend selection.
- The lane structure and docs are in place, but backend certification is still future work.

## What Is Authoritative

- `AGENTS.md`
- `v3.3/CODEX.md`
- `v3.3/CLAUDE.md`
- `v3.3/CONVENTIONS.md`
- `v3.3/PRIVATE_SHOP_OPERATING_MODEL.md`
- `v3.3/CLOUD_1H_PROFILE.md`
- `v3.3/contracts/pipeline_contract.json`
- `v3.3/contracts/deploy_profiles.json`
- `v3.3/contracts/private_shop_controls.json`

## Current Focus

- Certify the maintained GPU retrain path.
- Use machines that match the lane intent instead of forcing CPU rescue logic.
- Keep training/inference parity and governance artifacts first-class.

## Next Sensible Steps

1. Keep `1h` GPU-specific work isolated to this lane unless it is shared infra.
2. Treat machine selection as part of the lane contract, not an afterthought.
3. Update the dated Vast matrix when a better `1h` candidate is proven.

## Resume Guardrails

- Do not let `1h` fall back into an implicit CPU lane.
- Do not treat GPU count as enough; certify the backend path itself.
- Keep same-run provenance and artifact completeness strict.
