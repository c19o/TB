# Session Resume - 4H Lane

Date: 2026-04-02
Branch: `lane/4h`
Folder: `lanes/4h`

## Read This First

If this lane is being resumed after a context reset:

1. Read this file fully.
2. Read `v3.3/LANE_SESSION_START.md`.
3. Read `v3.3/CLOUD_4H_PROFILE.md`.
4. Read the latest dated status docs under `v3.3/docs/status/`.

## Lane Purpose

- `4h` is the transition lane.
- It is the first serious hybrid lane between CPU-first and GPU-dominant behavior.
- This lane should certify the maintained hybrid path, not drift into ad hoc mixed logic.

## Current Truth

- `4h` is expected to magnify sparse transport and retrain backend issues more than `1d`.
- The private-shop lane structure is in place, but `4h` backend certification is still pending.
- This lane should be the first place where hybrid retrain becomes explicit and measurable.

## What Is Authoritative

- `AGENTS.md`
- `v3.3/CODEX.md`
- `v3.3/CLAUDE.md`
- `v3.3/CONVENTIONS.md`
- `v3.3/PRIVATE_SHOP_OPERATING_MODEL.md`
- `v3.3/CLOUD_4H_PROFILE.md`
- `v3.3/contracts/pipeline_contract.json`
- `v3.3/contracts/deploy_profiles.json`
- `v3.3/contracts/private_shop_controls.json`

## Current Focus

- Certify `4h` as the first serious hybrid lane.
- Keep CPU/GPU boundary decisions explicit.
- Use `4h` to prove the bridge between high-RAM CPU work and meaningful GPU acceleration.

## Next Sensible Steps

1. Keep shared infra changes on `private-shop-core`.
2. Use `4h` to test hybrid backend policy and memory envelopes.
3. Update the dated machine matrix when a better under-`$6/hr` `4h` box is validated.

## Resume Guardrails

- Do not quietly inherit `1d` CPU-only assumptions.
- Do not assume `1h/15m` GPU rules automatically apply here.
- Keep `4h` as its own certified lane, not a hand-wavy middle ground.
