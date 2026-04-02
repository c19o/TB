# Session Resume - 1W Lane

Date: 2026-04-02
Branch: `lane/1w`
Folder: `lanes/1w`

## Read This First

If this lane is being resumed after a context reset:

1. Read this file fully.
2. Read `v3.3/LANE_SESSION_START.md`.
3. Read `v3.3/CLOUD_1W_PROFILE.md`.
4. Read the latest dated status docs under `v3.3/docs/status/`.

## Lane Purpose

- Keep `1w` as the trimmed cheap validation lane.
- Preserve the maintained weekly contract.
- Do not let weekly inherit lower-TF cross complexity.

## Current Truth

- `1w` is CPU-first and trimmed.
- `1w` cross policy is forbidden in the maintained path.
- `1w` is the cheapest lane for framework validation and deployment smoke testing.
- The maintained private-shop framework, lane structure, and session bootstrap docs are in place.

## What Is Authoritative

- `AGENTS.md`
- `v3.3/CODEX.md`
- `v3.3/CLAUDE.md`
- `v3.3/CONVENTIONS.md`
- `v3.3/PRIVATE_SHOP_OPERATING_MODEL.md`
- `v3.3/CLOUD_1W_PROFILE.md`
- `v3.3/contracts/pipeline_contract.json`
- `v3.3/contracts/deploy_profiles.json`
- `v3.3/contracts/private_shop_controls.json`

## Current Focus

- Use `1w` as the proof lane for shared deploy/release/run-control changes.
- Keep weekly fast, reliable, and easy to rerun.
- Do not add lower-TF-only complexity here unless it is shared infrastructure.

## Known Good State

- The maintained lane/worktree structure exists.
- `1w` already has the private-shop docs and lane bootstrap flow.
- Shared infra changes should land on `private-shop-core` first and then be merged here.

## Next Sensible Steps

1. Use `1w` to validate shared deployment or contract changes.
2. Keep the lane clean and trimmed.
3. Push TF-specific weekly behavior only if it is truly weekly-specific.

## Resume Guardrails

- Do not reintroduce cross generation into maintained `1w`.
- Do not treat repo-root runtime artifacts as valid run state.
- Do not make weekly the dumping ground for lower-TF experiments.
