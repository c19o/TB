# Session Resume - 15M Lane

Date: 2026-04-02
Branch: `lane/15m`
Folder: `lanes/15m`

## Read This First

If this lane is being resumed after a context reset:

1. Read this file fully.
2. Read `v3.3/LANE_SESSION_START.md`.
3. Read `v3.3/CLOUD_15M_PROFILE.md`.
4. Read the latest dated status docs under `v3.3/docs/status/`.

## Lane Purpose

- `15m` is the strict same-machine GPU lane.
- This is the hardest runtime lane and the least tolerant of sloppy transport, checkpoint, or memory behavior.
- Work here should assume private-shop discipline from the start.

## Current Truth

- `15m` is where the current architecture will be stressed the most.
- Machine choice, GPU path quality, and same-machine checkpoint policy matter materially here.
- The lane structure is ready, but `15m` certification is still future work.

## What Is Authoritative

- `AGENTS.md`
- `v3.3/CODEX.md`
- `v3.3/CLAUDE.md`
- `v3.3/CONVENTIONS.md`
- `v3.3/PRIVATE_SHOP_OPERATING_MODEL.md`
- `v3.3/CLOUD_15M_PROFILE.md`
- `v3.3/contracts/pipeline_contract.json`
- `v3.3/contracts/deploy_profiles.json`
- `v3.3/contracts/private_shop_controls.json`

## Current Focus

- Keep `15m` as the strict same-machine GPU certification lane.
- Make checkpointing, observability, and resume truly trustworthy.
- Treat RAM, disk, and GPU transport as first-class engineering concerns.

## Next Sensible Steps

1. Do not start with cheap hacks here.
2. Certify backend, machine class, and checkpoint policy together.
3. Update dated status docs whenever `15m` assumptions materially change.

## Resume Guardrails

- Do not weaken the same-machine rule casually.
- Do not assume `1h` solutions scale to `15m` without evidence.
- Do not hide artifact or memory shortcuts here; they will surface later as production failures.
