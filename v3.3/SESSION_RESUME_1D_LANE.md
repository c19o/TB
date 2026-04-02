# Session Resume - 1D Lane

Date: 2026-04-02
Branch: `lane/1d`
Folder: `lanes/1d`

## Read This First

If this lane is being resumed after a context reset:

1. Read this file fully.
2. Read `v3.3/LANE_SESSION_START.md`.
3. Read `v3.3/CLOUD_1D_PROFILE.md`.
4. Read `v3.3/docs/status/2026-04-02_private_shop_upgrade_plan.md`.
5. Read `C:\Users\C\Documents\Savage22 Runtime\downloads\1d-34034370-20260402T203704Z_small_pull\FORENSICS.md`.

## Lane Purpose

- `1d` is the first heavy proof lane.
- It is CPU-first and high-RAM in the current maintained framework.
- This lane should certify a real private-shop-grade retrain backend instead of relying on fragile live fixes.

## Current Truth

- The old Norway `1d` run was preserved only as a small forensic pull.
- The big `v2_crosses_BTC_1d.npz` was intentionally not downloaded.
- There is no practical retrain resume point carried into this lane.
- The lane contains the current code, docs, contracts, and forensic evidence, not live run state.

## What Is Authoritative

- `AGENTS.md`
- `v3.3/CODEX.md`
- `v3.3/CLAUDE.md`
- `v3.3/CONVENTIONS.md`
- `v3.3/PRIVATE_SHOP_OPERATING_MODEL.md`
- `v3.3/CLOUD_1D_PROFILE.md`
- `v3.3/contracts/pipeline_contract.json`
- `v3.3/contracts/deploy_profiles.json`
- `v3.3/contracts/private_shop_controls.json`
- `C:\Users\C\Documents\Savage22 Runtime\downloads\1d-34034370-20260402T203704Z_small_pull\FORENSICS.md`

## Current Focus

- Certify `1d step5` with a maintained backend and transport.
- Keep `1d` CPU-first unless and until a GPU-native retrain path is actually certified.
- Use the forensic pull to avoid repeating the same mistakes blindly.

## Known Problems To Keep In Mind

- Prior `1d step5` attempts exposed fragile retrain transport behavior.
- The forensic bundle preserves logs and metadata, but not the expensive cross matrix cache.
- Any new `1d` proof run should be treated as a fresh run from a code perspective.

## Next Sensible Steps

1. Read the forensic pull before changing `1d step5`.
2. Keep backend selection explicit and fail-fast.
3. Document every meaningful `1d` change in a new dated status file.

## Resume Guardrails

- Do not assume the old Norway run can be resumed.
- Do not hide a sequential fallback inside `1d`.
- Do not optimize `1d` in ways that make `4h/1h/15m` worse without calling that out.
