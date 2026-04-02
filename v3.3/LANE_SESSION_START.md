# Lane Session Start

Date: 2026-04-02

Use this file when opening a fresh Codex terminal in any lane worktree.

## Open The Right Folder

- `lanes/1w` -> branch `lane/1w`
- `lanes/1d` -> branch `lane/1d`
- `lanes/4h` -> branch `lane/4h`
- `lanes/1h` -> branch `lane/1h`
- `lanes/15m` -> branch `lane/15m`

Do shared architecture work on `private-shop-core` first. Merge or cherry-pick shared fixes into the lanes.

## Read In This Order

1. Repo-root `AGENTS.md`
2. `v3.3/CODEX.md`
3. `v3.3/CLAUDE.md`
4. `v3.3/CONVENTIONS.md`
5. `v3.3/PRIVATE_SHOP_OPERATING_MODEL.md`
6. `v3.3/LANES.md`
7. The timeframe profile for the lane:
   - `v3.3/CLOUD_1W_PROFILE.md`
   - `v3.3/CLOUD_1D_PROFILE.md`
   - `v3.3/CLOUD_4H_PROFILE.md`
   - `v3.3/CLOUD_1H_PROFILE.md`
   - `v3.3/CLOUD_15M_PROFILE.md`
8. The latest dated status docs under `v3.3/docs/status/`

## Research Stack

Use this order for non-trivial work:

1. Olson KB / local Orgonite Master KB
2. Socraticode
3. Perplexity
4. Generic recall only as last resort

If Olson MCP is unavailable, use the local KB CLI:

```powershell
python "C:\Users\C\Desktop\MY GOOGLE DRIVE\Orgonite master\kb.py" smart "<query>" --limit 5 --json-output --group-by-book
```

If Perplexity is needed, follow the rules in:
- `AGENTS.md`
- `v3.3/skills/perplexity-guide/SKILL.md`

## Private-Shop Rules

- The repo is source-only. Runtime data belongs under `SAVAGE22_RUNTIME_HOME`.
- Heavy phases use certified backends only.
- No silent fallback from a certified parallel mode to sequential.
- Resume must be provenance-safe.
- Promotion depends on evidence, not just successful exit codes.

Private-shop authority lives in:
- `v3.3/PRIVATE_SHOP_OPERATING_MODEL.md`
- `v3.3/contracts/private_shop_controls.json`
- `v3.3/contracts/deploy_profiles.json`
- `v3.3/contracts/pipeline_contract.json`

## Runtime Home

Default local runtime home:

- `C:\Users\C\Documents\Savage22 Runtime`

Expected subdirs:
- `shared_db/`
- `runs/`
- `artifacts/`
- `logs/`
- `archives/`
- `downloads/`
- `cache/`

## Git And GitHub

- Shared branch: `private-shop-core`
- Lane branches:
  - `lane/1w`
  - `lane/1d`
  - `lane/4h`
  - `lane/1h`
  - `lane/15m`

Rules:
- commit shared infra to `private-shop-core`
- keep timeframe-specific changes on the matching lane branch
- do not assume dirty main-checkout files appear in the lanes

## Timeframe Intent

- `1w`: trimmed CPU-first lane
- `1d`: CPU-first high-RAM lane
- `4h`: hybrid lane
- `1h`: GPU lane
- `15m`: same-machine GPU lane

The per-lane machine policy is documented in the matching `CLOUD_<tf>_PROFILE.md` file and in the dated Vast matrix under `v3.3/docs/status/`.

## Current Dated Status Docs

Start with these:
- `v3.3/docs/status/2026-04-02_private_shop_upgrade_plan.md`
- `v3.3/docs/status/2026-04-02_vast_under_6usd_matrix.md`
- `v3.3/docs/status/2026-04-02_datacenter_gpu_feasibility.md`

## If You Are Starting Real Work

1. Confirm branch with `git branch --show-current`
2. Confirm the lane path is correct
3. Read the TF profile doc
4. Run the required KB pass before non-trivial changes
5. Keep edits scoped to the lane unless the change is shared infra
