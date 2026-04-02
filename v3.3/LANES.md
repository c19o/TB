# Timeframe Lanes

Date: 2026-04-02

The maintained private-shop layout uses one shared core branch plus one git worktree lane per timeframe.

## Branch Model

- Shared core: `private-shop-core`
- Timeframe lanes:
  - `lane/1w`
  - `lane/1d`
  - `lane/4h`
  - `lane/1h`
  - `lane/15m`

Rule:
- shared framework changes land on `private-shop-core` first
- timeframe-specific behavior changes land on the matching `lane/*` branch

## Folder Layout

Repo-root worktree folders:
- `lanes/1w`
- `lanes/1d`
- `lanes/4h`
- `lanes/1h`
- `lanes/15m`

These folders exist so one terminal can stay open in each timeframe lane without polluting the main checkout.

## Operating Rules

- Do not expect dirty changes in the main checkout to appear in the lanes automatically.
- Create or update lane branches from committed refs only.
- If a shared change should reach all lanes, merge or cherry-pick it from `private-shop-core`.
- Lane worktrees are ignored by the main repo to keep `git status` readable.

## Scripts

- Create lanes:
  - `powershell -File scripts/create_lane_worktrees.ps1 -CreateBaseBranch`
- Remove lanes:
  - `powershell -File scripts/remove_lane_worktrees.ps1`

## Recommended Sequence

1. Commit shared framework changes on `private-shop-core`.
2. Create the lane worktrees.
3. Open one Codex terminal in each `lanes/<tf>` folder.
4. Read `v3.3/LANE_SESSION_START.md` in the lane terminal.
5. Read the matching `v3.3/SESSION_RESUME_<TF>_LANE.md` file.
6. Keep local runtime data outside the repo under `SAVAGE22_RUNTIME_HOME`.
