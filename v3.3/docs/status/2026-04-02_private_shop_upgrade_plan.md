# 2026-04-02 Private-Shop Upgrade Plan

## What Changed

- Preserved a small forensic pull from the failed Norway `1d` run:
  - local path: `C:\Users\C\Documents\Savage22 Runtime\downloads\1d-34034370-20260402T203704Z_small_pull`
  - preserved logs, sentinels, heartbeat, release manifest, Optuna outputs, features, and cross-name metadata
  - intentionally did not pull the `17.35 GB` `v2_crosses_BTC_1d.npz`
- Destroyed Vast instance `34034370` after the download completed
- Switched local path defaults to the runtime home so ad hoc local runs stop targeting the repo tree
- Added private-shop operating authority:
  - `PRIVATE_SHOP_OPERATING_MODEL.md`
  - `contracts/private_shop_controls.json`
  - `private_shop_contracts.py`
- Added lane/worktree tooling:
  - `scripts/create_lane_worktrees.ps1`
  - `scripts/remove_lane_worktrees.ps1`
  - `LANES.md`
- Extended framework docs to reference:
  - private-shop controls
  - lane branches/worktrees
  - runtime-home-first local behavior
- Added the dated live-Vast machine matrix:
  - `docs/status/2026-04-02_vast_under_6usd_matrix.md`

## What Is Authoritative Now

- `CLOUD_DEPLOYMENT_FRAMEWORK.md`
- `PRIVATE_SHOP_OPERATING_MODEL.md`
- `contracts/pipeline_contract.json`
- `contracts/deploy_profiles.json`
- `contracts/private_shop_controls.json`
- `path_contract.py`
- `LANES.md`

## Branches And Commits

- shared core branch: `private-shop-core`
- lane branches:
  - `lane/1w`
  - `lane/1d`
  - `lane/4h`
  - `lane/1h`
  - `lane/15m`
- defining commits:
  - `701f348` `Establish private-shop core and timeframe lanes`
  - `bc12603` `Fix lane worktree PowerShell helpers`

## What Remains Legacy

- ad hoc local runs that assume the repo is also the artifact root
- historical ETA notes and training notes that are not aligned with the current contracts
- provider-specific launch scripts that are not thin wrappers over the maintained deploy engine

## Private-Shop Focus Areas Added

- per-timeframe backend certification
- rare-feature health reporting
- training/inference parity certification
- post-train model governance states

## Known Gaps

- The repo is still not clean. `audit_runtime_home.py` previously reported `277` runtime artifacts/build/archive/log items still living in the repo tree.
- The new private-shop controls are authoritative scaffolding, not yet full automatic artifact emitters for every TF.
- Lane worktrees still need to be created from a committed `private-shop-core` snapshot.
- Lower-TF retrain backend certification is still a live engineering task, especially `1d step5`.

## Next Steps

1. Commit the shared core on `private-shop-core`
2. Materialize `lanes/1w`, `lanes/1d`, `lanes/4h`, `lanes/1h`, `lanes/15m`
3. Move remaining repo clutter into `Savage22 Runtime`
4. Certify `1d step5` backend transport
5. Implement automatic governance / parity / rare-feature artifacts per run
