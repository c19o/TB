# Local GSD Board

## Status
- execution model: local Codex + GSD + git worktrees
- visibility model: direct
- company structure: mirrored from the prior company layout, but locally run
- max concurrent worker target: 20 (configurable in `C:\Users\C\.codex\config.toml`)

## Active Lanes

### Chief
- role: orchestrator and integrator
- status: active
- goal: keep project execution in visible local GSD flow
- current lanes: worktree routing, issue reconciliation, cloud launch prep

### Daemon Lane
- roles:
  - daemon
  - runtime ipc
  - supervisor contract
  - gpu
  - qa
- objective: determine whether `SAV-4` / `SAV-12` are actually fixed and what remains
- output required:
  - blocker list
  - exact ownership
  - verified next actions
- current issue cluster:
  - `SAV-4`
  - `SAV-44`
  - `SAV-63`
  - `SAV-64`
  - `SAV-68`
  - `SAV-69`

### 1w Readiness Lane
- roles:
  - ml
  - qa
  - devops
- objective: decide whether `1w` is safe to retrain now
- output required:
  - safe now vs must-fix first
  - speed-affecting changes needing owner approval
  - environment/runtime gaps
- current launch contract:
  - `CLOUD_TARGET_MACHINE.md`
  - `CLOUD_1W_LAUNCH_CONTRACT.md`

### Governance Lane
- roles:
  - qa
  - docs
  - matrix
- objective: eliminate prompt drift and enforce KB-first behavior
- output required:
  - prompt/policy gaps
  - ops evidence gaps
  - minimum autonomy fixes

### Audit Lane
- roles:
  - docs
  - qa
  - matrix
- objective: reconcile historical company issues with current code truth
- output required:
  - fixed locally vs still blocked
  - future backlog vs external dependency
  - stale-doc cleanup

## Owner Escalation Only
- cloud machine rent/destroy
- speed-affecting production changes
- matrix-thesis / protected-feature policy changes
- major rollback decisions

## Notes
- Historical issue history may still be referenced as prior evidence.
- New execution should default to local GSD workstreams and visible Codex coordination.
- The approved cloud path is Washington `Type 33923286`; do not rent without owner approval.
