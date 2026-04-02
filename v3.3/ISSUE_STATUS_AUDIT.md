# Imported Issue Status Audit

Last updated: 2026-04-01

This file reconciles the imported legacy issue snapshot with the current repo state.
The snapshot still shows `31` non-done issues, but several are already fixed locally and only stale in the imported company state.

## Fixed Locally But Still Stale In Snapshot

### `SAV-19`
- Status here: fixed locally
- Evidence:
  - [live_trader.py](/C:/Users/C/Documents/Savage22%20Server/v3.3/live_trader.py)
  - signal handlers and stop-event sleep path are implemented

### `SAV-51`
- Status here: fixed locally
- Evidence:
  - [validate.py](/C:/Users/C/Documents/Savage22%20Server/v3.3/validate.py)
  - `bagging_fraction >= 0.95` enforced in config-level and hardcoded scan checks

### `SAV-53`
- Status here: code already reflects the intended setting
- Evidence:
  - [cloud_run_tf.py](/C:/Users/C/Documents/Savage22%20Server/v3.3/cloud_run_tf.py)
  - `OMP_NUM_THREADS=4` and `NUMBA_NUM_THREADS=4` are already set
- Note:
  - still an owner-approval runtime knob because it affects throughput

### `SAV-54`
- Status here: fixed locally
- Evidence:
  - [validate.py](/C:/Users/C/Documents/Savage22%20Server/v3.3/validate.py)
  - local and cloud validation paths now both enforce `ALLOW_CPU=1` when cuDF is unavailable

### `SAV-56`
- Status here: fixed locally
- Evidence:
  - [validate.py](/C:/Users/C/Documents/Savage22%20Server/v3.3/validate.py)
  - symbol argparse/CLI slash validation added

### `SAV-57`
- Status here: fixed locally
- Evidence:
  - [PARAMETER_GUIDE.md](/C:/Users/C/Documents/Savage22%20Server/v3.3/PARAMETER_GUIDE.md)
  - [PRODUCTION_READINESS.md](/C:/Users/C/Documents/Savage22%20Server/v3.3/PRODUCTION_READINESS.md)
  - agent prompts under [agents](/C:/Users/C/Documents/Savage22%20Server/v3.3/agents)

### `SAV-67`
- Status here: mostly fixed locally
- Evidence:
  - Perplexity-first drift removed from main agent prompts
- Note:
  - still worth one final repo-wide audit pass before calling fully closed

## Does Not Block Fresh 1w But Still Needs Proof Or Cleanup

### `SAV-4`, `SAV-44`, `SAV-63`, `SAV-64`, `SAV-68`, `SAV-69`
- Status here: mostly code-fixed, not fully revalidated end-to-end
- Evidence:
  - daemon/cross-supervisor path looks materially repaired
  - remaining gap is proof on `1d+` rerun rather than an obvious live source-code defect

### `SAV-48`
- Status here: still needs deployment/rollback re-audit synthesis
- Note:
  - important for production hardening, but not a code blocker for fresh `1w`

### `SAV-61`
- Status here: parent meta-issue, not a single fix target
- Note:
  - replaced locally by this visible Codex+GSD execution model

### `SAV-62`
- Status here: still blocked by ownership/tooling boundary
- Note:
  - Discord/external-control parity is not a model-training blocker

## External Dependency / User Dependency

### `SAV-9`
- Status here: still blocked
- Reason:
  - depends on expanding the Orgonite KB with additional technical books / PDFs

### `SAV-38`
- Status here: intentionally blocked recurring tracker
- Reason:
  - ongoing KB-gap tracking, not a bug to "finish" once

## Future Backlog, Not Pre-1w Readiness

### `SAV-3`, `SAV-5`, `SAV-6`, `SAV-7`
- Status here: downstream training backlog
- Reason:
  - these are training execution tasks for lower timeframes, not pre-`1w` blockers

### `SAV-16`
- Status here: downstream modeling lane
- Reason:
  - valuable, but not required before a fresh `1w` run

### `SAV-29`
- Status here: downstream safety/ops hardening
- Reason:
  - this is post-train rollback and diagnosis infrastructure (`deploy_model.py` / `model_diagnosis.py`) and does not block executing a first fresh `1w` retrain

### `SAV-18`
- Status here: cloud deployment/training workflow
- Reason:
  - this is the operational launch lane for cloud training, not a bugfix issue

## Human Reading Of The Backlog

The imported company snapshot is useful history, but it is not an accurate live source of truth anymore.
For immediate readiness:

1. the local code gate is green
2. the local `1w` smoke path is green on the declared `ALLOW_CPU=1` environment
3. the remaining meaningful work is cloud execution, result audit, and daemon proof for lower TFs
