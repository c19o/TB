# Local Pre-1W Issues

Status: in progress

This file is the local visible replacement for "what is still blocking before a fresh 1w retrain?"

## Buckets

### Must Fix Before 1W
- `SAV-53` `OMP_NUM_THREADS=4` cross-gen tuning
  - currently flagged as owner-approval runtime/performance change before production rollout
- `SAV-40` observability/watchdog gap in runtime path
  - requires owner review and approval before considering production usage
- Local `1w` smoke/retrain with artifact capture and validation gates
  - this remains the execution checkpoint for readiness, as captured on the visible board

### Can Fix After 1W
- `SAV-3` Train 1D timeframe on cloud
- `SAV-5` Train 4H timeframe
- `SAV-6` Train 1H timeframe
- `SAV-7` Train 15M timeframe
- `SAV-16` Regime CPCV + decomposed targets + LSTM meta-stacker
- `SAV-29` Implement model rollback + automatic diagnosis on bad results

### Owner Approval Required
- `SAV-53` (runtime/threading throughput tuning)
- `SAV-40` (watchdog/observability behavior)

## Inputs
- `ISSUE_STATUS_AUDIT.md`
- `PRE_1W_ISSUE_BOARD.md`
- `PRODUCTION_READINESS.md`
- `SESSION_RESUME.md`
- `VALIDATION_WARNINGS.md`
- daemon and QA audit docs

## Notes
- This board should track the local truth, not stale external status labels.
- Issues that block lower TFs but do not block a fresh `1w` retrain should be called out explicitly.
