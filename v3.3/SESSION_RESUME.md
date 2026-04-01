# V3.3 Session Resume - 2026-04-01

## Instruction To New Session
Read this file completely. Then read `v3.3/CLAUDE.md`. Resume from "Next Steps".

---

## Current Timeframe Status

### 1w
- Status: COMPLETE
- Validation: PASS
- Training: PASS (all steps)
- Metrics: CPCV AUC 57.5%, model AUC 79.3%
- Artifacts: `v3.3/1w_cloud_artifacts_v3/`
- Latest smoke test: `smoke_test_pipeline.py --tf 1w` passed 10/10 after latest parity fix update.

### 1d
- Status: BLOCKED (partial progress)
- Step 2 cross-gen V4:
  - First two cross steps run (dx, ax)
  - Later steps still blocked due to remaining cross-supervisor issues
- Known blocker details:
  - `gpu_daemon.py` RELOAD fixes were applied (3 fixes complete)
  - Remaining issues are in `cross_supervisor.py`:
    - return value mismatch (2 vs 3)
    - `_reload_csc_to_gpu` memory leak

### 4h
- Status: NOT STARTED for V3.3 full training
- Dependency: waits on 1d cross-gen path stabilization

### 1h
- Status: NOT STARTED for V3.3 full training
- Dependency: waits on 1d cross-gen path stabilization

### 15m
- Status: NOT STARTED for V3.3 full training
- Dependency: waits on 1d cross-gen path stabilization
- Historical context: high OOM risk on GPU cross-gen; CPU fallback likely required

---

## Active Cloud Machines

### Machine A
- ID: 33876301
- Provider/Region: Vast.ai Sichuan
- Specs: 8x RTX 3090 (24 GB each)
- Cost: $1.12/hr
- Purpose: V3.3 multi-GPU training and cross-gen
- Status: PAUSED
- Notes: Existing deploy/symlink flow already established

### Candidate Machines (not active)
- ID 33924989, CPU score 453, fastest option
- ID 33862959, CPU score 240, middle option
- ID 28736882, CPU score 237, lowest-cost option

---

## Blocking Issues
1. 1d cross-gen step 3+ still blocked by remaining `cross_supervisor.py` defects.
2. Downstream 4h/1h/15m full runs are queued behind 1d stabilization.

Non-blocking resolved items:
- LightGBM import failure resolved.
- Convention gate violations resolved to zero.
- `validate.py` now passes 96/96 (2 warnings).

---

## Recent Completions (2026-04-01)
1. SAV-15 feature package finalized, including SAV-32 config variables.
2. QA verification complete: validation PASS and convention gate ALL PASS.
3. `gpu_daemon.py` RELOAD path patched with 3 concrete fixes.
4. `PARAMETER_GUIDE.md` completed.
5. KB gap analysis completed with 4 missing-paper links captured.
6. `ops_kb.py` Unicode output hardening applied to prevent Windows cp1252 crash during list/smart output.
7. Documentation Lead DoD checks executed: `import ops_kb` PASS, `validate.py` PASS (96/96), `smoke_test_pipeline.py --tf 1w` FAIL due cuDF/`ALLOW_CPU` environment requirement.
8. SAV-8 parity fix landed in `feature_library.py`: added `px_pc213_x_rsi_os` and `px_pc213_x_macd_high`.
9. Follow-up `smoke_test_pipeline.py --tf 1w` passed 10/10.
10. Paperclip SAV-38 heartbeat sync complete: KB gap download list remains current (4 papers), no new `kb_gap` deltas; task moved to BLOCKED pending user PDF download into Orgonite `drop_here/` and ingest via SAV-9.

---

## Matrix Thesis Context (Do Not Dilute)
System relies on 2.9M+ sparse binary features, including rare esoteric signals
(gematria, numerology, astrology, space weather, calendar/time-cycle effects)
that may fire only a few times per year. These rare signals are intentional edge
features and must remain protected.

---

## Next Steps
1. Resolve remaining `cross_supervisor.py` issues (return mismatch and reload leak).
2. Resume 1d Step 2 cross-gen from daemon path and confirm no fallback to legacy OOM path.
3. When 1d cross-gen is stable, execute 1d Steps 3-7 and capture actual timings.
4. Advance full pipeline in order: 4h, then 1h, then 15m.
5. After every step completion or machine change, update `ETA_CHART.md` and log ops_kb entry.
