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
- Latest smoke test: default path still fails without `ALLOW_CPU=1` on local CUDA13+ due cuDF gate; latest gated run passed 10/10 (`smoke_test_1w.json`, total_time=5.4s).

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
- Latest smoke test: `smoke_test_15m.json` passed 10/10 with `ALLOW_CPU=1` (`total_time=8.9s`).

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
3. CUDA13+ local release path mismatch: default smoke path fails unless `ALLOW_CPU=1` is explicitly set (document/enforce fallback behavior in release flow).
4. Documentation governance gap: `v3.3/PARAMETER_GUIDE.md` still states `validate.py` has 74 checks; current reality is 96/96. Reopened in Paperclip for correction.
5. Paperclip governance readiness gaps: no automated KB-evidence gate, no enforced auto-failover between mirror agents, and workflow status can remain `todo` while execution runs are active.

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
11. SAV-38 KB-first compliance probe (retry-after-skill-sync) completed via KB-only path: three Orgonite queries returned sufficient results, `KB_QUERY` and `KB_SOURCE` were logged to ops_kb, and tracker issue returned to TODO for ongoing sweeps. Probe bug logged: hyphenated query text can trigger SQLite `no such column` in `kb.py smart`.
12. Documentation Lead DoD rerun: `python -c "import ops_kb"` PASS, `validate.py` PASS (96/96, 2 warnings), `smoke_test_pipeline.py --tf 1w` FAIL due cuDF unavailable on CUDA13 path when `ALLOW_CPU` is not set.
13. SAV-38 forced fallback smoke test completed: ran 3 KB-first phrasings for off-domain local-services query, classified KB output as weak/irrelevant, logged `KB_QUERY` + `KB_GAP` + `PERPLEXITY_SOURCE` (Task=`SAV-38-PERPLEXITY`), captured nearby result for Fairfield Bay (`501 Pressure Washing`), then returned SAV-38 to BLOCKED recurring-tracker state.
14. Post-fallback DoD rerun completed: `import ops_kb` PASS, `validate.py` PASS (96/96, 2 warnings), `smoke_test_pipeline.py --tf 1w` FAIL again on cuDF CUDA13 gate unless `ALLOW_CPU=1`.
15. SAV-38 repo probe completed (read-only environment test): `git rev-parse --show-toplevel` confirms repo root is parent `Savage22 Server` directory (not `v3.3`), `git remote -v` confirms `tb33` remote exists (`https://github.com/c19o/TB-3.3.git`), and issue returned to BLOCKED recurring-tracker state.
16. Post-repo-probe DoD rerun completed: `import ops_kb` PASS, `validate.py` PASS (96/96, 2 warnings), `smoke_test_pipeline.py --tf 1w` FAIL again with same cuDF CUDA13 gate (`ALLOW_CPU=1` needed for fallback).
17. New `smoke_test_1w.json` artifact generated at 2026-04-01 12:08 (local): smoke test summary shows PASS (`passed=true`, 10/10 steps), `total_time=5.5s`, and no errors.
18. Documentation Lead DoD rerun after doc refresh: `python -c "import ops_kb"` PASS, `validate.py` PASS (96/96, 2 warnings), `smoke_test_pipeline.py --tf 1w` FAIL at feature build with same cuDF CUDA13 gate unless `ALLOW_CPU=1` is set.
19. SAV-4 follow-up (GPU/RAM) logged in ops_kb: `v2_cross_generator.py` daemon dispatch now accepts both 2-value and 3-value `run_cross_step` return contracts; tuple-unpack mismatch no longer forces legacy fallback path (commit `62f25d4`).
20. SAV-43 production-readiness audit completed: convention gate PASS, `validate.py` PASS (96/96, 2 warnings), 1w smoke FAIL without `ALLOW_CPU=1` but PASS with it; 15m smoke PASS with `ALLOW_CPU=1`; CUDA13 fallback mismatch logged as deployment blocker.
21. GPU/RAM safety defaults enforced in `v2_cross_generator.py`: `OMP_NUM_THREADS` default set to 4 and `RIGHT_CHUNK` default fixed at 500 (env override retained) to prevent RAM-driven oversized chunks and non-1w OOM risk.
22. New smoke artifacts observed after SAV-43/GPU-RAM updates: `smoke_test_1w.json` PASS (`total_time=6.5s`) and `smoke_test_15m.json` PASS (`total_time=8.9s`), both aligned with `ALLOW_CPU=1` fallback context on local CUDA13 environment.
23. Documentation Lead DoD rerun after SAV-43/GPU-RAM doc sync: `python -c "import ops_kb"` PASS, `validate.py` PASS (96/96, 2 warnings), and `smoke_test_pipeline.py --tf 1w` FAIL again on CUDA13 cuDF gate without `ALLOW_CPU=1`.
24. `gpu_daemon.py` CUDA sparse-and-batch path patched for NNZ safety (ops_kb ID 64): CSC `indptr` now preserved as int64 end-to-end (`cp.int64`, kernel pointer/loop variables migrated to `long long`) to avoid >2^31 overflow risk.
25. New `smoke_test_1w.json` artifact after daemon int64 patch: PASS 10/10 (`total_time=5.4s`) with `ALLOW_CPU=1` fallback context on local CUDA13 environment.
26. Documentation Lead DoD rerun after daemon-int64 documentation sync: `python -c "import ops_kb"` PASS, `validate.py` PASS (96/96, 2 warnings), and `smoke_test_pipeline.py --tf 1w` FAIL again on default CUDA13 cuDF gate without `ALLOW_CPU=1`.
27. SAV-50 re-audit completed: reviewed SAV-27/SAV-28/SAV-38/SAV-42 against current repo and issue state, found stale governance claim in `PARAMETER_GUIDE.md` (`74 checks`), and reopened SAV-28 for correction; SAV-42 remains TODO (governance audit still pending).
28. Post-SAV-50 DoD rerun: `python -c "import ops_kb"` PASS, `validate.py` PASS (96/96, 2 warnings), and `smoke_test_pipeline.py --tf 1w` FAIL on the same CUDA13 cuDF gate unless `ALLOW_CPU=1` is set.
29. SAV-42 production-readiness governance audit completed (Paperclip company): strengths confirmed (skills attached to all agents, run-linked checkout traces), and major gaps documented (mirror failover not automatic, KB-first evidence not system-enforced, status/execution mismatch on active tasks). Recommended controls posted to issue.
30. Post-SAV-42 DoD rerun: `python -c "import ops_kb"` PASS, `validate.py` PASS (96/96, 2 warnings), and `smoke_test_pipeline.py --tf 1w` FAIL on the known CUDA13 cuDF gate unless `ALLOW_CPU=1` is set.
31. SAV-27 unblock path created without ownership violation: opened [SAV-62] (Discord/Paperclip status parity implementation task), assigned to Chief Engineer [Codex], and left SAV-27 blocked pending code-owner implementation plus verification.

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
6. Track closure of governance controls from SAV-42 (KB-evidence gate, mirror failover policy, workflow status consistency) before relying on autonomous multi-agent execution for production-critical training steps.
