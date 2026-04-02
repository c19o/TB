# V3.3 Session Resume - 2026-04-01

## Instruction To New Session
Read this file completely. Then read [CODEX.md](/C:/Users/C/Documents/Savage22%20Server/v3.3/CODEX.md), [CONVENTIONS.md](/C:/Users/C/Documents/Savage22%20Server/v3.3/CONVENTIONS.md), and [CLOUD_1W_LAUNCH_CONTRACT.md](/C:/Users/C/Documents/Savage22%20Server/v3.3/CLOUD_1W_LAUNCH_CONTRACT.md). Resume from `Next Steps`.

---

## Current Operating Model

- Native local Codex + GSD control plane is now the active company model.
- Project-local Paperclip / Claude-CEO control residue was removed.
- Use local workstreams, direct shell visibility, and git worktrees where useful.
- Research policy is strict:
  - database / KB first
  - Perplexity fallback only
  - Matrix Thesis preserved
- Any speed-affecting runtime or training change is owner-approved only unless the 2026-04-01 owner override applies: speed-positive, matrix-safe, accuracy-safe changes are pre-approved.

---

## Current Truth

### 1w
- Status: ready for real rerun on the updated path
- Maintained launch authority:
  - `CLOUD_1W_LAUNCH_CONTRACT.md`
  - `TRAINING_1W.md` is historical/local-only and does not govern maintained runs
- Launch protocol updated:
  - immutable remote release dirs under `/workspace/releases/v3.3_<run_id>`
  - per-run state under `/workspace/runs/<run_id>`
  - heartbeat now belongs in the run dir, not `/workspace/cloud_run_1w_heartbeat.json`
  - launcher preflight now hard-fails on missing release/artifact layout instead of trusting a mutable `/workspace/v3.3`
- Maintained `1w` policy:
  - base-features-only
  - step2_crosses is a validated skip, not a generation step
- Root cause of the earlier slowdown is known:
  - final retrain was slowed by conservative scheduling policy
  - Optuna used the machine better than final retrain
  - weak hardware was not the problem
- Current code state:
  - final retrain decision path is explicit and logged
  - policy knobs are in place
  - machine-aware default `auto` policy was implemented
  - default `FINAL_RETRAIN_PARALLEL_MIN_ROWS` is now `512`, not `2000`
  - weekly launch now uses `OPTUNA_SKIP_FINAL_RETRAIN=1` / `--search-only` so Optuna does not do a redundant final retrain before production retrain
  - hot-path training controls are wired in
  - artifact freshness / cache preservation / duplicate upload waste were cleaned up
- Validation state:
  - `validate.py` passes
  - convention gate passes

### 1d
- Status: still proof-pending; no authoritative post-speed-pass run yet
- Important caution:
  - `1d` had prior OOM behavior in earlier experimentation
  - do not force GPU Optuna search here by default
- What is already believed:
  - lower-TF cost will be more sensitive to daemon reuse, RELOAD behavior, and orchestration overhead than weekly
  - `1d` remains the first real proof lane after `1w`

### 4h / 1h / 15m
- Status: code-side speed pass is largely implemented; runtime proof is still pending
- What is already believed:
  - lower timeframes will magnify:
    - fold overhead
    - sparse handoff cost
    - daemon / reload coordination cost
    - checkpoint and orchestration tax
  - `15m` is probably possible on `512 GB RAM` with proper management, but this is not proven until a real run / partial profiling run
  - the best current lower-TF posture is:
    - GPU for cross-gen and final fold work
    - CPU-first Optuna search
    - conservative memory-aware concurrency
    - one machine only, with aggressive cache reuse

---

## Approved Owner Decisions Already Made

1. Move from hidden hardcoded speed rules to explicit machine-aware speed policy: approved
2. Benchmark and use more aggressive parallel final retrain behavior on the approved `2x4090` cloud target if justified: approved
3. Implement what we already know now instead of waiting for a perfect full audit: approved

---

## Cloud Target

- Type ID: `33923286`
- Region: approved cloud target at time of writing
- GPUs: `2x RTX 4090`
- CPU: `AMD EPYC 7B13 64-Core Processor`
- RAM: `516 GB`
- Price: `$0.832/hr` plus bandwidth

Important:
- a cloud machine was rented and used during this session
- the live `1w` run was intentionally paused/stopped
- no active training progress should be assumed

---

## What Was Completed This Session

1. Speed pass implementation across the pipeline
- duplicate Optuna final retrain removed from the weekly/cloud path
- hot-path training controls added to `ml_multi_tf.py`
- slower non-critical analytics moved off the critical path by default
- checkpoint / artifact / freshness handling improved
- explicit large-machine scheduling path added

2. Cross-gen / daemon / sparse-path improvements
- shared-memory `RELOAD_SHM` path added to supervisor + daemon
- repeated HMM overlay work reduced by precomputing sequential sparse fold overlays once
- merged daemon CSR reuse improved
- adaptive `RIGHT_CHUNK`, flush, and checkpoint logic already existed and remains in place

3. Lower-TF / machine-aware policy work
- owner override is documented: speed-positive, matrix-safe, accuracy-safe changes are pre-approved
- `4h`, `1h`, and `15m` Optuna row subsampling was removed; current config is full-row search for all TFs
- lower-TF GPU Optuna search is present only as an opt-in path via `OPTUNA_MULTI_GPU_SEARCH=1`
- default stance is CPU-first search because `1d` OOM history makes GPU search unsafe by default

4. KB / research infrastructure cleanup
- Orgonite KB ingest now uses manual topic tags only; auto-tagging is disabled in `ingest.py`, `ocr_gpu.py`, and `ocr_pdf.py`
- newly added performance docs and repo companions were ingested with correct manual tags
- targeted retag pass cleaned up the obvious technical/performance documents already in the KB

5. Verification
- `python -m py_compile` passed on the touched pipeline files and KB scripts
- `python -X utf8 validate.py` passed (`97/97`)
- `python convention_gate.py research-audit SPEED_IMPL_2026-04-01 --hours 72` passed

---

## Most Important Technical Findings

1. The main speed enemy is underlapped compute
- too much serialized work
- too much memory/setup overhead relative to useful GPU compute
- too many hardcoded scheduling assumptions

2. Weekly final retrain was too conservative
- final retrain had been effectively optimized for caution, not for a strong `2x4090` cloud box
- this is now partially corrected through explicit machine-aware policy

3. Lower timeframes are expected to amplify the same weaknesses
- daemon and RELOAD coordination
- sparse matrix handoff cost
- fold scheduling overhead
- checkpoint/orchestration tax

4. The newly ingested performance books did not reveal a hidden turnkey framework
- they reinforce profiling discipline, experiment design, observability, and capacity planning
- they support CPU-first search + GPU-heavy cross-gen/final work as the current safest lower-TF posture
- they do not eliminate the need for real runs; the remaining decisions are measurement-driven

5. GPU Optuna is possible but not safe-by-default
- explicit GPU-trial reservation / release code was added
- the path is intentionally opt-in (`OPTUNA_MULTI_GPU_SEARCH=1`)
- do not enable it casually on lower TFs until a memory-safe profile is proven

---

## What Is Still Open

1. Real runtime proof
- rerun `1w` on the approved cloud target under the updated path
- collect stage times and confirm final retrain now overlaps the box correctly

2. Lower-TF proof sequence after `1w`
- `1d`
- `4h`
- `1h`
- `15m`

3. Profiling-driven decisions still to be made
- whether `15m` is stable on `512 GB RAM`
- whether `2 TB RAM` is justified by real memory pressure vs orchestration waste
- whether GPU Optuna should ever be enabled for lower TFs

4. Optional documentation cleanup
- some older TF training guides remain stale historical context (`TRAINING_4H.md`, `TRAINING_1H.md`, `TRAINING_15M.md`)

---

## Next Steps

1. Run `1w` on the approved cloud target using the maintained contract.
2. During the run, capture:
- wall time by stage
- final retrain GPU / fold overlap behavior
- any checkpoint / reload / artifact delays
3. Audit the completed `1w` run before touching lower TFs.
4. If `1w` looks clean, move to `1d` as the first lower-TF proof lane.
5. Do not enable `OPTUNA_MULTI_GPU_SEARCH=1` on lower TFs unless there is an explicit memory-safe test plan.

### Immediate Run Intent
- The next session should treat the code-side optimization pass as largely complete.
- The priority is no longer “find another static optimization.”
- The priority is “run `1w`, measure it, and decide from evidence.”

---

## Guardrails

- Database first on every non-trivial issue
- Perplexity fallback only
- Matrix Thesis preserved
- No pruning without owner review and evidence
- Any further speed-affecting change still needs to be surfaced clearly to the owner unless it falls under the 2026-04-01 owner override: speed-positive and matrix-safe changes are pre-approved.
