# Lower-Timeframe Research & Audit (1d/4h/1h/15m)

Scope: `1d`, `4h`, `1h`, `15m` best-practice review for sequencing, daemon path, and scaling.  
Last updated: 2026-04-01.

## Source Order Used

1. Orgonite KB (repo-local) was checked first:
   - `python kb.py smart "lunar cycles trading timeframe" -n 3`
   - `python kb.py search "Schumann resonance market" -n 5`
   - `python kb.py smart "astro trading frequency" -n 3`
   - `python kb.py smart "day of year intraday market cycles" -n 3`

Result summary:
- KB gives lunar-cycle evidence (mostly `lunar cycles stock returns.pdf` with full/new moon framing) but does not provide precise implementation guidance by 1d/4h/1h/15m windowing, CPCV, or hyperparameter defaults.
- No Schumann-resonance implementation guidance returned from KB search.
- Therefore repo truth is used for sequencing/daemon/scaling decisions.

## Sequencing Check

- Canonical hierarchy in code is explicit:
  - `TF_HIERARCHY = ['1w', '1d', '4h', '1h', '15m']` (`v3.3/config.py`).
- Optuna warm-start mapping is explicit and aligned:
  - `1d <- 1w`, `4h <- 1d`, `1h <- 4h`, `15m <- 1h` (`v3.3/run_optuna_local.py`).
- `cloud_run_tf.py` uses runtime sequence `_TF_SEQUENCE = ['1w', '1d', '4h', '1h', '15m']` and stages background feature prefetch for the next TF only.
- `pipeline_orchestrator.py` confirms the same ordered assembly logic in `run_assembly_line()`.

Verdict: sequence is currently correct.

## Daemon Path: Status by Component

Current path contracts are internally consistent:
- `cross_supervisor.py` publishes the IPC protocol as plain tuples:
  - `RELOAD -> ('RELOAD', left_npy, right_npy, n_left_cols)`
  - `BATCH -> ('BATCH', batch_id, pairs, out_path, pair_id_offset)`
  - responses: `('READY', gpu_id)` / `('RESULT', batch_id, idx_path, total_nnz, status, error_msg)`
- `gpu_daemon.py` supports the same tuple protocol and sends matching `READY`/`RESULT`.
- Daemon main paths explicitly avoid scipy and build CSC from in-memory binary arrays before GPU intersection.
- Supervisor uses spawn-based child processes and round-robin dispatch with `multiprocessing.connection.wait()`.

Operationally:
- For `15m`, `cloud_run_tf.py` sets `V2_RIGHT_CHUNK=500` and `V2_BATCH_MAX=500`.
- `v2_cross_generator.py` already has per-step checkpointing and daemon-based batch dispatch through `run_cross_step`.

Risk status:
- `v3.3/DAEMON_RELOAD_AUDIT.md` remains consistent with code status: contract appears fixed, but 1d+ evidence is still proof-pending (no full rerun log sequence showing `RELOAD->READY->RESULT` end-to-end for each downstream TF).

## Scaling and Resource Posture

- Config scaling is coherent with row/feature growth:
  - `TF_CPCV_GROUPS`: 1d `(5,2)`, 4h `(10,2)`, 1h `(10,2)`, 15m `(10,2)`.
  - `TF_MIN_DATA_IN_LEAF`: 1d/4h/1h/15m all 8 (recently unified floor).
  - `TF_NUM_LEAVES`: 1d 15, 4h 31, 1h 63, 15m 127.
  - 15m-only force-row-wise boosting path (`TF_FORCE_ROW_WISE = {'15m'}`), matching 15m depth.
- Cloud RAM gates are explicit and increasing with workload:
  - 1d 128GB, 4h 256GB, 1h 512GB, 15m 768GB (`v3.3/cloud_run_tf.py`).
- Optuna sampling is reduced by TF:
  - 1d 1.0, 4h 1.0, 1h 0.5, 15m 0.25 (`config.py`).
- `TF_CPCV` sampling controls are present (`CPCV_SAMPLE_PATHS`, `CPCV_OPTUNA_SAMPLE_PATHS`) and can cap path explosion.
- Cross generation scaling is already tuned for memory by TF:
  - `USE_MONTH_INSTEAD_OF_DOY = {'1w', '1d'}` in `v2_cross_generator.py`.
  - `MIN_CO_OCCURRENCE` default still configurable and lower than early versions.

## Mismatch Between Docs and Repo Truth

- `TRAINING_PLAN.md` and some TF-specific guides still describe outdated parallel assumptions and older CPCV/param numbers (for example 4h `(6,2)` references, older `num_leaves/min_data_in_leaf`, old warm-start notes).
- `TRAINING_15M.md` still documents:
  - rows as ~227K and DOY min_nonzero assumptions that predate current `v2_cross_generator` behavior in some spots.
  - `TF_CPCV_GROUPS` for 15m shown as `(6,2)` in multiple lines, while code now uses `(10,2)`.
- `TRAINING_4H.md` shows stale 4h CPCV `(6,2)` and larger min_data assumptions.

Action: these docs should be treated as historical context only until refreshed to current code truth.

## Per-Timeframe Accuracy / Speed Ideas

### 1d
- Accuracy: keep warm-start from 1w as implemented, but enforce hard checks on holdout directional class recall before any post-1w promotion. Use CPCV/OOS artifacts plus confusion split by class for LONG/SHORT vs HOLD drift.
- Speed: validate whether full-row Optuna (1.0 subsample) still outperforms 0.75/0.5 in this data regime; if search wall-time is the limiter, use a controlled A/B with lower `OPTUNA_TF_PHASE1_TRIALS` and fixed seeds only for smoke verification, then return to 25 for production.

### 4h
- Accuracy: ensure 4h uses DOY-based contexts (not month windows) via `v2_cross_generator` TF routing and confirm monthly-cycle-only contamination is not unintentionally introduced.
- Speed: pre-existing cross generation and CPCV are CPU-heavy at 23K rows. Keep `save_binary`/checkpoint behavior, and use existing NUMA/thread binding path in `cloud_run_tf.py` rather than ad-hoc manual overrides.

### 1h
- Accuracy: maintain warm-start from 4h and inspect class balance around SHORT under the current 20% Optuna subsample (0.5) to catch any directional collapse before final retrain.
- Speed: keep 1h as the boundary that can reveal throughput bottlenecks. If 1h is slower than expected, fix I/O + daemon dispatch first before touching model hyperparameters; model side currently has only moderate scaling lever room.

### 15m
- Accuracy: keep 15m in sparse + row-wise mode and validate row-wise-only behavior against non-row-wise A/B only in a sandbox (not production), because row-wise is likely needed for rare-signal retention at 15m.
- Speed: 15m is primarily constrained by cross feature IO + matrix growth. Tune in order: `V2_RIGHT_CHUNK`, `V2_BATCH_MAX`, and checkpoint hygiene (`_cross_checkpoint_15m_*`) before changing CPCV widths.

## What Must Wait Until 1w Completes

Do not finalize any production 1d/4h/1h/15m run until the current 1w pass reaches clean completion with:
1. `optuna_configs_1w.json` present and non-empty.
2. `model_1w.json` and 1w pipeline artifacts stable.
3. Daemon reload evidence recorded for 1w path behavior.

Then sequence can proceed:
- 1d can safely warm-start from 1w.
- 4h from 1d.
- 1h from 4h.
- 15m from 1h.

Until that sequence is complete, any 1d+ daemon or accuracy claims remain non-authoritative.
