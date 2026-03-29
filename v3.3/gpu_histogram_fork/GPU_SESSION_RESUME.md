# V3.3 GPU Histogram Fork — Session Resume

## INSTRUCTION TO NEW SESSION
Read this file completely. It contains the full state of training, cloud machines, and what needs doing next.

## ACTIVE MACHINES (2026-03-29 ~03:55 UTC)

**ALL MACHINES DESTROYED/STOPPED. $0/hr burn.**

**1d: COMPLETE** — Norway m:29579 (destroyed)
- Full pipeline done in 5.6 hours
- All artifacts: `v3.3/1d_COMPLETE.tar.gz` (402MB)

**4h: m:32021 (Belgium)** — STOPPED
- SSH: `ssh -p 28348 -i ~/.ssh/id_ed25519 root@ssh8.vast.ai`
- Cross gen DONE (5.4M features). Artifacts: `v3.3/4h_cross_artifacts.tar.gz` (1.1GB)
- **BLOCKER**: Optuna OOM on 498GB dense matrix. Need fix in run_optuna_local.py.
- 16/16 DBs verified, full 23K row parquet rebuilt.

**1h: m:49722 (Belgium)** — EXITED (host unavailable)
- All fixes deployed. Needs new machine when ready.

## PREVIOUS MACHINES (DESTROYED)
- m:15340 (8x RTX 5090, 755GB, $3.76/hr) — destroyed after cross gen OOM. Artifacts downloaded.

## TRAINING STATUS

| TF | Status | Machine | Accuracy | Notes |
|----|--------|---------|----------|-------|
| 1w | **DONE** | destroyed | 58% CPCV | Artifacts in v3.3/cloud_results_1w/ |
| 1d | **DONE** | Norway (destroyed) | 51.2% CPCV, PBO 0.20 | Full pipeline complete. All artifacts downloaded locally. |
| 4h | **STOPPED (OOM)** | Belgium m:32021 | — | Cross gen DONE (5.4M features, downloaded). Optuna OOM on 498GB dense. Need sparse fix in run_optuna_local.py. |
| 1h | **CROSS GEN** | Belgium m:49722 | — | Cross 1 done, continuing |
| 15m | NOT STARTED | — | — | Need separate machine |

## OOM FIX HISTORY (CRITICAL)

Cross gen OOM'd **4 times** on the 755GB machine. Root cause: unbounded memory accumulation at 3 levels:

1. **`_csr_chunks` in `gpu_batch_cross`** — fixed with `MAX_CHUNKS_IN_RAM` + disk sub-checkpoints
2. **COO lists in `_gpu_cross_chunk`** — fixed with `_flush_coo_to_csr()` every `FLUSH_FEATS` features
3. **`_csr_out` in `_gpu_cross_chunk`** — fixed with `_flush_csr_to_disk()` every `MAX_CSR_IN_RAM` chunks

**Current thresholds:**
- `FLUSH_FEATS = max(5000, min(50000, int(ram_gb * 50)))` — COO→CSR every ~50K features
- `MAX_CSR_IN_RAM = max(2, min(5, int(ram_gb / 300)))` — CSR→disk every 2 chunks on 1TB, 5 on 2TB
- `MAX_CHUNKS_IN_RAM = max(100, ...)` — sub-checkpoint in gpu_batch_cross (outer level)

**Also applied to `_cpu_cross_chunk`** — same periodic COO→CSR flush pattern.

## FIXES APPLIED THIS SESSION

| Fix | Files | Impact |
|-----|-------|--------|
| CuPy Blackwell sm_120 | 6 files (CUPY_COMPILE_WITH_PTX=1) | CuPy works on CUDA 13+ |
| CUDA 13 CuPy unblock | feature_library, build_features_v2, v2_cross_generator | No longer blocks CuPy on driver 580+ |
| PBO is_metrics | backtest_validation, smoke_test, mini_train | All callers pass is_metrics |
| COO→CSR flush | v2_cross_generator (_gpu_cross_chunk + _cpu_cross_chunk) | Bounds RAM during cross multiply |
| CSR→disk flush | v2_cross_generator (_gpu_cross_chunk) | Dumps CSR chunks to disk periodically |
| Sub-checkpoint flush | v2_cross_generator (gpu_batch_cross) | Disk-backs RIGHT_CHUNK accumulation |
| Streaming CSC splice | v2_cross_generator (final assembly) | Loads one source at a time |
| int64 indptr | v2_cross_generator (_streaming_csc_splice) | Prevents overflow at >2B NNZ |
| DB list restored | cloud_run_tf.py | All 16 DBs required (reverted optional) |

## GPU FORK STATUS: COMPLETE
- 7 bugs fixed, 78x SpMV speedup, EFB compatible
- CUDA 13 verified working
- Integrated into run_optuna_local.py

## NEXT STEPS
1. Monitor all 3 machines for cross gen completion
2. 4h: when NPZ appears, relaunch pipeline (training + Optuna)
3. Download all artifacts before destroying machines
4. 15m: pick machine after 1h results are in
5. Re-run 1w trade optimizer on non-Blackwell machine (CuPy sm_120 fix untested on optimizer)

## GIT STATUS
Branch: v3.3, uncommitted changes (OOM fixes + CuPy + PBO)
