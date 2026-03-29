# V3.3 GPU Histogram Fork — Session Resume

## INSTRUCTION TO NEW SESSION
Read this file completely. It contains the full state of training, cloud machines, and what needs doing next.

## ACTIVE MACHINES (2026-03-29)

**1d: m:28851559** — 8× RTX 5090, Score 1423, 755GB RAM, $3.73/hr
- SSH: `ssh -p 23400 -i ~/.ssh/id_ed25519 root@ssh6.vast.ai`
- Monitor: `tail -f /workspace/train_1d.log`
- Status: RELAUNCHED after OOM fix. Has checkpoints for dx, ax, ax2, ta2. Resuming from Cross 5 (ex2).
- After 1d completes: RUN 4H ON THIS SAME MACHINE (755GB RAM is enough)

**4h machine was destroyed** — OOM'd at 479GB. Will run on 1d machine after 1d finishes.

## TRAINING STATUS

| TF | Status | Machine | Accuracy | Notes |
|----|--------|---------|----------|-------|
| 1w | **DONE** | m:32572512 (destroyed) | 58% CPCV, 29% SHORT | Artifacts downloaded to v3.3/cloud_results_1w/ |
| 1d | **RUNNING** | m:28851559 ($3.73/hr) | — | Cross gen with checkpoint fix, resuming Cross 5 |
| 4h | **QUEUED** | Run on 1d machine after | — | Previous machine OOM'd at 479GB |
| 1h | **QUEUED** | Run on 1d machine after 4h | — | 755GB should be enough |
| 15m | **NOT STARTED** | Need 1TB+ machine | — | Separate machine needed |

## 1w RESULTS (Downloaded)
- model_1w.json (90MB) — 58% CPCV, 62.5% final accuracy
- Optuna params: num_leaves=29, lambda_l2=10.94, min_data_in_leaf=14
- SHORT precision: 29% (was 0% with symmetric barriers) — WORKING
- Trade optimizer: FAILED (CuPy incompatible with Blackwell sm_120)
- PBO: FAILED (missing is_metrics parameter)
- Artifacts in: C:\Users\C\Documents\Savage22 Server\v3.3\cloud_results_1w\

## OOM LOG (CRITICAL — USE FOR MACHINE SELECTION)

| TF | RAM | Cross Type | Result |
|----|-----|------------|--------|
| 1w | 193GB | Full pipeline | SUCCESS (818 rows, tiny) |
| 1d | 193GB | Cross 2 (ax_) | OOM |
| 1d | 193GB | Training | OOM |
| 1d | 755GB | Cross 5 (ex2_) | OOM (chunks not freed — FIXED) |
| 4h | 479GB | Cross 2 (ax_) | OOM (pre-fix AND post-fix) |

**Lesson**: Cross gen accumulates CSR chunks in RAM. The checkpoint fix (d449be9) now hstacks ALL chunks, saves to disk, clears entire list. Peak RAM should now be max(single batch within one cross type), not sum(all types).

## CUDA 13 STATUS: VERIFIED WORKING
- Built + tested on RTX PRO 5000 Blackwell (driver 580.126.09, nvcc 13.2)
- cuSPARSE 13.x backward compatible
- Fix: std::max explicit cast for nvcc 13.2 + GCC 14.2
- build_linux.sh ships pre-patched LightGBM source (14MB tar)
- Must copy .so to BOTH lightgbm/ AND lightgbm/lib/
- All vast.ai machines compatible

## KEY FIXES THIS SESSION
1. **PROJECT_DIR undefined** in cloud_run_tf.py — added before stale cleanup
2. **TF_MIN_RAM too high** — lowered after targeted crossing (4h: 768→256, 1h: 1024→512)
3. **Cross gen OOM** — checkpoint saves ALL chunks (not just last), clears entire list, reloads for final assembly
4. **std::max nvcc 13.2** — explicit cast for int64_t vs long long compatibility
5. **SSH key re-registration** — vast.ai keys need re-creating when machines can't connect

## GPU FORK STATUS: COMPLETE
- 7 bugs fixed, 78x SpMV speedup, EFB compatible
- CUDA 13 verified working
- GPU is SLOWER than CPU on 1w/1d (small data, overhead dominates)
- GPU helps on 4h+ (23K+ rows)
- Multi-GPU: gpu_device_id round-robin committed, true parallelism TODO

## OPTUNA STATUS: Phase 1 + Validation Gate
- 20-30 trials total (was 150)
- Dataset.subset() for instant fold construction
- PatientPruner (patience=5)
- Warm-start cascade: 1w→1d→4h→1h→15m
- Optuna takes ~0.5x final training time (was 11x)

## ARCHITECTURAL CHANGES IMPLEMENTED
- Asymmetric triple-barrier labels (tp/sl split per TF)
- CPCV K=2 for final evaluation (5,2)/(6,2)
- Targeted crossing (ALL→TA for crosses 6-12, rdx_ removed)
- 4-tier binarization KEPT
- Cross gen per-type checkpoint + resume
- Model backup (_prev.json)
- Accuracy floor (40% minimum)
- save_binary bridge (Optuna→training)

## NEXT STEPS
1. Wait for 1d to complete on m:28851559
2. Run 4h on same machine
3. Run 1h on same machine (if 755GB is enough)
4. Pick 1TB+ machine for 15m
5. Re-run 1w trade optimizer on a non-Blackwell machine
6. Fix PBO (missing is_metrics parameter)

## GIT STATUS
Latest commit: d449be9 — fix: checkpoint saves ALL chunks not just last
Branch: v3.3, pushed to origin
