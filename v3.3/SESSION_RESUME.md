# V3.3 Session Resume — 2026-04-01

## INSTRUCTION TO NEW SESSION
Read this file completely. Then read v3.3/CLAUDE.md. Resume from "Next Steps" below.

---

## Recent Feature Additions (2026-04-01)

### SAV-15: Soft Labels + AFML + Optuna 30 Trials — COMPLETE
- **Soft label smoothing**: y_soft = (1-epsilon)y + epsilon/2
  - Default epsilon=0.10, 1w uses 0.15 (more smoothing for tiny dataset)
  - LightGBM objective switches to 'cross_entropy' when smoothing enabled
  - Reduces overconfidence, improves calibration
- **AFML feature elimination**: Iterative SHAP-based pruning
  - Protects esoteric prefixes (gematria, numerology, astrology, space weather)
  - 80/20 train/val split for elimination before CPCV
  - Optional via ENABLE_AFML_ELIMINATION config flag (defaults False)
- **Optuna trials**: 1w bumped from 15 to 30 trials for more thorough search
- **Status**: Implementation complete, validation passed (93/94)
- **Pending**: Config variables (SAV-32, Chief Engineer)

---

## Current State

### 🔴 CRITICAL BLOCKERS (2026-04-01)

#### 1. LightGBM Import Failure
- **Error**: `lib_lightgbm.dll` missing from Python312 site-packages
- **validate.py**: 93/94 checks (only failure: `import lightgbm`)
- **Owner**: DevOps

#### 2. Convention Gate Violations (92 total)
- **SPARSE violations**: `.toarray()`/`.todense()` on cross features
  - `train_1w_cached.py:259`
  - `test_gpu_accuracy.py:337, 428`
  - `v2_cross_generator.py:473, 474`
- **SACRED violations**: `feature_pre_filter` issues
  - `run_optuna_local.py:1535, 1692`
- **Owners**: GPU Specialist + ML Pipeline Engineer + QA Lead
- **Impact**: Blocks ALL training completion

**Status**: Both blockers escalated via Paperclip. Documentation Lead cannot fix (outside ownership zone).

### 1w Training: COMPLETE (but can't retrain until LightGBM fixed)
- CPCV: 57.5%, Model: 79.3%, Binary mode, all steps PASS
- Artifacts: v3.3/1w_cloud_artifacts_v3/
- OOS 70%+ confidence = 83-100% accuracy

### 1d Training: BLOCKED on TWO issues
1. **LightGBM import** (priority 1 - blocks validation)
2. **Daemon RELOAD bug** (priority 2 - blocks cross-gen step 3+)
   - V4 daemon architecture PROVEN: 138K features in 4s, 44GB RAM (was 700GB+ OOM)
   - First 2 cross steps work via daemons (dx, ax)
   - Steps 3+ fail: daemons die during RELOAD (matrix swap between cross steps)
   - Falls back to legacy path which OOMs at 700GB+

### 4h/1h/15m: Waiting on 1d blockers resolved

---

## Machine
Sichuan 8x RTX 3090, instance 33876301, $1.12/hr, PAUSED
SSH: ssh1.vast.ai:36300
CRITICAL: v3.3/*.py must be SYMLINKS to /workspace/*.py (already set up)

---

## V4 Daemon Architecture

### Files
- v3.3/gpu_daemon.py — persistent GPU process, zero scipy, Pipe IPC
- v3.3/cross_supervisor.py — supervisor + CSR builder
- v3.3/v2_cross_generator.py — daemon_handles flow: __main__ → generate_all_crosses → _execute_one_step → gpu_batch_cross → _gpu_cross_chunk

### What Works
- 8/8 daemons start from __main__ (before CUDA init)
- CUDA kernel compiles once per daemon
- Batch dispatch: 5K pairs/batch, round-robin 8 GPUs
- First 2 cross steps complete: 138K features, 44GB RAM
- Side-channel debug: /tmp/xgen_daemon_debug.log

### The Bug
- Daemons die during RELOAD for cross step 3+ (vx, asp, etc.)
- RELOAD sends new left/right matrix paths → daemon loads, builds CSC, uploads to GPU
- Something in the RELOAD handler crashes the daemon
- Error: "All daemons dead after RELOAD" → falls back to legacy OOM path

### Root Cause Investigation Needed
- Read gpu_daemon.py RELOAD handler (around line 231-260)
- Check if matrix dimensions change between cross steps
- Check if np.hstack([left, right]) fails for different-sized matrices
- Check if CSC construction without scipy has a bug for certain shapes

---

## 10 OOM Attempts (Legacy Path)
1. All 8 simultaneous → 714GB OOM
2. Disable pinned pool → 693GB
3. Stagger launch → all launched
4. RSS measurement → too low
5. RAM stability gate → spike later
6. Sparse kernel → 283GB start, grew to 691GB
7. Flush 5K → 699GB
8. Stream-to-disk → 707GB
9. Deferred merge → 722GB
10. 15% headroom → not tested

Root causes: scipy fragmentation, CuPy state, np.ascontiguousarray copies, csr_chunks unbounded

---

## 20 Speed Optimizations Implemented
See git log for details. Key ones:
- CUDA sparse kernel (O(nnz) memory)
- gpu_use_dp=False (2-4x GPU speed)
- Persistent GPU daemons (zero scipy in workers)
- Bitpack POPCNT cache optimization
- Two-pass → single-pass Numba fusion
- Parallel final retrain across GPUs
- jemalloc + NUMA + THP tuning

---

## Deploy Protocol
1. Edit locally, commit, push
2. python v3.3/deploy_manifest.py
3. SCP *.py + manifest to /workspace/
4. Symlink: for f in /workspace/*.py; do ln -sf $f /workspace/v3.3/$(basename $f); done
5. deploy_verify.py must PASS
6. Launch training

---

## Next Steps
1. **URGENT**: Fix LightGBM import (lib_lightgbm.dll missing) — blocks ALL training
2. Fix daemon RELOAD bug in gpu_daemon.py (blocks 1d cross-gen step 3+)
3. Once both fixed: 1d trains with 44GB RAM on 774GB machine (proven)
4. Then 4h → 1h → 15m
5. Long-term: custom CUDA C++ pipeline (Option B, 4-5 months)
