# DOUBLE AUDIT #3: Game-Changers #6-10 + Bug Fix Verification

**Auditor**: Error Checker Agent (READ-ONLY)
**Date**: 2026-03-30
**Branch**: v3.3

---

## GAME-CHANGERS #6-10

### #6 Cost-Sort (Pairs sorted by NNZ descending)

**STATUS: NOT IMPLEMENTED** | Severity: HIGH

- No NNZ-based pair sorting found anywhere in `v2_cross_generator.py`
- `_compute_cooccurrence_pairs()` returns pairs in arbitrary order
- `_parallel_cross_multiply()` (line 973) uses raw `prange` without cost-aware scheduling
- CPU path (ThreadPoolExecutor) uses static block sizes — no cost awareness
- GPU path uses VRAM-adaptive batching but no NNZ-based pair reordering
- `EXPERT_COMPILER.md` lines 184-189 documents the optimization but code does NOT implement it

**Impact**: Load imbalance in prange — some threads get heavy pairs, others idle.

---

### #7 .npy Format (save_csr_npy / load_csr_npy with mmap_mode='r')

**STATUS: NOT IMPLEMENTED** | Severity: MEDIUM

- Code still saves `.npz` everywhere:
  - `v2_cross_generator.py:663` — `sparse.save_npz(_npz_path, _merged, compressed=False)`
  - `v2_cross_generator.py:881` — `sparse.save_npz(_path, _merged, compressed=False)`
  - `v2_cross_generator.py:1570` — `sparse.save_npz(_ckpt_npz, _merged, compressed=False)`
- No `save_csr_npy` or `load_csr_npy` functions exist in executable code
- `EXPERT_LINUX_TUNING.md:120-141` documents that `.npz` mmap_mode is **silently ignored**
- `EXPERT_NVME_IO.md:62-69` defines correct `load_csr_memmap()` with `mmap_mode='r'` — but it's only in docs

**Impact**: Cannot mmap CSR arrays; full copies loaded into RAM on every read.

---

### #8 THP=madvise (Transparent Huge Pages)

**STATUS: BUG — 3 SCRIPTS SET `always` INSTEAD OF `madvise`** | Severity: HIGH

All EXPERT docs correctly specify `madvise`:
- `EXPERT_CPU_CACHE.md:334` — `echo madvise`
- `EXPERT_MEMORY_ALLOC.md:161,350` — `echo madvise`
- `EXPERT_NVME_IO.md:169,332` — `echo madvise`
- `EXPERT_LINUX_TUNING.md:64,264` — `echo madvise`

**BUT all 3 deployment scripts set `always`:**
- `setup.sh:62` — `echo always > /sys/kernel/mm/transparent_hugepage/enabled` ← **WRONG**
- `deploy_vastai.sh:633` — `echo always > /sys/kernel/mm/transparent_hugepage/enabled` ← **WRONG**
- `deploy_vastai_quick.sh:131` — `echo always > /sys/kernel/mm/transparent_hugepage/enabled` ← **WRONG**

The comment on `setup.sh:58` even says: *"THP 'always' causes multi-second stalls during LightGBM's small-alloc phase"* — then line 62 sets it to `always` anyway. Contradicts itself.

Defrag is correctly set to `defer+madvise` in all scripts.

**Impact**: Multi-second stalls during LightGBM sparse allocation phase. The exact problem the comment warns about.

---

### #9 Process Isolation (subprocess saves, SharedMemory, deadlock safety)

**STATUS: SAFE** | Severity: NONE

- `build_features_v2.py:245` — `multiprocessing.set_start_method('spawn', force=True)` — correct, no fork deadlocks
- `cloud_run_tf.py:92-110` — `subprocess.Popen()` for tee'd output, no locks held across fork
- `cloud_run_tf.py:226-247` — file-based lockfile prevents duplicate pipeline runs (safe)
- SharedMemory is documented in `EXPERT_PYTHON_PERF.md:46-56` but NOT used in training code — data passed via `.npy` files, avoiding complexity
- No deadlock risk identified

---

### #10 fastmath (ONLY on _parallel_cross_multiply)

**STATUS: NOT IMPLEMENTED** | Severity: LOW

- No `fastmath=True` found on any Numba decorator in the codebase
- `v2_cross_generator.py:973` — `@njit(parallel=True, cache=True)` — missing `fastmath=True`
- `v2_cross_generator.py:260,348,358` — binarize functions correctly do NOT have fastmath (NaN-handling)
- `EXPERT_COMPILER.md:57-58` recommends `fastmath=True` on `_parallel_cross_multiply` only
- `EXPERT_COMPILER.md:66` correctly warns against fastmath on `_binarize_batch_4tier`

**Impact**: Missing 5-10% speedup on dense multiply step. Safety is correct (no fastmath on NaN code).

---

## BUG FIX VERIFICATION

### BUG-C1: Warp Shuffle (__ballot_sync)

**STATUS: NOT IN KERNEL CODE — DOCUMENTATION ONLY**

- `__ballot_sync` appears only in `EXPERT_CUSPARSE.md:111` as a design recommendation
- NOT found in any `.cu` or `.cuh` kernel files in `gpu_histogram_fork/`
- Only `__shfl_sync` found in Eigen headers (LightGBM dependency, not our code)
- The warp shuffle optimization exists as a documented pattern, not as implemented code

---

### BUG-C2: cuda_sparse device_type

**STATUS: FULLY IMPLEMENTED** ✓

Correctly integrated across 15+ files:
- `bench_trial.py:68` — `gpu_params["device_type"] = "cuda_sparse"`
- `ml_multi_tf.py:352,502` — `params['device_type'] = 'cuda_sparse'`
- `run_optuna_local.py:512,537,702,914` — multiple cuda_sparse assignments
- `validate.py:537` — cuda_sparse validation tests
- `gpu_histogram_fork/src/train_pipeline.py:392,502` — pipeline integration
- `gpu_histogram_fork/src/cloud_gpu_integration.py:429` — cloud integration
- `gpu_histogram_fork/check_gpu.py:159-176` — detection function `detect_lgbm_cuda_sparse()`
- `runtime_checks.py:168-178` — runtime smoke tests

---

### BUG-H1: GPU Contention Guard

**STATUS: IMPLEMENTED VIA CUDA_VISIBLE_DEVICES** ✓

- `gpu_histogram_fork/src/multi_gpu.py:286` — `env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)` per subprocess
- `gpu_histogram_fork/src/cuda_compat.py:607` — `os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_index)`
- `cloud_runner.py:26,78` — explicitly pops CUDA_VISIBLE_DEVICES to expose all GPUs when needed
- `v2_cloud_runner.py:390,750` — per-process GPU assignment via env_overrides

No mutex/semaphore locking — isolation is via environment variable routing per subprocess.

---

### BUG-H2: Single GPU Guard (num_gpus >= 2)

**STATUS: IMPLEMENTED** ✓

- `ml_multi_tf.py:1172` — `if _num_gpus > 1:` (functionally equivalent to `>= 2`)
- Multi-GPU: ThreadPoolExecutor for parallel CPCV (no pickle overhead)
- Single GPU: sequential CPCV (lines 1176-1179)
- GPU count detected via nvidia-smi at line 1168

---

### NUMA Fix: --interleave=all

**STATUS: IMPLEMENTED** ✓

Present in all deployment paths:
- `setup.sh:153` — `NUMA_BIND_CMD="numactl --interleave=all"`
- `setup.sh:240` — `exec numactl --interleave=all "$@"`
- `deploy_vastai.sh:671-672` — numactl wrapper
- `deploy_vastai_quick.sh:148` — numactl wrapper

---

### Pruning Warmup: n_warmup_steps=120

**STATUS: IMPLEMENTED** ✓

- `run_optuna_local.py:1273` — `n_warmup_steps=120`
- Comment at line 1255 explains: *"discover rare signals — n_warmup_steps=120 ensures both folds complete"*
- Note: `runpod_train.py:557` has `n_warmup_steps=5` (different context — lightweight RunPod tests)

---

## SUMMARY

| Item | Status | Severity | Action Required |
|------|--------|----------|-----------------|
| **#6 Cost-Sort** | NOT IMPLEMENTED | HIGH | Implement NNZ-descending sort before prange |
| **#7 .npy Format** | NOT IMPLEMENTED | MEDIUM | Replace save_npz with separate .npy saves |
| **#8 THP=madvise** | **BUG: 3 scripts say `always`** | **HIGH** | Change to `madvise` in setup.sh:62, deploy_vastai.sh:633, deploy_vastai_quick.sh:131 |
| **#9 Process Isolation** | SAFE ✓ | — | None |
| **#10 fastmath** | NOT IMPLEMENTED | LOW | Add fastmath=True to line 973 only |
| **BUG-C1 __ballot_sync** | DOC ONLY, NOT IN KERNEL | INFO | Implement in CUDA kernel or remove from bug tracker |
| **BUG-C2 cuda_sparse** | FULLY IMPLEMENTED ✓ | — | None |
| **BUG-H1 GPU contention** | IMPLEMENTED ✓ | — | None |
| **BUG-H2 Single GPU** | IMPLEMENTED ✓ | — | None |
| **NUMA --interleave=all** | IMPLEMENTED ✓ | — | None |
| **Warmup n_warmup_steps=120** | IMPLEMENTED ✓ | — | None |

### Critical Findings Count
- **2 HIGH**: #6 cost-sort missing, #8 THP `always` bug (contradicts own comments + all EXPERT docs)
- **1 MEDIUM**: #7 .npy format missing
- **1 LOW**: #10 fastmath missing
- **1 INFO**: BUG-C1 documented but not in actual kernel code
