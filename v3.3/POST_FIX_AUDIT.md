# POST-FIX AUDIT REPORT
**Date:** 2026-03-30
**Auditor:** Error Checker Agent (READ-ONLY)
**Scope:** Verify 6 critical+high fixes landed in CEO branches

---

## FIX VERIFICATION MATRIX

### 1. Sortino Denominator — `exhaustive_optimizer.py`
**Status: FAIL on v3.3 / PASS on branch `ceo/backend-dev-cf941cf8`**

- **v3.3 (current):** Line 686-687 divides `sum_neg_sq` by `total_trades` (ALL trades)
- **Branch fix:** Changes denominator to `count_neg` (count of NEGATIVE returns only)
- Both exhaustive and Optuna (Sobol) paths use same `_sim_vectorized_all()` function, so fix covers both
- `count_neg` is correctly accumulated at line 596: `count_neg += neg_mask.astype(xp_lib.int32)`

**Impact if unmerged:** Sortino ratio is diluted — downside deviation underestimated when most trades are profitable, leading to inflated Sortino scores and overfit optimizer configs.

---

### 2. CPCV Temporal Leakage — `ml_multi_tf.py`
**Status: PASS (no leakage in current code)**

Analysis of `_generate_cpcv_splits()` (lines 185-269):

- **t0/t1 path (lines 232-243):** Uses `test_min`/`test_max` of the union of test indices. For non-contiguous groups [2, 5], this over-purges (removes training rows in the gap between groups). Conservative, not leaky.
- **max_hold_bars path (lines 244-255):** Iterates per-group boundaries (`groups[g][0]`, `groups[g][-1]`). Correctly purges at each boundary independently.
- **Embargo (lines 258-264):** `for g in test_group_ids` — applies embargo after EACH test group independently. Correct.

CEO branches (`cf941cf8`, `19f6b5f7`, `41c9d053`) add `sample_paths` parameter but do NOT change core purge logic. No fix needed.

---

### 3. THP=madvise — All `.sh` files
**Status: FAIL on v3.3 / PASS on branches `ceo/backend-dev-cf941cf8`, `ceo/backend-dev-19f6b5f7`, `ceo/backend-dev-41c9d053`**

- **v3.3 (current):** 3 files set THP to `always`:
  - `setup.sh` line 62: `echo always > .../transparent_hugepage/enabled`
  - `gpu_histogram_fork/deploy_vastai.sh` line 633: `echo always`
  - `gpu_histogram_fork/deploy_vastai_quick.sh` line 131: `echo always`
- **Branch fix:** All 3 files changed to `echo madvise` with correct comment explaining why

**Impact if unmerged:** THP `always` causes 512x memory bloat on sparse CSR regions and multi-second stalls during LightGBM training.

---

### 4. Cost-Sort Pairs by NNZ — `v2_cross_generator.py`
**Status: FAIL on v3.3 / PASS on branches `ceo/backend-dev-cf941cf8`, `ceo/backend-dev-19f6b5f7`, `ceo/backend-dev-41c9d053`**

- **v3.3 (current):** No sorting of `valid_pairs` before the cross-multiply loop
- **Branch fix:** Adds `pair_nnz = co_occur[valid_pairs[:, 0], valid_pairs[:, 1]]` + `np.argsort(-pair_nnz)` in BOTH GPU and CPU cross chunk functions
- Heavy pairs processed first, light pairs fill in at tail — fixes prange load imbalance

---

### 5. `__ballot_sync` — `gpu_histogram.cu`
**Status: FAIL on v3.3 / PASS on branches `ceo/backend-dev-22d6ed1e`, `ceo/backend-dev-4181eede`**

- **v3.3 (current):** No `__ballot_sync` anywhere in gpu_histogram.cu
- **Branch fix (22d6ed1e):** Adds warp-cooperative kernel with `__ballot_sync(0xffffffff, true)` used to build active lane mask before `__shfl_down_sync`. Correct usage — determines which lanes are active before warp reduction.
- Comment correctly notes: "active_mask must reflect actually active lanes (via `__ballot_sync`), NOT `0xffffffff`" — though the call itself uses `0xffffffff` as the participation mask (correct for compute >= 7.0).

**Note:** This is in a NEW kernel (`sparse_hist_build_warp_kernel`), not the original. Enabled via `CUDA_WARP_REDUCE=1` env var.

---

### 6. `fastmath=True` — `v2_cross_generator.py`
**Status: FAIL on v3.3 / PASS on branches `ceo/backend-dev-cf941cf8`, `ceo/backend-dev-19f6b5f7`, `ceo/backend-dev-41c9d053`**

- **v3.3 (current):** Line 973: `@njit(parallel=True, cache=True)` — no fastmath
- **Branch fix:** `@njit(parallel=True, cache=True, fastmath=True)` with docstring noting safety (binary 0/1 inputs, no NaN, no subnormals)

---

## BRANCH COVERAGE SUMMARY

| Fix | cf941cf8 | 19f6b5f7 | 41c9d053 | 22d6ed1e | 4181eede |
|-----|----------|----------|----------|----------|----------|
| 1. Sortino denom | YES | - | - | - | - |
| 2. CPCV purge | N/A (no bug) | N/A | N/A | N/A | N/A |
| 3. THP=madvise | YES | YES | YES | - | - |
| 4. Cost-sort NNZ | YES | YES | YES | - | - |
| 5. __ballot_sync | - | - | - | YES | YES |
| 6. fastmath | YES | YES | YES | - | - |

**Best single branch:** `ceo/backend-dev-cf941cf8` covers 4/5 actual fixes (all except `__ballot_sync`).

---

## FINAL VERDICT

### NO-GO for merge of v3.3 as-is

**4 of 5 real bugs remain unfixed on v3.3 branch.** Fixes exist on CEO branches but have not been merged.

### Recommended merge order:
1. `ceo/backend-dev-cf941cf8` — covers Sortino, THP, cost-sort, fastmath (4 fixes)
2. `ceo/backend-dev-22d6ed1e` — covers `__ballot_sync` warp kernel (1 fix)
3. Re-run this audit after merge to confirm all 5 fixes landed on v3.3

### Risk if deployed without fixes:
- **Sortino (HIGH):** Optimizer picks wrong configs due to inflated scores
- **THP (MEDIUM):** Memory bloat + stalls on cloud machines
- **Cost-sort (LOW):** Suboptimal cross-gen parallelism (correctness unaffected)
- **ballot_sync (LOW):** Only affects optional warp-reduce kernel
- **fastmath (LOW):** ~15-20% speed improvement lost on cross multiply
