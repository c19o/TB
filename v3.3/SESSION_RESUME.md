# V3.3 Session Resume — 2026-03-25 (Updated after bug fixes)

## MACHINES (ALL NEED REDEPLOYMENT)

All 5 machines are running OLD code with single-threaded bug. Must kill, upload fixed code, relaunch.

| TF | Instance | Port | Host | CPU | $/hr | Status |
|----|----------|------|------|-----|------|--------|
| 1w | 33496064 | 16064 | ssh1.vast.ai | EPYC 9654 384c 3.7GHz | $0.80 | STALE — single-threaded, needs redeploy |
| 1d | 33497394 | 17394 | ssh2.vast.ai | EPYC 7773X 128c 3.5GHz | $0.32 | STALE — single-threaded, needs redeploy |
| 4h | 33497395 | 17394 | ssh6.vast.ai | EPYC 7B13 128c 3.5GHz | $0.23 | STALE — single-threaded, needs redeploy |
| 1h | 33496904 | 16904 | ssh1.vast.ai | EPYC 9K84 384c 3.7GHz | $0.80 | STALE — single-threaded 5.5hrs wasted |
| 15m | 33497779 | 17778 | ssh9.vast.ai | Ryzen 9950X 32c 8.8GHz | $0.77 | STALE — cross gen stuck, needs redeploy |

## SSH COMMAND
```bash
SSH_OPTS="-o StrictHostKeyChecking=no -o IdentityFile=/c/Users/C/.ssh/vast_key -o IdentitiesOnly=yes"
ssh $SSH_OPTS -p <PORT> root@<HOST> "tail -20 /workspace/<TF>_pipeline.log"
```

## BUGS FIXED THIS SESSION (7 fixes, 3 clean audit passes)

### Fix A (CRITICAL): Single-threaded training root cause
- `_X_all_is_sparse = True` hardcoded → `hasattr(X_all, 'nnz')` tracks actual type
- Dense machines now use sequential path with ALL cores per fold
- Sparse machines (15m) use parallel path with N single-threaded workers
- File: ml_multi_tf.py line 653

### Fix B: Step 5 (--search-mode) overwrote Step 4 model
- Removed Step 5 entirely from cloud_run_tf.py
- Added model backup: model_{TF}_cpcv_backup.json after Step 4
- File: cloud_run_tf.py lines 379-383

### Fix C: SHAP .toarray() OOM
- Replaced pred_contrib with split/gain importance only
- No dense materialization of 2.9M+ cross matrix
- File: cloud_run_tf.py lines 440-490

### Fix D: Audit .db vs .parquet
- Early return on missing .db now checks parquet too
- File: backtesting_audit.py lines 153-159

### Fix E: Artifact name mismatch
- Updated artifact list to match actual filenames
- File: cloud_run_tf.py lines 103-115

### Fix F: NNZ guard for 15m int32 overflow
- Auto-subsample rows when NNZ > 2B (LightGBM int32 limit)
- min 1000 rows floor, keeps most recent data
- File: ml_multi_tf.py lines 609-621

### Fix G: 15m GPU skip
- Skip GPU cross gen when N > 100K rows (guaranteed OOM)
- CPU fallback works, just slower
- File: v2_cross_generator.py line 326

### Additional fixes found during audit:
- psutil replaced with /proc/meminfo fallback (no extra pip dependency)
- df subsampled in dense conversion path (was missing)
- cross_cols initialized as [] not None (both ml_multi_tf.py AND backtesting_audit.py)
- _target_rows has max(1000, ...) floor guard

## PROTOCOL HARDENING ADDED
- Pre-flight checklist (7 items to verify before every deploy)
- Post-launch validation (check within 60s: load avg, RSS, log output)
- Periodic health checks (load avg >> 1.0, no FAIL/Error, log growing)
- Step gates (verify artifacts after each pipeline step)
- Updated CLOUD_TRAINING_PROTOCOL.md and CLAUDE.md

## NEXT STEPS
1. **Get user permission to kill all 5 machines** (all running stale single-threaded code)
2. SCP fixed code to each machine (reuse, don't re-rent)
3. Relaunch with fixed code
4. Post-launch validation within 60s per machine
5. Monitor with periodic health checks per protocol
6. 15m: NNZ guard will subsample 227K→~111K rows. Training stays sparse+single-threaded. Fastest clock machine (9950X 5.7GHz) is correct for this.

## EXPECTED PERFORMANCE WITH FIXES
| TF | Cores | Path | Estimated Time |
|----|-------|------|---------------|
| 1w | 384 | Dense sequential, all cores | ~5-10 min |
| 1d | 128 | Dense sequential, all cores | ~30-60 min |
| 4h | 128 | Dense sequential, all cores | ~30-60 min |
| 1h | 384 | Dense sequential, all cores | ~1-2 hrs |
| 15m | 32 | Sparse parallel (6 workers × 1 thread each) | ~4-8 hrs |
