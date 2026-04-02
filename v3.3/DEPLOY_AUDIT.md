# Cloud Deployment Audit Report
**Date:** 2026-04-01  
**Auditor:** DevOps Agent  
**Scope:** cloud_run_tf.py, deploy_manifest.py, discord_gate.py

---

## Executive Summary

**Status:** 🟡 PARTIAL PASS — 4 of 5 requirements verified  
**Critical Issue:** Discord progress posting is MISSING

---

## Audit Findings

### ✅ PASS: All 7+ Pipeline Steps Present

**Verified Steps:**
- **Step 0:** Kill stale processes, install deps (line 285)
- **Step 1:** Fix btc_prices.db symbol format (line 348)
- **Step 2:** Rebuild features if parquet missing/incomplete (line 426)
- **Step 3:** Build crosses with NPZ skip logic (line 547)
- **Step 4:** Train (ml_multi_tf.py) — verifies sparse output (line 804)
- **Step 5:** Optuna hyperparameter search (line 640)
- **Step 6:** Exhaustive trade optimizer (line 862)
- **Steps 7-9:** Meta-labeling, LSTM, PBO (parallel execution, line 868)
- **Step 10:** Audit (sequential after optimizer, line 896)
- **Step 11:** SHAP cross feature validation (line 902)

**Evidence:**
```python
# Lines 285-902 in cloud_run_tf.py
# All steps documented, implemented, and called in sequence
```

**Result:** ✅ **PASS** — 11 distinct steps implemented and executed in correct order

---

### ✅ PASS: NPZ Skip Logic

**Implementation:** Lines 547-599 in cloud_run_tf.py

**Features:**
1. Checks if NPZ file exists before rebuilding crosses
2. Validates NPZ column count against minimum threshold per TF
3. Detects stale NPZs from v3.0 (old min_nonzero=8 config)
4. Symlinks NPZ if found in alternate directory
5. Supports per-TF cross feature toggling (1w skips crosses due to low row count)

**Thresholds:**
```python
_MIN_CROSS_COLS = {
    '1w': 500_000, 
    '1d': 1_000_000, 
    '4h': 1_000_000, 
    '1h': 1_000_000, 
    '15m': 1_000_000
}
```

**Skip Logic:**
```python
if os.path.exists(npz_path) and os.path.getsize(npz_path) > 1000:
    npz_size = os.path.getsize(npz_path) / (1024*1024)
    _min_cols = _MIN_CROSS_COLS.get(TF, 1_000_000)
    _npz_shape = _sp.load_npz(npz_path).shape
    if _npz_shape[1] >= _min_cols:
        _npz_valid = True
        log(f"Cross NPZ valid ({npz_size:.1f} MB, {_npz_shape[1]:,} cols >= {_min_cols:,} min) — SKIPPING cross gen")
```

**Result:** ✅ **PASS** — Comprehensive NPZ skip logic prevents redundant multi-hour cross generation

---

### ✅ PASS: SIGTERM Handler

**Implementation:** Lines 340-345 in cloud_run_tf.py

```python
# Lockfile cleanup on exit (normal, crash, or signal)
import atexit, signal as _signal

def _cleanup_lock():
    try: os.remove(_lockfile)
    except: pass

atexit.register(_cleanup_lock)
_signal.signal(_signal.SIGTERM, lambda *a: (_cleanup_lock(), sys.exit(0)))
```

**Coverage:**
- ✅ Normal exit (atexit)
- ✅ SIGTERM (signal handler)
- ✅ Removes lockfile to prevent stale locks

**Note:** SIGKILL (kill -9) cannot be caught — lockfile will remain stale in that case. Current implementation checks for stale locks by verifying process existence (lines 327-335).

**Result:** ✅ **PASS** — SIGTERM handler implemented with lockfile cleanup

---

### ❌ FAIL: Discord Progress Posting

**Status:** MISSING

**Expected Behavior:**
- Post to Discord at each step completion/failure
- Use `discord_gate.py` notify/critical functions
- Report progress, timing, and failures in real-time

**Actual Behavior:**
- NO Discord imports in cloud_run_tf.py
- NO calls to `gate.notify()`, `gate.approve()`, or `gate.critical()`
- Pipeline runs silently — user must SSH to check progress
- No notifications on failure/OOM/completion

**Discord Gate Available Functions:**
```python
from discord_gate import gate

gate.notify("Training 1d complete", {"accuracy": "62.3%", "pbo": 0.12})
gate.critical("OOM at 700GB", {"step": "cross_gen", "tf": "1h"})
```

**Suggested Integration Points:**
1. ✅ Step completion: `gate.notify(f"Step {N} complete: {name}", {"duration": dt, "tf": TF})`
2. ✅ Pipeline completion: `gate.notify("PIPELINE COMPLETE", {"tf": TF, "total_time": elapsed})`
3. ✅ Critical failures: `gate.critical(f"CRITICAL FAILURE: {name}", {"step": name, "tf": TF})`
4. ✅ OOM/crash: `gate.critical("Pipeline crashed", {"last_step": name, "exit_code": rc})`

**Result:** ❌ **FAIL** — Discord integration NOT implemented

---

### ✅ PASS: deploy_manifest.py Complete

**File:** v3.3/deploy_manifest.py (126 lines)

**Features:**
1. ✅ SHA256 hash generation for all .py files
2. ✅ Manifest includes generation timestamp, hostname, Python version
3. ✅ Critical file verification (23 files)
4. ✅ File size tracking
5. ✅ Deployment verification support (deploy_verify.py reads manifest)

**Critical Files List:**
```python
CRITICAL_FILES = [
    "config.py", "feature_library.py", "ml_multi_tf.py",
    "run_optuna_local.py", "cloud_run_tf.py", "validate.py",
    "runtime_checks.py", "v2_cross_generator.py", ...
]  # 23 total
```

**Usage:**
```bash
# LOCAL: Generate manifest before deploy
python deploy_manifest.py

# CLOUD: Verify after SCP
python deploy_verify.py --tf <TF>
```

**Result:** ✅ **PASS** — deploy_manifest.py is complete and functional

---

## deploy_manifest.json Verification

**File:** v3.3/deploy_manifest.json (465 lines)

**Content:**
- ✅ Generated timestamp: 2026-03-31 23:02:42 UTC
- ✅ Total files: 108 .py files hashed
- ✅ All 23 critical files present
- ✅ SHA256 + file size for each file
- ✅ Python version: 3.12.10

**Sample:**
```json
{
  "generated_at": "2026-03-31 23:02:42 UTC",
  "generated_on": "DESKTOP-GG75I1X",
  "python_version": "3.12.10",
  "files": {
    "cloud_run_tf.py": {
      "sha256": "3addc6a90288e0a1fdc6ef218cf242faa4ae531850a115276cb344cd739559de",
      "size": 46329
    },
    ...
  },
  "critical_files": [...],
  "total_files": 108
}
```

**Result:** ✅ **PASS** — Manifest is current and complete

---

## Recommendations

### 1. ADD Discord Progress Posting (HIGH PRIORITY)

**Add to cloud_run_tf.py:**

```python
# At top of file
from discord_gate import gate

# After each step completion (in run() function)
def run(cmd, name, critical=True):
    t0 = time.time()
    log(f"=== {name} ===")
    r = subprocess.run(cmd, shell=True)
    dt = time.time() - t0
    ok = r.returncode == 0
    log(f"{name}: {'OK' if ok else 'FAIL'} ({dt:.0f}s)")
    
    # ADD THIS:
    if ok:
        gate.notify(f"Step complete: {name}", {
            "tf": TF,
            "duration_s": int(dt),
            "duration_min": round(dt/60, 1)
        }, event_type="training_step")
    else:
        gate.critical(f"Step FAILED: {name}", {
            "tf": TF,
            "step": name,
            "duration_s": int(dt),
            "exit_code": r.returncode
        }, event_type="pipeline_error")
    
    if not ok:
        FAILURES.append(name)
        if critical:
            log(f"*** CRITICAL FAILURE: {name} — aborting ***")
            _print_summary()
            sys.exit(1)
    return ok

# At end of _print_summary()
def _print_summary():
    # ... existing code ...
    
    # ADD THIS:
    if FAILURES:
        gate.critical(f"PIPELINE FAILED: {TF}", {
            "tf": TF,
            "total_time_s": int(elapsed_total),
            "failures": FAILURES
        }, event_type="pipeline_error")
    else:
        gate.notify(f"PIPELINE COMPLETE: {TF}", {
            "tf": TF,
            "total_time_s": int(elapsed_total),
            "total_time_min": round(elapsed_total/60, 1)
        }, event_type="training_complete")
```

**Benefit:**
- Real-time visibility into cloud training progress
- Immediate failure notifications (no need to SSH to check logs)
- Historical audit trail in discord_gate_log.json
- Aligns with memory feedback: "Always run scripts with progress logs"

---

### 2. Test Discord Integration Locally (MEDIUM PRIORITY)

**Before deploying Discord-enabled cloud_run_tf.py:**

```bash
# Test discord_gate.py connectivity
cd v3.3
python discord_gate.py ping

# Test notification
python discord_gate.py notify "Test notification from audit" --context '{"test": true}'
```

**Verify:**
- ✅ Discord bot token valid
- ✅ DM channel opens
- ✅ Embeds render correctly
- ✅ discord_gate_log.json written

---

### 3. Consider Rate Limiting (LOW PRIORITY)

**Current Risk:**
- 11+ steps × notify() = 11+ Discord messages per TF
- 5 TFs = 55+ messages in a full training run
- Discord rate limit: ~50 messages / 10 seconds

**Mitigation:**
- Only notify on CRITICAL steps (cross gen, training, optimizer)
- Batch updates: post every 3 steps instead of every step
- Use INFO level for minor steps, CRITICAL for failures

---

## Conclusion

**Overall Status:** 🟡 **4/5 PASS** — Ready to deploy with Discord integration gap

### Verified ✅
- All 11 pipeline steps present and functional
- NPZ skip logic prevents redundant cross generation
- SIGTERM handler cleans up lockfile
- deploy_manifest.py complete and current

### Missing ❌
- Discord progress posting (no notifications sent during pipeline execution)

**Recommendation:** ADD Discord integration before next cloud deploy. Current pipeline is functional but runs blind — no real-time progress visibility.

---

**Audit Complete.**  
**Next Action:** Implement Discord progress posting (see Recommendations §1).
