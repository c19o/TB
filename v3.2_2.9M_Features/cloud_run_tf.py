#!/usr/bin/env python3
"""
cloud_run_tf.py — Run full pipeline for ONE timeframe on cloud.

Usage: python -u cloud_run_tf.py --tf 1w

Tar should extract flat to /workspace/ with:
  - All .py, .json, .pkl files from v3.2
  - features_BTC_{tf}.parquet (or will rebuild if missing/incomplete)
  - btc_prices.db (root version with BTC/USDT, or multi_asset_prices.db to be fixed)
  - V1 DBs (needed only if feature rebuild required)
  - kp_history.txt (needed only if feature rebuild required)

Steps:
  0. Kill stale processes, install deps
  1. Fix btc_prices.db symbol format if needed
  2. Rebuild features if parquet missing or < 2000 cols
  3. Build crosses (v2_cross_generator.py --symbol BTC --save-sparse)
  4. Train (ml_multi_tf.py --tf TF) — VERIFY SPARSE
  5-9. Optuna, optimizer, meta, LSTM, PBO, audit
  10. Verify all artifacts exist
"""
import os, sys, subprocess, time, json, glob, sqlite3

os.environ['PYTHONUNBUFFERED'] = '1'
# V30_DATA_DIR fallback to /workspace so config.py doesn't resolve to /v3.0 (LGBM)
os.environ.setdefault('V30_DATA_DIR', '/workspace')
os.environ.setdefault('SAVAGE22_DB_DIR', '/workspace')
os.chdir('/workspace')

TF = sys.argv[sys.argv.index('--tf') + 1] if '--tf' in sys.argv else '1d'

# Min base feature threshold — parquets with fewer cols need rebuild
MIN_BASE_FEATURES = 1000  # 15m has 1,284 base features (correct for intraday)

START = time.time()
FAILURES = []

def elapsed():
    return f"[{time.time()-START:.0f}s]"

def log(msg):
    print(f"{elapsed()} {msg}", flush=True)

def run(cmd, name, critical=True):
    """Run command. If critical=True and it fails, abort entire pipeline."""
    t0 = time.time()
    log(f"=== {name} ===")
    r = subprocess.run(cmd, shell=True)
    dt = time.time() - t0
    ok = r.returncode == 0
    log(f"{name}: {'OK' if ok else 'FAIL'} ({dt:.0f}s)")
    if not ok:
        FAILURES.append(name)
        if critical:
            log(f"*** CRITICAL FAILURE: {name} — aborting ***")
            _print_summary()
            sys.exit(1)
    return ok

def run_tee(cmd, name, logfile, critical=True):
    """Run command with output tee'd to logfile for post-verification."""
    t0 = time.time()
    log(f"=== {name} ===")
    # pipefail ensures pipe returns the python exit code, not tee's
    full_cmd = f'set -o pipefail && {{ {cmd} ; }} 2>&1 | tee -a {logfile}'
    r = subprocess.run(['bash', '-c', full_cmd])
    dt = time.time() - t0
    ok = r.returncode == 0
    log(f"{name}: {'OK' if ok else 'FAIL'} ({dt:.0f}s)")
    if not ok:
        FAILURES.append(name)
        if critical:
            log(f"*** CRITICAL FAILURE: {name} — aborting ***")
            _print_summary()
            sys.exit(1)
    return ok

def _print_summary():
    elapsed_total = time.time() - START
    print(f"\n{'='*60}", flush=True)
    if FAILURES:
        print(f"  PIPELINE FAILED: {TF} ({elapsed_total:.0f}s / {elapsed_total/60:.1f} min)", flush=True)
        print(f"  Failures: {', '.join(FAILURES)}", flush=True)
    else:
        print(f"  PIPELINE COMPLETE: {TF} ({elapsed_total:.0f}s / {elapsed_total/60:.1f} min)", flush=True)
    print(f"{'='*60}", flush=True)
    # List all artifacts
    artifacts = [
        f'model_{TF}.json', f'optuna_configs_{TF}.json', f'exhaustive_configs_{TF}.json',
        f'meta_model_{TF}.pkl', f'platt_{TF}.pkl', f'lstm_{TF}.pt',
        f'features_{TF}_all.json', f'cpcv_oos_predictions_{TF}.pkl',
        f'v2_crosses_BTC_{TF}.npz', f'v2_cross_names_BTC_{TF}.json',
        f'features_BTC_{TF}.parquet', f'feature_importance_top500_{TF}.json',
        f'feature_importance_summary.json',
    ]
    print("  Artifacts:", flush=True)
    for a in artifacts:
        if os.path.exists(a):
            sz = os.path.getsize(a) / (1024*1024)
            print(f"    {sz:8.1f} MB  {a}", flush=True)
        else:
            print(f"    MISSING     {a}", flush=True)


# ============================================================
# HEADER
# ============================================================
try:
    ram_gb = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024**3)
except (ValueError, OSError):
    ram_gb = 0
print(f"{'='*60}", flush=True)
print(f"  CLOUD PIPELINE: {TF}", flush=True)
print(f"  Cores: {os.cpu_count()}", flush=True)
print(f"  RAM:   {ram_gb:.0f} GB", flush=True)
print(f"  CWD:   {os.getcwd()}", flush=True)
print(f"{'='*60}", flush=True)

# ============================================================
# STEP 0: Kill stale python, install deps
# ============================================================
# Kill stale pipeline processes (NOT this script — exclude own PID)
_my_pid = os.getpid()
os.system(f'pgrep -f "python.*(ml_multi_tf|cross_generator|optuna|exhaustive|meta_label|lstm_seq|backtest|backtesting|build_.*features)" | grep -v {_my_pid} | xargs -r kill -9 2>/dev/null; true')
time.sleep(1)

run('pip install -q lightgbm optuna scipy numba scikit-learn pandas pyarrow psutil hmmlearn torch 2>&1 | tail -5',
    'Install deps')

# ============================================================
# STEP 1: Fix btc_prices.db symbol format
# ============================================================
log("=== Verify btc_prices.db ===")

if not os.path.exists('btc_prices.db') or os.path.getsize('btc_prices.db') == 0:
    # Try multi_asset_prices.db as fallback
    if os.path.exists('multi_asset_prices.db'):
        log("btc_prices.db missing — copying from multi_asset_prices.db")
        import shutil
        shutil.copy2('multi_asset_prices.db', 'btc_prices.db')
    else:
        log("*** CRITICAL: No price database found! ***")
        sys.exit(1)

# Check symbol format and fix if needed
conn = sqlite3.connect('btc_prices.db')
r_usdt = conn.execute("SELECT COUNT(*) FROM ohlcv WHERE timeframe='1d' AND symbol='BTC/USDT'").fetchone()[0]
r_bare = conn.execute("SELECT COUNT(*) FROM ohlcv WHERE timeframe='1d' AND symbol='BTC'").fetchone()[0]
log(f"  BTC/USDT rows: {r_usdt}, BTC rows: {r_bare}")

if r_usdt == 0 and r_bare > 0:
    log("  Symbol format is 'BTC' — adding '/USDT' suffix for compatibility...")
    conn.execute("UPDATE ohlcv SET symbol = symbol || '/USDT' WHERE symbol NOT LIKE '%/%'")
    conn.commit()
    # Verify fix
    r_check = conn.execute("SELECT COUNT(*) FROM ohlcv WHERE timeframe='1d' AND symbol='BTC/USDT'").fetchone()[0]
    log(f"  After fix: {r_check} daily BTC/USDT rows")
    if r_check == 0:
        log("*** CRITICAL: btc_prices.db symbol fix failed ***")
        conn.close()
        sys.exit(1)
elif r_usdt > 0:
    log(f"  btc_prices.db OK: {r_usdt} daily BTC/USDT rows")
else:
    log("*** CRITICAL: No BTC data in btc_prices.db ***")
    conn.close()
    sys.exit(1)

# Log available timeframes for BTC
tfs = conn.execute(
    "SELECT timeframe, COUNT(*) FROM ohlcv WHERE symbol='BTC/USDT' GROUP BY timeframe ORDER BY timeframe"
).fetchall()
for tf_name, cnt in tfs:
    log(f"    {tf_name}: {cnt} rows")
conn.close()

# ============================================================
# STEP 2: Rebuild features if parquet missing or incomplete
# ============================================================
parquet_path = f'features_BTC_{TF}.parquet'
need_rebuild = False

if not os.path.exists(parquet_path):
    # Also check non-BTC name (some build scripts save without symbol)
    alt_path = f'features_{TF}.parquet'
    if os.path.exists(alt_path):
        os.rename(alt_path, parquet_path)
        log(f"Renamed {alt_path} → {parquet_path}")
    else:
        log(f"Parquet {parquet_path} not found — need rebuild")
        need_rebuild = True

if not need_rebuild and os.path.exists(parquet_path):
    import pandas as pd
    pq = pd.read_parquet(parquet_path)
    n_cols = pq.shape[1]
    n_rows = pq.shape[0]
    log(f"Parquet check: {n_rows} rows x {n_cols} cols")
    del pq
    if n_cols < MIN_BASE_FEATURES:
        log(f"  Only {n_cols} cols (need {MIN_BASE_FEATURES}+) — need rebuild")
        need_rebuild = True
    else:
        log(f"  Parquet OK: {n_cols} base features")

if need_rebuild:
    build_script = f'build_{TF}_features.py'
    if not os.path.exists(build_script):
        # Try build_features_complete.py as fallback
        build_script = 'build_features_complete.py'
    if not os.path.exists(build_script):
        log(f"*** CRITICAL: No build script for {TF} ***")
        sys.exit(1)

    log(f"Rebuilding {TF} features using {build_script}...")
    run_tee(f'python -X utf8 -u {build_script}',
            f'Rebuild {TF} features', f'rebuild_{TF}.log')

    # The build script saves as features_{TF}.parquet — rename to features_BTC_{TF}.parquet
    if not os.path.exists(parquet_path):
        alt = f'features_{TF}.parquet'
        if os.path.exists(alt):
            os.rename(alt, parquet_path)
            log(f"Renamed {alt} → {parquet_path}")
        else:
            log(f"*** CRITICAL: Feature rebuild produced no parquet ***")
            sys.exit(1)

    # Verify rebuilt parquet
    import pandas as pd
    pq = pd.read_parquet(parquet_path)
    log(f"Rebuilt parquet: {pq.shape[0]} rows x {pq.shape[1]} cols")
    if pq.shape[1] < MIN_BASE_FEATURES:
        log(f"*** CRITICAL: Rebuilt parquet still only {pq.shape[1]} cols ***")
        sys.exit(1)
    del pq

# Create non-BTC symlink for scripts that look for features_{tf}.parquet (e.g. LSTM)
plain_pq = f'features_{TF}.parquet'
if os.path.exists(parquet_path) and not os.path.exists(plain_pq):
    os.symlink(parquet_path, plain_pq)
    log(f"Symlinked {plain_pq} → {parquet_path}")

# ============================================================
# STEP 3: Build crosses (skip if NPZ already exists)
# ============================================================
npz_path = f'v2_crosses_BTC_{TF}.npz'
if os.path.exists(npz_path) and os.path.getsize(npz_path) > 1000:
    npz_size = os.path.getsize(npz_path) / (1024*1024)
    log(f"Cross NPZ already exists ({npz_size:.1f} MB) — SKIPPING cross gen")
else:
    run_tee(f'python -X utf8 -u v2_cross_generator.py --tf {TF} --symbol BTC --save-sparse',
            f'Build {TF} crosses', f'cross_{TF}.log')
    if not os.path.exists(npz_path):
        log(f"*** CRITICAL: {npz_path} not created by cross generator ***")
        sys.exit(1)
    npz_size = os.path.getsize(npz_path) / (1024*1024)
    log(f"Cross NPZ: {npz_size:.1f} MB")

# ============================================================
# STEP 4: Train — MUST produce SPARSE output
# ============================================================
train_log = f'train_{TF}.log'
run_tee(f'python -X utf8 -u ml_multi_tf.py --tf {TF}',
        f'Train {TF}', train_log)

# CRITICAL VERIFICATION: Check for SPARSE in training log
log("=== SPARSE VERIFICATION ===")
sparse_found = False
dense_found = False
combined_line = ""
if os.path.exists(train_log):
    with open(train_log, 'r', errors='replace') as f:
        for line in f:
            if 'SPARSE' in line and 'Features:' in line:
                sparse_found = True
                log(f"  VERIFIED: {line.strip()}")
            if 'DENSE' in line and 'Features:' in line:
                dense_found = True
                log(f"  WARNING: {line.strip()}")
            if 'Combined sparse' in line:
                combined_line = line.strip()

if sparse_found and not dense_found:
    log("  PASS: Training used SPARSE cross features")
    if combined_line:
        log(f"  {combined_line}")
else:
    log("*** CRITICAL: Training did NOT use sparse crosses! ***")
    log("  This means crosses failed to load. Check cross_{TF}.log and train_{TF}.log")
    if dense_found:
        log("  Found DENSE — training ran with base features only (USELESS)")
    sys.exit(1)

# ============================================================
# STEP 5: Optuna — SKIPPED (run later with parquet + NPZ)
# Optuna is fully stateless — can run on any machine anytime
# using just the saved parquet + NPZ cross files.
# ============================================================
log("Optuna SKIPPED — will run later (decoupled from training pipeline)")

# ============================================================
# STEP 6: Exhaustive optimizer — SKIPPED (run with Optuna later)
# ============================================================
log("Exhaustive optimizer SKIPPED — will run later")

# ============================================================
# STEP 7: Meta-labeling
# ============================================================
run_tee(f'python -X utf8 -u meta_labeling.py --tf {TF}',
        f'Meta {TF}', f'meta_{TF}.log', critical=False)

# ============================================================
# STEP 8: LSTM
# ============================================================
run_tee(f'python -X utf8 -u lstm_sequence_model.py --tf {TF} --train',
        f'LSTM {TF}', f'lstm_{TF}.log', critical=False)

# ============================================================
# STEP 9: PBO
# ============================================================
run_tee(f'python -X utf8 -u backtest_validation.py --tf {TF}',
        f'PBO {TF}', f'pbo_{TF}.log', critical=False)

# ============================================================
# STEP 10: Audit
# ============================================================
run_tee(f'python -X utf8 -u backtesting_audit.py --tf {TF}',
        f'Audit {TF}', f'audit_{TF}.log', critical=False)

# ============================================================
# FINAL SUMMARY
# ============================================================
_print_summary()

# Signal completion
with open(f'DONE_{TF}', 'w') as f:
    f.write(f"Completed at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n")
    f.write(f"Total time: {time.time()-START:.0f}s\n")
    f.write(f"Failures: {', '.join(FAILURES) if FAILURES else 'None'}\n")
log(f"Wrote DONE_{TF} marker file")
