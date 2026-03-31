#!/usr/bin/env python3
"""
cloud_run_tf.py — Run full pipeline for ONE timeframe on cloud.

Usage: python -u cloud_run_tf.py --tf 1w

Tar should extract flat to /workspace/ with:
  - All .py, .json, .pkl files from v3.3
  - features_BTC_{tf}.parquet (or will rebuild if missing/incomplete)
  - btc_prices.db (root version with BTC/USDT, or multi_asset_prices.db to be fixed)
  - V1 DBs (needed only if feature rebuild required)
  - kp_history_gfz.txt (needed only if feature rebuild required)

Steps:
  0. Kill stale processes, install deps
  1. Fix btc_prices.db symbol format if needed
  2. Rebuild features if parquet missing or < 2000 cols
  3. Build crosses (v2_cross_generator.py --symbol BTC --save-sparse)
  5. Optuna hyperparameter search (saves optuna_configs_{tf}.json — params only, no model)
  4. Train (ml_multi_tf.py --tf TF) — reads Optuna params, VERIFY crosses loaded (SPARSE or DENSE)
  6-10. Optimizer, meta, LSTM, PBO, audit, SHAP
  11. Verify all artifacts exist
"""
import os, sys, subprocess, time, json, glob, sqlite3, threading

os.environ['PYTHONUNBUFFERED'] = '1'
# FIX #43: PyTorch expandable segments — reduces fragmentation-induced OOM on LSTM/meta steps
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
# CuPy Blackwell (sm_120) compat: PTX JIT lets CUDA 13.0 driver compile for sm_120
os.environ.setdefault('CUPY_COMPILE_WITH_PTX', '1')
# V30_DATA_DIR fallback to /workspace/v3.3 so config.py doesn't resolve to /v3.0 (LGBM)
os.environ.setdefault('V30_DATA_DIR', '/workspace/v3.3')
os.environ.setdefault('SAVAGE22_DB_DIR', '/workspace')
# V1 DBs (tweets, astro, ephemeris, etc.) are in /workspace alongside everything else
os.environ.setdefault('SAVAGE22_V1_DIR', '/workspace')
os.chdir('/workspace/v3.3' if os.path.isdir('/workspace/v3.3') else '/workspace')
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = _SCRIPT_DIR  # v3.3 directory

# ── OPT: Transparent Huge Pages — reduces TLB misses for large sparse matrices ──
try:
    with open('/sys/kernel/mm/transparent_hugepage/enabled', 'w') as f:
        f.write('always')
except (PermissionError, FileNotFoundError, OSError):
    pass  # container may not have permission

# ── OPT: jemalloc — better malloc for sparse alloc/free patterns (reduces fragmentation) ──
_jemalloc_paths = [
    '/usr/lib/x86_64-linux-gnu/libjemalloc.so.2',
    '/usr/lib/x86_64-linux-gnu/libjemalloc.so',
    '/usr/lib/libjemalloc.so.2',
    '/usr/lib/libjemalloc.so',
]
_jemalloc_found = None
for _jp in _jemalloc_paths:
    if os.path.exists(_jp):
        _jemalloc_found = _jp
        break
if _jemalloc_found:
    _existing_preload = os.environ.get('LD_PRELOAD', '')
    if 'jemalloc' not in _existing_preload:
        os.environ['LD_PRELOAD'] = f"{_jemalloc_found}:{_existing_preload}" if _existing_preload else _jemalloc_found
else:
    _jemalloc_found = None  # will log warning after log() is defined

# ── Fix #5: jemalloc tuning — reduce fragmentation for sparse alloc/free patterns ──
if _jemalloc_found:
    os.environ['MALLOC_CONF'] = 'background_thread:true,dirty_decay_ms:1000,muzzy_decay_ms:1000,narenas:32,tcache_max:4096,metadata_thp:auto'

# ── Fix #6: NUMA balancing off — prevents page migration storms during cross gen ──
try:
    with open('/proc/sys/kernel/numa_balancing', 'w') as f:
        f.write('0')
    print("  NUMA auto-balancing: disabled (prevents page migration storms)")
except (PermissionError, FileNotFoundError):
    pass

# ── Fix #7: Dirty page limits + swappiness — keep CSR in RAM ──
for path, val, desc in [
    ('/proc/sys/vm/swappiness', '10', 'swappiness=10 (keep CSR in RAM)'),
    ('/proc/sys/vm/dirty_background_bytes', str(int(1.5e9)), 'dirty_bg=1.5GB'),
    ('/proc/sys/vm/dirty_bytes', str(int(4e9)), 'dirty=4GB'),
]:
    try:
        with open(path, 'w') as f:
            f.write(val)
        print(f"  Kernel: {desc}")
    except (PermissionError, FileNotFoundError):
        pass

def _script(name):
    """Resolve script path — check CWD first, then script directory."""
    if os.path.exists(name):
        return name
    alt = os.path.join(_SCRIPT_DIR, name)
    if os.path.exists(alt):
        return alt
    return name  # let it fail with clear error

TF = sys.argv[sys.argv.index('--tf') + 1] if '--tf' in sys.argv else '1d'
ASSEMBLY_LINE = '--assembly-line' in sys.argv

# ── Assembly-line: TF sequence for prefetching next TF's features ──
_TF_SEQUENCE = ['1w', '1d', '4h', '1h', '15m']
_next_tf_build_thread = None  # Background thread handle for assembly-line overlap
_next_tf_build_ok = None      # Result of background feature build

# --- 15m-specific env vars to prevent OOM during cross gen ---
if TF == '15m':
    os.environ.setdefault('V2_RIGHT_CHUNK', '500')
    os.environ.setdefault('V2_BATCH_MAX', '500')

# Min base feature threshold — parquets with fewer cols need rebuild
# All TFs should have ~2,600-3,400 base features when built with V2 layers.
# A parquet with <2000 cols means V2 layers were NOT applied (old V1 build path).
MIN_BASE_FEATURES = 2000

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
    """Run command with output tee'd to logfile — Python Popen drain, no shell pipe.
    Replaces bash 'tee' pipeline which breaks on long runs (SIGPIPE, buffer saturation)."""
    from pathlib import Path
    t0 = time.time()
    log(f"=== {name} ===")
    Path(logfile).parent.mkdir(parents=True, exist_ok=True)
    with open(logfile, 'wb', buffering=0) as logf:
        proc = subprocess.Popen(
            ['bash', '-c', cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            bufsize=0,
        )
        try:
            while True:
                chunk = proc.stdout.read(64 * 1024)
                if not chunk:
                    break
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                logf.write(chunk)
        finally:
            if proc.stdout:
                proc.stdout.close()
        rc = proc.wait()
    dt = time.time() - t0
    ok = rc == 0
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
    # Assembly-line: report background build status if active
    if _next_tf_build_thread is not None:
        if _next_tf_build_thread.is_alive():
            print(f"  [ASSEMBLY-LINE] Background feature build still running for next TF", flush=True)
        elif _next_tf_build_ok is not None:
            _status = 'OK' if _next_tf_build_ok else 'FAILED'
            print(f"  [ASSEMBLY-LINE] Next TF feature build: {_status}", flush=True)
    # List all artifacts
    artifacts = [
        f'model_{TF}.json', f'model_{TF}_cpcv_backup.json',
        f'optuna_configs_{TF}.json',
        f'meta_model_{TF}.pkl', f'platt_{TF}.pkl', f'lstm_{TF}.pt',
        f'features_{TF}_all.json', f'cpcv_oos_predictions_{TF}.pkl',
        f'v2_crosses_BTC_{TF}.npz', f'v2_cross_names_BTC_{TF}.json',
        f'features_BTC_{TF}.parquet',
        f'feature_importance_stability_{TF}.json',
        f'shap_analysis_{TF}.json',
        'ml_multi_tf_configs.json',
        # Inference cross artifacts (for live trading)
        f'inference_{TF}_thresholds.json', f'inference_{TF}_cross_pairs.npz',
        f'inference_{TF}_ctx_names.json', f'inference_{TF}_base_cols.json',
        f'inference_{TF}_cross_names.json',
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
    from hardware_detect import get_available_ram_gb, get_cpu_count
    ram_gb = get_available_ram_gb()
    cpu_count = get_cpu_count()
except ImportError:
    try:
        ram_gb = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024**3)
    except (ValueError, OSError):
        ram_gb = 0
    cpu_count = os.cpu_count() or 1
# Cloud prerequisite: verify we get the bulk of host RAM (not a tiny cgroup slice)
if os.path.exists('/proc/meminfo'):
    try:
        _host_ram_gb = 0
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    _host_ram_gb = int(line.split()[1]) / (1024 * 1024)
                    break
        if _host_ram_gb > 0:
            _ram_pct = (ram_gb / _host_ram_gb) * 100 if _host_ram_gb > 0 else 100
            if ram_gb < _host_ram_gb * 0.5:
                print(f"  WARNING: cgroup reports {ram_gb:.0f}GB ({_ram_pct:.0f}%) of {_host_ram_gb:.0f}GB host RAM", flush=True)
                print(f"  Using host RAM value ({_host_ram_gb:.0f}GB) — cgroup detection may be inaccurate", flush=True)
                ram_gb = _host_ram_gb
            elif ram_gb < _host_ram_gb:
                print(f"  RAM: {ram_gb:.0f}GB cgroup / {_host_ram_gb:.0f}GB host ({_ram_pct:.0f}%)", flush=True)
                ram_gb = _host_ram_gb  # use host value for min-RAM checks
    except Exception:
        pass
print(f"{'='*60}", flush=True)
print(f"  CLOUD PIPELINE: {TF}", flush=True)
print(f"  Cores: {cpu_count} (cgroup-aware)", flush=True)
print(f"  RAM:   {ram_gb:.0f} GB (cgroup-aware)", flush=True)
print(f"  CWD:   {os.getcwd()}", flush=True)
if _jemalloc_found:
    print(f"  jemalloc: {_jemalloc_found} (LD_PRELOAD active)", flush=True)
else:
    print(f"  jemalloc: NOT FOUND — using system malloc (install libjemalloc-dev for less fragmentation)", flush=True)
print(f"{'='*60}", flush=True)

# --- RAM validation per TF ---
TF_MIN_RAM = {'1w': 64, '1d': 128, '4h': 256, '1h': 512, '15m': 768}  # Reduced: targeted crossing + int8 cut RAM 50-70%
if ram_gb > 0 and ram_gb < TF_MIN_RAM.get(TF, 64):
    print(f"ERROR: {TF} needs {TF_MIN_RAM[TF]}GB RAM, only {ram_gb:.0f}GB available", flush=True)
    sys.exit(1)
else:
    print(f"  RAM check: {ram_gb:.0f} GB >= {TF_MIN_RAM.get(TF, 64)} GB required for {TF} — OK", flush=True)

# ============================================================
# PRE-FLIGHT 0: Deployment verification (catches stale files, wrong binaries, missing DBs)
# ============================================================
run(f'{sys.executable} deploy_verify.py --tf {TF}', 'Deployment verification')

# ============================================================
# PRE-FLIGHT 1: Deterministic validation (catches config bugs before $$ is spent)
# ============================================================
run(f'{sys.executable} validate.py --tf {TF} --cloud', 'Pre-flight validation')

# ============================================================
# STEP 0: Kill stale python, install deps
# ============================================================
# Kill stale pipeline processes (NOT this script — exclude own PID)
_my_pid = os.getpid()
os.system(f'pgrep -f "python.*(ml_multi_tf|cross_generator|optuna|exhaustive|meta_label|lstm_seq|backtest|backtesting|build_.*features)" | grep -v {_my_pid} | xargs -r kill -9 2>/dev/null; true')
time.sleep(1)

# Install ALL dependencies — works on any base image (pytorch, ubuntu, etc.).
run('pip install -q lightgbm scikit-learn scipy ephem astropy pytz joblib '
    '"pandas<3.0" "numpy<2.3" pyarrow optuna hmmlearn numba tqdm pyyaml '
    'alembic cmaes colorlog sqlalchemy threadpoolctl sparse-dot-mkl 2>&1 | tail -5',
    'Install deps')

# OPT-14: Install numactl for NUMA-aware process binding on multi-socket cloud machines
run('apt-get install -y -qq numactl 2>/dev/null || true', 'Install numactl', critical=False)

# --- Stale artifact nuclear clean (delete old-version features/crosses) ---
log("=== Stale artifact cleanup ===")
_stale_patterns = ['features_*_all.json', 'v2_cross_names_*.json', 'v2_crosses_*.npz']  # NPZ+JSON must be deleted together
_stale_count = 0
_kept_count = 0
for _pat in _stale_patterns:
    for _stale in glob.glob(os.path.join(PROJECT_DIR, _pat)):
        # Keep current TF's artifacts — they may represent hours of cross gen work
        _basename = os.path.basename(_stale)
        if f'_{TF}.' in _basename or f'_BTC_{TF}.' in _basename:
            log(f"  Keeping current TF artifact: {_basename}")
            _kept_count += 1
            continue
        log(f"  Removing stale artifact: {_basename}")
        os.remove(_stale)
        _stale_count += 1
if _stale_count:
    log(f"  Removed {_stale_count} stale artifacts from previous runs")
if _kept_count:
    log(f"  Preserved {_kept_count} artifacts for current TF ({TF})")
if not _stale_count and not _kept_count:
    log(f"  No stale artifacts found")

# --- FIX 24: Lockfile — prevent duplicate pipeline runs for same TF ---
_lockfile = f'/tmp/cloud_run_{TF}.lock'
if os.path.exists(_lockfile):
    # Check if the process that created it is still alive
    try:
        _lock_pid = int(open(_lockfile).read().strip())
        os.kill(_lock_pid, 0)  # Check if PID exists
        log(f"*** ABORT: Another pipeline running for {TF} (PID {_lock_pid}) ***")
        sys.exit(1)
    except (ValueError, ProcessLookupError, OSError):
        log(f"  Stale lockfile found — previous run crashed. Removing.")
        os.remove(_lockfile)
with open(_lockfile, 'w') as f:
    f.write(str(os.getpid()))

# --- Lockfile cleanup on exit (normal, crash, or signal) ---
import atexit, signal as _signal
def _cleanup_lock():
    try: os.remove(_lockfile)
    except: pass
atexit.register(_cleanup_lock)
_signal.signal(_signal.SIGTERM, lambda *a: (_cleanup_lock(), sys.exit(0)))

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

# --- FIX 25: Disk space check before feature rebuild / cross gen ---
import shutil
_disk = shutil.disk_usage('.')
_free_gb = _disk.free / (1024**3)
if _free_gb < 20:
    log(f"*** ABORT: Only {_free_gb:.1f} GB free disk space (need 20+) ***")
    sys.exit(1)
log(f"Disk space OK: {_free_gb:.1f} GB free")

# --- 16-DB verification at startup ---
_REQUIRED_DBS = [
    'btc_prices.db', 'tweets.db', 'news_articles.db', 'sports_results.db',
    'space_weather.db', 'onchain_data.db', 'macro_data.db', 'astrology_full.db',
    'ephemeris_cache.db', 'fear_greed.db', 'funding_rates.db', 'google_trends.db',
    'open_interest.db', 'multi_asset_prices.db', 'llm_cache.db', 'v2_signals.db',
]
log("=== Database verification ===")
_db_missing = []
for _db in _REQUIRED_DBS:
    # Check both CWD and /workspace
    _found = os.path.exists(_db) or os.path.exists(os.path.join('/workspace', _db))
    _status = 'PASS' if _found else 'FAIL'
    if not _found:
        _db_missing.append(_db)
    log(f"  {_status}: {_db}")
if _db_missing:
    log(f"*** CRITICAL: {len(_db_missing)} required databases missing: {', '.join(_db_missing)} ***")
    log(f"  Upload ALL .db files before running pipeline. See CLOUD_TRAINING_PROTOCOL.md")
    sys.exit(1)
log(f"  All {len(_REQUIRED_DBS)} databases present — OK")

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
    pq_cols = set(pq.columns)
    log(f"Parquet check: {n_rows} rows x {n_cols} cols")
    del pq
    if n_cols < MIN_BASE_FEATURES:
        log(f"  Only {n_cols} cols (need {MIN_BASE_FEATURES}+) — need rebuild")
        need_rebuild = True
    else:
        # V3.3 FEATURE FINGERPRINT: detect stale parquets missing new features.
        # If feature_library.py was updated with new features but parquet was built
        # with old code, the parquet is stale and must be rebuilt.
        _V33_FINGERPRINT_COLS = [
            'vortex_family_group',     # compute_vortex_sacred_geometry_features
            'mars_speed',              # compute_planetary_expansion_features (get_planetary_speeds)
            'moon_distance_norm',      # compute_lunar_electromagnetic_features
            'loshu_row',               # compute_numerology_expansion_features
        ]
        _missing_v33 = [c for c in _V33_FINGERPRINT_COLS if c not in pq_cols]
        if _missing_v33:
            log(f"  STALE PARQUET: missing v3.3 features: {_missing_v33}")
            log(f"  Parquet was built with old feature_library.py — forcing rebuild")
            need_rebuild = True
        else:
            log(f"  Parquet OK: {n_cols} base features (v3.3 fingerprint verified)")

if need_rebuild:
    # Prefer build_features_v2.py (includes V2 layers: 4-tier binarization, entropy,
    # hurst, fib levels, moon signs, aspects, extra lags, etc.)
    # The old build_{TF}_features.py scripts only call build_all_features() without
    # V2 layers, producing ~1,200 base features instead of ~3,000+.
    # Look for build scripts in both CWD and the script's own directory
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    def _find_script(name):
        if os.path.exists(name):
            return name
        alt = os.path.join(_script_dir, name)
        if os.path.exists(alt):
            return alt
        return None

    _v2 = _find_script('build_features_v2.py')
    if _v2:
        build_script = _v2
        build_cmd = f'python -X utf8 -u {build_script} --symbol BTC --tf {TF}'
    else:
        build_script = _find_script(f'build_{TF}_features.py')
        if not build_script:
            build_script = _find_script('build_features_complete.py')
        if not build_script:
            log(f"*** CRITICAL: No build script for {TF} ***")
            sys.exit(1)
        build_cmd = f'python -X utf8 -u {build_script}'

    log(f"Rebuilding {TF} features using {build_script}...")
    run_tee(build_cmd,
            f'Rebuild {TF} features', f'rebuild_{TF}.log')

    # The build script may save to _SCRIPT_DIR instead of CWD — check both locations
    if not os.path.exists(parquet_path):
        # Check all possible locations
        candidates = [
            f'features_{TF}.parquet',
            os.path.join(_SCRIPT_DIR, f'features_BTC_{TF}.parquet'),
            os.path.join(_SCRIPT_DIR, f'features_{TF}.parquet'),
        ]
        found = None
        for alt in candidates:
            if os.path.exists(alt):
                found = alt
                break
        if found:
            if found != parquet_path:
                os.symlink(os.path.abspath(found), parquet_path)
                log(f"Symlinked {parquet_path} → {found}")
        else:
            log(f"*** CRITICAL: Feature rebuild produced no parquet ***")
            log(f"  Checked: {parquet_path}, {', '.join(candidates)}")
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

# --- FIX 25: Disk space check before cross gen ---
_disk = shutil.disk_usage('.')
_free_gb = _disk.free / (1024**3)
if _free_gb < 20:
    log(f"*** ABORT: Only {_free_gb:.1f} GB free disk space (need 20+) ***")
    sys.exit(1)

# ============================================================
# STEP 3: Build crosses (skip if NPZ already exists)
# ============================================================
# Per-TF cross feature toggle: 1w has too few rows (1158) for 2.8M crosses to be meaningful.
# Base features alone (TA + esoteric + astro + gematria + numerology) give better signal on 1w.
# The matrix thesis scales with DATA — more rows = more crosses add value.
SKIP_CROSSES_TFS = {'1w'}  # Base features only — no cross gen
if TF in SKIP_CROSSES_TFS:
    log(f"  SKIP CROSSES for {TF} — base features only (too few rows for cross features to add signal)")
    log(f"  Matrix signal comes from base feature diversity on {TF}")

# --- Dynamic OMP/NUMBA for cross gen phase ---
# Full core count for both OMP (MKL SpGEMM) and NUMBA (prange kernels).
# Thread exhaustion is prevented by threadpoolctl scoping in _mkl_dot(), not
# by global throttling. MKL_DYNAMIC=FALSE prevents MKL from auto-shrinking.
os.environ['OMP_NUM_THREADS'] = str(cpu_count)
os.environ['NUMBA_NUM_THREADS'] = str(cpu_count)
os.environ['MKL_DYNAMIC'] = 'FALSE'
os.environ.setdefault('OMP_PROC_BIND', 'spread')
os.environ.setdefault('OMP_PLACES', 'cores')
os.environ.setdefault('OMP_SCHEDULE', 'static')
log(f"Cross gen phase: OMP_NUM_THREADS={cpu_count}, NUMBA_NUM_THREADS={cpu_count}, MKL_DYNAMIC=FALSE (all {cpu_count} cores active)")

npz_path = f'v2_crosses_BTC_{TF}.npz'
# Check both CWD and _SCRIPT_DIR for existing NPZ
if not os.path.exists(npz_path):
    alt_npz = os.path.join(_SCRIPT_DIR, npz_path)
    if os.path.exists(alt_npz):
        os.symlink(os.path.abspath(alt_npz), npz_path)
        log(f"Symlinked {npz_path} → {alt_npz}")
    # Also symlink cross names
    cn = f'v2_cross_names_BTC_{TF}.json'
    alt_cn = os.path.join(_SCRIPT_DIR, cn)
    if not os.path.exists(cn) and os.path.exists(alt_cn):
        os.symlink(os.path.abspath(alt_cn), cn)

_npz_valid = False
_skip_crosses = TF in SKIP_CROSSES_TFS
if _skip_crosses:
    _npz_valid = True  # pretend NPZ exists — training will use base features only
    log(f"  Crosses DISABLED for {TF} — training on base features only")
elif os.path.exists(npz_path) and os.path.getsize(npz_path) > 1000:
    npz_size = os.path.getsize(npz_path) / (1024*1024)
    # Validate NPZ col count — stale NPZs from v3.0 (min_nonzero=8) have far fewer crosses
    _MIN_CROSS_COLS = {'1w': 500_000, '1d': 1_000_000, '4h': 1_000_000, '1h': 1_000_000, '15m': 1_000_000}
    _min_cols = _MIN_CROSS_COLS.get(TF, 1_000_000)
    try:
        from scipy import sparse as _sp
        _npz_shape = _sp.load_npz(npz_path).shape
        if _npz_shape[1] >= _min_cols:
            _npz_valid = True
            log(f"Cross NPZ valid ({npz_size:.1f} MB, {_npz_shape[1]:,} cols >= {_min_cols:,} min) — SKIPPING cross gen")
        else:
            log(f"Cross NPZ STALE ({_npz_shape[1]:,} cols < {_min_cols:,} min) — will rebuild")
            os.remove(npz_path)
            cn_path = npz_path.replace('v2_crosses_', 'v2_cross_names_').replace('.npz', '.json')
            if os.path.exists(cn_path):
                os.remove(cn_path)
                log(f"  Removed stale cross names: {cn_path}")
    except Exception as _e:
        log(f"  NPZ validation failed ({_e}) — will rebuild")
if _npz_valid:
    pass  # NPZ valid, skip cross gen
else:
    run_tee(f'python -X utf8 -u {_script("v2_cross_generator.py")} --tf {TF} --symbol BTC --save-sparse',
            f'Build {TF} crosses', f'cross_{TF}.log')
    # Check both locations for output
    if not os.path.exists(npz_path):
        alt_npz = os.path.join(_SCRIPT_DIR, npz_path)
        if os.path.exists(alt_npz):
            os.symlink(os.path.abspath(alt_npz), npz_path)
            log(f"Symlinked {npz_path} → {alt_npz}")
            cn = f'v2_cross_names_BTC_{TF}.json'
            alt_cn = os.path.join(_SCRIPT_DIR, cn)
            if not os.path.exists(cn) and os.path.exists(alt_cn):
                os.symlink(os.path.abspath(alt_cn), cn)
        else:
            log(f"*** CRITICAL: {npz_path} not created by cross generator ***")
            sys.exit(1)
    npz_size = os.path.getsize(npz_path) / (1024*1024)
    log(f"Cross NPZ: {npz_size:.1f} MB")

# Verify inference artifacts were created by cross generator
_inf_artifacts = [f'inference_{TF}_thresholds.json', f'inference_{TF}_cross_pairs.npz',
                  f'inference_{TF}_ctx_names.json', f'inference_{TF}_base_cols.json',
                  f'inference_{TF}_cross_names.json']
_inf_missing = [a for a in _inf_artifacts if not os.path.exists(a)]
if _inf_missing:
    log(f"  WARNING: Missing inference artifacts: {', '.join(_inf_missing)}")
    log(f"  Live trading will not have cross features for {TF}")
else:
    log(f"  Inference artifacts OK: all {len(_inf_artifacts)} files present")

# ============================================================
# STEP 5: Optuna hyperparameter search (BEFORE training)
# Saves optuna_configs_{tf}.json with best params — does NOT save a production model.
# Step 4 reads optuna_configs_{tf}.json and uses those params for the real CPCV training.
# ============================================================
# --- FIX #45: Reset thread pools between phases to prevent MKL/OpenMP/Numba conflicts ---
# Cross gen sets OMP_NUM_THREADS=cpu_count + MKL_DYNAMIC=FALSE. Training needs a clean slate.
os.environ.pop('OMP_NUM_THREADS', None)
os.environ.pop('NUMBA_NUM_THREADS', None)
os.environ.pop('MKL_DYNAMIC', None)
os.environ.pop('OMP_PROC_BIND', None)
os.environ.pop('OMP_PLACES', None)
os.environ.pop('OMP_SCHEDULE', None)
# OPT: Re-set OpenMP tuning for training phase (spread threads across cores for LightGBM)
os.environ.setdefault('OMP_PROC_BIND', 'spread')
os.environ.setdefault('OMP_PLACES', 'cores')
os.environ.setdefault('OMP_SCHEDULE', 'static')
try:
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=cpu_count, user_api='blas')
    threadpool_limits(limits=cpu_count, user_api='openmp')
    log(f"Training phase: thread pools reset to {cpu_count} cores (blas+openmp), OMP_PROC_BIND=spread")
except ImportError:
    log(f"Training phase: threadpoolctl not available, env vars cleared")

# OPT-14: NUMA-aware process binding for multi-socket cloud machines
# Detect NUMA topology and bind training to node 0 for memory locality
_NUMA_PREFIX = ''
try:
    _numa_out = subprocess.run(['numactl', '--hardware'], capture_output=True, text=True, timeout=5)
    if _numa_out.returncode == 0:
        _numa_nodes = [l for l in _numa_out.stdout.splitlines() if l.startswith('node') and 'cpus:' in l]
        _n_numa_nodes = len(_numa_nodes)
        if _n_numa_nodes > 1:
            # Test if numactl actually works (containers may lack SYS_NICE capability)
            _numa_test = subprocess.run(['numactl', '--interleave=all', 'true'], capture_output=True, timeout=5)
            if _numa_test.returncode == 0:
                log(f"NUMA: {_n_numa_nodes} nodes detected — using memory interleave across all nodes")
                _NUMA_PREFIX = 'numactl --interleave=all '
            else:
                log(f"NUMA: {_n_numa_nodes} nodes detected but numactl lacks permission (container?) — skipping")
            for _nl in _numa_nodes:
                log(f"  {_nl.strip()}")
        else:
            log(f"NUMA: single node — no binding needed")
    else:
        log(f"NUMA: numactl not available or failed")
except (FileNotFoundError, subprocess.TimeoutExpired):
    log(f"NUMA: numactl not installed — skipping NUMA binding")

# ============================================================
# ASSEMBLY-LINE: Prefetch next TF's features in background thread
# Feature builds are CPU-bound, Optuna/training are GPU-bound — no contention.
# Only overlaps feature BUILD (not cross gen, which needs GPU).
# ============================================================
def _assembly_line_build_next_tf():
    """Build features for the next TF in a background thread.
    Sets global _next_tf_build_ok with the result."""
    global _next_tf_build_ok
    try:
        _idx = _TF_SEQUENCE.index(TF)
    except ValueError:
        _next_tf_build_ok = False
        return
    if _idx + 1 >= len(_TF_SEQUENCE):
        return  # Last TF, nothing to prefetch
    next_tf = _TF_SEQUENCE[_idx + 1]
    next_parquet = f'features_BTC_{next_tf}.parquet'

    # Check if already built
    if os.path.exists(next_parquet):
        import pandas as _pd_check
        try:
            _ncols = _pd_check.read_parquet(next_parquet, columns=None).shape[1]
            if _ncols >= MIN_BASE_FEATURES:
                log(f"  [ASSEMBLY-LINE] {next_tf} features already exist ({_ncols} cols) — skip prefetch")
                _next_tf_build_ok = True
                return
            else:
                log(f"  [ASSEMBLY-LINE] {next_tf} parquet stale ({_ncols} cols) — rebuilding")
        except Exception:
            log(f"  [ASSEMBLY-LINE] {next_tf} parquet unreadable — rebuilding")

    # Find the build script
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    def _find(name):
        if os.path.exists(name):
            return name
        alt = os.path.join(_script_dir, name)
        if os.path.exists(alt):
            return alt
        return None

    _v2 = _find('build_features_v2.py')
    if _v2:
        cmd = f'python -X utf8 -u {_v2} --symbol BTC --tf {next_tf}'
    else:
        _alt = _find(f'build_{next_tf}_features.py') or _find('build_features_complete.py')
        if not _alt:
            log(f"  [ASSEMBLY-LINE] No build script for {next_tf} — cannot prefetch")
            _next_tf_build_ok = False
            return
        cmd = f'python -X utf8 -u {_alt}'

    log(f"  [ASSEMBLY-LINE] Starting background feature build: {next_tf}")
    t0 = time.time()
    logfile = f'assembly_build_{next_tf}.log'
    try:
        with open(logfile, 'w') as lf:
            proc = subprocess.Popen(
                cmd, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                cwd=os.getcwd(),
                env={**os.environ, 'PYTHONUNBUFFERED': '1'},
                bufsize=1, universal_newlines=True,
            )
            for line in proc.stdout:
                lf.write(line)
                lf.flush()
            proc.wait()
        dt = time.time() - t0
        if proc.returncode == 0:
            log(f"  [ASSEMBLY-LINE] {next_tf} feature build OK ({dt:.0f}s) — see {logfile}")
            _next_tf_build_ok = True
        else:
            log(f"  [ASSEMBLY-LINE] {next_tf} feature build FAILED (exit {proc.returncode}, {dt:.0f}s) — see {logfile}")
            _next_tf_build_ok = False
    except Exception as e:
        log(f"  [ASSEMBLY-LINE] {next_tf} feature build ERROR: {e}")
        _next_tf_build_ok = False

if ASSEMBLY_LINE:
    try:
        _idx = _TF_SEQUENCE.index(TF)
        if _idx + 1 < len(_TF_SEQUENCE):
            _next_tf_name = _TF_SEQUENCE[_idx + 1]
            log(f"[ASSEMBLY-LINE] Will prefetch {_next_tf_name} features while Optuna/training runs")
            _next_tf_build_thread = threading.Thread(
                target=_assembly_line_build_next_tf,
                name=f'assembly-build-{_next_tf_name}',
                daemon=True,
            )
            _next_tf_build_thread.start()
        else:
            log(f"[ASSEMBLY-LINE] {TF} is the last TF in sequence — nothing to prefetch")
    except ValueError:
        log(f"[ASSEMBLY-LINE] {TF} not in sequence {_TF_SEQUENCE} — skipping prefetch")

optuna_config_path = f'optuna_configs_{TF}.json'
if os.path.exists(optuna_config_path):
    log(f"Optuna config already exists ({optuna_config_path}) — skipping search")
else:
    run_tee(f'{_NUMA_PREFIX}python -X utf8 -u {_script("run_optuna_local.py")} --tf {TF}',
            f'Optuna search {TF}', f'optuna_{TF}.log', critical=False)
    if os.path.exists(optuna_config_path):
        import json as _json_step5
        with open(optuna_config_path) as _f5:
            _oc = _json_step5.load(_f5)
        log(f"Optuna search complete: accuracy={_oc.get('final_mean_accuracy', 'N/A')}, "
            f"sortino={_oc.get('final_mean_sortino', 'N/A')}")
        log(f"Best params: {_oc.get('best_params', {})}")
    else:
        log(f"WARNING: Optuna search did not produce {optuna_config_path} — Step 4 will use config.py defaults")

# ============================================================
# STEP 4: Train — MUST produce SPARSE output
# ============================================================

# Clean stale CPCV checkpoint — prevents resuming from a previous run's completed folds
# which would skip all training and produce results from old/different data
_cpcv_ckpt = f'cpcv_checkpoint_{TF}.pkl'
if os.path.exists(_cpcv_ckpt):
    os.remove(_cpcv_ckpt)
    log(f"  Removed stale CPCV checkpoint: {_cpcv_ckpt}")

train_log = f'train_{TF}.log'
run_tee(f'{_NUMA_PREFIX}python -X utf8 -u {_script("ml_multi_tf.py")} --tf {TF}',
        f'Train {TF}', train_log)

# CRITICAL VERIFICATION: Check that cross features were loaded (SPARSE or DENSE both valid)
log("=== CROSS FEATURE VERIFICATION ===")
crosses_loaded = False
combined_line = ""
if os.path.exists(train_log):
    with open(train_log, 'r', errors='replace') as f:
        for line in f:
            if 'Features:' in line and ('SPARSE' in line or 'DENSE' in line):
                crosses_loaded = True
                log(f"  VERIFIED: {line.strip()}")
            if 'Combined sparse' in line or 'Combined' in line:
                combined_line = line.strip()

if crosses_loaded:
    log("  PASS: Training loaded cross features")
    if combined_line:
        log(f"  {combined_line}")
else:
    log("*** CRITICAL: Training did NOT load cross features! ***")
    log("  Check cross_{TF}.log and train_{TF}.log")
    sys.exit(1)

# CRITICAL: Verify model was actually saved (accuracy floor can silently skip save)
_model_path = f'model_{TF}.json'
if not os.path.exists(_model_path):
    log(f"*** CRITICAL: model_{TF}.json NOT FOUND after training ***")
    log(f"  Training exited OK but model was not saved.")
    log(f"  Most likely cause: final accuracy < 0.40 (accuracy floor).")
    log(f"  Check {train_log} for 'ACCURACY BELOW FLOOR' message.")
    FAILURES.append(f'Model {TF} missing')
    _print_summary()
    sys.exit(1)

# PROTECT Step 4 model from any downstream overwrite
import shutil
_backup_path = f'model_{TF}_cpcv_backup.json'
if os.path.exists(_model_path):
    shutil.copy2(_model_path, _backup_path)
    log(f"  Model backed up: {_backup_path} ({os.path.getsize(_model_path)/1024:.0f} KB)")

# NOTE: Step 5 (Optuna search) now runs BEFORE Step 4. It saves only params (optuna_configs_{tf}.json),
# NOT a model. Step 4 reads those params and uses them for full CPCV training.

# ============================================================
# STEP 6: Exhaustive trade optimizer
# ============================================================
run_tee(f'python -X utf8 -u {_script("exhaustive_optimizer.py")} --tf {TF}',
        f'Optimizer {TF}', f'optimizer_{TF}.log', critical=False)

# ============================================================
# STEPS 7,8,9 — Run in PARALLEL (all depend only on Step 4)
# Step 10 (Audit) runs AFTER — depends on Step 6 optimizer output
# ============================================================
from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed

def _run_step(name, cmd, logfile):
    """Wrapper for parallel step execution."""
    try:
        run_tee(cmd, name, logfile, critical=False)
        return (name, True)
    except Exception as e:
        log(f"  {name} failed: {e}")
        return (name, False)

_parallel_steps = [
    (f'Meta {TF}',  f'python -X utf8 -u {_script("meta_labeling.py")} --tf {TF}',  f'meta_{TF}.log'),
    (f'LSTM {TF}',  f'python -X utf8 -u {_script("lstm_sequence_model.py")} --tf {TF} --train',  f'lstm_{TF}.log'),
    (f'PBO {TF}',   f'python -X utf8 -u {_script("backtest_validation.py")} --tf {TF}',  f'pbo_{TF}.log'),
]

log(f"=== Steps 7,8,9 launching in parallel ({len(_parallel_steps)} tasks) ===")
with ThreadPoolExecutor(max_workers=len(_parallel_steps)) as _step_pool:
    _step_futures = {_step_pool.submit(_run_step, n, c, l): n for n, c, l in _parallel_steps}
    for _sf in _as_completed(_step_futures):
        _sname, _sok = _sf.result()
        log(f"  {_sname}: {'OK' if _sok else 'FAIL'}")

# ============================================================
# STEP 10: Audit (sequential — depends on Step 6 optimizer output)
# ============================================================
run_tee(f'python -X utf8 -u {_script("backtesting_audit.py")} --tf {TF}',
        f'Audit {TF}', f'audit_{TF}.log', critical=False)

# ============================================================
# STEP 11: SHAP Cross Feature Validation (non-fatal)
# ============================================================
# === SHAP Cross Feature Validation ===
log(f"=== SHAP cross feature analysis ===")
try:
    import json
    import numpy as np
    import lightgbm as lgb

    # Load model
    model = lgb.Booster(model_file=f'model_{TF}.json')

    # Get features with non-zero split count (reduces 3.34M -> likely ~50K-200K active)
    all_features = model.feature_name()
    split_scores = dict(zip(all_features, model.feature_importance(importance_type='split')))
    split_importance = split_scores
    active_features = [f for f, v in split_importance.items() if v > 0]
    log(f"  Active features (split > 0): {len(active_features)} / {len(split_importance)}")

    # Count cross features that are active
    cross_prefixes = ('dx_', 'ax_', 'ax2_', 'ta2_', 'ex2_', 'sw_', 'hod_', 'mx_', 'vx_', 'asp_', 'mn_', 'pn_')
    active_crosses = [f for f in active_features if f.startswith(cross_prefixes)]
    active_base = [f for f in active_features if not f.startswith(cross_prefixes)]
    log(f"  Active cross features: {len(active_crosses)}")
    log(f"  Active base features: {len(active_base)}")

    # Use split importance only — pred_contrib + .toarray() OOMs on 2.9M+ sparse crosses
    # Split importance is memory-safe and already shows which cross features contribute
    import pandas as pd

    gain_scores = dict(zip(all_features, model.feature_importance(importance_type='gain')))
    gain_importance = gain_scores

    # Build importance DataFrame
    imp_df = pd.DataFrame([
        {'feature': f, 'splits': split_importance.get(f, 0), 'gain': gain_importance.get(f, 0)}
        for f in all_features
    ])
    imp_df['is_cross'] = imp_df['feature'].str.startswith(cross_prefixes)

    # Family aggregation
    def get_family(name):
        for p in cross_prefixes:
            if name.startswith(p): return p.rstrip('_')
        parts = name.split('_')
        return parts[0] if parts else 'unknown'

    imp_df['family'] = imp_df['feature'].apply(get_family)
    family_imp = imp_df.groupby('family')['gain'].agg(['sum', 'mean', 'count']).sort_values('sum', ascending=False)

    # Report
    log(f"  Top 20 feature families by gain:")
    for i, (fam, row) in enumerate(family_imp.head(20).iterrows()):
        log(f"    {i+1:2d}. {fam:<20s} gain_sum={row['sum']:10.1f}  mean={row['mean']:.2f}  count={int(row['count'])}")

    # Cross vs base comparison
    cross_gain = imp_df[imp_df['is_cross']]['gain'].sum()
    base_gain = imp_df[~imp_df['is_cross']]['gain'].sum()
    cross_pct = 100 * cross_gain / (cross_gain + base_gain) if (cross_gain + base_gain) > 0 else 0
    log(f"  Cross feature gain: {cross_pct:.1f}% of total ({cross_gain:.1f} / {cross_gain + base_gain:.1f})")

    # Save report
    shap_report = {
        'active_features': len(active_features),
        'active_crosses': len(active_crosses),
        'active_base': len(active_base),
        'cross_shap_pct': round(cross_pct, 2),
        'method': 'split_importance (pred_contrib skipped — OOM on sparse crosses)',
        'top_20_families': family_imp.head(20).to_dict(),
        'top_50_features': imp_df.nlargest(50, 'gain')[['feature', 'gain', 'splits', 'is_cross']].to_dict('records'),
    }
    with open(f'shap_analysis_{TF}.json', 'w') as f:
        json.dump(shap_report, f, indent=2, default=str)
    log(f"  Saved: shap_analysis_{TF}.json")

except Exception as e:
    log(f"  SHAP analysis error (non-fatal): {e}")

# ============================================================
# ASSEMBLY-LINE: Wait for background feature build to finish
# ============================================================
if _next_tf_build_thread is not None and _next_tf_build_thread.is_alive():
    _next_tf_name = _TF_SEQUENCE[_TF_SEQUENCE.index(TF) + 1]
    log(f"[ASSEMBLY-LINE] Waiting for {_next_tf_name} feature build to finish...")
    _next_tf_build_thread.join(timeout=7200)  # 2 hour max wait
    if _next_tf_build_thread.is_alive():
        log(f"[ASSEMBLY-LINE] WARNING: {_next_tf_name} build still running after 2h timeout — proceeding")
    elif _next_tf_build_ok:
        log(f"[ASSEMBLY-LINE] {_next_tf_name} features ready for next pipeline run")
    else:
        log(f"[ASSEMBLY-LINE] {_next_tf_name} feature build failed — next run will rebuild")

# ============================================================
# FINAL SUMMARY
# ============================================================
_print_summary()

# Signal completion — only write DONE marker if zero failures
if len(FAILURES) == 0:
    with open(f'DONE_{TF}', 'w') as f:
        f.write(f"Completed at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n")
        f.write(f"Total time: {time.time()-START:.0f}s\n")
        f.write(f"Failures: None\n")
    log(f"Wrote DONE_{TF} marker file")
else:
    log(f"*** NOT writing DONE_{TF} — {len(FAILURES)} failures: {', '.join(FAILURES)} ***")

# --- FIX 24: Remove lockfile on clean exit ---
if os.path.exists(_lockfile):
    os.remove(_lockfile)
