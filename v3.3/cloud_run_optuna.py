#!/usr/bin/env python3
"""
cloud_run_optuna.py — Run ONLY Optuna hyperparameter search for one TF.

Expects parquet + NPZ + btc_prices.db already in /workspace/.
Does NOT build crosses or train base model — those are already done.

Usage: python -u cloud_run_optuna.py --tf 1d
"""
import os, sys, subprocess, time, glob, sqlite3

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ.setdefault('V30_DATA_DIR', '/workspace')
os.environ.setdefault('SAVAGE22_DB_DIR', '/workspace')
os.chdir('/workspace')

TF = sys.argv[sys.argv.index('--tf') + 1] if '--tf' in sys.argv else '1d'
START = time.time()

def log(msg):
    print(f"[{time.time()-START:.0f}s] {msg}", flush=True)

def run_tee(cmd, name, logfile):
    t0 = time.time()
    log(f"=== {name} ===")
    full_cmd = f'set -o pipefail && {{ {cmd} ; }} 2>&1 | tee -a {logfile}'
    r = subprocess.run(['bash', '-c', full_cmd])
    log(f"{name}: {'OK' if r.returncode == 0 else 'FAIL'} ({time.time()-t0:.0f}s)")
    return r.returncode == 0

# Header
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
print(f"{'='*60}", flush=True)
print(f"  OPTUNA SEARCH: {TF}", flush=True)
print(f"  Cores: {cpu_count} (cgroup-aware), RAM: {ram_gb:.0f} GB (cgroup-aware)", flush=True)
print(f"{'='*60}", flush=True)

# Install deps
subprocess.run('pip install -q lightgbm optuna scipy numba scikit-learn pandas pyarrow psutil hmmlearn 2>&1 | tail -3', shell=True)

# Fix btc_prices.db symbol if needed
if os.path.exists('btc_prices.db'):
    conn = sqlite3.connect('btc_prices.db')
    r_usdt = conn.execute("SELECT COUNT(*) FROM ohlcv WHERE timeframe='1d' AND symbol='BTC/USDT'").fetchone()[0]
    r_bare = conn.execute("SELECT COUNT(*) FROM ohlcv WHERE timeframe='1d' AND symbol='BTC'").fetchone()[0]
    if r_usdt == 0 and r_bare > 0:
        log("Fixing btc_prices.db symbol format (BTC → BTC/USDT)...")
        conn.execute("UPDATE ohlcv SET symbol = symbol || '/USDT' WHERE symbol NOT LIKE '%/%'")
        conn.commit()
    conn.close()

# Create symlinks for LSTM compatibility
parquet_path = f'features_BTC_{TF}.parquet'
plain_pq = f'features_{TF}.parquet'
if os.path.exists(parquet_path) and not os.path.exists(plain_pq):
    os.symlink(parquet_path, plain_pq)

# Verify required files
npz = f'v2_crosses_BTC_{TF}.npz'
assert os.path.exists(parquet_path), f"Missing {parquet_path}"
assert os.path.exists(npz), f"Missing {npz}"
log(f"Parquet: {parquet_path} ({os.path.getsize(parquet_path)/1e6:.1f} MB)")
log(f"NPZ: {npz} ({os.path.getsize(npz)/1e6:.1f} MB)")

# Run Optuna
run_tee(f'python -X utf8 -u run_optuna_local.py --tf {TF}',
        f'Optuna {TF}', f'optuna_{TF}.log')

# Run exhaustive optimizer (uses model from Optuna)
run_tee(f'python -X utf8 -u exhaustive_optimizer.py --tf {TF}',
        f'Optimizer {TF}', f'optimizer_{TF}.log')

# Summary
elapsed = time.time() - START
print(f"\n{'='*60}", flush=True)
print(f"  OPTUNA COMPLETE: {TF} ({elapsed:.0f}s / {elapsed/60:.1f} min)", flush=True)
print(f"{'='*60}", flush=True)

for f in [f'optuna_configs_{TF}.json', f'optuna_model_{TF}.json', f'cpcv_oos_predictions_{TF}.pkl',
          f'optuna_{TF}.db', f'optuna_search_results.json', f'platt_{TF}.pkl',
          f'features_{TF}_all.json', f'feature_importance_top500_{TF}.json']:
    if os.path.exists(f):
        print(f"  {os.path.getsize(f)/1e6:.1f} MB  {f}", flush=True)

with open(f'DONE_OPTUNA_{TF}', 'w') as f:
    f.write(f"Completed at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n")
    f.write(f"Total time: {elapsed:.0f}s\n")
log(f"Wrote DONE_OPTUNA_{TF}")
