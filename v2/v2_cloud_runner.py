#!/usr/bin/env python
"""
v2_cloud_runner.py — V2 Production Cloud Training Orchestrator
================================================================
6-phase pipeline: manifest checkpoint/resume, heartbeat monitoring,
OOM retry with halved batch, background download, smart dep install.

ALL GPUs work in unison per step — no CUDA_VISIBLE_DEVICES pinning.

Phases:
  1. Feature Builds     — build_features_v2.py per symbol/tf
  2. Training           — v2_multi_asset_trainer.py (XGBoost/LightGBM)
  3. Optimization       — exhaustive_optimizer.py per tf
  4. Validation         — PBO + Deflated Sharpe from OOS predictions
  5. Meta-Labeling      — Logistic/shallow XGB meta-model per tf
  6. LSTM               — v2_lstm_trainer.py per tf, all GPUs

Usage:
  python v2_cloud_runner.py                                    # full pipeline
  python v2_cloud_runner.py --phase 1 --build-tf 4h 1h        # just builds
  python v2_cloud_runner.py --phase 2 3 4 5 --tf 1d 4h        # train + validate
  python v2_cloud_runner.py --resume                           # resume from manifest
  python v2_cloud_runner.py --download-to user@IP:/path/       # auto-download
  python v2_cloud_runner.py --dry-run                          # show plan
"""

import os, sys, time, json, subprocess, threading, queue, argparse, glob, gc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

V2_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, V2_DIR)
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['SAVAGE22_DB_DIR'] = V2_DIR

START = time.time()
MANIFEST_PATH = os.path.join(V2_DIR, 'pipeline_manifest.json')

OOM_PATTERNS = ['CUDA out of memory', 'MemoryError', 'std::bad_alloc',
                'cuMemAlloc failed', 'OutOfMemoryError',
                'CUBLAS_STATUS_ALLOC_FAILED', 'out of memory']
OOM_RETURNCODES = {-9, 137, -7, 134}


# ── Logging ──────────────────────────────────────────────────

def log(msg):
    """[HH:MM:SS] timestamped log."""
    e = int(time.time() - START)
    print(f"[{e//3600:02d}:{e%3600//60:02d}:{e%60:02d}] {msg}", flush=True)


def gpu_cleanup(phase_name):
    """Free GPU memory after each phase (torch + CuPy + gc)."""
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        import cupy as cp
        pool = cp.get_default_memory_pool()
        pool.free_all_blocks()
    except Exception:
        pass
    log(f"GPU cleanup after {phase_name}")


# ── Pipeline State (thread-safe) ─────────────────────────────

class PipelineState:
    def __init__(self):
        self.active = {}       # {step_id: description}
        self.completed = {}    # {step_id: 'OK' or 'FAIL:reason'}
        self.phase = 0
        self.done = False
        self.lock = threading.Lock()
        self.dph = 0.0

    def start_step(self, step_id, desc):
        with self.lock:
            self.active[step_id] = desc

    def finish_step(self, step_id, status='OK'):
        with self.lock:
            self.active.pop(step_id, None)
            self.completed[step_id] = status

    def get_counts(self):
        with self.lock:
            ok = sum(1 for v in self.completed.values() if v == 'OK')
            fail = sum(1 for v in self.completed.values() if v != 'OK')
            return ok, fail, len(self.active)

    def get_active_names(self):
        with self.lock:
            return list(self.active.values())


# ── Heartbeat Thread ─────────────────────────────────────────

def _gpu_stats():
    try:
        nv = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5)
        if nv.returncode == 0:
            lines = []
            for line in nv.stdout.strip().split('\n'):
                p = [x.strip() for x in line.split(',')]
                if len(p) >= 3:
                    lines.append(f"GPU {len(lines)}: {p[0]}% util, {p[1]}/{p[2]} MB")
            return lines
    except Exception:
        pass
    return ['nvidia-smi unavailable']


def _ram_stats():
    try:
        import psutil
        vm = psutil.virtual_memory()
        used = (vm.total - vm.available) / (1024**3)
        total = vm.total / (1024**3)
        return f"{used:.1f}/{total:.1f} GB ({vm.percent}%)"
    except ImportError:
        pass
    try:
        with open('/proc/meminfo') as f:
            info = {}
            for line in f:
                parts = line.split()
                if parts[0] in ('MemTotal:', 'MemAvailable:'):
                    info[parts[0]] = int(parts[1])
            if len(info) == 2:
                total = info['MemTotal:'] / (1024**2)
                avail = info['MemAvailable:'] / (1024**2)
                return f"{total-avail:.1f}/{total:.1f} GB ({(total-avail)/total*100:.0f}%)"
    except Exception:
        pass
    return 'unavailable'


def heartbeat_thread(state, interval=60):
    """Print GPU, RAM, progress, cost every `interval` seconds."""
    while not state.done:
        time.sleep(interval)
        if state.done:
            break
        elapsed_h = (time.time() - START) / 3600
        ok, fail, active = state.get_counts()
        lines = ['', '-' * 60,
                 f'  HEARTBEAT | Phase {state.phase} | {elapsed_h:.2f}h',
                 f'  Steps: {ok} OK, {fail} FAIL, {active} active']
        for name in state.get_active_names():
            lines.append(f'    -> {name}')
        for g in _gpu_stats():
            lines.append(f'  {g}')
        lines.append(f'  RAM: {_ram_stats()}')
        if state.dph > 0:
            lines.append(f'  Cost: ${elapsed_h * state.dph:.2f} (${state.dph:.2f}/hr)')
        lines.append('-' * 60)
        for ln in lines:
            print(ln, flush=True)


# ── Subprocess Execution ─────────────────────────────────────

def _make_env():
    """Env for subprocesses: unbuffered, ALL GPUs visible, V1 DB path."""
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env.pop('CUDA_VISIBLE_DEVICES', None)  # ALL GPUs in unison
    env['SAVAGE22_V1_DIR'] = os.path.dirname(V2_DIR)
    return env


def run_step(cmd, step_name, timeout=7200):
    """Run subprocess with streaming output and OOM detection.
    Returns (success, oom_detected, output_lines).
    """
    log(f"START [{step_name}]: {cmd}")
    output_lines = []
    oom_detected = False
    try:
        proc = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, bufsize=1, env=_make_env(), cwd=V2_DIR)
        deadline = time.time() + timeout if timeout else None
        for line in proc.stdout:
            text = line.rstrip('\n')
            output_lines.append(text)
            print(f"  [{step_name}] {text}", flush=True)
            for pat in OOM_PATTERNS:
                if pat.lower() in text.lower():
                    oom_detected = True
            if deadline and time.time() > deadline:
                log(f"TIMEOUT [{step_name}] after {timeout}s — killing")
                proc.kill()
                proc.wait()
                return False, False, output_lines
        proc.wait()
        if proc.returncode in OOM_RETURNCODES:
            oom_detected = True
        success = (proc.returncode == 0)
        log(f"{'DONE' if success else 'FAIL'} [{step_name}]"
            f"{' (OOM)' if oom_detected else ''}"
            f"{'' if success else f' exit={proc.returncode}'}")
        return success, oom_detected, output_lines
    except Exception as e:
        log(f"ERROR [{step_name}]: {e}")
        return False, False, output_lines


def run_step_with_oom_retry(cmd, step_name, timeout=7200,
                             env_var=None, reduction=0.5, max_retries=4):
    """Wraps run_step. On OOM + env_var provided: progressively reduce value, retry up to max_retries."""
    success, oom, output = run_step(cmd, step_name, timeout)
    if success or not oom or not env_var:
        return success, oom, output

    current = os.environ.get(env_var)
    defaults = {'V2_RIGHT_CHUNK': 500, 'V2_BATCH_SIZE': 256, 'V2_GPU_BATCH': 25}
    base = int(current) if current else defaults.get(env_var, 100)

    for attempt in range(max_retries):
        new_val = max(1, int(base * (reduction ** (attempt + 1))))
        log(f"OOM RETRY {attempt+1}/{max_retries} [{step_name}]: {env_var} -> {new_val}")
        os.environ[env_var] = str(new_val)
        s2, o2, out2 = run_step(cmd, f"{step_name}_retry{attempt+1}", timeout)
        if s2 or not o2:
            # Restore original
            if current: os.environ[env_var] = current
            else: os.environ.pop(env_var, None)
            return s2, o2, out2

    if current: os.environ[env_var] = current
    else: os.environ.pop(env_var, None)
    return False, True, out2


# ── Dependency Installation ──────────────────────────────────

REQUIRED = {'lightgbm': 'lightgbm', 'sklearn': 'scikit-learn', 'scipy': 'scipy',
            'hmmlearn': 'hmmlearn', 'psutil': 'psutil', 'optuna': 'optuna',
            'numba': 'numba', 'torch': 'torch', 'xgboost': 'xgboost',
            'cupy': 'cupy-cuda12x', 'cudf': 'cudf', 'cuml': 'cuml',
            'pandas': 'pandas', 'dask_cuda': 'dask-cuda', 'dask': 'dask',
            'distributed': 'distributed', 'pyarrow': 'pyarrow'}

def install_deps():
    """Only pip-install packages that fail to import."""
    log("Checking dependencies...")
    missing = []
    for imp, pip in REQUIRED.items():
        try:
            __import__(imp)
        except ImportError:
            missing.append(pip)
    if not missing:
        log("All dependencies installed")
        return
    log(f"Installing: {', '.join(missing)}")
    run_step(f"{sys.executable} -m pip install {' '.join(missing)} --quiet",
             'pip_install', timeout=600)
    for imp, pip in REQUIRED.items():
        try:
            __import__(imp)
        except ImportError:
            log(f"WARNING: {pip} still missing after install")


# ── Manifest (checkpoint/resume) ─────────────────────────────

def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, 'r') as f:
                m = json.load(f)
            log(f"Loaded manifest: {len(m.get('phases', {}))} phases tracked")
            return m
        except (json.JSONDecodeError, IOError) as e:
            log(f"WARNING: corrupt manifest, starting fresh: {e}")
    return {'version': 2, 'phases': {}, 'hardware': {},
            'started_at': None, 'completed_at': None}


def save_manifest(manifest):
    try:
        from atomic_io import atomic_save_json
        atomic_save_json(manifest, MANIFEST_PATH)
    except ImportError:
        tmp = MANIFEST_PATH + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(manifest, f, indent=2, default=str)
        os.replace(tmp, MANIFEST_PATH)


def update_step(manifest, phase, step_id, status, **kw):
    pk = f'phase_{phase}'
    if pk not in manifest['phases']:
        manifest['phases'][pk] = {}
    manifest['phases'][pk][step_id] = {
        'status': status, 'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ'), **kw}
    save_manifest(manifest)


def is_step_done(manifest, phase, step_id):
    return manifest.get('phases', {}).get(f'phase_{phase}', {}).get(
        step_id, {}).get('status') == 'OK'


def is_phase_done(manifest, phase):
    steps = manifest.get('phases', {}).get(f'phase_{phase}', {})
    return bool(steps) and all(s.get('status') == 'OK' for s in steps.values())


# ── Phase 1: Feature Builds ─────────────────────────────────

def run_phase_1(state, manifest, args):
    log("=" * 60)
    log("PHASE 1: FEATURE BUILDS")
    log("=" * 60)
    from config import TRAINING_CRYPTO, ALL_TRAINING

    build_tfs = args.build_tf or ['4h', '1h', '15m', '5m']
    log(f"Build timeframes: {build_tfs}")

    tf_assets = {}
    for tf in build_tfs:
        if tf in ('1d', '1w'):
            tf_assets[tf] = ALL_TRAINING
        elif tf in ('4h', '1h'):
            tf_assets[tf] = TRAINING_CRYPTO
        else:
            tf_assets[tf] = ['BTC']

    total = sum(len(a) for a in tf_assets.values())
    done = 0
    completed_tfs = []   # track TFs whose builds all finished (for phase pipelining)

    def _build_one(symbol, tf):
        """Build a single symbol/tf. Returns (step_id, status)."""
        step_id = f'build_{symbol}_{tf}'
        parquet = os.path.join(V2_DIR, f'features_{symbol}_{tf}.parquet')
        npz = os.path.join(V2_DIR, f'v2_crosses_{symbol}_{tf}.npz')

        # Skip if already done (manifest or files on disk)
        if is_step_done(manifest, 1, step_id):
            log(f"  SKIP {step_id} (manifest)")
            return step_id, 'SKIP'
        if os.path.exists(parquet) and os.path.exists(npz):
            log(f"  SKIP {step_id} (files exist)")
            update_step(manifest, 1, step_id, 'OK', parquet=parquet, npz=npz)
            return step_id, 'SKIP'

        state.start_step(step_id, f'Building {symbol} {tf}')
        cmd = f"{sys.executable} -u build_features_v2.py --symbol {symbol} --tf {tf}"
        success, oom, _ = run_step_with_oom_retry(
            cmd, step_id, timeout=7200, env_var='V2_RIGHT_CHUNK', reduction=0.5)

        status = 'OK' if success else ('FAIL:OOM' if oom else 'FAIL:exit')
        update_step(manifest, 1, step_id, status, parquet=parquet, npz=npz)
        state.finish_step(step_id, status)
        if not success:
            log(f"  FAILED: {step_id}")
        gpu_cleanup(f"build_{step_id}")
        return step_id, status

    # Detect GPU count for parallel builds
    # NOTE: _build_one is a closure (captures manifest/state) so it can't be pickled
    # for ProcessPoolExecutor. We use ThreadPoolExecutor here because _build_one
    # launches subprocesses via run_step() — the actual builds run in separate
    # Python processes, each seeing ALL GPUs. The ThreadPool just manages
    # concurrent subprocess launches.
    try:
        from hardware_detect import detect_hardware
        _hw = detect_hardware()
        n_gpus_build = _hw['n_gpus'] or 1
    except Exception:
        n_gpus_build = 1
    build_workers = min(n_gpus_build, 4)

    for tf in build_tfs:
        assets = tf_assets[tf]
        log(f"  TF={tf}: {len(assets)} assets ({build_workers} concurrent workers)")

        with ThreadPoolExecutor(max_workers=build_workers) as build_pool:
            futures = {build_pool.submit(_build_one, symbol, tf): symbol
                       for symbol in assets}
            for future in as_completed(futures):
                step_id, status = future.result()
                symbol = futures[future]
                done += 1
                log(f"  Progress: {done}/{total}")
                gpu_cleanup(f"asset_{symbol}_{tf}")
        gpu_cleanup(f"phase1_tf_{tf}")

        completed_tfs.append(tf)

    log(f"Phase 1 complete: {done}/{total} processed")
    return completed_tfs


# ── Phase 2: Training ───────────────────────────────────────

def _run_phase_2_for_tf(state, manifest, args, tf):
    """Run Phase 2 training for a single timeframe. Thread-safe for background use."""
    step_id = f'train_{args.mode}_{args.engine}_{tf}'

    if is_step_done(manifest, 2, step_id):
        log(f"SKIP {step_id} (done)")
        return True

    state.start_step(step_id, f'Training {args.mode} {args.engine} TF={tf}')
    cmd = (f"{sys.executable} -u v2_multi_asset_trainer.py "
           f"--mode {args.mode} --tf {tf} --resume "
           f"--engine {args.engine} --boost-rounds {args.boost_rounds} "
           f"--parallel-splits")
    # Dask-XGBoost: opt-in only (sparse 4-6M features OOM when dense-converted)
    # Trainer has safety guard: auto-falls back to parallel-splits if dense > 50% RAM
    if getattr(args, 'use_dask', False):
        cmd += " --use-dask"

    success, oom, _ = run_step_with_oom_retry(
        cmd, step_id, timeout=14400, env_var='V2_BATCH_SIZE', reduction=0.5)

    status = 'OK' if success else ('FAIL:OOM' if oom else 'FAIL:exit')
    update_step(manifest, 2, step_id, status,
                mode=args.mode, engine=args.engine, tfs=[tf])
    state.finish_step(step_id, status)
    if not success:
        log(f"Phase 2 FAILED: {step_id}")
    return success


def run_phase_2(state, manifest, args, exclude_tfs=None):
    """Run Phase 2 for all TFs not in exclude_tfs (those already pipelined from Phase 1)."""
    log("=" * 60)
    log("PHASE 2: TRAINING")
    log("=" * 60)

    exclude_tfs = exclude_tfs or set()
    remaining_tfs = [tf for tf in args.tf if tf not in exclude_tfs]

    if not remaining_tfs:
        log("Phase 2: all TFs already handled by pipelined Phase 2, nothing to do")
        return

    tf_str = ' '.join(remaining_tfs)
    step_id = f'train_{args.mode}_{args.engine}_{"_".join(remaining_tfs)}'

    if is_step_done(manifest, 2, step_id):
        log(f"SKIP {step_id} (done)")
        return

    state.start_step(step_id, f'Training {args.mode} {args.engine} TFs={tf_str}')
    cmd = (f"{sys.executable} -u v2_multi_asset_trainer.py "
           f"--mode {args.mode} --tf {tf_str} --resume "
           f"--engine {args.engine} --boost-rounds {args.boost_rounds} "
           f"--parallel-splits")
    # Dask-XGBoost: opt-in only (sparse 4-6M features OOM when dense-converted)
    # Trainer has safety guard: auto-falls back to parallel-splits if dense > 50% RAM
    if getattr(args, 'use_dask', False):
        cmd += " --use-dask"

    success, oom, _ = run_step_with_oom_retry(
        cmd, step_id, timeout=14400, env_var='V2_BATCH_SIZE', reduction=0.5)

    status = 'OK' if success else ('FAIL:OOM' if oom else 'FAIL:exit')
    update_step(manifest, 2, step_id, status,
                mode=args.mode, engine=args.engine, tfs=remaining_tfs)
    state.finish_step(step_id, status)
    if not success:
        log(f"Phase 2 FAILED: {step_id}")


# ── Phase 3: Optimization ───────────────────────────────────

def run_phase_3(state, manifest, args):
    log("=" * 60)
    log("PHASE 3: OPTIMIZATION")
    log("=" * 60)

    # Detect GPU count for parallel TF optimization
    try:
        from hardware_detect import detect_hardware as _dh3
        _hw3 = _dh3()
        n_gpus_opt = max(_hw3['n_gpus'], 1)
    except Exception:
        n_gpus_opt = 1

    # Collect TFs that need optimization
    tfs_to_optimize = []
    for tf in args.tf:
        step_id = f'optimize_{tf}'
        if is_step_done(manifest, 3, step_id):
            log(f"SKIP {step_id} (done)")
            continue
        config_file = os.path.join(V2_DIR, f'exhaustive_configs_{tf}.json')
        if os.path.exists(config_file):
            log(f"SKIP {step_id} ({os.path.basename(config_file)} exists)")
            update_step(manifest, 3, step_id, 'OK', config=config_file)
            continue
        tfs_to_optimize.append(tf)

    if not tfs_to_optimize:
        log("Phase 3: nothing to optimize")
        return

    # Launch optimizer for ALL needed TFs in a single subprocess (it parallelizes internally)
    # The optimizer itself uses ProcessPoolExecutor with one TF per GPU
    tf_str = ' '.join(f'--tf {tf}' for tf in tfs_to_optimize)
    combined_step = f'optimize_{"_".join(tfs_to_optimize)}'
    state.start_step(combined_step, f'Optimizing {" ".join(tfs_to_optimize)} ({n_gpus_opt} GPUs)')

    cmd = f"{sys.executable} -u exhaustive_optimizer.py {tf_str} --resume"
    log(f"  Launching optimizer for {len(tfs_to_optimize)} TFs (internal parallelism: {n_gpus_opt} GPUs)")
    success, oom, _ = run_step(cmd, combined_step, timeout=21600)

    # Mark individual TF steps based on output files
    for tf in tfs_to_optimize:
        step_id = f'optimize_{tf}'
        config_file = os.path.join(V2_DIR, f'exhaustive_configs_{tf}.json')
        if os.path.exists(config_file):
            update_step(manifest, 3, step_id, 'OK', config=config_file)
            state.finish_step(step_id, 'OK')
        else:
            status = 'FAIL:OOM' if oom else ('FAIL:exit' if not success else 'FAIL:no_output')
            update_step(manifest, 3, step_id, status, config=config_file)
            state.finish_step(step_id, status)
            log(f"Optimizer FAILED for {tf}")

    state.finish_step(combined_step, 'OK' if success else 'FAIL')
    gpu_cleanup("phase3_optimization")
    log("Phase 3 complete")


# ── Phase 4: Validation (in-process) ────────────────────────

def run_phase_4(state, manifest, args):
    log("=" * 60)
    log("PHASE 4: VALIDATION")
    log("=" * 60)
    for tf in args.tf:
        step_id = f'validate_{tf}'
        if is_step_done(manifest, 4, step_id):
            log(f"SKIP {step_id} (done)")
            continue

        existing = glob.glob(os.path.join(V2_DIR, f'validation_report_*_{tf}.json'))
        if existing:
            log(f"SKIP {step_id} (report exists)")
            update_step(manifest, 4, step_id, 'OK', report=existing[0])
            continue

        oos_files = glob.glob(os.path.join(V2_DIR, f'oos_predictions_*_{tf}.pkl'))
        if not oos_files:
            log(f"SKIP {step_id} — no OOS predictions for {tf}")
            update_step(manifest, 4, step_id, 'SKIP:no_oos')
            continue

        state.start_step(step_id, f'Validating {tf}')
        try:
            import pickle
            from backtest_validation import validation_report
            for oos_path in oos_files:
                bn = os.path.basename(oos_path)
                mode_part = bn.replace('oos_predictions_', '').replace(f'_{tf}.pkl', '')
                rpt_path = os.path.join(V2_DIR, f'validation_report_{mode_part}_{tf}.json')
                if os.path.exists(rpt_path):
                    log(f"  Exists: {os.path.basename(rpt_path)}")
                    continue
                log(f"  Validating: {bn}")
                with open(oos_path, 'rb') as f:
                    oos = pickle.load(f)
                # Extract IS metrics from OOS predictions for proper PBO
                is_metrics = None
                if isinstance(oos, list) and oos and 'is_accuracy' in oos[0]:
                    is_metrics = [{'path': p.get('path_idx', i),
                                   'is_accuracy': p.get('is_accuracy'),
                                   'is_sharpe': p.get('is_sharpe')}
                                  for i, p in enumerate(oos)]
                    log(f"    IS metrics found: {len(is_metrics)} folds")
                report = validation_report(oos, tf_name=tf, is_metrics=is_metrics)
                try:
                    from atomic_io import atomic_save_json
                    atomic_save_json(report, rpt_path)
                except ImportError:
                    with open(rpt_path, 'w') as f:
                        json.dump(report, f, indent=2, default=str)
                log(f"  Saved: {os.path.basename(rpt_path)}")
                if isinstance(report, dict):
                    pbo = report.get('pbo', {})
                    if isinstance(pbo, dict):
                        log(f"    PBO lambda={pbo.get('pbo_lambda','N/A')}, "
                            f"p={pbo.get('pbo_pvalue','N/A')}")
                    ds = report.get('deflated_sharpe', {})
                    if isinstance(ds, dict):
                        log(f"    Deflated Sharpe p={ds.get('p_value','N/A')}")
            update_step(manifest, 4, step_id, 'OK')
            state.finish_step(step_id, 'OK')
        except Exception as e:
            log(f"  Validation FAILED for {tf}: {e}")
            import traceback; traceback.print_exc()
            update_step(manifest, 4, step_id, f'FAIL:{e}')
            state.finish_step(step_id, f'FAIL:{e}')
        gpu_cleanup(f"phase4_validate_{tf}")
    log("Phase 4 complete")


# ── Phase 5: Meta-Labeling (in-process) ─────────────────────

def run_phase_5(state, manifest, args):
    log("=" * 60)
    log("PHASE 5: META-LABELING")
    log("=" * 60)
    for tf in args.tf:
        step_id = f'meta_{tf}'
        if is_step_done(manifest, 5, step_id):
            log(f"SKIP {step_id} (done)")
            continue

        existing = glob.glob(os.path.join(V2_DIR, f'meta_model_*_{tf}.pkl'))
        if existing:
            log(f"SKIP {step_id} (model exists)")
            update_step(manifest, 5, step_id, 'OK', model=existing[0])
            continue

        oos_files = glob.glob(os.path.join(V2_DIR, f'oos_predictions_*_{tf}.pkl'))
        if not oos_files:
            log(f"SKIP {step_id} — no OOS predictions for {tf}")
            update_step(manifest, 5, step_id, 'SKIP:no_oos')
            continue

        state.start_step(step_id, f'Meta-labeling {tf}')
        try:
            import pickle
            from meta_labeling import train_meta_model
            for oos_path in oos_files:
                bn = os.path.basename(oos_path)
                mode_part = bn.replace('oos_predictions_', '').replace(f'_{tf}.pkl', '')
                model_path = os.path.join(V2_DIR, f'meta_model_{tf}.pkl')
                if os.path.exists(model_path):
                    log(f"  Exists: {os.path.basename(model_path)}")
                    continue
                log(f"  Training meta-model from: {bn}")
                with open(oos_path, 'rb') as f:
                    oos = pickle.load(f)
                result = train_meta_model(oos, tf_name=tf, db_dir=V2_DIR)
                if result is not None:
                    try:
                        from atomic_io import atomic_save_pickle
                        atomic_save_pickle(result, model_path)
                    except ImportError:
                        with open(model_path, 'wb') as f:
                            pickle.dump(result, f)
                    log(f"  Saved: {os.path.basename(model_path)}")
                    m = result.get('metrics', {})
                    if m:
                        log(f"    acc={m.get('accuracy','N/A')}, "
                            f"auc={m.get('auc','N/A')}, "
                            f"thr={result.get('threshold','N/A')}")
                else:
                    log(f"  Meta-model returned None for {bn}")
            update_step(manifest, 5, step_id, 'OK')
            state.finish_step(step_id, 'OK')
        except Exception as e:
            log(f"  Meta-labeling FAILED for {tf}: {e}")
            import traceback; traceback.print_exc()
            update_step(manifest, 5, step_id, f'FAIL:{e}')
            state.finish_step(step_id, f'FAIL:{e}')
        gpu_cleanup(f"phase5_meta_{tf}")
    log("Phase 5 complete")


# ── Phase 6: LSTM ────────────────────────────────────────────

def run_phase_6(state, manifest, args):
    log("=" * 60)
    log("PHASE 6: LSTM TRAINING")
    log("=" * 60)
    for tf in args.tf:
        step_id = f'lstm_{tf}'
        if is_step_done(manifest, 6, step_id):
            log(f"SKIP {step_id} (done)")
            continue

        state.start_step(step_id, f'LSTM training {tf}')
        oos_candidates = glob.glob(os.path.join(V2_DIR, f'oos_predictions_*_{tf}.pkl'))
        oos_probs = oos_candidates[0] if oos_candidates else None
        cmd = f"{sys.executable} -u v2_lstm_trainer.py --tf {tf} --resume --alpha-search"
        if oos_probs and os.path.exists(oos_probs):
            cmd += f" --xgb-probs {oos_probs}"
            log(f"  Blending with XGB probs: {os.path.basename(oos_probs)}")

        success, oom, _ = run_step_with_oom_retry(
            cmd, step_id, timeout=14400, env_var='V2_BATCH_SIZE', reduction=0.5)

        status = 'OK' if success else ('FAIL:OOM' if oom else 'FAIL:exit')
        update_step(manifest, 6, step_id, status, tf=tf)
        state.finish_step(step_id, status)
        if not success:
            log(f"LSTM FAILED for {tf}")
        gpu_cleanup(f"phase6_lstm_{tf}")
    log("Phase 6 complete")


def run_phase_7(state, manifest, args):
    """Phase 7: Backtesting Audit Report — runs after all models + configs ready."""
    log("=" * 60)
    log("PHASE 7: BACKTESTING AUDIT REPORT")
    log("=" * 60)
    step_id = 'audit_report'
    if is_step_done(manifest, 7, step_id):
        log(f"SKIP {step_id} (done)")
        return

    state.start_step(step_id, 'Generating audit report')
    tf_str = ' '.join(args.tf)
    cmd = f"{sys.executable} -u backtesting_audit.py --tf {tf_str}"
    success, oom, _ = run_step_with_oom_retry(
        cmd, step_id, timeout=600, env_var=None, reduction=1.0)

    status = 'OK' if success else 'FAIL:exit'
    update_step(manifest, 7, step_id, status)
    state.finish_step(step_id, status)
    if success:
        log("Audit report generated: audit_report.json, audit_report.txt, audit_heatmap.html")
    else:
        log("Audit report FAILED (non-critical, continuing)")
    log("Phase 7 complete")


# ── Download Worker (background thread) ──────────────────────

def download_worker(download_queue, target, method='rsync'):
    """Pull (phase, artifact_paths) from queue; rsync/scp to target. None = stop."""
    log(f"Download worker started -> {target} ({method})")
    while True:
        item = download_queue.get()
        if item is None:
            log("Download worker: shutdown")
            break
        phase, patterns = item
        files = []
        for pat in patterns:
            files.extend(glob.glob(pat))
        if not files:
            log(f"Download worker: no files for phase {phase}")
            continue
        flist = ' '.join(f'"{f}"' for f in files)
        log(f"Download worker: sending {len(files)} files (phase {phase})")
        try:
            if method == 'scp':
                cmd = f'scp {flist} "{target}"'
            else:
                cmd = f'rsync -avz --progress {flist} "{target}"'
            proc = subprocess.run(cmd, shell=True, capture_output=True,
                                  text=True, timeout=3600, cwd=V2_DIR)
            if proc.returncode == 0:
                log(f"Download worker: phase {phase} OK ({len(files)} files)")
            else:
                log(f"Download worker: phase {phase} FAIL: {proc.stderr[:200]}")
        except subprocess.TimeoutExpired:
            log(f"Download worker: phase {phase} timed out")
        except Exception as e:
            log(f"Download worker: phase {phase} error: {e}")


# ── Phase Artifacts ──────────────────────────────────────────

def get_phase_artifacts(phase):
    """Glob patterns for artifacts produced by each phase."""
    p = {
        1: ['features_*.parquet', 'v2_crosses_*.npz', 'v2_cross_names_*.json'],
        2: ['model_v2_*.*', 'importance_v2_*.json',
            'oos_predictions_*.pkl', 'training_report_*.json',
            'features_v2_*.json'],
        3: ['exhaustive_configs_*.json', 'optimizer_checkpoint_*.json'],
        4: ['validation_report_*.json'],
        5: ['meta_model_*.pkl'],
        6: ['lstm_*.pt', 'lstm_*_checkpoint.pt',
            'platt_lstm_*.pkl', 'lstm_report_*.json',
            'blend_config_*.json'],
        7: ['audit_report.json', 'audit_report.txt', 'audit_heatmap.html'],
    }
    common = ['pipeline_manifest.json']
    return [os.path.join(V2_DIR, g) for g in p.get(phase, []) + common]


# ── Data Inventory ───────────────────────────────────────────

def log_data_inventory():
    cats = [('Parquets', 'features_*.parquet'),
            ('Sparse .npz', 'v2_crosses_*.npz'),
            ('Models', 'model_v2_*.*'),
            ('OOS preds', 'oos_predictions_*.pkl'),
            ('Opt configs', 'exhaustive_configs_*.json'),
            ('Val reports', 'validation_report_*.json'),
            ('Meta models', 'meta_model_*.pkl'),
            ('LSTM models', 'lstm_*.pt'),
            ('LSTM ckpts', 'lstm_*_checkpoint.pt')]
    log("DATA INVENTORY:")
    total_mb = 0
    for label, pat in cats:
        files = glob.glob(os.path.join(V2_DIR, pat))
        mb = sum(os.path.getsize(f) for f in files) / 1e6
        total_mb += mb
        if files:
            log(f"  {label}: {len(files)} ({mb:.1f} MB)")
    log(f"  TOTAL: {total_mb:.1f} MB")


# ── Dashboard ────────────────────────────────────────────────

def print_dashboard(manifest):
    names = {1: 'Feature Builds', 2: 'Training', 3: 'Optimization',
             4: 'Validation', 5: 'Meta-Labeling', 6: 'LSTM'}
    print('\n' + '=' * 70, flush=True)
    print('  PIPELINE DASHBOARD', flush=True)
    print('=' * 70, flush=True)
    t_ok = t_fail = t_skip = 0
    for p in range(1, 7):
        steps = manifest.get('phases', {}).get(f'phase_{p}', {})
        ok = sum(1 for s in steps.values() if s.get('status') == 'OK')
        fail = sum(1 for s in steps.values() if s.get('status', '').startswith('FAIL'))
        skip = sum(1 for s in steps.values() if s.get('status', '').startswith('SKIP'))
        t_ok += ok; t_fail += fail; t_skip += skip
        if not steps:
            st = 'NOT STARTED'
        elif fail:
            st = f'{ok} OK, {fail} FAIL' + (f', {skip} SKIP' if skip else '')
        else:
            st = f'{ok} OK' + (f', {skip} SKIP' if skip else '') + ' DONE'
        print(f'  Phase {p}: {names[p]:20s} | {st}', flush=True)
        for sid, sd in steps.items():
            if sd.get('status', '').startswith('FAIL'):
                print(f'    X {sid}: {sd["status"]}', flush=True)
    print('-' * 70, flush=True)
    print(f'  TOTAL: {t_ok} OK, {t_fail} FAIL, {t_skip} SKIP', flush=True)
    hw = manifest.get('hardware', {})
    if hw:
        print(f'  HW: {hw.get("n_gpus",0)} GPUs ({hw.get("total_vram_gb",0)} GB), '
              f'{hw.get("total_ram_gb",0)} GB RAM, {hw.get("cpu_count",0)} CPUs', flush=True)
    for k in ('started_at', 'completed_at'):
        if manifest.get(k):
            print(f'  {k}: {manifest[k]}', flush=True)
    print('=' * 70 + '\n', flush=True)


# ── Dry Run Plan ─────────────────────────────────────────────

def print_dry_run_plan(args, manifest):
    from config import TRAINING_CRYPTO, ALL_TRAINING
    phases = args.phase or [1, 2, 3, 4, 5, 6]
    print('\n' + '=' * 70, flush=True)
    print('  DRY RUN PLAN', flush=True)
    print('=' * 70, flush=True)
    for p in phases:
        if args.resume and is_phase_done(manifest, p):
            print(f'  Phase {p}: SKIP (done)', flush=True)
            continue
        if p == 1:
            print(f'  Phase 1: Feature Builds', flush=True)
            for tf in (args.build_tf or ['4h', '1h', '15m', '5m']):
                if tf in ('1d', '1w'): assets = ALL_TRAINING
                elif tf in ('4h', '1h'): assets = TRAINING_CRYPTO
                else: assets = ['BTC']
                skip = sum(1 for s in assets if is_step_done(manifest, 1, f'build_{s}_{tf}'))
                print(f'    {tf}: {len(assets)-skip} to build, {skip} done', flush=True)
        elif p == 2:
            print(f'  Phase 2: Training — mode={args.mode}, engine={args.engine}, '
                  f'TFs={args.tf}, rounds={args.boost_rounds}', flush=True)
        elif p == 3:
            print(f'  Phase 3: Optimization', flush=True)
            for tf in args.tf:
                ex = os.path.exists(os.path.join(V2_DIR, f'exhaustive_configs_{tf}.json'))
                print(f'    {tf}: {"exists" if ex else "WILL RUN"}', flush=True)
        elif p == 4:
            print(f'  Phase 4: Validation', flush=True)
            for tf in args.tf:
                n = len(glob.glob(os.path.join(V2_DIR, f'oos_predictions_*_{tf}.pkl')))
                print(f'    {tf}: {n} OOS files', flush=True)
        elif p == 5:
            print(f'  Phase 5: Meta-Labeling', flush=True)
            for tf in args.tf:
                n = len(glob.glob(os.path.join(V2_DIR, f'oos_predictions_*_{tf}.pkl')))
                print(f'    {tf}: {n} OOS files', flush=True)
        elif p == 6:
            print(f'  Phase 6: LSTM — TFs={args.tf}, all GPUs', flush=True)
    if args.download_to:
        print(f'  Download: {args.download_to} ({args.download_method})', flush=True)
    print(f'  Rate: ${args.dph:.2f}/hr', flush=True)
    print('=' * 70 + '\n', flush=True)


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='V2 Production Cloud Training Orchestrator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python v2_cloud_runner.py                                    # full pipeline
  python v2_cloud_runner.py --phase 1 --build-tf 4h 1h        # just builds
  python v2_cloud_runner.py --phase 2 3 4 5 --tf 1d 4h        # train + validate
  python v2_cloud_runner.py --resume                           # resume from manifest
  python v2_cloud_runner.py --download-to user@IP:/path/       # auto-download
  python v2_cloud_runner.py --dry-run                          # show plan
""")
    # Phase selection
    parser.add_argument('--phase', nargs='+', type=int, default=None,
                        help='Phases to run (1-6). Default: all')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from pipeline_manifest.json')
    # Phase 1
    parser.add_argument('--build-tf', nargs='+', default=None,
                        help='TFs to build (default: 4h 1h 15m 5m)')
    # Phase 2
    parser.add_argument('--mode', default='all',
                        choices=['per-asset', 'unified', 'production', 'all'])
    parser.add_argument('--tf', nargs='+', default=['1d'],
                        help='TFs for train/optimize/validate')
    parser.add_argument('--engine', default='xgboost',
                        choices=['xgboost', 'lightgbm', 'both'])
    parser.add_argument('--boost-rounds', type=int, default=500)
    # Infrastructure
    parser.add_argument('--skip-install', action='store_true')
    parser.add_argument('--download-to', default=None,
                        help='Auto-download target (user@IP:/path/)')
    parser.add_argument('--download-method', default='rsync',
                        choices=['rsync', 'scp'])
    parser.add_argument('--heartbeat-interval', type=int, default=60)
    parser.add_argument('--dph', type=float, default=2.88,
                        help='Cost per hour for estimates')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    # Banner
    print('\n' + '=' * 70, flush=True)
    print('  SAVAGE22 V2 — PRODUCTION CLOUD ORCHESTRATOR', flush=True)
    print('=' * 70, flush=True)
    print(f'  Started:  {time.strftime("%Y-%m-%d %H:%M:%S UTC")}', flush=True)
    print(f'  Phases:   {args.phase or "ALL (1-7)"}', flush=True)
    print(f'  TFs:      {args.tf}', flush=True)
    print(f'  Engine:   {args.engine} | Mode: {args.mode}', flush=True)
    print(f'  Resume:   {args.resume} | Cost: ${args.dph:.2f}/hr', flush=True)
    print('=' * 70 + '\n', flush=True)

    # 1. Hardware detection
    try:
        from hardware_detect import detect_hardware, log_hardware
        hw = detect_hardware()
        log_hardware(hw)
    except ImportError:
        log("WARNING: hardware_detect.py not found")
        hw = {'n_gpus': 0, 'gpu_names': [], 'vram_per_gpu_gb': [],
              'total_vram_gb': 0, 'total_ram_gb': 32, 'available_ram_gb': 16,
              'cpu_count': os.cpu_count() or 1, 'cpu_ghz': None}

    # 2. Install deps
    if not args.skip_install:
        install_deps()

    # 3. Load/create manifest
    manifest = load_manifest()
    manifest['hardware'] = hw
    manifest['started_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ')
    save_manifest(manifest)

    # 4. Data inventory
    log_data_inventory()

    # 5. Dry run
    if args.dry_run:
        print_dry_run_plan(args, manifest)
        print_dashboard(manifest)
        log("DRY RUN — no execution")
        return

    # 6. Heartbeat
    state = PipelineState()
    state.dph = args.dph
    hb = threading.Thread(target=heartbeat_thread,
                          args=(state, args.heartbeat_interval), daemon=True)
    hb.start()

    # 7. Download thread
    dl_queue = queue.Queue()
    if args.download_to:
        dl = threading.Thread(target=download_worker,
                              args=(dl_queue, args.download_to, args.download_method),
                              daemon=True)
        dl.start()
        log(f"Download worker -> {args.download_to}")

    # 8. Run phases — with Phase 1 → Phase 2 pipelining
    phases = args.phase or [1, 2, 3, 4, 5, 6, 7]
    phase_fns = {1: run_phase_1, 3: run_phase_3,
                 4: run_phase_4, 5: run_phase_5, 6: run_phase_6,
                 7: run_phase_7}

    # Phase 2 pipelining: launch training for each TF as soon as Phase 1
    # completes all assets for that TF (independent TFs don't need to wait).
    # E.g. 1d Phase 2 starts while 4h Phase 1 builds are still running.
    phase2_pool = None
    phase2_futures = {}      # {tf: future}
    pipelined_tfs = set()    # TFs whose Phase 2 was already launched from pipelining

    if 1 in phases and 2 in phases:
        # N_GPUS concurrent training jobs — each TF trains simultaneously
        try:
            from hardware_detect import detect_hardware as _dh2
            _hw2 = _dh2()
            _n_gpus_p2 = max(_hw2['n_gpus'], 1)
        except Exception:
            _n_gpus_p2 = 1
        phase2_pool = ThreadPoolExecutor(max_workers=_n_gpus_p2)

    for p in phases:
        if p == 2:
            # Phase 2 handled specially: wait for any pipelined Phase 2 futures,
            # then run Phase 2 for remaining TFs that weren't pipelined.
            state.phase = 2
            log(f"{'='*60}")
            log(f"ENTERING PHASE 2 (collecting pipelined + remaining)")
            log(f"{'='*60}")

            # Collect pipelined Phase 2 results
            for tf, fut in phase2_futures.items():
                try:
                    success = fut.result()
                    log(f"  Pipelined Phase 2 for {tf}: {'OK' if success else 'FAILED'}")
                except Exception as e:
                    log(f"  Pipelined Phase 2 for {tf} ERROR: {e}")
                    import traceback; traceback.print_exc()

            # Run Phase 2 for any TFs not already pipelined
            try:
                run_phase_2(state, manifest, args, exclude_tfs=pipelined_tfs)
                if args.download_to:
                    dl_queue.put((2, get_phase_artifacts(2)))
            except KeyboardInterrupt:
                log(f"INTERRUPTED during phase 2")
                manifest['interrupted_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ')
                save_manifest(manifest)
                break
            except Exception as e:
                log(f"Phase 2 FAILED: {e}")
                import traceback; traceback.print_exc()
                update_step(manifest, 2, 'phase_2_exception', f'FAIL:{e}')
            continue

        if p not in phase_fns:
            log(f"Unknown phase {p}, skipping")
            continue
        if args.resume and is_phase_done(manifest, p):
            log(f"Phase {p} done (manifest), skipping")
            continue
        state.phase = p
        log(f"{'='*60}")
        log(f"ENTERING PHASE {p}")
        log(f"{'='*60}")
        try:
            result = phase_fns[p](state, manifest, args)

            # Phase 1 → Phase 2 pipelining: after Phase 1 completes each TF,
            # immediately submit Phase 2 for that TF in the background.
            if p == 1 and phase2_pool is not None and result:
                completed_tfs = result  # run_phase_1 returns list of completed TFs
                for tf in completed_tfs:
                    if tf in args.tf and tf not in pipelined_tfs:
                        # Verify all required parquets exist for this TF
                        from config import TRAINING_CRYPTO, ALL_TRAINING
                        if tf in ('1d', '1w'):
                            check_assets = ALL_TRAINING
                        elif tf in ('4h', '1h'):
                            check_assets = TRAINING_CRYPTO
                        else:
                            check_assets = ['BTC']
                        all_exist = all(
                            os.path.exists(os.path.join(V2_DIR, f'features_{s}_{tf}.parquet'))
                            and os.path.exists(os.path.join(V2_DIR, f'v2_crosses_{s}_{tf}.npz'))
                            for s in check_assets)
                        if all_exist:
                            log(f"  PIPELINE: submitting Phase 2 for {tf} in background")
                            fut = phase2_pool.submit(
                                _run_phase_2_for_tf, state, manifest, args, tf)
                            phase2_futures[tf] = fut
                            pipelined_tfs.add(tf)

            if args.download_to:
                dl_queue.put((p, get_phase_artifacts(p)))
        except KeyboardInterrupt:
            log(f"INTERRUPTED during phase {p}")
            manifest['interrupted_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ')
            save_manifest(manifest)
            break
        except Exception as e:
            log(f"Phase {p} FAILED: {e}")
            import traceback; traceback.print_exc()
            update_step(manifest, p, f'phase_{p}_exception', f'FAIL:{e}')

    # Shutdown phase2 pool
    if phase2_pool is not None:
        phase2_pool.shutdown(wait=True)

    # 9. Final summary
    state.done = True
    if args.download_to:
        dl_queue.put(None)
        log("Waiting for download worker...")
    manifest['completed_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ')
    save_manifest(manifest)
    print_dashboard(manifest)

    elapsed_h = (time.time() - START) / 3600
    all_files = []
    for p in range(1, 8):
        for pat in get_phase_artifacts(p):
            all_files.extend(glob.glob(pat))
    total_mb = sum(os.path.getsize(f) for f in all_files) / 1e6
    log(f"PIPELINE COMPLETE — {elapsed_h:.1f}h, ${elapsed_h * args.dph:.2f}, "
        f"{len(all_files)} artifacts ({total_mb:.1f} MB)")


if __name__ == '__main__':
    main()
