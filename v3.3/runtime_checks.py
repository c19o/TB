"""
Savage22 V3.3 Runtime Checks
==============================
Catches failures that only appear DURING training:
- OOM from memory miscalculation
- Single-threaded training on multi-core machines
- GPU idle / VRAM overflow
- NaN trials, stuck training, memory leaks

Three layers:
1. preflight_training()  — after data load, before training
2. TrainingMonitor       — daemon thread during training
3. post_trial_check()    — between Optuna trials
"""
import os
import sys
import time
import logging
import threading
import subprocess
import numpy as np

log = logging.getLogger('runtime_checks')
if not log.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter('[RT %(levelname)s] %(message)s'))
    log.addHandler(_h)
    log.setLevel(logging.INFO)


# ══════════════════════════════════════════════════════════════
# LAYER 1: Pre-Training Runtime Checks
# ══════════════════════════════════════════════════════════════
def preflight_training(X, y, tf, params, n_jobs=1):
    """Run after data is loaded, before any training starts.
    Raises RuntimeError on critical failures. Logs warnings for non-critical."""
    failures = []

    def check(name, condition, msg):
        if condition:
            log.info(f"[PASS] {name}")
        else:
            failures.append(f"{name}: {msg}")
            log.error(f"[FAIL] {name} -- {msg}")

    def warn(name, condition, msg):
        if not condition:
            log.warning(f"[WARN] {name} -- {msg}")

    log.info(f"{'='*55}")
    log.info(f"  RUNTIME PRE-FLIGHT: {tf} | shape={X.shape} | nnz={X.nnz:,}")
    log.info(f"{'='*55}")

    # ── Memory estimation ──
    ram_gb = _get_ram_gb()
    csr_bytes = X.nnz * 5 + (X.shape[0] + 1) * 8  # data + indices + indptr
    csr_gb = csr_bytes / (1024**3)
    peak_gb = csr_gb * 2.5  # LightGBM internal overhead (binning, sorting, histograms)
    dense_gb = X.shape[0] * X.shape[1] * 8 / (1024**3)

    log.info(f"  RAM: {ram_gb:.0f}GB | CSR: {csr_gb:.1f}GB | Peak est: {peak_gb:.1f}GB | Dense would be: {dense_gb:.0f}GB")

    check("peak memory fits in RAM",
          peak_gb < ram_gb * 0.75,
          f"Estimated peak {peak_gb:.1f}GB > 75% of {ram_gb:.0f}GB RAM. "
          f"Likely OOM during Dataset construction.")

    if dense_gb > ram_gb * 0.5:
        log.info(f"  Dense would be {dense_gb:.0f}GB > 50% of RAM — sparse path mandatory (correct)")
    else:
        log.info(f"  Dense would be {dense_gb:.1f}GB — fits in RAM but sparse still preferred")

    if n_jobs > 1:
        per_job_gb = peak_gb / n_jobs
        warn("per-trial memory reasonable",
             per_job_gb < ram_gb * 0.4,
             f"Each of {n_jobs} parallel trials needs ~{per_job_gb:.1f}GB. "
             f"Total {per_job_gb * n_jobs:.1f}GB may exceed {ram_gb:.0f}GB RAM.")

    # ── Sparse matrix integrity ──
    import scipy.sparse as sp
    check("matrix is CSR format",
          sp.issparse(X) and X.format == 'csr',
          f"Matrix format is {getattr(X, 'format', type(X).__name__)}. Must be CSR for LightGBM.")

    if hasattr(X, 'nnz'):
        if X.nnz > 2**31 - 1:
            check("int64 indptr for large NNZ",
                  X.indptr.dtype == np.int64,
                  f"NNZ={X.nnz:,} > 2^31 but indptr is {X.indptr.dtype}. Will overflow.")
        else:
            log.info(f"  NNZ={X.nnz:,} (within int32 range)")

    check("matrix dtype is float",
          X.dtype in (np.float32, np.float64, np.bool_, np.uint8),
          f"Matrix dtype={X.dtype}. LightGBM expects float32/64. FIX: X = X.astype(np.float32)")

    # ── Label checks ──
    nan_count = np.sum(np.isnan(y))
    nan_pct = nan_count / len(y) * 100 if len(y) > 0 else 0
    if nan_pct > 5:
        check("NaN labels < 5%",
              False,
              f"{nan_count} NaN labels ({nan_pct:.1f}%) — too many. Check triple barrier config.")
    elif nan_count > 0:
        warn("NaN labels present (will be filtered by valid_mask)",
             False,
             f"{nan_count} NaN labels ({nan_pct:.1f}%) — normal for end-of-data. "
             f"Ensure run_optuna_local.py applies valid_mask = ~np.isnan(y).")

    unique, counts = np.unique(y.astype(int), return_counts=True)
    label_dist = dict(zip(unique, counts / len(y) * 100))
    log.info(f"  Label distribution: { {int(k): f'{v:.1f}%' for k,v in label_dist.items()} }")

    hold_pct = label_dist.get(1, 0)
    short_pct = label_dist.get(0, 0)
    long_pct = label_dist.get(2, 0)

    warn("HOLD < 20%",
         hold_pct < 20,
         f"HOLD={hold_pct:.1f}% — model will learn 'do nothing'. "
         f"Adjust triple barrier config (increase hold_bars).")

    warn("SHORT > 5%",
         short_pct > 5,
         f"SHORT={short_pct:.1f}% — too few SHORT labels. "
         f"Model can't learn SHORT direction. Check barrier asymmetry.")

    warn("LONG > 5%",
         long_pct > 5,
         f"LONG={long_pct:.1f}% — too few LONG labels.")

    # ── Thread/CPU configuration ──
    try:
        physical_cores = len(os.sched_getaffinity(0))
    except AttributeError:
        physical_cores = os.cpu_count() or 4

    num_threads = params.get('num_threads', 0)
    if num_threads == 0:
        num_threads = physical_cores

    omp_threads = os.environ.get('OMP_NUM_THREADS')
    if omp_threads:
        omp_val = int(omp_threads)
        warn("OMP_NUM_THREADS not stuck at cross-gen value",
             omp_val > 4 or omp_val == num_threads,
             f"OMP_NUM_THREADS={omp_val} (likely leftover from cross gen phase). "
             f"LightGBM will use {omp_val} threads instead of {num_threads}. "
             f"FIX: unset OMP_NUM_THREADS or set to {num_threads}")

    numba_threads = os.environ.get('NUMBA_NUM_THREADS')
    if numba_threads:
        warn("NUMBA_NUM_THREADS not stuck at 4",
             int(numba_threads) > 4,
             f"NUMBA_NUM_THREADS={numba_threads} (leftover from cross gen). "
             f"Won't affect LightGBM but may slow Numba-compiled code.")

    if n_jobs > 1:
        total_threads = num_threads * n_jobs
        warn("total threads <= physical cores",
             total_threads <= physical_cores * 1.5,
             f"{num_threads} threads x {n_jobs} jobs = {total_threads} > {physical_cores} cores. "
             f"Oversubscription causes cache thrashing. Reduce n_jobs or num_threads.")

    # ── GPU checks ──
    n_gpus = int(os.environ.get('LGBM_NUM_GPUS', '0'))
    if n_gpus > 0 or params.get('device_type') in ('cuda', 'cuda_sparse', 'gpu'):
        _check_gpu_budget(X, params, n_gpus, n_jobs)

        # Verify cuda_sparse actually works (not silent CPU fallback)
        if params.get('device_type') == 'cuda_sparse':
            import scipy.sparse as _sp
            _test_X = _sp.random(50, 100, density=0.1, format='csr', dtype=np.float32)
            _test_y = np.random.randint(0, 3, 50)
            import lightgbm as _lgb
            _test_ds = _lgb.Dataset(_test_X, label=_test_y, params={'feature_pre_filter': False})
            _test_p = {'objective': 'multiclass', 'num_class': 3, 'device_type': 'cuda_sparse',
                        'gpu_device_id': 0, 'num_iterations': 1, 'verbose': -1}
            try:
                _lgb.train(_test_p, _test_ds)
                log.info("  cuda_sparse smoke test: PASSED (GPU 0)")
            except Exception as _e:
                failures.append(f"cuda_sparse broken: {_e}")
                log.error(f"[FAIL] cuda_sparse test failed: {_e}")

        # Verify all assigned GPUs respond
        if n_gpus > 1:
            import lightgbm as _lgb
            import scipy.sparse as _sp
            _dead_gpus = []
            for _gid in range(n_gpus):
                _test_X = _sp.random(50, 100, density=0.1, format='csr', dtype=np.float32)
                _test_y = np.random.randint(0, 3, 50)
                _test_ds = _lgb.Dataset(_test_X, label=_test_y, params={'feature_pre_filter': False})
                _test_p = {'objective': 'multiclass', 'num_class': 3,
                           'device_type': params.get('device_type', 'cuda'),
                           'gpu_device_id': _gid, 'num_iterations': 1, 'verbose': -1}
                try:
                    _lgb.train(_test_p, _test_ds)
                except Exception:
                    _dead_gpus.append(_gid)
            check(f"all {n_gpus} GPUs respond",
                  len(_dead_gpus) == 0,
                  f"GPUs {_dead_gpus} failed smoke test. Only {n_gpus - len(_dead_gpus)}/{n_gpus} working.")

    # ── Summary ──
    if failures:
        log.error(f"\n{'='*55}")
        log.error(f"  RUNTIME PRE-FLIGHT FAILED: {len(failures)} critical issues")
        for f in failures:
            log.error(f"    - {f}")
        log.error(f"{'='*55}")
        raise RuntimeError(f"Runtime pre-flight failed: {len(failures)} issues. See log above.")
    else:
        log.info(f"  RUNTIME PRE-FLIGHT PASSED")


def _check_gpu_budget(X, params, n_gpus, n_jobs):
    """Estimate GPU VRAM budget and warn if insufficient."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return
        vram_mbs = [int(x.strip()) for x in result.stdout.strip().split('\n')]
        min_vram_mb = min(vram_mbs)

        # CSR resident on GPU
        csr_mb = (X.nnz * 5 + (X.shape[0] + 1) * 8) / (1024**2)
        # Histogram pool: num_leaves * total_bins * 12 bytes
        num_leaves = params.get('num_leaves', 63)
        max_bin = params.get('max_bin', 255)
        # Rough estimate: EFB reduces features 100x for binary
        efb_bundles = max(1000, X.shape[1] // max_bin)
        hist_mb = num_leaves * efb_bundles * 12 / (1024**2)
        # Gradients
        grad_mb = X.shape[0] * 16 / (1024**2)
        total_mb = csr_mb + hist_mb + grad_mb + 200  # 200MB overhead

        trials_per_gpu = max(1, n_jobs // max(1, n_gpus))
        per_gpu_mb = total_mb * trials_per_gpu

        log.info(f"  GPU VRAM budget: CSR={csr_mb:.0f}MB + hist={hist_mb:.0f}MB + grad={grad_mb:.0f}MB = {total_mb:.0f}MB")
        log.info(f"  GPU VRAM available: {min_vram_mb}MB (smallest GPU), {trials_per_gpu} trials/GPU")

        if per_gpu_mb > min_vram_mb * 0.85:
            log.warning(f"[WARN] GPU VRAM may overflow: {per_gpu_mb:.0f}MB needed > {min_vram_mb * 0.85:.0f}MB budget (85%)")
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════
# LAYER 2: During-Training Monitor (daemon thread)
# ══════════════════════════════════════════════════════════════
class TrainingMonitor:
    """Lightweight daemon thread that monitors system health during training."""

    def __init__(self, interval=30):
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = None
        self._snapshots = []
        self._low_cpu_count = 0
        self._high_mem_count = 0

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True, name='training-monitor')
        self._thread.start()
        log.info(f"[MONITOR] Started (interval={self.interval}s)")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        log.info(f"[MONITOR] Stopped ({len(self._snapshots)} snapshots)")

    def _run(self):
        try:
            import psutil
        except ImportError:
            log.warning("[MONITOR] psutil not installed — monitor disabled")
            return

        prev_rss = None
        while not self._stop_event.wait(self.interval):
            try:
                snap = self._take_snapshot(psutil, prev_rss)
                self._snapshots.append(snap)
                prev_rss = snap.get('rss_gb')
            except Exception as e:
                log.warning(f"[MONITOR] Snapshot failed: {e}")

    def _take_snapshot(self, psutil, prev_rss):
        proc = psutil.Process()
        mem = psutil.virtual_memory()
        rss_gb = proc.memory_info().rss / (1024**3)
        avail_gb = mem.available / (1024**3)
        total_gb = mem.total / (1024**3)
        cpu_pct = psutil.cpu_percent(interval=1)
        swap_gb = psutil.swap_memory().used / (1024**3)

        snap = {
            'time': time.time(),
            'cpu_pct': cpu_pct,
            'rss_gb': rss_gb,
            'avail_gb': avail_gb,
            'swap_gb': swap_gb,
        }

        # CPU utilization check
        cores = os.cpu_count() or 1
        if cpu_pct < 30:
            self._low_cpu_count += 1
            if self._low_cpu_count >= 2:
                log.warning(f"[MONITOR] CPU {cpu_pct:.0f}% for {self._low_cpu_count} checks — likely single-threaded on {cores} cores!")
        else:
            self._low_cpu_count = 0

        # Memory growth check
        if prev_rss and rss_gb > prev_rss * 1.2:
            log.warning(f"[MONITOR] RSS grew {prev_rss:.1f}GB -> {rss_gb:.1f}GB (+{(rss_gb-prev_rss):.1f}GB) — possible memory leak")

        # Low available RAM
        if avail_gb < total_gb * 0.10:
            self._high_mem_count += 1
            if self._high_mem_count >= 2:
                log.warning(f"[MONITOR] Only {avail_gb:.1f}GB free of {total_gb:.0f}GB — approaching OOM!")
        else:
            self._high_mem_count = 0

        # Swap usage
        if swap_gb > 1.0:
            log.warning(f"[MONITOR] Swap={swap_gb:.1f}GB — performance catastrophically degraded. Training is thrashing.")

        # GPU check
        self._check_gpu()

        # Periodic status log
        log.info(f"[MONITOR] CPU={cpu_pct:.0f}% | RSS={rss_gb:.1f}GB | Free={avail_gb:.1f}GB | Swap={swap_gb:.1f}GB")
        return snap

    def _check_gpu(self):
        n_expected = int(os.environ.get('LGBM_NUM_GPUS', '0'))
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                return
            active_gpus = 0
            gpu_lines = result.stdout.strip().split('\n')
            for i, line in enumerate(gpu_lines):
                parts = [x.strip() for x in line.split(',')]
                if len(parts) >= 3:
                    gpu_util = int(parts[0])
                    mem_used = int(parts[1])
                    mem_total = int(parts[2])
                    mem_pct = mem_used / mem_total * 100 if mem_total > 0 else 0
                    if mem_used > 100:
                        active_gpus += 1
                    if gpu_util < 5 and mem_used > 100:  # GPU has data but isn't computing
                        log.warning(f"[MONITOR] GPU{i} util={gpu_util}% but mem={mem_pct:.0f}% — GPU stalled or between trials")
                    if mem_pct > 90:
                        log.warning(f"[MONITOR] GPU{i} VRAM {mem_pct:.0f}% — approaching OOM!")

            # Multi-GPU: verify expected GPUs are active
            if n_expected > 0 and active_gpus == 0:
                log.warning(f"[MONITOR] 0/{n_expected} GPUs have VRAM allocated — cuda_sparse may have silently fallen back to CPU!")
            elif n_expected > 1 and active_gpus < n_expected:
                log.warning(f"[MONITOR] Only {active_gpus}/{n_expected} GPUs active — some trials may be CPU-only. "
                            f"Check device_type=cuda_sparse in params.")
        except Exception:
            pass

    def report(self):
        """Print summary after training completes."""
        if not self._snapshots:
            return
        cpus = [s['cpu_pct'] for s in self._snapshots]
        rsss = [s['rss_gb'] for s in self._snapshots]
        log.info(f"[MONITOR] Summary: {len(self._snapshots)} snapshots over {(self._snapshots[-1]['time'] - self._snapshots[0]['time'])/60:.0f} min")
        log.info(f"  CPU: avg={np.mean(cpus):.0f}% min={np.min(cpus):.0f}% max={np.max(cpus):.0f}%")
        log.info(f"  RSS: start={rsss[0]:.1f}GB end={rsss[-1]:.1f}GB peak={np.max(rsss):.1f}GB")
        if rsss[-1] > rsss[0] * 1.5:
            log.warning(f"  RSS grew {rsss[-1]/rsss[0]:.1f}x during training — investigate memory leak")


# ══════════════════════════════════════════════════════════════
# LAYER 3: Post-Trial Validation
# ══════════════════════════════════════════════════════════════
_trial_times = []


def post_trial_check(trial_number, trial_value, trial_params, elapsed_seconds,
                     feature_importance=None):
    """Run after each Optuna trial. Warns on anomalies."""
    global _trial_times
    _trial_times.append(elapsed_seconds)

    # NaN/Inf check
    if trial_value is None or not np.isfinite(trial_value):
        log.error(f"[TRIAL {trial_number}] Loss is {trial_value} — trial produced garbage. "
                  f"Check for NaN in features or labels.")
        return

    # Timing anomaly
    if len(_trial_times) >= 3:
        median_time = np.median(_trial_times[:-1])
        if elapsed_seconds > median_time * 3:
            log.warning(f"[TRIAL {trial_number}] Took {elapsed_seconds:.0f}s — {elapsed_seconds/median_time:.1f}x median ({median_time:.0f}s). "
                        f"Possible memory thrashing or EFB rebuild.")

    # Accuracy check (multiclass: random baseline = 33%)
    # mlogloss: random baseline for 3 classes = -ln(1/3) = 1.099
    if trial_value > 1.05:
        log.warning(f"[TRIAL {trial_number}] Loss={trial_value:.4f} > random baseline (1.099). "
                    f"Model learned nothing useful. Check feature loading.")

    # Feature importance: esoteric signals present?
    if feature_importance is not None:
        esoteric_prefixes = ('gem_', 'dr_', 'moon_', 'vedic_', 'bazi_', 'tzolkin',
                             'hebrew_', 'shmita', 'sw_', 'schumann_', 'arabic_',
                             'asp_', 'eclipse_', 'equinox_')
        top_100 = list(feature_importance.keys())[:100] if isinstance(feature_importance, dict) else []
        esoteric_in_top = sum(1 for f in top_100 if any(f.startswith(p) for p in esoteric_prefixes))
        if esoteric_in_top == 0 and len(top_100) > 0:
            log.warning(f"[TRIAL {trial_number}] 0 esoteric features in top 100 — matrix signals may be excluded. "
                        f"Check feature_fraction range.")

    log.info(f"[TRIAL {trial_number}] loss={trial_value:.4f} time={elapsed_seconds:.0f}s "
             f"params={{ {', '.join(f'{k}={v}' for k,v in list(trial_params.items())[:4])} }}")


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════
def _get_ram_gb():
    """Get available RAM in GB (cgroup-aware)."""
    import platform
    if platform.system() != 'Linux':
        try:
            import psutil
            return psutil.virtual_memory().total / (1024**3)
        except Exception:
            return 0
    for path in ['/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory/memory.limit_in_bytes']:
        try:
            with open(path) as f:
                val = f.read().strip()
                if val != 'max':
                    v = int(val)
                    if v < 2**62:
                        return v / (1024**3)
        except Exception:
            continue
    try:
        import psutil
        return psutil.virtual_memory().total / (1024**3)
    except Exception:
        return 0
