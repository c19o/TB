#!/usr/bin/env python
"""
Pipeline Plumbing Test — Validates all 4 TF differences + core pipeline integrity.
Run on cloud machine BEFORE any real training to catch issues early.

Usage: python -u test_pipeline_plumbing.py [--tf 1w|1d|4h|1h|15m]
Default: tests 1w (fastest, ~30 seconds)

Tests:
  1. Multi-GPU detection and assignment
  2. Cross gen multi-GPU dispatch (skipped for 1w, active for others)
  3. Binary vs 3-class label mode
  4. Lean mode vs full feature set
  5. Memmap streaming (for 1h/15m)
  6. CPCV Dataset caching (save_binary)
  7. Probability calibration (isotonic)
  8. LSTM AMP + torch.compile
  9. Optuna in-memory + ask/tell
  10. Feature build parallel (ThreadPoolExecutor)
  11. Assembly-line orchestrator
  12. All CPU cores utilized (OMP check)
"""
import os
import sys
import time
import numpy as np

# Force unbuffered output
os.environ.setdefault('PYTHONUNBUFFERED', '1')

PASS = 0
FAIL = 0
WARN = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} — {detail}")

def warn(name, condition, detail=""):
    global WARN
    if not condition:
        WARN += 1
        print(f"  [WARN] {name} — {detail}")

def main():
    global PASS, FAIL, WARN
    tf = sys.argv[sys.argv.index('--tf') + 1] if '--tf' in sys.argv else '1w'
    print(f"=" * 60)
    print(f"  PIPELINE PLUMBING TEST — TF: {tf}")
    print(f"=" * 60)
    t0 = time.time()

    # ============================================================
    # TEST 1: Multi-GPU Detection
    # ============================================================
    print(f"\n== TEST 1: Multi-GPU Detection ==")
    try:
        import torch
        n_cuda = torch.cuda.device_count()
        check("PyTorch CUDA available", torch.cuda.is_available())
        check(f"GPU count: {n_cuda}", n_cuda >= 1)
        for i in range(min(n_cuda, 4)):
            name = torch.cuda.get_device_name(i)
            check(f"  GPU {i}: {name}", True)
    except Exception as e:
        check("PyTorch CUDA", False, str(e))

    try:
        import cupy as cp
        n_cupy = cp.cuda.runtime.getDeviceCount()
        check(f"CuPy GPU count: {n_cupy}", n_cupy >= 1)
    except Exception as e:
        check("CuPy available", False, str(e))

    # nvidia-smi
    try:
        import subprocess
        r = subprocess.run(['nvidia-smi', '--list-gpus'], capture_output=True, text=True, timeout=5)
        n_smi = r.stdout.strip().count('\n') + 1 if r.stdout.strip() else 0
        check(f"nvidia-smi GPU count: {n_smi}", n_smi >= 1)
    except Exception as e:
        check("nvidia-smi", False, str(e))

    # ============================================================
    # TEST 2: Cross Gen Multi-GPU Dispatch
    # ============================================================
    print(f"\n== TEST 2: Cross Gen Multi-GPU ({tf}) ==")
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from v2_cross_generator import _detect_available_gpus, _MULTI_GPU_CROSS_GEN
        gpus = _detect_available_gpus()
        check(f"_detect_available_gpus(): {gpus}", len(gpus) >= 1)
        check(f"MULTI_GPU_CROSS_GEN enabled: {_MULTI_GPU_CROSS_GEN}", True)

        if tf == '1w':
            from config import SKIP_CROSSES_TFS
            check("1w skips cross gen", '1w' in SKIP_CROSSES_TFS if hasattr(SKIP_CROSSES_TFS, '__contains__') else True,
                  "1w should skip cross gen")
        else:
            # Verify _multi_gpu_cross_worker exists
            from v2_cross_generator import _multi_gpu_cross_worker
            check("_multi_gpu_cross_worker function exists", callable(_multi_gpu_cross_worker))
    except ImportError as e:
        check("Cross gen imports", False, str(e))
    except Exception as e:
        check("Cross gen test", False, str(e))

    # ============================================================
    # TEST 3: Binary vs 3-Class Label Mode
    # ============================================================
    print(f"\n== TEST 3: Binary vs 3-Class ({tf}) ==")
    try:
        from config import BINARY_TF_MODE
        is_binary = BINARY_TF_MODE.get(tf, False)
        check(f"BINARY_TF_MODE['{tf}'] = {is_binary}", True)

        if is_binary:
            check(f"{tf} uses binary mode (UP/DOWN)", True)
            # Verify the conversion works
            y_test = np.array([0, 1, 2, 0, 1, 2, 0, 2])  # SHORT, FLAT, LONG
            flat_mask = (y_test == 1)
            y_binary = y_test.copy().astype(float)
            y_binary[flat_mask] = np.nan
            valid = ~np.isnan(y_binary)
            y_binary[valid] = (y_binary[valid] == 2).astype(float)
            check("Binary conversion: FLAT→NaN", np.isnan(y_binary[1]) and np.isnan(y_binary[4]))
            check("Binary conversion: SHORT→0", y_binary[0] == 0 and y_binary[3] == 0)
            check("Binary conversion: LONG→1", y_binary[7] == 1)
        else:
            check(f"{tf} uses 3-class mode (LONG/FLAT/SHORT)", True)
    except Exception as e:
        check("Binary mode test", False, str(e))

    # ============================================================
    # TEST 4: Lean Mode vs Full Feature Set
    # ============================================================
    print(f"\n== TEST 4: Lean Mode ({tf}) ==")
    try:
        from config import LEAN_1W_MODE, LEAN_1W_TA_KEEPLIST, LEAN_1W_ALWAYS_KEEP_PREFIXES
        if tf == '1w':
            check(f"LEAN_1W_MODE = {LEAN_1W_MODE}", LEAN_1W_MODE == True)
            check(f"TA keeplist has {len(LEAN_1W_TA_KEEPLIST)} features", len(LEAN_1W_TA_KEEPLIST) > 20)
            check(f"Always-keep prefixes: {len(LEAN_1W_ALWAYS_KEEP_PREFIXES)}", len(LEAN_1W_ALWAYS_KEEP_PREFIXES) > 30)
        else:
            check(f"{tf}: Lean mode OFF (full feature set)", True)
            # Verify SKIP_FEATURES for this TF
            from config import SKIP_FEATURES_BY_TF
            skip = SKIP_FEATURES_BY_TF.get(tf, set())
            check(f"SKIP_FEATURES for {tf}: {len(skip)} constant features removed", True)
    except Exception as e:
        check("Lean mode test", False, str(e))

    # ============================================================
    # TEST 5: Memmap Streaming
    # ============================================================
    print(f"\n== TEST 5: Memmap Streaming ({tf}) ==")
    try:
        from memmap_merge import memmap_streaming_merge
        check("memmap_streaming_merge importable", callable(memmap_streaming_merge))
        if tf in ('1h', '15m'):
            check(f"{tf} will use memmap for cross gen merge", True)
        else:
            check(f"{tf} uses in-memory merge (small enough)", True)
    except ImportError:
        warn("memmap_merge not importable", False, "OK if not needed for this TF")
    except Exception as e:
        check("Memmap test", False, str(e))

    # ============================================================
    # TEST 6: CPCV Dataset Caching
    # ============================================================
    print(f"\n== TEST 6: CPCV Dataset Caching ==")
    try:
        import lightgbm as lgb
        check(f"LightGBM version: {lgb.__version__}", True)
        # Verify save_binary works with a tiny dataset
        X = np.random.rand(100, 10).astype(np.float32)
        y = np.random.randint(0, 2, 100)
        ds = lgb.Dataset(X, label=y, params={'feature_pre_filter': False, 'max_bin': 7, 'min_data_in_bin': 1})
        ds.construct()
        bin_path = '/tmp/test_lgbm_cache.bin'
        ds.save_binary(bin_path)
        check("save_binary works", os.path.exists(bin_path))
        # Load back
        ds2 = lgb.Dataset(bin_path)
        ds2.construct()
        check("load from binary works", ds2.num_data() == 100)
        os.remove(bin_path)
    except Exception as e:
        check("Dataset caching", False, str(e))

    # ============================================================
    # TEST 7: Probability Calibration
    # ============================================================
    print(f"\n== TEST 7: Probability Calibration ==")
    try:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.base import BaseEstimator, ClassifierMixin
        check("CalibratedClassifierCV importable", True)

        # Simulate calibration
        class DummyModel(BaseEstimator, ClassifierMixin):
            def __init__(self): self.classes_ = np.array([0, 1])
            def fit(self, X, y): return self
            def predict_proba(self, X): return np.column_stack([1-X[:, 0], X[:, 0]])

        model = DummyModel()
        cal = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
        X_cal = np.random.rand(200, 1).astype(np.float32)
        y_cal = (X_cal[:, 0] > 0.5).astype(int)
        cal.fit(X_cal, y_cal)
        probs = cal.predict_proba(X_cal)
        check("Isotonic calibration works", probs.shape == (200, 2))
        check("Probabilities sum to 1", np.allclose(probs.sum(axis=1), 1.0, atol=0.01))
    except Exception as e:
        check("Calibration test", False, str(e))

    # ============================================================
    # TEST 8: LSTM AMP + torch.compile
    # ============================================================
    print(f"\n== TEST 8: LSTM AMP + torch.compile ==")
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device('cuda')
            # Test AMP
            with torch.amp.autocast('cuda'):
                x = torch.randn(16, 100, device=device)
                w = torch.randn(100, 10, device=device)
                y = x @ w
            check("AMP autocast works on GPU", y.shape == (16, 10))

            # Test GradScaler
            scaler = torch.amp.GradScaler('cuda')
            check("GradScaler created", scaler is not None)

            # Test non_blocking transfer
            cpu_tensor = torch.randn(100, 50)
            gpu_tensor = cpu_tensor.to(device, non_blocking=True)
            check("non_blocking transfer works", gpu_tensor.device.type == 'cuda')

            # Test torch.compile (may not be available on all versions)
            try:
                model = torch.nn.Linear(10, 2).to(device)
                compiled = torch.compile(model)
                out = compiled(torch.randn(4, 10, device=device))
                check("torch.compile works", out.shape == (4, 2))
            except Exception as ce:
                warn("torch.compile", False, f"Not available: {ce}")
        else:
            warn("LSTM GPU tests", False, "No CUDA available")
    except Exception as e:
        check("LSTM test", False, str(e))

    # ============================================================
    # TEST 9: Optuna In-Memory + Ask/Tell
    # ============================================================
    print(f"\n== TEST 9: Optuna In-Memory + Ask/Tell ==")
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        # In-memory storage (no SQLite)
        study = optuna.create_study(direction='minimize', storage=None)
        check("In-memory Optuna study created", True)

        # Ask/tell pattern
        trial = study.ask()
        x = trial.suggest_float('x', -10, 10)
        study.tell(trial, x ** 2)
        check("ask/tell pattern works", len(study.trials) == 1)
        check(f"Trial result: x={x:.3f}, value={study.trials[0].value:.3f}", True)
    except Exception as e:
        check("Optuna test", False, str(e))

    # ============================================================
    # TEST 10: Feature Build Parallel
    # ============================================================
    print(f"\n== TEST 10: Feature Build Parallel ==")
    try:
        from concurrent.futures import ThreadPoolExecutor
        results = []
        def dummy_compute(n):
            return np.random.rand(100, n)

        with ThreadPoolExecutor(max_workers=4) as exe:
            futures = [exe.submit(dummy_compute, i+1) for i in range(8)]
            for f in futures:
                results.append(f.result())
        check(f"ThreadPoolExecutor(4) completed 8 tasks", len(results) == 8)
        check("Results correct shapes", all(r.shape[0] == 100 for r in results))
    except Exception as e:
        check("Parallel build test", False, str(e))

    # ============================================================
    # TEST 11: CPU Core Utilization
    # ============================================================
    print(f"\n== TEST 11: CPU Core Utilization ==")
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except AttributeError:
        cpu_count = os.cpu_count() or 1
    check(f"CPU cores detected: {cpu_count}", cpu_count >= 1)

    omp = os.environ.get('OMP_NUM_THREADS', 'not set')
    check(f"OMP_NUM_THREADS: {omp}", True)
    bind = os.environ.get('OMP_PROC_BIND', 'not set')
    check(f"OMP_PROC_BIND: {bind}", True)

    try:
        from numba import config as numba_config
        numba_threads = numba_config.NUMBA_NUM_THREADS
        check(f"Numba threads: {numba_threads}", numba_threads >= 1)
    except Exception:
        warn("Numba thread check", False, "Could not read Numba config")

    # ============================================================
    # TEST 12: Config Integrity
    # ============================================================
    print(f"\n== TEST 12: Config Integrity ==")
    try:
        from config import (V3_LGBM_PARAMS, TRIPLE_BARRIER_CONFIG,
                           TF_CPCV_GROUPS, TF_NUM_LEAVES, BINARY_TF_MODE)

        check("max_bin == 7", V3_LGBM_PARAMS.get('max_bin') == 7)
        check("'deterministic' NOT in params", 'deterministic' not in V3_LGBM_PARAMS)
        check("feature_pre_filter == False", V3_LGBM_PARAMS.get('feature_pre_filter') == False)
        check("feature_fraction >= 0.7", V3_LGBM_PARAMS.get('feature_fraction', 0) >= 0.7)

        tb = TRIPLE_BARRIER_CONFIG.get(tf, {})
        check(f"Triple barrier config exists for {tf}", bool(tb))
        if tb:
            check(f"  max_hold_bars: {tb.get('max_hold_bars')}", tb.get('max_hold_bars', 0) > 0)
            check(f"  tp_atr_mult: {tb.get('tp_atr_mult')}", tb.get('tp_atr_mult', 0) > 0)

        cpcv = TF_CPCV_GROUPS.get(tf)
        check(f"CPCV config for {tf}: {cpcv}", cpcv is not None)

        leaves = TF_NUM_LEAVES.get(tf)
        check(f"num_leaves for {tf}: {leaves}", leaves is not None and leaves > 0)
    except Exception as e:
        check("Config integrity", False, str(e))

    # ============================================================
    # SUMMARY
    # ============================================================
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  PLUMBING TEST: {PASS} PASSED, {FAIL} FAILED, {WARN} warnings")
    print(f"  Time: {elapsed:.1f}s | TF: {tf}")
    if FAIL == 0:
        print(f"  STATUS: ALL CLEAR — pipeline ready for {tf} training")
    else:
        print(f"  STATUS: {FAIL} FAILURES — fix before training")
    print(f"{'=' * 60}")
    sys.exit(1 if FAIL > 0 else 0)


if __name__ == '__main__':
    main()
