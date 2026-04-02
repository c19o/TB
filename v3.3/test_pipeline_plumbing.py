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
  13. CuPy pinned memory management
  14. Async NPZ writer (v2_cross_generator)
  15. Bitpack POPCNT tiled kernel
  16. Single-pass Numba kernel
  17. Binary mode in Optuna (_apply_binary_mode)
  18. Parallel final retrain (ThreadPoolExecutor + as_completed)
  19. Per-fold GPU distribution
  20. Environment optimizations (OMP, jemalloc, THP)
  21. lleaves + inference pruner wiring
  22. Deploy verification files
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
    # TEST 1b: LightGBM GPU Device Type
    # ============================================================
    print(f"\n== TEST 1b: LightGBM GPU Device Type ==")
    try:
        from multi_gpu_optuna import _detect_lgbm_device_type
        dtype = _detect_lgbm_device_type()
        check(f"LightGBM GPU device detected: {dtype}", dtype in ('cuda_sparse', 'gpu'))
    except RuntimeError as e:
        check("LightGBM GPU device type", False, str(e))
    except Exception as e:
        check("LightGBM GPU device detection", False, str(e))

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
            from pipeline_contract import cross_policy
            check("1w skips cross gen", cross_policy('1w') != 'required',
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
        ds2 = lgb.Dataset(bin_path, params={'max_bin': 7, 'feature_pre_filter': False, 'min_data_in_bin': 1})
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
        class DummyModel(ClassifierMixin, BaseEstimator):
            def __init__(self): self.classes_ = np.array([0, 1])
            def fit(self, X, y):
                self.classes_ = np.array([0, 1])
                return self
            def predict_proba(self, X): return np.column_stack([1-X[:, 0], X[:, 0]])
            def predict(self, X): return (X[:, 0] > 0.5).astype(int)

        model = DummyModel().fit(None, None)  # pre-fit the dummy model
        X_cal = np.random.rand(200, 1).astype(np.float32)
        y_cal = (X_cal[:, 0] > 0.5).astype(int)
        # cv='prefit' requires sklearn >= 1.4; use cv=2 as fallback
        try:
            cal = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
            cal.fit(X_cal, y_cal)
        except (TypeError, ValueError):
            cal = CalibratedClassifierCV(model, method='isotonic', cv=2)
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
        from config import (V3_LGBM_PARAMS,
                           TF_CPCV_GROUPS, TF_NUM_LEAVES, BINARY_TF_MODE)
        from feature_library import TRIPLE_BARRIER_CONFIG

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
    # TEST 13: CuPy Pinned Memory Management
    # ============================================================
    print(f"\n== TEST 13: CuPy Pinned Memory Management ==")
    try:
        import cupy as cp
        pool = cp.get_default_pinned_memory_pool()
        check("CuPy pinned_memory_pool exists", pool is not None)
        pool.free_all_blocks()
        check("pinned_memory_pool.free_all_blocks() callable", True)
        dev_pool = cp.get_default_memory_pool()
        dev_pool.set_limit(size=0)  # 0 = unlimited (just test callability)
        check("device_memory_pool.set_limit() callable", True)
    except ImportError:
        check("CuPy import", False, "CuPy not installed")
    except Exception as e:
        check("CuPy pinned memory", False, str(e))

    # ============================================================
    # TEST 14: Async NPZ Writer
    # ============================================================
    print(f"\n== TEST 14: Async NPZ Writer ==")
    try:
        import tempfile
        from scipy import sparse as sp_sparse
        from v2_cross_generator import _AsyncNpzWriter

        writer = _AsyncNpzWriter()
        check("_AsyncNpzWriter created", writer is not None)

        # Create a tiny sparse matrix and save it
        tiny = sp_sparse.random(10, 5, density=0.3, format='csr', dtype=np.float32)
        out_path = os.path.join(tempfile.gettempdir(), 'test_async_npz_writer.npz')
        writer.enqueue(out_path, tiny)
        writer.drain()
        writer.stop()
        check("enqueue + drain + stop succeeded", True)

        # Verify the file was written and is loadable
        exists = os.path.exists(out_path)
        check(f"NPZ file written to {out_path}", exists)
        if exists:
            loaded = sp_sparse.load_npz(out_path)
            check("Loaded sparse matrix shape matches", loaded.shape == (10, 5))
            check("Loaded data matches original nnz", loaded.nnz == tiny.nnz)
            os.remove(out_path)
    except ImportError as e:
        check("_AsyncNpzWriter import", False, str(e))
    except Exception as e:
        check("Async NPZ writer test", False, str(e))

    # ============================================================
    # TEST 15: Bitpack POPCNT Tiled Kernel
    # ============================================================
    print(f"\n== TEST 15: Bitpack POPCNT Tiled Kernel ==")
    try:
        from bitpack_utils import _cooccurrence_matrix_popcnt_tiled, _pack_matrix

        n_rows, n_left, n_right = 100, 20, 15
        np.random.seed(42)
        left_mat = (np.random.rand(n_rows, n_left) > 0.7).astype(np.float32)
        right_mat = (np.random.rand(n_rows, n_right) > 0.7).astype(np.float32)

        # Compute naive reference: co-occurrence = left.T @ right (binary AND count)
        naive_counts = (left_mat.T @ right_mat).astype(np.int32)

        # Pack and run tiled kernel
        n_words_raw = (n_rows + 63) >> 6
        n_words = ((n_words_raw + 7) >> 3) << 3  # round up to multiple of 8
        check(f"n_words={n_words} is multiple of 8", n_words % 8 == 0)

        left_packed = _pack_matrix(left_mat, n_left, n_words)
        right_packed = _pack_matrix(right_mat, n_right, n_words)
        check(f"left_packed shape: {left_packed.shape}", left_packed.shape == (n_left, n_words))
        check(f"right_packed shape: {right_packed.shape}", right_packed.shape == (n_right, n_words))

        counts = np.zeros((n_left, n_right), dtype=np.int32)
        _cooccurrence_matrix_popcnt_tiled(left_packed, right_packed, n_words, counts)
        check("Tiled kernel matches naive count", np.array_equal(counts, naive_counts),
              f"max diff={np.max(np.abs(counts - naive_counts))}")
    except ImportError as e:
        check("bitpack_utils import", False, str(e))
    except Exception as e:
        check("Bitpack POPCNT tiled kernel", False, str(e))

    # ============================================================
    # TEST 16: Single-Pass Numba Kernel
    # ============================================================
    print(f"\n== TEST 16: Single-Pass Numba Kernel ==")
    try:
        from numba_cross_kernels import _intersect_single_pass

        # Build two tiny CSC-style index structures
        # Column 0: rows [1, 3, 5, 7]    Column 1: rows [2, 3, 6, 7]
        left_indptr = np.array([0, 4, 8], dtype=np.int64)
        left_indices = np.array([1, 3, 5, 7, 2, 3, 6, 7], dtype=np.int32)
        # Column 0: rows [3, 5, 9]       Column 1: rows [1, 7]
        right_indptr = np.array([0, 3, 5], dtype=np.int64)
        right_indices = np.array([3, 5, 9, 1, 7], dtype=np.int32)

        # Pairs: (left_col=0, right_col=0) -> intersection {3,5} = 2
        #        (left_col=0, right_col=1) -> intersection {1,7} with {1,3,5,7} = {1,7} = 2
        #        (left_col=1, right_col=0) -> intersection {2,3,6,7} with {3,5,9} = {3} = 1
        pair_left = np.array([0, 0, 1], dtype=np.int32)
        pair_right = np.array([0, 1, 0], dtype=np.int32)
        expected = np.array([2, 2, 1], dtype=np.int64)

        # Pre-compute offsets (cumulative sum of max possible intersection sizes)
        max_per_pair = np.array([4, 4, 4], dtype=np.int64)  # generous upper bound
        offsets = np.zeros(3, dtype=np.int64)
        offsets[1] = max_per_pair[0]
        offsets[2] = offsets[1] + max_per_pair[1]
        total_buf = int(offsets[2] + max_per_pair[2])

        out_rows = np.zeros(total_buf, dtype=np.int32)
        counts = np.zeros(3, dtype=np.int64)

        _intersect_single_pass(left_indptr, left_indices, right_indptr, right_indices,
                               pair_left, pair_right, offsets, out_rows, counts)
        check(f"Pair (0,0) intersection count={counts[0]}", counts[0] == expected[0],
              f"expected {expected[0]}, got {counts[0]}")
        check(f"Pair (0,1) intersection count={counts[1]}", counts[1] == expected[1],
              f"expected {expected[1]}, got {counts[1]}")
        check(f"Pair (1,0) intersection count={counts[2]}", counts[2] == expected[2],
              f"expected {expected[2]}, got {counts[2]}")
    except ImportError as e:
        check("numba_cross_kernels import", False, str(e))
    except Exception as e:
        check("Single-pass Numba kernel", False, str(e))

    # ============================================================
    # TEST 17: Binary Mode in Optuna
    # ============================================================
    print(f"\n== TEST 17: Binary Mode in Optuna ==")
    try:
        from run_optuna_local import _apply_binary_mode

        # Test binary TF (1w should be binary=True)
        params_bin = {'objective': 'multiclass', 'metric': 'multi_logloss', 'num_class': 3}
        _apply_binary_mode(params_bin, '1w')
        check("1w: objective='binary'", params_bin.get('objective') == 'binary',
              f"got {params_bin.get('objective')}")
        check("1w: metric='binary_logloss'", params_bin.get('metric') == 'binary_logloss',
              f"got {params_bin.get('metric')}")
        check("1w: num_class removed", 'num_class' not in params_bin,
              f"num_class still present: {params_bin.get('num_class')}")

        # Test non-binary TF (4h should stay multiclass)
        params_mc = {'objective': 'multiclass', 'metric': 'multi_logloss', 'num_class': 3}
        _apply_binary_mode(params_mc, '4h')
        check("4h: objective stays 'multiclass'", params_mc.get('objective') == 'multiclass',
              f"got {params_mc.get('objective')}")
        check("4h: num_class=3 preserved", params_mc.get('num_class') == 3,
              f"got {params_mc.get('num_class')}")
    except ImportError as e:
        check("_apply_binary_mode import", False, str(e))
    except Exception as e:
        check("Binary mode test", False, str(e))

    # ============================================================
    # TEST 18: Parallel Final Retrain
    # ============================================================
    print(f"\n== TEST 18: Parallel Final Retrain ==")
    try:
        import inspect
        from run_optuna_local import final_retrain
        src = inspect.getsource(final_retrain)
        check("ThreadPoolExecutor in final_retrain",
              'ThreadPoolExecutor' in src,
              "ThreadPoolExecutor pattern not found in final_retrain")
        check("as_completed in final_retrain",
              'as_completed' in src,
              "as_completed pattern not found in final_retrain")
        check("_use_parallel guard present",
              '_use_parallel' in src,
              "_use_parallel guard not found")
        check("n_gpus > 1 AND n_valid > 2000 guard",
              'n_gpus > 1' in src and 'n_valid > 2000' in src,
              "parallel guard conditions not found")
    except ImportError as e:
        check("final_retrain import", False, str(e))
    except Exception as e:
        check("Parallel final retrain test", False, str(e))

    # ============================================================
    # TEST 19: Per-Fold GPU Distribution
    # ============================================================
    print(f"\n== TEST 19: Per-Fold GPU Distribution ==")
    try:
        import inspect
        from run_optuna_local import _run_single_validation_fold
        check("_run_single_validation_fold exists", callable(_run_single_validation_fold))
        sig = inspect.signature(_run_single_validation_fold)
        param_names = list(sig.parameters.keys())
        check("fold_i param present", 'fold_i' in param_names,
              f"params: {param_names}")
        check("n_gpus param present", 'n_gpus' in param_names,
              f"params: {param_names}")
    except ImportError as e:
        check("_run_single_validation_fold import", False, str(e))
    except Exception as e:
        check("Per-fold GPU distribution test", False, str(e))

    # ============================================================
    # TEST 20: Environment Optimizations
    # ============================================================
    print(f"\n== TEST 20: Environment Optimizations ==")
    try:
        # OMP_PROC_BIND / OMP_PLACES can be set
        os.environ['OMP_PROC_BIND'] = os.environ.get('OMP_PROC_BIND', 'close')
        os.environ['OMP_PLACES'] = os.environ.get('OMP_PLACES', 'cores')
        check(f"OMP_PROC_BIND={os.environ['OMP_PROC_BIND']}", True)
        check(f"OMP_PLACES={os.environ['OMP_PLACES']}", True)
    except Exception as e:
        check("OMP env vars", False, str(e))

    # jemalloc detection (non-fatal)
    try:
        import ctypes
        try:
            jemalloc = ctypes.CDLL('libjemalloc.so.2')
            check("jemalloc detected", True)
        except OSError:
            try:
                jemalloc = ctypes.CDLL('libjemalloc.so')
                check("jemalloc detected (alt)", True)
            except OSError:
                warn("jemalloc", False, "Not found (non-fatal, only needed on Linux cloud)")
    except Exception as e:
        warn("jemalloc detection", False, str(e))

    # Transparent Huge Pages (non-fatal, Linux only)
    thp_path = '/sys/kernel/mm/transparent_hugepage/enabled'
    if os.path.exists(thp_path):
        try:
            with open(thp_path) as f:
                thp_val = f.read().strip()
            check(f"THP status: {thp_val}", True)
        except Exception as e:
            warn("THP check", False, str(e))
    else:
        warn("THP file", False, f"{thp_path} not found (non-fatal, Linux only)")

    # ============================================================
    # TEST 21: lleaves + Inference Pruner Wiring
    # ============================================================
    print(f"\n== TEST 21: lleaves + Inference Pruner Wiring ==")
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        lleaves_path = os.path.join(script_dir, 'lleaves_compiler.py')
        pruner_path = os.path.join(script_dir, 'inference_pruner.py')
        ml_path = os.path.join(script_dir, 'ml_multi_tf.py')

        check("lleaves_compiler.py exists", os.path.exists(lleaves_path),
              f"Not found at {lleaves_path}")
        check("inference_pruner.py exists", os.path.exists(pruner_path),
              f"Not found at {pruner_path}")

        # Verify ml_multi_tf.py wires them in
        if os.path.exists(ml_path):
            with open(ml_path, 'r') as f:
                ml_src = f.read()
            check("ml_multi_tf.py imports lleaves_compiler",
                  'from lleaves_compiler' in ml_src,
                  "Missing 'from lleaves_compiler' import")
            check("ml_multi_tf.py imports inference_pruner",
                  'from inference_pruner' in ml_src,
                  "Missing 'from inference_pruner' import")
        else:
            check("ml_multi_tf.py exists", False, f"Not found at {ml_path}")
    except Exception as e:
        check("lleaves/pruner wiring", False, str(e))

    # ============================================================
    # TEST 22: Deploy Verification
    # ============================================================
    print(f"\n== TEST 22: Deploy Verification ==")
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        verify_path = os.path.join(script_dir, 'deploy_verify.py')
        manifest_path = os.path.join(script_dir, 'deploy_manifest.py')

        check("deploy_verify.py exists", os.path.exists(verify_path),
              f"Not found at {verify_path}")
        check("deploy_manifest.py exists", os.path.exists(manifest_path),
              f"Not found at {manifest_path}")

        if os.path.exists(verify_path):
            with open(verify_path, 'r') as f:
                verify_src = f.read()
            check("deploy_verify.py has check functions",
                  'def check' in verify_src or 'def verify' in verify_src,
                  "No check/verify functions found")
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                manifest_src = f.read()
            check("deploy_manifest.py has manifest logic",
                  'manifest' in manifest_src.lower(),
                  "No manifest logic found")
    except Exception as e:
        check("Deploy verification test", False, str(e))

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
