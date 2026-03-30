#!/usr/bin/env python3
"""
test_1w_end_to_end.py — Complete end-to-end GPU histogram benchmark
====================================================================

Loads real 1w cross features (v2_crosses_BTC_1w.npz), trains a LightGBM
model on CPU (baseline), benchmarks GPU histogram building via CuPy/cuSPARSE,
and reports accuracy, feature importance comparison, and estimated speedup.

Works RIGHT NOW with stock LightGBM (CPU) + CuPy (GPU histogram benchmark).
No custom LightGBM fork required.

Run from v3.3/:
    python -u gpu_histogram_fork/test_1w_end_to_end.py

Or from gpu_histogram_fork/:
    python -u test_1w_end_to_end.py

Matrix thesis: ALL features preserved. Both CPU and GPU paths use identical
feature matrices — zero filtering, zero subsetting, zero degradation.
"""

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Windows CUDA DLL path (must happen before any CuPy import)
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    _cuda_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin",
    ]
    for _p in _cuda_paths:
        if os.path.isdir(_p):
            try:
                os.add_dll_directory(_p)
            except (OSError, AttributeError):
                pass

# ---------------------------------------------------------------------------
# Resolve paths — works from v3.3/ or gpu_histogram_fork/
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_V33_DIR = (
    os.path.dirname(_THIS_DIR)
    if os.path.basename(_THIS_DIR) == "gpu_histogram_fork"
    else _THIS_DIR
)
_PROJECT_ROOT = os.path.dirname(_V33_DIR)

# Add v3.3 to path for feature_library imports
if _V33_DIR not in sys.path:
    sys.path.insert(0, _V33_DIR)

# Data directories
DB_DIR = os.environ.get("SAVAGE22_DB_DIR", _V33_DIR)
V30_DATA_DIR = os.environ.get(
    "V30_DATA_DIR",
    os.path.join(_PROJECT_ROOT, "v3.0 (LGBM)"),
)
V32_DATA_DIR = os.path.join(_PROJECT_ROOT, "v3.2_2.9M_Features")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Data Loading
# ═══════════════════════════════════════════════════════════════════════════
def _find_file(candidates, label):
    """Return first existing path from candidates, or None."""
    for p in candidates:
        if os.path.isfile(p):
            return p
    log(f"WARNING: {label} not found. Searched:")
    for p in candidates:
        log(f"  {p}")
    return None


def load_1w_data():
    """Load 1w parquet (labels + base features) + cross NPZ.

    Returns (X_combined_csr, y, feature_names, X_crosses_csr, n_base_cols)
    or None if data files are missing.
    """
    # --- Parquet (base features + labels) ---
    parquet_candidates = [
        os.path.join(DB_DIR, "v2_base_1w.parquet"),
        os.path.join(DB_DIR, "features_BTC_1w.parquet"),
        os.path.join(V32_DATA_DIR, "features_BTC_1w.parquet"),
        os.path.join(V30_DATA_DIR, "features_BTC_1w.parquet"),
    ]
    parquet_path = _find_file(parquet_candidates, "1w parquet")
    if parquet_path is None:
        return None

    log(f"Loading parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    log(f"  {len(df)} rows x {len(df.columns)} columns")

    # --- Labels (triple-barrier: 0=SHORT, 1=FLAT, 2=LONG) ---
    if "triple_barrier_label" in df.columns:
        y = pd.to_numeric(df["triple_barrier_label"], errors="coerce").values
        log(f"  Using pre-computed triple_barrier_label")
    elif "close" in df.columns and "high" in df.columns and "low" in df.columns:
        log(f"  Computing triple-barrier labels from OHLC...")
        y = _compute_labels_from_ohlc(df)
    else:
        log("ERROR: No labels and no OHLC data to compute them")
        return None

    valid = ~np.isnan(y)
    log(f"  Labels: LONG={int((y==2).sum())}, SHORT={int((y==0).sum())}, "
        f"FLAT={int((y==1).sum())}, NaN={int((~valid).sum())}")

    # --- Feature columns (exclude meta + target-like) ---
    meta_cols = {
        "timestamp", "date", "open", "high", "low", "close", "volume",
        "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote",
        "open_time", "date_norm",
    }
    target_like = {
        c for c in df.columns
        if "next_" in c.lower() or c == "triple_barrier_label"
    }
    # Keep directional features that are inputs (wyckoff, bos, knn, etc.)
    # but exclude next_* and the label itself
    exclude = meta_cols | target_like
    feature_cols = [c for c in df.columns if c not in exclude]
    log(f"  Base feature columns: {len(feature_cols)}")

    X_base = df[feature_cols].values.astype(np.float32)
    X_base = np.where(np.isinf(X_base), np.nan, X_base)
    n_base_cols = len(feature_cols)

    # --- Cross features NPZ ---
    npz_candidates = [
        os.path.join(DB_DIR, "v2_crosses_BTC_1w.npz"),
        os.path.join(V32_DATA_DIR, "v2_crosses_BTC_1w.npz"),
        os.path.join(V30_DATA_DIR, "v2_crosses_BTC_1w.npz"),
    ]
    npz_path = _find_file(npz_candidates, "v2_crosses_BTC_1w.npz")

    cross_matrix = None
    cross_names = []

    if npz_path is not None:
        log(f"Loading crosses: {npz_path}")
        cross_matrix = sp.load_npz(npz_path).tocsr()

        # Enforce LightGBM-compatible dtypes
        if cross_matrix.indices.dtype != np.int32:
            cross_matrix.indices = cross_matrix.indices.astype(np.int32)
        if cross_matrix.indptr.dtype != np.int64:
            cross_matrix.indptr = cross_matrix.indptr.astype(np.int64)
        if cross_matrix.data.dtype != np.float64:
            cross_matrix.data = cross_matrix.data.astype(np.float64)

        # Load column names
        names_candidates = [
            npz_path.replace(".npz", "_columns.json"),
            os.path.join(DB_DIR, "v2_cross_names_BTC_1w.json"),
            os.path.join(V32_DATA_DIR, "v2_cross_names_BTC_1w.json"),
            os.path.join(V30_DATA_DIR, "v2_cross_names_BTC_1w.json"),
        ]
        names_path = _find_file(names_candidates, "cross names JSON")
        if names_path:
            with open(names_path) as f:
                cross_names = json.load(f)
        else:
            cross_names = [f"cross_{i}" for i in range(cross_matrix.shape[1])]

        density = 100 * cross_matrix.nnz / max(1, cross_matrix.shape[0] * cross_matrix.shape[1])
        log(f"  Crosses: {cross_matrix.shape[0]} x {cross_matrix.shape[1]:,} "
            f"({cross_matrix.nnz:,} nnz, {density:.4f}% dense)")

        if cross_matrix.shape[0] != X_base.shape[0]:
            log(f"  WARNING: cross rows ({cross_matrix.shape[0]}) != base rows "
                f"({X_base.shape[0]}), skipping crosses")
            cross_matrix = None
            cross_names = []
    else:
        log("  No cross NPZ found, using base features only")

    # --- Combine base + crosses into single sparse CSR ---
    if cross_matrix is not None:
        X_base_sp = sp.csr_matrix(X_base)
        X_combined = sp.hstack([X_base_sp, cross_matrix], format="csr")
        all_names = feature_cols + cross_names
        log(f"  Combined: {X_combined.shape[0]} x {X_combined.shape[1]:,} "
            f"({n_base_cols} base + {len(cross_names):,} crosses)")
    else:
        X_combined = sp.csr_matrix(X_base)
        all_names = feature_cols

    return X_combined, y, all_names, cross_matrix, n_base_cols


def _compute_labels_from_ohlc(df):
    """Fallback triple-barrier labeling using ATR."""
    c = df["close"].astype(float).values
    h = df["high"].astype(float).values
    lo = df["low"].astype(float).values
    n = len(c)

    # ATR(14)
    tr = np.empty(n)
    tr[0] = h[0] - lo[0]
    for i in range(1, n):
        tr[i] = max(h[i] - lo[i], abs(h[i] - c[i - 1]), abs(lo[i] - c[i - 1]))
    atr = np.full(n, np.nan)
    for i in range(13, n):
        atr[i] = np.mean(tr[i - 13 : i + 1])

    # Triple-barrier: tp=3xATR, sl=3xATR, max_hold=6
    labels = np.full(n, np.nan)
    tp_mult, sl_mult, max_hold = 3.0, 3.0, 6
    for i in range(n):
        if np.isnan(atr[i]) or np.isnan(c[i]):
            continue
        tp_price = c[i] + tp_mult * atr[i]
        sl_price = c[i] - sl_mult * atr[i]
        end_bar = min(i + max_hold, n - 1)
        if i + 1 > end_bar:
            continue
        hit = False
        for j in range(i + 1, end_bar + 1):
            if h[j] >= tp_price:
                labels[i] = 2.0
                hit = True
                break
            if lo[j] <= sl_price:
                labels[i] = 0.0
                hit = True
                break
        if not hit:
            labels[i] = 1.0
    return labels


# ═══════════════════════════════════════════════════════════════════════════
# 2. Chronological Train/Test Split
# ═══════════════════════════════════════════════════════════════════════════
def split_data(X, y, holdout_frac=0.2):
    """Chronological split on valid-label rows. No shuffle, no lookahead."""
    valid_idx = np.where(~np.isnan(y))[0]
    n_train = int(len(valid_idx) * (1 - holdout_frac))

    train_idx = valid_idx[:n_train]
    test_idx = valid_idx[n_train:]

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx].astype(np.int32)
    y_test = y[test_idx].astype(np.int32)

    return X_train, X_test, y_train, y_test


# ═══════════════════════════════════════════════════════════════════════════
# 3. CPU Training (Stock LightGBM)
# ═══════════════════════════════════════════════════════════════════════════
def train_cpu(X_train, y_train, X_test, y_test):
    """Train LightGBM on CPU. Returns (model, train_time_s, accuracy, preds, probs)."""
    import lightgbm as lgb

    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "device_type": "cpu",
        "max_bin": 255,
        "num_leaves": 63,
        "learning_rate": 0.03,
        "feature_pre_filter": False,
        "force_col_wise": True,
        "is_enable_sparse": True,
        "min_data_in_leaf": 3,
        "min_gain_to_split": 2.0,
        "feature_fraction": 0.9,  # NEVER < 0.7 — kills rare esoteric EFB bundles
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "deterministic": True,
        "num_threads": 0,
        "verbosity": 0,
        "seed": 42,
    }

    log(f"\n{'=' * 60}")
    log(f"CPU TRAINING (stock LightGBM)")
    log(f"{'=' * 60}")
    log(f"  X_train: {X_train.shape[0]} x {X_train.shape[1]:,} "
        f"({'sparse' if sp.issparse(X_train) else 'dense'})")
    log(f"  Params: num_leaves={params['num_leaves']}, lr={params['learning_rate']}, "
        f"feature_fraction={params['feature_fraction']}")

    ds_params = {"feature_pre_filter": False}

    t0 = time.perf_counter()
    dtrain = lgb.Dataset(X_train, label=y_train, params=ds_params, free_raw_data=False)
    dtest = lgb.Dataset(X_test, label=y_test, reference=dtrain, params=ds_params, free_raw_data=False)
    ds_time = time.perf_counter() - t0
    log(f"  Dataset construction: {ds_time:.1f}s")

    callbacks = [
        lgb.log_evaluation(period=50),
        lgb.early_stopping(stopping_rounds=50, verbose=False),
    ]

    t0 = time.perf_counter()
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=300,
        valid_sets=[dtest],
        valid_names=["holdout"],
        callbacks=callbacks,
    )
    train_time = time.perf_counter() - t0

    # Predictions
    probs = model.predict(X_test)
    preds = np.argmax(probs, axis=1)
    accuracy = np.mean(preds == y_test) * 100

    log(f"  Training time:    {train_time:.1f}s ({train_time / 60:.2f} min)")
    log(f"  Best iteration:   {model.best_iteration}")
    log(f"  Holdout accuracy: {accuracy:.1f}%")
    log(f"  Preds: SHORT={int((preds==0).sum())}, FLAT={int((preds==1).sum())}, "
        f"LONG={int((preds==2).sum())}")

    return model, train_time, accuracy, preds, probs


# ═══════════════════════════════════════════════════════════════════════════
# 4. GPU Histogram Benchmark (CuPy/cuSPARSE)
# ═══════════════════════════════════════════════════════════════════════════
def benchmark_gpu_histograms(X_crosses_csr):
    """Benchmark cuSPARSE SpMV histogram building on real cross feature matrix.

    Returns dict with timing results, or None if GPU unavailable.
    """
    if X_crosses_csr is None:
        log("\nSKIP: No cross matrix available for GPU histogram benchmark")
        return None

    try:
        import cupy as cp
        from cupyx.scipy import sparse as cp_sparse
    except ImportError:
        log("\nSKIP: CuPy not installed (pip install cupy-cuda12x)")
        return None
    except Exception as e:
        log(f"\nSKIP: GPU not available: {e}")
        return None

    n_rows, n_cols = X_crosses_csr.shape
    nnz = X_crosses_csr.nnz
    density = 100 * nnz / max(1, n_rows * n_cols)

    log(f"\n{'=' * 60}")
    log(f"GPU HISTOGRAM BENCHMARK (cuSPARSE SpMV)")
    log(f"{'=' * 60}")
    log(f"  Matrix: {n_rows:,} x {n_cols:,} ({nnz:,} nnz, {density:.4f}% dense)")
    csr_mb = (X_crosses_csr.data.nbytes + X_crosses_csr.indices.nbytes +
              X_crosses_csr.indptr.nbytes) / 1e6
    log(f"  CSR size: {csr_mb:.1f} MB")

    # GPU info
    try:
        props = cp.cuda.runtime.getDeviceProperties(0)
        gpu_name = props.get("name", b"Unknown")
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode()
    except Exception:
        gpu_name = "Unknown"
    vram_free, vram_total = cp.cuda.Device(0).mem_info
    log(f"  GPU: {gpu_name}")
    log(f"  VRAM: {vram_total / (1024**3):.1f} GB total, {vram_free / (1024**3):.1f} GB free")

    # Simulated gradient + hessian (mimics LightGBM 3-class training)
    rng = np.random.default_rng(42)
    grad_cpu = rng.standard_normal(n_rows).astype(np.float64)
    hess_cpu = np.abs(rng.standard_normal(n_rows).astype(np.float64)) + 0.1

    # ─── CPU baseline (scipy) ────────────────────────────────────────────
    log(f"\n  --- CPU Histogram (scipy SpMV) ---")
    X_T_cpu = X_crosses_csr.T.tocsr()

    # Warmup
    _ = X_T_cpu @ grad_cpu

    n_cpu_iters = 20
    cpu_times = []
    for _ in range(n_cpu_iters):
        t0 = time.perf_counter()
        hist_grad_cpu = X_T_cpu @ grad_cpu
        hist_hess_cpu = X_T_cpu @ hess_cpu
        cpu_times.append(time.perf_counter() - t0)

    cpu_median_ms = np.median(cpu_times) * 1000
    log(f"  CPU (grad+hess): median={cpu_median_ms:.1f} ms over {n_cpu_iters} iters")

    # ─── GPU upload + pre-transpose ──────────────────────────────────────
    log(f"\n  --- GPU Setup ---")
    t0 = time.perf_counter()
    csr_gpu = cp_sparse.csr_matrix(X_crosses_csr)
    cp.cuda.Stream.null.synchronize()
    upload_ms = (time.perf_counter() - t0) * 1000
    log(f"  Upload to GPU: {upload_ms:.1f} ms")

    t0 = time.perf_counter()
    X_AT_gpu = csr_gpu.T.tocsr()
    cp.cuda.Stream.null.synchronize()
    transpose_ms = (time.perf_counter() - t0) * 1000
    log(f"  Pre-transpose (one-time): {transpose_ms:.1f} ms")

    grad_gpu = cp.asarray(grad_cpu)
    hess_gpu = cp.asarray(hess_cpu)

    # ─── GPU SpMV benchmark ──────────────────────────────────────────────
    log(f"\n  --- GPU Histogram (cuSPARSE SpMV) ---")
    n_warmup = 10
    n_gpu_iters = 100

    # Warmup
    for _ in range(n_warmup):
        _ = X_AT_gpu @ grad_gpu
        cp.cuda.Stream.null.synchronize()

    # Benchmark: 2x SpMV (grad + hess)
    gpu_times = []
    for _ in range(n_gpu_iters):
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        h_grad = X_AT_gpu @ grad_gpu
        h_hess = X_AT_gpu @ hess_gpu
        cp.cuda.Stream.null.synchronize()
        gpu_times.append(time.perf_counter() - t0)

    gpu_median_ms = np.median(gpu_times) * 1000
    gpu_min_ms = np.min(gpu_times) * 1000
    gpu_p95_ms = np.percentile(gpu_times, 95) * 1000

    log(f"  GPU (grad+hess, {n_gpu_iters} iters):")
    log(f"    Median: {gpu_median_ms:.3f} ms")
    log(f"    Min:    {gpu_min_ms:.3f} ms")
    log(f"    P95:    {gpu_p95_ms:.3f} ms")

    spmv_speedup = cpu_median_ms / gpu_median_ms
    log(f"    Speedup vs CPU: {spmv_speedup:.0f}x")

    # ─── Correctness verification ────────────────────────────────────────
    max_err_grad = np.max(np.abs(hist_grad_cpu - h_grad.get()))
    max_err_hess = np.max(np.abs(hist_hess_cpu - h_hess.get()))
    max_val = max(np.max(np.abs(hist_grad_cpu)), 1e-15)
    rel_err = max(max_err_grad, max_err_hess) / max_val
    correct = rel_err < 1e-6
    log(f"    Correctness: {'PASS' if correct else 'FAIL'} "
        f"(max relative error: {rel_err:.2e})")

    # ─── Effective bandwidth ─────────────────────────────────────────────
    bytes_read = nnz * (8 + 4) + (n_cols + 1) * 8 + n_rows * 8
    bytes_written = n_cols * 8
    total_bytes = (bytes_read + bytes_written) * 2  # 2x SpMV
    eff_bw = total_bytes / (gpu_median_ms / 1000) / 1e9
    log(f"    Effective bandwidth: {eff_bw:.1f} GB/s")

    # ─── Training time estimation ────────────────────────────────────────
    # LightGBM: num_leaves=63 -> ~32 histogram builds per tree (subtraction trick)
    # 3-class -> 3 trees per round
    # 300 rounds (with early stopping, typically ~200 effective)
    n_rounds = 200  # typical with early stopping
    n_classes = 3
    n_hist_per_tree = 32  # ~half of 63 leaves
    total_hist_calls = n_rounds * n_classes * n_hist_per_tree

    gpu_hist_total_s = total_hist_calls * gpu_median_ms / 1000
    cpu_hist_total_s = total_hist_calls * cpu_median_ms / 1000

    log(f"\n  --- Training Time Estimate ---")
    log(f"    Rounds: {n_rounds} (typical early-stopped)")
    log(f"    Histogram calls: {total_hist_calls:,} "
        f"({n_rounds} rounds x {n_classes} classes x {n_hist_per_tree} leaves)")
    log(f"    CPU histogram total: {cpu_hist_total_s:.1f}s ({cpu_hist_total_s/60:.1f} min)")
    log(f"    GPU histogram total: {gpu_hist_total_s:.1f}s ({gpu_hist_total_s/60:.2f} min)")

    # Histogram is ~50% of total LightGBM training time
    hist_fraction = 0.50
    gpu_train_est_s = gpu_hist_total_s / hist_fraction
    cpu_train_est_s = cpu_hist_total_s / hist_fraction

    log(f"    Estimated full CPU train: {cpu_train_est_s:.1f}s ({cpu_train_est_s/60:.1f} min)")
    log(f"    Estimated full GPU train: {gpu_train_est_s:.1f}s ({gpu_train_est_s/60:.1f} min)")
    log(f"    Estimated speedup: {cpu_train_est_s/gpu_train_est_s:.1f}x")

    # Cleanup
    del csr_gpu, X_AT_gpu, grad_gpu, hess_gpu
    cp.get_default_memory_pool().free_all_blocks()

    return {
        "gpu_name": gpu_name,
        "vram_gb": vram_total / (1024**3),
        "cpu_median_ms": cpu_median_ms,
        "gpu_median_ms": gpu_median_ms,
        "gpu_min_ms": gpu_min_ms,
        "spmv_speedup": spmv_speedup,
        "correct": correct,
        "rel_err": rel_err,
        "eff_bw_gbs": eff_bw,
        "upload_ms": upload_ms,
        "transpose_ms": transpose_ms,
        "total_hist_calls": total_hist_calls,
        "gpu_hist_total_s": gpu_hist_total_s,
        "cpu_hist_total_s": cpu_hist_total_s,
        "gpu_train_est_s": gpu_train_est_s,
        "cpu_train_est_s": cpu_train_est_s,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. Feature Importance Comparison
# ═══════════════════════════════════════════════════════════════════════════
def analyze_feature_importance(model, feature_names, n_base_cols, top_n=20):
    """Analyze which features the model actually uses."""
    log(f"\n{'=' * 60}")
    log(f"FEATURE IMPORTANCE ANALYSIS")
    log(f"{'=' * 60}")

    imp = model.feature_importance(importance_type="gain")
    n_used = int((imp > 0).sum())
    n_total = len(imp)

    log(f"  Features used: {n_used:,} / {n_total:,} ({100*n_used/n_total:.2f}%)")
    log(f"  Base features used: {int((imp[:n_base_cols] > 0).sum())} / {n_base_cols}")
    if n_total > n_base_cols:
        cross_used = int((imp[n_base_cols:] > 0).sum())
        cross_total = n_total - n_base_cols
        log(f"  Cross features used: {cross_used:,} / {cross_total:,} ({100*cross_used/cross_total:.2f}%)")

    # Top features
    top_idx = np.argsort(imp)[-top_n:][::-1]
    log(f"\n  Top-{top_n} features by gain:")
    for rank, idx in enumerate(top_idx):
        name = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
        source = "BASE" if idx < n_base_cols else "CROSS"
        log(f"    {rank+1:>3}. [{source:5s}] {name[:65]:<65s} gain={imp[idx]:.1f}")

    # Importance by category
    base_imp = imp[:n_base_cols].sum()
    cross_imp = imp[n_base_cols:].sum() if n_total > n_base_cols else 0
    total_imp = base_imp + cross_imp
    if total_imp > 0:
        log(f"\n  Importance share: BASE={100*base_imp/total_imp:.1f}%, "
            f"CROSS={100*cross_imp/total_imp:.1f}%")

    return imp


# ═══════════════════════════════════════════════════════════════════════════
# 6. Save Models + Artifacts
# ═══════════════════════════════════════════════════════════════════════════
def save_artifacts(model, accuracy, train_time, gpu_results):
    """Save model and results JSON to gpu_histogram_fork/."""
    out_dir = _THIS_DIR
    model_path = os.path.join(out_dir, "test_1w_model_cpu.txt")
    results_path = os.path.join(out_dir, "test_1w_results.json")

    try:
        model.save_model(model_path)
        log(f"  Saved CPU model: {model_path}")
    except Exception as e:
        log(f"  WARNING: Could not save model: {e}")

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cpu_accuracy": round(accuracy, 2),
        "cpu_train_time_s": round(train_time, 2),
        "best_iteration": model.best_iteration,
    }
    if gpu_results:
        results.update({
            "gpu_name": gpu_results["gpu_name"],
            "gpu_vram_gb": round(gpu_results["vram_gb"], 1),
            "spmv_speedup": round(gpu_results["spmv_speedup"], 1),
            "gpu_histogram_correct": gpu_results["correct"],
            "gpu_median_ms": round(gpu_results["gpu_median_ms"], 3),
            "cpu_histogram_median_ms": round(gpu_results["cpu_median_ms"], 1),
            "estimated_gpu_train_s": round(gpu_results["gpu_train_est_s"], 1),
            "estimated_speedup": round(gpu_results["cpu_train_est_s"] / gpu_results["gpu_train_est_s"], 1),
            "effective_bandwidth_gbs": round(gpu_results["eff_bw_gbs"], 1),
        })

    # Convert numpy types to native Python for JSON serialization
    def _to_native(obj):
        if isinstance(obj, (np.bool_, np.generic)):
            return obj.item()
        return obj

    results = {k: _to_native(v) for k, v in results.items()}
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"  Saved results: {results_path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    log("=" * 60)
    log("1W END-TO-END GPU HISTOGRAM TEST")
    log("=" * 60)
    log(f"v3.3 dir:     {_V33_DIR}")
    log(f"DB_DIR:       {DB_DIR}")
    log(f"V30_DATA_DIR: {V30_DATA_DIR}")
    log(f"V32_DATA_DIR: {V32_DATA_DIR}")

    # ── 1. Load data ─────────────────────────────────────────────────────
    result = load_1w_data()
    if result is None:
        log("\nABORT: Cannot load 1w data files")
        sys.exit(1)

    X_combined, y, feature_names, X_crosses_csr, n_base_cols = result

    # ── 2. Split ─────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = split_data(X_combined, y, holdout_frac=0.2)
    log(f"\nTrain: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
    log(f"Train labels: SHORT={int((y_train==0).sum())}, "
        f"FLAT={int((y_train==1).sum())}, LONG={int((y_train==2).sum())}")
    log(f"Test labels:  SHORT={int((y_test==0).sum())}, "
        f"FLAT={int((y_test==1).sum())}, LONG={int((y_test==2).sum())}")
    log(f"Total features: {X_train.shape[1]:,} ({n_base_cols} base + "
        f"{X_train.shape[1] - n_base_cols:,} crosses)")

    # ── 3. CPU Training ──────────────────────────────────────────────────
    model_cpu, time_cpu, acc_cpu, preds_cpu, probs_cpu = train_cpu(
        X_train, y_train, X_test, y_test
    )

    # ── 4. Feature Importance ────────────────────────────────────────────
    imp = analyze_feature_importance(model_cpu, feature_names, n_base_cols)

    # ── 5. GPU Histogram Benchmark ───────────────────────────────────────
    gpu_results = benchmark_gpu_histograms(X_crosses_csr)

    # ── 6. Save artifacts ────────────────────────────────────────────────
    save_artifacts(model_cpu, acc_cpu, time_cpu, gpu_results)

    # ── 7. Final Summary ─────────────────────────────────────────────────
    log(f"\n{'=' * 60}")
    log(f"=== 1W END-TO-END GPU TEST ===")
    log(f"{'=' * 60}")
    log(f"Data:               {X_combined.shape[0]} rows x {X_combined.shape[1]:,} features")
    log(f"                    ({n_base_cols} base + {X_combined.shape[1] - n_base_cols:,} crosses)")
    log(f"CPU Training:       {time_cpu:.1f}s ({time_cpu/60:.2f} min)")
    log(f"CPU Best Iteration: {model_cpu.best_iteration}")
    log(f"CPU Accuracy:       {acc_cpu:.1f}%")

    if gpu_results:
        log(f"")
        log(f"GPU:                {gpu_results['gpu_name']} ({gpu_results['vram_gb']:.0f} GB)")
        log(f"GPU Histogram:      {gpu_results['gpu_median_ms']:.3f} ms/call "
            f"x {gpu_results['total_hist_calls']:,} calls "
            f"= {gpu_results['gpu_hist_total_s']:.1f}s")
        log(f"CPU Histogram:      {gpu_results['cpu_median_ms']:.1f} ms/call "
            f"x {gpu_results['total_hist_calls']:,} calls "
            f"= {gpu_results['cpu_hist_total_s']:.1f}s")
        log(f"SpMV Speedup:       {gpu_results['spmv_speedup']:.0f}x")
        log(f"Est. GPU Training:  {gpu_results['gpu_train_est_s']:.1f}s "
            f"({gpu_results['gpu_train_est_s']/60:.1f} min)")
        log(f"Est. Full Speedup:  {gpu_results['cpu_train_est_s']/gpu_results['gpu_train_est_s']:.1f}x")
        log(f"Histogram Correct:  {'PASS' if gpu_results['correct'] else 'FAIL'} "
            f"(rel err: {gpu_results['rel_err']:.2e})")
        log(f"Eff. Bandwidth:     {gpu_results['eff_bw_gbs']:.1f} GB/s")
    else:
        log(f"GPU Histogram:      SKIPPED (CuPy not available)")

    log(f"")
    log(f"Baseline (v3.3 prod): 71.9% (CPCV, full Optuna tuning)")
    log(f"NOTE: This test uses simple holdout, not CPCV.")
    log(f"      Lower accuracy than baseline is expected.")
    log(f"{'=' * 60}")


if __name__ == "__main__":
    main()
