#!/usr/bin/env python3
"""
test_real_1w.py — Test GPU sparse histogram fork on REAL 1w training data.
==========================================================================

Loads actual 1w pipeline data (features_BTC_1w.parquet + v2_crosses_BTC_1w.npz),
prepares triple-barrier labels, and compares CPU vs GPU (cuda_sparse) LightGBM
training. Also benchmarks standalone cuSPARSE SpMV histogram building.

Baseline: 1w model = 71.9% accuracy, ~818 rows, ~2.2M features after cross gen.

Run from the v3.3/ directory:
    python -u gpu_histogram_fork/test_real_1w.py

Or from the gpu_histogram_fork/ directory:
    python -u test_real_1w.py
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
# Resolve paths — works whether run from v3.3/ or gpu_histogram_fork/
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_V33_DIR = os.path.dirname(_THIS_DIR) if os.path.basename(_THIS_DIR) == "gpu_histogram_fork" else _THIS_DIR

# Add v3.3 to path for feature_library imports
if _V33_DIR not in sys.path:
    sys.path.insert(0, _V33_DIR)

# Data directories — same logic as ml_multi_tf.py
DB_DIR = os.environ.get("SAVAGE22_DB_DIR", _V33_DIR)
V30_DATA_DIR = os.environ.get(
    "V30_DATA_DIR",
    os.path.join(os.path.dirname(_V33_DIR), "v3.0 (LGBM)"),
)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 1. Load real 1w data
# ---------------------------------------------------------------------------
def load_1w_data():
    """Load 1w parquet + cross NPZ. Returns (X_all, y, feature_cols, X_crosses_csr)."""
    # --- Parquet (base features) ---
    parquet_candidates = [
        os.path.join(DB_DIR, "features_BTC_1w.parquet"),
        os.path.join(V30_DATA_DIR, "features_BTC_1w.parquet"),
    ]
    parquet_path = None
    for p in parquet_candidates:
        if os.path.exists(p):
            parquet_path = p
            break
    if parquet_path is None:
        log(f"SKIP: features_BTC_1w.parquet not found. Searched:")
        for p in parquet_candidates:
            log(f"  {p}")
        return None

    log(f"Loading parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    log(f"  {len(df)} rows x {len(df.columns)} columns")

    # --- Timestamp ---
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    elif "date" in df.columns:
        df["timestamp"] = pd.to_datetime(df["date"])

    # --- Triple-barrier labels (0=SHORT, 1=FLAT, 2=LONG) ---
    if "triple_barrier_label" in df.columns:
        tb_labels = pd.to_numeric(df["triple_barrier_label"], errors="coerce").values
        log("  Using pre-computed triple_barrier_label column")
    else:
        log("  Computing triple-barrier labels on-the-fly...")
        try:
            from feature_library import compute_triple_barrier_labels
            tb_labels = compute_triple_barrier_labels(df, "1w")
        except ImportError:
            log("  WARNING: feature_library not importable, computing labels manually")
            tb_labels = _compute_labels_standalone(df)

    y = tb_labels.copy()
    valid_mask = ~np.isnan(y)
    n_long = int((y == 2).sum())
    n_short = int((y == 0).sum())
    n_flat = int((y == 1).sum())
    n_nan = int((~valid_mask).sum())
    log(f"  Labels: {n_long} LONG, {n_short} SHORT, {n_flat} FLAT, {n_nan} NaN")

    # --- Feature columns (exclude meta + target-like) ---
    meta_cols = {
        "timestamp", "date", "open", "high", "low", "close", "volume",
        "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote",
        "open_time", "date_norm",
    }
    target_like = {
        c for c in df.columns
        if "next_" in c.lower() or "target" in c.lower()
        or "direction" in c.lower() or c == "triple_barrier_label"
    }
    exclude_cols = meta_cols | target_like
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    log(f"  Base feature columns: {len(feature_cols)}")

    X_base = df[feature_cols].values.astype(np.float32)
    X_base = np.where(np.isinf(X_base), np.nan, X_base)

    # --- Cross features NPZ ---
    npz_candidates = [
        os.path.join(DB_DIR, "v2_crosses_BTC_1w.npz"),
        os.path.join(V30_DATA_DIR, "v2_crosses_BTC_1w.npz"),
    ]
    npz_path = None
    for p in npz_candidates:
        if os.path.exists(p):
            npz_path = p
            break

    cross_matrix = None
    cross_cols = []
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
        cols_path = npz_path.replace(".npz", "_columns.json")
        if not os.path.exists(cols_path):
            cols_path = os.path.join(DB_DIR, "v2_cross_names_BTC_1w.json")
        if not os.path.exists(cols_path):
            cols_path = os.path.join(V30_DATA_DIR, "v2_cross_names_BTC_1w.json")
        if os.path.exists(cols_path):
            with open(cols_path) as f:
                cross_cols = json.load(f)
        else:
            cross_cols = [f"cross_{i}" for i in range(cross_matrix.shape[1])]

        log(f"  Crosses: {cross_matrix.shape[0]} x {cross_matrix.shape[1]} "
            f"({cross_matrix.nnz:,} nnz, "
            f"{100 * cross_matrix.nnz / max(1, cross_matrix.shape[0] * cross_matrix.shape[1]):.4f}% dense)")

        if cross_matrix.shape[0] != X_base.shape[0]:
            log(f"  WARNING: cross rows ({cross_matrix.shape[0]}) != base rows ({X_base.shape[0]}), skipping crosses")
            cross_matrix = None
            cross_cols = []
    else:
        log("  No cross NPZ found, using base features only")

    # --- Combine base + crosses ---
    if cross_matrix is not None:
        X_base_sp = sp.csr_matrix(X_base)
        X_all = sp.hstack([X_base_sp, cross_matrix], format="csr")
        all_cols = feature_cols + cross_cols
        log(f"  Combined: {X_all.shape[0]} x {X_all.shape[1]:,} "
            f"({len(feature_cols)} base + {len(cross_cols)} crosses)")
    else:
        X_all = sp.csr_matrix(X_base)
        all_cols = feature_cols
        log(f"  Base only: {X_all.shape[0]} x {X_all.shape[1]:,}")

    # Store raw crosses CSR for standalone SpMV benchmark
    X_crosses_csr = cross_matrix

    return X_all, y, all_cols, X_crosses_csr, df


def _compute_labels_standalone(df):
    """Fallback triple-barrier labeling if feature_library is not importable."""
    from numba import njit

    @njit(cache=True)
    def _atr_kernel(h, l, c, n):
        tr = np.empty(n)
        tr[0] = h[0] - l[0]
        for i in range(1, n):
            hl = h[i] - l[i]
            hc = abs(h[i] - c[i - 1])
            lc = abs(l[i] - c[i - 1])
            tr[i] = max(hl, max(hc, lc))
        atr = np.full(n, np.nan)
        for i in range(13, n):
            s = 0.0
            for k in range(i - 13, i + 1):
                s += tr[k]
            atr[i] = s / 14.0
        return atr

    @njit(cache=True)
    def _label_kernel(c, h, l, atr, tp_mult, sl_mult, max_hold):
        n = len(c)
        labels = np.full(n, np.nan)
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
                if l[j] <= sl_price:
                    labels[i] = 0.0
                    hit = True
                    break
            if not hit:
                labels[i] = 1.0
        return labels

    c = df["close"].astype(float).values
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values
    # 1w config: tp=3.0xATR, sl=3.0xATR, max_hold=6
    atr = _atr_kernel(h, l, c, len(c))
    return _label_kernel(c, h, l, atr, 3.0, 3.0, 6)


# ---------------------------------------------------------------------------
# 2. Train/test split (chronological — no shuffle, no lookahead)
# ---------------------------------------------------------------------------
def split_data(X_all, y, holdout_frac=0.2):
    """Chronological train/test split. Returns only rows with valid labels."""
    valid_mask = ~np.isnan(y)
    valid_idx = np.where(valid_mask)[0]
    n_valid = len(valid_idx)
    n_train = int(n_valid * (1 - holdout_frac))

    train_idx = valid_idx[:n_train]
    test_idx = valid_idx[n_train:]

    if sp.issparse(X_all):
        X_train = X_all[train_idx]
        X_test = X_all[test_idx]
    else:
        X_train = X_all[train_idx]
        X_test = X_all[test_idx]

    y_train = y[train_idx].astype(np.int32)
    y_test = y[test_idx].astype(np.int32)

    return X_train, X_test, y_train, y_test, train_idx, test_idx


# ---------------------------------------------------------------------------
# 3. LightGBM training — CPU baseline
# ---------------------------------------------------------------------------
def train_lgbm(X_train, y_train, X_test, y_test, params, label="CPU"):
    """Train LightGBM and return (model, train_time, accuracy, predictions)."""
    import lightgbm as lgb

    log(f"\n{'='*60}")
    log(f"Training LightGBM [{label}]")
    log(f"  X_train: {X_train.shape[0]} x {X_train.shape[1]:,} "
        f"({'sparse' if sp.issparse(X_train) else 'dense'})")
    log(f"  Params: device={params.get('device_type', params.get('device', 'cpu'))}")
    log(f"{'='*60}")

    ds_params = {"feature_pre_filter": False}

    t0 = time.perf_counter()
    dtrain = lgb.Dataset(X_train, label=y_train, params=ds_params, free_raw_data=False)
    dtest = lgb.Dataset(X_test, label=y_test, reference=dtrain, params=ds_params, free_raw_data=False)
    ds_time = time.perf_counter() - t0
    log(f"  Dataset construction: {ds_time:.1f}s")

    callbacks = [lgb.log_evaluation(period=50)]
    early_stop = lgb.early_stopping(stopping_rounds=50, verbose=True)

    t0 = time.perf_counter()
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=300,
        valid_sets=[dtest],
        valid_names=["holdout"],
        callbacks=[callbacks, early_stop],
    )
    train_time = time.perf_counter() - t0

    # Predictions
    preds_raw = model.predict(X_test)
    preds = np.argmax(preds_raw, axis=1)
    accuracy = np.mean(preds == y_test) * 100

    log(f"  Training time:   {train_time:.1f}s")
    log(f"  Best iteration:  {model.best_iteration}")
    log(f"  Holdout accuracy: {accuracy:.1f}%")
    log(f"  Class distribution (pred): "
        f"SHORT={int((preds==0).sum())}, FLAT={int((preds==1).sum())}, LONG={int((preds==2).sum())}")

    return model, train_time, accuracy, preds, preds_raw


# ---------------------------------------------------------------------------
# 4. Feature importance comparison
# ---------------------------------------------------------------------------
def compare_feature_importance(model_cpu, model_gpu, feature_cols, top_n=50):
    """Compare top-N feature importance between CPU and GPU models."""
    log(f"\n{'='*60}")
    log(f"Feature Importance Comparison (top {top_n})")
    log(f"{'='*60}")

    imp_cpu = model_cpu.feature_importance(importance_type="gain")
    imp_gpu = model_gpu.feature_importance(importance_type="gain")

    # Rank correlation across all features
    from scipy.stats import spearmanr
    rho_all, pval_all = spearmanr(imp_cpu, imp_gpu)
    log(f"  Spearman rank correlation (all {len(imp_cpu):,} features): rho={rho_all:.4f}, p={pval_all:.2e}")

    # Top-N overlap
    top_cpu = set(np.argsort(imp_cpu)[-top_n:])
    top_gpu = set(np.argsort(imp_gpu)[-top_n:])
    overlap = len(top_cpu & top_gpu)
    log(f"  Top-{top_n} overlap: {overlap}/{top_n} ({100*overlap/top_n:.0f}%)")

    # Pearson on top-50 importance values
    top_cpu_idx = np.argsort(imp_cpu)[-top_n:]
    corr = np.corrcoef(imp_cpu[top_cpu_idx], imp_gpu[top_cpu_idx])[0, 1]
    log(f"  Top-{top_n} Pearson correlation (CPU ranking): {corr:.4f}")

    # Show top-10 from each
    log(f"\n  Top-10 features (CPU):")
    for i, idx in enumerate(np.argsort(imp_cpu)[-10:][::-1]):
        name = feature_cols[idx] if idx < len(feature_cols) else f"feat_{idx}"
        log(f"    {i+1:>2}. {name[:60]:<60s} gain={imp_cpu[idx]:.1f}")

    log(f"\n  Top-10 features (GPU):")
    for i, idx in enumerate(np.argsort(imp_gpu)[-10:][::-1]):
        name = feature_cols[idx] if idx < len(feature_cols) else f"feat_{idx}"
        log(f"    {i+1:>2}. {name[:60]:<60s} gain={imp_gpu[idx]:.1f}")

    return rho_all, overlap


# ---------------------------------------------------------------------------
# 5. Prediction agreement
# ---------------------------------------------------------------------------
def compare_predictions(preds_cpu, preds_gpu, probs_cpu, probs_gpu, y_test):
    """Compare predictions between CPU and GPU models."""
    log(f"\n{'='*60}")
    log(f"Prediction Agreement")
    log(f"{'='*60}")

    agreement = np.mean(preds_cpu == preds_gpu) * 100
    log(f"  Class agreement: {agreement:.1f}% ({int((preds_cpu == preds_gpu).sum())}/{len(preds_cpu)})")

    # Per-class agreement
    for cls, name in [(0, "SHORT"), (1, "FLAT"), (2, "LONG")]:
        mask = (preds_cpu == cls) | (preds_gpu == cls)
        if mask.sum() > 0:
            agree = np.mean(preds_cpu[mask] == preds_gpu[mask]) * 100
            log(f"  {name} agreement: {agree:.1f}% (of {int(mask.sum())} where either predicts {name})")

    # Probability difference
    if probs_cpu is not None and probs_gpu is not None:
        prob_diff = np.abs(probs_cpu - probs_gpu)
        log(f"  Mean |prob_cpu - prob_gpu|: {prob_diff.mean():.4f}")
        log(f"  Max  |prob_cpu - prob_gpu|: {prob_diff.max():.4f}")

    # Both correct / one correct / both wrong
    both_correct = ((preds_cpu == y_test) & (preds_gpu == y_test)).sum()
    cpu_only = ((preds_cpu == y_test) & (preds_gpu != y_test)).sum()
    gpu_only = ((preds_cpu != y_test) & (preds_gpu == y_test)).sum()
    both_wrong = ((preds_cpu != y_test) & (preds_gpu != y_test)).sum()
    log(f"  Both correct: {both_correct}  |  CPU-only correct: {cpu_only}  "
        f"|  GPU-only correct: {gpu_only}  |  Both wrong: {both_wrong}")

    return agreement


# ---------------------------------------------------------------------------
# 6. Standalone cuSPARSE SpMV benchmark on real data
# ---------------------------------------------------------------------------
def benchmark_cusparse_real(X_crosses_csr):
    """Benchmark cuSPARSE SpMV histogram building on real cross feature matrix."""
    if X_crosses_csr is None:
        log("\nSKIP: No cross matrix available for cuSPARSE benchmark")
        return

    try:
        import cupy as cp
        from cupyx.scipy import sparse as cp_sparse
    except ImportError:
        log("\nSKIP: CuPy not installed (pip install cupy-cuda12x)")
        return
    except Exception as e:
        log(f"\nSKIP: GPU not available: {e}")
        return

    n_rows, n_cols = X_crosses_csr.shape
    nnz = X_crosses_csr.nnz
    density = 100 * nnz / max(1, n_rows * n_cols)

    log(f"\n{'='*60}")
    log(f"cuSPARSE SpMV Benchmark (REAL 1w cross features)")
    log(f"{'='*60}")
    log(f"  Matrix: {n_rows:,} x {n_cols:,} ({nnz:,} nnz, {density:.4f}% dense)")
    log(f"  CSR size: {(X_crosses_csr.data.nbytes + X_crosses_csr.indices.nbytes + X_crosses_csr.indptr.nbytes) / 1e6:.1f} MB")

    try:
        props = cp.cuda.runtime.getDeviceProperties(0)
        gpu_name = props.get("name", "Unknown")
    except Exception:
        gpu_name = "Unknown"
    vram_gb = cp.cuda.Device(0).mem_info[1] / (1024**3)
    log(f"  GPU: {gpu_name}, VRAM: {vram_gb:.1f} GB")

    # Simulated gradient vector (mimics LightGBM gradients)
    rng = np.random.default_rng(42)
    grad_cpu = rng.standard_normal(n_rows).astype(np.float64)
    hess_cpu = np.abs(rng.standard_normal(n_rows).astype(np.float64))

    # --- CPU baseline ---
    log("\n  CPU (scipy) histogram building...")
    t0 = time.perf_counter()
    hist_grad_cpu = X_crosses_csr.T @ grad_cpu
    cpu_grad_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    hist_hess_cpu = X_crosses_csr.T @ hess_cpu
    cpu_hess_time = time.perf_counter() - t0

    cpu_total = cpu_grad_time + cpu_hess_time
    log(f"  CPU grad: {cpu_grad_time*1000:.1f} ms | hess: {cpu_hess_time*1000:.1f} ms | total: {cpu_total*1000:.1f} ms")

    # --- GPU: upload + pre-transpose ---
    log("\n  GPU upload + pre-transpose...")
    t0 = time.perf_counter()
    csr_gpu = cp_sparse.csr_matrix(X_crosses_csr)
    upload_time = time.perf_counter() - t0
    log(f"  Upload: {upload_time*1000:.1f} ms")

    t0 = time.perf_counter()
    csr_AT_gpu = csr_gpu.T.tocsr()
    transpose_time = time.perf_counter() - t0
    log(f"  Pre-transpose: {transpose_time*1000:.1f} ms (one-time cost)")

    grad_gpu = cp.asarray(grad_cpu)
    hess_gpu = cp.asarray(hess_cpu)

    # --- GPU: SpMV histogram (pre-transposed) ---
    n_warmup = 10
    n_trials = 100

    # Warmup
    for _ in range(n_warmup):
        _ = csr_AT_gpu @ grad_gpu
        cp.cuda.Stream.null.synchronize()

    # Benchmark: 2x SpMV (grad + hess separately — faster than fused SpMM)
    times_gpu = []
    for _ in range(n_trials):
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        h_grad = csr_AT_gpu @ grad_gpu
        h_hess = csr_AT_gpu @ hess_gpu
        cp.cuda.Stream.null.synchronize()
        times_gpu.append(time.perf_counter() - t0)

    med_gpu = np.median(times_gpu) * 1000
    min_gpu = np.min(times_gpu) * 1000
    p95_gpu = np.percentile(times_gpu, 95) * 1000

    log(f"\n  GPU SpMV (2x separate, {n_trials} trials):")
    log(f"    Median: {med_gpu:.3f} ms | Min: {min_gpu:.3f} ms | P95: {p95_gpu:.3f} ms")
    log(f"    Speedup vs CPU: {cpu_total*1000/med_gpu:.0f}x")

    # Verify correctness
    max_err_grad = np.max(np.abs(hist_grad_cpu - h_grad.get()))
    max_err_hess = np.max(np.abs(hist_hess_cpu - h_hess.get()))
    rel_err = max(max_err_grad, max_err_hess) / max(np.max(np.abs(hist_grad_cpu)), 1e-15)
    status = "PASS" if rel_err < 1e-10 else "FAIL"
    log(f"    Correctness: {status} (max relative error: {rel_err:.2e})")

    # Effective bandwidth
    bytes_read = nnz * (8 + 4) + (n_cols + 1) * 8 + n_rows * 8
    bytes_written = n_cols * 8
    total_bytes = (bytes_read + bytes_written) * 2  # 2x SpMV
    eff_bw = total_bytes / (med_gpu / 1000) / 1e9
    log(f"    Effective bandwidth: {eff_bw:.1f} GB/s")

    # --- GPU: Fused SpMM (grad + hess in one call) ---
    gh_gpu = cp.column_stack([grad_gpu, hess_gpu])

    for _ in range(n_warmup):
        _ = csr_AT_gpu @ gh_gpu
        cp.cuda.Stream.null.synchronize()

    times_fused = []
    for _ in range(n_trials):
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        h_both = csr_AT_gpu @ gh_gpu
        cp.cuda.Stream.null.synchronize()
        times_fused.append(time.perf_counter() - t0)

    med_fused = np.median(times_fused) * 1000
    log(f"\n  GPU SpMM fused (1 call, 2 RHS): {med_fused:.3f} ms")
    log(f"    2x SpMV vs 1x SpMM: {'SpMV faster' if med_gpu < med_fused else 'SpMM faster'} "
        f"({min(med_gpu, med_fused):.3f} vs {max(med_gpu, med_fused):.3f} ms)")

    # --- Per-boosting-round estimate ---
    # LightGBM builds histograms per leaf node. With num_leaves=63, worst case ~62 histogram builds.
    # But histogram subtraction halves it (~31 actual SpMV calls).
    n_leaves = 63
    n_hist_calls = n_leaves // 2 + 1  # ~32 with subtraction trick
    cpu_per_round = cpu_total * n_hist_calls
    gpu_per_round = med_gpu * n_hist_calls
    log(f"\n  Per-round estimate (num_leaves={n_leaves}, ~{n_hist_calls} histogram builds):")
    log(f"    CPU: {cpu_per_round:.0f} ms/round")
    log(f"    GPU: {gpu_per_round:.1f} ms/round")
    log(f"    Speedup: {cpu_per_round/gpu_per_round:.0f}x")

    # Cleanup
    del csr_gpu, csr_AT_gpu, grad_gpu, hess_gpu, gh_gpu
    cp.get_default_memory_pool().free_all_blocks()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    log("=" * 60)
    log("GPU Sparse Histogram Fork — Real 1w Data Test")
    log("=" * 60)
    log(f"v3.3 dir: {_V33_DIR}")
    log(f"DB_DIR:   {DB_DIR}")

    # --- Load data ---
    result = load_1w_data()
    if result is None:
        log("ABORT: Cannot load 1w data")
        sys.exit(1)

    X_all, y, feature_cols, X_crosses_csr, df = result

    # --- Split ---
    X_train, X_test, y_train, y_test, train_idx, test_idx = split_data(X_all, y, holdout_frac=0.2)
    log(f"\nTrain: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
    log(f"Train labels: SHORT={int((y_train==0).sum())}, FLAT={int((y_train==1).sum())}, LONG={int((y_train==2).sum())}")
    log(f"Test labels:  SHORT={int((y_test==0).sum())}, FLAT={int((y_test==1).sum())}, LONG={int((y_test==2).sum())}")

    # --- CPU training (baseline) ---
    params_cpu = {
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
        "feature_fraction": 0.05,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "deterministic": True,
        "num_threads": 0,
        "verbosity": 1,
        "seed": 42,
    }

    model_cpu, time_cpu, acc_cpu, preds_cpu, probs_cpu = train_lgbm(
        X_train, y_train, X_test, y_test, params_cpu, label="CPU baseline"
    )

    # --- GPU training (cuda_sparse fork) ---
    # Check if our fork's cuda_sparse device is available
    gpu_available = False
    try:
        import lightgbm as lgb
        # Test if cuda_sparse is a recognized device_type
        # (only available with our custom LightGBM fork)
        log("\nChecking for cuda_sparse device support...")

        # Method 1: Check if lightgbm was built with CUDA
        lgb_config = lgb.basic._LIB.LGBM_DumpParamAliases
        gpu_available = True  # We'll try and see if it errors
        log("  LightGBM loaded, attempting cuda_sparse training...")
    except Exception as e:
        log(f"  LightGBM import issue: {e}")

    model_gpu = None
    time_gpu = None
    acc_gpu = None
    preds_gpu = None
    probs_gpu = None

    if gpu_available:
        params_gpu = params_cpu.copy()
        params_gpu["device_type"] = "cuda_sparse"
        params_gpu["seed"] = 42
        # force_col_wise is not compatible with cuda_sparse
        if "force_col_wise" in params_gpu:
            del params_gpu["force_col_wise"]

        try:
            model_gpu, time_gpu, acc_gpu, preds_gpu, probs_gpu = train_lgbm(
                X_train, y_train, X_test, y_test, params_gpu, label="cuda_sparse (GPU fork)"
            )
        except Exception as e:
            log(f"\n  cuda_sparse training FAILED: {e}")
            log("  This is expected if running stock LightGBM (no custom fork installed)")
            log("  Falling back to cuda device_type...")

            # Try standard cuda device
            params_gpu2 = params_cpu.copy()
            params_gpu2["device_type"] = "cuda"
            params_gpu2["seed"] = 42
            if "force_col_wise" in params_gpu2:
                del params_gpu2["force_col_wise"]
            try:
                model_gpu, time_gpu, acc_gpu, preds_gpu, probs_gpu = train_lgbm(
                    X_train, y_train, X_test, y_test, params_gpu2, label="cuda (stock LightGBM)"
                )
            except Exception as e2:
                log(f"  cuda training also FAILED: {e2}")
                log("  GPU LightGBM not available, skipping GPU vs CPU comparison")

    # --- Comparison ---
    if model_gpu is not None:
        compare_feature_importance(model_cpu, model_gpu, feature_cols, top_n=50)
        compare_predictions(preds_cpu, preds_gpu, probs_cpu, probs_gpu, y_test)

        log(f"\n{'='*60}")
        log(f"TRAINING TIME COMPARISON")
        log(f"{'='*60}")
        log(f"  CPU:  {time_cpu:.1f}s")
        log(f"  GPU:  {time_gpu:.1f}s")
        speedup = time_cpu / time_gpu if time_gpu > 0 else float("inf")
        log(f"  Speedup: {speedup:.1f}x")
    else:
        log(f"\n  CPU-only results: {acc_cpu:.1f}% accuracy in {time_cpu:.1f}s")

    # --- Standalone cuSPARSE benchmark (independent of LightGBM fork) ---
    benchmark_cusparse_real(X_crosses_csr)

    # --- Summary ---
    log(f"\n{'='*60}")
    log(f"SUMMARY")
    log(f"{'='*60}")
    log(f"  Data: {X_all.shape[0]} rows x {X_all.shape[1]:,} features (1w BTC)")
    log(f"  CPU accuracy:  {acc_cpu:.1f}%")
    log(f"  CPU time:      {time_cpu:.1f}s")
    if acc_gpu is not None:
        log(f"  GPU accuracy:  {acc_gpu:.1f}%")
        log(f"  GPU time:      {time_gpu:.1f}s")
        log(f"  Speedup:       {time_cpu/time_gpu:.1f}x")
        agreement = np.mean(preds_cpu == preds_gpu) * 100
        log(f"  Pred agreement: {agreement:.1f}%")
    log(f"  Baseline (v3.3 production): 71.9% (CPCV, full Optuna tuning)")
    log(f"  NOTE: This test uses a simple holdout split, not CPCV.")
    log(f"         Lower accuracy than baseline is expected.")


if __name__ == "__main__":
    main()
