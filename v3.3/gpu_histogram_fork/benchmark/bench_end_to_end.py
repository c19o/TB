#!/usr/bin/env python3
"""
End-to-end LightGBM benchmark: CPU vs GPU histogram on synthetic 4h data.

Generates sparse binary CSR data matching our 4h training profile, trains
a 3-class LightGBM classifier (SHORT/FLAT/LONG) via both stock CPU and
our custom GPU histogram builder, then compares wall time, accuracy,
feature importance correlation, and prediction agreement.

Matrix Thesis Context:
  - Binary cross features, 3-class classification (SHORT/FLAT/LONG)
  - EFB bundling, feature_pre_filter=False, ALL features preserved
  - Sparse CSR with int32 indices, int64 indptr
  - max_bin=255 for maximum EFB compression
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy import sparse as sp_sparse
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, log_loss

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------------------------------------------------------
# Check for GPU histogram availability
# -------------------------------------------------------------------------
_GPU_AVAILABLE = False
_GPU_REASON = ""

try:
    _src_dir = str(Path(__file__).resolve().parent.parent / "src")
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)
    from gpu_histogram_atomic import is_available as gpu_is_available
    if gpu_is_available():
        _GPU_AVAILABLE = True
    else:
        _GPU_REASON = "CuPy/CUDA not detected"
except ImportError as e:
    _GPU_REASON = f"Import error: {e}"
except Exception as e:
    _GPU_REASON = f"Init error: {e}"

try:
    import lightgbm as lgb
except ImportError:
    print("FATAL: lightgbm not installed. pip install lightgbm")
    sys.exit(1)


# -------------------------------------------------------------------------
# Data generation
# -------------------------------------------------------------------------

def generate_sparse_binary_csr(rows: int, cols: int, density: float,
                               seed: int = 42) -> sp_sparse.csr_matrix:
    """
    Generate sparse binary CSR matrix matching real cross-feature profile.

    Binary 0/1 features with int32 indices, int64 indptr.
    Structural zeros = 0.0 (feature OFF, correct for binary crosses).
    """
    rng = np.random.RandomState(seed)
    nnz_per_row = max(1, int(cols * density))

    indptr = np.zeros(rows + 1, dtype=np.int64)
    indices_parts = []

    CHUNK = 200_000
    for start in range(0, rows, CHUNK):
        end = min(start + CHUNK, rows)
        chunk_rows = end - start
        chunk_idx = np.empty(chunk_rows * nnz_per_row, dtype=np.int32)
        for i in range(chunk_rows):
            chunk_idx[i * nnz_per_row:(i + 1) * nnz_per_row] = \
                rng.choice(cols, size=nnz_per_row, replace=False).astype(np.int32)
        indices_parts.append(chunk_idx)
        for i in range(chunk_rows):
            indptr[start + i + 1] = indptr[start + i] + nnz_per_row

    indices = np.concatenate(indices_parts)
    data = np.ones(len(indices), dtype=np.float32)

    mat = sp_sparse.csr_matrix((data, indices, indptr), shape=(rows, cols))
    mat.indices = mat.indices.astype(np.int32)
    mat.indptr = mat.indptr.astype(np.int64)
    return mat


def generate_labels(rows: int, seed: int = 42) -> np.ndarray:
    """
    Generate 3-class labels matching real distribution.
    ~30% LONG (2), ~40% FLAT (1), ~30% SHORT (0).
    """
    rng = np.random.RandomState(seed)
    probs = [0.30, 0.40, 0.30]  # SHORT, FLAT, LONG
    labels = rng.choice(3, size=rows, p=probs)
    return labels.astype(np.int32)


# -------------------------------------------------------------------------
# LightGBM params matching our config
# -------------------------------------------------------------------------

def get_lgbm_params(device: str = "cpu", num_rounds: int = 100) -> dict:
    """
    Return LightGBM params matching v3.3 config.

    device: "cpu" or "gpu"
    """
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "max_bin": 255,
        "num_leaves": 63,
        "learning_rate": 0.03,
        "feature_fraction": 0.8,
        "min_data_in_leaf": 5,
        "feature_pre_filter": False,
        "force_col_wise": True,
        "num_threads": 0,
        "verbose": -1,
        "seed": 42,
    }
    if device == "gpu":
        params["device_type"] = "gpu"
        params["gpu_use_dp"] = True
    else:
        params["device_type"] = "cpu"
    return params


# -------------------------------------------------------------------------
# Training harness
# -------------------------------------------------------------------------

def train_once(X_train, y_train, X_test, y_test, params: dict,
               num_rounds: int) -> dict:
    """
    Train LightGBM once and return metrics.

    Returns dict with: wall_time, accuracy, logloss, feature_importance (top 50),
                       predictions.
    """
    ds_train = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    ds_train.construct()

    t0 = time.perf_counter()
    model = lgb.train(
        params,
        ds_train,
        num_boost_round=num_rounds,
        valid_sets=[],
        callbacks=[],
    )
    wall_time = time.perf_counter() - t0

    # Predict on holdout
    y_proba = model.predict(X_test)
    y_pred = np.argmax(y_proba, axis=1)

    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_proba, labels=[0, 1, 2])

    # Feature importance (gain)
    importance = model.feature_importance(importance_type="gain")
    top_50_idx = np.argsort(importance)[::-1][:50]
    top_50_imp = importance[top_50_idx]

    return {
        "wall_time": wall_time,
        "accuracy": acc,
        "logloss": ll,
        "top_50_idx": top_50_idx,
        "top_50_imp": top_50_imp,
        "predictions": y_pred,
        "probabilities": y_proba,
    }


# -------------------------------------------------------------------------
# Comparison metrics
# -------------------------------------------------------------------------

def compare_results(cpu_runs: list, gpu_runs: list) -> dict:
    """
    Compute comparison metrics between CPU and GPU training runs.

    Returns dict with all comparison metrics.
    """
    cpu_times = [r["wall_time"] for r in cpu_runs]
    gpu_times = [r["wall_time"] for r in gpu_runs]

    cpu_accs = [r["accuracy"] for r in cpu_runs]
    gpu_accs = [r["accuracy"] for r in gpu_runs]

    cpu_ll = [r["logloss"] for r in cpu_runs]
    gpu_ll = [r["logloss"] for r in gpu_runs]

    # Speedup
    cpu_mean = np.mean(cpu_times)
    gpu_mean = np.mean(gpu_times)
    speedup = cpu_mean / gpu_mean if gpu_mean > 0 else float("inf")

    # Feature importance rank correlation (Spearman on top-50)
    # Use the last run from each for importance comparison
    cpu_top50 = cpu_runs[-1]["top_50_idx"]
    gpu_top50 = gpu_runs[-1]["top_50_idx"]

    # Build rank arrays: for each feature in the union of top-50 from both,
    # assign its rank in CPU and GPU importance lists
    all_features = np.union1d(cpu_top50, gpu_top50)
    cpu_ranks = np.zeros(len(all_features))
    gpu_ranks = np.zeros(len(all_features))
    for i, feat in enumerate(all_features):
        cpu_pos = np.where(cpu_top50 == feat)[0]
        cpu_ranks[i] = cpu_pos[0] if len(cpu_pos) > 0 else 51  # not in top-50 = rank 51
        gpu_pos = np.where(gpu_top50 == feat)[0]
        gpu_ranks[i] = gpu_pos[0] if len(gpu_pos) > 0 else 51

    if len(all_features) > 1:
        rho, p_val = spearmanr(cpu_ranks, gpu_ranks)
    else:
        rho, p_val = 1.0, 0.0

    # Prediction agreement (last run)
    cpu_preds = cpu_runs[-1]["predictions"]
    gpu_preds = gpu_runs[-1]["predictions"]
    agreement = np.mean(cpu_preds == gpu_preds)

    return {
        "cpu_time_mean": cpu_mean,
        "cpu_time_std": np.std(cpu_times),
        "gpu_time_mean": gpu_mean,
        "gpu_time_std": np.std(gpu_times),
        "speedup": speedup,
        "cpu_accuracy_mean": np.mean(cpu_accs),
        "cpu_accuracy_std": np.std(cpu_accs),
        "gpu_accuracy_mean": np.mean(gpu_accs),
        "gpu_accuracy_std": np.std(gpu_accs),
        "accuracy_delta": np.mean(gpu_accs) - np.mean(cpu_accs),
        "cpu_logloss_mean": np.mean(cpu_ll),
        "gpu_logloss_mean": np.mean(gpu_ll),
        "feature_importance_spearman_rho": rho,
        "feature_importance_spearman_p": p_val,
        "prediction_agreement": agreement,
    }


# -------------------------------------------------------------------------
# ASCII summary
# -------------------------------------------------------------------------

def print_summary(comp: dict, data_info: dict):
    """Print a formatted ASCII summary table."""
    w = 72
    print()
    print("=" * w)
    print("  END-TO-END BENCHMARK RESULTS")
    print("=" * w)

    print(f"\n  Data Profile:")
    print(f"    Rows:       {data_info['rows']:>12,}")
    print(f"    Features:   {data_info['features']:>12,}")
    print(f"    Density:    {data_info['density']:>12.4f}")
    print(f"    NNZ:        {data_info['nnz']:>12,}")
    print(f"    NNZ bytes:  {data_info['nnz_mb']:>11.1f} MB")
    print(f"    Rounds:     {data_info['rounds']:>12}")
    print(f"    Runs:       {data_info['runs']:>12}")

    print(f"\n  {'-' * (w - 4)}")
    print(f"  {'Metric':<40} {'CPU':>12} {'GPU':>12}")
    print(f"  {'-' * (w - 4)}")

    print(f"  {'Wall time / fold (s)':<40} "
          f"{comp['cpu_time_mean']:>10.2f}s  "
          f"{comp['gpu_time_mean']:>10.2f}s")
    print(f"  {'  std (s)':<40} "
          f"{comp['cpu_time_std']:>10.3f}s  "
          f"{comp['gpu_time_std']:>10.3f}s")
    print(f"  {'Accuracy':<40} "
          f"{comp['cpu_accuracy_mean']:>11.4f}  "
          f"{comp['gpu_accuracy_mean']:>11.4f}")
    print(f"  {'  std':<40} "
          f"{comp['cpu_accuracy_std']:>11.4f}  "
          f"{comp['gpu_accuracy_std']:>11.4f}")
    print(f"  {'Log-loss':<40} "
          f"{comp['cpu_logloss_mean']:>11.4f}  "
          f"{comp['gpu_logloss_mean']:>11.4f}")

    print(f"\n  {'-' * (w - 4)}")
    print(f"  {'Comparison Metric':<40} {'Value':>12}")
    print(f"  {'-' * (w - 4)}")
    print(f"  {'Speedup (CPU/GPU)':<40} {comp['speedup']:>11.2f}x")
    print(f"  {'Accuracy delta (GPU - CPU)':<40} {comp['accuracy_delta']:>+11.4f}")
    print(f"  {'Top-50 importance Spearman rho':<40} {comp['feature_importance_spearman_rho']:>11.4f}")
    print(f"  {'  p-value':<40} {comp['feature_importance_spearman_p']:>11.2e}")
    print(f"  {'Prediction agreement':<40} {comp['prediction_agreement']:>10.1%}")

    print(f"\n  {'-' * (w - 4)}")

    # Verdict
    if comp["speedup"] > 1.0:
        verdict = f"GPU is {comp['speedup']:.1f}x FASTER"
    elif comp["speedup"] < 1.0:
        verdict = f"CPU is {1.0 / comp['speedup']:.1f}x FASTER"
    else:
        verdict = "CPU and GPU are equal"

    if abs(comp["accuracy_delta"]) < 0.005:
        acc_verdict = "Accuracy: EQUIVALENT (delta < 0.5%)"
    elif comp["accuracy_delta"] > 0:
        acc_verdict = f"Accuracy: GPU better by {comp['accuracy_delta']:.4f}"
    else:
        acc_verdict = f"Accuracy: CPU better by {-comp['accuracy_delta']:.4f}"

    print(f"  VERDICT: {verdict}")
    print(f"  {acc_verdict}")
    print(f"  Feature importance: {'CORRELATED' if comp['feature_importance_spearman_rho'] > 0.8 else 'DIVERGENT'} (rho={comp['feature_importance_spearman_rho']:.3f})")
    print("=" * w)


# -------------------------------------------------------------------------
# Main benchmark
# -------------------------------------------------------------------------

def run_benchmark(rows: int, features: int, density: float,
                  num_rounds: int, num_runs: int,
                  output: str | None = None):
    """
    Full end-to-end benchmark: generate data, train CPU, train GPU, compare.
    """
    w = 72
    print("=" * w)
    print("  LightGBM End-to-End Benchmark: CPU vs GPU Histogram")
    print("=" * w)
    print(f"  Rows:     {rows:,}")
    print(f"  Features: {features:,}")
    print(f"  Density:  {density}")
    print(f"  Rounds:   {num_rounds}")
    print(f"  Runs:     {num_runs} (+ 1 warmup)")
    print(f"  GPU:      {'AVAILABLE' if _GPU_AVAILABLE else f'NOT AVAILABLE ({_GPU_REASON})'}")
    print()

    # ------------------------------------------------------------------
    # 1. Generate data
    # ------------------------------------------------------------------
    print("[1/6] Generating synthetic data ...")
    t0 = time.perf_counter()
    X = generate_sparse_binary_csr(rows, features, density, seed=42)
    y = generate_labels(rows, seed=42)
    gen_time = time.perf_counter() - t0

    nnz = X.nnz
    nnz_mb = (X.data.nbytes + X.indices.nbytes + X.indptr.nbytes) / 1024**2
    print(f"  Shape: {X.shape}, NNZ: {nnz:,} ({100 * nnz / (rows * features):.3f}%)")
    print(f"  CSR memory: {nnz_mb:.1f} MB")
    print(f"  Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"  Generated in {gen_time:.1f}s")

    # 80/20 train/test split (deterministic)
    split_idx = int(rows * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"  Train: {X_train.shape[0]:,} rows, Test: {X_test.shape[0]:,} rows")

    data_info = {
        "rows": rows,
        "features": features,
        "density": density,
        "nnz": nnz,
        "nnz_mb": round(nnz_mb, 1),
        "rounds": num_rounds,
        "runs": num_runs,
        "train_rows": X_train.shape[0],
        "test_rows": X_test.shape[0],
        "label_dist": {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
    }

    # ------------------------------------------------------------------
    # 2. CPU training
    # ------------------------------------------------------------------
    print(f"\n[2/6] CPU warmup (1 run) ...")
    cpu_params = get_lgbm_params("cpu", num_rounds)
    warmup_result = train_once(X_train, y_train, X_test, y_test,
                               cpu_params, num_rounds)
    print(f"  Warmup: {warmup_result['wall_time']:.2f}s, "
          f"acc={warmup_result['accuracy']:.4f}")

    print(f"\n[3/6] CPU timed runs ({num_runs}) ...")
    cpu_runs = []
    for i in range(num_runs):
        # Vary seed slightly to get variance, but keep data identical
        params_i = cpu_params.copy()
        params_i["seed"] = 42 + i
        result = train_once(X_train, y_train, X_test, y_test,
                           params_i, num_rounds)
        cpu_runs.append(result)
        print(f"  Run {i + 1}/{num_runs}: {result['wall_time']:.2f}s, "
              f"acc={result['accuracy']:.4f}, logloss={result['logloss']:.4f}")

    # ------------------------------------------------------------------
    # 3. GPU training (if available)
    # ------------------------------------------------------------------
    if _GPU_AVAILABLE:
        print(f"\n[4/6] GPU warmup (1 run) ...")
        gpu_params = get_lgbm_params("gpu", num_rounds)
        try:
            warmup_gpu = train_once(X_train, y_train, X_test, y_test,
                                    gpu_params, num_rounds)
            print(f"  Warmup: {warmup_gpu['wall_time']:.2f}s, "
                  f"acc={warmup_gpu['accuracy']:.4f}")
            gpu_ok = True
        except Exception as e:
            print(f"  GPU warmup FAILED: {e}")
            print(f"  Falling back to CPU-only benchmark.")
            gpu_ok = False

        if gpu_ok:
            print(f"\n[5/6] GPU timed runs ({num_runs}) ...")
            gpu_runs = []
            for i in range(num_runs):
                params_i = gpu_params.copy()
                params_i["seed"] = 42 + i
                result = train_once(X_train, y_train, X_test, y_test,
                                   params_i, num_rounds)
                gpu_runs.append(result)
                print(f"  Run {i + 1}/{num_runs}: {result['wall_time']:.2f}s, "
                      f"acc={result['accuracy']:.4f}, logloss={result['logloss']:.4f}")
        else:
            gpu_runs = None
    else:
        print(f"\n[4/6] GPU training SKIPPED ({_GPU_REASON})")
        print(f"[5/6] GPU training SKIPPED")
        gpu_runs = None

    # ------------------------------------------------------------------
    # 4. Compare
    # ------------------------------------------------------------------
    print(f"\n[6/6] Computing comparison metrics ...")

    if gpu_runs is not None:
        comp = compare_results(cpu_runs, gpu_runs)
    else:
        # CPU-only: report CPU stats, mark GPU as N/A
        cpu_times = [r["wall_time"] for r in cpu_runs]
        cpu_accs = [r["accuracy"] for r in cpu_runs]
        cpu_ll = [r["logloss"] for r in cpu_runs]
        comp = {
            "cpu_time_mean": np.mean(cpu_times),
            "cpu_time_std": np.std(cpu_times),
            "gpu_time_mean": None,
            "gpu_time_std": None,
            "speedup": None,
            "cpu_accuracy_mean": np.mean(cpu_accs),
            "cpu_accuracy_std": np.std(cpu_accs),
            "gpu_accuracy_mean": None,
            "gpu_accuracy_std": None,
            "accuracy_delta": None,
            "cpu_logloss_mean": np.mean(cpu_ll),
            "gpu_logloss_mean": None,
            "feature_importance_spearman_rho": None,
            "feature_importance_spearman_p": None,
            "prediction_agreement": None,
        }

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    if gpu_runs is not None:
        print_summary(comp, data_info)
    else:
        print()
        print("=" * 72)
        print("  CPU-ONLY BENCHMARK RESULTS (GPU not available)")
        print("=" * 72)
        print(f"\n  Data: {rows:,} rows x {features:,} features, "
              f"density={density}, {num_rounds} rounds")
        print(f"\n  CPU wall time:  {comp['cpu_time_mean']:.2f}s "
              f"(std {comp['cpu_time_std']:.3f}s)")
        print(f"  CPU accuracy:   {comp['cpu_accuracy_mean']:.4f} "
              f"(std {comp['cpu_accuracy_std']:.4f})")
        print(f"  CPU log-loss:   {comp['cpu_logloss_mean']:.4f}")
        print(f"\n  GPU benchmark requires CuPy + CUDA. "
              f"Install with: pip install cupy-cuda12x")
        print("=" * 72)

    # ------------------------------------------------------------------
    # 6. JSON output
    # ------------------------------------------------------------------
    output_data = {
        "data": data_info,
        "cpu": {
            "time_mean": comp["cpu_time_mean"],
            "time_std": comp["cpu_time_std"],
            "accuracy_mean": comp["cpu_accuracy_mean"],
            "accuracy_std": comp.get("cpu_accuracy_std"),
            "logloss_mean": comp["cpu_logloss_mean"],
            "per_run_times": [r["wall_time"] for r in cpu_runs],
            "per_run_accs": [r["accuracy"] for r in cpu_runs],
        },
        "gpu": None,
        "comparison": None,
    }

    if gpu_runs is not None:
        output_data["gpu"] = {
            "time_mean": comp["gpu_time_mean"],
            "time_std": comp["gpu_time_std"],
            "accuracy_mean": comp["gpu_accuracy_mean"],
            "accuracy_std": comp["gpu_accuracy_std"],
            "logloss_mean": comp["gpu_logloss_mean"],
            "per_run_times": [r["wall_time"] for r in gpu_runs],
            "per_run_accs": [r["accuracy"] for r in gpu_runs],
        }
        output_data["comparison"] = {
            "speedup": comp["speedup"],
            "accuracy_delta": comp["accuracy_delta"],
            "feature_importance_spearman_rho": comp["feature_importance_spearman_rho"],
            "feature_importance_spearman_p": comp["feature_importance_spearman_p"],
            "prediction_agreement": comp["prediction_agreement"],
        }

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output_data, f, indent=2, default=_json_default)
        print(f"\nResults saved to: {out_path.resolve()}")

    return output_data


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end LightGBM benchmark: CPU vs GPU histogram. "
                    "Trains on synthetic sparse binary data matching 4h profile.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (small)
  python bench_end_to_end.py --rows 1000 --features 10000 --rounds 50

  # 4h profile (default)
  python bench_end_to_end.py

  # Full scale with JSON output
  python bench_end_to_end.py --rows 17520 --features 100000 --rounds 200 --output results.json

  # Minimal for CI
  python bench_end_to_end.py --rows 500 --features 5000 --rounds 20 --runs 2
""",
    )
    parser.add_argument(
        "--rows", type=int, default=17_520,
        help="Number of rows / samples (default: 17520 = 4h BTC profile)"
    )
    parser.add_argument(
        "--features", type=int, default=100_000,
        help="Number of features (default: 100000, reduced from 3M for speed)"
    )
    parser.add_argument(
        "--density", type=float, default=0.003,
        help="Sparse matrix density (default: 0.003 = 0.3%%, matches 4h profile)"
    )
    parser.add_argument(
        "--rounds", type=int, default=100,
        help="LightGBM boosting rounds per run (default: 100)"
    )
    parser.add_argument(
        "--runs", type=int, default=5,
        help="Number of timed runs for mean/std (default: 5)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save JSON results (default: print only)"
    )

    args = parser.parse_args()

    run_benchmark(
        rows=args.rows,
        features=args.features,
        density=args.density,
        num_rounds=args.rounds,
        num_runs=args.runs,
        output=args.output,
    )


if __name__ == "__main__":
    main()
