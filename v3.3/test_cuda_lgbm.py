#!/usr/bin/env python3
"""
test_cuda_lgbm.py — Test LightGBM CUDA/GPU backends with sparse binary features post-EFB.

BUILDING LIGHTGBM WITH CUDA SUPPORT:
=====================================
# 1. Install CUDA toolkit (11.x or 12.x)
# 2. Build from source:
#    pip uninstall lightgbm -y
#    git clone --recursive https://github.com/microsoft/LightGBM.git
#    cd LightGBM
#    mkdir build && cd build
#    cmake .. -DUSE_CUDA=1
#    make -j$(nproc)
#    cd ../python-package && pip install .
#
# For OpenCL (device_type="gpu"):
#    cmake .. -DUSE_GPU=1
#    make -j$(nproc)
#
# Verify:
#    python -c "import lightgbm; print(lightgbm.__version__)"

Matrix thesis context:
  - 2-10M sparse binary cross features at 1-2% density
  - EFB (Exclusive Feature Bundling) compresses to ~12-18K bundles
  - Post-EFB data is compact enough for GPU memory
  - This script tests whether CUDA backend works after EFB compression
"""

import time
import sys
import warnings
import traceback
import numpy as np
from scipy import sparse

try:
    import lightgbm as lgb
    print(f"LightGBM version: {lgb.__version__}")
except ImportError:
    print("FATAL: lightgbm not installed. pip install lightgbm")
    sys.exit(1)

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_ROWS = 17_520        # ~4h candles over 8 years
N_FEATURES = 100_000   # smaller than real 2M+ for quick test
DENSITY = 0.015        # 1.5% density (binary 0/1)
N_CLASSES = 3          # SHORT=0, FLAT=1, LONG=2
N_ROUNDS = 100
N_FOLDS = 1            # single train/val split for speed
VAL_RATIO = 0.2
SEED = 42

BASE_PARAMS = {
    "objective": "multiclass",
    "num_class": N_CLASSES,
    "metric": "multi_logloss",
    "max_bin": 255,
    "num_leaves": 63,
    "learning_rate": 0.03,
    "feature_pre_filter": False,
    "verbose": 1,
    "seed": SEED,
    "num_threads": -1,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_vram_mb():
    """Return (used_mb, total_mb) or (None, None) if pynvml unavailable."""
    if not HAS_PYNVML:
        return None, None
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used = info.used / (1024 ** 2)
        total = info.total / (1024 ** 2)
        pynvml.nvmlShutdown()
        return used, total
    except Exception:
        return None, None


def generate_sparse_binary_data():
    """Generate synthetic sparse binary CSR matrix + 3-class labels."""
    print(f"\n{'='*70}")
    print(f"Generating synthetic data: {N_ROWS:,} rows x {N_FEATURES:,} features")
    print(f"Density: {DENSITY*100:.1f}%, binary 0/1")
    print(f"{'='*70}")

    rng = np.random.RandomState(SEED)
    t0 = time.time()

    X = sparse.random(N_ROWS, N_FEATURES, density=DENSITY, format="csr",
                      dtype=np.float32, random_state=rng)
    # Make binary: any nonzero -> 1.0
    X.data[:] = 1.0

    # 3-class labels with slight imbalance (like real data)
    y = rng.choice([0, 1, 2], size=N_ROWS, p=[0.25, 0.50, 0.25]).astype(np.int32)

    elapsed = time.time() - t0
    nnz = X.nnz
    print(f"Generated in {elapsed:.1f}s")
    print(f"NNZ: {nnz:,} ({nnz/(N_ROWS*N_FEATURES)*100:.2f}%)")
    print(f"CSR memory: {(X.data.nbytes + X.indices.nbytes + X.indptr.nbytes) / 1e6:.1f} MB")
    print(f"Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Train/val split
    split = int(N_ROWS * (1 - VAL_RATIO))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    return X_train, y_train, X_val, y_val


def run_training(device_type, X_train, y_train, X_val, y_val):
    """Train LightGBM with given device_type. Returns result dict."""
    result = {
        "device": device_type,
        "status": "UNKNOWN",
        "accuracy": None,
        "logloss": None,
        "train_time": None,
        "efb_bundles": None,
        "vram_before": None,
        "vram_after": None,
        "error": None,
        "warnings": [],
    }

    print(f"\n{'='*70}")
    print(f"TESTING: device_type='{device_type}'")
    print(f"{'='*70}")

    params = BASE_PARAMS.copy()
    params["device_type"] = device_type

    # force_col_wise is CPU-only; CUDA/GPU may reject it
    if device_type == "cpu":
        params["force_col_wise"] = True
    # For CUDA, LightGBM docs say force_col_wise is ignored; remove to avoid warnings
    # For GPU (OpenCL), same thing

    # VRAM before
    vram_before, vram_total = get_vram_mb()
    result["vram_before"] = vram_before
    if vram_before is not None:
        print(f"VRAM before: {vram_before:.0f} / {vram_total:.0f} MB")

    try:
        # Build datasets — pass sparse CSR directly, let LightGBM/EFB handle it
        print(f"Building LightGBM Dataset from sparse CSR ({X_train.shape[1]:,} features)...")
        t0 = time.time()

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")

            dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=False)

            # Construct to trigger EFB
            dtrain.construct()
            dval.construct()

        dataset_time = time.time() - t0
        print(f"Dataset construction: {dataset_time:.1f}s")

        # Check for EFB bundle count
        try:
            num_features_after = dtrain.num_feature()
            print(f"Features before EFB: {X_train.shape[1]:,}")
            print(f"Features after EFB:  {num_features_after:,}")
            result["efb_bundles"] = num_features_after
        except Exception:
            pass

        # Log any warnings from dataset construction
        for w in caught_warnings:
            msg = str(w.message)
            result["warnings"].append(f"[Dataset] {msg}")
            print(f"  WARNING: {msg}")

        # Train
        print(f"Training {N_ROUNDS} rounds with device_type='{device_type}'...")
        t0 = time.time()

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")

            callbacks = [lgb.log_evaluation(period=25)]
            model = lgb.train(
                params,
                dtrain,
                num_boost_round=N_ROUNDS,
                valid_sets=[dval],
                valid_names=["val"],
                callbacks=callbacks,
            )

        train_time = time.time() - t0
        result["train_time"] = train_time

        for w in caught_warnings:
            msg = str(w.message)
            result["warnings"].append(f"[Train] {msg}")
            print(f"  WARNING: {msg}")

        # Evaluate
        y_pred_proba = model.predict(X_val)
        y_pred = np.argmax(y_pred_proba, axis=1)
        accuracy = np.mean(y_pred == y_val)
        result["accuracy"] = accuracy

        # Final logloss from best iteration
        best_score = model.best_score.get("val", {}).get("multi_logloss", None)
        result["logloss"] = best_score

        # VRAM after
        vram_after, _ = get_vram_mb()
        result["vram_after"] = vram_after
        if vram_after is not None:
            print(f"VRAM after:  {vram_after:.0f} MB (delta: {vram_after - vram_before:+.0f} MB)")

        result["status"] = "PASS"
        print(f"\nRESULT: PASS")
        print(f"  Accuracy:   {accuracy:.4f}")
        print(f"  Logloss:    {best_score}")
        print(f"  Train time: {train_time:.1f}s")

    except Exception as e:
        result["status"] = "FAIL"
        result["error"] = str(e)
        print(f"\nRESULT: FAIL")
        print(f"  Error: {e}")
        traceback.print_exc()

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("LightGBM CUDA/GPU Backend Test — Sparse Binary Features + EFB")
    print("=" * 70)

    # System info
    vram_used, vram_total = get_vram_mb()
    if vram_total:
        print(f"GPU VRAM: {vram_total:.0f} MB total, {vram_used:.0f} MB used")
    else:
        print("GPU VRAM: pynvml not available (pip install pynvml)")

    # Generate data once
    X_train, y_train, X_val, y_val = generate_sparse_binary_data()

    # Run all three backends
    results = {}
    for device in ["cpu", "cuda", "gpu"]:
        results[device] = run_training(device, X_train, y_train, X_val, y_val)

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Backend':<10} {'Status':<8} {'Accuracy':<10} {'Logloss':<12} {'Time (s)':<10} {'EFB feat':<10} {'Speedup':<8}")
    print("-" * 70)

    cpu_time = results["cpu"]["train_time"]

    for device in ["cpu", "cuda", "gpu"]:
        r = results[device]
        acc = f"{r['accuracy']:.4f}" if r["accuracy"] is not None else "N/A"
        loss = f"{r['logloss']:.4f}" if r["logloss"] is not None else "N/A"
        t = f"{r['train_time']:.1f}" if r["train_time"] is not None else "N/A"
        efb = f"{r['efb_bundles']:,}" if r["efb_bundles"] is not None else "N/A"

        if r["train_time"] is not None and cpu_time is not None and cpu_time > 0:
            speedup = f"{cpu_time / r['train_time']:.2f}x"
        else:
            speedup = "N/A"

        status = r["status"]
        print(f"{device:<10} {status:<8} {acc:<10} {loss:<12} {t:<10} {efb:<10} {speedup:<8}")

    # VRAM delta
    print()
    for device in ["cuda", "gpu"]:
        r = results[device]
        if r["vram_before"] is not None and r["vram_after"] is not None:
            delta = r["vram_after"] - r["vram_before"]
            print(f"VRAM delta ({device}): {delta:+.0f} MB")

    # Warnings
    any_warnings = False
    for device in ["cpu", "cuda", "gpu"]:
        for w in results[device]["warnings"]:
            if not any_warnings:
                print("\nWarnings:")
                any_warnings = True
            print(f"  [{device}] {w}")

    # Errors
    for device in ["cuda", "gpu"]:
        if results[device]["error"]:
            print(f"\n{device.upper()} ERROR DETAIL:")
            print(f"  {results[device]['error']}")

    # Verdict
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")

    if results["cuda"]["status"] == "PASS":
        speedup = cpu_time / results["cuda"]["train_time"] if cpu_time and results["cuda"]["train_time"] else 0
        acc_delta = (results["cuda"]["accuracy"] - results["cpu"]["accuracy"]) if results["cpu"]["accuracy"] else 0
        print(f"CUDA backend WORKS with sparse binary features after EFB.")
        print(f"  Speedup vs CPU: {speedup:.2f}x")
        print(f"  Accuracy delta: {acc_delta:+.4f}")
        if speedup > 1.5:
            print(f"  --> Worth using for training. Significant speedup.")
        elif speedup > 1.0:
            print(f"  --> Marginal speedup. May help more with larger feature counts.")
        else:
            print(f"  --> No speedup. CPU is faster at this scale.")
    else:
        print(f"CUDA backend FAILED. Error: {results['cuda']['error']}")
        print(f"  --> Stick with CPU + sparse CSR for training.")

    if results["gpu"]["status"] == "PASS":
        speedup = cpu_time / results["gpu"]["train_time"] if cpu_time and results["gpu"]["train_time"] else 0
        print(f"\nOpenCL (gpu) backend also works. Speedup: {speedup:.2f}x")
    elif results["gpu"]["status"] == "FAIL":
        print(f"\nOpenCL (gpu) backend FAILED: {results['gpu']['error']}")

    print()
    return 0 if results["cuda"]["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
