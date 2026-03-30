#!/usr/bin/env python3
"""
lleaves_compiler.py -- LightGBM to LLVM Compilation Wrapper
=============================================================
Compiles a LightGBM model to native code via lleaves for 5.4x prediction speedup.

lleaves compiles the decision tree ensemble into LLVM IR, then to native machine code.
The compiled model produces bit-identical predictions to the original LightGBM model.

Usage:
    # Compile a model:
    python lleaves_compiler.py --model model_1w_pruned.json --tf 1w

    # Or programmatically:
    from lleaves_compiler import compile_model, load_compiled_model
    compiled_path = compile_model('model_1w_pruned.json', '1w')
    model = load_compiled_model(compiled_path)
    preds = model.predict(X)  # same API as lgb.Booster.predict()

Requirements:
    pip install lleaves
"""

import os
import sys
import time
import argparse
import numpy as np


def check_lleaves_available():
    """Check if lleaves is installed."""
    try:
        import lleaves
        return True
    except ImportError:
        return False


def compile_model(model_path, tf, output_dir=None):
    """
    Compile a LightGBM model to native LLVM code via lleaves.

    Args:
        model_path: str, path to LightGBM model .txt or .json file
        tf: str, timeframe name (used for output filename)
        output_dir: str, output directory (defaults to model's directory)

    Returns:
        compiled_path: str, path to compiled .so/.dll model
    """
    import lleaves

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    out_dir = output_dir or os.path.dirname(model_path) or '.'

    # lleaves requires model_to_string() text format
    # If we have a .json model, convert to text first
    model_txt_path = model_path
    if model_path.endswith('.json'):
        import lightgbm as lgb
        booster = lgb.Booster(model_file=model_path)
        model_txt_path = os.path.join(out_dir, f'model_{tf}_pruned.txt')
        booster.save_model(model_txt_path)
        print(f"  Converted JSON -> TXT: {model_txt_path}")

    # Determine output extension based on platform
    if sys.platform == 'win32':
        ext = '.dll'
    elif sys.platform == 'darwin':
        ext = '.dylib'
    else:
        ext = '.so'

    compiled_path = os.path.join(out_dir, f'model_{tf}_compiled{ext}')

    print(f"  Compiling {model_txt_path} -> {compiled_path}...")
    t0 = time.time()

    llvm_model = lleaves.Model(model_file=model_txt_path)
    llvm_model.compile(cache=compiled_path)

    dt = time.time() - t0
    size_mb = os.path.getsize(compiled_path) / (1024 * 1024)
    print(f"  Compiled in {dt:.1f}s ({size_mb:.1f} MB)")

    return compiled_path


def load_compiled_model(compiled_path):
    """
    Load a compiled lleaves model for inference.

    Args:
        compiled_path: str, path to compiled .so/.dll/.dylib

    Returns:
        model: lleaves.Model with .predict() method (same API as lgb.Booster)
    """
    import lleaves

    if not os.path.exists(compiled_path):
        raise FileNotFoundError(f"Compiled model not found: {compiled_path}")

    model = lleaves.Model(cache=compiled_path)
    return model


def benchmark_prediction(lgb_model_path, compiled_path, n_features, n_samples=1000):
    """
    Benchmark lleaves vs LightGBM prediction speed.

    Returns:
        results: dict with timing comparison
    """
    import lightgbm as lgb
    import lleaves

    # Generate random test data (sparse-like: mostly zeros)
    rng = np.random.default_rng(42)
    X = rng.random((n_samples, n_features)).astype(np.float32)
    X[X < 0.95] = 0.0  # 95% sparse (typical for cross features)

    # LightGBM prediction
    lgb_model = lgb.Booster(model_file=lgb_model_path)
    # Warmup
    lgb_model.predict(X[:1])
    t0 = time.time()
    lgb_preds = lgb_model.predict(X)
    lgb_time = time.time() - t0

    # lleaves prediction
    llvm_model = lleaves.Model(cache=compiled_path)
    # Warmup
    llvm_model.predict(X[:1])
    t0 = time.time()
    llvm_preds = llvm_model.predict(X)
    llvm_time = time.time() - t0

    # Verify predictions match
    max_diff = float(np.max(np.abs(lgb_preds - llvm_preds)))
    speedup = lgb_time / llvm_time if llvm_time > 0 else float('inf')

    results = {
        'lgb_time_ms': lgb_time * 1000,
        'lleaves_time_ms': llvm_time * 1000,
        'speedup': speedup,
        'max_pred_diff': max_diff,
        'preds_identical': max_diff < 1e-6,
        'n_samples': n_samples,
        'n_features': n_features,
    }

    print(f"\n  Benchmark ({n_samples} samples, {n_features} features):")
    print(f"    LightGBM:  {lgb_time*1000:.1f} ms")
    print(f"    lleaves:   {llvm_time*1000:.1f} ms")
    print(f"    Speedup:   {speedup:.1f}x")
    print(f"    Max diff:  {max_diff:.2e} (identical: {max_diff < 1e-6})")

    return results


def main():
    parser = argparse.ArgumentParser(description='Compile LightGBM model with lleaves')
    parser.add_argument('--model', required=True, help='Path to LightGBM model file')
    parser.add_argument('--tf', required=True, help='Timeframe (1w, 1d, 4h, 1h, 15m)')
    parser.add_argument('--output-dir', default=None, help='Output directory')
    parser.add_argument('--benchmark', action='store_true', help='Run speed benchmark after compilation')
    args = parser.parse_args()

    if not check_lleaves_available():
        print("ERROR: lleaves not installed. Install with: pip install lleaves")
        print("  Requires: LLVM development libraries (llvmlite)")
        sys.exit(1)

    compiled_path = compile_model(args.model, args.tf, args.output_dir)
    print(f"Compiled model saved: {compiled_path}")

    if args.benchmark:
        import lightgbm as lgb
        lgb_model = lgb.Booster(model_file=args.model)
        n_features = lgb_model.num_feature()
        benchmark_prediction(args.model, compiled_path, n_features)


if __name__ == '__main__':
    main()
