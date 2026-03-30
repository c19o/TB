#!/usr/bin/env python
"""
synthetic_pipeline_test.py -- Pipeline Validation + RAM Estimation
=================================================================
Generates synthetic data at REAL feature dimensions but fewer rows.
Runs the full ml_multi_tf.py pipeline. Measures RAM + timing per stage.

NEVER used for real training -- validation + RAM estimation only.
Random features -> ~33% accuracy (chance level). Expected.

Usage:
    python -u synthetic_pipeline_test.py                # all TFs
    python -u synthetic_pipeline_test.py --tf 1w        # single TF
    python -u synthetic_pipeline_test.py --tf 1w --tf 1d  # multiple
"""

import os, sys, time, json, argparse, gc, shutil, subprocess
import numpy as np
import pandas as pd
from scipy import sparse as sp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# ── TF specs: real feature dimensions, synthetic row counts ──
TF_SPECS = {
    '1w':  {'rows': 200,  'cross_cols': 2_200_000,  'density': 0.027, 'real_rows': 818,    'freq': 'W-MON'},
    '1d':  {'rows': 500,  'cross_cols': 6_000_000,  'density': 0.025, 'real_rows': 5_733,  'freq': 'D'},
    '4h':  {'rows': 1000, 'cross_cols': 8_000_000,  'density': 0.020, 'real_rows': 23_352, 'freq': '4h'},
    '1h':  {'rows': 500,  'cross_cols': 9_000_000,  'density': 0.015, 'real_rows': 75_406, 'freq': '1h'},
    '15m': {'rows': 500,  'cross_cols': 10_000_000, 'density': 0.012, 'real_rows': 294_000,'freq': '15min'},
}

BASE_FEATURE_COUNT = 50  # minimal base features for pipeline to work


def get_rss_mb():
    """Current process RSS in MB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024**2
    except ImportError:
        try:
            with open(f'/proc/{os.getpid()}/status') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        return int(line.split()[1]) / 1024
        except Exception:
            return 0


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def generate_synthetic_parquet(tf, spec, output_dir):
    """Generate minimal OHLCV + base feature parquet with synthetic data."""
    t0 = time.time()
    rows = spec['rows']
    rng = np.random.default_rng(42)

    # Random walk price data (needed for triple-barrier labels)
    close = 30000 + np.cumsum(rng.normal(0, 300, rows))
    close = np.maximum(close, 1000)  # floor at 1000

    df = pd.DataFrame({
        'timestamp': pd.date_range('2019-01-01', periods=rows, freq=spec['freq']),
        'open': close * rng.uniform(0.995, 1.005, rows),
        'high': close * rng.uniform(1.001, 1.025, rows),
        'low':  close * rng.uniform(0.975, 0.999, rows),
        'close': close,
        'volume': rng.uniform(1e6, 1e9, rows),
    })

    # Synthetic base features (random, named like real features)
    for i in range(BASE_FEATURE_COUNT):
        df[f'synth_feat_{i:04d}'] = rng.standard_normal(rows).astype(np.float32)

    # Add some "real-looking" columns that ml_multi_tf.py checks for
    df['ema50_declining'] = rng.integers(0, 2, rows).astype(np.float32)
    df['ema50_rising'] = (1 - df['ema50_declining']).astype(np.float32)
    df['triple_barrier_label'] = rng.choice([0, 1, 2], size=rows, p=[0.15, 0.60, 0.25]).astype(np.float32)

    path = os.path.join(output_dir, f'features_BTC_{tf}.parquet')
    df.to_parquet(path)
    elapsed = time.time() - t0
    log(f"  Parquet: {rows} rows × {len(df.columns)} cols -> {path} ({elapsed:.1f}s)")
    return path


def generate_synthetic_npz(tf, spec, output_dir):
    """Generate random binary sparse CSR at real feature dimensions."""
    t0 = time.time()
    rows = spec['rows']
    cols = spec['cross_cols']
    density = spec['density']

    nnz = int(rows * cols * density)
    rng = np.random.default_rng(42)

    log(f"  Generating sparse matrix: {rows} × {cols:,} @ {density*100:.1f}% density = {nnz:,} NNZ...")
    ram_before = get_rss_mb()

    # Fast COO construction (NOT scipy.sparse.random which is slow)
    row_idx = rng.integers(0, rows, size=nnz, dtype=np.int32)
    col_idx = rng.integers(0, cols, size=nnz, dtype=np.int32)
    data = np.ones(nnz, dtype=np.float32)

    X = sp.csr_matrix((data, (row_idx, col_idx)), shape=(rows, cols))
    X.sum_duplicates()  # deduplicate random collisions

    ram_after = get_rss_mb()
    npz_path = os.path.join(output_dir, f'v2_crosses_BTC_{tf}.npz')
    sp.save_npz(npz_path, X)
    npz_mb = os.path.getsize(npz_path) / 1e6

    elapsed = time.time() - t0
    log(f"  NPZ: {X.shape[0]} × {X.shape[1]:,} ({X.nnz:,} NNZ) -> {npz_mb:.1f} MB ({elapsed:.1f}s)")
    log(f"  RAM for sparse gen: {ram_after - ram_before:.0f} MB")

    # Generate matching cross names JSON
    names = [f'synth_cross_{i}' for i in range(X.shape[1])]
    names_path = os.path.join(output_dir, f'v2_cross_names_BTC_{tf}.json')
    with open(names_path, 'w') as f:
        json.dump(names, f)
    log(f"  Cross names: {len(names):,} -> {names_path}")

    del X, row_idx, col_idx, data
    gc.collect()
    return npz_path


def run_training(tf, output_dir, boost_rounds=50):
    """Run ml_multi_tf.py on synthetic data. Returns (elapsed, peak_rss_mb, success)."""
    t0 = time.time()

    env = os.environ.copy()
    env['SAVAGE22_DB_DIR'] = os.path.dirname(output_dir)  # parent dir for V1 DBs
    env['V30_DATA_DIR'] = output_dir
    env['PYTHONUNBUFFERED'] = '1'
    env['SKIP_LLM'] = '1'

    cmd = [
        sys.executable, '-u', os.path.join(output_dir, 'ml_multi_tf.py'),
        '--tf', tf,
        '--no-parallel-splits',
        '--boost-rounds', str(boost_rounds),
    ]

    log(f"  Running: {' '.join(cmd[-6:])}")
    log(f"  V30_DATA_DIR={output_dir}")
    log(f"  SAVAGE22_DB_DIR={env['SAVAGE22_DB_DIR']}")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1800, env=env, cwd=output_dir
        )
        elapsed = time.time() - t0

        # Parse output for key metrics
        lines = result.stdout.split('\n')
        acc_lines = [l for l in lines if 'Acc=' in l]
        feature_lines = [l for l in lines if 'Features:' in l]
        error_lines = [l for l in lines if 'Error' in l.lower() or 'Traceback' in l or 'CRITICAL' in l]
        converting_lines = [l for l in lines if 'Converting sparse' in l or 'Keeping SPARSE' in l]
        sparse_lines = [l for l in lines if 'is_enable_sparse' in l]

        success = result.returncode == 0 and len(acc_lines) > 0

        log(f"  Exit code: {result.returncode} | Time: {elapsed:.0f}s | Success: {success}")
        for l in converting_lines[:3]:
            log(f"    {l.strip()}")
        for l in sparse_lines[:3]:
            log(f"    {l.strip()}")
        for l in feature_lines[:3]:
            log(f"    {l.strip()}")
        for l in acc_lines[:5]:
            log(f"    {l.strip()}")
        if error_lines:
            log(f"  ERRORS ({len(error_lines)}):")
            for l in error_lines[:10]:
                log(f"    {l.strip()}")
        if not success and result.stderr:
            log(f"  STDERR (last 500 chars): {result.stderr[-500:]}")

        # Extract peak RSS from the subprocess if possible
        peak_rss = 0
        rss_lines = [l for l in lines if 'RSS' in l.upper() or 'memory' in l.lower()]
        for l in rss_lines[:3]:
            log(f"    {l.strip()}")

        return elapsed, peak_rss, success, result.stdout

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        log(f"  TIMEOUT after {elapsed:.0f}s")
        return elapsed, 0, False, ""
    except Exception as e:
        elapsed = time.time() - t0
        log(f"  EXCEPTION: {e}")
        return elapsed, 0, False, ""


def cleanup_synthetic(tf, output_dir):
    """Remove synthetic artifacts."""
    patterns = [
        f'features_BTC_{tf}.parquet',
        f'v2_crosses_BTC_{tf}.npz',
        f'v2_cross_names_BTC_{tf}.json',
        f'model_{tf}.json',
        f'model_{tf}_fold*.txt',
        f'platt_{tf}.pkl',
        f'cpcv_checkpoint_{tf}.pkl',
        f'lgbm_ckpt_{tf}_*.txt',
        f'ml_multi_tf_results.txt',
        f'ml_multi_tf_configs.json',
    ]
    import glob
    count = 0
    for pat in patterns:
        for f in glob.glob(os.path.join(output_dir, pat)):
            os.remove(f)
            count += 1
    log(f"  Cleaned {count} synthetic artifacts")


def estimate_real_ram(tf, spec, synth_ram_mb):
    """Extrapolate from synthetic RAM to real RAM."""
    rows_synth = spec['rows']
    rows_real = spec['real_rows']
    cols = spec['cross_cols']
    density = spec['density']

    # Fixed component: histogram memory (feature-count-driven, independent of rows)
    # EFB bundles: ~cols/254 bundles × 2 bins × 8 bytes × 3 classes
    n_bundles = max(1, cols // 254)
    hist_ram_gb = n_bundles * 2 * 8 * 3 / 1e9

    # Row-proportional component: CSR data + gradients
    nnz_real = rows_real * cols * density
    csr_ram_gb = nnz_real * 5 / 1e9  # data(4B) + indices(4B) ~ 5 bytes avg with indptr

    # LightGBM internal overhead: ~1.5-2x CSR
    lgbm_overhead_gb = csr_ram_gb * 1.5

    total_gb = hist_ram_gb + csr_ram_gb + lgbm_overhead_gb
    return total_gb, hist_ram_gb, csr_ram_gb


def main():
    parser = argparse.ArgumentParser(description='Synthetic pipeline validation + RAM estimation')
    parser.add_argument('--tf', action='append', help='TF to test (can repeat). Default: all')
    parser.add_argument('--boost-rounds', type=int, default=50, help='LightGBM rounds (default 50)')
    parser.add_argument('--no-cleanup', action='store_true', help='Keep synthetic artifacts')
    args = parser.parse_args()

    tfs = args.tf if args.tf else list(TF_SPECS.keys())

    log("=" * 70)
    log("SYNTHETIC PIPELINE VALIDATION + RAM ESTIMATION")
    log("=" * 70)
    log(f"TFs to test: {tfs}")
    log(f"Boost rounds: {args.boost_rounds}")
    log(f"Working dir: {SCRIPT_DIR}")
    log("")

    results = {}

    for tf in tfs:
        if tf not in TF_SPECS:
            log(f"SKIP: Unknown TF '{tf}'")
            continue

        spec = TF_SPECS[tf]
        log(f"\n{'='*70}")
        log(f"  {tf.upper()} -- {spec['rows']} synth rows, {spec['cross_cols']:,} features, {spec['density']*100:.1f}% density")
        log(f"  Real: {spec['real_rows']:,} rows")
        log(f"{'='*70}")

        ram_start = get_rss_mb()
        t_start = time.time()

        # Stage 1: Generate synthetic parquet
        log(f"\n  [Stage 1] Generate synthetic parquet...")
        t1 = time.time()
        generate_synthetic_parquet(tf, spec, SCRIPT_DIR)
        t_parquet = time.time() - t1

        # Stage 2: Generate synthetic NPZ
        log(f"\n  [Stage 2] Generate synthetic NPZ...")
        t2 = time.time()
        generate_synthetic_npz(tf, spec, SCRIPT_DIR)
        ram_after_gen = get_rss_mb()
        t_npz = time.time() - t2

        gc.collect()

        # Stage 3: Run training pipeline
        log(f"\n  [Stage 3] Run ml_multi_tf.py training...")
        t_train, peak_rss, success, stdout = run_training(tf, SCRIPT_DIR, args.boost_rounds)

        # Stage 4: RAM estimation
        log(f"\n  [Stage 4] RAM extrapolation...")
        est_total_gb, est_hist_gb, est_csr_gb = estimate_real_ram(tf, spec, ram_after_gen - ram_start)
        log(f"  Fixed histogram RAM: {est_hist_gb:.2f} GB")
        log(f"  CSR data RAM (real rows): {est_csr_gb:.1f} GB")
        log(f"  Estimated total (with LightGBM overhead): {est_total_gb:.0f} GB")

        # Cleanup
        if not args.no_cleanup:
            log(f"\n  [Cleanup]")
            cleanup_synthetic(tf, SCRIPT_DIR)

        t_total = time.time() - t_start

        results[tf] = {
            'synth_rows': spec['rows'],
            'real_rows': spec['real_rows'],
            'features': spec['cross_cols'],
            'density': spec['density'],
            't_parquet': t_parquet,
            't_npz': t_npz,
            't_train': t_train,
            't_total': t_total,
            'success': success,
            'est_ram_gb': est_total_gb,
            'est_hist_gb': est_hist_gb,
            'est_csr_gb': est_csr_gb,
            'ram_gen_mb': ram_after_gen - ram_start,
        }

    # ── Final Report ──
    log(f"\n\n{'='*70}")
    log("RESULTS SUMMARY")
    log(f"{'='*70}\n")

    log(f"{'TF':<6} {'Status':<8} {'Features':>12} {'Synth':>8} {'NPZ Gen':>8} {'Train':>8} {'Total':>8} {'Est RAM':>10} {'Machine':>12}")
    log("-" * 95)
    for tf, r in results.items():
        status = "PASS" if r['success'] else "FAIL"
        machine = (
            "64GB" if r['est_ram_gb'] < 50 else
            "256GB" if r['est_ram_gb'] < 200 else
            "512GB" if r['est_ram_gb'] < 400 else
            "1TB" if r['est_ram_gb'] < 800 else
            "2TB+"
        )
        synth_label = f"{r['synth_rows']}r"
        log(f"{tf:<6} {status:<8} {r['features']:>12,} {synth_label:>8} {r['t_npz']:>7.0f}s {r['t_train']:>7.0f}s {r['t_total']:>7.0f}s {r['est_ram_gb']:>8.0f} GB {machine:>12}")

    log(f"\n  RAM estimates include ~1.5x LightGBM overhead on top of CSR data.")
    log(f"  Machine recommendations based on estimated RAM + 30% headroom.\n")

    # Extrapolated training times
    log(f"{'TF':<6} {'Real Rows':>10} {'Synth Rows':>11} {'Scale':>6} {'Synth Train':>12} {'Est Real Train':>15}")
    log("-" * 70)
    for tf, r in results.items():
        scale = r['real_rows'] / r['synth_rows']
        # Training time scales roughly with rows (histogram constant, boosting linear)
        # But histogram dominates at high feature count, so scale < linear
        est_real = r['t_train'] * (scale ** 0.7)  # sub-linear scaling
        log(f"{tf:<6} {r['real_rows']:>10,} {r['synth_rows']:>11,} {scale:>5.0f}x {r['t_train']:>11.0f}s {est_real:>13.0f}s ({est_real/3600:.1f}h)")

    log(f"\n  Time estimates use rows^0.7 scaling (histogram overhead is fixed, only boosting scales with rows).")
    log(f"  These are ROUGH estimates -- actual training with ES=333 may vary 2-3x.")


if __name__ == '__main__':
    main()
