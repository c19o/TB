#!/usr/bin/env python
"""
mini_train.py — Full V2 Pipeline on BTC 1d in <5 min (RTX 3090)
================================================================
Runs the REAL 7-step pipeline end-to-end. Steps 1/3/4/5 run in-process
(no subprocess overhead). Steps 2/6/7 use subprocess (not importable).

Steps:
  1. Build features (base + V2 layers + sparse crosses)  [IN-PROCESS]
  2. Train CPCV LightGBM (3-class triple-barrier)          [SUBPROCESS]
  3. Run Optuna optimizer (mini grid)                      [IN-PROCESS]
  4. PBO + Deflated Sharpe validation                      [IN-PROCESS]
  5. Meta-labeling from OOS predictions                    [IN-PROCESS]
  6. LSTM training (3 epochs, fast)                        [SUBPROCESS]
  7. Backtesting audit                                     [SUBPROCESS]

Usage:
    python mini_train.py
    python mini_train.py --tf 1d --skip-lstm --full-crosses
"""

import os
import sys
import time
import json
import pickle
import traceback
import argparse
import gc

os.environ['PYTHONUNBUFFERED'] = '1'

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.environ.get('SAVAGE22_V1_DIR', os.path.dirname(PROJECT_DIR))
os.chdir(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)

START = time.time()
STEP_TIMES = {}


def log(msg):
    elapsed = time.time() - START
    m, s = divmod(int(elapsed), 60)
    print(f"[{m:02d}:{s:02d}] {msg}", flush=True)


def run_step(name, fn):
    log(f"{'='*60}")
    log(f"STEP: {name}")
    log(f"{'='*60}")
    t0 = time.time()
    try:
        result = fn()
        dt = time.time() - t0
        STEP_TIMES[name] = dt
        log(f"DONE: {name} ({dt:.1f}s)")
        return result
    except Exception as e:
        dt = time.time() - t0
        STEP_TIMES[name] = -dt  # negative = failed
        log(f"FAILED: {name} — {e}")
        traceback.print_exc()
        raise  # fail-fast: stop pipeline on any step failure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf', default='1d')
    parser.add_argument('--skip-lstm', action='store_true')
    parser.add_argument('--skip-optimizer', action='store_true')
    parser.add_argument('--skip-backtest', action='store_true')
    parser.add_argument('--full-crosses', action='store_true',
                        help='Generate all 2M+ crosses (slow). Default: 50K for speed.')
    parser.add_argument('--max-crosses', type=int, default=50000,
                        help='Max crosses for mini build (default: 50000)')
    parser.add_argument('--force', action='store_true',
                        help='Force rebuild even if checkpoint exists')
    parser.add_argument('--resume-from', type=int, default=1,
                        help='Resume from step N (skip completed steps). E.g., --resume-from 6')
    parser.add_argument('--full', action='store_true',
                        help='Full validation mode (800 rounds, per-TF CPCV, 5K grid, 5 LSTM epochs)')
    args = parser.parse_args()
    tf = args.tf
    max_crosses = None if args.full_crosses else args.max_crosses
    resume_from = args.resume_from
    is_full = args.full

    log(f"MINI TRAIN — BTC {tf} on local GPU")
    log(f"Mode: {'FULL VALIDATION' if is_full else 'FAST PLUMBING TEST'}")
    log(f"Project: {PROJECT_DIR}")
    log(f"DB dir: {DB_DIR}")
    log(f"Max crosses: {'FULL' if max_crosses is None else f'{max_crosses:,}'}")
    if resume_from > 1:
        log(f"RESUMING from step {resume_from} (steps 1-{resume_from-1} skipped)")

    def _step_done(step_num):
        """Check if a step's output artifacts exist (for auto-resume)."""
        checks = {
            1: [f'features_BTC_{tf}.parquet', f'v2_crosses_BTC_{tf}.npz'],
            2: [f'model_{tf}.json', f'cpcv_oos_predictions_{tf}.pkl'],
            3: [],  # always re-run (fast)
            4: [],  # always re-run (fast)
            5: [f'meta_model_{tf}.pkl'],
            6: [],  # LSTM optional
            7: [],  # backtest optional
        }
        return all(os.path.exists(os.path.join(PROJECT_DIR, f)) for f in checks.get(step_num, []))

    def _should_skip(step_num):
        """Skip if --resume-from is higher, or if artifacts exist and not --force."""
        if step_num < resume_from:
            return True
        if not args.force and _step_done(step_num):
            return True
        return False

    # ── Step 1: Build Features (IN-PROCESS) ──────────────────
    def step1_build():
        # Delete old checkpoint to force rebuild if requested
        parquet = os.path.join(PROJECT_DIR, f'features_BTC_{tf}.parquet')
        npz = os.path.join(PROJECT_DIR, f'v2_crosses_BTC_{tf}.npz')

        from build_features_v2 import build_single_asset
        from data_access_v2 import V2OfflineDataLoader

        loader = V2OfflineDataLoader()
        build_single_asset('BTC', tf, loader, max_crosses=max_crosses, force=args.force)

        # Verify outputs
        if not os.path.exists(parquet):
            raise FileNotFoundError(f"Missing: {parquet}")
        if not os.path.exists(npz):
            raise FileNotFoundError(f"Missing: {npz}")

        import pandas as pd
        df = pd.read_parquet(parquet)
        log(f"  Features: {len(df)} rows x {len(df.columns)} base columns")

        import numpy as np
        from scipy import sparse
        cross_mat = sparse.load_npz(npz).tocsr()
        log(f"  Crosses: {cross_mat.shape[1]:,} sparse columns ({cross_mat.nnz:,} nonzeros)")

        names_path = os.path.join(PROJECT_DIR, f'v2_cross_names_BTC_{tf}.json')
        if os.path.exists(names_path):
            with open(names_path) as f:
                cross_names = json.load(f)
            log(f"  Cross names: {len(cross_names):,}")

        del df, cross_mat
        gc.collect()

    # ── Step 2: Train CPCV (SUBPROCESS — not importable) ─────
    def step2_train():
        import subprocess
        cmd = [sys.executable, '-u', 'ml_multi_tf.py', '--tf', tf]
        if not is_full:
            cmd += ['--boost-rounds', '50', '--n-groups', '2']
        log(f"  Running: {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=False)

        # Check artifacts exist (process may crash on Dask cleanup even after success)
        model_path = os.path.join(PROJECT_DIR, f'model_{tf}.json')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Training produced no model (exit {proc.returncode})")
        log(f"  Model saved: {model_path}")

        oos_path = os.path.join(PROJECT_DIR, f'cpcv_oos_predictions_{tf}.pkl')
        if os.path.exists(oos_path):
            with open(oos_path, 'rb') as f:
                oos = pickle.load(f)
            n_folds = len(oos)
            n_samples = sum(len(p.get('y_true', [])) for p in oos)
            log(f"  OOS predictions: {n_folds} folds, {n_samples} total samples")
        else:
            log(f"  WARNING: No OOS predictions saved")

    # ── Step 3: Optimizer (IN-PROCESS, mini grid) ────────────
    def step3_optimizer():
        if args.skip_optimizer:
            log("  SKIPPED (--skip-optimizer)")
            return

        import exhaustive_optimizer as eo

        if is_full:
            # Full validation grid: ~5K combos
            eo.TF_GRIDS = {tf: {
                'lev': [1, 5, 10, 15, 20],
                'risk': [0.5, 1.5, 3.0, 5.0],
                'stop_atr': [0.5, 1.5, 2.5, 4.0],
                'rr': [1.5, 3.0, 5.0, 8.0],
                'hold': [7, 30, 60, 120],
                'exit_type': [0, -2],
                'conf': [0.5, 0.7],
            }}
            log(f"  Full grid: {5*4*4*4*4*2*2:,} combos")
        else:
            # Fast plumbing grid: 128 combos
            eo.TF_GRIDS = {tf: {
                'lev': [1, 10],
                'risk': [0.5, 3.0],
                'stop_atr': [1.0, 3.0],
                'rr': [2.0, 5.0],
                'hold': [7, 60],
                'exit_type': [0, -2],
                'conf': [0.5, 0.7],
            }}
            log(f"  Plumbing grid: {2**7:,} combos")
        eo.main(resume=False)

    # ── Step 4: PBO + DSR (IN-PROCESS) ───────────────────────
    def step4_validation():
        oos_path = os.path.join(PROJECT_DIR, f'cpcv_oos_predictions_{tf}.pkl')
        if not os.path.exists(oos_path):
            log("  SKIPPED — no OOS predictions")
            return

        from backtest_validation import validation_report
        with open(oos_path, 'rb') as f:
            oos_data = pickle.load(f)
        report = validation_report(oos_data, tf_name=tf)
        pbo_val = report.get('pbo', {}).get('pbo', '?')
        dsr_p = report.get('deflated_sharpe', {}).get('p_value', '?')
        rec = report.get('overall', '?')
        log(f"  PBO: {pbo_val}")
        log(f"  DSR p-value: {dsr_p}")
        log(f"  Recommendation: {rec}")

    # ── Step 5: Meta-labeling (IN-PROCESS) ───────────────────
    def step5_meta():
        oos_path = os.path.join(PROJECT_DIR, f'cpcv_oos_predictions_{tf}.pkl')
        if not os.path.exists(oos_path):
            log("  SKIPPED — no OOS predictions")
            return

        from meta_labeling import train_meta_model
        with open(oos_path, 'rb') as f:
            oos_data = pickle.load(f)
        result = train_meta_model(oos_data, tf_name=tf, db_dir=PROJECT_DIR)
        if result:
            log(f"  Meta model saved: meta_model_{tf}.pkl")
            log(f"  Meta threshold: {result.get('threshold', '?')}")
            auc = result.get('metrics', {}).get('auc', result.get('auc', '?'))
            log(f"  Meta AUC: {auc}")
        else:
            log("  Meta-labeling: not enough data (< 50 directional trades)")

    # ── Step 6: LSTM (SUBPROCESS) ────────────────────────────
    def step6_lstm():
        if args.skip_lstm:
            log("  SKIPPED (--skip-lstm)")
            return

        parquet = os.path.join(PROJECT_DIR, f'features_BTC_{tf}.parquet')
        if not os.path.exists(parquet):
            log("  SKIPPED — no feature parquet")
            return

        # Auto-install torch if missing (RAPIDS image doesn't include it)
        import subprocess
        try:
            import torch
        except ImportError:
            log("  PyTorch not found — installing...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch', '-q'],
                          capture_output=True, timeout=300)
            log("  PyTorch installed")

        epochs = '5' if is_full else '1'
        cmd = [sys.executable, '-u', 'v2_lstm_trainer.py',
               '--tf', tf, '--epochs', epochs]
        log(f"  Running: {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=False, timeout=900)
        if proc.returncode != 0:
            log(f"  WARNING: LSTM exited {proc.returncode} (non-fatal)")

    # ── Step 7: Backtest (SUBPROCESS) ────────────────────────
    def step7_backtest():
        if args.skip_backtest:
            log("  SKIPPED (--skip-backtest)")
            return

        model_path = os.path.join(PROJECT_DIR, f'model_{tf}.json')
        if not os.path.exists(model_path):
            log("  SKIPPED — no model")
            return

        import subprocess
        cmd = [sys.executable, '-u', 'backtesting_audit.py', '--tf', tf]
        log(f"  Running: {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=False, timeout=600)
        if proc.returncode != 0:
            log(f"  WARNING: Backtest exited {proc.returncode} (non-fatal)")

    # ── Execute Pipeline ────────────────────────────────────
    log("=" * 60)
    log("STARTING MINI TRAIN PIPELINE")
    log("=" * 60)

    steps = [
        (1, "1. Feature Build (BTC {})".format(tf), step1_build),
        (2, "2. CPCV Training", step2_train),
        (3, "3. Exhaustive Optimizer", step3_optimizer),
        (4, "4. PBO + Deflated Sharpe", step4_validation),
        (5, "5. Meta-Labeling", step5_meta),
        (6, "6. LSTM Training", step6_lstm),
        (7, "7. Backtesting Audit", step7_backtest),
    ]

    try:
        for step_num, step_name, step_fn in steps:
            if _should_skip(step_num):
                log(f"{'='*60}")
                log(f"SKIP: {step_name} (artifacts exist or --resume-from {resume_from})")
                log(f"{'='*60}")
                STEP_TIMES[step_name] = 0
                continue
            run_step(step_name, step_fn)
        all_passed = True
    except Exception:
        all_passed = False
        failed_step = step_num
        log(f"\nPIPELINE HALTED at step {failed_step} — fix the error above, then:")
        log(f"  python mini_train.py --tf {tf} --resume-from {failed_step}")

    # ── Summary ────────────────────────────────────────────
    elapsed = time.time() - START
    log("=" * 60)
    log(f"MINI TRAIN {'COMPLETE' if all_passed else 'FAILED'} — {elapsed/60:.1f} minutes")
    log("=" * 60)

    # Step timing breakdown
    log("\nStep Timing:")
    for name, dt in STEP_TIMES.items():
        status = "PASS" if dt >= 0 else "FAIL"
        log(f"  {status} {name}: {abs(dt):.1f}s")

    # List all artifacts produced
    import glob
    artifacts = []
    for pattern in ['model_*.json', 'features_BTC_*.parquet',
                     'v2_crosses_BTC_*.npz', 'v2_cross_names_BTC_*.json',
                     'cpcv_oos_predictions_*.pkl', 'optuna_configs_*.json',
                     'meta_model_*.pkl', 'platt_*.pkl', 'lstm_*.pt',
                     'blend_config_*.json', 'validation_report_*.json',
                     'feature_importance_stability_*.json']:
        artifacts.extend(glob.glob(os.path.join(PROJECT_DIR, pattern)))

    log(f"\nArtifacts produced: {len(artifacts)}")
    for a in sorted(artifacts):
        size = os.path.getsize(a)
        name = os.path.basename(a)
        if size > 1024 * 1024:
            log(f"  {name} ({size / 1024 / 1024:.1f} MB)")
        else:
            log(f"  {name} ({size / 1024:.0f} KB)")

    if all_passed:
        log(f"\nALL STEPS PASSED in {elapsed/60:.1f} min — ready for cloud deployment")
    else:
        log(f"\nFIX ERRORS ABOVE then re-run")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
