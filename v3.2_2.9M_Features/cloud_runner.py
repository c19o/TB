#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cloud_runner.py — Full 5-TF Pipeline with GPU Pinning
=======================================================
Assembly-line execution: each TF gets a GPU, runs build→train→optimizer→PBO→meta→LSTM→audit.
Training starts as soon as build completes — no waiting.
TFs: 1w, 1d, 4h, 1h, 15m (no 5m).

Usage:
    tmux new -s train 'cd /workspace && PYTHONUNBUFFERED=1 python cloud_runner.py 2>&1 | tee train.log'
"""
import os
os.environ['PYTHONUNBUFFERED'] = '1'

import sys, time, json, argparse, threading, subprocess, traceback, pickle, gc
import numpy as np

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
os.chdir(WORKSPACE)
START = time.time()
SEED = 42
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# ALL processes see ALL GPUs — no CUDA_VISIBLE_DEVICES pinning
# DOY cross code distributes across GPUs automatically via CuPy multi-GPU
# Phase 1: 1w+1d+4h parallel (tiny, share GPUs fine)
# Phase 2: 1h→15m sequential (each gets full VRAM)

PHASE1_PARALLEL = ['1w', '1d', '4h']
PHASE2_SEQUENTIAL = ['1h', '15m']


def log(msg):
    print(f"[{time.time()-START:6.0f}s] {msg}", flush=True)


# ============================================================
# HEARTBEAT — shows ALL active processes every 30s
# ============================================================
class State:
    def __init__(self):
        self.active = {}  # {tf: 'step description'}
        self.completed = {}  # {tf: 'OK' or 'FAIL:reason'}
        self.done = False

def heartbeat(state):
    while not state.done:
        time.sleep(30)
        if state.done:
            break
        elapsed = time.time() - START
        active = ' | '.join(f"{tf}:{step}" for tf, step in state.active.items()) or 'idle'
        done_count = sum(1 for v in state.completed.values() if v == 'OK')
        fail_count = sum(1 for v in state.completed.values() if v.startswith('FAIL'))
        # GPU utilization
        gpu = ''
        try:
            r = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                               '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                parts = [f"G{i}:{l.split(',')[0].strip()}%/{l.split(',')[1].strip()}MB"
                         for i, l in enumerate(r.stdout.strip().split('\n'))]
                gpu = ' '.join(parts)
        except Exception:
            pass
        print(f"[HEARTBEAT {elapsed:.0f}s] {done_count} done {fail_count} fail | {active} | {gpu}", flush=True)


# ============================================================
# RUN A STEP (subprocess with output streaming)
# ============================================================
def run_step(tf, cmd, step_name, state):
    """Run a subprocess with ALL GPUs visible, streaming output."""
    state.active[tf] = step_name
    env = os.environ.copy()
    env.pop('CUDA_VISIBLE_DEVICES', None)  # ensure ALL GPUs visible
    env['PYTHONUNBUFFERED'] = '1'

    log(f"[{tf}] START {step_name} (all GPUs)")
    t0 = time.time()

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, universal_newlines=True, cwd=WORKSPACE, env=env
    )
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            print(f"  [{tf}|{step_name}] {line}", flush=True)
    proc.wait()

    dt = time.time() - t0
    if proc.returncode != 0:
        log(f"[{tf}] FAIL {step_name} (exit {proc.returncode}) after {dt:.0f}s")
        return False
    log(f"[{tf}] DONE {step_name} in {dt:.0f}s")
    return True


def gpu_cleanup(phase_name):
    """Free GPU memory between major phases."""
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        import cupy as cp
        pool = cp.get_default_memory_pool()
        pool.free_all_blocks()
    except Exception:
        pass
    log(f"GPU memory cleanup after {phase_name}")


# ============================================================
# PER-TF FULL PIPELINE
# ============================================================
def build_and_train_tf(tf, state):
    """Build features + train CPCV for one TF. Optimizer/PBO/meta/LSTM run later."""
    log(f"\n{'='*50}")
    log(f"[{tf}] BUILD + TRAIN (all GPUs)")
    log(f"{'='*50}")

    # 1. BUILD FEATURES
    parquet = f'{WORKSPACE}/features_{tf}.parquet'
    if os.path.exists(parquet):
        sz = os.path.getsize(parquet) / 1e6
        log(f"[{tf}] SKIP build — features_{tf}.parquet exists ({sz:.0f}MB)")
    else:
        ok = run_step(tf, [sys.executable, '-u', f'build_{tf}_features.py'],
                      'build', state)
        if not ok:
            state.completed[tf] = 'FAIL:build'
            state.active.pop(tf, None)
            return

    # Validate parquet
    try:
        import pyarrow.parquet as pq
        meta = pq.read_metadata(parquet)
        log(f"[{tf}] Parquet OK: {meta.num_columns} columns, {meta.num_rows} rows")
    except Exception as e:
        log(f"[{tf}] Parquet INVALID: {e}")
        state.completed[tf] = 'FAIL:validate'
        state.active.pop(tf, None)
        return

    # 2. TRAIN LightGBM CPCV
    model_path = f'{WORKSPACE}/model_{tf}.json'
    if os.path.exists(model_path):
        log(f"[{tf}] SKIP train — model_{tf}.json exists")
    else:
        ok = run_step(tf, [sys.executable, '-u', 'ml_multi_tf.py', '--tf', tf],
                      'train', state)
        if not ok:
            state.completed[tf] = 'FAIL:train'
            state.active.pop(tf, None)
            return

    state.completed[tf] = 'OK:trained'
    state.active.pop(tf, None)
    log(f"[{tf}] BUILD + TRAIN COMPLETE")


def _get_optimizer_trial_count(tf):
    """Compute actual trial count from exhaustive_optimizer grid for a TF."""
    try:
        from exhaustive_optimizer import TF_GRIDS, count_grid
        if tf in TF_GRIDS:
            return count_grid(TF_GRIDS[tf])
    except Exception:
        pass
    # Fallback: known grid sizes per TF
    known = {'15m': 5_808_000, '1h': 7_260_000,
             '4h': 5_445_000, '1d': 3_630_000, '1w': 3_630_000}
    return known.get(tf, 5_000_000)


def run_pbo_meta(tf, state):
    """Run PBO + meta-labeling for one TF."""
    # PBO + DEFLATED SHARPE (with proper IS metrics + simulated returns for DSR)
    state.active[tf] = 'pbo'
    try:
        from backtest_validation import validation_report, save_report
        oos_path = f'{WORKSPACE}/cpcv_oos_predictions_{tf}.pkl'
        if os.path.exists(oos_path):
            with open(oos_path, 'rb') as f:
                oos = pickle.load(f)
            n_trials = _get_optimizer_trial_count(tf)
            log(f"[{tf}] PBO using n_optimizer_trials={n_trials:,}")

            # Extract IS metrics from OOS predictions (saved by ml_multi_tf.py)
            is_metrics = None
            if oos and 'is_accuracy' in oos[0]:
                is_metrics = [{'path': p['path'],
                               'is_accuracy': p.get('is_accuracy', 0.0),
                               'is_sharpe': p.get('is_sharpe', 0.0),
                               'is_mlogloss': p.get('is_mlogloss', 0.0)}
                              for p in oos]
                log(f"[{tf}] IS metrics found for {len(is_metrics)} paths (proper PBO)")
            else:
                log(f"[{tf}] WARNING: No IS metrics in OOS pkl — PBO will use fallback half-split")

            # Compute simulated returns from OOS predictions for DSR
            all_returns = []
            for p in oos:
                y_true = p['y_true']
                y_probs = p['y_pred_probs']
                pred_labels = np.argmax(y_probs, axis=1)
                # +1 if correct directional call, -1 if wrong
                sim_ret = np.where(pred_labels == y_true, 1.0, -1.0)
                all_returns.append(sim_ret)
            if all_returns:
                returns = np.concatenate(all_returns)
                obs_std = np.std(returns, ddof=1)
                observed_sharpe = float(np.mean(returns) / max(obs_std, 1e-10) * np.sqrt(252))
                log(f"[{tf}] DSR: {len(returns)} OOS return obs, observed Sharpe={observed_sharpe:.3f}")
            else:
                returns = None
                observed_sharpe = None

            report = validation_report(
                oos,
                observed_sharpe=observed_sharpe,
                n_optimizer_trials=n_trials,
                returns=returns,
                tf_name=tf,
                is_metrics=is_metrics,
            )
            save_report(report, db_dir=WORKSPACE)
            pbo = report.get('pbo', {}).get('pbo', 'N/A')
            pbo_method = report.get('pbo', {}).get('method', '?')
            dsr_p = report.get('deflated_sharpe', {}).get('p_value', 'N/A')
            log(f"[{tf}] PBO={pbo} (method={pbo_method}), DSR p={dsr_p}, Overall={report.get('overall', '?')}")
    except Exception as e:
        log(f"[{tf}] PBO failed: {e}")
        import traceback as _tb
        log(f"[{tf}] PBO traceback: {_tb.format_exc()}")

    # META-LABELING
    state.active[tf] = 'meta'
    try:
        from meta_labeling import train_meta_model
        oos_path = f'{WORKSPACE}/cpcv_oos_predictions_{tf}.pkl'
        if os.path.exists(oos_path):
            with open(oos_path, 'rb') as f:
                oos = pickle.load(f)
            result = train_meta_model(oos, tf_name=tf, db_dir=WORKSPACE)
            if result:
                log(f"[{tf}] Meta-model AUC={result['metrics']['auc']:.3f}")
    except Exception as e:
        log(f"[{tf}] Meta-labeling failed: {e}")
    state.active.pop(tf, None)


# ============================================================
# GPU QUEUE WORKER
# ============================================================
def tf_worker(tf, state):
    """Thread target for parallel build+train."""
    build_and_train_tf(tf, state)


# ============================================================
# SANITY CHECK
# ============================================================
def sanity_check():
    log("=== SANITY CHECK ===")
    try:
        import cupy as cp
        n = cp.cuda.runtime.getDeviceCount()
        for i in range(n):
            p = cp.cuda.runtime.getDeviceProperties(i)
            name = p['name'].decode() if isinstance(p['name'], bytes) else p['name']
            log(f"  GPU {i}: {name}")
    except Exception as e:
        log(f"  GPU check failed: {e}")
        return False

    try:
        import lightgbm as lgb
        log(f"  LightGBM: v{lgb.__version__} OK")
    except Exception as e:
        log(f"  LightGBM: FAIL ({e})")

    try:
        import psutil
        log(f"  RAM: {psutil.virtual_memory().total/1e9:.0f}GB total, {psutil.virtual_memory().available/1e9:.0f}GB free")
    except Exception:
        pass

    required = ['feature_library.py', 'ml_multi_tf.py', 'btc_prices.db']
    for f in required:
        if os.path.exists(f):
            log(f"  {f}: OK")
        else:
            log(f"  {f}: MISSING!")
            return False

    log("=== SANITY CHECK PASSED ===")
    return True


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf', action='append', help='Specific TFs (default: all 5)')
    parser.add_argument('--skip-build', action='store_true', help='Skip feature builds')
    args = parser.parse_args()

    state = State()
    hb = threading.Thread(target=heartbeat, args=(state,), daemon=True)
    hb.start()

    log(f"{'='*60}")
    log("FULL 5-TF PIPELINE — GPU-PINNED ASSEMBLY LINE")
    log(f"{'='*60}")

    if not sanity_check():
        state.done = True
        return 1

    # Determine which TFs to run
    all_tfs = args.tf or (PHASE1_PARALLEL + PHASE2_SEQUENTIAL)

    if args.tf:
        # Specific TFs — build+train sequentially, then optimizer+PBO+meta+LSTM
        for tf in args.tf:
            build_and_train_tf(tf, state)
    else:
        # PHASE 1: Build + Train 1w/1d/4h parallel
        log(f"\n{'='*60}")
        log("PHASE 1: BUILD+TRAIN 1w + 1d + 4h PARALLEL")
        log(f"{'='*60}")
        threads = []
        for tf in PHASE1_PARALLEL:
            t = threading.Thread(target=build_and_train_tf, args=(tf, state))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        p1 = [f'{tf}:{state.completed.get(tf, "?")}' for tf in PHASE1_PARALLEL]
        log(f"PHASE 1 COMPLETE: {p1}")
        gpu_cleanup('build+train phase 1')

        # PHASE 2: Build + Train 1h → 15m sequential
        log(f"\n{'='*60}")
        log("PHASE 2: BUILD+TRAIN 1h, 15m SEQUENTIAL")
        log(f"{'='*60}")
        for tf in PHASE2_SEQUENTIAL:
            build_and_train_tf(tf, state)
        gpu_cleanup('build+train phase 2')

    # PHASE 3: Exhaustive optimizer (ALL TFs at once)
    trained_tfs = [tf for tf in all_tfs if state.completed.get(tf, '').startswith('OK')]
    if trained_tfs and os.path.exists(f'{WORKSPACE}/exhaustive_optimizer.py'):
        log(f"\n{'='*60}")
        log(f"PHASE 3: EXHAUSTIVE OPTIMIZER (all {len(trained_tfs)} TFs)")
        log(f"{'='*60}")
        state.active['optimizer'] = 'all_tfs'
        opt_args = []
        for tf in trained_tfs:
            opt_args.extend(['--tf', tf])
        run_step('optimizer', [sys.executable, '-u', 'exhaustive_optimizer.py'] + opt_args,
                 'optimizer', state)
        state.active.pop('optimizer', None)
        gpu_cleanup('optimizer phase 3')

    # PHASE 4: PBO + Meta-labeling (per TF, lightweight)
    log(f"\n{'='*60}")
    log("PHASE 4: PBO + META-LABELING")
    log(f"{'='*60}")
    for tf in trained_tfs:
        run_pbo_meta(tf, state)
    gpu_cleanup('pbo+meta phase 4')

    # PHASE 5: LSTM training (all TFs at once)
    if trained_tfs and os.path.exists(f'{WORKSPACE}/lstm_sequence_model.py'):
        log(f"\n{'='*60}")
        log(f"PHASE 5: LSTM TRAINING (--all)")
        log(f"{'='*60}")
        run_step('lstm', [sys.executable, '-u', 'lstm_sequence_model.py', '--train', '--all'],
                 'lstm_all', state)
        gpu_cleanup('lstm phase 5')

    # Mark all trained TFs as fully complete
    for tf in trained_tfs:
        state.completed[tf] = 'OK'

    # PHASE 7: Backtesting audit (per TF)
    if trained_tfs and os.path.exists(f'{WORKSPACE}/backtesting_audit.py'):
        log(f"\n{'='*60}")
        log(f"PHASE 7: BACKTESTING AUDIT ({len(trained_tfs)} TFs)")
        log(f"{'='*60}")
        for tf in trained_tfs:
            run_step(tf, [sys.executable, '-u', 'backtesting_audit.py', '--tf', tf],
                     'audit', state)

    # SUMMARY
    total = time.time() - START
    log(f"\n{'='*60}")
    log("PIPELINE COMPLETE")
    log(f"{'='*60}")
    for tf, status in state.completed.items():
        log(f"  {tf}: {status}")
    log(f"Total time: {total:.0f}s ({total/60:.1f}min)")

    # List outputs
    import glob
    log(f"\nOutputs:")
    for pattern in ['model_*.json', 'features_*_all.json', 'features_*.parquet',
                     'cpcv_*.pkl', 'meta_model_*.pkl', 'platt_*.pkl',
                     'validation_report_*.json', 'feature_importance_*.json',
                     'optuna_configs*.json']:
        for f in sorted(glob.glob(f'{WORKSPACE}/{pattern}')):
            sz = os.path.getsize(f) / 1e6
            log(f"  {os.path.basename(f)} ({sz:.1f}MB)")

    state.done = True
    n_fail = sum(1 for v in state.completed.values() if v.startswith('FAIL'))
    return 1 if n_fail > 0 else 0


if __name__ == '__main__':
    sys.exit(main())
