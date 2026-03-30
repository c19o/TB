#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pipeline_orchestrator.py — Modular, Crash-Safe Training Pipeline (v3.1)
========================================================================
Every step is independently resumable. Crash mid-run? Restart and it picks up
exactly where it left off. Each asset × TF × phase is a separate checkpoint.

Usage:
    python pipeline_orchestrator.py                  # run full pipeline
    python pipeline_orchestrator.py --phase train    # run only training phase
    python pipeline_orchestrator.py --tf 1d          # run only 1d timeframe
    python pipeline_orchestrator.py --asset BTC      # run only BTC
    python pipeline_orchestrator.py --status          # show pipeline status
    python pipeline_orchestrator.py --reset phase tf  # reset a specific step

Checkpoint file: pipeline_manifest.json
"""

import os, sys, json, time, hashlib, subprocess, argparse, traceback
from datetime import datetime, timezone

# Ensure unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'

WORKSPACE = os.path.dirname(os.path.abspath(__file__))
os.chdir(WORKSPACE)
sys.path.insert(0, WORKSPACE)

from config import (PROJECT_DIR, V30_DATA_DIR, PIPELINE_MANIFEST,
                    TRAINING_CRYPTO, TIMEFRAMES_ALL_ASSETS, TIMEFRAMES_CRYPTO_ONLY)

# ── Pipeline Definition ──
# Each phase produces specific artifacts. If artifact exists → step is done.
PHASES = [
    'features',     # 1. Build feature parquets + cross NPZ
    'train',        # 2. LightGBM CPCV training → model + OOS predictions
    'optuna',       # 3. Optuna optimizer → trade configs
    'meta',         # 4. Meta-labeling from OOS predictions
    'lstm',         # 5. LSTM sequence model + Platt calibration
    'pbo',          # 6. PBO + Deflated Sharpe validation
    'audit',        # 7. Backtesting audit report
]

# TFs to train (no 5m in v3.1)
ALL_TFS = ['1w', '1d', '4h', '1h', '15m']

# Assets per TF
def get_assets_for_tf(tf):
    if tf in TIMEFRAMES_ALL_ASSETS:
        return ['BTC']  # daily/weekly: BTC primary (multi-asset features baked in)
    return ['BTC']  # crypto intraday: BTC primary


# ══════════════════════════════════════════════════════════════
# MANIFEST: tracks every step's completion state
# ══════════════════════════════════════════════════════════════

def load_manifest():
    if os.path.exists(PIPELINE_MANIFEST):
        with open(PIPELINE_MANIFEST, 'r') as f:
            return json.load(f)
    return {'steps': {}, 'started': None, 'version': '3.1'}


def save_manifest(manifest):
    """Atomic write — crash-safe via temp file + rename."""
    tmp = PIPELINE_MANIFEST + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, PIPELINE_MANIFEST)  # atomic on both Windows and Linux


def step_key(phase, tf, asset='BTC'):
    return f"{phase}:{asset}:{tf}"


def mark_done(manifest, phase, tf, asset='BTC', info=None):
    key = step_key(phase, tf, asset)
    manifest['steps'][key] = {
        'status': 'done',
        'completed_at': datetime.now(timezone.utc).isoformat(),
        'info': info or {},
    }
    save_manifest(manifest)
    log(f"  [OK] CHECKPOINT: {key} done")


def mark_failed(manifest, phase, tf, asset='BTC', error=''):
    key = step_key(phase, tf, asset)
    manifest['steps'][key] = {
        'status': 'failed',
        'failed_at': datetime.now(timezone.utc).isoformat(),
        'error': str(error)[:500],
    }
    save_manifest(manifest)


def is_done(manifest, phase, tf, asset='BTC'):
    key = step_key(phase, tf, asset)
    return manifest['steps'].get(key, {}).get('status') == 'done'


def reset_step(manifest, phase, tf, asset='BTC'):
    key = step_key(phase, tf, asset)
    if key in manifest['steps']:
        del manifest['steps'][key]
        save_manifest(manifest)
        log(f"  Reset: {key}")


# ══════════════════════════════════════════════════════════════
# ARTIFACT CHECKS: verify outputs exist on disk
# ══════════════════════════════════════════════════════════════

def check_features_exist(tf, asset='BTC'):
    """Check if feature parquet exists (either in v3.1 or v3.0).
    NPZ (cross features) are optional — ml_multi_tf.py loads them if present."""
    parquet = os.path.join(PROJECT_DIR, f'features_{asset}_{tf}.parquet')
    parquet_v30 = os.path.join(V30_DATA_DIR, f'features_{asset}_{tf}.parquet')
    return os.path.exists(parquet) or os.path.exists(parquet_v30)


def check_model_exists(tf, asset='BTC'):
    return os.path.exists(os.path.join(PROJECT_DIR, f'model_{tf}.json'))


def check_optuna_exists(tf):
    return os.path.exists(os.path.join(PROJECT_DIR, f'optuna_configs_{tf}.json'))


def check_meta_exists(tf):
    return os.path.exists(os.path.join(PROJECT_DIR, f'meta_model_{tf}.pkl'))


def check_lstm_exists(tf):
    return (os.path.exists(os.path.join(PROJECT_DIR, f'lstm_{tf}.pt')) or
            os.path.exists(os.path.join(PROJECT_DIR, f'platt_{tf}.pkl')))


def check_pbo_exists(tf):
    return os.path.exists(os.path.join(PROJECT_DIR, f'pbo_results_{tf}.json'))


def check_audit_exists(tf):
    return os.path.exists(os.path.join(PROJECT_DIR, f'audit_{tf}.json'))


ARTIFACT_CHECKS = {
    'features': check_features_exist,
    'train': check_model_exists,
    'optuna': lambda tf, asset='BTC': check_optuna_exists(tf),
    'meta': lambda tf, asset='BTC': check_meta_exists(tf),
    'lstm': lambda tf, asset='BTC': check_lstm_exists(tf),
    'pbo': lambda tf, asset='BTC': check_pbo_exists(tf),
    'audit': lambda tf, asset='BTC': check_audit_exists(tf),
}


# ══════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════

_START = time.time()

def log(msg):
    elapsed = time.time() - _START
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts} +{elapsed:.0f}s] {msg}", flush=True)


# ══════════════════════════════════════════════════════════════
# PHASE RUNNERS
# ══════════════════════════════════════════════════════════════

def run_features(tf, asset, manifest):
    """Build features for one asset × TF. Checks v3.0 first."""
    if check_features_exist(tf, asset):
        # Check if data is in v3.0 — create symlink or note in manifest
        parquet_local = os.path.join(PROJECT_DIR, f'features_{asset}_{tf}.parquet')
        parquet_v30 = os.path.join(V30_DATA_DIR, f'features_{asset}_{tf}.parquet')
        source = 'local' if os.path.exists(parquet_local) else 'v3.0'
        mark_done(manifest, 'features', tf, asset, {'source': source})
        return True

    # Need to build — run the appropriate build script
    build_scripts = {
        '1w': 'build_1w_features.py',
        '1d': 'build_1d_features.py',
        '4h': 'build_4h_features.py',
        '1h': 'build_1h_features.py',
        '15m': 'build_15m_features.py',
    }
    script = build_scripts.get(tf)
    if not script or not os.path.exists(os.path.join(PROJECT_DIR, script)):
        log(f"  SKIP features {asset} {tf} — no build script")
        return False

    log(f"  Building features: {asset} {tf} ...")
    ret = run_subprocess(f'python -u {script}', f'build_{asset}_{tf}')
    if ret == 0 and check_features_exist(tf, asset):
        mark_done(manifest, 'features', tf, asset, {'source': 'built'})
        return True
    else:
        mark_failed(manifest, 'features', tf, asset, f'exit code {ret}')
        return False


def run_train(tf, asset, manifest):
    """Train LightGBM model for one TF."""
    log(f"  Training LightGBM: {tf} ...")
    ret = run_subprocess(f'python -u ml_multi_tf.py --tf {tf}', f'train_{tf}')
    if ret == 0 and check_model_exists(tf, asset):
        # Read model info
        info = {}
        feat_path = os.path.join(PROJECT_DIR, f'features_{tf}_all.json')
        if os.path.exists(feat_path):
            with open(feat_path) as f:
                info['n_features'] = len(json.load(f))
        mark_done(manifest, 'train', tf, asset, info)
        return True
    else:
        mark_failed(manifest, 'train', tf, asset, f'exit code {ret}')
        return False


def run_optuna(tf, asset, manifest):
    """Run Optuna optimizer for one TF."""
    log(f"  Optuna optimizer: {tf} ...")
    script = 'run_optuna_local.py'
    if not os.path.exists(os.path.join(PROJECT_DIR, script)):
        script = 'exhaustive_optimizer.py'
    ret = run_subprocess(f'python -u {script} --tf {tf}', f'optuna_{tf}')
    if ret == 0:
        mark_done(manifest, 'optuna', tf, asset)
        return True
    else:
        mark_failed(manifest, 'optuna', tf, asset, f'exit code {ret}')
        return False


def run_meta(tf, asset, manifest):
    """Train meta-labeling model for one TF."""
    log(f"  Meta-labeling: {tf} ...")
    ret = run_subprocess(f'python -u meta_labeling.py --tf {tf}', f'meta_{tf}')
    if ret == 0:
        mark_done(manifest, 'meta', tf, asset)
        return True
    else:
        mark_failed(manifest, 'meta', tf, asset, f'exit code {ret}')
        return False


def run_lstm(tf, asset, manifest):
    """Train LSTM for one TF (GPU)."""
    log(f"  LSTM training: {tf} (GPU) ...")
    ret = run_subprocess(f'python -u lstm_sequence_model.py --tf {tf}', f'lstm_{tf}')
    if ret == 0:
        mark_done(manifest, 'lstm', tf, asset)
        return True
    else:
        mark_failed(manifest, 'lstm', tf, asset, f'exit code {ret}')
        return False


def run_pbo(tf, asset, manifest):
    """PBO + Deflated Sharpe validation for one TF."""
    log(f"  PBO validation: {tf} ...")
    # PBO is usually part of ml_multi_tf.py output, check if separate script exists
    script = 'backtest_validation.py'
    if not os.path.exists(os.path.join(PROJECT_DIR, script)):
        log(f"  SKIP PBO — {script} not found, marking done (PBO computed in training)")
        mark_done(manifest, 'pbo', tf, asset, {'note': 'computed during training'})
        return True
    ret = run_subprocess(f'python -u {script} --tf {tf}', f'pbo_{tf}')
    if ret == 0:
        mark_done(manifest, 'pbo', tf, asset)
        return True
    else:
        mark_failed(manifest, 'pbo', tf, asset, f'exit code {ret}')
        return False


def run_audit(tf, asset, manifest):
    """Backtesting audit for one TF."""
    log(f"  Backtesting audit: {tf} ...")
    ret = run_subprocess(f'python -u backtesting_audit.py --tf {tf}', f'audit_{tf}')
    if ret == 0:
        mark_done(manifest, 'audit', tf, asset)
        return True
    else:
        mark_failed(manifest, 'audit', tf, asset, f'exit code {ret}')
        return False


PHASE_RUNNERS = {
    'features': run_features,
    'train': run_train,
    'optuna': run_optuna,
    'meta': run_meta,
    'lstm': run_lstm,
    'pbo': run_pbo,
    'audit': run_audit,
}


# ══════════════════════════════════════════════════════════════
# SUBPROCESS RUNNER (with live output + log file)
# ══════════════════════════════════════════════════════════════

def run_subprocess(cmd, log_name):
    """Run a command with live output and log file. Returns exit code."""
    log_path = os.path.join(PROJECT_DIR, f'{log_name}.log')
    log(f"  CMD: {cmd}")
    log(f"  LOG: {log_path}")

    try:
        with open(log_path, 'w', encoding='utf-8') as logf:
            proc = subprocess.Popen(
                cmd, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                cwd=WORKSPACE, env={**os.environ, 'PYTHONUNBUFFERED': '1'},
                bufsize=1, universal_newlines=True,
            )
            for line in proc.stdout:
                sys.stdout.write(f"    | {line}")
                sys.stdout.flush()
                logf.write(line)
                logf.flush()
            proc.wait()
        return proc.returncode
    except Exception as e:
        log(f"  SUBPROCESS ERROR: {e}")
        return -1


# ══════════════════════════════════════════════════════════════
# DISK SPACE CHECK
# ══════════════════════════════════════════════════════════════

def check_disk_space(min_gb=10):
    """Check available disk space. Warn if below threshold."""
    try:
        import shutil
        total, used, free = shutil.disk_usage(PROJECT_DIR)
        free_gb = free / (1024**3)
        log(f"  Disk: {free_gb:.1f} GB free / {total/(1024**3):.0f} GB total ({used/total*100:.0f}% used)")
        if free_gb < min_gb:
            log(f"  WARNING: Only {free_gb:.1f} GB free! Need at least {min_gb} GB.")
            return False
        return True
    except Exception:
        return True  # can't check, proceed anyway


# ══════════════════════════════════════════════════════════════
# STATUS DISPLAY
# ══════════════════════════════════════════════════════════════

def show_status(manifest):
    """Pretty-print pipeline status."""
    print("\n" + "=" * 70)
    print("  PIPELINE STATUS — v3.1")
    print("=" * 70)

    for tf in ALL_TFS:
        assets = get_assets_for_tf(tf)
        for asset in assets:
            print(f"\n  {asset} {tf}:")
            for phase in PHASES:
                key = step_key(phase, tf, asset)
                step = manifest['steps'].get(key, {})
                status = step.get('status', 'pending')
                if status == 'done':
                    when = step.get('completed_at', '?')[:19]
                    info = step.get('info', {})
                    extra = ''
                    if 'source' in info:
                        extra = f" (from {info['source']})"
                    if 'n_features' in info:
                        extra = f" ({info['n_features']} features)"
                    print(f"    [DONE] {phase:12s} {when}{extra}")
                elif status == 'failed':
                    err = step.get('error', '?')[:60]
                    print(f"    [FAIL] {phase:12s} {err}")
                else:
                    # Check if artifact exists on disk even without manifest entry
                    checker = ARTIFACT_CHECKS.get(phase)
                    if checker and checker(tf, asset):
                        print(f"    [ OK ] {phase:12s} (artifact exists, not in manifest)")
                    else:
                        print(f"    [    ] {phase:12s}")

    # Disk space
    print()
    check_disk_space()
    print()


# ══════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════

def run_pipeline(phase_filter=None, tf_filter=None, asset_filter=None):
    """Run the full pipeline with checkpoint/resume."""
    manifest = load_manifest()

    if manifest['started'] is None:
        manifest['started'] = datetime.now(timezone.utc).isoformat()
        save_manifest(manifest)

    log("=" * 70)
    log("  SAVAGE22 v3.1 — MODULAR TRAINING PIPELINE")
    log(f"  Started: {datetime.now(timezone.utc).isoformat()}")
    log(f"  Filters: phase={phase_filter or 'all'} tf={tf_filter or 'all'} asset={asset_filter or 'all'}")
    log("=" * 70)

    if not check_disk_space(min_gb=15):
        log("ABORT: Insufficient disk space. Free up space and retry.")
        return False

    phases_to_run = [phase_filter] if phase_filter else PHASES
    tfs_to_run = [tf_filter] if tf_filter else ALL_TFS

    total_steps = 0
    done_steps = 0
    failed_steps = 0

    for phase in phases_to_run:
        if phase not in PHASE_RUNNERS:
            log(f"  Unknown phase: {phase}")
            continue

        log(f"\n{'='*60}")
        log(f"  PHASE: {phase.upper()}")
        log(f"{'='*60}")

        runner = PHASE_RUNNERS[phase]

        for tf in tfs_to_run:
            assets = get_assets_for_tf(tf)
            if asset_filter:
                assets = [a for a in assets if a == asset_filter]

            for asset in assets:
                total_steps += 1

                # Skip if already done
                if is_done(manifest, phase, tf, asset):
                    log(f"  SKIP {asset} {tf} {phase} — already done")
                    done_steps += 1
                    continue

                # Check prerequisites
                if phase != 'features':
                    if not is_done(manifest, 'features', tf, asset):
                        # Try to validate from disk
                        if check_features_exist(tf, asset):
                            mark_done(manifest, 'features', tf, asset,
                                      {'source': 'pre-existing'})
                        else:
                            log(f"  SKIP {asset} {tf} {phase} — features not ready")
                            continue

                if phase in ('optuna', 'meta', 'lstm', 'pbo', 'audit'):
                    if not is_done(manifest, 'train', tf, asset):
                        if check_model_exists(tf, asset):
                            mark_done(manifest, 'train', tf, asset,
                                      {'source': 'pre-existing'})
                        else:
                            log(f"  SKIP {asset} {tf} {phase} — model not trained")
                            continue

                # Run the step
                log(f"\n  >>> {asset} {tf} {phase}")
                try:
                    ok = runner(tf, asset, manifest)
                    if ok:
                        done_steps += 1
                    else:
                        failed_steps += 1
                        log(f"  FAILED: {asset} {tf} {phase}")
                except KeyboardInterrupt:
                    log(f"\n  INTERRUPTED at {asset} {tf} {phase}")
                    log(f"  Progress saved. Restart to resume from this point.")
                    save_manifest(manifest)
                    return False
                except Exception as e:
                    failed_steps += 1
                    mark_failed(manifest, phase, tf, asset, str(e))
                    log(f"  ERROR: {asset} {tf} {phase}: {e}")
                    traceback.print_exc()

    # Summary
    log(f"\n{'='*70}")
    log(f"  PIPELINE COMPLETE")
    log(f"  Total: {total_steps} | Done: {done_steps} | Failed: {failed_steps} | Skipped: {total_steps - done_steps - failed_steps}")
    log(f"  Elapsed: {time.time() - _START:.0f}s ({(time.time() - _START)/60:.1f} min)")
    log(f"{'='*70}")

    manifest['completed'] = datetime.now(timezone.utc).isoformat()
    save_manifest(manifest)
    return failed_steps == 0


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Savage22 v3.1 Modular Pipeline')
    parser.add_argument('--phase', choices=PHASES, help='Run only this phase')
    parser.add_argument('--tf', choices=ALL_TFS, help='Run only this timeframe')
    parser.add_argument('--asset', help='Run only this asset')
    parser.add_argument('--status', action='store_true', help='Show pipeline status')
    parser.add_argument('--reset', nargs=2, metavar=('PHASE', 'TF'),
                        help='Reset a specific step (e.g. --reset train 1d)')
    parser.add_argument('--reset-phase', help='Reset all steps for a phase')
    parser.add_argument('--reset-all', action='store_true', help='Reset entire manifest')
    args = parser.parse_args()

    manifest = load_manifest()

    if args.status:
        show_status(manifest)
        sys.exit(0)

    if args.reset_all:
        manifest = {'steps': {}, 'started': None, 'version': '3.1'}
        save_manifest(manifest)
        log("  Manifest reset.")
        sys.exit(0)

    if args.reset:
        phase, tf = args.reset
        for asset in get_assets_for_tf(tf):
            reset_step(manifest, phase, tf, asset)
        sys.exit(0)

    if args.reset_phase:
        phase = args.reset_phase
        keys_to_del = [k for k in manifest['steps'] if k.startswith(f"{phase}:")]
        for k in keys_to_del:
            del manifest['steps'][k]
        save_manifest(manifest)
        log(f"  Reset {len(keys_to_del)} steps for phase '{phase}'")
        sys.exit(0)

    ok = run_pipeline(args.phase, args.tf, args.asset)
    sys.exit(0 if ok else 1)
