#!/usr/bin/env python
"""
build_features_v2.py — V2 Unified Multi-Asset Feature Builder
===============================================================
Builds features for ANY asset at ANY timeframe.
Pipeline: OHLCV → V1 base features → V2 layers → V2 crosses → sparse output.

Usage:
  python build_features_v2.py --symbol BTC --tf 1d
  python build_features_v2.py --symbol SPY --tf 1d
  python build_features_v2.py --all-daily          # All 31 assets at 1D
  python build_features_v2.py --all-crypto-1h      # All 14 crypto at 1H
  python build_features_v2.py --symbol BTC --tf 1h 4h 1d
"""

import os, sys, time, argparse, warnings, gc, multiprocessing

os.environ['PYTHONUNBUFFERED'] = '1'
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

warnings.filterwarnings('ignore')

# ── CUDA 13 detection (must run BEFORE any CuPy GPU calls) ──
# feature_library.py sets V2_SKIP_GPU=1 on CUDA 13+, but it's imported lazily
# inside build_base_features(). GPU ops in _gpu_gc() and build_single_asset()
# run BEFORE that import, so we need early detection here too.
if os.environ.get('V2_SKIP_GPU') != '1':
    try:
        import subprocess as _sp
        _nv = _sp.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                       capture_output=True, text=True, timeout=5)
        _drv = int(_nv.stdout.strip().split('.')[0])
        if _drv >= 580:  # CUDA 13.0+ — RAPIDS/CuPy GPU ops segfault
            os.environ['V2_SKIP_GPU'] = '1'
            print(f"[build_features_v2] GPU DISABLED — driver {_drv} (CUDA 13+). Using CPU mode.")
    except Exception:
        pass  # nvidia-smi not available or parse error — assume CUDA 12 compatible

# Force CuPy memory pool cleanup helper
def _gpu_gc():
    """Release GPU memory pool + Python GC."""
    gc.collect()
    if os.environ.get('V2_SKIP_GPU') == '1':
        return  # CUDA 13+ — CuPy GPU ops segfault, skip entirely
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass

V2_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, V2_DIR)

from config import (ALL_TRAINING, TRAINING_CRYPTO,
                    TIMEFRAMES_ALL_ASSETS, TIMEFRAMES_CRYPTO_ONLY)
from data_access_v2 import V2OfflineDataLoader
from v2_feature_layers import add_all_v2_layers
from v2_cross_generator import generate_all_crosses


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_base_features(ohlcv_df, symbol, tf, astro_cache, tweets=None, news=None,
                        sports=None, onchain=None, macro=None, htf_data=None):
    """
    Build base features using v2/feature_library.py.
    ALL assets get ALL esoteric data — the matrix is universal.
    Returns DataFrame with ~1000-3000 base features.
    """
    from feature_library import build_all_features

    esoteric_frames = {}
    if tweets is not None and not tweets.empty:
        esoteric_frames['tweets'] = tweets
    if news is not None and not news.empty:
        esoteric_frames['news'] = news
    if sports is not None:
        # feature_library expects esoteric_frames['sports'] as dict with 'games'/'horse_races' keys
        if isinstance(sports, dict):
            esoteric_frames['sports'] = sports
        else:
            esoteric_frames['sports'] = sports  # DataFrame fallback — feature_library handles both
    if onchain is not None:
        # feature_library expects esoteric_frames['onchain'] as dict with 'daily'/'timestamped' keys
        if isinstance(onchain, dict):
            esoteric_frames['onchain'] = onchain
        else:
            esoteric_frames['onchain'] = onchain  # DataFrame fallback — feature_library handles both
    if macro is not None and not macro.empty:
        esoteric_frames['macro'] = macro

    df = build_all_features(
        ohlcv=ohlcv_df,
        esoteric_frames=esoteric_frames,
        tf_name=tf,
        htf_data=htf_data or {},
        astro_cache=astro_cache,
        space_weather_df=astro_cache.get('space_weather'),
    )
    return df



def build_single_asset(symbol, tf, loader, save=True, max_crosses=None, force=False, gpu_id=0):
    """Build all features for a single asset + timeframe."""
    # CUDA_VISIBLE_DEVICES is set per-subprocess by cloud runner (round-robin).
    # It remaps so the subprocess always sees its assigned GPU as device 0.
    # When called in-process (mini_train), gpu_id param is used directly.
    # CUDA 13+ (driver 580+): skip GPU device selection — RAPIDS/CuPy segfault.
    if os.environ.get('V2_SKIP_GPU') != '1':
        if os.environ.get('CUDA_VISIBLE_DEVICES'):
            _gpu_id = 0  # remapped by env var
        else:
            _gpu_id = gpu_id  # direct assignment
        try:
            import cupy as cp
            cp.cuda.Device(_gpu_id).use()
        except Exception:
            pass

    t0 = time.time()
    log(f"\n{'='*60}")
    log(f"Building {symbol} — {tf}")
    log(f"{'='*60}")

    # Load data
    log("  Loading data...")
    data = loader.load_all_data_for_asset(symbol, tf)
    ohlcv = data['ohlcv']

    if ohlcv.empty:
        log(f"  [SKIP] No OHLCV data for {symbol} {tf}")
        return None

    # Checkpoint: skip if already built
    out_path = os.path.join(V2_DIR, f'features_{symbol}_{tf}.parquet')
    sparse_path = os.path.join(V2_DIR, f'v2_crosses_{symbol}_{tf}.npz')
    if not force and os.path.exists(out_path) and os.path.exists(sparse_path):
        log(f"  [SKIP] Already built: {out_path}")
        return None

    log(f"  OHLCV: {len(ohlcv):,} bars, {ohlcv.index[0].date()} to {ohlcv.index[-1].date()}")

    # Step 1: Base features (full V1 pipeline — NO fallback, NO TA-only mode)
    log("  Step 1: Base features...")
    # ALL assets get the full pipeline — the matrix is universal.
    # Same sky, same tweets, same energy, same calendar = same features.
    # Only TA differs (computed from each asset's OHLCV).
    # Only on-chain/funding is crypto-specific.
    df = build_base_features(
        ohlcv, symbol, tf, data['astro_cache'],
        tweets=data.get('tweets'),
        news=data.get('news'),
        sports=data.get('sports'),
        onchain=data.get('onchain'),
        macro=data.get('macro'),
        htf_data=data.get('htf_data'),
    )

    # Deduplicate columns (can happen when cross sections overlap)
    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].tolist()
        log(f"  WARNING: {len(dupes)} duplicate columns removed: {dupes[:5]}...")
        df = df.loc[:, ~df.columns.duplicated()]
    log(f"  Base features: {len(df.columns):,} cols")

    # Step 2: V2 feature layers (all 20 new layers)
    log("  Step 2: V2 feature layers...")
    ephemeris_df = data['astro_cache'].get('ephemeris')
    inverse_signals = data['astro_cache'].get('inverse_signals')

    df = add_all_v2_layers(
        df, symbol=symbol, tf=tf,
        astro_cache=data['astro_cache'],
        inverse_signals=inverse_signals,
    )
    # Verify no duplicate columns (root-caused: V2 layers use distinct names)
    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Unexpected duplicate columns after V2 layers: {dupes[:10]}")
    log(f"  After V2 layers: {len(df.columns):,} cols")

    # Step 3: Cross generation (everything × everything → sparse)
    log("  Step 3: Cross generation...")
    df._v2_symbol = symbol  # Tag for per-symbol sparse output naming
    _gpu_id = int(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0])
    df = generate_all_crosses(df, tf=tf, gpu_id=_gpu_id, save_sparse=True, output_dir=V2_DIR,
                              max_crosses=max_crosses)
    # Note: crosses saved as sparse .npz separately (too big for dense DataFrame)
    # df only contains base + V2 layer features

    # Save base features (atomic: temp file → rename)
    if save:
        from atomic_io import atomic_save_parquet
        out_path = os.path.join(V2_DIR, f'features_{symbol}_{tf}.parquet')
        atomic_save_parquet(df, out_path)
        size_mb = os.path.getsize(out_path) / 1e6
        log(f"  Saved base: {out_path} ({size_mb:.1f} MB)")

    elapsed = time.time() - t0
    log(f"  DONE: {symbol} {tf} — {len(df):,} rows × {len(df.columns):,} cols ({elapsed:.0f}s)")

    # Force memory cleanup between assets (GPU + CPU)
    del df
    _gpu_gc()

    return None  # Data is saved to files, don't hold in memory


def build_worker(args_tuple):
    """Worker for builds. Each asset is independent.
    GPU routed via CUDA_VISIBLE_DEVICES environment variable."""
    symbol, tf = args_tuple[0], args_tuple[1]
    try:
        loader = V2OfflineDataLoader()
        build_single_asset(symbol, tf, loader)
        return (symbol, tf, 'OK')
    except Exception as e:
        log(f"  [FAILED] {symbol} {tf}: {e}")
        return (symbol, tf, f'FAILED: {e}')


def _parallel_build(tasks, max_workers, label='builds'):
    """Run build tasks in parallel with ProcessPoolExecutor.
    Tasks are (symbol, tf) or (symbol, tf, gpu_id) tuples."""
    from concurrent.futures import ProcessPoolExecutor
    results = []
    log(f"  Launching {len(tasks)} {label} with {max_workers} workers...")
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for r in pool.map(build_worker, tasks):
            results.append(r)
            log(f"  {r[0]} {r[1]}: {r[2]}")
    return results


def main():
    multiprocessing.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser(description='V2 Multi-Asset Feature Builder')
    parser.add_argument('--symbol', nargs='+', help='Asset symbol(s)')
    parser.add_argument('--tf', nargs='+', default=['1d'], help='Timeframe(s)')
    parser.add_argument('--all-daily', action='store_true', help='All 31 assets at 1D')
    parser.add_argument('--all-crypto-1h', action='store_true', help='All crypto at 1H')
    parser.add_argument('--all-crypto-intraday', action='store_true', help='All crypto at all intraday TFs')
    parser.add_argument('--parallel', type=int, default=0,
                        help='Number of parallel builds (0 = auto-detect GPUs)')
    parser.add_argument('--full', action='store_true', help='Full build: all daily + crypto intraday')
    args = parser.parse_args()

    if args.full:
        # FULL BUILD — parallel workers, all GPUs visible per process, fault-tolerant
        log("=" * 70)
        log("V2 FULL BUILD — ALL ASSETS, ALL TIMEFRAMES")
        log("  Strategy: parallel builds (ProcessPoolExecutor), ALL GPUs visible per worker")
        log("=" * 70)

        from hardware_detect import detect_hardware, log_hardware
        hw = detect_hardware()
        log_hardware(hw)
        n_gpus = hw['n_gpus'] or 1
        max_workers = min(n_gpus, 4)
        log(f"  Parallel workers: {max_workers} (N_GPUS={n_gpus}, capped at 4)")

        results = []
        loader = V2OfflineDataLoader()

        # Step 1.1: BTC 1d first (catches pipeline errors early)
        log("\n--- Step 1.1: BTC 1d (pipeline test) ---")
        build_single_asset('BTC', '1d', loader)

        # Step 1.2: Remaining 30 daily assets (parallel, all GPUs visible)
        log(f"\n--- Step 1.2: 30 daily assets ({max_workers} parallel workers) ---")
        daily_tasks = [(symbol, '1d') for symbol in ALL_TRAINING if symbol != 'BTC']
        results.extend(_parallel_build(daily_tasks, max_workers, label='daily builds'))
        gc.collect()

        # GC between TF phases — flush accumulated RAM
        log("\n  [GC] Flushing memory before intraday builds...")
        del loader
        _gpu_gc()
        loader = V2OfflineDataLoader()

        # Step 1.3: Crypto 4h (parallel, all GPUs visible)
        log(f"\n--- Step 1.3: 14 crypto 4h ({max_workers} parallel workers) ---")
        crypto_4h_tasks = [(symbol, '4h') for symbol in TRAINING_CRYPTO]
        results.extend(_parallel_build(crypto_4h_tasks, max_workers, label='crypto 4h builds'))
        gc.collect()

        # GC between phases
        log("\n  [GC] Flushing memory before 1h builds...")
        del loader
        _gpu_gc()
        loader = V2OfflineDataLoader()

        # Step 1.4: Crypto 1h (parallel, all GPUs visible)
        log(f"\n--- Step 1.4: 14 crypto 1h ({max_workers} parallel workers) ---")
        crypto_1h_tasks = [(symbol, '1h') for symbol in TRAINING_CRYPTO]
        results.extend(_parallel_build(crypto_1h_tasks, max_workers, label='crypto 1h builds'))

        # GC before large TFs
        log("\n  [GC] Flushing memory before 15m build...")
        del loader
        _gpu_gc()
        loader = V2OfflineDataLoader()

        # Step 1.5: BTC 15m (all GPUs, high RAM)
        for tf in ['15m']:
            log(f"\n--- Step 1.5/6: BTC {tf} (all GPUs) ---")
            try:
                build_single_asset('BTC', tf, loader)
                results.append(('BTC', tf, 'OK'))
            except Exception as e:
                log(f"  [FAILED] BTC {tf}: {e}")
                results.append(('BTC', tf, f'FAILED: {e}'))
            # GC between TF builds
            _gpu_gc()

        # Summary
        log("\n" + "=" * 70)
        log("BUILD SUMMARY")
        log("=" * 70)
        ok = sum(1 for r in results if r[2] == 'OK')
        failed = [r for r in results if r[2] != 'OK']
        log(f"  OK: {ok}, Failed: {len(failed)}")
        for r in failed:
            log(f"  FAILED: {r[0]} {r[1]} — {r[2]}")

        # Verify outputs
        import glob
        parquets = glob.glob(os.path.join(V2_DIR, 'features_*_*.parquet'))
        sparse_files = glob.glob(os.path.join(V2_DIR, 'v2_crosses_*_*.npz'))
        log(f"\n  Parquet files: {len(parquets)}")
        log(f"  Sparse files: {len(sparse_files)}")

    elif args.all_daily:
        from hardware_detect import detect_hardware
        hw = detect_hardware()
        n_gpus = hw['n_gpus'] or 1
        max_workers = min(n_gpus, 4)
        log(f"Building ALL {len(ALL_TRAINING)} assets at 1D ({max_workers} parallel workers)...")
        tasks = [(symbol, '1d') for symbol in ALL_TRAINING]
        _parallel_build(tasks, max_workers, label='daily builds')

    elif args.all_crypto_1h:
        from hardware_detect import detect_hardware
        hw = detect_hardware()
        n_gpus = hw['n_gpus'] or 1
        max_workers = min(n_gpus, 4)
        log(f"Building ALL {len(TRAINING_CRYPTO)} crypto at 1H ({max_workers} parallel workers)...")
        tasks = [(symbol, '1h') for symbol in TRAINING_CRYPTO]
        _parallel_build(tasks, max_workers, label='crypto 1h builds')

    elif args.all_crypto_intraday:
        from hardware_detect import detect_hardware
        hw = detect_hardware()
        n_gpus = hw['n_gpus'] or 1
        max_workers = min(n_gpus, 4)
        log(f"Building ALL crypto at all intraday TFs ({max_workers} parallel workers)...")
        for tf in TIMEFRAMES_CRYPTO_ONLY:
            tasks = [(symbol, tf) for symbol in TRAINING_CRYPTO]
            _parallel_build(tasks, max_workers, label=f'crypto {tf} builds')
            gc.collect()

    elif args.symbol:
        loader = V2OfflineDataLoader()
        for symbol in args.symbol:
            for tf in args.tf:
                build_single_asset(symbol, tf, loader)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
