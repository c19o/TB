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

import os, sys, time, argparse, warnings, gc
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings('ignore')

# Force CuPy memory pool cleanup helper
def _gpu_gc():
    """Release GPU memory pool + Python GC."""
    gc.collect()
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass

V2_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, V2_DIR)

from config import (ALL_TRAINING, TRAINING_CRYPTO, TRAINING_STOCKS,
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
        if isinstance(sports, dict):
            if not sports.get('games', pd.DataFrame()).empty:
                esoteric_frames['sports_games'] = sports['games']
            if not sports.get('horse_races', pd.DataFrame()).empty:
                esoteric_frames['sports_horses'] = sports['horse_races']
    if onchain is not None:
        if isinstance(onchain, dict):
            if not onchain.get('daily', pd.DataFrame()).empty:
                esoteric_frames['onchain_daily'] = onchain['daily']
            if not onchain.get('timestamped', pd.DataFrame()).empty:
                esoteric_frames['onchain'] = onchain['timestamped']
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



def build_single_asset(symbol, tf, loader, save=True):
    """Build all features for a single asset + timeframe."""
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
    if os.path.exists(out_path) and os.path.exists(sparse_path):
        log(f"  [SKIP] Already built: {out_path}")
        return None

    log(f"  OHLCV: {len(ohlcv):,} bars, {ohlcv.index[0].date()} to {ohlcv.index[-1].date()}")

    # Step 1: Base features (V1 pipeline or TA-only fallback)
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
    log(f"  After V2 layers: {len(df.columns):,} cols")

    # Step 3: Cross generation (everything × everything → sparse)
    log("  Step 3: Cross generation...")
    df._v2_symbol = symbol  # Tag for per-symbol sparse output naming
    df = generate_all_crosses(df, tf=tf, save_sparse=True, output_dir=V2_DIR)
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
    ALL GPUs visible — no CUDA_VISIBLE_DEVICES pinning."""
    symbol, tf = args_tuple[0], args_tuple[1]
    try:
        loader = V2OfflineDataLoader()
        build_single_asset(symbol, tf, loader)
        return (symbol, tf, 'OK')
    except Exception as e:
        log(f"  [FAILED] {symbol} {tf}: {e}")
        return (symbol, tf, f'FAILED: {e}')


def _detect_gpu_count():
    """Detect number of available GPUs for parallel routing."""
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount()
    except Exception:
        return 0


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
        log("\n  [GC] Flushing memory before 15m/5m builds...")
        del loader
        _gpu_gc()
        loader = V2OfflineDataLoader()

        # Step 1.5-1.6: BTC 15m + 5m (sequential, all GPUs, high RAM)
        for tf in ['15m', '5m']:
            log(f"\n--- Step 1.5/6: BTC {tf} (all GPUs) ---")
            try:
                build_single_asset('BTC', tf, loader)
                results.append(('BTC', tf, 'OK'))
            except Exception as e:
                log(f"  [FAILED] BTC {tf}: {e}")
                results.append(('BTC', tf, f'FAILED: {e}'))
            # GC between 15m and 5m
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
