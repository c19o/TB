#!/usr/bin/env python
"""verify_gpu_features.py — Verify GPU feature output matches CPU

Builds features for 4H timeframe using both GPU and CPU paths,
then compares every column to ensure GPU acceleration doesn't
introduce numerical drift beyond tolerance (1e-6).
"""
import sys
import os
import time
import warnings
import importlib
import traceback

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import sqlite3

DB_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_data():
    """Load 4H OHLCV data (smallest meaningful TF, ~14K rows)."""
    db_path = os.path.join(DB_DIR, 'btc_prices.db')
    if not os.path.exists(db_path):
        print(f"  ERROR: {db_path} not found")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    try:
        ohlcv = pd.read_sql_query("SELECT * FROM btc_4h ORDER BY timestamp", conn)
    except Exception as e:
        print(f"  ERROR reading btc_4h table: {e}")
        sys.exit(1)
    finally:
        conn.close()

    if ohlcv.empty:
        print("  ERROR: btc_4h table is empty")
        sys.exit(1)

    ohlcv['timestamp'] = pd.to_datetime(ohlcv['timestamp'])
    ohlcv = ohlcv.set_index('timestamp')
    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c in ohlcv.columns:
            ohlcv[c] = pd.to_numeric(ohlcv[c], errors='coerce')
        else:
            print(f"  WARNING: column '{c}' missing from btc_4h")
    return ohlcv


def load_aux_data():
    """Load auxiliary data needed by feature_library (tweets, news, etc.)."""
    aux = {}

    # Simple single-table sources
    simple_sources = {
        'tweets':  ('tweets.db',         'tweets'),
        'news':    ('news_articles.db',   'news_articles'),
        'onchain': ('onchain_data.db',    'onchain_data'),
        'macro':   ('macro_data.db',      'macro_data'),
    }
    for key, (db_file, table) in simple_sources.items():
        path = os.path.join(DB_DIR, db_file)
        if not os.path.exists(path):
            print(f"  INFO: {db_file} not found, skipping '{key}'")
            continue
        try:
            conn = sqlite3.connect(path)
            aux[key] = pd.read_sql_query(f"SELECT * FROM {table}", conn)
            conn.close()
            print(f"  Loaded {key}: {len(aux[key])} rows")
        except Exception as e:
            print(f"  WARNING: failed to load {key} from {db_file}: {e}")

    # Sports: expects {'games': df, 'horse_races': df}
    sports_path = os.path.join(DB_DIR, 'sports_results.db')
    if os.path.exists(sports_path):
        try:
            conn = sqlite3.connect(sports_path)
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            sports = {}
            if 'games' in tables:
                sports['games'] = pd.read_sql_query("SELECT * FROM games", conn)
            if 'horse_races' in tables:
                sports['horse_races'] = pd.read_sql_query("SELECT * FROM horse_races", conn)
            conn.close()
            if sports:
                aux['sports'] = sports
                print(f"  Loaded sports: {list(sports.keys())}")
        except Exception as e:
            print(f"  WARNING: failed to load sports: {e}")
    else:
        print(f"  INFO: sports_results.db not found, skipping 'sports'")

    return aux


def load_astro_cache():
    """Load astrology/ephemeris/fear-greed/trends caches for feature_library."""
    cache = {}

    sources = {
        'astrology':      ('astrology_full.db',   'daily_astrology'),
        'ephemeris':      ('ephemeris_cache.db',   'ephemeris'),
        'fear_greed':     ('fear_greed.db',        'fear_greed'),
        'google_trends':  ('google_trends.db',     'google_trends'),
    }
    for key, (db_file, table) in sources.items():
        path = os.path.join(DB_DIR, db_file)
        if not os.path.exists(path):
            continue
        try:
            conn = sqlite3.connect(path)
            # Check table exists
            exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table,)).fetchone()
            if exists:
                cache[key] = pd.read_sql_query(f"SELECT * FROM {table}", conn)
                print(f"  Loaded astro/{key}: {len(cache[key])} rows")
            conn.close()
        except Exception as e:
            print(f"  WARNING: failed to load {key}: {e}")

    # Funding daily from funding_rates.db
    fr_path = os.path.join(DB_DIR, 'funding_rates.db')
    if os.path.exists(fr_path):
        try:
            conn = sqlite3.connect(fr_path)
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            for t in ['funding_daily', 'funding_rates']:
                if t in tables:
                    cache['funding_daily'] = pd.read_sql_query(f"SELECT * FROM {t}", conn)
                    print(f"  Loaded funding_daily from {t}: {len(cache['funding_daily'])} rows")
                    break
            conn.close()
        except Exception as e:
            print(f"  WARNING: failed to load funding: {e}")

    return cache if cache else None


def load_space_weather():
    """Load space weather data (Kp index, solar flux, etc.)."""
    # Try space_weather.db first
    sw_path = os.path.join(DB_DIR, 'space_weather.db')
    if os.path.exists(sw_path):
        try:
            conn = sqlite3.connect(sw_path)
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            for t in ['space_weather', 'kp_data', 'solar_data']:
                if t in tables:
                    df = pd.read_sql_query(f"SELECT * FROM {t}", conn)
                    conn.close()
                    if not df.empty:
                        # Try to set date index
                        for col in ['date', 'timestamp', 'datetime']:
                            if col in df.columns:
                                df[col] = pd.to_datetime(df[col], errors='coerce')
                                df = df.set_index(col)
                                break
                        print(f"  Loaded space_weather from {t}: {len(df)} rows")
                        return df
            conn.close()
        except Exception as e:
            print(f"  WARNING: failed to load space_weather.db: {e}")

    # Fallback: kp_history_gfz.txt
    kp_path = os.path.join(DB_DIR, 'kp_history_gfz.txt')
    if not os.path.exists(kp_path):
        print("  INFO: No space weather data found")
        return None

    rows = []
    try:
        with open(kp_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        rows.append({
                            'date': pd.to_datetime(parts[0]),
                            'kp_index': float(parts[1]),
                        })
                    except (ValueError, TypeError):
                        pass
    except Exception as e:
        print(f"  WARNING: failed to read kp_history_gfz.txt: {e}")
        return None

    if rows:
        df = pd.DataFrame(rows).set_index('date')
        print(f"  Loaded kp_history_gfz.txt: {len(df)} rows")
        return df
    return None


# ---------------------------------------------------------------------------
# Feature building
# ---------------------------------------------------------------------------

def build_features(ohlcv, aux, space_weather, astro_cache, use_gpu):
    """Build features with GPU enabled or forced off."""
    # Fresh import each time so _HAS_GPU can be overridden cleanly
    import feature_library as fl
    importlib.reload(fl)

    if not use_gpu:
        fl._HAS_GPU = False
        print("  [CPU mode] _HAS_GPU forced to False")
    else:
        print(f"  [GPU mode] _HAS_GPU = {fl._HAS_GPU}")
        if not fl._HAS_GPU:
            print("  WARNING: CuPy not available — GPU build will also run CPU path")
            print("  The comparison will still verify code-path equivalence.")

    result = fl.build_all_features(
        ohlcv=ohlcv.copy(),
        esoteric_frames=aux,
        tf_name='4h',
        mode='backfill',
        space_weather_df=space_weather.copy() if space_weather is not None else None,
        astro_cache={k: v.copy() for k, v in astro_cache.items()} if astro_cache else None,
        include_targets=False,
        include_knn=False,  # Skip KNN for speed
    )
    return result


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

def compare(cpu_df, gpu_df, tol=1e-6):
    """Compare two DataFrames column by column.

    Returns True if all columns match within tolerance.
    """
    print(f"\n{'='*60}")
    print("VERIFICATION RESULTS")
    print(f"{'='*60}")

    # Align row counts (take intersection of indices)
    common_idx = cpu_df.index.intersection(gpu_df.index)
    if len(common_idx) < len(cpu_df) or len(common_idx) < len(gpu_df):
        print(f"  WARNING: row count differs — CPU={len(cpu_df)}, GPU={len(gpu_df)}")
        print(f"           using {len(common_idx)} common rows for comparison")
        cpu_df = cpu_df.loc[common_idx]
        gpu_df = gpu_df.loc[common_idx]

    cpu_cols = set(cpu_df.columns)
    gpu_cols = set(gpu_df.columns)
    only_cpu = sorted(cpu_cols - gpu_cols)
    only_gpu = sorted(gpu_cols - cpu_cols)
    common = sorted(cpu_cols & gpu_cols)

    print(f"  CPU columns:    {len(cpu_cols)}")
    print(f"  GPU columns:    {len(gpu_cols)}")
    print(f"  Common columns: {len(common)}")
    print(f"  Rows compared:  {len(cpu_df)}")

    if only_cpu:
        shown = only_cpu[:10]
        extra = f"... (+{len(only_cpu)-10} more)" if len(only_cpu) > 10 else ""
        print(f"  Only in CPU ({len(only_cpu)}): {shown}{extra}")
    if only_gpu:
        shown = only_gpu[:10]
        extra = f"... (+{len(only_gpu)-10} more)" if len(only_gpu) > 10 else ""
        print(f"  Only in GPU ({len(only_gpu)}): {shown}{extra}")

    mismatches = []
    nan_mismatches = []
    non_numeric = []
    matches = 0

    for col in common:
        # Try numeric conversion
        try:
            cpu_vals = cpu_df[col].values.astype(np.float64)
            gpu_vals = gpu_df[col].values.astype(np.float64)
        except (ValueError, TypeError):
            # Non-numeric column — exact string comparison
            if cpu_df[col].astype(str).equals(gpu_df[col].astype(str)):
                matches += 1
            else:
                non_numeric.append(col)
            continue

        # NaN position check
        cpu_nan = np.isnan(cpu_vals)
        gpu_nan = np.isnan(gpu_vals)
        nan_pos_match = np.array_equal(cpu_nan, gpu_nan)

        if not nan_pos_match:
            cpu_nan_count = int(np.sum(cpu_nan))
            gpu_nan_count = int(np.sum(gpu_nan))
            nan_mismatches.append((col, cpu_nan_count, gpu_nan_count))
            continue

        # Compare non-NaN values only
        valid = ~cpu_nan
        if not np.any(valid):
            # All NaN in both — that's a match
            matches += 1
            continue

        cpu_valid = cpu_vals[valid]
        gpu_valid = gpu_vals[valid]
        max_diff = np.max(np.abs(cpu_valid - gpu_valid))

        if max_diff > tol:
            mismatches.append((col, float(max_diff)))
        else:
            matches += 1

    # Report
    print(f"\n  Matching columns:   {matches}/{len(common)}")

    if nan_mismatches:
        print(f"  NaN-position mismatches: {len(nan_mismatches)}")
        for col, cn, gn in sorted(nan_mismatches)[:15]:
            print(f"    {col}: CPU has {cn} NaNs, GPU has {gn} NaNs")
        if len(nan_mismatches) > 15:
            print(f"    ... and {len(nan_mismatches)-15} more")

    if non_numeric:
        print(f"  Non-numeric mismatches: {len(non_numeric)}")
        for col in non_numeric[:10]:
            print(f"    {col}")

    if mismatches:
        print(f"  Value mismatches (tol={tol}): {len(mismatches)}")
        for col, diff in sorted(mismatches, key=lambda x: -x[1])[:20]:
            print(f"    {col}: max_diff={diff:.10f}")
        if len(mismatches) > 20:
            print(f"    ... and {len(mismatches)-20} more")
    else:
        if not nan_mismatches and not non_numeric:
            print("  ALL COLUMNS MATCH within tolerance!")

    total_issues = len(only_cpu) + len(only_gpu) + len(mismatches) + len(nan_mismatches) + len(non_numeric)
    if total_issues == 0:
        print(f"\n  VERIFICATION PASSED")
        return True
    else:
        print(f"\n  VERIFICATION FAILED — {total_issues} issue(s) found")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("GPU vs CPU Feature Verification")
    print("=" * 60)

    # Pre-flight: check feature_library is importable
    try:
        import feature_library as fl
        print(f"  feature_library loaded, _HAS_GPU={fl._HAS_GPU}")
    except ImportError as e:
        print(f"  FATAL: Cannot import feature_library: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"  FATAL: Error loading feature_library: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Load all data
    print("\nLoading data...")
    ohlcv = load_data()
    print(f"  OHLCV: {len(ohlcv)} rows, columns: {list(ohlcv.columns[:8])}")

    print("\nLoading auxiliary data...")
    aux = load_aux_data()

    print("\nLoading astro cache...")
    astro_cache = load_astro_cache()

    print("\nLoading space weather...")
    sw = load_space_weather()

    # -----------------------------------------------------------------------
    # CPU build (run first so GPU memory state doesn't affect it)
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("CPU BUILD")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        cpu_result = build_features(ohlcv, aux, sw, astro_cache, use_gpu=False)
    except Exception as e:
        print(f"  FATAL: CPU build failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    cpu_time = time.time() - t0
    print(f"  CPU result: {len(cpu_result)} rows x {len(cpu_result.columns)} cols in {cpu_time:.1f}s")

    # -----------------------------------------------------------------------
    # GPU build (reimport resets _HAS_GPU from CuPy detection)
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("GPU BUILD")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        gpu_result = build_features(ohlcv, aux, sw, astro_cache, use_gpu=True)
    except Exception as e:
        print(f"  FATAL: GPU build failed: {e}")
        traceback.print_exc()
        sys.exit(1)
    gpu_time = time.time() - t0
    print(f"  GPU result: {len(gpu_result)} rows x {len(gpu_result.columns)} cols in {gpu_time:.1f}s")

    if gpu_time > 0:
        print(f"\n  Speedup: {cpu_time / gpu_time:.2f}x")
    else:
        print(f"\n  Speedup: N/A (GPU time too small to measure)")

    # -----------------------------------------------------------------------
    # Compare
    # -----------------------------------------------------------------------
    passed = compare(cpu_result, gpu_result)
    sys.exit(0 if passed else 1)
