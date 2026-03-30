#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gcp_feature_builder.py — Build GCP (Global Consciousness Project) features
==========================================================================
Reads the cached hourly GCP stats from heartbeat_data/results/gcp_hourly_cache.json
and produces a DataFrame aligned to any OHLCV DatetimeIndex with rolling features.

Features produced:
  - gcp_deviation_mean (1h raw deviation)
  - gcp_deviation_max (1h raw max_var)
  - gcp_deviation_std (1h raw std_var)
  - gcp_deviation_4h (rolling 4h mean deviation)
  - gcp_deviation_24h (rolling 24h mean deviation)
  - gcp_max_var_4h (rolling 4h max of max_var)
  - gcp_max_var_24h (rolling 24h max of max_var)
  - gcp_rate_of_change (deviation change from previous hour)
  - gcp_extreme (top 1% deviation flag)
"""

import os
import json
import numpy as np
import pandas as pd


_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'heartbeat_data', 'results', 'gcp_hourly_cache.json'
)


def load_gcp_cache(path=None):
    """Load GCP hourly cache and return a DatetimeIndex DataFrame."""
    path = path or _CACHE_PATH
    if not os.path.exists(path):
        print(f"  [GCP] Cache not found: {path}")
        return pd.DataFrame()

    with open(path, 'r') as f:
        data = json.load(f)

    rows = []
    for ts_str, vals in data.items():
        rows.append({
            'timestamp': pd.Timestamp(int(ts_str), unit='s', tz='UTC'),
            'gcp_deviation_mean': vals.get('deviation', np.nan),
            'gcp_max_var': vals.get('max_var', np.nan),
            'gcp_std_var': vals.get('std_var', np.nan),
            'gcp_n_samples': vals.get('n_samples', 0),
        })

    df = pd.DataFrame(rows).set_index('timestamp').sort_index()
    return df


def build_gcp_features(ohlcv_df, gcp_cache_path=None):
    """
    Build GCP features aligned to an OHLCV DataFrame's DatetimeIndex.

    Args:
        ohlcv_df: DataFrame with DatetimeIndex (1h candles expected)
        gcp_cache_path: optional override path to gcp_hourly_cache.json

    Returns:
        DataFrame with GCP features, same index as ohlcv_df
    """
    gcp = load_gcp_cache(gcp_cache_path)
    out = pd.DataFrame(index=ohlcv_df.index)

    if gcp.empty:
        print("  [GCP] No data — returning NaN features")
        for col in ['gcp_deviation_mean', 'gcp_deviation_max', 'gcp_deviation_std',
                     'gcp_deviation_4h', 'gcp_deviation_24h',
                     'gcp_max_var_4h', 'gcp_max_var_24h',
                     'gcp_rate_of_change', 'gcp_extreme']:
            out[col] = np.nan
        return out

    # Align GCP to OHLCV index via nearest-hour merge
    # GCP is hourly, OHLCV should be hourly too
    dev = gcp['gcp_deviation_mean']
    max_var = gcp['gcp_max_var']
    std_var = gcp['gcp_std_var']

    # Reindex to match OHLCV
    out['gcp_deviation_mean'] = dev.reindex(ohlcv_df.index, method='nearest', tolerance='2h')
    out['gcp_deviation_max'] = max_var.reindex(ohlcv_df.index, method='nearest', tolerance='2h')
    out['gcp_deviation_std'] = std_var.reindex(ohlcv_df.index, method='nearest', tolerance='2h')

    # Rolling windows
    out['gcp_deviation_4h'] = out['gcp_deviation_mean'].rolling(4, min_periods=1).mean()
    out['gcp_deviation_24h'] = out['gcp_deviation_mean'].rolling(24, min_periods=1).mean()
    out['gcp_max_var_4h'] = out['gcp_deviation_max'].rolling(4, min_periods=1).max()
    out['gcp_max_var_24h'] = out['gcp_deviation_max'].rolling(24, min_periods=1).max()

    # Rate of change
    out['gcp_rate_of_change'] = out['gcp_deviation_mean'] - out['gcp_deviation_mean'].shift(1)

    # Extreme flag (top 1% by absolute deviation)
    abs_dev = out['gcp_deviation_mean'].abs()
    threshold = abs_dev.quantile(0.99)
    out['gcp_extreme'] = (abs_dev > threshold).astype(int) if threshold > 0 else 0

    print(f"  [GCP] Features built: {out.notna().sum().sum()} non-NaN values across {len(out.columns)} columns")
    non_null = out['gcp_deviation_mean'].notna().sum()
    print(f"  [GCP] Coverage: {non_null}/{len(out)} rows ({non_null/len(out)*100:.1f}%)")

    return out


if __name__ == "__main__":
    print("Testing GCP feature builder...")
    gcp = load_gcp_cache()
    print(f"Loaded {len(gcp)} hourly GCP records")
    print(f"Date range: {gcp.index.min()} to {gcp.index.max()}")
    print(f"Columns: {list(gcp.columns)}")
    print(f"\nSample:\n{gcp.head()}")
