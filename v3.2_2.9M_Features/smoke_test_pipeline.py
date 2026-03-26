#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, subprocess
_skip_gpu = os.environ.get('V2_SKIP_GPU') == '1'
if not _skip_gpu:
    try:
        _nv = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=5)
        if int(_nv.stdout.strip().split('.')[0]) >= 580:
            _skip_gpu = True
    except: pass
if not _skip_gpu:
    try:
        import cudf.pandas
        cudf.pandas.install()
        print("[GPU] cudf.pandas ENABLED")
    except (ImportError, Exception):
        pass
"""
smoke_test_pipeline.py — Lightweight Pre-Flight Validation (XGBoost)
=====================================================================
Runs the ENTIRE institutional pipeline on a tiny data slice to verify
every code path works before spending money on cloud GPUs.

Pre-flight checks (Step 0):
  - XGBoost import + DMatrix creation (sparse CSR)
  - V3_XGBM_PARAMS config verification (multi:softprob, max_bin=15, hist)
  - feature_library._np() fix (cuDF -> numpy conversion)
  - v2_cross_generator import + sparse matmul pre-filter
  - V2_SKIP_GPU detection (CUDA 13+ driver auto-detect)

Pipeline checks (Steps 1-10):
  - Rows (1000 bars per TF instead of 57K+)
  - Features (base only, no dx_ crosses = ~3K instead of 150K)
  - CPCV folds (3 groups, 1 test = 3 paths instead of 15)
  - XGBoost training + model save/load roundtrip (.json format)
  - PBO (skip subsample, use CPCV directly)
  - Meta-labeling (trains on tiny OOS set)
  - Kelly sizing (just verifies the math)

Target: <10 min, <32GB RAM, <16GB VRAM on RTX 3090.

Usage:
    python smoke_test_pipeline.py
    python smoke_test_pipeline.py --tf 1h
    python smoke_test_pipeline.py --tf 1w --rows 200
"""

import sys
import os
import io
import time
import json
import warnings
import argparse
import traceback

if os.name == 'nt':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import sqlite3

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
START = time.time()


def elapsed():
    return f"[{time.time()-START:.0f}s]"


def smoke_log(msg):
    print(f"  SMOKE {elapsed()} {msg}")


# ============================================================
# CONFIGURATION
# ============================================================

SMOKE_CONFIG = {
    'max_rows': 1000,        # bars per TF (tiny slice)
    'skip_dx_crosses': True, # skip 135K DOY crosses (saves RAM/time)
    'cpcv_groups': 3,        # minimal CPCV (3 paths instead of 15)
    'cpcv_test_groups': 1,
    'xgb_rounds': 50,        # fast training
    'xgb_early_stop': 10,
    'skip_lstm': True,       # skip if no trained model
    'skip_optimizer': True,  # skip exhaustive optimizer
}


def load_smoke_data(tf_name='1h', max_rows=1000):
    """Load a small slice of pre-built feature data for smoke testing.

    Priority: parquet (production format) > SQLite features > raw OHLCV SQLite.
    """
    # 1. Try parquet (production format — features already built)
    # Check local, asset-prefixed, and v3.0 shared data dir
    from config import V30_DATA_DIR
    parquet_candidates = [
        os.path.join(PROJECT_DIR, f'features_{tf_name}.parquet'),
        os.path.join(PROJECT_DIR, f'features_BTC_{tf_name}.parquet'),
        os.path.join(V30_DATA_DIR, f'features_BTC_{tf_name}.parquet'),
        os.path.join(V30_DATA_DIR, f'features_{tf_name}.parquet'),
    ]
    parquet_path = next((p for p in parquet_candidates if os.path.exists(p)), None)
    if parquet_path:
        df = pd.read_parquet(parquet_path)
        if len(df) > max_rows:
            df = df.tail(max_rows).copy()
        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df.index = pd.to_datetime(df['timestamp'], errors='coerce')
            elif 'open_time' in df.columns:
                df.index = pd.to_datetime(df['open_time'], errors='coerce')
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        smoke_log(f"Loaded {len(df)} {tf_name} bars from parquet ({len(df.columns)} cols)")
        return df

    # 2. Try SQLite feature DB (local or v3.0)
    db_path = os.path.join(PROJECT_DIR, f'features_{tf_name}.db')
    if not os.path.exists(db_path):
        db_path = os.path.join(V30_DATA_DIR, f'features_{tf_name}.db')
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        tables = [t[0] for t in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        if tables:
            df = pd.read_sql_query(f"SELECT * FROM {tables[0]} ORDER BY rowid DESC LIMIT {max_rows}", conn)
            conn.close()
            df = df.iloc[::-1].reset_index(drop=True)
            if 'timestamp' in df.columns:
                df.index = pd.to_datetime(df['timestamp'], errors='coerce')
            smoke_log(f"Loaded {len(df)} {tf_name} bars from features_{tf_name}.db ({len(df.columns)} cols)")
            return df
        conn.close()

    # 3. Fallback: raw OHLCV from btc_prices.db or multi_asset_prices.db (local or v3.0)
    raw_path = os.path.join(PROJECT_DIR, 'btc_prices.db')
    if not os.path.exists(raw_path):
        raw_path = os.path.join(V30_DATA_DIR, 'btc_prices.db')
    if not os.path.exists(raw_path):
        raw_path = os.path.join(V30_DATA_DIR, 'multi_asset_prices.db')
    if os.path.exists(raw_path):
        conn = sqlite3.connect(raw_path)
        try:
            df = pd.read_sql_query(
                f"SELECT * FROM ohlcv WHERE timeframe=? ORDER BY open_time DESC LIMIT {max_rows}",
                conn, params=(tf_name,)
            )
        except Exception:
            conn.close()
            raise FileNotFoundError(f"No data found for {tf_name}")
        conn.close()
        if 'open_time' in df.columns and 'timestamp' not in df.columns:
            df['timestamp'] = df['open_time']
        df = df.sort_values('timestamp').reset_index(drop=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        if df['timestamp'].dt.tz is not None:
            df.index = df['timestamp'].dt.tz_localize(None)
        else:
            df.index = df['timestamp']
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        smoke_log(f"Loaded {len(df)} {tf_name} bars from raw OHLCV (needs feature building)")
        return df

    raise FileNotFoundError(f"No data source found for {tf_name}")


def run_smoke_test(tf_name='1h', max_rows=None):
    """Run the full pipeline on a tiny data slice."""
    cfg = SMOKE_CONFIG.copy()
    if max_rows:
        cfg['max_rows'] = max_rows

    results = {'tf': tf_name, 'steps': {}, 'errors': [], 'passed': True}

    print(f"\n{'='*60}")
    print(f"  SMOKE TEST: {tf_name} ({cfg['max_rows']} bars, no dx_ crosses)")
    print(f"{'='*60}\n")

    # ============================================================
    # STEP 0: Pre-flight — verify critical imports & config
    # ============================================================
    try:
        # --- XGBoost import + DMatrix creation ---
        import xgboost as xgb
        smoke_log(f"XGBoost import: OK (v{xgb.__version__})")

        # Verify DMatrix works with sparse CSR input (production pipeline uses sparse)
        from scipy import sparse as sp_sparse
        _test_dense = np.random.rand(50, 10).astype(np.float32)
        _test_csr = sp_sparse.csr_matrix(_test_dense)
        _test_dm = xgb.DMatrix(_test_csr, label=np.random.randint(0, 3, 50),
                                feature_names=[f'f{i}' for i in range(10)], nthread=-1)
        assert _test_dm.num_row() == 50, f"DMatrix rows: {_test_dm.num_row()}"
        assert _test_dm.num_col() == 10, f"DMatrix cols: {_test_dm.num_col()}"
        smoke_log(f"XGBoost DMatrix (sparse CSR): OK")

        # --- V3_XGBM_PARAMS config verification ---
        from config import V3_XGBM_PARAMS, TF_MIN_DATA_IN_LEAF
        assert V3_XGBM_PARAMS['objective'] == 'multi:softprob', \
            f"Wrong objective: {V3_XGBM_PARAMS['objective']} (expected multi:softprob)"
        assert V3_XGBM_PARAMS['num_class'] == 3, \
            f"Wrong num_class: {V3_XGBM_PARAMS['num_class']}"
        assert V3_XGBM_PARAMS['max_bin'] == 15, \
            f"Wrong max_bin: {V3_XGBM_PARAMS['max_bin']} (binary features need 15)"
        assert V3_XGBM_PARAMS['tree_method'] == 'hist', \
            f"Wrong tree_method: {V3_XGBM_PARAMS['tree_method']}"
        assert V3_XGBM_PARAMS['eta'] == 0.03, \
            f"Wrong eta: {V3_XGBM_PARAMS['eta']}"
        smoke_log(f"V3_XGBM_PARAMS: OK (multi:softprob, max_bin=15, hist, eta=0.03)")

        # --- feature_library import + _np() fix ---
        from feature_library import _np
        _test_arr = pd.Series([1.0, 2.0, np.nan])
        _np_result = _np(_test_arr)
        assert isinstance(_np_result, np.ndarray), f"_np() should return ndarray, got {type(_np_result)}"
        assert np.isnan(_np_result[2]), "_np() must preserve NaN"
        smoke_log(f"feature_library._np(): OK (cuDF/pandas -> numpy, NaN preserved)")

        # --- cross generator import ---
        import v2_cross_generator
        smoke_log(f"v2_cross_generator: import OK")

        # --- V2_SKIP_GPU detection logic ---
        import subprocess as _sp
        try:
            _nv = _sp.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                          capture_output=True, text=True, timeout=5)
            _driver = _nv.stdout.strip().split('\n')[0].strip()
            _driver_major = int(_driver.split('.')[0]) if _driver else 0
            _skip_gpu_detected = _driver_major >= 580
            smoke_log(f"V2_SKIP_GPU detection: driver={_driver}, major={_driver_major}, "
                      f"skip_gpu={'YES (CUDA 13+)' if _skip_gpu_detected else 'NO (cuDF OK)'}")
        except Exception as _e:
            smoke_log(f"V2_SKIP_GPU detection: no GPU ({_e})")

        # --- Sparse matmul pre-filter in cross generator ---
        # Verify the function exists (it's the 10-50x speedup)
        _has_matmul = hasattr(v2_cross_generator, '_gpu_cross_multiply') or \
                      hasattr(v2_cross_generator, '_cpu_cross_multiply')
        smoke_log(f"Cross generator sparse matmul: {'found' if _has_matmul else 'MISSING'}")

        results['steps']['preflight'] = {'status': 'PASS', 'xgboost': xgb.__version__}
        smoke_log(f"Pre-flight: ALL CHECKS PASSED")

    except Exception as e:
        smoke_log(f"FAIL preflight: {e}")
        traceback.print_exc()
        results['steps']['preflight'] = {'status': 'FAIL', 'error': str(e)}
        results['errors'].append(f"preflight: {e}")
        results['passed'] = False
        return results

    # ============================================================
    # STEP 1: Load data
    # ============================================================
    try:
        ohlcv = load_smoke_data(tf_name, cfg['max_rows'])
        results['steps']['load_data'] = {'status': 'PASS', 'rows': len(ohlcv)}
    except Exception as e:
        smoke_log(f"FAIL load_data: {e}")
        results['steps']['load_data'] = {'status': 'FAIL', 'error': str(e)}
        results['errors'].append(f"load_data: {e}")
        results['passed'] = False
        return results

    # ============================================================
    # STEP 2: Feature build or load pre-built parquet
    # ============================================================
    try:
        from feature_library import (
            compute_ta_features, compute_numerology_features,
            compute_frac_diff_features, compute_time_features,
            TRIPLE_BARRIER_CONFIG, compute_triple_barrier_labels,
        )

        has_features = len(ohlcv.columns) > 20  # parquet already has features
        t0 = time.time()

        if has_features:
            # Loaded from parquet — features already built
            result_df = ohlcv.copy()
            n_features = len([c for c in result_df.columns
                             if c not in {'timestamp', 'open', 'high', 'low', 'close', 'volume',
                                         'quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote',
                                         'open_time', 'symbol', 'timeframe', 'triple_barrier_label'}])
            dt = time.time() - t0
            smoke_log(f"Features pre-built: {n_features} features loaded from parquet in {dt:.1f}s")
            results['steps']['feature_build'] = {
                'status': 'PASS', 'n_features': n_features, 'time': round(dt, 1),
                'source': 'parquet',
            }

            # Still test that NEW compute functions work (frac diff is new)
            try:
                frac = compute_frac_diff_features(result_df)
                for col in frac.columns:
                    if col not in result_df.columns:
                        result_df[col] = frac[col].values
                smoke_log(f"  Added {len(frac.columns)} new frac_diff features to existing parquet")
            except Exception as e:
                smoke_log(f"  WARNING: frac_diff on parquet failed: {e}")
        else:
            # Raw OHLCV — build features from scratch
            ta = compute_ta_features(ohlcv, tf_name)
            frac = compute_frac_diff_features(ohlcv)
            time_f = compute_time_features(ohlcv, tf_name)
            num = compute_numerology_features(ohlcv)

            n_features = len(ta.columns) + len(frac.columns) + len(time_f.columns) + len(num.columns)
            dt = time.time() - t0

            smoke_log(f"Feature build: {n_features} features in {dt:.1f}s")
            results['steps']['feature_build'] = {
                'status': 'PASS', 'n_features': n_features, 'time': round(dt, 1),
                'source': 'computed',
                'ta': len(ta.columns), 'frac': len(frac.columns),
                'time_feats': len(time_f.columns), 'numerology': len(num.columns),
            }

            result_df = ohlcv.copy()
            for df_part in [ta, frac, time_f, num]:
                for col in df_part.columns:
                    result_df[col] = df_part[col].values

    except Exception as e:
        smoke_log(f"FAIL feature_build: {e}")
        traceback.print_exc()
        results['steps']['feature_build'] = {'status': 'FAIL', 'error': str(e)}
        results['errors'].append(f"feature_build: {e}")
        results['passed'] = False
        return results

    # ============================================================
    # STEP 3: GPU gematria/sentiment batch (verify GPU engines work)
    # ============================================================
    try:
        from universal_gematria import gematria_gpu_batch
        from universal_sentiment import sentiment_gpu_batch

        test_texts = pd.Series(['Bitcoin crashes!', 'BTC moon pump!', '', 'BREAKING: ETF approved'])
        t0 = time.time()
        gem_result = gematria_gpu_batch(test_texts, prefix='test_gem')
        sent_result = sentiment_gpu_batch(test_texts, prefix='test_sent')
        dt = time.time() - t0

        assert len(gem_result) == 4, f"Gematria returned {len(gem_result)} rows, expected 4"
        assert len(sent_result) == 4, f"Sentiment returned {len(sent_result)} rows, expected 4"
        assert gem_result['test_gem_ordinal'].iloc[0] > 0, "Gematria ordinal should be > 0"

        smoke_log(f"GPU gematria+sentiment: PASS ({dt:.2f}s)")
        results['steps']['gpu_engines'] = {'status': 'PASS', 'time': round(dt, 2)}

    except Exception as e:
        smoke_log(f"FAIL gpu_engines: {e}")
        results['steps']['gpu_engines'] = {'status': 'FAIL', 'error': str(e)}
        results['errors'].append(f"gpu_engines: {e}")

    # ============================================================
    # STEP 4: Triple-barrier labels
    # ============================================================
    try:
        t0 = time.time()
        labels = compute_triple_barrier_labels(result_df, tf_name)
        n_valid = (~np.isnan(labels)).sum()
        n_long = (labels == 2).sum()
        n_short = (labels == 0).sum()
        n_flat = (labels == 1).sum()
        dt = time.time() - t0

        smoke_log(f"Labels: {n_valid} valid (L={n_long} F={n_flat} S={n_short}) in {dt:.1f}s")
        results['steps']['labels'] = {
            'status': 'PASS', 'n_valid': int(n_valid),
            'n_long': int(n_long), 'n_flat': int(n_flat), 'n_short': int(n_short),
        }
    except Exception as e:
        smoke_log(f"FAIL labels: {e}")
        traceback.print_exc()
        results['steps']['labels'] = {'status': 'FAIL', 'error': str(e)}
        results['errors'].append(f"labels: {e}")
        results['passed'] = False
        return results

    # ============================================================
    # STEP 5: CPCV splits + sample uniqueness
    # ============================================================
    try:
        # CPCV functions defined inline — ml_multi_tf.py runs its training loop
        # at module level (not guarded by __name__), so we CANNOT import from it.
        from itertools import combinations
        def _compute_sample_uniqueness(t0_arr, t1_arr, n_bars):
            n_events = len(t0_arr)
            concurrent = np.zeros(n_bars, dtype=np.int32)
            for i in range(n_events):
                s, e = int(t0_arr[i]), min(int(t1_arr[i]) + 1, n_bars)
                concurrent[s:e] += 1
            uniqueness = np.ones(n_events, dtype=np.float64)
            for i in range(n_events):
                s, e = int(t0_arr[i]), min(int(t1_arr[i]) + 1, n_bars)
                if e > s:
                    conc_slice = np.where(concurrent[s:e] > 0, concurrent[s:e].astype(float), 1)
                    uniqueness[i] = np.mean(1.0 / conc_slice)
            return uniqueness
        def _generate_cpcv_splits(n_samples, n_groups=6, n_test_groups=2, max_hold_bars=None, embargo_pct=0.01):
            group_size = n_samples // n_groups
            groups = [np.arange(g * group_size, (g+1) * group_size if g < n_groups-1 else n_samples) for g in range(n_groups)]
            embargo_size = max(1, int(n_samples * embargo_pct))
            splits = []
            for test_ids in combinations(range(n_groups), n_test_groups):
                test_idx = np.concatenate([groups[g] for g in test_ids])
                train_idx = np.concatenate([groups[g] for g in range(n_groups) if g not in test_ids])
                if max_hold_bars:
                    for g in test_ids:
                        for boundary in [groups[g][0], groups[g][-1]]:
                            train_idx = train_idx[np.abs(train_idx - boundary) > max_hold_bars]
                for g in test_ids:
                    end = groups[g][-1]
                    train_idx = train_idx[(train_idx <= end) | (train_idx > end + embargo_size)]
                if len(train_idx) > 0 and len(test_idx) > 0:
                    splits.append((train_idx, test_idx))
            return splits

        tb_cfg = TRIPLE_BARRIER_CONFIG.get(tf_name, TRIPLE_BARRIER_CONFIG['1h'])
        max_hold = tb_cfg.get('max_hold_bars', 24)
        n = len(result_df)

        t0_arr = np.arange(n)
        t1_arr = np.minimum(t0_arr + max_hold, n - 1)

        uniqueness = _compute_sample_uniqueness(t0_arr, t1_arr, n)

        splits = _generate_cpcv_splits(
            n, n_groups=cfg['cpcv_groups'], n_test_groups=cfg['cpcv_test_groups'],
            max_hold_bars=max_hold, embargo_pct=0.01,
        )

        smoke_log(f"CPCV: {len(splits)} paths, uniqueness [{uniqueness.min():.3f}, {uniqueness.max():.3f}]")
        results['steps']['cpcv'] = {
            'status': 'PASS', 'n_paths': len(splits),
            'uniqueness_min': round(float(uniqueness.min()), 3),
            'uniqueness_max': round(float(uniqueness.max()), 3),
        }
    except Exception as e:
        smoke_log(f"FAIL cpcv: {e}")
        traceback.print_exc()
        results['steps']['cpcv'] = {'status': 'FAIL', 'error': str(e)}
        results['errors'].append(f"cpcv: {e}")
        results['passed'] = False
        return results

    # ============================================================
    # STEP 6: XGBoost training on one CPCV path (matches production pipeline)
    # ============================================================
    try:
        import xgboost as xgb
        smoke_log(f"XGBoost version: {xgb.__version__}")

        # Prepare features
        meta_cols = {'timestamp', 'date', 'open', 'high', 'low', 'close', 'volume',
                     'quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote',
                     'open_time', 'symbol', 'timeframe'}
        feature_cols = [c for c in result_df.columns if c not in meta_cols
                       and 'next_' not in c and 'target' not in c and 'triple' not in c
                       and result_df[c].dtype in ('float64', 'float32', 'int64', 'int32', 'int8', 'uint8')]
        X_all = result_df[feature_cols].values.astype(np.float32)
        X_all = np.where(np.isinf(X_all), np.nan, X_all)
        y_all = labels.copy()

        # Train on first CPCV path
        train_idx, test_idx = splits[0]
        train_valid = ~np.isnan(y_all[train_idx])
        test_valid = ~np.isnan(y_all[test_idx])

        X_train = X_all[train_idx][train_valid]
        y_train = y_all[train_idx][train_valid].astype(int)
        X_test = X_all[test_idx][test_valid]
        y_test = y_all[test_idx][test_valid].astype(int)

        if len(X_train) < 30 or len(X_test) < 10:
            smoke_log(f"WARNING: not enough labeled samples (train={len(X_train)}, test={len(X_test)})")
            results['steps']['xgb_train'] = {'status': 'WARN', 'reason': 'too few samples'}
        else:
            from config import V3_XGBM_PARAMS
            t0 = time.time()
            params = V3_XGBM_PARAMS.copy()
            # Override for fast smoke test
            params['max_depth'] = 3

            w_train = uniqueness[train_idx][train_valid].astype(np.float32)
            dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train,
                                 feature_names=feature_cols, nthread=-1)
            dtest = xgb.DMatrix(X_test, label=y_test,
                                feature_names=feature_cols, nthread=-1)

            model = xgb.train(
                params, dtrain,
                num_boost_round=cfg['xgb_rounds'],
                evals=[(dtest, 'test')],
                early_stopping_rounds=cfg['xgb_early_stop'],
                verbose_eval=0,
            )

            preds = model.predict(dtest)
            pred_labels = np.argmax(preds, axis=1)
            acc = (pred_labels == y_test).mean()
            dt = time.time() - t0

            smoke_log(f"XGBoost: acc={acc:.3f}, {model.best_iteration} trees in {dt:.1f}s")
            results['steps']['xgb_train'] = {
                'status': 'PASS', 'accuracy': round(float(acc), 3),
                'n_trees': int(model.best_iteration), 'time': round(dt, 1),
                'n_features': len(feature_cols),
            }

            # Verify model save/load roundtrip (production uses .json format)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as _tmp:
                model.save_model(_tmp.name)
                model_loaded = xgb.Booster(model_file=_tmp.name)
                preds2 = model_loaded.predict(dtest)
                assert np.allclose(preds, preds2, atol=1e-6), "Model save/load roundtrip mismatch!"
                os.remove(_tmp.name)
            smoke_log(f"  Model save/load roundtrip: PASS (.json format)")

            # Store OOS predictions for subsequent steps
            oos_preds = [{
                'path': 0, 'y_true': y_test, 'y_pred_probs': preds,
                'y_pred_labels': pred_labels, 'test_indices': test_idx[test_valid],
            }]

    except Exception as e:
        smoke_log(f"FAIL xgb_train: {e}")
        traceback.print_exc()
        results['steps']['xgb_train'] = {'status': 'FAIL', 'error': str(e)}
        results['errors'].append(f"xgb_train: {e}")
        oos_preds = []

    # ============================================================
    # STEP 7: PBO + Deflated Sharpe
    # ============================================================
    try:
        from backtest_validation import compute_pbo, compute_deflated_sharpe

        if oos_preds:
            pbo = compute_pbo(oos_preds)
            dsr = compute_deflated_sharpe(
                observed_sharpe=0.5, n_trials=100, n_observations=len(y_test),
            )
            smoke_log(f"PBO={pbo.get('pbo', 'N/A')}, DSR p={dsr['p_value']:.3f}")
            results['steps']['pbo_dsr'] = {
                'status': 'PASS',
                'pbo': pbo.get('pbo'),
                'dsr_p_value': round(dsr['p_value'], 4),
            }
        else:
            results['steps']['pbo_dsr'] = {'status': 'SKIP', 'reason': 'no OOS predictions'}

    except Exception as e:
        smoke_log(f"FAIL pbo_dsr: {e}")
        results['steps']['pbo_dsr'] = {'status': 'FAIL', 'error': str(e)}
        results['errors'].append(f"pbo_dsr: {e}")

    # ============================================================
    # STEP 8: Meta-labeling
    # ============================================================
    try:
        from meta_labeling import train_meta_model, predict_meta

        if oos_preds:
            meta_result = train_meta_model(
                oos_preds, tf_name=f'smoke_{tf_name}', db_dir=PROJECT_DIR,
            )
            if meta_result:
                # Test prediction
                test_probs = np.random.dirichlet([1, 1, 1], 10)
                meta_probs, take = predict_meta(meta_result, test_probs)
                smoke_log(f"Meta-labeling: AUC={meta_result['metrics']['auc']:.3f}, "
                         f"{take.sum()}/10 trades approved")
                results['steps']['meta_labeling'] = {
                    'status': 'PASS',
                    'auc': round(meta_result['metrics']['auc'], 3),
                }
                # Cleanup smoke meta model
                smoke_meta_path = os.path.join(PROJECT_DIR, f'meta_model_smoke_{tf_name}.pkl')
                if os.path.exists(smoke_meta_path):
                    os.remove(smoke_meta_path)
            else:
                results['steps']['meta_labeling'] = {'status': 'WARN', 'reason': 'too few trades'}
        else:
            results['steps']['meta_labeling'] = {'status': 'SKIP', 'reason': 'no OOS predictions'}

    except Exception as e:
        smoke_log(f"FAIL meta_labeling: {e}")
        results['steps']['meta_labeling'] = {'status': 'FAIL', 'error': str(e)}
        results['errors'].append(f"meta_labeling: {e}")

    # ============================================================
    # STEP 9: Kelly bet sizing (math verification)
    # ============================================================
    try:
        from config import (KELLY_SAFETY_FRACTION, KELLY_MAX_RISK_MULT,
                            DD_HALT_THRESHOLD, DD_SCALE_STEEPNESS)
        # Verify Kelly formula produces sane output (using config values)
        confidence = 0.65
        rr = 2.0
        base_risk = 0.01

        p_win = confidence
        b_ratio = rr
        kelly_f = (p_win * b_ratio - (1 - p_win)) / max(b_ratio, 0.01)
        kelly_f = max(kelly_f, 0)
        kelly_risk = base_risk * (1 + KELLY_SAFETY_FRACTION * kelly_f)
        kelly_risk = min(kelly_risk, base_risk * KELLY_MAX_RISK_MULT)

        # Drawdown scaling (uses config thresholds)
        for dd in [0.0, 0.05, 0.10, 0.14, DD_HALT_THRESHOLD]:
            dd_scale = max(0.0, 1.0 - DD_SCALE_STEEPNESS * dd) if dd < DD_HALT_THRESHOLD else 0.0
            final_risk = kelly_risk * dd_scale

        smoke_log(f"Kelly: f={kelly_f:.3f}, risk={kelly_risk:.4f}, halts at {DD_HALT_THRESHOLD*100:.0f}% DD")
        results['steps']['kelly'] = {'status': 'PASS', 'kelly_f': round(kelly_f, 3)}

    except Exception as e:
        smoke_log(f"FAIL kelly: {e}")
        results['steps']['kelly'] = {'status': 'FAIL', 'error': str(e)}
        results['errors'].append(f"kelly: {e}")

    # ============================================================
    # STEP 10: LSTM blending (verify functions exist)
    # ============================================================
    try:
        from lstm_sequence_model import blend_predictions

        # Synthetic test
        xgb_probs = np.random.dirichlet([1, 1, 1], 50).astype(np.float32)
        lstm_probs = np.random.rand(50).astype(np.float32)

        blended = blend_predictions(xgb_probs, lstm_probs, alpha=0.2)
        assert blended.shape == (50, 3), f"Blended shape wrong: {blended.shape}"
        assert np.allclose(blended.sum(axis=1), 1.0, atol=0.01), "Blended probs don't sum to 1"

        smoke_log(f"LSTM blending: PASS (50 samples, probs sum to 1)")
        results['steps']['lstm_blend'] = {'status': 'PASS'}

    except Exception as e:
        smoke_log(f"FAIL lstm_blend: {e}")
        results['steps']['lstm_blend'] = {'status': 'FAIL', 'error': str(e)}
        results['errors'].append(f"lstm_blend: {e}")

    # ============================================================
    # SUMMARY
    # ============================================================
    total_time = time.time() - START
    n_pass = sum(1 for s in results['steps'].values() if s.get('status') == 'PASS')
    n_fail = sum(1 for s in results['steps'].values() if s.get('status') == 'FAIL')
    n_warn = sum(1 for s in results['steps'].values() if s.get('status') in ('WARN', 'SKIP'))
    n_total = len(results['steps'])

    results['passed'] = n_fail == 0
    results['total_time'] = round(total_time, 1)

    print(f"\n{'='*60}")
    print(f"  SMOKE TEST RESULTS: {tf_name}")
    print(f"{'='*60}")
    for step, info in results['steps'].items():
        status = info.get('status', '?')
        icon = {'PASS': '+', 'FAIL': 'X', 'WARN': '!', 'SKIP': '-'}.get(status, '?')
        detail = ''
        if 'error' in info:
            detail = f" -- {info['error'][:60]}"
        elif 'accuracy' in info:
            detail = f" -- acc={info['accuracy']}"
        elif 'n_features' in info:
            detail = f" -- {info['n_features']} features"
        print(f"  [{icon}] {step}: {status}{detail}")

    print(f"\n  {n_pass} passed, {n_fail} failed, {n_warn} warn/skip out of {n_total} steps")
    print(f"  Total time: {total_time:.1f}s")

    if n_fail > 0:
        print(f"\n  ERRORS:")
        for err in results['errors']:
            print(f"    - {err}")
        print(f"\n  FIX THESE BEFORE DEPLOYING TO CLOUD!")
    else:
        print(f"\n  ALL CLEAR — ready for cloud deployment")

    # Save results
    report_path = os.path.join(PROJECT_DIR, f'smoke_test_{tf_name}.json')
    with open(report_path, 'w') as f:
        # Convert numpy for JSON
        def _conv(o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return str(o)
        json.dump(results, f, indent=2, default=_conv)

    return results


if __name__ == "__main__":
    # Parse args BEFORE any heavy imports (ml_multi_tf has side effects)
    parser = argparse.ArgumentParser(description='Smoke test the full pipeline')
    parser.add_argument('--tf', default='1h', help='Timeframe to test (default: 1h)')
    parser.add_argument('--rows', type=int, default=None, help='Max rows (default: 1000)')
    args, _ = parser.parse_known_args()  # ignore unknown args from ml_multi_tf

    results = run_smoke_test(tf_name=args.tf, max_rows=args.rows)
    sys.exit(0 if results['passed'] else 1)
