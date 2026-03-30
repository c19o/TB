#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
smoke_test_v3.py — V3.0 LightGBM Pipeline End-to-End Smoke Test
=================================================================
Validates ALL plumbing: imports, file formats, data flow, LightGBM train/infer,
co-occurrence filter, Optuna TPE, and meta-labeling in ~3-5 minutes on an
i9-13900K + RTX 3090.

Usage:
    python smoke_test_v3.py --quick
"""

import sys
import os
import io
import time
import json
import pickle
import tempfile
import shutil
import warnings
import argparse
import traceback

# ── Encoding fix for Windows ──
if os.name == 'nt':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

warnings.filterwarnings('ignore')

# ── Paths ──
V3_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(V3_DIR)

# Add v3.0 and parent to sys.path for imports
sys.path.insert(0, V3_DIR)
sys.path.insert(1, PARENT_DIR)

# ── Globals ──
START = time.time()
RESULTS = {}  # step_name -> (PASS/FAIL, message)


def elapsed():
    return f"[{time.time() - START:6.1f}s]"


def header(step_num, title):
    print(f"\n{'='*70}")
    print(f"  STEP {step_num}: {title}")
    print(f"{'='*70}")


def result(step_name, passed, msg=""):
    status = "PASS" if passed else "FAIL"
    RESULTS[step_name] = (status, msg)
    tag = f"\033[92m PASS \033[0m" if passed else f"\033[91m FAIL \033[0m"
    print(f"  {elapsed()} [{tag}] {step_name}: {msg}")


# ==============================================================
# STEP 1: Import Check
# ==============================================================
def step_import_check():
    header(1, "IMPORT CHECK — All Critical V3 Modules")

    modules = [
        ("config",                "config"),
        ("feature_library",       "feature_library"),
        ("v2_cross_generator",    "v2_cross_generator"),
        ("v2_multi_asset_trainer","v2_multi_asset_trainer"),
        ("ml_multi_tf",           "ml_multi_tf"),
        ("exhaustive_optimizer",  "exhaustive_optimizer"),
        ("meta_labeling",         "meta_labeling"),
        ("v2_lstm_trainer",       "v2_lstm_trainer"),
        ("v2_feature_layers",     "v2_feature_layers"),
        ("data_access_v2",        "data_access_v2"),
    ]

    all_pass = True
    imported = {}
    for display_name, mod_name in modules:
        try:
            mod = __import__(mod_name)
            imported[mod_name] = mod
            result(f"import_{display_name}", True, f"imported {mod_name}")
        except Exception as e:
            result(f"import_{display_name}", False, f"{type(e).__name__}: {e}")
            all_pass = False

    result("step1_imports", all_pass,
           f"{sum(1 for k,v in RESULTS.items() if k.startswith('import_') and v[0]=='PASS')}/{len(modules)} modules imported")
    return imported


# ==============================================================
# STEP 2: Feature Load (parquet + sparse .npz)
# ==============================================================
def step_feature_load():
    header(2, "FEATURE LOAD — Parquet + Sparse .npz")
    import numpy as np
    import pandas as pd
    from scipy import sparse

    # ── Parquet ──
    parquet_candidates = [
        os.path.join(V3_DIR, "features_BTC_1d.parquet"),
        os.path.join(PARENT_DIR, "v2", "features_BTC_1d.parquet"),
        os.path.join(PARENT_DIR, "features_BTC_1d.parquet"),
    ]

    df = None
    parquet_path = None
    for p in parquet_candidates:
        if os.path.exists(p):
            parquet_path = p
            break

    if parquet_path:
        try:
            df = pd.read_parquet(parquet_path)
            result("load_parquet", True,
                   f"{parquet_path} => shape {df.shape} ({df.shape[1]} cols)")
        except Exception as e:
            result("load_parquet", False, f"{type(e).__name__}: {e}")
    else:
        result("load_parquet", False, "features_BTC_1d.parquet not found in any candidate path")

    # ── Sparse .npz ──
    npz_candidates = [
        os.path.join(V3_DIR, "v2_crosses_BTC_1d.npz"),
        os.path.join(PARENT_DIR, "v2", "v2_crosses_BTC_1d.npz"),
        os.path.join(PARENT_DIR, "v2_crosses_BTC_1d.npz"),
    ]

    sparse_mat = None
    npz_path = None
    for p in npz_candidates:
        if os.path.exists(p):
            npz_path = p
            break

    if npz_path:
        try:
            sparse_mat = sparse.load_npz(npz_path)
            result("load_sparse_npz", True,
                   f"{npz_path} => shape {sparse_mat.shape}, nnz={sparse_mat.nnz:,}")
        except Exception as e:
            result("load_sparse_npz", False, f"{type(e).__name__}: {e}")
    else:
        result("load_sparse_npz", False, "v2_crosses_BTC_1d.npz not found in any candidate path")

    result("step2_feature_load", df is not None and sparse_mat is not None,
           f"parquet={'OK' if df is not None else 'MISSING'}, npz={'OK' if sparse_mat is not None else 'MISSING'}")
    return df, sparse_mat, parquet_path, npz_path


# ==============================================================
# STEP 3: Co-occurrence Filter Test
# ==============================================================
def step_co_occurrence_filter(sparse_mat):
    header(3, "CO-OCCURRENCE FILTER — MIN_CO_OCCURRENCE=3")
    import numpy as np
    from scipy import sparse

    if sparse_mat is None:
        result("step3_co_occurrence", False, "No sparse matrix loaded — skipped")
        return None

    try:
        MIN_CO_OCCURRENCE = 3  # matches production min_nonzero=3 and min_data_in_leaf=3
        cols_before = sparse_mat.shape[1]

        # Count nonzeros per column (convert to CSC for efficient column ops)
        csc = sparse_mat.tocsc()
        nnz_per_col = np.diff(csc.indptr)  # fast: O(n_cols)

        # Columns surviving the filter
        keep_mask = nnz_per_col >= MIN_CO_OCCURRENCE
        cols_after = int(keep_mask.sum())
        cols_dropped = cols_before - cols_after

        # Actually apply the filter
        filtered = csc[:, keep_mask].tocsr()

        result("co_occurrence_count", True,
               f"Before: {cols_before:,} cols, After: {cols_after:,} cols, Dropped: {cols_dropped:,}")
        result("co_occurrence_shape", True,
               f"Filtered sparse shape: {filtered.shape}, nnz={filtered.nnz:,}")
        result("step3_co_occurrence", True,
               f"{cols_before:,} -> {cols_after:,} columns (dropped {cols_dropped:,} with <{MIN_CO_OCCURRENCE} nonzeros)")
        return filtered
    except Exception as e:
        result("step3_co_occurrence", False, f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return None


# ==============================================================
# STEP 4: LightGBM Mini-Train
# ==============================================================
def step_lgbm_mini_train(df, filtered_sparse, tmp_dir):
    header(4, "LIGHTGBM MINI-TRAIN — 100 trees, 3-class, save/load/predict")
    import numpy as np
    import pandas as pd
    from scipy import sparse
    import lightgbm as lgb

    if df is None:
        result("step4_lgbm", False, "No parquet data loaded — skipped")
        return None

    try:
        # ── Build feature matrix ──
        # Base features: all numeric columns except meta/target
        meta_cols = {'timestamp', 'date', 'open', 'high', 'low', 'close', 'volume',
                     'quote_volume', 'trades', 'taker_buy_volume', 'taker_buy_quote',
                     'open_time', 'date_norm'}
        target_like = {c for c in df.columns if 'next_' in c.lower() or 'target' in c.lower() or 'direction' in c.lower()}
        exclude_cols = meta_cols | target_like
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        # Vectorized numeric conversion (no .apply per-column)
        X_base = df[feature_cols].values.astype(np.float32)
        print(f"  Base features: {X_base.shape}")

        # Combine base + filtered sparse (if available)
        if filtered_sparse is not None:
            # Align row counts (parquet rows must match sparse rows)
            n_rows = min(X_base.shape[0], filtered_sparse.shape[0])
            X_base_trimmed = X_base[:n_rows]
            sparse_trimmed = filtered_sparse[:n_rows]

            # Convert base to sparse and hstack
            X_base_sp = sparse.csr_matrix(X_base_trimmed)
            X_all = sparse.hstack([X_base_sp, sparse_trimmed], format='csr')
            print(f"  Combined features: {X_all.shape} (base {X_base_trimmed.shape[1]} + sparse {sparse_trimmed.shape[1]})")
        else:
            n_rows = X_base.shape[0]
            X_all = sparse.csr_matrix(X_base)
            print(f"  Base-only features (no sparse): {X_all.shape}")

        # ── Simple triple-barrier label from next-bar returns (test-only shortcut) ──
        close = pd.to_numeric(df['close'], errors='coerce').values[:n_rows]
        returns = np.diff(close, prepend=close[0]) / np.where(close == 0, 1, close)
        # 3-class: SHORT(0), FLAT(1), LONG(2) based on return thresholds
        y = np.where(returns > 0.005, 2, np.where(returns < -0.005, 0, 1)).astype(np.int32)

        # Drop rows with NaN labels at the boundaries
        valid_mask = np.isfinite(close)
        X_all = X_all[valid_mask]
        y = y[valid_mask]
        n = X_all.shape[0]
        print(f"  Valid samples: {n}, label distribution: SHORT={np.sum(y==0)}, FLAT={np.sum(y==1)}, LONG={np.sum(y==2)}")

        # ── 80/20 split ──
        split = int(n * 0.8)
        X_train, X_test = X_all[:split], X_all[split:]
        y_train, y_test = y[:split], y[split:]
        print(f"  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

        # ── V3 LightGBM params from config.py (100 trees for smoke test) ──
        try:
            from config import V3_LGBM_PARAMS
            params = V3_LGBM_PARAMS.copy()
        except ImportError:
            params = {
                "objective": "multiclass",
                "num_class": 3,
                "metric": "multi_logloss",
                "boosting_type": "gbdt",
                "device": "cpu",
                "force_col_wise": True,
                "max_bin": 255,
                "num_threads": 0,
                "is_enable_sparse": True,
                "min_data_in_leaf": 3,
                "min_gain_to_split": 2.0,
                "lambda_l1": 0.5,
                "lambda_l2": 3.0,
                "feature_fraction": 0.9,
                "feature_fraction_bynode": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 1,
                "num_leaves": 63,
                "learning_rate": 0.03,
                "verbosity": -1,
                "feature_pre_filter": False,  # CRITICAL: True silently kills rare esoteric features
            }

        dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        dtest = lgb.Dataset(X_test, label=y_test, reference=dtrain, free_raw_data=False)

        print(f"  Training LightGBM (100 rounds, {X_all.shape[1]:,} features)...")
        t0 = time.time()
        model = lgb.train(
            params, dtrain, num_boost_round=100,
            valid_sets=[dtest], valid_names=['test'],
            callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)]
        )
        train_time = time.time() - t0
        result("lgbm_train", True, f"Trained in {train_time:.1f}s, best_iter={model.best_iteration}")

        # ── Save model ──
        model_path = os.path.join(tmp_dir, "smoke_model_1d.lightgbm")
        model.save_model(model_path)
        model_size = os.path.getsize(model_path)
        result("lgbm_save", True, f"Saved to {model_path} ({model_size/1024:.1f} KB)")

        # ── Load model back ──
        model_loaded = lgb.Booster(model_file=model_path)
        result("lgbm_load", True, "Model loaded from file")

        # ── Predict on test set ──
        y_pred_probs = model_loaded.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Verify 3-class output shape
        assert y_pred_probs.shape == (X_test.shape[0], 3), \
            f"Expected shape ({X_test.shape[0]}, 3), got {y_pred_probs.shape}"
        result("lgbm_predict_shape", True,
               f"Predictions: {y_pred_probs.shape} (3-class confirmed)")

        # Accuracy
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_test, y_pred)
        result("lgbm_accuracy", True, f"Test accuracy: {acc:.4f} (random baseline ~0.33)")

        # ── Save mock OOS predictions as .pkl (matches meta-labeling input format) ──
        oos_preds = [{
            'path': 0,
            'y_true': y_test,
            'y_pred_probs': y_pred_probs,
            'test_indices': np.arange(split, n),
        }]
        oos_path = os.path.join(tmp_dir, "cpcv_oos_predictions_1d.pkl")
        with open(oos_path, 'wb') as f:
            pickle.dump(oos_preds, f)
        result("lgbm_save_oos", True, f"OOS predictions saved: {oos_path}")

        result("step4_lgbm", True,
               f"train={train_time:.1f}s, acc={acc:.4f}, shape={y_pred_probs.shape}")
        return oos_preds, y_pred_probs

    except Exception as e:
        result("step4_lgbm", False, f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return None, None


# ==============================================================
# STEP 5: Optuna Mini-Test
# ==============================================================
def step_optuna_mini_test(tmp_dir):
    header(5, "OPTUNA MINI-TEST — TPESampler + 5 Trials")
    import numpy as np

    try:
        import optuna
        from optuna.samplers import TPESampler
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        result("optuna_import", True, f"optuna {optuna.__version__}")
    except ImportError as e:
        result("optuna_import", False, str(e))
        result("step5_optuna", False, "optuna not installed")
        return

    try:
        # Try importing simulate_batch from exhaustive_optimizer
        simulate_batch_available = False
        try:
            from exhaustive_optimizer import simulate_batch
            simulate_batch_available = True
            result("optuna_simulate_batch_import", True, "simulate_batch imported")
        except Exception as e:
            result("optuna_simulate_batch_import", False,
                   f"simulate_batch not importable ({type(e).__name__}: {e}) — using mock objective")

        if simulate_batch_available:
            # Run simulate_batch with random data + 5 Optuna trials
            n_bars = 500
            rng = np.random.RandomState(42)
            confs = rng.uniform(0.3, 0.9, n_bars).astype(np.float32)
            dirs = rng.choice([-1, 0, 1], n_bars).astype(np.float32)
            closes = np.cumsum(rng.randn(n_bars) * 0.5 + 50000).astype(np.float32)
            closes = np.abs(closes) + 10000  # keep positive
            atrs = np.full(n_bars, 500.0, dtype=np.float32)
            highs = closes + rng.uniform(0, 200, n_bars).astype(np.float32)
            lows = closes - rng.uniform(0, 200, n_bars).astype(np.float32)
            regime = rng.choice([0, 1, 2, 3], n_bars).astype(np.float32)

            def objective(trial):
                lev = trial.suggest_int('lev', 1, 20)
                risk = trial.suggest_float('risk', 0.1, 2.0)
                stop_atr = trial.suggest_float('stop_atr', 0.5, 3.0)
                rr = trial.suggest_float('rr', 1.0, 5.0)
                hold = trial.suggest_int('hold', 5, 60)
                exit_type = trial.suggest_categorical('exit_type', [0, 25, 50, 75])
                conf = trial.suggest_float('conf', 0.45, 0.85)

                params_batch = np.array([[lev, risk, stop_atr, rr, hold, exit_type, conf]],
                                        dtype=np.float32)
                results_arr = simulate_batch(params_batch, confs, dirs, closes, atrs,
                                             highs, lows, regime, np)
                # results_arr shape: (1, 7) — [balance, max_dd, win_rate, trade_count, roi, sortino, total_trades]
                sortino = float(results_arr[0, 5])
                return sortino if np.isfinite(sortino) else -999.0

        else:
            # Mock objective (just verify Optuna plumbing)
            def objective(trial):
                x = trial.suggest_float('x', -5, 5)
                y = trial.suggest_float('y', -5, 5)
                return -(x ** 2 + y ** 2)

        sampler = TPESampler(seed=42, n_startup_trials=3)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials=5, show_progress_bar=False)

        best = study.best_trial
        result("optuna_5_trials", True,
               f"Best trial #{best.number}: value={best.value:.4f}, params={best.params}")

        # Save mock optuna_configs_1d.json
        configs = {
            'tf': '1d',
            'best_params': best.params,
            'best_value': best.value,
            'n_trials': len(study.trials),
            'smoke_test': True,
        }
        config_path = os.path.join(tmp_dir, "optuna_configs_1d.json")
        with open(config_path, 'w') as f:
            json.dump(configs, f, indent=2, default=str)
        result("optuna_save_config", True, f"Saved {config_path}")

        result("step5_optuna", True,
               f"5 trials completed, best={best.value:.4f}, simulate_batch={'used' if simulate_batch_available else 'mock'}")

    except Exception as e:
        result("step5_optuna", False, f"{type(e).__name__}: {e}")
        traceback.print_exc()


# ==============================================================
# STEP 6: Meta-Labeling Mini-Test
# ==============================================================
def step_meta_labeling_test(oos_preds, tmp_dir):
    header(6, "META-LABELING — train_meta_model + predict_meta")
    import numpy as np

    if oos_preds is None:
        # Create mock OOS predictions (same format as CPCV output)
        print("  No OOS predictions from step 4 — creating mock data...")
        rng = np.random.RandomState(42)
        oos_preds = []
        for i in range(10):
            n = 300
            y_true = rng.randint(0, 3, n)
            y_probs = rng.dirichlet([1, 1, 1], n).astype(np.float32)
            for j in range(n):
                y_probs[j, y_true[j]] += 0.2  # slight edge
            y_probs /= y_probs.sum(axis=1, keepdims=True)
            oos_preds.append({
                'path': i,
                'y_true': y_true,
                'y_pred_probs': y_probs,
                'test_indices': np.arange(i * n, (i + 1) * n),
            })

    try:
        from meta_labeling import train_meta_model, predict_meta

        # Train meta-model (logistic = fast)
        print("  Training meta-model (logistic regression)...")
        meta_result = train_meta_model(
            oos_preds,
            tf_name='1d',
            model_type='logistic',
            db_dir=tmp_dir,
        )

        if meta_result is None:
            result("meta_train", False, "train_meta_model returned None (too few directional predictions)")
            result("step6_meta", False, "Meta-model training failed")
            return

        result("meta_train", True,
               f"AUC={meta_result['metrics']['auc']:.3f}, "
               f"acc={meta_result['metrics']['accuracy']:.3f}, "
               f"threshold={meta_result['threshold']:.2f}")

        # Verify meta_model_1d.pkl was saved
        meta_pkl_path = os.path.join(tmp_dir, "meta_model_1d.pkl")
        if os.path.exists(meta_pkl_path):
            result("meta_save", True, f"meta_model_1d.pkl saved ({os.path.getsize(meta_pkl_path)} bytes)")
        else:
            result("meta_save", False, "meta_model_1d.pkl not found in tmp_dir")

        # predict_meta on new data
        rng = np.random.RandomState(99)
        test_probs = rng.dirichlet([1, 1, 1], 50).astype(np.float32)
        meta_probs, take_trade = predict_meta(meta_result, test_probs)

        assert meta_probs.shape == (50,), f"Expected (50,), got {meta_probs.shape}"
        assert take_trade.shape == (50,), f"Expected (50,), got {take_trade.shape}"
        assert take_trade.dtype == bool, f"Expected bool, got {take_trade.dtype}"

        result("meta_predict", True,
               f"shape={meta_probs.shape}, approved={take_trade.sum()}/{len(take_trade)}, "
               f"prob_range=[{meta_probs.min():.3f}, {meta_probs.max():.3f}]")

        result("step6_meta", True,
               f"train+predict OK, AUC={meta_result['metrics']['auc']:.3f}, "
               f"{take_trade.sum()}/{len(take_trade)} trades approved")

    except Exception as e:
        result("step6_meta", False, f"{type(e).__name__}: {e}")
        traceback.print_exc()


# ==============================================================
# SUMMARY
# ==============================================================
def print_summary():
    total_time = time.time() - START
    print(f"\n{'='*70}")
    print(f"  SMOKE TEST SUMMARY — V3.0 LightGBM Pipeline")
    print(f"{'='*70}")

    # Group by step
    steps = [
        ("Step 1: Imports",        [k for k in RESULTS if k.startswith("step1") or k.startswith("import_")]),
        ("Step 2: Feature Load",   [k for k in RESULTS if k.startswith("step2") or k.startswith("load_")]),
        ("Step 3: Co-occurrence",  [k for k in RESULTS if k.startswith("step3") or k.startswith("co_occurrence")]),
        ("Step 4: LightGBM",      [k for k in RESULTS if k.startswith("step4") or k.startswith("lgbm_")]),
        ("Step 5: Optuna",         [k for k in RESULTS if k.startswith("step5") or k.startswith("optuna_")]),
        ("Step 6: Meta-Labeling",  [k for k in RESULTS if k.startswith("step6") or k.startswith("meta_")]),
    ]

    total_pass = 0
    total_fail = 0

    for step_name, keys in steps:
        step_keys = [k for k in keys if k.startswith("step")]
        detail_keys = [k for k in keys if not k.startswith("step")]

        # Step-level result
        step_status = "---"
        for sk in step_keys:
            if sk in RESULTS:
                step_status = RESULTS[sk][0]

        icon = "+" if step_status == "PASS" else ("-" if step_status == "FAIL" else "?")
        print(f"\n  [{icon}] {step_name}: {step_status}")

        for dk in detail_keys:
            if dk in RESULTS:
                st, msg = RESULTS[dk]
                sub_icon = "+" if st == "PASS" else "-"
                print(f"      [{sub_icon}] {dk}: {msg}")
                if st == "PASS":
                    total_pass += 1
                else:
                    total_fail += 1

    print(f"\n{'─'*70}")
    print(f"  TOTAL: {total_pass} passed, {total_fail} failed, {total_time:.1f}s elapsed")

    if total_fail == 0:
        print(f"  ALL CHECKS PASSED — V3.0 pipeline plumbing is healthy")
    else:
        print(f"  {total_fail} FAILURES — investigate before proceeding")
    print(f"{'='*70}\n")

    return total_fail == 0


# ==============================================================
# MAIN
# ==============================================================
def main():
    parser = argparse.ArgumentParser(description="V3.0 LightGBM Smoke Test")
    parser.add_argument('--quick', action='store_true',
                        help='Run in quick/minimal mode (required)')
    args = parser.parse_args()

    if not args.quick:
        print("Usage: python smoke_test_v3.py --quick")
        print("The --quick flag is required to run in minimal smoke test mode.")
        sys.exit(1)

    print(f"{'='*70}")
    print(f"  V3.0 LIGHTGBM PIPELINE — END-TO-END SMOKE TEST")
    print(f"  Directory: {V3_DIR}")
    print(f"  Mode: --quick (minimal, ~3-5 min target)")
    print(f"{'='*70}")

    # Create temp directory for all outputs (don't pollute project)
    tmp_dir = tempfile.mkdtemp(prefix="smoke_v3_")
    print(f"  Temp dir: {tmp_dir}")

    try:
        # Step 1: Import check
        imported = step_import_check()

        # Step 2: Feature load
        df, sparse_mat, parquet_path, npz_path = step_feature_load()

        # Step 3: Co-occurrence filter
        filtered_sparse = step_co_occurrence_filter(sparse_mat)

        # Step 4: LightGBM mini-train
        oos_preds, y_pred_probs = step_lgbm_mini_train(df, filtered_sparse, tmp_dir)

        # Step 5: Optuna mini-test
        step_optuna_mini_test(tmp_dir)

        # Step 6: Meta-labeling
        step_meta_labeling_test(oos_preds, tmp_dir)

    except KeyboardInterrupt:
        print("\n\n  INTERRUPTED by user")
    except Exception as e:
        print(f"\n  FATAL: {type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        # Summary
        all_pass = print_summary()

        # Cleanup temp dir
        try:
            shutil.rmtree(tmp_dir)
            print(f"  Cleaned up temp dir: {tmp_dir}")
        except Exception:
            print(f"  WARNING: Could not clean up {tmp_dir}")

        sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
