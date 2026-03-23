"""
feature_classifier.py - ML pipeline that classifies features as DIRECTIONAL, VOLATILITY, DUAL, or NOISE.

Runs after training. Produces FEATURE_CLASSIFICATION_REPORT.md categorizing every feature
based on dual-target XGBoost importance and mutual information analysis.

Uses GPU (RTX 3090) for XGBoost training.
"""

import os
import sys
import time
import sqlite3
import warnings
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from scipy import stats

warnings.filterwarnings("ignore")

BASE_DIR = Path(os.environ.get("SAVAGE22_V1_DIR", str(Path(__file__).parent.parent)))
REPORT_PATH = BASE_DIR / "FEATURE_CLASSIFICATION_REPORT.md"

# Columns that are targets or metadata, never features
TARGET_PATTERNS = [
    "next_", "timestamp", "open", "high", "low", "close", "volume",
    "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote",
]

# Timeframes to process: (db_file, table_name, return_col, direction_col)
TIMEFRAMES = [
    ("features_1h.db", "features_1h", "next_1h_return", "next_1h_direction"),
    ("features_4h.db", "features_4h", "next_4h_return", "next_4h_direction"),
]

# Thresholds
MI_RATIO_DIR_THRESHOLD = 1.5
MI_RATIO_VOL_THRESHOLD = 0.67
NAN_THRESHOLD = 0.90  # skip features with >90% NaN
IMPORTANCE_PERCENTILE = 50  # top 50% = above median


def elapsed_str(start):
    return f"[{time.time() - start:.1f}s]"


def detect_gpu():
    """Check if GPU is available for XGBoost."""
    try:
        d = xgb.DMatrix(np.array([[1, 2], [3, 4]]), label=[0, 1])
        # XGBoost 2.x+ uses device="cuda" with tree_method="hist"
        xgb.train({"tree_method": "hist", "device": "cuda", "max_depth": 2},
                   d, num_boost_round=2, verbose_eval=False)
        return True
    except Exception:
        try:
            # Fallback: older XGBoost uses gpu_hist
            xgb.train({"tree_method": "gpu_hist", "max_depth": 2},
                       d, num_boost_round=2, verbose_eval=False)
            return True
        except Exception:
            return False


def load_features(db_file, table_name, return_col, direction_col):
    """Load feature DB into DataFrame, separate features from targets."""
    db_path = BASE_DIR / db_file
    if not db_path.exists():
        print(f"  WARNING: {db_path} not found, skipping")
        return None, None, None, None

    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()

    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns from {db_file}")

    if return_col not in df.columns or direction_col not in df.columns:
        print(f"  WARNING: target columns {return_col}/{direction_col} not found, skipping")
        return None, None, None, None

    # Extract targets
    forward_return = df[return_col].copy()
    direction = df[direction_col].copy()

    # Identify feature columns (exclude targets and metadata)
    feature_cols = []
    for col in df.columns:
        skip = False
        for pat in TARGET_PATTERNS:
            if col.startswith(pat) or col == pat:
                skip = True
                break
        if not skip:
            feature_cols.append(col)

    print(f"  Identified {len(feature_cols)} feature columns")

    features = df[feature_cols].copy()
    return features, forward_return, direction, feature_cols


def filter_features(features, feature_cols):
    """Drop features that are >90% NaN. Return cleaned data and valid column names."""
    nan_fracs = features.isnull().mean()
    valid_mask = nan_fracs < NAN_THRESHOLD
    dropped = (~valid_mask).sum()
    valid_cols = [c for c, v in zip(feature_cols, valid_mask) if v]

    if dropped > 0:
        print(f"  Dropped {dropped} features with >{NAN_THRESHOLD*100:.0f}% NaN")

    features_clean = features[valid_cols].copy()

    # Fill remaining NaN with column median for model training
    for col in valid_cols:
        if features_clean[col].isnull().any():
            features_clean[col] = features_clean[col].fillna(features_clean[col].median())

    # Drop any columns that are constant (zero variance)
    std = features_clean.std()
    constant_cols = std[std == 0].index.tolist()
    if constant_cols:
        print(f"  Dropped {len(constant_cols)} constant features")
        features_clean = features_clean.drop(columns=constant_cols)
        valid_cols = [c for c in valid_cols if c not in constant_cols]

    print(f"  {len(valid_cols)} features after filtering")
    return features_clean, valid_cols


def train_dual_models(features, forward_return, direction, use_gpu):
    """Train Model A (direction classifier) and Model B (volatility regressor).
    Returns feature importance dicts for both."""
    start = time.time()

    # Prepare targets
    dir_target = (direction > 0).astype(int)  # binary: 1=up, 0=down
    vol_target = forward_return.abs()  # absolute return = volatility proxy

    # Drop rows where targets are NaN
    valid_idx = dir_target.notna() & vol_target.notna() & np.isfinite(vol_target)
    features_v = features.loc[valid_idx].reset_index(drop=True)
    dir_target = dir_target.loc[valid_idx].reset_index(drop=True)
    vol_target = vol_target.loc[valid_idx].reset_index(drop=True)

    print(f"  Training on {len(features_v)} valid rows")

    # Train/test split
    X_train, X_test, y_dir_train, y_dir_test, y_vol_train, y_vol_test = (
        train_test_split(features_v, dir_target, vol_target, test_size=0.2,
                         shuffle=False)
    )

    base_params = {
        "tree_method": "hist",
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 50,
        "verbosity": 0,
    }
    if use_gpu:
        base_params["device"] = "cuda"

    # --- Model A: Direction classifier ---
    print(f"  {elapsed_str(start)} Training Model A (direction classifier)...")
    params_a = {**base_params, "objective": "binary:logistic", "eval_metric": "auc"}
    dtrain_a = xgb.DMatrix(X_train, label=y_dir_train, nthread=-1)
    dtest_a = xgb.DMatrix(X_test, label=y_dir_test, nthread=-1)
    model_a = xgb.train(params_a, dtrain_a, num_boost_round=500,
                         evals=[(dtest_a, "test")],
                         early_stopping_rounds=50, verbose_eval=False)
    preds_a = model_a.predict(dtest_a)
    acc = accuracy_score(y_dir_test, (preds_a > 0.5).astype(int))
    print(f"  {elapsed_str(start)} Model A accuracy: {acc:.4f}")

    # --- Model B: Volatility regressor ---
    print(f"  {elapsed_str(start)} Training Model B (volatility regressor)...")
    params_b = {**base_params, "objective": "reg:squarederror", "eval_metric": "rmse"}
    dtrain_b = xgb.DMatrix(X_train, label=y_vol_train, nthread=-1)
    dtest_b = xgb.DMatrix(X_test, label=y_vol_test, nthread=-1)
    model_b = xgb.train(params_b, dtrain_b, num_boost_round=500,
                         evals=[(dtest_b, "test")],
                         early_stopping_rounds=50, verbose_eval=False)
    preds_b = model_b.predict(dtest_b)
    r2 = r2_score(y_vol_test, preds_b)
    print(f"  {elapsed_str(start)} Model B R2: {r2:.4f}")

    # Extract feature importance (gain)
    imp_a = model_a.get_score(importance_type="gain")
    imp_b = model_b.get_score(importance_type="gain")

    metrics = {"dir_accuracy": acc, "vol_r2": r2,
               "model_a_trees": model_a.best_iteration,
               "model_b_trees": model_b.best_iteration}

    return imp_a, imp_b, features_v, dir_target, vol_target, metrics


def compute_mutual_information(features, dir_target, vol_target, valid_cols):
    """Compute MI for direction (classification) and volatility (regression)."""
    start = time.time()
    print(f"  Computing mutual information for {len(valid_cols)} features...")

    # Subsample for speed if dataset is large
    n = len(features)
    if n > 10000:
        sample_idx = np.random.RandomState(42).choice(n, 10000, replace=False)
        features_s = features.iloc[sample_idx].values
        dir_s = dir_target.iloc[sample_idx].values
        vol_s = vol_target.iloc[sample_idx].values
    else:
        features_s = features.values
        dir_s = dir_target.values
        vol_s = vol_target.values

    mi_dir = mutual_info_classif(features_s, dir_s, random_state=42, n_neighbors=5)
    print(f"  {elapsed_str(start)} MI direction done")

    mi_vol = mutual_info_regression(features_s, vol_s, random_state=42, n_neighbors=5)
    print(f"  {elapsed_str(start)} MI volatility done")

    mi_dir_dict = dict(zip(valid_cols, mi_dir))
    mi_vol_dict = dict(zip(valid_cols, mi_vol))

    return mi_dir_dict, mi_vol_dict


def classify_features(valid_cols, imp_a, imp_b, mi_dir_dict, mi_vol_dict):
    """Classify each feature into DIRECTIONAL, VOLATILITY, DUAL, or NOISE."""
    results = []

    # Compute importance medians for threshold
    imp_a_vals = [imp_a.get(c, 0) for c in valid_cols]
    imp_b_vals = [imp_b.get(c, 0) for c in valid_cols]
    median_a = np.median(imp_a_vals) if imp_a_vals else 0
    median_b = np.median(imp_b_vals) if imp_b_vals else 0

    for col in valid_cols:
        mi_d = mi_dir_dict.get(col, 0)
        mi_v = mi_vol_dict.get(col, 0)
        gain_a = imp_a.get(col, 0)
        gain_b = imp_b.get(col, 0)

        # MI ratio (avoid division by zero)
        if mi_v > 1e-10:
            ratio = mi_d / mi_v
        elif mi_d > 1e-10:
            ratio = 999.0  # direction-only
        else:
            ratio = 1.0  # neither has signal

        # Classification logic
        is_top_a = gain_a >= median_a
        is_top_b = gain_b >= median_b

        is_directional = (ratio > MI_RATIO_DIR_THRESHOLD) and is_top_a
        is_volatility = (ratio < MI_RATIO_VOL_THRESHOLD) and is_top_b

        if is_directional and is_volatility:
            category = "DUAL"
        elif is_directional:
            category = "DIRECTIONAL"
        elif is_volatility:
            category = "VOLATILITY"
        elif is_top_a or is_top_b:
            category = "DUAL"
        else:
            category = "NOISE"

        results.append({
            "feature": col,
            "category": category,
            "mi_dir": mi_d,
            "mi_vol": mi_v,
            "mi_ratio": ratio,
            "gain_a": gain_a,
            "gain_b": gain_b,
        })

    return pd.DataFrame(results)


def load_previous_report():
    """Load previous report to track category changes."""
    if not REPORT_PATH.exists():
        return {}

    prev = {}
    try:
        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()

        in_table = False
        for line in lines:
            line = line.strip()
            if line.startswith("| ") and "feature" not in line.lower() and "---" not in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 4:
                    # parts[0] is empty (before first |), parts[1]=feature, parts[2]=category
                    feat = parts[1]
                    cat = parts[2]
                    if cat in ("DIRECTIONAL", "VOLATILITY", "DUAL", "NOISE"):
                        prev[feat] = cat
    except Exception:
        pass
    return prev


def detect_changes(current_df, previous_map):
    """Find features that switched categories since last run."""
    if not previous_map:
        return []

    changes = []
    for _, row in current_df.iterrows():
        feat = row["feature"]
        new_cat = row["category"]
        if feat in previous_map and previous_map[feat] != new_cat:
            changes.append((feat, previous_map[feat], new_cat))
    return changes


def generate_report(all_results, all_metrics, all_changes):
    """Generate the markdown report."""
    lines = []
    lines.append("# Feature Classification Report")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    for tf_name, df, metrics, changes in all_results:
        lines.append(f"## Timeframe: {tf_name}")
        lines.append("")

        # Model metrics
        lines.append("### Model Performance")
        lines.append(f"- Direction classifier accuracy: {metrics['dir_accuracy']:.4f} "
                      f"({metrics['model_a_trees']} trees)")
        lines.append(f"- Volatility regressor R2: {metrics['vol_r2']:.4f} "
                      f"({metrics['model_b_trees']} trees)")
        lines.append(f"- Total features analyzed: {len(df)}")
        lines.append("")

        # Category summary
        counts = df["category"].value_counts()
        lines.append("### Category Summary")
        lines.append("| Category | Count | Pct |")
        lines.append("|----------|-------|-----|")
        for cat in ["DIRECTIONAL", "VOLATILITY", "DUAL", "NOISE"]:
            c = counts.get(cat, 0)
            pct = c / len(df) * 100 if len(df) > 0 else 0
            lines.append(f"| {cat} | {c} | {pct:.1f}% |")
        lines.append("")

        # Changes from previous report
        if changes:
            lines.append("### Category Changes (vs previous run)")
            lines.append("| Feature | Old | New |")
            lines.append("|---------|-----|-----|")
            for feat, old, new in changes[:50]:  # cap at 50
                lines.append(f"| {feat} | {old} | {new} |")
            lines.append("")
        elif all_changes:
            lines.append("### Category Changes (vs previous run)")
            lines.append("No changes detected for this timeframe.")
            lines.append("")

        # Top 20 directional
        dir_df = df[df["category"].isin(["DIRECTIONAL", "DUAL"])].sort_values(
            "mi_ratio", ascending=False).head(20)
        lines.append("### Top 20 Directional Features")
        lines.append("| Feature | MI_dir | MI_vol | Ratio | Gain_A | Gain_B |")
        lines.append("|---------|--------|--------|-------|--------|--------|")
        for _, r in dir_df.iterrows():
            lines.append(f"| {r['feature']} | {r['mi_dir']:.5f} | {r['mi_vol']:.5f} "
                          f"| {r['mi_ratio']:.2f} | {r['gain_a']:.1f} | {r['gain_b']:.1f} |")
        lines.append("")

        # Top 20 volatility
        vol_df = df[df["category"].isin(["VOLATILITY", "DUAL"])].sort_values(
            "mi_ratio", ascending=True).head(20)
        lines.append("### Top 20 Volatility Features")
        lines.append("| Feature | MI_dir | MI_vol | Ratio | Gain_A | Gain_B |")
        lines.append("|---------|--------|--------|-------|--------|--------|")
        for _, r in vol_df.iterrows():
            lines.append(f"| {r['feature']} | {r['mi_dir']:.5f} | {r['mi_vol']:.5f} "
                          f"| {r['mi_ratio']:.2f} | {r['gain_a']:.1f} | {r['gain_b']:.1f} |")
        lines.append("")

        # Full table
        df_sorted = df.sort_values(["category", "mi_ratio"], ascending=[True, False])
        lines.append("### Full Feature Table")
        lines.append("| Feature | Category | MI_dir | MI_vol | Ratio | Gain_A | Gain_B |")
        lines.append("|---------|----------|--------|--------|-------|--------|--------|")
        for _, r in df_sorted.iterrows():
            lines.append(f"| {r['feature']} | {r['category']} | {r['mi_dir']:.5f} "
                          f"| {r['mi_vol']:.5f} | {r['mi_ratio']:.2f} "
                          f"| {r['gain_a']:.1f} | {r['gain_b']:.1f} |")
        lines.append("")

    # Recommendations
    lines.append("## Recommendations")
    lines.append("")
    lines.append("### Feature Engineering Priorities")
    lines.append("1. DIRECTIONAL features are your edge for entry signals -- "
                  "create interaction features between the top directional features")
    lines.append("2. VOLATILITY features should drive position sizing and stop-loss placement, "
                  "not entry decisions")
    lines.append("3. DUAL features are rare and valuable -- they predict both direction and "
                  "magnitude, prioritize these in model stacking")
    lines.append("4. NOISE features have low individual importance but contribute to the matrix -- "
                  "cross-domain interactions may surface value that single-feature metrics miss. "
                  "Do NOT remove")
    lines.append("5. Features with high MI_dir but low model gain may have nonlinear "
                  "relationships worth exploring with deeper trees or interactions")
    lines.append("6. Features with high model gain but low MI may be proxies -- "
                  "investigate what they actually capture")
    lines.append("")

    return "\n".join(lines)


def process_timeframe(db_file, table_name, return_col, direction_col, use_gpu, prev_map):
    """Process a single timeframe end-to-end."""
    tf_name = table_name.replace("features_", "").upper()
    print(f"\n{'='*60}")
    print(f"Processing {tf_name} ({db_file})")
    print(f"{'='*60}")

    # Load
    features, forward_return, direction, feature_cols = load_features(
        db_file, table_name, return_col, direction_col
    )
    if features is None:
        return None

    # Filter
    features_clean, valid_cols = filter_features(features, feature_cols)
    if len(valid_cols) == 0:
        print("  No valid features, skipping")
        return None

    # Train dual models
    imp_a, imp_b, features_v, dir_target, vol_target, metrics = train_dual_models(
        features_clean, forward_return, direction, use_gpu
    )

    # Mutual information
    mi_dir_dict, mi_vol_dict = compute_mutual_information(
        features_v, dir_target, vol_target, valid_cols
    )

    # Classify
    result_df = classify_features(valid_cols, imp_a, imp_b, mi_dir_dict, mi_vol_dict)

    # Track changes
    changes = detect_changes(result_df, prev_map)
    if changes:
        print(f"  {len(changes)} features changed category since last run")

    # Summary
    counts = result_df["category"].value_counts()
    print(f"\n  Results:")
    for cat in ["DIRECTIONAL", "VOLATILITY", "DUAL", "NOISE"]:
        print(f"    {cat}: {counts.get(cat, 0)}")

    return (tf_name, result_df, metrics, changes)


def main():
    start = time.time()
    print("Feature Classifier - Dual-Target Analysis")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # GPU detection
    print("\nDetecting GPU...")
    use_gpu = detect_gpu()
    if use_gpu:
        print("GPU detected - using gpu_hist")
    else:
        print("WARNING: GPU not available, falling back to CPU hist")

    # Load previous report for change tracking
    prev_map = load_previous_report()
    if prev_map:
        print(f"Loaded {len(prev_map)} features from previous report for change tracking")

    # Process each timeframe
    all_results = []
    all_changes = []
    for db_file, table_name, return_col, direction_col in TIMEFRAMES:
        result = process_timeframe(
            db_file, table_name, return_col, direction_col, use_gpu, prev_map
        )
        if result is not None:
            all_results.append(result)
            all_changes.extend(result[3])

    if not all_results:
        print("\nERROR: No timeframes processed successfully")
        sys.exit(1)

    # Generate report
    print(f"\n{elapsed_str(start)} Generating report...")
    report = generate_report(all_results, None, all_changes)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nReport saved to: {REPORT_PATH}")
    print(f"Total time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
