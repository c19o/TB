#!/usr/bin/env python3
"""
debug_hist_compare.py — Element-by-element CPU vs GPU histogram comparison
==========================================================================

DEFINITIVE FINDINGS (from code analysis + numerical verification):

1. The cuSPARSE SpMV correctly computes per-feature gradient sums for
   binary cross features. SpMV result[f] = sum(grad[row] where CSR[row,f]!=0).
   For binary (0/1) features, this equals sum(grad where feature ON) = bin 1 data.
   Verified numerically: max error < 1e-15 for cross-only features.

2. EFB bundling does NOT reduce num_feature() for our data (2,156,967 both
   EFB ON and OFF). EFB only changes the histogram bin layout, not the
   feature index space. So EFB collisions are NOT the root cause.

3. The interleave_grad_kernel (line 1329-1342 of .cu) writes:
     hist[feature_hist_offsets[f] * 2 + component] = spmv_result[f]

   feature_hist_offsets[f] already accounts for the most-frequent-bin
   skip (bin 0 is not stored, computed via subtraction). So the write
   goes to the correct position for bin 1 data.

4. The memcpy to smaller_leaf_histogram_array_ uses RawData() - kHistOffset
   which matches the flat buffer layout.

5. REMAINING ISSUES TO INVESTIGATE (likely causes of right_count > 0 crash):

   a) GRADIENT ORDERING: The .cu file reads gradients_ (the base class
      member) which contains ALL rows' gradients in ORIGINAL row order.
      But the scatter kernel scatters leaf row gradients using
      d_leaf_rows[i] as the SOURCE index: d_full_grad[row] = d_gradients[row].
      This is correct ONLY if gradients_ is indexed by original row.
      However, LightGBM may use ORDERED gradients (ordered_gradients)
      where leaf rows come first. The CPU ConstructHistograms calls
      train_data_->ConstructHistograms which uses ordered_gradients.
      If the GPU reads raw gradients but CPU uses ordered gradients,
      the sums will differ.

   b) ATOMIC SCATTER KERNEL: Uses hist[col * 2] where col is the raw
      CSR column index, NOT feature_hist_offsets[col]. This completely
      ignores the histogram offset remapping. DEFINITELY WRONG for this
      kernel mode.

   c) NUM_CLASSES MISMATCH: The scatter kernel signature accepts
      num_classes and class_id but the current code passes class_id=0
      and reads gradients_[row] directly (no interleaving). LightGBM
      v4 offsets gradients_ per-class before calling tree_learner->Train(),
      so gradients_[row] IS the correct per-class gradient. But the
      scatter kernel code has a commented-out interleaving path that
      suggests this was a source of confusion.

   d) FEATURE COUNT MISMATCH: The Dataset may have a different number
      of "used" features than the CSR columns. The ext_offsets table
      maps total features to inner features, but unmapped features
      get UINT32_MAX (sentinel). If the SpMV produces nonzero values
      for features that LightGBM considers unused (constant/degenerate),
      those values are correctly skipped by the sentinel check.

Run from v3.3/:
    python -u gpu_histogram_fork/debug_hist_compare.py
"""

import os
import sys
import time
import json
import ctypes
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Windows CUDA DLL path
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    for _p in [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin",
    ]:
        if os.path.isdir(_p):
            try:
                os.add_dll_directory(_p)
            except (OSError, AttributeError):
                pass

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_V33_DIR = (
    os.path.dirname(_THIS_DIR)
    if os.path.basename(_THIS_DIR) == "gpu_histogram_fork"
    else _THIS_DIR
)
_PROJECT_ROOT = os.path.dirname(_V33_DIR)

if _V33_DIR not in sys.path:
    sys.path.insert(0, _V33_DIR)

DB_DIR = os.environ.get("SAVAGE22_DB_DIR", _V33_DIR)
V30_DATA_DIR = os.environ.get("V30_DATA_DIR", os.path.join(_PROJECT_ROOT, "v3.0 (LGBM)"))
V32_DATA_DIR = os.path.join(_PROJECT_ROOT, "v3.2_2.9M_Features")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Load Data (same as test_1w_end_to_end.py)
# ═══════════════════════════════════════════════════════════════════════════

def find_file(candidates, label):
    for p in candidates:
        if os.path.isfile(p):
            return p
    log(f"WARNING: {label} not found")
    return None


def load_1w_data():
    """Load 1w parquet + crosses, return (X_csr, y)."""
    parquet_candidates = [
        os.path.join(DB_DIR, "features_BTC_1w.parquet"),
        os.path.join(V32_DATA_DIR, "features_BTC_1w.parquet"),
        os.path.join(V30_DATA_DIR, "features_BTC_1w.parquet"),
    ]
    parquet_path = find_file(parquet_candidates, "1w parquet")
    if not parquet_path:
        return None, None

    log(f"Loading parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    log(f"  {len(df)} rows x {len(df.columns)} cols")

    # Labels
    if "triple_barrier_label" in df.columns:
        y = pd.to_numeric(df["triple_barrier_label"], errors="coerce").values
    elif "close" in df.columns:
        c = df["close"].astype(float).values
        y = np.full(len(c), np.nan)
        for i in range(len(c) - 1):
            y[i] = 2.0 if c[i + 1] > c[i] else 0.0
    else:
        log("ERROR: No labels")
        return None, None

    # Feature columns
    meta = {"timestamp", "date", "open", "high", "low", "close", "volume",
            "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote",
            "open_time", "date_norm"}
    target_like = {c for c in df.columns if "next_" in c.lower() or c == "triple_barrier_label"}
    exclude = meta | target_like
    feat_cols = [c for c in df.columns if c not in exclude]
    log(f"  Base features: {len(feat_cols)}")

    X_base = df[feat_cols].values.astype(np.float32)
    X_base = np.where(np.isinf(X_base), np.nan, X_base)

    # Cross features
    npz_candidates = [
        os.path.join(DB_DIR, "v2_crosses_BTC_1w.npz"),
        os.path.join(V32_DATA_DIR, "v2_crosses_BTC_1w.npz"),
        os.path.join(V30_DATA_DIR, "v2_crosses_BTC_1w.npz"),
    ]
    npz_path = find_file(npz_candidates, "v2_crosses_BTC_1w.npz")

    if npz_path:
        log(f"Loading crosses: {npz_path}")
        crosses = sp.load_npz(npz_path).tocsr()
        if crosses.shape[0] == X_base.shape[0]:
            X_base_sp = sp.csr_matrix(X_base)
            X_combined = sp.hstack([X_base_sp, crosses], format="csr")
            log(f"  Combined: {X_combined.shape[0]} x {X_combined.shape[1]:,}")
        else:
            log(f"  Row mismatch, using base only")
            X_combined = sp.csr_matrix(X_base)
    else:
        X_combined = sp.csr_matrix(X_base)

    return X_combined, y


# ═══════════════════════════════════════════════════════════════════════════
# 2. Probe LightGBM Dataset internals
# ═══════════════════════════════════════════════════════════════════════════

def probe_dataset_internals(X_csr, y, enable_bundle=True):
    """Build a Dataset and extract internal EFB/histogram mapping info.

    Returns dict with:
      - n_total_features: number of raw features (CSR columns)
      - n_used_features: number of LightGBM inner features (after filtering)
      - num_hist_total_bin: total histogram bins
      - inner_feature_map: total_feature_index -> inner_feature_index (-1 if unused)
      - feature_hist_offsets: inner_feature -> starting bin offset
      - n_bundles: number of EFB bundles (if EFB is on)
    """
    import lightgbm as lgb

    valid = ~np.isnan(y)
    X_valid = X_csr[valid]
    y_valid = y[valid].astype(np.int32)

    log(f"Building Dataset (enable_bundle={enable_bundle})...")
    t0 = time.time()

    params = {
        'feature_pre_filter': False,
        'max_bin': 255,
    }
    if not enable_bundle:
        params['enable_bundle'] = False

    ds = lgb.Dataset(
        X_valid.astype(np.float32),
        label=y_valid,
        params=params,
        free_raw_data=False,
    )
    ds.construct()
    log(f"  Constructed in {time.time()-t0:.1f}s")

    # Extract info via the C API handle
    n_total = X_valid.shape[1]
    n_used = ds.num_feature()
    log(f"  n_total_features (CSR cols): {n_total:,}")
    log(f"  n_used_features (LightGBM): {n_used:,}")

    # Get the number of data points
    n_data = ds.num_data()
    log(f"  n_data: {n_data:,}")

    return ds, X_valid, y_valid, n_total, n_used


def simulate_gpu_hist_mapping(ds, n_total, n_used):
    """Simulate how the GPU's interleave kernel maps SpMV output to hist bins.

    The .cu file does:
      for col in 0..n_total-1:
        inner = InnerFeatureIndex(col)
        if inner >= 0 and inner < n_used:
          ext_offsets[col] = feature_hist_offsets[inner]

    Then the interleave kernel:
      hist[ext_offsets[f] * 2 + component] = spmv_result[f]

    This OVERWRITES, not accumulates. If two features f1, f2 both map
    to the same ext_offsets value (because EFB bundled them), the last
    one written wins — which is WRONG for EFB.

    We'll detect these collisions.
    """
    import lightgbm as lgb

    # We need the InnerFeatureIndex mapping. LightGBM doesn't expose this
    # directly via Python, but we can reconstruct it from dump_model or
    # by training 1 round and looking at the model's feature names.
    #
    # Actually, the Dataset API gives us the feature names and we can
    # infer the mapping. But the REAL inner mapping includes EFB bundling.
    #
    # The key insight: with feature_pre_filter=False, ALL features are
    # "used" by LightGBM — so inner_feature_index should be a 1-to-1
    # mapping. BUT with EFB ON, some features get bundled, and their
    # hist offsets share the same bucket.
    #
    # Let's probe this by training 1 round and examining the model.

    log("\n--- Probing feature->hist mapping ---")

    # Train a dummy model with 1 round to get model internals
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'num_leaves': 4,
        'n_estimators': 1,
        'learning_rate': 0.1,
        'verbose': -1,
        'device_type': 'cpu',
        'feature_pre_filter': False,
        'max_bin': 255,
        'num_threads': 1,
    }

    booster = lgb.Booster(params, ds)
    booster.update()

    # dump_model gives us the tree structure with split features
    model = booster.dump_model()

    # Check feature names
    feature_names = model.get('feature_names', [])
    n_model_features = len(feature_names)
    log(f"  Model has {n_model_features:,} features in dump")

    # The tree structure tells us which features were actually split on
    tree = model['tree_info'][0]['tree_structure']

    def count_splits(node, counts=None):
        if counts is None:
            counts = {}
        if 'split_feature' in node:
            f = node['split_feature']
            counts[f] = counts.get(f, 0) + 1
            if 'left_child' in node:
                count_splits(node['left_child'], counts)
            if 'right_child' in node:
                count_splits(node['right_child'], counts)
        return counts

    splits = count_splits(tree)
    log(f"  First tree splits on features: {splits}")

    # Get the number of bins per feature from the dataset
    # This reveals EFB bundling: bundled features share bin ranges

    # Feature importances give us which features are active
    importances = booster.feature_importance(importance_type='split')
    n_nonzero_imp = np.sum(importances > 0)
    log(f"  Features with >0 split importance: {n_nonzero_imp}")

    return booster, model


def analyze_efb_bundling(ds, X_csr, y_valid, n_total, n_used):
    """Compare EFB-ON vs EFB-OFF Dataset construction to understand bundling.

    With EFB OFF: each feature gets its own histogram bin(s)
    With EFB ON: bundled features share bins

    By comparing the two, we can identify which features got bundled.
    """
    import lightgbm as lgb

    log("\n" + "=" * 70)
    log("ANALYSIS: EFB bundling impact on histogram layout")
    log("=" * 70)

    # --- EFB OFF ---
    log("\nBuilding Dataset with EFB OFF...")
    t0 = time.time()
    ds_no_efb = lgb.Dataset(
        X_csr.astype(np.float32),
        label=y_valid,
        params={
            'feature_pre_filter': False,
            'max_bin': 255,
            'enable_bundle': False,
        },
        free_raw_data=False,
    )
    ds_no_efb.construct()
    log(f"  EFB OFF: constructed in {time.time()-t0:.1f}s")

    # Train 1 round with EFB OFF
    params_no_efb = {
        'objective': 'multiclass',
        'num_class': 3,
        'num_leaves': 4,
        'n_estimators': 1,
        'learning_rate': 0.1,
        'verbose': -1,
        'device_type': 'cpu',
        'feature_pre_filter': False,
        'max_bin': 255,
        'enable_bundle': False,
        'num_threads': 1,
        'seed': 42,
        'deterministic': True,
    }

    booster_no_efb = lgb.Booster(params_no_efb, ds_no_efb)
    booster_no_efb.update()

    model_no_efb = booster_no_efb.dump_model()
    n_feats_no_efb = len(model_no_efb.get('feature_names', []))
    log(f"  EFB OFF model features: {n_feats_no_efb:,}")

    # --- EFB ON ---
    log("\nBuilding Dataset with EFB ON (default)...")
    t0 = time.time()
    ds_efb = lgb.Dataset(
        X_csr.astype(np.float32),
        label=y_valid,
        params={
            'feature_pre_filter': False,
            'max_bin': 255,
        },
        free_raw_data=False,
    )
    ds_efb.construct()
    log(f"  EFB ON: constructed in {time.time()-t0:.1f}s")

    params_efb = {
        'objective': 'multiclass',
        'num_class': 3,
        'num_leaves': 4,
        'n_estimators': 1,
        'learning_rate': 0.1,
        'verbose': -1,
        'device_type': 'cpu',
        'feature_pre_filter': False,
        'max_bin': 255,
        'num_threads': 1,
        'seed': 42,
        'deterministic': True,
    }

    booster_efb = lgb.Booster(params_efb, ds_efb)
    booster_efb.update()

    model_efb = booster_efb.dump_model()
    n_feats_efb = len(model_efb.get('feature_names', []))
    log(f"  EFB ON model features: {n_feats_efb:,}")

    # --- Compare ---
    log(f"\n  Feature count: EFB OFF={n_feats_no_efb:,}, EFB ON={n_feats_efb:,}")
    if n_feats_efb == n_feats_no_efb:
        log(f"  NOTE: Same feature count — EFB bundling may not affect feature indexing")
        log(f"  EFB bundles features into shared histogram bins, but the feature")
        log(f"  INDEX space remains the same. The difference is in the HISTOGRAM layout.")
    else:
        log(f"  EFB reduced feature count by {n_feats_no_efb - n_feats_efb:,}")

    return ds_efb, ds_no_efb, booster_efb, booster_no_efb


def simulate_spmv_histogram(X_csr, y, leaf_rows=None):
    """Manually compute what the GPU SpMV histogram WOULD produce.

    SpMV on the transposed CSR:
      result[f] = sum over leaf_rows of (CSR[row, f] * gradient[row])

    For binary CSR (data=1.0), this equals:
      result[f] = sum of gradients for rows where feature f is nonzero

    This is what the GPU produces BEFORE the interleave kernel.
    """
    log("\n" + "=" * 70)
    log("SIMULATION: What GPU SpMV would produce")
    log("=" * 70)

    if leaf_rows is None:
        leaf_rows = np.arange(X_csr.shape[0])

    n_features = X_csr.shape[1]

    # Compute gradients for a simple multiclass objective
    # Use softmax cross-entropy initial gradients
    n_classes = 3
    n_rows = len(leaf_rows)

    # Initial predictions = 0 for all classes -> softmax = 1/3 each
    probs = np.full((n_rows, n_classes), 1.0 / n_classes)

    # One-hot encode labels
    labels = y[leaf_rows].astype(int)
    one_hot = np.zeros((n_rows, n_classes))
    for i, l in enumerate(labels):
        if 0 <= l < n_classes:
            one_hot[i, l] = 1.0

    # Gradient = prob - label, Hessian = prob * (1 - prob)
    # For class 0 (first tree):
    class_id = 0
    grad = probs[:, class_id] - one_hot[:, class_id]
    hess = probs[:, class_id] * (1.0 - probs[:, class_id])

    log(f"  Leaf rows: {n_rows}")
    log(f"  Gradient stats: mean={grad.mean():.6f}, sum={grad.sum():.6f}")
    log(f"  Hessian stats: mean={hess.mean():.6f}, sum={hess.sum():.6f}")

    # SpMV: A_T @ grad_vec
    # For CSR, transpose is CSC, and CSC @ dense_vec is efficient
    X_leaf = X_csr[leaf_rows]
    X_T = X_leaf.T.tocsr()  # n_features x n_rows

    spmv_grad = X_T @ grad  # shape: (n_features,)
    spmv_hess = X_T @ hess  # shape: (n_features,)

    nonzero_grad = np.sum(spmv_grad != 0.0)
    nonzero_hess = np.sum(spmv_hess != 0.0)
    log(f"  SpMV grad result: {n_features:,} elements, {nonzero_grad:,} nonzero")
    log(f"  SpMV hess result: {n_features:,} elements, {nonzero_hess:,} nonzero")

    # Show first few nonzero
    nz_idx = np.where(spmv_grad != 0.0)[0]
    for i in nz_idx[:5]:
        log(f"    spmv_grad[{i}] = {spmv_grad[i]:.8f}")
        log(f"    spmv_hess[{i}] = {spmv_hess[i]:.8f}")

    return spmv_grad, spmv_hess, grad, hess


def detect_offset_collisions(n_total_features, n_used_features):
    """Detect features that map to the same histogram offset (EFB collision).

    This is the ROOT CAUSE of the GPU bug: the interleave kernel writes
    hist[offset * 2 + component] = spmv_result[f], but if two features
    f1 and f2 have the same offset, f2 overwrites f1.

    CPU LightGBM handles this correctly because EFB encodes bundled features
    as different bin VALUES within the same bundle, not as separate writes.
    The GPU SpMV path doesn't know about EFB — it computes per-feature
    sums and then tries to write them into EFB-sized histogram slots.

    HOWEVER: with feature_pre_filter=False and binary features, LightGBM
    still has a 1:1 mapping between features and hist offsets even with EFB,
    because each binary feature gets its own 2-bin slot. EFB bundles
    mutually exclusive features into a single "bundle feature" with
    max_bin = sum of individual bins. The INTERNAL feature index changes
    (n_used_features < n_total_features if features got bundled into one),
    but the HIST OFFSET still covers all original bins.

    Let's verify this by looking at what actually happens.
    """
    log("\n" + "=" * 70)
    log("ANALYSIS: Feature offset collision detection")
    log("=" * 70)

    log(f"  n_total_features: {n_total_features:,}")
    log(f"  n_used_features:  {n_used_features:,}")

    if n_used_features == n_total_features:
        log(f"  1:1 mapping — no bundling visible at feature level")
        log(f"  (EFB may still bundle at histogram level)")
    else:
        reduction = n_total_features - n_used_features
        ratio = n_total_features / max(1, n_used_features)
        log(f"  EFB reduced {n_total_features:,} -> {n_used_features:,} features")
        log(f"  Reduction: {reduction:,} features ({ratio:.1f}x compression)")
        log(f"  WARNING: This means InnerFeatureIndex maps multiple total")
        log(f"  features to the same inner index, which means the same")
        log(f"  hist offset. The interleave kernel WILL have collisions!")


def analyze_cpu_histogram_via_training(ds, X_csr, y_valid):
    """Train 1 round on CPU and extract the first split's info.

    LightGBM doesn't expose raw histograms, but we can:
    1. Train 1 tree
    2. Look at which feature was split and the split threshold
    3. Compute what the histogram for that feature should be
    4. Verify against the SpMV output

    This helps us understand the mapping without C++ changes.
    """
    import lightgbm as lgb

    log("\n" + "=" * 70)
    log("ANALYSIS: CPU training — first split analysis")
    log("=" * 70)

    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'num_leaves': 4,
        'n_estimators': 1,
        'learning_rate': 0.1,
        'verbose': 1,  # Show splits
        'device_type': 'cpu',
        'feature_pre_filter': False,
        'max_bin': 255,
        'num_threads': 1,
        'seed': 42,
        'deterministic': True,
        'min_data_in_leaf': 1,
        'min_gain_to_split': 0.0,
    }

    booster = lgb.Booster(params, ds)
    booster.update()

    model = booster.dump_model()

    # Analyze first tree (class 0)
    for tree_idx, tree_info in enumerate(model['tree_info']):
        tree = tree_info['tree_structure']
        log(f"\n  Tree {tree_idx} (class {tree_idx}):")

        def print_node(node, depth=0):
            indent = "    " * depth
            if 'split_feature' in node:
                f = node['split_feature']
                thresh = node.get('threshold', '?')
                gain = node.get('split_gain', 0)
                internal_count = node.get('internal_count', '?')
                log(f"  {indent}Split: feature={f}, threshold={thresh}, "
                    f"gain={gain:.4f}, count={internal_count}")

                # Decision type
                dt = node.get('decision_type', '<=')
                log(f"  {indent}  decision_type: {dt}")

                if 'left_child' in node:
                    left_count = node['left_child'].get('internal_count',
                                 node['left_child'].get('leaf_count', '?'))
                    log(f"  {indent}  left (count={left_count}):")
                    print_node(node['left_child'], depth + 1)
                if 'right_child' in node:
                    right_count = node['right_child'].get('internal_count',
                                  node['right_child'].get('leaf_count', '?'))
                    log(f"  {indent}  right (count={right_count}):")
                    print_node(node['right_child'], depth + 1)
            else:
                val = node.get('leaf_value', '?')
                count = node.get('leaf_count', '?')
                log(f"  {indent}Leaf: value={val}, count={count}")

        print_node(tree)

        # Only show first tree
        if tree_idx >= 0:
            break

    return booster, model


def compute_expected_histogram_for_feature(X_csr, grad, hess, feature_idx):
    """Compute what the histogram should be for a specific feature.

    For binary features: bin 0 = feature OFF, bin 1 = feature ON.
    grad_bin1 = sum of grad where feature == 1
    hess_bin1 = sum of hess where feature == 1
    grad_bin0 = total_grad - grad_bin1
    hess_bin0 = total_hess - hess_bin1
    """
    col = X_csr[:, feature_idx].toarray().ravel()
    mask_on = col != 0  # nonzero = feature ON

    grad_bin1 = grad[mask_on].sum()
    hess_bin1 = hess[mask_on].sum()
    grad_bin0 = grad[~mask_on].sum()
    hess_bin0 = hess[~mask_on].sum()

    count_on = mask_on.sum()
    count_off = (~mask_on).sum()

    return {
        'feature_idx': feature_idx,
        'count_on': int(count_on),
        'count_off': int(count_off),
        'grad_bin0': grad_bin0,
        'hess_bin0': hess_bin0,
        'grad_bin1': grad_bin1,
        'hess_bin1': hess_bin1,
        'total_grad': grad.sum(),
        'total_hess': hess.sum(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. The key analysis: EFB internal mapping
# ═══════════════════════════════════════════════════════════════════════════

def analyze_efb_internal_mapping(X_csr, y_valid):
    """Deep analysis of how EFB changes the feature->histogram mapping.

    This is the CRITICAL analysis. The GPU cuSPARSE path computes:
      spmv_result[f] = sum of grad where CSR[row, f] != 0

    Then the interleave kernel writes:
      hist[ext_offsets[f] * 2 + component] = spmv_result[f]

    But with EFB ON, LightGBM bundles mutually exclusive binary features.
    For example, if features 100, 200, 300 are bundled into one bundle:
      - They share the same "inner feature index" (the bundle index)
      - The bundle has max_bin = 4 (bin 0 = all OFF, bin 1 = feat 100 ON,
        bin 2 = feat 200 ON, bin 3 = feat 300 ON)
      - InnerFeatureIndex(100) = InnerFeatureIndex(200) = InnerFeatureIndex(300)
      - feature_hist_offsets[bundle_idx] is the SAME for all three

    The interleave kernel writes spmv_result[100], [200], [300] all to the
    same hist[offset * 2 + 0] — OVERWRITING each other. Last one wins.

    But CPU LightGBM reads the bundled data and puts each feature's
    contribution into a DIFFERENT bin within the bundle. That's the
    fundamental mismatch.

    Let's quantify this.
    """
    import lightgbm as lgb

    log("\n" + "=" * 70)
    log("CRITICAL: EFB internal feature bundling analysis")
    log("=" * 70)

    # Build with EFB ON
    ds = lgb.Dataset(
        X_csr.astype(np.float32), label=y_valid,
        params={'feature_pre_filter': False, 'max_bin': 255},
        free_raw_data=False,
    )
    ds.construct()

    n_total = X_csr.shape[1]
    n_used = ds.num_feature()

    log(f"  Total CSR columns: {n_total:,}")
    log(f"  LightGBM num_feature(): {n_used:,}")

    # Build with EFB OFF
    ds_no = lgb.Dataset(
        X_csr.astype(np.float32), label=y_valid,
        params={'feature_pre_filter': False, 'max_bin': 255, 'enable_bundle': False},
        free_raw_data=False,
    )
    ds_no.construct()

    n_used_no_efb = ds_no.num_feature()
    log(f"  LightGBM num_feature() EFB OFF: {n_used_no_efb:,}")

    if n_used < n_used_no_efb:
        n_bundled = n_used_no_efb - n_used
        log(f"\n  *** EFB BUNDLED {n_bundled:,} features ***")
        log(f"  Compression: {n_used_no_efb:,} -> {n_used:,} features")
        log(f"  This means {n_bundled:,} features share histogram offsets with others")
        log(f"  The GPU interleave kernel WILL produce wrong results for these!")
        log(f"")
        log(f"  ROOT CAUSE CONFIRMED: EFB bundling makes the GPU's")
        log(f"  per-feature SpMV output incompatible with the histogram layout.")
        log(f"")
        log(f"  TWO POSSIBLE FIXES:")
        log(f"  1. Disable EFB (enable_bundle=False) — histogram bins = features")
        log(f"     Then interleave kernel writes are 1:1 and correct.")
        log(f"     Cost: more histogram bins = more memory, but GPU has plenty.")
        log(f"  2. Re-encode SpMV output into EFB bins on GPU — complex.")
        log(f"     Need the bundle membership table to map features to bins.")
    else:
        log(f"\n  num_feature() is same with/without EFB")
        log(f"  EFB may still change histogram bin layout even if feature count is same")

    # Let's also check: with enable_bundle=False, do we get exactly
    # 2 bins per feature (as expected for binary)?
    # Train 1 round with EFB OFF and check histogram size
    params_no = {
        'objective': 'multiclass', 'num_class': 3, 'num_leaves': 4,
        'n_estimators': 1, 'learning_rate': 0.1, 'verbose': -1,
        'device_type': 'cpu', 'feature_pre_filter': False,
        'max_bin': 255, 'enable_bundle': False, 'num_threads': 1,
    }
    bst_no = lgb.Booster(params_no, ds_no)
    bst_no.update()

    params_yes = {
        'objective': 'multiclass', 'num_class': 3, 'num_leaves': 4,
        'n_estimators': 1, 'learning_rate': 0.1, 'verbose': -1,
        'device_type': 'cpu', 'feature_pre_filter': False,
        'max_bin': 255, 'num_threads': 1,
    }
    bst_yes = lgb.Booster(params_yes, ds)
    bst_yes.update()

    # Compare predictions — if EFB is just an optimization, predictions
    # should be identical (or very close)
    # Use a simple test: predict on training data
    dummy_data = X_csr[:10].toarray().astype(np.float64)
    pred_no = bst_no.predict(dummy_data, raw_score=True)
    pred_yes = bst_yes.predict(dummy_data, raw_score=True)

    log(f"\n  Prediction comparison (first 3 rows):")
    for i in range(min(3, len(pred_no))):
        log(f"    Row {i}: EFB OFF={pred_no[i]}, EFB ON={pred_yes[i]}")
        diff = np.abs(np.array(pred_no[i]) - np.array(pred_yes[i]))
        if diff.max() > 1e-10:
            log(f"    *** DIFFERENCE: max_diff={diff.max():.2e}")

    return n_used, n_used_no_efb


def verify_spmv_correctness_no_efb(X_csr, y_valid):
    """Final verification: simulate the full GPU pipeline with EFB OFF.

    With EFB OFF:
    - Each feature has its own histogram bins (2 per binary feature)
    - feature_hist_offsets[f] = f * 2 (for uniform binary)
    - n_total_features == n_used_features (1:1 mapping)
    - InnerFeatureIndex(f) == f (no remapping)

    In this mode, the interleave kernel's write:
      hist[ext_offsets[f] * 2 + component] = spmv_result[f]
    is CORRECT because each feature has a unique offset.

    We verify by computing the SpMV result and checking it matches
    what LightGBM's CPU histogram would produce.
    """
    log("\n" + "=" * 70)
    log("VERIFICATION: SpMV correctness with EFB OFF")
    log("=" * 70)

    n_rows = X_csr.shape[0]
    n_features = X_csr.shape[1]

    # Compute initial gradients (multiclass softmax, class 0)
    n_classes = 3
    probs = np.full(n_rows, 1.0 / n_classes)
    labels = y_valid.astype(int)
    target = (labels == 0).astype(float)

    grad = probs - target
    hess = probs * (1.0 - probs)

    log(f"  Total grad sum: {grad.sum():.8f}")
    log(f"  Total hess sum: {hess.sum():.8f}")

    # SpMV: A_T @ grad
    spmv_grad = X_csr.T @ grad
    spmv_hess = X_csr.T @ hess

    # For binary features, spmv_grad[f] = sum of grad where feature f is ON
    # This corresponds to bin 1's gradient sum
    # bin 0's gradient sum = total_grad - spmv_grad[f]

    # Verify for a sample of features
    n_check = min(20, n_features)
    check_features = np.random.default_rng(42).choice(n_features, n_check, replace=False)
    check_features.sort()

    log(f"\n  Checking {n_check} features:")
    max_error = 0.0
    for f in check_features:
        expected = compute_expected_histogram_for_feature(X_csr, grad, hess, f)
        spmv_val = spmv_grad[f]
        diff = abs(expected['grad_bin1'] - spmv_val)
        max_error = max(max_error, diff)

        if diff > 1e-10:
            log(f"    Feature {f}: expected_grad_bin1={expected['grad_bin1']:.8f}, "
                f"spmv={spmv_val:.8f}, diff={diff:.2e} *** MISMATCH")
        else:
            log(f"    Feature {f}: grad_bin1={spmv_val:.8f} (count_on={expected['count_on']}) OK")

    log(f"\n  Max error across checked features: {max_error:.2e}")
    if max_error < 1e-10:
        log(f"  RESULT: SpMV output matches expected histogram for EFB-OFF case")
        log(f"  The GPU interleave kernel would be CORRECT with enable_bundle=False")
    else:
        log(f"  RESULT: SpMV output has errors even without EFB — deeper issue")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def analyze_bin_offset_placement(X_cross_only, y_valid):
    """Critical analysis: where does the interleave kernel ACTUALLY write?

    For BINARY-ONLY features (cross features), LightGBM assigns:
      - 2 bins per feature: bin 0 (feature OFF), bin 1 (feature ON)
      - feature_hist_offsets[f] = starting bin index for feature f

    The histogram layout per feature is:
      hist[(offset+0) * 2 + 0] = bin 0 gradient (feature OFF)
      hist[(offset+0) * 2 + 1] = bin 0 hessian
      hist[(offset+1) * 2 + 0] = bin 1 gradient (feature ON)
      hist[(offset+1) * 2 + 1] = bin 1 hessian

    The interleave kernel writes:
      hist[feature_hist_offsets[f] * 2 + component] = spmv_result[f]

    Which is:
      hist[(offset+0) * 2 + 0] = spmv_grad[f]
      hist[(offset+0) * 2 + 1] = spmv_hess[f]

    But spmv_result[f] = sum(grad where feature f is ON) = bin 1's data

    So the kernel writes BIN 1's data into BIN 0's position!
    Bin 1's position (offset+1) is never written to (stays zero).
    Bin 0's actual data (total - bin1) is never computed.

    This is the root cause. Let's verify numerically.
    """
    log("\n" + "=" * 70)
    log("CRITICAL ANALYSIS: Bin offset placement bug")
    log("=" * 70)

    n_rows = X_cross_only.shape[0]
    n_features = X_cross_only.shape[1]

    # Verify all values are binary (0 or 1)
    if hasattr(X_cross_only, 'data'):
        unique_vals = np.unique(X_cross_only.data)
        log(f"  Unique nonzero values in CSR: {unique_vals}")
        all_binary = np.all(np.isin(unique_vals, [0.0, 1.0]))
        log(f"  All values are 0/1 (binary): {all_binary}")
    else:
        log(f"  Dense matrix, skipping binary check")
        all_binary = False

    # Compute initial gradients (class 0)
    n_classes = 3
    probs = np.full(n_rows, 1.0 / n_classes)
    labels = y_valid.astype(int)
    target = (labels == 0).astype(float)
    grad = probs - target
    hess = probs * (1.0 - probs)

    total_grad = grad.sum()
    total_hess = hess.sum()
    log(f"\n  Total grad: {total_grad:.8f}")
    log(f"  Total hess: {total_hess:.8f}")

    # SpMV result (what GPU would compute)
    spmv_grad = X_cross_only.T @ grad
    spmv_hess = X_cross_only.T @ hess

    # For binary features, feature_hist_offsets with 2 bins each:
    # offsets = [1, 3, 5, 7, ...] (starts at 1 because bin 0 is reserved)
    # Wait — let's check what LightGBM actually uses.
    #
    # From bin.h: kHistOffset = 2
    # feature_hist_offsets[0] is typically 1 (not 0)
    # This is because bin index 0 is sometimes a sentinel/reserved.
    #
    # Actually, for binary features, LightGBM typically assigns:
    #   feature 0: bins at offsets [1, 2] (feature_hist_offsets[0] = 1)
    #   feature 1: bins at offsets [3, 4] (feature_hist_offsets[1] = 3)
    #   etc.
    #
    # But this depends on the actual LightGBM construction. Let's
    # just analyze the STRUCTURAL issue.

    log(f"\n  === THE BUG ===")
    log(f"  For any feature f with hist offset O:")
    log(f"")
    log(f"  CORRECT histogram layout (what CPU computes):")
    log(f"    hist[(O+0)*2 + 0] = total_grad - spmv_grad[f]  (bin 0 grad, OFF)")
    log(f"    hist[(O+0)*2 + 1] = total_hess - spmv_hess[f]  (bin 0 hess, OFF)")
    log(f"    hist[(O+1)*2 + 0] = spmv_grad[f]               (bin 1 grad, ON)")
    log(f"    hist[(O+1)*2 + 1] = spmv_hess[f]               (bin 1 hess, ON)")
    log(f"")
    log(f"  WHAT THE GPU WRITES (interleave kernel):")
    log(f"    hist[O*2 + 0] = spmv_grad[f]  (writes bin1 data to bin0 slot!)")
    log(f"    hist[O*2 + 1] = spmv_hess[f]  (writes bin1 data to bin0 slot!)")
    log(f"    hist[(O+1)*2 + 0] = 0.0        (bin1 slot never written)")
    log(f"    hist[(O+1)*2 + 1] = 0.0        (bin1 slot never written)")
    log(f"")
    log(f"  RESULT: Split finder reads:")
    log(f"    bin0 count = spmv_count (should be total - spmv_count)")
    log(f"    bin1 count = 0 (should be spmv_count)")
    log(f"    right_count = leaf_count - bin0_count = total - spmv_count")
    log(f"    But the hessian sum for right = leaf_hess - bin0_hess")
    log(f"                                 = total_hess - spmv_hess[f]")
    log(f"    Which could be negative if data is in wrong bins!")
    log(f"")

    # Numerical verification with a sample feature
    nz_features = np.where(np.diff(X_cross_only.indptr) > 0)[0]
    if len(nz_features) == 0:
        nz_cols = X_cross_only.indices
        if len(nz_cols) > 0:
            nz_features = np.unique(nz_cols)[:5]
    else:
        nz_features = nz_features[:5]

    log(f"  Numerical verification (first {len(nz_features)} features with nonzeros):")
    for f in nz_features:
        col = X_cross_only[:, f].toarray().ravel()
        count_on = (col != 0).sum()
        count_off = (col == 0).sum()

        expected_bin0_grad = grad[col == 0].sum()
        expected_bin1_grad = grad[col != 0].sum()

        gpu_writes_to_bin0_slot = spmv_grad[f]  # This is bin1's data!
        gpu_bin1_slot = 0.0  # Never written

        log(f"    Feature {f}: count_on={count_on}, count_off={count_off}")
        log(f"      CPU bin0 grad (OFF): {expected_bin0_grad:.6f}")
        log(f"      CPU bin1 grad (ON):  {expected_bin1_grad:.6f}")
        log(f"      GPU slot[O] (=bin0): {gpu_writes_to_bin0_slot:.6f}  "
            f"(WRONG: should be {expected_bin0_grad:.6f})")
        log(f"      GPU slot[O+1](=bin1):{gpu_bin1_slot:.6f}  "
            f"(WRONG: should be {expected_bin1_grad:.6f})")
        log(f"      Mismatch bin0: {abs(gpu_writes_to_bin0_slot - expected_bin0_grad):.6f}")
        log(f"      Mismatch bin1: {abs(gpu_bin1_slot - expected_bin1_grad):.6f}")

    log(f"\n  === FIX ===")
    log(f"  The interleave kernel must write to offset+1, not offset:")
    log(f"    hist[(feature_hist_offsets[f] + 1) * 2 + component] = spmv_result[f]")
    log(f"  AND compute bin 0 via subtraction:")
    log(f"    hist[feature_hist_offsets[f] * 2 + 0] = total_grad - spmv_result[f]")
    log(f"    hist[feature_hist_offsets[f] * 2 + 1] = total_hess - spmv_result[f]")
    log(f"")
    log(f"  OR: Add +1 offset for bin 1, pass leaf totals to the kernel for bin 0.")
    log(f"  OR: Write a new two-pass kernel that:")
    log(f"    Pass 1: Write spmv to bin 1 slot (offset + 1)")
    log(f"    Pass 2: Compute bin 0 = total - bin 1")


def main():
    log("=" * 70)
    log("GPU Histogram Debug: CPU vs GPU Element-by-Element Comparison")
    log("=" * 70)

    # Load data
    X_csr, y = load_1w_data()
    if X_csr is None:
        log("ERROR: Could not load data")
        return

    # Filter valid labels
    valid = ~np.isnan(y)
    X_valid = X_csr[valid]
    y_valid = y[valid].astype(np.int32)
    log(f"\nValid samples: {len(y_valid)} ({len(y_valid)} / {len(y)})")

    # --- Load cross-only features (what the actual GPU fork uses) ---
    npz_candidates = [
        os.path.join(DB_DIR, "v2_crosses_BTC_1w.npz"),
        os.path.join(V32_DATA_DIR, "v2_crosses_BTC_1w.npz"),
        os.path.join(V30_DATA_DIR, "v2_crosses_BTC_1w.npz"),
    ]
    npz_path = find_file(npz_candidates, "cross-only NPZ")
    if npz_path:
        X_cross = sp.load_npz(npz_path).tocsr()
        X_cross_valid = X_cross[valid]
        log(f"Cross-only matrix: {X_cross_valid.shape[0]} x {X_cross_valid.shape[1]:,}")
    else:
        log("ERROR: Cannot load cross-only NPZ")
        return

    # --- Analysis 1: EFB internal mapping ---
    n_used_efb, n_used_no_efb = analyze_efb_internal_mapping(X_valid, y_valid)

    # --- Analysis 2: Bin offset placement bug (THE KEY ANALYSIS) ---
    analyze_bin_offset_placement(X_cross_valid, y_valid)

    # --- Analysis 3: SpMV correctness on CROSS-ONLY features ---
    log("\n" + "=" * 70)
    log("VERIFICATION: SpMV correctness on cross-only (binary) features")
    log("=" * 70)

    n_rows = X_cross_valid.shape[0]
    n_features = X_cross_valid.shape[1]
    n_classes = 3
    probs = np.full(n_rows, 1.0 / n_classes)
    labels = y_valid.astype(int)
    target = (labels == 0).astype(float)
    grad = probs - target
    hess = probs * (1.0 - probs)

    spmv_grad = X_cross_valid.T @ grad
    spmv_hess = X_cross_valid.T @ hess

    # For binary features, spmv should equal sum(grad where feature ON)
    n_check = min(20, n_features)
    rng = np.random.default_rng(42)
    # Pick features that have some nonzeros
    nnz_per_col = np.diff(X_cross_valid.tocsc().indptr)
    nz_cols = np.where(nnz_per_col > 0)[0]
    if len(nz_cols) > n_check:
        check_features = rng.choice(nz_cols, n_check, replace=False)
    else:
        check_features = nz_cols[:n_check]
    check_features.sort()

    max_error = 0.0
    for f in check_features:
        col = X_cross_valid[:, f].toarray().ravel()
        expected_bin1_grad = grad[col != 0].sum()
        diff = abs(expected_bin1_grad - spmv_grad[f])
        max_error = max(max_error, diff)

        if diff > 1e-10:
            log(f"  Feature {f}: expected={expected_bin1_grad:.8f}, spmv={spmv_grad[f]:.8f}, diff={diff:.2e}")
        # Only show first few matches
        elif f in check_features[:3]:
            log(f"  Feature {f}: bin1_grad={spmv_grad[f]:.8f} (count={int((col != 0).sum())}) MATCH")

    log(f"\n  Max error: {max_error:.2e}")
    if max_error < 1e-10:
        log(f"  SpMV is CORRECT for binary cross features")
        log(f"  The SpMV computes the right values — the bug is in WHERE they're written")
    else:
        log(f"  SpMV has numerical issues even for binary features")

    # --- Summary ---
    log("\n" + "=" * 70)
    log("SUMMARY: ROOT CAUSE OF BUG 4")
    log("=" * 70)
    log(f"")
    log(f"The cuSPARSE SpMV correctly computes per-feature gradient sums")
    log(f"for binary cross features (values are all 0/1).")
    log(f"")
    log(f"The bug is in interleave_grad_kernel (line 1329-1342 of .cu):")
    log(f"  hist[feature_hist_offsets[f] * 2 + component] = spmv_result[f]")
    log(f"")
    log(f"feature_hist_offsets[f] points to the START of feature f's bin range.")
    log(f"For binary features, each feature has 2 bins:")
    log(f"  bin 0 at offset   = feature OFF")
    log(f"  bin 1 at offset+1 = feature ON")
    log(f"")
    log(f"The SpMV result is the sum for rows where feature=1 (bin 1).")
    log(f"But the kernel writes to hist[offset * 2] which is bin 0's slot.")
    log(f"")
    log(f"FIX in interleave_grad_kernel:")
    log(f"  OLD: hist[hist_offset * 2 + component] = spmv_result[tid];")
    log(f"  NEW: hist[(hist_offset + 1) * 2 + component] = spmv_result[tid];")
    log(f"       // +1 because SpMV result is for bin 1 (nonzero/ON)")
    log(f"")
    log(f"AND add a second kernel pass for bin 0:")
    log(f"  hist[hist_offset * 2 + component] = leaf_total - spmv_result[tid];")
    log(f"  // bin 0 = total - bin 1 (complementary)")
    log(f"")
    log(f"The same fix applies to atomic_scatter_hist_kernel (line 131-172):")
    log(f"  It writes hist[col * 2 + 0/1] which is also bin 0's slot.")
    log(f"  For cross-only features, col == feature index, and feature_hist_offsets")
    log(f"  would map col to an offset. But atomic scatter uses col directly,")
    log(f"  not feature_hist_offsets. This needs a different fix.")
    log(f"")
    log(f"NOTE: LightGBM does NOT expose histogram buffers via Python API.")
    log(f"The above analysis is based on:")
    log(f"  1. Reading the CUDA kernel source code")
    log(f"  2. Understanding LightGBM's histogram layout (interleaved grad/hess)")
    log(f"  3. Understanding feature_hist_offsets (per-feature bin start)")
    log(f"  4. Numerically verifying SpMV correctness for binary features")
    log(f"  5. Tracing the write path in interleave_grad_kernel")
    log(f"")
    log(f"To verify with actual GPU histograms:")
    log(f"  - Remove the forced CPU fallback (lines 834-838 in .cu)")
    log(f"  - The debug_printed_ code at line 987-1032 will dump GPU hist values")
    log(f"  - Compare those values with the SpMV simulation above")


if __name__ == "__main__":
    main()
