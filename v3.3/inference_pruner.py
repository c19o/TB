#!/usr/bin/env python3
"""
inference_pruner.py -- Feature Pruning for Deployment Models
============================================================
Extracts used features from a trained LightGBM model (split count > 0),
creates a pruned feature mapping, and retrains a deployment model on the
reduced feature set for faster inference.

Key rules (from EXPERT_INFERENCE.md):
- Use split count > 0, NOT gain importance (low-gain features still appear in branches)
- Do NOT hand-edit model.txt -- retrain on reduced feature set instead
- Store feature mapping for interpretability

Usage:
    # Extract active features from trained model:
    python inference_pruner.py --model model_1w.json --tf 1w

    # Or use programmatically:
    from inference_pruner import extract_active_features, save_pruned_artifacts
    active, mapping = extract_active_features('model_1w.json')
    save_pruned_artifacts(active, mapping, '1w', output_dir='.')
"""

import os
import sys
import json
import time
import argparse
import numpy as np

try:
    import lightgbm as lgb
except ImportError:
    print("ERROR: lightgbm required. pip install lightgbm")
    sys.exit(1)

from scipy import sparse as sp_sparse


def extract_active_features(model_path):
    """
    Extract features with split_count > 0 from a trained LightGBM model.

    Returns:
        active_features: list of str -- feature names used in at least one split
        mapping: dict -- {
            'original_to_pruned': {orig_name: pruned_idx},
            'pruned_to_original': {pruned_idx: orig_name},
            'active_feature_names': [str],
            'split_counts': {name: count},
            'total_original': int,
            'total_pruned': int,
            'pruning_ratio': float,
        }
    """
    model = lgb.Booster(model_file=model_path)
    all_features = model.feature_name()
    split_importance = model.feature_importance(importance_type='split')

    # Active = split count > 0 (used in at least one tree split)
    split_counts = dict(zip(all_features, split_importance))
    active_features = [f for f, count in split_counts.items() if count > 0]

    # Preserve original order (important for reproducibility)
    active_set = set(active_features)
    active_features_ordered = [f for f in all_features if f in active_set]

    # Build mapping: original feature name -> pruned index
    original_to_pruned = {name: idx for idx, name in enumerate(active_features_ordered)}
    pruned_to_original = {idx: name for idx, name in enumerate(active_features_ordered)}

    # Only keep non-zero split counts in output
    active_split_counts = {f: int(split_counts[f]) for f in active_features_ordered}

    mapping = {
        'original_to_pruned': original_to_pruned,
        'pruned_to_original': pruned_to_original,
        'active_feature_names': active_features_ordered,
        'split_counts': active_split_counts,
        'total_original': len(all_features),
        'total_pruned': len(active_features_ordered),
        'pruning_ratio': len(active_features_ordered) / len(all_features) if all_features else 0,
    }

    return active_features_ordered, mapping


def save_pruned_artifacts(active_features, mapping, tf, output_dir='.'):
    """
    Save pruned feature artifacts for inference and interpretability.

    Saves:
        - features_{tf}_pruned.json: ordered list of active feature names
        - feature_mapping_{tf}.json: full mapping with split counts + index maps
    """
    # Pruned feature list (used by live_trader.py for inference)
    pruned_path = os.path.join(output_dir, f'features_{tf}_pruned.json')
    with open(pruned_path, 'w') as f:
        json.dump(active_features, f)

    # Full mapping (for interpretability and debugging)
    mapping_path = os.path.join(output_dir, f'feature_mapping_{tf}.json')
    # Convert int keys to str for JSON serialization
    serializable_mapping = {
        'active_feature_names': mapping['active_feature_names'],
        'split_counts': mapping['split_counts'],
        'total_original': mapping['total_original'],
        'total_pruned': mapping['total_pruned'],
        'pruning_ratio': mapping['pruning_ratio'],
        'original_to_pruned': mapping['original_to_pruned'],
        'pruned_to_original': {str(k): v for k, v in mapping['pruned_to_original'].items()},
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
    }
    with open(mapping_path, 'w') as f:
        json.dump(serializable_mapping, f, indent=2)

    return pruned_path, mapping_path


def prune_sparse_matrix(X_sparse, all_feature_names, active_features):
    """
    Prune a sparse CSR matrix to only include active feature columns.

    Args:
        X_sparse: scipy.sparse.csr_matrix (n_samples, n_original_features)
        all_feature_names: list of str, original feature names
        active_features: list of str, features to keep

    Returns:
        X_pruned: scipy.sparse.csr_matrix (n_samples, n_active_features)
    """
    active_set = set(active_features)
    col_indices = [i for i, name in enumerate(all_feature_names) if name in active_set]
    col_indices = np.array(col_indices, dtype=np.int32)

    # Extract columns from CSR (convert to CSC for efficient column slicing)
    X_csc = X_sparse.tocsc()
    X_pruned = X_csc[:, col_indices].tocsr()

    # Preserve int64 indptr for large NNZ (LightGBM PR #1719)
    if X_pruned.indptr.dtype != np.int64:
        X_pruned.indptr = X_pruned.indptr.astype(np.int64)

    return X_pruned


def retrain_pruned_model(X_train, y_train, active_features, params, tf,
                         X_val=None, y_val=None, output_dir='.'):
    """
    Retrain a LightGBM model on the pruned feature set for deployment.

    Args:
        X_train: sparse CSR matrix (n_samples, n_active_features)
        y_train: labels array
        active_features: list of str, pruned feature names
        params: dict, LightGBM training parameters
        tf: str, timeframe name
        X_val: optional validation matrix
        y_val: optional validation labels
        output_dir: str, directory for saving model

    Returns:
        model: lgb.Booster, trained on pruned features
        model_path: str, path to saved model
    """
    train_data = lgb.Dataset(
        X_train, label=y_train,
        feature_name=active_features,
        free_raw_data=False,
    )

    valid_sets = [train_data]
    valid_names = ['train']
    if X_val is not None and y_val is not None:
        val_data = lgb.Dataset(
            X_val, label=y_val,
            feature_name=active_features,
            free_raw_data=False,
            reference=train_data,
        )
        valid_sets.append(val_data)
        valid_names.append('valid')

    # Ensure deployment-safe params
    train_params = dict(params)
    train_params['feature_pre_filter'] = False  # NEVER filter rare features
    train_params['verbose'] = -1

    num_rounds = train_params.pop('num_boost_round', train_params.pop('n_estimators', 1000))

    model = lgb.train(
        train_params,
        train_data,
        num_boost_round=num_rounds,
        valid_sets=valid_sets,
        valid_names=valid_names,
    )

    model_path = os.path.join(output_dir, f'model_{tf}_pruned.json')
    model.save_model(model_path)
    print(f"  Saved pruned model: {model_path} ({len(active_features)} features)")

    return model, model_path


def analyze_pruning(mapping, tf):
    """Print pruning analysis summary."""
    total = mapping['total_original']
    pruned = mapping['total_pruned']
    ratio = mapping['pruning_ratio']
    splits = mapping['split_counts']

    # Categorize active features
    cross_prefixes = ('dx_', 'ax_', 'ax2_', 'ta2_', 'ex2_', 'sw_', 'hod_',
                      'mx_', 'vx_', 'asp_', 'mn_', 'pn_', 'cross_')
    active_crosses = [f for f in mapping['active_feature_names'] if f.startswith(cross_prefixes)]
    active_base = [f for f in mapping['active_feature_names'] if not f.startswith(cross_prefixes)]

    # Top features by split count
    top_by_splits = sorted(splits.items(), key=lambda x: x[1], reverse=True)[:20]

    print(f"\n{'='*60}")
    print(f"  FEATURE PRUNING ANALYSIS — {tf.upper()}")
    print(f"{'='*60}")
    print(f"  Original features:  {total:,}")
    print(f"  Active features:    {pruned:,} ({ratio*100:.2f}%)")
    print(f"  Pruned (unused):    {total - pruned:,} ({(1-ratio)*100:.2f}%)")
    print(f"  Active base:        {len(active_base):,}")
    print(f"  Active crosses:     {len(active_crosses):,}")
    print(f"\n  Top 20 features by split count:")
    for i, (feat, count) in enumerate(top_by_splits):
        print(f"    {i+1:2d}. {feat:<50s}  splits={count:,}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Extract active features from LightGBM model')
    parser.add_argument('--model', required=True, help='Path to model .json file')
    parser.add_argument('--tf', required=True, help='Timeframe (1w, 1d, 4h, 1h, 15m)')
    parser.add_argument('--output-dir', default='.', help='Output directory')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        sys.exit(1)

    print(f"Extracting active features from {args.model}...")
    active_features, mapping = extract_active_features(args.model)

    analyze_pruning(mapping, args.tf)

    pruned_path, mapping_path = save_pruned_artifacts(
        active_features, mapping, args.tf, args.output_dir
    )
    print(f"Saved: {pruned_path}")
    print(f"Saved: {mapping_path}")


if __name__ == '__main__':
    main()
