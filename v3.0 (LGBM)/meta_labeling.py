#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
meta_labeling.py — Secondary Classifier for Trade Filtering
=============================================================
Answers: "Given LightGBM says LONG/SHORT, should we actually take this trade?"

The base model predicts direction; the meta-model predicts whether
the base model is RIGHT. Uses CPCV OOS predictions as training data
(no leakage — base model never saw these samples).

Usage:
    from meta_labeling import train_meta_model, predict_meta
    meta_model = train_meta_model(oos_predictions, feature_data, tf_name)
    meta_prob = predict_meta(meta_model, base_prob, context_features)
"""

import numpy as np
import pandas as pd
import os
import pickle
import json

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression


def _build_meta_features(y_pred_probs, indices, feature_data, feature_cols):
    """Build meta-features for the meta-labeler.

    Meta-features:
    - Base model confidence (max probability)
    - Base model direction probabilities (prob_long, prob_short)
    - Probability margin (top - second)
    - Recent model performance (rolling accuracy proxy)

    Context features from the original data are NOT included here
    to keep the meta-model simple and avoid overfitting.
    """
    n = len(y_pred_probs)
    meta_X = np.zeros((n, 5), dtype=np.float32)

    # 1. Max probability (confidence)
    meta_X[:, 0] = np.max(y_pred_probs, axis=1)
    # 2. prob_long (class 2)
    meta_X[:, 1] = y_pred_probs[:, 2]
    # 3. prob_short (class 0)
    meta_X[:, 2] = y_pred_probs[:, 0]
    # 4. Probability margin (top - second best)
    sorted_probs = np.sort(y_pred_probs, axis=1)
    meta_X[:, 3] = sorted_probs[:, 2] - sorted_probs[:, 1]
    # 5. Entropy of prediction (low = confident, high = uncertain)
    probs_clipped = np.clip(y_pred_probs, 1e-10, 1.0)
    meta_X[:, 4] = -np.sum(probs_clipped * np.log(probs_clipped), axis=1)

    meta_feature_names = [
        'meta_confidence', 'meta_prob_long', 'meta_prob_short',
        'meta_prob_margin', 'meta_entropy',
    ]

    return meta_X, meta_feature_names


def train_meta_model(oos_predictions, feature_data=None, feature_cols=None,
                     tf_name='1h', model_type='logistic', db_dir='.'):
    """Train a meta-labeling model from CPCV OOS predictions.

    The meta-model is deliberately simple (logistic regression or
    max_depth=2 LightGBM) to avoid overfitting the meta-layer.

    Args:
        oos_predictions: list of CPCV OOS prediction dicts
        feature_data: full feature matrix (X_all) — optional, for context features
        feature_cols: feature column names — optional
        tf_name: timeframe name
        model_type: 'logistic' (recommended) or 'lgbm_shallow'
        db_dir: directory to save model

    Returns:
        dict with 'model', 'meta_feature_names', 'threshold', 'metrics'
    """
    # Auto-load from file if path provided
    if isinstance(oos_predictions, str):
        with open(oos_predictions, 'rb') as f:
            oos_predictions = pickle.load(f)

    # Concatenate all OOS predictions
    all_y_true = []
    all_y_pred_probs = []
    all_indices = []

    for pred in oos_predictions:
        all_y_true.append(pred['y_true'])
        all_y_pred_probs.append(pred['y_pred_probs'])
        all_indices.append(pred.get('test_indices', np.arange(len(pred['y_true']))))

    if not all_y_true:
        return None

    y_true = np.concatenate(all_y_true)
    y_pred_probs = np.concatenate(all_y_pred_probs)
    indices = np.concatenate(all_indices)

    # Create meta-labels: 1 if base model's predicted direction was correct
    pred_labels = np.argmax(y_pred_probs, axis=1)
    meta_y = (pred_labels == y_true).astype(np.int32)

    # Only train on samples where the model actually predicted a direction
    # (not FLAT — we only want to filter LONG/SHORT trades)
    trade_mask = pred_labels != 1  # not FLAT
    if trade_mask.sum() < 50:
        print(f"  Meta-labeling: too few directional predictions ({trade_mask.sum()}), skipping")
        return None

    meta_X, meta_feature_names = _build_meta_features(
        y_pred_probs[trade_mask], indices[trade_mask], feature_data, feature_cols
    )
    meta_y_trades = meta_y[trade_mask]

    # Split meta data with its own time-based split (last 30% for validation)
    n_meta = len(meta_X)
    split_point = int(n_meta * 0.7)
    X_meta_train = meta_X[:split_point]
    y_meta_train = meta_y_trades[:split_point]
    X_meta_test = meta_X[split_point:]
    y_meta_test = meta_y_trades[split_point:]

    print(f"  Meta-labeling: {n_meta} directional trades, "
          f"train={len(X_meta_train)} test={len(X_meta_test)}, "
          f"base accuracy={meta_y_trades.mean():.3f}")

    # Train meta-model (deliberately simple)
    if model_type == 'logistic':
        model = LogisticRegression(C=1.0, max_iter=500, class_weight='balanced')
        model.fit(X_meta_train, y_meta_train)
        meta_probs_test = model.predict_proba(X_meta_test)[:, 1]
    elif model_type == 'lgbm_shallow' and lgb is not None:
        dtrain = lgb.Dataset(X_meta_train, label=y_meta_train,
                             feature_name=meta_feature_names, free_raw_data=False)
        dtest = lgb.Dataset(X_meta_test, label=y_meta_test,
                            feature_name=meta_feature_names, free_raw_data=False)
        params = {
            'objective': 'binary',
            'max_depth': 2,
            'num_leaves': 4,
            'learning_rate': 0.1,
            'min_data_in_leaf': 20,
            'lambda_l2': 5.0,
            'lambda_l1': 2.0,
            'metric': 'auc',
            'verbosity': -1,
        }
        model = lgb.train(params, dtrain, num_boost_round=100,
                          valid_sets=[dtest], valid_names=['test'],
                          callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
        meta_probs_test = model.predict(X_meta_test)
    else:
        model = LogisticRegression(C=1.0, max_iter=500, class_weight='balanced')
        model.fit(X_meta_train, y_meta_train)
        meta_probs_test = model.predict_proba(X_meta_test)[:, 1]

    # Evaluate
    try:
        auc = roc_auc_score(y_meta_test, meta_probs_test)
    except ValueError:
        auc = 0.5
    meta_pred = (meta_probs_test > 0.5).astype(int)
    meta_acc = accuracy_score(y_meta_test, meta_pred)
    base_acc = y_meta_test.mean()

    # Find optimal threshold (maximize accuracy)
    best_thresh = 0.5
    best_meta_acc = meta_acc
    for thresh in np.arange(0.3, 0.7, 0.05):
        acc = accuracy_score(y_meta_test, (meta_probs_test > thresh).astype(int))
        if acc > best_meta_acc:
            best_meta_acc = acc
            best_thresh = thresh

    print(f"  Meta-labeling: AUC={auc:.3f}, acc={meta_acc:.3f} (base={base_acc:.3f}), "
          f"best_thresh={best_thresh:.2f}")

    # Save meta-model
    meta_path = os.path.join(db_dir, f'meta_model_{tf_name}.pkl')
    with open(meta_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'model_type': model_type,
            'meta_feature_names': meta_feature_names,
            'threshold': best_thresh,
        }, f)
    print(f"  Meta-model saved: {meta_path}")

    return {
        'model': model,
        'model_type': model_type,
        'meta_feature_names': meta_feature_names,
        'threshold': best_thresh,
        'metrics': {
            'auc': auc,
            'accuracy': meta_acc,
            'base_accuracy': base_acc,
            'best_threshold': best_thresh,
            'n_train': len(X_meta_train),
            'n_test': len(X_meta_test),
        },
    }


def predict_meta(meta_result, base_pred_probs):
    """Get meta-labeling probability for new predictions.

    Args:
        meta_result: dict from train_meta_model()
        base_pred_probs: (N, 3) array of base model probabilities

    Returns:
        meta_probs: (N,) array of P(base model is correct)
        take_trade: (N,) boolean array (meta_prob > threshold)
    """
    if meta_result is None:
        return np.ones(len(base_pred_probs)), np.ones(len(base_pred_probs), dtype=bool)

    meta_X, _ = _build_meta_features(
        base_pred_probs,
        np.arange(len(base_pred_probs)),
        None, None,
    )

    model = meta_result['model']
    threshold = meta_result['threshold']

    if meta_result['model_type'] == 'logistic':
        meta_probs = model.predict_proba(meta_X)[:, 1]
    elif meta_result['model_type'] == 'lgbm_shallow':
        meta_probs = model.predict(meta_X)
    else:
        meta_probs = np.ones(len(base_pred_probs))

    take_trade = meta_probs > threshold
    return meta_probs, take_trade


def load_oos_predictions(mode, tf, db_dir='.'):
    """Load OOS predictions from the standard file path.

    Args:
        mode: 'unified', 'per-asset', or 'prod_{symbol}'
        tf: timeframe name ('1d', '1h', etc.)
        db_dir: directory containing the .pkl files

    Returns:
        list of OOS prediction dicts, or None if file not found
    """
    path = os.path.join(db_dir, f'oos_predictions_{mode}_{tf}.pkl')
    if not os.path.exists(path):
        print(f"  OOS predictions not found: {path}")
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    print("=== Meta-Labeling Test ===")
    np.random.seed(42)

    # Simulate CPCV OOS predictions (base model ~55% accurate)
    fake_oos = []
    for i in range(10):
        n = 300
        y_true = np.random.randint(0, 3, n)
        y_probs = np.random.dirichlet([1, 1, 1], n)
        for j in range(n):
            y_probs[j, y_true[j]] += 0.2  # slight edge
        y_probs /= y_probs.sum(axis=1, keepdims=True)
        fake_oos.append({
            'path': i,
            'y_true': y_true,
            'y_pred_probs': y_probs,
            'test_indices': np.arange(i * n, (i + 1) * n),
        })

    result = train_meta_model(fake_oos, tf_name='test', db_dir='.')
    if result:
        print(f"\n  Model type: {result['model_type']}")
        print(f"  AUC: {result['metrics']['auc']:.3f}")
        print(f"  Threshold: {result['threshold']:.2f}")

        # Test prediction
        test_probs = np.random.dirichlet([1, 1, 1], 20)
        meta_probs, take = predict_meta(result, test_probs)
        print(f"\n  Test: {take.sum()}/{len(take)} trades approved")
        print(f"  Meta prob range: [{meta_probs.min():.3f}, {meta_probs.max():.3f}]")

        # Cleanup test file
        os.remove('meta_model_test.pkl')
