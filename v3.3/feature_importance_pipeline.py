#!/usr/bin/env python3
"""
feature_importance_pipeline.py -- 6M-Scale Feature Importance Analysis
======================================================================
Designed for LightGBM CPCV with 6M+ sparse binary cross features.

Pipeline stages:
  1. Gain importance extraction across all CPCV folds
  2. Stability selection (principled N via elbow detection)
  3. Sign consistency from tree structure traversal
  4. Composite signal strength = gain x stability x sign_consistency
  5. Permutation importance on top-1K (optional, ~1 min)
  6. SHAP on top-200 (optional, ~30 sec)
  7. Random injection test for overfitting detection
  8. Cross-TF importance comparison
  9. Visualization (heatmap, Pareto, category breakdown, overfitting dashboard)
  10. Save results to JSON

Usage:
    from feature_importance_pipeline import FeatureImportancePipeline

    pipeline = FeatureImportancePipeline(
        fold_boosters=[model_fold0, model_fold1, model_fold2, model_fold3],
        feature_names=feature_name_list,
        tf_name='1w',
        output_dir='/workspace/v3.3',
    )
    results = pipeline.run()
"""

import os
import sys
import time
import json
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import stats
from scipy.stats import rankdata, kendalltau, binom

warnings.filterwarnings('ignore')


class FeatureImportancePipeline:
    """6M-scale feature importance pipeline for LightGBM CPCV models."""

    def __init__(self, fold_boosters, feature_names, tf_name,
                 output_dir='.', X_val=None, y_val=None,
                 feature_categories=None, other_tf_importances=None):
        """
        Args:
            fold_boosters: list of lgb.Booster objects, one per CPCV fold
            feature_names: list of str, feature names (length = n_features)
            tf_name: str, timeframe name ('1w', '1d', '4h', etc.)
            output_dir: str, directory for saving results
            X_val: optional sparse/dense matrix for permutation/SHAP (n_samples, n_features)
            y_val: optional labels for permutation importance
            feature_categories: optional dict {feature_name: category_str}
                                categories like 'TECHNICAL', 'ESOTERIC', 'MACRO', 'CROSS_TECH_ESO'
            other_tf_importances: optional dict {tf_name: np.array of gains} for cross-TF analysis
        """
        self.boosters = fold_boosters
        self.feature_names = np.array(feature_names)
        self.n_features = len(feature_names)
        self.n_folds = len(fold_boosters)
        self.tf_name = tf_name
        self.output_dir = output_dir
        self.X_val = X_val
        self.y_val = y_val
        self.feature_categories = feature_categories or {}
        self.other_tf_importances = other_tf_importances or {}

        # Results accumulator
        self.results = {}
        self.timings = {}

    def log(self, msg):
        print(f"[FI-{self.tf_name.upper()}] {msg}")
        sys.stdout.flush()

    # ================================================================
    # STAGE 1: Gain importance extraction
    # ================================================================
    def _extract_gains(self):
        """Extract gain importance from all fold boosters."""
        t0 = time.time()
        self.log(f"Stage 1: Extracting gain importance from {self.n_folds} folds...")

        fold_gains = []
        for i, booster in enumerate(self.boosters):
            gains = booster.feature_importance(importance_type='gain')
            if len(gains) != self.n_features:
                self.log(f"  WARNING: Fold {i} has {len(gains)} features, expected {self.n_features}")
                # Pad or truncate
                padded = np.zeros(self.n_features)
                n = min(len(gains), self.n_features)
                padded[:n] = gains[:n]
                gains = padded
            fold_gains.append(gains.astype(np.float64))

        self.fold_gains = fold_gains
        all_gains = np.stack(fold_gains, axis=1)  # (n_features, n_folds)
        self.mean_gains = all_gains.mean(axis=1)

        n_nonzero_per_fold = [(g > 0).sum() for g in fold_gains]
        n_nonzero_any = (all_gains.max(axis=1) > 0).sum()

        self.log(f"  Non-zero gain per fold: {n_nonzero_per_fold}")
        self.log(f"  Non-zero in ANY fold: {n_nonzero_any:,} / {self.n_features:,} "
                 f"({100*n_nonzero_any/self.n_features:.3f}%)")

        # Gain concentration
        sorted_mean = np.sort(self.mean_gains)[::-1]
        if sorted_mean.sum() > 0:
            top1pct_n = max(1, n_nonzero_any // 100)
            top1pct_share = sorted_mean[:top1pct_n].sum() / sorted_mean.sum()
            self.log(f"  Top 1% of non-zero features hold {top1pct_share:.1%} of total gain")

        self.results['n_nonzero_per_fold'] = n_nonzero_per_fold
        self.results['n_nonzero_any_fold'] = int(n_nonzero_any)
        self.timings['stage1_gain'] = time.time() - t0
        self.log(f"  Done in {self.timings['stage1_gain']:.1f}s")

    # ================================================================
    # STAGE 2: Stability selection with principled N
    # ================================================================
    def _stability_selection(self):
        """Count how many folds each feature appears in top-N by gain."""
        t0 = time.time()
        self.log("Stage 2: Stability selection...")

        # --- Find principled N via elbow detection ---
        sorted_mean = np.sort(self.mean_gains)[::-1]
        nonzero_gains = sorted_mean[sorted_mean > 0]

        if len(nonzero_gains) < 10:
            self.log("  WARNING: Fewer than 10 non-zero features. Using N=10.")
            N = 10
        else:
            # Log-scale elbow detection
            log_gain = np.log1p(nonzero_gains)
            grad = np.gradient(log_gain)
            slope_threshold = 0.01 * np.abs(grad).max()
            elbow_candidates = np.where(np.abs(grad) < slope_threshold)[0]
            N_elbow = elbow_candidates[0] if len(elbow_candidates) > 0 else len(nonzero_gains)

            # Meinshausen-Buhlmann false-discovery bound
            N_meinshausen = int(np.sqrt((2 * self.n_folds - 1) * self.n_features))

            N = max(500, min(N_elbow, N_meinshausen, 5000))
            self.log(f"  Elbow N={N_elbow:,}, Meinshausen N={N_meinshausen:,}, Using N={N:,}")

        self.stability_N = N

        # --- Count fold appearances in top-N (using O(n) argpartition) ---
        topN_per_fold = []
        for gains in self.fold_gains:
            nonzero_mask = gains > 0
            n_nonzero = nonzero_mask.sum()
            if n_nonzero <= N:
                topN_idx = set(np.where(nonzero_mask)[0].tolist())
            else:
                topN_idx_arr = np.argpartition(gains, -N)[-N:]
                topN_idx = set(topN_idx_arr[gains[topN_idx_arr] > 0].tolist())
            topN_per_fold.append(topN_idx)

        # Union of all candidates
        all_candidates = set.union(*topN_per_fold) if topN_per_fold else set()

        # Build stability counts
        stability_data = []
        for feat_idx in all_candidates:
            n_folds_in = sum(feat_idx in fold_set for fold_set in topN_per_fold)
            fold_gains_feat = [self.fold_gains[k][feat_idx] for k in range(self.n_folds)]
            mean_gain = np.mean(fold_gains_feat)
            gain_cv = np.std(fold_gains_feat) / (mean_gain + 1e-12)

            stability_data.append({
                'feature_idx': int(feat_idx),
                'feature': self.feature_names[feat_idx],
                'stability_count': n_folds_in,
                'stability_score': n_folds_in / self.n_folds,
                'mean_gain': float(mean_gain),
                'gain_cv': float(gain_cv),
                'category': self.feature_categories.get(self.feature_names[feat_idx], 'UNKNOWN'),
            })

        self.stability_df = pd.DataFrame(stability_data)
        if not self.stability_df.empty:
            self.stability_df.sort_values('mean_gain', ascending=False, inplace=True)

        n_stable = (self.stability_df['stability_score'] >= 0.75).sum() if not self.stability_df.empty else 0
        self.log(f"  Candidates in union of top-{N}: {len(all_candidates):,}")
        self.log(f"  Stable (in >= 75% folds): {n_stable:,}")

        self.results['stability_N'] = N
        self.results['n_candidates'] = len(all_candidates)
        self.results['n_stable_75pct'] = int(n_stable)
        self.timings['stage2_stability'] = time.time() - t0
        self.log(f"  Done in {self.timings['stage2_stability']:.1f}s")

    # ================================================================
    # STAGE 3: Sign consistency from tree traversal
    # ================================================================
    def _sign_consistency(self):
        """Extract directional effect (sign) of each feature from tree structure."""
        t0 = time.time()
        self.log("Stage 3: Extracting sign consistency from tree structures...")

        # Only analyze stable candidates (top features from stability selection)
        if self.stability_df.empty:
            self.log("  WARNING: No stable features found. Skipping sign analysis.")
            self.timings['stage3_sign'] = time.time() - t0
            return

        # Focus on features stable in >= 50% of folds
        candidates = set(
            self.stability_df[self.stability_df['stability_score'] >= 0.5]['feature_idx'].tolist()
        )
        self.log(f"  Analyzing sign for {len(candidates):,} candidate features...")

        fold_signs = defaultdict(list)  # {feat_idx: [(sign, weighted_effect), ...]}

        for fold_i, booster in enumerate(self.boosters):
            t_fold = time.time()
            model_dict = booster.dump_model()
            n_splits_found = 0

            def get_subtree_leaf_values(node):
                if 'leaf_value' in node:
                    return [node['leaf_value']]
                vals = []
                if 'left_child' in node:
                    vals.extend(get_subtree_leaf_values(node['left_child']))
                if 'right_child' in node:
                    vals.extend(get_subtree_leaf_values(node['right_child']))
                return vals

            def traverse(node):
                nonlocal n_splits_found
                if 'leaf_value' in node:
                    return
                feat_idx = node.get('split_feature')
                if feat_idx is not None and feat_idx in candidates:
                    left_vals = get_subtree_leaf_values(node.get('left_child', {}))
                    right_vals = get_subtree_leaf_values(node.get('right_child', {}))
                    if left_vals and right_vals:
                        mean_left = np.mean(left_vals)
                        mean_right = np.mean(right_vals)
                        split_gain = node.get('split_gain', 1.0)
                        effect = mean_right - mean_left
                        fold_signs[feat_idx].append((
                            fold_i,
                            int(np.sign(effect)) if effect != 0 else 0,
                            effect * split_gain,
                            split_gain,
                        ))
                        n_splits_found += 1
                if 'left_child' in node:
                    traverse(node['left_child'])
                if 'right_child' in node:
                    traverse(node['right_child'])

            for tree_info in model_dict.get('tree_info', []):
                traverse(tree_info.get('tree_structure', {}))

            self.log(f"    Fold {fold_i}: {n_splits_found:,} splits analyzed ({time.time()-t_fold:.1f}s)")

        # Aggregate: per-feature sign consistency across folds
        sign_results = {}
        for feat_idx, entries in fold_signs.items():
            # Group by fold, get per-fold dominant sign
            per_fold = defaultdict(list)
            for fold_i, sign, weighted_eff, gain in entries:
                per_fold[fold_i].append((sign, weighted_eff, gain))

            fold_dominant_signs = []
            for fold_i, splits in per_fold.items():
                total_gain = sum(g for _, _, g in splits)
                if total_gain > 0:
                    weighted_mean = sum(we for _, we, _ in splits) / total_gain
                    fold_dominant_signs.append(int(np.sign(weighted_mean)))

            if not fold_dominant_signs:
                continue

            n_positive = sum(1 for s in fold_dominant_signs if s > 0)
            n_negative = sum(1 for s in fold_dominant_signs if s < 0)
            n_total = len(fold_dominant_signs)
            consistency = max(n_positive, n_negative) / n_total
            dominant = 1 if n_positive >= n_negative else -1

            sign_results[feat_idx] = {
                'sign_consistency': float(consistency),
                'dominant_sign': int(dominant),
                'n_folds_with_sign': n_total,
                'n_positive': n_positive,
                'n_negative': n_negative,
            }

        self.sign_results = sign_results
        self.log(f"  Sign extracted for {len(sign_results):,} features")

        # Add to stability_df
        if not self.stability_df.empty:
            self.stability_df['sign_consistency'] = self.stability_df['feature_idx'].map(
                lambda x: sign_results.get(x, {}).get('sign_consistency', 0.5)
            )
            self.stability_df['dominant_sign'] = self.stability_df['feature_idx'].map(
                lambda x: sign_results.get(x, {}).get('dominant_sign', 0)
            )

        self.timings['stage3_sign'] = time.time() - t0
        self.log(f"  Done in {self.timings['stage3_sign']:.1f}s")

    # ================================================================
    # STAGE 4: Composite signal strength
    # ================================================================
    def _composite_signal_strength(self):
        """Compute gain x stability x sign_consistency composite score."""
        t0 = time.time()
        self.log("Stage 4: Computing composite signal strength...")

        if self.stability_df.empty:
            self.log("  WARNING: No data for composite scoring.")
            self.timings['stage4_composite'] = time.time() - t0
            return

        df = self.stability_df.copy()

        # Gain component: rank percentile of mean log-gain (handles skew)
        mean_log_gain = np.log1p(df['mean_gain'].values)
        nonzero = mean_log_gain > 0
        gain_rank = np.zeros(len(df))
        if nonzero.sum() > 0:
            ranks = rankdata(mean_log_gain[nonzero], method='average')
            gain_rank[nonzero] = ranks / ranks.max()
        df['gain_score'] = gain_rank

        # Stability component: already in [0, 1]
        # Sign consistency: rescale from [0.5, 1] to [0, 1]
        df['sign_score'] = ((df['sign_consistency'] - 0.5) / 0.5).clip(0, 1)

        # Composite: multiplicative (any zero kills it)
        df['signal_strength'] = df['gain_score'] * df['stability_score'] * df['sign_score']
        df['signed_signal_strength'] = df['signal_strength'] * df['dominant_sign']

        df.sort_values('signal_strength', ascending=False, inplace=True)

        self.signal_df = df

        # Diagnostics
        n_pass_all = (
            (df['gain_score'] > 0.1) &
            (df['stability_score'] >= 0.75) &
            (df['sign_score'] >= 0.5)
        ).sum()
        self.log(f"  Features passing all 3 filters (gain>0.1, stab>=0.75, sign>=0.5): {n_pass_all:,}")

        # Top 20 preview
        top20 = df.head(20)[['feature', 'mean_gain', 'stability_score',
                              'sign_consistency', 'dominant_sign', 'signal_strength', 'category']]
        self.log(f"  Top 20 by signal strength:\n{top20.to_string(index=False)}")

        self.results['n_pass_all_filters'] = int(n_pass_all)
        self.timings['stage4_composite'] = time.time() - t0
        self.log(f"  Done in {self.timings['stage4_composite']:.1f}s")

    # ================================================================
    # STAGE 5: Permutation importance on top-1K (optional)
    # ================================================================
    def _permutation_importance(self, top_k=1000, n_repeats=10, max_val_rows=15000):
        """Run permutation importance on top-K features. Requires X_val, y_val."""
        if self.X_val is None or self.y_val is None:
            self.log("Stage 5: SKIPPED (no X_val/y_val provided)")
            return

        t0 = time.time()
        self.log(f"Stage 5: Permutation importance on top-{top_k} features...")

        import lightgbm as lgb
        from sklearn.inspection import permutation_importance
        from scipy import sparse as sp_sparse

        # Get top-K feature indices by signal strength
        if hasattr(self, 'signal_df') and not self.signal_df.empty:
            top_indices = self.signal_df.head(top_k)['feature_idx'].values
        else:
            top_indices = np.argsort(self.mean_gains)[::-1][:top_k]

        # Subset validation data
        if sp_sparse.issparse(self.X_val):
            X_sub = self.X_val[:, top_indices]
        else:
            X_sub = self.X_val[:, top_indices]

        # Cap rows for speed
        n_rows = min(max_val_rows, X_sub.shape[0])
        if n_rows < X_sub.shape[0]:
            idx = np.random.choice(X_sub.shape[0], n_rows, replace=False)
            X_sub = X_sub[idx]
            y_sub = self.y_val[idx]
        else:
            y_sub = self.y_val

        self.log(f"  Retraining on {X_sub.shape[0]} rows x {X_sub.shape[1]} features...")

        # Retrain clean model on subset
        model_sub = lgb.LGBMClassifier(
            n_estimators=300, num_leaves=31, n_jobs=-1,
            verbose=-1, force_col_wise=True, max_bin=7,
            feature_pre_filter=False,
        )
        model_sub.fit(X_sub, y_sub)

        self.log(f"  Running permutation importance (n_repeats={n_repeats})...")
        result = permutation_importance(
            model_sub, X_sub, y_sub,
            n_repeats=n_repeats, n_jobs=-1,
            scoring='accuracy', random_state=42,
        )

        perm_df = pd.DataFrame({
            'feature_idx': top_indices,
            'feature': self.feature_names[top_indices],
            'perm_importance_mean': result.importances_mean,
            'perm_importance_std': result.importances_std,
        }).sort_values('perm_importance_mean', ascending=False)

        # Significant features: mean > 2*std > 0
        significant = perm_df[
            perm_df['perm_importance_mean'] - 2 * perm_df['perm_importance_std'] > 0
        ]
        self.log(f"  Significant by permutation: {len(significant):,} / {len(perm_df):,}")

        self.perm_df = perm_df
        self.results['n_perm_significant'] = len(significant)
        self.timings['stage5_permutation'] = time.time() - t0
        self.log(f"  Done in {self.timings['stage5_permutation']:.1f}s")

    # ================================================================
    # STAGE 6: SHAP on top-200 (optional)
    # ================================================================
    def _shap_analysis(self, top_k=200, sample_rows=10000):
        """Run SHAP TreeExplainer on top-K features. Requires X_val."""
        if self.X_val is None:
            self.log("Stage 6: SKIPPED (no X_val provided)")
            return

        t0 = time.time()
        self.log(f"Stage 6: SHAP analysis on top-{top_k} features...")

        try:
            import shap
            import lightgbm as lgb
            from scipy import sparse as sp_sparse
        except ImportError:
            self.log("  SKIPPED: shap not installed")
            return

        # Get top-K indices
        if hasattr(self, 'signal_df') and not self.signal_df.empty:
            top_indices = self.signal_df.head(top_k)['feature_idx'].values
        else:
            top_indices = np.argsort(self.mean_gains)[::-1][:top_k]

        # Subset and sample
        if sp_sparse.issparse(self.X_val):
            X_sub = self.X_val[:, top_indices]
            if X_sub.shape[0] > sample_rows:
                idx = np.random.choice(X_sub.shape[0], sample_rows, replace=False)
                X_sub = X_sub[idx].toarray()
            else:
                X_sub = X_sub.toarray()
        else:
            X_sub = self.X_val[:, top_indices]
            if X_sub.shape[0] > sample_rows:
                idx = np.random.choice(X_sub.shape[0], sample_rows, replace=False)
                X_sub = X_sub[idx]

        # Retrain on subset
        model_sub = lgb.LGBMClassifier(
            n_estimators=300, num_leaves=31, n_jobs=-1,
            verbose=-1, force_col_wise=True, max_bin=7,
        )
        model_sub.fit(X_sub, self.y_val[:len(X_sub)] if len(self.y_val) > len(X_sub) else self.y_val)

        self.log(f"  Computing SHAP values for {X_sub.shape[0]} samples x {X_sub.shape[1]} features...")
        explainer = shap.TreeExplainer(model_sub)
        shap_values = explainer.shap_values(X_sub)

        # For multiclass, shap_values is a list of arrays per class
        if isinstance(shap_values, list):
            # Use class 2 (LONG) - class 0 (SHORT) for directional SHAP
            shap_importance = np.abs(shap_values[2]).mean(axis=0) if len(shap_values) > 2 else np.abs(shap_values[0]).mean(axis=0)
        else:
            shap_importance = np.abs(shap_values).mean(axis=0)

        shap_df = pd.DataFrame({
            'feature_idx': top_indices,
            'feature': self.feature_names[top_indices],
            'shap_importance': shap_importance,
        }).sort_values('shap_importance', ascending=False)

        self.shap_df = shap_df
        self.log(f"  Top 10 by SHAP:")
        for _, row in shap_df.head(10).iterrows():
            self.log(f"    {row['feature']}: {row['shap_importance']:.6f}")

        self.timings['stage6_shap'] = time.time() - t0
        self.log(f"  Done in {self.timings['stage6_shap']:.1f}s")

    # ================================================================
    # STAGE 7: Random injection overfitting test
    # ================================================================
    def _random_injection_test(self, n_random=100, top_k=1000):
        """Check if random features can infiltrate the top-K."""
        t0 = time.time()
        self.log(f"Stage 7: Random injection test ({n_random} random features)...")

        if self.X_val is None or self.y_val is None:
            self.log("  SKIPPED (no X_val/y_val provided)")
            return

        import lightgbm as lgb
        from scipy import sparse as sp_sparse

        # Inject random binary columns
        np.random.seed(42)
        n_rows = self.X_val.shape[0]
        # Random binary features with ~5% density (similar to real cross features)
        random_cols = (np.random.random((n_rows, n_random)) < 0.05).astype(np.float32)

        if sp_sparse.issparse(self.X_val):
            from scipy.sparse import hstack, csr_matrix
            X_augmented = hstack([self.X_val, csr_matrix(random_cols)], format='csr')
        else:
            X_augmented = np.hstack([self.X_val, random_cols])

        # Feature names for augmented
        random_names = [f'__RANDOM_{i}__' for i in range(n_random)]

        self.log(f"  Training on augmented data ({X_augmented.shape[1]:,} features)...")
        model_aug = lgb.LGBMClassifier(
            n_estimators=300, num_leaves=31, n_jobs=-1,
            verbose=-1, force_col_wise=True, max_bin=7,
            colsample_bytree=0.9,
        )
        model_aug.fit(X_augmented, self.y_val)

        gains = model_aug.booster_.feature_importance(importance_type='gain')

        # Check random feature rankings
        random_gains = gains[-n_random:]
        real_gains = gains[:-n_random]

        # Rank all features
        all_ranks = rankdata(-gains)  # higher gain = lower rank (rank 1 = best)
        random_ranks = all_ranks[-n_random:]

        n_in_topk = (random_ranks <= top_k).sum()
        expected_by_chance = n_random * top_k / len(gains)
        noise_floor_95 = np.percentile(random_gains[random_gains > 0], 95) if (random_gains > 0).any() else 0

        overfit_signal = n_in_topk > max(3 * expected_by_chance, 1)

        self.log(f"  Random features injected: {n_random}")
        self.log(f"  Random features in top-{top_k}: {n_in_topk}")
        self.log(f"  Expected by chance: {expected_by_chance:.1f}")
        self.log(f"  Noise floor (95th pct gain): {noise_floor_95:.6f}")
        self.log(f"  Overfitting signal: {'YES -- WARNING' if overfit_signal else 'NO -- clean'}")

        self.results['random_injection'] = {
            'n_random': n_random,
            'n_in_topk': int(n_in_topk),
            'expected_by_chance': round(expected_by_chance, 2),
            'noise_floor_95pct': float(noise_floor_95),
            'overfit_signal': overfit_signal,
        }
        self.timings['stage7_injection'] = time.time() - t0
        self.log(f"  Done in {self.timings['stage7_injection']:.1f}s")

    # ================================================================
    # STAGE 8: Cross-TF importance comparison
    # ================================================================
    def _cross_tf_analysis(self):
        """Compare feature importance across timeframes."""
        if not self.other_tf_importances:
            self.log("Stage 8: SKIPPED (no other TF importances provided)")
            return

        t0 = time.time()
        self.log(f"Stage 8: Cross-TF importance comparison...")

        current_gains = self.mean_gains
        tau_results = {}

        for other_tf, other_gains in self.other_tf_importances.items():
            if len(other_gains) != self.n_features:
                self.log(f"  WARNING: {other_tf} has {len(other_gains)} features, expected {self.n_features}. Skipping.")
                continue
            tau, p = kendalltau(current_gains, other_gains)
            tau_results[other_tf] = {'tau': float(tau), 'p_value': float(p)}
            self.log(f"  {self.tf_name} vs {other_tf}: tau={tau:.4f}, p={p:.4f}")

        # Cross-TF universal features: nonzero in this TF AND at least one other
        if tau_results:
            other_arrays = [v for v in self.other_tf_importances.values()
                           if len(v) == self.n_features]
            if other_arrays:
                all_tf_gains = np.stack([current_gains] + other_arrays, axis=1)
                nonzero_all = (all_tf_gains > 0).all(axis=1)
                n_universal = nonzero_all.sum()
                self.log(f"  Features with non-zero gain in ALL {len(other_arrays)+1} TFs: {n_universal:,}")

                # Geometric mean composite for universal features
                if n_universal > 0:
                    eps = 1e-10
                    normed = all_tf_gains / (all_tf_gains.max(axis=0, keepdims=True) + eps)
                    geo_mean = np.exp(np.mean(np.log(normed + eps), axis=1))
                    top_universal = np.argsort(geo_mean)[::-1][:20]
                    self.log(f"  Top 20 universal features (geometric mean):")
                    for idx in top_universal:
                        if geo_mean[idx] > eps:
                            self.log(f"    {self.feature_names[idx]}: geo_mean={geo_mean[idx]:.6f}")

        self.results['cross_tf_tau'] = tau_results
        self.timings['stage8_cross_tf'] = time.time() - t0
        self.log(f"  Done in {self.timings['stage8_cross_tf']:.1f}s")

    # ================================================================
    # STAGE 9: Esoteric validation (binomial baseline)
    # ================================================================
    def _esoteric_validation(self, top_k=1000):
        """Test if esoteric features appear more than expected by chance."""
        t0 = time.time()
        self.log("Stage 9: Esoteric validation (binomial baseline)...")

        if not self.feature_categories or not hasattr(self, 'signal_df') or self.signal_df.empty:
            self.log("  SKIPPED (no categories or no signal data)")
            return

        # Count features per category in total and in top-K
        total_counts = defaultdict(int)
        for cat in self.feature_categories.values():
            total_counts[cat] += 1

        top_features = self.signal_df.head(top_k)
        topk_counts = top_features['category'].value_counts().to_dict()
        actual_topk = len(top_features)

        self.log(f"\n  Category distribution (top-{actual_topk} vs total {self.n_features:,}):")
        self.log(f"  {'Category':<25} {'Total':>10} {'Base%':>8} {'Top-K':>8} {'Expected':>10} {'Lift':>8}")
        self.log(f"  {'-'*75}")

        esoteric_results = {}
        for cat in sorted(total_counts.keys()):
            n_total = total_counts[cat]
            n_topk = topk_counts.get(cat, 0)
            base_rate = n_total / self.n_features
            expected = actual_topk * base_rate
            lift = n_topk / expected if expected > 0 else 0

            # Binomial test: is observed count significantly above expected?
            p_value = 1 - binom.cdf(n_topk - 1, actual_topk, base_rate)

            esoteric_results[cat] = {
                'n_total': n_total,
                'base_rate': round(base_rate, 4),
                'n_in_topk': n_topk,
                'expected': round(expected, 1),
                'lift': round(lift, 2),
                'p_value': round(p_value, 4),
            }

            sig = '*' if p_value < 0.05 else ' '
            self.log(f"  {cat:<25} {n_total:>10,} {base_rate:>7.1%} {n_topk:>8} {expected:>10.1f} {lift:>7.2f}x {sig}")

        self.results['esoteric_validation'] = esoteric_results
        self.timings['stage9_esoteric'] = time.time() - t0
        self.log(f"  Done in {self.timings['stage9_esoteric']:.1f}s")

    # ================================================================
    # STAGE 10: Overfitting diagnostics
    # ================================================================
    def _overfitting_diagnostics(self):
        """Compute entropy, Jaccard, and concentration metrics."""
        t0 = time.time()
        self.log("Stage 10: Overfitting diagnostics...")

        diagnostics = {}

        # --- Importance entropy ---
        nonzero_gains = self.mean_gains[self.mean_gains > 0]
        if len(nonzero_gains) > 1:
            from scipy.stats import entropy as scipy_entropy
            probs = nonzero_gains / nonzero_gains.sum()
            H = scipy_entropy(probs)
            H_max = np.log(len(probs))
            H_norm = H / H_max if H_max > 0 else 0

            # Gini concentration
            n = len(probs)
            sorted_p = np.sort(probs)
            gini = (2 * np.sum(np.arange(1, n+1) * sorted_p) - (n + 1)) / n

            effective_n = np.exp(H)

            diagnostics['entropy'] = {
                'H': round(H, 4),
                'H_normalized': round(H_norm, 4),
                'gini_concentration': round(gini, 4),
                'effective_features': round(effective_n, 1),
                'interpretation': (
                    'HEALTHY' if 0.4 <= H_norm <= 0.7 else
                    'OVER-CONCENTRATED (possible memorization)' if H_norm < 0.3 else
                    'NEAR-UNIFORM (possible noise fitting)' if H_norm > 0.8 else
                    'BORDERLINE'
                ),
            }
            self.log(f"  Entropy: H_norm={H_norm:.3f}, Gini={gini:.3f}, "
                     f"Effective N={effective_n:.0f} -- {diagnostics['entropy']['interpretation']}")

        # --- Jaccard similarity of top-K sets across folds ---
        top_k_jaccard = min(1000, self.n_features)
        fold_sets = []
        for gains in self.fold_gains:
            n_nz = (gains > 0).sum()
            k = min(top_k_jaccard, n_nz)
            if k > 0:
                topk_idx = set(np.argpartition(gains, -k)[-k:].tolist())
                fold_sets.append(topk_idx)
            else:
                fold_sets.append(set())

        jaccard_values = []
        for i in range(len(fold_sets)):
            for j in range(i+1, len(fold_sets)):
                union = fold_sets[i] | fold_sets[j]
                inter = fold_sets[i] & fold_sets[j]
                jac = len(inter) / len(union) if union else 0
                jaccard_values.append(jac)

        mean_jaccard = np.mean(jaccard_values) if jaccard_values else 0
        diagnostics['jaccard'] = {
            'mean_jaccard': round(mean_jaccard, 4),
            'pairwise_values': [round(j, 4) for j in jaccard_values],
            'interpretation': (
                'STABLE (genuine signal)' if mean_jaccard > 0.6 else
                'MODERATE (proceed with caution)' if mean_jaccard > 0.3 else
                'UNSTABLE (likely noise memorization)'
            ),
        }
        self.log(f"  Jaccard (top-{top_k_jaccard} overlap): mean={mean_jaccard:.3f} -- "
                 f"{diagnostics['jaccard']['interpretation']}")

        # --- Semantic cluster concentration ---
        if hasattr(self, 'signal_df') and not self.signal_df.empty and self.feature_categories:
            top100 = self.signal_df.head(100)
            type_counts = top100['category'].value_counts()
            if len(type_counts) >= 3:
                top3_share = type_counts.head(3).sum() / len(top100)
                diagnostics['concentration'] = {
                    'top3_category_share': round(top3_share, 3),
                    'category_distribution': type_counts.to_dict(),
                    'interpretation': (
                        'HEALTHY DIVERSITY' if 0.30 <= top3_share <= 0.70 else
                        'HIGH CONCENTRATION (over-reliant on few signal types)' if top3_share > 0.70 else
                        'LOW CONCENTRATION (near-uniform, noise risk)'
                    ),
                }
                self.log(f"  Concentration: top-3 categories = {top3_share:.1%} -- "
                         f"{diagnostics['concentration']['interpretation']}")

        self.results['overfitting_diagnostics'] = diagnostics
        self.timings['stage10_diagnostics'] = time.time() - t0
        self.log(f"  Done in {self.timings['stage10_diagnostics']:.1f}s")

    # ================================================================
    # STAGE 11: Save results
    # ================================================================
    def _save_results(self):
        """Save all results to JSON files."""
        t0 = time.time()
        self.log("Stage 11: Saving results...")

        prefix = f'fi_pipeline_{self.tf_name}'

        # Main signal strength table (top-5K)
        if hasattr(self, 'signal_df') and not self.signal_df.empty:
            top5k = self.signal_df.head(5000)
            signal_path = os.path.join(self.output_dir, f'{prefix}_signal_strength.json')
            records = top5k.to_dict(orient='records')
            # Convert numpy types for JSON
            for r in records:
                for k, v in r.items():
                    if isinstance(v, (np.integer,)):
                        r[k] = int(v)
                    elif isinstance(v, (np.floating,)):
                        r[k] = float(v)
            with open(signal_path, 'w') as f:
                json.dump(records, f, indent=2, default=str)
            self.log(f"  Saved: {signal_path} ({len(records)} features)")

        # Summary report
        summary = {
            'tf': self.tf_name,
            'n_total_features': self.n_features,
            'n_folds': self.n_folds,
            'timings': self.timings,
            **self.results,
        }
        summary_path = os.path.join(self.output_dir, f'{prefix}_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        self.log(f"  Saved: {summary_path}")

        # Permutation importance (if computed)
        if hasattr(self, 'perm_df'):
            perm_path = os.path.join(self.output_dir, f'{prefix}_permutation.json')
            self.perm_df.to_json(perm_path, orient='records', indent=2)
            self.log(f"  Saved: {perm_path}")

        # SHAP importance (if computed)
        if hasattr(self, 'shap_df'):
            shap_path = os.path.join(self.output_dir, f'{prefix}_shap.json')
            self.shap_df.to_json(shap_path, orient='records', indent=2)
            self.log(f"  Saved: {shap_path}")

        self.timings['stage11_save'] = time.time() - t0
        self.log(f"  Done in {self.timings['stage11_save']:.1f}s")

    # ================================================================
    # STAGE 12: Visualization
    # ================================================================
    def _visualize(self):
        """Generate all plots."""
        t0 = time.time()
        self.log("Stage 12: Generating visualizations...")

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            import seaborn as sns
        except ImportError:
            self.log("  SKIPPED: matplotlib/seaborn not installed")
            return

        if not hasattr(self, 'signal_df') or self.signal_df.empty:
            self.log("  SKIPPED: no signal data")
            return

        prefix = f'fi_pipeline_{self.tf_name}'
        cat_colors = {
            'TECHNICAL': '#2196F3', 'ESOTERIC': '#FF9800',
            'CROSS_TECH_ESO': '#4CAF50', 'MACRO': '#9C27B0',
            'UNKNOWN': '#607D8B',
        }

        # --- PLOT 1: Heatmap of top-100 ---
        try:
            top100 = self.signal_df.head(100).copy()
            fig, ax = plt.subplots(figsize=(14, 20))

            metrics = ['gain_score', 'stability_score', 'sign_score']
            heat_data = top100[metrics].values

            im = ax.imshow(heat_data, aspect='auto', cmap='YlOrRd', interpolation='nearest')
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(['Gain Score', 'Stability', 'Sign Consistency'],
                               fontsize=12, fontweight='bold')
            ax.set_yticks(range(len(top100)))
            ax.set_yticklabels(top100['feature'].values, fontsize=5.5)

            # Category color strips
            for i, cat in enumerate(top100['category'].values):
                color = cat_colors.get(cat, '#607D8B')
                ax.add_patch(mpatches.Rectangle(
                    (-0.55, i - 0.5), 0.45, 1.0,
                    color=color, transform=ax.transData, clip_on=False
                ))

            plt.colorbar(im, ax=ax, shrink=0.3, label='Score')
            legend_patches = [mpatches.Patch(color=v, label=k) for k, v in cat_colors.items()
                             if k in top100['category'].values]
            if legend_patches:
                ax.legend(handles=legend_patches, loc='upper right',
                         bbox_to_anchor=(1.35, 1.0), fontsize=9)
            ax.set_title(f'{self.tf_name.upper()} Top-100 Features: Gain x Stability x Sign',
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{prefix}_heatmap.png'),
                       dpi=150, bbox_inches='tight')
            plt.close()
            self.log("  Saved heatmap")
        except Exception as e:
            self.log(f"  Heatmap failed: {e}")

        # --- PLOT 2: Pareto frontier ---
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            df_plot = self.signal_df[self.signal_df['mean_gain'] > 0].head(2000)

            colors = df_plot['category'].map(cat_colors).fillna('#607D8B')
            sizes = 15 + (df_plot['sign_score'].values * 60)

            ax.scatter(df_plot['stability_score'], df_plot['mean_gain'],
                      c=colors, s=sizes, alpha=0.5, linewidths=0.2, edgecolors='white')

            # Pareto frontier
            pts = df_plot[['stability_score', 'mean_gain']].values
            is_pareto = np.ones(len(pts), dtype=bool)
            for i in range(len(pts)):
                if is_pareto[i]:
                    dominated = np.all(pts[is_pareto] >= pts[i], axis=1) & \
                                np.any(pts[is_pareto] > pts[i], axis=1)
                    mask_indices = np.where(is_pareto)[0]
                    is_pareto[mask_indices[dominated]] = False

            pareto_pts = df_plot[is_pareto].sort_values('stability_score')
            if len(pareto_pts) > 0:
                ax.plot(pareto_pts['stability_score'], pareto_pts['mean_gain'],
                       'r--', lw=2.0, label='Pareto Frontier', zorder=5)

            ax.set_xlabel('Stability Score', fontsize=12)
            ax.set_ylabel('Mean Gain (log scale)', fontsize=12)
            ax.set_yscale('log')
            ax.set_title(f'{self.tf_name.upper()} Pareto Frontier: Gain vs Stability\n'
                        f'Dot size = sign consistency, color = category',
                        fontsize=13, fontweight='bold')

            legend_patches = [mpatches.Patch(color=v, label=k) for k, v in cat_colors.items()
                             if k in df_plot['category'].values]
            if pareto_pts is not None and len(pareto_pts) > 0:
                legend_patches.append(plt.Line2D([0], [0], color='red', linestyle='--',
                                                  label='Pareto Frontier'))
            ax.legend(handles=legend_patches, fontsize=9)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{prefix}_pareto.png'),
                       dpi=150, bbox_inches='tight')
            plt.close()
            self.log("  Saved Pareto frontier")
        except Exception as e:
            self.log(f"  Pareto plot failed: {e}")

        # --- PLOT 3: Category breakdown ---
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Pie chart
            cat_counts = self.signal_df.head(1000)['category'].value_counts()
            colors_pie = [cat_colors.get(c, '#607D8B') for c in cat_counts.index]
            axes[0].pie(cat_counts.values, labels=cat_counts.index,
                       autopct='%1.1f%%', colors=colors_pie, startangle=90)
            axes[0].set_title(f'{self.tf_name.upper()} Top-1K Category Breakdown',
                             fontsize=12, fontweight='bold')

            # Bar chart by decile
            n_per_dec = max(1, len(self.signal_df.head(1000)) // 10)
            decile_data = defaultdict(list)
            cats_present = sorted(set(self.signal_df['category'].values))
            for i in range(10):
                chunk = self.signal_df.iloc[i*n_per_dec:(i+1)*n_per_dec]
                for cat in cats_present:
                    decile_data[cat].append((chunk['category'] == cat).sum())

            x = np.arange(10)
            width = 0.8 / max(len(cats_present), 1)
            for j, cat in enumerate(cats_present):
                axes[1].bar(x + j*width, decile_data[cat], width,
                           label=cat, color=cat_colors.get(cat, '#607D8B'), alpha=0.85)
            axes[1].set_xticks(x + width * len(cats_present) / 2)
            axes[1].set_xticklabels([f'D{i+1}' for i in range(10)], fontsize=9)
            axes[1].set_ylabel('Count')
            axes[1].set_title('Category per Importance Decile', fontsize=12, fontweight='bold')
            axes[1].legend(fontsize=8)

            plt.suptitle(f'{self.tf_name.upper()} Feature Category Analysis',
                        fontsize=14, y=1.02)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'{prefix}_categories.png'),
                       dpi=150, bbox_inches='tight')
            plt.close()
            self.log("  Saved category breakdown")
        except Exception as e:
            self.log(f"  Category plot failed: {e}")

        self.timings['stage12_viz'] = time.time() - t0
        self.log(f"  Done in {self.timings['stage12_viz']:.1f}s")

    # ================================================================
    # MAIN RUN METHOD
    # ================================================================
    def run(self, skip_permutation=False, skip_shap=False,
            skip_injection=False, skip_viz=False):
        """
        Run the complete feature importance pipeline.

        Args:
            skip_permutation: skip stage 5 (needs X_val, y_val)
            skip_shap: skip stage 6 (needs X_val, shap library)
            skip_injection: skip stage 7 (needs X_val, y_val)
            skip_viz: skip stage 12 (needs matplotlib)

        Returns:
            dict with all results
        """
        t_total = time.time()
        self.log(f"="*70)
        self.log(f"FEATURE IMPORTANCE PIPELINE -- {self.tf_name.upper()}")
        self.log(f"  Features: {self.n_features:,}, Folds: {self.n_folds}")
        self.log(f"  Val data: {'provided' if self.X_val is not None else 'NOT provided'}")
        self.log(f"  Categories: {'provided' if self.feature_categories else 'NOT provided'}")
        self.log(f"  Other TFs: {list(self.other_tf_importances.keys()) if self.other_tf_importances else 'none'}")
        self.log(f"="*70)

        # Core stages (always run)
        self._extract_gains()          # Stage 1
        self._stability_selection()    # Stage 2
        self._sign_consistency()       # Stage 3
        self._composite_signal_strength()  # Stage 4

        # Optional stages
        if not skip_permutation:
            self._permutation_importance()  # Stage 5

        if not skip_shap:
            self._shap_analysis()          # Stage 6

        if not skip_injection:
            self._random_injection_test()  # Stage 7

        self._cross_tf_analysis()      # Stage 8
        self._esoteric_validation()    # Stage 9
        self._overfitting_diagnostics()  # Stage 10
        self._save_results()           # Stage 11

        if not skip_viz:
            self._visualize()          # Stage 12

        total_time = time.time() - t_total
        self.log(f"\n{'='*70}")
        self.log(f"PIPELINE COMPLETE -- {total_time:.1f}s total")
        self.log(f"  Stage timings:")
        for stage, t in sorted(self.timings.items()):
            self.log(f"    {stage}: {t:.1f}s")
        self.log(f"{'='*70}")

        return {
            'signal_df': self.signal_df if hasattr(self, 'signal_df') else None,
            'results': self.results,
            'timings': self.timings,
        }


# ================================================================
# INTEGRATION: Run from CPCV checkpoint
# ================================================================
def run_from_cpcv_checkpoint(checkpoint_path, tf_name, output_dir,
                              feature_names=None, X_val=None, y_val=None):
    """
    Load CPCV fold models from checkpoint and run the pipeline.

    Args:
        checkpoint_path: path to cpcv_oos_predictions_{tf}.pkl
        tf_name: timeframe name
        output_dir: output directory
        feature_names: list of feature names
        X_val: validation data (optional)
        y_val: validation labels (optional)
    """
    import pickle
    import lightgbm as lgb

    # Load checkpoint
    with open(checkpoint_path, 'rb') as f:
        ckpt = pickle.load(f)

    # The checkpoint contains oos_predictions with model info
    # But we need the actual booster objects -- load from saved model files
    model_dir = os.path.dirname(checkpoint_path)
    boosters = []
    for i in range(10):  # check up to 10 folds
        model_path = os.path.join(model_dir, f'model_{tf_name}_fold{i}.txt')
        if os.path.exists(model_path):
            boosters.append(lgb.Booster(model_file=model_path))
        else:
            break

    if not boosters:
        print(f"ERROR: No fold model files found at {model_dir}/model_{tf_name}_fold*.txt")
        print("  Models must be saved during CPCV training. Add to ml_multi_tf.py:")
        print("    model.save_model(f'model_{tf_name}_fold{wi}.txt')")
        return None

    if feature_names is None:
        feature_names = boosters[0].feature_name()

    pipeline = FeatureImportancePipeline(
        fold_boosters=boosters,
        feature_names=feature_names,
        tf_name=tf_name,
        output_dir=output_dir,
        X_val=X_val,
        y_val=y_val,
    )
    return pipeline.run()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Feature Importance Pipeline')
    parser.add_argument('--tf', required=True, help='Timeframe (1w, 1d, 4h, 1h, 15m)')
    parser.add_argument('--model-dir', default='.', help='Directory with fold model files')
    parser.add_argument('--output-dir', default='.', help='Output directory')
    parser.add_argument('--skip-perm', action='store_true', help='Skip permutation importance')
    parser.add_argument('--skip-shap', action='store_true', help='Skip SHAP analysis')
    parser.add_argument('--skip-injection', action='store_true', help='Skip random injection test')
    parser.add_argument('--skip-viz', action='store_true', help='Skip visualization')
    args = parser.parse_args()

    import lightgbm as lgb

    # Load fold models
    boosters = []
    for i in range(20):
        path = os.path.join(args.model_dir, f'model_{args.tf}_fold{i}.txt')
        if os.path.exists(path):
            boosters.append(lgb.Booster(model_file=path))

    if not boosters:
        print(f"No model files found: {args.model_dir}/model_{args.tf}_fold*.txt")
        sys.exit(1)

    feature_names = boosters[0].feature_name()
    print(f"Loaded {len(boosters)} fold models with {len(feature_names):,} features")

    pipeline = FeatureImportancePipeline(
        fold_boosters=boosters,
        feature_names=feature_names,
        tf_name=args.tf,
        output_dir=args.output_dir,
    )
    pipeline.run(
        skip_permutation=args.skip_perm,
        skip_shap=args.skip_shap,
        skip_injection=args.skip_injection,
        skip_viz=args.skip_viz,
    )
