#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
backtest_validation.py — PBO + Deflated Sharpe Ratio
=====================================================
Post-training validation: separates real edge from overfitting luck.

Usage:
    from backtest_validation import compute_pbo, compute_deflated_sharpe, validation_report
    report = validation_report(oos_predictions, optimizer_results, tf_name)
"""

import numpy as np
import pandas as pd
import os
import json
from scipy import stats


# ============================================================
# PROBABILITY OF BACKTEST OVERFITTING (PBO)
# ============================================================

def compute_pbo(oos_predictions, metric_fn=None, is_metrics=None):
    """Compute Probability of Backtest Overfitting from CPCV OOS predictions.

    PBO = fraction of CPCV paths where the best in-sample strategy
    ranks below median out-of-sample.

    When is_metrics is provided, uses ACTUAL IS performance from training
    (the correct methodology per Lopez de Prado) instead of splitting OOS
    data in half to fabricate fake IS/OOS pairs.

    Args:
        oos_predictions: list of dicts from CPCV, OR a file path string to a .pkl
            Each dict has: 'y_true', 'y_pred_probs', 'path'
        metric_fn: function(y_true, y_pred_probs) -> float (higher = better)
                   Default: accuracy
        is_metrics: list of dicts with IS performance from training.
            Each dict has: 'path' (int), and one or more of:
              'is_sharpe' (float), 'is_accuracy' (float), 'is_mlogloss' (float)
            When provided, IS ranks come from training; OOS ranks from OOS predictions.

    Returns:
        dict with 'pbo': float, 'omega_bar': float (logit), 'n_paths': int,
              'is_scores': list, 'oos_scores': list, 'recommendation': str
    """
    # Auto-load from file if path provided
    if isinstance(oos_predictions, str):
        import pickle
        with open(oos_predictions, 'rb') as f:
            oos_predictions = pickle.load(f)
    if metric_fn is None:
        def metric_fn(y_true, y_pred_probs):
            return (np.argmax(y_pred_probs, axis=1) == y_true).mean()

    n_paths = len(oos_predictions)
    if n_paths < 4:
        return {'pbo': np.nan, 'n_paths': n_paths, 'warning': 'too few paths'}

    # ------------------------------------------------------------------
    # PROPER PBO: actual IS metrics from training vs OOS metrics
    # ------------------------------------------------------------------
    if is_metrics is not None:
        # Build lookup: path_id -> IS score
        is_lookup = {}
        for m in is_metrics:
            pid = m['path']
            # Prefer Sharpe, fall back to accuracy
            if 'is_sharpe' in m and np.isfinite(m['is_sharpe']):
                is_lookup[pid] = m['is_sharpe']
            elif 'is_accuracy' in m:
                is_lookup[pid] = m['is_accuracy']

        # Compute OOS score per path
        is_scores = []
        oos_scores = []
        for pred in oos_predictions:
            pid = pred['path']
            if pid not in is_lookup:
                continue
            y_true = pred['y_true']
            y_probs = pred['y_pred_probs']
            if len(y_true) < 10:
                continue
            # OOS Sharpe from simulated returns
            pred_labels = np.argmax(y_probs, axis=1)
            # +1 if correct directional call, -1 if wrong
            sim_returns = np.where(pred_labels == y_true, 1.0, -1.0)
            oos_sharpe = sharpe_from_returns(sim_returns, periods_per_year=252)
            is_scores.append(is_lookup[pid])
            oos_scores.append(oos_sharpe)

        if len(is_scores) < 4:
            return {'pbo': np.nan, 'n_paths': len(is_scores),
                    'warning': 'too few matched IS/OOS paths', 'method': 'proper'}

        is_scores = np.array(is_scores)
        oos_scores = np.array(oos_scores)

        # Rank: higher score = lower rank number (rank 1 = best)
        is_ranks = stats.rankdata(-is_scores)
        oos_ranks = stats.rankdata(-oos_scores)

        n = len(is_scores)
        median_rank = n / 2

        # PBO = fraction of top-half IS performers that rank below median OOS
        is_top_mask = is_ranks <= median_rank
        oos_below_mask = oos_ranks > median_rank
        n_below_median = int((is_top_mask & oos_below_mask).sum())
        n_is_top = int(is_top_mask.sum())
        pbo = n_below_median / max(n_is_top, 1)

        # Omega-bar (logit): mean of logit(relative OOS rank) for IS-optimal paths
        # relative rank w_i = oos_rank_i / (n + 1), logit(w) = ln(w / (1 - w))
        relative_ranks = oos_ranks / (n + 1)
        # Clamp to avoid log(0)
        relative_ranks = np.clip(relative_ranks, 1e-6, 1 - 1e-6)
        logits = np.log(relative_ranks / (1 - relative_ranks))
        # omega_bar = mean logit for IS top-half paths
        omega_bar = float(np.mean(logits[is_top_mask]))

        return {
            'pbo': pbo,
            'omega_bar': omega_bar,
            'n_paths': n,
            'best_is_oos_rank': int(oos_ranks[np.argmin(is_ranks)]),
            'median_rank': median_rank,
            'is_scores': is_scores.tolist(),
            'oos_scores': oos_scores.tolist(),
            'method': 'proper',
            'recommendation': 'DEPLOY' if pbo < 0.15 else ('INVESTIGATE' if pbo < 0.30 else 'REJECT'),
        }

    # ------------------------------------------------------------------
    # FALLBACK: split OOS data in half (old method — NOT correct PBO)
    # ------------------------------------------------------------------
    import warnings as _w
    _w.warn(
        "compute_pbo: is_metrics not provided — falling back to OOS half-split. "
        "This is NOT proper PBO (Lopez de Prado). Pass is_metrics from training "
        "for correct IS vs OOS rank comparison.",
        UserWarning, stacklevel=2,
    )

    is_scores = []
    oos_scores = []

    for pred in oos_predictions:
        y_true = pred['y_true']
        y_probs = pred['y_pred_probs']
        n = len(y_true)
        if n < 10:
            continue
        mid = n // 2
        is_score = metric_fn(y_true[:mid], y_probs[:mid])
        oos_score = metric_fn(y_true[mid:], y_probs[mid:])
        is_scores.append(is_score)
        oos_scores.append(oos_score)

    if len(is_scores) < 4:
        return {'pbo': np.nan, 'n_paths': len(is_scores), 'warning': 'too few valid paths',
                'method': 'half_split_fallback'}

    is_scores = np.array(is_scores)
    oos_scores = np.array(oos_scores)

    # For each path, compute IS rank and OOS rank
    is_ranks = stats.rankdata(-is_scores)  # higher score = lower rank (rank 1 = best)
    oos_ranks = stats.rankdata(-oos_scores)

    n = len(is_scores)
    median_rank = n / 2

    # PBO = fraction of paths where best IS performer ranks below median OOS
    best_is_idx = np.argmin(is_ranks)  # index of best IS path
    best_is_oos_rank = oos_ranks[best_is_idx]

    # Vectorized: for paths in top-half IS, count those ranking below median OOS
    is_top_mask = is_ranks <= median_rank
    oos_below_mask = oos_ranks > median_rank
    n_below_median = int((is_top_mask & oos_below_mask).sum())
    n_is_top = int(is_top_mask.sum())
    pbo = n_below_median / max(n_is_top, 1)

    # Omega-bar for fallback too
    relative_ranks = oos_ranks / (n + 1)
    relative_ranks = np.clip(relative_ranks, 1e-6, 1 - 1e-6)
    logits = np.log(relative_ranks / (1 - relative_ranks))
    omega_bar = float(np.mean(logits[is_top_mask]))

    return {
        'pbo': pbo,
        'omega_bar': omega_bar,
        'n_paths': n,
        'best_is_oos_rank': int(best_is_oos_rank),
        'median_rank': median_rank,
        'is_scores': is_scores.tolist(),
        'oos_scores': oos_scores.tolist(),
        'method': 'half_split_fallback',
        'recommendation': 'DEPLOY' if pbo < 0.15 else ('INVESTIGATE' if pbo < 0.30 else 'REJECT'),
    }


# ============================================================
# DEFLATED SHARPE RATIO
# ============================================================

def compute_deflated_sharpe(observed_sharpe, n_trials, n_observations,
                            skewness=0.0, kurtosis=3.0):
    """Compute Deflated Sharpe Ratio (Bailey & Lopez de Prado).

    Adjusts observed Sharpe for multiple testing, non-normality.

    Args:
        observed_sharpe: the Sharpe ratio of the selected strategy
        n_trials: total number of strategies/configs tested (e.g., 30M)
        n_observations: number of return observations used to compute Sharpe
        skewness: skewness of the return series (0 for normal)
        kurtosis: kurtosis of the return series (3 for normal)

    Returns:
        dict with 'dsr': float, 'p_value': float, 'expected_max_sharpe': float
    """
    # Expected maximum Sharpe under null (all strategies are noise)
    # E[max(SR)] ≈ sqrt(2 * ln(N_trials)) for N_trials strategies
    # More precise: Euler-Mascheroni correction
    euler_mascheroni = 0.5772156649
    if n_trials > 1:
        expected_max_sr = np.sqrt(2 * np.log(n_trials)) - \
            (euler_mascheroni + np.log(np.log(n_trials))) / \
            (2 * np.sqrt(2 * np.log(n_trials)))
    else:
        expected_max_sr = 0.0

    # Standard error of Sharpe ratio (adjusted for non-normality)
    # SE(SR) = sqrt((1 - skew*SR + (kurtosis-1)/4 * SR^2) / (n-1))
    sr = observed_sharpe
    se_sr = np.sqrt(
        (1 - skewness * sr + ((kurtosis - 1) / 4) * sr ** 2) /
        max(n_observations - 1, 1)
    )

    # Deflated Sharpe = (SR_observed - E[max(SR)]) / SE(SR)
    if se_sr > 0:
        dsr = (sr - expected_max_sr) / se_sr
    else:
        dsr = 0.0

    # p-value: probability of observing this DSR or higher under null
    p_value = 1 - stats.norm.cdf(dsr)

    return {
        'dsr': dsr,
        'p_value': p_value,
        'observed_sharpe': observed_sharpe,
        'expected_max_sharpe': expected_max_sr,
        'se_sharpe': se_sr,
        'n_trials': n_trials,
        'n_observations': n_observations,
        'recommendation': 'DEPLOY' if p_value < 0.05 else ('INVESTIGATE' if p_value < 0.10 else 'REJECT'),
    }


# ============================================================
# SHARPE FROM EQUITY CURVE
# ============================================================

def sharpe_from_returns(returns, periods_per_year=252):
    """Compute annualized Sharpe ratio from a return series."""
    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[~np.isnan(returns)]
    if len(returns) < 10:
        return 0.0
    mean_r = np.mean(returns)
    std_r = np.std(returns, ddof=1)
    if std_r == 0:
        return 0.0
    return (mean_r / std_r) * np.sqrt(periods_per_year)


# ============================================================
# COMBINED VALIDATION REPORT
# ============================================================

def validation_report(oos_predictions, observed_sharpe=None,
                      n_optimizer_trials=30_000_000, n_observations=None,
                      returns=None, tf_name='1h', n_timeframes=6,
                      is_metrics=None):
    """Generate full validation report for a timeframe.

    Args:
        oos_predictions: list of CPCV OOS prediction dicts
        observed_sharpe: Sharpe ratio of best strategy (if known)
        n_optimizer_trials: number of param combos tested per TF by exhaustive optimizer
        n_observations: number of bars used for Sharpe computation
        returns: array of returns for the selected strategy (for auto Sharpe)
        tf_name: timeframe name for logging
        n_timeframes: number of TFs tested (total trials = n_optimizer_trials * n_timeframes)
        is_metrics: list of dicts with IS performance from training (for proper PBO).
            Each dict has: 'path' (int), 'is_sharpe' or 'is_accuracy'.

    Returns:
        dict with full report
    """
    # Auto-load from file if path provided
    if isinstance(oos_predictions, str):
        import pickle
        with open(oos_predictions, 'rb') as f:
            oos_predictions = pickle.load(f)

    report = {'tf': tf_name}

    # PBO — use proper methodology when IS metrics available
    pbo_result = compute_pbo(oos_predictions, is_metrics=is_metrics)
    report['pbo'] = pbo_result

    # Deflated Sharpe
    if returns is not None:
        returns = np.asarray(returns, dtype=np.float64)
        returns = returns[~np.isnan(returns)]
        # Annualized Sharpe — periods depend on TF
        tf_periods = {
            '5m': 365 * 288,  # 288 5-min bars per day
            '15m': 365 * 96,
            '1h': 365 * 24,
            '4h': 365 * 6,
            '1d': 365,
            '1w': 52,
        }
        periods = tf_periods.get(tf_name, 365 * 24)
        observed_sharpe = sharpe_from_returns(returns, periods_per_year=periods)
        n_observations = len(returns)
        skewness = float(stats.skew(returns))
        kurtosis = float(stats.kurtosis(returns, fisher=False))
    else:
        skewness = 0.0
        kurtosis = 3.0
        if n_observations is None:
            n_observations = 1000

    if observed_sharpe is not None:
        # Total trials = per-TF combos * number of TFs tested
        total_trials = n_optimizer_trials * n_timeframes
        dsr_result = compute_deflated_sharpe(
            observed_sharpe, total_trials, n_observations,
            skewness=skewness, kurtosis=kurtosis,
        )
        report['deflated_sharpe'] = dsr_result
    else:
        report['deflated_sharpe'] = {'warning': 'no Sharpe provided'}

    # Overall recommendation
    pbo_ok = pbo_result.get('pbo', 1.0) < 0.30
    dsr_ok = report.get('deflated_sharpe', {}).get('p_value', 1.0) < 0.10
    if pbo_ok and dsr_ok:
        report['overall'] = 'DEPLOY'
    elif pbo_ok or dsr_ok:
        report['overall'] = 'INVESTIGATE'
    else:
        report['overall'] = 'REJECT'

    return report


def save_report(report, db_dir='.'):
    """Save validation report to JSON."""
    tf = report.get('tf', 'unknown')
    path = os.path.join(db_dir, f'validation_report_{tf}.json')

    # Convert numpy types for JSON serialization
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    clean = json.loads(json.dumps(report, default=_convert))
    with open(path, 'w') as f:
        json.dump(clean, f, indent=2)
    return path


if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser(description='PBO + Deflated Sharpe validation')
    parser.add_argument('--tf', type=str, default=None, help='Timeframe (e.g. 1w, 1d)')
    parser.add_argument('--db-dir', type=str, default='.', help='Directory with predictions')
    args = parser.parse_args()

    if args.tf:
        # Production: load real OOS predictions and run validation
        oos_path = os.path.join(args.db_dir, f'cpcv_oos_predictions_{args.tf}.pkl')
        if not os.path.exists(oos_path):
            print(f"ERROR: {oos_path} not found — run training first")
            sys.exit(1)
        print(f"=== PBO Validation: {args.tf} ===")
        print(f"  Loading: {oos_path}")
        report = validation_report(oos_path, tf_name=args.tf)
        save_report(report, db_dir=args.db_dir)
        print(f"  Overall: {report['overall']}")
        print(f"  PBO: {report['pbo']['pbo']:.3f} ({report['pbo']['recommendation']})")
        if 'dsr' in report.get('deflated_sharpe', {}):
            print(f"  DSR p-value: {report['deflated_sharpe']['p_value']:.4f}")
        print(f"  Saved: validation_report_{args.tf}.json")
    else:
        # Test mode with synthetic data
        print("=== PBO Test (synthetic) ===")
        np.random.seed(42)
        fake_oos = []
        for i in range(15):
            n = 200
            y_true = np.random.randint(0, 3, n)
            y_probs = np.random.dirichlet([1, 1, 1], n)
            for j in range(n):
                y_probs[j, y_true[j]] += 0.3
            y_probs /= y_probs.sum(axis=1, keepdims=True)
            fake_oos.append({'path': i, 'y_true': y_true, 'y_pred_probs': y_probs})
        pbo = compute_pbo(fake_oos)
        print(f"  PBO = {pbo['pbo']:.3f} ({pbo['recommendation']})")
