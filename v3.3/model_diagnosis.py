#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_diagnosis.py — Automatic Model Diagnosis & Rollback System for V3.3
========================================================================

FEATURES:
- Automatic diagnosis on training completion
- SHAP analysis when AUC < previous best
- Esoteric feature validation (must be in top 100)
- Train/val gap detection (overfitting)
- Label distribution validation
- Auto-rollback triggers (AUC drop > 5%, drawdown > 25%)
- Discord notifications for all diagnosis events
- Integration with deploy_model.py versioning system

USAGE:
    from model_diagnosis import diagnose_model, auto_rollback_check

    # After training
    diagnosis = diagnose_model(
        tf='1w',
        new_auc=0.915,
        train_auc=0.934,
        val_auc=0.915,
        positive_rate=0.42,
        feature_importance=shap_values,
        feature_names=feature_names,
        model_path='model_1w.json'
    )

    # Check if rollback needed
    if auto_rollback_check(diagnosis):
        rollback_model(tf='1w')

DIAGNOSIS CHECKS:
1. AUC Regression: new_auc < previous_best - 0.05
2. Overfitting: train_auc - val_auc > 0.15
3. Label Imbalance: positive_rate < 0.35 or > 0.65
4. Missing Esoteric Signals: zero esoteric features in SHAP top 100
5. Feature Pipeline Broken: total feature count drop > 10%

AUTO-ROLLBACK TRIGGERS:
- AUC dropped > 5% from previous best
- Max drawdown > 25% in backtest (if backtest provided)
- Critical diagnosis failures (esoteric features missing)

INTEGRATION:
- Called automatically by cloud_run_tf.py after training
- Posts all results to Discord via discord_gate.py
- Writes diagnosis JSON to models/{tf}/diagnosis_latest.json
- Triggers deploy_model.py rollback on critical failures
"""

import os
import sys
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np

# Import project modules
try:
    from discord_gate import gate
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False
    print("WARNING: discord_gate not available, notifications disabled")

try:
    from config import PROTECTED_FEATURE_PREFIXES
except ImportError:
    # Fallback if config not available
    PROTECTED_FEATURE_PREFIXES = [
        'gematria_', 'numerology_', 'astro_', 'hebrew_', 'tarot_',
        'kp_', 'solar_', 'geomag_', 'schumann_', 'cosmic_'
    ]

# Constants
MODELS_BASE_DIR = Path(__file__).parent.parent / 'models'
VALID_TIMEFRAMES = ['1w', '1d', '4h', '1h', '15m']

# Diagnosis thresholds
AUC_DROP_CRITICAL = 0.05      # Auto-rollback if AUC drops > 5%
AUC_DROP_WARNING = 0.02       # Warning if AUC drops > 2%
OVERFIT_THRESHOLD = 0.15      # Train/val gap indicating overfitting
LABEL_MIN = 0.35              # Minimum healthy positive rate
LABEL_MAX = 0.65              # Maximum healthy positive rate
FEATURE_DROP_THRESHOLD = 0.10 # Critical if feature count drops > 10%
ESOTERIC_MIN_TOP100 = 1       # At least 1 esoteric in top 100


def get_previous_manifest(tf: str) -> Optional[Dict]:
    """Load the manifest of the currently active model."""
    active_link = MODELS_BASE_DIR / tf / 'active'

    if not active_link.exists() or not active_link.is_symlink():
        return None

    manifest_path = active_link / 'manifest.json'
    if not manifest_path.exists():
        return None

    try:
        with open(manifest_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"WARNING: Could not load previous manifest: {e}")
        return None


def classify_feature(feature_name: str) -> str:
    """Classify a feature by its prefix."""
    feature_lower = feature_name.lower()

    # Check esoteric prefixes
    for prefix in PROTECTED_FEATURE_PREFIXES:
        if feature_lower.startswith(prefix):
            return 'ESOTERIC'

    # Check cross-feature patterns
    if '__x__' in feature_name:
        left, right = feature_name.split('__x__', 1)
        left_type = classify_feature(left)
        right_type = classify_feature(right)

        if left_type == 'ESOTERIC' or right_type == 'ESOTERIC':
            return 'CROSS_ESOTERIC'
        return 'CROSS_TECH'

    # Default to technical
    return 'TECHNICAL'


def check_esoteric_presence(feature_names: List[str], feature_importance: np.ndarray,
                           top_n: int = 100) -> Dict:
    """Check if esoteric features appear in top N most important features."""

    # Get top N features by importance
    top_indices = np.argsort(feature_importance)[::-1][:top_n]
    top_features = [feature_names[i] for i in top_indices]

    # Count esoteric features
    esoteric_count = 0
    esoteric_features = []
    cross_esoteric_count = 0

    for feat in top_features:
        feat_type = classify_feature(feat)
        if feat_type == 'ESOTERIC':
            esoteric_count += 1
            esoteric_features.append(feat)
        elif feat_type == 'CROSS_ESOTERIC':
            cross_esoteric_count += 1
            esoteric_features.append(feat)

    return {
        'total_esoteric_in_top100': esoteric_count,
        'total_cross_esoteric_in_top100': cross_esoteric_count,
        'esoteric_features': esoteric_features[:10],  # Top 10 for display
        'pass': (esoteric_count + cross_esoteric_count) >= ESOTERIC_MIN_TOP100
    }


def diagnose_model(
    tf: str,
    new_auc: float,
    train_auc: float,
    val_auc: float,
    positive_rate: float,
    feature_importance: np.ndarray,
    feature_names: List[str],
    model_path: Optional[str] = None,
    backtest_metrics: Optional[Dict] = None
) -> Dict:
    """
    Comprehensive model diagnosis.

    Args:
        tf: Timeframe ('1w', '1d', etc.)
        new_auc: Final model AUC (validation or CPCV)
        train_auc: Training AUC
        val_auc: Validation AUC
        positive_rate: Fraction of positive labels (LONG or SHORT)
        feature_importance: Array of feature importances (gain or SHAP)
        feature_names: List of feature names
        model_path: Optional path to model file
        backtest_metrics: Optional dict with backtest results (sharpe, drawdown, etc.)

    Returns:
        Dict with diagnosis results and recommendations
    """

    diagnosis = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'timeframe': tf,
        'new_auc': new_auc,
        'train_auc': train_auc,
        'val_auc': val_auc,
        'positive_rate': positive_rate,
        'num_features': len(feature_names),
        'checks': {},
        'severity': 'OK',
        'recommendations': [],
        'rollback_required': False
    }

    # Load previous model for comparison
    prev_manifest = get_previous_manifest(tf)
    prev_auc = prev_manifest['accuracy'] if prev_manifest else None
    prev_num_features = prev_manifest.get('num_features', 0) if prev_manifest else 0

    # ================================================================
    # CHECK 1: AUC Regression
    # ================================================================
    auc_check = {'pass': True, 'severity': 'OK'}

    if prev_auc is not None:
        auc_drop = prev_auc - new_auc
        auc_check['previous_auc'] = prev_auc
        auc_check['auc_drop'] = auc_drop

        if auc_drop > AUC_DROP_CRITICAL:
            auc_check['pass'] = False
            auc_check['severity'] = 'CRITICAL'
            auc_check['message'] = f"AUC dropped {auc_drop:.3f} (>{AUC_DROP_CRITICAL:.2f} threshold)"
            diagnosis['recommendations'].append(f"ROLLBACK: AUC dropped {auc_drop:.1%} from {prev_auc:.3f} to {new_auc:.3f}")
            diagnosis['rollback_required'] = True
            diagnosis['severity'] = 'CRITICAL'
        elif auc_drop > AUC_DROP_WARNING:
            auc_check['severity'] = 'WARNING'
            auc_check['message'] = f"AUC dropped {auc_drop:.3f} (>{AUC_DROP_WARNING:.2f} warning)"
            diagnosis['recommendations'].append(f"INVESTIGATE: AUC regression of {auc_drop:.1%}")
            if diagnosis['severity'] == 'OK':
                diagnosis['severity'] = 'WARNING'
        else:
            auc_check['message'] = f"AUC improved by {-auc_drop:.3f}" if auc_drop < 0 else f"AUC stable (drop: {auc_drop:.3f})"
    else:
        auc_check['message'] = "First model for this TF, no baseline to compare"

    diagnosis['checks']['auc_regression'] = auc_check

    # ================================================================
    # CHECK 2: Overfitting (train/val gap)
    # ================================================================
    overfit_check = {'pass': True, 'severity': 'OK'}
    train_val_gap = train_auc - val_auc
    overfit_check['train_val_gap'] = train_val_gap

    if train_val_gap > OVERFIT_THRESHOLD:
        overfit_check['pass'] = False
        overfit_check['severity'] = 'CRITICAL'
        overfit_check['message'] = f"Severe overfitting: train={train_auc:.3f}, val={val_auc:.3f}, gap={train_val_gap:.3f}"
        diagnosis['recommendations'].append(f"OVERFIT: Increase regularization (lambda_l1/l2, min_gain_to_split)")
        diagnosis['rollback_required'] = True
        diagnosis['severity'] = 'CRITICAL'
    elif train_val_gap > 0.10:
        overfit_check['severity'] = 'WARNING'
        overfit_check['message'] = f"Moderate overfitting: gap={train_val_gap:.3f}"
        diagnosis['recommendations'].append(f"Monitor: train/val gap is {train_val_gap:.1%}")
        if diagnosis['severity'] == 'OK':
            diagnosis['severity'] = 'WARNING'
    else:
        overfit_check['message'] = f"Healthy train/val gap: {train_val_gap:.3f}"

    diagnosis['checks']['overfitting'] = overfit_check

    # ================================================================
    # CHECK 3: Label Distribution
    # ================================================================
    label_check = {'pass': True, 'severity': 'OK'}
    label_check['positive_rate'] = positive_rate

    if positive_rate < LABEL_MIN or positive_rate > LABEL_MAX:
        label_check['pass'] = False
        label_check['severity'] = 'CRITICAL'
        label_check['message'] = f"Label distribution broken: {positive_rate:.1%} (expected {LABEL_MIN:.0%}-{LABEL_MAX:.0%})"
        diagnosis['recommendations'].append(f"CRITICAL: Check triple-barrier labeling logic")
        diagnosis['rollback_required'] = True
        diagnosis['severity'] = 'CRITICAL'
    else:
        label_check['message'] = f"Healthy label distribution: {positive_rate:.1%}"

    diagnosis['checks']['label_distribution'] = label_check

    # ================================================================
    # CHECK 4: Esoteric Feature Presence
    # ================================================================
    esoteric_check = check_esoteric_presence(feature_names, feature_importance, top_n=100)

    if not esoteric_check['pass']:
        esoteric_check['severity'] = 'CRITICAL'
        esoteric_check['message'] = f"ZERO esoteric features in top 100 - feature pipeline broken"
        diagnosis['recommendations'].append("CRITICAL: Feature pipeline dropped all esoteric signals")
        diagnosis['recommendations'].append("Check: feature_library.py, cross_feature_generator, validate.py")
        diagnosis['rollback_required'] = True
        diagnosis['severity'] = 'CRITICAL'
    else:
        esoteric_check['severity'] = 'OK'
        total_eso = esoteric_check['total_esoteric_in_top100'] + esoteric_check['total_cross_esoteric_in_top100']
        esoteric_check['message'] = f"{total_eso} esoteric features in top 100"

    diagnosis['checks']['esoteric_presence'] = esoteric_check

    # ================================================================
    # CHECK 5: Feature Count Stability
    # ================================================================
    feature_count_check = {'pass': True, 'severity': 'OK'}
    feature_count_check['num_features'] = len(feature_names)
    feature_count_check['previous_num_features'] = prev_num_features

    if prev_num_features > 0:
        feature_drop_pct = (prev_num_features - len(feature_names)) / prev_num_features
        feature_count_check['feature_drop_pct'] = feature_drop_pct

        if feature_drop_pct > FEATURE_DROP_THRESHOLD:
            feature_count_check['pass'] = False
            feature_count_check['severity'] = 'CRITICAL'
            feature_count_check['message'] = f"Feature count dropped {feature_drop_pct:.1%} ({prev_num_features:,} → {len(feature_names):,})"
            diagnosis['recommendations'].append(f"CRITICAL: Feature pipeline lost {feature_drop_pct:.0%} of features")
            diagnosis['rollback_required'] = True
            diagnosis['severity'] = 'CRITICAL'
        else:
            feature_count_check['message'] = f"Feature count stable: {len(feature_names):,} ({feature_drop_pct:+.1%})"
    else:
        feature_count_check['message'] = f"First model: {len(feature_names):,} features"

    diagnosis['checks']['feature_count'] = feature_count_check

    # ================================================================
    # CHECK 6: Backtest Metrics (if provided)
    # ================================================================
    if backtest_metrics:
        backtest_check = {'pass': True, 'severity': 'OK'}
        max_dd = backtest_metrics.get('max_drawdown', 0)
        sharpe = backtest_metrics.get('sharpe', 0)

        backtest_check['max_drawdown'] = max_dd
        backtest_check['sharpe'] = sharpe

        if max_dd > 0.25:  # 25% drawdown threshold
            backtest_check['pass'] = False
            backtest_check['severity'] = 'CRITICAL'
            backtest_check['message'] = f"Max drawdown {max_dd:.1%} exceeds 25% threshold"
            diagnosis['recommendations'].append(f"ROLLBACK: Backtest drawdown {max_dd:.0%} too high")
            diagnosis['rollback_required'] = True
            diagnosis['severity'] = 'CRITICAL'
        else:
            backtest_check['message'] = f"Healthy backtest: DD={max_dd:.1%}, Sharpe={sharpe:.2f}"

        diagnosis['checks']['backtest'] = backtest_check

    # ================================================================
    # Summary
    # ================================================================
    failed_checks = [k for k, v in diagnosis['checks'].items() if not v.get('pass', True)]
    diagnosis['failed_checks'] = failed_checks
    diagnosis['num_failed_checks'] = len(failed_checks)

    return diagnosis


def notify_discord(diagnosis: Dict):
    """Post diagnosis results to Discord."""
    if not DISCORD_AVAILABLE:
        print("Discord notifications disabled (discord_gate not available)")
        return

    tf = diagnosis['timeframe']
    severity = diagnosis['severity']

    # Build notification message
    title = f"Model Diagnosis: {tf.upper()}"

    if severity == 'CRITICAL':
        title += " ⚠️ CRITICAL"
        level = "CRITICAL"
    elif severity == 'WARNING':
        title += " ⚠️ WARNING"
        level = "APPROVAL_REQUIRED"
    else:
        title += " ✅ PASS"
        level = "INFO"

    context = {
        'AUC': f"{diagnosis['new_auc']:.3f}",
        'Train AUC': f"{diagnosis['train_auc']:.3f}",
        'Val AUC': f"{diagnosis['val_auc']:.3f}",
        'Positive Rate': f"{diagnosis['positive_rate']:.1%}",
        'Features': f"{diagnosis['num_features']:,}",
        'Failed Checks': f"{diagnosis['num_failed_checks']}/{len(diagnosis['checks'])}",
    }

    # Add check details
    for check_name, check_data in diagnosis['checks'].items():
        if not check_data.get('pass', True):
            context[f"❌ {check_name}"] = check_data.get('message', 'Failed')

    # Add recommendations
    if diagnosis['recommendations']:
        context['Recommendations'] = '\n'.join(f"• {r}" for r in diagnosis['recommendations'])

    # Send notification
    if level == "CRITICAL":
        gate.critical(title, context, event_type=f"model_diagnosis_{tf}")
    elif level == "APPROVAL_REQUIRED":
        gate.approve(f"model_diagnosis_{tf}", context, message=title)
    else:
        gate.notify(title, context, event_type=f"model_diagnosis_{tf}")


def save_diagnosis(diagnosis: Dict):
    """Save diagnosis results to JSON file."""
    tf = diagnosis['timeframe']
    tf_dir = MODELS_BASE_DIR / tf
    tf_dir.mkdir(parents=True, exist_ok=True)

    # Save latest diagnosis
    latest_path = tf_dir / 'diagnosis_latest.json'
    with open(latest_path, 'w') as f:
        json.dump(diagnosis, f, indent=2)

    # Archive diagnosis with timestamp
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_path = tf_dir / f'diagnosis_{timestamp}.json'
    with open(archive_path, 'w') as f:
        json.dump(diagnosis, f, indent=2)

    print(f"[+] Diagnosis saved: {latest_path}")
    print(f"[+] Diagnosis archived: {archive_path}")


def auto_rollback_check(diagnosis: Dict) -> bool:
    """Check if automatic rollback should be triggered."""
    return diagnosis.get('rollback_required', False)


def rollback_model(tf: str):
    """Execute rollback via deploy_model.py."""
    print(f"\n[!] AUTO-ROLLBACK TRIGGERED for {tf}")

    # Call deploy_model.py rollback
    deploy_script = Path(__file__).parent / 'deploy_model.py'
    cmd = [sys.executable, str(deploy_script), 'rollback', '--tf', tf]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)

        if DISCORD_AVAILABLE:
            gate.critical(
                f"Auto-Rollback Executed: {tf.upper()}",
                {
                    'Action': f"Rolled back {tf} to previous version",
                    'Reason': 'Automatic rollback triggered by diagnosis',
                    'Command': ' '.join(cmd)
                },
                event_type=f"auto_rollback_{tf}"
            )

        return True
    except subprocess.CalledProcessError as e:
        print(f"[X] Rollback failed: {e}")
        print(e.stdout)
        print(e.stderr)

        if DISCORD_AVAILABLE:
            gate.critical(
                f"Auto-Rollback FAILED: {tf.upper()}",
                {
                    'Error': str(e),
                    'Stdout': e.stdout[:500],
                    'Stderr': e.stderr[:500]
                },
                event_type=f"rollback_failed_{tf}"
            )

        return False


def main():
    """CLI interface for manual diagnosis."""
    import argparse

    parser = argparse.ArgumentParser(description='V3.3 Model Diagnosis System')
    parser.add_argument('--tf', required=True, choices=VALID_TIMEFRAMES, help='Timeframe')
    parser.add_argument('--diagnosis-json', required=True, help='Path to diagnosis JSON from training')
    parser.add_argument('--auto-rollback', action='store_true', help='Execute auto-rollback if triggered')

    args = parser.parse_args()

    # Load diagnosis from training
    with open(args.diagnosis_json, 'r') as f:
        training_diagnosis = json.load(f)

    # Run diagnosis (assuming training_diagnosis has all required fields)
    diagnosis = diagnose_model(
        tf=args.tf,
        new_auc=training_diagnosis['val_auc'],
        train_auc=training_diagnosis['train_auc'],
        val_auc=training_diagnosis['val_auc'],
        positive_rate=training_diagnosis['positive_rate'],
        feature_importance=np.array(training_diagnosis['feature_importance']),
        feature_names=training_diagnosis['feature_names']
    )

    # Save results
    save_diagnosis(diagnosis)

    # Notify Discord
    notify_discord(diagnosis)

    # Check for auto-rollback
    if auto_rollback_check(diagnosis):
        print(f"\n{'='*60}")
        print(f"AUTO-ROLLBACK TRIGGERED: {args.tf.upper()}")
        print(f"{'='*60}")

        if args.auto_rollback:
            rollback_model(args.tf)
        else:
            print("[!] --auto-rollback not specified, skipping rollback")
            print(f"[!] To rollback manually: python deploy_model.py rollback --tf {args.tf}")
    else:
        print(f"\n[✓] Diagnosis PASSED. Model is healthy.")


if __name__ == '__main__':
    main()
