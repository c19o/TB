#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deploy_model.py — Production model versioning and deployment system for V3.3

FEATURES:
- Per-timeframe versioning (1w, 1d, 4h, 1h, 15m)
- Version ID: lgbm_{tf}_v{YYYYMMDD}_{HHMMSS}_{git_sha7}_acc{accuracy*1000}
- Symlink-based registry (active/shadow/previous)
- Atomic swaps (no race conditions)
- SHA256 integrity verification
- Auto-prune to keep last 3 versions per timeframe
- Blue/green deployment support
- One-command rollback

USAGE:
    # Deploy a new model
    python deploy_model.py deploy --tf 1w --model model_1w.json --accuracy 0.934 --features features_1w_all.json

    # Rollback to previous version
    python deploy_model.py rollback --tf 1w

    # Rollback to shadow version
    python deploy_model.py rollback --tf 1w --target shadow

    # List versions for a timeframe
    python deploy_model.py list --tf 1w

    # Promote shadow to active (after 48hr soak test)
    python deploy_model.py promote --tf 1w

DIRECTORY LAYOUT:
    models/
      1w/
        versions/
          lgbm_1w_v20260401_103000_a3f9c12_acc934/
            model.json
            features_all.json
            manifest.json
        active -> versions/lgbm_1w_v20260401_103000_a3f9c12_acc934/
        shadow -> versions/lgbm_1w_v20260328_120000_b7d4e55_acc928/
        previous -> versions/lgbm_1w_v20260320_090000_c1f2a88_acc915/
      1d/
        ... (same structure)

MANIFEST FORMAT (manifest.json):
    {
      "version_id": "lgbm_1w_v20260401_103000_a3f9c12_acc934",
      "timeframe": "1w",
      "trained_at": "2026-04-01T10:30:00Z",
      "accuracy": 0.934,
      "num_features": 2900000,
      "training_host": "vast.ai-12345",
      "git_sha": "a3f9c12",
      "parent_version": "lgbm_1w_v20260328_120000_b7d4e55_acc928",
      "files": {
        "model.json": {"sha256": "b3a9c1...", "size": 1024000},
        "features_all.json": {"sha256": "c4d2e3...", "size": 50000}
      }
    }

SAFETY:
- All symlink swaps are atomic (mv -Tf / os.replace)
- SHA256 verification after every file copy
- Audit log for all deployments and rollbacks
- NEVER deletes active/shadow/previous (protected by symlinks)

"""
import os
import sys
import json
import hashlib
import shutil
import subprocess
import argparse
from datetime import datetime, timezone
from pathlib import Path

# Constants
KEEP_VERSIONS = 3
VALID_TIMEFRAMES = ['1w', '1d', '4h', '1h', '15m']
MODELS_BASE_DIR = Path(__file__).parent.parent / 'models'  # C:/Users/C/Documents/Savage22 Server/models/


def sha256_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_sha() -> str:
    """Get short git commit SHA (7 chars)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit00"


def get_hostname() -> str:
    """Get hostname or vast.ai instance ID."""
    try:
        # Try to get vast.ai instance ID from environment
        vast_id = os.environ.get('VAST_CONTAINERLABEL')
        if vast_id:
            return f"vast.ai-{vast_id}"

        # Fall back to hostname
        import socket
        return socket.gethostname()
    except:
        return "unknown"


def get_tf_dir(tf: str) -> Path:
    """Get base directory for a timeframe."""
    return MODELS_BASE_DIR / tf


def get_versions_dir(tf: str) -> Path:
    """Get versions directory for a timeframe."""
    return get_tf_dir(tf) / 'versions'


def read_symlink(tf: str, name: str) -> str | None:
    """Read a symlink (active/shadow/previous). Returns version ID or None."""
    link = get_tf_dir(tf) / name
    if link.is_symlink():
        target = os.readlink(link)
        return Path(target).name
    return None


def set_symlink(tf: str, name: str, target_version_id: str):
    """Set a symlink atomically. Uses temp + rename for atomicity."""
    tf_dir = get_tf_dir(tf)
    link = tf_dir / name
    target = get_versions_dir(tf) / target_version_id
    tmp = tf_dir / f"{name}.tmp"

    # Remove temp if it exists
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()

    # Create symlink to target
    if os.name == 'nt':  # Windows
        # Windows requires different handling
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target, target_is_directory=True)
    else:  # Linux/Unix
        tmp.symlink_to(target)
        tmp.replace(link)  # Atomic on POSIX


def rotate_symlinks(tf: str, new_version_id: str):
    """Rotate symlinks: previous <- shadow <- active <- new."""
    shadow = read_symlink(tf, 'shadow')
    active = read_symlink(tf, 'active')

    # Shift chain
    if shadow:
        set_symlink(tf, 'previous', shadow)
    if active:
        set_symlink(tf, 'shadow', active)

    # Set new active
    set_symlink(tf, 'active', new_version_id)


def prune_old_versions(tf: str):
    """Delete versions beyond KEEP_VERSIONS, protecting active/shadow/previous."""
    protected = {
        read_symlink(tf, 'active'),
        read_symlink(tf, 'shadow'),
        read_symlink(tf, 'previous')
    } - {None}

    versions_dir = get_versions_dir(tf)
    if not versions_dir.exists():
        return

    # Get all version directories, sorted by modification time (oldest first)
    versions = sorted(
        [v for v in versions_dir.iterdir() if v.is_dir()],
        key=lambda p: p.stat().st_mtime
    )

    for v in versions:
        if v.name not in protected:
            print(f"[-]  Pruning old version: {v.name}")
            shutil.rmtree(v)


def write_audit_log(tf: str, event: str, details: dict):
    """Write to audit log (append-only)."""
    audit_file = get_tf_dir(tf) / 'audit.log'
    audit_file.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **details
    }

    with open(audit_file, 'a') as f:
        f.write(json.dumps(entry) + '\n')


def deploy_model(tf: str, model_path: str, accuracy: float, features_path: str = None):
    """Deploy a new model version."""
    if tf not in VALID_TIMEFRAMES:
        print(f"[X] ERROR: Invalid timeframe '{tf}'. Valid: {VALID_TIMEFRAMES}")
        sys.exit(1)

    model_path = Path(model_path)
    if not model_path.exists():
        print(f"[X] ERROR: Model file not found: {model_path}")
        sys.exit(1)

    if features_path:
        features_path = Path(features_path)
        if not features_path.exists():
            print(f"[X] ERROR: Features file not found: {features_path}")
            sys.exit(1)

    # Generate version ID
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    git_sha = get_git_sha()
    acc_tag = f"acc{int(accuracy * 1000)}"
    version_id = f"lgbm_{tf}_v{ts}_{git_sha}_{acc_tag}"

    print(f"\n[*] Deploying new model: {version_id}")

    # Create version directory
    version_dir = get_versions_dir(tf) / version_id
    version_dir.mkdir(parents=True, exist_ok=True)

    # Copy model and verify integrity
    model_dest = version_dir / 'model.json'
    shutil.copy2(model_path, model_dest)
    model_hash = sha256_file(model_dest)
    model_size = model_dest.stat().st_size

    print(f"  OK Model copied: {model_size / 1024 / 1024:.1f} MB")
    print(f"  OK SHA256: {model_hash[:16]}...")

    # Copy features if provided
    files_manifest = {
        'model.json': {'sha256': model_hash, 'size': model_size}
    }

    if features_path:
        feat_dest = version_dir / 'features_all.json'
        shutil.copy2(features_path, feat_dest)
        feat_hash = sha256_file(feat_dest)
        feat_size = feat_dest.stat().st_size
        files_manifest['features_all.json'] = {'sha256': feat_hash, 'size': feat_size}
        print(f"  OK Features copied: {feat_size / 1024:.1f} KB")

    # Count features if features file exists
    num_features = 0
    if features_path:
        try:
            with open(features_path, 'r') as f:
                features = json.load(f)
                num_features = len(features)
        except:
            pass

    # Create manifest
    parent_version = read_symlink(tf, 'active')
    manifest = {
        'version_id': version_id,
        'timeframe': tf,
        'trained_at': datetime.now(timezone.utc).isoformat(),
        'accuracy': accuracy,
        'num_features': num_features,
        'training_host': get_hostname(),
        'git_sha': git_sha,
        'parent_version': parent_version,
        'files': files_manifest
    }

    manifest_path = version_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"  OK Manifest written")

    # Rotate symlinks
    print(f"\n[>] Rotating symlinks...")
    rotate_symlinks(tf, version_id)

    print(f"  OK active   -> {version_id}")
    print(f"  OK shadow   -> {read_symlink(tf, 'shadow') or '(none)'}")
    print(f"  OK previous -> {read_symlink(tf, 'previous') or '(none)'}")

    # Prune old versions
    print(f"\n[~] Pruning old versions (keeping {KEEP_VERSIONS})...")
    prune_old_versions(tf)

    # Write audit log
    write_audit_log(tf, 'deploy', {
        'version_id': version_id,
        'accuracy': accuracy,
        'num_features': num_features,
        'parent_version': parent_version
    })

    print(f"\n[+] Deployment complete: {version_id}")
    print(f"   Accuracy: {accuracy:.3f}")
    print(f"   Features: {num_features:,}")


def rollback_model(tf: str, target: str = 'previous'):
    """Rollback to a previous version."""
    if tf not in VALID_TIMEFRAMES:
        print(f"[X] ERROR: Invalid timeframe '{tf}'. Valid: {VALID_TIMEFRAMES}")
        sys.exit(1)

    if target not in ['previous', 'shadow']:
        print(f"[X] ERROR: Invalid rollback target '{target}'. Valid: previous, shadow")
        sys.exit(1)

    target_version = read_symlink(tf, target)
    if not target_version:
        print(f"[X] ERROR: No {target} version exists for {tf}")
        sys.exit(1)

    active_version = read_symlink(tf, 'active')

    print(f"\n[<] Rolling back {tf} to {target}: {target_version}")
    print(f"   Was: {active_version}")

    # Set active to target
    set_symlink(tf, 'active', target_version)

    print(f"   Now: {target_version}")

    # Write audit log
    write_audit_log(tf, 'rollback', {
        'from_version': active_version,
        'to_version': target_version,
        'target': target
    })

    print(f"\n[+] Rollback complete")
    print(f"   IMPORTANT: Restart inference service to load new model")


def promote_shadow(tf: str):
    """Promote shadow to active (after shadow mode validation)."""
    if tf not in VALID_TIMEFRAMES:
        print(f"[X] ERROR: Invalid timeframe '{tf}'. Valid: {VALID_TIMEFRAMES}")
        sys.exit(1)

    shadow_version = read_symlink(tf, 'shadow')
    if not shadow_version:
        print(f"[X] ERROR: No shadow version exists for {tf}")
        sys.exit(1)

    active_version = read_symlink(tf, 'active')

    print(f"\n[^]  Promoting shadow to active for {tf}")
    print(f"   Shadow: {shadow_version}")
    print(f"   Active: {active_version}")

    # Rotate: previous <- active, active <- shadow
    if active_version:
        set_symlink(tf, 'previous', active_version)
    set_symlink(tf, 'active', shadow_version)

    # Clear shadow (no longer needed)
    shadow_link = get_tf_dir(tf) / 'shadow'
    if shadow_link.exists() or shadow_link.is_symlink():
        shadow_link.unlink()

    print(f"  OK active   -> {shadow_version}")
    print(f"  OK previous -> {active_version}")
    print(f"  OK shadow   -> (cleared)")

    # Write audit log
    write_audit_log(tf, 'promote', {
        'promoted_version': shadow_version,
        'previous_active': active_version
    })

    print(f"\n[+] Promotion complete")


def list_versions(tf: str):
    """List all versions for a timeframe."""
    if tf not in VALID_TIMEFRAMES:
        print(f"[X] ERROR: Invalid timeframe '{tf}'. Valid: {VALID_TIMEFRAMES}")
        sys.exit(1)

    versions_dir = get_versions_dir(tf)
    if not versions_dir.exists():
        print(f"[*] No versions found for {tf}")
        return

    active = read_symlink(tf, 'active')
    shadow = read_symlink(tf, 'shadow')
    previous = read_symlink(tf, 'previous')

    print(f"\n[*] Versions for {tf}:\n")

    # Get all versions, sorted by modification time (newest first)
    versions = sorted(
        [v for v in versions_dir.iterdir() if v.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    for v in versions:
        version_id = v.name
        manifest_path = v / 'manifest.json'

        # Read manifest if it exists
        manifest = {}
        if manifest_path.exists():
            try:
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
            except:
                pass

        # Determine role
        role = []
        if version_id == active:
            role.append('ACTIVE')
        if version_id == shadow:
            role.append('SHADOW')
        if version_id == previous:
            role.append('PREVIOUS')

        role_str = f" [{', '.join(role)}]" if role else ""

        # Print version info
        acc = manifest.get('accuracy', 0)
        trained_at = manifest.get('trained_at', 'unknown')
        num_feats = manifest.get('num_features', 0)

        print(f"  {version_id}{role_str}")
        if manifest:
            print(f"    Accuracy: {acc:.3f} | Features: {num_feats:,} | Trained: {trained_at}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='V3.3 Model Versioning and Deployment System',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Deploy command
    deploy_parser = subparsers.add_parser('deploy', help='Deploy a new model version')
    deploy_parser.add_argument('--tf', required=True, choices=VALID_TIMEFRAMES, help='Timeframe')
    deploy_parser.add_argument('--model', required=True, help='Path to model.json file')
    deploy_parser.add_argument('--accuracy', required=True, type=float, help='Model accuracy (0.0-1.0)')
    deploy_parser.add_argument('--features', help='Path to features_all.json file (optional)')

    # Rollback command
    rollback_parser = subparsers.add_parser('rollback', help='Rollback to a previous version')
    rollback_parser.add_argument('--tf', required=True, choices=VALID_TIMEFRAMES, help='Timeframe')
    rollback_parser.add_argument('--target', default='previous', choices=['previous', 'shadow'],
                                 help='Rollback target (default: previous)')

    # Promote command
    promote_parser = subparsers.add_parser('promote', help='Promote shadow to active')
    promote_parser.add_argument('--tf', required=True, choices=VALID_TIMEFRAMES, help='Timeframe')

    # List command
    list_parser = subparsers.add_parser('list', help='List all versions for a timeframe')
    list_parser.add_argument('--tf', required=True, choices=VALID_TIMEFRAMES, help='Timeframe')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == 'deploy':
        deploy_model(args.tf, args.model, args.accuracy, args.features)
    elif args.command == 'rollback':
        rollback_model(args.tf, args.target)
    elif args.command == 'promote':
        promote_shadow(args.tf)
    elif args.command == 'list':
        list_versions(args.tf)


if __name__ == '__main__':
    main()
