#!/usr/bin/env python3
"""
deploy_manifest.py — Generate SHA256 manifest of all .py files for cloud deployment verification.

Run LOCALLY before SCP:
    python deploy_manifest.py

Generates deploy_manifest.json with:
  - SHA256 hash of every .py file in v3.3/
  - Timestamp of generation
  - Python version used to generate
  - Total file count

The manifest is SCP'd alongside code. deploy_verify.py reads it on the cloud
to detect stale/corrupt/missing files.
"""
import hashlib
import json
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(SCRIPT_DIR, "deploy_manifest.json")

# Files that are deployment-critical (pipeline will fail without them)
CRITICAL_FILES = [
    "config.py",
    "feature_library.py",
    "ml_multi_tf.py",
    "run_optuna_local.py",
    "cloud_run_tf.py",
    "validate.py",
    "runtime_checks.py",
    "v2_cross_generator.py",
    "multi_gpu_optuna.py",
    "data_access.py",
    "v2_feature_layers.py",
    "meta_labeling.py",
    "lstm_sequence_model.py",
    "exhaustive_optimizer.py",
    "memmap_merge.py",
    "efb_prebundler.py",
    "deploy_verify.py",
    "test_pipeline_plumbing.py",
    "backtest_validation.py",
    "inference_crosses.py",
    "live_trader.py",
    "hardware_detect.py",
    "bitpack_utils.py",
]

# Root-level files that get deployed alongside v3.3/
# (astrology_engine.py was moved into v3.3/ -- add here if any root .py files are SCP'd)
ROOT_FILES = []


def sha256_file(filepath):
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print(f"Generating deployment manifest from: {SCRIPT_DIR}")

    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "generated_on": os.uname().nodename if hasattr(os, "uname") else os.environ.get("COMPUTERNAME", "unknown"),
        "python_version": sys.version,
        "files": {},
        "critical_files": CRITICAL_FILES,
    }

    # Hash all .py files in v3.3/
    py_files = sorted(f for f in os.listdir(SCRIPT_DIR) if f.endswith(".py"))
    for fname in py_files:
        fpath = os.path.join(SCRIPT_DIR, fname)
        if os.path.isfile(fpath):
            manifest["files"][fname] = {
                "sha256": sha256_file(fpath),
                "size": os.path.getsize(fpath),
            }

    # Hash root-level files that get SCP'd
    root_dir = os.path.dirname(SCRIPT_DIR)
    for fname in ROOT_FILES:
        fpath = os.path.join(root_dir, fname)
        if os.path.isfile(fpath):
            manifest["files"][f"../{fname}"] = {
                "sha256": sha256_file(fpath),
                "size": os.path.getsize(fpath),
            }
        else:
            print(f"  WARNING: root file not found: {fname}")

    manifest["total_files"] = len(manifest["files"])

    # Verify all critical files are present
    missing_critical = []
    for cf in CRITICAL_FILES:
        if cf not in manifest["files"]:
            missing_critical.append(cf)
            print(f"  CRITICAL MISSING: {cf}")

    if missing_critical:
        print(f"\nERROR: {len(missing_critical)} critical files missing. Fix before deploying.")
        sys.exit(1)

    # Write manifest
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest written: {MANIFEST_PATH}")
    print(f"  Total .py files: {manifest['total_files']}")
    print(f"  Critical files:  {len(CRITICAL_FILES)} (all present)")
    print(f"\nNext: SCP this manifest alongside your code tarball.")
    print(f"  Then on cloud: python deploy_verify.py --tf <TF>")


if __name__ == "__main__":
    main()
