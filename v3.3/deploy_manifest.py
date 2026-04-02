#!/usr/bin/env python3
"""
deploy_manifest.py - Generate SHA256 manifest for release-eligible files.

Run locally before SCP:
    python deploy_manifest.py

Generates deploy_manifest.json with:
  - SHA256 hash of every shipped source/control file in v3.3/
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
    "live_trader.py",
    "hardware_detect.py",
    "bitpack_utils.py",
]

ALLOWED_TOP_LEVEL_JSON = {
    "WEEKLY_1W_ARTIFACT_CONTRACT.json",
}

EXCLUDED_PREFIXES = (
    "_cross_checkpoint_",
    "cpcv_oos_",
    "feature_importance_",
    "features_",
    "inference_",
    "lgbm_dataset_",
    "lgbm_parent_",
    "meta_model_",
    "ml_multi_tf_configs",
    "model_",
    "optuna_model_",
    "optuna_search",
    "pipeline_manifest",
    "shap_analysis_",
    "validation_report_",
    "v2_cross_names_",
    "v2_crosses_",
)


def sha256_file(filepath):
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def should_include_release_file(fname):
    """Return True for files that may ship in a maintained release bundle."""
    if fname in ALLOWED_TOP_LEVEL_JSON:
        return True

    ext = os.path.splitext(fname)[1].lower()
    if ext not in {".py", ".md", ".sh"}:
        return False

    if any(fname.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
        return False

    return True


def main():
    print(f"Generating deployment manifest from: {SCRIPT_DIR}")

    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "generated_on": os.uname().nodename if hasattr(os, "uname") else os.environ.get("COMPUTERNAME", "unknown"),
        "python_version": sys.version,
        "files": {},
        "critical_files": CRITICAL_FILES,
    }

    release_files = sorted(
        f for f in os.listdir(SCRIPT_DIR)
        if os.path.isfile(os.path.join(SCRIPT_DIR, f))
        and should_include_release_file(f)
    )
    for fname in release_files:
        fpath = os.path.join(SCRIPT_DIR, fname)
        manifest["files"][fname] = {
            "sha256": sha256_file(fpath),
            "size": os.path.getsize(fpath),
        }

    manifest["total_files"] = len(manifest["files"])

    missing_critical = []
    for cf in CRITICAL_FILES:
        if cf not in manifest["files"]:
            missing_critical.append(cf)
            print(f"  CRITICAL MISSING: {cf}")

    if missing_critical:
        print(f"\nERROR: {len(missing_critical)} critical files missing. Fix before deploying.")
        sys.exit(1)

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest written: {MANIFEST_PATH}")
    print(f"  Total shipped files: {manifest['total_files']}")
    print(f"  Critical files:  {len(CRITICAL_FILES)} (all present)")
    print("\nNext: SCP this manifest alongside your code tarball.")
    print("  Then on cloud: python deploy_verify.py --tf <TF>")


if __name__ == "__main__":
    main()
