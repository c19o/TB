#!/usr/bin/env python3
"""Generate a recursive SHA256 manifest for maintained release files."""

import hashlib
import fnmatch
import json
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(SCRIPT_DIR, "deploy_manifest.json")

# Files that are deployment-critical (pipeline will fail without them)
CRITICAL_FILES = [
    "config.py",
    "path_contract.py",
    "pipeline_contract.py",
    "deploy_profiles.py",
    "deploy_tf.py",
    "runtime_home.py",
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
    "contracts/pipeline_contract.json",
    "contracts/deploy_profiles.json",
]

ALLOWED_TOP_LEVEL_JSON = {
    "WEEKLY_1W_ARTIFACT_CONTRACT.json",
}

RECURSIVE_INCLUDE_DIRS = {
    "contracts",
    "docs",
    "gpu_histogram_fork/src",
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

EXCLUDED_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".git",
    ".worktrees",
    "_build",
    "node_modules",
}

EXCLUDED_GLOBS = (
    "cloud_results_*",
    "old_run_holddominated*",
    "v2_run_balanced_labels*",
    "*.db",
    "*.parquet",
    "*.pkl",
    "*.npz",
    "*.npy",
    "*.bin",
    "*.zip",
    "*.tgz",
    "*.tar.gz",
    "*.pyc",
    "*.pyo",
)


def sha256_file(filepath):
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _relpath(path):
    return os.path.relpath(path, SCRIPT_DIR).replace("\\", "/")


def should_include_release_file(relpath):
    """Return True for files that may ship in a maintained release bundle."""
    base = os.path.basename(relpath)

    if base in ALLOWED_TOP_LEVEL_JSON and "/" not in relpath:
        return True

    if any(part in EXCLUDED_DIR_NAMES for part in relpath.split("/")):
        return False

    for pattern in EXCLUDED_GLOBS:
        if os.path.basename(relpath) and fnmatch.fnmatch(base, pattern):
            return False

    if any(base.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
        return False

    ext = os.path.splitext(base)[1].lower()
    top_level = "/" not in relpath
    if top_level:
        return ext in {".py", ".md", ".sh", ".json"}

    if relpath.startswith("contracts/"):
        return ext == ".json"
    if relpath.startswith("docs/"):
        return ext == ".md"
    if relpath.startswith("gpu_histogram_fork/src/"):
        return ext in {".py", ".cu", ".h"}
    return False


def iter_release_files():
    release_files = []
    for root, dirnames, filenames in os.walk(SCRIPT_DIR):
        rel_root = _relpath(root)
        if rel_root == ".":
            rel_root = ""
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIR_NAMES]
        for fname in filenames:
            full = os.path.join(root, fname)
            rel = _relpath(full)
            if should_include_release_file(rel):
                release_files.append(rel)
    return sorted(set(release_files))


def build_manifest():
    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "generated_on": os.uname().nodename if hasattr(os, "uname") else os.environ.get("COMPUTERNAME", "unknown"),
        "python_version": sys.version,
        "files": {},
        "critical_files": CRITICAL_FILES,
    }

    for relpath in iter_release_files():
        fpath = os.path.join(SCRIPT_DIR, relpath.replace("/", os.sep))
        manifest["files"][relpath] = {
            "sha256": sha256_file(fpath),
            "size": os.path.getsize(fpath),
        }

    manifest["total_files"] = len(manifest["files"])
    return manifest


def main():
    print(f"Generating deployment manifest from: {SCRIPT_DIR}")

    manifest = build_manifest()

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
