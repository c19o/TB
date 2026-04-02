#!/usr/bin/env python
"""Helpers for local runtime-home management and source-only repo hygiene."""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CODE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_ROOT.parent
DEFAULT_RUNTIME_HOME = PROJECT_ROOT.parent / "Savage22 Runtime"
RUNTIME_HOME_ENV = "SAVAGE22_RUNTIME_HOME"
RUNTIME_SUBDIRS = (
    "shared_db",
    "runs",
    "artifacts",
    "logs",
    "archives",
    "downloads",
    "cache",
)
SCAN_ROOTS = (PROJECT_ROOT, CODE_ROOT)
SKIP_NAMES = {
    ".git",
    ".worktrees",
    ".codex",
    ".planning",
    ".venv",
    "node_modules",
    "__pycache__",
}

FILE_CATEGORY_RULES: tuple[tuple[str, str], ...] = (
    ("*.db", "shared_db"),
    ("kp_history_gfz.txt", "shared_db"),
    ("features_*.parquet", "artifacts"),
    ("v2_crosses_*.npz", "artifacts"),
    ("v2_cross_names_*.json", "artifacts"),
    ("inference_*.json", "artifacts"),
    ("inference_*.npz", "artifacts"),
    ("_cross_checkpoint_*", "artifacts"),
    ("model_*.json", "artifacts"),
    ("model_*.txt", "artifacts"),
    ("optuna_configs_*.json", "artifacts"),
    ("optimizer_configs_*.json", "artifacts"),
    ("platt_*.pkl", "artifacts"),
    ("cpcv_oos_predictions_*.pkl", "artifacts"),
    ("meta_model_*.pkl", "artifacts"),
    ("lstm_*.pt", "artifacts"),
    ("validation_report_*.json", "artifacts"),
    ("backend_certification_*.json", "artifacts"),
    ("rare_feature_health_*.json", "artifacts"),
    ("training_inference_parity_*.json", "artifacts"),
    ("model_governance_*.json", "artifacts"),
    ("audit_report.*", "artifacts"),
    ("audit_heatmap.html", "artifacts"),
    ("unified_audit_report.*", "artifacts"),
    ("release_manifest.json", "runs"),
    ("pipeline_manifest.json", "runs"),
    ("*.tgz", "archives"),
    ("*.tar.gz", "archives"),
    ("*.zip", "archives"),
    ("*.npy", "cache"),
    ("*.bin", "cache"),
    ("*.log", "logs"),
)
DIR_CATEGORY_RULES: tuple[tuple[str, str], ...] = (
    ("cloud_results_*", "downloads"),
    ("runpod_output", "downloads"),
    ("old_run_holddominated", "archives"),
    ("v2_run_balanced_labels", "archives"),
    ("_build", "cache"),
)


@dataclass(frozen=True)
class RuntimeFinding:
    path: Path
    category: str
    suggested_destination: Path
    is_dir: bool


def runtime_home() -> Path:
    raw = os.environ.get(RUNTIME_HOME_ENV, "").strip()
    return Path(raw).expanduser().resolve() if raw else DEFAULT_RUNTIME_HOME.resolve()


def runtime_dir(name: str) -> Path:
    if name not in RUNTIME_SUBDIRS:
        raise KeyError(f"unknown runtime subdir: {name}")
    return runtime_home() / name


def ensure_runtime_home() -> dict[str, Path]:
    root = runtime_home()
    root.mkdir(parents=True, exist_ok=True)
    created = {}
    for name in RUNTIME_SUBDIRS:
        path = root / name
        path.mkdir(parents=True, exist_ok=True)
        created[name] = path
    return created


def _matches_any(name: str, patterns: Iterable[tuple[str, str]]) -> str | None:
    for pattern, category in patterns:
        if fnmatch.fnmatch(name, pattern):
            return category
    return None


def classify_entry(path: Path) -> str | None:
    category = _matches_any(path.name, DIR_CATEGORY_RULES if path.is_dir() else FILE_CATEGORY_RULES)
    if category:
        return category
    if path.parent.name == "__pycache__":
        return "cache"
    if path.suffix.lower() in {".parquet", ".npz", ".pkl"}:
        return "artifacts"
    return None


def should_skip(path: Path) -> bool:
    return any(part in SKIP_NAMES for part in path.parts)


def has_runtime_parent(path: Path) -> bool:
    for parent in path.parents:
        if parent in SCAN_ROOTS:
            break
        if _matches_any(parent.name, DIR_CATEGORY_RULES):
            return True
    return False


def suggested_destination(path: Path, category: str) -> Path:
    root_tag = "v3.3" if CODE_ROOT in path.parents or path == CODE_ROOT else "repo"
    return runtime_dir(category) / root_tag / path.name


def inventory_runtime_clutter() -> list[RuntimeFinding]:
    findings: list[RuntimeFinding] = []
    seen: set[Path] = set()
    for scan_root in SCAN_ROOTS:
        if not scan_root.exists():
            continue
        for current_root, dirnames, filenames in os.walk(scan_root):
            root_path = Path(current_root)
            if root_path != scan_root:
                root_category = classify_entry(root_path)
                if root_category and root_path.resolve() not in seen and not should_skip(root_path):
                    seen.add(root_path.resolve())
                    findings.append(
                        RuntimeFinding(
                            path=root_path,
                            category=root_category,
                            suggested_destination=suggested_destination(root_path, root_category),
                            is_dir=True,
                        )
                    )
                    dirnames[:] = []
                    continue
            dirnames[:] = [name for name in dirnames if name not in SKIP_NAMES]
            for dirname in list(dirnames):
                candidate = root_path / dirname
                if candidate.resolve() in seen or should_skip(candidate) or has_runtime_parent(candidate):
                    continue
                category = classify_entry(candidate)
                if category:
                    dirnames.remove(dirname)
                    seen.add(candidate.resolve())
                    findings.append(
                        RuntimeFinding(
                            path=candidate,
                            category=category,
                            suggested_destination=suggested_destination(candidate, category),
                            is_dir=True,
                        )
                    )
            for filename in filenames:
                candidate = root_path / filename
                if candidate.resolve() in seen or should_skip(candidate) or has_runtime_parent(candidate):
                    continue
                category = classify_entry(candidate)
                if not category:
                    continue
                seen.add(candidate.resolve())
                findings.append(
                    RuntimeFinding(
                        path=candidate,
                        category=category,
                        suggested_destination=suggested_destination(candidate, category),
                        is_dir=False,
                    )
                )
    return sorted(findings, key=lambda item: str(item.path))
