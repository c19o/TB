#!/usr/bin/env python
"""
Canonical path contract for Savage22 release-mode runs.

Separates immutable code from shared DB inputs, per-run artifacts,
and per-run control state.
"""

from __future__ import annotations

import os
from pathlib import Path


def _real_dir(path: str) -> str:
    return os.path.realpath(os.path.abspath(path))


def _prefer_existing_runtime_root(runtime_path: str, fallback_path: str) -> str:
    if os.environ.get("SAVAGE22_RUNTIME_HOME"):
        return runtime_path
    return runtime_path if os.path.exists(runtime_path) else fallback_path


def _default_runtime_home() -> str:
    code_root = Path(__file__).resolve().parent
    project_root = code_root.parent
    return str((project_root.parent / "Savage22 Runtime").resolve())


RUNTIME_HOME = _real_dir(os.environ.get("SAVAGE22_RUNTIME_HOME", _default_runtime_home()))
_runtime_db_root = os.path.join(RUNTIME_HOME, "shared_db")
_runtime_artifact_root = os.path.join(RUNTIME_HOME, "artifacts")
_runtime_run_root = os.path.join(RUNTIME_HOME, "runs")


CODE_ROOT = _real_dir(os.path.dirname(__file__))
_legacy_db_root = os.path.dirname(CODE_ROOT)
SHARED_DB_ROOT = _real_dir(os.environ.get("SAVAGE22_DB_DIR", _prefer_existing_runtime_root(_runtime_db_root, _legacy_db_root)))
V1_ROOT = _real_dir(os.environ.get("SAVAGE22_V1_DIR", SHARED_DB_ROOT))
ARTIFACT_ROOT = _real_dir(
    os.environ.get(
        "SAVAGE22_ARTIFACT_DIR",
        os.environ.get("V30_DATA_DIR", _runtime_artifact_root),
    )
)
RUN_ROOT = _real_dir(
    os.environ.get(
        "SAVAGE22_RUN_DIR",
        _runtime_run_root,
    )
)
QUARANTINE_ROOT = os.path.join(RUN_ROOT, "quarantine")


def ensure_runtime_dirs() -> None:
    for path in (RUNTIME_HOME, SHARED_DB_ROOT, ARTIFACT_ROOT, RUN_ROOT, QUARANTINE_ROOT):
        os.makedirs(path, exist_ok=True)


def artifact_path(*parts: str) -> str:
    return os.path.join(ARTIFACT_ROOT, *parts)


def run_path(*parts: str) -> str:
    return os.path.join(RUN_ROOT, *parts)


def quarantine_path(*parts: str) -> str:
    return os.path.join(QUARANTINE_ROOT, *parts)


def db_path(name: str) -> str:
    return os.path.join(SHARED_DB_ROOT, name)


def v1_path(name: str) -> str:
    return os.path.join(V1_ROOT, name)


def code_path(*parts: str) -> str:
    return os.path.join(CODE_ROOT, *parts)


def artifact_candidates(*names: str) -> list[str]:
    return [artifact_path(name) for name in names]


def first_existing(paths: list[str]) -> str | None:
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def run_scoped_dir(namespace: str, *parts: str) -> str:
    safe = namespace.replace(os.sep, "_")
    return os.path.join(ARTIFACT_ROOT, "_runtime", safe, *parts)


def artifact_provenance_path(name: str) -> str:
    return artifact_path(f"{name}.meta.json")


def is_under(child: str, parent: str) -> bool:
    try:
        Path(child).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False
