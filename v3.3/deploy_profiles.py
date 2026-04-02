#!/usr/bin/env python
"""Loader and helpers for maintained deployment profiles."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from pipeline_contract import load_timeframe_contract


_CODE_DIR = Path(__file__).resolve().parent
_CONTRACT_DIR = _CODE_DIR / "contracts"
_DEPLOY_PROFILES_PATH = _CONTRACT_DIR / "deploy_profiles.json"
GPU_REQUIRED_EXECUTION_MODES = {"gpu_required", "gpu_required_same_machine"}


def _as_path(path: str | Path | None) -> Path:
    if path is None:
        return _DEPLOY_PROFILES_PATH
    return Path(path).expanduser().resolve()


@lru_cache(maxsize=None)
def _load_json(path_str: str) -> dict[str, Any]:
    with Path(path_str).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_shape(data: dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise ValueError("deploy profiles must be a JSON object")
    if not isinstance(data.get("timeframes"), dict) or not data["timeframes"]:
        raise ValueError("deploy profiles must contain non-empty timeframes object")


@lru_cache(maxsize=None)
def load_deploy_profiles(path: str | Path | None = None) -> dict[str, Any]:
    resolved = _as_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"deploy profiles not found: {resolved}")
    data = _load_json(str(resolved))
    _validate_shape(data)
    return data


def load_timeframe_profile(tf: str, path: str | Path | None = None) -> dict[str, Any]:
    profiles = load_deploy_profiles(path)
    try:
        return profiles["timeframes"][tf]
    except KeyError as exc:
        raise KeyError(f"timeframe {tf!r} not present in deploy profiles") from exc


def runtime_home_default(path: str | Path | None = None) -> str:
    profiles = load_deploy_profiles(path)
    if os.name == "nt":
        return str(profiles.get("runtime_home_default_windows", ""))
    return str(profiles.get("runtime_home_default_posix", ""))


def env_defaults(tf: str, path: str | Path | None = None) -> dict[str, str]:
    profile = load_timeframe_profile(tf, path)
    payload = profile.get("env_defaults", {})
    if not isinstance(payload, dict):
        raise ValueError(f"env_defaults for {tf!r} must be an object")
    return {str(k): str(v) for k, v in payload.items()}


def execution_mode(tf: str, path: str | Path | None = None) -> str:
    profile = load_timeframe_profile(tf, path)
    mode = str(profile.get("execution_mode", "")).strip()
    if not mode:
        raise ValueError(f"execution_mode missing for {tf!r}")
    return mode


def execution_policy(tf: str, path: str | Path | None = None) -> dict[str, Any]:
    profile = load_timeframe_profile(tf, path)
    payload = profile.get("execution_policy", {})
    if not isinstance(payload, dict):
        raise ValueError(f"execution_policy for {tf!r} must be an object")
    return dict(payload)


def requires_cupy(tf: str, path: str | Path | None = None) -> bool:
    if execution_mode(tf, path) in GPU_REQUIRED_EXECUTION_MODES:
        return True
    contract = load_timeframe_contract(tf)
    optimizer_phase = contract.get("phases", {}).get("step6_optimizer", {})
    required_artifacts = optimizer_phase.get("required_artifacts", [])
    policy = str(optimizer_phase.get("policy", "")).strip().lower()
    return bool(required_artifacts) and policy != "skipped"


def warm_start_parent(tf: str, path: str | Path | None = None) -> str | None:
    return load_timeframe_profile(tf, path).get("warm_start_parent")


def machine_policy(tf: str, path: str | Path | None = None) -> dict[str, Any]:
    profile = load_timeframe_profile(tf, path)
    payload = profile.get("machine_policy", {})
    if not isinstance(payload, dict):
        raise ValueError(f"machine_policy for {tf!r} must be an object")
    return payload


def post_train_required(tf: str, path: str | Path | None = None) -> list[str]:
    profile = load_timeframe_profile(tf, path)
    payload = profile.get("post_train_required", [])
    if not isinstance(payload, list):
        raise ValueError(f"post_train_required for {tf!r} must be a list")
    return list(payload)


def summary(tf: str, path: str | Path | None = None) -> dict[str, Any]:
    profile = load_timeframe_profile(tf, path)
    return {
        "tf": tf,
        "warm_start_parent": profile.get("warm_start_parent"),
        "execution_mode": execution_mode(tf, path),
        "same_machine_required": bool(profile.get("same_machine_required", False)),
        "machine_policy": machine_policy(tf, path),
        "env_defaults": env_defaults(tf, path),
        "post_train_required": post_train_required(tf, path),
    }
