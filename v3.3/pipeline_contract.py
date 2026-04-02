#!/usr/bin/env python
"""Canonical timeframe-aware pipeline contract loader.

This module is the data source for maintained pipeline contracts. It does not
execute runtime logic; it only loads, normalizes, and queries contract data so
launchers, validators, and training code can converge on one source of truth.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


_CODE_DIR = Path(__file__).resolve().parent
_CONTRACT_DIR = _CODE_DIR / "contracts"
_PIPELINE_CONTRACT_PATH = _CONTRACT_DIR / "pipeline_contract.json"
_LEGACY_WEEKLY_CONTRACT_PATH = _CODE_DIR / "WEEKLY_1W_ARTIFACT_CONTRACT.json"

_DEFAULT_HEARTBEAT_STATUSES = ["running", "validated", "failed", "complete"]
_DEFAULT_RESUME_BOUNDARIES = [
    "features_validated",
    "optuna_validated",
    "train_validated",
    "optimizer_validated",
]


def _as_path(path: str | Path | None) -> Path:
    if path is None:
        return _PIPELINE_CONTRACT_PATH
    return Path(path).expanduser().resolve()


@lru_cache(maxsize=None)
def _load_json(path_str: str) -> dict[str, Any]:
    path = Path(path_str)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _canonicalize_legacy_weekly_contract(raw: dict[str, Any]) -> dict[str, Any]:
    tf = raw.get("tf", "1w")
    wrapped = {
        "schema_version": raw.get("schema_version", 1),
        "contract_name": raw.get("contract_name", "legacy weekly artifact contract"),
        "heartbeat_statuses": raw.get("heartbeat_statuses", list(_DEFAULT_HEARTBEAT_STATUSES)),
        "timeframes": {
            tf: raw,
        },
    }
    tf_entry = wrapped["timeframes"][tf]
    tf_entry.setdefault("cross_policy", "forbidden" if tf == "1w" else "required")
    tf_entry.setdefault("resume_boundaries", list(_DEFAULT_RESUME_BOUNDARIES))
    return wrapped


def _validate_contract_shape(contract: dict[str, Any]) -> None:
    if not isinstance(contract, dict):
        raise ValueError("pipeline contract must be a JSON object")
    if "timeframes" not in contract or not isinstance(contract["timeframes"], dict):
        raise ValueError("pipeline contract must contain a 'timeframes' object")
    if "heartbeat_statuses" in contract:
        statuses = contract["heartbeat_statuses"]
        if not isinstance(statuses, list) or not statuses:
            raise ValueError("pipeline contract heartbeat_statuses must be a non-empty list")
    for tf, tf_contract in contract["timeframes"].items():
        if not isinstance(tf_contract, dict):
            raise ValueError(f"timeframe contract for {tf!r} must be an object")
        phases = tf_contract.get("phases")
        if not isinstance(phases, dict) or not phases:
            raise ValueError(f"timeframe contract for {tf!r} must define phases")
        for phase_name, phase_cfg in phases.items():
            if not isinstance(phase_cfg, dict):
                raise ValueError(f"phase {phase_name!r} for {tf!r} must be an object")
            if "phase_seq" not in phase_cfg:
                raise ValueError(f"phase {phase_name!r} for {tf!r} must define phase_seq")
            required = phase_cfg.get("required_artifacts", [])
            if not isinstance(required, list):
                raise ValueError(f"phase {phase_name!r} for {tf!r} required_artifacts must be a list")


@lru_cache(maxsize=None)
def load_pipeline_contract(contract_path: str | Path | None = None) -> dict[str, Any]:
    """Load the unified pipeline contract and normalize legacy 1w-only files.

    If a canonical unified contract exists under v3.3/contracts/pipeline_contract.json,
    it is preferred. A legacy weekly contract file is accepted as a compatibility
    fallback and wrapped into the unified schema for read-only query helpers.
    """

    path = _as_path(contract_path)
    if path.exists():
        raw = _load_json(str(path))
    elif path == _PIPELINE_CONTRACT_PATH and _LEGACY_WEEKLY_CONTRACT_PATH.exists():
        raw = _load_json(str(_LEGACY_WEEKLY_CONTRACT_PATH))
    else:
        raise FileNotFoundError(f"pipeline contract not found: {path}")

    if "timeframes" not in raw:
        raw = _canonicalize_legacy_weekly_contract(raw)

    _validate_contract_shape(raw)
    return raw


def list_timeframes(contract_path: str | Path | None = None) -> list[str]:
    contract = load_pipeline_contract(contract_path)
    return list(contract["timeframes"].keys())


def load_timeframe_contract(tf: str, contract_path: str | Path | None = None) -> dict[str, Any]:
    contract = load_pipeline_contract(contract_path)
    try:
        tf_contract = contract["timeframes"][tf]
    except KeyError as exc:
        raise KeyError(f"timeframe {tf!r} not present in pipeline contract") from exc
    return tf_contract


def ordered_phase_items(tf: str, contract_path: str | Path | None = None) -> list[tuple[str, dict[str, Any]]]:
    phases = load_timeframe_contract(tf, contract_path)["phases"]
    return sorted(phases.items(), key=lambda item: item[1].get("phase_seq", 10_000))


def ordered_phase_names(tf: str, contract_path: str | Path | None = None) -> list[str]:
    return [name for name, _ in ordered_phase_items(tf, contract_path)]


def phase_contract(tf: str, phase: str, contract_path: str | Path | None = None) -> dict[str, Any]:
    tf_contract = load_timeframe_contract(tf, contract_path)
    try:
        return tf_contract["phases"][phase]
    except KeyError as exc:
        raise KeyError(f"phase {phase!r} not present for timeframe {tf!r}") from exc


def required_artifacts(tf: str, phase: str, contract_path: str | Path | None = None) -> list[str]:
    return list(phase_contract(tf, phase, contract_path).get("required_artifacts", []))


def cross_policy(tf: str, contract_path: str | Path | None = None) -> str:
    return str(load_timeframe_contract(tf, contract_path).get("cross_policy", "required"))


def is_cross_required(tf: str, contract_path: str | Path | None = None) -> bool:
    return cross_policy(tf, contract_path) == "required"


def resume_boundaries(tf: str, contract_path: str | Path | None = None) -> list[str]:
    tf_contract = load_timeframe_contract(tf, contract_path)
    boundaries = tf_contract.get("resume_boundaries", list(_DEFAULT_RESUME_BOUNDARIES))
    if not isinstance(boundaries, list):
        raise ValueError(f"resume_boundaries for {tf!r} must be a list")
    return list(boundaries)


def heartbeat_statuses(contract_path: str | Path | None = None) -> list[str]:
    contract = load_pipeline_contract(contract_path)
    statuses = contract.get("heartbeat_statuses", list(_DEFAULT_HEARTBEAT_STATUSES))
    if not isinstance(statuses, list):
        raise ValueError("heartbeat_statuses must be a list")
    return list(statuses)


def complete_artifacts(tf: str, contract_path: str | Path | None = None) -> list[str]:
    """Return the union of required artifacts across non-skipped phases."""

    required: list[str] = []
    for phase_name, phase_cfg in ordered_phase_items(tf, contract_path):
        if phase_name == "complete":
            continue
        if phase_cfg.get("policy") == "skipped":
            continue
        for artifact in phase_cfg.get("required_artifacts", []):
            if artifact not in required:
                required.append(artifact)
    return required


def summary(tf: str, contract_path: str | Path | None = None) -> dict[str, Any]:
    tf_contract = load_timeframe_contract(tf, contract_path)
    return {
        "tf": tf,
        "cross_policy": tf_contract.get("cross_policy", "required"),
        "resume_boundaries": resume_boundaries(tf, contract_path),
        "phases": ordered_phase_names(tf, contract_path),
        "complete_artifacts": complete_artifacts(tf, contract_path),
    }
