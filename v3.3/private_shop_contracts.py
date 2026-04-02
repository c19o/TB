#!/usr/bin/env python
"""Load the private-shop controls contract."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PRIVATE_SHOP_CONTROLS = SCRIPT_DIR / "contracts" / "private_shop_controls.json"


@lru_cache(maxsize=4)
def load_private_shop_controls(path: str | Path | None = None) -> dict[str, Any]:
    target = Path(path) if path else DEFAULT_PRIVATE_SHOP_CONTROLS
    with target.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def timeframe_private_shop_controls(tf: str, path: str | Path | None = None) -> dict[str, Any]:
    payload = load_private_shop_controls(path)
    try:
        return payload["timeframes"][tf]
    except KeyError as exc:
        raise KeyError(f"Unknown private-shop timeframe: {tf}") from exc


def governance_states(path: str | Path | None = None) -> list[str]:
    payload = load_private_shop_controls(path)
    return list(payload.get("governance_states", []))
