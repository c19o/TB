#!/usr/bin/env python
"""Inventory or migrate runtime clutter out of the maintained source tree."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from runtime_home import ensure_runtime_home, inventory_runtime_clutter


def _serialize(findings):
    return [
        {
            "path": str(item.path),
            "category": item.category,
            "is_dir": item.is_dir,
            "suggested_destination": str(item.suggested_destination),
        }
        for item in findings
    ]


def _move_item(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError(f"destination already exists: {dest}")
    shutil.move(str(src), str(dest))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--migrate", action="store_true", help="Move detected runtime clutter into SAVAGE22_RUNTIME_HOME")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()

    ensure_runtime_home()
    findings = inventory_runtime_clutter()

    if args.migrate:
        for item in findings:
            _move_item(item.path, item.suggested_destination)
        findings = inventory_runtime_clutter()

    if args.json:
        json.dump(
            {
                "remaining_findings": _serialize(findings),
                "remaining_count": len(findings),
                "migrated": bool(args.migrate),
            },
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
        return 0

    print(f"Runtime home audit findings: {len(findings)}")
    for item in findings:
        kind = "DIR " if item.is_dir else "FILE"
        print(f"  [{kind}] {item.path} -> {item.suggested_destination} ({item.category})")
    if args.migrate:
        print("Migration complete.")
    else:
        print("Dry run only. Re-run with --migrate to move these paths.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
