#!/usr/bin/env python3
"""
Usage monitor for KB-first + Perplexity-fallback evidence.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import convention_gate
import ops_kb


def _project_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _format_entry(entry: dict, max_preview: int = 120) -> str:
    preview = entry["content"].replace("\n", " ")[:max_preview]
    if len(entry["content"]) > max_preview:
        preview += "..."
    return f"[{entry['id']}] {entry['added_at']} | {entry['kind'].upper()} | {preview}"


def _timestamp(entry: dict) -> datetime:
    return convention_gate._entry_dt(entry) or datetime.min


def _records(task: str | None, hours: int) -> list:
    return sorted(
        convention_gate._load_research_entries(_project_dir(), task_token=task, hours=hours),
        key=lambda entry: (_timestamp(entry), entry["id"]),
    )


def cmd_timeline(task: str | None, hours: int) -> int:
    entries = _records(task, hours)
    if not entries:
        print(f"No research evidence found for task '{task}' in last {hours}h.")
        return 0

    print(f"RESEARCH TIMELINE: task={task} hours={hours}")
    print("-" * 80)
    for entry in entries:
        src = entry.get("source") or convention_gate._primary_source_hint(entry)
        print(f"{_format_entry(entry)} | provider={src}")
    return 0


def _pairing_failures(entries: list) -> list[str]:
    ordered = sorted(entries, key=lambda entry: (_timestamp(entry), entry["id"]))
    failures = []
    if not ordered:
        return failures

    gap_positions = [i for i, e in enumerate(ordered) if e["kind"] == "kb_gap"]
    perplexity_positions = [i for i, e in enumerate(ordered) if e["kind"] == "perplexity_source"]

    if perplexity_positions and not gap_positions:
        failures.append("perplexity_source exists without any kb_gap marker.")
    if perplexity_positions:
        first_perplexity = perplexity_positions[0]
        if all(g > first_perplexity for g in gap_positions):
            failures.append("earliest perplexity_source appears before any kb_gap.")
    for gap_pos in gap_positions:
        gap_added = ordered[gap_pos]["added_at"]
        if not any(p > gap_pos for p in perplexity_positions):
            failures.append(f"kb_gap at {gap_added} has no later perplexity_source.")
    return failures


def cmd_audit(task: str | None, hours: int, require_kb_sources: bool, require_perplexity: bool) -> int:
    entries = _records(task, hours)
    if not entries:
        print("No research evidence found.")
        return 1

    failures = []
    if require_kb_sources:
        providers = {
            convention_gate._primary_source_hint(entry)
            for entry in entries
            if entry["kind"] == "kb_query"
        }
        if not {"socraticode", "orgonite"} & providers:
            failures.append("kb_query entries are missing explicit SocratiCode/Orgonite markers.")
    if require_perplexity and not any(entry["kind"] == "perplexity_source" for entry in entries):
        failures.append("--require-perplexity requested, but no perplexity_source found.")
    failures.extend(_pairing_failures(entries))

    counts = {kind: 0 for kind in ("kb_query", "kb_source", "kb_gap", "perplexity_source")}
    for entry in entries:
        if entry["kind"] in counts:
            counts[entry["kind"]] += 1
    if counts["kb_query"] == 0:
        failures.append("No kb_query log found.")

    for failure in failures:
        print(f"FAIL: {failure}")

    print("SUMMARY:")
    print(f"  KB_QUERY: {counts['kb_query']}")
    print(f"  KB_SOURCE: {counts['kb_source']}")
    print(f"  KB_GAP: {counts['kb_gap']}")
    print(f"  PERPLEXITY_SOURCE: {counts['perplexity_source']}")
    print("\nRecent timeline:")
    cmd_timeline(task, hours)
    return 1 if failures else 0


def cmd_db_audit(hours: int) -> int:
    entries = _records(None, hours)
    if not entries:
        print(f"No KB research records found in last {hours}h.")
        return 0

    summary = {
        "kb_query": 0,
        "kb_source": 0,
        "kb_gap": 0,
        "perplexity_source": 0,
        "other": 0,
    }
    task_map = {}
    for entry in entries:
        summary[entry["kind"]] = summary.get(entry["kind"], 0) + 1
        task = ops_kb._extract_task_token(entry["content"]) or "ungrouped"
        task_map.setdefault(task, {"kb_query": 0, "kb_source": 0, "kb_gap": 0, "perplexity_source": 0})
        task_map[task][entry["kind"]] = task_map[task].get(entry["kind"], 0) + 1

    print(f"DB AUDIT WINDOW: last {hours}h")
    print(f"  KB_QUERY={summary['kb_query']}")
    print(f"  KB_SOURCE={summary['kb_source']}")
    print(f"  KB_GAP={summary['kb_gap']}")
    print(f"  PERPLEXITY_SOURCE={summary['perplexity_source']}\n")
    print("SUSPECT TASKS (perplexity without gap):")
    printed = False
    for task, counts in sorted(task_map.items()):
        if counts["perplexity_source"] > 0 and counts["kb_gap"] == 0:
            print(f"  {task}: gaps={counts['kb_gap']} perplexity={counts['perplexity_source']}")
            printed = True
    if not printed:
        print("  none")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monitor KB and Perplexity usage evidence.")
    sub = parser.add_subparsers(dest="command", required=True)

    timeline = sub.add_parser("timeline", help="Show research timeline for a task.")
    timeline.add_argument("--task", "-t", required=True)
    timeline.add_argument("--hours", "-H", type=int, default=72)

    audit = sub.add_parser("assert", help="Validate KB-first + fallback order.")
    audit.add_argument("--task", "-t", required=True)
    audit.add_argument("--hours", "-H", type=int, default=72)
    audit.add_argument("--require-kb-first-sources", action="store_true")
    audit.add_argument("--require-perplexity", action="store_true")

    db_audit = sub.add_parser("db", help="Print DB-wide KB-source/perplexity usage summary.")
    db_audit.add_argument("--hours", "-H", type=int, default=168)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "timeline":
        return cmd_timeline(args.task, args.hours)
    if args.command == "assert":
        return cmd_audit(
            args.task,
            args.hours,
            require_kb_sources=args.require_kb_first_sources,
            require_perplexity=args.require_perplexity,
        )
    if args.command == "db":
        return cmd_db_audit(args.hours)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
