#!/usr/bin/env python3
"""
Savage22 Operational Memory KB
================================
Lightweight SQLite FTS5 knowledge base for project operational memory.
Stores what's been tried, training results, decisions, bug attempts.

Usage:
    python ops_kb.py add "FACT: 15m batch_size=16 OOMs on 44GB A40" --topic oom
    python ops_kb.py add path/to/file.txt --topic training_result
    python ops_kb.py search "batch size 15m"
    python ops_kb.py smart "what was tried for daemon RELOAD bug"
    python ops_kb.py stats
    python ops_kb.py list [--topic <tag>] [--limit N]

Topics: training_result, bug_attempt, oom, deployment, decision, feature_audit, general
"""
import sys
import os
import sqlite3
import json
import datetime
import textwrap
import click

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ops_kb", "db", "ops_kb.db")
VALID_TOPICS = {"training_result", "bug_attempt", "oom", "deployment", "decision", "feature_audit", "general"}

# Prevent Windows cp1252 console crashes when entries contain unicode symbols.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# --- DB Init ---

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS entries (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            topic    TEXT NOT NULL DEFAULT 'general',
            content  TEXT NOT NULL,
            source   TEXT,
            added_at TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS entries_fts USING fts5(
            content,
            topic,
            content='entries',
            content_rowid='id'
        );
        CREATE TRIGGER IF NOT EXISTS entries_ai AFTER INSERT ON entries BEGIN
            INSERT INTO entries_fts(rowid, content, topic) VALUES (new.id, new.content, new.topic);
        END;
        CREATE TRIGGER IF NOT EXISTS entries_ad AFTER DELETE ON entries BEGIN
            INSERT INTO entries_fts(entries_fts, rowid, content, topic) VALUES ('delete', old.id, old.content, old.topic);
        END;
    """)
    conn.commit()
    return conn


def add_entry(content: str, topic: str = "general", source: str = None):
    conn = init_db()
    now = datetime.datetime.now().isoformat(timespec="seconds")
    conn.execute(
        "INSERT INTO entries (topic, content, source, added_at) VALUES (?, ?, ?, ?)",
        (topic, content.strip(), source, now)
    )
    conn.commit()
    conn.close()


def ingest_file(path: str, topic: str = "general"):
    """Ingest a text/JSON file as entries (one entry per logical chunk)."""
    ext = os.path.splitext(path)[1].lower()
    added = 0
    if ext == ".json":
        with open(path, encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                text = json.dumps(item, ensure_ascii=False)
                add_entry(text, topic=topic, source=os.path.basename(path))
                added += 1
        else:
            add_entry(json.dumps(data, ensure_ascii=False), topic=topic, source=os.path.basename(path))
            added = 1
    else:
        with open(path, encoding="utf-8", errors="replace") as f:
            raw = f.read()
        # Chunk into ~1500 char pieces
        chunk_size = 1500
        for i in range(0, len(raw), chunk_size - 200):
            chunk = raw[i:i + chunk_size].strip()
            if chunk:
                add_entry(chunk, topic=topic, source=os.path.basename(path))
                added += 1
    return added


def _do_search(query: str, topic: str = None, limit: int = 10):
    conn = init_db()
    try:
        if topic:
            rows = conn.execute(
                "SELECT e.id, e.topic, e.content, e.added_at, e.source FROM entries_fts f "
                "JOIN entries e ON e.id = f.rowid "
                "WHERE entries_fts MATCH ? AND f.topic = ? ORDER BY rank LIMIT ?",
                (query, topic, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT e.id, e.topic, e.content, e.added_at, e.source FROM entries_fts f "
                "JOIN entries e ON e.id = f.rowid "
                "WHERE entries_fts MATCH ? ORDER BY rank LIMIT ?",
                (query, limit)
            ).fetchall()
    except Exception:
        rows = []
    conn.close()
    return rows


def list_entries(topic: str = None, limit: int = 20):
    conn = init_db()
    if topic:
        rows = conn.execute(
            "SELECT id, topic, content, added_at, source FROM entries WHERE topic=? ORDER BY id DESC LIMIT ?",
            (topic, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, topic, content, added_at, source FROM entries ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()
    conn.close()
    return rows


def get_stats():
    conn = init_db()
    total = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
    by_topic = conn.execute(
        "SELECT topic, COUNT(*) FROM entries GROUP BY topic ORDER BY COUNT(*) DESC"
    ).fetchall()
    conn.close()
    return total, by_topic


# --- CLI ---

@click.group()
def cli():
    """Savage22 Operational Memory KB — track what's been tried."""


@cli.command()
@click.argument("target")
@click.option("--topic", "-t", default="general", help="Topic tag")
def add(target, topic):
    """Add a fact string or ingest a file/folder.

    \b
    Examples:
        ops_kb.py add "FACT: 1d cross-gen took 52min, NNZ=1.4B" --topic training_result
        ops_kb.py add discord_gate_log.json --topic deployment
        ops_kb.py add SESSION_RESUME.md --topic training_result
    """
    if topic not in VALID_TOPICS:
        click.echo(f"Warning: unknown topic '{topic}'. Valid: {', '.join(sorted(VALID_TOPICS))}")

    if os.path.isfile(target):
        n = ingest_file(target, topic=topic)
        click.echo(f"Ingested {n} chunks from {os.path.basename(target)} [{topic}]")
    elif os.path.isdir(target):
        total = 0
        for fname in os.listdir(target):
            fpath = os.path.join(target, fname)
            if os.path.isfile(fpath) and os.path.splitext(fname)[1].lower() in {".txt", ".json", ".md"}:
                n = ingest_file(fpath, topic=topic)
                total += n
                click.echo(f"  {fname}: {n} chunks")
        click.echo(f"Total: {total} chunks ingested")
    else:
        # Treat as a raw fact string
        add_entry(target, topic=topic)
        click.echo(f"Added [{topic}]: {target[:80]}{'...' if len(target) > 80 else ''}")


@cli.command()
@click.argument("query")
@click.option("--topic", "-t", default=None, help="Filter by topic")
@click.option("--n", "-n", default=10, help="Max results")
def search(query, topic, n):
    """Keyword search (FTS5/BM25). Use for exact terms."""
    rows = _do_search(query, topic=topic, limit=n)
    _print_results(rows, query)


@cli.command()
@click.argument("query")
@click.option("--topic", "-t", default=None, help="Filter by topic")
@click.option("--n", "-n", default=10, help="Max results")
def smart(query, topic, n):
    """Smart search — tries keyword + OR expansion + phrase match."""
    results = {}
    for q in [query, query.replace(" ", " OR "), f'"{query}"']:
        for row in _do_search(q, topic=topic, limit=n):
            if row[0] not in results:
                results[row[0]] = row
    rows = list(results.values())[:n]
    _print_results(rows, query)


@cli.command("list")
@click.option("--topic", "-t", default=None, help="Filter by topic")
@click.option("--limit", "-n", default=20, help="Max entries")
def list_cmd(topic, limit):
    """List recent entries."""
    rows = list_entries(topic=topic, limit=limit)
    if not rows:
        click.echo("No entries yet.")
        return
    for row in rows:
        eid, etopic, content, added_at, source = row
        preview = content[:100].replace("\n", " ")
        src = f" | {source}" if source else ""
        click.echo(f"[{eid}] {added_at} [{etopic}{src}]")
        click.echo(f"  {preview}{'...' if len(content) > 100 else ''}")
        click.echo()


@cli.command()
def stats():
    """Show database statistics."""
    total, by_topic = get_stats()
    click.echo(f"\n  Savage22 Ops KB — {total} entries\n")
    click.echo(f"  {'Topic':<20} {'Count':>6}")
    click.echo(f"  {'-'*20} {'-'*6}")
    for t, c in by_topic:
        click.echo(f"  {t:<20} {c:>6}")
    click.echo()


def _print_results(rows, query):
    if not rows:
        click.echo(f"No results for: {query}")
        return
    click.echo(f"\n  {len(rows)} result(s) for: '{query}'\n")
    for row in rows:
        eid, etopic, content, added_at, source = row
        src = f" | {source}" if source else ""
        click.echo(f"  [{eid}] [{etopic}{src}] {added_at}")
        wrapped = textwrap.fill(content[:400], width=80, initial_indent="    ", subsequent_indent="    ")
        click.echo(wrapped)
        if len(content) > 400:
            click.echo("    ...")
        click.echo()


if __name__ == "__main__":
    cli()
