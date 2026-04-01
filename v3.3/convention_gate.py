#!/usr/bin/env python3
"""
convention_gate.py — Automated convention enforcement for Savage22 V3.3

Three layers:
  1. Pattern detector (grep-style forbidden patterns)
  2. AST structural checks (binarization, batch assignment, Numba)
  3. Meta-audit (discover ungated convention violations in recent commits)

Usage:
  python convention_gate.py check                    # Run all gates on modified files
  python convention_gate.py check feature_library.py  # Run on specific file
  python convention_gate.py meta-audit               # Find ungated violations in recent commits
  python convention_gate.py full                     # Both check + meta-audit

Allowlist mechanism:
  Add # noqa: convention to any line to suppress violations on that line.
  Example:
    result = X.toarray()  # noqa: convention  (small co-occurrence matrix)
"""

import ast
import re
import sys
import os
import sqlite3
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter

# Fix Windows cp1252 encoding
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Layer 1: Forbidden Pattern Detector ──────────────────────────────────────

FORBIDDEN_PATTERNS = [
    # Sparse matrix violations (full matrix conversion — small-subset .toarray() is allowed)
    (r"\.toarray\(\)", "SPARSE: .toarray() converts sparse to dense — forbidden on cross features"),
    (r"\.todense\(\)", "SPARSE: .todense() converts sparse to dense — forbidden on cross features"),

    # XGBoost
    (r"import\s+xgboost", "XGBOOST: XGBoost import forbidden — LightGBM only"),
    (r"from\s+xgboost", "XGBOOST: XGBoost import forbidden — LightGBM only"),
    (r"import\s+xgb\b", "XGBOOST: XGBoost alias import forbidden"),

    # NaN handling — only flag in training/ML files, not in feature construction
    # (feature_library.py legitimately uses fillna(0) on raw metadata like earthquake counts)
    (r"fillna\s*\(\s*0\s*\)", "NAN: fillna(0) converts NaN→0 — LightGBM handles NaN natively"),
    (r"np\.nan_to_num\(", "NAN: nan_to_num converts NaN→0 — use NaN for missing values"),
    (r"\.replace\(\s*np\.nan\s*,\s*0", "NAN: replace(nan, 0) — LightGBM handles NaN natively"),

    # Feature fraction / bagging (in Optuna configs)
    (r"suggest_float\s*\(\s*['\"]feature_fraction['\"].*?,\s*0\.[0-6]", "SACRED: feature_fraction floor < 0.7 — kills rare cross signals"),
    (r"suggest_float\s*\(\s*['\"]bagging_fraction['\"].*?,\s*0\.[0-8]", "SACRED: bagging_fraction floor < 0.9 — kills rare signal training examples"),

    # Dense conversion hints
    (r"\.to_numpy\(\).*dense", "SPARSE: possible dense conversion — verify not on cross features"),

    # feature_pre_filter — match only when the VALUE is True (not another param on same line)
    (r"feature_pre_filter['\"\s:=]+True", "SACRED: feature_pre_filter must be False — True silently kills rare features"),

    # Performance: .apply(lambda) on DataFrames (vectorize instead)
    (r"\.apply\(\s*lambda", "PERF: .apply(lambda) on DataFrame — vectorize with numpy/pandas ops for 10-100x speedup"),

    # Hardcoded Windows paths (use Path() or os.path.join)
    (r"['\"]C:\\\\", "PATH: Hardcoded Windows path C:\\ — use Path() or config.py for cross-platform compatibility"),
    (r"['\"]D:\\\\", "PATH: Hardcoded Windows path D:\\ — use Path() or config.py for cross-platform compatibility"),
]

# Patterns to SKIP (false positive zones — comments, docstrings, test files)
SKIP_PATTERNS = [
    r"^\s*#",       # Comments
    r"^\s*\"\"\"",  # Docstrings
    r"^\s*\'\'\'",  # Docstrings
]

# Files excluded from ALL pattern scanning (they check FOR violations, not commit them)
EXCLUDED_FILES = {
    "validate.py",        # The validator itself references forbidden patterns in check messages
    "convention_gate.py", # This file defines the forbidden patterns
    "test_gpu_accuracy.py",   # Test files use .toarray() for accuracy verification
    "train_1w_cached.py",     # Chunked .toarray() for GPU prediction (small blocks)
}

# Per-file pattern exclusions: {filename: set of message prefixes to skip}
# These are known legitimate uses that would be false positives
PER_FILE_EXCLUSIONS = {
    "feature_library.py": {"NAN"},     # fillna(0)/nan_to_num on raw metadata, not model features
    "v2_cross_generator.py": {"SPARSE"},  # Co-occurrence matrices are small, not full cross features
    "validate.py": {"NAN"},            # Validator checks FOR violations, doesn't commit them
}

def check_forbidden_patterns(filepath: str) -> list:
    """Scan file for forbidden patterns."""
    basename = os.path.basename(filepath)
    if basename in EXCLUDED_FILES:
        return []

    file_exclusions = PER_FILE_EXCLUSIONS.get(basename, set())
    violations = []
    try:
        source = Path(filepath).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    for i, line in enumerate(source.splitlines(), 1):
        # Skip comments and docstrings
        if any(re.match(sp, line) for sp in SKIP_PATTERNS):
            continue

        # Skip lines with noqa: convention comment
        if re.search(r'#\s*noqa:\s*convention', line, re.IGNORECASE):
            continue

        # Strip inline comments before matching (prevents false positives
        # where the pattern appears only in a trailing comment)
        code_part = line.split('#')[0] if '#' in line else line
        for pattern, message in FORBIDDEN_PATTERNS:
            # Skip patterns excluded for this file
            msg_prefix = message.split(':')[0]
            if msg_prefix in file_exclusions:
                continue
            if re.search(pattern, code_part):
                # Allow .toarray()/.todense() on sliced subsets and co-occurrence matrices
                if '.toarray()' in code_part or '.todense()' in code_part:
                    # Safe patterns: sliced columns X[:,idx], sliced rows X[i:j],
                    # subset variables, co-occurrence results, single column extraction
                    if re.search(r'\[.*[:,:].*\]\.to(?:array|dense)\(\)', code_part):
                        continue  # sliced subset — safe
                    if re.search(r'_(?:slice|subset|small|esoteric)\.to(?:array|dense)\(\)', code_part):
                        continue  # named subset/small variable — safe
                    if re.search(r'(?:cooc|co_oc|cooccur).*\.to(?:array|dense)\(\)', code_part):
                        continue  # co-occurrence matrix variable — small result
                    if re.search(r'(?:\.T\s*@|@.*\.T).*\.to(?:array|dense)\(\)', code_part):
                        continue  # co-occurrence matrix (sparse.T @ sparse) — small result
                    if re.search(r'_mkl_dot\(.*\.to(?:array|dense)\(\)', code_part):
                        continue  # MKL sparse dot product result — small
                    if re.search(r'getcol\(.*\.to(?:array|dense)\(\)', code_part):
                        continue  # single column extraction — safe
                    if re.search(r'getrow\(.*\.to(?:array|dense)\(\)', code_part):
                        continue  # single row extraction — safe
                    if re.search(r'asnumpy\(.*\.to(?:array|dense)\(\)', code_part):
                        continue  # GPU co-occurrence result — small
                    if re.search(r'\[\s*:\s*,\s*(?:astro|moon|eclipse|gem|hebrew).*\]\.to(?:array|dense)\(\)', code_part):
                        continue  # esoteric feature subset — small
                violations.append({
                    "file": filepath,
                    "line": i,
                    "code": line.strip()[:100],
                    "message": message,
                    "layer": "PATTERN",
                })
    return violations


# ── Layer 2: AST Structural Checks ──────────────────────────────────────────

class OneAtATimeAssignChecker(ast.NodeVisitor):
    """Detect df[col] = val inside for/while loops (should use batch assignment)."""

    def __init__(self, filepath):
        self.filepath = filepath
        self.violations = []
        self._in_loop = False

    def visit_For(self, node):
        old = self._in_loop
        self._in_loop = True
        self.generic_visit(node)
        self._in_loop = old

    def visit_While(self, node):
        old = self._in_loop
        self._in_loop = True
        self.generic_visit(node)
        self._in_loop = old

    def visit_Assign(self, node):
        if self._in_loop:
            for target in node.targets:
                if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                    if target.value.id in ("df", "features", "X", "result", "out"):
                        self.violations.append({
                            "file": self.filepath,
                            "line": node.lineno,
                            "code": f"df[...] = ... inside loop",
                            "message": "BATCH: One-at-a-time column assignment inside loop — use dict accumulation + pd.concat",
                            "layer": "AST",
                        })
        self.generic_visit(node)


class XGBoostImportChecker(ast.NodeVisitor):
    """Detect any XGBoost imports at AST level."""

    def __init__(self, filepath):
        self.filepath = filepath
        self.violations = []

    def visit_Import(self, node):
        for alias in node.names:
            if "xgboost" in alias.name.lower() or alias.name == "xgb":
                self.violations.append({
                    "file": self.filepath,
                    "line": node.lineno,
                    "code": f"import {alias.name}",
                    "message": "XGBOOST: XGBoost import forbidden — LightGBM only",
                    "layer": "AST",
                })

    def visit_ImportFrom(self, node):
        if node.module and "xgboost" in node.module.lower():
            self.violations.append({
                "file": self.filepath,
                "line": node.lineno,
                "code": f"from {node.module} import ...",
                "message": "XGBOOST: XGBoost import forbidden — LightGBM only",
                "layer": "AST",
            })


# Files excluded from AST batch-assignment checks (known tech debt, separate refactor)
AST_BATCH_EXCLUDED = {
    "feature_library.py",    # 50+ loop assignments — massive refactor, tracked separately
    "v2_cross_generator.py", # Cross gen builds columns in batch loops by design
}

def check_ast(filepath: str) -> list:
    """Run AST-based structural checks on a Python file."""
    basename = os.path.basename(filepath)
    if basename in EXCLUDED_FILES:
        return []

    violations = []
    try:
        source = Path(filepath).read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except SyntaxError:
        return [{"file": filepath, "line": 0, "code": "", "message": "SYNTAX: File has syntax errors", "layer": "AST"}]
    except Exception:
        return []

    # Build a set of lines with noqa comments for fast lookup
    source_lines = source.splitlines()
    noqa_lines = set()
    for i, line in enumerate(source_lines, 1):
        if re.search(r'#\s*noqa:\s*convention', line, re.IGNORECASE):
            noqa_lines.add(i)

    checkers = [XGBoostImportChecker]
    if basename not in AST_BATCH_EXCLUDED:
        checkers.append(OneAtATimeAssignChecker)

    for CheckerClass in checkers:
        checker = CheckerClass(filepath)
        checker.visit(tree)
        # Filter out violations on noqa lines
        for v in checker.violations:
            if v['line'] not in noqa_lines:
                violations.append(v)

    return violations


# ── Layer 3: Protected Prefix Coverage ───────────────────────────────────────

def check_prefix_coverage(project_dir: str) -> list:
    """Check that all feature prefixes in feature_library.py are in PROTECTED_FEATURE_PREFIXES."""
    violations = []
    config_path = os.path.join(project_dir, "config.py")
    fl_path = os.path.join(project_dir, "feature_library.py")

    if not os.path.exists(config_path) or not os.path.exists(fl_path):
        return []

    config_src = Path(config_path).read_text(encoding="utf-8", errors="replace")
    fl_src = Path(fl_path).read_text(encoding="utf-8", errors="replace")

    # Known prefixes that MUST be protected if used
    REQUIRED_PREFIXES = [
        "gem_", "dr_", "moon_", "nakshatra", "eclipse", "vedic_", "bazi_",
        "hebrew_", "sw_", "aspect_", "vortex_", "sephirah", "chakra_",
        "jupiter_", "mercury_", "planetary_", "angel", "master_", "palindrome",
        "doy_", "dx_", "ax_", "ex2_", "asp_", "pn_", "mn_", "bio_",
        "saros_", "metonic_", "rahu_", "ketu_", "fib_", "gann_", "biorhythm_",
        "fib_ret_", "fib_ext_", "near_fib",
    ]

    for prefix in REQUIRED_PREFIXES:
        # Is it used in feature_library.py?
        if f'"{prefix}' in fl_src or f"'{prefix}" in fl_src:
            # Is it in config.py PROTECTED_FEATURE_PREFIXES?
            if f'"{prefix}' not in config_src and f"'{prefix}" not in config_src:
                violations.append({
                    "file": config_path,
                    "line": 0,
                    "code": f"prefix: {prefix}",
                    "message": f"PREFIX: '{prefix}' used in feature_library.py but MISSING from PROTECTED_FEATURE_PREFIXES",
                    "layer": "PREFIX",
                })

    return violations


# ── Layer 4: Meta-Audit (Gate Gap Analysis) ──────────────────────────────────

# Convention rules that are NOT yet in automated gates
# When meta-audit finds these firing frequently, they become gate candidates
UNGATED_CONVENTIONS = [
    ("no_inplace_sort", r"sort_values.*inplace\s*=\s*True", "STYLE: inplace sort_values — prefer assignment"),
    ("no_apply_loops", r"\.apply\(\s*lambda", "PERF: .apply(lambda) is slow — use vectorized ops or Numba"),
    ("no_raw_python_loop_on_prices", r"for\s+\w+\s+in\s+.*prices", "PERF: Raw Python loop on prices — use Numba @njit"),
    ("no_hardcoded_paths", r"['\"]C:\\\\|['\"]C:/Users", "DEPLOY: Hardcoded Windows path — use config.py"),
    ("no_print_debug", r"\bprint\s*\(\s*['\"]debug|print\s*\(\s*f['\"]debug", "STYLE: Debug print statement left in code"),
    ("no_magic_numbers_in_optuna", r"suggest_(?:float|int)\([^)]*\b(?:100|1000|0\.001|0\.0001)\b", "OPTUNA: Suspicious magic number in Optuna range"),
    ("no_dense_matrix_multiply", r"np\.dot\(.*sparse|sparse.*np\.dot", "SPARSE: np.dot on sparse matrix — use sparse_dot_mkl or @ operator"),
]


def meta_audit(project_dir: str, days: int = 30) -> list:
    """Find ungated convention violations in recent commits."""
    proposals = []

    # Get recently changed Python files (last N commits)
    repo_root = project_dir
    while repo_root and not os.path.isdir(os.path.join(repo_root, ".git")):
        parent = os.path.dirname(repo_root)
        if parent == repo_root:
            break
        repo_root = parent

    try:
        result = subprocess.run(
            ["git", "log", f"-{days}", "--name-only", "--pretty=format:"],
            capture_output=True, text=True, cwd=repo_root
        )
        recent_files = sorted(set(
            f for f in result.stdout.strip().split("\n")
            if f.endswith(".py") and f.strip() and os.path.exists(os.path.join(repo_root, f))
        ))
        # Make full paths
        recent_files = [os.path.join(repo_root, f) for f in recent_files]
    except Exception:
        recent_files = []

    if not recent_files:
        print(f"No Python files changed in last {days} days.")
        return []

    hit_counts = Counter()
    hit_details = {}

    for filepath in recent_files:
        full_path = filepath  # already full path after fix above
        try:
            source = Path(full_path).read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        for rule_id, pattern, message in UNGATED_CONVENTIONS:
            matches = [(i, line.strip()) for i, line in enumerate(source.splitlines(), 1)
                       if re.search(pattern, line) and not re.match(r"^\s*#", line)]
            if matches:
                hit_counts[rule_id] += 1
                hit_details.setdefault(rule_id, []).extend(
                    [(filepath, ln, code[:80]) for ln, code in matches[:3]]
                )

    # Rules firing in >20% of changed files → propose as gate candidates
    threshold = max(1, len(recent_files) * 0.2)
    for rule_id, count in hit_counts.most_common():
        status = "PROMOTE_TO_GATE" if count >= threshold else "MONITOR"
        message = next(m for rid, _, m in UNGATED_CONVENTIONS if rid == rule_id)
        proposals.append({
            "rule_id": rule_id,
            "hits": count,
            "files_changed": len(recent_files),
            "pct": round(100 * count / len(recent_files), 1),
            "status": status,
            "message": message,
            "examples": hit_details.get(rule_id, [])[:3],
        })

    return sorted(proposals, key=lambda x: -x["hits"])


# ── Main Runner ──────────────────────────────────────────────────────────────

def get_modified_files(project_dir: str) -> list:
    """Get Python files modified (uncommitted + last 5 commits)."""
    files = set()
    repo_root = project_dir
    # Walk up to find repo root
    while repo_root and not os.path.isdir(os.path.join(repo_root, ".git")):
        parent = os.path.dirname(repo_root)
        if parent == repo_root:
            break
        repo_root = parent

    try:
        # Uncommitted changes
        for cmd in [["git", "diff", "--name-only", "HEAD"], ["git", "diff", "--name-only", "--staged"]]:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root)
            files.update(f for f in result.stdout.strip().split("\n") if f.strip())

        # Last 5 commits (catch recently committed agent work)
        result = subprocess.run(
            ["git", "log", "-5", "--name-only", "--pretty=format:"],
            capture_output=True, text=True, cwd=repo_root
        )
        files.update(f for f in result.stdout.strip().split("\n") if f.strip())
    except Exception:
        pass

    return [
        os.path.join(repo_root, f)
        for f in files
        if f.endswith(".py") and os.path.exists(os.path.join(repo_root, f))
    ]


def run_check(project_dir: str, specific_file: str = None):
    """Run all gate checks."""
    if specific_file:
        files = [os.path.join(project_dir, specific_file)] if not os.path.isabs(specific_file) else [specific_file]
    else:
        files = get_modified_files(project_dir)

    if not files:
        print("No modified Python files to check.")
        return 0

    all_violations = []

    for filepath in files:
        if not filepath.endswith(".py"):
            continue
        all_violations.extend(check_forbidden_patterns(filepath))
        all_violations.extend(check_ast(filepath))

    # Always check prefix coverage
    all_violations.extend(check_prefix_coverage(project_dir))

    if all_violations:
        print(f"\n{'='*60}")
        print(f"  CONVENTION GATE: {len(all_violations)} violation(s) found")
        print(f"{'='*60}\n")
        for v in all_violations:
            print(f"  [{v['layer']}] {os.path.basename(v['file'])}:{v['line']}")
            print(f"    {v['message']}")
            if v['code']:
                print(f"    Code: {v['code']}")
            print()
        return 1
    else:
        print("CONVENTION GATE: ALL CHECKS PASSED")
        return 0


def run_meta_audit(project_dir: str):
    """Run the gate gap analysis."""
    proposals = meta_audit(project_dir)

    if not proposals:
        print("META-AUDIT: No ungated violations found in recent commits.")
        return

    print(f"\n{'='*60}")
    print(f"  META-AUDIT: Gate Gap Analysis")
    print(f"{'='*60}\n")

    for p in proposals:
        marker = ">>> PROMOTE" if p["status"] == "PROMOTE_TO_GATE" else "    monitor"
        print(f"  {marker} | {p['rule_id']} | {p['hits']}/{p['files_changed']} files ({p['pct']}%)")
        print(f"    {p['message']}")
        for filepath, ln, code in p["examples"]:
            print(f"    Example: {filepath}:{ln} → {code}")
        print()


def _parse_iso_timestamp(value: str):
    """Parse ops_kb timestamp values safely."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _ops_kb_db_path(project_dir: str) -> str:
    """Resolve the ops_kb SQLite path."""
    return os.path.join(project_dir, "ops_kb", "db", "ops_kb.db")


def _entry_kind(entry: dict) -> str:
    """Normalize research evidence types across old/new topic conventions."""
    topic = (entry.get("topic") or "").lower()
    content = entry.get("content") or ""
    if topic == "kb_query" or "KB_QUERY:" in content:
        return "kb_query"
    if topic == "kb_source" or "KB_SOURCE:" in content:
        return "kb_source"
    if topic == "perplexity_source" or "PERPLEXITY_SOURCE:" in content:
        return "perplexity_source"
    if topic == "kb_gap" or "KB_GAP:" in content:
        return "kb_gap"
    return "other"


def _load_research_entries(project_dir: str, hours: int = 72, task_token: str = None) -> list:
    """Load recent research evidence from ops_kb, optionally scoped to one task token."""
    db_path = _ops_kb_db_path(project_dir)
    if not os.path.exists(db_path):
        return []

    cutoff = datetime.now() - timedelta(hours=hours)
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT id, topic, content, added_at FROM entries ORDER BY id ASC"
    ).fetchall()
    conn.close()

    task_needle = task_token.lower() if task_token else None
    entries = []
    for row_id, topic, content, added_at in rows:
        parsed = _parse_iso_timestamp(added_at)
        if parsed and parsed < cutoff:
            continue
        haystack = f"{topic}\n{content}".lower()
        if task_needle and task_needle not in haystack:
            continue
        if _entry_kind({"topic": topic, "content": content}) == "other":
            continue
        entries.append({
            "id": row_id,
            "topic": topic,
            "content": content,
            "added_at": added_at,
        })
    return entries


def _entry_dt(entry: dict):
    """Return parsed datetime for a research entry when available."""
    return _parse_iso_timestamp(entry.get("added_at"))


def run_research_audit(project_dir: str, task_token: str = None, hours: int = 72, require_perplexity: bool = False):
    """Audit KB-first and Perplexity-fallback evidence in ops_kb."""
    entries = _load_research_entries(project_dir, hours=hours, task_token=task_token)
    scope = task_token or "recent research activity"

    if not entries:
        print(f"RESEARCH AUDIT: FAIL - no research evidence found for {scope} in last {hours}h.")
        print("Expected at minimum: KB_QUERY plus KB_SOURCE or KB_GAP/PERPLEXITY_SOURCE.")
        return 1

    counts = Counter(_entry_kind(entry) for entry in entries)
    failures = []

    if counts["kb_query"] == 0:
        failures.append("Missing KB_QUERY log. KB-first cannot be proven.")

    if counts["kb_source"] == 0 and counts["kb_gap"] == 0:
        failures.append("Missing KB_SOURCE or KB_GAP log. KB sufficiency verdict was never recorded.")

    if counts["perplexity_source"] > 0 and counts["kb_gap"] == 0:
        failures.append("PERPLEXITY_SOURCE exists without a prior KB_GAP. Fallback was not justified.")

    if counts["kb_gap"] > 0 and counts["perplexity_source"] == 0:
        failures.append("KB_GAP logged without PERPLEXITY_SOURCE fallback evidence.")

    if require_perplexity and counts["perplexity_source"] == 0:
        failures.append("--require-perplexity set, but no PERPLEXITY_SOURCE was found.")

    gap_entries = [entry for entry in entries if _entry_kind(entry) == "kb_gap"]
    perplexity_entries = [entry for entry in entries if _entry_kind(entry) == "perplexity_source"]
    if gap_entries and perplexity_entries:
        latest_gap = max(gap_entries, key=lambda entry: (_entry_dt(entry) or datetime.min, entry["id"]))
        latest_perplexity = max(perplexity_entries, key=lambda entry: (_entry_dt(entry) or datetime.min, entry["id"]))
        gap_dt = _entry_dt(latest_gap)
        perplexity_dt = _entry_dt(latest_perplexity)
        if gap_dt and perplexity_dt:
            if perplexity_dt < gap_dt:
                failures.append("Latest KB_GAP is newer than latest PERPLEXITY_SOURCE. Fallback log sequence is incomplete.")
        elif latest_perplexity["id"] < latest_gap["id"]:
            failures.append("Latest KB_GAP is newer than latest PERPLEXITY_SOURCE. Fallback log sequence is incomplete.")

    print(f"\n{'='*60}")
    print(f"  RESEARCH AUDIT: {scope}")
    print(f"{'='*60}\n")
    print(f"  Window: last {hours}h")
    print(f"  KB_QUERY: {counts['kb_query']}")
    print(f"  KB_SOURCE: {counts['kb_source']}")
    print(f"  KB_GAP: {counts['kb_gap']}")
    print(f"  PERPLEXITY_SOURCE: {counts['perplexity_source']}\n")

    for entry in entries[-5:]:
        kind = _entry_kind(entry).upper()
        preview = entry["content"][:140].replace("\n", " ")
        print(f"  [{entry['id']}] {entry['added_at']} [{kind}] {preview}")

    if failures:
        print(f"\nRESEARCH AUDIT: FAIL ({len(failures)} issue(s))")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\nRESEARCH AUDIT: PASS")
    return 0


if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.abspath(__file__))

    if len(sys.argv) < 2:
        print("Usage: python convention_gate.py [check|meta-audit|research-audit|full] [file|task]")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "check":
        specific = sys.argv[2] if len(sys.argv) > 2 else None
        exit_code = run_check(project_dir, specific)
        sys.exit(exit_code)
    elif cmd == "meta-audit":
        run_meta_audit(project_dir)
    elif cmd == "research-audit":
        task_token = None
        hours = 72
        require_perplexity = False

        args = sys.argv[2:]
        i = 0
        while i < len(args):
            arg = args[i]
            if arg == "--hours" and i + 1 < len(args):
                hours = int(args[i + 1])
                i += 2
            elif arg == "--require-perplexity":
                require_perplexity = True
                i += 1
            elif not task_token:
                task_token = arg
                i += 1
            else:
                print(f"Unknown argument: {arg}")
                sys.exit(1)

        exit_code = run_research_audit(
            project_dir,
            task_token=task_token,
            hours=hours,
            require_perplexity=require_perplexity,
        )
        sys.exit(exit_code)
    elif cmd == "full":
        exit_code = run_check(project_dir)
        run_meta_audit(project_dir)
        sys.exit(exit_code)
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
