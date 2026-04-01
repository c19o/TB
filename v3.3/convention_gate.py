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
"""

import ast
import re
import sys
import os
import subprocess
from pathlib import Path
from collections import Counter

# Fix Windows cp1252 encoding
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Layer 1: Forbidden Pattern Detector ──────────────────────────────────────

FORBIDDEN_PATTERNS = [
    # Sparse matrix violations
    (r"\.toarray\(\)", "SPARSE: .toarray() converts sparse to dense — forbidden on cross features"),
    (r"\.todense\(\)", "SPARSE: .todense() converts sparse to dense — forbidden on cross features"),

    # XGBoost
    (r"import\s+xgboost", "XGBOOST: XGBoost import forbidden — LightGBM only"),
    (r"from\s+xgboost", "XGBOOST: XGBoost import forbidden — LightGBM only"),
    (r"import\s+xgb\b", "XGBOOST: XGBoost alias import forbidden"),

    # NaN handling
    (r"fillna\s*\(\s*0\s*\)", "NAN: fillna(0) converts NaN→0 — LightGBM handles NaN natively"),
    (r"np\.nan_to_num\(", "NAN: nan_to_num converts NaN→0 — use NaN for missing values"),
    (r"\.replace\(\s*np\.nan\s*,\s*0", "NAN: replace(nan, 0) — LightGBM handles NaN natively"),

    # Feature fraction / bagging (in Optuna configs)
    (r"suggest_float\s*\(\s*['\"]feature_fraction['\"].*?,\s*0\.[0-6]", "SACRED: feature_fraction floor < 0.7 — kills rare cross signals"),
    (r"suggest_float\s*\(\s*['\"]bagging_fraction['\"].*?,\s*0\.[0-8]", "SACRED: bagging_fraction floor < 0.9 — kills rare signal training examples"),

    # Dense conversion hints
    (r"\.to_numpy\(\).*dense", "SPARSE: possible dense conversion — verify not on cross features"),

    # feature_pre_filter
    (r"feature_pre_filter.*True", "SACRED: feature_pre_filter must be False — True silently kills rare features"),
]

# Patterns to SKIP (false positive zones — comments, docstrings, test files)
SKIP_PATTERNS = [
    r"^\s*#",       # Comments
    r"^\s*\"\"\"",  # Docstrings
    r"^\s*\'\'\'",  # Docstrings
]

def check_forbidden_patterns(filepath: str) -> list:
    """Scan file for forbidden patterns."""
    violations = []
    try:
        source = Path(filepath).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    for i, line in enumerate(source.splitlines(), 1):
        # Skip comments
        if any(re.match(sp, line) for sp in SKIP_PATTERNS):
            continue
        for pattern, message in FORBIDDEN_PATTERNS:
            if re.search(pattern, line):
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


def check_ast(filepath: str) -> list:
    """Run AST-based structural checks on a Python file."""
    violations = []
    try:
        source = Path(filepath).read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except SyntaxError:
        return [{"file": filepath, "line": 0, "code": "", "message": "SYNTAX: File has syntax errors", "layer": "AST"}]
    except Exception:
        return []

    for CheckerClass in [OneAtATimeAssignChecker, XGBoostImportChecker]:
        checker = CheckerClass(filepath)
        checker.visit(tree)
        violations.extend(checker.violations)

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


if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.abspath(__file__))

    if len(sys.argv) < 2:
        print("Usage: python convention_gate.py [check|meta-audit|full] [file]")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "check":
        specific = sys.argv[2] if len(sys.argv) > 2 else None
        exit_code = run_check(project_dir, specific)
        sys.exit(exit_code)
    elif cmd == "meta-audit":
        run_meta_audit(project_dir)
    elif cmd == "full":
        exit_code = run_check(project_dir)
        run_meta_audit(project_dir)
        sys.exit(exit_code)
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
