# GSD Rollout

## Status

GSD is installed locally in both:

- `C:/Users/C/Documents/Savage22 Server`
- `C:/Users/C/Documents/Savage22 Server/v3.3`

Installed for:

- Claude Code
- Codex

This covers the mixed historical agent `cwd` layout:

- root-scoped agents use the root install
- `v3.3`-scoped agents use the `v3.3` install

## What Was Added

For each install location:

- local Claude GSD commands under `.claude/commands/gsd`
- local Codex GSD skills under `.codex/skills/gsd-*`
- local GSD agent configs under `.codex/agents`
- local GSD workflow assets under `.codex/get-shit-done`

## Issues Fixed During Rollout

### Agent prompt drift

The company prompts had stale examples that were causing real failures in live runs:

- `ops_kb.py smart ... --limit` was wrong; the CLI expects `-n`
- several prompts referenced `cross_feature_generator_v4.py`, but the real file in this repo is `v2_cross_generator.py`

Those fixes were applied under `v3.3/agents/`.

### Orgonite KB query failure

The Orgonite master KB was failing on punctuation-heavy natural-language queries with SQLite FTS parse errors such as:

- `no such column: feature`
- `no such column: 4`
- `no such column: lived`

Root cause:

- raw natural-language strings were being passed directly into SQLite FTS5
- punctuation-heavy tokens like `3+`, `2.9M+`, and `long-lived` could break parsing

Fix:

- `C:/Users/C/Desktop/MY GOOGLE DRIVE/Orgonite master/database.py`
- search now falls back to a sanitized tokenized FTS query when the raw query trips the parser

### GSD install defect

The installer left one stale `.claude` path reference inside the generated Codex debugger docs:

- `.codex/agents/gsd-debugger.md`
- `v3.3/.codex/agents/gsd-debugger.md`

Those references were corrected to `.codex`.

## Live Agent Monitoring

Use the native Codex control session as the source of truth:

- active agent board in-session
- direct shell output
- git diff / artifact checks
- visible worktree ownership

## Recommended Resume Pattern

When local company execution is resumed:

1. resume only the roles needed for the next task wave
2. restart agents after local GSD install so their next sessions load the new local config
3. for non-trivial work, use GSD-style flow:
   - discuss/assumptions
   - plan
   - execute
   - verify
4. keep Codex + local docs as the control plane
5. use GSD as the execution workflow inside each worktree/lane

## Practical Guidance

- trivial tasks: use fast/quick execution paths
- complex bugfixes and production work: use plan/execute/verify flow
- daemon and training issues should keep KB-first research requirements from the existing company prompts
- Native local Codex control is the company OS; GSD is the repo-local execution framework
