# Draft: 1W Cloud Training Plan + MCP Diagnostics

## Requirements (confirmed)
- User requested reading and synthesizing:
  - `v3.3/SESSION_RESUME.md`
  - `v3.3/CODEX.md`
  - `v3.3/CONVENTIONS.md`
  - `v3.3/CLOUD_1W_LAUNCH_CONTRACT.md`
- User requested: "plan to train 1w model"
- User requested: test Perplexity MCP server
- User requested: test Olson/ops KB
- User requested: diagnose Socraticode

## Technical Decisions
- Planning mode only (no implementation / no training execution).
- Use documented 1W launch contract as source-of-truth run path.
- Include MCP/KB diagnostics as preflight work in plan scope.

## Research Findings
- Session resume indicates code-side optimization pass is largely complete; immediate priority is rerun 1W and measure evidence.
- Contract machine: Washington `2x RTX 4090`, `516 GB RAM`, type `33923286`.
- Canonical 1W sequence uses explicit stepwise commands (features → crosses → baseline → optuna search-only → retrain → optimizer).
- `OPTUNA_SKIP_FINAL_RETRAIN=1` and machine-aware final retrain policy are already part of current posture.
- `ops_kb.py smart` syntax uses `-n` (not `--limit`).
- `ops_kb.py smart "1w cloud launch contract weekly training" -n 5` returned results successfully.
- Orgonite KB query succeeded (`python kb.py smart ... --limit 5`), returning semantic matches.
- Perplexity MCP server process smoke test: `index.mjs` exists and starts (`RUNNING` observed, then stopped intentionally).
- Socraticode CLI is installed (`npx` exists; `npx -y socraticode --version` exit code 0), but MCP invocation via `skill_mcp` cannot see `socraticode` server in this session (only `playwright` MCP exposed).

## Scope Boundaries
- INCLUDE: 1W training work plan and preflight diagnostics plan.
- EXCLUDE: actual training execution, code edits, cloud runtime operations.

## Open Questions
- Should the plan stop at preflight + launch commands, or include full post-run audit checklist and go/no-go gates for `1d`?
- Preferred evidence format for timings/artifact audit: markdown checklist only vs checklist + structured CSV/JSON capture targets?
- For Socraticode diagnosis, should remediation include environment/integration fixes in scope, or diagnosis-only handoff?
