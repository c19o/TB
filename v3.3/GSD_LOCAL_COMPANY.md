# GSD Local Company Mode

## Purpose
Run Savage22 from the local Codex workspace with full visibility, using GSD for planning/execution, git worktrees for isolation, and the same role structure as the prior external company.

Active project work should be coordinated locally first.

## Visibility Rules
- All active work should be visible from this terminal.
- Use local Codex workstreams/subagents instead of opaque heartbeat loops where possible.
- Split complex lanes into separate git worktrees so unrelated fixes do not collide.
- Report progress in plain language with concrete files, commands, and blockers.
- No hidden "company state" should be required for the owner to understand what is happening.

## Local Company Roles
- Chief: orchestration, prioritization, reopening weak work, integrating findings
- Daemon: `gpu_daemon.py`, `cross_supervisor.py`, reload/runtime failures
- GPU: memory lifecycle, CSC/CSR safety, throughput bottlenecks
- ML: training pipeline, validation, Optuna/runtime guards, 1w retrain gates
- Matrix: feature thesis, KB alignment, protected-signal correctness
- QA: re-audits, regression checks, gate enforcement
- Docs: session resume, audit reports, owner-facing status
- DevOps: deployment/runtime contracts, cloud scripts, machine hygiene
- Worktree lanes: `daemon`, `deploy`, `governance`, `audit`

## Research Policy
- Knowledge Base first
- Perplexity fallback only
- Any non-trivial ML/CUDA/training/daemon/runtime/deployment work must leave KB-first evidence
- SocratiCode should be used for codebase navigation when the repo index is available

## GSD Workflow
For any non-trivial lane:
1. Discuss assumptions
2. Plan the phase
3. Execute in bounded workstreams
4. Verify outcome before closing
5. Record the outcome in the corresponding audit/status doc

## Owner Escalation Rules
Escalate to the owner only for:
- cloud machine rental/destruction
- any speed-affecting training/runtime change
- Matrix Thesis / protected-signal policy changes
- rollback decisions after failed runs

## Immediate Working Mode
- Use local Codex-led workstreams with visible updates
- Keep the same company role decomposition
- Prefer direct local inspection and code work over external orchestration layers
