# Local GSD Company

## Purpose
Run Savage22 locally with the same practical company role structure as the prior external company, but with direct visibility, git worktrees, and tighter control.

Implementation and debugging happen locally through Codex + GSD + visible subagent work in dedicated worktrees.

## Operating Model

- `CEO / Monitor`: this Codex session
- `Execution framework`: local GSD install under `.codex/`
- `Specialists`: local Codex subagents spawned from this session
- `Workspace model`: isolated git worktrees under `.worktrees/`
- `Source of truth`: repo files, local diffs, validation runs, and explicit status updates in this session

## Mirrored Company Roles

- `Chief`: planning, delegation, integration
- `Daemon`: gpu daemon, supervisor, runtime reload path
- `GPU`: CUDA memory, CSC lifecycle, OOM, sparse path
- `ML`: training, Optuna, runtime guards, 1w retraining gate
- `QA`: validate, smoke, regression, convention and research audits
- `Doc`: session resume, operational logging, status docs
- `DevOps`: cloud/runtime/deploy checks
- `Matrix`: feature and thesis compliance

These roles should be represented as local workstreams and subagent scopes, not opaque external heartbeats.
Current worktrees map to the main lanes:
- `daemon`
- `deploy`
- `governance`
- `audit`

## Visibility Rules

- No hidden company state should be required to understand progress.
- Every active lane should have:
  - clear owner
  - explicit scope
  - current blocker or next action
- I will report what subagents are doing in plain language and cite file references for conclusions.
- I can inspect command output, diffs, and artifacts directly here.
- I cannot guarantee access to hidden private reasoning from any model runtime; what I can guarantee is visible execution and visible results.

## Research Rules

- Knowledge Base first for any non-trivial work
- Perplexity fallback only
- ML, CUDA, daemon, runtime, training, deployment, validation, and feature work are never "simple"
- SocratiCode first for repo navigation when the codebase is already indexed

## Governance Rules

- Any speed-affecting production change must be escalated to owner before use
- Any cloud rental or destruction must be escalated to owner before use
- Rare-signal retention is mandatory
- LightGBM only

## Local Workflow

1. Create or select a visible workstream.
2. Use GSD planning/execution patterns for non-trivial work.
3. Spawn local subagents only for clearly separated scopes.
4. Keep findings and fixes integrated in this session.
5. Verify with validation, smoke tests, or direct code evidence before calling work done.

## Immediate Focus

- Daemon RELOAD lane
- 1w production readiness lane
- Governance / KB-first enforcement lane
- Re-audit of weak or previously opaque completed work
- Cloud 1w launch lane for the approved Washington 2x RTX 4090 machine
