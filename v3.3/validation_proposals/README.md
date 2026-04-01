# Validation Proposals

Any agent can propose a new validate.py check. Only QA Lead + User can implement it.

## How to Propose

Create a YAML file in this directory: `YYYY-MM-DD-short-name.yaml`

```yaml
id: VAL-NNNN          # Next sequential ID (check existing proposals)
title: Short name
predicate: "exact condition to check"
severity: critical|high|medium|low
scope:
  - training_config    # or: feature_library, cross_gen, deployment, etc.
rationale: |
  Why this check matters. Include incident/evidence.
evidence:
  - "SAV-33: bagging_fraction=0.7 allowed by Optuna, should be >=0.95"
examples:
  passing:
    - "bagging_fraction: 0.95"
  failing:
    - "bagging_fraction: 0.70"
message: "Human-readable failure message for validate.py output"
rollout: hard_gate|soft_gate|warn|observe
proposed_by: agent-name
date: YYYY-MM-DD
status: proposed|approved|implemented|rejected
```

## Process

1. **Agent** creates proposal YAML + commits to branch
2. **QA Lead** reviews: is the invariant real? False positive risk?
3. If approved: QA Lead adds check to validate.py, updates .validate_hash
4. **Stop hook** enforces new check on all future agent runs
5. Proposal status updated to `implemented`

## Rules

- NEVER edit validate.py directly to add a check — propose here first
- Every check needs rationale + evidence (not vibes)
- QA Lead can reject with reason
- Hard gates require at least one real incident as evidence
