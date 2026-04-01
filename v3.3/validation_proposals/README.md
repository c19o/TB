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

## Meta-Audit Workflow (Weekly)

convention_gate.py includes a meta-audit feature that analyzes recent commits to discover ungated convention violations:

```bash
cd v3.3
python convention_gate.py meta-audit
```

**Interpreting Results**:
- `>>> PROMOTE` = Rule fires in >50% of changed files → candidate for validation_proposals/
- `monitor` = Rule fires but below threshold → watch for future occurrences

**Promotion Flow**:
1. Run meta-audit weekly (or after major feature additions)
2. Review PROMOTE rules — are they real violations or false positives?
3. For real violations:
   - Create YYYY-MM-DD-short-name.yaml proposal in this directory
   - Include evidence from meta-audit output (file:line examples)
   - Set severity based on impact (critical = kills rare signals, high = accuracy loss, medium = tech debt)
4. QA Lead reviews and implements via validate.py
5. convention_gate.py updated with the new hard gate

**Example**:
```yaml
id: VAL-0097
title: No .apply(lambda) on large DataFrames
predicate: "No DataFrame.apply(lambda ...) calls in feature construction"
severity: high
scope:
  - feature_library
rationale: |
  .apply(lambda) is 10-100x slower than vectorized pandas/numpy ops.
  On 500K rows, this can turn a 30-second feature build into 10 minutes.
evidence:
  - "meta-audit found .apply(lambda) in 4/7 changed files"
  - "feature_library.py:234 — 45-second overhead on 1h TF"
examples:
  passing:
    - "df['result'] = np.where(condition, value_if_true, value_if_false)"
  failing:
    - "df['result'] = df.apply(lambda row: some_func(row), axis=1)"
message: "PERF: .apply(lambda) on DataFrame — vectorize for 10-100x speedup"
rollout: hard_gate
proposed_by: meta-audit
date: 2026-04-XX
status: proposed
```
