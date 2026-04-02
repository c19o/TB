# Savage22 V3.3 - Pipeline ETA Chart

Last updated: 2026-04-01

## Current Launch Target

- Machine: approved Washington `2x RTX 4090`
- Type ID: `33923286`
- Status: not rented yet
- Purpose: fresh `1w` cloud run

Historical Sichuan notes are no longer the active launch plan.

---

## Current Readiness Snapshot

| Item | Status | Notes |
|------|--------|-------|
| `validate.py` | PASS | `97/97` |
| convention gate | PASS | no live violations |
| `1w` smoke path | PASS | local fallback path healthy |
| daemon for `1w` | READY ENOUGH | lower-TF proof still pending |
| Washington launch script | READY | explicit and machine-constrained |
| lower TF ladder | WAITING | depends on fresh `1d` proof |

---

## ETA Per Timeframe

| Step | Description | 1w | 1d | 4h | 1h | 15m |
|------|-------------|----|----|----|----|-----|
| 0 | Preflight gates | DONE | DONE | DONE | DONE | DONE |
| 1 | Feature DB build | launch run | done after 1w | later | later | later |
| 2 | Cross-gen | launch run | proof rerun needed | later | later | later |
| 3 | Label generation | launch run | later | later | later | later |
| 4 | LightGBM + Optuna | launch run | later | later | later | later |
| 5 | LSTM ensemble | launch run | later | later | later | later |
| 6 | Portfolio optimization | launch run | later | later | later | later |
| 7 | Evaluation + SHAP | launch run | later | later | later | later |

---

## Operator Notes

1. `1w` is the only immediate cloud objective.
2. `1d` and below are not the current launch decision.
3. Artifact download at every checkpoint is mandatory.
4. Any speed-affecting runtime change still requires owner signoff.
