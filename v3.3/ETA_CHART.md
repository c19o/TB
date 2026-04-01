# Savage22 V3.3 — Pipeline ETA Chart

> **IMPORTANT**: Update this file after every pipeline change, machine change, or step completion.
> Actual times must replace estimates once known. Documentation Lead owns this file.

## Current Machine: Sichuan 8x RTX 3090 (ID: 33876301) — $1.12/hr

_Last updated: 2026-03-31_
_Machine specs: 8x RTX 3090 (24GB each), ~64 cores_

---

## ETA Per Step Per Timeframe

| Step | Description | 1w | 1d | 4h | 1h | 15m |
|------|-------------|----|----|----|----|-----|
| **0** | validate.py (74 checks) | ~30s | ~30s | ~30s | ~30s | ~30s |
| **1** | Feature DB build (local) | ✅ DONE | ✅ DONE | ✅ DONE | ✅ DONE | ✅ DONE |
| **2** | Cross-gen V4 (GPU/CPU) | ✅ DONE | ~45m | ~2h | ~4h | ~8h |
| **3** | Label generation | ✅ DONE | ~5m | ~10m | ~20m | ~40m |
| **4** | LightGBM + Optuna (CPCV) | ✅ DONE | ~3h | ~6h | ~10h | ~20h |
| **5** | LSTM ensemble | ✅ DONE | ~1h | ~2h | ~3h | ~5h |
| **6** | Portfolio optimization | ✅ DONE | ~20m | ~30m | ~45m | ~1h |
| **7** | Evaluation + SHAP | ✅ DONE | ~20m | ~30m | ~45m | ~1h |
| — | **TOTAL** | ✅ DONE | **~5h 20m** | **~11h** | **~19h** | **~36h** |

> ETAs above are estimates based on 1w actuals. Update with real times as each TF completes.

---

## Actual vs ETA Log

| Date | TF | Step | ETA | Actual | Notes |
|------|----|------|-----|--------|-------|
| 2026-03-28 | 1w | Full | ~6h | ~6h | All steps PASS. AUC 57.5% |
| — | 1d | — | ~5h 20m | TBD | Blocked: RELOAD bug (SAV-4) |

---

## Resource Health Check (update on machine change)

| Resource | Advertised | Actual (on connect) | Status |
|----------|-----------|---------------------|--------|
| GPU count | 8x RTX 3090 | TBD | — |
| GPU VRAM total | 8 × 24GB = 192GB | TBD | — |
| System RAM | ~256GB | TBD | — |
| CPU cores | ~64 | TBD | — |
| Disk | ~500GB | TBD | — |

> Run `nvidia-smi` + `free -h` + `nproc` on connect. Fill in Actual column. If any resource is <85% of advertised, escalate to Discord before starting work.

---

## How to Update This File

After every training step completes:
```
| 2026-XX-XX | 1d | Step 2 Cross-gen | 45m | 52m | GPU OOM once, retry OK |
```

After any machine change: update the machine header + Resource Health Check table.

After any pipeline code change: re-estimate affected steps and update the ETA column.
