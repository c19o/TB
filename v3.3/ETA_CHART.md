# Savage22 V3.3 - Pipeline ETA Chart

> IMPORTANT: Update this file after every pipeline step completion, machine change, or pipeline code change.
> Fill actual timings as soon as known; re-estimate remaining pipeline ETAs immediately.

## Current Machine: Sichuan 8x RTX 3090 (ID: 33876301) - $1.12/hr (PAUSED)

Last updated: 2026-04-01
Machine specs: 8x RTX 3090 (24GB each), CPU score reference ~240-453 on candidate pool

---

## ETA Per Step Per Timeframe

| Step | Description | 1w | 1d | 4h | 1h | 15m |
|------|-------------|----|----|----|----|-----|
| 0 | validate.py (96 checks) | DONE | 30s | 30s | 30s | 30s |
| 1 | Feature DB build (local) | DONE | DONE | DONE | DONE | DONE |
| 2 | Cross-gen V4 (GPU/CPU) | DONE | BLOCKED (step 3+ reload path) | 2h | 4h | 8h (CPU likely) |
| 3 | Label generation | DONE | 5m | 10m | 20m | 40m |
| 4 | LightGBM + Optuna (CPCV) | DONE | 3h | 6h | 10h | 20h |
| 5 | LSTM ensemble | DONE | 1h | 2h | 3h | 5h |
| 6 | Portfolio optimization | DONE | 20m | 30m | 45m | 1h |
| 7 | Evaluation + SHAP | DONE | 20m | 30m | 45m | 1h |
| - | TOTAL | DONE (~6h actual) | ~5h 20m once unblocked | ~11h | ~19h | ~36h |

Notes:
- 1d ETA assumes cross-supervisor defects are fixed and daemon path remains stable.
- 15m remains highest memory risk; prior attempts indicate CPU fallback may be required for cross-gen.

---

## Actual vs ETA Log

| Date | TF | Step | ETA | Actual | Notes |
|------|----|------|-----|--------|-------|
| 2026-03-28 | 1w | Full | ~6h | ~6h | All steps PASS. CPCV AUC 57.5% |
| 2026-04-01 | all | validate + convention gate | ~30s | pass | validate.py 96/96 PASS (2 warnings), convention gate ALL PASS |
| 2026-04-01 | 1d | Step 2 cross-gen V4 | ~45m | partial | First two cross steps run; step 3+ blocked by remaining cross-supervisor bugs |
| 2026-04-01 | 1w | smoke test pipeline | ~2-5m | fail | Feature build failed on cuDF import; requires cuDF install or `ALLOW_CPU=1` |
| 2026-04-01 | all | pipeline code change (SAV-8 parity) | n/a | complete | Added `px_pc213_x_rsi_os` and `px_pc213_x_macd_high` for pc213 parity |
| 2026-04-01 | 1w | smoke test pipeline (retry) | ~2-5m | pass (10/10) | Post-parity-fix rerun succeeded |
| 2026-04-01 | 1w | smoke test pipeline (DoD rerun) | ~2-5m | fail | cuDF unavailable on CUDA13 path; run requires cuDF or `ALLOW_CPU=1` |

---

## Resource Health Check (update on machine change)

| Resource | Advertised | Actual (on connect) | Status |
|----------|-----------|---------------------|--------|
| GPU count | 8x RTX 3090 | TBD (machine paused) | pending reconnect |
| GPU VRAM total | 192GB | TBD | pending reconnect |
| System RAM | ~256GB | TBD | pending reconnect |
| CPU score | provider-specific | 240-453 candidates seen | informational |
| Disk | ~500GB | TBD | pending reconnect |

Run `nvidia-smi`, `free -h`, and `nproc` immediately after machine connect and update this table.
If any resource is under 85% of advertised, escalate before starting training.

---

## Update Protocol

After every training step completion, append one row to Actual vs ETA Log.
After any machine rent/pause/destroy event, update machine header + resource table + notes.
After any pipeline code change affecting runtime, update impacted ETA estimates.
