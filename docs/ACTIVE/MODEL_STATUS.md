# Model Training Status — Savage22 V3.3

Last updated: 2026-03-29

> **HOW TO READ**: WF Accuracy = CPCV walk-forward (OOS). Final Accuracy = trained on full dataset. Confidence threshold = minimum probability to open a trade. SUSPECT = data leakage detected, results unreliable until retrain.

---

## Per-TF Status

| TF | Status | WF Accuracy | Final Accuracy | Conf Threshold (LONG) | Conf Threshold (SHORT) | Last Trained | Notes |
|----|--------|-------------|----------------|-----------------------|------------------------|-------------|-------|
| **1w** | DONE v5 — **SUSPECT** | 57.9% CPCV | 75.4%@0.70 | 0.70 | N/A | 2026-03-29 | purge=6 < max_hold=50 → leakage. Retrain with purge=50. |
| **1d** | CROSS GEN DONE | — | — | — | — | — | 4.69M cross names + inference artifacts downloaded. Needs CPCV training on cloud (512GB RAM). |
| **4h** | BASE BUILT | — | — | — | — | — | features_BTC_4h.parquet ready (8,794×3,904). Cross gen + training on cloud. |
| **1h** | NOT STARTED | — | — | — | — | — | Needs feature build + cross gen on cloud. Requires 2TB+ RAM. |
| **15m** | NOT STARTED | — | — | — | — | — | Needs cloud only (100GB+ RAM cross gen). **User picks machine personally.** |

---

## 1w Detail (SUSPECT RUN — do not trade with this model)

```
Run date:       2026-03-29
Rows:           818
Features:       2,156,945 (2.16M = 3,298 base + 2,153,647 crosses)
CPCV:           4 paths, N=4 groups, K=1 test, purge=6, embargo=1%
WF Accuracy:    57.9% (SUSPECT — purge=6, max_hold=50 → 44 bars leakage)
Final Accuracy: 75.4% @ 0.70 confidence threshold
PrecL (CPCV):   0.453 (LONGs)
PrecS (CPCV):   0.000 (SHORTs never fired — class imbalance issue)
Top feature:    range_position (gain=146.8)
Esoteric in top 50: 6 features (synodic_jupiter_saturn, full_moon_decay, chinese_year_element)
```

**Known Issue (Bug 1.1):** CPCV purge=6 bars but max_hold_bars=50 for 1w. Bars 7-50 of each fold boundary leak training labels into test period. Perplexity-confirmed: 57.9% must be treated as suspect. Fix: `purge = TRIPLE_BARRIER_CONFIG[tf]['max_hold_bars']` for all TFs.

**Known Issue (Bug 1.2):** Class weight array misalignment — SHORT samples may be getting weight 1.0 instead of 3.0. Explains PrecS=0.000.

---

## 1d Detail

```
Cross gen:      DONE (ran on cloud, OOM'd locally)
Artifacts:      inference_1d_base_cols.json, inference_1d_cross_names.json,
                inference_1d_ctx_names.json, inference_1d_thresholds.json
Cross features: 4.69M cross names
Base parquet:   features_BTC_1d.parquet (10.7MB, 5,733×3,796)
Next step:      Rent 512GB+ RAM cloud machine, run cloud_run_tf.py --tf 1d
```

---

## 4h Detail

```
Base parquet:   features_BTC_4h.parquet (13.5MB, 8,794×3,904)
Cross gen:      NOT STARTED (requires 512GB+ RAM cloud)
Next step:      Rent cloud machine, run cross gen + training
```

---

## Retrain Queue

| Priority | TF | Reason | Blocker |
|----------|----|--------|---------|
| 1 | **1w** | purge=6 leakage (Bug 1.1) + class weight misalignment (Bug 1.2) | Fix config.py purge, rebuild |
| 2 | **1d** | Never trained | Cloud machine (512GB RAM) |
| 3 | **4h** | Never trained | Cloud machine (512GB RAM) |
| 4 | **1h** | Never trained | Cloud machine (2TB+ RAM) |
| 5 | **15m** | Never trained | User picks machine |

---

## GPU Fork Status

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1 (standalone benchmark) | COMPLETE | 99x standalone SpMV, 78x integrated (RTX 3090) |
| Phase 2 (LightGBM integration) | COMPLETE | Fork builds (42/42), `device_type="cuda_sparse"` accepted |
| Phase 3 (CSR bridge) | COMPLETE | Dangling pointer fixed, DLL rebuilt |
| Phase 4 (full GPU pipeline) | COMPLETE | GPU vs CPU accuracy: EXACT MATCH (77.64% both) |
| Production deployment | PENDING | Need Bug 1.1+1.2 fixed first |
