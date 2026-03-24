# Session Resume — v3.1 (2026-03-23)

## STATUS: V3.1 PREPPED — NEEDS AUDIT -> SMOKE TEST -> TRAIN -> PAPER TRADE

All code forked from v3.0, 5m dropped, Tier 1 institutional risk added, modular pipeline built.
Disk space tight: 72 GB free, user will delete games to free more.

## WHAT TO DO NEXT

### Phase 1: Full Audit (3 clean passes)
Follow the CLAUDE.md 7-pass audit pipeline:
- Pass 0: Philosophy gate (no filtering, no fallbacks, NaN preserved, esoteric protected)
- Pass 1: Dead code (parallel with Pass 2)
- Pass 2: Config & hardcoded values (parallel with Pass 1)
- Pass 3: Cross-file consistency (after Pass 0)
- Pass 4: GPU & performance (after Pass 1+2)
- Pass 5: Live trading safety (after Pass 3)
- Pass 6: Cascade regression + integration smoke
Run 3 FULL LOOPS until zero issues found.

### Phase 2: Smoke Test
- python smoke_test_pipeline.py -- must pass all checks
- Verify: config loads, features resolve from v3.0, model can initialize, live_trader starts in paper

### Phase 3: Full Training Pipeline
- python pipeline_orchestrator.py -- runs all 7 phases with checkpoint/resume
- Or: python pipeline_orchestrator.py --status to check progress
- Each step independently resumable after crash/pause

### Phase 4: Paper Trade
- Start live_trader.py --mode paper
- Monitor for 48-72 hours (200+ trades or 30 days per promotion pipeline)

## DISK SPACE
- 72 GB free (97% used on 1.9 TB drive)
- User will delete games to free ~50-100 GB
- v3.1 reads parquets/npz from v3.0 (no duplication of 30 GB data)
- Deletable now: v3.0 .tmp.npz (6.1 GB), cross_names JSON (0.9 GB), root zip (6.4 GB)
- 15m NPZ will need ~84 GB if building from scratch

## V3.1 ARCHITECTURE

### What changed from v3.0
- Dropped 5m -- esoteric signals meaningless at 5m, saves 4-6 hrs build + 84 GB disk
- Tier 1 Risk Framework -- hard limits, drawdown protocol, circuit breakers, model versioning
- Modular Pipeline -- pipeline_orchestrator.py with per-step checkpoint/resume
- Shared Data -- v3.1 reads v3.0 parquets/npz via V30_DATA_DIR, no duplication
- Local Training -- i9-13900K + RTX 3090 + 64GB RAM. No cloud needed. No accuracy loss.

### Key Files
- pipeline_orchestrator.py -- crash-safe 7-phase orchestrator with manifest
- pipeline_manifest.json -- checkpoint state (auto-created on first run)
- config.py -- all risk limits, TF configs, shared data paths
- ml_multi_tf.py -- LightGBM CPCV training (resolves data from v3.0)
- live_trader.py -- Tier 1 risk enforcement + model versioning
- run_full_pipeline.sh -- local-only shell script (alternative to orchestrator)
- TIER_2_3_ROADMAP.md -- future upgrades for $10K-$100K+

### Pipeline Phases (checkpoint/resume each)
| Phase | What | Output |
|-------|------|--------|
| 1. features | Build parquets + cross NPZ | features_BTC_{tf}.parquet + v2_crosses_BTC_{tf}.npz |
| 2. train | LightGBM CPCV per TF | model_{tf}.json + features_{tf}_all.json |
| 3. optuna | Optuna 200 TPE trials | optuna_configs_{tf}.json |
| 4. meta | Meta-labeling from OOS | meta_model_{tf}.pkl |
| 5. lstm | LSTM + Platt calibration | lstm_{tf}.pt + platt_{tf}.pkl |
| 6. pbo | PBO + Deflated Sharpe | pbo_results_{tf}.json |
| 7. audit | Backtesting report | audit_{tf}.json |

### Reusable from v3.0 (saves hours)
- All 1w/1d/4h/1h parquets (37 files, 0.58 GB)
- All 1w/1d/4h/1h cross NPZ (35 files, 31.6 GB)
- multi_asset_prices.db (1.3 GB)
- NO 15m data exists in v3.0 (must build from scratch)

## ACCURACY: LOCAL = CLOUD
Perplexity-verified: max_bin=15 irrelevant for binary features (95% of matrix), num_leaves=63 optimal for sparse, CPU=GPU quality, local=distributed quality.
