# V2 SESSION RESUME — Read this file, then execute next unchecked item

## WHAT IS V2
Multi-asset ML trading system. 31 assets (14 crypto + 17 stocks). 15-20M sparse features per asset from "everything x everything" crosses (astro x TA x gematria x numerology x space weather x all). XGBoost + LightGBM on GPU. Target: 80% trade accuracy.

## WHERE WE ARE — STATUS: CERTIFIED FOR TRAINING

Full audit loop completed 2026-03-21 with 3 consecutive clean passes across ALL dimensions:
- Philosophy (10 rules) — 3x CLEAN
- Training readiness (17 checks) — 3x CLEAN
- Production readiness (9 checks) — 3x CLEAN
- Data format compatibility (2 checks) — 3x CLEAN
- NaN semantics sweep — 0 violations
- Cascading fix verification — CLEAN

**~35 fixes applied across 20+ files. Pushed to https://github.com/c19o/TB**

### COMPLETED — Full System Audit & Certification (2026-03-21)
Key fixes applied:
- [x] Platt calibration → uses OOS predictions (not training data)
- [x] eliminate_zeros() removed from all feature sparse matrices
- [x] KNN A/B test removed (XGBoost decides, no post-hoc removal)
- [x] All 13 cross types in live inference (added vx_ volatility crosses)
- [x] Esoteric/volatility/moon/aspect signal groups properly routed
- [x] V2 failures raise (LSTM, meta, features, crosses — no silent fallbacks)
- [x] platt_lstm_{tf}.pkl naming (no collision with XGBoost Platt)
- [x] blend_config saves both 'alpha' and 'best_alpha' keys
- [x] LSTM checkpoint includes means/stds for normalization
- [x] load_v2_data() returns 5 values, all callers updated
- [x] v2_cloud_runner: SAVAGE22_V1_DIR set, pyarrow added, V2_BATCH_SIZE wired
- [x] Artifact patterns complete (cross_names, feature_lists, platt_lstm, blend_config, manifest)
- [x] OOS prediction filename uses glob (not hardcoded)
- [x] live_trader: V2_DIR-first search for ALL artifacts
- [x] live_trader: load_dotenv + bitget_api dual key names
- [x] live_trader: SIGINT/SIGTERM handler with graceful shutdown
- [x] Batch column assignment in v2_feature_layers (110 one-at-a-time → 0)
- [x] Leakage check: 3-class, purge gap, sparse cross loading
- [x] Backtesting: OOS-only predictions, sparse cross loading, isinf sanitization
- [x] PROTECTED_FEATURE_PREFIXES with 58+ esoteric prefixes
- [x] range_position fillna(0.5) removed
- [x] asp_ removed from skip_pre (aspect crosses now fire)
- [x] portfolio_aggregator: np.full(NaN) not np.zeros
- [x] build_sports: targeted fillna on count cols only
- [x] Space weather methods ported to v2 LiveDataLoader
- [x] Cross column name dual-path loading in ml_multi_tf.py
- [x] v2_multi_asset_trainer: _hw init, pickle load, OOS dual keys, V2_BATCH_SIZE

### NEXT: Deploy to Cloud + Train
- [ ] Rent GPU on vast.ai
- [ ] Upload v2/ scripts + V1 databases (14 .db files) + kp_history.txt
- [ ] Run: `cd /workspace/v2 && python -u v2_cloud_runner.py --dph <cost> 2>&1 | tee pipeline.log`
- [ ] Phase 1: Feature builds (4h/1h/15m/5m) — needs restart, OOM fixed
- [ ] Phase 2: CPCV training all TFs
- [ ] Phase 3: Exhaustive optimizer
- [ ] Phase 4: PBO + meta-labeling
- [ ] Phase 5-6: LSTM on local 3090 (H200 weak CPU)
- [ ] Phase 7: Backtest + paper validation

### AFTER CLOUD: Paper Trading
- [ ] Verify all model files downloaded to local v2/ directory
- [ ] Start streamer supervisor: `python streamer_supervisor.py`
- [ ] `python live_trader.py --mode paper`
- [ ] Monitor trades.db for 48-72 hours
- [ ] Run self_learner.py after 10+ trades

## KEY RULES (NON-NEGOTIABLE)
- The matrix is UNIVERSAL — every asset gets full esoteric pipeline
- NO FILTERING of features. XGBoost decides via tree splits
- NO FALLBACKS. One pipeline. If it breaks, fix it
- Esoteric signals ARE the edge. Never regularize them away
- NEVER convert NaN to 0 — NaN is "missing", 0 is "the value is zero"
- Batch column assignment only. Never df[col]=val one at a time
- Sparse CSR for cross features. COO triplet accumulation
- 4-tier binarization on ALL numeric columns
- No eliminate_zeros() on feature sparse matrices
- Always progress logs. Never run blind
- Never deviate from plan. Follow checklist exactly

## GPU ARCHITECTURE
- ALL GPUs visible to every process (no CUDA_VISIBLE_DEVICES pinning)
- XGBoost: --parallel-splits distributes CPCV folds across GPUs
- NO --use-dask (OOM on sparse features)
- LSTM: nn.DataParallel splits batches across GPUs
- Batch sizes auto-scale via hardware_detect.py
- OOM retry: orchestrator halves V2_BATCH_SIZE and retries

## KEY FILES
```
v2/config.py                     — 31 assets, paths, PROTECTED_FEATURE_PREFIXES
v2/feature_library.py            — 16 GPU-native compute functions
v2/v2_feature_layers.py          — 20 V2 layers + 4-tier binarization
v2/v2_cross_generator.py         — 13 cross types, streaming sparse
v2/build_features_v2.py          — Feature builder with checkpointing
v2/ml_multi_tf.py                — CPCV training, triple-barrier, sample uniqueness
v2/v2_multi_asset_trainer.py     — Multi-asset XGBoost/LightGBM
v2/exhaustive_optimizer.py       — GPU grid search (30M combos)
v2/backtest_validation.py        — PBO + Deflated Sharpe
v2/meta_labeling.py              — Trade gate from OOS predictions
v2/v2_lstm_trainer.py            — LSTM DataParallel + Platt + alpha
v2/v2_cloud_runner.py            — 7-phase cloud orchestrator
v2/backtesting_audit.py          — Full-history audit report
v2/live_trader.py                — Production trading engine
v2/data_access.py                — LiveDataLoader (all sources)
v2/atomic_io.py                  — Atomic save helpers
v2/hardware_detect.py            — GPU/RAM detection
v2/CLAUDE.md                     — Rules + lessons learned
```
