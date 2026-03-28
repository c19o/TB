# V3.3 Full System Audit — 4 Passes Complete (2026-03-26)

## STATUS: PASS 5 running. Need 3 consecutive clean passes before implementing fixes.

## CUMULATIVE FINDINGS (85+ issues across 4 passes, 36 agents)

---

### TIER 1: TRAINING-BLOCKING (must fix before ANY training run)

| # | Issue | File:Line | Impact |
|---|-------|-----------|--------|
| 1 | `feature_pre_filter=True` (LightGBM default) silently kills rare features before trees grow | config.py (not set → defaults True) | Features firing <min_data_in_leaf in fold are PRE-ELIMINATED. Kills esoteric signals on 1h/15m. |
| 2 | `OMP_NUM_THREADS=4` persists into training phase | All TRAINING_*.md launch commands | LightGBM capped at 4 threads on 192-core machine |
| 3 | `NUMBA_NUM_THREADS=4` throttles cross gen prange to 4 cores | All TRAINING_*.md launch commands | Cross gen uses 4 of 192 cores |
| 4 | `num_threads=-1` invalid for core LightGBM API | config.py:235 | Undocumented behavior, defers to OMP=4 |
| 5 | `is_enable_sparse=True` default — single-threaded when data stays sparse | config.py:236 | 15m/1h always sparse → single-threaded LightGBM |
| 6 | Final model never sets `num_threads` explicitly | ml_multi_tf.py:1263 | Gets -1 from config, caps at 4 OMP threads |
| 7 | Sequential CPCV path never sets `num_threads` | ml_multi_tf.py:1046 | 15m sequential path gets 4 threads on 512 cores |
| 8 | RAM-per-worker underestimates by 3x in cross gen | v2_cross_generator.py:652 | OOM on 1d/4h/1h cross gen |
| 9 | `_X_all_is_sparse=True` set unconditionally → `.tocsr()` crash | ml_multi_tf.py:630,929 | Crashes 1w/1d sequential training path |
| 10 | `run_optuna_local.py:230` unconditional `.toarray()` | run_optuna_local.py:230 | OOMs on 15m (8TB) and 1h (425GB) |

**FIX for #1:** Add `"feature_pre_filter": False` to V3_LGBM_PARAMS in config.py
**FIX for #2-3:** Remove OMP/NUMBA exports from launch commands. Set dynamically per phase in cloud_run_tf.py
**FIX for #4:** Change `"num_threads": -1` to `"num_threads": 0` in config.py
**FIX for #5:** Change `"is_enable_sparse": True` to `False` in config.py (workers convert fold slices to dense)
**FIX for #6-7:** Set `final_params['num_threads'] = os.cpu_count()` after copying V2_LGBM_PARAMS
**FIX for #8:** Change `* 4` to `* 12` at v2_cross_generator.py:652
**FIX for #9:** Change line 630 to `_X_all_is_sparse = not _converted_to_dense`
**FIX for #10:** Add RAM check before .toarray(), matching ml_multi_tf.py:617-621 pattern

---

### TIER 2: OPTUNA BROKEN (must fix before Optuna runs)

| # | Issue | File:Line |
|---|-------|-----------|
| 11 | Objective is mlogloss, NOT Sortino (violates spec) | run_optuna_local.py:439 |
| 12 | min_data_in_leaf range [1,30] ignores per-TF config | run_optuna_local.py:299 |
| 13 | min_gain_to_split range [0.5,5.0] too aggressive | run_optuna_local.py:305 |
| 14 | lambda_l1 up to 10.0 zeros out rare features | run_optuna_local.py:303 |
| 15 | lambda_l2 up to 50.0 over-regularizes | run_optuna_local.py:304 |
| 16 | No esoteric feature usage penalty (documented, never implemented) | run_optuna_local.py |
| 17 | Missing esoteric sample weights in load_tf_data | run_optuna_local.py:239 |
| 18 | max_bin tunable [15,31,63] (should be fixed at 15) | run_optuna_local.py:306 |
| 19 | max_conflict_rate/path_smooth/extra_trees = doc fiction (zero in code) | run_optuna_local.py |
| 20 | num_leaves up to 127 overfits 818 rows (1w) | run_optuna_local.py:298 |
| 21 | No max_depth in search space | run_optuna_local.py |
| 22 | Only 40 trials (spec says 200) | config.py:274-275 |
| 23 | V3_LGBM_PARAMS imported but never used | run_optuna_local.py |
| 24 | TF_MIN_DATA_IN_LEAF imported but never used | run_optuna_local.py |
| 25 | learning_rate excluded from search | run_optuna_local.py:309 |
| 26 | Stage 1 uses 30% subsample (noise for 1w: 245 rows) | config.py:286 |
| 27 | class_weight='balanced' documented but absent | run_optuna_local.py |
| 28 | n_jobs auto hardcodes 4 | run_optuna_local.py:854-857 |
| 29 | Sortino computation is binary hit/miss, not actual returns | run_optuna_local.py:410 |
| 30 | Stage 2 enqueues duplicate on resume | run_optuna_local.py:734 |
| 31 | HyperbandPruner incompatible with CPCV folds | run_optuna_local.py:624-633 |
| 32 | Stage 2 sampler missing group=True | run_optuna_local.py:725 |
| 33 | exhaustive_optimizer n_jobs=4 hardcoded | exhaustive_optimizer.py:780 |
| 34 | exhaustive_optimizer in-memory study (no persistence) | exhaustive_optimizer.py:769 |
| 35 | exhaustive_optimizer doesn't use CPCV OOS predictions | exhaustive_optimizer.py:267-302 |
| 36 | exhaustive_optimizer default 200 trials (spec says 500) | exhaustive_optimizer.py:919 |

---

### TIER 3: CPCV/TRAINING QUALITY

| # | Issue | File:Line |
|---|-------|-----------|
| 37 | Final model trains on last CPCV split (75% data), not full dataset | ml_multi_tf.py:1236 |
| 38 | Validation set missing weight= (3 locations) | ml_multi_tf.py:256,1065,1280 |
| 39 | class_weight='balanced' defined in config but NEVER used | config.py:261, ml_multi_tf.py |
| 40 | Worker count ignores RAM constraints | ml_multi_tf.py:815 |
| 41 | Final model missing interaction_constraints | ml_multi_tf.py:1263 |
| 42 | Parallel path skips per-fold HMM re-fitting | ml_multi_tf.py:807 |
| 43 | Pickle copies CSR per worker (no shared memory) | ml_multi_tf.py:830-835 |
| 44 | Dense→sparse→dense triple conversion | ml_multi_tf.py:621,823,219 |
| 45 | CPCV n_groups/n_test_groups hardcoded, not in config.py | ml_multi_tf.py:729-737 |
| 46 | 1w num_leaves=63 overfit risk (should be 31) | config.py:245 |
| 47 | ESOTERIC_KEYWORDS missing 15+ categories vs PROTECTED list | ml_multi_tf.py:656-664 |

---

### TIER 4: CLOUD DEPLOYMENT

| # | Issue | File:Line |
|---|-------|-----------|
| 48 | No V1 database verification (only checks btc_prices.db) | cloud_run_tf.py |
| 49 | Step 0 kill is NOT TF-aware — kills sibling TFs | cloud_run_tf.py:146 |
| 50 | No atexit/signal handler for lockfile cleanup | cloud_run_tf.py |
| 51 | 13 sys.exit(1) calls bypass lockfile cleanup | cloud_run_tf.py |
| 52 | cloud_run_tf.py has ZERO TF-specific branching | cloud_run_tf.py |
| 53 | ml_multi_tf_configs.json missing from artifact list | cloud_run_tf.py:104-117 |
| 54 | Model save is not atomic (no temp+rename) | ml_multi_tf.py:1314 |
| 55 | 15m env vars (RIGHT_CHUNK, BATCH_MAX) manual only | cloud_run_tf.py |
| 56 | No RAM validation per TF | cloud_run_tf.py |
| 57 | Cross gen caps at 128 threads | v2_cross_generator.py:654 |
| 58 | vast_launch.py emits NUMBA_DISABLE_CUDA=1 (contradicts CLAUDE.md) | vast_launch.py:294 |

---

### TIER 5: CROSS-GEN PERFORMANCE

| # | Issue | File:Line |
|---|-------|-----------|
| 59 | Numba thread pool global/shared — contention with ThreadPoolExecutor | v2_cross_generator.py:694+576 |
| 60 | cgroup v2 not detected (returns host RAM) | v2_cross_generator.py:84 |
| 61 | cgroup returns TOTAL not AVAILABLE | v2_cross_generator.py:84-87 |
| 62 | np.nonzero + np.unique single-threaded per block | v2_cross_generator.py:578-587 |
| 63 | Inconsistent RAM source between batch sizing and thread calc | v2_cross_generator.py:147 vs 651 |
| 64 | gpu_cross_builder.py thread cap = 32 | gpu_cross_builder.py:200 |
| 65 | ThreadPoolExecutor recreated per window | v2_cross_generator.py:694 |
| 66 | Dedup fancy indexing resets int64 indices to int32 | v2_cross_generator.py:1220 |

---

### TIER 6: LIVE TRADER / INFERENCE (non-functional until wired)

| # | Issue | File:Line |
|---|-------|-----------|
| 67 | InferenceCrossComputer not wired into live_trader.py | live_trader.py |
| 68 | No feature order validation against model.feature_name() | live_trader.py:930 |
| 69 | ~15% cross features always 0 (multi-signal combos) | inference_crosses.py:254 |
| 70 | bars_held uses wrong TF divisor — premature exits | live_trader.py:1287 |
| 71 | portfolio_aggregator loads models from root dir, not v3.3/ | portfolio_aggregator.py:27 |
| 72 | portfolio_aggregator can_open_position mutates Signal | portfolio_aggregator.py:183-187 |
| 73 | Fee calculation undercounts by ~50% (1x not 2x round-trip) | portfolio_aggregator.py:100, live_trader.py:1304 |
| 74 | bitget_api no retry logic for network errors | bitget_api.py:104-118 |
| 75 | bitget_api uses Binance.US as primary price source | bitget_api.py:321-335 |
| 76 | Cross-TF price contamination in exit checks | live_trader.py:1287-1291 |
| 77 | datetime.utcnow() tz-naive inconsistent with tz-aware | bitget_api.py:160,181 |

---

### TIER 7: MINOR / PERFORMANCE

| # | Issue | File:Line |
|---|-------|-----------|
| 78 | threadpoolctl installed but never used | cloud_run_tf.py:152 |
| 79 | Dead code: _subsample_start | ml_multi_tf.py:605,649 |
| 80 | backtesting_audit.py raw .nnz (safe today, fragile) | backtesting_audit.py:283 |
| 81 | v2_multi_asset_trainer.py is_enable_sparse=True (single-threaded) | v2_multi_asset_trainer.py |
| 82 | trade_journal post-exit fields always None (dead schema) | trade_journal.py:569-571 |
| 83 | trade_journal re-inits DB every connection | trade_journal.py:205 |
| 84 | No model hot-reload in live_trader | live_trader.py |
| 85 | Silent streamer gap handling | live_trader.py |
| 86 | lstm_sequence_model hardcodes num_workers=4 | lstm_sequence_model.py:359 |
| 87 | lstm_precursor_model uses num_workers=0 | lstm_precursor_model.py:346 |
| 88 | portfolio_aggregator references 5m (dropped) | portfolio_aggregator.py:8 |
| 89 | DONE marker written on partial failures | cloud_run_tf.py:560 |

---

### 15m BLOCKERS (NOT IMPLEMENTED — block 15m training)

| # | Issue |
|---|-------|
| 90 | Row-partitioned incremental boosting — zero init_model code |
| 91 | NNZ overflow detection — logs but never checks INT32_MAX |
| 92 | scipy int64 indices — no .astype(np.int64) in ml_multi_tf.py |

---

## DYNAMIC CONFIG MATRIX (from PASS 2)

| Setting | 1w | 1d | 4h | 1h | 15m |
|---------|----|----|----|----|-----|
| Rows | 818 | 5,727 | 4,380 | 17,520 | 227,577 |
| Expected crosses | ~2.2M | ~5-6M | ~3-4M | ~7-8M | ~10M+ |
| Dense matrix GB | 7.2 | 126 | 61 | 526 | 9,108 |
| Can fit dense? (512GB) | YES | YES | YES | NO | NO |
| min_data_in_leaf | 3 | 3 | 5 | 8 | 15 |
| num_leaves (SHOULD BE) | 31 | 63 | 63 | 63 | 127 |
| CPCV groups | 4(K=1) | 4(K=1) | 5(K=2) | 6(K=2) | 6(K=2) |
| NNZ overflow risk | None | None | None | LOW | BORDERLINE |
| Row partition needed | No | No | No | No | MAYBE |

## IMPLEMENTATION ORDER

1. **TIER 1 (training-blocking):** Fix all 10 issues → re-run 1w → verify full parallelism
2. **TIER 3 (training quality):** Fix CPCV/quality issues → better models
3. **TIER 2 (Optuna):** Fix all 26 Optuna issues → proper hyperparameter search
4. **TIER 4 (deploy):** Fix cloud deployment issues
5. **TIER 5 (cross-gen perf):** Fix performance bottlenecks
6. **TIER 6 (live trader):** Wire InferenceCrossComputer, fix all inference issues
7. **TIER 7 (minor):** Clean up minor issues

## MACHINE STATUS (paused)
| ID | TF | Status |
|----|-----|--------|
| 33598908 | 1w | PAUSED — training killed for audit |
| 33598909 | 1d | PAUSED |
| 33598156 | 4h | PAUSED |
| 33598910 | 1h | DESTROY — too small |
| 33598158 | 15m | PAUSED — has BLOCKERS |
