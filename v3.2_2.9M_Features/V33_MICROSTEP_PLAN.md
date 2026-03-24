# V3.3 Microstep Plan — Pick Up Here Next Session

## STATUS: v3.2 TRAINING COMPLETE. v3.3 PREP IN PROGRESS.

## WHAT'S DONE
- All 5 TFs trained with full cross features (1w/1d/4h/1h/15m)
- All results downloaded to v32_cloud_results/
- All 21 code fixes verified PASS
- Phase 1 esoteric analysis COMPLETE (all 5 TFs)
- inference_crosses.py BUILT (not integrated yet)
- v3.3 training plan written with full lessons
- 3 audit agents ran: streamers, training pipeline, code fixes
- Zero cloud machines running

## WHAT NEEDS TO BE DONE (in order)

### STEP 1: Fix Streamers (BLOCKS RETRAINING)

**1a. Tweet Color Detection**
- Files: `tweet_streamer.py`, `scrape_tw.py`
- Problem: has_gold/has_red/has_green never populated (100% zeros)
- Fix: implement color detection from media_urls or emoji/text patterns
- Consult Perplexity on best approach for tweet image color extraction

**1b. Tweet Gematria Storage**
- Files: `scrape_tw.py`
- Problem: gematria_simple, gematria_english never inserted into tweets.db
- Fix: compute gematria on full_text at insert time using universal_gematria

**1c. Sports Lookahead Removal**
- Files: `sports_streamer.py`, `feature_library.py` (lines 2340-2350)
- Problem: games mapped to ALL bars on that date including pre-game
- Fix: use game_timestamp (UTC), only map to bars AFTER game completed
- feature_library.py must filter: game_timestamp < bar_close_time

**1d. DB Path Standardization**
- Files: `config.py`, `data_access.py`, all streamers
- Problem: streamers write to v3.2/ but live_trader reads from parent
- Fix: single DB_DIR in config.py, all streamers and readers use it

**1e. Space Weather Missing Columns**
- Files: `space_weather_streamer.py`
- Problem: sunspot_number, solar_flux_f107 not populated
- Fix: add NOAA F10.7 + SILSO sunspot endpoints

**1f. Horse Racing**
- Files: `sports_streamer.py`
- Problem: hardcoded 10 dates/year, no live API
- Fix: find free API or remove horse features entirely

**1g. v2_easy_streamers Integration**
- Files: `v2_easy_streamers.py`, `feature_library.py`
- Problem: collects DeFi TVL, BTC.D, mining stats but nothing reads it
- Fix: wire into feature_library or remove from supervisor

### STEP 2: 15m Base Feature Expansion
- Files: `build_15m_features.py`, `feature_library.py`
- Problem: 15m has 1,284 base features vs 3,000+ for other TFs
- Fix: ensure same TA windows compute at all TFs

### STEP 3: Integrate Inference Crosses into Live Trader
- Files: `v2_cross_generator.py` (add save_inference_artifacts call), `live_trader.py` (load + compute per bar)
- inference_crosses.py already built — needs integration
- Add save step to training pipeline (cloud_run_tf.py)

### STEP 4: Add is_unbalance=True for 1d
- Files: `ml_multi_tf.py` or `config.py`
- Problem: 1d 60% FLAT → model collapses to FLAT at high confidence
- Fix: add is_unbalance=True to LightGBM params for 1d

### STEP 5: 3 Clean Audit Passes
- Agent team audit pass 1: streamer data flow (all fixes working)
- Agent team audit pass 2: training pipeline (all code correct)
- Agent team audit pass 3: full integration (no import errors, no skew)
- ALL THREE must come back clean before cloud training

### STEP 6: Rebuild Feature Parquets
- Run all build scripts locally with fixed streamers
- Verify feature counts match or exceed v3.2
- New features from fixed tweets/sports should increase column counts

### STEP 7: Cloud Training (user picks machines)
- Fastest path on vast.ai
- User chooses machines at training time
- 15m needs 2TB Belgium-class machine
- All other TFs: 384c Norway-class ($0.80-0.94/hr)
- Pipeline: cross gen → train → Optuna → meta → LSTM → PBO
- Download + verify + agent review before killing

### STEP 8: Post-Training (AFTER models trained)
- Phase B: Temporal cascade (4h→1h→15m signal flow)
- Phase C: Scale-in/out (fractional Kelly per bar)
- Phase E: Dynamic exits (lower TF reversal)
- Phase F: SHAP cross feature validation
- Phase 2: Vector DB mining for new esoteric features

## KEY RESULTS FROM V3.2 (reference)

| TF | Directional Acc (>80% conf) | PBO | Top Esoteric |
|---|---|---|---|
| 1w | 90.2% overall | REJECT | arabic_lot_treachery #36 |
| 1d | 62.6% (FLAT-biased) | DEPLOY 0.00 | bars_since_full_moon #6 |
| 4h | 94% dir at >70% | REJECT | planetary_day_dr_combo #3 |
| 1h | 98% dir at >80% | DEPLOY 0.14 | mayan_sign_idx #12 |
| 15m | 49.9% (broken data) | — | cross_new_moon_x_bull #6 |

## 16 UNIVERSAL ESOTERIC FEATURES (all 5 TFs)
bars_since_full_moon, pi_cycle_ratio, golden_ratio_ext, golden_ratio_dist,
schumann_783d_sin, equinox_proximity, gann_sq9_level, arabic_lot_commerce,
schumann_143d_sin, full_moon_decay_slow, schumann_133d_sin, schumann_133d_cos,
high_fear_decay_fast, moon_approach_return_5d, high_fear_decay_slow, bars_since_gem_match

## EXPANSION TARGETS (missing from all models)
HIGH: planetary retrogrades, solstice, fibonacci time sequences
MEDIUM: Hebrew calendar, master numbers, Tesla 369 vortex math
LOWER: Shmita, number patterns (113, 322, 666, 777), palindromes

## TRAINING INFRASTRUCTURE REMINDERS
- LightGBM sparse = single-threaded. MUST convert to dense.
- 15m dense = 1.89TB. Need 2TB machine. Auto-subsample if tight.
- save_binary() after first Dataset = skip 3hr loading on reruns
- max_bin=63 not 255 for 4x faster histograms
- Optuna: 20 S1 trials, 150 rounds, 2-fold search, reference=dtrain
- NPZ skip logic in cloud_run_tf.py
- ALWAYS download NPZ before killing machines
- ALWAYS verify with agent before killing
- REUSE machines instead of killing + renting new ones
- Consult Perplexity MCP before any changes, keeping matrix intact
