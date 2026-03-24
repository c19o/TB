# V3.3 Training Plan — Complete Lessons from v3.2 (2026-03-24)
# THIS IS THE AUTHORITATIVE TRAINING REFERENCE. READ BEFORE ANY CLOUD DEPLOY.

## CRITICAL LESSONS (v3.2 Session)

### 1. LightGBM Sparse = SINGLE-THREADED
**The #1 surprise.** LightGBM with `force_col_wise=True` and scipy sparse CSR spawns OpenMP threads but ONLY the main thread does work. Load average = 1.0 on a 256-core machine. 449 threads spawned, 448 idle.

**Fix:** Convert sparse→dense before creating LightGBM Dataset:
```python
X_all = sp_sparse.hstack([X_base_sparse, cross_matrix], format='csr')
X_all = X_all.toarray()  # REQUIRED for multi-core LightGBM
```

**RAM for dense:**
| TF | Dense Size | Min RAM |
|---|---|---|
| 1w | 2.2 GB | 32 GB |
| 1d | 66 GB | 128 GB |
| 4h | 56 GB | 128 GB |
| 1h | 300 GB | 512 GB |
| 15m | 455 GB | 768 GB |

### 2. Cross Generator is Always Single-Threaded
v2_cross_generator.py builds crosses on 1 CPU core regardless of machine size. Cross gen for 1d takes ~5 min, 1h takes ~15 min, 15m takes ~45 min. Machine core count doesn't help.

### 3. Ten Bugs That Killed Every Previous Attempt
All 10 previous cloud runs trained with base-only (3,314 features) because crosses never loaded:

1. `--asset BTC` → must be `--symbol BTC` in v2_cross_generator.py
2. `df._v2_symbol` not set from CLI args → NPZ saved without BTC prefix
3. multi_asset_prices.db has symbol='BTC', code queries 'BTC/USDT' → SQL UPDATE fix
4. meta_labeling.py had test-only __main__ → added --tf CLI
5. backtest_validation.py had test-only __main__ → added --tf CLI
6. V30_DATA_DIR resolved to /v3.0 (LGBM) on cloud → env var fix
7. LSTM looks for features_{tf}.parquet (no BTC prefix) → symlink fix
8. killall python kills the launcher script itself → targeted pgrep fix
9. tee masks exit codes → bash pipefail fix
10. 15m parquet has 1,284 base features (correct for intraday) → lower threshold

### 4. 15m Has Fewer Base Features (1,284 vs ~3,000)
Not a bug. The feature library generates fewer TA indicators for 15m resolution (176 vs 400+ for daily). Crosses still expand to 500K+. Combined with 227K rows, this is the most data-rich TF.

### 5. Upload Strategy — Use Root btc_prices.db
- Root btc_prices.db (539MB) has BTC/USDT → queries work natively
- multi_asset_prices.db (1.3GB) has BTC → needs SQL UPDATE to add '/USDT'
- For 15m feature rebuild: need multi_asset_prices.db (more data) + V1 DBs + kp_history.txt
- For all other TFs: root btc_prices.db is sufficient

## Machine Selection (vast.ai)

**Key: need HIGH RAM for dense conversion, not just high core count.**

| TF | Min Cores | Min RAM | Ideal |
|---|---|---|---|
| 1w | 64 | 32 GB | Any cheap 128c |
| 1d | 128 | 128 GB | 128c, 252GB+ |
| 4h | 128 | 128 GB | 128c, 252GB+ |
| 1h | 128 | 512 GB | 256c, 755GB+ |
| 15m | 192 | 768 GB | 384c, 1TB+ |

## Feature Counts (Verified 2026-03-24)

| TF | Base | Crosses | Total | Rows |
|---|---|---|---|---|
| 1w | 2,656 | 654,202 | 656,825 | 818 |
| 1d | 3,077 | 2,881,220 | 2,884,297 | 5,727 |
| 4h | 3,289 | 3,198,974 | 3,202,263 | 4,380 |
| 1h | 3,350 | 4,264,060 | 4,267,410 | 17,520 |
| 15m | 1,284 | ~500K-1M | ~500K-1M | 227,577 |

## Deploy Script: cloud_run_tf.py

Fixed version handles:
- Symbol format detection and fix (BTC → BTC/USDT)
- Feature rebuild if parquet < 1000 cols
- Symlinks for LSTM compatibility
- SPARSE verification (abort if DENSE)
- pipefail for proper exit code propagation
- Targeted process kill (not self)
- DONE marker file for monitoring
- V30_DATA_DIR and SAVAGE22_DB_DIR env vars

## Pipeline Steps (per TF, ~1-6 hours depending on TF)

1. Cross build (single-threaded, 5-45 min)
2. Train CPCV (multi-core with dense fix, 10-120 min)
3. Optuna hyperparameter search (multi-core, 30-180 min)
4. Exhaustive trade optimizer
5. Meta-labeling
6. LSTM
7. PBO + Audit

## Cost (actual v3.2 run)

5 machines @ $3.32/hr total. $32 budget. 1w done in 3 min.

---

## PERPLEXITY-VALIDATED INSIGHTS (2026-03-24)

### PushDataToMultiValBin — THE Real Bottleneck
LightGBM issue #5205: data loading is O(rows × ALL_features), not O(NNZ). For 227K × 2.3M = 520B iterations, single-threaded. Takes 3+ hours. Bug unfixed as of 2026.

**Workarounds (ranked by speed):**
1. **Dense conversion** (bypasses entirely) — needs rows × features × 4 bytes RAM
2. **`save_binary()`** after first load — all reruns skip loading (seconds vs hours). CRITICAL for Optuna.
3. **`max_bin=63`** not 255 — 4x less histogram work
4. **`force_col_wise=True`** — signals column-oriented structures, may avoid row-wise path
5. **`feature_pre_filter=True`** (default) — drops zero-variance before construction

**DO NOT USE:** `two_round_loading=True` (slower), CSR→CSC conversion (makes it worse)

### Per-TF Dense RAM Requirements
| TF | Rows | Features | Dense (float32) | Min Machine RAM |
|---|---|---|---|---|
| 1w | 818 | 657K | 2.2 GB | 32 GB |
| 1d | 5,727 | 2.88M | 66 GB | 128 GB |
| 4h | 4,380 | 3.2M | 56 GB | 128 GB |
| 1h | 17,520 | 4.27M | 300 GB | 512 GB |
| 15m | 227,577 | 2.28M | 1.89 TB | **2 TB (subsample to 210K = 1.81 TB)** |

### Sparse int64 — Safe in LightGBM v4.6
- scipy auto-promotes to int64 when NNZ > 2^31
- LightGBM v4.6 Python wrapper handles int64 correctly (fixed in PR #1719)
- C++ PushDataToMultiValBin has separate issue (#5205) — speed, not corruption
- GPU prediction segfaults with int64 (issue #7101) — CPU only

### Optuna is Fully Decoupled
Per Perplexity: Optuna is stateless. Needs only parquet + NPZ + btc_prices.db.
Can run on any machine, any time, even weeks later.
Use `save_binary()` to avoid re-loading for each of 80 trials.

### 1d Class Imbalance Fix — CRITICAL FOR V3.3
60% FLAT on daily → model predicts FLAT 90% of the time. Above 65% confidence it's 100% FLAT.
LONG edge exists (75% directional accuracy at 45-50% conf) but model won't commit at high confidence.
SHORT is dead (55 predictions, 21.8% precision).

**Root cause:** Class imbalance. 60.5% FLAT true labels → model learns FLAT = safe = high confidence.

**v3.3 fixes (per Perplexity, matrix-safe — ALL features stay):**
1. `is_unbalance=True` in LightGBM params — auto-reweights by inverse class frequency
2. OR `scale_pos_weight` for fine-grained control: weight LONG/SHORT = n_flat / n_directional (~1.5x)
3. Lower confidence threshold for directional classes from 0.5 to 0.3-0.35
4. Consider focal loss to down-weight easy FLAT predictions
5. Tighten FLAT definition (narrow the ±X% band) → fewer bars qualify as FLAT → natural rebalance

**DO NOT use SMOTE/oversampling** — creates synthetic leakage with 2.9M engineered features.

**Also noted:** Directional edge decays over time (Fold 0: 74% LONG precision → Fold 3: 30%). Investigate regime dependency. Walk-forward retraining (Phase 6) should help.

**v3.2 daily model is still valuable:**
- PBO=0.00 (zero overfit) confirms the FLAT signal is real and stable
- LONG at 45-50% conf = 75% directional accuracy on 300-400 trades
- 30% of top 50 features are esoteric (highest of any TF) — moon, Arabic lots, planetary cycles
- Use as regime filter: when daily says FLAT with >80% conf, don't trade any TF

### Feature Chunking Loses Cross-Interactions
Per Perplexity: splitting features into chunks and ensembling LOSES cross-chunk interactions.
"RSI only matters when Mercury is retrograde" requires BOTH features in ONE model.
**NEVER chunk features. Always keep all features in one model.** Row subsample if needed.

### v3.2 Training Results
| TF | Directional Acc | LSTM | PBO | Production? |
|---|---|---|---|---|
| 1w | 62.4% | 55.3% | REJECT | No (818 rows) |
| 1d | 62.6% overall | ✅ | **DEPLOY (0.00)** | Yes (needs class rebalance) |
| 4h | 59.4% | 53.5% | REJECT | No (needs Optuna tuning) |
| 1h | **93.9% at >90% conf** | 53.9% | **DEPLOY (0.14)** | **YES — production ready** |

### 1h Confidence Breakdown (the money model)
```
>90% conf: 93.9% directional accuracy (605 trades)
>85% conf: 92.2% (2,091 trades)
>80% conf: 88.1% (4,830 trades)
>75% conf: 85.8% (9,018 trades)
>70% conf: 83.0% (14,921 trades)
```

---

## V3.3 FEATURE ENGINEERING — REVERSE-ENGINEER THE ALPHA

### Context
The v3.2 models showed a massive 14x preference for esoteric signals over traditional TA. We need to understand exactly what math, frequencies, or patterns the model is latching onto, and expand on it using the local vector database.

### Phase 1: Isolate the Alpha (Feature Analysis)
Read our v3.2 model artifacts (specifically `feature_stability_{tf}.json` containing the MDI/MDA rankings and the LightGBM gain metrics). Identify the top 5% of highest-contributing esoteric features.

Categorize these top features: Are they numerology (digital roots), astrology/planetary combos, or GCP consciousness data? What exact numerical thresholds, ratios, or specific categories drove the highest information gain?

### Phase 2: Vector Database Mining
Local vector database of esoteric texts at `C:\Users\C\Desktop\MY GOOGLE DRIVE\Orgonite master`. Contains deep literature on vortex math, Tesla's 3-6-9, sacred geometry, and more.

Using the top-performing categories identified in Phase 1 as semantic search context, query this database. Find the 'missing links.' For example, if the model heavily weighted a specific numerological sequence, what sacred geometry or vortex math concept in the database mathematically aligns with it that we are currently missing? Let the v3.2 winning data dictate what we search for.

### Phase 3: Engineer v3.3 Scalable Feature Logic
Do NOT give a small static list of features. We operate at a scale of over 2.9 million feature permutations based on daily cycles, global events, and market regimes.

Based on the texts retrieved from the vector database, write Python logic/functions for new Feature Groups to add to `feature_library.py`:

- If a new 3-6-9 harmonic resonance pattern is found, write the pandas/cuDF function that dynamically calculates this resonance against every traditional TA indicator (RSI, MACD, Bollinger Bands) across all market regimes (bull, bear, consolidation).
- If sacred geometry ratios are found, write the dynamic cross-feature logic that maps these ratios to existing 'Group 15: Decay features' (exponential decay since esoteric events).

The goal: write scalable code that mathematically generates thousands of new, logically sound permutations (Esoteric × TA × Regime) to feed into training.

---

## V3.3 TEMPORAL CASCADE — SUB-CANDLE ENTRIES & EXITS

### The Opportunity (validated by v3.2 results)
- 4h model: 93% win rate at >71.7% confidence, but enters/exits on 4h candle closes only
- 1h model: 93.9% directional at >90% confidence
- These two models are independent — no cross-TF communication

### Phase 4: Temporal Cascade Ensemble
Use higher-TF models for DIRECTION, lower-TF models for TIMING:

```
4h model signals LONG at >71.7% conf
  → 1h model times the exact entry within that 4h window
  → 15m model finds the optimal sub-hour entry tick
  → Same cascade for exits: 4h says "exit zone", 1h/15m finds exact exit
```

**Implementation:**
1. 4h predictions feed as meta-features to 1h model
2. 1h predictions + 4h context feed as meta-features to 15m model
3. Entry: wait for 4h signal, then 1h confirmation, then 15m trigger
4. Exit: 4h model says hold period ending, 15m model finds the exit candle

**Per Perplexity:** This is a "temporal cascade ensemble" — multi-scale information flow. NOT simple model averaging. Each TF contributes its unique resolution of the market.

### Phase 5: Dynamic Capital Allocation + Execution Overhaul

**v3.2 bottleneck identified:** 1h model has 98% directional accuracy at >80% confidence across 2,205 bars — but Optuna optimizer only takes 138 trades in 2 years (285% ROI). The alpha is there, the execution wastes it.

**Problems to fix in v3.3:**
1. **Only 138 of 2,205 high-accuracy signals used** — optimizer is too conservative with entry logic
2. **1h candle entries/exits** — bad fills. 15m timing can improve every entry by 0.5-1%
3. **Flat position sizing** — no compounding gains. With 98% accuracy, compounding is massive
4. **No pyramiding** — when consecutive bars confirm LONG at >80% conf, should add to position
5. **Fixed hold periods (44-62 hrs)** — should exit on signal reversal, not arbitrary time limit
6. **Single-TF execution** — no cross-TF confirmation or cascade

**v3.3 execution upgrades:**
- **Lower confidence threshold for entry when multi-TF confirms**: 4h+1h+15m all LONG >70% = enter, even though individual TF threshold might be 80%
- **Kelly criterion sizing**: 98% win rate × 6:1 RR = Kelly says bet big
- **Compound gains**: reinvest profits, not flat $100 per trade
- **Pyramid on confirmation**: if already LONG and new bar confirms >80% LONG, add 50% to position
- **15m exit timing**: don't hold to fixed bar count — exit when 15m model signals reversal or momentum exhaustion
- **Trade frequency target**: 2,205 qualifying bars / 2 years = ~3 trades per day at >80% confidence. Even capturing 50% = 1.5 trades/day = massive improvement over 1 trade per 5 days

### Phase 6: Walk-Forward Retrain
- Retrain models monthly on rolling 2-year window
- Use `save_binary()` to cache LightGBM datasets (skip 3hr loading)
- Optuna configs from v3.2 as warm-start for v3.3 search

---

## V3.3 TRAINING INFRASTRUCTURE LESSONS (from v3.2)

### LightGBM Sparse = Single-Threaded
- MUST convert to dense for multi-core training
- 15m (227K × 2.3M) needs 2TB+ RAM for dense
- Use `save_binary()` after first Dataset construction — all reruns skip 3hr loading
- `max_bin=63` not 255 — 4x faster histograms
- `force_col_wise=True` always

### Feature Chunking = BAD (violates matrix philosophy)
- Splitting features across models LOSES cross-chunk interactions
- "RSI only matters when Mercury is retrograde" requires BOTH in ONE model
- NEVER chunk features. Subsample ROWS if RAM is tight.

### Optuna is Fully Decoupled
- Run separately from training pipeline
- Only needs parquet + NPZ + btc_prices.db
- Use `save_binary()` to avoid reloading for each trial

### Optuna Speed Fixes (Perplexity-validated, 2026-03-24)
Original: 60 S1 + 20 S2 trials × 4-fold CPCV × 300 rounds = **13+ hours on 2.9M features**
Fixed: 20 S1 + 20 S2 trials × 2-fold search × 150 rounds = **~90 min**

| Setting | Old | New | Why |
|---|---|---|---|
| S1 trials | 60 | **20** | TPE gain flattens after ~20 |
| Search rounds | 300 | **150** | Early stopping finds optimum; final uses 800 |
| Search CPCV folds | 4 | **2** | Search needs direction, not low-variance estimates |
| `reference=dtrain` | No | **Yes** | Reuses bin boundaries for validation set |
| `save_binary()` | No | **TODO** | Pre-cache fold datasets = skip re-binning entirely |
| `bagging_freq` | 1 | 1 | Verified: 30% subsample IS active |

**All features stay. Model quality identical on final retrain (800 rounds, full CPCV).**
Future: implement `save_binary()` per-fold caching for Stage 2 + final retrain = additional 3-4x speedup.

### Machine Requirements for v3.3
| TF | Dense RAM | Min Cores | Notes |
|---|---|---|---|
| 1w | 2 GB | 64 | Tiny, any machine |
| 1d | 66 GB | 128 | Standard |
| 4h | 56 GB | 128 | Standard |
| 1h | 300 GB | 256 | Need 512GB+ machine |
| 15m | 1.89 TB | 256 | Need 2TB+ machine (rare) or AWS x2idn Spot |

---

## ⚠ 15m REQUIRES SPECIAL MACHINE — 2TB+ RAM

15m is the problem child. 227,577 rows × 2,283,592 features = **1.89 TB dense**. Normal cloud machines top out at 1TB.

**Options (ranked):**
1. **vast.ai Belgium 30818859** — 2TB RAM, 256c, $1.15/hr. ONLY 2TB machine on vast.ai. SSH can be unstable.
2. **AWS Spot x2idn.32xlarge** — 2TB RAM, 128c, ~$3-5/hr Spot. Needs quota increase (128 vCPU for X family, takes 1-2 days).
3. **Vultr MI355X** — 3TB RAM, 256c, $20.72/hr. Expensive but guaranteed available.
4. **Row subsample to 210K** — 1.81TB, fits on 2TB machine with headroom. Loses 7% of rows (keeps most recent).
5. **Sparse single-threaded** — works on any machine with 64GB+ RAM, but takes 8-12 hours. Valid per Perplexity (int64 safe in v4.6).

**For v3.3:** If feature count grows beyond 2.3M, 15m dense may exceed even 2TB. Plan for AWS x2idn or Vultr. Request AWS quota increase NOW before training day.

**Cross gen for 15m also takes 2+ hours** (single-threaded, 227K rows). Budget this time. Never kill a running 15m cross gen.

---

## OPERATIONAL RULES (learned the hard way in v3.2)

### Deployment
1. **Run 1 TF at a time starting with smallest (1w)** — verify FULL pipeline before scaling
2. **Audit ALL scripts for the same bug pattern** before deploying. When you fix a bug in ml_multi_tf.py, grep every other .py file for the same pattern
3. **Verify Docker image tags exist** before launching instances
4. **Never use nohup bash wrappers** — use cloud_run_tf.py directly
5. **Pipeline must be fully modular** — every asset/TF independently resumable across machines
6. **NPZ skip logic** — if v2_crosses_BTC_{tf}.npz already exists, skip cross gen. Never waste 2hrs rebuilding

### Monitoring
7. **Always run scripts with progress logs** (tee/unbuffered). Never run blind
8. **vast.ai machines die without warning** — download partial results FREQUENTLY, not just at DONE
9. **VERIFY outputs before killing machines** — check file sizes, SPARSE confirmed, accuracy present. Once killed = gone forever
10. **Download NPZ from every machine** — it's the one artifact you can't reconstruct without rerunning cross gen

### LSTM
11. **LSTM crashes on NaN features** — must `np.nan_to_num()` after z-score normalization. LightGBM handles NaN natively but LSTM propagates NaN → CUDA assertion crash
12. **LSTM uses `features_{tf}.parquet`** (no BTC prefix) — must create symlink to `features_BTC_{tf}.parquet`

### Cross Generator
13. **Cross gen is ALWAYS single-threaded** — core count doesn't help. 15m takes ~2 hours
14. **Never kill a running cross gen** — no checkpointing, all progress lost
15. **`--symbol BTC` not `--asset BTC`** — and set `df._v2_symbol = args.symbol` after read_parquet

### LightGBM Specifics
16. **Sparse NNZ > 2^31**: Python wrapper handles int64 (safe), but C++ PushDataToMultiValBin has separate speed issue (#5205)
17. **`save_binary()` after first Dataset construction** — CRITICAL for Optuna (80 trials = 80 loads skipped)
18. **`max_bin=63`** not 255 — 4x faster histogram construction with minimal accuracy loss
19. **`is_unbalance=True`** for TFs with FLAT class imbalance (1d = 60% FLAT)
20. **`.toarray()` cascade** — after converting sparse→dense, guard ALL downstream `.toarray()` calls with `hasattr(X, 'toarray')`
21. **`nnz` cascade** — after dense conversion, guard `.nnz` with `hasattr(X, 'nnz')`

### Cost Management
22. **Ask user before renting machines** — never pick without approval
23. **Kill machines as soon as downloads verified** — $0.80-1.36/hr adds up fast
24. **Maximize parallelism** — run independent TFs on separate machines simultaneously
25. **Upload tars (~200MB) is always faster than rebuilding** — 30s upload vs hours of cross gen
26. **REUSE running machines** — when fixing code, scp new files + relaunch. Don't kill + rent new. Saves boot time + money every time.

### 4h Optuna Results (v3.2 validated)
```
Conservative: 93% win rate, Sortino 209, 14 trades, $10K→$10,455
Aggressive:   69% win rate, Sortino 1.27, 83 trades, $10K→$27,301
```
