# SAVAGE22 DEFINITIVE TRAINING PLAN
## Modular, Resumable, Microstep-Level Execution Guide
### Last Updated: 2026-03-21

---

## TABLE OF CONTENTS
1. [Decision: V1 vs V2 Priority](#decision-v1-vs-v2-priority)
2. [Pre-Flight: Local Bug Fixes (Before Renting Cloud)](#phase-0-pre-flight-local-bug-fixes)
3. [Pre-Flight: V2 Intraday Feature Builds (Local)](#phase-0b-v2-intraday-feature-builds-local)
4. [Pre-Flight: Smoke Tests & Upload Checklist](#phase-0c-smoke-tests--upload-checklist)
5. [Cloud Phase 1: Feature Builds (V1 + V2 Intraday)](#cloud-phase-1-feature-builds)
6. [Cloud Phase 2: XGBoost Training](#cloud-phase-2-training)
7. [Cloud Phase 3: Exhaustive Optimizer](#cloud-phase-3-exhaustive-optimizer)
8. [Cloud Phase 4: PBO + Deflated Sharpe Validation](#cloud-phase-4-validation)
9. [Cloud Phase 5: Meta-Labeling](#cloud-phase-5-meta-labeling)
10. [Cloud Phase 6: LSTM Training](#cloud-phase-6-lstm)
11. [Cloud Phase 7: Backtesting Audit](#cloud-phase-7-backtesting-audit)
12. [Post-Training: Download & Verify](#post-training-download--verify)
13. [Post-Training: Paper Trading Activation](#post-training-paper-trading)

---

## DECISION: V1 VS V2 PRIORITY

**Run V2 first. V1 second (or skip if V2 succeeds).**

Rationale:
- V2 has 31 assets, 15-20M sparse features, multi-asset validation -- strictly superior
- V2 daily builds are DONE (31/31 parquets + npz)
- V2 only needs intraday builds (4h, 1h, 15m, 5m) + full training pipeline
- V1 has 6 TFs for BTC-only, 150K features -- already battle-tested code but needs rebuild after bug fixes
- Both share the same core scripts (feature_library.py, exhaustive_optimizer.py, backtest_validation.py, meta_labeling.py)
- V2 cloud_runner is more mature (881 lines, manifest checkpoint, OOM retry, heartbeat)

**V1 can run AFTER V2 on the same cloud instance if time/budget allows.** The V1 cloud_runner.py is simpler but functional.

---

## PHASE 0: PRE-FLIGHT — LOCAL BUG FIXES
**Location: Local (13900K + RTX 3090)**
**Estimated time: 2-4 hours**
**Must complete BEFORE renting cloud GPU**

### Bug Cross-Reference: What Is Already Fixed vs Still Open

From the 14-bug audit list, cross-referenced against session_resume.md:

| # | Bug | Status | Notes |
|---|-----|--------|-------|
| 1 | V1 optimizer direction (Bug A) | FIXED | simulate_batch receives directions |
| 2 | V1 optimizer close-only SL/TP (Bug C) | FIXED | Uses highs/lows |
| 3 | V1 PBO n_trials hardcoded 100 | FIXED | Now uses actual grid size |
| 4 | 5m/15m SQL missing columns | NEEDS VERIFY | May be fixed by parquet-first saves |
| 5 | Timestamped onchain data never used | NEEDS FIX | ~15 features always NaN in V1 builds |
| 6 | News schema inversion | NEEDS FIX | build scripts vs data_access mismatch |
| 7 | Space weather format mismatch | PARTIALLY FIXED | V2 live_trader fixed, V1 live_trader may still have issue |
| 8 | HMM state mapping inverted in live | NEEDS VERIFY | V2 fixed, V1 may still be wrong |
| 9 | GCP features missing in live (1h) | NEEDS FIX | 22 features NaN in V1 1h model |
| 10 | Sports columns missing in live | NEEDS FIX | ~38 features NaN in V1 live |
| 11 | V2 live_trader 3-class prediction | FIXED | multi:softprob 3-class parsing |
| 12 | V2 live_trader missing space_weather/HMM | FIXED | Added to LiveDataLoader |
| 13 | compute_sample_uniqueness needs @njit | NEEDS FIX | CPU bottleneck in training |
| 14 | Early stopping on test fold | FIXED | 85/15 validation split |

### 0.1 — Fix compute_sample_uniqueness @njit
- **Input:** `ml_multi_tf.py` (V1), `v2_multi_asset_trainer.py` (V2)
- **What:** Add `@njit` decorator to the inner loop of `_compute_sample_uniqueness()`. The function iterates over 500K+ events with Python for-loops -- must be Numba-compiled.
- **Output:** Both files updated with `@njit` on the inner loop function
- **Verification:** `python -c "from ml_multi_tf import _compute_sample_uniqueness; print('OK')"`
- **Time estimate:** 15 min
- **If interrupted:** No state -- just re-edit the file
- **Resume check:** grep for `@njit` above `_compute_sample_uniqueness` in both files

### 0.2 — Fix onchain timestamped data (V1)
- **Input:** `data_access.py`, `feature_library.py`, onchain_data.db schema
- **What:** The onchain data has timestamped rows but the merge/join logic drops them. ~15 features are always NaN. Fix the join to properly align timestamps.
- **Output:** `data_access.py` updated with correct onchain merge logic
- **Verification:** `python -c "from data_access import get_onchain_data; df=get_onchain_data(); print(df.notna().sum())"` -- should show non-zero counts for all columns
- **Time estimate:** 30 min
- **If interrupted:** No state -- just re-edit
- **Resume check:** Run verification command above

### 0.3 — Fix news schema inversion (V1)
- **Input:** `data_access.py`, build_*_features.py scripts, news_articles.db schema
- **What:** Build scripts expect columns in different order/names than data_access returns. Fix data_access to match what build scripts expect (or vice versa).
- **Output:** Consistent column names between data_access and build scripts
- **Verification:** `python -c "from data_access import get_news_data; df=get_news_data(); print(df.columns.tolist())"`
- **Time estimate:** 20 min
- **If interrupted:** No state -- just re-edit
- **Resume check:** Run verification command

### 0.4 — Fix V1 space weather live format
- **Input:** `live_trader.py` (V1), `data_access.py`
- **What:** Space weather data has RangeIndex in live but DatetimeIndex in training. 17 features are all-NaN in V1 live inference. Ensure consistent index handling.
- **Output:** `live_trader.py` converts space weather to DatetimeIndex at load time
- **Verification:** Manually inspect that space weather features are non-NaN in a test inference call
- **Time estimate:** 20 min
- **If interrupted:** No state -- just re-edit
- **Resume check:** grep for `DatetimeIndex` or `set_index` near space_weather in live_trader.py

### 0.5 — Fix V1 HMM state mapping
- **Input:** `live_trader.py` (V1)
- **What:** HMM bull_prob and bear_prob were swapped in live inference. Verify the mapping matches training.
- **Output:** Correct state-to-label mapping in live_trader.py
- **Verification:** Compare HMM state means (higher mean = bull state) in both train and live code paths
- **Time estimate:** 15 min
- **If interrupted:** No state -- just re-edit
- **Resume check:** Inspect HMM state mapping in live_trader.py

### 0.6 — Fix V1 GCP/sports features missing in live
- **Input:** `live_trader.py` (V1), `data_access.py`
- **What:** 1h model uses ~22 GCP features and ~38 sports features that are never computed in live inference. Either add them to the live pipeline or ensure they are correctly set to NaN (not 0) so XGBoost handles them as missing.
- **Output:** Live inference computes or correctly handles these features
- **Verification:** Run a test inference and check that feature names match training feature list
- **Time estimate:** 45 min
- **If interrupted:** No state -- just re-edit
- **Resume check:** Compare `model.feature_names` vs live feature DataFrame columns

### 0.7 — Fix 5m/15m SQL column completeness
- **Input:** `data_access.py`, btc_prices.db schema
- **What:** 5m and 15m tables may be missing `quote_volume`, `trades`, `taker_buy_quote` columns. With parquet-first saves this may be moot, but verify the source data has these columns.
- **Output:** Confirm columns exist or add fallback NaN columns in build scripts
- **Verification:** `python -c "import sqlite3; c=sqlite3.connect('btc_prices.db'); print([x[1] for x in c.execute('PRAGMA table_info(btc_5m)').fetchall()])"`
- **Time estimate:** 15 min
- **If interrupted:** No state
- **Resume check:** Run verification command

### 0.8 — Verify all V2 bug fixes are applied
- **Input:** V2 session_resume.md fix list
- **What:** Confirm BUG 1-6 and all HIGH fixes from the second audit are still in the codebase (no file reverts).
- **Output:** Checkmark for each fix
- **Verification:** grep for key fix signatures:
  - `compute_features_live.*returns.*dict.*DataFrame` (Bug 1)
  - `get_space_weather` in `live_trader.py` V2 (Bug 2)
  - `hmm_bull_prob` computation in V2 live (Bug 3)
  - `crash` regime with `rvol > 2` (Bug 4 HIGH)
  - `classify_expected_trade_type` called (Bug 5 HIGH)
  - `TF_CAPITAL_ALLOC` imported from config.py (Bug 6 HIGH)
- **Time estimate:** 20 min
- **If interrupted:** No state
- **Resume check:** All greps return matches

### PHASE 0 GATE
**All 8 sub-steps must PASS before proceeding. If any fail, fix and re-verify.**
**Checkpoint file: `preflight_bugs_fixed.json`** -- create this file with timestamps for each fix after all pass.

---

## PHASE 0B: V2 INTRADAY FEATURE BUILDS (LOCAL)
**Location: Local (13900K + RTX 3090, 128GB RAM)**
**Estimated time: 3-5 hours (staggered)**
**Purpose: Build the 4 intraday TFs that OOM'd on cloud**

The V2 daily builds (31/31) are complete. Intraday builds (4h, 1h, 15m, 5m) failed on cloud due to RAM OOM. Build locally where we can stagger and control memory.

### Why local, not cloud:
- Feature builds are CPU-bound (.apply() UDFs eliminated but cuDF rolling still benefits from fast CPU)
- Local 13900K has the fastest single-core speed available
- Staggered builds keep RAM under 128GB
- No cloud rental cost for the slow CPU-bound phase
- Upload parquets to cloud for GPU-only training

### 0B.1 — Build 14 crypto x 4h features (parallel batch 1)
- **Input:** `v2/build_features_v2.py`, `multi_asset_prices.db`, all V1 databases
- **Command:**
  ```bash
  cd "C:\Users\C\Documents\Savage22 Server\v2"
  python -u build_features_v2.py --tf 4h --parallel 2 2>&1 | tee build_4h.log
  ```
- **Output:** 14 files: `features_{SYMBOL}_4h.parquet` + `v2_crosses_{SYMBOL}_4h.npz`
- **Peak RAM:** ~40-50 GB (2 concurrent builds)
- **Time estimate:** ~60 min
- **Progress indicator:** Builder logs per-symbol start/end with column counts
- **Verification:**
  ```bash
  ls -la features_*_4h.parquet | wc -l  # should be 14
  ls -la v2_crosses_*_4h.npz | wc -l   # should be 14
  python -c "import pandas as pd; df=pd.read_parquet('features_BTC_4h.parquet'); print(f'{len(df)} rows, {len(df.columns)} cols')"
  ```
- **If interrupted:** Checkpointed -- existing parquets are skipped on restart. Just re-run same command.
- **Resume check:** Count parquet + npz files. If 14 each, skip this step.

### 0B.2 — Build 14 crypto x 1h features (parallel batch 2)
- **Command:**
  ```bash
  python -u build_features_v2.py --tf 1h --parallel 2 2>&1 | tee build_1h.log
  ```
- **Output:** 14 files each (parquet + npz)
- **Peak RAM:** ~60-80 GB (2 concurrent builds, 56K rows each)
- **Time estimate:** ~75 min
- **If interrupted:** Checkpointed. Re-run.
- **Resume check:** `ls features_*_1h.parquet | wc -l` == 14

### 0B.3 — Build BTC 15m features (solo)
- **Command:**
  ```bash
  python -u build_features_v2.py --symbol BTC --tf 15m 2>&1 | tee build_15m.log
  ```
- **Output:** `features_BTC_15m.parquet` + `v2_crosses_BTC_15m.npz`
- **Peak RAM:** ~130 GB (217K rows). Will use swap if only 128GB physical -- monitor with `free -h`.
- **Time estimate:** ~45 min
- **If interrupted:** Partially built parquet is NOT saved (atomic save). Must restart from scratch.
- **Resume check:** `features_BTC_15m.parquet` exists and is non-zero size
- **FALLBACK if OOM:** Set `V2_RIGHT_CHUNK=250` (half default) to reduce cross batch memory:
  ```bash
  V2_RIGHT_CHUNK=250 python -u build_features_v2.py --symbol BTC --tf 15m 2>&1 | tee build_15m.log
  ```

### 0B.4 — Build BTC 5m features (solo, largest)
- **Command:**
  ```bash
  python -u build_features_v2.py --symbol BTC --tf 5m 2>&1 | tee build_5m.log
  ```
- **Output:** `features_BTC_5m.parquet` + `v2_crosses_BTC_5m.npz`
- **Peak RAM:** ~250 GB. **WILL NOT FIT in 128GB.** Options:
  1. **Option A (recommended):** Build on cloud only. Include 5m in cloud Phase 1.
  2. **Option B:** Use V2_RIGHT_CHUNK=100 for smaller cross batches + heavy swap. Very slow (~4 hours).
  3. **Option C:** Skip 5m for V2 (V1 covers BTC 5m). Can add later.
- **Time estimate:** Option A: cloud only. Option B: ~4 hours local. Option C: skip.
- **Decision:** **Build 5m on cloud (Option A).** Only 1 asset, single build -- fast even on cloud.
- **Resume check:** `features_BTC_5m.parquet` exists

### 0B.5 — Verify all V2 intraday builds
- **Command:**
  ```bash
  python -c "
  import os, pandas as pd
  v2 = r'C:\Users\C\Documents\Savage22 Server\v2'
  for tf in ['4h', '1h', '15m']:
      pqs = [f for f in os.listdir(v2) if f.startswith('features_') and f.endswith(f'_{tf}.parquet')]
      npzs = [f for f in os.listdir(v2) if f.startswith('v2_crosses_') and f.endswith(f'_{tf}.npz')]
      print(f'{tf}: {len(pqs)} parquets, {len(npzs)} npz')
      if pqs:
          df = pd.read_parquet(os.path.join(v2, pqs[0]))
          print(f'  Sample: {pqs[0]} -> {len(df)} rows, {len(df.columns)} cols')
  "
  ```
- **Expected output:**
  - 4h: 14 parquets, 14 npz
  - 1h: 14 parquets, 14 npz
  - 15m: 1 parquet, 1 npz
- **Time estimate:** 2 min
- **Checkpoint file:** `v2_intraday_builds_complete.json` with file counts and sizes

### PHASE 0B GATE
**Minimum requirement: 4h (14) + 1h (14) + 15m (1) complete. 5m is optional (cloud build).**
**Total local build time: ~3 hours staggered**

---

## PHASE 0C: SMOKE TESTS & UPLOAD CHECKLIST
**Location: Local**
**Estimated time: 30-60 min**

### 0C.1 — Run V2 smoke test
- **Command:**
  ```bash
  cd "C:\Users\C\Documents\Savage22 Server\v2"
  python -u smoke_test_pipeline.py 2>&1 | tee smoke_test.log
  ```
- **Expected:** 10/10 PASS in ~5s
- **If any FAIL:** Fix the specific test before proceeding. Do not deploy broken code.
- **Time estimate:** 5 min

### 0C.2 — Run V1 smoke test
- **Command:**
  ```bash
  cd "C:\Users\C\Documents\Savage22 Server"
  python -u smoke_test_pipeline.py 2>&1 | tee smoke_test_v1.log
  ```
- **Expected:** 10/10 PASS
- **Time estimate:** 5 min

### 0C.3 — Validate all databases exist
- **Command:**
  ```bash
  cd "C:\Users\C\Documents\Savage22 Server"
  for db in btc_prices.db astrology_full.db ephemeris_cache.db fear_greed.db \
    funding_rates.db google_trends.db news_articles.db tweets.db macro_data.db \
    onchain_data.db open_interest.db space_weather.db sports_results.db llm_cache.db; do
    if [ -f "$db" ]; then
      echo "OK: $db ($(du -h $db | cut -f1))"
    else
      echo "MISSING: $db"
    fi
  done
  ```
- **All must show OK. Any MISSING is a blocker.**
- **Time estimate:** 1 min

### 0C.4 — Check vast.ai balance and add funds
- **Command:**
  ```bash
  vastai show user
  ```
- **Required balance:** $15+ (est. $7-12 for V2, $5-8 for V1 if running both)
- **If balance < $15:** Add funds via vast.ai website before proceeding
- **Time estimate:** 5 min (just checking, adding funds is external)

### 0C.5 — Select cloud machine
- **Target specs:**
  - 4+ GPUs with 24GB+ VRAM each (RTX 4090 ideal, A100/H100 if budget allows)
  - 512GB+ RAM (for 5m build if running on cloud)
  - Fast CPU (3.5GHz+) helps but less critical since builds are mostly done
  - Docker: `rapidsai/base:25.02-cuda12.5-py3.12`
- **Search command:**
  ```bash
  vastai search offers 'num_gpus >= 4 gpu_ram >= 20 cpu_ram >= 500 reliability > 0.95 dph <= 5.0' -o 'dph'
  ```
- **Previously selected:** 8x RTX 4090, 774GB RAM, $2.88/hr, Sichuan China (ID 30228399)
- **Verify it's still available** before starting builds
- **Time estimate:** 10 min

### 0C.6 — Prepare upload file list

**V2 Scripts (upload to `/workspace/v2/`):**
```
v2/config.py
v2/hardware_detect.py
v2/atomic_io.py
v2/data_access_v2.py
v2/feature_library.py
v2/v2_feature_layers.py
v2/v2_cross_generator.py
v2/build_features_v2.py
v2/v2_multi_asset_trainer.py
v2/exhaustive_optimizer.py
v2/backtest_validation.py
v2/meta_labeling.py
v2/v2_lstm_trainer.py
v2/v2_cloud_runner.py
v2/backtesting_audit.py
v2/smoke_test_pipeline.py
v2/ml_multi_tf.py
v2/gpu_cross_builder.py
v2/knn_feature_engine.py
v2/universal_gematria.py
v2/universal_sentiment.py
v2/universal_numerology.py
v2/universal_astro.py
v2/portfolio_optimizer.py
v2/streamer_supervisor.py
v2/live_trader.py
```

**V2 Pre-built Features (upload to `/workspace/v2/`):**
```
v2/features_*_1d.parquet    (31 files)
v2/v2_crosses_*_1d.npz     (31 files)
v2/features_*_4h.parquet    (14 files, after local build)
v2/v2_crosses_*_4h.npz     (14 files)
v2/features_*_1h.parquet    (14 files)
v2/v2_crosses_*_1h.npz     (14 files)
v2/features_BTC_15m.parquet (1 file)
v2/v2_crosses_BTC_15m.npz  (1 file)
```

**V1 Scripts (upload to `/workspace/`, only if running V1):**
```
feature_library.py
ml_multi_tf.py
exhaustive_optimizer.py
lstm_sequence_model.py
gpu_cross_builder.py
knn_feature_engine.py
build_{1h,4h,15m,5m,1d,1w}_features.py
universal_{gematria,numerology,astro,sentiment}.py
data_access.py
cloud_runner.py
backtest_validation.py
meta_labeling.py
smoke_test_pipeline.py
backtesting_audit.py
```

**Databases (upload to `/workspace/`):**
```
btc_prices.db
astrology_full.db
ephemeris_cache.db
fear_greed.db
funding_rates.db
google_trends.db
news_articles.db
tweets.db
macro_data.db
onchain_data.db
open_interest.db
space_weather.db
sports_results.db
llm_cache.db
v2/multi_asset_prices.db
v2/v2_signals.db
```

**Data Files:**
```
kp_history.txt
dst_index.json
dynamic_config.json
systematic_cross_results_1h.csv
systematic_cross_results_4h.csv
```

### 0C.7 — Estimate upload size and time
- **Command:**
  ```bash
  du -sh v2/features_*_1d.parquet v2/v2_crosses_*_1d.npz | tail -1
  du -sh *.db v2/*.db | tail -1
  ```
- **Expected total:** ~4-8 GB (parquets ~2GB, npz ~1.5GB, databases ~2GB, scripts ~2MB)
- **Upload time at 10 Mbps:** ~60 min. At 50 Mbps: ~15 min.
- **Use rsync with compression:** `rsync -avz --progress`

### PHASE 0C GATE
**All smoke tests pass. All DBs exist. Balance >= $15. Machine selected. File list confirmed.**

---

## CLOUD DEPLOYMENT

### Rent & Connect
```bash
# 1. Start instance
vastai create instance <OFFER_ID> --image rapidsai/base:25.02-cuda12.5-py3.12 --disk 100

# 2. Get SSH info
vastai show instances --raw | python -m json.tool

# 3. Connect
ssh -p <PORT> root@<HOST>
```

### Upload
```bash
# From local machine:
# V2 scripts
rsync -avz --progress "C:/Users/C/Documents/Savage22 Server/v2/" root@<HOST>:/workspace/v2/

# V1 scripts (if running V1)
rsync -avz --progress --include="*.py" --include="*.sh" --exclude="*" \
  "C:/Users/C/Documents/Savage22 Server/" root@<HOST>:/workspace/

# Databases
rsync -avz --progress "C:/Users/C/Documents/Savage22 Server/"*.db root@<HOST>:/workspace/
rsync -avz --progress "C:/Users/C/Documents/Savage22 Server/v2/"*.db root@<HOST>:/workspace/v2/

# Data files
rsync -avz --progress "C:/Users/C/Documents/Savage22 Server/kp_history.txt" \
  "C:/Users/C/Documents/Savage22 Server/dst_index.json" \
  "C:/Users/C/Documents/Savage22 Server/dynamic_config.json" \
  "C:/Users/C/Documents/Savage22 Server/systematic_cross_results_"*.csv \
  root@<HOST>:/workspace/
```

### Verify Upload
```bash
ssh -p <PORT> root@<HOST> 'ls /workspace/v2/features_*_1d.parquet | wc -l'  # 31
ssh -p <PORT> root@<HOST> 'ls /workspace/*.db | wc -l'                       # 14
```

---

## CLOUD PHASE 1: FEATURE BUILDS
**Location: Cloud GPU**
**Only needed for: BTC 5m (too large for local RAM)**
**Estimated time: ~30-40 min for 5m only**
**Cost: ~$1.50 at $2.88/hr**

### 1.1 — Build BTC 5m on cloud
- **Command (on cloud):**
  ```bash
  cd /workspace/v2
  python -u build_features_v2.py --symbol BTC --tf 5m 2>&1 | tee build_5m.log
  ```
- **Output:** `features_BTC_5m.parquet` + `v2_crosses_BTC_5m.npz`
- **Peak RAM:** ~250 GB (cloud has 774GB, safe)
- **Time estimate:** 30-40 min
- **Progress indicator:** Per-compute-function timing logs, cross batch progress
- **Verification:**
  ```bash
  python -c "import pandas as pd; df=pd.read_parquet('features_BTC_5m.parquet'); print(f'{len(df)} rows, {len(df.columns)} cols')"
  # Expected: ~547K rows, ~10K+ cols
  ```
- **If interrupted:** Atomic saves mean partial file is not written. Re-run same command.
- **Resume check:** `features_BTC_5m.parquet` exists with non-zero size

### 1.2 — Verify all feature files present
- **Command:**
  ```bash
  cd /workspace/v2
  python -c "
  import os
  tfs = {'1d': 31, '4h': 14, '1h': 14, '15m': 1, '5m': 1}
  for tf, expected in tfs.items():
      pqs = [f for f in os.listdir('.') if f.startswith('features_') and f.endswith(f'_{tf}.parquet')]
      npzs = [f for f in os.listdir('.') if f.startswith('v2_crosses_') and f.endswith(f'_{tf}.npz')]
      status = 'OK' if len(pqs) == expected and len(npzs) == expected else 'FAIL'
      print(f'{tf}: {len(pqs)}/{expected} parquets, {len(npzs)}/{expected} npz -> {status}')
  "
  ```
- **All must show OK. Any FAIL is a blocker for Phase 2.**

---

## CLOUD PHASE 2: TRAINING
**Location: Cloud GPU**
**Estimated time: ~50 min**
**Cost: ~$2.40 at $2.88/hr**

### Launch via orchestrator (recommended):
```bash
cd /workspace/v2
tmux new -s pipeline 'python -u v2_cloud_runner.py \
  --phase 2 \
  --tf 1d 4h 1h 15m 5m \
  --mode production \
  --engine xgboost \
  --boost-rounds 500 \
  --resume \
  --dph 2.88 \
  2>&1 | tee phase2.log'
```

### Or run each TF manually for more control:

### 2.1 — Train daily models (31 assets x 1d)
- **Command:**
  ```bash
  python -u v2_multi_asset_trainer.py --mode production --tf 1d --engine xgboost \
    --boost-rounds 500 --parallel-splits --resume 2>&1 | tee train_1d.log
  ```
- **Output:**
  - `model_v2_production_xgboost_1d.json` (XGBoost model)
  - `importance_v2_production_1d.json` (feature importance)
  - `oos_predictions_production_1d.pkl` (CPCV OOS predictions)
  - `training_report_production_1d.json` (metrics)
- **Time estimate:** ~15 min (31 assets, CPCV with 15 paths, GPU histogram)
- **Progress indicator:** Per-fold training logs with accuracy, logloss. GPU utilization in heartbeat.
- **Verification:**
  ```bash
  python -c "
  import json
  r = json.load(open('training_report_production_1d.json'))
  print(f'Accuracy: {r.get(\"test_accuracy\", \"N/A\")}')
  print(f'Folds: {r.get(\"n_folds\", \"N/A\")}')
  "
  ```
- **If interrupted:** `--resume` flag skips completed folds. Re-run same command.
- **Resume check:** `model_v2_production_xgboost_1d.json` exists

### 2.2 — Train 4h crypto models
- **Command:**
  ```bash
  python -u v2_multi_asset_trainer.py --mode production --tf 4h --engine xgboost \
    --boost-rounds 500 --parallel-splits --resume 2>&1 | tee train_4h.log
  ```
- **Output:** Same pattern as 2.1 but for 4h
- **Time estimate:** ~10 min (14 crypto assets, more rows per asset)
- **Resume check:** `model_v2_production_xgboost_4h.json` exists

### 2.3 — Train 1h crypto models
- **Command:**
  ```bash
  python -u v2_multi_asset_trainer.py --mode production --tf 1h --engine xgboost \
    --boost-rounds 500 --parallel-splits --resume 2>&1 | tee train_1h.log
  ```
- **Output:** Same pattern for 1h
- **Time estimate:** ~10 min
- **Resume check:** `model_v2_production_xgboost_1h.json` exists

### 2.4 — Train 15m BTC model
- **Command:**
  ```bash
  python -u v2_multi_asset_trainer.py --mode production --tf 15m --engine xgboost \
    --boost-rounds 500 --parallel-splits --resume 2>&1 | tee train_15m.log
  ```
- **Output:** Same pattern for 15m
- **Time estimate:** ~8 min (BTC only, 217K rows)
- **Resume check:** `model_v2_production_xgboost_15m.json` exists

### 2.5 — Train 5m BTC model
- **Command:**
  ```bash
  python -u v2_multi_asset_trainer.py --mode production --tf 5m --engine xgboost \
    --boost-rounds 500 --parallel-splits --resume 2>&1 | tee train_5m.log
  ```
- **Output:** Same pattern for 5m
- **Time estimate:** ~12 min (BTC only, 547K rows, largest)
- **Resume check:** `model_v2_production_xgboost_5m.json` exists
- **VRAM note:** 547K rows x 10K+ features as sparse CSR. XGBoost DMatrix handles sparse natively. If OOM, reduce `V2_BATCH_SIZE` env var.

### 2.6 — Verify all models
- **Command:**
  ```bash
  for tf in 1d 4h 1h 15m 5m; do
    if [ -f "model_v2_production_xgboost_${tf}.json" ]; then
      echo "OK: model_v2_production_xgboost_${tf}.json"
    else
      echo "MISSING: model_v2_production_xgboost_${tf}.json"
    fi
    if [ -f "oos_predictions_production_${tf}.pkl" ]; then
      echo "OK: oos_predictions_production_${tf}.pkl"
    else
      echo "MISSING: oos_predictions_production_${tf}.pkl"
    fi
  done
  ```
- **All must show OK. OOS predictions are required for Phases 4, 5, 6.**

### Phase 2 GPU Memory Cleanup
```bash
python -c "import gc; gc.collect(); import torch; torch.cuda.empty_cache(); import cupy; cupy.get_default_memory_pool().free_all_blocks(); print('GPU cleaned')"
```

---

## CLOUD PHASE 3: EXHAUSTIVE OPTIMIZER
**Location: Cloud GPU**
**Estimated time: ~35 min**
**Cost: ~$1.70 at $2.88/hr**

### Launch via orchestrator:
```bash
tmux new -s opt 'cd /workspace/v2 && python -u v2_cloud_runner.py \
  --phase 3 \
  --tf 1d 4h 1h 15m 5m \
  --resume \
  --dph 2.88 \
  2>&1 | tee phase3.log'
```

### Or manually per TF:

### 3.1 — Optimize all TFs (parallel across GPUs)
- **Command:**
  ```bash
  python -u exhaustive_optimizer.py --tf 1d --tf 4h --tf 1h --tf 15m --tf 5m --resume \
    2>&1 | tee optimize.log
  ```
- **What it does:** Exhaustive grid search over ~30M parameter combinations per TF:
  - Leverage: 1-10x
  - Risk: 0.5-5%
  - Stop-loss: 0.5-5%
  - R:R: 1-5
  - Hold: varies by TF
  - Now with: direction-aware entries, intrabar SL/TP on highs/lows, regime adaptation
- **Output per TF:** `exhaustive_configs_{tf}.json`
- **Time estimate:** ~7 min per TF, ~35 min total (parallelized across GPUs)
- **Progress indicator:** Prints every 5% batch with Sortino, win rate, trade count
- **Checkpoint:** Saves progress every 5% to `optimizer_checkpoint_{tf}.json`. On `--resume`, continues from last checkpoint.
- **Verification:**
  ```bash
  for tf in 1d 4h 1h 15m 5m; do
    if [ -f "exhaustive_configs_${tf}.json" ]; then
      python -c "import json; c=json.load(open('exhaustive_configs_${tf}.json')); print(f'${tf}: {len(c)} configs, best Sortino={c[0][\"sortino\"]:.2f}')"
    else
      echo "MISSING: exhaustive_configs_${tf}.json"
    fi
  done
  ```
- **If interrupted:** `--resume` continues from last 5% checkpoint. Re-run same command.
- **Resume check:** All 5 config files exist
- **CRITICAL NOTE:** Old configs are INVALID (direction bug was fixed). Must regenerate ALL.

---

## CLOUD PHASE 4: VALIDATION (PBO + Deflated Sharpe)
**Location: Cloud (in-process, lightweight)**
**Estimated time: ~5 min total**
**Cost: ~$0.25**

### 4.1 — Run validation for all TFs
- **Command (via orchestrator):**
  ```bash
  python -u v2_cloud_runner.py --phase 4 --tf 1d 4h 1h 15m 5m --resume --dph 2.88 \
    2>&1 | tee phase4.log
  ```
- **Or manually per TF:**
  ```bash
  python -c "
  import pickle, json
  from backtest_validation import validation_report
  for tf in ['1d', '4h', '1h', '15m', '5m']:
      oos_path = f'oos_predictions_production_{tf}.pkl'
      try:
          with open(oos_path, 'rb') as f:
              oos = pickle.load(f)
          report = validation_report(oos, tf_name=tf)
          rpt_path = f'validation_report_production_{tf}.json'
          with open(rpt_path, 'w') as f:
              json.dump(report, f, indent=2, default=str)
          pbo = report.get('pbo', {})
          dsr = report.get('deflated_sharpe', {})
          print(f'{tf}: PBO lambda={pbo.get(\"pbo_lambda\",\"N/A\")}, DSR p={dsr.get(\"p_value\",\"N/A\")}')
      except FileNotFoundError:
          print(f'{tf}: no OOS predictions, skipping')
  "
  ```
- **Output per TF:** `validation_report_production_{tf}.json`
- **Key metrics to check:**
  - PBO lambda < 0.5 (probability of backtest overfitting)
  - Deflated Sharpe p-value < 0.05 (statistically significant after trial correction)
- **Interpretation:**
  - PBO > 0.5 = model is likely overfit. Consider reducing feature count or regularization.
  - DSR p > 0.05 = Sharpe ratio not significant given number of trials. May be luck.
- **If interrupted:** Stateless -- just re-run.
- **Resume check:** All 5 validation report files exist

### 4.2 — Review validation results
- **GATE CHECK:** If PBO > 0.5 for ANY TF, that TF should NOT go to production.
  - Action: Increase regularization (reg_lambda, min_child_weight) and re-train (Phase 2) then re-validate.
  - Do NOT proceed with a known-overfit model.
- **If all pass:** Continue to Phase 5.

---

## CLOUD PHASE 5: META-LABELING
**Location: Cloud (in-process, lightweight)**
**Estimated time: ~5 min total**
**Cost: ~$0.25**

### 5.1 — Train meta-models for all TFs
- **Command:**
  ```bash
  python -u v2_cloud_runner.py --phase 5 --tf 1d 4h 1h 15m 5m --resume --dph 2.88 \
    2>&1 | tee phase5.log
  ```
- **What it does:** Trains a logistic regression meta-classifier per TF on CPCV OOS predictions. Learns which predictions to trust (confidence, margin, entropy as meta-features).
- **Output per TF:** `meta_model_production_{tf}.pkl`
- **Key metrics:**
  - AUC > 0.55 = meta-model adds value (filters out bad trades)
  - Threshold printed = confidence cutoff for trade execution
- **Verification:**
  ```bash
  for tf in 1d 4h 1h 15m 5m; do
    if [ -f "meta_model_production_${tf}.pkl" ]; then
      python -c "
  import pickle
  with open('meta_model_production_${tf}.pkl', 'rb') as f:
      m = pickle.load(f)
  print(f'${tf}: AUC={m[\"metrics\"][\"auc\"]:.3f}, threshold={m.get(\"threshold\",\"N/A\")}')
  "
    else
      echo "MISSING: meta_model_production_${tf}.pkl"
    fi
  done
  ```
- **If interrupted:** Stateless -- re-run.
- **Resume check:** All 5 meta_model files exist

---

## CLOUD PHASE 6: LSTM TRAINING
**Location: Cloud GPU (DataParallel across all GPUs)**
**Estimated time: ~20-30 min**
**Cost: ~$1.20 at $2.88/hr**

**NOTE on H200 CPU bottleneck:** If the cloud machine has a slow CPU (<3GHz), LSTM DataLoader will bottleneck. For V2 this is less of an issue because the trainer uses DataParallel across all GPUs. But if LSTM training is extremely slow on cloud, consider training LSTMs locally on 13900K+3090 after downloading OOS predictions.

### 6.1 — Train LSTM for each TF (sequential, all GPUs per model)
- **Command:**
  ```bash
  python -u v2_cloud_runner.py --phase 6 --tf 1d 4h 1h 15m 5m --resume --dph 2.88 \
    2>&1 | tee phase6.log
  ```
- **Or manually per TF:**
  ```bash
  for tf in 1d 4h 1h 15m 5m; do
    echo "=== LSTM $tf ==="
    python -u v2_lstm_trainer.py --tf $tf --resume --alpha-search \
      --xgb-probs oos_predictions_production_${tf}.pkl \
      2>&1 | tee lstm_${tf}.log
    # GPU cleanup between TFs
    python -c "import gc; gc.collect(); import torch; torch.cuda.empty_cache(); print('cleaned')"
  done
  ```
- **What it does per TF:**
  1. Trains LSTM sequence model on feature sequences
  2. Platt calibration (maps LSTM logits to calibrated probabilities)
  3. Alpha grid search (finds optimal blend weight: p_blend = alpha*p_lstm + (1-alpha)*p_xgb)
- **Output per TF:**
  - `lstm_model_{tf}.pt` (PyTorch model weights)
  - `lstm_calibration_{tf}.pkl` (Platt scaling parameters)
  - `lstm_report_{tf}.json` (training metrics, optimal alpha)
  - Possibly `blend_config_{tf}.json` (alpha weight for XGBoost blending)
- **Time estimate:** ~5 min per TF, ~25 min total
- **Progress indicator:** Per-epoch loss/accuracy, GPU memory usage, calibration metrics
- **Checkpoint:** `lstm_checkpoint_{tf}.pt` saved per epoch. `--resume` continues from last epoch.
- **Verification:**
  ```bash
  for tf in 1d 4h 1h 15m 5m; do
    if [ -f "lstm_model_${tf}.pt" ]; then
      echo "OK: lstm_model_${tf}.pt ($(du -h lstm_model_${tf}.pt | cut -f1))"
    else
      echo "MISSING: lstm_model_${tf}.pt"
    fi
  done
  ```
- **If interrupted:** `--resume` continues from epoch checkpoint. Re-run same command.
- **Resume check:** All 5 `lstm_model_{tf}.pt` files exist

---

## CLOUD PHASE 7: BACKTESTING AUDIT
**Location: Cloud (uses optimizer configs + models)**
**Estimated time: ~5 min**
**Cost: ~$0.25**

### 7.1 — Generate full-history audit report
- **Command:**
  ```bash
  python -u backtesting_audit.py --tf 1d 4h 1h 15m 5m 2>&1 | tee phase7.log
  ```
- **What it does:** Full-history simulation (2019-2026) using trained models + optimized configs:
  - Monthly P&L breakdown
  - Weekly P&L breakdown
  - Regime performance (bull/bear/sideways/crash)
  - Trade-type breakdown (scalp/day/swing/position)
  - Named period deep dives (COVID crash, Luna collapse, FTX, etc.)
  - P&L heatmap visualization
- **Output:**
  - `audit_report.json` (machine-readable full results)
  - `audit_report.txt` (human-readable summary)
  - `audit_heatmap.html` (visual monthly P&L heatmap)
- **Verification:**
  ```bash
  python -c "
  import json
  r = json.load(open('audit_report.json'))
  print(f'Total trades: {r.get(\"total_trades\", \"N/A\")}')
  print(f'Win rate: {r.get(\"win_rate\", \"N/A\")}')
  print(f'Sharpe: {r.get(\"sharpe\", \"N/A\")}')
  print(f'Max DD: {r.get(\"max_drawdown\", \"N/A\")}')
  "
  ```
- **If interrupted:** Stateless -- re-run.
- **Resume check:** `audit_report.json` exists

---

## FULL PIPELINE COMMAND (ALL PHASES, SINGLE COMMAND)

If you want to run everything as one orchestrated pipeline with heartbeat, manifest checkpoint/resume, OOM retry, and progress reporting:

```bash
cd /workspace/v2
tmux new -s pipeline 'python -u v2_cloud_runner.py \
  --build-tf 5m \
  --tf 1d 4h 1h 15m 5m \
  --mode production \
  --engine xgboost \
  --boost-rounds 500 \
  --resume \
  --dph 2.88 \
  2>&1 | tee pipeline.log'
```

**Monitor from another tmux pane:**
```bash
# Live log
tail -f /workspace/v2/pipeline.log

# Manifest status
cat /workspace/v2/pipeline_manifest.json | python -m json.tool

# GPU utilization
watch -n 5 nvidia-smi
```

**Total estimated time:** ~2-2.5 hours
**Total estimated cost:** ~$6-8 at $2.88/hr

---

## POST-TRAINING: DOWNLOAD & VERIFY
**Location: Local**
**Estimated time: 30-60 min**

### D.1 — Download all artifacts from cloud
- **Command (from local):**
  ```bash
  # All V2 artifacts
  rsync -avz --progress root@<HOST>:/workspace/v2/model_v2_*.json \
    root@<HOST>:/workspace/v2/importance_v2_*.json \
    root@<HOST>:/workspace/v2/oos_predictions_*.pkl \
    root@<HOST>:/workspace/v2/training_report_*.json \
    root@<HOST>:/workspace/v2/exhaustive_configs_*.json \
    root@<HOST>:/workspace/v2/validation_report_*.json \
    root@<HOST>:/workspace/v2/meta_model_*.pkl \
    root@<HOST>:/workspace/v2/lstm_model_*.pt \
    root@<HOST>:/workspace/v2/lstm_calibration_*.pkl \
    root@<HOST>:/workspace/v2/lstm_report_*.json \
    root@<HOST>:/workspace/v2/blend_config_*.json \
    root@<HOST>:/workspace/v2/audit_report.* \
    root@<HOST>:/workspace/v2/pipeline_manifest.json \
    "C:/Users/C/Documents/Savage22 Server/v2/"

  # 5m parquet + npz (if built on cloud)
  rsync -avz --progress root@<HOST>:/workspace/v2/features_BTC_5m.parquet \
    root@<HOST>:/workspace/v2/v2_crosses_BTC_5m.npz \
    "C:/Users/C/Documents/Savage22 Server/v2/"
  ```
- **Time estimate:** 10-30 min depending on connection speed
- **Verification:** Count downloaded files

### D.2 — Verify model file integrity
- **Command:**
  ```bash
  cd "C:\Users\C\Documents\Savage22 Server\v2"
  python -c "
  import os, json, pickle
  artifacts = {
      'Models': ('model_v2_production_xgboost_{tf}.json', ['1d','4h','1h','15m','5m']),
      'OOS Preds': ('oos_predictions_production_{tf}.pkl', ['1d','4h','1h','15m','5m']),
      'Opt Configs': ('exhaustive_configs_{tf}.json', ['1d','4h','1h','15m','5m']),
      'Val Reports': ('validation_report_production_{tf}.json', ['1d','4h','1h','15m','5m']),
      'Meta Models': ('meta_model_production_{tf}.pkl', ['1d','4h','1h','15m','5m']),
      'LSTM Models': ('lstm_model_{tf}.pt', ['1d','4h','1h','15m','5m']),
  }
  total = ok = 0
  for cat, (pattern, tfs) in artifacts.items():
      for tf in tfs:
          total += 1
          fname = pattern.format(tf=tf)
          if os.path.exists(fname):
              sz = os.path.getsize(fname)
              print(f'  OK: {fname} ({sz/1e6:.1f} MB)')
              ok += 1
          else:
              print(f'  MISSING: {fname}')
  print(f'\n{ok}/{total} artifacts present')
  "
  ```
- **Expected:** 30/30 (6 types x 5 TFs)
- **If any missing:** Re-download from cloud before stopping instance

### D.3 — Verify model quality from audit report
- **Command:**
  ```bash
  python -c "
  import json
  r = json.load(open('audit_report.json'))
  print('=== AUDIT RESULTS ===')
  print(f'Total trades: {r.get(\"total_trades\", \"N/A\")}')
  print(f'Win rate: {r.get(\"win_rate\", \"N/A\")}')
  print(f'Sharpe: {r.get(\"sharpe\", \"N/A\")}')
  print(f'Sortino: {r.get(\"sortino\", \"N/A\")}')
  print(f'Max DD: {r.get(\"max_drawdown\", \"N/A\")}')
  print(f'Profit factor: {r.get(\"profit_factor\", \"N/A\")}')
  "
  ```
- **Minimum quality thresholds:**
  - Win rate > 50% (for 3-class model)
  - Sharpe > 1.0
  - Max drawdown < 25%
  - Total trades > 100 (statistical significance)
- **If quality is poor:** Do NOT proceed to paper trading. Investigate feature importance, retrain with different params.

### D.4 — Review PBO results
- **Command:**
  ```bash
  for tf in 1d 4h 1h 15m 5m; do
    python -c "
  import json
  r = json.load(open('validation_report_production_${tf}.json'))
  pbo = r.get('pbo', {})
  dsr = r.get('deflated_sharpe', {})
  print(f'${tf}: PBO={pbo.get(\"pbo_lambda\",\"N/A\"):.3f}, DSR_p={dsr.get(\"p_value\",\"N/A\")}')
  " 2>/dev/null || echo "${tf}: no validation report"
  done
  ```
- **GATE:** PBO < 0.5 and DSR p < 0.05 for each TF
- **If any TF fails:** That TF's model should NOT be used in production

### D.5 — Stop cloud instance
- **ONLY after all artifacts are downloaded and verified**
- **Command:**
  ```bash
  vastai stop instance <INSTANCE_ID>
  ```
- **WARNING:** Once stopped, the instance may not restart. Verify all downloads first.

---

## POST-TRAINING: PAPER TRADING ACTIVATION
**Location: Local (13900K + RTX 3090)**
**Estimated time: Ongoing (48-72 hour validation)**

### P.1 — Verify all model files are in correct locations
- **V2 models should be in:** `C:\Users\C\Documents\Savage22 Server\v2\`
- **Required files per TF (5 TFs = 30+ files):**
  - `model_v2_production_xgboost_{tf}.json`
  - `exhaustive_configs_{tf}.json`
  - `meta_model_production_{tf}.pkl`
  - `lstm_model_{tf}.pt`
  - `lstm_calibration_{tf}.pkl`
  - `blend_config_{tf}.json`

### P.2 — Start data streamers
- **Command:**
  ```bash
  cd "C:\Users\C\Documents\Savage22 Server\v2"
  python -u streamer_supervisor.py 2>&1 | tee streamers.log &
  ```
- **What it does:** Manages 7 data streamers with auto-restart, exponential backoff, freshness checks
- **Verification:** After 60 seconds, check all streamers are running:
  ```bash
  python -c "
  import sqlite3, datetime
  conn = sqlite3.connect('v2_signals.db')
  tables = [r[0] for r in conn.execute(\"SELECT name FROM sqlite_master WHERE type='table'\").fetchall()]
  for t in tables:
      try:
          last = conn.execute(f'SELECT MAX(timestamp) FROM {t}').fetchone()[0]
          print(f'{t}: last update = {last}')
      except: pass
  "
  ```

### P.3 — Start paper trader
- **Command:**
  ```bash
  cd "C:\Users\C\Documents\Savage22 Server\v2"
  python -u live_trader.py --mode paper 2>&1 | tee paper_trade.log &
  ```
- **What it does:**
  - Loads all 5 TF models + meta-models + LSTM + blend configs
  - Computes features from live data
  - Generates sparse crosses at inference time
  - Applies: XGBoost prediction -> LSTM blend -> meta-label gate -> Kelly sizing -> regime scaling
  - Logs all trades to trades.db (paper mode, no real orders)
- **Verification (after 15 min):**
  ```bash
  python -c "
  import sqlite3
  conn = sqlite3.connect('trades.db')
  trades = conn.execute('SELECT COUNT(*) FROM trades').fetchone()[0]
  print(f'Paper trades logged: {trades}')
  latest = conn.execute('SELECT * FROM trades ORDER BY timestamp DESC LIMIT 3').fetchall()
  for t in latest:
      print(t)
  "
  ```

### P.4 — Validate feature pipeline in live
- **Check AFTER first inference cycle:**
  ```bash
  # Check for NaN features (the exact bugs we fixed)
  python -c "
  # This should be run inside the paper trader or as a diagnostic
  # Verify these feature groups are non-NaN:
  print('Check paper_trade.log for these:')
  print('  - LSTM features: should show non-NaN values')
  print('  - Space weather: Kp, solar flux should be non-NaN')
  print('  - HMM features: hmm_bull_prob should be non-NaN')
  print('  - KNN features: should be non-NaN')
  print('  - Sparse crosses: should show 4-6M features computed')
  "
  ```
- **grep the log for warnings:**
  ```bash
  grep -i "nan\|missing\|error\|warning\|crash" paper_trade.log
  ```

### P.5 — Monitor for 48-72 hours
- **Metrics to track:**
  - Number of trades per TF per day
  - Win/loss ratio
  - Average trade duration
  - Feature NaN rate (should be 0% after fixes)
  - Model prediction distribution (should be roughly balanced, not all-long or all-short)
  - Memory usage (should not grow over time -- memory leak check)
- **Alerting (manual for now):**
  - Check `paper_trade.log` every 4-6 hours
  - Look for crashes, errors, stuck states
  - Verify trades are being generated across all TFs

### P.6 — Run unified backtest for comparison
- **After 48 hours of paper trading, compare:**
  ```bash
  python -u backtesting_audit.py --unified 2>&1 | tee backtest_comparison.log
  ```
- **Compare paper trade results vs historical backtest**
- **If paper trading is significantly worse than backtest:** Investigate feature drift, data freshness issues

---

## V1 PIPELINE (OPTIONAL, AFTER V2)
**Run this if V2 succeeds and there is remaining cloud budget/time.**

The V1 pipeline uses the same cloud instance. It's simpler (BTC-only, 6 TFs).

### V1.1 — Build all 6 V1 features on cloud
```bash
cd /workspace
tmux new -s v1 'python -u cloud_runner.py 2>&1 | tee v1_pipeline.log'
```
- **Phases:** 1w+1d+4h parallel -> 1h -> 15m -> 5m sequential
- **Time:** ~60 min build + train
- **Output:** Same model/config/meta/LSTM files but in `/workspace/` (not v2/)

### V1.2 — Download V1 artifacts
```bash
rsync -avz --progress root@<HOST>:/workspace/model_*.json \
  root@<HOST>:/workspace/exhaustive_configs*.json \
  root@<HOST>:/workspace/cpcv_*.pkl \
  root@<HOST>:/workspace/meta_model_*.pkl \
  root@<HOST>:/workspace/platt_*.pkl \
  root@<HOST>:/workspace/validation_report_*.json \
  root@<HOST>:/workspace/features_*.parquet \
  "C:/Users/C/Documents/Savage22 Server/"
```

---

## INTERRUPTION HANDLING SUMMARY

| Phase | Checkpoint Mechanism | Resume Command | What's Lost if Killed |
|-------|---------------------|----------------|----------------------|
| 0 (bugs) | None (file edits) | Re-edit files | Nothing (git tracks) |
| 0B (builds) | Per-asset parquet | Re-run same command | Current asset's build (~15 min) |
| 1 (cloud build) | Per-asset parquet + manifest | `--resume` | Current asset's build |
| 2 (training) | Per-fold checkpoint + manifest | `--resume` | Current fold (~2 min) |
| 3 (optimizer) | 5% batch checkpoint + manifest | `--resume` | Current 5% batch (~30 sec) |
| 4 (validation) | Per-TF report file + manifest | `--resume` | Current TF validation (~30 sec) |
| 5 (meta) | Per-TF model file + manifest | `--resume` | Current TF meta (~30 sec) |
| 6 (LSTM) | Per-epoch checkpoint + manifest | `--resume` | Current epoch (~1 min) |
| 7 (audit) | Report file + manifest | `--resume` | Audit run (~2 min) |

---

## COST ESTIMATES

| Phase | Time | Cost at $2.88/hr |
|-------|------|-----------------|
| Phase 1 (5m build only) | 35 min | $1.68 |
| Phase 2 (training) | 50 min | $2.40 |
| Phase 3 (optimizer) | 35 min | $1.68 |
| Phase 4 (validation) | 5 min | $0.24 |
| Phase 5 (meta) | 5 min | $0.24 |
| Phase 6 (LSTM) | 25 min | $1.20 |
| Phase 7 (audit) | 5 min | $0.24 |
| **TOTAL** | **~2.5 hours** | **~$7.68** |

Add ~$1-2 buffer for upload/download time and idle periods.
**Total budget needed: ~$10-12 on vast.ai**
**Current balance: $11.93 -- should be sufficient if machine is $2.88/hr**

---

## PROGRESS TRACKING

Create this file on cloud at pipeline start:
```bash
cat > /workspace/v2/progress.sh << 'PROGRESS'
#!/bin/bash
echo "=== PIPELINE PROGRESS ==="
echo "Manifest:"
cat pipeline_manifest.json 2>/dev/null | python -m json.tool | grep -E '"status"|"phase_'
echo ""
echo "Files:"
echo "  Parquets: $(ls features_*.parquet 2>/dev/null | wc -l)"
echo "  NPZ: $(ls v2_crosses_*.npz 2>/dev/null | wc -l)"
echo "  Models: $(ls model_v2_*.json 2>/dev/null | wc -l)"
echo "  OOS: $(ls oos_predictions_*.pkl 2>/dev/null | wc -l)"
echo "  Configs: $(ls exhaustive_configs_*.json 2>/dev/null | wc -l)"
echo "  Val Reports: $(ls validation_report_*.json 2>/dev/null | wc -l)"
echo "  Meta: $(ls meta_model_*.pkl 2>/dev/null | wc -l)"
echo "  LSTM: $(ls lstm_model_*.pt 2>/dev/null | wc -l)"
echo "  Audit: $(ls audit_report.* 2>/dev/null | wc -l)"
echo ""
echo "GPU:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null
echo ""
echo "RAM:"
free -h | head -2
echo ""
echo "Cost:"
HOURS=$(awk "BEGIN {printf \"%.2f\", $(cat /proc/uptime | awk '{print $1}')/3600}")
echo "  Uptime: ${HOURS}h, Cost: \$$(awk "BEGIN {printf \"%.2f\", $HOURS * 2.88}")"
PROGRESS
chmod +x /workspace/v2/progress.sh
```

Run anytime: `bash /workspace/v2/progress.sh`

---

## DECISION TREE: WHAT TO DO WHEN THINGS GO WRONG

```
Problem: OOM during feature build
  -> Is it GPU OOM?
     -> YES: Set V2_RIGHT_CHUNK=250 (or 100) and retry
     -> NO (RAM OOM):
        -> Is it 5m?
           -> YES: Need 250GB+ RAM. Build on cloud only.
           -> NO: Reduce --parallel to 1 and retry
        -> Still OOM? Kill other processes, gc.collect(), retry

Problem: OOM during XGBoost training
  -> Reduce V2_BATCH_SIZE env var (halve it)
  -> v2_cloud_runner.py does this automatically on OOM

Problem: Training accuracy very low (<40%)
  -> Check feature count: "import json; print(len(json.load(open('importance_v2_production_1d.json'))))"
  -> If 0 features: data loading failed silently. Check parquet column names.
  -> If features exist but low accuracy: check label distribution (imbalanced classes?)

Problem: PBO > 0.5 (overfit)
  -> Increase reg_lambda by 2x
  -> Increase min_child_weight by 2x
  -> Reduce max_depth by 1
  -> Re-train (Phase 2) and re-validate (Phase 4)

Problem: LSTM won't converge
  -> Check input normalization (means/stds should be finite)
  -> Reduce learning rate by 10x
  -> Increase sequence length

Problem: Meta-model AUC < 0.5
  -> Meta-model is worse than random. Do NOT use it.
  -> Set meta-label gate threshold to 0 (pass all trades through)

Problem: Cloud instance won't start
  -> vast.ai instances can become unavailable after stop
  -> Search for a new machine with same specs
  -> All progress is in pipeline_manifest.json -- upload to new instance and --resume

Problem: Upload too slow
  -> Use scp instead of rsync (sometimes faster)
  -> Compress large files: tar czf v2_data.tar.gz features_*.parquet v2_crosses_*.npz
  -> Upload compressed, extract on cloud

Problem: Paper trader crashes
  -> Check paper_trade.log for stack trace
  -> Most common: feature name mismatch between train and live
  -> Fix: ensure live_trader.py uses exact same feature_library.py as training
```

---

## QUICK REFERENCE: SINGLE-LINE COMMANDS

```bash
# Pre-flight (local)
python -u smoke_test_pipeline.py                                          # smoke test
vastai show user                                                          # check balance

# Cloud deploy
vastai create instance <ID> --image rapidsai/base:25.02-cuda12.5-py3.12 --disk 100
rsync -avz --progress v2/ root@<HOST>:/workspace/v2/                     # upload scripts
rsync -avz --progress *.db root@<HOST>:/workspace/                       # upload DBs

# Full pipeline (cloud)
python -u v2_cloud_runner.py --tf 1d 4h 1h 15m 5m --mode production --engine xgboost --boost-rounds 500 --resume --dph 2.88 --build-tf 5m

# Monitor (cloud)
tail -f pipeline.log
bash progress.sh

# Download (local)
rsync -avz root@<HOST>:/workspace/v2/{model_v2_*,exhaustive_configs_*,meta_model_*,lstm_model_*,validation_report_*,audit_report*,blend_config_*,lstm_calibration_*} v2/

# Paper trade (local)
python -u streamer_supervisor.py &
python -u live_trader.py --mode paper
```
