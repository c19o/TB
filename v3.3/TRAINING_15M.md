# 15M Training Guide — V3.3
# **READY — All blockers resolved.** Int64 indptr fix applied (LightGBM PR #1719).
# Row-partitioned boosting REJECTED (Perplexity-confirmed: kills rare signals).
# THE HARDEST TIMEFRAME. Read every section. Another Claude session will use this.

## Machine Requirements
- **RAM:** 2TB+ **cgroup** (NOT host RAM — containers cap lower)
- **Verify cgroup:** `cat /sys/fs/cgroup/memory/memory.limit_in_bytes` — must show >= 2,147,483,648,000 (2TB)
  - Host may report 2TB but container caps at 1.33TB. **Always check cgroup, not free -g.**
  - If cgroup shows `9223372036854775807` (max int64), there's no cgroup limit — use `free -g` instead.
- **Cores:** 256+ (cross gen is multi-threaded, parallel CPCV needs cores)
- **CPU Score:** 500+ (cores x base GHz)
- **Disk:** 150GB+ free
- **USER PICKS THIS MACHINE** from vast.ai lineup. Never auto-select.

---

## Required Databases (ALL 16 — ZERO MISSING)

**From project root (-> /workspace/):**
```
btc_prices.db          # 1.3GB — BTC OHLCV 2010-2026
tweets.db              # tweet text + gematria
news_articles.db       # news headlines
astrology_full.db      # planetary positions
ephemeris_cache.db     # ephemeris calculations
fear_greed.db          # fear/greed index
sports_results.db      # sports event correlations
space_weather.db       # Kp index, solar flux
macro_data.db          # macro indicators
onchain_data.db        # on-chain metrics
funding_rates.db       # funding rates
open_interest.db       # open interest
google_trends.db       # google trends
llm_cache.db           # LLM feature cache
```

**From v3.3/ (-> /workspace/v3.3/):**
```
multi_asset_prices.db  # 1.3GB — multi-asset data
v2_signals.db          # DeFi TVL, BTC dominance, mining stats
```

**Also required (non-DB):**
```
kp_history_gfz.txt     # historical Kp index data
astrology_engine.py     # must be in v3.3/ directory (feature_library.py imports it)
```

### Verify Script (run BEFORE launching — ALL must say OK)
```bash
echo "=== DB Verification ==="
FAIL=0
for db in tweets.db news_articles.db astrology_full.db ephemeris_cache.db \
  fear_greed.db sports_results.db space_weather.db macro_data.db \
  onchain_data.db funding_rates.db open_interest.db google_trends.db \
  llm_cache.db btc_prices.db; do
  if [ -f /workspace/$db ] || [ -f /workspace/v3.3/$db ]; then
    echo "OK   $db"
  else
    echo "MISS $db"; FAIL=1
  fi
done
for db in multi_asset_prices.db v2_signals.db; do
  if [ -f /workspace/v3.3/$db ]; then
    echo "OK   v3.3/$db"
  else
    echo "MISS v3.3/$db"; FAIL=1
  fi
done
# Non-DB files
[ -f /workspace/v3.3/kp_history_gfz.txt ] || [ -f /workspace/kp_history_gfz.txt ] && echo "OK   kp_history_gfz.txt" || { echo "MISS kp_history_gfz.txt"; FAIL=1; }
[ -f /workspace/v3.3/astrology_engine.py ] && echo "OK   astrology_engine.py" || { echo "MISS astrology_engine.py"; FAIL=1; }
echo ""
if [ $FAIL -eq 1 ]; then echo "STOP: Missing files. Upload before launching."; else echo "ALL OK — safe to launch."; fi
```
**If ANY says MISS -> STOP. Upload the missing file first. Missing DB = broken matrix = invalid model.**

---

## Nuclear Clean (delete ALL old artifacts before fresh run)
```bash
cd /workspace && rm -f *.npz *.json *.pkl *.parquet *.log DONE_* RUNNING_* *.lock 2>/dev/null  # NOTE: no *.txt — would kill kp_history_gfz.txt
cd /workspace/v3.3 && rm -f v2_crosses_*.npz v2_cross_names_*.json v2_base_*.parquet \
  features_BTC_*.parquet features_*_all.json model_*.json platt_*.pkl cpcv_oos_*.pkl \
  feature_importance_*.json shap_analysis_*.json validation_report_*.json meta_model_*.pkl \
  ml_multi_tf_*.* optuna_configs_all.json lstm_*.pt DONE_* RUNNING_* *.lock 2>/dev/null
echo '{"steps": {}, "version": "3.3"}' > pipeline_manifest.json
echo 'Nuclear clean done'
```
**Old NPZs built with min_nonzero=8 produce fewer features. Old cross names JSON truncates column count. DELETE BOTH.**

---

## Verify Parquet Freshness
```bash
python3 -c "
import pandas as pd, os
p = '/workspace/v3.3/features_BTC_15m.parquet'
if os.path.exists(p):
    df = pd.read_parquet(p)
    print(f'Parquet: {len(df)} rows x {len(df.columns)} cols')
    print(f'Modified: {pd.Timestamp(os.path.getmtime(p), unit=\"s\")}')
else:
    print('No parquet found — will be built fresh')
"
```

---

## Data Profile
- **Rows:** ~293,980 (15m bars, 2017-11-01 to 2026). **Was 227,577 — started from 2019 instead of 2017.**
- **Base features:** ~4,000+ cols after feature_library.py
- **Cross features:** ~10M+ expected (min_nonzero=3)
- **No pre-built NPZ** — full cross gen required from scratch
- **NNZ estimate:** ~29B+ (293,980 rows x 10M cols x ~1% density) — **massively exceeds int32 limit**
- **int64 indptr is the PRIMARY fix. Row-partitioned boosting is backup for extreme cases only.**
- **Data range:** Binance BTC/USDT 15m from 2017-11-01 (download_btc.py via Binance global API)

---

## CRITICAL: NNZ int32 Overflow Problem

LightGBM's sparse CSR uses int32 for `indptr` and `indices` arrays. Maximum NNZ = 2^31 - 1 = 2,147,483,647.

15m at 293,980 rows x 10M+ features will have NNZ far exceeding this limit.

### How ml_multi_tf.py handles it (v3.3 — FIXED):
`_ensure_lgbm_sparse_dtypes()` is applied after NPZ load AND after hstack:
1. `indptr` cast to int64 — handles cumulative NNZ values > 2^31
2. `indices` cast to int32 — column IDs (max ~10M, always fits int32)
3. LightGBM C API accepts int64 indptr since PR #1719 (2018)
4. If NNZ > int32 max: logs info, forces sequential CPCV, training proceeds normally
5. Dense conversion will correctly skip (dense is 8.5TB, won't fit) → stays sparse
6. `_predict_chunked()` handles IS predictions on large train sets

### NO row-partitioned boosting (Perplexity-confirmed UNSAFE):
- Row partitioning kills rare signals (dilutes occurrence counts below min_data_in_leaf)
- Instead: full matrix, single training pass, int64 indptr for overflow safety

---

## NNZ Overflow Handling (RESOLVED via int64 indptr)

### Primary fix: int64 indptr
`_ensure_lgbm_sparse_dtypes()` casts `indptr` to int64, which handles cumulative NNZ values > 2^31. LightGBM C API accepts int64 indptr since PR #1719 (2018). This is the PRIMARY fix applied to all TFs.

### Row-partitioned boosting: BACKUP ONLY for extreme cases
Row-partitioned `init_model` continuation is a backup for extreme NNZ cases where int64 indptr alone is insufficient. **WARNING:** Row partitioning dilutes rare signal occurrence counts below min_data_in_leaf, killing esoteric signals. Perplexity confirmed this risk. Use ONLY if int64 indptr fails at runtime.

### Risk if int64 indptr not applied:
LightGBM trains on corrupted int32-overflowed sparse matrix. Model looks trained but predictions are garbage. There is NO error message — it is completely silent.

---

## Dense vs Sparse on 15m

At 293,980 rows x 10M features:
- **Dense:** 293,980 x 10,000,000 x 4 bytes = **~11 TB** -> IMPOSSIBLE. No machine has this.
- **Sparse:** Stays as CSR. `is_enable_sparse=True` in LightGBM params (set automatically by ml_multi_tf.py since dense conversion fails).
- **Consequence:** LightGBM trains on sparse CSR. This is SLOWER than dense (sparse serializes OpenMP histogram building). But it's the only option.
- **EFB still works on sparse:** LightGBM's Exclusive Feature Bundling handles sparse binary features optimally. max_bin=15 keeps it fast.

---

## Environment Variables (15m-specific)
```bash
export V2_RIGHT_CHUNK=200     # Memory-safe chunking for cross gen (RC=500 OOM'd at 1892G on 2TB machine)
export V2_BATCH_MAX=500       # Cap dense intermediate arrays in cross gen
## OMP_NUM_THREADS / NUMBA_NUM_THREADS — set dynamically by cloud_run_tf.py per phase
```

### Why V2_RIGHT_CHUNK=200:
RC=500 OOM'd at 1892G on 2TB machine (293,980 rows). RC=200 peaked at 574G (29% usage on 2TB).
RC=300 estimated ~1100-1300G peak — possible but tight on 2TB, no margin for error. **RC=200 is the safe choice.**

### Why V2_BATCH_MAX=500:
Caps the number of feature pairs processed per batch in the parallel cross multiply. Each worker holds arrays of (N x BATCH x 4 bytes). At 293,980 rows x 500 pairs x 4 bytes = ~560MB per worker.

### Cross gen thread cap:
v2_cross_generator.py caps at 128 threads (line 654: `n_threads = min(_ram_limited, n_cpus, 128)`). RAM-limited: each worker's memory footprint is estimated and total workers capped to fit available RAM.

### min_nonzero=3:
Default in v2_cross_generator.py line 179: `MIN_CO_OCCURRENCE = 3`. This is correct — matches min_data_in_leaf=3. Preserves rare esoteric crosses. Can be overridden via env: `V2_MIN_CO_OCCURRENCE=3`.

### min_data_in_leaf=15:
Per `TF_MIN_DATA_IN_LEAF` in config.py — higher than other TFs due to more rows. 15m has ~294K rows vs 5,727 for 1d, so rare signals still fire 100+ times even with leaf=15.

---

## Pipeline Steps with Estimated Times

| Step | What | Output | Est. Time | Notes |
|------|------|--------|-----------|-------|
| 1 | Feature build | features_BTC_15m.parquet | ~2-3 min | CPU pandas (no cuDF on cloud). ~4000 cols. |
| 2 | Cross gen | v2_crosses_BTC_15m.npz | **3-8 hrs** | 294K rows, ~10M+ crosses. Bottleneck step. |
| 3 | LightGBM CPCV | model_15m.json | **10 hrs** | 4 folds × 150 min. Sparse CSR + int64 indptr. Sequential CPCV. |
| 4 | Optuna | optuna_configs_15m.json | **4-8 hrs** | 200 TPE trials, each on sparse CSR. |
| 5 | Meta-labeling | meta_model_15m.pkl | ~30 min | Logistic regression on CPCV OOS predictions. |
| 6 | LSTM | lstm_15m.pt + platt_15m.pkl | **Run locally** | 13900K + RTX 3090. Cloud H200 has weak CPU for DataLoader. |
| 7 | PBO/Audit | validation_report_15m.json | ~10 min | PBO on CPCV OOS equity curves. |
| **Total** | | | **13-19 hrs** | No Optuna. 4 folds (4,1). Production model identical regardless of fold count. See FOLD_STRATEGY.md. |

---

## Launch Command
```bash
cd /workspace/v3.3 && \
  export SAVAGE22_DB_DIR=/workspace && \
  export V30_DATA_DIR=/workspace/v3.3 && \
  export PYTHONUNBUFFERED=1 && \
  export V2_RIGHT_CHUNK=200 && \
  export V2_BATCH_MAX=500 && \
  nohup python -u cloud_run_tf.py --symbol BTC --tf 15m > /workspace/15m_log.txt 2>&1 &
```

---

## Memory Monitoring During Cross Gen

Cross gen is the most RAM-intensive step. Monitor continuously.

### Watch command (run in separate SSH session):
```bash
watch -n 10 'echo "=== RAM ===" && free -g && echo "" && echo "=== Process ===" && ps aux | grep cloud_run_tf | grep -v grep && echo "" && echo "=== Cgroup ===" && cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null | awk "{printf \"Cgroup used: %.1f GB\n\", \$1/1073741824}"'
```

### What to watch for:
- RAM usage climbing past 80% of cgroup limit -> approaching OOM kill
- If OOM killed, process just disappears — check `dmesg | tail -20` for "oom-kill"
- Cross gen logs batch progress: `Parallel cross: N pairs, M batches, T threads`
- If single-threaded (load avg ~1.0 on 256-core machine), something is wrong

---

## Download Partial Results After Each Step (MACHINES DIE)

vast.ai machines die without warning. Download after EVERY critical step.

### After feature build (Step 1):
```bash
scp -i ~/.ssh/vast_key -P {PORT} root@{HOST}:/workspace/v3.3/features_BTC_15m.parquet ./v3.3/
```

### After cross gen (Step 2) — THIS IS THE BIG ONE:
```bash
scp -i ~/.ssh/vast_key -P {PORT} root@{HOST}:/workspace/v3.3/v2_crosses_BTC_15m.npz ./v3.3/
scp -i ~/.ssh/vast_key -P {PORT} root@{HOST}:/workspace/v3.3/v2_cross_names_BTC_15m.json ./v3.3/
scp -i ~/.ssh/vast_key -P {PORT} root@{HOST}:/workspace/v3.3/v2_base_BTC_15m.parquet ./v3.3/
```
**The NPZ can be 10-50GB. Start the download as soon as cross gen finishes. If the machine dies during training, you only lose training — cross gen is preserved.**

### After training (Step 3):
```bash
scp -i ~/.ssh/vast_key -P {PORT} root@{HOST}:/workspace/v3.3/model_15m.json ./v3.3/
scp -i ~/.ssh/vast_key -P {PORT} root@{HOST}:/workspace/v3.3/cpcv_oos_15m.pkl ./v3.3/
```

### After Optuna (Step 4):
```bash
scp -i ~/.ssh/vast_key -P {PORT} root@{HOST}:/workspace/v3.3/optuna_configs_all.json ./v3.3/
```

### After meta-labeling (Step 5):
```bash
scp -i ~/.ssh/vast_key -P {PORT} root@{HOST}:/workspace/v3.3/meta_model_15m.pkl ./v3.3/
```

### After PBO (Step 7):
```bash
scp -i ~/.ssh/vast_key -P {PORT} root@{HOST}:/workspace/v3.3/validation_report_15m.json ./v3.3/
```

### Download everything at once (if machine is still alive at the end):
```bash
ssh -i ~/.ssh/vast_key -p {PORT} root@{HOST} "cd /workspace/v3.3 && tar czf /workspace/15m_results.tar.gz model_15m.json optuna_configs_all.json meta_model_15m.pkl cpcv_oos_15m.pkl validation_report_15m.json v2_crosses_BTC_15m.npz v2_cross_names_BTC_15m.json features_BTC_15m.parquet feature_importance_*.json 2>/dev/null"
scp -i ~/.ssh/vast_key -P {PORT} root@{HOST}:/workspace/15m_results.tar.gz ./v3.3/
```

---

## v3.2 Baseline
**NONE.** 15m has never completed training in any previous version. The Texas machine session ended before 15m was attempted.

Expected v3.3 targets (based on other TF patterns):
- Accuracy: 55-58% (more noise at 15m, more rows help)
- PBO: DEPLOY
- This is the first-ever 15m model. Any validated result is a milestone.

---

## Verification After Launch (first 30 lines)

```bash
sleep 30 && head -30 /workspace/15m_log.txt
```

Must see:
- "All 16 databases present" or zero "MISS" lines
- Row count: ~293,980
- Feature count: ~4,000+ base cols
- Correct paths: DB_DIR=/workspace, V30_DATA_DIR=/workspace/v3.3
- No "WARNING: DB missing" for any esoteric DB

---

## BLOCKERS — ALL RESOLVED

### BLOCKER 1: Row-Partitioned Incremental Boosting — **REJECTED**
- Perplexity confirmed: row-partitioned `init_model` continuation KILLS rare signals
- 294K rows / 13 chunks = ~22K rows/chunk → features firing 15 times globally → ~1.1 per chunk
- Below min_data_in_leaf=15 → LightGBM NEVER splits on rare esoteric signals
- **NEVER use row-partitioned boosting. It violates the matrix thesis.**

### BLOCKER 2: NNZ Overflow Detection — **RESOLVED**
- `_ensure_lgbm_sparse_dtypes()` enforces: indptr=int64 (handles NNZ > 2^31), indices=int32 (column IDs)
- LightGBM PR #1719 (2018) fixed int64 indptr support in C API
- If NNZ > int32 max: logs warning, forces sequential CPCV, training proceeds normally
- Crash-loud assertion if column indices exceed int32 (would mean >2B features)

### BLOCKER 3: scipy int64 — **RESOLVED**
- `_ensure_lgbm_sparse_dtypes()` applied after NPZ load AND after hstack on ALL TFs
- indptr always int64 (row pointers, values can exceed int32)
- indices always int32 (column IDs, values < 10M, fits int32)
- No conditional logic, one code path for all TFs

---

## LAUNCH CHECKLIST:
1. Machine with 2TB+ cgroup rented (user picks — high CPU score, <$2/hr)
2. All 16 DBs verified present
3. Nuclear clean run
4. Cross gen completes (~3-8 hrs) and NPZ is downloaded as backup
5. First CPCV fold validates int64 indptr works (crash-loud if not)

---

## STATUS
**READY** — (4,1)=4 folds, no Optuna. Est. 13-19 hrs, ~$23-33. Single machine.
