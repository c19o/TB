# 4H Training Guide

## Machine Requirements
- **RAM:** 1.5-2TB MINIMUM (1TB OOM'd, 2TB worked at RC=500 — peaked 1213GB)
- **Cores:** 128+ (parallel CPCV + cross gen)
- **CPU Score:** 400+
- **Disk:** 50GB+
- **RIGHT_CHUNK:** `export V2_RIGHT_CHUNK=500` (MANDATORY — auto=2000 OOM'd on 1TB. RC=500 peaked 1213GB on 2TB machine)

## Required Databases (ALL 16 — ZERO MISSING)
```
FROM PROJECT ROOT (-> /workspace/):
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

FROM v3.3/ (-> /workspace/v3.3/):
  multi_asset_prices.db  # 1.3GB — multi-asset data
  v2_signals.db          # DeFi TVL, BTC dominance, mining stats
```
**Also required:** kp_history_gfz.txt, astrology_engine.py (both in v3.3/)

### DB Verification Script (RUN BEFORE LAUNCH — ALL must say OK)
```bash
FAIL=0
for db in btc_prices.db tweets.db news_articles.db sports_results.db space_weather.db onchain_data.db macro_data.db astrology_full.db ephemeris_cache.db fear_greed.db funding_rates.db google_trends.db open_interest.db llm_cache.db; do
  if [ -f /workspace/$db ] || [ -f /workspace/v3.3/$db ]; then echo "OK   $db"; else echo "MISSING: $db"; FAIL=$((FAIL+1)); fi
done
for db in multi_asset_prices.db v2_signals.db; do
  if [ -f /workspace/v3.3/$db ]; then echo "OK   v3.3/$db"; else echo "MISSING: v3.3/$db"; FAIL=$((FAIL+1)); fi
done
[ -f /workspace/kp_history_gfz.txt ] || [ -f /workspace/v3.3/kp_history_gfz.txt ] && echo "OK   kp_history_gfz.txt" || { echo "MISSING: kp_history_gfz.txt"; FAIL=$((FAIL+1)); }
[ -f /workspace/v3.3/astrology_engine.py ] && echo "OK   astrology_engine.py" || { echo "MISSING: astrology_engine.py"; FAIL=$((FAIL+1)); }
echo ""
echo "DB check: $FAIL missing"
[ $FAIL -eq 0 ] && echo "ALL 16 DBs + kp_history + astrology_engine PRESENT" || { echo "STOP — fix missing files before proceeding"; exit 1; }
```
**If ANY says MISSING -> STOP. Do not launch. Upload the missing file first.**

## Data
- **Rows:** ~17,520 (4h bars, 2017-08-17 to 2026). **Was 4,380 — only had 2024+ data from Binance.US.**
- **Base features:** ~3,904 cols
- **Cross features:** ~3-4M expected (min_nonzero=3, no pre-built NPZ)
- **Full cross gen required** (no pre-existing NPZ — build from scratch)
- **Dense matrix:** Inconsistent with observed 1213GB peak RAM — actual memory usage dominated by cross gen intermediates, not dense training matrix.
- **Context signals:** 4269 total (astro:333, ta:2576, esoteric:909, macro:123, regime:27, session:4, aspect:38, price_num:4, moon:23, volatility:232)
- **Data range:** Binance BTC/USDT 4h from 2017-08-17 (download_btc.py via Binance global API)

## Nuclear Clean (MANDATORY — delete ALL old artifacts before launch)
Must delete BOTH v2_crosses*.npz AND v2_cross_names*.json — old NPZs were built with min_nonzero=8, old cross names JSONs truncate column count.
```bash
cd /workspace && rm -f *.npz *.json *.pkl *.parquet *.log DONE_* RUNNING_* *.lock 2>/dev/null  # NOTE: no *.txt — would kill kp_history_gfz.txt
cd /workspace/v3.3 && rm -f v2_crosses_*.npz v2_cross_names_*.json v2_base_*.parquet \
  features_BTC_*.parquet features_*_all.json model_*.json platt_*.pkl cpcv_oos_*.pkl \
  feature_importance_*.json shap_analysis_*.json validation_report_*.json meta_model_*.pkl \
  ml_multi_tf_*.* optuna_configs_all.json lstm_*.pt DONE_* RUNNING_* *.lock 2>/dev/null
echo '{"steps": {}, "version": "3.3"}' > pipeline_manifest.json
echo 'Clean done'
```

## Verify Parquet Freshness
```bash
python3 -c "import pandas as pd; df=pd.read_parquet('/workspace/v3.3/features_BTC_4h.parquet'); print(f'Cols: {len(df.columns)}, Rows: {len(df)}')"
# Expected: columns should match feature_library.py output count
```

## Pipeline Steps
1. Feature build (~35s) -> features_BTC_4h.parquet
2. Cross gen (Step 5/13 alone took 9.5+ hours. Total estimated 18+ hours) -> v2_crosses_BTC_4h.npz + v2_cross_names_BTC_4h.json
3. SPARSE + SEQUENTIAL CPCV (parallel disabled for >1M features — pickle bottleneck)
4. LightGBM CPCV (4 folds sequential, (4,1) config) -> model_4h.json
5. Optuna (200 trials) -> optuna_configs_4h.json
6. Meta-labeling -> meta_model_4h.pkl
7. LSTM -> lstm_4h.pt + platt_4h.pkl
8. PBO/Audit -> validation_report_4h.json

## CRITICAL LESSON: is_enable_sparse=False
When data is converted from sparse to dense (Step 3), LightGBM MUST have `is_enable_sparse=False` set. Without this, LightGBM uses single-threaded sparse histogram construction even on dense data, wasting all but 1 core. The code in ml_multi_tf.py handles this automatically (line 780), but verify in logs:
```
is_enable_sparse=False (data converted to dense -- enables multi-core LightGBM)
```
If you do NOT see this line, training is single-threaded. STOP and investigate.

## Install Dependencies
```bash
pip install lightgbm scikit-learn scipy ephem astropy pytz joblib pandas numpy pyarrow optuna hmmlearn numba tqdm pyyaml psutil
python -c "import pandas, numpy, scipy, sklearn, lightgbm, ephem, astropy, pyarrow, optuna, numba, hmmlearn, yaml, tqdm; print('ALL OK')"
```

## Launch Command
```bash
cd /workspace/v3.3 && \
  export SAVAGE22_DB_DIR=/workspace && \
  export V30_DATA_DIR=/workspace/v3.3 && \
  export PYTHONUNBUFFERED=1 && \
  nohup python -u cloud_run_tf.py --symbol BTC --tf 4h > /workspace/4h_log.txt 2>&1 &
```

## Verify First 30 Lines (wait ~30s after launch)
```bash
head -30 /workspace/4h_log.txt
```
Must see:
- "All 16 databases present" (or zero "MISS" / "WARNING: DB missing")
- Correct row count: ~17,520
- Correct base feature count: ~3,904
- V30_DATA_DIR pointing to /workspace/v3.3 (NOT v3.0)

## Verify Multi-Threaded Execution
After training starts (Step 4 — CPCV), check load average:
```bash
# Load average should be >> 1.0 on 128+ core machine
uptime
# Expected: load average > 30 (LightGBM threads (sequential CPCV))
# If load average ~ 1.0, training is SINGLE-THREADED — this is a critical bug

# Also check RSS
ps aux | grep cloud_run_tf | grep -v grep
```

## Verify Crosses Not Truncated
After cross gen completes (Step 2), check the cross feature count:
```bash
python3 -c "
import scipy.sparse as sp
X = sp.load_npz('/workspace/v3.3/v2_crosses_BTC_4h.npz')
print(f'Cross shape: {X.shape}')
print(f'Expected: ~3-4M columns (min_nonzero=3)')
print(f'NNZ: {X.nnz:,}')
import json
with open('/workspace/v3.3/v2_cross_names_BTC_4h.json') as f:
    names = json.load(f)
print(f'Cross names: {len(names)}')
assert X.shape[1] == len(names), 'MISMATCH: NPZ cols != JSON names count'
print('OK — NPZ and JSON match')
"
```
If cross count is < 2M, something is wrong (min_nonzero may not be 3). Check logs.

## Monitor Commands
```bash
# Live log tail
tail -f /workspace/4h_log.txt

# Check process is alive
ps aux | grep cloud_run_tf | grep -v grep

# Check memory usage
free -g

# Check disk space
df -h /workspace

# Check for OOM kills:
dmesg | tail -20 | grep -i oom
```

## v3.2 Baseline (targets to beat)
| Metric | v3.2 Value |
|--------|-----------|
| Meta AUC | 0.616 |
| PBO | REJECT (17,520 rows) |

## v3.3 Expected (min_nonzero=3)
| Metric | Expected |
|--------|----------|
| Cross features | ~3-4M (was unknown at min_nonzero=8) |
| CPCV mode | SPARSE + SEQUENTIAL |
| Training time | 18+ hrs cross gen + training |

## Download Results When Done
```bash
scp -P PORT root@HOST:/workspace/v3.3/model_4h.json .
scp -P PORT root@HOST:/workspace/v3.3/optuna_configs_4h.json .
scp -P PORT root@HOST:/workspace/v3.3/features_4h_all.json .
scp -P PORT root@HOST:/workspace/v3.3/ml_multi_tf_results.txt .
scp -P PORT root@HOST:/workspace/v3.3/ml_multi_tf_configs.json .
```

## Notes
- 4h has ~17,520 rows (2017-08-17 to 2026, Binance global API)
- 4 CPCV paths (N=4, K=1) — production model identical regardless of fold count (trains on ALL data). See FOLD_STRATEGY.md.
- min_data_in_leaf=5 for 4h (per TF_MIN_DATA_IN_LEAF in config.py)
- Full cross gen from scratch — no pre-built NPZ exists for 4h
- PBO result with 17,520 rows should be more reliable than with old 4,380 rows

## STATUS
Cross gen step 5/13 when machine destroyed. Only parquet saved.
