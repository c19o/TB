# 1H Training Guide

## Machine Requirements
- **RAM:** 2TB+ MINIMUM (cross gen peaks at ~1.5TB with RC=500)
- **Cores:** 256+ (parallel CPCV + cross gen)
- **CPU Score:** 500+
- **Disk:** 80GB+
- **RIGHT_CHUNK:** `export V2_RIGHT_CHUNK=500` (MANDATORY — auto=2000 OOMs)
- **Training stays sparse** — dense would be ~2.3TB, won't fit. int64 indptr handles NNZ > 2^31.

## CRITICAL: OLD MACHINE DESTROYED — DO NOT USE 33598910
The 504GB machine subsampled from 17,520 to 13,243 rows (lost 24% of data). Subsampling code has been REMOVED from ml_multi_tf.py. If dense doesn't fit, training keeps sparse (slower but no data loss). **Rent 768GB+ machine to get dense speed with full matrix.**

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
- **Rows:** ~75,405 (1h bars, 2017-08-17 to 2026). **Was 17,520 — only had 2024+ data from Binance.US.**
- **Base features:** ~3,968 cols
- **Cross features:** ~7-8M expected (min_nonzero=3, up from 6.06M at min_nonzero=8)
- **NPZ from v3.3 cloud_results:** STALE (built with min_nonzero=8). Must regenerate.
- **Dense matrix:** ~2.3TB (73K rows x 8M features x 4 bytes). **Row-partitioned boosting likely needed (NNZ will exceed int32).**
- **NNZ estimate:** ~8B+ (73K rows × 8M cols × ~1.5% density) — **exceeds int32 limit, needs row-partitioned training**
- **Data range:** Binance BTC/USDT 1h from 2017-08-17 (download_btc.py via Binance global API)

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
python3 -c "import pandas as pd; df=pd.read_parquet('/workspace/v3.3/features_BTC_1h.parquet'); print(f'Cols: {len(df.columns)}, Rows: {len(df)}')"
# Expected: columns should match feature_library.py output count
```

## Pipeline Steps
1. Feature build (~60s) -> features_BTC_1h.parquet
2. Cross gen (~2 hrs, min_nonzero=3) -> v2_crosses_BTC_1h.npz + v2_cross_names_BTC_1h.json
3. Dense conversion (~425GB). **768GB+ machine required.** is_enable_sparse=False for multi-core LightGBM.
4. LightGBM CPCV (folds parallel) -> model_1h.json
5. Optuna (200 trials) -> optuna_configs_1h.json
6. Meta-labeling -> meta_model_1h.pkl
7. LSTM -> lstm_1h.pt + platt_1h.pkl
8. PBO/Audit -> validation_report_1h.json

## CRITICAL LESSON: is_enable_sparse=False
When data is converted from sparse to dense (Step 3), LightGBM MUST have `is_enable_sparse=False` set. Without this, LightGBM uses single-threaded sparse histogram construction even on dense data, wasting all but 1 core. The code in ml_multi_tf.py handles this automatically (line 780), but verify in logs:
```
is_enable_sparse=False (data converted to dense -- enables multi-core LightGBM)
```
If you do NOT see this line and data was converted to dense, training is single-threaded. STOP and investigate.

If the machine has < 768GB cgroup RAM, dense conversion will be SKIPPED and training stays sparse. This is slower but correct — no data is lost. Parallel CPCV still works on sparse via ProcessPoolExecutor.

## CRITICAL: Dense Matrix = 425GB. Needs 768GB+ Machine.
- Dense conversion threshold: dense_bytes < 70% of available RAM
- 425GB / 0.70 = 607GB minimum cgroup RAM for dense
- 1TB machine = ~1000GB cgroup -> 700GB available -> 425GB fits with 275GB headroom
- 504GB machine = ~504GB cgroup -> 353GB available -> 425GB DOES NOT FIT -> stays sparse (slow)
- **Belgium A40 1TB ($0.58/hr) is the correct machine class**

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
  nohup python -u cloud_run_tf.py --symbol BTC --tf 1h > /workspace/1h_log.txt 2>&1 &
```

## Verify Cgroup RAM BEFORE Launch
```bash
# Check actual cgroup RAM (host may advertise more than container gets)
python3 -c "
import os
try:
    with open('/sys/fs/cgroup/memory/memory.limit_in_bytes') as f:
        ram = int(f.read().strip())
    print(f'Cgroup RAM: {ram/1e9:.0f} GB')
    if ram < 768e9:
        print('WARNING: < 768GB — dense conversion will be SKIPPED, training stays sparse (slower)')
    else:
        print('OK — dense conversion will proceed')
except:
    import psutil
    print(f'System RAM: {psutil.virtual_memory().total/1e9:.0f} GB')
"
```

## Verify First 30 Lines (wait ~30s after launch)
```bash
head -30 /workspace/1h_log.txt
```
Must see:
- "All 16 databases present" (or zero "MISS" / "WARNING: DB missing")
- Correct row count: ~75,405
- Correct base feature count: ~3,968
- V30_DATA_DIR pointing to /workspace/v3.3 (NOT v3.0)

## Verify Multi-Threaded Execution
After training starts (Step 4 — CPCV), check load average:
```bash
# Load average should be >> 1.0 on 128+ core machine
uptime
# Expected: load average > 30 (parallel CPCV folds + LightGBM threads)
# If load average ~ 1.0, training is SINGLE-THREADED — this is a critical bug
# Check for "is_enable_sparse=False" in log if data was converted to dense

# Also check RSS matches expected dense matrix size
ps aux | grep cloud_run_tf | grep -v grep
# RSS should be ~425GB+ if dense conversion happened
```

## Verify Crosses Not Truncated
After cross gen completes (Step 2), check the cross feature count:
```bash
python3 -c "
import scipy.sparse as sp
X = sp.load_npz('/workspace/v3.3/v2_crosses_BTC_1h.npz')
print(f'Cross shape: {X.shape}')
print(f'Expected: ~7-8M columns (min_nonzero=3)')
print(f'v3.2 baseline was 6,061,813 columns (min_nonzero=8)')
print(f'NNZ: {X.nnz:,}')
if X.shape[1] < 6_000_000:
    print('WARNING: Fewer crosses than v3.2 — min_nonzero may not be 3')
import json
with open('/workspace/v3.3/v2_cross_names_BTC_1h.json') as f:
    names = json.load(f)
print(f'Cross names: {len(names)}')
assert X.shape[1] == len(names), 'MISMATCH: NPZ cols != JSON names count'
print('OK — NPZ and JSON match')
"
```

## Monitor Commands
```bash
# Live log tail
tail -f /workspace/1h_log.txt

# Check process is alive
ps aux | grep cloud_run_tf | grep -v grep

# Check memory usage (dense matrix should show ~425GB RSS)
free -g

# Check disk space
df -h /workspace

# Check load average (should be >> 1.0 during training)
uptime

# Check for OOM kills:
dmesg | tail -20 | grep -i oom
```

## v3.2 Baseline (targets to beat)
| Metric | v3.2 Value |
|--------|-----------|
| Accuracy | 58.1% (OOS mean across 15 CPCV paths) |
| Dir Acc @>70% conf | 84.9% |
| Meta AUC | 0.648 |
| PBO | DEPLOY (0.14) |
| Cross Features | 6,061,813 (min_nonzero=8) |

## v3.3 Expected (min_nonzero=3)
| Metric | Expected |
|--------|----------|
| Cross features | ~7-8M (up from 6.06M) |
| Dense matrix | ~425GB |
| Training time | ~5 hrs total |

## Download Results When Done
```bash
scp -P PORT root@HOST:/workspace/v3.3/model_1h.json .
scp -P PORT root@HOST:/workspace/v3.3/optuna_configs_1h.json .
scp -P PORT root@HOST:/workspace/v3.3/features_1h_all.json .
scp -P PORT root@HOST:/workspace/v3.3/ml_multi_tf_results.txt .
scp -P PORT root@HOST:/workspace/v3.3/ml_multi_tf_configs.json .
```

## Notes
- min_data_in_leaf=8 for 1h (per TF_MIN_DATA_IN_LEAF in config.py)
- NNZ close to int32 limit (~2B) but should stay under. If it exceeds 2^31, LightGBM SILENTLY produces garbage. Monitor NNZ count after cross gen.
- Subsampling code REMOVED from ml_multi_tf.py. If dense doesn't fit, it stays sparse. No data loss.
- Download ALL artifacts after each step — vast.ai machines die without warning.
