# 1H Training Guide

## Machine Requirements
- **RAM:** 2TB+ MINIMUM (cross gen peaked at 1871G/2003G — near OOM at both RC=500 and RC=300)
- **Cores:** 256+ (LightGBM OpenMP threads + cross gen)
- **CPU Score:** 1000+ recommended (cores x GHz). GPU REQUIRED for training.
- **Disk:** 80GB+
- **RIGHT_CHUNK:** `export V2_RIGHT_CHUNK=300` (MANDATORY — RC=500 peaked at 1871G/2003G, near OOM. RC=300 also near-OOM'd — NOT safe, just slightly better.)
- **Training stays sparse** — dense would be ~2.3TB, won't fit. int64 indptr handles NNZ > 2^31.
- **GPU:** REQUIRED (A100/H100 or multi-GPU). 75K rows = GPU sweet spot. 3.5x speedup.

### Machine Recommendation: Cloud 2TB+ RAM, GPU REQUIRED
GPU is REQUIRED for training at 75K rows. Without GPU, CPCV alone takes 7 hrs and Optuna takes 18 hrs.
- **Cloud option:** vast.ai or Lambda with A100/H100, 2TB+ RAM, ~$3-4/hr.
- **CPU Score 1000+ recommended.** Multi-GPU ideal for maximum throughput.
- **Cost:** ~$37 without Optuna (CPU), ~$35 with Optuna (GPU), ~$102 with Optuna (CPU).
- **NEVER subsample.** Subsampling code REMOVED. Rent bigger machine.

### GPU vs CPU Per Step
| Step | Engine | Reason |
|------|--------|--------|
| Feature build | GPU (cuDF) | Rolling/ewm on GPU. ~1 hr. |
| Cross gen | GPU (cuSPARSE + streaming) | ~45 min GPU vs 2.3 hrs CPU. |
| Optuna search | CPU (n_jobs=4) | CPU parallel search stage, GPU for final retrain only. |
| Final CPCV | GPU (histogram fork) | ~75K rows = GPU sweet spot. ~2 hrs GPU vs 7 hrs CPU. |
| Trade optimizer | GPU | cuDF-accelerated parameter sweep. |

### Optuna Machine Strategy
- Same GPU machine for Optuna and CPCV. CPU Score 1000+ for search stage (n_jobs=4).
- GPU for final retrain only (after search selects best config).
- Warm-started from 4h: 50+30 trials. ~18 hrs CPU / ~5.5 hrs GPU.
- Separate Optuna machine from training machine ONLY if parallelizing across TFs.

### Revised ETAs (Machine: Score 1000+ CPU, A100/H100 GPU)
| Stage | Time (CPU) | Time (GPU est.) | Status |
|-------|-----------|----------------|--------|
| Feature build | 1 hr | 1 hr | PENDING |
| Cross gen (cuSPARSE + streaming) | 2.3 hrs | ~45 min | PENDING |
| save_binary | 30 min | 30 min | PENDING |
| CPCV (4 folds) | 7 hrs | ~2 hrs (GPU hist) | PENDING |
| Optuna (50+30 warm, n_jobs=4, pruning) | ~18 hrs | ~5.5 hrs (GPU hist) | PENDING |
| Meta + PBO + SHAP | 20 min | 20 min | PENDING |
| **TOTAL (CPU)** | **~30 hrs ($102)** | | |
| **TOTAL (GPU est.)** | | **~10 hrs ($35)** | |
| **Without Optuna (CPU)** | **~11 hrs ($37)** | | |

## CRITICAL: OLD MACHINE DESTROYED — DO NOT USE 33598910
The 504GB machine subsampled from 75,405 to 13,243 rows (lost 24% of data). Subsampling code has been REMOVED from ml_multi_tf.py. If dense doesn't fit, training keeps sparse (slower but no data loss). **Rent 768GB+ machine to get dense speed with full matrix.**

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
- **Rows:** ~75,405 (1h bars, 2017-08-17 to 2026). **Was ~56,875 (only had 2019-2025 data from Binance)**
- **Base features:** ~3,968 cols
- **Cross features:** ~7-8M expected (min_nonzero=3, up from 6.06M at min_nonzero=8)
- **NPZ from v3.3 cloud_results:** STALE (built with min_nonzero=8). Must regenerate.
- **Dense matrix:** ~2.3TB (73K rows x 8M features x 4 bytes). Will NOT fit in any machine — stays sparse.
- **NNZ estimate:** ~8B+ (75K rows x 8M cols x ~1.5% density) — **exceeds int32 limit. int64 indptr fixes NNZ > 2^31. Row-partitioned boosting is BANNED (kills rare signals — Perplexity confirmed).**
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
2. Cross gen (50+ hours estimated, min_nonzero=3) -> v2_crosses_BTC_1h.npz + v2_cross_names_BTC_1h.json
3. Stays SPARSE (~2.3TB dense impossible). SPARSE + SEQUENTIAL CPCV.
4. LightGBM CPCV (4 folds sequential, (4,1) config) -> model_1h.json
5. Optuna (200 trials) -> optuna_configs_1h.json
6. Meta-labeling -> meta_model_1h.pkl
7. LSTM -> lstm_1h.pt + platt_1h.pkl
8. PBO/Audit -> validation_report_1h.json

## SPARSE TRAINING (1H always stays sparse)
Dense matrix would be ~2.3TB — impossible on any available machine. 1H always stays sparse.
- `is_enable_sparse=True` stays set (default from config.py)
- LightGBM trains on sparse CSR with EFB bundling
- SPARSE + SEQUENTIAL CPCV (parallel disabled for >1M features — pickle bottleneck)
- int64 indptr handles NNZ > 2^31 (applied by `_ensure_lgbm_sparse_dtypes()`)
- Verify in log: `"Keeping SPARSE (dense would need XXX GB, only YYY GB avail)"`

## CRITICAL: Dense Matrix = ~2.3TB. IMPOSSIBLE — Stays Sparse.
- Dense conversion threshold: dense_bytes < 70% of available RAM
- 2.3TB dense matrix cannot fit in any available machine
- Training MUST stay sparse. SPARSE + SEQUENTIAL CPCV (parallel disabled for >1M features — pickle bottleneck)
- int64 indptr fixes NNZ > 2^31. Row-partitioned boosting is BANNED (kills rare signals — Perplexity confirmed).

## Install Dependencies
```bash
pip install -q lightgbm scikit-learn scipy ephem astropy pytz joblib pandas numpy \
  pyarrow optuna hmmlearn numba tqdm pyyaml alembic cmaes colorlog sqlalchemy \
  threadpoolctl psutil 2>&1 | tail -3
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
    if ram < 2000e9:
        print(f'WARNING: < 2TB — cross gen may OOM (peaked 1871G/2003G at RC=300)')
    else:
        print('OK — 2TB+ available for cross gen')
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
After training starts (Step 4 -- CPCV), check load average:
```bash
# Load average should be >> 1.0 on 128+ core machine
uptime
# Expected: load average > 30 (LightGBM threads within sequential CPCV fold)
# If load average ~ 1.0, training is SINGLE-THREADED — this is a critical bug
# 1H always stays sparse (no dense conversion). Check log for "Keeping SPARSE".

# Also check RSS (stays sparse — no dense conversion for 1h)
ps aux | grep cloud_run_tf | grep -v grep
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

# Check memory usage (stays sparse — no dense conversion)
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
| Dense matrix | ~2.3TB (impossible — stays sparse) |
| Training time | 80+ hours estimated total |

## STATUS
**READY** — (4,1)=4 folds, no Optuna. Est. 9-15 hrs, ~$16-26. Single machine.

## Failure Modes and What to Check

| Symptom | Cause | Fix |
|---------|-------|-----|
| "WARNING: DB missing" in log | Missing database file | Upload the missing .db, re-run verify script |
| Cross cols < 6M | Old v2_cross_names JSON left over (min_nonzero=8) | Delete BOTH npz AND json, restart |
| OOM during cross gen | V2_RIGHT_CHUNK too large | Set V2_RIGHT_CHUNK=300 (lower to 200 if still OOMing) |
| "ModuleNotFoundError: astrology_engine" | astrology_engine.py not in v3.3/ | Copy from project root |
| Feature count mismatch | Stale parquet from old feature_library.py | Delete features_BTC_1h.parquet, restart |
| V30_DATA_DIR shows v3.0 path | Env var not set | Verify `export V30_DATA_DIR=/workspace/v3.3` |
| Cross gen very slow (>72 hrs) | Single-threaded sparse matmul | Normal for 75K rows x 7-8M crosses. Wait. |
| LSTM crashes with NaN | Features have NaN not imputed for LSTM | ml_multi_tf.py imputes NaN->0 for LSTM only. Check log. |
| NNZ overflow (silent corruption) | int32 indptr on >2B NNZ | Verify `_ensure_lgbm_sparse_dtypes()` applied. Check log for int64 indptr. |

---

## OPTUNA DEPLOYMENT

### Upload Size for Optuna
- **Total upload: ~54 GB** (parquet + NPZ + cross_names JSON + all DBs)
- **Recommend same machine as training** — 54 GB upload is large and slow to transfer
- Warm-started from 4h: 50+30 trials (vs 100+50 cold)

### Optuna Timing
- Optuna takes ~2-3x final training time (target after optimizations)
- 1h CPCV = ~7 hrs CPU / ~2 hrs GPU, so Optuna = ~18 hrs CPU / ~5.5 hrs GPU
- save_binary bridge eliminates redundant EFB construction (Dataset parsed once, reused across all trials)

### If Running Optuna on a Separate Machine (NOT recommended)
Upload these files:
```
features_BTC_1h.parquet          # base feature parquet
v2_crosses_BTC_1h.npz            # cross feature sparse matrix (~45-50 GB)
v2_cross_names_BTC_1h.json       # cross feature column names
lgbm_dataset_1h.bin              # save_binary output (if available — skips EFB rebuild)
model_1h.json                    # trained model (warm-start seed)
optuna_configs_4h.json           # 4h Optuna results (warm-start cascade source)
All 16 .db files + kp_history_gfz.txt + astrology_engine.py
v33_code.tar.gz                  # all v3.3/*.py code
```
Total: **~54 GB.** At 100 Mbps upload = ~1.2 hrs transfer. At 1 Gbps = ~7 min.
**Recommend keeping Optuna on the same machine as training to avoid this transfer.**

---

## Download Results When Done

```bash
# Check if pipeline completed
grep "DONE\|pipeline complete\|All steps" /workspace/1h_log.txt

# Download all artifacts
scp -P {PORT} root@{HOST}:/workspace/v3.3/model_1h.json .
scp -P {PORT} root@{HOST}:/workspace/v3.3/optuna_configs_1h.json .
scp -P {PORT} root@{HOST}:/workspace/v3.3/meta_model_1h.pkl .
scp -P {PORT} root@{HOST}:/workspace/v3.3/lstm_1h.pt .
scp -P {PORT} root@{HOST}:/workspace/v3.3/platt_1h.pkl .
scp -P {PORT} root@{HOST}:/workspace/v3.3/validation_report_1h.json .
scp -P {PORT} root@{HOST}:/workspace/v3.3/feature_importance_*.json .
scp -P {PORT} root@{HOST}:/workspace/1h_log.txt .
```

**Download partial results after each critical step (vast.ai machines die without warning).**

## LightGBM Config (from config.py)

| Parameter | Value | Source |
|-----------|-------|--------|
| min_data_in_leaf | 8 | TF_MIN_DATA_IN_LEAF['1h'] |
| num_leaves | 63 | TF_NUM_LEAVES['1h'] |
| max_bin | 255 | V3_LGBM_PARAMS (binary crosses always get 2 bins regardless) |
| CPCV folds | (4,1) = 4 folds | TF_CPCV_GROUPS['1h'] |
| save_binary | Not feasible | ~2.3TB dense matrix, stays sparse |

## Notes
- min_data_in_leaf=8 for 1h (per TF_MIN_DATA_IN_LEAF in config.py)
- 4 CPCV paths (N=4, K=1) — was (6,2)=15 folds. Production model identical regardless (trains on ALL data). See FOLD_STRATEGY.md.
- NNZ will exceed int32 limit (~2B). int64 indptr fixes NNZ > 2^31. Row-partitioned boosting is BANNED (kills rare signals -- Perplexity confirmed).
- Subsampling code REMOVED from ml_multi_tf.py. If dense doesn't fit, it stays sparse. No data loss.
- Download ALL artifacts after each step -- vast.ai machines die without warning.
