# V3.3 Cloud Training Protocol — AUTHORITATIVE
# Follow this EXACTLY. No shortcuts. No improvisation.

## STACK

```
Image:  PICK BY DRIVER VERSION:
  Driver 535-579 (CUDA 12.x) → rapidsai/base:25.12-cuda12-py3.12
  Driver 580+    (CUDA 13.x) → rapidsai/base:25.12-cuda13-py3.12
Pip:    lightgbm optuna scipy scikit-learn ephem
Fix:    Disable numba_cuda package on EVERY machine (see Step 4b)
```

**CUDA 12 and 13 are NOT cross-compatible.** Container CUDA major must match host driver CUDA major. This is a hard boundary — no workaround.

**CUDA Init Fix (learned 2026-03-25):**
RAPIDS 25.02 ships `numba-cuda` as a separate package. It tries CUDA context init on import.
- `NUMBA_DISABLE_CUDA=1` is WRONG — numba_cuda reads it and THROWS instead of degrading
- `pip uninstall numba-cuda` doesn't survive relaunches
- Root cause: `CUDA_VISIBLE_DEVICES` may be empty on some vast.ai machines → numba_cuda enumerates 0 devices → IndexError

**Fix (multi-layer, server-agnostic):**
1. Filter vast.ai for `CUDA 12.8` in the search results — container CUDA must match host CUDA. Driver 570.x = CUDA 12.8. Driver 580.x = CUDA 13.0 (INCOMPATIBLE with numba in CUDA 12.8 container). Driver 590.x = CUDA 13.1 (also incompatible).
2. Disable numba_cuda external package: `mv numba_cuda numba_cuda_DISABLED && mv _numba_cuda_redirector.py DISABLED && mv .pth DISABLED`
3. Set `NUMBA_DISABLE_CUDA=1` to suppress numba's built-in CUDA probe (safe after numba_cuda package is removed)
4. Keep `CUDA_VISIBLE_DEVICES=0` for CuPy/cuDF (they use their own CUDA path, not numba's)

**CRITICAL: The smoke test MUST include cuDF-to-numpy conversion:**
```python
import cudf, numpy as np
gdf = cudf.DataFrame({'a': np.random.randn(100)})
arr = gdf['a'].to_numpy()  # This is the actual crash path — tests cuDF→numba.cuda→numpy
```
If this fails, the machine is incompatible. Do NOT proceed.

**Why NOT plain CUDA image:** RAPIDS C++ libs (libcudf, librmm) are complex to install at runtime. Pre-built image saves 10+ min per machine.

## RENT COMMAND (copy-paste ready)

```bash
vastai create instance <OFFER_ID> \
  --image rapidsai/base:25.02-cuda12.8-py3.12 \
  --disk 50 --ssh \
  --onstart-cmd 'pip install --no-cache-dir lightgbm optuna scipy scikit-learn ephem'
```

## MACHINE SELECTION TABLE FORMAT

Always present with CPU Score:

| # | Machine | Cores | GHz | CPU Score | RAM | $/hr | Est. Time | Est. Cost | ID |

CPU Score = Cores × GHz.

### Per-TF Requirements (v3.3 with 3.34M features)

| TF | Rows | Dense GB | Min RAM | Min Cores | Min Driver |
|----|------|----------|---------|-----------|------------|
| 1w | 818 | 2.5 | 32 GB | 64 | 570+ |
| 1d | 5,727 | 71 | 128 GB | 128 | 570+ |
| 4h | 4,380 | 55 | 128 GB | 128 | 570+ |
| 1h | 17,520 | 300 | 512 GB | 256 | 570+ |
| 15m | 227,577 | 1,900 | 2 TB | 256 | 570+ |

### vast.ai Search Filter

```bash
vastai search offers 'cpu_ram>=128 cpu_cores>=64 driver_version>=570 rentable=true' -o 'dph_total' --limit 15
```

## DEPLOY PROTOCOL (step by step, no skipping)

### Step 1: Rent Machine
Use the rent command above. Wait for status=running.

```bash
vastai show instances --raw | python -c "import json,sys; [print(f'Status: {i[\"actual_status\"]} SSH: ssh -p {i[\"ssh_port\"]} root@{i[\"ssh_host\"]}') for i in json.load(sys.stdin) if i['id']==<ID>]"
```

### Step 2: Verify SSH
```bash
SSH="ssh -o StrictHostKeyChecking=no -o IdentityFile=~/.ssh/vast_key -o IdentitiesOnly=yes -p <PORT> root@<HOST>"
$SSH "echo CONNECTED && nproc && free -h | head -2"
```

### Step 3: Upload Tar
```bash
SCP="scp -o StrictHostKeyChecking=no -o IdentityFile=~/.ssh/vast_key -o IdentitiesOnly=yes -P <PORT>"
$SCP /tmp/v33_upload.tar.gz root@<HOST>:/workspace/
```

### Step 4: Extract + Symlink
```bash
$SSH "cd /workspace && tar xzf v33_upload.tar.gz && for f in *.db kp_history_gfz.txt; do ln -sf /workspace/\$f /workspace/v3.2_2.9M_Features/\$f; done && ln -sf /workspace/astrology_engine.py /workspace/v3.2_2.9M_Features/"
```

### Step 5: SMOKE TEST (MANDATORY — never skip)
```bash
$SSH "export NUMBA_DISABLE_CUDA=1 && python -c \"
import os
os.environ['NUMBA_DISABLE_CUDA'] = '1'
print('=== SMOKE TEST ===')

# 1. numba CPU JIT
from numba import njit
import numpy as np
@njit
def _t(x): return x * 2.0
assert _t(np.float64(3.0)) == 6.0
print('OK: numba @njit CPU')

# 2. cuDF
import cudf
gdf = cudf.DataFrame({'a': [1,2,3]})
assert len(gdf) == 3
print('OK: cuDF')

# 3. CuPy
import cupy as cp
x = cp.array([1,2,3])
print(f'OK: CuPy on GPU')

# 4. scipy sparse
from scipy.sparse import csr_matrix
m = csr_matrix(np.eye(100))
print(f'OK: scipy sparse')

# 5. LightGBM
import lightgbm as lgb
print(f'OK: LightGBM {lgb.__version__}')

# 6. ephem
import ephem
print(f'OK: ephem {ephem.__version__}')

# 7. Config check
import sys; sys.path.insert(0, '/workspace/v3.2_2.9M_Features')
from config import V3_LGBM_PARAMS, TF_CLASS_WEIGHT
assert V3_LGBM_PARAMS['max_bin'] == 63
assert V3_LGBM_PARAMS['max_conflict_rate'] == 0.0
assert 'path_smooth' in V3_LGBM_PARAMS
assert TF_CLASS_WEIGHT == {'1d': 'balanced', '1w': 'balanced'}
print('OK: Config verified')

print('=== ALL PASSED ===')
\""
```

**IF ANY CHECK FAILS: DO NOT PROCEED. Fix the issue first.**

### Step 6: Create Launcher
```bash
cat > /tmp/launch.sh << 'SCRIPT'
#!/bin/bash
cd /workspace/v3.2_2.9M_Features
export SAVAGE22_DB_DIR=/workspace
export V30_DATA_DIR=/workspace/v3.2_2.9M_Features
export PYTHONUNBUFFERED=1
export NUMBA_DISABLE_CUDA=1
python -u cloud_run_tf.py --symbol BTC --tf <TF> > /workspace/<TF>_pipeline.log 2>&1 &
echo "PID: $!"
SCRIPT
$SCP /tmp/launch.sh root@<HOST>:/workspace/launch.sh
$SSH "bash /workspace/launch.sh"
```

### Step 7: Monitor
```bash
# Check every 30s max
$SSH "tail -20 /workspace/<TF>_pipeline.log"

# Check for DONE marker
$SSH "ls /workspace/DONE_<TF> 2>/dev/null && echo DONE || echo RUNNING"
```

### Step 8: Download Results (BEFORE killing!)
```bash
# Download ALL artifacts
for f in model_<TF>.json v2_crosses_BTC_<TF>.npz optuna_configs_<TF>.json \
         exhaustive_configs_<TF>.json meta_model_<TF>.pkl platt_<TF>.pkl \
         lstm_<TF>.pt features_BTC_<TF>.parquet feature_importance_top500_<TF>.json \
         feature_importance_summary.json shap_analysis_<TF>.json \
         inference_<TF>_thresholds.json inference_<TF>_cross_pairs.npz \
         inference_<TF>_ctx_names.json inference_<TF>_base_cols.json \
         inference_<TF>_cross_names.json cpcv_oos_predictions_<TF>.pkl; do
    $SCP root@<HOST>:/workspace/v3.2_2.9M_Features/$f . 2>/dev/null && echo "OK: $f" || echo "MISSING: $f"
done
$SCP root@<HOST>:/workspace/<TF>_pipeline.log .
```

### Step 9: Verify Downloads
```bash
# Model must be >1MB
ls -lh model_<TF>.json
# NPZ must be >10MB
ls -lh v2_crosses_BTC_<TF>.npz
# Check accuracy in model
python -c "import json; m=json.load(open('model_<TF>.json')); print(f'Accuracy: {m.get(\"test_accuracy\",\"MISSING\")}')"
# Check SHAP report
python -c "import json; s=json.load(open('shap_analysis_<TF>.json')); print(f'Cross SHAP: {s[\"cross_shap_pct\"]}%')"
```

### Step 10: Kill Machine
```bash
vastai destroy instance <ID>
```

## PIPELINE STEPS (what cloud_run_tf.py runs)

| Step | What | Est. Time (1w/1d/4h/1h/15m) |
|------|------|---------------------------|
| 1 | Install deps | 30s |
| 2 | Verify DB | 5s |
| 3 | Feature rebuild | 3m / 15m / 15m / 30m / 90m |
| 4 | Cross generation | 2m / 5m / 5m / 15m / 120m |
| 5 | CPCV Training | 2m / 20m / 20m / 60m / 120m |
| 6 | Optuna search | 5m / 90m / 90m / 180m / 480m |
| 7 | Exhaustive optimizer | 8m / 12m / 12m / 20m / 35m |
| 8 | Meta-labeling | 2m / 5m / 5m / 10m / 15m |
| 9 | LSTM | 5m / 15m / 15m / 30m / 45m |
| 10 | PBO + Audit | 1m / 3m / 3m / 5m / 5m |
| 11 | SHAP analysis | 1m / 5m / 5m / 10m / 15m |
| **TOTAL** | | **~29m / ~2.9h / ~2.9h / ~6h / ~15.4h** |

## CRITICAL RULES

1. **Disable numba_cuda package** on EVERY machine: `mv numba_cuda DISABLED && mv _numba_cuda_redirector.py DISABLED && mv .pth DISABLED` in site-packages.
1b. **Symlink artifacts to /workspace** — cloud_run_tf.py CWD is /workspace but build scripts save to /workspace/v3.2_2.9M_Features/. After feature build, symlink: `ln -sf /workspace/v3.2_2.9M_Features/features_BTC_*.parquet /workspace/ && ln -sf /workspace/v3.2_2.9M_Features/v2_crosses_BTC_*.npz /workspace/ && ln -sf /workspace/v3.2_2.9M_Features/v2_cross_names_BTC_*.json /workspace/`
1c. **Verify zero "WARNING: DB missing"** in first 30s of log. If any appear, STOP.
1d. **cloud_run_tf.py os.chdir('/workspace') breaks ALL script paths** — tar extracts to /workspace/v3.2_2.9M_Features/ but CWD is /workspace/. All `python script.py` calls must use `_script()` helper to resolve full path. Fixed in v3.3. Also symlink parquets/NPZs from v3.2 dir to /workspace after feature build.
2. **SMOKE TEST before training** — never skip Step 5
3. **Download before killing** — once destroyed, artifacts are GONE
4. **Download NPZ after cross gen** — if machine dies mid-training, NPZ is the irreplaceable artifact
5. **One TF per machine** — no concurrent training on same machine
6. **Verify SPARSE in training logs** — if log says DENSE, abort (wrong code path)
7. **`--symbol BTC` not `--asset BTC`**
8. **Reuse running machines** — scp new files + relaunch, don't kill + rent new
9. **Poll every 30s max** — use run_in_background, never block the conversation
10. **Always respond to user first** — never make them Esc to get attention

## GPU COMPATIBILITY

| GPU | Compute | Min Driver | RAPIDS 25.02 |
|-----|---------|-----------|-------------|
| RTX 3060-3090 | sm_86 | 570+ | OK |
| RTX 4070-4090 | sm_89 | 570+ | OK |
| RTX 5060-5090 | sm_100 | 570+ | OK (PTX JIT) |
| A100 | sm_80 | 570+ | OK |
| H100 | sm_90 | 570+ | OK |

Filter vast.ai: `driver_version >= 570`

## UPLOAD TAR CONTENTS

```bash
cd "C:/Users/C/Documents/Savage22 Server"
tar czf /tmp/v33_upload.tar.gz \
  v3.2_2.9M_Features/*.py \
  astrology_engine.py \
  btc_prices.db tweets.db news_articles.db sports_results.db \
  space_weather.db onchain_data.db macro_data.db \
  astrology_full.db ephemeris_cache.db fear_greed.db \
  funding_rates.db google_trends.db \
  kp_history_gfz.txt
```

**ALL 12 databases MUST be included. Missing ANY = invalid model. Check with:**
```bash
ls /workspace/*.db | wc -l  # Must be >= 12
```

Expected size: ~165MB. Upload time: ~15-30s on fast connections.
