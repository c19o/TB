# Cloud Deployment Checklist — Savage22 V3.3

Extracted from `v3.3/CLAUDE.md` Section 6. Use this as the single-page pre-flight before every cloud deploy.

Last updated: 2026-03-29

---

## PRE-DEPLOY (local, before renting)

- [ ] `python validate.py` passes (96 checks — ZERO failures allowed)
- [ ] All code audited: 3 consecutive clean passes (5x audit with 13 agents, then 3 clean passes)
- [ ] No mid-run patches planned — ALL bugs fixed before deploy
- [ ] Know which TF you're running (one at a time, smallest first)
- [ ] Machine selected by user (NEVER auto-pick — ask first)
- [ ] Machine table includes CPU Score (cores×GHz) and GHz clock speed
- [ ] New machine is FASTER than the previous one (never rent slower)

---

## MACHINE SELECTION CRITERIA

| Requirement | Minimum | Notes |
|------------|---------|-------|
| RAM | 512GB for 1d/4h, 2TB+ for 1h/15m | Dense matrix peak RAM: 1h=526GB, 15m=9TB |
| GPU | NVIDIA with 20GB+ VRAM | RTX 3090 or better |
| CPU | Maximise cores×GHz | Cross gen is CPU-bound |
| Driver | NVIDIA 535+ | Any driver (535+) must work — no driver-specific fixes |
| Image | pytorch/pytorch base | pip cached — no rapids/cuda-specific images |

**15m machine: USER PICKS PERSONALLY from vast.ai lineup. Never auto-select.**

---

## STEP 1: PIP INSTALL

```bash
pip install lightgbm scikit-learn scipy ephem astropy pytz joblib \
    pandas numpy pyarrow optuna hmmlearn numba tqdm pyyaml sparse-dot-mkl
```

**Verify ALL imports before proceeding:**
```bash
python -c "import pandas, numpy, scipy, sklearn, lightgbm, ephem, astropy, \
    pyarrow, optuna, numba, hmmlearn, yaml, tqdm, sparse_dot_mkl; print('ALL OK')"
```

- [ ] Import test prints `ALL OK` (zero errors)
- [ ] `lightgbm.__version__` matches local dev version

---

## STEP 2: SCP UPLOAD SEQUENCE

```bash
# 1. Code tarball
scp -P <PORT> v33_code_latest.tar.gz root@<HOST>:/workspace/

# 2. ALL .db files (16+ databases)
scp -P <PORT> *.db root@<HOST>:/workspace/

# 3. Verify DB count IMMEDIATELY after transfer
ssh -p <PORT> root@<HOST> "ls /workspace/*.db | wc -l"
```

- [ ] DB count >= 16 (missing DB = weaker model = INVALID RUN — stop immediately)
- [ ] Extract code: `tar -xzf v33_code_latest.tar.gz -C /workspace/`
- [ ] Symlink DBs: `ln -sf /workspace/*.db /workspace/v3.3/`

**Required DB list:**
```
btc_prices.db          astrology_full.db      ephemeris_cache.db
fear_greed.db          funding_rates.db       google_trends.db
news_articles.db       tweets.db              macro_data.db
onchain_data.db        open_interest.db       space_weather.db
sports_results.db      llm_cache.db           (+ 2 more project-specific)
```

**Required non-DB uploads:**
- All `v3.3/*.py` and `*.json` files
- `astrology_engine.py` (from project root — NOT inside v3.3/)
- `inference_1d_*.json` / `inference_1w_*.json` artifacts (if running inference TF)

---

## STEP 3: VALIDATE.PY PRE-FLIGHT

```bash
cd /workspace/v3.3
python validate.py
```

- [ ] All 96 checks pass (zero failures)
- [ ] No `WARNING: DB missing` in first 30 lines of output
- [ ] All config paths visible in first 10 log lines (DB_DIR, V30_DATA_DIR, etc.)

**If validate.py fails:** STOP. Fix locally. Re-upload. Do not proceed.

---

## STEP 4: GPU FORK .SO SWAP (if using cuda_sparse device)

GPU fork location: `v3.3/gpu_histogram_fork/`

```bash
# Verify fork .so exists
ls /workspace/v3.3/gpu_histogram_fork/_build/LightGBM/lib_lightgbm.so

# Swap into active LightGBM installation
LGBM_PATH=$(python -c "import lightgbm; import os; print(os.path.dirname(lightgbm.__file__))")
cp /workspace/v3.3/gpu_histogram_fork/_build/LightGBM/lib_lightgbm.so $LGBM_PATH/lib_lightgbm.so

# Verify GPU detected
python -c "import lightgbm; print(lightgbm.__version__); \
    import ctypes; lib=ctypes.cdll.LoadLibrary(lightgbm.basic.find_lib_path()[0]); print('GPU fork loaded')"
```

- [ ] `.so` swap successful
- [ ] LightGBM imports without error after swap
- [ ] `device_type="cuda_sparse"` accepted (will error if CPU-only build)

**Skip this step if using CPU mode.**

---

## STEP 5: LAUNCH TRAINING

```bash
cd /workspace/v3.3
python -u cloud_run_tf.py --symbol BTC --tf 1w 2>&1 | tee train_1w.log
```

**First 30 seconds — mandatory checks:**
- [ ] `validate.py` auto-runs as Step 0 (visible in log)
- [ ] No `WARNING: DB missing` in log
- [ ] `SPARSE=True` visible in log (confirms CSR pipeline active)
- [ ] Config paths correct (DB_DIR, V30_DATA_DIR)
- [ ] `OMP_NUM_THREADS` NOT hardcoded to 4 in env

**DO NOT run concurrent TFs on same machine. One TF at a time.**

---

## STEP 6: MONITORING COMMANDS

```bash
# GPU utilization
watch -n 5 nvidia-smi

# Training progress (tail log)
tail -f train_1w.log

# RAM usage
free -h

# Process status (read-only — NEVER send signals to running process)
ps aux | grep cloud_run_tf

# Disk space
df -h /workspace
```

**Monitoring rules:**
- Poll progress every 30 seconds maximum
- NEVER kill running cross gen (add NPZ skip logic instead)
- NEVER send signals to running processes (use read-only /proc/status, ps, free only)
- Download partial results after EACH critical step (machines die without warning)

---

## STEP 7: ARTIFACT DOWNLOAD (at each checkpoint)

```bash
# Download after cross gen completes
scp -P <PORT> root@<HOST>:/workspace/v3.3/v2_crosses_BTC_1w.npz ./v3.3/
scp -P <PORT> root@<HOST>:/workspace/v3.3/inference_1w_*.json ./v3.3/

# Download after training completes
scp -P <PORT> root@<HOST>:/workspace/v3.3/model_1w.json ./v3.3/
scp -P <PORT> root@<HOST>:/workspace/v3.3/ml_multi_tf_results.txt ./v3.3/
scp -P <PORT> root@<HOST>:/workspace/v3.3/cpcv_oos_predictions_1w.pkl ./v3.3/
scp -P <PORT> root@<HOST>:/workspace/v3.3/platt_1w.pkl ./v3.3/
scp -P <PORT> root@<HOST>:/workspace/v3.3/feature_importance_stability_1w.json ./v3.3/
```

**Download at EVERY checkpoint. Never wait for DONE. Machines die without warning.**

---

## STEP 8: EVALUATE BEFORE DESTROYING

- [ ] Review CPCV WF Accuracy (target: better than previous run)
- [ ] Review PrecL and PrecS (both should be > 0)
- [ ] Check TOP 30 FEATURES for esoteric signal presence (target: 6+ in top 50)
- [ ] Confidence threshold validation table reviewed
- [ ] Model file downloaded and verified locally: `python -c "import lightgbm; m=lightgbm.Booster(model_file='v3.3/model_1w.json'); print(m.num_trees())"`

**NEVER destroy machine before evaluating results. NEVER say "it's fine" about missing data.**

---

## NON-NEGOTIABLE RULES (from CLAUDE.md)

1. NEVER deploy with missing databases
2. NEVER say "it's fine" about missing data
3. NEVER skip validate.py smoke test
4. NEVER deploy code that hasn't had 3 clean audit passes
5. NEVER run concurrent TF builds on same machine
6. NEVER kill running processes without explicit user permission
7. NEVER restart cross gen — add NPZ skip logic instead
8. NEVER use nohup bash wrappers — use cloud_run_tf.py directly
9. NEVER rent slower machine than previous — fix in-place
10. Feature fraction >= 0.7 minimum (low values kill rare esoteric signals silently)
