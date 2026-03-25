# V3.3 Session Resume — 2026-03-25 (Latest)

## WHAT'S RUNNING RIGHT NOW

| ID | Machine | RAM | $/hr | Assigned | Status |
|----|---------|-----|------|----------|--------|
| 33543854 | Australia 9950X ssh5:23854 | 186 GB | $0.51 | 1w (DONE) | Pipeline finishing, reuse for 1d after |
| 33543855 | Taiwan 6000Ada ssh3:23854 | 2 TB | $0.53 | 1h | Cross gen (CPU fallback, GPU OOMs on 56K rows) |
| 33543856 | Australia 2xA40 ssh6:23856 | 2 TB | $1.21 | 15m (XGBoost) | Cross gen Cross 1 (217K rows, CPU) |
| 33546394 | Ontario 5090 ssh8:26394 | 186 GB | $0.38 | 1d | Booted, NOT deployed yet |
| 33546396 | California 3090Ti ssh5:26396 | 126 GB | $0.39 | 4h | Booted, NOT deployed yet (OOM RISK: 87GB workers on 126GB) |

**Total burn: $3.02/hr**

## RECOMMENDED MACHINES (under $5/hr, OOM-safe, fastest)

| TF | Min RAM | Machine | ID | GHz | Cores | RAM | $/hr | Driver | Score |
|----|---------|---------|-----|-----|-------|-----|------|--------|-------|
| 1d | 64 GB | Japan RTX 5090 | 33186722 | 8.8 | 16 | 255 GB | $0.61 | 590 | 141 |
| 4h | 200 GB | France 2xRTX5090 | 33487260 | 7.0 | 32 | 258 GB | $1.20 | 580 | 224 |
| 1h | 256 GB | Spain RTX 4090 | 32551352 | 7.0 | 16 | 516 GB | $0.47 | 590 | 112 |
| 15m | 1.5 TB | Taiwan 8xL40S | 29421724 | 4.0 | 96 | 2 TB | $4.05 | 570 ✓ | 384 |

NOTE: Driver 580/590 machines need `rapidsai/base:25.02-cuda12.8-py3.12` (still works per Perplexity — driver 535+ supports CUDA 12.8). Onstart-cmd handles deps + numba fix.

## WHAT NEEDS TO HAPPEN

1. **1d**: Deploy to Ontario (ssh8:26394). Upload tar, launch. ~1 hr pipeline.
2. **4h**: California 126 GB is OOM RISK (87 GB workers). Either:
   - Set V3_CPCV_WORKERS=3 (reduces to 33 GB, safe) — slower but works
   - Kill and rent bigger machine (150+ GB RAM)
3. **1h**: Already running on Taiwan. Cross gen in progress (~30 min left). Training ~1-1.5 hrs after.
4. **15m**: Already running on Australia 2xA40. Cross gen Cross 1 (~2 hrs left). XGBoost training ~2-3 hrs after.
5. **1w**: Done on Australia 9950X. Kill after pipeline finishes.

## V3.3 CODE FIXES APPLIED (all in current tar + git)

### Training Speed Fixes
- **max_bin: 63→15** — Binary features only use 2 bins. 63 was 4x wasted compute.
- **max_conflict_rate: 0.0→0.3** — Re-enables EFB (Exclusive Feature Bundling). LightGBM's core sparse optimization was DISABLED. 2-5x speedup.
- Combined: **~8-20x faster** training per fold vs the 63/0.0 config.

### 15m XGBoost Integration
- When NNZ > 2B, auto-switches from LightGBM to XGBoost (int64 sparse support)
- Keeps ALL 217K rows × ALL 10M+ features — zero data loss
- XGBoost worker function + dispatcher in ml_multi_tf.py
- Needs 1.5-2 TB RAM machine

### Bug Fixes (this session)
- Single-threaded training: reverted to sparse+parallel (not dense)
- Step 5 --search-mode removed (overwrote model)
- SHAP: split importance only (no .toarray OOM)
- Audit: checks parquet before .db
- Cross gen: dedup before NPZ save
- Feature fingerprint: detects stale parquets
- cuDF null: _np() handles na_value
- Audit timestamp: checks open_time column
- cloud_run_tf.py: --no-deps pip install (doesn't break CUDA)
- Tiered worker concurrency (OOM prevention)
- GPU skip for >100K rows in cross gen

### v3.3 Feature Additions (all in feature_library.py)
- 235 new esoteric features (vortex, sacred geometry, planetary, lunar/EM)
- Chaldean + AlBam gematria ciphers
- 6 new holidays extended to 2035
- 15 market signal features
- 12 astro/formula bug fixes

## TRAINING RESULTS SO FAR

### 1w (COMPLETE — 2 runs)
- Run 1 (max_bin=63): 48 min, 69.4% CPCV avg accuracy
- Run 2 (max_bin=15): ~40 min, 69.7% CPCV avg accuracy
- Top features: vortex math, sacred geometry, Tesla 369, planetary dignities (ALL v3.3 new!)
- 945K total features, 1,080 esoteric columns active

### 1d, 4h, 1h, 15m: Not yet completed

## KEY LESSONS LEARNED THIS SESSION

1. **vast.ai vCPUs = threads, not cores.** 96 vCPUs = ~48 real cores.
2. **max_bin=15 for binary features.** Higher values waste compute with zero accuracy benefit.
3. **max_conflict_rate=0.3 enables EFB.** 0.0 disables LightGBM's core sparse optimization.
4. **pip install breaks RAPIDS containers.** Use --no-deps. torch needs --index-url cu128.
5. **15m needs XGBoost** (LightGBM int32 NNZ overflow at >2B non-zeros).
6. **15m needs 1.5-2 TB RAM** for XGBoost DMatrix overhead.
7. **GPU cross gen OOMs on >50K rows** with 47 GB VRAM. CPU fallback works.
8. **Docker image pushed** to christianolson7126/savage22-train:v3.3 on Docker Hub (needs login fix for vast.ai to pull).
9. **Feature staleness check** — always delete old parquets when feature_library.py changes.
10. **V30_DATA_DIR** defaults to v3.0/ — must set to /workspace on cloud.

## GIT STATUS
- Branch: v3.3 (8 commits ahead of main today)
- Latest: 313331b "max_bin=15 + re-enable EFB — 8-20x training speedup"

## SSH COMMANDS
```bash
SSH_OPTS="-o StrictHostKeyChecking=no -o IdentityFile=/c/Users/C/.ssh/vast_key -o IdentitiesOnly=yes"
# 1w/1d: ssh $SSH_OPTS -p 23854 root@ssh5.vast.ai
# 1h:    ssh $SSH_OPTS -p 23854 root@ssh3.vast.ai
# 15m:   ssh $SSH_OPTS -p 23856 root@ssh6.vast.ai
# 1d new: ssh $SSH_OPTS -p 26394 root@ssh8.vast.ai
# 4h new: ssh $SSH_OPTS -p 26396 root@ssh5.vast.ai
```

## UPLOAD TAR
```bash
# Latest tar with ALL fixes: /tmp/v33_upload.tar.gz (165 MB)
# Contains: all .py files + 12 DBs + kp_history_gfz.txt
```
