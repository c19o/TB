# V3.3 Full Training Pipeline Document

Last updated: 2026-03-28

This document describes the complete v3.3 training pipeline for the Savage22 BTC prediction system. The pipeline trains LightGBM models on millions of cross-features derived from esoteric, astrological, gematria, numerology, space weather, and technical analysis signals. The core thesis: more diverse signals = stronger predictions. Every feature matters. No filtering, no subsampling. LightGBM with EFB on sparse CSR is the only acceptable architecture.

Orchestrator script: `cloud_run_tf.py --symbol BTC --tf {TF}`

---

## Pipeline Steps (In Order)

### Step 0: Environment Setup

**What it does:** Kills stale processes, installs Python dependencies, verifies all 16 required databases are present, checks disk space (20GB+ free), validates RAM meets TF minimums, and removes stale artifacts from previous runs.

**Script:** `cloud_run_tf.py` (inline, lines 165-300)

**Dependencies installed:**
```
lightgbm scikit-learn scipy ephem astropy pytz joblib pandas numpy
pyarrow optuna hmmlearn numba tqdm pyyaml alembic cmaes colorlog
sqlalchemy threadpoolctl
```

**Required databases (all 16 must be present):**
btc_prices.db, tweets.db, news_articles.db, sports_results.db, space_weather.db, onchain_data.db, macro_data.db, astrology_full.db, ephemeris_cache.db, fear_greed.db, funding_rates.db, google_trends.db, open_interest.db, multi_asset_prices.db, llm_cache.db, v2_signals.db

Also required: `kp_history_gfz.txt` (Kp geomagnetic index history)

**RAM/CPU/GPU:** Minimal. CPU only. ~1 minute.

**TF-specific caveats:**
- 15m: Sets `V2_RIGHT_CHUNK=500` and `V2_BATCH_MAX=500` env vars to prevent cross gen OOM
- All TFs: Lockfile prevents duplicate pipeline runs

**Known issues:**
- `btc_prices.db` may have bare "BTC" symbols instead of "BTC/USDT" -- auto-fixed by adding "/USDT" suffix
- `config.py` V30_DATA_DIR defaults to `v3.0 (LGBM)/` which has OLD data. Cloud sets V30_DATA_DIR=/workspace. Locally MUST export correct path.
- `astrology_engine.py` lives in project root, not v3.3/ -- must be copied into v3.3/ for cloud deploys

**Estimated time:** 1-2 minutes

---

### Step 1: Base Feature Build

**What it does:** Builds ~3,000-3,400 base features from 16 data sources. Includes TA indicators, esoteric calendar signals, gematria on tweet/news text, astrology (planetary positions, aspects, moon phases), numerology (vortex math, sacred geometry, Lo Shu), space weather (Kp, solar flux, Schumann), sports correlations, on-chain metrics, and macro data. All numeric features get 4-tier binarization (low/mid_low/mid_high/high). V2 layers add entropy, Hurst exponent, Fibonacci levels, moon signs, extra lags, etc.

**Script:** `build_features_v2.py --symbol BTC --tf {TF}`

**Called by:** `cloud_run_tf.py` (only if parquet missing, <2000 cols, or fails v3.3 fingerprint check)

**Fingerprint columns (must all be present):** vortex_family_group, mars_speed, moon_distance_norm, loshu_row

**RAM/CPU/GPU:**
- CPU-bound (pandas .apply() was eliminated -- all vectorized/Numba now)
- GPU accelerated via cuDF if CUDA <13.0. On CUDA 13.0+ (driver 580+), falls back to pandas CPU.
- RAM: 8-16GB sufficient for all TFs
- Uses batch column assignment (dict accumulation + pd.concat) to avoid DataFrame reindex overhead

**TF-specific caveats:**
| TF | Rows | Build time (est.) | Notes |
|----|------|-------------------|-------|
| 1w | ~818 | 2-5 min | Smallest, fastest |
| 1d | ~5,700 | 3-8 min | Standard |
| 4h | ~23,000 | 5-15 min | Moderate |
| 1h | ~91,000 | 15-30 min | Large |
| 15m | ~227,000 | 30-60 min | Largest, highest RAM during binarization |

**Known issues:**
- Parquet col count >2000 causes pipeline to skip rebuild even if 235 features are missing. The v3.3 fingerprint check catches this.
- SQLite has 2000 column limit -- features saved as parquet, not DB
- cuDF doesn't support `.map(dict)` -- replaced with merge pattern
- cuDF doesn't support `reindex(method='ffill')` -- uses reindex + ffill separately

**Output:** `features_BTC_{TF}.parquet`

**Estimated time:** 2-60 min depending on TF

---

### Step 2: Cross Feature Generation (THE BIG ONE)

**What it does:** Generates millions of cross-features by multiplying every binarized left-side feature (TA indicators) against every binarized right-side context (esoteric, astro, gematria, numerology, space weather, sentiment). This is where the matrix thesis manifests: RSI_high x full_moon, MACD_cross x mercury_retrograde, etc. Uses 4-tier binarization on ALL numeric columns. Co-occurrence filter of 3 (min_nonzero=3, matches min_data_in_leaf).

**Script:** `v2_cross_generator.py --tf {TF} --symbol BTC --save-sparse`

**Called by:** `cloud_run_tf.py` (skipped if valid NPZ already exists with sufficient column count)

**RAM/CPU/GPU:**
- This step dominates pipeline time and RAM
- CPU path: Multi-threaded Numba prange for feature multiplication
- GPU path: CuPy sparse matmul, requires 20GB+ VRAM. GPUs with <20GB are SLOWER than multi-core CPU.
- Thread count: `ram_gb * 0.10 / per_worker_gb` (reserves 90% RAM for data accumulation)
- `OMP_NUM_THREADS=4` during cross gen (set by cloud_run_tf.py) to prevent thread exhaustion

**TF-specific caveats:**

| TF | Rows | Est. Features | Peak RAM | Min RAM | Est. Time | Notes |
|----|------|--------------|----------|---------|-----------|-------|
| 1w | 818 | ~2.2M | ~370GB | 64GB | 1-4 hrs | Fits in RAM easily. GPU path viable. |
| 1d | 5,700 | ~6M | ~500GB | 128GB | 4-12 hrs | Needs disk flush architecture |
| 4h | 23,000 | ~5M | ~600GB | 256GB | 6-18 hrs | Needs disk flush architecture |
| 1h | 91,000 | ~8M | ~1.5TB | 512GB | 12-36 hrs | Needs memmap OR tcmalloc |
| 15m | 227,000 | ~10M+ | ~2TB+ | 768GB | 24-72 hrs | int64 indptr required (NNZ > 2^31) |

**NPZ validation thresholds (stale NPZ detection):**
- 1w: >= 1.5M cols
- 1d: >= 5M cols
- 4h: >= 3M cols
- 1h: >= 5M cols
- 15m: >= 5M cols

**Output:** `v2_crosses_BTC_{TF}.npz`, `v2_cross_names_BTC_{TF}.json`, inference artifacts (5 files)

**Estimated time:** 1-72 hours depending on TF and machine specs

---

### Step 3: Optuna Hyperparameter Search

**What it does:** Runs a 2-phase Optuna search to find optimal LightGBM hyperparameters. Phase 1: 25 trials (2 seeded + 8 random + 15 TPE) with 2-fold CPCV, fast LR (0.15), 60 max rounds. Phase 2: Top-3 candidates validated with 4-fold CPCV, 200 rounds, LR 0.08. Uses `Dataset.subset()` (1000x faster than `reference=`), PatientPruner, and warm-start boosting. Sortino ratio is the optimization objective.

**Script:** `run_optuna_local.py --tf {TF}`

**Called by:** `cloud_run_tf.py` (skipped if `optuna_configs_{TF}.json` exists)

**RAM/CPU/GPU:**
- Loads full sparse cross matrix into RAM (same as training)
- CPU: All cores via LightGBM num_threads=0
- GPU: Not used for training (LightGBM CPU with EFB on sparse CSR)
- OMP_NUM_THREADS and NUMBA_NUM_THREADS unset for this phase (LightGBM auto-detects)

**TF-specific caveats:**
| TF | Optuna Rows Subsample | num_leaves cap | min_data_in_leaf |
|----|----------------------|----------------|------------------|
| 1w | 1.0 (all 818 rows) | 31 | 5 |
| 1d | configurable | 127 | 5 |
| 4h | configurable | 255 | 5 |
| 1h | configurable | 511 | 8 |
| 15m | configurable | 511 | 15 |

**Known issues:**
- v3.2's best model used num_leaves=7; Optuna floor lowered to 4 to capture shallow-tree optima
- `feature_pre_filter=False` is CRITICAL in all lgb.Dataset() calls -- True silently kills rare esoteric features
- Warm-start mode for downstream TFs uses fewer trials (15) and validation candidates (2)

**Output:** `optuna_configs_{TF}.json`

**Estimated time:** 30 min - 4 hrs depending on TF and feature count

---

### Step 4: CPCV Training

**What it does:** Trains the production LightGBM model using Combinatorial Purged Cross-Validation (CPCV) with K=2 test groups. Reads Optuna-tuned params from `optuna_configs_{TF}.json`. Trains on full sparse CSR matrix (base features + cross features). Saves in-sample metrics (accuracy + Sharpe) per fold for PBO validation. Uses asymmetric triple-barrier labels to fix SHORT precision (1.5x ATR profit / 3x ATR stop for SHORT on upward-biased BTC).

**Script:** `ml_multi_tf.py --tf {TF}`

**Called by:** `cloud_run_tf.py`

**RAM/CPU/GPU:**
- Loads full sparse cross matrix into RAM
- CPU: All cores via LightGBM num_threads=0
- GPU: LightGBM CPU-only (GPU doesn't support sparse CSR with EFB)
- RAM: Same as cross gen peak -- the sparse matrix must fit in RAM
- Sequential CPCV for >1M features (parallel CPCV abandoned due to pickle bottleneck)

**TF-specific caveats:**

| TF | CPCV Groups (N,K) | Splits | Unique Paths | Train % | Notes |
|----|-------------------|--------|-------------|---------|-------|
| 1w | (5, 2) | 10 | 4 | 60% | Fastest. ~10-30 min. |
| 1d | (5, 2) | 10 | 4 | 60% | ~1-3 hrs |
| 4h | (6, 2) | 15 | 5 | 67% | ~2-6 hrs |
| 1h | (6, 2) | 15 | 5 | 67% | ~4-12 hrs |
| 15m | (6, 2) | 15 | 5 | 67% | ~8-24 hrs. int64 indptr required. |

**Critical parameters (from config.py V3_LGBM_PARAMS):**
- max_bin=255 (maximum EFB compression; binary features still get 2 bins)
- feature_pre_filter=False (protects rare esoteric features)
- is_enable_sparse=True (direct sparse CSR input)
- deterministic=True (reproducible sparse training)
- class_weight='balanced' for 1d/1w (fixes multiclass imbalance)

**Known issues:**
- NEVER use XGBoost -- accuracy dropped 12% when accidentally swapped. LightGBM EFB is architecturally critical.
- Sparse CSR + EFB: structural zero = 0.0 (feature OFF, correct for binary crosses). No NaN in cross features.
- Model backed up to `model_{TF}_cpcv_backup.json` immediately after training

**Output:** `model_{TF}.json`, `model_{TF}_cpcv_backup.json`, `cpcv_oos_predictions_{TF}.pkl`, `feature_importance_stability_{TF}.json`, `ml_multi_tf_configs.json`

**Post-step verification:** cloud_run_tf.py checks train log for "Features:" line containing "SPARSE" or "DENSE" -- pipeline aborts if crosses were not loaded.

**Estimated time:** 10 min - 24 hrs depending on TF

---

### Step 5: Trade Optimizer

**What it does:** Exhaustive search over 13-dimensional trade parameter space: entry thresholds, exit thresholds, stop-loss, take-profit, position sizing, hold periods, etc. Uses 500 Optuna TPE trials with Sortino objective.

**Script:** `exhaustive_optimizer.py --tf {TF}`

**Called by:** `cloud_run_tf.py` (non-critical -- pipeline continues on failure)

**RAM/CPU/GPU:** Moderate CPU. Loads model predictions, not full feature matrix.

**TF-specific caveats:** Hold period ranges differ by TF (1w has longer holds than 15m).

**Output:** Updated `ml_multi_tf_configs.json` with optimized trade parameters

**Estimated time:** 10-60 min

---

### Steps 6, 7, 8: Parallel Execution (Meta-labeling, LSTM, PBO)

These three steps run in parallel via ThreadPoolExecutor. All depend only on Step 4 output.

#### Step 6: Meta-labeling

**What it does:** Trains a secondary model that predicts whether the primary model's signal will be correct. Used to filter low-confidence trades. Saves a calibrated probability model (Platt scaling).

**Script:** `meta_labeling.py --tf {TF}`

**RAM/CPU/GPU:** Moderate. Loads OOS predictions from CPCV, not full feature matrix.

**Output:** `meta_model_{TF}.pkl`, `platt_{TF}.pkl`

**Estimated time:** 5-30 min

#### Step 7: LSTM Ensemble

**What it does:** Trains an LSTM sequence model on the same features as a secondary predictor. Provides temporal pattern recognition that tree-based models miss. Alpha search aligns LSTM validation set with exact CPCV OOS samples.

**Script:** `lstm_sequence_model.py --tf {TF} --train`

**RAM/CPU/GPU:**
- GPU preferred (RTX 3090 locally, cloud GPU)
- H200 has weak CPU -- bottlenecks DataLoader. Use local 13900K+3090 for LSTM.
- Must impute NaN to 0 AFTER z-score normalization (LightGBM handles NaN but LSTM cannot)

**Known issues:**
- NaN features crash LSTM -- mandatory NaN imputation in preprocessing
- H200 CPU too weak for DataLoader -- train LSTM locally

**Output:** `lstm_{TF}.pt`

**Estimated time:** 15-120 min depending on GPU

#### Step 8: PBO Validation

**What it does:** Probability of Backtest Overfitting validation using in-sample metrics from CPCV folds. Requires K>=2 for meaningful combinatorial paths (K=1 gives only 1 path -- useless).

**Script:** `backtest_validation.py --tf {TF}`

**RAM/CPU/GPU:** Minimal. Uses saved fold metrics only.

**Known issues:**
- Previous runs with K=1 had no meaningful PBO. Now using K=2 across all TFs.

**Output:** PBO metrics in validation report

**Estimated time:** 2-10 min

---

### Step 9: Backtesting Audit

**What it does:** Comprehensive audit of the trained model's backtest performance. Depends on Step 5 optimizer output.

**Script:** `backtesting_audit.py --tf {TF}`

**Called by:** `cloud_run_tf.py` (non-critical)

**RAM/CPU/GPU:** Moderate CPU.

**Output:** `audit_{TF}.log`

**Estimated time:** 5-15 min

---

### Step 10: SHAP Analysis

**What it does:** Analyzes feature importance using split importance and gain scores from the trained LightGBM model. Cannot use pred_contrib (SHAP values) because .toarray() OOMs on 2.9M+ sparse crosses. Reports: active features (split > 0), cross vs base importance ratio, top 20 feature families by gain, top 50 individual features.

**Script:** Inline in `cloud_run_tf.py` (lines 608-682)

**RAM/CPU/GPU:** Loads model only (not full matrix). Moderate RAM.

**Known issues:**
- Full SHAP (pred_contrib) impossible on sparse crosses due to OOM on .toarray()
- Uses split/gain importance as proxy -- sufficient for family-level analysis

**Output:** `shap_analysis_{TF}.json`

**Estimated time:** 1-5 min

---

### Step 11: Final Artifact Verification

**What it does:** Lists all expected artifacts with sizes, checks for MISSING files, writes `DONE_{TF}` marker if zero failures.

**Script:** `cloud_run_tf.py` `_print_summary()` (lines 102-134)

**Expected artifacts:**
- `model_{TF}.json` -- Production LightGBM model
- `model_{TF}_cpcv_backup.json` -- Backup of production model
- `optuna_configs_{TF}.json` -- Tuned hyperparameters
- `meta_model_{TF}.pkl` -- Meta-labeling model
- `platt_{TF}.pkl` -- Platt calibration
- `lstm_{TF}.pt` -- LSTM ensemble model
- `features_{TF}_all.json` -- Feature list
- `cpcv_oos_predictions_{TF}.pkl` -- OOS predictions for PBO
- `v2_crosses_BTC_{TF}.npz` -- Sparse cross feature matrix
- `v2_cross_names_BTC_{TF}.json` -- Cross feature names
- `features_BTC_{TF}.parquet` -- Base features
- `feature_importance_stability_{TF}.json` -- Feature importance
- `shap_analysis_{TF}.json` -- SHAP report
- `ml_multi_tf_configs.json` -- Training + trade config
- `inference_{TF}_thresholds.json` -- Live inference thresholds
- `inference_{TF}_cross_pairs.npz` -- Live inference cross pairs
- `inference_{TF}_ctx_names.json` -- Live inference context names
- `inference_{TF}_base_cols.json` -- Live inference base columns
- `inference_{TF}_cross_names.json` -- Live inference cross names

**Output:** `DONE_{TF}` marker file (only if zero failures)

---

## Cross Gen Deep Dive (Step 2)

This step has been the single most problematic part of the pipeline. It OOM'd 6 times on a 755GB machine before being solved. The fundamental challenge: materializing millions of cross-features from sparse binary multiplications while keeping RAM bounded.

### The 3-Level Flush Architecture

Cross generation accumulates data at three nested levels. Each level has its own flush mechanism to prevent unbounded RAM growth.

**Level 1: COO Lists (innermost)**
Inside `_gpu_cross_chunk()` / `_cpu_cross_chunk()`, individual cross features are computed as COO (coordinate format) arrays -- row indices, column indices, and data values. These arrays grow with every feature pair.

- Flush trigger: Every `FLUSH_FEATS` features
- Formula: `FLUSH_FEATS = max(5000, min(50000, int(ram_gb * 50)))`
- Example: 755GB RAM -> ~37,750 features between flushes
- Action: `_flush_coo_to_csr()` converts accumulated COO arrays into a single CSR matrix and frees the COO arrays

**Level 2: CSR Chunks (middle)**
After COO-to-CSR conversion, CSR matrices accumulate inside the chunk function. These can collectively consume hundreds of GB.

- Flush trigger: Every `MAX_CSR_IN_RAM` CSR chunks
- Formula: `MAX_CSR_IN_RAM = max(2, min(5, int(ram_gb / 300)))`
- Example: 755GB RAM -> 2 CSR chunks max before flush; 2TB RAM -> 5 chunks
- Action: `_flush_csr_to_disk()` writes CSR chunks to numbered NPZ files on disk and frees RAM

**Level 3: RIGHT_CHUNK Accumulation (outermost)**
The outer `gpu_batch_cross()` function processes the right-side context array in chunks of `RIGHT_CHUNK` columns. Each chunk produces a CSR result that accumulates.

- Flush trigger: `MAX_CHUNKS_IN_RAM` threshold
- Formula: `MAX_CHUNKS_IN_RAM = max(100, ...)`
- Action: `_flush_chunks_to_disk()` writes accumulated chunks as sub-checkpoint NPZ files

**Final Assembly: Streaming CSC Splice**
After all crosses are generated (possibly across dozens of disk-backed NPZ files), `_streaming_csc_splice()` loads one checkpoint at a time, converts to CSC, and splices columns together. Never loads all checkpoints into RAM simultaneously. Uses int64 indptr for 15m (NNZ > 2^31).

### Thread Count Formula

CPU threads for cross multiplication must balance parallelism against RAM pressure from thread intermediates.

```
per_worker_gb = N_rows * 8 * 6 / 1e9  (6 arrays per worker: left_cols, right_cols, product, rows, cols, data)
n_threads = ram_gb * 0.10 / per_worker_gb  (reserve 90% for data accumulation + OS)
```

**Why 0.10 (not 0.40)?** Thread intermediates (left_cols, right_cols, crosses arrays) compete with COO/CSR accumulation for RAM. Old formula (0.40) OOM'd on cgroup-limited machines where physical RAM != available RAM.

**Cgroup awareness:** `_get_available_ram_gb()` reads `/sys/fs/cgroup/memory.max` (not physical RAM). vast.ai sets cgroup limits ~5-10% below physical (e.g., 1032GB physical = 967GB cgroup).

### BATCH Size Formula

BATCH controls how many feature pairs are computed per thread invocation.

```
BATCH = min(MAX_BATCH, max(500, n_pairs // n_threads))
target: BATCH = target_ram / (n_threads * N_rows * 8 * 6)
```

Key insight: **Smaller batches = more threads = faster windows + faster merge = lower peak RAM.** On a 512-core machine with 967GB cgroup, optimal was 128 threads x ~5.5K BATCH (not 14 threads x 50K BATCH).

### GPU vs CPU Path Selection

```python
if cupy_available and cp.cuda.runtime.memGetInfo()[1] / 1024**3 >= 20:
    # GPU path: CuPy sparse matmul for pre-filter + batch multiply
else:
    # CPU path: Numba prange for parallel feature multiplication
```

**20GB VRAM threshold:** GPUs with <20GB are SLOWER than multi-core CPU for cross gen (tiny VRAM = tiny batches = more kernel launch overhead than computation).

GPU path uses:
- CuPy sparse matmul for co-occurrence pre-filter (`left_sp.T @ right_sp`)
- CuPy element-wise multiply for cross computation
- cuSPARSE SpGEMM on supported GPUs

CPU path uses:
- scipy sparse matmul for pre-filter
- Numba @njit with prange for parallel cross multiplication
- ThreadPoolExecutor for batch dispatch

**CUDA 13+ (driver 580+) caveat:** cuDF/CuPy compiled for CUDA 12.x SEGFAULT on CUDA 13. CuPy works with `CUPY_COMPILE_WITH_PTX=1` env var (PTX JIT compilation). cuSPARSE on small GPUs (RTX 3060 Ti, 8GB) still crashes even with PTX.

### Per-TF RAM Estimates

| TF | Rows | Est. Features | Peak RAM (cross gen) | Peak RAM (training) | RIGHT_CHUNK |
|----|------|--------------|---------------------|--------------------|-|
| 1w | 818 | ~2.2M | ~370GB | ~64GB | 2000 (or env override) |
| 1d | 5,700 | ~6M | ~500GB | ~128-256GB | 1000-2000 |
| 4h | 23,000 | ~5M | ~600GB | ~256-512GB | 500-1000 |
| 1h | 91,000 | ~8M | ~1.5TB | ~512GB-1TB | 500 |
| 15m | 227,000 | ~10M+ | ~2TB+ | ~768GB-1.5TB | 500 (forced via env) |

### The 8 Bugs Found and Fixed (2026-03-29 Session)

| # | Bug | Root Cause | Fix | Files |
|---|-----|-----------|-----|-------|
| 1 | COO lists grew unbounded | No periodic flush in _gpu_cross_chunk | `_flush_coo_to_csr()` every FLUSH_FEATS features | v2_cross_generator.py |
| 2 | CSR chunks accumulated in RAM | No disk flush between COO conversions | `_flush_csr_to_disk()` every MAX_CSR_IN_RAM chunks | v2_cross_generator.py |
| 3 | RIGHT_CHUNK accumulation OOM | gpu_batch_cross stored all chunks in memory | `_flush_chunks_to_disk()` with MAX_CHUNKS_IN_RAM | v2_cross_generator.py |
| 4 | Final assembly loaded all NPZs | _streaming_csc_splice loaded everything at once | Stream one checkpoint at a time, free before next | v2_cross_generator.py |
| 5 | CuPy Blackwell sm_120 crash | CuPy couldn't JIT for sm_120 architecture | `CUPY_COMPILE_WITH_PTX=1` env var | 6 files |
| 6 | scipy 2.x `len(csr_matrix)` error | scipy 2.x deprecated len() on sparse matrices | Detect CSR via `hasattr(rows_list[0], 'indptr')` | v2_cross_generator.py |
| 7 | Thread count too aggressive | Old formula (0.40) OOM'd on cgroup-limited machines | Reduced to 0.10, added cgroup-aware RAM detection | v2_cross_generator.py |
| 8 | int64 indptr overflow at 15m | 227K rows x 10M features exceeds int32 NNZ limit (2^31) | int64 indptr in _streaming_csc_splice | v2_cross_generator.py |

---

## TF-Specific Caveats Table

| Issue | 1w | 1d | 4h | 1h | 15m |
|-------|----|----|----|----|-----|
| Disk flush architecture needed | No | Yes | Yes | Yes | Yes |
| int64 indptr required | No | No | No | Maybe | Yes |
| Memmap/tcmalloc needed for cross gen | No | No | No | Yes | Yes |
| RIGHT_CHUNK env override | No | No | No | Recommended | Mandatory (500) |
| BATCH_MAX env override | No | No | No | Recommended | Mandatory (500) |
| CPCV sequential (pickle bottleneck) | No (parallel OK) | Yes | Yes | Yes | Yes |
| GPU path viable (cross gen) | Yes | Yes | Yes | Depends on VRAM | Depends on VRAM |
| class_weight='balanced' | Yes | Yes | No | No | No |
| LSTM: train locally (H200 CPU weak) | Recommended | Recommended | Recommended | Yes | Yes |
| EFB bundle viable | Yes | Yes | Yes | Maybe (10M features) | Maybe (10M features) |
| min_data_in_leaf | 5 | 5 | 5 | 8 | 15 |
| CPCV groups (N, K) | (5, 2) | (5, 2) | (6, 2) | (6, 2) | (6, 2) |
| num_leaves cap | 31 | 127 | 255 | 511 | 511 |
| Cross gen OMP_NUM_THREADS | 4 | 4 | 4 | 4 | 4 |
| NPZ min cols (stale check) | 1.5M | 5M | 3M | 5M | 5M |

---

## Machine Requirements Table

| TF | Min RAM | Recommended RAM | Min Cores | Recommended Cores | GPU Needed | Est. Total Time | Est. Cost |
|----|---------|----------------|-----------|-------------------|-----------|-----------------|-----------|
| 1w | 64 GB | 512 GB | 32 | 128+ | No (helpful) | 2-6 hrs | $2-10 |
| 1d | 128 GB | 512 GB | 64 | 256+ | No (helpful) | 6-18 hrs | $8-30 |
| 4h | 256 GB | 1 TB | 64 | 256+ | No (helpful) | 8-24 hrs | $10-40 |
| 1h | 512 GB | 2 TB | 128 | 256+ | Optional | 18-48 hrs | $20-60 |
| 15m | 768 GB | 2 TB+ | 128 | 512+ | Optional | 36-96 hrs | $40-120 |

**Machine selection priority:** CPU Score (cores x base GHz) > RAM > GPU. Cross gen and LightGBM training are CPU-bound. GPU helps only for feature builds (cuDF) and cross gen pre-filter (CuPy sparse matmul).

**Cloud provider notes:**
- vast.ai: Sets cgroup limits ~5-10% below advertised RAM. Always check `memory.max`.
- vast.ai machines can die without warning -- download partial results after each critical step.
- Any provider works: vast.ai, RunPod, Lambda, GCP, Azure. Use pip+SCP deployment, not custom Docker.
- NEVER rent a slower machine than what was already being used. Fix issues in-place.

---

## Cloud Deployment Checklist

1. Rent machine with lightweight cached image (e.g., `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`)
2. SSH in, install deps (see Step 0)
3. Test ALL imports: `python -c "import pandas, numpy, scipy, sklearn, lightgbm, ephem, astropy, pyarrow, optuna, numba, hmmlearn, yaml, tqdm; print('ALL OK')"`
4. SCP code tar (~11MB) + DB tar separately
5. Extract, symlink DBs: `ln -sf /workspace/*.db /workspace/v3.3/`
6. Verify DB count: `ls /workspace/*.db | wc -l` must be >= 16
7. Run: `cd /workspace/v3.3 && python -u cloud_run_tf.py --symbol BTC --tf {TF}`
8. Monitor: `tail -f /workspace/train_{TF}.log`
9. Verify multi-threaded execution: load average > cores x 0.3 within 60s of launch
10. Download ALL artifacts before destroying machine

---

## Performance Optimization Notes

**Memory allocator:** tcmalloc (google-perftools) via LD_PRELOAD. Intercepts numpy, LightGBM C++, Numba allocations. 5-15% speedup.

**THP (Transparent Huge Pages):** `enabled=always` + `defrag=defer+madvise`. NOT `defrag=always` (causes multi-second stalls during LightGBM histogram phase). 5-20% speedup.

**NUMA binding:** Multi-socket machines need numactl. Single-process: `numactl --interleave=all`. Parallel Optuna: one worker per NUMA node. 10-30% speedup on multi-socket.

**tcmalloc + fork() danger:** Deadlock risk if forking after LightGBM initializes. Use spawn, not fork.

**Combined speedup:** 20-40% wall-time on multi-socket machines.

---

## Architecture Invariants (Non-Negotiable)

1. **LightGBM only.** Never XGBoost. EFB on sparse CSR is the only architecture that handles millions of binary cross features efficiently.
2. **No feature filtering.** No MI, no variance filter, no support thresholds. The model decides via tree splits. `feature_pre_filter=False` always.
3. **No subsampling.** All rows, all features, all signals. Row-partitioned boosting was REJECTED (kills rare signals).
4. **No fallbacks.** No TA-only mode, no base-only mode. Full pipeline or crash.
5. **Sparse CSR throughout.** Dense conversion only where absolutely required (LSTM input). Cross features stored as sparse NPZ.
6. **4-tier binarization on everything.** TA, gematria, sentiment, astro positions, numerology values -- all get low/mid_low/mid_high/high tiers.
7. **NaN semantics preserved.** NaN = missing (LightGBM learns split direction). 0 = value is zero. Structural zero in CSR = feature OFF (correct for binary crosses). Never fillna(0) on features.
8. **max_bin=255.** Maximum EFB compression. Binary features get 2 bins regardless. Lower max_bin forces excessive bundle fragmentation.
9. **Co-occurrence filter = 3.** Matches min_data_in_leaf for 1d/1w. Ensures every cross feature has enough samples for at least one leaf.
10. **The edge IS the matrix.** Same sky, same calendar, same energy for all assets. More diverse signals = stronger predictions. Esoteric signals are the edge, never regularize them away.
