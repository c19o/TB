# V2 Project Rules & Lessons Learned

## Codebase Intelligence (Socraticode)

Socraticode MCP server provides semantic search over the indexed codebase:
- `codebase_search` — Hybrid semantic + keyword search across all source files
- `codebase_graph_query` — Import/dependency analysis (circular deps, import chains)
- `codebase_graph_visualize` — Mermaid diagrams of dependency relationships
- `codebase_context_search` — Search indexed artifacts (config, feature library, cross generator)
- `codebase_update` — Trigger incremental re-index after major changes
- `codebase_watch` — File watcher keeps index current automatically

First session after setup: run `codebase_index` to build the initial index.

## STRUCTURED AUDIT PIPELINE — AGENT-BASED

When the user says "audit", "audit the system", or "full audit", execute the following multi-pass pipeline. Each pass is a SCOPED agent task. Do NOT combine passes. Do NOT fix things outside the current pass scope. Launch passes in parallel where marked.

### CASCADE DETECTION RULE (applies to EVERY pass that makes changes)
After ANY fix in ANY pass, BEFORE moving to the next checklist item, run a cascade check:
1. **Signature changes** (added/removed args, changed return tuple length):
   - Grep ALL callers of the modified function across v2/*.py
   - Verify every call site matches the new signature
   - Verify every unpacking site matches the new return shape
2. **Filename/path changes** (renamed files, moved artifacts, changed extensions):
   - Grep ALL consumers: import statements, open(), os.path.join(), config references
   - Verify every consumer uses the new name/path
3. **Config value changes** (moved to config.py, changed default, renamed key):
   - Grep ALL readers of the old name/value across v2/*.py
   - Verify every reader uses the new config.KEY, not a stale local default
4. **Interface contract changes** (column names, dict keys, class attributes):
   - Trace the data flow: producer → intermediate → consumer
   - Verify the contract holds at every stage

**If a cascade fix itself triggers another cascade, follow the chain until stable.**
Log every cascade chain in the pass output:
```
FIX: renamed foo() return from 3→5 values
  CASCADE: bar.py:42 unpacks foo() — updated
  CASCADE: baz.py:88 unpacks foo() — updated
    CASCADE-2: baz.py:90 passes result to qux() — verified compatible
```

### MATRIX PHILOSOPHY GATE (PASS 0 — always first, blocks everything)
Verify every file in v2/ against these non-negotiable rules. ANY violation = immediate fix.
Apply CASCADE DETECTION RULE after each fix.
- [ ] No feature filtering/pre-screening before LightGBM (no MI, no variance filter, no support thresholds)
- [ ] No fallback modes (no TA-only, no base-only, no graceful degradation)
- [ ] No fillna(0) on feature data (NaN = missing signal, 0 = "value is zero")
- [ ] No blanket try/except that masks failures (crash > silent degradation)
- [ ] Esoteric features protected from regularization/pruning (SHAP, Optuna, CPCV)
- [ ] Every asset gets full pipeline (esoteric + astro + gematria + numerology + space weather)
- [ ] Sparse CSR preserves NaN semantics (structural zero = missing, explicit NaN = missing, explicit 0.0 = WRONG)
- [ ] V2 cross + V2 layers mandatory in live_trader (no silent fallback to base-only)
- [ ] min_data_in_leaf respects sparse signal frequency (1d/1w=3, 4h=5, 1h=8, 15m=15)

### PASS 1 — Dead Code & Abandoned Experiments (agent 1, parallel with pass 2)
Use `codebase_search` + `codebase_graph_query` to find:
Apply CASCADE DETECTION RULE after each fix (deleting a function may break an import elsewhere).
- [ ] Unused imports across all v2/*.py
- [ ] Functions/classes defined but never called (grep for def, check callers via graph)
- [ ] Commented-out code blocks (>3 lines)
- [ ] Variables assigned but never read
- [ ] Files in v2/ that nothing imports
- [ ] Old v1 references in v2 code
- DELETE dead code. Don't comment it out. Git has history.

### PASS 2 — Config & Hardcoded Values (agent 2, parallel with pass 1)
Use `codebase_search("hardcoded", "magic number", "threshold")` to find:
Apply CASCADE DETECTION RULE after each fix (moving a value to config.py means every reader must update).
- [ ] Magic numbers not in config.py (thresholds, window sizes, fees, retry counts)
- [ ] Paper/live mode divergence (different code paths, different defaults)
- [ ] Environment-specific paths hardcoded (C:\Users\..., /root/..., etc.)
- [ ] API keys/secrets in source code (should be in .env)
- [ ] Inconsistent defaults between files (e.g., confidence threshold 0.6 in one file, 0.65 in another)
- All tunable values → config.py. All secrets → .env. All paths → config.py with env override.

### PASS 3 — Cross-File Consistency (agent 3, after pass 0)
Use `codebase_graph_query` for import chains + `codebase_search` for interface contracts:
Apply CASCADE DETECTION RULE after each fix.
- [ ] Feature column names match between builder → trainer → live_trader
- [ ] Sparse matrix shapes align (base features + cross features + layers)
- [ ] Config values used consistently (same asset list, same TF list, same paths)
- [ ] Error handling consistent (all crash-on-failure, no mixed strategies)
- [ ] Data types consistent across pipeline stages (cuDF vs pandas, float32 vs float64)
- [ ] NaN handling consistent (never converted to 0, always preserved for LightGBM)
- [ ] Function signatures match between definition and ALL call sites
- [ ] Return value tuple lengths match between producer and ALL unpacking sites
- [ ] Artifact filenames match between writer (training) and reader (inference/audit)

### PASS 4 — GPU & Performance (agent 4, after pass 1+2)
Apply CASCADE DETECTION RULE after each fix.
- [ ] No CPU fallbacks where GPU path exists (no silent sklearn instead of cuML)
- [ ] No .apply() or Python for-loops on arrays (Numba @njit or vectorized)
- [ ] No one-at-a-time df[col]=val (batch with dict + pd.concat)
- [ ] Batch sizes respect VRAM (auto-adapt, not hardcoded)
- [ ] Memory cleanup between builds (del + gc.collect)
- [ ] CUDA_VISIBLE_DEVICES not pinned per-process (let all GPUs be visible)

### PASS 5 — Live Trading Safety (agent 5, after pass 3)
Apply CASCADE DETECTION RULE after each fix.
- [ ] Kill switch exists and works
- [ ] Max position size enforced
- [ ] Stale data detection (if features older than X bars, halt)
- [ ] Order rejection handling (retry logic, not silent fail)
- [ ] Paper/live use identical inference path (only broker adapter differs)
- [ ] Logging captures EVERYTHING (entry reason, feature snapshot, model confidence, exit reason)
- [ ] No lookahead bias in live feature computation

### PASS 6 — Cascade Regression + Integration Smoke Test (sequential, last)
This pass exists specifically to catch cascades that slipped through per-fix checks.
- [ ] Import every v2 module — no ImportError
- [ ] Config loads without error
- [ ] Feature builder can initialize (not full build — just import + config parse)
- [ ] Live trader can initialize in paper mode (not trade — just startup)
- [ ] All file paths in config.py actually exist or have creation logic
- [ ] **Signature audit**: For every function modified in passes 0-5, grep all callers and verify args + return unpacking
- [ ] **Filename audit**: For every file renamed/moved in passes 0-5, grep all references and verify paths
- [ ] **Config audit**: For every config value added/moved in passes 0-5, grep all readers and verify they use config.KEY
- [ ] If ANY cascade found here → fix it, then re-run Pass 6 until clean (max 3 iterations)

### AUDIT OUTPUT
After all passes, produce a single summary:
```
AUDIT RESULTS — [date]
Pass 0 (Philosophy): X violations found, X fixed, X cascades traced
Pass 1 (Dead Code): X items removed, X cascades traced
Pass 2 (Config): X hardcoded values moved, X cascades traced
Pass 3 (Consistency): X mismatches fixed, X cascades traced
Pass 4 (GPU/Perf): X CPU fallbacks eliminated, X cascades traced
Pass 5 (Live Safety): X gaps closed, X cascades traced
Pass 6 (Regression): PASS/FAIL (iteration count), X late cascades caught

CASCADE CHAINS (if any):
- [original fix] → [cascade 1] → [cascade 2] → stable

REMAINING ISSUES (if any):
- [issue]: [why it can't be auto-fixed, needs user decision]
```

## RULES (NON-NEGOTIABLE)

### Philosophy
- The matrix is UNIVERSAL — same sky, same calendar, same energy for ALL assets
- Every asset gets the FULL pipeline (esoteric, astro, gematria, numerology, space weather)
- NO FILTERING of features. LightGBM decides via tree splits, not us
- NO FALLBACKS. One pipeline for all. If it breaks, fix it
- Esoteric signals ARE the edge. Never regularize them away
- More diverse signals = stronger predictions. The edge is the matrix

### Process
- NEVER deviate from the plan. Follow it exactly as written
- Always make a checklist and verify each step before moving to the next
- Always keep this file updated with lessons learned
- Always run scripts with progress logs (tee/unbuffered). Never run blind
- NEVER kill processes without explicit user permission
- Don't search user's PC for files. Use the sources they provide
- Use Perplexity MCP for technical/parameter queries. For esoteric content, use the local vector DB at Orgonite master instead.
- Stagger feature builds: small TFs parallel, then 15m, then 5m solo
- Build features on GPU (cuDF rolling/ewm) + cloud for parallelism. Train on cloud GPU
- Always maximize parallelism. Launch multiple agents simultaneously

### Data & Deployment Integrity (NON-NEGOTIABLE)
- **NEVER deploy with missing data, missing code, or untested changes.** Every shortcut weakens the model. If ANYTHING is incomplete, STOP and fix it first.
- **NEVER deploy with missing databases.** ALL .db files must be in the upload tar. Missing DB = missing features = weaker model = INVALID RUN. Kill it immediately.
- **NEVER treat "WARNING" log messages as acceptable.** Every warning is a lost signal or a silent failure. Investigate and fix before proceeding.
- **NEVER say "it's fine" about missing data.** The matrix requires ALL data sources. Missing ANY source violates the core philosophy.
- **NEVER skip the smoke test.** If smoke test fails, the machine is incompatible. Do NOT try workarounds — find a compatible machine.
- **NEVER deploy code that hasn't been audited.** 3 clean audit passes before any cloud deploy.
- **NEVER assume a fix works without verifying.** Test the actual pipeline path, not just individual components.
- **Upload tar must include EVERY .db file:** btc_prices.db, tweets.db, news_articles.db, sports_results.db, space_weather.db, onchain_data.db, macro_data.db, astrology_full.db, ephemeris_cache.db, fear_greed.db, funding_rates.db, google_trends.db, kp_history_gfz.txt
- **Verify DB count after extract:** `ls /workspace/*.db | wc -l` must be >= 12. If not, STOP.
- **Verify zero "WARNING: DB missing" in first 30s of pipeline log.** If any appear, STOP.
- **NEVER compromise training speed.** Pick machines with the highest CPU Score (cores × base GHz). Match CUDA image to driver version. Use GPU (cuDF/CuPy) for feature building. Never fall back to CPU-only when GPU is available.
- **NEVER use bandaids that hurt performance.** If a fix slows training (e.g., disabling GPU, reducing batch size, limiting cores), it's a bandaid. Find the real fix.
- **ACTIVE monitoring is MANDATORY.** When machines are training, check monitor output every cycle. If ANY log shows FAIL/CRITICAL/Error, FIX IT IMMEDIATELY — don't wait for the user to ask. A monitor that runs but isn't acted on is useless.
- **Monitor must detect failures, not just liveness.** Check for "FAIL", "CRITICAL", "Error", "Traceback" in logs — not just whether the process is alive. A dead process with errors needs immediate action.
- **Monitor must verify MULTI-THREADED execution.** After launch, check load average > cores × 0.3. If load ≈ 1.0 on a 128+ core machine, training is single-threaded — this is a critical bug. Check RSS matches expected dense matrix size.
- **After fixing a bug, grep ALL files for the SAME pattern.** The .nnz bug was fixed in one place but existed in another. The parquet symlink bug hit 1w, then 1h, then 1d, then 4h — same bug, never properly fixed at the root.
- **Pre-flight code audit before EVERY deploy.** Run the pre-flight checklist in CLOUD_TRAINING_PROTOCOL.md. Never deploy code that hasn't been verified against the checklist. The single-threaded bug (v3.3) could have been caught by checking load average within 60 seconds of launch.
- **STALE PARQUET CHECK is MANDATORY.** If feature_library.py changed since parquets were built, DELETE old parquets AND NPZs. The pipeline skips rebuild when parquet exists with >2000 cols — it does NOT detect new features. cloud_run_tf.py now has a v3.3 fingerprint check, but ALWAYS verify parquet col count matches expected after any feature_library.py change.
- **VERIFY DATA DIRECTORY BEFORE EVERY RUN.** config.py's V30_DATA_DIR defaults to `v3.0 (LGBM)/` which has OLD data. Cloud sets V30_DATA_DIR=/workspace. Locally, MUST set `export V30_DATA_DIR=/path/to/v3.2_2.9M_Features` or training silently uses v3.0 features. ALWAYS check the "Loaded from parquet:" log line — if it shows v3.0 path, STOP.
- **max_bin=15 for binary sparse features.** Our cross features are binary (0/1) — they only use 2 bins regardless of max_bin. Setting max_bin=63 caused 4x slowdown with zero accuracy benefit (Perplexity confirmed). Keep max_bin=15.
- **max_conflict_rate=0.3 NOT 0.0.** Setting 0.0 DISABLES Exclusive Feature Bundling (EFB) which is LightGBM's core sparse optimization. EFB bundles mutually exclusive features to dramatically reduce histogram work. 0.3 allows bundling with up to 30% conflict rate — safe for our binary crosses.
- **15m uses XGBoost (not LightGBM)** when NNZ > 2B. LightGBM has int32 index overflow. XGBoost supports int64 natively. All 217K rows × 10M+ features preserved. Needs 1.5-2 TB RAM machine.
- **CHECK ALL CONFIG PATHS IN FIRST 10 LOG LINES.** DB_DIR, V30_DATA_DIR, SAVAGE22_DB_DIR, PROJECT_DIR — verify all point to correct directories. Wrong paths = training on wrong/stale data = wasted time and money.
- **NEVER pip/conda install packages that touch CUDA deps in RAPIDS containers.** Both pip and conda can upgrade cuda-python/cuda-bindings to versions incompatible with cudf, PERMANENTLY breaking the environment. SAFE onstart-cmd: `pip install --no-cache-dir --no-deps lightgbm optuna ephem && pip install --no-cache-dir alembic cmaes colorlog sqlalchemy tqdm PyYAML joblib threadpoolctl`. The --no-deps flag prevents pip from resolving CUDA dependencies. scipy + scikit-learn + numpy are ALREADY in the RAPIDS container.

### Code
- No fallback modes. No TA-only mode. Full pipeline or fix it
- Batch column assignment (pd.concat / dict accumulation), NEVER one-at-a-time df[col]=val
- Always use GPU (RTX 3090) for processing, never default to CPU
- 4-tier binarization on ALL numeric columns (gematria, sentiment, astro, TA, everything)
- Sparse CSR for cross features — never create dense DataFrames for 100K+ columns
- NEVER convert NaN to 0 in sparse matrices — NaN is "missing" (LightGBM learns split direction), 0 is "the value is zero" (different signal). Structural zeros in CSR = missing. Explicit NaN = missing. Explicit 0.0 = stored bloat + wrong semantics
- Use native cuDF (not cudf.pandas) for compute_ta_features — keeps rolling ops on GPU
- Stateful loops must use Numba @njit — never raw Python for-loops on price arrays
- Cloud docker: rapidsai/base (cuDF+CuPy pre-installed), pip install lightgbm+optuna+scipy+scikit-learn+ephem (numba comes with RAPIDS, don't pip install separately)
- LightGBM CUDA does NOT support sparse — always use device="cpu" with force_col_wise=True
- min_data_in_leaf=3 (1d/1w) with min_gain_to_split=2.0 as compensating guard for rare signals
- Co-occurrence filter of 8 on cross features (math constraint: <8 can't appear in both CPCV splits)
- Optuna replaces exhaustive 30M grid (200 TPE trials for LightGBM params, 500 for trade optimizer (13D search space), Sortino objective)

## LESSONS LEARNED

### GPU Acceleration of Feature Pipeline (2026-03-21)
- ALL 16 compute functions GPU-native — zero _to_cpu() conversions remain
- All rolling/ewm/shift ops on cuDF GPU across entire pipeline
- External DataFrames (astro, tweets, news, space weather) converted to cuDF before merge
- `.map(dict)` replaced with cuDF merge pattern (cuDF doesn't support .map with dicts)
- `.reindex()` + `.ffill()` used separately (cuDF doesn't support `reindex(method='ffill')`)
- cuML KNeighborsClassifier replaces sklearn KNN (50x faster on GPU)
- CuPy convolve replaces np.convolve for fractional differentiation
- _bars_since_event compiled with Numba @njit (forward scan pattern)
- _rolling_percentile_vec compiled with Numba @njit (91.6M loop iterations eliminated)
- Cross generator batch size auto-adapts to GPU VRAM (A100: BATCH=200, 3090: BATCH~20)
- Stateful loops (SAR, Supertrend, Wyckoff, Volume Profile, Elliott, Gann) compiled with Numba @njit
- Shannon entropy and Hurst vectorized with numpy sliding_window_view (no per-bar loops)
- Intraday builds parallelized with ProcessPoolExecutor + CUDA_VISIBLE_DEVICES per worker
- CPCV training splits parallelized across GPUs (--parallel-splits flag)
- Cloud docker: rapidsai/base (cuDF+CuPy+XGBoost+cuML pre-installed)
- On cloud, use `--parallel 0` flag to auto-detect GPU count for worker routing

### Pandas Column Assignment Bottleneck (2026-03-21)
- `df[col] = val` called 135K+ times is the #1 build bottleneck
- Each assignment triggers DataFrame internal reindex/copy overhead
- On a 2.2GHz cloud CPU, this dominated build time (GPU sat idle)
- FIX: Accumulate in dict, then `pd.DataFrame(dict, index=df.index)` + `df[new.columns] = new`
- This alone cuts build time by ~60%
- Applied to: ALL trend cross sections (tx_, ex_, px_, vwap, range, DOY x ALL)
- Context matrix chunked into groups of 200 to avoid 8.7GB column_stack OOM

### Feature Builds ARE GPU-Compatible (CORRECTED 2026-03-21)
- V1 institutional upgrades ALREADY converted ALL .apply() calls to GPU batch ops
- Zero .apply() calls remain in feature_library.py — confirmed by grep audit
- The _safe_gem_* and _safe_sent_* functions are dead code (defined but never called)
- cuDF.pandas mode accelerates all rolling TA, CuPy handles gematria/crosses
- Previous lesson "CPU-bound" was from BEFORE the GPU conversion
- On a RAPIDS cloud machine (cuDF + CuPy), the ENTIRE build runs on GPU
- Batch sizes auto-adapt to RAM: 200 (64GB), 500 (128GB), 1000 (256GB), 2000 (512GB+)
- Cloud with 512GB RAM + 8x GPU = fastest possible build

### H200 Has Weak CPU (2026-03-20)
- RunPod H200 pod bottlenecks LSTM training (heavy DataLoader CPU work)
- Use local 13900K + RTX 3090 for LSTM
- Use cloud GPU for LightGBM training and optimizer only

### vast.ai Lessons (2026-03-20)
- Do NOT pin CUDA_VISIBLE_DEVICES per process → causes OOM on single GPU
- Let ALL processes see ALL GPUs → CuPy distributes across all
- cuDF doesn't support timezone-aware datetimes → strip TZ at pipeline start
- CuPy requires contiguous arrays → np.ascontiguousarray() fix
- SQLite has 2000 column limit → save parquet FIRST
- PYTHONUNBUFFERED=1 essential for output capture
- __name__ guard required on ml_multi_tf.py

### Sparse Matrices Must Preserve NaN — Never nan_to_num (2026-03-21)
- Converting NaN → 0 before sparse storage is WRONG for two reasons:
  1. **Semantics**: LightGBM treats NaN as "missing" and learns optimal split directions for absent data. Converting to 0 tells the model "the value is zero" — completely different signal. Missing RSI (first 14 bars) is NOT the same as RSI=0
  2. **Storage bloat**: Sparse CSR only saves space by NOT storing zeros. If you convert NaN to explicit 0.0, those zeros get stored as entries in the data array. Base features (~3000 cols × 200K rows) are mostly non-zero values — adding millions of explicit zeros turns a 50MB sparse matrix into gigabytes
- FIX: `sp_sparse.csr_matrix(X_base)` directly (NaN stored as explicit entries, true zeros are structural)
- Then `X_all.eliminate_zeros()` removes only true zeros, NaN stays
- LightGBM Dataset handles NaN in sparse matrices natively — treats them as missing
- For variance screening on sparse with NaN: temporarily replace NaN with 0 in a COPY for computation, then discard. Never mutate the actual training matrix
- Cross features (.npz) are already correct — binary 0/1 with structural zeros = missing

### System RAM OOM After Many Sequential Builds (2026-03-21)
- After building 31 daily assets sequentially, system RAM filled up (~128GB)
- Cross generator holds dict with millions of arrays — GC doesn't free fast enough
- 4H and 1H builds ALL failed with tiny allocation errors (68 KiB = system fully exhausted)
- FIX: Force `del df; gc.collect()` at end of each asset build
- FIX: Run 4H/1H as separate invocations, not in same process as daily
- Daily builds all completed successfully (checkpointed) — only intraday needs restart

### GPU OOM on Large Cross Batches (2026-03-21)
- v2_cross_generator.py GPU batch cross with BATCH=50 OOMs on 3090 when right-side has 2,601+ contexts
- Shape (5727, 50, 2601) = 2.77 GB — exceeds available VRAM after other allocations
- CPU fallback works fine, just slower (~30s vs ~5s for the batch)
- FIX for next run: reduce BATCH to 20 or 10 when n_right > 1000
- Not critical — CPU fallback handles it, but GPU would be faster

### FRED API Blocked (2026-03-21)
- FRED CSV download times out from user's network (even with proxy)
- Not a problem — we have actual ETF price data (SPY, GLD, USO) + V1 macro_data.db
- FRED macro is redundant, skip it

### 4-Tier Binarization Scope (2026-03-21)
- Originally only 22 TA indicators had 4-tier
- User corrected: EVERYTHING numeric gets 4-tier (gematria, sentiment, astro positions, etc.)
- This massively increases binarized contexts: ~400 → ~8,000-10,000
- Which massively increases crosses: ~300K → ~15-20M per asset

### Every Asset Gets Full Esoteric (2026-03-21)
- Initially had TA-only fallback for non-BTC assets — WRONG
- The matrix is universal: same dates, same moon, same tweets, same energy
- SPY under the same full moon with the same caution tweet = same signal
- SPY has LESS noise than BTC → cleaner signal validates the pattern is real
- Stocks are the textbook, crypto is the exam

### Optuna Must Protect Esoteric Signals (2026-03-21)
- Tuning must NOT regularize esoteric features away
- If a param combo uses 0 esoteric features in top 100 splits → penalize it
- The model SHOULD use gematria, astro, numerology if they have signal
- Secondary objective in Optuna: esoteric feature usage count

### Philosophy Audit Fixes (2026-03-21)
- **Removed MI pre-screening** from CPCV training folds — was dropping esoteric features before XGBoost saw them
- **Removed zero-variance filter** from CPCV folds — XGBoost ignores constant features via 0-gain splits naturally
- **Removed dx_ support=50 filter** from feature_library.py — esoteric calendar signals are SPARSE by nature, that's the edge
- **Per-TF min_data_in_leaf**: 1d/1w=3, 4h=5, 1h=8, 15m=15. Rare astro conjunctions fire 10-20x on daily — old value of 50 killed them all
- **Removed fillna(0)** from build_1h GCP cross contexts — NaN means "unknown interaction" (XGBoost learns optimal split), 0 means "no interaction" (wrong signal)
- **Made V2_CROSS + V2_LAYERS mandatory** in live_trader.py — no silent degradation to base-only features
- **Removed blanket try/except** around V2 layer computation — if V2 layers fail, CRASH and fix, don't mask with degraded features
- **Empty binarized contexts = ValueError** — force investigation, not empty sparse fallback
- **IS metrics per CPCV fold** — XGBoost/LightGBM training now saves is_accuracy + is_sharpe for proper PBO
- **LSTM alpha search alignment** — uses test_indices from CPCV folds to match XGBoost OOS with LSTM val set on exact same samples
- **Meta-labeling filename** — cloud runner now looks for meta_model_{tf}.pkl (matches what meta_labeling.py saves)
- Lesson: any pre-filtering violates "NO FILTERING" even if it seems like it removes "noise." Sparse = the edge, not noise.

## CLOUD DEPLOYMENT PROTOCOL — MANDATORY

Follow `v3.3/CLOUD_TRAINING_PROTOCOL.md` EXACTLY for every cloud deployment. No shortcuts.

**If deployment fails:**
1. Consult Perplexity MCP for the fix — keep in mind training efficiency and the matrix thesis
2. Fix the root cause (not a bandaid)
3. Deploy successfully
4. Update `v3.3/CLOUD_TRAINING_PROTOCOL.md` with the lesson learned
5. Add the fix to the `--onstart-cmd` or smoke test so it NEVER happens again

**Key rules from protocol:**
- Install `cuda_init_fix.py` as sitecustomize on EVERY machine (NEVER use NUMBA_DISABLE_CUDA=1 — it breaks numba_cuda)
- Mandatory smoke test (Step 5) before training — NEVER skip
- `rapidsai/base:25.02-cuda12.8-py3.12` is the standard image
- `driver_version >= 570` filter on vast.ai search
- Download ALL artifacts before killing machine
- Always respond to user before running long tools

## V3.3 BUILD STATUS

### Code Changes (COMPLETE)
- [x] 173 new esoteric features (vortex math, sacred geometry, planetary expansion, numerology, lunar/EM)
- [x] 2 new gematria ciphers (Chaldean, AlBam) on all 10 text sources
- [x] 6 new holiday windows extended to 2035
- [x] 15 market signal features (DeFi TVL, BTC.D, mining stats)
- [x] 12 bug fixes (eclipse formula, BaZi branch, Tzolkin tone/kin, VOC moon, etc.)
- [x] is_unbalance → class_weight='balanced' (multiclass fix)
- [x] max_bin: 15 → 63 (4x resolution on continuous features)
- [x] max_conflict_rate: 0.0 (protect cross feature co-occurrence from EFB)
- [x] path_smooth: 0.1, extra_trees: False (new regularizers)
- [x] SHAP cross validation as pipeline Step 11
- [x] Inference crosses wired into live_trader.py
- [x] Sports lookahead removal (intraday uses game_timestamp)
- [x] Tweet color detection + gematria in all scrapers
- [x] Space weather: sunspot + solar flux endpoints added
- [x] DB path standardization (config.DB_DIR)

### Training (IN PROGRESS)
- [ ] 1w validation run (Norway 384c, confirms pipeline)
- [ ] Pick machines for 1d, 4h, 1h, 15m
- [ ] Train all 5 TFs
- [ ] Download + verify all artifacts
- [ ] SHAP analysis on cross features

### Post-Training
- [ ] Scale-in/Kelly/dynamic exits in exhaustive_optimizer
- [ ] LSTM projection + attention upgrade
- [ ] Temporal cascade ensemble (4h→1h→15m)
- [ ] Paper trading validation
