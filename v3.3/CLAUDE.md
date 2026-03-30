# V3.3 Project Rules

## 1. VALIDATION SYSTEM (TOP RULE — NON-NEGOTIABLE)

- `validate.py` MUST pass before ANY training run (local or cloud)
- NEVER skip validation, even for "quick tests"
- After ANY config change: run `python validate.py` before committing
- After ANY new training failure: add a `check()` to validate.py BEFORE fixing the bug (ensures it never recurs)
- `validate.py` is the SINGLE SOURCE OF TRUTH for all parameter constraints
- If validate.py and this file disagree, **validate.py wins** (it's deterministic)
- All numerical parameter rules live in validate.py, NOT in this file

## 2. PHILOSOPHY (NON-NEGOTIABLE)

- The matrix is UNIVERSAL — same sky, same calendar, same energy for ALL assets
- Every asset gets the FULL pipeline (esoteric, astro, gematria, numerology, space weather)
- NO FILTERING of features — the model decides via tree splits, not us
- NO FALLBACKS — one pipeline for all. If it breaks, fix it
- Esoteric signals ARE the edge. Never regularize them away
- More diverse signals = stronger predictions. The edge is the matrix
- NO NaN->0 conversion — NaN = "missing" (model learns split direction), 0 = "value is zero" (different signal)
- Structural zeros in CSR = 0.0 (feature OFF, correct for binary crosses)

## 3. PROCESS RULES

- NEVER deviate from the plan. Follow it exactly as written
- Always make a checklist and verify each step before moving to the next
- Always run scripts with progress logs (tee/unbuffered). Never run blind
- NEVER kill processes without explicit user permission
- Don't search user's PC for files. Use the sources they provide
- Use Perplexity MCP for technical queries — ALWAYS include matrix thesis context
- Stagger feature builds: small TFs parallel, then 15m separately
- Build features locally (13900K), train on cloud GPU
- Always maximize parallelism. Launch up to 20 agents simultaneously
- NEVER rent a slower machine than what we had. Fix issues in-place
- Fix ALL issues before deploying. No mid-run patches. Test locally first
- Download artifacts at EVERY checkpoint. Cloud machines die without warning
- Evaluate results (accuracy, SHAP, confidence) BEFORE destroying cloud machines
- One TF at a time, smallest first. Verify full pipeline before scaling
- Keep session resume file updated after every significant step
- After fixing a bug, grep ALL files for same pattern before deploying

## 4. CODE RULES

- LightGBM is the ONLY training backend (EFB, sparse CSR). NO XGBoost
- Sparse CSR preserved through training — no dense conversion
- EFB (enable_bundle) ALWAYS True for ALL timeframes
- Batch column assignment (pd.concat / dict accumulation), NEVER one-at-a-time df[col]=val
- Always use GPU (RTX 3090) for processing, never default to CPU
- 4-tier binarization on ALL numeric columns
- Stateful loops must use Numba @njit — never raw Python for-loops on price arrays
- No fallback modes. Full pipeline or fix it
- `feature_pre_filter=False` always — True silently kills rare features
- `num_threads=0` for lgb.train() (0 = auto-detect). NEVER use -1
- Sparse CSR: indptr=int64 (NNZ > 2^31 fix), indices=int32
- NO 5m timeframe (only 1w, 1d, 4h, 1h, 15m)

## 5. AUDIT PIPELINE

When the user says "audit", execute this multi-pass pipeline:

### CASCADE DETECTION RULE (applies to EVERY pass)
After ANY fix, BEFORE moving to next item:
1. Signature changes -> grep ALL callers, verify match
2. Filename/path changes -> grep ALL consumers, verify paths
3. Config value changes -> grep ALL readers, verify they use config.KEY
4. Interface contract changes -> trace data flow end-to-end

### PASS 0: Matrix Philosophy Gate (first, blocks everything)
- No feature filtering/pre-screening before LightGBM
- No fallback modes
- No fillna(0) on feature data
- No blanket try/except that masks failures
- Esoteric features protected from pruning
- V2 cross + V2 layers mandatory in live_trader

### PASS 1: Dead Code (parallel with pass 2)
- Unused imports, functions, files. DELETE dead code.

### PASS 2: Config & Hardcoded Values (parallel with pass 1)
- Magic numbers -> config.py. Secrets -> .env. Paths -> config.py with env override.

### PASS 3: Cross-File Consistency (after pass 0)
- Feature column names match across pipeline stages
- Sparse matrix shapes align
- Function signatures match ALL call sites

### PASS 4: GPU & Performance (after pass 1+2)
- No CPU fallbacks where GPU path exists
- No .apply() or Python for-loops on arrays

### PASS 5: Live Trading Safety (after pass 3)
- Kill switch, max position size, stale data detection, identical paper/live inference

### PASS 6: Cascade Regression (last, sequential)
- Import every module, config loads, feature builder initializes
- Re-audit ALL signatures/filenames/config changes from passes 0-5

## 6. CLOUD DEPLOYMENT PROTOCOL

### Pip + SCP (ONLY METHOD)
1. Rent machine (pytorch base image, pip cached)
2. `pip install lightgbm scikit-learn scipy ephem astropy pytz joblib pandas numpy pyarrow optuna hmmlearn numba tqdm pyyaml sparse-dot-mkl`
3. Test ALL imports: `python -c "import pandas, numpy, scipy, sklearn, lightgbm, ephem, astropy, pyarrow, optuna, numba, hmmlearn, yaml, tqdm; print('ALL OK')"`
4. SCP code tar + ALL .db files
5. Symlink DBs: `ln -sf /workspace/*.db /workspace/v3.3/`
6. Run: `cd /workspace/v3.3 && python -u cloud_run_tf.py --symbol BTC --tf 1w`
7. validate.py runs automatically as first step

### Required uploads:
- All `v3.3/*.py` and `*.json` files
- `astrology_engine.py` (from project root)
- ALL .db files (16+ databases). Missing DB = weaker model = INVALID RUN

### Non-negotiable deployment rules:
- NEVER deploy with missing databases
- NEVER say "it's fine" about missing data
- NEVER skip the smoke test
- NEVER deploy code that hasn't been audited (3 clean passes)
- Verify DB count after extract: `ls /workspace/*.db | wc -l` >= 16
- Check first 30s of log for "WARNING: DB missing" -> STOP if any
- CHECK ALL CONFIG PATHS IN FIRST 10 LOG LINES (DB_DIR, V30_DATA_DIR, etc.)

## Codebase Intelligence (Socraticode)

- `codebase_search` — Semantic search across source files
- `codebase_graph_query` — Import/dependency analysis
- `codebase_context_search` — Search indexed artifacts
- `codebase_watch` — File watcher keeps index current
