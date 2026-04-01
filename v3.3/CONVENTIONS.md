# Savage22 V3.3 — Agent Conventions

Injected into every agent session. These are non-negotiable.

---

## THE MATRIX THESIS (NON-NEGOTIABLE)

- More diverse signals = stronger predictions
- Rare signals firing 2x/year ARE the edge. NEVER prune them.
- Standard ML advice on feature pruning does NOT apply
- Every esoteric signal prefix is protected: gem_, dr_, moon_, nakshatra, eclipse, vedic_, bazi_, hebrew_, sw_, aspect_, vortex_, sephirah, chakra_, jupiter_, mercury_, planetary_, angel, master_, palindrome, doy_, dx_, ax_, ex2_, asp_, pn_, mn_, bio_, saros_, metonic_, rahu_, ketu_, fib_

## SPARSE MATRIX STANDARDS

- ALL cross features stored as sparse CSR (scipy.sparse.csr_matrix)
- int64 indptr required for NNZ > 2^31
- NEVER call .toarray() or .todense() on cross feature matrices
- EFB pre-bundling: 127 binary features per bundle
- max_bin=7 (binary features need 2 bins max)

## LIGHTGBM PARAMETERS (SACRED)

- feature_fraction >= 0.7 (CRITICAL: lower values kill rare cross signals)
- feature_pre_filter = False (True silently kills rare features)
- is_enable_sparse = True
- bagging_fraction >= 0.95 (preserves P(10-fire in bag) = 59.9%)
- NEVER use XGBoost. LightGBM only. EFB is architecturally correct for the matrix.

## CPCV CONVENTIONS

- Purge width = max_hold_bars from triple_barrier_config
- K=2 for all TFs (combinatorial)
- Path sampling: 1w=all 28, others=30 deterministic (seed=42)
- NEVER modify fold logic without dual sign-off (ML Pipeline + QA Lead)

## FEATURE ENGINEERING RULES

- Batch column assignment (pd.concat/dict accumulation), NEVER one-at-a-time df[col]=val
- All numeric features get 4-tier binarization
- No NaN->0 conversion (LightGBM handles NaN natively, LSTM needs impute)
- Numba @njit for any price array loops
- Add new feature prefixes to PROTECTED_FEATURE_PREFIXES in config.py

## CLOUD DEPLOYMENT

- symbol='BTC' not 'BTC/USDT'
- V2_RIGHT_CHUNK=500 for cross gen (Auto=2000 OOMs)
- OMP_NUM_THREADS=4 for thread exhaustion prevention
- Download artifacts at EVERY checkpoint (machines die without warning)
- killall python before launching new training
- NEVER use nohup bash wrappers — use cloud_run_tf.py directly

## LOGGING FORMAT

- All ops_kb entries: "FACT: <structured description>. Date: YYYY-MM-DD"
- All Discord notifications: emoji prefix + TF + step + key metric
- All training logs: tee to disk + unbuffered output

## MANDATORY KB RESEARCH — NON-NEGOTIABLE
Any code that touches training or features MUST query the Knowledge Base BEFORE writing code:
```bash
cd "C:/Users/C/Desktop/MY GOOGLE DRIVE/Orgonite master"
python kb.py smart "<what you're implementing>" --limit 10
python kb.py smart "<alternative phrasing>" --limit 10
python kb.py smart "<third phrasing>" --limit 10
```
Also query ops_kb for what's been tried before:
```bash
cd "C:/Users/C/Documents/Savage22 Server/v3.3"
python ops_kb.py smart "<what you're doing>" --limit 5
```
Files that REQUIRE KB research before ANY edit:
- feature_library.py (features/signals)
- ml_multi_tf.py (training pipeline)
- config.py (parameters)
- v2_cross_generator.py (cross features)
- cloud_run_tf.py (deployment)
- gpu_daemon.py (GPU operations)

If you edit these files without KB queries, your work will be REJECTED and REVERTED.
The KB has 947 docs: AFML full book, LightGBM paper, CUDA guides, 42 academic papers on every signal type. USE THEM.

## POST-IMPLEMENTATION VERIFICATION — NON-NEGOTIABLE
After writing ANY code that touches training or features, you MUST verify:

### Step 1: KB Cross-Check
Query the KB with what you just implemented to confirm it aligns:
```bash
cd "C:/Users/C/Desktop/MY GOOGLE DRIVE/Orgonite master"
python kb.py smart "<what I just implemented> best practices" --limit 5
python kb.py smart "<parameter I just set> recommended values" --limit 5
```
If KB says your approach is wrong → FIX IT before committing.

### Step 2: Perplexity Cross-Check (if Perplexity was used)
If you used Perplexity advice to write code, verify it against the KB:
```bash
python kb.py smart "<the Perplexity recommendation>" --limit 5
```
PERPLEXITY GIVES GENERIC ML ADVICE. Our system is NOT generic:
- 2.9M sparse binary features (not 50 dense features)
- Rare signals fire 2x/year (standard pruning advice DESTROYS our edge)
- LightGBM with EFB (not XGBoost, not neural nets)
- Sparse CSR end-to-end (not dense matrices)

If Perplexity says "prune features with low importance" → WRONG for us.
If Perplexity says "use feature_fraction 0.3" → WRONG for us (minimum 0.7).
If Perplexity says "convert to dense for speed" → WRONG for us (498GB OOM).
If Perplexity says "reduce max_bin to 2" → CHECK against KB first.

ALWAYS trust the KB over Perplexity. The KB has OUR books, OUR papers, OUR architecture.

### Step 3: validate.py
```bash
cd "C:/Users/C/Documents/Savage22 Server/v3.3"
python validate.py
```
ALL 74 checks must pass. If any fail → fix BEFORE committing.

### Step 4: Self-Review Checklist
Before committing, answer YES to ALL:
- [ ] Does my code match what the KB says about this topic?
- [ ] Did I preserve all PROTECTED_FEATURE_PREFIXES?
- [ ] Did I use batch column assignment (not one-at-a-time)?
- [ ] Is feature_fraction still >= 0.7?
- [ ] No NaN→0 conversions?
- [ ] No dense matrix conversions on cross features?
- [ ] No imports of XGBoost?
- [ ] validate.py passes?

If ANY box is unchecked → FIX before committing.

## CODE STYLE

- No docstrings/comments on code you didn't change
- No "improvements" beyond task scope
- No error handling for scenarios that can't happen
- No helpers/utilities for one-time operations
- Three similar lines > premature abstraction
