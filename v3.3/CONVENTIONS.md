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
python kb.py smart "<what you're implementing>" --n 10
python kb.py smart "<alternative phrasing>" --n 10
python kb.py smart "<third phrasing>" --n 10
```
Also query ops_kb for what's been tried before:
```bash
cd "C:/Users/C/Documents/Savage22 Server/v3.3"
python ops_kb.py smart "<what you're doing>" --n 5
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

## CODE STYLE

- No docstrings/comments on code you didn't change
- No "improvements" beyond task scope
- No error handling for scenarios that can't happen
- No helpers/utilities for one-time operations
- Three similar lines > premature abstraction
