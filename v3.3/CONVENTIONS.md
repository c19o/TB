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

## OWNER APPROVAL GATES

- The user is the company owner, not the runtime manager. Keep them out of the loop unless escalation is required.
- Escalate to the owner BEFORE any production change that can materially affect training speed, throughput, or runtime behavior.
- Owner override in effect as of 2026-04-01: speed-positive changes are pre-approved if they preserve Matrix Thesis, rare-signal retention, OOS accuracy, and calibration. Do not re-ask for approval on speed work that stays inside those bounds.
- Examples that REQUIRE owner approval before production use:
  - dense vs sparse execution-path changes
  - GPU vs CPU execution-path changes
  - machine-selection changes that alter speed, throughput, or cost profile
- Also escalate for:
  - renting or destroying cloud machines
  - Matrix Thesis / protected-feature policy changes
  - rollback decisions after a bad run
- If a change is intended to increase speed, the burden of proof is: faster WITHOUT loss of OOS accuracy, calibration, or rare-signal retention.
- RIGHT_CHUNK / batch sizing / fold parallelism / num_threads / daemon scheduling / cache / checkpoint / I/O-path changes are authorized under the owner override above unless they introduce a credible matrix, calibration, or cost-risk exception.
- Under the 2026-04-01 owner override, implement the change directly if the evidence says it improves speed and does not weaken matrix richness, calibration, or OOS behavior. Escalate only if those protections are uncertain.

## LOGGING FORMAT

- All ops_kb entries: "FACT: <structured description>. Date: YYYY-MM-DD"
- All Discord notifications: emoji prefix + TF + step + key metric
- All training logs: tee to disk + unbuffered output

## CODEX RULEBOOK

- `CODEX.md` is the Codex-specific operating rulebook for this repo.
- `CONVENTIONS.md` remains the shared enforcement contract.
- If a Codex behavior question comes up, read `CODEX.md` and follow the stricter rule.

## MANDATORY KB RESEARCH — NON-NEGOTIABLE
Any non-trivial technical task MUST query the Knowledge Base BEFORE planning or writing code.
This includes bug diagnosis, runtime failures, dependency issues, calibration issues, sklearn compatibility issues, ML, CUDA, training, feature engineering, deployment/runtime, GPU memory work, and cross-generation work.
Only truly simple tasks are exempt, such as typo fixes, pure formatting, path corrections, or documentation wording that does not change technical meaning.

ML / CUDA / training / feature / deployment-runtime / daemon / calibration / compatibility examples are NEVER "simple tasks" for the purpose of this rule.

Repo docs and code inspection are REQUIRED, but they do NOT replace KB/database research for non-trivial issues.
If the vector database has books or technical docs on the issue, those MUST be consulted before Perplexity.

At minimum, any issue touching training, features, calibration, CUDA, GPU memory, cloud runtime, daemons, supervisors, validation gates, dependencies, or model deployment MUST query the Knowledge Base first:
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
Before the KB search, log the exact task token you are researching so the audit can tie the evidence back to one issue:
```bash
python ops_kb.py add "KB_QUERY: Task=[SAV-29 or file path]. Query1=[first KB query]. Query2=[second KB query]. Query3=[third KB query]. ResultCounts=[n1,n2,n3]. Verdict=[definitive|weak]" --topic kb_query
```
If the KB gave a definitive answer, log the source you actually used before coding:
```bash
python ops_kb.py add "KB_SOURCE: Task=[SAV-29 or file path]. Sources=[book/doc names from KB]. Key finding=[one-line summary]. Confidence=[high/medium/low]" --topic kb_source
```
Files that REQUIRE KB research before ANY edit:
- feature_library.py (features/signals)
- ml_multi_tf.py (training pipeline)
- live_trader.py (inference/runtime decisions)
- config.py (parameters)
- v2_cross_generator.py (cross features)
- cloud_run_tf.py (deployment)
- gpu_daemon.py (GPU operations)
- cross_supervisor.py (daemon/supervisor runtime)
- run_optuna_local.py (Optuna/training runtime)
- validate.py (release gates)
- test_pipeline_plumbing.py (deployment/runtime contract)
- deploy_manifest.json / deploy_manifest.py (deployment runtime contract)

If you edit these files without KB queries, your work will be REJECTED and REVERTED.
The KB has 947 docs: AFML full book, LightGBM paper, CUDA guides, 42 academic papers on every signal type. USE THEM.

### KB Gap + Perplexity Source Logging — MANDATORY
If KB returns <3 relevant results across your 3 queries, you MUST:
1. Log the gap to ops_kb:
   ```bash
   python ops_kb.py add "KB_GAP: Task=[SAV-29 or file path]. Queried [your 3 queries]. <3 relevant results. Topic needed: [what's missing]. Suggested text: [paper/book if known]" --topic kb_gap
   ```
2. THEN use Perplexity (with matrix thesis context as always). Never use Perplexity first for a non-trivial issue.
3. After Perplexity returns, log its sources to ops_kb:
   ```bash
   python ops_kb.py add "PERPLEXITY_SOURCE: Task=[SAV-29 or file path]. Query=[what you asked]. Sources=[list URLs/paper names Perplexity cited]. Key finding=[one-line summary]. Confidence=[high/medium/low based on source quality]" --topic perplexity_source
   ```
4. If Perplexity cites a paper/textbook we don't have in KB, add it to the gap log:
   ```bash
   python ops_kb.py add "KB_GAP_DOWNLOAD: Task=[SAV-29 or file path]. [paper/book title] by [author]. URL: [if available]. Reason: Perplexity cited this for [topic] and we don't have it in KB." --topic kb_gap
   ```

This creates an audit trail: KB gap found -> Perplexity used -> sources logged -> texts identified for download -> user downloads -> KB ingested -> gap closed.
Run this before declaring the task done:
```bash
python convention_gate.py research-audit SAV-29 --hours 72
```

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
ALL 96 checks must pass. If any fail → fix BEFORE committing.

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

## VALIDATION PROPOSAL PROCESS

When you discover a new invariant that should be enforced (e.g., a parameter range, a config consistency rule):

1. **DO NOT edit validate.py directly** — it's QA Lead + User only
2. Create a proposal file: `v3.3/validation_proposals/YYYY-MM-DD-short-name.yaml`
3. Include: predicate, severity, rationale, evidence, pass/fail examples
4. Commit the proposal file
5. QA Lead reviews → implements in validate.py → updates .validate_hash
6. Stop hook enforces the new check on all future agent runs

See `v3.3/validation_proposals/README.md` for the full YAML schema.

## STOP HOOK ENFORCEMENT (AUTOMATIC — YOU CANNOT BYPASS)

Before you can finish any task, the Stop hook checks:
- **Gate 1**: validate.py must pass (all checks green)
- **Gate 2**: ops_kb must have your log entry
- **Gate 2A**: training/feature/research tasks must have `KB_QUERY`
- **Gate 2B**: definitive KB use must have `KB_SOURCE`
- **Gate 2C**: KB gaps must have `PERPLEXITY_SOURCE` after the gap
- **Gate 3**: PROTECTED_FEATURE_PREFIXES must cover all feature prefixes in feature_library.py
- **Gate 4**: Sacred parameter ranges (bagging_fraction >= 0.95, feature_fraction >= 0.7)

If ANY gate fails → you are BLOCKED from finishing. Fix the violations first.

## CODE STYLE

- No docstrings/comments on code you didn't change
- No "improvements" beyond task scope
- No error handling for scenarios that can't happen
- No helpers/utilities for one-time operations
- Three similar lines > premature abstraction
