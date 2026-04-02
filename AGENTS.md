# Savage22 Agent Rules

## MCP TOOLS — MANDATORY RESEARCH PROTOCOL

### Primary: Olson KB / Orgonite Master (MCP: `olson-kb`)
The Olson Knowledge Base is the same system as the local Orgonite Master knowledge base at `C:\Users\C\Desktop\MY GOOGLE DRIVE\Orgonite master`.
It is the **first research source for EVERY non-trivial issue** — coding bugs, runtime failures, CUDA problems, ML issues, esoteric signals, calibration, deployment, architecture decisions, ALL of it. Not just esoteric topics.

**Tools:** `olson_kb_search`, `olson_kb_stats`, `olson_kb_topics`, `olson_kb_list`

Before planning or writing code for any non-trivial task:
```
Use olson-kb olson_kb_search with your query (try 2-3 phrasings)
```
If the `olson-kb` MCP tools are unavailable in-session, use the local KB CLI instead:
```
cd "C:/Users/C/Desktop/MY GOOGLE DRIVE/Orgonite master"
python kb.py smart "<query>" --limit 10
python kb.py smart "<second phrasing>" --limit 10
python kb.py smart "<third phrasing>" --limit 10
```

### Fallback: Perplexity (MCP: `perplexity-browser`)
Use Perplexity **ONLY** when the Olson KB / Orgonite Master KB returns <3 relevant results across your queries.

**Tools:** `perplexity_search`, `perplexity_reason`, `perplexity_followup`, `perplexity_deep_research`, `perplexity_models`

Every Perplexity query MUST include this context:
> "Our system uses 2.9M+ sparse binary features where individual signals (esoteric: gematria, numerology, astrology, space weather) may fire only 2-3 times per year but are highly predictive when they do. Standard ML advice to prune, regularize, or drop rare features does NOT apply. We use LightGBM with EFB (Exclusive Feature Bundling) which handles sparse features natively."

### Codebase Search: Socraticode (MCP: `socraticode`)
Use Socraticode for semantic code search and dependency analysis within the project.

**Tools:** `codebase_search`, `codebase_graph_query`, `codebase_watch`, `codebase_context_search`

At session start, run `codebase_watch` with `action: "start"` for the project directory.
When searching code: use `codebase_search` for semantic queries before falling back to grep/glob.

### Research Priority Order (NON-NEGOTIABLE)
1. **Olson KB / Orgonite Master** — always first, for any non-trivial issue
2. **Socraticode** — for codebase-specific questions
3. **Perplexity** — fallback only, when KB is weak
4. Generic knowledge — last resort

### What Counts as Non-Trivial (ALWAYS requires Olson KB / Orgonite Master first)
- Bugs and runtime failures
- Training, calibration, and validation issues
- CUDA, GPU, sparse-matrix, LightGBM, and Optuna issues
- Cloud, deployment, daemon, supervisor, and orchestration issues
- Model inference, live trading, and artifact-contract issues
- Architecture or dependency compatibility issues
- Feature engineering decisions
- Any code change to protected files (feature_library.py, ml_multi_tf.py, config.py, etc.)

### Trivial (KB research exempt)
- Typo fixes
- Formatting-only changes
- Path corrections with no behavior change

## PROJECT RULES
See `v3.3/CLAUDE.md` for the full project rulebook (validation, philosophy, process, code rules, audit pipeline, cloud deployment, anti-laziness overrides).

See `v3.3/CONVENTIONS.md` for agent conventions (Matrix Thesis, sparse standards, LightGBM params, CPCV, feature engineering, cloud deployment, owner approval gates).

See `v3.3/CODEX.md` for the Codex-specific operating rules (database-first research protocol, source priority, completion/audit rules).
