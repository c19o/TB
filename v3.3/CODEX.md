# V3.3 Codex Operating Rules

This is the Codex-specific rulebook for Savage22. Read this before investigating, planning, or implementing any non-trivial task.

## Core Policy

- Database first on every non-trivial bug, issue, failure, or technical decision.
- Perplexity is fallback only, never the first research step.
- Repo code and project docs are part of the database-first workflow, not a substitute for it.
- Do not implement before research evidence exists for any non-trivial technical task.
- Owner override in effect as of 2026-04-01: speed-positive changes are pre-approved if they preserve Matrix Thesis, rare-signal retention, OOS accuracy, and calibration. Do not re-ask for approval on speed work that stays inside those bounds.

## What Counts As Non-Trivial

These are NEVER "simple tasks" and always require database-first research:

- bugs and runtime failures
- training, calibration, and validation issues
- CUDA, GPU, sparse-matrix, LightGBM, and Optuna issues
- cloud, deployment, daemon, supervisor, and orchestration issues
- model inference, live trading, and artifact-contract issues
- architecture or dependency compatibility issues

Only trivial work is exempt:

- typo-only fixes
- formatting-only changes
- wording-only documentation edits that do not change technical meaning
- path corrections with no technical behavior change

## Mandatory Research Protocol

For every non-trivial bug or issue:

1. Read the repo truth first.
2. Query the local databases / KBs second.
3. Only if the KB is weak, log the gap and use Perplexity.
4. Record what source actually informed the decision.

Minimum required sequence:

```bash
cd "C:/Users/C/Documents/Savage22 Server/v3.3"
python ops_kb.py smart "<what is failing>" --limit 5

cd "C:/Users/C/Desktop/MY GOOGLE DRIVE/Orgonite master"
python kb.py smart "<bug or issue description>" --limit 10
python kb.py smart "<second phrasing>" --limit 10
python kb.py smart "<third phrasing>" --limit 10
```

Before KB research, log the task token:

```bash
cd "C:/Users/C/Documents/Savage22 Server/v3.3"
python ops_kb.py add "KB_QUERY: Task=[issue id or file path]. Query1=[first query]. Query2=[second query]. Query3=[third query]. ResultCounts=[n1,n2,n3]. Verdict=[definitive|weak]" --topic kb_query
```

If KB is definitive:

```bash
python ops_kb.py add "KB_SOURCE: Task=[issue id or file path]. Sources=[book/doc names]. Key finding=[one-line summary]. Confidence=[high/medium/low]" --topic kb_source
```

If KB is weak:

```bash
python ops_kb.py add "KB_GAP: Task=[issue id or file path]. Queried [your 3 queries]. <3 relevant results. Topic needed: [what is missing]." --topic kb_gap
```

Then and only then use Perplexity with Matrix Thesis context, and log:

```bash
python ops_kb.py add "PERPLEXITY_SOURCE: Task=[issue id or file path]. Query=[what you asked]. Sources=[URLs/paper names]. Key finding=[one-line summary]. Confidence=[high/medium/low]" --topic perplexity_source
```

## Database-First Means Real Technical Sources

Do not treat the KB as only esoteric books. It also includes technical material relevant to:

- trading algorithms
- calibration and probability mapping
- CUDA / NVIDIA best practices
- sparse ML and LightGBM behavior
- deployment and runtime engineering

If the vector database has books or technical documents on the issue, those must be consulted before fallback web research.

## Source Priority

When sources disagree, use this order:

1. live code and deterministic repo checks
2. project rule files and project audit docs
3. local KB / vector database material
4. Perplexity fallback research
5. generic intuition

Perplexity does not outrank the local KB when they conflict.

## Completion / Audit Rule

For any non-trivial researched task, run:

```bash
cd "C:/Users/C/Documents/Savage22 Server/v3.3"
python convention_gate.py research-audit <TASK_OR_FILE_TOKEN> --hours 72
```

If the research trail is missing, the task is not complete.
