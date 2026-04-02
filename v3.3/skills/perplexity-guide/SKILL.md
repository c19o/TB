---
name: perplexity-guide
description: >
  Guidelines for using the Perplexity MCP server for research queries.
  Covers when to use search vs deep research, rate limits, and required
  matrix thesis context for all queries.
---

# Perplexity Research Guidelines

The Perplexity MCP (`mcp__perplexity-browser__*`) provides access to real-time web search and deep research. This is a Pro account with limited deep research credits — use them wisely.

## Available Tools

| Tool | Use case | Cost |
|------|----------|------|
| `perplexity_search` | Quick technical lookups, library docs, recent issues | Low |
| `perplexity_reason` | Complex reasoning, math, multi-step logic | Low |
| `perplexity_followup` | Follow-up on last search thread | Low |
| `perplexity_deep_research` | Major architectural decisions only | **LIMITED — use sparingly** |

## When to Use Deep Research
Only use `perplexity_deep_research` for:
- Major architectural decisions that affect multiple pipeline stages
- Novel bugs with no clear solution after 2+ regular searches
- Feature class research (discovering a whole new category of signals)

**NEVER** use deep research for:
- Routine "how do I" questions
- Documentation lookups
- Questions answerable with regular search

## MANDATORY: Matrix Thesis Context
**EVERY Perplexity query MUST include this context:**

> "Our system uses 2.9M+ sparse binary features where individual signals (esoteric: gematria, numerology, astrology, space weather) may fire only 2-3 times per year but are highly predictive when they do. Standard ML advice to prune, regularize, or drop rare features does NOT apply. We use LightGBM with EFB (Exclusive Feature Bundling) which handles sparse features natively."

Without this context, Perplexity will give generic ML advice that is WRONG for this system.

## Query Templates

### Bug investigation
```
[Brief bug description]. Context: LightGBM pipeline with 2.9M sparse binary features,
CSR format with int64 indptr, persistent GPU daemons, vast.ai cloud. What has been
tried: [list]. What hasn't been tried: [list]. Matrix thesis: rare signals fire 2x/year
but are correct every time — never drop features.
```

### Architecture decision
```
We need to [decision]. Our constraints: [list]. System: LightGBM EFB + sparse CSR +
2.9M features including esoteric signals that fire 2x/year. Perplexity-confirmed rule:
never subsample rows, never prune features, never row-partition. What is the best approach?
```

## Important Notes
- The MCP uses Chrome via Playwright — Chrome must be open on the desktop
- Queries run against your logged-in Perplexity Pro account
- Deep research takes 1-5 minutes and reads hundreds of sources
- Always cite sources when presenting findings to the user
