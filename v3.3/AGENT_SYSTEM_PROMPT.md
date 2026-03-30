# Matrix Thesis System Prompt — ALL AGENTS MUST READ

## THE MATRIX THESIS (NON-NEGOTIABLE)
This BTC trading system uses "the matrix" — a massive cross-product of esoteric signals (astrology, gematria, numerology, space weather, moon phases, planetary alignments, sacred geometry) combined with traditional financial features. The thesis: the same sky, same calendar, same energy affects ALL markets. More diverse signals = stronger predictions. The edge IS the matrix.

## CRITICAL RULES — VIOLATING ANY OF THESE INVALIDATES THE WORK
1. **NO FEATURE FILTERING** — The model (LightGBM) decides via tree splits, NOT us. Never pre-screen, whitelist, or prune features before training.
2. **NO NaN→0 CONVERSION** — NaN = "missing" (model learns split direction). 0 = "value is zero" (different signal). Converting NaN→0 destroys information.
3. **NO FALLBACKS** — One pipeline for all. If it breaks, fix it. No degraded modes.
4. **feature_pre_filter=False ALWAYS** — True silently kills rare features.
5. **feature_fraction >= 0.7** — Lower values kill rare esoteric cross signals silently.
6. **bagging_fraction >= 0.7** — Low values make 10-fire signals appear in only 11% of trees.
7. **min_data_in_leaf <= 10** — Rare signals (10-20 fires) must be able to form leaves.
8. **lambda_l1 <= 4.0** — Higher values zero leaf weights for signals firing ≤33 times.
9. **Sparse CSR preserved** — No dense conversion. LightGBM EFB handles sparse natively.
10. **LightGBM ONLY** — No XGBoost. LightGBM's EFB is architecturally correct for binary crosses.
11. **All 2.9M features must reach the model** — Any silent filtering is a critical bug.
12. **Structural zeros in CSR = 0.0** — Feature OFF, correct for binary crosses. Not the same as NaN.

## PIPELINE OVERVIEW
Feature Build → Cross Gen (13 steps, sparse CSR) → EFB Pre-Bundler → LightGBM Dataset → Optuna HPO → LightGBM Training (cuda_sparse) → CPCV Validation → Trade Optimizer (Sobol) → PBO + Audit → Inference

## TIMEFRAMES (NO 5m)
1w (818 rows), 1d (5,733 rows), 4h (8,794 rows), 1h (90K rows), 15m (227K rows)

## PERPLEXITY QUERIES
When using Perplexity MCP, ALWAYS include this context:
"We're building a BTC trading system with 2.9M features from cross-products of esoteric signals (astrology, gematria, numerology, space weather) and traditional financial data. LightGBM with EFB on sparse CSR matrices. The thesis is that more diverse signals = stronger predictions. Rare signals (10-20 fires across 818-227K rows) are THE edge — they must never be filtered, regularized away, or lost in any pipeline step."

## YOUR OUTPUT MUST INCLUDE
1. **Matrix Adherence Statement** — Explicitly confirm no feature filtering, NaN preservation, sparse integrity
2. **Specific Findings** — File:line references for every issue found
3. **Recommended Changes** — Exact code diffs, not vague suggestions
4. **Risk Assessment** — What could go wrong if this change is applied incorrectly
