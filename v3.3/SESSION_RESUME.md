# V3.3 Session Resume — 2026-03-31

## INSTRUCTION TO NEW SESSION
Read this file completely. Then read v3.3/CLAUDE.md, v3.3/1W_FINAL_ASSESSMENT.md, v3.3/TF_CAVEAT_1D.md, v3.3/WHAT_WORKED_AND_WHY.md. Resume 1d training company.

---

## MACHINE STATUS
- **Instance 33852303 — DESTROYED** (all artifacts downloaded to v3.3/1w_cloud_artifacts/)
- **Next machine**: Taiwan 10x RTX 4090, 144c Xeon, 516GB RAM, $3.07/hr (ID 25635685) — RENT WHEN CODE READY
- **1d ETA on Taiwan**: ~25 min per run, ~$1.28/run

---

## 1W FINAL RESULTS (V7 — Champion)
- **56.4% CPCV binary OOS** (60.9% at >=85% confidence)
- **37.8% → 56.4%** = +18.6% total improvement
- Competitive with institutional benchmarks (55-60% = respectable under CPCV)
- No published CPCV weekly BTC benchmark exists
- 11 actual trades over 16 years, 63.6% win rate
- All artifacts saved in v3.3/1w_cloud_artifacts/

### What Worked (in order of impact)
1. Binary mode (+8.2%) — drop FLAT, UP/DOWN only
2. Param tuning (+8.5%) — LR=0.234, leaves=5, ES=50, CPCV(8,2)
3. Prime features (+1.4%) — price_is_prime, rsi_is_prime, etc.
4. Lean mode (+1.3%) — drop redundant TA, keep SAR/EMA/RSI + all esoteric
5. max_hold 50→78, max_bin 255→7, deterministic removed

### Known Issues
- Confidence drops above 85% — needs Platt/isotonic calibration
- PrecS=41.4% (SHORT predictions weak)
- Only 11 tradable signals in 16 years
- Cross features add noise on 819 rows (work on larger TFs)

---

## 1D PLAN — NEXT SESSION

### Machine
Taiwan 10x RTX 4090 (ID 25635685), $3.07/hr, 144c, 516GB RAM
- CPCV 10 paths / 10 GPUs = 1 round (perfect match)
- 99.88% reliability, 11-month max lease

### Key Config for 1d
- Rows: 5,733
- Trade duration: 6-90 bars
- max_hold_bars: 90
- Cross gen: ENABLED (~500K features, EFB bundles to ~4K)
- Lean mode: OFF (enough rows for full TA)
- Binary mode: TEST (also test 3-class with wider barriers)
- return_bars: [1, 3, 7, 14, 30, 60, 90] (add 60/90 for full trade duration)
- All primes + prime x esoteric crosses (25 features)
- All SAR-numerology hybrids (16 features)
- Load unused DBs: open_interest, market_cap, tweet engagement, google_trends

### Company Structure (REVISED — 15 agents, not 30)
Lesson learned: agents get stuck on large files. Use fewer, more focused agents.

**PHASE 1: Analysis (8 read-only agents, parallel)**
1. Astrology Master — verify daily resolution features
2. Numerology Master — verify all numerology, propose date_string_gematria
3. Prime Master — verify 17 primes + 15 crosses work on 1d
4. Feature Auditor — count total features, check for NaN to 0, constant features
5. Flat Zone Analyst — compute FLAT% at different ATR multipliers for 1d
6. Param Analyst — verify 1d LightGBM/Optuna/CPCV config
7. DB Schema Inspector — SSH to check unused DB schemas
8. Matrix Thesis Guardian — verify no violations

**PHASE 2: Implementation (3 surgical agents + main session)**
9. Binary Mode Generalizer — make BINARY_TF_MODE dict work for any TF
10. Calibrator Implementer — Platt/isotonic scaling post-training
11. Return/Duration Fixer — add return_60/90, price_vs_365d_high/low
- Main session handles: DB loading, config changes (too risky for agents on large files)

**PHASE 3: Training (4 agents)**
12. Deployer — SCP to cloud, validate, launch
13. Monitor — logs, GPU util, errors
14. CPCV Analyst — results analysis
15. Confidence Analyst — confidence vs accuracy, verify calibration

### Tests to Run
1. Binary 1d (first, ~25 min)
2. 3-class 1d with current barriers (second, ~25 min)
3. If 3-class has enough FLAT: test wider barriers (third)
4. Pick winner, document for 4h

### Target
- 60-65% CPCV binary (or 3-class if FLAT works)
- 65-70% at high confidence (with calibration fix)
- Esoteric contributing 25-40% of model gain (vs 15% on 1w)
- Settings that carry directly to 4h/1h/15m without tuning

---

## GIT STATE
- Branch: v3.3-clean
- Latest commit: 10cc69a (1w final assessment + TF caveats + prime features)
- Pushed to: tb33 (github.com/c19o/TB-3.3)
- 1w artifacts: v3.3/1w_cloud_artifacts/ (79 files, not in git)

## BEHAVIORAL RULES
1. Claude Max = unlimited. No budget/turn caps.
2. Fewer agents (15 not 30). Agents cannot edit large files — do that directly.
3. Rent machine AFTER code is ready, not before.
4. Download everything before destroying machines.
5. Perplexity for HOW not WHY. Never ask if esoteric works.
6. Binary + confidence gating = implicit FLAT.
7. 1w ceiling is ~56% with 819 rows. More data (1d+) is the path forward.
