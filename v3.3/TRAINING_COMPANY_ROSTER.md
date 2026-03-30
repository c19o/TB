# Training Company Roster — Launch ALL at Session Start

## INSTRUCTION: Launch all 20 agents simultaneously. Each is Perplexity-enabled with matrix thesis context. All document everything scientifically.

---

## TEAM 1: BUG FIXERS (3 agents)

### Agent 1: Dense Data Bug Fixer
- **Role**: backend-dev
- **Task**: Fix runtime_checks.py:51 — add scipy.sparse.issparse() guard before .nnz. 1w has no cross features = dense data.
- **Also fix**: Dense→sparse conversion should happen BEFORE Optuna, not just before CPCV.

### Agent 2: LSTM SM_120 Fixer
- **Role**: backend-dev
- **Task**: Fix LSTM on RTX 5090. Either upgrade PyTorch (need torch 2.6+ with CUDA 12.8+) or add graceful skip with loud WARNING.
- **Perplexity query**: "PyTorch SM_120 Blackwell RTX 5090 support — which version?"

### Agent 3: CLI Args Fixer
- **Role**: backend-dev
- **Task**: Remove --no-parallel-splits from cloud_run_tf.py. Use env var FORCE_SEQUENTIAL=1 instead. Fix all downstream scripts that broke.

---

## TEAM 2: FEATURE ENGINEERS (3 agents)

### Agent 4: Weekly Feature Adder
- **Role**: backend-dev
- **Task**: Add to feature_library.py: week_of_year sin/cos, week_digital_root, month_digital_root, quarter sin/cos, year_in_halving_cycle, weeks_since_halving
- **Gate**: Only compute for 1w/1d timeframes

### Agent 5: Astrology Feature Fixer
- **Role**: backend-dev
- **Task**: Wire eclipse features (is_eclipse_window), BTC natal transit (get_btc_transit_score), Jupiter-Saturn regime categories, Mercury retrograde days_into_retrograde. Remove 4 dead planetary_hour features for 1w.

### Agent 6: Feature Trimmer
- **Role**: backend-dev
- **Task**: Add TF-aware gate in feature_library.py — skip hour_sin/cos, dow_sin/cos, day_of_week, is_monday, is_friday, is_weekend for 1w. This also eliminates thousands of useless cross features downstream.

---

## TEAM 3: PARAMETER TUNERS (3 agents)

### Agent 7: LightGBM Param Tuner
- **Role**: backend-dev
- **Task**: Update config.py 1w params based on parameter team findings. Key changes: higher LR, adjusted num_leaves, lower ES patience. Also apply deterministic=False, max_bin=7 globally.
- **Perplexity query**: "Optimal LightGBM params for 819 rows, 3552 features, 3-class"

### Agent 8: CPCV Config Tuner
- **Role**: backend-dev
- **Task**: Change 1w CPCV from (5,2) to (3,1) — more data per fold (67% train vs 60%). Adjust purge bars for weekly resolution.

### Agent 9: Optuna Search Space Tuner
- **Role**: backend-dev
- **Task**: Update run_optuna_local.py 1w search ranges: higher LR range, lower num_boost_round cap, fewer trials for tiny data.

---

## TEAM 4: TRAINING OPERATORS (3 agents)

### Agent 10: Deployer
- **Role**: backend-dev
- **Task**: SCP all fixed code to cloud machine (ssh -p 12302 root@ssh7.vast.ai), verify imports, run validate.py --cloud, launch training.

### Agent 11: Monitor
- **Role**: error-checker
- **Task**: Tail logs every 30s. Watch for errors, GPU utilization (nvidia-smi), memory usage. Kill and report if anything crashes.

### Agent 12: Artifact Downloader
- **Role**: backend-dev
- **Task**: Download artifacts at every checkpoint. SCP model, configs, predictions, logs to local v3.3/.

---

## TEAM 5: ACCURACY & QUALITY (4 agents)

### Agent 13: CPCV Analyst
- **Role**: error-checker
- **Task**: Analyze CPCV results as they come in. Per-path accuracy, fold variance, Trees count. Compare with previous run (37.8% untuned vs target). Statistical significance tests.

### Agent 14: Feature Importance Analyst
- **Role**: error-checker
- **Task**: After training, analyze which features are active. Are esoteric features used now (with new weekly features added)? Category breakdown (TA vs time vs esoteric).

### Agent 15: Esoteric Signal Validator
- **Role**: error-checker
- **Task**: Verify astrology, numerology, gematria features are correctly computed for weekly resolution. Check new features (eclipse, natal transit, halving cycle) have non-zero variance.
- **Perplexity-enabled**: Query about expected esoteric signal strength on weekly data.

### Agent 16: Trade Strategy Assessor
- **Role**: error-checker
- **Task**: After optimizer runs, analyze: Sortino, ROI, drawdown, trade count. Is the strategy tradable? Enough trades for statistical significance? PBO score assessment.

---

## TEAM 6: SPEED & INFRASTRUCTURE (2 agents)

### Agent 17: Speed Bottleneck Hunter
- **Role**: error-checker
- **Task**: Time every pipeline step. Compare with ETA estimates. Find any step that's slower than expected. Check GPU utilization, CPU utilization, I/O wait. Identify next bottleneck to optimize.

### Agent 18: Validation & Docs Updater
- **Role**: backend-dev
- **Task**: After all fixes applied, add new validate.py checks for every bug found. Update SESSION_RESUME.md with results. Update CLAUDE.md if new rules discovered.

---

## TEAM 7: STRATEGIC (2 agents)

### Agent 19: Next-TF Planner
- **Role**: error-checker
- **Task**: Based on 1w results, plan the 1d training run. What machine? What additional features? Should we skip 1w and focus on larger TFs where esoteric features have more data?
- **Perplexity query**: "Which BTC timeframe gives best signal-to-noise for esoteric features?"

### Agent 20: Matrix Thesis Guardian
- **Role**: error-checker
- **Task**: Verify NO changes violate the matrix thesis. All features preserved. No filtering. No subsampling. feature_fraction >= 0.7. NaN = missing. 0 = value. Every signal matters.

---

## SYSTEM PROMPT FOR ALL AGENTS

```
MATRIX THESIS: This BTC trading system uses 2.9M+ features including esoteric signals (astrology, gematria, numerology, sacred geometry, space weather). More diverse signals = stronger predictions. The edge IS the matrix. NEVER filter, subsample, or regularize away rare signals.

TF: 1w (weekly). 819 bars. This is the HARDEST timeframe — fewest rows, most noise. But we train it to validate the pipeline before moving to larger TFs.

MACHINE: vast.ai Instance 33852303, 1x RTX 5090 32GB, EPYC 7B12 128c, 258GB RAM
SSH: ssh -p 12302 root@ssh7.vast.ai

RULES:
- Consult Perplexity (with matrix thesis context) for ANY technical uncertainty
- Document EVERYTHING — this is scientific
- NO CPU fallbacks for training — fail loud
- Cut training short if results are clearly bad
- ALL cores and GPU utilized
- feature_pre_filter=False, feature_fraction >= 0.7, bagging >= 0.7
- NaN = missing, 0 = value. NEVER fillna(0).

GitHub: https://github.com/c19o/TB-3.3 (remote: tb33, branch: v3.3-clean)
```
