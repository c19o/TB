# AUDIT 8: Self-Learning Trading Systems — Perplexity Research Report
**Date:** 2026-03-21
**Source:** Perplexity AI (best model) — ML/engineering expertise only

---

## TABLE OF CONTENTS
1. [Self-Learning Trading Systems](#1-self-learning-trading-systems)
2. [Trade Outcome Analysis Frameworks](#2-trade-outcome-analysis-frameworks)
3. [Reinforcement Learning for Trade Execution](#3-reinforcement-learning-for-trade-execution)
4. [Feature Drift Detection in Production ML](#4-feature-drift-detection-in-production-ml)
5. [Continuous Model Validation](#5-continuous-model-validation)
6. [Actionable Recommendations for Savage22](#6-actionable-recommendations-for-savage22)

---

## 1. SELF-LEARNING TRADING SYSTEMS

### Online Learning with XGBoost (Warm-Start)

XGBoost does NOT support true single-sample online learning. The production pattern is **mini-batch warm-start retraining**:

1. Train initial model on large backtest window, save the booster
2. As new data accrues (daily/weekly), call `xgb.train()` with `xgb_model=existing_booster` to add trees
3. Control plasticity via `n_estimators`, `learning_rate`, and regularization

```python
import xgboost as xgb

# Initial training
booster = xgb.train(params, dtrain_initial, num_boost_round=200)
booster.save_model("model.json")

# Periodic warm-start update
booster = xgb.Booster()
booster.load_model("model.json")
booster = xgb.train(params, dtrain_new_batch, num_boost_round=50, xgb_model=booster)
booster.save_model("model.json")
```

**Critical warning:** Uncontrolled infinite warm-starts accumulate bias. Periodically rebuild from scratch on a fresh rolling window to "re-anchor" the model.

### Replay Buffers for Trading

For supervised alpha models (like Savage22's XGBoost), the analog of RL replay buffers is:

- Maintain a **rolling window dataset** of recent (features, label, realized PnL) tuples
- Cap history length (6-24 months) so ancient regimes don't dominate
- **Decay sample weights** instead of hard-dropping old data
- Store both DECISIONS (signals/orders) and OUTCOMES (slippage, PnL, drawdown) for later cost-aware retraining
- When retraining, **stratify mini-batches by market regime or volatility bucket** to avoid over-weighting high-vol days

### Feature Importance Drift Detection

Track over time:
- SHAP or gain-based importance rankings
- Alert when rankings or weights change materially across windows
- Combine with KS-test or PSI on key features vs training baseline

### Retrain Triggers — The Three-Trigger System

Production systems combine:

| Trigger Type | Example |
|---|---|
| **Time-based** | Weekly rolling retrain on last N months |
| **Performance-based** | OOS Sharpe or hit-rate drops below threshold for K days |
| **Drift-based** | ADWIN or KSWIN signals shift in error distribution |

**Recommended schedule for Savage22:**
- **Daily:** Score with frozen model; log features, predictions, realized returns
- **Weekly:** Recompute metrics, drift, and feature importance on last 3-6 months; skip retrain if stable
- **On trigger:** Retrain from scratch on rolling window, or warm-start if tests show no cumulative bias

### Avoiding Catastrophic Forgetting

| Strategy | Implementation |
|---|---|
| Wide training window | Include multiple regimes (both low- and high-volatility periods) |
| Sample weighting | Recent data gets higher weight, but old data is never zeroed out |
| Regularization | Strong L1/L2 and tree depth constraints on XGBoost |
| Periodic full rebuild | Treat warm-starting as short runs between full retrains, not infinite |
| Regime stratification | Balance sampling across bull/bear/sideways/high-vol/low-vol |

### Libraries
- `xgboost` — warm-start via `xgb_model` parameter
- `river` (formerly creme) — online drift detectors (ADWIN, Page-Hinkley, KSWIN)
- `shap` — feature importance drift analysis
- `stable-baselines3` / `ray[rllib]` — if adding RL overlay later

### Papers
- "Concept Drift Detection in Finance" (AUA capstone) — evaluates ADWIN, KSWIN, Page-Hinkley on financial time series
- "Applying Neural Networks for Concept Drift Detection in Financial Time Series" — drift-curve approach for multivariate financial data
- Stefan Jansen, *Machine Learning for Trading* — XGBoost/LightGBM practical retraining/validation

### Common Pitfalls
- Treating warm-start as truly online (it's not — bias accumulates)
- Leaking future information in feature engineering during simulated online retraining
- Overreacting to noise: triggering retrain on every short-term wobble
- Ignoring transaction costs when updating models faster than deployment allows
- No out-of-time validation for the retrain policy itself

---

## 2. TRADE OUTCOME ANALYSIS FRAMEWORKS

### MAE/MFE Analysis

**Maximum Adverse Excursion (MAE):** Worst unrealized loss from entry to exit
**Maximum Favorable Excursion (MFE):** Best unrealized profit during the life of the trade

**Implementation choices:**
- Compute BOTH price-based and PnL-based excursions
- Use bar high/low for theoretical extremes, OR running PnL snapshots for realistic fills
- Store per trade alongside entry/exit price, qty, fees, realized PnL, and end-trade drawdown

**Key uses:**
- **Stop-loss calibration:** Histogram MAE by entry regime, choose stop where most winners survive and most losers breach
- **Take-profit tuning:** Compare realized PnL to MFE to spot "exiting too early/late" patterns

### Trade Efficiency Metrics

| Metric | Formula | Meaning |
|---|---|---|
| **Exit Efficiency** | realized_profit / MFE | How much of possible profit was captured |
| **Entry Efficiency** | realized_profit relative to MAE | How much initial heat was paid for the edge |
| **Total Efficiency** | Fraction of total price move captured (-100% to +100%) | Net capture |

**Best practice:** Bucket trades by features (signal decile, volatility regime, holding period) and compute distributions. Low exit efficiency with high MFE = work on exits, not entries.

### Signal Attribution for Multi-Factor Models

From CFM's "Signal-wise Performance Attribution for Constrained Portfolio Optimisation":

1. At each rebalance, store signal vectors for each factor
2. Store optimizer outputs including Lagrange multipliers/dual variables
3. Construct per-signal "shadow portfolios" whose trades sum to actual trades
4. Compute per-signal PnL, turnover, risk contributions, and MAE/MFE distributions
5. Method explicitly handles constraints and transaction costs

**For Savage22:** Tag each trade with the signal state at entry time (all factor exposures, prediction score, confidence). This links every trade back to which signals drove it.

### Post-Trade Attribution — Four-Layer Stack

1. **Feature/Signal Diagnostics:** IC time series and decay, quantile return spreads, turnover
2. **Signal Analysis:** Compare models/signals on IC, hit-rate, Sharpe by bucket. Use SHAP to understand what drives predictions and failures
3. **Trade-Level Analysis:** Win rate, avg win/loss, profit factor, holding periods, slippage, excursion analysis. Trade-SHAP: explain worst trades to find consistent failure patterns
4. **Portfolio-Level Attribution:** PnL contribution by factor, risk contributions, drawdown decomposition

### Required Trade DataFrame Schema

```python
# Per-trade record
columns = [
    'trade_id', 'instrument', 'side', 'entry_time', 'exit_time',
    'entry_price', 'exit_price', 'qty', 'fees',
    'realized_pnl', 'holding_bars',
    'price_MAE', 'price_MFE',        # from bar high/low path
    'pnl_MAE', 'pnl_MFE',            # from running PnL vector
    'exit_efficiency', 'total_efficiency',
    'model_prediction', 'model_confidence',
    'signal_snapshot_json',            # all factor values at entry
    'regime_label', 'volatility_bucket'
]
```

### Libraries
- QuantConnect LEAN — TradeBuilder/Trade objects with built-in MAE/MFE
- `ml4t-diagnostic` (PyPI) — trade analysis, excursion-based TP/SL optimization, trade-SHAP
- `vectorbt` / `backtrader` — backtest engines with extensible trade logging
- Standard: `pandas`, `numpy`, `statsmodels`

### Papers
- John Sweeney, *Maximum Adverse Excursion* (1997) — foundational work on MAE/MFE distributions
- CFM, "Signal-wise Performance Attribution for Constrained Portfolio Optimisation" (2014) — multi-factor attribution
- "Machine Learning Enhanced Multi-Factor Quantitative Trading" — SHAP-driven trade diagnostics

### Common Pitfalls
- Using bar high/low for MAE/MFE when you can't actually trade inside the bar
- Ignoring partial fills / scale-ins / scale-outs (must recompute PnL path for evolving position size)
- Not conditioning MAE/MFE by regime (aggregated distributions hide regime-specific failures)
- Overfitting stops/targets to backtest MAE/MFE (treat as hyperparameters subject to walk-forward validation)

---

## 3. REINFORCEMENT LEARNING FOR TRADE EXECUTION

### RL for Entry/Exit Timing — Overlay Architecture

The consensus: do NOT let RL "discover" alpha. Feed it pre-computed signals and let it learn WHEN and HOW AGGRESSIVELY to act.

**Recommended architecture for Savage22:**
- Keep the current XGBoost signal-driven strategy as baseline
- RL outputs a **multiplicative overlay** (0-1 throttle on trade aggressiveness)
- This makes training stable and allows direct comparison vs the known benchmark

**State space:**
- Recent returns, your alpha signals (XGBoost predictions), volatility/regime indicators
- Current position, inventory, spread
- Confidence scores from the model

**Action space:**
- Discrete: hold / enter / exit / scale up / scale down
- OR continuous: target position between bounds

**Reward design:**
- PnL minus penalties for turnover, drawdown, inventory risk, and slippage
- Shape reward with both expectation and variance to control risk

### RL for Position Sizing (Kelly Alternative)

Kelly assumes known edge and variance — in practice, both are noisy and time-varying. RL can learn a mapping from state to size fraction.

**Practical approach — RL as Kelly scaler:**
- Define an "anchor" like fractional Kelly or volatility-targeted size
- RL outputs a factor that scales it up/down based on regime
- More robust than free-form sizing

**State for sizing agent:**
- Predicted win probability / Sharpe from XGBoost
- Recent realized Sharpe, volatility, drawdown
- Rolling hit-rate of the strategy

**Action:** Continuous size in [0, s_max] or discrete buckets (0, 0.25, 0.5, 0.75, 1) of max Kelly

### Model-Based vs Model-Free RL

| Approach | When to Use |
|---|---|
| **Model-Free** (PPO, SAC, DQN) | End-of-day allocation, slower rebalancing, lots of historical episodes |
| **Model-Based** (environment simulator) | Intraday execution, order type selection, fill probability modeling |
| **Hybrid** (recommended) | Use historical data as realistic simulator, run model-free algos inside it |

**For Savage22 (BTC, multi-timeframe):** Start with model-free PPO or SAC as an overlay on top of the existing XGBoost signals. The realistic approach is a throttle/overlay, not replacement.

### Implementation Stack

```
Backtester + Execution Simulator (custom, wrapping historical bar data)
    |
    v
Custom Gymnasium-style Env (states = existing signals + risk metrics)
    |
    v
Agent: PPO or SAC from Stable-Baselines3
    |
    v
Workflow: Train on rolling windows → validate OOS → paper trade → gradual live
```

### Libraries
- `stable-baselines3` — PPO, A2C, SAC, TD3, DQN (most used for trading research)
- `ray[rllib]` — distributed training, hyperparameter sweeps
- `FinRL` / `RL-FIN` — domain-specific DRL for finance with prebuilt crypto environments

### Papers
- Nevmyvaka et al. — Deep RL for optimal execution
- "Reinforcement Learning for Trade Execution with Market and Limit Orders" (2025)
- "Reinforcement Learning for Optimal Trade Execution" — reward shaping by cost expectation and variance
- Quantitative Brokers whitepaper, "Reinforcement Learning For Trade Execution" (2023)
- FinRL papers — end-to-end DRL trading pipelines

### Common Pitfalls
- **Reward hacking:** Agent exploits simulator quirks, not real market structure
- **Non-stationarity:** Policies trained on one regime degrade fast — need rolling retraining
- **Overfitting:** Deep agents + limited independent episodes = disaster without regularization
- **Poor state design:** Omitting inventory, spread, volatility causes unstable behavior
- **Ignoring partial observability:** Markets are not fully observable — consider LSTM/GRU in the agent

### Verdict for Savage22
RL is a **Phase 3+ addition**, not a Phase 1 priority. The immediate value is as an execution overlay or position sizing scaler on top of the existing XGBoost system. Do NOT attempt to replace the signal generation with RL.

---

## 4. FEATURE DRIFT DETECTION IN PRODUCTION ML

### Population Stability Index (PSI)

Compares binned feature distributions between reference and production windows.

```
PSI = SUM[ (q_b - p_b) * ln(q_b / p_b) ]  for each bin b
```

**Thresholds (industry standard from credit risk/finance):**
| PSI Value | Interpretation |
|---|---|
| < 0.10 | Negligible drift |
| 0.10 - 0.25 | Moderate, watch |
| > 0.25 | Material drift, investigate/alert |

**Best practice:** Use quantile bins (10-20 buckets), not equal-width, for fat-tailed returns data.

### KL Divergence and Alternatives

| Metric | Properties | Best For |
|---|---|---|
| **KL Divergence** | Asymmetric, unbounded | Theory, but handle zero-bins carefully |
| **Jensen-Shannon** | Symmetric, bounded | Embeddings, latent features |
| **Wasserstein / EMD** | Captures shape changes | When distribution shape matters (uni→bimodal) |
| **KS Test** | Hypothesis test with p-values | Small set of critical numeric features |

**Recommended combo for Savage22:**
- PSI for broad monitoring across many features (dashboarding)
- KS or Wasserstein for critical features (spread, realized vol, order-book imbalance)

### SHAP Value Drift — The Critical Layer

SHAP drift detects when the **model's decision logic changes**, not just when inputs shift.

**Why this matters more than raw feature drift:**
- Raw feature drift doesn't necessarily break the model if P(y|x) is stable
- SHAP drift highlights when "spread" suddenly dominates prediction where "order-book imbalance" used to

**Monitoring pattern:**
1. Choose reference period with good performance
2. Compute SHAP values on reference and rolling production windows (daily)
3. Monitor: drift in mean/variance of SHAP values per feature, changes in feature ranking, new SHAP clusters

### The Decision Framework: Retrain vs Regime Change

| Scenario | Feature Drift | Performance | SHAP Drift | Action |
|---|---|---|---|---|
| **1. Safe** | Elevated | Still OK | Stable | Watch, no urgent retrain |
| **2. Retrain needed** | Elevated | Degraded | Stable (same reasoning) | Retrain / extend training window |
| **3. Regime change** | Moderate | Degraded | SHAP reshuffled | Model architecture or feature-set change + regime modeling |
| **4. Known crisis** | Extreme | Degraded | N/A | Switch to dedicated crisis model |

**Alert runbook for Savage22:**
- **YELLOW:** 3+ critical features exceed PSI 0.25 or p-value < 0.01, but estimated performance within tolerance
- **RED (retrain):** PSI high AND PnL out of band → schedule retrain
- **RED (regime):** SHAP attribution drift (KS between SHAP distributions > threshold) AND regime indicators suggest structural change → review model design + risk limits

### Libraries — Comparison

| Goal | Library | Strengths |
|---|---|---|
| Many tabular features, reports | **Evidently AI** | PSI/KS/JS/Wasserstein out of box, dataset-level drift flag, HTML reports |
| Label-delayed performance | **NannyML** | CBPE to estimate performance without labels (PnL arrives late) |
| Custom/kernel drift | **alibi-detect** | MMD/KS/CvM on raw or latent features, online detectors |

```python
# Evidently: broad feature drift
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref_df, current_data=prod_df)
result = report.as_dict()
share_drifted = result['metrics'][0]['result']['dataset_drift']['share_of_drifted_columns']

# alibi-detect: MMD drift on feature matrix
from alibi_detect.cd import MMDDrift
import numpy as np

cd = MMDDrift(ref_df.values.astype(np.float32), p_val=0.05)
preds = cd.predict(prod_df.values.astype(np.float32))
drift_flag = preds['data']['is_drift']
```

### Papers
- "Explaining Drift using Shapley Values" (DBShap) — decomposes performance drift into real vs virtual drift
- "Feature-based Analyses of Concept Drift" — formal feature-level drift + feature selection for monitoring
- "Machine Learning for Financial Prediction Under Regime Change" — regime switching and concept drift in finance
- D3Bench — benchmarks Evidently, NannyML, alibi-detect empirically

### Common Pitfalls
- **Over-sensitivity at scale:** KS tests flag trivial shifts at production sample sizes — combine statistical significance with effect size (PSI magnitude)
- **Ignoring joint drift:** Per-feature PSI misses correlation drift. Use MMD-based multivariate detection
- **Using training data as reference:** Biases CBPE estimates and underestimates real drift
- **Not accounting for label delay:** If relying only on realized PnL, you react too slowly. CBPE or proxy labels give earlier warning
- **Treating every drift as "retrain now":** Frequent retrains on tiny drifts cause instability and overfit to noise

---

## 5. CONTINUOUS MODEL VALIDATION

### Rolling Sharpe Ratio Monitoring

```python
import pandas as pd
import numpy as np

def rolling_sharpe(pnl_series, window=126, periods_per_year=252):
    roll = pnl_series.rolling(window)
    mu = roll.mean()
    sigma = roll.std(ddof=1)
    sharpe = np.sqrt(periods_per_year) * (mu / sigma)
    return sharpe.replace([np.inf, -np.inf], np.nan)

df["rolling_sharpe_6m"] = rolling_sharpe(df["pnl"], window=126)
```

**Control limits for Savage22:**
- If 6-month rolling Sharpe < 0 AND 1-year < 0.3 → raise attention
- If both < 0 for N consecutive days → trigger formal review
- Track BOTH strategy Sharpe and relative Sharpe vs benchmark (buy-and-hold BTC)
- Combine with rolling max drawdown, volatility, and tail loss (Sharpe alone is blind to skew)

### Sequential Testing for Degradation

**CUSUM / Page-Hinkley:** Online change-point detection on PnL or per-trade edge.

Page-Hinkley formulation:
```
g_t = g_{t-1} + (x_t - v)         where v = small tolerance
G_t = min(G_{t-1}, g_t)
ALARM if g_t - G_t > h            where h = threshold
```

**In practice:**
- Run on per-trade log returns or per-bar PnL, normalized by volatility
- Tune v and h with historical walk-forward using synthetic drifts to measure detection delay vs false positives
- Reset detector after confirmed drift and model switch

```python
from river import drift

detector = drift.ADWIN()
for i, r in enumerate(trade_returns_stream):
    detector.update(r)
    if detector.change_detected:
        # Log, freeze position size, evaluate model switch
        detector.reset()
```

**Use these as soft signals feeding a higher-level decision layer, NOT as auto-kill switches.**

### Adaptive Model Selection — The Ensemble Router

Maintain a small set of candidate models:
- **"Recent"** — trained heavily on last N months (high plasticity)
- **"Stable/Old"** — long-history, lower variance (robustness)
- **"Regime-specific"** — optional dedicated model for high-vol or crisis regimes

**Selection mechanisms:**
| Method | Description |
|---|---|
| **Periodic re-ranking** | Every K trades/days, compare models on rolling OOS window, allocate proportionally to recent risk-adjusted performance |
| **Multi-armed bandit** | Treat models as arms, allocate more flow to higher recent reward with exploration |
| **Dynamic ensembles** | Combine with weights that evolve (Hedge / Exponentially Weighted Average Forecaster) |

**Switching policy:**
- Maintain per-model rolling Sharpe, drawdown, and drift detectors
- Define guardrails: minimum live sample size, max turnover in "who's primary," transaction-cost-aware switching thresholds
- Switch ONLY when:
  - Rolling Sharpe underperforms conservative baseline by a margin exceeding estimated transaction costs
  - Degradation confirmed by at least one sequential test AND persists for minimum dwell time
- On switch: reduce size or halt degraded model, promote best backup, optionally trigger retrain with recent-regime weighting

### The Two-Tier Monitoring Architecture

```
TIER 1 (Low Latency — Tripwires)
  CUSUM / Page-Hinkley / ADWIN on per-trade returns
  → Fast detection, may have false positives
  → Action: flag for review, optionally reduce position size

TIER 2 (High Confidence — Governor)
  Rolling Sharpe + drawdown + relative performance
  → Slower but more reliable
  → Action: model switch, retrain trigger, risk limit adjustment
```

### Libraries
- `river` — ADWIN, KSWIN, DDM, EDDM online drift detectors
- `alibi-detect` — MMD, KL, classifier-based drift
- `QSTrader` / `backtrader` — rolling returns and Sharpe computation
- Custom monitoring infrastructure for alerts

### Papers
- CUSUM / Page-Hinkley — standard statistical change-detection, repurposed for ML
- Adaptive model selection and dynamic ensembles in financial time series
- "Performance and Risk of an AI-Driven Trading Framework" — robust risk-adjusted metrics and live monitoring

### Common Pitfalls
- Windows too short or thresholds too tight → churn and cost bleed
- Ignoring transaction costs in switching decisions
- Optimizing drift-detection parameters on same history used to evaluate the strategy
- Using only Sharpe → ignoring fat tails and correlated failures across models

---

## 6. ACTIONABLE RECOMMENDATIONS FOR SAVAGE22

### Priority 1 — Implement Now (Low Effort, High Value)

**A. Trade Logging with MAE/MFE**
- Extend `paper_trades.db` / `trades.db` schema to include:
  - `price_MAE`, `price_MFE`, `pnl_MAE`, `pnl_MFE`
  - `exit_efficiency` (realized_pnl / MFE)
  - `signal_snapshot` (JSON of all factor values at entry time)
  - `regime_label`, `volatility_bucket`
- This is the foundation for everything else. Without per-trade signal attribution, you cannot diagnose what's working.

**B. Rolling Sharpe Monitoring**
- Add rolling 126-day and 252-day Sharpe to the paper trader dashboard
- Set alert thresholds: yellow if 6m Sharpe < 0.3, red if < 0
- Compare vs buy-and-hold BTC as baseline

**C. Feature Importance Logging**
- After each model training, save SHAP importance rankings with timestamp
- Weekly comparison of top-20 feature rankings vs previous training
- Alert if rankings reshuffle dramatically

### Priority 2 — Build Next (Medium Effort, High Value)

**D. Drift Monitoring Pipeline**
- Install Evidently AI for broad PSI/KS monitoring across all features
- Set up weekly drift reports comparing last 30 days vs training reference
- Implement the YELLOW/RED alert runbook from Section 4
- Add SHAP drift monitoring (compare SHAP distributions, not just feature distributions)

**E. Three-Trigger Retrain System**
- Time-based: weekly retrain eligibility check
- Performance-based: rolling Sharpe below threshold for 5+ days
- Drift-based: ADWIN detector on per-trade returns (from `river` library)
- Any trigger fires → retrain from scratch on rolling window (do NOT infinitely warm-start)

**F. Model Versioning and A/B Testing**
- Save every model artifact with timestamp, training window, and feature set
- Maintain "current" and "challenger" models running in parallel on paper
- Promote challenger only when it beats current on OOS for minimum dwell time

### Priority 3 — Future Phase (High Effort, Speculative Value)

**G. RL Execution Overlay**
- Build a Gymnasium-style environment wrapping the existing backtester
- State = XGBoost predictions + regime indicators + position + drawdown
- Action = trade aggressiveness throttle (0-1 multiplier)
- Start with PPO from Stable-Baselines3
- Only after the signal system is validated and stable

**H. RL Position Sizing Scaler**
- Define Kelly/volatility-target anchor
- RL learns to scale the anchor up/down based on regime
- Requires stable base system with sufficient trade history

**I. Adaptive Model Ensemble**
- Maintain "recent" + "stable" + "crisis" models
- Multi-armed bandit or exponentially weighted average for selection
- Requires extensive backtesting of the selection mechanism itself

### Implementation Order and Dependencies

```
Phase 1 (NOW):
  Trade Logging → Rolling Sharpe → Feature Importance Logging

Phase 2 (NEXT):
  Drift Monitoring → Three-Trigger Retrain → Model Versioning
  (Depends on Phase 1 data collection)

Phase 3 (FUTURE):
  RL Overlay → RL Sizing → Adaptive Ensemble
  (Depends on Phase 2 infrastructure being solid)
```

### Key Libraries to Install

```bash
pip install river           # Online drift detectors (ADWIN, Page-Hinkley, KSWIN)
pip install evidently        # Feature drift monitoring (PSI, KS, JS, Wasserstein)
pip install nannyml          # Performance estimation without labels (CBPE)
pip install alibi-detect     # Advanced drift detection (MMD, kernel-based)
pip install shap             # Already likely installed — SHAP value drift
pip install stable-baselines3  # RL agents (Phase 3)
pip install ml4t-diagnostic  # Trade diagnostics, excursion-based analysis
```

### Reading List (Prioritized)

1. **Stefan Jansen, Machine Learning for Trading** — XGBoost/LightGBM practical retraining (Chapter 12)
2. **John Sweeney, Maximum Adverse Excursion (1997)** — MAE/MFE foundational work
3. **CFM, "Signal-wise Performance Attribution for Constrained Portfolio Optimisation" (2014)** — multi-factor attribution
4. **"Explaining Drift using Shapley Values" (DBShap)** — SHAP-driven drift decomposition
5. **"Machine Learning for Financial Prediction Under Regime Change"** — regime switching survey
6. **D3Bench paper** — empirical comparison of Evidently, NannyML, alibi-detect
7. **"Concept Drift Detection in Finance" (AUA capstone)** — ADWIN/KSWIN/Page-Hinkley on financial data
8. **Quantitative Brokers, "Reinforcement Learning For Trade Execution" (2023)** — practitioner RL for execution

---

*Report generated by Audit Agent 8. All recommendations are ML/engineering best practices. Signal generation strategy (the Matrix) is out of scope for this audit.*
