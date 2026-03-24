# Institutional Upgrade Roadmap — Tier 2 & 3

## Tier 2 — Add at $10K-$50K

### 6. OMS Layer (Order Management System)
Proper order state machine: pending → new → partial → filled → canceled.
Idempotent handling, reconciliation vs exchange fills.
Not just "send market order" — track every order lifecycle event.

### 7. Execution Algos
TWAP/limit ladder for entries, spread-aware timing.
Sub-candle entry (v3.1) is the seed — extend to full execution algo framework.
Pegged orders, child orders, smart limit vs market based on spread/depth/volatility.

### 8. Live Risk Dashboard + Alerts
Real-time: current exposure, leverage, PnL vs limits, drawdown, regime state.
Alerts via Slack/Discord on: limit breach, data staleness, NaN model output,
order rejects, missed fills, 5-sigma PnL anomaly.
Liveness heartbeat for data feeds + execution connection.

### 9. Execution Quality Tracking
Log expected vs actual fill price on every trade.
Measure real slippage vs model assumption per TF, per venue, per time-of-day.
Feed measured slippage back into optimizer (close the loop).

### 10. Signal Decay Analysis
Measure alpha degradation with entry delay (1 bar, 2 bars, 5 bars).
Per-TF decay curves tell you how time-sensitive each signal is.
Informs sub-candle urgency and execution algo aggressiveness.

## Tier 3 — Add at $50K-$100K+

### 11. Capacity / Impact Analysis
At what position size does the edge disappear?
Max % of venue volume per trade. Empirical impact curves.
Stress test: 2x spread + 50% depth collapse. Check survival at target AUM.

### 12. Factor Attribution
Decompose PnL into: esoteric alpha, TA alpha, regime timing, execution slippage, fees.
Daily breakdown. Know WHERE the edge comes from and if it's shifting.
If esoteric features stop contributing, investigate (regime shift? data quality?).

### 13. Cross-Strategy Correlation Monitoring
Do TF models crash simultaneously? Track rolling correlation of per-TF PnLs.
Cap correlated exposure. If 1d and 4h both say long with 0.9 correlation,
that's one bet not two — size accordingly.

### 14. Stress Tests
Simulate: 50% gap down, spread 10x, exchange down 4hrs, funding rate spike.
Check survival at each AUM milestone (25K, 50K, 100K).
Tail risk hedging overlays if warranted.

### 15. Multi-Venue Smart Order Router
Route across exchanges by fee tier, rebate, and depth.
Even 2-3 bps saved per trade compounds massively at scale.
