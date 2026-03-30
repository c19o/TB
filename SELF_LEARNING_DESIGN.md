# Savage22 Self-Learning System Design
## Audit Agent 5: Trade Outcome Feedback Loop + Continuous Improvement

---

## 1. CURRENT STATE ASSESSMENT

### What Is Logged Per Trade Today

**trades.db `trades` table** (v1 + v2 live_trader.py):
```
id, tf, direction, confidence, entry_price, entry_time, exit_price, exit_time,
stop_price, tp_price, pnl, pnl_pct, bars_held, exit_reason, regime, leverage,
risk_pct, features_json, status, created_at
```

**What `features_json` contains**: ALL pruned feature values at entry time (the full model input vector), stored as `{feature_name: rounded_value}`. This is the single most valuable piece of data for self-learning — it already exists.

**What is NOT logged today (gaps)**:
- No features at EXIT time
- No max adverse excursion (MAE) or max favorable excursion (MFE) during the trade
- No bar-by-bar price path during the hold period
- No record of which individual signals were active vs inactive at entry
- No XGBoost class probabilities breakdown (only final `confidence`)
- No meta-labeling probability or LSTM probability (they modify `confidence` in-place before storing)
- No record of BLOCKED trades (confluence filter, meta-gate rejections, DD halts)
- No feature importance snapshot per trade
- No optimal exit timing comparison

**paper_trades.db** has a `signals_fired` table that tracks per-signal events — useful for signal quality tracking but disconnected from the ML pipeline.

---

## 2. DESIGN: SELF-LEARNING SYSTEM

### Architecture Overview

```
live_trader.py
    |
    +---> [ENTRY] ---> trade_journal.db::trade_snapshots (full feature vector + metadata)
    |
    +---> [TICK MONITOR] ---> trade_journal.db::price_path (MAE/MFE/bar-by-bar)
    |
    +---> [EXIT] ---> trade_journal.db::trade_outcomes (outcome analysis)
    |
    +---> [BLOCKED TRADES] ---> trade_journal.db::rejected_trades (what we DIDN'T take)
    |
    v
self_learner.py (periodic batch analysis)
    |
    +---> Signal Quality Tracker (rolling accuracy per signal)
    +---> Feature Drift Detector (importance shift over time)
    +---> Meta-Label Auditor (is the gate calibrated?)
    +---> Kelly Auditor (over/under-betting?)
    +---> Regime Calibrator (are multipliers correct?)
    +---> Retrain Trigger (when to schedule cloud retraining)
    |
    v
retrain_scheduler.py (triggers cloud retraining when thresholds met)
```

---

## A. TRADE OUTCOME FEEDBACK LOOP

### A1. New Database: `trade_journal.db`

Separate from `trades.db` to avoid bloating the live trading DB. The journal is append-only and designed for post-hoc analysis.

**Table: `trade_snapshots`** — Full state at entry time
```sql
CREATE TABLE trade_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id INTEGER NOT NULL,          -- FK to trades.db trades.id
    tf TEXT NOT NULL,
    direction TEXT NOT NULL,
    entry_time TEXT NOT NULL,
    entry_price REAL NOT NULL,

    -- Model outputs (BEFORE blending/gating)
    xgb_prob_long REAL,
    xgb_prob_flat REAL,
    xgb_prob_short REAL,
    lstm_prob REAL,                      -- NULL if no LSTM model
    meta_prob REAL,                      -- NULL if no meta-model
    blended_confidence REAL,             -- final after all blending

    -- Regime state
    regime TEXT,
    regime_idx INTEGER,
    hmm_bull_prob REAL,
    hmm_bear_prob REAL,
    hmm_neutral_prob REAL,
    hmm_state INTEGER,

    -- Position sizing inputs
    kelly_fraction REAL,
    base_risk_pct REAL,
    final_risk_pct REAL,
    leverage_used REAL,
    dd_scale REAL,
    portfolio_dd_pct REAL,
    tf_pool_dd_pct REAL,

    -- Trade params
    stop_atr_mult REAL,
    rr_ratio REAL,
    max_hold_bars INTEGER,
    atr_14 REAL,
    confluence_parent_tf TEXT,
    confluence_parent_dir INTEGER,
    confluence_scale REAL,

    -- Full feature vector (compressed JSON)
    features_json TEXT NOT NULL,         -- ALL feature values at entry

    -- Key signal states at entry (extracted from features for fast querying)
    -- Esoteric signals
    moon_phase REAL,
    mercury_retro INTEGER,
    kp_index REAL,
    gematria_sum_price INTEGER,
    numerology_date_sum INTEGER,
    astro_dominant_sign TEXT,

    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_snap_tf ON trade_snapshots(tf);
CREATE INDEX idx_snap_time ON trade_snapshots(entry_time);
```

**Table: `price_path`** — Bar-by-bar tracking during open position
```sql
CREATE TABLE price_path (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id INTEGER NOT NULL,
    bar_num INTEGER NOT NULL,            -- 0 = entry bar, 1 = next bar, etc.
    timestamp TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL,
    volume REAL,

    -- Running metrics
    unrealized_pnl_pct REAL,
    max_adverse_excursion REAL,          -- worst drawdown from entry (negative)
    max_favorable_excursion REAL,        -- best profit from entry (positive)
    distance_to_stop_pct REAL,           -- how close to SL (0 = at SL, 1 = at entry)
    distance_to_tp_pct REAL,             -- how close to TP (0 = at entry, 1 = at TP)

    UNIQUE(trade_id, bar_num)
);
CREATE INDEX idx_path_trade ON price_path(trade_id);
```

**Table: `trade_outcomes`** — Post-close analysis
```sql
CREATE TABLE trade_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id INTEGER NOT NULL UNIQUE,
    tf TEXT NOT NULL,
    direction TEXT NOT NULL,

    -- Outcome
    pnl REAL,
    pnl_pct REAL,
    exit_reason TEXT,                    -- SL, TP, TIME, MANUAL
    bars_held INTEGER,

    -- Excursion analysis
    max_adverse_excursion REAL,          -- worst point during trade
    max_favorable_excursion REAL,        -- best point during trade
    mae_bar INTEGER,                     -- which bar had worst point
    mfe_bar INTEGER,                     -- which bar had best point
    mfe_before_mae INTEGER,              -- 1 if best came before worst (was right, then reversed)

    -- Timing analysis
    optimal_exit_bar INTEGER,            -- bar with best P&L (hindsight)
    optimal_exit_pnl REAL,               -- P&L at optimal exit
    timing_efficiency REAL,              -- actual_pnl / optimal_pnl (0 to 1+)
    entry_timing_error_bars INTEGER,     -- bars between entry and actual turn

    -- Features at exit time
    exit_features_json TEXT,             -- full feature snapshot at exit

    -- What the model predicted vs reality
    predicted_direction TEXT,            -- LONG or SHORT
    actual_direction TEXT,               -- LONG or SHORT (based on price move)
    prediction_correct INTEGER,          -- 1 or 0
    predicted_confidence REAL,

    -- Signal breakdown
    active_signals_json TEXT,            -- which esoteric/TA signals were active at entry
    correct_signals_json TEXT,           -- which signals aligned with actual outcome
    incorrect_signals_json TEXT,         -- which signals were wrong

    -- Post-trade price action (what happened AFTER exit)
    price_1bar_after REAL,
    price_5bars_after REAL,
    price_10bars_after REAL,
    continued_in_direction INTEGER,      -- 1 if price kept going our way after exit

    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_outcome_tf ON trade_outcomes(tf);
CREATE INDEX idx_outcome_time ON trade_outcomes(created_at);
```

**Table: `rejected_trades`** — Trades we did NOT take (critical for learning)
```sql
CREATE TABLE rejected_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tf TEXT NOT NULL,
    direction TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    price REAL,
    confidence REAL,

    -- Why it was rejected
    rejection_reason TEXT,               -- 'meta_gate', 'confluence_block', 'dd_halt',
                                         -- 'below_threshold', 'tf_halted', 'duplicate'
    meta_prob REAL,                       -- meta-label probability (if meta_gate)
    confluence_parent TEXT,               -- parent TF (if confluence_block)
    confluence_parent_dir INTEGER,

    -- What would have happened (filled in post-hoc)
    hypothetical_exit_price REAL,
    hypothetical_pnl_pct REAL,
    hypothetical_exit_reason TEXT,
    was_correct_rejection INTEGER,        -- 1 if rejection saved money, 0 if missed profit

    features_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_rejected_tf ON rejected_trades(tf);
CREATE INDEX idx_rejected_reason ON rejected_trades(rejection_reason);
```

### A2. Modifications to `live_trader.py`

**At ENTRY (line ~699 in v1, where INSERT INTO trades happens):**

```python
# AFTER the INSERT INTO trades, get the trade_id:
trade_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

# Log full snapshot to trade_journal.db
_log_trade_snapshot(trade_id, tf, direction, now, price,
    p_long=p_long, p_flat=p_flat, p_short=p_short,
    lstm_prob=lstm_p_raw,  # capture BEFORE blending
    meta_prob=meta_prob_raw,
    blended_confidence=confidence,
    regime=regime, regime_idx=regime_idx,
    hmm_feats=hmm_feats,
    kelly_f=kelly_f, base_risk=base_risk, risk=risk,
    leverage=lev, dd_scale=dd_scale,
    portfolio_dd=portfolio_dd, tf_pool_dd=pool.get('dd', 0),
    stop_atr=stop_mult, rr=rr, max_hold=max_hold,
    atr=atr, feat_dict=feat_dict,
    confluence_parent=parent_tf, confluence_parent_dir=parent_dir,
    confluence_scale=conf_scale)
```

**During HOLD (new monitoring thread/function):**

```python
def _monitor_open_trades():
    """Called every bar close. Record price path for all open trades."""
    conn_j = sqlite3.connect(TRADE_JOURNAL_DB)
    conn_t = sqlite3.connect(TRADES_DB)

    open_trades = conn_t.execute(
        "SELECT id, tf, direction, entry_price, entry_time FROM trades WHERE status='open'"
    ).fetchall()

    for tid, tf, direction, entry_price, entry_time in open_trades:
        entry_dt = datetime.fromisoformat(entry_time)
        bar_num = _count_bars_since(entry_dt, tf)
        d = 1 if direction == 'LONG' else -1
        current_price = _get_current_price()

        # Get current OHLCV bar
        ohlcv = live_dal.get_ohlcv_window(tf, 2)
        if ohlcv is None or len(ohlcv) == 0:
            continue
        bar = ohlcv.iloc[-1]

        unrealized_pnl = (current_price - entry_price) / entry_price * d
        # Compute running MAE/MFE from all recorded bars
        prev_mae = conn_j.execute(
            "SELECT MIN(max_adverse_excursion) FROM price_path WHERE trade_id=?", (tid,)
        ).fetchone()[0] or 0
        prev_mfe = conn_j.execute(
            "SELECT MAX(max_favorable_excursion) FROM price_path WHERE trade_id=?", (tid,)
        ).fetchone()[0] or 0

        mae = min(prev_mae, unrealized_pnl)
        mfe = max(prev_mfe, unrealized_pnl)

        conn_j.execute("""INSERT OR REPLACE INTO price_path
            (trade_id, bar_num, timestamp, open, high, low, close, volume,
             unrealized_pnl_pct, max_adverse_excursion, max_favorable_excursion,
             distance_to_stop_pct, distance_to_tp_pct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (tid, bar_num, datetime.now(timezone.utc).isoformat(),
             float(bar.get('open', 0)), float(bar.get('high', 0)),
             float(bar.get('low', 0)), float(bar.get('close', 0)),
             float(bar.get('volume', 0)),
             unrealized_pnl, mae, mfe,
             _dist_to_stop(current_price, entry_price, sl, direction),
             _dist_to_tp(current_price, entry_price, tp, direction)))

    conn_j.commit()
    conn_j.close()
    conn_t.close()
```

**At EXIT (line ~765 in v1, where UPDATE trades SET status='closed'):**

```python
# AFTER closing the trade, compute full outcome analysis
_log_trade_outcome(tid, tf, direction, entry_price, price, pnl, pnl_pct,
    reason, bars_held, feat_dict_at_exit, feat_dict_at_entry,
    p_long_entry, p_short_entry, confidence_entry)
```

**At REJECTION points (confluence block, meta-gate, DD halt, below threshold):**

```python
# Each rejection point should log:
_log_rejected_trade(tf, direction, now, price, confidence,
    reason='meta_gate',  # or 'confluence_block', 'dd_halt', 'below_threshold'
    meta_prob=meta_prob, confluence_parent=parent_tf,
    features_json=json.dumps(key_feats))
```

---

## B. CONTINUOUS MODEL IMPROVEMENT

### B1. Retraining Strategy: Periodic Batch (NOT Online Learning)

Online learning (updating model weights after each trade) is **wrong** for this system because:
1. XGBoost is a batch learner -- no native online update mechanism
2. With ~1-5 trades per day across all TFs, the sample rate is too low for meaningful online updates
3. Esoteric signals (astrology, gematria) have seasonal/cyclical patterns that require full-history context
4. The matrix philosophy ("more signals = stronger") means the model needs to see ALL historical interactions

**Retraining approach: Scheduled batch retraining with expanding windows.**

### B2. Retrain Trigger System

New file: `retrain_scheduler.py`

```python
RETRAIN_TRIGGERS = {
    # Trigger 1: Trade count threshold
    'min_new_trades': 50,          # At least 50 new trades since last retrain

    # Trigger 2: Performance degradation
    'sharpe_floor': 0.5,           # Rolling 30-trade Sharpe drops below 0.5
    'win_rate_floor': 0.45,        # Rolling 30-trade win rate drops below 45%
    'max_consecutive_losses': 7,   # 7 losses in a row on any single TF

    # Trigger 3: Calendar-based
    'max_days_since_retrain': 14,  # Force retrain every 2 weeks regardless

    # Trigger 4: Feature drift detected
    'feature_drift_threshold': 0.3,  # >30% change in top-20 feature importance

    # Trigger 5: Regime shift
    'regime_change_detected': True,  # HMM state transition detected
}
```

**Retrain flow:**
1. `retrain_scheduler.py` runs locally as a cron job every 6 hours
2. Queries `trade_journal.db` for performance metrics
3. If ANY trigger fires, creates a retrain manifest:
   ```json
   {
     "trigger": "sharpe_floor",
     "triggered_at": "2026-03-21T12:00:00Z",
     "metrics": {"rolling_sharpe": 0.32, "rolling_win_rate": 0.41},
     "tfs_to_retrain": ["1h", "4h"],
     "include_new_trades": true,
     "new_trade_count": 87,
     "data_window": "2021-01-01 to 2026-03-21"
   }
   ```
4. Uploads latest `features_*.db` + manifest to vast.ai
5. Triggers cloud retraining via SSH command
6. Downloads new model files when complete
7. Hot-swaps models in live_trader.py (atomic file rename)

### B3. Expanding Window with Decay Weighting

When retraining, use the FULL historical feature DBs plus new live data, but weight recent samples more heavily:

```python
def compute_sample_weights(timestamps, half_life_days=90):
    """Exponential decay weighting. Recent trades matter more."""
    now = pd.Timestamp.now(tz='UTC')
    ages_days = (now - pd.to_datetime(timestamps, utc=True)).dt.total_seconds() / 86400
    weights = np.exp(-np.log(2) * ages_days / half_life_days)
    # Floor at 0.1 so old data still contributes (never forget old patterns)
    return np.maximum(weights, 0.1)
```

This is passed to XGBoost's `DMatrix(weight=...)` parameter. Old patterns (full/new moon effects across 5 years) are preserved but recent regime shifts get more weight.

### B4. Feature Importance Drift Detection

New file component in `self_learner.py`:

```python
def detect_feature_drift(current_model, previous_importance_json, threshold=0.3):
    """
    Compare current model's feature importance ranking to the previous snapshot.
    Returns drift_score (0-1) and list of features that changed significantly.
    """
    current_imp = dict(zip(
        current_model.feature_names,
        current_model.get_score(importance_type='total_gain').values()
    ))
    previous_imp = json.load(open(previous_importance_json))

    # Normalize both to rank-based comparison
    curr_rank = {f: r for r, f in enumerate(sorted(current_imp, key=current_imp.get, reverse=True))}
    prev_rank = {f: r for r, f in enumerate(sorted(previous_imp, key=previous_imp.get, reverse=True))}

    # Focus on top 50 features
    top_features = sorted(current_imp, key=current_imp.get, reverse=True)[:50]

    rank_changes = []
    for f in top_features:
        curr_r = curr_rank.get(f, len(curr_rank))
        prev_r = prev_rank.get(f, len(prev_rank))
        rank_changes.append(abs(curr_r - prev_r))

    drift_score = np.mean(rank_changes) / len(curr_rank)  # normalized

    # Categorize which signal TYPES are drifting
    esoteric_drift = [f for f in top_features if any(k in f for k in
        ['gematria', 'numerology', 'astro', 'moon', 'mercury', 'kp_'])]
    ta_drift = [f for f in top_features if any(k in f for k in
        ['rsi', 'macd', 'sma', 'ema', 'atr', 'bollinger', 'volume'])]

    return {
        'drift_score': drift_score,
        'triggered': drift_score > threshold,
        'esoteric_features_shifted': esoteric_drift,
        'ta_features_shifted': ta_drift,
        'top_10_current': top_features[:10],
    }
```

---

## C. SIGNAL QUALITY TRACKING

### C1. Per-Signal Rolling Accuracy

New table in `trade_journal.db`:

```sql
CREATE TABLE signal_quality (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_name TEXT NOT NULL,
    signal_category TEXT NOT NULL,       -- 'gematria', 'astro', 'numerology', 'ta', 'space_weather'
    window_end TEXT NOT NULL,            -- end of rolling window
    window_trades INTEGER,              -- number of trades in window

    -- Accuracy metrics
    accuracy REAL,                       -- % of trades where this signal aligned with outcome
    contribution_when_active REAL,       -- avg P&L when this signal was active
    contribution_when_inactive REAL,     -- avg P&L when this signal was inactive
    lift REAL,                           -- active_pnl - inactive_pnl (positive = signal helps)

    -- Frequency
    activation_rate REAL,                -- % of trades where signal was active
    correlation_with_outcome REAL,       -- point-biserial correlation

    UNIQUE(signal_name, window_end)
);
CREATE INDEX idx_sq_signal ON signal_quality(signal_name);
CREATE INDEX idx_sq_category ON signal_quality(signal_category);
```

### C2. Signal Quality Computation

```python
def compute_signal_quality(window_size=50):
    """
    For each signal feature, compute rolling accuracy over last N trades.

    A signal is "active" if its value is non-zero/non-NaN.
    A signal is "correct" if:
      - Signal > 0 (bullish) and trade was profitable, OR
      - Signal < 0 (bearish) and SHORT trade was profitable
    """
    conn = sqlite3.connect(TRADE_JOURNAL_DB)
    trades = pd.read_sql("""
        SELECT s.features_json, o.pnl, o.direction, o.prediction_correct
        FROM trade_snapshots s
        JOIN trade_outcomes o ON s.trade_id = o.trade_id
        ORDER BY s.entry_time DESC
        LIMIT ?
    """, conn, params=(window_size,))

    if len(trades) < 10:
        return {}

    # Parse features from JSON
    feature_dicts = [json.loads(f) for f in trades['features_json']]
    feature_df = pd.DataFrame(feature_dicts)

    results = {}
    for col in feature_df.columns:
        vals = feature_df[col].values
        pnls = trades['pnl'].values
        active = ~np.isnan(vals) & (vals != 0)

        if active.sum() < 3:
            continue

        active_pnl = pnls[active].mean() if active.sum() > 0 else 0
        inactive_pnl = pnls[~active].mean() if (~active).sum() > 0 else 0

        results[col] = {
            'accuracy': (pnls[active] > 0).mean() if active.sum() > 0 else 0.5,
            'contribution_active': active_pnl,
            'contribution_inactive': inactive_pnl,
            'lift': active_pnl - inactive_pnl,
            'activation_rate': active.mean(),
            'n_active': int(active.sum()),
        }

    # Categorize
    CATEGORIES = {
        'gematria': ['gem_', 'gematria_', 'ordinal_', 'cipher_'],
        'numerology': ['num_', 'angel_', 'master_', 'pythagorean_'],
        'astrology': ['astro_', 'moon_', 'mercury_', 'planet_', 'zodiac_'],
        'space_weather': ['kp_', 'solar_', 'schumann_', 'geomag_'],
        'ta': ['rsi', 'macd', 'sma_', 'ema_', 'atr_', 'bollinger_', 'volume_'],
    }

    for sig_name, metrics in results.items():
        cat = 'other'
        for category, prefixes in CATEGORIES.items():
            if any(sig_name.startswith(p) or sig_name.startswith(p.upper()) for p in prefixes):
                cat = category
                break
        metrics['category'] = cat

    return results
```

### C3. Dynamic Signal Weighting

Instead of modifying feature values (which would break the XGBoost model), signal quality informs:

1. **Retrain feature selection**: If a signal category's average lift is consistently negative over 100+ trades, flag it for investigation (but NEVER remove -- the matrix philosophy says more signals = stronger)
2. **Confidence adjustment**: Multiply blended confidence by a signal-quality factor:
   ```python
   def adjust_confidence_by_signal_quality(confidence, active_signals, signal_quality_db):
       """Adjust confidence based on how well the active signals have been performing."""
       if not active_signals:
           return confidence

       lifts = []
       for sig in active_signals:
           sq = signal_quality_db.get(sig)
           if sq and sq['n_active'] >= 10:
               lifts.append(sq['lift'])

       if not lifts:
           return confidence

       avg_lift = np.mean(lifts)
       # Scale: if average lift is strongly positive, boost confidence slightly
       # If negative, dampen slightly (never below 0.8x original)
       adjustment = 1.0 + np.clip(avg_lift * 10, -0.2, 0.2)
       return confidence * adjustment
   ```
3. **Dashboard reporting**: Show which signals are hot vs cold in the current regime

---

## D. META-LEARNING

### D1. Meta-Labeling Audit

```sql
CREATE TABLE meta_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    tf TEXT,
    window_trades INTEGER,

    -- Gate performance
    total_signals INTEGER,               -- total XGBoost signals (above conf_thresh)
    gate_allowed INTEGER,                -- signals that passed meta-gate
    gate_blocked INTEGER,                -- signals that meta-gate rejected
    allowed_win_rate REAL,               -- win rate of allowed trades
    blocked_hypothetical_win_rate REAL,  -- win rate of blocked trades (what if we took them?)

    -- Gate calibration
    gate_accuracy REAL,                  -- % of correct gate decisions
    false_positive_rate REAL,            -- allowed bad trades / total allowed
    false_negative_rate REAL,            -- blocked good trades / total blocked
    optimal_threshold REAL,              -- hindsight-optimal threshold
    current_threshold REAL,              -- threshold currently in use

    -- Value added
    pnl_with_gate REAL,                  -- actual PnL (with gate)
    pnl_without_gate REAL,               -- hypothetical PnL (no gate)
    gate_value_added REAL                -- difference
);
```

```python
def audit_meta_gate(window_size=100):
    """
    Compare trades that passed the meta-gate vs trades that were blocked.
    If blocked trades would have been profitable, the gate is too aggressive.
    If allowed trades are mostly losers, the gate is too permissive.
    """
    conn = sqlite3.connect(TRADE_JOURNAL_DB)

    # Allowed trades (actual outcomes)
    allowed = pd.read_sql("""
        SELECT o.pnl, o.prediction_correct, s.meta_prob, s.blended_confidence
        FROM trade_outcomes o
        JOIN trade_snapshots s ON o.trade_id = s.trade_id
        WHERE s.meta_prob IS NOT NULL
        ORDER BY o.created_at DESC LIMIT ?
    """, conn, params=(window_size,))

    # Blocked trades (hypothetical outcomes)
    blocked = pd.read_sql("""
        SELECT hypothetical_pnl_pct, was_correct_rejection, meta_prob, confidence
        FROM rejected_trades
        WHERE rejection_reason = 'meta_gate'
        ORDER BY created_at DESC LIMIT ?
    """, conn, params=(window_size,))

    if len(allowed) < 10:
        return None

    gate_accuracy = (
        (allowed['pnl'] > 0).sum() +  # correct allows
        blocked['was_correct_rejection'].sum()  # correct blocks
    ) / (len(allowed) + len(blocked)) if (len(allowed) + len(blocked)) > 0 else 0

    # Optimal threshold search
    all_probs = np.concatenate([allowed['meta_prob'].values,
                                 blocked['meta_prob'].values])
    all_correct = np.concatenate([
        (allowed['pnl'] > 0).values.astype(int),
        (1 - blocked['was_correct_rejection']).values.astype(int)
    ])

    best_thresh, best_acc = 0.5, 0
    for thresh in np.arange(0.3, 0.8, 0.01):
        pred = (all_probs >= thresh).astype(int)
        acc = (pred == all_correct).mean()
        if acc > best_acc:
            best_acc = acc
            best_thresh = thresh

    return {
        'gate_accuracy': gate_accuracy,
        'allowed_win_rate': (allowed['pnl'] > 0).mean(),
        'blocked_would_win_rate': (1 - blocked['was_correct_rejection']).mean() if len(blocked) > 0 else 0,
        'optimal_threshold': best_thresh,
        'value_added': allowed['pnl'].sum() - (allowed['pnl'].sum() + blocked.get('hypothetical_pnl_pct', pd.Series([0])).sum()),
    }
```

### D2. Kelly Sizing Audit

```sql
CREATE TABLE kelly_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    tf TEXT,
    window_trades INTEGER,

    -- Kelly calibration
    estimated_win_prob REAL,             -- P(win) used for Kelly at trade time
    actual_win_rate REAL,                -- real win rate in this window
    estimated_rr REAL,                   -- avg R:R used for Kelly
    actual_rr REAL,                      -- actual avg_win / avg_loss
    kelly_optimal_fraction REAL,         -- what Kelly SHOULD be
    kelly_used_fraction REAL,            -- what we actually used
    over_under_betting TEXT,             -- 'over', 'under', or 'calibrated'

    -- Sizing impact
    avg_risk_pct REAL,
    optimal_risk_pct REAL,
    pnl_actual REAL,
    pnl_optimal_kelly REAL              -- hypothetical with optimal sizing
);
```

```python
def audit_kelly_sizing(window_size=50):
    """Check if Kelly fraction is calibrated to actual win rate and R:R."""
    conn = sqlite3.connect(TRADE_JOURNAL_DB)
    trades = pd.read_sql("""
        SELECT s.kelly_fraction, s.base_risk_pct, s.final_risk_pct,
               s.blended_confidence, s.rr_ratio, s.leverage_used,
               o.pnl, o.pnl_pct, o.exit_reason
        FROM trade_snapshots s
        JOIN trade_outcomes o ON s.trade_id = o.trade_id
        ORDER BY s.entry_time DESC LIMIT ?
    """, conn, params=(window_size,))

    if len(trades) < 20:
        return None

    actual_win_rate = (trades['pnl'] > 0).mean()
    wins = trades[trades['pnl'] > 0]['pnl_pct']
    losses = trades[trades['pnl'] <= 0]['pnl_pct']
    actual_rr = abs(wins.mean() / losses.mean()) if len(losses) > 0 and losses.mean() != 0 else 1.0

    # Optimal full Kelly
    optimal_kelly = (actual_win_rate * actual_rr - (1 - actual_win_rate)) / max(actual_rr, 0.01)
    optimal_kelly = max(optimal_kelly, 0)

    avg_confidence = trades['blended_confidence'].mean()
    avg_rr_used = trades['rr_ratio'].mean()
    estimated_kelly = (avg_confidence * avg_rr_used - (1 - avg_confidence)) / max(avg_rr_used, 0.01)

    status = 'calibrated'
    if optimal_kelly > 0 and trades['kelly_fraction'].mean() > optimal_kelly * 1.5:
        status = 'over-betting'
    elif optimal_kelly > 0 and trades['kelly_fraction'].mean() < optimal_kelly * 0.5:
        status = 'under-betting'

    return {
        'actual_win_rate': actual_win_rate,
        'actual_rr': actual_rr,
        'optimal_kelly': optimal_kelly,
        'used_kelly': trades['kelly_fraction'].mean(),
        'status': status,
        'recommended_adjustment': optimal_kelly / max(trades['kelly_fraction'].mean(), 0.001),
    }
```

### D3. Regime Multiplier Calibration

```python
def audit_regime_multipliers(window_size=200):
    """
    Check if regime multipliers match actual performance per regime.
    If bull regime trades are losing, the bull multiplier is too aggressive.
    """
    conn = sqlite3.connect(TRADE_JOURNAL_DB)
    trades = pd.read_sql("""
        SELECT s.regime, s.regime_idx, s.leverage_used, s.final_risk_pct,
               o.pnl_pct, o.exit_reason, o.bars_held
        FROM trade_snapshots s
        JOIN trade_outcomes o ON s.trade_id = o.trade_id
        ORDER BY s.entry_time DESC LIMIT ?
    """, conn, params=(window_size,))

    results = {}
    for regime in trades['regime'].unique():
        r_trades = trades[trades['regime'] == regime]
        if len(r_trades) < 10:
            continue

        results[regime] = {
            'n_trades': len(r_trades),
            'win_rate': (r_trades['pnl_pct'] > 0).mean(),
            'avg_pnl': r_trades['pnl_pct'].mean(),
            'avg_leverage': r_trades['leverage_used'].mean(),
            'sharpe': r_trades['pnl_pct'].mean() / max(r_trades['pnl_pct'].std(), 0.001),
            'avg_bars_held': r_trades['bars_held'].mean(),
            'sl_rate': (r_trades['exit_reason'] == 'SL').mean(),
        }

        # Recommend multiplier adjustments
        if results[regime]['sharpe'] < 0:
            results[regime]['recommendation'] = 'REDUCE leverage and risk (negative Sharpe)'
        elif results[regime]['sl_rate'] > 0.5:
            results[regime]['recommendation'] = 'WIDEN stops or REDUCE leverage (>50% SL exits)'
        elif results[regime]['win_rate'] > 0.6 and results[regime]['avg_pnl'] > 0:
            results[regime]['recommendation'] = 'Current multipliers working well'
        else:
            results[regime]['recommendation'] = 'Monitor closely'

    return results
```

---

## E. IMPLEMENTATION PLAN

### E1. New Files to Create

| File | Purpose | Runs Where |
|------|---------|------------|
| `trade_journal.py` | Journal DB init, snapshot/path/outcome logging functions | Local (imported by live_trader.py) |
| `self_learner.py` | All analysis: signal quality, drift detection, meta/Kelly/regime audits | Local (cron or manual) |
| `retrain_scheduler.py` | Monitors performance, triggers cloud retraining when needed | Local (cron every 6h) |
| `hypothetical_tracker.py` | After each bar, simulates what rejected trades would have done | Local (called by self_learner) |

### E2. Modifications to Existing Files

| File | Changes |
|------|---------|
| `live_trader.py` | Add imports for `trade_journal`. At entry: call `log_trade_snapshot()`. At exit: call `log_trade_outcome()`. At every bar: call `monitor_price_paths()`. At every rejection point: call `log_rejected_trade()`. Capture individual probabilities (xgb, lstm, meta) BEFORE blending into `confidence`. |
| `v2/live_trader.py` | Same changes as v1, plus capture sparse cross contribution metrics. |
| `ml_multi_tf.py` | After training: save feature importance snapshot to `feature_importance_{tf}.json`. Save OOS prediction quality metrics. |
| `exhaustive_optimizer.py` | After optimization: save param sensitivity analysis (how much does Sharpe change per param?). |
| `meta_labeling.py` | Save threshold and calibration metrics to journal DB after training. |

### E3. New Database

**`trade_journal.db`** (separate from trades.db):
- `trade_snapshots` — full state at entry
- `price_path` — bar-by-bar during hold
- `trade_outcomes` — post-close analysis
- `rejected_trades` — what we didn't take
- `signal_quality` — rolling per-signal accuracy
- `meta_audit` — meta-gate calibration history
- `kelly_audit` — Kelly sizing calibration history
- `regime_audit` — regime multiplier performance history
- `feature_drift` — feature importance drift history
- `retrain_log` — when/why/results of each retraining

### E4. Integration with Cloud Training Pipeline

```
LOCAL (13900K + RTX 3090)                    CLOUD (vast.ai)
========================                     ==================
live_trader.py
  -> trade_journal.db

self_learner.py (every 6h)
  -> Computes all audits
  -> Checks retrain triggers

retrain_scheduler.py
  -> IF trigger fires:
     1. Export new trade data
     2. rsync features_*.db to cloud         ml_multi_tf.py (retrain)
     3. SSH: launch training                 -> XGBoost on GPU
     4. Poll for completion       <-------   -> Save model_*.json
     5. Download new models                  -> Save features_*.json
     6. Atomic swap in live_trader           -> Save importance_*.json
     7. Log retrain results

self_learner.py (post-retrain)
  -> Compare old vs new model
  -> Feature drift analysis
  -> Update signal_quality
```

**Key design principle**: Self-learning analysis runs LOCALLY (CPU-bound analysis, small data). Only model TRAINING runs on cloud GPU. This avoids cloud costs for continuous monitoring.

### E5. Execution Priority Order

**Phase 1 (Immediate -- enables all future learning):**
1. Create `trade_journal.py` with DB init and logging functions
2. Modify `live_trader.py` to capture snapshots at entry/exit/rejection
3. Add `_monitor_open_trades()` to track MAE/MFE during holds

**Phase 2 (After 50+ trades logged):**
4. Create `self_learner.py` with signal quality tracker
5. Create `hypothetical_tracker.py` for rejected trade simulation
6. Add feature importance drift detection

**Phase 3 (After 200+ trades):**
7. Create `retrain_scheduler.py` with trigger system
8. Meta-labeling audit
9. Kelly sizing audit
10. Regime multiplier calibration

**Phase 4 (Ongoing):**
11. Dashboard integration (display signal quality, drift alerts)
12. Automated retrain pipeline with vast.ai
13. A/B testing framework (run old vs new model in shadow mode)

### E6. Critical Design Constraints

1. **NEVER remove features based on self-learning**. The matrix philosophy is sacred. If gematria signals have negative lift for 100 trades, LOG IT but do not remove them. They may be cyclical.
2. **All analysis is read-only against the training pipeline**. Self-learning informs WHEN to retrain and WHAT to monitor, but never modifies feature computation or model architecture autonomously.
3. **Rejected trade tracking is essential**. Without it, you have survivorship bias in your analysis. You must know what would have happened with trades you didn't take.
4. **Decay weighting during retrain preserves old patterns**. The floor weight of 0.1 ensures full-moon patterns from 2021 still contribute. Recent data just gets MORE weight, not exclusive weight.
5. **Hot-swap models atomically**. Write new model to `model_{tf}_new.json`, then `os.rename()` to `model_{tf}.json`. This prevents the live trader from reading a half-written model.

---

## SUMMARY OF GAPS AND FIXES

| Gap | Current State | Fix |
|-----|--------------|-----|
| No exit-time features | Only entry features stored | Capture `feat_dict` at exit, store in `trade_outcomes.exit_features_json` |
| No MAE/MFE | Not tracked | Bar-by-bar monitoring via `price_path` table |
| No individual model probs | Blended into single `confidence` | Capture xgb/lstm/meta probs separately BEFORE blending |
| No rejected trade tracking | Completely missing | New `rejected_trades` table at every rejection point |
| No signal quality metrics | No per-signal accuracy | Rolling window analysis in `signal_quality` table |
| No feature drift detection | Feature importance never compared | Post-retrain comparison of importance rankings |
| No retrain triggers | Manual decision | Automated triggers based on Sharpe, win rate, trade count, drift |
| No meta-gate audit | No way to know if gate is helping | Compare allowed vs blocked trade outcomes |
| No Kelly calibration | Static 25% Kelly fraction | Audit actual vs optimal Kelly based on realized win rate + R:R |
| No regime audit | Static multiplier tables | Per-regime performance tracking with adjustment recommendations |
| No optimal exit analysis | Unknown if exits are well-timed | Compare actual exit to optimal bar (hindsight) via `timing_efficiency` |
| No post-exit tracking | Don't know if we exited too early | Track price 1/5/10 bars after exit via `continued_in_direction` |
