#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
portfolio_aggregator.py
========================
Multi-TF Portfolio Risk Aggregator.
Manages 5 concurrent execution models (1D context, 4H, 1H, 15m, 5m)
with independent risk budgets, correlation adjustment, and drawdown triggers.

Can be imported by paper_trader.py or run standalone for backtesting.
"""

import sys, io, time, json, os, warnings
if os.name == 'nt':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import lightgbm as lgb
import sqlite3
from datetime import datetime, timedelta
from collections import defaultdict

DB_DIR = os.environ.get("SAVAGE22_V1_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================
# MODEL CONFIGS (from NSGA-II results)
# ============================================================
# Load saved configs if available
def load_configs():
    config_path = f'{DB_DIR}/ml_multi_tf_configs.json'
    if os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    return {}

# Default risk parameters per TF
DEFAULT_RISK = {
    '4h': {'max_risk_pct': 0.75, 'max_concurrent': 5, 'max_leverage': 25, 'priority_weight': 1.3},
    '1h': {'max_risk_pct': 0.25, 'max_concurrent': 3, 'max_leverage': 50, 'priority_weight': 1.1},
    '15m': {'max_risk_pct': 0.15, 'max_concurrent': 3, 'max_leverage': 60, 'priority_weight': 1.0},
    '5m': {'max_risk_pct': 0.10, 'max_concurrent': 2, 'max_leverage': 75, 'priority_weight': 0.8},
}

# ============================================================
# SIGNAL CLASS
# ============================================================
class Signal:
    def __init__(self, tf, direction, confidence, timestamp, price, atr,
                 leverage=1, risk_pct=1.0, stop_atr=1.0, rr=2.0, max_hold=10):
        self.tf = tf
        self.direction = direction  # 1=LONG, -1=SHORT
        self.confidence = confidence
        self.timestamp = timestamp
        self.price = price
        self.atr = atr
        self.leverage = leverage
        self.risk_pct = risk_pct / 100  # convert to decimal
        self.stop_atr = stop_atr
        self.rr = rr
        self.max_hold = max_hold
        self.id = f"{tf}_{timestamp}_{direction}"

class Position:
    def __init__(self, signal, entry_price, balance):
        self.signal = signal
        self.tf = signal.tf
        self.direction = signal.direction
        self.entry_price = entry_price
        self.leverage = signal.leverage
        self.risk_amount = balance * signal.risk_pct
        self.stop_price = entry_price - signal.direction * signal.stop_atr * signal.atr
        self.tp_price = entry_price + signal.direction * signal.stop_atr * signal.atr * signal.rr
        self.bars_held = 0
        self.max_hold = signal.max_hold
        self.entry_time = signal.timestamp
        self.pnl = 0.0
        self.closed = False
        self.close_reason = ""

    def update(self, price, fee_rate=0.0012):
        """Check exits. Returns PnL if closed, None if still open."""
        self.bars_held += 1

        sl_hit = (self.direction == 1 and price <= self.stop_price) or \
                 (self.direction == -1 and price >= self.stop_price)
        tp_hit = (self.direction == 1 and price >= self.tp_price) or \
                 (self.direction == -1 and price <= self.tp_price)
        time_exit = self.bars_held >= self.max_hold

        if sl_hit or tp_hit or time_exit:
            price_change = (price - self.entry_price) / self.entry_price * self.direction
            gross = price_change * self.leverage
            fee = fee_rate * self.leverage
            net = gross - fee
            self.pnl = self.risk_amount * net
            self.closed = True
            self.close_reason = "SL" if sl_hit else ("TP" if tp_hit else "TIME")
            return self.pnl

        return None

# ============================================================
# PORTFOLIO AGGREGATOR
# ============================================================
class PortfolioAggregator:
    def __init__(self, initial_balance=10000, max_total_heat=0.03,
                 max_portfolio_dd=0.15, fee_rate=0.0012):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.peak_balance = initial_balance
        self.max_total_heat = max_total_heat  # 3% max total exposure
        self.max_portfolio_dd = max_portfolio_dd
        self.fee_rate = fee_rate

        self.positions = []  # active positions
        self.closed_trades = []
        self.model_equity = defaultdict(lambda: initial_balance)
        self.model_peak = defaultdict(lambda: initial_balance)
        self.model_suspended = defaultdict(bool)

        self.risk_params = DEFAULT_RISK.copy()
        self.signal_queue = []
        self.trade_log = []

    @property
    def total_exposure(self):
        """Current total risk as fraction of equity."""
        return sum(p.risk_amount for p in self.positions if not p.closed) / max(self.balance, 1)

    @property
    def net_direction(self):
        """Net directional exposure: +1 all long, -1 all short, 0 mixed."""
        if not self.positions:
            return 0
        dirs = [p.direction * p.risk_amount for p in self.positions if not p.closed]
        return sum(dirs) / max(sum(abs(d) for d in dirs), 1)

    @property
    def portfolio_dd(self):
        """Current drawdown from peak."""
        return (self.peak_balance - self.balance) / max(self.peak_balance, 1)

    def model_dd(self, tf):
        """Drawdown for a specific model."""
        return (self.model_peak[tf] - self.model_equity[tf]) / max(self.model_peak[tf], 1)

    def can_open_position(self, signal):
        """Check if a new position is allowed by risk rules."""
        tf = signal.tf
        params = self.risk_params.get(tf, {})

        # Model suspended?
        if self.model_suspended[tf]:
            return False, "Model suspended"

        # Max concurrent positions per TF
        active = sum(1 for p in self.positions if not p.closed and p.tf == tf)
        if active >= params.get('max_concurrent', 3):
            return False, f"Max concurrent ({active}/{params.get('max_concurrent', 3)})"

        # Total portfolio heat check
        new_risk = self.balance * signal.risk_pct / 100
        if self.total_exposure + new_risk / self.balance > self.max_total_heat:
            return False, f"Total heat exceeded ({self.total_exposure:.1%} + {new_risk/self.balance:.1%})"

        # Portfolio DD trigger
        if self.portfolio_dd > self.max_portfolio_dd:
            return False, f"Portfolio DD > {self.max_portfolio_dd:.0%}"

        # Model DD triggers
        model_dd = self.model_dd(tf)
        if model_dd > 0.10:
            return False, f"Model {tf} DD > 10% — suspended"
        if model_dd > 0.05:
            # Reduce risk by 50%
            signal.risk_pct *= 0.5

        # Leverage cap
        if signal.leverage > params.get('max_leverage', 75):
            signal.leverage = params['max_leverage']

        return True, "OK"

    def correlation_adjust(self, signal):
        """Reduce size if multiple models agree on same direction."""
        same_dir = sum(1 for p in self.positions
                      if not p.closed and p.direction == signal.direction)
        if same_dir >= 3:
            signal.risk_pct *= 0.5  # 50% reduction if 3+ models agree
        elif same_dir >= 2:
            signal.risk_pct *= 0.75  # 25% reduction if 2 models agree
        return signal

    def process_signal(self, signal):
        """Process a new signal from any TF model."""
        allowed, reason = self.can_open_position(signal)
        if not allowed:
            return None, reason

        signal = self.correlation_adjust(signal)

        pos = Position(signal, signal.price, self.balance)
        self.positions.append(pos)
        return pos, "Opened"

    def update_positions(self, current_prices):
        """Update all positions with current prices. Returns list of closed trades."""
        closed = []
        for pos in self.positions:
            if pos.closed:
                continue
            price = current_prices.get(pos.tf, current_prices.get('default', 0))
            pnl = pos.update(price, self.fee_rate)
            if pnl is not None:
                self.balance += pnl
                self.model_equity[pos.tf] += pnl

                if self.balance > self.peak_balance:
                    self.peak_balance = self.balance
                if self.model_equity[pos.tf] > self.model_peak[pos.tf]:
                    self.model_peak[pos.tf] = self.model_equity[pos.tf]

                # Check model suspension
                if self.model_dd(pos.tf) > 0.10:
                    self.model_suspended[pos.tf] = True

                closed.append({
                    'tf': pos.tf,
                    'direction': 'LONG' if pos.direction == 1 else 'SHORT',
                    'entry': pos.entry_price,
                    'exit': price,
                    'pnl': pnl,
                    'bars': pos.bars_held,
                    'reason': pos.close_reason,
                    'balance': self.balance,
                })
                self.closed_trades.append(closed[-1])

        # Clean up closed positions
        self.positions = [p for p in self.positions if not p.closed]
        return closed

    def get_status(self):
        """Get current portfolio status."""
        active = [p for p in self.positions if not p.closed]
        return {
            'balance': self.balance,
            'initial': self.initial_balance,
            'roi_pct': (self.balance - self.initial_balance) / self.initial_balance * 100,
            'peak': self.peak_balance,
            'dd_pct': self.portfolio_dd * 100,
            'total_exposure': self.total_exposure * 100,
            'net_direction': self.net_direction,
            'active_positions': len(active),
            'total_trades': len(self.closed_trades),
            'wins': sum(1 for t in self.closed_trades if t['pnl'] > 0),
            'losses': sum(1 for t in self.closed_trades if t['pnl'] <= 0),
            'per_model': {
                tf: {
                    'equity': self.model_equity[tf],
                    'dd_pct': self.model_dd(tf) * 100,
                    'suspended': self.model_suspended[tf],
                    'active': sum(1 for p in active if p.tf == tf),
                    'trades': sum(1 for t in self.closed_trades if t['tf'] == tf),
                } for tf in ['4h', '1h', '15m', '5m']
            },
        }

    def print_status(self):
        """Print formatted status."""
        s = self.get_status()
        print(f"\n{'='*60}")
        print(f"PORTFOLIO STATUS")
        print(f"{'='*60}")
        print(f"  Balance: ${s['balance']:,.2f} (ROI: {s['roi_pct']:+.1f}%)")
        print(f"  Peak: ${s['peak']:,.2f} | DD: {s['dd_pct']:.1f}%")
        print(f"  Exposure: {s['total_exposure']:.1f}% | Net: {s['net_direction']:+.2f}")
        print(f"  Active: {s['active_positions']} | Trades: {s['total_trades']} "
              f"({s['wins']}W/{s['losses']}L = {s['wins']/max(s['total_trades'],1)*100:.0f}%)")
        print(f"\n  Per-Model:")
        for tf, m in s['per_model'].items():
            status = "SUSPENDED" if m['suspended'] else f"Active({m['active']})"
            print(f"    {tf:>4s}: ${m['equity']:>10,.2f} DD={m['dd_pct']:.1f}% "
                  f"Trades={m['trades']:>4d} [{status}]")

# ============================================================
# MODEL LOADER
# ============================================================
class MultiTFPredictor:
    """Loads all trained models and generates predictions."""

    def __init__(self):
        self.models = {}
        self.features = {}
        self.configs = load_configs()

        for tf in ['4h', '1h', '15m', '5m']:
            model_path = f'{DB_DIR}/model_{tf}.json'
            feat_path = f'{DB_DIR}/features_{tf}_pruned.json'
            if os.path.exists(model_path) and os.path.exists(feat_path):
                self.models[tf] = lgb.Booster(model_file=model_path)
                with open(feat_path) as f:
                    self.features[tf] = json.load(f)
                print(f"  Loaded {tf} model: {len(self.features[tf])} features")
            else:
                print(f"  {tf} model not found")

    def predict(self, tf, feature_dict):
        """Generate prediction for a TF given a dict of feature values."""
        if tf not in self.models:
            return None, 0.0

        feat_names = self.features[tf]
        values = [float(feature_dict.get(f, float('nan'))) for f in feat_names]
        X = np.array([values], dtype=np.float32)
        preds = self.models[tf].predict(X)
        prob = float(preds[0])

        return prob, prob

# ============================================================
# BACKTEST MODE
# ============================================================
def backtest_portfolio():
    """Run a full backtest of the multi-TF portfolio."""
    print("=" * 70)
    print("MULTI-TF PORTFOLIO BACKTEST")
    print("=" * 70)

    agg = PortfolioAggregator(initial_balance=10000)

    # Load all feature databases and run through chronologically
    configs = load_configs()

    for tf in ['4h', '1h', '15m', '5m']:
        tf_cfg = configs.get(tf, {})
        if not tf_cfg or tf_cfg.get('context_only'):
            continue

        balanced = tf_cfg.get('configs', {}).get('balanced', {})
        if not balanced:
            continue

        params = balanced.get('params', [10, 1.0, 1.0, 2.0, 5, 0, 0.6])
        lev, risk, stop, rr, hold, ptp, conf_thresh = params

        # Load model
        model_path = f'{DB_DIR}/model_{tf}.json'
        feat_path = f'{DB_DIR}/features_{tf}_pruned.json'
        if not os.path.exists(model_path):
            continue

        model = lgb.Booster(model_file=model_path)
        with open(feat_path) as f:
            feat_names = json.load(f)

        # Load features
        db_map = {'4h': ('features_4h.db', 'features_4h'),
                  '1h': ('features_1h.db', 'features_1h'),
                  '15m': ('features_15m.db', 'features_15m'),
                  '5m': ('features_5m.db', 'features_5m')}

        db_file, table = db_map[tf]
        conn = sqlite3.connect(f'{DB_DIR}/{db_file}')
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        conn.close()

        # Get last 20% as test
        n = len(df)
        test_start = int(n * 0.8)
        df_test = df.iloc[test_start:].copy()

        closes = pd.to_numeric(df_test['close'], errors='coerce').values
        atrs = pd.to_numeric(df_test.get('atr_14', pd.Series(closes * 0.01)), errors='coerce').values

        # Get predictions
        available_feats = [f for f in feat_names if f in df_test.columns]
        if len(available_feats) < len(feat_names) * 0.8:
            print(f"  {tf}: Too few features available ({len(available_feats)}/{len(feat_names)})")
            continue

        X = df_test[available_feats].values.astype(np.float32)
        X = np.where(np.isinf(X), np.nan, X)

        # Pad missing features with NaN (LightGBM treats as missing)
        full_X = np.full((len(X), len(feat_names)), np.nan, dtype=np.float32)
        for i, fn in enumerate(feat_names):
            if fn in available_feats:
                col_idx = available_feats.index(fn)
                full_X[:, i] = X[:, col_idx]

        probs = model.predict(full_X)

        trade_count = 0
        for i in range(len(probs)):
            price = closes[i] if closes[i] > 0 else 1
            atr = atrs[i] if not np.isnan(atrs[i]) and atrs[i] > 0 else price * 0.01

            # Check for signal
            if probs[i] > conf_thresh:
                sig = Signal(tf, 1, probs[i], i, price, atr,
                            leverage=lev, risk_pct=risk, stop_atr=stop, rr=rr, max_hold=int(hold))
                pos, reason = agg.process_signal(sig)
                if pos:
                    trade_count += 1
            elif probs[i] < (1 - conf_thresh):
                sig = Signal(tf, -1, 1 - probs[i], i, price, atr,
                            leverage=lev, risk_pct=risk, stop_atr=stop, rr=rr, max_hold=int(hold))
                pos, reason = agg.process_signal(sig)
                if pos:
                    trade_count += 1

            # Update all positions
            agg.update_positions({'default': price, tf: price})

        print(f"  {tf}: {trade_count} trades opened, conf>{conf_thresh:.2f}")

    agg.print_status()

    # Save results
    status = agg.get_status()
    with open(f'{DB_DIR}/portfolio_backtest_results.json', 'w') as f:
        json.dump(status, f, indent=2, default=str)
    print(f"\n  Saved: portfolio_backtest_results.json")

    return agg

if __name__ == '__main__':
    backtest_portfolio()
