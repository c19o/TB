#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lstm_precursor_model.py — Precursor Pattern Detection LSTM
============================================================
Trains on what happens BEFORE signals/events fire.
Instead of predicting direction, predicts "is event X about to happen?"

For each event type, it:
1. Finds all historical timestamps where the event fired
2. Extracts the N bars BEFORE each event as a labeled sequence (class=1)
3. Samples equal non-event windows as negatives (class=0)
4. Trains an LSTM to recognize the precursor pattern

Output: probability scores for each event type, fed as features to LightGBM.

NOTE: LSTM precursor operates on ~3000 base features only, NOT the 2.9M sparse cross features.
Cross features are sparse binary (0/1) — LSTM can't learn temporal patterns from sparse binary sequences.
This is an architectural gap, not a bug. The LightGBM model handles cross features via EFB.

Events detected:
  - Wyckoff SC (selling climax about to happen)
  - Wyckoff Spring (spring about to fire)
  - Wyckoff Upthrust (upthrust about to fire)
  - Liquidation cascade (liq spike imminent)
  - CVD divergence resolution (divergence about to resolve)
  - Gematria caution convergence
  - Volume breakout (volume spike imminent)
  - FVG formation (fair value gap about to form)

Usage:
    python lstm_precursor_model.py --train --tf 1h
    python lstm_precursor_model.py --train --all
"""
import os
import sys
import json
import time
import logging
import argparse
import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# ============================================================
# CONFIG
# ============================================================

# Lookback window before each event (bars to look at BEFORE the event)
PRECURSOR_WINDOW = {
    '1h': 24, '4h': 18, '1d': 10, '1w': 8,
}

# ONLY rare, high-value Wyckoff events — not FVGs/BOS which fire every few bars
EVENT_DEFINITIONS = [
    ('wyckoff_spring', lambda x: x == 1, 'Wyckoff Spring'),
    ('wyckoff_upthrust', lambda x: x == 1, 'Wyckoff Upthrust'),
    ('wyckoff_sos', lambda x: x == 1, 'Sign of Strength'),
    ('wyckoff_sow', lambda x: x == 1, 'Sign of Weakness'),
]

# Dynamic rare events (computed from price action, not from pre-computed columns)
DYNAMIC_EVENTS = [
    'volume_climax',     # volume > 3x avg AND spread > 1.5x ATR (SC-like)
    'big_move_up',       # return > 2.5x ATR (rare breakouts only)
    'big_move_down',     # return < -2.5x ATR (rare breakdowns only)
]

# Only 1H through 1W — events on 5m/15m are too frequent to be meaningful
DB_MAP = {
    '1h': ('features_1h.db', 'features_1h'),
    '4h': ('features_4h.db', 'features_4h'),
    '1d': ('features_1d.db', 'features_1d'),
    '1w': ('features_1w.db', 'features_1w'),
}


# ============================================================
# MODEL
# ============================================================

class PrecursorLSTM(nn.Module):
    """Multi-task LSTM: predicts probability of each event type."""

    def __init__(self, input_size, hidden_size=128, num_layers=2, n_events=1, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, n_events)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last = self.dropout(lstm_out[:, -1, :])
        return self.sigmoid(self.fc(last))


# ============================================================
# DATA PREP
# ============================================================

def find_events(df, event_col, condition_fn):
    """Find indices where an event fires. Vectorized where possible."""
    if event_col not in df.columns:
        return np.array([], dtype=int)
    vals = pd.to_numeric(df[event_col], errors='coerce').values.astype(np.float64)
    not_nan = ~np.isnan(vals)
    # Apply condition vectorized on non-NaN values
    mask = np.zeros(len(vals), dtype=bool)
    valid_indices = np.where(not_nan)[0]
    if len(valid_indices) > 0:
        valid_vals = vals[valid_indices]
        try:
            # Try vectorized apply first (works for simple lambdas like > threshold)
            cond_results = np.array([condition_fn(v) for v in valid_vals], dtype=bool)
            mask[valid_indices] = cond_results
        except (TypeError, ValueError):
            pass
    return np.where(mask)[0]


def compute_dynamic_events(df):
    """Compute dynamic event columns — RARE events only."""
    events = {}

    # Volume climax: volume > 3x avg AND bar spread > 1.5x ATR (selling/buying climax conditions)
    if all(c in df.columns for c in ['volume', 'volume_ratio', 'atr_14', 'high', 'low']):
        spread = df['high'] - df['low']
        events['volume_climax'] = (
            (df['volume_ratio'] > 3.0) &
            (spread > 1.5 * df['atr_14'])
        ).astype(int).values

    # Big moves: > 2.5x ATR (rarer threshold than 2x)
    if 'close' in df.columns and 'atr_14' in df.columns:
        ret = df['close'].pct_change()
        atr_pct = df['atr_14'] / df['close']
        events['big_move_up'] = (ret > 2.5 * atr_pct).astype(int).values
        events['big_move_down'] = (ret < -2.5 * atr_pct).astype(int).values

    return events


def prepare_precursor_data(tf_name):
    """Build training data: windows before events (positive) + random windows (negative).

    KEY DESIGN DECISIONS (prevents data leakage + class imbalance):
    1. EXCLUDE the event's own columns from input features — prevents the model
       from seeing the event forming in its own features (e.g., wyckoff_spring=1
       would leak into the input when predicting springs)
    2. EXCLUDE the window containing the event itself — only use bars BEFORE the
       event window, with a GAP of 1 bar to prevent any same-bar leakage
    3. Balance classes 1:1 — equal positive and negative samples per event type
    4. Cap positive samples per event to prevent dominant events from drowning others
    """
    db_file, table = DB_MAP[tf_name]
    db_dir = os.environ.get('SAVAGE22_DB_DIR', PROJECT_DIR)
    db_path = os.path.join(db_dir, db_file)
    window = PRECURSOR_WINDOW.get(tf_name, 20)

    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"SELECT * FROM {table} ORDER BY rowid", conn)
    conn.close()
    log.info(f"Loaded {len(df)} rows from {db_path}")

    # LEAKAGE FIX: Identify all event-related columns to EXCLUDE from inputs
    # These are the columns that directly indicate the events we're predicting
    event_leak_columns = set()
    for event_col, _, _ in EVENT_DEFINITIONS:
        event_leak_columns.add(event_col)
    # Also exclude derived Wyckoff columns that encode event state
    for col in df.columns:
        if any(prefix in col for prefix in [
            'wyckoff_spring', 'wyckoff_upthrust', 'wyckoff_sos', 'wyckoff_sow',
            'wyckoff_lps_count', 'wyckoff_lpsy_count', 'wyckoff_directional_score',
            'wyckoff_phase',  # phase encodes event progression
            'fvg_bullish', 'fvg_bearish', 'fvg_nearest',
            'bos_direction', 'liquidity_sweep', 'order_block',
        ]):
            event_leak_columns.add(col)
    log.info(f"Excluding {len(event_leak_columns)} event-leak columns from inputs")

    # Get all numeric feature columns (exclude targets + meta + event leaks)
    target_cols = [c for c in df.columns if c.startswith('next_') or c == 'triple_barrier_label']
    meta_cols = ['open_time', 'close_time', 'symbol', 'timeframe', 'index', 'date']
    exclude = set(target_cols + meta_cols) | event_leak_columns
    feature_cols = [c for c in df.columns if c not in exclude and
                    df[c].dtype in ('float64', 'float32', 'int64', 'int32', 'int8')]
    log.info(f"Input features: {len(feature_cols)} (after excluding event columns)")

    # Normalize
    X_raw = df[feature_cols].values.astype(np.float32)
    means = np.nanmean(X_raw, axis=0)
    stds = np.nanstd(X_raw, axis=0)
    stds[stds == 0] = 1
    X_norm = (X_raw - means) / stds
    # After z-score normalization, BEFORE feeding to LSTM (3B.1):
    # Append missing_mask so LSTM can distinguish "value is 0" from "value was NaN"
    missing_mask = np.isnan(X_norm).astype(np.float32)
    X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)
    X_norm = np.concatenate([X_norm, missing_mask], axis=-1)  # preserves NaN semantics

    # Find all events
    all_event_indices = {}
    event_names = []

    for event_col, cond_fn, desc in EVENT_DEFINITIONS:
        indices = find_events(df, event_col, cond_fn)
        if len(indices) >= 20:  # need at least 20 events for train+test
            all_event_indices[desc] = indices
            event_names.append(desc)
            log.info(f"  {desc}: {len(indices)} events")
        else:
            log.info(f"  {desc}: {len(indices)} events (skipped, too few)")

    # Dynamic events
    dyn_events = compute_dynamic_events(df)
    for name, vals in dyn_events.items():
        indices = np.where(vals == 1)[0]
        if len(indices) >= 20:
            all_event_indices[name] = indices
            event_names.append(name)
            log.info(f"  {name}: {len(indices)} events")

    if not event_names:
        raise ValueError("No events found with enough occurrences")

    n_events = len(event_names)
    log.info(f"Training on {n_events} event types")

    # All event positions (union) for negative sampling
    # Mark event bar + small buffer (3 bars each side) as "near event"
    # Don't mark the entire lookback window — that's too aggressive and kills negatives
    BUFFER = 3
    all_event_positions = set()
    for indices in all_event_indices.values():
        for idx in indices:
            for j in range(max(0, idx - BUFFER), min(len(X_norm), idx + BUFFER + 1)):
                all_event_positions.add(j)

    # Build BALANCED training samples per event type
    # Cap at MAX_POS_PER_EVENT to prevent dominant events from taking over
    MAX_POS_PER_EVENT = 5000
    pos_sequences = []
    pos_labels = []

    for evt_idx, evt_name in enumerate(event_names):
        indices = all_event_indices[evt_name]
        # Subsample if too many events
        if len(indices) > MAX_POS_PER_EVENT:
            indices = np.random.choice(indices, MAX_POS_PER_EVENT, replace=False)
            indices.sort()

        count = 0
        for idx in indices:
            # Window ends 1 bar BEFORE the event (gap prevents same-bar leak)
            end = idx - 1
            start = end - window
            if start < 0:
                continue
            seq = X_norm[start:end]
            if seq.shape[0] != window:
                continue
            label = np.zeros(n_events, dtype=np.float32)
            label[evt_idx] = 1.0
            pos_sequences.append(seq)
            pos_labels.append(label)
            count += 1
        log.info(f"    {evt_name}: {count} positive samples")

    # BALANCED negative samples: 1:1 ratio with positives
    # Sample windows that are NOT near any event
    n_neg_target = len(pos_sequences)
    neg_sequences = []
    neg_labels = []

    # Build list of valid negative positions (not near any event)
    valid_neg_positions = []
    for i in range(window + 1, len(X_norm)):
        if i not in all_event_positions:
            valid_neg_positions.append(i)

    if len(valid_neg_positions) < n_neg_target:
        log.warning(f"Only {len(valid_neg_positions)} valid negative positions, "
                    f"need {n_neg_target}. Using all available.")
        n_neg_target = len(valid_neg_positions)

    # Random sample from valid positions
    neg_indices = np.random.choice(valid_neg_positions,
                                    min(n_neg_target, len(valid_neg_positions)),
                                    replace=False)
    for idx in neg_indices:
        end = idx
        start = end - window
        seq = X_norm[start:end]
        if seq.shape[0] == window:
            neg_sequences.append(seq)
            neg_labels.append(np.zeros(n_events, dtype=np.float32))

    log.info(f"Samples: {len(pos_sequences)} positive, {len(neg_sequences)} negative (1:1 balanced)")

    # Combine and shuffle
    all_seqs = np.array(pos_sequences + neg_sequences)
    all_labels = np.array(pos_labels + neg_labels)
    perm = np.random.permutation(len(all_seqs))
    all_seqs = all_seqs[perm]
    all_labels = all_labels[perm]

    return all_seqs, all_labels, feature_cols, event_names, means, stds, window


# ============================================================
# TRAINING
# ============================================================

def train_precursor(tf_name, device=None):
    """Train precursor detection LSTM."""
    if not HAS_TORCH:
        raise ImportError("PyTorch required")

    X, y, feature_names, event_names, means, stds, window = prepare_precursor_data(tf_name)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_features = X.shape[2]
    n_events = y.shape[1]

    # 80/20 split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    train_ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    test_ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_test), torch.FloatTensor(y_test))

    _num_workers = min(2, os.cpu_count() // 4) if os.cpu_count() else 1
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                              num_workers=_num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=64,
                             num_workers=_num_workers, pin_memory=True)

    model = PrecursorLSTM(n_features, hidden_size=128, num_layers=2,
                          n_events=n_events, dropout=0.3).to(device)

    # Multi-GPU: wrap with DataParallel if multiple GPUs available
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpus > 1:
        model = nn.DataParallel(model)
        log.info(f"DataParallel across {n_gpus} GPUs")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.BCELoss()

    log.info(f"Model: PrecursorLSTM(input={n_features}, events={n_events})")
    log.info(f"Events: {event_names}")
    log.info(f"Device: {device}")

    best_loss = float('inf')
    best_state = None
    patience = 0

    for epoch in range(100):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(yb)

        model.eval()
        test_loss = 0
        correct_per_event = np.zeros(n_events)
        total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                test_loss += loss.item() * len(yb)
                # Per-event accuracy (threshold 0.5)
                pred_binary = (pred > 0.5).float()
                correct_per_event += (pred_binary == yb).float().sum(dim=0).cpu().numpy()
                total += len(yb)

        avg_test_loss = test_loss / max(1, total)
        event_accs = correct_per_event / max(1, total)

        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            acc_str = " ".join([f"{event_names[i][:8]}={event_accs[i]:.3f}" for i in range(n_events)])
            log.info(f"  Epoch {epoch+1:3d}: loss={avg_test_loss:.4f} {acc_str}")

        if patience >= 15:
            log.info(f"  Early stopping at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)

    save_path = os.path.join(
        os.environ.get('SAVAGE22_DB_DIR', PROJECT_DIR),
        f'precursor_{tf_name}.pt')
    torch.save({
        'model_state': model.state_dict(),
        'feature_names': feature_names,
        'event_names': event_names,
        'means': means, 'stds': stds,
        'input_size': n_features,
        'n_events': n_events,
        'window': window,
        'best_loss': best_loss,
        'tf_name': tf_name,
    }, save_path)
    log.info(f"Saved to {save_path}")

    return model, event_names, best_loss


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--tf', type=str, default='1h')
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    if not HAS_TORCH:
        print("ERROR: PyTorch required")
        sys.exit(1)

    if args.train:
        timeframes = ['1h', '4h', '1d', '1w'] if args.all else [args.tf]
        for tf in timeframes:
            log.info(f"\n{'='*60}")
            log.info(f"Training Precursor LSTM for {tf}")
            log.info(f"{'='*60}")
            try:
                model, events, loss = train_precursor(tf)
                log.info(f"  {tf}: {len(events)} events, best_loss={loss:.4f}")
            except Exception as e:
                log.error(f"  {tf}: Failed — {e}")
                import traceback
                traceback.print_exc()


if __name__ == '__main__':
    main()
