#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lstm_sequence_model.py — LSTM Sequence Model for Savage22
==========================================================
Learns temporal patterns that XGBoost cannot:
  - Post-signal price dynamics (what happens 5-50 bars after esoteric events)
  - Wyckoff phase transition sequences (SC -> AR -> ST -> Spring timing)
  - Multi-feature convergence patterns across time
  - CVD divergence resolution dynamics

Architecture (from 2025 research):
  1. LSTM encodes last N bars of normalized features into hidden state
  2. Hidden state output becomes features for XGBoost (stacking)
  3. Also outputs its own directional probability for ensemble

Usage:
    # Train standalone
    python lstm_sequence_model.py --train --tf 1h

    # Extract features for XGBoost stacking
    from lstm_sequence_model import LSTMFeatureExtractor
    extractor = LSTMFeatureExtractor('1h')
    lstm_features = extractor.extract(feature_df)  # returns DataFrame with lstm_* columns

NOT wired into live_trader.py yet — standalone module for testing.
"""
import os
import sys
import time
import logging
import argparse
import numpy as np
import pandas as pd
import sqlite3

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: PyTorch not installed. Run: pip install torch")

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)


def _safe_cuda_device():
    """Return cuda device if supported, else cpu. Handles unsupported GPU archs (e.g. sm_120)."""
    if not HAS_TORCH or not torch.cuda.is_available():
        return torch.device('cpu')
    try:
        _test = torch.zeros(1, device='cuda')
        _ = _test + 1
        return torch.device('cuda')
    except RuntimeError as e:
        _arch = torch.cuda.get_device_capability()
        log.warning("=" * 80)
        log.warning(f"LSTM SKIPPED GPU: sm_{_arch[0]}{_arch[1]} not supported by PyTorch {torch.__version__}")
        log.warning(f"FIX: pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128")
        log.warning(f"LSTM falling back to CPU — ensemble signal will be SLOWER")
        log.warning("=" * 80)
        return torch.device('cpu')

# ============================================================
# CONFIGURATION
# ============================================================

# Config scaled for ~150 input dims (expanded esoteric + TA sequence features)
# Hidden sized for RTX 3090 24GB VRAM
LSTM_CONFIG = {
    '15m': {'window': 48,  'hidden': 384, 'layers': 2, 'lr': 0.0005, 'epochs': 80,  'batch': 48},
    '1h':  {'window': 48,  'hidden': 384, 'layers': 2, 'lr': 0.0005, 'epochs': 80,  'batch': 48},
    '4h':  {'window': 36,  'hidden': 256, 'layers': 2, 'lr': 0.0003, 'epochs': 100, 'batch': 32},
    '1d':  {'window': 20,  'hidden': 128, 'layers': 2, 'lr': 0.0003, 'epochs': 120, 'batch': 32},
    '1w':  {'window': 12,  'hidden': 64,  'layers': 1, 'lr': 0.0003, 'epochs': 150, 'batch': 16},
}

# Features the LSTM focuses on (normalized OHLCV + key signals)
# These are the features where SEQUENCE MATTERS (not just point-in-time value)
SEQUENCE_FEATURES = {
    # Raw price action (normalized)
    'price': ['close', 'high', 'low', 'open', 'volume'],
    # Wyckoff state machine (sequence of events matters most)
    'wyckoff': ['wyckoff_phase', 'wyckoff_in_range', 'wyckoff_range_position',
                'wyckoff_spring', 'wyckoff_upthrust', 'wyckoff_sos', 'wyckoff_sow',
                'wyckoff_effort_vs_result', 'wyckoff_directional_score',
                'wyckoff_volume_trend'],
    # Order flow / CVD (divergence patterns over time)
    'flow': ['delta_bar', 'delta_ratio', 'cvd_slope', 'cvd_price_divergence'],
    # Volume Profile dynamics
    'volume_profile': ['vpoc_distance', 'vpoc_migration', 'value_area_position'],
    # ICT / SMC
    'ict': ['fvg_bullish', 'fvg_bearish', 'bos_direction', 'liquidity_sweep'],
    # Momentum / trend
    'momentum': ['rsi_14', 'macd_histogram', 'bb_pctb_20', 'atr_14_pct',
                 'volume_ratio', 'ema50_slope'],
    # Esoteric event signals — FULL SUITE (LSTM learns post-event dynamics)
    'esoteric_lunar': ['moon_phase', 'days_to_full_moon', 'days_to_new_moon',
                       'is_full_moon', 'is_new_moon', 'lunar_sin', 'lunar_cos'],
    'esoteric_astro': ['mercury_retrograde', 'eclipse_window', 'nakshatra',
                       'nakshatra_nature', 'nakshatra_guna', 'planetary_hour_idx',
                       'saros_cycle_phase', 'metonic_cycle_phase'],
    'esoteric_calendar': ['bazi_btc_friendly', 'bazi_day_element_idx', 'bazi_day_stem',
                          'bazi_day_branch', 'mayan_tzolkin_num', 'mayan_haab_coeff',
                          'hebrew_holiday', 'chinese_new_year_window', 'diwali_window',
                          'ramadan_window', 'fomc_window', 'fomc_day',
                          'bonus_season', 'tax_loss_harvest'],
    'esoteric_numerology': ['date_dr', 'date_palindrome', 'is_angel_number',
                            'is_master_number', 'is_fib_day', 'golden_ratio_day',
                            'is_223', 'pi_cycle_cross'],
    'esoteric_gematria': ['caution_gematria_1h', 'tweet_gem_dr_ord_mode',
                          'news_gem_dr_ord_mode', 'sport_winner_gem_dr_mode',
                          'gem_match_date_sport', 'gem_match_price_sport',
                          'gem_match_tweet_sport', 'gem_match_sport_news'],
    'esoteric_space': ['kp_index', 'sunspot_number', 'solar_flux_f107',
                       'sw_storm_level', 'schumann_peak'],
    # Decay features — LSTM learns post-event price dynamics over time
    'decay': ['eclipse_decay_fast', 'eclipse_decay_slow',
              'full_moon_decay_fast', 'full_moon_decay_slow',
              'new_moon_decay_fast', 'new_moon_decay_slow',
              'gold_tweet_decay_fast', 'gold_tweet_decay_slow',
              'red_tweet_decay_fast', 'red_tweet_decay_slow',
              'green_tweet_decay_fast', 'green_tweet_decay_slow',
              'caps_tweet_decay_fast', 'news_caution_decay_fast',
              'gem_match_decay_fast', 'high_fear_decay_fast', 'high_fear_decay_slow',
              'high_greed_decay_fast', 'high_greed_decay_slow',
              'kp_storm_decay_fast', 'sw_storm_decay', 'sw_severe_decay',
              'sport_upset_decay_fast', 'sport_upset_decay_slow'],
    # Bars-since features — temporal distance from events
    'bars_since': ['bars_since_full_moon', 'bars_since_new_moon', 'bars_since_eclipse',
                   'bars_since_gold_tweet', 'bars_since_red_tweet', 'bars_since_green_tweet',
                   'bars_since_caps_tweet', 'bars_since_sport_upset', 'bars_since_news_caution',
                   'bars_since_gem_match', 'bars_since_high_fear', 'bars_since_high_greed',
                   'bars_since_kp_storm', 'sw_bars_since_storm', 'sw_bars_since_severe'],
    # Sentiment / fear-greed dynamics over time
    'sentiment': ['fear_greed', 'fear_greed_lag24', 'fear_greed_lag72',
                  'fear_greed_lag120', 'fear_greed_lag240', 'fg_x_moon_phase'],
    # Cross-domain interactions (sequence of convergences)
    'cross_signals': ['cross_eclipse_x_moon', 'cross_eclipse_x_funding',
                      'cross_fg_fear_x_eclipse', 'cross_fg_fear_x_moon',
                      'cross_fg_fear_x_kp_storm', 'cross_kp_storm_x_moon',
                      'cross_kp_storm_x_funding', 'cross_master_num_x_eclipse',
                      'cross_nakshatra_x_moon', 'cross_mercury_retro_x_news_sent',
                      'cross_schumann_peak_x_funding', 'cross_shmita_x_kp_storm',
                      'cycle_confluence_score'],
    # Institutional
    'institutional': ['funding_rate', 'funding_zscore_30d', 'funding_regime',
                      'coinbase_premium', 'funding_rate_high'],
}


# ============================================================
# DATASET
# ============================================================

class SequenceDataset(Dataset):
    """Sliding window dataset for LSTM training.
    Uses numpy stride_tricks for zero-copy sliding windows."""

    def __init__(self, X, y, window_size):
        """
        X: (n_samples, n_features) numpy array
        y: (n_samples,) numpy array of labels
        window_size: number of bars per sequence
        """
        self.window = window_size
        self.n_samples = max(0, len(X) - window_size)

        if self.n_samples > 0:
            # Zero-copy sliding window view using stride_tricks
            # This creates a VIEW, not a copy — uses ~0 extra memory
            from numpy.lib.stride_tricks import as_strided
            n, f = X.shape
            strides = (X.strides[0], X.strides[0], X.strides[1])
            self.X_windows = as_strided(X, shape=(self.n_samples, window_size, f), strides=strides)
            self.labels = y[window_size:window_size + self.n_samples]
        else:
            self.X_windows = np.empty((0, window_size, X.shape[1] if len(X.shape) > 1 else 0))
            self.labels = np.empty(0)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        seq = torch.from_numpy(self.X_windows[idx].copy()).float()
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return seq, label


# ============================================================
# MODEL
# ============================================================

class LSTMDirectionModel(nn.Module):
    """LSTM for directional probability prediction."""

    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, window, features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]
        out = self.dropout(last_hidden)
        out = self.relu(self.fc1(out))
        out = self.sigmoid(self.fc2(out))
        return out.squeeze(-1)

    def forward_with_hidden(self, x):
        """Single forward pass returning BOTH prediction AND hidden state.
        Eliminates the redundant second LSTM pass from extract_hidden()."""
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        out = self.dropout(last_hidden)
        out = self.relu(self.fc1(out))
        out = self.sigmoid(self.fc2(out))
        return out.squeeze(-1), last_hidden

    def extract_hidden(self, x):
        """Extract hidden state features for XGBoost stacking."""
        with torch.no_grad():
            lstm_out, (h_n, c_n) = self.lstm(x)
            last_hidden = lstm_out[:, -1, :]
            return last_hidden.cpu().numpy()


# ============================================================
# TRAINING
# ============================================================

def prepare_data(tf_name):
    """Load feature DB and prepare sequences."""
    cfg = LSTM_CONFIG.get(tf_name, LSTM_CONFIG['1h'])
    window = cfg['window']

    # Load features — parquet-first (pipeline now outputs parquet)
    parquet_path = os.path.join(PROJECT_DIR, f'features_{tf_name}.parquet')
    db_path = os.path.join(PROJECT_DIR, f'features_{tf_name}.db')
    table = f'features_{tf_name}'
    data_source = None

    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        data_source = parquet_path
        log.info(f"Loaded {len(df)} rows from parquet: {parquet_path}")
    elif os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        ext_check = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table + '_ext',)
        ).fetchone()
        if ext_check:
            df_main = pd.read_sql(f"SELECT * FROM {table} ORDER BY rowid", conn)
            df_ext = pd.read_sql(f"SELECT * FROM {table}_ext ORDER BY rowid", conn)
            df_ext = df_ext.drop(columns=['timestamp'], errors='ignore')
            df = pd.concat([df_main, df_ext], axis=1)
        else:
            df = pd.read_sql(f"SELECT * FROM {table} ORDER BY rowid", conn)
        conn.close()
        data_source = db_path
        log.info(f"Loaded {len(df)} rows from {db_path}")
    else:
        raise FileNotFoundError(f"No feature data found — checked {parquet_path} and {db_path}")

    # Get target column — triple_barrier_label is canonical for parquet pipeline
    target_col = 'triple_barrier_label'
    if target_col not in df.columns:
        # Fallback to legacy direction column
        target_col = f'next_{tf_name}_direction'
        if target_col not in df.columns:
            raise ValueError(f"No target column found in {data_source}. "
                             f"Expected 'triple_barrier_label' or 'next_{tf_name}_direction'. "
                             f"Available columns: {[c for c in df.columns if 'label' in c.lower() or 'target' in c.lower() or 'direction' in c.lower()]}")

    # Use ALL features from the DB — let the LSTM learn what matters
    # Exclude: raw OHLCV (already captured via returns), targets, metadata
    target_cols = [c for c in df.columns if c.startswith('next_') or c == 'triple_barrier_label']
    meta_cols = ['open_time', 'close_time', 'symbol', 'timeframe', 'index', 'date']
    exclude = set(target_cols + meta_cols)
    available = [c for c in df.columns if c not in exclude and df[c].dtype in ('float64', 'float32', 'int64', 'int32', 'int8')]
    log.info(f"LSTM input features: {len(available)} (ALL numeric features from DB)")

    if len(available) < 5:
        raise ValueError(f"Too few features available: {len(available)}")

    # Prepare X and y
    X = df[available].copy()
    y = df[target_col].copy()

    # Drop rows where target is NaN
    valid = y.notna()
    X = X[valid].reset_index(drop=True)
    y = y[valid].reset_index(drop=True)

    # Normalize features (z-score per column, handling NaN)
    means = X.mean()
    stds = X.std().replace(0, 1)
    X_norm = (X - means) / stds
    # NaN preserved — LSTM sees missing data as sequence gaps

    # Preserve NaN semantics for LSTM: add binary missing indicators (fix 3B.1)
    # Bare nan_to_num(nan=0.0) violates philosophy (NaN ≠ 0). Missing indicators let LSTM learn "this was missing."
    X_scaled = X_norm.values.astype(np.float32)
    missing_mask = np.isnan(X_scaled).astype(np.float32)  # binary: 1 = was missing
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    X_arr = np.concatenate([X_scaled, missing_mask], axis=1)  # doubles features but preserves NaN semantics
    y_arr = y.values.astype(np.float32)

    # For triple barrier: convert 0=SHORT, 1=FLAT, 2=LONG to binary
    # Only train on LONG (2) and SHORT (0), map to 1/0
    if target_col == 'triple_barrier_label':
        mask = y_arr != 1  # exclude FLAT
        X_arr = X_arr[mask]
        y_arr = y_arr[mask]
        y_arr = (y_arr == 2).astype(np.float32)  # LONG=1, SHORT=0

    log.info(f"Training data: {len(X_arr)} samples, {X_arr.shape[1]} features, window={window}")
    log.info(f"Class balance: {y_arr.mean():.3f} (long ratio)")

    return X_arr, y_arr, available, means.values, stds.values, cfg


def train_lstm(tf_name, device=None):
    """Train LSTM model for a timeframe.

    NOTE: LSTM operates on ~3000 base features only, NOT the 2.9M sparse cross features.
    Cross features are sparse binary (0/1) — LSTM can't learn temporal patterns from sparse binary sequences.
    LightGBM handles cross features via EFB (Exclusive Feature Bundling).
    The LSTM provides complementary sequential pattern detection on continuous base features.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required. Install: pip install torch")

    X_arr, y_arr, feature_names, means, stds, cfg = prepare_data(tf_name)

    window = cfg['window']
    hidden = cfg['hidden']
    layers = cfg['layers']
    lr = cfg['lr']
    epochs = cfg['epochs']
    batch_size = cfg['batch']

    # FIX #22: Dynamic batch size — larger batches = better GPU utilization
    if len(X_arr) > 5000:
        batch_size = max(batch_size, 64)

    if device is None:
        if torch.cuda.is_available():
            # Smoke test: verify GPU arch is supported (RTX 5090 sm_120 may not be)
            try:
                _test = torch.zeros(1, device='cuda')
                _ = _test + 1
                device = torch.device('cuda')
            except RuntimeError as _cuda_err:
                _arch = torch.cuda.get_device_capability()
                log.warning(f"WARNING: CUDA available but GPU compute sm_{_arch[0]}{_arch[1]} "
                            f"not supported by this PyTorch build: {_cuda_err}")
                log.warning(f"WARNING: Falling back to CPU for LSTM training")
                device = torch.device('cpu')
        else:
            device = torch.device('cpu')
    log.info(f"Device: {device}")

    # Walk-forward split (80% train, 20% test — no shuffling for time series)
    split_idx = int(len(X_arr) * 0.8)
    X_train, X_test = X_arr[:split_idx], X_arr[split_idx:]
    y_train, y_test = y_arr[:split_idx], y_arr[split_idx:]

    train_ds = SequenceDataset(X_train, y_train, window)
    test_ds = SequenceDataset(X_test, y_test, window)

    if len(train_ds) == 0 or len(test_ds) == 0:
        log.error(f"Not enough data for window={window}: train={len(X_train)}, test={len(X_test)}")
        return None

    _num_workers = min(4, os.cpu_count() // 2) if os.cpu_count() else 2
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=_num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=_num_workers, pin_memory=True)

    # Model
    input_size = X_arr.shape[1]
    model = LSTMDirectionModel(input_size, hidden, layers).to(device)
    # torch.compile for kernel fusion (10-30% speedup)
    try:
        model = torch.compile(model)
        log.info(f"  torch.compile enabled")
    except Exception:
        pass  # compile not available on older PyTorch

    # Multi-GPU: wrap with DataParallel if multiple GPUs available
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpus > 1:
        model = nn.DataParallel(model)
        log.info(f"DataParallel across {n_gpus} GPUs")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.BCELoss()

    # FIX #13: AMP (mixed precision) — 1.5-2x speedup on Tensor Cores
    use_amp = (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    log.info(f"Model: LSTM(input={input_size}, hidden={hidden}, layers={layers})")
    log.info(f"Training: {epochs} epochs, batch={batch_size}, lr={lr}, AMP={use_amp}")

    best_acc = 0
    best_state = None
    patience_counter = 0
    patience_limit = 20

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(y_batch)
            train_correct += ((pred > 0.5).float() == y_batch).sum().item()
            train_total += len(y_batch)

        # Eval
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    pred = model(X_batch)
                    loss = criterion(pred, y_batch)
                test_loss += loss.item() * len(y_batch)
                test_correct += ((pred > 0.5).float() == y_batch).sum().item()
                test_total += len(y_batch)

        train_acc = train_correct / max(1, train_total)
        test_acc = test_correct / max(1, test_total)
        avg_train_loss = train_loss / max(1, train_total)
        avg_test_loss = test_loss / max(1, test_total)

        scheduler.step(avg_test_loss)

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            log.info(f"  Epoch {epoch+1:3d}: train_acc={train_acc:.4f} test_acc={test_acc:.4f} "
                     f"train_loss={avg_train_loss:.4f} test_loss={avg_test_loss:.4f} "
                     f"best={best_acc:.4f}")

        if patience_counter >= patience_limit:
            log.info(f"  Early stopping at epoch {epoch+1}")
            break

    # Save best model
    if best_state:
        model.load_state_dict(best_state)

    save_path = os.path.join(PROJECT_DIR, f'lstm_{tf_name}.pt')
    torch.save({
        'model_state': model.state_dict(),
        'config': cfg,
        'feature_names': feature_names,
        'means': means,
        'stds': stds,
        'input_size': input_size,
        'best_accuracy': best_acc,
        'tf_name': tf_name,
    }, save_path)

    log.info(f"Saved model to {save_path}")
    log.info(f"Best test accuracy: {best_acc:.4f}")

    return model, best_acc


# ============================================================
# FEATURE EXTRACTION (for XGBoost stacking)
# ============================================================

class LSTMFeatureExtractor:
    """Extract LSTM hidden states as features for XGBoost."""

    def __init__(self, tf_name, device=None):
        self.tf_name = tf_name
        self.device = device or _safe_cuda_device()
        self.model = None
        self.feature_names = None
        self.means = None
        self.stds = None
        self.window = None
        self._load()

    def _load(self):
        """Load trained LSTM model."""
        path = os.path.join(PROJECT_DIR, f'lstm_{self.tf_name}.pt')
        if not os.path.exists(path):
            log.warning(f"No LSTM model found at {path}")
            return

        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        cfg = checkpoint['config']
        input_size = checkpoint['input_size']

        self.model = LSTMDirectionModel(
            input_size, cfg['hidden'], cfg['layers']
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()

        self.feature_names = checkpoint['feature_names']
        try:
            self.means = checkpoint['means']
            self.stds = checkpoint['stds']
        except (KeyError, TypeError):
            log.warning(f"LSTM checkpoint for {self.tf_name} missing means/stds — will compute at inference")
            self.means = None
            self.stds = None
        self.window = cfg['window']

        log.info(f"Loaded LSTM model for {self.tf_name}: {input_size} features, "
                 f"window={self.window}, acc={checkpoint.get('best_accuracy', 'N/A')}")

    def extract(self, feature_df):
        """
        Extract LSTM features from a feature DataFrame.

        Args:
            feature_df: DataFrame with columns matching self.feature_names

        Returns:
            DataFrame with columns: lstm_prob, lstm_hidden_0..N
        """
        if self.model is None:
            # Return NaN columns if no model
            n = len(feature_df)
            return pd.DataFrame({
                'lstm_prob': np.full(n, np.nan),
                'lstm_hidden_mean': np.full(n, np.nan),
                'lstm_hidden_std': np.full(n, np.nan),
            }, index=feature_df.index)

        # Select and normalize features
        available = [f for f in self.feature_names if f in feature_df.columns]
        if len(available) < len(self.feature_names):
            missing = set(self.feature_names) - set(available)
            log.warning(f"Missing {len(missing)} LSTM features: {list(missing)[:5]}...")

        X = feature_df[available].copy()

        # Use stored normalization params (only for available features)
        feat_idx = [self.feature_names.index(f) for f in available]
        if self.means is not None and self.stds is not None:
            means = self.means[feat_idx]
            stds = self.stds[feat_idx]
        else:
            # Fallback: compute from data if checkpoint had no means/stds
            means = X.mean().values
            stds = X.std().values
            stds = np.where(stds == 0, 1, stds)
        X_norm = (X - means) / np.where(stds == 0, 1, stds)
        # NaN preserved — LSTM sees missing data as sequence gaps
        X_arr = X_norm.values.astype(np.float32)

        n = len(X_arr)
        window = self.window

        # Output arrays
        probs = np.full(n, np.nan)
        hidden_means = np.full(n, np.nan)
        hidden_stds = np.full(n, np.nan)

        if n <= window:
            return pd.DataFrame({
                'lstm_prob': probs,
                'lstm_hidden_mean': hidden_means,
                'lstm_hidden_std': hidden_stds,
            }, index=feature_df.index)

        # Process sequences in batches of 48 (matches training batch size)
        # Uses forward_with_hidden() to get prediction + hidden in ONE forward pass
        # instead of model() + extract_hidden() = two redundant LSTM passes
        self.model.eval()
        batch_size = 48
        with torch.no_grad():
            # Pre-build all sequence indices
            indices = list(range(window, n))
            for batch_start in range(0, len(indices), batch_size):
                batch_indices = indices[batch_start:batch_start + batch_size]
                # Stack sequences into a single batch tensor
                batch_seqs = np.stack([X_arr[i - window:i] for i in batch_indices])
                batch_tensor = torch.FloatTensor(batch_seqs).to(self.device)
                # Single forward pass: prediction + hidden state together
                batch_probs, batch_hidden = self.model.forward_with_hidden(batch_tensor)
                # Extract results from batch output
                batch_probs_np = batch_probs.cpu().numpy()
                batch_hidden_np = batch_hidden.cpu().numpy()
                for j, idx in enumerate(batch_indices):
                    probs[idx] = batch_probs_np[j]
                    hidden_means[idx] = batch_hidden_np[j].mean()
                    hidden_stds[idx] = batch_hidden_np[j].std()

        return pd.DataFrame({
            'lstm_prob': probs,
            'lstm_hidden_mean': hidden_means,
            'lstm_hidden_std': hidden_stds,
        }, index=feature_df.index)


# ============================================================
# PLATT CALIBRATION + BLENDING
# ============================================================

def calibrate_lstm_platt(lstm_probs, y_true, save_path=None):
    """Fit Platt scaling (logistic regression) to calibrate LSTM probabilities.

    Args:
        lstm_probs: array of raw LSTM probabilities (0-1)
        y_true: array of true labels (0=SHORT, 1=FLAT, 2=LONG)
        save_path: optional path to save calibration model

    Returns:
        calibration model (sklearn LogisticRegression) or None
    """
    from sklearn.linear_model import LogisticRegression

    # Convert to binary: 1 if LONG (class 2), 0 otherwise
    # LSTM prob is P(price goes up), so calibrate against LONG label
    valid = ~np.isnan(lstm_probs)
    if valid.sum() < 50:
        return None

    X_cal = lstm_probs[valid].reshape(-1, 1)
    y_cal = (y_true[valid] == 2).astype(int)

    cal_model = LogisticRegression(C=1e10, max_iter=1000)  # C=large = no regularization (pure Platt)
    cal_model.fit(X_cal, y_cal)

    if save_path:
        import pickle
        with open(save_path, 'wb') as f:
            pickle.dump(cal_model, f)

    return cal_model


def apply_platt_calibration(lstm_probs, cal_model):
    """Apply Platt calibration to LSTM probabilities."""
    if cal_model is None:
        return lstm_probs
    valid = ~np.isnan(lstm_probs)
    result = lstm_probs.copy()
    if valid.sum() > 0:
        result[valid] = cal_model.predict_proba(lstm_probs[valid].reshape(-1, 1))[:, 1]
    return result


def blend_predictions(xgb_probs_3c, lstm_prob_calibrated, alpha=0.2):
    """Blend XGBoost and LSTM predictions.

    Single probability pipeline:
        p_blend = alpha * p_lstm + (1-alpha) * p_xgb

    Args:
        xgb_probs_3c: (N, 3) XGBoost probabilities [short, flat, long]
        lstm_prob_calibrated: (N,) calibrated LSTM P(long)
        alpha: LSTM weight (default 0.2 — conservative start)

    Returns:
        blended_probs_3c: (N, 3) blended probabilities
    """
    n = len(xgb_probs_3c)
    blended = xgb_probs_3c.copy()

    valid = ~np.isnan(lstm_prob_calibrated)
    if valid.sum() == 0:
        return blended

    # Blend the LONG probability
    blended[valid, 2] = alpha * lstm_prob_calibrated[valid] + (1 - alpha) * xgb_probs_3c[valid, 2]
    # Blend the SHORT probability (1 - lstm_prob is P(short-ish))
    blended[valid, 0] = alpha * (1 - lstm_prob_calibrated[valid]) + (1 - alpha) * xgb_probs_3c[valid, 0]
    # FLAT gets the remainder
    blended[valid, 1] = 1 - blended[valid, 2] - blended[valid, 0]
    blended[valid, 1] = np.maximum(blended[valid, 1], 0)  # ensure non-negative

    # Re-normalize
    row_sums = blended.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1)
    blended /= row_sums

    return blended


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='LSTM Sequence Model for Savage22')
    parser.add_argument('--train', action='store_true', help='Train LSTM model')
    parser.add_argument('--tf', type=str, default='1h', help='Timeframe (default: 1h)')
    parser.add_argument('--all', action='store_true', help='Train all timeframes')
    parser.add_argument('--test', action='store_true', help='Test feature extraction')
    args = parser.parse_args()

    if not HAS_TORCH:
        print("ERROR: PyTorch not installed. Run: pip install torch")
        sys.exit(1)

    if args.train:
        timeframes = list(LSTM_CONFIG.keys()) if args.all else [args.tf]
        for tf in timeframes:
            log.info(f"\n{'='*60}")
            log.info(f"Training LSTM for {tf}")
            log.info(f"{'='*60}")
            try:
                model, acc = train_lstm(tf)
                log.info(f"  {tf}: Best accuracy = {acc:.4f}")
            except Exception as e:
                log.error(f"  {tf}: Failed — {e}")

    elif args.test:
        log.info(f"Testing LSTM feature extraction for {args.tf}")
        extractor = LSTMFeatureExtractor(args.tf)

        # Load features — parquet-first (pipeline now outputs parquet)
        _pq_p = os.path.join(PROJECT_DIR, f'features_{args.tf}.parquet')
        _db_p = os.path.join(PROJECT_DIR, f'features_{args.tf}.db')
        _table = f'features_{args.tf}'
        if os.path.exists(_pq_p):
            df = pd.read_parquet(_pq_p)
        elif os.path.exists(_db_p):
            conn = sqlite3.connect(_db_p)
            _ext = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (_table + '_ext',)
            ).fetchone()
            if _ext:
                df_main = pd.read_sql(f"SELECT * FROM {_table} ORDER BY rowid", conn)
                df_ext = pd.read_sql(f"SELECT * FROM {_table}_ext ORDER BY rowid", conn)
                df_ext = df_ext.drop(columns=['timestamp'], errors='ignore')
                df = pd.concat([df_main, df_ext], axis=1)
            else:
                df = pd.read_sql(f"SELECT * FROM {_table} ORDER BY rowid", conn)
            conn.close()
        else:
            log.error(f"No feature data found — checked {_pq_p} and {_db_p}")
            sys.exit(1)

        # Extract features
        lstm_feats = extractor.extract(df.tail(500))
        for col in lstm_feats.columns:
            non_null = lstm_feats[col].notna().sum()
            log.info(f"  {col}: {non_null} non-null values")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
