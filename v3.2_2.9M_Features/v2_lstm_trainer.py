#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2_lstm_trainer.py — V2 LSTM Training Wrapper
===============================================
Wraps lstm_sequence_model.py for V2 multi-asset parquet data.

- Loads dense features from V2 parquets (NOT sparse .npz crosses)
- ALL GPUs work in unison via nn.DataParallel (no CUDA_VISIBLE_DEVICES pinning)
- Sequential TF training: one TF at a time, all GPUs per TF
- Epoch checkpointing, Platt calibration, alpha grid search

Usage:
    python v2_lstm_trainer.py --tf 1d 4h 1h 15m 5m          # sequential, all GPUs each
    python v2_lstm_trainer.py --tf 1d --resume                # resume from checkpoint
    python v2_lstm_trainer.py --tf 1d --alpha-search --xgb-probs oos_predictions_prod_BTC_1d.pkl
"""

import os
import sys
import json
import time
import math
import glob
import logging
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import cpu_count

# ── V2 directory ──
V2_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, V2_DIR)

# ── V2 imports ──
from lstm_sequence_model import (
    LSTMDirectionModel, SequenceDataset, LSTM_CONFIG, blend_predictions
)
from config import ALL_TRAINING, TRAINING_CRYPTO
from hardware_detect import detect_hardware, log_hardware
from atomic_io import atomic_save_torch, atomic_save_pickle, atomic_save_json

# ── PyTorch ──
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# ── Alpha grid for blending search ──
ALPHA_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

# ── Target column patterns ──
TARGET_PREFIXES = ('next_', 'triple_barrier_label')
META_COLS = {'open_time', 'close_time', 'symbol', 'timeframe', 'index', 'date',
             'timestamp', 'open', 'high', 'low', 'close', 'volume'}


# ============================================================
# 1) LOAD V2 DATA
# ============================================================

def load_v2_data(tf, symbols=None):
    """Load V2 parquet features for a timeframe (dense only, no sparse crosses).

    Args:
        tf: timeframe string ('1d', '4h', '1h', '15m', '5m')
        symbols: list of symbols to load (None = auto from config)

    Returns:
        X_array: (n_samples, n_features) float32 numpy array (z-score normalized)
        y_array: (n_samples,) float32 numpy array
        feature_names: list of feature column names
        means: per-column means used for z-score normalization
        stds: per-column stds used for z-score normalization
    """
    # Auto-select symbols from config
    if symbols is None:
        if tf in ('1d', '1w'):
            symbols = ALL_TRAINING
        else:
            symbols = TRAINING_CRYPTO

    log.info(f"Loading V2 data for {tf}: {len(symbols)} symbols")

    dfs = []
    for sym in symbols:
        parquet_path = os.path.join(V2_DIR, f"features_{sym}_{tf}.parquet")
        if not os.path.exists(parquet_path):
            log.warning(f"  Missing: {parquet_path}")
            continue
        df = pd.read_parquet(parquet_path)
        df['_symbol'] = sym
        dfs.append(df)
        log.info(f"  {sym}: {len(df)} rows, {len(df.columns)} cols")

    if not dfs:
        raise FileNotFoundError(f"No V2 parquets found for tf={tf} in {V2_DIR}")

    combined = pd.concat(dfs, ignore_index=True)
    log.info(f"Combined: {len(combined)} rows, {len(combined.columns)} columns")

    # ── Identify target column ──
    target_col = f"next_{tf}_direction"
    if target_col not in combined.columns:
        target_col = 'triple_barrier_label'
        if target_col not in combined.columns:
            # Search for any next_*_direction column
            candidates = [c for c in combined.columns if c.startswith('next_') and 'direction' in c]
            if candidates:
                target_col = candidates[0]
                log.info(f"  Using target: {target_col}")
            else:
                raise ValueError(f"No target column found. Columns: {list(combined.columns)[:20]}...")

    # ── Select numeric feature columns ──
    target_cols = {c for c in combined.columns if c.startswith('next_') or c == 'triple_barrier_label'}
    exclude = target_cols | META_COLS | {'_symbol'}
    numeric_dtypes = ('float64', 'float32', 'float16', 'int64', 'int32', 'int16', 'int8')
    feature_cols = [
        c for c in combined.columns
        if c not in exclude and str(combined[c].dtype) in numeric_dtypes
    ]
    log.info(f"Feature columns: {len(feature_cols)}")

    if len(feature_cols) < 5:
        raise ValueError(f"Too few numeric features: {len(feature_cols)}")

    # ── Extract X and y ──
    X = combined[feature_cols].copy()
    y = combined[target_col].copy()

    # Drop rows where target is NaN
    valid = y.notna()
    X = X[valid].reset_index(drop=True)
    y = y[valid].reset_index(drop=True)

    # ── For triple barrier: convert to binary ──
    if target_col == 'triple_barrier_label':
        mask = y != 1  # exclude FLAT
        X = X[mask].reset_index(drop=True)
        y = y[mask].reset_index(drop=True)
        y = (y == 2).astype(np.float32)  # LONG=1, SHORT=0

    # ── Normalize (z-score per column) ──
    means = X.mean()
    stds = X.std().replace(0, 1)
    X_norm = (X - means) / stds
    # NaN preserved — LSTM sees missing data as sequence gaps

    X_array = X_norm.values.astype(np.float32)
    y_array = y.values.astype(np.float32)

    log.info(f"Final dataset: {X_array.shape[0]} samples, {X_array.shape[1]} features")
    log.info(f"Class balance: {y_array.mean():.4f} (long ratio)")

    return X_array, y_array, feature_cols, means, stds


# ============================================================
# 2) TRAIN ONE TIMEFRAME
# ============================================================

def train_one_tf(tf, epochs=100, lr=0.0005, resume=False):
    """Train LSTM for one timeframe using all available GPUs via DataParallel.

    Args:
        tf: timeframe string
        epochs: max training epochs
        lr: learning rate
        resume: whether to resume from checkpoint

    Returns:
        dict with training metrics
    """
    t0 = time.time()

    # ── Hardware detection ──
    hw = detect_hardware()
    log_hardware(hw)
    n_gpus = hw['n_gpus']
    device = torch.device('cuda' if torch.cuda.is_available() and n_gpus > 0 else 'cpu')
    log.info(f"Training device: {device} | GPUs: {n_gpus}")

    # ── LSTM config ──
    cfg = LSTM_CONFIG.get(tf, LSTM_CONFIG['1h'])
    window = cfg['window']
    hidden = cfg['hidden']
    layers = cfg['layers']
    batch_size = cfg['batch']

    # ── V2_BATCH_SIZE env override (set by cloud runner OOM retry) ──
    env_batch = int(os.environ.get('V2_BATCH_SIZE', 0))
    if env_batch > 0:
        batch_size = env_batch
        log.info(f"V2_BATCH_SIZE override: batch_size={batch_size}")

    # DataParallel splits each batch across GPUs — keep base batch per-GPU
    # Do NOT scale batch_size linearly (OOMs on 8+ GPUs)
    if n_gpus > 1:
        log.info(f"DataParallel across {n_gpus} GPUs, batch_size={batch_size} per-GPU")

    # ── Load data ──
    X_arr, y_arr, feature_names, means, stds = load_v2_data(tf)
    input_size = X_arr.shape[1]

    # ── Train/val split: last 20% for validation ──
    split_idx = int(len(X_arr) * 0.8)
    X_train, X_val = X_arr[:split_idx], X_arr[split_idx:]
    y_train, y_val = y_arr[:split_idx], y_arr[split_idx:]

    train_ds = SequenceDataset(X_train, y_train, window)
    val_ds = SequenceDataset(X_val, y_val, window)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(f"Not enough data for window={window}: "
                         f"train={len(X_train)}, val={len(X_val)}")

    log.info(f"Train sequences: {len(train_ds)} | Val sequences: {len(val_ds)}")

    # ── DataLoaders (optimized) ──
    n_workers = min(32, max(1, cpu_count() // 8))  # 48 workers on 384-core, 8 on local
    log.info(f"DataLoader: num_workers={n_workers}, pin_memory=True, persistent_workers=True")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=n_workers, pin_memory=True, persistent_workers=True,
        prefetch_factor=2, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=n_workers, pin_memory=True, persistent_workers=True,
        prefetch_factor=2
    )

    # ── Model ──
    model = LSTMDirectionModel(input_size, hidden, layers).to(device)
    log.info(f"Model: LSTM(input={input_size}, hidden={hidden}, layers={layers})")

    # Wrap in DataParallel if multiple GPUs
    if n_gpus > 1:
        model = nn.DataParallel(model)
        log.info(f"Wrapped model in nn.DataParallel across {n_gpus} GPUs")

    # ── LR scaling for multi-GPU (sqrt scaling rule) ──
    effective_lr = lr * math.sqrt(n_gpus) if n_gpus > 1 else lr
    if n_gpus > 1:
        log.info(f"LR scaled for {n_gpus} GPUs: {lr} -> {effective_lr:.6f} (sqrt scaling)")

    # ── Mixed precision (AMP) ──
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
    log.info(f"AMP mixed precision: {'enabled' if device.type == 'cuda' else 'disabled (CPU)'}")

    optimizer = torch.optim.Adam(model.parameters(), lr=effective_lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.BCELoss()

    # ── Resume from checkpoint ──
    start_epoch = 0
    best_acc = 0.0
    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 20

    checkpoint_path = os.path.join(V2_DIR, f"lstm_{tf}_checkpoint.pt")
    best_path = os.path.join(V2_DIR, f"lstm_{tf}.pt")

    if resume and os.path.exists(checkpoint_path):
        log.info(f"Resuming from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        # Handle DataParallel state dict
        state_dict = ckpt['model_state']
        if n_gpus > 1 and not any(k.startswith('module.') for k in state_dict):
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}
        elif n_gpus <= 1 and any(k.startswith('module.') for k in state_dict):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch = ckpt['epoch'] + 1
        best_acc = ckpt.get('best_acc', 0.0)
        best_loss = ckpt.get('best_loss', float('inf'))
        patience_counter = ckpt.get('patience_counter', 0)
        log.info(f"Resumed at epoch {start_epoch}, best_acc={best_acc:.4f}")

    # ── Training loop ──
    log.info(f"Training {tf}: {epochs} epochs, batch={batch_size}, lr={lr}, "
             f"patience={patience_limit}")
    log.info(f"{'='*70}")

    best_state = None

    for epoch in range(start_epoch, epochs):
        epoch_t0 = time.time()

        # ── Train phase ──
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
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

        # ── Validation phase ──
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                    pred = model(X_batch)
                    loss = criterion(pred, y_batch)
                val_loss += loss.item() * len(y_batch)
                val_correct += ((pred > 0.5).float() == y_batch).sum().item()
                val_total += len(y_batch)

        train_acc = train_correct / max(1, train_total)
        val_acc = val_correct / max(1, val_total)
        avg_train_loss = train_loss / max(1, train_total)
        avg_val_loss = val_loss / max(1, val_total)
        epoch_time = time.time() - epoch_t0

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # ── Track best ──
        improved = False
        if val_acc > best_acc:
            best_acc = val_acc
            best_loss = avg_val_loss
            # Save raw state dict (unwrap DataParallel)
            raw_model = model.module if isinstance(model, nn.DataParallel) else model
            best_state = {k: v.cpu().clone() for k, v in raw_model.state_dict().items()}
            patience_counter = 0
            improved = True
        else:
            patience_counter += 1

        # ── Logging ──
        marker = " *BEST*" if improved else ""
        if (epoch + 1) % 5 == 0 or epoch == start_epoch or improved:
            log.info(
                f"  Epoch {epoch+1:3d}/{epochs} | "
                f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} | "
                f"train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} | "
                f"lr={current_lr:.6f} | {epoch_time:.1f}s{marker}"
            )

        # ── Checkpoint only on validation improvement (saves I/O) ──
        if improved:
            raw_model = model.module if isinstance(model, nn.DataParallel) else model
            ckpt_data = {
                'epoch': epoch,
                'model_state': raw_model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_acc': best_acc,
                'best_loss': best_loss,
                'patience_counter': patience_counter,
                'config': cfg,
                'input_size': input_size,
                'feature_names': feature_names,
                'tf': tf,
                'means': means,
                'stds': stds,
            }
            atomic_save_torch(ckpt_data, checkpoint_path)
            log.info(f"  Checkpoint saved (val_acc={best_acc:.4f})")

        # ── Early stopping ──
        if patience_counter >= patience_limit:
            log.info(f"  Early stopping at epoch {epoch+1} (patience={patience_limit})")
            break

    # ── Save best model ──
    if best_state is not None:
        save_data = {
            'model_state': best_state,
            'config': cfg,
            'feature_names': feature_names,
            'input_size': input_size,
            'best_accuracy': best_acc,
            'best_loss': best_loss,
            'tf_name': tf,
            'n_train': len(train_ds),
            'n_val': len(val_ds),
            'means': means,
            'stds': stds,
        }
        atomic_save_torch(save_data, best_path)
        log.info(f"Saved best model to {best_path} (val_acc={best_acc:.4f})")

    total_time = time.time() - t0
    log.info(f"Training complete for {tf}: {total_time:.1f}s | best_val_acc={best_acc:.4f}")

    # ── Reload best model state before Platt calibration ──
    if best_state is not None:
        raw_model = model.module if isinstance(model, nn.DataParallel) else model
        raw_model.load_state_dict(best_state)
        log.info(f"Reloaded best model state (val_acc={best_acc:.4f}) for Platt calibration")
    log.info(f"Running Platt calibration on validation set...")
    calibrate_platt(model, val_loader, device, tf)

    return {
        'tf': tf,
        'best_val_acc': best_acc,
        'best_val_loss': best_loss,
        'n_train': len(train_ds),
        'n_val': len(val_ds),
        'n_features': input_size,
        'total_time_s': total_time,
    }


# ============================================================
# 3) PLATT CALIBRATION
# ============================================================

def calibrate_platt(model, val_loader, device, tf):
    """Collect LSTM probs on validation set and fit Platt scaling.

    Saves calibration model to platt_{tf}.pkl via atomic_save_pickle.
    """
    from sklearn.linear_model import LogisticRegression

    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            pred = model(X_batch)
            all_probs.append(pred.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)

    if len(probs) < 50:
        log.warning(f"Too few validation samples ({len(probs)}) for Platt calibration")
        return None

    # Fit Platt scaling (logistic regression with no regularization)
    X_cal = probs.reshape(-1, 1)
    y_cal = labels.astype(int)

    cal_model = LogisticRegression(C=1e10, max_iter=1000)
    cal_model.fit(X_cal, y_cal)

    # Evaluate calibration
    cal_probs = cal_model.predict_proba(X_cal)[:, 1]
    cal_acc = ((cal_probs > 0.5).astype(int) == y_cal).mean()
    log.info(f"Platt calibration: raw_acc={(probs > 0.5).astype(int).mean():.4f} -> "
             f"cal_acc={cal_acc:.4f}")

    # Save atomically (platt_lstm_ to avoid collision with XGBoost 3-class platt_{tf}.pkl)
    platt_path = os.path.join(V2_DIR, f"platt_lstm_{tf}.pkl")
    atomic_save_pickle(cal_model, platt_path)
    log.info(f"Saved Platt calibrator to {platt_path}")

    return cal_model


# ============================================================
# 4) ALPHA GRID SEARCH
# ============================================================

def search_alpha(tf, xgb_probs_path):
    """Grid search optimal alpha for blending LSTM + XGBoost predictions.

    Args:
        tf: timeframe string
        xgb_probs_path: path to pickle with XGBoost OOS probabilities
            Expected format: dict with 'probs_3c' (N,3) and 'y_true' (N,) arrays

    Saves blend_config_{tf}.json with best alpha + metrics.
    """
    log.info(f"Alpha search for {tf}")
    log.info(f"  XGBoost probs: {xgb_probs_path}")

    # ── Load XGBoost OOS probs (supports both CPCV fold list and legacy dict format) ──
    if not os.path.exists(xgb_probs_path):
        raise FileNotFoundError(f"XGBoost probs not found: {xgb_probs_path}")

    with open(xgb_probs_path, 'rb') as f:
        xgb_data = pickle.load(f)

    if isinstance(xgb_data, list):
        # CPCV fold format from v2_multi_asset_trainer: list of dicts with test_indices
        all_indices, all_probs, all_labels = [], [], []
        for fold in xgb_data:
            if 'test_indices' in fold:
                all_indices.append(fold['test_indices'])
                all_probs.append(fold['y_pred_probs'])
                all_labels.append(fold['y_true'])
        if not all_indices:
            raise ValueError("OOS predictions have no test_indices — cannot align with LSTM")
        indices = np.concatenate(all_indices)
        probs = np.concatenate(all_probs)
        labels = np.concatenate(all_labels)
        order = np.argsort(indices)
        xgb_probs_3c = probs[order]
        y_true = labels[order]
        xgb_indices = indices[order]  # absolute row indices into full dataset
        log.info(f"  XGBoost OOS: {len(xgb_probs_3c)} samples from {len(xgb_data)} CPCV folds")
    elif isinstance(xgb_data, dict):
        xgb_probs_3c = xgb_data['probs_3c']
        y_true = xgb_data['y_true']
        xgb_indices = None  # Legacy format
        log.info(f"  XGBoost OOS: {len(xgb_probs_3c)} samples (legacy dict)")
    else:
        raise ValueError(f"Unexpected OOS format: {type(xgb_data)}")

    # ── Load LSTM model + get probs on validation set ──
    hw = detect_hardware()
    n_gpus = hw['n_gpus']
    device = torch.device('cuda' if torch.cuda.is_available() and n_gpus > 0 else 'cpu')

    best_path = os.path.join(V2_DIR, f"lstm_{tf}.pt")
    if not os.path.exists(best_path):
        raise FileNotFoundError(f"No trained LSTM found: {best_path}")

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    cfg = ckpt['config']
    input_size = ckpt['input_size']

    model = LSTMDirectionModel(input_size, cfg['hidden'], cfg['layers']).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    if n_gpus > 1:
        model = nn.DataParallel(model)

    # ── Load matching validation data ──
    X_arr, y_arr, _, _, _ = load_v2_data(tf)
    split_idx = int(len(X_arr) * 0.8)
    X_val = X_arr[split_idx:]
    y_val = y_arr[split_idx:]

    window = cfg['window']
    val_ds = SequenceDataset(X_val, y_val, window)
    n_workers = min(32, max(1, cpu_count() // 8))  # 48 workers on 384-core, 8 on local
    val_loader = DataLoader(
        val_ds, batch_size=cfg['batch'], shuffle=False,
        num_workers=n_workers, pin_memory=True, persistent_workers=True
    )

    # Collect LSTM probs
    lstm_probs = []
    lstm_labels = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            pred = model(X_batch)
            lstm_probs.append(pred.cpu().numpy())
            lstm_labels.append(y_batch.numpy())

    lstm_probs = np.concatenate(lstm_probs)
    lstm_labels = np.concatenate(lstm_labels)

    # ── Apply Platt calibration if available (E5: defensive null check) ──
    platt_path = os.path.join(V2_DIR, f"platt_lstm_{tf}.pkl")
    if os.path.exists(platt_path):
        with open(platt_path, 'rb') as f:
            cal_model = pickle.load(f)
        if cal_model is not None and hasattr(cal_model, 'predict_proba'):
            try:
                lstm_probs_cal = cal_model.predict_proba(lstm_probs.reshape(-1, 1))[:, 1]
                log.info(f"  Applied Platt calibration from {platt_path}")
            except Exception as e:
                lstm_probs_cal = lstm_probs
                log.warning(f"  Platt calibration failed ({e}), using raw LSTM probs")
        else:
            lstm_probs_cal = lstm_probs
            log.warning(f"  Platt calibrator is None/invalid, using raw LSTM probs")
    else:
        lstm_probs_cal = lstm_probs
        log.info(f"  No Platt calibrator found, using raw LSTM probs")

    # ── Align XGBoost and LSTM on the SAME samples ──
    if xgb_indices is not None:
        # CPCV format: align via test_indices that fall in LSTM validation range
        val_mask = xgb_indices >= split_idx
        xgb_val = xgb_probs_3c[val_mask]
        y_val_xgb = y_true[val_mask]
        xgb_val_relative = xgb_indices[val_mask] - split_idx

        n_lstm_val = len(lstm_probs_cal)
        valid = xgb_val_relative < n_lstm_val
        xgb_aligned = xgb_val[valid]
        lstm_aligned = lstm_probs_cal[xgb_val_relative[valid]]
        y_aligned = y_val_xgb[valid]
        log.info(f"  Aligned via test_indices: {len(y_aligned)} samples "
                 f"(XGB OOS in val range: {val_mask.sum()}, valid: {valid.sum()})")
    else:
        # Legacy dict format: use minimum length
        n_common = min(len(xgb_probs_3c), len(lstm_probs_cal))
        xgb_aligned = xgb_probs_3c[-n_common:]
        lstm_aligned = lstm_probs_cal[-n_common:]
        y_aligned = y_true[-n_common:]
        log.info(f"  Aligned samples (legacy): {n_common}")

    n_common = len(y_aligned)
    if n_common < 50:
        raise ValueError(f"Too few aligned samples: {len(y_aligned)}")

    # ── Vectorized grid search (all alphas at once) ──
    alphas = np.array(ALPHA_GRID, dtype=np.float32)
    # Blend all alphas simultaneously: shape (n_alphas, n_samples, n_classes)
    # blend = (1-alpha)*xgb + alpha*lstm
    xgb_exp = xgb_aligned[np.newaxis, :, :]   # (1, N, 3)
    lstm_exp = lstm_aligned[np.newaxis, :, :]  # (1, N, 3)
    alphas_exp = alphas[:, np.newaxis, np.newaxis]  # (A, 1, 1)
    all_blended = (1 - alphas_exp) * xgb_exp + alphas_exp * lstm_exp  # (A, N, 3)

    # Predicted classes for all alphas: (A, N)
    all_pred_classes = np.argmax(all_blended, axis=2)

    is_binary = y_aligned.max() <= 1
    eps = 1e-15
    all_blended_clipped = np.clip(all_blended, eps, 1 - eps)

    results = []
    for ai, alpha in enumerate(ALPHA_GRID):
        pred_class = all_pred_classes[ai]
        if is_binary:
            correct = ((pred_class == 2) == (y_aligned == 1)).sum()
        else:
            correct = (pred_class == y_aligned).sum()
        acc = correct / len(y_aligned)

        blended_clipped = all_blended_clipped[ai]
        if is_binary:
            long_prob = blended_clipped[:, 2]
            logloss = -np.mean(
                y_aligned * np.log(long_prob) + (1 - y_aligned) * np.log(1 - long_prob)
            )
        else:
            n_classes = all_blended.shape[2]
            logloss = 0.0
            for c in range(n_classes):
                mask = y_aligned == c
                if mask.sum() > 0:
                    logloss -= np.mean(np.log(blended_clipped[mask, c]))
            logloss /= n_classes

        results.append({
            'alpha': alpha,
            'accuracy': float(acc),
            'logloss': float(logloss),
        })
        log.info(f"  alpha={alpha:.2f}: acc={acc:.4f}, logloss={logloss:.4f}")

    # ── Find best alpha (by accuracy, then by logloss if tied) ──
    best = max(results, key=lambda r: (r['accuracy'], -r['logloss']))
    log.info(f"  BEST: alpha={best['alpha']:.2f}, acc={best['accuracy']:.4f}, "
             f"logloss={best['logloss']:.4f}")

    # ── Save blend config ──
    blend_config = {
        'tf': tf,
        'best_alpha': best['alpha'],
        'alpha': best['alpha'],
        'best_accuracy': best['accuracy'],
        'best_logloss': best['logloss'],
        'grid_results': results,
        'n_samples': n_common,
        'platt_calibrated': os.path.exists(platt_path),
    }
    config_path = os.path.join(V2_DIR, f"blend_config_{tf}.json")
    atomic_save_json(blend_config, config_path)
    log.info(f"Saved blend config to {config_path}")

    return blend_config


# ============================================================
# 5) MAIN — CLI + SEQUENTIAL TF LOOP
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='V2 LSTM Trainer — sequential TF training with all GPUs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python v2_lstm_trainer.py --tf 1d 4h 1h 15m 5m          # Train all TFs sequentially
  python v2_lstm_trainer.py --tf 1d --resume               # Resume from checkpoint
  python v2_lstm_trainer.py --tf 1d --alpha-search --xgb-probs oos_predictions_production_1d.pkl
        """
    )
    parser.add_argument('--tf', nargs='+', default=['1d'],
                        help='Timeframe(s) to train, processed sequentially (default: 1d)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Max epochs per TF (default: 100)')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate (default: 0.0005)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint')
    parser.add_argument('--alpha-search', action='store_true',
                        help='Run alpha grid search for LSTM/XGBoost blending')
    parser.add_argument('--xgb-probs', type=str, default=None,
                        help='Path to XGBoost OOS probabilities pickle (for --alpha-search)')
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("  V2 LSTM TRAINER")
    log.info("=" * 70)

    # ── Hardware report ──
    hw = detect_hardware()
    log_hardware(hw)

    if hw['n_gpus'] == 0:
        log.warning("NO GPU DETECTED — training will be very slow on CPU!")

    # ── Process each TF sequentially ──
    all_results = {}

    for tf in args.tf:
        log.info(f"\n{'='*70}")
        log.info(f"  TIMEFRAME: {tf}")
        log.info(f"{'='*70}")

        try:
            if args.alpha_search:
                # Alpha grid search mode
                if args.xgb_probs is None:
                    # Default: glob for any OOS predictions file for this TF
                    oos_candidates = glob.glob(os.path.join(V2_DIR, f"oos_predictions_*_{tf}.pkl"))
                    xgb_path = oos_candidates[0] if oos_candidates else os.path.join(V2_DIR, f"oos_predictions_prod_BTC_{tf}.pkl")
                else:
                    xgb_path = args.xgb_probs
                result = search_alpha(tf, xgb_path)
                all_results[tf] = result
            else:
                # Training mode
                result = train_one_tf(
                    tf,
                    epochs=args.epochs,
                    lr=args.lr,
                    resume=args.resume,
                )
                all_results[tf] = result

        except Exception as e:
            log.error(f"FAILED for {tf}: {e}", exc_info=True)
            all_results[tf] = {'error': str(e)}

    # ── Summary ──
    log.info(f"\n{'='*70}")
    log.info("  TRAINING SUMMARY")
    log.info(f"{'='*70}")
    for tf, result in all_results.items():
        if 'error' in result:
            log.info(f"  {tf}: FAILED — {result['error']}")
        elif 'best_alpha' in result:
            log.info(f"  {tf}: best_alpha={result['best_alpha']:.2f}, "
                     f"acc={result['best_accuracy']:.4f}")
        else:
            log.info(f"  {tf}: val_acc={result['best_val_acc']:.4f}, "
                     f"features={result['n_features']}, "
                     f"time={result['total_time_s']:.0f}s")
    log.info(f"{'='*70}")


if __name__ == '__main__':
    main()
