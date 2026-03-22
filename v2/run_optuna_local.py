#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_optuna_local.py — Local Optuna LSTM Hyperparameter Search
==============================================================
Runs on local 13900K + RTX 3090. Finds optimal LSTM hyperparameters
per TF and trains 5-seed ensemble with the best config.

Usage:
    python run_optuna_local.py              # all TFs
    python run_optuna_local.py --tf 1h      # single TF
"""
import os
import sys
import json
import time
import argparse
import logging
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('SAVAGE22_DB_DIR', PROJECT_DIR)
os.environ.setdefault('SKIP_LLM', '1')

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lstm_sequence_model import prepare_data, LSTMDirectionModel, SequenceDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TF_ORDER = ["1w", "1d", "4h", "1h", "15m", "5m"]
N_TRIALS = {"1w": 100, "1d": 100, "4h": 200, "1h": 200, "15m": 200, "5m": 200}
N_SEEDS = 5
TIMEOUT_PER_TF = 1200  # 20 min max per TF


def run_optuna_for_tf(tf):
    """Run Optuna search + 5-seed ensemble for one timeframe."""
    log.info(f"\n{'='*60}")
    log.info(f"Optuna search for {tf}")
    log.info(f"{'='*60}")

    try:
        X_arr, y_arr, feat_names, means, stds, cfg = prepare_data(tf)
    except Exception as e:
        log.error(f"Skip {tf}: {e}")
        return None

    n_features = X_arr.shape[1]
    split = int(len(X_arr) * 0.8)
    X_train, X_test = X_arr[:split], X_arr[split:]
    y_train, y_test = y_arr[:split], y_arr[split:]

    log.info(f"  {len(X_arr)} samples, {n_features} features")
    log.info(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    def objective(trial):
        window = trial.suggest_int("window", 8, min(80, len(X_train) // 4))
        hidden = trial.suggest_categorical("hidden", [64, 128, 256, 512])
        layers = trial.suggest_int("layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.1, 0.6)
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        batch = trial.suggest_categorical("batch", [32, 64, 128])

        train_ds = SequenceDataset(X_train, y_train, window)
        test_ds = SequenceDataset(X_test, y_test, window)
        if len(train_ds) < batch or len(test_ds) < batch:
            return 0.5

        train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False, pin_memory=True)

        model = LSTMDirectionModel(n_features, hidden, layers, dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.BCELoss()

        best_acc = 0
        patience = 0
        for epoch in range(40):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            correct = total = 0
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb)
                    correct += ((pred > 0.5).float() == yb).sum().item()
                    total += len(yb)
            acc = correct / max(1, total)
            if acc > best_acc:
                best_acc = acc
                patience = 0
            else:
                patience += 1
            if patience >= 8:
                break

            trial.report(acc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_acc

    n_trials = N_TRIALS.get(tf, 100)
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )

    log.info(f"  Running {n_trials} trials (timeout {TIMEOUT_PER_TF}s)...")
    start = time.time()
    study.optimize(objective, n_trials=n_trials, timeout=TIMEOUT_PER_TF)
    elapsed = time.time() - start

    best = study.best_trial
    log.info(f"  Best accuracy: {best.value:.4f} in {elapsed:.0f}s")
    log.info(f"  Best params: {best.params}")

    # Train 5-seed ensemble with best params
    bp = best.params
    window = bp["window"]
    train_ds = SequenceDataset(X_train, y_train, window)
    test_ds = SequenceDataset(X_test, y_test, window)
    train_loader = DataLoader(train_ds, batch_size=bp["batch"], shuffle=True,
                              num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bp["batch"], shuffle=False,
                             num_workers=0, pin_memory=True)

    ensemble_states = []
    ensemble_accs = []
    for seed in range(N_SEEDS):
        torch.manual_seed(seed * 42 + 7)
        np.random.seed(seed * 42 + 7)
        m = LSTMDirectionModel(n_features, bp["hidden"], bp["layers"], bp["dropout"]).to(device)
        opt = torch.optim.Adam(m.parameters(), lr=bp["lr"], weight_decay=1e-5)
        crit = nn.BCELoss()
        best_s = None
        best_a = 0
        no_improve = 0
        for epoch in range(80):
            m.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = crit(m(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
            m.eval()
            c2 = t2 = 0
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = m(xb)
                    c2 += ((pred > 0.5).float() == yb).sum().item()
                    t2 += len(yb)
            a2 = c2 / max(1, t2)
            if a2 > best_a:
                best_a = a2
                best_s = {k: v.cpu().clone() for k, v in m.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if epoch > 20 and no_improve >= 8:
                break
        if best_s:
            ensemble_states.append(best_s)
            ensemble_accs.append(best_a)
            log.info(f"    Seed {seed}: {best_a:.4f}")

    if ensemble_states:
        save_path = os.path.join(PROJECT_DIR, f'lstm_{tf}.pt')
        torch.save({
            "ensemble_states": ensemble_states,
            "ensemble_accs": ensemble_accs,
            "config": bp,
            "feature_names": feat_names,
            "means": means, "stds": stds,
            "input_size": n_features,
            "best_accuracy": max(ensemble_accs),
            "mean_accuracy": sum(ensemble_accs) / len(ensemble_accs),
            "tf_name": tf,
            "n_seeds": N_SEEDS,
        }, save_path)
        log.info(f"  Saved {N_SEEDS}-seed ensemble: mean={sum(ensemble_accs)/len(ensemble_accs):.4f} best={max(ensemble_accs):.4f}")

    return {
        "accuracy": best.value,
        "params": best.params,
        "ensemble_mean": sum(ensemble_accs) / len(ensemble_accs) if ensemble_accs else 0,
        "ensemble_best": max(ensemble_accs) if ensemble_accs else 0,
        "time": elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf', type=str, nargs='*', default=None)
    args = parser.parse_args()

    timeframes = args.tf if args.tf else TF_ORDER

    log.info(f"Device: {device}")
    if torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
    log.info(f"Timeframes: {timeframes}")
    log.info(f"Trials: {N_TRIALS}")

    results = {}
    total_start = time.time()

    for tf in timeframes:
        result = run_optuna_for_tf(tf)
        if result:
            results[tf] = result

    # Save results
    results_path = os.path.join(PROJECT_DIR, 'lstm_optuna_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    total_elapsed = time.time() - total_start
    log.info(f"\n{'='*60}")
    log.info(f"ALL DONE in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    log.info(f"{'='*60}")
    for tf, r in results.items():
        log.info(f"  {tf}: best={r['accuracy']:.4f} ensemble_mean={r['ensemble_mean']:.4f} ({r['time']:.0f}s)")
    log.info(f"Results saved to {results_path}")


if __name__ == '__main__':
    main()
