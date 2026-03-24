#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
runpod_train.py -- Automated RunPod Training Pipeline for Savage22
==================================================================
Creates an H200 SXM pod, uploads data in waves (train while uploading),
runs full pipeline: LightGBM retrain + LSTM train + Optuna LSTM tuning +
Optuna optimizer.

Wave strategy (train while big files upload):
  Wave 1: Scripts + small DBs (1W, 1D) -> start training 1W/1D immediately
  Wave 2: Medium DBs (4H, 1H) -> train 4H/1H + LSTM on small TFs
  Wave 3: Big DBs (15m, 5m) -> train 15m/5m + LSTM + optimizer

Usage:
    python runpod_train.py              # Full pipeline
    python runpod_train.py --dry-run    # Show plan without creating pod
    python runpod_train.py --skip-upload # Skip upload (data already on pod)
    python runpod_train.py --xgboost-only  # Only retrain LightGBM
    python runpod_train.py --lstm-only     # Only train LSTM
    python runpod_train.py --optimizer-only # Only run exhaustive optimizer
"""

import os
import sys
import time
import argparse
import subprocess

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Pod spec — H200 SXM (fastest for training + optimizer)
POD_NAME = "savage22-h200"
POD_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
GPU_TYPE = "NVIDIA H200"
GPU_COUNT = 1
VOLUME_GB = 100
CONTAINER_DISK_GB = 50
CLOUD_TYPE = "SECURE"  # H200 usually secure cloud
COST_PER_HOUR = 3.99   # H200 SXM typical rate

# SSH
SSH_KEY = os.path.expanduser("~/.ssh/id_ed25519")
REMOTE_DIR = "/workspace"

# Wave-based upload: smallest first, start training while big files transfer
WAVE_1_FILES = [
    # Scripts (tiny)
    "ml_multi_tf.py", "feature_library.py", "lstm_sequence_model.py",
    "exhaustive_optimizer.py", "knn_feature_engine.py",
    "universal_gematria.py", "universal_numerology.py",
    "universal_astro.py", "universal_sentiment.py",
    "feature_classifier.py", "data_access.py", "config.py",
    "llm_features.py",
    # Small DBs + data
    "features_1w.db", "features_1d.db",
    "kp_history.txt", "llm_cache.db",
]

WAVE_2_FILES = [
    # Medium DBs
    "features_4h.db", "features_1h.db",
    "btc_prices.db",
]

WAVE_3_FILES = [
    # Big DBs
    "features_15m.db", "features_5m.db",
]

ALL_UPLOAD_FILES = WAVE_1_FILES + WAVE_2_FILES + WAVE_3_FILES

# Files/patterns to download after training
DOWNLOAD_PATTERNS = [
    "model_*.json",
    "features_*_all.json",
    "platt_*.pkl",
    "lstm_*.pt",
    "ml_multi_tf_results.txt",
    "ml_multi_tf_configs.json",
    "optuna_configs*.json",
    "lstm_optuna_results.json",
    "training.log",
]

# Pip packages needed on the pod
PIP_PACKAGES = [
    "lightgbm",
    "scikit-learn",
    "pandas",
    "numpy",
    "scipy",
    "hmmlearn",
    "cupy-cuda12x",
    "optuna",
]


def load_api_key():
    """Load RUNPOD_API_KEY from .env file."""
    env_path = os.path.join(PROJECT_DIR, ".env")
    if not os.path.exists(env_path):
        print("ERROR: .env file not found at %s" % env_path)
        sys.exit(1)

    api_key = None
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("RUNPOD_API_KEY="):
                api_key = line.split("=", 1)[1].strip()
                break

    if not api_key:
        print("ERROR: RUNPOD_API_KEY not found in .env")
        sys.exit(1)

    return api_key


def health_check(api_key):
    """Verify SDK can connect before creating a pod."""
    import runpod
    runpod.api_key = api_key

    print("[HEALTH CHECK] Testing RunPod API connection...")
    try:
        gpus = runpod.get_gpus()
        if gpus is None:
            print("  ERROR: get_gpus() returned None -- check your API key")
            sys.exit(1)
        print("  OK -- API key valid, %d GPU types available" % len(gpus))

        # Check 5090 availability
        found_5090 = False
        for g in gpus:
            gid = g.get("id", "")
            if "5090" in gid:
                found_5090 = True
                print("  Found: %s" % gid)
                break
        if not found_5090:
            print("  WARNING: RTX 5090 not listed -- may be unavailable")
    except Exception as e:
        print("  ERROR: API connection failed -- %s" % str(e))
        sys.exit(1)


def create_pod(api_key):
    """Create a 2x RTX 5090 pod and return pod dict."""
    import runpod
    runpod.api_key = api_key

    print("\n[1/6] Creating pod: %s" % POD_NAME)
    print("  Image: %s" % POD_IMAGE)
    print("  GPU: %s x%d" % (GPU_TYPE, GPU_COUNT))
    print("  Volume: %dGB persistent, %dGB container" % (VOLUME_GB, CONTAINER_DISK_GB))
    print("  Cloud: %s" % CLOUD_TYPE)
    print("  Estimated cost: $%.2f/hr" % COST_PER_HOUR)

    pod = runpod.create_pod(
        name=POD_NAME,
        image_name=POD_IMAGE,
        gpu_type_id=GPU_TYPE,
        gpu_count=GPU_COUNT,
        cloud_type=CLOUD_TYPE,
        volume_in_gb=VOLUME_GB,
        container_disk_in_gb=CONTAINER_DISK_GB,
        ports="22/tcp,8787/tcp",
        volume_mount_path=REMOTE_DIR,
    )

    pod_id = pod["id"]
    print("  Pod created: %s" % pod_id)
    return pod


def wait_for_pod(api_key, pod_id, timeout=600):
    """Wait for pod to reach RUNNING state with SSH available."""
    import runpod
    runpod.api_key = api_key

    print("\n[2/6] Waiting for pod %s to be ready..." % pod_id)
    start = time.time()
    last_status = ""
    timeout = max(timeout, 600)  # minimum 10 min timeout

    while True:
        elapsed = time.time() - start
        if elapsed > timeout:
            print("  TIMEOUT after %ds -- pod never became ready" % timeout)
            return None

        pod = runpod.get_pod(pod_id)
        runtime = pod.get("runtime", {})
        if runtime is None:
            runtime = {}

        status = pod.get("desiredStatus", "UNKNOWN")
        ports = runtime.get("ports", [])

        # Extract SSH connection info from runtime ports
        ssh_host = None
        ssh_port = None
        if ports:
            for port_info in ports:
                if port_info.get("privatePort") == 22:
                    ssh_host = port_info.get("ip")
                    ssh_port = port_info.get("publicPort")
                    break

        if status != last_status:
            print("  Status: %s (%.0fs elapsed)" % (status, elapsed))
            last_status = status

        if status == "RUNNING" and ssh_host and ssh_port:
            print("  Pod ready! Testing SSH connection...")
            print("  SSH: ssh root@%s -p %s -i %s" % (ssh_host, ssh_port, SSH_KEY))
            # Actually test SSH connectivity with retries
            for attempt in range(12):  # 12 attempts x 15s = 3 min
                rc, out, err = ssh_command(ssh_host, int(ssh_port), "echo SSH_OK", timeout=10)
                if rc == 0 and "SSH_OK" in out:
                    print("  SSH connected!")
                    return {
                        "id": pod_id,
                        "ssh_host": ssh_host,
                        "ssh_port": int(ssh_port),
                        "cost_per_hr": COST_PER_HOUR,
                        "start_time": start,
                    }
                print("  Waiting for SSH... (attempt %d/12)" % (attempt + 1))
                time.sleep(15)
            print("  ERROR: SSH not available after 3 minutes of retries")
            return None

        if status in ("EXITED", "TERMINATED", "ERROR"):
            print("  ERROR: Pod entered state: %s" % status)
            return None

        time.sleep(10)


def ssh_command(host, port, cmd, timeout=None):
    """Run a command on the pod via SSH. Returns (returncode, stdout)."""
    ssh_args = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        "-p", str(port),
        "-i", SSH_KEY,
        "root@%s" % host,
        cmd,
    ]
    try:
        result = subprocess.run(
            ssh_args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "SSH command timed out"
    except Exception as e:
        return -1, "", str(e)


def wait_for_ssh(host, port, timeout=120):
    """Wait until SSH is accepting connections."""
    print("  Waiting for SSH to accept connections...")
    start = time.time()
    while time.time() - start < timeout:
        rc, out, err = ssh_command(host, port, "echo ready", timeout=10)
        if rc == 0 and "ready" in out:
            print("  SSH connected!")
            return True
        time.sleep(5)
    print("  ERROR: SSH not available after %ds" % timeout)
    return False


def upload_files(host, port):
    """Upload data files and scripts to the pod via SCP."""
    print("\n[3/6] Uploading files to pod...")

    missing = []
    to_upload = []
    total_size = 0

    for filename in ALL_UPLOAD_FILES:
        local_path = os.path.join(PROJECT_DIR, filename)
        if not os.path.exists(local_path):
            missing.append(filename)
        else:
            size = os.path.getsize(local_path)
            total_size += size
            to_upload.append((filename, local_path, size))

    if missing:
        print("  WARNING: Missing files (will skip):")
        for m in missing:
            print("    - %s" % m)

    print("  Uploading %d files (%.1f MB total)..." % (len(to_upload), total_size / 1024 / 1024))

    for i, (filename, local_path, size) in enumerate(to_upload):
        size_mb = size / 1024 / 1024
        print("  [%d/%d] %s (%.1f MB)" % (i + 1, len(to_upload), filename, size_mb))

        scp_args = [
            "scp",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR",
            "-P", str(port),
            "-i", SSH_KEY,
            local_path,
            "root@%s:%s/%s" % (host, REMOTE_DIR, filename),
        ]
        result = subprocess.run(scp_args, capture_output=True, text=True)
        if result.returncode != 0:
            print("    ERROR uploading %s: %s" % (filename, result.stderr.strip()))
            # Continue -- some files may be optional
        else:
            print("    OK")

    print("  Upload complete.")


def install_deps(host, port):
    """Install Python dependencies on the pod."""
    print("\n  Installing dependencies...")
    pip_cmd = "pip install --no-cache-dir %s" % " ".join(PIP_PACKAGES)
    rc, out, err = ssh_command(host, port, pip_cmd, timeout=600)
    if rc != 0:
        print("  WARNING: pip install returned code %d" % rc)
        if err:
            # Print last few lines of error
            lines = err.strip().split("\n")
            for line in lines[-5:]:
                print("    %s" % line)
    else:
        print("  Dependencies installed OK")


def stream_ssh_command(host, port, cmd, pod_info, label=""):
    """Run a command on pod via SSH and stream output with cost tracker."""
    ssh_args = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR",
        "-o", "ServerAliveInterval=30",
        "-p", str(port),
        "-i", SSH_KEY,
        "root@%s" % host,
        cmd,
    ]

    start = time.time()
    process = subprocess.Popen(
        ssh_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    try:
        for line in process.stdout:
            elapsed = time.time() - pod_info["start_time"]
            cost = (elapsed / 3600) * COST_PER_HOUR
            prefix = "  [$%.2f]" % cost
            if label:
                prefix += " [%s]" % label
            sys.stdout.write("%s %s" % (prefix, line))
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n  INTERRUPTED by user")
        process.terminate()

    process.wait()
    elapsed = time.time() - start
    return process.returncode, elapsed


def upload_wave(host, port, files, wave_name):
    """Upload a wave of files."""
    print("\n  --- Uploading %s ---" % wave_name)
    to_upload = []
    for filename in files:
        local_path = os.path.join(PROJECT_DIR, filename)
        if os.path.exists(local_path):
            size = os.path.getsize(local_path)
            to_upload.append((filename, local_path, size))
        else:
            print("  [SKIP] %s (not found)" % filename)

    total_mb = sum(s for _, _, s in to_upload) / 1024 / 1024
    print("  %d files, %.1f MB" % (len(to_upload), total_mb))

    for i, (filename, local_path, size) in enumerate(to_upload):
        size_mb = size / 1024 / 1024
        print("  [%d/%d] %s (%.1f MB)" % (i + 1, len(to_upload), filename, size_mb))
        scp_args = [
            "scp", "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
            "-P", str(port), "-i", SSH_KEY,
            local_path, "root@%s:%s/%s" % (host, REMOTE_DIR, filename),
        ]
        result = subprocess.run(scp_args, capture_output=True, text=True)
        if result.returncode != 0:
            print("    ERROR: %s" % result.stderr.strip())
    print("  %s upload complete." % wave_name)


def run_full_pipeline(host, port, pod_info, xgboost=True, lstm=True, optimizer=True):  # param name kept for CLI compat
    """Run the full training pipeline with wave-based uploading."""
    env = "SAVAGE22_DB_DIR=%s SKIP_LLM=1" % REMOTE_DIR

    # Install deps
    print("\n[3/6] Setting up pod environment...")
    install_deps(host, port)

    # Verify GPU
    gpu_check = "python -c \"import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); print('VRAM:', torch.cuda.get_device_properties(0).total_mem // 1024**3, 'GB') if torch.cuda.is_available() else None\""
    rc, out, _ = ssh_command(host, port, gpu_check, timeout=30)
    if rc == 0:
        print("  %s" % out.strip().replace("\n", "\n  "))

    # ================================================================
    # PHASE 1: LightGBM retrain (all 6 TFs)
    # ================================================================
    if xgboost:
        print("\n" + "=" * 60)
        print("[PHASE 1] LightGBM Retrain — All 6 Timeframes")
        print("=" * 60)

        train_cmd = "cd %s && %s python -u ml_multi_tf.py 2>&1 | tee training.log" % (REMOTE_DIR, env)
        rc, elapsed = stream_ssh_command(host, port, train_cmd, pod_info, "LightGBM")
        print("\n  LightGBM retrain: %s in %.0fs" % ("OK" if rc == 0 else "FAILED", elapsed))

    # ================================================================
    # PHASE 2: LSTM training (all 6 TFs, all features)
    # ================================================================
    if lstm:
        print("\n" + "=" * 60)
        print("[PHASE 2] LSTM Training — All 6 Timeframes, All Features")
        print("=" * 60)

        lstm_cmd = "cd %s && %s python -u lstm_sequence_model.py --train --all 2>&1 | tee lstm_training.log" % (REMOTE_DIR, env)
        rc, elapsed = stream_ssh_command(host, port, lstm_cmd, pod_info, "LSTM")
        print("\n  LSTM training: %s in %.0fs" % ("OK" if rc == 0 else "FAILED", elapsed))

    # ================================================================
    # PHASE 3: Optuna LSTM Hyperparameter Search
    # ================================================================
    if lstm:
        print("\n" + "=" * 60)
        print("[PHASE 3] Optuna LSTM Hyperparameter Search")
        print("=" * 60)

        # Create optuna search script on the pod
        optuna_script = '''
import json, sys, os
sys.path.insert(0, "/workspace")
os.environ["SAVAGE22_DB_DIR"] = "/workspace"
os.environ["SKIP_LLM"] = "1"

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("Optuna not installed, skipping")
    sys.exit(0)

import torch
from lstm_sequence_model import prepare_data, LSTM_CONFIG, LSTMDirectionModel, SequenceDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results = {}

for tf in ["1w", "1d", "4h", "1h", "15m"]:
    print(f"\\n=== Optuna search for {tf} ===")
    try:
        X_arr, y_arr, feat_names, means, stds, cfg = prepare_data(tf)
    except Exception as e:
        print(f"  Skip {tf}: {e}")
        continue

    n_features = X_arr.shape[1]
    split = int(len(X_arr) * 0.8)
    X_train, X_test = X_arr[:split], X_arr[split:]
    y_train, y_test = y_arr[:split], y_arr[split:]

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

        train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False)

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

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )
    n_trials = 200 if tf in ("1h", "4h", "15m") else 100
    study.optimize(objective, n_trials=n_trials, timeout=1200)

    best = study.best_trial
    print(f"  Best accuracy: {best.value:.4f}")
    print(f"  Best params: {best.params}")
    results[tf] = {"accuracy": best.value, "params": best.params}

    # Retrain with best params and save
    bp = best.params
    window = bp["window"]
    train_ds = SequenceDataset(X_train, y_train, window)
    test_ds = SequenceDataset(X_test, y_test, window)
    train_loader = DataLoader(train_ds, batch_size=bp["batch"], shuffle=True)

    model = LSTMDirectionModel(n_features, bp["hidden"], bp["layers"], bp["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=bp["lr"], weight_decay=1e-5)
    criterion = nn.BCELoss()

    best_state = None
    best_acc = 0
    for epoch in range(80):
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
            for xb, yb in DataLoader(test_ds, batch_size=bp["batch"]):
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                correct += ((pred > 0.5).float() == yb).sum().item()
                total += len(yb)
        acc = correct / max(1, total)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Multi-seed ensemble: train 5 models with best params, save all
    N_SEEDS = 5
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
                for xb, yb in DataLoader(test_ds, batch_size=bp["batch"]):
                    xb, yb = xb.to(device), yb.to(device)
                    pred = m(xb)
                    c2 += ((pred > 0.5).float() == yb).sum().item()
                    t2 += len(yb)
            a2 = c2 / max(1, t2)
            if a2 > best_a:
                best_a = a2
                best_s = {k: v.cpu().clone() for k, v in m.state_dict().items()}
            if a2 <= best_a and epoch > 20:
                break
        if best_s:
            ensemble_states.append(best_s)
            ensemble_accs.append(best_a)
            print(f"    Seed {seed}: {best_a:.4f}")

    if ensemble_states:
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
        }, f"/workspace/lstm_{tf}.pt")
        print(f"  Saved {N_SEEDS}-seed ensemble for {tf}: mean={sum(ensemble_accs)/len(ensemble_accs):.4f} best={max(ensemble_accs):.4f}")

with open("/workspace/lstm_optuna_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\\nOptuna results saved to lstm_optuna_results.json")
'''
        # Write optuna script to pod
        escaped = optuna_script.replace("'", "'\\''")
        write_cmd = "cat > %s/run_optuna.py << 'OPTUNA_EOF'\n%s\nOPTUNA_EOF" % (REMOTE_DIR, optuna_script)
        ssh_command(host, port, write_cmd, timeout=30)

        optuna_cmd = "cd %s && python -u run_optuna.py 2>&1 | tee optuna.log" % REMOTE_DIR
        rc, elapsed = stream_ssh_command(host, port, optuna_cmd, pod_info, "Optuna")
        print("\n  Optuna search: %s in %.0fs" % ("OK" if rc == 0 else "FAILED", elapsed))

    # ================================================================
    # PHASE 4: Exhaustive Trading Parameter Optimizer
    # ================================================================
    if optimizer:
        print("\n" + "=" * 60)
        print("[PHASE 4] Exhaustive Trading Parameter Optimizer")
        print("=" * 60)

        opt_cmd = "cd %s && %s python -u exhaustive_optimizer.py 2>&1 | tee optimizer.log" % (REMOTE_DIR, env)
        rc, elapsed = stream_ssh_command(host, port, opt_cmd, pod_info, "Optimizer")
        print("\n  Optimizer: %s in %.0fs" % ("OK" if rc == 0 else "FAILED", elapsed))

    print("\n" + "=" * 60)
    total_cost = ((time.time() - pod_info["start_time"]) / 3600) * COST_PER_HOUR
    print("  ALL PHASES COMPLETE. Running cost: $%.2f" % total_cost)
    print("=" * 60)


def download_results(host, port):
    """Download trained models and output files from the pod."""
    print("\n[5/6] Downloading results...")

    output_dir = os.path.join(PROJECT_DIR, "runpod_output")
    os.makedirs(output_dir, exist_ok=True)

    # List files matching our patterns on the remote
    find_cmd = "cd %s && ls -la %s 2>/dev/null" % (REMOTE_DIR, " ".join(DOWNLOAD_PATTERNS))
    rc, out, _ = ssh_command(host, port, find_cmd, timeout=30)
    if rc == 0 and out.strip():
        print("  Remote files found:")
        for line in out.strip().split("\n"):
            print("    %s" % line)

    # Download each pattern
    for pattern in DOWNLOAD_PATTERNS:
        # Get list of matching files
        ls_cmd = "cd %s && ls %s 2>/dev/null" % (REMOTE_DIR, pattern)
        rc, out, _ = ssh_command(host, port, ls_cmd, timeout=15)
        if rc != 0 or not out.strip():
            continue

        for remote_file in out.strip().split("\n"):
            remote_file = remote_file.strip()
            if not remote_file:
                continue
            local_file = os.path.join(output_dir, os.path.basename(remote_file))
            print("  Downloading: %s" % remote_file)

            scp_args = [
                "scp",
                "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null",
                "-o", "LogLevel=ERROR",
                "-P", str(port),
                "-i", SSH_KEY,
                "root@%s:%s/%s" % (host, REMOTE_DIR, remote_file),
                local_file,
            ]
            result = subprocess.run(scp_args, capture_output=True, text=True)
            if result.returncode != 0:
                print("    ERROR: %s" % result.stderr.strip())
            else:
                print("    -> %s" % local_file)

    # Also copy models to project dir root (where live_trader.py expects them)
    print("\n  Copying models to project root...")
    for f in os.listdir(output_dir):
        src = os.path.join(output_dir, f)
        dst = os.path.join(PROJECT_DIR, f)
        if os.path.isfile(src):
            import shutil
            shutil.copy2(src, dst)
            print("    %s -> %s" % (f, dst))

    print("  Download complete. Results in: %s" % output_dir)


def terminate_pod(api_key, pod_id, pod_info=None):
    """Terminate the pod to stop all charges."""
    import runpod
    runpod.api_key = api_key

    print("\n[6/6] Terminating pod %s..." % pod_id)
    try:
        runpod.terminate_pod(pod_id)
        print("  Pod terminated. All charges stopped.")
    except Exception as e:
        print("  ERROR terminating pod: %s" % str(e))
        print("  MANUAL ACTION REQUIRED: Go to https://www.console.runpod.io/pods")
        print("  and terminate pod %s to avoid charges!" % pod_id)

    if pod_info:
        total_elapsed = time.time() - pod_info["start_time"]
        total_cost = (total_elapsed / 3600) * COST_PER_HOUR
        print("\n  Session summary:")
        print("    Duration: %.0f seconds (%.1f hours)" % (total_elapsed, total_elapsed / 3600))
        print("    Estimated cost: $%.2f" % total_cost)


def dry_run():
    """Show what would happen without actually creating a pod."""
    print("=" * 60)
    print("DRY RUN -- No pod will be created")
    print("=" * 60)

    print("\n[1] Would create pod:")
    print("  Name: %s" % POD_NAME)
    print("  Image: %s" % POD_IMAGE)
    print("  GPU: %s x%d" % (GPU_TYPE, GPU_COUNT))
    print("  Volume: %dGB persistent, %dGB container" % (VOLUME_GB, CONTAINER_DISK_GB))
    print("  Cloud: %s" % CLOUD_TYPE)
    print("  Cost: $%.2f/hr" % COST_PER_HOUR)

    print("\n[2] Would upload %d files:" % len(ALL_UPLOAD_FILES))
    total_size = 0
    for filename in ALL_UPLOAD_FILES:
        local_path = os.path.join(PROJECT_DIR, filename)
        if os.path.exists(local_path):
            size = os.path.getsize(local_path)
            total_size += size
            print("  [OK] %s (%.1f MB)" % (filename, size / 1024 / 1024))
        else:
            print("  [MISSING] %s" % filename)
    print("  Total: %.1f MB" % (total_size / 1024 / 1024))

    print("\n[3] Would install: %s" % ", ".join(PIP_PACKAGES))

    print("\n[4] Would run: ml_multi_tf.py")
    print("  Timeframes: 1w, 1d, 4h, 1h, 15m, 5m")
    print("  Multi-GPU: Dask with %d workers" % GPU_COUNT)

    print("\n[5] Would download:")
    for p in DOWNLOAD_PATTERNS:
        print("  %s" % p)

    print("\n[6] Would terminate pod")

    # Cost estimate
    print("\n  Estimated training time: 2-6 hours")
    print("  Estimated cost: $%.2f - $%.2f" % (2 * COST_PER_HOUR, 6 * COST_PER_HOUR))

    # Check SSH key
    if os.path.exists(SSH_KEY):
        print("\n  SSH key: %s [OK]" % SSH_KEY)
    else:
        print("\n  SSH key: %s [MISSING] -- add your key to RunPod console" % SSH_KEY)


def main():
    parser = argparse.ArgumentParser(description="RunPod training pipeline for Savage22")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without creating pod")
    parser.add_argument("--skip-upload", action="store_true", help="Skip file upload")
    parser.add_argument("--xgboost-only", action="store_true", help="Only retrain LightGBM")
    parser.add_argument("--lstm-only", action="store_true", help="Only train LSTM + Optuna")
    parser.add_argument("--optimizer-only", action="store_true", help="Only run Optuna optimizer")
    args = parser.parse_args()

    api_key = load_api_key()

    if args.dry_run:
        health_check(api_key)
        dry_run()
        return

    health_check(api_key)

    if not os.path.exists(SSH_KEY):
        print("WARNING: SSH key not found at %s" % SSH_KEY)
        print("Make sure your public key is added to RunPod console.")
        resp = input("Continue anyway? (y/N): ").strip().lower()
        if resp != "y":
            print("Aborted.")
            return

    # Determine what to run
    do_xgb = not (args.lstm_only or args.optimizer_only)
    do_lstm = not (args.xgboost_only or args.optimizer_only)
    do_opt = not (args.xgboost_only or args.lstm_only)

    # Create pod
    pod = create_pod(api_key)
    pod_id = pod["id"]
    print("  Pod ID: %s -- remember this in case of errors!" % pod_id)

    # Everything from here is wrapped in try/finally to ensure cleanup
    pod_info = None
    try:
        # Wait for pod to be ready
        pod_info = wait_for_pod(api_key, pod_id, timeout=600)
        if pod_info is None:
            print("ERROR: Pod failed to start. Terminating...")
            terminate_pod(api_key, pod_id)
            return

        host = pod_info["ssh_host"]
        port = pod_info["ssh_port"]

        # Wait for SSH
        if not wait_for_ssh(host, port, timeout=120):
            print("ERROR: SSH not available. Terminating pod...")
            terminate_pod(api_key, pod_id, pod_info)
            return

        # Upload files in waves
        if not args.skip_upload:
            upload_wave(host, port, WAVE_1_FILES, "Wave 1 (scripts + small DBs)")
            upload_wave(host, port, WAVE_2_FILES, "Wave 2 (medium DBs)")
            upload_wave(host, port, WAVE_3_FILES, "Wave 3 (large DBs)")
        else:
            print("\n  SKIPPED upload (--skip-upload)")

        # Run full pipeline
        run_full_pipeline(host, port, pod_info,
                         xgboost=do_xgb, lstm=do_lstm, optimizer=do_opt)

        # Download results
        download_results(host, port)

    except KeyboardInterrupt:
        print("\n\nINTERRUPTED by user!")
        print("Pod %s is still running -- terminating..." % pod_id)
    except Exception as e:
        print("\nERROR: %s" % str(e))
        print("Pod %s may still be running!" % pod_id)
    finally:
        # Always terminate to avoid charges
        terminate_pod(api_key, pod_id, pod_info)

    print("\nDone! Check ./runpod_output/ for trained models.")


if __name__ == "__main__":
    main()
