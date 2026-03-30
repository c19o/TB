#!/usr/bin/env python3
"""
cloud_auto.py — Fully automated cloud training orchestrator.

Automates: search → rent → wait → SSH → deploy → run → monitor → download → destroy
across multiple providers with failover: vast.ai → GCP spot (c3d-highmem-360).

Usage:
  # Full automated pipeline for a timeframe:
  python v3.3/cloud_auto.py --tf 1w

  # Search only (show available machines, don't rent):
  python v3.3/cloud_auto.py --tf 1d --search-only

  # Split pipeline: cross-gen on big-RAM machine, train on cheaper machine:
  python v3.3/cloud_auto.py --tf 1h --split-pipeline

  # Resume from checkpoint after machine death:
  python v3.3/cloud_auto.py --tf 4h --resume

  # Use specific provider:
  python v3.3/cloud_auto.py --tf 1w --provider vast

  # Dry run (show what would happen):
  python v3.3/cloud_auto.py --tf 1d --dry-run

Environment variables:
  VAST_API_KEY        — vast.ai API key (required for vast.ai provider)
  GCP_PROJECT         — GCP project ID (required for GCP provider)
  GCP_ZONE            — GCP zone (default: us-central1-a)
  SSH_KEY_PATH        — Path to SSH private key (default: ~/.ssh/vast_key)
  CHECKPOINT_DIR      — Local checkpoint directory (default: ./checkpoints)
  UPLOAD_TAR          — Path to code tar (default: /tmp/v33_upload.tar.gz)
  DB_TAR              — Path to DB tar (default: /tmp/v33_dbs.tar.gz)
"""

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional


# =============================================================================
# CONFIGURATION
# =============================================================================

class Timeframe(str, Enum):
    TF_1W  = "1w"
    TF_1D  = "1d"
    TF_4H  = "4h"
    TF_1H  = "1h"
    TF_15M = "15m"


@dataclass
class TFSpec:
    """Resource requirements per timeframe."""
    ram_gb: int           # Minimum system RAM in GB
    min_cores: int        # Minimum CPU cores (vCPUs)
    budget_usd: float     # Total budget for this TF
    est_hours: tuple      # (min_hours, max_hours) expected wallclock
    disk_gb: int          # Minimum disk space in GB
    # For split pipeline: training needs less RAM than cross-gen
    train_ram_gb: int     # RAM needed for training phase only
    train_cores: int      # Cores needed for training phase only


TF_SPECS = {
    Timeframe.TF_1W:  TFSpec(64,   64,  2,   (0.3, 0.7),  30,  32,   32),
    Timeframe.TF_1D:  TFSpec(256,  128, 10,  (2, 4),       50,  128,  64),
    Timeframe.TF_4H:  TFSpec(768,  128, 20,  (4, 8),       50,  256,  128),
    Timeframe.TF_1H:  TFSpec(1536, 256, 50,  (12, 24),     80,  512,  128),
    Timeframe.TF_15M: TFSpec(2048, 256, 100, (24, 48),     150, 768,  256),
}

# Docker image — lightweight, usually cached on vast.ai
BASE_IMAGE = 'pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime'

# SSH settings
SSH_RETRIES_MAX    = 8
SSH_BACKOFF_BASE   = 5     # seconds
SSH_CONNECT_TIMEOUT = 20   # seconds
HEARTBEAT_INTERVAL = 30    # seconds
HEARTBEAT_FAILURES = 3     # consecutive failures before declaring death

# Checkpoint settings
CHECKPOINT_INTERVAL = 300  # seconds between artifact downloads

# Remote paths
REMOTE_WORKSPACE = '/workspace'
REMOTE_V33_DIR   = '/workspace/v3.3'

# Pipeline artifacts to download after each critical step
CRITICAL_ARTIFACTS = {
    'cross_gen': [
        'v2_crosses_BTC_{tf}.npz',
        'v2_cross_names_BTC_{tf}.json',
        'features_BTC_{tf}.parquet',
    ],
    'training': [
        'model_{tf}.json',
        'model_{tf}_cpcv_backup.json',
        'ml_multi_tf_configs.json',
        'features_{tf}_all.json',
        'cpcv_oos_predictions_{tf}.pkl',
        'feature_importance_stability_{tf}.json',
    ],
    'optimizer': [
        'optuna_configs_{tf}.json',
        'optimization_results.csv',
    ],
    'meta': [
        'meta_model_{tf}.pkl',
        'platt_{tf}.pkl',
    ],
    'lstm': [
        'lstm_{tf}.pt',
    ],
    'inference': [
        'inference_{tf}_thresholds.json',
        'inference_{tf}_cross_pairs.npz',
        'inference_{tf}_ctx_names.json',
        'inference_{tf}_base_cols.json',
        'inference_{tf}_cross_names.json',
    ],
    'shap': [
        'shap_analysis_{tf}.json',
    ],
}

# Pip dependencies (same as cloud_run_tf.py)
PIP_DEPS = (
    'lightgbm scikit-learn scipy ephem astropy pytz joblib '
    'pandas numpy pyarrow optuna hmmlearn numba tqdm pyyaml '
    'alembic cmaes colorlog sqlalchemy threadpoolctl'
)


# =============================================================================
# LOGGING
# =============================================================================

_START = time.time()

def log(msg, level='INFO'):
    elapsed = time.time() - _START
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] [{elapsed:.0f}s] [{level}] {msg}", flush=True)

def log_error(msg):
    log(msg, 'ERROR')

def log_warn(msg):
    log(msg, 'WARN')


# =============================================================================
# STATE MANAGEMENT — persists across machine deaths
# =============================================================================

@dataclass
class PipelineState:
    """Tracks pipeline progress. Serialized to JSON for resume."""
    tf: str
    phase: str = 'init'          # init, cross_gen, training, optimizer, meta, lstm, shap, done
    provider: str = ''
    instance_id: str = ''
    ssh_host: str = ''
    ssh_port: int = 0
    machine_rented_at: str = ''
    total_cost_usd: float = 0.0
    completed_steps: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    machines_used: list = field(default_factory=list)

    def save(self, path: str):
        with open(path + '.tmp', 'w') as f:
            json.dump(self.__dict__, f, indent=2)
        os.replace(path + '.tmp', path)

    @classmethod
    def load(cls, path: str) -> 'PipelineState':
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            state = cls(tf=data.get('tf', '1w'))
            for k, v in data.items():
                if hasattr(state, k):
                    setattr(state, k, v)
            return state
        return None


# =============================================================================
# SSH CLIENT — robust connection with exponential backoff
# =============================================================================

class SSHClient:
    """Wraps SSH/SCP operations with retry logic and heartbeat monitoring."""

    def __init__(self, host: str, port: int, key_path: str, user: str = 'root'):
        self.host = host
        self.port = port
        self.key_path = key_path
        self.user = user
        self._ssh_opts = (
            f'-o StrictHostKeyChecking=no '
            f'-o UserKnownHostsFile=/dev/null '
            f'-o IdentityFile={key_path} '
            f'-o IdentitiesOnly=yes '
            f'-o ConnectTimeout={SSH_CONNECT_TIMEOUT} '
            f'-o ServerAliveInterval=15 '
            f'-o ServerAliveCountMax=3 '
            f'-o LogLevel=ERROR'
        )
        self._heartbeat_thread = None
        self._heartbeat_stop = threading.Event()
        self._dead = False

    @property
    def is_dead(self):
        return self._dead

    def connect(self):
        """Test SSH connection with exponential backoff retry."""
        delay = SSH_BACKOFF_BASE
        for attempt in range(SSH_RETRIES_MAX):
            try:
                result = subprocess.run(
                    f'ssh {self._ssh_opts} -p {self.port} {self.user}@{self.host} "echo OK"',
                    shell=True, capture_output=True, text=True,
                    timeout=SSH_CONNECT_TIMEOUT + 10
                )
                if result.returncode == 0 and 'OK' in result.stdout:
                    log(f"SSH connected to {self.host}:{self.port} (attempt {attempt+1})")
                    return True
            except subprocess.TimeoutExpired:
                pass
            except Exception as e:
                log_warn(f"SSH attempt {attempt+1}/{SSH_RETRIES_MAX} failed: {e}")

            if attempt < SSH_RETRIES_MAX - 1:
                log(f"SSH retry in {delay}s (attempt {attempt+1}/{SSH_RETRIES_MAX})")
                time.sleep(delay)
                delay = min(delay * 2, 120)

        raise ConnectionError(f"SSH connection failed after {SSH_RETRIES_MAX} attempts")

    def run(self, cmd: str, timeout: int = 3600, check: bool = True, env: dict = None) -> subprocess.CompletedProcess:
        """Execute command over SSH. Returns CompletedProcess."""
        env_prefix = ''
        if env:
            env_prefix = ' '.join(f'{k}={v}' for k, v in env.items()) + ' '

        full_cmd = f'ssh {self._ssh_opts} -p {self.port} {self.user}@{self.host} "{env_prefix}{cmd}"'
        result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True, timeout=timeout)

        if check and result.returncode != 0:
            raise RuntimeError(
                f"Remote command failed (exit {result.returncode}): {cmd}\n"
                f"stderr: {result.stderr[:500]}"
            )
        return result

    def run_streaming(self, cmd: str, env: dict = None, logfile: str = None) -> int:
        """Execute command with live stdout streaming. Returns exit code."""
        env_prefix = ''
        if env:
            env_prefix = ' '.join(f'{k}={v}' for k, v in env.items()) + ' '

        full_cmd = f'ssh {self._ssh_opts} -p {self.port} {self.user}@{self.host} "{env_prefix}{cmd}"'
        if logfile:
            full_cmd = f'set -o pipefail && {{ {full_cmd} ; }} 2>&1 | tee -a {logfile}'
            proc = subprocess.run(['bash', '-c', full_cmd])
        else:
            proc = subprocess.run(full_cmd, shell=True)
        return proc.returncode

    def upload(self, local_path: str, remote_path: str, timeout: int = 600):
        """SCP upload a file."""
        cmd = f'scp {self._ssh_opts} -P {self.port} "{local_path}" {self.user}@{self.host}:{remote_path}'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(f"SCP upload failed: {result.stderr[:300]}")
        log(f"Uploaded {local_path} -> {remote_path}")

    def download(self, remote_path: str, local_path: str, timeout: int = 600):
        """SCP download a file."""
        os.makedirs(os.path.dirname(local_path) or '.', exist_ok=True)
        cmd = f'scp {self._ssh_opts} -P {self.port} {self.user}@{self.host}:{remote_path} "{local_path}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(f"SCP download failed: {result.stderr[:300]}")

    def heartbeat(self) -> bool:
        """Single heartbeat check. Returns True if alive."""
        try:
            result = self.run('echo heartbeat', timeout=15, check=False)
            return result.returncode == 0
        except Exception:
            return False

    def start_heartbeat_monitor(self, on_dead_callback=None):
        """Start background thread that monitors machine liveness."""
        self._heartbeat_stop.clear()
        self._dead = False

        def _monitor():
            consecutive_failures = 0
            while not self._heartbeat_stop.is_set():
                if self.heartbeat():
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    log_warn(f"Heartbeat failed ({consecutive_failures}/{HEARTBEAT_FAILURES})")
                    if consecutive_failures >= HEARTBEAT_FAILURES:
                        self._dead = True
                        log_error("Machine declared DEAD — heartbeat failed 3x")
                        if on_dead_callback:
                            on_dead_callback()
                        return
                self._heartbeat_stop.wait(HEARTBEAT_INTERVAL)

        self._heartbeat_thread = threading.Thread(target=_monitor, daemon=True)
        self._heartbeat_thread.start()

    def stop_heartbeat_monitor(self):
        self._heartbeat_stop.set()
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=10)


# =============================================================================
# AUTO-TUNING — detect hardware and compute optimal parameters
# =============================================================================

def autotune(ssh: SSHClient, tf: str, num_features: int = 0) -> dict:
    """
    Detect remote hardware and compute optimal RIGHT_CHUNK, OMP_NUM_THREADS, etc.

    Returns dict of environment variables to set for the pipeline.
    """
    # Detect cores
    result = ssh.run('nproc --all', check=True)
    cores = int(result.stdout.strip())

    # Detect RAM (cgroup-aware)
    result = ssh.run(
        'python3 -c "'
        'import os; '
        'try: ram=os.sysconf(\"SC_PAGE_SIZE\")*os.sysconf(\"SC_PHYS_PAGES\")/(1024**3); '
        'except: ram=0; '
        'print(int(ram))'
        '"',
        check=True
    )
    ram_gb = int(result.stdout.strip())

    # Detect CPU model and GHz
    result = ssh.run('lscpu | grep "Model name" | head -1', check=False)
    cpu_model = result.stdout.strip().split(':')[-1].strip() if result.returncode == 0 else 'unknown'

    result = ssh.run('lscpu | grep "CPU max MHz" | head -1', check=False)
    try:
        max_mhz = float(result.stdout.strip().split(':')[-1].strip())
        ghz = max_mhz / 1000
    except (ValueError, IndexError):
        ghz = 0

    cpu_score = cores * ghz
    log(f"Hardware: {cores} cores, {ram_gb}GB RAM, {cpu_model}")
    log(f"CPU Score: {cpu_score:.0f} (cores={cores} x GHz={ghz:.2f})")

    # Detect disk speed (quick dd test)
    result = ssh.run(
        'dd if=/dev/zero of=/tmp/testfile bs=1M count=256 oflag=direct 2>&1 | tail -1',
        check=False, timeout=30
    )
    disk_speed = result.stdout.strip() if result.returncode == 0 else 'unknown'
    log(f"Disk: {disk_speed}")

    # --- Compute optimal parameters ---

    # OMP_NUM_THREADS: LightGBM trains best with moderate thread count
    # Too many threads = contention. Sweet spot is usually cores * 0.75
    omp_threads = max(1, int(cores * 0.75))
    # Cap at 128 — beyond that, memory bandwidth saturates
    omp_threads = min(omp_threads, 128)

    # NUMBA_NUM_THREADS: for cross gen parallelism
    numba_threads = max(1, min(cores, 96))

    # RIGHT_CHUNK: how many right-side columns to process at once during cross gen
    # Formula: available_ram_bytes / (n_rows * bytes_per_float32 * overhead_factor)
    # But since we don't know n_rows at this point, use TF-based heuristic
    tf_rows = {'1w': 818, '1d': 5727, '4h': 22908, '1h': 91632, '15m': 227000}
    n_rows = tf_rows.get(tf, 5727)

    # Cross gen needs: left_chunk (n_rows x batch) * right_chunk (n_rows x chunk)
    # Each float32 = 4 bytes. Overhead factor 2.5x for intermediate arrays
    # Target: use 65% of RAM for cross gen
    ram_bytes = ram_gb * (1024 ** 3)
    usable_ram = ram_bytes * 0.65
    # Per-chunk memory: n_rows * right_chunk * 4 bytes * 2.5 overhead
    bytes_per_col = n_rows * 4 * 2.5
    right_chunk = int(usable_ram / bytes_per_col) if bytes_per_col > 0 else 500
    # Clamp to sane range
    right_chunk = max(100, min(right_chunk, 2000))
    # 15m is special — always 500 max to prevent OOM
    if tf == '15m':
        right_chunk = min(right_chunk, 500)
    # 1h also needs capping
    if tf == '1h':
        right_chunk = min(right_chunk, 500)

    env = {
        'OMP_NUM_THREADS': str(omp_threads),
        'NUMBA_NUM_THREADS': str(numba_threads),
        'V2_RIGHT_CHUNK': str(right_chunk),
        'V2_BATCH_MAX': str(min(right_chunk, 500)),
        'PYTHONUNBUFFERED': '1',
        'V30_DATA_DIR': REMOTE_V33_DIR,
        'SAVAGE22_DB_DIR': REMOTE_WORKSPACE,
        'SAVAGE22_V1_DIR': REMOTE_WORKSPACE,
    }

    log(f"Auto-tuned: OMP={omp_threads}, NUMBA={numba_threads}, "
        f"RIGHT_CHUNK={right_chunk}, BATCH_MAX={min(right_chunk, 500)}")

    return env


# =============================================================================
# PROVIDER ABSTRACTION
# =============================================================================

@dataclass
class MachineInfo:
    """Info about a rented machine."""
    provider: str
    instance_id: str
    ssh_host: str
    ssh_port: int
    vcpus: int
    ram_gb: int
    hourly_price: float
    gpu_name: str = ''
    cpu_model: str = ''
    cpu_score: float = 0.0


class ProviderBase:
    """Base class for cloud providers."""
    name: str = 'base'

    def search(self, spec: TFSpec, max_results: int = 15) -> list:
        """Search for matching machines. Returns list of offer dicts."""
        raise NotImplementedError

    def rent(self, offer_id: str, disk_gb: int = 50) -> MachineInfo:
        """Rent a specific machine. Returns MachineInfo."""
        raise NotImplementedError

    def get_status(self, instance_id: str) -> str:
        """Get instance status. Returns: running, stopped, terminated, unknown."""
        raise NotImplementedError

    def destroy(self, instance_id: str):
        """Destroy/terminate an instance."""
        raise NotImplementedError

    def is_available(self) -> bool:
        """Check if this provider is configured and usable."""
        raise NotImplementedError


class VastProvider(ProviderBase):
    """vast.ai provider — primary choice for cost-effective GPU machines."""
    name = 'vast'

    def __init__(self):
        self.api_key = os.environ.get('VAST_API_KEY', '')

    def is_available(self) -> bool:
        if not self.api_key:
            # Check if vastai CLI is configured
            try:
                result = subprocess.run(
                    ['vastai', 'show', 'instances', '--raw'],
                    capture_output=True, text=True, timeout=15
                )
                return result.returncode == 0
            except (FileNotFoundError, subprocess.TimeoutExpired):
                return False
        return True

    def _run_vastai(self, args: list, raw: bool = True) -> any:
        """Run vastai CLI command, return parsed JSON."""
        cmd = ['vastai'] + args
        if raw and '--raw' not in args:
            cmd.append('--raw')
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f"vastai failed: {result.stderr[:300]}")
        if raw:
            return json.loads(result.stdout)
        return result.stdout.strip()

    def search(self, spec: TFSpec, max_results: int = 15) -> list:
        """Search vast.ai for matching offers sorted by CPU Score (fastest first)."""
        query = (
            f'cpu_ram>={spec.ram_gb} '
            f'cpu_cores>={spec.min_cores} '
            f'disk_space>={spec.disk_gb} '
            f'cuda_vers>=12.0 '
            f'rentable=true '
            f'num_gpus>=1'
        )

        # Budget filter: max hourly price = budget / max_hours
        max_hourly = spec.budget_usd / max(spec.est_hours[1], 0.5)
        query += f' dph_total<={max_hourly:.2f}'

        offers = self._run_vastai(
            ['search', 'offers', query, '-o', 'cpu_cores_effective-', '--limit', str(max_results)]
        )
        if not isinstance(offers, list):
            return []

        # Enrich with CPU Score and sort by it (fastest first)
        for o in offers:
            cores = o.get('cpu_cores', 0)
            ghz = o.get('cpu_ghz', 0) or 0
            o['cpu_score'] = cores * ghz
            o['ram_gb'] = round((o.get('cpu_ram', 0) or 0) / 1024, 0)

        offers.sort(key=lambda o: -o['cpu_score'])
        return offers

    def display_offers(self, offers: list):
        """Print formatted table of offers."""
        if not offers:
            log_warn("No matching offers found on vast.ai")
            return

        print()
        print(f"{'#':>3}  {'ID':>10}  {'GPU':20s}  {'vCPUs':>5}  {'GHz':>5}  {'CPU Score':>9}  "
              f"{'RAM GB':>6}  {'Driver':>10}  {'$/hr':>6}")
        print("-" * 105)

        for i, o in enumerate(offers):
            gpu_name = o.get('gpu_name', '?')
            num_gpus = o.get('num_gpus', 1)
            gpu_label = f"{num_gpus}x {gpu_name}" if num_gpus > 1 else gpu_name
            if len(gpu_label) > 20:
                gpu_label = gpu_label[:19] + '.'

            print(f"{i+1:>3}  {o.get('id','?'):>10}  {gpu_label:20s}  "
                  f"{o.get('cpu_cores',0):>5}  {o.get('cpu_ghz',0) or 0:>5.2f}  "
                  f"{o.get('cpu_score',0):>9.1f}  {o.get('ram_gb',0):>6.0f}  "
                  f"{o.get('driver_version','?'):>10}  "
                  f"${o.get('dph_total',0):>5.2f}")
        print()

    def rent(self, offer_id: str, disk_gb: int = 50) -> MachineInfo:
        """Rent a specific vast.ai offer."""
        create_output = self._run_vastai(
            ['create', 'instance', str(offer_id),
             '--image', BASE_IMAGE,
             '--disk', str(disk_gb),
             '--ssh'],
            raw=False
        )

        # Parse instance ID from response
        match = re.search(r"'new_contract':\s*(\d+)", create_output)
        if not match:
            match = re.search(r'"new_contract":\s*(\d+)', create_output)
        if not match:
            raise RuntimeError(f"Could not parse instance ID from: {create_output}")

        instance_id = match.group(1)
        log(f"Created vast.ai instance {instance_id}")

        # Wait for running + SSH info
        ssh_host, ssh_port = self._wait_for_running(int(instance_id))

        return MachineInfo(
            provider='vast',
            instance_id=instance_id,
            ssh_host=ssh_host,
            ssh_port=ssh_port,
            vcpus=0,   # filled later by autotune
            ram_gb=0,
            hourly_price=0,
        )

    def _wait_for_running(self, instance_id: int, timeout: int = 600) -> tuple:
        """Wait for vast.ai instance to reach running state. Returns (host, port)."""
        log(f"Waiting for instance {instance_id} to start...")
        start = time.time()
        while time.time() - start < timeout:
            instances = self._run_vastai(['show', 'instances'])
            for inst in (instances if isinstance(instances, list) else []):
                if inst.get('id') == instance_id:
                    status = inst.get('actual_status', '')
                    if status == 'running':
                        ssh_host = inst.get('ssh_host', '')
                        ssh_port = inst.get('ssh_port', '')
                        if ssh_host and ssh_port:
                            log(f"Instance {instance_id} running at {ssh_host}:{ssh_port}")
                            return ssh_host, int(ssh_port)
                    elif status in ('exited', 'error', 'offline'):
                        raise RuntimeError(f"Instance {instance_id} entered state '{status}'")
                    log(f"  Status: {status} ({time.time()-start:.0f}s)")
            time.sleep(10)

        raise TimeoutError(f"Instance {instance_id} did not start within {timeout}s")

    def get_status(self, instance_id: str) -> str:
        try:
            instances = self._run_vastai(['show', 'instances'])
            for inst in (instances if isinstance(instances, list) else []):
                if str(inst.get('id')) == str(instance_id):
                    status = inst.get('actual_status', '')
                    if status == 'running':
                        return 'running'
                    elif status in ('exited', 'error', 'offline', 'stopped'):
                        return 'terminated'
                    return 'unknown'
        except Exception:
            pass
        return 'unknown'

    def destroy(self, instance_id: str):
        try:
            self._run_vastai(['destroy', 'instance', str(instance_id)], raw=False)
            log(f"Destroyed vast.ai instance {instance_id}")
        except Exception as e:
            log_error(f"Failed to destroy instance {instance_id}: {e}")


class GCPProvider(ProviderBase):
    """GCP c3d-highmem-360 spot instances — 360 vCPUs, 2880 GiB RAM."""
    name = 'gcp'

    def __init__(self):
        self.project = os.environ.get('GCP_PROJECT', '')
        self.zone = os.environ.get('GCP_ZONE', 'us-central1-a')

    def is_available(self) -> bool:
        if not self.project:
            return False
        try:
            result = subprocess.run(
                ['gcloud', 'compute', 'instances', 'list', '--project', self.project,
                 '--format=json', '--limit=1'],
                capture_output=True, text=True, timeout=30
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def search(self, spec: TFSpec, max_results: int = 15) -> list:
        """GCP has fixed machine types. Return c3d-highmem-360 if it meets requirements."""
        # c3d-highmem-360: 360 vCPUs, 2880 GiB RAM
        if spec.ram_gb <= 2880 and spec.min_cores <= 360:
            # Spot price ~$3.50-7.85/hr depending on region
            spot_price = 5.50  # conservative estimate
            return [{
                'id': 'c3d-highmem-360',
                'machine_type': 'c3d-highmem-360',
                'vcpus': 360,
                'ram_gb': 2880,
                'hourly_price': spot_price,
                'provider': 'gcp',
                'cpu_score': 360 * 3.7,  # ~3.7 GHz Sapphire Rapids
            }]
        return []

    def rent(self, offer_id: str = 'c3d-highmem-360', disk_gb: int = 200) -> MachineInfo:
        """Create a GCP spot VM."""
        name = f"train-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        startup_script = f"""#!/bin/bash
apt-get update -qq && apt-get install -y -qq python3-pip openssh-server
pip3 install -q {PIP_DEPS}
"""
        # Create the instance
        cmd = [
            'gcloud', 'compute', 'instances', 'create', name,
            '--project', self.project,
            '--zone', self.zone,
            '--machine-type', 'c3d-highmem-360',
            '--provisioning-model', 'SPOT',
            '--instance-termination-action', 'STOP',
            '--boot-disk-size', f'{disk_gb}GB',
            '--boot-disk-type', 'pd-ssd',
            '--image-family', 'debian-12',
            '--image-project', 'debian-cloud',
            f'--metadata=startup-script={startup_script}',
            '--format=json',
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"GCP instance creation failed: {result.stderr[:500]}")

        data = json.loads(result.stdout)
        instance = data[0] if isinstance(data, list) else data

        # Get external IP
        nat_ip = None
        for nic in instance.get('networkInterfaces', []):
            for ac in nic.get('accessConfigs', []):
                if ac.get('natIP'):
                    nat_ip = ac['natIP']
                    break

        if not nat_ip:
            raise RuntimeError("No external IP assigned to GCP instance")

        log(f"Created GCP instance {name} at {nat_ip}")

        return MachineInfo(
            provider='gcp',
            instance_id=name,
            ssh_host=nat_ip,
            ssh_port=22,
            vcpus=360,
            ram_gb=2880,
            hourly_price=5.50,
        )

    def get_status(self, instance_id: str) -> str:
        try:
            result = subprocess.run(
                ['gcloud', 'compute', 'instances', 'describe', instance_id,
                 '--project', self.project, '--zone', self.zone, '--format=json'],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                status = data.get('status', '')
                if status == 'RUNNING':
                    return 'running'
                elif status in ('TERMINATED', 'STOPPED', 'STOPPING'):
                    return 'terminated'
            return 'unknown'
        except Exception:
            return 'unknown'

    def destroy(self, instance_id: str):
        try:
            subprocess.run(
                ['gcloud', 'compute', 'instances', 'delete', instance_id,
                 '--project', self.project, '--zone', self.zone, '--quiet'],
                capture_output=True, text=True, timeout=60
            )
            log(f"Destroyed GCP instance {instance_id}")
        except Exception as e:
            log_error(f"Failed to destroy GCP instance {instance_id}: {e}")


# =============================================================================
# PROVIDER REGISTRY & FAILOVER
# =============================================================================

def get_providers() -> list:
    """Return list of available providers in priority order."""
    providers = []

    vast = VastProvider()
    if vast.is_available():
        providers.append(vast)

    gcp = GCPProvider()
    if gcp.is_available():
        providers.append(gcp)

    return providers


def search_all_providers(spec: TFSpec, providers: list) -> dict:
    """Search all providers and return combined results."""
    results = {}
    for p in providers:
        try:
            offers = p.search(spec)
            if offers:
                results[p.name] = {'provider': p, 'offers': offers}
                log(f"  {p.name}: {len(offers)} offers found")
            else:
                log(f"  {p.name}: no matching offers")
        except Exception as e:
            log_warn(f"  {p.name}: search failed — {e}")
    return results


def rent_with_failover(spec: TFSpec, providers: list, offer_choice: dict = None) -> tuple:
    """
    Rent a machine with failover across providers.
    If offer_choice is provided, use that specific offer.
    Returns (provider, MachineInfo).
    """
    if offer_choice:
        provider = offer_choice['provider']
        offer_id = offer_choice['offer_id']
        log(f"Renting {provider.name} offer {offer_id}...")
        machine = provider.rent(str(offer_id), spec.disk_gb)
        return provider, machine

    errors = []
    for provider in providers:
        try:
            offers = provider.search(spec)
            if not offers:
                errors.append((provider.name, "no matching offers"))
                continue

            # Pick the highest CPU Score offer
            best = offers[0]
            offer_id = best.get('id', '')
            log(f"Auto-selecting {provider.name} offer {offer_id} "
                f"(CPU Score: {best.get('cpu_score', 0):.0f}, "
                f"${best.get('dph_total', best.get('hourly_price', 0)):.2f}/hr)")

            machine = provider.rent(str(offer_id), spec.disk_gb)
            return provider, machine

        except Exception as e:
            errors.append((provider.name, str(e)))
            log_warn(f"  {provider.name} failed: {e}")

    raise RuntimeError(f"All providers failed: {errors}")


# =============================================================================
# DEPLOYMENT — upload code, DBs, verify
# =============================================================================

def deploy(ssh: SSHClient, tf: str, upload_tar: str, db_tar: str = None):
    """Deploy code and data to remote machine."""

    # 1. Install dependencies
    log("Installing dependencies...")
    exit_code = ssh.run_streaming(
        f'pip install -q {PIP_DEPS} 2>&1 | tail -5'
    )
    if exit_code != 0:
        raise RuntimeError("pip install failed")

    # 2. Verify all imports
    log("Verifying Python imports...")
    ssh.run(
        'python3 -c "import pandas,numpy,scipy,sklearn,lightgbm,ephem,astropy,'
        'pyarrow,optuna,numba,hmmlearn,yaml,tqdm; print(\'ALL_IMPORTS_OK\')"',
        check=True
    )

    # 3. Upload code tar
    if upload_tar and os.path.exists(upload_tar):
        log(f"Uploading code tar ({os.path.getsize(upload_tar)/(1024*1024):.1f} MB)...")
        ssh.upload(upload_tar, f'{REMOTE_WORKSPACE}/v33_upload.tar.gz')
        ssh.run(f'cd {REMOTE_WORKSPACE} && tar xzf v33_upload.tar.gz', check=True)
    else:
        log_warn(f"Code tar not found at {upload_tar} — skipping upload")

    # 4. Upload DB tar (if provided)
    if db_tar and os.path.exists(db_tar):
        log(f"Uploading DB tar ({os.path.getsize(db_tar)/(1024*1024):.1f} MB)...")
        ssh.upload(db_tar, f'{REMOTE_WORKSPACE}/v33_dbs.tar.gz', timeout=1800)
        ssh.run(f'cd {REMOTE_WORKSPACE} && tar xzf v33_dbs.tar.gz', check=True)

    # 5. Create symlinks
    log("Creating symlinks...")
    ssh.run(
        f'cd {REMOTE_WORKSPACE} && '
        f'for f in *.db kp_history_gfz.txt; do '
        f'  [ -f "$f" ] && ln -sf {REMOTE_WORKSPACE}/"$f" {REMOTE_V33_DIR}/"$f" 2>/dev/null; '
        f'done && '
        f'[ -f astrology_engine.py ] && ln -sf {REMOTE_WORKSPACE}/astrology_engine.py {REMOTE_V33_DIR}/ 2>/dev/null; '
        f'ln -sf {REMOTE_V33_DIR}/btc_prices.db {REMOTE_WORKSPACE}/btc_prices.db 2>/dev/null; '
        f'true',
        check=False
    )

    # 6. Verify DB count
    result = ssh.run(f'ls {REMOTE_WORKSPACE}/*.db 2>/dev/null | wc -l', check=False)
    db_count = int(result.stdout.strip()) if result.returncode == 0 else 0
    if db_count < 10:
        log_warn(f"Only {db_count} DBs found — expected >= 16. Some features may be missing.")
    else:
        log(f"DB count: {db_count}")

    # 7. Upload checkpoints if resuming
    checkpoint_dir = os.environ.get('CHECKPOINT_DIR', './checkpoints')
    tf_checkpoint = os.path.join(checkpoint_dir, tf)
    if os.path.isdir(tf_checkpoint):
        log(f"Uploading checkpoint artifacts from {tf_checkpoint}...")
        for f in os.listdir(tf_checkpoint):
            local = os.path.join(tf_checkpoint, f)
            if os.path.isfile(local):
                ssh.upload(local, f'{REMOTE_V33_DIR}/{f}')
        log(f"Uploaded {len(os.listdir(tf_checkpoint))} checkpoint files")


# =============================================================================
# ARTIFACT DOWNLOAD — checkpoint after critical steps
# =============================================================================

def download_artifacts(ssh: SSHClient, tf: str, phase: str, local_dir: str):
    """Download artifacts for a given phase."""
    os.makedirs(local_dir, exist_ok=True)

    patterns = CRITICAL_ARTIFACTS.get(phase, [])
    downloaded = 0
    for pattern in patterns:
        filename = pattern.format(tf=tf)
        remote_path = f'{REMOTE_V33_DIR}/{filename}'
        local_path = os.path.join(local_dir, filename)

        # Check if file exists on remote
        result = ssh.run(f'test -f {remote_path} && stat -c%s {remote_path}', check=False)
        if result.returncode == 0:
            size_bytes = int(result.stdout.strip())
            if size_bytes > 0:
                try:
                    ssh.download(remote_path, local_path)
                    size_mb = size_bytes / (1024 * 1024)
                    log(f"  Downloaded {filename} ({size_mb:.1f} MB)")
                    downloaded += 1
                except Exception as e:
                    log_warn(f"  Failed to download {filename}: {e}")

    log(f"Downloaded {downloaded}/{len(patterns)} artifacts for phase '{phase}'")
    return downloaded


def download_all_artifacts(ssh: SSHClient, tf: str, local_dir: str):
    """Download ALL artifacts (for final collection or before machine death)."""
    for phase in CRITICAL_ARTIFACTS:
        download_artifacts(ssh, tf, phase, local_dir)


# =============================================================================
# PIPELINE EXECUTION
# =============================================================================

def run_pipeline(ssh: SSHClient, tf: str, env: dict, state: PipelineState,
                 checkpoint_dir: str):
    """
    Run the full cloud pipeline on a remote machine.
    Checkpoints artifacts after each critical step.
    """
    tf_dir = os.path.join(checkpoint_dir, tf)
    os.makedirs(tf_dir, exist_ok=True)

    # The pipeline command — delegates to cloud_run_tf.py which handles all steps
    env_str = ' '.join(f'export {k}={v} &&' for k, v in env.items())
    pipeline_cmd = (
        f'cd {REMOTE_V33_DIR} && '
        f'{env_str} '
        f'python3 -u cloud_run_tf.py --symbol BTC --tf {tf}'
    )

    log(f"Starting pipeline for {tf}...")
    log(f"Remote command: {pipeline_cmd[:200]}...")

    # Run pipeline with streaming output
    exit_code = ssh.run_streaming(
        pipeline_cmd,
        logfile=f'{REMOTE_WORKSPACE}/{tf}_pipeline.log'
    )

    if exit_code != 0:
        log_error(f"Pipeline exited with code {exit_code}")
        # Download whatever we can before failing
        log("Downloading partial artifacts...")
        download_all_artifacts(ssh, tf, tf_dir)
        state.errors.append(f"Pipeline exit code {exit_code}")
        state.save(os.path.join(checkpoint_dir, f'state_{tf}.json'))
        raise RuntimeError(f"Pipeline failed with exit code {exit_code}")

    # Pipeline completed — download everything
    log("Pipeline completed! Downloading all artifacts...")
    download_all_artifacts(ssh, tf, tf_dir)

    # Also download the log
    try:
        ssh.download(
            f'{REMOTE_WORKSPACE}/{tf}_pipeline.log',
            os.path.join(tf_dir, f'{tf}_pipeline.log')
        )
    except Exception:
        pass

    state.phase = 'done'
    state.completed_steps.append('full_pipeline')
    state.save(os.path.join(checkpoint_dir, f'state_{tf}.json'))

    # Check for DONE marker
    result = ssh.run(f'test -f {REMOTE_WORKSPACE}/DONE_{tf}', check=False)
    if result.returncode == 0:
        log(f"DONE marker found — {tf} pipeline fully complete!")
    else:
        log_warn(f"No DONE marker — pipeline may not have completed all steps")


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Automated cloud training orchestrator for v3.3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python v3.3/cloud_auto.py --tf 1w                    # Full automated run
  python v3.3/cloud_auto.py --tf 1d --search-only       # Just show available machines
  python v3.3/cloud_auto.py --tf 4h --provider vast      # Force vast.ai
  python v3.3/cloud_auto.py --tf 1h --resume             # Resume from checkpoint
  python v3.3/cloud_auto.py --tf 15m --dry-run           # Show plan without executing
        """
    )
    parser.add_argument('--tf', required=True, choices=['1w', '1d', '4h', '1h', '15m'],
                        help='Timeframe to train')
    parser.add_argument('--search-only', action='store_true',
                        help='Only search and display available machines')
    parser.add_argument('--provider', choices=['vast', 'gcp'],
                        help='Force a specific provider')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last checkpoint')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show plan without executing')
    parser.add_argument('--auto-select', action='store_true',
                        help='Auto-select best machine (no prompt)')
    parser.add_argument('--upload-tar', default=os.environ.get('UPLOAD_TAR', '/tmp/v33_upload.tar.gz'),
                        help='Path to code tar')
    parser.add_argument('--db-tar', default=os.environ.get('DB_TAR', ''),
                        help='Path to DB tar')
    parser.add_argument('--checkpoint-dir', default=os.environ.get('CHECKPOINT_DIR', './checkpoints'),
                        help='Local checkpoint directory')
    parser.add_argument('--ssh-key', default=os.path.expanduser('~/.ssh/vast_key'),
                        help='SSH private key path')
    parser.add_argument('--max-price', type=float,
                        help='Override max $/hr filter')
    args = parser.parse_args()

    tf = Timeframe(args.tf)
    spec = TF_SPECS[tf]

    log(f"Cloud Auto Orchestrator — TF: {args.tf}")
    log(f"Requirements: {spec.ram_gb}GB RAM, {spec.min_cores}+ cores, "
        f"budget ${spec.budget_usd}, est {spec.est_hours[0]}-{spec.est_hours[1]} hrs")

    # --- Load or create state ---
    state_path = os.path.join(args.checkpoint_dir, f'state_{args.tf}.json')
    if args.resume:
        state = PipelineState.load(state_path)
        if state:
            log(f"Resuming from phase '{state.phase}', "
                f"completed: {state.completed_steps}")
        else:
            log("No checkpoint found, starting fresh")
            state = PipelineState(tf=args.tf)
    else:
        state = PipelineState(tf=args.tf)

    # --- Get available providers ---
    providers = get_providers()
    if args.provider:
        providers = [p for p in providers if p.name == args.provider]
    if not providers:
        log_error("No providers available. Set VAST_API_KEY or GCP_PROJECT.")
        sys.exit(1)

    log(f"Available providers: {[p.name for p in providers]}")

    # --- Search phase ---
    log("Searching for machines...")
    all_results = search_all_providers(spec, providers)

    if not all_results:
        log_error("No matching machines found on any provider!")
        log(f"Requirements: {spec.ram_gb}GB RAM, {spec.min_cores}+ cores")
        if spec.ram_gb >= 1536:
            log("TIP: For 1.5TB+ RAM, try GCP c3d-highmem-360 (set GCP_PROJECT env var)")
            log("TIP: vast.ai rarely has 1.5TB+ RAM machines — check manually")
        sys.exit(1)

    # Display results
    for pname, data in all_results.items():
        provider = data['provider']
        log(f"\n--- {pname.upper()} offers ---")
        if hasattr(provider, 'display_offers'):
            provider.display_offers(data['offers'])
        else:
            for o in data['offers']:
                log(f"  {o.get('id')}: {o.get('vcpus', '?')} vCPUs, "
                    f"{o.get('ram_gb', '?')}GB RAM, "
                    f"${o.get('hourly_price', '?')}/hr")

    if args.search_only:
        log("(--search-only mode, exiting)")
        return

    if args.dry_run:
        log("(--dry-run mode)")
        best_provider = list(all_results.values())[0]['provider']
        best_offer = list(all_results.values())[0]['offers'][0]
        log(f"Would rent: {best_provider.name} offer {best_offer.get('id')} "
            f"(CPU Score: {best_offer.get('cpu_score', 0):.0f})")
        log(f"Would upload: {args.upload_tar}")
        log(f"Would run: cloud_run_tf.py --symbol BTC --tf {args.tf}")
        log(f"Would checkpoint to: {args.checkpoint_dir}/{args.tf}/")
        return

    # --- Machine selection ---
    if args.auto_select:
        # Pick highest CPU Score across all providers
        best_score = -1
        offer_choice = None
        for pname, data in all_results.items():
            for o in data['offers']:
                score = o.get('cpu_score', 0)
                if score > best_score:
                    best_score = score
                    offer_choice = {
                        'provider': data['provider'],
                        'offer_id': o.get('id'),
                    }
        log(f"Auto-selected best CPU Score: {best_score:.0f}")
    else:
        # Interactive selection
        # Flatten all offers with provider info
        flat_offers = []
        for pname, data in all_results.items():
            for o in data['offers']:
                o['_provider'] = data['provider']
                o['_provider_name'] = pname
                flat_offers.append(o)
        flat_offers.sort(key=lambda o: -o.get('cpu_score', 0))

        print(f"\nAll offers sorted by CPU Score:")
        print(f"{'#':>3}  {'Provider':>8}  {'ID':>12}  {'vCPUs':>5}  {'CPU Score':>9}  "
              f"{'RAM GB':>6}  {'$/hr':>6}")
        print("-" * 65)
        for i, o in enumerate(flat_offers):
            print(f"{i+1:>3}  {o['_provider_name']:>8}  {str(o.get('id','')):>12}  "
                  f"{o.get('cpu_cores', o.get('vcpus', 0)):>5}  "
                  f"{o.get('cpu_score',0):>9.1f}  "
                  f"{o.get('ram_gb',0):>6.0f}  "
                  f"${o.get('dph_total', o.get('hourly_price', 0)):>5.2f}")
        print()

        while True:
            choice = input(f"Pick offer # (1-{len(flat_offers)}), or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                log("Aborted by user.")
                return
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(flat_offers):
                    selected = flat_offers[idx]
                    offer_choice = {
                        'provider': selected['_provider'],
                        'offer_id': selected.get('id'),
                    }
                    break
            except ValueError:
                pass
            print(f"Invalid choice. Enter 1-{len(flat_offers)} or 'q'.")

    # --- Rent machine ---
    provider = offer_choice['provider']
    log(f"Renting {provider.name} offer {offer_choice['offer_id']}...")

    confirm = input(f"Rent this machine? [y/N]: ").strip().lower()
    if confirm != 'y':
        log("Aborted by user.")
        return

    provider_obj, machine = rent_with_failover(spec, providers, offer_choice)

    state.provider = provider_obj.name
    state.instance_id = machine.instance_id
    state.ssh_host = machine.ssh_host
    state.ssh_port = machine.ssh_port
    state.machine_rented_at = datetime.now().isoformat()
    state.machines_used.append({
        'provider': provider_obj.name,
        'instance_id': machine.instance_id,
        'rented_at': state.machine_rented_at,
    })
    state.save(state_path)

    # --- Connect SSH ---
    ssh = SSHClient(machine.ssh_host, machine.ssh_port, args.ssh_key)
    try:
        ssh.connect()
    except ConnectionError as e:
        log_error(f"SSH connection failed: {e}")
        log("Machine may need more time to boot. Try --resume later.")
        sys.exit(1)

    # --- Start heartbeat monitor ---
    machine_died = threading.Event()

    def on_machine_death():
        machine_died.set()
        log_error("MACHINE DEATH DETECTED — downloading artifacts...")
        try:
            download_all_artifacts(ssh, args.tf,
                                   os.path.join(args.checkpoint_dir, args.tf))
        except Exception:
            pass
        state.errors.append(f"Machine died at {datetime.now().isoformat()}")
        state.save(state_path)

    ssh.start_heartbeat_monitor(on_dead_callback=on_machine_death)

    try:
        # --- Auto-tune ---
        env = autotune(ssh, args.tf)

        # --- Deploy ---
        deploy(ssh, args.tf, args.upload_tar, args.db_tar if args.db_tar else None)

        # --- Run pipeline ---
        run_pipeline(ssh, args.tf, env, state, args.checkpoint_dir)

        # --- Success ---
        log("=" * 60)
        log(f"TRAINING COMPLETE: {args.tf}")
        log(f"Artifacts saved to: {os.path.join(args.checkpoint_dir, args.tf)}")
        log("=" * 60)

    except Exception as e:
        log_error(f"Pipeline error: {e}")
        # Try to download whatever we can
        if not machine_died.is_set():
            log("Downloading partial artifacts...")
            try:
                download_all_artifacts(ssh, args.tf,
                                       os.path.join(args.checkpoint_dir, args.tf))
            except Exception:
                pass

    finally:
        ssh.stop_heartbeat_monitor()

        # --- Destroy machine ---
        if not machine_died.is_set():
            print()
            destroy = input(f"Destroy {provider_obj.name} instance {machine.instance_id}? [y/N]: ").strip().lower()
            if destroy == 'y':
                provider_obj.destroy(machine.instance_id)
            else:
                log(f"Instance {machine.instance_id} LEFT RUNNING — remember to destroy it!")
                log(f"  vastai destroy instance {machine.instance_id}")

    # Final state
    state.save(state_path)
    log("Done.")


if __name__ == '__main__':
    main()
