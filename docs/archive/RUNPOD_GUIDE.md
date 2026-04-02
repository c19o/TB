# RunPod GPU Cloud - Complete Guide for ML Training

Last updated: 2026-03-18

---

## Table of Contents

1. [Overview](#overview)
2. [Setup and Installation](#setup-and-installation)
3. [API Key Configuration](#api-key-configuration)
4. [RTX 5090 Pricing and Availability](#rtx-5090-pricing-and-availability)
5. [Creating Pods with Python SDK](#creating-pods-with-python-sdk)
6. [Creating Pods with REST API](#creating-pods-with-rest-api)
7. [Managing Pod Lifecycle](#managing-pod-lifecycle)
8. [Connecting to Pods](#connecting-to-pods)
9. [Uploading Data to Pods](#uploading-data-to-pods)
10. [Running Training Jobs](#running-training-jobs)
11. [Downloading Results](#downloading-results)
12. [Storage Options](#storage-options)
13. [Serverless Alternative](#serverless-alternative)
14. [Full Workflow Example](#full-workflow-example)
15. [GPU Pricing Comparison](#gpu-pricing-comparison)
16. [Gotchas and Best Practices](#gotchas-and-best-practices)

---

## Overview

RunPod is a GPU cloud platform for ML/AI workloads. Three deployment models:

- **Pods** - Dedicated GPU instances (what we want for training)
- **Serverless** - Pay-per-second, auto-scaling (better for inference)
- **Flash** (Beta) - Run Python functions on remote GPUs from local terminal

For training jobs, Pods are the right choice. You get full SSH access, persistent
storage, and can run arbitrary code.

---

## Setup and Installation

```bash
pip install runpod
```

Verify installation:
```bash
python -c "import runpod; print(runpod.__version__)"
```

Install from GitHub for latest:
```bash
pip install git+https://github.com/runpod/runpod-python.git
```

Also install the CLI tool (pre-installed on pods, needed locally for file transfer):
```bash
# Download runpodctl from https://github.com/runpod/runpodctl/releases
# Or on Linux:
# wget https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-amd64
# chmod +x runpodctl-linux-amd64 && mv runpodctl-linux-amd64 /usr/local/bin/runpodctl
```

Requires Python 3.8+.

---

## API Key Configuration

1. Go to https://www.console.runpod.io/ -> Settings -> API Keys
2. Click "Create API Key"
3. Set permission level to "Read/Write" for full access
4. Save the key immediately - RunPod does NOT store it after creation

In your .env file:
```
RUNPOD_API_KEY=your_key_here
```

In Python:
```python
import os
import runpod
from dotenv import load_dotenv

load_dotenv()
runpod.api_key = os.getenv("RUNPOD_API_KEY")
```

NEVER hardcode API keys in source code.

---

## RTX 5090 Pricing and Availability

The RTX 5090 is available on RunPod with 32 GB GDDR7 VRAM.

| Metric | Value |
|--------|-------|
| VRAM | 32 GB GDDR7 |
| Memory Bandwidth | 1.79 TB/s |
| FP16 Tensor Performance | 0.42 PFLOPS |
| System RAM | 35 GB |
| vCPUs | 9 |

### Pricing

| Cloud Type | On-Demand | 6-Month Commit | 1-Year Commit |
|------------|-----------|----------------|---------------|
| Community Cloud | $0.69/hr | -- | -- |
| Secure Cloud | $0.89/hr | $0.774/hr | $0.757/hr |

### Cost Estimates

| Duration | Community | Secure |
|----------|-----------|--------|
| 1 hour | $0.69 | $0.89 |
| 10 hours | $6.90 | $8.90 |
| 100 hours | $69.00 | $89.00 |
| 24/7 month | $503.70 | $649.80 |

### Availability Warning

RTX 5090 availability is NOT guaranteed. Community reports indicate some regions
(e.g., IS-1) have been "nearly always unavailable" for weeks at a time. Tips:

- Check multiple regions before deploying
- Use `gpuTypePriority: "availability"` to let RunPod pick the best region
- Have fallback GPU types ready (RTX 4090 at $0.34/hr is a good backup)
- Consider Secure Cloud for more reliable availability

The GPU type string for the RTX 5090 is: `"NVIDIA GeForce RTX 5090"`

---

## Creating Pods with Python SDK

### Minimal Example

```python
import os
import runpod
from dotenv import load_dotenv

load_dotenv()
runpod.api_key = os.getenv("RUNPOD_API_KEY")

pod = runpod.create_pod(
    name="savage22-training",
    image_name="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
    gpu_type_id="NVIDIA GeForce RTX 5090",
)
print(f"Pod ID: {pod['id']}")
```

### Full Example with All Options

```python
import os
import runpod
from dotenv import load_dotenv

load_dotenv()
runpod.api_key = os.getenv("RUNPOD_API_KEY")

pod = runpod.create_pod(
    name="savage22-training",
    image_name="runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
    gpu_type_id="NVIDIA GeForce RTX 5090",
    volume_in_gb=50,           # Persistent storage (survives restarts)
    container_disk_in_gb=50,   # Temporary storage (wiped on restart)
    # template_id="xxx",       # Optional: use a saved template
)

print(f"Pod created: {pod['id']}")
print(f"Status: {pod.get('desiredStatus')}")
```

### With Fallback GPUs

If RTX 5090 is not available, the SDK only takes one gpu_type_id. Use the REST
API for fallback GPU lists (see next section).

---

## Creating Pods with REST API

The REST API gives more control, including fallback GPU types.

### Endpoint

```
POST https://rest.runpod.io/v1/pods
Authorization: Bearer YOUR_API_KEY
Content-Type: application/json
```

### Full Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| name | string | "my pod" | Pod name (max 191 chars) |
| imageName | string | required | Docker image to run |
| computeType | string | "GPU" | "GPU" or "CPU" |
| cloudType | string | "SECURE" | "SECURE" or "COMMUNITY" |
| gpuCount | integer | 1 | Number of GPUs |
| gpuTypeIds | array | -- | GPU types in priority order |
| gpuTypePriority | string | "availability" | "availability" or "custom" |
| minRAMPerGPU | integer | 8 | Min GB RAM per GPU |
| minVCPUPerGPU | integer | 2 | Min vCPUs per GPU |
| containerDiskInGb | integer | 50 | Temp disk (wiped on restart) |
| volumeInGb | integer | 20 | Persistent volume |
| volumeMountPath | string | "/workspace" | Where volume mounts |
| networkVolumeId | string | -- | Attach existing network volume |
| ports | array | ["8888/http","22/tcp"] | Exposed ports |
| env | object | {} | Environment variables |
| dockerStartCmd | array | [] | Override start command |
| interruptible | boolean | false | Spot pricing (cheaper, can be interrupted) |
| supportPublicIp | boolean | -- | Request public IP |
| dataCenterIds | array | [all] | Preferred data centers |

### Python Example with REST API

```python
import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("RUNPOD_API_KEY")

response = requests.post(
    "https://rest.runpod.io/v1/pods",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    },
    json={
        "name": "savage22-training",
        "imageName": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
        "computeType": "GPU",
        "cloudType": "COMMUNITY",  # Cheaper ($0.69/hr vs $0.89/hr)
        "gpuCount": 1,
        "gpuTypeIds": [
            "NVIDIA GeForce RTX 5090",  # First choice
            "NVIDIA GeForce RTX 4090",  # Fallback
        ],
        "gpuTypePriority": "custom",
        "containerDiskInGb": 50,
        "volumeInGb": 50,
        "volumeMountPath": "/workspace",
        "ports": ["8888/http", "22/tcp"],
        "env": {
            "CUDA_VISIBLE_DEVICES": "0",
        },
    },
)

pod = response.json()
print(f"Pod ID: {pod['id']}")
print(f"GPU: {pod['gpu']['displayName']}")
print(f"Cost: ${pod['costPerHr']}/hr")
print(f"Public IP: {pod.get('publicIp')}")
print(f"SSH Port: {pod.get('portMappings', {}).get('22')}")
```

### Example Response

```json
{
  "id": "xedezhzb9la3ye",
  "name": "savage22-training",
  "desiredStatus": "RUNNING",
  "costPerHr": 0.69,
  "gpu": {
    "id": "NVIDIA GeForce RTX 5090",
    "count": 1,
    "displayName": "NVIDIA GeForce RTX 5090"
  },
  "memoryInGb": 35,
  "vcpuCount": 9,
  "containerDiskInGb": 50,
  "volumeInGb": 50,
  "volumeMountPath": "/workspace",
  "publicIp": "100.65.0.119",
  "portMappings": {
    "22": 10341,
    "8888": 10342
  }
}
```

---

## Managing Pod Lifecycle

### With Python SDK

```python
import os
import runpod
from dotenv import load_dotenv

load_dotenv()
runpod.api_key = os.getenv("RUNPOD_API_KEY")

# List all pods
pods = runpod.get_pods()
for p in pods:
    print(f"{p['id']}: {p['name']} - {p.get('desiredStatus')}")

# Get specific pod info
pod = runpod.get_pod("POD_ID")

# Stop pod (releases GPU, keeps /workspace data, still charges for volume storage)
runpod.stop_pod("POD_ID")

# Resume a stopped pod
runpod.resume_pod("POD_ID")

# Terminate pod (destroys everything, stops all charges)
runpod.terminate_pod("POD_ID")
```

### With CLI

```bash
# Set API key
runpodctl config --apiKey YOUR_API_KEY

# List pods
runpodctl get pod

# Start/stop/terminate
runpodctl start pod POD_ID
runpodctl stop pod POD_ID
runpodctl remove pod POD_ID
```

### With REST API

```bash
# List pods
curl https://rest.runpod.io/v1/pods \
  -H "Authorization: Bearer YOUR_API_KEY"

# Get specific pod
curl https://rest.runpod.io/v1/pods/POD_ID \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### Important: Stop vs Terminate

- **Stop**: Releases GPU, preserves /workspace data. You still pay for volume storage.
- **Terminate**: Destroys everything. All charges stop. Data is GONE.

---

## Connecting to Pods

### SSH (Recommended for Training)

1. Add your SSH public key to RunPod console (Settings -> SSH Keys)
2. Get connection details from pod info (IP + mapped port for port 22)

```bash
# Connect via SSH
ssh root@POD_PUBLIC_IP -p MAPPED_PORT -i ~/.ssh/id_ed25519
```

### Web Terminal

Available directly in the RunPod console. Good for quick commands but sessions
do not persist.

### JupyterLab

Available at: `https://POD_ID-8888.proxy.runpod.net`
Pre-configured in PyTorch templates.

### VS Code / Cursor

Use the Remote-SSH extension with the same SSH credentials.

---

## Uploading Data to Pods

Four methods, from simplest to most robust:

### Method 1: runpodctl (Simplest, No SSH Needed)

Pre-installed on all pods. Peer-to-peer transfer.

From your local machine:
```bash
runpodctl send my_features.db
# Output: Sending 'my_features.db' (2.1 GB)
# Code: 8338-galileo-collect-fidel
```

On the pod:
```bash
cd /workspace
runpodctl receive 8338-galileo-collect-fidel
```

### Method 2: SCP (Standard, Requires SSH)

```bash
# Upload a file
scp -P MAPPED_PORT -i ~/.ssh/id_ed25519 \
  ./features.db root@POD_IP:/workspace/features.db

# Upload a directory
scp -r -P MAPPED_PORT -i ~/.ssh/id_ed25519 \
  ./data/ root@POD_IP:/workspace/data/

# Download a file
scp -P MAPPED_PORT -i ~/.ssh/id_ed25519 \
  root@POD_IP:/workspace/model.pt ./model.pt
```

### Method 3: rsync (Best for Large Datasets)

Supports incremental transfers - only sends changed files.

```bash
# Upload dataset with compression and progress
rsync -avzP -e "ssh -p MAPPED_PORT -i ~/.ssh/id_ed25519" \
  ./data/ root@POD_IP:/workspace/data/

# Download trained models
rsync -avzP -e "ssh -p MAPPED_PORT -i ~/.ssh/id_ed25519" \
  root@POD_IP:/workspace/output/ ./output/
```

Key rsync flags:
- `-a` = archive mode (preserves permissions, timestamps)
- `-v` = verbose
- `-z` = compress during transfer
- `-P` = show progress + allow resume

### Method 4: wget from Cloud Storage (Best for Large/Repeated Transfers)

Upload your data to cloud storage first, then pull from the pod:

```bash
# On the pod:
cd /workspace
wget https://your-s3-bucket.s3.amazonaws.com/features.db
# or
pip install gdown
gdown https://drive.google.com/uc?id=FILE_ID
```

This avoids local upload speed bottlenecks.

### Windows Notes

- SCP and rsync work from Git Bash or WSL
- runpodctl has a Windows binary available
- For rsync on Windows, use WSL (Windows Subsystem for Linux)

---

## Running Training Jobs

### Option A: Interactive (SSH In and Run)

```bash
# SSH into the pod
ssh root@POD_IP -p MAPPED_PORT -i ~/.ssh/id_ed25519

# On the pod:
cd /workspace
pip install -r requirements.txt
python train.py --config configs/train.yaml
```

For long-running jobs, use tmux or screen:
```bash
tmux new -s training
python train.py --epochs 100
# Ctrl+B, D to detach
# tmux attach -t training to reattach
```

### Option B: Automated via Docker Start Command

Set the training command when creating the pod:

```python
response = requests.post(
    "https://rest.runpod.io/v1/pods",
    headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
    json={
        "name": "savage22-auto-train",
        "imageName": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
        "gpuTypeIds": ["NVIDIA GeForce RTX 5090"],
        "cloudType": "COMMUNITY",
        "containerDiskInGb": 50,
        "volumeInGb": 50,
        "dockerStartCmd": [
            "bash", "-c",
            "cd /workspace && pip install -r requirements.txt && python train.py"
        ],
    },
)
```

### Option C: Serverless Handler (For Repeated Jobs)

Package training as a serverless handler:

```python
import runpod

def train_handler(job):
    job_input = job["input"]
    epochs = job_input.get("epochs", 10)
    dataset_url = job_input.get("dataset_url")

    # Download data
    # ... your training code ...

    return {"status": "complete", "model_url": "https://..."}

runpod.serverless.start({"handler": train_handler})
```

---

## Downloading Results

### SCP

```bash
# Download single model file
scp -P MAPPED_PORT -i ~/.ssh/id_ed25519 \
  root@POD_IP:/workspace/output/model.pt ./models/model.pt

# Download entire output directory
scp -r -P MAPPED_PORT -i ~/.ssh/id_ed25519 \
  root@POD_IP:/workspace/output/ ./output/
```

### rsync (Recommended for Large Outputs)

```bash
rsync -avzP -e "ssh -p MAPPED_PORT -i ~/.ssh/id_ed25519" \
  root@POD_IP:/workspace/output/ ./output/
```

### runpodctl

On the pod:
```bash
runpodctl send /workspace/output/model.pt
# Note the transfer code
```

On your local machine:
```bash
runpodctl receive TRANSFER_CODE
```

---

## Storage Options

### Container Disk
- Temporary storage, wiped on pod restart or termination
- Default: 50 GB
- Use for: pip packages, temporary files

### Pod Volume
- Persistent across pod restarts (but lost on termination)
- Default: 20 GB, mounted at /workspace
- Use for: training data, checkpoints, code

### Network Volume
- Persists independently of pods
- Can be attached to multiple pods
- Survives pod termination
- Use for: datasets you reuse across training runs, final models
- Has S3-compatible API access

For training workflows, put everything important in /workspace (the volume mount).

---

## Serverless Alternative

For inference or batch processing (not interactive training), serverless may be
cheaper since you only pay for actual compute time.

```python
import runpod

# Create an endpoint for inference
endpoint = runpod.Endpoint("ENDPOINT_ID")

# Synchronous (blocks up to 90 seconds)
result = endpoint.run_sync({"input_data": "..."})

# Asynchronous
run = endpoint.run({"input_data": "..."})
status = run.status()   # Check status
output = run.output()   # Blocks until complete
```

Payload limits: 10 MB for async (/run), 20 MB for sync (/runsync).
For larger data, use URLs pointing to cloud storage.

---

## Full Workflow Example

Complete Python script for the Savage22 training pipeline:

```python
"""
RunPod Training Pipeline for Savage22
Handles: Pod creation -> Data upload -> Training -> Model download -> Cleanup
"""

import os
import time
import subprocess
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("RUNPOD_API_KEY")
BASE_URL = "https://rest.runpod.io/v1"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# --- Configuration ---
POD_CONFIG = {
    "name": "savage22-training",
    "imageName": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
    "computeType": "GPU",
    "cloudType": "COMMUNITY",
    "gpuCount": 1,
    "gpuTypeIds": [
        "NVIDIA GeForce RTX 5090",   # First choice: 32GB, $0.69/hr
        "NVIDIA GeForce RTX 4090",   # Fallback: 24GB, $0.34/hr
    ],
    "gpuTypePriority": "custom",
    "containerDiskInGb": 50,
    "volumeInGb": 50,
    "volumeMountPath": "/workspace",
    "ports": ["8888/http", "22/tcp"],
}

SSH_KEY = os.path.expanduser("~/.ssh/id_ed25519")

# Files to upload
LOCAL_FILES = [
    ("./features.db", "/workspace/features.db"),
    ("./train.py", "/workspace/train.py"),
    ("./requirements.txt", "/workspace/requirements.txt"),
]


def create_pod():
    """Create a GPU pod and return pod info."""
    print("[1/6] Creating pod...")
    resp = requests.post(f"{BASE_URL}/pods", headers=HEADERS, json=POD_CONFIG)
    resp.raise_for_status()
    pod = resp.json()
    print(f"  Pod ID: {pod['id']}")
    print(f"  GPU: {pod['gpu']['displayName']}")
    print(f"  Cost: ${pod['costPerHr']}/hr")
    return pod


def wait_for_pod(pod_id, timeout=300):
    """Wait for pod to be fully running with SSH available."""
    print("[2/6] Waiting for pod to be ready...")
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.get(f"{BASE_URL}/pods/{pod_id}", headers=HEADERS)
        pod = resp.json()
        status = pod.get("desiredStatus", "")
        public_ip = pod.get("publicIp")
        port_mappings = pod.get("portMappings", {})

        if status == "RUNNING" and public_ip and port_mappings.get("22"):
            print(f"  Pod ready! IP: {public_ip}, SSH Port: {port_mappings['22']}")
            return pod
        print(f"  Status: {status}, waiting...")
        time.sleep(10)
    raise TimeoutError(f"Pod {pod_id} did not become ready within {timeout}s")


def upload_files(pod_ip, ssh_port):
    """Upload training data and scripts to the pod via SCP."""
    print("[3/6] Uploading files...")
    for local_path, remote_path in LOCAL_FILES:
        if not os.path.exists(local_path):
            print(f"  SKIP (not found): {local_path}")
            continue
        print(f"  Uploading: {local_path} -> {remote_path}")
        cmd = [
            "scp", "-P", str(ssh_port),
            "-i", SSH_KEY,
            "-o", "StrictHostKeyChecking=no",
            local_path,
            f"root@{pod_ip}:{remote_path}",
        ]
        subprocess.run(cmd, check=True)
    print("  Upload complete.")


def run_training(pod_ip, ssh_port):
    """SSH into the pod and run the training script."""
    print("[4/6] Running training...")
    train_cmd = (
        "cd /workspace && "
        "pip install -r requirements.txt && "
        "python train.py 2>&1 | tee /workspace/training.log"
    )
    cmd = [
        "ssh", "-p", str(ssh_port),
        "-i", SSH_KEY,
        "-o", "StrictHostKeyChecking=no",
        f"root@{pod_ip}",
        train_cmd,
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  WARNING: Training exited with code {result.returncode}")
    else:
        print("  Training complete.")


def download_results(pod_ip, ssh_port, local_output="./output"):
    """Download trained models and logs from the pod."""
    print("[5/6] Downloading results...")
    os.makedirs(local_output, exist_ok=True)
    cmd = [
        "scp", "-r", "-P", str(ssh_port),
        "-i", SSH_KEY,
        "-o", "StrictHostKeyChecking=no",
        f"root@{pod_ip}:/workspace/output/",
        local_output,
    ]
    subprocess.run(cmd, check=True)
    # Also grab the training log
    subprocess.run([
        "scp", "-P", str(ssh_port),
        "-i", SSH_KEY,
        "-o", "StrictHostKeyChecking=no",
        f"root@{pod_ip}:/workspace/training.log",
        os.path.join(local_output, "training.log"),
    ], check=False)
    print(f"  Results saved to {local_output}")


def terminate_pod(pod_id):
    """Terminate the pod to stop all charges."""
    print("[6/6] Terminating pod...")
    import runpod
    runpod.api_key = API_KEY
    runpod.terminate_pod(pod_id)
    print("  Pod terminated. All charges stopped.")


def main():
    pod = create_pod()
    pod_id = pod["id"]

    try:
        pod = wait_for_pod(pod_id)
        pod_ip = pod["publicIp"]
        ssh_port = pod["portMappings"]["22"]

        upload_files(pod_ip, ssh_port)
        run_training(pod_ip, ssh_port)
        download_results(pod_ip, ssh_port)
    finally:
        terminate_pod(pod_id)

    print("\nDone! Check ./output/ for trained models.")


if __name__ == "__main__":
    main()
```

---

## GPU Pricing Comparison

As of March 2026:

| GPU | VRAM | Community $/hr | Secure $/hr | Best For |
|-----|------|---------------|-------------|----------|
| RTX 4090 | 24 GB | $0.34 | $0.44 | Budget training, smaller models |
| RTX 5090 | 32 GB | $0.69 | $0.89 | Mid-size training, more VRAM headroom |
| A100 PCIe | 80 GB | $1.19 | -- | Large models, VRAM-heavy workloads |
| H100 PCIe | 80 GB | $1.99 | -- | Maximum throughput |
| A100 SXM | 80 GB | $1.64 | -- | Multi-GPU training |
| H100 SXM | 80 GB | $2.49 | -- | Enterprise multi-GPU |

Additional costs:
- Volume storage: charged while pod exists (even when stopped)
- Network volumes: separate storage charge
- No ingress/egress fees (uploads/downloads are free)

---

## Multi-GPU Training (2x RTX 5090)

To use 2 GPUs, set `gpuCount=2` in pod creation. But XGBoost does NOT auto-scale
to multiple GPUs with just `device="cuda"`. You need **Dask + XGBoost distributed**.

### Pod Creation (2x 5090)

```python
pod = runpod.create_pod(
    name="savage22-2x5090",
    image_name="nvidia/cuda:12.3.2-devel-ubuntu22.04",
    gpu_type_id="NVIDIA GeForce RTX 5090",
    gpu_count=2,                    # <-- 2 GPUs
    volume_in_gb=100,
    container_disk_in_gb=50,
    min_vcpu_count=8,
    min_memory_in_gb=64,
    ports="22/tcp,8787/tcp",        # SSH + Dask dashboard
    volume_mount_path="/workspace",
)
```

### Multi-GPU XGBoost Training Script

```python
# Install on pod: pip install dask-cuda xgboost dask distributed
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
import xgboost as xgb
import dask.dataframe as dd

# 1. Create Dask cluster with 1 worker per GPU
cluster = LocalCUDACluster(n_workers=2, threads_per_worker=1)
client = Client(cluster)

# 2. Load data as Dask DataFrame
ddf = dd.read_parquet("/workspace/features.parquet")
X = ddf.drop(columns=["target"])
y = ddf["target"]

# 3. Create DaskQuantileDMatrix (GPU-efficient)
dtrain = xgb.dask.DaskQuantileDMatrix(client, X, y)

# 4. Train with distributed XGBoost
params = {
    "tree_method": "hist",
    "device": "cuda",
    "max_depth": 8,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "multi:softprob",
    "num_class": 3,
}
output = xgb.dask.train(client, params, dtrain, num_boost_round=500)

# 5. Save model
output["booster"].save_model("/workspace/model.json")
client.close()
cluster.close()
```

### Important Notes for 2x GPU
- `dask-cuda` auto-detects both GPUs and pins one worker to each
- `QuantileDMatrix` reduces memory pressure vs regular DMatrix
- Benchmark 1 vs 2 GPUs -- for small datasets (<100K rows), 1 GPU may be faster (PCIe overhead)
- Our datasets (56K-547K rows, 400-843 features) are in the sweet spot where 2 GPUs should help
- Enable memory spilling if VRAM gets tight: `LocalCUDACluster(enable_tcp_over_ucx=True)`
- Cost: 2x 5090 = $1.38/hr Community Cloud

---

## Gotchas and Best Practices

### Critical

1. **Stop vs Terminate**: Stopping a pod releases the GPU but you STILL PAY for
   volume storage. Terminate to stop all charges.

2. **Container disk is temporary**: Anything not in /workspace (the volume mount)
   is LOST when the pod restarts. Always save checkpoints to /workspace.

3. **RTX 5090 availability is spotty**: Always have fallback GPU types in your
   gpuTypeIds list. The 4090 at $0.34/hr is a solid backup.

4. **SSH port is NOT 22 externally**: RunPod maps port 22 to a random high port.
   Check portMappings in the pod response to get the actual port.

5. **100-second HTTP timeout**: Cloudflare enforces this on the HTTP proxy.
   Use SSH for long-running connections, not the web terminal.

### Performance

6. **Use PyTorch CUDA images**: Start with `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`
   or similar. Building CUDA from scratch wastes time and money.

7. **Upload speed is your bottleneck**: Your home upload is likely slower than
   the pod's download. For large datasets (>10 GB), upload to cloud storage
   first (S3, GCS, Hugging Face) then wget from the pod.

8. **Use rsync over SCP for large transfers**: rsync supports resume and only
   sends changed data. Essential for multi-GB feature databases.

9. **Use Network Volumes for repeated training**: If you retrain regularly, a
   network volume persists your data across pod terminations. No re-uploading.

### Cost Control

10. **Community Cloud is ~22% cheaper**: $0.69 vs $0.89/hr for RTX 5090. Less
    reliable but fine for training (you can restart if interrupted).

11. **Use interruptible (spot) instances for non-critical training**: Even cheaper
    but can be preempted. Save checkpoints frequently.

12. **Always terminate when done**: Set up alerts or auto-termination. A forgotten
    pod at $0.69/hr costs ~$500/month.

13. **Monitor with the console**: https://www.console.runpod.io/pods shows all
    running pods and current charges.

### Security

14. **Never commit API keys**: Use .env files and load_dotenv().

15. **Exposed services are public**: If you expose ports (JupyterLab, etc.),
    anyone can access them. Add authentication or use SSH tunneling.

16. **Docker Compose is NOT supported**: Use custom templates with all
    dependencies baked into a single container.

17. **UDP is NOT supported**: Only TCP connections work on RunPod pods.

---

## Quick Reference

```bash
# Install
pip install runpod python-dotenv

# Create pod (Python)
import runpod; runpod.api_key = "..."; pod = runpod.create_pod("name", "image", "GPU_TYPE")

# Upload (SCP)
scp -P PORT -i KEY file root@IP:/workspace/

# Download (SCP)
scp -P PORT -i KEY root@IP:/workspace/model.pt ./

# Stop (saves data, releases GPU)
runpod.stop_pod("POD_ID")

# Terminate (destroys everything)
runpod.terminate_pod("POD_ID")
```

---

## References

- RunPod Docs: https://docs.runpod.io/
- Python SDK: https://github.com/runpod/runpod-python
- Pod API Reference: https://docs.runpod.io/api-reference/pods/POST/pods
- GPU Pricing: https://www.runpod.io/gpu-pricing
- RTX 5090 Page: https://www.runpod.io/gpu-models/rtx-5090
- File Transfer Guide: https://docs.runpod.io/pods/storage/transfer-files
- Console: https://www.console.runpod.io/
