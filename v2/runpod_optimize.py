"""
RunPod Exhaustive Optimizer — 4x RTX 5090
Runs exhaustive_optimizer.py on RunPod cloud with 4 GPUs.
"""
import os
import sys
import time
import subprocess

# Reuse the training script's infrastructure
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Override GPU count for optimizer
os.environ['RUNPOD_GPU_COUNT'] = '4'

# Load .env
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                k, v = line.split('=', 1)
                os.environ.setdefault(k, v)

api_key = os.environ.get('RUNPOD_API_KEY', '')
if not api_key:
    print("ERROR: RUNPOD_API_KEY not found in .env")
    sys.exit(1)

import runpod
runpod.api_key = api_key

DB_DIR = os.path.dirname(os.path.abspath(__file__))
SSH_KEY = os.path.expanduser("~/.ssh/id_ed25519")
REMOTE_DIR = "/workspace"
POD_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
GPU_TYPE = "NVIDIA GeForce RTX 5090"
GPU_COUNT = 4
COST_PER_HOUR = 2.76  # 4x 5090

def ssh_cmd(host, port, cmd, timeout=None):
    args = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR", "-p", str(port), "-i", SSH_KEY,
            "root@%s" % host, cmd]
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except:
        return -1, "", "timeout"

def scp_upload(host, port, local, remote):
    args = ["scp", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR", "-P", str(port), "-i", SSH_KEY,
            local, "root@%s:%s" % (host, remote)]
    r = subprocess.run(args, capture_output=True, text=True, timeout=600)
    return r.returncode == 0

def scp_download(host, port, remote, local):
    args = ["scp", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR", "-P", str(port), "-i", SSH_KEY,
            "root@%s:%s" % (host, remote), local]
    r = subprocess.run(args, capture_output=True, text=True, timeout=300)
    return r.returncode == 0

def main():
    start = time.time()
    pod_id = None

    try:
        # Health check
        gpus = runpod.get_gpus()
        print("API OK, %d GPU types" % len(gpus))

        # Create pod
        print("\nCreating 4x RTX 5090 pod ($%.2f/hr)..." % COST_PER_HOUR)
        pod = runpod.create_pod(
            name="savage22-optimizer-4x5090",
            image_name=POD_IMAGE,
            gpu_type_id=GPU_TYPE,
            gpu_count=GPU_COUNT,
            volume_in_gb=200,
            container_disk_in_gb=50,
            min_vcpu_count=16,
            min_memory_in_gb=64,
            ports="22/tcp",
            volume_mount_path=REMOTE_DIR,
            cloud_type="COMMUNITY",
        )
        pod_id = pod["id"]
        print("Pod: %s" % pod_id)

        # Wait for SSH
        print("Waiting for SSH...")
        for _ in range(30):
            time.sleep(10)
            p = runpod.get_pod(pod_id)
            rt = p.get("runtime") or {}
            ports = rt.get("ports") or []
            for pi in ports:
                if pi.get("privatePort") == 22:
                    host = pi["ip"]
                    port = pi["publicPort"]
                    # Test SSH
                    for attempt in range(12):
                        rc, out, _ = ssh_cmd(host, int(port), "echo OK", timeout=10)
                        if rc == 0 and "OK" in out:
                            print("SSH connected: %s:%s" % (host, port))
                            goto_upload = True
                            break
                        time.sleep(15)
                    if 'goto_upload' in dir():
                        break
            if 'goto_upload' in dir():
                break
        else:
            print("SSH timeout!")
            return

        port = int(port)

        # Upload optimizer + dependencies
        files = [
            "exhaustive_optimizer.py",
            "features_5m.db", "features_15m.db", "features_1h.db",
            "features_4h.db", "features_1d.db", "features_1w.db",
            "model_5m.json", "model_15m.json", "model_1h.json",
            "model_4h.json", "model_1d.json", "model_1w.json",
            "features_5m_all.json", "features_15m_all.json", "features_1h_all.json",
            "features_4h_all.json", "features_1d_all.json", "features_1w_all.json",
        ]
        print("\nUploading %d files..." % len(files))
        for f in files:
            local = os.path.join(DB_DIR, f)
            if os.path.exists(local):
                sz = os.path.getsize(local) / 1e6
                print("  %s (%.1f MB)..." % (f, sz), end=" ", flush=True)
                if scp_upload(host, port, local, REMOTE_DIR + "/" + f):
                    print("OK")
                else:
                    print("FAILED")
            else:
                print("  %s -- MISSING (skip)" % f)

        # Install deps
        print("\nInstalling deps...")
                # Install CuPy from source for RTX 5090 Blackwell support
        ssh_cmd(host, port,
            "pip install xgboost numpy pandas scikit-learn scipy hmmlearn 2>&1 | tail -3 && "
            "CUPY_NUM_BUILD_JOBS=8 pip install cupy-cuda12x --no-binary cupy-cuda12x 2>&1 | tail -5",
            timeout=600)  # source build takes a few minutes

        # Run optimizer
        print("\nRunning exhaustive optimizer on 4x RTX 5090...")
        print("This will take a while (2.88B combos)...\n")

        # Run 4 optimizer instances in parallel, each on a different GPU, each doing different TFs
        # GPU 0: 5m, GPU 1: 15m, GPU 2: 1h+4h, GPU 3: 1d+1w
        opt_cmd = (
            "cd %s && "
            "SAVAGE22_DB_DIR=%s CUDA_VISIBLE_DEVICES=0 python -u exhaustive_optimizer.py --tf 5m  > opt_5m.log 2>&1 & "
            "SAVAGE22_DB_DIR=%s CUDA_VISIBLE_DEVICES=1 python -u exhaustive_optimizer.py --tf 15m > opt_15m.log 2>&1 & "
            "SAVAGE22_DB_DIR=%s CUDA_VISIBLE_DEVICES=2 python -u exhaustive_optimizer.py --tf 1h --tf 4h > opt_1h_4h.log 2>&1 & "
            "SAVAGE22_DB_DIR=%s CUDA_VISIBLE_DEVICES=3 python -u exhaustive_optimizer.py --tf 1d --tf 1w > opt_1d_1w.log 2>&1 & "
            "wait && echo ALL_DONE && "
            "cat opt_5m.log opt_15m.log opt_1h_4h.log opt_1d_1w.log > optimizer.log && "
            "python -c \"import json,glob; configs={}; [configs.update(json.load(open(f))) for f in glob.glob('exhaustive_configs_*.json')]; json.dump(configs, open('exhaustive_configs.json','w'), indent=2); print('Merged', len(configs), 'TF configs')\""
        ) % (REMOTE_DIR, REMOTE_DIR, REMOTE_DIR, REMOTE_DIR, REMOTE_DIR)

        proc = subprocess.Popen(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
             "-o", "LogLevel=ERROR", "-o", "ServerAliveInterval=30",
             "-p", str(port), "-i", SSH_KEY, "root@%s" % host, opt_cmd],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        for line in proc.stdout:
            elapsed_hr = (time.time() - start) / 3600
            cost = elapsed_hr * COST_PER_HOUR
            print("[$%.2f] %s" % (cost, line.rstrip()))
        proc.wait()

        # Download results
        print("\nDownloading results...")
        os.makedirs(os.path.join(DB_DIR, "runpod_output"), exist_ok=True)
        for f in ["exhaustive_configs.json", "optimizer.log"]:
            remote = REMOTE_DIR + "/" + f
            local = os.path.join(DB_DIR, "runpod_output", f)
            if scp_download(host, port, remote, local):
                print("  %s -> %s" % (f, local))
                # Also copy to project root
                import shutil
                shutil.copy2(local, os.path.join(DB_DIR, f))

        elapsed = time.time() - start
        cost = (elapsed / 3600) * COST_PER_HOUR
        print("\nDone! Duration: %.0fs, Cost: $%.2f" % (elapsed, cost))

    finally:
        if pod_id:
            print("\nStopping pod %s (volume preserved for restart)..." % pod_id)
            try:
                runpod.stop_pod(pod_id)
                print("Stopped. Data preserved. Resume with: runpod.resume_pod('%s')" % pod_id)
                print("To terminate and delete: runpod.terminate_pod('%s')" % pod_id)
            except:
                print("ERROR stopping -- check console!")

if __name__ == "__main__":
    main()
