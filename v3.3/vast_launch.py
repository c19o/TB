#!/usr/bin/env python3
"""
vast_launch.py — vast.ai machine search, rent, and deploy helper.

Searches vast.ai for machines matching requirements, rents, and prints
ready-to-use SSH/SCP/pip install commands for deployment.

Usage:
  python v3.3/vast_launch.py --tf 1w --ram 128 --cores 64
  python v3.3/vast_launch.py --tf 1d --ram 128 --cores 128 --max-price 1.50
  python v3.3/vast_launch.py --tf 15m --ram 2048 --cores 256 --max-price 5.00
  python v3.3/vast_launch.py --search-only --ram 128 --cores 64

The script:
  1. Searches vast.ai for matching offers (sorted by $/hr)
  2. Displays a table with CPU Score, driver, CUDA, RAM, and $/hr
  3. Asks for confirmation before renting
  4. Creates the instance with a lightweight base image
  5. Waits for SSH to come up
  6. Prints ready-to-use SSH/SCP + pip install commands

Requirements:
  - vastai CLI installed and API key configured
  - SSH key at ~/.ssh/vast_key
"""

import argparse
import json
import os
import subprocess
import sys
import time
import re


# ---------------------------------------------------------------------------
# Base image — lightweight, usually cached on vast.ai
# ---------------------------------------------------------------------------
BASE_IMAGE = 'pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime'

# Per-TF resource requirements (from CLOUD_TRAINING_PROTOCOL.md)
TF_REQUIREMENTS = {
    '1w':  {'ram': 64,   'cores': 64,  'disk': 30},
    '1d':  {'ram': 192,  'cores': 128, 'disk': 50},
    '4h':  {'ram': 768,  'cores': 128, 'disk': 50},
    '1h':  {'ram': 1024, 'cores': 256, 'disk': 80},
    '15m': {'ram': 2048, 'cores': 256, 'disk': 150},
}


def run_vastai(args_list, raw=False):
    """Run a vastai CLI command, return parsed JSON if --raw, else stdout string."""
    cmd = ['vastai'] + args_list
    if raw and '--raw' not in args_list:
        cmd.append('--raw')
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        print(f"ERROR: vastai command failed: {' '.join(cmd)}")
        print(f"  stderr: {result.stderr.strip()}")
        sys.exit(1)
    if raw:
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            print(f"ERROR: Failed to parse vastai JSON output")
            print(f"  stdout: {result.stdout[:500]}")
            sys.exit(1)
    return result.stdout.strip()


def search_offers(min_ram, min_cores, max_price=None, limit=20):
    """Search vast.ai for matching offers, return list of offer dicts.

    Note: vast.ai search filter accepts cpu_ram in GB (e.g., cpu_ram>=128),
    but the response value is in MB (e.g., 131072). Filter in GB, display in MB/1024.
    """
    query = (
        f'cpu_ram>={min_ram} cpu_cores>={min_cores} '
        f'cuda_vers>=12.0 '
        f'rentable=true '
        f'num_gpus>=1'
    )
    if max_price:
        query += f' dph_total<={max_price}'

    offers = run_vastai(
        ['search', 'offers', query, '-o', 'dph_total', '--limit', str(limit)],
        raw=True
    )
    return offers if isinstance(offers, list) else []


def display_offers(offers):
    """Print a formatted table of offers with auto-selected images."""
    if not offers:
        print("No matching offers found.")
        return

    # Header
    print()
    print(f"{'#':>3}  {'ID':>10}  {'GPU':20s}  {'vCPUs':>5}  {'GHz':>5}  {'CPU Score':>9}  "
          f"{'RAM GB':>6}  {'Driver':>10}  {'CUDA':>5}  {'$/hr':>6}")
    print("-" * 110)

    for i, o in enumerate(offers):
        gpu_name = o.get('gpu_name', '?')
        num_gpus = o.get('num_gpus', 1)
        gpu_label = f"{num_gpus}x {gpu_name}" if num_gpus > 1 else gpu_name
        if len(gpu_label) > 20:
            gpu_label = gpu_label[:19] + '.'

        cores = o.get('cpu_cores', 0)  # Total vCPUs (vast.ai vCPU = thread, not core)
        # vast.ai reports cpu_ghz as the clock speed
        ghz = o.get('cpu_ghz', 0) or 0
        cpu_score = cores * ghz
        ram_gb = round((o.get('cpu_ram', 0) or 0) / 1024, 0)  # vast.ai reports RAM in MB
        driver = o.get('driver_version', '?')
        cuda = o.get('cuda_max_good', o.get('cuda_vers', '?'))
        dph = o.get('dph_total', 0)
        offer_id = o.get('id', '?')

        print(f"{i+1:>3}  {offer_id:>10}  {gpu_label:20s}  {cores:>5}  {ghz:>5.2f}  "
              f"{cpu_score:>9.1f}  {ram_gb:>6.0f}  {driver:>10}  {cuda!s:>5}  "
              f"${dph:>5.2f}")

    print()


def wait_for_instance(instance_id, timeout=300):
    """Wait for instance to reach 'running' status with SSH info."""
    print(f"Waiting for instance {instance_id} to start (timeout {timeout}s)...")
    start = time.time()
    while time.time() - start < timeout:
        instances = run_vastai(['show', 'instances'], raw=True)
        for inst in (instances if isinstance(instances, list) else []):
            if inst.get('id') == instance_id:
                status = inst.get('actual_status', '')
                if status == 'running':
                    ssh_host = inst.get('ssh_host', '')
                    ssh_port = inst.get('ssh_port', '')
                    if ssh_host and ssh_port:
                        return ssh_host, int(ssh_port)
                elif status in ('exited', 'error', 'offline'):
                    print(f"ERROR: Instance entered state '{status}'. Check vast.ai dashboard.")
                    sys.exit(1)
                print(f"  Status: {status} ({time.time()-start:.0f}s elapsed)")
        time.sleep(10)

    print(f"ERROR: Instance did not start within {timeout}s")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Unified vast.ai launcher with auto CUDA image selection')
    parser.add_argument('--tf', type=str, help='Timeframe (1w, 1d, 4h, 1h, 15m). Sets default RAM/cores.')
    parser.add_argument('--ram', type=int, help='Minimum RAM in GB (overrides TF default)')
    parser.add_argument('--cores', type=int, help='Minimum CPU cores (overrides TF default)')
    parser.add_argument('--disk', type=int, default=50, help='Disk space in GB (default: 50)')
    parser.add_argument('--max-price', type=float, help='Max $/hr filter')
    parser.add_argument('--limit', type=int, default=15, help='Number of offers to show (default: 15)')
    parser.add_argument('--search-only', action='store_true', help='Only search and display, do not rent')
    parser.add_argument('--offer-id', type=int, help='Skip search, rent this specific offer ID')
    parser.add_argument('--ssh-key', type=str, default='~/.ssh/vast_key', help='SSH key path')
    args = parser.parse_args()

    # Resolve requirements
    if args.tf and args.tf in TF_REQUIREMENTS:
        reqs = TF_REQUIREMENTS[args.tf]
        min_ram = args.ram or reqs['ram']
        min_cores = args.cores or reqs['cores']
        disk = args.disk or reqs['disk']
    else:
        min_ram = args.ram or 128
        min_cores = args.cores or 64
        disk = args.disk or 50

    ssh_key = os.path.expanduser(args.ssh_key)

    if args.offer_id:
        # Direct rent
        print(f"Renting offer {args.offer_id}...")

    else:
        # Search for offers
        print(f"Searching vast.ai: RAM >= {min_ram} GB, Cores >= {min_cores}"
              + (f", Max ${args.max_price:.2f}/hr" if args.max_price else ""))
        offers = search_offers(min_ram, min_cores, args.max_price, args.limit)

        if not offers:
            print("No matching offers found. Try relaxing constraints.")
            sys.exit(1)

        display_offers(offers)

        if args.search_only:
            print("(--search-only mode, not renting)")
            return

        # Ask user to pick
        while True:
            choice = input(f"Pick offer # (1-{len(offers)}), or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                print("Aborted.")
                return
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(offers):
                    offer = offers[idx]
                    break
            except ValueError:
                pass
            print(f"Invalid choice. Enter 1-{len(offers)} or 'q'.")

        args.offer_id = offer['id']

    # Confirm
    print()
    print(f"  Offer ID:  {args.offer_id}")
    print(f"  Image:     {BASE_IMAGE}")
    print(f"  Disk:      {disk} GB")
    tf_str = f" (for {args.tf} training)" if args.tf else ""
    print(f"  Purpose:   Training{tf_str}")
    print()
    confirm = input("Rent this machine? [y/N]: ").strip().lower()
    if confirm != 'y':
        print("Aborted.")
        return

    # Create instance
    print(f"\nCreating instance on offer {args.offer_id} with {BASE_IMAGE}...")
    create_output = run_vastai([
        'create', 'instance', str(args.offer_id),
        '--image', BASE_IMAGE,
        '--disk', str(disk),
        '--ssh',
    ])
    print(f"  vast.ai response: {create_output}")

    # Parse instance ID from response
    # Response format: "Started. {'new_contract': 12345678}"
    instance_id = None
    try:
        if 'new_contract' in create_output:
            # Extract the number after 'new_contract':
            match = re.search(r"'new_contract':\s*(\d+)", create_output)
            if match:
                instance_id = int(match.group(1))
    except Exception:
        pass

    if not instance_id:
        print("WARNING: Could not parse instance ID from response.")
        print("Check vast.ai dashboard for your new instance.")
        return

    print(f"  Instance ID: {instance_id}")

    # Wait for it to come up
    ssh_host, ssh_port = wait_for_instance(instance_id)

    # Print ready-to-use commands
    print()
    print("=" * 70)
    print(f"INSTANCE READY: {instance_id}")
    print("=" * 70)
    print()
    print("# SSH connection:")
    ssh_opts = f"-o StrictHostKeyChecking=no -o IdentityFile={ssh_key} -o IdentitiesOnly=yes"
    print(f'SSH="{ssh_opts} -p {ssh_port} root@{ssh_host}"')
    print(f'ssh $SSH_OPTS -p {ssh_port} root@{ssh_host}')
    print()
    print("# Install deps:")
    print(f'ssh $SSH_OPTS -p {ssh_port} root@{ssh_host} '
          '"pip install -q lightgbm scikit-learn scipy ephem astropy pytz joblib '
          'pandas numpy pyarrow optuna hmmlearn numba tqdm pyyaml '
          'alembic cmaes colorlog sqlalchemy threadpoolctl 2>&1 | tail -5"')
    print()
    print("# Upload tar:")
    print(f'scp {ssh_opts} -P {ssh_port} /tmp/v33_upload.tar.gz root@{ssh_host}:/workspace/')
    print()
    print("# Extract + symlink:")
    print(f'ssh $SSH_OPTS -p {ssh_port} root@{ssh_host} '
          '"cd /workspace && tar xzf v33_upload.tar.gz && '
          'for f in *.db kp_history_gfz.txt; do ln -sf /workspace/\\$f /workspace/v3.3/\\$f; done && '
          'ln -sf /workspace/astrology_engine.py /workspace/v3.3/"')
    print()
    if args.tf:
        print(f"# Launch {args.tf} pipeline:")
        print(f'ssh $SSH_OPTS -p {ssh_port} root@{ssh_host} '
              f'"cd /workspace/v3.3 && '
              f'export SAVAGE22_DB_DIR=/workspace && '
              f'export V30_DATA_DIR=/workspace/v3.3 && '
              f'export PYTHONUNBUFFERED=1 && '
              f'nohup python -u cloud_run_tf.py --symbol BTC --tf {args.tf} '
              f'> /workspace/{args.tf}_pipeline.log 2>&1 &"')
        print()
        print(f"# Monitor:")
        print(f'ssh $SSH_OPTS -p {ssh_port} root@{ssh_host} "tail -20 /workspace/{args.tf}_pipeline.log"')
        print()
        print(f"# Check for completion:")
        print(f'ssh $SSH_OPTS -p {ssh_port} root@{ssh_host} '
              f'"ls /workspace/DONE_{args.tf} 2>/dev/null && echo DONE || echo RUNNING"')
    print()
    print(f"# Kill when done:")
    print(f'vastai destroy instance {instance_id}')
    print()


if __name__ == '__main__':
    main()
