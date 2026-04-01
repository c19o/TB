#!/usr/bin/env python3
"""
vast_selector.py — Intelligent vast.ai Machine Selector with Discord Approval
=============================================================================
Searches vast.ai for machines matching 4 predefined profiles, ranks by CPU Score
(cores × GHz), and requires Discord approval before renting.

Machine Profiles:
  smoke    - Cheapest 24GB+ GPU for pipeline validation (≤ $0.50/hr, auto-approved)
  light    - 1w/1d training (128GB RAM, 64+ cores, 1x 24GB GPU)
  medium   - 4h/1h training (768GB RAM, 128+ cores, 4x 24GB GPU)
  heavy    - 15m training (2TB RAM, 256+ cores, 4x 80GB GPU, user picks)

Usage:
  python v3.3/vast_selector.py --profile smoke --search-only
  python v3.3/vast_selector.py --profile light --rent
  python v3.3/vast_selector.py --profile heavy --rent

Design:
  - CPU Score = cores × GHz (higher = faster builds)
  - Smoke tests auto-approved (< $0.50/hr)
  - All other rentals require Discord approval
  - Never auto-select for 15m (user picks from table)
  - Fastest machine > cheapest machine (memory from v3.3 feedback)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Import Discord gate for approval workflow
try:
    from discord_gate import gate
except ImportError:
    print("ERROR: discord_gate.py not found. Approval workflow disabled.")
    gate = None


# ---------------------------------------------------------------------------
# Machine Profiles (4 tiers matching v3.3 requirements)
# ---------------------------------------------------------------------------

PROFILES = {
    'smoke': {
        'name': 'Smoke Test / Bug Verification',
        'ram': 32,
        'cores': 16,
        'disk': 30,
        'min_gpus': 1,
        'min_vram': 24,
        'max_price': 0.50,
        'auto_approve': True,
        'note': 'Cheapest 24GB+ GPU for pipeline validation',
    },
    'light': {
        'name': '1w/1d Training',
        'ram': 128,
        'cores': 64,
        'disk': 50,
        'min_gpus': 1,
        'min_vram': 24,
        'max_price': None,
        'auto_approve': False,
        'note': 'Base features + month crosses (~3M features)',
    },
    'medium': {
        'name': '4h/1h Training',
        'ram': 768,
        'cores': 128,
        'disk': 80,
        'min_gpus': 4,
        'min_vram': 24,
        'max_price': None,
        'auto_approve': False,
        'note': 'DOY crosses + multi-GPU Optuna (~6M features)',
    },
    'heavy': {
        'name': '15m Training',
        'ram': 2048,
        'cores': 256,
        'disk': 150,
        'min_gpus': 4,
        'min_vram': 80,
        'max_price': None,
        'auto_approve': False,
        'note': 'Largest TF, user must pick machine personally',
    },
}

# Base Docker image (lightweight, cached on vast.ai)
BASE_IMAGE = 'pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime'


# ---------------------------------------------------------------------------
# vast.ai CLI Wrappers
# ---------------------------------------------------------------------------

def run_vastai(args_list, raw=False):
    """Run a vastai CLI command, return parsed JSON if --raw, else stdout string."""
    cmd = ['vastai'] + args_list
    if raw and '--raw' not in args_list:
        cmd.append('--raw')
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        print("ERROR: vastai command timed out after 60s")
        sys.exit(1)

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


def search_offers(profile):
    """Search vast.ai for machines matching profile, return scored/sorted list."""
    p = PROFILES[profile]

    # Build query string (RAM in GB, VRAM per GPU in GB)
    query = (
        f'cpu_ram>={p["ram"]} '
        f'cpu_cores>={p["cores"]} '
        f'num_gpus>={p["min_gpus"]} '
        f'gpu_ram>={p["min_vram"]} '
        f'cuda_vers>=12.0 '
        f'driver_version>=535.00.00 '
        f'rentable=true '
        f'verified=true'
    )
    if p['max_price']:
        query += f' dph_total<={p["max_price"]}'

    # Search (sort by CPU score descending = fastest first)
    offers = run_vastai(
        ['search', 'offers', query, '-o', 'cpu_cores_effective-,cpu_ghz-', '--limit', '20'],
        raw=True
    )

    if not isinstance(offers, list):
        return []

    # Compute CPU Score for each offer
    for o in offers:
        cores = o.get('cpu_cores_effective', o.get('cpu_cores', 0))
        ghz = o.get('cpu_ghz', 0) or 0
        o['_cpu_score'] = cores * ghz
        o['_ram_gb'] = round((o.get('cpu_ram', 0) or 0) / 1024, 0)

    # Sort by CPU score descending (fastest first)
    offers.sort(key=lambda o: o['_cpu_score'], reverse=True)

    return offers


def display_offers(offers, profile):
    """Print formatted table of offers, ranked by CPU Score."""
    if not offers:
        print(f"No machines found matching profile '{profile}'")
        return

    p = PROFILES[profile]
    print()
    print(f"  PROFILE: {p['name']}")
    print(f"  Requirements: {p['ram']}GB RAM, {p['cores']}+ cores, "
          f"{p['min_gpus']}x {p['min_vram']}GB VRAM")
    if p['max_price']:
        print(f"  Max Price: ${p['max_price']:.2f}/hr")
    print()
    print(f"{'#':>3}  {'ID':>10}  {'GPU':25s}  {'Cores':>5}  {'GHz':>5}  {'CPU Score':>9}  "
          f"{'RAM GB':>6}  {'VRAM':>5}  {'$/hr':>6}  {'$/day':>7}")
    print("-" * 115)

    for i, o in enumerate(offers):
        gpu_name = o.get('gpu_name', '?')
        num_gpus = o.get('num_gpus', 1)
        gpu_label = f"{num_gpus}x {gpu_name}" if num_gpus > 1 else gpu_name
        if len(gpu_label) > 25:
            gpu_label = gpu_label[:24] + '…'

        cores = o.get('cpu_cores_effective', o.get('cpu_cores', 0))
        ghz = o.get('cpu_ghz', 0) or 0
        cpu_score = o.get('_cpu_score', 0)
        ram_gb = o.get('_ram_gb', 0)
        vram_gb = round(o.get('gpu_ram', 0) / 1024, 0) if o.get('gpu_ram', 0) > 100 else o.get('gpu_ram', 0)
        dph = o.get('dph_total', 0)
        dpd = dph * 24
        offer_id = o.get('id', '?')

        marker = " ← FASTEST" if i == 0 else ""
        print(f"{i+1:>3}  {offer_id:>10}  {gpu_label:25s}  {cores:>5}  {ghz:>5.2f}  "
              f"{cpu_score:>9.1f}  {ram_gb:>6.0f}  {vram_gb:>5.0f}  ${dph:>5.2f}  ${dpd:>6.1f}{marker}")

    print()


def request_discord_approval(profile, offer):
    """Request user approval via Discord for machine rental."""
    if gate is None:
        print("WARNING: Discord gate not available. Proceeding without approval.")
        return True

    p = PROFILES[profile]

    # Auto-approve smoke tests (< $0.50/hr)
    if p.get('auto_approve', False):
        print(f"  Auto-approved: Smoke test machine ${offer['dph_total']:.2f}/hr")
        return True

    # Build approval request metadata
    cores = offer.get('cpu_cores_effective', offer.get('cpu_cores', 0))
    ghz = offer.get('cpu_ghz', 0) or 0
    gpu_name = offer.get('gpu_name', '?')
    num_gpus = offer.get('num_gpus', 1)

    metadata = {
        'profile': profile,
        'offer_id': offer['id'],
        'gpu': f"{num_gpus}x {gpu_name}",
        'cpu_score': f"{cores * ghz:.0f} (cores={cores}, GHz={ghz:.2f})",
        'ram_gb': round((offer.get('cpu_ram', 0) or 0) / 1024, 0),
        'vram_gb': round(offer.get('gpu_ram', 0) / 1024, 0),
        'price_per_hour': f"${offer['dph_total']:.2f}",
        'price_per_day': f"${offer['dph_total'] * 24:.2f}",
        'location': offer.get('geolocation', '?'),
    }

    # 15m profile requires user to pick personally (never auto)
    if profile == 'heavy':
        metadata['note'] = 'USER MUST PICK 15m MACHINE PERSONALLY (memory rule)'

    print(f"\n  Requesting Discord approval for {p['name']} machine...")
    approved = gate.approve('rent_machine', metadata)

    if approved:
        print("  ✓ APPROVED by user")
    else:
        print("  ✗ DENIED by user")

    return approved


def rent_machine(offer_id, profile):
    """Rent a vast.ai instance with the base image."""
    p = PROFILES[profile]
    disk = p['disk']

    print(f"\n  Creating instance on offer {offer_id}...")
    create_output = run_vastai([
        'create', 'instance', str(offer_id),
        '--image', BASE_IMAGE,
        '--disk', str(disk),
        '--ssh',
    ])
    print(f"  vast.ai response: {create_output}")

    # Parse instance ID from response
    # Response format: "Started. {'new_contract': 12345678}"
    import re
    instance_id = None
    try:
        if 'new_contract' in create_output:
            match = re.search(r"'new_contract':\s*(\d+)", create_output)
            if match:
                instance_id = int(match.group(1))
    except Exception:
        pass

    if not instance_id:
        print("WARNING: Could not parse instance ID. Check vast.ai dashboard.")
        return None

    print(f"  Instance ID: {instance_id}")
    return instance_id


def wait_for_ssh(instance_id, timeout=300):
    """Wait for instance to reach 'running' status with SSH info."""
    print(f"  Waiting for instance {instance_id} to start (timeout {timeout}s)...")
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
                print(f"    Status: {status} ({time.time()-start:.0f}s elapsed)")
        time.sleep(10)

    print(f"ERROR: Instance did not start within {timeout}s")
    sys.exit(1)


def print_deploy_commands(instance_id, ssh_host, ssh_port, profile):
    """Print ready-to-use SSH/SCP/pip commands for deployment."""
    print()
    print("=" * 80)
    print(f"INSTANCE READY: {instance_id} ({PROFILES[profile]['name']})")
    print("=" * 80)
    print()
    print("# SSH connection:")
    print(f'ssh -p {ssh_port} root@{ssh_host}')
    print()
    print("# Install dependencies:")
    print(f'ssh -p {ssh_port} root@{ssh_host} "pip install -q lightgbm scikit-learn scipy '
          'ephem astropy pytz joblib pandas numpy pyarrow optuna hmmlearn numba tqdm pyyaml '
          'sparse-dot-mkl datasketch 2>&1 | tail -5"')
    print()
    print("# Upload code + DBs:")
    print(f'scp -P {ssh_port} /tmp/v33_upload.tar.gz root@{ssh_host}:/workspace/')
    print()
    print("# Extract + symlink:")
    print(f'ssh -p {ssh_port} root@{ssh_host} "cd /workspace && tar xzf v33_upload.tar.gz && '
          'for f in *.db kp_history_gfz.txt; do ln -sf /workspace/$f /workspace/v3.3/$f; done"')
    print()
    print(f"# Destroy when done:")
    print(f'vastai destroy instance {instance_id}')
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Intelligent vast.ai machine selector with 4 profiles and Discord approval'
    )
    parser.add_argument('--profile', type=str, required=True,
                        choices=['smoke', 'light', 'medium', 'heavy'],
                        help='Machine profile (smoke, light, medium, heavy)')
    parser.add_argument('--search-only', action='store_true',
                        help='Only search and display, do not rent')
    parser.add_argument('--rent', action='store_true',
                        help='Search, get approval, and rent the machine')
    parser.add_argument('--offer-id', type=int,
                        help='Skip search, rent this specific offer ID (still requires approval)')
    args = parser.parse_args()

    profile = args.profile
    p = PROFILES[profile]

    if args.offer_id:
        # Direct rent with approval
        print(f"  Profile: {p['name']}")
        print(f"  Offer ID: {args.offer_id}")
        print()

        # Fetch offer details for approval
        offers = run_vastai(['search', 'offers', f'id={args.offer_id}'], raw=True)
        if not offers or len(offers) == 0:
            print(f"ERROR: Offer {args.offer_id} not found or not rentable")
            sys.exit(1)

        offer = offers[0]

        # Request approval
        if not request_discord_approval(profile, offer):
            print("  Rental cancelled by user")
            sys.exit(0)

        # Rent
        instance_id = rent_machine(args.offer_id, profile)
        if not instance_id:
            sys.exit(1)

        ssh_host, ssh_port = wait_for_ssh(instance_id)
        print_deploy_commands(instance_id, ssh_host, ssh_port, profile)

    else:
        # Search for offers
        print(f"  Searching vast.ai for '{p['name']}' machines...")
        offers = search_offers(profile)

        if not offers:
            print(f"No machines found matching profile '{profile}'")
            sys.exit(1)

        display_offers(offers, profile)

        if args.search_only:
            print("(--search-only mode, not renting)")
            return

        if not args.rent:
            print("Tip: Use --search-only to browse, or --rent to proceed with rental")
            return

        # Heavy profile: user MUST pick from table (never auto-select #1)
        if profile == 'heavy':
            print("⚠️  15m PROFILE: You must pick a machine personally (memory rule)")
            print()
            while True:
                choice = input(f"Pick machine # (1-{len(offers)}), or 'q' to quit: ").strip()
                if choice.lower() == 'q':
                    print("Cancelled.")
                    return
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(offers):
                        offer = offers[idx]
                        break
                except ValueError:
                    pass
                print(f"Invalid choice. Enter 1-{len(offers)} or 'q'.")
        else:
            # Other profiles: present top 3, user picks or defaults to #1
            print(f"Top 3 machines by CPU Score:")
            for i in range(min(3, len(offers))):
                o = offers[i]
                print(f"  {i+1}. ID {o['id']}: {o.get('gpu_name', '?')} × {o.get('num_gpus', 1)}, "
                      f"Score {o['_cpu_score']:.0f}, ${o['dph_total']:.2f}/hr")
            print()

            choice = input(f"Pick machine # (1-3, or Enter for #1): ").strip()
            if not choice:
                offer = offers[0]
                print(f"  Using fastest: #{offer['id']}")
            else:
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < min(3, len(offers)):
                        offer = offers[idx]
                    else:
                        print("Invalid choice. Aborted.")
                        return
                except ValueError:
                    print("Invalid input. Aborted.")
                    return

        # Request Discord approval
        if not request_discord_approval(profile, offer):
            print("  Rental cancelled")
            return

        # Rent the machine
        instance_id = rent_machine(offer['id'], profile)
        if not instance_id:
            sys.exit(1)

        ssh_host, ssh_port = wait_for_ssh(instance_id)
        print_deploy_commands(instance_id, ssh_host, ssh_port, profile)

        # Log rental to ops_kb
        try:
            subprocess.run([
                'python', 'ops_kb.py', 'add',
                f"MACHINE: Rented {offer['id']} ({profile} profile): "
                f"{offer.get('num_gpus', 1)}x {offer.get('gpu_name', '?')}, "
                f"Score {offer['_cpu_score']:.0f}, ${offer['dph_total']:.2f}/hr, "
                f"instance {instance_id}",
                '--topic', 'deployment'
            ], cwd=Path(__file__).parent, timeout=5)
        except Exception:
            pass


if __name__ == '__main__':
    main()
