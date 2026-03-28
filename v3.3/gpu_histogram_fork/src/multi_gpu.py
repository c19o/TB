#!/usr/bin/env python3
"""
multi_gpu.py -- Multi-GPU parallel timeframe trainer
=====================================================
Orchestrates training different TFs on different GPUs simultaneously.
Each TF gets full CUDA_VISIBLE_DEVICES isolation -- no cross-TF interference.

Example (4x A100):
    GPU 0: training 1w   (3 GB VRAM)
    GPU 1: training 1d   (6 GB VRAM)
    GPU 2: training 4h  (13 GB VRAM)
    GPU 3: training 1h  (26 GB VRAM)

Usage:
    python multi_gpu.py --tfs 1w,1d,4h,1h --gpus 0,1,2,3
    python multi_gpu.py --tfs 1w,1d,4h,1h          # auto-assign
    python multi_gpu.py --tfs 1w,1d,4h,1h --dry-run # show plan only

Each TF runs as a separate subprocess with its own CUDA_VISIBLE_DEVICES.
The full matrix is preserved per TF -- no feature sharing or interference.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from typing import Optional

# ---------------------------------------------------------------------------
# Constants -- VRAM budgets from train_pipeline.py / ARCHITECTURE.md
# Includes CSR resident + histogram pool + gradients + kernel overhead
# ---------------------------------------------------------------------------
TF_VRAM_BUDGET_GB = {
    '1w':   3.0,    #  818 rows x 2.2M features
    '1d':   6.0,    # 5733 rows x 6M features
    '4h':  13.0,    #  23K rows x 4M features
    '1h':  26.0,    # 100K rows x 10M features
    '15m': 41.0,    # 227K rows x 10M features
}

# System RAM needed per TF (from cloud_run_tf.py TF_MIN_RAM)
TF_MIN_RAM_GB = {
    '1w':   64,
    '1d':  192,
    '4h':  768,
    '1h': 1024,
    '15m': 1500,
}

# Ordering: largest TF first (needs most resources, takes longest)
TF_SIZE_ORDER = ['15m', '1h', '4h', '1d', '1w']

# Safety margin -- never allocate more than 85% of VRAM
_VRAM_SAFETY = 0.85

# Valid timeframes (no 5m -- dropped in v3.1)
VALID_TFS = {'1w', '1d', '4h', '1h', '15m'}


# ---------------------------------------------------------------------------
# GPU Detection
# ---------------------------------------------------------------------------

def detect_gpus() -> list[dict]:
    """Detect all CUDA GPUs with VRAM info.

    Returns list of dicts sorted by device ID:
        {id, name, vram_total_gb, vram_free_gb}
    Falls back to nvidia-smi if CuPy unavailable.
    """
    # Try CuPy first (most accurate, matches train_pipeline.py)
    try:
        import cupy as cp
        n = cp.cuda.runtime.getDeviceCount()
        devices = []
        for i in range(n):
            with cp.cuda.Device(i):
                props = cp.cuda.runtime.getDeviceProperties(i)
                free, total = cp.cuda.runtime.memGetInfo()
                name = props['name']
                if isinstance(name, bytes):
                    name = name.decode()
                devices.append({
                    'id': i,
                    'name': name.strip(),
                    'vram_total_gb': total / (1024 ** 3),
                    'vram_free_gb': free / (1024 ** 3),
                })
        return devices
    except Exception:
        pass

    # Fallback: nvidia-smi
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free',
             '--format=csv,noheader,nounits'],
            text=True, timeout=10,
        )
        devices = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 4:
                devices.append({
                    'id': int(parts[0]),
                    'name': parts[1],
                    'vram_total_gb': float(parts[2]) / 1024,
                    'vram_free_gb': float(parts[3]) / 1024,
                })
        return devices
    except Exception:
        pass

    return []


def get_system_ram_gb() -> float:
    """Get total system RAM in GB (cgroup-aware)."""
    try:
        from hardware_detect import get_available_ram_gb
        return get_available_ram_gb()
    except ImportError:
        pass
    try:
        return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024 ** 3)
    except (ValueError, OSError, AttributeError):
        pass
    return 0.0


# ---------------------------------------------------------------------------
# TF-to-GPU Assignment
# ---------------------------------------------------------------------------

def assign_tfs_to_gpus(
    tf_list: list[str],
    gpu_ids: Optional[list[int]] = None,
) -> list[dict]:
    """Assign TFs to GPUs based on VRAM requirements.

    Strategy:
    - Sort TFs by VRAM need (largest first)
    - Sort GPUs by VRAM available (largest first)
    - Assign largest TF to largest GPU
    - Validate each assignment fits within VRAM safety margin

    Returns list of dicts:
        {tf, gpu_id, gpu_name, vram_needed_gb, vram_available_gb, fits}
    """
    devices = detect_gpus()
    if not devices:
        print("ERROR: No GPUs detected. Cannot assign TFs.", file=sys.stderr)
        sys.exit(1)

    # Filter to requested GPU IDs
    if gpu_ids is not None:
        device_map = {d['id']: d for d in devices}
        missing = [g for g in gpu_ids if g not in device_map]
        if missing:
            print(f"ERROR: GPU IDs {missing} not found. Available: "
                  f"{[d['id'] for d in devices]}", file=sys.stderr)
            sys.exit(1)
        devices = [device_map[g] for g in gpu_ids]

    if len(tf_list) > len(devices):
        print(f"ERROR: {len(tf_list)} TFs requested but only {len(devices)} GPUs "
              f"available. Cannot assign 1:1.", file=sys.stderr)
        sys.exit(1)

    # Sort TFs by VRAM need descending (largest TF first)
    sorted_tfs = sorted(tf_list, key=lambda t: TF_VRAM_BUDGET_GB.get(t, 0),
                        reverse=True)

    # Sort GPUs by free VRAM descending (largest GPU first)
    sorted_gpus = sorted(devices, key=lambda d: d['vram_free_gb'], reverse=True)

    assignments = []
    for tf, gpu in zip(sorted_tfs, sorted_gpus):
        vram_needed = TF_VRAM_BUDGET_GB.get(tf, 10.0)
        vram_usable = gpu['vram_free_gb'] * _VRAM_SAFETY
        assignments.append({
            'tf': tf,
            'gpu_id': gpu['id'],
            'gpu_name': gpu['name'],
            'vram_needed_gb': vram_needed,
            'vram_available_gb': gpu['vram_free_gb'],
            'fits': vram_needed <= vram_usable,
        })

    return assignments


def print_assignment_table(assignments: list[dict]) -> None:
    """Print a formatted table of TF -> GPU assignments."""
    print()
    print("=" * 72)
    print(f"  {'TF':<6} {'GPU':<6} {'GPU Name':<22} {'VRAM Need':>10} "
          f"{'VRAM Free':>10} {'Status':>8}")
    print("-" * 72)
    for a in assignments:
        status = "OK" if a['fits'] else "WARNING"
        print(f"  {a['tf']:<6} {a['gpu_id']:<6} {a['gpu_name']:<22} "
              f"{a['vram_needed_gb']:>8.1f}GB {a['vram_available_gb']:>8.1f}GB "
              f"{status:>8}")
    print("=" * 72)

    # RAM check
    total_ram = get_system_ram_gb()
    if total_ram > 0:
        tfs_in_play = [a['tf'] for a in assignments]
        max_ram_needed = max(TF_MIN_RAM_GB.get(tf, 64) for tf in tfs_in_play)
        total_ram_needed = sum(TF_MIN_RAM_GB.get(tf, 64) for tf in tfs_in_play)
        print(f"\n  System RAM: {total_ram:.0f} GB")
        print(f"  Combined TF RAM need: {total_ram_needed} GB "
              f"(largest single: {max_ram_needed} GB)")
        if total_ram < total_ram_needed:
            print(f"  WARNING: System RAM ({total_ram:.0f}GB) < combined need "
                  f"({total_ram_needed}GB).")
            print(f"           TFs will compete for RAM. Consider running fewer "
                  f"in parallel.")
    print()


# ---------------------------------------------------------------------------
# Training Orchestrator
# ---------------------------------------------------------------------------

def train_parallel_tfs(
    tf_list: list[str],
    gpu_ids: Optional[list[int]] = None,
    symbol: str = 'BTC',
    extra_args: Optional[list[str]] = None,
    log_dir: str = '.',
) -> dict[str, int]:
    """Train multiple TFs in parallel on different GPUs.

    Each TF runs as a separate subprocess with:
    - CUDA_VISIBLE_DEVICES set to its assigned GPU
    - stdout/stderr tee'd to {tf}_log.txt
    - Full isolation -- no shared GPU memory

    Args:
        tf_list:    List of timeframes to train (e.g., ['1w', '1d', '4h'])
        gpu_ids:    Explicit GPU IDs (auto-assigns if None)
        symbol:     Trading symbol (default 'BTC')
        extra_args: Additional args passed to cloud_run_tf.py
        log_dir:    Directory for log files (default: current dir)

    Returns:
        Dict mapping TF -> exit code (0 = success)
    """
    assignments = assign_tfs_to_gpus(tf_list, gpu_ids)
    print_assignment_table(assignments)

    # Check for VRAM warnings
    warnings = [a for a in assignments if not a['fits']]
    if warnings:
        print("WARNING: The following TFs may not fit in GPU VRAM:")
        for a in warnings:
            print(f"  {a['tf']}: needs {a['vram_needed_gb']:.1f}GB but "
                  f"GPU {a['gpu_id']} only has {a['vram_available_gb']:.1f}GB free")
        print("  Training will proceed -- LightGBM may fall back to CPU for "
              "these TFs.\n")

    # Resolve cloud_run_tf.py path
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))  # v3.3/
    cloud_script = os.path.join(script_dir, 'cloud_run_tf.py')
    if not os.path.exists(cloud_script):
        # Flat workspace layout (cloud deployment)
        cloud_script = 'cloud_run_tf.py'

    os.makedirs(log_dir, exist_ok=True)

    # Launch all TFs
    processes: list[dict] = []
    t_start = time.time()

    for assignment in assignments:
        tf = assignment['tf']
        gpu_id = assignment['gpu_id']

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        env['PYTHONUNBUFFERED'] = '1'

        cmd = [
            sys.executable, '-u', cloud_script,
            '--tf', tf,
            '--symbol', symbol,
        ]
        if extra_args:
            cmd.extend(extra_args)

        log_path = os.path.join(log_dir, f'{tf}_log.txt')
        log_fh = open(log_path, 'w')

        print(f"  Launching {tf} on GPU {gpu_id} ({assignment['gpu_name']}) "
              f"-> {log_path}")

        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            # Start in same CWD (cloud_run_tf.py will chdir to /workspace)
        )

        processes.append({
            'tf': tf,
            'gpu_id': gpu_id,
            'gpu_name': assignment['gpu_name'],
            'proc': proc,
            'log_path': log_path,
            'log_fh': log_fh,
            'start_time': time.time(),
            'exit_code': None,
        })

    print(f"\n  All {len(processes)} TFs launched. Monitoring...\n")

    # Monitor until all complete
    results = _monitor_training(processes)

    # Summary
    elapsed = time.time() - t_start
    _print_final_summary(results, elapsed)

    return {r['tf']: r['exit_code'] for r in results}


def _monitor_training(processes: list[dict]) -> list[dict]:
    """Monitor running processes, report completions and failures.

    Polls every 15 seconds. When a TF finishes, reports its status.
    Remaining TFs continue regardless of failures (no cross-TF dependency).
    """
    active = list(processes)

    while active:
        still_running = []
        for p in active:
            ret = p['proc'].poll()
            if ret is not None:
                # Process finished
                p['exit_code'] = ret
                p['log_fh'].close()
                elapsed = time.time() - p['start_time']
                status = "DONE" if ret == 0 else f"FAILED (exit {ret})"
                print(f"  [{_fmt_elapsed(elapsed)}] {p['tf']} (GPU {p['gpu_id']}): "
                      f"{status}")
                if ret != 0:
                    # Show last 10 lines of log for failed TFs
                    _show_log_tail(p['log_path'], p['tf'], lines=10)
            else:
                still_running.append(p)

        active = still_running

        if active:
            # Brief status line
            running_tfs = ', '.join(f"{p['tf']}(GPU{p['gpu_id']})" for p in active)
            total_elapsed = time.time() - processes[0]['start_time']
            print(f"  [{_fmt_elapsed(total_elapsed)}] Running: {running_tfs}", end='\r')
            time.sleep(15)

    print()  # Clear the \r line
    return processes


def _show_log_tail(log_path: str, tf: str, lines: int = 10) -> None:
    """Print last N lines of a log file for debugging failed TFs."""
    try:
        with open(log_path, 'r') as f:
            all_lines = f.readlines()
        tail = all_lines[-lines:] if len(all_lines) >= lines else all_lines
        print(f"\n  --- Last {len(tail)} lines of {tf} log ---")
        for line in tail:
            print(f"    {line.rstrip()}")
        print(f"  --- End {tf} log ---\n")
    except Exception as e:
        print(f"  (Could not read {log_path}: {e})")


def _fmt_elapsed(seconds: float) -> str:
    """Format seconds as HH:MM:SS or MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


def _print_final_summary(processes: list[dict], total_elapsed: float) -> None:
    """Print final summary table after all TFs complete."""
    print()
    print("=" * 60)
    print(f"  MULTI-GPU TRAINING COMPLETE  ({_fmt_elapsed(total_elapsed)})")
    print("-" * 60)

    succeeded = []
    failed = []

    for p in processes:
        tf = p['tf']
        ec = p['exit_code']
        elapsed = time.time() - p['start_time']
        status = "OK" if ec == 0 else f"FAIL (exit {ec})"
        print(f"  {tf:<6} GPU {p['gpu_id']:<4} {_fmt_elapsed(elapsed):>10}  {status}")
        if ec == 0:
            succeeded.append(tf)
        else:
            failed.append(tf)

    print("-" * 60)
    print(f"  Succeeded: {len(succeeded)}/{len(processes)} "
          f"({', '.join(succeeded) if succeeded else 'none'})")
    if failed:
        print(f"  Failed:    {len(failed)}/{len(processes)} "
              f"({', '.join(failed)})")
        print(f"\n  Check logs for failed TFs:")
        for p in processes:
            if p['exit_code'] != 0:
                print(f"    {p['tf']}: {p['log_path']}")
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# Graceful shutdown -- forward SIGINT/SIGTERM to children
# ---------------------------------------------------------------------------

_CHILD_PROCS: list[subprocess.Popen] = []


def _signal_handler(sig, frame):
    """Forward termination signal to all child processes."""
    signame = signal.Signals(sig).name
    print(f"\n  Received {signame} -- terminating all TF processes...")
    for p in _CHILD_PROCS:
        try:
            p.terminate()
        except Exception:
            pass
    # Give children 5s to exit gracefully, then kill
    time.sleep(5)
    for p in _CHILD_PROCS:
        try:
            p.kill()
        except Exception:
            pass
    sys.exit(128 + sig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Multi-GPU parallel timeframe trainer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train 4 TFs on 4 GPUs (auto-assign largest TF to largest GPU):
  python multi_gpu.py --tfs 1w,1d,4h,1h

  # Explicit GPU assignment:
  python multi_gpu.py --tfs 1w,1d,4h,1h --gpus 0,1,2,3

  # Just show the assignment plan:
  python multi_gpu.py --tfs 1w,1d,4h,1h --dry-run

  # Train with extra args passed to cloud_run_tf.py:
  python multi_gpu.py --tfs 1w,1d --extra-args "--parallel 0"

  # Monitor existing logs (if processes are already running):
  python multi_gpu.py --monitor-only --tfs 1w,1d,4h,1h
""",
    )
    parser.add_argument(
        '--tfs', required=True,
        help='Comma-separated list of timeframes (e.g., 1w,1d,4h,1h,15m)',
    )
    parser.add_argument(
        '--gpus', default=None,
        help='Comma-separated GPU IDs (default: auto-assign)',
    )
    parser.add_argument(
        '--symbol', default='BTC',
        help='Trading symbol (default: BTC)',
    )
    parser.add_argument(
        '--log-dir', default='.',
        help='Directory for per-TF log files (default: current dir)',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Show assignment plan without launching training',
    )
    parser.add_argument(
        '--extra-args', default=None,
        help='Extra arguments to pass to cloud_run_tf.py (quoted string)',
    )
    parser.add_argument(
        '--tail', default=None, type=str,
        help='Tail a specific TF log file (e.g., --tail 1w)',
    )

    args = parser.parse_args()

    # Parse TFs
    tf_list = [t.strip() for t in args.tfs.split(',')]
    invalid = [t for t in tf_list if t not in VALID_TFS]
    if invalid:
        print(f"ERROR: Invalid timeframes: {invalid}. "
              f"Valid: {sorted(VALID_TFS)}", file=sys.stderr)
        sys.exit(1)

    # Parse GPU IDs
    gpu_ids = None
    if args.gpus:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(',')]

    # Tail mode -- just tail a log file
    if args.tail:
        log_path = os.path.join(args.log_dir, f'{args.tail}_log.txt')
        if not os.path.exists(log_path):
            print(f"ERROR: Log file not found: {log_path}", file=sys.stderr)
            sys.exit(1)
        os.execvp('tail', ['tail', '-f', log_path])
        return

    # Dry run -- show plan only
    if args.dry_run:
        print("\n  DRY RUN -- showing assignment plan only\n")
        assignments = assign_tfs_to_gpus(tf_list, gpu_ids)
        print_assignment_table(assignments)

        # Show what commands would be run
        script_dir = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        cloud_script = os.path.join(script_dir, 'cloud_run_tf.py')
        print("  Commands that would be launched:")
        for a in assignments:
            cmd = (f"  CUDA_VISIBLE_DEVICES={a['gpu_id']} "
                   f"python -u {cloud_script} --tf {a['tf']} "
                   f"--symbol {args.symbol}")
            print(f"    {cmd}")
        print()
        return

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Parse extra args
    extra = args.extra_args.split() if args.extra_args else None

    # Launch training
    print(f"\n  Multi-GPU Training: {len(tf_list)} TFs on "
          f"{len(gpu_ids) if gpu_ids else 'auto'} GPUs")
    print(f"  Symbol: {args.symbol}")
    print(f"  Log dir: {os.path.abspath(args.log_dir)}\n")

    results = train_parallel_tfs(
        tf_list=tf_list,
        gpu_ids=gpu_ids,
        symbol=args.symbol,
        extra_args=extra,
        log_dir=args.log_dir,
    )

    # Track child procs for signal handler
    # (already done in train_parallel_tfs, but register them here too
    #  for the signal handler to reference)

    # Exit with non-zero if any TF failed
    if any(code != 0 for code in results.values()):
        sys.exit(1)


if __name__ == '__main__':
    main()
