#!/usr/bin/env python
"""
hardware_detect.py — Shared hardware detection for V2 pipeline
===============================================================
Importable by all scripts: build, train, optimize, LSTM, cloud runner.
Auto-detects GPUs, VRAM, system RAM, CPU — no hardcoded values.
"""

import os, subprocess


def detect_hardware() -> dict:
    """Detect all available hardware. Returns dict with:
        n_gpus, gpu_names, vram_per_gpu_gb, total_vram_gb,
        total_ram_gb, available_ram_gb, cpu_count, cpu_ghz
    """
    hw = {
        'n_gpus': 0,
        'gpu_names': [],
        'vram_per_gpu_gb': [],
        'total_vram_gb': 0.0,
        'total_ram_gb': 0.0,
        'available_ram_gb': 0.0,
        'cpu_count': os.cpu_count() or 1,
        'cpu_ghz': None,
    }

    # ── GPU detection via nvidia-smi ──
    try:
        nv = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )
        if nv.returncode == 0:
            for line in nv.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 2:
                    hw['gpu_names'].append(parts[0])
                    vram_mb = float(parts[1])
                    hw['vram_per_gpu_gb'].append(round(vram_mb / 1024, 1))
            hw['n_gpus'] = len(hw['gpu_names'])
            hw['total_vram_gb'] = round(sum(hw['vram_per_gpu_gb']), 1)
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        # Fallback: try CuPy
        try:
            import cupy as cp
            n = cp.cuda.runtime.getDeviceCount()
            for i in range(n):
                d = cp.cuda.Device(i)
                mem = d.mem_info
                vram_gb = round(mem[1] / (1024**3), 1)
                hw['gpu_names'].append(f'GPU {i}')
                hw['vram_per_gpu_gb'].append(vram_gb)
            hw['n_gpus'] = n
            hw['total_vram_gb'] = round(sum(hw['vram_per_gpu_gb']), 1)
        except Exception:
            pass

    # ── RAM detection ──
    try:
        import psutil
        vm = psutil.virtual_memory()
        hw['total_ram_gb'] = round(vm.total / (1024**3), 1)
        hw['available_ram_gb'] = round(vm.available / (1024**3), 1)
    except ImportError:
        try:
            with open('/proc/meminfo') as f:
                for line in f:
                    if 'MemTotal' in line:
                        hw['total_ram_gb'] = round(int(line.split()[1]) / (1024**2), 1)
                    elif 'MemAvailable' in line:
                        hw['available_ram_gb'] = round(int(line.split()[1]) / (1024**2), 1)
        except (FileNotFoundError, Exception):
            hw['total_ram_gb'] = 32.0
            hw['available_ram_gb'] = 16.0

    # ── CPU frequency ──
    try:
        import psutil
        freq = psutil.cpu_freq()
        if freq and freq.max > 0:
            hw['cpu_ghz'] = round(freq.max / 1000, 2)
    except Exception:
        pass

    return hw


def get_min_vram_gb(hw: dict) -> float:
    """Return smallest GPU VRAM (bottleneck for batch sizing)."""
    if hw['vram_per_gpu_gb']:
        return min(hw['vram_per_gpu_gb'])
    return 0.0


def get_right_chunk(hw: dict) -> int:
    """Compute optimal RIGHT_CHUNK for cross generator based on RAM + VRAM."""
    # Env var override (for OOM retry from orchestrator)
    env_val = os.environ.get('V2_RIGHT_CHUNK')
    if env_val:
        return int(env_val)

    ram_gb = hw['total_ram_gb']
    if ram_gb >= 512:
        ram_chunk = 2000
    elif ram_gb >= 256:
        ram_chunk = 1000
    elif ram_gb >= 128:
        ram_chunk = 500
    elif ram_gb >= 64:
        ram_chunk = 200
    else:
        ram_chunk = 100

    # Also cap by VRAM if GPUs present
    min_vram = get_min_vram_gb(hw)
    if min_vram > 0:
        vram_chunk = int(min_vram * 80)
        return min(ram_chunk, vram_chunk)
    return ram_chunk


def get_gpu_batch(hw: dict, n_bars: int, n_right: int) -> int:
    """Compute optimal GPU batch for cross generation."""
    min_vram = get_min_vram_gb(hw)
    if min_vram <= 0:
        return 10
    available_bytes = min_vram * 0.7 * (1024**3)
    bytes_per_elem = n_bars * n_right * 4
    if bytes_per_elem <= 0:
        return 10
    return max(1, int(available_bytes / bytes_per_elem))


def log_hardware(hw: dict):
    """Print formatted hardware summary."""
    print("=" * 60, flush=True)
    print("  HARDWARE DETECTION", flush=True)
    print("=" * 60, flush=True)
    print(f"  GPUs:  {hw['n_gpus']}", flush=True)
    for i, (name, vram) in enumerate(zip(hw['gpu_names'], hw['vram_per_gpu_gb'])):
        print(f"    [{i}] {name} — {vram} GB VRAM", flush=True)
    print(f"  Total VRAM: {hw['total_vram_gb']} GB", flush=True)
    print(f"  System RAM: {hw['total_ram_gb']} GB (available: {hw['available_ram_gb']} GB)", flush=True)
    print(f"  CPUs: {hw['cpu_count']} cores", flush=True)
    if hw['cpu_ghz']:
        print(f"  CPU freq: {hw['cpu_ghz']} GHz", flush=True)
    print("=" * 60, flush=True)


if __name__ == '__main__':
    hw = detect_hardware()
    log_hardware(hw)
    print(f"\nRecommended RIGHT_CHUNK: {get_right_chunk(hw)}")
    print(f"Min VRAM: {get_min_vram_gb(hw)} GB")
