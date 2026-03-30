#!/usr/bin/env python3
"""
cuda_compat.py -- CUDA compatibility layer for diverse GPU environments
========================================================================
Handles wildly different driver/CUDA combos across vast.ai, RunPod, GCP, etc.

Our fork is compiled with CUDA 12.6, SM targets 80;86;89;90 + PTX fallback.
This module detects the runtime, verifies compatibility, installs matching
CuPy, and configures the environment for optimal operation.

Driver compatibility matrix:
  Driver 525  -> CUDA 12.0 max
  Driver 535  -> CUDA 12.2 max
  Driver 545  -> CUDA 12.3 max
  Driver 550  -> CUDA 12.4 max
  Driver 560  -> CUDA 12.6 max
  Driver 570+ -> CUDA 12.8+ max
  Driver 580+ -> CUDA 13.0 -- cuDF breaks!
"""

import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

# The CUDA version our fork was compiled against
COMPILED_CUDA_VERSION = (12, 6)
# SM architectures we have native SASS for (PTX fallback for newer)
COMPILED_SM_TARGETS = [80, 86, 89, 90]

# Driver -> max CUDA version mapping (from NVIDIA compatibility table)
DRIVER_CUDA_MAP = {
    525: (12, 0),
    535: (12, 2),
    545: (12, 3),
    550: (12, 4),
    555: (12, 5),
    560: (12, 6),
    565: (12, 7),
    570: (12, 8),
    575: (12, 9),
    580: (13, 0),
    585: (13, 1),
}


def _run_cmd(cmd: str, timeout: int = 10) -> str:
    """Run shell command, return stdout or empty string on failure."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def _parse_version_tuple(ver_str: str) -> Optional[Tuple[int, ...]]:
    """Parse '12.6.1' or '12.6' into (12, 6, 1) or (12, 6)."""
    m = re.search(r'(\d+)\.(\d+)(?:\.(\d+))?', ver_str)
    if m:
        parts = [int(m.group(1)), int(m.group(2))]
        if m.group(3) is not None:
            parts.append(int(m.group(3)))
        return tuple(parts)
    return None


def _driver_major(driver_str: str) -> Optional[int]:
    """Extract major version from driver string like '560.35.03'."""
    m = re.match(r'(\d+)', driver_str)
    return int(m.group(1)) if m else None


def _max_cuda_for_driver(driver_major: int) -> Tuple[int, int]:
    """Look up max CUDA version supported by a driver major version.
    Uses the closest known entry at or below the given driver major.
    """
    best_driver = None
    for d in sorted(DRIVER_CUDA_MAP.keys()):
        if d <= driver_major:
            best_driver = d
    if best_driver is not None:
        return DRIVER_CUDA_MAP[best_driver]
    # Unknown old driver -- assume CUDA 11.8 minimum
    return (11, 8)


# ---------------------------------------------------------------------------
# 1. detect_cuda_env()
# ---------------------------------------------------------------------------

def detect_cuda_env() -> Dict:
    """Detect full CUDA environment. Returns dict with:
        driver_version       - str, e.g. '560.35.03'
        driver_major         - int, e.g. 560
        max_cuda_version     - tuple, e.g. (12, 6)
        nvcc_version         - str or None
        cudart_version       - tuple or None (runtime)
        gpu_name             - str
        gpu_vram_mb          - int
        gpu_vram_gb          - float
        compute_capability   - tuple, e.g. (8, 6)
        sm_count             - int
        n_gpus               - int
        gpus                 - list of per-GPU dicts
        cupy_version         - str or None
        cupy_cuda_version    - str or None
    """
    env = {
        'driver_version': None,
        'driver_major': None,
        'max_cuda_version': None,
        'nvcc_version': None,
        'cudart_version': None,
        'gpu_name': None,
        'gpu_vram_mb': 0,
        'gpu_vram_gb': 0.0,
        'compute_capability': None,
        'sm_count': 0,
        'n_gpus': 0,
        'gpus': [],
        'cupy_version': None,
        'cupy_cuda_version': None,
    }

    # -- Strategy 1: pynvml (most reliable) --
    _detect_via_pynvml(env)

    # -- Strategy 2: nvidia-smi (fallback) --
    if env['driver_version'] is None:
        _detect_via_nvidia_smi(env)

    # -- Strategy 3: /proc/driver/nvidia (last resort on Linux) --
    if env['driver_version'] is None:
        _detect_via_proc(env)

    # Derive max CUDA from driver
    if env['driver_major'] is not None:
        env['max_cuda_version'] = _max_cuda_for_driver(env['driver_major'])

    # -- nvcc version --
    nvcc_raw = _run_cmd("nvcc --version 2>/dev/null")
    if nvcc_raw:
        for line in nvcc_raw.split('\n'):
            if 'release' in line.lower():
                m = re.search(r'release\s+(\d+\.\d+)', line, re.IGNORECASE)
                if m:
                    env['nvcc_version'] = m.group(1)
                break

    # -- CUDA runtime version (via CuPy or ctypes) --
    env['cudart_version'] = _detect_cudart_version()

    # -- CuPy detection --
    try:
        import cupy as cp
        env['cupy_version'] = cp.__version__

        # Detect which CUDA CuPy was built against
        try:
            cuda_ver = cp.cuda.runtime.runtimeGetVersion()
            major = cuda_ver // 1000
            minor = (cuda_ver % 1000) // 10
            env['cupy_cuda_version'] = f"{major}.{minor}"
        except Exception:
            pass
    except ImportError:
        pass
    except Exception:
        pass

    return env


def _detect_via_pynvml(env: Dict):
    """Populate env dict using pynvml (NVIDIA Management Library bindings)."""
    try:
        import pynvml
        pynvml.nvmlInit()

        # Driver version
        driver = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode('utf-8')
        env['driver_version'] = driver
        env['driver_major'] = _driver_major(driver)

        # GPU enumeration
        n_gpus = pynvml.nvmlDeviceGetCount()
        env['n_gpus'] = n_gpus
        gpus = []

        for i in range(n_gpus):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_mb = mem_info.total // (1024 * 1024)

            try:
                cc_major = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                # Returns (major, minor) tuple on newer pynvml
                if isinstance(cc_major, tuple):
                    cc = cc_major
                else:
                    cc_minor_val = 0
                    cc = (cc_major, cc_minor_val)
            except (AttributeError, pynvml.NVMLError):
                cc = None

            try:
                sm = pynvml.nvmlDeviceGetNumGpuCores(handle)
            except (AttributeError, pynvml.NVMLError):
                sm = 0

            gpu = {
                'index': i,
                'name': name,
                'vram_mb': vram_mb,
                'vram_gb': round(vram_mb / 1024, 1),
                'compute_capability': cc,
                'sm_count': sm,
            }
            gpus.append(gpu)

        env['gpus'] = gpus
        if gpus:
            env['gpu_name'] = gpus[0]['name']
            env['gpu_vram_mb'] = gpus[0]['vram_mb']
            env['gpu_vram_gb'] = gpus[0]['vram_gb']
            env['compute_capability'] = gpus[0]['compute_capability']
            env['sm_count'] = gpus[0]['sm_count']

        pynvml.nvmlShutdown()
    except ImportError:
        pass
    except Exception:
        pass


def _detect_via_nvidia_smi(env: Dict):
    """Populate env dict using nvidia-smi CLI."""
    # GPU info
    raw = _run_cmd(
        "nvidia-smi --query-gpu=index,name,memory.total,driver_version "
        "--format=csv,noheader,nounits"
    )
    if not raw:
        return

    gpus = []
    for line in raw.strip().split('\n'):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 4:
            idx = int(parts[0])
            name = parts[1]
            try:
                vram_mb = int(float(parts[2]))
            except ValueError:
                vram_mb = 0
            driver = parts[3]

            if env['driver_version'] is None:
                env['driver_version'] = driver
                env['driver_major'] = _driver_major(driver)

            gpu = {
                'index': idx,
                'name': name,
                'vram_mb': vram_mb,
                'vram_gb': round(vram_mb / 1024, 1),
                'compute_capability': None,
                'sm_count': 0,
            }
            gpus.append(gpu)

    env['gpus'] = gpus
    env['n_gpus'] = len(gpus)
    if gpus:
        env['gpu_name'] = gpus[0]['name']
        env['gpu_vram_mb'] = gpus[0]['vram_mb']
        env['gpu_vram_gb'] = gpus[0]['vram_gb']

    # Max CUDA from nvidia-smi header
    header = _run_cmd("nvidia-smi")
    if header:
        for line in header.split('\n'):
            if 'CUDA Version' in line:
                m = re.search(r'CUDA Version:\s*(\d+\.\d+)', line)
                if m:
                    ver = _parse_version_tuple(m.group(1))
                    if ver:
                        env['max_cuda_version'] = ver[:2]
                break

    # Compute capability via CuPy if nvidia-smi doesn't give it
    try:
        import cupy as cp
        for gpu in env['gpus']:
            dev = cp.cuda.Device(gpu['index'])
            attrs = dev.attributes
            cc_maj = attrs.get('ComputeCapabilityMajor', 0)
            cc_min = attrs.get('ComputeCapabilityMinor', 0)
            if cc_maj > 0:
                gpu['compute_capability'] = (cc_maj, cc_min)
                gpu['sm_count'] = attrs.get('MultiProcessorCount', 0)
        if env['gpus']:
            env['compute_capability'] = env['gpus'][0]['compute_capability']
            env['sm_count'] = env['gpus'][0]['sm_count']
    except Exception:
        pass


def _detect_via_proc(env: Dict):
    """Last resort: read /proc/driver/nvidia/version on Linux."""
    try:
        with open('/proc/driver/nvidia/version', 'r') as f:
            content = f.read()
        m = re.search(r'Kernel Module\s+(\d+\.\d+\.\d+)', content)
        if m:
            env['driver_version'] = m.group(1)
            env['driver_major'] = _driver_major(m.group(1))
    except (FileNotFoundError, PermissionError):
        pass


def _detect_cudart_version() -> Optional[Tuple[int, int]]:
    """Detect CUDA runtime version via CuPy or ctypes."""
    # Try CuPy first
    try:
        import cupy as cp
        ver = cp.cuda.runtime.runtimeGetVersion()
        return (ver // 1000, (ver % 1000) // 10)
    except Exception:
        pass

    # Try ctypes + libcudart
    try:
        import ctypes
        import ctypes.util
        libname = ctypes.util.find_library('cudart')
        if libname:
            lib = ctypes.CDLL(libname)
            ver = ctypes.c_int()
            lib.cudaRuntimeGetVersion(ctypes.byref(ver))
            v = ver.value
            return (v // 1000, (v % 1000) // 10)
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# 2. install_matching_cupy()
# ---------------------------------------------------------------------------

def install_matching_cupy(env: Optional[Dict] = None, force: bool = False) -> bool:
    """Install the CuPy package matching the detected CUDA version.

    Args:
        env: Pre-detected env dict (calls detect_cuda_env() if None).
        force: If True, reinstall even if CuPy already present.

    Returns:
        True if CuPy is available after this call, False otherwise.
    """
    if env is None:
        env = detect_cuda_env()

    # Already installed and working?
    if not force and env.get('cupy_version'):
        try:
            import cupy as cp
            _ = cp.zeros(1)  # Verify it actually works (not just importable)
            print(f"[cuda_compat] CuPy {env['cupy_version']} already working", flush=True)
            return True
        except Exception:
            pass  # Installed but broken -- reinstall

    max_cuda = env.get('max_cuda_version')
    if max_cuda is None:
        print("[cuda_compat] No CUDA detected -- skipping CuPy install", flush=True)
        return False

    major, minor = max_cuda[0], max_cuda[1]

    if major >= 13:
        # CUDA 13.0+ -- cupy-cuda12x may still work via compat, try it
        pkg = "cupy-cuda12x"
        print(f"[cuda_compat] WARNING: CUDA {major}.{minor} detected (driver 580+). "
              f"Trying {pkg} -- cuDF will NOT work.", flush=True)
    elif major == 12:
        pkg = "cupy-cuda12x"
    elif major == 11:
        pkg = "cupy-cuda11x"
    else:
        print(f"[cuda_compat] CUDA {major}.{minor} too old for CuPy", flush=True)
        return False

    print(f"[cuda_compat] Installing {pkg} for CUDA {major}.{minor}...", flush=True)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", pkg],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0:
            print(f"[cuda_compat] {pkg} installed successfully", flush=True)
            # Verify
            try:
                import importlib
                if 'cupy' in sys.modules:
                    importlib.reload(sys.modules['cupy'])
                else:
                    import cupy
                return True
            except Exception as e:
                print(f"[cuda_compat] {pkg} installed but import failed: {e}", flush=True)
                return False
        else:
            print(f"[cuda_compat] pip install failed: {result.stderr[:500]}", flush=True)
            return False
    except subprocess.TimeoutExpired:
        print("[cuda_compat] pip install timed out (300s)", flush=True)
        return False
    except Exception as e:
        print(f"[cuda_compat] install error: {e}", flush=True)
        return False


# ---------------------------------------------------------------------------
# 3. check_fork_compatibility()
# ---------------------------------------------------------------------------

def check_fork_compatibility(env: Optional[Dict] = None) -> Tuple[bool, str]:
    """Verify our compiled fork .so/.dll works on this machine.

    Checks:
      1. CUDA runtime >= compiled version (12.6) -- forward compat via PTX
      2. Compute capability has native SASS or PTX fallback
      3. Driver supports the CUDA runtime we need

    Returns:
        (compatible: bool, reason: str)
    """
    if env is None:
        env = detect_cuda_env()

    # No GPU at all
    if env['n_gpus'] == 0:
        return False, "No NVIDIA GPU detected"

    # Check driver supports CUDA 12.6+
    max_cuda = env.get('max_cuda_version')
    if max_cuda is None:
        return False, "Cannot determine max CUDA version from driver"

    compiled = COMPILED_CUDA_VERSION  # (12, 6)
    if max_cuda < compiled:
        return False, (
            f"Driver {env['driver_version']} supports max CUDA {max_cuda[0]}.{max_cuda[1]}, "
            f"but fork was compiled with CUDA {compiled[0]}.{compiled[1]}. "
            f"Need driver >= 560."
        )

    # Check CUDA runtime (if detected)
    cudart = env.get('cudart_version')
    if cudart is not None and cudart < compiled:
        return False, (
            f"CUDA runtime {cudart[0]}.{cudart[1]} < compiled {compiled[0]}.{compiled[1]}. "
            f"Install CUDA toolkit >= {compiled[0]}.{compiled[1]} or use a container with it."
        )

    # Check compute capability
    cc = env.get('compute_capability')
    if cc is not None:
        sm = cc[0] * 10 + cc[1]
        has_native = sm in COMPILED_SM_TARGETS
        # PTX fallback works for any GPU with CC >= lowest compiled target
        min_sm = min(COMPILED_SM_TARGETS)
        if sm < min_sm:
            return False, (
                f"GPU compute capability {cc[0]}.{cc[1]} (SM {sm}) is below "
                f"minimum compiled target SM {min_sm}. Needs Ampere (SM 80) or newer."
            )

        if has_native:
            note = f"Native SASS for SM {sm}"
        else:
            note = f"PTX fallback for SM {sm} (native: {COMPILED_SM_TARGETS})"
    else:
        note = "Compute capability unknown -- assuming PTX compat"

    # CUDA 13.0+ warning
    if max_cuda >= (13, 0):
        return True, (
            f"COMPATIBLE with warning: Driver {env['driver_version']} "
            f"supports CUDA {max_cuda[0]}.{max_cuda[1]}. "
            f"Fork compiled for 12.6 will use compat mode. "
            f"cuDF will NOT work. {note}"
        )

    return True, f"COMPATIBLE: {note}. Driver {env['driver_version']}, max CUDA {max_cuda[0]}.{max_cuda[1]}"


# ---------------------------------------------------------------------------
# 4. get_best_gpu()
# ---------------------------------------------------------------------------

def get_best_gpu(
    min_vram_gb: float = 0,
    env: Optional[Dict] = None,
) -> Optional[Dict]:
    """Select the best GPU on multi-GPU machines.

    Selection priority:
      1. Filter by min_vram_gb
      2. Prefer highest compute capability (newer arch)
      3. Tie-break by most VRAM

    Args:
        min_vram_gb: Minimum VRAM in GB to consider.
        env: Pre-detected env dict (calls detect_cuda_env() if None).

    Returns:
        GPU dict with index/name/vram_gb/compute_capability, or None.
    """
    if env is None:
        env = detect_cuda_env()

    gpus = env.get('gpus', [])
    if not gpus:
        return None

    # Filter by VRAM
    candidates = [g for g in gpus if g['vram_gb'] >= min_vram_gb]
    if not candidates:
        return None

    def _sort_key(g):
        cc = g.get('compute_capability') or (0, 0)
        sm = cc[0] * 10 + cc[1]
        vram = g.get('vram_gb', 0)
        return (sm, vram)

    candidates.sort(key=_sort_key, reverse=True)
    return candidates[0]


def get_all_gpus(
    min_vram_gb: float = 0,
    env: Optional[Dict] = None,
) -> List[Dict]:
    """Return all GPUs meeting the VRAM threshold, sorted best-first."""
    if env is None:
        env = detect_cuda_env()

    gpus = env.get('gpus', [])
    candidates = [g for g in gpus if g['vram_gb'] >= min_vram_gb]

    def _sort_key(g):
        cc = g.get('compute_capability') or (0, 0)
        sm = cc[0] * 10 + cc[1]
        vram = g.get('vram_gb', 0)
        return (sm, vram)

    candidates.sort(key=_sort_key, reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# 5. setup_cuda_env()
# ---------------------------------------------------------------------------

def setup_cuda_env(
    env: Optional[Dict] = None,
    gpu_index: Optional[int] = None,
    verbose: bool = True,
) -> Dict:
    """Configure environment variables for CUDA operation.

    Sets:
      - CUDA_VISIBLE_DEVICES (if gpu_index specified, otherwise leave all visible)
      - LD_LIBRARY_PATH / PATH for CUDA libs
      - Workarounds for known driver/CUDA issues

    Args:
        env: Pre-detected env dict.
        gpu_index: Pin to specific GPU (None = all visible).
        verbose: Print what was set.

    Returns:
        Dict of env vars that were set.
    """
    if env is None:
        env = detect_cuda_env()

    changes = {}

    # -- CUDA_VISIBLE_DEVICES --
    if gpu_index is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_index)
        changes['CUDA_VISIBLE_DEVICES'] = str(gpu_index)
    # If not specified, do NOT override -- let all GPUs be visible
    # (per CLAUDE.md: "Do NOT pin CUDA_VISIBLE_DEVICES per-process")

    # -- LD_LIBRARY_PATH (Linux) --
    if sys.platform != 'win32':
        cuda_lib_paths = [
            '/usr/local/cuda/lib64',
            '/usr/local/cuda/extras/CUPTI/lib64',
            '/usr/lib/x86_64-linux-gnu',
        ]
        # Also check for toolkit-specific paths
        for ver in ['12.6', '12.4', '12.2', '12.0', '11.8']:
            cuda_lib_paths.append(f'/usr/local/cuda-{ver}/lib64')

        existing_ld = os.environ.get('LD_LIBRARY_PATH', '')
        new_paths = []
        for p in cuda_lib_paths:
            if os.path.isdir(p) and p not in existing_ld:
                new_paths.append(p)

        if new_paths:
            combined = ':'.join(new_paths)
            if existing_ld:
                combined = combined + ':' + existing_ld
            os.environ['LD_LIBRARY_PATH'] = combined
            changes['LD_LIBRARY_PATH'] = combined
    else:
        # Windows: ensure CUDA bin is on PATH
        cuda_paths = [
            r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin',
            r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin',
        ]
        existing_path = os.environ.get('PATH', '')
        added = []
        for p in cuda_paths:
            if os.path.isdir(p) and p not in existing_path:
                added.append(p)
                break  # Only add the first one found
        if added:
            os.environ['PATH'] = ';'.join(added) + ';' + existing_path
            changes['PATH'] = os.environ['PATH']

    # -- Workaround: CUDA 13.0+ (driver 580+) breaks cuDF --
    max_cuda = env.get('max_cuda_version')
    if max_cuda and max_cuda >= (13, 0):
        os.environ['SAVAGE22_NO_CUDF'] = '1'
        changes['SAVAGE22_NO_CUDF'] = '1'
        if verbose:
            print(
                f"[cuda_compat] WARNING: CUDA {max_cuda[0]}.{max_cuda[1]} "
                f"(driver 580+) -- cuDF disabled, using pandas CPU for feature builds.",
                flush=True,
            )

    # -- Workaround: MPS (Multi-Process Service) can cause issues with our fork --
    # Disable if running as non-root and MPS is active
    if sys.platform != 'win32':
        mps_check = _run_cmd("nvidia-smi -q -d COMPUTE 2>/dev/null | grep -i 'MPS'")
        if 'Enabled' in mps_check:
            os.environ['CUDA_MPS_PIPE_DIRECTORY'] = ''
            changes['CUDA_MPS_PIPE_DIRECTORY'] = '(disabled)'
            if verbose:
                print("[cuda_compat] MPS detected and disabled for fork compatibility", flush=True)

    # -- Workaround: ECC memory overhead (reduce reported VRAM) --
    # Just informational -- no env change needed

    # -- tcmalloc preload for memory allocator perf (Linux) --
    if sys.platform != 'win32':
        tcmalloc_paths = [
            '/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4',
            '/usr/lib/libtcmalloc_minimal.so.4',
            '/usr/lib64/libtcmalloc_minimal.so.4',
        ]
        for p in tcmalloc_paths:
            if os.path.isfile(p):
                existing_preload = os.environ.get('LD_PRELOAD', '')
                if 'tcmalloc' not in existing_preload:
                    new_preload = p if not existing_preload else f"{p}:{existing_preload}"
                    os.environ['LD_PRELOAD'] = new_preload
                    changes['LD_PRELOAD'] = new_preload
                    if verbose:
                        print(f"[cuda_compat] tcmalloc preloaded: {p}", flush=True)
                break

    # -- PYTHONMALLOC=malloc routes Python allocs through tcmalloc too --
    if 'LD_PRELOAD' in os.environ and 'tcmalloc' in os.environ.get('LD_PRELOAD', ''):
        os.environ['PYTHONMALLOC'] = 'malloc'
        changes['PYTHONMALLOC'] = 'malloc'

    if verbose and changes:
        print("[cuda_compat] Environment configured:", flush=True)
        for k, v in changes.items():
            display = v if len(str(v)) < 120 else str(v)[:117] + '...'
            print(f"  {k}={display}", flush=True)

    return changes


# ---------------------------------------------------------------------------
# Convenience: full setup in one call
# ---------------------------------------------------------------------------

def auto_setup(
    min_vram_gb: float = 0,
    install_cupy: bool = True,
    verbose: bool = True,
) -> Dict:
    """One-call setup: detect, verify compatibility, install CuPy, configure env.

    Returns dict with:
        env          - full detect_cuda_env() result
        compatible   - bool
        reason       - compatibility message
        best_gpu     - best GPU dict or None
        env_changes  - dict of env vars that were set
        cupy_ok      - bool
    """
    env = detect_cuda_env()

    if verbose:
        print("=" * 60, flush=True)
        print("  CUDA COMPATIBILITY CHECK", flush=True)
        print("=" * 60, flush=True)
        if env['driver_version']:
            print(f"  Driver:    {env['driver_version']}", flush=True)
        if env['max_cuda_version']:
            mc = env['max_cuda_version']
            print(f"  Max CUDA:  {mc[0]}.{mc[1]}", flush=True)
        if env['nvcc_version']:
            print(f"  nvcc:      {env['nvcc_version']}", flush=True)
        if env['cudart_version']:
            cr = env['cudart_version']
            print(f"  Runtime:   {cr[0]}.{cr[1]}", flush=True)
        print(f"  GPUs:      {env['n_gpus']}", flush=True)
        for g in env.get('gpus', []):
            cc_str = f"SM {g['compute_capability'][0]}.{g['compute_capability'][1]}" if g.get('compute_capability') else "CC unknown"
            print(f"    [{g['index']}] {g['name']} -- {g['vram_gb']} GB, {cc_str}", flush=True)

    compatible, reason = check_fork_compatibility(env)

    if verbose:
        status = "PASS" if compatible else "FAIL"
        print(f"\n  Fork compat: {status}", flush=True)
        print(f"  {reason}", flush=True)

    best = get_best_gpu(min_vram_gb=min_vram_gb, env=env)

    cupy_ok = False
    if install_cupy and compatible:
        cupy_ok = install_matching_cupy(env)

    env_changes = setup_cuda_env(env, verbose=verbose)

    if verbose:
        print("=" * 60, flush=True)

    return {
        'env': env,
        'compatible': compatible,
        'reason': reason,
        'best_gpu': best,
        'env_changes': env_changes,
        'cupy_ok': cupy_ok,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import json

    result = auto_setup(verbose=True)

    # Also dump as JSON for scripting
    print("\n--- JSON dump ---", flush=True)
    # Make env JSON-serializable (tuples -> lists)
    def _serialize(obj):
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, set):
            return list(obj)
        raise TypeError(f"Not serializable: {type(obj)}")

    print(json.dumps(result['env'], indent=2, default=_serialize), flush=True)
