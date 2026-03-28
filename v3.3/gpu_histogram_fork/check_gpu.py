#!/usr/bin/env python3
"""
check_gpu.py -- GPU Capability Reporter & Training Speedup Estimator
====================================================================
Reports GPU hardware, CUDA stack, CuPy/cuSPARSE availability, and estimates
per-timeframe training speedup for the Savage22 sparse histogram fork.

Usage:
    python check_gpu.py              # Full report
    python check_gpu.py --bench      # Report + cuSPARSE SpMV benchmark
    python check_gpu.py --json       # Machine-readable JSON output
    python check_gpu.py --tf 1d      # Detailed estimate for one TF
"""
import argparse
import json as json_mod
import subprocess
import sys
import time

import numpy as np
import scipy.sparse as sp

# ── TF Profiles (from real v3.3 training data) ──
TF_PROFILES = [
    {"tf": "1w",  "rows":    818, "features": 2_200_000, "csr_gb": 2.0,  "cpu_fold_min": 2},
    {"tf": "1d",  "rows":  5_727, "features": 6_000_000, "csr_gb": 5.0,  "cpu_fold_min": 15},
    {"tf": "4h",  "rows": 22_908, "features": 4_000_000, "csr_gb": 12.0, "cpu_fold_min": 45},
    {"tf": "1h",  "rows": 91_632, "features":10_000_000, "csr_gb": 25.0, "cpu_fold_min": 180},
    {"tf": "15m", "rows":227_000, "features":10_000_000, "csr_gb": 40.0, "cpu_fold_min": 600},
]

# GPU memory bandwidth (GB/s) -- for Amdahl speedup estimation
# Histogram building is memory-bandwidth bound (SpMV kernel)
GPU_BANDWIDTH = {
    'RTX 3090':    936,
    'RTX 4090':   1008,
    'A100':       2039,
    'A100 80GB':  2039,
    'H100':       3350,
    'H100 80GB':  3350,
    'H200':       4800,
    'L40S':       864,
    'L40':        864,
    'A40':        696,
    'A30':        933,
    'V100':       900,
    'T4':         320,
    'RTX 4080':   717,
    'RTX 4070':   504,
    'B200':       8000,
}
CPU_BW_REF = 100  # GB/s (AMD EPYC 7763 128c reference)


def _run_cmd(cmd: str) -> str:
    """Run shell command, return stdout or empty string on failure."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return result.stdout.strip()
    except Exception:
        return ""


def detect_gpu():
    """Detect GPU model, VRAM, compute capability, SM count via nvidia-smi."""
    info = {
        "model": "Not detected",
        "vram_mb": 0,
        "vram_gb": 0.0,
        "compute_cap": "N/A",
        "sm_count": 0,
        "driver_version": "N/A",
        "cuda_driver_version": "N/A",
        "gpu_count": 0,
    }

    # nvidia-smi basic info
    raw = _run_cmd("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits")
    if raw:
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        info["gpu_count"] = len(lines)
        if lines:
            parts = [p.strip() for p in lines[0].split(",")]
            if len(parts) >= 3:
                info["model"] = parts[0]
                try:
                    info["vram_mb"] = int(float(parts[1]))
                    info["vram_gb"] = round(info["vram_mb"] / 1024, 1)
                except ValueError:
                    pass
                info["driver_version"] = parts[2]

    # CUDA driver version from nvidia-smi header
    header = _run_cmd("nvidia-smi")
    if header:
        for line in header.split("\n"):
            if "CUDA Version" in line:
                try:
                    idx = line.index("CUDA Version:")
                    ver = line[idx + 14:].strip().split()[0]
                    info["cuda_driver_version"] = ver
                except (ValueError, IndexError):
                    pass
                break

    # Compute capability + SM count via CuPy or deviceQuery
    try:
        import cupy as cp
        dev = cp.cuda.Device(0)
        cc_major = dev.attributes["ComputeCapabilityMajor"]
        cc_minor = dev.attributes["ComputeCapabilityMinor"]
        info["compute_cap"] = f"{cc_major}.{cc_minor}"
        info["sm_count"] = dev.attributes.get("MultiProcessorCount", 0)
    except Exception:
        raw_cc = _run_cmd(
            "nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null"
        )
        if raw_cc and raw_cc[0].isdigit():
            info["compute_cap"] = raw_cc.split("\n")[0].strip()

    return info


def detect_cuda_toolkit():
    """Detect CUDA toolkit version (nvcc)."""
    # Check PATH first, then common locations
    for cmd in ["nvcc --version 2>/dev/null",
                "/usr/local/cuda/bin/nvcc --version 2>/dev/null"]:
        raw = _run_cmd(cmd)
        if raw:
            for line in raw.split("\n"):
                if "release" in line.lower():
                    try:
                        idx = line.lower().index("release")
                        ver = line[idx + 8:].strip().split(",")[0]
                        return ver
                    except (ValueError, IndexError):
                        pass
    return "Not installed"


def detect_cupy():
    """Check CuPy availability and version."""
    try:
        import cupy as cp
        version = cp.__version__
        try:
            from cupyx.scipy import sparse as cusp
            cusparse_ok = True
        except ImportError:
            cusparse_ok = False
        return {"available": True, "version": version, "cusparse": cusparse_ok}
    except ImportError:
        return {"available": False, "version": "N/A", "cusparse": False}
    except Exception as e:
        return {"available": False, "version": f"Error: {e}", "cusparse": False}


def detect_lgbm_cuda_sparse():
    """Check if LightGBM was built with CUDA sparse histogram support."""
    try:
        import lightgbm as lgb
        version = lgb.__version__

        try:
            X = sp.random(10, 5, density=0.3, format="csr", dtype=np.float32)
            y = np.random.randint(0, 2, 10)
            ds = lgb.Dataset(X, label=y, free_raw_data=False)
            params = {
                "device_type": "cuda_sparse",
                "num_iterations": 1,
                "verbose": -1,
                "num_leaves": 4,
            }
            lgb.train(params, ds, num_boost_round=1, verbose_eval=False)
            return {"available": True, "version": version, "detail": "cuda_sparse supported"}
        except lgb.basic.LightGBMError as e:
            err = str(e).lower()
            if "cuda" in err or "device" in err or "not compiled" in err:
                return {"available": False, "version": version, "detail": f"Not compiled: {e}"}
            return {"available": False, "version": version, "detail": str(e)}
        except Exception as e:
            return {"available": False, "version": version, "detail": str(e)}
    except ImportError:
        return {"available": False, "version": "Not installed", "detail": "LightGBM not found"}


def estimate_speedup(gpu_name, vram_gb):
    """Estimate GPU histogram speedup per TF using Amdahl's law.

    Model: histogram building is ~50% of LightGBM training time.
    GPU SpMV is memory-bandwidth bound, speedup = GPU_BW/CPU_BW * utilization.
    Overall speedup via Amdahl: 1 / ((1 - f) + f/speedup) where f=0.50.
    """
    # Match GPU to bandwidth
    gpu_bw = 500  # conservative default
    for key, bw in GPU_BANDWIDTH.items():
        if key.lower() in gpu_name.lower():
            gpu_bw = bw
            break

    results = []
    for profile in TF_PROFILES:
        csr_gb = profile['csr_gb']
        fits = vram_gb * 0.85 >= csr_gb

        # Utilization factor (smaller matrices have more launch overhead)
        rows = profile['rows']
        if rows < 1000:
            util = 0.45
        elif rows < 10000:
            util = 0.65
        elif rows < 50000:
            util = 0.75
        else:
            util = 0.80

        raw_speedup = (gpu_bw / CPU_BW_REF) * util
        hist_fraction = 0.50
        overall = 1.0 / ((1.0 - hist_fraction) + hist_fraction / raw_speedup)

        cpu_fold_min = profile['cpu_fold_min']
        gpu_fold_min = cpu_fold_min / overall

        results.append({
            'tf': profile['tf'],
            'fits_vram': fits,
            'csr_gb': csr_gb,
            'hist_speedup': round(raw_speedup, 1),
            'overall_speedup': round(overall, 1),
            'cpu_fold_min': cpu_fold_min,
            'gpu_fold_min': round(gpu_fold_min, 1),
        })
    return results


def run_benchmark():
    """Run cuSPARSE SpMV benchmark: 1000 x 100K sparse binary CSR, 50 iterations."""
    print("\n" + "=" * 70)
    print("BENCHMARK: cuSPARSE SpMV (1000 x 100K sparse binary CSR)")
    print("=" * 70)

    rows, cols = 1000, 100_000
    density = 0.01
    n_iters = 50

    print(f"  Generating {rows}x{cols} CSR (density={density}, nnz~{int(rows * cols * density):,})...")
    A_scipy = sp.random(rows, cols, density=density, format="csr", dtype=np.float32)
    A_scipy.data[:] = 1.0
    x_np = np.random.randn(cols).astype(np.float32)

    # SciPy baseline
    print(f"  SciPy SpMV x{n_iters}...")
    _ = A_scipy @ x_np  # warmup
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = A_scipy @ x_np
    scipy_time = time.perf_counter() - t0
    scipy_per_iter = scipy_time / n_iters
    print(f"    Total: {scipy_time:.4f}s  |  Per-iter: {scipy_per_iter * 1000:.3f}ms")

    # CuPy / cuSPARSE
    try:
        import cupy as cp
        from cupyx.scipy import sparse as cusp

        print(f"  CuPy cuSPARSE SpMV x{n_iters}...")
        A_gpu = cusp.csr_matrix(A_scipy)
        x_gpu = cp.asarray(x_np)

        _ = A_gpu @ x_gpu
        cp.cuda.Stream.null.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = A_gpu @ x_gpu
        cp.cuda.Stream.null.synchronize()
        cupy_time = time.perf_counter() - t0
        cupy_per_iter = cupy_time / n_iters

        nnz = A_scipy.nnz
        bytes_moved = nnz * 8 + (rows + 1) * 4 + cols * 4
        bw_gbps = (bytes_moved * n_iters) / cupy_time / 1e9

        speedup = scipy_time / cupy_time

        print(f"    Total: {cupy_time:.4f}s  |  Per-iter: {cupy_per_iter * 1000:.3f}ms")
        print(f"    Effective bandwidth: {bw_gbps:.1f} GB/s")
        print(f"    Speedup vs SciPy: {speedup:.1f}x")

        del A_gpu, x_gpu
        cp.get_default_memory_pool().free_all_blocks()

    except ImportError:
        print("    CuPy not available -- skipping GPU benchmark")
    except Exception as e:
        print(f"    GPU benchmark failed: {e}")


def print_report(gpu_info, toolkit_ver, cupy_info, lgbm_info, tf_filter=None):
    """Print the full capability report."""
    print("=" * 70)
    print("  Savage22 GPU Capability Report -- Sparse Histogram Fork")
    print("=" * 70)

    # Section 1: GPU Hardware
    print("\n1. GPU Hardware")
    print(f"   Model:              {gpu_info['model']}")
    print(f"   VRAM:               {gpu_info['vram_mb']} MB ({gpu_info['vram_gb']} GB)")
    print(f"   Compute Capability: {gpu_info['compute_cap']}")
    print(f"   SM Count:           {gpu_info['sm_count'] or 'N/A'}")
    print(f"   GPU Count:          {gpu_info['gpu_count']}")

    # Section 2: CUDA Stack
    print("\n2. CUDA Stack")
    print(f"   Driver Version:     {gpu_info['driver_version']}")
    print(f"   CUDA (driver):      {gpu_info['cuda_driver_version']}")
    print(f"   CUDA Toolkit:       {toolkit_ver}")

    # Section 3: CuPy
    print("\n3. CuPy")
    if cupy_info["available"]:
        print(f"   Version:            {cupy_info['version']}")
        print(f"   cuSPARSE:           {'Available' if cupy_info['cusparse'] else 'NOT available'}")
    else:
        print(f"   Status:             NOT available ({cupy_info['version']})")

    # Section 4: LightGBM CUDA Sparse
    print("\n4. LightGBM cuda_sparse Fork")
    if lgbm_info["available"]:
        print(f"   Version:            {lgbm_info['version']}")
        print(f"   cuda_sparse:        SUPPORTED")
    else:
        print(f"   Version:            {lgbm_info['version']}")
        print(f"   cuda_sparse:        NOT supported")
        print(f"   Detail:             {lgbm_info['detail']}")

    # Section 5: Per-TF VRAM Fit + Speedup Estimates
    vram_gb = gpu_info["vram_gb"]
    estimates = estimate_speedup(gpu_info["model"], vram_gb)

    print(f"\n5. Per-Timeframe Analysis (GPU VRAM = {vram_gb} GB)")
    print("   " + "-" * 78)
    print(f"   {'TF':<5} {'CSR Size':<11} {'Fits?':<7} {'Hist Spd':<10} {'Overall':<9} {'CPU/fold':<10} {'GPU/fold':<10}")
    print("   " + "-" * 78)

    gpu_tfs = []
    cpu_tfs = []

    for e in estimates:
        if tf_filter and e['tf'] != tf_filter:
            continue
        fits = e['fits_vram']
        fits_str = "YES" if fits else "NO"
        note = ""
        if not fits and vram_gb > 0:
            needed = int(np.ceil(e['csr_gb'] / 0.85))
            note = f" (need {needed}GB+)"

        if fits:
            gpu_tfs.append(e['tf'])
        else:
            cpu_tfs.append(e['tf'])

        print(f"   {e['tf']:<5} ~{e['csr_gb']:<9.1f}GB {fits_str:<7} "
              f"{e['hist_speedup']:<10.1f}x {e['overall_speedup']:<9.1f}x "
              f"~{e['cpu_fold_min']:<9}min ~{e['gpu_fold_min']:<8.1f}min{note}")

    print("   " + "-" * 78)

    # Detailed breakdown for --tf
    if tf_filter:
        for e in estimates:
            if e['tf'] != tf_filter:
                continue
            profile = next(p for p in TF_PROFILES if p['tf'] == tf_filter)
            print(f"\n   Detailed: {tf_filter}")
            print(f"     Matrix: {profile['rows']:,} rows x {profile['features']:,} features")
            print(f"     CSR size: ~{e['csr_gb']}GB")
            print(f"     VRAM fits: {'YES' if e['fits_vram'] else 'NO -- will use CPU fallback'}")
            print(f"     Histogram kernel speedup: {e['hist_speedup']}x (bandwidth-bound SpMV)")
            print(f"     Overall training speedup: {e['overall_speedup']}x (Amdahl: hist=50% of training)")
            print(f"     CPU per fold: ~{e['cpu_fold_min']} min")
            print(f"     GPU per fold: ~{e['gpu_fold_min']} min (estimated)")

    # Section 6: Multi-GPU
    if gpu_info['gpu_count'] > 1:
        print(f"\n6. Multi-GPU ({gpu_info['gpu_count']} GPUs)")
        print("   LightGBM uses 1 GPU for histogram building.")
        print("   Use extra GPUs for parallel Optuna workers:")
        for i in range(gpu_info['gpu_count']):
            print(f"     GPU {i}: CUDA_VISIBLE_DEVICES={i} lgbm-run python -u cloud_run_tf.py --tf <TF>")

    # Section 7: Recommendations
    print(f"\n{'7' if gpu_info['gpu_count'] > 1 else '6'}. Recommendations")

    if not lgbm_info["available"]:
        print("   [ACTION] Install GPU histogram fork:")
        print("            bash deploy_vastai.sh         (build from source)")
        print("            bash deploy_vastai_quick.sh   (pre-built wheel)")
    else:
        print("   [OK] LightGBM cuda_sparse fork installed")

    if toolkit_ver == "Not installed":
        print("   [ACTION] Install CUDA toolkit (needed for build from source)")
    else:
        print(f"   [OK] CUDA toolkit {toolkit_ver}")

    if not cupy_info["available"]:
        print("   [ACTION] Install CuPy: pip install cupy-cuda12x")
    else:
        print(f"   [OK] CuPy {cupy_info['version']}")

    if gpu_tfs:
        print(f"   [OK] GPU histograms for: {', '.join(gpu_tfs)}")
    if cpu_tfs:
        print(f"   [INFO] CPU fallback for: {', '.join(cpu_tfs)} (need more VRAM)")

    if vram_gb >= 80:
        print("\n   This GPU can handle ALL timeframes on-GPU.")
    elif vram_gb >= 24:
        print(f"\n   This GPU ({vram_gb}GB) handles 1w/1d/4h on-GPU.")
        print("   1h/15m require 80GB+ VRAM (A100/H100/H200).")
    elif vram_gb > 0:
        print(f"\n   This GPU ({vram_gb}GB) may only handle 1w/1d on-GPU.")
        print("   Consider cloud GPU (A100 80GB) for larger timeframes.")

    print()


def print_json(gpu_info, toolkit_ver, cupy_info, lgbm_info):
    """Print machine-readable JSON report."""
    estimates = estimate_speedup(gpu_info["model"], gpu_info["vram_gb"])
    report = {
        "gpu": gpu_info,
        "cuda_toolkit": toolkit_ver,
        "cupy": cupy_info,
        "lightgbm": lgbm_info,
        "speedup_estimates": {e['tf']: e for e in estimates},
    }
    print(json_mod.dumps(report, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(
        description="GPU capability reporter for Savage22 sparse histogram fork"
    )
    parser.add_argument(
        "--bench", action="store_true",
        help="Run cuSPARSE SpMV benchmark (1000x100K sparse binary CSR, 50 iterations)"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output machine-readable JSON"
    )
    parser.add_argument(
        "--tf", type=str, default=None, choices=["1w", "1d", "4h", "1h", "15m"],
        help="Show detailed estimate for one timeframe"
    )
    args = parser.parse_args()

    # Detect everything
    gpu_info = detect_gpu()
    toolkit_ver = detect_cuda_toolkit()
    cupy_info = detect_cupy()
    lgbm_info = detect_lgbm_cuda_sparse()

    if args.json:
        print_json(gpu_info, toolkit_ver, cupy_info, lgbm_info)
    else:
        print_report(gpu_info, toolkit_ver, cupy_info, lgbm_info, tf_filter=args.tf)

    # Optional benchmark
    if args.bench and not args.json:
        run_benchmark()

    # Exit code: 0 if GPU usable for at least 1w, 1 otherwise
    usable = gpu_info["vram_gb"] >= 2.0
    sys.exit(0 if usable else 1)


if __name__ == "__main__":
    main()
