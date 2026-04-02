#!/usr/bin/env python
"""
v2_cross_generator.py — V2 Everything × Everything Cross Feature Generator
============================================================================
Generates ALL V2 cross features using sparse batch operations.
MEMORY-OPTIMIZED: Streams results to sparse chunks instead of accumulating
dense arrays in a dict. Peak RAM stays under ~4GB regardless of feature count.

Cross types generated:
  dx_  = DOY window (±2 days) × context
  ax_  = astro × TA
  ax2_ = multi-astro × TA
  ta2_ = multi-TA × DOY + astro
  ex2_ = esoteric × TA
  sw_  = space weather × all
  hod_ = hour-of-day × all
  mx_  = macro × all
  vx_  = volatility regime × all
  asp_ = planetary aspects × all
  pn_  = price numerology × all
  3x regime-aware DOY crosses

Usage:
  python v2_cross_generator.py --tf 1d
  python v2_cross_generator.py --tf 1d --gpu 0
  python v2_cross_generator.py --tf 1h --save-sparse
"""

import os, sys, time, argparse, warnings, gc, glob, tempfile, queue, threading, shutil
import ctypes
from gpu_worker_estimator import preflight_check as _preflight_check
try:
    _libc = ctypes.cdll.LoadLibrary("libc.so.6")
    def _malloc_trim():
        _libc.malloc_trim(0)
except Exception:
    def _malloc_trim():
        pass
import numpy as np
import pandas as pd
from scipy import sparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from atomic_io import (
    cleanup_crossgen_scratch_dir,
    crossgen_scratch_dir,
    get_crossgen_namespace,
    get_crossgen_run_id,
    validate_sparse_names_contract,
)
# ── Bitpacked POPCNT co-occurrence pre-filter (Phase 1D) ──
_USE_BITPACK_COOCCURRENCE = os.environ.get('USE_BITPACK_COOCCURRENCE', '1') == '1'
try:
    from bitpack_utils import bitpack_cooccurrence_filter
    _HAS_BITPACK = True
except ImportError:
    _HAS_BITPACK = False
    if _USE_BITPACK_COOCCURRENCE:
        print("[v2_cross_generator] WARNING: bitpack_utils not found, falling back to sparse matmul for co-occurrence")
try:
    from sparse_dot_mkl import dot_product_mkl as _dot_product_mkl
    _USE_MKL = True
except ImportError:
    _USE_MKL = False
try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
    _HAS_THREADPOOLCTL = True
except ImportError:
    _HAS_THREADPOOLCTL = False
# Set Numba/OMP thread count BEFORE import — can't change after first use.
# Strategy: OMP_NUM_THREADS controls MKL baseline, but _mkl_dot() overrides
# via threadpoolctl for SpGEMM calls. Numba gets full core count for prange.
# MKL_DYNAMIC=FALSE prevents MKL from auto-shrinking its thread pool.
if 'NUMBA_NUM_THREADS' not in os.environ:
    try:
        _ncpu = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        _ncpu = os.cpu_count() or 1
    _numba_threads = max(4, _ncpu)  # Full cores for Numba prange kernels
    os.environ['NUMBA_NUM_THREADS'] = str(_numba_threads)
if 'OMP_NUM_THREADS' not in os.environ:
    # Matrix convention: cap OMP threads to avoid thread exhaustion.
    os.environ['OMP_NUM_THREADS'] = '4'
os.environ.setdefault('MKL_DYNAMIC', 'FALSE')

# Module-level CPU count for MKL threadpoolctl — resolves after NUMBA env, before Numba import
try:
    _cpu_count = len(os.sched_getaffinity(0))
except (AttributeError, OSError):
    _cpu_count = os.cpu_count() or 1

from numba import njit, prange

# ── Numba CSC cross gen (Optimization #1 + #6) ──
# Enable with USE_NUMBA_CROSS=1 for 3-8x speedup on cross gen
_USE_NUMBA_CROSS = os.environ.get('USE_NUMBA_CROSS', '0') == '1'
if _USE_NUMBA_CROSS:
    try:
        from numba_cross_kernels import numba_csc_cross, warmup_numba_kernels, sort_pairs_l2_friendly
        print("[v2_cross_generator] USE_NUMBA_CROSS=1 — Numba CSC intersection enabled (Opt #1+#6)")
        warmup_numba_kernels()
    except ImportError as _ncx_err:
        print(f"[v2_cross_generator] WARNING: USE_NUMBA_CROSS=1 but import failed: {_ncx_err}")
        _USE_NUMBA_CROSS = False

warnings.filterwarnings('ignore')

V2_DIR = os.path.dirname(os.path.abspath(__file__))

# ── CUDA version detection ──
# CuPy works on CUDA 13+ with CUPY_COMPILE_WITH_PTX=1 (Blackwell sm_120 compat)
os.environ.setdefault('CUPY_COMPILE_WITH_PTX', '1')
_CUDA_MAJOR = 12
try:
    import subprocess as _sp
    _nv = _sp.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                   capture_output=True, text=True, timeout=5)
    _drv = int(_nv.stdout.strip().split('.')[0])
    _CUDA_MAJOR = 13 if _drv >= 580 else 12
except Exception:
    pass

# ── GPU setup — CuPy works on CUDA 13+ with PTX JIT ──
# GPUs under RTX 4090 (< 24GB VRAM) are too small for cross gen — CPU with many cores is faster.
_MIN_VRAM_GB = 20  # RTX 4090 = 24GB, anything below = CPU only
try:
    import cupy as cp
    import cupyx.scipy.sparse as cusp
    cp.array([1.0]) + cp.array([2.0])  # verify GPU works
    _vram_gb = cp.cuda.runtime.memGetInfo()[1] / 1024**3
    _gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    if _vram_gb < _MIN_VRAM_GB:
        # Small GPU — CPU path is faster for cross gen
        cp = None
        cusp = None
        GPU = False
        print(f"[v2_cross_generator] {_gpu_name} ({_vram_gb:.0f}GB VRAM) too small for cross gen — using CPU (faster with many cores)")
    else:
        GPU = True
        if _CUDA_MAJOR >= 13:
            print(f"[v2_cross_generator] CuPy GPU verified: {_gpu_name} ({_vram_gb:.0f}GB) (CUDA {_CUDA_MAJOR}.x — PTX JIT)")
        else:
            print(f"[v2_cross_generator] CuPy GPU verified: {_gpu_name} ({_vram_gb:.0f}GB)")
except (ImportError, Exception) as _gpu_err:
    if os.environ.get('ALLOW_CPU', '0') != '1':
        raise RuntimeError(f"GPU REQUIRED: CuPy failed ({_gpu_err}). Set ALLOW_CPU=1 for CPU mode.")
    cp = None
    cusp = None
    GPU = False
    print(f"[v2_cross_generator] ALLOW_CPU=1 — CuPy unavailable ({_gpu_err}). Using CPU mode.")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        safe = line.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8', errors='replace')
        print(safe, flush=True)


def _mkl_dot(a, b):
    """MKL sparse SpGEMM with threadpoolctl override.
    Uses full CPU count for MKL — sparse matmul at 0.3% density is memory-bandwidth
    bound and benefits from all cores up to NUMA saturation. threadpoolctl scopes
    the boost to this call only, preventing thread exhaustion in other phases.
    """
    if _HAS_THREADPOOLCTL:
        with _threadpool_limits(limits=_cpu_count, user_api='openmp'):
            return _dot_product_mkl(a, b, cast=True)
    return _dot_product_mkl(a, b, cast=True)


def _get_available_ram_gb():
    """Get available RAM in GB (cgroup v1+v2 aware). Delegates to shared hardware_detect."""
    try:
        from hardware_detect import get_available_ram_gb
        return get_available_ram_gb()
    except ImportError:
        pass
    # Inline fallback if hardware_detect not available
    try:
        import psutil
        return psutil.virtual_memory().available / (1024**3)
    except (ImportError, Exception):
        pass
    return 64.0  # conservative fallback


# ── Adaptive batch sizing based on available RAM ──
def _get_right_chunk():
    """Fixed safe default to prevent OOM across TFs."""
    return 500


def _get_cross_batch_size(n_rows):
    """Auto-size batch for vectorized cross computation. Targets 25% of available RAM."""
    try:
        import psutil
        avail = psutil.virtual_memory().available
    except (ImportError, Exception):
        try:
            with open('/proc/meminfo') as f:
                for line in f:
                    if 'MemAvailable' in line:
                        avail = int(line.split()[1]) * 1024
                        break
                else:
                    avail = 16 * 1024**3
        except Exception:
            avail = 16 * 1024**3
    target = int(avail * 0.25)
    # Each pair: 2 column reads (N×4B) + 1 result (N×4B) = 12N bytes
    bytes_per_pair = n_rows * 12
    batch = max(500, min(50000, target // max(1, bytes_per_pair)))
    _env_batch = os.environ.get("V2_BATCH_MAX")
    if _env_batch:
        batch = min(batch, int(_env_batch))
    return batch


# Env var override for orchestrator OOM retry — now serves as MAX CAP for adaptive controller
_env_chunk = os.environ.get('V2_RIGHT_CHUNK')
RIGHT_CHUNK = int(_env_chunk) if _env_chunk else _get_right_chunk()
log(f"RIGHT_CHUNK = {RIGHT_CHUNK} ({'env override' if _env_chunk else 'fixed safe default'})")

# ── Parallel cross steps toggle ──
PARALLEL_CROSS_STEPS = os.environ.get('PARALLEL_CROSS_STEPS', '0') == '1'

# ── RAM ceiling for memory-aware scheduling ──
_RAM_CEILING_PCT = float(os.environ.get('V2_RAM_CEILING_PCT', '70'))


def _get_mem_percent():
    """Get current RAM usage as percentage (0-100)."""
    try:
        import psutil
        return psutil.virtual_memory().percent
    except (ImportError, Exception):
        return 50.0  # conservative fallback


def _get_rss_bytes():
    """Get current process RSS in bytes."""
    try:
        import psutil
        return psutil.Process().memory_info().rss
    except (ImportError, Exception):
        return 0


def _log_memory(label):
    """Log current memory usage with label."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        rss = psutil.Process().memory_info().rss
        log(f"  [MEM] {label}: RSS={rss/1e9:.2f}GB, System={vm.percent:.1f}% "
            f"(used={vm.used/1e9:.1f}GB / total={vm.total/1e9:.1f}GB, avail={vm.available/1e9:.1f}GB)")
    except (ImportError, Exception):
        log(f"  [MEM] {label}: (psutil unavailable)")


class AdaptiveChunkController:
    """Rolling RSS-based RIGHT_CHUNK sizing.

    - Pilots with RC=16 for first 2 chunks to measure actual bytes/col
    - Sizes subsequent chunks from worst-case of last 3 measurements
    - On MemoryError: halves chunk size and retries
    - V2_RIGHT_CHUNK env var is max cap (not fixed value)
    - Target: stay below 70% of available RAM
    """
    PILOT_SIZE = 16
    PILOT_CHUNKS = 2

    def __init__(self, max_cap=None, target_pct=None):
        self.max_cap = max_cap or RIGHT_CHUNK
        self.target_pct = target_pct or _RAM_CEILING_PCT
        self._measurements = []  # list of bytes_per_col from recent chunks
        self._chunk_count = 0
        self._current_size = self.PILOT_SIZE
        self._halved = False

    def get_chunk_size(self):
        """Return the current recommended chunk size."""
        return self._current_size

    def record_chunk(self, n_cols, rss_before, rss_after):
        """Record RSS measurement after processing a chunk.
        Updates internal state for next chunk sizing."""
        self._chunk_count += 1
        delta_bytes = max(0, rss_after - rss_before)
        if n_cols > 0:
            bytes_per_col = delta_bytes / n_cols
            self._measurements.append(bytes_per_col)
            # Keep only last 3 measurements
            if len(self._measurements) > 3:
                self._measurements = self._measurements[-3:]

        # After pilot phase, compute optimal chunk size
        if self._chunk_count >= self.PILOT_CHUNKS and self._measurements:
            self._resize()

    def halve(self):
        """Called on MemoryError — halve current chunk size."""
        self._current_size = max(4, self._current_size // 2)
        self._halved = True
        log(f"    [AdaptiveChunk] MemoryError → halved to {self._current_size}")

    def _resize(self):
        """Compute optimal chunk size from worst-case of recent measurements."""
        if not self._measurements:
            return
        worst_bpc = max(self._measurements)
        if worst_bpc <= 0:
            self._current_size = self.max_cap
            return

        try:
            import psutil
            avail = psutil.virtual_memory().available
        except (ImportError, Exception):
            avail = 16 * 1024**3

        # Target: use up to (target_pct)% of available RAM for this chunk
        target_bytes = avail * (self.target_pct / 100.0) * 0.5  # 50% of ceiling for safety
        optimal = int(target_bytes / worst_bpc)
        self._current_size = max(4, min(optimal, self.max_cap))

    def __repr__(self):
        return (f"AdaptiveChunkController(size={self._current_size}, cap={self.max_cap}, "
                f"chunks={self._chunk_count}, measurements={len(self._measurements)})")

# Minimum co-occurrence threshold for cross features.
# Crosses firing fewer than this many times cannot reliably appear in both
# CPCV train AND validation splits (98.3% coverage at n=8 with 5-fold CPCV).
# This is a math constraint on the validation pipeline, not a signal filter.
_env_co_occur = os.environ.get('V2_MIN_CO_OCCURRENCE')
MIN_CO_OCCURRENCE = int(_env_co_occur) if _env_co_occur else 3  # Lowered from 8: matches min_data_in_leaf=3, preserves rare esoteric crosses


def _apply_correlation_clustering(sparse_mat, feature_names, tf, threshold=0.95):
    """
    Cluster highly correlated cross features using MinHash LSH + sparse Pearson.

    PROTECTED FEATURES: Never clusters esoteric signals (PROTECTED_FEATURE_PREFIXES).
    Only clusters redundant TA × TA crosses (r > threshold).

    Algorithm (from Perplexity research 2026-04-01):
      1. Pre-filter: separate rare (< 1% non-zero) and esoteric features
      2. MinHash LSH: candidate pair generation (Jaccard ≈ Pearson for binary)
      3. Exact sparse Pearson: verify candidates above threshold
      4. Union-Find: cluster correlated features
      5. Keep one representative per cluster (highest sparsity)

    Args:
        sparse_mat: scipy CSC sparse matrix (n_samples × n_features)
        feature_names: list of feature names (length = n_features)
        tf: timeframe (for logging)
        threshold: correlation threshold for clustering (default 0.95)

    Returns:
        (clustered_sparse_mat, clustered_feature_names)
    """
    from scipy.sparse import csc_matrix
    import json
    from config import PROTECTED_FEATURE_PREFIXES

    n_samples, n_features = sparse_mat.shape
    log(f"  [CLUSTER] Input: {n_features:,} features, {n_samples:,} samples")

    # Step 1: Identify protected features (esoteric + rare)
    sparse_mat_csc = sparse_mat.tocsc() if not isinstance(sparse_mat, csc_matrix) else sparse_mat
    nnz_per_feature = np.diff(sparse_mat_csc.indptr)
    sparsity = nnz_per_feature / n_samples

    # Rare features (< 1% non-zero) are protected
    rare_mask = sparsity < 0.01

    # Esoteric features (protected prefixes) are protected
    esoteric_mask = np.array([
        any(name.startswith(p) for p in PROTECTED_FEATURE_PREFIXES)
        for name in feature_names
    ])

    # Protected = rare OR esoteric
    protected_mask = rare_mask | esoteric_mask
    clusterable_mask = ~protected_mask

    n_protected = protected_mask.sum()
    n_clusterable = clusterable_mask.sum()

    log(f"  [CLUSTER] Protected: {n_protected:,} ({rare_mask.sum():,} rare + "
        f"{(esoteric_mask & ~rare_mask).sum():,} esoteric)")
    log(f"  [CLUSTER] Clusterable: {n_clusterable:,} TA crosses")

    if n_clusterable < 100:
        log(f"  [CLUSTER] Too few clusterable features, skipping")
        return sparse_mat, feature_names

    # Step 2: MinHash LSH candidate pair generation
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        log(f"  [CLUSTER] datasketch not installed, skipping clustering")
        return sparse_mat, feature_names

    log(f"  [CLUSTER] Building MinHash LSH index (threshold={threshold})...")
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    minhashes = {}
    clusterable_indices = np.where(clusterable_mask)[0]

    for i in clusterable_indices:
        col = sparse_mat_csc.getcol(i)
        nonzero_rows = col.nonzero()[0]
        if len(nonzero_rows) == 0:
            continue  # Skip empty columns

        m = MinHash(num_perm=128)
        for r in nonzero_rows:
            m.update(r.to_bytes(8, 'little'))

        key = f"f{i}"
        lsh.insert(key, m)
        minhashes[key] = m

    # Query to get candidate pairs
    log(f"  [CLUSTER] Querying LSH for candidate pairs...")
    candidate_pairs = set()
    for key, m in minhashes.items():
        neighbors = lsh.query(m)
        for n in neighbors:
            if n != key:
                i_key = int(key[1:])
                j_key = int(n[1:])
                if i_key < j_key:  # Avoid duplicates
                    candidate_pairs.add((i_key, j_key))

    log(f"  [CLUSTER] LSH found {len(candidate_pairs):,} candidate pairs")

    if len(candidate_pairs) == 0:
        log(f"  [CLUSTER] No candidates found, returning original matrix")
        return sparse_mat, feature_names

    # Step 3: Exact sparse Pearson verification
    log(f"  [CLUSTER] Computing exact Pearson correlations...")
    confirmed_pairs = []

    for i, j in candidate_pairs:
        ci = np.asarray(sparse_mat_csc.getcol(i).todense()).flatten()
        cj = np.asarray(sparse_mat_csc.getcol(j).todense()).flatten()

        # Compute Pearson correlation
        if np.std(ci) == 0 or np.std(cj) == 0:
            continue  # Skip constant columns

        r = np.corrcoef(ci, cj)[0, 1]
        if abs(r) > threshold:
            confirmed_pairs.append((i, j, r))

    log(f"  [CLUSTER] Confirmed: {len(confirmed_pairs):,} pairs with |r| > {threshold}")

    if len(confirmed_pairs) == 0:
        log(f"  [CLUSTER] No confirmed pairs, returning original matrix")
        return sparse_mat, feature_names

    # Step 4: Union-Find clustering
    log(f"  [CLUSTER] Building clusters (Union-Find)...")
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        parent[find(x)] = find(y)

    for i, j, r in confirmed_pairs:
        union(i, j)

    # Group into clusters
    from collections import defaultdict
    clusters = defaultdict(list)
    for idx in clusterable_indices:
        if idx in parent or any(idx == p[0] or idx == p[1] for p in confirmed_pairs):
            clusters[find(idx)].append(idx)

    # Step 5: Select representatives (keep sparsest = most specific signal)
    representatives = []
    clustered_count = 0

    for root, members in clusters.items():
        if len(members) == 1:
            representatives.append(members[0])
        else:
            # Keep the sparsest (most specific) feature in each cluster
            sparsities = [sparsity[m] for m in members]
            best_idx = members[np.argmin(sparsities)]
            representatives.append(best_idx)
            clustered_count += len(members) - 1

    # Add all protected features
    protected_indices = np.where(protected_mask)[0].tolist()
    all_keep_indices = sorted(protected_indices + representatives)

    log(f"  [CLUSTER] Clusters: {len(clusters):,} groups")
    log(f"  [CLUSTER] Removed: {clustered_count:,} redundant features")
    log(f"  [CLUSTER] Kept: {len(all_keep_indices):,} features "
        f"({len(protected_indices):,} protected + {len(representatives):,} representatives)")

    # Build reduced matrix
    reduced_mat = sparse_mat_csc[:, all_keep_indices].tocsc()
    reduced_names = [feature_names[i] for i in all_keep_indices]

    return reduced_mat, reduced_names


def _compute_cooccurrence_pairs(left_mat, right_mat, min_nonzero):
    """Compute co-occurrence counts and return valid pairs meeting threshold.

    Uses bitpacked AND + hardware POPCNT when available (8-21ms for 7M+ pairs),
    falls back to sparse matmul (cuSPARSE GPU → MKL → scipy).

    Parameters
    ----------
    left_mat : ndarray (n_rows, n_left) — binary 0/1 float32
    right_mat : ndarray (n_rows, n_right) — binary 0/1 float32
    min_nonzero : int — minimum co-occurrence threshold

    Returns
    -------
    valid_pairs : ndarray (n_valid, 2) — [left_idx, right_idx] pairs
    method : str — which method was used ('bitpack', 'cuSPARSE', 'MKL', 'scipy')
    """
    n_left = left_mat.shape[1]
    n_right = right_mat.shape[1]
    n_total_pairs = n_left * n_right

    # ── Bitpacked POPCNT path (fastest: 8-21ms for 7M+ pairs) ──
    if _USE_BITPACK_COOCCURRENCE and _HAS_BITPACK:
        t_bp = time.time()
        valid_pairs, _counts = bitpack_cooccurrence_filter(left_mat, right_mat, min_co=min_nonzero)
        dt_bp = time.time() - t_bp
        log(f"    Bitpack POPCNT: {n_total_pairs:,} pairs → {len(valid_pairs):,} valid "
            f"(≥{min_nonzero} co-occur) in {dt_bp*1000:.1f}ms")
        # Build sparse co_occur matrix from bitpack counts for caller pair-sorting
        co_occur = np.zeros((n_left, n_right), dtype=np.int32)
        if len(valid_pairs) > 0:
            co_occur[valid_pairs[:, 0], valid_pairs[:, 1]] = _counts
        return valid_pairs, 'bitpack', co_occur

    # ── Sparse matmul fallback ──
    t_sp = time.time()
    if cp is not None and cusp is not None:
        try:
            left_gpu_sp = cusp.csc_matrix(cp.asarray(left_mat.astype(np.float32)))
            right_gpu_sp = cusp.csc_matrix(cp.asarray(right_mat.astype(np.float32)))
            co_occur = cp.asnumpy((left_gpu_sp.T @ right_gpu_sp).toarray())
            del left_gpu_sp, right_gpu_sp
            cp.get_default_memory_pool().free_all_blocks()
            method = 'cuSPARSE'
        except Exception as e:
            log(f"  cuSPARSE SpGEMM failed ({e}), falling back to MKL/scipy")
            left_sp = sparse.csr_matrix(left_mat.astype(np.float32))
            right_sp = sparse.csr_matrix(right_mat.astype(np.float32))
            if _USE_MKL:
                co_occur = _mkl_dot(left_sp.T, right_sp).toarray()
                method = 'MKL'
            else:
                co_occur = (left_sp.T @ right_sp).toarray()
                method = 'scipy'
    else:
        left_sp = sparse.csr_matrix(left_mat.astype(np.float32))
        right_sp = sparse.csr_matrix(right_mat.astype(np.float32))
        if _USE_MKL:
            co_occur = _mkl_dot(left_sp.T, right_sp).toarray()
            method = 'MKL'
        else:
            co_occur = (left_sp.T @ right_sp).toarray()
            method = 'scipy'

    valid_pairs = np.argwhere(co_occur >= min_nonzero)
    dt_sp = time.time() - t_sp
    log(f"    Sparse matmul ({method}): {n_total_pairs:,} pairs → {len(valid_pairs):,} valid "
        f"(≥{min_nonzero} co-occur) in {dt_sp*1000:.1f}ms")
    return valid_pairs, method, co_occur


# ── GPU VRAM-adaptive batch sizing ──
def _get_gpu_vram_gb(gpu_id=0):
    """Detect GPU VRAM in GB. Returns 0 if CuPy unavailable."""
    if cp is None:
        return 0.0
    try:
        mem = cp.cuda.Device(gpu_id).mem_info  # returns (free, total)
        return mem[1] / (1024**3)  # total VRAM in GB
    except Exception:
        return 12.0  # default: assume 3090


def _get_optimal_batch(n_bars, n_right, gpu_vram_gb=None, gpu_id=0):
    """Auto-size GPU batch based on available VRAM. Fully dynamic — no hard cap.
    Scales from BATCH=5 on 12GB 3090 to BATCH=200+ on 96GB H100."""
    if gpu_vram_gb is None:
        gpu_vram_gb = _get_gpu_vram_gb(gpu_id=gpu_id)
    # Leave 30% headroom for kernel overhead + other GPU operations
    available_bytes = gpu_vram_gb * 0.7 * (1024**3)
    # Shape is (n_bars, BATCH, n_right) in float32 = 4 bytes
    bytes_per_batch_elem = n_bars * n_right * 4
    if bytes_per_batch_elem <= 0:
        return 10  # safe minimum
    max_batch = int(available_bytes / bytes_per_batch_elem)
    return max(1, max_batch)


# ============================================================
# BINARIZATION
# ============================================================

@njit(parallel=True, cache=True)
def _binarize_batch_4tier(values, n_cols):
    """Compute 4-tier binarization (XH/H/L/XL) for all columns in parallel.

    Parameters
    ----------
    values : float32 array (n_rows, n_cols) — contiguous, NaN-preserving
    n_cols : int

    Returns
    -------
    masks : float32 array (n_cols * 4, n_rows) — row-major per mask
        Layout: [col0_XH, col0_H, col0_L, col0_XL, col1_XH, ...]
    thresholds : float32 array (n_cols, 4) — [q95, q75, q25, q5]
    mask_sums : float32 array (n_cols * 4) — nansum of each mask (for filtering)
    """
    n_rows = values.shape[0]
    masks = np.zeros((n_cols * 4, n_rows), dtype=np.float32)
    thresholds = np.zeros((n_cols, 4), dtype=np.float32)
    mask_sums = np.zeros(n_cols * 4, dtype=np.float32)

    for col_idx in prange(n_cols):
        col = values[:, col_idx]

        # Gather non-NaN values
        valid_count = 0
        for i in range(n_rows):
            if not np.isnan(col[i]):
                valid_count += 1
        if valid_count < 10:
            # Near-empty — masks stay zero, thresholds stay zero
            continue

        valid = np.empty(valid_count, dtype=np.float32)
        vi = 0
        for i in range(n_rows):
            if not np.isnan(col[i]):
                valid[vi] = col[i]
                vi += 1

        # Count non-zero valid entries
        nz_count = 0
        for i in range(valid_count):
            if valid[i] != 0.0:
                nz_count += 1

        # Use non-zero subset if >100 non-zero values, else use all valid
        if nz_count > 100:
            nz = np.empty(nz_count, dtype=np.float32)
            ni = 0
            for i in range(valid_count):
                if valid[i] != 0.0:
                    nz[ni] = valid[i]
                    ni += 1
        else:
            nz = valid

        # Sort for percentile computation
        sorted_vals = np.sort(nz)
        n = len(sorted_vals)
        # Compute percentiles via index (matches numpy nearest-rank behavior)
        q95 = sorted_vals[min(int(n * 0.95), n - 1)]
        q75 = sorted_vals[min(int(n * 0.75), n - 1)]
        q25 = sorted_vals[min(int(n * 0.25), n - 1)]
        q5 = sorted_vals[min(int(n * 0.05), n - 1)]
        thresholds[col_idx, 0] = q95
        thresholds[col_idx, 1] = q75
        thresholds[col_idx, 2] = q25
        thresholds[col_idx, 3] = q5

        base = col_idx * 4
        s_xh = np.float32(0.0)
        s_h = np.float32(0.0)
        s_l = np.float32(0.0)
        s_xl = np.float32(0.0)
        for i in range(n_rows):
            v = col[i]
            # NaN comparisons yield False — masks stay 0.0 for NaN rows
            if v > q95:
                masks[base, i] = 1.0
                s_xh += 1.0
            if v > q75:
                masks[base + 1, i] = 1.0
                s_h += 1.0
            if v < q25:
                masks[base + 2, i] = 1.0
                s_l += 1.0
            if v < q5:
                masks[base + 3, i] = 1.0
                s_xl += 1.0
        mask_sums[base] = s_xh
        mask_sums[base + 1] = s_h
        mask_sums[base + 2] = s_l
        mask_sums[base + 3] = s_xl

    return masks, thresholds, mask_sums


@njit(parallel=True, cache=True)
def _binarize_batch_2tier(values, n_cols):
    """Compute 2-tier binarization (H/L) for all columns in parallel.

    Same structure as 4-tier but with q80/q20 thresholds.
    """
    n_rows = values.shape[0]
    masks = np.zeros((n_cols * 2, n_rows), dtype=np.float32)
    thresholds = np.zeros((n_cols, 2), dtype=np.float32)
    mask_sums = np.zeros(n_cols * 2, dtype=np.float32)

    for col_idx in prange(n_cols):
        col = values[:, col_idx]

        valid_count = 0
        for i in range(n_rows):
            if not np.isnan(col[i]):
                valid_count += 1
        if valid_count < 10:
            continue

        valid = np.empty(valid_count, dtype=np.float32)
        vi = 0
        for i in range(n_rows):
            if not np.isnan(col[i]):
                valid[vi] = col[i]
                vi += 1

        nz_count = 0
        for i in range(valid_count):
            if valid[i] != 0.0:
                nz_count += 1

        if nz_count > 100:
            nz = np.empty(nz_count, dtype=np.float32)
            ni = 0
            for i in range(valid_count):
                if valid[i] != 0.0:
                    nz[ni] = valid[i]
                    ni += 1
        else:
            nz = valid

        sorted_vals = np.sort(nz)
        n = len(sorted_vals)
        q80 = sorted_vals[min(int(n * 0.80), n - 1)]
        q20 = sorted_vals[min(int(n * 0.20), n - 1)]
        thresholds[col_idx, 0] = q80
        thresholds[col_idx, 1] = q20

        base = col_idx * 2
        s_h = np.float32(0.0)
        s_l = np.float32(0.0)
        for i in range(n_rows):
            v = col[i]
            if v > q80:
                masks[base, i] = 1.0
                s_h += 1.0
            if v < q20:
                masks[base + 1, i] = 1.0
                s_l += 1.0
        mask_sums[base] = s_h
        mask_sums[base + 1] = s_l

    return masks, thresholds, mask_sums


def binarize_contexts(df, four_tier=True):
    """
    Binarize all suitable columns into binary context arrays.
    Returns: (ctx_names, ctx_arrays) where ctx_arrays is list of float32 arrays.

    With four_tier=True:
      EXTREME_HIGH (>95th), HIGH (>75th), LOW (<25th), EXTREME_LOW (<5th)

    Uses Numba @njit(parallel=True) to compute all percentiles + masks in one
    parallel kernel instead of 3000+ sequential np.percentile() calls.
    """
    skip_pre = ('tx_', 'px_', 'ex_', 'dx_', 'cross_', 'next_', 'target_',
                'doy_', 'ax_', 'ax2_', 'ta2_', 'ex2_', 'sw_', 'hod_',
                'mx_', 'vx_', 'pn_', 'seq_', 'roc_', 'harm_')
    skip_ex = {'timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume',
               'trades', 'taker_buy_volume', 'taker_buy_quote', 'triple_barrier_label',
               'open_time', 'open_time_ms'}

    N = len(df)
    ctx_names = []
    ctx_arrays = []

    # ── Pass 1: classify columns (Python — handles strings, dedup, filtering) ──
    binary_cols = []     # (col_name, vals_array) for <=3 unique values
    multi_cols = []      # (col_name, col_index_in_batch) for >3 unique values
    multi_vals_list = [] # corresponding float32 arrays to stack

    seen_cols = set()
    for col in df.columns:
        if col in seen_cols:
            continue
        seen_cols.add(col)
        if col.startswith(skip_pre) or col in skip_ex:
            continue

        raw = df[col]
        if isinstance(raw, pd.DataFrame):
            raw = raw.iloc[:, 0]
        vals = pd.to_numeric(raw, errors='coerce').values.astype(np.float32)
        uniq = np.unique(vals[~np.isnan(vals)])
        if len(uniq) <= 1:
            continue

        if len(uniq) <= 3:
            # Binary/ternary — handle directly (no percentile needed)
            b = (vals > 0).astype(np.float32)
            if 5 < np.nansum(b) < N * 0.98:
                binary_cols.append((col, b))
        else:
            multi_cols.append((col, len(multi_vals_list)))
            multi_vals_list.append(vals)

    # ── Add binary columns directly ──
    for col_name, b in binary_cols:
        ctx_names.append(col_name)
        ctx_arrays.append(b)

    # ── Pass 2: Numba parallel kernel for all multi-valued columns ──
    if multi_vals_list:
        # Stack into contiguous (n_rows, n_multi_cols) array
        batch = np.column_stack(multi_vals_list)  # (N, n_multi_cols) float32
        n_multi = len(multi_vals_list)

        if four_tier:
            masks, _thresholds, mask_sums = _binarize_batch_4tier(batch, n_multi)
            tags = ('XH', 'H', 'L', 'XL')
            n_tiers = 4
        else:
            masks, _thresholds, mask_sums = _binarize_batch_2tier(batch, n_multi)
            tags = ('H', 'L')
            n_tiers = 2

        # ── Pass 3: construct output from kernel results (Python — string ops) ──
        for col_name, batch_idx in multi_cols:
            base = batch_idx * n_tiers
            for t, tag in enumerate(tags):
                if mask_sums[base + t] > 5:
                    ctx_names.append(f'{col_name}_{tag}')
                    ctx_arrays.append(masks[base + t].copy())

        del batch, masks, mask_sums
        gc.collect()

    return ctx_names, ctx_arrays


# ============================================================
# SIGNAL GROUP EXTRACTION
# ============================================================

def extract_signal_groups(df, ctx_names, ctx_arrays):
    """
    Extract named groups of binary signals for targeted crossing.
    Returns dict of {group_name: [(name, array), ...]}.
    """
    groups = {
        'astro': [], 'ta': [], 'esoteric': [], 'space_weather': [],
        'macro': [], 'regime': [], 'session': [], 'aspect': [],
        'price_num': [], 'moon': [], 'volatility': [],
    }

    astro_keys = ('moon_phase', 'moon_illumination', 'moon_distance',
                  'new_moon', 'full_moon',
                  'retro', 'eclipse', 'nakshatra', 'void_of_course',
                  'planetary_hour', 'planetary_day', 'bazi', 'tzolkin',
                  'equinox', 'solstice', 'tithi', 'karana')
    ta_keys = ('rsi', 'macd', 'stoch', 'volume_', 'ema', 'sma',
               'obv', 'ichimoku', 'supertrend',
               'vwap', 'adx', 'cci', 'mfi', 'entropy', 'hurst')
    esoteric_keys = (
        # gematria / digital root
        'gem_', 'dr_', 'gematria', 'digital_root',
        # tweet / sentiment / caution
        'tweet', 'caution', 'pump', 'misdirection', 'sentiment',
        # on-chain
        'onchain', 'oi_', 'funding', 'liq_', 'whale', 'mempool',
        'hash_rate', 'block_height', 'coinbase_premium', 'cvd',
        # hebrew calendar / shmita
        'hebrew', 'shmita',
        # esoteric time cycles
        'fibonacci_time', 'gann_time', 'tesla',
        # numerology
        'master_', 'angel', 'contains_', 'palindrome', 'friday_13',
        # text features
        'caps_', 'excl_', 'word_count',
        # fear & greed
        'fear_greed', 'f_g_',
    )
    volatility_keys = ('atr', 'bb_width', 'realized_vol', 'vol_', 'keltner',
                        'donchian', 'true_range', 'natr', 'bb_')
    space_keys = ('kp_', 'solar', 'sunspot')
    macro_keys = ('vix', 'dxy', 'yield', 'spx', 'dominance', 'tvl', 'cot_')
    regime_keys = ('regime', 'bull', 'bear', 'sideways')
    session_keys = ('session_', 'kill_zone', 'hod_')
    aspect_keys = ('asp_',)
    price_num_keys = ('price_dr', 'price_angel', 'price_near', 'price_master')
    moon_keys = ('moon_in_', 'moon_fire', 'moon_earth', 'moon_air', 'moon_water',
                 'moon_sign', 'moon_cardinal', 'moon_fixed', 'moon_mutable')

    for name, arr in zip(ctx_names, ctx_arrays):
        nl = name.lower()
        if any(k in nl for k in astro_keys):
            groups['astro'].append((name, arr))
        elif any(k in nl for k in esoteric_keys):
            groups['esoteric'].append((name, arr))
        elif any(k in nl for k in volatility_keys):
            groups['volatility'].append((name, arr))
        elif any(k in nl for k in ta_keys):
            groups['ta'].append((name, arr))
        elif any(k in nl for k in space_keys):
            groups['space_weather'].append((name, arr))
        elif any(k in nl for k in macro_keys):
            groups['macro'].append((name, arr))
        elif any(k in nl for k in regime_keys):
            groups['regime'].append((name, arr))
        elif any(k in nl for k in session_keys):
            groups['session'].append((name, arr))
        elif any(k in nl for k in aspect_keys):
            groups['aspect'].append((name, arr))
        elif any(k in nl for k in price_num_keys):
            groups['price_num'].append((name, arr))
        elif any(k in nl for k in moon_keys):
            groups['moon'].append((name, arr))
        else:
            groups['ta'].append((name, arr))  # default to TA

    return groups


# ============================================================
# HELPERS: dict -> COO triplet conversion
# ============================================================

    # _dict_to_coo_triplets removed — GPU/CPU chunks now return COO directly


# ============================================================
# MULTI-GPU DETECTION
# ============================================================

def _detect_available_gpus():
    """Detect all available GPUs. Returns list of GPU IDs."""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--list-gpus'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            gpu_count = result.stdout.strip().count('\n') + 1 if result.stdout.strip() else 0
            return list(range(gpu_count))
    except Exception:
        pass
    # Fallback: try CuPy
    try:
        import cupy as cp
        return list(range(cp.cuda.runtime.getDeviceCount()))
    except Exception:
        return [0]

_MULTI_GPU_CROSS_GEN = os.environ.get('MULTI_GPU_CROSS_GEN', '1') == '1'
_GPU_SERVER_BATCH = int(os.environ.get('GPU_SERVER_BATCH', '5000'))


# ============================================================
# PERSISTENT GPU SERVER (long-lived, queue-based, ZERO scipy in hot loop)
# ============================================================

def _gpu_server(gpu_id, work_queue, result_queue, left_npy_path, right_npy_path,
                n_left_cols, N, out_dir):
    """
    Persistent GPU server process. Initializes CUDA + uploads CSC matrix ONCE,
    then processes batches from work_queue until poison pill (None).

    ZERO scipy in hot loop — only numpy + cupy + raw binary file I/O.
    Memory is CONSTANT: CSC indices on GPU + per-batch kernel buffers (freed each batch).
    Per-server host RAM: ~5-10GB (CSC indptr/indices + col_nnz array).

    .idx binary format (per pair with nnz > 0):
        pair_local_idx (int32) | nnz (int32) | row_indices (int32[nnz])

    Work queue message: (batch_pairs_remapped: ndarray[int32, (n,2)], batch_idx: int)
    Result queue message: (gpu_id: int, batch_idx: int, total_nnz: int, idx_path: str|None)
    Error signal: (gpu_id, -1, -1, error_message_str)
    Poison pill: None (triggers clean exit)
    """
    import os, time, gc, traceback
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ.setdefault('CUPY_COMPILE_WITH_PTX', '1')

    import numpy as np

    # Import CuPy fresh — CUDA_VISIBLE_DEVICES already set above
    import cupy as _cp
    _cp.cuda.Device(0).use()
    _cp.cuda.set_pinned_memory_allocator(None)
    mem_free, mem_total = _cp.cuda.Device(0).mem_info
    vram_gb = mem_total / (1024**3)
    _cp.get_default_memory_pool().set_limit(size=int(vram_gb * 0.85 * 1024**3))

    def _log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] [GPU-SRV-{gpu_id}] {msg}", flush=True)

    batches_done = 0
    try:
        _log(f"Initializing — VRAM: {vram_gb:.1f}GB")

        # ── ONE-TIME INIT: Load matrices, build combined CSC, upload to GPU ──
        left_mat = np.load(left_npy_path, mmap_mode='r')
        right_mat = np.load(right_npy_path, mmap_mode='r')

        # Build combined CSC [left | right] — scipy used ONLY here at init
        from scipy import sparse as _sp_init
        combined = np.hstack([np.asarray(left_mat), np.asarray(right_mat)])
        combined_csc = _sp_init.csc_matrix(combined.astype(np.float32))
        del combined, left_mat, right_mat, _sp_init
        gc.collect()

        # Upload CSC structure to GPU (just indptr + indices — no data, binary AND)
        indptr_gpu = _cp.asarray(combined_csc.indptr.astype(np.int32))
        indices_gpu = _cp.asarray(combined_csc.indices.astype(np.int32))

        # Per-column nnz for VRAM-adaptive sub-batching
        col_nnz = np.diff(combined_csc.indptr).astype(np.int64)
        del combined_csc
        gc.collect()

        _log(f"CSC uploaded — {len(col_nnz)} cols, max_nnz={int(col_nnz.max())}, "
             f"GPU mem: {_cp.cuda.Device(0).mem_info[0]/1e9:.1f}GB free")

        # ── Compile CUDA kernel (same logic as _get_sparse_and_kernel, local to this process) ──
        kernel = _cp.RawKernel(r"""
        extern "C" __global__
        void sparse_and_batch(
            const int* __restrict__ indptr,
            const int* __restrict__ indices,
            const int* __restrict__ col_pairs,
            int*       result_indices,
            int*       result_counts,
            const int  max_out_per_pair
        ) {
            int pair = blockIdx.x;
            int col_a = col_pairs[pair * 2];
            int col_b = col_pairs[pair * 2 + 1];
            int a = indptr[col_a], a_end = indptr[col_a + 1];
            int b = indptr[col_b], b_end = indptr[col_b + 1];
            int* out = result_indices + pair * max_out_per_pair;
            int cnt = 0;
            while (a < a_end && b < b_end && cnt < max_out_per_pair) {
                int ra = indices[a], rb = indices[b];
                if (ra == rb) { out[cnt++] = ra; a++; b++; }
                else if (ra < rb) { a++; }
                else { b++; }
            }
            result_counts[pair] = cnt;
        }
        """, "sparse_and_batch")

        _log("Kernel compiled, entering work loop")

        # ── WORK LOOP: Process batches until poison pill ──
        while True:
            msg = work_queue.get()
            if msg is None:  # poison pill → clean exit
                break

            batch_pairs, batch_idx = msg
            # batch_pairs: ndarray (n_pairs, 2) int32 — already remapped (right += n_left_cols)
            n_pairs = len(batch_pairs)
            t0 = time.time()

            # Per-pair max_nnz for tight buffer sizing (avoids over-allocation)
            left_nnz = col_nnz[batch_pairs[:, 0]]
            right_nnz = col_nnz[batch_pairs[:, 1]]
            pair_max_nnz = np.minimum(left_nnz, right_nnz)
            max_out = int(pair_max_nnz.max()) if n_pairs > 0 else 0

            if max_out == 0:
                result_queue.put((gpu_id, batch_idx, 0, None))
                continue
            max_out = min(max_out, N)

            # VRAM-adaptive sub-batching — never exceed 60% of free VRAM
            result_bytes = n_pairs * max_out * 4
            mem_free_now = _cp.cuda.Device(0).mem_info[0]
            if result_bytes > mem_free_now * 0.6:
                sub_batch_size = max(100, int(n_pairs * (mem_free_now * 0.5) / result_bytes))
            else:
                sub_batch_size = n_pairs

            # Process sub-batches, stream results directly to .idx file — ZERO accumulation
            idx_path = os.path.join(out_dir, f'gpu{gpu_id}_batch{batch_idx}.idx')
            total_nnz = 0

            with open(idx_path, 'wb') as f:
                for sb_start in range(0, n_pairs, sub_batch_size):
                    sb_end = min(sb_start + sub_batch_size, n_pairs)
                    sb_pairs = batch_pairs[sb_start:sb_end]
                    sb_max_nnz = pair_max_nnz[sb_start:sb_end]
                    n_sb = sb_end - sb_start

                    sb_max_out = int(sb_max_nnz.max()) if n_sb > 0 else 0
                    if sb_max_out == 0:
                        continue
                    sb_max_out = min(sb_max_out, N)

                    # Launch kernel
                    pairs_gpu = _cp.asarray(sb_pairs.astype(np.int32).ravel())
                    result_buf = _cp.zeros(n_sb * sb_max_out, dtype=_cp.int32)
                    result_counts = _cp.zeros(n_sb, dtype=_cp.int32)

                    kernel((n_sb,), (1,),
                           (indptr_gpu, indices_gpu, pairs_gpu,
                            result_buf, result_counts, np.int32(sb_max_out)))
                    _cp.cuda.Stream.null.synchronize()

                    counts_cpu = _cp.asnumpy(result_counts)

                    # Stream results to .idx file — NO host-side accumulation
                    for i in range(n_sb):
                        cnt = int(counts_cpu[i])
                        if cnt > 0:
                            offset = i * sb_max_out
                            rows = _cp.asnumpy(result_buf[offset:offset + cnt])
                            # Format: pair_local_idx(int32) | nnz(int32) | row_indices(int32[nnz])
                            f.write(np.int32(sb_start + i).tobytes())
                            f.write(np.int32(cnt).tobytes())
                            f.write(rows.astype(np.int32).tobytes())
                            total_nnz += cnt

                    del pairs_gpu, result_buf, result_counts
                    _cp.get_default_memory_pool().free_all_blocks()

            dt = time.time() - t0
            batches_done += 1

            if total_nnz == 0:
                # Remove empty file
                try:
                    os.remove(idx_path)
                except OSError:
                    pass
                idx_path = None

            result_queue.put((gpu_id, batch_idx, total_nnz, idx_path))

            if batches_done % 3 == 0:
                _log(f"Batch {batch_idx}: {n_pairs} pairs → {total_nnz:,} nnz in {dt:.1f}s "
                     f"({batches_done} batches done)")

    except Exception as e:
        _log(f"FATAL ERROR: {e}")
        traceback.print_exc()
        # Signal error to supervisor via result_queue
        result_queue.put((gpu_id, -1, -1, str(e)))

    finally:
        # Cleanup GPU memory
        try:
            del indptr_gpu, indices_gpu
        except NameError:
            pass
        try:
            _cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
        _log(f"Server exiting ({batches_done} batches processed)")


# ============================================================
# ASYNC NPZ WRITER (double-buffer pattern for non-blocking disk I/O)
# ============================================================

class _AsyncNpzWriter:
    """Background thread for non-blocking NPZ writes. Double-buffer pattern.

    maxsize=2 means the caller blocks on the 3rd enqueue until the writer
    finishes one, giving natural back-pressure (at most 2 CSR matrices
    queued in RAM beyond what the caller holds).
    """
    def __init__(self):
        self._queue = queue.Queue(maxsize=2)
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            item = self._queue.get()
            if item is None:
                break
            path, csr_mat = item
            try:
                _tmp = path + '.tmp'
                sparse.save_npz(_tmp, csr_mat, compressed=False)
                # scipy may append .npz to the filename
                if not _tmp.endswith('.npz'):
                    os.rename(_tmp + '.npz', _tmp)
                os.replace(_tmp, path)
            except Exception as e:
                self._error = e
            finally:
                del csr_mat
            self._queue.task_done()

    def enqueue(self, path, csr_mat):
        if self._error:
            raise self._error
        self._queue.put((path, csr_mat))

    def drain(self):
        self._queue.join()
        if self._error:
            raise self._error

    def stop(self):
        self.drain()
        self._queue.put(None)
        self._thread.join(timeout=60)


# ============================================================
# CUDA SPARSE AND KERNEL — O(nnz) memory cross generation
# ============================================================
# Two-pointer sorted merge on CSC column indices. Memory is O(nnz) not O(rows),
# which prevents OOM on 1d/4h/1h/15m timeframes. Perplexity-validated approach.

_SPARSE_AND_KERNEL = None


def _get_sparse_and_kernel():
    """Lazy-init the CUDA RawKernel for sparse AND (sorted intersection)."""
    global _SPARSE_AND_KERNEL
    if _SPARSE_AND_KERNEL is not None:
        return _SPARSE_AND_KERNEL
    if cp is None:
        return None
    _SPARSE_AND_KERNEL = cp.RawKernel(r"""
    extern "C" __global__
    void sparse_and_batch(
        const int* __restrict__ indptr,
        const int* __restrict__ indices,
        const int* __restrict__ col_pairs,
        int*       result_indices,
        int*       result_counts,
        const int  max_out_per_pair
    ) {
        int pair = blockIdx.x;
        int col_a = col_pairs[pair * 2];
        int col_b = col_pairs[pair * 2 + 1];
        int a = indptr[col_a], a_end = indptr[col_a + 1];
        int b = indptr[col_b], b_end = indptr[col_b + 1];
        int* out = result_indices + pair * max_out_per_pair;
        int cnt = 0;
        while (a < a_end && b < b_end && cnt < max_out_per_pair) {
            int ra = indices[a], rb = indices[b];
            if (ra == rb) { out[cnt++] = ra; a++; b++; }
            else if (ra < rb) { a++; }
            else { b++; }
        }
        result_counts[pair] = cnt;
    }
    """, "sparse_and_batch")
    return _SPARSE_AND_KERNEL


def _sparse_gpu_cross_batch(left_csc, right_csc, valid_pairs, all_names,
                            N, prefix, gpu_id, max_features, feats_so_far):
    """
    GPU sparse cross generation using CUDA RawKernel two-pointer intersection.

    Instead of dense element-wise multiply (O(N×BATCH) VRAM), this operates on
    CSC indices directly: O(nnz) memory. Works for all timeframes including 15m (228K rows).

    Args:
        left_csc:  scipy.sparse.csc_matrix — left feature matrix
        right_csc: scipy.sparse.csc_matrix — right feature matrix
        valid_pairs: np.ndarray (n_pairs, 2) — column index pairs to cross
        all_names: list[str] — pre-built feature names
        N: int — number of rows
        prefix: str — cross type prefix
        gpu_id: int — GPU device ID
        max_features: int or None — cap
        feats_so_far: int — features already generated

    Returns: same format as _gpu_cross_chunk single-GPU path:
        (names, disk_files_or_csr, 'disk' or None, tmp_dir or None, n_cols)
    """
    kernel = _get_sparse_and_kernel()
    if kernel is None:
        return None  # fallback to dense path

    _dev = 0 if os.environ.get('CUDA_VISIBLE_DEVICES') else gpu_id
    cp.cuda.Device(_dev).use()
    cp.cuda.set_pinned_memory_allocator(None)

    n_left_cols = left_csc.shape[1]
    n_right_cols = right_csc.shape[1]
    n_valid = len(valid_pairs)

    # Build unified CSC: stack [left | right] so one indptr/indices array on GPU.
    # Right columns are offset by n_left_cols in the unified matrix.
    combined_csc = sparse.hstack([left_csc, right_csc], format='csc')

    # Upload CSC structure to GPU (just indptr + indices, NOT data — binary AND)
    indptr_gpu = cp.asarray(combined_csc.indptr.astype(np.int32))
    indices_gpu = cp.asarray(combined_csc.indices.astype(np.int32))
    del combined_csc

    # Remap pairs: left col_a stays as-is, right col_b += n_left_cols
    remapped_pairs = valid_pairs.copy()
    remapped_pairs[:, 1] += n_left_cols

    # Estimate max output per pair from co-occurrence counts (upper bound)
    # Use min(nnz(col_a), nnz(col_b)) as upper bound for intersection size
    indptr_cpu = cp.asnumpy(indptr_gpu)
    left_nnz = indptr_cpu[remapped_pairs[:, 0] + 1] - indptr_cpu[remapped_pairs[:, 0]]
    right_nnz = indptr_cpu[remapped_pairs[:, 1] + 1] - indptr_cpu[remapped_pairs[:, 1]]
    pair_max_nnz = np.minimum(left_nnz, right_nnz)

    # Batch processing parameters
    PAIRS_PER_BATCH = 25000  # 25K pairs per kernel launch
    gpu_vram_gb = _get_gpu_vram_gb(gpu_id=_dev)

    # Accumulate results
    names_out = []
    csr_chunks = []
    _disk_csr_files = []
    col_idx = 0

    _ram_gb = _get_available_ram_gb()
    FLUSH_FEATS = max(5000, min(50000, int(_ram_gb * 50)))
    MAX_CSR_IN_RAM = max(2, min(5, int(_ram_gb / 300)))
    _gpu_tmp_dir = os.path.join(os.environ.get('V30_DATA_DIR', '.'),
                                f'_gpu_csr_{prefix}_{os.getpid()}')

    _gpu_writer = _AsyncNpzWriter()

    _coo_names = []
    _coo_rows = []
    _coo_data = []
    _coo_col_local = 0

    def _flush_csr_to_disk():
        if not csr_chunks:
            return
        os.makedirs(_gpu_tmp_dir, exist_ok=True)
        if len(csr_chunks) == 1:
            _merged = csr_chunks[0]
        else:
            _merged = sparse.hstack(csr_chunks, format='csr')
        _path = os.path.join(_gpu_tmp_dir, f'gpu_csr_{len(_disk_csr_files):04d}.npz')
        _disk_csr_files.append(_path)
        csr_chunks.clear()
        _gpu_writer.enqueue(_path, _merged)
        gc.collect()
        _malloc_trim()
        log(f"      Sparse CSR disk flush #{len(_disk_csr_files)}: {col_idx:,} feats total")

    def _flush_coo_to_csr():
        nonlocal _coo_col_local
        if not _coo_rows:
            return
        _r_all = np.concatenate(_coo_rows)
        _c_parts = []
        for _ci, _r in enumerate(_coo_rows):
            _c_parts.append(np.full(len(_r), _ci, dtype=np.int32))
        _c_all = np.concatenate(_c_parts)
        _d_all = np.concatenate(_coo_data)
        _csr = sparse.coo_matrix((_d_all, (_r_all, _c_all)),
                                  shape=(N, _coo_col_local)).tocsr()
        csr_chunks.append(_csr)
        names_out.extend(_coo_names)
        _coo_names.clear()
        _coo_rows.clear()
        _coo_data.clear()
        _coo_col_local = 0
        del _r_all, _c_all, _d_all, _csr, _c_parts
        gc.collect()
        if len(csr_chunks) >= MAX_CSR_IN_RAM:
            _flush_csr_to_disk()

    log(f"  SPARSE KERNEL: {n_valid:,} pairs, batch={PAIRS_PER_BATCH}, "
        f"GPU={gpu_vram_gb:.1f}GB VRAM")

    # FIX #9: CUDA multi-stream — overlap kernel execution with D2H transfer.
    # stream1 runs the kernel for the current sub-batch while stream2 transfers
    # the previous sub-batch's results back to host concurrently.
    stream1 = cp.cuda.Stream(non_blocking=True)  # kernel execution
    stream2 = cp.cuda.Stream(non_blocking=True)  # D2H result transfer
    _prev_result_idx_gpu = None
    _prev_result_cnt_gpu = None
    _prev_pairs_gpu = None
    _prev_sb_max_out = 0
    _prev_n_sb = 0
    _prev_b_start = 0
    _prev_sb_start = 0

    def _drain_prev_results():
        """Transfer and process results from previous sub-batch via stream2."""
        nonlocal _prev_result_idx_gpu, _prev_result_cnt_gpu, _prev_pairs_gpu
        nonlocal _prev_sb_max_out, _prev_n_sb, _prev_b_start, _prev_sb_start
        nonlocal col_idx, _coo_col_local
        if _prev_result_idx_gpu is None:
            return
        # D2H transfer on stream2
        with stream2:
            result_counts = cp.asnumpy(_prev_result_cnt_gpu)
            result_indices_flat = cp.asnumpy(_prev_result_idx_gpu)
        stream2.synchronize()

        for k in range(_prev_n_sb):
            cnt = int(result_counts[k])
            if cnt == 0:
                continue
            global_pair_idx = _prev_b_start + _prev_sb_start + k
            nz_rows = result_indices_flat[k * _prev_sb_max_out: k * _prev_sb_max_out + cnt]
            _coo_names.append(all_names[global_pair_idx])
            _coo_rows.append(nz_rows.copy())
            _coo_data.append(np.ones(cnt, dtype=np.float32))
            _coo_col_local += 1
            col_idx += 1

        del result_counts, result_indices_flat
        del _prev_result_idx_gpu, _prev_result_cnt_gpu, _prev_pairs_gpu
        _prev_result_idx_gpu = None
        _prev_result_cnt_gpu = None
        _prev_pairs_gpu = None
        cp.get_default_memory_pool().free_all_blocks()

    for b_start in range(0, n_valid, PAIRS_PER_BATCH):
        b_end = min(b_start + PAIRS_PER_BATCH, n_valid)
        batch_pairs = remapped_pairs[b_start:b_end]
        batch_max_nnz = pair_max_nnz[b_start:b_end]
        n_batch = b_end - b_start

        max_out = int(batch_max_nnz.max()) if len(batch_max_nnz) > 0 else 0
        if max_out == 0:
            continue
        max_out = min(max_out, N)

        # Account for double-buffering: previous sub-batch results may still be on GPU
        result_bytes = n_batch * max_out * 4
        mem_free = cp.cuda.Device(_dev).mem_info[0]
        _vram_factor = 0.35 if _prev_result_idx_gpu is not None else 0.7
        if result_bytes > mem_free * _vram_factor:
            sub_batch_size = max(100, int(n_batch * (mem_free * (_vram_factor * 0.7)) / result_bytes))
            log(f"    VRAM limit: sub-batching {n_batch} -> {sub_batch_size} pairs "
                f"(result buf would be {result_bytes/1e9:.1f}GB, free={mem_free/1e9:.1f}GB)")
        else:
            sub_batch_size = n_batch

        for sb_start in range(0, n_batch, sub_batch_size):
            sb_end = min(sb_start + sub_batch_size, n_batch)
            sb_pairs = batch_pairs[sb_start:sb_end]
            sb_max_nnz = batch_max_nnz[sb_start:sb_end]
            n_sb = sb_end - sb_start

            sb_max_out = int(sb_max_nnz.max()) if len(sb_max_nnz) > 0 else 0
            if sb_max_out == 0:
                continue
            sb_max_out = min(sb_max_out, N)

            # Upload batch pairs and launch kernel on stream1
            with stream1:
                pairs_gpu = cp.asarray(sb_pairs.astype(np.int32).ravel())
                result_idx_gpu = cp.zeros(n_sb * sb_max_out, dtype=cp.int32)
                result_cnt_gpu = cp.zeros(n_sb, dtype=cp.int32)

                kernel((n_sb,), (1,),
                       (indptr_gpu, indices_gpu, pairs_gpu,
                        result_idx_gpu, result_cnt_gpu, np.int32(sb_max_out)))

            # While stream1 kernel runs, drain previous sub-batch results on stream2
            _drain_prev_results()

            # Wait for current kernel to finish
            stream1.synchronize()

            # Stash current results for next iteration's overlapped D2H transfer
            _prev_result_idx_gpu = result_idx_gpu
            _prev_result_cnt_gpu = result_cnt_gpu
            _prev_pairs_gpu = pairs_gpu
            _prev_sb_max_out = sb_max_out
            _prev_n_sb = n_sb
            _prev_b_start = b_start
            _prev_sb_start = sb_start

        # Periodic flush
        if _coo_col_local >= FLUSH_FEATS:
            _drain_prev_results()
            _flush_coo_to_csr()
            log(f"    Sparse batch flush: {len(csr_chunks)} CSR chunks, "
                f"{col_idx:,}/{n_valid:,} features")

        if max_features and (feats_so_far + col_idx) >= max_features:
            break

    # Drain final pending results and cleanup streams
    _drain_prev_results()
    del stream1, stream2

    # Final flush
    _flush_coo_to_csr()
    if csr_chunks:
        _flush_csr_to_disk()

    _gpu_writer.drain()
    _gpu_writer.stop()

    del indptr_gpu, indices_gpu
    cp.get_default_memory_pool().free_all_blocks()

    log(f"  SPARSE KERNEL done: {col_idx:,} features generated")

    if _disk_csr_files:
        return names_out, _disk_csr_files, 'disk', _gpu_tmp_dir, col_idx
    if csr_chunks:
        return names_out, csr_chunks, None, None, col_idx
    return names_out, [], [], [], col_idx


# ============================================================
# BATCH GPU CROSS MULTIPLICATION (CHUNKED RIGHT-SIDE)
# ============================================================

def gpu_batch_cross(left_names, left_arrays, right_names, right_arrays, prefix,
                    gpu_id=0, min_nonzero=None, max_features=None, col_offset=0,
                    daemon_handles=None):
    """
    Cross every left signal with every right signal using GPU batch multiply.
    Returns (feature_names_list, rows_list, cols_list, data_list, n_new_cols)
    as COO triplet arrays for efficient assembly.

    Right-side is processed in chunks of RIGHT_CHUNK to cap memory.
    col_offset: starting column index for global COO assembly.

    NOTE: The RIGHT_CHUNK loop CANNOT be parallelized across GPUs.
    Each chunk allocates a tensor of shape (n_bars, BATCH, n_right) that
    consumes most of GPU VRAM. Running multiple chunks simultaneously would
    OOM. The chunking exists precisely because the full cross product exceeds
    VRAM. Sequential chunking is the correct approach here — the bottleneck
    is VRAM capacity, not GPU compute utilization.
    """
    if min_nonzero is None:
        min_nonzero = MIN_CO_OCCURRENCE

    if not left_arrays or not right_arrays:
        return [], [], [], [], 0

    N = len(left_arrays[0])
    left_mat = np.column_stack(left_arrays)  # (N, n_left) — usually small

    all_names = []
    all_rows = []
    all_cols = []
    all_data = []
    _local_csr_chunks = []  # CSR chunks flushed per RIGHT_CHUNK
    _disk_npz_files = []    # temp NPZ files for sub-checkpoint flush
    _disk_tmp_dirs = []     # tmp dirs to clean up
    total_feats = 0
    current_offset = col_offset

    # Sub-checkpoint: flush accumulated CSR chunks to disk when chunk count
    # exceeds threshold. Bounds peak RAM regardless of RIGHT_CHUNK iteration count.
    # (RIGHT_CHUNK may be large enough to produce ALL features in 1-2 iterations.)
    _ram_gb = _get_available_ram_gb()
    # Flush when accumulated chunks hit this count. Lower = less RAM, more disk I/O.
    MAX_CHUNKS_IN_RAM = max(100, min(2000, int(_ram_gb * 0.25 / max(0.01, N * 0.001))))
    _tmp_dir = crossgen_scratch_dir(
        'sub_checkpoint',
        symbol=os.environ.get('SAVAGE22_XGEN_SYMBOL'),
        tf=os.environ.get('SAVAGE22_XGEN_TF'),
        prefix=prefix,
        run_id=os.environ.get('SAVAGE22_RUN_ID'),
    )
    # Opt #7H: Time-based flush interval (seconds). Flush accumulated chunks to disk
    # even if count threshold not reached, preventing data loss during long cross types.
    _FLUSH_INTERVAL = int(os.environ.get('V2_FLUSH_INTERVAL_SEC', '1200'))
    _last_time_flush = time.time()

    def _flush_chunks_to_disk():
        """Hstack accumulated CSR chunks, enqueue async NPZ write, clear from RAM."""
        nonlocal _local_csr_chunks
        if not _local_csr_chunks:
            return
        os.makedirs(_tmp_dir, exist_ok=True)
        if len(_local_csr_chunks) == 1:
            _merged = _local_csr_chunks[0]
        else:
            _merged = sparse.hstack(_local_csr_chunks, format='csr')
        _npz_path = os.path.join(_tmp_dir, f'sub_{len(_disk_npz_files):04d}.npz')
        _disk_npz_files.append(_npz_path)
        _n_flushed = sum(c.shape[1] for c in _local_csr_chunks)
        _local_csr_chunks.clear()
        # Async write — writer thread owns _merged from here
        _writer.enqueue(_npz_path, _merged)
        gc.collect()
        _malloc_trim()
        log(f"      Sub-flush #{len(_disk_npz_files)}: {_n_flushed:,} cols to disk (async), RAM freed")

    # Async NPZ writer — overlaps disk I/O with next chunk's compute
    _writer = _AsyncNpzWriter()

    # Process right-side in adaptive chunks (was static RIGHT_CHUNK)
    _chunk_ctrl = AdaptiveChunkController(max_cap=RIGHT_CHUNK, target_pct=_RAM_CEILING_PCT)
    rc_start = 0
    # FIX #40: Disable GC during batch multiply loop — 100+ gc.collect() calls in hot paths
    # add ~5-10% overhead. We do explicit gc.collect() only at chunk boundaries and flush points.
    gc.disable()
    while rc_start < len(right_names):
        _chunk_size = _chunk_ctrl.get_chunk_size()
        rc_end = min(rc_start + _chunk_size, len(right_names))
        r_names_chunk = right_names[rc_start:rc_end]
        r_arrays_chunk = right_arrays[rc_start:rc_end]
        _rss_before = _get_rss_bytes()
        try:
            right_mat_chunk = np.column_stack(r_arrays_chunk)  # (N, <=chunk_size)

            # GPU now uses sparse matmul pre-filter — no 3D tensor, works for ANY row count.
            if GPU and os.environ.get('V2_SKIP_GPU') != '1':
                c_names, c_rows, c_cols, c_data, c_ncols = _gpu_cross_chunk(
                    left_names, left_mat, r_names_chunk, right_mat_chunk,
                    prefix, gpu_id, min_nonzero, max_features, total_feats,
                    col_offset=current_offset,
                    daemon_handles=daemon_handles
                )
            elif _USE_NUMBA_CROSS:
                c_names, c_rows, c_cols, c_data, c_ncols = _numba_cross_chunk(
                    left_names, left_mat, r_names_chunk, right_mat_chunk,
                    prefix, min_nonzero, max_features, total_feats,
                    col_offset=current_offset
                )
            else:
                c_names, c_rows, c_cols, c_data, c_ncols = _cpu_cross_chunk(
                    left_names, left_mat, r_names_chunk, right_mat_chunk,
                    prefix, min_nonzero, max_features, total_feats,
                    col_offset=current_offset
                )

            if c_names:
                all_names.extend(c_names)
                # Check if return is already CSR chunks (from GPU/CPU flush path)
                # or disk-backed paths — pass through directly to local lists
                if c_cols == 'disk':
                    # Disk-backed: c_rows = list of NPZ paths, c_data = tmp_dir
                    _disk_npz_files.extend(c_rows)
                    _disk_tmp_dirs.append(c_data)
                    current_offset += c_ncols
                    total_feats += len(c_names)
                    c_rows = c_cols = c_data = c_names = None
                    gc.collect()
                elif c_rows and hasattr(c_rows[0], 'indptr'):
                    # Already CSR chunks — append to local list
                    _local_csr_chunks.extend(c_rows)
                    current_offset += c_ncols
                    total_feats += len(c_names)
                    c_rows = c_cols = c_data = c_names = None
                    gc.collect()
                elif c_rows:
                    # Legacy COO path — convert to CSR
                    _total_nnz = sum(len(r) for r in c_rows)
                    _avg_nnz = _total_nnz / max(1, len(c_rows))
                    # Max entries per sub-batch concat = 125M (500MB / 4 bytes)
                    _max_entries = 125_000_000
                    SUB_FLUSH = max(100, min(5000, int(_max_entries / max(1, _avg_nnz))))

                    n_cols_total = len(c_rows)
                    for sf_start in range(0, n_cols_total, SUB_FLUSH):
                        sf_end = min(sf_start + SUB_FLUSH, n_cols_total)
                        sf_n_cols = sf_end - sf_start
                        # Build local COO with column indices 0..sf_n_cols-1
                        sf_r_parts = []
                        sf_c_parts = []
                        sf_d_parts = []
                        for local_col, global_col in enumerate(range(sf_start, sf_end)):
                            r_arr = c_rows[global_col]
                            if len(r_arr) > 0:
                                sf_r_parts.append(r_arr)
                                sf_c_parts.append(np.full(len(r_arr), local_col, dtype=np.int32))
                                sf_d_parts.append(c_data[global_col])
                        if sf_r_parts:
                            _sf_r = np.concatenate(sf_r_parts)
                            _sf_c = np.concatenate(sf_c_parts)
                            _sf_d = np.concatenate(sf_d_parts)
                            if cp is not None and cusp is not None:
                                try:
                                    _sf_d_gpu = cp.asarray(_sf_d)
                                    _sf_r_gpu = cp.asarray(_sf_r)
                                    _sf_c_gpu = cp.asarray(_sf_c)
                                    _sf_csr = cusp.coo_matrix((_sf_d_gpu, (_sf_r_gpu, _sf_c_gpu)), shape=(N, sf_n_cols)).tocsr()
                                    _sf_csr = _sf_csr.get()  # Transfer back to scipy CSR on CPU
                                    del _sf_d_gpu, _sf_r_gpu, _sf_c_gpu
                                    cp.get_default_memory_pool().free_all_blocks()
                                except Exception:
                                    _sf_csr = sparse.coo_matrix((_sf_d, (_sf_r, _sf_c)), shape=(N, sf_n_cols)).tocsr()
                            else:
                                _sf_csr = sparse.coo_matrix((_sf_d, (_sf_r, _sf_c)), shape=(N, sf_n_cols)).tocsr()
                            _local_csr_chunks.append(_sf_csr)
                            del _sf_r, _sf_c, _sf_d, _sf_csr
                        del sf_r_parts, sf_c_parts, sf_d_parts
                        gc.collect()
                    current_offset += c_ncols
                    total_feats += len(c_names)

            del right_mat_chunk, r_arrays_chunk
            c_names = c_rows = c_cols = c_data = None
            _rss_after = _get_rss_bytes()
            _chunk_ctrl.record_chunk(rc_end - rc_start, _rss_before, _rss_after)
            gc.collect()
            _malloc_trim()

            # Sub-checkpoint: flush accumulated CSR chunks to disk when count exceeds threshold
            # OR when time interval elapsed (Opt #7H: prevents data loss during long runs)
            _time_since_flush = time.time() - _last_time_flush
            if len(_local_csr_chunks) >= MAX_CHUNKS_IN_RAM or (
                    _local_csr_chunks and _time_since_flush >= _FLUSH_INTERVAL):
                if _time_since_flush >= _FLUSH_INTERVAL:
                    log(f"      Time flush ({_time_since_flush:.0f}s): {len(_local_csr_chunks)} chunks to disk")
                _flush_chunks_to_disk()
                _last_time_flush = time.time()

            if max_features and total_feats >= max_features:
                break

            rc_start = rc_end  # advance to next chunk

        except MemoryError:
            # Adaptive retry: halve chunk size, retry same rc_start
            del r_names_chunk, r_arrays_chunk
            gc.collect()
            _malloc_trim()
            _chunk_ctrl.halve()
            log(f"      MemoryError at rc_start={rc_start}, retrying with chunk={_chunk_ctrl.get_chunk_size()}")
            continue

    del left_mat
    gc.enable()  # FIX #40: Re-enable GC after batch multiply loop
    gc.collect()

    # Flush any remaining in-memory chunks to disk
    if _disk_npz_files and _local_csr_chunks:
        _flush_chunks_to_disk()

    # Wait for all async NPZ writes to complete before returning disk paths
    _writer.drain()
    _writer.stop()

    n_total_cols = current_offset - col_offset

    if _disk_npz_files:
        # Return disk file paths instead of in-memory CSR chunks.
        # Caller (_collect_cross / _save_checkpoint) streams from disk.
        # Combine all tmp dirs for cleanup
        _all_tmp_dirs = list(set(_disk_tmp_dirs + ([_tmp_dir] if os.path.exists(_tmp_dir) else [])))
        log(f"    {len(_disk_npz_files)} sub-checkpoints on disk, {n_total_cols:,} total cols")
        return all_names, _disk_npz_files, 'disk', _all_tmp_dirs, n_total_cols
    # Return CSR chunks if we flushed (memory-safe path)
    if _local_csr_chunks:
        return all_names, _local_csr_chunks, None, None, n_total_cols
    return all_names, all_rows, all_cols, all_data, n_total_cols


def _gpu_cross_chunk(left_names, left_mat, right_names, right_mat, prefix,
                     gpu_id, min_nonzero, max_features, feats_so_far,
                     col_offset=0, daemon_handles=None):
    """
    GPU cross multiply with co-occurrence pre-filter.
    Step 1: Bitpacked POPCNT (or sparse matmul fallback) for co-occurrence filtering.
    Step 2: GPU element-wise multiply ONLY for valid pairs.
    No 3D tensor — works for ANY row count including 217K+ (15m).

    When daemon_handles is provided (V4 architecture), dispatches to persistent
    GPU daemons via Pipe IPC. Otherwise falls through to legacy paths.
    """
    names = []
    rows_list = []
    cols_list = []
    data_list = []
    col_idx = 0

    # ── Step 1: Co-occurrence pre-filter (bitpack POPCNT or sparse matmul) ──
    valid_pairs, _co_method, co_occur = _compute_cooccurrence_pairs(left_mat, right_mat, min_nonzero)
    n_valid = len(valid_pairs)

    if n_valid == 0:
        return names, rows_list, cols_list, data_list, col_idx

    # Sort pairs by co-occurrence count (nnz) descending — heavy pairs first.
    # Fixes load imbalance in prange: heavy work up front, light pairs fill in at the tail.
    pair_nnz = co_occur[valid_pairs[:, 0], valid_pairs[:, 1]]
    sort_idx = np.argsort(-pair_nnz)
    valid_pairs = valid_pairs[sort_idx]

    gpu_vram_gb = _get_gpu_vram_gb(gpu_id=gpu_id)
    log(f"  GPU VRAM: {gpu_vram_gb:.1f} GB | {n_valid} valid pairs (pre-filtered from {left_mat.shape[1]}×{right_mat.shape[1]})")

    # Pre-build all feature names (preserves exact order)
    all_names = [f'{prefix}_{left_names[int(p[0])][:40]}_{right_names[int(p[1])][:40]}'
                 for p in valid_pairs]

    # ── V4 DAEMON DISPATCH PATH (persistent daemons, zero scipy in GPU workers) ──
    # When daemon_handles is provided, use the new architecture:
    # - Daemons are ALREADY running (started in generate_all_crosses)
    # - RELOAD uploads new CSC once per cross step (not 48× per RIGHT_CHUNK)
    # - Batches dispatched via Pipe IPC, results as .idx files
    # - CSR built from .idx files AFTER all GPU work completes
    # SIDE-CHANNEL DEBUG: write to file to bypass run_tee buffering
    with open('/tmp/xgen_daemon_debug.log', 'a') as _dbg:
        _dbg.write(f"[{time.strftime('%H:%M:%S')}] _gpu_cross_chunk: daemon_handles={'PRESENT('+str(len(daemon_handles))+')' if daemon_handles else 'None'}, n_valid={n_valid}\n")
        if daemon_handles:
            for _h in daemon_handles:
                _dbg.write(f"  GPU-{_h.gpu_id}: status={_h.status}, alive={_h.process.is_alive()}\n")
        _dbg.flush()
    print(f"[XGEN-DEBUG] _gpu_cross_chunk: daemon_handles={'PRESENT('+str(len(daemon_handles))+')' if daemon_handles else 'None'}, n_valid={n_valid}", flush=True)
    if daemon_handles is not None and len(daemon_handles) >= 2:
        n_active = sum(1 for h in daemon_handles if h.status != 'dead')
        with open('/tmp/xgen_daemon_debug.log', 'a') as _dbg:
            _dbg.write(f"  n_active={n_active}, threshold={n_active * 100}, n_valid={n_valid}\n")
            _dbg.flush()
        if n_active >= 2 and n_valid >= n_active * 100:
            log(f"  V4 DAEMON DISPATCH: {n_valid:,} pairs → {n_active} persistent daemons")
            try:
                from cross_supervisor import run_cross_step, build_csr_from_idx_files

                N = left_mat.shape[0]
                _out_dir = os.environ.get('V30_DATA_DIR', '.')
                _run_id = os.environ.get('SAVAGE22_RUN_ID')
                _symbol = os.environ.get('SAVAGE22_XGEN_SYMBOL')
                _tf = os.environ.get('SAVAGE22_XGEN_TF')

                _step_result = run_cross_step(
                    handles=daemon_handles,
                    left_mat=left_mat,
                    right_mat=right_mat,
                    valid_pairs=valid_pairs,
                    all_names=all_names,
                    N=N,
                    out_dir=_out_dir,
                    prefix=prefix,
                    symbol=_symbol,
                    tf=_tf,
                    run_id=_run_id,
                )
                # IPC contract drift guard: accept both legacy 2-tuple and
                # newer 3-tuple return signatures from run_cross_step().
                if not isinstance(_step_result, tuple):
                    raise RuntimeError(
                        f"run_cross_step returned non-tuple: {type(_step_result)}"
                    )
                if len(_step_result) == 3:
                    idx_files, total_nnz, _ = _step_result
                elif len(_step_result) == 2:
                    idx_files, total_nnz = _step_result
                else:
                    raise RuntimeError(
                        f"run_cross_step returned {len(_step_result)} values; expected 2 or 3"
                    )

                if not idx_files:
                    log(f"  V4 daemon dispatch: no .idx files (0 nnz)")
                    return all_names[:0], [], [], [], 0

                # Build CSR from .idx files (Stage 3)
                merged_names, merged_npz, total_cols = build_csr_from_idx_files(
                    idx_files=idx_files,
                    all_names=all_names,
                    N=N,
                    out_dir=_out_dir,
                    symbol=_symbol,
                    tf=_tf,
                    prefix=prefix,
                    run_id=_run_id,
                )
                if not merged_npz:
                    raise RuntimeError("build_csr_from_idx_files returned empty output path")

                # Persist each assembled chunk under a unique name so later checkpoint
                # assembly does not accidentally reference the same merged file path
                # multiple times after subsequent dispatches overwrite it.
                _dispatch_dir = crossgen_scratch_dir(
                    'dispatch_merge',
                    symbol=_symbol,
                    tf=_tf,
                    prefix=prefix,
                    run_id=_run_id,
                )
                out_path = os.path.join(
                    _dispatch_dir,
                    f"{get_crossgen_namespace(symbol=_symbol, tf=_tf, prefix=prefix, run_id=_run_id)}__dispatch_{time.time_ns()}.npz"
                )
                os.replace(merged_npz, out_path)
                gc.collect()

                log(f"  V4 DAEMON DONE: {total_cols:,} features from {n_active} GPUs, "
                    f"{total_nnz:,} total nnz")
                _scratch_dirs = []
                if idx_files:
                    _scratch_dirs.append(os.path.dirname(idx_files[0]))
                _assembly_dir = os.path.dirname(merged_npz)
                if _assembly_dir != _dispatch_dir:
                    _scratch_dirs.append(_assembly_dir)
                _scratch_dirs.append(_dispatch_dir)
                return merged_names[:total_cols], [out_path], 'disk', _scratch_dirs, total_cols

            except Exception as _v4_err:
                log(f"  V4 daemon dispatch FAILED ({_v4_err}), aborting fast-path run")
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"Persistent GPU daemon path failed for {prefix}: {_v4_err}") from _v4_err

    # ── Legacy Multi-GPU dispatch (Step 2 across N GPUs) ──
    available_gpus = _detect_available_gpus()
    n_gpus = len(available_gpus) if _MULTI_GPU_CROSS_GEN else 1

    if n_gpus > 1 and n_valid >= n_gpus * 100:
        import multiprocessing as _mp
        log(f"  MULTI-GPU: sharding {n_valid:,} pairs across {n_gpus} GPUs ({available_gpus})")

        N = left_mat.shape[0]
        # ── Preflight memory estimate — throttle BEFORE launching ──
        _density = np.count_nonzero(right_mat) / max(right_mat.size, 1)
        _max_conc = _preflight_check(
            tf_name=prefix, n_rows=N,
            n_cols_left=left_mat.shape[1], n_cols_right=right_mat.shape[1],
            density=_density, n_valid_pairs=n_valid,
            n_gpus=n_gpus, gpu_vram_gb=gpu_vram_gb,
        )
        if _max_conc < n_gpus:
            log(f"  PREFLIGHT THROTTLE: capping workers {n_gpus} → {_max_conc}")
            n_gpus = _max_conc
            available_gpus = available_gpus[:n_gpus]
        _mgpu_tmp = crossgen_scratch_dir(
            'legacy_mgpu',
            symbol=os.environ.get('SAVAGE22_XGEN_SYMBOL'),
            tf=os.environ.get('SAVAGE22_XGEN_TF'),
            prefix=prefix,
            run_id=os.environ.get('SAVAGE22_RUN_ID'),
        )

        # Save shared data to temp .npy for workers (mmap-friendly)
        _left_path = os.path.join(_mgpu_tmp, 'left_mat.npy')
        _right_path = os.path.join(_mgpu_tmp, 'right_mat.npy')
        _pairs_path = os.path.join(_mgpu_tmp, 'valid_pairs.npy')
        _names_path = os.path.join(_mgpu_tmp, 'all_names.npy')
        np.save(_left_path, np.ascontiguousarray(left_mat))
        np.save(_right_path, np.ascontiguousarray(right_mat))
        np.save(_pairs_path, valid_pairs)
        np.save(_names_path, np.array(all_names, dtype=object))

        # ── PERSISTENT GPU SERVER ARCHITECTURE ──
        # Each GPU gets a long-lived server process (CUDA init once).
        # Pairs are split into small batches dispatched via Queue.
        # Servers write raw index files (ZERO scipy) → constant memory.
        # Post-processing builds CSR after all servers exit (full RAM available).
        import json as _json
        ctx = _mp.get_context('spawn')
        n_left_cols = left_mat.shape[1]

        # Batch sizing: split pairs into chunks of 5000
        BATCH_SIZE = 5000
        all_batches = []
        for b_start in range(0, n_valid, BATCH_SIZE):
            b_end = min(b_start + BATCH_SIZE, n_valid)
            all_batches.append((valid_pairs[b_start:b_end], b_start // BATCH_SIZE))

        log(f"  GPU servers: {n_gpus} GPUs, {len(all_batches)} batches of {BATCH_SIZE}")

        # Create queues
        work_queue = ctx.Queue()
        result_queue = ctx.Queue()

        # Start persistent GPU servers
        servers = []
        for g_id in available_gpus:
            p = ctx.Process(
                target=_gpu_server,
                args=(g_id, work_queue, result_queue,
                      _left_path, _right_path, n_left_cols, N, _mgpu_tmp),
                daemon=False,
            )
            p.start()
            servers.append((g_id, p))
            time.sleep(1)  # stagger CUDA context init

        # Dispatch all batches round-robin
        for batch_pairs, batch_idx in all_batches:
            # Remap pairs: left cols stay [0,n_left), right cols become [n_left, n_left+n_right)
            remapped = batch_pairs.copy()
            remapped[:, 1] += n_left_cols
            work_queue.put((remapped, batch_idx))

        # Send poison pills
        for _ in servers:
            work_queue.put(None)

        # Collect results
        completed_batches = 0
        idx_files = []
        failed = False
        while completed_batches < len(all_batches):
            try:
                gpu_id, batch_idx, nnz, path = result_queue.get(timeout=600)
                if batch_idx == -1:  # error signal
                    log(f"  ERROR from GPU-{gpu_id}: {path}")
                    failed = True
                    break
                if path:
                    idx_files.append(path)
                completed_batches += 1
                if completed_batches % 2 == 0 or completed_batches == len(all_batches):
                    log(f"  Batches: {completed_batches}/{len(all_batches)} "
                        f"({nnz} nnz in last batch)")
            except Exception as e:
                log(f"  Queue timeout or error: {e}")
                failed = True
                break

        # Join servers
        for g_id, p in servers:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()

        if failed:
            shutil.rmtree(_mgpu_tmp, ignore_errors=True)
            raise RuntimeError("GPU server cross gen failed. Check logs above.")

        # ── POST-PROCESSING: Build CSR from raw index files ──
        # All GPU servers have exited — full RAM available for merge.
        log(f"  Building CSR from {len(idx_files)} index files...")
        all_coo_rows = []
        all_coo_cols = []
        col_offset = 0
        merged_names = list(all_names)

        for idx_path in sorted(idx_files):
            with open(idx_path, 'rb') as f:
                data = f.read()
            pos = 0
            while pos < len(data):
                pair_local = int(np.frombuffer(data[pos:pos+4], dtype=np.int32)[0])
                pos += 4
                nnz = int(np.frombuffer(data[pos:pos+4], dtype=np.int32)[0])
                pos += 4
                rows = np.frombuffer(data[pos:pos+nnz*4], dtype=np.int32)
                pos += nnz * 4
                # pair_local is index into the batch — need global column index
                # For now, use sequential column assignment
                all_coo_rows.append(rows)
                all_coo_cols.append(np.full(nnz, col_offset, dtype=np.int32))
                col_offset += 1

        total_cols = col_offset
        if all_coo_rows:
            _r = np.concatenate(all_coo_rows)
            _c = np.concatenate(all_coo_cols)
            _d = np.ones(len(_r), dtype=np.float32)
            final_csr = sparse.csr_matrix((_d, (_r, _c)), shape=(N, total_cols))
            del _r, _c, _d, all_coo_rows, all_coo_cols
        else:
            final_csr = sparse.csr_matrix((N, 0), dtype=np.float32)

        # Clean up index files + temp dir
        shutil.rmtree(_mgpu_tmp, ignore_errors=True)

        log(f"  MULTI-GPU done: {total_cols:,} features from {n_gpus} GPUs")

        # Save merged CSR
        _mgpu_out_dir = os.path.join(os.environ.get('V30_DATA_DIR', '.'),
                                     f'_gpu_csr_{prefix}_{os.getpid()}')
        os.makedirs(_mgpu_out_dir, exist_ok=True)
        out_path = os.path.join(_mgpu_out_dir, 'gpu_csr_0000.npz')
        sparse.save_npz(out_path, final_csr, compressed=False)
        del final_csr
        gc.collect()
        return merged_names[:total_cols], [out_path], 'disk', _mgpu_out_dir, total_cols

    # ── Step 2a: CUDA sparse kernel path (O(nnz) memory — preferred) ──
    # Uses two-pointer sorted merge on CSC indices instead of dense multiply.
    # Memory is O(nnz) not O(rows×batch), preventing OOM on 1h/15m timeframes.
    if cp is not None and _get_sparse_and_kernel() is not None:
        try:
            left_csc = sparse.csc_matrix(left_mat.astype(np.float32))
            right_csc = sparse.csc_matrix(right_mat.astype(np.float32))
            result = _sparse_gpu_cross_batch(
                left_csc, right_csc, valid_pairs, all_names,
                left_mat.shape[0], prefix, gpu_id, max_features, feats_so_far)
            if result is not None:
                return result
            log("  Sparse kernel returned None — falling back to dense GPU path")
        except Exception as _sparse_err:
            log(f"  Sparse kernel failed ({_sparse_err}) — falling back to dense GPU path")

    # ── Step 2b: Dense GPU batch multiply (fallback single-GPU path) ──
    # Flush COO→CSR every FLUSH_INTERVAL batches to prevent unbounded RAM growth.
    # Without this, 1.38M features accumulate ~16GB+ of COO arrays before returning.
    _dev = 0 if os.environ.get('CUDA_VISIBLE_DEVICES') else gpu_id
    cp.cuda.Device(_dev).use()
    cp.cuda.set_pinned_memory_allocator(None)  # use pageable memory, saves host RAM on single-GPU path
    left_gpu = cp.asarray(np.ascontiguousarray(left_mat))
    right_gpu = cp.asarray(np.ascontiguousarray(right_mat))

    N = left_mat.shape[0]
    avail_vram = int(gpu_vram_gb * 0.5 * 1024**3)
    bytes_per_pair = N * 12
    BATCH = max(100, min(50000, avail_vram // max(1, bytes_per_pair)))

    # Accumulate CSR chunks, flush to disk when too many to prevent OOM.
    _csr_out = []  # list of CSR matrices in RAM
    _disk_csr_files = []  # list of NPZ file paths flushed to disk
    _coo_names = []
    _coo_rows = []
    _coo_data = []
    _coo_col_local = 0

    _ram_gb = _get_available_ram_gb()
    FLUSH_FEATS = max(5000, min(50000, int(_ram_gb * 50)))  # COO→CSR every ~37K features
    # Flush CSR chunks to disk every N chunks to prevent _csr_out from eating all RAM
    MAX_CSR_IN_RAM = max(2, min(5, int(_ram_gb / 300)))  # ~2 on 755GB, ~6 on 2TB
    _gpu_tmp_dir = os.path.join(os.environ.get('V30_DATA_DIR', '.'),
                                f'_gpu_csr_{prefix}_{os.getpid()}')

    # Async NPZ writer for GPU cross chunk — overlaps disk I/O with GPU compute
    _gpu_writer = _AsyncNpzWriter()

    def _flush_csr_to_disk():
        """Save accumulated CSR chunks to disk NPZ (async), clear from RAM."""
        if not _csr_out:
            return
        os.makedirs(_gpu_tmp_dir, exist_ok=True)
        if len(_csr_out) == 1:
            _merged = _csr_out[0]
        else:
            _merged = sparse.hstack(_csr_out, format='csr')
        _path = os.path.join(_gpu_tmp_dir, f'gpu_csr_{len(_disk_csr_files):04d}.npz')
        _disk_csr_files.append(_path)
        _csr_out.clear()
        # Async write — writer thread owns _merged from here
        _gpu_writer.enqueue(_path, _merged)
        gc.collect()
        _malloc_trim()
        log(f"      CSR disk flush #{len(_disk_csr_files)}: {col_idx:,} feats total, RAM freed (async)")

    def _flush_coo_to_csr():
        """Convert accumulated COO arrays to CSR, append to _csr_out, clear COO."""
        nonlocal _coo_col_local
        if not _coo_rows:
            return
        _r_all = np.concatenate(_coo_rows)
        _c_all_parts = []
        for _ci, _r in enumerate(_coo_rows):
            _c_all_parts.append(np.full(len(_r), _ci, dtype=np.int32))
        _c_all = np.concatenate(_c_all_parts)
        _d_all = np.concatenate(_coo_data)
        _csr = sparse.coo_matrix((_d_all, (_r_all, _c_all)),
                                  shape=(N, _coo_col_local)).tocsr()
        _csr_out.append(_csr)
        names.extend(_coo_names)
        _coo_names.clear()
        _coo_rows.clear()
        _coo_data.clear()
        _coo_col_local = 0
        del _r_all, _c_all, _d_all, _csr, _c_all_parts
        gc.collect()
        # Flush CSR to disk if too many accumulated
        if len(_csr_out) >= MAX_CSR_IN_RAM:
            _flush_csr_to_disk()

    for b_start in range(0, n_valid, BATCH):
        b_end = min(b_start + BATCH, n_valid)
        chunk = valid_pairs[b_start:b_end]

        # Vectorized batch multiply on GPU
        left_cols = left_gpu[:, chunk[:, 0]]
        right_cols = right_gpu[:, chunk[:, 1]]
        crosses_gpu = left_cols * right_cols
        del left_cols, right_cols

        # GPU nonzero extraction
        nz_rows_gpu, nz_cols_gpu = cp.nonzero(crosses_gpu)
        nz_rows_all = cp.asnumpy(nz_rows_gpu)
        nz_cols_all = cp.asnumpy(nz_cols_gpu)
        del nz_rows_gpu, nz_cols_gpu

        del crosses_gpu
        cp.cuda.Stream.null.synchronize()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        cp.get_default_memory_pool().free_all_blocks()

        if len(nz_rows_all) > 0:
            unique_cols, col_starts = np.unique(nz_cols_all, return_index=True)
            col_ends = np.append(col_starts[1:], len(nz_cols_all))

            for k in range(len(unique_cols)):
                c = int(unique_cols[k])
                s, e = int(col_starts[k]), int(col_ends[k])
                nz = nz_rows_all[s:e]
                _coo_names.append(all_names[b_start + c])
                _coo_rows.append(nz)
                _coo_data.append(np.ones(len(nz), dtype=np.float32))
                _coo_col_local += 1
                col_idx += 1

        # Periodic flush: convert accumulated COO to CSR to free RAM
        if _coo_col_local >= FLUSH_FEATS:
            _flush_coo_to_csr()
            log(f"    GPU batch flush: {len(_csr_out)} CSR chunks, {col_idx:,}/{n_valid:,} features")

        if max_features and (feats_so_far + col_idx) >= max_features:
            break

    # Final flush of remaining COO → CSR → disk
    _flush_coo_to_csr()
    if _csr_out:
        _flush_csr_to_disk()

    # Wait for all async GPU NPZ writes to complete before returning
    _gpu_writer.drain()
    _gpu_writer.stop()

    del left_gpu, right_gpu
    cp.get_default_memory_pool().free_all_blocks()

    # Return disk-backed CSR files if any were created
    if _disk_csr_files:
        return names, _disk_csr_files, 'disk', _gpu_tmp_dir, col_idx
    # Return in-memory CSR chunks
    if _csr_out:
        return names, _csr_out, None, None, col_idx
    return names, rows_list, cols_list, data_list, col_idx


# ============================================================
# NUMBA CSC CROSS CHUNK (Optimization #1 + #6)
# ============================================================

def _numba_cross_chunk(left_names, left_mat, right_names, right_mat, prefix,
                       min_nonzero, max_features, feats_so_far, col_offset=0):
    """
    Numba CSC sorted-index intersection cross chunk.
    Drop-in replacement for _cpu_cross_chunk when USE_NUMBA_CROSS=1.

    Instead of dense multiply + np.nonzero, converts to CSC and does
    two-pointer sorted intersection. L2 cache-friendly pair sorting
    keeps the same left column hot across all its right partners.

    Returns same format as _cpu_cross_chunk: (names, csr_chunks, None, None, n_cols)
    """
    N = left_mat.shape[0]

    # ── Sparse matmul pre-filter for co-occurrence (same as _cpu_cross_chunk) ──
    if cp is not None and cusp is not None:
        try:
            left_gpu_sp = cusp.csc_matrix(cp.asarray(left_mat.astype(np.float32)))
            right_gpu_sp = cusp.csc_matrix(cp.asarray(right_mat.astype(np.float32)))
            co_occur = cp.asnumpy((left_gpu_sp.T @ right_gpu_sp).toarray())
            del left_gpu_sp, right_gpu_sp
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            left_sp = sparse.csr_matrix(left_mat.astype(np.float32))
            right_sp = sparse.csr_matrix(right_mat.astype(np.float32))
            if _USE_MKL:
                co_occur = _mkl_dot(left_sp.T, right_sp).toarray()
            else:
                co_occur = (left_sp.T @ right_sp).toarray()
    else:
        left_sp = sparse.csr_matrix(left_mat.astype(np.float32))
        right_sp = sparse.csr_matrix(right_mat.astype(np.float32))
        if _USE_MKL:
            co_occur = _mkl_dot(left_sp.T, right_sp).toarray()
        else:
            co_occur = (left_sp.T @ right_sp).toarray()

    valid_pairs = np.argwhere(co_occur >= min_nonzero)
    if len(valid_pairs) == 0:
        return [], [], [], [], 0

    # Apply max_features limit
    if max_features is not None:
        remaining = max_features - feats_so_far
        if remaining <= 0:
            return [], [], [], [], 0
        if len(valid_pairs) > remaining:
            valid_pairs = valid_pairs[:remaining]

    all_names = [f'{prefix}_{left_names[int(p[0])][:40]}_{right_names[int(p[1])][:40]}'
                 for p in valid_pairs]

    # ── Numba CSC intersection (Opt #1 + #6) ──
    names, csr_chunks, n_features = numba_csc_cross(
        left_mat, right_mat, valid_pairs, all_names,
        n_rows=N, min_nonzero=min_nonzero, progress=True
    )

    if not names:
        return [], [], [], [], 0

    return names, csr_chunks, None, None, n_features



@njit(parallel=True, cache=True, fastmath=True)
def _parallel_cross_multiply(left, right, out):
    """fastmath=True is safe here: inputs are binary 0/1, no NaN, no subnormals."""
    n_rows, n_pairs = left.shape
    for j in prange(n_pairs):
        for i in range(n_rows):
            out[i, j] = left[i, j] * right[i, j]


def _process_cross_block(left_mat, right_mat, valid_pairs, all_names,
                         b_start, b_end, col_offset_base):
    """
    Process a block of cross pairs. Thread-safe — numpy element-wise ops release GIL.
    Returns (names, rows_list, cols_list, data_list) for this block.
    """
    chunk = valid_pairs[b_start:b_end]
    left_cols = np.ascontiguousarray(left_mat[:, chunk[:, 0]])     # (N, chunk_size)
    right_cols = np.ascontiguousarray(right_mat[:, chunk[:, 1]])   # (N, chunk_size)
    crosses = np.empty_like(left_cols)
    _parallel_cross_multiply(left_cols, right_cols, crosses)       # Numba parallel prange

    nz_rows_all, nz_cols_all = np.nonzero(crosses)
    del crosses  # free dense array early

    names = []
    rows_list = []
    cols_list = []
    data_list = []

    if len(nz_rows_all) > 0:
        unique_cols, col_starts = np.unique(nz_cols_all, return_index=True)
        col_ends = np.append(col_starts[1:], len(nz_cols_all))

        for k in range(len(unique_cols)):
            c = int(unique_cols[k])
            s, e = int(col_starts[k]), int(col_ends[k])
            nz = nz_rows_all[s:e]
            names.append(all_names[b_start + c])
            rows_list.append(nz)
            # col indices assigned later during merge to ensure correct global ordering
            data_list.append(np.ones(len(nz), dtype=np.float32))

    return names, rows_list, data_list


def _cpu_cross_chunk(left_names, left_mat, right_names, right_mat, prefix,
                     min_nonzero, max_features, feats_so_far, col_offset=0):
    """
    CPU cross multiply with co-occurrence pre-filter + MULTI-THREADED execution.
    Step 1: Bitpacked POPCNT (or sparse matmul fallback) for co-occurrence filtering.
    Step 2: Only compute actual crosses for valid pairs (skip ~92% of work).
    Step 3: Parallel thread execution — numpy element-wise ops release GIL.
    Returns COO triplets directly, never stores dense columns.
    """
    names = []
    rows_list = []
    cols_list = []
    data_list = []
    col_idx = 0

    # ── Co-occurrence pre-filter (bitpack POPCNT or sparse matmul) ──
    valid_pairs, _co_method, co_occur = _compute_cooccurrence_pairs(left_mat, right_mat, min_nonzero)

    if len(valid_pairs) == 0:
        return names, rows_list, cols_list, data_list, col_idx

    # Sort pairs by co-occurrence count (nnz) descending — heavy pairs first.
    # Fixes load imbalance in parallel batches: heavy work up front, light pairs fill in.
    pair_nnz = co_occur[valid_pairs[:, 0], valid_pairs[:, 1]]
    sort_idx = np.argsort(-pair_nnz)
    valid_pairs = valid_pairs[sort_idx]

    all_names = [f'{prefix}_{left_names[int(p[0])][:40]}_{right_names[int(p[1])][:40]}'
                 for p in valid_pairs]

    # ── Multi-threaded batch cross multiply ──
    # numpy element-wise * releases the GIL, so threads give true parallelism.
    # Cap at 64 threads — memory bandwidth saturates before that on most machines.
    N = left_mat.shape[0]
    n_valid = len(valid_pairs)

    # Determine thread count first, then size batches to fill threads
    try:
        from hardware_detect import get_cpu_count
        n_cpus = get_cpu_count()
    except ImportError:
        try:
            n_cpus = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            n_cpus = os.cpu_count() or 1
    # Size batches based on available RAM — cap so thread spike stays under 10% of RAM
    _ram_gb = _get_available_ram_gb()
    BATCH_MAX = _get_cross_batch_size(N)
    # Cap BATCH so all threads' dense arrays fit in 10% of RAM
    # Each thread: 3 arrays of (N, BATCH) × 8 bytes + nonzero overhead (~2x)
    _target_thread_ram_gb = _ram_gb * 0.10
    _n_target_threads = max(4, min(64, n_cpus // 4))
    _max_batch_for_ram = max(500, int(_target_thread_ram_gb * 1e9 / (_n_target_threads * N * 8 * 6)))
    BATCH_MAX = min(BATCH_MAX, _max_batch_for_ram)

    _ram_per_worker_gb = max(0.1, N * BATCH_MAX * 8 * 3 / 1e9)
    _ram_limited = max(4, int(_ram_gb * 0.10 / _ram_per_worker_gb))
    n_threads = min(_ram_limited, n_cpus, max(4, _ram_limited))

    # Size batches so we get at least n_threads batches (saturate all threads)
    # But don't make batches too small (< 500 pairs) — overhead dominates
    BATCH = min(BATCH_MAX, max(500, n_valid // n_threads))
    n_batches = (n_valid + BATCH - 1) // BATCH
    n_threads = min(n_threads, n_batches)

    if n_threads <= 1 or n_batches <= 1:
        # Single-threaded fast path — still flush COO→CSR periodically
        _st_csr = []
        _st_names = []
        _st_rows = []
        _st_data = []
        _st_col = 0
        _FLUSH_ST = max(5000, min(50000, int(_ram_gb * 50)))
        for b_start in range(0, n_valid, BATCH):
            b_end = min(b_start + BATCH, n_valid)
            blk_names, blk_rows, blk_data = _process_cross_block(
                left_mat, right_mat, valid_pairs, all_names, b_start, b_end, 0)
            for i in range(len(blk_names)):
                _st_names.append(blk_names[i])
                _st_rows.append(blk_rows[i])
                _st_data.append(blk_data[i])
                _st_col += 1
                col_idx += 1
            if _st_col >= _FLUSH_ST:
                _r = np.concatenate(_st_rows)
                _cp = [np.full(len(r), ci, dtype=np.int32) for ci, r in enumerate(_st_rows)]
                _c = np.concatenate(_cp)
                _d = np.concatenate(_st_data)
                _st_csr.append(sparse.coo_matrix((_d, (_r, _c)), shape=(N, _st_col)).tocsr())
                names.extend(_st_names)
                _st_names.clear(); _st_rows.clear(); _st_data.clear(); _st_col = 0
                del _r, _c, _d, _cp; gc.collect()
            if max_features and (feats_so_far + col_idx) >= max_features:
                break
        # Final flush
        if _st_rows:
            _r = np.concatenate(_st_rows)
            _cp = [np.full(len(r), ci, dtype=np.int32) for ci, r in enumerate(_st_rows)]
            _c = np.concatenate(_cp)
            _d = np.concatenate(_st_data)
            _st_csr.append(sparse.coo_matrix((_d, (_r, _c)), shape=(N, _st_col)).tocsr())
            names.extend(_st_names)
            del _r, _c, _d, _cp; gc.collect()
        if _st_csr:
            return names, _st_csr, None, None, col_idx
        return names, rows_list, cols_list, data_list, col_idx

    # ── Multi-threaded execution with streaming merge ──
    # Process batches in windows. Merge results WITHIN each window (not after all).
    # This prevents 1.38M features of per-column arrays from accumulating.
    batch_ranges = []
    for b_start in range(0, n_valid, BATCH):
        b_end = min(b_start + BATCH, n_valid)
        batch_ranges.append((b_start, b_end))

    log(f"    Parallel cross: {n_valid} pairs, {len(batch_ranges)} batches, {n_threads} threads")

    _csr_out = []
    _coo_names_local = []
    _coo_rows_local = []
    _coo_data_local = []
    _coo_col_local = 0
    _ram_gb_cpu = _get_available_ram_gb()
    _FLUSH_FEATS_CPU = max(5000, min(50000, int(_ram_gb_cpu * 50)))

    def _flush_cpu_coo():
        nonlocal _coo_col_local
        if not _coo_rows_local:
            return
        _r = np.concatenate(_coo_rows_local)
        _c_parts = [np.full(len(r), ci, dtype=np.int32) for ci, r in enumerate(_coo_rows_local)]
        _c = np.concatenate(_c_parts)
        _d = np.concatenate(_coo_data_local)
        _csr = sparse.coo_matrix((_d, (_r, _c)), shape=(N, _coo_col_local)).tocsr()
        _csr_out.append(_csr)
        names.extend(_coo_names_local)
        _coo_names_local.clear()
        _coo_rows_local.clear()
        _coo_data_local.clear()
        _coo_col_local = 0
        del _r, _c, _d, _csr, _c_parts
        gc.collect()

    WINDOW = n_threads  # one batch per thread per window
    _hit_limit = False
    for win_start in range(0, len(batch_ranges), WINDOW):
        win_end = min(win_start + WINDOW, len(batch_ranges))
        win_ranges = batch_ranges[win_start:win_end]
        # Execute window
        win_results = [None] * len(win_ranges)
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = {}
            for idx_offset, (b_start, b_end) in enumerate(win_ranges):
                f = executor.submit(_process_cross_block,
                                    left_mat, right_mat, valid_pairs, all_names,
                                    b_start, b_end, 0)
                futures[f] = idx_offset
            for f in as_completed(futures):
                try:
                    win_results[futures[f]] = f.result()
                except Exception as _thread_err:
                    log(f"    WARNING: thread {futures[f]} failed: {_thread_err}")
                    win_results[futures[f]] = ([], [], [])  # empty result
        # Merge this window's results immediately (frees per-column arrays)
        for blk_names, blk_rows, blk_data in win_results:
            if blk_names is None:
                blk_names, blk_rows, blk_data = [], [], []
            if blk_names is None:
                continue
            for i in range(len(blk_names)):
                _coo_names_local.append(blk_names[i])
                _coo_rows_local.append(blk_rows[i])
                _coo_data_local.append(blk_data[i])
                _coo_col_local += 1
                col_idx += 1
            if _coo_col_local >= _FLUSH_FEATS_CPU:
                _flush_cpu_coo()
                log(f"    CPU merge flush: {len(_csr_out)} CSR chunks, {col_idx:,}/{n_valid:,} features")
            if max_features and (feats_so_far + col_idx) >= max_features:
                _hit_limit = True
                break
        del win_results
        gc.collect()
        _malloc_trim()
        if _hit_limit:
            break

    _flush_cpu_coo()  # final flush

    if _csr_out:
        return names, _csr_out, None, None, col_idx
    return names, rows_list, cols_list, data_list, col_idx


# ============================================================
# DOY WINDOW GENERATION (±2 days)
# ============================================================

def create_doy_windows(df, window=2):
    """
    Create DOY ±window day flags. Each window covers 2*window+1 days.
    Returns (doy_names, doy_arrays).
    """
    if hasattr(df.index, 'dayofyear'):
        doy_vals = df.index.dayofyear.values
    elif 'day_of_year' in df.columns:
        doy_vals = pd.to_numeric(df['day_of_year'], errors='coerce').values.astype(int)
    else:
        log("  [WARN] No DOY source, skipping DOY windows")
        return [], []

    names = []
    arrays = []
    for d in range(1, 366):
        # Window wraps around year boundary — vectorized with np.isin
        targets = [(d + offset - 1) % 365 + 1 for offset in range(-window, window + 1)]
        mask = np.isin(doy_vals, targets).astype(np.float32)

        if mask.sum() > 0:
            names.append(f'dw_{d}')
            arrays.append(mask)

    return names, arrays


def create_month_windows(df):
    """
    Create month-of-year flags (12 months). Replaces DOY for TFs where
    daily granularity is noise (1w, 1d). Each flag fires ~480 times on 1d
    vs ~16 times for each DOY flag.
    Returns (month_names, month_arrays).
    """
    if hasattr(df.index, 'month'):
        month_vals = df.index.month.values
    else:
        log("  [WARN] No month source, skipping month windows")
        return [], []

    names = []
    arrays = []
    month_labels = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                    'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    for m in range(1, 13):
        mask = (month_vals == m).astype(np.float32)
        if mask.sum() > 0:
            names.append(f'moy_{month_labels[m-1]}')
            arrays.append(mask)

    return names, arrays


# Per-TF: use month windows instead of DOY windows for low-granularity TFs
# DOY crosses create 365 × contexts = ~1M features. Month crosses = 12 × contexts = ~30K.
# DOY is noise on weekly/daily. Month captures real seasonal patterns.
USE_MONTH_INSTEAD_OF_DOY = {'1w', '1d'}


# ============================================================
# REGIME-AWARE DOY (3x)
# ============================================================

def create_regime_doy(doy_names, doy_arrays, df):
    """
    Split DOY windows by regime: bull, bear, sideways.
    Returns additional (names, arrays) — triples the DOY features.
    """
    # Find regime column
    regime_col = None
    for col in ('regime', 'hmm_regime', 'regime_label'):
        if col in df.columns:
            regime_col = col
            break

    # If no regime column, try ema50_rising as proxy
    if regime_col is None:
        if 'ema50_rising' in df.columns:
            vals = pd.to_numeric(df['ema50_rising'], errors='coerce').values
            # NaN preserved — LightGBM treats missing as unknown split direction
            bull = (vals > 0).astype(np.float32)
            bear = (vals == 0).astype(np.float32)
            sideways = np.zeros(len(df), dtype=np.float32)  # No sideways without HMM
        else:
            return [], []
    else:
        regime_vals = df[regime_col].values
        bull = (regime_vals == 'bull').astype(np.float32) if regime_vals.dtype == object else \
               (regime_vals == 0).astype(np.float32)
        bear = (regime_vals == 'bear').astype(np.float32) if regime_vals.dtype == object else \
               (regime_vals == 1).astype(np.float32)
        sideways = (regime_vals == 'sideways').astype(np.float32) if regime_vals.dtype == object else \
                   (regime_vals == 2).astype(np.float32)

    names = []
    arrays = []
    for dn, da in zip(doy_names, doy_arrays):
        for tag, regime_mask in [('B', bull), ('R', bear), ('S', sideways)]:
            cross = da * regime_mask
            if cross.sum() > 0:
                names.append(f'{dn}_{tag}')
                arrays.append(cross)

    return names, arrays


# ============================================================
# MULTI-SIGNAL COMBINATIONS
# ============================================================

def create_multi_signal_combos(signals, prefix, max_pairs=100):
    """
    Create 2-way combinations of signals firing simultaneously.
    Returns (names, arrays, formulas) where formulas = {combo_name: [comp_a, comp_b]}.
    The formula dict persists the full component signal names so inference can
    reconstruct combo contexts from their parent binarized signals.
    """
    names_list = [s[0] for s in signals]
    arrays_list = [s[1] for s in signals]
    n = len(signals)

    combo_names = []
    combo_arrays = []
    combo_formulas = {}  # {combo_name: [full_name_i, full_name_j]}
    count = 0

    # Batched inner loop: for each i, multiply against all j>i at once
    for i in range(n):
        remaining = n - (i + 1)
        if remaining <= 0:
            break

        # Stack all j>i arrays into a matrix (N, remaining)
        right_mat = np.column_stack(arrays_list[i + 1:])  # (N, remaining)

        # Broadcast multiply: (N, 1) * (N, remaining) = (N, remaining)
        combos_batch = arrays_list[i][:, None] * right_mat  # vectorized

        # Compute sums along axis=0 to filter
        sums = combos_batch.sum(axis=0)  # (remaining,)

        # Extract valid combos
        valid_mask = sums >= MIN_CO_OCCURRENCE
        valid_indices = np.nonzero(valid_mask)[0]

        for idx in valid_indices:
            j = i + 1 + idx
            cname = f'{prefix}_{names_list[i][:15]}_{names_list[j][:15]}'
            combo_names.append(cname)
            combo_arrays.append(combos_batch[:, idx].copy())
            # Persist full component names (not truncated) for inference reconstruction
            combo_formulas[cname] = [names_list[i], names_list[j]]
            count += 1
            if count >= max_pairs:
                return combo_names, combo_arrays, combo_formulas

        del right_mat, combos_batch, sums
        if count >= max_pairs:
            break

    return combo_names, combo_arrays, combo_formulas


# ============================================================
# MAIN CROSS GENERATION
# ============================================================

def generate_all_crosses(df, tf='1d', gpu_id=0, save_sparse=False, output_dir=None, max_crosses=None, daemon_handles=None):
    """
    Generate ALL V2 cross features for a single asset's feature DataFrame.

    MEMORY-OPTIMIZED: Each cross type streams directly to sparse chunks.
    We never hold more than one batch of dense arrays at a time.
    Peak RAM stays under ~4GB regardless of total feature count.

    Args:
        daemon_handles: Pre-started GPU daemon handles from __main__ (V4 architecture).
                        If None, will attempt to start daemons here (fallback).

    Returns base DataFrame (crosses are in sparse .npz file, not in df).
    """
    global PARALLEL_CROSS_STEPS
    print(f"[XGEN-DEBUG] generate_all_crosses() ENTERED tf={tf} gpu_id={gpu_id} daemon_handles={'PASSED' if daemon_handles else 'None'}", flush=True)
    t0 = time.time()
    N = len(df)
    out_dir = output_dir or V2_DIR
    symbol_tag = getattr(df, '_v2_symbol', None) or df.attrs.get('symbol') or 'GLOBAL'
    run_id = get_crossgen_run_id()
    os.environ['SAVAGE22_RUN_ID'] = run_id
    os.environ['SAVAGE22_XGEN_SYMBOL'] = str(symbol_tag)
    os.environ['SAVAGE22_XGEN_TF'] = str(tf)

    # Pin to GPU — CUDA_VISIBLE_DEVICES remaps, so always use Device(0) when set
    print(f"[XGEN-DEBUG] GPU={GPU}, about to pin device", flush=True)
    if GPU:
        if os.environ.get('CUDA_VISIBLE_DEVICES'):
            cp.cuda.Device(0).use()
        else:
            cp.cuda.Device(gpu_id).use()
    print(f"[XGEN-DEBUG] GPU pin done", flush=True)

    log(f"V2 Cross Generator — {tf.upper()}")
    log(f"  Input: {N:,} rows × {len(df.columns):,} cols")
    log(f"  Run scope: {run_id} | symbol={symbol_tag}")

    # ── GPU daemons (V4 architecture: persistent daemons, zero scipy) ──
    # If daemon_handles passed from __main__, use them directly.
    # Otherwise, try to start here (fallback for non-CLI callers).
    _daemon_handles = daemon_handles
    print(f"[XGEN-DEBUG] daemon_handles from caller: {'PRESENT' if _daemon_handles else 'None'}", flush=True)

    if _daemon_handles is None:
        # Fallback: try to start daemons here (original path)
        available_gpus = _detect_available_gpus() if GPU else []
        n_gpus = len(available_gpus) if _MULTI_GPU_CROSS_GEN else (1 if GPU else 0)
        print(f"[XGEN-DEBUG] Fallback daemon start: GPU={GPU}, n_gpus={n_gpus}, MULTI_GPU={_MULTI_GPU_CROSS_GEN}, available_gpus={available_gpus}", flush=True)
        log(f"  [DEBUG] GPU={GPU}, n_gpus={n_gpus}, MULTI_GPU={_MULTI_GPU_CROSS_GEN}, available_gpus={available_gpus}")
        if n_gpus > 1 and GPU:
            try:
                print(f"[XGEN-DEBUG] Importing gpu_daemon for fallback prestage...", flush=True)
                from gpu_daemon import prestage_gpu_daemons
                log(f"  Pre-starting {n_gpus} GPU daemons (V4 zero-scipy architecture)...")
                _daemon_handles = prestage_gpu_daemons(
                    n_gpus=n_gpus,
                    available_gpu_ids=available_gpus,
                    vram_limit_pct=0.85
                )
                _active = sum(1 for h in _daemon_handles if h.status == 'idle')
                if _active < 2:
                    log(f"  ERROR: Only {_active}/{n_gpus} daemons active; persistent GPU path unavailable")
                    from gpu_daemon import shutdown_daemons
                    shutdown_daemons(_daemon_handles, timeout=10)
                    raise RuntimeError("Persistent GPU daemons failed to prestart")
                else:
                    log(f"  {_active}/{n_gpus} GPU daemons ready (CUDA init hidden behind binarization)")
            except Exception as _daemon_err:
                import traceback as _tb
                print(f"[XGEN-DEBUG] DAEMON PRESTAGE EXCEPTION: {_daemon_err}", flush=True)
                _tb.print_exc()
                log(f"  ERROR: GPU daemon prestage failed ({_daemon_err})")
                raise
    else:
        # Daemons passed from __main__ — validate they're alive
        _active = sum(1 for h in _daemon_handles if h.status == 'idle')
        log(f"  Using {_active}/{len(_daemon_handles)} pre-started GPU daemons from __main__")
        print(f"[XGEN-DEBUG] Pre-started daemons: {_active}/{len(_daemon_handles)} active", flush=True)
        if _active < 2:
            log(f"  ERROR: Only {_active} daemons alive; persistent GPU path unavailable")
            from gpu_daemon import shutdown_daemons
            shutdown_daemons(_daemon_handles, timeout=10)
            raise RuntimeError("Persistent GPU daemons unavailable at cross-gen start")

    # ── Step 1: Binarize all contexts (4-tier) ──
    log("  Step 1: Binarizing contexts (4-tier)...")
    ctx_names, ctx_arrays = binarize_contexts(df, four_tier=True)
    log(f"    {len(ctx_names)} context signals")

    # ── Step 2: Extract signal groups ──
    groups = extract_signal_groups(df, ctx_names, ctx_arrays)
    for g, sigs in groups.items():
        if sigs:
            log(f"    {g}: {len(sigs)} signals")

    # ── Step 3: DOY or Month windows (per-TF) ──
    if tf in USE_MONTH_INSTEAD_OF_DOY:
        log(f"  Step 2: Creating month-of-year windows (DOY replaced for {tf})...")
        doy_names, doy_arrays = create_month_windows(df)
        log(f"    {len(doy_names)} month windows (replaces 365 DOY windows)")
    else:
        log("  Step 2: Creating DOY ±2 windows...")
        doy_names, doy_arrays = create_doy_windows(df)
        log(f"    {len(doy_names)} DOY windows")

    # ── Pre-extract TA and astro arrays for targeted crossing ──
    # Crosses 6-12 use targeted right-sides (TA, TA+astro, TA+DOY) instead of ALL contexts.
    # Extract once here so they're available to all cross types.
    ta_n = [s[0] for s in groups['ta']] if groups['ta'] else []
    ta_a = [s[1] for s in groups['ta']] if groups['ta'] else []
    astro_n = [s[0] for s in groups['astro']] if groups['astro'] else []
    astro_a = [s[1] for s in groups['astro']] if groups['astro'] else []

    # ── Accumulate sparse CSR chunks + column names across all cross types ──
    # PERF: Convert each cross type to CSR immediately instead of accumulating
    # one massive COO. hstack of small CSR matrices is much faster than
    # converting a single giant COO→CSR (scipy COO→CSR is single-threaded).
    _csr_chunks = []    # list of CSR matrices, one per cross type
    all_cross_names = []    # flat list of feature names
    all_combo_formulas = {}  # {combo_name: [comp_a, comp_b]} — persisted for inference
    col_offset = 0      # running column offset (for gpu_batch_cross)

    _total_collected = 0

    # ── Per-cross-type checkpoint/resume ──
    # Each completed cross type is saved as an individual NPZ + JSON checkpoint.
    # On OOM at cross 12, types 1-11 are recoverable from checkpoint files.
    import json as _json_mod
    _ckpt_dir = crossgen_scratch_dir(
        'checkpoints', symbol=symbol_tag, tf=tf, prefix='checkpoint', run_id=run_id
    )
    _ckpt_pattern = os.path.join(_ckpt_dir, '*__checkpoint.npz')
    _completed_prefixes = set()
    _checkpoint_files = sorted(glob.glob(_ckpt_pattern))
    _saved_checkpoint_files = []
    if _checkpoint_files:
        log(f"  Found {len(_checkpoint_files)} checkpoint(s), resuming...")
        for _cf in _checkpoint_files:
            _cf_base = os.path.basename(_cf)
            if not _cf_base.endswith('__checkpoint.npz'):
                continue
            _cf_prefix = _cf_base[:-len('__checkpoint.npz')].split('__')[-1]
            _completed_prefixes.add(_cf_prefix)
            # Load the CSR chunk (backward-compat: handles both scipy and indices-only formats)
            from atomic_io import load_npz_auto
            _resume_csr = load_npz_auto(_cf)
            _csr_chunks.append(_resume_csr)
            _saved_checkpoint_files.append(_cf)
            # Load matching names
            _cf_names_path = os.path.join(
                _ckpt_dir,
                f"{get_crossgen_namespace(symbol=symbol_tag, tf=tf, prefix=_cf_prefix, run_id=run_id)}__names.json",
            )
            if os.path.exists(_cf_names_path):
                with open(_cf_names_path, 'r') as _f:
                    _resume_names = _json_mod.load(_f)
                all_cross_names.extend(_resume_names)
                _total_collected += len(_resume_names)
                col_offset += _resume_csr.shape[1]
                log(f"    Restored checkpoint: {_cf_prefix} ({_resume_csr.shape[1]:,} features, {len(_resume_names):,} names)")
            else:
                log(f"    WARNING: checkpoint {_cf_prefix} missing names file, skipping")
                _csr_chunks.pop()
                _completed_prefixes.discard(_cf_prefix)
            del _resume_csr
        gc.collect()
        _malloc_trim()
        log(f"  Resumed {len(_completed_prefixes)} cross type(s): {sorted(_completed_prefixes)}")
        log(f"  Accumulated: {_total_collected:,} features so far")

    _pending_disk_groups = []  # list of (npz_file_list, tmp_dir) from disk-backed cross types

    # ── Opt #7H: Intra-step time flush — flush accumulated chunks every N seconds ──
    _FLUSH_INTERVAL_SEC = int(os.environ.get('V2_FLUSH_INTERVAL_SEC', '1200'))  # default 20 min
    _last_flush_time = time.time()

    def _streaming_csc_splice(sources, n_rows):
        """Stream-assemble CSR from mix of in-memory CSR matrices and on-disk NPZ files.
        Loads one source at a time to bound peak RAM."""
        all_data_parts = []
        all_indices_parts = []
        indptr_parts = []
        cumulative_nnz = np.int64(0)
        total_cols = 0
        for src in sources:
            if isinstance(src, str):
                csc = sparse.load_npz(src).tocsc()
            else:
                csc = src.tocsc()
            all_data_parts.append(csc.data)
            all_indices_parts.append(csc.indices)
            # int64 cast BEFORE addition to prevent overflow at >2B NNZ
            indptr_parts.append(csc.indptr[:-1].astype(np.int64) + cumulative_nnz)
            cumulative_nnz += np.int64(csc.nnz)
            total_cols += csc.shape[1]
            del csc
            gc.collect()
        indptr_parts.append(np.array([cumulative_nnz], dtype=np.int64))
        _all_data = np.concatenate(all_data_parts)
        _all_indices = np.concatenate(all_indices_parts)
        _new_indptr = np.concatenate(indptr_parts)
        del all_data_parts, all_indices_parts, indptr_parts
        result = sparse.csc_matrix((_all_data, _all_indices, _new_indptr),
                                   shape=(n_rows, total_cols)).tocsr()
        del _all_data, _all_indices, _new_indptr
        gc.collect()
        return result

    def _save_checkpoint(cross_prefix):
        """Save checkpoint from in-memory CSR chunks + disk-backed sub-checkpoints.
        Streams from disk to avoid loading all sub-checkpoints at once."""
        from atomic_io import atomic_save_json, atomic_save_npz
        _has_memory = bool(_csr_chunks)
        _has_disk = bool(_pending_disk_groups)
        if not _has_memory and not _has_disk:
            return

        # Build source list: disk NPZ paths + in-memory CSR matrices
        _sources = []
        _tmp_dirs_to_clean = []
        for _npz_files, _tmp_dirs in _pending_disk_groups:
            _sources.extend(_npz_files)
            if isinstance(_tmp_dirs, list):
                _tmp_dirs_to_clean.extend(_tmp_dirs)
            else:
                _tmp_dirs_to_clean.append(_tmp_dirs)
        _sources.extend(_csr_chunks)

        try:
            if len(_sources) == 1 and not isinstance(_sources[0], str):
                _merged = _sources[0]
            else:
                log(f"    Streaming CSC splice: {len(_sources)} sources ({sum(1 for s in _sources if isinstance(s, str))} disk, {len(_csr_chunks)} memory)")
                _merged = _streaming_csc_splice(_sources, N)

            _n_cols_this = _merged.shape[1]
            _these_names = validate_sparse_names_contract(
                _merged,
                all_cross_names[-_n_cols_this:],
                expected_prefix=cross_prefix,
            )
            _ckpt_stem = get_crossgen_namespace(
                symbol=symbol_tag, tf=tf, prefix=cross_prefix, run_id=run_id
            )
            _ckpt_npz = os.path.join(_ckpt_dir, f'{_ckpt_stem}__checkpoint.npz')
            _ckpt_names = os.path.join(_ckpt_dir, f'{_ckpt_stem}__names.json')
            atomic_save_npz(_merged, _ckpt_npz)
            atomic_save_json(_these_names, _ckpt_names)
            if not os.path.exists(_ckpt_npz):
                raise RuntimeError(f"Checkpoint save missing after write: {_ckpt_npz}")
            if not os.path.exists(_ckpt_names):
                raise RuntimeError(f"Checkpoint names missing after write: {_ckpt_names}")
            _saved_checkpoint_files.append(_ckpt_npz)
            log(f"    Checkpoint saved: {cross_prefix} ({_n_cols_this:,} features)")
            del _merged
        finally:
            # ALWAYS cleanup: free RAM + remove temp dirs (even on OOM)
            _csr_chunks.clear()
            _pending_disk_groups.clear()
            for _td in _tmp_dirs_to_clean:
                cleanup_crossgen_scratch_dir(_td, run_id=run_id)
            gc.collect()
            _malloc_trim()
            log(f"    RAM freed (all chunks offloaded to disk)")

    def _maybe_time_flush(cross_prefix):
        """Opt #7H: Flush accumulated chunks to disk if flush interval elapsed.
        Prevents data loss during long-running cross types (15m can take hours)."""
        nonlocal _last_flush_time
        if not _csr_chunks and not _pending_disk_groups:
            return
        elapsed = time.time() - _last_flush_time
        if elapsed >= _FLUSH_INTERVAL_SEC:
            log(f"    Time flush ({elapsed:.0f}s elapsed): saving {cross_prefix} checkpoint...")
            _save_checkpoint(cross_prefix)
            _last_flush_time = time.time()

    def _cleanup_checkpoints():
        """Delete all checkpoint files after successful final save."""
        _removed = 1 if cleanup_crossgen_scratch_dir(_ckpt_dir, run_id=run_id) else 0
        _saved_checkpoint_files.clear()
        if _removed:
            log("  Cleaned up checkpoint scratch")

    def _collect_cross(label, names, rows_list, cols_list, data_list, n_new_cols):
        """Convert cross type results to CSR. Handles CSR-flushed, disk-backed, and legacy COO paths."""
        nonlocal col_offset, _total_collected
        count = len(names)
        if count > 0:
            all_cross_names.extend(names)
            # BUG-M1 FIX: capture col_offset BEFORE incrementing so COO→local
            # subtraction uses the correct base (especially in parallel mode
            # where each step returns 0-based columns).
            _col_offset_base = col_offset
            # Disk-backed path: sub-checkpoints on disk from gpu_batch_cross
            if cols_list == 'disk' and isinstance(rows_list, list):
                # data_list = list of tmp_dirs or single tmp_dir string
                _tmp_dirs = data_list if isinstance(data_list, list) else [data_list]
                _pending_disk_groups.append((rows_list, _tmp_dirs))  # (npz_files, tmp_dirs)
                col_offset += n_new_cols
                _total_collected += count
                log(f"    {label} crosses: {count:,} (total: {_total_collected:,}) [disk-backed]")
                return count
            # Check if rows_list contains pre-flushed CSR chunks
            if rows_list is not None and isinstance(rows_list, list) and len(rows_list) > 0 and hasattr(rows_list[0], "indptr"):
                _csr_chunks.extend(rows_list)
                col_offset += n_new_cols
                _total_collected += count
                gc.collect()
                _malloc_trim()
            elif rows_list is not None and cols_list is not None and isinstance(rows_list, list) and len(rows_list) > 0:
                _r = np.concatenate(rows_list)
                _c = np.concatenate(cols_list)
                _d = np.concatenate(data_list)
                _c_local = _c - _col_offset_base
                if cp is not None and cusp is not None:
                    try:
                        _d_gpu = cp.asarray(_d)
                        _r_gpu = cp.asarray(_r)
                        _c_local_gpu = cp.asarray(_c_local)
                        chunk = cusp.coo_matrix((_d_gpu, (_r_gpu, _c_local_gpu)), shape=(N, n_new_cols)).tocsr()
                        chunk = chunk.get()  # Transfer back to scipy CSR on CPU
                        del _d_gpu, _r_gpu, _c_local_gpu
                        cp.get_default_memory_pool().free_all_blocks()
                    except Exception:
                        chunk = sparse.coo_matrix((_d, (_r, _c_local)), shape=(N, n_new_cols)).tocsr()
                else:
                    chunk = sparse.coo_matrix((_d, (_r, _c_local)), shape=(N, n_new_cols)).tocsr()
                _csr_chunks.append(chunk)
                del _r, _c, _c_local, _d, chunk
                gc.collect()
                _malloc_trim()
                col_offset += n_new_cols
                _total_collected += count
            else:
                col_offset += n_new_cols
                _total_collected += count
            log(f"    {label} crosses: {count:,} (total: {_total_collected:,})")
        return count

    def _remaining():
        """How many more crosses can we generate before hitting max_crosses."""
        if max_crosses is None:
            return None
        return max(0, max_crosses - _total_collected)

    def _at_limit():
        """True if we've hit the max_crosses cap."""
        return max_crosses is not None and _total_collected >= max_crosses

    # ── Thread count: keep full cores for matmul phase ──
    # MKL thread exhaustion is handled by threadpoolctl scoping in _mkl_dot(),
    # not by global throttling. Numba prange kernels use all cores.
    log(f"  Matmul phase: NUMBA_NUM_THREADS={os.environ.get('NUMBA_NUM_THREADS', 'unset')}, "
        f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')} (full cores, scoped via threadpoolctl)")

    # ── Define all 12 cross steps as descriptors ──
    # Each step: (step_num, prefix, label, left_n, left_a, right_n, right_a, pre_fn, est_ram_gb)
    # pre_fn: optional callable returning (left_n, left_a, combo_formulas) for steps needing combo creation
    # est_ram_gb: rough RAM estimate for memory-aware scheduling (based on left×right sizes)

    def _step_ax2_prep():
        ax2_n, ax2_a, ax2_f = create_multi_signal_combos(groups['astro'], 'a2', max_pairs=50)
        return ax2_n, ax2_a, ax2_f

    def _step_ta2_prep():
        ta2_n, ta2_a, ta2_f = create_multi_signal_combos(groups['ta'][:60], 'ta2', max_pairs=30)
        combined_n = list(doy_names) + list(astro_n)
        combined_a = list(doy_arrays) + list(astro_a)
        return ta2_n, ta2_a, ta2_f, combined_n, combined_a

    def _est_ram(n_left, n_right):
        """Rough RAM estimate in GB for a cross step."""
        return max(0.1, n_left * n_right * N * 4 / 1e9 * 0.15)

    _cross_steps = []  # will be populated below

    # Step 1: DOY × ALL contexts
    if 'dx' not in _completed_prefixes:
        _cross_steps.append({
            'num': 1, 'prefix': 'dx', 'label': 'dx_',
            'desc': 'DOY windows × ALL contexts',
            'left_n': doy_names, 'left_a': doy_arrays,
            'right_n': ctx_names, 'right_a': ctx_arrays,
            'est_ram': _est_ram(len(doy_names), len(ctx_names)),
        })
    else:
        log("  Cross 1 (dx): SKIPPED (checkpoint)")

    # Step 2: Astro × TA
    if 'ax' not in _completed_prefixes and astro_n and ta_n:
        _cross_steps.append({
            'num': 2, 'prefix': 'ax', 'label': 'ax_',
            'desc': 'Astro × TA',
            'left_n': astro_n, 'left_a': astro_a,
            'right_n': ta_n, 'right_a': ta_a,
            'est_ram': _est_ram(len(astro_n), len(ta_n)),
        })
    elif 'ax' in _completed_prefixes:
        log("  Cross 2 (ax): SKIPPED (checkpoint)")

    # Step 3: Multi-astro combos × TA (needs prep)
    if 'ax2' not in _completed_prefixes and len(groups['astro']) >= 2 and ta_n:
        _cross_steps.append({
            'num': 3, 'prefix': 'ax2', 'label': 'ax2_',
            'desc': 'Multi-astro combos × TA',
            'prep_fn': _step_ax2_prep,
            'right_n': ta_n, 'right_a': ta_a,
            'est_ram': _est_ram(min(50, len(groups['astro'])**2), len(ta_n)),
        })
    elif 'ax2' in _completed_prefixes:
        log("  Cross 3 (ax2): SKIPPED (checkpoint)")

    # Step 4: Multi-TA combos × DOY + astro (needs prep)
    if 'ta2' not in _completed_prefixes and len(groups['ta']) >= 2:
        _cross_steps.append({
            'num': 4, 'prefix': 'ta2', 'label': 'ta2_',
            'desc': 'Multi-TA combos × DOY + astro',
            'prep_fn': _step_ta2_prep,
            'est_ram': _est_ram(min(30, len(groups['ta'][:60])**2), len(doy_names) + len(astro_n)),
        })
    elif 'ta2' in _completed_prefixes:
        log("  Cross 4 (ta2): SKIPPED (checkpoint)")

    # Step 5: Esoteric × TA
    if 'ex2' not in _completed_prefixes and groups['esoteric'] and ta_n:
        eso_n = [s[0] for s in groups['esoteric']]
        eso_a = [s[1] for s in groups['esoteric']]
        _cross_steps.append({
            'num': 5, 'prefix': 'ex2', 'label': 'ex2_',
            'desc': 'Esoteric × TA',
            'left_n': eso_n, 'left_a': eso_a,
            'right_n': ta_n, 'right_a': ta_a,
            'est_ram': _est_ram(len(eso_n), len(ta_n)),
        })
    elif 'ex2' in _completed_prefixes:
        log("  Cross 5 (ex2): SKIPPED (checkpoint)")

    # Step 6: Space weather × TA
    if 'sw' not in _completed_prefixes and groups['space_weather'] and ta_n:
        sw_n = [s[0] for s in groups['space_weather']]
        sw_a = [s[1] for s in groups['space_weather']]
        _cross_steps.append({
            'num': 6, 'prefix': 'sw', 'label': 'sw_',
            'desc': 'Space weather × TA',
            'left_n': sw_n, 'left_a': sw_a,
            'right_n': ta_n, 'right_a': ta_a,
            'est_ram': _est_ram(len(sw_n), len(ta_n)),
        })
    elif 'sw' in _completed_prefixes:
        log("  Cross 6 (sw): SKIPPED (checkpoint)")

    # Step 7: Session × TA+astro
    if 'hod' not in _completed_prefixes and groups['session'] and (ta_n or astro_n):
        hod_n = [s[0] for s in groups['session']]
        hod_a = [s[1] for s in groups['session']]
        hod_right_n = ta_n + astro_n
        hod_right_a = ta_a + astro_a
        _cross_steps.append({
            'num': 7, 'prefix': 'hod', 'label': 'hod_',
            'desc': 'Session × TA+astro',
            'left_n': hod_n, 'left_a': hod_a,
            'right_n': hod_right_n, 'right_a': hod_right_a,
            'est_ram': _est_ram(len(hod_n), len(hod_right_n)),
        })
    elif 'hod' in _completed_prefixes:
        log("  Cross 7 (hod): SKIPPED (checkpoint)")

    # Step 8: Macro × TA
    if 'mx' not in _completed_prefixes and groups['macro'] and ta_n:
        mx_n = [s[0] for s in groups['macro']]
        mx_a = [s[1] for s in groups['macro']]
        _cross_steps.append({
            'num': 8, 'prefix': 'mx', 'label': 'mx_',
            'desc': 'Macro × TA',
            'left_n': mx_n, 'left_a': mx_a,
            'right_n': ta_n, 'right_a': ta_a,
            'est_ram': _est_ram(len(mx_n), len(ta_n)),
        })
    elif 'mx' in _completed_prefixes:
        log("  Cross 8 (mx): SKIPPED (checkpoint)")

    # Step 9: Volatility × TA+DOY
    if 'vx' not in _completed_prefixes and groups['volatility'] and (ta_n or doy_names):
        vx_n = [s[0] for s in groups['volatility']]
        vx_a = [s[1] for s in groups['volatility']]
        vx_right_n = ta_n + list(doy_names)
        vx_right_a = ta_a + list(doy_arrays)
        _cross_steps.append({
            'num': 9, 'prefix': 'vx', 'label': 'vx_',
            'desc': 'Volatility × TA+DOY',
            'left_n': vx_n, 'left_a': vx_a,
            'right_n': vx_right_n, 'right_a': vx_right_a,
            'est_ram': _est_ram(len(vx_n), len(vx_right_n)),
        })
    elif 'vx' in _completed_prefixes:
        log("  Cross 9 (vx): SKIPPED (checkpoint)")

    # Step 10: Aspects × TA
    if 'asp' not in _completed_prefixes and groups['aspect'] and ta_n:
        asp_n = [s[0] for s in groups['aspect']]
        asp_a = [s[1] for s in groups['aspect']]
        _cross_steps.append({
            'num': 10, 'prefix': 'asp', 'label': 'asp_',
            'desc': 'Aspects × TA',
            'left_n': asp_n, 'left_a': asp_a,
            'right_n': ta_n, 'right_a': ta_a,
            'est_ram': _est_ram(len(asp_n), len(ta_n)),
        })
    elif 'asp' in _completed_prefixes:
        log("  Cross 10 (asp): SKIPPED (checkpoint)")

    # Step 11: Price numerology × TA
    if 'pn' not in _completed_prefixes and groups['price_num'] and ta_n:
        pn_n = [s[0] for s in groups['price_num']]
        pn_a = [s[1] for s in groups['price_num']]
        _cross_steps.append({
            'num': 11, 'prefix': 'pn', 'label': 'pn_',
            'desc': 'Price numerology × TA',
            'left_n': pn_n, 'left_a': pn_a,
            'right_n': ta_n, 'right_a': ta_a,
            'est_ram': _est_ram(len(pn_n), len(ta_n)),
        })
    elif 'pn' in _completed_prefixes:
        log("  Cross 11 (pn): SKIPPED (checkpoint)")

    # Step 12: Moon × TA
    if 'mn' not in _completed_prefixes and groups['moon'] and ta_n:
        mn_n = [s[0] for s in groups['moon']]
        mn_a = [s[1] for s in groups['moon']]
        _cross_steps.append({
            'num': 12, 'prefix': 'mn', 'label': 'mn_',
            'desc': 'Moon × TA',
            'left_n': mn_n, 'left_a': mn_a,
            'right_n': ta_n, 'right_a': ta_a,
            'est_ram': _est_ram(len(mn_n), len(ta_n)),
        })
    elif 'mn' in _completed_prefixes:
        log("  Cross 12 (mn): SKIPPED (checkpoint)")

    # Cross 13: REMOVED — Regime-aware DOY was redundant with dx_ (Cross 1)

    def _execute_one_step(step, daemon_handles=None):
        """Execute a single cross step. Returns (prefix, label, names, r, c, d, nc, combo_formulas)."""
        print(f"[XGEN-DEBUG] _execute_one_step: Cross {step['num']} ({step['prefix']}), daemon_handles={'PRESENT' if daemon_handles else 'None'}", flush=True)
        _log_memory(f"Cross {step['num']} ({step['prefix']}) START")
        t_step = time.time()
        log(f"  Cross {step['num']}: {step['desc']}...")

        combo_formulas = {}
        left_n = step.get('left_n')
        left_a = step.get('left_a')
        right_n = step.get('right_n')
        right_a = step.get('right_a')

        # Run prep function if present (creates combo signals)
        if 'prep_fn' in step:
            result = step['prep_fn']()
            if step['prefix'] == 'ta2':
                left_n, left_a, combo_formulas, right_n, right_a = result
            else:
                left_n, left_a, combo_formulas = result

        if not left_n or not right_n:
            _log_memory(f"Cross {step['num']} ({step['prefix']}) END (empty)")
            return step['prefix'], step['label'], [], [], [], [], 0, combo_formulas

        # Each parallel step uses col_offset=0 — offsets are reassigned during collection
        names, r, c, d, nc = gpu_batch_cross(
            left_n, left_a, right_n, right_a,
            step['prefix'], gpu_id=gpu_id, col_offset=0,
            max_features=max_crosses,
            daemon_handles=daemon_handles
        )

        elapsed_step = time.time() - t_step
        _log_memory(f"Cross {step['num']} ({step['prefix']}) END ({elapsed_step:.1f}s)")
        return step['prefix'], step['label'], names, r, c, d, nc, combo_formulas

    # ── Execute cross steps: parallel or sequential ──
    # Parallel mode uses a GPU lock so only 1 step touches GPU at a time,
    # but CPU-bound work (COO→CSR assembly, disk flush) overlaps between steps.
    # Steps split into "small" (can overlap) and "large" (run alone).
    _LARGE_STEP_RAM_GB = float(os.environ.get('V2_LARGE_STEP_RAM_GB', '2.0'))
    _LARGE_STEP_PAIRS = int(os.environ.get('V2_LARGE_STEP_PAIRS', '50000'))

    def _classify_steps(steps):
        """Split steps into small (parallelizable) and large (sequential)."""
        small, large = [], []
        for s in steps:
            n_left = len(s.get('left_n') or [])
            n_right = len(s.get('right_n') or [])
            n_pairs = n_left * n_right
            if s['est_ram'] > _LARGE_STEP_RAM_GB or n_pairs > _LARGE_STEP_PAIRS:
                large.append(s)
            else:
                small.append(s)
        return small, large

    if PARALLEL_CROSS_STEPS and len(_cross_steps) > 1:
        import threading
        from concurrent.futures import ThreadPoolExecutor as _TPE
        import concurrent.futures as _cf

        _small_steps, _large_steps = _classify_steps(_cross_steps)
        log(f"  PARALLEL MODE: {len(_cross_steps)} steps "
            f"({len(_small_steps)} small parallel, {len(_large_steps)} large sequential), "
            f"RAM ceiling={_RAM_CEILING_PCT}%")
        for s in _small_steps:
            log(f"    small: Cross {s['num']} ({s['prefix']}, ~{s['est_ram']:.1f}GB)")
        for s in _large_steps:
            log(f"    large: Cross {s['num']} ({s['prefix']}, ~{s['est_ram']:.1f}GB)")

        # GPU lock: only 1 thread can use GPU at a time.
        # gpu_batch_cross does chunk loops with GPU compute + CPU assembly.
        # Between chunks one thread does CPU work while another can use GPU.
        _gpu_lock = threading.Lock()

        def _execute_one_step_locked(step):
            """Execute a cross step with GPU lock to prevent VRAM contention."""
            with _gpu_lock:
                return _execute_one_step(step, daemon_handles=_daemon_handles)

        def _parallel_execute(steps_to_run):
            """Run steps in ThreadPoolExecutor with memory-aware backpressure."""
            results = []
            if not steps_to_run:
                return results
            steps_to_run.sort(key=lambda s: s['est_ram'])

            # GPU present: 2 workers (1 on GPU, 1 doing CPU assembly)
            # CPU-only: up to 4 workers bounded by available RAM
            if cp is not None:
                _max_w = 2
            else:
                _avail_gb = _get_available_ram_gb()
                _max_w = max(1, min(4, int(_avail_gb / max(0.1, _LARGE_STEP_RAM_GB))))
            log(f"    ThreadPool: {_max_w} workers (GPU={'yes' if cp is not None else 'no'})")

            with _TPE(max_workers=_max_w) as pool:
                futures = {}
                pending = list(steps_to_run)

                while pending or futures:
                    while pending and _get_mem_percent() < _RAM_CEILING_PCT and len(futures) < _max_w:
                        step = pending.pop(0)
                        log(f"    Submitting Cross {step['num']} ({step['prefix']}, "
                            f"~{step['est_ram']:.1f}GB est)")
                        fut = pool.submit(_execute_one_step_locked, step)
                        futures[fut] = step

                    if not futures:
                        break

                    done_set, _ = _cf.wait(futures.keys(), return_when=_cf.FIRST_COMPLETED)
                    for fut in done_set:
                        step = futures.pop(fut)
                        try:
                            result = fut.result()
                            results.append(result)
                            log(f"    Completed Cross {step['num']} ({step['prefix']})")
                        except Exception as e:
                            log(f"    FAILED Cross {step['num']} ({step['prefix']}): {e}")
                            import traceback
                            traceback.print_exc()
                            raise RuntimeError(
                                f"Cross step {step['num']} ({step['prefix']}) failed: {e}"
                            ) from e
            return results

        # Phase 1: Run small steps in parallel (GPU-locked, overlap CPU work)
        _all_results = []
        if _small_steps:
            log(f"  Phase 1: {len(_small_steps)} small steps in parallel")
            _all_results.extend(_parallel_execute(_small_steps))

        # Phase 2: Run large steps sequentially (full resources, no contention)
        if _large_steps:
            log(f"  Phase 2: {len(_large_steps)} large steps sequential")
            for step in _large_steps:
                if _at_limit():
                    log(f"  Cross {step['num']} ({step['prefix']}): SKIPPED (max_crosses reached)")
                    continue
                result = _execute_one_step(step, daemon_handles=_daemon_handles)
                _all_results.append(result)
                log(f"    Completed Cross {step['num']} ({step['prefix']})")

        # Collect results in canonical step order
        _step_order = ['dx', 'ax', 'ax2', 'ta2', 'ex2', 'sw', 'hod', 'mx', 'vx', 'asp', 'pn', 'mn']
        _result_map = {r[0]: r for r in _all_results}
        for prefix in _step_order:
            if prefix not in _result_map:
                continue
            _pfx, _lbl, _names, _r, _c, _d, _nc, _combo_f = _result_map[prefix]
            if _combo_f:
                all_combo_formulas.update(_combo_f)
            if _names:
                _collect_cross(_lbl, _names, _r, _c, _d, _nc)
                _save_checkpoint(_pfx)
            del _names, _r, _c, _d
            gc.collect()
            _malloc_trim()

    else:
        # Sequential mode (default)
        if _cross_steps:
            log(f"  SEQUENTIAL MODE: {len(_cross_steps)} steps")
        for step in _cross_steps:
            if _at_limit():
                log(f"  Cross {step['num']} ({step['prefix']}): SKIPPED (max_crosses reached)")
                continue
            _pfx, _lbl, _names, _r, _c, _d, _nc, _combo_f = _execute_one_step(step, daemon_handles=_daemon_handles)
            if _combo_f:
                all_combo_formulas.update(_combo_f)
            if _names:
                _collect_cross(_lbl, _names, _r, _c, _d, _nc)
                _save_checkpoint(_pfx)
            del _names, _r, _c, _d
            gc.collect()
            _malloc_trim()

    # ── Shutdown GPU daemons (V4: must happen BEFORE final assembly) ──
    # Only shutdown daemons we created locally (not ones passed from __main__)
    _owns_daemons = (daemon_handles is None and _daemon_handles is not None)
    if _owns_daemons:
        try:
            from gpu_daemon import shutdown_daemons
            log("  Shutting down GPU daemons (locally-created)...")
            shutdown_daemons(_daemon_handles, timeout=60)
            log("  GPU daemons shut down — full RAM available for assembly")
        except Exception as _sd_err:
            log(f"  WARNING: Daemon shutdown error: {_sd_err}")
        _daemon_handles = None
        gc.collect()
        _malloc_trim()
    elif _daemon_handles is not None:
        log("  GPU daemons owned by __main__ — NOT shutting down (reuse for next TF)")

    # ── Save inference artifacts (for live cross computation) ──
    if len(all_cross_names) > 0:
        try:
            from inference_crosses import save_inference_artifacts
            # Combine all context names: binarized + DOY windows + regime DOY
            # The cross generator uses ctx_names (binarized) + doy_names as left/right
            # for different cross types. Merge them all for inference.
            all_ctx_names = list(ctx_names) + list(doy_names)
            all_ctx_arrays = list(ctx_arrays) + list(doy_arrays)
            log("  Saving inference artifacts...")
            save_inference_artifacts(
                all_ctx_names, all_ctx_arrays, all_cross_names, df, tf,
                output_dir=out_dir,
                combo_formulas=all_combo_formulas if all_combo_formulas else None,
            )
        except Exception as e:
            log(f"  WARNING: Failed to save inference artifacts: {e}")
            import traceback
            traceback.print_exc()

    # ── Free context arrays — no longer needed ──
    del ctx_names, ctx_arrays, doy_names, doy_arrays, groups, ta_n, ta_a, astro_n, astro_a
    gc.collect()
    _malloc_trim()

    # ── SAVE CROSSES AS SPARSE (memory efficient) ──
    total_crosses = len(all_cross_names)
    log(f"\n  TOTAL NEW FEATURES: {total_crosses:,}")

    t_assign = time.time()

    # Final assembly: stream from checkpoint files (one at a time) to bound peak RAM
    _ckpt_files = list(dict.fromkeys(_saved_checkpoint_files)) or sorted(
        glob.glob(os.path.join(_ckpt_dir, '*__checkpoint.npz'))
    )

    if total_crosses > 0 and (_ckpt_files or _csr_chunks):
        cross_names = all_cross_names

        # Build source list: checkpoint file paths + any remaining in-memory chunks
        _sources = list(_ckpt_files) + list(_csr_chunks)
        log(f"  Assembly: {len(_ckpt_files)} disk + {len(_csr_chunks)} memory = {len(_sources)} sources")

        # Decide merge strategy: memmap (1h/15m) vs in-memory (1w/1d/4h)
        from memmap_merge import should_use_memmap, memmap_streaming_merge, cleanup_memmap_files
        _use_memmap = should_use_memmap(tf)
        _memmap_tmp_files = []

        if _use_memmap:
            log(f"  Using MEMMAP merge (tf={tf}) — peak RAM bounded to ~5-15GB")
            sparse_mat, _memmap_tmp_files = memmap_streaming_merge(_sources, N, tmp_dir=out_dir)
        else:
            log(f"  Using in-memory merge (tf={tf})")
            sparse_mat = _streaming_csc_splice(_sources, N)

        _csr_chunks.clear()
        gc.collect()
        _malloc_trim()

        # Force int64 to prevent silent overflow at >2B NNZ
        if sparse_mat.nnz > 2**30:
            sparse_mat.indices = sparse_mat.indices.astype(np.int64)
            sparse_mat.indptr = sparse_mat.indptr.astype(np.int64)
            log(f"  Upgraded to int64 indices (NNZ={sparse_mat.nnz:,})")


        log(f"  Sparse matrix: {sparse_mat.shape}, {sparse_mat.nnz:,} non-zeros, "
            f"density={sparse_mat.nnz / (sparse_mat.shape[0] * sparse_mat.shape[1]) * 100:.3f}%")

        # ── CORRELATION CLUSTERING (optional, gated by env var) ──
        # Clusters highly correlated TA × TA crosses to reduce redundancy.
        # NEVER touches esoteric features (protected prefixes).
        _USE_CORRELATION_CLUSTERING = os.environ.get('USE_CORRELATION_CLUSTERING', '0') == '1'
        if _USE_CORRELATION_CLUSTERING and total_crosses > 1000:
            log(f"\n  CORRELATION CLUSTERING: Enabled (reducing TA cross redundancy)")
            try:
                sparse_mat, cross_names = _apply_correlation_clustering(
                    sparse_mat, all_cross_names, tf, threshold=0.95
                )
                log(f"  Clustering complete: {sparse_mat.shape[1]:,} features retained "
                    f"({len(all_cross_names) - sparse_mat.shape[1]:,} clustered)")
            except Exception as _cluster_err:
                log(f"  WARNING: Correlation clustering failed (non-fatal): {_cluster_err}")
                log(f"  Proceeding with full feature set")
                import traceback
                traceback.print_exc()
                cross_names = all_cross_names
        else:
            cross_names = all_cross_names

        # Save sparse matrix + column names
        symbol_file_tag = getattr(df, '_v2_symbol', None) or ''
        if symbol_file_tag:
            npz_path = os.path.join(out_dir, f'v2_crosses_{symbol_file_tag}_{tf}.npz')
            names_path = os.path.join(out_dir, f'v2_cross_names_{symbol_file_tag}_{tf}.json')
        else:
            npz_path = os.path.join(out_dir, f'v2_crosses_{tf}.npz')
            names_path = os.path.join(out_dir, f'v2_cross_names_{tf}.json')

        from atomic_io import atomic_save_npz, atomic_save_json, save_npz_indices_only, _NPZ_INDICES_ONLY, save_csr_npy
        # Ensure cross names are native Python strings (not np.str_) for JSON
        # Dedup names from truncation collisions — keep first occurrence, drop duplicates
        # CRITICAL: dedup BEFORE saving NPZ so matrix cols match name count
        seen = set()
        dedup_indices = []
        for i, n in enumerate(cross_names):
            s = str(n)
            if s in seen:
                continue
            seen.add(s)
            dedup_indices.append(i)
        if len(dedup_indices) < len(cross_names):
            n_dups = len(cross_names) - len(dedup_indices)
            log(f"  Deduplicating {n_dups} cross names from truncation collisions")
            cross_names = [str(cross_names[i]) for i in dedup_indices]
            # Also remove duplicate columns from sparse matrix
            sparse_mat = sparse_mat[:, dedup_indices]
        else:
            cross_names = [str(n) for n in cross_names]
        cross_names = validate_sparse_names_contract(sparse_mat, cross_names)
        if _NPZ_INDICES_ONLY:
            save_npz_indices_only(sparse_mat, npz_path)
            log(f"  Saved indices-only NPZ (no data array — all 1.0)")
        else:
            atomic_save_npz(sparse_mat, npz_path)
        # FIX 26: Verify NPZ integrity immediately after save
        try:
            from atomic_io import load_npz_auto
            _verify = load_npz_auto(npz_path)
            assert _verify.shape == sparse_mat.shape, f"NPZ verify failed: shape {_verify.shape} != {sparse_mat.shape}"
            assert _verify.nnz == sparse_mat.nnz, f"NPZ verify failed: nnz {_verify.nnz} != {sparse_mat.nnz}"
            del _verify
            log(f"  NPZ verified: {npz_path} ({os.path.getsize(npz_path)/1e6:.1f} MB)")
        except Exception as e:
            log(f"  NPZ CORRUPT: {e} — deleting and retrying")
            os.remove(npz_path)
            raise
        atomic_save_json(cross_names, names_path)
        with open(names_path, 'r', encoding='utf-8') as _fh:
            _saved_names = _json_mod.load(_fh)
        if len(_saved_names) != sparse_mat.shape[1]:
            raise RuntimeError(
                f"Final sparse/name mismatch after save: {len(_saved_names)} names vs {sparse_mat.shape[1]} cols"
            )

        # FIX 14: Also save as separate .npy files for mmap loading
        # NPZ silently ignores mmap_mode — .npy enables zero-copy load
        if symbol_file_tag:
            npy_dir = os.path.join(out_dir, f'v2_crosses_{symbol_file_tag}_{tf}_npy')
        else:
            npy_dir = os.path.join(out_dir, f'v2_crosses_{tf}_npy')
        try:
            save_csr_npy(sparse_mat, npy_dir)
            log(f"  Saved .npy memmap dir: {npy_dir}")
        except Exception as _npy_err:
            log(f"  WARNING: .npy save failed (NPZ still valid): {_npy_err}")

        # Final save succeeded — clean up per-cross-type checkpoints
        _cleanup_checkpoints()

        # Clean up memmap temp files (only exist if memmap merge was used)
        if _memmap_tmp_files:
            cleanup_memmap_files(_memmap_tmp_files)
            _memmap_tmp_files = []

        size_mb = os.path.getsize(npz_path) / 1e6
        log(f"  Saved: {npz_path} ({size_mb:.1f} MB)")

        # Also save base features (non-cross) as parquet
        base_path = os.path.join(out_dir, f'v2_base_{tf}.parquet')
        df.to_parquet(base_path)
        log(f"  Saved base: {base_path}")

        del sparse_mat
        gc.collect()
        _malloc_trim()

        # ── EFB Pre-Bundling ──
        # Pre-bundle binary cross features externally to bypass LightGBM's O(F^2) conflict graph.
        # Config-gated per TF. Output: *_bundled.npz + mapping JSON.
        try:
            from config import EFB_PREBUNDLE_ENABLED
            if EFB_PREBUNDLE_ENABLED.get(tf, False):
                log(f"\n  EFB PRE-BUNDLING ({tf})...")
                from efb_prebundler import prebundle_from_files
                prebundle_from_files(npz_path, names_path, output_dir=out_dir, tf=tf)
                log(f"  EFB pre-bundling complete")
            else:
                log(f"  EFB pre-bundling disabled for {tf} (config)")
        except Exception as e:
            log(f"  WARNING: EFB pre-bundling failed (non-fatal): {e}")
            log(f"  Training will use raw sparse + LightGBM internal EFB")
    elif total_crosses > 0:
        raise RuntimeError(
            f"Cross generation produced {total_crosses:,} features but no durable sparse sources "
            f"(checkpoints={len(_ckpt_files)}, in_memory={len(_csr_chunks)})"
        )

    log(f"  Sparse build: {time.time()-t_assign:.1f}s")

    elapsed = time.time() - t0
    log(f"\n  DONE: {N:,} rows, {len(df.columns):,} base cols + {total_crosses:,} cross cols ({elapsed:.0f}s)")

    # Final cleanup
    del all_cross_names
    gc.collect()
    _malloc_trim()

    # Return base df (crosses are in sparse file, not in df — too big for dense)
    return df


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    print("[XGEN-DEBUG] __main__ ENTERED", flush=True)
    parser = argparse.ArgumentParser(description='V2 Cross Feature Generator')
    parser.add_argument('--tf', nargs='+', default=['1d'], help='Timeframes to process')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--save-sparse', action='store_true', help='Save sparse .npz files')
    parser.add_argument('--input', help='Input parquet path (overrides auto-detect)')
    parser.add_argument('--symbol', default='BTC', help='Asset symbol')
    args = parser.parse_args()
    print(f"[XGEN-DEBUG] args={args}", flush=True)

    # ── V4: Pre-start GPU daemons BEFORE any CuPy-heavy code ──
    # This runs in __main__ (top-level process) so daemons fork from a clean state.
    # Previously this was inside generate_all_crosses() but the code path was never
    # reached due to unknown control flow issue. Moving it here guarantees execution.
    _main_daemon_handles = None
    if GPU and _MULTI_GPU_CROSS_GEN:
        print(f"[XGEN-DEBUG] __main__: Detecting GPUs for daemon prestage...", flush=True)
        _main_available_gpus = _detect_available_gpus()
        _main_n_gpus = len(_main_available_gpus)
        print(f"[XGEN-DEBUG] __main__: Detected {_main_n_gpus} GPUs: {_main_available_gpus}", flush=True)
        if _main_n_gpus > 1:
            try:
                from gpu_daemon import prestage_gpu_daemons, shutdown_daemons
                print(f"[XGEN-DEBUG] __main__: Calling prestage_gpu_daemons(n_gpus={_main_n_gpus})...", flush=True)
                _main_daemon_handles = prestage_gpu_daemons(
                    n_gpus=_main_n_gpus,
                    available_gpu_ids=_main_available_gpus,
                    vram_limit_pct=0.85
                )
                _main_active = sum(1 for h in _main_daemon_handles if h.status == 'idle')
                print(f"[XGEN-DEBUG] __main__: Daemon prestage done: {_main_active}/{_main_n_gpus} active", flush=True)
                log(f"[DAEMON] {_main_active}/{_main_n_gpus} GPU daemons pre-started from __main__")
                if _main_active < 2:
                    log(f"[DAEMON] WARNING: Only {_main_active} alive, will fall back to legacy path")
                    shutdown_daemons(_main_daemon_handles, timeout=10)
                    _main_daemon_handles = None
            except Exception as _main_daemon_err:
                import traceback
                print(f"[XGEN-DEBUG] __main__: DAEMON PRESTAGE FAILED: {_main_daemon_err}", flush=True)
                traceback.print_exc()
                log(f"[DAEMON] WARNING: prestage failed ({_main_daemon_err}), will use legacy path")
                _main_daemon_handles = None
    else:
        print(f"[XGEN-DEBUG] __main__: Skipping daemon prestage (GPU={GPU}, MULTI_GPU={_MULTI_GPU_CROSS_GEN})", flush=True)

    for tf in args.tf:
        print(f"[XGEN-DEBUG] __main__: Processing tf={tf}", flush=True)
        # Look for input parquet
        if args.input:
            path = args.input
        else:
            # Try V2 path first, then V1
            candidates = [
                os.path.join(V2_DIR, f'features_{args.symbol}_{tf}.parquet'),
                os.path.join(V2_DIR, f'features_{tf}.parquet'),
            ]
            path = None
            for c in candidates:
                if os.path.exists(c):
                    path = c
                    break

            if path is None:
                log(f"[SKIP] No parquet found for {args.symbol} {tf}")
                continue

        log(f"Loading {path}...")
        print(f"[XGEN-DEBUG] __main__: Loading parquet from {path}...", flush=True)
        df = pd.read_parquet(path)
        print(f"[XGEN-DEBUG] __main__: Parquet loaded, {len(df)} rows, calling generate_all_crosses()", flush=True)
        # Set symbol attribute so output files include symbol in name
        # (v2_crosses_BTC_{tf}.npz, not v2_crosses_{tf}.npz)
        # pd.read_parquet doesn't preserve custom attributes
        df._v2_symbol = args.symbol
        df = generate_all_crosses(df, tf=tf, gpu_id=args.gpu,
                                   save_sparse=args.save_sparse,
                                   output_dir=V2_DIR,
                                   daemon_handles=_main_daemon_handles)

        # Save expanded parquet
        out_path = os.path.join(V2_DIR, f'features_{args.symbol}_{tf}_v2.parquet')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_parquet(out_path)
        log(f"Saved: {out_path}")

    # Shutdown daemons after all TFs processed
    if _main_daemon_handles is not None:
        try:
            from gpu_daemon import shutdown_daemons
            print(f"[XGEN-DEBUG] __main__: Shutting down daemons after all TFs...", flush=True)
            shutdown_daemons(_main_daemon_handles, timeout=60)
        except Exception:
            pass
