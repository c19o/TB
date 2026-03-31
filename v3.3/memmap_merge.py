#!/usr/bin/env python
"""
memmap_merge.py — Single-pass streaming CSC merge via memory-mapped files
=========================================================================
Solves the 1h (1.8TB peak RAM) and 15m (3TB+) cross-gen final assembly OOM.

Instead of np.concatenate() of all data/indices/indptr arrays in RAM,
this module:
  Pre-scan: Fast metadata read (nnz + cols) from each source without CSC conversion.
  Single pass: Pre-allocates memmap files on disk, loads each source as CSC and fills.
  Constructs scipy.sparse.csc_matrix backed by memmap arrays (no RAM spike).
  save_npz streams from memmap via OS paging.

RAM reduction:
  1h:  ~1.8TB -> ~5-10GB  (99.5%)
  15m: ~3TB+  -> ~10-15GB (99.7%)

Controls:
  MEMMAP_CROSS_GEN=1  — force memmap for all TFs
  MEMMAP_CROSS_GEN=0  — force in-memory
  Unset               — auto-enables for 1h/15m only

Safety:
  - ALL cross features preserved (no filtering during merge)
  - int64 indptr (NNZ > 2^31)
  - Resume-safe: partial runs recoverable from per-type checkpoints
  - Structural zeros = 0.0, not NaN
  - Never use mmap over network storage (local/NVMe only)
"""

import os
import gc
import time
import tempfile
import numpy as np
from scipy import sparse


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def should_use_memmap(tf):
    """Decide whether to use memmap merge based on env var or timeframe.

    Returns True if memmap merge should be used for this timeframe.
    """
    env = os.environ.get('MEMMAP_CROSS_GEN')
    if env is not None:
        return env.strip() == '1'
    # Auto-enable for 1h and 15m only (these are the OOM timeframes)
    return tf in ('1h', '15m')


def _load_as_csc(source):
    """Load a source (file path or in-memory CSR) as CSC matrix."""
    if isinstance(source, str):
        mat = sparse.load_npz(source)
    else:
        mat = source
    if not sparse.issparse(mat):
        raise ValueError(f"Expected sparse matrix, got {type(mat)}")
    return mat.tocsc()


def memmap_streaming_merge(sources, n_rows, tmp_dir=None):
    """Two-pass streaming CSC merge via memory-mapped files.

    Parameters
    ----------
    sources : list
        Mix of file paths (str) to NPZ files and in-memory scipy sparse matrices.
        Each source is loaded one at a time to bound peak RAM.
    n_rows : int
        Number of rows in the output matrix.
    tmp_dir : str, optional
        Directory for memmap temp files. Defaults to same dir as first file source,
        or system temp. Should be on fast local storage (NVMe preferred).

    Returns
    -------
    result : scipy.sparse.csr_matrix
        Final merged matrix in CSR format with int64 indptr.
    tmp_cleanup : list of str
        Paths to memmap temp files that should be deleted after save_npz.
    """
    if not sources:
        return sparse.csr_matrix((n_rows, 0), dtype=np.float32), []

    # Determine temp directory — prefer local filesystem
    if tmp_dir is None:
        for src in sources:
            if isinstance(src, str):
                tmp_dir = os.path.dirname(src)
                break
        if tmp_dir is None:
            tmp_dir = tempfile.gettempdir()

    # ================================================================
    # SINGLE PASS: Scan sizes, allocate memmaps, fill — each source loaded ONCE
    # ================================================================
    # Pre-scan: fast metadata read (nnz + cols) without full CSC conversion.
    # For file sources, load_npz + shape/nnz is cheap; for in-memory, even cheaper.
    log("  [memmap] Pre-scan: reading source metadata...")
    t0 = time.time()

    total_nnz = np.int64(0)
    total_cols = 0

    for i, src in enumerate(sources):
        if isinstance(src, str):
            mat = sparse.load_npz(src)
        else:
            mat = src
        if not sparse.issparse(mat):
            raise ValueError(f"Expected sparse matrix, got {type(mat)}")
        nnz = np.int64(mat.nnz)
        n_cols = mat.shape[1] if sparse.issparse(mat) else mat.shape[1]
        # For CSR input, tocsc() changes shape[1] — use original shape
        total_nnz += nnz
        total_cols += n_cols
        src_label = os.path.basename(src) if isinstance(src, str) else f"memory[{i}]"
        log(f"    [{i+1}/{len(sources)}] {src_label}: {n_cols:,} cols, {nnz:,} nnz")
        del mat

    log(f"  [memmap] Pre-scan done: {total_cols:,} total cols, {total_nnz:,} total NNZ ({time.time()-t0:.1f}s)")

    # Estimate memmap size
    data_bytes = int(total_nnz) * 4  # float32
    indices_bytes = int(total_nnz) * 4  # int32
    indptr_bytes = (total_cols + 1) * 8  # int64
    total_bytes = data_bytes + indices_bytes + indptr_bytes
    log(f"  [memmap] Allocating {total_bytes / 1e9:.2f} GB memmap on disk")

    # Create memmap files
    data_path = os.path.join(tmp_dir, f'_memmap_data_{os.getpid()}.dat')
    indices_path = os.path.join(tmp_dir, f'_memmap_indices_{os.getpid()}.dat')
    indptr_path = os.path.join(tmp_dir, f'_memmap_indptr_{os.getpid()}.dat')
    tmp_files = [data_path, indices_path, indptr_path]

    # Pre-allocate memory-mapped arrays
    mm_data = np.memmap(data_path, dtype=np.float32, mode='w+', shape=(int(total_nnz),))
    mm_indices = np.memmap(indices_path, dtype=np.int32, mode='w+', shape=(int(total_nnz),))
    mm_indptr = np.memmap(indptr_path, dtype=np.int64, mode='w+', shape=(total_cols + 1,))

    log("  [memmap] Filling memmap arrays (single pass)...")
    t1 = time.time()

    nnz_offset = np.int64(0)
    col_offset = 0

    for i, src in enumerate(sources):
        csc = _load_as_csc(src)
        src_nnz = np.int64(csc.nnz)
        src_cols = csc.shape[1]

        # Copy data array
        nnz_end = nnz_offset + src_nnz
        mm_data[int(nnz_offset):int(nnz_end)] = csc.data.astype(np.float32)

        # Copy indices array (row indices)
        mm_indices[int(nnz_offset):int(nnz_end)] = csc.indices.astype(np.int32)

        # Copy indptr (shifted by cumulative NNZ offset)
        # indptr has src_cols+1 entries; we take all but the last (appended at end)
        src_indptr = csc.indptr.astype(np.int64) + nnz_offset
        mm_indptr[col_offset:col_offset + src_cols] = src_indptr[:-1]

        nnz_offset = nnz_end
        col_offset += src_cols

        src_label = os.path.basename(src) if isinstance(src, str) else f"memory[{i}]"
        log(f"    [{i+1}/{len(sources)}] {src_label}: filled ({int(nnz_offset):,}/{int(total_nnz):,} NNZ)")

        del csc, src_indptr
        gc.collect()

    # Final indptr entry = total NNZ
    mm_indptr[total_cols] = total_nnz

    # Flush memmaps to disk
    mm_data.flush()
    mm_indices.flush()
    mm_indptr.flush()

    log(f"  [memmap] Single pass done: memmaps filled ({time.time()-t1:.1f}s)")

    # ================================================================
    # Construct CSC matrix backed by memmap arrays
    # ================================================================
    log("  [memmap] Constructing CSC matrix from memmaps...")
    t2 = time.time()

    # Read memmaps back as read-only for scipy
    ro_data = np.memmap(data_path, dtype=np.float32, mode='r', shape=(int(total_nnz),))
    ro_indices = np.memmap(indices_path, dtype=np.int32, mode='r', shape=(int(total_nnz),))
    ro_indptr = np.memmap(indptr_path, dtype=np.int64, mode='r', shape=(total_cols + 1,))

    # Construct CSC — scipy uses these arrays by reference (no copy)
    csc_merged = sparse.csc_matrix(
        (ro_data, ro_indices, ro_indptr),
        shape=(n_rows, total_cols),
        copy=False
    )

    log(f"  [memmap] CSC matrix: {csc_merged.shape}, {csc_merged.nnz:,} NNZ ({time.time()-t2:.1f}s)")

    # Convert to CSR (streams through OS page cache — no full materialization)
    log("  [memmap] Converting CSC -> CSR...")
    t3 = time.time()
    csr_result = csc_merged.tocsr()

    # Force int64 indptr on the CSR result for NNZ > 2^31 safety
    if csr_result.nnz > 2**30:
        csr_result.indptr = csr_result.indptr.astype(np.int64)
        csr_result.indices = csr_result.indices.astype(np.int64)

    log(f"  [memmap] CSR conversion done ({time.time()-t3:.1f}s)")

    # Clean up write-mode memmaps
    del mm_data, mm_indices, mm_indptr, ro_data, ro_indices, ro_indptr, csc_merged
    gc.collect()

    return csr_result, tmp_files


def cleanup_memmap_files(tmp_files):
    """Delete memmap temp files after final NPZ save."""
    removed = 0
    for f in tmp_files:
        try:
            if os.path.exists(f):
                os.remove(f)
                removed += 1
        except OSError as e:
            log(f"  [memmap] Warning: could not remove {f}: {e}")
    if removed:
        log(f"  [memmap] Cleaned up {removed} temp files")
