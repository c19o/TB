#!/usr/bin/env python
"""
atomic_io.py — Atomic file save helpers for V2 pipeline
=========================================================
Writes to temp file, then os.replace() for atomicity.
If process crashes mid-write, no corrupt files left behind.
"""

import os
import numpy as np
from contextlib import contextmanager


@contextmanager
def atomic_save(final_path, suffix='.tmp'):
    """Context manager for atomic file writes.

    Usage:
        with atomic_save('features_BTC_1d.parquet') as tmp:
            df.to_parquet(tmp)
        # Now atomically at features_BTC_1d.parquet
    """
    tmp_path = final_path + suffix
    try:
        yield tmp_path
        os.replace(tmp_path, final_path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_save_parquet(df, path, downcast_float64=True, **kwargs):
    """Atomic parquet save.

    Args:
        downcast_float64: If True, convert float64 columns to float32 before saving.
            LightGBM uses float32 histograms internally, so float64 wastes 2x RAM
            on disk and on load with zero benefit to training quality.
    """
    import numpy as np
    if downcast_float64:
        f64_cols = [c for c in df.columns if df[c].dtype == np.float64]
        if f64_cols:
            df = df.copy()
            df[f64_cols] = df[f64_cols].astype(np.float32)
    with atomic_save(path) as tmp:
        df.to_parquet(tmp, **kwargs)


def atomic_save_npz(sparse_mat, path):
    """Atomic scipy sparse npz save. compressed=False: skips lzma, ~5x faster
    load at cost of ~3x larger file — critical for Optuna restarts that reload
    the cross matrix on every trial."""
    from scipy import sparse
    with atomic_save(path) as tmp:
        sparse.save_npz(tmp, sparse_mat, compressed=False)
        # scipy appends .npz if tmp doesn't already end with it,
        # creating e.g. "crosses.npz.tmp.npz" instead of "crosses.npz.tmp".
        # Rename back so os.replace() in atomic_save finds the expected path.
        if not tmp.endswith('.npz'):
            os.rename(tmp + '.npz', tmp)


def atomic_save_json(obj, path, indent=2):
    """Atomic JSON save with numpy type handling."""
    import json
    import numpy as np

    def _convert(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.str_, np.bytes_)):
            return str(o)
        raise TypeError(f"Not serializable: {type(o)}")

    with atomic_save(path) as tmp:
        with open(tmp, 'w') as f:
            json.dump(obj, f, indent=indent, default=_convert)


def atomic_save_pickle(obj, path):
    """Atomic pickle save."""
    import pickle
    with atomic_save(path) as tmp:
        with open(tmp, 'wb') as f:
            pickle.dump(obj, f)


def atomic_save_torch(state_dict, path):
    """Atomic PyTorch checkpoint save."""
    import torch
    with atomic_save(path) as tmp:
        torch.save(state_dict, tmp)


def save_csr_npy(sparse_mat, directory):
    """Save CSR matrix as separate .npy files for mmap loading.

    Saves data.npy, indices.npy, indptr.npy, shape.npy separately.
    Enables np.load(mmap_mode='r') for zero-copy loading — no deserialization,
    no memory allocation until pages are actually touched.

    Use NPZ for cloud transfer (single file), .npy for local training (zero-copy).
    """
    from scipy import sparse
    if not sparse.issparse(sparse_mat):
        raise ValueError("Input must be a scipy sparse matrix")
    csr = sparse_mat.tocsr()
    os.makedirs(directory, exist_ok=True)
    with atomic_save(os.path.join(directory, 'data.npy')) as tmp:
        np.save(tmp, csr.data)
    with atomic_save(os.path.join(directory, 'indices.npy')) as tmp:
        np.save(tmp, csr.indices)
    with atomic_save(os.path.join(directory, 'indptr.npy')) as tmp:
        np.save(tmp, csr.indptr)
    with atomic_save(os.path.join(directory, 'shape.npy')) as tmp:
        np.save(tmp, np.array(csr.shape, dtype=np.int64))


def load_csr_npy(directory, mmap_mode='r'):
    """Load CSR matrix from separate .npy files with optional mmap.

    mmap_mode='r' gives zero-copy read-only access (OS pages in on demand).
    mmap_mode=None loads fully into RAM (use for training where random access is heavy).
    """
    import numpy as np
    from scipy import sparse
    data = np.load(os.path.join(directory, 'data.npy'), mmap_mode=mmap_mode)
    indices = np.load(os.path.join(directory, 'indices.npy'), mmap_mode=mmap_mode)
    indptr = np.load(os.path.join(directory, 'indptr.npy'), mmap_mode=mmap_mode)
    shape = tuple(np.load(os.path.join(directory, 'shape.npy')))
    return sparse.csr_matrix((data, indices, indptr), shape=shape, copy=False)


# ============================================================
# INDICES-ONLY NPZ (Optimization #8)
# ============================================================
# Binary crosses are all 1.0. Storing data array wastes ~40% disk.
# Store only indptr + indices; reconstruct data=np.ones on load.
# Toggle: NPZ_INDICES_ONLY=1 (default ON).

_NPZ_INDICES_ONLY = os.environ.get('NPZ_INDICES_ONLY', '1') != '0'


def save_npz_indices_only(csr_mat, path):
    """Save CSR matrix storing only indptr + indices (no data array).
    Binary crosses have all data=1.0, so data is redundant.
    Atomic: writes to temp file, then os.replace().
    Preserves int64 indptr for NNZ > 2^31.
    """
    import numpy as np
    tmp_path = path + '.tmp'
    try:
        np.savez(tmp_path,
                 indptr=csr_mat.indptr,
                 indices=csr_mat.indices,
                 shape=np.array(csr_mat.shape, dtype=np.int64),
                 _indices_only=np.array([1], dtype=np.int8))
        # np.savez appends .npz if path doesn't end with it
        actual_tmp = tmp_path if os.path.exists(tmp_path) else tmp_path + '.npz'
        actual_final = path if path.endswith('.npz') else path + '.npz'
        os.replace(actual_tmp, actual_final)
    except BaseException:
        for p in (tmp_path, tmp_path + '.npz'):
            try:
                os.unlink(p)
            except OSError:
                pass
        raise


def load_npz_auto(path):
    """Load NPZ with backward compatibility.
    Detects indices-only format (has '_indices_only' key) vs scipy sparse format.
    Returns scipy CSR matrix in both cases.
    """
    import numpy as np
    from scipy import sparse

    # Try indices-only format first (faster check: just look for our marker key)
    try:
        with np.load(path, allow_pickle=False) as npz:
            if '_indices_only' in npz:
                indptr = npz['indptr']
                indices = npz['indices']
                shape = tuple(npz['shape'])
                data = np.ones(len(indices), dtype=np.float32)
                return sparse.csr_matrix((data, indices, indptr), shape=shape)
    except Exception:
        pass

    # Fall back to scipy sparse format
    return sparse.load_npz(path)
