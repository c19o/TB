#!/usr/bin/env python
"""
atomic_io.py — Atomic file save helpers for V2 pipeline
=========================================================
Writes to temp file, then os.replace() for atomicity.
If process crashes mid-write, no corrupt files left behind.
"""

import os
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


def atomic_save_parquet(df, path, **kwargs):
    """Atomic parquet save."""
    with atomic_save(path) as tmp:
        df.to_parquet(tmp, **kwargs)


def atomic_save_npz(sparse_mat, path):
    """Atomic scipy sparse npz save."""
    from scipy import sparse
    with atomic_save(path) as tmp:
        sparse.save_npz(tmp, sparse_mat)
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
