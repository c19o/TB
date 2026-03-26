#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
knn_feature_engine.py — GPU-Accelerated KNN Pattern Similarity Feature Generator
=================================================================================
Uses CuPy (RTX 3090 CUDA) for batched distance computation + walk-forward masking.
Falls back to NumPy on CPU if CuPy not available.

Features per bar:
  - knn_direction:       time-decay weighted avg direction of K neighbors
  - knn_confidence:      agreement ratio (% of K neighbors same direction)
  - knn_avg_return:      time-decay weighted mean return after similar patterns
  - knn_best_match_dist: distance to closest match
  - knn_pattern_std:     std of current pattern (volatility context)

Walk-forward safe: only uses patterns BEFORE the current bar.
GPU batched: processes 4000-8000 queries at once, minutes not hours.
"""

import os
import numpy as np
import time as _time
from numpy.lib.stride_tricks import sliding_window_view

# GPU backend — respects V2_SKIP_GPU env var (set by feature_library.py on CUDA 13+)
# and does its own driver version check as a safety net.
def _cuda_major():
    """Detect CUDA major version from driver. Returns 13 if driver >= 580, else 12."""
    try:
        import subprocess as _sp
        _nv = _sp.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                       capture_output=True, text=True, timeout=5)
        _drv = int(_nv.stdout.strip().split('.')[0])
        return 13 if _drv >= 580 else 12
    except Exception:
        return 12

_skip_gpu = os.environ.get('V2_SKIP_GPU', '') == '1' or _cuda_major() >= 13

if _skip_gpu:
    cp = None
    GPU_KNN = False
    _reason = "V2_SKIP_GPU=1" if os.environ.get('V2_SKIP_GPU') == '1' else "CUDA 13+ driver (580+)"
    print(f"[KNN] GPU disabled — {_reason}. CuPy/RAPIDS segfault on CUDA 13. Using CPU fallback.")
else:
    try:
        import cupy as cp
        GPU_KNN = True
        print(f"[KNN] CuPy GPU detected — GPU-accelerated KNN")
    except ImportError:
        cp = None
        GPU_KNN = False
        print(f"[KNN] CuPy not available — CPU fallback")

# Per-TF config
KNN_TF_CONFIG = {
    '15m': {'pattern_len': 30, 'k': 50, 'max_lookback_bars': 70080},
    '1h':  {'pattern_len': 48, 'k': 50, 'max_lookback_bars': 26280},
    '4h':  {'pattern_len': 30, 'k': 50, 'max_lookback_bars': 6570},
    '1d':  {'pattern_len': 20, 'k': 50, 'max_lookback_bars': 1460},
    '1w':  {'pattern_len': 12, 'k': 30, 'max_lookback_bars': 208},
}

# Time decay: lambda for exp(-lambda * bars_ago)
DECAY_LAMBDAS = {
    '15m': 0.00002,
    '1h':  0.00008,
    '4h':  0.0003,
    '1d':  0.002,
    '1w':  0.013,
}


def build_pattern_matrix(pct_changes, pattern_len):
    """Build sliding-window pattern matrix. Row i = pattern ending at bar i."""
    n = len(pct_changes)
    if n < pattern_len:
        return np.full((n, pattern_len), np.nan, dtype=np.float32)
    windows = sliding_window_view(pct_changes, pattern_len)
    result = np.full((n, pattern_len), np.nan, dtype=np.float32)
    result[pattern_len - 1:] = windows
    return result


def zscore_normalize(patterns):
    """Z-score normalize each pattern row. Returns (normalized, stds)."""
    means = np.nanmean(patterns, axis=1, keepdims=True)
    stds = np.nanstd(patterns, axis=1, keepdims=True)
    stds_safe = np.where(stds > 1e-10, stds, np.nan)
    normalized = (patterns - means) / stds_safe
    return normalized, stds.squeeze()


def _sq_euclidean_gpu(Q, R):
    """Squared Euclidean distance matrix on GPU. Q:(B,d), R:(N,d) -> (B,N)
    Uses element-wise ops to avoid cuBLAS dependency."""
    B, d = Q.shape
    N = R.shape[0]
    # For d=30, expanding is fine memory-wise: B*N*30*4 bytes
    # Q: (B,1,d), R: (1,N,d) -> diff: (B,N,d) -> sum -> (B,N)
    # But this uses B*N*d memory which could be large.
    # Split into sub-batches if needed.
    MAX_ELEMENTS = 500_000_000  # ~2GB for float32
    elements = B * N * d
    if elements <= MAX_ELEMENTS:
        diff = Q[:, None, :] - R[None, :, :]  # (B, N, d)
        D = cp.sum(diff ** 2, axis=2)  # (B, N)
        return D
    else:
        # Sub-batch over queries
        sub_b = max(1, MAX_ELEMENTS // (N * d))
        D = cp.empty((B, N), dtype=cp.float32)
        for si in range(0, B, sub_b):
            se = min(si + sub_b, B)
            diff = Q[si:se, None, :] - R[None, :, :]
            D[si:se] = cp.sum(diff ** 2, axis=2)
        return D


def _sq_euclidean_cpu(Q, R):
    """Squared Euclidean distance matrix on CPU. Q:(B,d), R:(N,d) -> (B,N)"""
    Q_norm = np.sum(Q ** 2, axis=1, keepdims=True)
    R_norm = np.sum(R ** 2, axis=1, keepdims=True).T
    cross = Q @ R.T
    D = Q_norm + R_norm - 2 * cross
    return np.maximum(D, 0)


def compute_knn_features_walkforward(pct_changes, next_returns, tf_name,
                                      timestamps_unix=None):
    """
    GPU-accelerated walk-forward KNN feature computation.

    Batches queries, computes distance matrix on GPU, masks future bars,
    finds top-K per row, computes weighted features.
    """
    cfg = KNN_TF_CONFIG.get(tf_name, KNN_TF_CONFIG['1h'])
    pl = cfg['pattern_len']
    k = cfg['k']
    max_lookback = cfg['max_lookback_bars']
    decay_lambda = DECAY_LAMBDAS.get(tf_name, 0.0001)

    n = len(pct_changes)
    min_start = pl + k + 1

    # Build and normalize patterns
    raw_patterns = build_pattern_matrix(pct_changes, pl)
    norm_patterns, pattern_stds = zscore_normalize(raw_patterns)

    # Mark valid rows (no NaN in pattern AND has a valid next_return)
    valid_pattern = ~np.isnan(norm_patterns).any(axis=1)
    valid_return = ~np.isnan(next_returns)

    # Output arrays
    knn_direction = np.full(n, np.nan, dtype=np.float32)
    knn_confidence = np.full(n, np.nan, dtype=np.float32)
    knn_avg_return = np.full(n, np.nan, dtype=np.float32)
    knn_best_match_dist = np.full(n, np.nan, dtype=np.float32)
    knn_pattern_std = pattern_stds.astype(np.float32)

    if n <= min_start:
        return {
            'knn_direction': knn_direction, 'knn_confidence': knn_confidence,
            'knn_avg_return': knn_avg_return, 'knn_best_match_dist': knn_best_match_dist,
            'knn_pattern_std': knn_pattern_std,
        }

    # NaN patterns/returns are already masked out by valid_pattern and valid_return masks (lines 133-134).
    # Distance computation uses those masks to set invalid entries to 1e30, so NaN rows never contribute.
    X = norm_patterns.astype(np.float32)
    returns_clean = next_returns.astype(np.float32)

    # Batch size for queries
    B_q = 4000 if GPU_KNN else 2000
    t0 = _time.time()
    total_computed = 0

    for q_start in range(min_start, n, B_q):
        q_end = min(n, q_start + B_q)
        b_size = q_end - q_start

        # Reference window: all bars that ANY query in this batch could see
        global_ref_start = max(0, q_start - max_lookback)
        global_ref_end = q_end  # queries see up to their own index (exclusive)

        R_total = global_ref_end - global_ref_start

        # Get query and reference slices
        Q_np = X[q_start:q_end]  # (b_size, d)
        R_np = X[global_ref_start:global_ref_end]  # (R_total, d)

        if GPU_KNN:
            Q_gpu = cp.asarray(Q_np)
            R_gpu = cp.asarray(R_np)

            # Squared Euclidean distances: (b_size, R_total)
            D = _sq_euclidean_gpu(Q_gpu, R_gpu)

            # Walk-forward mask: query i can only see refs with global index < i
            q_global = cp.arange(q_start, q_end, dtype=cp.int32)  # (b_size,)
            r_global = cp.arange(global_ref_start, global_ref_end, dtype=cp.int32)  # (R_total,)
            # Mask: ref >= query (future bars) -> inf
            future_mask = r_global[None, :] >= q_global[:, None]  # (b_size, R_total)

            # Also mask invalid patterns (NaN original) and invalid returns
            ref_valid_pattern = cp.asarray(valid_pattern[global_ref_start:global_ref_end])
            ref_valid_return = cp.asarray(valid_return[global_ref_start:global_ref_end])
            invalid_ref = ~(ref_valid_pattern & ref_valid_return)  # (R_total,)
            invalid_mask = invalid_ref[None, :] | future_mask  # (b_size, R_total)

            # Also mask queries that have invalid patterns
            q_valid = cp.asarray(valid_pattern[q_start:q_end])  # (b_size,)

            D = cp.where(invalid_mask, cp.float32(1e30), D)

            # Lookback mask: ref too far in the past
            lookback_mask = (q_global[:, None] - r_global[None, :]) > max_lookback
            D = cp.where(lookback_mask, cp.float32(1e30), D)

            # Top-K per row via argpartition
            k_actual = min(k, R_total)
            if k_actual < 3:
                continue

            # argpartition: get indices of K smallest distances per row
            idx_part = cp.argpartition(D, k_actual - 1, axis=1)[:, :k_actual]  # (b_size, k)
            row_idx = cp.arange(b_size)[:, None]
            d_part = D[row_idx, idx_part]  # (b_size, k)

            # Sort within K
            order = cp.argsort(d_part, axis=1)
            idx_sorted = idx_part[row_idx, order]  # (b_size, k)
            d_sorted = d_part[row_idx, order]  # (b_size, k)

            # Map to global bar indices
            idx_global = idx_sorted + global_ref_start  # (b_size, k) global bar indices

            # Get returns for neighbors
            neighbor_returns = cp.asarray(returns_clean)[idx_global]  # (b_size, k)

            # Distance weights: 1/sqrt(d) (using sqrt of squared dist)
            d_sqrt = cp.sqrt(d_sorted + 1e-8)
            dist_weights = 1.0 / (d_sqrt + 1e-8)

            # Time-decay weights
            bars_ago = q_global[:, None] - idx_global  # (b_size, k)
            time_weights = cp.exp(-decay_lambda * bars_ago.astype(cp.float32))

            # Combined weights
            weights = dist_weights * time_weights  # (b_size, k)

            # Mask out invalid entries (distance was 1e30)
            valid_k = d_sorted < 1e29
            weights = cp.where(valid_k, weights, 0.0)
            valid_k_count = valid_k.sum(axis=1)  # (b_size,)

            total_w = weights.sum(axis=1, keepdims=True)  # (b_size, 1)
            total_w = cp.where(total_w > 0, total_w, 1.0)  # avoid div by 0

            # Weighted features
            w_return = (weights * neighbor_returns).sum(axis=1) / total_w.squeeze()
            direction_sign = cp.where(neighbor_returns > 0, 1.0, -1.0)
            w_direction = (weights * direction_sign).sum(axis=1) / total_w.squeeze()

            up_count = (valid_k & (neighbor_returns > 0)).sum(axis=1).astype(cp.float32)
            dn_count = (valid_k & (neighbor_returns <= 0)).sum(axis=1).astype(cp.float32)
            agreement = cp.maximum(up_count, dn_count) / cp.maximum(valid_k_count.astype(cp.float32), 1.0)

            best_dist = cp.sqrt(d_sorted[:, 0] + 1e-8)

            # Transfer back to CPU
            w_direction_np = w_direction.get()
            agreement_np = agreement.get()
            w_return_np = w_return.get()
            best_dist_np = best_dist.get()
            q_valid_np = q_valid.get()
            valid_k_count_np = valid_k_count.get()

            # Free GPU memory
            del Q_gpu, R_gpu, D, future_mask, invalid_mask, lookback_mask
            del idx_part, d_part, idx_sorted, d_sorted, idx_global
            del neighbor_returns, weights, valid_k
            cp.get_default_memory_pool().free_all_blocks()

        else:
            # CPU fallback
            D = _sq_euclidean_cpu(Q_np, R_np)

            q_global = np.arange(q_start, q_end, dtype=np.int32)
            r_global = np.arange(global_ref_start, global_ref_end, dtype=np.int32)

            future_mask = r_global[None, :] >= q_global[:, None]
            ref_valid = valid_pattern[global_ref_start:global_ref_end] & valid_return[global_ref_start:global_ref_end]
            invalid_mask = ~ref_valid[None, :] | future_mask
            lookback_mask = (q_global[:, None] - r_global[None, :]) > max_lookback

            D = np.where(invalid_mask | lookback_mask, 1e30, D)

            k_actual = min(k, R_total)
            if k_actual < 3:
                continue

            idx_part = np.argpartition(D, k_actual - 1, axis=1)[:, :k_actual]
            row_idx = np.arange(b_size)[:, None]
            d_part = D[row_idx, idx_part]
            order = np.argsort(d_part, axis=1)
            idx_sorted = idx_part[row_idx, order]
            d_sorted = d_part[row_idx, order]

            idx_global_np = idx_sorted + global_ref_start
            neighbor_returns_np = returns_clean[idx_global_np]

            d_sqrt = np.sqrt(d_sorted + 1e-8)
            dist_weights = 1.0 / (d_sqrt + 1e-8)
            bars_ago = q_global[:, None] - idx_global_np
            time_weights = np.exp(-decay_lambda * bars_ago.astype(np.float32))
            weights = dist_weights * time_weights

            valid_k = d_sorted < 1e29
            weights = np.where(valid_k, weights, 0.0)
            valid_k_count_np = valid_k.sum(axis=1)

            total_w = weights.sum(axis=1, keepdims=True)
            total_w = np.where(total_w > 0, total_w, 1.0)

            w_return_np = (weights * neighbor_returns_np).sum(axis=1) / total_w.squeeze()
            direction_sign = np.where(neighbor_returns_np > 0, 1.0, -1.0)
            w_direction_np = (weights * direction_sign).sum(axis=1) / total_w.squeeze()

            up_count = (valid_k & (neighbor_returns_np > 0)).sum(axis=1).astype(np.float32)
            dn_count = (valid_k & (neighbor_returns_np <= 0)).sum(axis=1).astype(np.float32)
            agreement_np = np.maximum(up_count, dn_count) / np.maximum(valid_k_count_np.astype(np.float32), 1.0)

            best_dist_np = np.sqrt(d_sorted[:, 0] + 1e-8)
            q_valid_np = valid_pattern[q_start:q_end]

        # Write results (only for valid queries with enough neighbors)
        for j in range(b_size):
            gi = q_start + j
            if not q_valid_np[j] or valid_k_count_np[j] < 3:
                continue
            knn_direction[gi] = w_direction_np[j]
            knn_confidence[gi] = agreement_np[j]
            knn_avg_return[gi] = w_return_np[j]
            knn_best_match_dist[gi] = best_dist_np[j]

        total_computed += b_size
        elapsed = _time.time() - t0
        rate = total_computed / max(elapsed, 0.01)
        remaining = n - min_start - total_computed
        eta = remaining / max(rate, 1) / 60
        if q_start % (B_q * 5) == 0 or q_end >= n:
            print(f"    KNN [{tf_name}]: {total_computed}/{n - min_start} bars "
                  f"({total_computed/(n-min_start)*100:.1f}%) | "
                  f"{rate:.0f} bars/s | ETA: {eta:.1f}min")

    total_time = _time.time() - t0
    valid_count = (~np.isnan(knn_direction)).sum()
    print(f"    KNN [{tf_name}] DONE: {valid_count} valid features in {total_time:.1f}s "
          f"({'GPU' if GPU_KNN else 'CPU'})")

    return {
        'knn_direction': knn_direction,
        'knn_confidence': knn_confidence,
        'knn_avg_return': knn_avg_return,
        'knn_best_match_dist': knn_best_match_dist,
        'knn_pattern_std': knn_pattern_std,
    }


def compute_knn_features_live(pct_changes, next_returns, tf_name,
                               timestamps_unix=None):
    """Compute KNN features for the LAST bar only (live trading). Fast single query."""
    cfg = KNN_TF_CONFIG.get(tf_name, KNN_TF_CONFIG['1h'])
    pl = cfg['pattern_len']
    k = cfg['k']
    max_lookback = cfg['max_lookback_bars']
    decay_lambda = DECAY_LAMBDAS.get(tf_name, 0.0001)

    n = len(pct_changes)
    nan_result = {
        'knn_direction': np.nan, 'knn_confidence': np.nan,
        'knn_avg_return': np.nan, 'knn_best_match_dist': np.nan,
        'knn_pattern_std': np.nan,
    }

    if n < pl + k + 1:
        return nan_result

    raw_patterns = build_pattern_matrix(pct_changes, pl)
    norm_patterns, pattern_stds = zscore_normalize(raw_patterns)

    i = n - 1
    cp_vec = norm_patterns[i]
    if np.any(np.isnan(cp_vec)):
        nan_result['knn_pattern_std'] = float(pattern_stds[i]) if not np.isnan(pattern_stds[i]) else np.nan
        return nan_result

    pool_start = max(pl - 1, i - max_lookback)
    pool_end = i

    pool_patterns = norm_patterns[pool_start:pool_end]
    pool_returns = next_returns[pool_start:pool_end]

    valid_mask = ~np.isnan(pool_patterns).any(axis=1) & ~np.isnan(pool_returns)
    if valid_mask.sum() < 3:
        nan_result['knn_pattern_std'] = float(pattern_stds[i])
        return nan_result

    valid_patterns = pool_patterns[valid_mask].astype(np.float32)
    valid_returns = pool_returns[valid_mask].astype(np.float32)
    valid_bar_offsets = np.arange(pool_start, pool_end)[valid_mask]

    # Distance computation (single query — CPU is fine)
    diffs = valid_patterns - cp_vec.astype(np.float32)
    dists = np.sqrt(np.einsum('ij,ij->i', diffs, diffs))

    k_actual = min(k, len(dists))
    if k_actual < 3:
        nan_result['knn_pattern_std'] = float(pattern_stds[i])
        return nan_result

    top_k_idx = np.argpartition(dists, k_actual)[:k_actual] if k_actual < len(dists) else np.arange(len(dists))
    top_k_dists = dists[top_k_idx]
    top_k_returns = valid_returns[top_k_idx]
    bars_ago = i - valid_bar_offsets[top_k_idx]

    dist_weights = 1.0 / (top_k_dists + 1e-8)
    time_weights = np.exp(-decay_lambda * bars_ago.astype(np.float64))
    weights = dist_weights * time_weights
    total_w = weights.sum()

    if total_w <= 0:
        nan_result['knn_pattern_std'] = float(pattern_stds[i])
        return nan_result

    weighted_return = np.dot(weights, top_k_returns) / total_w
    direction_sign = np.where(top_k_returns > 0, 1.0, -1.0)
    weighted_direction = np.dot(weights, direction_sign) / total_w
    up_count = (top_k_returns > 0).sum()
    agreement = max(up_count, k_actual - up_count) / k_actual

    return {
        'knn_direction': float(weighted_direction),
        'knn_confidence': float(agreement),
        'knn_avg_return': float(weighted_return),
        'knn_best_match_dist': float(top_k_dists.min()),
        'knn_pattern_std': float(pattern_stds[i]),
    }


def knn_features_from_ohlcv(opens, closes, tf_name, timestamps_unix=None,
                             walkforward=True):
    """Compute KNN features from raw OHLCV data."""
    pct_changes = np.zeros(len(closes), dtype=np.float64)
    pct_changes[1:] = np.where(
        closes[:-1] > 0,
        (closes[1:] - closes[:-1]) / closes[:-1] * 100,
        0
    )
    next_returns = np.full(len(closes), np.nan, dtype=np.float64)
    next_returns[:-1] = pct_changes[1:]

    if walkforward:
        return compute_knn_features_walkforward(
            pct_changes, next_returns, tf_name, timestamps_unix)
    else:
        return compute_knn_features_live(
            pct_changes, next_returns, tf_name, timestamps_unix)


if __name__ == '__main__':
    """Quick self-test with BTC data."""
    import sqlite3

    DB_DIR = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(DB_DIR, 'btc_prices.db')

    if not os.path.exists(db_path):
        print("btc_prices.db not found — skipping self-test")
        exit()

    conn = sqlite3.connect(db_path)

    for tf in ['1d', '4h', '1h', '15m']:
        print(f"\n{'='*50}")
        print(f"Testing KNN features for {tf}")

        rows = conn.execute(
            "SELECT open_time, open, close FROM ohlcv "
            "WHERE symbol='BTC/USDT' AND timeframe=? ORDER BY open_time",
            (tf,)
        ).fetchall()

        if not rows:
            print(f"  No data for {tf}")
            continue

        opens = np.array([r[1] for r in rows], dtype=np.float64)
        closes = np.array([r[2] for r in rows], dtype=np.float64)

        print(f"  Bars: {len(closes)}")

        t0 = _time.time()
        features = knn_features_from_ohlcv(opens, closes, tf, walkforward=True)
        elapsed = _time.time() - t0

        for name, arr in features.items():
            valid = ~np.isnan(arr)
            if valid.sum() > 0:
                print(f"  {name:25s}: {valid.sum():6d} valid of {len(arr)}, "
                      f"range [{np.nanmin(arr):+.4f}, {np.nanmax(arr):+.4f}]")
            else:
                print(f"  {name:25s}: 0 valid")

        print(f"  Time: {elapsed:.1f}s ({'GPU' if GPU_KNN else 'CPU'})")

    conn.close()
    print("\nDone.")
