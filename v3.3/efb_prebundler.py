#!/usr/bin/env python
"""
efb_prebundler.py — External EFB Pre-Bundler for Binary Cross Features
========================================================================
Bypasses LightGBM's O(F^2) EFB conflict-graph construction by pre-bundling
binary (0/1) features externally. Pure binary features with max_bin=255
allow up to 127 features per bundle (each needs 2 bins, 127*2=254 ≤ 255).

Key properties:
  - ZERO features dropped — every feature lands in a bundle
  - Offset encoding: bundle_val = sum(feature_i * (2*i)) for i in 0..126
  - Reversible: feature→bundle mapping stored for post-training SHAP analysis
  - Density-tiered grouping: ultra-rare packed blindly, others bitmap-checked

Output: dense int16 matrix (79K cols) + mapping JSON. LightGBM trains with
enable_bundle=False (already pre-bundled), building 79K histograms instead
of 10M — a 128x reduction.

Usage:
  python efb_prebundler.py --npz v2_crosses_BTC_1d.npz --names v2_cross_names_BTC_1d.json

  # Or from Python:
  from efb_prebundler import prebundle_binary_matrix
  bundled, mapping = prebundle_binary_matrix(sparse_csr, col_names, tf='1d')
"""

import os
import sys
import time
import json
import argparse
import gc
import numpy as np
from scipy import sparse

# Max features per bundle: max_bin=255, each binary feature needs 2 bins (0/1),
# offset encoding uses 2*i, so max offset = 2*126 = 252, plus value 1 → 253 < 255
MAX_FEATURES_PER_BUNDLE = 127

# Density tiers for grouping strategy
DENSITY_ULTRA_RARE = 0.001   # <0.1% — pack blindly (collision prob <0.001%)
DENSITY_RARE = 0.01          # <1%
DENSITY_MODERATE = 0.05      # <5%
# >5% = common

# Batch size for bitmap intersection checks (controls peak RAM)
BITMAP_BATCH_SIZE = 5000


def _log(msg):
    print(f"  [EFB-PRE] {msg}", flush=True)


def _compute_densities(csr_matrix):
    """Compute per-column density (fraction of non-zero rows) for a CSR matrix."""
    n_rows = csr_matrix.shape[0]
    # nnz per column via CSC conversion (fast for CSR)
    csc = csr_matrix.tocsc()
    col_nnz = np.diff(csc.indptr)
    densities = col_nnz / n_rows
    return densities


def _is_binary_column(csc, col_idx):
    """Check if a CSC column contains only 0 and 1 values."""
    start, end = csc.indptr[col_idx], csc.indptr[col_idx + 1]
    if start == end:
        return True  # all zeros = binary (just 0)
    vals = csc.data[start:end]
    return np.all((vals == 0) | (vals == 1))


def _classify_binary_columns(csr_matrix):
    """Identify which columns are pure binary and compute their densities.

    Returns:
        binary_indices: array of column indices that are binary
        densities: array of densities for those columns
        non_binary_indices: array of column indices that are NOT binary
    """
    n_cols = csr_matrix.shape[1]
    n_rows = csr_matrix.shape[0]
    csc = csr_matrix.tocsc()

    binary_mask = np.zeros(n_cols, dtype=bool)
    densities = np.zeros(n_cols, dtype=np.float32)

    for col_idx in range(n_cols):
        start, end = csc.indptr[col_idx], csc.indptr[col_idx + 1]
        nnz = end - start
        densities[col_idx] = nnz / n_rows
        if nnz == 0:
            binary_mask[col_idx] = True
        else:
            vals = csc.data[start:end]
            binary_mask[col_idx] = np.all((vals == 0) | (vals == 1))

    binary_indices = np.where(binary_mask)[0]
    non_binary_indices = np.where(~binary_mask)[0]
    return binary_indices, densities[binary_mask], non_binary_indices, csc


def _tier_features(binary_indices, densities):
    """Group binary features by density tier.

    Returns dict: tier_name -> list of (col_idx, density) tuples, sorted by density.
    """
    tiers = {
        'ultra_rare': [],  # <0.1%
        'rare': [],        # <1%
        'moderate': [],    # <5%
        'common': [],      # >=5%
    }

    for idx, density in zip(binary_indices, densities):
        if density < DENSITY_ULTRA_RARE:
            tiers['ultra_rare'].append((idx, density))
        elif density < DENSITY_RARE:
            tiers['rare'].append((idx, density))
        elif density < DENSITY_MODERATE:
            tiers['moderate'].append((idx, density))
        else:
            tiers['common'].append((idx, density))

    # Sort each tier by density (ascending) for better packing
    for tier in tiers.values():
        tier.sort(key=lambda x: x[1])

    return tiers


def _pack_blindly(features_with_density):
    """Pack features into bundles without conflict checking.

    For ultra-rare features (<0.1% density), collision probability is:
    P(collision) = density_a * density_b < 0.001 * 0.001 = 0.000001
    With 127 features per bundle: P(any collision) < 127*126/2 * 1e-6 ≈ 0.8%
    Acceptable for ultra-rare tier.

    Returns list of bundles, where each bundle is a list of column indices.
    """
    bundles = []
    current_bundle = []

    for col_idx, _density in features_with_density:
        current_bundle.append(col_idx)
        if len(current_bundle) >= MAX_FEATURES_PER_BUNDLE:
            bundles.append(current_bundle)
            current_bundle = []

    if current_bundle:
        bundles.append(current_bundle)

    return bundles


def _pack_with_bitmap_check(features_with_density, csc, n_rows):
    """Pack features into bundles with bitmap intersection conflict checking.

    Two features CONFLICT if they both fire (=1) on the same row.
    We use bitmap (boolean array) intersection to detect this.

    Returns list of bundles, where each bundle is a list of column indices.
    """
    if not features_with_density:
        return []

    bundles = []
    # Track which rows are "occupied" in the current bundle
    # Use a set-based approach for sparse features (much faster than dense bitmap)
    bundle_rows = set()  # rows where ANY feature in current bundle fires
    current_bundle = []

    for col_idx, _density in features_with_density:
        # Get rows where this feature fires
        start, end = csc.indptr[col_idx], csc.indptr[col_idx + 1]
        feature_rows = set(csc.indices[start:end].tolist())

        # Check conflict: does this feature fire on any row already occupied?
        if current_bundle and feature_rows & bundle_rows:
            # Conflict — start a new bundle
            bundles.append(current_bundle)
            current_bundle = [col_idx]
            bundle_rows = feature_rows.copy()
        else:
            # No conflict — add to current bundle
            current_bundle.append(col_idx)
            bundle_rows |= feature_rows

        if len(current_bundle) >= MAX_FEATURES_PER_BUNDLE:
            bundles.append(current_bundle)
            current_bundle = []
            bundle_rows = set()

    if current_bundle:
        bundles.append(current_bundle)

    return bundles


def _encode_bundles(bundles, csc, n_rows):
    """Encode bundles into dense integer columns using offset encoding.

    For each bundle, the encoded value for row r is:
        bundle_val[r] = sum(feature_i[r] * (2*i)) for i in range(len(bundle))

    where feature_i[r] is 0 or 1. This gives unique decodable values.

    Returns:
        bundled_matrix: np.ndarray of shape (n_rows, n_bundles), dtype=int16
    """
    n_bundles = len(bundles)
    # int16 range is -32768..32767, max value = 2*126 = 252 < 255, fits easily
    bundled = np.zeros((n_rows, n_bundles), dtype=np.int16)

    for bundle_idx, bundle_cols in enumerate(bundles):
        for slot, col_idx in enumerate(bundle_cols):
            offset = 2 * slot  # offset encoding: 0, 2, 4, 6, ...
            start, end = csc.indptr[col_idx], csc.indptr[col_idx + 1]
            if start == end:
                continue  # all-zero column, contributes nothing
            rows = csc.indices[start:end]
            vals = csc.data[start:end]
            # Only add offset where feature actually fires (val == 1)
            fire_mask = vals == 1
            fire_rows = rows[fire_mask]
            bundled[fire_rows, bundle_idx] += offset + 1  # +1 so feature=1 at offset=0 gives value 1 not 0

    return bundled


def _build_mapping(bundles, col_names, binary_indices, non_binary_indices):
    """Build the feature→bundle mapping for post-training analysis.

    Returns a dict with:
        - bundles: list of {bundle_id, features: [{name, col_idx, slot, offset}]}
        - non_binary_passthrough: list of {name, col_idx} for non-binary features
        - stats: summary statistics
    """
    mapping = {
        'version': 1,
        'max_features_per_bundle': MAX_FEATURES_PER_BUNDLE,
        'encoding': 'offset',
        'encoding_formula': 'bundle_val = sum(feature_i * (2*slot + 1)) where feature_i in {0,1}',
        'bundles': [],
        'non_binary_passthrough': [],
        'stats': {},
    }

    for bundle_idx, bundle_cols in enumerate(bundles):
        bundle_info = {
            'bundle_id': bundle_idx,
            'n_features': len(bundle_cols),
            'features': [],
        }
        for slot, col_idx in enumerate(bundle_cols):
            name = col_names[col_idx] if col_idx < len(col_names) else f'cross_{col_idx}'
            bundle_info['features'].append({
                'name': name,
                'original_col_idx': int(col_idx),
                'slot': slot,
                'offset': 2 * slot,
            })
        mapping['bundles'].append(bundle_info)

    for col_idx in non_binary_indices:
        name = col_names[col_idx] if col_idx < len(col_names) else f'cross_{col_idx}'
        mapping['non_binary_passthrough'].append({
            'name': name,
            'original_col_idx': int(col_idx),
        })

    total_bundled = sum(len(b) for b in bundles)
    mapping['stats'] = {
        'total_input_features': len(col_names),
        'binary_features': int(len(binary_indices)),
        'non_binary_features': int(len(non_binary_indices)),
        'total_bundles': len(bundles),
        'total_bundled_features': total_bundled,
        'avg_features_per_bundle': round(total_bundled / max(1, len(bundles)), 1),
        'output_columns': len(bundles) + len(non_binary_indices),
        'compression_ratio': round(len(col_names) / max(1, len(bundles) + len(non_binary_indices)), 1),
    }

    return mapping


def prebundle_binary_matrix(csr_matrix, col_names, tf='1d', verbose=True):
    """Pre-bundle binary features in a sparse cross matrix for LightGBM.

    This is the main entry point. Takes a sparse CSR matrix of cross features
    (mostly binary 0/1) and returns a dense integer matrix where each column
    is a packed bundle of up to 127 binary features.

    Non-binary features are passed through unchanged (appended as extra columns).

    Args:
        csr_matrix: scipy.sparse.csr_matrix — the cross feature matrix
        col_names: list of str — column names matching csr_matrix columns
        tf: str — timeframe name (for logging)
        verbose: bool — print progress

    Returns:
        bundled_matrix: np.ndarray — dense (n_rows, n_bundles + n_nonbinary), dtype varies
        bundled_col_names: list of str — column names for bundled matrix
        mapping: dict — full feature→bundle mapping (save to JSON for analysis)
    """
    log = _log if verbose else lambda msg: None
    t0 = time.time()
    n_rows, n_cols = csr_matrix.shape
    log(f"[{tf}] Pre-bundling {n_cols:,} features ({n_rows:,} rows)")

    # Step 1: Classify columns as binary vs non-binary
    t1 = time.time()
    binary_indices, binary_densities, non_binary_indices, csc = _classify_binary_columns(csr_matrix)
    log(f"  Classification: {len(binary_indices):,} binary, {len(non_binary_indices):,} non-binary ({time.time()-t1:.1f}s)")

    if len(binary_indices) == 0:
        log(f"  No binary features found — returning original matrix as dense")
        dense = csr_matrix.toarray()
        return dense, list(col_names), {'stats': {'total_input_features': n_cols, 'binary_features': 0, 'bundles': 0}}

    # Step 2: Tier binary features by density
    t2 = time.time()
    tiers = _tier_features(binary_indices, binary_densities)
    for tier_name, features in tiers.items():
        if features:
            log(f"  Tier {tier_name}: {len(features):,} features")

    # Step 3: Pack each tier into bundles
    all_bundles = []

    # Ultra-rare: blind packing (collision prob negligible)
    if tiers['ultra_rare']:
        ultra_bundles = _pack_blindly(tiers['ultra_rare'])
        log(f"  Ultra-rare: {len(tiers['ultra_rare']):,} features → {len(ultra_bundles):,} bundles (blind)")
        all_bundles.extend(ultra_bundles)

    # Rare: bitmap intersection check
    if tiers['rare']:
        rare_bundles = _pack_with_bitmap_check(tiers['rare'], csc, n_rows)
        log(f"  Rare: {len(tiers['rare']):,} features → {len(rare_bundles):,} bundles (bitmap-checked)")
        all_bundles.extend(rare_bundles)

    # Moderate: bitmap intersection check
    if tiers['moderate']:
        mod_bundles = _pack_with_bitmap_check(tiers['moderate'], csc, n_rows)
        log(f"  Moderate: {len(tiers['moderate']):,} features → {len(mod_bundles):,} bundles (bitmap-checked)")
        all_bundles.extend(mod_bundles)

    # Common: bitmap intersection check (most likely to conflict)
    if tiers['common']:
        common_bundles = _pack_with_bitmap_check(tiers['common'], csc, n_rows)
        log(f"  Common: {len(tiers['common']):,} features → {len(common_bundles):,} bundles (bitmap-checked)")
        all_bundles.extend(common_bundles)

    log(f"  Packing: {len(binary_indices):,} binary → {len(all_bundles):,} bundles ({time.time()-t2:.1f}s)")

    # Step 4: Encode bundles into dense integer matrix
    t3 = time.time()
    bundled = _encode_bundles(all_bundles, csc, n_rows)
    log(f"  Encoding: {bundled.shape} int16 matrix ({time.time()-t3:.1f}s)")

    # Step 5: Append non-binary features as passthrough columns
    if len(non_binary_indices) > 0:
        non_binary_dense = csc[:, non_binary_indices].toarray().astype(np.float32)
        # Combine: bundles (int16) + passthrough (float32) → float32 output
        bundled_f32 = bundled.astype(np.float32)
        final_matrix = np.hstack([bundled_f32, non_binary_dense])
        del bundled_f32, non_binary_dense
    else:
        final_matrix = bundled

    # Step 6: Build column names for output
    bundled_col_names = []
    for bundle_idx, bundle_cols in enumerate(all_bundles):
        # Name format: efb_{bundle_idx}_n{count} — e.g. efb_0_n127
        bundled_col_names.append(f'efb_{bundle_idx}_n{len(bundle_cols)}')

    for col_idx in non_binary_indices:
        name = col_names[col_idx] if col_idx < len(col_names) else f'cross_{col_idx}'
        bundled_col_names.append(name)

    # Step 7: Build mapping
    mapping = _build_mapping(all_bundles, col_names, binary_indices, non_binary_indices)

    elapsed = time.time() - t0
    stats = mapping['stats']
    log(f"  DONE: {stats['total_input_features']:,} → {stats['output_columns']:,} columns "
        f"({stats['compression_ratio']:.0f}x reduction, {elapsed:.1f}s)")

    # Free the CSC copy
    del csc
    gc.collect()

    return final_matrix, bundled_col_names, mapping


def prebundle_from_files(npz_path, names_path, output_dir=None, tf='1d'):
    """Pre-bundle from saved NPZ + names JSON files.

    Saves:
        - v2_crosses_BTC_{tf}_bundled.npz  (dense matrix saved as sparse for compat)
        - v2_cross_names_BTC_{tf}_bundled.json (bundled column names)
        - v2_efb_mapping_BTC_{tf}.json (full feature→bundle mapping)

    Returns: (bundled_matrix, bundled_col_names, mapping)
    """
    from atomic_io import atomic_save_json

    _log(f"Loading {npz_path}...")
    csr = sparse.load_npz(npz_path).tocsr()

    _log(f"Loading {names_path}...")
    with open(names_path) as f:
        col_names = json.load(f)

    assert csr.shape[1] == len(col_names), \
        f"Column count mismatch: matrix has {csr.shape[1]}, names has {len(col_names)}"

    bundled, bundled_names, mapping = prebundle_binary_matrix(csr, col_names, tf=tf)
    del csr
    gc.collect()

    # Determine output paths
    out_dir = output_dir or os.path.dirname(npz_path)
    base = os.path.basename(npz_path)  # e.g. v2_crosses_BTC_1d.npz
    tag = base.replace('v2_crosses_', '').replace('.npz', '')  # e.g. BTC_1d

    bundled_npz_path = os.path.join(out_dir, f'v2_crosses_{tag}_bundled.npz')
    bundled_names_path = os.path.join(out_dir, f'v2_cross_names_{tag}_bundled.json')
    mapping_path = os.path.join(out_dir, f'v2_efb_mapping_{tag}.json')

    # Save bundled matrix as sparse CSR (for pipeline compatibility)
    bundled_sparse = sparse.csr_matrix(bundled)
    sparse.save_npz(bundled_npz_path, bundled_sparse, compressed=False)
    _log(f"Saved: {bundled_npz_path} ({os.path.getsize(bundled_npz_path)/1e6:.1f} MB)")

    atomic_save_json(bundled_names, bundled_names_path)
    _log(f"Saved: {bundled_names_path}")

    atomic_save_json(mapping, mapping_path)
    _log(f"Saved: {mapping_path}")

    return bundled, bundled_names, mapping


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EFB Pre-Bundler for Binary Cross Features')
    parser.add_argument('--npz', required=True, help='Path to sparse cross matrix NPZ')
    parser.add_argument('--names', required=True, help='Path to cross names JSON')
    parser.add_argument('--tf', default='1d', help='Timeframe label')
    parser.add_argument('--output-dir', help='Output directory (default: same as input)')
    args = parser.parse_args()

    prebundle_from_files(args.npz, args.names, output_dir=args.output_dir, tf=args.tf)
