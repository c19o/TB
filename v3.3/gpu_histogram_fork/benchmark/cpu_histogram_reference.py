"""
CPU reference implementation for LightGBM histogram building.

Builds histograms from sparse CSR the same way LightGBM does internally,
serving as the ground-truth reference for GPU kernel validation.

Matrix Thesis Context:
  - Binary cross features (0/1), ~3-6M raw features
  - EFB bundles mutually exclusive features into ~12-23K bundles (max_bin=255)
  - Histogram = per-bundle-bin gradient/hessian sums
  - For binary features: bin 0 = feature OFF, bin 1 = feature ON
  - We NEVER filter features — the model decides via tree splits

LightGBM histogram inner loop (from serial_tree_learner.cpp):
  For each row in the leaf:
    1. Look up gradient and hessian for that row
    2. Walk CSR nonzero entries for that row (indptr[row]:indptr[row+1])
    3. For each nonzero: data[j] is the pre-computed EFB bundle bin index
    4. Accumulate: hist[data[j]].grad += gradient, hist[data[j]].hess += hessian

Histogram layout: float64, shape (num_bins, 2) — interleaved [grad_sum, hess_sum] per bin.
"""

import time
import numpy as np
import numba
from numba import njit, prange
from scipy import sparse as sp_sparse


# ---------------------------------------------------------------------------
# Numba JIT inner kernels (these ARE the reference — just fast)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _build_hist_inner(indptr, indices, data, grad, hess, row_indices, hist):
    """
    Core histogram accumulation loop — single leaf.

    Walks CSR rows specified by row_indices. For each nonzero entry,
    data[j] holds the pre-computed EFB bundle bin index. Accumulates
    gradient and hessian into hist[bin_idx, 0] and hist[bin_idx, 1].

    Parameters
    ----------
    indptr : int64 array, shape (n_rows + 1,)
    indices : int32 array, shape (nnz,)  — column indices (unused in bundled mode)
    data : uint8/int32 array, shape (nnz,) — EFB bin indices
    grad : float64 array, shape (n_rows,)
    hess : float64 array, shape (n_rows,)
    row_indices : int32/int64 array — rows belonging to this leaf
    hist : float64 array, shape (num_bins, 2) — output, zeroed on entry
    """
    for i in range(row_indices.shape[0]):
        row = row_indices[i]
        g = grad[row]
        h = hess[row]
        start = indptr[row]
        end = indptr[row + 1]
        for j in range(start, end):
            bin_idx = data[j]
            hist[bin_idx, 0] += g
            hist[bin_idx, 1] += h


@njit(cache=True)
def _build_hist_inner_efb(indptr, indices, data, grad, hess, row_indices,
                          col_to_bundle, col_to_bin, hist):
    """
    Histogram accumulation with EFB mapping from raw column indices.

    When CSR stores raw (unbundled) column indices, we map through
    col_to_bundle and col_to_bin to get the EFB-compressed bin index.
    This is what LightGBM does when building histograms from unbundled data.

    The final bin index into hist is:
        bundle_offset + col_to_bin[col]
    where bundle_offset is the cumulative bin count for bundles before this one.

    Parameters
    ----------
    indptr : int64 array, shape (n_rows + 1,)
    indices : int32 array, shape (nnz,) — raw column indices
    data : float64/uint8 array, shape (nnz,) — raw feature values (unused for binary)
    grad : float64 array, shape (n_rows,)
    hess : float64 array, shape (n_rows,)
    row_indices : int32/int64 array
    col_to_bundle : int32 array, shape (n_features,) — maps column -> bundle id
    col_to_bin : int32 array, shape (n_features,) — maps column -> bin within bundle
    hist : float64 array, shape (total_bins, 2) — output, zeroed on entry
    """
    for i in range(row_indices.shape[0]):
        row = row_indices[i]
        g = grad[row]
        h = hess[row]
        start = indptr[row]
        end = indptr[row + 1]
        for j in range(start, end):
            col = indices[j]
            bundle_id = col_to_bundle[col]
            bin_in_bundle = col_to_bin[col]
            # For binary features: bin_in_bundle is 0 or 1
            # The actual hist index = bundle_start_bin + bin_in_bundle
            # We store bundle_id * max_bin + bin_in_bundle for simplicity
            # But the caller pre-computes flat bin offsets:
            #   flat_idx = bundle_offsets[bundle_id] + bin_in_bundle
            # Here we assume col_to_bin already encodes the flat index:
            hist_idx = col_to_bin[col]
            hist[hist_idx, 0] += g
            hist[hist_idx, 1] += h


@njit(parallel=True, cache=True)
def _build_hist_all_leaves(indptr, indices, data, grad, hess,
                           leaf_assignment, num_leaves, num_bins):
    """
    Build histograms for ALL leaves in one pass (parallel over leaves).

    Parameters
    ----------
    indptr : int64 array, shape (n_rows + 1,)
    indices : int32 array, shape (nnz,)
    data : uint8/int32 array, shape (nnz,) — EFB bin indices
    grad : float64 array, shape (n_rows,)
    hess : float64 array, shape (n_rows,)
    leaf_assignment : int32 array, shape (n_rows,) — which leaf each row belongs to
    num_leaves : int
    num_bins : int

    Returns
    -------
    hists : float64 array, shape (num_leaves, num_bins, 2)
    """
    n_rows = grad.shape[0]
    hists = np.zeros((num_leaves, num_bins, 2), dtype=np.float64)

    # First pass: count rows per leaf (for pre-allocation)
    leaf_counts = np.zeros(num_leaves, dtype=np.int64)
    for row in range(n_rows):
        leaf = leaf_assignment[row]
        if 0 <= leaf < num_leaves:
            leaf_counts[leaf] += 1

    # Build per-leaf row index arrays
    leaf_offsets = np.zeros(num_leaves, dtype=np.int64)
    leaf_rows = np.empty(n_rows, dtype=np.int64)
    leaf_starts = np.zeros(num_leaves, dtype=np.int64)

    # Compute starts
    cumsum = np.int64(0)
    for lf in range(num_leaves):
        leaf_starts[lf] = cumsum
        cumsum += leaf_counts[lf]

    # Fill row indices per leaf
    leaf_offsets[:] = leaf_starts[:]
    for row in range(n_rows):
        leaf = leaf_assignment[row]
        if 0 <= leaf < num_leaves:
            pos = leaf_offsets[leaf]
            leaf_rows[pos] = row
            leaf_offsets[leaf] = pos + 1

    # Parallel histogram build over leaves
    for lf in prange(num_leaves):
        start_idx = leaf_starts[lf]
        count = leaf_counts[lf]
        for i in range(count):
            row = leaf_rows[start_idx + i]
            g = grad[row]
            h = hess[row]
            csr_start = indptr[row]
            csr_end = indptr[row + 1]
            for j in range(csr_start, csr_end):
                bin_idx = data[j]
                hists[lf, bin_idx, 0] += g
                hists[lf, bin_idx, 1] += h

    return hists


# ---------------------------------------------------------------------------
# Public API — Python wrappers with validation + timing
# ---------------------------------------------------------------------------

def build_histogram_cpu(csr_matrix, gradients, hessians, row_indices, num_bins):
    """
    Build histogram for a single leaf from sparse CSR data.

    This replicates LightGBM's SerialTreeLearner::ConstructHistograms()
    for one leaf node. For each row in row_indices, walks the CSR nonzero
    entries and accumulates gradient/hessian into the histogram bin specified
    by data[j] (the pre-computed EFB bundle bin index).

    Parameters
    ----------
    csr_matrix : scipy.sparse.csr_matrix or csr_array
        Shape (n_rows, n_features). data array contains EFB bin indices (uint8).
    gradients : np.ndarray, shape (n_rows,), dtype float64
        Per-row gradient from the objective function.
    hessians : np.ndarray, shape (n_rows,), dtype float64
        Per-row hessian from the objective function.
    row_indices : np.ndarray, dtype int32 or int64
        Rows belonging to this leaf node.
    num_bins : int
        Total number of histogram bins (sum of bins across all EFB bundles).

    Returns
    -------
    hist : np.ndarray, shape (num_bins, 2), dtype float64
        hist[b, 0] = sum of gradients for bin b
        hist[b, 1] = sum of hessians for bin b
    elapsed : float
        Wall-clock seconds for the histogram build (excludes validation).
    """
    # --- Validation ---
    if not sp_sparse.issparse(csr_matrix):
        raise TypeError(f"Expected sparse CSR matrix, got {type(csr_matrix)}")
    csr = csr_matrix.tocsr()

    n_rows = csr.shape[0]
    if gradients.shape[0] != n_rows:
        raise ValueError(f"gradients length {gradients.shape[0]} != matrix rows {n_rows}")
    if hessians.shape[0] != n_rows:
        raise ValueError(f"hessians length {hessians.shape[0]} != matrix rows {n_rows}")
    if row_indices.max() >= n_rows or row_indices.min() < 0:
        raise ValueError(f"row_indices out of range [0, {n_rows})")

    # Ensure correct dtypes for Numba
    indptr = np.ascontiguousarray(csr.indptr, dtype=np.int64)
    indices = np.ascontiguousarray(csr.indices, dtype=np.int32)
    data = np.ascontiguousarray(csr.data, dtype=np.int32)
    grad = np.ascontiguousarray(gradients, dtype=np.float64)
    hess = np.ascontiguousarray(hessians, dtype=np.float64)
    rows = np.ascontiguousarray(row_indices, dtype=np.int64)

    hist = np.zeros((num_bins, 2), dtype=np.float64)

    # --- Timed section ---
    t0 = time.perf_counter()
    _build_hist_inner(indptr, indices, data, grad, hess, rows, hist)
    elapsed = time.perf_counter() - t0

    return hist, elapsed


def build_histogram_cpu_all_leaves(csr_matrix, gradients, hessians,
                                   leaf_assignment, num_leaves, num_bins):
    """
    Build histograms for ALL leaves in one pass.

    Parallel over leaves using Numba prange. This matches LightGBM's approach
    of building multiple leaf histograms per tree level.

    Parameters
    ----------
    csr_matrix : scipy.sparse.csr_matrix
        Shape (n_rows, n_features). data = EFB bin indices.
    gradients : np.ndarray, shape (n_rows,), dtype float64
    hessians : np.ndarray, shape (n_rows,), dtype float64
    leaf_assignment : np.ndarray, shape (n_rows,), dtype int32
        Which leaf each row belongs to. Values in [0, num_leaves).
        Use -1 for rows not in any active leaf.
    num_leaves : int
    num_bins : int

    Returns
    -------
    hists : np.ndarray, shape (num_leaves, num_bins, 2), dtype float64
    elapsed : float
        Wall-clock seconds.
    """
    csr = csr_matrix.tocsr()
    n_rows = csr.shape[0]

    if gradients.shape[0] != n_rows:
        raise ValueError(f"gradients length {gradients.shape[0]} != matrix rows {n_rows}")
    if hessians.shape[0] != n_rows:
        raise ValueError(f"hessians length {hessians.shape[0]} != matrix rows {n_rows}")
    if leaf_assignment.shape[0] != n_rows:
        raise ValueError(f"leaf_assignment length {leaf_assignment.shape[0]} != matrix rows {n_rows}")

    indptr = np.ascontiguousarray(csr.indptr, dtype=np.int64)
    indices = np.ascontiguousarray(csr.indices, dtype=np.int32)
    data = np.ascontiguousarray(csr.data, dtype=np.int32)
    grad = np.ascontiguousarray(gradients, dtype=np.float64)
    hess = np.ascontiguousarray(hessians, dtype=np.float64)
    leaves = np.ascontiguousarray(leaf_assignment, dtype=np.int32)

    t0 = time.perf_counter()
    hists = _build_hist_all_leaves(indptr, indices, data, grad, hess,
                                   leaves, num_leaves, num_bins)
    elapsed = time.perf_counter() - t0

    return hists, elapsed


def build_histogram_with_efb(csr_matrix, gradients, hessians, row_indices,
                             col_to_bundle, col_to_bin, num_bundles, max_bin=2):
    """
    Build histogram with EFB mapping from raw (unbundled) CSR data.

    This is what LightGBM does when the Dataset has NOT been pre-bundled:
    raw column indices are mapped through EFB tables to get the final
    histogram bin index.

    For binary cross features with max_bin=2:
      - Each bundle has 2 bins: bin 0 (OFF) and bin 1 (ON)
      - Total histogram bins = num_bundles * max_bin
      - Flat bin index = bundle_offsets[bundle_id] + bin_in_bundle

    Parameters
    ----------
    csr_matrix : scipy.sparse.csr_matrix
        Raw unbundled CSR. data values are feature values (0/1 for binary).
    gradients : np.ndarray, shape (n_rows,), dtype float64
    hessians : np.ndarray, shape (n_rows,), dtype float64
    row_indices : np.ndarray, dtype int32/int64
        Rows in this leaf.
    col_to_bundle : np.ndarray, shape (n_features,), dtype int32
        Maps raw column index -> EFB bundle id.
    col_to_bin : np.ndarray, shape (n_features,), dtype int32
        Maps raw column index -> FLAT bin index in the histogram.
        Computed as: bundle_offsets[col_to_bundle[col]] + bin_within_bundle[col]
    num_bundles : int
        Number of EFB bundles.
    max_bin : int, default 2
        Max bins per bundle. For binary features = 2.

    Returns
    -------
    hist : np.ndarray, shape (total_bins, 2), dtype float64
        total_bins = num_bundles * max_bin
    elapsed : float
    """
    csr = csr_matrix.tocsr()
    n_rows = csr.shape[0]
    total_bins = num_bundles * max_bin

    if gradients.shape[0] != n_rows:
        raise ValueError(f"gradients length {gradients.shape[0]} != matrix rows {n_rows}")
    if col_to_bundle.shape[0] != csr.shape[1]:
        raise ValueError(f"col_to_bundle length {col_to_bundle.shape[0]} != matrix cols {csr.shape[1]}")
    if col_to_bin.shape[0] != csr.shape[1]:
        raise ValueError(f"col_to_bin length {col_to_bin.shape[0]} != matrix cols {csr.shape[1]}")

    indptr = np.ascontiguousarray(csr.indptr, dtype=np.int64)
    indices = np.ascontiguousarray(csr.indices, dtype=np.int32)
    data = np.ascontiguousarray(csr.data, dtype=np.float64)
    grad = np.ascontiguousarray(gradients, dtype=np.float64)
    hess = np.ascontiguousarray(hessians, dtype=np.float64)
    rows = np.ascontiguousarray(row_indices, dtype=np.int64)
    c2b = np.ascontiguousarray(col_to_bundle, dtype=np.int32)
    c2bin = np.ascontiguousarray(col_to_bin, dtype=np.int32)

    hist = np.zeros((total_bins, 2), dtype=np.float64)

    t0 = time.perf_counter()
    _build_hist_inner_efb(indptr, indices, data, grad, hess, rows,
                          c2b, c2bin, hist)
    elapsed = time.perf_counter() - t0

    return hist, elapsed


def build_histogram_subtraction(parent_hist, child_hist):
    """
    Compute sibling histogram via subtraction trick.

    LightGBM optimization: after splitting a node, only build the histogram
    for the smaller child. The larger child's histogram is:
        sibling = parent - smaller_child

    This halves the histogram computation per tree level.

    Parameters
    ----------
    parent_hist : np.ndarray, shape (num_bins, 2), dtype float64
    child_hist : np.ndarray, shape (num_bins, 2), dtype float64

    Returns
    -------
    sibling_hist : np.ndarray, shape (num_bins, 2), dtype float64
    elapsed : float
    """
    if parent_hist.shape != child_hist.shape:
        raise ValueError(
            f"Shape mismatch: parent {parent_hist.shape} vs child {child_hist.shape}"
        )
    if parent_hist.ndim != 2 or parent_hist.shape[1] != 2:
        raise ValueError(
            f"Expected shape (num_bins, 2), got {parent_hist.shape}"
        )

    t0 = time.perf_counter()
    sibling_hist = parent_hist - child_hist
    elapsed = time.perf_counter() - t0

    return sibling_hist, elapsed


# ---------------------------------------------------------------------------
# Synthetic data generators for benchmarking
# ---------------------------------------------------------------------------

def generate_synthetic_csr(n_rows, n_features, density=0.001, num_bins=256,
                           rng=None):
    """
    Generate a synthetic sparse CSR matrix mimicking EFB-bundled data.

    data values are random EFB bin indices in [0, num_bins).
    For binary features (our case), num_bins=2 and data is 0 or 1.

    Parameters
    ----------
    n_rows : int
    n_features : int
    density : float
        Fraction of nonzero entries. Binary crosses are typically 0.001-0.01.
    num_bins : int
        Range of bin indices stored in data array.
    rng : np.random.Generator or None

    Returns
    -------
    csr : scipy.sparse.csr_matrix
    """
    if rng is None:
        rng = np.random.default_rng(42)

    csr = sp_sparse.random(n_rows, n_features, density=density,
                           format='csr', dtype=np.float64,
                           random_state=rng.integers(0, 2**31))
    # Replace random values with EFB bin indices
    csr.data = rng.integers(0, num_bins, size=csr.data.shape[0]).astype(np.int32)
    # Eliminate any zeros that crept in (structural zeros != stored zeros)
    csr.eliminate_zeros()
    return csr


def generate_efb_mapping(n_features, num_bundles, max_bin=2, rng=None):
    """
    Generate synthetic EFB bundle mapping.

    Randomly assigns each raw feature to a bundle and a bin within that bundle.

    Parameters
    ----------
    n_features : int
    num_bundles : int
    max_bin : int
        Bins per bundle (2 for binary features).
    rng : np.random.Generator or None

    Returns
    -------
    col_to_bundle : np.ndarray, shape (n_features,), dtype int32
    col_to_bin : np.ndarray, shape (n_features,), dtype int32
        Flat bin index = bundle_id * max_bin + bin_within_bundle
    """
    if rng is None:
        rng = np.random.default_rng(42)

    col_to_bundle = rng.integers(0, num_bundles, size=n_features).astype(np.int32)
    bin_within_bundle = rng.integers(0, max_bin, size=n_features).astype(np.int32)
    # Flat bin index for direct histogram indexing
    col_to_bin = (col_to_bundle * max_bin + bin_within_bundle).astype(np.int32)

    return col_to_bundle, col_to_bin


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(n_rows=5733, n_features=23000, density=0.01, num_bins=256,
                  num_leaves=63, warmup=2, repeats=10):
    """
    Run the full benchmark suite.

    Default parameters match 1d BTC training:
      - 5,733 rows (daily bars 2010-2026)
      - 23,000 EFB bundles (from ~6M raw features with max_bin=255)
      - density 0.01 (~1.3M nonzeros)
      - 63 leaves (num_leaves from config)

    Parameters
    ----------
    n_rows : int
    n_features : int — number of EFB bundles (NOT raw features)
    density : float
    num_bins : int — total histogram bins
    num_leaves : int
    warmup : int — JIT warmup iterations (not timed)
    repeats : int — timed iterations

    Returns
    -------
    results : dict — timing results
    """
    rng = np.random.default_rng(42)
    print(f"=" * 70)
    print(f"CPU Histogram Reference Benchmark")
    print(f"=" * 70)
    print(f"  Rows:         {n_rows:,}")
    print(f"  EFB bundles:  {n_features:,}")
    print(f"  Density:      {density}")
    print(f"  Num bins:     {num_bins}")
    print(f"  Num leaves:   {num_leaves}")
    print(f"  Warmup:       {warmup}")
    print(f"  Repeats:      {repeats}")

    # Generate data
    print(f"\nGenerating synthetic CSR...")
    t0 = time.perf_counter()
    csr = generate_synthetic_csr(n_rows, n_features, density, num_bins, rng)
    nnz = csr.nnz
    print(f"  CSR: {csr.shape}, nnz={nnz:,} ({nnz / (n_rows * n_features) * 100:.2f}%)")
    print(f"  Data size: {csr.data.nbytes / 1024**2:.1f} MB")
    print(f"  Generated in {time.perf_counter() - t0:.3f}s")

    gradients = rng.standard_normal(n_rows).astype(np.float64)
    hessians = np.abs(rng.standard_normal(n_rows)).astype(np.float64) + 0.01

    # Split rows into leaves
    leaf_assignment = rng.integers(0, num_leaves, size=n_rows).astype(np.int32)
    leaf_0_rows = np.where(leaf_assignment == 0)[0].astype(np.int64)
    all_rows = np.arange(n_rows, dtype=np.int64)

    results = {}

    # --- 1. Single leaf histogram ---
    print(f"\n--- 1. Single leaf histogram (leaf 0, {leaf_0_rows.shape[0]} rows) ---")
    # Warmup (triggers Numba JIT compilation)
    for _ in range(warmup):
        build_histogram_cpu(csr, gradients, hessians, leaf_0_rows, num_bins)

    times = []
    for _ in range(repeats):
        _, elapsed = build_histogram_cpu(csr, gradients, hessians,
                                         leaf_0_rows, num_bins)
        times.append(elapsed)
    avg = np.mean(times)
    std = np.std(times)
    print(f"  Avg: {avg * 1000:.3f} ms  (std: {std * 1000:.3f} ms)")
    print(f"  Throughput: {nnz * leaf_0_rows.shape[0] / n_rows / avg / 1e9:.2f} G entries/s")
    results['single_leaf_ms'] = avg * 1000

    # --- 2. Full-tree histogram (all rows, one leaf = root) ---
    print(f"\n--- 2. Root histogram (all {n_rows} rows) ---")
    for _ in range(warmup):
        build_histogram_cpu(csr, gradients, hessians, all_rows, num_bins)

    times = []
    for _ in range(repeats):
        _, elapsed = build_histogram_cpu(csr, gradients, hessians,
                                         all_rows, num_bins)
        times.append(elapsed)
    avg = np.mean(times)
    std = np.std(times)
    print(f"  Avg: {avg * 1000:.3f} ms  (std: {std * 1000:.3f} ms)")
    print(f"  Throughput: {nnz / avg / 1e9:.2f} G entries/s")
    results['root_hist_ms'] = avg * 1000

    # --- 3. All-leaves parallel histogram ---
    print(f"\n--- 3. All-leaves parallel ({num_leaves} leaves) ---")
    for _ in range(warmup):
        build_histogram_cpu_all_leaves(csr, gradients, hessians,
                                      leaf_assignment, num_leaves, num_bins)

    times = []
    for _ in range(repeats):
        _, elapsed = build_histogram_cpu_all_leaves(csr, gradients, hessians,
                                                    leaf_assignment,
                                                    num_leaves, num_bins)
        times.append(elapsed)
    avg = np.mean(times)
    std = np.std(times)
    print(f"  Avg: {avg * 1000:.3f} ms  (std: {std * 1000:.3f} ms)")
    results['all_leaves_ms'] = avg * 1000

    # --- 4. Histogram subtraction ---
    print(f"\n--- 4. Histogram subtraction ---")
    parent = rng.standard_normal((num_bins, 2)).astype(np.float64)
    child = rng.standard_normal((num_bins, 2)).astype(np.float64)

    times = []
    for _ in range(repeats):
        _, elapsed = build_histogram_subtraction(parent, child)
        times.append(elapsed)
    avg = np.mean(times)
    print(f"  Avg: {avg * 1e6:.1f} us")
    results['subtraction_us'] = avg * 1e6

    # --- 5. EFB-mapped histogram ---
    print(f"\n--- 5. EFB-mapped histogram (raw columns -> bundles) ---")
    n_raw_features = 100_000  # Subset of raw features for benchmark
    n_bundles = 500
    max_bin = 2
    total_efb_bins = n_bundles * max_bin

    csr_raw = generate_synthetic_csr(n_rows, n_raw_features, density=0.005,
                                     num_bins=2, rng=rng)
    col_to_bundle, col_to_bin = generate_efb_mapping(n_raw_features, n_bundles,
                                                     max_bin, rng)

    for _ in range(warmup):
        build_histogram_with_efb(csr_raw, gradients, hessians, all_rows,
                                 col_to_bundle, col_to_bin, n_bundles, max_bin)

    times = []
    for _ in range(repeats):
        _, elapsed = build_histogram_with_efb(csr_raw, gradients, hessians,
                                              all_rows, col_to_bundle,
                                              col_to_bin, n_bundles, max_bin)
        times.append(elapsed)
    avg = np.mean(times)
    std = np.std(times)
    print(f"  Raw features: {n_raw_features:,}, bundles: {n_bundles}, "
          f"max_bin: {max_bin}")
    print(f"  Avg: {avg * 1000:.3f} ms  (std: {std * 1000:.3f} ms)")
    results['efb_mapped_ms'] = avg * 1000

    # --- 6. Correctness validation ---
    print(f"\n--- 6. Correctness validation ---")
    _validate_correctness(rng)

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")
    for key, val in results.items():
        unit = 'us' if 'us' in key else 'ms'
        print(f"  {key:30s}: {val:10.3f} {unit}")

    # Estimate full-tree cost (LightGBM context)
    # 800 rounds x 3 classes x ~31 hist builds/tree = ~74,400 histogram ops
    hist_ops = 800 * 3 * 31
    single_leaf_s = results['single_leaf_ms'] / 1000
    estimated_total = hist_ops * single_leaf_s
    print(f"\n  Estimated full training histogram cost:")
    print(f"    {hist_ops:,} histogram ops x {single_leaf_s * 1000:.3f} ms "
          f"= {estimated_total:.1f}s ({estimated_total / 60:.1f} min)")
    print(f"    (This is what the GPU kernel replaces)")

    return results


def _validate_correctness(rng):
    """
    Validate histogram implementations against brute-force reference.
    """
    n_rows, n_cols, num_bins = 100, 50, 8
    density = 0.2

    csr = generate_synthetic_csr(n_rows, n_cols, density, num_bins, rng)
    grad = rng.standard_normal(n_rows).astype(np.float64)
    hess = np.abs(rng.standard_normal(n_rows)).astype(np.float64) + 0.01
    rows = np.arange(n_rows, dtype=np.int64)

    # --- Brute force reference (pure Python, no Numba) ---
    expected = np.zeros((num_bins, 2), dtype=np.float64)
    csr_coo = csr.tocsr()
    for row in range(n_rows):
        start = csr_coo.indptr[row]
        end = csr_coo.indptr[row + 1]
        for j in range(start, end):
            b = int(csr_coo.data[j])
            expected[b, 0] += grad[row]
            expected[b, 1] += hess[row]

    # Test single-leaf
    hist, _ = build_histogram_cpu(csr, grad, hess, rows, num_bins)
    assert np.allclose(hist, expected, atol=1e-10), \
        f"Single-leaf mismatch! Max diff: {np.abs(hist - expected).max()}"
    print("  [PASS] Single-leaf histogram matches brute-force reference")

    # Test all-leaves (with all rows in leaf 0)
    leaf_assign = np.zeros(n_rows, dtype=np.int32)
    hists, _ = build_histogram_cpu_all_leaves(csr, grad, hess,
                                              leaf_assign, 1, num_bins)
    assert np.allclose(hists[0], expected, atol=1e-10), \
        f"All-leaves mismatch! Max diff: {np.abs(hists[0] - expected).max()}"
    print("  [PASS] All-leaves histogram matches brute-force reference")

    # Test subtraction
    parent = expected.copy()
    child = expected * 0.4
    sibling, _ = build_histogram_subtraction(parent, child)
    expected_sibling = parent - child
    assert np.allclose(sibling, expected_sibling, atol=1e-15), \
        "Subtraction mismatch!"
    print("  [PASS] Histogram subtraction correct")

    # Test EFB mapping
    n_bundles = 5
    max_bin = 2
    total_bins = n_bundles * max_bin
    col_to_bundle, col_to_bin = generate_efb_mapping(n_cols, n_bundles,
                                                     max_bin, rng)
    hist_efb, _ = build_histogram_with_efb(csr, grad, hess, rows,
                                           col_to_bundle, col_to_bin,
                                           n_bundles, max_bin)

    # Brute-force EFB reference
    expected_efb = np.zeros((total_bins, 2), dtype=np.float64)
    for row in range(n_rows):
        start = csr_coo.indptr[row]
        end = csr_coo.indptr[row + 1]
        for j in range(start, end):
            col = csr_coo.indices[j]
            flat_bin = int(col_to_bin[col])
            expected_efb[flat_bin, 0] += grad[row]
            expected_efb[flat_bin, 1] += hess[row]

    assert np.allclose(hist_efb, expected_efb, atol=1e-10), \
        f"EFB mismatch! Max diff: {np.abs(hist_efb - expected_efb).max()}"
    print("  [PASS] EFB-mapped histogram matches brute-force reference")

    print("  [ALL PASSED] All correctness checks passed")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='CPU reference histogram benchmark for GPU validation')
    parser.add_argument('--rows', type=int, default=5733,
                        help='Number of rows (default: 5733 = 1d BTC)')
    parser.add_argument('--features', type=int, default=23000,
                        help='Number of EFB bundles (default: 23000)')
    parser.add_argument('--density', type=float, default=0.01,
                        help='CSR density (default: 0.01)')
    parser.add_argument('--bins', type=int, default=256,
                        help='Number of histogram bins (default: 256)')
    parser.add_argument('--leaves', type=int, default=63,
                        help='Number of leaves (default: 63)')
    parser.add_argument('--warmup', type=int, default=2,
                        help='JIT warmup iterations (default: 2)')
    parser.add_argument('--repeats', type=int, default=10,
                        help='Timed iterations (default: 10)')
    parser.add_argument('--validate-only', action='store_true',
                        help='Run only correctness validation, skip benchmark')

    args = parser.parse_args()

    if args.validate_only:
        print("Running correctness validation only...")
        _validate_correctness(np.random.default_rng(42))
    else:
        run_benchmark(
            n_rows=args.rows,
            n_features=args.features,
            density=args.density,
            num_bins=args.bins,
            num_leaves=args.leaves,
            warmup=args.warmup,
            repeats=args.repeats,
        )
