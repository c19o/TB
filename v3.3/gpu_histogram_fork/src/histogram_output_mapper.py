"""
Histogram Output Mapper — GPU histogram to LightGBM internal format.

Converts GPU SpMV histogram output (per-feature grad/hess sums for bin=1 only)
into LightGBM's expected interleaved flat array layout.

LightGBM's internal histogram format:
  - Flat double array, interleaved: [grad_bin0, hess_bin0, grad_bin1, hess_bin1, ...]
  - Per-feature offsets stored in feature_hist_offsets_ (cumulative bin counts)
  - Bin numbering starts at 1; bin 0 is reserved for missing/most_freq_bin sentinel
  - The hist_t* pointer is back-shifted by kHistOffset so that
    hist_ptr[bin_value] indexes directly without subtraction

Our GPU SpMV produces:
  - Per-feature gradient sums: one scalar per feature = sum of gradients
    for rows where feature=1 (sparse CSR nonzero entries)
  - Per-feature hessian sums: same
  - These correspond to bin=1 ONLY (feature ON)
  - bin=0 (feature OFF) = leaf_total - bin_1

For binary cross features (max_bin=2, the only case in our system):
  - Each feature has exactly 2 bins: bin 0 (OFF) and bin 1 (ON)
  - feature_hist_offsets_ = [0, 2, 4, 6, ...] (each feature contributes 2 bins)
  - The flat array has 2 * n_features entries (interleaved grad/hess per bin)
  - Total array size = n_features * 2 bins * 2 doubles (grad+hess) = n_features * 4

Correctness invariant:
  For every feature f:
    hist[offset(f) + 0].grad + hist[offset(f) + 1].grad == leaf_total_grad
    hist[offset(f) + 0].hess + hist[offset(f) + 1].hess == leaf_total_hess
  This is guaranteed by construction (bin0 = total - bin1).

All accumulation in float64 to match LightGBM's internal precision.
"""

import numpy as np
from typing import Optional


# LightGBM interleaves grad/hess as pairs of doubles.
# For a single bin entry, the layout is: [gradient, hessian]
# Index 0 = gradient, index 1 = hessian within each bin's pair.
_GRAD_IDX = 0
_HESS_IDX = 1
_PAIR_SIZE = 2  # doubles per bin (grad + hess)


def gpu_hist_to_lgbm_format(
    gpu_grad_sums: np.ndarray,
    gpu_hess_sums: np.ndarray,
    leaf_total_grad: float,
    leaf_total_hess: float,
    feature_hist_offsets: np.ndarray,
    num_total_bins: int,
) -> np.ndarray:
    """
    Convert GPU histogram output to LightGBM's expected interleaved format.

    GPU output: per-feature grad/hess sums (for bin=1 only)
    LightGBM expects: per-bin grad/hess for all bins (including bin=0)

    For binary features with EFB:
    - Each feature has 2 bins: bin=0 (OFF), bin=1 (ON)
    - bin=1 grad/hess = GPU output directly
    - bin=0 grad/hess = leaf_total - bin=1

    Parameters
    ----------
    gpu_grad_sums : np.ndarray, shape (n_features,), float64
        Per-feature gradient sums from GPU SpMV. Each entry is the sum of
        gradients for rows where that feature's value = 1.
    gpu_hess_sums : np.ndarray, shape (n_features,), float64
        Per-feature hessian sums from GPU SpMV. Same layout as gpu_grad_sums.
    leaf_total_grad : float
        Sum of all gradients for rows in this leaf. float64.
    leaf_total_hess : float
        Sum of all hessians for rows in this leaf. float64.
    feature_hist_offsets : np.ndarray, shape (n_features + 1,), int32 or int64
        Cumulative bin counts per feature. feature_hist_offsets[f] is the
        starting bin index for feature f. feature_hist_offsets[n_features]
        = num_total_bins. For binary features: [0, 2, 4, 6, ...].
    num_total_bins : int
        Total number of bins across all features. For n binary features
        with 2 bins each: num_total_bins = 2 * n_features.

    Returns
    -------
    hist : np.ndarray, shape (num_total_bins * 2,), float64
        Interleaved [grad_bin0, hess_bin0, grad_bin1, hess_bin1, ...]
        for each feature in sequence. This is the exact layout that
        LightGBM's split finder reads from hist_t*.

    Raises
    ------
    ValueError
        If array dimensions are inconsistent or inputs are invalid.
    """
    n_features = len(gpu_grad_sums)

    # --- Input validation ---
    if len(gpu_hess_sums) != n_features:
        raise ValueError(
            f"gpu_grad_sums length ({n_features}) != "
            f"gpu_hess_sums length ({len(gpu_hess_sums)})"
        )
    if len(feature_hist_offsets) != n_features + 1:
        raise ValueError(
            f"feature_hist_offsets length ({len(feature_hist_offsets)}) != "
            f"n_features + 1 ({n_features + 1})"
        )
    if feature_hist_offsets[-1] != num_total_bins:
        raise ValueError(
            f"feature_hist_offsets[-1] ({feature_hist_offsets[-1]}) != "
            f"num_total_bins ({num_total_bins})"
        )
    if n_features == 0:
        return np.zeros(0, dtype=np.float64)

    # Ensure float64 precision
    gpu_grad_sums = np.asarray(gpu_grad_sums, dtype=np.float64)
    gpu_hess_sums = np.asarray(gpu_hess_sums, dtype=np.float64)
    leaf_total_grad = np.float64(leaf_total_grad)
    leaf_total_hess = np.float64(leaf_total_hess)

    # Allocate output: interleaved pairs for each bin
    hist = np.zeros(num_total_bins * _PAIR_SIZE, dtype=np.float64)

    # --- Vectorized path for uniform binary features (2 bins each) ---
    # Check if all features have exactly 2 bins (the common case for our system)
    bins_per_feature = np.diff(feature_hist_offsets)
    all_binary = np.all(bins_per_feature == 2)

    if all_binary:
        # Fast vectorized path: all features are binary (2 bins)
        # Layout: feature f occupies hist[f*4 : f*4 + 4]
        #   hist[f*4 + 0] = grad_bin0  (feature OFF)
        #   hist[f*4 + 1] = hess_bin0
        #   hist[f*4 + 2] = grad_bin1  (feature ON)
        #   hist[f*4 + 3] = hess_bin1

        # bin 1 (feature ON) = GPU output
        hist[2::4] = gpu_grad_sums      # grad_bin1 at positions 2, 6, 10, ...
        hist[3::4] = gpu_hess_sums      # hess_bin1 at positions 3, 7, 11, ...

        # bin 0 (feature OFF) = total - bin1
        hist[0::4] = leaf_total_grad - gpu_grad_sums   # grad_bin0 at 0, 4, 8, ...
        hist[1::4] = leaf_total_hess - gpu_hess_sums   # hess_bin0 at 1, 5, 9, ...

    else:
        # General path: features may have different bin counts (EFB bundles)
        # For non-binary features, we only populate bin 0 and bin 1.
        # Bins 2..N-1 are left as zero (GPU SpMV only computes bin=1 sums).
        for f in range(n_features):
            offset = int(feature_hist_offsets[f])
            n_bins_f = int(bins_per_feature[f])

            if n_bins_f < 2:
                # Degenerate: single-bin feature (all same value). Skip.
                continue

            # bin 0 (OFF): total - bin1
            base = offset * _PAIR_SIZE
            hist[base + _GRAD_IDX] = leaf_total_grad - gpu_grad_sums[f]
            hist[base + _HESS_IDX] = leaf_total_hess - gpu_hess_sums[f]

            # bin 1 (ON): GPU output
            hist[base + _PAIR_SIZE + _GRAD_IDX] = gpu_grad_sums[f]
            hist[base + _PAIR_SIZE + _HESS_IDX] = gpu_hess_sums[f]

            # bins 2..N-1: zero (no data from GPU for these bins)
            # Already zero from np.zeros initialization.

    return hist


def gpu_hist_to_lgbm_format_batched(
    gpu_grad_sums: np.ndarray,
    gpu_hess_sums: np.ndarray,
    leaf_total_grad: float,
    leaf_total_hess: float,
    n_features: int,
) -> np.ndarray:
    """
    Fast path for uniform binary features — skip offset lookup entirely.

    When ALL features are binary (2 bins each), the offsets are trivially
    [0, 2, 4, ...] and we can skip the indirection. This is the common
    case for our system (all cross features are binary 0/1).

    Parameters
    ----------
    gpu_grad_sums : np.ndarray, shape (n_features,), float64
        Per-feature gradient sums from GPU (bin=1 only).
    gpu_hess_sums : np.ndarray, shape (n_features,), float64
        Per-feature hessian sums from GPU (bin=1 only).
    leaf_total_grad : float
        Total gradient for this leaf.
    leaf_total_hess : float
        Total hessian for this leaf.
    n_features : int
        Number of features (must match array lengths).

    Returns
    -------
    hist : np.ndarray, shape (n_features * 4,), float64
        Interleaved LightGBM format: [g0, h0, g1, h1] per feature.
    """
    if len(gpu_grad_sums) != n_features or len(gpu_hess_sums) != n_features:
        raise ValueError(
            f"Array length mismatch: gpu_grad_sums={len(gpu_grad_sums)}, "
            f"gpu_hess_sums={len(gpu_hess_sums)}, n_features={n_features}"
        )

    gpu_grad_sums = np.asarray(gpu_grad_sums, dtype=np.float64)
    gpu_hess_sums = np.asarray(gpu_hess_sums, dtype=np.float64)

    # Allocate: 2 bins * 2 doubles (grad+hess) per feature
    hist = np.empty(n_features * 4, dtype=np.float64)

    # bin 0 (OFF) = total - bin1
    hist[0::4] = np.float64(leaf_total_grad) - gpu_grad_sums
    hist[1::4] = np.float64(leaf_total_hess) - gpu_hess_sums

    # bin 1 (ON) = GPU output
    hist[2::4] = gpu_grad_sums
    hist[3::4] = gpu_hess_sums

    return hist


def lgbm_hist_to_per_feature(
    hist: np.ndarray,
    feature_hist_offsets: np.ndarray,
    n_features: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract per-feature (n_features, n_bins) arrays from LightGBM flat format.

    Useful for debugging and comparison with cpu_histogram_reference output.

    Parameters
    ----------
    hist : np.ndarray, shape (num_total_bins * 2,), float64
        LightGBM interleaved histogram.
    feature_hist_offsets : np.ndarray, shape (n_features + 1,)
        Cumulative bin offsets.
    n_features : int
        Number of features.

    Returns
    -------
    grad : np.ndarray, shape (n_features, max_bins_per_feature), float64
        Gradient histogram. Padded with 0 if features have different bin counts.
    hess : np.ndarray, shape (n_features, max_bins_per_feature), float64
        Hessian histogram.
    """
    bins_per_feature = np.diff(feature_hist_offsets)
    max_bins = int(np.max(bins_per_feature)) if n_features > 0 else 0

    grad = np.zeros((n_features, max_bins), dtype=np.float64)
    hess = np.zeros((n_features, max_bins), dtype=np.float64)

    for f in range(n_features):
        offset = int(feature_hist_offsets[f])
        n_bins_f = int(bins_per_feature[f])
        for b in range(n_bins_f):
            base = (offset + b) * _PAIR_SIZE
            grad[f, b] = hist[base + _GRAD_IDX]
            hess[f, b] = hist[base + _HESS_IDX]

    return grad, hess


def make_binary_feature_hist_offsets(n_features: int) -> np.ndarray:
    """
    Build feature_hist_offsets for n uniform binary features.

    Each feature has 2 bins, so offsets are [0, 2, 4, ..., 2*n_features].

    Parameters
    ----------
    n_features : int
        Number of binary features.

    Returns
    -------
    offsets : np.ndarray, shape (n_features + 1,), int32
    """
    return np.arange(n_features + 1, dtype=np.int32) * 2


def validate_histogram_invariant(
    hist: np.ndarray,
    feature_hist_offsets: np.ndarray,
    n_features: int,
    leaf_total_grad: float,
    leaf_total_hess: float,
    atol: float = 1e-10,
) -> bool:
    """
    Verify the bin0 + bin1 == total invariant for every binary feature.

    For binary features, sum of all bins for any feature must equal the
    leaf total. This is the fundamental correctness check.

    Parameters
    ----------
    hist : np.ndarray
        LightGBM interleaved histogram.
    feature_hist_offsets : np.ndarray
        Cumulative bin offsets.
    n_features : int
        Number of features.
    leaf_total_grad : float
        Expected total gradient.
    leaf_total_hess : float
        Expected total hessian.
    atol : float
        Absolute tolerance for comparison.

    Returns
    -------
    valid : bool
        True if invariant holds for all features.

    Raises
    ------
    AssertionError
        If any feature violates the invariant (with details).
    """
    bins_per_feature = np.diff(feature_hist_offsets)
    violations = []

    for f in range(n_features):
        offset = int(feature_hist_offsets[f])
        n_bins_f = int(bins_per_feature[f])

        sum_grad = 0.0
        sum_hess = 0.0
        for b in range(n_bins_f):
            base = (offset + b) * _PAIR_SIZE
            sum_grad += hist[base + _GRAD_IDX]
            sum_hess += hist[base + _HESS_IDX]

        if abs(sum_grad - leaf_total_grad) > atol:
            violations.append(
                f"feature {f}: grad sum {sum_grad:.15e} != "
                f"total {leaf_total_grad:.15e} (diff={abs(sum_grad - leaf_total_grad):.2e})"
            )
        if abs(sum_hess - leaf_total_hess) > atol:
            violations.append(
                f"feature {f}: hess sum {sum_hess:.15e} != "
                f"total {leaf_total_hess:.15e} (diff={abs(sum_hess - leaf_total_hess):.2e})"
            )

    if violations:
        raise AssertionError(
            f"Histogram invariant violated in {len(violations)} checks:\n"
            + "\n".join(violations[:20])
            + (f"\n... and {len(violations) - 20} more" if len(violations) > 20 else "")
        )

    return True
