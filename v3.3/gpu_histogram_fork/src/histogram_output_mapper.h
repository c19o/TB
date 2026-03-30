/*
 * histogram_output_mapper.h — Map GPU histogram output to LightGBM format.
 *
 * Converts GPU SpMV per-feature grad/hess sums (bin=1 only) into LightGBM's
 * expected interleaved histogram layout for the split finder.
 *
 * LightGBM's internal histogram format:
 *   - hist_t is double (when use_fp64=true, which we always set)
 *   - Interleaved pairs: [grad_bin0, hess_bin0, grad_bin1, hess_bin1, ...]
 *   - Per-feature offsets via feature_hist_offsets_ (cumulative bin counts)
 *   - Bin 0 = missing/most_freq_bin sentinel; bin 1+ = actual bin values
 *   - The hist pointer is back-shifted by kHistOffset so bin_value indexes
 *     directly into the array without subtraction
 *
 * GPU SpMV output:
 *   - gpu_grad[f] = sum of gradients for rows where feature f = 1
 *   - gpu_hess[f] = sum of hessians  for rows where feature f = 1
 *   - These correspond to bin=1 ONLY (feature ON in sparse CSR)
 *
 * For binary cross features (2 bins per feature):
 *   - bin 1 (ON)  = GPU output directly
 *   - bin 0 (OFF) = leaf_total - bin_1 (subtraction trick)
 *
 * Correctness invariant:
 *   For every feature f:
 *     sum(hist[offset(f)..offset(f+1)].grad) == leaf_grad_sum
 *     sum(hist[offset(f)..offset(f+1)].hess) == leaf_hess_sum
 *
 * Copyright (c) Savage22 Server Project. Licensed under MIT.
 */

#ifndef GPU_HISTOGRAM_OUTPUT_MAPPER_H_
#define GPU_HISTOGRAM_OUTPUT_MAPPER_H_

#include <cstdint>
#include <cstring>   /* memset */
#include <cmath>     /* fabs   */

#ifdef __cplusplus
extern "C" {
#endif

/* hist_t = double in LightGBM when compiled with USE_DOUBLE=ON.
 * We always use float64 for bit-exact matching with CPU LightGBM. */
typedef double hist_t;

/* Number of doubles per histogram bin entry (gradient + hessian). */
#define HIST_PAIR_SIZE 2

/* =========================================================================
 * MapGPUHistToLGBM — Main mapping function
 *
 * Converts GPU per-feature sums into LightGBM's interleaved flat array.
 *
 * Parameters:
 *   gpu_grad            [in]  Per-feature gradient sums from GPU, length n_features.
 *                             gpu_grad[f] = sum of gradients where feature f == 1.
 *   gpu_hess            [in]  Per-feature hessian sums from GPU, length n_features.
 *   leaf_grad_sum       [in]  Total gradient for all rows in this leaf.
 *   leaf_hess_sum       [in]  Total hessian for all rows in this leaf.
 *   feature_hist_offsets [in] Cumulative bin offsets, length n_features + 1.
 *                             feature_hist_offsets[f] = starting bin index for feature f.
 *                             feature_hist_offsets[n_features] = total bins.
 *                             For binary features: [0, 2, 4, ..., 2*n_features].
 *   n_features          [in]  Number of features.
 *   output_hist         [out] Caller-allocated buffer, length = total_bins * 2 doubles.
 *                             total_bins = feature_hist_offsets[n_features].
 *                             Filled with interleaved [grad, hess] per bin.
 *
 * Memory:
 *   output_hist must be pre-allocated by the caller. This function writes
 *   total_bins * 2 * sizeof(double) bytes. The function does NOT allocate
 *   or free memory.
 *
 * Thread safety:
 *   Safe to call from multiple threads if output_hist buffers do not overlap.
 *   The function is stateless and reads gpu_grad/gpu_hess/feature_hist_offsets
 *   without modification.
 * ========================================================================= */

static inline void MapGPUHistToLGBM(
    const double* gpu_grad,
    const double* gpu_hess,
    double leaf_grad_sum,
    double leaf_hess_sum,
    const int32_t* feature_hist_offsets,
    int32_t n_features,
    hist_t* output_hist)
{
    if (n_features <= 0) return;

    const int32_t total_bins = feature_hist_offsets[n_features];

    /* Zero the output buffer. Bins beyond 0 and 1 for any feature
     * with >2 bins will remain zero (GPU only computes bin=1 sums). */
    memset(output_hist, 0, (size_t)total_bins * HIST_PAIR_SIZE * sizeof(hist_t));

    for (int32_t f = 0; f < n_features; ++f) {
        const int32_t offset = feature_hist_offsets[f];
        const int32_t n_bins_f = feature_hist_offsets[f + 1] - offset;

        if (n_bins_f < 2) continue;  /* Degenerate single-bin feature */

        hist_t* base = output_hist + (size_t)offset * HIST_PAIR_SIZE;

        /* bin 0 (feature OFF) = leaf_total - bin_1 */
        base[0] = leaf_grad_sum - gpu_grad[f];  /* grad_bin0 */
        base[1] = leaf_hess_sum - gpu_hess[f];  /* hess_bin0 */

        /* bin 1 (feature ON) = GPU output directly */
        base[HIST_PAIR_SIZE + 0] = gpu_grad[f]; /* grad_bin1 */
        base[HIST_PAIR_SIZE + 1] = gpu_hess[f]; /* hess_bin1 */
    }
}


/* =========================================================================
 * MapGPUHistToLGBM_Binary — Optimized for uniform binary features
 *
 * When ALL features have exactly 2 bins (the common case for cross features),
 * we skip the offset lookup and use a simple stride-4 pattern.
 * ~2x faster than the general path due to sequential memory access.
 *
 * Parameters: same as MapGPUHistToLGBM except no feature_hist_offsets
 *   (offsets are implicit: feature f starts at bin 2*f).
 *
 * output_hist must have space for n_features * 2 bins * 2 doubles
 *   = n_features * 4 doubles.
 * ========================================================================= */

static inline void MapGPUHistToLGBM_Binary(
    const double* gpu_grad,
    const double* gpu_hess,
    double leaf_grad_sum,
    double leaf_hess_sum,
    int32_t n_features,
    hist_t* output_hist)
{
    for (int32_t f = 0; f < n_features; ++f) {
        hist_t* base = output_hist + (size_t)f * (HIST_PAIR_SIZE * 2);

        /* bin 0 (OFF) = total - bin1 */
        base[0] = leaf_grad_sum - gpu_grad[f];
        base[1] = leaf_hess_sum - gpu_hess[f];

        /* bin 1 (ON) = GPU output */
        base[2] = gpu_grad[f];
        base[3] = gpu_hess[f];
    }
}


/* =========================================================================
 * ValidateHistogramInvariant — Debug-mode correctness check
 *
 * Verifies that bin0 + bin1 == leaf_total for every binary feature.
 * Returns 0 if valid, or the 1-based index of the first failing feature.
 *
 * Only called in debug/test builds — not on the hot path.
 * ========================================================================= */

static inline int32_t ValidateHistogramInvariant(
    const hist_t* hist,
    const int32_t* feature_hist_offsets,
    int32_t n_features,
    double leaf_grad_sum,
    double leaf_hess_sum,
    double atol)
{
    for (int32_t f = 0; f < n_features; ++f) {
        const int32_t offset = feature_hist_offsets[f];
        const int32_t n_bins_f = feature_hist_offsets[f + 1] - offset;

        double sum_grad = 0.0;
        double sum_hess = 0.0;
        for (int32_t b = 0; b < n_bins_f; ++b) {
            const hist_t* entry = hist + (size_t)(offset + b) * HIST_PAIR_SIZE;
            sum_grad += entry[0];
            sum_hess += entry[1];
        }

        if (fabs(sum_grad - leaf_grad_sum) > atol ||
            fabs(sum_hess - leaf_hess_sum) > atol) {
            return f + 1;  /* 1-based index of failing feature */
        }
    }
    return 0;  /* All features passed */
}


#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* GPU_HISTOGRAM_OUTPUT_MAPPER_H_ */
