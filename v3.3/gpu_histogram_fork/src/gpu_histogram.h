/*
 * gpu_histogram.h — GPU Histogram Co-Processor Interface
 *
 * Contract between LightGBM's SerialTreeLearner::ConstructHistograms() and
 * the CUDA backend that replaces it. The GPU builds gradient/hessian histograms
 * over EFB-bundled sparse CSR data; the CPU does everything else (split finding,
 * tree structure, gradient computation, EFB bundling).
 *
 * Matrix thesis: ALL features preserved. Binary cross features in sparse CSR.
 * Structural zeros = feature OFF (correct for binary crosses). No filtering,
 * no subsampling, no row partitioning. The model decides via tree splits.
 *
 * Data flow:
 *   INIT  — CSR matrix uploaded to GPU once (stays resident for all rounds)
 *   ROUND — CPU computes gradients, uploads to GPU via pinned memory
 *   NODE  — GPU builds histogram for one leaf's rows, CPU finds best split
 *   SUBTRACT — GPU computes sibling histogram (parent - child) on-device
 *
 * Memory ownership:
 *   - All host-side arrays (gradients, row_indices, histograms) are OWNED BY
 *     THE CALLER. The library reads from them during the call and does not
 *     retain pointers after the function returns.
 *   - All GPU-side memory is owned by the GpuHistContext (opaque handle).
 *     It is allocated at gpu_hist_init() and freed at gpu_hist_cleanup().
 *   - The CSR arrays passed to gpu_hist_init() are copied to GPU memory.
 *     The caller may free the host-side CSR arrays after init returns.
 *
 * Thread safety:
 *   - A single GpuHistHandle is NOT thread-safe. All calls on the same handle
 *     must be serialized by the caller.
 *   - Multiple handles on different GPU devices are independent and may be
 *     used from different threads concurrently (multi-GPU feature-parallel).
 *   - Internally the library uses multiple CUDA streams for H2D/compute/D2H
 *     overlap, but this is transparent to the caller.
 *
 * Error handling:
 *   - Every function returns GpuHistError. On error, the handle remains valid
 *     and may be retried or cleaned up. No partial state is leaked.
 *   - GPU_HIST_OOM means the requested operation exceeded the VRAM budget.
 *     The caller should fall back to CPU histogram building.
 *
 * Determinism:
 *   - When row_indices are sorted in ascending order, the GPU produces
 *     bit-exact results matching CPU LightGBM (float64 accumulation,
 *     deterministic reduction order). The caller MUST pass sorted row_indices
 *     to guarantee determinism (required when LightGBM deterministic=True).
 *
 * Supported GPUs (fat binary):
 *   sm_80  (A100, A30)
 *   sm_86  (RTX 3090, A40)
 *   sm_89  (RTX 4090, L40)
 *   sm_90  (H100, H200)
 *   sm_100 (B200, B100)
 *   PTX fallback for future architectures.
 */

#ifndef GPU_HISTOGRAM_H_
#define GPU_HISTOGRAM_H_

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Error codes
 * ========================================================================= */

typedef enum {
    GPU_HIST_OK          = 0,  /* Success                                    */
    GPU_HIST_NO_DEVICE   = 1,  /* No CUDA-capable GPU found or device_id     */
                               /* out of range                               */
    GPU_HIST_OOM         = 2,  /* GPU memory allocation failed (CSR too      */
                               /* large for VRAM budget). Caller should      */
                               /* fall back to CPU histogram.                */
    GPU_HIST_INVALID_ARG = 3,  /* Null pointer, negative size, mismatched    */
                               /* dimensions, or other invalid parameter.    */
    GPU_HIST_CUDA_ERROR  = 4,  /* Unexpected CUDA runtime error. Call        */
                               /* gpu_hist_get_last_error() for details.     */
    GPU_HIST_NOT_INIT    = 5,  /* Handle is NULL or was already cleaned up.  */
} GpuHistError;

/* =========================================================================
 * Configuration
 *
 * Passed to gpu_hist_init(). All fields have sensible defaults when
 * zero-initialized (C99 designated initializers or memset).
 * ========================================================================= */

typedef struct {
    int    device_id;       /* CUDA device ordinal. Default 0.               */
                            /* Must be in [0, cudaGetDeviceCount()-1].       */

    size_t max_vram_bytes;  /* Maximum GPU memory the library may allocate.  */
                            /* 0 = auto-detect 85% of total VRAM.           */
                            /* The remaining 15% is reserved for CUDA        */
                            /* context, driver, and external allocations.    */

    int    use_fp64;        /* If nonzero, accumulate histograms in float64  */
                            /* (matching LightGBM's CPU path exactly).      */
                            /* If zero, use float64 anyway — this field is   */
                            /* reserved for a future float32 fast path.     */
                            /* Default: 1 (always float64 for bit-exact).   */

    int    num_streams;     /* Number of CUDA streams for H2D/compute/D2H   */
                            /* overlap. Default 3. Minimum 1, maximum 8.    */
                            /* More streams help when the kernel is short    */
                            /* relative to transfer latency.                */

    int    num_classes;     /* Number of output classes. 1 = binary          */
                            /* classification / regression, 3 = multiclass. */
                            /* Must match the gradient array layout.         */
                            /* Default 3 (our 3-class setup).               */
} GpuHistConfig;

/* =========================================================================
 * Opaque handle
 *
 * Holds all GPU-side state: CSR matrix, gradient buffers, histogram pool,
 * leaf partition array, CUDA streams, and pinned host memory for double-
 * buffered gradient uploads.
 *
 * Created by gpu_hist_init(), destroyed by gpu_hist_cleanup().
 * ========================================================================= */

typedef struct GpuHistContext_* GpuHistHandle;

/* =========================================================================
 * Lifecycle — Init
 *
 * Uploads the CSR matrix to GPU memory where it stays resident for the
 * entire training run. Also allocates the histogram buffer pool, gradient
 * double-buffers, and leaf partition array.
 *
 * The CSR represents the EFB-bundled feature matrix:
 *   - indptr[i] .. indptr[i+1] are the nonzero offsets for row i
 *   - indices[j] gives the EFB bundle-bin index for nonzero j
 *   - No data array is needed for raw binary features (value always 1),
 *     but for EFB-encoded features the data array IS the bin index.
 *     Pass NULL for csr_data if features are raw binary (pre-EFB).
 *     Pass non-NULL csr_data (uint8) for EFB-encoded bundle bins.
 *
 * Parameters:
 *   handle         [out] Receives the new handle on success.
 *   config         [in]  Configuration. May be stack-allocated; the library
 *                        copies what it needs. NULL = all defaults.
 *   csr_indptr     [in]  CSR row pointer array, length n_rows_plus1.
 *                        int64 to support NNZ > 2^31 (15m timeframe).
 *   n_rows_plus1   [in]  Number of rows + 1 (i.e., indptr length).
 *   csr_indices    [in]  CSR column indices, length nnz.
 *                        For raw binary: feature index.
 *                        For EFB-encoded: bundle-bin index.
 *   nnz            [in]  Number of nonzero entries in the CSR matrix.
 *                        May exceed INT32_MAX (int64 indptr handles this).
 *   csr_data       [in]  EFB bundle bin values, length nnz. uint8 because
 *                        max_bin=255 fits in one byte. NULL = raw binary
 *                        features (all nonzeros treated as bin 1).
 *   n_features     [in]  Total number of features (columns) in the original
 *                        matrix. Used for histogram output sizing when
 *                        operating in raw binary mode (no EFB).
 *   n_bundles      [in]  Number of EFB bundles. 0 = raw binary mode
 *                        (n_bundles treated as n_features, 2 bins each).
 *   max_bin        [in]  Maximum bin count per bundle (from LightGBM config).
 *                        255 for our setup. Determines histogram buffer size
 *                        per bundle.
 *
 * Returns:
 *   GPU_HIST_OK          on success
 *   GPU_HIST_NO_DEVICE   if device_id is invalid or no GPU found
 *   GPU_HIST_OOM         if CSR + buffers exceed max_vram_bytes
 *   GPU_HIST_INVALID_ARG if handle==NULL, csr_indptr==NULL, etc.
 *
 * After this call, the caller may free the host-side CSR arrays.
 * ========================================================================= */

GpuHistError gpu_hist_init(
    GpuHistHandle*   handle,
    const GpuHistConfig* config,
    /* CSR matrix (copied to GPU, stays resident) */
    const int64_t*   csr_indptr,       /* [n_rows_plus1]              */
    int64_t          n_rows_plus1,
    const int32_t*   csr_indices,      /* [nnz]                       */
    int64_t          nnz,
    const uint8_t*   csr_data,         /* [nnz] or NULL for raw binary */
    int32_t          n_features,
    int32_t          n_bundles,        /* 0 = raw binary mode          */
    int32_t          max_bin           /* 255 for our config           */
);

/* =========================================================================
 * Per-Round Gradient Update
 *
 * Called once per boosting round BEFORE any histogram builds for that round.
 * Uploads the new gradients and hessians to GPU via pinned memory with
 * double-buffering (the previous round's upload may still be in flight;
 * the library handles synchronization internally).
 *
 * For multiclass (num_classes=3), gradients are interleaved:
 *   gradients[row * num_classes + class_id]
 *   hessians [row * num_classes + class_id]
 *
 * For binary/regression (num_classes=1):
 *   gradients[row]
 *   hessians [row]
 *
 * Parameters:
 *   handle     [in] Valid handle from gpu_hist_init().
 *   gradients  [in] Gradient array, length n_rows * num_classes. float64
 *                   to match LightGBM's internal precision.
 *   hessians   [in] Hessian array, same layout as gradients.
 *
 * Memory: The library copies from gradients/hessians into pinned memory
 *         and initiates async H2D transfer. The caller's arrays may be
 *         modified after this call returns.
 *
 * Returns:
 *   GPU_HIST_OK          on success
 *   GPU_HIST_NOT_INIT    if handle is NULL
 *   GPU_HIST_INVALID_ARG if gradients or hessians is NULL
 *   GPU_HIST_CUDA_ERROR  on transfer failure
 * ========================================================================= */

GpuHistError gpu_hist_update_gradients(
    GpuHistHandle    handle,
    const double*    gradients,        /* [n_rows * num_classes]       */
    const double*    hessians          /* [n_rows * num_classes]       */
);

/* =========================================================================
 * Build Histogram for a Single Leaf
 *
 * This is the hot path — called up to 62 times per tree (num_leaves=63,
 * minus subtraction trick which halves it to ~31 calls per tree).
 *
 * The GPU scans all CSR rows belonging to this leaf and accumulates
 * gradient/hessian sums per EFB bundle bin. For binary crosses, most
 * entries are bin 0 (feature OFF) — the kernel skips these, computing
 * bin-0 by subtraction from the total.
 *
 * DETERMINISM: row_indices MUST be sorted in ascending order for the
 * GPU to produce bit-exact results matching CPU LightGBM. The library
 * does NOT sort internally (to avoid allocation/overhead on every call).
 *
 * Parameters:
 *   handle           [in]  Valid handle.
 *   row_indices      [in]  Sorted row indices belonging to this leaf.
 *                          Length n_leaf_rows. int32 is sufficient since
 *                          max rows = 227K (15m) < INT32_MAX.
 *   n_leaf_rows      [in]  Number of rows in this leaf. Must be > 0.
 *   class_id         [in]  Which class to build histograms for.
 *                          Range [0, num_classes-1]. For multiclass
 *                          LightGBM, each boosting round trains one class.
 *   output_histogram [out] Caller-allocated buffer receiving the histogram.
 *                          Layout: interleaved grad/hess pairs per bin.
 *                          Size: n_hist_bins * 2 doubles.
 *                          output_histogram[bin * 2 + 0] = sum of gradients
 *                          output_histogram[bin * 2 + 1] = sum of hessians
 *                          The library copies from GPU to this buffer via
 *                          D2H transfer before returning.
 *   n_hist_bins      [in]  Total number of histogram bins across all
 *                          bundles (sum of per-bundle bin counts).
 *                          For raw binary mode: n_features (2 bins each,
 *                          but bin 0 is computed by subtraction, so
 *                          n_hist_bins = n_features).
 *                          For EFB mode: sum over all bundles of their
 *                          individual bin counts.
 *
 * Performance:
 *   Internally uses shared memory tiling over bundles. Tile size adapts
 *   to the GPU's shared memory capacity (see ARCHITECTURE.md Section 4.3).
 *   Binary skip-zero optimization eliminates ~95% of atomicAdd calls.
 *
 * Returns:
 *   GPU_HIST_OK          on success
 *   GPU_HIST_NOT_INIT    if handle is NULL
 *   GPU_HIST_INVALID_ARG if row_indices==NULL, n_leaf_rows<=0,
 *                        output_histogram==NULL, class_id out of range,
 *                        or n_hist_bins mismatches the init configuration
 *   GPU_HIST_CUDA_ERROR  on kernel launch or D2H failure
 * ========================================================================= */

GpuHistError gpu_hist_build(
    GpuHistHandle    handle,
    const int32_t*   row_indices,      /* [n_leaf_rows], sorted asc    */
    int32_t          n_leaf_rows,
    int32_t          class_id,
    double*          output_histogram, /* [n_hist_bins * 2]            */
    int32_t          n_hist_bins
);

/* =========================================================================
 * Histogram Subtraction (Sibling = Parent - Child)
 *
 * LightGBM's subtraction trick: after splitting a leaf, only the SMALLER
 * child's histogram is built by scanning rows. The larger child's histogram
 * is computed as parent - smaller_child. This halves the number of expensive
 * GPU histogram builds per tree level.
 *
 * This subtraction is done on the GPU to avoid a D2H + CPU subtract + H2D
 * round-trip. The parent and child histograms are already on-device from
 * the gpu_hist_build() call.
 *
 * Parameters:
 *   handle       [in]  Valid handle.
 *   parent_hist  [in]  Parent leaf histogram, length n_bins * 2 doubles.
 *                      This is a HOST pointer — the library uploads it to
 *                      GPU if not already resident. For the common case
 *                      where the parent was computed by a prior gpu_hist_build(),
 *                      the library may use the on-device copy directly
 *                      (implementation-defined optimization).
 *   child_hist   [in]  Smaller child histogram (from gpu_hist_build()),
 *                      length n_bins * 2 doubles. HOST pointer.
 *   sibling_hist [out] Receives parent - child, length n_bins * 2 doubles.
 *                      HOST pointer. May alias parent_hist (in-place).
 *   n_bins       [in]  Number of histogram bins (same as n_hist_bins in
 *                      gpu_hist_build).
 *
 * Returns:
 *   GPU_HIST_OK          on success
 *   GPU_HIST_NOT_INIT    if handle is NULL
 *   GPU_HIST_INVALID_ARG if any pointer is NULL or n_bins <= 0
 *   GPU_HIST_CUDA_ERROR  on kernel or transfer failure
 * ========================================================================= */

GpuHistError gpu_hist_subtract(
    GpuHistHandle    handle,
    const double*    parent_hist,       /* [n_bins * 2]                */
    const double*    child_hist,        /* [n_bins * 2]                */
    double*          sibling_hist,      /* [n_bins * 2], may alias parent */
    int32_t          n_bins
);

/* =========================================================================
 * GPU-Side Row Partition Update
 *
 * After the CPU finds the best split, this function tells the GPU to
 * update its internal leaf_id[] array. This avoids sending full row index
 * lists for each leaf — instead, only 12 bytes (the split decision) are
 * transferred per node.
 *
 * The GPU scans rows in the old leaf, evaluates the split condition using
 * the resident CSR data, and assigns rows to left or right child.
 *
 * Parameters:
 *   handle          [in] Valid handle.
 *   old_leaf_id     [in] Leaf being split (0-based).
 *   new_left_id     [in] ID for the left child leaf.
 *   new_right_id    [in] ID for the right child leaf.
 *   split_feature   [in] Feature index (or EFB bundle-bin index) to split on.
 *   split_threshold [in] Split threshold. Rows with feature value <=
 *                        threshold go left, > threshold go right.
 *                        For binary features: threshold = 0.5 means
 *                        value=0 -> left, value=1 -> right.
 *   left_count      [out] If non-NULL, receives the number of rows in left child.
 *   right_count     [out] If non-NULL, receives the number of rows in right child.
 *
 * Returns:
 *   GPU_HIST_OK          on success
 *   GPU_HIST_NOT_INIT    if handle is NULL
 *   GPU_HIST_INVALID_ARG if leaf IDs are out of range
 *   GPU_HIST_CUDA_ERROR  on kernel failure
 * ========================================================================= */

GpuHistError gpu_hist_update_partition(
    GpuHistHandle    handle,
    int32_t          old_leaf_id,
    int32_t          new_left_id,
    int32_t          new_right_id,
    int32_t          split_feature,
    float            split_threshold,
    int32_t*         left_count,        /* [out] or NULL               */
    int32_t*         right_count        /* [out] or NULL               */
);

/* =========================================================================
 * Reset Partition for New Tree
 *
 * Resets the GPU-side leaf_id[] array so all rows belong to leaf 0 (root).
 * Called at the start of each new tree within a boosting round.
 *
 * Returns:
 *   GPU_HIST_OK          on success
 *   GPU_HIST_NOT_INIT    if handle is NULL
 *   GPU_HIST_CUDA_ERROR  on kernel failure
 * ========================================================================= */

GpuHistError gpu_hist_reset_partition(
    GpuHistHandle    handle
);

/* =========================================================================
 * Cleanup
 *
 * Frees all GPU memory, destroys CUDA streams, releases pinned host memory.
 * After this call, the handle is invalid and must not be used.
 *
 * Safe to call with NULL handle (no-op). Safe to call multiple times
 * (second call is a no-op).
 *
 * Returns:
 *   GPU_HIST_OK          always (cleanup never fails)
 * ========================================================================= */

GpuHistError gpu_hist_cleanup(
    GpuHistHandle    handle
);

/* =========================================================================
 * Query — VRAM Usage
 *
 * Reports current GPU memory usage by this handle and total device VRAM.
 * Useful for logging, auto-tuning, and deciding whether to fall back to
 * CPU for larger timeframes.
 *
 * Parameters:
 *   handle      [in]  Valid handle.
 *   used_bytes  [out] Bytes allocated by this handle on the GPU.
 *   total_bytes [out] Total device VRAM in bytes.
 *
 * Returns:
 *   GPU_HIST_OK          on success
 *   GPU_HIST_NOT_INIT    if handle is NULL
 *   GPU_HIST_INVALID_ARG if used_bytes or total_bytes is NULL
 * ========================================================================= */

GpuHistError gpu_hist_get_vram_usage(
    GpuHistHandle    handle,
    size_t*          used_bytes,
    size_t*          total_bytes
);

/* =========================================================================
 * Query — Last CUDA Error String
 *
 * When a function returns GPU_HIST_CUDA_ERROR, call this to get a
 * human-readable description of the underlying CUDA error.
 *
 * Parameters:
 *   handle  [in] Valid handle (or NULL for init-phase errors).
 *
 * Returns:
 *   Static string describing the last CUDA error, or "no error" if none.
 *   The string is valid until the next call on the same handle.
 * ========================================================================= */

const char* gpu_hist_get_last_error(
    GpuHistHandle    handle
);

/* =========================================================================
 * Query — Device Info
 *
 * Reports GPU hardware info for logging and adaptive tile sizing.
 *
 * Parameters:
 *   handle               [in]  Valid handle.
 *   device_name          [out] Buffer to receive device name (null-terminated).
 *   device_name_len      [in]  Length of device_name buffer.
 *   compute_major        [out] Compute capability major version.
 *   compute_minor        [out] Compute capability minor version.
 *   sm_count             [out] Number of streaming multiprocessors.
 *   shared_mem_per_sm    [out] Shared memory per SM in bytes.
 *
 * Any output pointer may be NULL (skipped).
 *
 * Returns:
 *   GPU_HIST_OK          on success
 *   GPU_HIST_NOT_INIT    if handle is NULL
 * ========================================================================= */

GpuHistError gpu_hist_get_device_info(
    GpuHistHandle    handle,
    char*            device_name,
    int32_t          device_name_len,
    int32_t*         compute_major,
    int32_t*         compute_minor,
    int32_t*         sm_count,
    size_t*          shared_mem_per_sm
);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* GPU_HISTOGRAM_H_ */
