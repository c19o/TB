/*
 * gpu_histogram.cu — CUDA kernel implementation for sparse CSR histogram building
 *
 * Part of the GPU Histogram Co-Processor Fork for LightGBM (v3.3)
 *
 * Implements the interface defined in gpu_histogram.h. The GPU replaces ONLY
 * LightGBM's SerialTreeLearner::ConstructHistograms() hot path. Everything
 * else (split finding, tree growing, EFB bundling, gradient computation)
 * stays on CPU.
 *
 * Matrix Thesis: ALL features preserved. NO filtering. NO subsampling.
 * Sparse binary cross features ARE the edge. This kernel processes the
 * full CSR matrix — structural zeros = feature OFF (correct for binary
 * crosses after binarization).
 *
 * Architecture: Row-parallel atomic scatter. Each thread handles one leaf
 * row, walks its CSR nonzeros, atomicAdd grad/hess into histogram bins.
 * Ultra-sparse data (~99.7% zeros) means near-zero atomic contention.
 *
 * Compile (fat binary for all supported architectures):
 *   nvcc -O3 -shared -Xcompiler -fPIC \
 *        -gencode arch=compute_80,code=sm_80 \
 *        -gencode arch=compute_86,code=sm_86 \
 *        -gencode arch=compute_89,code=sm_89 \
 *        -gencode arch=compute_90,code=sm_90 \
 *        -gencode arch=compute_100,code=sm_100 \
 *        -gencode arch=compute_100,code=compute_100 \
 *        -o libgpu_histogram.so gpu_histogram.cu
 *
 * The final -gencode with code=compute_100 embeds PTX for forward compat
 * with future architectures.
 *
 * Requires: CUDA 12.0+ (sm_100 needs CUDA 12.8+), driver 535+
 */

#include "gpu_histogram.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>


/* ===================================================================
 * Error Checking Macros
 * =================================================================== */

/* Set last error message and return GpuHistError */
#define SET_ERROR(ctx, cuda_err)                                               \
    do {                                                                        \
        if (ctx) {                                                              \
            snprintf((ctx)->last_error_msg, sizeof((ctx)->last_error_msg),      \
                     "%s:%d — %s", __FILE__, __LINE__,                         \
                     cudaGetErrorString(cuda_err));                            \
        }                                                                       \
    } while (0)

#define CUDA_CHECK(ctx, call)                                                  \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            SET_ERROR(ctx, err__);                                              \
            return GPU_HIST_CUDA_ERROR;                                         \
        }                                                                       \
    } while (0)

/* Variant for init-phase where we need to do cleanup on failure */
#define CUDA_CHECK_INIT(call, cleanup_label)                                   \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            SET_ERROR(ctx, err__);                                              \
            result = GPU_HIST_CUDA_ERROR;                                       \
            goto cleanup_label;                                                 \
        }                                                                       \
    } while (0)

#define CUDA_CHECK_OOM(call, cleanup_label)                                    \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ == cudaErrorMemoryAllocation) {                               \
            SET_ERROR(ctx, err__);                                              \
            result = GPU_HIST_OOM;                                              \
            goto cleanup_label;                                                 \
        } else if (err__ != cudaSuccess) {                                      \
            SET_ERROR(ctx, err__);                                              \
            result = GPU_HIST_CUDA_ERROR;                                       \
            goto cleanup_label;                                                 \
        }                                                                       \
    } while (0)


/* ===================================================================
 * Constants
 * =================================================================== */

/* Threads per block. 256 = 8 warps: enough to hide memory latency,
 * allows 8 blocks per SM on most architectures, good register balance. */
static constexpr int BLOCK_SIZE = 256;

/* Tiled kernel: minimum rows before tiling is worthwhile.
 * Below this, global-memory atomics have lower launch overhead. */
static constexpr int TILED_MIN_ROWS = 1024;

/* Maximum bins per shared memory tile. Caps memory even on GPUs that
 * report very large configurable shared memory. */
static constexpr int MAX_TILE_BINS = 8192;

/* Maximum streams */
static constexpr int MAX_STREAMS = 8;

/* Default VRAM fraction if max_vram_bytes == 0 */
static constexpr double VRAM_AUTO_FRACTION = 0.85;


/* ===================================================================
 * Internal Context Structure
 *
 * This is the concrete type behind the opaque GpuHistHandle.
 * All GPU memory is pre-allocated at init and reused. No dynamic
 * allocation during the training loop.
 * =================================================================== */

struct GpuHistContext_ {
    /* ---- CSR data (GPU-resident, read-only after init) ---- */
    int64_t*  d_indptr;        /* [n_rows + 1] */
    int32_t*  d_indices;       /* [nnz] */
    uint8_t*  d_data;          /* [nnz] EFB bin values, or NULL for raw binary */

    /* ---- Gradients/Hessians (double-buffered pinned + device) ---- */
    double*   d_grad;          /* [n_rows * num_classes] on GPU */
    double*   d_hess;          /* [n_rows * num_classes] on GPU */
    double*   h_grad_pinned;   /* pinned host buffer A */
    double*   h_hess_pinned;   /* pinned host buffer A */

    /* ---- Leaf row indices ---- */
    int32_t*  d_leaf_rows;     /* [n_rows] max — reused per node */

    /* ---- GPU-side partition map ---- */
    int8_t*   d_leaf_id;       /* [n_rows] leaf assignment */
    int32_t*  d_count_buf;     /* small device buffer for atomics (2 ints) */

    /* ---- Histogram buffers (on device) ---- */
    double*   d_hist;          /* current histogram being built */
    double*   d_hist_parent;   /* saved parent for subtraction trick */
    double*   d_hist_sibling;  /* result of subtraction */

    /* ---- Pinned host histogram for D2H ---- */
    double*   h_hist_pinned;

    /* ---- CUDA streams ---- */
    cudaStream_t streams[MAX_STREAMS];
    int          num_streams;

    /* Stream roles (indices into streams[]) */
    int          stream_h2d;     /* host-to-device transfers */
    int          stream_compute; /* kernel execution */
    int          stream_d2h;     /* device-to-host transfers */

    /* ---- Dimensions ---- */
    int64_t   n_rows;
    int64_t   nnz;
    int32_t   n_features;
    int32_t   n_bundles;         /* 0 = raw binary mode */
    int32_t   max_bin;
    int32_t   num_classes;
    int32_t   total_hist_bins;   /* n_features for raw binary, or sum(bundle bins) */
    int64_t   hist_elements;     /* total_hist_bins * 2 (interleaved grad/hess) */

    /* ---- GPU info ---- */
    int       device_id;
    int       smem_bytes_per_block; /* max shared mem per block (configurable) */
    int       tile_size;            /* features per shared-memory tile (0=disabled) */
    int       sm_count;
    char      device_name[256];
    int       compute_major;
    int       compute_minor;

    /* ---- Memory tracking ---- */
    size_t    gpu_bytes_allocated;
    size_t    total_device_vram;

    /* ---- Error state ---- */
    char      last_error_msg[512];

    /* ---- Lifecycle flag ---- */
    int       initialized;
};


/* ===================================================================
 * KERNEL 1: Sparse Histogram Build — Global Memory Atomics
 *
 * Primary kernel. Each thread processes one leaf row. Walks CSR
 * nonzeros and atomicAdd grad/hess into interleaved histogram bins.
 *
 * For raw binary mode (data == NULL): every stored nonzero is bin=1.
 * For EFB mode (data != NULL): data[j] gives the EFB bundle bin index.
 *   bin=0 means all features in this bundle are OFF for this row —
 *   skip it (bin=0 histogram computed by subtraction later).
 *
 * Histogram layout: [total_hist_bins * 2] doubles
 *   hist[bin * 2 + 0] = gradient sum
 *   hist[bin * 2 + 1] = hessian sum
 * =================================================================== */

__global__ void sparse_hist_build_kernel(
    const int64_t* __restrict__ indptr,
    const int32_t* __restrict__ indices,
    const uint8_t* __restrict__ data,        /* NULL for raw binary */
    const double*  __restrict__ gradients,   /* [n_rows * num_classes] */
    const double*  __restrict__ hessians,
    const int32_t* __restrict__ leaf_rows,
    int32_t                     n_leaf_rows,
    int32_t                     class_id,
    int32_t                     num_classes,
    double*        __restrict__ hist
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_leaf_rows) return;

    int32_t row = leaf_rows[tid];

    /* Index into the class-specific gradient/hessian */
    double g = gradients[row * num_classes + class_id];
    double h = hessians[row * num_classes + class_id];

    int64_t start = indptr[row];
    int64_t end   = indptr[row + 1];

    if (data == nullptr) {
        /* Raw binary mode: every nonzero entry is bin=1 (feature ON).
         * indices[j] is the feature column index.
         * Atomic contention analysis: ~9K nonzeros per row out of 3M+
         * features means P(collision) = (9K/3M)^2 * warps ≈ negligible. */
        for (int64_t j = start; j < end; j++) {
            int32_t col = indices[j];
            atomicAdd(&hist[col * 2],     g);
            atomicAdd(&hist[col * 2 + 1], h);
        }
    } else {
        /* EFB mode: data[j] is the pre-computed bundle bin index.
         * bin=0 means all features in the bundle are OFF — skip it.
         * indices[j] is the EFB bundle-bin combined index. */
        for (int64_t j = start; j < end; j++) {
            uint8_t bin = data[j];
            if (bin == 0) continue;  /* structural zero in this bundle */
            int32_t col = indices[j];
            atomicAdd(&hist[col * 2],     g);
            atomicAdd(&hist[col * 2 + 1], h);
        }
    }
}


/* ===================================================================
 * KERNEL 2: Sparse Histogram Build — Shared Memory Tiled
 *
 * For GPUs with large shared memory (A100: 164KB, H100: 228KB).
 * The full histogram is too large for shared memory, so we tile
 * over feature column ranges. Each block accumulates into a local
 * shared-memory histogram tile, then flushes to global with atomicAdd.
 *
 * Each thread block processes all its assigned rows but only
 * accumulates features in [tile_start, tile_end). Multiple kernel
 * launches (one per tile) cover the full feature range.
 *
 * Net win when n_leaf_rows > TILED_MIN_ROWS: shared memory atomics
 * are ~10x faster than global memory atomics due to lower latency.
 * =================================================================== */

__global__ void sparse_hist_build_tiled_kernel(
    const int64_t* __restrict__ indptr,
    const int32_t* __restrict__ indices,
    const uint8_t* __restrict__ data,
    const double*  __restrict__ gradients,
    const double*  __restrict__ hessians,
    const int32_t* __restrict__ leaf_rows,
    int32_t                     n_leaf_rows,
    int32_t                     class_id,
    int32_t                     num_classes,
    double*        __restrict__ hist,
    int32_t                     tile_start,
    int32_t                     tile_end
) {
    /* Dynamic shared memory: tile_size * 2 doubles (grad + hess per bin) */
    extern __shared__ double smem_hist[];

    int tile_size = tile_end - tile_start;
    int smem_elems = tile_size * 2;

    /* Zero shared memory cooperatively across the block */
    for (int i = threadIdx.x; i < smem_elems; i += blockDim.x) {
        smem_hist[i] = 0.0;
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_leaf_rows) {
        int32_t row = leaf_rows[tid];
        double g = gradients[row * num_classes + class_id];
        double h = hessians[row * num_classes + class_id];

        int64_t start = indptr[row];
        int64_t end   = indptr[row + 1];

        if (data == nullptr) {
            for (int64_t j = start; j < end; j++) {
                int32_t col = indices[j];
                if (col >= tile_start && col < tile_end) {
                    int local = (col - tile_start) * 2;
                    atomicAdd(&smem_hist[local],     g);
                    atomicAdd(&smem_hist[local + 1], h);
                }
            }
        } else {
            for (int64_t j = start; j < end; j++) {
                uint8_t bin = data[j];
                if (bin == 0) continue;
                int32_t col = indices[j];
                if (col >= tile_start && col < tile_end) {
                    int local = (col - tile_start) * 2;
                    atomicAdd(&smem_hist[local],     g);
                    atomicAdd(&smem_hist[local + 1], h);
                }
            }
        }
    }
    __syncthreads();

    /* Flush shared memory to global histogram.
     * Skip zero entries to reduce global atomicAdd pressure. */
    for (int i = threadIdx.x; i < smem_elems; i += blockDim.x) {
        if (smem_hist[i] != 0.0) {
            atomicAdd(&hist[tile_start * 2 + i], smem_hist[i]);
        }
    }
}


/* ===================================================================
 * KERNEL 3: Histogram Subtraction
 *
 * sibling[i] = parent[i] - child[i]
 * Used for LightGBM's histogram subtraction trick.
 * =================================================================== */

__global__ void hist_subtract_kernel(
    const double* __restrict__ parent,
    const double* __restrict__ child,
    double*       __restrict__ sibling,
    int32_t                    n_elements
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_elements) {
        sibling[i] = parent[i] - child[i];
    }
}


/* ===================================================================
 * KERNEL 4: Histogram Zero
 *
 * Kernel-based memset on the compute stream. Avoids potential
 * stream sync issues with cudaMemsetAsync on some driver versions.
 * =================================================================== */

__global__ void hist_zero_kernel(
    double*  __restrict__ hist,
    int32_t              n_elements
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_elements) {
        hist[i] = 0.0;
    }
}


/* ===================================================================
 * KERNEL 5: Partition Update
 *
 * After CPU finds best split, update leaf_id[] on GPU.
 * Avoids transferring full row index arrays — only the split
 * decision (12 bytes) is needed.
 *
 * For raw binary features: check if the feature column is nonzero
 * for this row by scanning CSR. For EFB features: compare data[j]
 * against threshold.
 *
 * Outputs left/right counts via atomic counters.
 * =================================================================== */

__global__ void partition_update_kernel(
    const int64_t* __restrict__ indptr,
    const int32_t* __restrict__ indices,
    const uint8_t* __restrict__ data,
    int8_t*        __restrict__ leaf_id,
    int32_t                     n_rows,
    int32_t                     old_leaf,
    int32_t                     new_left,
    int32_t                     new_right,
    int32_t                     split_feature,
    float                       split_threshold,
    int32_t*       __restrict__ counts    /* [2]: left, right */
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    if (leaf_id[row] != (int8_t)old_leaf) return;

    /* Find whether this row has the split feature nonzero.
     * Binary search through the CSR row's sorted indices. */
    int64_t lo = indptr[row];
    int64_t hi = indptr[row + 1] - 1;
    float feat_val = 0.0f;  /* default: feature is OFF (structural zero) */

    while (lo <= hi) {
        int64_t mid = lo + (hi - lo) / 2;
        int32_t col = indices[mid];
        if (col == split_feature) {
            /* Feature is present in this row */
            if (data != nullptr) {
                feat_val = (float)data[mid];
            } else {
                feat_val = 1.0f;  /* raw binary: present = 1 */
            }
            break;
        } else if (col < split_feature) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }

    /* LightGBM convention: <= threshold goes left, > threshold goes right */
    if (feat_val <= split_threshold) {
        leaf_id[row] = (int8_t)new_left;
        if (counts) atomicAdd(&counts[0], 1);
    } else {
        leaf_id[row] = (int8_t)new_right;
        if (counts) atomicAdd(&counts[1], 1);
    }
}


/* ===================================================================
 * KERNEL 6: Gather Leaf Rows
 *
 * Compact rows matching a given leaf_id into a dense array.
 * Uses a global atomic counter for the output position.
 * =================================================================== */

__global__ void gather_leaf_rows_kernel(
    const int8_t*  __restrict__ leaf_id,
    int32_t*       __restrict__ leaf_rows,
    int32_t*       __restrict__ count,
    int8_t                      target_leaf,
    int32_t                     n_rows
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_rows && leaf_id[i] == target_leaf) {
        int pos = atomicAdd(count, 1);
        leaf_rows[pos] = i;
    }
}


/* ===================================================================
 * KERNEL 7: Partition Reset
 *
 * Set all leaf_id entries to 0 (root leaf). Called at the start
 * of each new tree.
 * =================================================================== */

__global__ void partition_reset_kernel(
    int8_t*  __restrict__ leaf_id,
    int32_t              n_rows
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_rows) {
        leaf_id[i] = 0;
    }
}


/* ===================================================================
 * Helper: grid dimensions
 * =================================================================== */

static inline int grid_blocks(int n_elements) {
    return (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

static inline int grid_blocks64(int64_t n_elements) {
    return (int)((n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE);
}


/* ===================================================================
 * HOST FUNCTION IMPLEMENTATIONS
 * =================================================================== */

extern "C" {

/* ---- Init ---- */

GpuHistError gpu_hist_init(
    GpuHistHandle*       handle,
    const GpuHistConfig* config,
    const int64_t*       csr_indptr,
    int64_t              n_rows_plus1,
    const int32_t*       csr_indices,
    int64_t              nnz,
    const uint8_t*       csr_data,
    int32_t              n_features,
    int32_t              n_bundles,
    int32_t              max_bin
) {
    /* ---- Argument validation ---- */
    if (!handle)      return GPU_HIST_INVALID_ARG;
    if (!csr_indptr)  return GPU_HIST_INVALID_ARG;
    if (!csr_indices) return GPU_HIST_INVALID_ARG;
    if (n_rows_plus1 < 2) return GPU_HIST_INVALID_ARG;
    if (nnz < 0)     return GPU_HIST_INVALID_ARG;
    if (n_features < 1) return GPU_HIST_INVALID_ARG;
    if (max_bin < 2)  return GPU_HIST_INVALID_ARG;

    *handle = nullptr;

    /* ---- Apply config defaults ---- */
    GpuHistConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    if (config) {
        cfg = *config;
    }
    if (cfg.num_streams < 1) cfg.num_streams = 3;
    if (cfg.num_streams > MAX_STREAMS) cfg.num_streams = MAX_STREAMS;
    if (cfg.num_classes < 1) cfg.num_classes = 3;

    /* ---- Allocate context ---- */
    GpuHistContext_* ctx = (GpuHistContext_*)calloc(1, sizeof(GpuHistContext_));
    if (!ctx) return GPU_HIST_CUDA_ERROR;

    GpuHistError result = GPU_HIST_OK;

    ctx->n_rows      = n_rows_plus1 - 1;
    ctx->nnz         = nnz;
    ctx->n_features  = n_features;
    ctx->n_bundles   = n_bundles;
    ctx->max_bin     = max_bin;
    ctx->num_classes = cfg.num_classes;
    ctx->device_id   = cfg.device_id;
    ctx->num_streams = cfg.num_streams;

    /* Assign stream roles */
    ctx->stream_h2d     = 0;
    ctx->stream_compute = (cfg.num_streams >= 2) ? 1 : 0;
    ctx->stream_d2h     = (cfg.num_streams >= 3) ? 2 : 0;

    /* Histogram sizing:
     * Raw binary mode (n_bundles == 0): n_features bins, each feature has 2 bins
     *   but we only store bin=1 (bin=0 = total - bin=1).
     *   So total_hist_bins = n_features.
     * EFB mode: total bins = sum of per-bundle bin counts. For simplicity,
     *   we allocate n_bundles * max_bin bins (upper bound). LightGBM provides
     *   exact counts via bundle metadata — for now, we use the generous bound. */
    if (n_bundles > 0) {
        ctx->total_hist_bins = n_bundles * max_bin;
    } else {
        ctx->total_hist_bins = n_features;
    }
    ctx->hist_elements = (int64_t)ctx->total_hist_bins * 2;

    /* ---- Select and validate GPU device ---- */
    int device_count = 0;
    CUDA_CHECK_INIT(cudaGetDeviceCount(&device_count), fail_free_ctx);
    if (device_count == 0 || cfg.device_id >= device_count) {
        result = GPU_HIST_NO_DEVICE;
        goto fail_free_ctx;
    }
    CUDA_CHECK_INIT(cudaSetDevice(cfg.device_id), fail_free_ctx);

    /* ---- Query device properties ---- */
    {
        cudaDeviceProp prop;
        CUDA_CHECK_INIT(cudaGetDeviceProperties(&prop, cfg.device_id), fail_free_ctx);

        strncpy(ctx->device_name, prop.name, sizeof(ctx->device_name) - 1);
        ctx->compute_major = prop.major;
        ctx->compute_minor = prop.minor;
        ctx->sm_count = prop.multiProcessorCount;
        ctx->total_device_vram = prop.totalGlobalMem;

        /* Use opt-in shared memory for tiling (sm_80+ supports configurable smem) */
        ctx->smem_bytes_per_block = (int)prop.sharedMemPerBlockOptin;

        /* Compute tile size: each bin uses 2 doubles = 16 bytes in smem.
         * Use 90% of available smem, cap at MAX_TILE_BINS. */
        int usable_smem = (int)(ctx->smem_bytes_per_block * 0.9);
        ctx->tile_size = usable_smem / (2 * (int)sizeof(double));
        if (ctx->tile_size > MAX_TILE_BINS) ctx->tile_size = MAX_TILE_BINS;
        if (ctx->tile_size < 64) ctx->tile_size = 0;  /* too small, disable */

        fprintf(stderr, "[gpu_hist] Device %d: %s (sm_%d%d), %d SMs, "
                "VRAM: %zu MB, Shared mem/block: %d KB, Tile: %d bins\n",
                cfg.device_id, prop.name, prop.major, prop.minor,
                prop.multiProcessorCount,
                prop.totalGlobalMem / (1024 * 1024),
                ctx->smem_bytes_per_block / 1024,
                ctx->tile_size);
    }

    /* ---- VRAM budget check ---- */
    {
        size_t csr_bytes = (size_t)(ctx->n_rows + 1) * sizeof(int64_t)
                         + (size_t)nnz * sizeof(int32_t)
                         + (csr_data ? (size_t)nnz * sizeof(uint8_t) : 0);
        size_t grad_bytes = (size_t)ctx->n_rows * ctx->num_classes
                          * sizeof(double) * 2;  /* grad + hess */
        size_t leaf_bytes = (size_t)ctx->n_rows * sizeof(int32_t)   /* leaf_rows */
                          + (size_t)ctx->n_rows * sizeof(int8_t)    /* leaf_id */
                          + 2 * sizeof(int32_t);                    /* count_buf */
        size_t hist_bytes = (size_t)ctx->hist_elements * sizeof(double) * 3;
        size_t pinned_bytes = (size_t)ctx->n_rows * ctx->num_classes
                            * sizeof(double) * 2   /* grad+hess pinned */
                            + (size_t)ctx->hist_elements * sizeof(double);
        size_t total_gpu = csr_bytes + grad_bytes + leaf_bytes + hist_bytes;

        size_t max_allowed = cfg.max_vram_bytes;
        if (max_allowed == 0) {
            max_allowed = (size_t)(ctx->total_device_vram * VRAM_AUTO_FRACTION);
        }

        if (total_gpu > max_allowed) {
            fprintf(stderr, "[gpu_hist] VRAM budget exceeded: need %zu MB, "
                    "limit %zu MB (%.0f%% of %zu MB)\n",
                    total_gpu / (1024*1024), max_allowed / (1024*1024),
                    VRAM_AUTO_FRACTION * 100, ctx->total_device_vram / (1024*1024));
            result = GPU_HIST_OOM;
            goto fail_free_ctx;
        }

        fprintf(stderr, "[gpu_hist] VRAM estimate: CSR %zu MB, grad %zu MB, "
                "hist %zu MB, total %zu MB / %zu MB\n",
                csr_bytes / (1024*1024), grad_bytes / (1024*1024),
                hist_bytes / (1024*1024), total_gpu / (1024*1024),
                max_allowed / (1024*1024));

        (void)pinned_bytes;  /* tracked but not part of GPU VRAM */
    }

    /* ---- Create CUDA streams ---- */
    for (int i = 0; i < ctx->num_streams; i++) {
        CUDA_CHECK_OOM(
            cudaStreamCreateWithFlags(&ctx->streams[i], cudaStreamNonBlocking),
            fail_cleanup);
    }

    /* ---- Allocate and transfer CSR data ---- */
    CUDA_CHECK_OOM(
        cudaMalloc(&ctx->d_indptr, (ctx->n_rows + 1) * sizeof(int64_t)),
        fail_cleanup);
    ctx->gpu_bytes_allocated += (ctx->n_rows + 1) * sizeof(int64_t);

    CUDA_CHECK_OOM(
        cudaMalloc(&ctx->d_indices, (size_t)nnz * sizeof(int32_t)),
        fail_cleanup);
    ctx->gpu_bytes_allocated += (size_t)nnz * sizeof(int32_t);

    /* Synchronous transfer for CSR — this is the one-time init cost */
    CUDA_CHECK_INIT(
        cudaMemcpy(ctx->d_indptr, csr_indptr,
                   (ctx->n_rows + 1) * sizeof(int64_t), cudaMemcpyHostToDevice),
        fail_cleanup);
    CUDA_CHECK_INIT(
        cudaMemcpy(ctx->d_indices, csr_indices,
                   (size_t)nnz * sizeof(int32_t), cudaMemcpyHostToDevice),
        fail_cleanup);

    /* Optional EFB data array */
    if (csr_data) {
        CUDA_CHECK_OOM(
            cudaMalloc(&ctx->d_data, (size_t)nnz * sizeof(uint8_t)),
            fail_cleanup);
        ctx->gpu_bytes_allocated += (size_t)nnz * sizeof(uint8_t);

        CUDA_CHECK_INIT(
            cudaMemcpy(ctx->d_data, csr_data,
                       (size_t)nnz * sizeof(uint8_t), cudaMemcpyHostToDevice),
            fail_cleanup);
    }

    /* ---- Allocate gradient/hessian device buffers ---- */
    {
        size_t g_bytes = (size_t)ctx->n_rows * ctx->num_classes * sizeof(double);
        CUDA_CHECK_OOM(cudaMalloc(&ctx->d_grad, g_bytes), fail_cleanup);
        CUDA_CHECK_OOM(cudaMalloc(&ctx->d_hess, g_bytes), fail_cleanup);
        ctx->gpu_bytes_allocated += g_bytes * 2;

        /* Pinned host buffers for async H2D */
        CUDA_CHECK_OOM(cudaMallocHost(&ctx->h_grad_pinned, g_bytes), fail_cleanup);
        CUDA_CHECK_OOM(cudaMallocHost(&ctx->h_hess_pinned, g_bytes), fail_cleanup);
    }

    /* ---- Allocate leaf row and partition buffers ---- */
    CUDA_CHECK_OOM(
        cudaMalloc(&ctx->d_leaf_rows, (size_t)ctx->n_rows * sizeof(int32_t)),
        fail_cleanup);
    ctx->gpu_bytes_allocated += (size_t)ctx->n_rows * sizeof(int32_t);

    CUDA_CHECK_OOM(
        cudaMalloc(&ctx->d_leaf_id, (size_t)ctx->n_rows * sizeof(int8_t)),
        fail_cleanup);
    ctx->gpu_bytes_allocated += (size_t)ctx->n_rows * sizeof(int8_t);

    CUDA_CHECK_OOM(
        cudaMalloc(&ctx->d_count_buf, 2 * sizeof(int32_t)),
        fail_cleanup);
    ctx->gpu_bytes_allocated += 2 * sizeof(int32_t);

    /* Initialize leaf_id to 0 (all rows in root) */
    CUDA_CHECK_INIT(
        cudaMemset(ctx->d_leaf_id, 0, (size_t)ctx->n_rows * sizeof(int8_t)),
        fail_cleanup);

    /* ---- Allocate 3 histogram buffers (build, parent, sibling) ---- */
    {
        size_t h_bytes = (size_t)ctx->hist_elements * sizeof(double);
        CUDA_CHECK_OOM(cudaMalloc(&ctx->d_hist, h_bytes), fail_cleanup);
        CUDA_CHECK_OOM(cudaMalloc(&ctx->d_hist_parent, h_bytes), fail_cleanup);
        CUDA_CHECK_OOM(cudaMalloc(&ctx->d_hist_sibling, h_bytes), fail_cleanup);
        ctx->gpu_bytes_allocated += h_bytes * 3;

        /* Pinned host buffer for histogram D2H */
        CUDA_CHECK_OOM(cudaMallocHost(&ctx->h_hist_pinned, h_bytes), fail_cleanup);
    }

    /* ---- Done ---- */
    ctx->initialized = 1;
    *handle = ctx;

    fprintf(stderr, "[gpu_hist] Init complete: %ld rows, %ld nnz, %d features, "
            "%d classes, %zu MB GPU allocated\n",
            (long)ctx->n_rows, (long)ctx->nnz, ctx->n_features,
            ctx->num_classes, ctx->gpu_bytes_allocated / (1024*1024));

    return GPU_HIST_OK;

fail_cleanup:
    /* Free everything that was allocated */
    if (ctx->d_indptr)       cudaFree(ctx->d_indptr);
    if (ctx->d_indices)      cudaFree(ctx->d_indices);
    if (ctx->d_data)         cudaFree(ctx->d_data);
    if (ctx->d_grad)         cudaFree(ctx->d_grad);
    if (ctx->d_hess)         cudaFree(ctx->d_hess);
    if (ctx->h_grad_pinned)  cudaFreeHost(ctx->h_grad_pinned);
    if (ctx->h_hess_pinned)  cudaFreeHost(ctx->h_hess_pinned);
    if (ctx->d_leaf_rows)    cudaFree(ctx->d_leaf_rows);
    if (ctx->d_leaf_id)      cudaFree(ctx->d_leaf_id);
    if (ctx->d_count_buf)    cudaFree(ctx->d_count_buf);
    if (ctx->d_hist)         cudaFree(ctx->d_hist);
    if (ctx->d_hist_parent)  cudaFree(ctx->d_hist_parent);
    if (ctx->d_hist_sibling) cudaFree(ctx->d_hist_sibling);
    if (ctx->h_hist_pinned)  cudaFreeHost(ctx->h_hist_pinned);
    for (int i = 0; i < ctx->num_streams; i++) {
        if (ctx->streams[i]) cudaStreamDestroy(ctx->streams[i]);
    }

fail_free_ctx:
    free(ctx);
    return result;
}


/* ---- Gradient Update ---- */

GpuHistError gpu_hist_update_gradients(
    GpuHistHandle handle,
    const double* gradients,
    const double* hessians
) {
    if (!handle || !handle->initialized) return GPU_HIST_NOT_INIT;
    if (!gradients || !hessians) return GPU_HIST_INVALID_ARG;

    GpuHistContext_* ctx = handle;
    size_t bytes = (size_t)ctx->n_rows * ctx->num_classes * sizeof(double);

    /* Copy to pinned staging buffers (fast, L1/L2 bandwidth) */
    memcpy(ctx->h_grad_pinned, gradients, bytes);
    memcpy(ctx->h_hess_pinned, hessians, bytes);

    /* Async H2D via pinned memory — overlaps with prior D2H or compute */
    cudaStream_t s = ctx->streams[ctx->stream_h2d];
    CUDA_CHECK(ctx, cudaMemcpyAsync(ctx->d_grad, ctx->h_grad_pinned, bytes,
                                    cudaMemcpyHostToDevice, s));
    CUDA_CHECK(ctx, cudaMemcpyAsync(ctx->d_hess, ctx->h_hess_pinned, bytes,
                                    cudaMemcpyHostToDevice, s));

    return GPU_HIST_OK;
}


/* ---- Build Histogram ---- */

GpuHistError gpu_hist_build(
    GpuHistHandle  handle,
    const int32_t* row_indices,
    int32_t        n_leaf_rows,
    int32_t        class_id,
    double*        output_histogram,
    int32_t        n_hist_bins
) {
    if (!handle || !handle->initialized) return GPU_HIST_NOT_INIT;
    if (!row_indices || !output_histogram) return GPU_HIST_INVALID_ARG;
    if (n_leaf_rows <= 0) return GPU_HIST_INVALID_ARG;
    if (class_id < 0 || class_id >= handle->num_classes) return GPU_HIST_INVALID_ARG;

    GpuHistContext_* ctx = handle;
    cudaStream_t s_h2d = ctx->streams[ctx->stream_h2d];
    cudaStream_t s_comp = ctx->streams[ctx->stream_compute];
    cudaStream_t s_d2h = ctx->streams[ctx->stream_d2h];

    /* Upload leaf row indices to GPU */
    CUDA_CHECK(ctx, cudaMemcpyAsync(ctx->d_leaf_rows, row_indices,
                                    (size_t)n_leaf_rows * sizeof(int32_t),
                                    cudaMemcpyHostToDevice, s_h2d));

    /* Ensure H2D (gradients + leaf rows) complete before compute */
    {
        cudaEvent_t h2d_done;
        CUDA_CHECK(ctx, cudaEventCreate(&h2d_done));
        CUDA_CHECK(ctx, cudaEventRecord(h2d_done, s_h2d));
        CUDA_CHECK(ctx, cudaStreamWaitEvent(s_comp, h2d_done, 0));
        CUDA_CHECK(ctx, cudaEventDestroy(h2d_done));
    }

    /* Zero histogram buffer on compute stream */
    int n_hist_elems = n_hist_bins * 2;
    hist_zero_kernel<<<grid_blocks(n_hist_elems), BLOCK_SIZE, 0, s_comp>>>(
        ctx->d_hist, n_hist_elems);

    int n_blocks = grid_blocks(n_leaf_rows);

    /* Choose kernel variant based on leaf size and GPU capabilities.
     *
     * Tiled kernel wins when:
     * 1. Shared memory is large enough for meaningful tiles (tile_size > 0)
     * 2. Enough rows that contention reduction matters (>= TILED_MIN_ROWS)
     * 3. Multiple tiles are needed (n_features > tile_size)
     *
     * For ultra-sparse data, global atomics usually win because contention
     * is negligible. But tiling helps on larger leaves (e.g., root node
     * with all rows). */
    bool use_tiled = (ctx->tile_size > 0
                      && n_leaf_rows >= TILED_MIN_ROWS
                      && ctx->total_hist_bins > ctx->tile_size);

    if (use_tiled) {
        /* Set max shared memory for this kernel (opt-in for sm_80+) */
        int tile_sz = ctx->tile_size;
        size_t max_smem = (size_t)tile_sz * 2 * sizeof(double);

        cudaFuncSetAttribute(sparse_hist_build_tiled_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)max_smem);

        /* Launch one tile at a time over the feature range */
        for (int32_t ts = 0; ts < ctx->total_hist_bins; ts += tile_sz) {
            int32_t te = ts + tile_sz;
            if (te > ctx->total_hist_bins) te = ctx->total_hist_bins;

            int actual_tile = te - ts;
            size_t smem = (size_t)actual_tile * 2 * sizeof(double);

            sparse_hist_build_tiled_kernel
                <<<n_blocks, BLOCK_SIZE, smem, s_comp>>>(
                ctx->d_indptr, ctx->d_indices, ctx->d_data,
                ctx->d_grad, ctx->d_hess,
                ctx->d_leaf_rows, n_leaf_rows,
                class_id, ctx->num_classes,
                ctx->d_hist, ts, te);
        }
    } else {
        /* Global memory atomic kernel — lower overhead, great for sparse */
        sparse_hist_build_kernel
            <<<n_blocks, BLOCK_SIZE, 0, s_comp>>>(
            ctx->d_indptr, ctx->d_indices, ctx->d_data,
            ctx->d_grad, ctx->d_hess,
            ctx->d_leaf_rows, n_leaf_rows,
            class_id, ctx->num_classes,
            ctx->d_hist);
    }

    /* Check for kernel launch errors */
    CUDA_CHECK(ctx, cudaGetLastError());

    /* D2H: copy histogram result to pinned host, then to caller's buffer.
     * Wait for compute to finish first. */
    {
        cudaEvent_t comp_done;
        CUDA_CHECK(ctx, cudaEventCreate(&comp_done));
        CUDA_CHECK(ctx, cudaEventRecord(comp_done, s_comp));
        CUDA_CHECK(ctx, cudaStreamWaitEvent(s_d2h, comp_done, 0));
        CUDA_CHECK(ctx, cudaEventDestroy(comp_done));
    }

    size_t hist_bytes = (size_t)n_hist_elems * sizeof(double);
    CUDA_CHECK(ctx, cudaMemcpyAsync(ctx->h_hist_pinned, ctx->d_hist,
                                    hist_bytes, cudaMemcpyDeviceToHost, s_d2h));

    /* Synchronize D2H and copy to caller's buffer */
    CUDA_CHECK(ctx, cudaStreamSynchronize(s_d2h));
    memcpy(output_histogram, ctx->h_hist_pinned, hist_bytes);

    return GPU_HIST_OK;
}


/* ---- Histogram Subtraction ---- */

GpuHistError gpu_hist_subtract(
    GpuHistHandle  handle,
    const double*  parent_hist,
    const double*  child_hist,
    double*        sibling_hist,
    int32_t        n_bins
) {
    if (!handle || !handle->initialized) return GPU_HIST_NOT_INIT;
    if (!parent_hist || !child_hist || !sibling_hist) return GPU_HIST_INVALID_ARG;
    if (n_bins <= 0) return GPU_HIST_INVALID_ARG;

    GpuHistContext_* ctx = handle;
    cudaStream_t s_comp = ctx->streams[ctx->stream_compute];
    cudaStream_t s_h2d  = ctx->streams[ctx->stream_h2d];
    cudaStream_t s_d2h  = ctx->streams[ctx->stream_d2h];

    int n_elements = n_bins * 2;  /* interleaved grad/hess */
    size_t bytes = (size_t)n_elements * sizeof(double);

    /* Upload parent and child histograms to GPU */
    CUDA_CHECK(ctx, cudaMemcpyAsync(ctx->d_hist_parent, parent_hist,
                                    bytes, cudaMemcpyHostToDevice, s_h2d));
    CUDA_CHECK(ctx, cudaMemcpyAsync(ctx->d_hist, child_hist,
                                    bytes, cudaMemcpyHostToDevice, s_h2d));

    /* Wait for uploads */
    {
        cudaEvent_t h2d_done;
        CUDA_CHECK(ctx, cudaEventCreate(&h2d_done));
        CUDA_CHECK(ctx, cudaEventRecord(h2d_done, s_h2d));
        CUDA_CHECK(ctx, cudaStreamWaitEvent(s_comp, h2d_done, 0));
        CUDA_CHECK(ctx, cudaEventDestroy(h2d_done));
    }

    /* Subtract on GPU */
    hist_subtract_kernel<<<grid_blocks(n_elements), BLOCK_SIZE, 0, s_comp>>>(
        ctx->d_hist_parent, ctx->d_hist, ctx->d_hist_sibling, n_elements);
    CUDA_CHECK(ctx, cudaGetLastError());

    /* D2H sibling result */
    {
        cudaEvent_t comp_done;
        CUDA_CHECK(ctx, cudaEventCreate(&comp_done));
        CUDA_CHECK(ctx, cudaEventRecord(comp_done, s_comp));
        CUDA_CHECK(ctx, cudaStreamWaitEvent(s_d2h, comp_done, 0));
        CUDA_CHECK(ctx, cudaEventDestroy(comp_done));
    }

    CUDA_CHECK(ctx, cudaMemcpyAsync(ctx->h_hist_pinned, ctx->d_hist_sibling,
                                    bytes, cudaMemcpyDeviceToHost, s_d2h));
    CUDA_CHECK(ctx, cudaStreamSynchronize(s_d2h));
    memcpy(sibling_hist, ctx->h_hist_pinned, bytes);

    return GPU_HIST_OK;
}


/* ---- Partition Update ---- */

GpuHistError gpu_hist_update_partition(
    GpuHistHandle handle,
    int32_t       old_leaf_id,
    int32_t       new_left_id,
    int32_t       new_right_id,
    int32_t       split_feature,
    float         split_threshold,
    int32_t*      left_count,
    int32_t*      right_count
) {
    if (!handle || !handle->initialized) return GPU_HIST_NOT_INIT;

    GpuHistContext_* ctx = handle;
    cudaStream_t s_comp = ctx->streams[ctx->stream_compute];

    /* Zero the count buffer */
    CUDA_CHECK(ctx, cudaMemsetAsync(ctx->d_count_buf, 0,
                                    2 * sizeof(int32_t), s_comp));

    int n_blocks = grid_blocks64(ctx->n_rows);

    partition_update_kernel<<<n_blocks, BLOCK_SIZE, 0, s_comp>>>(
        ctx->d_indptr, ctx->d_indices, ctx->d_data,
        ctx->d_leaf_id, (int32_t)ctx->n_rows,
        old_leaf_id, new_left_id, new_right_id,
        split_feature, split_threshold,
        ctx->d_count_buf);
    CUDA_CHECK(ctx, cudaGetLastError());

    /* Read back counts if requested */
    if (left_count || right_count) {
        int32_t counts[2] = {0, 0};
        CUDA_CHECK(ctx, cudaStreamSynchronize(s_comp));
        CUDA_CHECK(ctx, cudaMemcpy(counts, ctx->d_count_buf,
                                   2 * sizeof(int32_t), cudaMemcpyDeviceToHost));
        if (left_count)  *left_count  = counts[0];
        if (right_count) *right_count = counts[1];
    }

    return GPU_HIST_OK;
}


/* ---- Reset Partition ---- */

GpuHistError gpu_hist_reset_partition(GpuHistHandle handle) {
    if (!handle || !handle->initialized) return GPU_HIST_NOT_INIT;

    GpuHistContext_* ctx = handle;
    cudaStream_t s_comp = ctx->streams[ctx->stream_compute];

    int n_blocks = grid_blocks64(ctx->n_rows);
    partition_reset_kernel<<<n_blocks, BLOCK_SIZE, 0, s_comp>>>(
        ctx->d_leaf_id, (int32_t)ctx->n_rows);
    CUDA_CHECK(ctx, cudaGetLastError());

    return GPU_HIST_OK;
}


/* ---- Cleanup ---- */

GpuHistError gpu_hist_cleanup(GpuHistHandle handle) {
    if (!handle) return GPU_HIST_OK;

    GpuHistContext_* ctx = handle;

    /* Synchronize all streams before freeing */
    for (int i = 0; i < ctx->num_streams; i++) {
        if (ctx->streams[i]) cudaStreamSynchronize(ctx->streams[i]);
    }

    /* Free device memory */
    if (ctx->d_indptr)       cudaFree(ctx->d_indptr);
    if (ctx->d_indices)      cudaFree(ctx->d_indices);
    if (ctx->d_data)         cudaFree(ctx->d_data);
    if (ctx->d_grad)         cudaFree(ctx->d_grad);
    if (ctx->d_hess)         cudaFree(ctx->d_hess);
    if (ctx->d_leaf_rows)    cudaFree(ctx->d_leaf_rows);
    if (ctx->d_leaf_id)      cudaFree(ctx->d_leaf_id);
    if (ctx->d_count_buf)    cudaFree(ctx->d_count_buf);
    if (ctx->d_hist)         cudaFree(ctx->d_hist);
    if (ctx->d_hist_parent)  cudaFree(ctx->d_hist_parent);
    if (ctx->d_hist_sibling) cudaFree(ctx->d_hist_sibling);

    /* Free pinned host memory */
    if (ctx->h_grad_pinned)  cudaFreeHost(ctx->h_grad_pinned);
    if (ctx->h_hess_pinned)  cudaFreeHost(ctx->h_hess_pinned);
    if (ctx->h_hist_pinned)  cudaFreeHost(ctx->h_hist_pinned);

    /* Destroy streams */
    for (int i = 0; i < ctx->num_streams; i++) {
        if (ctx->streams[i]) cudaStreamDestroy(ctx->streams[i]);
    }

    ctx->initialized = 0;

    fprintf(stderr, "[gpu_hist] Cleanup: freed %zu MB GPU memory\n",
            ctx->gpu_bytes_allocated / (1024*1024));

    free(ctx);
    return GPU_HIST_OK;
}


/* ---- VRAM Usage Query ---- */

GpuHistError gpu_hist_get_vram_usage(
    GpuHistHandle handle,
    size_t*       used_bytes,
    size_t*       total_bytes
) {
    if (!handle || !handle->initialized) return GPU_HIST_NOT_INIT;
    if (!used_bytes || !total_bytes) return GPU_HIST_INVALID_ARG;

    *used_bytes  = handle->gpu_bytes_allocated;
    *total_bytes = handle->total_device_vram;
    return GPU_HIST_OK;
}


/* ---- Last Error String ---- */

const char* gpu_hist_get_last_error(GpuHistHandle handle) {
    if (!handle) {
        /* For init-phase errors, return the last CUDA error globally */
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return cudaGetErrorString(err);
        return "no error";
    }
    if (handle->last_error_msg[0] != '\0') {
        return handle->last_error_msg;
    }
    return "no error";
}


/* ---- Device Info Query ---- */

GpuHistError gpu_hist_get_device_info(
    GpuHistHandle handle,
    char*         device_name,
    int32_t       device_name_len,
    int32_t*      compute_major,
    int32_t*      compute_minor,
    int32_t*      sm_count,
    size_t*       shared_mem_per_sm
) {
    if (!handle || !handle->initialized) return GPU_HIST_NOT_INIT;

    GpuHistContext_* ctx = handle;

    if (device_name && device_name_len > 0) {
        strncpy(device_name, ctx->device_name,
                (size_t)(device_name_len - 1));
        device_name[device_name_len - 1] = '\0';
    }
    if (compute_major) *compute_major = ctx->compute_major;
    if (compute_minor) *compute_minor = ctx->compute_minor;
    if (sm_count)      *sm_count = ctx->sm_count;
    if (shared_mem_per_sm) *shared_mem_per_sm = (size_t)ctx->smem_bytes_per_block;

    return GPU_HIST_OK;
}


} /* extern "C" */
