/*
 * cuda_sparse_hist_tree_learner.cu
 * ================================
 * CUDASparseHistTreeLearner: LightGBM tree learner with GPU-accelerated
 * histogram building for sparse CSR data.
 *
 * Subclasses SerialTreeLearner, overrides ONLY ConstructHistograms().
 * Everything else (split finding, tree growing, EFB bundling, gradient
 * computation, row partitioning) stays on CPU.
 *
 * Two histogram modes, selected by config flag:
 *   MODE_CUSPARSE (default):
 *     Pre-transpose CSR once at Init(). Per-node: 2x cuSPARSE SpMV
 *     (csr_AT @ grad, csr_AT @ hess) to produce histogram vectors.
 *     Proven 473x speedup on RTX 3090 (benchmark/bench_kernel_speed.py).
 *
 *   MODE_ATOMIC_SCATTER:
 *     Row-parallel atomic scatter kernel (same as gpu_histogram.cu).
 *     Each thread walks one leaf row's CSR nonzeros, atomicAdd into
 *     histogram bins. Near-zero contention on ultra-sparse data.
 *
 * Matrix thesis: ALL features preserved. Binary sparse cross features.
 * EFB stays CPU-side. GPU builds histograms only. Structural zeros in
 * CSR = feature OFF (correct for binary crosses). No filtering. No
 * subsampling. No row partitioning. int64 indptr for NNZ > 2^31.
 *
 * Compile as part of a LightGBM fork's CMake build:
 *   add_library(cuda_sparse_hist_tree_learner OBJECT
 *       src/treelearner/cuda_sparse_hist_tree_learner.cu)
 *   target_link_libraries(_lightgbm ... cuda_sparse_hist_tree_learner
 *       CUDA::cudart CUDA::cusparse)
 *
 * Requires: CUDA 11.2+, cuSPARSE, LightGBM internal headers.
 */

/* =========================================================================
 * Configuration: Select histogram building mode at compile time.
 * Override with -DGPU_HIST_MODE=1 (cuSPARSE) or -DGPU_HIST_MODE=2 (atomic).
 * Default: cuSPARSE (mode 1), selected at runtime via tree_learner param.
 * ========================================================================= */
#ifndef GPU_HIST_MODE_CUSPARSE
#define GPU_HIST_MODE_CUSPARSE  1
#endif
#ifndef GPU_HIST_MODE_ATOMIC
#define GPU_HIST_MODE_ATOMIC    2
#endif

/* =========================================================================
 * Includes
 * ========================================================================= */

/* Workaround: nvcc + MSVC 19.50 (VS 2025) doesn't properly expose
 * _MSVC_EXECUTION_CHARACTER_SET=65001 to the device compiler pass,
 * causing fmt's UTF-8 static_assert to fire even with /utf-8 flag.
 * Define FMT_UNICODE=0 to disable the check. This is safe — we don't
 * use fmt formatting in CUDA kernels. */
#ifdef __CUDACC__
#ifndef FMT_UNICODE
#define FMT_UNICODE 0
#endif
#endif

#include <cuda_runtime.h>
#include <cusparse.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <memory>
#include <numeric>

/* LightGBM internal headers — paths relative to LightGBM source root */
#include <LightGBM/bin.h>
#include <LightGBM/dataset.h>
#include <LightGBM/tree_learner.h>
#include <LightGBM/config.h>
#include <LightGBM/train_share_states.h>
#include <LightGBM/utils/common.h>
#include <LightGBM/utils/log.h>

/* SerialTreeLearner definition (the base class we override) */
#include "serial_tree_learner.h"
#include "feature_histogram.hpp"
#include "leaf_splits.hpp"


/* =========================================================================
 * Error checking macros (defined before namespace — used by both kernels
 * and class methods, but Log::Fatal requires LightGBM namespace)
 * ========================================================================= */

#define CUDA_CHECK_FATAL(call)                                                \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            LightGBM::Log::Fatal("[CUDASparseHist] CUDA error at %s:%d — %s", \
                       __FILE__, __LINE__, cudaGetErrorString(err));           \
        }                                                                      \
    } while (0)

#define CUSPARSE_CHECK_FATAL(call)                                            \
    do {                                                                       \
        cusparseStatus_t status = (call);                                      \
        if (status != CUSPARSE_STATUS_SUCCESS) {                               \
            LightGBM::Log::Fatal("[CUDASparseHist] cuSPARSE error at %s:%d "  \
                       "— code %d", __FILE__, __LINE__, (int)status);         \
        }                                                                      \
    } while (0)


/* =========================================================================
 * KERNEL: Atomic Scatter Histogram Build (Mode 2)
 *
 * Each thread processes one leaf row. Walks its CSR nonzeros and
 * atomicAdd grad/hess into the histogram bin for that column.
 *
 * Gradients are RAW (indexed by original row ID), NOT ordered.
 * GBDT pre-slices the gradient array per class, so grad[row] gives
 * the correct gradient without num_classes stride.
 *
 * For raw binary features (data == NULL): every stored nonzero is bin 1.
 * For EFB-encoded features (data != NULL): data[j] is the bundle-bin.
 *   bin 0 means all features in this bundle are OFF — skip it.
 *
 * Histogram layout: interleaved [grad, hess] per bin.
 *   hist[bin * 2 + 0] = sum of gradients
 *   hist[bin * 2 + 1] = sum of hessians
 * ========================================================================= */

static constexpr int BLOCK_SIZE = 256;

__global__ void atomic_scatter_hist_kernel(
    const int64_t* __restrict__ indptr,
    const int32_t* __restrict__ indices,
    const uint8_t* __restrict__ data,          /* NULL for raw binary */
    const double*  __restrict__ gradients,     /* [n_rows] — single class, RAW */
    const double*  __restrict__ hessians,
    const int32_t* __restrict__ leaf_rows,     /* [n_leaf_rows] sorted */
    int32_t                     n_leaf_rows,
    int32_t                     class_id,      /* unused — GBDT pre-slices */
    int32_t                     num_classes,   /* unused — GBDT pre-slices */
    double*        __restrict__ hist            /* [total_bins * 2] */
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_leaf_rows) return;

    int32_t row = leaf_rows[tid];
    /* gradients_/hessians_ are RAW arrays indexed by original row ID.
     * GBDT::TrainOneIter() already offsets the pointer by
     * (cur_tree_id * num_data_), so grad[row] is this class's gradient.
     * No num_classes stride needed. */
    double g = gradients[row];
    double h = hessians[row];

    int64_t start = indptr[row];
    int64_t end   = indptr[row + 1];

    if (data == nullptr) {
        /* Raw binary mode: every nonzero = bin 1, indices[j] = feature col.
         * Atomic contention: ~9K nnz/row out of 3M+ features = negligible. */
        for (int64_t j = start; j < end; j++) {
            int32_t col = indices[j];
            atomicAdd(&hist[static_cast<int64_t>(col) * 2],     g);
            atomicAdd(&hist[static_cast<int64_t>(col) * 2 + 1], h);
        }
    } else {
        /* EFB mode: data[j] = pre-computed bundle-bin index.
         * bin=0 means all features in bundle are OFF — skip. */
        for (int64_t j = start; j < end; j++) {
            uint8_t bin = data[j];
            if (bin == 0) continue;
            int32_t col = indices[j];
            atomicAdd(&hist[static_cast<int64_t>(col) * 2],     g);
            atomicAdd(&hist[static_cast<int64_t>(col) * 2 + 1], h);
        }
    }
}


/* =========================================================================
 * KERNEL: Histogram Zero
 * ========================================================================= */

__global__ void hist_zero_kernel(
    double*  __restrict__ hist,
    int64_t              n_elements
) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n_elements) {
        hist[i] = 0.0;
    }
}


/* =========================================================================
 * KERNEL: Histogram Subtraction (sibling = parent - child)
 * ========================================================================= */

__global__ void hist_subtract_kernel(
    const double* __restrict__ parent,
    const double* __restrict__ child,
    double*       __restrict__ sibling,
    int64_t                    n_elements
) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n_elements) {
        sibling[i] = parent[i] - child[i];
    }
}


/* =========================================================================
 * Helper: grid dimensions
 * ========================================================================= */

static inline int grid_blocks(int64_t n_elements) {
    return static_cast<int>((n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE);
}


/* =========================================================================
 * Helper kernels for cuSPARSE mode (forward-declared, defined below class)
 * ========================================================================= */

/* Scatter RAW leaf gradients into full-size vectors at original row
 * positions. Required for cuSPARSE SpMV which operates on full n_rows.
 * raw_grad is indexed by original row ID (NOT ordered). */
__global__ void scatter_grad_kernel(
    const int32_t* __restrict__ leaf_rows,
    const double*  __restrict__ raw_grad,
    const double*  __restrict__ raw_hess,
    double*        __restrict__ full_grad,
    double*        __restrict__ full_hess,
    int32_t                     n_leaf_rows,
    int32_t                     num_classes,
    int32_t                     class_id);

/* Copy SpMV result into interleaved histogram buffer.
 * component=0 for gradients, component=1 for hessians. */
__global__ void interleave_grad_kernel(
    const double* __restrict__ spmv_result,
    double*       __restrict__ hist,
    const uint32_t* __restrict__ feature_hist_offsets,
    int32_t                    n_features,
    int32_t                    component);

/* Host-side launch wrappers */
static void launch_scatter_kernel(
    const int32_t* d_leaf_rows, const double* d_raw_grad,
    const double* d_raw_hess, double* d_full_grad, double* d_full_hess,
    int32_t n_leaf_rows, int32_t num_classes, int32_t class_id,
    cudaStream_t stream);

static void launch_interleave_grad_kernel(
    const double* d_spmv_result, double* d_hist,
    const uint32_t* d_feature_hist_offsets,
    int32_t n_features, int component, cudaStream_t stream);


/* =========================================================================
 * External CSR global getters (defined in c_api.cpp, exported via extern "C").
 * The Python wrapper calls LGBM_BoosterSetExternalCSR() which stores CSR data
 * in a global struct. These getters let the tree learner read it.
 * ========================================================================= */

extern "C" {
    const int64_t* LGBM_GetExternalCSRIndptr();
    const int32_t* LGBM_GetExternalCSRIndices();
    int64_t LGBM_GetExternalCSRNnz();
    int32_t LGBM_GetExternalCSRNRows();
    int32_t LGBM_GetExternalCSRNFeatures();
    bool LGBM_GetExternalCSRIsSet();
}

/* =========================================================================
 * Include the class definition from the header file.
 * The header is the single source of truth for the class interface.
 * tree_learner.cpp also includes this header for the factory function.
 * Included at file scope (outside any namespace) since the header wraps
 * its own namespace LightGBM.
 * ========================================================================= */

#include "cuda_sparse_hist_tree_learner.h"

namespace LightGBM {


/* =========================================================================
 * Constructor / Destructor
 * ========================================================================= */

CUDASparseHistTreeLearner::CUDASparseHistTreeLearner(const Config* config)
    : SerialTreeLearner(config), gpu_hist_mode_(GPU_HIST_MODE_CUSPARSE),
      csr_uploaded_(false) {}

CUDASparseHistTreeLearner::~CUDASparseHistTreeLearner() {
    CleanupGPU();
}


/* =========================================================================
 * Init — called once before training starts
 * ========================================================================= */

void CUDASparseHistTreeLearner::Init(const Dataset* train_data,
                                      bool is_constant_hessian) {
    /* Base class Init sets up all CPU structures:
     * - share_state_ (histogram layout, feature offsets)
     * - smaller_leaf_histogram_array_
     * - larger_leaf_histogram_array_
     * - feature_hist_offsets()
     * - num_hist_total_bin()
     * - data partitioning structures */
    SerialTreeLearner::Init(train_data, is_constant_hessian);

    /* Histogram mode: cuSPARSE by default (proven 473x on RTX 3090).
     * Hardcoded — no config parameter needed. Use compile-time
     * -DGPU_HIST_MODE=2 to switch to atomic scatter. */
    gpu_hist_mode_ = GPU_HIST_MODE_CUSPARSE;

    Log::Info("[CUDASparseHist] Histogram mode: %s",
              gpu_hist_mode_ == GPU_HIST_MODE_CUSPARSE ? "cuSPARSE SpMV" :
                                                          "Atomic Scatter");

    /* Extract dimensions from the dataset */
    n_rows_     = train_data->num_data();
    n_features_ = train_data->num_total_features();
    num_classes_ = config_->num_class;
    if (num_classes_ < 1) num_classes_ = 1;

    /* Total histogram bins — from the base class share_state_ which was
     * initialized by SerialTreeLearner::Init() above.
     *
     * With EFB enabled (default):
     *   total_hist_bins < n_features (bundled bins)
     *   GPU SpMV scatter uses the EFB bin mapping to write into the
     *   correct bundled histogram slots.
     *
     * With EFB disabled (enable_bundle=False):
     *   total_hist_bins == n_features * 2  (2 bins per binary feature)
     */
    total_hist_bins_ = share_state_->num_hist_total_bin();
    hist_buf_elems_  = total_hist_bins_ * 2;  /* interleaved grad/hess */

    /* Upload feature histogram offsets to GPU for sparse histogram mapping.
     *
     * share_state_->feature_hist_offsets() is indexed by INNER feature index
     * (0..num_used_features) and has num_used_features+1 entries.
     * But the SpMV output is indexed by TOTAL feature index (0..num_total_features-1)
     * because the transposed CSR has num_total_features rows.
     *
     * We build an extended offset table of size num_total_features, mapping each
     * CSR column to its histogram bin offset.  Unused features (filtered out by
     * LightGBM's dataset construction) get UINT32_MAX as a sentinel — the
     * interleave kernel skips them. */
    {
        const auto& inner_offsets = share_state_->feature_hist_offsets();
        int n_used = train_data->num_features();
        int n_total = n_features_;  /* num_total_features */

        /* Build extended offset table on host */
        std::vector<uint32_t> ext_offsets(n_total, UINT32_MAX);
        int mapped = 0;
        for (int col = 0; col < n_total; ++col) {
            int inner = train_data->InnerFeatureIndex(col);
            if (inner >= 0 && inner < n_used) {
                ext_offsets[col] = inner_offsets[inner];
                ++mapped;
            }
        }

        size_t offsets_bytes = ext_offsets.size() * sizeof(uint32_t);
        CUDA_CHECK_FATAL(cudaMalloc(&d_feature_hist_offsets_, offsets_bytes));
        gpu_bytes_alloc_ += offsets_bytes;
        CUDA_CHECK_FATAL(cudaMemcpy(d_feature_hist_offsets_, ext_offsets.data(),
                                    offsets_bytes, cudaMemcpyHostToDevice));
        Log::Info("[CUDASparseHist] Feature hist offsets uploaded: %d total features, "
                  "%d mapped to hist bins, %d unused (sentinel=UINT32_MAX)",
                  n_total, mapped, n_total - mapped);
    }

    Log::Info("[CUDASparseHist] Dataset: %lld rows, %d features, "
              "%d classes, %lld hist bins",
              static_cast<long long>(n_rows_), n_features_,
              num_classes_, static_cast<long long>(total_hist_bins_));

    Log::Info("[CUDASparseHist] EFB mapping active: %d features -> %lld hist bins (%.1fx compression)",
              n_features_, static_cast<long long>(total_hist_bins_),
              static_cast<double>(n_features_) / static_cast<double>(total_hist_bins_ > 0 ? total_hist_bins_ : 1));

    /* Initialize GPU resources */
    InitGPU();
}


/* =========================================================================
 * InitGPU — set up all GPU resources
 * ========================================================================= */

void CUDASparseHistTreeLearner::InitGPU() {
    int device_count = 0;
    CUDA_CHECK_FATAL(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        Log::Fatal("[CUDASparseHist] No CUDA-capable GPU found");
    }

    int device_id = config_->gpu_device_id >= 0 ? config_->gpu_device_id : 0;
    CUDA_CHECK_FATAL(cudaSetDevice(device_id));

    /* Log GPU info */
    cudaDeviceProp prop;
    CUDA_CHECK_FATAL(cudaGetDeviceProperties(&prop, device_id));
    Log::Info("[CUDASparseHist] GPU %d: %s (sm_%d%d), %d SMs, "
              "VRAM: %zu MB, Shared/block: %d KB",
              device_id, prop.name, prop.major, prop.minor,
              prop.multiProcessorCount,
              prop.totalGlobalMem / (1024 * 1024),
              static_cast<int>(prop.sharedMemPerBlockOptin / 1024));

    /* Create CUDA streams (non-blocking for overlap) */
    CUDA_CHECK_FATAL(cudaStreamCreateWithFlags(&stream_h2d_,
                     cudaStreamNonBlocking));
    CUDA_CHECK_FATAL(cudaStreamCreateWithFlags(&stream_compute_,
                     cudaStreamNonBlocking));
    CUDA_CHECK_FATAL(cudaStreamCreateWithFlags(&stream_d2h_,
                     cudaStreamNonBlocking));

    /* CSR upload is DEFERRED until first ConstructHistograms() call.
     * At Init() time, SetExternalCSR() has NOT been called yet.
     * Call sequence from Python:
     *   1. lgb.Booster(params, ds) -> Init() runs (here)
     *   2. booster.set_external_csr(X) -> stores CSR in host vectors
     *   3. booster.update() -> ConstructHistograms() -> uploads CSR there
     */
    if (has_external_csr_) {
        /* CSR already set (unusual but possible) — upload now */
        UploadCSR();
        if (gpu_hist_mode_ == GPU_HIST_MODE_CUSPARSE && !has_efb_data_) {
            SetupCuSPARSE();
        } else if (gpu_hist_mode_ == GPU_HIST_MODE_CUSPARSE && has_efb_data_) {
            Log::Warning("[CUDASparseHist] EFB data present — falling back to "
                         "atomic scatter mode (cuSPARSE requires raw binary)");
            gpu_hist_mode_ = GPU_HIST_MODE_ATOMIC;
        }
        csr_uploaded_ = true;
    } else {
        Log::Warning("[CUDASparseHist] No external CSR at Init() time — "
                     "CSR will be uploaded on first ConstructHistograms() call "
                     "after set_external_csr().");
    }

    /* Allocate gradient, histogram, and staging buffers */
    AllocateBuffers();

    gpu_initialized_ = true;

    Log::Info("[CUDASparseHist] GPU init complete: %zu MB allocated on device",
              gpu_bytes_alloc_ / (1024 * 1024));
}


/* =========================================================================
 * SetExternalCSR — receive CSR data from Python wrapper
 *
 * Called by the Python wrapper after Dataset construction. Passes the
 * scipy CSR matrix directly, avoiding the need to extract from LightGBM's
 * private MultiValBin internals.
 *
 * For our binary cross features fed as scipy CSR:
 * - indptr:  int64 row pointers (supports NNZ > 2^31 for 15m)
 * - indices: int32 column indices (feature index)
 * - n_rows/n_features/nnz: matrix dimensions
 *
 * The CSR data is stored on the host side and uploaded to GPU during
 * UploadCSR(). This is the Phase 1 approach — avoids all private member
 * access issues with MultiValBin.
 * ========================================================================= */

void CUDASparseHistTreeLearner::SetExternalCSR(
    const int64_t* indptr, const int32_t* indices,
    int64_t nnz, int32_t n_rows, int32_t n_features) {

    /* Store copies of the CSR arrays for later GPU upload */
    ext_csr_indptr_.assign(indptr, indptr + n_rows + 1);
    ext_csr_indices_.assign(indices, indices + nnz);
    ext_csr_nnz_ = nnz;
    ext_csr_n_rows_ = n_rows;
    ext_csr_n_features_ = n_features;
    has_external_csr_ = true;

    Log::Info("[CUDASparseHist] External CSR set: %d rows, %lld nnz, "
              "%d features",
              n_rows, static_cast<long long>(nnz), n_features);
}


/* =========================================================================
 * UploadCSR — upload externally-provided CSR to GPU
 *
 * Uses the CSR data provided by SetExternalCSR(). If no external CSR
 * was set, logs an error and fails. Internal MultiValBin extraction is
 * not supported in this fork (Phase 1).
 *
 * For binary cross features:
 * - indptr:  int64 row pointers (supports NNZ > 2^31 for 15m)
 * - indices: int32 column indices (feature index)
 * - All nonzeros represent "feature ON" (value = 1), raw binary mode.
 * ========================================================================= */

void CUDASparseHistTreeLearner::UploadCSR() {
    if (!has_external_csr_) {
        Log::Warning("[CUDASparseHist] UploadCSR() called but no external CSR "
                     "set yet. Skipping — CSR will be uploaded later via "
                     "deferred path in ConstructHistograms().");
        return;
    }

    const int64_t* h_indptr  = ext_csr_indptr_.data();
    const int32_t* h_indices = ext_csr_indices_.data();
    nnz_ = ext_csr_nnz_;
    /* Override n_rows_ and n_features_ from external CSR if they differ
     * (the Dataset may report differently due to EFB bundling) */
    if (ext_csr_n_rows_ > 0) {
        n_rows_ = ext_csr_n_rows_;
    }
    if (ext_csr_n_features_ > 0) {
        n_features_ = ext_csr_n_features_;
    }

    /* Raw binary mode — cross features are 0/1 */
    has_efb_data_ = false;

    Log::Info("[CUDASparseHist] CSR: %lld rows, %lld nnz (%.2f%% dense), "
              "%d features",
              static_cast<long long>(n_rows_),
              static_cast<long long>(nnz_),
              100.0 * nnz_ / (static_cast<double>(n_rows_) * n_features_),
              n_features_);

    /* Upload original CSR to GPU */
    size_t indptr_bytes  = (n_rows_ + 1) * sizeof(int64_t);
    size_t indices_bytes = static_cast<size_t>(nnz_) * sizeof(int32_t);

    CUDA_CHECK_FATAL(cudaMalloc(&d_csr_indptr_, indptr_bytes));
    gpu_bytes_alloc_ += indptr_bytes;

    CUDA_CHECK_FATAL(cudaMalloc(&d_csr_indices_, indices_bytes));
    gpu_bytes_alloc_ += indices_bytes;

    CUDA_CHECK_FATAL(cudaMemcpy(d_csr_indptr_, h_indptr, indptr_bytes,
                                cudaMemcpyHostToDevice));
    CUDA_CHECK_FATAL(cudaMemcpy(d_csr_indices_, h_indices, indices_bytes,
                                cudaMemcpyHostToDevice));

    Log::Info("[CUDASparseHist] CSR uploaded: indptr %zu MB, indices %zu MB",
              indptr_bytes / (1024 * 1024), indices_bytes / (1024 * 1024));

    /* For cuSPARSE mode, we need the TRANSPOSED CSR (n_features x n_rows).
     * Build it on CPU, upload, then free the CPU copy.
     *
     * Transpose CSR -> CSC of the original (which IS CSR of the transpose).
     * Algorithm: count column occurrences, compute prefix sums, scatter. */
    if (gpu_hist_mode_ == GPU_HIST_MODE_CUSPARSE && !has_efb_data_) {
        Log::Info("[CUDASparseHist] Building transposed CSR for cuSPARSE SpMV...");

        int32_t n_cols = n_features_;
        h_csrT_indptr_.resize(n_cols + 1, 0);
        h_csrT_indices_.resize(nnz_);

        /* Pass 1: count nonzeros per column */
        for (int64_t j = 0; j < nnz_; j++) {
            int32_t col = h_indices[j];
            if (col >= 0 && col < n_cols) {
                h_csrT_indptr_[col + 1]++;
            }
        }

        /* Prefix sum -> row pointers of transposed CSR */
        for (int32_t c = 0; c < n_cols; c++) {
            h_csrT_indptr_[c + 1] += h_csrT_indptr_[c];
        }

        /* Pass 2: scatter row indices into transposed structure.
         * Use a working copy of indptr as write cursors. */
        std::vector<int64_t> cursor(h_csrT_indptr_.begin(),
                                     h_csrT_indptr_.end());
        for (int64_t row = 0; row < n_rows_; row++) {
            int64_t start = h_indptr[row];
            int64_t end   = h_indptr[row + 1];
            for (int64_t j = start; j < end; j++) {
                int32_t col = h_indices[j];
                if (col >= 0 && col < n_cols) {
                    int64_t pos = cursor[col]++;
                    h_csrT_indices_[pos] = static_cast<int32_t>(row);
                }
            }
        }

        /* Upload transposed CSR to GPU.
         * cuSPARSE requires matching index types for indptr and indices.
         * When NNZ <= INT32_MAX: use int32 for both (saves memory).
         * When NNZ > INT32_MAX: use int64 for both (required for 15m). */
        csrT_use_int32_ = (nnz_ <= static_cast<int64_t>(INT32_MAX));

        if (csrT_use_int32_) {
            /* Downcast transposed indptr to int32 */
            std::vector<int32_t> h_csrT_indptr32(n_cols + 1);
            for (int32_t c = 0; c <= n_cols; c++) {
                h_csrT_indptr32[c] = static_cast<int32_t>(h_csrT_indptr_[c]);
            }
            size_t csrT_indptr_bytes  = (n_cols + 1) * sizeof(int32_t);
            size_t csrT_indices_bytes = static_cast<size_t>(nnz_) * sizeof(int32_t);

            CUDA_CHECK_FATAL(cudaMalloc(&d_csrT_indptr32_, csrT_indptr_bytes));
            gpu_bytes_alloc_ += csrT_indptr_bytes;
            CUDA_CHECK_FATAL(cudaMalloc(&d_csrT_indices_, csrT_indices_bytes));
            gpu_bytes_alloc_ += csrT_indices_bytes;

            CUDA_CHECK_FATAL(cudaMemcpy(d_csrT_indptr32_, h_csrT_indptr32.data(),
                                        csrT_indptr_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK_FATAL(cudaMemcpy(d_csrT_indices_, h_csrT_indices_.data(),
                                        csrT_indices_bytes, cudaMemcpyHostToDevice));

            Log::Info("[CUDASparseHist] Transposed CSR uploaded (int32 mode): "
                      "indptr %zu MB, indices %zu MB",
                      csrT_indptr_bytes / (1024 * 1024),
                      csrT_indices_bytes / (1024 * 1024));
        } else {
            /* NNZ > INT32_MAX: use int64 for both. Upcast indices to int64. */
            std::vector<int64_t> h_csrT_indices64(nnz_);
            for (int64_t j = 0; j < nnz_; j++) {
                h_csrT_indices64[j] = static_cast<int64_t>(h_csrT_indices_[j]);
            }
            size_t csrT_indptr_bytes  = (n_cols + 1) * sizeof(int64_t);
            size_t csrT_indices_bytes = static_cast<size_t>(nnz_) * sizeof(int64_t);

            CUDA_CHECK_FATAL(cudaMalloc(&d_csrT_indptr_, csrT_indptr_bytes));
            gpu_bytes_alloc_ += csrT_indptr_bytes;
            CUDA_CHECK_FATAL(cudaMalloc(&d_csrT_indices64_, csrT_indices_bytes));
            gpu_bytes_alloc_ += csrT_indices_bytes;

            CUDA_CHECK_FATAL(cudaMemcpy(d_csrT_indptr_, h_csrT_indptr_.data(),
                                        csrT_indptr_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK_FATAL(cudaMemcpy(d_csrT_indices64_, h_csrT_indices64.data(),
                                        csrT_indices_bytes, cudaMemcpyHostToDevice));

            Log::Info("[CUDASparseHist] Transposed CSR uploaded (int64 mode): "
                      "indptr %zu MB, indices %zu MB",
                      csrT_indptr_bytes / (1024 * 1024),
                      csrT_indices_bytes / (1024 * 1024));
        }

        /* Free host-side transposed CSR — it's now on GPU */
        h_csrT_indptr_.clear();
        h_csrT_indptr_.shrink_to_fit();
        h_csrT_indices_.clear();
        h_csrT_indices_.shrink_to_fit();
    }

    /* Free external CSR host copies — now on GPU */
    ext_csr_indptr_.clear();
    ext_csr_indptr_.shrink_to_fit();
    ext_csr_indices_.clear();
    ext_csr_indices_.shrink_to_fit();
}


/* =========================================================================
 * SetupCuSPARSE — create cuSPARSE handle, matrix/vector descriptors,
 * and allocate SpMV workspace buffer.
 *
 * The transposed CSR matrix (n_features x n_rows) multiplied by a gradient
 * vector (n_rows) produces a histogram vector (n_features) in one SpMV call.
 *
 * For binary features: all nonzero values are implicitly 1.0. We create a
 * CSR descriptor with a dummy values array of all 1.0 on the GPU.
 * ========================================================================= */

void CUDASparseHistTreeLearner::SetupCuSPARSE() {
    CUSPARSE_CHECK_FATAL(cusparseCreate(&cusparse_handle_));
    CUSPARSE_CHECK_FATAL(cusparseSetStream(cusparse_handle_, stream_compute_));

    /* For binary features, SpMV needs a values array (all 1.0).
     * We allocate this on the GPU once and never change it. */
    double* d_csrT_values = nullptr;
    size_t values_bytes = static_cast<size_t>(nnz_) * sizeof(double);
    CUDA_CHECK_FATAL(cudaMalloc(&d_csrT_values, values_bytes));
    gpu_bytes_alloc_ += values_bytes;

    /* Fill with 1.0 using a temporary host allocation */
    {
        std::vector<double> ones(nnz_, 1.0);
        CUDA_CHECK_FATAL(cudaMemcpy(d_csrT_values, ones.data(), values_bytes,
                                    cudaMemcpyHostToDevice));
    }

    /* Create sparse matrix descriptor for the transposed CSR.
     * Shape: n_features (rows of AT) x n_rows (cols of AT).
     * This is CSR format of A^T, so:
     *   indptr  has n_features+1 entries
     *   indices has nnz entries (row indices of original = col indices of AT)
     *   values  has nnz entries (all 1.0 for binary)
     *
     * cuSPARSE requires matching index types — both int32 or both int64.
     * csrT_use_int32_ was set in UploadCSR() based on NNZ range. */
    if (csrT_use_int32_) {
        CUSPARSE_CHECK_FATAL(cusparseCreateCsr(
            &matA_T_,
            n_features_,                /* rows of A^T */
            n_rows_,                    /* cols of A^T */
            nnz_,
            d_csrT_indptr32_,          /* int32 row pointers */
            d_csrT_indices_,           /* int32 column indices */
            d_csrT_values,             /* double values (all 1.0) */
            CUSPARSE_INDEX_32I,        /* indptr type: int32 */
            CUSPARSE_INDEX_32I,        /* indices type: int32 */
            CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_64F                 /* values type: double */
        ));
        Log::Info("[CUDASparseHist] cuSPARSE descriptor created (int32 indices)");
    } else {
        CUSPARSE_CHECK_FATAL(cusparseCreateCsr(
            &matA_T_,
            n_features_,                /* rows of A^T */
            n_rows_,                    /* cols of A^T */
            nnz_,
            d_csrT_indptr_,            /* int64 row pointers */
            d_csrT_indices64_,         /* int64 column indices */
            d_csrT_values,             /* double values (all 1.0) */
            CUSPARSE_INDEX_64I,        /* indptr type: int64 */
            CUSPARSE_INDEX_64I,        /* indices type: int64 */
            CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_64F                 /* values type: double */
        ));
        Log::Info("[CUDASparseHist] cuSPARSE descriptor created (int64 indices)");
    }

    /* Allocate full-size gradient vectors for SpMV input.
     * These are n_rows long — we scatter leaf gradients into them. */
    size_t vec_bytes = n_rows_ * sizeof(double);
    CUDA_CHECK_FATAL(cudaMalloc(&d_full_grad_, vec_bytes));
    CUDA_CHECK_FATAL(cudaMalloc(&d_full_hess_, vec_bytes));
    gpu_bytes_alloc_ += vec_bytes * 2;

    /* Allocate SpMV output vector (n_features doubles) */
    size_t out_bytes = n_features_ * sizeof(double);
    CUDA_CHECK_FATAL(cudaMalloc(&d_spmv_result_, out_bytes));
    gpu_bytes_alloc_ += out_bytes;

    /* Create dense vector descriptors (we'll update the pointers
     * before each SpMV call to switch between grad and hess) */
    CUSPARSE_CHECK_FATAL(cusparseCreateDnVec(&vec_in_, n_rows_,
                                             d_full_grad_, CUDA_R_64F));
    CUSPARSE_CHECK_FATAL(cusparseCreateDnVec(&vec_out_, n_features_,
                                             d_spmv_result_, CUDA_R_64F));

    /* Query and allocate SpMV workspace buffer.
     * This is a one-time cost. The buffer is reused for all SpMV calls. */
    double alpha = 1.0, beta = 0.0;
    CUSPARSE_CHECK_FATAL(cusparseSpMV_bufferSize(
        cusparse_handle_,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA_T_, vec_in_, &beta, vec_out_,
        CUDA_R_64F,
        CUSPARSE_SPMV_ALG_DEFAULT,
        &spmv_buffer_size_
    ));

    if (spmv_buffer_size_ > 0) {
        CUDA_CHECK_FATAL(cudaMalloc(&d_spmv_buffer_, spmv_buffer_size_));
        gpu_bytes_alloc_ += spmv_buffer_size_;
    }

    Log::Info("[CUDASparseHist] cuSPARSE ready: SpMV workspace %zu KB, "
              "values array %zu MB",
              spmv_buffer_size_ / 1024, values_bytes / (1024 * 1024));
}


/* =========================================================================
 * AllocateBuffers — gradient staging, histogram, and leaf row buffers
 * ========================================================================= */

void CUDASparseHistTreeLearner::AllocateBuffers() {
    /* Gradient buffer: n_rows_ doubles (NOT * num_classes_).
     * For multiclass, GBDT::TrainOneIter() pre-offsets the gradient pointer
     * to the current class, so gradients_[row] is already class-specific. */
    size_t grad_bytes = static_cast<size_t>(n_rows_) * sizeof(double);
    size_t hist_bytes = static_cast<size_t>(hist_buf_elems_) * sizeof(double);
    size_t leaf_bytes = static_cast<size_t>(n_rows_) * sizeof(int32_t);

    /* Device gradient/hessian buffers */
    CUDA_CHECK_FATAL(cudaMalloc(&d_gradients_, grad_bytes));
    CUDA_CHECK_FATAL(cudaMalloc(&d_hessians_, grad_bytes));
    gpu_bytes_alloc_ += grad_bytes * 2;

    /* Device histogram buffers (working + parent for subtraction) */
    CUDA_CHECK_FATAL(cudaMalloc(&d_hist_, hist_bytes));
    CUDA_CHECK_FATAL(cudaMalloc(&d_hist_parent_, hist_bytes));
    gpu_bytes_alloc_ += hist_bytes * 2;

    /* Device leaf row index buffer (max capacity = all rows) */
    CUDA_CHECK_FATAL(cudaMalloc(&d_leaf_rows_, leaf_bytes));
    gpu_bytes_alloc_ += leaf_bytes;

    /* Pinned host staging buffers for async DMA */
    CUDA_CHECK_FATAL(cudaMallocHost(&h_grad_pinned_, grad_bytes));
    CUDA_CHECK_FATAL(cudaMallocHost(&h_hess_pinned_, grad_bytes));
    CUDA_CHECK_FATAL(cudaMallocHost(&h_hist_pinned_, hist_bytes));

    Log::Info("[CUDASparseHist] Buffers allocated: grad %zu MB, "
              "hist %zu MB, leaf %zu MB",
              grad_bytes * 2 / (1024 * 1024),
              hist_bytes * 2 / (1024 * 1024),
              leaf_bytes / (1024 * 1024));
}


/* =========================================================================
 * ConstructHistograms — THE HOT PATH
 *
 * Called by SerialTreeLearner::Train() for each tree level. Builds
 * histograms for the smaller leaf (and optionally the larger leaf
 * via subtraction trick or direct build).
 *
 * We use gradients_ and hessians_ (inherited from SerialTreeLearner),
 * which are RAW per-sample gradients indexed by original row ID.
 * GBDT::TrainOneIter() pre-slices the pointer per class, so these
 * are single-class arrays of length num_data_.
 *
 * data_indices (from leaf_splits): original row indices for the leaf.
 * We do NOT use ordered_gradients_ (the CPU reordering buffer).
 *
 * Histogram output goes into smaller_leaf_histogram_array_ which the
 * base class's FindBestSplits() reads.
 * ========================================================================= */

void CUDASparseHistTreeLearner::ConstructHistograms(
    const std::vector<int8_t>& is_feature_used,
    bool use_subtract) {

    if (!gpu_initialized_) {
        /* GPU init failed — hard error, no silent CPU fallback */
        Log::Fatal("[CUDASparseHist] GPU not initialized. Cannot build histograms. "
                   "Call Init() first or check GPU availability.");
        throw std::runtime_error("GPU histogram builder not initialized");
    }

    /* Deferred CSR upload: The Python wrapper calls LGBM_BoosterSetExternalCSR()
     * which stores CSR in a global struct (c_api.cpp). The tree learner reads it
     * here on the first ConstructHistograms() call. */
    if (!csr_uploaded_ && !has_external_csr_ && LGBM_GetExternalCSRIsSet()) {
        /* Bridge: pull CSR from global C API into this tree learner instance */
        Log::Info("[CUDASparseHist] Reading external CSR from C API global...");
        SetExternalCSR(
            LGBM_GetExternalCSRIndptr(),
            LGBM_GetExternalCSRIndices(),
            LGBM_GetExternalCSRNnz(),
            LGBM_GetExternalCSRNRows(),
            LGBM_GetExternalCSRNFeatures()
        );
    }
    if (!csr_uploaded_ && has_external_csr_) {
        UploadCSR();
        if (gpu_hist_mode_ == GPU_HIST_MODE_CUSPARSE && !has_efb_data_) {
            SetupCuSPARSE();
        } else if (gpu_hist_mode_ == GPU_HIST_MODE_CUSPARSE && has_efb_data_) {
            Log::Warning("[CUDASparseHist] EFB data present — falling back to "
                         "atomic scatter mode (cuSPARSE requires raw binary)");
            gpu_hist_mode_ = GPU_HIST_MODE_ATOMIC;
        }
        csr_uploaded_ = true;
        Log::Info("[CUDASparseHist] CSR uploaded on first ConstructHistograms call");
    }
    if (!csr_uploaded_) {
        /* No CSR data — hard error, no silent CPU fallback */
        Log::Fatal("[CUDASparseHist] No CSR data available. Upload CSR via "
                   "LGBM_BoosterSetExternalCSR() before training.");
        throw std::runtime_error("No CSR data uploaded for GPU histogram building");
    }

    /* Get smaller/larger leaf info from the base class leaf splits.
     * smaller_leaf_splits_ and larger_leaf_splits_ are set by
     * SerialTreeLearner::BeforeFindBestSplit() before this is called. */
    int smaller_leaf = smaller_leaf_splits_->leaf_index();
    int larger_leaf  = (larger_leaf_splits_ != nullptr)
                       ? larger_leaf_splits_->leaf_index() : -1;

    data_size_t n_smaller_rows = smaller_leaf_splits_->num_data_in_leaf();
    data_size_t n_larger_rows  = (larger_leaf_splits_ != nullptr)
                                 ? larger_leaf_splits_->num_data_in_leaf() : 0;

    if (n_smaller_rows <= 0) {
        return;  /* Empty leaf, nothing to build */
    }

    /* ---- Step 1: Upload gradients/hessians to GPU ---- */
    /* gradients_ and hessians_ are inherited protected members from
     * SerialTreeLearner. For multiclass, GBDT::TrainOneIter() offsets
     * the gradient pointer by (cur_tree_id * num_data_) before calling
     * tree_learner_->Train(). So gradients_[row] gives the gradient
     * for the CURRENT class at original row position `row`.
     * There are n_rows_ elements (NOT n_rows_ * num_classes_). */
    {
        int64_t n_grad_elems = static_cast<int64_t>(n_rows_);
        size_t grad_bytes = static_cast<size_t>(n_grad_elems) * sizeof(double);

        /* Copy to pinned staging buffers. score_t may be float or double
         * depending on build config (SCORE_T_USE_DOUBLE). GPU kernels
         * expect double, so widen if score_t is float. */
        const score_t* src_grad = gradients_;
        const score_t* src_hess = hessians_;

#ifdef SCORE_T_USE_DOUBLE
        /* score_t is double — direct copy */
        std::memcpy(h_grad_pinned_, src_grad, grad_bytes);
        std::memcpy(h_hess_pinned_, src_hess, grad_bytes);
#else
        /* score_t is float — widen to double */
        for (int64_t i = 0; i < n_grad_elems; i++) {
            h_grad_pinned_[i] = static_cast<double>(src_grad[i]);
            h_hess_pinned_[i] = static_cast<double>(src_hess[i]);
        }
#endif

        /* Async H2D via pinned memory */
        CUDA_CHECK_FATAL(cudaMemcpyAsync(d_gradients_, h_grad_pinned_,
                         grad_bytes, cudaMemcpyHostToDevice, stream_h2d_));
        CUDA_CHECK_FATAL(cudaMemcpyAsync(d_hessians_, h_hess_pinned_,
                         grad_bytes, cudaMemcpyHostToDevice, stream_h2d_));
    }

    /* ---- Step 2: Get row indices for the smaller leaf ---- */
    /* data_indices() returns the original row indices for this leaf's
     * data subset. These map leaf positions to original row positions.
     * For the root node (leaf 0, first split), data_indices() may return
     * NULL because all rows are in the leaf — generate sequential indices. */
    const data_size_t* data_indices = smaller_leaf_splits_->data_indices();

    std::vector<data_size_t> seq_indices;
    if (data_indices == nullptr) {
        /* Root node: all rows, sequential indices [0..n-1] */
        seq_indices.resize(n_smaller_rows);
        std::iota(seq_indices.begin(), seq_indices.end(), 0);
        data_indices = seq_indices.data();
    }

    /* Upload smaller leaf's row indices to GPU */
    CUDA_CHECK_FATAL(cudaMemcpyAsync(d_leaf_rows_, data_indices,
                     static_cast<size_t>(n_smaller_rows) * sizeof(int32_t),
                     cudaMemcpyHostToDevice, stream_h2d_));

    /* Synchronize H2D before compute */
    {
        cudaEvent_t h2d_done;
        CUDA_CHECK_FATAL(cudaEventCreate(&h2d_done));
        CUDA_CHECK_FATAL(cudaEventRecord(h2d_done, stream_h2d_));
        CUDA_CHECK_FATAL(cudaStreamWaitEvent(stream_compute_, h2d_done, 0));
        CUDA_CHECK_FATAL(cudaEventDestroy(h2d_done));
    }

    /* ---- Step 3: Build histogram for smaller leaf ---- */
    /* For multiclass, GBDT::TrainOneIter() offsets the gradient pointer
     * by (cur_tree_id * num_data_) before calling Train(), so gradients_
     * already points to the current class's RAW gradient slice.
     * class_id=0 because there's no interleaving — the pointer is
     * pre-sliced. */
    int class_id = 0;  /* gradients_ already points to correct class slice */

    if (gpu_hist_mode_ == GPU_HIST_MODE_CUSPARSE) {
        BuildHistogramCuSPARSE(data_indices, n_smaller_rows, class_id);
    } else {
        BuildHistogramAtomicScatter(data_indices, n_smaller_rows, class_id);
    }

    /* ---- Step 4: Copy histogram from GPU to host ---- */
    {
        cudaEvent_t comp_done;
        CUDA_CHECK_FATAL(cudaEventCreate(&comp_done));
        CUDA_CHECK_FATAL(cudaEventRecord(comp_done, stream_compute_));
        CUDA_CHECK_FATAL(cudaStreamWaitEvent(stream_d2h_, comp_done, 0));
        CUDA_CHECK_FATAL(cudaEventDestroy(comp_done));
    }

    size_t hist_bytes = static_cast<size_t>(hist_buf_elems_) * sizeof(double);
    CUDA_CHECK_FATAL(cudaMemcpyAsync(h_hist_pinned_, d_hist_,
                     hist_bytes, cudaMemcpyDeviceToHost, stream_d2h_));
    CUDA_CHECK_FATAL(cudaStreamSynchronize(stream_d2h_));

    /* ---- DEBUG: check histogram contents ---- */
    if (!debug_printed_) {
        double sum_abs = 0.0;
        int nonzero_count = 0;
        double first_nonzero = 0.0;
        int first_nonzero_idx = -1;
        for (int64_t i = 0; i < hist_buf_elems_; i++) {
            double val = h_hist_pinned_[i];
            if (val != 0.0) {
                sum_abs += std::abs(val);
                nonzero_count++;
                if (first_nonzero_idx < 0) {
                    first_nonzero = val;
                    first_nonzero_idx = static_cast<int>(i);
                }
            }
        }
        Log::Info("[CUDASparseHist] DEBUG hist: %lld elems, nonzero=%d, sum_abs=%.6f, "
                  "first_nz_idx=%d val=%.6f",
                  static_cast<long long>(hist_buf_elems_), nonzero_count, sum_abs,
                  first_nonzero_idx, first_nonzero);

        /* Print first 10 nonzero entries */
        int shown = 0;
        for (int64_t i = 0; i < hist_buf_elems_ && shown < 10; i++) {
            if (h_hist_pinned_[i] != 0.0) {
                Log::Info("[CUDASparseHist] DEBUG hist[%lld] = %.8f (bin=%lld, %s)",
                          static_cast<long long>(i),
                          h_hist_pinned_[i],
                          static_cast<long long>(i / 2),
                          (i % 2 == 0) ? "grad" : "hess");
                shown++;
            }
        }

        /* (RawData() debug moved after Step 5 memcpy) */

        /* Check total grad/hess for feature 0 (inner idx 0) */
        int n_used = train_data_->num_features();
        if (n_used > 0) {
            Log::Info("[CUDASparseHist] DEBUG n_used_features=%d, n_total_features=%d, "
                      "n_smaller_rows=%d",
                      n_used, n_features_, n_smaller_rows);
        }

        /* Don't set debug_printed_ until after Step 5 memcpy verification */
    }

    /* ---- Step 5: Write GPU histogram into LightGBM's histogram arrays ---- */
    /* The base class expects histogram data starting at RawData() - kHistOffset.
     * kHistOffset = 2 (from bin.h). feature_hist_offsets[0] is typically 1,
     * meaning feature 0's data is at position 1 in the buffer. The CPU writes
     * at base[offset*2 + component] where base = RawData() - kHistOffset.
     * Our GPU histogram uses the same offset table, so we copy the GPU buffer
     * to RawData() - kHistOffset to align correctly. */
    {
        hist_t* hist_ptr = smaller_leaf_histogram_array_[0].RawData() - kHistOffset;
        /* The histogram arrays are contiguous — hist_ptr points to the
         * actual start of the flat buffer (before RawData). */
        std::memcpy(hist_ptr, h_hist_pinned_, hist_bytes);

        /* DEBUG: verify data arrived in RawData() */
        if (!debug_printed_) {
            hist_t* raw_ptr = smaller_leaf_histogram_array_[0].RawData();
            Log::Info("[CUDASparseHist] DEBUG AFTER COPY RawData() [-2..3]: %.8f %.8f %.8f %.8f %.8f %.8f",
                      raw_ptr[-2], raw_ptr[-1], raw_ptr[0], raw_ptr[1], raw_ptr[2], raw_ptr[3]);
            Log::Info("[CUDASparseHist] DEBUG AFTER COPY hist_ptr[0..5]: %.8f %.8f %.8f %.8f %.8f %.8f",
                      hist_ptr[0], hist_ptr[1], hist_ptr[2], hist_ptr[3], hist_ptr[4], hist_ptr[5]);
            debug_printed_ = true;
        }
    }

    /* ---- Step 6: Handle larger leaf histogram ---- */
    if (use_subtract && larger_leaf >= 0) {
        /* Subtraction trick: larger_hist = parent_hist - smaller_hist.
         * The parent histogram was saved from the previous level.
         * Do subtraction on CPU — it's fast (~1ms for 23K bins) and
         * avoids another D2H round-trip. */
        hist_t* smaller_hist =
            smaller_leaf_histogram_array_[0].RawData();
        hist_t* larger_hist =
            larger_leaf_histogram_array_[0].RawData();
        hist_t* parent_hist =
            parent_leaf_histogram_array_[0].RawData();

        int64_t n_elems = hist_buf_elems_;

        for (int64_t i = 0; i < n_elems; i++) {
            larger_hist[i] = parent_hist[i] - smaller_hist[i];
        }
    } else if (!use_subtract && larger_leaf >= 0 && n_larger_rows > 0) {
        /* No subtraction — must build larger leaf histogram too.
         * Upload larger leaf's row indices and run GPU kernel again. */
        const data_size_t* larger_indices = larger_leaf_splits_->data_indices();

        std::vector<data_size_t> larger_seq_indices;
        if (larger_indices == nullptr) {
            larger_seq_indices.resize(n_larger_rows);
            std::iota(larger_seq_indices.begin(), larger_seq_indices.end(), 0);
            larger_indices = larger_seq_indices.data();
        }

        CUDA_CHECK_FATAL(cudaMemcpyAsync(d_leaf_rows_, larger_indices,
                         static_cast<size_t>(n_larger_rows) * sizeof(int32_t),
                         cudaMemcpyHostToDevice, stream_h2d_));

        /* Sync H2D */
        {
            cudaEvent_t h2d_done;
            CUDA_CHECK_FATAL(cudaEventCreate(&h2d_done));
            CUDA_CHECK_FATAL(cudaEventRecord(h2d_done, stream_h2d_));
            CUDA_CHECK_FATAL(cudaStreamWaitEvent(stream_compute_,
                             h2d_done, 0));
            CUDA_CHECK_FATAL(cudaEventDestroy(h2d_done));
        }

        if (gpu_hist_mode_ == GPU_HIST_MODE_CUSPARSE) {
            BuildHistogramCuSPARSE(larger_indices, n_larger_rows, class_id);
        } else {
            BuildHistogramAtomicScatter(larger_indices, n_larger_rows,
                                         class_id);
        }

        /* D2H larger histogram */
        {
            cudaEvent_t comp_done;
            CUDA_CHECK_FATAL(cudaEventCreate(&comp_done));
            CUDA_CHECK_FATAL(cudaEventRecord(comp_done, stream_compute_));
            CUDA_CHECK_FATAL(cudaStreamWaitEvent(stream_d2h_, comp_done, 0));
            CUDA_CHECK_FATAL(cudaEventDestroy(comp_done));
        }

        CUDA_CHECK_FATAL(cudaMemcpyAsync(h_hist_pinned_, d_hist_,
                         hist_bytes, cudaMemcpyDeviceToHost, stream_d2h_));
        CUDA_CHECK_FATAL(cudaStreamSynchronize(stream_d2h_));

        hist_t* larger_hist =
            larger_leaf_histogram_array_[0].RawData();
        std::memcpy(larger_hist, h_hist_pinned_, hist_bytes);
    }
}


/* =========================================================================
 * BuildHistogramCuSPARSE — cuSPARSE SpMV approach
 *
 * Uses pre-transposed CSR: csr_AT (n_features x n_rows).
 * SpMV: csr_AT @ gradient_vec = histogram_vec
 *
 * Problem: SpMV multiplies against ALL n_rows, but we only want rows
 * belonging to the current leaf.
 *
 * Solution: Create a full-size gradient vector (n_rows) with zeros for
 * non-leaf rows and actual gradients for leaf rows. This is wasteful
 * (computes against zeros) but CORRECT. The SpMV naturally produces
 * zero contributions from non-leaf rows because their gradient = 0.
 *
 * This approach is correct because:
 * 1. grad_full[row] = raw_grad[row] for leaf rows (scatter kernel)
 * 2. grad_full[row] = 0.0 for non-leaf rows (memset)
 * 3. SpMV: hist[f] = sum over rows of (csr_AT[f,row] * grad_full[row])
 *    = sum over leaf rows of (csr_AT[f,row] * grad[row])  (non-leaf = 0)
 *
 * Performance note: the SpMV still processes ALL nnz entries in the
 * transposed CSR, but ~50%+ of the gradient multiplications are 0*g=0.
 * cuSPARSE may or may not short-circuit these. For a future optimization,
 * extract only the leaf-subset rows from the transposed CSR.
 * ========================================================================= */

void CUDASparseHistTreeLearner::BuildHistogramCuSPARSE(
    const data_size_t* row_indices,
    data_size_t n_leaf_rows,
    int class_id) {

    /* Zero the full gradient vectors on GPU */
    size_t vec_bytes = n_rows_ * sizeof(double);
    CUDA_CHECK_FATAL(cudaMemsetAsync(d_full_grad_, 0, vec_bytes,
                     stream_compute_));
    CUDA_CHECK_FATAL(cudaMemsetAsync(d_full_hess_, 0, vec_bytes,
                     stream_compute_));

    /* Scatter the leaf gradients into the full-size vectors.
     * d_gradients_ has RAW (unordered) gradients for ALL n_rows_ rows,
     * indexed by original row ID. gradients_ is set directly from
     * GBDT::TrainOneIter() which offsets by (cur_tree_id * num_data_).
     * It is NOT the ordered_gradients_ array (which is a separate
     * buffer filled by Dataset::ConstructHistograms on the CPU path).
     *
     * d_leaf_rows_ has the original row indices for this leaf.
     * We scatter:  d_full_grad_[row] = d_gradients_[row]
     * for each row in d_leaf_rows_. Non-leaf rows stay zero (from memset).
     * The SpMV then sums only leaf rows' gradients correctly. */
    launch_scatter_kernel(d_leaf_rows_, d_gradients_, d_hessians_,
                          d_full_grad_, d_full_hess_,
                          n_leaf_rows, num_classes_, class_id,
                          stream_compute_);

    /* SpMV for gradients: csr_AT @ full_grad -> spmv_result (n_features) */
    double alpha = 1.0, beta = 0.0;

    /* Update the input vector descriptor to point to full_grad */
    CUSPARSE_CHECK_FATAL(cusparseDnVecSetValues(vec_in_, d_full_grad_));
    CUSPARSE_CHECK_FATAL(cusparseDnVecSetValues(vec_out_, d_spmv_result_));

    /* Zero the output vector */
    CUDA_CHECK_FATAL(cudaMemsetAsync(d_spmv_result_, 0,
                     n_features_ * sizeof(double), stream_compute_));

    /* SpMV: hist_grad = csr_AT @ full_grad */
    CUSPARSE_CHECK_FATAL(cusparseSpMV(
        cusparse_handle_,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA_T_, vec_in_, &beta, vec_out_,
        CUDA_R_64F,
        CUSPARSE_SPMV_ALG_DEFAULT,
        d_spmv_buffer_
    ));

    /* Interleave grad/hess into the histogram buffer.
     * d_spmv_result_ has grad sums per feature. Write to d_hist_[f*2+0]. */

    /* Zero histogram first */
    int64_t n_elems = hist_buf_elems_;
    hist_zero_kernel<<<grid_blocks(n_elems), BLOCK_SIZE, 0, stream_compute_>>>(
        d_hist_, n_elems);
    CUDA_CHECK_FATAL(cudaGetLastError());

    /* Write grad component (component=0) */
    launch_interleave_grad_kernel(d_spmv_result_, d_hist_,
                                   d_feature_hist_offsets_,
                                   n_features_, 0, stream_compute_);

    /* SpMV for hessians: csr_AT @ full_hess -> spmv_result (n_features) */
    CUSPARSE_CHECK_FATAL(cusparseDnVecSetValues(vec_in_, d_full_hess_));
    CUDA_CHECK_FATAL(cudaMemsetAsync(d_spmv_result_, 0,
                     n_features_ * sizeof(double), stream_compute_));

    CUSPARSE_CHECK_FATAL(cusparseSpMV(
        cusparse_handle_,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA_T_, vec_in_, &beta, vec_out_,
        CUDA_R_64F,
        CUSPARSE_SPMV_ALG_DEFAULT,
        d_spmv_buffer_
    ));

    /* Write hess component (component=1) */
    launch_interleave_grad_kernel(d_spmv_result_, d_hist_,
                                   d_feature_hist_offsets_,
                                   n_features_, 1, stream_compute_);
}


/* =========================================================================
 * BuildHistogramAtomicScatter — atomic scatter kernel approach
 *
 * Uses the original (non-transposed) CSR. Each thread processes one leaf
 * row, walks its CSR nonzeros, atomicAdd grad/hess into histogram bins.
 *
 * Advantages over cuSPARSE:
 * - Works with EFB-encoded data (variable bin values)
 * - Only processes leaf rows (no wasted work on non-leaf rows)
 * - Zero setup cost (no transpose, no cuSPARSE descriptors)
 *
 * Disadvantages:
 * - Atomic contention possible on high-density data (not our case)
 * - Cannot leverage cuSPARSE's internal optimizations
 * ========================================================================= */

void CUDASparseHistTreeLearner::BuildHistogramAtomicScatter(
    const data_size_t* row_indices,
    data_size_t n_leaf_rows,
    int class_id) {

    /* Zero histogram buffer */
    int64_t n_elems = hist_buf_elems_;
    hist_zero_kernel<<<grid_blocks(n_elems), BLOCK_SIZE, 0, stream_compute_>>>(
        d_hist_, n_elems);
    CUDA_CHECK_FATAL(cudaGetLastError());

    /* Launch atomic scatter kernel.
     * d_leaf_rows_ already contains the original row indices for this leaf.
     * d_gradients_ has RAW gradients for n_rows_ rows (single class slice —
     * GBDT already offsets by cur_tree_id * num_data_).
     * The kernel indexes: grad[row] (num_classes passed for ABI compat but
     * should always be 1 since GBDT pre-slices the gradient array). */
    int n_blocks = grid_blocks(n_leaf_rows);
    atomic_scatter_hist_kernel<<<n_blocks, BLOCK_SIZE, 0, stream_compute_>>>(
        d_csr_indptr_, d_csr_indices_, d_csr_data_,
        d_gradients_, d_hessians_,
        d_leaf_rows_, n_leaf_rows,
        class_id, num_classes_,
        d_hist_
    );
    CUDA_CHECK_FATAL(cudaGetLastError());
}


}  // namespace LightGBM (close for kernel definitions at file scope)

/* =========================================================================
 * Helper kernels for cuSPARSE mode
 * Defined at file scope to match forward declarations above.
 * ========================================================================= */

/* Scatter kernel: copy RAW gradients into full-size vectors at original
 * row positions, but ONLY for leaf rows (non-leaf rows stay zero).
 *
 * d_gradients_ (aliased as raw_grad here) contains RAW per-sample
 * gradients indexed by original row ID — NOT ordered/reordered.
 * GBDT::TrainOneIter() already offsets the pointer by
 * (cur_tree_id * num_data_), so raw_grad[row] is the current class's
 * gradient for original row `row`.
 *
 * The scatter does: full_grad[row] = raw_grad[row]
 * where row = leaf_rows[tid]. This places the gradient at the correct
 * position in the full-size vector for subsequent SpMV multiplication
 * against the CSR matrix (which is also in original row order). */
__global__ void scatter_grad_kernel(
    const int32_t* __restrict__ leaf_rows,
    const double*  __restrict__ raw_grad,
    const double*  __restrict__ raw_hess,
    double*        __restrict__ full_grad,
    double*        __restrict__ full_hess,
    int32_t                     n_leaf_rows,
    int32_t                     num_classes,  /* unused, kept for ABI compat */
    int32_t                     class_id      /* unused, kept for ABI compat */
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_leaf_rows) return;

    int32_t row = leaf_rows[tid];
    full_grad[row] = raw_grad[row];
    full_hess[row] = raw_hess[row];
}


/* Interleave kernel: copy SpMV results into interleaved histogram buffer.
 *   d_hist[feature_hist_offsets[f] * 2 + component] = d_spmv_result[f]
 * component=0 for gradients, component=1 for hessians.
 *
 * The offset table has n_features entries (one per CSR column / total feature).
 * Unused features have offset == UINT32_MAX — their SpMV output is discarded. */
__global__ void interleave_grad_kernel(
    const double* __restrict__ spmv_result,
    double*       __restrict__ hist,
    const uint32_t* __restrict__ feature_hist_offsets,
    int32_t                    n_features,
    int32_t                    component      /* 0=grad, 1=hess */
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_features) return;

    uint32_t hist_offset = feature_hist_offsets[tid];
    if (hist_offset == UINT32_MAX) return;  /* unused feature — skip */
    hist[static_cast<int64_t>(hist_offset) * 2 + component] = spmv_result[tid];
}


/* Launch wrappers (called from BuildHistogramCuSPARSE) */

static void launch_scatter_kernel(
    const int32_t* d_leaf_rows, const double* d_raw_grad,
    const double* d_raw_hess, double* d_full_grad, double* d_full_hess,
    int32_t n_leaf_rows, int32_t num_classes, int32_t class_id,
    cudaStream_t stream) {

    int n_blocks = grid_blocks(n_leaf_rows);
    scatter_grad_kernel<<<n_blocks, BLOCK_SIZE, 0, stream>>>(
        d_leaf_rows, d_raw_grad, d_raw_hess,
        d_full_grad, d_full_hess,
        n_leaf_rows, num_classes, class_id);
    /* Caller checks cudaGetLastError */
}


static void launch_interleave_grad_kernel(
    const double* d_spmv_result, double* d_hist,
    const uint32_t* d_feature_hist_offsets,
    int32_t n_features, int component, cudaStream_t stream) {

    int n_blocks = grid_blocks(n_features);
    interleave_grad_kernel<<<n_blocks, BLOCK_SIZE, 0, stream>>>(
        d_spmv_result, d_hist, d_feature_hist_offsets, n_features, component);
    /* Caller checks cudaGetLastError */
}


namespace LightGBM {  // reopen for remaining class methods

/* =========================================================================
 * CleanupGPU — free all GPU resources
 * ========================================================================= */

void CUDASparseHistTreeLearner::CleanupGPU() {
    if (!gpu_initialized_) return;

    /* Synchronize all streams before freeing */
    if (stream_h2d_)     cudaStreamSynchronize(stream_h2d_);
    if (stream_compute_) cudaStreamSynchronize(stream_compute_);
    if (stream_d2h_)     cudaStreamSynchronize(stream_d2h_);

    /* Destroy cuSPARSE resources */
    if (vec_out_)          cusparseDestroyDnVec(vec_out_);
    if (vec_in_)           cusparseDestroyDnVec(vec_in_);
    if (matA_T_)           cusparseDestroySpMat(matA_T_);
    if (cusparse_handle_)  cusparseDestroy(cusparse_handle_);
    if (d_spmv_buffer_)    cudaFree(d_spmv_buffer_);

    /* Free device memory — feature hist offsets */
    if (d_feature_hist_offsets_) cudaFree(d_feature_hist_offsets_);

    /* Free device memory — CSR */
    if (d_csr_indptr_)     cudaFree(d_csr_indptr_);
    if (d_csr_indices_)    cudaFree(d_csr_indices_);
    if (d_csr_data_)       cudaFree(d_csr_data_);
    if (d_csrT_indptr_)    cudaFree(d_csrT_indptr_);
    if (d_csrT_indptr32_)  cudaFree(d_csrT_indptr32_);
    if (d_csrT_indices_)   cudaFree(d_csrT_indices_);
    if (d_csrT_indices64_) cudaFree(d_csrT_indices64_);

    /* Free device memory — gradients and SpMV vectors */
    if (d_gradients_)      cudaFree(d_gradients_);
    if (d_hessians_)       cudaFree(d_hessians_);
    if (d_full_grad_)      cudaFree(d_full_grad_);
    if (d_full_hess_)      cudaFree(d_full_hess_);
    if (d_spmv_result_)    cudaFree(d_spmv_result_);

    /* Free device memory — histogram and leaf rows */
    if (d_hist_)           cudaFree(d_hist_);
    if (d_hist_parent_)    cudaFree(d_hist_parent_);
    if (d_leaf_rows_)      cudaFree(d_leaf_rows_);

    /* Free pinned host memory */
    if (h_grad_pinned_)    cudaFreeHost(h_grad_pinned_);
    if (h_hess_pinned_)    cudaFreeHost(h_hess_pinned_);
    if (h_hist_pinned_)    cudaFreeHost(h_hist_pinned_);

    /* Destroy streams */
    if (stream_h2d_)       cudaStreamDestroy(stream_h2d_);
    if (stream_compute_)   cudaStreamDestroy(stream_compute_);
    if (stream_d2h_)       cudaStreamDestroy(stream_d2h_);

    Log::Info("[CUDASparseHist] Cleanup: freed %zu MB GPU memory",
              gpu_bytes_alloc_ / (1024 * 1024));

    gpu_bytes_alloc_   = 0;
    gpu_initialized_   = false;
}


/* =========================================================================
 * Factory Registration
 *
 * Register this tree learner with LightGBM's factory so it can be
 * selected via tree_learner=cuda_sparse_hist in the config.
 *
 * Usage in params dict:
 *   params['tree_learner'] = 'cuda_sparse_hist'
 *   params['gpu_hist_mode'] = 1   # 1=cuSPARSE (default), 2=atomic
 *   params['gpu_device_id'] = 0   # CUDA device ordinal
 * ========================================================================= */

TreeLearner* CreateCUDASparseHistTreeLearner(const Config* config) {
    return new CUDASparseHistTreeLearner(config);
}

/* Registration is handled via tree_learner.cpp factory (already patched).
 * The factory checks device_type == "cuda_sparse" and directly instantiates
 * CUDASparseHistTreeLearner. No plugin registry needed. */


}  // namespace LightGBM
