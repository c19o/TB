/*!
 * \file cuda_sparse_hist_tree_learner.h
 * \brief GPU-accelerated histogram building for sparse CSR feature matrices.
 *
 * Subclasses SerialTreeLearner and overrides ONLY ConstructHistograms().
 * Everything else (split finding, tree structure, EFB bundling, gradient
 * computation, row partitioning) stays on CPU in SerialTreeLearner.
 *
 * Design constraints (matrix thesis — non-negotiable):
 *   - ALL features preserved. No filtering, no subsampling, no pruning.
 *   - Sparse binary cross features (2-10M) ARE the edge.
 *   - EFB bundling stays CPU-side (one-time at Dataset construction).
 *   - GPU only builds gradient/hessian histograms from EFB-bundled CSR.
 *   - feature_pre_filter=False always.
 *   - Bit-exact float64 accumulation matching CPU LightGBM output.
 *
 * Memory model:
 *   - CSR matrix uploaded to GPU on first ConstructHistograms() call
 *     (deferred from Init() because SetExternalCSR() hasn't been called yet).
 *     Stays resident for all boosting rounds. Freed at destructor or
 *     ResetTrainingData().
 *   - Gradients uploaded per boosting round via pinned memory + async H2D.
 *   - Histograms computed on GPU, transferred back via async D2H into
 *     pinned host buffer, then copied into LightGBM's histogram arrays.
 *   - Three CUDA streams overlap H2D / compute / D2H for pipelining.
 *
 * Phase 1 approach:
 *   - CSR data provided externally via SetExternalCSR() (called from Python
 *     after Dataset construction). Avoids accessing LightGBM's private
 *     MultiValBin internals.
 *
 * Supported GPUs (fat binary, set in CMakeLists.txt):
 *   sm_80 (A100/A30), sm_86 (RTX 3090/A40), sm_89 (RTX 4090/L40),
 *   sm_90 (H100/H200), PTX fallback.
 *
 * Copyright (c) Savage22 Server Project. Licensed under MIT.
 */

#ifndef LIGHTGBM_TREELEARNER_CUDA_SPARSE_HIST_TREE_LEARNER_H_
#define LIGHTGBM_TREELEARNER_CUDA_SPARSE_HIST_TREE_LEARNER_H_

#include <LightGBM/config.h>
#include <LightGBM/dataset.h>
#include <LightGBM/utils/log.h>

#include "serial_tree_learner.h"

#include <cuda_runtime.h>
#include <cusparse.h>

#include <cstdint>
#include <vector>

namespace LightGBM {

/*!
 * \brief Tree learner that offloads histogram construction to GPU via
 *        sparse CSR kernel, while keeping all other operations on CPU.
 *
 * Inherits from SerialTreeLearner (NOT CUDASingleGPUTreeLearner) because
 * we only replace the histogram inner loop.
 */
class CUDASparseHistTreeLearner : public SerialTreeLearner {
 public:
  explicit CUDASparseHistTreeLearner(const Config* config);
  ~CUDASparseHistTreeLearner() override;

  /* Disable copy/move — GPU resources are not trivially copyable. */
  CUDASparseHistTreeLearner(const CUDASparseHistTreeLearner&) = delete;
  CUDASparseHistTreeLearner& operator=(const CUDASparseHistTreeLearner&) = delete;
  CUDASparseHistTreeLearner(CUDASparseHistTreeLearner&&) = delete;
  CUDASparseHistTreeLearner& operator=(CUDASparseHistTreeLearner&&) = delete;

  void Init(const Dataset* train_data, bool is_constant_hessian) override;

  void ConstructHistograms(const std::vector<int8_t>& is_feature_used,
                           bool use_subtract) override;

  /*!
   * \brief Provide external CSR data from Python wrapper.
   *
   * Called after Dataset construction, before training. Passes the scipy
   * CSR matrix directly, avoiding private MultiValBin access.
   *
   * \param indptr     Row pointer array, int64 (length n_rows + 1)
   * \param indices    Column index array, int32 (length nnz)
   * \param nnz        Number of nonzero entries
   * \param n_rows     Number of rows
   * \param n_features Number of features (columns)
   */
  void SetExternalCSR(const int64_t* indptr, const int32_t* indices,
                      int64_t nnz, int32_t n_rows, int32_t n_features);

 private:
  /* ---- Initialization helpers ---- */
  void InitGPU();
  void UploadCSR();
  void SetupCuSPARSE();
  void AllocateBuffers();
  void CleanupGPU();

  /* ---- Histogram building modes ---- */
  void BuildHistogramCuSPARSE(const data_size_t* row_indices,
                               data_size_t n_leaf_rows, int class_id);
  void BuildHistogramAtomicScatter(const data_size_t* row_indices,
                                    data_size_t n_leaf_rows, int class_id);

  /* ---- GPU mode selection ---- */
  int gpu_hist_mode_;

  /* ---- External CSR storage (from SetExternalCSR, before GPU upload) ---- */
  std::vector<int64_t> ext_csr_indptr_;
  std::vector<int32_t> ext_csr_indices_;
  int64_t ext_csr_nnz_        = 0;
  int32_t ext_csr_n_rows_     = 0;
  int32_t ext_csr_n_features_ = 0;
  bool    has_external_csr_   = false;

  /* ---- CSR data on GPU (stays resident for entire training) ---- */
  int64_t*  d_csr_indptr_   = nullptr;
  int32_t*  d_csr_indices_  = nullptr;
  uint8_t*  d_csr_data_     = nullptr;

  /* Transposed CSR for cuSPARSE SpMV (features x rows).
   * cuSPARSE requires matching index types (both int32 or both int64).
   * When NNZ <= INT32_MAX: use int32 for both (d_csrT_indptr32_ + d_csrT_indices_).
   * When NNZ > INT32_MAX: use int64 for both (d_csrT_indptr_ + d_csrT_indices64_). */
  int64_t*  d_csrT_indptr_     = nullptr;
  int32_t*  d_csrT_indptr32_   = nullptr;  /* int32 copy when NNZ fits */
  int32_t*  d_csrT_indices_    = nullptr;
  int64_t*  d_csrT_indices64_  = nullptr;  /* int64 copy when NNZ > INT32_MAX */
  bool      csrT_use_int32_    = true;     /* true = int32 for both */

  /* ---- Dimensions ---- */
  int64_t   n_rows_         = 0;
  int64_t   nnz_            = 0;
  int32_t   n_features_     = 0;
  int32_t   num_classes_    = 1;
  int64_t   total_hist_bins_ = 0;
  int64_t   hist_buf_elems_ = 0;

  /* ---- Gradient/Hessian buffers on GPU ---- */
  double*   d_gradients_    = nullptr;
  double*   d_hessians_     = nullptr;

  /* ---- Leaf row indices on GPU ---- */
  int32_t*  d_leaf_rows_    = nullptr;

  /* ---- Histogram buffers on GPU ---- */
  double*   d_hist_         = nullptr;
  double*   d_hist_parent_  = nullptr;

  /* ---- Pinned host staging buffers (async DMA) ---- */
  double*   h_grad_pinned_  = nullptr;
  double*   h_hess_pinned_  = nullptr;
  double*   h_hist_pinned_  = nullptr;

  /* ---- Full gradient vector for cuSPARSE SpMV ---- */
  double*   d_full_grad_    = nullptr;
  double*   d_full_hess_    = nullptr;
  double*   d_spmv_result_  = nullptr;

  /* ---- cuSPARSE handles ---- */
  cusparseHandle_t     cusparse_handle_  = nullptr;
  cusparseSpMatDescr_t matA_T_           = nullptr;
  cusparseDnVecDescr_t vec_in_           = nullptr;
  cusparseDnVecDescr_t vec_out_          = nullptr;
  size_t               spmv_buffer_size_ = 0;
  void*                d_spmv_buffer_    = nullptr;

  /* ---- CUDA streams ---- */
  cudaStream_t stream_h2d_     = nullptr;
  cudaStream_t stream_compute_ = nullptr;
  cudaStream_t stream_d2h_     = nullptr;

  /* ---- State tracking ---- */
  bool   gpu_initialized_   = false;
  bool   csr_uploaded_       = false;  /* set true after first UploadCSR() */
  bool   has_efb_data_      = false;
  size_t gpu_bytes_alloc_   = 0;

  /* ---- Host-side transposed CSR (built once, uploaded, then freed) ---- */
  std::vector<int64_t> h_csrT_indptr_;
  std::vector<int32_t> h_csrT_indices_;
};

}  // namespace LightGBM

#endif  // LIGHTGBM_TREELEARNER_CUDA_SPARSE_HIST_TREE_LEARNER_H_
