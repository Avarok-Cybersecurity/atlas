// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_runtime_api.h>

#include <cublasLt.h>

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"

template <int TB_M, int TB_N, int TB_K, int W_M, int W_N, int W_K>
int atlas_cutlass_bf16_gemm_act_weight_t_impl(
    const void* act,
    const void* weight,
    void* out,
    int m,
    int n,
    int k,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream) {
  using Element = cutlass::bfloat16_t;
  using Gemm = cutlass::gemm::device::Gemm<
      Element,
      cutlass::layout::RowMajor,
      Element,
      cutlass::layout::ColumnMajor,
      Element,
      cutlass::layout::RowMajor,
      float,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<TB_M, TB_N, TB_K>,
      cutlass::gemm::GemmShape<W_M, W_N, W_K>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<Element, 8, float, float>>;

  Gemm gemm;
  typename Gemm::Arguments args(
      {m, n, k},
      {static_cast<Element const*>(act), k},
      {static_cast<Element const*>(weight), k},
      {static_cast<Element const*>(out), n},
      {static_cast<Element*>(out), n},
      {1.0f, 0.0f});

  size_t needed = Gemm::get_workspace_size(args);
  if (needed > workspace_size) {
    return -2;
  }
  cutlass::Status status = gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return static_cast<int>(status);
  }
  status = gemm.initialize(args, workspace, stream);
  if (status != cutlass::Status::kSuccess) {
    return static_cast<int>(status);
  }
  status = gemm(stream);
  return static_cast<int>(status);
}

extern "C" int atlas_cutlass_bf16_gemm_act_weight_t(
    const void* act,
    const void* weight,
    void* out,
    int m,
    int n,
    int k,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream) {
  return atlas_cutlass_bf16_gemm_act_weight_t_impl<128, 128, 32, 64, 64, 32>(
      act, weight, out, m, n, k, workspace, workspace_size, stream);
}

extern "C" int atlas_cutlass_bf16_gemm_act_weight_t_128x256(
    const void* act,
    const void* weight,
    void* out,
    int m,
    int n,
    int k,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream) {
  return atlas_cutlass_bf16_gemm_act_weight_t_impl<128, 256, 32, 64, 64, 32>(
      act, weight, out, m, n, k, workspace, workspace_size, stream);
}

extern "C" int atlas_cutlass_bf16_gemm_act_weight_t_256x128(
    const void* act,
    const void* weight,
    void* out,
    int m,
    int n,
    int k,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream) {
  return atlas_cutlass_bf16_gemm_act_weight_t_impl<256, 128, 32, 64, 64, 32>(
      act, weight, out, m, n, k, workspace, workspace_size, stream);
}

extern "C" int atlas_cutlass_bf16_gemm_act_weight_t_64x128(
    const void* act,
    const void* weight,
    void* out,
    int m,
    int n,
    int k,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream) {
  return atlas_cutlass_bf16_gemm_act_weight_t_impl<64, 128, 32, 32, 64, 32>(
      act, weight, out, m, n, k, workspace, workspace_size, stream);
}

extern "C" int atlas_cutlass_bf16_gemm_act_weight_t_128x64(
    const void* act,
    const void* weight,
    void* out,
    int m,
    int n,
    int k,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream) {
  return atlas_cutlass_bf16_gemm_act_weight_t_impl<128, 64, 32, 64, 32, 32>(
      act, weight, out, m, n, k, workspace, workspace_size, stream);
}

extern "C" int atlas_cutlass_bf16_gemm_act_weight_t_64x64(
    const void* act,
    const void* weight,
    void* out,
    int m,
    int n,
    int k,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream) {
  return atlas_cutlass_bf16_gemm_act_weight_t_impl<64, 64, 32, 32, 32, 32>(
      act, weight, out, m, n, k, workspace, workspace_size, stream);
}

extern "C" int atlas_cublaslt_bf16_gemm_act_weight_t_algo(
    const void* act,
    const void* weight,
    void* out,
    int m,
    int n,
    int k,
    void* workspace,
    size_t workspace_size,
    cudaStream_t stream,
    int algo_index,
    int* returned_count) {
  static cublasLtHandle_t handle = nullptr;
  if (handle == nullptr) {
    cublasStatus_t st = cublasLtCreate(&handle);
    if (st != CUBLAS_STATUS_SUCCESS) {
      return static_cast<int>(st);
    }
  }

  cublasLtMatmulDesc_t desc = nullptr;
  cublasLtMatrixLayout_t layout_a = nullptr;
  cublasLtMatrixLayout_t layout_b = nullptr;
  cublasLtMatrixLayout_t layout_d = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;

  auto cleanup = [&]() {
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (layout_a) cublasLtMatrixLayoutDestroy(layout_a);
    if (layout_b) cublasLtMatrixLayoutDestroy(layout_b);
    if (layout_d) cublasLtMatrixLayoutDestroy(layout_d);
    if (desc) cublasLtMatmulDescDestroy(desc);
  };

  cublasStatus_t st =
      cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  if (st != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return static_cast<int>(st);
  }
  cublasOperation_t trans_a = CUBLAS_OP_T;
  cublasOperation_t trans_b = CUBLAS_OP_N;
  st = cublasLtMatmulDescSetAttribute(
      desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a));
  if (st != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return static_cast<int>(st);
  }
  st = cublasLtMatmulDescSetAttribute(
      desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b));
  if (st != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return static_cast<int>(st);
  }

  st = cublasLtMatrixLayoutCreate(&layout_a, CUDA_R_16BF, k, n, k);
  if (st != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return static_cast<int>(st);
  }
  st = cublasLtMatrixLayoutCreate(&layout_b, CUDA_R_16BF, k, m, k);
  if (st != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return static_cast<int>(st);
  }
  st = cublasLtMatrixLayoutCreate(&layout_d, CUDA_R_16BF, n, m, n);
  if (st != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return static_cast<int>(st);
  }
  st = cublasLtMatmulPreferenceCreate(&pref);
  if (st != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return static_cast<int>(st);
  }
  st = cublasLtMatmulPreferenceSetAttribute(
      pref,
      CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &workspace_size,
      sizeof(workspace_size));
  if (st != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return static_cast<int>(st);
  }

  constexpr int kMaxAlgos = 16;
  cublasLtMatmulHeuristicResult_t results[kMaxAlgos];
  int returned = 0;
  st = cublasLtMatmulAlgoGetHeuristic(
      handle,
      desc,
      layout_a,
      layout_b,
      layout_d,
      layout_d,
      pref,
      kMaxAlgos,
      results,
      &returned);
  if (returned_count) {
    *returned_count = returned;
  }
  if (st != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return static_cast<int>(st);
  }
  if (algo_index < 0 || algo_index >= returned) {
    cleanup();
    return -3;
  }
  if (results[algo_index].state != CUBLAS_STATUS_SUCCESS ||
      results[algo_index].workspaceSize > workspace_size) {
    cleanup();
    return -4;
  }

  float alpha = 1.0f;
  float beta = 0.0f;
  st = cublasLtMatmul(
      handle,
      desc,
      &alpha,
      weight,
      layout_a,
      act,
      layout_b,
      &beta,
      out,
      layout_d,
      out,
      layout_d,
      &results[algo_index].algo,
      workspace,
      workspace_size,
      stream);
  cleanup();
  return static_cast<int>(st);
}
