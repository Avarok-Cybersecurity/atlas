// SPDX-License-Identifier: AGPL-3.0-only AND Apache-2.0
// Portions copied from vLLM (https://github.com/vllm-project/vllm, commit 8a36f08),
// csrc/quantization/marlin/marlin.cuh:1-152, Copyright contributors to the vLLM project,
// Apache-2.0.
// Copyright (C) Marlin.2024 Elias Frantar, Apache-2.0, adapted from
// https://github.com/IST-DASLab/marlin.
// Changes (by Atlas contributors), listed in VLLM-MARLIN-ATTRIBUTION.md:
//   * `div_ceil` is `__host__ __device__`: the kernel calls it at run time, and the Atlas
//     kernel build does not pass --expt-relaxed-constexpr;
//   * the pre-sm_80 plain-load fallbacks of the cp.async helpers are removed (this file
//     builds for sm_89 only), and so are the unused `<iostream>` and `<cuda.h>` includes;
//   * the include guard is `#pragma once`.

// Marlin launch constants, the register vector type and the cp.async helpers, shared by
// marlin_template_ada.cuh and marlin_ada.cu.
//
// The constants are the HOST's contract as much as the kernel's: L18b's Rust dispatcher
// (crates/spark-model/src/port/marlin/config.rs) mirrors `min_thread_n`, `min_thread_k`,
// `max_thread_n`, `tile_size` and `max_par` when it picks a thread configuration and sizes
// the workspace, exactly as vLLM's marlin.cu does.
#pragma once

#include <cuda_fp16.h>
#include <cstdint>

#ifndef MARLIN_NAMESPACE_NAME
  #define MARLIN_NAMESPACE_NAME marlin
#endif

namespace MARLIN_NAMESPACE_NAME {

// Marlin params

// 8 warps are a good choice since every SM has 4 schedulers and having more
// than 1 warp per schedule allows some more latency hiding. At the same time,
// we want relatively few warps to have many registers per warp and small tiles.
static constexpr int default_threads = 256;

static constexpr int pipe_stages =
    4;  // 4 pipeline stages fit into shared memory

static constexpr int min_thread_n = 64;
static constexpr int min_thread_k = 64;
static constexpr int max_thread_n = 256;

static constexpr int tile_size = 16;
static constexpr int max_par = 16;

// Repack params
static constexpr int repack_stages = 8;

static constexpr int repack_threads = 256;

static constexpr int tile_k_size = tile_size;
static constexpr int tile_n_size = tile_k_size * 4;

// Helpers
template <typename T, int n>
struct Vec {
  T elems[n];
  __device__ T& operator[](int i) { return elems[i]; }
};

using I4 = Vec<int, 4>;

__host__ __device__ constexpr int div_ceil(int a, int b) { return (a + b - 1) / b; }

__device__ inline void cp_async1_ca_pred(void* smem_ptr, const void* glob_ptr,
                                         bool pred = true) {
  const int BYTES = 4;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      "   .reg .pred p;\n"
      "   setp.ne.b32 p, %0, 0;\n"
      "   @p cp.async.ca.shared.global [%1], [%2], %3;\n"
      "}\n" ::"r"((int)pred),
      "r"(smem), "l"(glob_ptr), "n"(BYTES));
}

__device__ inline void cp_async2_ca_pred(void* smem_ptr, const void* glob_ptr,
                                         bool pred = true) {
  const int BYTES = 8;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      "   .reg .pred p;\n"
      "   setp.ne.b32 p, %0, 0;\n"
      "   @p cp.async.ca.shared.global [%1], [%2], %3;\n"
      "}\n" ::"r"((int)pred),
      "r"(smem), "l"(glob_ptr), "n"(BYTES));
}

__device__ inline void cp_async4_pred(void* smem_ptr, const void* glob_ptr,
                                      bool pred = true) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      "   .reg .pred p;\n"
      "   setp.ne.b32 p, %0, 0;\n"
      "   @p cp.async.cg.shared.global [%1], [%2], %3;\n"
      "}\n" ::"r"((int)pred),
      "r"(smem), "l"(glob_ptr), "n"(BYTES));
}

__device__ inline void cp_async4(void* smem_ptr, const void* glob_ptr) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "{\n"
      "   cp.async.cg.shared.global [%0], [%1], %2;\n"
      "}\n" ::"r"(smem),
      "l"(glob_ptr), "n"(BYTES));
}

__device__ inline void cp_async_fence() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int n>
__device__ inline void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
}

}  // namespace MARLIN_NAMESPACE_NAME
