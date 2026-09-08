// SPDX-License-Identifier: MIT
//
// Vendored from turboderp's ExLlamaV3 (https://github.com/turboderp-org/exllamav3)
// Copyright (c) 2025 turboderp — MIT license.
// Snapshot original: .research/exllamav3_ref/exl3_kernel_map.cuh.
// Adaptation: host-side selection prototypes and the function-pointer instance
// table macros are dropped — Atlas selects kernels BY NAME from the PTX module
// (see exl3_matmul.cu), so only the argument-list macros and the shape tables
// survive. Values untouched.
//
// Shape table (shape_idx 1..4):
//            TILESIZE_M  TILESIZE_K  TILESIZE_N  SH_STAGES  FRAG_STAGES  blockdim
//   shape 1      16          16         128          6           5          256
//   shape 2      16          32         128          4           3          512
//   shape 3      16          32         256          4           3          512
//   shape 4      16          16         512          4           3          256
// blockdim = EXL3_GEMM_BASE_THREADS * TILESIZE_K / 16.
// Host-side shape heuristic for Blackwell (select_gemm_shape, CC_BLACKWELL
// branch, verbatim from exl3_kernel_map.cu — the host wrapper must reproduce
// this):
//   mod_256 = (size_n % 256 == 0); mod_512 = (size_n % 512 == 0);   // BEFORE bszm scaling
//   size_k *= bszm_in; size_n *= bszm_out;
//   if ((K == 4 || K == 2) && !multi) { if (size_k <= 2048) return 1; }
//   if (K >= 7) {
//       if (mod_256 && size_n <= 8192) return size_k > 32768 ? 3 : 2;
//       if (mod_512 && size_n > 32768) return 4;
//       return 2;
//   }
//   if (mod_256 && size_n <= 4096) return size_k > 8192 && K >= 3 ? 3 : 2;
//   if (mod_512 && size_n > 16384) return 4;
//   if (mod_256) return 3;
//   return 2;
// Compat check: size_k % TILESIZE_K == 0 && size_n % TILESIZE_N == 0.

#pragma once

#define EXL3_GEMM_T_ARGS \
    const int bits, \
    const bool c_fp32, \
    const int cb, \
    const int TILESIZE_M, \
    const int TILESIZE_K, \
    const int TILESIZE_N, \
    const int SH_STAGES, \
    const int FRAG_STAGES

#define EXL3_GEMM_ARGS \
    const half* __restrict__  A, \
    const uint16_t* __restrict__ B, \
    void* __restrict__ C, \
    const int size_m, \
    const int size_k, \
    const int size_n, \
    int* __restrict__ locks, \
    const half* __restrict__ suh, \
    half* __restrict__ A_had, \
    const half* __restrict__ svh

#define EXL3_MGEMM_ARGS \
    const half* __restrict__  A, \
    const uint16_t** __restrict__ B_list, \
    void* __restrict__ C, \
    const int size_m, \
    const int size_k, \
    const int size_n, \
    int* __restrict__ locks, \
    const half** __restrict__ suh_list, \
    half* __restrict__ A_had, \
    const half** __restrict__ svh_list, \
    int64_t* B_indices, \
    half* B_weights, \
    const int bszm_in, \
    const int bszm_out, \
    const int min_index, \
    const int max_index, \
    const int num_tokens, \
    const int* __restrict__ size_n_list, \
    void** __restrict__ C_list

#define EXL3_GEMM_SHAPE_1     16,     16,    128,     6,     5
#define EXL3_GEMM_SHAPE_2     16,     32,    128,     4,     3
#define EXL3_GEMM_SHAPE_3     16,     32,    256,     4,     3
#define EXL3_GEMM_SHAPE_4     16,     16,    512,     4,     3

#define EXL3_GEMM_TILESIZE_K  0, 16, 32, 32, 16
#define EXL3_GEMM_TILESIZE_N  0, 128, 128, 256, 512
#define EXL3_GEMM_BLOCKDIM  0, 256, 512, 512, 256

#define EXL3_GEMM_NUM_SHAPES 4

#define EXL3_GEMM_BASE_THREADS 256
