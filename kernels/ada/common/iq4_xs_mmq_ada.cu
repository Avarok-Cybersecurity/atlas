// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from llama.cpp (ggml-org/llama.cpp), MIT License, Copyright (c) 2023-2026 The ggml
// authors: ggml/src/ggml-cuda/mmq.cuh (mul_mat_q's conventional-tiling setup around
// mul_mat_q_process_tile), through the vendored copy in kernels/gb10/qwen3.6-27b/nvfp4/q4k_vendor/.
// Changes: no MoE ids, one channel and sample, bf16 output, the compile-time type switched to IQ4_XS, Q4_K
// and Q6_K, Ada entry names, by Atlas contributors. A copy of Atlas's
// kernels/gb10/qwen3.6-27b/nvfp4/q4k_mmq.cu with the type switched, as q2_0_mmq.cu and kquant_moe.cu do.
//
// Responsibility: the keep-packed PREFILL GEMMs (M > 8 rows) of the GGUF 4-bit route on Ada (port lane
// L19, row `gguf_native_iq4`). The vendored MMQ tensor-core engine (int8 mma.sync, Turing and later, so
// sm_89 runs the MMA path) reads the raw super-blocks as they sit in the file and the activations as
// block_q8_1_mmq from `atlas_q8_1_mmq_{d4,ds4}_bf16_ada` (kquant_q8_1_ada.cu): IQ4_XS and Q6_K take the
// D4 layout, Q4_K the DS4 layout (`mmq_get_q8_1_ds_layout`). Host: crates/spark-model/src/port/gguf_q4/.
//
// Entry points (module `iq4_xs_mmq_ada`), mmq_x = mmq_y = 128, 8 warps (block (32, 8, 1)), grid
// (ceil(N / 128), ceil(M / 128), 1), dynamic shared memory ATLAS_MMQ128_SMEM(type) bytes (57,856 for all
// three types; above 48 KiB, so the launch opts in, which the registry does for any launch over 48 KiB):
//   atlas_iq4_xs_mmq128_{nc,wc}, atlas_q4_k_mmq128_{nc,wc}, atlas_q6_k_mmq128_{nc,wc}
//   (x = weight blocks [N][K/256], y = q8_1_mmq activations, dst = bf16 [M][N], nrows_x = N,
//    ncols_dst = M, ncols_x = K, stride_row_x = K/256, ncols_y = M, stride_col_dst = N).
// `_wc` bounds-checks the last row tile (N % 128 != 0); `_nc` does not.

#include <cuda_bf16.h>
// The vendored ggml headers, included from gb10's 27B tree and never edited.
#include "../../gb10/qwen3.6-27b/nvfp4/q4k_vendor/mmq.cuh"

// ids (mmq_x ints) + the weight tile (mmq_y rows of the type's MMA tile width) + the activation tile
// (mmq_x block_q8_1_mmq), as `mmq_get_nbytes_shared` sizes it on the MMA path.
#define ATLAS_MMQ128_SMEM(TYPE)                                                                \
    ((unsigned int)(128 * sizeof(int) + 128 * mmq_get_mma_tile_x_k(TYPE) * sizeof(int) +      \
                    128 * sizeof(block_q8_1_mmq)))
// The activation tile needs no padding at mmq_x = 128 (GGML_PAD to nwarps * warp_size ints is a no-op),
// and the host passes the constant below; both hold only while these asserts do.
static_assert((128 * sizeof(block_q8_1_mmq)) % (8 * 32 * sizeof(int)) == 0, "activation tile is padded");
static_assert(ATLAS_MMQ128_SMEM(GGML_TYPE_IQ4_XS) == 57856u, "host constant GGUF_Q4_MMQ_SMEM (IQ4_XS)");
static_assert(ATLAS_MMQ128_SMEM(GGML_TYPE_Q4_K) == 57856u, "host constant GGUF_Q4_MMQ_SMEM (Q4_K)");
static_assert(ATLAS_MMQ128_SMEM(GGML_TYPE_Q6_K) == 57856u, "host constant GGUF_Q4_MMQ_SMEM (Q6_K)");

template <ggml_type type, int mmq_x, bool need_check>
static __device__ __forceinline__ void atlas_q4_mmq_tile(
        const char * __restrict__ x, const int * __restrict__ y, __nv_bfloat16 * __restrict__ dst,
        const int nrows_x, const int ncols_dst, const int ncols_x,
        const int stride_row_x, const int ncols_y, const int stride_col_dst) {
    constexpr int nwarps    = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int qk        = ggml_cuda_type_traits<type>::qk;
    constexpr int mmq_y     = get_mmq_y_device();

    extern __shared__ int ids_dst_shared[];
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps*warp_size) {
        const int j = j0 + threadIdx.y*warp_size + threadIdx.x;
        if (j0 + nwarps*warp_size > mmq_x && j >= mmq_x) break;
        ids_dst_shared[j] = j;
    }
    __syncthreads();

    const int it = blockIdx.x;   // tile over nrows_x (N output features)
    const int jt = blockIdx.y;   // tile over ncols_dst (M tokens)

    const int offset_y   = jt*mmq_x*(int)(sizeof(block_q8_1_mmq)/sizeof(int));
    const int offset_dst = jt*mmq_x*stride_col_dst + it*mmq_y;
    const int tile_x_max_i = nrows_x   - it*mmq_y - 1;
    const int tile_y_max_j = ncols_dst - jt*mmq_x - 1;
    const int offset_x = it*mmq_y*stride_row_x;
    const int kb0_stop = ncols_x / qk;   // K super-blocks per row (K/256)

    mul_mat_q_process_tile<type, mmq_x, need_check, /*fixup=*/false, __nv_bfloat16>(
        x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, nullptr,
        stride_row_x, ncols_y, stride_col_dst, tile_x_max_i, tile_y_max_j, 0, kb0_stop);
}

#define ATLAS_Q4_MMQ_ENTRY(NAME, TYPE, CHECK)                                                        \
extern "C" __global__ void __launch_bounds__(256, 1) NAME(                                           \
        const char* x, const int* y, __nv_bfloat16* dst,                                             \
        int nrows_x, int ncols_dst, int ncols_x, int stride_row_x, int ncols_y, int stride_col_dst) {\
    atlas_q4_mmq_tile<TYPE, 128, CHECK>(x, y, dst, nrows_x, ncols_dst, ncols_x, stride_row_x,        \
                                        ncols_y, stride_col_dst);                                     \
}
ATLAS_Q4_MMQ_ENTRY(atlas_iq4_xs_mmq128_nc, GGML_TYPE_IQ4_XS, false)
ATLAS_Q4_MMQ_ENTRY(atlas_iq4_xs_mmq128_wc, GGML_TYPE_IQ4_XS, true)
ATLAS_Q4_MMQ_ENTRY(atlas_q4_k_mmq128_nc, GGML_TYPE_Q4_K, false)
ATLAS_Q4_MMQ_ENTRY(atlas_q4_k_mmq128_wc, GGML_TYPE_Q4_K, true)
ATLAS_Q4_MMQ_ENTRY(atlas_q6_k_mmq128_nc, GGML_TYPE_Q6_K, false)
ATLAS_Q4_MMQ_ENTRY(atlas_q6_k_mmq128_wc, GGML_TYPE_Q6_K, true)
