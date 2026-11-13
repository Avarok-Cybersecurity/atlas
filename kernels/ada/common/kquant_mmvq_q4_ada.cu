// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from llama.cpp (ggml-org/llama.cpp), MIT License, Copyright (c) 2023-2026 The ggml
// authors: ggml/src/ggml-cuda/mmvq.cu (mul_mat_vec_q: the lane-to-block split, `blocks_per_iter`, the
// per-row warp reduction) over the vendored dots in
// kernels/gb10/qwen3.6-27b/nvfp4/q4k_vendor/vecdotq.cuh (vec_dot_q4_K_q8_1, vec_dot_iq4_xs_q8_1,
// vec_dot_q6_K_q8_1). Changes: Atlas launch shapes (one block per output row with a fixed cross-warp
// order, and a warp-per-row variant), bf16 output, up to 8 activation rows sharing every weight read, the
// entry names of an Ada file, by Atlas contributors. Structure after Atlas's
// kernels/gb10/deepseek-v4-flash/nvfp4/kquant_moe.cu (`kq_mmvq`, `kq_mmvq_warp_s`).
//
// Responsibility: the keep-packed DECODE GEMVs (M = 1 to 8 rows: plain decode and the MTP verify tier) of
// the GGUF 4-bit route on Ada (port lane L19, row `gguf_native_iq4`): IQ4_XS (136 B per 256 weights,
// 4.25 bpw), Q4_K (144 B, 4.5 bpw) and Q6_K (210 B, 6.56 bpw; the 27B's output head and the odd Q6_K
// projection of a mixed "UD" file). The weights stay the file's raw super-blocks; the activation is
// `block_q8_1` rows from `atlas_q8_1_rows_bf16_ada` (kquant_q8_1_ada.cu); each super-block is scored by the
// vendored ggml dot (dp4a on the packed codes). Host: crates/spark-model/src/port/gguf_q4/launch.rs.
//
// Layout: vx = [N rows][K/256 super-blocks] raw; vy = block_q8_1 [m][K/32]; dst = bf16 [m][N].
// K % 256 == 0, 1 <= m <= 8.
//
// Row invariance: activation row j accumulates into its own register in a loop order that does not depend
// on m, and the cross-lane and cross-warp sums run in a fixed order, so row j's bits are the same for any
// m (the W4Linear contract, port plan 3.7.2).
//
// Entry points (module `kquant_mmvq_q4_ada`), each with the same arguments
// (vx, vy, dst, ncols_x = K, nrows_x = N, m):
//   * `atlas_mmvq_{iq4_xs,q4_k,q6_k}_ada`     one block (4 warps) per output row; grid (N, 1, 1),
//                                            block (32, 4, 1). llama.cpp's own shape for m <= 8.
//   * `atlas_mmvq_{iq4_xs,q4_k,q6_k}_w_ada`   one WARP per output row, four rows a block; grid
//                                            (ceil(N / 4), 1, 1), block (32, 4, 1).
// The two shapes compute the same per-lane products and differ only in the order of the final sum, so they
// agree to bf16 rounding, not bit for bit. The host serves the block shape; the warp shape is the A/B arm
// of examples/iq4xs_gemv_microtest.rs.

#include <cuda_bf16.h>
// The vendored ggml headers, included from gb10's 27B tree and never edited (master plan 3.1).
#include "../../gb10/qwen3.6-27b/nvfp4/q4k_vendor/mmq.cuh"

#define ATLAS_Q4_NWARPS 4
#define ATLAS_Q4_MAX_M  8

// Per-type constants and the vendored dot. `vdr` is llama.cpp's VDR_*_Q8_1_MMVQ: the number of 32-bit
// quant words one lane scores per call; `qi` / `vdr` lanes cover one 256-weight super-block.
template <ggml_type type> struct atlas_q4_traits;
template <> struct atlas_q4_traits<GGML_TYPE_IQ4_XS> {
    static constexpr int vdr = VDR_IQ4_XS_Q8_1_MMVQ;   // 4: 8 lanes a super-block, iqs = 0, 4, .., 28
    typedef block_iq4_xs block;
    static __device__ __forceinline__ float dot(const void* vbq, const block_q8_1* y, const int kbx, const int iqs) {
        return vec_dot_iq4_xs_q8_1(vbq, y, kbx, iqs);
    }
};
template <> struct atlas_q4_traits<GGML_TYPE_Q4_K> {
    static constexpr int vdr = VDR_Q4_K_Q8_1_MMVQ;     // 2: 16 lanes a super-block, iqs = 0, 2, .., 30
    typedef block_q4_K block;
    static __device__ __forceinline__ float dot(const void* vbq, const block_q8_1* y, const int kbx, const int iqs) {
        return vec_dot_q4_K_q8_1(vbq, y, kbx, iqs);
    }
};
template <> struct atlas_q4_traits<GGML_TYPE_Q6_K> {
    static constexpr int vdr = VDR_Q6_K_Q8_1_MMVQ;     // 1: 32 lanes a super-block, iqs = 0, 1, .., 31
    typedef block_q6_K block;
    static __device__ __forceinline__ float dot(const void* vbq, const block_q8_1* y, const int kbx, const int iqs) {
        return vec_dot_q6_K_q8_1(vbq, y, kbx, iqs);
    }
};

// One block of ATLAS_Q4_NWARPS warps per output row `blockIdx.x`. The 128 lanes stride the row's
// super-blocks `blocks_per_iter` at a time; per-row partial sums meet in shared memory and warp 0 adds
// the warps in index order.
template <ggml_type type>
static __device__ __forceinline__ void atlas_q4_mmvq_block(
        const void* __restrict__ vx, const block_q8_1* __restrict__ y, __nv_bfloat16* __restrict__ dst,
        const int ncols_x, const int nrows_x, const int m) {
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;   // 256
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;   // 32 for all three types
    constexpr int vdr = atlas_q4_traits<type>::vdr;
    constexpr int blocks_per_iter = vdr * ATLAS_Q4_NWARPS * 32 / qi;
    typedef typename atlas_q4_traits<type>::block block_t;

    const int tid = 32 * threadIdx.y + threadIdx.x;
    const int row = blockIdx.x;
    const int blocks_per_row_x = ncols_x / qk;
    const int blocks_per_col_y = ncols_x / QK8_1;
    const char* x_row = (const char*)vx + (size_t)row * blocks_per_row_x * sizeof(block_t);

    float tmp[ATLAS_Q4_MAX_M];
#pragma unroll
    for (int j = 0; j < ATLAS_Q4_MAX_M; ++j) tmp[j] = 0.0f;

    for (int kbx = tid / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);   // 8 q8_1 blocks per super-block
        const int kqs = vdr * (tid % (qi / vdr));
#pragma unroll
        for (int j = 0; j < ATLAS_Q4_MAX_M; ++j) {
            if (j < m) {
                tmp[j] += atlas_q4_traits<type>::dot(x_row, &y[(size_t)j * blocks_per_col_y + kby], kbx, kqs);
            }
        }
    }

    __shared__ float red[ATLAS_Q4_NWARPS][ATLAS_Q4_MAX_M];
#pragma unroll
    for (int j = 0; j < ATLAS_Q4_MAX_M; ++j) {
        const float v = warp_reduce_sum(tmp[j]);
        if (threadIdx.x == 0) red[threadIdx.y][j] = v;
    }
    __syncthreads();
    if (threadIdx.y == 0 && (int)threadIdx.x < m) {
        float s = 0.0f;
#pragma unroll
        for (int w = 0; w < ATLAS_Q4_NWARPS; ++w) s += red[w][threadIdx.x];
        dst[(size_t)threadIdx.x * nrows_x + row] = __float2bfloat16(s);
    }
}

// One WARP per output row (row = blockIdx.x * ATLAS_Q4_NWARPS + threadIdx.y): the 32 lanes stride the
// row's super-blocks, one warp shuffle reduces, no shared memory and no __syncthreads.
template <ggml_type type>
static __device__ __forceinline__ void atlas_q4_mmvq_warp(
        const void* __restrict__ vx, const block_q8_1* __restrict__ y, __nv_bfloat16* __restrict__ dst,
        const int ncols_x, const int nrows_x, const int m) {
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = atlas_q4_traits<type>::vdr;
    constexpr int blocks_per_iter = vdr * 32 / qi;
    typedef typename atlas_q4_traits<type>::block block_t;

    const int lane = threadIdx.x;
    const int row = blockIdx.x * ATLAS_Q4_NWARPS + threadIdx.y;
    if (row >= nrows_x) return;
    const int blocks_per_row_x = ncols_x / qk;
    const int blocks_per_col_y = ncols_x / QK8_1;
    const char* x_row = (const char*)vx + (size_t)row * blocks_per_row_x * sizeof(block_t);

    float tmp[ATLAS_Q4_MAX_M];
#pragma unroll
    for (int j = 0; j < ATLAS_Q4_MAX_M; ++j) tmp[j] = 0.0f;
    for (int kbx = lane / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);
        const int kqs = vdr * (lane % (qi / vdr));
#pragma unroll
        for (int j = 0; j < ATLAS_Q4_MAX_M; ++j) {
            if (j < m) {
                tmp[j] += atlas_q4_traits<type>::dot(x_row, &y[(size_t)j * blocks_per_col_y + kby], kbx, kqs);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < ATLAS_Q4_MAX_M; ++j) {
        const float v = warp_reduce_sum(tmp[j]);
        if (lane == 0 && j < m) dst[(size_t)j * nrows_x + row] = __float2bfloat16(v);
    }
}

#define ATLAS_Q4_MMVQ_ENTRIES(SUFFIX, TYPE)                                                                \
    extern "C" __global__ void __launch_bounds__(32 * ATLAS_Q4_NWARPS) atlas_mmvq_##SUFFIX##_ada(           \
            const void* __restrict__ vx, const void* __restrict__ vy, __nv_bfloat16* __restrict__ dst,     \
            unsigned int ncols_x, unsigned int nrows_x, unsigned int m) {                                  \
        atlas_q4_mmvq_block<TYPE>(vx, (const block_q8_1*)vy, dst, (int)ncols_x, (int)nrows_x, (int)m);     \
    }                                                                                                      \
    extern "C" __global__ void __launch_bounds__(32 * ATLAS_Q4_NWARPS) atlas_mmvq_##SUFFIX##_w_ada(         \
            const void* __restrict__ vx, const void* __restrict__ vy, __nv_bfloat16* __restrict__ dst,     \
            unsigned int ncols_x, unsigned int nrows_x, unsigned int m) {                                  \
        atlas_q4_mmvq_warp<TYPE>(vx, (const block_q8_1*)vy, dst, (int)ncols_x, (int)nrows_x, (int)m);      \
    }

ATLAS_Q4_MMVQ_ENTRIES(iq4_xs, GGML_TYPE_IQ4_XS)
ATLAS_Q4_MMVQ_ENTRIES(q4_k, GGML_TYPE_Q4_K)
ATLAS_Q4_MMVQ_ENTRIES(q6_k, GGML_TYPE_Q6_K)

// The block geometry the host's byte arithmetic assumes (port/gguf_q4/launch.rs).
static_assert(sizeof(block_iq4_xs) == 136, "IQ4_XS super-block is 136 bytes");
static_assert(sizeof(block_q4_K) == 144, "Q4_K super-block is 144 bytes");
static_assert(sizeof(block_q6_K) == 210, "Q6_K super-block is 210 bytes");
static_assert(sizeof(block_q8_1) == 36, "block_q8_1 is 36 bytes");
static_assert(QI4_XS == 32 && QI4_K == 32 && QI6_K == 32, "one super-block is 32 quant words");
