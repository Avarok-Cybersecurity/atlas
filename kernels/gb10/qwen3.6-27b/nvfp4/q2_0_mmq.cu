// SPDX-License-Identifier: AGPL-3.0-only
//
// ATLAS native Ternary-Bonsai Q2_0 prefill GEMM via the vendored llama.cpp MMQ
// engine — the Tier-2 replacement for the transient-dequant prefill stopgap.
// The 2-bit weight stays PACKED: `load_tiles_q2_0` (q4k_vendor/mmq.cuh) unpacks
// codes into int8 `(code-1)` in-register into the Q8_0 tensor-core tile, then
// the stock `vec_dot_q8_0_q8_1_mma` runs the int8 MMA and folds the fp16 block
// scale d and the q8_1 activation scale. No BF16 weight scratch, no dequant tax,
// no shared-buffer co-dispatch race. Output is BF16 [M,N] (fused store, F32 accum).
//
// Structurally identical to q4k_mmq.cu (same 2D tiling, same mmq_x=mmq_y=128,
// same __nv_bfloat16 fused-store template arg); only the compile-time `type`
// differs. The q8_1 activation quantizer is SHARED with Q4_K — reuse
// `atlas_q8_1_quantize_ds4_bf16` (DS4 layout) from q4k_mmq.cu; none is defined
// here. Q2_0 uses DS4 too (the `(code-1)*d` dequant never reads q8_1's `s` term).
#include <cuda_bf16.h>
#include "q4k_vendor/mmq.cuh"
#include "q4k_vendor/quantize_impl.cuh"

// Conventional-tiling setup (no MoE ids, nchannels_y=nsamples_y=1) mirroring
// atlas_q4k_tile — calls the shared __device__ mul_mat_q_process_tile with the
// Q2_0 compile-time type so the Q2_0 trait table (load_tiles_q2_0 + q8_0 vec_dot)
// is selected. stride_row_x is K/QK2_0 (K/128).
template <int mmq_x, bool need_check>
static __device__ __forceinline__ void atlas_q2_0_tile(
        const char * __restrict__ x, const int * __restrict__ y, __nv_bfloat16 * __restrict__ dst,
        const int nrows_x, const int ncols_dst, const int ncols_x,
        const int stride_row_x, const int ncols_y, const int stride_col_dst) {
    constexpr ggml_type type = GGML_TYPE_Q2_0;
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
    const int kb0_stop = ncols_x / qk;   // number of K-blocks (K/128)

    mul_mat_q_process_tile<type, mmq_x, need_check, /*fixup=*/false, __nv_bfloat16>(
        x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, nullptr,
        stride_row_x, ncols_y, stride_col_dst, tile_x_max_i, tile_y_max_j, 0, kb0_stop);
}

// mmq_x=128 entries (need_check = nrows_x not a multiple of mmq_y=128).
extern "C" __global__ void __launch_bounds__(256, 1) atlas_q2_0_mmq128_nc(
        const char* x, const int* y, __nv_bfloat16* dst,
        int nrows_x, int ncols_dst, int ncols_x, int stride_row_x, int ncols_y, int stride_col_dst) {
    atlas_q2_0_tile<128, false>(x, y, dst, nrows_x, ncols_dst, ncols_x, stride_row_x, ncols_y, stride_col_dst);
}
extern "C" __global__ void __launch_bounds__(256, 1) atlas_q2_0_mmq128_wc(
        const char* x, const int* y, __nv_bfloat16* dst,
        int nrows_x, int ncols_dst, int ncols_x, int stride_row_x, int ncols_y, int stride_col_dst) {
    atlas_q2_0_tile<128, true>(x, y, dst, nrows_x, ncols_dst, ncols_x, stride_row_x, ncols_y, stride_col_dst);
}
