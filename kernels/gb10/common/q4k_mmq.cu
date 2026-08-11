// SPDX-License-Identifier: AGPL-3.0-only
//
// ATLAS FFN prefill GEMM via vendored llama.cpp Q4_K MMQ (verified 56 TFLOP/s, 1.3x faith2).
// extern-C entries launched from Rust (ops/q4k_mmq.rs). Conventional 2D tiling (no MoE ids,
// single channel/sample); our prefill shapes have thousands of tiles >> 48 SMs so stream-k's
// load-balancing buys ~nothing. Output is F32 [M,N] row-major (dst[m*N+n]); Rust casts to BF16.
// Vendored headers in q4k_vendor/ (pristine except quantize_impl.cuh worker extraction).
#include <cuda_bf16.h>
#include "q4k_vendor/mmq.cuh"
#include "q4k_vendor/quantize_impl.cuh"

// Conventional-tiling setup mirroring mul_mat_q's pre-VOLTA path, specialized: no ids,
// nchannels_y=nsamples_y=1 (blockIdx.z==0). Calls the existing __device__ process_tile.
template <int mmq_x, bool need_check>
static __device__ __forceinline__ void atlas_q4k_tile(
        const char * __restrict__ x, const int * __restrict__ y, __nv_bfloat16 * __restrict__ dst,
        const int nrows_x, const int ncols_dst, const int ncols_x,
        const int stride_row_x, const int ncols_y, const int stride_col_dst) {
    constexpr ggml_type type = GGML_TYPE_Q4_K;
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
    const int kb0_stop = ncols_x / qk;   // number of K-blocks

    mul_mat_q_process_tile<type, mmq_x, need_check, /*fixup=*/false, __nv_bfloat16>(
        x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, nullptr,
        stride_row_x, ncols_y, stride_col_dst, tile_x_max_i, tile_y_max_j, 0, kb0_stop);
}

// mmq_x=128 entries (need_check = nrows_x not a multiple of mmq_y=128).
extern "C" __global__ void __launch_bounds__(256, 1) atlas_q4k_mmq128_nc(
        const char* x, const int* y, __nv_bfloat16* dst,
        int nrows_x, int ncols_dst, int ncols_x, int stride_row_x, int ncols_y, int stride_col_dst) {
    atlas_q4k_tile<128, false>(x, y, dst, nrows_x, ncols_dst, ncols_x, stride_row_x, ncols_y, stride_col_dst);
}
extern "C" __global__ void __launch_bounds__(256, 1) atlas_q4k_mmq128_wc(
        const char* x, const int* y, __nv_bfloat16* dst,
        int nrows_x, int ncols_dst, int ncols_x, int stride_row_x, int ncols_y, int stride_col_dst) {
    atlas_q4k_tile<128, true>(x, y, dst, nrows_x, ncols_dst, ncols_x, stride_row_x, ncols_y, stride_col_dst);
}

// Activation quantizer: f32 [ne1=M rows, ne00=K] -> block_q8_1_mmq (DS4 layout for Q4_K).
// grid (ne1, ceil(ne0/(4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ)), 1), block (CUDA_QUANTIZE_BLOCK_SIZE_MMQ).
extern "C" __global__ void atlas_q8_1_quantize_ds4(
        const float* x, void* vy, long ne00, long s01, long ne0, int ne1) {
    quantize_mmq_q8_1_worker<MMQ_Q8_1_DS_LAYOUT_DS4, float>(x, nullptr, vy, ne00, s01, 0, 0, ne0, ne1, 1);
}
// bf16-input variant (Atlas activations are bf16) — avoids a bf16->f32 cast + scratch.
extern "C" __global__ void atlas_q8_1_quantize_ds4_bf16(
        const __nv_bfloat16* x, void* vy, long ne00, long s01, long ne0, int ne1) {
    quantize_mmq_q8_1_worker<MMQ_Q8_1_DS_LAYOUT_DS4, __nv_bfloat16>(x, nullptr, vy, ne00, s01, 0, 0, ne0, ne1, 1);
}
// D4-layout q8_1 activation quantizer (Q6_K MMQ expects DS_LAYOUT_D4, not DS4).
extern "C" __global__ void atlas_q8_1_quantize_d4_bf16(
        const __nv_bfloat16* x, void* vy, long ne00, long s01, long ne0, int ne1) {
    quantize_mmq_q8_1_worker<MMQ_Q8_1_DS_LAYOUT_D4, __nv_bfloat16>(x, nullptr, vy, ne00, s01, 0, 0, ne0, ne1, 1);
}

// ── Grouped (MoE) tile: one grid over (N-tiles, worst-case M-tiles, num_experts).
// Each CTA self-selects its expert via blockIdx.z and reads that expert's row
// range [col_low,col_high) from the DEVICE expert_offsets — no host readback, so
// this is CUDA-graph-capture-legal. Weights are one contiguous per-proj stack
// (emit_experts_packed): expert e's blocks start at e*stride_channel_x. Output is
// written in SORTED (expert-contiguous) order via ids_dst = global sorted row, so
// the caller's moe_unpermute_reduce_indexed scatters it unchanged. Mirrors the
// vendored mul_mat_q pre-VOLTA grouped path (mmq.cuh) as conventional tiling.
template <ggml_type type, int mmq_x, bool need_check>
static __device__ __forceinline__ void atlas_kq_tile_grouped(
        const char * __restrict__ x, const int * __restrict__ y,
        const int32_t * __restrict__ expert_offsets, __nv_bfloat16 * __restrict__ dst,
        const int nrows_x, const int ncols_x, const int stride_row_x,
        const int stride_channel_x, const int ncols_y, const int stride_col_dst) {
    constexpr int nwarps    = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int qk        = ggml_cuda_type_traits<type>::qk;
    constexpr int mmq_y     = get_mmq_y_device();

    const int zt = blockIdx.z;   // expert
    const int jt = blockIdx.y;   // tile over this expert's M (token) rows
    const int it = blockIdx.x;   // tile over nrows_x (N output features)

    const int col_low  = expert_offsets[zt];
    const int col_high = expert_offsets[zt + 1];
    const int col_diff = col_high - col_low;
    if (jt*mmq_x >= col_diff) {
        return;   // this tile has no rows for this expert
    }

    extern __shared__ int ids_dst_shared[];
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps*warp_size) {
        const int j = j0 + threadIdx.y*warp_size + threadIdx.x;
        if (j0 + nwarps*warp_size > mmq_x && j >= mmq_x) break;
        ids_dst_shared[j] = col_low + jt*mmq_x + j;   // global sorted dst row
    }
    __syncthreads();

    const int offset_y   = (col_low + jt*mmq_x) * (int)(sizeof(block_q8_1_mmq)/sizeof(int));
    const int offset_x   = zt*stride_channel_x + it*mmq_y*stride_row_x;
    const int offset_dst = it*mmq_y;   // N-feature tile; row comes from ids_dst
    const int tile_x_max_i = nrows_x - it*mmq_y - 1;
    const int tile_y_max_j = col_diff - jt*mmq_x - 1;
    const int kb0_stop   = ncols_x / qk;

    mul_mat_q_process_tile<type, mmq_x, need_check, /*fixup=*/false, __nv_bfloat16>(
        x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, nullptr,
        stride_row_x, ncols_y, stride_col_dst, tile_x_max_i, tile_y_max_j, 0, kb0_stop);
}

// extern-C grouped entries. grid=(ceil(nrows_x/128), max_m_tiles, num_experts),
// block=(32,8). need_check(wc) only guards nrows_x%128!=0; Laguna N (inter/hidden)
// is 128-aligned so nc is used, but both are provided.
extern "C" __global__ void __launch_bounds__(256, 1) atlas_q4k_mmq128_grouped_nc(
        const char* x, const int* y, const int* expert_offsets, __nv_bfloat16* dst,
        int nrows_x, int ncols_x, int stride_row_x, int stride_channel_x, int ncols_y, int stride_col_dst) {
    atlas_kq_tile_grouped<GGML_TYPE_Q4_K, 128, false>(x, y, expert_offsets, dst, nrows_x, ncols_x, stride_row_x, stride_channel_x, ncols_y, stride_col_dst);
}
extern "C" __global__ void __launch_bounds__(256, 1) atlas_q4k_mmq128_grouped_wc(
        const char* x, const int* y, const int* expert_offsets, __nv_bfloat16* dst,
        int nrows_x, int ncols_x, int stride_row_x, int stride_channel_x, int ncols_y, int stride_col_dst) {
    atlas_kq_tile_grouped<GGML_TYPE_Q4_K, 128, true>(x, y, expert_offsets, dst, nrows_x, ncols_x, stride_row_x, stride_channel_x, ncols_y, stride_col_dst);
}
extern "C" __global__ void __launch_bounds__(256, 1) atlas_q6k_mmq128_grouped_nc(
        const char* x, const int* y, const int* expert_offsets, __nv_bfloat16* dst,
        int nrows_x, int ncols_x, int stride_row_x, int stride_channel_x, int ncols_y, int stride_col_dst) {
    atlas_kq_tile_grouped<GGML_TYPE_Q6_K, 128, false>(x, y, expert_offsets, dst, nrows_x, ncols_x, stride_row_x, stride_channel_x, ncols_y, stride_col_dst);
}
extern "C" __global__ void __launch_bounds__(256, 1) atlas_q6k_mmq128_grouped_wc(
        const char* x, const int* y, const int* expert_offsets, __nv_bfloat16* dst,
        int nrows_x, int ncols_x, int stride_row_x, int stride_channel_x, int ncols_y, int stride_col_dst) {
    atlas_kq_tile_grouped<GGML_TYPE_Q6_K, 128, true>(x, y, expert_offsets, dst, nrows_x, ncols_x, stride_row_x, stride_channel_x, ncols_y, stride_col_dst);
}

// ── Fused (MoE) grouped gate+up tile: ONE CTA computes BOTH projections. gate
// and up are shape-identical [inter,h], so the empty-expert guard + ids_dst
// setup run once and mul_mat_q_process_tile (verified) runs twice against the
// same activation range. grid.x = ceil(inter/128) (NOT doubled) => half the
// CTAs of two separate grouped launches; sum resets to 64 floats per call so
// no register spill; smem unchanged at 57856. Writes gate_dst and up_dst in
// SORTED order for the caller's silu_mul + unpermute-reduce.
template <ggml_type type, int mmq_x, bool need_check>
static __device__ __forceinline__ void atlas_kq_tile_grouped_gate_up(
        const char * __restrict__ gate_x, const char * __restrict__ up_x,
        const int * __restrict__ y, const int32_t * __restrict__ expert_offsets,
        __nv_bfloat16 * __restrict__ gate_dst, __nv_bfloat16 * __restrict__ up_dst,
        const int nrows_x, const int ncols_x, const int stride_row_x,
        const int stride_channel_x, const int ncols_y, const int stride_col_dst) {
    constexpr int nwarps    = mmq_get_nwarps_device();
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int qk        = ggml_cuda_type_traits<type>::qk;
    constexpr int mmq_y     = get_mmq_y_device();

    const int zt = blockIdx.z;   // expert
    const int jt = blockIdx.y;   // tile over this expert's M (token) rows
    const int it = blockIdx.x;   // tile over nrows_x (N output features)

    const int col_low  = expert_offsets[zt];
    const int col_high = expert_offsets[zt + 1];
    const int col_diff = col_high - col_low;
    if (jt*mmq_x >= col_diff) {
        return;   // empty-expert CTA: skip BOTH gate and up in one shot
    }

    extern __shared__ int ids_dst_shared[];
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps*warp_size) {
        const int j = j0 + threadIdx.y*warp_size + threadIdx.x;
        if (j0 + nwarps*warp_size > mmq_x && j >= mmq_x) break;
        ids_dst_shared[j] = col_low + jt*mmq_x + j;   // global sorted dst row
    }
    __syncthreads();

    const int offset_y     = (col_low + jt*mmq_x) * (int)(sizeof(block_q8_1_mmq)/sizeof(int));
    const int offset_x     = zt*stride_channel_x + it*mmq_y*stride_row_x; // SAME for gate & up
    const int offset_dst   = it*mmq_y;
    const int tile_x_max_i = nrows_x - it*mmq_y - 1;
    const int tile_y_max_j = col_diff - jt*mmq_x - 1;
    const int kb0_stop     = ncols_x / qk;

    // GATE: fresh sum (64 regs), loads tile_x=gate + tile_y from global.
    mul_mat_q_process_tile<type, mmq_x, need_check, /*fixup=*/false, __nv_bfloat16>(
        gate_x, offset_x, y + offset_y, ids_dst_shared, gate_dst + offset_dst, nullptr,
        stride_row_x, ncols_y, stride_col_dst, tile_x_max_i, tile_y_max_j, 0, kb0_stop);

    // WAR fence before UP overwrites tile_x/tile_y in shared memory.
    __syncthreads();

    // UP: sum resets to 0, SAME y+offset_y => numerically identical to the
    // separate up launch.
    mul_mat_q_process_tile<type, mmq_x, need_check, /*fixup=*/false, __nv_bfloat16>(
        up_x, offset_x, y + offset_y, ids_dst_shared, up_dst + offset_dst, nullptr,
        stride_row_x, ncols_y, stride_col_dst, tile_x_max_i, tile_y_max_j, 0, kb0_stop);
}

extern "C" __global__ void __launch_bounds__(256, 1) atlas_q4k_mmq128_grouped_gate_up_nc(
        const char* gate_x, const char* up_x, const int* y, const int* expert_offsets,
        __nv_bfloat16* gate_dst, __nv_bfloat16* up_dst,
        int nrows_x, int ncols_x, int stride_row_x, int stride_channel_x, int ncols_y, int stride_col_dst) {
    atlas_kq_tile_grouped_gate_up<GGML_TYPE_Q4_K, 128, false>(
        gate_x, up_x, y, expert_offsets, gate_dst, up_dst,
        nrows_x, ncols_x, stride_row_x, stride_channel_x, ncols_y, stride_col_dst);
}
extern "C" __global__ void __launch_bounds__(256, 1) atlas_q4k_mmq128_grouped_gate_up_wc(
        const char* gate_x, const char* up_x, const int* y, const int* expert_offsets,
        __nv_bfloat16* gate_dst, __nv_bfloat16* up_dst,
        int nrows_x, int ncols_x, int stride_row_x, int stride_channel_x, int ncols_y, int stride_col_dst) {
    atlas_kq_tile_grouped_gate_up<GGML_TYPE_Q4_K, 128, true>(
        gate_x, up_x, y, expert_offsets, gate_dst, up_dst,
        nrows_x, ncols_x, stride_row_x, stride_channel_x, ncols_y, stride_col_dst);
}
