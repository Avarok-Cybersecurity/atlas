// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b
//
// DeepSeek-V4.1 Flash routed experts on RAW Q2_K / Q3_K blocks (the Q2_K GGUF:
// gate/up are Q2_K, down is Q3_K). The expert bytes stream from the SSD and are
// consumed as-is: no BF16 expansion, no re-quantization. Measured on the real
// shards, a CPU dequant of one expert is 851 ms, which is 204 s per token; this
// file is what makes the streaming design serve at all.
//
// Two shapes, one activation format each:
//   * DECODE (M = 1, and the M <= 8 verify tier): `kquant_mmvq_q{2,3}_k`, the
//     llama.cpp mmvq form. The activation is quantised to plain `block_q8_1`
//     (32 values, d + sum) by `kquant_q8_1_rows_bf16`; each K super-block is
//     scored by the vendored `vec_dot_q{2,3}_K_q8_1` (dp4a on the packed codes,
//     no unpack to bytes). One CUDA block per output row, four warps.
//   * PREFILL (M up to the 2048 cap): `atlas_q{2,3}_k_mmq128_{nc,wc}`, the
//     vendored MMQ tensor-core engine with the compile-time type switched to
//     Q2_K / Q3_K, exactly as q2_0_mmq.cu does for Q2_0. Q2_K wants the D2S6
//     q8_1 layout and Q3_K the D4 layout (`mmq_get_q8_1_ds_layout`), so the two
//     activation quantisers below select those layouts.
//
// Math is the vendored ggml-cuda math; `--fmad=false` is inherited from the
// target's KERNEL.toml. Oracle: `ops/kquant_mmq_tests.rs` (CPU dequant + f32).

#include <cuda_bf16.h>
// Spelled from kernels/, not from this hardware set: kernels/hopper and
// kernels/b200 compile this file through a per-file symlink, and a quoted
// include resolves against the symlink's own directory, where no
// qwen3.6-27b tree exists on b200. The vendor headers live in gb10's tree
// for every hardware set that inherits it.
#include "../../../gb10/qwen3.6-27b/nvfp4/q4k_vendor/mmq.cuh"
#include "../../../gb10/qwen3.6-27b/nvfp4/q4k_vendor/quantize_impl.cuh"

// ── prefill: MMQ tiles with the K-quant types ────────────────────────────────

template <ggml_type type, int mmq_x, bool need_check>
static __device__ __forceinline__ void atlas_kq_tile(
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

#define KQ_MMQ_ENTRY(NAME, TYPE, CHECK)                                                              \
extern "C" __global__ void __launch_bounds__(256, 1) NAME(                                           \
        const char* x, const int* y, __nv_bfloat16* dst,                                             \
        int nrows_x, int ncols_dst, int ncols_x, int stride_row_x, int ncols_y, int stride_col_dst) {\
    atlas_kq_tile<TYPE, 128, CHECK>(x, y, dst, nrows_x, ncols_dst, ncols_x, stride_row_x, ncols_y, stride_col_dst); \
}
KQ_MMQ_ENTRY(atlas_q2_k_mmq128_nc, GGML_TYPE_Q2_K, false)
KQ_MMQ_ENTRY(atlas_q2_k_mmq128_wc, GGML_TYPE_Q2_K, true)
KQ_MMQ_ENTRY(atlas_q3_k_mmq128_nc, GGML_TYPE_Q3_K, false)
KQ_MMQ_ENTRY(atlas_q3_k_mmq128_wc, GGML_TYPE_Q3_K, true)

// MMQ activation quantisers, bf16 in, in the layout each type's dot expects.
extern "C" __global__ void atlas_q8_1_quantize_d2s6_bf16(
        const __nv_bfloat16* x, void* vy, long ne00, long s01, long ne0, int ne1) {
    quantize_mmq_q8_1_worker<MMQ_Q8_1_DS_LAYOUT_D2S6, __nv_bfloat16>(x, nullptr, vy, ne00, s01, 0, 0, ne0, ne1, 1);
}
extern "C" __global__ void atlas_q8_1_quantize_d4_bf16(
        const __nv_bfloat16* x, void* vy, long ne00, long s01, long ne0, int ne1) {
    quantize_mmq_q8_1_worker<MMQ_Q8_1_DS_LAYOUT_D4, __nv_bfloat16>(x, nullptr, vy, ne00, s01, 0, 0, ne0, ne1, 1);
}

// ── decode: plain q8_1 rows + mmvq ───────────────────────────────────────────

// bf16 [M, K] -> block_q8_1 [M, K/32]. One warp per 32-value block: d = amax/127,
// q = round(x/d), ds = (d, sum x). K must be a multiple of 32.
extern "C" __global__ void kquant_q8_1_rows_bf16(
        const __nv_bfloat16* __restrict__ x, void* __restrict__ vy,
        unsigned int K, unsigned int M) {
    const unsigned int nblk = K / QK8_1;
    const unsigned int gw   = (blockIdx.x * blockDim.x + threadIdx.x) / 32u;
    const unsigned int lane = threadIdx.x % 32u;
    if (gw >= M * nblk) return;
    const unsigned int m  = gw / nblk;
    const unsigned int ib = gw % nblk;
    const float xi = __bfloat162float(x[(size_t)m * K + (size_t)ib * QK8_1 + lane]);
    float amax = fabsf(xi);
    float sum  = xi;
    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);
    const float d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : (int8_t)roundf(xi / d);
    block_q8_1* y = (block_q8_1*)vy;
    y[gw].qs[lane] = q;
    if (lane == 0) y[gw].ds = make_half2(d, sum);
}

#define KQ_NWARPS 4
#define KQ_MAX_M  8

template <ggml_type type>
static __device__ __forceinline__ float kq_vec_dot(const void* vbq, const block_q8_1* bq8_1, const int kbx, const int iqs);
template <> __device__ __forceinline__ float kq_vec_dot<GGML_TYPE_Q2_K>(const void* vbq, const block_q8_1* bq8_1, const int kbx, const int iqs) {
    return vec_dot_q2_K_q8_1(vbq, bq8_1, kbx, iqs);
}
template <> __device__ __forceinline__ float kq_vec_dot<GGML_TYPE_Q3_K>(const void* vbq, const block_q8_1* bq8_1, const int kbx, const int iqs) {
    return vec_dot_q3_K_q8_1(vbq, bq8_1, kbx, iqs);
}

// One block (32 x KQ_NWARPS threads) per output row `blockIdx.x`; `m` activation
// rows (<= KQ_MAX_M) share every weight read. dst[j * nrows_x + row], bf16.
template <ggml_type type>
static __device__ __forceinline__ void kq_mmvq(
        const void* __restrict__ x_row, const block_q8_1* __restrict__ y,
        __nv_bfloat16* __restrict__ dst, const int ncols_x, const int nrows_x, const int m) {
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;   // 256
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;   // 16
    constexpr int vdr = 1;                                  // VDR_Q{2,3}_K_Q8_1_MMVQ
    constexpr int blocks_per_iter = vdr * KQ_NWARPS * 32 / qi;   // 8
    const int tid = 32 * threadIdx.y + threadIdx.x;
    const int row = blockIdx.x;
    const int blocks_per_row_x = ncols_x / qk;
    const int blocks_per_col_y = ncols_x / QK8_1;

    float tmp[KQ_MAX_M];
#pragma unroll
    for (int j = 0; j < KQ_MAX_M; ++j) tmp[j] = 0.0f;

    for (int kbx = tid / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);          // 8 q8_1 blocks per super-block
        const int kqs = vdr * (tid % (qi / vdr));
        for (int j = 0; j < m; ++j) {
            tmp[j] += kq_vec_dot<type>(x_row, &y[(size_t)j * blocks_per_col_y + kby], kbx, kqs);
        }
    }

    __shared__ float red[KQ_NWARPS][KQ_MAX_M];
#pragma unroll
    for (int j = 0; j < KQ_MAX_M; ++j) {
        const float v = warp_reduce_sum(tmp[j]);
        if (threadIdx.x == 0) red[threadIdx.y][j] = v;
    }
    __syncthreads();
    if (threadIdx.y == 0 && (int)threadIdx.x < m) {
        float s = 0.0f;
#pragma unroll
        for (int w = 0; w < KQ_NWARPS; ++w) s += red[w][threadIdx.x];
        dst[(size_t)threadIdx.x * nrows_x + row] = __float2bfloat16(s);
    }
}

// vx: [N rows][K/256 blocks] raw; vy: block_q8_1 [m][K/32]; dst: bf16 [m][N].
extern "C" __global__ void __launch_bounds__(128) kquant_mmvq_q2_k(
        const void* __restrict__ vx, const void* __restrict__ vy, __nv_bfloat16* __restrict__ dst,
        unsigned int ncols_x, unsigned int nrows_x, unsigned int m) {
    const char* x_row = (const char*)vx + (size_t)blockIdx.x * (ncols_x / QK_K) * sizeof(block_q2_K);
    kq_mmvq<GGML_TYPE_Q2_K>(x_row, (const block_q8_1*)vy, dst, (int)ncols_x, (int)nrows_x, (int)m);
}
extern "C" __global__ void __launch_bounds__(128) kquant_mmvq_q3_k(
        const void* __restrict__ vx, const void* __restrict__ vy, __nv_bfloat16* __restrict__ dst,
        unsigned int ncols_x, unsigned int nrows_x, unsigned int m) {
    const char* x_row = (const char*)vx + (size_t)blockIdx.x * (ncols_x / QK_K) * sizeof(block_q3_K);
    kq_mmvq<GGML_TYPE_Q3_K>(x_row, (const block_q8_1*)vy, dst, (int)ncols_x, (int)nrows_x, (int)m);
}

// Batched over experts for the single-token step: blockIdx.y picks the expert,
// whose blocks come from the pointer table `vxs`; its activation is
// `vy + blockIdx.y * y_stride_bytes` (0 = one activation shared by every
// expert, the token itself) and its output `dst + blockIdx.y * m * nrows_x`.
// The per-row math is kq_mmvq unchanged, so each expert's numbers match the
// one-expert launch bit for bit; the six launches a projection become one.
//
// Grid: (nrows_x, n_experts, 1)  Block: (32, KQ_NWARPS, 1)
extern "C" __global__ void __launch_bounds__(128) kquant_mmvq_q2_k_experts(
        const void* const* __restrict__ vxs, const void* __restrict__ vy, __nv_bfloat16* __restrict__ dst,
        unsigned int ncols_x, unsigned int nrows_x, unsigned int m, unsigned int y_stride_bytes) {
    const unsigned int e = blockIdx.y;
    const char* x_row = (const char*)vxs[e] + (size_t)blockIdx.x * (ncols_x / QK_K) * sizeof(block_q2_K);
    const block_q8_1* y = (const block_q8_1*)((const char*)vy + (size_t)e * y_stride_bytes);
    kq_mmvq<GGML_TYPE_Q2_K>(x_row, y, dst + (size_t)e * m * nrows_x, (int)ncols_x, (int)nrows_x, (int)m);
}
extern "C" __global__ void __launch_bounds__(128) kquant_mmvq_q3_k_experts(
        const void* const* __restrict__ vxs, const void* __restrict__ vy, __nv_bfloat16* __restrict__ dst,
        unsigned int ncols_x, unsigned int nrows_x, unsigned int m, unsigned int y_stride_bytes) {
    const unsigned int e = blockIdx.y;
    const char* x_row = (const char*)vxs[e] + (size_t)blockIdx.x * (ncols_x / QK_K) * sizeof(block_q3_K);
    const block_q8_1* y = (const block_q8_1*)((const char*)vy + (size_t)e * y_stride_bytes);
    kq_mmvq<GGML_TYPE_Q3_K>(x_row, y, dst + (size_t)e * m * nrows_x, (int)ncols_x, (int)nrows_x, (int)m);
}

// Warp-per-row variant. The rows here are short (ncols_x / 256 = 20
// super-blocks at dim 5120, 9 at inter 2304), so one 128-thread block plus a
// shared-memory reduction per row is mostly scheduling and reduction. Each
// WARP owns one output row (row = blockIdx.x * KQ_NWARPS + threadIdx.y): the
// 32 lanes stride the row's super-blocks two at a time (qi = 16 lanes per
// super-block), the same kq_vec_dot per lane and the same m activation rows
// sharing every weight read, reduced with one warp shuffle and no
// __syncthreads. The per-lane products are the existing kernel's; only the
// order of the final sum differs, so results agree to bf16 rounding.
template <ggml_type type>
static __device__ __forceinline__ void kq_mmvq_warp(
        const char* __restrict__ x_rows, const size_t row_bytes, const block_q8_1* __restrict__ y,
        __nv_bfloat16* __restrict__ dst, const int ncols_x, const int nrows_x, const int m) {
    constexpr int qk  = ggml_cuda_type_traits<type>::qk;   // 256
    constexpr int qi  = ggml_cuda_type_traits<type>::qi;   // 16
    constexpr int vdr = 1;
    constexpr int blocks_per_iter = vdr * 32 / qi;          // 2 super-blocks per warp iteration
    const int lane = threadIdx.x;
    const int row = blockIdx.x * KQ_NWARPS + threadIdx.y;
    if (row >= nrows_x) return;
    const char* x_row = x_rows + (size_t)row * row_bytes;
    const int blocks_per_row_x = ncols_x / qk;
    const int blocks_per_col_y = ncols_x / QK8_1;
    float tmp[KQ_MAX_M];
#pragma unroll
    for (int j = 0; j < KQ_MAX_M; ++j) tmp[j] = 0.0f;
    for (int kbx = lane / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1);
        const int kqs = vdr * (lane % (qi / vdr));
        for (int j = 0; j < m; ++j) {
            tmp[j] += kq_vec_dot<type>(x_row, &y[(size_t)j * blocks_per_col_y + kby], kbx, kqs);
        }
    }
#pragma unroll
    for (int j = 0; j < KQ_MAX_M; ++j) {
        const float v = warp_reduce_sum(tmp[j]);
        if (lane == 0 && j < m) dst[(size_t)j * nrows_x + row] = __float2bfloat16(v);
    }
}

// Grid: (ceil(nrows_x / KQ_NWARPS), 1, 1)  Block: (32, KQ_NWARPS, 1)
extern "C" __global__ void __launch_bounds__(128) kquant_mmvq_q2_k_w(
        const void* __restrict__ vx, const void* __restrict__ vy, __nv_bfloat16* __restrict__ dst,
        unsigned int ncols_x, unsigned int nrows_x, unsigned int m) {
    kq_mmvq_warp<GGML_TYPE_Q2_K>((const char*)vx, (size_t)(ncols_x / QK_K) * sizeof(block_q2_K),
                                 (const block_q8_1*)vy, dst, (int)ncols_x, (int)nrows_x, (int)m);
}
extern "C" __global__ void __launch_bounds__(128) kquant_mmvq_q3_k_w(
        const void* __restrict__ vx, const void* __restrict__ vy, __nv_bfloat16* __restrict__ dst,
        unsigned int ncols_x, unsigned int nrows_x, unsigned int m) {
    kq_mmvq_warp<GGML_TYPE_Q3_K>((const char*)vx, (size_t)(ncols_x / QK_K) * sizeof(block_q3_K),
                                 (const block_q8_1*)vy, dst, (int)ncols_x, (int)nrows_x, (int)m);
}

// Grid: (ceil(nrows_x / KQ_NWARPS), n_experts, 1)  Block: (32, KQ_NWARPS, 1)
extern "C" __global__ void __launch_bounds__(128) kquant_mmvq_q2_k_experts_w(
        const void* const* __restrict__ vxs, const void* __restrict__ vy, __nv_bfloat16* __restrict__ dst,
        unsigned int ncols_x, unsigned int nrows_x, unsigned int m, unsigned int y_stride_bytes) {
    const unsigned int e = blockIdx.y;
    const block_q8_1* y = (const block_q8_1*)((const char*)vy + (size_t)e * y_stride_bytes);
    kq_mmvq_warp<GGML_TYPE_Q2_K>((const char*)vxs[e], (size_t)(ncols_x / QK_K) * sizeof(block_q2_K),
                                 y, dst + (size_t)e * m * nrows_x, (int)ncols_x, (int)nrows_x, (int)m);
}
extern "C" __global__ void __launch_bounds__(128) kquant_mmvq_q3_k_experts_w(
        const void* const* __restrict__ vxs, const void* __restrict__ vy, __nv_bfloat16* __restrict__ dst,
        unsigned int ncols_x, unsigned int nrows_x, unsigned int m, unsigned int y_stride_bytes) {
    const unsigned int e = blockIdx.y;
    const block_q8_1* y = (const block_q8_1*)((const char*)vy + (size_t)e * y_stride_bytes);
    kq_mmvq_warp<GGML_TYPE_Q3_K>((const char*)vxs[e], (size_t)(ncols_x / QK_K) * sizeof(block_q3_K),
                                 y, dst + (size_t)e * m * nrows_x, (int)ncols_x, (int)nrows_x, (int)m);
}
