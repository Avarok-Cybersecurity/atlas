// SPDX-License-Identifier: AGPL-3.0-only AND MIT
// After ExLlamaV3 (turboderp-org/exllamav3), MIT, Copyright (c) 2025 Turboderp, via TensorFold
// zig/kernels/cuda/exl3_linear.cu:107-127 (rot_in_body, tf_exl3_rot_in) and
// zig/src/core/exl3_format.zig:174-187 (dequantize: W = diag(suh) H_K Wq H_N diag(svh)) (Apache-2.0,
// ashhart/TensorFold, commit 3885dce). See EXL3-ATTRIBUTION.md and LICENSES/ExLlamaV3-MIT.txt.
// Changes for Atlas: the rotation reads bf16 only (the dtype switch is dropped) and its PDL calls are
// dropped (sm_90 only); entry renamed exl3_ada_rot_in_bf16 with plain parameters; the BF16 dequant kernel
// is new for Atlas: TensorFold's dequantize is a scalar FP64 CPU reference, translated here into a
// shared-memory CUDA kernel over the same lane decoder and butterfly as the linear kernel.

// Ada (sm_89) EXL3 Hadamard-domain transforms: the input rotation of the decode-row linear and the BF16
// dequant that serves the W4Linear prefill fallback and the oracle.
//
// exl3_ada_rot_in_bf16: xh [M, K] fp16 = fp16(((x * suh) @ H) / sqrt(128)) per 128-value block, x bf16.
//   Grid (ceil(K / 512), M), 128 threads (one warp a 128-value block). Arithmetic, which the CPU mirror
//   (port/exl3/reference_math.rs, rot_in_mirror) reproduces bit for bit: v = f32(x) * f32(suh), the fixed
//   fp32 butterfly fwht128, v * HAD_SCALE, one round to fp16.
//
// exl3_ada_dequant_bf16_<K2>_<CB>: out [N, K] bf16 row-major (the DenseWeight layout), out[n][k] = W[k][n].
//   Grid (N / 128, K / 128), 256 threads, dynamic shared memory 128 x 129 fp32 (66,048 bytes; the
//   registry opts the function in above 48 KiB). Per 128 x 128 block: the 64 tiles decoded to fp32
//   (exact fp16 values), every row along n through fwht128 then * HAD_SCALE, every column along k through
//   fwht128 then ((v * HAD_SCALE) * suh[k]) * svh[n], one round to bf16. Ada common builds with
//   --fmad=false, so no product is contracted and the CPU mirror (dequant_bf16_mirror) is bit-exact.

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "exl3_decode_ada.cuh"

namespace avarok_exl3 {

// Shared-memory row pitch of the dequant block (129: the row and column passes stay at most 4-way
// bank-conflicted).
constexpr int DQ_LD = 129;
constexpr int DQ_THREADS = 256;

__device__ __forceinline__ void rot_in_body(const __nv_bfloat16* __restrict__ x, const half* __restrict__ suh,
                                            half* __restrict__ xh, int kdim) {
    const int blk = blockIdx.x * 4 + (threadIdx.x >> 5), row = blockIdx.y, lane = threadIdx.x & 31;
    if (blk * 128 >= kdim) return;                       // a whole warp leaves together (blk is warp-uniform)
    const int k = blk * 128 + 4 * lane;
    const size_t at = (size_t)row * kdim + k;
    const uint2 ux = *reinterpret_cast<const uint2*>(x + at);
    const float2 xa = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&ux.x));
    const float2 xb = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&ux.y));
    const uint2 us = *reinterpret_cast<const uint2*>(suh + k);
    const float2 sa = __half22float2(*reinterpret_cast<const half2*>(&us.x));
    const float2 sb = __half22float2(*reinterpret_cast<const half2*>(&us.y));
    float v[4] = {xa.x * sa.x, xa.y * sa.y, xb.x * sb.x, xb.y * sb.y};
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] *= HAD_SCALE;
    half2 a = __floats2half2_rn(v[0], v[1]), b = __floats2half2_rn(v[2], v[3]);
    uint2 u;
    u.x = *reinterpret_cast<uint32_t*>(&a);
    u.y = *reinterpret_cast<uint32_t*>(&b);
    *reinterpret_cast<uint2*>(xh + at) = u;
}

template <int K2, int CB>
__device__ __forceinline__ void dequant_body(const uint32_t* __restrict__ t, long long stride_k, long long stride_nb,
                                             const half* __restrict__ suh, const half* __restrict__ svh,
                                             __nv_bfloat16* __restrict__ out, int K) {
    constexpr int TW = tile_words<K2>(), LW = lane_words<K2>();
    extern __shared__ __align__(16) float s[];           // [128][DQ_LD]
    const int nb = blockIdx.x, kb = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const uint32_t* base = t + nb * stride_nb + (long long)(kb * 8) * stride_k;

    // 1. the block's 8 x 8 tiles, 8 a warp, as exact fp32 values of the decoded fp16 states
#pragma unroll 1
    for (int i = 0; i < 8; ++i) {
        const int q = warp * 8 + i, tk = q >> 3, tn = q & 7;
        uint32_t w[LW];
        ldg_lane_words<K2>(base + (long long)tk * stride_k + tn * TW, lane, w);
        uint32_t b0[2], b1[2];
        decode_lane<K2, CB>(w, lane, b0, b1);
        const uint32_t hv[4] = {b0[0], b0[1], b1[0], b1[1]};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const half2 pair = *reinterpret_cast<const half2*>(&hv[j >> 1]);
            const float f = __half2float((j & 1) ? __high2half(pair) : __low2half(pair));
            s[(tk * 16 + value_row(lane, j)) * DQ_LD + tn * 16 + value_col(lane, j)] = f;
        }
    }
    __syncthreads();

    // 2. Wq H_N / sqrt(128): every k row along n, 16 rows a warp
#pragma unroll 1
    for (int r = warp * 16; r < warp * 16 + 16; ++r) {
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = s[r * DQ_LD + 4 * lane + j];
        fwht128(v, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) s[r * DQ_LD + 4 * lane + j] = v[j] * HAD_SCALE;
    }
    __syncthreads();

    // 3. H_K (..) / sqrt(128), then the scales: every n column along k, 16 columns a warp, stored as row n
#pragma unroll 1
    for (int c = warp * 16; c < warp * 16 + 16; ++c) {
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = s[(4 * lane + j) * DQ_LD + c];
        fwht128(v, lane);
        const int n = nb * 128 + c, k0 = kb * 128 + 4 * lane;
        const float sv = __half2float(svh[n]);
#pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = v[j] * HAD_SCALE * __half2float(suh[k0 + j]) * sv;
        __nv_bfloat162 a = __floats2bfloat162_rn(v[0], v[1]), b = __floats2bfloat162_rn(v[2], v[3]);
        uint2 u;
        u.x = *reinterpret_cast<uint32_t*>(&a);
        u.y = *reinterpret_cast<uint32_t*>(&b);
        *reinterpret_cast<uint2*>(out + (size_t)n * K + k0) = u;
    }
}

}  // namespace avarok_exl3

extern "C" __global__ void __launch_bounds__(128)
    exl3_ada_rot_in_bf16(const __nv_bfloat16* __restrict__ x, const half* __restrict__ suh, half* __restrict__ xh,
                         int m, int k) {
    (void)m;
    avarok_exl3::rot_in_body(x, suh, xh, k);
}

#define EXL3_ADA_DEQUANT(K2, CB)                                                                         \
    extern "C" __global__ void __launch_bounds__(avarok_exl3::DQ_THREADS) exl3_ada_dequant_bf16_##K2##_##CB( \
        const uint32_t* __restrict__ t, long long stride_k, long long stride_nb, const half* __restrict__ suh, \
        const half* __restrict__ svh, __nv_bfloat16* __restrict__ out, int n, int k) {                   \
        (void)n;                                                                                         \
        avarok_exl3::dequant_body<K2, CB>(t, stride_k, stride_nb, suh, svh, out, k);                    \
    }

// The same width and codebook set as exl3_linear_ada.cu, written out at file scope for the same reason:
// the build's shadow detector expands one level of instantiation macro, not a nested one.
EXL3_ADA_DEQUANT(2, 0) EXL3_ADA_DEQUANT(4, 0) EXL3_ADA_DEQUANT(6, 0) EXL3_ADA_DEQUANT(8, 0)  // 3inst
EXL3_ADA_DEQUANT(10, 0) EXL3_ADA_DEQUANT(12, 0) EXL3_ADA_DEQUANT(14, 0) EXL3_ADA_DEQUANT(16, 0)
EXL3_ADA_DEQUANT(2, 1) EXL3_ADA_DEQUANT(4, 1) EXL3_ADA_DEQUANT(6, 1) EXL3_ADA_DEQUANT(8, 1)  // mcg
EXL3_ADA_DEQUANT(10, 1) EXL3_ADA_DEQUANT(12, 1) EXL3_ADA_DEQUANT(14, 1) EXL3_ADA_DEQUANT(16, 1)
EXL3_ADA_DEQUANT(2, 2) EXL3_ADA_DEQUANT(4, 2) EXL3_ADA_DEQUANT(6, 2) EXL3_ADA_DEQUANT(8, 2)  // mul1
EXL3_ADA_DEQUANT(10, 2) EXL3_ADA_DEQUANT(12, 2) EXL3_ADA_DEQUANT(14, 2) EXL3_ADA_DEQUANT(16, 2)
EXL3_ADA_DEQUANT(3, 2) EXL3_ADA_DEQUANT(5, 2) EXL3_ADA_DEQUANT(7, 2) EXL3_ADA_DEQUANT(9, 2)
EXL3_ADA_DEQUANT(11, 2) EXL3_ADA_DEQUANT(13, 2)

#undef EXL3_ADA_DEQUANT
