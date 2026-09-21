// SPDX-License-Identifier: AGPL-3.0-only
//
// Runtime BF16 -> FP8 (E4M3) quantization with 128x128 BLOCK scales.
//
// This is the load-time weight quantizer for checkpoints that ship plain BF16
// with no calibration metadata (LongCat-Flash-Lite is the first). Atlas's only
// runtime quantizer for that case was BF16 -> NVFP4, which costs real output
// quality on models whose weights are 90% routed experts; FP8 sits between the
// two and fits where BF16 does not.
//
// Output layout is the one `moe_fp8_grouped_gemm.cu` already consumes, so no
// kernel on the consuming side changes:
//
//   B[N, K]                 uint8 E4M3, row-major
//   block_scale[N/128, K/128]  FP32, row-major, indexed S[n_block*k_blocks + k_block]
//   dequant: bf16_val = E4M3_LUT[byte] * block_scale[n/128, k/128]
//
// Per [128,128] tile:  scale = max|X| / 448   (floored, never zero)
//                      byte  = round_e4m3(X / scale)   saturating to +-448
//
// One CTA per tile, 256 threads, two passes over the tile's 16384 elements
// (absmax, then encode). Partial edge tiles are handled — N and K need not be
// multiples of 128, though every shape Atlas feeds this today is.
//
// NOTE on `k_blocks`: it is ceil(K/128), matching the consumer's
// `(K + FP8_BLOCK - 1) / FP8_BLOCK`. Getting this rounding wrong silently
// shears the scale table by one column per row, which reads as "FP8 is
// inaccurate" rather than as an indexing bug.
//
// Grid: (k_blocks, n_blocks, 1)   Block: (256, 1, 1)

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define QBS_FP8_MAX 448.0f
#define QBS_BLOCK 128
#define QBS_THREADS 256

#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
// gfx1151/SCALE software SATFINITE E4M3 encode — mirrors quant_rowwise_fp8.cu;
// __nv_cvt_float_to_fp8 is non-standard there.
__device__ __forceinline__ unsigned char qbs_enc_fp8(float v) {
    if (v != v) return 0x7F;
    unsigned int bb = __float_as_uint(v); unsigned int sign = (bb >> 31) & 1u;
    int e = (int)((bb >> 23) & 0xFF) - 127; unsigned int man = bb & 0x7FFFFFu;
    int ee = e + 7; unsigned int em;
    if (ee < 1) { ee = 0; em = 0; if (e >= -10) { float a = v < 0 ? -v : v; em = (unsigned int)(a / 0.001953125f + 0.5f); if (em > 7u) em = 7u; } }
    else if (ee > 15) { ee = 15; em = 6; }
    else { em = (man + (1u << 19)) >> 20; if (em > 7u) { em = 0; ee++; if (ee > 15) { ee = 15; em = 6; } } }
    return (unsigned char)((sign << 7) | ((unsigned)ee << 3) | em);
}
#endif

extern "C" __global__ void quantize_bf16_to_fp8_blockscaled(
    const __nv_bfloat16* __restrict__ X,   // [N, K] BF16
    unsigned char* __restrict__ X_fp8,     // [N, K] FP8 E4M3
    float* __restrict__ block_scale,       // [ceil(N/128), ceil(K/128)] FP32
    unsigned int N,
    unsigned int K
) {
    const unsigned int k_block = blockIdx.x;
    const unsigned int n_block = blockIdx.y;
    const unsigned int k_blocks = (K + QBS_BLOCK - 1) / QBS_BLOCK;

    const unsigned int n0 = n_block * QBS_BLOCK;
    const unsigned int k0 = k_block * QBS_BLOCK;
    if (n0 >= N || k0 >= K) return;

    const unsigned int n_len = min((unsigned int)QBS_BLOCK, N - n0);
    const unsigned int k_len = min((unsigned int)QBS_BLOCK, K - k0);
    const unsigned int count = n_len * k_len;

    const unsigned int tid = threadIdx.x;

    // 1. Tile absmax.
    float my_max = 0.0f;
    for (unsigned int i = tid; i < count; i += QBS_THREADS) {
        const unsigned int r = i / k_len;
        const unsigned int c = i - r * k_len;
        const unsigned long long off = (unsigned long long)(n0 + r) * K + (k0 + c);
        my_max = fmaxf(my_max, fabsf(__bfloat162float(X[off])));
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        my_max = fmaxf(my_max, __shfl_down_sync(0xFFFFFFFF, my_max, o));
    }
    __shared__ float smem_warp_max[QBS_THREADS / 32];
    const unsigned int warp_id = tid >> 5;
    const unsigned int lane = tid & 31;
    if (lane == 0) smem_warp_max[warp_id] = my_max;
    __syncthreads();

    __shared__ float smem_scale;
    if (tid == 0) {
        float gmax = 0.0f;
        #pragma unroll
        for (int i = 0; i < QBS_THREADS / 32; i++) gmax = fmaxf(gmax, smem_warp_max[i]);
        float s = gmax / QBS_FP8_MAX;
        // An all-zero tile is real (pruned experts exist). Flooring keeps the
        // dequant multiply finite; the encoded bytes are zero either way.
        if (s < 1e-12f) s = 1e-12f;
        block_scale[n_block * k_blocks + k_block] = s;
        smem_scale = s;
    }
    __syncthreads();

    // 2. Encode.
    const float s = smem_scale;
    for (unsigned int i = tid; i < count; i += QBS_THREADS) {
        const unsigned int r = i / k_len;
        const unsigned int c = i - r * k_len;
        const unsigned long long off = (unsigned long long)(n0 + r) * K + (k0 + c);
        float v = __bfloat162float(X[off]) / s;
        v = fmaxf(fminf(v, QBS_FP8_MAX), -QBS_FP8_MAX);
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
        X_fp8[off] = qbs_enc_fp8(v);
#else
        X_fp8[off] = (unsigned char)__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
#endif
    }
}
