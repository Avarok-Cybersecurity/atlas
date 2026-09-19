// SPDX-License-Identifier: AGPL-3.0-only
//
// Per-ROW FP8 (E4M3) quantization — for cuBLASLt OUTER_VEC (row-wise) scaling.
// GB10/sm_121 supports per-tensor and OUTER_VEC (row-wise) fp8 matmul but NOT
// 128-block-scaled fp8, so row-wise is the viable fp8 GEMM path on this silicon.
//
// For each row r of an [R, K] matrix:
//   scale[r]   = max_k |X[r,k]| / 448.0     (floored to 1e-12 to avoid div0)
//   X_fp8[r,k] = round_e4m3( X[r,k] / scale[r] )   (saturating to ±448)
//
// Used for BOTH the per-output-channel WEIGHT re-quant ([N,K] → scale[N]) and
// the per-token ACTIVATION quant ([M,K] → scale[M]). cuBLASLt OUTER_VEC_32F
// then folds A_scale[i] * B_scale[j] into the FP32 accumulator epilogue.
//
// Grid: (R, 1, 1)  Block: (256, 1, 1) — 8 warps stride over K.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define RW_FP8_MAX 448.0f
#define RW_THREADS 256

#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
// gfx1151/SCALE software SATFINITE E4M3 encode (matches the repo's other
// quantizers; __nv_cvt_float_to_fp8 is non-standard there).
__device__ __forceinline__ unsigned char rw_enc_fp8(float v) {
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

extern "C" __global__ void quant_rowwise_fp8(
    const __nv_bfloat16* __restrict__ X,   // [R, K] BF16
    unsigned char* __restrict__ X_fp8,     // [R, K] FP8 E4M3
    float* __restrict__ scale,             // [R] FP32 per-row scale
    unsigned int R,
    unsigned int K
) {
    const unsigned int r = blockIdx.x;
    if (r >= R) return;
    const unsigned int tid = threadIdx.x;
    const unsigned long long base = (unsigned long long)r * K;

    // 1. Row max over K (strided over 256 threads).
    float my_max = 0.0f;
    for (unsigned int k = tid; k < K; k += RW_THREADS) {
        my_max = fmaxf(my_max, fabsf(__bfloat162float(X[base + k])));
    }

    // 2. Block-reduce max (warp shuffle + smem across 8 warps).
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        my_max = fmaxf(my_max, __shfl_down_sync(0xFFFFFFFF, my_max, off));
    }
    __shared__ float smem_warp_max[RW_THREADS / 32];
    const unsigned int warp_id = tid >> 5;
    const unsigned int lane = tid & 31;
    if (lane == 0) smem_warp_max[warp_id] = my_max;
    __syncthreads();

    __shared__ float smem_scale;
    if (tid == 0) {
        float gmax = 0.0f;
        #pragma unroll
        for (int i = 0; i < RW_THREADS / 32; i++) gmax = fmaxf(gmax, smem_warp_max[i]);
        float s = gmax / RW_FP8_MAX;
        if (s < 1e-12f) s = 1e-12f;
        scale[r] = s;
        smem_scale = s;
    }
    __syncthreads();

    // 3. Quantize each element to FP8 E4M3.
    const float inv = smem_scale;
    for (unsigned int k = tid; k < K; k += RW_THREADS) {
        float v = __bfloat162float(X[base + k]) / inv;
        v = fmaxf(fminf(v, RW_FP8_MAX), -RW_FP8_MAX);
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
        X_fp8[base + k] = rw_enc_fp8(v);
#else
        X_fp8[base + k] = (unsigned char)__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
#endif
    }
}
