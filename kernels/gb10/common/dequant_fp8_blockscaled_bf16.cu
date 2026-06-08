// SPDX-License-Identifier: AGPL-3.0-only

// FP8 block-scaled → BF16 dequantization on GPU.
//
// Replaces the CPU dequant loop in
// `crates/spark-model/src/weight_map/quant_helpers.rs::dequant_fp8_blockscaled_to_bf16`
// which dominates load time for FP8-MoE models when
// ATLAS_FP8_DEQUANT_MOE_TO_BF16=1 (256 experts × 3 projs × 40 layers =
// ~30k dequant calls, each a D2H + single-threaded CPU loop + H2D).
//
// Math (matches the CPU version 1:1):
//   bf16_out[n, k] = E4M3_LUT[fp8_in[n, k]] * scale_inv[n/block_n, k/block_k]
//
// Block scale dtype: BF16 (Qwen3.6 / DeepSeek-V3) or FP32 (MiniMax-M2).
// Selected via `scale_is_fp32` flag.
//
// Grid:  (ceil(K/64), ceil(N/4), 1)
// Block: (64, 4, 1)  — each block processes a 4 × 64 tile.

#include <cuda_bf16.h>

__device__ __constant__ float E4M3_LUT_DEQ[256] = {
    0.0f, 0.001953125f, 0.00390625f, 0.005859375f,
    0.0078125f, 0.009765625f, 0.01171875f, 0.013671875f,
    0.015625f, 0.017578125f, 0.01953125f, 0.021484375f,
    0.0234375f, 0.025390625f, 0.02734375f, 0.029296875f,
    0.03125f, 0.03515625f, 0.0390625f, 0.04296875f,
    0.046875f, 0.05078125f, 0.0546875f, 0.05859375f,
    0.0625f, 0.0703125f, 0.078125f, 0.0859375f,
    0.09375f, 0.1015625f, 0.109375f, 0.1171875f,
    0.125f, 0.140625f, 0.15625f, 0.171875f,
    0.1875f, 0.203125f, 0.21875f, 0.234375f,
    0.25f, 0.28125f, 0.3125f, 0.34375f,
    0.375f, 0.40625f, 0.4375f, 0.46875f,
    0.5f, 0.5625f, 0.625f, 0.6875f,
    0.75f, 0.8125f, 0.875f, 0.9375f,
    1.0f, 1.125f, 1.25f, 1.375f,
    1.5f, 1.625f, 1.75f, 1.875f,
    2.0f, 2.25f, 2.5f, 2.75f,
    3.0f, 3.25f, 3.5f, 3.75f,
    4.0f, 4.5f, 5.0f, 5.5f,
    6.0f, 6.5f, 7.0f, 7.5f,
    8.0f, 9.0f, 10.0f, 11.0f,
    12.0f, 13.0f, 14.0f, 15.0f,
    16.0f, 18.0f, 20.0f, 22.0f,
    24.0f, 26.0f, 28.0f, 30.0f,
    32.0f, 36.0f, 40.0f, 44.0f,
    48.0f, 52.0f, 56.0f, 60.0f,
    64.0f, 72.0f, 80.0f, 88.0f,
    96.0f, 104.0f, 112.0f, 120.0f,
    128.0f, 144.0f, 160.0f, 176.0f,
    192.0f, 208.0f, 224.0f, 240.0f,
    256.0f, 288.0f, 320.0f, 352.0f,
    384.0f, 416.0f, 448.0f, 0.0f,
    -0.0f, -0.001953125f, -0.00390625f, -0.005859375f,
    -0.0078125f, -0.009765625f, -0.01171875f, -0.013671875f,
    -0.015625f, -0.017578125f, -0.01953125f, -0.021484375f,
    -0.0234375f, -0.025390625f, -0.02734375f, -0.029296875f,
    -0.03125f, -0.03515625f, -0.0390625f, -0.04296875f,
    -0.046875f, -0.05078125f, -0.0546875f, -0.05859375f,
    -0.0625f, -0.0703125f, -0.078125f, -0.0859375f,
    -0.09375f, -0.1015625f, -0.109375f, -0.1171875f,
    -0.125f, -0.140625f, -0.15625f, -0.171875f,
    -0.1875f, -0.203125f, -0.21875f, -0.234375f,
    -0.25f, -0.28125f, -0.3125f, -0.34375f,
    -0.375f, -0.40625f, -0.4375f, -0.46875f,
    -0.5f, -0.5625f, -0.625f, -0.6875f,
    -0.75f, -0.8125f, -0.875f, -0.9375f,
    -1.0f, -1.125f, -1.25f, -1.375f,
    -1.5f, -1.625f, -1.75f, -1.875f,
    -2.0f, -2.25f, -2.5f, -2.75f,
    -3.0f, -3.25f, -3.5f, -3.75f,
    -4.0f, -4.5f, -5.0f, -5.5f,
    -6.0f, -6.5f, -7.0f, -7.5f,
    -8.0f, -9.0f, -10.0f, -11.0f,
    -12.0f, -13.0f, -14.0f, -15.0f,
    -16.0f, -18.0f, -20.0f, -22.0f,
    -24.0f, -26.0f, -28.0f, -30.0f,
    -32.0f, -36.0f, -40.0f, -44.0f,
    -48.0f, -52.0f, -56.0f, -60.0f,
    -64.0f, -72.0f, -80.0f, -88.0f,
    -96.0f, -104.0f, -112.0f, -120.0f,
    -128.0f, -144.0f, -160.0f, -176.0f,
    -192.0f, -208.0f, -224.0f, -240.0f,
    -256.0f, -288.0f, -320.0f, -352.0f,
    -384.0f, -416.0f, -448.0f, -0.0f,
};

extern "C" __global__ void dequant_fp8_blockscaled_bf16(
    const unsigned char* __restrict__ fp8_in,   // [N, K] FP8 E4M3 row-major
    const void* __restrict__ scale_inv,         // [sn, sk] BF16 or FP32
    __nv_bfloat16* __restrict__ bf16_out,       // [N, K] BF16 row-major
    unsigned int N,
    unsigned int K,
    unsigned int block_n,
    unsigned int block_k,
    unsigned int sk,
    unsigned int scale_is_fp32
) {
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int n = blockIdx.y * blockDim.y + threadIdx.y;
    if (n >= N || k >= K) return;

    unsigned int sn_idx = n / block_n;
    unsigned int sk_idx = k / block_k;
    unsigned int scale_offset = sn_idx * sk + sk_idx;

    float scale;
    if (scale_is_fp32) {
        scale = ((const float*)scale_inv)[scale_offset];
    } else {
        unsigned short raw = ((const unsigned short*)scale_inv)[scale_offset];
        scale = __bfloat162float(*(const __nv_bfloat16*)&raw);
    }

    unsigned char fp8_byte = fp8_in[n * K + k];
    float val = E4M3_LUT_DEQ[fp8_byte] * scale;
    bf16_out[n * K + k] = __float2bfloat16(val);
}
