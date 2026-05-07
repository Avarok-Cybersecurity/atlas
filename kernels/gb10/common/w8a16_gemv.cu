// SPDX-License-Identifier: AGPL-3.0-only

// Atlas W8A16 GEMV — Fused FP8-E4M3 weight dequant + BF16 GEMV for M=1 decode.
//
// out[n] = sum_k A[0,k] * E4M3_LUT[B[n,k]] * block_scale[n/BS, k/BS]
//
// Uses a 256-entry E4M3 LUT in shared memory for branchless dequant.
// Supports 2D block-scaled FP8 (block_size=128 in both N and K dimensions),
// matching the Qwen/Qwen3.5-35B-A3B-FP8 checkpoint format.
//
// FP8-E4M3 weight format:
//   B: [N, K] uint8 — one byte per weight (FP8 E4M3)
//   block_scale: [N/BS, K/BS] BF16 — per-block scale (scale_inv from checkpoint)
//   BS: block size (128)
//
// 4 outputs per block, 64 threads per output, vectorized 16-byte reads.
// Grid: (ceil(N / 4), 1, 1)   Block: (256, 1, 1)

#include <cuda_bf16.h>

#define BLOCK_SIZE 256
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define FP8_BLOCK 128

// ── E4M3 Lookup Table ──────────────────────────────────────────────
//
// FP8 E4M3: sign(1) + exponent(4) + mantissa(3), bias=7
// 256 entries mapping every possible byte value to its f32 equivalent.
// Range: [-448, 448], NaN (0x7F/0xFF) mapped to 0.0.

__device__ __constant__ float E4M3_LUT[256] = {
    // Positive (0x00..0x7F)
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
    // Negative (0x80..0xFF)
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

// ── W8A16 GEMV with 2D block scales ────────────────────────────────
//
// C[n] = sum_k A[k] * E4M3_LUT[B[n,k]] * block_scale[n/BS, k/BS]
//
// block_scale is [ceil(N/BS), ceil(K/BS)] in BF16, stored row-major.
// Each 128×128 block of weights shares one scale factor.

extern "C" __global__ void w8a16_gemv(
    const __nv_bfloat16* __restrict__ A,            // [1, K]
    const unsigned char* __restrict__ B,             // [N, K] FP8 E4M3
    const __nv_bfloat16* __restrict__ block_scale,   // [N/BS, K/BS] BF16
    __nv_bfloat16* __restrict__ C,                   // [1, N]
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;  // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int K16 = K / 16;
    const unsigned int k_blocks = (K + FP8_BLOCK - 1) / FP8_BLOCK;  // ceil(K/128)
    const unsigned int n_block = n / FP8_BLOCK;

    // Load E4M3 LUT into shared memory (256 entries, 1 KB)
    __shared__ float s_lut[256];
    __shared__ float smem[N_PER_BLOCK * 2];
    s_lut[threadIdx.x] = E4M3_LUT[threadIdx.x];
    __syncthreads();

    float acc = 0.0f;

    // Process 16 K-values per iteration, applying per-block scale
    for (unsigned int k16 = lane; k16 < K16; k16 += threads_per_out) {
        const unsigned int base_k = k16 * 16;

        // Determine which K-block this chunk falls in and load its scale
        const unsigned int k_block = base_k / FP8_BLOCK;
        float scale = __bfloat162float(block_scale[n_block * k_blocks + k_block]);

        // Load 16 FP8 weights as uint4
        uint4 b_data = ((const uint4*)(B + (unsigned long long)n * K))[k16];

        // Load 16 BF16 activations as 2 × uint4
        uint4 a_data0 = ((const uint4*)A)[k16 * 2];
        uint4 a_data1 = ((const uint4*)A)[k16 * 2 + 1];

        // First 8: b_data.x, b_data.y with a_data0
        const unsigned int b_raw0[2] = {b_data.x, b_data.y};
        const unsigned int a_raw0[4] = {a_data0.x, a_data0.y, a_data0.z, a_data0.w};

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            unsigned int w32 = b_raw0[i];
            unsigned int a32_lo = a_raw0[i * 2];
            unsigned int a32_hi = a_raw0[i * 2 + 1];

            float w0 = s_lut[(w32      ) & 0xFF] * scale;
            float w1 = s_lut[(w32 >>  8) & 0xFF] * scale;
            float w2 = s_lut[(w32 >> 16) & 0xFF] * scale;
            float w3 = s_lut[(w32 >> 24) & 0xFF] * scale;

            __nv_bfloat16 a0, a1, a2, a3;
            *(unsigned short*)&a0 = (unsigned short)(a32_lo & 0xFFFF);
            *(unsigned short*)&a1 = (unsigned short)(a32_lo >> 16);
            *(unsigned short*)&a2 = (unsigned short)(a32_hi & 0xFFFF);
            *(unsigned short*)&a3 = (unsigned short)(a32_hi >> 16);

            acc += __bfloat162float(a0) * w0;
            acc += __bfloat162float(a1) * w1;
            acc += __bfloat162float(a2) * w2;
            acc += __bfloat162float(a3) * w3;
        }

        // Next 8: b_data.z, b_data.w with a_data1
        const unsigned int b_raw1[2] = {b_data.z, b_data.w};
        const unsigned int a_raw1[4] = {a_data1.x, a_data1.y, a_data1.z, a_data1.w};

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            unsigned int w32 = b_raw1[i];
            unsigned int a32_lo = a_raw1[i * 2];
            unsigned int a32_hi = a_raw1[i * 2 + 1];

            float w0 = s_lut[(w32      ) & 0xFF] * scale;
            float w1 = s_lut[(w32 >>  8) & 0xFF] * scale;
            float w2 = s_lut[(w32 >> 16) & 0xFF] * scale;
            float w3 = s_lut[(w32 >> 24) & 0xFF] * scale;

            __nv_bfloat16 a0, a1, a2, a3;
            *(unsigned short*)&a0 = (unsigned short)(a32_lo & 0xFFFF);
            *(unsigned short*)&a1 = (unsigned short)(a32_lo >> 16);
            *(unsigned short*)&a2 = (unsigned short)(a32_hi & 0xFFFF);
            *(unsigned short*)&a3 = (unsigned short)(a32_hi >> 16);

            acc += __bfloat162float(a0) * w0;
            acc += __bfloat162float(a1) * w1;
            acc += __bfloat162float(a2) * w2;
            acc += __bfloat162float(a3) * w3;
        }
    }

    // Cross-warp reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    unsigned int warp_in_out = lane / WARP_SIZE;
    if (lane % WARP_SIZE == 0) {
        smem[local_out * 2 + warp_in_out] = acc;
    }
    __syncthreads();

    if (lane == 0) {
        float result = smem[local_out * 2] + smem[local_out * 2 + 1];
        C[n] = __float2bfloat16(result);
    }
}
