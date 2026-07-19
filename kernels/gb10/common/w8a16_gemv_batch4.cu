// SPDX-License-Identifier: AGPL-3.0-only

// Atlas W8A16 batched GEMV (M<=4) — block-scaled FP8 weight, BF16 activations.
//
//   C[t, n] = sum_k A[t, k] * E4M3_LUT[B[n, k]] * block_scale[n/128, k/128]
//             for t in 0..M  (M <= 4)
//
// The M=4 sibling of `w8a16_gemv` (M=1). At n=4 batched decode the SSM QKVZ /
// out_proj projections share weights across the 4 sequences, so a single pass
// over the FP8 weight matrix serves all M rows: the weight byte is dequantized
// (LUT * block_scale) ONCE and MAC'd into M independent FP32 accumulators. This
// replaces `w8a16_gemm_pipelined` for n<=4, which pads M to a 128-row MMA tile
// (32x compute over-provision, issue/occupancy-bound). One DRAM pass over the
// same weight bytes, no tensor cores, no M padding.
//
// Per-row accumulation order is IDENTICAL to `w8a16_gemv`, so the output is
// bit-identical to running `w8a16_gemv` M times (verify with a cos>=0.9999
// microtest). A:[M,K] BF16, B:[N,K] FP8 E4M3, block_scale:[N/128,K/128] FP32,
// C:[M,N] BF16. Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1).

#include <cuda_bf16.h>

#define BLOCK_SIZE 256
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define FP8_BLOCK 128

// FP8 E4M3 -> f32 lookup table (256 entries). Identical table to w8a16_gemv.cu.
__device__ __constant__ float E4M3_LUT_B4[256] = {
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

// Templated on the MAX rows (register array bound + unroll). The runtime `M`
// (<= MAX_M) selects how many activation rows are actually computed, so one
// instantiation serves all n in [1, MAX_M]. MAX_M=4 is the optimal common path
// (n<=4); MAX_M=16 covers high-concurrency decode (n=5..16) without falling
// back to the M-padded MMA.
template <int MAX_M>
__device__ __forceinline__ void w8a16_gemv_batchm_impl(
    const __nv_bfloat16* __restrict__ A,    // [M, K] BF16
    const unsigned char* __restrict__ B,     // [N, K] FP8 E4M3
    const float* __restrict__ block_scale,   // [N/128, K/128] FP32
    __nv_bfloat16* __restrict__ C,           // [M, N] BF16
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;  // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;

    __shared__ float s_lut[256];
    s_lut[threadIdx.x] = E4M3_LUT_B4[threadIdx.x];
    __syncthreads();

    if (n >= N) return;

    const unsigned int K16 = K / 16;
    const unsigned int k_blocks = (K + FP8_BLOCK - 1) / FP8_BLOCK;
    const unsigned int n_block = n / FP8_BLOCK;

    float acc[MAX_M];
    #pragma unroll
    for (int t = 0; t < MAX_M; t++) acc[t] = 0.0f;

    for (unsigned int k16 = lane; k16 < K16; k16 += threads_per_out) {
        const unsigned int base_k = k16 * 16;
        const unsigned int k_block = base_k / FP8_BLOCK;
        const float scale = block_scale[n_block * k_blocks + k_block];

        // 16 FP8 weight bytes, dequantized + scaled ONCE for all M rows.
        uint4 b_data = ((const uint4*)(B + (unsigned long long)n * K))[k16];
        const unsigned int b_raw[4] = {b_data.x, b_data.y, b_data.z, b_data.w};
        float wf[16];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            unsigned int w32 = b_raw[i];
            wf[i * 4 + 0] = s_lut[(w32      ) & 0xFF] * scale;
            wf[i * 4 + 1] = s_lut[(w32 >>  8) & 0xFF] * scale;
            wf[i * 4 + 2] = s_lut[(w32 >> 16) & 0xFF] * scale;
            wf[i * 4 + 3] = s_lut[(w32 >> 24) & 0xFF] * scale;
        }

        // Reuse the scaled weights across each activation row (k-order matches
        // w8a16_gemv exactly -> bit-identical per-row reduction).
        #pragma unroll
        for (int t = 0; t < MAX_M; t++) {
            if ((unsigned int)t >= M) continue;
            const __nv_bfloat16* At = A + (unsigned long long)t * K;
            uint4 a_lo = ((const uint4*)At)[k16 * 2];
            uint4 a_hi = ((const uint4*)At)[k16 * 2 + 1];
            const unsigned int ar[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                                        a_hi.x, a_hi.y, a_hi.z, a_hi.w};
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                __nv_bfloat16 lo, hi;
                *(unsigned short*)&lo = (unsigned short)(ar[j] & 0xFFFF);
                *(unsigned short*)&hi = (unsigned short)(ar[j] >> 16);
                acc[t] += __bfloat162float(lo) * wf[j * 2]
                        + __bfloat162float(hi) * wf[j * 2 + 1];
            }
        }
    }

    // Cross-warp reduction (threads_per_out=64 -> 2 warps/output), per row.
    __shared__ float smem[MAX_M][N_PER_BLOCK * 2];
    const unsigned int warp_in_out = lane / WARP_SIZE;
    #pragma unroll
    for (int t = 0; t < MAX_M; t++) {
        if ((unsigned int)t >= M) continue;
        float a = acc[t];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            a += __shfl_down_sync(0xFFFFFFFF, a, offset);
        }
        if (lane % WARP_SIZE == 0) smem[t][local_out * 2 + warp_in_out] = a;
    }
    __syncthreads();

    if (lane == 0) {
        #pragma unroll
        for (int t = 0; t < MAX_M; t++) {
            if ((unsigned int)t >= M) continue;
            float r = smem[t][local_out * 2] + smem[t][local_out * 2 + 1];
            C[(unsigned long long)t * N + n] = __float2bfloat16(r);
        }
    }
}

// M<=4 (common-path batched decode). Optimal: no wasted unroll past 4 rows.
extern "C" __global__ void w8a16_gemv_batch4(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    const float* __restrict__ block_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    w8a16_gemv_batchm_impl<4>(A, B, block_scale, C, M, N, K);
}

// M<=16 (high-concurrency decode, n=5..16). Same weight-streaming pass; one
// extra weight read serves up to 16 activation rows instead of the M-padded
// MMA the pipelined kernel would use at these batch sizes.
extern "C" __global__ void w8a16_gemv_batch16(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    const float* __restrict__ block_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    w8a16_gemv_batchm_impl<16>(A, B, block_scale, C, M, N, K);
}
