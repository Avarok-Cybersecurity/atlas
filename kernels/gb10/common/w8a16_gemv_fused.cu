// SPDX-License-Identifier: AGPL-3.0-only

// Atlas W8A16 GEMV Fused — dual projection + silu-input variants (FP8 E4M3).
//
// FP8 mirror of w4a16_gemv_fused.cu (which decodes NVFP4 E2M1). The ONLY
// differences vs the NVFP4 fused kernels are the weight/scale decode:
//   - weights are 1 byte / weight over the FULL K (not K/2 nibble-packed),
//     decoded through a 256-entry E4M3 LUT (not a 16-entry E2M1 LUT),
//   - scales are a 2D FP32 block_scale[N/128, K/128] (one scale per 128 K),
//     NOT per-group(16) FP8-E4M3 group scales × a per-tensor FP32 scale2.
// The input (BF16, uint4-vectorized) handling and the two-stage reduction /
// output are byte-for-byte the same geometry as the NVFP4 kernels and the
// scalar w8a16_gemv.cu. K-iteration order matches w8a16_gemv.cu (16 K / iter)
// so outputs are bit-identical to three separate w8a16_gemv launches.
//
// Reduces the dense shared-FFN decode path from 4 kernels to 2 per layer:
//   Before: gate (1) + up (1) + silu_mul (1) + down (1) = 4 launches/layer
//   After:  gate_up_dual (1) + silu_down (1) = 2 launches/layer
//
// w8a16_gemv_dual: blockIdx.z selects projection 0 (gate) vs 1 (up).
//   Both projections share the same BF16 input A[1, K].
//   Grid: (ceil(N/4), 1, 2)  Block: (256, 1, 1)
//
// w8a16_gemv_silu_input: reads gate_out + up_out BF16 vectors, computes
//   silu(gate)*up inline as activation, then GEMV with FP8 down weights.
//   Eliminates the separate silu_mul kernel entirely.
//   Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
//
// FP8-E4M3 weight format (per projection):
//   B:           [N, K]      uint8 — one byte per weight (FP8 E4M3)
//   block_scale: [N/128, K/128] FP32 — per-128×128-block scale (scale_inv
//                widened to FP32 at load; applied in full FP32 precision,
//                matching vLLM / DeepGEMM / HF block-FP8 numerics)

#include <cuda_bf16.h>

#define BLOCK_SIZE 256
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define FP8_BLOCK 128

// ── E4M3 Lookup Table ──────────────────────────────────────────────
//
// FP8 E4M3: sign(1) + exponent(4) + mantissa(3), bias=7.
// 256 entries mapping every byte value to its f32 equivalent.
// Range: [-448, 448], NaN bytes (0x7F/0xFF) mapped to 0.0.
// Byte-identical to the table in w8a16_gemv.cu (single-source semantics).

__device__ __constant__ float E4M3_LUT_FUSED_W8[256] = {
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

// ── W8A16 GEMV Dual Projection (FP8 E4M3) ──
//
// blockIdx.z = 0: first projection (gate), blockIdx.z = 1: second (up).
// Both read the same shared BF16 input A[1, K] with different FP8 weights.
// Grid: (ceil(N/4), 1, 2)  Block: (256, 1, 1)
extern "C" __global__ void w8a16_gemv_dual(
    const __nv_bfloat16* __restrict__ A,            // [1, K] shared input
    const unsigned char* __restrict__ B1,            // [N, K] proj 0 FP8 E4M3
    const float* __restrict__ B1_scale,              // [N/128, K/128] proj 0 FP32
    __nv_bfloat16* __restrict__ C1,                  // [1, N] proj 0 output
    const unsigned char* __restrict__ B2,            // [N, K] proj 1 FP8 E4M3
    const float* __restrict__ B2_scale,              // [N/128, K/128] proj 1 FP32
    __nv_bfloat16* __restrict__ C2,                  // [1, N] proj 1 output
    unsigned int N,
    unsigned int K
) {
    const unsigned int proj = blockIdx.z;
    const unsigned char* B = proj == 0 ? B1 : B2;
    const float* block_scale = proj == 0 ? B1_scale : B2_scale;
    __nv_bfloat16* C = proj == 0 ? C1 : C2;

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;  // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int K16 = K / 16;
    const unsigned int k_blocks = (K + FP8_BLOCK - 1) / FP8_BLOCK;  // ceil(K/128)
    const unsigned int n_block = n / FP8_BLOCK;

    // 256-entry E4M3 LUT in shared memory (branchless dequant; smem services
    // the data-dependent divergent byte indices in parallel vs __constant__
    // broadcast-cache serialization).
    __shared__ float s_lut[256];
    __shared__ float smem[N_PER_BLOCK * 2];
    s_lut[threadIdx.x] = E4M3_LUT_FUSED_W8[threadIdx.x];
    __syncthreads();

    float acc = 0.0f;

    // 16 K-values per iteration, applying the per-128-block scale.
    for (unsigned int k16 = lane; k16 < K16; k16 += threads_per_out) {
        const unsigned int base_k = k16 * 16;

        const unsigned int k_block = base_k / FP8_BLOCK;
        float scale = block_scale[n_block * k_blocks + k_block];

        // 16 FP8 weights as one uint4.
        uint4 b_data = ((const uint4*)(B + (unsigned long long)n * K))[k16];
        // 16 BF16 activations as 2 × uint4.
        uint4 a_data0 = ((const uint4*)A)[k16 * 2];
        uint4 a_data1 = ((const uint4*)A)[k16 * 2 + 1];

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

    // Two-stage reduction: 32-lane shuffle, then cross-warp via smem.
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

// ── W8A16 GEMV with SiLU-fused Input (FP8 E4M3) ──
//
// Reads gate_out[K] and up_out[K] BF16, computes silu(gate)*up inline as
// the activation, then GEMV with FP8 down weights. Eliminates silu_mul.
// Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1)
extern "C" __global__ void w8a16_gemv_silu_input(
    const __nv_bfloat16* __restrict__ gate_out,     // [1, K] gate proj output
    const __nv_bfloat16* __restrict__ up_out,       // [1, K] up proj output
    const unsigned char* __restrict__ B,             // [N, K] down FP8 E4M3
    const float* __restrict__ block_scale,           // [N/128, K/128] FP32
    __nv_bfloat16* __restrict__ C,                   // [1, N] output
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

    __shared__ float s_lut[256];
    __shared__ float smem[N_PER_BLOCK * 2];
    s_lut[threadIdx.x] = E4M3_LUT_FUSED_W8[threadIdx.x];
    __syncthreads();

    float acc = 0.0f;

    for (unsigned int k16 = lane; k16 < K16; k16 += threads_per_out) {
        const unsigned int base_k = k16 * 16;

        const unsigned int k_block = base_k / FP8_BLOCK;
        float scale = block_scale[n_block * k_blocks + k_block];

        // 16 FP8 down-weights as one uint4.
        uint4 b_data = ((const uint4*)(B + (unsigned long long)n * K))[k16];
        // 16 BF16 gate + 16 BF16 up activations as 2 × uint4 each.
        uint4 g_data0 = ((const uint4*)gate_out)[k16 * 2];
        uint4 g_data1 = ((const uint4*)gate_out)[k16 * 2 + 1];
        uint4 u_data0 = ((const uint4*)up_out)[k16 * 2];
        uint4 u_data1 = ((const uint4*)up_out)[k16 * 2 + 1];

        const unsigned int b_raw0[2] = {b_data.x, b_data.y};
        const unsigned int g_raw0[4] = {g_data0.x, g_data0.y, g_data0.z, g_data0.w};
        const unsigned int u_raw0[4] = {u_data0.x, u_data0.y, u_data0.z, u_data0.w};

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            unsigned int w32 = b_raw0[i];
            unsigned int g32_lo = g_raw0[i * 2];
            unsigned int g32_hi = g_raw0[i * 2 + 1];
            unsigned int u32_lo = u_raw0[i * 2];
            unsigned int u32_hi = u_raw0[i * 2 + 1];

            float w0 = s_lut[(w32      ) & 0xFF] * scale;
            float w1 = s_lut[(w32 >>  8) & 0xFF] * scale;
            float w2 = s_lut[(w32 >> 16) & 0xFF] * scale;
            float w3 = s_lut[(w32 >> 24) & 0xFF] * scale;

            __nv_bfloat16 g0, g1, g2, g3, u0, u1, u2, u3;
            *(unsigned short*)&g0 = (unsigned short)(g32_lo & 0xFFFF);
            *(unsigned short*)&g1 = (unsigned short)(g32_lo >> 16);
            *(unsigned short*)&g2 = (unsigned short)(g32_hi & 0xFFFF);
            *(unsigned short*)&g3 = (unsigned short)(g32_hi >> 16);
            *(unsigned short*)&u0 = (unsigned short)(u32_lo & 0xFFFF);
            *(unsigned short*)&u1 = (unsigned short)(u32_lo >> 16);
            *(unsigned short*)&u2 = (unsigned short)(u32_hi & 0xFFFF);
            *(unsigned short*)&u3 = (unsigned short)(u32_hi >> 16);

            float gf0 = __bfloat162float(g0), gf1 = __bfloat162float(g1);
            float gf2 = __bfloat162float(g2), gf3 = __bfloat162float(g3);

            // SiLU(gate) * up = (gate / (1 + exp(-gate))) * up
            float a0 = (gf0 / (1.0f + __expf(-gf0))) * __bfloat162float(u0);
            float a1 = (gf1 / (1.0f + __expf(-gf1))) * __bfloat162float(u1);
            float a2 = (gf2 / (1.0f + __expf(-gf2))) * __bfloat162float(u2);
            float a3 = (gf3 / (1.0f + __expf(-gf3))) * __bfloat162float(u3);

            acc += a0 * w0;
            acc += a1 * w1;
            acc += a2 * w2;
            acc += a3 * w3;
        }

        const unsigned int b_raw1[2] = {b_data.z, b_data.w};
        const unsigned int g_raw1[4] = {g_data1.x, g_data1.y, g_data1.z, g_data1.w};
        const unsigned int u_raw1[4] = {u_data1.x, u_data1.y, u_data1.z, u_data1.w};

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            unsigned int w32 = b_raw1[i];
            unsigned int g32_lo = g_raw1[i * 2];
            unsigned int g32_hi = g_raw1[i * 2 + 1];
            unsigned int u32_lo = u_raw1[i * 2];
            unsigned int u32_hi = u_raw1[i * 2 + 1];

            float w0 = s_lut[(w32      ) & 0xFF] * scale;
            float w1 = s_lut[(w32 >>  8) & 0xFF] * scale;
            float w2 = s_lut[(w32 >> 16) & 0xFF] * scale;
            float w3 = s_lut[(w32 >> 24) & 0xFF] * scale;

            __nv_bfloat16 g0, g1, g2, g3, u0, u1, u2, u3;
            *(unsigned short*)&g0 = (unsigned short)(g32_lo & 0xFFFF);
            *(unsigned short*)&g1 = (unsigned short)(g32_lo >> 16);
            *(unsigned short*)&g2 = (unsigned short)(g32_hi & 0xFFFF);
            *(unsigned short*)&g3 = (unsigned short)(g32_hi >> 16);
            *(unsigned short*)&u0 = (unsigned short)(u32_lo & 0xFFFF);
            *(unsigned short*)&u1 = (unsigned short)(u32_lo >> 16);
            *(unsigned short*)&u2 = (unsigned short)(u32_hi & 0xFFFF);
            *(unsigned short*)&u3 = (unsigned short)(u32_hi >> 16);

            float gf0 = __bfloat162float(g0), gf1 = __bfloat162float(g1);
            float gf2 = __bfloat162float(g2), gf3 = __bfloat162float(g3);

            float a0 = (gf0 / (1.0f + __expf(-gf0))) * __bfloat162float(u0);
            float a1 = (gf1 / (1.0f + __expf(-gf1))) * __bfloat162float(u1);
            float a2 = (gf2 / (1.0f + __expf(-gf2))) * __bfloat162float(u2);
            float a3 = (gf3 / (1.0f + __expf(-gf3))) * __bfloat162float(u3);

            acc += a0 * w0;
            acc += a1 * w1;
            acc += a2 * w2;
            acc += a3 * w3;
        }
    }

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
