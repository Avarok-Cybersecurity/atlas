// SPDX-License-Identifier: AGPL-3.0-only

// Atlas Dense FP8-Weight dual-GEMV (batch=2) kernel for SM121 (GB10).
//
// Computes two output rows from ONE pass over the FP8 weight matrix:
//   C[t, n] = dot(A[t, :], dequant(B_fp8[n, :])) * row_scale[n]   for t in {0,1}
// where:
//   A:         [2, K] BF16 (two activation rows, row-major)
//   B:         [N, K] FP8 E4M3 (quantized weights, row-major)
//   row_scale: [N] FP32 (per-row dequant scale)
//   C:         [2, N] BF16 (output, row-major)
//
// This is the batch=2 sibling of `dense_gemv_fp8w`: each block streams an
// expert/output column's weight row ONCE and applies it to both activation
// rows, halving FP8 weight bandwidth vs two separate M=1 GEMV launches. The
// dot products are accumulated independently per token, so the result is
// bit-identical to running `dense_gemv_fp8w` twice (the per-token reduction
// order is unchanged). Used by the K=2 MTP verify path (lm_head, attention
// Q/K/V/O, SSM out_proj) where the two verify positions share the same
// weights but have distinct activations.
//
// Grid: (ceil(N / N_PER_BLOCK), 1, 1)   Block: (256, 1, 1)

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ float scl_fp8_b2(unsigned char b) {
    unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u; float v;
    if (e == 0u)               v = (float)m * 0.001953125f;
    else if (e == 15u && m == 7u) v = 0.0f;
    else                       v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return s ? -v : v;
}
#endif

#define BLOCK_SIZE 256
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define VEC_SIZE 16

// Decode one uint32 (4 FP8 bytes) into 4 floats.
__device__ __forceinline__ void decode4_fp8(unsigned int w32, float& f0, float& f1,
                                            float& f2, float& f3) {
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
    f0 = scl_fp8_b2((unsigned char)(w32 & 0xFF));
    f1 = scl_fp8_b2((unsigned char)((w32 >> 8) & 0xFF));
    f2 = scl_fp8_b2((unsigned char)((w32 >> 16) & 0xFF));
    f3 = scl_fp8_b2((unsigned char)((w32 >> 24) & 0xFF));
#else
    __nv_fp8_e4m3 a, b, c, d;
    *(unsigned char*)&a = (unsigned char)(w32 & 0xFF);
    *(unsigned char*)&b = (unsigned char)((w32 >> 8) & 0xFF);
    *(unsigned char*)&c = (unsigned char)((w32 >> 16) & 0xFF);
    *(unsigned char*)&d = (unsigned char)((w32 >> 24) & 0xFF);
    f0 = (float)a; f1 = (float)b; f2 = (float)c; f3 = (float)d;
#endif
}

// Accumulate one 4-FP8 chunk against one activation row's 4 BF16 values.
__device__ __forceinline__ void mac4(float& acc, unsigned int a32_lo, unsigned int a32_hi,
                                     float wf0, float wf1, float wf2, float wf3) {
    __nv_bfloat16 a0, a1, a2, a3;
    *(unsigned short*)&a0 = (unsigned short)(a32_lo & 0xFFFF);
    *(unsigned short*)&a1 = (unsigned short)(a32_lo >> 16);
    *(unsigned short*)&a2 = (unsigned short)(a32_hi & 0xFFFF);
    *(unsigned short*)&a3 = (unsigned short)(a32_hi >> 16);
    acc += __bfloat162float(a0) * wf0 + __bfloat162float(a1) * wf1
         + __bfloat162float(a2) * wf2 + __bfloat162float(a3) * wf3;
}

extern "C" __global__ void dense_gemv_fp8w_batch2(
    const __nv_bfloat16* __restrict__ A,    // [2, K] BF16
    const unsigned char* __restrict__ B,     // [N, K] FP8 E4M3
    const float* __restrict__ row_scale,     // [N] f32
    __nv_bfloat16* __restrict__ C,           // [2, N] BF16
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK; // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const float scale = row_scale[n];
    float acc0 = 0.0f, acc1 = 0.0f;

    const unsigned int K_VEC = K / VEC_SIZE;
    const uint4* B_vec = (const uint4*)(B + (unsigned long long)n * K);
    const __nv_bfloat16* A0 = A;                      // token 0 row
    const __nv_bfloat16* A1 = A + (unsigned long long)K; // token 1 row

    for (unsigned int kv = lane; kv < K_VEC; kv += threads_per_out) {
        uint4 b_data = B_vec[kv];
        const unsigned int b_raw[4] = {b_data.x, b_data.y, b_data.z, b_data.w};

        // 16 BF16 activations per token via 2× uint4
        uint4 a0_d0 = ((const uint4*)A0)[kv * 2];
        uint4 a0_d1 = ((const uint4*)A0)[kv * 2 + 1];
        uint4 a1_d0 = ((const uint4*)A1)[kv * 2];
        uint4 a1_d1 = ((const uint4*)A1)[kv * 2 + 1];
        const unsigned int a0_raw0[4] = {a0_d0.x, a0_d0.y, a0_d0.z, a0_d0.w};
        const unsigned int a0_raw1[4] = {a0_d1.x, a0_d1.y, a0_d1.z, a0_d1.w};
        const unsigned int a1_raw0[4] = {a1_d0.x, a1_d0.y, a1_d0.z, a1_d0.w};
        const unsigned int a1_raw1[4] = {a1_d1.x, a1_d1.y, a1_d1.z, a1_d1.w};

        // First 8 weights (b_raw[0], b_raw[1])
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            float w0, w1, w2, w3;
            decode4_fp8(b_raw[i], w0, w1, w2, w3);
            mac4(acc0, a0_raw0[i * 2], a0_raw0[i * 2 + 1], w0, w1, w2, w3);
            mac4(acc1, a1_raw0[i * 2], a1_raw0[i * 2 + 1], w0, w1, w2, w3);
        }
        // Next 8 weights (b_raw[2], b_raw[3])
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            float w0, w1, w2, w3;
            decode4_fp8(b_raw[i + 2], w0, w1, w2, w3);
            mac4(acc0, a0_raw1[i * 2], a0_raw1[i * 2 + 1], w0, w1, w2, w3);
            mac4(acc1, a1_raw1[i * 2], a1_raw1[i * 2 + 1], w0, w1, w2, w3);
        }
    }

    acc0 *= scale;
    acc1 *= scale;

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, offset);
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
    }

    __shared__ float smem0[N_PER_BLOCK * 2];
    __shared__ float smem1[N_PER_BLOCK * 2];
    if (warp_lane == 0) {
        unsigned int smem_idx = local_out * 2 + (lane / WARP_SIZE);
        smem0[smem_idx] = acc0;
        smem1[smem_idx] = acc1;
    }
    __syncthreads();

    if (lane == 0) {
        float r0 = smem0[local_out * 2] + smem0[local_out * 2 + 1];
        float r1 = smem1[local_out * 2] + smem1[local_out * 2 + 1];
        C[n] = __float2bfloat16(r0);                       // token 0 → C[0, n]
        C[(unsigned long long)N + n] = __float2bfloat16(r1); // token 1 → C[1, n]
    }
}
