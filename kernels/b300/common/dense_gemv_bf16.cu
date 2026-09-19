// SPDX-License-Identifier: AGPL-3.0-only

// Atlas Dense BF16 GEMV kernel for SM121 (GB10).
//
// out[n] = dot(A[0,:], B[n,:])  where:
//   A: [1, K] BF16 (single activation row, row-major)
//   B: [N, K] BF16 (weights, row-major — standard HuggingFace layout)
//   C: [1, N] BF16 (output, row-major)
//
// Specialized for M=1 decode: replaces dense_gemm_bf16 which wastes
// 15/16 threads (6.25% utilization) at M=1 with 16x16 tiles.
//
// Vectorized: 128-bit (uint4) loads read 8 BF16 per memory transaction,
// improving bandwidth utilization from ~38% to ~70%+ of LPDDR5X peak.
//
// Design: each block computes N_PER_BLOCK output elements.
// 256 threads cooperatively reduce K dimension per output element.
// Uses warp shuffle for final reduction — no shared memory barrier needed.

#include <cuda_bf16.h>

#define BLOCK_SIZE 256
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define VEC_SIZE 8  // BF16 values per vectorized load (uint4 = 16 bytes)

// GEMV: C[n] = sum_k A[k] * B[n, k]  for n in [block_n .. block_n + N_PER_BLOCK)
//
// Grid: (ceil(N / N_PER_BLOCK), 1, 1)   Block: (256, 1, 1)
//
// 4 outputs per block, 64 threads (2 warps) per output. Cross-warp smem
// reduction. Grid: (ceil(N / 4), 1, 1)  Block: (256, 1, 1)
extern "C" __global__ void dense_gemv_bf16(
    const __nv_bfloat16* __restrict__ A,  // [1, K]
    const __nv_bfloat16* __restrict__ B,  // [N, K]
    __nv_bfloat16* __restrict__ C,         // [1, N]
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;  // 32
    const unsigned int local_out = threadIdx.x / threads_per_out;   // which of 8 outputs
    const unsigned int lane = threadIdx.x % threads_per_out;        // position within warp

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    float acc = 0.0f;

    // Vectorized K-reduction: 8 BF16 per load (uint4 = 128 bits)
    const unsigned int K_VEC = K / VEC_SIZE;
    const uint4* A_vec = (const uint4*)A;
    const uint4* B_vec = (const uint4*)(B + (unsigned long long)n * K);

    for (unsigned int kv = lane; kv < K_VEC; kv += threads_per_out) {
        uint4 a_data = A_vec[kv];
        uint4 b_data = B_vec[kv];

        // Extract 8 BF16 pairs from uint4 (4 x uint32, each holding 2 BF16)
        const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
        const unsigned int b_raw[4] = {b_data.x, b_data.y, b_data.z, b_data.w};

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            __nv_bfloat16 a_lo, a_hi, b_lo, b_hi;
            *(unsigned short*)&a_lo = (unsigned short)(a_raw[i] & 0xFFFF);
            *(unsigned short*)&a_hi = (unsigned short)(a_raw[i] >> 16);
            *(unsigned short*)&b_lo = (unsigned short)(b_raw[i] & 0xFFFF);
            *(unsigned short*)&b_hi = (unsigned short)(b_raw[i] >> 16);
            acc += __bfloat162float(a_lo) * __bfloat162float(b_lo);
            acc += __bfloat162float(a_hi) * __bfloat162float(b_hi);
        }
    }

    // Scalar tail for K not divisible by VEC_SIZE (never hits for model dims)
    {
        const unsigned int tail_start = K_VEC * VEC_SIZE;
        const __nv_bfloat16* B_row = B + (unsigned long long)n * K;
        for (unsigned int k = tail_start + lane; k < K; k += threads_per_out) {
            acc += __bfloat162float(A[k]) * __bfloat162float(B_row[k]);
        }
    }

    // Warp shuffle reduction within each group of 64 threads
    // First reduce within each warp (32 threads)
    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    // threads_per_out=64 means 2 warps per output. Use shared memory for cross-warp reduce.
    __shared__ float smem[N_PER_BLOCK * 2];  // 2 warps per output x 4 outputs

    if (warp_lane == 0) {
        // Each warp writes its partial sum
        unsigned int smem_idx = local_out * 2 + (lane / WARP_SIZE);
        smem[smem_idx] = acc;
    }
    __syncthreads();

    // First thread of each output group writes final result
    if (lane == 0) {
        float result = smem[local_out * 2] + smem[local_out * 2 + 1];
        C[n] = __float2bfloat16(result);
    }
}

// FP32-output variant of dense_gemv_bf16.
//
// Same body but the final store keeps the FP32 accumulator without
// downcast to BF16. Used for the LM head step on Gemma-4-31B where the
// 0.125-logit margin between the top two tokens at decode step 1 sits
// exactly on a BF16 representable boundary at value 16-32 (BF16 step
// size = 0.125 there). Truncating the FP32 accumulator at that point
// flips the greedy argmax tiebreak the wrong way and self-reinforces
// into a stop-word loop ("Blue a a a a..." for haiku prompts).
//
// Internal arithmetic is identical; only the output dtype differs.
extern "C" __global__ void dense_gemv_bf16_fp32out(
    const __nv_bfloat16* __restrict__ A,  // [1, K]
    const __nv_bfloat16* __restrict__ B,  // [N, K]
    float* __restrict__ C,                 // [1, N] FP32 (no downcast)
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    float acc = 0.0f;

    const unsigned int K_VEC = K / VEC_SIZE;
    const uint4* A_vec = (const uint4*)A;
    const uint4* B_vec = (const uint4*)(B + (unsigned long long)n * K);

    for (unsigned int kv = lane; kv < K_VEC; kv += threads_per_out) {
        uint4 a_data = A_vec[kv];
        uint4 b_data = B_vec[kv];

        const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
        const unsigned int b_raw[4] = {b_data.x, b_data.y, b_data.z, b_data.w};

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            __nv_bfloat16 a_lo, a_hi, b_lo, b_hi;
            *(unsigned short*)&a_lo = (unsigned short)(a_raw[i] & 0xFFFF);
            *(unsigned short*)&a_hi = (unsigned short)(a_raw[i] >> 16);
            *(unsigned short*)&b_lo = (unsigned short)(b_raw[i] & 0xFFFF);
            *(unsigned short*)&b_hi = (unsigned short)(b_raw[i] >> 16);
            acc += __bfloat162float(a_lo) * __bfloat162float(b_lo);
            acc += __bfloat162float(a_hi) * __bfloat162float(b_hi);
        }
    }

    {
        const unsigned int tail_start = K_VEC * VEC_SIZE;
        const __nv_bfloat16* B_row = B + (unsigned long long)n * K;
        for (unsigned int k = tail_start + lane; k < K; k += threads_per_out) {
            acc += __bfloat162float(A[k]) * __bfloat162float(B_row[k]);
        }
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    __shared__ float smem[N_PER_BLOCK * 2];

    if (warp_lane == 0) {
        unsigned int smem_idx = local_out * 2 + (lane / WARP_SIZE);
        smem[smem_idx] = acc;
    }
    __syncthreads();

    if (lane == 0) {
        float result = smem[local_out * 2] + smem[local_out * 2 + 1];
        C[n] = result;  // keep FP32
    }
}
