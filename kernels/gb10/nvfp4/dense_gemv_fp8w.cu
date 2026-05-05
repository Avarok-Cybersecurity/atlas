// SPDX-License-Identifier: AGPL-3.0-only

// Atlas Dense FP8-Weight GEMV kernel for SM121 (GB10).
//
// out[n] = dot(A[0,:], dequant(B_fp8[n,:])) * row_scale[n]  where:
//   A:         [1, K] BF16 (single activation row, row-major)
//   B:         [N, K] FP8 E4M3 (quantized weights, row-major)
//   row_scale: [N] FP32 (per-row dequant scale)
//   C:         [1, N] BF16 (output, row-major)
//
// BF16 weights quantized to FP8 at model load time with per-row scaling.
// Halves weight bandwidth vs dense_gemv_bf16: 1 byte/weight instead of 2.
//
// Vectorized: uint4 loads read 16 FP8 values (16 bytes) per weight load,
// and 2 x uint4 reads 16 BF16 activations (32 bytes) per iteration.
// Total: 48 bytes per 16 K-elements (vs 64 bytes for BF16 GEMV).
//
// Design: same as dense_gemv_bf16 — N_PER_BLOCK=4 outputs per block,
// 64 threads (2 warps) per output, cross-warp shared memory reduction.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define BLOCK_SIZE 256
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define VEC_SIZE 16  // FP8 values per vectorized load (uint4 = 16 bytes)

// ── Runtime BF16 → FP8 E4M3 quantization ──────────────────────────
//
// Quantizes a [N, K] BF16 weight matrix to [N, K] FP8 with per-row f32 scales.
// Called once at model load time (not on the decode hot path).
//
// Grid: (N, 1, 1)  Block: (256, 1, 1)
// Each block quantizes one row of K elements.
extern "C" __global__ void quantize_bf16_to_fp8(
    const __nv_bfloat16* __restrict__ input,  // [N, K] BF16
    unsigned char* __restrict__ output,        // [N, K] FP8 E4M3 bytes
    float* __restrict__ row_scales,            // [N] per-row f32 scales
    unsigned int N,
    unsigned int K
) {
    unsigned int row = blockIdx.x;
    if (row >= N) return;

    const __nv_bfloat16* row_in = input + (unsigned long long)row * K;
    unsigned char* row_out = output + (unsigned long long)row * K;

    // Step 1: Find max absolute value in this row (parallel reduction)
    float local_max = 0.0f;
    for (unsigned int k = threadIdx.x; k < K; k += blockDim.x) {
        float absval = fabsf(__bfloat162float(row_in[k]));
        if (absval > local_max) local_max = absval;
    }

    // Warp-level max reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xFFFFFFFF, local_max, offset);
        if (other > local_max) local_max = other;
    }

    // Cross-warp reduction via shared memory
    __shared__ float smem_q[8];  // up to 8 warps (256/32)
    unsigned int warp_id = threadIdx.x / WARP_SIZE;
    unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    if (warp_lane == 0) smem_q[warp_id] = local_max;
    __syncthreads();

    if (threadIdx.x == 0) {
        float max_val = 0.0f;
        for (int w = 0; w < (int)(blockDim.x / WARP_SIZE); w++) {
            if (smem_q[w] > max_val) max_val = smem_q[w];
        }
        // Scale: normalize weights to FP8 E4M3 range [-448, 448]
        float scale = (max_val > 0.0f) ? (max_val / 448.0f) : 1.0f;
        row_scales[row] = scale;
        smem_q[0] = 1.0f / scale;  // broadcast inv_scale
    }
    __syncthreads();
    float inv_scale = smem_q[0];

    // Step 2: Quantize each element to FP8 E4M3
    for (unsigned int k = threadIdx.x; k < K; k += blockDim.x) {
        float val = __bfloat162float(row_in[k]) * inv_scale;
        // Clamp to FP8 E4M3 range (saturation)
        val = fminf(fmaxf(val, -448.0f), 448.0f);
        __nv_fp8_e4m3 fp8 = (__nv_fp8_e4m3)val;
        row_out[k] = *(unsigned char*)&fp8;
    }
}

// ── FP8-Weight GEMV ───────────────────────────────────────────────

// GEMV: C[n] = sum_k A[k] * (B_fp8[n,k] * row_scale[n])
//
// Grid: (ceil(N / N_PER_BLOCK), 1, 1)   Block: (256, 1, 1)
extern "C" __global__ void dense_gemv_fp8w(
    const __nv_bfloat16* __restrict__ A,          // [1, K]
    const unsigned char* __restrict__ B,           // [N, K] FP8 E4M3
    const float* __restrict__ row_scale,           // [N] per-row f32
    __nv_bfloat16* __restrict__ C,                 // [1, N]
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;  // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    // Load per-row scale once
    const float scale = row_scale[n];

    float acc = 0.0f;

    // Vectorized K-reduction: 16 FP8 weights per uint4 load
    const unsigned int K_VEC = K / VEC_SIZE;
    const uint4* B_vec = (const uint4*)(B + (unsigned long long)n * K);

    for (unsigned int kv = lane; kv < K_VEC; kv += threads_per_out) {
        // Load 16 FP8 weights via uint4 (16 bytes)
        uint4 b_data = B_vec[kv];

        // Load 16 BF16 activations via 2 x uint4 (32 bytes)
        // Activation offset: kv * VEC_SIZE elements = kv * 16 BF16 = kv * 32 bytes
        // As uint4 (8 BF16 each): indices kv*2 and kv*2+1
        uint4 a_data0 = ((const uint4*)A)[kv * 2];
        uint4 a_data1 = ((const uint4*)A)[kv * 2 + 1];

        // Extract weight bytes from uint4 (4 x uint32, each holding 4 FP8 values)
        const unsigned int b_raw[4] = {b_data.x, b_data.y, b_data.z, b_data.w};

        // Process first 8 FP8 weights with first 8 BF16 activations
        const unsigned int a_raw0[4] = {a_data0.x, a_data0.y, a_data0.z, a_data0.w};

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            // Each uint32 holds 4 FP8 values (bytes 0,1,2,3)
            unsigned int w32 = b_raw[i];
            unsigned int a32_lo = a_raw0[i * 2];
            unsigned int a32_hi = a_raw0[i * 2 + 1];

            // Byte 0,1 → pair with a32_lo (2 BF16)
            __nv_fp8_e4m3 fp8_0, fp8_1, fp8_2, fp8_3;
            *(unsigned char*)&fp8_0 = (unsigned char)(w32 & 0xFF);
            *(unsigned char*)&fp8_1 = (unsigned char)((w32 >> 8) & 0xFF);
            *(unsigned char*)&fp8_2 = (unsigned char)((w32 >> 16) & 0xFF);
            *(unsigned char*)&fp8_3 = (unsigned char)((w32 >> 24) & 0xFF);

            __nv_bfloat16 a_lo, a_hi;
            *(unsigned short*)&a_lo = (unsigned short)(a32_lo & 0xFFFF);
            *(unsigned short*)&a_hi = (unsigned short)(a32_lo >> 16);
            acc += __bfloat162float(a_lo) * (float)fp8_0;
            acc += __bfloat162float(a_hi) * (float)fp8_1;

            *(unsigned short*)&a_lo = (unsigned short)(a32_hi & 0xFFFF);
            *(unsigned short*)&a_hi = (unsigned short)(a32_hi >> 16);
            acc += __bfloat162float(a_lo) * (float)fp8_2;
            acc += __bfloat162float(a_hi) * (float)fp8_3;
        }

        // Process next 8 FP8 weights with next 8 BF16 activations
        const unsigned int a_raw1[4] = {a_data1.x, a_data1.y, a_data1.z, a_data1.w};

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            unsigned int w32 = b_raw[i + 2];
            unsigned int a32_lo = a_raw1[i * 2];
            unsigned int a32_hi = a_raw1[i * 2 + 1];

            __nv_fp8_e4m3 fp8_0, fp8_1, fp8_2, fp8_3;
            *(unsigned char*)&fp8_0 = (unsigned char)(w32 & 0xFF);
            *(unsigned char*)&fp8_1 = (unsigned char)((w32 >> 8) & 0xFF);
            *(unsigned char*)&fp8_2 = (unsigned char)((w32 >> 16) & 0xFF);
            *(unsigned char*)&fp8_3 = (unsigned char)((w32 >> 24) & 0xFF);

            __nv_bfloat16 a_lo, a_hi;
            *(unsigned short*)&a_lo = (unsigned short)(a32_lo & 0xFFFF);
            *(unsigned short*)&a_hi = (unsigned short)(a32_lo >> 16);
            acc += __bfloat162float(a_lo) * (float)fp8_0;
            acc += __bfloat162float(a_hi) * (float)fp8_1;

            *(unsigned short*)&a_lo = (unsigned short)(a32_hi & 0xFFFF);
            *(unsigned short*)&a_hi = (unsigned short)(a32_hi >> 16);
            acc += __bfloat162float(a_lo) * (float)fp8_2;
            acc += __bfloat162float(a_hi) * (float)fp8_3;
        }
    }

    // Apply per-row scale after full K accumulation (better precision)
    acc *= scale;

    // Warp shuffle reduction within each group of 64 threads
    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    // Cross-warp shared memory reduction (2 warps per output)
    __shared__ float smem[N_PER_BLOCK * 2];

    if (warp_lane == 0) {
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
