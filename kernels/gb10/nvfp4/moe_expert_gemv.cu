// SPDX-License-Identifier: AGPL-3.0-only

// Atlas Batched MoE Expert GEMV — All top-K experts in one kernel launch.
//
// Replaces 10 individual w4a16_gemv + 10 scaled_add with:
//   1 batched GEMV (gate) + 1 batched GEMV (up) + 1 SiLU + 1 batched GEMV (down) + 1 weighted sum
//
// Pointer indirection: kernel reads expert_id from device-side indices array,
// then looks up weight pointers from device-side tables (set up at model load).
// No D2H sync needed — everything stays on device.
//
// Vectorized: reads 4 packed weight bytes (uint32_t = 8 FP4 values) and
// 8 BF16 activations (uint4 = 16 bytes) per iteration for better bandwidth.
//
// NVFP4 weight layout: [N, K/2] packed, [N, K/GROUP_SIZE] scales (K-dim packing).
//
// 4 outputs per block, 32 threads (1 warp) per output. Warp shuffle reduction.
// Grid: (ceil(N / 4), top_k, 1)
// Block: (128, 1, 1)
// blockIdx.y = expert slot (0..top_k-1)

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define BLOCK_SIZE 128
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define GROUP_SIZE 16

__device__ __constant__ float E2M1_LUT_EXP[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// Batched MoE Expert W4A16 GEMV (vectorized)
//
// For top-K experts simultaneously:
//   C[slot, n] = sum_k A[slot_or_shared, k] * dequant(B[expert_id, n, k])
//
// input_stride: 0 = shared input (gate/up: all experts read same A[1,K])
//               K = strided input (down: slot e reads from A + e * K)
extern "C" __global__ void moe_expert_gemv(
    const __nv_bfloat16* __restrict__ A,               // [1, K] or [top_k, K]
    const unsigned long long* __restrict__ packed_ptrs, // [num_experts] B_packed device ptrs
    const unsigned long long* __restrict__ scale_ptrs,  // [num_experts] B_scale device ptrs
    const float* __restrict__ scale2_vals,              // [num_experts] per-expert scale2
    __nv_bfloat16* __restrict__ C,                      // [top_k, N] contiguous output
    const unsigned int* __restrict__ expert_indices,    // [top_k] from GPU top-K
    unsigned int N,
    unsigned int K,
    unsigned int top_k,
    unsigned int input_stride                           // 0 = shared, K = per-expert
) {
    const unsigned int expert_slot = blockIdx.y;
    if (expert_slot >= top_k) return;

    // Look up which expert this slot maps to
    const unsigned int expert_id = expert_indices[expert_slot];

    // Get expert's weight pointers via indirection
    const unsigned char* B_packed = (const unsigned char*)packed_ptrs[expert_id];
    const unsigned char* B_scale = (const unsigned char*)scale_ptrs[expert_id];
    const float scale2 = scale2_vals[expert_id];

    // EP: NULL pointer means remote expert — write zero output and return
    if (B_packed == 0) {
        const unsigned int n_base = blockIdx.x * N_PER_BLOCK;
        for (unsigned int i = threadIdx.x; i < N_PER_BLOCK && n_base + i < N; i += BLOCK_SIZE) {
            C[expert_slot * N + n_base + i] = __float2bfloat16(0.0f);
        }
        return;
    }

    // Input pointer: shared (stride=0) or per-expert (stride=K)
    const __nv_bfloat16* input = A + (input_stride > 0 ? (unsigned long long)expert_slot * input_stride : 0);

    // Standard GEMV logic (vectorized, same structure as w4a16_gemv)
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;  // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    // Load E2M1 LUT into shared memory to avoid __constant__ serialization
    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_EXP[threadIdx.x];
    __syncthreads();

    float acc = 0.0f;

    // Vectorized: process 8 K-values per iteration
    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;

        // Load 8 BF16 activations as uint4 (128-bit)
        uint4 a_data = ((const uint4*)input)[k8];
        const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};

        // Load 4 packed weight bytes as uint32_t (8 FP4 values)
        unsigned int packed4 = *(const unsigned int*)(B_packed + (unsigned long long)n * half_K + k8 * 4);

        // Load single FP8 scale (8 values always in same group: base_k is 8-aligned, GROUP_SIZE=16)
        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + scale_group];
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = scale_byte;
        float scale = (float)fp8 * scale2;

        // Unpack 4 bytes x 2 nibbles = 8 weight values, FMA with activations
        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char byte_val = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = s_lut[byte_val & 0xF] * scale;
            float w_hi = s_lut[byte_val >> 4] * scale;

            __nv_bfloat16 a_lo, a_hi;
            *(unsigned short*)&a_lo = (unsigned short)(a_raw[b] & 0xFFFF);
            *(unsigned short*)&a_hi = (unsigned short)(a_raw[b] >> 16);
            acc += __bfloat162float(a_lo) * w_lo;
            acc += __bfloat162float(a_hi) * w_hi;
        }
    }

    // Warp shuffle reduction (1 warp per output, no cross-warp sync needed)
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (lane == 0) {
        C[(unsigned long long)expert_slot * N + n] = __float2bfloat16(acc);
    }
}

// Weighted sum of batched expert outputs (device-side weights from GPU top-K).
//
// output[j] = sum_{e=0}^{top_k-1} weights[e] * expert_out[e * hidden + j]
//
// Grid: (ceil(hidden/256), 1, 1)  Block: (256, 1, 1)
extern "C" __global__ void moe_weighted_sum(
    __nv_bfloat16* __restrict__ output,              // [1, hidden]
    const __nv_bfloat16* __restrict__ expert_out,    // [top_k, hidden]
    const float* __restrict__ expert_weights,         // [top_k] from GPU top-K (device)
    unsigned int hidden,
    unsigned int top_k
) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= hidden) return;

    float acc = 0.0f;
    for (unsigned int e = 0; e < top_k; e++) {
        float val = __bfloat162float(expert_out[(unsigned long long)e * hidden + j]);
        acc += expert_weights[e] * val;
    }
    output[j] = __float2bfloat16(acc);
}

// Fused weighted sum + sigmoid blend + gate scalar GEMV.
//
// Computes in a single kernel launch:
//   gate_scalar = dot(input[K], gate_weight[K])
//   output[j] = sum_{e} weights[e] * expert_out[e, j]
//             + sigmoid(gate_scalar) * shared_out[j]
//
// Each block independently computes the gate scalar dot product (redundant
// but only 2×K = 8KB of reads per block, 8 blocks total — negligible).
// Eliminates the separate dense_gemv kernel for shared expert gate scalar.
// Saves 1 graph node per MoE layer (48 total).
//
// Grid: (ceil(hidden/256), 1, 1)  Block: (256, 1, 1)
extern "C" __global__ void moe_weighted_sum_blend(
    __nv_bfloat16* __restrict__ output,              // [hidden] output
    const __nv_bfloat16* __restrict__ expert_out,    // [top_k, hidden]
    const float* __restrict__ expert_weights,         // [top_k]
    const __nv_bfloat16* __restrict__ shared_out,    // [hidden]
    const __nv_bfloat16* __restrict__ input,         // [1, K] gate GEMV input (= MoE input)
    const __nv_bfloat16* __restrict__ gate_weight,   // [1, K] shared expert gate weight
    unsigned int hidden,
    unsigned int top_k,
    unsigned int K
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane = tid % WARP_SIZE;

    // ── Phase 1: All threads cooperatively compute gate scalar ──
    // Each block computes the full dot product independently.
    // K=2048, 256 threads → 1 vectorized iteration per thread (8 elements).
    // NULL gate_weight = no gate modulation → sigmoid=1.0 (always include shared expert).
    // Models like Mistral have a shared expert with no gate — it's always-on.
    __shared__ float s_warp_sums[8];
    __shared__ float sigmoid_val;

    if (gate_weight == 0) {
        // No gate: shared expert always included at full weight
        if (tid == 0) sigmoid_val = 1.0f;
        __syncthreads();
    } else {

    float dot_acc = 0.0f;
    unsigned int K8 = K / 8;
    for (unsigned int k8 = tid; k8 < K8; k8 += 256) {
        uint4 a_data = ((const uint4*)input)[k8];
        uint4 w_data = ((const uint4*)gate_weight)[k8];
        const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
        const unsigned int w_raw[4] = {w_data.x, w_data.y, w_data.z, w_data.w};

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            __nv_bfloat16 a_lo, a_hi, w_lo, w_hi;
            *(unsigned short*)&a_lo = (unsigned short)(a_raw[b] & 0xFFFF);
            *(unsigned short*)&a_hi = (unsigned short)(a_raw[b] >> 16);
            *(unsigned short*)&w_lo = (unsigned short)(w_raw[b] & 0xFFFF);
            *(unsigned short*)&w_hi = (unsigned short)(w_raw[b] >> 16);
            dot_acc += __bfloat162float(a_lo) * __bfloat162float(w_lo);
            dot_acc += __bfloat162float(a_hi) * __bfloat162float(w_hi);
        }
    }

    // Warp shuffle reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        dot_acc += __shfl_down_sync(0xFFFFFFFF, dot_acc, offset);
    }
    if (lane == 0) {
        s_warp_sums[warp_id] = dot_acc;
    }
    __syncthreads();

    // Thread 0: final reduce + sigmoid
    if (tid == 0) {
        float gate_scalar = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; w++) {
            gate_scalar += s_warp_sums[w];
        }
        sigmoid_val = 1.0f / (1.0f + __expf(-gate_scalar));
    }
    __syncthreads();

    } // end else (gate_weight != NULL)

    // ── Phase 2: Weighted sum + blend ──
    unsigned int j = blockIdx.x * blockDim.x + tid;
    if (j >= hidden) return;

    float acc = 0.0f;
    for (unsigned int e = 0; e < top_k; e++) {
        acc += expert_weights[e] * __bfloat162float(expert_out[(unsigned long long)e * hidden + j]);
    }
    acc += sigmoid_val * __bfloat162float(shared_out[j]);
    output[j] = __float2bfloat16(acc);
}
