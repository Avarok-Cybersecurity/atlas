// SPDX-License-Identifier: AGPL-3.0-only

// Atlas Fused MoE Expert GEMV — Gate+Up in one launch, SiLU+Down in one launch.
//
// Reduces MoE expert kernels from 4 to 2 per layer (saves 96 launches total):
//   Before: gate (1) + up (1) + silu_mul (1) + down (1) = 4 per layer × 48 = 192
//   After:  gate_up (1) + silu_down (1) = 2 per layer × 48 = 96
//
// moe_expert_gemv_gate_up: blockIdx.z selects gate (0) vs up (1) projection.
//   Both projections share the same input and expert_indices.
//   Grid: (ceil(N/4), top_k, 2)
//
// moe_expert_gemv_silu_down: reads gate_out + up_out, computes silu(gate)*up
//   inline as the activation, then GEMV with down weights.
//   Eliminates separate silu_mul kernel entirely.
//   Grid: (ceil(N/4), top_k, 1)

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define BLOCK_SIZE 128
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define GROUP_SIZE 16

__device__ __constant__ float E2M1_LUT_FUSED[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// ── Fused Gate+Up Expert GEMV ──
//
// blockIdx.z = 0: gate projection, blockIdx.z = 1: up projection
// Both read same shared input A[1, K] and same expert_indices.
// Grid: (ceil(N / 4), top_k, 2)   Block: (128, 1, 1)
extern "C" __global__ void moe_expert_gemv_gate_up(
    const __nv_bfloat16* __restrict__ A,                    // [1, K] shared input
    const unsigned long long* __restrict__ gate_packed_ptrs, // [num_experts]
    const unsigned long long* __restrict__ gate_scale_ptrs,  // [num_experts]
    const float* __restrict__ gate_scale2_vals,              // [num_experts]
    __nv_bfloat16* __restrict__ gate_out,                    // [top_k, N]
    const unsigned long long* __restrict__ up_packed_ptrs,   // [num_experts]
    const unsigned long long* __restrict__ up_scale_ptrs,    // [num_experts]
    const float* __restrict__ up_scale2_vals,                // [num_experts]
    __nv_bfloat16* __restrict__ up_out,                      // [top_k, N]
    const unsigned int* __restrict__ expert_indices,          // [top_k]
    unsigned int N,
    unsigned int K,
    unsigned int top_k
) {
    const unsigned int expert_slot = blockIdx.y;
    if (expert_slot >= top_k) return;

    const unsigned int proj = blockIdx.z;  // 0=gate, 1=up

    const unsigned int expert_id = expert_indices[expert_slot];

    // Select weight pointers and output based on projection
    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float scale2;
    __nv_bfloat16* C;

    if (proj == 0) {
        B_packed = (const unsigned char*)gate_packed_ptrs[expert_id];
        B_scale = (const unsigned char*)gate_scale_ptrs[expert_id];
        scale2 = gate_scale2_vals[expert_id];
        C = gate_out;
    } else {
        B_packed = (const unsigned char*)up_packed_ptrs[expert_id];
        B_scale = (const unsigned char*)up_scale_ptrs[expert_id];
        scale2 = up_scale2_vals[expert_id];
        C = up_out;
    }

    // EP: NULL pointer means remote expert — write zero output and return
    if (B_packed == 0) {
        const unsigned int n_base = blockIdx.x * N_PER_BLOCK;
        for (unsigned int i = threadIdx.x; i < N_PER_BLOCK && n_base + i < N; i += BLOCK_SIZE) {
            C[expert_slot * N + n_base + i] = __float2bfloat16(0.0f);
        }
        return;
    }

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_FUSED[threadIdx.x];
    __syncthreads();

    float acc = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;

        uint4 a_data = ((const uint4*)A)[k8];
        const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};

        unsigned int packed4 = *(const unsigned int*)(B_packed + (unsigned long long)n * half_K + k8 * 4);

        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + scale_group];
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = scale_byte;
        float scale = (float)fp8 * scale2;

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

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (lane == 0) {
        C[(unsigned long long)expert_slot * N + n] = __float2bfloat16(acc);
    }
}

// ── Register-Tiled Fused Gate+Up Expert GEMV ──
//
// Same as moe_expert_gemv_gate_up but each thread computes 2 output rows,
// reusing the shared input vector from registers. Doubles outstanding
// weight reads per iteration for better LPDDR5X bandwidth utilization.
//
// 4 groups × 32 threads, each group handles 2 rows → 8 outputs per block.
// Grid: (ceil(N / 8), top_k, 2)   Block: (128, 1, 1)
extern "C" __global__ void moe_expert_gemv_gate_up_2x(
    const __nv_bfloat16* __restrict__ A,                    // [1, K] shared input
    const unsigned long long* __restrict__ gate_packed_ptrs, // [num_experts]
    const unsigned long long* __restrict__ gate_scale_ptrs,  // [num_experts]
    const float* __restrict__ gate_scale2_vals,              // [num_experts]
    __nv_bfloat16* __restrict__ gate_out,                    // [top_k, N]
    const unsigned long long* __restrict__ up_packed_ptrs,   // [num_experts]
    const unsigned long long* __restrict__ up_scale_ptrs,    // [num_experts]
    const float* __restrict__ up_scale2_vals,                // [num_experts]
    __nv_bfloat16* __restrict__ up_out,                      // [top_k, N]
    const unsigned int* __restrict__ expert_indices,          // [top_k]
    unsigned int N,
    unsigned int K,
    unsigned int top_k
) {
    const unsigned int expert_slot = blockIdx.y;
    if (expert_slot >= top_k) return;

    const unsigned int proj = blockIdx.z;  // 0=gate, 1=up
    const unsigned int expert_id = expert_indices[expert_slot];

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    __nv_bfloat16* C;

    if (proj == 0) {
        B_packed = (const unsigned char*)gate_packed_ptrs[expert_id];
        B_scale = (const unsigned char*)gate_scale_ptrs[expert_id];
        s2 = gate_scale2_vals[expert_id];
        C = gate_out;
    } else {
        B_packed = (const unsigned char*)up_packed_ptrs[expert_id];
        B_scale = (const unsigned char*)up_scale_ptrs[expert_id];
        s2 = up_scale2_vals[expert_id];
        C = up_out;
    }

    // EP: NULL pointer means remote expert — write zero output and return
    if (B_packed == 0) {
        const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
        for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE) {
            C[expert_slot * N + n_base + i] = __float2bfloat16(0.0f);
        }
        return;
    }

    // 4 groups of 32 threads, each group handles 2 consecutive output rows
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;  // 32
    const unsigned int local_out = threadIdx.x / threads_per_out;    // 0..3
    const unsigned int lane = threadIdx.x % threads_per_out;         // 0..31

    const unsigned int n1 = blockIdx.x * (N_PER_BLOCK * 2) + local_out * 2;
    const unsigned int n2 = n1 + 1;
    if (n1 >= N) return;
    const bool have_n2 = (n2 < N);

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_FUSED[threadIdx.x];
    __syncthreads();

    float acc1 = 0.0f;
    float acc2 = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;

        // Load input (shared between both output rows)
        uint4 a_data = ((const uint4*)A)[k8];
        const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};

        // Load weights for row n1
        unsigned int packed4_1 = *(const unsigned int*)(B_packed + (unsigned long long)n1 * half_K + k8 * 4);
        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char sb1 = B_scale[(unsigned long long)n1 * num_groups + scale_group];
        __nv_fp8_e4m3 fp8_1;
        *(unsigned char*)&fp8_1 = sb1;
        float scale_1 = (float)fp8_1 * s2;

        // Load weights for row n2
        unsigned int packed4_2 = have_n2 ?
            *(const unsigned int*)(B_packed + (unsigned long long)n2 * half_K + k8 * 4) : 0;
        unsigned char sb2 = have_n2 ?
            B_scale[(unsigned long long)n2 * num_groups + scale_group] : 0;
        __nv_fp8_e4m3 fp8_2;
        *(unsigned char*)&fp8_2 = sb2;
        float scale_2 = have_n2 ? (float)fp8_2 * s2 : 0.0f;

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            // Dequant weights for row n1
            unsigned char bv1 = (packed4_1 >> (b * 8)) & 0xFF;
            float w1_lo = s_lut[bv1 & 0xF] * scale_1;
            float w1_hi = s_lut[bv1 >> 4] * scale_1;

            // Dequant weights for row n2
            unsigned char bv2 = (packed4_2 >> (b * 8)) & 0xFF;
            float w2_lo = s_lut[bv2 & 0xF] * scale_2;
            float w2_hi = s_lut[bv2 >> 4] * scale_2;

            // Shared input (reused from registers)
            __nv_bfloat16 a_lo, a_hi;
            *(unsigned short*)&a_lo = (unsigned short)(a_raw[b] & 0xFFFF);
            *(unsigned short*)&a_hi = (unsigned short)(a_raw[b] >> 16);
            float af_lo = __bfloat162float(a_lo);
            float af_hi = __bfloat162float(a_hi);

            acc1 += af_lo * w1_lo + af_hi * w1_hi;
            acc2 += af_lo * w2_lo + af_hi * w2_hi;
        }
    }

    // Warp shuffle reduction for acc1
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
    }
    if (lane == 0) {
        C[(unsigned long long)expert_slot * N + n1] = __float2bfloat16(acc1);
    }

    // Warp shuffle reduction for acc2
    if (have_n2) {
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
        }
        if (lane == 0) {
            C[(unsigned long long)expert_slot * N + n2] = __float2bfloat16(acc2);
        }
    }
}

// ── Fused SiLU+Down Expert GEMV ──
//
// Reads gate_out[slot, k] and up_out[slot, k], computes
// activation = silu(gate) * up inline, then GEMV with down weights.
// Eliminates the separate silu_mul kernel entirely.
//
// Grid: (ceil(N / 4), top_k, 1)   Block: (128, 1, 1)
extern "C" __global__ void moe_expert_gemv_silu_down(
    const __nv_bfloat16* __restrict__ gate_out,              // [top_k, K] gate outputs
    const __nv_bfloat16* __restrict__ up_out,                // [top_k, K] up outputs
    const unsigned long long* __restrict__ packed_ptrs,      // [num_experts] down B_packed
    const unsigned long long* __restrict__ scale_ptrs,       // [num_experts] down B_scale
    const float* __restrict__ scale2_vals,                   // [num_experts] down scale2
    __nv_bfloat16* __restrict__ C,                           // [top_k, N] output
    const unsigned int* __restrict__ expert_indices,          // [top_k]
    unsigned int N,
    unsigned int K,
    unsigned int top_k
) {
    const unsigned int expert_slot = blockIdx.y;
    if (expert_slot >= top_k) return;

    const unsigned int expert_id = expert_indices[expert_slot];

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

    // Per-expert strided input (silu(gate) * up computed inline)
    const __nv_bfloat16* g_ptr = gate_out + (unsigned long long)expert_slot * K;
    const __nv_bfloat16* u_ptr = up_out + (unsigned long long)expert_slot * K;

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_FUSED[threadIdx.x];
    __syncthreads();

    float acc = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;

        // Load 8 gate values and 8 up values, compute silu(gate) * up
        uint4 g_data = ((const uint4*)g_ptr)[k8];
        uint4 u_data = ((const uint4*)u_ptr)[k8];

        unsigned int packed4 = *(const unsigned int*)(B_packed + (unsigned long long)n * half_K + k8 * 4);

        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + scale_group];
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = scale_byte;
        float scale = (float)fp8 * scale2;

        // Process 8 elements: silu(gate) * up * dequant(weight)
        const unsigned int g_raw[4] = {g_data.x, g_data.y, g_data.z, g_data.w};
        const unsigned int u_raw[4] = {u_data.x, u_data.y, u_data.z, u_data.w};

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char byte_val = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = s_lut[byte_val & 0xF] * scale;
            float w_hi = s_lut[byte_val >> 4] * scale;

            // Gate values
            __nv_bfloat16 g_lo, g_hi;
            *(unsigned short*)&g_lo = (unsigned short)(g_raw[b] & 0xFFFF);
            *(unsigned short*)&g_hi = (unsigned short)(g_raw[b] >> 16);
            float gf_lo = __bfloat162float(g_lo);
            float gf_hi = __bfloat162float(g_hi);

            // Up values
            __nv_bfloat16 u_lo, u_hi;
            *(unsigned short*)&u_lo = (unsigned short)(u_raw[b] & 0xFFFF);
            *(unsigned short*)&u_hi = (unsigned short)(u_raw[b] >> 16);

            // SiLU(gate) * up = (gate / (1 + exp(-gate))) * up
            float a_lo = (gf_lo / (1.0f + __expf(-gf_lo))) * __bfloat162float(u_lo);
            float a_hi = (gf_hi / (1.0f + __expf(-gf_hi))) * __bfloat162float(u_hi);

            acc += a_lo * w_lo;
            acc += a_hi * w_hi;
        }
    }

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (lane == 0) {
        C[(unsigned long long)expert_slot * N + n] = __float2bfloat16(acc);
    }
}

// ── Register-Tiled Fused SiLU+Down Expert GEMV ──
//
// Same as moe_expert_gemv_silu_down but each thread computes 2 output rows,
// reusing the SiLU(gate)*up activation from registers. Doubles outstanding
// weight reads per iteration for better LPDDR5X bandwidth utilization.
// Critical for K=512 where base kernel only does 2 iterations per thread.
//
// 4 groups × 32 threads, each group handles 2 rows → 8 outputs per block.
// Grid: (ceil(N / 8), top_k, 1)   Block: (128, 1, 1)
extern "C" __global__ void moe_expert_gemv_silu_down_2x(
    const __nv_bfloat16* __restrict__ gate_out,              // [top_k, K] gate outputs
    const __nv_bfloat16* __restrict__ up_out,                // [top_k, K] up outputs
    const unsigned long long* __restrict__ packed_ptrs,      // [num_experts] down B_packed
    const unsigned long long* __restrict__ scale_ptrs,       // [num_experts] down B_scale
    const float* __restrict__ scale2_vals,                   // [num_experts] down scale2
    __nv_bfloat16* __restrict__ C,                           // [top_k, N] output
    const unsigned int* __restrict__ expert_indices,          // [top_k]
    unsigned int N,
    unsigned int K,
    unsigned int top_k
) {
    const unsigned int expert_slot = blockIdx.y;
    if (expert_slot >= top_k) return;

    const unsigned int expert_id = expert_indices[expert_slot];

    const unsigned char* B_packed = (const unsigned char*)packed_ptrs[expert_id];
    const unsigned char* B_scale = (const unsigned char*)scale_ptrs[expert_id];
    const float scale2 = scale2_vals[expert_id];

    // EP: NULL pointer means remote expert — write zero output and return
    if (B_packed == 0) {
        const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
        for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE) {
            C[expert_slot * N + n_base + i] = __float2bfloat16(0.0f);
        }
        return;
    }

    const __nv_bfloat16* g_ptr = gate_out + (unsigned long long)expert_slot * K;
    const __nv_bfloat16* u_ptr = up_out + (unsigned long long)expert_slot * K;

    // 4 groups of 32 threads, each group handles 2 consecutive output rows
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;  // 32
    const unsigned int local_out = threadIdx.x / threads_per_out;    // 0..3
    const unsigned int lane = threadIdx.x % threads_per_out;         // 0..31

    const unsigned int n1 = blockIdx.x * (N_PER_BLOCK * 2) + local_out * 2;
    const unsigned int n2 = n1 + 1;
    if (n1 >= N) return;
    const bool have_n2 = (n2 < N);

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_FUSED[threadIdx.x];
    __syncthreads();

    float acc1 = 0.0f;
    float acc2 = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;

        // Load gate+up and compute SiLU activation (shared between both output rows)
        uint4 g_data = ((const uint4*)g_ptr)[k8];
        uint4 u_data = ((const uint4*)u_ptr)[k8];

        // Load weights for row n1
        unsigned int packed4_1 = *(const unsigned int*)(B_packed + (unsigned long long)n1 * half_K + k8 * 4);
        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char sb1 = B_scale[(unsigned long long)n1 * num_groups + scale_group];
        __nv_fp8_e4m3 fp8_1;
        *(unsigned char*)&fp8_1 = sb1;
        float scale_1 = (float)fp8_1 * scale2;

        // Load weights for row n2
        unsigned int packed4_2 = have_n2 ?
            *(const unsigned int*)(B_packed + (unsigned long long)n2 * half_K + k8 * 4) : 0;
        unsigned char sb2 = have_n2 ?
            B_scale[(unsigned long long)n2 * num_groups + scale_group] : 0;
        __nv_fp8_e4m3 fp8_2;
        *(unsigned char*)&fp8_2 = sb2;
        float scale_2 = have_n2 ? (float)fp8_2 * scale2 : 0.0f;

        const unsigned int g_raw[4] = {g_data.x, g_data.y, g_data.z, g_data.w};
        const unsigned int u_raw[4] = {u_data.x, u_data.y, u_data.z, u_data.w};

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            // Dequant weights for both rows
            unsigned char bv1 = (packed4_1 >> (b * 8)) & 0xFF;
            float w1_lo = s_lut[bv1 & 0xF] * scale_1;
            float w1_hi = s_lut[bv1 >> 4] * scale_1;

            unsigned char bv2 = (packed4_2 >> (b * 8)) & 0xFF;
            float w2_lo = s_lut[bv2 & 0xF] * scale_2;
            float w2_hi = s_lut[bv2 >> 4] * scale_2;

            // SiLU(gate) * up activation (shared between both rows)
            __nv_bfloat16 g_lo, g_hi;
            *(unsigned short*)&g_lo = (unsigned short)(g_raw[b] & 0xFFFF);
            *(unsigned short*)&g_hi = (unsigned short)(g_raw[b] >> 16);
            float gf_lo = __bfloat162float(g_lo);
            float gf_hi = __bfloat162float(g_hi);

            __nv_bfloat16 u_lo, u_hi;
            *(unsigned short*)&u_lo = (unsigned short)(u_raw[b] & 0xFFFF);
            *(unsigned short*)&u_hi = (unsigned short)(u_raw[b] >> 16);

            float a_lo = (gf_lo / (1.0f + __expf(-gf_lo))) * __bfloat162float(u_lo);
            float a_hi = (gf_hi / (1.0f + __expf(-gf_hi))) * __bfloat162float(u_hi);

            acc1 += a_lo * w1_lo + a_hi * w1_hi;
            acc2 += a_lo * w2_lo + a_hi * w2_hi;
        }
    }

    // Warp shuffle reduction for acc1
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
    }
    if (lane == 0) {
        C[(unsigned long long)expert_slot * N + n1] = __float2bfloat16(acc1);
    }

    // Warp shuffle reduction for acc2
    if (have_n2) {
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
        }
        if (lane == 0) {
            C[(unsigned long long)expert_slot * N + n2] = __float2bfloat16(acc2);
        }
    }
}

// ── Wide SiLU+Down Expert GEMV (optimized for small K) ──
//
// Same as moe_expert_gemv_silu_down but with 16 outputs per block instead of 4.
// For K=512: 8 inner iterations per thread (vs 2), dramatically better
// memory latency hiding. Uses sub-warp (width=8) shuffle reduction.
//
// Grid: (ceil(N / 16), top_k, 1)   Block: (128, 1, 1)
extern "C" __global__ void moe_expert_gemv_silu_down_wide(
    const __nv_bfloat16* __restrict__ gate_out,              // [top_k, K]
    const __nv_bfloat16* __restrict__ up_out,                // [top_k, K]
    const unsigned long long* __restrict__ packed_ptrs,      // [num_experts]
    const unsigned long long* __restrict__ scale_ptrs,       // [num_experts]
    const float* __restrict__ scale2_vals,                   // [num_experts]
    __nv_bfloat16* __restrict__ C,                           // [top_k, N]
    const unsigned int* __restrict__ expert_indices,          // [top_k]
    unsigned int N,
    unsigned int K,
    unsigned int top_k
) {
    const unsigned int expert_slot = blockIdx.y;
    if (expert_slot >= top_k) return;

    const unsigned int expert_id = expert_indices[expert_slot];

    const unsigned char* B_packed = (const unsigned char*)packed_ptrs[expert_id];
    const unsigned char* B_scale = (const unsigned char*)scale_ptrs[expert_id];
    const float scale2 = scale2_vals[expert_id];

    // EP: NULL pointer means remote expert — write zero output and return
    if (B_packed == 0) {
        const unsigned int N_WIDE = 16;
        const unsigned int n_base = blockIdx.x * N_WIDE;
        for (unsigned int i = threadIdx.x; i < N_WIDE && n_base + i < N; i += BLOCK_SIZE) {
            C[expert_slot * N + n_base + i] = __float2bfloat16(0.0f);
        }
        return;
    }

    const __nv_bfloat16* g_ptr = gate_out + (unsigned long long)expert_slot * K;
    const __nv_bfloat16* u_ptr = up_out + (unsigned long long)expert_slot * K;

    // 16 outputs per block, 8 threads per output
    const unsigned int N_WIDE = 16;
    const unsigned int tpo = BLOCK_SIZE / N_WIDE;  // 8
    const unsigned int local_out = threadIdx.x / tpo;
    const unsigned int lane = threadIdx.x % tpo;

    const unsigned int n = blockIdx.x * N_WIDE + local_out;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_FUSED[threadIdx.x];
    __syncthreads();

    float acc = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += tpo) {
        const unsigned int base_k = k8 * 8;

        uint4 g_data = ((const uint4*)g_ptr)[k8];
        uint4 u_data = ((const uint4*)u_ptr)[k8];

        unsigned int packed4 = *(const unsigned int*)(B_packed + (unsigned long long)n * half_K + k8 * 4);

        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + scale_group];
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = scale_byte;
        float scale = (float)fp8 * scale2;

        const unsigned int g_raw[4] = {g_data.x, g_data.y, g_data.z, g_data.w};
        const unsigned int u_raw[4] = {u_data.x, u_data.y, u_data.z, u_data.w};

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char byte_val = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = s_lut[byte_val & 0xF] * scale;
            float w_hi = s_lut[byte_val >> 4] * scale;

            __nv_bfloat16 g_lo, g_hi;
            *(unsigned short*)&g_lo = (unsigned short)(g_raw[b] & 0xFFFF);
            *(unsigned short*)&g_hi = (unsigned short)(g_raw[b] >> 16);
            float gf_lo = __bfloat162float(g_lo);
            float gf_hi = __bfloat162float(g_hi);

            __nv_bfloat16 u_lo, u_hi;
            *(unsigned short*)&u_lo = (unsigned short)(u_raw[b] & 0xFFFF);
            *(unsigned short*)&u_hi = (unsigned short)(u_raw[b] >> 16);

            float a_lo = (gf_lo / (1.0f + __expf(-gf_lo))) * __bfloat162float(u_lo);
            float a_hi = (gf_hi / (1.0f + __expf(-gf_hi))) * __bfloat162float(u_hi);

            acc += a_lo * w_lo;
            acc += a_hi * w_hi;
        }
    }

    // Sub-warp reduction (width=8): 3 shuffle steps
    #pragma unroll
    for (int offset = tpo / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset, tpo);
    }

    if (lane == 0) {
        C[(unsigned long long)expert_slot * N + n] = __float2bfloat16(acc);
    }
}
