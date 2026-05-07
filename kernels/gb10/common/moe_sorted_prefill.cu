// SPDX-License-Identifier: AGPL-3.0-only

// Atlas Sorted MoE GEMV — L2-optimized prefill.
//
// Tokens are pre-sorted by expert assignment so consecutive CUDA blocks
// process the same expert. This keeps expert weights hot in L2 cache,
// reducing DRAM reads by ~30x for typical configurations.
//
// Routed experts only — shared expert handled separately via w4a16_gemm.
//
// Gate+Up Grid:   (ceil(inter/8), total_expanded, 2)   Block: (128,1,1)
// SiLU+Down Grid: (ceil(hidden/8), total_expanded, 1)  Block: (128,1,1)

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define BLOCK_SIZE 128
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define GROUP_SIZE 16

__device__ __constant__ float E2M1_LUT_SORTED[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// ============================================================================
// Gate+Up projection — sorted routed experts only
// ============================================================================
//
// Grid: (ceil(N_inter/8), total_expanded, 2)  Block: (128, 1, 1)
//
// blockIdx.y indexes into sorted_token_ids / sorted_expert_ids.
// Consecutive y values map to the same expert → L2 reuse for weights.

extern "C" __global__ void moe_sorted_gate_up(
    const __nv_bfloat16* __restrict__ A,            // [num_tokens, K] BF16
    const unsigned long long* __restrict__ gate_packed_ptrs,
    const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals,
    __nv_bfloat16* __restrict__ gate_out,           // [total_expanded, N_inter] sorted order
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ up_out,             // [total_expanded, N_inter] sorted order
    const int* __restrict__ sorted_token_ids,       // [total_expanded] original token index
    const int* __restrict__ sorted_expert_ids,      // [total_expanded] expert ID
    unsigned int N,             // N_inter (intermediate dimension)
    unsigned int K,             // hidden_size (input dimension)
    unsigned int total_expanded
) {
    const unsigned int y = blockIdx.y;
    if (y >= total_expanded) return;

    const unsigned int proj = blockIdx.z;  // 0=gate, 1=up

    // Sorted lookup — consecutive y values share same expert → L2 cache hit
    const int token = sorted_token_ids[y];
    const int expert_id = sorted_expert_ids[y];

    const __nv_bfloat16* A_token = A + (unsigned long long)token * K;

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    __nv_bfloat16* C;

    if (proj == 0) {
        B_packed = (const unsigned char*)gate_packed_ptrs[expert_id];
        B_scale = (const unsigned char*)gate_scale_ptrs[expert_id];
        s2 = gate_scale2_vals[expert_id];
        C = gate_out + (unsigned long long)y * N;
    } else {
        B_packed = (const unsigned char*)up_packed_ptrs[expert_id];
        B_scale = (const unsigned char*)up_scale_ptrs[expert_id];
        s2 = up_scale2_vals[expert_id];
        C = up_out + (unsigned long long)y * N;
    }

    // EP: NULL pointer means remote expert — write zero
    if (B_packed == 0) {
        const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
        for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE) {
            C[n_base + i] = __float2bfloat16(0.0f);
        }
        return;
    }

    // GEMV: compute N_PER_BLOCK*2 output elements per block
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n1 = blockIdx.x * (N_PER_BLOCK * 2) + local_out * 2;
    const unsigned int n2 = n1 + 1;
    if (n1 >= N) return;
    const bool have_n2 = (n2 < N);

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_SORTED[threadIdx.x];
    __syncthreads();

    float acc1 = 0.0f, acc2 = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        uint4 a_data = ((const uint4*)A_token)[k8];
        const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
        const unsigned int base_k = k8 * 8;

        unsigned int packed4_1 = *(const unsigned int*)(B_packed + (unsigned long long)n1 * half_K + k8 * 4);
        unsigned int sg = base_k / GROUP_SIZE;
        unsigned char sb1 = B_scale[(unsigned long long)n1 * num_groups + sg];
        __nv_fp8_e4m3 fp8_1; *(unsigned char*)&fp8_1 = sb1;
        float sc1 = (float)fp8_1 * s2;

        unsigned int packed4_2 = have_n2 ?
            *(const unsigned int*)(B_packed + (unsigned long long)n2 * half_K + k8 * 4) : 0;
        unsigned char sb2 = have_n2 ? B_scale[(unsigned long long)n2 * num_groups + sg] : 0;
        __nv_fp8_e4m3 fp8_2; *(unsigned char*)&fp8_2 = sb2;
        float sc2 = have_n2 ? (float)fp8_2 * s2 : 0.0f;

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char bv1 = (packed4_1 >> (b * 8)) & 0xFF;
            float w1l = s_lut[bv1 & 0xF] * sc1, w1h = s_lut[bv1 >> 4] * sc1;
            unsigned char bv2 = (packed4_2 >> (b * 8)) & 0xFF;
            float w2l = s_lut[bv2 & 0xF] * sc2, w2h = s_lut[bv2 >> 4] * sc2;
            __nv_bfloat16 al, ah;
            *(unsigned short*)&al = (unsigned short)(a_raw[b] & 0xFFFF);
            *(unsigned short*)&ah = (unsigned short)(a_raw[b] >> 16);
            float afl = __bfloat162float(al), afh = __bfloat162float(ah);
            acc1 += afl * w1l + afh * w1h;
            acc2 += afl * w2l + afh * w2h;
        }
    }

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
    if (lane == 0) C[n1] = __float2bfloat16(acc1);

    if (have_n2) {
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
        if (lane == 0) C[n2] = __float2bfloat16(acc2);
    }
}

// ============================================================================
// SiLU+Down projection — sorted routed experts only
// ============================================================================
//
// Grid: (ceil(N_hidden/8), total_expanded, 1)  Block: (128, 1, 1)
//
// Reads gate_out/up_out in sorted order, computes SiLU(gate)*up,
// then GEMV with down projection. Output in sorted order.

extern "C" __global__ void moe_sorted_silu_down(
    const __nv_bfloat16* __restrict__ gate_out,     // [total_expanded, K_inter] sorted
    const __nv_bfloat16* __restrict__ up_out,       // [total_expanded, K_inter] sorted
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,                  // [total_expanded, N_hidden] sorted
    const int* __restrict__ sorted_expert_ids,      // [total_expanded] expert ID
    unsigned int N,             // N_hidden (output dimension)
    unsigned int K,             // K_inter (intermediate, input to down proj)
    unsigned int total_expanded
) {
    const unsigned int y = blockIdx.y;
    if (y >= total_expanded) return;

    const int expert_id = sorted_expert_ids[y];

    const unsigned char* B_packed = (const unsigned char*)packed_ptrs[expert_id];
    const unsigned char* B_scale = (const unsigned char*)scale_ptrs[expert_id];
    float s2 = scale2_vals[expert_id];

    // EP: NULL pointer means remote expert — write zero
    if (B_packed == 0) {
        __nv_bfloat16* out = C + (unsigned long long)y * N;
        const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
        for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE) {
            out[n_base + i] = __float2bfloat16(0.0f);
        }
        return;
    }

    const __nv_bfloat16* g_ptr = gate_out + (unsigned long long)y * K;
    const __nv_bfloat16* u_ptr = up_out + (unsigned long long)y * K;

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n1 = blockIdx.x * (N_PER_BLOCK * 2) + local_out * 2;
    const unsigned int n2 = n1 + 1;
    if (n1 >= N) return;
    const bool have_n2 = (n2 < N);

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    __shared__ float s_lut[16];
    __shared__ float s_act[1024]; // max K=1024

    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_SORTED[threadIdx.x];

    // Phase 1: Cooperatively precompute SiLU(gate)*up into shared memory
    for (unsigned int i = threadIdx.x; i < K; i += BLOCK_SIZE) {
        float gf = __bfloat162float(g_ptr[i]);
        float uf = __bfloat162float(u_ptr[i]);
        s_act[i] = (gf / (1.0f + __expf(-gf))) * uf;
    }
    __syncthreads();

    // Phase 2: GEMV reading precomputed activation from shared memory
    float acc1 = 0.0f, acc2 = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;

        unsigned int packed4_1 = *(const unsigned int*)(B_packed + (unsigned long long)n1 * half_K + k8 * 4);
        unsigned int sg = base_k / GROUP_SIZE;
        unsigned char sb1 = B_scale[(unsigned long long)n1 * num_groups + sg];
        __nv_fp8_e4m3 fp8_1; *(unsigned char*)&fp8_1 = sb1;
        float sc1 = (float)fp8_1 * s2;

        unsigned int packed4_2 = have_n2 ?
            *(const unsigned int*)(B_packed + (unsigned long long)n2 * half_K + k8 * 4) : 0;
        unsigned char sb2 = have_n2 ? B_scale[(unsigned long long)n2 * num_groups + sg] : 0;
        __nv_fp8_e4m3 fp8_2; *(unsigned char*)&fp8_2 = sb2;
        float sc2 = have_n2 ? (float)fp8_2 * s2 : 0.0f;

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            float al = s_act[base_k + b * 2];
            float ah = s_act[base_k + b * 2 + 1];

            unsigned char bv1 = (packed4_1 >> (b * 8)) & 0xFF;
            float w1l = s_lut[bv1 & 0xF] * sc1, w1h = s_lut[bv1 >> 4] * sc1;
            unsigned char bv2 = (packed4_2 >> (b * 8)) & 0xFF;
            float w2l = s_lut[bv2 & 0xF] * sc2, w2h = s_lut[bv2 >> 4] * sc2;

            acc1 += al * w1l + ah * w1h;
            acc2 += al * w2l + ah * w2h;
        }
    }

    __nv_bfloat16* out = C + (unsigned long long)y * N;

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
    if (lane == 0) out[n1] = __float2bfloat16(acc1);

    if (have_n2) {
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
        if (lane == 0) out[n2] = __float2bfloat16(acc2);
    }
}
