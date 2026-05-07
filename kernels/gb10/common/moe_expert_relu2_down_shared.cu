// SPDX-License-Identifier: AGPL-3.0-only

// Atlas Fused MoE Expert relu²+down GEMV — for 2-projection experts (Nemotron-H).
//
// Nemotron-H MoE experts have only up_proj + down_proj (no gate_proj).
// Activation: relu(x)^2 (ReLU-squared) instead of SiLU(gate)*up.
//
// Same structure as moe_expert_silu_down_shared.cu but with relu² activation.
// blockIdx.y < top_k: routed expert
// blockIdx.y == top_k: shared expert (direct weight pointers, different K)
//
// Grid: (ceil(N/8), top_k+1, 1)  Block: (128, 1, 1)

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define BLOCK_SIZE 128
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define GROUP_SIZE 16

__device__ __constant__ float E2M1_LUT_R2[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// Fused relu²+down 2x GEMV with shared expert.
//
// Reads up_out, applies relu², loads result into shared memory,
// then performs down-proj GEMV using the activated values as input.
//
// Routed experts: up_out at [slot*K_routed, K_routed], weights from pointer table
// Shared expert:  up_out at sh_up_in [K_shared], weights from direct pointers
//
// K_routed and K_shared may differ (e.g., 1856 vs 3712 for Nemotron-H).
extern "C" __global__ void moe_expert_relu2_down_shared(
    const __nv_bfloat16* __restrict__ up_out,       // [top_k, K_routed] BF16
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,                   // [top_k, N] routed output
    const unsigned int* __restrict__ expert_indices,
    // Shared expert
    const __nv_bfloat16* __restrict__ sh_up_in,     // [K_shared] BF16
    const unsigned char* __restrict__ sh_down_packed,
    const unsigned char* __restrict__ sh_down_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,         // [N_shared] BF16
    unsigned int N,             // output dim (hidden_size for routed)
    unsigned int K_routed,      // input dim for routed experts (moe_intermediate_size)
    unsigned int K_shared,      // input dim for shared expert (shared_expert_intermediate_size)
    unsigned int N_shared,      // output dim for shared expert (hidden_size)
    unsigned int top_k
) {
    const unsigned int expert_slot = blockIdx.y;
    const bool is_shared = (expert_slot == top_k);

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    unsigned int K;
    unsigned int N_out;

    const __nv_bfloat16* u_ptr;

    if (is_shared) {
        B_packed = sh_down_packed;
        B_scale = sh_down_scale;
        s2 = sh_down_s2;
        u_ptr = sh_up_in;
        K = K_shared;
        N_out = N_shared;
    } else {
        const unsigned int expert_id = expert_indices[expert_slot];
        B_packed = (const unsigned char*)packed_ptrs[expert_id];
        B_scale = (const unsigned char*)scale_ptrs[expert_id];
        s2 = scale2_vals[expert_id];
        u_ptr = up_out + (unsigned long long)expert_slot * K_routed;
        K = K_routed;
        N_out = N;
        // EP: NULL pointer means remote expert
        if (B_packed == 0) {
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N_out; i += BLOCK_SIZE) {
                C[expert_slot * N + n_base + i] = __float2bfloat16(0.0f);
            }
            return;
        }
    }

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n1 = blockIdx.x * (N_PER_BLOCK * 2) + local_out * 2;
    const unsigned int n2 = n1 + 1;
    if (n1 >= N_out) return;
    const bool have_n2 = (n2 < N_out);

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    // Shared memory: E2M1 LUT + precomputed relu²(up) activation
    __shared__ float s_lut[16];
    // max K = 3712 (shared expert), fits in shared memory (14.8 KB)
    extern __shared__ float s_act[];

    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_R2[threadIdx.x];

    // Phase 1: Cooperatively precompute relu²(up) into shared memory
    for (unsigned int i = threadIdx.x; i < K; i += BLOCK_SIZE) {
        float u = __bfloat162float(u_ptr[i]);
        float r = fmaxf(u, 0.0f);
        s_act[i] = r * r;
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

    // Output: shared writes to sh_down_out, routed writes to C[slot*N]
    __nv_bfloat16* out = is_shared ? sh_down_out : (C + (unsigned long long)expert_slot * N);

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
