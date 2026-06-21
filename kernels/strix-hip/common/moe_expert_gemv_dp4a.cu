// SPDX-License-Identifier: AGPL-3.0-only
//
// W4A8 integer-DP4A MoE expert GEMV for AMD gfx1151 (RDNA3.5 / Strix Halo).
//
// DP4A analog of moe_expert_gemv.cu (per-expert pointer indirection, NVFP4 4-bit
// weights). ADDITIVE: the float moe_expert_gemv / FP8 expert path are untouched and
// remain the gb10/NVIDIA defaults. Selected only on strix-hip behind a flag, and only
// once the FP8 experts have been requantized to NVFP4 at load.
//
// Decode profile (ATLAS_PROFILE, 35B-A3B-FP8) put the routed-expert GEMVs at ~24% of
// per-token decode, run in FP8 (1 byte/weight). Routing them to NVFP4 (0.5 byte/weight)
// halves the dominant weight traffic on the bandwidth-bound LPDDR5X part, and DP4A
// replaces the FP32 FMA epilogue with hardware `v_dot4` (`__builtin_amdgcn_sudot4`).
//
// FAITHFULNESS: integer codebook = E2M1 * 2 = {0,1,2,3,4,6,8,12} (+neg), exact; x0.5
// folded into the per-group scale. The only new error vs the float NVFP4 path is the
// int8 activation quant (block-q8_1 style, shared across gate/up via quantize once).
//
// Activations are pre-quantized to int8 by quantize_act_int8_g16 (see w4a16_gemv_dp4a.cu)
// ONCE per layer and shared across the gate/up (and reused for down) GEMVs.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ float moe_dp4a_scl_fp8(unsigned char b) {
    unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u; float v;
    if (e == 0u)                  v = (float)m * 0.001953125f;
    else if (e == 15u && m == 7u) v = 0.0f;
    else                          v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return s ? -v : v;
}

__device__ __constant__ signed char MOE_DP4A_CODEBOOK[16] = {
    0, 1, 2, 3, 4, 6, 8, 12,
    0, -1, -2, -3, -4, -6, -8, -12
};

#define MOE_DP4A_BLOCK_SIZE 256
#define MOE_DP4A_N_PER_BLOCK 4
#define MOE_DP4A_WARP_SIZE 32
#define MOE_DP4A_GROUP_SIZE 16

#if defined(__HIP_PLATFORM_AMD__) || defined(__SCALE__)
#define MOE_DP4A_DOT(a, b, c) __builtin_amdgcn_sudot4(true, (a), true, (b), (c), false)
#else
#define MOE_DP4A_DOT(a, b, c) __dp4a((a), (b), (c))
#endif

// out[expert_slot, n] = a_scale-weighted sum_k aq[k] * wint(expert_id, n, k)
// Grid: (ceil(N/N_PER_BLOCK), top_k, 1)  Block: (256,1,1)  — same geometry as moe_expert_gemv.
extern "C" __global__ void moe_expert_gemv_dp4a(
    const signed char* __restrict__ a_q,                // [1,K] or [top_k,K] int8 acts
    const float* __restrict__ a_scale,                  // [K/16] or [top_k, K/16] act scales
    const unsigned long long* __restrict__ packed_ptrs, // [num_experts] B_packed device ptrs
    const unsigned long long* __restrict__ scale_ptrs,  // [num_experts] B_scale device ptrs
    const float* __restrict__ scale2_vals,              // [num_experts] per-expert scale2
    __nv_bfloat16* __restrict__ C,                      // [top_k, N]
    const unsigned int* __restrict__ expert_indices,    // [top_k]
    unsigned int N,
    unsigned int K,
    unsigned int top_k,
    unsigned int input_stride                           // 0 = shared act row, K = per-slot act row
) {
    const unsigned int expert_slot = blockIdx.y;
    if (expert_slot >= top_k) return;
    const unsigned int expert_id = expert_indices[expert_slot];
    const unsigned char* B_packed = (const unsigned char*)packed_ptrs[expert_id];
    const unsigned char* B_scale = (const unsigned char*)scale_ptrs[expert_id];
    const float scale2 = scale2_vals[expert_id];

    const unsigned int threads_per_out = MOE_DP4A_BLOCK_SIZE / MOE_DP4A_N_PER_BLOCK; // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int n = blockIdx.x * MOE_DP4A_N_PER_BLOCK + local_out;

    if (B_packed == 0) { // EP remote expert -> zero
        if (n < N && lane == 0) C[(unsigned long long)expert_slot * N + n] = __float2bfloat16(0.0f);
        return;
    }
    if (n >= N) return;

    // Shared (stride=0) or per-slot (stride=K) activation + matching per-group scales.
    const unsigned int num_groups = K / MOE_DP4A_GROUP_SIZE;
    const signed char* aq = a_q + (input_stride > 0 ? (unsigned long long)expert_slot * K : 0);
    const float* asc = a_scale + (input_stride > 0 ? (unsigned long long)expert_slot * num_groups : 0);

    const unsigned int half_K = K / 2;
    const unsigned int K16 = K / 16;

    __shared__ signed char s_cb[16];
    __shared__ float smem[MOE_DP4A_N_PER_BLOCK * 2];
    if (threadIdx.x < 16) s_cb[threadIdx.x] = MOE_DP4A_CODEBOOK[threadIdx.x];
    __syncthreads();

    float acc = 0.0f;
    for (unsigned int k16 = lane; k16 < K16; k16 += threads_per_out) {
        const unsigned int base_k = k16 * 16;
        int4 aq4 = *(const int4*)(aq + base_k);
        const int av[4] = {aq4.x, aq4.y, aq4.z, aq4.w};
        unsigned long long packed8 = *(const unsigned long long*)(B_packed + (unsigned long long)n * half_K + k16 * 8);

        int wint[4];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            unsigned int packed = 0;
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                int elem = j * 4 + e;
                int b = elem >> 1;
                unsigned char byte_val = (unsigned char)(packed8 >> (b * 8));
                unsigned char nib = (elem & 1) ? (byte_val >> 4) : (byte_val & 0xF);
                packed |= ((unsigned int)(unsigned char)s_cb[nib]) << (e * 8);
            }
            wint[j] = (int)packed;
        }
        int sumi = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++) sumi = MOE_DP4A_DOT(av[j], wint[j], sumi);

        unsigned int scale_group = base_k / MOE_DP4A_GROUP_SIZE;
        float wscale = moe_dp4a_scl_fp8(B_scale[(unsigned long long)n * num_groups + scale_group]);
        acc += (float)sumi * asc[k16] * (wscale * 0.5f * scale2);
    }

    const unsigned int warp_lane = threadIdx.x % MOE_DP4A_WARP_SIZE;
    #pragma unroll
    for (int offset = MOE_DP4A_WARP_SIZE / 2; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    if (warp_lane == 0) smem[local_out * 2 + (lane / MOE_DP4A_WARP_SIZE)] = acc;
    __syncthreads();
    if (lane == 0) C[(unsigned long long)expert_slot * N + n] = __float2bfloat16(smem[local_out * 2] + smem[local_out * 2 + 1]);
}
