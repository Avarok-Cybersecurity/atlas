// SPDX-License-Identifier: AGPL-3.0-only
//
// Transposed-layout decode MoE — FP8 (single-token) variant. Companion
// to moe_shared_expert_fused_t.cu but with native FP8 expert weights
// (1 byte per element, no NVFP4 nibble packing) + BF16 block scales.
//
// Weight layout `[K, N]` FP8 (transposed from `[N, K]`).
// Scale layout `[K/FP8_BLOCK, N/FP8_BLOCK]` BF16
// (transposed from `[N/FP8_BLOCK, K/FP8_BLOCK]`).

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define BLOCK_SIZE 32
#define FP8_BLOCK 128

extern "C" __global__ void moe_expert_gate_up_shared_fp8_t(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ gate_weight_t_ptrs,
    const unsigned long long* __restrict__ gate_block_scale_t_ptrs,
    __nv_bfloat16* __restrict__ gate_out,
    const unsigned long long* __restrict__ up_weight_t_ptrs,
    const unsigned long long* __restrict__ up_block_scale_t_ptrs,
    __nv_bfloat16* __restrict__ up_out,
    const unsigned int* __restrict__ expert_indices,
    const unsigned char* __restrict__ sh_gate_t_weight,
    const __nv_bfloat16* __restrict__ sh_gate_t_block_scale,
    __nv_bfloat16* __restrict__ sh_gate_out,
    const unsigned char* __restrict__ sh_up_t_weight,
    const __nv_bfloat16* __restrict__ sh_up_t_block_scale,
    __nv_bfloat16* __restrict__ sh_up_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    const unsigned int expert_slot = blockIdx.y;
    const unsigned int proj = blockIdx.z;
    const bool is_shared = (expert_slot == top_k);

    const unsigned char* B_weight;
    const __nv_bfloat16* B_block_scale;
    __nv_bfloat16* C;

    if (is_shared) {
        if (proj == 0) {
            B_weight = sh_gate_t_weight; B_block_scale = sh_gate_t_block_scale; C = sh_gate_out;
        } else {
            B_weight = sh_up_t_weight; B_block_scale = sh_up_t_block_scale; C = sh_up_out;
        }
    } else {
        const unsigned int expert_id = expert_indices[expert_slot];
        if (proj == 0) {
            B_weight = (const unsigned char*)gate_weight_t_ptrs[expert_id];
            B_block_scale = (const __nv_bfloat16*)gate_block_scale_t_ptrs[expert_id];
            C = gate_out;
        } else {
            B_weight = (const unsigned char*)up_weight_t_ptrs[expert_id];
            B_block_scale = (const __nv_bfloat16*)up_block_scale_t_ptrs[expert_id];
            C = up_out;
        }
        if (B_weight == 0) {
            const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
            if (n < N) C[(unsigned long long)expert_slot * N + n] = __float2bfloat16(0.0f);
            return;
        }
    }

    const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (n >= N) return;

    // Iterate K, accumulate. Block scale changes every FP8_BLOCK k's.
    const unsigned int n_block = n / FP8_BLOCK;
    const unsigned int n_blocks = (N + FP8_BLOCK - 1) / FP8_BLOCK;
    const unsigned int k_blocks = (K + FP8_BLOCK - 1) / FP8_BLOCK;
    float acc = 0.0f;

    for (unsigned int k_block = 0; k_block < k_blocks; k_block++) {
        // Transposed scale layout: [k_blocks, n_blocks]
        float sc = __bfloat162float(B_block_scale[(unsigned long long)k_block * n_blocks + n_block]);
        const unsigned int k_start = k_block * FP8_BLOCK;
        const unsigned int k_end = (k_start + FP8_BLOCK) < K ? (k_start + FP8_BLOCK) : K;
        #pragma unroll 8
        for (unsigned int k = k_start; k < k_end; k++) {
            unsigned char w_byte = B_weight[(unsigned long long)k * N + n];
            __nv_fp8_e4m3 fp8w; *(unsigned char*)&fp8w = w_byte;
            float wf = (float)fp8w * sc;
            float af = __bfloat162float(A[k]);
            acc += wf * af;
        }
    }

    if (is_shared) {
        C[n] = __float2bfloat16(acc);
    } else {
        C[(unsigned long long)expert_slot * N + n] = __float2bfloat16(acc);
    }
}

extern "C" __global__ void moe_expert_silu_down_shared_fp8_t(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned long long* __restrict__ weight_t_ptrs,
    const unsigned long long* __restrict__ block_scale_t_ptrs,
    __nv_bfloat16* __restrict__ C,
    const unsigned int* __restrict__ expert_indices,
    const __nv_bfloat16* __restrict__ sh_gate_in,
    const __nv_bfloat16* __restrict__ sh_up_in,
    const unsigned char* __restrict__ sh_down_t_weight,
    const __nv_bfloat16* __restrict__ sh_down_t_block_scale,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    const unsigned int expert_slot = blockIdx.y;
    const bool is_shared = (expert_slot == top_k);

    const unsigned char* B_weight;
    const __nv_bfloat16* B_block_scale;
    const __nv_bfloat16* g_ptr;
    const __nv_bfloat16* u_ptr;
    if (is_shared) {
        B_weight = sh_down_t_weight; B_block_scale = sh_down_t_block_scale;
        g_ptr = sh_gate_in; u_ptr = sh_up_in;
    } else {
        const unsigned int expert_id = expert_indices[expert_slot];
        B_weight = (const unsigned char*)weight_t_ptrs[expert_id];
        B_block_scale = (const __nv_bfloat16*)block_scale_t_ptrs[expert_id];
        g_ptr = gate_out + (unsigned long long)expert_slot * K;
        u_ptr = up_out + (unsigned long long)expert_slot * K;
        if (B_weight == 0) {
            const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
            if (n < N) C[(unsigned long long)expert_slot * N + n] = __float2bfloat16(0.0f);
            return;
        }
    }

    const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const bool valid = (n < N);

    extern __shared__ float s_act[];
    for (unsigned int i = threadIdx.x; i < K; i += BLOCK_SIZE) {
        float gf = __bfloat162float(g_ptr[i]);
        float uf = __bfloat162float(u_ptr[i]);
        s_act[i] = (gf / (1.0f + __expf(-gf))) * uf;
    }
    __syncthreads();
    if (!valid) return;

    const unsigned int n_block = n / FP8_BLOCK;
    const unsigned int n_blocks = (N + FP8_BLOCK - 1) / FP8_BLOCK;
    const unsigned int k_blocks = (K + FP8_BLOCK - 1) / FP8_BLOCK;
    float acc = 0.0f;

    for (unsigned int k_block = 0; k_block < k_blocks; k_block++) {
        float sc = __bfloat162float(B_block_scale[(unsigned long long)k_block * n_blocks + n_block]);
        const unsigned int k_start = k_block * FP8_BLOCK;
        const unsigned int k_end = (k_start + FP8_BLOCK) < K ? (k_start + FP8_BLOCK) : K;
        #pragma unroll 8
        for (unsigned int k = k_start; k < k_end; k++) {
            unsigned char w_byte = B_weight[(unsigned long long)k * N + n];
            __nv_fp8_e4m3 fp8w; *(unsigned char*)&fp8w = w_byte;
            acc += (float)fp8w * sc * s_act[k];
        }
    }

    if (is_shared) {
        sh_down_out[n] = __float2bfloat16(acc);
    } else {
        C[(unsigned long long)expert_slot * N + n] = __float2bfloat16(acc);
    }
}
