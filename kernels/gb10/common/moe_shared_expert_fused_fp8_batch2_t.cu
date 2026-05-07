// SPDX-License-Identifier: AGPL-3.0-only
//
// Transposed-layout decode MoE — FP8 K=2 batch variant.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define BLOCK_SIZE 32
#define FP8_BLOCK 128

extern "C" __global__ void moe_expert_gate_up_shared_fp8_batch2_t(
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
    const unsigned int total_routed = 2 * top_k;
    const unsigned int y = blockIdx.y;
    const unsigned int proj = blockIdx.z;
    const bool is_shared = (y >= total_routed);
    unsigned int token, expert_slot;
    if (is_shared) { token = y - total_routed; expert_slot = 0; }
    else { token = y / top_k; expert_slot = y % top_k; }

    const __nv_bfloat16* A_token = A + (unsigned long long)token * K;
    const unsigned char* B_weight;
    const __nv_bfloat16* B_block_scale;
    __nv_bfloat16* C;
    unsigned long long c_offset = 0;

    if (is_shared) {
        if (proj == 0) {
            B_weight = sh_gate_t_weight; B_block_scale = sh_gate_t_block_scale;
            C = sh_gate_out; c_offset = (unsigned long long)token * N;
        } else {
            B_weight = sh_up_t_weight; B_block_scale = sh_up_t_block_scale;
            C = sh_up_out; c_offset = (unsigned long long)token * N;
        }
    } else {
        const unsigned int expert_id = expert_indices[token * top_k + expert_slot];
        const unsigned int flat_slot = token * top_k + expert_slot;
        if (proj == 0) {
            B_weight = (const unsigned char*)gate_weight_t_ptrs[expert_id];
            B_block_scale = (const __nv_bfloat16*)gate_block_scale_t_ptrs[expert_id];
            C = gate_out;
        } else {
            B_weight = (const unsigned char*)up_weight_t_ptrs[expert_id];
            B_block_scale = (const __nv_bfloat16*)up_block_scale_t_ptrs[expert_id];
            C = up_out;
        }
        c_offset = (unsigned long long)flat_slot * N;
        if (B_weight == 0) {
            const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
            if (n < N) C[c_offset + n] = __float2bfloat16(0.0f);
            return;
        }
    }

    const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (n >= N) return;
    const unsigned int n_block = n / FP8_BLOCK;
    const unsigned int n_blocks = (N + FP8_BLOCK - 1) / FP8_BLOCK;
    const unsigned int k_blocks = (K + FP8_BLOCK - 1) / FP8_BLOCK;
    float acc = 0.0f;
    for (unsigned int kb = 0; kb < k_blocks; kb++) {
        float sc = __bfloat162float(B_block_scale[(unsigned long long)kb * n_blocks + n_block]);
        const unsigned int k_start = kb * FP8_BLOCK;
        const unsigned int k_end = (k_start + FP8_BLOCK) < K ? (k_start + FP8_BLOCK) : K;
        #pragma unroll 8
        for (unsigned int k = k_start; k < k_end; k++) {
            unsigned char w_byte = B_weight[(unsigned long long)k * N + n];
            __nv_fp8_e4m3 fp8w; *(unsigned char*)&fp8w = w_byte;
            acc += (float)fp8w * sc * __bfloat162float(A_token[k]);
        }
    }
    C[c_offset + n] = __float2bfloat16(acc);
}

extern "C" __global__ void moe_expert_silu_down_shared_fp8_batch2_t(
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
    const unsigned int total_routed = 2 * top_k;
    const unsigned int y = blockIdx.y;
    const bool is_shared = (y >= total_routed);
    unsigned int token, expert_slot;
    if (is_shared) { token = y - total_routed; expert_slot = 0; }
    else { token = y / top_k; expert_slot = y % top_k; }

    const unsigned char* B_weight;
    const __nv_bfloat16* B_block_scale;
    const __nv_bfloat16* g_ptr;
    const __nv_bfloat16* u_ptr;
    unsigned long long c_offset;

    if (is_shared) {
        B_weight = sh_down_t_weight; B_block_scale = sh_down_t_block_scale;
        g_ptr = sh_gate_in + (unsigned long long)token * K;
        u_ptr = sh_up_in + (unsigned long long)token * K;
        c_offset = (unsigned long long)token * N;
    } else {
        const unsigned int expert_id = expert_indices[token * top_k + expert_slot];
        const unsigned int flat_slot = token * top_k + expert_slot;
        B_weight = (const unsigned char*)weight_t_ptrs[expert_id];
        B_block_scale = (const __nv_bfloat16*)block_scale_t_ptrs[expert_id];
        g_ptr = gate_out + (unsigned long long)flat_slot * K;
        u_ptr = up_out + (unsigned long long)flat_slot * K;
        c_offset = (unsigned long long)flat_slot * N;
        if (B_weight == 0) {
            const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
            if (n < N) C[c_offset + n] = __float2bfloat16(0.0f);
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
    for (unsigned int kb = 0; kb < k_blocks; kb++) {
        float sc = __bfloat162float(B_block_scale[(unsigned long long)kb * n_blocks + n_block]);
        const unsigned int k_start = kb * FP8_BLOCK;
        const unsigned int k_end = (k_start + FP8_BLOCK) < K ? (k_start + FP8_BLOCK) : K;
        #pragma unroll 8
        for (unsigned int k = k_start; k < k_end; k++) {
            unsigned char w_byte = B_weight[(unsigned long long)k * N + n];
            __nv_fp8_e4m3 fp8w; *(unsigned char*)&fp8w = w_byte;
            acc += (float)fp8w * sc * s_act[k];
        }
    }
    if (is_shared) {
        sh_down_out[c_offset + n] = __float2bfloat16(acc);
    } else {
        C[c_offset + n] = __float2bfloat16(acc);
    }
}
