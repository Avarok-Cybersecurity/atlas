// SPDX-License-Identifier: AGPL-3.0-only
//
// Transposed-layout decode MoE — K=3 batch variant. Same structure as
// moe_shared_expert_fused_batch2_t.cu with 3 tokens instead of 2 (used
// by MTP K=3 verify on MiniMax / Qwen3 MoE). See
// moe_shared_expert_fused_t.cu for layout rationale.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define BLOCK_SIZE 32
#define GROUP_SIZE 16

__device__ __constant__ float E2M1_LUT_BATCH3_T[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

extern "C" __global__ void moe_expert_gate_up_shared_batch3_t(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ gate_packed_t_ptrs,
    const unsigned long long* __restrict__ gate_scale_t_ptrs,
    const float* __restrict__ gate_scale2_vals,
    __nv_bfloat16* __restrict__ gate_out,
    const unsigned long long* __restrict__ up_packed_t_ptrs,
    const unsigned long long* __restrict__ up_scale_t_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ up_out,
    const unsigned int* __restrict__ expert_indices,
    const unsigned char* __restrict__ sh_gate_t_packed,
    const unsigned char* __restrict__ sh_gate_t_scale,
    float sh_gate_s2,
    __nv_bfloat16* __restrict__ sh_gate_out,
    const unsigned char* __restrict__ sh_up_t_packed,
    const unsigned char* __restrict__ sh_up_t_scale,
    float sh_up_s2,
    __nv_bfloat16* __restrict__ sh_up_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    const unsigned int total_routed = 3 * top_k;
    const unsigned int y = blockIdx.y;
    const unsigned int proj = blockIdx.z;
    const bool is_shared = (y >= total_routed);

    unsigned int token, expert_slot;
    if (is_shared) {
        token = y - total_routed;
        expert_slot = 0;
    } else {
        token = y / top_k;
        expert_slot = y % top_k;
    }

    const __nv_bfloat16* A_token = A + (unsigned long long)token * K;
    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    __nv_bfloat16* C;
    unsigned long long c_offset = 0;

    if (is_shared) {
        if (proj == 0) {
            if (sh_gate_t_packed == 0) {
                const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
                if (n < N) sh_gate_out[(unsigned long long)token * N + n] = __float2bfloat16(0.0f);
                return;
            }
            B_packed = sh_gate_t_packed; B_scale = sh_gate_t_scale; s2 = sh_gate_s2;
            C = sh_gate_out; c_offset = (unsigned long long)token * N;
        } else {
            if (sh_up_t_packed == 0) {
                const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
                if (n < N) sh_up_out[(unsigned long long)token * N + n] = __float2bfloat16(0.0f);
                return;
            }
            B_packed = sh_up_t_packed; B_scale = sh_up_t_scale; s2 = sh_up_s2;
            C = sh_up_out; c_offset = (unsigned long long)token * N;
        }
    } else {
        const unsigned int expert_id = expert_indices[token * top_k + expert_slot];
        const unsigned int flat_slot = token * top_k + expert_slot;
        if (proj == 0) {
            B_packed = (const unsigned char*)gate_packed_t_ptrs[expert_id];
            B_scale = (const unsigned char*)gate_scale_t_ptrs[expert_id];
            s2 = gate_scale2_vals[expert_id]; C = gate_out;
        } else {
            B_packed = (const unsigned char*)up_packed_t_ptrs[expert_id];
            B_scale = (const unsigned char*)up_scale_t_ptrs[expert_id];
            s2 = up_scale2_vals[expert_id]; C = up_out;
        }
        c_offset = (unsigned long long)flat_slot * N;
        if (B_packed == 0) {
            const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
            if (n < N) C[c_offset + n] = __float2bfloat16(0.0f);
            return;
        }
    }

    const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const bool valid = (n < N);

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_BATCH3_T[threadIdx.x];
    __syncthreads();
    if (!valid) return;

    const unsigned int num_groups = K / GROUP_SIZE;
    float acc = 0.0f;
    for (unsigned int sg = 0; sg < num_groups; sg++) {
        unsigned char sb = B_scale[(unsigned long long)sg * N + n];
        __nv_fp8_e4m3 fp8_s; *(unsigned char*)&fp8_s = sb;
        float sc = (float)fp8_s * s2;
        const unsigned int kh_base = sg * 8;
        #pragma unroll
        for (unsigned int kh_off = 0; kh_off < 8; kh_off++) {
            unsigned int k_half = kh_base + kh_off;
            unsigned char byte = B_packed[(unsigned long long)k_half * N + n];
            float a_lo = __bfloat162float(A_token[k_half * 2]);
            float a_hi = __bfloat162float(A_token[k_half * 2 + 1]);
            float w_lo = s_lut[byte & 0xFu] * sc;
            float w_hi = s_lut[(byte >> 4) & 0xFu] * sc;
            acc += a_lo * w_lo + a_hi * w_hi;
        }
    }
    C[c_offset + n] = __float2bfloat16(acc);
}

extern "C" __global__ void moe_expert_silu_down_shared_batch3_t(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned long long* __restrict__ packed_t_ptrs,
    const unsigned long long* __restrict__ scale_t_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const unsigned int* __restrict__ expert_indices,
    const __nv_bfloat16* __restrict__ sh_gate_in,
    const __nv_bfloat16* __restrict__ sh_up_in,
    const unsigned char* __restrict__ sh_down_t_packed,
    const unsigned char* __restrict__ sh_down_t_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    const unsigned int total_routed = 3 * top_k;
    const unsigned int y = blockIdx.y;
    const bool is_shared = (y >= total_routed);
    unsigned int token, expert_slot;
    if (is_shared) {
        token = y - total_routed;
        expert_slot = 0;
    } else {
        token = y / top_k;
        expert_slot = y % top_k;
    }

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    const __nv_bfloat16* g_ptr;
    const __nv_bfloat16* u_ptr;
    unsigned long long c_offset;

    if (is_shared) {
        if (sh_down_t_packed == 0) {
            const unsigned int n = blockIdx.x * BLOCK_SIZE + threadIdx.x;
            if (n < N) sh_down_out[(unsigned long long)token * N + n] = __float2bfloat16(0.0f);
            return;
        }
        B_packed = sh_down_t_packed; B_scale = sh_down_t_scale; s2 = sh_down_s2;
        g_ptr = sh_gate_in + (unsigned long long)token * K;
        u_ptr = sh_up_in + (unsigned long long)token * K;
        c_offset = (unsigned long long)token * N;
    } else {
        const unsigned int expert_id = expert_indices[token * top_k + expert_slot];
        const unsigned int flat_slot = token * top_k + expert_slot;
        B_packed = (const unsigned char*)packed_t_ptrs[expert_id];
        B_scale = (const unsigned char*)scale_t_ptrs[expert_id];
        s2 = scale2_vals[expert_id];
        g_ptr = gate_out + (unsigned long long)flat_slot * K;
        u_ptr = up_out + (unsigned long long)flat_slot * K;
        c_offset = (unsigned long long)flat_slot * N;
        if (B_packed == 0) {
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
    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_BATCH3_T[threadIdx.x];
    __syncthreads();
    if (!valid) return;

    const unsigned int num_groups = K / GROUP_SIZE;
    float acc = 0.0f;
    for (unsigned int sg = 0; sg < num_groups; sg++) {
        unsigned char sb = B_scale[(unsigned long long)sg * N + n];
        __nv_fp8_e4m3 fp8_s; *(unsigned char*)&fp8_s = sb;
        float sc = (float)fp8_s * s2;
        const unsigned int kh_base = sg * 8;
        #pragma unroll
        for (unsigned int kh_off = 0; kh_off < 8; kh_off++) {
            unsigned int k_half = kh_base + kh_off;
            unsigned char byte = B_packed[(unsigned long long)k_half * N + n];
            float w_lo = s_lut[byte & 0xFu] * sc;
            float w_hi = s_lut[(byte >> 4) & 0xFu] * sc;
            acc += s_act[k_half * 2] * w_lo + s_act[k_half * 2 + 1] * w_hi;
        }
    }
    if (is_shared) {
        sh_down_out[c_offset + n] = __float2bfloat16(acc);
    } else {
        C[c_offset + n] = __float2bfloat16(acc);
    }
}
