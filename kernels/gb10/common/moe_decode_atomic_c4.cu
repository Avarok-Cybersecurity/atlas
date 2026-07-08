// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define BLOCK_SIZE 128
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define GROUP_SIZE 16

__device__ __constant__ float E2M1_LUT_ATOMIC_C4[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ float atlas_dec_e4m3_atomic_c4(unsigned char b) {
    unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u;
    float v;
    if (e == 0u) {
        v = (float)m * 0.001953125f;
    } else if (e == 15u && m == 7u) {
        v = 0.0f;
    } else {
        v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    }
    return s ? -v : v;
}
#else
__device__ __forceinline__ float atlas_dec_e4m3_atomic_c4(unsigned char b) {
    __nv_fp8_e4m3 f;
    *(unsigned char*)&f = b;
    return (float)f;
}
#endif

extern "C" __global__ void moe_decode_atomic_c4_silu_down_accum(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    const unsigned int* __restrict__ expert_indices,
    const float* __restrict__ expert_weights,
    float* __restrict__ routed_accum,
    const __nv_bfloat16* __restrict__ sh_gate_in,
    const __nv_bfloat16* __restrict__ sh_up_in,
    const unsigned char* __restrict__ sh_down_packed,
    const unsigned char* __restrict__ sh_down_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int hidden,
    unsigned int inter,
    unsigned int top_k,
    unsigned int num_tokens
) {
    const unsigned int total_routed = num_tokens * top_k;
    const unsigned int y = blockIdx.y;
    const bool is_shared = (y >= total_routed);

    unsigned int token;
    unsigned int expert_slot;
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

    if (is_shared) {
        if (sh_down_packed == 0) {
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < hidden; i += BLOCK_SIZE) {
                sh_down_out[(unsigned long long)token * hidden + n_base + i] = __float2bfloat16(0.0f);
            }
            return;
        }
        B_packed = sh_down_packed;
        B_scale = sh_down_scale;
        s2 = sh_down_s2;
        g_ptr = sh_gate_in + (unsigned long long)token * inter;
        u_ptr = sh_up_in + (unsigned long long)token * inter;
    } else {
        const unsigned int route = token * top_k + expert_slot;
        const unsigned int expert_id = expert_indices[route];
        B_packed = (const unsigned char*)packed_ptrs[expert_id];
        B_scale = (const unsigned char*)scale_ptrs[expert_id];
        s2 = scale2_vals[expert_id];
        g_ptr = gate_out + (unsigned long long)route * inter;
        u_ptr = up_out + (unsigned long long)route * inter;
        if (B_packed == 0) return;
    }

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n1 = blockIdx.x * (N_PER_BLOCK * 2) + local_out * 2;
    const unsigned int n2 = n1 + 1;
    if (n1 >= hidden) return;
    const bool have_n2 = (n2 < hidden);

    const unsigned int half_K = inter / 2;
    const unsigned int num_groups = inter / GROUP_SIZE;
    const unsigned int K8 = inter / 8;

    __shared__ float s_lut[16];
    extern __shared__ float s_act[];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_ATOMIC_C4[threadIdx.x];

    for (unsigned int i = threadIdx.x; i < inter; i += BLOCK_SIZE) {
        float gf = __bfloat162float(g_ptr[i]);
        float uf = __bfloat162float(u_ptr[i]);
        s_act[i] = (gf / (1.0f + __expf(-gf))) * uf;
    }
    __syncthreads();

    float acc1 = 0.0f;
    float acc2 = 0.0f;
    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;
        unsigned int packed4_1 = *(const unsigned int*)(B_packed + (unsigned long long)n1 * half_K + k8 * 4);
        unsigned int sg = base_k / GROUP_SIZE;
        unsigned char sb1 = B_scale[(unsigned long long)n1 * num_groups + sg];
        float sc1 = atlas_dec_e4m3_atomic_c4(sb1) * s2;

        unsigned int packed4_2 = have_n2
            ? *(const unsigned int*)(B_packed + (unsigned long long)n2 * half_K + k8 * 4)
            : 0;
        unsigned char sb2 = have_n2 ? B_scale[(unsigned long long)n2 * num_groups + sg] : 0;
        float sc2 = have_n2 ? atlas_dec_e4m3_atomic_c4(sb2) * s2 : 0.0f;

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            float al = s_act[base_k + b * 2];
            float ah = s_act[base_k + b * 2 + 1];
            unsigned char bv1 = (packed4_1 >> (b * 8)) & 0xFF;
            float w1l = s_lut[bv1 & 0xF] * sc1;
            float w1h = s_lut[bv1 >> 4] * sc1;
            unsigned char bv2 = (packed4_2 >> (b * 8)) & 0xFF;
            float w2l = s_lut[bv2 & 0xF] * sc2;
            float w2h = s_lut[bv2 >> 4] * sc2;
            acc1 += al * w1l + ah * w1h;
            acc2 += al * w2l + ah * w2h;
        }
    }

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);

    if (lane == 0) {
        if (is_shared) {
            sh_down_out[(unsigned long long)token * hidden + n1] = __float2bfloat16(acc1);
        } else {
            const float w = expert_weights[token * top_k + expert_slot];
            atomicAdd(routed_accum + (unsigned long long)token * hidden + n1, acc1 * w);
        }
    }

    if (have_n2) {
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
        if (lane == 0) {
            if (is_shared) {
                sh_down_out[(unsigned long long)token * hidden + n2] = __float2bfloat16(acc2);
            } else {
                const float w = expert_weights[token * top_k + expert_slot];
                atomicAdd(routed_accum + (unsigned long long)token * hidden + n2, acc2 * w);
            }
        }
    }
}

extern "C" __global__ void moe_decode_atomic_c4_finalize(
    __nv_bfloat16* __restrict__ output,
    const float* __restrict__ routed_accum,
    const __nv_bfloat16* __restrict__ shared_out,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ gate_weight,
    unsigned int hidden,
    unsigned int num_tokens,
    unsigned int include_shared
) {
    const unsigned int token = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int j = blockIdx.x * blockDim.x + tid;

    __shared__ float s_warp_sums[8];
    __shared__ float sigmoid_val;

    if (include_shared != 0) {
        float dot_acc = 0.0f;
        if (gate_weight != 0) {
            const __nv_bfloat16* input_t = input + (unsigned long long)token * hidden;
            const unsigned int K8 = hidden / 8;
            for (unsigned int k8 = tid; k8 < K8; k8 += 256) {
                uint4 a_data = ((const uint4*)input_t)[k8];
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
        }

        const unsigned int warp_id = tid / WARP_SIZE;
        const unsigned int lane = tid % WARP_SIZE;
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            dot_acc += __shfl_down_sync(0xFFFFFFFF, dot_acc, offset);
        if (lane == 0) s_warp_sums[warp_id] = dot_acc;
        __syncthreads();

        if (tid == 0) {
            if (gate_weight == 0) {
                sigmoid_val = 1.0f;
            } else {
                float total = 0.0f;
                #pragma unroll
                for (int w = 0; w < 8; w++) total += s_warp_sums[w];
                sigmoid_val = 1.0f / (1.0f + __expf(-total));
            }
        }
        __syncthreads();
    }

    if (j >= hidden || token >= num_tokens) return;

    float acc = routed_accum[(unsigned long long)token * hidden + j];
    if (include_shared != 0) {
        acc += sigmoid_val * __bfloat162float(shared_out[(unsigned long long)token * hidden + j]);
    }
    output[(unsigned long long)token * hidden + j] = __float2bfloat16(acc);
}
