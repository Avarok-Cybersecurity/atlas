// SPDX-License-Identifier: AGPL-3.0-only

// Atlas Fused MoE Expert+Shared GEMV — shared expert as extra blockIdx.y slot.
//
// Same as moe_expert_gemv_fused.cu gate_up_2x / silu_down_2x but with
// blockIdx.y == top_k serving the shared expert using direct weight pointers.
// The shared expert blocks run concurrently with routed expert blocks within
// the same kernel launch, eliminating 2 separate kernel launches per layer
// (96 graph nodes across 48 MoE layers).
//
// Grid: gate_up (ceil(N/8), top_k+1, 2),  silu_down (ceil(N/8), top_k+1, 1)

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define BLOCK_SIZE 128
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define GROUP_SIZE 16

__device__ __constant__ float E2M1_LUT_SHARED[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// ── Fused Gate+Up 2x with shared expert ──
//
// blockIdx.y < top_k: routed expert (pointer table lookup)
// blockIdx.y == top_k: shared expert (direct weight pointers)
// Grid: (ceil(N/8), top_k+1, 2)  Block: (128, 1, 1)
extern "C" __global__ void moe_expert_gate_up_shared(
    const __nv_bfloat16* __restrict__ A,
    // Routed expert tables
    const unsigned long long* __restrict__ gate_packed_ptrs,
    const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals,
    __nv_bfloat16* __restrict__ gate_out,
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ up_out,
    const unsigned int* __restrict__ expert_indices,
    // Shared expert direct pointers
    const unsigned char* __restrict__ sh_gate_packed,
    const unsigned char* __restrict__ sh_gate_scale,
    float sh_gate_s2,
    __nv_bfloat16* __restrict__ sh_gate_out,
    const unsigned char* __restrict__ sh_up_packed,
    const unsigned char* __restrict__ sh_up_scale,
    float sh_up_s2,
    __nv_bfloat16* __restrict__ sh_up_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    const unsigned int expert_slot = blockIdx.y;
    const unsigned int proj = blockIdx.z;
    const bool is_shared = (expert_slot == top_k);

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;
    __nv_bfloat16* C;

    if (is_shared) {
        // NULL shared expert: model has no shared expert weights (e.g., Mistral).
        // Write zeros and return to prevent NULL pointer dereference.
        if (sh_gate_packed == 0) {
            __nv_bfloat16* out = (proj == 0) ? sh_gate_out : sh_up_out;
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE)
                out[n_base + i] = __float2bfloat16(0.0f);
            return;
        }
        if (proj == 0) {
            B_packed = sh_gate_packed; B_scale = sh_gate_scale;
            s2 = sh_gate_s2; C = sh_gate_out;
        } else {
            B_packed = sh_up_packed; B_scale = sh_up_scale;
            s2 = sh_up_s2; C = sh_up_out;
        }
    } else {
        const unsigned int expert_id = expert_indices[expert_slot];
        if (proj == 0) {
            B_packed = (const unsigned char*)gate_packed_ptrs[expert_id];
            B_scale = (const unsigned char*)gate_scale_ptrs[expert_id];
            s2 = gate_scale2_vals[expert_id]; C = gate_out;
        } else {
            B_packed = (const unsigned char*)up_packed_ptrs[expert_id];
            B_scale = (const unsigned char*)up_scale_ptrs[expert_id];
            s2 = up_scale2_vals[expert_id]; C = up_out;
        }
        // EP: NULL pointer means remote expert — write zero output and return
        if (B_packed == 0) {
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE) {
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
    if (n1 >= N) return;
    const bool have_n2 = (n2 < N);

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_SHARED[threadIdx.x];
    __syncthreads();

    float acc1 = 0.0f, acc2 = 0.0f;

    // 16 K-values per iteration: uint64 weight + 2×uint4 activation
    for (unsigned int k16 = lane; k16 < K16; k16 += threads_per_out) {
        const unsigned int base_k = k16 * 16;
        uint4 a_lo = ((const uint4*)A)[k16 * 2];
        uint4 a_hi = ((const uint4*)A)[k16 * 2 + 1];
        const unsigned int a_raw[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                                        a_hi.x, a_hi.y, a_hi.z, a_hi.w};

        unsigned long long packed8_1 = *(const unsigned long long*)(B_packed + (unsigned long long)n1 * half_K + k16 * 8);
        unsigned int sg = base_k / GROUP_SIZE;
        unsigned char sb1 = B_scale[(unsigned long long)n1 * num_groups + sg];
        __nv_fp8_e4m3 fp8_1; *(unsigned char*)&fp8_1 = sb1;
        float sc1 = (float)fp8_1 * s2;

        unsigned long long packed8_2 = have_n2 ?
            *(const unsigned long long*)(B_packed + (unsigned long long)n2 * half_K + k16 * 8) : 0;
        unsigned char sb2 = have_n2 ? B_scale[(unsigned long long)n2 * num_groups + sg] : 0;
        __nv_fp8_e4m3 fp8_2; *(unsigned char*)&fp8_2 = sb2;
        float sc2 = have_n2 ? (float)fp8_2 * s2 : 0.0f;

        #pragma unroll
        for (int b = 0; b < 8; b++) {
            unsigned char bv1 = (unsigned char)(packed8_1 >> (b * 8));
            float w1l = s_lut[bv1 & 0xF] * sc1, w1h = s_lut[bv1 >> 4] * sc1;
            unsigned char bv2 = (unsigned char)(packed8_2 >> (b * 8));
            float w2l = s_lut[bv2 & 0xF] * sc2, w2h = s_lut[bv2 >> 4] * sc2;
            __nv_bfloat16 al, ah;
            *(unsigned short*)&al = (unsigned short)(a_raw[b] & 0xFFFF);
            *(unsigned short*)&ah = (unsigned short)(a_raw[b] >> 16);
            float afl = __bfloat162float(al), afh = __bfloat162float(ah);
            acc1 += afl * w1l + afh * w1h;
            acc2 += afl * w2l + afh * w2h;
        }
    }

    // Output offset: shared expert writes at [0..N], routed at [slot*N..N]
    const unsigned long long base = is_shared ? 0 : (unsigned long long)expert_slot * N;

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
    if (lane == 0) C[base + n1] = __float2bfloat16(acc1);

    if (have_n2) {
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
        if (lane == 0) C[base + n2] = __float2bfloat16(acc2);
    }
}

// ── Fused SiLU+Down 2x with shared expert ──
//
// Precomputes SiLU(gate)*up in shared memory once per block, eliminating
// redundant SiLU compute across all 4 thread groups and replacing global
// gate/up loads with fast shared memory reads in the GEMV inner loop.
//
// blockIdx.y < top_k: routed expert (pointer table + expert_gate_out/up_out)
// blockIdx.y == top_k: shared expert (direct pointers + sh_gate_in/up_in)
// Grid: (ceil(N/8), top_k+1, 1)  Block: (128, 1, 1)
extern "C" __global__ void moe_expert_silu_down_shared(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const unsigned int* __restrict__ expert_indices,
    // Shared expert
    const __nv_bfloat16* __restrict__ sh_gate_in,
    const __nv_bfloat16* __restrict__ sh_up_in,
    const unsigned char* __restrict__ sh_down_packed,
    const unsigned char* __restrict__ sh_down_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int top_k
) {
    const unsigned int expert_slot = blockIdx.y;
    const bool is_shared = (expert_slot == top_k);

    const unsigned char* B_packed;
    const unsigned char* B_scale;
    float s2;

    const __nv_bfloat16* g_ptr;
    const __nv_bfloat16* u_ptr;

    if (is_shared) {
        // NULL shared expert: write zeros and return
        if (sh_down_packed == 0) {
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE)
                sh_down_out[n_base + i] = __float2bfloat16(0.0f);
            return;
        }
        B_packed = sh_down_packed; B_scale = sh_down_scale; s2 = sh_down_s2;
        g_ptr = sh_gate_in; u_ptr = sh_up_in;
    } else {
        const unsigned int expert_id = expert_indices[expert_slot];
        B_packed = (const unsigned char*)packed_ptrs[expert_id];
        B_scale = (const unsigned char*)scale_ptrs[expert_id];
        s2 = scale2_vals[expert_id];
        g_ptr = gate_out + (unsigned long long)expert_slot * K;
        u_ptr = up_out + (unsigned long long)expert_slot * K;
        // EP: NULL pointer means remote expert — write zero output and return
        if (B_packed == 0) {
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE) {
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
    if (n1 >= N) return;
    const bool have_n2 = (n2 < N);

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    __shared__ float s_lut[16];
    extern __shared__ float s_act[];

    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_SHARED[threadIdx.x];

    // Phase 1: Cooperatively precompute SiLU(gate)*up into shared memory
    for (unsigned int i = threadIdx.x; i < K; i += BLOCK_SIZE) {
        float gf = __bfloat162float(g_ptr[i]);
        float uf = __bfloat162float(u_ptr[i]);
        // GeGLU: GELU(gate) * up — Gemma-4 26B
        float inner = 0.7978845608f * (gf + 0.044715f * gf * gf * gf);
        s_act[i] = (0.5f * gf * (1.0f + tanhf(inner))) * uf;
    }
    __syncthreads();

    // Phase 2: GEMV with 16 K-values per iteration
    float acc1 = 0.0f, acc2 = 0.0f;

    for (unsigned int k16 = lane; k16 < K16; k16 += threads_per_out) {
        const unsigned int base_k = k16 * 16;

        unsigned long long packed8_1 = *(const unsigned long long*)(B_packed + (unsigned long long)n1 * half_K + k16 * 8);
        unsigned int sg = base_k / GROUP_SIZE;
        unsigned char sb1 = B_scale[(unsigned long long)n1 * num_groups + sg];
        __nv_fp8_e4m3 fp8_1; *(unsigned char*)&fp8_1 = sb1;
        float sc1 = (float)fp8_1 * s2;

        unsigned long long packed8_2 = have_n2 ?
            *(const unsigned long long*)(B_packed + (unsigned long long)n2 * half_K + k16 * 8) : 0;
        unsigned char sb2 = have_n2 ? B_scale[(unsigned long long)n2 * num_groups + sg] : 0;
        __nv_fp8_e4m3 fp8_2; *(unsigned char*)&fp8_2 = sb2;
        float sc2 = have_n2 ? (float)fp8_2 * s2 : 0.0f;

        #pragma unroll
        for (int b = 0; b < 8; b++) {
            float al = s_act[base_k + b * 2];
            float ah = s_act[base_k + b * 2 + 1];

            unsigned char bv1 = (unsigned char)(packed8_1 >> (b * 8));
            float w1l = s_lut[bv1 & 0xF] * sc1, w1h = s_lut[bv1 >> 4] * sc1;
            unsigned char bv2 = (unsigned char)(packed8_2 >> (b * 8));
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
