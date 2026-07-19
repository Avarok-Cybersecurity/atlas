// SPDX-License-Identifier: AGPL-3.0-only

// Atlas Fused MoE Expert+Shared GEMV — K=2 multi-token batch, BF16 weight variant.
//
// For models loaded via the FP8-dequant-on-load path (bf16_*_weight_ptrs). The
// MTP K=2 verify step processes 2 tokens; the per-token BF16 decode kernels
// (moe_expert_*_shared_bf16) had to be launched twice, serializing the two
// tokens' expert dispatch and doubling the launch count. This variant collapses
// both tokens into single kernel launches by expanding blockIdx.y to hold two
// sets of (top_k routed + 1 shared) experts, exactly mirroring the proven FP8
// K=2 structure in moe_shared_expert_fused_fp8_batch2.cu — but with direct BF16
// weight pointers (2 bytes/weight, no scale, no LUT).
//
// Token layout in blockIdx.y:
//   y in [0, 2*top_k)  -> routed experts (token = y/top_k, slot = y%top_k)
//   y == 2*top_k       -> shared expert, computes BOTH tokens from a SINGLE
//                         pass over the shared weight (the guaranteed overlap
//                         between the two verify tokens — halves the shared
//                         expert's weight traffic vs a per-token dispatch).
//
// Grid: gate_up_batch2  (ceil(N/8), 2*top_k + 1, 2)
//       silu_down_batch2 (ceil(N/8), 2*top_k + 1, 1)
//
// Output layout matches the FP8 batch2 path so moe_weighted_sum_blend_batch2
// consumes it unchanged:
//   routed gate/up/down : C[(token*top_k + slot) * N + n]
//   shared gate/up/down : sh_out[token * N + n]  (both tokens written)

#include <cuda_bf16.h>

#define BLOCK_SIZE 128
#define N_PER_BLOCK 4
#define WARP_SIZE 32

// ── Fused Gate+Up 2x with shared expert — K=2 batch, BF16 variant ──
//
// blockIdx.z: 0 = gate, 1 = up
// Grid: (ceil(N/8), 2*(top_k+1), 2)  Block: (128, 1, 1)
extern "C" __global__ void moe_expert_gate_up_shared_bf16_batch2(
    const __nv_bfloat16* __restrict__ A,       // [2, K] BF16 input (2 tokens)
    // Routed expert tables (one ptr per expert per projection)
    const unsigned long long* __restrict__ gate_weight_ptrs,
    __nv_bfloat16* __restrict__ gate_out,      // [2*top_k, N] BF16
    const unsigned long long* __restrict__ up_weight_ptrs,
    __nv_bfloat16* __restrict__ up_out,        // [2*top_k, N] BF16
    const unsigned int* __restrict__ expert_indices,  // [2*top_k] u32
    // Shared expert direct pointers
    const __nv_bfloat16* __restrict__ sh_gate_weight,
    __nv_bfloat16* __restrict__ sh_gate_out,   // [2, N] BF16
    const __nv_bfloat16* __restrict__ sh_up_weight,
    __nv_bfloat16* __restrict__ sh_up_out,     // [2, N] BF16
    unsigned int N, unsigned int K, unsigned int top_k
) {
    const unsigned int total_routed = 2 * top_k;
    const unsigned int y = blockIdx.y;
    const unsigned int proj = blockIdx.z;
    const bool is_shared = (y == total_routed);   // single shared block, 2 tokens

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int n1 = blockIdx.x * (N_PER_BLOCK * 2) + local_out * 2;
    const unsigned int n2 = n1 + 1;
    const unsigned int K8 = K / 8;

    if (is_shared) {
        // Shared expert: read weight ONCE, compute BOTH tokens (the guaranteed
        // overlap → half the shared-weight traffic vs a per-token dispatch).
        const __nv_bfloat16* B_weight = (proj == 0) ? sh_gate_weight : sh_up_weight;
        __nv_bfloat16* C0 = ((proj == 0) ? sh_gate_out : sh_up_out);          // token 0
        __nv_bfloat16* C1 = C0 + (unsigned long long)N;                        // token 1
        // NULL shared weight (model without shared expert): zero both tokens.
        if (B_weight == 0) {
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE) {
                C0[n_base + i] = __float2bfloat16(0.0f);
                C1[n_base + i] = __float2bfloat16(0.0f);
            }
            return;
        }
        if (n1 >= N) return;
        const bool have_n2 = (n2 < N);
        const __nv_bfloat16* A0 = A;                       // token 0 input
        const __nv_bfloat16* A1 = A + (unsigned long long)K;  // token 1 input

        float a0n1 = 0.0f, a0n2 = 0.0f, a1n1 = 0.0f, a1n2 = 0.0f;
        for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
            uint4 w_n1 = ((const uint4*)(B_weight + (unsigned long long)n1 * K))[k8];
            uint4 w_n2;
            if (have_n2) w_n2 = ((const uint4*)(B_weight + (unsigned long long)n2 * K))[k8];
            else { w_n2.x = 0; w_n2.y = 0; w_n2.z = 0; w_n2.w = 0; }
            uint4 a0 = ((const uint4*)A0)[k8];
            uint4 a1 = ((const uint4*)A1)[k8];
            const unsigned int w1[4] = {w_n1.x, w_n1.y, w_n1.z, w_n1.w};
            const unsigned int w2[4] = {w_n2.x, w_n2.y, w_n2.z, w_n2.w};
            const unsigned int a0w[4] = {a0.x, a0.y, a0.z, a0.w};
            const unsigned int a1w[4] = {a1.x, a1.y, a1.z, a1.w};
            #pragma unroll
            for (int b = 0; b < 4; b++) {
                __nv_bfloat16 w1v0, w1v1, w2v0, w2v1, a0v0, a0v1, a1v0, a1v1;
                *(unsigned short*)&w1v0 = (unsigned short)(w1[b] & 0xFFFF);
                *(unsigned short*)&w1v1 = (unsigned short)(w1[b] >> 16);
                *(unsigned short*)&w2v0 = (unsigned short)(w2[b] & 0xFFFF);
                *(unsigned short*)&w2v1 = (unsigned short)(w2[b] >> 16);
                *(unsigned short*)&a0v0 = (unsigned short)(a0w[b] & 0xFFFF);
                *(unsigned short*)&a0v1 = (unsigned short)(a0w[b] >> 16);
                *(unsigned short*)&a1v0 = (unsigned short)(a1w[b] & 0xFFFF);
                *(unsigned short*)&a1v1 = (unsigned short)(a1w[b] >> 16);
                float wf1_0 = __bfloat162float(w1v0), wf1_1 = __bfloat162float(w1v1);
                float wf2_0 = __bfloat162float(w2v0), wf2_1 = __bfloat162float(w2v1);
                float a0f0 = __bfloat162float(a0v0), a0f1 = __bfloat162float(a0v1);
                float a1f0 = __bfloat162float(a1v0), a1f1 = __bfloat162float(a1v1);
                a0n1 += a0f0 * wf1_0 + a0f1 * wf1_1;
                a0n2 += a0f0 * wf2_0 + a0f1 * wf2_1;
                a1n1 += a1f0 * wf1_0 + a1f1 * wf1_1;
                a1n2 += a1f0 * wf2_0 + a1f1 * wf2_1;
            }
        }
        #pragma unroll
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            a0n1 += __shfl_down_sync(0xFFFFFFFF, a0n1, off);
            a1n1 += __shfl_down_sync(0xFFFFFFFF, a1n1, off);
        }
        if (lane == 0) { C0[n1] = __float2bfloat16(a0n1); C1[n1] = __float2bfloat16(a1n1); }
        if (have_n2) {
            #pragma unroll
            for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
                a0n2 += __shfl_down_sync(0xFFFFFFFF, a0n2, off);
                a1n2 += __shfl_down_sync(0xFFFFFFFF, a1n2, off);
            }
            if (lane == 0) { C0[n2] = __float2bfloat16(a0n2); C1[n2] = __float2bfloat16(a1n2); }
        }
        return;
    }

    // ── Routed expert: per token (data-dependent selection) ──
    const unsigned int token = y / top_k;          // 0 or 1
    const unsigned int expert_slot = y % top_k;     // 0..top_k-1
    const __nv_bfloat16* A_token = A + (unsigned long long)token * K;
    const unsigned int expert_id = expert_indices[token * top_k + expert_slot];
    const unsigned int flat_slot = token * top_k + expert_slot;
    const __nv_bfloat16* B_weight;
    __nv_bfloat16* C;
    if (proj == 0) {
        B_weight = (const __nv_bfloat16*)gate_weight_ptrs[expert_id];
        C = gate_out + (unsigned long long)flat_slot * N;
    } else {
        B_weight = (const __nv_bfloat16*)up_weight_ptrs[expert_id];
        C = up_out + (unsigned long long)flat_slot * N;
    }
    // EP: NULL pointer means remote expert — write zero output and return.
    if (B_weight == 0) {
        const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
        for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE)
            C[n_base + i] = __float2bfloat16(0.0f);
        return;
    }

    if (n1 >= N) return;
    const bool have_n2 = (n2 < N);
    float acc1 = 0.0f, acc2 = 0.0f;
    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        uint4 a_data = ((const uint4*)A_token)[k8];
        uint4 w_n1 = ((const uint4*)(B_weight + (unsigned long long)n1 * K))[k8];
        uint4 w_n2;
        if (have_n2) w_n2 = ((const uint4*)(B_weight + (unsigned long long)n2 * K))[k8];
        else { w_n2.x = 0; w_n2.y = 0; w_n2.z = 0; w_n2.w = 0; }
        const unsigned int aw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
        const unsigned int w1[4] = {w_n1.x, w_n1.y, w_n1.z, w_n1.w};
        const unsigned int w2[4] = {w_n2.x, w_n2.y, w_n2.z, w_n2.w};
        #pragma unroll
        for (int b = 0; b < 4; b++) {
            __nv_bfloat16 av0, av1, w1v0, w1v1, w2v0, w2v1;
            *(unsigned short*)&av0 = (unsigned short)(aw[b] & 0xFFFF);
            *(unsigned short*)&av1 = (unsigned short)(aw[b] >> 16);
            *(unsigned short*)&w1v0 = (unsigned short)(w1[b] & 0xFFFF);
            *(unsigned short*)&w1v1 = (unsigned short)(w1[b] >> 16);
            *(unsigned short*)&w2v0 = (unsigned short)(w2[b] & 0xFFFF);
            *(unsigned short*)&w2v1 = (unsigned short)(w2[b] >> 16);
            float af0 = __bfloat162float(av0), af1 = __bfloat162float(av1);
            float wf1_0 = __bfloat162float(w1v0), wf1_1 = __bfloat162float(w1v1);
            float wf2_0 = __bfloat162float(w2v0), wf2_1 = __bfloat162float(w2v1);
            acc1 += af0 * wf1_0 + af1 * wf1_1;
            acc2 += af0 * wf2_0 + af1 * wf2_1;
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

// ── Fused SiLU+Down 2x with shared expert — K=2 batch, BF16 variant ──
//
// Phase 1: cooperatively precompute SiLU(gate)*up into shared memory.
// Phase 2: GEMV against down_proj reading precomputed activation from smem.
//
// Grid: (ceil(N/8), 2*(top_k+1), 1)  Block: (128, 1, 1)
extern "C" __global__ void moe_expert_silu_down_shared_bf16_batch2(
    const __nv_bfloat16* __restrict__ gate_out,  // [2*top_k, inter] BF16
    const __nv_bfloat16* __restrict__ up_out,    // [2*top_k, inter] BF16
    const unsigned long long* __restrict__ weight_ptrs,
    __nv_bfloat16* __restrict__ C,               // [2*top_k, N] BF16
    const unsigned int* __restrict__ expert_indices,  // [2*top_k] u32
    // Shared expert
    const __nv_bfloat16* __restrict__ sh_gate_in,   // [2, inter] BF16
    const __nv_bfloat16* __restrict__ sh_up_in,     // [2, inter] BF16
    const __nv_bfloat16* __restrict__ sh_down_weight,
    __nv_bfloat16* __restrict__ sh_down_out,        // [2, N] BF16
    unsigned int N, unsigned int K, unsigned int top_k
) {
    const unsigned int total_routed = 2 * top_k;
    const unsigned int y = blockIdx.y;
    const bool is_shared = (y == total_routed);   // single shared block, 2 tokens

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int n1 = blockIdx.x * (N_PER_BLOCK * 2) + local_out * 2;
    const unsigned int n2 = n1 + 1;
    const unsigned int K8 = K / 8;

    // Single smem tile shared by both paths (SSOT for occupancy): routed uses
    // [0,K); the shared-expert path uses [0,2K) for both tokens. Sized 2048
    // floats (8 KB) — same as the per-token BF16 decode kernel — so occupancy
    // is unchanged. Requires 2*K <= 2048 for the shared path (K = MoE
    // intermediate size; 512–768 on the A3B models this path serves).
    __shared__ float s_act[2048];

    if (is_shared) {
        // Shared expert: read down weight ONCE, compute BOTH tokens.
        __nv_bfloat16* O0 = sh_down_out;                          // token 0
        __nv_bfloat16* O1 = sh_down_out + (unsigned long long)N;  // token 1
        if (sh_down_weight == 0) {
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE) {
                O0[n_base + i] = __float2bfloat16(0.0f);
                O1[n_base + i] = __float2bfloat16(0.0f);
            }
            return;
        }
        const __nv_bfloat16* B_weight = sh_down_weight;
        // SiLU(gate)*up for BOTH tokens into smem: [0,K) token0, [K,2K) token1.
        for (unsigned int i = threadIdx.x; i < K; i += BLOCK_SIZE) {
            float g0 = __bfloat162float(sh_gate_in[i]);
            float u0 = __bfloat162float(sh_up_in[i]);
            s_act[i] = (g0 / (1.0f + __expf(-g0))) * u0;
            float g1 = __bfloat162float(sh_gate_in[(unsigned long long)K + i]);
            float u1 = __bfloat162float(sh_up_in[(unsigned long long)K + i]);
            s_act[K + i] = (g1 / (1.0f + __expf(-g1))) * u1;
        }
        __syncthreads();
        if (n1 >= N) return;
        const bool have_n2 = (n2 < N);
        float a0n1 = 0.0f, a0n2 = 0.0f, a1n1 = 0.0f, a1n2 = 0.0f;
        for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
            const unsigned int base_k = k8 * 8;
            uint4 w_n1 = ((const uint4*)(B_weight + (unsigned long long)n1 * K))[k8];
            uint4 w_n2;
            if (have_n2) w_n2 = ((const uint4*)(B_weight + (unsigned long long)n2 * K))[k8];
            else { w_n2.x = 0; w_n2.y = 0; w_n2.z = 0; w_n2.w = 0; }
            const unsigned int w1[4] = {w_n1.x, w_n1.y, w_n1.z, w_n1.w};
            const unsigned int w2[4] = {w_n2.x, w_n2.y, w_n2.z, w_n2.w};
            #pragma unroll
            for (int b = 0; b < 4; b++) {
                __nv_bfloat16 w1v0, w1v1, w2v0, w2v1;
                *(unsigned short*)&w1v0 = (unsigned short)(w1[b] & 0xFFFF);
                *(unsigned short*)&w1v1 = (unsigned short)(w1[b] >> 16);
                *(unsigned short*)&w2v0 = (unsigned short)(w2[b] & 0xFFFF);
                *(unsigned short*)&w2v1 = (unsigned short)(w2[b] >> 16);
                float wf1_0 = __bfloat162float(w1v0), wf1_1 = __bfloat162float(w1v1);
                float wf2_0 = __bfloat162float(w2v0), wf2_1 = __bfloat162float(w2v1);
                float al0 = s_act[base_k + b * 2],       al1 = s_act[base_k + b * 2 + 1];
                float bl0 = s_act[K + base_k + b * 2],   bl1 = s_act[K + base_k + b * 2 + 1];
                a0n1 += al0 * wf1_0 + al1 * wf1_1;
                a0n2 += al0 * wf2_0 + al1 * wf2_1;
                a1n1 += bl0 * wf1_0 + bl1 * wf1_1;
                a1n2 += bl0 * wf2_0 + bl1 * wf2_1;
            }
        }
        #pragma unroll
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            a0n1 += __shfl_down_sync(0xFFFFFFFF, a0n1, off);
            a1n1 += __shfl_down_sync(0xFFFFFFFF, a1n1, off);
        }
        if (lane == 0) { O0[n1] = __float2bfloat16(a0n1); O1[n1] = __float2bfloat16(a1n1); }
        if (have_n2) {
            #pragma unroll
            for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
                a0n2 += __shfl_down_sync(0xFFFFFFFF, a0n2, off);
                a1n2 += __shfl_down_sync(0xFFFFFFFF, a1n2, off);
            }
            if (lane == 0) { O0[n2] = __float2bfloat16(a0n2); O1[n2] = __float2bfloat16(a1n2); }
        }
        return;
    }

    // ── Routed expert: per token ──
    const unsigned int token = y / top_k;
    const unsigned int expert_slot = y % top_k;
    const unsigned int expert_id = expert_indices[token * top_k + expert_slot];
    const unsigned int flat_slot = token * top_k + expert_slot;
    const __nv_bfloat16* B_weight = (const __nv_bfloat16*)weight_ptrs[expert_id];
    const __nv_bfloat16* g_ptr = gate_out + (unsigned long long)flat_slot * K;
    const __nv_bfloat16* u_ptr = up_out + (unsigned long long)flat_slot * K;
    if (B_weight == 0) {
        const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
        for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE)
            C[(unsigned long long)flat_slot * N + n_base + i] = __float2bfloat16(0.0f);
        return;
    }

    // SiLU(gate)*up for this (token,expert) into the shared s_act tile.
    for (unsigned int i = threadIdx.x; i < K; i += BLOCK_SIZE) {
        float gf = __bfloat162float(g_ptr[i]);
        float uf = __bfloat162float(u_ptr[i]);
        s_act[i] = (gf / (1.0f + __expf(-gf))) * uf;
    }
    __syncthreads();

    if (n1 >= N) return;
    const bool have_n2 = (n2 < N);
    float acc1 = 0.0f, acc2 = 0.0f;
    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;
        uint4 w_n1 = ((const uint4*)(B_weight + (unsigned long long)n1 * K))[k8];
        uint4 w_n2;
        if (have_n2) w_n2 = ((const uint4*)(B_weight + (unsigned long long)n2 * K))[k8];
        else { w_n2.x = 0; w_n2.y = 0; w_n2.z = 0; w_n2.w = 0; }
        const unsigned int w1[4] = {w_n1.x, w_n1.y, w_n1.z, w_n1.w};
        const unsigned int w2[4] = {w_n2.x, w_n2.y, w_n2.z, w_n2.w};
        #pragma unroll
        for (int b = 0; b < 4; b++) {
            __nv_bfloat16 w1v0, w1v1, w2v0, w2v1;
            *(unsigned short*)&w1v0 = (unsigned short)(w1[b] & 0xFFFF);
            *(unsigned short*)&w1v1 = (unsigned short)(w1[b] >> 16);
            *(unsigned short*)&w2v0 = (unsigned short)(w2[b] & 0xFFFF);
            *(unsigned short*)&w2v1 = (unsigned short)(w2[b] >> 16);
            float wf1_0 = __bfloat162float(w1v0), wf1_1 = __bfloat162float(w1v1);
            float wf2_0 = __bfloat162float(w2v0), wf2_1 = __bfloat162float(w2v1);
            float al0 = s_act[base_k + b * 2];
            float al1 = s_act[base_k + b * 2 + 1];
            acc1 += al0 * wf1_0 + al1 * wf1_1;
            acc2 += al0 * wf2_0 + al1 * wf2_1;
        }
    }

    __nv_bfloat16* out = C + (unsigned long long)flat_slot * N;
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
