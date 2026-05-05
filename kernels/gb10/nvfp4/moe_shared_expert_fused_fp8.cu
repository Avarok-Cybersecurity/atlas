// SPDX-License-Identifier: AGPL-3.0-only

// Atlas Fused MoE Expert+Shared GEMV — FP8 (E4M3) weight variant.
//
// Same grid layout as moe_shared_expert_fused.cu but with FP8 weight format:
//   weight: [N, K] uint8 — one byte per weight (FP8 E4M3)
//   block_scale: [N/BS, K/BS] BF16 — per 128×128 block scale
//
// Grid: gate_up (ceil(N/8), top_k+1, 2),  silu_down (ceil(N/8), top_k+1, 1)

#include <cuda_bf16.h>

#define BLOCK_SIZE 128
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define FP8_BLOCK 128

// ── E4M3 Lookup Table ──────────────────────────────────────────────
//
// FP8 E4M3: sign(1) + exponent(4) + mantissa(3), bias=7
// 256 entries mapping every possible byte value to its f32 equivalent.
// Range: [-448, 448], NaN (0x7F/0xFF) mapped to 0.0.

__device__ __constant__ float E4M3_LUT_MOE_SHARED[256] = {
    // Positive (0x00..0x7F)
    0.0f, 0.001953125f, 0.00390625f, 0.005859375f,
    0.0078125f, 0.009765625f, 0.01171875f, 0.013671875f,
    0.015625f, 0.017578125f, 0.01953125f, 0.021484375f,
    0.0234375f, 0.025390625f, 0.02734375f, 0.029296875f,
    0.03125f, 0.03515625f, 0.0390625f, 0.04296875f,
    0.046875f, 0.05078125f, 0.0546875f, 0.05859375f,
    0.0625f, 0.0703125f, 0.078125f, 0.0859375f,
    0.09375f, 0.1015625f, 0.109375f, 0.1171875f,
    0.125f, 0.140625f, 0.15625f, 0.171875f,
    0.1875f, 0.203125f, 0.21875f, 0.234375f,
    0.25f, 0.28125f, 0.3125f, 0.34375f,
    0.375f, 0.40625f, 0.4375f, 0.46875f,
    0.5f, 0.5625f, 0.625f, 0.6875f,
    0.75f, 0.8125f, 0.875f, 0.9375f,
    1.0f, 1.125f, 1.25f, 1.375f,
    1.5f, 1.625f, 1.75f, 1.875f,
    2.0f, 2.25f, 2.5f, 2.75f,
    3.0f, 3.25f, 3.5f, 3.75f,
    4.0f, 4.5f, 5.0f, 5.5f,
    6.0f, 6.5f, 7.0f, 7.5f,
    8.0f, 9.0f, 10.0f, 11.0f,
    12.0f, 13.0f, 14.0f, 15.0f,
    16.0f, 18.0f, 20.0f, 22.0f,
    24.0f, 26.0f, 28.0f, 30.0f,
    32.0f, 36.0f, 40.0f, 44.0f,
    48.0f, 52.0f, 56.0f, 60.0f,
    64.0f, 72.0f, 80.0f, 88.0f,
    96.0f, 104.0f, 112.0f, 120.0f,
    128.0f, 144.0f, 160.0f, 176.0f,
    192.0f, 208.0f, 224.0f, 240.0f,
    256.0f, 288.0f, 320.0f, 352.0f,
    384.0f, 416.0f, 448.0f, 0.0f,
    // Negative (0x80..0xFF)
    -0.0f, -0.001953125f, -0.00390625f, -0.005859375f,
    -0.0078125f, -0.009765625f, -0.01171875f, -0.013671875f,
    -0.015625f, -0.017578125f, -0.01953125f, -0.021484375f,
    -0.0234375f, -0.025390625f, -0.02734375f, -0.029296875f,
    -0.03125f, -0.03515625f, -0.0390625f, -0.04296875f,
    -0.046875f, -0.05078125f, -0.0546875f, -0.05859375f,
    -0.0625f, -0.0703125f, -0.078125f, -0.0859375f,
    -0.09375f, -0.1015625f, -0.109375f, -0.1171875f,
    -0.125f, -0.140625f, -0.15625f, -0.171875f,
    -0.1875f, -0.203125f, -0.21875f, -0.234375f,
    -0.25f, -0.28125f, -0.3125f, -0.34375f,
    -0.375f, -0.40625f, -0.4375f, -0.46875f,
    -0.5f, -0.5625f, -0.625f, -0.6875f,
    -0.75f, -0.8125f, -0.875f, -0.9375f,
    -1.0f, -1.125f, -1.25f, -1.375f,
    -1.5f, -1.625f, -1.75f, -1.875f,
    -2.0f, -2.25f, -2.5f, -2.75f,
    -3.0f, -3.25f, -3.5f, -3.75f,
    -4.0f, -4.5f, -5.0f, -5.5f,
    -6.0f, -6.5f, -7.0f, -7.5f,
    -8.0f, -9.0f, -10.0f, -11.0f,
    -12.0f, -13.0f, -14.0f, -15.0f,
    -16.0f, -18.0f, -20.0f, -22.0f,
    -24.0f, -26.0f, -28.0f, -30.0f,
    -32.0f, -36.0f, -40.0f, -44.0f,
    -48.0f, -52.0f, -56.0f, -60.0f,
    -64.0f, -72.0f, -80.0f, -88.0f,
    -96.0f, -104.0f, -112.0f, -120.0f,
    -128.0f, -144.0f, -160.0f, -176.0f,
    -192.0f, -208.0f, -224.0f, -240.0f,
    -256.0f, -288.0f, -320.0f, -352.0f,
    -384.0f, -416.0f, -448.0f, -0.0f,
};

// ── Fused Gate+Up 2x with shared expert — FP8 variant ──
//
// blockIdx.y < top_k: routed expert (pointer table lookup)
// blockIdx.y == top_k: shared expert (direct weight pointers)
// Grid: (ceil(N/8), top_k+1, 2)  Block: (128, 1, 1)
extern "C" __global__ void moe_expert_gate_up_shared_fp8(
    const __nv_bfloat16* __restrict__ A,
    // Routed expert tables (2 tables: weight + block_scale)
    const unsigned long long* __restrict__ gate_weight_ptrs,
    const unsigned long long* __restrict__ gate_block_scale_ptrs,
    __nv_bfloat16* __restrict__ gate_out,
    const unsigned long long* __restrict__ up_weight_ptrs,
    const unsigned long long* __restrict__ up_block_scale_ptrs,
    __nv_bfloat16* __restrict__ up_out,
    const unsigned int* __restrict__ expert_indices,
    // Shared expert direct pointers
    const unsigned char* __restrict__ sh_gate_weight,
    const __nv_bfloat16* __restrict__ sh_gate_block_scale,
    __nv_bfloat16* __restrict__ sh_gate_out,
    const unsigned char* __restrict__ sh_up_weight,
    const __nv_bfloat16* __restrict__ sh_up_block_scale,
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
            B_weight = sh_gate_weight; B_block_scale = sh_gate_block_scale;
            C = sh_gate_out;
        } else {
            B_weight = sh_up_weight; B_block_scale = sh_up_block_scale;
            C = sh_up_out;
        }
    } else {
        const unsigned int expert_id = expert_indices[expert_slot];
        if (proj == 0) {
            B_weight = (const unsigned char*)gate_weight_ptrs[expert_id];
            B_block_scale = (const __nv_bfloat16*)gate_block_scale_ptrs[expert_id];
            C = gate_out;
        } else {
            B_weight = (const unsigned char*)up_weight_ptrs[expert_id];
            B_block_scale = (const __nv_bfloat16*)up_block_scale_ptrs[expert_id];
            C = up_out;
        }
        // EP: NULL pointer means remote expert — write zero output and return
        if (B_weight == 0) {
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

    const unsigned int K16 = K / 16;
    const unsigned int k_blocks = (K + FP8_BLOCK - 1) / FP8_BLOCK;
    const unsigned int n1_block = n1 / FP8_BLOCK;
    const unsigned int n2_block = n2 / FP8_BLOCK;

    // Load E4M3 LUT into shared memory (256 entries, each thread loads 2)
    __shared__ float s_lut[256];
    s_lut[threadIdx.x] = E4M3_LUT_MOE_SHARED[threadIdx.x];
    s_lut[threadIdx.x + BLOCK_SIZE] = E4M3_LUT_MOE_SHARED[threadIdx.x + BLOCK_SIZE];
    __syncthreads();

    float acc1 = 0.0f, acc2 = 0.0f;

    // Process 16 K-elements per iteration: uint4 weight load (16 FP8 bytes)
    // + 2× uint4 activation load (16 BF16 elements). Halves loop iterations
    // vs the K8 version and improves instruction-level parallelism.
    for (unsigned int k16 = lane; k16 < K16; k16 += threads_per_out) {
        const unsigned int base_k = k16 * 16;

        // Block scale for this K chunk
        const unsigned int k_block = base_k / FP8_BLOCK;
        float sc1 = __bfloat162float(B_block_scale[n1_block * k_blocks + k_block]);
        float sc2 = have_n2 ? __bfloat162float(B_block_scale[n2_block * k_blocks + k_block]) : 0.0f;

        // Load 16 BF16 activations as 2 × uint4
        uint4 a_data0 = ((const uint4*)A)[k16 * 2];
        uint4 a_data1 = ((const uint4*)A)[k16 * 2 + 1];

        // Load 16 FP8 weights for n1 as uint4 (128-bit coalesced read)
        uint4 w_n1 = *(const uint4*)(B_weight + (unsigned long long)n1 * K + base_k);
        // Load 16 FP8 weights for n2
        uint4 w_n2;
        if (have_n2) {
            w_n2 = *(const uint4*)(B_weight + (unsigned long long)n2 * K + base_k);
        } else {
            w_n2.x = 0; w_n2.y = 0; w_n2.z = 0; w_n2.w = 0;
        }

        // Process 16 elements: 4 groups of 4
        const unsigned int w1[4] = {w_n1.x, w_n1.y, w_n1.z, w_n1.w};
        const unsigned int w2[4] = {w_n2.x, w_n2.y, w_n2.z, w_n2.w};
        const unsigned int a0[4] = {a_data0.x, a_data0.y, a_data0.z, a_data0.w};
        const unsigned int a1[4] = {a_data1.x, a_data1.y, a_data1.z, a_data1.w};

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned int w32_1 = w1[b];
            unsigned int w32_2 = w2[b];
            unsigned int a32_lo = (b < 2) ? a0[b * 2] : a1[(b - 2) * 2];
            unsigned int a32_hi = (b < 2) ? a0[b * 2 + 1] : a1[(b - 2) * 2 + 1];

            float wf1_0 = s_lut[(w32_1      ) & 0xFF] * sc1;
            float wf1_1 = s_lut[(w32_1 >>  8) & 0xFF] * sc1;
            float wf1_2 = s_lut[(w32_1 >> 16) & 0xFF] * sc1;
            float wf1_3 = s_lut[(w32_1 >> 24) & 0xFF] * sc1;

            float wf2_0 = s_lut[(w32_2      ) & 0xFF] * sc2;
            float wf2_1 = s_lut[(w32_2 >>  8) & 0xFF] * sc2;
            float wf2_2 = s_lut[(w32_2 >> 16) & 0xFF] * sc2;
            float wf2_3 = s_lut[(w32_2 >> 24) & 0xFF] * sc2;

            __nv_bfloat16 av0, av1, av2, av3;
            *(unsigned short*)&av0 = (unsigned short)(a32_lo & 0xFFFF);
            *(unsigned short*)&av1 = (unsigned short)(a32_lo >> 16);
            *(unsigned short*)&av2 = (unsigned short)(a32_hi & 0xFFFF);
            *(unsigned short*)&av3 = (unsigned short)(a32_hi >> 16);
            float af0 = __bfloat162float(av0), af1 = __bfloat162float(av1);
            float af2 = __bfloat162float(av2), af3 = __bfloat162float(av3);

            acc1 += af0 * wf1_0 + af1 * wf1_1 + af2 * wf1_2 + af3 * wf1_3;
            acc2 += af0 * wf2_0 + af1 * wf2_1 + af2 * wf2_2 + af3 * wf2_3;
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

// ── Fused SiLU+Down 2x with shared expert — FP8 variant ──
//
// Precomputes SiLU(gate)*up in shared memory once per block, eliminating
// redundant SiLU compute across all 4 thread groups and replacing global
// gate/up loads with fast shared memory reads in the GEMV inner loop.
//
// blockIdx.y < top_k: routed expert (pointer table + expert_gate_out/up_out)
// blockIdx.y == top_k: shared expert (direct pointers + sh_gate_in/up_in)
// Grid: (ceil(N/8), top_k+1, 1)  Block: (128, 1, 1)
extern "C" __global__ void moe_expert_silu_down_shared_fp8(
    const __nv_bfloat16* __restrict__ gate_out,
    const __nv_bfloat16* __restrict__ up_out,
    const unsigned long long* __restrict__ weight_ptrs,
    const unsigned long long* __restrict__ block_scale_ptrs,
    __nv_bfloat16* __restrict__ C,
    const unsigned int* __restrict__ expert_indices,
    // Shared expert
    const __nv_bfloat16* __restrict__ sh_gate_in,
    const __nv_bfloat16* __restrict__ sh_up_in,
    const unsigned char* __restrict__ sh_down_weight,
    const __nv_bfloat16* __restrict__ sh_down_block_scale,
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
        B_weight = sh_down_weight; B_block_scale = sh_down_block_scale;
        g_ptr = sh_gate_in; u_ptr = sh_up_in;
    } else {
        const unsigned int expert_id = expert_indices[expert_slot];
        B_weight = (const unsigned char*)weight_ptrs[expert_id];
        B_block_scale = (const __nv_bfloat16*)block_scale_ptrs[expert_id];
        g_ptr = gate_out + (unsigned long long)expert_slot * K;
        u_ptr = up_out + (unsigned long long)expert_slot * K;
        // EP: NULL pointer means remote expert — write zero output and return
        if (B_weight == 0) {
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

    const unsigned int K8 = K / 8;
    const unsigned int k_blocks = (K + FP8_BLOCK - 1) / FP8_BLOCK;
    const unsigned int n1_block = n1 / FP8_BLOCK;
    const unsigned int n2_block = n2 / FP8_BLOCK;

    // Shared memory: E4M3 LUT + precomputed SiLU(gate)*up activation.
    __shared__ float s_lut[256];
    __shared__ float s_act[1024]; // max K=1024 (actual K=512)

    // Load E4M3 LUT (256 entries, each thread loads 2)
    s_lut[threadIdx.x] = E4M3_LUT_MOE_SHARED[threadIdx.x];
    s_lut[threadIdx.x + BLOCK_SIZE] = E4M3_LUT_MOE_SHARED[threadIdx.x + BLOCK_SIZE];

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

        // Block scale for this K chunk
        const unsigned int k_block = base_k / FP8_BLOCK;
        float sc1 = __bfloat162float(B_block_scale[n1_block * k_blocks + k_block]);
        float sc2 = have_n2 ? __bfloat162float(B_block_scale[n2_block * k_blocks + k_block]) : 0.0f;

        // Load 8 FP8 weights for n1
        unsigned int w4_1a = *(const unsigned int*)(B_weight + (unsigned long long)n1 * K + k8 * 8);
        unsigned int w4_1b = *(const unsigned int*)(B_weight + (unsigned long long)n1 * K + k8 * 8 + 4);

        // Load 8 FP8 weights for n2
        unsigned int w4_2a = have_n2 ?
            *(const unsigned int*)(B_weight + (unsigned long long)n2 * K + k8 * 8) : 0;
        unsigned int w4_2b = have_n2 ?
            *(const unsigned int*)(B_weight + (unsigned long long)n2 * K + k8 * 8 + 4) : 0;

        #pragma unroll
        for (int b = 0; b < 2; b++) {
            unsigned int w32_1 = (b == 0) ? w4_1a : w4_1b;
            unsigned int w32_2 = (b == 0) ? w4_2a : w4_2b;

            float al0 = s_act[base_k + b * 4];
            float al1 = s_act[base_k + b * 4 + 1];
            float al2 = s_act[base_k + b * 4 + 2];
            float al3 = s_act[base_k + b * 4 + 3];

            float wf1_0 = s_lut[(w32_1      ) & 0xFF] * sc1;
            float wf1_1 = s_lut[(w32_1 >>  8) & 0xFF] * sc1;
            float wf1_2 = s_lut[(w32_1 >> 16) & 0xFF] * sc1;
            float wf1_3 = s_lut[(w32_1 >> 24) & 0xFF] * sc1;

            float wf2_0 = s_lut[(w32_2      ) & 0xFF] * sc2;
            float wf2_1 = s_lut[(w32_2 >>  8) & 0xFF] * sc2;
            float wf2_2 = s_lut[(w32_2 >> 16) & 0xFF] * sc2;
            float wf2_3 = s_lut[(w32_2 >> 24) & 0xFF] * sc2;

            acc1 += al0 * wf1_0 + al1 * wf1_1 + al2 * wf1_2 + al3 * wf1_3;
            acc2 += al0 * wf2_0 + al1 * wf2_1 + al2 * wf2_2 + al3 * wf2_3;
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
