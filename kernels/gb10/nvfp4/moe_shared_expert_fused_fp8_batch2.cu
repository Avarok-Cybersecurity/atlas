// SPDX-License-Identifier: AGPL-3.0-only

// Atlas Fused MoE Expert+Shared GEMV — K=2 multi-token batch, FP8 (E4M3) variant.
//
// Processes 2 tokens through MoE in single kernel launches by expanding
// blockIdx.y to accommodate 2 sets of (top_k routed + 1 shared) experts.
// FP8 weight format: 1 byte per weight + BF16 per-128x128-block scale.
//
// Token layout in blockIdx.y:
//   y in [0, 2*top_k)         -> routed experts (token = y/top_k, slot = y%top_k)
//   y in [2*top_k, 2*top_k+2) -> shared expert  (token = y - 2*top_k)
//
// Grid: gate_up_batch2  (ceil(N/8), 2*(top_k+1), 2)
//       silu_down_batch2 (ceil(N/8), 2*(top_k+1), 1)

#include <cuda_bf16.h>

#define BLOCK_SIZE 128
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define FP8_BLOCK 128

__device__ __constant__ float E4M3_LUT_MOE_BATCH2[256] = {
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

// ── Fused Gate+Up 2x with shared expert — K=2 batch, FP8 variant ──
//
// Grid: (ceil(N/8), 2*(top_k+1), 2)  Block: (128, 1, 1)
// blockIdx.y: 0..2*top_k-1 = routed (token=y/top_k, slot=y%top_k)
//             2*top_k..2*top_k+1 = shared (token=y-2*top_k)
extern "C" __global__ void moe_expert_gate_up_shared_fp8_batch2(
    const __nv_bfloat16* __restrict__ A,       // [2, H] BF16 input (2 tokens)
    // Routed expert tables (2 tables: weight + block_scale)
    const unsigned long long* __restrict__ gate_weight_ptrs,
    const unsigned long long* __restrict__ gate_block_scale_ptrs,
    __nv_bfloat16* __restrict__ gate_out,      // [2*top_k, inter] BF16
    const unsigned long long* __restrict__ up_weight_ptrs,
    const unsigned long long* __restrict__ up_block_scale_ptrs,
    __nv_bfloat16* __restrict__ up_out,        // [2*top_k, inter] BF16
    const unsigned int* __restrict__ expert_indices,  // [2*top_k] u32
    // Shared expert direct pointers
    const unsigned char* __restrict__ sh_gate_weight,
    const __nv_bfloat16* __restrict__ sh_gate_block_scale,
    __nv_bfloat16* __restrict__ sh_gate_out,   // [2, inter] BF16
    const unsigned char* __restrict__ sh_up_weight,
    const __nv_bfloat16* __restrict__ sh_up_block_scale,
    __nv_bfloat16* __restrict__ sh_up_out,     // [2, inter] BF16
    unsigned int N, unsigned int K, unsigned int top_k
) {
    const unsigned int total_routed = 2 * top_k;
    const unsigned int y = blockIdx.y;
    const unsigned int proj = blockIdx.z;
    const bool is_shared = (y >= total_routed);

    // Determine token index and expert slot
    unsigned int token, expert_slot;
    if (is_shared) {
        token = y - total_routed;  // 0 or 1
        expert_slot = 0;           // unused for shared
    } else {
        token = y / top_k;         // 0 or 1
        expert_slot = y % top_k;   // 0..top_k-1
    }

    // Select input for this token
    const __nv_bfloat16* A_token = A + (unsigned long long)token * K;

    const unsigned char* B_weight;
    const __nv_bfloat16* B_block_scale;
    __nv_bfloat16* C;

    if (is_shared) {
        if (proj == 0) {
            B_weight = sh_gate_weight; B_block_scale = sh_gate_block_scale;
            C = sh_gate_out + (unsigned long long)token * N;
        } else {
            B_weight = sh_up_weight; B_block_scale = sh_up_block_scale;
            C = sh_up_out + (unsigned long long)token * N;
        }
    } else {
        const unsigned int expert_id = expert_indices[token * top_k + expert_slot];
        const unsigned int flat_slot = token * top_k + expert_slot;
        if (proj == 0) {
            B_weight = (const unsigned char*)gate_weight_ptrs[expert_id];
            B_block_scale = (const __nv_bfloat16*)gate_block_scale_ptrs[expert_id];
            C = gate_out + (unsigned long long)flat_slot * N;
        } else {
            B_weight = (const unsigned char*)up_weight_ptrs[expert_id];
            B_block_scale = (const __nv_bfloat16*)up_block_scale_ptrs[expert_id];
            C = up_out + (unsigned long long)flat_slot * N;
        }
        // EP: NULL pointer means remote expert — write zero output and return
        if (B_weight == 0) {
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE) {
                C[n_base + i] = __float2bfloat16(0.0f);
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

    // Load E4M3 LUT into shared memory (256 entries, each thread loads 2)
    __shared__ float s_lut[256];
    s_lut[threadIdx.x] = E4M3_LUT_MOE_BATCH2[threadIdx.x];
    s_lut[threadIdx.x + BLOCK_SIZE] = E4M3_LUT_MOE_BATCH2[threadIdx.x + BLOCK_SIZE];
    __syncthreads();

    float acc1 = 0.0f, acc2 = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        uint4 a_data = ((const uint4*)A_token)[k8];
        const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
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
            unsigned int a32_lo = a_raw[b * 2];
            unsigned int a32_hi = a_raw[b * 2 + 1];

            float wf1_0 = s_lut[(w32_1      ) & 0xFF] * sc1;
            float wf1_1 = s_lut[(w32_1 >>  8) & 0xFF] * sc1;
            float wf1_2 = s_lut[(w32_1 >> 16) & 0xFF] * sc1;
            float wf1_3 = s_lut[(w32_1 >> 24) & 0xFF] * sc1;

            float wf2_0 = s_lut[(w32_2      ) & 0xFF] * sc2;
            float wf2_1 = s_lut[(w32_2 >>  8) & 0xFF] * sc2;
            float wf2_2 = s_lut[(w32_2 >> 16) & 0xFF] * sc2;
            float wf2_3 = s_lut[(w32_2 >> 24) & 0xFF] * sc2;

            __nv_bfloat16 a0, a1, a2, a3;
            *(unsigned short*)&a0 = (unsigned short)(a32_lo & 0xFFFF);
            *(unsigned short*)&a1 = (unsigned short)(a32_lo >> 16);
            *(unsigned short*)&a2 = (unsigned short)(a32_hi & 0xFFFF);
            *(unsigned short*)&a3 = (unsigned short)(a32_hi >> 16);
            float af0 = __bfloat162float(a0), af1 = __bfloat162float(a1);
            float af2 = __bfloat162float(a2), af3 = __bfloat162float(a3);

            acc1 += af0 * wf1_0 + af1 * wf1_1 + af2 * wf1_2 + af3 * wf1_3;
            acc2 += af0 * wf2_0 + af1 * wf2_1 + af2 * wf2_2 + af3 * wf2_3;
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

// ── Fused SiLU+Down 2x with shared expert — K=2 batch, FP8 variant ──
//
// Grid: (ceil(N/8), 2*(top_k+1), 1)  Block: (128, 1, 1)
extern "C" __global__ void moe_expert_silu_down_shared_fp8_batch2(
    const __nv_bfloat16* __restrict__ gate_out,  // [2*top_k, inter] BF16
    const __nv_bfloat16* __restrict__ up_out,    // [2*top_k, inter] BF16
    const unsigned long long* __restrict__ weight_ptrs,
    const unsigned long long* __restrict__ block_scale_ptrs,
    __nv_bfloat16* __restrict__ C,               // [2*top_k, H] BF16
    const unsigned int* __restrict__ expert_indices,  // [2*top_k] u32
    // Shared expert
    const __nv_bfloat16* __restrict__ sh_gate_in,  // [2, inter] BF16
    const __nv_bfloat16* __restrict__ sh_up_in,    // [2, inter] BF16
    const unsigned char* __restrict__ sh_down_weight,
    const __nv_bfloat16* __restrict__ sh_down_block_scale,
    __nv_bfloat16* __restrict__ sh_down_out,       // [2, H] BF16
    unsigned int N, unsigned int K, unsigned int top_k
) {
    const unsigned int total_routed = 2 * top_k;
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

    const unsigned char* B_weight;
    const __nv_bfloat16* B_block_scale;

    const __nv_bfloat16* g_ptr;
    const __nv_bfloat16* u_ptr;

    if (is_shared) {
        B_weight = sh_down_weight; B_block_scale = sh_down_block_scale;
        g_ptr = sh_gate_in + (unsigned long long)token * K;
        u_ptr = sh_up_in + (unsigned long long)token * K;
    } else {
        const unsigned int expert_id = expert_indices[token * top_k + expert_slot];
        const unsigned int flat_slot = token * top_k + expert_slot;
        B_weight = (const unsigned char*)weight_ptrs[expert_id];
        B_block_scale = (const __nv_bfloat16*)block_scale_ptrs[expert_id];
        g_ptr = gate_out + (unsigned long long)flat_slot * K;
        u_ptr = up_out + (unsigned long long)flat_slot * K;
        // EP: NULL pointer means remote expert — write zero output and return
        if (B_weight == 0) {
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE) {
                C[(unsigned long long)(token * top_k + expert_slot) * N + n_base + i] = __float2bfloat16(0.0f);
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

    __shared__ float s_lut[256];
    __shared__ float s_act[1024];

    // Load E4M3 LUT (256 entries, each thread loads 2)
    s_lut[threadIdx.x] = E4M3_LUT_MOE_BATCH2[threadIdx.x];
    s_lut[threadIdx.x + BLOCK_SIZE] = E4M3_LUT_MOE_BATCH2[threadIdx.x + BLOCK_SIZE];

    // Phase 1: Precompute SiLU(gate)*up into shared memory
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

    // Output: shared at sh_down_out[token*N], routed at C[flat_slot*N]
    __nv_bfloat16* out = is_shared
        ? (sh_down_out + (unsigned long long)token * N)
        : (C + (unsigned long long)(token * top_k + expert_slot) * N);

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

// ── Weighted sum + sigmoid blend — K=2 batch variant ──
//
// Identical to NVFP4 version (operates on BF16 intermediate results, no weight format dependency).
// Included for completeness — the blend kernel is format-agnostic.
//
// Grid: (ceil(hidden/256), 2, 1)  Block: (256, 1, 1)
extern "C" __global__ void moe_weighted_sum_blend_fp8_batch2(
    __nv_bfloat16* __restrict__ output,              // [2, hidden] BF16
    const __nv_bfloat16* __restrict__ expert_out,    // [2*top_k, hidden] BF16
    const float* __restrict__ expert_weights,         // [2*top_k] f32
    const __nv_bfloat16* __restrict__ shared_out,    // [2, hidden] BF16
    const __nv_bfloat16* __restrict__ input,         // [2, K] BF16 (MoE input)
    const __nv_bfloat16* __restrict__ gate_weight,   // [1, K] BF16 (shared gate)
    unsigned int hidden,
    unsigned int top_k,
    unsigned int K
) {
    const unsigned int token = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane = tid % WARP_SIZE;

    // Per-token input pointer
    const __nv_bfloat16* my_input = input + (unsigned long long)token * K;
    const float* my_weights = expert_weights + token * top_k;
    const __nv_bfloat16* my_expert_out = expert_out + (unsigned long long)token * top_k * hidden;
    const __nv_bfloat16* my_shared_out = shared_out + (unsigned long long)token * hidden;
    __nv_bfloat16* my_output = output + (unsigned long long)token * hidden;

    // ── Phase 1: Compute gate scalar (dot product + sigmoid) ──
    __shared__ float s_warp_sums[8];
    __shared__ float sigmoid_val;

    float dot_acc = 0.0f;
    unsigned int K8 = K / 8;
    for (unsigned int k8 = tid; k8 < K8; k8 += 256) {
        uint4 a_data = ((const uint4*)my_input)[k8];
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

    // Warp shuffle reduction
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        dot_acc += __shfl_down_sync(0xFFFFFFFF, dot_acc, offset);
    }
    if (lane == 0) {
        s_warp_sums[warp_id] = dot_acc;
    }
    __syncthreads();

    if (tid == 0) {
        float gate_scalar = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; w++) {
            gate_scalar += s_warp_sums[w];
        }
        sigmoid_val = 1.0f / (1.0f + __expf(-gate_scalar));
    }
    __syncthreads();

    // ── Phase 2: Weighted sum + blend ──
    unsigned int j = blockIdx.x * blockDim.x + tid;
    if (j >= hidden) return;

    float acc = 0.0f;
    for (unsigned int e = 0; e < top_k; e++) {
        acc += my_weights[e] * __bfloat162float(my_expert_out[(unsigned long long)e * hidden + j]);
    }
    acc += sigmoid_val * __bfloat162float(my_shared_out[j]);
    my_output[j] = __float2bfloat16(acc);
}
