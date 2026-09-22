// SPDX-License-Identifier: AGPL-3.0-only

// Atlas cross-row GROUPED fused MoE decode GEMV — FP8 (E4M3) weight variant.
//
// The batch2/batch3 siblings put the (token, slot) pair on blockIdx.y, so two
// rows routed to the same expert stream that expert's [N, K] weight twice. Here
// blockIdx.y indexes the COMPACTED list of ACTIVE experts (built from
// `expert_offsets` by `moe_fp8_grouped_compact`): a CTA streams each of its
// weight rows ONCE and applies them to every row routed to that expert (rows
// are grouped by expert with `moe_sort_by_expert`; `expert_offsets[e]..
// expert_offsets[e+1]` is the sorted-position range of expert e and
// `sorted_token_ids[pos]` its input row). The grid is sized to the FIXED cap
// `min(num_tokens*top_k, num_experts)` so a captured graph stays valid for
// every routing; CTAs past `active_count` exit at once. blockIdx.y == cap is
// the shared expert, which owns every row.
//
// Rows are processed in register passes of GROUP_ROWS; a pass holds one FP32
// accumulator pair per row, so an expert with more rows than GROUP_ROWS reads
// its weights ceil(rows / GROUP_ROWS) times — still never once per row.
//
// NUMERICS: the per-row arithmetic is copied from the single-token kernels
// this path replaces (moe_shared_expert_fused_fp8.cu): the gate/up loop keeps
// that kernel's 16-element lane partition and `acc += a0*w0 + a1*w1 + a2*w2 +
// a3*w3` grouping, silu/down keeps its 8-element partition, and the reduction
// is the same 32-lane butterfly, so each row's BF16 output is bit-identical to
// the per-token loop. (The batch2/3 gate/up kernels use an 8-element partition
// and therefore differ from BOTH in FP32 summation order.)
//
// Intermediates are laid out by SORTED position: gate_out/up_out/down C row
// `pos` belongs to `sorted_token_ids[pos]`; the blend maps a token's slot k
// back through `token_to_perm[token*top_k + k]`.
//
// Grid: compact   (1, 1, 1)                     Block (256,1,1)
//       gate_up   (ceil(N/8),  cap+1, 2)          Block (128,1,1)
//       silu_down (ceil(N/32), cap+1, 1)          Block (256,1,1)
//                 dynamic smem GROUP_ROWS*K*4 bytes
// silu_down owns 32 output columns per CTA (8 warps x 4): the 8-row
// SiLU(gate)*up activation block is computed ONCE per CTA into shared memory
// and read by every warp, so the activation traffic per output column is 4x
// lower than an 8-column tile, and it is read as float4 (an 8-float lane
// stride of scalar reads is 8-way bank conflicted).
// The blend lives in moe_fp8_grouped_blend.cu (module of the same name).

#include <cuda_bf16.h>

#define BLOCK_SIZE 128
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define FP8_BLOCK 128
#define GROUP_ROWS 8
#define DOWN_BLOCK 256
#define DOWN_COLS_PER_WARP 4
#define DOWN_COLS_PER_CTA ((DOWN_BLOCK / WARP_SIZE) * DOWN_COLS_PER_WARP)

__device__ __constant__ float E4M3_LUT_MOE_GROUPED[256] = {
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

// Rows [row0, row0+cnt) of this CTA's expert: the input row of local row r.
__device__ __forceinline__ unsigned int grouped_a_row(
    const int* __restrict__ sorted_token_ids, bool is_shared, unsigned int pos
) {
    return is_shared ? pos : (unsigned int)sorted_token_ids[pos];
}

// Compacted active-expert list: ascending expert ids with at least one row.
// Thread 0 walks the 256-entry offsets table; this is ~1 us and keeps the
// list order deterministic. `active_count[0]` is the list length.
extern "C" __global__ void moe_fp8_grouped_compact(
    const int* __restrict__ expert_offsets,   // [num_experts + 1]
    int* __restrict__ active_experts,         // [cap] (out)
    int* __restrict__ active_count,           // [1]   (out)
    unsigned int num_experts
) {
    if (threadIdx.x != 0) return;
    int n = 0;
    for (unsigned int e = 0; e < num_experts; e++) {
        if (expert_offsets[e + 1] > expert_offsets[e]) active_experts[n++] = (int)e;
    }
    active_count[0] = n;
}

extern "C" __global__ void moe_expert_gate_up_shared_fp8_grouped(
    const __nv_bfloat16* __restrict__ A,       // [num_tokens, K] BF16
    const unsigned long long* __restrict__ gate_weight_ptrs,
    const unsigned long long* __restrict__ gate_block_scale_ptrs,
    __nv_bfloat16* __restrict__ gate_out,      // [num_tokens*top_k, N] BF16, SORTED rows
    const unsigned long long* __restrict__ up_weight_ptrs,
    const unsigned long long* __restrict__ up_block_scale_ptrs,
    __nv_bfloat16* __restrict__ up_out,        // [num_tokens*top_k, N] BF16, SORTED rows
    const int* __restrict__ expert_offsets,    // [num_experts + 1]
    const int* __restrict__ sorted_token_ids,  // [num_tokens*top_k] -> input row
    const int* __restrict__ active_experts,    // [cap] compacted expert ids
    const int* __restrict__ active_count,      // [1]
    const unsigned char* __restrict__ sh_gate_weight,
    const float* __restrict__ sh_gate_block_scale,
    __nv_bfloat16* __restrict__ sh_gate_out,   // [num_tokens, N] BF16
    const unsigned char* __restrict__ sh_up_weight,
    const float* __restrict__ sh_up_block_scale,
    __nv_bfloat16* __restrict__ sh_up_out,     // [num_tokens, N] BF16
    unsigned int N, unsigned int K, unsigned int cap, unsigned int num_tokens
) {
    const unsigned int y = blockIdx.y;
    const unsigned int proj = blockIdx.z;
    const bool is_shared = (y == cap);

    unsigned int begin, end, expert = 0;
    if (is_shared) {
        begin = 0; end = num_tokens;
    } else {
        if ((int)y >= active_count[0]) return;
        expert = (unsigned int)active_experts[y];
        begin = (unsigned int)expert_offsets[expert];
        end = (unsigned int)expert_offsets[expert + 1];
    }
    if (begin >= end) return;

    const unsigned char* B_weight;
    const float* B_block_scale;
    __nv_bfloat16* C;
    if (is_shared) {
        if (proj == 0) { B_weight = sh_gate_weight; B_block_scale = sh_gate_block_scale; C = sh_gate_out; }
        else           { B_weight = sh_up_weight;   B_block_scale = sh_up_block_scale;   C = sh_up_out; }
    } else {
        if (proj == 0) {
            B_weight = (const unsigned char*)gate_weight_ptrs[expert];
            B_block_scale = (const float*)gate_block_scale_ptrs[expert];
            C = gate_out;
        } else {
            B_weight = (const unsigned char*)up_weight_ptrs[expert];
            B_block_scale = (const float*)up_block_scale_ptrs[expert];
            C = up_out;
        }
        // EP: NULL pointer means remote expert — zero every row of it and return.
        if (B_weight == 0) {
            const unsigned int n_base = blockIdx.x * (N_PER_BLOCK * 2);
            for (unsigned int pos = begin; pos < end; pos++) {
                for (unsigned int i = threadIdx.x; i < N_PER_BLOCK * 2 && n_base + i < N; i += BLOCK_SIZE) {
                    C[(unsigned long long)pos * N + n_base + i] = __float2bfloat16(0.0f);
                }
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

    __shared__ float s_lut[256];
    s_lut[threadIdx.x] = E4M3_LUT_MOE_GROUPED[threadIdx.x];
    s_lut[threadIdx.x + BLOCK_SIZE] = E4M3_LUT_MOE_GROUPED[threadIdx.x + BLOCK_SIZE];
    __syncthreads();

    for (unsigned int row0 = begin; row0 < end; row0 += GROUP_ROWS) {
        const unsigned int cnt = min((unsigned int)GROUP_ROWS, end - row0);
        const __nv_bfloat16* a_rows[GROUP_ROWS];
        #pragma unroll
        for (int r = 0; r < GROUP_ROWS; r++) {
            a_rows[r] = (r < (int)cnt)
                ? A + (unsigned long long)grouped_a_row(sorted_token_ids, is_shared, row0 + r) * K
                : A;
        }
        float acc1[GROUP_ROWS], acc2[GROUP_ROWS];
        #pragma unroll
        for (int r = 0; r < GROUP_ROWS; r++) { acc1[r] = 0.0f; acc2[r] = 0.0f; }

        for (unsigned int k16 = lane; k16 < K16; k16 += threads_per_out) {
            const unsigned int base_k = k16 * 16;
            const unsigned int k_block = base_k / FP8_BLOCK;
            float sc1 = B_block_scale[n1_block * k_blocks + k_block];
            float sc2 = have_n2 ? B_block_scale[n2_block * k_blocks + k_block] : 0.0f;

            // The weight rows are read ONCE per pass for every row of the expert.
            uint4 w_n1 = *(const uint4*)(B_weight + (unsigned long long)n1 * K + base_k);
            uint4 w_n2;
            if (have_n2) {
                w_n2 = *(const uint4*)(B_weight + (unsigned long long)n2 * K + base_k);
            } else {
                w_n2.x = 0; w_n2.y = 0; w_n2.z = 0; w_n2.w = 0;
            }
            const unsigned int w1[4] = {w_n1.x, w_n1.y, w_n1.z, w_n1.w};
            const unsigned int w2[4] = {w_n2.x, w_n2.y, w_n2.z, w_n2.w};

            #pragma unroll
            for (int r = 0; r < GROUP_ROWS; r++) {
                if (r < (int)cnt) {
                    uint4 a_data0 = ((const uint4*)a_rows[r])[k16 * 2];
                    uint4 a_data1 = ((const uint4*)a_rows[r])[k16 * 2 + 1];
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

                        acc1[r] += af0 * wf1_0 + af1 * wf1_1 + af2 * wf1_2 + af3 * wf1_3;
                        acc2[r] += af0 * wf2_0 + af1 * wf2_1 + af2 * wf2_2 + af3 * wf2_3;
                    }
                }
            }
        }

        #pragma unroll
        for (int r = 0; r < GROUP_ROWS; r++) {
            if (r < (int)cnt) {
                const unsigned long long base = (unsigned long long)(row0 + r) * N;
                float v1 = acc1[r];
                #pragma unroll
                for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                    v1 += __shfl_down_sync(0xFFFFFFFF, v1, offset);
                if (lane == 0) C[base + n1] = __float2bfloat16(v1);
                if (have_n2) {
                    float v2 = acc2[r];
                    #pragma unroll
                    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                        v2 += __shfl_down_sync(0xFFFFFFFF, v2, offset);
                    if (lane == 0) C[base + n2] = __float2bfloat16(v2);
                }
            }
        }
    }
}

extern "C" __global__ void moe_expert_silu_down_shared_fp8_grouped(
    const __nv_bfloat16* __restrict__ gate_out,  // [num_tokens*top_k, K] BF16, SORTED rows
    const __nv_bfloat16* __restrict__ up_out,    // [num_tokens*top_k, K] BF16, SORTED rows
    const unsigned long long* __restrict__ weight_ptrs,
    const unsigned long long* __restrict__ block_scale_ptrs,
    __nv_bfloat16* __restrict__ C,               // [num_tokens*top_k, N] BF16, SORTED rows
    const int* __restrict__ expert_offsets,      // [num_experts + 1]
    const int* __restrict__ active_experts,      // [cap] compacted expert ids
    const int* __restrict__ active_count,        // [1]
    const __nv_bfloat16* __restrict__ sh_gate_in,  // [num_tokens, K] BF16
    const __nv_bfloat16* __restrict__ sh_up_in,    // [num_tokens, K] BF16
    const unsigned char* __restrict__ sh_down_weight,
    const float* __restrict__ sh_down_block_scale,
    __nv_bfloat16* __restrict__ sh_down_out,       // [num_tokens, N] BF16
    unsigned int N, unsigned int K, unsigned int cap, unsigned int num_tokens
) {
    const unsigned int y = blockIdx.y;
    const bool is_shared = (y == cap);

    unsigned int begin, end, expert = 0;
    if (is_shared) {
        begin = 0; end = num_tokens;
    } else {
        if ((int)y >= active_count[0]) return;
        expert = (unsigned int)active_experts[y];
        begin = (unsigned int)expert_offsets[expert];
        end = (unsigned int)expert_offsets[expert + 1];
    }
    if (begin >= end) return;

    const unsigned char* B_weight;
    const float* B_block_scale;
    const __nv_bfloat16* g_base;
    const __nv_bfloat16* u_base;
    __nv_bfloat16* out_base;
    if (is_shared) {
        B_weight = sh_down_weight; B_block_scale = sh_down_block_scale;
        g_base = sh_gate_in; u_base = sh_up_in; out_base = sh_down_out;
    } else {
        B_weight = (const unsigned char*)weight_ptrs[expert];
        B_block_scale = (const float*)block_scale_ptrs[expert];
        g_base = gate_out; u_base = up_out; out_base = C;
        if (B_weight == 0) {
            const unsigned int n_base = blockIdx.x * DOWN_COLS_PER_CTA;
            for (unsigned int pos = begin; pos < end; pos++) {
                for (unsigned int i = threadIdx.x; i < DOWN_COLS_PER_CTA && n_base + i < N; i += DOWN_BLOCK) {
                    C[(unsigned long long)pos * N + n_base + i] = __float2bfloat16(0.0f);
                }
            }
            return;
        }
    }

    const unsigned int warp = threadIdx.x / WARP_SIZE;
    const unsigned int lane = threadIdx.x % WARP_SIZE;
    // This warp's 4 output columns; a column past N is computed on a zero
    // scale (loads clamped to column 0) and never written.
    const unsigned int n0 = blockIdx.x * DOWN_COLS_PER_CTA + warp * DOWN_COLS_PER_WARP;
    const bool active = (n0 < N);
    unsigned int ncol[DOWN_COLS_PER_WARP];
    bool have[DOWN_COLS_PER_WARP];
    #pragma unroll
    for (int c = 0; c < DOWN_COLS_PER_WARP; c++) {
        have[c] = (n0 + c < N);
        ncol[c] = have[c] ? n0 + c : 0;
    }

    const unsigned int K8 = K / 8;
    const unsigned int k_blocks = (K + FP8_BLOCK - 1) / FP8_BLOCK;

    __shared__ float s_lut[256];
    extern __shared__ float s_act[];  // [GROUP_ROWS, K]
    s_lut[threadIdx.x] = E4M3_LUT_MOE_GROUPED[threadIdx.x];

    for (unsigned int row0 = begin; row0 < end; row0 += GROUP_ROWS) {
        const unsigned int cnt = min((unsigned int)GROUP_ROWS, end - row0);
        __syncthreads();  // previous pass finished reading s_act
        // Phase 1: SiLU(gate)*up for every row of this pass, ONCE per CTA,
        // same expression as the single-token kernel.
        for (unsigned int idx = threadIdx.x; idx < cnt * K; idx += DOWN_BLOCK) {
            const unsigned int r = idx / K;
            const unsigned int i = idx - r * K;
            const unsigned long long row = (unsigned long long)(row0 + r) * K;
            float gf = __bfloat162float(g_base[row + i]);
            float uf = __bfloat162float(u_base[row + i]);
            s_act[idx] = (gf / (1.0f + __expf(-gf))) * uf;
        }
        __syncthreads();
        if (!active) continue;

        float acc[GROUP_ROWS][DOWN_COLS_PER_WARP];
        #pragma unroll
        for (int r = 0; r < GROUP_ROWS; r++)
            #pragma unroll
            for (int c = 0; c < DOWN_COLS_PER_WARP; c++) acc[r][c] = 0.0f;

        for (unsigned int k8 = lane; k8 < K8; k8 += WARP_SIZE) {
            const unsigned int base_k = k8 * 8;
            const unsigned int k_block = base_k / FP8_BLOCK;
            float sc[DOWN_COLS_PER_WARP];
            unsigned int wa[DOWN_COLS_PER_WARP], wb[DOWN_COLS_PER_WARP];
            #pragma unroll
            for (int c = 0; c < DOWN_COLS_PER_WARP; c++) {
                sc[c] = have[c] ? B_block_scale[(ncol[c] / FP8_BLOCK) * k_blocks + k_block] : 0.0f;
                const unsigned char* wrow = B_weight + (unsigned long long)ncol[c] * K + base_k;
                wa[c] = have[c] ? *(const unsigned int*)(wrow) : 0u;
                wb[c] = have[c] ? *(const unsigned int*)(wrow + 4) : 0u;
            }

            #pragma unroll
            for (int b = 0; b < 2; b++) {
                float wf[DOWN_COLS_PER_WARP][4];
                #pragma unroll
                for (int c = 0; c < DOWN_COLS_PER_WARP; c++) {
                    const unsigned int w32 = (b == 0) ? wa[c] : wb[c];
                    wf[c][0] = s_lut[(w32      ) & 0xFF] * sc[c];
                    wf[c][1] = s_lut[(w32 >>  8) & 0xFF] * sc[c];
                    wf[c][2] = s_lut[(w32 >> 16) & 0xFF] * sc[c];
                    wf[c][3] = s_lut[(w32 >> 24) & 0xFF] * sc[c];
                }
                #pragma unroll
                for (int r = 0; r < GROUP_ROWS; r++) {
                    if (r < (int)cnt) {
                        const float4 al = *(const float4*)(s_act + r * K + base_k + b * 4);
                        #pragma unroll
                        for (int c = 0; c < DOWN_COLS_PER_WARP; c++) {
                            acc[r][c] += al.x * wf[c][0] + al.y * wf[c][1] + al.z * wf[c][2] + al.w * wf[c][3];
                        }
                    }
                }
            }
        }

        #pragma unroll
        for (int r = 0; r < GROUP_ROWS; r++) {
            if (r < (int)cnt) {
                __nv_bfloat16* out = out_base + (unsigned long long)(row0 + r) * N;
                #pragma unroll
                for (int c = 0; c < DOWN_COLS_PER_WARP; c++) {
                    float v = acc[r][c];
                    #pragma unroll
                    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                        v += __shfl_down_sync(0xFFFFFFFF, v, offset);
                    if (lane == 0 && have[c]) out[ncol[c]] = __float2bfloat16(v);
                }
            }
        }
    }
}
