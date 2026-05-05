// SPDX-License-Identifier: AGPL-3.0-only

// Atlas FP8 Grouped MoE GEMM — Sorted expert dispatch with FP8 E4M3 block-scaled weights.
//
// C[M_expert,N] = A[M_expert,K] (BF16) @ dequant(B_expert[N,K] (FP8 E4M3))
//
// Each CTA processes one (expert, m_tile, n_tile) block. Expert weights are
// accessed via pointer tables indexed by expert_id. Tokens are sorted by expert
// so each expert's tokens are contiguous.
//
// FP8 weight format: B[N,K] uint8 with block_scale[N/128, K/128] BF16.
// Dequant: bf16_val = E4M3_LUT[byte] * block_scale[n/128, k/128]
//
// Grid: (ceil(N/64), max_m_tiles, num_experts)  Block: (128, 1, 1)

#include <cuda_bf16.h>

#define M_TILE 64
#define N_TILE 64
#define K_STEP 16
#define PAD 2
#define FP8_BLOCK 128

__device__ __constant__ float E4M3_LUT_GMOE[256] = {
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

// MMA compute (shared with other GEMM kernels)
__device__ __forceinline__ void fp8_moe_mma(
    __nv_bfloat16 smem_A[][K_STEP + PAD],
    __nv_bfloat16 smem_B[][N_TILE + PAD],
    float acc[8][4],
    unsigned int warp_m_offset, unsigned int group_id, unsigned int tid
) {
    const unsigned int a_stride = K_STEP + PAD;
    const unsigned int b_stride = N_TILE + PAD;
    const unsigned short* sA = (const unsigned short*)smem_A;
    const unsigned short* sB = (const unsigned short*)smem_B;

    unsigned int frag_r0 = warp_m_offset + group_id;
    unsigned int frag_r1 = warp_m_offset + group_id + 8;
    unsigned int frag_c0 = tid * 2;
    unsigned int frag_c1 = tid * 2 + 8;

    unsigned int a0 = *(const unsigned int*)&sA[frag_r0 * a_stride + frag_c0];
    unsigned int a1 = *(const unsigned int*)&sA[frag_r1 * a_stride + frag_c0];
    unsigned int a2 = *(const unsigned int*)&sA[frag_r0 * a_stride + frag_c1];
    unsigned int a3 = *(const unsigned int*)&sA[frag_r1 * a_stride + frag_c1];

    #pragma unroll
    for (int n_tile = 0; n_tile < 8; n_tile++) {
        unsigned int n_col = n_tile * 8 + group_id;
        unsigned int k0 = tid * 2;
        unsigned int k1 = tid * 2 + 8;

        unsigned int b0 = ((unsigned int)sB[(k0 + 1) * b_stride + n_col] << 16) |
                          (unsigned int)sB[k0 * b_stride + n_col];
        unsigned int b1 = ((unsigned int)sB[(k1 + 1) * b_stride + n_col] << 16) |
                          (unsigned int)sB[k1 * b_stride + n_col];

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%10, %11, %12, %13};"
            : "=f"(acc[n_tile][0]), "=f"(acc[n_tile][1]),
              "=f"(acc[n_tile][2]), "=f"(acc[n_tile][3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1),
              "f"(acc[n_tile][0]), "f"(acc[n_tile][1]),
              "f"(acc[n_tile][2]), "f"(acc[n_tile][3])
        );
    }
}

/// FP8 grouped GEMM for sorted MoE dispatch.
///
/// BF16 activations × FP8 E4M3 block-scaled weights per expert.
/// Expert weights accessed via pointer tables. Tokens sorted by expert.
///
/// Grid: (ceil(N/64), max_m_tiles, num_experts)  Block: (128, 1, 1)
extern "C" __global__ void moe_fp8_grouped_gemm(
    const __nv_bfloat16* __restrict__ A,                   // [total_tokens, K] BF16
    const unsigned long long* __restrict__ B_weight_ptrs,   // [num_experts] → [N, K] FP8
    const unsigned long long* __restrict__ B_scale_ptrs,    // [num_experts] → [N/128, K/128] BF16
    __nv_bfloat16* __restrict__ C,                         // [total_expanded, N] BF16
    const int* __restrict__ expert_offsets,                 // [num_experts + 1]
    const int* __restrict__ sorted_token_ids,              // [total_expanded]
    unsigned int num_experts,
    unsigned int N,
    unsigned int K
) {
    const unsigned int expert_id = blockIdx.z;
    if (expert_id >= num_experts) return;

    const int m_start = expert_offsets[expert_id];
    const int m_end = expert_offsets[expert_id + 1];
    const int M_expert = m_end - m_start;
    if (M_expert <= 0) return;

    const int cta_m_local = blockIdx.y * M_TILE;
    if (cta_m_local >= M_expert) return;

    const unsigned int cta_n = blockIdx.x * N_TILE;

    // Load expert weight/scale pointers from table
    const unsigned char* B_exp = (const unsigned char*)B_weight_ptrs[expert_id];
    const __nv_bfloat16* S_exp = (const __nv_bfloat16*)B_scale_ptrs[expert_id];
    if (B_exp == 0) return;  // NULL → remote expert under EP

    const unsigned int k_blocks = (K + FP8_BLOCK - 1) / FP8_BLOCK;

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[M_TILE][K_STEP + PAD];
    __shared__ __nv_bfloat16 smem_B[K_STEP][N_TILE + PAD];

    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    for (unsigned int k_base = 0; k_base < K; k_base += K_STEP) {
        // Load A tile: gather from sorted token positions
        {
            #pragma unroll
            for (unsigned int i = 0; i < 8; i++) {
                unsigned int idx = threadIdx.x * 8 + i;
                unsigned int row = idx / K_STEP;
                unsigned int col = idx % K_STEP;
                unsigned int m_idx = cta_m_local + row;
                unsigned int gc = k_base + col;

                if (m_idx < (unsigned int)M_expert && gc < K) {
                    int sorted_idx = m_start + m_idx;
                    // If sorted_token_ids is NULL, use direct indexing (already sorted input)
                    int token_id = sorted_token_ids ? sorted_token_ids[sorted_idx] : sorted_idx;
                    smem_A[row][col] = A[(unsigned long long)token_id * K + gc];
                } else {
                    smem_A[row][col] = __float2bfloat16(0.0f);
                }
            }
        }

        // Dequant B tile: FP8 E4M3 → BF16 via LUT × block_scale
        {
            #pragma unroll
            for (unsigned int i = 0; i < 8; i++) {
                unsigned int idx = threadIdx.x * 8 + i;
                unsigned int k = idx / N_TILE;
                unsigned int n = idx % N_TILE;
                unsigned int gk = k_base + k;
                unsigned int gn = cta_n + n;

                if (gk < K && gn < N) {
                    unsigned char weight_byte = B_exp[(unsigned long long)gn * K + gk];
                    unsigned int n_block = gn / FP8_BLOCK;
                    unsigned int k_block = gk / FP8_BLOCK;
                    float scale = __bfloat162float(S_exp[n_block * k_blocks + k_block]);
                    float dequant_val = E4M3_LUT_GMOE[weight_byte] * scale;
                    smem_B[k][n] = __float2bfloat16(dequant_val);
                } else {
                    smem_B[k][n] = __float2bfloat16(0.0f);
                }
            }
        }

        __syncthreads();
        fp8_moe_mma(smem_A, smem_B, acc, warp_m_offset, group_id, tid);
        __syncthreads();
    }

    // Store C tile — write to sorted position in output
    #pragma unroll
    for (int n_tile = 0; n_tile < 8; n_tile++) {
        unsigned int base_n = cta_n + n_tile * 8;
        unsigned int col0 = base_n + (tid * 2);
        unsigned int col1 = col0 + 1;
        unsigned int row0 = cta_m_local + warp_m_offset + group_id;
        unsigned int row1 = row0 + 8;

        if (row0 < (unsigned int)M_expert) {
            unsigned int out_row = m_start + row0;
            if (col0 < N) C[(unsigned long long)out_row * N + col0] = __float2bfloat16(acc[n_tile][0]);
            if (col1 < N) C[(unsigned long long)out_row * N + col1] = __float2bfloat16(acc[n_tile][1]);
        }
        if (row1 < (unsigned int)M_expert) {
            unsigned int out_row = m_start + row1;
            if (col0 < N) C[(unsigned long long)out_row * N + col0] = __float2bfloat16(acc[n_tile][2]);
            if (col1 < N) C[(unsigned long long)out_row * N + col1] = __float2bfloat16(acc[n_tile][3]);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Coalesced-load variant — same MMA logic, new thread mapping on A/B smem
// staging so neighbouring threads in a warp hit contiguous global memory.
// ═══════════════════════════════════════════════════════════════════
//
// Bug in v1 (moe_fp8_grouped_gemm): the thread-to-smem mapping
//     unsigned int idx = threadIdx.x * 8 + i;
//     unsigned int k   = idx / N_TILE;
//     unsigned int n   = idx % N_TILE;
// causes within-warp strided access when loading B[gn * K + gk] — 32
// threads of a warp land on 32 different rows of B (stride K bytes),
// each hitting a distinct cache line. On GB10 that shows up as ~8×
// lower memory throughput than the LPDDR5X ceiling on MoE FFN prefill.
//
// v2 layout: 8 thread-groups × 16 threads. Each group owns 8 rows of the
// K_STEP × N_TILE tile (spread across the 8 `i` iterations). Within a
// group, 16 threads load 16 contiguous K values for one (row, k_base)
// position — exactly one coalesced 16-byte transaction per load per
// group. 2 cache-line fills per warp instruction instead of 32.
//
// Same output semantics as v1: smem cells are written with identical
// data, just by different threads. MMA stage reads smem_A / smem_B
// after __syncthreads() so the producer→consumer thread mapping does
// not need to match.
//
// Gated at the Rust dispatch layer via ATLAS_FP8_MOE_COALESCED=1 to
// preserve the v1 codepath as the validated default.
extern "C" __global__ void moe_fp8_grouped_gemm_v2(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ B_weight_ptrs,
    const unsigned long long* __restrict__ B_scale_ptrs,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    unsigned int num_experts,
    unsigned int N,
    unsigned int K
) {
    const unsigned int expert_id = blockIdx.z;
    if (expert_id >= num_experts) return;

    const int m_start = expert_offsets[expert_id];
    const int m_end = expert_offsets[expert_id + 1];
    const int M_expert = m_end - m_start;
    if (M_expert <= 0) return;

    const int cta_m_local = blockIdx.y * M_TILE;
    if (cta_m_local >= M_expert) return;

    const unsigned int cta_n = blockIdx.x * N_TILE;

    const unsigned char* B_exp = (const unsigned char*)B_weight_ptrs[expert_id];
    const __nv_bfloat16* S_exp = (const __nv_bfloat16*)B_scale_ptrs[expert_id];
    if (B_exp == 0) return;

    const unsigned int k_blocks = (K + FP8_BLOCK - 1) / FP8_BLOCK;

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[M_TILE][K_STEP + PAD];
    __shared__ __nv_bfloat16 smem_B[K_STEP][N_TILE + PAD];

    // Coalesced mapping: 8 groups × 16 threads.
    // thread_group = threadIdx.x / 16, k_offset = threadIdx.x % 16.
    // Each group owns 8 rows (A) / 8 columns (B) across the 8 `i` iters.
    const unsigned int thread_group = threadIdx.x >> 4;      // 0..7
    const unsigned int k_offset     = threadIdx.x & 15;      // 0..K_STEP-1 (K_STEP=16)
    const unsigned int row_base     = thread_group * 8;      // 0, 8, 16, ..., 56

    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    for (unsigned int k_base = 0; k_base < K; k_base += K_STEP) {
        const unsigned int gk = k_base + k_offset;

        // Load A tile [M_TILE=64 rows][K_STEP=16 cols] from global into smem.
        // smem_A is indexed by LOCAL row (0..M_TILE-1); the expert-relative
        // offset adds cta_m_local, and the expert-bounds check uses that.
        // Within a warp, 16 threads of the same thread_group share one m-row
        // and vary k_offset 0..15 — 16 contiguous BF16 elements from the
        // same A row (tokenised, so different warps may hit different pages
        // via sorted_token_ids but a single warp's 16-thread group is one
        // coalesced 32-byte burst).
        #pragma unroll
        for (unsigned int i = 0; i < 8; i++) {
            unsigned int local_row  = row_base + i;
            unsigned int m_global   = cta_m_local + local_row;
            if (m_global < (unsigned int)M_expert && gk < K) {
                int sorted_idx = m_start + m_global;
                int token_id = sorted_token_ids ? sorted_token_ids[sorted_idx] : sorted_idx;
                smem_A[local_row][k_offset] = A[(unsigned long long)token_id * K + gk];
            } else {
                smem_A[local_row][k_offset] = __float2bfloat16(0.0f);
            }
        }

        // Dequant B tile [K_STEP=16 rows][N_TILE=64 cols].
        // smem_B[k_offset][n_local] where n_local = row_base + i is the
        // column within the N_TILE. Global B addr = B_exp[gn * K + gk]
        // with gn = cta_n + n_local. Within a warp, 16 threads of the same
        // thread_group share one (gn, k_base) row and vary gk 0..15 →
        // one coalesced 16-byte load per group.
        #pragma unroll
        for (unsigned int i = 0; i < 8; i++) {
            unsigned int n_local = row_base + i;
            unsigned int gn = cta_n + n_local;
            if (gk < K && gn < N) {
                unsigned char weight_byte = B_exp[(unsigned long long)gn * K + gk];
                unsigned int n_block = gn / FP8_BLOCK;
                unsigned int k_block = gk / FP8_BLOCK;
                float scale = __bfloat162float(S_exp[n_block * k_blocks + k_block]);
                float dequant_val = E4M3_LUT_GMOE[weight_byte] * scale;
                smem_B[k_offset][n_local] = __float2bfloat16(dequant_val);
            } else {
                smem_B[k_offset][n_local] = __float2bfloat16(0.0f);
            }
        }

        __syncthreads();
        fp8_moe_mma(smem_A, smem_B, acc, warp_m_offset, group_id, tid);
        __syncthreads();
    }

    // Store C tile — write to sorted position in output. Identical to v1.
    #pragma unroll
    for (int n_tile = 0; n_tile < 8; n_tile++) {
        unsigned int base_n = cta_n + n_tile * 8;
        unsigned int col0 = base_n + (tid * 2);
        unsigned int col1 = col0 + 1;
        unsigned int row0 = cta_m_local + warp_m_offset + group_id;
        unsigned int row1 = row0 + 8;

        if (row0 < (unsigned int)M_expert) {
            unsigned int out_row = m_start + row0;
            if (col0 < N) C[(unsigned long long)out_row * N + col0] = __float2bfloat16(acc[n_tile][0]);
            if (col1 < N) C[(unsigned long long)out_row * N + col1] = __float2bfloat16(acc[n_tile][1]);
        }
        if (row1 < (unsigned int)M_expert) {
            unsigned int out_row = m_start + row1;
            if (col0 < N) C[(unsigned long long)out_row * N + col0] = __float2bfloat16(acc[n_tile][2]);
            if (col1 < N) C[(unsigned long long)out_row * N + col1] = __float2bfloat16(acc[n_tile][3]);
        }
    }
}
