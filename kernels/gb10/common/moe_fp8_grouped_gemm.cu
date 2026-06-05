// SPDX-License-Identifier: AGPL-3.0-only

// Atlas FP8 Grouped MoE GEMM — Sorted expert dispatch with FP8 E4M3 block-scaled weights.
//
// C[M_expert,N] = A[M_expert,K] (BF16) @ dequant(B_expert[N,K] (FP8 E4M3))
//
// Each CTA processes one (expert, m_tile, n_tile) block. Expert weights are
// accessed via pointer tables indexed by expert_id. Tokens are sorted by expert
// so each expert's tokens are contiguous.
//
// FP8 weight format: B[N,K] uint8 with block_scale[N/128, K/128] FP32.
// (scale_inv is widened to FP32 at load; applied in full FP32 precision.)
// Dequant: bf16_val = E4M3_LUT[byte] * block_scale[n/128, k/128]
//
// Numerics SSOT (Phase 2b, 2026-05-24): all f32 -> BF16 conversions in
// this file use `__float2bfloat16(x)`, which on sm_80+ lowers to
// `cvt.rn.bf16.f32` (round-to-nearest-even). This matches the
// load-time CPU dequant in `weight_map::fp8_lut::f32_to_bf16` and
// `atlas_quant::fp8::f32_to_bf16`, so the routed-expert kernel-side
// dequant agrees byte-exact with the shared-expert load-time dequant
// AND with PyTorch's `torch.float32 -> torch.bfloat16` reference.
//
// Grid: (ceil(N/64), max_m_tiles, num_experts)  Block: (128, 1, 1)

#include <cuda_bf16.h>

#define M_TILE 64
#define N_TILE 64
#define K_STEP 16
#define PAD 2
#define FP8_BLOCK 128
// Inner-promotion stride. Smaller than FP8_BLOCK applies the same scale at
// finer granularity within a scale-block — mathematically identical for
// FP32 accumulators ((a+b)*s == a*s + b*s), but matches DeepGEMM's
// "4 WGMMA per promote" structure and exposes any subtle two-level path
// bugs more frequently. Must divide FP8_BLOCK evenly.
#define K_PROMOTE 64

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
    const unsigned long long* __restrict__ B_scale_ptrs,    // [num_experts] → [N/128, K/128] FP32
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

    // Load expert weight/scale pointers from table.
    // mantissa noise from the previous BF16-cast-at-use path.
    const unsigned char* B_exp = (const unsigned char*)B_weight_ptrs[expert_id];
    const float* S_exp = (const float*)B_scale_ptrs[expert_id];
    if (B_exp == 0) return;  // NULL → remote expert under EP

    const unsigned int k_blocks = (K + FP8_BLOCK - 1) / FP8_BLOCK;

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[M_TILE][K_STEP + PAD];
    __shared__ __nv_bfloat16 smem_B[K_STEP][N_TILE + PAD];

    // Two-level FP32 accumulation (DeepGEMM pattern, A5/A10 2026-05-25):
    //   inner_acc accumulates unscaled BF16×BF16 products within one
    //     scale-block (K=128). E4M3_LUT[byte] cast to BF16 is LOSSLESS
    //     because FP8 E4M3 has only 3 mantissa bits and BF16 has 7.
    //   outer_acc accumulates scale * inner_acc per K-block.
    // Decouples the BF16-scale-truncation from per-element multiplication:
    // previously every product was `BF16(LUT * scale)` so the scale's
    // 7-bit mantissa truncated each dequanted weight. Now the scale is
    // applied ONCE to the FP32-accumulated K=128 sum, preserving the
    // full LUT precision through the inner reduction.
    float outer_acc[8][4];
    float inner_acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        outer_acc[i][0] = 0.0f; outer_acc[i][1] = 0.0f;
        outer_acc[i][2] = 0.0f; outer_acc[i][3] = 0.0f;
        inner_acc[i][0] = 0.0f; inner_acc[i][1] = 0.0f;
        inner_acc[i][2] = 0.0f; inner_acc[i][3] = 0.0f;
    }

    // N_TILE=64 < FP8_BLOCK=128, cta_n always aligned to N_TILE — so all
    // 64 N-cols of this CTA fall within a single N-block and share one
    // scale per K-block.
    const unsigned int n_block = cta_n / FP8_BLOCK;

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

        // Dequant B tile: FP8 E4M3 → BF16 via LUT (NO scale — applied
        // post-MMA to inner_acc). Lossless because FP8 has 3-bit
        // mantissa, BF16 has 7-bit mantissa.
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
                    smem_B[k][n] = __float2bfloat16(E4M3_LUT_GMOE[weight_byte]);
                } else {
                    smem_B[k][n] = __float2bfloat16(0.0f);
                }
            }
        }

        __syncthreads();
        fp8_moe_mma(smem_A, smem_B, inner_acc, warp_m_offset, group_id, tid);
        __syncthreads();

        // End-of-K-block: scale inner_acc, accumulate to outer_acc, reset.
        unsigned int next_k = k_base + K_STEP;
        if (next_k % K_PROMOTE == 0 || next_k >= K) {
            unsigned int k_block = k_base / FP8_BLOCK;
            float scale = S_exp[n_block * k_blocks + k_block];
            #pragma unroll
            for (int n_tile = 0; n_tile < 8; n_tile++) {
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    outer_acc[n_tile][j] += inner_acc[n_tile][j] * scale;
                    inner_acc[n_tile][j] = 0.0f;
                }
            }
        }
    }

    // Store C tile — write to sorted position in output (from outer_acc)
    #pragma unroll
    for (int n_tile = 0; n_tile < 8; n_tile++) {
        unsigned int base_n = cta_n + n_tile * 8;
        unsigned int col0 = base_n + (tid * 2);
        unsigned int col1 = col0 + 1;
        unsigned int row0 = cta_m_local + warp_m_offset + group_id;
        unsigned int row1 = row0 + 8;

        if (row0 < (unsigned int)M_expert) {
            unsigned int out_row = m_start + row0;
            if (col0 < N) C[(unsigned long long)out_row * N + col0] = __float2bfloat16(outer_acc[n_tile][0]);
            if (col1 < N) C[(unsigned long long)out_row * N + col1] = __float2bfloat16(outer_acc[n_tile][1]);
        }
        if (row1 < (unsigned int)M_expert) {
            unsigned int out_row = m_start + row1;
            if (col0 < N) C[(unsigned long long)out_row * N + col0] = __float2bfloat16(outer_acc[n_tile][2]);
            if (col1 < N) C[(unsigned long long)out_row * N + col1] = __float2bfloat16(outer_acc[n_tile][3]);
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
    const float* S_exp = (const float*)B_scale_ptrs[expert_id];
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

    // Two-level FP32 accumulation (DeepGEMM pattern, see v1 above for
    // rationale): inner accumulates unscaled BF16(LUT) products within
    // one K=128 scale-block, outer applies scale per block.
    float outer_acc[8][4];
    float inner_acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        outer_acc[i][0] = 0.0f; outer_acc[i][1] = 0.0f;
        outer_acc[i][2] = 0.0f; outer_acc[i][3] = 0.0f;
        inner_acc[i][0] = 0.0f; inner_acc[i][1] = 0.0f;
        inner_acc[i][2] = 0.0f; inner_acc[i][3] = 0.0f;
    }

    const unsigned int n_block = cta_n / FP8_BLOCK;

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

        // Dequant B tile [K_STEP=16 rows][N_TILE=64 cols] — NO scale (applied
        // post-MMA per K-block to inner_acc; see v1 above for rationale).
        #pragma unroll
        for (unsigned int i = 0; i < 8; i++) {
            unsigned int n_local = row_base + i;
            unsigned int gn = cta_n + n_local;
            if (gk < K && gn < N) {
                unsigned char weight_byte = B_exp[(unsigned long long)gn * K + gk];
                smem_B[k_offset][n_local] = __float2bfloat16(E4M3_LUT_GMOE[weight_byte]);
            } else {
                smem_B[k_offset][n_local] = __float2bfloat16(0.0f);
            }
        }

        __syncthreads();
        fp8_moe_mma(smem_A, smem_B, inner_acc, warp_m_offset, group_id, tid);
        __syncthreads();

        // End-of-K-block: scale inner_acc → outer_acc, reset inner.
        unsigned int next_k = k_base + K_STEP;
        if (next_k % K_PROMOTE == 0 || next_k >= K) {
            unsigned int k_block = k_base / FP8_BLOCK;
            float scale = S_exp[n_block * k_blocks + k_block];
            #pragma unroll
            for (int n_tile = 0; n_tile < 8; n_tile++) {
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    outer_acc[n_tile][j] += inner_acc[n_tile][j] * scale;
                    inner_acc[n_tile][j] = 0.0f;
                }
            }
        }
    }

    // Store C tile from outer_acc (DeepGEMM-style scaled FP32 sum).
    #pragma unroll
    for (int n_tile = 0; n_tile < 8; n_tile++) {
        unsigned int base_n = cta_n + n_tile * 8;
        unsigned int col0 = base_n + (tid * 2);
        unsigned int col1 = col0 + 1;
        unsigned int row0 = cta_m_local + warp_m_offset + group_id;
        unsigned int row1 = row0 + 8;

        if (row0 < (unsigned int)M_expert) {
            unsigned int out_row = m_start + row0;
            if (col0 < N) C[(unsigned long long)out_row * N + col0] = __float2bfloat16(outer_acc[n_tile][0]);
            if (col1 < N) C[(unsigned long long)out_row * N + col1] = __float2bfloat16(outer_acc[n_tile][1]);
        }
        if (row1 < (unsigned int)M_expert) {
            unsigned int out_row = m_start + row1;
            if (col0 < N) C[(unsigned long long)out_row * N + col0] = __float2bfloat16(outer_acc[n_tile][2]);
            if (col1 < N) C[(unsigned long long)out_row * N + col1] = __float2bfloat16(outer_acc[n_tile][3]);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// v3 — Fix-B occupancy + cp.async rewrite (sm_121/GB10).
// ═══════════════════════════════════════════════════════════════════
//
// Same math + number formats as v1/v2 (BF16 acts × FP8 E4M3 block-scaled
// weights, grouped per-expert dispatch, two-level FP32 accumulation PRESERVED
// EXACTLY). Structure ported from the Fix-A w8a16_gemm_pipelined rewrite, but
// the perf sweep below diverges from Fix-A's "occupancy is everything" result.
//
// v2 baseline: M_TILE=64 / N_TILE=64, 128 threads, [8][4]×2 = 64 accumulator
// regs → 132 regs/thread → 3 CTAs/SM (25% occupancy on a 1536-thread SM).
// Scalar smem loads through __constant__ LUT, NO cp.async, NO prefetch.
// Measured 5.0 TFLOP/s at the representative 8×256×N2048×K2048 size.
//
// v3: 128×64 (M×N) tile, 8 warps (256 threads), [8][8]/[4] inner+outer
// accumulators. PM3_STAGES-deep cp.async.cg software pipeline (this kernel
// previously had NONE): A-tile gathered per sorted token-id row, raw FP8 B
// bytes prefetched as contiguous 16-B K-runs, then cooperatively dequantized
// into [k][n] BF16 via a SHARED-MEMORY LUT.
//
// Three levers, measured kernel-only (CUDA events) at 8×256×2048×2048:
//   1. OCCUPANCY (Fix-A's lever): N_TILE 64→32 + 256 threads dropped the
//      accumulator to 64 regs → 4 CTAs/SM (66.7%). 5.0 → 12.6 TFLOP/s (2.5×).
//      cp.async included here but contributed little on its own (this kernel,
//      like Fix-A, is NOT global-load-latency-bound: STAGES 2 vs 3 is noise).
//   2. SHARED-MEMORY LUT: the dequant indexes the E4M3 table with DATA-
//      DEPENDENT bytes; in __constant__ memory those divergent lookups
//      SERIALIZE (constant cache broadcasts only on uniform addresses) and
//      dominated the inner loop. A 1 KB smem copy (byte-identical values, so
//      numerics preserved) → 12.6 → 18.0 TFLOP/s (+43%). THE biggest lever.
//   3. N_TILE 32→64 + __launch_bounds__(256,3): unlike Fix-A, a LARGER N-tile
//      wins here despite dropping occupancy to 3 CTAs/SM (50%). More N-tiles
//      per warp improves B-tile reuse + MMA-issue density per __syncthreads,
//      and that outweighs the occupancy loss once the dequant is cheap (lever
//      2). 18.0 → 23.5 TFLOP/s (+30%). N_TILE must stay ≤128 and a power-of-2
//      factor of FP8_BLOCK so a tile lives in ONE 128-N scale-block (N=96
//      straddles two blocks and breaks the single-scale-per-tile invariant).
// Net: 5.0 → 23.5 TFLOP/s = 4.7× over v2 at the representative size.
//
// Differences from w8a16_gemm_pipelined (all in the dispatch wrapper, not the
// inner pipeline): expert_id = blockIdx.z selects the per-expert weight/scale
// pointer + the [m_start,m_end) token band; A rows are GATHERED through
// sorted_token_ids (so the A cp.async resolves a fresh row pointer per row, but
// each row is still one contiguous K-run); output stores to the sorted
// position m_start+row, matching v1/v2.
//
// cp.async.cg (sm_80+) is correct on sm_121; TMA / cp.async.bulk AVOIDED
// (silent corruption on sm_121). cp.async.cg needs 16-byte-aligned smem dsts +
// strides, hence __align__(16) buffers and the 24-BF16 A stride.
//
// Grid: (ceil(N/PM3_N_TILE), ceil(max_tok_per_expert/PM3_M_TILE), num_experts)
// Block: (PM3_THREADS=256, 1, 1).

#define PM3_M_TILE 128
#define PM3_N_TILE 64
#define PM3_K_STEP 16
#define PM3_PAD 2
#define PM3_A_STRIDE 24                          // 16 K-cols + 8 pad, mult. of 8 BF16 (16 B)
#define PM3_WARPS 8
#define PM3_THREADS (PM3_WARPS * 32)             // 256
#define PM3_N_TILES_PER_WARP (PM3_N_TILE / 8)    // m16n8k16 N-tiles per warp (=8 at N_TILE=64)
#define PM3_STAGES 2

__device__ __forceinline__ void pm3_cp_async_cg_16(void* smem_ptr, const void* gmem_ptr) {
    unsigned int s = (unsigned int)__cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem_ptr));
}
__device__ __forceinline__ void pm3_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void pm3_cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void pm3_cp_async_wait_le(unsigned int n) {
    switch (n) {
        case 0:  pm3_cp_async_wait_group<0>(); break;
        case 1:  pm3_cp_async_wait_group<1>(); break;
        case 2:  pm3_cp_async_wait_group<2>(); break;
        default: pm3_cp_async_wait_group<3>(); break;
    }
}

// MMA over one resident K_STEP (16 K-elements) into inner[PM3_N_TILES_PER_WARP][4].
// smem_A is [PM3_M_TILE][PM3_A_STRIDE]; smem_B is [PM3_K_STEP][PM3_N_TILE+PM3_PAD]
// (already-dequantized BF16). Fragment layout byte-for-byte identical to v1/v2's
// fp8_moe_mma — only the N-tile count per warp changes (4 vs 8).
__device__ __forceinline__ void pm3_mma_kstep(
    const __nv_bfloat16* smem_A,
    const __nv_bfloat16* smem_B,
    float inner[PM3_N_TILES_PER_WARP][4],
    unsigned int warp_m_offset, unsigned int group_id, unsigned int tid
) {
    const unsigned int a_stride = PM3_A_STRIDE;
    const unsigned int b_stride = PM3_N_TILE + PM3_PAD;
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
    for (int n_tile = 0; n_tile < PM3_N_TILES_PER_WARP; n_tile++) {
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
            : "=f"(inner[n_tile][0]), "=f"(inner[n_tile][1]),
              "=f"(inner[n_tile][2]), "=f"(inner[n_tile][3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1),
              "f"(inner[n_tile][0]), "f"(inner[n_tile][1]),
              "f"(inner[n_tile][2]), "f"(inner[n_tile][3])
        );
    }
}

/// FP8 grouped GEMM v3 — occupancy-tuned, cp.async-pipelined.
/// SAME signature as v1/v2.
/// Grid: (ceil(N/PM3_N_TILE), ceil(max_tok_per_expert/PM3_M_TILE), num_experts)
/// Block: (PM3_THREADS=256, 1, 1)
extern "C" __global__ void __launch_bounds__(PM3_THREADS, 3) moe_fp8_grouped_gemm_v3(
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

    const unsigned int cta_m_local = blockIdx.y * PM3_M_TILE;   // expert-relative M base
    if ((int)cta_m_local >= M_expert) return;

    const unsigned int cta_n = blockIdx.x * PM3_N_TILE;

    const unsigned char* B_exp = (const unsigned char*)B_weight_ptrs[expert_id];
    const float* S_exp = (const float*)B_scale_ptrs[expert_id];
    if (B_exp == 0) return;   // NULL → remote expert under EP

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;           // 8 warps × 16 = 128 M-rows
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    // Pipelined smem. cp.async destinations (smem_A, smem_Braw) are __align__(16)
    // with 16-byte-aligned row strides. smem_B holds the MMA-ready [k][n] BF16
    // weights, cooperatively dequantized ONCE per K-step by all 256 threads.
    // Per stage: smem_A 128*24*2=6144 B + smem_B 16*34*2=1088 B + smem_Braw
    // 32*16=512 B ≈ 7744 B → 15488 B for 2 stages (well under the 48 KB limit).
    __shared__ __align__(16) __nv_bfloat16 smem_A[PM3_STAGES][PM3_M_TILE][PM3_A_STRIDE];
    __shared__ __nv_bfloat16 smem_B[PM3_STAGES][PM3_K_STEP][PM3_N_TILE + PM3_PAD];
    __shared__ __align__(16) unsigned char smem_Braw[PM3_STAGES][PM3_N_TILE][PM3_K_STEP];

    // E4M3 LUT staged into shared memory. The dequant reads it with DATA-
    // DEPENDENT (divergent) indices — in __constant__ memory those serialize
    // (constant cache broadcasts only when all lanes of a warp read the same
    // address), which dominated v3's inner loop. A 1 KB smem copy turns each
    // lookup into a banked smem read (no warp-serialization), and the VALUES
    // are byte-identical to E4M3_LUT_GMOE so the two-level FP32 numerics are
    // preserved exactly. Filled cooperatively once before the K-loop.
    __shared__ float lut_s[256];
    #pragma unroll
    for (unsigned int i = threadIdx.x; i < 256; i += PM3_THREADS) {
        lut_s[i] = E4M3_LUT_GMOE[i];
    }

    // Two-level FP32 accumulation — PRESERVED EXACTLY (inner over a 128-K block,
    // outer += inner * block_scale at the boundary; scale never per-element).
    float inner_acc[PM3_N_TILES_PER_WARP][4];
    float outer_acc[PM3_N_TILES_PER_WARP][4];
    #pragma unroll
    for (int i = 0; i < PM3_N_TILES_PER_WARP; i++) {
        inner_acc[i][0] = 0.0f; inner_acc[i][1] = 0.0f;
        inner_acc[i][2] = 0.0f; inner_acc[i][3] = 0.0f;
        outer_acc[i][0] = 0.0f; outer_acc[i][1] = 0.0f;
        outer_acc[i][2] = 0.0f; outer_acc[i][3] = 0.0f;
    }

    const unsigned int k_blocks = (K + FP8_BLOCK - 1) / FP8_BLOCK;
    const unsigned int k_steps_per_block = FP8_BLOCK / PM3_K_STEP;   // 8
    const unsigned int n_block = cta_n / FP8_BLOCK;
    const unsigned int n_steps = (K + PM3_K_STEP - 1) / PM3_K_STEP;

    // A-tile cp.async: 128 rows × 16 K-cols BF16. Each row = 16 BF16 = 32 B =
    // two 16-B chunks → 256 chunks total → 1 chunk/thread (256 threads). Unlike
    // the dense w8a16 kernel, each A row is GATHERED through sorted_token_ids,
    // so the row pointer is resolved per chunk; the 16-B run within a row is
    // still contiguous in K (one coalesced transaction).
    const unsigned int a_chunks = (PM3_M_TILE * PM3_K_STEP) / 8;     // 256

    auto prefetch = [&](unsigned int step, unsigned int stage) {
        unsigned int k_base = step * PM3_K_STEP;

        // ── A: contiguous 16-B (8 BF16) chunks along K, gathered per row ──
        #pragma unroll
        for (unsigned int c = threadIdx.x; c < a_chunks; c += PM3_THREADS) {
            unsigned int row = (c * 8) / PM3_K_STEP;          // 0..127
            unsigned int col = (c * 8) % PM3_K_STEP;          // 0 or 8
            unsigned int m_global = cta_m_local + row;        // expert-relative
            unsigned int gc = k_base + col;
            __nv_bfloat16* dst = &smem_A[stage][row][col];
            if (m_global < (unsigned int)M_expert && gc + 8 <= K) {
                int sorted_idx = m_start + (int)m_global;
                int token_id = sorted_token_ids ? sorted_token_ids[sorted_idx] : sorted_idx;
                pm3_cp_async_cg_16(dst, &A[(unsigned long long)token_id * K + gc]);
            } else {
                #pragma unroll
                for (unsigned int e = 0; e < 8; e++) {
                    unsigned int gcol = gc + e;
                    if (m_global < (unsigned int)M_expert && gcol < K) {
                        int sorted_idx = m_start + (int)m_global;
                        int token_id = sorted_token_ids ? sorted_token_ids[sorted_idx] : sorted_idx;
                        dst[e] = A[(unsigned long long)token_id * K + gcol];
                    } else {
                        dst[e] = __float2bfloat16(0.0f);
                    }
                }
            }
        }

        // ── B raw: one contiguous 16-B run of K-bytes per N-row ──
        #pragma unroll
        for (unsigned int nrow = threadIdx.x; nrow < PM3_N_TILE; nrow += PM3_THREADS) {
            unsigned int gn = cta_n + nrow;
            unsigned char* dst = &smem_Braw[stage][nrow][0];
            if (gn < N && k_base + PM3_K_STEP <= K) {
                pm3_cp_async_cg_16(dst, &B_exp[(unsigned long long)gn * K + k_base]);
            } else {
                #pragma unroll
                for (unsigned int e = 0; e < PM3_K_STEP; e++) {
                    unsigned int gk = k_base + e;
                    dst[e] = (gn < N && gk < K) ? B_exp[(unsigned long long)gn * K + gk] : 0;
                }
            }
        }
        pm3_cp_async_commit();
    };

    // LUT-dequant just-arrived raw B for `stage` into [k][n] BF16 (no scale —
    // folded on the FP32 accumulator at the block boundary). Cooperative across
    // all 256 threads so each weight is converted once and reused by all 8 warps.
    auto dequant_B = [&](unsigned int stage) {
        #pragma unroll
        for (unsigned int idx = threadIdx.x; idx < PM3_K_STEP * PM3_N_TILE; idx += PM3_THREADS) {
            unsigned int k = idx / PM3_N_TILE;     // 0..15
            unsigned int n = idx % PM3_N_TILE;     // 0..PM3_N_TILE-1
            unsigned char wb = smem_Braw[stage][n][k];
            smem_B[stage][k][n] = __float2bfloat16(lut_s[wb]);
        }
    };

    // ── Software-pipelined main loop (PM3_STAGES-deep cp.async) ──
    #pragma unroll
    for (unsigned int p = 0; p < PM3_STAGES - 1; p++) {
        if (p < n_steps) {
            prefetch(p, p % PM3_STAGES);
        }
    }
    unsigned int k_step_in_block = 0;

    for (unsigned int step = 0; step < n_steps; step++) {
        unsigned int cur = step % PM3_STAGES;

        unsigned int ahead = step + (PM3_STAGES - 1);
        if (ahead < n_steps) {
            prefetch(ahead, ahead % PM3_STAGES);
        }
        unsigned int committed = min(n_steps, PM3_STAGES + step);
        unsigned int target = committed - (step + 1);
        pm3_cp_async_wait_le(target);
        __syncthreads();   // raw B for `cur` resident for all threads

        dequant_B(cur);
        __syncthreads();   // smem_B[cur] fully written before MMA reads it

        pm3_mma_kstep(&smem_A[cur][0][0], &smem_B[cur][0][0],
                      inner_acc, warp_m_offset, group_id, tid);
        __syncthreads();   // done reading smem_*[cur]; safe for reuse

        // K_BLOCK boundary: fold scaled inner into outer, reset inner.
        k_step_in_block++;
        if (k_step_in_block == k_steps_per_block) {
            const unsigned int k_block = (step * PM3_K_STEP) / FP8_BLOCK;
            const float scale = S_exp[n_block * k_blocks + k_block];
            #pragma unroll
            for (int i = 0; i < PM3_N_TILES_PER_WARP; i++) {
                outer_acc[i][0] += inner_acc[i][0] * scale;
                outer_acc[i][1] += inner_acc[i][1] * scale;
                outer_acc[i][2] += inner_acc[i][2] * scale;
                outer_acc[i][3] += inner_acc[i][3] * scale;
                inner_acc[i][0] = 0.0f; inner_acc[i][1] = 0.0f;
                inner_acc[i][2] = 0.0f; inner_acc[i][3] = 0.0f;
            }
            k_step_in_block = 0;
        }
    }

    // Fold any incomplete trailing K_BLOCK (only when K % FP8_BLOCK != 0).
    if (k_step_in_block != 0) {
        const unsigned int k_block = (K - 1) / FP8_BLOCK;
        const float scale = S_exp[n_block * k_blocks + k_block];
        #pragma unroll
        for (int i = 0; i < PM3_N_TILES_PER_WARP; i++) {
            outer_acc[i][0] += inner_acc[i][0] * scale;
            outer_acc[i][1] += inner_acc[i][1] * scale;
            outer_acc[i][2] += inner_acc[i][2] * scale;
            outer_acc[i][3] += inner_acc[i][3] * scale;
        }
    }

    // ── Store C tile: f32 outer accumulators → BF16, sorted output position ──
    #pragma unroll
    for (int n_tile = 0; n_tile < PM3_N_TILES_PER_WARP; n_tile++) {
        unsigned int base_n = cta_n + n_tile * 8;
        unsigned int col0 = base_n + (tid * 2);
        unsigned int col1 = col0 + 1;
        unsigned int row0 = cta_m_local + warp_m_offset + group_id;   // expert-relative
        unsigned int row1 = row0 + 8;

        if (row0 < (unsigned int)M_expert) {
            unsigned int out_row = m_start + row0;
            if (col0 < N) C[(unsigned long long)out_row * N + col0] = __float2bfloat16(outer_acc[n_tile][0]);
            if (col1 < N) C[(unsigned long long)out_row * N + col1] = __float2bfloat16(outer_acc[n_tile][1]);
        }
        if (row1 < (unsigned int)M_expert) {
            unsigned int out_row = m_start + row1;
            if (col0 < N) C[(unsigned long long)out_row * N + col0] = __float2bfloat16(outer_acc[n_tile][2]);
            if (col1 < N) C[(unsigned long long)out_row * N + col1] = __float2bfloat16(outer_acc[n_tile][3]);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// v4 — Fix-B follow-ups on v3: K_STEP 16→32 + K-contiguous smem_B.
// ═══════════════════════════════════════════════════════════════════
//
// Same math + number formats as v1/v2/v3 (BF16 acts × FP8 E4M3 block-scaled
// weights, grouped per-expert dispatch, two-level FP32 accumulation PRESERVED
// EXACTLY, shared-memory E4M3 LUT). Starts from v3's 128×64 tile / 256-thread /
// cp.async-pipelined structure and applies the two levers that the dense
// w8a16_gemm_pipelined kernel proved (12→26 TFLOP/s, commit dd7d7bd):
//
//   LEVER A — K_STEP 16→32 (PM4_K_SUB=16, 2 sub-MMAs per step). v3 issued one
//     m16n8k16 MMA per resident K-step, paying the full barrier triple
//     (raw-B sync → dequant → smem_B sync → MMA → reuse sync) once per 16 K.
//     v4 keeps TWO 16-K sub-MMAs resident per step, so the barrier triple is
//     amortized over 2× the MMA-issue work — halving the __syncthreads count
//     per K traversed. Since v3 is MMA-issue/barrier-bound (per its own header:
//     "once the dequant is cheap ... K_STEP and the smem_B load pattern matter
//     most"), this is the primary expected lever.
//
//   LEVER B — K-contiguous smem_B [n][k] (transposed from v3's [k][n]). The MMA
//     B fragment packs two consecutive-K BF16 weights (k, k+1) per 32-bit
//     register. v3 stored smem_B [k][n], so that pair was two STRIDED 16-bit
//     loads + a shift/or:  ((u16)sB[(k+1)*b_stride+n]<<16) | (u16)sB[k*b_stride+n].
//     With [n][k] the pair is ADJACENT in smem → a SINGLE aligned 32-bit load:
//     *(u32*)&sB[n*b_stride + k]. Halves the smem instruction count on the B
//     fragment and removes the bit-shuffle ALU.
//
// Both levers preserve numerics EXACTLY: smem_B holds the identical unscaled
// BF16-cast E4M3 values, only the storage axis changes; the two sub-MMAs sum
// into the same inner_acc the single MMA did, in the same K order; the scale is
// still applied ONCE per 128-K block on the FP32 outer accumulator.
//
// smem budget per stage (N_TILE=64, K_STEP=32): smem_A 128×40×2 = 10240 B +
// smem_B 64×34×2 = 4352 B + smem_Braw 64×32 = 2048 B ≈ 16640 B → 33280 B for
// 2 stages + 1 KB LUT = 34304 B (well under the 101 KB cap).
//
//   LAUNCH_BOUNDS (256,2), NOT (256,3) as v3 uses. The K_STEP-32 sub-MMA loop
//   needs more live registers (A fragments for 2 sub-K windows + 8 N-tile
//   accumulators). At v3's (256,3) hint ptxas caps to 80 regs and SPILLS
//   (192 B store / 96 B load), measured 25.9 TFLOP/s. The 34 KB smem already
//   caps this kernel to 2 CTAs/SM (3×34 KB > 100 KB carveout) regardless of
//   the reg hint, so (256,2) costs ZERO occupancy yet lets ptxas use 125 regs
//   with NO spill → 31.3 TFLOP/s (+23% vs the spilling (256,3) build). Probed
//   K_STEP=64 (4 sub-MMAs): 63 KB smem → 1 CTA/SM, regressed to 22 TFLOP/s —
//   confirming the kernel rewards barrier-amortization + tile reuse at 2 CTAs
//   over raw occupancy, but only until smem collapses the CTA count.
//
// cp.async.cg / mma.sync.bf16 only; NO TMA / cp.async.bulk / e2m1 (corrupt on
// sm_121). Grid/block identical to v3 (M_TILE=128 unchanged).
//
// Grid: (ceil(N/PM4_N_TILE), ceil(max_tok_per_expert/PM4_M_TILE), num_experts)
// Block: (PM4_THREADS=256, 1, 1).

#define PM4_M_TILE 128
#define PM4_N_TILE 64
#define PM4_K_STEP 32
#define PM4_K_SUB 16                             // one MMA's K-width
#define PM4_K_SUBS (PM4_K_STEP / PM4_K_SUB)      // = 2 sub-MMAs per K-step
#define PM4_PAD 2
#define PM4_A_STRIDE (PM4_K_STEP + 8)            // K_STEP K-cols + 8 pad, mult. of 8 BF16 (16 B)
#define PM4_WARPS 8
#define PM4_THREADS (PM4_WARPS * 32)             // 256
#define PM4_N_TILES_PER_WARP (PM4_N_TILE / 8)    // m16n8k16 N-tiles per warp (=8 at N_TILE=64)
#define PM4_STAGES 2

__device__ __forceinline__ void pm4_cp_async_cg_16(void* smem_ptr, const void* gmem_ptr) {
    unsigned int s = (unsigned int)__cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem_ptr));
}
__device__ __forceinline__ void pm4_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void pm4_cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void pm4_cp_async_wait_le(unsigned int n) {
    switch (n) {
        case 0:  pm4_cp_async_wait_group<0>(); break;
        case 1:  pm4_cp_async_wait_group<1>(); break;
        case 2:  pm4_cp_async_wait_group<2>(); break;
        default: pm4_cp_async_wait_group<3>(); break;
    }
}

// MMA over one resident K_STEP (PM4_K_STEP=32 K-elements = PM4_K_SUBS=2 m16n8k16
// sub-MMAs of 16-K each) into inner[PM4_N_TILES_PER_WARP][4]. smem_B is [n][k]
// K-CONTIGUOUS (Lever B): the (k, k+1) BF16 pair of each MMA B fragment is a
// single aligned 32-bit load. The two sub-MMAs (Lever A) sum into the same
// inner accumulator in K order — byte-for-byte the same products v3 issued,
// just batched 2-at-a-time per barrier.
__device__ __forceinline__ void pm4_mma_kstep(
    const __nv_bfloat16* smem_A,   // [PM4_M_TILE][PM4_A_STRIDE]
    const __nv_bfloat16* smem_B,   // [PM4_N_TILE][PM4_K_STEP + PM4_PAD] (K-contiguous)
    float inner[PM4_N_TILES_PER_WARP][4],
    unsigned int warp_m_offset, unsigned int group_id, unsigned int tid
) {
    const unsigned int a_stride = PM4_A_STRIDE;
    const unsigned int b_stride = PM4_K_STEP + PM4_PAD;   // [n][k] K-contiguous stride
    const unsigned short* sA = (const unsigned short*)smem_A;
    const unsigned short* sB = (const unsigned short*)smem_B;

    unsigned int frag_r0 = warp_m_offset + group_id;
    unsigned int frag_r1 = warp_m_offset + group_id + 8;

    #pragma unroll
    for (int s = 0; s < PM4_K_SUBS; s++) {
        const unsigned int k_off = s * PM4_K_SUB;   // K offset of this sub-MMA within the step
        unsigned int frag_c0 = k_off + tid * 2;
        unsigned int frag_c1 = k_off + tid * 2 + 8;

        unsigned int a0 = *(const unsigned int*)&sA[frag_r0 * a_stride + frag_c0];
        unsigned int a1 = *(const unsigned int*)&sA[frag_r1 * a_stride + frag_c0];
        unsigned int a2 = *(const unsigned int*)&sA[frag_r0 * a_stride + frag_c1];
        unsigned int a3 = *(const unsigned int*)&sA[frag_r1 * a_stride + frag_c1];

        #pragma unroll
        for (int n_tile = 0; n_tile < PM4_N_TILES_PER_WARP; n_tile++) {
            unsigned int n_col = n_tile * 8 + group_id;
            unsigned int k0 = k_off + tid * 2;
            unsigned int k1 = k_off + tid * 2 + 8;

            // [n][k] K-contiguous: (k, k+1) adjacent → single aligned u32.
            unsigned int b0 = *(const unsigned int*)&sB[n_col * b_stride + k0];
            unsigned int b1 = *(const unsigned int*)&sB[n_col * b_stride + k1];

            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9}, "
                "{%10, %11, %12, %13};"
                : "=f"(inner[n_tile][0]), "=f"(inner[n_tile][1]),
                  "=f"(inner[n_tile][2]), "=f"(inner[n_tile][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(b0), "r"(b1),
                  "f"(inner[n_tile][0]), "f"(inner[n_tile][1]),
                  "f"(inner[n_tile][2]), "f"(inner[n_tile][3])
            );
        }
    }
}

/// FP8 grouped GEMM v4 — v3 + K_STEP 32 + K-contiguous smem_B.
/// SAME signature as v1/v2/v3.
/// Grid: (ceil(N/PM4_N_TILE), ceil(max_tok_per_expert/PM4_M_TILE), num_experts)
/// Block: (PM4_THREADS=256, 1, 1)
extern "C" __global__ void __launch_bounds__(PM4_THREADS, 2) moe_fp8_grouped_gemm_v4(
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

    const unsigned int cta_m_local = blockIdx.y * PM4_M_TILE;   // expert-relative M base
    if ((int)cta_m_local >= M_expert) return;

    const unsigned int cta_n = blockIdx.x * PM4_N_TILE;

    const unsigned char* B_exp = (const unsigned char*)B_weight_ptrs[expert_id];
    const float* S_exp = (const float*)B_scale_ptrs[expert_id];
    if (B_exp == 0) return;   // NULL → remote expert under EP

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;           // 8 warps × 16 = 128 M-rows
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    // Pipelined smem. cp.async destinations (smem_A, smem_Braw) are __align__(16)
    // with 16-byte-aligned row strides. smem_B holds the MMA-ready [n][k]
    // K-CONTIGUOUS BF16 weights (Lever B), cooperatively dequantized ONCE per
    // K-step by all 256 threads. Per stage (N_TILE=64, K_STEP=32):
    // smem_A 128*40*2=10240 B + smem_B 64*34*2=4352 B + smem_Braw 64*32=2048 B.
    __shared__ __align__(16) __nv_bfloat16 smem_A[PM4_STAGES][PM4_M_TILE][PM4_A_STRIDE];
    __shared__ __nv_bfloat16 smem_B[PM4_STAGES][PM4_N_TILE][PM4_K_STEP + PM4_PAD];
    __shared__ __align__(16) unsigned char smem_Braw[PM4_STAGES][PM4_N_TILE][PM4_K_STEP];

    // E4M3 LUT staged into shared memory (same rationale + byte-identical values
    // as v3 — data-dependent divergent lookups serialize in __constant__ memory).
    __shared__ float lut_s[256];
    #pragma unroll
    for (unsigned int i = threadIdx.x; i < 256; i += PM4_THREADS) {
        lut_s[i] = E4M3_LUT_GMOE[i];
    }

    // Two-level FP32 accumulation — PRESERVED EXACTLY (inner over a 128-K block,
    // outer += inner * block_scale at the boundary; scale never per-element).
    float inner_acc[PM4_N_TILES_PER_WARP][4];
    float outer_acc[PM4_N_TILES_PER_WARP][4];
    #pragma unroll
    for (int i = 0; i < PM4_N_TILES_PER_WARP; i++) {
        inner_acc[i][0] = 0.0f; inner_acc[i][1] = 0.0f;
        inner_acc[i][2] = 0.0f; inner_acc[i][3] = 0.0f;
        outer_acc[i][0] = 0.0f; outer_acc[i][1] = 0.0f;
        outer_acc[i][2] = 0.0f; outer_acc[i][3] = 0.0f;
    }

    const unsigned int k_blocks = (K + FP8_BLOCK - 1) / FP8_BLOCK;
    const unsigned int k_steps_per_block = FP8_BLOCK / PM4_K_STEP;   // 4 at K_STEP=32
    const unsigned int n_block = cta_n / FP8_BLOCK;
    const unsigned int n_steps = (K + PM4_K_STEP - 1) / PM4_K_STEP;

    // A-tile cp.async: 128 rows × PM4_K_STEP K-cols BF16. Each 16-B chunk = 8
    // BF16. At K_STEP=32: 128×32/8 = 512 chunks → 2 chunks/thread (256 threads).
    // Each A row is GATHERED through sorted_token_ids; the 16-B K-run within a
    // row is still contiguous (one coalesced transaction).
    const unsigned int a_chunks = (PM4_M_TILE * PM4_K_STEP) / 8;     // 512

    auto prefetch = [&](unsigned int step, unsigned int stage) {
        unsigned int k_base = step * PM4_K_STEP;

        // ── A: contiguous 16-B (8 BF16) chunks along K, gathered per row ──
        #pragma unroll
        for (unsigned int c = threadIdx.x; c < a_chunks; c += PM4_THREADS) {
            unsigned int row = (c * 8) / PM4_K_STEP;          // 0..127
            unsigned int col = (c * 8) % PM4_K_STEP;          // 0, 8, 16, 24
            unsigned int m_global = cta_m_local + row;        // expert-relative
            unsigned int gc = k_base + col;
            __nv_bfloat16* dst = &smem_A[stage][row][col];
            if (m_global < (unsigned int)M_expert && gc + 8 <= K) {
                int sorted_idx = m_start + (int)m_global;
                int token_id = sorted_token_ids ? sorted_token_ids[sorted_idx] : sorted_idx;
                pm4_cp_async_cg_16(dst, &A[(unsigned long long)token_id * K + gc]);
            } else {
                #pragma unroll
                for (unsigned int e = 0; e < 8; e++) {
                    unsigned int gcol = gc + e;
                    if (m_global < (unsigned int)M_expert && gcol < K) {
                        int sorted_idx = m_start + (int)m_global;
                        int token_id = sorted_token_ids ? sorted_token_ids[sorted_idx] : sorted_idx;
                        dst[e] = A[(unsigned long long)token_id * K + gcol];
                    } else {
                        dst[e] = __float2bfloat16(0.0f);
                    }
                }
            }
        }

        // ── B raw: contiguous 16-B (16 FP8-byte) chunks of K per N-row ──
        // smem_Braw[stage][n][k] mirrors global B[n, k_base + k] contiguously.
        // At K_STEP=32 each N-row is 32 bytes = two 16-B chunks.
        const unsigned int b_chunks = (PM4_N_TILE * PM4_K_STEP) / 16;   // 128 at K_STEP=32
        #pragma unroll
        for (unsigned int c = threadIdx.x; c < b_chunks; c += PM4_THREADS) {
            unsigned int nrow = (c * 16) / PM4_K_STEP;        // 0..PM4_N_TILE-1
            unsigned int kcol = (c * 16) % PM4_K_STEP;        // 0 or 16
            unsigned int gn = cta_n + nrow;
            unsigned int gk = k_base + kcol;
            unsigned char* dst = &smem_Braw[stage][nrow][kcol];
            if (gn < N && gk + 16 <= K) {
                pm4_cp_async_cg_16(dst, &B_exp[(unsigned long long)gn * K + gk]);
            } else {
                #pragma unroll
                for (unsigned int e = 0; e < 16; e++) {
                    unsigned int gke = gk + e;
                    dst[e] = (gn < N && gke < K) ? B_exp[(unsigned long long)gn * K + gke] : 0;
                }
            }
        }
        pm4_cp_async_commit();
    };

    // LUT-dequant just-arrived raw B for `stage` into the MMA-ready BF16 buffer.
    // smem_B is [n][k] K-contiguous, matching smem_Braw, so this is a same-layout
    // element-wise dequant (no transpose). NO scale (folded post-MMA at the
    // block boundary). Cooperative across all 256 threads (each weight converted
    // once, reused by all 8 warps).
    auto dequant_B = [&](unsigned int stage) {
        #pragma unroll
        for (unsigned int idx = threadIdx.x; idx < PM4_K_STEP * PM4_N_TILE; idx += PM4_THREADS) {
            unsigned int n = idx / PM4_K_STEP;     // 0..PM4_N_TILE-1
            unsigned int k = idx % PM4_K_STEP;     // 0..PM4_K_STEP-1
            unsigned char wb = smem_Braw[stage][n][k];
            smem_B[stage][n][k] = __float2bfloat16(lut_s[wb]);
        }
    };

    // ── Software-pipelined main loop (PM4_STAGES-deep cp.async) ──
    #pragma unroll
    for (unsigned int p = 0; p < PM4_STAGES - 1; p++) {
        if (p < n_steps) {
            prefetch(p, p % PM4_STAGES);
        }
    }
    unsigned int k_step_in_block = 0;

    for (unsigned int step = 0; step < n_steps; step++) {
        unsigned int cur = step % PM4_STAGES;

        unsigned int ahead = step + (PM4_STAGES - 1);
        if (ahead < n_steps) {
            prefetch(ahead, ahead % PM4_STAGES);
        }
        unsigned int committed = min(n_steps, PM4_STAGES + step);
        unsigned int target = committed - (step + 1);
        pm4_cp_async_wait_le(target);
        __syncthreads();   // raw B for `cur` resident for all threads

        dequant_B(cur);
        __syncthreads();   // smem_B[cur] fully written before MMA reads it

        pm4_mma_kstep(&smem_A[cur][0][0], &smem_B[cur][0][0],
                      inner_acc, warp_m_offset, group_id, tid);
        __syncthreads();   // done reading smem_*[cur]; safe for reuse

        // K_BLOCK boundary: fold scaled inner into outer, reset inner.
        k_step_in_block++;
        if (k_step_in_block == k_steps_per_block) {
            const unsigned int k_block = (step * PM4_K_STEP) / FP8_BLOCK;
            const float scale = S_exp[n_block * k_blocks + k_block];
            #pragma unroll
            for (int i = 0; i < PM4_N_TILES_PER_WARP; i++) {
                outer_acc[i][0] += inner_acc[i][0] * scale;
                outer_acc[i][1] += inner_acc[i][1] * scale;
                outer_acc[i][2] += inner_acc[i][2] * scale;
                outer_acc[i][3] += inner_acc[i][3] * scale;
                inner_acc[i][0] = 0.0f; inner_acc[i][1] = 0.0f;
                inner_acc[i][2] = 0.0f; inner_acc[i][3] = 0.0f;
            }
            k_step_in_block = 0;
        }
    }

    // Fold any incomplete trailing K_BLOCK (only when K % FP8_BLOCK != 0).
    if (k_step_in_block != 0) {
        const unsigned int k_block = (K - 1) / FP8_BLOCK;
        const float scale = S_exp[n_block * k_blocks + k_block];
        #pragma unroll
        for (int i = 0; i < PM4_N_TILES_PER_WARP; i++) {
            outer_acc[i][0] += inner_acc[i][0] * scale;
            outer_acc[i][1] += inner_acc[i][1] * scale;
            outer_acc[i][2] += inner_acc[i][2] * scale;
            outer_acc[i][3] += inner_acc[i][3] * scale;
        }
    }

    // ── Store C tile: f32 outer accumulators → BF16, sorted output position ──
    #pragma unroll
    for (int n_tile = 0; n_tile < PM4_N_TILES_PER_WARP; n_tile++) {
        unsigned int base_n = cta_n + n_tile * 8;
        unsigned int col0 = base_n + (tid * 2);
        unsigned int col1 = col0 + 1;
        unsigned int row0 = cta_m_local + warp_m_offset + group_id;   // expert-relative
        unsigned int row1 = row0 + 8;

        if (row0 < (unsigned int)M_expert) {
            unsigned int out_row = m_start + row0;
            if (col0 < N) C[(unsigned long long)out_row * N + col0] = __float2bfloat16(outer_acc[n_tile][0]);
            if (col1 < N) C[(unsigned long long)out_row * N + col1] = __float2bfloat16(outer_acc[n_tile][1]);
        }
        if (row1 < (unsigned int)M_expert) {
            unsigned int out_row = m_start + row1;
            if (col0 < N) C[(unsigned long long)out_row * N + col0] = __float2bfloat16(outer_acc[n_tile][2]);
            if (col1 < N) C[(unsigned long long)out_row * N + col1] = __float2bfloat16(outer_acc[n_tile][3]);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// v5 — grid-compaction: persistent 96-CTA grid over a COMPACTED work-list.
// ═══════════════════════════════════════════════════════════════════
//
// Same math + number formats as v1/v2/v3/v4 (BF16 acts × FP8 E4M3 block-scaled
// weights, grouped per-expert dispatch, two-level FP32 accumulation PRESERVED
// EXACTLY, shared-memory E4M3 LUT, K_STEP=32 + K-contiguous smem_B). The
// PER-TILE compute is BYTE-FOR-BYTE IDENTICAL to v4 — it reuses pm4_mma_kstep
// and v4's exact warp/lane setup, accumulator init, prefetch+dequant lambdas,
// pipelined K-loop, two-level fold, and store guard.
//
// The ONLY change vs v4: instead of deriving (expert, m_tile, n_tile) from
// blockIdx.{z,y,x} on a dense 3D grid `[ceil(N/64), max_m_tiles=1300,
// num_experts]` (≈16M CTAs/layer of which 99.5% early-exit because most
// (m_tile, expert) pairs are out of range), v5 launches a PERSISTENT 1D grid of
// PM5_PERSIST_CTAS=96 CTAs that stride over a COMPACTED work-list built by
// `moe_build_tile_worklist` (moe_permute.cu). Each work-item is exactly one
// tile v4 would NOT have early-exited on:
//   worklist[wid*2 + 0] = expert_id
//   worklist[wid*2 + 1] = (m_tile << 6) | n_tile
//   *total_tiles        = number of real tiles
// This collapses the launch overhead (the v4 dense grid was LAUNCH-bound at
// 1.7% of peak on the MoE FFN prefill) while keeping the inner GEMM unchanged.
//
// SAME-STREAM INVARIANT (R3): the builder + this kernel MUST be enqueued on the
// SAME stream so the read of *total_tiles / worklist[] here happens-after the
// builder's write. The Rust launcher enforces this (no cross-stream event).
//
// RISK #1 (smem reuse across work-items): a single CTA now processes MANY tiles
// in its persistent loop, all reusing the SAME smem_A/smem_B/smem_Braw staging
// buffers. A `__syncthreads()` at the TOP of each iteration fences the prior
// tile's last pipeline stage (still being read by stragglers in the final MMA
// reuse-sync) before this iteration re-primes the cp.async pipeline — otherwise
// a stale stage could be overwritten while still in use. The smem LUT (lut_s)
// is resident across the whole loop and filled ONCE before it.
//
// Grid: (PM5_PERSIST_CTAS=96, 1, 1)  Block: (PM4_THREADS=256, 1, 1)
#define PM5_PERSIST_CTAS 96   // 48 SMs × 2 CTAs/SM. SSOT — mirrored in the Rust
                              // launch wrapper ops::moe_fp8_grouped_gemm_v5.

extern "C" __global__ void __launch_bounds__(PM4_THREADS, 2) moe_fp8_grouped_gemm_v5(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ B_weight_ptrs,
    const unsigned long long* __restrict__ B_scale_ptrs,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    unsigned int num_experts,
    unsigned int N,
    unsigned int K,
    const unsigned int* __restrict__ worklist,   // [*total_tiles * 2] (expert, packed m/n)
    const int* __restrict__ total_tiles          // [1] (read-after-write on same stream)
) {
    // E4M3 LUT staged into shared memory (same rationale + byte-identical values
    // as v4 — data-dependent divergent lookups serialize in __constant__ memory).
    // RISK #1 / Step-3: filled ONCE before the persistent loop, resident across
    // all work-items (NOT re-filled per iteration).
    __shared__ float lut_s[256];
    #pragma unroll
    for (unsigned int i = threadIdx.x; i < 256; i += PM4_THREADS) {
        lut_s[i] = E4M3_LUT_GMOE[i];
    }

    // Pipelined smem — identical layout/budget to v4. Shared across all
    // work-items this CTA processes (see RISK #1 fence at the loop top).
    __shared__ __align__(16) __nv_bfloat16 smem_A[PM4_STAGES][PM4_M_TILE][PM4_A_STRIDE];
    __shared__ __nv_bfloat16 smem_B[PM4_STAGES][PM4_N_TILE][PM4_K_STEP + PM4_PAD];
    __shared__ __align__(16) unsigned char smem_Braw[PM4_STAGES][PM4_N_TILE][PM4_K_STEP];

    // Warp/lane setup — IDENTICAL to v4 (CTA-uniform, hoisted out of the loop).
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;           // 8 warps × 16 = 128 M-rows
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    const int total = *total_tiles;

    // Persistent loop: stride PM5_PERSIST_CTAS over the compacted work-list.
    for (int wid = blockIdx.x; wid < total; wid += PM5_PERSIST_CTAS) {
        __syncthreads();   // RISK #1 fix: fence smem reuse before re-priming pipeline

        // ── Coords from the work-list instead of blockIdx.{z,y,x} ──
        // Named mt/nt (not m_tile/n_tile) so they do NOT shadow the store
        // loop's `n_tile` MMA index below — keeping the store body byte-
        // identical to v4.
        unsigned int expert_id = worklist[wid * 2 + 0];
        unsigned int packed    = worklist[wid * 2 + 1];
        unsigned int mt = packed >> 6;
        unsigned int nt = packed & 0x3F;

        const int m_start = expert_offsets[expert_id];
        const int M_expert = expert_offsets[expert_id + 1] - m_start;

        const unsigned char* B_exp = (const unsigned char*)B_weight_ptrs[expert_id];
        const float* S_exp = (const float*)B_scale_ptrs[expert_id];
        if (B_exp == 0) continue;   // NULL → remote expert under EP (builder skips these too)

        const unsigned int cta_m_local = mt * PM4_M_TILE;   // expert-relative M base
        const unsigned int cta_n = nt * PM4_N_TILE;

        // ── v4 BODY VERBATIM from accumulator-init onward ──
        // Two-level FP32 accumulation — PRESERVED EXACTLY (inner over a 128-K block,
        // outer += inner * block_scale at the boundary; scale never per-element).
        // Step-4: RE-ZEROED at the top of each work-item (per-tile accumulators).
        float inner_acc[PM4_N_TILES_PER_WARP][4];
        float outer_acc[PM4_N_TILES_PER_WARP][4];
        #pragma unroll
        for (int i = 0; i < PM4_N_TILES_PER_WARP; i++) {
            inner_acc[i][0] = 0.0f; inner_acc[i][1] = 0.0f;
            inner_acc[i][2] = 0.0f; inner_acc[i][3] = 0.0f;
            outer_acc[i][0] = 0.0f; outer_acc[i][1] = 0.0f;
            outer_acc[i][2] = 0.0f; outer_acc[i][3] = 0.0f;
        }

        const unsigned int k_blocks = (K + FP8_BLOCK - 1) / FP8_BLOCK;
        const unsigned int k_steps_per_block = FP8_BLOCK / PM4_K_STEP;   // 4 at K_STEP=32
        const unsigned int n_block = cta_n / FP8_BLOCK;
        const unsigned int n_steps = (K + PM4_K_STEP - 1) / PM4_K_STEP;

        // A-tile cp.async: 128 rows × PM4_K_STEP K-cols BF16. Each 16-B chunk = 8
        // BF16. At K_STEP=32: 128×32/8 = 512 chunks → 2 chunks/thread (256 threads).
        // Each A row is GATHERED through sorted_token_ids; the 16-B K-run within a
        // row is still contiguous (one coalesced transaction).
        const unsigned int a_chunks = (PM4_M_TILE * PM4_K_STEP) / 8;     // 512

        auto prefetch = [&](unsigned int step, unsigned int stage) {
            unsigned int k_base = step * PM4_K_STEP;

            // ── A: contiguous 16-B (8 BF16) chunks along K, gathered per row ──
            #pragma unroll
            for (unsigned int c = threadIdx.x; c < a_chunks; c += PM4_THREADS) {
                unsigned int row = (c * 8) / PM4_K_STEP;          // 0..127
                unsigned int col = (c * 8) % PM4_K_STEP;          // 0, 8, 16, 24
                unsigned int m_global = cta_m_local + row;        // expert-relative
                unsigned int gc = k_base + col;
                __nv_bfloat16* dst = &smem_A[stage][row][col];
                if (m_global < (unsigned int)M_expert && gc + 8 <= K) {
                    int sorted_idx = m_start + (int)m_global;
                    int token_id = sorted_token_ids ? sorted_token_ids[sorted_idx] : sorted_idx;
                    pm4_cp_async_cg_16(dst, &A[(unsigned long long)token_id * K + gc]);
                } else {
                    #pragma unroll
                    for (unsigned int e = 0; e < 8; e++) {
                        unsigned int gcol = gc + e;
                        if (m_global < (unsigned int)M_expert && gcol < K) {
                            int sorted_idx = m_start + (int)m_global;
                            int token_id = sorted_token_ids ? sorted_token_ids[sorted_idx] : sorted_idx;
                            dst[e] = A[(unsigned long long)token_id * K + gcol];
                        } else {
                            dst[e] = __float2bfloat16(0.0f);
                        }
                    }
                }
            }

            // ── B raw: contiguous 16-B (16 FP8-byte) chunks of K per N-row ──
            // smem_Braw[stage][n][k] mirrors global B[n, k_base + k] contiguously.
            // At K_STEP=32 each N-row is 32 bytes = two 16-B chunks.
            const unsigned int b_chunks = (PM4_N_TILE * PM4_K_STEP) / 16;   // 128 at K_STEP=32
            #pragma unroll
            for (unsigned int c = threadIdx.x; c < b_chunks; c += PM4_THREADS) {
                unsigned int nrow = (c * 16) / PM4_K_STEP;        // 0..PM4_N_TILE-1
                unsigned int kcol = (c * 16) % PM4_K_STEP;        // 0 or 16
                unsigned int gn = cta_n + nrow;
                unsigned int gk = k_base + kcol;
                unsigned char* dst = &smem_Braw[stage][nrow][kcol];
                if (gn < N && gk + 16 <= K) {
                    pm4_cp_async_cg_16(dst, &B_exp[(unsigned long long)gn * K + gk]);
                } else {
                    #pragma unroll
                    for (unsigned int e = 0; e < 16; e++) {
                        unsigned int gke = gk + e;
                        dst[e] = (gn < N && gke < K) ? B_exp[(unsigned long long)gn * K + gke] : 0;
                    }
                }
            }
            pm4_cp_async_commit();
        };

        // LUT-dequant just-arrived raw B for `stage` into the MMA-ready BF16 buffer.
        // smem_B is [n][k] K-contiguous, matching smem_Braw, so this is a same-layout
        // element-wise dequant (no transpose). NO scale (folded post-MMA at the
        // block boundary). Cooperative across all 256 threads (each weight converted
        // once, reused by all 8 warps).
        auto dequant_B = [&](unsigned int stage) {
            #pragma unroll
            for (unsigned int idx = threadIdx.x; idx < PM4_K_STEP * PM4_N_TILE; idx += PM4_THREADS) {
                unsigned int n = idx / PM4_K_STEP;     // 0..PM4_N_TILE-1
                unsigned int k = idx % PM4_K_STEP;     // 0..PM4_K_STEP-1
                unsigned char wb = smem_Braw[stage][n][k];
                smem_B[stage][n][k] = __float2bfloat16(lut_s[wb]);
            }
        };

        // ── Software-pipelined main loop (PM4_STAGES-deep cp.async) ──
        #pragma unroll
        for (unsigned int p = 0; p < PM4_STAGES - 1; p++) {
            if (p < n_steps) {
                prefetch(p, p % PM4_STAGES);
            }
        }
        unsigned int k_step_in_block = 0;

        for (unsigned int step = 0; step < n_steps; step++) {
            unsigned int cur = step % PM4_STAGES;

            unsigned int ahead = step + (PM4_STAGES - 1);
            if (ahead < n_steps) {
                prefetch(ahead, ahead % PM4_STAGES);
            }
            unsigned int committed = min(n_steps, PM4_STAGES + step);
            unsigned int target = committed - (step + 1);
            pm4_cp_async_wait_le(target);
            __syncthreads();   // raw B for `cur` resident for all threads

            dequant_B(cur);
            __syncthreads();   // smem_B[cur] fully written before MMA reads it

            pm4_mma_kstep(&smem_A[cur][0][0], &smem_B[cur][0][0],
                          inner_acc, warp_m_offset, group_id, tid);
            __syncthreads();   // done reading smem_*[cur]; safe for reuse

            // K_BLOCK boundary: fold scaled inner into outer, reset inner.
            k_step_in_block++;
            if (k_step_in_block == k_steps_per_block) {
                const unsigned int k_block = (step * PM4_K_STEP) / FP8_BLOCK;
                const float scale = S_exp[n_block * k_blocks + k_block];
                #pragma unroll
                for (int i = 0; i < PM4_N_TILES_PER_WARP; i++) {
                    outer_acc[i][0] += inner_acc[i][0] * scale;
                    outer_acc[i][1] += inner_acc[i][1] * scale;
                    outer_acc[i][2] += inner_acc[i][2] * scale;
                    outer_acc[i][3] += inner_acc[i][3] * scale;
                    inner_acc[i][0] = 0.0f; inner_acc[i][1] = 0.0f;
                    inner_acc[i][2] = 0.0f; inner_acc[i][3] = 0.0f;
                }
                k_step_in_block = 0;
            }
        }

        // Fold any incomplete trailing K_BLOCK (only when K % FP8_BLOCK != 0).
        if (k_step_in_block != 0) {
            const unsigned int k_block = (K - 1) / FP8_BLOCK;
            const float scale = S_exp[n_block * k_blocks + k_block];
            #pragma unroll
            for (int i = 0; i < PM4_N_TILES_PER_WARP; i++) {
                outer_acc[i][0] += inner_acc[i][0] * scale;
                outer_acc[i][1] += inner_acc[i][1] * scale;
                outer_acc[i][2] += inner_acc[i][2] * scale;
                outer_acc[i][3] += inner_acc[i][3] * scale;
            }
        }

        // ── Store C tile: f32 outer accumulators → BF16, sorted output position ──
        #pragma unroll
        for (int n_tile = 0; n_tile < PM4_N_TILES_PER_WARP; n_tile++) {
            unsigned int base_n = cta_n + n_tile * 8;
            unsigned int col0 = base_n + (tid * 2);
            unsigned int col1 = col0 + 1;
            unsigned int row0 = cta_m_local + warp_m_offset + group_id;   // expert-relative
            unsigned int row1 = row0 + 8;

            if (row0 < (unsigned int)M_expert) {
                unsigned int out_row = m_start + row0;
                if (col0 < N) C[(unsigned long long)out_row * N + col0] = __float2bfloat16(outer_acc[n_tile][0]);
                if (col1 < N) C[(unsigned long long)out_row * N + col1] = __float2bfloat16(outer_acc[n_tile][1]);
            }
            if (row1 < (unsigned int)M_expert) {
                unsigned int out_row = m_start + row1;
                if (col0 < N) C[(unsigned long long)out_row * N + col0] = __float2bfloat16(outer_acc[n_tile][2]);
                if (col1 < N) C[(unsigned long long)out_row * N + col1] = __float2bfloat16(outer_acc[n_tile][3]);
            }
        }
    }
}
