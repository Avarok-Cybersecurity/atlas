// SPDX-License-Identifier: AGPL-3.0-only

// Atlas BF16 Grouped MoE GEMM — for FP8-source models dequanted to BF16 at load.
//
// C[M_expert,N] = A[M_expert,K] (BF16) @ B_expert[N,K] (BF16)
//
// Same tile/MMA layout and thread mapping as the FP8 grouped-GEMM variant. The
// only difference is the B tile staging:
//   - FP8 variant: loads `unsigned char` weight bytes, runs them through
//     `E4M3_LUT[byte]` and `__float2bfloat16`, then uses two-level FP32
//     accumulation with a per-K-block scale factor.
//   - BF16 variant (this file): loads `__nv_bfloat16` directly into smem,
//     single-level FP32 accumulator across the full K loop. No LUT, no scale.
//
// Rationale: the cosine harness in `bench/fp8_dgx2_drift/cosine_run.py`
// confirmed Atlas's existing FP8 path has only ~0.001 cosine/layer headroom
// vs a perfect FP8 dequant reference (mean C cosine 0.995). The remaining
// ~0.004/layer drift to BF16-reference (A vs C: 0.996 vs 0.995 mean) comes
// from FP8 quantization itself. Loading BF16 weights eliminates the
// quantization step entirely, matching vLLM-BF16 reference quality.
//
// Grid: (ceil(N/64), max_m_tiles, num_experts)  Block: (128, 1, 1)

#include <cuda_bf16.h>

#define M_TILE 64
#define N_TILE 64
#define K_STEP 16
#define PAD 2

// MMA compute (identical layout to moe_fp8_grouped_gemm — shared output semantics).
__device__ __forceinline__ void bf16_moe_mma(
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

/// BF16 grouped GEMM for sorted MoE dispatch.
///
/// Coalesced thread mapping (8 groups × 16 threads). Same output layout as
/// the FP8 v2 kernel — just BF16 weights instead of FP8 + scale.
///
/// Grid: (ceil(N/64), max_m_tiles, num_experts)  Block: (128, 1, 1)
extern "C" __global__ void moe_bf16_grouped_gemm(
    const __nv_bfloat16* __restrict__ A,                   // [total_tokens, K] BF16
    const unsigned long long* __restrict__ B_weight_ptrs,   // [num_experts] → [N, K] BF16
    __nv_bfloat16* __restrict__ C,                         // [total_expanded, N] BF16
    const int* __restrict__ expert_offsets,                 // [num_experts + 1]
    const int* __restrict__ sorted_token_ids,              // [total_expanded] or NULL
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

    const __nv_bfloat16* B_exp = (const __nv_bfloat16*)B_weight_ptrs[expert_id];
    if (B_exp == 0) return;  // NULL → remote expert under EP, or absent

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[M_TILE][K_STEP + PAD];
    __shared__ __nv_bfloat16 smem_B[K_STEP][N_TILE + PAD];

    const unsigned int thread_group = threadIdx.x >> 4;      // 0..7
    const unsigned int k_offset     = threadIdx.x & 15;      // 0..K_STEP-1
    const unsigned int row_base     = thread_group * 8;      // 0, 8, 16, ..., 56

    // Single-level FP32 accumulator. No per-K-block scale needed (BF16 weights
    // already at target precision). The MMA itself accumulates in FP32 across
    // the entire K loop — no precision loss vs the FP8 two-level dance.
    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    for (unsigned int k_base = 0; k_base < K; k_base += K_STEP) {
        const unsigned int gk = k_base + k_offset;

        // Load A tile [M_TILE=64 rows][K_STEP=16 cols] from global into smem.
        // 16 threads of the same thread_group share one m-row, vary k_offset
        // 0..15 — coalesced 32-byte burst per warp instruction.
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

        // Load B tile directly as BF16 — no dequant, no LUT, no scale.
        #pragma unroll
        for (unsigned int i = 0; i < 8; i++) {
            unsigned int n_local = row_base + i;
            unsigned int gn = cta_n + n_local;
            if (gk < K && gn < N) {
                smem_B[k_offset][n_local] = B_exp[(unsigned long long)gn * K + gk];
            } else {
                smem_B[k_offset][n_local] = __float2bfloat16(0.0f);
            }
        }

        __syncthreads();
        bf16_moe_mma(smem_A, smem_B, acc, warp_m_offset, group_id, tid);
        __syncthreads();
    }

    // Store C tile from FP32 accumulator.
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
