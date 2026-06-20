// SPDX-License-Identifier: AGPL-3.0-only
//
// Atlas W8A8 + FP32 epilogue MoE Grouped GEMM — vLLM-equivalent numerics.
//
// Same shape/layout as `moe_fp8_grouped_gemm.cu` but:
//   - A is FP8 E4M3 (one byte per element), pre-quantized per-token-per-128
//     via `per_token_group_quant_fp8`. Dequanted to BF16 in smem via LUT
//     (lossless: FP8 3-bit mantissa fits in BF16 7-bit).
//   - a_scale[M_total, K/128] FP32 — looked up via sorted_token_ids[m_start + m_idx].
//   - b_scale[N/128, K/128] FP32 — scale_inv widened to FP32 at load (read once per fold).
//   - Two-level FP32 accumulation: inner_acc over K=128 block (4× K_STEP=16),
//     then outer_acc += inner_acc × (a_scale[row, kb] × b_scale[col, kb]).
//
//   C[M_expert, N] = bf16( Σ_kb ( Σ_(k∈kb) bf16(LUT[A_fp8[m,k]]) * bf16(LUT[B_fp8[n,k]]) )
//                       * a_scale[orig_token(m), kb] * b_scale[n/128, kb] )
//
// Grid: (ceil(N/64), max_m_tiles, num_experts)  Block: (128, 1, 1)

#include <cuda_bf16.h>

#define M_TILE 64
#define N_TILE 64
#define K_STEP 16
#define PAD 2
#define FP8_BLOCK 128
#define K_PROMOTE 64

__device__ __constant__ float E4M3_LUT_W8A8[256] = {
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

__device__ __forceinline__ void fp8_w8a8_mma(
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

extern "C" __global__ void moe_w8a8_grouped_gemm(
    const unsigned char* __restrict__ A_fp8,                // [total_tokens, K] FP8 E4M3
    const float* __restrict__ a_scale,                      // [total_tokens, K/128] FP32
    const unsigned long long* __restrict__ B_weight_ptrs,   // [num_experts] → [N, K] FP8
    const unsigned long long* __restrict__ B_scale_ptrs,    // [num_experts] → [N/128, K/128] FP32
    __nv_bfloat16* __restrict__ C,                          // [total_expanded, N] BF16
    const int* __restrict__ expert_offsets,                 // [num_experts + 1]
    const int* __restrict__ sorted_token_ids,               // [total_expanded] or NULL
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
    // Per-warp cache of the original-token-id for each of its M-rows (group_id 0..7
    // → 8 row indices per warp). Stored once per CTA to avoid re-resolving every
    // K_STEP. We need rows [warp_m_offset .. warp_m_offset+15] for the warp's
    // two fragments (r0_global = warp_m_offset+group_id, r1 = +8).
    // CTA covers M_TILE=64 rows total — store all 64 here.
    __shared__ int smem_token_id[M_TILE];
    if (threadIdx.x < M_TILE) {
        int m_idx = threadIdx.x;
        if (m_idx + cta_m_local < M_expert) {
            int sorted_idx = m_start + cta_m_local + m_idx;
            smem_token_id[m_idx] = sorted_token_ids ? sorted_token_ids[sorted_idx] : sorted_idx;
        } else {
            smem_token_id[m_idx] = -1;
        }
    }
    __syncthreads();

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
        // Load A tile: gather FP8 from sorted token positions, dequant to BF16 (lossless).
        {
            #pragma unroll
            for (unsigned int i = 0; i < 8; i++) {
                unsigned int idx = threadIdx.x * 8 + i;
                unsigned int row = idx / K_STEP;
                unsigned int col = idx % K_STEP;
                unsigned int m_idx = cta_m_local + row;
                unsigned int gc = k_base + col;

                if (m_idx < (unsigned int)M_expert && gc < K) {
                    int token_id = smem_token_id[row];
                    if (token_id >= 0) {
                        unsigned char a_byte = A_fp8[(unsigned long long)token_id * K + gc];
                        smem_A[row][col] = __float2bfloat16(E4M3_LUT_W8A8[a_byte]);
                    } else {
                        smem_A[row][col] = __float2bfloat16(0.0f);
                    }
                } else {
                    smem_A[row][col] = __float2bfloat16(0.0f);
                }
            }
        }

        // Dequant B tile: FP8 E4M3 → BF16 via LUT (lossless).
        {
            #pragma unroll
            for (unsigned int i = 0; i < 8; i++) {
                unsigned int idx = threadIdx.x * 8 + i;
                unsigned int k = idx / N_TILE;
                unsigned int n = idx % N_TILE;
                unsigned int gk = k_base + k;
                unsigned int gn = cta_n + n;

                if (gk < K && gn < N) {
                    unsigned char b_byte = B_exp[(unsigned long long)gn * K + gk];
                    smem_B[k][n] = __float2bfloat16(E4M3_LUT_W8A8[b_byte]);
                } else {
                    smem_B[k][n] = __float2bfloat16(0.0f);
                }
            }
        }

        __syncthreads();
        fp8_w8a8_mma(smem_A, smem_B, inner_acc, warp_m_offset, group_id, tid);
        __syncthreads();

        unsigned int next_k = k_base + K_STEP;
        if (next_k % K_PROMOTE == 0 || next_k >= K) {
            unsigned int k_block = k_base / FP8_BLOCK;
            const float bs = S_exp[n_block * k_blocks + k_block];
            // a_scale lookup per row. Row 0..7 use r0 = warp_m_offset+group_id,
            // row 8..15 use r1 = r0+8. For each n_tile, acc[][0,1] write to row r0,
            // acc[][2,3] to row r1.
            unsigned int r0 = warp_m_offset + group_id;
            unsigned int r1 = r0 + 8;
            int t0 = (r0 < M_TILE) ? smem_token_id[r0] : -1;
            int t1 = (r1 < M_TILE) ? smem_token_id[r1] : -1;
            const float as0 = (t0 >= 0)
                ? a_scale[(unsigned long long)t0 * k_blocks + k_block]
                : 0.0f;
            const float as1 = (t1 >= 0)
                ? a_scale[(unsigned long long)t1 * k_blocks + k_block]
                : 0.0f;
            const float s0 = as0 * bs;
            const float s1 = as1 * bs;
            #pragma unroll
            for (int n_tile = 0; n_tile < 8; n_tile++) {
                outer_acc[n_tile][0] += inner_acc[n_tile][0] * s0;
                outer_acc[n_tile][1] += inner_acc[n_tile][1] * s0;
                outer_acc[n_tile][2] += inner_acc[n_tile][2] * s1;
                outer_acc[n_tile][3] += inner_acc[n_tile][3] * s1;
                inner_acc[n_tile][0] = 0.0f; inner_acc[n_tile][1] = 0.0f;
                inner_acc[n_tile][2] = 0.0f; inner_acc[n_tile][3] = 0.0f;
            }
        }
    }

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
