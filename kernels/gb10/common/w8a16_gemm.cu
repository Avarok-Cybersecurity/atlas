// SPDX-License-Identifier: AGPL-3.0-only

// Atlas W8A16 Dequant+GEMM — Fused FP8-E4M3 weight dequant + BF16 Tensor Core GEMM.
//
// C[M,N] = A[M,K] (BF16 activations) * dequant(B[N,K] (FP8 E4M3 weights))
//
// FP8-E4M3 weight format (2D block-scaled):
//   B:           [N, K] uint8 — one byte per weight (FP8 E4M3)
//   block_scale: [N/128, K/128] FP32 — per-block scale factor (scale_inv
//                widened to FP32 at load; applied in full FP32 precision)
//
// Dequant: bf16_val = E4M3_LUT[byte] * block_scale[n/128, k/128]
//
// Uses mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
//
// Grid: (ceil(N/64), ceil(M/64), 1), Block: (128, 1, 1)

#include <cuda_bf16.h>
// E4M3_LUT (256-entry __constant__ FP32) extracted to a shared header so the
// pipelined Fix-A rewrite (w8a16_gemm_pipelined.cu) reuses the SAME table
// (SSOT) rather than duplicating it. Behaviour of this kernel is unchanged —
// the table contents are byte-identical to the prior inline definition.
#include "e4m3_lut.cuh"

#define M_TILE 64
#define N_TILE 64
#define K_STEP 16
#define PAD 2
#define FP8_BLOCK 128

// MMA compute — shared between potential layout variants.
// Operates on already-loaded smem_A[M_TILE][K_STEP+PAD] and smem_B[K_STEP][N_TILE+PAD].
// Structurally identical to w4a16_mma_and_store (both see BF16 tiles in smem).
__device__ __forceinline__ void w8a16_mma_and_store(
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

/// W8A16 GEMM: B[N, K] row-major FP8 E4M3 with 2D block scales.
///
/// block_scale[N/128, K/128] FP32 — one scale per 128x128 weight block.
/// Dequant during B tile load: bf16_val = E4M3_LUT[byte] * block_scale
extern "C" __global__ void w8a16_gemm(
    const __nv_bfloat16* __restrict__ A,            // [M, K] BF16 activations
    const unsigned char* __restrict__ B,             // [N, K] FP8 E4M3
    const float* __restrict__ block_scale,           // [N/128, K/128] FP32
    __nv_bfloat16* __restrict__ C,                   // [M, N] BF16 output
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int cta_m = blockIdx.y * M_TILE;
    const unsigned int cta_n = blockIdx.x * N_TILE;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[M_TILE][K_STEP + PAD];
    __shared__ __nv_bfloat16 smem_B[K_STEP][N_TILE + PAD];

    // Two-level FP32 accumulation (vLLM W8A8 / DeepGEMM pattern):
    //   inner_acc — runs through K_STEPS_PER_BLOCK MMA steps within one K_BLOCK,
    //               smem_B holds UNSCALED BF16-cast FP8 weights (lossless cast
    //               since E4M3 has 3 mantissa bits, BF16 has 7).
    //   outer_acc — accumulates inner_acc × block_scale (FP32) at each K_BLOCK
    //               boundary. The scale is applied ONCE per block on the FP32
    //               accumulator, not per-element on a BF16 narrowing — which
    //               is the bug we are removing.
    //
    // n_block is constant per CTA because N_TILE=64 ≤ FP8_BLOCK=128 and cta_n
    // is N_TILE-aligned (so all N values within a CTA share the same n_block).
    float inner_acc[8][4];
    float outer_acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        inner_acc[i][0] = 0.0f; inner_acc[i][1] = 0.0f;
        inner_acc[i][2] = 0.0f; inner_acc[i][3] = 0.0f;
        outer_acc[i][0] = 0.0f; outer_acc[i][1] = 0.0f;
        outer_acc[i][2] = 0.0f; outer_acc[i][3] = 0.0f;
    }

    const unsigned int k_blocks = K / FP8_BLOCK;
    const unsigned int k_steps_per_block = FP8_BLOCK / K_STEP;
    const unsigned int n_block = cta_n / FP8_BLOCK;
    unsigned int k_step_in_block = 0;

    for (unsigned int k_base = 0; k_base < K; k_base += K_STEP) {
        // === Load A tile: [M_TILE, K_STEP] BF16 from global → smem ===
        {
            const unsigned int elems_per_thread = (M_TILE * K_STEP) / 128;
            #pragma unroll
            for (unsigned int i = 0; i < elems_per_thread; i++) {
                unsigned int idx = threadIdx.x * elems_per_thread + i;
                unsigned int row = idx / K_STEP;
                unsigned int col = idx % K_STEP;
                unsigned int gr = cta_m + row;
                unsigned int gc = k_base + col;
                smem_A[row][col] = (gr < M && gc < K) ? A[gr * K + gc] : __float2bfloat16(0.0f);
            }
        }

        // === Dequant B: FP8 E4M3 → BF16 via LUT (NO scale yet) ===
        // B tile: [K_STEP, N_TILE] — 16×64 = 1024 elements, 128 threads → 8 per thread.
        // The LUT result is exact (FP8 E4M3 fits in BF16); narrowing here is
        // lossless. Scale is folded onto the FP32 accumulator below.
        {
            #pragma unroll
            for (unsigned int i = 0; i < 8; i++) {
                unsigned int idx = threadIdx.x * 8 + i;
                unsigned int k = idx / N_TILE;
                unsigned int n = idx % N_TILE;
                unsigned int gk = k_base + k;
                unsigned int gn = cta_n + n;

                if (gk < K && gn < N) {
                    unsigned char weight_byte = B[(unsigned long long)gn * K + gk];
                    smem_B[k][n] = __float2bfloat16(E4M3_LUT[weight_byte]);
                } else {
                    smem_B[k][n] = __float2bfloat16(0.0f);
                }
            }
        }

        __syncthreads();
        w8a16_mma_and_store(smem_A, smem_B, inner_acc, warp_m_offset, group_id, tid);
        __syncthreads();

        // K_BLOCK boundary: fold scaled inner into outer, reset inner.
        k_step_in_block++;
        if (k_step_in_block == k_steps_per_block) {
            const unsigned int k_block = k_base / FP8_BLOCK;
            const float scale = block_scale[n_block * k_blocks + k_block];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
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

    // Fold any incomplete trailing K_BLOCK (only fires when K is not a
    // multiple of FP8_BLOCK — for Qwen3.6 hidden_dim=2048 this is dead code,
    // but keeps the kernel correct on other configs).
    if (k_step_in_block != 0) {
        const unsigned int k_block = (K - 1) / FP8_BLOCK;
        const float scale = block_scale[n_block * k_blocks + k_block];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            outer_acc[i][0] += inner_acc[i][0] * scale;
            outer_acc[i][1] += inner_acc[i][1] * scale;
            outer_acc[i][2] += inner_acc[i][2] * scale;
            outer_acc[i][3] += inner_acc[i][3] * scale;
        }
    }

    // === Store C tile: f32 outer accumulators → BF16 output ===
    #pragma unroll
    for (int n_tile = 0; n_tile < 8; n_tile++) {
        unsigned int base_n = cta_n + n_tile * 8;
        unsigned int col0 = base_n + (tid * 2);
        unsigned int col1 = col0 + 1;
        unsigned int row0 = cta_m + warp_m_offset + group_id;
        unsigned int row1 = row0 + 8;

        if (row0 < M && col0 < N) C[row0 * N + col0] = __float2bfloat16(outer_acc[n_tile][0]);
        if (row0 < M && col1 < N) C[row0 * N + col1] = __float2bfloat16(outer_acc[n_tile][1]);
        if (row1 < M && col0 < N) C[row1 * N + col0] = __float2bfloat16(outer_acc[n_tile][2]);
        if (row1 < M && col1 < N) C[row1 * N + col1] = __float2bfloat16(outer_acc[n_tile][3]);
    }
}

/// Standalone W8A16 dequant: B_fp8 → B_bf16 [N, K]
/// Each thread handles one FP8 byte → 1 BF16 output.
extern "C" __global__ void w8a16_dequant(
    const unsigned char* __restrict__ B,             // [N, K] FP8 E4M3
    const float* __restrict__ block_scale,           // [N/128, K/128] FP32
    __nv_bfloat16* __restrict__ B_bf16,              // [N, K] BF16 output
    unsigned int K,
    unsigned int N
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = N * K;
    if (idx >= total) return;

    unsigned int n = idx / K;
    unsigned int k = idx % K;

    unsigned char weight_byte = B[idx];

    unsigned int k_blocks = K / FP8_BLOCK;
    unsigned int n_block = n / FP8_BLOCK;
    unsigned int k_block = k / FP8_BLOCK;
    float scale = block_scale[n_block * k_blocks + k_block];

    float val = E4M3_LUT[weight_byte] * scale;
    B_bf16[idx] = __float2bfloat16(val);
}
