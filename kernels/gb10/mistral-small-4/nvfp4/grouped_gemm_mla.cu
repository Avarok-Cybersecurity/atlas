// SPDX-License-Identifier: AGPL-3.0-only

// Grouped GEMM for MLA Q absorption and V extraction.
//
// Processes G independent small GEMMs in one kernel launch:
//   C_g[M, N_g] = A_g[M, K_g] @ B_g[N_g, K_g]^T   for g = 0..G-1
//
// Input layout: A[M, G * K_g] where A_g starts at offset g * K_g per row
// Weight layout: B[G * N_g, K_g] where B_g starts at row g * N_g
// Output layout: C[M, G * N_g] where C_g starts at offset g * N_g per row
//
// This is equivalent to a block-diagonal GEMM but without zero padding.
// The kernel processes each (token, group) pair as an independent GEMM row.
//
// Q absorption: G=32, M=N_tokens, K_g=nope=64, N_g=kv_lora=256
//   A = Q_nope[M, 32*64=2048] (extracted from Q_full, nope portion per head)
//   B = W_UK[32*256, 64] = [8192, 64]
//   C = Q_absorbed[M, 32*256=8192]
//
// V extraction: G=32, M=N_tokens, K_g=kv_lora=256, N_g=v_dim=128
//   A = attn_latent[M, 32*256=8192] (kv_lora portion of attention output per head)
//   B = W_UV[32*128, 256] = [4096, 256]
//   C = V_extracted[M, 32*128=4096]

#include <cuda_bf16.h>

// Each block computes one (token, group) pair's output row: [1, N_g].
// Grid: (M * G, ceil(N_g / TILE_N), 1)
// Block: (256, 1, 1)
// TILE_N = 4 output elements per block (each of 64 threads reduces K_g dims)

#define GG_BLOCK 256
#define GG_TILE_N 4  // outputs per block

extern "C" __global__ void grouped_gemm_mla(
    const __nv_bfloat16* __restrict__ A,    // [M, G * K_g]
    const __nv_bfloat16* __restrict__ B,    // [G * N_g, K_g]
    __nv_bfloat16* __restrict__ C,           // [M, G * N_g]
    unsigned int M,          // number of tokens
    unsigned int G,          // number of groups (= num_heads)
    unsigned int K_g,        // input dim per group
    unsigned int N_g,        // output dim per group
    unsigned int A_stride,   // G * K_g (elements per row in A)
    unsigned int C_stride    // G * N_g (elements per row in C)
) {
    // Decode (token, group, output_tile) from grid position
    unsigned int mg_idx = blockIdx.x;  // token * G + group
    unsigned int n_tile = blockIdx.y;  // which tile of N_g outputs
    unsigned int token = mg_idx / G;
    unsigned int group = mg_idx % G;

    if (token >= M) return;

    unsigned int tid = threadIdx.x;

    // Each block computes GG_TILE_N output elements.
    // 256 threads / GG_TILE_N = 64 threads per output element for K reduction.
    const unsigned int threads_per_out = GG_BLOCK / GG_TILE_N;  // 64
    const unsigned int local_n = tid / threads_per_out;           // 0..3
    const unsigned int k_lane = tid % threads_per_out;            // 0..63

    unsigned int n_idx = n_tile * GG_TILE_N + local_n;
    if (n_idx >= N_g) return;

    // Pointers for this (token, group)
    const __nv_bfloat16* A_row = A + (unsigned long long)token * A_stride + group * K_g;
    const __nv_bfloat16* B_row = B + (unsigned long long)(group * N_g + n_idx) * K_g;

    // Dot product: A_row[k] * B_row[k] for k = k_lane, k_lane+64, ...
    float acc = 0.0f;
    for (unsigned int k = k_lane; k < K_g; k += threads_per_out) {
        float a_val = __bfloat162float(A_row[k]);
        float b_val = __bfloat162float(B_row[k]);
        acc += a_val * b_val;
    }

    // Warp reduction (threads_per_out=64 = 2 warps)
    // First reduce within each 32-thread warp
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    // Cross-warp reduction: lane 0 of each warp writes to shared memory
    __shared__ float s_partial[GG_TILE_N][2];  // [output_idx][warp_within_output]
    unsigned int warp_in_out = k_lane / 32;    // 0 or 1
    unsigned int lane_in_warp = k_lane % 32;
    if (lane_in_warp == 0) {
        s_partial[local_n][warp_in_out] = acc;
    }
    __syncthreads();

    // Final sum and write
    if (k_lane == 0) {
        float sum = s_partial[local_n][0] + s_partial[local_n][1];
        C[(unsigned long long)token * C_stride + group * N_g + n_idx] = __float2bfloat16(sum);
    }
}
