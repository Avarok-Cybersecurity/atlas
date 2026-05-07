// SPDX-License-Identifier: AGPL-3.0-only

// Split-K Dense GEMM for SM121 — saturates all SMs on skinny M matrices.
//
// Standard GEMM: Grid = (ceil(N/TN), ceil(M/TM)) — for M=22, N=256: only 32 blocks
// Split-K GEMM: Grid = (ceil(N/TN), ceil(M/TM), K_splits) — 32 * K_splits blocks
//
// Each block computes a PARTIAL result over K_chunk = K/K_splits dimensions.
// A reduction kernel then sums the K_splits partial results per output element.
//
// C_partial[split, M, N] += A[M, k_start:k_end] * B[N, k_start:k_end]^T
// C[M, N] = sum(C_partial[split, M, N])
//
// This is a 2-phase approach:
//   Phase 1: dense_gemm_splitk_partial — compute partial products (FP32 workspace)
//   Phase 2: dense_gemm_splitk_reduce — sum partials and convert to BF16

#include <cuda_bf16.h>

#define SK_TILE_M 16
#define SK_TILE_N 16
#define SK_TILE_K 16

// Phase 1: Compute partial GEMM over K_chunk dimensions.
// Grid: (ceil(N/TN), ceil(M/TM), K_splits)
// Block: (TN, TM) = (16, 16) = 256 threads
extern "C" __global__ void dense_gemm_splitk_partial(
    const __nv_bfloat16* __restrict__ A,  // [M, K]
    const __nv_bfloat16* __restrict__ B,  // [N, K] (read transposed)
    float* __restrict__ C_partial,         // [K_splits, M, N] FP32 workspace
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int K_splits
) {
    unsigned int split = blockIdx.z;
    unsigned int row = blockIdx.y * SK_TILE_M + threadIdx.y;
    unsigned int col = blockIdx.x * SK_TILE_N + threadIdx.x;

    // K range for this split
    unsigned int k_chunk = (K + K_splits - 1) / K_splits;
    unsigned int k_start = split * k_chunk;
    unsigned int k_end = min(k_start + k_chunk, K);

    __shared__ __nv_bfloat16 smem_A[SK_TILE_M][SK_TILE_K];
    __shared__ __nv_bfloat16 smem_B[SK_TILE_K][SK_TILE_N];

    float acc = 0.0f;

    for (unsigned int k_base = k_start; k_base < k_end; k_base += SK_TILE_K) {
        unsigned int k_local = k_base + threadIdx.x;
        if (row < M && k_local < k_end) {
            smem_A[threadIdx.y][threadIdx.x] = A[row * K + k_local];
        } else {
            smem_A[threadIdx.y][threadIdx.x] = __float2bfloat16(0.0f);
        }

        unsigned int k_local_y = k_base + threadIdx.y;
        if (k_local_y < k_end && col < N) {
            smem_B[threadIdx.y][threadIdx.x] = B[(unsigned long long)col * K + k_local_y];
        } else {
            smem_B[threadIdx.y][threadIdx.x] = __float2bfloat16(0.0f);
        }

        __syncthreads();

        unsigned int tile_end = min(SK_TILE_K, k_end - k_base);
        for (unsigned int kk = 0; kk < tile_end; kk++) {
            acc += __bfloat162float(smem_A[threadIdx.y][kk])
                 * __bfloat162float(smem_B[kk][threadIdx.x]);
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C_partial[(unsigned long long)split * M * N + row * N + col] = acc;
    }
}

// Phase 2: Reduce K_splits partial results and write BF16 output.
// Grid: (ceil(N/256), M, 1)
// Block: (256, 1, 1)
extern "C" __global__ void dense_gemm_splitk_reduce(
    const float* __restrict__ C_partial,  // [K_splits, M, N]
    __nv_bfloat16* __restrict__ C,         // [M, N]
    unsigned int M,
    unsigned int N,
    unsigned int K_splits
) {
    unsigned int row = blockIdx.y;
    unsigned int col = blockIdx.x * 256 + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (unsigned int s = 0; s < K_splits; s++) {
        sum += C_partial[(unsigned long long)s * M * N + row * N + col];
    }

    C[row * N + col] = __float2bfloat16(sum);
}
