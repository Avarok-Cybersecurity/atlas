// SPDX-License-Identifier: AGPL-3.0-only

// Atlas Dense GEMM kernel for SM121 (GB10).
//
// C = A * B^T  where:
//   A: [M, K] BF16 (activations, row-major)
//   B: [N, K] BF16 (weights, row-major — standard HuggingFace layout)
//   C: [M, N] BF16 (output, row-major)
//
// The kernel reads B transposed: B^T[k,n] = B[n,k] = B[n*K + k].
//
// Phase 1: Correct scalar implementation with shared memory tiling.
// Phase 2: Will add mma.sync.aligned.m16n8k16 BF16 tensor cores.

#include <cuda_bf16.h>

#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

// Tiled GEMM: C[M,N] = A[M,K] * B[N,K]^T
// All matrices in BF16, accumulation in FP32.
//
// Grid: (ceil(N/TILE_N), ceil(M/TILE_M))
// Block: (TILE_N, TILE_M) — each thread computes one output element
extern "C" __global__ void dense_gemm_bf16(
    const __nv_bfloat16* __restrict__ A,  // [M, K] row-major
    const __nv_bfloat16* __restrict__ B,  // [N, K] row-major (read transposed)
    __nv_bfloat16* __restrict__ C,         // [M, N] row-major
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    // Each thread computes one element of C
    unsigned int row = blockIdx.y * TILE_M + threadIdx.y;
    unsigned int col = blockIdx.x * TILE_N + threadIdx.x;

    // Shared memory tiles
    __shared__ __nv_bfloat16 smem_A[TILE_M][TILE_K];
    __shared__ __nv_bfloat16 smem_B[TILE_K][TILE_N];

    float acc = 0.0f;

    // Loop over K in TILE_K chunks
    for (unsigned int k_base = 0; k_base < K; k_base += TILE_K) {
        // Load A tile
        if (row < M && (k_base + threadIdx.x) < K) {
            smem_A[threadIdx.y][threadIdx.x] = A[row * K + k_base + threadIdx.x];
        } else {
            smem_A[threadIdx.y][threadIdx.x] = __float2bfloat16(0.0f);
        }

        // Load B tile (B is [N,K] row-major, read as B^T[K,N])
        if ((k_base + threadIdx.y) < K && col < N) {
            smem_B[threadIdx.y][threadIdx.x] = B[(unsigned long long)col * K + (k_base + threadIdx.y)];
        } else {
            smem_B[threadIdx.y][threadIdx.x] = __float2bfloat16(0.0f);
        }

        __syncthreads();

        // Compute partial dot product
        for (unsigned int kk = 0; kk < TILE_K; kk++) {
            acc += __bfloat162float(smem_A[threadIdx.y][kk])
                 * __bfloat162float(smem_B[kk][threadIdx.x]);
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        C[row * N + col] = __float2bfloat16(acc);
    }
}

// Fused SiLU(gate) * up activation — vectorized 2-wide BF16 loads/stores.
// Input: [N, inter_size*2] where first half is gate, second half is up.
// Output: [N, inter_size]
// out[i] = silu(gate[i]) * up[i]  where silu(x) = x * sigmoid(x)
extern "C" __global__ void fused_silu_mul(
    const __nv_bfloat16* __restrict__ gate_up,  // [num_tokens, inter_size * 2]
    __nv_bfloat16* __restrict__ output,          // [num_tokens, inter_size]
    unsigned int num_tokens,
    unsigned int inter_size
) {
    // Each thread processes 2 elements (vectorized BF16x2)
    unsigned int idx2 = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int half_total = (num_tokens * inter_size) / 2;
    if (idx2 >= half_total) return;

    // Map linear index to (token, col_pair)
    unsigned int half_inter = inter_size / 2;
    unsigned int token = idx2 / half_inter;
    unsigned int col_pair = idx2 % half_inter;

    // Vectorized loads: 2 BF16 per 32-bit read
    const unsigned int* gate32 = (const unsigned int*)(gate_up + token * (inter_size * 2));
    const unsigned int* up32 = (const unsigned int*)(gate_up + token * (inter_size * 2) + inter_size);
    unsigned int* out32 = (unsigned int*)(output + token * inter_size);

    unsigned int g_packed = gate32[col_pair];
    unsigned int u_packed = up32[col_pair];

    float g0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(g_packed & 0xFFFF)));
    float g1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(g_packed >> 16)));
    float u0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(u_packed & 0xFFFF)));
    float u1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(u_packed >> 16)));

    // SiLU(gate) * up for both elements
    float sg0 = 1.0f / (1.0f + __expf(-g0));
    float sg1 = 1.0f / (1.0f + __expf(-g1));
    float r0 = g0 * sg0 * u0;
    float r1 = g1 * sg1 * u1;

    // Vectorized store
    unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(r0));
    unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(r1));
    out32[col_pair] = lo | (hi << 16);
}
