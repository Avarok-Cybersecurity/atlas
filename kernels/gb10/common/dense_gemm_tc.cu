// SPDX-License-Identifier: AGPL-3.0-only

// Tensor Core BF16 GEMM for SM121 (GB10) — m16n8k16 MMA.
//
// C[M,N] = A[M,K] @ B[N,K]^T  (BF16 in, BF16 out, FP32 accumulation)
//
// Tile: 16M × 64N per block. 4 warps, each computes 16×16 via 2 MMA ops (16×8 each).
// K loop: iterate in chunks of 16 (MMA K dimension).
//
// Grid: (ceil(N/64), ceil(M/16), 1)
// Block: (128, 1, 1) = 4 warps
//
// SM121 workaround: ldmatrix.x4 is broken, use manual uint32 loads from shared memory.

#include <cuda_bf16.h>

#define TC_TM 16      // M-tile per block
#define TC_TN 64      // N-tile per block (8 MMA 16×8 tiles = 4 warps × 2 N-tiles each)
#define TC_TK 16      // K-tile (MMA K dimension)
#define TC_PAD 8      // shared memory padding for bank-conflict-free access
#define TC_BLOCK 128   // threads per block (4 warps)

extern "C" __global__ void dense_gemm_tc(
    const __nv_bfloat16* __restrict__ A,  // [M, K] row-major
    const __nv_bfloat16* __restrict__ B,  // [N, K] row-major (read as B^T)
    __nv_bfloat16* __restrict__ C,         // [M, N] row-major
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int m_block = blockIdx.y * TC_TM;
    const unsigned int n_block = blockIdx.x * TC_TN;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;
    const unsigned int group_id = lane_id >> 2;   // 0..7
    const unsigned int tid_in_group = lane_id & 3; // 0..3

    // Each warp handles 2 MMA tiles: N-tiles at offset warp_id*16 and warp_id*16+8
    // Output per warp: 16×16 (via 2 × m16n8k16 = 16×8 + 16×8)
    const unsigned int n_warp_base = warp_id * 16;

    // Shared memory: A_tile[16][TK+PAD], B_tile[TK][64+PAD]
    __shared__ __nv_bfloat16 smem_A[TC_TM][TC_TK + TC_PAD];
    __shared__ __nv_bfloat16 smem_B[TC_TK][TC_TN + TC_PAD];

    // Output accumulators: 2 N-tiles × 4 floats per MMA = 8 FP32 values per thread
    float acc[2][4];
    #pragma unroll
    for (int t = 0; t < 2; t++) {
        acc[t][0] = 0.0f; acc[t][1] = 0.0f;
        acc[t][2] = 0.0f; acc[t][3] = 0.0f;
    }

    const unsigned short* sA_u16 = (const unsigned short*)smem_A;
    const unsigned short* sB_u16 = (const unsigned short*)smem_B;
    const unsigned int sA_stride = TC_TK + TC_PAD;
    const unsigned int sB_stride = TC_TN + TC_PAD;

    // K loop
    for (unsigned int k_base = 0; k_base < K; k_base += TC_TK) {
        // Cooperative load: 128 threads load A[16][16] + B[16][64]
        // A: 16*16 = 256 elements, 128 threads → 2 elements per thread
        {
            // Load A tile (loop needed: 256 elements, 128 threads)
            for (unsigned int idx = tid; idx < TC_TM * TC_TK; idx += TC_BLOCK) {
                unsigned int r = idx / TC_TK;
                unsigned int c = idx % TC_TK;
                unsigned int gr = m_block + r;
                unsigned int gc = k_base + c;
                smem_A[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : __float2bfloat16(0.0f);
            }
            // Load B tile: B[N,K] read as B^T[K,N] → smem_B[k][n]
            // B[n, k] at offset n*K + k → store at smem_B[k][n]
            // 128 threads load 16*64 = 1024 elements → 8 per thread
            for (unsigned int i = tid; i < TC_TK * TC_TN; i += TC_BLOCK) {
                unsigned int bk = i / TC_TN;
                unsigned int bn = i % TC_TN;
                unsigned int gn = n_block + bn;
                unsigned int gk = k_base + bk;
                smem_B[bk][bn] = (gn < N && gk < K) ? B[(unsigned long long)gn * K + gk] : __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        // MMA: A[16,16] × B^T[16,64] → C[16,64]
        // Each warp: 2 MMA tiles at n_warp_base and n_warp_base+8

        // A fragment: same for both N-tiles (shared M rows)
        unsigned int ar0 = group_id;       // row 0..7
        unsigned int ar1 = group_id + 8;   // row 8..15
        unsigned int ac0 = tid_in_group * 2;       // col pair 0 (0,1 or 2,3 or 4,5 or 6,7)
        unsigned int ac1 = tid_in_group * 2 + 8;   // col pair 1 (8,9 or 10,11 or 12,13 or 14,15)
        unsigned int a0 = *(const unsigned int*)&sA_u16[ar0 * sA_stride + ac0];
        unsigned int a1 = *(const unsigned int*)&sA_u16[ar1 * sA_stride + ac0];
        unsigned int a2 = *(const unsigned int*)&sA_u16[ar0 * sA_stride + ac1];
        unsigned int a3 = *(const unsigned int*)&sA_u16[ar1 * sA_stride + ac1];

        // B fragments: 2 N-tiles per warp
        #pragma unroll
        for (int nt = 0; nt < 2; nt++) {
            unsigned int n_col = n_warp_base + nt * 8 + group_id;
            unsigned int k0 = tid_in_group * 2;
            unsigned int k1 = tid_in_group * 2 + 8;
            unsigned int b0 = ((unsigned int)sB_u16[k0 * sB_stride + n_col] |
                              ((unsigned int)sB_u16[(k0+1) * sB_stride + n_col] << 16));
            unsigned int b1 = ((unsigned int)sB_u16[k1 * sB_stride + n_col] |
                              ((unsigned int)sB_u16[(k1+1) * sB_stride + n_col] << 16));

            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9}, "
                "{%10, %11, %12, %13};"
                : "=f"(acc[nt][0]), "=f"(acc[nt][1]),
                  "=f"(acc[nt][2]), "=f"(acc[nt][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(b0), "r"(b1),
                  "f"(acc[nt][0]), "f"(acc[nt][1]),
                  "f"(acc[nt][2]), "f"(acc[nt][3])
            );
        }

        __syncthreads();
    }

    // Write output: MMA result layout → row-major C[M,N]
    // Each thread owns 4 output elements per N-tile:
    //   (row0, col0), (row0, col1), (row1, col0), (row1, col1)
    // where row0 = group_id, row1 = group_id+8
    //       col0 = nt*8 + tid_in_group*2, col1 = col0+1
    #pragma unroll
    for (int nt = 0; nt < 2; nt++) {
        unsigned int r0 = m_block + group_id;
        unsigned int r1 = m_block + group_id + 8;
        unsigned int c0 = n_block + n_warp_base + nt * 8 + tid_in_group * 2;
        unsigned int c1 = c0 + 1;

        if (r0 < M && c0 < N) C[r0 * N + c0] = __float2bfloat16(acc[nt][0]);
        if (r0 < M && c1 < N) C[r0 * N + c1] = __float2bfloat16(acc[nt][1]);
        if (r1 < M && c0 < N) C[r1 * N + c0] = __float2bfloat16(acc[nt][2]);
        if (r1 < M && c1 < N) C[r1 * N + c1] = __float2bfloat16(acc[nt][3]);
    }
}
