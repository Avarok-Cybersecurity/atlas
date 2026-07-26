// SPDX-License-Identifier: AGPL-3.0-only

// Tensor Core BF16 GEMM for gfx1151 (Strix Halo) — AMD WMMA 16x16x16.
// HIP port of the NVIDIA SM121 m16n8k16 mma.sync version.
//
// C[M,N] = A[M,K] @ B[N,K]^T  (BF16 in, BF16 out, FP32 accumulation)
//
// Tile: 16M × 64N per block. 4 warps, each computes one 16×16 WMMA op.
//   (NVIDIA version did 2× m16n8k16 per warp = 16×16; AMD WMMA n16 does it in 1 op.)
// K loop: iterate in chunks of 16 (WMMA K dimension).
//
// Grid: (ceil(N/64), ceil(M/16), 1)
// Block: (128, 1, 1) = 4 warps
//
// WMMA fragment layout (validated in w4a16_wmma_ref.hip):
//   A load (M×K row-major smem): lane l → a[i] = smem_A[m_row_base + (l&15)][i], i=0..15
//   B load (K×N smem):           lane l → b[k] = smem_B[k][n_base + (l&15)], k=0..15
//   Store: lane l, acc element e(0..7) → C[row_base + 2*e + (l>>4)][col_base + (l&15)]

#include <cuda_bf16.h>

typedef __bf16 v16bf __attribute__((ext_vector_type(16)));
typedef float  v8f   __attribute__((ext_vector_type(8)));

#define TC_TM 16      // M-tile per block (one WMMA M tile)
#define TC_TN 64      // N-tile per block (4 warps × 16 N each)
#define TC_TK 16      // K-tile (WMMA K dimension)
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

    // Each warp owns one 16×16 N-tile at offset warp_id*16.
    const unsigned int n_warp_base = warp_id * 16;

    // Shared memory: A_tile[16][TK+PAD], B_tile[TK][64+PAD]
    __shared__ __nv_bfloat16 smem_A[TC_TM][TC_TK + TC_PAD];
    __shared__ __nv_bfloat16 smem_B[TC_TK][TC_TN + TC_PAD];

    v8f acc = v8f{0, 0, 0, 0, 0, 0, 0, 0};

    // K loop
    for (unsigned int k_base = 0; k_base < K; k_base += TC_TK) {
        // Cooperative load: 128 threads load A[16][16] + B[16][64]
        {
            unsigned int idx = tid;
            // Load A tile: 16*16 = 256 elements, 128 threads → 2 per thread
            if (idx < TC_TM * TC_TK) {
                unsigned int r = idx / TC_TK;
                unsigned int c = idx % TC_TK;
                unsigned int gr = m_block + r;
                unsigned int gc = k_base + c;
                smem_A[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : __float2bfloat16(0.0f);
            }
            // Load B tile: B[N,K] read as B^T[K,N] → smem_B[k][n]
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

        // WMMA: A[16,16] × B^T[16,64] → C[16,64]. This warp does its 16×16 N-tile.
        v16bf a;
        #pragma unroll
        for (int i = 0; i < 16; i++) a[i] = (__bf16)(float)smem_A[lane_id & 15][i];
        v16bf b;
        #pragma unroll
        for (int k = 0; k < 16; k++) b[k] = (__bf16)(float)smem_B[k][n_warp_base + (lane_id & 15)];
        acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc);

        __syncthreads();
    }

    // Write output: WMMA fragment → row-major C[M,N]
    #pragma unroll
    for (int e = 0; e < 8; e++) {
        unsigned int r = m_block + 2 * e + (lane_id >> 4);
        unsigned int c = n_block + n_warp_base + (lane_id & 15);
        if (r < M && c < N) C[r * N + c] = __float2bfloat16(acc[e]);
    }
}
