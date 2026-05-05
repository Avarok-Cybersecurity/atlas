// SPDX-License-Identifier: AGPL-3.0-only

// Atlas W8A16 Transposed GEMM — FP8 E4M3 block-scaled with coalesced weight reads.
//
// C[M,N] = A[M,K] (BF16) * dequant(B_t[K,N] (FP8 E4M3, transposed at load time))
//
// Key optimization over w8a16_gemm: weights stored as B_t[K, N] instead of B[N, K].
// This makes the N-dimension contiguous in memory, enabling coalesced 128-byte reads
// when 128 threads each read one N-element in the same K-row.
//
// Block scales: block_scale_t[K/128, N/128] BF16 (also transposed)
// Dequant: bf16_val = E4M3_LUT[byte] * block_scale_t[k/128, n/128]
//
// Uses mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
// Grid: (ceil(N/N_TILE), ceil(M/M_TILE), 1), Block: (128, 1, 1)

#include <cuda_bf16.h>

#define M_TILE 64
#define N_TILE 128   // 2x wider N-tile: reads 128 N elements per K-step
#define K_STEP 32    // 2x deeper K-step: better compute-to-memory ratio
#define PAD 2
#define FP8_BLOCK 128

// E4M3 lookup table
__device__ __constant__ float E4M3_LUT_T[256] = {
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

// MMA compute: shared between both tile halves.
// smem_A: [M_TILE][K_STEP_HALF+PAD], smem_B: [K_STEP_HALF][N_TILE_HALF+PAD]
__device__ __forceinline__ void w8a16_mma_and_store_t(
    __nv_bfloat16 smem_A[][16 + PAD],
    __nv_bfloat16 smem_B[][64 + PAD],
    float acc[8][4],
    unsigned int warp_m_offset, unsigned int group_id, unsigned int tid
) {
    const unsigned int a_stride = 16 + PAD;
    const unsigned int b_stride = 64 + PAD;
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

/// W8A16 GEMM with transposed weight layout for coalesced reads.
///
/// B_t: [K, N] FP8 E4M3 — transposed from checkpoint's [N, K]
/// block_scale_t: [K/128, N/128] BF16 — transposed block scales
///
/// Thread mapping: 128 threads load a [16, 64] B-tile where the
/// 64 N-elements are contiguous in memory → coalesced 128-byte reads.
extern "C" __global__ void w8a16_gemm_t(
    const __nv_bfloat16* __restrict__ A,               // [M, K] BF16
    const unsigned char* __restrict__ B_t,              // [K, N] FP8 E4M3 transposed
    const __nv_bfloat16* __restrict__ block_scale_t,   // [K/128, N/128] BF16 transposed
    __nv_bfloat16* __restrict__ C,                     // [M, N] BF16
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int cta_m = blockIdx.y * M_TILE;
    const unsigned int cta_n = blockIdx.x * 64;  // Each CTA handles 64 N columns
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[M_TILE][16 + PAD];
    __shared__ __nv_bfloat16 smem_B[16][64 + PAD];

    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int n_scale_blocks = (N + FP8_BLOCK - 1) / FP8_BLOCK;

    for (unsigned int k_base = 0; k_base < K; k_base += 16) {
        // === Load A tile: [M_TILE, 16] BF16 from global → smem ===
        {
            // 128 threads, M_TILE*16 = 1024 elements → 8 per thread
            #pragma unroll
            for (unsigned int i = 0; i < 8; i++) {
                unsigned int idx = threadIdx.x * 8 + i;
                unsigned int row = idx / 16;
                unsigned int col = idx % 16;
                unsigned int gr = cta_m + row;
                unsigned int gc = k_base + col;
                smem_A[row][col] = (gr < M && gc < K) ? A[gr * K + gc] : __float2bfloat16(0.0f);
            }
        }

        // === Dequant B: transposed [K, N] layout — coalesced N reads ===
        // B_t tile: [16, 64] = 1024 elements, 128 threads → 8 per thread.
        // Layout: each thread reads from B_t[gk * N + gn] where gn varies
        // within a warp → adjacent threads access adjacent N addresses.
        {
            #pragma unroll
            for (unsigned int i = 0; i < 8; i++) {
                unsigned int idx = threadIdx.x * 8 + i;
                unsigned int k = idx / 64;       // 0..15 (K dimension)
                unsigned int n = idx % 64;       // 0..63 (N dimension, contiguous)
                unsigned int gk = k_base + k;
                unsigned int gn = cta_n + n;

                if (gk < K && gn < N) {
                    // Coalesced read: adjacent threads read adjacent N addresses
                    unsigned char weight_byte = B_t[(unsigned long long)gk * N + gn];

                    // Block scale: transposed layout [K/128, N/128]
                    unsigned int k_block = gk / FP8_BLOCK;
                    unsigned int n_block = gn / FP8_BLOCK;
                    float scale = __bfloat162float(block_scale_t[k_block * n_scale_blocks + n_block]);

                    float dequant_val = E4M3_LUT_T[weight_byte] * scale;
                    smem_B[k][n] = __float2bfloat16(dequant_val);
                } else {
                    smem_B[k][n] = __float2bfloat16(0.0f);
                }
            }
        }

        __syncthreads();
        w8a16_mma_and_store_t(smem_A, smem_B, acc, warp_m_offset, group_id, tid);
        __syncthreads();
    }

    // === Store C tile: f32 accumulators → BF16 output ===
    #pragma unroll
    for (int n_tile = 0; n_tile < 8; n_tile++) {
        unsigned int base_n = cta_n + n_tile * 8;
        unsigned int col0 = base_n + (tid * 2);
        unsigned int col1 = col0 + 1;
        unsigned int row0 = cta_m + warp_m_offset + group_id;
        unsigned int row1 = row0 + 8;

        if (row0 < M && col0 < N) C[row0 * N + col0] = __float2bfloat16(acc[n_tile][0]);
        if (row0 < M && col1 < N) C[row0 * N + col1] = __float2bfloat16(acc[n_tile][1]);
        if (row1 < M && col0 < N) C[row1 * N + col0] = __float2bfloat16(acc[n_tile][2]);
        if (row1 < M && col1 < N) C[row1 * N + col1] = __float2bfloat16(acc[n_tile][3]);
    }
}

/// Transpose FP8 weight matrix: B[N,K] → B_t[K,N]
/// Each thread transposes one element.
extern "C" __global__ void transpose_fp8(
    const unsigned char* __restrict__ B,      // [N, K] FP8 E4M3
    unsigned char* __restrict__ B_t,          // [K, N] transposed
    unsigned int N,
    unsigned int K
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = N * K;
    if (idx >= total) return;

    unsigned int n = idx / K;
    unsigned int k = idx % K;
    B_t[(unsigned long long)k * N + n] = B[(unsigned long long)n * K + k];
}

/// Transpose block scales: scale[N/128, K/128] → scale_t[K/128, N/128]
extern "C" __global__ void transpose_block_scale(
    const __nv_bfloat16* __restrict__ scale,        // [N/128, K/128]
    __nv_bfloat16* __restrict__ scale_t,            // [K/128, N/128]
    unsigned int N_blocks,    // N/128
    unsigned int K_blocks     // K/128
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = N_blocks * K_blocks;
    if (idx >= total) return;

    unsigned int nb = idx / K_blocks;
    unsigned int kb = idx % K_blocks;
    scale_t[kb * N_blocks + nb] = scale[nb * K_blocks + kb];
}
