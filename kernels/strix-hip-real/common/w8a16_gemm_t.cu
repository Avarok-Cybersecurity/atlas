// SPDX-License-Identifier: AGPL-3.0-only

// Atlas W8A16 Transposed GEMM — FP8 E4M3 block-scaled, coalesced weight reads.
// HIP/gfx1151 (AMD WMMA) port of the NVIDIA mma.sync version. Mirrors the
// proven idiom in w8a16_gemm.cu (the BF16 WMMA GEMM port).
//
// C[M,N] = A[M,K] (BF16) * dequant(B_t[K,N] (FP8 E4M3, transposed))
//
// Weights stored as B_t[K, N] so the N-dimension is contiguous (coalesced reads).
// Block scales: block_scale_t[K/128, N/128] BF16. Dequant: LUT[byte] * scale.
//
// WMMA: __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32 (16x16x16, n16).
//   One WMMA op == two NVIDIA m16n8k16 n-tiles. A 64-wide N tile = 4 WMMA ops.
//   Store: lane l, acc element e: row = row_base + 2*e + (l>>4), col = col_base + (l&15)
//
// Grid: (ceil(N/64), ceil(M/64), 1), Block: (128, 1, 1)

#include <cuda_bf16.h>

typedef __bf16 v16bf __attribute__((ext_vector_type(16)));
typedef float  v8f   __attribute__((ext_vector_type(8)));

#define M_TILE 64
#define K_STEP 16
#define N_TILE 64       // 64 N columns per CTA (matches w8a16_gemm.cu)
#define PAD 2
#define FP8_BLOCK 128

// E4M3 lookup table (SSOT with w8a16_gemm.cu / w8a16_gemv.cu).
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

// WMMA compute over loaded smem_A[M_TILE][K_STEP+PAD], smem_B[K_STEP][N_TILE+PAD].
// acc[4] = 4 WMMA n16-tiles (64 N). Identical to w8a16_gemm.cu's compute.
__device__ __forceinline__ void w8a16_wmma_compute_t(
    __nv_bfloat16 smem_A[][K_STEP + PAD],
    __nv_bfloat16 smem_B[][N_TILE + PAD],
    v8f acc[4],
    unsigned int warp_m_offset, unsigned int lane
) {
    v16bf a;
    #pragma unroll
    for (int i = 0; i < 16; i++) a[i] = (__bf16)smem_A[warp_m_offset + (lane & 15)][i];
    #pragma unroll
    for (int nb = 0; nb < 4; nb++) {
        v16bf b;
        #pragma unroll
        for (int k = 0; k < 16; k++) b[k] = (__bf16)smem_B[k][nb * 16 + (lane & 15)];
        acc[nb] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc[nb]);
    }
}

/// W8A16 GEMM with transposed weight layout for coalesced reads.
/// B_t: [K, N] FP8 E4M3. block_scale_t: [K/128, N/128] BF16.
extern "C" __global__ void w8a16_gemm_t(
    const __nv_bfloat16* __restrict__ A,               // [M, K] BF16
    const unsigned char* __restrict__ B_t,              // [K, N] FP8 E4M3 transposed
    const float* __restrict__ block_scale_t,   // [K/128, N/128] BF16
    __nv_bfloat16* __restrict__ C,                     // [M, N] BF16
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int cta_m = blockIdx.y * M_TILE;
    const unsigned int cta_n = blockIdx.x * N_TILE;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;

    __shared__ __nv_bfloat16 smem_A[M_TILE][K_STEP + PAD];
    __shared__ __nv_bfloat16 smem_B[K_STEP][N_TILE + PAD];

    v8f acc[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) acc[i] = v8f{0, 0, 0, 0, 0, 0, 0, 0};

    const unsigned int n_scale_blocks = (N + FP8_BLOCK - 1) / FP8_BLOCK;

    for (unsigned int k_base = 0; k_base < K; k_base += K_STEP) {
        // === Load A tile: [M_TILE, K_STEP] BF16 ===
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

        // === Dequant B_t: transposed [K, N] — coalesced N reads ===
        {
            #pragma unroll
            for (unsigned int i = 0; i < 8; i++) {
                unsigned int idx = threadIdx.x * 8 + i;
                unsigned int k = idx / N_TILE;       // 0..15 (K)
                unsigned int n = idx % N_TILE;       // 0..63 (N, contiguous)
                unsigned int gk = k_base + k;
                unsigned int gn = cta_n + n;
                if (gk < K && gn < N) {
                    unsigned char weight_byte = B_t[(unsigned long long)gk * N + gn];
                    unsigned int k_block = gk / FP8_BLOCK;
                    unsigned int n_block = gn / FP8_BLOCK;
                    float scale = block_scale_t[k_block * n_scale_blocks + n_block];
                    float dequant_val = E4M3_LUT_T[weight_byte] * scale;
                    smem_B[k][n] = __float2bfloat16(dequant_val);
                } else {
                    smem_B[k][n] = __float2bfloat16(0.0f);
                }
            }
        }

        __syncthreads();
        w8a16_wmma_compute_t(smem_A, smem_B, acc, warp_m_offset, lane_id);
        __syncthreads();
    }

    // === Store C tile ===
    #pragma unroll
    for (int nb = 0; nb < 4; nb++) {
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            unsigned int row = cta_m + warp_m_offset + 2 * e + (lane_id >> 4);
            unsigned int col = cta_n + nb * 16 + (lane_id & 15);
            if (row < M && col < N) C[row * N + col] = __float2bfloat16(acc[nb][e]);
        }
    }
}

/// Transpose FP8 weight matrix: B[N,K] → B_t[K,N]. One element per thread.
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
    const float* __restrict__ scale,        // [N/128, K/128]
    float* __restrict__ scale_t,            // [K/128, N/128]
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
