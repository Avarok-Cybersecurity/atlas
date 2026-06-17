// SPDX-License-Identifier: AGPL-3.0-only

// Atlas W8A16 Dequant+GEMM — Fused FP8-E4M3 weight dequant + BF16 WMMA GEMM.
// HIP/gfx1151 (AMD WMMA) port of the NVIDIA mma.sync version.
//
// C[M,N] = A[M,K] (BF16 activations) * dequant(B[N,K] (FP8 E4M3 weights))
//
// FP8-E4M3 weight format (2D block-scaled):
//   B:           [N, K] uint8 — one byte per weight (FP8 E4M3)
//   block_scale: [N/128, K/128] BF16 — per-block scale factor
//
// Dequant: bf16_val = E4M3_LUT[byte] * block_scale[n/128, k/128]
//
// WMMA: __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32 (16x16x16, n16).
//   One WMMA op == two NVIDIA m16n8k16 n-tiles. A 64-wide N tile = 4 WMMA ops.
//   Store mapping: lane l, acc element e: row = row_base + 2*e + (l>>4), col = col_base + (l&15)
//
// Grid: (ceil(N/64), ceil(M/64), 1), Block: (128, 1, 1)

#include <cuda_bf16.h>

typedef __bf16 v16bf __attribute__((ext_vector_type(16)));
typedef float  v8f   __attribute__((ext_vector_type(8)));

// Tile geometry. A CTA computes a M_TILE×N_TILE output block with a
// THREADS-thread workgroup (THREADS/32 warps, each owning 16 M-rows).
// 256×128 / 512-thread (16-warp) tile: large M reuse + 8 WMMA n-sub-tiles per
// loaded A row gives the best gfx1151 occupancy/intensity for prefill GEMM.
#define M_TILE 256
#define N_TILE 128
#define K_STEP 16
#define PAD 2
#define FP8_BLOCK 128
#define THREADS 512
#define N_SUBTILES (N_TILE / 16)        // 8 WMMA 16×16 n-sub-tiles
#define A_ELEMS_PER_THREAD ((M_TILE * K_STEP) / THREADS)  // 8
#define B_ELEMS_PER_THREAD ((K_STEP * N_TILE) / THREADS)  // 4

// E4M3 lookup table: 256-entry byte → FP32 value.
// Copied from w8a16_gemv.cu (SSOT: same LUT used for both GEMV and GEMM).
__device__ __constant__ float E4M3_LUT[256] = {
    // Positive (0x00..0x7F)
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
    // Negative (0x80..0xFF)
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

// WMMA compute — operates on already-loaded smem_A/smem_B BF16 tiles.
// acc[N_SUBTILES] holds the WMMA n-sub-tiles (N_SUBTILES × 16 = N_TILE).
__device__ __forceinline__ void w8a16_wmma_compute(
    __nv_bfloat16 smem_A[][K_STEP + PAD],
    __nv_bfloat16 smem_B[][N_TILE + PAD],
    v8f acc[N_SUBTILES],
    unsigned int warp_m_offset, unsigned int lane
) {
    v16bf a;
    #pragma unroll
    for (int i = 0; i < 16; i++) a[i] = (__bf16)smem_A[warp_m_offset + (lane & 15)][i];
    #pragma unroll
    for (int nb = 0; nb < N_SUBTILES; nb++) {
        v16bf b;
        #pragma unroll
        for (int k = 0; k < 16; k++) b[k] = (__bf16)smem_B[k][nb * 16 + (lane & 15)];
        acc[nb] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc[nb]);
    }
}

__device__ __forceinline__ void w8a16_wmma_store(
    __nv_bfloat16* __restrict__ C, v8f acc[N_SUBTILES],
    unsigned int cta_m, unsigned int cta_n, unsigned int warp_m_offset,
    unsigned int lane, unsigned int M, unsigned int N
) {
    #pragma unroll
    for (int nb = 0; nb < N_SUBTILES; nb++) {
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            unsigned int row = cta_m + warp_m_offset + 2 * e + (lane >> 4);
            unsigned int col = cta_n + nb * 16 + (lane & 15);
            if (row < M && col < N) C[row * N + col] = __float2bfloat16(acc[nb][e]);
        }
    }
}

/// W8A16 GEMM: B[N, K] row-major FP8 E4M3 with 2D block scales.
//
// Register-prefetch software pipelining (double-buffered, ping-pong smem).
// gfx1151 has NO cp.async / global_load_lds (target feature
// `vmem-to-lds-load-insts` is absent), so the only way to overlap global
// loads with WMMA compute is to issue the VMEM reads for tile k+1 into VGPRs
// (reg_A / reg_B below), let them stay in-flight across the WMMA of tile k,
// then commit those registers into the *other* smem buffer.
//
// LDS budget (gfx1151 cap = 64 KB/workgroup):
//   smem_A[2][256][18] bf16 = 2 * 256 * 18 * 2 = 18432 B
//   smem_B[2][16][130] bf16 = 2 * 16 * 130 * 2 =  8320 B
//   total = 26752 B  (well under 64 KB; both operands double-buffered).
//
// Per-thread prefetch registers: each of the THREADS=512 threads stages
// A_ELEMS_PER_THREAD (8) A elements and B_ELEMS_PER_THREAD (4) dequantized B
// elements per K-tile, covering the full M_TILE×K_STEP and K_STEP×N_TILE tiles.
__device__ __forceinline__ void w8a16_load_A_regs(
    const __nv_bfloat16* __restrict__ A,
    __nv_bfloat16 reg_A[A_ELEMS_PER_THREAD],
    unsigned int cta_m, unsigned int k_base, unsigned int M, unsigned int K
) {
    #pragma unroll
    for (unsigned int i = 0; i < A_ELEMS_PER_THREAD; i++) {
        unsigned int idx = threadIdx.x * A_ELEMS_PER_THREAD + i;
        unsigned int row = idx / K_STEP;
        unsigned int col = idx % K_STEP;
        unsigned int gr = cta_m + row;
        unsigned int gc = k_base + col;
        reg_A[i] = (gr < M && gc < K) ? A[gr * K + gc] : __float2bfloat16(0.0f);
    }
}

__device__ __forceinline__ void w8a16_load_B_regs(
    const unsigned char* __restrict__ B,
    const float* __restrict__ block_scale,
    __nv_bfloat16 reg_B[B_ELEMS_PER_THREAD],
    unsigned int cta_n, unsigned int k_base,
    unsigned int N, unsigned int K, unsigned int k_blocks
) {
    #pragma unroll
    for (unsigned int i = 0; i < B_ELEMS_PER_THREAD; i++) {
        unsigned int idx = threadIdx.x * B_ELEMS_PER_THREAD + i;
        unsigned int k = idx / N_TILE;
        unsigned int n = idx % N_TILE;
        unsigned int gk = k_base + k;
        unsigned int gn = cta_n + n;
        if (gk < K && gn < N) {
            unsigned char weight_byte = B[(unsigned long long)gn * K + gk];
            unsigned int n_block = gn / FP8_BLOCK;
            unsigned int k_block = gk / FP8_BLOCK;
            float scale = block_scale[n_block * k_blocks + k_block];
            float dequant_val = E4M3_LUT[weight_byte] * scale;
            reg_B[i] = __float2bfloat16(dequant_val);
        } else {
            reg_B[i] = __float2bfloat16(0.0f);
        }
    }
}

__device__ __forceinline__ void w8a16_store_A_regs(
    __nv_bfloat16 smem_A[][K_STEP + PAD], const __nv_bfloat16 reg_A[A_ELEMS_PER_THREAD]
) {
    #pragma unroll
    for (unsigned int i = 0; i < A_ELEMS_PER_THREAD; i++) {
        unsigned int idx = threadIdx.x * A_ELEMS_PER_THREAD + i;
        unsigned int row = idx / K_STEP;
        unsigned int col = idx % K_STEP;
        smem_A[row][col] = reg_A[i];
    }
}

__device__ __forceinline__ void w8a16_store_B_regs(
    __nv_bfloat16 smem_B[][N_TILE + PAD], const __nv_bfloat16 reg_B[B_ELEMS_PER_THREAD]
) {
    #pragma unroll
    for (unsigned int i = 0; i < B_ELEMS_PER_THREAD; i++) {
        unsigned int idx = threadIdx.x * B_ELEMS_PER_THREAD + i;
        unsigned int k = idx / N_TILE;
        unsigned int n = idx % N_TILE;
        smem_B[k][n] = reg_B[i];
    }
}

extern "C" __global__ void w8a16_gemm(
    const __nv_bfloat16* __restrict__ A,            // [M, K] BF16 activations
    const unsigned char* __restrict__ B,             // [N, K] FP8 E4M3
    const float* __restrict__ block_scale,   // [N/128, K/128] BF16
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

    // Double-buffered (ping-pong) smem — overlaps tile-(k+1) global loads,
    // staged in VGPRs, with tile-k WMMA compute.
    __shared__ __nv_bfloat16 smem_A[2][M_TILE][K_STEP + PAD];
    __shared__ __nv_bfloat16 smem_B[2][K_STEP][N_TILE + PAD];

    v8f acc[N_SUBTILES];
    #pragma unroll
    for (int i = 0; i < N_SUBTILES; i++) acc[i] = v8f{0, 0, 0, 0, 0, 0, 0, 0};

    const unsigned int k_blocks = K / FP8_BLOCK;
    const unsigned int num_tiles = (K + K_STEP - 1) / K_STEP;

    // ── Prologue: load tile 0 (global → regs → smem[0]) ──
    __nv_bfloat16 reg_A[A_ELEMS_PER_THREAD];
    __nv_bfloat16 reg_B[B_ELEMS_PER_THREAD];
    w8a16_load_A_regs(A, reg_A, cta_m, 0, M, K);
    w8a16_load_B_regs(B, block_scale, reg_B, cta_n, 0, N, K, k_blocks);
    w8a16_store_A_regs(smem_A[0], reg_A);
    w8a16_store_B_regs(smem_B[0], reg_B);
    __syncthreads();

    // ── Steady state: prefetch tile k into regs while computing tile k-1 ──
    //
    // Single barrier per K-step: the ping-pong double-buffer makes a
    // post-compute barrier redundant. The publish barrier below, executed at
    // the end of iteration kt-1, already guarantees every warp finished reading
    // buf[(kt-1)&1] before iteration kt overwrites the *other* buffer buf[kt&1]
    // (whose previous reader was kt-1's compute, fenced by that same barrier).
    for (unsigned int kt = 1; kt < num_tiles; kt++) {
        const unsigned int k_base = kt * K_STEP;
        // (a) issue global reads for tile kt into registers (no wait)
        w8a16_load_A_regs(A, reg_A, cta_m, k_base, M, K);
        w8a16_load_B_regs(B, block_scale, reg_B, cta_n, k_base, N, K, k_blocks);
        // (b) compute the already-resident previous tile
        w8a16_wmma_compute(smem_A[(kt - 1) & 1], smem_B[(kt - 1) & 1], acc, warp_m_offset, lane_id);
        // (c) commit prefetched registers into the other buffer
        w8a16_store_A_regs(smem_A[kt & 1], reg_A);
        w8a16_store_B_regs(smem_B[kt & 1], reg_B);
        // (d) publish smem[kt&1] (also fences this iter's compute-read of the
        //     buffer the *next* iteration will overwrite)
        __syncthreads();
    }

    // ── Epilogue: compute the last tile ──
    w8a16_wmma_compute(smem_A[(num_tiles - 1) & 1], smem_B[(num_tiles - 1) & 1], acc, warp_m_offset, lane_id);

    w8a16_wmma_store(C, acc, cta_m, cta_n, warp_m_offset, lane_id, M, N);
}

/// Standalone W8A16 dequant: B_fp8 → B_bf16 [N, K]  (no tensor core; portable as-is)
/// Each thread handles one FP8 byte → 1 BF16 output.
extern "C" __global__ void w8a16_dequant(
    const unsigned char* __restrict__ B,             // [N, K] FP8 E4M3
    const float* __restrict__ block_scale,   // [N/128, K/128] BF16
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
