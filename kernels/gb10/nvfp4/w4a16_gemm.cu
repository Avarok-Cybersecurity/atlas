// SPDX-License-Identifier: AGPL-3.0-only

// Atlas W4A16 Dequant+GEMM — Fused NVFP4 weight dequant + BF16 Tensor Core GEMM.
//
// C[M,N] = A[M,K] (BF16 activations) * dequant(B_fp4[N,K/2] (packed E2M1 weights))
//
// NVFP4 weight format (HuggingFace/compressed-tensors):
//   B_packed: [N, K/2] uint8 — byte at [n, j] holds W[n, 2j] (low) and W[n, 2j+1] (high)
//   B_scale:  [N, K/group_size] FP8-E4M3 — one scale per group of 16 K-dim values
//   B_scale2: scalar FP32 — per-tensor second-level scale
//
// Dequant: bf16_val = e2m1_to_bf16(nibble) * fp8_scale * scale2
//
// Uses mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
//
// Two variants:
//   w4a16_gemm   — original layout: B_packed[N, K/2], B_scale[N, K/GROUP_SIZE]
//   w4a16_gemm_t — transposed layout: B_packed[K/2, N], B_scale[K/GROUP_SIZE, N]
//                   Coalesced N-dim reads for better LPDDR5X bandwidth.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define M_TILE 64
#define N_TILE 64
#define K_STEP 16
#define PAD 2
#define GROUP_SIZE 16

// E2M1 lookup table: 4-bit index → FP32 value
__device__ __constant__ float E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,   // positive
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f  // negative
};

// MMA compute + store — shared between both layout variants.
// Operates on already-loaded smem_A and smem_B tiles.
__device__ __forceinline__ void w4a16_mma_and_store(
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

/// Original layout: B_packed[N, K/2], B_scale[N, K/GROUP_SIZE]
extern "C" __global__ void w4a16_gemm(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,     // [N, K/2]
    const unsigned char* __restrict__ B_scale,      // [N, K/GROUP_SIZE]
    const float scale2,
    __nv_bfloat16* __restrict__ C,
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

    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;

    for (unsigned int k_base = 0; k_base < K; k_base += K_STEP) {
        // === Load A tile ===
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

        // === Dequant B: original [N, K/2] layout ===
        {
            #pragma unroll
            for (unsigned int i = 0; i < 8; i++) {
                unsigned int idx = threadIdx.x * 8 + i;
                unsigned int k = idx / N_TILE;
                unsigned int n = idx % N_TILE;
                unsigned int gk = k_base + k;
                unsigned int gn = cta_n + n;

                if (gk < K && gn < N) {
                    unsigned int k_pair = gk / 2;
                    unsigned char packed_byte = B_packed[(unsigned long long)gn * half_K + k_pair];
                    unsigned int nibble = (gk & 1) ? (packed_byte >> 4) : (packed_byte & 0xF);

                    unsigned int scale_group = gk / GROUP_SIZE;
                    unsigned char scale_byte = B_scale[(unsigned long long)gn * num_groups + scale_group];
                    __nv_fp8_e4m3 fp8;
                    *(unsigned char*)&fp8 = scale_byte;
                    float dequant_val = E2M1_LUT[nibble] * (float)fp8 * scale2;
                    smem_B[k][n] = __float2bfloat16(dequant_val);
                } else {
                    smem_B[k][n] = __float2bfloat16(0.0f);
                }
            }
        }

        __syncthreads();
        w4a16_mma_and_store(smem_A, smem_B, acc, warp_m_offset, group_id, tid);
        __syncthreads();
    }

    // === Store results ===
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

/// Transposed layout: B_packed[K/2, N], B_scale[K/GROUP_SIZE, N]
/// Coalesced N-dim reads — consecutive threads read consecutive N addresses.
extern "C" __global__ void w4a16_gemm_t(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,     // [K/2, N] transposed
    const unsigned char* __restrict__ B_scale,      // [K/GROUP_SIZE, N] transposed
    const float scale2,
    __nv_bfloat16* __restrict__ C,
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

    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    for (unsigned int k_base = 0; k_base < K; k_base += K_STEP) {
        // === Load A tile ===
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

        // === Dequant B: transposed [K/2, N] layout — coalesced ===
        {
            #pragma unroll
            for (unsigned int i = 0; i < 8; i++) {
                unsigned int idx = threadIdx.x * 8 + i;
                unsigned int k = idx / N_TILE;
                unsigned int n = idx % N_TILE;
                unsigned int gk = k_base + k;
                unsigned int gn = cta_n + n;

                if (gk < K && gn < N) {
                    unsigned int k_pair = gk / 2;
                    unsigned char packed_byte = B_packed[(unsigned long long)k_pair * N + gn];
                    unsigned int nibble = (gk & 1) ? (packed_byte >> 4) : (packed_byte & 0xF);

                    unsigned int scale_group = gk / GROUP_SIZE;
                    unsigned char scale_byte = B_scale[(unsigned long long)scale_group * N + gn];
                    __nv_fp8_e4m3 fp8;
                    *(unsigned char*)&fp8 = scale_byte;
                    float dequant_val = E2M1_LUT[nibble] * (float)fp8 * scale2;
                    smem_B[k][n] = __float2bfloat16(dequant_val);
                } else {
                    smem_B[k][n] = __float2bfloat16(0.0f);
                }
            }
        }

        __syncthreads();
        w4a16_mma_and_store(smem_A, smem_B, acc, warp_m_offset, group_id, tid);
        __syncthreads();
    }

    // === Store results ===
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

/// Standalone W4A16 dequant: B_fp4 → B_bf16 [N, K]
/// Layout: B_packed[N, K/2], K-dim packing.
/// Each thread handles one packed byte → 2 BF16 outputs for consecutive K.
extern "C" __global__ void w4a16_dequant(
    const unsigned char* __restrict__ B_packed,     // [N, K/2] uint8
    const unsigned char* __restrict__ B_scale,      // [N, K/GROUP_SIZE] FP8-E4M3
    const float scale2,
    __nv_bfloat16* __restrict__ B_bf16,             // [N, K] BF16 output
    unsigned int K,
    unsigned int N
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_bytes = N * (K / 2);
    if (idx >= total_bytes) return;

    unsigned int n = idx / (K / 2);
    unsigned int k_pair = idx % (K / 2);
    unsigned int k0 = k_pair * 2;
    unsigned int k1 = k0 + 1;

    unsigned char packed = B_packed[idx];
    unsigned int nib0 = packed & 0xF;  // W[n, k0]
    unsigned int nib1 = packed >> 4;   // W[n, k1]

    unsigned int num_groups = K / GROUP_SIZE;
    unsigned int scale_group = k0 / GROUP_SIZE;
    // Both k0 and k1 are in the same group (consecutive K values)
    unsigned char s_byte = B_scale[(unsigned long long)n * num_groups + scale_group];

    __nv_fp8_e4m3 fp8;
    *(unsigned char*)&fp8 = s_byte;
    float s = (float)fp8 * scale2;

    float v0 = E2M1_LUT[nib0] * s;
    float v1 = E2M1_LUT[nib1] * s;

    B_bf16[n * K + k0] = __float2bfloat16(v0);
    if (k1 < K) B_bf16[n * K + k1] = __float2bfloat16(v1);
}
