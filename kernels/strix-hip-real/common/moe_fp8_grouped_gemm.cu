// SPDX-License-Identifier: AGPL-3.0-only

// Atlas FP8 Grouped MoE GEMM — gfx1151 (RDNA3.5, wave32) HIP/WMMA port.
//
// Faithful port of kernels/gb10/common/moe_fp8_grouped_gemm.cu (NVIDIA
// mma.sync m16n8k16) to AMD WMMA (__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32)
// using the proven gfx1151 idiom from kernels/strix-hip-real/common/w8a16_gemm.cu.
//
//   C[M_expert,N] = A[M_expert,K] (BF16) @ dequant(B_expert[N,K] (FP8 E4M3))
//
// Each CTA processes one (expert, m_tile, n_tile) block. Expert weights are
// accessed via pointer tables indexed by expert_id. Tokens are sorted by
// expert so each expert's tokens are contiguous.
//
// FP8 weight format: B[N,K] uint8 with block_scale[N/128, K/128] FP32.
//   (scale_inv is widened to FP32 at load; applied in full FP32 precision —
//    the per-expert scale pointer table entries are `const float*`.)
//   Dequant: bf16_val = E4M3_LUT[byte] * block_scale[n/128, k/128]
//
// Numerics SSOT: identical two-level FP32 accumulation to gb10. inner_acc
// accumulates unscaled BF16(LUT) products within one K=128 scale-block;
// outer_acc applies the FP32 scale once per K-block ((a+b)*s == a*s + b*s).
// Lossless BF16(LUT) cast: FP8 E4M3 has 3 mantissa bits, BF16 has 7.
//
// gfx1151 mapping vs gb10:
//   gb10 uses mma.sync.m16n8k16 (one CTA warp pair → m16). Two NVIDIA
//   m16n8k16 n-tiles == one AMD WMMA 16x16x16 n16 op; a 64-wide N tile = 4
//   WMMA ops (acc[4]). The AMD C-fragment store mapping (validated in
//   w8a16_gemm.cu) is: lane l, acc element e ->
//       row = m_base + 2*e + (l>>4),  col = n_base + nb*16 + (l&15)
//   This produces byte-identical output positions to gb10's combined
//   {row0,row1}×{col0,col1}×n_tile fragment store (same C[out_row*N+col]).
//
// Grid: (ceil(N/64), max_m_tiles, num_experts)  Block: (128, 1, 1)

#include <cuda_bf16.h>

typedef __bf16 v16bf __attribute__((ext_vector_type(16)));
typedef float  v8f   __attribute__((ext_vector_type(8)));

#define M_TILE 64
#define N_TILE 64
#define K_STEP 16
#define PAD 2
#define FP8_BLOCK 128
// Inner-promotion stride: applies the scale at K_PROMOTE granularity within a
// scale-block. Mathematically identical for FP32 accumulators; mirrors gb10.
#define K_PROMOTE 64
#define WARP_SIZE 32

__device__ __constant__ float E4M3_LUT_GMOE[256] = {
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

// WMMA compute — accumulates one K_STEP=16 slice of BF16×BF16 products into
// the 4 WMMA n-sub-tiles (4 × 16 = N_TILE=64). Mirrors w8a16_wmma_compute.
//   A fragment: 16 contiguous K values of the warp's m-row (lane & 15).
//   B fragment: 16 contiguous K values of column (nb*16 + (lane & 15)).
__device__ __forceinline__ void fp8_moe_wmma_compute(
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

/// FP8 grouped GEMM for sorted MoE dispatch.
///
/// BF16 activations × FP8 E4M3 block-scaled weights per expert.
/// Expert weights accessed via pointer tables. Tokens sorted by expert.
///
/// Grid: (ceil(N/64), max_m_tiles, num_experts)  Block: (128, 1, 1)
extern "C" __global__ void moe_fp8_grouped_gemm(
    const __nv_bfloat16* __restrict__ A,                    // [total_tokens, K] BF16
    const unsigned long long* __restrict__ B_weight_ptrs,   // [num_experts] → [N, K] FP8
    const unsigned long long* __restrict__ B_scale_ptrs,    // [num_experts] → [N/128, K/128] FP32
    __nv_bfloat16* __restrict__ C,                          // [total_expanded, N] BF16
    const int* __restrict__ expert_offsets,                 // [num_experts + 1]
    const int* __restrict__ sorted_token_ids,               // [total_expanded]
    unsigned int num_experts,
    unsigned int N,
    unsigned int K
) {
    const unsigned int expert_id = blockIdx.z;
    if (expert_id >= num_experts) return;

    const int m_start = expert_offsets[expert_id];
    const int m_end = expert_offsets[expert_id + 1];
    const int M_expert = m_end - m_start;
    if (M_expert <= 0) return;

    const int cta_m_local = blockIdx.y * M_TILE;
    if (cta_m_local >= M_expert) return;

    const unsigned int cta_n = blockIdx.x * N_TILE;

    // Per-expert weight/scale pointers. Scale is FP32 (loader produces FP32).
    const unsigned char* B_exp = (const unsigned char*)B_weight_ptrs[expert_id];
    const float* S_exp = (const float*)B_scale_ptrs[expert_id];
    if (B_exp == 0) return;  // NULL → remote expert under EP

    const unsigned int k_blocks = (K + FP8_BLOCK - 1) / FP8_BLOCK;

    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    const unsigned int lane_id = threadIdx.x % WARP_SIZE;
    const unsigned int warp_m_offset = warp_id * 16;

    __shared__ __nv_bfloat16 smem_A[M_TILE][K_STEP + PAD];
    __shared__ __nv_bfloat16 smem_B[K_STEP][N_TILE + PAD];

    // Two-level FP32 accumulation (gb10 SSOT): inner accumulates unscaled
    // BF16(LUT) products within one K=128 scale-block; outer applies scale.
    v8f outer_acc[4];
    v8f inner_acc[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        outer_acc[i] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
        inner_acc[i] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
    }

    // N_TILE=64 < FP8_BLOCK=128, cta_n aligned to N_TILE — all 64 N-cols of
    // this CTA fall within a single N-block and share one scale per K-block.
    const unsigned int n_block = cta_n / FP8_BLOCK;

    for (unsigned int k_base = 0; k_base < K; k_base += K_STEP) {
        // Load A tile: gather from sorted token positions.
        {
            #pragma unroll
            for (unsigned int i = 0; i < 8; i++) {
                unsigned int idx = threadIdx.x * 8 + i;
                unsigned int row = idx / K_STEP;
                unsigned int col = idx % K_STEP;
                unsigned int m_idx = cta_m_local + row;
                unsigned int gc = k_base + col;

                if (m_idx < (unsigned int)M_expert && gc < K) {
                    int sorted_idx = m_start + m_idx;
                    // NULL sorted_token_ids → direct (already sorted) indexing.
                    int token_id = sorted_token_ids ? sorted_token_ids[sorted_idx] : sorted_idx;
                    smem_A[row][col] = A[(unsigned long long)token_id * K + gc];
                } else {
                    smem_A[row][col] = __float2bfloat16(0.0f);
                }
            }
        }

        // Dequant B tile: FP8 E4M3 → BF16 via LUT (NO scale — applied post-MMA
        // to inner_acc). Lossless: FP8 3-bit mantissa < BF16 7-bit mantissa.
        {
            #pragma unroll
            for (unsigned int i = 0; i < 8; i++) {
                unsigned int idx = threadIdx.x * 8 + i;
                unsigned int k = idx / N_TILE;
                unsigned int n = idx % N_TILE;
                unsigned int gk = k_base + k;
                unsigned int gn = cta_n + n;

                if (gk < K && gn < N) {
                    unsigned char weight_byte = B_exp[(unsigned long long)gn * K + gk];
                    smem_B[k][n] = __float2bfloat16(E4M3_LUT_GMOE[weight_byte]);
                } else {
                    smem_B[k][n] = __float2bfloat16(0.0f);
                }
            }
        }

        __syncthreads();
        fp8_moe_wmma_compute(smem_A, smem_B, inner_acc, warp_m_offset, lane_id);
        __syncthreads();

        // End-of-K-block: scale inner_acc, accumulate to outer_acc, reset.
        unsigned int next_k = k_base + K_STEP;
        if (next_k % K_PROMOTE == 0 || next_k >= K) {
            unsigned int k_block = k_base / FP8_BLOCK;
            float scale = S_exp[n_block * k_blocks + k_block];
            #pragma unroll
            for (int nb = 0; nb < 4; nb++) {
                #pragma unroll
                for (int e = 0; e < 8; e++) {
                    outer_acc[nb][e] += inner_acc[nb][e] * scale;
                    inner_acc[nb][e] = 0.0f;
                }
            }
        }
    }

    // Store C tile — write to sorted position in output (from outer_acc).
    // AMD WMMA store mapping: lane l, acc element e ->
    //   row = m_base + 2*e + (l>>4),  col = n_base + nb*16 + (l&15)
    #pragma unroll
    for (int nb = 0; nb < 4; nb++) {
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            unsigned int row_local = warp_m_offset + 2 * e + (lane_id >> 4);
            unsigned int col = cta_n + nb * 16 + (lane_id & 15);
            if (row_local < (unsigned int)M_expert && col < N) {
                unsigned int out_row = m_start + row_local;
                C[(unsigned long long)out_row * N + col] = __float2bfloat16(outer_acc[nb][e]);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Coalesced-load variant — same WMMA logic, new thread mapping on A/B smem
// staging so neighbouring threads in a warp hit contiguous global memory.
// Mirrors gb10 moe_fp8_grouped_gemm_v2. Same output semantics as v1.
// ═══════════════════════════════════════════════════════════════════
extern "C" __global__ void moe_fp8_grouped_gemm_v2(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ B_weight_ptrs,
    const unsigned long long* __restrict__ B_scale_ptrs,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    unsigned int num_experts,
    unsigned int N,
    unsigned int K
) {
    const unsigned int expert_id = blockIdx.z;
    if (expert_id >= num_experts) return;

    const int m_start = expert_offsets[expert_id];
    const int m_end = expert_offsets[expert_id + 1];
    const int M_expert = m_end - m_start;
    if (M_expert <= 0) return;

    const int cta_m_local = blockIdx.y * M_TILE;
    if (cta_m_local >= M_expert) return;

    const unsigned int cta_n = blockIdx.x * N_TILE;

    const unsigned char* B_exp = (const unsigned char*)B_weight_ptrs[expert_id];
    const float* S_exp = (const float*)B_scale_ptrs[expert_id];
    if (B_exp == 0) return;

    const unsigned int k_blocks = (K + FP8_BLOCK - 1) / FP8_BLOCK;

    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    const unsigned int lane_id = threadIdx.x % WARP_SIZE;
    const unsigned int warp_m_offset = warp_id * 16;

    __shared__ __nv_bfloat16 smem_A[M_TILE][K_STEP + PAD];
    __shared__ __nv_bfloat16 smem_B[K_STEP][N_TILE + PAD];

    // Coalesced mapping: 8 groups × 16 threads.
    const unsigned int thread_group = threadIdx.x >> 4;      // 0..7
    const unsigned int k_offset     = threadIdx.x & 15;      // 0..K_STEP-1
    const unsigned int row_base     = thread_group * 8;      // 0, 8, ..., 56

    v8f outer_acc[4];
    v8f inner_acc[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        outer_acc[i] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
        inner_acc[i] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
    }

    const unsigned int n_block = cta_n / FP8_BLOCK;

    for (unsigned int k_base = 0; k_base < K; k_base += K_STEP) {
        const unsigned int gk = k_base + k_offset;

        // Load A tile [M_TILE=64][K_STEP=16]: 16 threads of a group share one
        // m-row and vary k_offset 0..15 — one coalesced burst per group.
        #pragma unroll
        for (unsigned int i = 0; i < 8; i++) {
            unsigned int local_row  = row_base + i;
            unsigned int m_global   = cta_m_local + local_row;
            if (m_global < (unsigned int)M_expert && gk < K) {
                int sorted_idx = m_start + m_global;
                int token_id = sorted_token_ids ? sorted_token_ids[sorted_idx] : sorted_idx;
                smem_A[local_row][k_offset] = A[(unsigned long long)token_id * K + gk];
            } else {
                smem_A[local_row][k_offset] = __float2bfloat16(0.0f);
            }
        }

        // Dequant B tile [K_STEP=16][N_TILE=64] — NO scale (applied post-MMA).
        #pragma unroll
        for (unsigned int i = 0; i < 8; i++) {
            unsigned int n_local = row_base + i;
            unsigned int gn = cta_n + n_local;
            if (gk < K && gn < N) {
                unsigned char weight_byte = B_exp[(unsigned long long)gn * K + gk];
                smem_B[k_offset][n_local] = __float2bfloat16(E4M3_LUT_GMOE[weight_byte]);
            } else {
                smem_B[k_offset][n_local] = __float2bfloat16(0.0f);
            }
        }

        __syncthreads();
        fp8_moe_wmma_compute(smem_A, smem_B, inner_acc, warp_m_offset, lane_id);
        __syncthreads();

        // End-of-K-block: scale inner_acc → outer_acc, reset inner.
        unsigned int next_k = k_base + K_STEP;
        if (next_k % K_PROMOTE == 0 || next_k >= K) {
            unsigned int k_block = k_base / FP8_BLOCK;
            float scale = S_exp[n_block * k_blocks + k_block];
            #pragma unroll
            for (int nb = 0; nb < 4; nb++) {
                #pragma unroll
                for (int e = 0; e < 8; e++) {
                    outer_acc[nb][e] += inner_acc[nb][e] * scale;
                    inner_acc[nb][e] = 0.0f;
                }
            }
        }
    }

    // Store C tile from outer_acc — same AMD WMMA mapping as v1.
    #pragma unroll
    for (int nb = 0; nb < 4; nb++) {
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            unsigned int row_local = warp_m_offset + 2 * e + (lane_id >> 4);
            unsigned int col = cta_n + nb * 16 + (lane_id & 15);
            if (row_local < (unsigned int)M_expert && col < N) {
                unsigned int out_row = m_start + row_local;
                C[(unsigned long long)out_row * N + col] = __float2bfloat16(outer_acc[nb][e]);
            }
        }
    }
}
