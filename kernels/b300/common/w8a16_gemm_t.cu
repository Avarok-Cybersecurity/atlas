// SPDX-License-Identifier: AGPL-3.0-only

// Atlas W8A16 Transposed GEMM — FP8 E4M3 block-scaled with coalesced weight reads.
//
// C[M,N] = A[M,K] (BF16) * dequant(B_t[K,N] (FP8 E4M3, transposed at load time))
//
// Key optimization over w8a16_gemm: weights stored as B_t[K, N] instead of B[N, K].
// This makes the N-dimension contiguous in memory, enabling coalesced 128-byte reads
// when 128 threads each read one N-element in the same K-row.
//
// Block scales: block_scale_t[K/128, N/128] FP32 (also transposed; the
// scale_inv is widened to FP32 at load and transposed as FP32 here)
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
/// block_scale_t: [K/128, N/128] FP32 — transposed block scales
///
/// Thread mapping: 128 threads load a [16, 64] B-tile where the
/// 64 N-elements are contiguous in memory → coalesced 128-byte reads.
extern "C" __global__ void w8a16_gemm_t(
    const __nv_bfloat16* __restrict__ A,               // [M, K] BF16
    const unsigned char* __restrict__ B_t,              // [K, N] FP8 E4M3 transposed
    const float* __restrict__ block_scale_t,           // [K/128, N/128] FP32 transposed
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

    // Two-level FP32 accumulation (vLLM W8A8 / DeepGEMM pattern). See
    // w8a16_gemm.cu for the design rationale. n_block constant per CTA
    // because N_TILE=64 ≤ FP8_BLOCK=128 and cta_n is N_TILE-aligned.
    float inner_acc[8][4];
    float outer_acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        inner_acc[i][0] = 0.0f; inner_acc[i][1] = 0.0f;
        inner_acc[i][2] = 0.0f; inner_acc[i][3] = 0.0f;
        outer_acc[i][0] = 0.0f; outer_acc[i][1] = 0.0f;
        outer_acc[i][2] = 0.0f; outer_acc[i][3] = 0.0f;
    }

    const unsigned int n_scale_blocks = (N + FP8_BLOCK - 1) / FP8_BLOCK;
    const unsigned int k_step_inner = 16;
    const unsigned int k_steps_per_block = FP8_BLOCK / k_step_inner;
    const unsigned int n_block = cta_n / FP8_BLOCK;
    unsigned int k_step_in_block = 0;

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
        // Store LUT[byte] as BF16 (lossless cast); apply block scale on the
        // FP32 outer accumulator at each K_BLOCK boundary below.
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
                    smem_B[k][n] = __float2bfloat16(E4M3_LUT_T[weight_byte]);
                } else {
                    smem_B[k][n] = __float2bfloat16(0.0f);
                }
            }
        }

        __syncthreads();
        w8a16_mma_and_store_t(smem_A, smem_B, inner_acc, warp_m_offset, group_id, tid);
        __syncthreads();

        // K_BLOCK boundary: fold scaled inner into outer, reset inner.
        k_step_in_block++;
        if (k_step_in_block == k_steps_per_block) {
            const unsigned int k_block = k_base / FP8_BLOCK;
            // Transposed scale layout: [K/128, N/128]
            const float scale = block_scale_t[k_block * n_scale_blocks + n_block];
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                outer_acc[i][0] += inner_acc[i][0] * scale;
                outer_acc[i][1] += inner_acc[i][1] * scale;
                outer_acc[i][2] += inner_acc[i][2] * scale;
                outer_acc[i][3] += inner_acc[i][3] * scale;
                inner_acc[i][0] = 0.0f; inner_acc[i][1] = 0.0f;
                inner_acc[i][2] = 0.0f; inner_acc[i][3] = 0.0f;
            }
            k_step_in_block = 0;
        }
    }

    // Fold any incomplete trailing K_BLOCK (K % FP8_BLOCK != 0; dead code
    // for Qwen3.6 hidden=2048 but keeps the kernel general).
    if (k_step_in_block != 0) {
        const unsigned int k_block = (K - 1) / FP8_BLOCK;
        const float scale = block_scale_t[k_block * n_scale_blocks + n_block];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            outer_acc[i][0] += inner_acc[i][0] * scale;
            outer_acc[i][1] += inner_acc[i][1] * scale;
            outer_acc[i][2] += inner_acc[i][2] * scale;
            outer_acc[i][3] += inner_acc[i][3] * scale;
        }
    }

    // === Store C tile: f32 outer accumulators → BF16 output ===
    #pragma unroll
    for (int n_tile = 0; n_tile < 8; n_tile++) {
        unsigned int base_n = cta_n + n_tile * 8;
        unsigned int col0 = base_n + (tid * 2);
        unsigned int col1 = col0 + 1;
        unsigned int row0 = cta_m + warp_m_offset + group_id;
        unsigned int row1 = row0 + 8;

        if (row0 < M && col0 < N) C[row0 * N + col0] = __float2bfloat16(outer_acc[n_tile][0]);
        if (row0 < M && col1 < N) C[row0 * N + col1] = __float2bfloat16(outer_acc[n_tile][1]);
        if (row1 < M && col0 < N) C[row1 * N + col0] = __float2bfloat16(outer_acc[n_tile][2]);
        if (row1 < M && col1 < N) C[row1 * N + col1] = __float2bfloat16(outer_acc[n_tile][3]);
    }
}

// ============================================================================
// w8a16_gemm_t_pipelined — occupancy-tuned tensor-core rewrite of w8a16_gemm_t
// ============================================================================
//
// Same TRANSPOSED contract + same number formats as w8a16_gemm_t above, but
// applies the proven w8a16_gemm_pipelined playbook (commits dd7d7bd/4bb1cca/
// 9a41575) adapted to the transposed B_t[K,N] + block_scale_t[K/128,N/128]
// layout:
//
//   Lever 1 (biggest): SHARED-memory E4M3 LUT, not __constant__. The dequant
//     indexes the table with DATA-DEPENDENT weight bytes; __constant__ memory
//     is a broadcast cache that SERIALIZES across a warp on divergent indices.
//     The original w8a16_gemm_t (above) still pays this. We stage the 256-entry
//     table into smem once per CTA; smem services divergent indices in parallel.
//   Lever 2: K_STEP=32 (two m16n8k16 sub-MMAs per resident step) amortizes the
//     per-K-step barrier triple over 2× the MMA work.
//   Lever 3: MMA-ready smem_B stored [n][k] (K-CONTIGUOUS) so the B fragment's
//     two consecutive-K BF16 elements are a SINGLE aligned 32-bit smem load.
//   Lever 4 (occupancy): 128×32 output tile (8 warps) keeps the per-thread
//     two-level FP32 accumulator small enough to co-resident multiple CTAs/SM.
//   cp.async.cg multistage prefetch (sm_80+, correct on sm_121; NO TMA/bulk).
//
// TRANSPOSED-SPECIFIC TWIST: B_t[K,N] is N-contiguous, but the MMA fragment
// wants K-contiguous. We resolve the tension by loading N-contiguous chunks
// (COALESCED 16-byte cp.async: 16 adjacent N-bytes of one K-row) into a raw
// smem buffer laid out [k][n], then TRANSPOSING during the cooperative dequant
// (free — dequant already touches every element) into the MMA-ready [n][k]
// K-contiguous smem_B. This gives coalesced global loads AND fast 32-bit MMA
// fragment loads AND the shared-memory LUT, all at once.
//
// Two-level FP32 accumulation PRESERVED EXACTLY (deep-layer FP8 floor):
//   inner accumulates UNSCALED BF16-cast E4M3 weights across the 8 K-steps of
//   one 128-K block; at each block boundary outer += inner * block_scale_t.
//
// Grid: (ceil(N/PT_N_TILE), ceil(M/PT_M_TILE), 1), Block: (256,1,1).

#include "e4m3_lut.cuh"   // shared E4M3_LUT SSOT (also used by non-transposed sibling)

#define PT_M_TILE 128
#define PT_N_TILE 32                          // occupancy sweet spot (see sibling note)
#define PT_K_STEP 32                          // 2 sub-MMAs per resident K-step
#define PT_K_SUB 16                           // one m16n8k16's K-width
#define PT_K_SUBS (PT_K_STEP / PT_K_SUB)      // = 2
#define PT_PAD 2
// A-tile smem row stride: 32 real K-cols + 8 pad = 40 BF16 = 80 bytes, a
// multiple of 16 so every 16-byte cp.async.cg chunk lands aligned (a 36-byte
// PAD=2 stride misaligns odd rows → CUDA_ERROR_MISALIGNED_ADDRESS); the pad
// also breaks shared-memory bank conflicts on the MMA's u32 reads.
#define PT_A_STRIDE 40
#define PT_FP8_BLOCK 128
#define PT_WARPS 8
#define PT_THREADS (PT_WARPS * 32)            // 256
#define PT_N_TILES_PER_WARP (PT_N_TILE / 8)   // = 4
#define PT_STAGES 2

// cp.async.cg 16-byte (cache-global) copy: smem <- global. sm_80+; correct on
// sm_121 (unlike TMA / cp.async.bulk). Requires 16-byte-aligned addresses.
__device__ __forceinline__ void pt_cp_async_cg_16(void* smem_ptr, const void* gmem_ptr) {
    unsigned int s = (unsigned int)__cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem_ptr));
}
__device__ __forceinline__ void pt_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void pt_cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void pt_cp_async_wait_le(unsigned int n) {
    switch (n) {
        case 0:  pt_cp_async_wait_group<0>(); break;
        case 1:  pt_cp_async_wait_group<1>(); break;
        case 2:  pt_cp_async_wait_group<2>(); break;
        default: pt_cp_async_wait_group<3>(); break;
    }
}

// MMA over one resident K_STEP (= PT_K_SUBS m16n8k16 sub-MMAs). Reads K-CONTIGUOUS
// smem_B[n][k] — fragment-identical to the non-transposed pipelined sibling.
__device__ __forceinline__ void pt_mma_kstep(
    const __nv_bfloat16* smem_A,   // [PT_M_TILE][PT_A_STRIDE]
    const __nv_bfloat16* smem_B,   // [PT_N_TILE][PT_K_STEP + PT_PAD] (K-contiguous)
    float inner[PT_N_TILES_PER_WARP][4],
    unsigned int warp_m_offset, unsigned int group_id, unsigned int tid
) {
    const unsigned int a_stride = PT_A_STRIDE;
    const unsigned int b_stride = PT_K_STEP + PT_PAD;
    const unsigned short* sA = (const unsigned short*)smem_A;
    const unsigned short* sB = (const unsigned short*)smem_B;

    unsigned int frag_r0 = warp_m_offset + group_id;
    unsigned int frag_r1 = warp_m_offset + group_id + 8;

    #pragma unroll
    for (int s = 0; s < PT_K_SUBS; s++) {
        const unsigned int k_off = s * PT_K_SUB;
        unsigned int frag_c0 = k_off + tid * 2;
        unsigned int frag_c1 = k_off + tid * 2 + 8;

        unsigned int a0 = *(const unsigned int*)&sA[frag_r0 * a_stride + frag_c0];
        unsigned int a1 = *(const unsigned int*)&sA[frag_r1 * a_stride + frag_c0];
        unsigned int a2 = *(const unsigned int*)&sA[frag_r0 * a_stride + frag_c1];
        unsigned int a3 = *(const unsigned int*)&sA[frag_r1 * a_stride + frag_c1];

        #pragma unroll
        for (int n_tile = 0; n_tile < PT_N_TILES_PER_WARP; n_tile++) {
            unsigned int n_col = n_tile * 8 + group_id;
            unsigned int k0 = k_off + tid * 2;
            unsigned int k1 = k_off + tid * 2 + 8;

            // [n][k] K-contiguous: (k, k+1) adjacent → single aligned u32.
            unsigned int b0 = *(const unsigned int*)&sB[n_col * b_stride + k0];
            unsigned int b1 = *(const unsigned int*)&sB[n_col * b_stride + k1];

            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9}, "
                "{%10, %11, %12, %13};"
                : "=f"(inner[n_tile][0]), "=f"(inner[n_tile][1]),
                  "=f"(inner[n_tile][2]), "=f"(inner[n_tile][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(b0), "r"(b1),
                  "f"(inner[n_tile][0]), "f"(inner[n_tile][1]),
                  "f"(inner[n_tile][2]), "f"(inner[n_tile][3])
            );
        }
    }
}

/// W8A16 pipelined TRANSPOSED GEMM: B_t[K,N] N-contiguous FP8 E4M3 + 2D
/// transposed block scales. 128×32 tile (M×N), 8 warps, PT_STAGES-deep
/// cp.async prefetch, shared-memory LUT, transpose-on-dequant.
extern "C" __global__ void w8a16_gemm_t_pipelined(
    const __nv_bfloat16* __restrict__ A,            // [M, K] BF16 activations
    const unsigned char* __restrict__ B_t,           // [K, N] FP8 E4M3 transposed
    const float* __restrict__ block_scale_t,         // [K/128, N/128] FP32 transposed
    __nv_bfloat16* __restrict__ C,                   // [M, N] BF16 output
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int cta_m = blockIdx.y * PT_M_TILE;
    const unsigned int cta_n = blockIdx.x * PT_N_TILE;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    // smem_A / smem_Braw are cp.async destinations → __align__(16), 16-byte row
    // strides. smem_B is the MMA-ready K-contiguous dequantized buffer.
    //   Per stage: smem_A 128*40*2 = 10240 B + smem_B 32*34*2 = 2176 B +
    //   smem_Braw 32*32 = 1024 B ≈ 13440 B → 26880 B for 2 stages + 1 KB LUT.
    __shared__ __align__(16) __nv_bfloat16 smem_A[PT_STAGES][PT_M_TILE][PT_A_STRIDE];
    __shared__ __nv_bfloat16 smem_B[PT_STAGES][PT_N_TILE][PT_K_STEP + PT_PAD];
    // Raw FP8 bytes, N-CONTIGUOUS [k][n] mirroring global B_t (coalesced loads).
    __shared__ __align__(16) unsigned char smem_Braw[PT_STAGES][PT_K_STEP][PT_N_TILE];

    // ── Lever 1: stage the E4M3→FP32 LUT in SHARED memory (SSOT = E4M3_LUT) ──
    __shared__ float smem_lut[256];
    smem_lut[threadIdx.x] = E4M3_LUT[threadIdx.x];   // PT_THREADS == 256, exact cover
    __syncthreads();

    // Two-level FP32 accumulation (PRESERVED EXACTLY).
    float inner_acc[PT_N_TILES_PER_WARP][4];
    float outer_acc[PT_N_TILES_PER_WARP][4];
    #pragma unroll
    for (int i = 0; i < PT_N_TILES_PER_WARP; i++) {
        inner_acc[i][0] = 0.0f; inner_acc[i][1] = 0.0f;
        inner_acc[i][2] = 0.0f; inner_acc[i][3] = 0.0f;
        outer_acc[i][0] = 0.0f; outer_acc[i][1] = 0.0f;
        outer_acc[i][2] = 0.0f; outer_acc[i][3] = 0.0f;
    }

    const unsigned int n_scale_blocks = (N + PT_FP8_BLOCK - 1) / PT_FP8_BLOCK;
    const unsigned int k_steps_per_block = PT_FP8_BLOCK / PT_K_STEP;
    const unsigned int n_block = cta_n / PT_FP8_BLOCK;
    const unsigned int n_steps = (K + PT_K_STEP - 1) / PT_K_STEP;

    // A-tile cp.async: 128 rows × PT_K_STEP K-cols BF16, contiguous along K.
    const unsigned int a_chunks = (PT_M_TILE * PT_K_STEP) / 8;     // 512 at K_STEP=32

    // Issue cp.async loads for K-step `step` into double-buffer `stage`.
    auto prefetch = [&](unsigned int step, unsigned int stage) {
        unsigned int k_base = step * PT_K_STEP;

        // ── A: contiguous 16-B (8 BF16) chunks along K ──
        #pragma unroll
        for (unsigned int c = threadIdx.x; c < a_chunks; c += PT_THREADS) {
            unsigned int row = (c * 8) / PT_K_STEP;          // 0..127
            unsigned int col = (c * 8) % PT_K_STEP;          // 0,8,16,24
            unsigned int gr = cta_m + row;
            unsigned int gc = k_base + col;
            __nv_bfloat16* dst = &smem_A[stage][row][col];
            if (gr < M && gc + 8 <= K) {
                pt_cp_async_cg_16(dst, &A[(unsigned long long)gr * K + gc]);
            } else {
                #pragma unroll
                for (unsigned int e = 0; e < 8; e++) {
                    unsigned int gcol = gc + e;
                    dst[e] = (gr < M && gcol < K) ? A[(unsigned long long)gr * K + gcol]
                                                  : __float2bfloat16(0.0f);
                }
            }
        }

        // ── B_t raw: COALESCED 16-B (16 FP8-byte) chunks of N per K-row ──
        // smem_Braw[stage][k][n] mirrors global B_t[k_base + k, n] contiguously.
        // At N_TILE=32 each K-row is 32 bytes = two 16-B chunks; iterate over
        // (PT_K_STEP × PT_N_TILE/16) chunks, one cp.async per chunk.
        const unsigned int b_chunks = (PT_K_STEP * PT_N_TILE) / 16;   // 64 at N_TILE=32
        #pragma unroll
        for (unsigned int c = threadIdx.x; c < b_chunks; c += PT_THREADS) {
            unsigned int krow = (c * 16) / PT_N_TILE;        // 0..PT_K_STEP-1
            unsigned int ncol = (c * 16) % PT_N_TILE;        // 0 or 16
            unsigned int gk = k_base + krow;
            unsigned int gn = cta_n + ncol;
            unsigned char* dst = &smem_Braw[stage][krow][ncol];
            if (gk < K && gn + 16 <= N) {
                pt_cp_async_cg_16(dst, &B_t[(unsigned long long)gk * N + gn]);
            } else {
                #pragma unroll
                for (unsigned int e = 0; e < 16; e++) {
                    unsigned int gne = gn + e;
                    dst[e] = (gk < K && gne < N) ? B_t[(unsigned long long)gk * N + gne] : 0;
                }
            }
        }
        pt_cp_async_commit();
    };

    // LUT-dequant + TRANSPOSE just-arrived raw B for `stage`: read N-contiguous
    // smem_Braw[k][n], write K-contiguous smem_B[n][k]. The transpose is free
    // (dequant touches every element anyway) and is the whole point — it gives
    // the MMA single-u32 K-contiguous fragment loads from a coalesced N-load.
    // No scale here (folded on the FP32 accumulator at the block boundary).
    auto dequant_B = [&](unsigned int stage) {
        #pragma unroll
        for (unsigned int idx = threadIdx.x; idx < PT_K_STEP * PT_N_TILE; idx += PT_THREADS) {
            unsigned int k = idx / PT_N_TILE;     // 0..PT_K_STEP-1
            unsigned int n = idx % PT_N_TILE;     // 0..PT_N_TILE-1
            unsigned char wb = smem_Braw[stage][k][n];
            smem_B[stage][n][k] = __float2bfloat16(smem_lut[wb]);
        }
    };

    // ── Software-pipelined main loop (PT_STAGES-deep cp.async) ──
    #pragma unroll
    for (unsigned int p = 0; p < PT_STAGES - 1; p++) {
        if (p < n_steps) {
            prefetch(p, p % PT_STAGES);
        }
    }
    unsigned int k_step_in_block = 0;

    for (unsigned int step = 0; step < n_steps; step++) {
        unsigned int cur = step % PT_STAGES;

        unsigned int ahead = step + (PT_STAGES - 1);
        if (ahead < n_steps) {
            prefetch(ahead, ahead % PT_STAGES);
        }
        unsigned int committed = min(n_steps, PT_STAGES + step);
        unsigned int target = committed - (step + 1);
        pt_cp_async_wait_le(target);
        __syncthreads();   // raw B for `cur` resident for all threads

        dequant_B(cur);
        __syncthreads();   // smem_B[cur] fully written before MMA reads it

        pt_mma_kstep(&smem_A[cur][0][0], &smem_B[cur][0][0],
                     inner_acc, warp_m_offset, group_id, tid);
        __syncthreads();   // done reading smem_*[cur]; safe for reuse

        // K_BLOCK boundary: fold scaled inner into outer, reset inner.
        k_step_in_block++;
        if (k_step_in_block == k_steps_per_block) {
            const unsigned int k_block = (step * PT_K_STEP) / PT_FP8_BLOCK;
            // Transposed scale layout: [K/128, N/128].
            const float scale = block_scale_t[k_block * n_scale_blocks + n_block];
            #pragma unroll
            for (int i = 0; i < PT_N_TILES_PER_WARP; i++) {
                outer_acc[i][0] += inner_acc[i][0] * scale;
                outer_acc[i][1] += inner_acc[i][1] * scale;
                outer_acc[i][2] += inner_acc[i][2] * scale;
                outer_acc[i][3] += inner_acc[i][3] * scale;
                inner_acc[i][0] = 0.0f; inner_acc[i][1] = 0.0f;
                inner_acc[i][2] = 0.0f; inner_acc[i][3] = 0.0f;
            }
            k_step_in_block = 0;
        }
    }

    // Fold any incomplete trailing K_BLOCK (only when K % FP8_BLOCK != 0).
    if (k_step_in_block != 0) {
        const unsigned int k_block = (K - 1) / PT_FP8_BLOCK;
        const float scale = block_scale_t[k_block * n_scale_blocks + n_block];
        #pragma unroll
        for (int i = 0; i < PT_N_TILES_PER_WARP; i++) {
            outer_acc[i][0] += inner_acc[i][0] * scale;
            outer_acc[i][1] += inner_acc[i][1] * scale;
            outer_acc[i][2] += inner_acc[i][2] * scale;
            outer_acc[i][3] += inner_acc[i][3] * scale;
        }
    }

    // ── Store C tile: f32 outer accumulators → BF16 output ──
    #pragma unroll
    for (int n_tile = 0; n_tile < PT_N_TILES_PER_WARP; n_tile++) {
        unsigned int base_n = cta_n + n_tile * 8;
        unsigned int col0 = base_n + (tid * 2);
        unsigned int col1 = col0 + 1;
        unsigned int row0 = cta_m + warp_m_offset + group_id;
        unsigned int row1 = row0 + 8;

        if (row0 < M && col0 < N) C[row0 * N + col0] = __float2bfloat16(outer_acc[n_tile][0]);
        if (row0 < M && col1 < N) C[row0 * N + col1] = __float2bfloat16(outer_acc[n_tile][1]);
        if (row1 < M && col0 < N) C[row1 * N + col0] = __float2bfloat16(outer_acc[n_tile][2]);
        if (row1 < M && col1 < N) C[row1 * N + col1] = __float2bfloat16(outer_acc[n_tile][3]);
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
/// FP32 scales (widened from the checkpoint BF16/FP32 at load time).
extern "C" __global__ void transpose_block_scale(
    const float* __restrict__ scale,        // [N/128, K/128] FP32
    float* __restrict__ scale_t,            // [K/128, N/128] FP32
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
