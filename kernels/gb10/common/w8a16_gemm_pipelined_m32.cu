// SPDX-License-Identifier: AGPL-3.0-only

// Atlas W8A16 Pipelined Dequant+GEMM — 32-row M-tile twin of `w8a16_gemm_pipelined`.
//
// C[r, n] = sum_k A[r, k] * dequant(B[n, k]) * block_scale[n/128, k/128]
//           r < M, n < N;  A rows `lda` elements apart, C rows `ldc` apart.
//
// WHY (G18, MoE C=16 profile 2026-09-22/23, R = 32 verify rows): the GDN
// in_proj_qkvz / out_proj and, past 16 rows, the attention Q/K/V and o_proj
// FP8 projections all ran `w8a16_gemm_pipelined`, whose 128-row M tile is
// 75 % zero-padding at M=32 — every CTA streams a 128x32 A tile (96 rows of
// scalar zero-fill), dequantizes a 32x32 B step and issues 64 MMAs of which
// 48 multiply zeros, all behind three `__syncthreads` per 32-wide K-step. At
// M=32, N=12288, K=2048 that measured 296 us for 25.2 MB of weight (85 GB/s
// of a 273 GB/s bus); `out_proj` (N=2048, K=4096) launches 64 CTAs on 48 SMs
// and took 166 us for 8.4 MB, because each CTA is a 128-step chain of
// barrier latency with nothing co-resident to hide it.
//
// This twin changes the TILE, not the math:
//   * PM32_M_TILE = 32 — the A tile is the real rows (no padded rows past 32);
//     8 warps are laid out 2 (M) x 4 (N), each owning a 16x8 m16n8k16 slab.
//   * PM32_K_STEP = 128 — ONE FP8 scale block per resident K-step, so a CTA
//     runs K/128 barrier triples instead of K/32 (32 instead of 128 for
//     out_proj) and the scale fold happens once per step.
//   * `lda` / `ldc` row pitches, so the multi-seq attention Q/K/V tier can
//     write all n rows straight into the `[n, per_seq_qkv]` decode buffer
//     (Q at 0, K after Q, V after K) the way its `_strided` GEMVs do.
//
// NUMERICS — BIT-IDENTICAL to `w8a16_gemm_pipelined`, by construction, and
// the oracle (`examples/native_fp8_gdn_proj_m32_microtest`) byte-compares
// the two. Every output element sees the same sequence of operations:
// m16n8k16 sub-MMAs over the same 16-aligned K windows in ascending order,
// accumulating into an FP32 `inner` that is folded onto `outer` with the
// block scale after every 8 windows (= one 128-K block), exactly as the
// 128-tile kernel folds after its 4 x 2 sub-MMAs. Same instruction, same
// fragment values (E4M3 -> BF16 is lossless), same fold points => same bits.
// K must be a multiple of 128 (the wrapper enforces it): with K_STEP equal to
// the scale block there is no partial trailing block to special-case, and the
// callers' K (hidden 2048/5120, value_dim 4096, q_dim 8192) all satisfy it.
//
// Relative to the SCALAR `w8a16_gemv` family this reassociates the K
// reduction like every tensor-core tier does (<= 2 BF16 ULP or the
// accumulation floor — `layers::dense_ffn::m16_tc::oracle`); it is the
// numerics the same weights already see in prefill (`w8a16_gemm_pipelined`)
// and in the GDN verify at R > 16.
//
// Shared memory (static, per CTA): A 2 x 32 x 136 x 2 B = 17,408; B (BF16,
// [n][k] K-contiguous, +8 pad) 17,408; B raw 2 x 32 x 128 = 8,192; LUT 1,024
// => 44,032 B, under the 48 KB static limit. Both padded strides are 68
// words, 68 mod 32 = 4, so the eight `group_id` rows of an MMA fragment read
// land on banks 0,4,...,28 (+tid) — conflict-free, the same property the
// 128-tile kernel's 40-short A stride has.
//
// The cp.async helpers below duplicate the three in
// `w8a16_gemm_pipelined.cu` VERBATIM. That file is the prefill kernel of
// every FP8 model and this twin was written under a no-compile rule
// (certification campaign on the box); moving the helpers into a shared
// `.cuh` is the right end state and the first follow-up once a build can
// prove the production PTX unchanged.
//
// Grid: (ceil(N/32), ceil(M/32), 1), Block: (256,1,1) = 8 warps.

#include <cuda_bf16.h>
#include "e4m3_lut.cuh"

#define PM32_M_TILE 32
#define PM32_N_TILE 32
#define PM32_K_STEP 128                         // == PM32_FP8_BLOCK: one scale block per step
#define PM32_K_SUB 16                           // one m16n8k16's K width
#define PM32_K_SUBS (PM32_K_STEP / PM32_K_SUB)  // = 8
#define PM32_PAD 8                              // shorts; keeps rows 16 B-aligned and banks spread
#define PM32_A_STRIDE (PM32_K_STEP + PM32_PAD)  // 136 BF16 = 272 B
#define PM32_B_STRIDE (PM32_K_STEP + PM32_PAD)  // 136 BF16
#define PM32_FP8_BLOCK 128
#define PM32_WARPS 8
#define PM32_THREADS (PM32_WARPS * 32)          // 256
#define PM32_WARPS_M (PM32_M_TILE / 16)         // 2
#define PM32_WARPS_N (PM32_WARPS / PM32_WARPS_M) // 4; PM32_N_TILE / 8 == PM32_WARPS_N: one n-tile per warp
#define PM32_STAGES 2

__device__ __forceinline__ void pm32_cp_async_cg_16(void* smem_ptr, const void* gmem_ptr) {
    unsigned int s = (unsigned int)__cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem_ptr));
}
__device__ __forceinline__ void pm32_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void pm32_cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
// Runtime-count wait over the (<= PM32_STAGES) legal values; the default arm
// drains fully, which is correct for any deeper pipeline.
__device__ __forceinline__ void pm32_cp_async_wait_le(unsigned int n) {
    switch (n) {
        case 0:  pm32_cp_async_wait_group<0>(); break;
        case 1:  pm32_cp_async_wait_group<1>(); break;
        default: pm32_cp_async_wait_group<2>(); break;
    }
}

// One resident K-step = PM32_K_SUBS m16n8k16 sub-MMAs over the warp's single
// 16x8 slab, accumulated into `inner[4]`. Fragment addressing is the
// 128-tile kernel's `pm_mma_kstep` with one n-tile per warp and the warp's
// column offset folded into `n_col`.
__device__ __forceinline__ void pm32_mma_kstep(
    const __nv_bfloat16* smem_A,   // [PM32_M_TILE][PM32_A_STRIDE]
    const __nv_bfloat16* smem_B,   // [PM32_N_TILE][PM32_B_STRIDE] (K-contiguous)
    float inner[4],
    unsigned int warp_m_offset, unsigned int warp_n_offset,
    unsigned int group_id, unsigned int tid
) {
    const unsigned short* sA = (const unsigned short*)smem_A;
    const unsigned short* sB = (const unsigned short*)smem_B;
    const unsigned int frag_r0 = warp_m_offset + group_id;
    const unsigned int frag_r1 = frag_r0 + 8;
    const unsigned int n_col = warp_n_offset + group_id;

    #pragma unroll
    for (int s = 0; s < PM32_K_SUBS; s++) {
        const unsigned int k_off = s * PM32_K_SUB;
        const unsigned int c0 = k_off + tid * 2;
        const unsigned int c1 = c0 + 8;

        unsigned int a0 = *(const unsigned int*)&sA[frag_r0 * PM32_A_STRIDE + c0];
        unsigned int a1 = *(const unsigned int*)&sA[frag_r1 * PM32_A_STRIDE + c0];
        unsigned int a2 = *(const unsigned int*)&sA[frag_r0 * PM32_A_STRIDE + c1];
        unsigned int a3 = *(const unsigned int*)&sA[frag_r1 * PM32_A_STRIDE + c1];
        unsigned int b0 = *(const unsigned int*)&sB[n_col * PM32_B_STRIDE + c0];
        unsigned int b1 = *(const unsigned int*)&sB[n_col * PM32_B_STRIDE + c1];

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%10, %11, %12, %13};"
            : "=f"(inner[0]), "=f"(inner[1]), "=f"(inner[2]), "=f"(inner[3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1),
              "f"(inner[0]), "f"(inner[1]), "f"(inner[2]), "f"(inner[3])
        );
    }
}

/// W8A16 pipelined GEMM, 32x32 (M x N) tile, 128-K steps, 8 warps, strided
/// A/C. `M` may be any value: rows past M in the last M tile are zero-filled
/// and never stored (grid.y = ceil(M/32)).
extern "C" __global__ void w8a16_gemm_pipelined_m32(
    const __nv_bfloat16* __restrict__ A,            // [M, lda] BF16, K used per row
    const unsigned char* __restrict__ B,             // [N, K] FP8 E4M3
    const float* __restrict__ block_scale,           // [N/128, K/128] FP32
    __nv_bfloat16* __restrict__ C,                   // [M, ldc] BF16, N written per row
    unsigned int M,
    unsigned int N,
    unsigned int K,                                  // multiple of 128
    unsigned int lda,                                // elements; multiple of 8 (16 B rows)
    unsigned int ldc                                 // elements
) {
    const unsigned int cta_m = blockIdx.y * PM32_M_TILE;
    const unsigned int cta_n = blockIdx.x * PM32_N_TILE;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = (warp_id % PM32_WARPS_M) * 16;
    const unsigned int warp_n_offset = (warp_id / PM32_WARPS_M) * 8;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __align__(16) __nv_bfloat16 smem_A[PM32_STAGES][PM32_M_TILE][PM32_A_STRIDE];
    __shared__ __align__(16) __nv_bfloat16 smem_B[PM32_STAGES][PM32_N_TILE][PM32_B_STRIDE];
    __shared__ __align__(16) unsigned char smem_Braw[PM32_STAGES][PM32_N_TILE][PM32_K_STEP];
    __shared__ float smem_lut[256];
    smem_lut[threadIdx.x] = E4M3_LUT[threadIdx.x];   // PM32_THREADS == 256, exact cover
    __syncthreads();

    float inner_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float outer_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    const unsigned int k_blocks = K / PM32_FP8_BLOCK;
    const unsigned int n_block = cta_n / PM32_FP8_BLOCK;
    const unsigned int n_steps = K / PM32_K_STEP;   // K % 128 == 0: no partial step

    // A: 32 rows x 128 K BF16 = 512 16-B chunks (2 per thread).
    // B: 32 rows x 128 K FP8  = 256 16-B chunks (1 per thread).
    const unsigned int a_chunks = (PM32_M_TILE * PM32_K_STEP) / 8;
    const unsigned int b_chunks = (PM32_N_TILE * PM32_K_STEP) / 16;

    auto prefetch = [&](unsigned int step, unsigned int stage) {
        const unsigned int k_base = step * PM32_K_STEP;
        #pragma unroll
        for (unsigned int c = threadIdx.x; c < a_chunks; c += PM32_THREADS) {
            const unsigned int row = (c * 8) / PM32_K_STEP;
            const unsigned int col = (c * 8) % PM32_K_STEP;
            const unsigned int gr = cta_m + row;
            const unsigned int gc = k_base + col;
            __nv_bfloat16* dst = &smem_A[stage][row][col];
            if (gr < M) {
                pm32_cp_async_cg_16(dst, &A[(unsigned long long)gr * lda + gc]);
            } else {
                #pragma unroll
                for (unsigned int e = 0; e < 8; e++) dst[e] = __float2bfloat16(0.0f);
            }
        }
        #pragma unroll
        for (unsigned int c = threadIdx.x; c < b_chunks; c += PM32_THREADS) {
            const unsigned int nrow = (c * 16) / PM32_K_STEP;
            const unsigned int kcol = (c * 16) % PM32_K_STEP;
            const unsigned int gn = cta_n + nrow;
            const unsigned int gk = k_base + kcol;
            unsigned char* dst = &smem_Braw[stage][nrow][kcol];
            if (gn < N) {
                pm32_cp_async_cg_16(dst, &B[(unsigned long long)gn * K + gk]);
            } else {
                #pragma unroll
                for (unsigned int e = 0; e < 16; e++) dst[e] = 0;
            }
        }
        pm32_cp_async_commit();
    };

    // Same-layout element-wise LUT dequant of the arrived raw step (32 x 128
    // bytes, 16 per thread); no scale here — folded on the FP32 accumulator.
    auto dequant_B = [&](unsigned int stage) {
        #pragma unroll
        for (unsigned int idx = threadIdx.x; idx < PM32_K_STEP * PM32_N_TILE; idx += PM32_THREADS) {
            const unsigned int n = idx / PM32_K_STEP;
            const unsigned int k = idx % PM32_K_STEP;
            smem_B[stage][n][k] = __float2bfloat16(smem_lut[smem_Braw[stage][n][k]]);
        }
    };

    #pragma unroll
    for (unsigned int p = 0; p < PM32_STAGES - 1; p++) {
        if (p < n_steps) prefetch(p, p % PM32_STAGES);
    }

    for (unsigned int step = 0; step < n_steps; step++) {
        const unsigned int cur = step % PM32_STAGES;
        const unsigned int ahead = step + (PM32_STAGES - 1);
        if (ahead < n_steps) prefetch(ahead, ahead % PM32_STAGES);
        // Groups complete FIFO; keep min(n_steps, STAGES+step) - (step+1)
        // in flight so `cur` (the oldest) is complete — the 128-tile kernel's
        // drain rule verbatim.
        const unsigned int committed = min(n_steps, PM32_STAGES + step);
        pm32_cp_async_wait_le(committed - (step + 1));
        __syncthreads();

        dequant_B(cur);
        __syncthreads();

        pm32_mma_kstep(&smem_A[cur][0][0], &smem_B[cur][0][0],
                       inner_acc, warp_m_offset, warp_n_offset, group_id, tid);
        __syncthreads();

        // One K-step == one 128-K scale block: fold every step.
        const float scale = block_scale[n_block * k_blocks + step];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            outer_acc[i] += inner_acc[i] * scale;
            inner_acc[i] = 0.0f;
        }
    }

    const unsigned int col0 = cta_n + warp_n_offset + tid * 2;
    const unsigned int col1 = col0 + 1;
    const unsigned int row0 = cta_m + warp_m_offset + group_id;
    const unsigned int row1 = row0 + 8;
    if (row0 < M && col0 < N) C[(unsigned long long)row0 * ldc + col0] = __float2bfloat16(outer_acc[0]);
    if (row0 < M && col1 < N) C[(unsigned long long)row0 * ldc + col1] = __float2bfloat16(outer_acc[1]);
    if (row1 < M && col0 < N) C[(unsigned long long)row1 * ldc + col0] = __float2bfloat16(outer_acc[2]);
    if (row1 < M && col1 < N) C[(unsigned long long)row1 * ldc + col1] = __float2bfloat16(outer_acc[3]);
}
