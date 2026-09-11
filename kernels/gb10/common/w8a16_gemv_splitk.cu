// SPDX-License-Identifier: AGPL-3.0-only

// Atlas W8A16 decode GEMV, SPLIT-K — block-scaled FP8 E4M3 weights, BF16
// activations, M=1. Computes exactly what `w8a16_gemv.cu` computes:
//
//   C[n] = sum_k A[k] * E4M3_LUT[B[n, k]] * block_scale[n/128, k/128]
//
// but hands each output's K range to `splits` CTAs instead of one, so a
// short-N/long-K shape stops being starved of blocks.
//
// WHY (#928, C=1 decode). nsys, 1xH100, Qwen/Qwen3.8-27B-FP8 native FP8,
// 2026-09-11 round 7, steady-state decode step 21.891 ms:
//
//   kernel                    shape              grid   us/launch   GB/s
//   w8a16_gemv_dual           N=17408x2 K=5120   4352      90.1     1,979
//   w8a16_gemv                N=16384   K=5120   4096       -       1,852
//   w8a16_gemv_silu_input     N=5120    K=17408  1280     103.9       858
//   w8a16_gemv (k/v proj)     N=1024    K=5120    256       -         861
//
// Read the grid column, not the K column: this kernel family's effective
// bandwidth tracks HOW MANY CTAs the shape produces, because `N_PER_BLOCK=4`
// outputs per 256-thread CTA makes the CTA count a pure function of N. H100
// has 132 SMs and this CTA is 256 threads, so ~8 CTAs/SM co-reside:
//
//   grid 4352 -> 33 CTAs/SM -> ~4.1 full waves -> tail costs ~2%   (1,979 GB/s)
//   grid 1280 -> 9.7 CTAs/SM -> ~1.2 waves, i.e. ONE full wave plus a
//                224-CTA tail that leaves 83% of the machine idle for the
//                whole second wave -> ~60% of the achievable rate
//   grid  256 -> 1.9 CTAs/SM -> under-occupied outright; there are never
//                enough warps resident to cover an HBM round trip (861 GB/s)
//
// The 8 CTAs/SM is ptxas-pinned, not estimated: `nvcc -cubin -Xptxas -v
// -arch=sm_90a --fmad=false` (CUDA 13.0, 2026-09-11) reports 32 registers,
// 1,056 B smem and 0 spills for BOTH `w8a16_gemv` and this kernel, and
// 32 x 256 x 8 = 65,536 is exactly the SM register file. So the split buys
// CTAs at no occupancy cost. (`w8a16_gemv_silu_input` needs 53 registers =
// 13,568/CTA = only 4 CTAs/SM, which is a second reason the fused SwiGLU
// variant is slow and a reason to stage the activation instead.)
//
// Split-K restores the CTA count without touching the per-lane work: at
// splits=4 the FFN down projection (N=5120, K=17408) launches 5,120 CTAs and
// each lane walks 5 chunk iterations instead of 17 — the same per-lane profile
// as `w8a16_gemv_dual`, which is the shape already measured at 1,979 GB/s.
//
// 🟡 NUMERICS — the per-lane chains are BYTE-IDENTICAL sub-chains of
// `w8a16_gemv`'s; only the final `splits`-way combine is new.
//   * the lane -> k16 map is unchanged WITHIN a split (`threads_per_out` is
//     still 64, the stride is still 64, the start is still `lane`), and each
//     split owns a CONTIGUOUS, 64-chunk-aligned range of the same chunk
//     sequence, so every partial sum is a prefix/interior run of the operands
//     the scalar kernel adds in that same order;
//   * the in-CTA reduction (shfl_down tree, then the two-partial add in shared
//     memory) is unchanged, and the result is kept in FP32 — the BF16 round
//     happens once, in the reduce pass, exactly where the scalar kernel does
//     it;
//   * `w8a16_gemv_splitk_reduce` adds the partials in INCREASING split order,
//     s = 0, 1, .. splits-1. That order is the contract; it is fixed here so
//     the result is deterministic run to run and machine to machine.
//   * therefore `splits == 1` is BIT-IDENTICAL to `w8a16_gemv`, and that is
//     the oracle the microtest asserts (`unequal=0`). `splits > 1` differs
//     only by FP32 reassociation of at most 8 addends, which is why the
//     dispatch keeps it behind `ATLAS_FFN_DOWN_SPLITK` (presence, default
//     OFF). Oracle: `examples/native_fp8_ffn_down_gemv_microtest`.
//
// Host contract (SSOT: `layers::ops::w8a16_decode_gemv`): `iters_per_split` is
// ceil(ceil(K/16) / 64 / splits); the caller sizes `partials` as
// [W8A16_SPLITK_MAX, N] FP32 and never launches an empty split.
//
// Entry points:
//   w8a16_gemv_splitk(A, B, block_scale, partials, N, K, iters_per_split)
//     Grid: (ceil(N/4), 1, splits)  Block: (256, 1, 1)
//   w8a16_gemv_splitk_reduce(partials, C, N, splits)
//     Grid: (ceil(N/256), 1, 1)     Block: (256, 1, 1)

#include <cuda_bf16.h>

#include "e4m3_lut.cuh"

#define BLOCK_SIZE 256
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define FP8_BLOCK 128

// ── Stage 1: per-split partial sums ────────────────────────────────

extern "C" __global__ void w8a16_gemv_splitk(
    const __nv_bfloat16* __restrict__ A,            // [1, K] BF16
    const unsigned char* __restrict__ B,             // [N, K] FP8 E4M3
    const float* __restrict__ block_scale,           // [N/128, K/128] FP32
    float* __restrict__ partials,                    // [splits, N] FP32
    unsigned int N,
    unsigned int K,
    unsigned int iters_per_split
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;  // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int K16 = K / 16;
    const unsigned int k_blocks = (K + FP8_BLOCK - 1) / FP8_BLOCK;  // ceil(K/128)
    const unsigned int n_block = n / FP8_BLOCK;

    // This split's contiguous, 64-chunk-aligned slice of the chunk sequence.
    // `chunk_hi < chunk_lo` (an over-long grid.z) makes the loop a no-op and
    // the partial a clean 0.0f rather than a wild read.
    const unsigned int chunk_lo = blockIdx.z * iters_per_split * threads_per_out;
    unsigned int chunk_hi = chunk_lo + iters_per_split * threads_per_out;
    if (chunk_hi > K16) {
        chunk_hi = K16;
    }

    __shared__ float s_lut[256];
    __shared__ float smem[N_PER_BLOCK * 2];
    s_lut[threadIdx.x] = E4M3_LUT[threadIdx.x];
    __syncthreads();

    float acc = 0.0f;

    // 16 K-values per iteration, applying the per-128-block scale. Body is
    // `w8a16_gemv.cu`'s, character for character; only the bounds differ.
    for (unsigned int k16 = chunk_lo + lane; k16 < chunk_hi; k16 += threads_per_out) {
        const unsigned int base_k = k16 * 16;

        const unsigned int k_block = base_k / FP8_BLOCK;
        float scale = block_scale[n_block * k_blocks + k_block];

        uint4 b_data = ((const uint4*)(B + (unsigned long long)n * K))[k16];
        uint4 a_data0 = ((const uint4*)A)[k16 * 2];
        uint4 a_data1 = ((const uint4*)A)[k16 * 2 + 1];

        const unsigned int b_raw0[2] = {b_data.x, b_data.y};
        const unsigned int a_raw0[4] = {a_data0.x, a_data0.y, a_data0.z, a_data0.w};

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            unsigned int w32 = b_raw0[i];
            unsigned int a32_lo = a_raw0[i * 2];
            unsigned int a32_hi = a_raw0[i * 2 + 1];

            float w0 = s_lut[(w32      ) & 0xFF] * scale;
            float w1 = s_lut[(w32 >>  8) & 0xFF] * scale;
            float w2 = s_lut[(w32 >> 16) & 0xFF] * scale;
            float w3 = s_lut[(w32 >> 24) & 0xFF] * scale;

            __nv_bfloat16 a0, a1, a2, a3;
            *(unsigned short*)&a0 = (unsigned short)(a32_lo & 0xFFFF);
            *(unsigned short*)&a1 = (unsigned short)(a32_lo >> 16);
            *(unsigned short*)&a2 = (unsigned short)(a32_hi & 0xFFFF);
            *(unsigned short*)&a3 = (unsigned short)(a32_hi >> 16);

            acc += __bfloat162float(a0) * w0;
            acc += __bfloat162float(a1) * w1;
            acc += __bfloat162float(a2) * w2;
            acc += __bfloat162float(a3) * w3;
        }

        const unsigned int b_raw1[2] = {b_data.z, b_data.w};
        const unsigned int a_raw1[4] = {a_data1.x, a_data1.y, a_data1.z, a_data1.w};

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            unsigned int w32 = b_raw1[i];
            unsigned int a32_lo = a_raw1[i * 2];
            unsigned int a32_hi = a_raw1[i * 2 + 1];

            float w0 = s_lut[(w32      ) & 0xFF] * scale;
            float w1 = s_lut[(w32 >>  8) & 0xFF] * scale;
            float w2 = s_lut[(w32 >> 16) & 0xFF] * scale;
            float w3 = s_lut[(w32 >> 24) & 0xFF] * scale;

            __nv_bfloat16 a0, a1, a2, a3;
            *(unsigned short*)&a0 = (unsigned short)(a32_lo & 0xFFFF);
            *(unsigned short*)&a1 = (unsigned short)(a32_lo >> 16);
            *(unsigned short*)&a2 = (unsigned short)(a32_hi & 0xFFFF);
            *(unsigned short*)&a3 = (unsigned short)(a32_hi >> 16);

            acc += __bfloat162float(a0) * w0;
            acc += __bfloat162float(a1) * w1;
            acc += __bfloat162float(a2) * w2;
            acc += __bfloat162float(a3) * w3;
        }
    }

    // Two-stage in-CTA reduction, unchanged from `w8a16_gemv`.
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    unsigned int warp_in_out = lane / WARP_SIZE;
    if (lane % WARP_SIZE == 0) {
        smem[local_out * 2 + warp_in_out] = acc;
    }
    __syncthreads();

    if (lane == 0) {
        // FP32 out — the single BF16 round stays in the reduce pass, so
        // splits==1 reproduces the scalar kernel's bits exactly.
        partials[(unsigned long long)blockIdx.z * N + n] =
            smem[local_out * 2] + smem[local_out * 2 + 1];
    }
}

// ── Stage 2: deterministic combine ─────────────────────────────────
//
// Adds the partials in increasing split order and rounds once to BF16. At
// splits==1 this is `C[n] = __float2bfloat16(result)` — the scalar kernel's
// own last line — which is what makes the splits==1 bit-identity oracle work.

extern "C" __global__ void w8a16_gemv_splitk_reduce(
    const float* __restrict__ partials,              // [splits, N] FP32
    __nv_bfloat16* __restrict__ C,                   // [1, N] BF16
    unsigned int N,
    unsigned int splits
) {
    const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    float acc = partials[n];
    for (unsigned int s = 1; s < splits; s++) {
        acc += partials[(unsigned long long)s * N + n];
    }
    C[n] = __float2bfloat16(acc);
}
