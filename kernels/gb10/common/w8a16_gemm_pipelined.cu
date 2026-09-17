// SPDX-License-Identifier: AGPL-3.0-only

// Atlas W8A16 Pipelined Dequant+GEMM — Fix-A tensor-core rewrite.
//
// C[M,N] = A[M,K] (BF16 activations) * dequant(B[N,K] (FP8 E4M3 weights))
//
// Same math + same number formats as the production `w8a16_gemm`, but tuned
// for OCCUPANCY on GB10/sm_121 (the dominant lever — see the PM_N_TILE note):
// a 128×32 output tile (8 warps) keeps the per-thread FP32 accumulator small
// enough to fit 4 CTAs (32 warps) per SM, so warp-level parallelism hides the
// per-K-step barriers / MMA-issue / smem latency. A 2-stage cp.async pipeline
// prefetches the next K-step. 56 regs, no spill, 15.5 KB smem/CTA. Measured
// ~12 TFLOP/s on the large shapes = 2.1× the 64×64 production kernel (5.6) and
// +72% over the original 128×128 1-stage-occupancy pipelined draft (7.0).
//
// FP8-E4M3 weight format (2D block-scaled), identical to w8a16_gemm:
//   B:           [N, K] uint8 — one byte per weight (FP8 E4M3)
//   block_scale: [N/128, K/128] FP32 — per-block scale factor
//
// Two-level FP32 accumulation (PRESERVED EXACTLY from w8a16_gemm — holds a
// deep-layer FP8 precision floor):
//   inner_acc accumulates MMA outputs across the 8 K_STEPs of one 128-K
//   block, where smem_B holds UNSCALED BF16-cast E4M3 weights (lossless,
//   E4M3 has 3 mantissa bits ⊂ BF16's 7). At each 128-K-block boundary:
//       outer_acc += inner_acc * block_scale[n_block, k_block]; inner_acc = 0
//   The scale is applied ONCE per block on the FP32 accumulator — never
//   per-element, never folded into BF16.
//
// Pipeline (PM_STAGES-deep, default 2): a cp.async software pipeline prefetches
// look-ahead K-steps' A tiles (BF16, contiguous K-run) and RAW FP8 B bytes
// (per-N contiguous K-run) into multi-buffered smem while the MMAs consume the
// current K-step. cp.async.cg (sm_80+) is correct on sm_121; TMA /
// cp.async.bulk are AVOIDED (they silently corrupt on sm_121).
//
// Uses mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32.
//
// Grid: (ceil(N/PM_N_TILE), ceil(M/PM_M_TILE), 1), Block: (256,1,1) = 8 warps.
// Static shared memory only (15.5 KB at 2 stages), under the 48 KB limit.

#include <cuda_bf16.h>
#include "e4m3_lut.cuh"

#define PM_M_TILE 128
// N-tile = 32 (was 128). OCCUPANCY is the dominant lever on this kernel (ptxas
// + perf sweep): the per-thread two-level FP32 accumulator tile is the register
// hog, and registers cap how many CTAs co-reside per SM (= how many warps hide
// each __syncthreads / MMA-issue / smem latency).
//   N=128 → inner+outer acc = 16 N-tiles ×4 ×2 = 128 acc regs → 168 regs/thread
//           → 1 CTA (8 warps) per SM = 12.5% occ → SM fully stalls on barriers.
//   N=64  → 64 acc regs → 95 regs/thread → 2 CTAs (16 warps) = 25% occ. +45%.
//   N=32  → 32 acc regs → 56 regs/thread → 4 CTAs (32 warps) = 50% occ. +68%.
//   N=16  → 40 regs but B re-stream + tiny MMA tiles dominate → REGRESSES.
// N=32 is the sweet spot. The smaller tile re-streams more B from DRAM, but the
// kernel is occupancy/issue-bound (not DRAM-bound), so that trade is free.
#define PM_N_TILE 32
// K_STEP = 32 (Lever 2, was 16). Each resident K-step now carries TWO
// m16n8k16 MMAs (16-K each), so the per-K-step barrier triple (raw-B sync →
// dequant → smem_B sync → MMA → reuse sync) is amortized over 2× the MMA
// work, halving the barrier count per K traversed. smem_Braw/smem_B grow with
// K_STEP but stay well under budget.
#define PM_K_STEP 32
#define PM_K_SUB 16                           // one MMA's K-width; PM_K_STEP / PM_K_SUB sub-MMAs
#define PM_K_SUBS (PM_K_STEP / PM_K_SUB)      // = 2
#define PM_PAD 2
// A-tile smem row stride MUST be a multiple of 8 BF16 (16 bytes) so every
// 16-B cp.async chunk lands on a 16-byte boundary (cp.async.cg requires
// 16-byte alignment — a 36-byte PAD=2 stride misaligns odd rows and faults
// with CUDA_ERROR_MISALIGNED_ADDRESS). 32 real K-cols + 8 pad = 40 BF16 =
// 80 bytes (multiple of 16); the pad also breaks shared-memory bank
// conflicts on the MMA's u32 reads.
#define PM_A_STRIDE 40                        // 32 K-cols + 8 pad, mult. of 8 BF16
#define PM_FP8_BLOCK 128
#define PM_WARPS 8
#define PM_THREADS (PM_WARPS * 32)            // 256
#define PM_N_TILES_PER_WARP (PM_N_TILE / 8)   // m16n8k16 N-tiles per warp (=4 at N_TILE=32)
// cp.async pipeline depth. SWEEP RESULT: 2/3/4 stages are within noise
// (12.1 / 11.85 / 11.93 TFLOP/s on 512×2048×4096) — the kernel is NOT
// global-load-latency-bound (deepening the prefetch does not help), it is
// occupancy/issue-bound. 2 stages wins marginally and uses the least smem
// (15.5 KB/CTA → most headroom should the register budget ever drop enough to
// fit a 5th CTA/SM), so keep the pipeline shallow.
#define PM_STAGES 2

// cp.async 16-byte (cg = cache-global) copy: smem <- global. sm_80+; correct
// on sm_121 (unlike TMA / cp.async.bulk). Requires 16-byte-aligned addresses.
__device__ __forceinline__ void cp_async_cg_16(void* smem_ptr, const void* gmem_ptr) {
    unsigned int s = (unsigned int)__cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem_ptr));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// Wait until at most `n` cp.async groups remain in flight, where `n` is a
// runtime value in [0, PM_STAGES-1]. cp.async.wait_group requires a
// compile-time immediate, so dispatch the runtime count through a small switch
// over the (≤ PM_STAGES) legal values. The default arm covers any deeper
// pipeline configuration safely (drains fully).
__device__ __forceinline__ void cp_async_wait_le(unsigned int n) {
    switch (n) {
        case 0:  cp_async_wait_group<0>(); break;
        case 1:  cp_async_wait_group<1>(); break;
        case 2:  cp_async_wait_group<2>(); break;
        default: cp_async_wait_group<3>(); break;
    }
}

// MMA over one resident K_STEP (PM_K_STEP K-elements = PM_K_SUBS m16n8k16
// sub-MMAs of PM_K_SUB=16 K-each). Accumulates into inner[PM_N_TILES_PER_WARP][4]
// (one accumulator per N-tile, summed across all sub-MMAs). Reads dequantized
// BF16 weights from smem_B [k][n] (cooperatively dequantized ONCE per K-step by
// all 256 threads — amortizing the E4M3 conversion across the 8 warps that share
// each weight). Each sub-MMA's fragment layout is byte-for-byte identical to
// w8a16_gemm's helper, just shifted by the sub-K offset (s*PM_K_SUB).
__device__ __forceinline__ void pm_mma_kstep(
    const __nv_bfloat16* smem_A,   // [PM_M_TILE][PM_A_STRIDE]
    const __nv_bfloat16* smem_B,   // [PM_N_TILE][PM_K_STEP + PM_PAD] (K-contiguous)
    float inner[PM_N_TILES_PER_WARP][4],
    unsigned int warp_m_offset, unsigned int group_id, unsigned int tid
) {
    const unsigned int a_stride = PM_A_STRIDE;
    const unsigned int b_stride = PM_K_STEP + PM_PAD;   // [n][k] K-contiguous stride
    const unsigned short* sA = (const unsigned short*)smem_A;
    const unsigned short* sB = (const unsigned short*)smem_B;

    unsigned int frag_r0 = warp_m_offset + group_id;
    unsigned int frag_r1 = warp_m_offset + group_id + 8;

    #pragma unroll
    for (int s = 0; s < PM_K_SUBS; s++) {
        const unsigned int k_off = s * PM_K_SUB;     // K offset of this sub-MMA within the step
        // A fragment columns for this sub-MMA (16-K window starting at k_off).
        unsigned int frag_c0 = k_off + tid * 2;
        unsigned int frag_c1 = k_off + tid * 2 + 8;

        unsigned int a0 = *(const unsigned int*)&sA[frag_r0 * a_stride + frag_c0];
        unsigned int a1 = *(const unsigned int*)&sA[frag_r1 * a_stride + frag_c0];
        unsigned int a2 = *(const unsigned int*)&sA[frag_r0 * a_stride + frag_c1];
        unsigned int a3 = *(const unsigned int*)&sA[frag_r1 * a_stride + frag_c1];

        #pragma unroll
        for (int n_tile = 0; n_tile < PM_N_TILES_PER_WARP; n_tile++) {
            unsigned int n_col = n_tile * 8 + group_id;
            unsigned int k0 = k_off + tid * 2;
            unsigned int k1 = k_off + tid * 2 + 8;

            // [n][k] K-contiguous: (k, k+1) are adjacent → single aligned u32.
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

/// W8A16 pipelined GEMM: B[N,K] row-major FP8 E4M3 with 2D block scales.
/// 128×32 tile (M×N), 8 warps, PM_STAGES-deep cp.async prefetch pipeline.
extern "C" __global__ void w8a16_gemm_pipelined(
    const __nv_bfloat16* __restrict__ A,            // [M, K] BF16 activations
    const unsigned char* __restrict__ B,             // [N, K] FP8 E4M3
    const float* __restrict__ block_scale,           // [N/128, K/128] FP32
    __nv_bfloat16* __restrict__ C,                   // [M, N] BF16 output
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int cta_m = blockIdx.y * PM_M_TILE;
    const unsigned int cta_n = blockIdx.x * PM_N_TILE;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;   // 8 warps × 16 = 128 M-rows
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    // Pipelined smem. cp.async destinations (smem_A, smem_Braw) are
    // __align__(16) with 16-byte-aligned row strides so every 16-B
    // cp.async.cg chunk is naturally aligned. smem_B (MMA-ready [k][n] BF16) is
    // the cooperatively-dequantized weight buffer: all 256 threads convert the
    // K-step's weights ONCE (amortized across the 8 warps that share them) —
    // fusing dequant into the MMA instead multiplies the LUT cost 8× and
    // serializes the constant cache (measured 5× slowdown). Per stage at
    // N_TILE=32: smem_A 128*24*2=6144 B + smem_B 16*34*2=1088 B + smem_Braw
    // 32*16=512 B ≈ 7744 B → 15488 B for 2 stages.
    __shared__ __align__(16) __nv_bfloat16 smem_A[PM_STAGES][PM_M_TILE][PM_A_STRIDE];
    // Lever 3: smem_B stored [n][k] (K-CONTIGUOUS), transposed from the prior
    // [k][n] layout. The MMA B fragment packs two consecutive-K BF16 elements
    // (k, k+1) per 32-bit register; with K contiguous that pair is a SINGLE
    // aligned 32-bit smem load instead of two strided 16-bit loads + a shift/or.
    // Row stride = PM_K_STEP + PM_PAD shorts (34 at K_STEP=32) keeps the u32
    // reads off the same bank and the rows 16-bit aligned for the u32 cast.
    __shared__ __nv_bfloat16 smem_B[PM_STAGES][PM_N_TILE][PM_K_STEP + PM_PAD];
    __shared__ __align__(16) unsigned char smem_Braw[PM_STAGES][PM_N_TILE][PM_K_STEP];

    // ── Lever 1: stage the E4M3→FP32 LUT in SHARED memory ──
    // The dequant indexes the table with DATA-DEPENDENT weight bytes; reads from
    // __constant__ memory SERIALIZE across a warp whenever the 32 lanes hit
    // divergent table entries (the constant cache is a broadcast cache, optimal
    // only for UNIFORM access). Shared memory services divergent indices in
    // parallel (one transaction per distinct bank, no broadcast restriction).
    // Copy the 256-entry table from the __constant__ SSOT into smem ONCE per CTA
    // (256 threads → one element each), then dequant reads the smem copy.
    __shared__ float smem_lut[256];
    smem_lut[threadIdx.x] = E4M3_LUT[threadIdx.x];   // PM_THREADS == 256, exact cover
    __syncthreads();

    // Two-level FP32 accumulation (see file header — PRESERVED EXACTLY).
    float inner_acc[PM_N_TILES_PER_WARP][4];
    float outer_acc[PM_N_TILES_PER_WARP][4];
    #pragma unroll
    for (int i = 0; i < PM_N_TILES_PER_WARP; i++) {
        inner_acc[i][0] = 0.0f; inner_acc[i][1] = 0.0f;
        inner_acc[i][2] = 0.0f; inner_acc[i][3] = 0.0f;
        outer_acc[i][0] = 0.0f; outer_acc[i][1] = 0.0f;
        outer_acc[i][2] = 0.0f; outer_acc[i][3] = 0.0f;
    }

    const unsigned int k_blocks = K / PM_FP8_BLOCK;
    const unsigned int k_steps_per_block = PM_FP8_BLOCK / PM_K_STEP;
    const unsigned int n_block = cta_n / PM_FP8_BLOCK;
    const unsigned int n_steps = (K + PM_K_STEP - 1) / PM_K_STEP;

    // A-tile cp.async: 128 rows × PM_K_STEP K-cols BF16. Each 16-B chunk = 8
    // BF16. At K_STEP=32: 128×32/8 = 512 chunks → 2 chunks/thread (256 threads).
    const unsigned int a_chunks = (PM_M_TILE * PM_K_STEP) / 8;     // 512 at K_STEP=32

    // Issue cp.async loads for K-step `step` into double-buffer `stage`. The
    // copies are contiguous along K (the contiguous global axis for both A
    // [M,K] and B [N,K]) so each 16-B cp.async pulls a real run. Bounds /
    // K-tail fall back to a masked scalar copy (cp.async cannot predicate).
    auto prefetch = [&](unsigned int step, unsigned int stage) {
        unsigned int k_base = step * PM_K_STEP;

        // ── A: contiguous 16-B (8 BF16) chunks along K ──
        #pragma unroll
        for (unsigned int c = threadIdx.x; c < a_chunks; c += PM_THREADS) {
            unsigned int row = (c * 8) / PM_K_STEP;          // 0..127
            unsigned int col = (c * 8) % PM_K_STEP;          // 0 or 8
            unsigned int gr = cta_m + row;
            unsigned int gc = k_base + col;
            __nv_bfloat16* dst = &smem_A[stage][row][col];
            if (gr < M && gc + 8 <= K) {
                cp_async_cg_16(dst, &A[(unsigned long long)gr * K + gc]);
            } else {
                #pragma unroll
                for (unsigned int e = 0; e < 8; e++) {
                    unsigned int gcol = gc + e;
                    dst[e] = (gr < M && gcol < K) ? A[(unsigned long long)gr * K + gcol]
                                                  : __float2bfloat16(0.0f);
                }
            }
        }

        // ── B raw: contiguous 16-B (16 FP8-byte) chunks of K per N-row ──
        // smem_Braw[stage][n][k] mirrors global B[n, k_base + k] contiguously.
        // At K_STEP=32 each N-row is 32 bytes = two 16-B chunks; iterate over
        // (PM_N_TILE × PM_K_STEP/16) chunks, one cp.async per chunk.
        const unsigned int b_chunks = (PM_N_TILE * PM_K_STEP) / 16;   // 64 at K_STEP=32
        #pragma unroll
        for (unsigned int c = threadIdx.x; c < b_chunks; c += PM_THREADS) {
            unsigned int nrow = (c * 16) / PM_K_STEP;        // 0..PM_N_TILE-1
            unsigned int kcol = (c * 16) % PM_K_STEP;        // 0 or 16
            unsigned int gn = cta_n + nrow;
            unsigned int gk = k_base + kcol;
            unsigned char* dst = &smem_Braw[stage][nrow][kcol];
            if (gn < N && gk + 16 <= K) {
                cp_async_cg_16(dst, &B[(unsigned long long)gn * K + gk]);
            } else {
                #pragma unroll
                for (unsigned int e = 0; e < 16; e++) {
                    unsigned int gke = gk + e;
                    dst[e] = (gn < N && gke < K) ? B[(unsigned long long)gn * K + gke] : 0;
                }
            }
        }
        cp_async_commit();
    };

    // LUT-dequant just-arrived raw B for `stage` into the MMA-ready BF16 buffer.
    // Lever 3: smem_B is now ALSO [n][k] (K-contiguous), matching smem_Braw, so
    // this is a same-layout element-wise dequant (no transpose). No scale here
    // (folded on the FP32 accumulator at the block boundary). Cooperative across
    // all 256 threads so each weight is converted once and reused by all 8 warps.
    // (Tested a direct BF16-bits LUT to skip the float→BF16 cast: bit-identical
    // but ~1% SLOWER and no register change — the float-LUT + __float2bfloat16
    // path already fuses optimally, so it is kept.)
    auto dequant_B = [&](unsigned int stage) {
        #pragma unroll
        for (unsigned int idx = threadIdx.x; idx < PM_K_STEP * PM_N_TILE; idx += PM_THREADS) {
            unsigned int n = idx / PM_K_STEP;     // 0..PM_N_TILE-1
            unsigned int k = idx % PM_K_STEP;     // 0..PM_K_STEP-1
            unsigned char wb = smem_Braw[stage][n][k];
            smem_B[stage][n][k] = __float2bfloat16(smem_lut[wb]);
        }
    };

    // ── Software-pipelined main loop (PM_STAGES-deep cp.async) ──
    // Prologue: issue the first PM_STAGES-1 prefetches so that, on entering the
    // main loop, the consuming stage for step 0 is already committed together
    // with PM_STAGES-2 look-ahead stages. Each prefetch commits one cp.async
    // group; we drain them from the head (FIFO) as we consume.
    #pragma unroll
    for (unsigned int p = 0; p < PM_STAGES - 1; p++) {
        if (p < n_steps) {
            prefetch(p, p % PM_STAGES);
        }
    }
    unsigned int k_step_in_block = 0;

    for (unsigned int step = 0; step < n_steps; step++) {
        unsigned int cur = step % PM_STAGES;

        // Prefetch the stage PM_STAGES-1 K-steps ahead (keeps the pipeline
        // full) before consuming the current one, so its global loads overlap
        // the dequant + MMA below.
        unsigned int ahead = step + (PM_STAGES - 1);
        if (ahead < n_steps) {
            prefetch(ahead, ahead % PM_STAGES);
        }
        // Drain so the `cur` stage (the oldest outstanding cp.async group) is
        // guaranteed complete. Groups complete FIFO. After this step's
        // prefetch, the total committed = min(n_steps, PM_STAGES + step) and
        // `cur` is the step-th (0-indexed); the number of groups committed
        // AFTER cur that should stay in flight is therefore:
        //     target = min(n_steps, PM_STAGES + step) - (step + 1)
        // which is PM_STAGES-1 in steady state and shrinks to 0 at the tail.
        unsigned int committed = min(n_steps, PM_STAGES + step);
        unsigned int target = committed - (step + 1);
        cp_async_wait_le(target);
        __syncthreads();   // raw B for `cur` is now resident for all threads

        dequant_B(cur);
        __syncthreads();   // smem_B[cur] BF16 fully written before MMA reads it

        pm_mma_kstep(&smem_A[cur][0][0], &smem_B[cur][0][0],
                     inner_acc, warp_m_offset, group_id, tid);
        __syncthreads();   // done reading smem_*[cur]; safe for next reuse

        // K_BLOCK boundary: fold scaled inner into outer, reset inner.
        k_step_in_block++;
        if (k_step_in_block == k_steps_per_block) {
            const unsigned int k_block = (step * PM_K_STEP) / PM_FP8_BLOCK;
            const float scale = block_scale[n_block * k_blocks + k_block];
            #pragma unroll
            for (int i = 0; i < PM_N_TILES_PER_WARP; i++) {
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
        const unsigned int k_block = (K - 1) / PM_FP8_BLOCK;
        const float scale = block_scale[n_block * k_blocks + k_block];
        #pragma unroll
        for (int i = 0; i < PM_N_TILES_PER_WARP; i++) {
            outer_acc[i][0] += inner_acc[i][0] * scale;
            outer_acc[i][1] += inner_acc[i][1] * scale;
            outer_acc[i][2] += inner_acc[i][2] * scale;
            outer_acc[i][3] += inner_acc[i][3] * scale;
        }
    }

    // ── Store C tile: f32 outer accumulators → BF16 output ──
    #pragma unroll
    for (int n_tile = 0; n_tile < PM_N_TILES_PER_WARP; n_tile++) {
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
