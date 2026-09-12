// SPDX-License-Identifier: AGPL-3.0-only

// Atlas W8A16 tensor-core DECODE GEMM — 16-row M tile, FP8 E4M3 block-scaled.
//
//   C[M,N] = A[M,K] (BF16) * dequant(B[N,K] (FP8 E4M3)),  1 <= M <= 16
//
// WHY (#927, 1xH100, 2026-09-11, Qwen/Qwen3.8-27B-FP8, tip `2962cfed7`). At a
// decode batch of 16 the step costs 86.7 ms; the 48 SSM layers are 63.3 ms of
// it and the dense FFN inside them is 63% (~833 us/layer). The tier that
// serves those widths today, `w8a16_gemv_batch16`, is bit-exact but
// FP32-FMA-BOUND at M=16: 0.260 ms / 342 GB/s on gate/up (N=17408, K=5120) and
// 0.330 ms / 270 GB/s on down (N=5120, K=17408). The 89 MB FP8 weight matrix
// should stream in ~30 us on HBM3 (~3,000 GB/s), so the GEMV is leaving ~9x on
// the floor. Its arithmetic is the wall, not the memory: per weight BYTE it
// runs ~16 scalar FFMA plus 16 BF16->FP32 converts plus a LUT lookup and a
// scale multiply — ~37 ALU ops/byte, which at the SM's FP32 issue rate caps it
// near 350 GB/s no matter how fast the DRAM is.
//
// This kernel replaces those 16 scalar FFMA per byte with ONE m16n8k16 MMA
// lane-slot (the M tile IS 16, so nothing is padded — that is the whole
// difference from the tile GEMMs, which pad M to 128 and waste 7/8 of every
// tile) and cuts the dequant to ~2 instructions per weight byte. The remaining
// per-byte cost is ~0.17 ALU ops, i.e. the kernel becomes weight-BANDWIDTH
// bound, which is what the shape has always been.
//
// ── NUMERICS: REASSOCIATED, NOT BIT-IDENTICAL ─────────────────────────────
// `w8a16_gemv` / `w8a16_gemv_batch{4,16}` reduce each output in ONE FP32
// accumulator walked in strict K order. An MMA reduces 16 K-products in the
// tensor core's own (unspecified, but fixed) order before it reaches the FP32
// accumulator, and the accumulator is then summed across 4 sub-MMAs per K-step.
// So this kernel is NOT bit-identical to the scalar GEMV, and it is not meant
// to be — the contract is <= 2 BF16 ULP per element, which
// `examples/native_fp8_ffn_m16_tc_microtest.rs` measures.
//
// That is not a new seam. The arm the FFN used at these widths BEFORE #927 was
// `w8a16_gemm_n128_m128` / `w8a16_gemm_pipelined`, both m16n8k16 MMA kernels
// with exactly this reassociation. #927 moved 5..=32 onto the bit-exact GEMV;
// this kernel moves them back onto an MMA, which is why it is behind
// `ATLAS_FFN_M16_TC` and OFF by default until an H100 receipt says it wins.
// The FFN dispatch rule states the same thing (`dense_ffn_m16_tc.rs`).
//
// The two-level FP32 fold is PRESERVED EXACTLY from `w8a16_gemm_pipelined` /
// `fp8_gemm_t_blockscaled`: the MMAs accumulate UNSCALED (smem holds the
// lossless BF16 cast of the E4M3 byte — 3 mantissa bits and a 4-bit exponent
// are a strict subset of BF16's 7 and 8), and at each 128-K block boundary
//     outer += inner * block_scale[n_block, k_block];  inner = 0
// so the block scale is applied ONCE per block on an FP32 accumulator, never
// per element and never folded into BF16. That is what holds the deep-layer
// FP8 precision floor.
//
// DEQUANT. `cvt.rn.f16x2.e4m3x2` (sm_89+) decodes two E4M3 bytes per
// instruction; FP16 represents every finite E4M3 value exactly (5-bit exponent
// vs 4, 10 mantissa bits vs 3), and FP32->BF16 round-to-nearest-even is exact
// for <= 3 mantissa bits, so the hardware path yields the SAME BF16 bits as the
// shared `E4M3_LUT`. The one divergence is NaN: E4M3 0x7F/0xFF are the format's
// only NaNs and `E4M3_LUT` decodes them to +-0, where `cvt` yields NaN. A
// block-scaled FP8 checkpoint cannot contain those bytes (they are not in the
// quantizer's output alphabet) and the oracle draws from the same 0x00..0x7E /
// 0x80..0xFE alphabet the batch4 oracle uses. Arches below sm_89 take the
// `E4M3_LUT` fallback below and keep the +-0 behaviour.
//
// ── GEOMETRY ──────────────────────────────────────────────────────────────
// CTA = 4 warps (128 threads) covering [16 M x 32 N]. Each warp owns 8 N
// columns = exactly ONE m16n8k16 tile, so every MMA slot does real work at
// M=16 and the accumulator is 4 inner + 4 outer FP32 registers.
//
// N_TILE is a TEMPLATE PARAMETER with two instantiations, 32 (the default) and
// 64 (`w8a16_gemm_m16_n64`, opt-in via `ATLAS_FFN_M16_TC_NTILE=64`).
//
// 32 is the default because it is the one with a receipt. 64 amortizes the
// (shared) A fragment loads over twice the weight bytes, but it gives the down
// projection (N=5120) only ceil(5120/64) = 80 CTAs on a 132-SM H100 — 52 SMs
// idle and ~10 KB of in-flight cp.async per SM, which is not enough
// memory-level parallelism to cover HBM latency. At N_TILE=32 the grid is 160
// CTAs for down and 544 for gate/up (N=17408); every SM is fed and all CTAs are
// resident at once. The cost is that each A element is read by twice as many
// CTAs, but A is 16 rows (<= 550 KB for the whole K) and stays L2-resident, so
// that traffic never reaches DRAM.
//
// WHY 64 EXISTS ANYWAY (round 6, 1xH100, 2026-09-11). The microtest says this
// kernel is worth 3.71x over `w8a16_gemv_batch16` at M=16 on gate/up
// (1,296.5 vs 349.6 GB/s) and 3.39x on down (929.9 vs 274.3), yet the SERVING
// A/B at bs16 with the lever on measured the attention tiers -21.7% and the
// SSM-layer FFN +13.7% — the FFN arm LOSES on the box where the microtest
// wins. The two differ only in N: the attention tiers run N=6144/1024/5120
// (192/32/160 CTAs) and the FFN runs N=17408 (544 CTAs). At 19,456 B of smem
// and `__launch_bounds__(128, 4)` an H100 SM holds 4 CTAs, so 132 SMs hold 528
// — the FFN's 544 CTAs are ONE FULL WAVE PLUS A 16-CTA TAIL, and that tail
// serialises a second whole wave's worth of latency behind 3% of the work,
// while the attention tiers all fit inside a single partial wave. N_TILE=64
// halves the FFN's grid to 272 CTAs (well inside one wave) and doubles the A
// reuse. It is a HYPOTHESIS with no receipt yet — hence opt-in, default 32 —
// and the A/B that would settle it is `ATLAS_FFN_M16_TC_NTILE=64` against the
// same serve. Full reasoning and the competing L2 hypothesis:
// `dense_ffn_m16_tc.rs`.
//
// K_STEP=64 with a 4-stage cp.async pipeline on the RAW FP8 bytes (32 n x 64 k
// = 2 KB per stage) and on the 16x64 BF16 activation slice (2 KB per stage).
// No TMA, no cp.async.bulk — they silently corrupt on sm_121 (see
// w8a16_gemm_pipelined.cu); cp.async.cg is correct there.
//
// smem: A 4*16*72*2 = 9,216 B + Braw 4*N_TILE*80 = 10,240 B at N_TILE=32
// (19,456 B total, 19 KB) or 20,480 B at N_TILE=64 (29,696 B, 29 KB), so 5 resp.
// 3 CTAs/SM fit sm_121's 100 KB budget, 4 resp. 4 fit H100's 228 KB, and the
// 48 KB static limit is untouched by either.
// Row pitches are padded (A: 64+8 BF16 = 144 B; B: 64+16 = 80 B) so that both
// the 16-byte cp.async chunks stay aligned AND the 32 lanes of a warp hit 32
// distinct shared-memory banks on every fragment load.
//
// NO ldmatrix: the fragment registers are built from plain smem loads, the way
// `dense_gemm_tc.cu` does it (ldmatrix.x4 is broken on sm_121).
//
// Rows >= M are ZERO-FILLED in smem and never stored, so a caller may pass any
// 1 <= M <= 16 and the kernel neither reads nor writes outside [M, N].
//
// Grid: (ceil(N/N_TILE), 1, 1)  Block: (128, 1, 1). Requires K % 128 == 0 (the
// block-scale granularity) and, for the strided entry point, an activation row
// pitch that is a multiple of 8 BF16 (the cp.async chunks are 16 B).
//
// Entry points:
//   `w8a16_gemm_m16`          N_TILE=32, contiguous A [M,K] and C [M,N]
//   `w8a16_gemm_m16_strided`  N_TILE=32, caller-supplied A/C row pitches in
//                             ELEMENTS, for the multi-seq QKV buffer, which is
//                             [n, per_seq_qkv] with Q/K/V at fixed offsets
//                             inside each row
//   `w8a16_gemm_m16_n64`      N_TILE=64, contiguous — the wide-tile A/B arm.
//                             No strided twin: the only strided consumer is the
//                             multi-seq QKV tier, whose N (1024/6144) is
//                             already CTA-starved at 32 and would be worse at
//                             64. Add one when a shape asks for it.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "e4m3_lut.cuh"   // shared E4M3 -> FP32 SSOT (pre-sm_89 fallback)

#define M16_M_TILE 16
// The two instantiated N tiles. 32 is the default and the one with a receipt;
// 64 is the opt-in wide arm (`ATLAS_FFN_M16_TC_NTILE=64`). Both must divide
// M16_FP8_BLOCK so a CTA's columns lie inside ONE 128-wide scale block.
#define M16_N_TILE 32
#define M16_N_TILE_WIDE 64
#define M16_K_STEP 64
#define M16_K_SUB 16                                // one m16n8k16's K width
#define M16_K_SUBS (M16_K_STEP / M16_K_SUB)         // = 4
#define M16_WARPS 4
#define M16_THREADS (M16_WARPS * 32)                // 128
#define M16_N_PER_MMA 8                             // one m16n8k16's N width
#define M16_STAGES 4
#define M16_FP8_BLOCK 128
#define M16_A_STRIDE 72                             // BF16 elems: 64 + 8 pad
#define M16_B_STRIDE 80                             // bytes: 64 + 16 pad

// cp.async.cg 16-byte (cache-global) copy: smem <- global. sm_80+; correct on
// sm_121, unlike TMA / cp.async.bulk. Requires 16-byte-aligned addresses.
__device__ __forceinline__ void m16_cp_async_cg_16(void* smem_ptr, const void* gmem_ptr) {
    unsigned int s = (unsigned int)__cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem_ptr));
}
__device__ __forceinline__ void m16_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void m16_cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// Wait until at most `n` cp.async groups remain in flight. The PTX operand must
// be a compile-time immediate, so dispatch the runtime count (always in
// [0, M16_STAGES-1]) through a switch over its legal values.
__device__ __forceinline__ void m16_cp_async_wait_le(unsigned int n) {
    switch (n) {
        case 0:  m16_cp_async_wait_group<0>(); break;
        case 1:  m16_cp_async_wait_group<1>(); break;
        case 2:  m16_cp_async_wait_group<2>(); break;
        default: m16_cp_async_wait_group<3>(); break;
    }
}

// Two E4M3 bytes (low = weight k, high = weight k+1) -> one BF16x2 register in
// the SAME halves, which is exactly the m16n8k16 B fragment's packing. See the
// DEQUANT note in the header for why the two paths agree bit-for-bit on every
// byte a checkpoint can contain.
__device__ __forceinline__ unsigned int m16_dequant_pair(unsigned short raw) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 890)
    unsigned int h2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(raw));
    float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h2));
    __nv_bfloat162 b = __floats2bfloat162_rn(f.x, f.y);
    return *reinterpret_cast<const unsigned int*>(&b);
#else
    __nv_bfloat162 b = __floats2bfloat162_rn(E4M3_LUT[raw & 0xFFu], E4M3_LUT[raw >> 8]);
    return *reinterpret_cast<const unsigned int*>(&b);
#endif
}

/// W8A16 tensor-core decode GEMM body. `a_row_stride` / `c_row_stride` are the
/// A and C row pitches in ELEMENTS; the contiguous entry point passes K and N.
///
/// `N_TILE` is the CTA's N width: 32 (default) or 64 (the wide arm). It must be
/// a multiple of 32 — one 8-wide MMA tile per warp per sub-step — and a divisor
/// of the 128-wide FP8 scale block, so `n_block` stays constant for the CTA.
template <int N_TILE>
__device__ __forceinline__ void w8a16_gemm_m16_impl(
    const __nv_bfloat16* __restrict__ A,     // [M, a_row_stride] BF16, K used
    const unsigned char* __restrict__ B,      // [N, K] FP8 E4M3
    const float* __restrict__ block_scale,    // [N/128, K/128] FP32
    __nv_bfloat16* __restrict__ C,            // [M, c_row_stride] BF16, N used
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int a_row_stride,
    unsigned int c_row_stride
) {
    static_assert(N_TILE % 32 == 0, "N_TILE must be a whole number of MMA N-tiles per warp");
    static_assert(M16_FP8_BLOCK % N_TILE == 0, "a CTA's columns must lie in ONE scale block");
    // MMA N-tiles per warp (1 at N_TILE=32, 2 at 64) and 16-byte B chunks per
    // thread per stage (same numbers, for the same reason: both scale with the
    // weight bytes the CTA stages).
    constexpr int N_PER_WARP = N_TILE / M16_WARPS;
    constexpr int N_SUBS = N_PER_WARP / M16_N_PER_MMA;
    constexpr int B_CHUNKS = N_TILE / 32;

    const unsigned int cta_n = blockIdx.x * N_TILE;
    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int lane_id = threadIdx.x & 31;
    const unsigned int group_id = lane_id >> 2;     // 0..7: MMA fragment row
    const unsigned int quad = lane_id & 3;          // 0..3: MMA fragment K pair
    const unsigned int warp_n = warp_id * N_PER_WARP;

    __shared__ __align__(16) __nv_bfloat16 smem_A[M16_STAGES][M16_M_TILE][M16_A_STRIDE];
    __shared__ __align__(16) unsigned char smem_Braw[M16_STAGES][N_TILE][M16_B_STRIDE];

    // Two-level FP32 accumulation (see header). N_SUBS m16n8k16 tiles per warp,
    // so the whole accumulator is 8 * N_SUBS registers.
    float inner[4 * N_SUBS];
    float outer[4 * N_SUBS];
    #pragma unroll
    for (int i = 0; i < 4 * N_SUBS; i++) {
        inner[i] = 0.0f;
        outer[i] = 0.0f;
    }

    const unsigned int n_steps = K / M16_K_STEP;
    const unsigned int k_blocks = K / M16_FP8_BLOCK;
    const unsigned int k_steps_per_block = M16_FP8_BLOCK / M16_K_STEP;   // 2
    // The CTA's 32 N-columns start at a multiple of 32 and span 32, so they lie
    // inside ONE 128-wide scale block: n_block is constant for the whole CTA.
    const unsigned int n_block = cta_n / M16_FP8_BLOCK;

    // Stage one K-step into `stage`. The A tile is 2 KB = 128 chunks of 16 B,
    // i.e. EXACTLY one cp.async per thread; the B tile is 2 KB per 32 N rows, so
    // B_CHUNKS per thread. The copies run along K, the contiguous global axis
    // for A [M,K] and B [N,K] alike. Out-of-range rows (activation rows >= M,
    // weight rows >= N) are zero-filled by hand — cp.async cannot predicate, and
    // zero weights/activations contribute nothing to the MMA, which is what
    // makes any 1 <= M <= 16 legal.
    auto prefetch = [&](unsigned int step, unsigned int stage) {
        const unsigned int k_base = step * M16_K_STEP;
        {
            const unsigned int row = threadIdx.x >> 3;              // 0..15
            const unsigned int col = (threadIdx.x & 7) * 8;         // 0..56
            __nv_bfloat16* dst = &smem_A[stage][row][col];
            if (row < M) {
                m16_cp_async_cg_16(dst, &A[(unsigned long long)row * a_row_stride + k_base + col]);
            } else {
                #pragma unroll
                for (int e = 0; e < 8; e++) dst[e] = __float2bfloat16(0.0f);
            }
        }
        #pragma unroll
        for (int c = 0; c < B_CHUNKS; c++) {
            const unsigned int row = (threadIdx.x >> 2) + c * 32;   // 0..N_TILE-1
            const unsigned int col = (threadIdx.x & 3) * 16;        // 0..48
            const unsigned int gn = cta_n + row;
            unsigned char* dst = &smem_Braw[stage][row][col];
            if (gn < N) {
                m16_cp_async_cg_16(dst, &B[(unsigned long long)gn * K + k_base + col]);
            } else {
                #pragma unroll
                for (int e = 0; e < 16; e++) dst[e] = 0;
            }
        }
        m16_cp_async_commit();
    };

    #pragma unroll
    for (unsigned int p = 0; p < M16_STAGES - 1; p++) {
        if (p < n_steps) prefetch(p, p);
    }

    unsigned int k_step_in_block = 0;
    for (unsigned int step = 0; step < n_steps; step++) {
        const unsigned int cur = step % M16_STAGES;
        // Groups complete FIFO. Before this iteration issues its own prefetch,
        // min(n_steps, STAGES-1+step) groups are committed and `cur` is the
        // step-th, so the number that may stay in flight is:
        const unsigned int committed = min(n_steps, M16_STAGES - 1 + step);
        m16_cp_async_wait_le(committed - (step + 1));
        // ONE barrier per K-step. It does double duty: stage `cur` is now
        // visible to every warp, AND every warp has finished the MMA of step-1,
        // which is what makes the prefetch below (it targets stage
        // (step-1) % STAGES) safe to issue without a second barrier.
        __syncthreads();
        const unsigned int ahead = step + M16_STAGES - 1;
        if (ahead < n_steps) prefetch(ahead, ahead % M16_STAGES);

        const unsigned short* sA = (const unsigned short*)&smem_A[cur][0][0];
        const unsigned char* sB = &smem_Braw[cur][0][0];
        #pragma unroll
        for (int s = 0; s < M16_K_SUBS; s++) {
            // m16n8k16 row.col fragments, built straight from smem (no
            // ldmatrix): A rows {group_id, group_id+8} x K pairs
            // {quad*2, quad*2+8}; B row (warp_n + j*8 + group_id) at the same K
            // pairs. The A fragment is loaded ONCE and reused by all N_SUBS
            // MMAs — that reuse is the whole point of the wide tile.
            const unsigned int kc0 = s * M16_K_SUB + quad * 2;
            const unsigned int kc1 = kc0 + 8;
            const unsigned int r0 = group_id * M16_A_STRIDE;
            const unsigned int r1 = (group_id + 8) * M16_A_STRIDE;
            const unsigned int a0 = *(const unsigned int*)&sA[r0 + kc0];
            const unsigned int a1 = *(const unsigned int*)&sA[r1 + kc0];
            const unsigned int a2 = *(const unsigned int*)&sA[r0 + kc1];
            const unsigned int a3 = *(const unsigned int*)&sA[r1 + kc1];
            #pragma unroll
            for (int j = 0; j < N_SUBS; j++) {
                const unsigned char* brow =
                    &sB[(warp_n + j * M16_N_PER_MMA + group_id) * M16_B_STRIDE];
                const unsigned int b0 = m16_dequant_pair(*(const unsigned short*)&brow[kc0]);
                const unsigned int b1 = m16_dequant_pair(*(const unsigned short*)&brow[kc1]);
                // Index `inner` directly rather than through a pointer: `j` is
                // a compile-time constant under the unroll, so these stay
                // registers. A `float*` into the array would force it to local
                // memory and the accumulator would spill every K-step.
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5, %6, %7}, "
                    "{%8, %9}, "
                    "{%10, %11, %12, %13};"
                    : "=f"(inner[j * 4 + 0]), "=f"(inner[j * 4 + 1]),
                      "=f"(inner[j * 4 + 2]), "=f"(inner[j * 4 + 3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                      "r"(b0), "r"(b1),
                      "f"(inner[j * 4 + 0]), "f"(inner[j * 4 + 1]),
                      "f"(inner[j * 4 + 2]), "f"(inner[j * 4 + 3])
                );
            }
        }

        // 128-K block boundary: fold the UNSCALED inner accumulator onto the
        // outer one exactly once, with this block's scale. One scale for the
        // whole CTA — every N_SUBS tile is inside the same 128-wide block.
        if (++k_step_in_block == k_steps_per_block) {
            const float scale = block_scale[n_block * k_blocks + step / k_steps_per_block];
            #pragma unroll
            for (int i = 0; i < 4 * N_SUBS; i++) {
                outer[i] += inner[i] * scale;
                inner[i] = 0.0f;
            }
            k_step_in_block = 0;
        }
    }

    // ── Store: FP32 outer accumulators -> BF16, masked to [M, N] ──
    const unsigned int row0 = group_id;
    const unsigned int row1 = group_id + 8;
    const unsigned long long o0 = (unsigned long long)row0 * c_row_stride;
    const unsigned long long o1 = (unsigned long long)row1 * c_row_stride;
    #pragma unroll
    for (int j = 0; j < N_SUBS; j++) {
        const unsigned int col0 = cta_n + warp_n + j * M16_N_PER_MMA + quad * 2;
        const unsigned int col1 = col0 + 1;
        if (row0 < M && col0 < N) C[o0 + col0] = __float2bfloat16(outer[j * 4 + 0]);
        if (row0 < M && col1 < N) C[o0 + col1] = __float2bfloat16(outer[j * 4 + 1]);
        if (row1 < M && col0 < N) C[o1 + col0] = __float2bfloat16(outer[j * 4 + 2]);
        if (row1 < M && col1 < N) C[o1 + col1] = __float2bfloat16(outer[j * 4 + 3]);
    }
}

/// Contiguous A `[M, K]` and C `[M, N]` — the dense-FFN gate/up/down shape.
extern "C" __global__
__launch_bounds__(M16_THREADS, 4)
void w8a16_gemm_m16(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    const float* __restrict__ block_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    w8a16_gemm_m16_impl<M16_N_TILE>(A, B, block_scale, C, M, N, K, K, N);
}

/// N_TILE=64 twin of [`w8a16_gemm_m16`] — SAME arguments, SAME per-output
/// arithmetic (the two-level fold and the m16n8k16 K order are untouched; only
/// which CTA owns a column changes), HALF the CTAs, and each staged A tile
/// feeds twice the weight bytes. Grid is `ceil(N/64)`.
///
/// Opt-in (`ATLAS_FFN_M16_TC_NTILE=64`) and NOT the default: it exists to A/B
/// the round-6 FFN regression (+13.7% on the SSM-layer FFN at bs16 while the
/// attention tiers went -21.7%), whose leading hypothesis is the FFN's 544 CTAs
/// overrunning the H100's 528-CTA residency by one 16-CTA tail. 64 takes that
/// to 272. See `dense_ffn_m16_tc.rs` for the full argument and the competing
/// L2-thrash hypothesis; neither has a receipt yet.
extern "C" __global__
__launch_bounds__(M16_THREADS, 4)
void w8a16_gemm_m16_n64(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    const float* __restrict__ block_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    w8a16_gemm_m16_impl<M16_N_TILE_WIDE>(A, B, block_scale, C, M, N, K, K, N);
}

/// Caller-supplied A and C row pitches in ELEMENTS — identical math and
/// identical accumulation order to `w8a16_gemm_m16`, for the multi-seq decode
/// Q/K/V buffer (`[n, per_seq_qkv]`, Q at 0, K after Q, V after K), which the
/// contiguous writer cannot address. Same split the `w8a16_gemv_batch4` /
/// `_strided` pair uses, and for the same reason.
extern "C" __global__
__launch_bounds__(M16_THREADS, 4)
void w8a16_gemm_m16_strided(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    const float* __restrict__ block_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int a_row_stride,
    unsigned int c_row_stride
) {
    w8a16_gemm_m16_impl<M16_N_TILE>(A, B, block_scale, C, M, N, K, a_row_stride, c_row_stride);
}
