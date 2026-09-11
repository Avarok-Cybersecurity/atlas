// SPDX-License-Identifier: AGPL-3.0-only

// Atlas DENSE BF16 tensor-core decode GEMM — 16-row M tile.
//
//   C[M,N] = A[M,K] (BF16) * B[N,K]^T (BF16),  1 <= M <= 16
//
// WHY (#927/#928, 1xH100, nsys round 7, 2026-09-11, Qwen/Qwen3.8-27B-FP8 with
// `--lm-head-dtype bf16` and `ATLAS_LM_HEAD_BATCHM_MAX=16`, decode batch 16).
// The BF16 LM head is the third-largest kernel in the 43.6 ms step and the only
// top-5 entry that is a SINGLE launch: `dense_gemv_bf16_batchm` costs
// **3,571 µs = 8.19% of the step** for one pass over the 2.54 GB vocab weight
// ([N=248077, K=5120] BF16), i.e. **~710 GB/s** against ~3,350 GB/s of HBM3.
// The same kernel at C=1 takes 798 µs — 3.2 TB/s-class, already at the memory
// roofline — so the 16-row tier costs 4.47x the 1-row tier for an IDENTICAL
// weight read. The arithmetic is the wall, not the memory.
//
// `dense_gemv_bf16_batchm` spends, per weight BYTE: 8 scalar FFMA (one per
// batched row at M=16, per BF16 element = 2 bytes), 8 BF16->FP32 converts of
// the activation and 0.5 of the weight, plus the smem round trip for the staged
// A tile. That is ~17 ALU ops per weight byte, which at the SM's FP32 issue
// rate caps the kernel near 700 GB/s no matter how fast the DRAM is — exactly
// where it measured.
//
// This kernel replaces those per-row FFMA with ONE `mma.sync.m16n8k16` lane
// slot. The M tile IS 16, so nothing is padded — that is the whole difference
// from `dense_gemm_tc.cu`, whose 16Mx64N tile pads the same way but reads the
// weight through a scalar inner loop, and from the W4A16/W8A16 tile GEMMs that
// pad M to 128 and waste 7/8 of every tile. The remaining per-byte cost is
// ~0.1 ALU ops, so the shape becomes weight-BANDWIDTH bound, which is what a
// 2.54 GB weight and 16 activation rows has always been.
//
// TARGET: <= 1.3 ms at M=16 on the real head shape = >= 1,950 GB/s, i.e. >=
// 2.75x the batched GEMV. Receipt:
// `examples/native_bf16_lm_head_m16_microtest.rs`.
//
// ── NUMERICS: REASSOCIATED, NOT BIT-IDENTICAL ─────────────────────────────
// `dense_gemv_bf16` / `dense_gemv_bf16_batchm` reduce each output in ONE FP32
// accumulator walked in strict K order (64 lanes striding by 64 over 8-wide
// uint4 chunks, lo-then-hi within each chunk, then a shfl butterfly and one
// cross-warp add). An MMA reduces 16 K-products in the tensor core's own
// (unspecified, but fixed) order before they reach the FP32 accumulator. So
// this kernel is NOT bit-identical to the scalar GEMV, and it is not meant to
// be — the contract is <= 2 BF16 ULP per element (or the accumulation floor
// for outputs that have catastrophically cancelled), the same predicate
// `w8a16_gemm_m16` is held to: `layers::dense_ffn::m16_tc::within_m16_tc_budget`.
//
// That is a seam at the LM head specifically, because a near-tie argmax flip
// changes the emitted token — which is why the head arm is behind
// `ATLAS_LM_HEAD_M16_TC` and OFF by default until an H100 receipt says it wins.
// `model/trait_impl/lm_head_batched.rs` states the same rule at the dispatch.
//
// NO BLOCK SCALE, NO DEQUANT. Unlike `w8a16_gemm_m16` (FP8 E4M3 weights, a
// 128-K block scale and a two-level FP32 fold), B here is already BF16 — the
// exact input type of the MMA — so a staged weight word IS a B fragment
// register and the accumulator is ONE level. That removes the `cvt` pair, the
// `E4M3_LUT` fallback, the `k % 128 == 0` scale-block requirement and the
// per-block fold entirely; the K constraint drops to the 64-wide pipeline step.
//
// ── GEOMETRY ──────────────────────────────────────────────────────────────
// CTA = 4 warps (128 threads) covering [16 M x N_TILE N]. Each warp owns
// N_TILE/4 columns = N_TILE/32 m16n8k16 tiles, so every MMA slot does real work
// at M=16 and the accumulator is 4 * N_SUBS FP32 registers (4 at N_TILE=32).
//
// N_TILE is a TEMPLATE PARAMETER with two instantiations, 32 (the default) and
// 64 (`dense_gemm_m16_bf16_n64`, opt-in via `ATLAS_LM_HEAD_M16_TC_NTILE=64`),
// mirroring `w8a16_gemm_m16`. 32 is the default because it is the tile the FFN
// receipt was taken on, and because `w8a16_gemm_m16`'s wave-quantisation
// argument does NOT apply here: at the vocab's N=248077 the grid is
// ceil(N/32) = 7,753 CTAs against an H100 residency of ~528, i.e. ~15 full
// waves, so a 16-CTA tail is 0.2% of the work rather than a second wave.
//
// WHY 64 EXISTS ANYWAY. The A tile is re-read by every CTA: at N_TILE=32 the
// 7,753 CTAs read 16x5120x2 = 160 KB each = 1.27 GB of L2 traffic against
// 2.54 GB of HBM traffic. A stays comfortably L2-resident (160 KB of a 50 MB
// L2), so none of it reaches DRAM, but it is not free either — it is ~50% more
// L2 reads than weight bytes. N_TILE=64 halves it to 0.63 GB for the same
// 2.54 GB weight read, at the cost of doubling the per-CTA smem. This is a
// HYPOTHESIS with no receipt; the A/B that settles it is the microtest's
// n64 column, and until it lands the default stays 32.
//
// K_STEP=64 with a 4-stage cp.async pipeline on the BF16 weights (N_TILE n x
// 64 k x 2 B = 4 KB per stage at N_TILE=32) and on the 16x64 BF16 activation
// slice (2 KB per stage). No TMA, no cp.async.bulk — they silently corrupt on
// sm_121 (see w8a16_gemm_pipelined.cu); cp.async.cg is correct there.
//
// smem: A 4*16*72*2 = 9,216 B + B 4*N_TILE*72*2 = 18,432 B at N_TILE=32
// (27,648 B total, 27 KB) or 36,864 B at N_TILE=64 (46,080 B, 45 KB). So 8
// resp. 4 CTAs/SM fit H100's 228 KB budget and 3 resp. 2 fit sm_121's 100 KB;
// both stay under the 48 KB static per-block limit, so no dynamic-smem opt-in.
//
// MEASURED (`nvcc -cubin -Xptxas -v --fmad=false`, CUDA 13.0.r13.0,
// 2026-09-11) — the numbers above are what ptxas actually produced, on every
// arch this file is built for:
//
//   entry                      sm_90a        sm_121f       sm_100f
//   dense_gemm_m16_bf16        44 reg 27,648 44 reg 27,648 44 reg 27,648
//   dense_gemm_m16_bf16_n64    60 reg 46,080 56 reg 46,080 56 reg 46,080
//
// 0 stack frame, 0 spill stores, 0 spill loads on all six. The 44/60 registers
// are what `__launch_bounds__(128, 4)` has to fit: 4 CTAs x 128 threads x 60
// regs = 30,720 of the SM's 65,536, so the register file is never the limiter
// and the smem figures above are.
// Row pitches are padded to 72 BF16 = 144 B on BOTH tiles so that the 16-byte
// cp.async chunks stay aligned (144 = 9 x 16) AND the 32 lanes of a warp hit 32
// distinct banks on every fragment load: the fragment word index is
// row*36 + s*8 + quad, and row*36 mod 32 = row*4 mod 32 walks 0,4,...,28 across
// the 8 fragment rows while quad walks 0..3 inside each.
//
// NO ldmatrix: the fragment registers are built from plain smem loads, the way
// `dense_gemm_tc.cu` and `w8a16_gemm_m16.cu` do it (ldmatrix.x4 is broken on
// sm_121).
//
// Rows >= M and weight rows >= N are ZERO-FILLED in smem and never stored, so a
// caller may pass any 1 <= M <= 16 and any N, and the kernel neither reads nor
// writes outside [M, N]. That matters here: the vocab is 248,077 — ODD — so the
// last CTA is always partial.
//
// Grid: (ceil(N/N_TILE), 1, 1)  Block: (128, 1, 1). Requires K % 64 == 0 (the
// pipeline step) and an activation row pitch that is a multiple of 8 BF16 (the
// cp.async chunks are 16 B). The WEIGHT row pitch is K, so K % 8 == 0 — implied
// by K % 64 == 0 — is what keeps every weight row 16-byte aligned.
//
// Entry points:
//   `dense_gemm_m16_bf16`      N_TILE=32, caller-supplied A/C row pitches in
//                              ELEMENTS (K and N for the contiguous LM head)
//   `dense_gemm_m16_bf16_n64`  N_TILE=64, same signature — the wide-tile arm

#include <cuda_bf16.h>

#define DGM16_M_TILE 16
// The two instantiated N tiles. 32 is the default; 64 is the opt-in wide arm
// (`ATLAS_LM_HEAD_M16_TC_NTILE=64`). Both must be a multiple of 32 so each of
// the 4 warps owns a whole number of 8-wide MMA tiles.
#define DGM16_N_TILE 32
#define DGM16_N_TILE_WIDE 64
#define DGM16_K_STEP 64
#define DGM16_K_SUB 16                                  // one m16n8k16's K width
#define DGM16_K_SUBS (DGM16_K_STEP / DGM16_K_SUB)       // = 4
#define DGM16_WARPS 4
#define DGM16_THREADS (DGM16_WARPS * 32)                // 128
#define DGM16_N_PER_MMA 8                               // one m16n8k16's N width
#define DGM16_STAGES 4
// BF16 elements per smem row on BOTH tiles: 64 + 8 pad = 144 B. See the bank /
// alignment note in the header.
#define DGM16_ROW_STRIDE 72
// 16-byte cp.async chunks: 8 BF16 each, 8 chunks per 64-wide row, 128 threads
// => one full 16-row A tile per pass and 16 weight rows per pass.
#define DGM16_ELEMS_PER_CHUNK 8
#define DGM16_ROWS_PER_PASS (DGM16_THREADS / (DGM16_K_STEP / DGM16_ELEMS_PER_CHUNK))

// cp.async.cg 16-byte (cache-global) copy: smem <- global. sm_80+; correct on
// sm_121, unlike TMA / cp.async.bulk. Requires 16-byte-aligned addresses.
__device__ __forceinline__ void dgm16_cp_async_cg_16(void* smem_ptr, const void* gmem_ptr) {
    unsigned int s = (unsigned int)__cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem_ptr));
}
__device__ __forceinline__ void dgm16_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void dgm16_cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// Wait until at most `n` cp.async groups remain in flight. The PTX operand must
// be a compile-time immediate, so dispatch the runtime count (always in
// [0, DGM16_STAGES-1]) through a switch over its legal values.
__device__ __forceinline__ void dgm16_cp_async_wait_le(unsigned int n) {
    switch (n) {
        case 0:  dgm16_cp_async_wait_group<0>(); break;
        case 1:  dgm16_cp_async_wait_group<1>(); break;
        case 2:  dgm16_cp_async_wait_group<2>(); break;
        default: dgm16_cp_async_wait_group<3>(); break;
    }
}

/// Dense BF16 tensor-core decode GEMM body. `a_row_stride` / `c_row_stride` are
/// the A and C row pitches in ELEMENTS; the weight pitch is K by definition of
/// the `[N, K]` HuggingFace layout.
///
/// `N_TILE` is the CTA's N width: 32 (default) or 64 (the wide arm). It must be
/// a multiple of 32 — one 8-wide MMA tile per warp per sub-step.
template <int N_TILE>
__device__ __forceinline__ void dense_gemm_m16_bf16_impl(
    const __nv_bfloat16* __restrict__ A,     // [M, a_row_stride] BF16, K used
    const __nv_bfloat16* __restrict__ B,     // [N, K] BF16 weights
    __nv_bfloat16* __restrict__ C,           // [M, c_row_stride] BF16, N used
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int a_row_stride,
    unsigned int c_row_stride
) {
    static_assert(N_TILE % 32 == 0, "N_TILE must be a whole number of MMA N-tiles per warp");
    // MMA N-tiles per warp (1 at N_TILE=32, 2 at 64) and 16-byte weight chunks
    // per thread per stage (2 resp. 4, for the same reason: both scale with the
    // weight bytes the CTA stages).
    constexpr int N_PER_WARP = N_TILE / DGM16_WARPS;
    constexpr int N_SUBS = N_PER_WARP / DGM16_N_PER_MMA;
    constexpr int B_CHUNKS = N_TILE / DGM16_ROWS_PER_PASS;

    const unsigned int cta_n = blockIdx.x * N_TILE;
    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int lane_id = threadIdx.x & 31;
    const unsigned int group_id = lane_id >> 2;     // 0..7: MMA fragment row
    const unsigned int quad = lane_id & 3;          // 0..3: MMA fragment K pair
    const unsigned int warp_n = warp_id * N_PER_WARP;

    __shared__ __align__(16) __nv_bfloat16 smem_A[DGM16_STAGES][DGM16_M_TILE][DGM16_ROW_STRIDE];
    __shared__ __align__(16) __nv_bfloat16 smem_B[DGM16_STAGES][N_TILE][DGM16_ROW_STRIDE];

    // ONE level of FP32 accumulation — there is no block scale to fold, so the
    // MMA chain runs uninterrupted from k=0 to k=K.
    float acc[4 * N_SUBS];
    #pragma unroll
    for (int i = 0; i < 4 * N_SUBS; i++) acc[i] = 0.0f;

    const unsigned int n_steps = K / DGM16_K_STEP;

    // Stage one K-step into `stage`. The A tile is 2 KB = 128 chunks of 16 B,
    // i.e. EXACTLY one cp.async per thread; the weight tile is 2 KB per 16 N
    // rows, so B_CHUNKS per thread. The copies run along K, the contiguous
    // global axis for A [M,K] and B [N,K] alike. Out-of-range rows (activation
    // rows >= M, weight rows >= N) are zero-filled by hand — cp.async cannot
    // predicate, and zero operands contribute nothing to the MMA, which is what
    // makes any 1 <= M <= 16 and any N legal.
    auto prefetch = [&](unsigned int step, unsigned int stage) {
        const unsigned int k_base = step * DGM16_K_STEP;
        const unsigned int col = (threadIdx.x & 7) * DGM16_ELEMS_PER_CHUNK;   // 0..56
        {
            const unsigned int row = threadIdx.x >> 3;                        // 0..15
            __nv_bfloat16* dst = &smem_A[stage][row][col];
            if (row < M) {
                dgm16_cp_async_cg_16(dst, &A[(unsigned long long)row * a_row_stride + k_base + col]);
            } else {
                #pragma unroll
                for (int e = 0; e < DGM16_ELEMS_PER_CHUNK; e++) dst[e] = __float2bfloat16(0.0f);
            }
        }
        #pragma unroll
        for (int c = 0; c < B_CHUNKS; c++) {
            const unsigned int row = (threadIdx.x >> 3) + c * DGM16_ROWS_PER_PASS;
            const unsigned int gn = cta_n + row;
            __nv_bfloat16* dst = &smem_B[stage][row][col];
            if (gn < N) {
                dgm16_cp_async_cg_16(dst, &B[(unsigned long long)gn * K + k_base + col]);
            } else {
                #pragma unroll
                for (int e = 0; e < DGM16_ELEMS_PER_CHUNK; e++) dst[e] = __float2bfloat16(0.0f);
            }
        }
        dgm16_cp_async_commit();
    };

    #pragma unroll
    for (unsigned int p = 0; p < DGM16_STAGES - 1; p++) {
        if (p < n_steps) prefetch(p, p);
    }

    for (unsigned int step = 0; step < n_steps; step++) {
        const unsigned int cur = step % DGM16_STAGES;
        // Groups complete FIFO. Before this iteration issues its own prefetch,
        // min(n_steps, STAGES-1+step) groups are committed and `cur` is the
        // step-th, so the number that may stay in flight is:
        const unsigned int committed = min(n_steps, DGM16_STAGES - 1 + step);
        dgm16_cp_async_wait_le(committed - (step + 1));
        // ONE barrier per K-step. It does double duty: stage `cur` is now
        // visible to every warp, AND every warp has finished the MMA of step-1,
        // which is what makes the prefetch below (it targets stage
        // (step-1) % STAGES) safe to issue without a second barrier.
        __syncthreads();
        const unsigned int ahead = step + DGM16_STAGES - 1;
        if (ahead < n_steps) prefetch(ahead, ahead % DGM16_STAGES);

        const unsigned short* sA = (const unsigned short*)&smem_A[cur][0][0];
        const unsigned short* sB = (const unsigned short*)&smem_B[cur][0][0];
        #pragma unroll
        for (int s = 0; s < DGM16_K_SUBS; s++) {
            // m16n8k16 row.col fragments, built straight from smem (no
            // ldmatrix): A rows {group_id, group_id+8} x K pairs
            // {quad*2, quad*2+8}; B row (warp_n + j*8 + group_id) at the same K
            // pairs. The A fragment is loaded ONCE and reused by all N_SUBS
            // MMAs — that reuse is the whole point of the wide tile. A staged
            // weight word IS a B fragment register: no dequant, no scale.
            const unsigned int kc0 = s * DGM16_K_SUB + quad * 2;
            const unsigned int kc1 = kc0 + 8;
            const unsigned int r0 = group_id * DGM16_ROW_STRIDE;
            const unsigned int r1 = (group_id + 8) * DGM16_ROW_STRIDE;
            const unsigned int a0 = *(const unsigned int*)&sA[r0 + kc0];
            const unsigned int a1 = *(const unsigned int*)&sA[r1 + kc0];
            const unsigned int a2 = *(const unsigned int*)&sA[r0 + kc1];
            const unsigned int a3 = *(const unsigned int*)&sA[r1 + kc1];
            #pragma unroll
            for (int j = 0; j < N_SUBS; j++) {
                const unsigned int brow =
                    (warp_n + j * DGM16_N_PER_MMA + group_id) * DGM16_ROW_STRIDE;
                const unsigned int b0 = *(const unsigned int*)&sB[brow + kc0];
                const unsigned int b1 = *(const unsigned int*)&sB[brow + kc1];
                // Index `acc` directly rather than through a pointer: `j` is a
                // compile-time constant under the unroll, so these stay
                // registers. A `float*` into the array would force it to local
                // memory and the accumulator would spill every K-step.
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5, %6, %7}, "
                    "{%8, %9}, "
                    "{%10, %11, %12, %13};"
                    : "=f"(acc[j * 4 + 0]), "=f"(acc[j * 4 + 1]),
                      "=f"(acc[j * 4 + 2]), "=f"(acc[j * 4 + 3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                      "r"(b0), "r"(b1),
                      "f"(acc[j * 4 + 0]), "f"(acc[j * 4 + 1]),
                      "f"(acc[j * 4 + 2]), "f"(acc[j * 4 + 3])
                );
            }
        }
    }

    // ── Store: FP32 accumulators -> BF16, masked to [M, N] ──
    const unsigned int row0 = group_id;
    const unsigned int row1 = group_id + 8;
    const unsigned long long o0 = (unsigned long long)row0 * c_row_stride;
    const unsigned long long o1 = (unsigned long long)row1 * c_row_stride;
    #pragma unroll
    for (int j = 0; j < N_SUBS; j++) {
        const unsigned int col0 = cta_n + warp_n + j * DGM16_N_PER_MMA + quad * 2;
        const unsigned int col1 = col0 + 1;
        if (row0 < M && col0 < N) C[o0 + col0] = __float2bfloat16(acc[j * 4 + 0]);
        if (row0 < M && col1 < N) C[o0 + col1] = __float2bfloat16(acc[j * 4 + 1]);
        if (row1 < M && col0 < N) C[o1 + col0] = __float2bfloat16(acc[j * 4 + 2]);
        if (row1 < M && col1 < N) C[o1 + col1] = __float2bfloat16(acc[j * 4 + 3]);
    }
}

/// The default 32-wide CTA. `a_row_stride` / `c_row_stride` are in ELEMENTS;
/// the BF16 LM head passes K and N (its `normed` rows are `h` apart and its
/// `logits` rows are `vocab_size` apart, both contiguous).
extern "C" __global__
__launch_bounds__(DGM16_THREADS, 4)
void dense_gemm_m16_bf16(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int a_row_stride,
    unsigned int c_row_stride
) {
    dense_gemm_m16_bf16_impl<DGM16_N_TILE>(A, B, C, M, N, K, a_row_stride, c_row_stride);
}

/// `N_TILE=64` twin of [`dense_gemm_m16_bf16`] — SAME arguments, SAME
/// per-output arithmetic (the m16n8k16 K order is untouched; only which CTA
/// owns a column changes), HALF the CTAs, and each staged A tile feeds twice
/// the weight bytes, halving the L2 traffic the A re-reads cost. Grid is
/// `ceil(N/64)`.
///
/// Opt-in (`ATLAS_LM_HEAD_M16_TC_NTILE=64`) and NOT the default: it is the A/B
/// arm for the A-reuse hypothesis in the header, and it has no receipt yet.
extern "C" __global__
__launch_bounds__(DGM16_THREADS, 4)
void dense_gemm_m16_bf16_n64(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int a_row_stride,
    unsigned int c_row_stride
) {
    dense_gemm_m16_bf16_impl<DGM16_N_TILE_WIDE>(A, B, C, M, N, K, a_row_stride, c_row_stride);
}
