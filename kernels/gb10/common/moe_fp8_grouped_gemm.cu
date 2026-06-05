// SPDX-License-Identifier: AGPL-3.0-only

// Atlas FP8 Grouped MoE GEMM — Sorted expert dispatch with FP8 E4M3 block-scaled weights.
//
// C[M_expert,N] = A[M_expert,K] (BF16) @ dequant(B_expert[N,K] (FP8 E4M3))
//
// THE routed-expert FP8 grouped-GEMM kernel for prefill (grid-compaction). A
// persistent PM5_PERSIST_CTAS=96 1D grid strides over a COMPACTED
// (expert, m_tile, n_tile) work-list built by `moe_build_tile_worklist`
// (moe_permute.cu) — collapsing the launch overhead that dominated the earlier
// dense `[ceil(N/64), max_m_tiles, num_experts]` 3D grid (99.5% early-exit CTAs)
// while keeping the inner GEMM byte-for-byte unchanged. Expert weights are
// accessed via pointer tables indexed by expert_id. Tokens are sorted by expert
// so each expert's tokens are contiguous.
//
// FP8 weight format: B[N,K] uint8 with block_scale[N/128, K/128] FP32.
// (scale_inv is widened to FP32 at load; applied in full FP32 precision.)
// Dequant: bf16_val = E4M3_LUT[byte] * block_scale[n/128, k/128]
//
// Numerics SSOT (Phase 2b, 2026-05-24): all f32 -> BF16 conversions in
// this file use `__float2bfloat16(x)`, which on sm_80+ lowers to
// `cvt.rn.bf16.f32` (round-to-nearest-even). This matches the
// load-time CPU dequant in `weight_map::fp8_lut::f32_to_bf16` and
// `atlas_quant::fp8::f32_to_bf16`, so the routed-expert kernel-side
// dequant agrees byte-exact with the shared-expert load-time dequant
// AND with PyTorch's `torch.float32 -> torch.bfloat16` reference.
//
// Grid: (PM5_PERSIST_CTAS=96, 1, 1)  Block: (PM4_THREADS=256, 1, 1)

#include <cuda_bf16.h>

#define FP8_BLOCK 128

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

// ═══════════════════════════════════════════════════════════════════
// Inner-GEMM primitives — K_STEP=32 + K-contiguous smem_B (PM4 geometry).
// ═══════════════════════════════════════════════════════════════════
//
// The PM4_* tile geometry, cp.async helpers, and `pm4_mma_kstep` below are the
// shared inner-GEMM building blocks consumed VERBATIM by the canonical
// grid-compaction kernel `moe_fp8_grouped_gemm` (further down). BF16 acts × FP8
// E4M3 block-scaled weights, grouped per-expert dispatch, two-level FP32
// accumulation, shared-memory E4M3 LUT. 128×64 tile / 256-thread /
// cp.async-pipelined, with two levers the dense w8a16_gemm_pipelined kernel
// proved (12→26 TFLOP/s, commit dd7d7bd):
//
//   LEVER A — K_STEP=32 (PM4_K_SUB=16, 2 sub-MMAs per step). A single
//     m16n8k16 MMA per resident K-step pays the full barrier triple
//     (raw-B sync → dequant → smem_B sync → MMA → reuse sync) once per 16 K.
//     Keeping TWO 16-K sub-MMAs resident per step amortizes the barrier triple
//     over 2× the MMA-issue work — halving the __syncthreads count per K
//     traversed. Since the loop is MMA-issue/barrier-bound once the dequant is
//     cheap, this is the primary lever.
//
//   LEVER B — K-contiguous smem_B [n][k]. The MMA B fragment packs two
//     consecutive-K BF16 weights (k, k+1) per 32-bit register. A [k][n] store
//     would make that pair two STRIDED 16-bit loads + a shift/or; with [n][k]
//     the pair is ADJACENT in smem → a SINGLE aligned 32-bit load:
//     *(u32*)&sB[n*b_stride + k]. Halves the smem instruction count on the B
//     fragment and removes the bit-shuffle ALU.
//
// Both levers preserve numerics EXACTLY: smem_B holds the identical unscaled
// BF16-cast E4M3 values, only the storage axis changes; the two sub-MMAs sum
// into the same inner_acc a single MMA did, in the same K order; the scale is
// still applied ONCE per 128-K block on the FP32 outer accumulator.
//
// smem budget per stage (N_TILE=64, K_STEP=32): smem_A 128×40×2 = 10240 B +
// smem_B 64×34×2 = 4352 B + smem_Braw 64×32 = 2048 B ≈ 16640 B → 33280 B for
// 2 stages + 1 KB LUT = 34304 B (well under the 101 KB cap).
//
//   LAUNCH_BOUNDS (256,2). The K_STEP-32 sub-MMA loop needs more live registers
//   (A fragments for 2 sub-K windows + 8 N-tile accumulators). At a (256,3)
//   hint ptxas caps to 80 regs and SPILLS (192 B store / 96 B load), measured
//   25.9 TFLOP/s. The 34 KB smem already caps the kernel to 2 CTAs/SM
//   (3×34 KB > 100 KB carveout) regardless of the reg hint, so (256,2) costs
//   ZERO occupancy yet lets ptxas use 125 regs with NO spill → 31.3 TFLOP/s
//   (+23% vs the spilling (256,3) build). Probed K_STEP=64 (4 sub-MMAs):
//   63 KB smem → 1 CTA/SM, regressed to 22 TFLOP/s — confirming the kernel
//   rewards barrier-amortization + tile reuse at 2 CTAs over raw occupancy,
//   but only until smem collapses the CTA count.
//
// cp.async.cg / mma.sync.bf16 only; NO TMA / cp.async.bulk / e2m1 (corrupt on
// sm_121). M_TILE=128.

#define PM4_M_TILE 128
#define PM4_N_TILE 64
#define PM4_K_STEP 32
#define PM4_K_SUB 16                             // one MMA's K-width
#define PM4_K_SUBS (PM4_K_STEP / PM4_K_SUB)      // = 2 sub-MMAs per K-step
#define PM4_PAD 2
#define PM4_A_STRIDE (PM4_K_STEP + 8)            // K_STEP K-cols + 8 pad, mult. of 8 BF16 (16 B)
#define PM4_WARPS 8
#define PM4_THREADS (PM4_WARPS * 32)             // 256
#define PM4_N_TILES_PER_WARP (PM4_N_TILE / 8)    // m16n8k16 N-tiles per warp (=8 at N_TILE=64)
#define PM4_STAGES 2

__device__ __forceinline__ void pm4_cp_async_cg_16(void* smem_ptr, const void* gmem_ptr) {
    unsigned int s = (unsigned int)__cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem_ptr));
}
__device__ __forceinline__ void pm4_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void pm4_cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void pm4_cp_async_wait_le(unsigned int n) {
    switch (n) {
        case 0:  pm4_cp_async_wait_group<0>(); break;
        case 1:  pm4_cp_async_wait_group<1>(); break;
        case 2:  pm4_cp_async_wait_group<2>(); break;
        default: pm4_cp_async_wait_group<3>(); break;
    }
}

// MMA over one resident K_STEP (PM4_K_STEP=32 K-elements = PM4_K_SUBS=2 m16n8k16
// sub-MMAs of 16-K each) into inner[PM4_N_TILES_PER_WARP][4]. smem_B is [n][k]
// K-CONTIGUOUS (Lever B): the (k, k+1) BF16 pair of each MMA B fragment is a
// single aligned 32-bit load. The two sub-MMAs (Lever A) sum into the same
// inner accumulator in K order — batched 2-at-a-time per barrier.
__device__ __forceinline__ void pm4_mma_kstep(
    const __nv_bfloat16* smem_A,   // [PM4_M_TILE][PM4_A_STRIDE]
    const __nv_bfloat16* smem_B,   // [PM4_N_TILE][PM4_K_STEP + PM4_PAD] (K-contiguous)
    float inner[PM4_N_TILES_PER_WARP][4],
    unsigned int warp_m_offset, unsigned int group_id, unsigned int tid
) {
    const unsigned int a_stride = PM4_A_STRIDE;
    const unsigned int b_stride = PM4_K_STEP + PM4_PAD;   // [n][k] K-contiguous stride
    const unsigned short* sA = (const unsigned short*)smem_A;
    const unsigned short* sB = (const unsigned short*)smem_B;

    unsigned int frag_r0 = warp_m_offset + group_id;
    unsigned int frag_r1 = warp_m_offset + group_id + 8;

    #pragma unroll
    for (int s = 0; s < PM4_K_SUBS; s++) {
        const unsigned int k_off = s * PM4_K_SUB;   // K offset of this sub-MMA within the step
        unsigned int frag_c0 = k_off + tid * 2;
        unsigned int frag_c1 = k_off + tid * 2 + 8;

        unsigned int a0 = *(const unsigned int*)&sA[frag_r0 * a_stride + frag_c0];
        unsigned int a1 = *(const unsigned int*)&sA[frag_r1 * a_stride + frag_c0];
        unsigned int a2 = *(const unsigned int*)&sA[frag_r0 * a_stride + frag_c1];
        unsigned int a3 = *(const unsigned int*)&sA[frag_r1 * a_stride + frag_c1];

        #pragma unroll
        for (int n_tile = 0; n_tile < PM4_N_TILES_PER_WARP; n_tile++) {
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

// ═══════════════════════════════════════════════════════════════════
// moe_fp8_grouped_gemm — THE routed-expert FP8 prefill kernel (grid-compaction).
// ═══════════════════════════════════════════════════════════════════
//
// BF16 acts × FP8 E4M3 block-scaled weights, grouped per-expert dispatch,
// two-level FP32 accumulation, shared-memory E4M3 LUT, K_STEP=32 + K-contiguous
// smem_B. The PER-TILE compute reuses the PM4 inner-GEMM primitives above
// (pm4_mma_kstep + the exact warp/lane setup, accumulator init, prefetch+dequant
// lambdas, pipelined K-loop, two-level fold, and store guard).
//
// Launch geometry: instead of deriving (expert, m_tile, n_tile) from
// blockIdx.{z,y,x} on a dense 3D grid `[ceil(N/64), max_m_tiles=1300,
// num_experts]` (≈16M CTAs/layer of which 99.5% early-exit because most
// (m_tile, expert) pairs are out of range), this kernel launches a PERSISTENT
// 1D grid of PM5_PERSIST_CTAS=96 CTAs that stride over a COMPACTED work-list
// built by `moe_build_tile_worklist` (moe_permute.cu). Each work-item is exactly
// one real (non-early-exit) tile:
//   worklist[wid*2 + 0] = expert_id
//   worklist[wid*2 + 1] = (m_tile << 6) | n_tile
//   *total_tiles        = number of real tiles
// This collapses the launch overhead (the prior dense grid was LAUNCH-bound at
// 1.7% of peak on the MoE FFN prefill) while keeping the inner GEMM unchanged.
//
// SAME-STREAM INVARIANT (R3): the builder + this kernel MUST be enqueued on the
// SAME stream so the read of *total_tiles / worklist[] here happens-after the
// builder's write. The Rust launcher enforces this (no cross-stream event).
//
// RISK #1 (smem reuse across work-items): a single CTA processes MANY tiles in
// its persistent loop, all reusing the SAME smem_A/smem_B/smem_Braw staging
// buffers. A `__syncthreads()` at the TOP of each iteration fences the prior
// tile's last pipeline stage (still being read by stragglers in the final MMA
// reuse-sync) before this iteration re-primes the cp.async pipeline — otherwise
// a stale stage could be overwritten while still in use. The smem LUT (lut_s)
// is resident across the whole loop and filled ONCE before it.
//
// Grid: (PM5_PERSIST_CTAS=96, 1, 1)  Block: (PM4_THREADS=256, 1, 1)
#define PM5_PERSIST_CTAS 96   // 48 SMs × 2 CTAs/SM. SSOT — mirrored in the Rust
                              // launch wrapper ops::moe_fp8_grouped_gemm.

extern "C" __global__ void __launch_bounds__(PM4_THREADS, 2) moe_fp8_grouped_gemm(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ B_weight_ptrs,
    const unsigned long long* __restrict__ B_scale_ptrs,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    unsigned int num_experts,
    unsigned int N,
    unsigned int K,
    const unsigned int* __restrict__ worklist,   // [*total_tiles * 2] (expert, packed m/n)
    const int* __restrict__ total_tiles          // [1] (read-after-write on same stream)
) {
    // E4M3 LUT staged into shared memory (data-dependent divergent lookups
    // serialize in __constant__ memory; the smem copy holds byte-identical
    // values). RISK #1: filled ONCE before the persistent loop, resident across
    // all work-items (NOT re-filled per iteration).
    __shared__ float lut_s[256];
    #pragma unroll
    for (unsigned int i = threadIdx.x; i < 256; i += PM4_THREADS) {
        lut_s[i] = E4M3_LUT_GMOE[i];
    }

    // Pipelined smem — PM4 layout/budget (see the inner-GEMM primitives above).
    // Shared across all work-items this CTA processes (see RISK #1 fence at the
    // loop top).
    __shared__ __align__(16) __nv_bfloat16 smem_A[PM4_STAGES][PM4_M_TILE][PM4_A_STRIDE];
    __shared__ __nv_bfloat16 smem_B[PM4_STAGES][PM4_N_TILE][PM4_K_STEP + PM4_PAD];
    __shared__ __align__(16) unsigned char smem_Braw[PM4_STAGES][PM4_N_TILE][PM4_K_STEP];

    // Warp/lane setup — CTA-uniform, hoisted out of the persistent loop.
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;           // 8 warps × 16 = 128 M-rows
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    const int total = *total_tiles;

    // Persistent loop: stride PM5_PERSIST_CTAS over the compacted work-list.
    for (int wid = blockIdx.x; wid < total; wid += PM5_PERSIST_CTAS) {
        __syncthreads();   // RISK #1 fix: fence smem reuse before re-priming pipeline

        // ── Coords from the work-list instead of blockIdx.{z,y,x} ──
        // Named mt/nt (not m_tile/n_tile) so they do NOT shadow the store
        // loop's `n_tile` MMA index below.
        unsigned int expert_id = worklist[wid * 2 + 0];
        unsigned int packed    = worklist[wid * 2 + 1];
        unsigned int mt = packed >> 6;
        unsigned int nt = packed & 0x3F;

        const int m_start = expert_offsets[expert_id];
        const int M_expert = expert_offsets[expert_id + 1] - m_start;

        const unsigned char* B_exp = (const unsigned char*)B_weight_ptrs[expert_id];
        const float* S_exp = (const float*)B_scale_ptrs[expert_id];
        if (B_exp == 0) continue;   // NULL → remote expert under EP (builder skips these too)

        const unsigned int cta_m_local = mt * PM4_M_TILE;   // expert-relative M base
        const unsigned int cta_n = nt * PM4_N_TILE;

        // ── Per-tile inner GEMM (accumulator-init onward) ──
        // Two-level FP32 accumulation — PRESERVED EXACTLY (inner over a 128-K block,
        // outer += inner * block_scale at the boundary; scale never per-element).
        // RE-ZEROED at the top of each work-item (per-tile accumulators).
        float inner_acc[PM4_N_TILES_PER_WARP][4];
        float outer_acc[PM4_N_TILES_PER_WARP][4];
        #pragma unroll
        for (int i = 0; i < PM4_N_TILES_PER_WARP; i++) {
            inner_acc[i][0] = 0.0f; inner_acc[i][1] = 0.0f;
            inner_acc[i][2] = 0.0f; inner_acc[i][3] = 0.0f;
            outer_acc[i][0] = 0.0f; outer_acc[i][1] = 0.0f;
            outer_acc[i][2] = 0.0f; outer_acc[i][3] = 0.0f;
        }

        const unsigned int k_blocks = (K + FP8_BLOCK - 1) / FP8_BLOCK;
        const unsigned int k_steps_per_block = FP8_BLOCK / PM4_K_STEP;   // 4 at K_STEP=32
        const unsigned int n_block = cta_n / FP8_BLOCK;
        const unsigned int n_steps = (K + PM4_K_STEP - 1) / PM4_K_STEP;

        // A-tile cp.async: 128 rows × PM4_K_STEP K-cols BF16. Each 16-B chunk = 8
        // BF16. At K_STEP=32: 128×32/8 = 512 chunks → 2 chunks/thread (256 threads).
        // Each A row is GATHERED through sorted_token_ids; the 16-B K-run within a
        // row is still contiguous (one coalesced transaction).
        const unsigned int a_chunks = (PM4_M_TILE * PM4_K_STEP) / 8;     // 512

        auto prefetch = [&](unsigned int step, unsigned int stage) {
            unsigned int k_base = step * PM4_K_STEP;

            // ── A: contiguous 16-B (8 BF16) chunks along K, gathered per row ──
            #pragma unroll
            for (unsigned int c = threadIdx.x; c < a_chunks; c += PM4_THREADS) {
                unsigned int row = (c * 8) / PM4_K_STEP;          // 0..127
                unsigned int col = (c * 8) % PM4_K_STEP;          // 0, 8, 16, 24
                unsigned int m_global = cta_m_local + row;        // expert-relative
                unsigned int gc = k_base + col;
                __nv_bfloat16* dst = &smem_A[stage][row][col];
                if (m_global < (unsigned int)M_expert && gc + 8 <= K) {
                    int sorted_idx = m_start + (int)m_global;
                    int token_id = sorted_token_ids ? sorted_token_ids[sorted_idx] : sorted_idx;
                    pm4_cp_async_cg_16(dst, &A[(unsigned long long)token_id * K + gc]);
                } else {
                    #pragma unroll
                    for (unsigned int e = 0; e < 8; e++) {
                        unsigned int gcol = gc + e;
                        if (m_global < (unsigned int)M_expert && gcol < K) {
                            int sorted_idx = m_start + (int)m_global;
                            int token_id = sorted_token_ids ? sorted_token_ids[sorted_idx] : sorted_idx;
                            dst[e] = A[(unsigned long long)token_id * K + gcol];
                        } else {
                            dst[e] = __float2bfloat16(0.0f);
                        }
                    }
                }
            }

            // ── B raw: contiguous 16-B (16 FP8-byte) chunks of K per N-row ──
            // smem_Braw[stage][n][k] mirrors global B[n, k_base + k] contiguously.
            // At K_STEP=32 each N-row is 32 bytes = two 16-B chunks.
            const unsigned int b_chunks = (PM4_N_TILE * PM4_K_STEP) / 16;   // 128 at K_STEP=32
            #pragma unroll
            for (unsigned int c = threadIdx.x; c < b_chunks; c += PM4_THREADS) {
                unsigned int nrow = (c * 16) / PM4_K_STEP;        // 0..PM4_N_TILE-1
                unsigned int kcol = (c * 16) % PM4_K_STEP;        // 0 or 16
                unsigned int gn = cta_n + nrow;
                unsigned int gk = k_base + kcol;
                unsigned char* dst = &smem_Braw[stage][nrow][kcol];
                if (gn < N && gk + 16 <= K) {
                    pm4_cp_async_cg_16(dst, &B_exp[(unsigned long long)gn * K + gk]);
                } else {
                    #pragma unroll
                    for (unsigned int e = 0; e < 16; e++) {
                        unsigned int gke = gk + e;
                        dst[e] = (gn < N && gke < K) ? B_exp[(unsigned long long)gn * K + gke] : 0;
                    }
                }
            }
            pm4_cp_async_commit();
        };

        // LUT-dequant just-arrived raw B for `stage` into the MMA-ready BF16 buffer.
        // smem_B is [n][k] K-contiguous, matching smem_Braw, so this is a same-layout
        // element-wise dequant (no transpose). NO scale (folded post-MMA at the
        // block boundary). Cooperative across all 256 threads (each weight converted
        // once, reused by all 8 warps).
        auto dequant_B = [&](unsigned int stage) {
            #pragma unroll
            for (unsigned int idx = threadIdx.x; idx < PM4_K_STEP * PM4_N_TILE; idx += PM4_THREADS) {
                unsigned int n = idx / PM4_K_STEP;     // 0..PM4_N_TILE-1
                unsigned int k = idx % PM4_K_STEP;     // 0..PM4_K_STEP-1
                unsigned char wb = smem_Braw[stage][n][k];
                smem_B[stage][n][k] = __float2bfloat16(lut_s[wb]);
            }
        };

        // ── Software-pipelined main loop (PM4_STAGES-deep cp.async) ──
        #pragma unroll
        for (unsigned int p = 0; p < PM4_STAGES - 1; p++) {
            if (p < n_steps) {
                prefetch(p, p % PM4_STAGES);
            }
        }
        unsigned int k_step_in_block = 0;

        for (unsigned int step = 0; step < n_steps; step++) {
            unsigned int cur = step % PM4_STAGES;

            unsigned int ahead = step + (PM4_STAGES - 1);
            if (ahead < n_steps) {
                prefetch(ahead, ahead % PM4_STAGES);
            }
            unsigned int committed = min(n_steps, PM4_STAGES + step);
            unsigned int target = committed - (step + 1);
            pm4_cp_async_wait_le(target);
            __syncthreads();   // raw B for `cur` resident for all threads

            dequant_B(cur);
            __syncthreads();   // smem_B[cur] fully written before MMA reads it

            pm4_mma_kstep(&smem_A[cur][0][0], &smem_B[cur][0][0],
                          inner_acc, warp_m_offset, group_id, tid);
            __syncthreads();   // done reading smem_*[cur]; safe for reuse

            // K_BLOCK boundary: fold scaled inner into outer, reset inner.
            k_step_in_block++;
            if (k_step_in_block == k_steps_per_block) {
                const unsigned int k_block = (step * PM4_K_STEP) / FP8_BLOCK;
                const float scale = S_exp[n_block * k_blocks + k_block];
                #pragma unroll
                for (int i = 0; i < PM4_N_TILES_PER_WARP; i++) {
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
            const unsigned int k_block = (K - 1) / FP8_BLOCK;
            const float scale = S_exp[n_block * k_blocks + k_block];
            #pragma unroll
            for (int i = 0; i < PM4_N_TILES_PER_WARP; i++) {
                outer_acc[i][0] += inner_acc[i][0] * scale;
                outer_acc[i][1] += inner_acc[i][1] * scale;
                outer_acc[i][2] += inner_acc[i][2] * scale;
                outer_acc[i][3] += inner_acc[i][3] * scale;
            }
        }

        // ── Store C tile: f32 outer accumulators → BF16, sorted output position ──
        #pragma unroll
        for (int n_tile = 0; n_tile < PM4_N_TILES_PER_WARP; n_tile++) {
            unsigned int base_n = cta_n + n_tile * 8;
            unsigned int col0 = base_n + (tid * 2);
            unsigned int col1 = col0 + 1;
            unsigned int row0 = cta_m_local + warp_m_offset + group_id;   // expert-relative
            unsigned int row1 = row0 + 8;

            if (row0 < (unsigned int)M_expert) {
                unsigned int out_row = m_start + row0;
                if (col0 < N) C[(unsigned long long)out_row * N + col0] = __float2bfloat16(outer_acc[n_tile][0]);
                if (col1 < N) C[(unsigned long long)out_row * N + col1] = __float2bfloat16(outer_acc[n_tile][1]);
            }
            if (row1 < (unsigned int)M_expert) {
                unsigned int out_row = m_start + row1;
                if (col0 < N) C[(unsigned long long)out_row * N + col0] = __float2bfloat16(outer_acc[n_tile][2]);
                if (col1 < N) C[(unsigned long long)out_row * N + col1] = __float2bfloat16(outer_acc[n_tile][3]);
            }
        }
    }
}
