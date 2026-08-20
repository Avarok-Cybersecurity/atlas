// SPDX-License-Identifier: AGPL-3.0-only

// Atlas W4A16 GEMV — Fused NVFP4 weight dequant + BF16 GEMV for M=1 decode.
//
// out[n] = dot(A[0,:], dequant(B_fp4[n,:]))
//
// Specialized for M=1 decode: replaces w4a16_gemm which wastes ~98% of
// threads at M=1 with 64x64 tiles + MMA tensor cores (MMA requires M>=16).
//
// Vectorized: reads 4 packed weight bytes (uint32_t = 8 FP4 values) and
// 8 BF16 activations (uint4 = 16 bytes) per iteration for better bandwidth.
//
// NVFP4 weight format (HuggingFace/compressed-tensors):
//   B_packed: [N, K/2] uint8 — byte at [n, j] holds W[n, 2j] (low) and W[n, 2j+1] (high)
//   B_scale:  [N, K/GROUP_SIZE] FP8-E4M3 — one scale per group of 16 K-dim values
//   scale2:   scalar FP32 — per-tensor second-level scale
//
// K-dim packing: each byte holds 2 consecutive input features for the same output.
// Vectorized reads of 4 bytes = 8 weight values, coalesced across warps.
//
// 4 outputs per block, 64 threads (2 warps) per output. Cross-warp smem reduction.
// Grid: (ceil(N / 4), 1, 1)   Block: (256, 1, 1)

#include <cuda_bf16.h>
#include <cuda_fp8.h>

// Standard E4M3 (1-4-3, bias 7) decode via pure bit-math. On real NVIDIA this is
// byte-identical to (float)__nv_fp8_e4m3; on SCALE/gfx1151 the built-in
// __nv_fp8_e4m3->float decode is a NON-STANDARD narrow format which mismatches the
// standard E4M3 scales written by the encoder -> corrupts every block scale.
// HIP/gfx1151 shares the same software path (no cvt.rn.satfinite.e4m3x2.f32 PTX).
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ float scl_fp8(unsigned char b) {
    unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u; float v;
    if (e == 0u)               v = (float)m * 0.001953125f;            // subnormal m*2^-9
    else if (e == 15u && m == 7u) v = 0.0f;                            // NaN -> 0
    else                       v = __uint_as_float(((e + 120u) << 23) | (m << 20)); // 2^(e-7)*(1+m/8)
    return s ? -v : v;
}
#endif

#define BLOCK_SIZE 256
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define GROUP_SIZE 16

// E2M1 lookup table (same as w4a16_gemm.cu)
__device__ __constant__ float E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// ── Stage the E2M1 table in SHARED memory before any decode inner loop ──
// The dequant indexes the table with a DATA-DEPENDENT weight nibble. A read of
// __constant__ memory SERIALIZES across a warp whenever the lanes hit divergent
// entries: the constant cache is a BROADCAST cache and replays once per distinct
// address, and 32 lanes drawing NVFP4 nibbles cover 16*(1-(15/16)^32) ~= 14 of
// the 16 entries, so one LUT read costs ~14 transactions. Shared memory services
// divergent indices in parallel — the 16 entries sit in 16 distinct banks and
// same-index lanes multicast, so the read is one conflict-free transaction.
// Same lever, same reasoning as `w8a16_gemm_pipelined.cu` (measured 5x there).
//
// This is NOT a numerical change: `s_lut[i]` is a bit-exact FP32 copy of
// `E2M1_LUT[i]`, so every fmaf sees the identical operand it saw before.
//
// ── HIP / SCALE portability ──────────────────────────────────────────────────
// This file is SYMLINKED into kernels/strix-hip/common/ and kernels/strix/common/,
// so hipcc compiles it verbatim. `__syncwarp()` is a CUDA intrinsic that target
// does not accept: the strix-hip fork of w4a16_gemm.cu contains zero
// `__syncwarp` and the tree carries no shim. Introducing it unguarded broke the
// windows-x86_64-amd-hip release build (#519).
//
// NOTE the block-wide staging elsewhere in this file (`__shared__ s_lut` +
// `__syncthreads()`, 9 pre-existing sites) is FINE on HIP and is untouched.
// Only the WARP-scoped publish is CUDA-only.
//
// On AMD the warp-scoped path keeps the PRE-EXISTING behaviour: no staging, the
// partial indexes `__constant__ E2M1_LUT` directly, exactly as shipped before
// this change. Provably compilable (it is the shipped code) and provably
// correct; it forgoes the staging win on a target we cannot measure. Doing
// better needs a wave-scoped publish (`__builtin_amdgcn_wave_barrier()` plus
// the right fence) VALIDATED on gfx1151 — no such box exists here, so it is not
// guessed at, for the same reason the strix-hip batched-prefill argument shift
// was left to someone with the hardware.
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
#define ATLAS_WARP_LUT_STAGED 0
#else
#define ATLAS_WARP_LUT_STAGED 1
#endif

// WARP-SCOPED variant, for the single-warp kernels: they early-return on a
// warp-uniform `n >= N`, so a block-wide __syncthreads() here would be a
// divergent barrier. Lanes 0..15 fill their own warp's copy; __syncwarp()
// publishes it. The reduction below stays barrier-free, as documented.
__device__ __forceinline__ void stage_e2m1_lut_warp(float* s_lut, unsigned int lane) {
#if ATLAS_WARP_LUT_STAGED
    if (lane < 16u) s_lut[lane] = E2M1_LUT[lane];
    __syncwarp();
#else
    (void)s_lut; (void)lane;   // AMD: nothing staged; callers pass the constant LUT
#endif
}

// W4A16 GEMV: C[n] = sum_k A[k] * dequant(B_fp4[n, k])
//
// Vectorized: 16 K-values per chunk (2× uint4 activation + uint64 weight),
// TWO chunks in flight per iteration with independent accumulators. Keeping 2
// outstanding weight loads per thread hides DRAM latency (ncu: 72% of warp
// stalls are long-scoreboard on GB10). The FP8 group scale is factored out of
// the inner 16-FMA block (exact regroup: sum(s*w*a) == s*sum(w*a)).
//
// Coalescing: within a warp, consecutive threads read consecutive weight and
// activation chunks. Perfectly coalesced.
//
// SSOT: `w4a16_gemv` and `w4a16_gemv_sw` share `w4a16_gemv_partial`. A previous
// SW copy used the older stride-64 sequential `acc += a*w` loop and was 1 ULP
// lossy vs this pipelined association (GB10 oracle: gdn in_proj / K-tail).

// One orig-lane's pipelined K16 partial. `orig_lane` is the 64-thread kernel's
// lane (0..63): k16 = orig_lane*2, stride 128, inner c={0,1} → kk pair.
__device__ __forceinline__ float w4a16_gemv_partial(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    unsigned int n, unsigned int half_K, unsigned int num_groups,
    unsigned int K16, unsigned int orig_lane,
    const float* __restrict__ lut)   // shared-staged E2M1_LUT copy; see stage_e2m1_lut_warp
{
    float acc0 = 0.0f, acc1 = 0.0f;
    const unsigned int stride2 = 128u; // threads_per_out (64) * 2
    for (unsigned int k16 = orig_lane * 2u; k16 < K16 + 1u; k16 += stride2) {
        #pragma unroll
        for (int c = 0; c < 2; c++) {
            const unsigned int kk = k16 + (unsigned int)c;
            if (kk >= K16) break;

            uint4 a_lo = ((const uint4*)A)[kk * 2];
            uint4 a_hi = ((const uint4*)A)[kk * 2 + 1];
            const unsigned int a_raw[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                                            a_hi.x, a_hi.y, a_hi.z, a_hi.w};
            unsigned long long packed8 = *(const unsigned long long*)(
                B_packed + (unsigned long long)n * half_K + kk * 8);
            unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + kk];
            __nv_fp8_e4m3 fp8;
            *(unsigned char*)&fp8 = scale_byte;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
            float scale = scl_fp8(scale_byte) * scale2;
#else
            float scale = (float)fp8 * scale2;
#endif
            float part = 0.0f;
            #pragma unroll
            for (int b = 0; b < 8; b++) {
                unsigned char byte_val = (unsigned char)(packed8 >> (b * 8));
                float2 af = __bfloat1622float2(*(const __nv_bfloat162*)&a_raw[b]);
                part = fmaf(af.x, lut[byte_val & 0xF], part);
                part = fmaf(af.y, lut[byte_val >> 4], part);
            }
            if (c == 0) acc0 = fmaf(scale, part, acc0);
            else        acc1 = fmaf(scale, part, acc1);
        }
    }
    return acc0 + acc1;
}

extern "C" __global__ void w4a16_gemv(
    const __nv_bfloat16* __restrict__ A,        // [1, K]
    const unsigned char* __restrict__ B_packed,  // [N, K/2] uint8
    const unsigned char* __restrict__ B_scale,   // [N, K/GROUP_SIZE] FP8-E4M3
    const float scale2,
    __nv_bfloat16* __restrict__ C,               // [1, N]
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;  // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 2];  // cross-warp reduction
    // Block-staged here (all 256 threads reach this barrier — no early return).
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    // Do not return early: N%4!=0 tail warps must still hit the smem barrier.
    float acc = 0.0f;
    if (n < N) {
        acc = w4a16_gemv_partial(A, B_packed, B_scale, scale2, n, half_K, num_groups, K16, lane, s_lut);
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (warp_lane == 0) {
        unsigned int smem_idx = local_out * 2 + (lane / WARP_SIZE);
        smem[smem_idx] = acc;
    }
    __syncthreads();

    if (lane == 0 && n < N) {
        float result = smem[local_out * 2] + smem[local_out * 2 + 1];
        C[n] = __float2bfloat16(result);
    }
}

// ============================================================
// W4A16 GEMV — SINGLE-WARP-PER-OUTPUT variant (lossless; default ON, kill ATLAS_NO_GEMV_SW=1).
//
// Bit-identical to w4a16_gemv: same `w4a16_gemv_partial` per orig-lane. 32
// threads (1 warp) per output instead of 64 (2 warps). 8 outputs per 256-thread
// block (was 4). The cross-warp __syncthreads() + smem round-trip is gone —
// the final combine is one FP32 add of two warp-shuffle results.
//
// BIT-IDENTICALITY: orig warp A is lanes 0..31, warp B is 32..63. Each SW lane
// holds those two orig-lane partials:
//   acc_a[lane] == orig acc[lane]     (k16 = lane*2,     stride 128)
//   acc_b[lane] == orig acc[lane+32]  (k16 = (lane+32)*2, stride 128)
// Shuffle-reduce each in the same tree, then `reduced_a + reduced_b` ==
// smem[0]+smem[1]. Grid: (ceil(N / 8), 1, 1)   Block: (256, 1, 1)

#define N_PER_BLOCK_SW 8

extern "C" __global__ void w4a16_gemv_sw(
    const __nv_bfloat16* __restrict__ A,        // [1, K]
    const unsigned char* __restrict__ B_packed,  // [N, K/2] uint8
    const unsigned char* __restrict__ B_scale,   // [N, K/GROUP_SIZE] FP8-E4M3
    const float scale2,
    __nv_bfloat16* __restrict__ C,               // [1, N]
    unsigned int N,
    unsigned int K
) {
    const unsigned int local_out = threadIdx.x / WARP_SIZE;       // 0..7
    const unsigned int lane = threadIdx.x % WARP_SIZE;            // 0..31
    const unsigned int n = blockIdx.x * N_PER_BLOCK_SW + local_out;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    // One private copy per warp (8 warps x 16 floats = 512 B/block): the block
    // barrier is unavailable here (warp-uniform early return above).
    __shared__ float s_lut[N_PER_BLOCK_SW][16];
    stage_e2m1_lut_warp(s_lut[local_out], lane);
#if ATLAS_WARP_LUT_STAGED
    const float* __restrict__ warp_lut = s_lut[local_out];
#else
    const float* __restrict__ warp_lut = E2M1_LUT;
#endif

    // acc_a reproduces orig lane `lane` (warp A); acc_b reproduces orig lane
    // `lane+32` (warp B). Same operands, same order as the 64-thread kernel.
    float acc_a = w4a16_gemv_partial(A, B_packed, B_scale, scale2, n, half_K, num_groups, K16, lane, warp_lut);
    float acc_b = w4a16_gemv_partial(A, B_packed, B_scale, scale2, n, half_K, num_groups, K16, lane + 32u, warp_lut);

    // Reduce each accumulator within the warp in the SAME tree order as orig.
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc_a += __shfl_down_sync(0xFFFFFFFF, acc_a, offset);
        acc_b += __shfl_down_sync(0xFFFFFFFF, acc_b, offset);
    }

    // lane 0 holds reduced warp-A (acc_a) and warp-B (acc_b). Final combine ==
    // smem[0] + smem[1] in the 64-thread kernel. Bit-identical.
    if (lane == 0) {
        float result = acc_a + acc_b;
        C[n] = __float2bfloat16(result);
    }
}

// ============================================================
// W4A16 GEMV with FP32 output (for LM head logits).
// Identical to w4a16_gemv but writes float instead of BF16.
// FP32 logits are critical for sampling quality — BF16 collapses
// similar logit values, making stochastic sampling random.
// ============================================================
extern "C" __global__ void w4a16_gemv_logits(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    float* __restrict__ C,  // FP32 output
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 2];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc = 0.0f;
    for (unsigned int k16 = lane; k16 < K16; k16 += threads_per_out) {
        const unsigned int base_k = k16 * 16;
        uint4 a_lo = ((const uint4*)A)[k16 * 2];
        uint4 a_hi = ((const uint4*)A)[k16 * 2 + 1];
        const unsigned int a_raw[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                                        a_hi.x, a_hi.y, a_hi.z, a_hi.w};
        unsigned long long packed8 = *(const unsigned long long*)(B_packed + (unsigned long long)n * half_K + k16 * 8);
        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + scale_group];
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = scale_byte;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
        float scale = scl_fp8(scale_byte) * scale2;
#else
        float scale = (float)fp8 * scale2;
#endif
        #pragma unroll
        for (int b = 0; b < 8; b++) {
            unsigned char byte_val = (unsigned char)(packed8 >> (b * 8));
            float w_lo = s_lut[byte_val & 0xF] * scale;
            float w_hi = s_lut[byte_val >> 4] * scale;
            __nv_bfloat16 a_lo_bf, a_hi_bf;
            *(unsigned short*)&a_lo_bf = (unsigned short)(a_raw[b] & 0xFFFF);
            *(unsigned short*)&a_hi_bf = (unsigned short)(a_raw[b] >> 16);
            acc += __bfloat162float(a_lo_bf) * w_lo;
            acc += __bfloat162float(a_hi_bf) * w_hi;
        }
    }
    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    if (warp_lane == 0) {
        unsigned int smem_idx = local_out * 2 + (lane / WARP_SIZE);
        smem[smem_idx] = acc;
    }
    __syncthreads();
    if (lane == 0) {
        C[n] = smem[local_out * 2] + smem[local_out * 2 + 1]; // FP32 output!
    }
}

// ============================================================
// W4A16 double-GEMV (M=2): reads weights once, computes 2 outputs
// ============================================================
// For K=2 speculative verification: processes 2 input vectors through
// the same weight matrix in a single pass. Eliminates the GEMM M=2
// tile waste (64x64 tiles at 3% M-utilization).
//
// A: [2, K] BF16 contiguous (row 0 and row 1)
// B: [N, K/2] NVFP4 packed weights
// C: [2, N] BF16 contiguous (row 0 and row 1)
//
// Same memory bandwidth as M=1 GEMV (weights dominate, read once).
// Extra cost: 2x activation reads (K*2 bytes per vector, fits in L1/L2).
//
// Grid: (ceil(N / 4), 1, 1)   Block: (256, 1, 1)
// `w4a16_gemv_batch2` is defined below, next to batch4/8/16/32, because it is
// now a thin instantiation of the shared `w4a16_gemv_batchm_impl` template.
// Its former standalone body carried the same three bit-parity divergences
// from `w4a16_gemv` documented on that template — see the note there.

// ============================================================
// W4A16 batched GEMV (M<=MAX_M) — the NVFP4 sibling of w8a16_gemv_batch4/16.
// ============================================================
// At M-token batched decode the SSM QKVZ / out_proj projections share the same
// NVFP4 weight matrix across all M sequences, so a SINGLE DRAM pass over the
// packed 4-bit weight (dequantized E2M1*scale ONCE) serves all M rows — MAC'd
// into M independent FP32 accumulators. This is what lets FP4 amortize the
// weight read the way w8a16_gemv_batch4/16 does for FP8; without it the FP4
// multi-seq path capped at batch3 and re-streamed the weight ~3x at C=8.
//
// Per-row accumulation order is IDENTICAL to `w4a16_gemv` (M=1), so the output
// is bit-identical to running w4a16_gemv M times.
//
// That claim was FALSE until 2026-08-12 and is now enforced by
// `crates/spark-model/examples/w4a16_batch_bitparity_microtest.rs`. The body
// below had THREE independent divergences from `w4a16_gemv`, each of which
// alone breaks bit-parity under the dir's `--fmad=false`:
//
//   1. K-chunk → lane mapping. `w4a16_gemv` walks `k16 = lane*2` with stride
//      128 in PAIRS (c = 0,1) into two accumulators; this walked `k16 = lane`
//      with stride 64 into one. Different lane partitions ⇒ different
//      partial sums ⇒ a different shuffle-reduction tree.
//   2. Block scale placement. `w4a16_gemv` factors the FP8 group scale OUT of
//      the 16-FMA block (`acc = fmaf(scale, part, acc)`); this pre-multiplied
//      it into every unpacked weight (`lut * scale`). Exact in real
//      arithmetic, NOT in FP32.
//   3. Fused vs split accumulation — the same defect found in
//      `w8a16_gemv_batch4.cu`: `acc += x*w0 + y*w1` associates as
//      `acc + (x·w0 + y·w1)`, where the M=1 kernel computes
//      `(acc + x·w0) + y·w1`.
//
// Measured at the production Nemotron projection shapes before the fix
// ([10304x2688], [2688x4096], [18560x4096], [4096x8192], seeds 1/99/12345):
// 178 of 180 legs differed, 1..62 BF16 elements per launch, max|delta| 0.0625.
// The body now reproduces `w4a16_gemv` FMA-for-FMA; the weight DRAM read, the
// packed8 unpack and the scale decode are still done ONCE per chunk for all M
// rows, which is the entire point of the tier.
//
// ── VIRTUAL-LANE REMAP (why this does not walk K the way `w4a16_gemv` does) ──
//
// The literal transcription — physical thread `l` walks `k16 = l*2` with stride
// 128 in pairs, exactly like the M=1 kernel — is bit-exact but 45% SLOWER than
// the (wrong) body it replaced, and NOT because of the second accumulator
// array. It is coalescing. Under that mapping a warp's 16-byte activation load
// has its lanes 64 B apart, so one instruction touches 2048 B to use 512 — half
// the sector efficiency of the stride-32 walk, and the activation stream is
// M times the weight stream, which is why the cost grew with M (batch2 +13%,
// batch16 +49%). Priced directly: holding the pair mapping and both
// accumulators but addressing activations with the stride-32 pattern lands on
// baseline (-0.4%); widening the WEIGHT load to 16 B recovers nothing (+44%).
//
// So this walks K COALESCED — consecutive lanes read consecutive chunks, the
// pattern the previous body used — and recovers bit-parity by remapping which
// REFERENCE accumulator each physical thread stands in for. Write p = physical
// thread (0..63) and recall that the M=1 kernel's lane l owns chunks
// {2l + c + 128j} in accumulator c. Then, for either value of `phase`,
//
//   chunks {p + 64*phase + 128j} == 2*(p/2 + 32*phase) + (p%2) + 128j
//                                == ref lane (p/2 + 32*phase), accumulator p%2
//
// — the same chunk set in the same order, so the same fmaf chain, so a partial
// that is bit-identical to the reference one it stands for. The kernel runs
// that chain TWICE, phase 0 then phase 1, with ONE accumulator array reused;
// the two phases read DISJOINT halves of the weight row, so total DRAM traffic
// is unchanged (each pass is a stride-1024 B walk of contiguous 512 B runs) and
// the register cost stays at the pre-fix `acc[MAX_M]`. Carrying both chains at
// once instead costs a second array and ~20% at batch8/16, measured.
//
// `acc_l = acc0_l + acc1_l` is then one `__shfl_xor_sync(...,1)` between the
// adjacent threads p and p^1, which hold accumulator 0 and 1 of the SAME
// reference lane and are always in the same warp. p and p^1 add in opposite
// operand order, which is exact — FP32 addition is commutative.
//
// That leaves acc_l for l = 0..63 on the even threads, so `s_vl` re-lands it in
// virtual-lane order and the UNCHANGED two-warp shuffle tree below reduces
// lanes 0..31 and 32..63 exactly as `w4a16_gemv` does. Cost: one shfl per row
// per phase and a 64-float smem round trip per (row, output) — not per chunk.
//
// Register cost lands at or BELOW the pre-fix body — batch16 56 vs 70 and
// batch32 80 vs 106, so 4 and 3 CTA/SM where the pre-fix body got 3 and 2 —
// and no `__launch_bounds__` is wanted on the WIDE tiers (16/32): pinning them
// all measured 12.70 ms against 11.49 ms free. The narrow M=5..8 tiers are the
// exception and are pinned to 5 CTA/SM; the register table and the reasoning
// are on `w4a16_gemv_batch5`.
//
// A:[M,K] BF16, B_packed:[N,K/2], B_scale:[N,K/16] FP8-E4M3, scale2 FP32,
// C:[M,N] BF16. Grid: (ceil(N/4),1,1) Block: (256,1,1).
template <int MAX_M>
__device__ __forceinline__ void w4a16_gemv_batchm_impl(
    const __nv_bfloat16* __restrict__ A,         // [M, K]
    const unsigned char* __restrict__ B_packed,   // [N, K/2] uint8
    const unsigned char* __restrict__ B_scale,    // [N, K/GROUP_SIZE] FP8-E4M3
    const float scale2,
    __nv_bfloat16* __restrict__ C,                // [M, N]
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;  // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    // ONE accumulator array, reused by both phases — see the two-phase note in
    // the header. `phase` picks WHICH reference lane this thread is standing in
    // for; within a phase the thread runs a single reference accumulator chain.
    float acc[MAX_M];
    __shared__ float s_vl[MAX_M][N_PER_BLOCK][2 * WARP_SIZE];

    #pragma unroll 1
    for (unsigned int phase = 0; phase < 2u; phase++) {
        #pragma unroll
        for (int t = 0; t < MAX_M; t++) acc[t] = 0.0f;

        // Coalesced walk: at every step the 64 lanes read CONSECUTIVE chunks,
        // so a warp's 8-byte weight load covers 256 contiguous bytes and its
        // 16-byte activation loads are 32 B apart — the pre-fix access pattern.
        for (unsigned int kk = lane + phase * threads_per_out; kk < K16;
             kk += threads_per_out * 2u) {
            // 8 packed weight bytes (16 FP4) + 1 group scale → read and
            // unpacked ONCE for all M rows. This is the weight-DRAM saving
            // the tier exists for; it is independent of the FP order below.
            unsigned long long packed8 =
                *(const unsigned long long*)(B_packed + (unsigned long long)n * half_K + kk * 8);
            unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + kk];
            __nv_fp8_e4m3 fp8;
            *(unsigned char*)&fp8 = scale_byte;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
            float scale = scl_fp8(scale_byte) * scale2;
#else
            float scale = (float)fp8 * scale2;
#endif
            // UNSCALED E2M1 values. The group scale stays factored OUT of the
            // 16-FMA block and is applied once via fmaf(scale, part, acc),
            // exactly as `w4a16_gemv` does — pre-multiplying it into each
            // weight is a different FP32 expression.
            float wl[16];
            #pragma unroll
            for (int b = 0; b < 8; b++) {
                unsigned char byte_val = (unsigned char)(packed8 >> (b * 8));
                wl[b * 2]     = s_lut[byte_val & 0xF];   // W[2j]   <-> act 2j
                wl[b * 2 + 1] = s_lut[byte_val >> 4];    // W[2j+1] <-> act 2j+1
            }

            #pragma unroll
            for (int t = 0; t < MAX_M; t++) {
                if ((unsigned int)t >= M) continue;
                const __nv_bfloat16* At = A + (unsigned long long)t * K;
                uint4 a_lo = ((const uint4*)At)[kk * 2];
                uint4 a_hi = ((const uint4*)At)[kk * 2 + 1];
                const unsigned int ar[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                                            a_hi.x, a_hi.y, a_hi.z, a_hi.w};
                // Per-chunk partial in the M=1 kernel's exact FMA order:
                // TWO separate fmaf per byte, never `part += x*w0 + y*w1`.
                float part = 0.0f;
                #pragma unroll
                for (int b = 0; b < 8; b++) {
                    // BF16 -> FP32 is EXACT and is nothing but `bits << 16`
                    // (BF16 is the high half of the FP32 word), so this is a
                    // bit-for-bit substitution for
                    // `__bfloat1622float2(*(const __nv_bfloat162*)&ar[b])`,
                    // NOT a numeric change: the intrinsic lowers to
                    // `mov.b32 %f, {0, %h}`, which forces ptxas to first
                    // EXTRACT each 16-bit half into its own register (one
                    // PRMT apiece) and then re-assemble it. Doing the two
                    // shifts on the packed word the load already produced
                    // skips the extraction entirely: measured on sm_121f
                    // (`cuobjdump -sass`, --fmad=false) it removes all 64
                    // PRMT from batch8 (824 -> 760 static instructions,
                    // -7.8%) and all 32 from batch4 (472 -> 440, -6.8%) at
                    // identical registers (48), smem and occupancy. The
                    // template is issue-bound at M>=5, not DRAM-bound (see
                    // the tier table on `w4a16_gemv_batch5`), so instructions
                    // removed from this loop are the whole game.
                    const float ax = __uint_as_float(ar[b] << 16);
                    const float ay = __uint_as_float(ar[b] & 0xFFFF0000u);
                    part = fmaf(ax, wl[b * 2], part);
                    part = fmaf(ay, wl[b * 2 + 1], part);
                }
                acc[t] = fmaf(scale, part, acc[t]);
            }
        }

        // Threads p and p^1 hold accumulator 0 and 1 of the SAME reference lane
        // (p/2 + 32*phase); fold them and land the result in virtual-lane order.
        // p adds acc0+acc1 and p^1 adds acc1+acc0 — the same FP32 value.
        #pragma unroll
        for (int t = 0; t < MAX_M; t++) {
            if ((unsigned int)t >= M) continue;
            const float v = acc[t] + __shfl_xor_sync(0xFFFFFFFF, acc[t], 1);
            if ((lane & 1u) == 0u) {
                s_vl[t][local_out][phase * WARP_SIZE + (lane >> 1)] = v;
            }
        }
    }
    __syncthreads();

    __shared__ float smem[MAX_M][N_PER_BLOCK * 2];  // 2 warps/output, per row
    const unsigned int warp_in_out = lane / WARP_SIZE;
    #pragma unroll
    for (int t = 0; t < MAX_M; t++) {
        if ((unsigned int)t >= M) continue;
        float a = s_vl[t][local_out][lane];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            a += __shfl_down_sync(0xFFFFFFFF, a, offset);
        }
        if (lane % WARP_SIZE == 0) smem[t][local_out * 2 + warp_in_out] = a;
    }
    __syncthreads();

    if (lane == 0) {
        #pragma unroll
        for (int t = 0; t < MAX_M; t++) {
            if ((unsigned int)t >= M) continue;
            float r = smem[t][local_out * 2] + smem[t][local_out * 2 + 1];
            C[(unsigned long long)t * N + n] = __float2bfloat16(r);
        }
    }
}

// M==2 (K=2 spec verify, and C=2 multi-seq decode: dense_ffn, qwen3 qkv/o_proj).
// Fixed-M signature (no `M` argument) — kept for its existing call sites.
//
// This used to be a hand-written standalone kernel that walked K in 8-value
// chunks with ONE accumulator per row and the group scale pre-multiplied into
// each weight. It therefore carried all three divergences documented on
// `w4a16_gemv_batchm_impl` above and was never byte-identical to 2 x
// `w4a16_gemv`. Routing it through the (now bit-exact) template fixes it by
// construction and deletes ~100 lines of duplicated dequant/reduce code. The
// template also reads 16 K-values per chunk with two chunks in flight, where
// the old body read 8 with one, so the weight loads get WIDER, not narrower.
extern "C" __global__ void w4a16_gemv_batch2(
    const __nv_bfloat16* __restrict__ A,          // [2, K]
    const unsigned char* __restrict__ B_packed,   // [N, K/2] uint8
    const unsigned char* __restrict__ B_scale,    // [N, K/GROUP_SIZE] FP8-E4M3
    const float scale2,
    __nv_bfloat16* __restrict__ C,                // [2, N]
    unsigned int N,
    unsigned int K
) {
    w4a16_gemv_batchm_impl<2>(A, B_packed, B_scale, scale2, C, 2u, N, K);
}

// M==3 (K=3 spec verify, C=3 multi-seq decode, MoE forward_k3). Same story as
// batch2 above: formerly a standalone body with the same three divergences,
// plus the fused `acc += x*w_lo + y*w_hi` form verbatim.
extern "C" __global__ void w4a16_gemv_batch3(
    const __nv_bfloat16* __restrict__ A,          // [3, K]
    const unsigned char* __restrict__ B_packed,   // [N, K/2] uint8
    const unsigned char* __restrict__ B_scale,    // [N, K/GROUP_SIZE] FP8-E4M3
    const float scale2,
    __nv_bfloat16* __restrict__ C,                // [3, N]
    unsigned int N,
    unsigned int K
) {
    w4a16_gemv_batchm_impl<3>(A, B_packed, B_scale, scale2, C, 3u, N, K);
}

// M<=4 (common-path batched decode) — sibling of w8a16_gemv_batch4.
extern "C" __global__ void w4a16_gemv_batch4(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    w4a16_gemv_batchm_impl<4>(A, B_packed, B_scale, scale2, C, M, N, K);
}

// ── EXACT-M TIERS 5/6/7 ─────────────────────────────────────────────────────
// Why they exist. `MAX_M` sizes `acc[]`, `s_vl[]` and `smem[]`, and — because
// the row loop is `#pragma unroll`ed — it also sizes the CODE. The `t >= M`
// guard skips the WORK of a dead row at run time but not its instructions:
// running M=5 on the MAX_M=8 tier still carries three dead unrolled row
// blocks through the hot loop. Measured on the real 27B qkv/o shape
// (N=5120 K=5120, cold-cycled weights, 273 GB/s peak) BEFORE this change:
//
//   tier    M    time      eff BW     % peak
//   batch4  4    70.5 us   209.2 GB/s  76.6%
//   batch8  4    74.8 us   197.3 GB/s  72.3%   <- same M, same work, +6.1%
//   batch8  5    89.0 us   165.7 GB/s  60.7%
//   batch8  6    91.5 us   161.2 GB/s  59.0%
//   batch8  8   106.4 us   138.5 GB/s  50.7%
//   batch16 8   113.3 us   130.1 GB/s  47.7%
//
// The batch8-at-M=4 row is the whole argument: identical rows, identical
// weight stream, +6.1%. It is NOT the `__launch_bounds__` — batch4 lands on
// 48 registers / 5 CTA with no pragma at all, exactly what the pragma pins
// batch8 to, so the two tiers run at the SAME occupancy. What differs is the
// unrolled body: 824 vs 472 static SASS instructions before the conversion
// fix above, 760 vs 440 after. Sizing the tier to the real row count is the
// only way to stop paying for rows that are not there.
//
// Occupancy. `__launch_bounds__(BLOCK_SIZE, 5)` on 5/6/7 for the same reason
// it is on 8, and it is free here — measured with `ptxas -v` on sm_121f:
//
//   MAX_M   free regs / CTA-per-SM      pinned to 5 CTA
//   5       48 / 5   (already 5)        48, no spill
//   6       55 / 4                      48, no spill
//   7       56 / 4                      48, no spill
//   8       56 / 4                      48, no spill  <- the existing pin
//
// i.e. 6 and 7 would otherwise DROP to 4 CTA/SM and lose the thing the tier
// was added to win. 48 is the ceiling at 5 CTA (5 x 256 x 48 = 61440 of the
// SM's 65536 registers; the next allocation step, 56, only fits 4 CTAs), and
// asking for 6 CTA spills 32 B at MAX_M=8 and 8 B at MAX_M=5.
//
// Dispatch picks the narrowest RESOLVED tier >= M
// (`spark-model/src/layers/w4a16_gemv_tiers.rs`); `ATLAS_NO_GEMV_EXACT_M_TIERS=1`
// hides 5/6/7 and restores the batch4/batch8-only decision.
//
// Bit-parity: same template, same `MAX_M`-independent FMA chain per row, so
// tier 5/6/7 output is bit-identical to tier 8 (and to M x `w4a16_gemv`) at
// the same M — enforced by `w4a16_batch_bitparity_microtest`.
extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 5) void w4a16_gemv_batch5(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    w4a16_gemv_batchm_impl<5>(A, B_packed, B_scale, scale2, C, M, N, K);
}

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 5) void w4a16_gemv_batch6(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    w4a16_gemv_batchm_impl<6>(A, B_packed, B_scale, scale2, C, M, N, K);
}

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 5) void w4a16_gemv_batch7(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    w4a16_gemv_batchm_impl<7>(A, B_packed, B_scale, scale2, C, M, N, K);
}

// M<=8 (chain-verify K=5..8) — keeps M=5..8 projections on the
// weight-streaming batched GEMV instead of falling to the tile GEMMs.
// batch16's acc[16] + smem[16][8] register/smem pressure is halved at
// MAX_M=8; per-row accumulation order is identical to batch4/batch16
// (same template, M-guarded), so output is bit-identical at matching M.
//
// The ONE tier that wants a `__launch_bounds__`: free allocation gives it 56
// registers / 4 CTA per SM, and pinning it to 48 / 5 measured 2.166 ms against
// 2.261 ms over the gate's M=5..8 legs (6 CTA is worse again, 2.437 ms). It is
// also the only tier still slower than the pre-fix body (1.969 ms, +10%);
// batch2/3/4/16/32 are all at or under it.
//
// DROPPING the pragma was re-examined 2026-08-17 and REJECTED: it does not
// reproduce batch4's codegen (batch4 is 48/5 for free), it produces the
// 56-register / 4-CTA variant the line above already measured as the slower
// of the two. The tiers above carry the same pin for the same reason.
extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 5) void w4a16_gemv_batch8(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    w4a16_gemv_batchm_impl<8>(A, B_packed, B_scale, scale2, C, M, N, K);
}

// M<=16 (high-concurrency decode, n=5..16) — sibling of w8a16_gemv_batch16.
extern "C" __global__ void w4a16_gemv_batch16(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    w4a16_gemv_batchm_impl<16>(A, B_packed, B_scale, scale2, C, M, N, K);
}

// M<=32 (native-bs32 batched propose, n=17..32) — same M-guarded template.
// Exists so the n=32 MTP re-propose reads the drafter LM head ONCE instead
// of two chunked batch16 sweeps (the measured 2 x 13.5 ms propose in the
// wave-10 K=2 step timing). acc[32] + smem[32][8] doubles batch16's
// per-thread state; the kernel is DRAM-bound on the shared weight read at
// grid ceil(N/4), so occupancy loss is secondary to halving weight traffic.
extern "C" __global__ void w4a16_gemv_batch32(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    w4a16_gemv_batchm_impl<32>(A, B_packed, B_scale, scale2, C, M, N, K);
}

// ============================================================
// W4A16 GEMV with inline Q/Gate deinterleave on output write
// ============================================================
// Same GEMV as w4a16_gemv but writes Q and Gate to separate halves.
// Eliminates the separate deinterleave_qg kernel (saves 12 graph nodes).
//
// Input layout (interleaved per head): [Q_h0(hd), G_h0(hd), Q_h1(hd), G_h1(hd), ...]
// Output layout (deinterleaved): [Q_h0..Q_nh | G_h0..G_nh]
//
// N = num_heads * head_dim * 2  (total Q+Gate elements)
//
// Grid: (ceil(N / 4), 1, 1)   Block: (256, 1, 1)
extern "C" __global__ void w4a16_gemv_qg(
    const __nv_bfloat16* __restrict__ A,        // [1, K]
    const unsigned char* __restrict__ B_packed,  // [N, K/2] uint8
    const unsigned char* __restrict__ B_scale,   // [N, K/GROUP_SIZE] FP8-E4M3
    const float scale2,
    __nv_bfloat16* __restrict__ C,               // [Q | G] deinterleaved
    unsigned int N,
    unsigned int K,
    unsigned int num_heads,
    unsigned int head_dim
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 2];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;
        uint4 a_data = ((const uint4*)A)[k8];
        const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
        unsigned int packed4 = *(const unsigned int*)(B_packed + (unsigned long long)n * half_K + k8 * 4);
        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + scale_group];
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = scale_byte;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
        float scale = scl_fp8(scale_byte) * scale2;
#else
        float scale = (float)fp8 * scale2;
#endif

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char byte_val = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = s_lut[byte_val & 0xF] * scale;
            float w_hi = s_lut[byte_val >> 4] * scale;
            __nv_bfloat16 a_lo, a_hi;
            *(unsigned short*)&a_lo = (unsigned short)(a_raw[b] & 0xFFFF);
            *(unsigned short*)&a_hi = (unsigned short)(a_raw[b] >> 16);
            acc += __bfloat162float(a_lo) * w_lo;
            acc += __bfloat162float(a_hi) * w_hi;
        }
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (warp_lane == 0) {
        smem[local_out * 2 + (lane / WARP_SIZE)] = acc;
    }
    __syncthreads();

    if (lane == 0) {
        float result = smem[local_out * 2] + smem[local_out * 2 + 1];

        // Deinterleave: n indexes interleaved [Q_h0(hd), G_h0(hd), Q_h1(hd), ...]
        // head = n / (2 * head_dim), is_gate = (n % (2 * head_dim)) >= head_dim
        unsigned int group_dim = 2 * head_dim;
        unsigned int h = n / group_dim;
        unsigned int idx = n % group_dim;
        unsigned int q_total = num_heads * head_dim;

        unsigned int out_idx;
        if (idx < head_dim) {
            out_idx = h * head_dim + idx;             // Q region
        } else {
            out_idx = q_total + h * head_dim + (idx - head_dim);  // Gate region
        }
        C[out_idx] = __float2bfloat16(result);
    }
}

// ============================================================
// W4A16 GEMV with inline QKVZ deinterleave on output write
// ============================================================
// Same GEMV as w4a16_gemv but writes to deinterleaved output locations.
// Eliminates the separate deinterleave_qkvz kernel (saves 36 graph nodes).
//
// QKVZ interleaved layout (N=12288, 16 groups of 768):
//   Group g: [Q_{g*128..128} | K_{g*128..128} | V_{g*256..256} | Z_{g*256..256}]
//
// Deinterleaved output: [Q_2048 | K_2048 | V_4096 | Z_4096]
//
// Grid: (ceil(N / 4), 1, 1)   Block: (256, 1, 1)
extern "C" __global__ void w4a16_gemv_qkvz(
    const __nv_bfloat16* __restrict__ A,        // [1, K]
    const unsigned char* __restrict__ B_packed,  // [N, K/2] uint8
    const unsigned char* __restrict__ B_scale,   // [N, K/GROUP_SIZE] FP8-E4M3
    const float scale2,
    __nv_bfloat16* __restrict__ C,               // [Q|K|V|Z] deinterleaved
    unsigned int N,
    unsigned int K,
    // Deinterleave params:
    unsigned int num_groups,        // 16
    unsigned int head_k_dim,        // 128
    unsigned int vheads_per_group,  // 2
    unsigned int head_v_dim         // 128
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups_k = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 2];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;
        uint4 a_data = ((const uint4*)A)[k8];
        const unsigned int a_raw[4] = {a_data.x, a_data.y, a_data.z, a_data.w};
        unsigned int packed4 = *(const unsigned int*)(B_packed + (unsigned long long)n * half_K + k8 * 4);
        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups_k + scale_group];
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = scale_byte;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
        float scale = scl_fp8(scale_byte) * scale2;
#else
        float scale = (float)fp8 * scale2;
#endif

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char byte_val = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = s_lut[byte_val & 0xF] * scale;
            float w_hi = s_lut[byte_val >> 4] * scale;
            __nv_bfloat16 a_lo, a_hi;
            *(unsigned short*)&a_lo = (unsigned short)(a_raw[b] & 0xFFFF);
            *(unsigned short*)&a_hi = (unsigned short)(a_raw[b] >> 16);
            acc += __bfloat162float(a_lo) * w_lo;
            acc += __bfloat162float(a_hi) * w_hi;
        }
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }

    if (warp_lane == 0) {
        smem[local_out * 2 + (lane / WARP_SIZE)] = acc;
    }
    __syncthreads();

    if (lane == 0) {
        float result = smem[local_out * 2] + smem[local_out * 2 + 1];

        // Compute deinterleaved output index
        unsigned int v_group_size = vheads_per_group * head_v_dim;
        unsigned int group_dim = 2 * head_k_dim + 2 * v_group_size;
        unsigned int g = n / group_dim;
        unsigned int idx = n % group_dim;
        unsigned int q_total = num_groups * head_k_dim;
        unsigned int k_total = num_groups * head_k_dim;

        unsigned int out_idx;
        if (idx < head_k_dim) {
            out_idx = g * head_k_dim + idx;
        } else if (idx < 2 * head_k_dim) {
            out_idx = q_total + g * head_k_dim + (idx - head_k_dim);
        } else if (idx < 2 * head_k_dim + v_group_size) {
            out_idx = q_total + k_total + g * v_group_size + (idx - 2 * head_k_dim);
        } else {
            out_idx = q_total + k_total + num_groups * v_group_size
                    + g * v_group_size + (idx - 2 * head_k_dim - v_group_size);
        }
        C[out_idx] = __float2bfloat16(result);
    }
}

// ============================================================
// W4A16 GEMV batch2 with inline Q/Gate deinterleave
// ============================================================
// Combines w4a16_gemv_batch2 (2-input) with w4a16_gemv_qg (deinterleave).
// Reads Q+Gate weight matrix once for 2 input tokens, produces 2 deinterleaved
// output vectors [Q_all | Gate_all] per token.
//
// Input:  A[2, K] BF16 (2 token hidden states)
// Output: C[2, N] BF16 (deinterleaved: C[0] = [Q0|G0], C[1] = [Q1|G1])
//
// FIFTH FAMILY, now fixed. The reference here is `w4a16_gemv_qg` (M=1), NOT
// `w4a16_gemv` — qg/qkvz have their own k8=lane chunking and pre-multiplied
// scale. Against that reference all four fused batched variants
// (`w4a16_gemv_qg_batch2`, `w4a16_gemv_dual_batch2`, `w4a16_gemv_qg_batch3`,
// `w4a16_gemv_dual_batch3`) diverged in exactly ONE way, defect 3 from the
// `w4a16_gemv_batchm_impl` note above: they computed
// `acc += x*w_lo + y*w_hi` where `w4a16_gemv_qg`/`w4a16_gemv_qkvz` do two
// separate `acc +=`. Chunking and scale placement already MATCHED, so the fix
// is a one-line split per accumulator — no lane remap, no register cost, and
// `ptxas -v` is unchanged at 40 registers / 6 CTAs per SM for all four.
// `w4a16_qg_batch_bitparity_microtest.rs` is the standing gate; the `dual`
// pair is compared against `w4a16_gemv_qg` driven with num_heads=1,
// head_dim=N, which makes its deinterleave the identity map.
//
// Grid: (ceil(N / 4), 1, 1)   Block: (256, 1, 1)
extern "C" __global__ void w4a16_gemv_qg_batch2(
    const __nv_bfloat16* __restrict__ A,        // [2, K]
    const unsigned char* __restrict__ B_packed,  // [N, K/2] uint8
    const unsigned char* __restrict__ B_scale,   // [N, K/GROUP_SIZE] FP8-E4M3
    const float scale2,
    __nv_bfloat16* __restrict__ C,               // [2, N] deinterleaved [Q|G] per token
    unsigned int N,
    unsigned int K,
    unsigned int num_heads,
    unsigned int head_dim
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    const __nv_bfloat16* __restrict__ A1 = A + K;
    __nv_bfloat16* __restrict__ C1 = C + N;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 4];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;

        uint4 a0_data = ((const uint4*)A)[k8];
        uint4 a1_data = ((const uint4*)A1)[k8];
        const unsigned int a0_raw[4] = {a0_data.x, a0_data.y, a0_data.z, a0_data.w};
        const unsigned int a1_raw[4] = {a1_data.x, a1_data.y, a1_data.z, a1_data.w};

        unsigned int packed4 = *(const unsigned int*)(B_packed + (unsigned long long)n * half_K + k8 * 4);
        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + scale_group];
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = scale_byte;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
        float scale = scl_fp8(scale_byte) * scale2;
#else
        float scale = (float)fp8 * scale2;
#endif

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char byte_val = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = s_lut[byte_val & 0xF] * scale;
            float w_hi = s_lut[byte_val >> 4] * scale;

            __nv_bfloat16 a0_lo, a0_hi;
            *(unsigned short*)&a0_lo = (unsigned short)(a0_raw[b] & 0xFFFF);
            *(unsigned short*)&a0_hi = (unsigned short)(a0_raw[b] >> 16);
            acc0 += __bfloat162float(a0_lo) * w_lo;
            acc0 += __bfloat162float(a0_hi) * w_hi;

            __nv_bfloat16 a1_lo, a1_hi;
            *(unsigned short*)&a1_lo = (unsigned short)(a1_raw[b] & 0xFFFF);
            *(unsigned short*)&a1_hi = (unsigned short)(a1_raw[b] >> 16);
            acc1 += __bfloat162float(a1_lo) * w_lo;
            acc1 += __bfloat162float(a1_hi) * w_hi;
        }
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, offset);
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
    }

    if (warp_lane == 0) {
        unsigned int warp_idx = lane / WARP_SIZE;
        smem[local_out * 4 + warp_idx * 2]     = acc0;
        smem[local_out * 4 + warp_idx * 2 + 1] = acc1;
    }
    __syncthreads();

    if (lane == 0) {
        float result0 = smem[local_out * 4]     + smem[local_out * 4 + 2];
        float result1 = smem[local_out * 4 + 1] + smem[local_out * 4 + 3];

        // Deinterleave: n indexes interleaved [Q_h0(hd), G_h0(hd), Q_h1(hd), ...]
        unsigned int group_dim = 2 * head_dim;
        unsigned int h = n / group_dim;
        unsigned int idx = n % group_dim;
        unsigned int q_total = num_heads * head_dim;

        unsigned int out_idx;
        if (idx < head_dim) {
            out_idx = h * head_dim + idx;
        } else {
            out_idx = q_total + h * head_dim + (idx - head_dim);
        }
        C[out_idx]  = __float2bfloat16(result0);
        C1[out_idx] = __float2bfloat16(result1);
    }
}

// ============================================================
// W4A16 GEMV dual batch2: K+V for 2 input tokens in one launch
// ============================================================
// Processes 2 separate weight matrices (K and V) with 2 input vectors each.
// blockIdx.z selects K (0) or V (1). Both projections compute 2 outputs.
//
// Input:  A[2, K_in] BF16 (2 token hidden states)
// Output: C[2, N] where blockIdx.z=0 writes K, blockIdx.z=1 writes V
//
// Grid: (ceil(N / 4), 1, 2)   Block: (256, 1, 1)
extern "C" __global__ void w4a16_gemv_dual_batch2(
    const __nv_bfloat16* __restrict__ A,         // [2, K_in] BF16
    const unsigned char* __restrict__ B0_packed,  // [N, K_in/2] first projection
    const unsigned char* __restrict__ B0_scale,
    float B0_scale2,
    __nv_bfloat16* __restrict__ C0,              // [2, N] first projection output
    const unsigned char* __restrict__ B1_packed,  // [N, K_in/2] second projection
    const unsigned char* __restrict__ B1_scale,
    float B1_scale2,
    __nv_bfloat16* __restrict__ C1,              // [2, N] second projection output
    unsigned int N,
    unsigned int K_in
) {
    const unsigned int proj = blockIdx.z;
    const unsigned char* B_packed = (proj == 0) ? B0_packed : B1_packed;
    const unsigned char* B_scale = (proj == 0) ? B0_scale : B1_scale;
    float s2 = (proj == 0) ? B0_scale2 : B1_scale2;
    __nv_bfloat16* C_out = (proj == 0) ? C0 : C1;

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int half_K = K_in / 2;
    const unsigned int num_groups = K_in / GROUP_SIZE;
    const unsigned int K8 = K_in / 8;

    const __nv_bfloat16* A1 = A + K_in;
    __nv_bfloat16* C_out1 = C_out + N;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 4];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc0 = 0.0f, acc1 = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;

        uint4 a0_data = ((const uint4*)A)[k8];
        uint4 a1_data = ((const uint4*)A1)[k8];
        const unsigned int a0_raw[4] = {a0_data.x, a0_data.y, a0_data.z, a0_data.w};
        const unsigned int a1_raw[4] = {a1_data.x, a1_data.y, a1_data.z, a1_data.w};

        unsigned int packed4 = *(const unsigned int*)(B_packed + (unsigned long long)n * half_K + k8 * 4);
        unsigned int sg = base_k / GROUP_SIZE;
        unsigned char sb = B_scale[(unsigned long long)n * num_groups + sg];
        __nv_fp8_e4m3 fp8; *(unsigned char*)&fp8 = sb;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
        float scale = scl_fp8(sb) * s2;
#else
        float scale = (float)fp8 * s2;
#endif

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char bv = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = s_lut[bv & 0xF] * scale;
            float w_hi = s_lut[bv >> 4] * scale;

            __nv_bfloat16 a0_lo, a0_hi;
            *(unsigned short*)&a0_lo = (unsigned short)(a0_raw[b] & 0xFFFF);
            *(unsigned short*)&a0_hi = (unsigned short)(a0_raw[b] >> 16);
            acc0 += __bfloat162float(a0_lo) * w_lo;
            acc0 += __bfloat162float(a0_hi) * w_hi;

            __nv_bfloat16 a1_lo, a1_hi;
            *(unsigned short*)&a1_lo = (unsigned short)(a1_raw[b] & 0xFFFF);
            *(unsigned short*)&a1_hi = (unsigned short)(a1_raw[b] >> 16);
            acc1 += __bfloat162float(a1_lo) * w_lo;
            acc1 += __bfloat162float(a1_hi) * w_hi;
        }
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, offset);
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
    }

    if (warp_lane == 0) {
        unsigned int warp_idx = lane / WARP_SIZE;
        smem[local_out * 4 + warp_idx * 2]     = acc0;
        smem[local_out * 4 + warp_idx * 2 + 1] = acc1;
    }
    __syncthreads();

    if (lane == 0) {
        float result0 = smem[local_out * 4]     + smem[local_out * 4 + 2];
        float result1 = smem[local_out * 4 + 1] + smem[local_out * 4 + 3];
        C_out[n]  = __float2bfloat16(result0);
        C_out1[n] = __float2bfloat16(result1);
    }
}

// ============================================================
// W4A16 triple-GEMV (M=3): reads weights once, computes 3 outputs
// ============================================================
// For K=3 speculative verification: processes 3 input vectors through
// the same weight matrix in a single pass.
//
// A: [3, K] BF16 contiguous (row 0, 1, 2)
// B: [N, K/2] NVFP4 packed weights
// C: [3, N] BF16 contiguous (row 0, 1, 2)
//
// Same memory bandwidth as M=1 GEMV (weights dominate, read once).
// Extra cost: 3x activation reads (K*2 bytes per vector, fits in L1/L2).
//
// Grid: (ceil(N / 4), 1, 1)   Block: (256, 1, 1)
// `w4a16_gemv_batch3` is defined above, next to batch4/8/16/32, because it is
// now a thin instantiation of the shared `w4a16_gemv_batchm_impl` template.
// Its former standalone body carried the same three bit-parity divergences
// from `w4a16_gemv` documented on that template — see the note there.

// ============================================================
// W4A16 GEMV batch3 with inline Q/Gate deinterleave
// ============================================================
// Combines w4a16_gemv_batch3 (3-input) with Q/Gate deinterleave.
// Reads Q+Gate weight matrix once for 3 input tokens, produces 3 deinterleaved
// output vectors [Q_all | Gate_all] per token.
//
// Input:  A[3, K] BF16 (3 token hidden states)
// Output: C[3, N] BF16 (deinterleaved: C[i] = [Qi|Gi])
//
// Grid: (ceil(N / 4), 1, 1)   Block: (256, 1, 1)
extern "C" __global__ void w4a16_gemv_qg_batch3(
    const __nv_bfloat16* __restrict__ A,        // [3, K]
    const unsigned char* __restrict__ B_packed,  // [N, K/2] uint8
    const unsigned char* __restrict__ B_scale,   // [N, K/GROUP_SIZE] FP8-E4M3
    const float scale2,
    __nv_bfloat16* __restrict__ C,               // [3, N] deinterleaved [Q|G] per token
    unsigned int N,
    unsigned int K,
    unsigned int num_heads,
    unsigned int head_dim
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K8 = K / 8;

    const __nv_bfloat16* __restrict__ A1 = A + K;
    const __nv_bfloat16* __restrict__ A2 = A + 2 * K;
    __nv_bfloat16* __restrict__ C1 = C + N;
    __nv_bfloat16* __restrict__ C2 = C + 2 * N;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 6];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;

        uint4 a0_data = ((const uint4*)A)[k8];
        uint4 a1_data = ((const uint4*)A1)[k8];
        uint4 a2_data = ((const uint4*)A2)[k8];
        const unsigned int a0_raw[4] = {a0_data.x, a0_data.y, a0_data.z, a0_data.w};
        const unsigned int a1_raw[4] = {a1_data.x, a1_data.y, a1_data.z, a1_data.w};
        const unsigned int a2_raw[4] = {a2_data.x, a2_data.y, a2_data.z, a2_data.w};

        unsigned int packed4 = *(const unsigned int*)(B_packed + (unsigned long long)n * half_K + k8 * 4);
        unsigned int scale_group = base_k / GROUP_SIZE;
        unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + scale_group];
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = scale_byte;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
        float scale = scl_fp8(scale_byte) * scale2;
#else
        float scale = (float)fp8 * scale2;
#endif

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char byte_val = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = s_lut[byte_val & 0xF] * scale;
            float w_hi = s_lut[byte_val >> 4] * scale;

            __nv_bfloat16 a0_lo, a0_hi;
            *(unsigned short*)&a0_lo = (unsigned short)(a0_raw[b] & 0xFFFF);
            *(unsigned short*)&a0_hi = (unsigned short)(a0_raw[b] >> 16);
            acc0 += __bfloat162float(a0_lo) * w_lo;
            acc0 += __bfloat162float(a0_hi) * w_hi;

            __nv_bfloat16 a1_lo, a1_hi;
            *(unsigned short*)&a1_lo = (unsigned short)(a1_raw[b] & 0xFFFF);
            *(unsigned short*)&a1_hi = (unsigned short)(a1_raw[b] >> 16);
            acc1 += __bfloat162float(a1_lo) * w_lo;
            acc1 += __bfloat162float(a1_hi) * w_hi;

            __nv_bfloat16 a2_lo, a2_hi;
            *(unsigned short*)&a2_lo = (unsigned short)(a2_raw[b] & 0xFFFF);
            *(unsigned short*)&a2_hi = (unsigned short)(a2_raw[b] >> 16);
            acc2 += __bfloat162float(a2_lo) * w_lo;
            acc2 += __bfloat162float(a2_hi) * w_hi;
        }
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, offset);
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
        acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
    }

    if (warp_lane == 0) {
        unsigned int warp_idx = lane / WARP_SIZE;
        smem[local_out * 6 + warp_idx * 3]     = acc0;
        smem[local_out * 6 + warp_idx * 3 + 1] = acc1;
        smem[local_out * 6 + warp_idx * 3 + 2] = acc2;
    }
    __syncthreads();

    if (lane == 0) {
        float result0 = smem[local_out * 6]     + smem[local_out * 6 + 3];
        float result1 = smem[local_out * 6 + 1] + smem[local_out * 6 + 4];
        float result2 = smem[local_out * 6 + 2] + smem[local_out * 6 + 5];

        unsigned int group_dim = 2 * head_dim;
        unsigned int h = n / group_dim;
        unsigned int idx = n % group_dim;
        unsigned int q_total = num_heads * head_dim;

        unsigned int out_idx;
        if (idx < head_dim) {
            out_idx = h * head_dim + idx;
        } else {
            out_idx = q_total + h * head_dim + (idx - head_dim);
        }
        C[out_idx]  = __float2bfloat16(result0);
        C1[out_idx] = __float2bfloat16(result1);
        C2[out_idx] = __float2bfloat16(result2);
    }
}

// ============================================================
// W4A16 GEMV dual batch3: K+V for 3 input tokens in one launch
// ============================================================
// Processes 2 separate weight matrices (K and V) with 3 input vectors each.
// blockIdx.z selects K (0) or V (1). Both projections compute 3 outputs.
//
// Input:  A[3, K_in] BF16 (3 token hidden states)
// Output: C[3, N] where blockIdx.z=0 writes K, blockIdx.z=1 writes V
//
// Grid: (ceil(N / 4), 1, 2)   Block: (256, 1, 1)
extern "C" __global__ void w4a16_gemv_dual_batch3(
    const __nv_bfloat16* __restrict__ A,         // [3, K_in] BF16
    const unsigned char* __restrict__ B0_packed,  // [N, K_in/2] first projection
    const unsigned char* __restrict__ B0_scale,
    float B0_scale2,
    __nv_bfloat16* __restrict__ C0,              // [3, N] first projection output
    const unsigned char* __restrict__ B1_packed,  // [N, K_in/2] second projection
    const unsigned char* __restrict__ B1_scale,
    float B1_scale2,
    __nv_bfloat16* __restrict__ C1,              // [3, N] second projection output
    unsigned int N,
    unsigned int K_in
) {
    const unsigned int proj = blockIdx.z;
    const unsigned char* B_packed = (proj == 0) ? B0_packed : B1_packed;
    const unsigned char* B_scale = (proj == 0) ? B0_scale : B1_scale;
    float s2 = (proj == 0) ? B0_scale2 : B1_scale2;
    __nv_bfloat16* C_out = (proj == 0) ? C0 : C1;

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const unsigned int half_K = K_in / 2;
    const unsigned int num_groups = K_in / GROUP_SIZE;
    const unsigned int K8 = K_in / 8;

    const __nv_bfloat16* A1 = A + K_in;
    const __nv_bfloat16* A2 = A + 2 * K_in;
    __nv_bfloat16* C_out1 = C_out + N;
    __nv_bfloat16* C_out2 = C_out + 2 * N;

    __shared__ float s_lut[16];
    __shared__ float smem[N_PER_BLOCK * 6];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;

    for (unsigned int k8 = lane; k8 < K8; k8 += threads_per_out) {
        const unsigned int base_k = k8 * 8;

        uint4 a0_data = ((const uint4*)A)[k8];
        uint4 a1_data = ((const uint4*)A1)[k8];
        uint4 a2_data = ((const uint4*)A2)[k8];
        const unsigned int a0_raw[4] = {a0_data.x, a0_data.y, a0_data.z, a0_data.w};
        const unsigned int a1_raw[4] = {a1_data.x, a1_data.y, a1_data.z, a1_data.w};
        const unsigned int a2_raw[4] = {a2_data.x, a2_data.y, a2_data.z, a2_data.w};

        unsigned int packed4 = *(const unsigned int*)(B_packed + (unsigned long long)n * half_K + k8 * 4);
        unsigned int sg = base_k / GROUP_SIZE;
        unsigned char sb = B_scale[(unsigned long long)n * num_groups + sg];
        __nv_fp8_e4m3 fp8; *(unsigned char*)&fp8 = sb;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
        float scale = scl_fp8(sb) * s2;
#else
        float scale = (float)fp8 * s2;
#endif

        #pragma unroll
        for (int b = 0; b < 4; b++) {
            unsigned char bv = (packed4 >> (b * 8)) & 0xFF;
            float w_lo = s_lut[bv & 0xF] * scale;
            float w_hi = s_lut[bv >> 4] * scale;

            __nv_bfloat16 a0_lo, a0_hi;
            *(unsigned short*)&a0_lo = (unsigned short)(a0_raw[b] & 0xFFFF);
            *(unsigned short*)&a0_hi = (unsigned short)(a0_raw[b] >> 16);
            acc0 += __bfloat162float(a0_lo) * w_lo;
            acc0 += __bfloat162float(a0_hi) * w_hi;

            __nv_bfloat16 a1_lo, a1_hi;
            *(unsigned short*)&a1_lo = (unsigned short)(a1_raw[b] & 0xFFFF);
            *(unsigned short*)&a1_hi = (unsigned short)(a1_raw[b] >> 16);
            acc1 += __bfloat162float(a1_lo) * w_lo;
            acc1 += __bfloat162float(a1_hi) * w_hi;

            __nv_bfloat16 a2_lo, a2_hi;
            *(unsigned short*)&a2_lo = (unsigned short)(a2_raw[b] & 0xFFFF);
            *(unsigned short*)&a2_hi = (unsigned short)(a2_raw[b] >> 16);
            acc2 += __bfloat162float(a2_lo) * w_lo;
            acc2 += __bfloat162float(a2_hi) * w_hi;
        }
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, offset);
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
        acc2 += __shfl_down_sync(0xFFFFFFFF, acc2, offset);
    }

    if (warp_lane == 0) {
        unsigned int warp_idx = lane / WARP_SIZE;
        smem[local_out * 6 + warp_idx * 3]     = acc0;
        smem[local_out * 6 + warp_idx * 3 + 1] = acc1;
        smem[local_out * 6 + warp_idx * 3 + 2] = acc2;
    }
    __syncthreads();

    if (lane == 0) {
        float result0 = smem[local_out * 6]     + smem[local_out * 6 + 3];
        float result1 = smem[local_out * 6 + 1] + smem[local_out * 6 + 4];
        float result2 = smem[local_out * 6 + 2] + smem[local_out * 6 + 5];
        C_out[n]  = __float2bfloat16(result0);
        C_out1[n] = __float2bfloat16(result1);
        C_out2[n] = __float2bfloat16(result2);
    }
}

// ── ROW-PARALLEL M=8 TIER (batch8_rp) ───────────────────────────────────────
// The batchm template is ISSUE-bound at M>=5 (see the tier table above): the
// unrolled 8-row FMA chain costs 760 static SASS instructions vs batch4's
// 440, and batch8@M=8 lands at 50.7% of peak while batch4@M=4 holds 76%.
// This tier splits the rows ACROSS threads instead of down each thread's
// chain: 2 outputs per 256-thread CTA (grid ceil(N/2)), 128 threads per
// output, organized as TWO 64-lane row-groups — group 0 runs rows 0..3,
// group 1 rows 4..7. Each group replays the template's EXACT 64-lane
// two-phase virtual-lane walk, so every row's fmaf chain is bit-identical
// to `w4a16_gemv` (same parity contract as the template; the microtest
// gates it). Each thread now carries a batch4-class 4-row chain — the
// instruction profile that measures 76% — while the weight chunk is loaded
// by both groups at the same iteration: the second read is an L1 hit, so
// DRAM weight traffic is unchanged.
//
// Occupancy: 256-thread CTA, __launch_bounds__(256, 5) — same 5-CTA/SM pin
// as the narrow exact-M tiers. Grid doubles (N/2 vs N/4); at the serve
// shapes (N >= 5120) that is 2560+ CTAs — saturation is not a concern.
extern "C" __global__ void __launch_bounds__(256, 5) w4a16_gemv_batch8_rp(
    const __nv_bfloat16* __restrict__ A,          // [M, K]
    const unsigned char* __restrict__ B_packed,   // [N, K/2] uint8
    const unsigned char* __restrict__ B_scale,    // [N, K/GROUP_SIZE] FP8-E4M3
    const float scale2,
    __nv_bfloat16* __restrict__ C,                // [M, N]
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int OUT_PER_BLOCK = 2;                 // outputs per CTA
    const unsigned int lane128 = threadIdx.x % 128u;      // thread within output
    const unsigned int local_out = threadIdx.x / 128u;    // 0..1
    const unsigned int group = lane128 / 64u;             // row-group 0|1
    const unsigned int glane = lane128 % 64u;             // lane within group
    const unsigned int row_base = group * 4u;             // rows [base, base+4)
    const unsigned int n = blockIdx.x * OUT_PER_BLOCK + local_out;

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    // 4 accumulators per thread (this group's rows), reused across phases —
    // identical structure to the template's acc[MAX_M] at MAX_M=4.
    float acc[4];
    // Virtual-lane landing zone: [row][output][64 lanes of that row's group].
    __shared__ float s_vl_rp[8][2][64];

    #pragma unroll 1
    for (unsigned int phase = 0; phase < 2u; phase++) {
        #pragma unroll
        for (int t = 0; t < 4; t++) acc[t] = 0.0f;

        for (unsigned int kk = glane + phase * 64u; kk < K16; kk += 128u) {
            unsigned long long packed8 =
                *(const unsigned long long*)(B_packed + (unsigned long long)n * half_K + kk * 8);
            unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + kk];
            __nv_fp8_e4m3 fp8;
            *(unsigned char*)&fp8 = scale_byte;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
            float scale = scl_fp8(scale_byte) * scale2;
#else
            float scale = (float)fp8 * scale2;
#endif
            float wl[16];
            #pragma unroll
            for (int b = 0; b < 8; b++) {
                unsigned char byte_val = (unsigned char)(packed8 >> (b * 8));
                wl[b * 2]     = s_lut[byte_val & 0xF];
                wl[b * 2 + 1] = s_lut[byte_val >> 4];
            }

            #pragma unroll
            for (int t = 0; t < 4; t++) {
                const unsigned int row = row_base + (unsigned int)t;
                if (row >= M) continue;
                const __nv_bfloat16* At = A + (unsigned long long)row * K;
                uint4 a_lo = ((const uint4*)At)[kk * 2];
                uint4 a_hi = ((const uint4*)At)[kk * 2 + 1];
                const unsigned int ar[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                                            a_hi.x, a_hi.y, a_hi.z, a_hi.w};
                float part = 0.0f;
                #pragma unroll
                for (int b = 0; b < 8; b++) {
                    const float ax = __uint_as_float(ar[b] << 16);
                    const float ay = __uint_as_float(ar[b] & 0xFFFF0000u);
                    part = fmaf(ax, wl[b * 2], part);
                    part = fmaf(ay, wl[b * 2 + 1], part);
                }
                acc[t] = fmaf(scale, part, acc[t]);
            }
        }

        // Same phase-fold as the template: threads p and p^1 of THIS group
        // hold accumulator 0 and 1 of the same reference lane.
        #pragma unroll
        for (int t = 0; t < 4; t++) {
            const unsigned int row = row_base + (unsigned int)t;
            if (row >= M) continue;
            const float v = acc[t] + __shfl_xor_sync(0xFFFFFFFF, acc[t], 1);
            if ((glane & 1u) == 0u) {
                s_vl_rp[row][local_out][phase * WARP_SIZE + (glane >> 1)] = v;
            }
        }
    }
    __syncthreads();

    // Per-row reduce over the owning group's 64 virtual lanes — the
    // template's two-warp shuffle tree, indexed by glane.
    __shared__ float smem_rp[8][2][2];
    const unsigned int warp_in_group = glane / WARP_SIZE;
    #pragma unroll
    for (int t = 0; t < 4; t++) {
        const unsigned int row = row_base + (unsigned int)t;
        if (row >= M) continue;
        float a = s_vl_rp[row][local_out][glane];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            a += __shfl_down_sync(0xFFFFFFFF, a, offset);
        }
        if (glane % WARP_SIZE == 0) smem_rp[row][local_out][warp_in_group] = a;
    }
    __syncthreads();

    if (glane == 0) {
        #pragma unroll
        for (int t = 0; t < 4; t++) {
            const unsigned int row = row_base + (unsigned int)t;
            if (row >= M) continue;
            float r = smem_rp[row][local_out][0] + smem_rp[row][local_out][1];
            C[(unsigned long long)row * N + n] = __float2bfloat16(r);
        }
    }
}

// ── PHASE-PARALLEL M=8 TIER v2 (batch8_pp) ──────────────────────────────────
// v1 (row-parallel, above) is bit-exact but LOSES ~10% to batch8: the two
// row-groups decode every weight chunk twice and the CTA amortizes the
// activation walk over half the outputs. v2 keeps the template's row layout
// and instead runs the TWO K-PHASES on different thread halves: 2 outputs
// per 256-thread CTA, 128 threads per output = 64 lanes x 2 phase-groups.
// Each thread carries the full 8-row accumulator set but walks only HALF
// the chunks (its phase's disjoint K half), so per-thread issue halves and
// no weight chunk is decoded twice. The fmaf chain per (reference lane,
// accumulator) is IDENTICAL to the template's sequential-phase execution —
// concurrency only changes which register file holds the partial — so
// bit-parity vs batch8 is preserved by construction (gate 2c).
//
// MEASURED (2026-08-19, cold-cycled, sm_121f): ALSO ~NEUTRAL — qkv/o
// 116.0 vs batch8's 113.6 us at M=8; ffn_down +7%, ffn_up/lm_head worse.
// Together with the v1 result this pins the diagnosis: the family is
// PER-SM ISSUE-BOUND (~164 inst per 8-byte weight chunk per output), and
// redistributing the same instructions across threads is invariant. A real
// M=8 win must CUT instructions/byte (packed HFMA2 math, or gemm_t-style
// tiled decode amortization) — and every such change alters the FP32
// chain, i.e. breaks the bit-parity contract this family is gated on.
// Next step is a POLICY decision (relax parity for the verify path and
// gate on accept-rate/argmax stability), not another thread layout. Both
// prototypes stay for the day that decision lands; nothing dispatches them.
extern "C" __global__ void __launch_bounds__(256, 5) w4a16_gemv_batch8_pp(
    const __nv_bfloat16* __restrict__ A,          // [M, K]
    const unsigned char* __restrict__ B_packed,   // [N, K/2] uint8
    const unsigned char* __restrict__ B_scale,    // [N, K/GROUP_SIZE] FP8-E4M3
    const float scale2,
    __nv_bfloat16* __restrict__ C,                // [M, N]
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int lane128 = threadIdx.x % 128u;
    const unsigned int local_out = threadIdx.x / 128u;    // 0..1
    const unsigned int phase = lane128 / 64u;             // 0|1
    const unsigned int lane = lane128 % 64u;              // template lane
    const unsigned int n = blockIdx.x * 2u + local_out;

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    if (n >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    float acc[8];
    #pragma unroll
    for (int t = 0; t < 8; t++) acc[t] = 0.0f;
    // Virtual-lane landing zone, identical shape to the template's
    // s_vl[MAX_M][N_PER_BLOCK][2*WARP_SIZE] at MAX_M=8, N_PER_BLOCK=2.
    __shared__ float s_vl_pp[8][2][2 * WARP_SIZE];

    for (unsigned int kk = lane + phase * 64u; kk < K16; kk += 128u) {
        unsigned long long packed8 =
            *(const unsigned long long*)(B_packed + (unsigned long long)n * half_K + kk * 8);
        unsigned char scale_byte = B_scale[(unsigned long long)n * num_groups + kk];
        __nv_fp8_e4m3 fp8;
        *(unsigned char*)&fp8 = scale_byte;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
        float scale = scl_fp8(scale_byte) * scale2;
#else
        float scale = (float)fp8 * scale2;
#endif
        float wl[16];
        #pragma unroll
        for (int b = 0; b < 8; b++) {
            unsigned char byte_val = (unsigned char)(packed8 >> (b * 8));
            wl[b * 2]     = s_lut[byte_val & 0xF];
            wl[b * 2 + 1] = s_lut[byte_val >> 4];
        }

        #pragma unroll
        for (int t = 0; t < 8; t++) {
            if ((unsigned int)t >= M) continue;
            const __nv_bfloat16* At = A + (unsigned long long)t * K;
            uint4 a_lo = ((const uint4*)At)[kk * 2];
            uint4 a_hi = ((const uint4*)At)[kk * 2 + 1];
            const unsigned int ar[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                                        a_hi.x, a_hi.y, a_hi.z, a_hi.w};
            float part = 0.0f;
            #pragma unroll
            for (int b = 0; b < 8; b++) {
                const float ax = __uint_as_float(ar[b] << 16);
                const float ay = __uint_as_float(ar[b] & 0xFFFF0000u);
                part = fmaf(ax, wl[b * 2], part);
                part = fmaf(ay, wl[b * 2 + 1], part);
            }
            acc[t] = fmaf(scale, part, acc[t]);
        }
    }

    // Phase fold: within a phase-group, threads p and p^1 hold accumulator
    // 0 and 1 of the same reference lane — identical to the template.
    #pragma unroll
    for (int t = 0; t < 8; t++) {
        if ((unsigned int)t >= M) continue;
        const float v = acc[t] + __shfl_xor_sync(0xFFFFFFFF, acc[t], 1);
        if ((lane & 1u) == 0u) {
            s_vl_pp[t][local_out][phase * WARP_SIZE + (lane >> 1)] = v;
        }
    }
    __syncthreads();

    // Template's two-warp reduce over the 64 virtual lanes; phase-group 0's
    // threads do the tree (phase-group 1 idles here — 64 lanes suffice).
    __shared__ float smem_pp[8][2][2];
    if (phase == 0u) {
        const unsigned int warp_in_out = lane / WARP_SIZE;
        #pragma unroll
        for (int t = 0; t < 8; t++) {
            if ((unsigned int)t >= M) continue;
            float a = s_vl_pp[t][local_out][lane];
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                a += __shfl_down_sync(0xFFFFFFFF, a, offset);
            }
            if (lane % WARP_SIZE == 0) smem_pp[t][local_out][warp_in_out] = a;
        }
    }
    __syncthreads();

    if (lane128 == 0) {
        #pragma unroll
        for (int t = 0; t < 8; t++) {
            if ((unsigned int)t >= M) continue;
            float r = smem_pp[t][local_out][0] + smem_pp[t][local_out][1];
            C[(unsigned long long)t * N + n] = __float2bfloat16(r);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════
// PORTED FROM PR #648 (Ronald R. Stesiak), commit acaf9533.
//
// Taken on its own rather than merging that branch: #648 is 171 files and
// conflicts with main, and most of it is a separate drafter lane. This
// kernel is the part that stands alone — the batch-GEMV tier is shared by
// every M<=8 verify shape, so it pays off independently of which drafter
// sits on top.
//
// Only the T=2 variant and its template are taken. The sibling experiments
// in #648 (pf / pf_free / pf2 / pf3 / rt4) are the search that found it;
// his own notes record the weight-prefetch line as a WASH at M=8, so
// carrying them here would be importing dead ends.
// ════════════════════════════════════════════════════════════════════════
// ── Register-tiled batchm (lever-1 v3: T outputs per thread) ────────────
// provenance-id: 526f6e616c6420522e205374657369616b
//
// The M-sweep verdict (batchm_bench 2026-08-19, clean runs): near-peak at
// M=1 (250 GB/s), monotonic decay to ~143 at M=8; batch16-vs-batch8 at
// equal M shows template footprint alone costs bandwidth; both prefetch
// families were washes. The remaining suspects — L2 activation traffic
// (256 B per (chunk,row) re-read per output), activation load-instruction
// count, and single-dependent-FMA-chain ILP — are ALL divided by T when
// each thread computes T adjacent output rows from ONE set of activation
// registers. Layout: N_PER_BLOCK lane-groups of 64 as before, each group
// covering T ADJACENT n rows; grid shrinks to ceil(N / (N_PER_BLOCK*T)).
// Each output row\'s chunk walk, per-chunk FMA sequence, pair-fold and
// reduction tree are IDENTICAL to `w4a16_gemv_batchm_impl` — bit-exact by
// construction, and batchm_bench gate 4 proves it, never assumes it.
template <int MAX_M, int T>
__device__ __forceinline__ void w4a16_gemv_batchm_impl_rt(
    const __nv_bfloat16* __restrict__ A,         // [M, K]
    const unsigned char* __restrict__ B_packed,   // [N, K/2] uint8
    const unsigned char* __restrict__ B_scale,    // [N, K/GROUP_SIZE] FP8-E4M3
    const float scale2,
    __nv_bfloat16* __restrict__ C,                // [M, N]
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;  // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    // This lane group covers T ADJACENT output rows n0 .. n0+T-1.
    const unsigned int n0 = (blockIdx.x * N_PER_BLOCK + local_out) * T;

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    if (n0 >= N) return;

    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;
    const unsigned int K16 = K / 16;

    float acc[T][MAX_M];
    // Virtual-lane slab per (row, group, OUTPUT) — the T dimension is what
    // the first draft of this kernel missed.
    __shared__ float s_vl[MAX_M][N_PER_BLOCK][T][2 * WARP_SIZE];

    #pragma unroll 1
    for (unsigned int phase = 0; phase < 2u; phase++) {
        #pragma unroll
        for (int o = 0; o < T; o++)
            #pragma unroll
            for (int t = 0; t < MAX_M; t++) acc[o][t] = 0.0f;

        for (unsigned int kk = lane + phase * threads_per_out; kk < K16;
             kk += threads_per_out * 2u) {
            // T weight chunks (one per adjacent output row) + T scales.
            unsigned long long packed8[T];
            float scale[T];
            #pragma unroll
            for (int o = 0; o < T; o++) {
                const unsigned long long n = n0 + o;
                if (n < N) {
                    packed8[o] = *(const unsigned long long*)(B_packed + n * half_K + kk * 8);
                    const unsigned char sb = B_scale[n * num_groups + kk];
                    __nv_fp8_e4m3 fp8;
                    *(unsigned char*)&fp8 = sb;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
                    scale[o] = scl_fp8(sb) * scale2;
#else
                    scale[o] = (float)fp8 * scale2;
#endif
                } else {
                    packed8[o] = 0ull;
                    scale[o] = 0.0f;
                }
            }
            float wl[T][16];
            #pragma unroll
            for (int o = 0; o < T; o++) {
                #pragma unroll
                for (int b = 0; b < 8; b++) {
                    unsigned char byte_val = (unsigned char)(packed8[o] >> (b * 8));
                    wl[o][b * 2]     = s_lut[byte_val & 0xF];
                    wl[o][b * 2 + 1] = s_lut[byte_val >> 4];
                }
            }

            #pragma unroll
            for (int t = 0; t < MAX_M; t++) {
                if ((unsigned int)t >= M) continue;
                const __nv_bfloat16* At = A + (unsigned long long)t * K;
                // ONE activation load per (chunk, row) feeding T chains.
                uint4 a_lo = ((const uint4*)At)[kk * 2];
                uint4 a_hi = ((const uint4*)At)[kk * 2 + 1];
                const unsigned int ar[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                                            a_hi.x, a_hi.y, a_hi.z, a_hi.w};
                #pragma unroll
                for (int o = 0; o < T; o++) {
                    // Per-output chain: the M=1 kernel\'s exact FMA order.
                    float part = 0.0f;
                    #pragma unroll
                    for (int b = 0; b < 8; b++) {
                        float2 af = __bfloat1622float2(*(const __nv_bfloat162*)&ar[b]);
                        part = fmaf(af.x, wl[o][b * 2], part);
                        part = fmaf(af.y, wl[o][b * 2 + 1], part);
                    }
                    acc[o][t] = fmaf(scale[o], part, acc[o][t]);
                }
            }
        }

        // Pair-fold + virtual-lane landing, per output row — same tree.
        #pragma unroll
        for (int o = 0; o < T; o++) {
            #pragma unroll
            for (int t = 0; t < MAX_M; t++) {
                if ((unsigned int)t >= M) continue;
                const float v = acc[o][t] + __shfl_xor_sync(0xFFFFFFFF, acc[o][t], 1);
                if ((lane & 1u) == 0u) {
                    s_vl[t][local_out][o][phase * WARP_SIZE + (lane >> 1)] = v;
                }
            }
        }
    }
    __syncthreads();

    __shared__ float smem[MAX_M][N_PER_BLOCK * T * 2];  // 2 warps per (out,o)
    const unsigned int warp_in_out = lane / WARP_SIZE;
    #pragma unroll
    for (int o = 0; o < T; o++) {
        #pragma unroll
        for (int t = 0; t < MAX_M; t++) {
            if ((unsigned int)t >= M) continue;
            float a = s_vl[t][local_out][o][lane];
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                a += __shfl_down_sync(0xFFFFFFFF, a, offset);
            }
            if (lane % WARP_SIZE == 0)
                smem[t][(local_out * T + o) * 2 + warp_in_out] = a;
        }
    }
    __syncthreads();

    if (lane == 0) {
        #pragma unroll
        for (int o = 0; o < T; o++) {
            const unsigned int n = n0 + o;
            if (n >= N) continue;
            #pragma unroll
            for (int t = 0; t < MAX_M; t++) {
                if ((unsigned int)t >= M) continue;
                float r = smem[t][(local_out * T + o) * 2]
                        + smem[t][(local_out * T + o) * 2 + 1];
                C[(unsigned long long)t * N + n] = __float2bfloat16(r);
            }
        }
    }
}

// Register-tiled batch8, T=2 outputs/thread. Grid: ceil(N/8).
extern "C" __global__ void w4a16_gemv_batch8_rt2(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    w4a16_gemv_batchm_impl_rt<8, 2>(A, B_packed, B_scale, scale2, C, M, N, K);
}
