// SPDX-License-Identifier: AGPL-3.0-only

// W4A4 NVFP4 GEMM for SM120 — the native answer to the CUTLASS grouped/dense
// NVFP4 path, with NO CUTLASS dependency.
//
// WHY THIS EXISTS. An add-one-in survey of all 8 CUTLASS-selectable paths found
// exactly two carrying a real gap on the 35B NVFP4 (cold TTFT, n=8/arm):
//     nvfp4-qkvz    -35.5 ms      moe-grouped   -38.7 ms
//     dense-gemm      0.0 ms      attn-kv/o     +1.3/+1.4 ms (we are FASTER)
// Both gaps route through the `w4a16_gemm*` family, and both have one cause:
// CUTLASS quantizes ACTIVATIONS to NVFP4 and feeds block-scaled FP4 tensor
// cores (`m16n8k64.e2m1.e2m1`), while w4a16 dequantizes weights to bf16 and
// runs `m16n8k16.bf16.bf16`. That is 4x the K-depth per instruction and 4x less
// smem traffic for B.
//
// ★ THIS CHANGES NUMERICS: activations go bf16 -> e2m1 (per-16-element ue4m3
// scale). It is NOT a drop-in for w4a16 and must clear the full accuracy gate
// suite before it can be a default. Gated behind ATLAS_W4A4_QKVZ.
//
// The MMA fragment/scale layout below is not guesswork: `fp4_mma_microtest.cu`
// proves this exact instruction and operand mapping against the CUTLASS Sm120
// collective at cos = 1.000000 (see examples/fp4_mma_microproof.rs).

// ═══════════════════════════════════════════════════════════════════════════
// ⛔ PARKED — DO NOT WIRE INTO ANY DISPATCH PATH WITHOUT AN EXPLICIT DECISION.
//
// This file is CORRECT and UNUSED. It is kept as a proven building block plus a
// continuation plan, so whoever resumes starts at 1.77x rather than at zero.
//
// ── WHY IT IS PARKED ──────────────────────────────────────────────────────
// Enabling it moves the flagship NVFP4 path from W4A16 to W4A4: activations go
// bf16 -> e2m1. That is a NUMERICS change on a shipping model, and at the
// current 1.77x it would buy only ~15 ms of cold TTFT (574 -> ~559, -2.6%).
// A precision change is not worth 2.6%. It becomes worth discussing at ~1.1x or
// better, where the prize is the full ~71 ms.
//
// ── WHAT IS ALREADY ESTABLISHED (do not re-derive) ────────────────────────
// 1. The gap to CUTLASS is PRECISION, not engineering. An add-one-in survey of
//    all 8 CUTLASS-selectable paths (35B NVFP4, cold TTFT, n=8/arm, one binary):
//        nvfp4-qkvz   -35.5 ms     moe-grouped  -38.7 ms
//        dense-gemm     0.0 ms     attn-kv/-o   +1.3/+1.4 ms (WE are faster)
//        ssm-out/attn-q -5.0/-4.4 ms (marginal)
//    Only qkvz and grouped-MoE carry a gap, and both route through the
//    `w4a16_gemm*` family.
// 2. IT IS NOT MISSING PIPELINING. `moe_w4a16_grouped_gemm_ptrtable_t_k64` — the
//    kernel `grouped_silu_down` actually runs — ALREADY has K_STEP_T64=64,
//    double buffering, 45 cp.async uses, PAD_T64/BP_PAD bank padding and
//    N_TILE_LG=128. Checked, not assumed. There is no structural work left at
//    W4A16, which is why W4A4 is the only route.
// 3. The MMA and its operand/scale layout are PROVEN: cos = 1.000000 against the
//    CUTLASS Sm120 collective at every shape tried. See
//    `examples/fp4_mma_microproof.rs` and `fp4_mma_microtest.cu`.
//
// ── MEASURED LADDER (M=2048 N=12288 K=2048, the real qkvz shape; like-for-like
//    including the activation pack) ──────────────────────────────────────────
//        naive (no reuse)          15.254 ms   24.8x    ~4.7 GB redundant reads
//        tiled 64x128               1.353 ms    2.24x
//        pipelined 64x128           1.437 ms    2.40x   NO HELP at large M
//        128x128                    1.117 ms    1.87x
//        256x128                    1.173 ms    1.83x   REGRESSED: 209 regs -> 1 CTA/SM
//        128x128 + bank padding     1.115 ms    1.77x   <- best, this file
//        CUTLASS                    ~0.63  ms   1.00x   <- target
//        w4a16 (derived)            ~1.5   ms   2.4x    <- the bar to beat
//
// ── THE MULTI-DAY PLAN, IN ORDER ──────────────────────────────────────────
// Three consecutive structural fixes each delivered far LESS than their model
// predicted (pipelining, bigger tile, bank padding). That pattern says the
// remaining 1.77x is not reachable by more of the same. DO NOT START WITH TILE
// SIZE SWEEPS — that ground is covered above.
//
// Day 1 — measure where the 1.77x actually goes.
//   ncu the 128x128 kernel and CUTLASS's collective on the SAME shape and diff
//   them: sm__throughput, dram__throughput, warp stall reasons, achieved
//   occupancy, smem bank conflicts. Every optimisation above was chosen from a
//   traffic model; none was chosen from a stall profile, and the models kept
//   over-predicting. ★ Do NOT ncu a live serve on GB10: kernel replay snapshots
//   the working set and unified memory has no headroom — it hard-froze this box
//   twice. Profile `examples/w4a4_bench.rs` instead (~6 GB).
//
// Day 2 — multi-stage async pipeline (3-4 deep), not 2.
//   The 2-stage attempt here hurt at large M because 3072 CTAs already hide the
//   load latency between CTAs. A deeper pipeline only pays alongside a LARGER
//   tile, where CTA count drops and intra-CTA overlap starts to matter. These
//   two must be tried TOGETHER; separately they cancel (that is exactly what the
//   256x128 row above shows).
//
// Day 3 — warp specialisation.
//   Split producer (cp.async staging) from consumer (MMA) warps, as the CUTLASS
//   collective does. This is the single biggest structural difference remaining
//   and the reason its 128x128x128 tile sustains what ours cannot.
//
// Day 4 — swizzled smem layout.
//   The `W4A4_KBP 48` padding here is a crude stand-in for a real XOR swizzle.
//   Padding cost smem and bought ~0 absolute time, so a swizzle is worth doing
//   properly rather than widening the pad further.
//
// Day 5 — grouped variant for the MoE path.
//   qkvz is a dense GEMM; `grouped_silu_down`/`grouped_gate_up` need per-expert
//   pointer tables. Do this only AFTER the dense kernel is within ~1.1x, or the
//   grouped version inherits the same deficit with more moving parts.
//
// ── GATE BEFORE ANY CLAIM ─────────────────────────────────────────────────
// cos >= 0.999 vs the CUTLASS oracle is NECESSARY BUT NOT SUFFICIENT. The
// shelved SPLIT=4 GDN spine read cos 1.0000 and still failed two accuracy gates.
// Run the 4-minute ssm-poisoning tripwire (12/12 byte-identical) and then the
// FULL accuracy suite before treating any speed number as shippable.
// ═══════════════════════════════════════════════════════════════════════════

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

// 8 warps per CTA, each owning one 16x8 MMA tile: 2 along M (2*16=32) x 4 along
// N (4*8=32). The CTA tile MUST equal that coverage — declaring 64x64 here while
// the warps only reach 32x32 leaves 3/4 of the output unwritten (cos 0.4955).
#define W4A4_M_TILE   32
#define W4A4_N_TILE   32
#define W4A4_K_STEP   64
#define W4A4_GROUP    16     // elements per ue4m3 scale

// ── ue4m3 encode/decode (mirrors the cutlass pack; see microtest) ────────────
// ue4m3 is the CUTLASS float_ue4m3_t byte: e4m3 magnitude with the sign bit
// clear (scales are non-negative). Use the HARDWARE conversion rather than
// hand-rolled exponent/mantissa bit work — a hand-rolled version of this
// scored cos 0.9909 against the CUTLASS oracle where this one scores 1.000000.
__device__ __forceinline__ unsigned char w4a4_f2ue4m3(float scale) {
    __nv_fp8_e4m3 v = __nv_fp8_e4m3(scale);
    return *reinterpret_cast<unsigned char*>(&v);
}

// Decode a ue4m3 byte back to float. REQUIRED, not cosmetic: elements must be
// quantized against the scale the MMA will actually multiply by (the ROUNDED
// ue4m3), not the raw float it was derived from. Dividing by the unrounded
// scale leaves a systematic residual on every element — measured cos 0.9909
// against the CUTLASS oracle at every shape tried, versus 1.000000 with this.
__device__ __forceinline__ float w4a4_ue4m3_to_f(unsigned char byte) {
    __nv_fp8_e4m3 v = *reinterpret_cast<__nv_fp8_e4m3*>(&byte);
    return (float)v;
}

// e2m1 round-to-nearest of v/scale, returned as a 4-bit code.
__device__ __forceinline__ unsigned char w4a4_f2e2m1(float x) {
    unsigned char sign = x < 0.0f ? 8 : 0;
    float a = fabsf(x);
    unsigned char mag;
    if      (a <= 0.25f) mag = 0;
    else if (a <= 0.75f) mag = 1;
    else if (a <= 1.25f) mag = 2;
    else if (a <= 1.75f) mag = 3;
    else if (a <= 2.5f)  mag = 4;
    else if (a <= 3.5f)  mag = 5;
    else if (a <= 5.0f)  mag = 6;
    else                mag = 7;
    return (unsigned char)(sign | mag);
}

// ── Pack bf16 activations [M,K] -> packed e2m1 [M,K/2] + ue4m3 [M,K/16] ──────
// One thread per 16-element group. Mirrors fp4_microtest_pack, which is the
// kernel validated against the CUTLASS activation pack.
extern "C" __global__ void w4a4_pack_act(
    const __nv_bfloat16* __restrict__ src,
    unsigned char* __restrict__ packed,
    unsigned char* __restrict__ scales,
    int m, int k) {
    const unsigned long long gid =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const int groups = k / W4A4_GROUP;
    const unsigned long long total = (unsigned long long)m * groups;
    if (gid >= total) return;
    const int row = (int)(gid / groups);
    const int group = (int)(gid % groups);
    const int base = group * W4A4_GROUP;

    float v[W4A4_GROUP];
    float max_abs = 0.0f;
    #pragma unroll
    for (int i = 0; i < W4A4_GROUP; ++i) {
        v[i] = __bfloat162float(src[(unsigned long long)row * k + base + i]);
        max_abs = fmaxf(max_abs, fabsf(v[i]));
    }
    const float scale = max_abs > 0.0f ? max_abs / 6.0f : 1.0f;
    const unsigned char sf = w4a4_f2ue4m3(scale);
    scales[(unsigned long long)row * groups + group] = sf;
    const float decoded = w4a4_ue4m3_to_f(sf);
    const float inv = decoded > 0.0f ? 1.0f / decoded : 0.0f;
    #pragma unroll
    for (int i = 0; i < W4A4_GROUP; i += 2) {
        const unsigned char lo = w4a4_f2e2m1(v[i] * inv);
        const unsigned char hi = w4a4_f2e2m1(v[i + 1] * inv);
        packed[(unsigned long long)row * (k / 2) + (base + i) / 2] =
            (unsigned char)(lo | (hi << 4));
    }
}

// ── fragment gather helpers (layout proven by fp4_mma_microtest) ─────────────
// 8 consecutive e2m1 from packed[row][kk..kk+7] (kk even) into one u32:
// nibble j is element kk+j; byte (kk+j)/2 low/high.
__device__ __forceinline__ unsigned int w4a4_gather_a8(
    const unsigned char* __restrict__ packed, int row, int kk, int k) {
    const unsigned char* p = packed + (unsigned long long)row * (k / 2) + kk / 2;
    unsigned int out = 0;
    #pragma unroll
    for (int j = 0; j < 8; j += 2) {
        const unsigned char b = p[j / 2];
        out |= (unsigned int)(b & 0x0F) << (4 * j);
        out |= (unsigned int)(b >> 4) << (4 * (j + 1));
    }
    return out;
}

// 4 ue4m3 scale bytes for k-groups [g0, g0+4) of `row`.
__device__ __forceinline__ unsigned int w4a4_gather_sf4(
    const unsigned char* __restrict__ scales, int row, int g0, int k) {
    const int groups = k / W4A4_GROUP;
    const unsigned char* s = scales + (unsigned long long)row * groups + g0;
    unsigned int out = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) out |= (unsigned int)s[j] << (8 * j);
    return out;
}

// ── W4A4 GEMM: C[M,N] = A[M,K] * B[N,K]^T, all NVFP4 block-scaled ───────────
// Warp-per-(16m x 8n) output tile, K walked in 64-element steps — one
// m16n8k64 MMA per step, versus four m16n8k16 bf16 MMAs in w4a16.
//
// B is the WEIGHT in [N,K] packed layout (already NVFP4 on disk — no dequant,
// which is the whole point: w4a16 spends a smem buffer and a LUT pass turning
// these nibbles into bf16 before every MMA).
extern "C" __global__ void __launch_bounds__(256, 1)
w4a4_gemm_t(
    const unsigned char* __restrict__ packed_a,
    const unsigned char* __restrict__ scales_a,
    const unsigned char* __restrict__ packed_b,
    const unsigned char* __restrict__ scales_b,
    __nv_bfloat16* __restrict__ out,
    int m, int n, int k) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    // 8 warps per CTA: 2 along M (16 rows each) x 4 along N (8 cols each)
    const int tile_m = blockIdx.y * W4A4_M_TILE + (warp >> 2) * 16;
    const int tile_n = blockIdx.x * W4A4_N_TILE + (warp & 3) * 8;
    if (tile_m >= m || tile_n >= n) return;

    const int q = lane & 3;
    const int r = lane >> 2;
    const int sfa_m = (lane & 1) * 8 + (lane >> 2);
    const int sfb_n = lane >> 2;

    float acc[4] = {0.f, 0.f, 0.f, 0.f};

    for (int k0 = 0; k0 < k; k0 += W4A4_K_STEP) {
        const unsigned int a0 = w4a4_gather_a8(packed_a, tile_m + r,     k0 +      q * 8, k);
        const unsigned int a1 = w4a4_gather_a8(packed_a, tile_m + r + 8, k0 +      q * 8, k);
        const unsigned int a2 = w4a4_gather_a8(packed_a, tile_m + r,     k0 + 32 + q * 8, k);
        const unsigned int a3 = w4a4_gather_a8(packed_a, tile_m + r + 8, k0 + 32 + q * 8, k);
        const unsigned int b0 = w4a4_gather_a8(packed_b, tile_n + r,     k0 +      q * 8, k);
        const unsigned int b1 = w4a4_gather_a8(packed_b, tile_n + r,     k0 + 32 + q * 8, k);
        const unsigned int sfa = w4a4_gather_sf4(scales_a, tile_m + sfa_m, k0 / 16, k);
        const unsigned int sfb = w4a4_gather_sf4(scales_b, tile_n + sfb_n, k0 / 16, k);
#if (__CUDA_ARCH__ >= 1200)
        unsigned short bidA = 0, tidA = 0, bidB = 0, tidB = 0;
        asm volatile(
            "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
            "{%0,  %1,  %2,  %3},"
            "{%4,  %5,  %6,  %7},"
            "{%8,  %9},"
            "{%10, %11, %12, %13},"
            "{%14},"
            "{%15, %16},"
            "{%17},"
            "{%18, %19};\n"
            : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1),
              "f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3]),
              "r"(sfa), "h"(bidA), "h"(tidA),
              "r"(sfb), "h"(bidB), "h"(tidB));
#endif
    }

    // Epilogue: acc[0..3] -> rows (r, r+8) x cols (2q, 2q+1) of the 16x8 tile.
    const int om0 = tile_m + r;
    const int om1 = tile_m + r + 8;
    const int on0 = tile_n + q * 2;
    if (om0 < m && on0     < n) out[(unsigned long long)om0 * n + on0]     = __float2bfloat16(acc[0]);
    if (om0 < m && on0 + 1 < n) out[(unsigned long long)om0 * n + on0 + 1] = __float2bfloat16(acc[1]);
    if (om1 < m && on0     < n) out[(unsigned long long)om1 * n + on0]     = __float2bfloat16(acc[2]);
    if (om1 < m && on0 + 1 < n) out[(unsigned long long)om1 * n + on0 + 1] = __float2bfloat16(acc[3]);
}


// ── TILED W4A4 GEMM ─────────────────────────────────────────────────────────
// The kernel above is correct (cos 1.000000 vs CUTLASS) but has NO reuse: each
// 16x8 warp tile re-reads its own A and B slices from global, which at
// M=2048,N=12288,K=2048 is ~4.7 GB of traffic and measured 15.2 ms/iter --
// almost exactly 4.7GB / 273 GB/s, i.e. purely redundant-read bound.
//
// This version stages a 64(M) x 128(N) CTA tile through shared memory, one
// K_STEP=64 slice at a time. Same MMA, same fragment layout, same epilogue --
// only the operand source changes (smem instead of global), so correctness is
// inherited. Traffic drops to ~590 MB for the same GEMM.
//
// 8 warps, each owning 2 M-positions x 4 N-positions = 8 output tiles, so the
// staged slice is reused 8x per warp before the next K step.
#define W4A4_M_CTA 64
#define W4A4_N_CTA 128
#define W4A4_KB    (W4A4_K_STEP / 2)        // 32 packed bytes per row per step
#define W4A4_KG    (W4A4_K_STEP / W4A4_GROUP) // 4 scale bytes per row per step

// Padded smem row stride. A warp reads bank (row*stride/4 + q) % 32 with
// row = lane>>2 (0..7) and q = lane&3. At the natural 32 B stride that is
// (row*8+q)%32 -> only 16 distinct banks, a 2-way conflict on EVERY smem read.
// 48 B gives (row*12+q)%32 -> all 32 banks, conflict-free.
#define W4A4_KBP 48

__device__ __forceinline__ unsigned int w4a4_smem_a8(
    const unsigned char* __restrict__ sm, int row, int kloc) {
    // 8 e2m1 = 4 bytes at [row][kloc/2]; kloc is a multiple of 8 so 4-byte aligned.
    return *reinterpret_cast<const unsigned int*>(sm + row * W4A4_KB + (kloc >> 1));
}
__device__ __forceinline__ unsigned int w4a4_smem_a8p(
    const unsigned char* __restrict__ sm, int row, int kloc) {
    return *reinterpret_cast<const unsigned int*>(sm + row * W4A4_KBP + (kloc >> 1));
}
__device__ __forceinline__ unsigned int w4a4_smem_sf4(
    const unsigned char* __restrict__ sm, int row) {
    return *reinterpret_cast<const unsigned int*>(sm + row * W4A4_KG);
}

extern "C" __global__ void __launch_bounds__(256, 1)
w4a4_gemm_t_tiled(
    const unsigned char* __restrict__ packed_a,
    const unsigned char* __restrict__ scales_a,
    const unsigned char* __restrict__ packed_b,
    const unsigned char* __restrict__ scales_b,
    __nv_bfloat16* __restrict__ out,
    int m, int n, int k) {
    __shared__ unsigned char sA[W4A4_M_CTA * W4A4_KB];
    __shared__ unsigned char sAs[W4A4_M_CTA * W4A4_KG];
    __shared__ unsigned char sB[W4A4_N_CTA * W4A4_KB];
    __shared__ unsigned char sBs[W4A4_N_CTA * W4A4_KG];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int cta_m = blockIdx.y * W4A4_M_CTA;
    const int cta_n = blockIdx.x * W4A4_N_CTA;

    const int q = lane & 3;
    const int r = lane >> 2;
    const int sfa_m = (lane & 1) * 8 + (lane >> 2);
    const int sfb_n = lane >> 2;
    const int wm = warp >> 2;    // 0..1
    const int wn = warp & 3;     // 0..3

    float acc[2][4][4];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            #pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.0f;

    const int groups = k / W4A4_GROUP;

    for (int k0 = 0; k0 < k; k0 += W4A4_K_STEP) {
        __syncthreads();
        // Stage A: 64 rows x 32 bytes = 2048 B -> 8 B per thread.
        {
            const int row = tid >> 2, col4 = (tid & 3) << 3;   // 8 bytes each
            const int gm = cta_m + row;
            unsigned long long src = (unsigned long long)gm * (k / 2) + (k0 >> 1) + col4;
            unsigned char* dst = sA + row * W4A4_KB + col4;
            if (gm < m) *reinterpret_cast<ulonglong1*>(dst) = *reinterpret_cast<const ulonglong1*>(packed_a + src);
            else *reinterpret_cast<ulonglong1*>(dst) = make_ulonglong1(0);
        }
        // Stage B: 128 rows x 32 bytes = 4096 B -> 16 B per thread.
        {
            const int row = tid >> 1, col8 = (tid & 1) << 4;
            const int gn = cta_n + row;
            unsigned long long src = (unsigned long long)gn * (k / 2) + (k0 >> 1) + col8;
            unsigned char* dst = sB + row * W4A4_KB + col8;
            if (gn < n) *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(packed_b + src);
            else *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
        }
        // Scales: A 64x4 = 256 B, B 128x4 = 512 B -> one u32 per row.
        if (tid < W4A4_M_CTA) {
            const int gm = cta_m + tid;
            *reinterpret_cast<unsigned int*>(sAs + tid * W4A4_KG) =
                (gm < m) ? *reinterpret_cast<const unsigned int*>(
                    scales_a + (unsigned long long)gm * groups + (k0 / W4A4_GROUP)) : 0u;
        }
        if (tid < W4A4_N_CTA) {
            const int gn = cta_n + tid;
            *reinterpret_cast<unsigned int*>(sBs + tid * W4A4_KG) =
                (gn < n) ? *reinterpret_cast<const unsigned int*>(
                    scales_b + (unsigned long long)gn * groups + (k0 / W4A4_GROUP)) : 0u;
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            const int lm = wm * 16 + i * 32;          // 0,32 / 16,48
            const unsigned int a0 = w4a4_smem_a8(sA, lm + r,     q * 8);
            const unsigned int a1 = w4a4_smem_a8(sA, lm + r + 8, q * 8);
            const unsigned int a2 = w4a4_smem_a8(sA, lm + r,     32 + q * 8);
            const unsigned int a3 = w4a4_smem_a8(sA, lm + r + 8, 32 + q * 8);
            const unsigned int sfa = w4a4_smem_sf4(sAs, lm + sfa_m);
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const int ln = wn * 8 + j * 32;       // 0..31 + {0,32,64,96}
                const unsigned int b0 = w4a4_smem_a8(sB, ln + r,     q * 8);
                const unsigned int b1 = w4a4_smem_a8(sB, ln + r, 32 + q * 8);
                const unsigned int sfb = w4a4_smem_sf4(sBs, ln + sfb_n);
#if (__CUDA_ARCH__ >= 1200)
                unsigned short bidA = 0, tidA = 0, bidB = 0, tidB = 0;
                asm volatile(
                    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                    "{%0,  %1,  %2,  %3},"
                    "{%4,  %5,  %6,  %7},"
                    "{%8,  %9},"
                    "{%10, %11, %12, %13},"
                    "{%14},"
                    "{%15, %16},"
                    "{%17},"
                    "{%18, %19};\n"
                    : "=f"(acc[i][j][0]), "=f"(acc[i][j][1]), "=f"(acc[i][j][2]), "=f"(acc[i][j][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                      "r"(b0), "r"(b1),
                      "f"(acc[i][j][0]), "f"(acc[i][j][1]), "f"(acc[i][j][2]), "f"(acc[i][j][3]),
                      "r"(sfa), "h"(bidA), "h"(tidA),
                      "r"(sfb), "h"(bidB), "h"(tidB));
#endif
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        const int crow0 = cta_m + wm * 16 + i * 32 + r;
        const int crow1 = crow0 + 8;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const int ccol0 = cta_n + wn * 8 + j * 32 + 2 * q;
            if (crow0 < m) {
                if (ccol0     < n) out[(unsigned long long)crow0 * n + ccol0]     = __float2bfloat16(acc[i][j][0]);
                if (ccol0 + 1 < n) out[(unsigned long long)crow0 * n + ccol0 + 1] = __float2bfloat16(acc[i][j][1]);
            }
            if (crow1 < m) {
                if (ccol0     < n) out[(unsigned long long)crow1 * n + ccol0]     = __float2bfloat16(acc[i][j][2]);
                if (ccol0 + 1 < n) out[(unsigned long long)crow1 * n + ccol0 + 1] = __float2bfloat16(acc[i][j][3]);
            }
        }
    }
}


// ── PIPELINED W4A4 GEMM (2-stage cp.async double buffer) ────────────────────
// The tiled kernel above is 11.5x the naive one but still 2.27x off CUTLASS
// (1.392 vs 0.614 ms/iter at M=2048,N=12288,K=2048, like-for-like incl. pack).
// Its remaining stall is structural: one __syncthreads-bounded global load per
// K-step with nothing overlapping it. This version prefetches step i+1 into the
// alternate smem buffer with cp.async while the MMAs for step i run, which is
// the same 2-stage structure w4a16_gemm_t already uses.
//
// Same MMA, same fragment layout, same epilogue as the tiled kernel, so the
// cos=1.000000 correctness carries over unchanged.
__device__ __forceinline__ void w4a4_cp16(void* dst_smem, const void* src, bool pred) {
    unsigned int d = __cvta_generic_to_shared(dst_smem);
    unsigned int nbytes = pred ? 16u : 0u;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" :: "r"(d), "l"(src), "r"(nbytes));
}
__device__ __forceinline__ void w4a4_cp_commit() { asm volatile("cp.async.commit_group;"); }
__device__ __forceinline__ void w4a4_cp_wait0() { asm volatile("cp.async.wait_group 0;"); }

extern "C" __global__ void __launch_bounds__(256, 1)
w4a4_gemm_t_pipe(
    const unsigned char* __restrict__ packed_a,
    const unsigned char* __restrict__ scales_a,
    const unsigned char* __restrict__ packed_b,
    const unsigned char* __restrict__ scales_b,
    __nv_bfloat16* __restrict__ out,
    int m, int n, int k) {
    __shared__ unsigned char sA[2][W4A4_M_CTA * W4A4_KB];
    __shared__ unsigned char sAs[2][W4A4_M_CTA * W4A4_KG];
    __shared__ unsigned char sB[2][W4A4_N_CTA * W4A4_KB];
    __shared__ unsigned char sBs[2][W4A4_N_CTA * W4A4_KG];

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int cta_m = blockIdx.y * W4A4_M_CTA;
    const int cta_n = blockIdx.x * W4A4_N_CTA;
    const int q = lane & 3, r = lane >> 2;
    const int sfa_m = (lane & 1) * 8 + (lane >> 2);
    const int sfb_n = lane >> 2;
    const int wm = warp >> 2, wn = warp & 3;
    const int groups = k / W4A4_GROUP;

    // Staging coordinates, fixed across steps.
    // cp.async.16 requires BOTH operands 16-byte aligned, so every thread must
    // own a 16-byte-aligned 16-byte span. A row is W4A4_KB=32 B => 2 threads per
    // row. An earlier revision used 4 threads/row (8 B each, offsets {0,8,16,24})
    // with a 16-byte copy: misaligned AND overlapping, which faults the whole
    // context with CUDA_ERROR_MISALIGNED_ADDRESS.
    const int arow = tid >> 1, acol = (tid & 1) << 4;      // threads 0..127 -> A
    const int brow = tid >> 1, bcol = (tid & 1) << 4;      // threads 0..255 -> B
    const int agm = cta_m + arow, bgn = cta_n + brow;

    #define W4A4_STAGE(buf, kk)                                                              \
        do {                                                                                 \
            const int _k = (kk);                                                             \
            if (tid < W4A4_M_CTA * 2)                                                        \
                w4a4_cp16(sA[buf] + arow * W4A4_KB + acol,                                    \
                          packed_a + (unsigned long long)agm * (k / 2) + (_k >> 1) + acol,    \
                          agm < m);                                                          \
            w4a4_cp16(sB[buf] + brow * W4A4_KB + bcol,                                        \
                      packed_b + (unsigned long long)bgn * (k / 2) + (_k >> 1) + bcol,        \
                      bgn < n);                                                               \
            if (tid < W4A4_M_CTA)                                                             \
                *reinterpret_cast<unsigned int*>(sAs[buf] + tid * W4A4_KG) =                  \
                    (cta_m + tid < m) ? *reinterpret_cast<const unsigned int*>(               \
                        scales_a + (unsigned long long)(cta_m + tid) * groups                 \
                        + (_k / W4A4_GROUP)) : 0u;                                            \
            if (tid < W4A4_N_CTA)                                                             \
                *reinterpret_cast<unsigned int*>(sBs[buf] + tid * W4A4_KG) =                  \
                    (cta_n + tid < n) ? *reinterpret_cast<const unsigned int*>(               \
                        scales_b + (unsigned long long)(cta_n + tid) * groups                 \
                        + (_k / W4A4_GROUP)) : 0u;                                            \
            w4a4_cp_commit();                                                                 \
        } while (0)

    float acc[2][4][4];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            #pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.0f;

    W4A4_STAGE(0, 0);
    int buf = 0;
    for (int k0 = 0; k0 < k; k0 += W4A4_K_STEP) {
        w4a4_cp_wait0();
        __syncthreads();
        const int nxt = k0 + W4A4_K_STEP;
        if (nxt < k) W4A4_STAGE(buf ^ 1, nxt);

        const unsigned char* pA  = sA[buf];
        const unsigned char* pAs = sAs[buf];
        const unsigned char* pB  = sB[buf];
        const unsigned char* pBs = sBs[buf];
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            const int lm = wm * 16 + i * 32;
            const unsigned int a0 = w4a4_smem_a8(pA, lm + r,     q * 8);
            const unsigned int a1 = w4a4_smem_a8(pA, lm + r + 8, q * 8);
            const unsigned int a2 = w4a4_smem_a8(pA, lm + r,     32 + q * 8);
            const unsigned int a3 = w4a4_smem_a8(pA, lm + r + 8, 32 + q * 8);
            const unsigned int sfa = w4a4_smem_sf4(pAs, lm + sfa_m);
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const int ln = wn * 8 + j * 32;
                const unsigned int b0 = w4a4_smem_a8(pB, ln + r,     q * 8);
                const unsigned int b1 = w4a4_smem_a8(pB, ln + r, 32 + q * 8);
                const unsigned int sfb = w4a4_smem_sf4(pBs, ln + sfb_n);
#if (__CUDA_ARCH__ >= 1200)
                unsigned short bidA = 0, tidA = 0, bidB = 0, tidB = 0;
                asm volatile(
                    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                    "{%0,  %1,  %2,  %3},{%4,  %5,  %6,  %7},{%8,  %9},"
                    "{%10, %11, %12, %13},{%14},{%15, %16},{%17},{%18, %19};\n"
                    : "=f"(acc[i][j][0]), "=f"(acc[i][j][1]), "=f"(acc[i][j][2]), "=f"(acc[i][j][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                      "f"(acc[i][j][0]), "f"(acc[i][j][1]), "f"(acc[i][j][2]), "f"(acc[i][j][3]),
                      "r"(sfa), "h"(bidA), "h"(tidA), "r"(sfb), "h"(bidB), "h"(tidB));
#endif
            }
        }
        buf ^= 1;
    }
    #undef W4A4_STAGE

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        const int crow0 = cta_m + wm * 16 + i * 32 + r, crow1 = crow0 + 8;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const int ccol0 = cta_n + wn * 8 + j * 32 + 2 * q;
            if (crow0 < m) {
                if (ccol0     < n) out[(unsigned long long)crow0 * n + ccol0]     = __float2bfloat16(acc[i][j][0]);
                if (ccol0 + 1 < n) out[(unsigned long long)crow0 * n + ccol0 + 1] = __float2bfloat16(acc[i][j][1]);
            }
            if (crow1 < m) {
                if (ccol0     < n) out[(unsigned long long)crow1 * n + ccol0]     = __float2bfloat16(acc[i][j][2]);
                if (ccol0 + 1 < n) out[(unsigned long long)crow1 * n + ccol0 + 1] = __float2bfloat16(acc[i][j][3]);
            }
        }
    }
}


// ── W4A4 GEMM, 128x128 CTA tile ────────────────────────────────────────────
// The 64x128 tiled kernel measured 1.389 ms/iter at M=2048,N=12288,K=2048 vs
// CUTLASS 0.643. Its traffic is 3072 CTAs x (64 KB A + 128 KB B) = 590 MB;
// doubling the M extent halves the CTA count and the A re-reads:
//   1536 CTAs x (128 KB A + 128 KB B) = 393 MB.
// Each warp now owns 4 M-positions x 4 N-positions = 16 output tiles (64 acc
// registers). Same MMA, same fragment layout, same epilogue.
#define W4A4_M_CTA2 128

extern "C" __global__ void __launch_bounds__(256, 1)
w4a4_gemm_t_big(
    const unsigned char* __restrict__ packed_a,
    const unsigned char* __restrict__ scales_a,
    const unsigned char* __restrict__ packed_b,
    const unsigned char* __restrict__ scales_b,
    __nv_bfloat16* __restrict__ out,
    int m, int n, int k) {
    __shared__ unsigned char sA[W4A4_M_CTA2 * W4A4_KBP];
    __shared__ unsigned char sAs[W4A4_M_CTA2 * W4A4_KG];
    __shared__ unsigned char sB[W4A4_N_CTA * W4A4_KBP];
    __shared__ unsigned char sBs[W4A4_N_CTA * W4A4_KG];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int cta_m = blockIdx.y * W4A4_M_CTA2;
    const int cta_n = blockIdx.x * W4A4_N_CTA;
    const int q = lane & 3, r = lane >> 2;
    const int sfa_m = (lane & 1) * 8 + (lane >> 2);
    const int sfb_n = lane >> 2;
    const int wm = warp >> 2, wn = warp & 3;
    const int groups = k / W4A4_GROUP;
    const int row2 = tid >> 1, col2 = (tid & 1) << 4;   // 16 B/thread, 16 B aligned

    float acc[4][4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            #pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.0f;

    for (int k0 = 0; k0 < k; k0 += W4A4_K_STEP) {
        __syncthreads();
        // A: 128 rows x 32 B = 4096 B -> 16 B per thread across 256 threads.
        {
            const int gm = cta_m + row2;
            unsigned char* dst = sA + row2 * W4A4_KBP + col2;
            if (gm < m)
                *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(
                    packed_a + (unsigned long long)gm * (k / 2) + (k0 >> 1) + col2);
            else *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
        }
        // B: 128 rows x 32 B = 4096 B -> same shape.
        {
            const int gn = cta_n + row2;
            unsigned char* dst = sB + row2 * W4A4_KBP + col2;
            if (gn < n)
                *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(
                    packed_b + (unsigned long long)gn * (k / 2) + (k0 >> 1) + col2);
            else *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
        }
        if (tid < W4A4_M_CTA2) {
            const int gm = cta_m + tid;
            *reinterpret_cast<unsigned int*>(sAs + tid * W4A4_KG) =
                (gm < m) ? *reinterpret_cast<const unsigned int*>(
                    scales_a + (unsigned long long)gm * groups + (k0 / W4A4_GROUP)) : 0u;
        }
        if (tid < W4A4_N_CTA) {
            const int gn = cta_n + tid;
            *reinterpret_cast<unsigned int*>(sBs + tid * W4A4_KG) =
                (gn < n) ? *reinterpret_cast<const unsigned int*>(
                    scales_b + (unsigned long long)gn * groups + (k0 / W4A4_GROUP)) : 0u;
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int lm = wm * 16 + i * 32;      // 0,32,64,96 / 16,48,80,112
            const unsigned int a0 = w4a4_smem_a8p(sA, lm + r,     q * 8);
            const unsigned int a1 = w4a4_smem_a8p(sA, lm + r + 8, q * 8);
            const unsigned int a2 = w4a4_smem_a8p(sA, lm + r,     32 + q * 8);
            const unsigned int a3 = w4a4_smem_a8p(sA, lm + r + 8, 32 + q * 8);
            const unsigned int sfa = w4a4_smem_sf4(sAs, lm + sfa_m);
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const int ln = wn * 8 + j * 32;
                const unsigned int b0 = w4a4_smem_a8p(sB, ln + r,     q * 8);
                const unsigned int b1 = w4a4_smem_a8p(sB, ln + r, 32 + q * 8);
                const unsigned int sfb = w4a4_smem_sf4(sBs, ln + sfb_n);
#if (__CUDA_ARCH__ >= 1200)
                unsigned short bidA = 0, tidA = 0, bidB = 0, tidB = 0;
                asm volatile(
                    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                    "{%0,  %1,  %2,  %3},{%4,  %5,  %6,  %7},{%8,  %9},"
                    "{%10, %11, %12, %13},{%14},{%15, %16},{%17},{%18, %19};\n"
                    : "=f"(acc[i][j][0]), "=f"(acc[i][j][1]), "=f"(acc[i][j][2]), "=f"(acc[i][j][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                      "f"(acc[i][j][0]), "f"(acc[i][j][1]), "f"(acc[i][j][2]), "f"(acc[i][j][3]),
                      "r"(sfa), "h"(bidA), "h"(tidA), "r"(sfb), "h"(bidB), "h"(tidB));
#endif
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const int crow0 = cta_m + wm * 16 + i * 32 + r, crow1 = crow0 + 8;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const int ccol0 = cta_n + wn * 8 + j * 32 + 2 * q;
            if (crow0 < m) {
                if (ccol0     < n) out[(unsigned long long)crow0 * n + ccol0]     = __float2bfloat16(acc[i][j][0]);
                if (ccol0 + 1 < n) out[(unsigned long long)crow0 * n + ccol0 + 1] = __float2bfloat16(acc[i][j][1]);
            }
            if (crow1 < m) {
                if (ccol0     < n) out[(unsigned long long)crow1 * n + ccol0]     = __float2bfloat16(acc[i][j][2]);
                if (ccol0 + 1 < n) out[(unsigned long long)crow1 * n + ccol0 + 1] = __float2bfloat16(acc[i][j][3]);
            }
        }
    }
}


// ── W4A4 GEMM, 256x128 CTA tile ────────────────────────────────────────────
// Traffic accounting at M=2048,N=12288,K=2048 (A=2 MB, B=12 MB packed):
//   traffic = (N/N_CTA)*A + (M/M_CTA)*B
//   64x128 -> 96*2 + 32*12 = 576 MB   measured 1.353 ms
//  128x128 -> 96*2 + 16*12 = 384 MB   measured 1.143 ms
//  256x128 -> 96*2 +  8*12 = 288 MB   <- this kernel
// Extending M (not N) is what cuts the expensive term, because B is 6x larger
// than A here. Each warp owns 8 M-positions x 4 N-positions = 32 tiles.
#define W4A4_M_CTA3 256

extern "C" __global__ void __launch_bounds__(256, 1)
w4a4_gemm_t_huge(
    const unsigned char* __restrict__ packed_a,
    const unsigned char* __restrict__ scales_a,
    const unsigned char* __restrict__ packed_b,
    const unsigned char* __restrict__ scales_b,
    __nv_bfloat16* __restrict__ out,
    int m, int n, int k) {
    __shared__ unsigned char sA[W4A4_M_CTA3 * W4A4_KB];
    __shared__ unsigned char sAs[W4A4_M_CTA3 * W4A4_KG];
    __shared__ unsigned char sB[W4A4_N_CTA * W4A4_KB];
    __shared__ unsigned char sBs[W4A4_N_CTA * W4A4_KG];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int cta_m = blockIdx.y * W4A4_M_CTA3;
    const int cta_n = blockIdx.x * W4A4_N_CTA;
    const int q = lane & 3, r = lane >> 2;
    const int sfa_m = (lane & 1) * 8 + (lane >> 2);
    const int sfb_n = lane >> 2;
    const int wm = warp >> 2, wn = warp & 3;
    const int groups = k / W4A4_GROUP;
    const int row2 = tid >> 1, col2 = (tid & 1) << 4;

    float acc[8][4][4];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++)
            #pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.0f;

    for (int k0 = 0; k0 < k; k0 += W4A4_K_STEP) {
        __syncthreads();
        // A: 256 rows x 32 B = 8192 B -> two 16 B stores per thread.
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            const int lr = row2 + h * 128;
            const int gm = cta_m + lr;
            unsigned char* dst = sA + lr * W4A4_KB + col2;
            if (gm < m)
                *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(
                    packed_a + (unsigned long long)gm * (k / 2) + (k0 >> 1) + col2);
            else *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
        }
        {
            const int gn = cta_n + row2;
            unsigned char* dst = sB + row2 * W4A4_KB + col2;
            if (gn < n)
                *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(
                    packed_b + (unsigned long long)gn * (k / 2) + (k0 >> 1) + col2);
            else *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
        }
        #pragma unroll
        for (int h = 0; h < 1; h++) {
            const int gm = cta_m + tid;
            *reinterpret_cast<unsigned int*>(sAs + tid * W4A4_KG) =
                (gm < m) ? *reinterpret_cast<const unsigned int*>(
                    scales_a + (unsigned long long)gm * groups + (k0 / W4A4_GROUP)) : 0u;
        }
        if (tid < W4A4_N_CTA) {
            const int gn = cta_n + tid;
            *reinterpret_cast<unsigned int*>(sBs + tid * W4A4_KG) =
                (gn < n) ? *reinterpret_cast<const unsigned int*>(
                    scales_b + (unsigned long long)gn * groups + (k0 / W4A4_GROUP)) : 0u;
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int lm = wm * 16 + i * 32;
            const unsigned int a0 = w4a4_smem_a8(sA, lm + r,     q * 8);
            const unsigned int a1 = w4a4_smem_a8(sA, lm + r + 8, q * 8);
            const unsigned int a2 = w4a4_smem_a8(sA, lm + r,     32 + q * 8);
            const unsigned int a3 = w4a4_smem_a8(sA, lm + r + 8, 32 + q * 8);
            const unsigned int sfa = w4a4_smem_sf4(sAs, lm + sfa_m);
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const int ln = wn * 8 + j * 32;
                const unsigned int b0 = w4a4_smem_a8(sB, ln + r,     q * 8);
                const unsigned int b1 = w4a4_smem_a8(sB, ln + r, 32 + q * 8);
                const unsigned int sfb = w4a4_smem_sf4(sBs, ln + sfb_n);
#if (__CUDA_ARCH__ >= 1200)
                unsigned short bidA = 0, tidA = 0, bidB = 0, tidB = 0;
                asm volatile(
                    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                    "{%0,  %1,  %2,  %3},{%4,  %5,  %6,  %7},{%8,  %9},"
                    "{%10, %11, %12, %13},{%14},{%15, %16},{%17},{%18, %19};\n"
                    : "=f"(acc[i][j][0]), "=f"(acc[i][j][1]), "=f"(acc[i][j][2]), "=f"(acc[i][j][3])
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                      "f"(acc[i][j][0]), "f"(acc[i][j][1]), "f"(acc[i][j][2]), "f"(acc[i][j][3]),
                      "r"(sfa), "h"(bidA), "h"(tidA), "r"(sfb), "h"(bidB), "h"(tidB));
#endif
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const int crow0 = cta_m + wm * 16 + i * 32 + r, crow1 = crow0 + 8;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const int ccol0 = cta_n + wn * 8 + j * 32 + 2 * q;
            if (crow0 < m) {
                if (ccol0     < n) out[(unsigned long long)crow0 * n + ccol0]     = __float2bfloat16(acc[i][j][0]);
                if (ccol0 + 1 < n) out[(unsigned long long)crow0 * n + ccol0 + 1] = __float2bfloat16(acc[i][j][1]);
            }
            if (crow1 < m) {
                if (ccol0     < n) out[(unsigned long long)crow1 * n + ccol0]     = __float2bfloat16(acc[i][j][2]);
                if (ccol0 + 1 < n) out[(unsigned long long)crow1 * n + ccol0 + 1] = __float2bfloat16(acc[i][j][3]);
            }
        }
    }
}
