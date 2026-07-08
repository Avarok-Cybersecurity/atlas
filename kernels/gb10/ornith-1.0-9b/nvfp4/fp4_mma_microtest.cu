// SPDX-License-Identifier: AGPL-3.0-only

// Hand-rolled Sm120 block-scaled FP4 (e2m1) MMA microproof.
//
// Goal: prove a single hand-written
//   mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
// matches the CUTLASS Sm120 block-scaled FP4 collective
// (nvfp4_gemm_bf16_act_weight_t) at cos >= 0.999. NO cp.async, NO grouping, NO
// gate+up fusion — the SIMPLEST possible single-tile-looped [M,N,K] GEMM that
// exercises the MMA + the SFA/SFB scale operand layout correctly.
//
// Operand / scale layout (extracted from ~/cutlass mma_sm120.hpp +
// mma_traits_sm120.hpp; see report). Per lane t in a 32-thread warp, with
//   q = t % 4   (K-octet group within a 32-wide half)
//   r = t / 4   (row index 0..7)
// for a 16x8x64 tile and k0 = current K base:
//   A frag (4 regs, 8 e2m1 each, low-nibble = lower k):
//     a0: m=r   , k = k0 +      q*8 + (0..7)
//     a1: m=r+8 , k = k0 +      q*8 + (0..7)
//     a2: m=r   , k = k0 + 32 + q*8 + (0..7)
//     a3: m=r+8 , k = k0 + 32 + q*8 + (0..7)
//   B frag (2 regs, 8 e2m1 each):
//     b0: n=r   , k = k0 +      q*8 + (0..7)
//     b1: n=r   , k = k0 + 32 + q*8 + (0..7)
//   SFA (1 u32 = 4 ue4m3 bytes, byte j = scale of k-group (k0/16 + j)):
//     row m = (t%2)*8 + (t/4)
//   SFB (1 u32 = 4 ue4m3 bytes, byte j = scale of k-group (k0/16 + j)):
//     col n = t/4
//
// Quantization is done in a SEPARATE pre-pass (fp4_microtest_pack) that mirrors
// atlas_cutlass_pack_bf16_act_nvfp4 EXACTLY (per-16-group scale = max_abs/6 as
// ue4m3, e2m1 round-to-nearest) into NATURAL-layout buffers:
//   packed[rows][K/2]   (low nibble = even k)
//   scales[rows][K/16]  (raw ue4m3 byte per group)
// so the MMA kernel only has to gather the right nibbles/scales — isolating MMA
// correctness from the quant math.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

// ───────────────────────── e2m1 / ue4m3 helpers (mirror the cutlass pack) ──
__device__ __forceinline__ unsigned char mt_float_to_e2m1(float x) {
    unsigned char sign = (x < 0.0f) ? 8u : 0u;
    float ax = fabsf(x);
    unsigned char mag;
    if (ax <= 0.25f)      mag = 0;
    else if (ax <= 0.75f) mag = 1;
    else if (ax <= 1.25f) mag = 2;
    else if (ax <= 1.75f) mag = 3;
    else if (ax <= 2.5f)  mag = 4;
    else if (ax <= 3.5f)  mag = 5;
    else if (ax <= 5.0f)  mag = 6;
    else                  mag = 7;
    return sign | mag;
}

// Encode a positive float scale into ue4m3 (unsigned e4m3, no sign bit, bias 7).
// Matches cutlass::float_ue4m3_t(scale) construction (round-to-nearest-even,
// saturating). We round-trip through __nv_fp8_e4m3 with the sign forced off:
// ue4m3 shares the e4m3 exponent/mantissa field semantics but uses all 8 bits
// for magnitude. CUTLASS's ue4m3 = e4m3 magnitude (bits [6:0] are m+e, bit7
// would be the e4m3 sign which ue4m3 repurposes). The simplest faithful
// encoder: build the byte via the standard e4m3 of |scale| then it already has
// sign=0; for ue4m3 the stored byte is the low 8 bits == same magnitude byte.
__device__ __forceinline__ unsigned char mt_float_to_ue4m3(float scale) {
    // ue4m3: 4-bit exp (bias 7), 3-bit mantissa, NO sign — same field layout as
    // the lower 7 bits of e4m3 plus one extra exponent... NO: ue4m3 is E4M3
    // *interpreted unsigned*: 8 bits = [4 exp][3 mant] would be only 7 bits.
    // CUTLASS float_ue4m3_t is an 8-bit type: 1 unused/0 sign? In practice the
    // SF byte stored by the cutlass pack is byte_of(float_ue4m3_t(scale)).
    // Empirically float_ue4m3_t == cutlass e4m3 magnitude in low 8 bits; encode
    // scale (always >= 0) via __nv_fp8_e4m3 and take the byte (sign bit 0).
    __nv_fp8_e4m3 v = __nv_fp8_e4m3(scale);
    unsigned char b = *reinterpret_cast<unsigned char*>(&v);
    return b; // scale>=0 => sign bit clear
}

__device__ __forceinline__ float mt_ue4m3_to_float(unsigned char byte) {
    __nv_fp8_e4m3 v;
    *reinterpret_cast<unsigned char*>(&v) = byte;
    return static_cast<float>(v);
}

// ───────────────────────── pack pre-pass ─────────────────────────
// One thread per (row, group-of-16). Reads bf16 src[rows][K], writes
// packed[rows][K/2] (e2m1 nibbles) + scales[rows][K/16] (ue4m3 byte).
extern "C" __global__ void fp4_microtest_pack(
    const __nv_bfloat16* __restrict__ src,
    unsigned char* __restrict__ packed,
    unsigned char* __restrict__ scales,
    int rows,
    int k) {
    int row = blockIdx.x;
    int group = blockIdx.y * blockDim.x + threadIdx.x;
    int groups = k / 16;
    if (row >= rows || group >= groups) return;

    int base = group * 16;
    float max_abs = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        float v = __bfloat162float(src[(unsigned long long)row * k + base + i]);
        max_abs = fmaxf(max_abs, fabsf(v));
    }
    float scale = max_abs > 0.0f ? max_abs / 6.0f : 1.0f;
    unsigned char sf = mt_float_to_ue4m3(scale);
    scales[(unsigned long long)row * groups + group] = sf;
    float decoded = mt_ue4m3_to_float(sf);
    float inv = decoded > 0.0f ? 1.0f / decoded : 0.0f;

#pragma unroll
    for (int i = 0; i < 16; i += 2) {
        float v0 = __bfloat162float(src[(unsigned long long)row * k + base + i])     * inv;
        float v1 = __bfloat162float(src[(unsigned long long)row * k + base + i + 1]) * inv;
        packed[(unsigned long long)row * (k / 2) + base / 2 + i / 2] =
            (unsigned char)(mt_float_to_e2m1(v0) | (mt_float_to_e2m1(v1) << 4));
    }
}

// ───────────────────────── helpers to gather fragments ─────────────────────
// Read 8 consecutive e2m1 from packed[row][kk..kk+7] (kk even) into a u32:
// nibble j (j=0..7) is element kk+j. Byte (kk+j)/2 low/high.
__device__ __forceinline__ unsigned int gather_a8(
    const unsigned char* __restrict__ packed, int row, int kk, int k) {
    const unsigned char* p = packed + (unsigned long long)row * (k / 2) + kk / 2;
    unsigned int r = 0;
#pragma unroll
    for (int j = 0; j < 8; j += 2) {
        unsigned char byte = p[j / 2];
        unsigned int lo = byte & 0xF;
        unsigned int hi = (byte >> 4) & 0xF;
        r |= lo << (4 * j);
        r |= hi << (4 * (j + 1));
    }
    return r;
}

// Pack 4 ue4m3 scale bytes (k-groups g0..g0+3 of `row`) into a u32 (byte j = group j).
__device__ __forceinline__ unsigned int gather_sf4(
    const unsigned char* __restrict__ scales, int row, int g0, int k) {
    int groups = k / 16;
    const unsigned char* p = scales + (unsigned long long)row * groups + g0;
    unsigned int r = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) r |= ((unsigned int)p[j]) << (8 * j);
    return r;
}

// ───────────────────────── the MMA kernel ─────────────────────────
// One warp per output tile. grid = (N/8, M/16). block = 32.
// A: packed_a[M][K/2], scales_a[M][K/16].  B: packed_b[N][K/2], scales_b[N][K/16].
// C: bf16[M][N], row-major.  out[m,n] = sum_k a[m,k]*b[n,k].
extern "C" __global__ void fp4_microtest_mma(
    const unsigned char* __restrict__ packed_a,
    const unsigned char* __restrict__ scales_a,
    const unsigned char* __restrict__ packed_b,
    const unsigned char* __restrict__ scales_b,
    __nv_bfloat16* __restrict__ out,
    int m, int n, int k) {
    int tile_n = blockIdx.x * 8;   // base N col
    int tile_m = blockIdx.y * 16;  // base M row
    int t = threadIdx.x;           // 0..31
    int q = t & 3;                 // 0..3
    int r = t >> 2;                // 0..7

    float acc[4] = {0.f, 0.f, 0.f, 0.f};

    // SFA row/col owners
    int sfa_m = (t & 1) * 8 + (t >> 2);   // (t%2)*8 + (t/4)
    int sfb_n = t >> 2;                   // t/4

    for (int k0 = 0; k0 < k; k0 += 64) {
        // A fragment
        unsigned int a0 = gather_a8(packed_a, tile_m + r,     k0 +      q * 8, k);
        unsigned int a1 = gather_a8(packed_a, tile_m + r + 8, k0 +      q * 8, k);
        unsigned int a2 = gather_a8(packed_a, tile_m + r,     k0 + 32 + q * 8, k);
        unsigned int a3 = gather_a8(packed_a, tile_m + r + 8, k0 + 32 + q * 8, k);
        // B fragment
        unsigned int b0 = gather_a8(packed_b, tile_n + r,     k0 +      q * 8, k);
        unsigned int b1 = gather_a8(packed_b, tile_n + r,     k0 + 32 + q * 8, k);
        // Scales: 4 k-groups starting at k0/16
        unsigned int sfa = gather_sf4(scales_a, tile_m + sfa_m, k0 / 16, k);
        unsigned int sfb = gather_sf4(scales_b, tile_n + sfb_n, k0 / 16, k);

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

    // C layout SM80_16x8_Row: thread t holds acc for rows {t/4, t/4+8}, cols {2*(t%4), 2*(t%4)+1}
    // acc[0],acc[1] -> row t/4 ; acc[2],acc[3] -> row t/4+8 ; col pair = 2*(t%4)+{0,1}
    int crow0 = tile_m + (t >> 2);
    int crow1 = crow0 + 8;
    int ccol0 = tile_n + 2 * (t & 3);
    int ccol1 = ccol0 + 1;
    if (crow0 < m) {
        if (ccol0 < n) out[(unsigned long long)crow0 * n + ccol0] = __float2bfloat16(acc[0]);
        if (ccol1 < n) out[(unsigned long long)crow0 * n + ccol1] = __float2bfloat16(acc[1]);
    }
    if (crow1 < m) {
        if (ccol0 < n) out[(unsigned long long)crow1 * n + ccol0] = __float2bfloat16(acc[2]);
        if (ccol1 < n) out[(unsigned long long)crow1 * n + ccol1] = __float2bfloat16(acc[3]);
    }
}
