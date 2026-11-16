// SPDX-License-Identifier: AGPL-3.0-only AND Apache-2.0
// Portions copied and adapted from TensorFold (https://github.com/ashhart/TensorFold), commit 3885dce,
// zig/kernels/cuda/qmm_prefill.cu:1-301, zig/kernels/cuda/w4a8_prefill.cu:1-172,
// zig/kernels/cuda/w4a8_store.cuh:1-20, zig/kernels/cuda/common/qmm_frag.cuh:1-174 (the primitives used here) and
// zig/kernels/cuda/common/cp_async.cuh:1-67 (the primitives used here).
// Copyright 2026 TensorFold contributors. Licensed under the Apache License, Version 2.0; code written
// before TensorFold 0.6.0 is also available under the MIT License. See TENSORFOLD-ATTRIBUTION.md.
// Changes for Atlas: namespaces renamed (avarok_affine_prefill); the used cp.async, ldmatrix, mma, pair, swizzle
// and tile_of primitives inlined from qmm_frag.cuh and cp_async.cuh, with the sm_90 `sub.rn.bf16x2` branch of
// `pair` dropped (this file builds for sm_89 only); `w4a8_store.cuh` inlined as `store2`; the BF16 body shared by
// the single and the many entry (upstream keeps a literal second copy for Blackwell SASS reasons that do not apply
// to sm_89); the int8 body takes its block index as an argument so it serves a new `_many` entry; tiles live in
// the dynamic shared memory of `prefill_dyn_smem.cuh` and every entry exports its launch size; extern "C" entry
// names; a device-side exceptional gate on every entry (the int8 entries skip on a raised flag, the BF16 entries
// can run only on a raised flag and count the fallback), replacing the host readback of upstream w4a8.zig.

// Ada (sm_89) prompt GEMMs (M > 16 rows) over MLX affine 4-bit weights in TensorFold's tiled layout, the prefill
// half of the affine route (the decode half is affine4_lane_ada.cu).
//
// ── WEIGHT LAYOUT (written by affine_pack_ada.cu, L17a) ─────────────────────────────────────────────────────
//
//   words   u32 [npad/64][K/64][8][32][2]   npad = N rounded up to 128; per (64-column tile, 64-input group):
//           8 n8 column slices x 32 lanes x 2 words. Lane l of slice j holds column t*64 + j*8 + l/4; with
//           c = l%4, word v nibble p is the code of input g*64 + 32v + 2c + {0,8,16,24,1,9,17,25}[p].
//   scales  bf16 [K/64][npad]   (columns past N are zero, so padded columns produce zeros)
//   biases  bf16 [K/64][npad]
//   value of a weight = s * q + b, q in 0..15 (MLX affine, group 64 only: K % 64 == 0)
//
// ── BF16 PROMPT GEMM: affine4_prefill_bf16_ada, affine4_prefill_bf16_many_ada ───────────────────────────────
//
//   out[m, n] = sum_k x[m, k] * bf16(s * q + b)        x: [M, K] BF16 with row stride ldx, out: [M, N] BF16
//
//   The dequantized operand is formed per pair by one `fma.rn.bf16x2` (q * s + b, one rounding to BF16) and fed
//   to `mma.sync.m16n8k16` with FP32 accumulation. 128 x 128 CTA tile, 2 x 4 warps (256 threads), 3 cp.async
//   stages of 20,992 bytes (62,976 dynamic bytes, one CTA an SM on Ada). Grid: rows_t * cols_t CTAs in row bands
//   of `group` row tiles (L2 reuse of the weight tiles; the host derives `group` from K).
//
// ── W4A8 INT8 PROMPT GEMM: affine4_w4a8_prefill_ada, affine4_w4a8_prefill_many_ada ─────────────────────────
//
//   out[m, n] = sum_g ( s * (a[m,g] * (q . xq[m, g])) + b * xs[m, g] )
//
//   xq, a, xs come from affine_swiglu_sums_ada.cu (int8 codes of x in mma slot order, scale amax/127 and the FP32
//   sum of the BF16 x, per row per 64-input group). The 4-bit codes are used as unsigned int8 bytes on
//   `mma.sync.m16n8k32.s32.s8.s8`, so the group's integer dot is EXACT. Its accumulators start at the bit pattern
//   of 12582912.0f (1.5 * 2^23): for |dot| < 2^22 (here |dot| <= 64 * 15 * 127 = 121,920) the int32 sum read as a
//   float is 12582912 + dot exactly, and one fma `fmaf(that, a, -12582912 * a)` removes the offset and applies the
//   activation scale. 128 x 128 tile, 2 x 4 warps, 4 stages of 13,824 bytes (55,296 dynamic bytes).
//
// ── NUMERICS ────────────────────────────────────────────────────────────────────────────────────────────────
//
//   * int8: every output is a deterministic function of (xq, a, xs, s, b, q): the integer dot is exact and the
//     epilogue is a fixed chain of explicit fmaf calls per group in group order, so the result is bit-exact to
//     the host model `port/affine/prefill_oracle.rs::int8_gemm` (the GPU microtest compares bytes). Its distance
//     to the exact affine product is bounded per group by the int8 rounding of x (|x - a*xq| <= a/2 per input,
//     times |s| q), the offset rounding (at most one unit of a), the FP32 group sum and the accumulation; the
//     oracle's `int8_bound` writes the bound out term by term.
//   * BF16: within |want| * 2^-8 + gamma(K) * sum |x * w| of the float64 product over the BF16-rounded weights.
//   * Row invariance: a row's output depends only on that row's inputs and the weights, never on M (same group
//     order, same per-element epilogue), so prefill rows are bit-equal at every M.
//   * `--fmad=false` (common/KERNEL.toml) does not touch the explicit fmaf / fma.rn.bf16x2 used here.
//
// ── EXCEPTIONAL GATE (no host sync) ─────────────────────────────────────────────────────────────────────────
//
//   flags[0] is raised by the activation quantizer when a group holds a non-finite value or its FP32 sum
//   overflows. The int8 entries return at once when flags[0] != 0; the BF16 entries take the same pointer as
//   `run_if` and run only when it is raised (and add the number of projections they ran to flags[1], the sticky
//   fallback counter the host reads at its own sync points). The BF16 result therefore replaces the int8 one on
//   the device, in eager runs and inside a captured graph alike. A NULL `run_if` runs the BF16 entry
//   unconditionally (the plain BF16 route).
//
// Constraints the host checks (port/affine/prefill.rs): K % 64 == 0, ldx % 8 == 0, 16-byte aligned x, words,
// scales and biases; npad = N rounded up to 128; M, N, K < 2^31.

#include <cuda_bf16.h>
#include <stddef.h>
#include <stdint.h>

#include "prefill_dyn_smem.cuh"

namespace avarok_affine_prefill {

constexpr int GS = 64;     // inputs a group (the tiled layout's only group size)
constexpr int PARTS = 4;   // projections a `_many` grid

// One projection of a `_many` grid. Layout pinned by the host (port/affine/prefill.rs, PART_BYTES / PARTS_BYTES).
struct Part {
    const uint32_t* w;
    const __nv_bfloat16* scales;
    const __nv_bfloat16* biases;
    __nv_bfloat16* out;
    int n, npad, first;    // first = the projection's first CTA in the shared grid
};

struct Parts {
    Part p[PARTS];
    int count;
};

static_assert(sizeof(Part) == 48, "Part must match port/affine/prefill.rs PART_BYTES");
static_assert(offsetof(Parts, count) == 192, "Parts.count must sit where port/affine/prefill.rs writes it");
static_assert(sizeof(Parts) == 200, "Parts must match port/affine/prefill.rs PARTS_BYTES");

// ── primitives (qmm_frag.cuh, cp_async.cuh) ─────────────────────────────────────────────────────────────────

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr(dst)), "l"(src));
}

// 16 bytes, or 16 zero bytes when `read` is false (the source is not read).
__device__ __forceinline__ void cp16z(void* dst, const void* src, bool read) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_addr(dst)), "l"(src),
                 "r"(read ? 16 : 0));
}

__device__ __forceinline__ void cp4(void* dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(smem_addr(dst)), "l"(src));
}

__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void ldmatrix4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_addr(p)));
}

__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_s8(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Nibbles at s and s + 16 of w as the exact BF16 pair (q_lo, q_hi): (128 + q) * 1 - 128 in one fma.
__device__ __forceinline__ uint32_t pair(uint32_t w, int s) {
    const uint32_t t = ((w >> s) & 0x000F000Fu) | 0x43004300u;
    uint32_t r;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(t), "r"(0x3F803F80u), "r"(0xC300C300u));
    return r;
}

__device__ __forceinline__ uint32_t fma2(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// Nibbles at s, s + 16, s + 4, s + 20 of w as four int8 bytes in that order (prmt: a.b0, a.b2, b.b0, b.b2).
__device__ __forceinline__ uint32_t bytes4(uint32_t w, int s) {
    const uint32_t t0 = (w >> s) & 0x000F000Fu, t1 = (w >> (s + 4)) & 0x000F000Fu;
    uint32_t r;
    asm("prmt.b32 %0, %1, %2, 0x6420;\n" : "=r"(r) : "r"(t0), "r"(t1));
    return r;
}

// CTA b's (row, column) origin: row tiles walk inside bands of `group` row tiles, column tiles across a band.
__device__ __forceinline__ int2 tile_of(int b, int M, int N, int BM, int BN, int group) {
    const int rows_t = (M + BM - 1) / BM, cols_t = (N + BN - 1) / BN, band = group * cols_t;
    const int first = b / band * group, in_band = b % band, height = min(group, rows_t - first);
    return make_int2((first + in_band % height) * BM, in_band / height * BN);
}

template <int CHUNKS>
__device__ __forceinline__ int swz(int r, int c) {
    return c ^ (r % CHUNKS);
}

// w4a8_store.cuh: a BF16 pair, as one 4-byte store when aligned, else per component with the same rounding.
__device__ __forceinline__ void store2(__nv_bfloat16* dst, int remaining, float a, float b) {
    if (remaining > 1 && (reinterpret_cast<uintptr_t>(dst) & 3) == 0)
        *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(a, b);
    else {
        dst[0] = __float2bfloat16_rn(a);
        if (remaining > 1) dst[1] = __float2bfloat16_rn(b);
    }
}

// ── BF16 prompt GEMM body (qmm_prefill.cu `prefill_body`, bf16 out) ──────────────────────────────────────────

template <int BM, int BN, int WM, int WN, int STAGES>
struct Tile {
    static constexpr int THREADS = WM * WN * 32;
    static constexpr int MT = BM / WM / 16;   // m16 tiles a warp
    static constexpr int NT = BN / WN / 8;    // n8 tiles a warp
    static constexpr int ROW = GS * 2;        // bytes of one input row a group
    static constexpr int CHUNKS = ROW / 16;
    static constexpr int X = BM * ROW;        // stage bytes: inputs,
    static constexpr int W = BN * GS / 2;     // weights,
    static constexpr int S = BN * 2;          // scales and biases (bf16)
    static constexpr int STAGE = X + W + 2 * S;
    static constexpr int SMEM = STAGES * STAGE;
};

using BfTile = Tile<128, 128, 2, 4, 3>;
struct BfSmem {
    unsigned char bytes[BfTile::SMEM];
};
static_assert(BfTile::SMEM == 62976, "port/affine/prefill.rs BF16_SMEM_BYTES");

template <int BM, int BN, int WM, int WN, int STAGES>
__device__ __forceinline__ void bf16_body(const __nv_bfloat16* __restrict__ x, const uint32_t* __restrict__ w,
                                          const __nv_bfloat16* __restrict__ scales,
                                          const __nv_bfloat16* __restrict__ biases, __nv_bfloat16* __restrict__ out,
                                          int M, int N, int K, int npad, int ldx, int group, int block) {
    using T = Tile<BM, BN, WM, WN, STAGES>;
    unsigned char* buf = avarok_prefill_smem;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const int KG = K / GS;
    const int2 at = tile_of(block, M, N, BM, BN, group);
    const int m0 = at.x, n0 = at.y;

    auto stage = [&](int s) { return buf + s * T::STAGE; };
    auto load = [&](int s, int g) {
        unsigned char* p = stage(s);
        for (int c = tid; c < BM * T::CHUNKS; c += T::THREADS) {
            const int r = c / T::CHUNKS, ch = c % T::CHUNKS;
            const int row = min(m0 + r, M - 1);
            cp16z(p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16, x + static_cast<size_t>(row) * ldx + g * GS + ch * 8,
                  m0 + r < M);
        }
        unsigned char* pw = p + T::X;
        constexpr int TILE_BYTES = 64 * GS / 2;   // one stored 64-column tile's group block
        for (int c = tid; c < T::W / 16; c += T::THREADS) {
            const int t = c / (TILE_BYTES / 16), off = c % (TILE_BYTES / 16);
            const size_t tile = static_cast<size_t>(n0 / 64 + t) * KG + g;
            if (n0 + t * 64 < npad)
                cp16(pw + c * 16, reinterpret_cast<const unsigned char*>(w) + tile * TILE_BYTES + off * 16);
        }
        unsigned char* ps = pw + T::W;
        for (int c = tid; c < 2 * (T::S / 16); c += T::THREADS) {
            const int which = c / (T::S / 16), off = c % (T::S / 16);
            const __nv_bfloat16* src = (which ? biases : scales) + static_cast<size_t>(g) * npad + n0 + off * 8;
            if (n0 + off * 8 < npad) cp16(ps + which * T::S + off * 16, src);
        }
    };

    float acc[T::MT][T::NT][4];
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KG) load(s, s);
        commit();
    }
    for (int it = 0; it < KG; ++it) {
        wait<STAGES - 2>();
        __syncthreads();
        const int next = it + STAGES - 1;
        if (next < KG) load(next % STAGES, next);
        commit();
        const unsigned char* p = stage(it % STAGES);
        const uint32_t* pw = reinterpret_cast<const uint32_t*>(p + T::X);
        const uint16_t* ps = reinterpret_cast<const uint16_t*>(p + T::X + T::W);
        uint32_t words[T::NT][GS / 32], sv[T::NT], bv[T::NT];
#pragma unroll
        for (int j = 0; j < T::NT; ++j) {
#pragma unroll
            for (int v = 0; v < GS / 32; ++v) words[j][v] = pw[((wn * T::NT + j) * 32 + lane) * (GS / 32) + v];
            const int col = wn * (BN / WN) + j * 8 + (lane >> 2);   // this lane's B-fragment column
            sv[j] = ps[col] * 0x10001u;                             // (s, s) and (b, b) as bf16 pairs
            bv[j] = ps[BN + col] * 0x10001u;
        }
#pragma unroll
        for (int kt = 0; kt < GS / 16; ++kt) {
            uint32_t a[T::MT][4];
#pragma unroll
            for (int i = 0; i < T::MT; ++i) {
                const int r = wm * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                const int ch = kt * 2 + (lane >> 4);
                ldmatrix4(a[i], p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16);
            }
#pragma unroll
            for (int j = 0; j < T::NT; ++j) {
                const uint32_t b0 = fma2(pair(words[j][kt / 2], (kt & 1) * 8), sv[j], bv[j]);
                const uint32_t b1 = fma2(pair(words[j][kt / 2], (kt & 1) * 8 + 4), sv[j], bv[j]);
#pragma unroll
                for (int i = 0; i < T::MT; ++i) mma(acc[i][j], a[i], b0, b1);
            }
        }
    }
    wait<0>();
    __syncthreads();
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT; ++j) {
            const int col = n0 + wn * (BN / WN) + j * 8 + (lane & 3) * 2;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm * (BM / WM) + i * 16 + (lane >> 2) + h * 8;
                if (row >= M) continue;
                const float v0 = acc[i][j][2 * h], v1 = acc[i][j][2 * h + 1];
                __nv_bfloat16* dst = out + static_cast<size_t>(row) * N + col;
                if (col + 1 < N && (N & 1) == 0)
                    *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v0, v1);
                else {
                    if (col < N) dst[0] = __float2bfloat16_rn(v0);
                    if (col + 1 < N) dst[1] = __float2bfloat16_rn(v1);
                }
            }
        }
}

// ── W4A8 int8 prompt GEMM body (w4a8_prefill.cu `prefill8_body`) ───────────────────────────────────────────

template <int BM, int BN, int WM, int WN, int STAGES>
struct Tile8 {
    static constexpr int THREADS = WM * WN * 32, MT = BM / WM / 16, NT = BN / WN / 8;
    static constexpr int X = BM * GS;        // int8 inputs a stage (64-byte rows)
    static constexpr int XF = BM * 4 * 2;    // their scales and sums
    static constexpr int W = BN * GS / 2;
    static constexpr int S = BN * 2;
    static constexpr int STAGE = X + XF + W + 2 * S;
    static constexpr int SMEM = STAGES * STAGE;
};

using I8Tile = Tile8<128, 128, 2, 4, 4>;
struct I8Smem {
    unsigned char bytes[I8Tile::SMEM];
};
static_assert(I8Tile::SMEM == 55296, "port/affine/prefill.rs INT8_SMEM_BYTES");

template <int BM, int BN, int WM, int WN, int STAGES>
__device__ __forceinline__ void i8_body(const uint8_t* __restrict__ xq, const float* __restrict__ xa,
                                        const float* __restrict__ xs, const uint32_t* __restrict__ w,
                                        const __nv_bfloat16* __restrict__ scales,
                                        const __nv_bfloat16* __restrict__ biases, __nv_bfloat16* __restrict__ out,
                                        int M, int N, int K, int npad, int group, int block) {
    using T = Tile8<BM, BN, WM, WN, STAGES>;
    unsigned char* buf = avarok_prefill_smem;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const int KG = K / GS;
    const int2 at = tile_of(block, M, N, BM, BN, group);
    const int m0 = at.x, n0 = at.y;
    auto stage = [&](int s) { return buf + s * T::STAGE; };
    auto load = [&](int s, int g) {
        unsigned char* p = stage(s);
        for (int c = tid; c < BM * 4; c += T::THREADS) {   // 4 chunks of 16 bytes a 64-byte row
            const int r = c >> 2, ch = c & 3, row = min(m0 + r, M - 1);
            cp16z(p + r * GS + swz<4>(r, ch) * 16, xq + (size_t)row * K + g * GS + ch * 16, m0 + r < M);
        }
        float* pf = reinterpret_cast<float*>(p + T::X);
        for (int c = tid; c < 2 * BM / 4; c += T::THREADS) {   // scales then sums, 4 rows a thread
            const int which = c / (BM / 4), r = (c % (BM / 4)) * 4;
            const float* src = which ? xs : xa;
            // rows are not contiguous per group: copy 4 bytes each
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int row = min(m0 + r + e, M - 1);
                cp4(pf + which * BM + r + e, src + (size_t)row * KG + g);
            }
        }
        unsigned char* pw = p + T::X + T::XF;
        constexpr int TILE_BYTES = 64 * GS / 2;
        for (int c = tid; c < T::W / 16; c += T::THREADS) {
            const int t = c / (TILE_BYTES / 16), off = c % (TILE_BYTES / 16);
            const size_t tile = (size_t)(n0 / 64 + t) * KG + g;
            if (n0 + t * 64 < npad)
                cp16(pw + c * 16, reinterpret_cast<const unsigned char*>(w) + tile * TILE_BYTES + off * 16);
        }
        unsigned char* ps = pw + T::W;
        for (int c = tid; c < 2 * (T::S / 16); c += T::THREADS) {
            const int which = c / (T::S / 16), off = c % (T::S / 16);
            const __nv_bfloat16* src = (which ? biases : scales) + (size_t)g * npad + n0 + off * 8;
            if (n0 + off * 8 < npad) cp16(ps + which * T::S + off * 16, src);
        }
    };
    float acc[T::MT][T::NT][4];
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KG) load(s, s);
        commit();
    }
    for (int it = 0; it < KG; ++it) {
        wait<STAGES - 2>();
        __syncthreads();
        const int next = it + STAGES - 1;
        if (next < KG) load(next % STAGES, next);
        commit();
        const unsigned char* p = stage(it % STAGES);
        const float* pf = reinterpret_cast<const float*>(p + T::X);
        const uint32_t* pw = reinterpret_cast<const uint32_t*>(p + T::X + T::XF);
        const __nv_bfloat16* ps = reinterpret_cast<const __nv_bfloat16*>(p + T::X + T::XF + T::W);
        int iacc[T::MT][T::NT][4];
#pragma unroll
        for (int i = 0; i < T::MT; ++i)
#pragma unroll
            for (int j = 0; j < T::NT; ++j)
                iacc[i][j][0] = iacc[i][j][1] = iacc[i][j][2] = iacc[i][j][3] = 0x4B400000;
#pragma unroll
        for (int kt = 0; kt < 2; ++kt) {   // two 32-input steps a group
            uint32_t a[T::MT][4];
#pragma unroll
            for (int i = 0; i < T::MT; ++i) {
                const int r = wm * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                const int ch = kt * 2 + (lane >> 4);
                ldmatrix4(a[i], p + r * GS + swz<4>(r, ch) * 16);
            }
#pragma unroll
            for (int j = 0; j < T::NT; ++j) {
                const uint32_t word = pw[((wn * T::NT + j) * 32 + lane) * 2 + kt];
                const uint32_t b0 = bytes4(word, 0), b1 = bytes4(word, 8);
#pragma unroll
                for (int i = 0; i < T::MT; ++i) mma_s8(iacc[i][j], a[i], b0, b1);
            }
        }
        // accumulators start at the float 12582912: one fma drops it and applies a (rc rounds: a dot unit at most)
        float ra[T::MT][2], rs[T::MT][2], rc[T::MT][2];
#pragma unroll
        for (int i = 0; i < T::MT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = wm * (BM / WM) + i * 16 + (lane >> 2) + h * 8;
                ra[i][h] = pf[r];
                rs[i][h] = pf[BM + r];
                rc[i][h] = -12582912.0f * ra[i][h];
            }
#pragma unroll
        for (int j = 0; j < T::NT; ++j) {
            const int c = wn * (BN / WN) + j * 8 + (lane & 3) * 2;
            const __nv_bfloat162 sp = *reinterpret_cast<const __nv_bfloat162*>(ps + c),
                                 bp = *reinterpret_cast<const __nv_bfloat162*>(ps + BN + c);
            const float s0 = __low2float(sp), s1 = __high2float(sp), b0 = __low2float(bp), b1 = __high2float(bp);
#pragma unroll
            for (int i = 0; i < T::MT; ++i)
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const float d0 = fmaf(__int_as_float(iacc[i][j][2 * h]), ra[i][h], rc[i][h]);
                    const float d1 = fmaf(__int_as_float(iacc[i][j][2 * h + 1]), ra[i][h], rc[i][h]);
                    acc[i][j][2 * h] = fmaf(b0, rs[i][h], fmaf(s0, d0, acc[i][j][2 * h]));
                    acc[i][j][2 * h + 1] = fmaf(b1, rs[i][h], fmaf(s1, d1, acc[i][j][2 * h + 1]));
                }
        }
    }
    wait<0>();
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT; ++j) {
            const int col = n0 + wn * (BN / WN) + j * 8 + (lane & 3) * 2;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm * (BM / WM) + i * 16 + (lane >> 2) + h * 8;
                if (row >= M || col >= N) continue;
                store2(out + (size_t)row * N + col, N - col, acc[i][j][2 * h], acc[i][j][2 * h + 1]);
            }
        }
}

// The projection of `_many` CTA `b`: the last part whose first CTA is at or below b.
__device__ __forceinline__ int part_of(const Parts& parts, int b) {
    int q = 0;
#pragma unroll
    for (int i = 1; i < PARTS; ++i)
        if (i < parts.count && b >= parts.p[i].first) q = i;
    return q;
}

// BF16 gate: NULL runs, otherwise only a raised flag runs; the projections run are added to flags[1].
__device__ __forceinline__ bool bf16_gate_closed(unsigned int* run_if, unsigned int projections) {
    if (run_if == nullptr) return false;
    if (*reinterpret_cast<const volatile unsigned int*>(run_if) == 0u) return true;
    if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(run_if + 1, projections);
    return false;
}

__device__ __forceinline__ bool int8_gate_closed(const unsigned int* flags) {
    return flags != nullptr && *reinterpret_cast<const volatile unsigned int*>(flags) != 0u;
}

}   // namespace avarok_affine_prefill

// Grid: (rows_t * cols_t)  Block: 256  Dynamic shared: 62,976 (exported below).
extern "C" __global__ void __launch_bounds__(256)
    affine4_prefill_bf16_ada(const __nv_bfloat16* __restrict__ x, const uint32_t* __restrict__ w,
                             const __nv_bfloat16* __restrict__ scales, const __nv_bfloat16* __restrict__ biases,
                             __nv_bfloat16* __restrict__ out, int M, int N, int K, int npad, int ldx, int group,
                             unsigned int* run_if) {
    if (avarok_affine_prefill::bf16_gate_closed(run_if, 1u)) return;
    avarok_affine_prefill::bf16_body<128, 128, 2, 4, 3>(x, w, scales, biases, out, M, N, K, npad, ldx, group,
                                                        blockIdx.x);
}
AVAROK_EXPORT_DYN_SMEM(affine4_prefill_bf16_ada, avarok_affine_prefill::BfSmem);

// Up to four projections that share x in one grid (gate + up, q + k + v): projection-local block indices.
// Grid: (sum of each part's rows_t * cols_t)  Block: 256  Dynamic shared: 62,976.
extern "C" __global__ void __launch_bounds__(256)
    affine4_prefill_bf16_many_ada(const __nv_bfloat16* __restrict__ x,
                                  const __grid_constant__ avarok_affine_prefill::Parts parts, int M, int K, int ldx,
                                  int group, unsigned int* run_if) {
    if (avarok_affine_prefill::bf16_gate_closed(run_if, (unsigned int)parts.count)) return;
    const int q = avarok_affine_prefill::part_of(parts, blockIdx.x);
    const avarok_affine_prefill::Part P = parts.p[q];
    avarok_affine_prefill::bf16_body<128, 128, 2, 4, 3>(x, P.w, P.scales, P.biases, P.out, M, P.n, K, P.npad, ldx,
                                                        group, (int)blockIdx.x - P.first);
}
AVAROK_EXPORT_DYN_SMEM(affine4_prefill_bf16_many_ada, avarok_affine_prefill::BfSmem);

// Grid: (rows_t * cols_t)  Block: 256  Dynamic shared: 55,296 (exported below).
extern "C" __global__ void __launch_bounds__(256)
    affine4_w4a8_prefill_ada(const uint8_t* __restrict__ xq, const float* __restrict__ xa,
                             const float* __restrict__ xs, const uint32_t* __restrict__ w,
                             const __nv_bfloat16* __restrict__ scales, const __nv_bfloat16* __restrict__ biases,
                             __nv_bfloat16* __restrict__ out, int M, int N, int K, int npad, int group,
                             const unsigned int* flags) {
    if (avarok_affine_prefill::int8_gate_closed(flags)) return;
    avarok_affine_prefill::i8_body<128, 128, 2, 4, 4>(xq, xa, xs, w, scales, biases, out, M, N, K, npad, group,
                                                      blockIdx.x);
}
AVAROK_EXPORT_DYN_SMEM(affine4_w4a8_prefill_ada, avarok_affine_prefill::I8Smem);

// Up to four projections over one quantized x. Grid: (sum of the parts' tiles)  Block: 256  Dynamic: 55,296.
extern "C" __global__ void __launch_bounds__(256)
    affine4_w4a8_prefill_many_ada(const uint8_t* __restrict__ xq, const float* __restrict__ xa,
                                  const float* __restrict__ xs,
                                  const __grid_constant__ avarok_affine_prefill::Parts parts, int M, int K,
                                  int group, const unsigned int* flags) {
    if (avarok_affine_prefill::int8_gate_closed(flags)) return;
    const int q = avarok_affine_prefill::part_of(parts, blockIdx.x);
    const avarok_affine_prefill::Part P = parts.p[q];
    avarok_affine_prefill::i8_body<128, 128, 2, 4, 4>(xq, xa, xs, P.w, P.scales, P.biases, P.out, M, P.n, K,
                                                      P.npad, group, (int)blockIdx.x - P.first);
}
AVAROK_EXPORT_DYN_SMEM(affine4_w4a8_prefill_many_ada, avarok_affine_prefill::I8Smem);
