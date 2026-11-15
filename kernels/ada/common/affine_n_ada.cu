// SPDX-License-Identifier: AGPL-3.0-only AND Apache-2.0
// Portions copied and adapted from TensorFold (https://github.com/ashhart/TensorFold), commit 3885dce,
// zig/kernels/cuda/affine8.cu:1-385 (Part, Tile, items, store, gemv, split, the narrow and 8-bit
// gemv/split instances, tf_affinen_pack, tf_affine8_pack8), zig/kernels/cuda/affine4_pack.cu:29-36
// (tf_transpose_pad16), zig/kernels/cuda/qwen35.cu:12-23,184-190 (group_sum, tf_q35_sums), and a GPU
// quantizer translated from zig/src/core/affine4_host.zig:41-66 (quantizeMlx).
// Copyright 2026 TensorFold contributors. Licensed under the Apache License, Version 2.0; code written
// before TensorFold 0.6.0 is also available under the MIT License. See TENSORFOLD-ATTRIBUTION.md.
// Changes for Atlas: Atlas entry names (`affine<B>_gemv_ada`, ...); plain scalar and pointer kernel
// arguments instead of a by-value `__grid_constant__` Part (Atlas's KernelLaunch passes scalars and
// pointers); the PDL grid_wait/grid_launch pair dropped (sm_90 only); the quantizer generalized from
// 4 bits to 2, 3, 5, 6 and 8 bits, run as one warp a 64-group with an optional clip search (an Atlas
// addition); a new tile-layout BF16 dequant kernel for the prefill fallback and the oracle;
// tf_affine8_pack4 and tf_affine8_widen not carried (no caller in Atlas); comments rewritten.
//
// MLX affine weights of 2, 3, 5, 6 or 8 bits a weight on sm_89.
// Host side: crates/spark-model/src/port/affine/widths.rs. Nothing here runs
// unless a W4 route row is on AND the route asked for a width other than 4 (`--w4-bits`); the 4-bit
// lane kernels are in `affine4_lane_ada.cu`.
//
// FORMAT. Per output row and 64-input group: codes q in [0, 2^bits - 1], a bf16 scale s and a bf16
// bias b, value = s * q + b (MLX's affine format). The checkpoint (MLX) layout is (n, k * bits / 32)
// little-endian u32 words, codes contiguous low first, plus (n, k / 64) scales and biases. The GEMV
// reads repacked tiles, npad = n rounded up to 128 rows:
//   * 2, 3, 5, 6 bits ("narrow"): [npad/64][k/64][4 warps][32 lanes][BITS] words. A lane's BITS
//     words hold its 16 MMA operand pairs bit-sliced: pair i's first code is bits [i*BITS, i*BITS +
//     BITS) of the low halves, its second code the same bits of the high halves.
//   * 8 bits: [npad/64][k/64][8][32 lanes][4] words; word kt holds inputs k0 + 2c + {0, 1, 8, 9}.
//   * scales and biases: [k/64][npad] bf16 (transposed, padded rows zero).
//
// GEMV. One 16-row m16n8k16 bf16 tile a CTA (128 threads, four warps of 16 columns), 64 columns a
// tile, one 64-group a cp.async stage. Inputs are bf16, codes enter the MMA as exact bf16 integers,
// and the epilogue applies acc += s * (x . q) + b * sum(x) per group with explicit fused fp32 FMAs.
// sum(x) per row and group comes from `affine_n_sums_ada`. Split-K is fixed by the weight's shape
// (host `split_k`), never by the row count, and the split form sums its slices in slice order from
// the first, exactly as the unsplit form does, so the two are bit-identical. Rows 1..16 run the same
// tile with the missing rows zero-filled, so a row's bits never depend on how many rows ran.
//
// SHARED MEMORY. gemv: 14208 + 3072 * BITS bytes (narrow) or 38784 (8-bit); split: 9472 + 2048 * BITS
// or 25856. All under the 48 KB default, so no opt-in attribute is needed.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "affine_codes_ada.cuh"

namespace avarok_affine_n {

struct Part {
    const uint32_t* w;
    const __nv_bfloat16* scales;
    const __nv_bfloat16* biases;
    void* out;
    int n, npad, sk, tiles;
};

// The lane tile with BITS bits a weight: 16 rows, BN columns over four warps, one GS-group a stage.
template <int GS, int BN, int STAGES, int BITS = 8>
struct Tile {
    static constexpr int THREADS = 128;
    static constexpr int NT = BN / 4 / 8;
    static constexpr int ROW = GS * 2;
    static constexpr int CHUNKS = ROW / 16;
    static constexpr int X = 16 * ROW;
    static constexpr int W = BN * GS * BITS / 8;
    static constexpr int S = BN * 2;
    static constexpr int XS = 16 * 4;
    static constexpr int STAGE = X + W + 2 * S + XS;
    static constexpr int SMEM = STAGES * STAGE;
};

// Stream `count` (group, column-tile) items through the cp.async pipeline, accumulating into acc;
// `after(it)` runs once an item's epilogue is applied.
template <int GS, int BN, int STAGES, int BITS, typename Map, typename After>
__device__ __forceinline__ void items(const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs, const Part& P,
                                      int M, int K, int ldx, int count, Map map,
                                      float (&acc)[Tile<GS, BN, STAGES, BITS>::NT][4], After after) {
    using T = Tile<GS, BN, STAGES, BITS>;
    static_assert(BITS == 8 || (GS == 64 && BN == 64 && BITS >= 2 && BITS <= 6), "narrow tiles are 64 by 64, 2 to 6 bits");
    constexpr int NT = T::NT;
    constexpr int KW = GS / 16;  // words a lane a group on the 8-bit tiles: one per k16 step
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, wn = tid >> 5;
    const int KG = K / GS;
    auto stage = [&](int s) { return buf + s * T::STAGE; };
    const int rows = min(16, M);
    constexpr int TILE_BYTES = 64 * GS * BITS / 8;
    constexpr int XC = (16 * T::CHUNKS + T::THREADS - 1) / T::THREADS, WC = (T::W / 16 + T::THREADS - 1) / T::THREADS;
    constexpr int SC = (2 * T::S / 16 + T::THREADS - 1) / T::THREADS;
    const __nv_bfloat16* xsrc[XC];
    size_t woff[WC];
    int soff[SC], xdst[XC], wdst[WC], sdst[SC];
    bool xok[XC], wok[WC], sok[SC], sbias[SC];
#pragma unroll
    for (int j = 0; j < XC; ++j) {
        const int c = tid + j * T::THREADS, r = c / T::CHUNKS, ch = c % T::CHUNKS;
        xok[j] = c < 16 * T::CHUNKS && r < rows;
        xsrc[j] = x + static_cast<size_t>(min(r, rows - 1)) * ldx + ch * 8;
        xdst[j] = r * T::ROW + swz<T::CHUNKS>(r, ch) * 16;
    }
#pragma unroll
    for (int j = 0; j < WC; ++j) {
        const int c = tid + j * T::THREADS, t = c / (TILE_BYTES / 16), off = c % (TILE_BYTES / 16);
        wok[j] = c < T::W / 16;
        woff[j] = static_cast<size_t>(t) * KG * TILE_BYTES + off * 16;
        wdst[j] = T::X + c * 16;
    }
#pragma unroll
    for (int j = 0; j < SC; ++j) {
        const int c = tid + j * T::THREADS, which = c / (T::S / 16), off = c % (T::S / 16);
        sok[j] = c < 2 * (T::S / 16);
        sbias[j] = which;
        soff[j] = off * 8;
        sdst[j] = T::X + T::W + which * T::S + off * 16;
    }
    const bool xsok = tid < rows;
    const float* xssrc = xs + static_cast<size_t>(min(tid, rows - 1)) * KG;
    auto load_x = [&](int s, int i) {
        const int g = map(i).x;
        unsigned char* p = stage(s);
#pragma unroll
        for (int j = 0; j < XC; ++j)
            if (xok[j]) cp16(p + xdst[j], xsrc[j] + g * GS);
        if (xsok) cp4(p + T::X + T::W + 2 * T::S + tid * 4, xssrc + g);
    };
    auto load_w = [&](int s, int i) {
        const int2 at = map(i);
        const int g = at.x, n0 = at.y;
        unsigned char* p = stage(s);
        const unsigned char* wb = reinterpret_cast<const unsigned char*>(P.w) + static_cast<size_t>(n0 / 64) * KG * TILE_BYTES +
                                  (n0 % 64) * GS * BITS / 8;
#pragma unroll
        for (int j = 0; j < WC; ++j)
            if (wok[j]) cp16(p + wdst[j], wb + woff[j] + static_cast<size_t>(g) * TILE_BYTES);
#pragma unroll
        for (int j = 0; j < SC; ++j)
            if (sok[j]) cp16(p + sdst[j], (sbias[j] ? P.biases : P.scales) + n0 + soff[j] + static_cast<size_t>(g) * P.npad);
    };

    // Rows at and above M stay zero in every stage, so they add nothing to any row's MMA.
    for (int c = tid; c < 16 * T::CHUNKS; c += T::THREADS)
        if (c / T::CHUNKS >= rows)
#pragma unroll
            for (int s = 0; s < STAGES; ++s)
                *reinterpret_cast<uint4*>(stage(s) + c / T::CHUNKS * T::ROW + c % T::CHUNKS * 16) = uint4{};
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < count) load_w(s, s);
        commit();
    }
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < count) load_x(s, s);
        commit();
    }
    for (int it = 0; it < count; ++it) {
        wait<STAGES - 2>();
        __syncthreads();
        const int next = it + STAGES - 1;
        if (next < count) {
            load_x(next % STAGES, next);
            load_w(next % STAGES, next);
        }
        commit();
        const unsigned char* p = stage(it % STAGES);
        const uint32_t* pw = reinterpret_cast<const uint32_t*>(p + T::X);
        const __nv_bfloat16* ps = reinterpret_cast<const __nv_bfloat16*>(p + T::X + T::W);
        const float* px = reinterpret_cast<const float*>(p + T::X + T::W + 2 * T::S);
        constexpr int LW = BITS == 8 ? NT * KW : BITS;  // a lane's words a group
        uint32_t lw[LW];
        if constexpr (BITS == 8) {
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int v = 0; v < KW; ++v) lw[j * KW + v] = pw[((wn * NT + j) * 32 + lane) * KW + v];
        } else if constexpr (BITS % 2 == 0) {
#pragma unroll
            for (int v = 0; v < BITS; v += 2) {
                const uint2 two = *reinterpret_cast<const uint2*>(pw + (wn * 32 + lane) * BITS + v);
                lw[v] = two.x;
                lw[v + 1] = two.y;
            }
        } else {
#pragma unroll
            for (int v = 0; v < BITS; ++v) lw[v] = pw[(wn * 32 + lane) * BITS + v];
        }
        float d[NT][4];
#pragma unroll
        for (int kt = 0; kt < GS / 16; ++kt) {
            uint32_t a[4];
            const int r = (lane & 7) + ((lane >> 3) & 1) * 8, ch = kt * 2 + (lane >> 4);
            ldmatrix4(a, p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16);
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                uint32_t b0, b1;
                if constexpr (BITS == 8) {
                    b0 = byte_pair(lw[j * KW + kt], 0);
                    b1 = byte_pair(lw[j * KW + kt], 16);
                } else {
                    b0 = pair_at<BITS>(lw, (j * 4 + kt) * 2);
                    b1 = pair_at<BITS>(lw, (j * 4 + kt) * 2 + 1);
                }
                if (kt == 0) mma0(d[j], a, b0, b1);
                else mma(d[j], a, b0, b1);
            }
        }
        const int row = lane >> 2;
        const float xv[2] = {px[row], px[row + 8]};
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            const int col = wn * (BN / 4) + j * 8 + (lane & 3) * 2;
            const __nv_bfloat162 s2 = *reinterpret_cast<const __nv_bfloat162*>(ps + col);
            const __nv_bfloat162 b2 = *reinterpret_cast<const __nv_bfloat162*>(ps + BN + col);
            const float sv[2] = {__low2float(s2), __high2float(s2)};
            const float bv[2] = {__low2float(b2), __high2float(b2)};
#pragma unroll
            for (int e = 0; e < 4; ++e)
                acc[j][e] = __fmaf_rn(xv[e >> 1], bv[e & 1], __fmaf_rn(d[j][e], sv[e & 1], acc[j][e]));
        }
        after(it);
    }
}

// Rows below M of the BN columns at n0 as bf16, from fp32 pairs got(j, h, row, col) in the
// accumulator's layout.
template <int BN, int NT, typename Got>
__device__ __forceinline__ void store(const Part& P, int M, int n0, Got got) {
    const int lane = threadIdx.x & 31, wn = threadIdx.x >> 5, row = lane >> 2, N = P.n;
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = n0 + wn * (BN / 4) + j * 8 + (lane & 3) * 2, r = row + h * 8;
            if (r >= M) continue;
            const float2 v = got(j, h, r, col - n0);
            auto* dst = reinterpret_cast<__nv_bfloat16*>(P.out) + static_cast<size_t>(r) * N + col;
            if (col + 1 < N && (N & 1) == 0) {
                *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v.x, v.y);
            } else {
                if (col < N) dst[0] = __float2bfloat16_rn(v.x);
                if (col + 1 < N) dst[1] = __float2bfloat16_rn(v.y);
            }
        }
}

// CTAs loop over column tiles; each tile's K slices are summed in slice order from the first.
template <int GS, int BN, int STAGES, int BITS>
__device__ __forceinline__ void gemv(const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs, const Part& P, int M,
                                     int K, int ldx) {
    constexpr int NT = Tile<GS, BN, STAGES, BITS>::NT;
    const int KG = K / GS, per = KG / P.sk;
    const int mine = P.tiles > (int)blockIdx.x ? (P.tiles - 1 - (int)blockIdx.x) / (int)gridDim.x + 1 : 0;
    auto tile_of_item = [&](int i) { return (int)blockIdx.x + (i / KG) * (int)gridDim.x; };
    float acc[NT][4], tot[NT][4];
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[j][e] = tot[j][e] = 0.0f;
    items<GS, BN, STAGES, BITS>(
        x, xs, P, M, K, ldx, mine * KG, [&](int i) { return make_int2(i % KG, tile_of_item(i) * BN); }, acc, [&](int it) {
            const int g = it % KG;
            if ((g + 1) % per != 0) return;
            const bool first = g + 1 == per;
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    tot[j][e] = first ? acc[j][e] : tot[j][e] + acc[j][e];
                    acc[j][e] = 0.0f;
                }
            if (g != KG - 1) return;
            store<BN, NT>(P, M, tile_of_item(it) * BN,
                          [&](int j, int h, int, int) { return make_float2(tot[j][2 * h], tot[j][2 * h + 1]); });
        });
}

// A CTA a tile's K slice; the tile's last CTA (by ticket) sums the slices in their order from the
// first and resets the ticket, so the ticket array is zero again after every launch.
template <int GS, int BN, int STAGES, int BITS>
__device__ __forceinline__ void split(const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs, const Part& P, int M,
                                      int K, int ldx, float* __restrict__ work, unsigned* __restrict__ tickets) {
    constexpr int NT = Tile<GS, BN, STAGES, BITS>::NT;
    __shared__ unsigned ticket;
    const int tid = threadIdx.x, lane = tid & 31, wn = tid >> 5;
    const int per = K / GS / P.sk;
    const int tile = (int)blockIdx.x / P.sk, slice = (int)blockIdx.x % P.sk, g0 = slice * per;
    float acc[NT][4];
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[j][e] = 0.0f;
    items<GS, BN, STAGES, BITS>(x, xs, P, M, K, ldx, per, [&](int i) { return make_int2(g0 + i, tile * BN); }, acc, [](int) {});
    const int row = lane >> 2;
    float* part = work + (static_cast<size_t>(tile) * P.sk + slice) * 16 * BN;
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int c = wn * (BN / 4) + j * 8 + (lane & 3) * 2, r = row + h * 8;
            if (r >= M) continue;
            part[r * BN + c] = acc[j][2 * h];
            part[r * BN + c + 1] = acc[j][2 * h + 1];
        }
    __threadfence();
    __syncthreads();
    if (tid == 0) ticket = atomicAdd(tickets + tile, 1u);
    __syncthreads();
    if (ticket != static_cast<unsigned>(P.sk - 1)) return;
    __threadfence();
    const float* base = work + static_cast<size_t>(tile) * P.sk * 16 * BN;
    store<BN, NT>(P, M, tile * BN, [&](int, int, int r, int c) {
        float t0 = __ldcg(base + r * BN + c), t1 = __ldcg(base + r * BN + c + 1);
        for (int s = 1; s < P.sk; ++s) {
            t0 = t0 + __ldcg(base + (s * 16 + r) * BN + c);
            t1 = t1 + __ldcg(base + (s * 16 + r) * BN + c + 1);
        }
        return make_float2(t0, t1);
    });
    if (tid == 0) tickets[tile] = 0;
}

__device__ __forceinline__ Part part_of(const uint32_t* w, const __nv_bfloat16* scales, const __nv_bfloat16* biases, void* out,
                                        int n, int npad, int sk, int tiles) {
    Part p;
    p.w = w;
    p.scales = scales;
    p.biases = biases;
    p.out = out;
    p.n = n;
    p.npad = npad;
    p.sk = sk;
    p.tiles = tiles;
    return p;
}

// One 64-entry group's sum, its entries added in order (x 16-byte aligned).
__device__ __forceinline__ float group_sum(const __nv_bfloat16* x) {
    float s = 0.0f;
#pragma unroll
    for (int j = 0; j < 64; j += 8) {
        const uint4 w = *reinterpret_cast<const uint4*>(x + j);
        const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&w);
#pragma unroll
        for (int i = 0; i < 8; i++) s = __fadd_rn(s, __bfloat162float(b[i]));
    }
    return s;
}

__device__ __forceinline__ float bf16_rounded(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }

// MLX's affine scale and bias for the range [l, h] at `nb` = 2^bits - 1 levels: the larger-magnitude
// edge lands exactly on the grid. False when the group cannot be represented (non-finite or zero).
__device__ __forceinline__ bool mlx_params(float l, float h, float nb, float& s, float& b) {
    const float step = fmaxf(__fdiv_rn(__fsub_rn(h, l), nb), 1e-7f);
    if (!isfinite(step)) return false;
    const bool low = fabsf(l) > fabsf(h);
    const float sg = low ? step : -step;
    const float edge = low ? l : h;
    const float q0 = roundf(__fdiv_rn(edge, sg));
    s = q0 == 0.0f ? sg : __fdiv_rn(edge, q0);
    b = q0 == 0.0f ? 0.0f : edge;
    if (!isfinite(s) || s == 0.0f || !isfinite(b)) return false;
    return isfinite(bf16_rounded(s)) && isfinite(bf16_rounded(b));
}

// The code of v: round half away from zero of (v - b) / s, clamped to [0, nb], from the UNROUNDED
// fp32 scale and bias (bf16 is the storage type only).
__device__ __forceinline__ float mlx_code(float v, float s, float b, float nb) {
    return fminf(fmaxf(roundf(__fdiv_rn(__fsub_rn(v, b), s)), 0.0f), nb);
}

}  // namespace avarok_affine_n

using avarok_affine_n::Tile;

static_assert(Tile<64, 64, 6>::SMEM == 38784, "affine8 gemv smem");
static_assert(Tile<64, 64, 4>::SMEM == 25856, "affine8 split smem");

// The gemv and split entries of one width B (2, 3, 5, 6 narrow; 8 on the 8-bit tiles).
//   gemv grid: min(tiles, resident CTAs), block 128, dynamic shared Tile<64, 64, 6, B>::SMEM.
//   split grid: tiles * sk, block 128, dynamic shared Tile<64, 64, 4, B>::SMEM; work holds
//   tiles * sk * 16 * 64 fp32, tickets holds `tiles` zeroed u32 (left zero on exit).
#define AVAROK_AFFINE_WIDTH(B)                                                                                              \
    extern "C" __global__ void __launch_bounds__(128)                                                                     \
        affine##B##_gemv_ada(const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs, const uint32_t* w,          \
                             const __nv_bfloat16* scales, const __nv_bfloat16* biases, __nv_bfloat16* out, int n, int npad, \
                             int sk, int tiles, int M, int K, int ldx) {                                                    \
        const avarok_affine_n::Part P = avarok_affine_n::part_of(w, scales, biases, out, n, npad, sk, tiles);              \
        avarok_affine_n::gemv<64, 64, 6, B>(x, xs, P, M, K, ldx);                                                           \
    }                                                                                                                       \
    extern "C" __global__ void __launch_bounds__(128)                                                                     \
        affine##B##_split_ada(const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs, const uint32_t* w,         \
                              const __nv_bfloat16* scales, const __nv_bfloat16* biases, __nv_bfloat16* out, int n,          \
                              int npad, int sk, int tiles, int M, int K, int ldx, float* __restrict__ work,                 \
                              unsigned* __restrict__ tickets) {                                                             \
        const avarok_affine_n::Part P = avarok_affine_n::part_of(w, scales, biases, out, n, npad, sk, tiles);              \
        avarok_affine_n::split<64, 64, 4, B>(x, xs, P, M, K, ldx, work, tickets);                                           \
    }

#define AVAROK_AFFINE_NARROW(B)                                                                    \
    static_assert(Tile<64, 64, 6, B>::SMEM == 14208 + 3072 * B, "narrow gemv smem");               \
    static_assert(Tile<64, 64, 4, B>::SMEM == 9472 + 2048 * B, "narrow split smem");               \
    AVAROK_AFFINE_WIDTH(B)

AVAROK_AFFINE_NARROW(2)
AVAROK_AFFINE_NARROW(3)
AVAROK_AFFINE_NARROW(5)
AVAROK_AFFINE_NARROW(6)
AVAROK_AFFINE_WIDTH(8)
#undef AVAROK_AFFINE_NARROW
#undef AVAROK_AFFINE_WIDTH

// 64-group sums of bf16 rows D wide (the GEMV's sum(x) per row and group), entries in order.
// Grid (rows, ceil(D / 64 / 64)), block 64, a thread a group; XS is [rows][D / 64] fp32.
extern "C" __global__ void __launch_bounds__(64) affine_n_sums_ada(const __nv_bfloat16* __restrict__ X, float* __restrict__ XS,
                                                                   int D) {
    const size_t row = blockIdx.x;
    const int g = blockIdx.y * blockDim.x + threadIdx.x;
    if (g < D / 64) XS[row * (D / 64) + g] = avarok_affine_n::group_sum(X + row * D + static_cast<size_t>(g) * 64);
}

// Squared error of the bf16-STORED dequantization of a lane's two entries under (s, b), summed over
// the warp by a fixed fp32 butterfly (the same value in every lane).
__device__ __forceinline__ float avarok_affine_n_err(float x0, float x1, float s, float b, float nb) {
    using namespace avarok_affine_n;
    const float sb = bf16_rounded(s), bb = bf16_rounded(b);
    const float d0 = __fsub_rn(__fadd_rn(__fmul_rn(sb, mlx_code(x0, s, b, nb)), bb), x0);
    const float d1 = __fsub_rn(__fadd_rn(__fmul_rn(sb, mlx_code(x1, s, b, nb)), bb), x1);
    float e = __fadd_rn(__fadd_rn(0.0f, __fmul_rn(d0, d0)), __fmul_rn(d1, d1));
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) e = __fadd_rn(e, __shfl_xor_sync(0xFFFFFFFFu, e, off));
    return e;
}

// MLX's affine quantizer for `bits` in {2, 3, 5, 6, 8}, groups of 64, from bf16 rows [n][k].
// One warp a group (lane l owns entries 2l and 2l + 1): min and max by warp reduction (a -0.0 edge
// canonicalized to +0.0), MLX's scale and bias (affine4_host.zig quantizeMlx with 2^bits - 1
// levels), codes OR-ed into `words` [n][k*bits/32] (which the caller zeroes), bf16 scale and bias bits
// into [n][k/64]. With clip > 0 (at most 10) it also shrinks [lo, hi] about its midpoint by
// 1 - 0.05 i for i = 1..clip and keeps the candidate with the least squared error of the stored
// dequantization; ties keep the earlier candidate, so clip = 0 is plain MLX. This is the clip search
// of the 4-bit quantizer (affine_pack_ada.cu, lane L17a), so `--w4-clip-search` means one thing at
// every width. A group with a non-finite entry or no representable scale writes its index to *bad by
// atomicMin (the caller seeds 0xFFFFFFFF). Grid ceil(n * k / 64 / 4), block 128.
extern "C" __global__ void __launch_bounds__(128)
    affine_n_quantize_ada(const __nv_bfloat16* __restrict__ src, int n, int k, int bits, int clip, uint32_t* __restrict__ words,
                          uint16_t* __restrict__ scales, uint16_t* __restrict__ biases, unsigned* __restrict__ bad) {
    using namespace avarok_affine_n;
    const int kg = k / 64;
    const long long groups = static_cast<long long>(n) * kg;
    const long long gw = static_cast<long long>(blockIdx.x) * 4 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (gw >= groups) return;
    const long long row = gw / kg;
    const int g = static_cast<int>(gw % kg);
    const __nv_bfloat16* xp = src + row * k + g * 64 + 2 * lane;
    const float x0 = __bfloat162float(xp[0]), x1 = __bfloat162float(xp[1]);
    if (__any_sync(0xFFFFFFFFu, !(isfinite(x0) && isfinite(x1)))) {
        if (lane == 0) atomicMin(bad, static_cast<unsigned>(gw));
        return;
    }
    float lo = fminf(x0, x1), hi = fmaxf(x0, x1);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        lo = fminf(lo, __shfl_xor_sync(0xFFFFFFFFu, lo, off));
        hi = fmaxf(hi, __shfl_xor_sync(0xFFFFFFFFu, hi, off));
    }
    lo = __fadd_rn(lo, 0.0f);  // -0.0 -> +0.0, so the result never depends on which zero won
    hi = __fadd_rn(hi, 0.0f);
    const float nb = static_cast<float>((1u << bits) - 1u);
    float best_s = 0.0f, best_b = 0.0f;
    if (!mlx_params(lo, hi, nb, best_s, best_b)) {
        if (lane == 0) atomicMin(bad, static_cast<unsigned>(gw));
        return;
    }
    if (clip > 0) {
        float best = avarok_affine_n_err(x0, x1, best_s, best_b, nb);
        const float mid = __fmul_rn(__fadd_rn(lo, hi), 0.5f);
        const float half_w = __fmul_rn(__fsub_rn(hi, lo), 0.5f);
        for (int i = 1; i <= min(clip, 10); ++i) {
            const float h = __fmul_rn(half_w, __fsub_rn(1.0f, __fmul_rn(0.05f, static_cast<float>(i))));
            float s, b;
            if (!mlx_params(__fsub_rn(mid, h), __fadd_rn(mid, h), nb, s, b)) continue;
            const float e = avarok_affine_n_err(x0, x1, s, b, nb);
            if (e < best) {
                best = e;
                best_s = s;
                best_b = b;
            }
        }
    }
    uint32_t* rw = words + row * (static_cast<long long>(k) * bits / 32);
#pragma unroll
    for (int e = 0; e < 2; ++e) {
        const uint32_t q = static_cast<uint32_t>(mlx_code(e == 0 ? x0 : x1, best_s, best_b, nb));
        const long long bit = (static_cast<long long>(g) * 64 + 2 * lane + e) * bits;
        const int sh = static_cast<int>(bit & 31);
        const long long wd = bit >> 5;
        atomicOr(rw + wd, q << sh);
        if (sh + bits > 32) atomicOr(rw + wd + 1, q >> (32 - sh));
    }
    if (lane == 0) {
        const __nv_bfloat16 sbf = __float2bfloat16_rn(best_s), bbf = __float2bfloat16_rn(best_b);
        scales[row * kg + g] = *reinterpret_cast<const uint16_t*>(&sbf);
        biases[row * kg + g] = *reinterpret_cast<const uint16_t*>(&bbf);
    }
}

// MLX words of 2, 3, 5 or 6 bits -> the narrow tiles of that width: a lane's pairs' first codes in
// the words' low halves, second codes in the high halves. One thread an output word; total =
// npad / 64 * kg * 128 * bits. Rows at and above n are zero.
extern "C" __global__ void __launch_bounds__(256) affine_n_pack_ada(const uint32_t* __restrict__ w, int bits, int n, int kw, int kg,
                                                                    uint32_t* __restrict__ out, long long total) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int v = static_cast<int>(idx % bits);
    const long long rest = idx / bits;
    const int lane = static_cast<int>(rest & 31), wn = static_cast<int>((rest >> 5) & 3);
    const long long tg = rest >> 7;
    const int g = static_cast<int>(tg % kg);
    const long long t = tg / kg;
    const int c = lane & 3;
    uint32_t word = 0;
    for (int b = 0; b < 16; ++b) {
        const int sb = 16 * v + b, i = sb / bits, at = sb % bits;
        const int j = i >> 3, kt = (i >> 1) & 3, h = i & 1;
        const long long row = t * 64 + (wn * 2 + j) * 8 + (lane >> 2);
        if (row >= n) continue;
        const uint32_t* src = w + row * kw;
        const long long k = static_cast<long long>(g) * 64 + 16 * kt + 2 * c + 8 * h;
        word |= ((avarok_affine_n::code_at(src, k, bits) >> at) & 1u) << b;
        word |= ((avarok_affine_n::code_at(src, k + 1, bits) >> at) & 1u) << (16 + b);
    }
    out[idx] = word;
}

// MLX words of 5 to 8 bits -> the 8-bit tiles [npad/64][kg][8][32][4]: word kt holds inputs
// k0 + 2c + {0, 1, 8, 9}, one a byte. One thread an output word; total = npad / 64 * kg * 1024.
extern "C" __global__ void __launch_bounds__(256) affine8_pack_ada(const uint32_t* __restrict__ w, int bits, int n, int kw, int kg,
                                                                   uint32_t* __restrict__ out, long long total) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int kt = idx & 3, lane = (idx >> 2) & 31, j = (idx >> 7) & 7;
    const long long tg = idx >> 10;
    const int g = static_cast<int>(tg % kg);
    const long long t = tg / kg;
    const long long row = t * 64 + j * 8 + (lane >> 2);
    const int k0 = g * 64 + 16 * kt + 2 * (lane & 3);
    const int offs[4] = {0, 1, 8, 9};
    uint32_t word = 0;
    if (row < n) {
        const uint32_t* src = w + row * kw;
#pragma unroll
        for (int p = 0; p < 4; ++p) word |= avarok_affine_n::code_at(src, k0 + offs[p], bits) << (8 * p);
    }
    out[idx] = word;
}

// (n, kg) bf16 scales or biases -> the tiles' (kg, npad) layout, rows at and above n zero.
extern "C" __global__ void __launch_bounds__(256) affine_n_transpose_pad16_ada(const uint16_t* __restrict__ x, int n, int kg, int npad,
                                                                              uint16_t* __restrict__ out) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<long long>(kg) * npad) return;
    const int g = static_cast<int>(idx / npad), col = static_cast<int>(idx % npad);
    out[idx] = col < n ? x[static_cast<long long>(col) * kg + g] : 0;
}

// Tiles of `bits` -> bf16 rows [n][k]: out = bf16(s * q + b), the product and the sum each rounded in
// fp32 (the fallback the prefill path runs through cuBLASLt, and the GEMV's oracle input).
// One thread an element; total = n * k.
extern "C" __global__ void __launch_bounds__(256)
    affine_n_dequant_bf16_ada(const uint32_t* __restrict__ w, const uint16_t* __restrict__ scales, const uint16_t* __restrict__ biases,
                              int n, int k, int npad, int bits, __nv_bfloat16* __restrict__ out, long long total) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const long long row = idx / k;
    const int kk = static_cast<int>(idx % k);
    const int kg = k / 64, g = kk / 64, in = kk % 64;
    const long long t = row / 64;
    const int rr = static_cast<int>(row % 64);
    const int kt = in / 16, rem = in % 16, h = rem / 8, c = (rem % 8) / 2, e = rem % 2;
    const int lane = (rr % 8) * 4 + c;
    uint32_t q = 0;
    if (bits == 8) {
        const uint32_t word = w[(((t * kg + g) * 8 + rr / 8) * 32 + lane) * 4 + kt];
        q = (word >> (8 * (h * 2 + e))) & 0xFFu;
    } else {
        const int wn = rr / 16, j = (rr / 8) % 2, i = j * 8 + kt * 2 + h;
        const uint32_t* lw = w + ((t * kg + g) * 128 + wn * 32 + lane) * bits;
        for (int at = 0; at < bits; ++at) {
            const int sb = i * bits + at;
            q |= ((lw[sb / 16] >> (sb % 16 + 16 * e)) & 1u) << at;
        }
    }
    const uint16_t sbits = scales[static_cast<long long>(g) * npad + row], bbits = biases[static_cast<long long>(g) * npad + row];
    const float s = __uint_as_float(static_cast<uint32_t>(sbits) << 16), b = __uint_as_float(static_cast<uint32_t>(bbits) << 16);
    out[idx] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(s, static_cast<float>(q)), b));
}
