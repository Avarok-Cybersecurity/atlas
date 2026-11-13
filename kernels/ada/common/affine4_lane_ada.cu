// SPDX-License-Identifier: AGPL-3.0-only AND Apache-2.0
// Portions copied and adapted from TensorFold (https://github.com/ashhart/TensorFold), commit 3885dce,
// zig/kernels/cuda/lane_gemv.cu:1-346.
// Copyright 2026 TensorFold contributors. Licensed under the Apache License, Version 2.0; code written
// before TensorFold 0.6.0 is also available under the MIT License. See TENSORFOLD-ATTRIBUTION.md.
// Changes for Atlas: namespace tf_lane_gemv renamed avarok_affine_lane and the entry points renamed
// affine4_lane_*; the group size is a template argument instantiated at 32, 64 and 128 (upstream
// instantiates the lane body at 64 only, and its group-32 decode tile is qmm.cu's qmm_kernel instance
// at qmm.cu:227-231, replaced here by the same lane body at GS = 32); gemv_kernel and many_body share one
// body that takes (block, blocks); the PDL grid_wait/grid_launch calls are dropped (sm_90 only); a Seg
// carries its projection's own ticket array instead of an offset into one shared array, so every weight
// owns its tickets; every entry point is an extern "C" wrapper.

// Ada (sm_89) affine 4-bit decode and verify GEMV: out[r, :] = x[r, :] . W^T for r < M <= 16.
//
// WEIGHT FORMAT. MLX affine 4-bit: per group of GS inputs a bf16 scale s and bias b, value = s * q + b,
// q in 0..15. The codes are tiled [npad/64][K/GS][8][32][GS/32] u32 (affine_pack_ada.cu,
// affine4_pack_tiles); scales and biases are [K/GS][npad] bf16 with zero columns past n.
//
// ARITHMETIC (upstream's, unchanged). The codes enter a BF16 m16n8k16 MMA as exact bf16 integers with no
// scale applied; per group the epilogue applies the affine identity
//     acc = fma(sum_g(x), b, fma(x . q, s, acc))
// where sum_g(x) is the group's input sum (affine_pack_ada.cu, affine_group_sums). Groups are visited in
// order. K is cut into sk slices fixed by the WEIGHT's shape (`split_k`, crates/spark-model/src/port/
// affine/weight.rs), never by M; slice partials start from zero and are added in slice order.
//
// ROW INVARIANCE. Row r's bits do not depend on M: each output row is computed from its own input row,
// with the same group order, the same MMA K16 order, the same epilogue chain and the same slice order at
// every M (rows past M are zero-filled once and never stored). An M = 1 decode row and an M = 4 verify row
// are therefore bit-identical, which is what lets MTP drafted output equal plain decoding on this route.
//
// ENTRY POINTS (module affine4_lane_ada; GS in {32, 64, 128}; 128 threads, one m16 x 64 tile a CTA):
//   affine4_lane_gemv_g<GS>   every K slice of a column tile in one CTA, CTAs looping over the tiles
//                             (grid: any; dynamic shared LaneTile<GS,16,64,1,4,STAGES_GEMV>::SMEM)
//   affine4_lane_split_g<GS>  one CTA a (tile, slice); the tile's last CTA sums the slices in order
//                             through a ticket (grid: tiles * sk; needs sk > 1, work and zeroed tickets)
//   affine4_lane_seg_g<GS>    up to four projections of one input in one launch, each as _split
//                             (or as one slice written directly when its sk is 1)
//   affine4_lane_many_g<GS>   up to four projections of one input, each as _gemv on its own CTA range
// All four give the same bits for the same weight and input. Tickets return to zero after every launch.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "affine_frag_ada.cuh"

namespace avarok_affine_lane {

using namespace avarok_affine;

// One projection: tiled codes, (kg, npad) scales and biases, the bf16 output, its column count, padded
// column count, K slices and 64-column tiles. Mirrored byte for byte by `Part` in
// crates/spark-model/src/port/affine/kernels.rs (48 bytes).
struct Part {
    const uint32_t* w;
    const __nv_bfloat16* scales;
    const __nv_bfloat16* biases;
    void* out;
    int n, npad, sk, tiles;
};

template <int GS, int BN, int STAGES>
using GemvTile = LaneTile<GS, 16, BN, 1, 4, STAGES>;

// Visit mapped groups in order, preserving the MMA and fma chain before after(i).
template <int GS, int BN, int STAGES, typename Map, typename After>
__device__ __forceinline__ void gemv_items(const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs, const Part& P, int M,
                                           int K, int ldx, int items, Map map, float (&acc)[GemvTile<GS, BN, STAGES>::NT][4],
                                           After after) {
    using T = GemvTile<GS, BN, STAGES>;
    static_assert(T::MT == 1 && T::THREADS == 128, "one m16 row tile, four column warps");
    constexpr int NT = T::NT;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, wn = tid >> 5;
    const int KG = K / GS;
    auto stage = [&](int s) { return buf + s * T::STAGE; };
    const int rows = min(16, M);
    constexpr int TILE_BYTES = 64 * GS / 2;
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
        const unsigned char* wb = reinterpret_cast<const unsigned char*>(P.w) +
                                  static_cast<size_t>(n0 / 64) * KG * TILE_BYTES + (n0 % 64) * GS / 2;
#pragma unroll
        for (int j = 0; j < WC; ++j)
            if (wok[j]) cp16(p + wdst[j], wb + woff[j] + static_cast<size_t>(g) * TILE_BYTES);
#pragma unroll
        for (int j = 0; j < SC; ++j)
            if (sok[j]) cp16(p + sdst[j], (sbias[j] ? P.biases : P.scales) + n0 + soff[j] + static_cast<size_t>(g) * P.npad);
    };
    uint32_t mask;
    asm volatile("mov.b32 %0, 0x000F000F;\n" : "=r"(mask));

    // Rows past M: zero once in every stage (the loads never touch them).
    for (int c = tid; c < 16 * T::CHUNKS; c += T::THREADS)
        if (c / T::CHUNKS >= rows)
#pragma unroll
            for (int s = 0; s < STAGES; ++s)
                *reinterpret_cast<uint4*>(stage(s) + c / T::CHUNKS * T::ROW + c % T::CHUNKS * 16) = uint4{};
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < items) load_w(s, s);
        commit();
    }
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < items) load_x(s, s);
        commit();
    }
    for (int it = 0; it < items; ++it) {
        wait<STAGES - 2>();
        __syncthreads();
        const int next = it + STAGES - 1;
        if (next < items) {
            load_x(next % STAGES, next);
            load_w(next % STAGES, next);
        }
        commit();
        const unsigned char* p = stage(it % STAGES);
        const uint32_t* pw = reinterpret_cast<const uint32_t*>(p + T::X);
        const __nv_bfloat16* ps = reinterpret_cast<const __nv_bfloat16*>(p + T::X + T::W);
        const float* px = reinterpret_cast<const float*>(p + T::X + T::W + 2 * T::S);
        uint32_t words[NT][GS / 32];
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int v = 0; v < GS / 32; ++v) words[j][v] = pw[((wn * NT + j) * 32 + lane) * (GS / 32) + v];
        float d[NT][4];
#pragma unroll
        for (int kt = 0; kt < GS / 16; ++kt) {
            uint32_t a[4];
            const int r = (lane & 7) + ((lane >> 3) & 1) * 8, ch = kt * 2 + (lane >> 4);
            ldmatrix4(a, p + r * T::ROW + swz<T::CHUNKS>(r, ch) * 16);
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                const uint32_t b0 = pairm(words[j][kt / 2], (kt & 1) * 8, mask);
                const uint32_t b1 = pairm(words[j][kt / 2], (kt & 1) * 8 + 4, mask);
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

// Rows below M of the 64 columns at n0 as bf16, from fp32 pairs got(j, h) in the accumulator's layout.
template <int BN, int NT, typename Got>
__device__ __forceinline__ void gemv_store(const Part& P, int M, int n0, Got got) {
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

// Column CTAs keep every K slice of their tile, summed in slice order. `block` of `blocks` CTAs walks
// tiles block, block + blocks, ... (upstream gemv_kernel with blockIdx/gridDim, and many_body with a
// projection-local index: one body).
template <int GS, int BN, int STAGES>
__device__ __forceinline__ void gemv_body(const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs, const Part& P,
                                          int M, int K, int ldx, int block, int blocks) {
    constexpr int NT = GemvTile<GS, BN, STAGES>::NT;
    const int KG = K / GS, per = KG / P.sk;
    const int mine = P.tiles > block ? (P.tiles - 1 - block) / blocks + 1 : 0;
    auto tile_of_item = [&](int i) { return block + (i / KG) * blocks; };
    float acc[NT][4], tot[NT][4];
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[j][e] = tot[j][e] = 0.0f;
    gemv_items<GS, BN, STAGES>(x, xs, P, M, K, ldx, mine * KG, [&](int i) { return make_int2(i % KG, tile_of_item(i) * BN); }, acc,
                               [&](int it) {
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
                                   gemv_store<BN, NT>(P, M, tile_of_item(it) * BN, [&](int j, int h, int, int) {
                                       return make_float2(tot[j][2 * h], tot[j][2 * h + 1]);
                                   });
                               });
}

// Each CTA writes one tile/slice partial; the tile's last CTA sums the slices in their original order.
// ONE: a projection with a single slice stores its sum directly (the in-order sum of one partial).
template <int GS, int BN, int STAGES, bool ONE>
__device__ __forceinline__ void gemv_split(const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs, const Part& P, int M,
                                           int K, int ldx, float* __restrict__ work, unsigned* __restrict__ tickets, int tile,
                                           int slice) {
    constexpr int NT = GemvTile<GS, BN, STAGES>::NT;
    __shared__ unsigned ticket;
    const int tid = threadIdx.x, lane = tid & 31, wn = tid >> 5;
    const int per = K / GS / P.sk, g0 = slice * per;
    float acc[NT][4];
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[j][e] = 0.0f;
    gemv_items<GS, BN, STAGES>(x, xs, P, M, K, ldx, per, [&](int i) { return make_int2(g0 + i, tile * BN); }, acc, [](int) {});
    if (ONE && P.sk == 1) {
        gemv_store<BN, NT>(P, M, tile * BN, [&](int j, int h, int, int) { return make_float2(acc[j][2 * h], acc[j][2 * h + 1]); });
        return;
    }
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
    gemv_store<BN, NT>(P, M, tile * BN, [&](int, int, int r, int c) {
        float t0 = __ldcg(base + r * BN + c), t1 = __ldcg(base + r * BN + c + 1);
        for (int s = 1; s < P.sk; ++s) {
            t0 = t0 + __ldcg(base + (s * 16 + r) * BN + c);
            t1 = t1 + __ldcg(base + (s * 16 + r) * BN + c + 1);
        }
        return make_float2(t0, t1);
    });
    if (tid == 0) tickets[tile] = 0;
}

constexpr int SEGS = 4;

// One projection of a shared input: its blocks start at `first`, its partials at work + work_at * 16 * 64
// floats, and `tickets` is the projection's own per-tile ticket array. Mirrored by `Seg` in kernels.rs
// (64 bytes).
struct Seg {
    Part p;
    unsigned* tickets;
    int first, work_at;
};

struct Segs {
    Seg s[SEGS];
    int count;
};

// One member of a _many launch: its persistent CTAs are blocks first .. first + blocks - 1. Mirrored by
// `ManyPart` in kernels.rs (56 bytes).
struct ManyPart {
    Part p;
    int first, blocks;
};

struct ManyParts {
    ManyPart p[SEGS];
    int count;
};

template <int GS, int STAGES>
__device__ __forceinline__ void seg_body(const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs, const Segs& S, int M,
                                         int K, int ldx, float* __restrict__ work) {
    Seg sg = S.s[0];  // in registers: the loads index it every group
#pragma unroll
    for (int i = 1; i < SEGS; ++i)
        if (i < S.count && (int)blockIdx.x >= S.s[i].first) sg = S.s[i];
    const int b = (int)blockIdx.x - sg.first;
    gemv_split<GS, 64, STAGES, true>(x, xs, sg.p, M, K, ldx, work + static_cast<size_t>(sg.work_at) * 16 * 64, sg.tickets,
                                     b / sg.p.sk, b % sg.p.sk);
}

template <int GS, int STAGES>
__device__ __forceinline__ void many_body(const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs, const ManyParts& parts,
                                          int M, int K, int ldx) {
    ManyPart member = parts.p[0];
#pragma unroll
    for (int i = 1; i < SEGS; ++i)
        if (i < parts.count && (int)blockIdx.x >= parts.p[i].first) member = parts.p[i];
    gemv_body<GS, 64, STAGES>(x, xs, member.p, M, K, ldx, (int)blockIdx.x - member.first, member.blocks);
}

}  // namespace avarok_affine_lane

// Stage counts. _gemv and _many: LaneTile<GS,16,64,1,4,S>::SMEM of 18,944 (GS 32, S 8), 35,328 (GS 64, S 8)
// and 34,048 (GS 128, S 4) bytes. _split and _seg: 9,472 (32, 4), 17,664 (64, 4) and 25,536 (128, 3).
// The host computes the same numbers (`lane_smem`, crates/spark-model/src/port/affine/weight.rs).
#define AFFINE4_LANE_ENTRIES(GS, GEMV_STAGES, SPLIT_STAGES)                                                          \
    extern "C" __global__ void __launch_bounds__(128) affine4_lane_gemv_g##GS(                                      \
        const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs,                                          \
        const __grid_constant__ avarok_affine_lane::Part P, int M, int K, int ldx) {                                \
        avarok_affine_lane::gemv_body<GS, 64, GEMV_STAGES>(x, xs, P, M, K, ldx, (int)blockIdx.x, (int)gridDim.x);   \
    }                                                                                                               \
    extern "C" __global__ void __launch_bounds__(128) affine4_lane_split_g##GS(                                     \
        const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs,                                          \
        const __grid_constant__ avarok_affine_lane::Part P, int M, int K, int ldx, float* __restrict__ work,        \
        unsigned* __restrict__ tickets) {                                                                           \
        avarok_affine_lane::gemv_split<GS, 64, SPLIT_STAGES, false>(x, xs, P, M, K, ldx, work, tickets,             \
                                                                    (int)blockIdx.x / P.sk, (int)blockIdx.x % P.sk); \
    }                                                                                                               \
    extern "C" __global__ void __launch_bounds__(128) affine4_lane_seg_g##GS(                                       \
        const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs,                                          \
        const __grid_constant__ avarok_affine_lane::Segs S, int M, int K, int ldx, float* __restrict__ work) {      \
        avarok_affine_lane::seg_body<GS, SPLIT_STAGES>(x, xs, S, M, K, ldx, work);                                  \
    }                                                                                                               \
    extern "C" __global__ void __launch_bounds__(128) affine4_lane_many_g##GS(                                      \
        const __nv_bfloat16* __restrict__ x, const float* __restrict__ xs,                                          \
        const __grid_constant__ avarok_affine_lane::ManyParts parts, int M, int K, int ldx) {                       \
        avarok_affine_lane::many_body<GS, GEMV_STAGES>(x, xs, parts, M, K, ldx);                                    \
    }

AFFINE4_LANE_ENTRIES(32, 8, 4)
AFFINE4_LANE_ENTRIES(64, 8, 4)
AFFINE4_LANE_ENTRIES(128, 4, 3)
