// SPDX-License-Identifier: AGPL-3.0-only AND MIT
// After ExLlamaV3 (turboderp-org/exllamav3), MIT, Copyright (c) 2025 Turboderp, via TensorFold
// zig/kernels/cuda/exl3_linear.cu:1-477 (Apache-2.0, ashhart/TensorFold, commit 3885dce): finish 71-81,
// load_step and step_lane_words 83-101, linear_body 143-280 and the instantiation macros 430-446. See
// EXL3-ATTRIBUTION.md and LICENSES/ExLlamaV3-MIT.txt.
// Changes for Atlas: namespace tf_exl3 renamed avarok_exl3 and the entry points renamed
// exl3_ada_linear_<K2>_<CB>_<WK>; the by-value LinearArgs struct is replaced by plain parameters; the
// output is always bf16 (the store4 dtype switch and y_dtype are dropped) and there is no bias (the host
// refuses biased groups, because the W4Linear contract has none); the PDL grid_wait/grid_launch calls are
// dropped (sm_90 only); only WK = 4 and 8 are instantiated (the warps the shape plan can pick); the
// mid-M (mpg), linear_wc and projection-group kernels are not ported: 17-128 rows run linear_body, which
// gives the same bits as those kernels by construction upstream. The input rotation lives in
// exl3_hadamard_ada.cu.

// Ada (sm_89) EXL3 decode-row linear: y[r, :] = (H(xh[r, :] @ Wq) / sqrt(128)) * svh for r < M <= 128,
// where xh is the Hadamard-rotated fp16 input of exl3_ada_rot_in_bf16 (exl3_hadamard_ada.cu).
//
// GEOMETRY. Grid (N / 128) * SK blocks, split-major (block b: column block b % NB, K split b / NB), WK warps
// a block, dynamic shared memory WK * min(M, 8) * 128 * 4 bytes (at most 32 KiB). Warp w of split s owns
// the contiguous k tiles [s * per_warp * WK + w * per_warp, +per_warp), per_warp = (K / 16) / SK / WK,
// which the host guarantees is a whole number. Each k step decodes the step's 8 tiles (128 columns) once
// and feeds 16-row passes of xh through mma.m16n8k16 (fp16 in, fp32 accumulate).
//
// SPLIT K. SK > 1 writes fp32 partials z [SK, M, N]; the block that arrives last on its (pass, column
// block) counter adds the partials in split order 0..SK-1, finishes, and resets the counter to zero, so
// the counters (M / 16 rounded up, times NB) are zero at rest and need no clearing between calls.
//
// ROW INVARIANCE. (SK, WK) come from the weight's shape alone (crates/spark-model/src/port/exl3/plan.rs),
// never from M. Row r's mma inputs are its own xh row, its warps' sums are added in warp order and its
// split partials in split order, so row r's bits do not depend on M: an MTP verify row equals the decode
// of the same token. Rows past M inside a pass read row M - 1 (in bounds) and are never stored.

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "exl3_decode_ada.cuh"

namespace avarok_exl3 {

__device__ __forceinline__ int imin(int a, int b) { return a < b ? a : b; }

__device__ __forceinline__ void store4_bf16(__nv_bfloat16* p, size_t i, const float (&v)[4]) {
    __nv_bfloat162 a = __floats2bfloat162_rn(v[0], v[1]), b = __floats2bfloat162_rn(v[2], v[3]);
    uint2 u;
    u.x = *reinterpret_cast<uint32_t*>(&a);
    u.y = *reinterpret_cast<uint32_t*>(&b);
    *reinterpret_cast<uint2*>(p + i) = u;
}

// The finished outputs of one row's 128 columns from their fp32 sums (4 a lane): H / sqrt(128), * svh.
__device__ __forceinline__ void finish(float (&v)[4], int lane, const half* svh, int col) {
    fwht128(v, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = v[j] * HAD_SCALE * __half2float(__ldg(svh + col + j));
}

// A k step's words for one lane: the step is 8 tiles back to back, each lane takes the same slice of every tile.
template <int K2>
__device__ __forceinline__ void load_step(const uint32_t* step, int lane, uint32_t (&raw)[8 * lane_words<K2>()]) {
    constexpr int TW = tile_words<K2>(), LW = lane_words<K2>();
    int word, offset;
    lane_start<K2>(lane, word, offset);
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int q = 0; q < LW; ++q) raw[j * LW + q] = __ldg(step + j * TW + (word + q) % TW);
}

// Tile j's lane words from a loaded step.
template <int K2>
__device__ __forceinline__ void step_lane_words(const uint32_t (&raw)[8 * lane_words<K2>()], int j,
                                                uint32_t (&w)[lane_words<K2>()]) {
#pragma unroll
    for (int q = 0; q < lane_words<K2>(); ++q) w[q] = raw[j * lane_words<K2>() + q];
}

// One program: (N / 128) * SK blocks, split-major like a (N / 128, SK) grid, WK warps each.
template <int K2, int CB, int WK>
__device__ __forceinline__ void linear_body(const half* __restrict__ xh, const uint32_t* __restrict__ t,
                                            long long stride_k, long long stride_nb, const half* __restrict__ svh,
                                            __nv_bfloat16* __restrict__ y, float* z, int* counters, int N, int SK,
                                            int M, int K) {
    constexpr int TW = tile_words<K2>(), LW = lane_words<K2>();
    constexpr bool PF = K2 <= 12;                             // the next k step's words load while this one decodes
    constexpr int SR = 8 * LW;
    extern __shared__ __align__(16) float red[];              // WK * RH * 128 floats
    __shared__ int last;
    const int NB = N >> 7;
    const int RH = imin(M, 8);                                // rows of red a warp
    const int bid = blockIdx.x;
    const int nb = bid % NB, split = bid / NB;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, tq = lane & 3;
    const int per_warp = (K >> 4) / SK / WK;
    const int kt0 = split * (per_warp * WK) + warp * per_warp;
    const uint32_t* tiles = t + nb * stride_nb;
    const int col0 = nb * 128;

    for (int m0 = 0, pass = 0; m0 < M; m0 += 16, ++pass) {
        const int R = imin(16, M - m0);

        float acc[8][2][4];
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

        // the walk's rows for the mma, clamped so rows past the pass read inside the buffer (their outputs are dropped)
        const int r0 = m0 + (g < R ? g : R - 1), r1 = m0 + (g + 8 < R ? g + 8 : R - 1);
        const half* x0 = xh + (size_t)r0 * K;
        const half* x1 = xh + (size_t)r1 * K;
        const uint32_t* tile = tiles + (size_t)kt0 * stride_k;
        uint32_t cur[PF ? SR : 1], nxt[PF ? SR : 1];
        if constexpr (PF) load_step<K2>(tile, lane, cur);

        // x's fragments of k step kt: rows g (and g + 8): [0] [2] k 2t.., 2t + 8..; [1] [3] the same of row g + 8
        auto load_x = [&](int kt, uint32_t (&u)[4]) {
            u[0] = __ldg(reinterpret_cast<const uint32_t*>(x0 + kt * 16 + 2 * tq));
            u[2] = __ldg(reinterpret_cast<const uint32_t*>(x0 + kt * 16 + 2 * tq + 8));
            u[1] = __ldg(reinterpret_cast<const uint32_t*>(x1 + kt * 16 + 2 * tq));
            u[3] = __ldg(reinterpret_cast<const uint32_t*>(x1 + kt * 16 + 2 * tq + 8));
        };
#pragma unroll 1
        for (int i = 0; i < per_warp; ++i) {
            uint32_t u[4];
            load_x(kt0 + i, u);
            if constexpr (PF) {
                if (i + 1 < per_warp) load_step<K2>(tile + (size_t)(i + 1) * stride_k, lane, nxt);
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                uint32_t w[LW];
                if constexpr (PF) step_lane_words<K2>(cur, j, w);
                else ldg_lane_words<K2>(tile + (size_t)i * stride_k + j * TW, lane, w);
                uint32_t b0[2], b1[2];
                decode_lane<K2, CB>(w, lane, b0, b1);
                mma16816(acc[j][0], u, b0);
                mma16816(acc[j][1], u, b1);
            }
            if constexpr (PF) {
#pragma unroll
                for (int q = 0; q < SR; ++q) cur[q] = nxt[q];
            }
        }

        // the warps' sums, added in warp order, rows 0-7 of the pass and then rows 8-15
        for (int rlo = 0; rlo < R; rlo += 8) {
            const int rn = imin(R - rlo, 8);
            __syncthreads();                         // red is reused by every half and pass
            if (g < RH) {
#pragma unroll
                for (int i = 0; i < 8; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int col = i * 16 + h * 8 + 2 * tq;
                        *reinterpret_cast<float2*>(red + (warp * RH + g) * 128 + col) =
                            rlo ? make_float2(acc[i][h][2], acc[i][h][3]) : make_float2(acc[i][h][0], acc[i][h][1]);
                    }
            }
            __syncthreads();

            if (SK == 1) {
                for (int r = warp; r < rn; r += WK) {
                    float v[4];
                    const float4 u = *reinterpret_cast<const float4*>(red + r * 128 + 4 * lane);
                    v[0] = u.x; v[1] = u.y; v[2] = u.z; v[3] = u.w;
#pragma unroll
                    for (int w = 1; w < WK; ++w) {
                        const float4 q = *reinterpret_cast<const float4*>(red + (w * RH + r) * 128 + 4 * lane);
                        v[0] += q.x; v[1] += q.y; v[2] += q.z; v[3] += q.w;
                    }
                    finish(v, lane, svh, col0 + 4 * lane);
                    store4_bf16(y, (size_t)(m0 + rlo + r) * N + col0 + 4 * lane, v);
                }
            } else {
                for (int idx = threadIdx.x; idx < rn * 32; idx += WK * 32) {
                    const int r = idx >> 5, c = 4 * (idx & 31);
                    float4 s = *reinterpret_cast<const float4*>(red + r * 128 + c);
#pragma unroll
                    for (int w = 1; w < WK; ++w) {
                        const float4 q = *reinterpret_cast<const float4*>(red + (w * RH + r) * 128 + c);
                        s.x += q.x; s.y += q.y; s.z += q.z; s.w += q.w;
                    }
                    *reinterpret_cast<float4*>(z + ((size_t)split * M + m0 + rlo + r) * N + col0 + c) = s;
                }
            }
        }
        if (SK > 1) {
            __threadfence();
            __syncthreads();
            if (threadIdx.x == 0) last = atomicAdd(counters + pass * NB + nb, 1) == SK - 1;
            __syncthreads();
            if (last) {
                __threadfence();
                for (int r = warp; r < R; r += WK) {
                    const size_t at = ((size_t)m0 + r) * N + col0 + 4 * lane;
                    float4 s = __ldcg(reinterpret_cast<const float4*>(z + at));
                    for (int q = 1; q < SK; ++q) {
                        const float4 u = __ldcg(reinterpret_cast<const float4*>(z + (size_t)q * M * N + at));
                        s.x += u.x; s.y += u.y; s.z += u.z; s.w += u.w;
                    }
                    float v[4] = {s.x, s.y, s.z, s.w};
                    finish(v, lane, svh, col0 + 4 * lane);
                    store4_bf16(y, (size_t)(m0 + r) * N + col0 + 4 * lane, v);
                }
                if (threadIdx.x == 0) counters[pass * NB + nb] = 0;   // every program of the block has arrived
            }
        }
        __syncthreads();                             // red is reused in the next pass
    }
}

}  // namespace avarok_exl3

// One extern "C" kernel per instantiation; the host resolves them by name
// (crates/spark-model/src/port/exl3/kernels.rs, `linear_entry`).
#define EXL3_ADA_LINEAR(K2, CB, WK)                                                                         \
    extern "C" __global__ void __launch_bounds__((WK) * 32) exl3_ada_linear_##K2##_##CB##_##WK(            \
        const half* __restrict__ xh, const uint32_t* __restrict__ t, long long stride_k, long long stride_nb, \
        const half* __restrict__ svh, __nv_bfloat16* __restrict__ y, float* z, int* counters, int n, int sk,  \
        int m, int k) {                                                                                     \
        avarok_exl3::linear_body<K2, CB, WK>(xh, t, stride_k, stride_nb, svh, y, z, counters, n, sk, m, k);  \
    }

// The warps the shape plan picks (4 or 8), for every width the format serves: whole bits (K2 even) on all
// three codebooks, and the mul1 half steps 1.5 to 6.5 bits (K2 = 3, 5, ..., 13); 7.5 bits is not a width.
// Every instantiation is written out at file scope, one (K2, CB) pair a line, rather than through nested
// width and codebook macros: the build's shadow detector (crates/avarok-kernels/build_shadow.rs) expands
// one level of instantiation macro, so a nested invocation would hide these entry points from it.
// codebook 0 (3inst)
EXL3_ADA_LINEAR(2, 0, 4) EXL3_ADA_LINEAR(2, 0, 8)
EXL3_ADA_LINEAR(4, 0, 4) EXL3_ADA_LINEAR(4, 0, 8)
EXL3_ADA_LINEAR(6, 0, 4) EXL3_ADA_LINEAR(6, 0, 8)
EXL3_ADA_LINEAR(8, 0, 4) EXL3_ADA_LINEAR(8, 0, 8)
EXL3_ADA_LINEAR(10, 0, 4) EXL3_ADA_LINEAR(10, 0, 8)
EXL3_ADA_LINEAR(12, 0, 4) EXL3_ADA_LINEAR(12, 0, 8)
EXL3_ADA_LINEAR(14, 0, 4) EXL3_ADA_LINEAR(14, 0, 8)
EXL3_ADA_LINEAR(16, 0, 4) EXL3_ADA_LINEAR(16, 0, 8)
// codebook 1 (mcg)
EXL3_ADA_LINEAR(2, 1, 4) EXL3_ADA_LINEAR(2, 1, 8)
EXL3_ADA_LINEAR(4, 1, 4) EXL3_ADA_LINEAR(4, 1, 8)
EXL3_ADA_LINEAR(6, 1, 4) EXL3_ADA_LINEAR(6, 1, 8)
EXL3_ADA_LINEAR(8, 1, 4) EXL3_ADA_LINEAR(8, 1, 8)
EXL3_ADA_LINEAR(10, 1, 4) EXL3_ADA_LINEAR(10, 1, 8)
EXL3_ADA_LINEAR(12, 1, 4) EXL3_ADA_LINEAR(12, 1, 8)
EXL3_ADA_LINEAR(14, 1, 4) EXL3_ADA_LINEAR(14, 1, 8)
EXL3_ADA_LINEAR(16, 1, 4) EXL3_ADA_LINEAR(16, 1, 8)
// codebook 2 (mul1), whole bits
EXL3_ADA_LINEAR(2, 2, 4) EXL3_ADA_LINEAR(2, 2, 8)
EXL3_ADA_LINEAR(4, 2, 4) EXL3_ADA_LINEAR(4, 2, 8)
EXL3_ADA_LINEAR(6, 2, 4) EXL3_ADA_LINEAR(6, 2, 8)
EXL3_ADA_LINEAR(8, 2, 4) EXL3_ADA_LINEAR(8, 2, 8)
EXL3_ADA_LINEAR(10, 2, 4) EXL3_ADA_LINEAR(10, 2, 8)
EXL3_ADA_LINEAR(12, 2, 4) EXL3_ADA_LINEAR(12, 2, 8)
EXL3_ADA_LINEAR(14, 2, 4) EXL3_ADA_LINEAR(14, 2, 8)
EXL3_ADA_LINEAR(16, 2, 4) EXL3_ADA_LINEAR(16, 2, 8)
// codebook 2 (mul1), half steps
EXL3_ADA_LINEAR(3, 2, 4) EXL3_ADA_LINEAR(3, 2, 8)
EXL3_ADA_LINEAR(5, 2, 4) EXL3_ADA_LINEAR(5, 2, 8)
EXL3_ADA_LINEAR(7, 2, 4) EXL3_ADA_LINEAR(7, 2, 8)
EXL3_ADA_LINEAR(9, 2, 4) EXL3_ADA_LINEAR(9, 2, 8)
EXL3_ADA_LINEAR(11, 2, 4) EXL3_ADA_LINEAR(11, 2, 8)
EXL3_ADA_LINEAR(13, 2, 4) EXL3_ADA_LINEAR(13, 2, 8)

#undef EXL3_ADA_LINEAR
