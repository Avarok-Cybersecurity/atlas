// SPDX-License-Identifier: AGPL-3.0-only AND Apache-2.0
// Portions copied and adapted from TensorFold (https://github.com/ashhart/TensorFold), commit 3885dce,
// zig/kernels/cuda/affine4_pack.cu:1-36 (tf_pack_dense, tf_transpose_pad16) and
// zig/kernels/cuda/qwen35.cu:12-23 and 184-190 (group_sum, tf_q35_sums), and translated from
// zig/src/core/affine4_host.zig:7-12 and 40-66 (f32of, bf16of, quantizeMlx).
// Copyright 2026 TensorFold contributors. Licensed under the Apache License, Version 2.0; code written
// before TensorFold 0.6.0 is also available under the MIT License. See TENSORFOLD-ATTRIBUTION.md.
// Changes for Atlas: tf_pack_dense generalized to groups of 32, 64 and 128 (GS/32 words a lane a group;
// at GS = 64 the layout and every bit are upstream's) and renamed affine4_pack_tiles; tf_transpose_pad16
// renamed affine_transpose_pad16; the sums kernel takes the group size and drops the device row counts P;
// quantizeMlx runs on the GPU one warp a group (warp min/max, f32 math through __f*_rn, round half away
// from zero, bf16 round to nearest even), reports the first bad group through atomicMin instead of
// returning it, canonicalizes a -0.0 group edge to +0.0, and gains an optional clip search (an Atlas
// addition, off at clip = 0); the dequantizer from the tiled layout to row-major BF16 is new.

// Ada (sm_89) load-time and activation-side kernels of the affine 4-bit route. Module affine_pack_ada.
//
//   affine4_quantize_bf16_g<GS>  BF16 [n, k] -> MLX affine 4-bit: words [n, k/8] u32 (low nibble first),
//                                scales and biases [n, k/GS] bf16; `bad` gets the first group whose range,
//                                scale or bias is not finite (it must hold 0xFFFFFFFF on entry).
//                                One warp a group, 128 threads a block.
//   affine4_pack_tiles           MLX words -> the lane GEMV's tiles [npad/64][kg][8][32][GS/32] u32.
//   affine_transpose_pad16       (n, kg) 16-bit -> (kg, npad), zero columns past n (scales, biases).
//   affine_group_sums            per row, per GS-input group: the sum of the bf16 inputs in order (fp32).
//   affine4_dequant_bf16         tiled codes, scales, biases -> BF16 [n, k] = bf16(s * q + b).
//
// Every one is bit-exact against its host mirror (crates/spark-model/src/port/affine/reference.rs); the
// microtest examples/affine_ada_microtest.rs compares them byte for byte.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace avarok_affine_pack {

constexpr unsigned FULL = 0xFFFFFFFFu;

__device__ __forceinline__ float f32of(uint16_t b) { return __uint_as_float(static_cast<uint32_t>(b) << 16); }

// The nearest bf16, ties to even; NaN stays NaN (upstream bf16of, bit for bit).
__device__ __forceinline__ uint16_t bf16of(float x) {
    const uint32_t u = __float_as_uint(x);
    if (isnan(x)) return static_cast<uint16_t>((u >> 16) | 0x40u);
    return static_cast<uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

// MLX's scale and bias for the range [lo, hi] (affine4_host.zig:48-56). false: not representable.
__device__ __forceinline__ bool mlx_params(float lo, float hi, float& s, float& b) {
    const float step = fmaxf(__fdiv_rn(__fsub_rn(hi, lo), 15.0f), 1e-7f);
    if (!isfinite(step)) return false;
    const bool low_edge = fabsf(lo) > fabsf(hi);
    const float sg = low_edge ? step : -step;
    const float edge = low_edge ? lo : hi;
    const float q0 = roundf(__fdiv_rn(edge, sg));
    s = q0 == 0.0f ? sg : __fdiv_rn(edge, q0);
    b = q0 == 0.0f ? 0.0f : edge;
    return isfinite(s) && s != 0.0f && isfinite(b);
}

// The code of x under the UNROUNDED f32 scale and bias, as MLX computes it.
__device__ __forceinline__ uint32_t code_of(float x, float s, float b) {
    const float q = roundf(__fdiv_rn(__fsub_rn(x, b), s));
    return static_cast<uint32_t>(fminf(fmaxf(q, 0.0f), 15.0f));
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = __fadd_rn(v, __shfl_xor_sync(FULL, v, off));
    return v;
}

// One GS-input group a warp. Lane l holds inputs l * V .. l * V + V - 1 of the group (V = GS / 32).
template <int GS>
__device__ __forceinline__ void quantize_group(const __nv_bfloat16* __restrict__ src, int n, int k, int clip,
                                               uint32_t* __restrict__ words, uint16_t* __restrict__ scales,
                                               uint16_t* __restrict__ biases, unsigned* __restrict__ bad) {
    constexpr int V = GS / 32;
    constexpr int L = 8 / V;  // lanes a u32 word spans
    const int lane = threadIdx.x & 31;
    const long long gid = static_cast<long long>(blockIdx.x) * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int kg = k / GS;
    if (gid >= static_cast<long long>(n) * kg) return;  // uniform across the warp
    const long long row = gid / kg;
    const int g = static_cast<int>(gid % kg);
    const __nv_bfloat16* p = src + row * k + static_cast<long long>(g) * GS + lane * V;
    float v[V];
    bool finite = true;
#pragma unroll
    for (int j = 0; j < V; ++j) {
        v[j] = __bfloat162float(p[j]);
        finite = finite && isfinite(v[j]);
    }
    float s = 0.0f, b = 0.0f;
    bool ok = __all_sync(FULL, finite);
    if (ok) {
        float lo = v[0], hi = v[0];
#pragma unroll
        for (int j = 1; j < V; ++j) {
            lo = fminf(lo, v[j]);
            hi = fmaxf(hi, v[j]);
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            lo = fminf(lo, __shfl_xor_sync(FULL, lo, off));
            hi = fmaxf(hi, __shfl_xor_sync(FULL, hi, off));
        }
        lo = __fadd_rn(lo, 0.0f);  // -0.0 -> +0.0, so the result never depends on which zero won
        hi = __fadd_rn(hi, 0.0f);
        ok = mlx_params(lo, hi, s, b) && isfinite(f32of(bf16of(s))) && isfinite(f32of(bf16of(b)));
        // Clip search (Atlas addition): shrink [lo, hi] about its midpoint by 1 - 0.05 i, i = 1..clip, and
        // keep the candidate with the least squared error of the STORED (bf16) dequantization. Ties keep
        // the earlier candidate, so clip = 0 and a group no candidate improves are plain MLX.
        if (ok && clip > 0) {
            auto err_of = [&](float cs, float cb) {
                const float sb = f32of(bf16of(cs)), bb = f32of(bf16of(cb));
                float e = 0.0f;
#pragma unroll
                for (int j = 0; j < V; ++j) {
                    const float q = static_cast<float>(code_of(v[j], cs, cb));
                    const float d = __fsub_rn(__fadd_rn(__fmul_rn(sb, q), bb), v[j]);
                    e = __fadd_rn(e, __fmul_rn(d, d));
                }
                return warp_sum(e);
            };
            float best = err_of(s, b);
            const float mid = __fmul_rn(__fadd_rn(lo, hi), 0.5f);
            const float half_w = __fmul_rn(__fsub_rn(hi, lo), 0.5f);
            for (int i = 1; i <= clip; ++i) {
                const float ratio = __fsub_rn(1.0f, __fmul_rn(0.05f, static_cast<float>(i)));
                const float h = __fmul_rn(half_w, ratio);
                float cs, cb;
                if (!mlx_params(__fsub_rn(mid, h), __fadd_rn(mid, h), cs, cb)) continue;
                if (!isfinite(f32of(bf16of(cs))) || !isfinite(f32of(bf16of(cb)))) continue;
                const float e = err_of(cs, cb);
                if (e < best) {
                    best = e;
                    s = cs;
                    b = cb;
                }
            }
        }
    }
    uint32_t bits = 0;
    if (ok) {
#pragma unroll
        for (int j = 0; j < V; ++j) bits |= code_of(v[j], s, b) << (4 * ((lane * V + j) % 8));
    } else {
        s = 0.0f;
        b = 0.0f;
        if (lane == 0) atomicMin(bad, static_cast<unsigned>(gid));
    }
#pragma unroll
    for (int off = 1; off < L; off <<= 1) bits |= __shfl_xor_sync(FULL, bits, off);
    if (lane % L == 0) words[(row * k + static_cast<long long>(g) * GS) / 8 + lane / L] = bits;
    if (lane == 0) {
        scales[gid] = ok ? bf16of(s) : 0;
        biases[gid] = ok ? bf16of(b) : 0;
    }
}

}  // namespace avarok_affine_pack

#define AFFINE4_QUANTIZE_ENTRY(GS)                                                                            \
    extern "C" __global__ void __launch_bounds__(128) affine4_quantize_bf16_g##GS(                           \
        const __nv_bfloat16* __restrict__ src, int n, int k, int clip, uint32_t* __restrict__ words,         \
        uint16_t* __restrict__ scales, uint16_t* __restrict__ biases, unsigned* __restrict__ bad) {          \
        avarok_affine_pack::quantize_group<GS>(src, n, k, clip, words, scales, biases, bad);                 \
    }

AFFINE4_QUANTIZE_ENTRY(32)
AFFINE4_QUANTIZE_ENTRY(64)
AFFINE4_QUANTIZE_ENTRY(128)

// MLX (n, k/8) words -> the tiled lane layout [npad/64][kg][8][32][gs/32] (one thread a tiled word).
// Word (t, g, j, lane, v) holds, at nibble p, the code of row t*64 + j*8 + lane/4 and input
// g*gs + 32*v + 2*(lane%4) + {0, 8, 16, 24, 1, 9, 17, 25}[p]: the B-fragment order of m16n8k16.
extern "C" __global__ void affine4_pack_tiles(const uint32_t* __restrict__ w, int n, int k8, int kg, int gs,
                                              uint32_t* __restrict__ out, long long total) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int V = gs / 32;
    const int v = static_cast<int>(idx % V);
    const int lane = static_cast<int>((idx / V) & 31), j = static_cast<int>((idx / (V * 32)) & 7);
    const long long tg = idx / (V * 256);
    const int g = static_cast<int>(tg % kg);
    const long long t = tg / kg;
    const long long row = t * 64 + j * 8 + (lane >> 2);
    const int c = lane & 3;
    const int offs[8] = {0, 8, 16, 24, 1, 9, 17, 25};
    uint32_t word = 0;
    if (row < n) {
#pragma unroll
        for (int p = 0; p < 8; ++p) {
            const int input = g * gs + 32 * v + 2 * c + offs[p];
            const uint32_t src = w[row * k8 + (input >> 3)];
            word |= ((src >> (4 * (input & 7))) & 0xFu) << (4 * p);
        }
    }
    out[idx] = word;
}

// (n, kg) 16-bit -> (kg, npad) with zero columns past n: the tiled scales' and biases' layout.
extern "C" __global__ void affine_transpose_pad16(const uint16_t* __restrict__ x, int n, int kg, int npad,
                                                  uint16_t* __restrict__ out) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<long long>(kg) * npad) return;
    const int g = static_cast<int>(idx / npad), col = static_cast<int>(idx % npad);
    out[idx] = col < n ? x[static_cast<long long>(col) * kg + g] : 0;
}

// gs-group sums of bf16 rows k wide (x rows contiguous, 16-byte aligned): xs[row * (k / gs) + g] is the
// in-order fp32 sum of x[row, g*gs .. g*gs + gs - 1]. Grid (rows, ceil(kg / 64)), 64 threads, a thread a
// group.
extern "C" __global__ void __launch_bounds__(64) affine_group_sums(const __nv_bfloat16* __restrict__ x, float* __restrict__ xs,
                                                                   int rows, int k, int gs) {
    const long long row = blockIdx.x;
    const int kg = k / gs;
    const int g = blockIdx.y * blockDim.x + threadIdx.x;
    if (row >= rows || g >= kg) return;
    const __nv_bfloat16* p = x + row * k + static_cast<long long>(g) * gs;
    float s = 0.0f;
    for (int j = 0; j < gs; j += 8) {
        const uint4 w = *reinterpret_cast<const uint4*>(p + j);
        const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&w);
#pragma unroll
        for (int i = 0; i < 8; ++i) s = __fadd_rn(s, __bfloat162float(e[i]));
    }
    xs[row * kg + g] = s;
}

// BF16 [n, k] = bf16(s * q + b) from the tiled layout (the inverse of affine4_pack_tiles' mapping), one
// thread an element. The fallback of a declined prompt GEMM and the oracle of the microtest.
extern "C" __global__ void affine4_dequant_bf16(const uint32_t* __restrict__ w, const uint16_t* __restrict__ scales,
                                                const uint16_t* __restrict__ biases, int n, int k, int npad, int gs,
                                                uint16_t* __restrict__ out) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<long long>(n) * k) return;
    const long long r = idx / k;
    const int col = static_cast<int>(idx % k);
    const int kg = k / gs, V = gs / 32;
    const int g = col / gs, ic = col % gs;
    const int v = ic / 32, rem = ic % 32;
    const int c = (rem % 8) / 2, p = rem / 8 + 4 * (rem % 2);
    const int lane = static_cast<int>(r % 8) * 4 + c, j = static_cast<int>((r % 64) / 8);
    const long long t = r / 64;
    const long long word = (((t * kg + g) * 8 + j) * 32 + lane) * V + v;
    const float q = static_cast<float>((w[word] >> (4 * p)) & 0xFu);
    const long long sb = static_cast<long long>(g) * npad + r;
    const float val = __fadd_rn(__fmul_rn(avarok_affine_pack::f32of(scales[sb]), q), avarok_affine_pack::f32of(biases[sb]));
    out[idx] = avarok_affine_pack::bf16of(val);
}
