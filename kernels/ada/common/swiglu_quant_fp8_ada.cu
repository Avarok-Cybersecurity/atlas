// SPDX-License-Identifier: AGPL-3.0-only

// Ada (sm_89) fused SwiGLU + per-token-group FP8 E4M3 quantization: the input
// of the dense FFN's W8A8 down projection in ONE pass.
//
//   r[m, k]        = bf16( silu(gate[m, k]) * up[m, k] )
//   a_scale[m, g]  = max(max_{k in g} |r[m, k]| / 448, 1e-12)
//   out_fp8[m, k]  = e4m3_satfinite( clamp(r[m, k] / a_scale[m, g], +-448) )
//
// g walks K in groups of 128. It replaces the pair the W8A8 arm of
// `layers/dense_ffn.rs` launches between the up and down GEMMs:
//
//   moe_silu_mul(gate, up, gate)                      (kernels/gb10/common/moe_silu_mul.cu)
//   per_token_group_quant_fp8(gate, out_fp8, a_scale) (kernels/gb10/common/per_token_group_quant_fp8.cu)
//
// That pair writes the BF16 product to memory and reads it back: 4 B in,
// 2 B out, 2 B in, 1 B out per element. This kernel reads gate and up once and
// writes the FP8 byte, 4 B in and 1 B out, and the BF16 product never leaves
// registers. On the 4B FFN at a 3,921-token prefill that is 145 MB less DRAM
// traffic per layer. gb10's own fused `silu_mul_quant_fp8` (same file as
// `moe_silu_mul`) cannot stand in: it holds a row's values in a 16-group
// register array (K <= 2048), and the Holo intermediates are 3,584 and 9,216.
//
// ── NUMERICS: bit-identical to the pair, by construction ─────────────────
//
//   * The product is `moe_silu_mul`'s expression, token for token:
//     `1.0f / (1.0f + __expf(-g))`, then `g * sigmoid_g * u`, then
//     `__float2bfloat16`. Compiled with the same flags (this file and
//     moe_silu_mul.cu both sit in kernels/ada/common, so both get
//     common/KERNEL.toml's `--fmad=false` and the hardware flags), it lowers
//     to the same sequence: `mul.rn.f32` by -log2(e), `ex2.approx.f32`,
//     `add.rn.f32`, the IEEE reciprocal `rcp.rn.f32`, then two separate
//     `mul.rn.f32` (no FMA contraction under --fmad=false).
//   * The quantizer input is that BF16 value widened back to FP32, which is
//     exactly what `per_token_group_quant_fp8` reads from memory.
//   * Scale: `amax / 448.0f` (IEEE division), floored at `1e-12f`. Element:
//     `v / scale` (IEEE division, NOT a reciprocal multiply), the same
//     `fmaxf(fminf(v, 448), -448)` clamp, the same
//     `__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3)`.
//   * The max reduction tree differs from the shared kernel's (16-lane XOR
//     butterfly here, warp shuffle-down plus a 4-slot cross-warp combine
//     there) and may: `fmaxf` over absolute values is exact, associative and
//     commutative, and both trees are seeded with `0.0f`, so a NaN product is
//     dropped by both wherever it sits (PTX `max.f32` returns the non-NaN
//     operand). This is the argument, and the layout, of the Hopper quantizer
//     twin `kernels/hopper/common/fp8_act_quant_hopper.cu`.
//
// The SiLU half is only identical while the dense FFN's `act_mul` handle is
// the common `moe_silu_mul`. A model that shadows moe_silu_mul.cu (the
// swiglu_limit models on gb10 do) would change the unfused product and not
// this one; `dense_ffn_swiglu_quant_tests.rs` fails if any ada model
// directory shadows that file, and the host only selects this kernel for
// `FfnActivation::SiLU`.
//
// Grid: (M, ceil(K/128 / 8), 1)  Block: (128, 1, 1). 16 threads per 128-wide
// group, 8 elements per thread (one 16-byte load of gate and one of up), 8
// groups per CTA, no shared memory and no barrier. The group span is derived
// from gridDim.y, so any Y extent in 1..=K/128 covers every group once.
// Requires K % 128 == 0 (the host selector's clause; the down projection's
// W8A8 gate already demands it).

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define SQ_GROUP_K 128
#define SQ_E4M3_MAX 448.0f
#define SQ_LANES_PER_GROUP 16
#define SQ_GROUPS_PER_CTA 8
#define SQ_ELEMS_PER_THREAD 8

// Eight bf16 widened to f32: one `uint4` when the address allows it, else
// eight scalar loads of the same values in the same order.
__device__ __forceinline__ void sq_load8(const __nv_bfloat16* __restrict__ p, float* v) {
    if ((reinterpret_cast<uintptr_t>(p) & 15u) == 0) {
        const uint4 w = *reinterpret_cast<const uint4*>(p);
        const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(&w);
#pragma unroll
        for (int i = 0; i < SQ_ELEMS_PER_THREAD; i++) v[i] = __bfloat162float(b[i]);
    } else {
#pragma unroll
        for (int i = 0; i < SQ_ELEMS_PER_THREAD; i++) v[i] = __bfloat162float(p[i]);
    }
}

// Eight FP8 bytes: one `uint2` store when aligned.
__device__ __forceinline__ void sq_store8(unsigned char* __restrict__ p, const unsigned char* b) {
    if ((reinterpret_cast<uintptr_t>(p) & 7u) == 0) {
        uint2 w;
        w.x = (unsigned int)b[0] | ((unsigned int)b[1] << 8) | ((unsigned int)b[2] << 16) |
              ((unsigned int)b[3] << 24);
        w.y = (unsigned int)b[4] | ((unsigned int)b[5] << 8) | ((unsigned int)b[6] << 16) |
              ((unsigned int)b[7] << 24);
        *reinterpret_cast<uint2*>(p) = w;
    } else {
#pragma unroll
        for (int i = 0; i < SQ_ELEMS_PER_THREAD; i++) p[i] = b[i];
    }
}

// `moe_silu_mul`'s per-element body, rounded to BF16 and widened back: the
// value the unfused quantizer reads from memory.
__device__ __forceinline__ float sq_swiglu_bf16(float g, float u) {
    float sigmoid_g = 1.0f / (1.0f + __expf(-g));
    float result = g * sigmoid_g * u;
    return __bfloat162float(__float2bfloat16(result));
}

extern "C" __global__ void __launch_bounds__(128)
    swiglu_quant_fp8_ada(const __nv_bfloat16* __restrict__ gate, // [M, K] gate GEMM output
                         const __nv_bfloat16* __restrict__ up,   // [M, K] up GEMM output
                         unsigned char* __restrict__ out_fp8,    // [M, K] E4M3
                         float* __restrict__ a_scale,            // [M, K/128] FP32
                         unsigned int M, unsigned int K) {
    const unsigned int m = blockIdx.x;
    if (m >= M) return;
    const unsigned int L = K / SQ_GROUP_K;
    if (L == 0) return;

    const unsigned int gpc = (L + gridDim.y - 1u) / gridDim.y;
    const unsigned int g0 = blockIdx.y * gpc;
    if (g0 >= L) return;
    const unsigned int g_end = (g0 + gpc < L) ? (g0 + gpc) : L;

    const unsigned int tid = threadIdx.x;
    const unsigned int sub = tid / SQ_LANES_PER_GROUP;  // which of the 8 groups
    const unsigned int lane = tid % SQ_LANES_PER_GROUP; // which 8-element chunk
    const size_t row = (size_t)m * (size_t)K;
    // CTA-uniform trip count, so the butterfly is never entered by a
    // divergent subset of a warp.
    const unsigned int iters = (gpc + SQ_GROUPS_PER_CTA - 1u) / SQ_GROUPS_PER_CTA;

    for (unsigned int it = 0; it < iters; ++it) {
        const unsigned int g = g0 + it * SQ_GROUPS_PER_CTA + sub;
        const bool live = (g < g_end);
        const size_t base = row + (size_t)g * SQ_GROUP_K + (size_t)lane * SQ_ELEMS_PER_THREAD;

        // 1. SwiGLU in registers, rounded through BF16, and its abs-max.
        float r[SQ_ELEMS_PER_THREAD];
        float amax = 0.0f;
        if (live) {
            float gv[SQ_ELEMS_PER_THREAD];
            float uv[SQ_ELEMS_PER_THREAD];
            sq_load8(gate + base, gv);
            sq_load8(up + base, uv);
#pragma unroll
            for (int i = 0; i < SQ_ELEMS_PER_THREAD; i++) {
                r[i] = sq_swiglu_bf16(gv[i], uv[i]);
                amax = fmaxf(amax, fabsf(r[i]));
            }
        }

        // 2. The group's max over its 16 lanes. XOR offsets 8, 4, 2, 1 stay
        //    inside a 16-lane half, so two groups share a warp unmixed.
#pragma unroll
        for (int off = SQ_LANES_PER_GROUP / 2; off > 0; off >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off));
        }

        // 3. The scale, as `per_token_group_quant_fp8` computes it.
        float scale = amax / SQ_E4M3_MAX;
        if (scale < 1e-12f) scale = 1e-12f;
        if (live && lane == 0) a_scale[(size_t)m * (size_t)L + (size_t)g] = scale;
        if (!live) continue;

        // 4. Quantize from registers.
        unsigned char q[SQ_ELEMS_PER_THREAD];
#pragma unroll
        for (int i = 0; i < SQ_ELEMS_PER_THREAD; i++) {
            float v = r[i] / scale;
            v = fmaxf(fminf(v, SQ_E4M3_MAX), -SQ_E4M3_MAX);
            q[i] = (unsigned char)__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
        }
        sq_store8(out_fp8 + base, q);
    }
}
