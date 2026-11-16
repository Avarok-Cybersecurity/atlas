// SPDX-License-Identifier: AGPL-3.0-only AND Apache-2.0
// Portions copied and adapted from TensorFold (https://github.com/ashhart/TensorFold), commit 3885dce,
// zig/kernels/cuda/w4a8_quantize.cuh:1-51 and zig/kernels/cuda/w4a8_normalize.h:1-14.
// Copyright 2026 TensorFold contributors. Licensed under the Apache License, Version 2.0; code written
// before TensorFold 0.6.0 is also available under the MIT License. See TENSORFOLD-ATTRIBUTION.md.
// Changes for Atlas: namespace renamed (avarok_affine_act); the per-group body split out as `quantize_group` so
// the plain quantizer and the new fused SwiGLU entry share it byte for byte; `w4a8_normalize.h` inlined; the
// exceptional mark is word 0 of a flags pair the prompt GEMMs gate on (affine_prefill_ada.cu); extern "C" entry
// names. The SwiGLU fusion (`affine4_swiglu_act_quant_ada`) is new Atlas code, written in the style of
// swiglu_quant_fp8_ada.cu.

// Ada (sm_89) activation side of the affine 4-bit W4A8 prompt GEMM: per row and
// per 64-input group, the int8 codes of x in the int8 mma's slot order, the group scale a = amax / 127 and the
// FP32 sum of the BF16 inputs, in one pass.
//
//   affine4_act_quant_ada          x (BF16, row stride ldx) -> xq, xa, xs
//   affine4_swiglu_act_quant_ada   gate, up (BF16) -> h = bf16(silu(gate) * up), then the same quantizer on h:
//                                  the dense FFN's SwiGLU, the down projection's group sums and its int8 input
//                                  in ONE launch (the unfused pair is moe_silu_mul then affine4_act_quant_ada)
//
// Outputs, row r, group g (KG = K / 64):
//   xq[r * K + g * 64 + step * 32 + s] = int8 code of input g * 64 + step * 32 + slot_input(s), s in 0..31
//   xa[r * KG + g] = a = amax / 127 (1 when amax == 0)      xs[r * KG + g] = FP32 sum of the 64 BF16 inputs
//
// The slot order is the one the packed 4-bit words give each lane of `mma.sync.m16n8k32` (`bytes4` in
// affine_prefill_ada.cu): within a 32-input step, slot s holds input (s & 16) + 2 * ((s & 15) >> 2) + (s & 1)
// + 8 * ((s & 3) >> 1). A code is round-half-even(v * 127 / amax) in -127..127.
//
// ── EXCEPTIONAL GROUPS ───────────────────────────────────────────────────────────────────────────────────────
//
// A group with a non-finite input, or whose FP32 sum overflows, cannot be represented in int8; its warp raises
// flags[0] and writes nothing for that group. The prompt GEMMs gate on flags[0] on the device: the int8 GEMM
// then skips and the BF16 prompt GEMM runs on the same BF16 input (for the fused entry: on h, which is written
// before the check), so a NaN in the input still reaches the output as it would on the BF16 route. The host
// clears flags[0] (cuMemsetD8Async) before every quantizer launch.
//
// ── NUMERICS ─────────────────────────────────────────────────────────────────────────────────────────────────
//
// Bit-exact host model: port/affine/prefill_oracle.rs (`quantize_rows`): amax and the sum are reduced by the
// warp butterfly below (lane value v0 + v1, then XOR offsets 16, 8, 4, 2, 1; IEEE addition is commutative, so
// every lane holds the same sum and lane 0's is stored), `amax / 127.0f` and `127.0f / amax` are IEEE divisions,
// codes are `__float2int_rn`. The fused entry's SwiGLU is moe_silu_mul's expression token for token
// (`1.0f / (1.0f + __expf(-g))`, then `g * sigmoid_g * u`, then `__float2bfloat16`), compiled with the same
// common/KERNEL.toml flags, so h is moe_silu_mul's output and the codes, scales and sums are the plain quantizer's
// on it: fused == pair, byte for byte (the GPU microtest pins it; `__expf` has no bit-exact host model, so the host
// model of the fusion is self-consistent only).
//
// Grid: (M, ceil(KG / 4))  Block: 128 (four warps, a group each). No shared memory. Requires K % 64 == 0.

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

namespace avarok_affine_act {

// Within each 32-input step the int8 mma's slot s holds input P(s), the order the packed 4-bit words give each lane.
__device__ __forceinline__ int slot_input(int s) {
    const int q = (s & 15) >> 2, i = s & 3;
    return (s & 16) + 2 * q + (i & 1) + 8 * (i >> 1);
}

// w4a8_normalize.h: keep the multiply whenever the reciprocal is finite.
__device__ __forceinline__ float normalized(float v, float amax, float inv) {
    return inv > 0x1.fffffep127f ? (v / amax) * 127.0f : v * inv;
}

// One warp, one 64-input group. Lane l holds slots 2l and 2l + 1 (mod 32) of step l / 16, as v0 and v1.
__device__ __forceinline__ void quantize_group(float v0, float v1, int lane, uint8_t* __restrict__ xq_group,
                                               float* __restrict__ xa_at, float* __restrict__ xs_at,
                                               unsigned int* __restrict__ flags) {
    if (__any_sync(0xffffffffu, !isfinite(v0) || !isfinite(v1))) {
        if (lane == 0) atomicOr(flags, 1u);
        return;
    }
    float amax = fmaxf(fabsf(v0), fabsf(v1)), sum = v0 + v1;
#pragma unroll
    for (int o = 16; o; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    const float a = amax > 0.0f ? amax / 127.0f : 1.0f, inv = amax > 0.0f ? 127.0f / amax : 1.0f;
    if (__any_sync(0xffffffffu, !isfinite(a) || !isfinite(sum))) {
        if (lane == 0) atomicOr(flags, 1u);
        return;
    }
    const int q0 = __float2int_rn(normalized(v0, amax, inv)), q1 = __float2int_rn(normalized(v1, amax, inv));
    const uint16_t q = (uint16_t)((q0 & 0xFF) | ((q1 & 0xFF) << 8));
    reinterpret_cast<uint16_t*>(xq_group)[lane] = q;
    if (lane == 0) {
        *xa_at = a;
        *xs_at = sum;
    }
}

// moe_silu_mul's per-element body (kernels/gb10/common/moe_silu_mul.cu), rounded to BF16 and widened back: the
// value the unfused quantizer reads from memory.
__device__ __forceinline__ float swiglu_bf16(float g, float u) {
    float sigmoid_g = 1.0f / (1.0f + __expf(-g));
    float result = g * sigmoid_g * u;
    return __bfloat162float(__float2bfloat16(result));
}

}   // namespace avarok_affine_act

// x [M, K] BF16, row stride ldx -> xq [M, K] int8 (slot order), xa, xs [M, K / 64] FP32; flags[0] |= exceptional.
extern "C" __global__ void __launch_bounds__(128)
    affine4_act_quant_ada(const __nv_bfloat16* __restrict__ x, uint8_t* __restrict__ xq, float* __restrict__ xa,
                          float* __restrict__ xs, int K, int ldx, unsigned int* __restrict__ flags) {
    using namespace avarok_affine_act;
    const int row = blockIdx.x, lane = threadIdx.x & 31, g = blockIdx.y * 4 + (threadIdx.x >> 5);
    if (g * 64 >= K) return;
    const __nv_bfloat16* src = x + (size_t)row * ldx + g * 64;
    const int s0 = (2 * lane) & 31, step = lane >> 4;
    const float v0 = __bfloat162float(src[step * 32 + slot_input(s0)]);
    const float v1 = __bfloat162float(src[step * 32 + slot_input(s0 + 1)]);
    const size_t at = (size_t)row * (K / 64) + g;
    quantize_group(v0, v1, lane, xq + (size_t)row * K + g * 64, xa + at, xs + at, flags);
}

// gate, up [M, K] BF16, row stride ld_in -> h [M, K] BF16 (row stride ld_h; NULL skips the store; h may alias
// gate when ld_h == ld_in: every element is read and written by the same lane), then xq, xa, xs of h as above.
extern "C" __global__ void __launch_bounds__(128)
    affine4_swiglu_act_quant_ada(const __nv_bfloat16* gate, const __nv_bfloat16* __restrict__ up,
                                 __nv_bfloat16* h, int ld_in, int ld_h, uint8_t* __restrict__ xq,
                                 float* __restrict__ xa, float* __restrict__ xs, int K,
                                 unsigned int* __restrict__ flags) {
    using namespace avarok_affine_act;
    const int row = blockIdx.x, lane = threadIdx.x & 31, g = blockIdx.y * 4 + (threadIdx.x >> 5);
    if (g * 64 >= K) return;
    const int s0 = (2 * lane) & 31, step = lane >> 4;
    const int i0 = g * 64 + step * 32 + slot_input(s0), i1 = g * 64 + step * 32 + slot_input(s0 + 1);
    const size_t in = (size_t)row * ld_in;
    const float r0 = swiglu_bf16(__bfloat162float(gate[in + i0]), __bfloat162float(up[in + i0]));
    const float r1 = swiglu_bf16(__bfloat162float(gate[in + i1]), __bfloat162float(up[in + i1]));
    if (h != nullptr) {
        const size_t o = (size_t)row * ld_h;
        h[o + i0] = __float2bfloat16(r0);   // exact: r0 is already a BF16 value
        h[o + i1] = __float2bfloat16(r1);
    }
    const size_t at = (size_t)row * (K / 64) + g;
    quantize_group(r0, r1, lane, xq + (size_t)row * K + g * 64, xa + at, xs + at, flags);
}
