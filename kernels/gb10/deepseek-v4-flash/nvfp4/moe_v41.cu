// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b
//
// DeepSeek-V4.1 Flash MoE glue around the K-quant expert kernels (kquant_moe.cu):
// the clamped SwiGLU between the gate/up and down projections, the f32
// accumulation of routed-expert outputs, and the final cast. Written to the CPU
// reference (deepseek_v41_ref::moe::expert): SwiGLU in f32 on the bf16 GEMM
// outputs, the routing weight multiplied in f32, the product cast to bf16 before
// w2, per-expert outputs summed in f32, one bf16 cast at the end.

#include <cuda_bf16.h>

// h[r, j] = bf16(silu(min(g, limit)) * clamp(u, -limit, limit) * w[r]); the
// clamp only when limit > 0, the weight only when `w` is non-null (the shared
// expert has none). Grid: ceil(rows * inter / 256). Block: 256.
extern "C" __global__ void moe_v41_swiglu(
    const __nv_bfloat16* __restrict__ gate, const __nv_bfloat16* __restrict__ up,
    const float* __restrict__ w, __nv_bfloat16* __restrict__ h,
    const unsigned int rows, const unsigned int inter, const float limit) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * inter) return;
    float g = __bfloat162float(gate[i]);
    float u = __bfloat162float(up[i]);
    if (limit > 0.0f) {
        u = fminf(fmaxf(u, -limit), limit);
        g = fminf(g, limit);
    }
    float v = (g / (1.0f + expf(-g))) * u;
    if (w != nullptr) v *= w[i / inter];
    h[i] = __float2bfloat16(v);
}

// acc[i] += src[i]. Grid: ceil(n / 256). Block: 256.
extern "C" __global__ void moe_v41_accumulate(
    float* __restrict__ acc, const __nv_bfloat16* __restrict__ src, const unsigned int n) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) acc[i] += __bfloat162float(src[i]);
}

// out[i] = bf16(acc[i]). Grid: ceil(n / 256). Block: 256.
extern "C" __global__ void moe_v41_finish(
    const float* __restrict__ acc, __nv_bfloat16* __restrict__ out, const unsigned int n) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(acc[i]);
}
