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

// out[r, :] = x[rows[r], :] for r < n_rows (bf16 rows of `dim`). Grid: (n_rows). Block: 256.
extern "C" __global__ void moe_v41_gather_rows(
    const __nv_bfloat16* __restrict__ x, const int* __restrict__ rows,
    __nv_bfloat16* __restrict__ out, const unsigned int dim) {
    const unsigned int r = blockIdx.x;
    const __nv_bfloat16* src = x + (size_t)rows[r] * dim;
    __nv_bfloat16* dst = out + (size_t)r * dim;
    for (unsigned int d = threadIdx.x; d < dim; d += blockDim.x) dst[d] = src[d];
}

// acc[rows[r], :] += src[r, :] (f32 += bf16). Rows of one group are distinct
// tokens, so no two blocks touch the same acc row. Grid: (n_rows). Block: 256.
extern "C" __global__ void moe_v41_scatter_add(
    float* __restrict__ acc, const __nv_bfloat16* __restrict__ src,
    const int* __restrict__ rows, const unsigned int dim) {
    const unsigned int r = blockIdx.x;
    float* dst = acc + (size_t)rows[r] * dim;
    const __nv_bfloat16* s = src + (size_t)r * dim;
    for (unsigned int d = threadIdx.x; d < dim; d += blockDim.x) dst[d] += __bfloat162float(s[d]);
}

// Router logits at decode: one thread per (token, expert) output, strict
// k = 0..K-1 accumulation in fp32 with the same expression as
// dense_gemm_bf16_f32out, so the logits are bit-identical to the tiled kernel
// (the router-numerics pin) while every gate row is read once instead of the
// 16x16 tile idling 15 of its rows at m = 1. K is a multiple of 8 (dim = 5120):
// 8 bf16 per 16-byte load, consumed in order.
//
// Grid: (ceil(N/64), M, 1)  Block: (64, 1, 1)
extern "C" __global__ void moe_v41_router_gemv_f32out(
    const __nv_bfloat16* __restrict__ A,  // [M, K] row-major
    const __nv_bfloat16* __restrict__ B,  // [N, K] row-major
    float* __restrict__ C,                // [M, N] row-major, FP32
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int n = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int t = blockIdx.y;
    if (n >= N || t >= M) return;
    const __nv_bfloat16* a = A + (unsigned long long)t * K;
    const __nv_bfloat16* b = B + (unsigned long long)n * K;
    const uint4* a4 = (const uint4*)a;
    const uint4* b4 = (const uint4*)b;
    float acc = 0.0f;
    const unsigned int k8n = K / 8;
    // Eight 16-byte pairs in flight per trip (64 weights), all loads issued
    // before any add; the adds then run in strict k order, so the sum is the
    // same bits as the one-load-at-a-time loop.
    const unsigned int k64n = k8n / 8;
    for (unsigned int k64 = 0; k64 < k64n; ++k64) {
        uint4 av[8], bv[8];
        #pragma unroll
        for (int u = 0; u < 8; ++u) { av[u] = a4[k64 * 8 + u]; bv[u] = b4[k64 * 8 + u]; }
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            const unsigned int ar[4] = {av[u].x, av[u].y, av[u].z, av[u].w};
            const unsigned int br[4] = {bv[u].x, bv[u].y, bv[u].z, bv[u].w};
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                __nv_bfloat16 a_lo, a_hi, b_lo, b_hi;
                *(unsigned short*)&a_lo = (unsigned short)(ar[i] & 0xFFFFu);
                *(unsigned short*)&a_hi = (unsigned short)(ar[i] >> 16);
                *(unsigned short*)&b_lo = (unsigned short)(br[i] & 0xFFFFu);
                *(unsigned short*)&b_hi = (unsigned short)(br[i] >> 16);
                acc += __bfloat162float(a_lo) * __bfloat162float(b_lo);
                acc += __bfloat162float(a_hi) * __bfloat162float(b_hi);
            }
        }
    }
    for (unsigned int k8 = k64n * 8; k8 < k8n; ++k8) {
        const uint4 av = a4[k8];
        const uint4 bv = b4[k8];
        const unsigned int ar[4] = {av.x, av.y, av.z, av.w};
        const unsigned int br[4] = {bv.x, bv.y, bv.z, bv.w};
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            __nv_bfloat16 a_lo, a_hi, b_lo, b_hi;
            *(unsigned short*)&a_lo = (unsigned short)(ar[i] & 0xFFFFu);
            *(unsigned short*)&a_hi = (unsigned short)(ar[i] >> 16);
            *(unsigned short*)&b_lo = (unsigned short)(br[i] & 0xFFFFu);
            *(unsigned short*)&b_hi = (unsigned short)(br[i] >> 16);
            acc += __bfloat162float(a_lo) * __bfloat162float(b_lo);
            acc += __bfloat162float(a_hi) * __bfloat162float(b_hi);
        }
    }
    for (unsigned int k = k8n * 8; k < K; ++k) {
        acc += __bfloat162float(a[k]) * __bfloat162float(b[k]);
    }
    C[(unsigned long long)t * N + n] = acc;
}
