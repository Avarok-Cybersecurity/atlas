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

// acc[:] += src[0, :] + src[1, :] + ... + src[n_rows-1, :], added one row at a
// time in row order (the same rounding sequence as n_rows sequential
// accumulate launches). The single-token routed path: every expert row lands
// in the ONE token row, so scatter_add's distinct-rows precondition does not
// hold there and this kernel replaces it. Grid: ceil(dim / 256). Block: 256.
extern "C" __global__ void moe_v41_sum_rows(
    float* __restrict__ acc, const __nv_bfloat16* __restrict__ src,
    const unsigned int n_rows, const unsigned int dim) {
    const unsigned int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;
    float a = acc[d];
    for (unsigned int r = 0; r < n_rows; ++r) a += __bfloat162float(src[(size_t)r * dim + d]);
    acc[d] = a;
}

// The router's strict k = 0..K-1 fp32 chain over one activation row `a` and
// one gate row `b` (K a multiple of 8, 16-byte loads consumed in order), the
// same expression as dense_gemm_bf16_f32out so the logits are bit-identical to
// the tiled kernel (the router-numerics pin). Shared by the direct and the
// staged entries below, so the two can never drift apart.
static __device__ __forceinline__ float moe_v41_router_chain(
        const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ b, unsigned int K) {
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
    return acc;
}

// Router logits at decode: one thread per (token, expert) output, the chain
// above straight from global memory; every gate row is read once instead of
// the 16x16 tile idling 15 of its rows at m = 1.
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
    C[(unsigned long long)t * N + n] = moe_v41_router_chain(
        A + (unsigned long long)t * K, B + (unsigned long long)n * K, K);
}

// The same logits, staged: a 256-thread block takes one token (blockIdx.y)
// and MOE_V41_ROUTER_RPB gate rows (blockIdx.x), copies the activation row and
// those gate rows into shared memory with every thread loading (coalesced
// 16-byte loads, all in flight), then lane 0 of warps 0..RPB-1 runs the chain
// over the staged rows. Same arithmetic, same order, same bits; the 4 MB of
// gate rows now stream through ceil(N/RPB) blocks instead of ceil(N/64) blocks
// of 64 threads each pulling a 10 KB row alone (134 us for 4 MB, nsys 09-19).
// Dynamic shared memory: (1 + RPB) * K * 2 bytes.
//
// Grid: (ceil(N/RPB), M, 1)  Block: (256, 1, 1)
#define MOE_V41_ROUTER_RPB 2u
extern "C" __global__ void __launch_bounds__(256) moe_v41_router_gemv_f32out_staged(
    const __nv_bfloat16* __restrict__ A,  // [M, K] row-major
    const __nv_bfloat16* __restrict__ B,  // [N, K] row-major
    float* __restrict__ C,                // [M, N] row-major, FP32
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    extern __shared__ uint4 moe_v41_router_smem[];
    const unsigned int t = blockIdx.y;
    const unsigned int n0 = blockIdx.x * MOE_V41_ROUTER_RPB;
    if (t >= M || n0 >= N) return;
    const unsigned int k8n = K / 8;
    uint4* sa = moe_v41_router_smem;
    uint4* sb = moe_v41_router_smem + k8n;
    const uint4* a4 = (const uint4*)(A + (unsigned long long)t * K);
    for (unsigned int i = threadIdx.x; i < k8n; i += blockDim.x) sa[i] = a4[i];
    for (unsigned int r = 0; r < MOE_V41_ROUTER_RPB; ++r) {
        const unsigned int n = n0 + r;
        if (n >= N) break;
        const uint4* b4 = (const uint4*)(B + (unsigned long long)n * K);
        for (unsigned int i = threadIdx.x; i < k8n; i += blockDim.x) sb[r * k8n + i] = b4[i];
    }
    __syncthreads();
    const unsigned int r = threadIdx.x / 32;
    if ((threadIdx.x % 32) != 0 || r >= MOE_V41_ROUTER_RPB) return;
    const unsigned int n = n0 + r;
    if (n >= N) return;
    C[(unsigned long long)t * N + n] = moe_v41_router_chain(
        (const __nv_bfloat16*)sa, (const __nv_bfloat16*)(sb + r * k8n), K);
}
