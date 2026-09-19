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

// The same logits a third way: the block's 256 threads compute the 5120
// PRODUCTS of one gate row against the activation row into shared memory
// (a bf16 x bf16 product is exact in f32, so this is the chain's own FMUL
// result), then lane 0 adds them in k order from 0.0f, MOE_V41_CHAIN_UNROLL
// at a time from registers, so no load sits on the FADD chain. The adds are
// the chain's adds in the chain's order: the same bits as the tiled kernel
// and the two entries above (bench 09-19: 64 -> ~50 us a launch, one row a
// block). Dynamic shared memory: K * 4 bytes.
//
// Grid: (N, M, 1)  Block: (256, 1, 1)
#define MOE_V41_CHAIN_UNROLL 8
extern "C" __global__ void __launch_bounds__(256) moe_v41_router_gemv_f32out_products(
    const __nv_bfloat16* __restrict__ A,  // [M, K] row-major
    const __nv_bfloat16* __restrict__ B,  // [N, K] row-major
    float* __restrict__ C,                // [M, N] row-major, FP32
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    extern __shared__ float moe_v41_products[];
    const unsigned int t = blockIdx.y;
    const unsigned int n = blockIdx.x;
    if (t >= M || n >= N) return;
    const unsigned int k8n = K / 8;
    const uint4* a4 = (const uint4*)(A + (unsigned long long)t * K);
    const uint4* b4 = (const uint4*)(B + (unsigned long long)n * K);
    for (unsigned int i = threadIdx.x; i < k8n; i += blockDim.x) {
        const uint4 av = a4[i];
        const uint4 bv = b4[i];
        const unsigned int ar[4] = {av.x, av.y, av.z, av.w};
        const unsigned int br[4] = {bv.x, bv.y, bv.z, bv.w};
        float pr[8];
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            __nv_bfloat16 a_lo, a_hi, b_lo, b_hi;
            *(unsigned short*)&a_lo = (unsigned short)(ar[q] & 0xFFFFu);
            *(unsigned short*)&a_hi = (unsigned short)(ar[q] >> 16);
            *(unsigned short*)&b_lo = (unsigned short)(br[q] & 0xFFFFu);
            *(unsigned short*)&b_hi = (unsigned short)(br[q] >> 16);
            pr[2 * q] = __bfloat162float(a_lo) * __bfloat162float(b_lo);
            pr[2 * q + 1] = __bfloat162float(a_hi) * __bfloat162float(b_hi);
        }
        float4* p4 = (float4*)(moe_v41_products + 8 * i);
        p4[0] = make_float4(pr[0], pr[1], pr[2], pr[3]);
        p4[1] = make_float4(pr[4], pr[5], pr[6], pr[7]);
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    float acc = 0.0f;
    const float4* p4 = (const float4*)moe_v41_products;
    const unsigned int k4n = K / 4;
    unsigned int k = 0;
    for (; k + MOE_V41_CHAIN_UNROLL <= k4n; k += MOE_V41_CHAIN_UNROLL) {
        float4 v[MOE_V41_CHAIN_UNROLL];
        #pragma unroll
        for (int u = 0; u < MOE_V41_CHAIN_UNROLL; ++u) v[u] = p4[k + u];
        #pragma unroll
        for (int u = 0; u < MOE_V41_CHAIN_UNROLL; ++u) {
            acc += v[u].x;
            acc += v[u].y;
            acc += v[u].z;
            acc += v[u].w;
        }
    }
    for (; k < k4n; ++k) {
        const float4 v = p4[k];
        acc += v.x;
        acc += v.y;
        acc += v.z;
        acc += v.w;
    }
    for (unsigned int kk = k4n * 4; kk < K; ++kk) {
        acc += __bfloat162float(A[(unsigned long long)t * K + kk]) * __bfloat162float(B[(unsigned long long)n * K + kk]);
    }
    C[(unsigned long long)t * N + n] = acc;
}

// ── The single-token selection on the device (ATLAS_DS41_DEVICE_ROUTE=1) ──
//
// `route_from_logits` bit for bit: score = sqrt(softplus(logit / temp)) with
// glibc 2.39's expf (table + three double FMAs, round-half-away) and log1pf
// (the Sun algorithm in the installed aarch64 libm's exact FMA pattern),
// both verified identical to the host over all 2^32 f32 inputs
// (docs/perf/ds41_prototypes/glibc_softplus_exact.cu); rank by score + bias
// descending with index ascending on ties (= Rust's stable sort_by), top-k,
// weights in pick order with the sum in pick order (+1e-20, / , * scale as
// separate roundings), then the plan in ascending expert id: the weights and
// the gate / up / down pointers from the device slot table (a slot of -1 sets
// the miss flag; the host then fetches and re-uploads the table as today).
// One block of n_routed threads per token; the host checks the two constants.
#define MOE_V41_ROUTE_NR 384
#define MOE_V41_ROUTE_TOPK 6

__device__ __constant__ unsigned long long MOE_V41_EXP2F_TAB[32] = {
0x3ff0000000000000ull, 0x3fefd9b0d3158574ull, 0x3fefb5586cf9890full, 0x3fef9301d0125b51ull, 0x3fef72b83c7d517bull, 0x3fef54873168b9aaull, 0x3fef387a6e756238ull, 0x3fef1e9df51fdee1ull,
0x3fef06fe0a31b715ull, 0x3feef1a7373aa9cbull, 0x3feedea64c123422ull, 0x3feece086061892dull, 0x3feebfdad5362a27ull, 0x3feeb42b569d4f82ull, 0x3feeab07dd485429ull, 0x3feea47eb03a5585ull,
0x3feea09e667f3bcdull, 0x3fee9f75e8ec5f74ull, 0x3feea11473eb0187ull, 0x3feea589994cce13ull, 0x3feeace5422aa0dbull, 0x3feeb737b0cdc5e5ull, 0x3feec49182a3f090ull, 0x3feed503b23e255dull,
0x3feee89f995ad3adull, 0x3feeff76f2fb5e47ull, 0x3fef199bdd85529cull, 0x3fef3720dcef9069ull, 0x3fef5818dcfba487ull, 0x3fef7c97337b9b5full, 0x3fefa4afa2a490daull, 0x3fefd0765b6e4540ull };

static __device__ __forceinline__ float moe_v41_glibc_expf(float x) {
  const double N = 32.0, InvLn2N = 0x1.71547652b82fep+0 * N, C0 = 0x1.c6af84b912394p-5 / N / N / N, C1 = 0x1.ebfce50fac4f3p-3 / N / N, C2 = 0x1.62e42ff0c52d6p-1 / N;
  unsigned int abstop = (__float_as_uint(x) >> 20) & 0x7ff;
  if (abstop >= ((__float_as_uint(88.0f) >> 20) & 0x7ff)) {
    if (__float_as_uint(x) == __float_as_uint(-INFINITY)) return 0.0f;
    if (abstop >= ((__float_as_uint(INFINITY) >> 20) & 0x7ff)) return x + x;
    if (x > 0x1.62e42ep6f) return INFINITY;
    if (x < -0x1.9fe368p6f) return 0.0f;
  }
  double z = __dmul_rn(InvLn2N, (double)x); double kd = round(z); long long ki = (long long)kd; double r = __dsub_rn(z, kd);
  unsigned long long t = MOE_V41_EXP2F_TAB[ki & 31]; t += (unsigned long long)ki << 47; double s = __longlong_as_double((long long)t);
  double zz = fma(C0, r, C1); double r2 = __dmul_rn(r, r); double y = fma(C2, r, 1.0); y = fma(zz, r2, y); y = __dmul_rn(y, s);
  return (float)y;
}
static __device__ __forceinline__ float moe_v41_glibc_log1pf(float x) {
  const float ln2_hi = 6.9313812256e-01f, ln2_lo = 9.0580006145e-06f, two25 = 3.355443200e+07f;
  const float Lp1 = 6.6666668653e-01f, Lp2 = 4.0000000596e-01f, Lp3 = 2.8571429849e-01f, Lp4 = 2.2222198546e-01f, Lp5 = 1.8183572590e-01f, Lp6 = 1.5313838422e-01f, Lp7 = 1.4798198640e-01f;
  float hfsq, f, c = 0.0f, s, z, R, u; int k, hx, hu, ax;
  hx = (int)__float_as_uint(x); ax = hx & 0x7fffffff; k = 1;
  if (hx < 0x3ed413d7) {
    if (ax >= 0x3f800000) { if (x == -1.0f) return -two25 / 0.0f; else return (x - x) / (x - x); }
    if (ax < 0x31000000) { if (ax < 0x24800000) return x; else return __fmaf_rn(-__fmul_rn(x, x), 0.5f, x); }
    if (hx > 0 || hx <= (int)0xbe95f61f) { k = 0; f = x; hu = 1; }
  }
  if (hx >= 0x7f800000) return x + x;
  if (k != 0) {
    if (hx < 0x5a000000) { u = __fadd_rn(1.0f, x); hu = (int)__float_as_uint(u); k = (hu >> 23) - 127; c = (k > 0) ? __fsub_rn(1.0f, __fsub_rn(u, x)) : __fsub_rn(x, __fsub_rn(u, 1.0f)); c = __fdiv_rn(c, u); }
    else { u = x; hu = (int)__float_as_uint(u); k = (hu >> 23) - 127; c = 0; }
    hu &= 0x007fffff;
    if (hu < 0x3504f7) { u = __uint_as_float((unsigned int)(hu | 0x3f800000)); } else { k += 1; u = __uint_as_float((unsigned int)(hu | 0x3f000000)); hu = (0x00800000 - hu) >> 2; }
    f = __fsub_rn(u, 1.0f);
  }
  hfsq = __fmul_rn(__fmul_rn(0.5f, f), f); float kf = (float)k;
  if (hu == 0) {
    if (f == 0.0f) { if (k == 0) return 0.0f; else { c = __fmaf_rn(kf, ln2_lo, c); return __fmaf_rn(kf, ln2_hi, c); } }
    R = __fmul_rn(__fmaf_rn(-0.66666666666666666f, f, 1.0f), hfsq);
    if (k == 0) return __fsub_rn(f, R); else return __fmaf_rn(kf, ln2_hi, -__fsub_rn(__fsub_rn(R, __fmaf_rn(kf, ln2_lo, c)), f));
  }
  s = __fdiv_rn(f, __fadd_rn(2.0f, f)); z = __fmul_rn(s, s);
  float p = __fmaf_rn(z, Lp7, Lp6); p = __fmaf_rn(p, z, Lp5); p = __fmaf_rn(p, z, Lp4); p = __fmaf_rn(p, z, Lp3); p = __fmaf_rn(p, z, Lp2); p = __fmaf_rn(p, z, Lp1);
  float hr = __fmaf_rn(p, z, hfsq);
  if (k == 0) return __fsub_rn(f, __fsub_rn(hfsq, __fmul_rn(s, hr)));
  else return __fmaf_rn(kf, ln2_hi, -__fsub_rn(__fsub_rn(hfsq, __fadd_rn(__fmul_rn(s, hr), __fmaf_rn(kf, ln2_lo, c))), f));
}

// hdr: [0] miss flag, [1..1+K) picks in top-k order, [1+K..1+2K) weight bits
// in top-k order, [1+2K..1+3K) plan slots (ascending expert id, -1 = miss).
// plan_w / plan_rows: `weight_dev` / `rows_dev` in plan order (m = 1: row 0).
// ptrs: [3][K] gate / up / down addresses in plan order from
// arena_base + slot * slot_bytes + off.
// Grid: (1, 1, 1)  Block: (MOE_V41_ROUTE_NR, 1, 1)
extern "C" __global__ void __launch_bounds__(MOE_V41_ROUTE_NR) moe_v41_route_select(
        const float* __restrict__ logits, const float* __restrict__ bias, float gate_temp, float route_scale,
        unsigned int norm_topk, const int* __restrict__ slot_of, unsigned long long arena_base,
        unsigned long long slot_bytes, unsigned long long gate_off, unsigned long long up_off,
        unsigned long long down_off, int* __restrict__ hdr, float* __restrict__ plan_w,
        int* __restrict__ plan_rows, unsigned long long* __restrict__ ptrs) {
  constexpr int NR = MOE_V41_ROUTE_NR, K = MOE_V41_ROUTE_TOPK;
  __shared__ float sc[NR]; __shared__ float key[NR]; __shared__ int pick[K]; __shared__ float wsum;
  const int e = threadIdx.x;
  const float x = __fdiv_rn(logits[e], gate_temp);
  const float s = sqrtf(x > 20.0f ? x : moe_v41_glibc_log1pf(moe_v41_glibc_expf(x)));
  sc[e] = s; key[e] = __fadd_rn(s, bias[e]);
  if (e == 0) hdr[0] = 0;
  __syncthreads();
  const float mk = key[e]; int rank = 0;
  for (int j = 0; j < NR; j++) { const float kj = key[j]; if (kj > mk || (kj == mk && j < e)) rank++; }
  if (rank < K) pick[rank] = e;
  __syncthreads();
  if (e == 0) { float sum = 0.0f; for (int i = 0; i < K; i++) sum = __fadd_rn(sum, sc[pick[i]]); wsum = __fadd_rn(sum, 1e-20f); }
  __syncthreads();
  if (e < K) {
    float w = sc[pick[e]];
    if (norm_topk) w = __fdiv_rn(w, wsum);
    w = __fmul_rn(w, route_scale);
    hdr[1 + e] = pick[e]; hdr[1 + K + e] = (int)__float_as_uint(w);
    int pos = 0; for (int i = 0; i < K; i++) if (pick[i] < pick[e]) pos++;
    plan_w[pos] = w; plan_rows[pos] = 0;
    const int sl = slot_of[pick[e]];
    hdr[1 + 2 * K + pos] = sl;
    if (sl < 0) { atomicOr(hdr, 1); } else {
      const unsigned long long base = arena_base + (unsigned long long)sl * slot_bytes;
      ptrs[pos] = base + gate_off; ptrs[K + pos] = base + up_off; ptrs[2 * K + pos] = base + down_off;
    }
  }
}

// The device slot table `[layers][n_routed]` (int, -1 = not resident) brought
// up to date with the cache: `n` (layer, expert, slot) triples.
extern "C" __global__ void moe_v41_slot_table_set(int* __restrict__ table, unsigned int n_routed,
                                                  const int* __restrict__ triples, unsigned int n) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  table[(unsigned int)triples[3 * i] * n_routed + (unsigned int)triples[3 * i + 1]] = triples[3 * i + 2];
}
