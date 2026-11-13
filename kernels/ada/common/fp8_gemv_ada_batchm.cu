// SPDX-License-Identifier: AGPL-3.0-only
// Portions derived from Cinference (https://github.com/satellitedown/cinference, commit 383e5db), itself
// derived from NInfer by Neroued and contributors (https://github.com/Neroued/ninfer), Apache License 2.0.
// Source: src/ops/common/memory.cuh:72-84 (the `createpolicy.fractional.L2::evict_first` policy for
// weights read once). Changes: used only by the `_l2ef` entry points below, on
// `ld.global.nc.L1::no_allocate.L2::cache_hint` vector loads, by Atlas contributors. The batch-M GEMV
// itself is Atlas-native (it is `fp8_gemv_ada.cu` with M accumulators).

// Ada (sm_89) FP8-E4M3 W8A16 GEMVs for M = 2..8 activation rows on ONE pass
// over the weight: the dense FFN's speculative VERIFY rows (MTP / DFlash K+1
// rows), port row `ffn_fp8_verify_gemv` (off by default).
//
//   C[t, n] = sum_k A[t, k] * E4M3(B[n, k]) * block_scale[n/128, k/128],  t < M
//
// Entry points, M in 2..=8 (M is part of the NAME, so each is a fully
// unrolled instantiation with M register accumulators):
//
//   fp8_gemv_ada_batchm_<M>               == M x fp8_gemv_ada
//   fp8_gemv_ada_batchm_dual_<M>          == M x fp8_gemv_ada_dual
//   fp8_gemv_ada_batchm_silu_input_<M>    == M x fp8_gemv_ada_silu_input
//   fp8_gemv_ada_batchm_l2ef_<M>, _l2ef_dual_<M>, _l2ef_silu_input_<M>:
//       the same with evict-first L2 weight loads (row `l2_evict_first_weights`)
//
// SAME weight format and SAME launch geometry as `fp8_gemv_ada.cu`: grid
// (ceil(N/4), 1, {1, 2}), 256 threads, B[N, K] E4M3 bytes, block_scale
// [ceil(N/128), ceil(K/128)] FP32, K % 16 == 0. A is [M, K] BF16 row-major,
// C is [M, N] BF16 row-major (the dual writes C1 and C2, each [M, N]); the
// SiLU input variant reads gate and up, each [M, K].
//
// ── WHY ──────────────────────────────────────────────────────────────────
//
// On the Ada FP8 route the dense FFN's verify forward (`forward_k2`, `_k3`,
// `_km`) runs the PREFILL ladder: `w8a16_gemv_batch4` at M <= 4 and the
// padded tile GEMMs or the lossy W8A8 arm at M 5..8. Each weight byte should be
// read once per verify pass whatever M is (the pass is bandwidth bound at
// these widths: 2 x 2560 x 9216 + 9216 x 2560 FP8 bytes per layer on the 4B),
// so the target is "one weight pass, M dot products", which is what this file
// does with the Ada register dequant (`cvt.rn.f16x2.e4m3x2`) instead of the
// gb10 shared-memory table.
//
// ── NUMERICS: byte-identical to M launches of the M=1 kernel ─────────────
//
// Row t's arithmetic is `fp8_gemv_ada`'s, operation for operation:
//   * lane l of 64 walks chunks l, l+64, l+128, ... (the 2-deep unroll moves
//     LOADS only, exactly as there);
//   * within a chunk, word j of the 16 weight bytes is decoded ONCE to
//     w0..w3 = decode(byte) * scale (`__fmul_rn`), then for each row t, in t
//     order, acc[t] += a[t][4j+i] * w_i for i = 0..3 as separate
//     `__fmul_rn` / `__fadd_rn`, i.e. row t sees the same sequence of rounded
//     operations in the same K order as the M=1 kernel. Interleaving rows does
//     not reorder any one row's operations;
//   * the same 5-step shfl.down butterfly per row, the same two-warp add
//     through shared memory, the same single `__float2bfloat16`;
//   * the SiLU input is `(g / (1 + __expf(-g))) * u` in FP32 per element, with
//     the same intrinsics, as `FgaActSilu` computes it.
// Activations are read as 8-byte words (4 BF16) per row instead of 16-byte
// words; that changes which bytes arrive together, not their values.
// `examples/fp8_ada_batchm_microtest.rs` byte-compares every entry against M
// launches of its M=1 counterpart (compare bytes, never a cosine: see the
// 2026-08-12 note in kernels/gb10/common/w8a16_gemv_batch4.cu).

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 890)
#error "fp8_gemv_ada_batchm.cu needs sm_89 or newer (cvt.rn.f16x2.e4m3x2)"
#endif

#define FGB_BLOCK 256
#define FGB_N_PER_BLOCK 4
#define FGB_LANES (FGB_BLOCK / FGB_N_PER_BLOCK) // 64 lanes per output row
#define FGB_WARP 32
#define FGB_FP8_BLOCK 128
#define FGB_CHUNK 16                                   // K values per uint4 of FP8 bytes
#define FGB_CHUNKS_PER_SCALE (FGB_FP8_BLOCK / FGB_CHUNK) // 8
#define FGB_UNROLL 2

// The hinted load is compiled out where Cinference documents the ptxas fault
// (CUDA 13.0 and 13.1); see the TOOLCHAIN CAVEAT in fp8_gemv_ada.cu.
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ == 13) && (__CUDACC_VER_MINOR__ < 2)
#define FGB_L2_HINT 0
#else
#define FGB_L2_HINT 1
#endif

// Two E4M3 bytes (low byte = lower K index) -> two FP32, as in fp8_gemv_ada.cu.
__device__ __forceinline__ float2 fgb_e4m3x2_to_f32x2(unsigned short raw) {
    unsigned int h2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(raw));
    return __half22float2(*reinterpret_cast<const __half2*>(&h2));
}

__device__ __forceinline__ float fgb_bf16_lo(unsigned int word) {
    return __uint_as_float(word << 16);
}
__device__ __forceinline__ float fgb_bf16_hi(unsigned int word) {
    return __uint_as_float(word & 0xFFFF0000u);
}

// ── Weight loaders ───────────────────────────────────────────────────────

// The base kernels' load: streaming hint.
struct FgbLoadStream {
    __device__ __forceinline__ static FgbLoadStream make() { return FgbLoadStream{}; }
    __device__ __forceinline__ uint4 operator()(const uint4* p) const { return __ldcs(p); }
};

// Evict-first L2 policy over the whole line, no L1 allocation.
struct FgbLoadEvictFirst {
    unsigned long long policy;

    __device__ __forceinline__ static FgbLoadEvictFirst make() {
        FgbLoadEvictFirst l;
        l.policy = 0ull;
#if FGB_L2_HINT
        asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(l.policy));
#endif
        return l;
    }

    __device__ __forceinline__ uint4 operator()(const uint4* p) const {
#if FGB_L2_HINT
        uint4 v;
        asm("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;"
            : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
            : "l"(p), "l"(policy));
        return v;
#else
        return __ldcs(p);
#endif
    }
};

// ── Activation sources ───────────────────────────────────────────────────

// Plain BF16 rows A[M, K]: the four values of row t at K = 16*k16 + 4*j + 0..3.
struct FgbActRows {
    const __nv_bfloat16* __restrict__ a;
    unsigned int k;

    __device__ __forceinline__ void quad(int t, unsigned int k16, int j, float* out) const {
        const uint2* row = reinterpret_cast<const uint2*>(a + (size_t)t * k);
        const uint2 w = __ldg(row + k16 * 4 + j);
        out[0] = fgb_bf16_lo(w.x);
        out[1] = fgb_bf16_hi(w.x);
        out[2] = fgb_bf16_lo(w.y);
        out[3] = fgb_bf16_hi(w.y);
    }
};

// silu(gate) * up computed inline per element, exactly as `FgaActSilu`
// (fp8_gemv_ada.cu) and `w8a16_gemv_silu_input` compute it.
struct FgbActSilu {
    const __nv_bfloat16* __restrict__ gate;
    const __nv_bfloat16* __restrict__ up;
    unsigned int k;

    __device__ __forceinline__ void quad(int t, unsigned int k16, int j, float* out) const {
        const size_t row = (size_t)t * k;
        const uint2 g2 = __ldg(reinterpret_cast<const uint2*>(gate + row) + k16 * 4 + j);
        const uint2 u2 = __ldg(reinterpret_cast<const uint2*>(up + row) + k16 * 4 + j);
        const float g[4] = {fgb_bf16_lo(g2.x), fgb_bf16_hi(g2.x), fgb_bf16_lo(g2.y), fgb_bf16_hi(g2.y)};
        const float u[4] = {fgb_bf16_lo(u2.x), fgb_bf16_hi(u2.x), fgb_bf16_lo(u2.y), fgb_bf16_hi(u2.y)};
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const float sig_den = __fadd_rn(1.0f, __expf(-g[i]));
            out[i] = __fmul_rn(__fdiv_rn(g[i], sig_den), u[i]);
        }
    }
};

// ── The GEMV ─────────────────────────────────────────────────────────────

// One 16-value chunk for all M rows: decode each weight word once, then for
// every row the M=1 kernel's four rounded multiply/add steps in K order.
template <int M, typename Act>
__device__ __forceinline__ void fgb_chunk(float (&acc)[M], uint4 b, float scale, const Act& act, unsigned int k16) {
    const unsigned int words[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const float2 lo = fgb_e4m3x2_to_f32x2((unsigned short)(words[j] & 0xFFFFu));
        const float2 hi = fgb_e4m3x2_to_f32x2((unsigned short)(words[j] >> 16));
        const float w0 = __fmul_rn(lo.x, scale);
        const float w1 = __fmul_rn(lo.y, scale);
        const float w2 = __fmul_rn(hi.x, scale);
        const float w3 = __fmul_rn(hi.y, scale);
#pragma unroll
        for (int t = 0; t < M; t++) {
            float a[4];
            act.quad(t, k16, j, a);
            acc[t] = __fadd_rn(acc[t], __fmul_rn(a[0], w0));
            acc[t] = __fadd_rn(acc[t], __fmul_rn(a[1], w1));
            acc[t] = __fadd_rn(acc[t], __fmul_rn(a[2], w2));
            acc[t] = __fadd_rn(acc[t], __fmul_rn(a[3], w3));
        }
    }
}

// One output row's partial dot products (all M rows) for lane `lane` of 64.
// `acc` arrives zeroed.
template <int M, typename Act, typename Load>
__device__ __forceinline__ void fgb_row(float (&acc)[M], const unsigned char* __restrict__ b_row,
                                        const float* __restrict__ scale_row, const Act& act, const Load& load,
                                        unsigned int k16_count, unsigned int lane) {
    const uint4* b4 = reinterpret_cast<const uint4*>(b_row);
    unsigned int k16 = lane;
    for (; k16 + (FGB_UNROLL - 1) * FGB_LANES < k16_count; k16 += FGB_UNROLL * FGB_LANES) {
        uint4 b[FGB_UNROLL];
        float s[FGB_UNROLL];
#pragma unroll
        for (int u = 0; u < FGB_UNROLL; u++) {
            const unsigned int c = k16 + u * FGB_LANES;
            b[u] = load(b4 + c);
            s[u] = __ldg(scale_row + c / FGB_CHUNKS_PER_SCALE);
        }
#pragma unroll
        for (int u = 0; u < FGB_UNROLL; u++) {
            fgb_chunk<M>(acc, b[u], s[u], act, k16 + u * FGB_LANES);
        }
    }
    for (; k16 < k16_count; k16 += FGB_LANES) {
        fgb_chunk<M>(acc, load(b4 + k16), __ldg(scale_row + k16 / FGB_CHUNKS_PER_SCALE), act, k16);
    }
}

// The M=1 kernel's reduction, once per row: 5-step shfl.down per warp, the
// two warps of an output added through shared memory by lane 0, one BF16
// round. Every thread of the CTA reaches the barrier.
template <int M>
__device__ __forceinline__ void fgb_reduce_store(float (&acc)[M], bool live, unsigned int local_out,
                                                 unsigned int lane, __nv_bfloat16* __restrict__ c, unsigned int n,
                                                 unsigned int N) {
    __shared__ float smem[M][FGB_N_PER_BLOCK * 2];
#pragma unroll
    for (int t = 0; t < M; t++) {
#pragma unroll
        for (int offset = FGB_WARP / 2; offset > 0; offset >>= 1) {
            acc[t] = __fadd_rn(acc[t], __shfl_down_sync(0xFFFFFFFFu, acc[t], offset));
        }
    }
    if (lane % FGB_WARP == 0) {
#pragma unroll
        for (int t = 0; t < M; t++) {
            smem[t][local_out * 2 + lane / FGB_WARP] = acc[t];
        }
    }
    __syncthreads();
    if (live && lane == 0) {
#pragma unroll
        for (int t = 0; t < M; t++) {
            c[(size_t)t * N + n] = __float2bfloat16(__fadd_rn(smem[t][local_out * 2], smem[t][local_out * 2 + 1]));
        }
    }
}

template <int M, typename Act, typename Load>
__device__ __forceinline__ void fgb_gemv(const Act& act, const Load& load, const unsigned char* __restrict__ B,
                                         const float* __restrict__ block_scale, __nv_bfloat16* __restrict__ C,
                                         unsigned int N, unsigned int K) {
    const unsigned int local_out = threadIdx.x / FGB_LANES;
    const unsigned int lane = threadIdx.x % FGB_LANES;
    const unsigned int n = blockIdx.x * FGB_N_PER_BLOCK + local_out;
    const bool live = n < N;
    float acc[M];
#pragma unroll
    for (int t = 0; t < M; t++) {
        acc[t] = 0.0f;
    }
    if (live) {
        const unsigned int k_blocks = (K + FGB_FP8_BLOCK - 1) / FGB_FP8_BLOCK;
        fgb_row<M>(acc, B + (unsigned long long)n * K, block_scale + (unsigned long long)(n / FGB_FP8_BLOCK) * k_blocks,
                   act, load, K / FGB_CHUNK, lane);
    }
    fgb_reduce_store<M>(acc, live, local_out, lane, C, n, N);
}

template <int M, typename Load>
__device__ __forceinline__ void fgb_plain(const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ B,
                                          const float* __restrict__ block_scale, __nv_bfloat16* __restrict__ C,
                                          unsigned int N, unsigned int K) {
    const FgbActRows act{A, K};
    const Load load = Load::make();
    fgb_gemv<M>(act, load, B, block_scale, C, N, K);
}

// Gate and up of the same rows; blockIdx.z picks which.
template <int M, typename Load>
__device__ __forceinline__ void fgb_dual(const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ B1,
                                         const float* __restrict__ B1_scale, __nv_bfloat16* __restrict__ C1,
                                         const unsigned char* __restrict__ B2, const float* __restrict__ B2_scale,
                                         __nv_bfloat16* __restrict__ C2, unsigned int N, unsigned int K) {
    const FgbActRows act{A, K};
    const Load load = Load::make();
    if (blockIdx.z == 0) {
        fgb_gemv<M>(act, load, B1, B1_scale, C1, N, K);
    } else {
        fgb_gemv<M>(act, load, B2, B2_scale, C2, N, K);
    }
}

template <int M, typename Load>
__device__ __forceinline__ void fgb_silu(const __nv_bfloat16* __restrict__ gate_out,
                                         const __nv_bfloat16* __restrict__ up_out, const unsigned char* __restrict__ B,
                                         const float* __restrict__ block_scale, __nv_bfloat16* __restrict__ C,
                                         unsigned int N, unsigned int K) {
    const FgbActSilu act{gate_out, up_out, K};
    const Load load = Load::make();
    fgb_gemv<M>(act, load, B, block_scale, C, N, K);
}

// ── Entry points ─────────────────────────────────────────────────────────
//
// Spelled out one by one (no macro), so every name is greppable in this file;
// `layers/dense_ffn_ada_kernels/verify.rs` holds the matching name tables and
// a CPU test checks each one appears here.

#define FGB_PLAIN_ARGS                                                                                         \
    const __nv_bfloat16 *__restrict__ A, const unsigned char *__restrict__ B, const float *__restrict__ block_scale, \
        __nv_bfloat16 *__restrict__ C, unsigned int N, unsigned int K
#define FGB_DUAL_ARGS                                                                                            \
    const __nv_bfloat16 *__restrict__ A, const unsigned char *__restrict__ B1, const float *__restrict__ B1_scale, \
        __nv_bfloat16 *__restrict__ C1, const unsigned char *__restrict__ B2, const float *__restrict__ B2_scale, \
        __nv_bfloat16 *__restrict__ C2, unsigned int N, unsigned int K
#define FGB_SILU_ARGS                                                                                 \
    const __nv_bfloat16 *__restrict__ gate_out, const __nv_bfloat16 *__restrict__ up_out,              \
        const unsigned char *__restrict__ B, const float *__restrict__ block_scale, __nv_bfloat16 *__restrict__ C, \
        unsigned int N, unsigned int K

// Plain rows, streaming loads.
extern "C" __global__ void __launch_bounds__(FGB_BLOCK) fp8_gemv_ada_batchm_2(FGB_PLAIN_ARGS) {
    fgb_plain<2, FgbLoadStream>(A, B, block_scale, C, N, K);
}
extern "C" __global__ void __launch_bounds__(FGB_BLOCK) fp8_gemv_ada_batchm_3(FGB_PLAIN_ARGS) {
    fgb_plain<3, FgbLoadStream>(A, B, block_scale, C, N, K);
}
extern "C" __global__ void __launch_bounds__(FGB_BLOCK) fp8_gemv_ada_batchm_4(FGB_PLAIN_ARGS) {
    fgb_plain<4, FgbLoadStream>(A, B, block_scale, C, N, K);
}
extern "C" __global__ void __launch_bounds__(FGB_BLOCK) fp8_gemv_ada_batchm_5(FGB_PLAIN_ARGS) {
    fgb_plain<5, FgbLoadStream>(A, B, block_scale, C, N, K);
}
extern "C" __global__ void __launch_bounds__(FGB_BLOCK) fp8_gemv_ada_batchm_6(FGB_PLAIN_ARGS) {
    fgb_plain<6, FgbLoadStream>(A, B, block_scale, C, N, K);
}
extern "C" __global__ void __launch_bounds__(FGB_BLOCK) fp8_gemv_ada_batchm_7(FGB_PLAIN_ARGS) {
    fgb_plain<7, FgbLoadStream>(A, B, block_scale, C, N, K);
}
extern "C" __global__ void __launch_bounds__(FGB_BLOCK) fp8_gemv_ada_batchm_8(FGB_PLAIN_ARGS) {
    fgb_plain<8, FgbLoadStream>(A, B, block_scale, C, N, K);
}

// Gate + up, streaming loads. Grid (ceil(N/4), 1, 2).
extern "C" __global__ void __launch_bounds__(FGB_BLOCK) fp8_gemv_ada_batchm_dual_2(FGB_DUAL_ARGS) {
    fgb_dual<2, FgbLoadStream>(A, B1, B1_scale, C1, B2, B2_scale, C2, N, K);
}
extern "C" __global__ void __launch_bounds__(FGB_BLOCK) fp8_gemv_ada_batchm_dual_3(FGB_DUAL_ARGS) {
    fgb_dual<3, FgbLoadStream>(A, B1, B1_scale, C1, B2, B2_scale, C2, N, K);
}
extern "C" __global__ void __launch_bounds__(FGB_BLOCK) fp8_gemv_ada_batchm_dual_4(FGB_DUAL_ARGS) {
    fgb_dual<4, FgbLoadStream>(A, B1, B1_scale, C1, B2, B2_scale, C2, N, K);
}
extern "C" __global__ void __launch_bounds__(FGB_BLOCK) fp8_gemv_ada_batchm_dual_5(FGB_DUAL_ARGS) {
    fgb_dual<5, FgbLoadStream>(A, B1, B1_scale, C1, B2, B2_scale, C2, N, K);
}
extern "C" __global__ void __launch_bounds__(FGB_BLOCK) fp8_gemv_ada_batchm_dual_6(FGB_DUAL_ARGS) {
    fgb_dual<6, FgbLoadStream>(A, B1, B1_scale, C1, B2, B2_scale, C2, N, K);
}
extern "C" __global__ void __launch_bounds__(FGB_BLOCK) fp8_gemv_ada_batchm_dual_7(FGB_DUAL_ARGS) {
    fgb_dual<7, FgbLoadStream>(A, B1, B1_scale, C1, B2, B2_scale, C2, N, K);
}
extern "C" __global__ void __launch_bounds__(FGB_BLOCK) fp8_gemv_ada_batchm_dual_8(FGB_DUAL_ARGS) {
    fgb_dual<8, FgbLoadStream>(A, B1, B1_scale, C1, B2, B2_scale, C2, N, K);
}

// down(silu(gate) * up), streaming loads.
extern "C" __global__ void __launch_bounds__(FGB_BLOCK) fp8_gemv_ada_batchm_silu_input_2(FGB_SILU_ARGS) {
    fgb_silu<2, FgbLoadStream>(gate_out, up_out, B, block_scale, C, N, K);
}
extern "C" __global__ void __launch_bounds__(FGB_BLOCK) fp8_gemv_ada_batchm_silu_input_3(FGB_SILU_ARGS) {
    fgb_silu<3, FgbLoadStream>(gate_out, up_out, B, block_scale, C, N, K);
}
extern "C" __global__ void __launch_bounds__(FGB_BLOCK) fp8_gemv_ada_batchm_silu_input_4(FGB_SILU_ARGS) {
    fgb_silu<4, FgbLoadStream>(gate_out, up_out, B, block_scale, C, N, K);
}
extern "C" __global__ void __launch_bounds__(FGB_BLOCK) fp8_gemv_ada_batchm_silu_input_5(FGB_SILU_ARGS) {
    fgb_silu<5, FgbLoadStream>(gate_out, up_out, B, block_scale, C, N, K);
}
extern "C" __global__ void __launch_bounds__(FGB_BLOCK) fp8_gemv_ada_batchm_silu_input_6(FGB_SILU_ARGS) {
    fgb_silu<6, FgbLoadStream>(gate_out, up_out, B, block_scale, C, N, K);
}
extern "C" __global__ void __launch_bounds__(FGB_BLOCK) fp8_gemv_ada_batchm_silu_input_7(FGB_SILU_ARGS) {
    fgb_silu<7, FgbLoadStream>(gate_out, up_out, B, block_scale, C, N, K);
}
extern "C" __global__ void __launch_bounds__(FGB_BLOCK) fp8_gemv_ada_batchm_silu_input_8(FGB_SILU_ARGS) {
    fgb_silu<8, FgbLoadStream>(gate_out, up_out, B, block_scale, C, N, K);
}

// Plain rows, evict-first L2 weight loads.
extern "C" __global__ void __launch_bounds__(FGB_BLOCK) fp8_gemv_ada_batchm_l2ef_2(FGB_PLAIN_ARGS) {
    fgb_plain<2, FgbLoadEvictFirst>(A, B, block_scale, C, N, K);
}
extern "C" __global__ void __launch_bounds__(FGB_BLOCK) fp8_gemv_ada_batchm_l2ef_3(FGB_PLAIN_ARGS) {
    fgb_plain<3, FgbLoadEvictFirst>(A, B, block_scale, C, N, K);
}
extern "C" __global__ void __launch_bounds__(FGB_BLOCK) fp8_gemv_ada_batchm_l2ef_4(FGB_PLAIN_ARGS) {
    fgb_plain<4, FgbLoadEvictFirst>(A, B, block_scale, C, N, K);
}
extern "C" __global__ void __launch_bounds__(FGB_BLOCK) fp8_gemv_ada_batchm_l2ef_5(FGB_PLAIN_ARGS) {
    fgb_plain<5, FgbLoadEvictFirst>(A, B, block_scale, C, N, K);
}
extern "C" __global__ void __launch_bounds__(FGB_BLOCK) fp8_gemv_ada_batchm_l2ef_6(FGB_PLAIN_ARGS) {
    fgb_plain<6, FgbLoadEvictFirst>(A, B, block_scale, C, N, K);
}
extern "C" __global__ void __launch_bounds__(FGB_BLOCK) fp8_gemv_ada_batchm_l2ef_7(FGB_PLAIN_ARGS) {
    fgb_plain<7, FgbLoadEvictFirst>(A, B, block_scale, C, N, K);
}
extern "C" __global__ void __launch_bounds__(FGB_BLOCK) fp8_gemv_ada_batchm_l2ef_8(FGB_PLAIN_ARGS) {
    fgb_plain<8, FgbLoadEvictFirst>(A, B, block_scale, C, N, K);
}

// Gate + up, evict-first L2 weight loads.
extern "C" __global__ void __launch_bounds__(FGB_BLOCK) fp8_gemv_ada_batchm_l2ef_dual_2(FGB_DUAL_ARGS) {
    fgb_dual<2, FgbLoadEvictFirst>(A, B1, B1_scale, C1, B2, B2_scale, C2, N, K);
}
extern "C" __global__ void __launch_bounds__(FGB_BLOCK) fp8_gemv_ada_batchm_l2ef_dual_3(FGB_DUAL_ARGS) {
    fgb_dual<3, FgbLoadEvictFirst>(A, B1, B1_scale, C1, B2, B2_scale, C2, N, K);
}
extern "C" __global__ void __launch_bounds__(FGB_BLOCK) fp8_gemv_ada_batchm_l2ef_dual_4(FGB_DUAL_ARGS) {
    fgb_dual<4, FgbLoadEvictFirst>(A, B1, B1_scale, C1, B2, B2_scale, C2, N, K);
}
extern "C" __global__ void __launch_bounds__(FGB_BLOCK) fp8_gemv_ada_batchm_l2ef_dual_5(FGB_DUAL_ARGS) {
    fgb_dual<5, FgbLoadEvictFirst>(A, B1, B1_scale, C1, B2, B2_scale, C2, N, K);
}
extern "C" __global__ void __launch_bounds__(FGB_BLOCK) fp8_gemv_ada_batchm_l2ef_dual_6(FGB_DUAL_ARGS) {
    fgb_dual<6, FgbLoadEvictFirst>(A, B1, B1_scale, C1, B2, B2_scale, C2, N, K);
}
extern "C" __global__ void __launch_bounds__(FGB_BLOCK) fp8_gemv_ada_batchm_l2ef_dual_7(FGB_DUAL_ARGS) {
    fgb_dual<7, FgbLoadEvictFirst>(A, B1, B1_scale, C1, B2, B2_scale, C2, N, K);
}
extern "C" __global__ void __launch_bounds__(FGB_BLOCK) fp8_gemv_ada_batchm_l2ef_dual_8(FGB_DUAL_ARGS) {
    fgb_dual<8, FgbLoadEvictFirst>(A, B1, B1_scale, C1, B2, B2_scale, C2, N, K);
}

// down(silu(gate) * up), evict-first L2 weight loads.
extern "C" __global__ void __launch_bounds__(FGB_BLOCK) fp8_gemv_ada_batchm_l2ef_silu_input_2(FGB_SILU_ARGS) {
    fgb_silu<2, FgbLoadEvictFirst>(gate_out, up_out, B, block_scale, C, N, K);
}
extern "C" __global__ void __launch_bounds__(FGB_BLOCK) fp8_gemv_ada_batchm_l2ef_silu_input_3(FGB_SILU_ARGS) {
    fgb_silu<3, FgbLoadEvictFirst>(gate_out, up_out, B, block_scale, C, N, K);
}
extern "C" __global__ void __launch_bounds__(FGB_BLOCK) fp8_gemv_ada_batchm_l2ef_silu_input_4(FGB_SILU_ARGS) {
    fgb_silu<4, FgbLoadEvictFirst>(gate_out, up_out, B, block_scale, C, N, K);
}
extern "C" __global__ void __launch_bounds__(FGB_BLOCK) fp8_gemv_ada_batchm_l2ef_silu_input_5(FGB_SILU_ARGS) {
    fgb_silu<5, FgbLoadEvictFirst>(gate_out, up_out, B, block_scale, C, N, K);
}
extern "C" __global__ void __launch_bounds__(FGB_BLOCK) fp8_gemv_ada_batchm_l2ef_silu_input_6(FGB_SILU_ARGS) {
    fgb_silu<6, FgbLoadEvictFirst>(gate_out, up_out, B, block_scale, C, N, K);
}
extern "C" __global__ void __launch_bounds__(FGB_BLOCK) fp8_gemv_ada_batchm_l2ef_silu_input_7(FGB_SILU_ARGS) {
    fgb_silu<7, FgbLoadEvictFirst>(gate_out, up_out, B, block_scale, C, N, K);
}
extern "C" __global__ void __launch_bounds__(FGB_BLOCK) fp8_gemv_ada_batchm_l2ef_silu_input_8(FGB_SILU_ARGS) {
    fgb_silu<8, FgbLoadEvictFirst>(gate_out, up_out, B, block_scale, C, N, K);
}
