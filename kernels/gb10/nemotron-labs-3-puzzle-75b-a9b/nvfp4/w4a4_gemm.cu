// SPDX-License-Identifier: AGPL-3.0-only
//
// W4A4 NVFP4 prefill GEMM for the dense FFN gate/up/down — native FP4 tensor cores (sm_121a).
// mma.sync kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64 (E2M1 x E2M1, E4M3 group-16 scales).
// Validated standalone (rel_err 0 vs reference, 52.4 TFLOP/s) — see /workspace/fp4-swing/.
// Activation A and weight B are BOTH native NVFP4 (E2M1 nibble-packed [.,K/2] + E4M3 [.,K/16]
// group-16 scale). Weight has an extra global FP32 scale2; activation scale2 passed as 1.0.
// MUST be compiled for sm_121a (FP4 MMA): KERNEL.toml extra_nvcc_flags=["-arch=sm_121a"].
//
// Output C is BF16 [M, N]. C[m,n] = scaleA2 * scaleB2 * sum_k deq(A) * deq(B).
#include <cuda_bf16.h>
#include <cstdint>

#define W4A4_BM 128
#define W4A4_BN 128
#define W4A4_KSTEP 64
#define W4A4_THREADS 256
#define W4A4_ABYTES 32   // 64 nibble-packed e2m1 = 32 bytes
#define W4A4_SFCNT  4     // 4 e4m3 group-16 scales per 64-K row

__device__ __forceinline__ void w4a4_cpa16(void* d, const void* s) {
    unsigned x = __cvta_generic_to_shared(d);
    asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n" ::"r"(x), "l"(s));
}
__device__ __forceinline__ void w4a4_cpa4(void* d, const void* s) {
    unsigned x = __cvta_generic_to_shared(d);
    asm volatile("cp.async.ca.shared.global [%0],[%1],4;\n" ::"r"(x), "l"(s));
}
__device__ __forceinline__ void w4a4_commit() { asm volatile("cp.async.commit_group;"); }
__device__ __forceinline__ void w4a4_wait() { asm volatile("cp.async.wait_group 0;"); }

__device__ __forceinline__ void w4a4_mma(float* acc,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1,
    uint32_t sfa, uint32_t sfb) {
    uint16_t z = 0;
    asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%11,%12},{%13},{%14,%15};\n"
      : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
        "r"(sfa), "h"(z), "h"(z), "r"(sfb), "h"(z), "h"(z));
}

// A_packed [M,K/2] e2m1; A_sf [M,K/16] e4m3; B_packed [N,K/2]; B_sf [N,K/16]; scaleA2,scaleB2 global.
__device__ __forceinline__ void w4a4_gemm_impl(
    const uint8_t* __restrict__ A_packed, const uint8_t* __restrict__ A_sf,
    const uint8_t* __restrict__ B_packed, const uint8_t* __restrict__ B_sf,
    __nv_bfloat16* __restrict__ C, float scaleA2, float scaleB2, int M, int N, int K,
    const unsigned cta_m, const unsigned cta_n) {
    const float sA2 = scaleA2;
    const unsigned warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const unsigned wm = warp * 16, gid = lane >> 2, tid = lane & 3;
    __shared__ uint8_t sAf[2][W4A4_BM][W4A4_ABYTES], sBf[2][W4A4_BN][W4A4_ABYTES];
    __shared__ uint8_t sSA[2][W4A4_BM][W4A4_SFCNT], sSB[2][W4A4_BN][W4A4_SFCNT];
    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) { acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0; }
    const int KB = K / 2, KS = K / 16;
    const unsigned Mm1 = (unsigned)(M - 1), Nm1 = (unsigned)(N - 1);
    // Clamp global row to [0,M-1]/[0,N-1] so M/N not a multiple of the tile never
    // reads OOB; padding rows compute garbage that the bounds-checked epilogue discards.
    #define W4A4_LOAD(buf, kb) do { \
        int ar = threadIdx.x >> 1, ac = (threadIdx.x & 1) << 4; \
        if (ar < W4A4_BM) { unsigned ga = min(cta_m + (unsigned)ar, Mm1); \
            w4a4_cpa16(&sAf[buf][ar][ac], A_packed + (size_t)ga * KB + (kb) / 2 + ac); } \
        for (int r = threadIdx.x; r < W4A4_BN * 2; r += W4A4_THREADS) { int n = r >> 1, c = (r & 1) << 4; \
            unsigned gb = min(cta_n + (unsigned)n, Nm1); \
            w4a4_cpa16(&sBf[buf][n][c], B_packed + (size_t)gb * KB + (kb) / 2 + c); } \
        if (threadIdx.x < W4A4_BM) { unsigned ga = min(cta_m + threadIdx.x, Mm1); \
            w4a4_cpa4(&sSA[buf][threadIdx.x][0], A_sf + (size_t)ga * KS + (kb) / 16); } \
        for (int n = threadIdx.x; n < W4A4_BN; n += W4A4_THREADS) { unsigned gb = min(cta_n + (unsigned)n, Nm1); \
            w4a4_cpa4(&sSB[buf][n][0], B_sf + (size_t)gb * KS + (kb) / 16); } \
    } while (0)
    W4A4_LOAD(0, 0); w4a4_commit(); w4a4_wait(); __syncthreads();
    int buf = 0;
    for (int kb = W4A4_KSTEP; kb < K; kb += W4A4_KSTEP) {
        int nb = buf ^ 1; W4A4_LOAD(nb, kb); w4a4_commit();
        unsigned fr0 = wm + gid, fr1 = fr0 + 8;
        uint32_t a0 = *(uint32_t*)&sAf[buf][fr0][4 * tid], a1 = *(uint32_t*)&sAf[buf][fr1][4 * tid];
        uint32_t a2 = *(uint32_t*)&sAf[buf][fr0][16 + 4 * tid], a3 = *(uint32_t*)&sAf[buf][fr1][16 + 4 * tid];
        uint32_t sfa = *(uint32_t*)&sSA[buf][wm + ((tid & 1) << 3) + gid][0];
        #pragma unroll
        for (int nt = 0; nt < 16; nt++) { unsigned nc = nt * 8 + gid;
            uint32_t b0 = *(uint32_t*)&sBf[buf][nc][4 * tid], b1 = *(uint32_t*)&sBf[buf][nc][16 + 4 * tid];
            uint32_t sfb = *(uint32_t*)&sSB[buf][nc][0];
            w4a4_mma(acc[nt], a0, a1, a2, a3, b0, b1, sfa, sfb); }
        w4a4_wait(); __syncthreads(); buf = nb;
    }
    { unsigned fr0 = wm + gid, fr1 = fr0 + 8;
      uint32_t a0 = *(uint32_t*)&sAf[buf][fr0][4 * tid], a1 = *(uint32_t*)&sAf[buf][fr1][4 * tid];
      uint32_t a2 = *(uint32_t*)&sAf[buf][fr0][16 + 4 * tid], a3 = *(uint32_t*)&sAf[buf][fr1][16 + 4 * tid];
      uint32_t sfa = *(uint32_t*)&sSA[buf][wm + ((tid & 1) << 3) + gid][0];
      #pragma unroll
      for (int nt = 0; nt < 16; nt++) { unsigned nc = nt * 8 + gid;
        uint32_t b0 = *(uint32_t*)&sBf[buf][nc][4 * tid], b1 = *(uint32_t*)&sBf[buf][nc][16 + 4 * tid];
        uint32_t sfb = *(uint32_t*)&sSB[buf][nc][0];
        w4a4_mma(acc[nt], a0, a1, a2, a3, b0, b1, sfa, sfb); } }
    const float g = sA2 * scaleB2;
    unsigned fr0 = wm + gid, fr1 = fr0 + 8;
    #pragma unroll
    for (int nt = 0; nt < 16; nt++) { unsigned base = cta_n + nt * 8;
        unsigned r0 = cta_m + fr0, r1 = cta_m + fr1, c0 = base + tid * 2, c1 = c0 + 1;
        if (r0 < (unsigned)M && c0 < (unsigned)N) C[(size_t)r0 * N + c0] = __float2bfloat16(acc[nt][0] * g);
        if (r0 < (unsigned)M && c1 < (unsigned)N) C[(size_t)r0 * N + c1] = __float2bfloat16(acc[nt][1] * g);
        if (r1 < (unsigned)M && c0 < (unsigned)N) C[(size_t)r1 * N + c0] = __float2bfloat16(acc[nt][2] * g);
        if (r1 < (unsigned)M && c1 < (unsigned)N) C[(size_t)r1 * N + c1] = __float2bfloat16(acc[nt][3] * g);
    }
}

// Grid wrappers over the same body. w4a4_gemm keeps the legacy n-fast order
// (qwen dense-FFN launcher shape); _mfast puts M on the fast axis so the
// M-blocks sharing a B panel run co-resident and the panel stays in L2 --
// with n-fast the full B matrix would be re-streamed once per M-block
// (~8x at 1k tokens), same failure mode measured on the FP8 GEMMs.
extern "C" __global__ __launch_bounds__(W4A4_THREADS, 2) void w4a4_gemm(
    const uint8_t* __restrict__ A_packed, const uint8_t* __restrict__ A_sf,
    const uint8_t* __restrict__ B_packed, const uint8_t* __restrict__ B_sf,
    __nv_bfloat16* __restrict__ C, float scaleA2, float scaleB2, int M, int N, int K) {
    w4a4_gemm_impl(A_packed, A_sf, B_packed, B_sf, C, scaleA2, scaleB2, M, N, K,
                   blockIdx.y * W4A4_BM, blockIdx.x * W4A4_BN);
}


// ─────────────────────────────────────────────────────────────────────
// w4a4_gemm_mfast — production schedule (v3 of the microbench sweep at
// /tmp scratchpad w4a4bench, 2026-07-14):
//   - fp8_fp8_gemm_t_m128 structure: 128 threads / 4 warps, each warp owns
//     16 rows of both 64-row chunks, so B fragments + SFB are loaded ONCE
//     per (k-step, n-tile) and feed both chunks' MMAs. The original 8-warp
//     one-chunk-per-warp schedule has no B reuse and measured 92 TFLOP/s;
//     this one 103.6.
//   - 3-stage cp.async pipeline (2 loads in flight): 106.5 TFLOP/s.
//   - M on the fast grid axis: the M-blocks sharing a B panel run
//     co-resident so the panel stays in L2 (n-fast re-streams all of B
//     once per M-block; same failure measured on the FP8 GEMMs).
// Bit-identical to w4a4_gemm (max_rel = 0 over 18.4M outputs at the
// in_proj shape M=1023 N=18048 K=4096).
// ─────────────────────────────────────────────────────────────────────
#define V3_THREADS 128

__device__ __forceinline__ void v3_cpa16(void* d, const void* s) {
    unsigned x = __cvta_generic_to_shared(d);
    asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n" ::"r"(x), "l"(s));
}
__device__ __forceinline__ void v3_cpa4(void* d, const void* s) {
    unsigned x = __cvta_generic_to_shared(d);
    asm volatile("cp.async.ca.shared.global [%0],[%1],4;\n" ::"r"(x), "l"(s));
}
__device__ __forceinline__ void v3_commit() { asm volatile("cp.async.commit_group;"); }
__device__ __forceinline__ void v3_wait1() { asm volatile("cp.async.wait_group 1;"); }
__device__ __forceinline__ void v3_wait() { asm volatile("cp.async.wait_group 0;"); }

__device__ __forceinline__ void v3_mma(float* acc,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1,
    uint32_t sfa, uint32_t sfb) {
    uint16_t z = 0;
    asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%11,%12},{%13},{%14,%15};\n"
      : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
        "r"(sfa), "h"(z), "h"(z), "r"(sfb), "h"(z), "h"(z));
}

extern "C" __global__ __launch_bounds__(V3_THREADS, 3) void w4a4_gemm_mfast(
    const uint8_t* __restrict__ A_packed, const uint8_t* __restrict__ A_sf,
    const uint8_t* __restrict__ B_packed, const uint8_t* __restrict__ B_sf,
    __nv_bfloat16* __restrict__ C, float scaleA2, float scaleB2, int M, int N, int K) {
    const unsigned cta_m = blockIdx.x * 128u;   // m-fast
    const unsigned cta_n = blockIdx.y * 128u;
    if (cta_m >= (unsigned)M) return;
    const unsigned warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const unsigned wm = warp * 16, gid = lane >> 2, tid = lane & 3;

    __shared__ uint8_t sAf[3][128][32], sBf[3][128][32];
    __shared__ uint8_t sSA[3][128][4],  sSB[3][128][4];

    float acc0[16][4], acc1[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc0[i][0]=acc0[i][1]=acc0[i][2]=acc0[i][3]=0;
        acc1[i][0]=acc1[i][1]=acc1[i][2]=acc1[i][3]=0;
    }
    const int KB = K / 2, KS = K / 16;
    const unsigned Mm1 = (unsigned)(M - 1), Nm1 = (unsigned)(N - 1);

    #define V3_LOAD(buf, kb) do { \
        _Pragma("unroll") \
        for (unsigned ch = 0; ch < 2u; ch++) { \
            unsigned idx = threadIdx.x + ch * 128u; \
            unsigned r = idx >> 1, c = (idx & 1u) << 4; \
            unsigned ga = min(cta_m + r, Mm1); \
            v3_cpa16(&sAf[buf][r][c], A_packed + (size_t)ga * KB + (kb)/2 + c); \
            unsigned gb = min(cta_n + r, Nm1); \
            v3_cpa16(&sBf[buf][r][c], B_packed + (size_t)gb * KB + (kb)/2 + c); \
        } \
        { unsigned r = threadIdx.x; \
          unsigned ga = min(cta_m + r, Mm1); \
          v3_cpa4(&sSA[buf][r][0], A_sf + (size_t)ga * KS + (kb)/16); \
          unsigned gb = min(cta_n + r, Nm1); \
          v3_cpa4(&sSB[buf][r][0], B_sf + (size_t)gb * KS + (kb)/16); } \
    } while (0)

    #define V3_COMPUTE(buf) do { \
        const unsigned fr0 = wm + gid, fr1 = fr0 + 8; \
        uint32_t a0 = *(uint32_t*)&sAf[buf][fr0][4*tid],      a1 = *(uint32_t*)&sAf[buf][fr1][4*tid]; \
        uint32_t a2 = *(uint32_t*)&sAf[buf][fr0][16 + 4*tid], a3 = *(uint32_t*)&sAf[buf][fr1][16 + 4*tid]; \
        uint32_t e0 = *(uint32_t*)&sAf[buf][64 + fr0][4*tid],      e1 = *(uint32_t*)&sAf[buf][64 + fr1][4*tid]; \
        uint32_t e2 = *(uint32_t*)&sAf[buf][64 + fr0][16 + 4*tid], e3 = *(uint32_t*)&sAf[buf][64 + fr1][16 + 4*tid]; \
        uint32_t sfa0 = *(uint32_t*)&sSA[buf][wm + ((tid & 1u) << 3) + gid][0]; \
        uint32_t sfa1 = *(uint32_t*)&sSA[buf][64 + wm + ((tid & 1u) << 3) + gid][0]; \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned nc = nt * 8 + gid; \
            uint32_t b0 = *(uint32_t*)&sBf[buf][nc][4*tid], b1 = *(uint32_t*)&sBf[buf][nc][16 + 4*tid]; \
            uint32_t sfb = *(uint32_t*)&sSB[buf][nc][0]; \
            v3_mma(acc0[nt], a0, a1, a2, a3, b0, b1, sfa0, sfb); \
            v3_mma(acc1[nt], e0, e1, e2, e3, b0, b1, sfa1, sfb); \
        } \
    } while (0)

    // 3-stage: 2 loads in flight; compute stage s while s+1 lands and s+2 issues.
    V3_LOAD(0, 0); v3_commit();
    if (K > 64) { V3_LOAD(1, 64); v3_commit(); }
    v3_wait1(); __syncthreads();
    int buf = 0;
    for (int kb = 128; kb < K; kb += 64) {
        int nb = (buf + 2) % 3;
        V3_LOAD(nb, kb); v3_commit();
        V3_COMPUTE(buf);
        v3_wait1(); __syncthreads();
        buf = (buf + 1) % 3;
    }
    if (K > 64) { v3_wait(); __syncthreads(); V3_COMPUTE(buf); buf = (buf + 1) % 3; }
    v3_wait(); __syncthreads();
    V3_COMPUTE(buf);
    #undef V3_LOAD
    #undef V3_COMPUTE

    const float g = scaleA2 * scaleB2;
    const unsigned fr0 = wm + gid, fr1 = fr0 + 8;
    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned c0 = cta_n + nt * 8 + tid * 2, c1 = c0 + 1;
        unsigned r0 = cta_m + fr0, r1 = cta_m + fr1;
        if (r0 < (unsigned)M && c0 < (unsigned)N) C[(size_t)r0 * N + c0] = __float2bfloat16(acc0[nt][0] * g);
        if (r0 < (unsigned)M && c1 < (unsigned)N) C[(size_t)r0 * N + c1] = __float2bfloat16(acc0[nt][1] * g);
        if (r1 < (unsigned)M && c0 < (unsigned)N) C[(size_t)r1 * N + c0] = __float2bfloat16(acc0[nt][2] * g);
        if (r1 < (unsigned)M && c1 < (unsigned)N) C[(size_t)r1 * N + c1] = __float2bfloat16(acc0[nt][3] * g);
        unsigned r2 = r0 + 64, r3 = r1 + 64;
        if (r2 < (unsigned)M && c0 < (unsigned)N) C[(size_t)r2 * N + c0] = __float2bfloat16(acc1[nt][0] * g);
        if (r2 < (unsigned)M && c1 < (unsigned)N) C[(size_t)r2 * N + c1] = __float2bfloat16(acc1[nt][1] * g);
        if (r3 < (unsigned)M && c0 < (unsigned)N) C[(size_t)r3 * N + c0] = __float2bfloat16(acc1[nt][2] * g);
        if (r3 < (unsigned)M && c1 < (unsigned)N) C[(size_t)r3 * N + c1] = __float2bfloat16(acc1[nt][3] * g);
    }
}


