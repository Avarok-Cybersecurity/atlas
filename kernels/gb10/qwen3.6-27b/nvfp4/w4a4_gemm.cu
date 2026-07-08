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
extern "C" __global__ __launch_bounds__(W4A4_THREADS, 2) void w4a4_gemm(
    const uint8_t* __restrict__ A_packed, const uint8_t* __restrict__ A_sf,
    const uint8_t* __restrict__ B_packed, const uint8_t* __restrict__ B_sf,
    __nv_bfloat16* __restrict__ C, float scaleA2, float scaleB2, int M, int N, int K) {
    const float sA2 = scaleA2;
    const unsigned cta_n = blockIdx.x * W4A4_BN, cta_m = blockIdx.y * W4A4_BM;
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
