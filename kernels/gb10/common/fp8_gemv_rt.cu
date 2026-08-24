// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b
//
// fp8_gemv_rt.cu — register-tiled batched FP8 GEMV for the DFlash drafter
// PROPOSE path (Phase G weights: B[N,K] FP8 E4M3 + per-row f32 scale).
//
//   C[M,N] (bf16) = A[M,K] (bf16) @ dequant(B)^T * row_scale[N]
//
// Why it exists (nsys node-trace + batchm_bench, 2026-08-19): the propose
// pipeline ran its M=8 GEMMs on prefill-class tile kernels
// (fp8_gemm_t_row_scaled M64-tile = 87% padding, _m16 = 50%), measuring
// ~100 GB/s while the register-tiled GEMV family proves 180+ GB/s at M=8
// on this memory system. This kernel is the FP8 twin of
// `w4a16_gemv_batch8_rt2`: T=2 adjacent output rows per 64-lane group, one
// activation load feeds both FMA chains (activation traffic, load
// instructions, and chain ILP all improved 2x vs one-output-per-group).
//
// DRAFTER-SIDE NUMERICS ARE CORRECTNESS-FREE: under strict-argmax accept,
// draft quality only moves the accept RATE, never the output tokens. So
// unlike the w4a16 verify family there is NO bit-order contract here; this
// kernel uses a plain single-phase accumulation and applies row_scale once
// at write-out. Accuracy is gated by accept-rate A/B on the live server
// (kill-switch: ATLAS_NO_DFLASH_FP8_RT=1 restores the tile kernels).
//
// Dequant uses the proven 256-entry E4M3 smem LUT (branchless, no
// cvt.rn.satfinite dependency on SM121 — same rationale as w8a16_gemv).
//
// Grid: (ceil(N/8), 1, 1)  Block: (256, 1, 1). Requires K % 16 == 0
// (holds for h=5120 and inter=17408). M clamped to 8 by the Rust launcher.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define RT_BLOCK 256
#define RT_GROUPS 4
#define RT_T 2
#define RT_MAXM 8

extern "C" __global__ void fp8_gemv_rowscale_batch8_rt2(
    const __nv_bfloat16* __restrict__ A,   // [M, K] bf16
    const unsigned char* __restrict__ B,   // [N, K] fp8 e4m3
    const float* __restrict__ row_scale,   // [N] f32
    __nv_bfloat16* __restrict__ C,         // [M, N] bf16
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int tpo = RT_BLOCK / RT_GROUPS;     // 64 lanes per group
    const unsigned int local_out = threadIdx.x / tpo;  // 0..3
    const unsigned int lane = threadIdx.x % tpo;       // 0..63
    const unsigned int n0 = (blockIdx.x * RT_GROUPS + local_out) * RT_T;

    __shared__ float s_lut[256];
    {
        __nv_fp8_e4m3 f;
        *(unsigned char*)&f = (unsigned char)threadIdx.x;
        s_lut[threadIdx.x] = (float)f;
    }
    __syncthreads();
    if (n0 >= N) return;

    const unsigned int K16 = K / 16;

    float acc[RT_T][RT_MAXM];
    #pragma unroll
    for (int o = 0; o < RT_T; o++)
        #pragma unroll
        for (int t = 0; t < RT_MAXM; t++) acc[o][t] = 0.0f;

    for (unsigned int kk = lane; kk < K16; kk += tpo) {
        // T weight chunks: 16 FP8 bytes each, one uint4 load per output.
        float wl[RT_T][16];
        #pragma unroll
        for (int o = 0; o < RT_T; o++) {
            const unsigned long long n = n0 + o;
            if (n < N) {
                uint4 wb = *(const uint4*)(B + n * K + (unsigned long long)kk * 16u);
                const unsigned int wr[4] = {wb.x, wb.y, wb.z, wb.w};
                #pragma unroll
                for (int w = 0; w < 4; w++)
                    #pragma unroll
                    for (int b = 0; b < 4; b++)
                        wl[o][w * 4 + b] = s_lut[(wr[w] >> (b * 8)) & 0xFF];
            } else {
                #pragma unroll
                for (int i = 0; i < 16; i++) wl[o][i] = 0.0f;
            }
        }

        #pragma unroll
        for (int t = 0; t < RT_MAXM; t++) {
            if ((unsigned int)t >= M) continue;
            const __nv_bfloat16* At = A + (unsigned long long)t * K;
            // ONE activation load per (chunk, row) feeding both chains.
            uint4 a_lo = ((const uint4*)At)[kk * 2];
            uint4 a_hi = ((const uint4*)At)[kk * 2 + 1];
            const unsigned int ar[8] = {a_lo.x, a_lo.y, a_lo.z, a_lo.w,
                                        a_hi.x, a_hi.y, a_hi.z, a_hi.w};
            #pragma unroll
            for (int o = 0; o < RT_T; o++) {
                float part = 0.0f;
                #pragma unroll
                for (int b = 0; b < 8; b++) {
                    float2 af = __bfloat1622float2(*(const __nv_bfloat162*)&ar[b]);
                    part = fmaf(af.x, wl[o][b * 2], part);
                    part = fmaf(af.y, wl[o][b * 2 + 1], part);
                }
                acc[o][t] += part;
            }
        }
    }

    // 64-lane reduce: 32-lane shuffle tree per warp, cross-warp via smem.
    __shared__ float s_red[RT_MAXM][RT_GROUPS * RT_T * 2];
    const unsigned int warp_in_out = lane / 32u;
    #pragma unroll
    for (int o = 0; o < RT_T; o++) {
        #pragma unroll
        for (int t = 0; t < RT_MAXM; t++) {
            if ((unsigned int)t >= M) continue;
            float a = acc[o][t];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                a += __shfl_down_sync(0xFFFFFFFF, a, off);
            if ((lane & 31u) == 0u)
                s_red[t][(local_out * RT_T + o) * 2 + warp_in_out] = a;
        }
    }
    __syncthreads();

    if (lane == 0) {
        #pragma unroll
        for (int o = 0; o < RT_T; o++) {
            const unsigned int n = n0 + o;
            if (n >= N) continue;
            const float rs = row_scale[n];
            #pragma unroll
            for (int t = 0; t < RT_MAXM; t++) {
                if ((unsigned int)t >= M) continue;
                float r = s_red[t][(local_out * RT_T + o) * 2]
                        + s_red[t][(local_out * RT_T + o) * 2 + 1];
                C[(unsigned long long)t * N + n] = __float2bfloat16(r * rs);
            }
        }
    }
}
