// SPDX-License-Identifier: AGPL-3.0-only

// Atlas W4A16 GEMM v2 — MiniMax-only shadow kernel.
//
// Baseline (`w4a16_gemm_t_m128` in `w4a16_gemm.cu`):
//   - 128 (M) × 128 (N) × 32 (K per step) CTA tile
//   - blockDim 128 (4 warps), 2-stage cp.async pipeline
//   - Chunk 0 (rows 0-63) and chunk 1 (rows 64-127) MMAs computed
//     **serially**: 4 warps run 64 MMAs for chunk 0, then 64 for chunk 1.
//
// v2 change (this file):
//   - blockDim 256 (8 warps). Warps 0-3 own chunk 0 rows, warps 4-7 own
//     chunk 1 rows → both chunks' MMAs run **in parallel**.
//   - SMEM layout and 2-stage pipeline unchanged (still 3 CTAs/SM).
//
// Occupancy: SMEM footprint identical to v1 (~29.8 KB/CTA → 3 CTAs/SM),
// so we get 3 × 256 = 768 threads/SM vs v1's 3 × 128 = 384 threads/SM.
// More warps/SM = more MMA pipeline slots in flight.
//
// Only a new `w4a16_gemm_t_m128_v2` symbol is emitted. No other kernels
// in this file — v1 symbols stay in `w4a16_gemm.cu` untouched.
// ═══════════════════════════════════════════════════════════════════

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define M_TILE     64
#define N_TILE_LG  128
#define K_STEP_T   32
#define PAD_T      8
#define BP_PAD     16
#define GROUP_SIZE 16

__device__ __constant__ float E2M1_LUT_V2[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

__device__ __forceinline__ void v2_cp_async_pred_16(void* dst_smem, const void* src_gmem, bool pred) {
    unsigned int dst = __cvta_generic_to_shared(dst_smem);
    unsigned int src_bytes = pred ? 16 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
                 :: "r"(dst), "l"(src_gmem), "r"(src_bytes));
}

__device__ __forceinline__ void v2_cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}

__device__ __forceinline__ void v2_cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;");
}

__device__ __forceinline__ unsigned int v2_bf16x4_to_e4m3x4(const unsigned short* src) {
    unsigned int p0 = *(const unsigned int*)src;
    unsigned int p1 = *(const unsigned int*)(src + 2);
    unsigned short bf0 = (unsigned short)(p0 & 0xFFFFu);
    unsigned short bf1 = (unsigned short)(p0 >> 16);
    unsigned short bf2 = (unsigned short)(p1 & 0xFFFFu);
    unsigned short bf3 = (unsigned short)(p1 >> 16);
    float f0, f1, f2, f3;
    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(f0) : "h"(bf0));
    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(f1) : "h"(bf1));
    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(f2) : "h"(bf2));
    asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(f3) : "h"(bf3));
    unsigned short h0, h1;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(h0) : "f"(f1), "f"(f0));
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(h1) : "f"(f3), "f"(f2));
    return ((unsigned int)h1 << 16) | (unsigned int)h0;
}

extern "C" __global__
__launch_bounds__(256, 3)
void w4a16_gemm_t_m128_v2(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * (2 * M_TILE);
    if (cta_m >= M) return;

    const unsigned int warp_id = threadIdx.x >> 5;    // 0..7
    const unsigned int lane_id = threadIdx.x & 31;
    const unsigned int chunk   = warp_id >> 2;        // 0 or 1
    const unsigned int sub     = warp_id & 3;         // 0..3 within chunk
    const unsigned int warp_m_offset = sub * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    // Same SMEM layout as v1's w4a16_gemm_t_m128 (2 stages).
    __shared__ __nv_bfloat16 smem_A[2][2 * M_TILE][K_STEP_T + PAD_T];
    __shared__ unsigned char smem_Bp[2][K_STEP_T / 2][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_Bs[2][K_STEP_T / GROUP_SIZE][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_B_fp8[N_TILE_LG][K_STEP_T];
    __shared__ float smem_LUT[16];

    if (threadIdx.x < 16) smem_LUT[threadIdx.x] = E2M1_LUT_V2[threadIdx.x];

    // Each warp owns 16 M-rows × 16 N-tiles of the output (16 MMAs/K-step).
    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.f; acc[i][1] = 0.f; acc[i][2] = 0.f; acc[i][3] = 0.f;
    }

    const unsigned int a_stride = K_STEP_T + PAD_T;

    // Load tile [buf] from global.
    //   A: 256 threads × 2 rounds × 16 B = 128×32×2 = 8192 B (one full tile)
    //   Bp: first 128 threads × 1 round × 16 B = 16×128 = 2048 B
    //   Bs: first 16 threads × 1 round × 16 B = 256 B effective
    #define V2_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 2;        /* 0..63 */ \
            unsigned int a_col      = (threadIdx.x & 3) << 3;  /* 0/8/16/24 */ \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 2; rnd++) { \
                unsigned int row = (unsigned int)(rnd * 64) + a_row_base; \
                unsigned int gr  = cta_m + row; \
                v2_cp_async_pred_16(&smem_A[(buf)][row][a_col], \
                    &A[(unsigned long long)gr * K + gc], \
                    (gr < M) && (gc + 7 < K)); \
            } \
        } \
        if (threadIdx.x < 128) { \
            unsigned int kp  = threadIdx.x >> 3;        /* 0..15 */ \
            unsigned int ns  = (threadIdx.x & 7) << 4;  /* 0/16/.../112 */ \
            unsigned int gke = (kb) + (kp << 1); \
            unsigned int gns = cta_n + ns; \
            v2_cp_async_pred_16(&smem_Bp[(buf)][kp][ns], \
                &B_packed[(unsigned long long)(gke >> 1) * N + gns], \
                (gke + 1 <= K) && (gns + 15 < N)); \
            if (kp < K_STEP_T / GROUP_SIZE) { \
                unsigned int sg = (kb) / GROUP_SIZE + kp; \
                v2_cp_async_pred_16(&smem_Bs[(buf)][kp][ns], \
                    &B_scale[(unsigned long long)sg * N + gns], \
                    (gns + 15 < N)); \
            } \
        } \
    } while(0)

    // Dequant buf's NVFP4 into shared FP8 E4M3 (128 threads, 1 N-col each).
    #define V2_DEQUANT(buf) do { \
        if (threadIdx.x < 128) { \
            unsigned int my_n = threadIdx.x; \
            unsigned char sb0 = smem_Bs[(buf)][0][my_n]; \
            unsigned char sb1 = smem_Bs[(buf)][1][my_n]; \
            __nv_fp8_e4m3 f0, f1; \
            *(unsigned char*)&f0 = sb0; *(unsigned char*)&f1 = sb1; \
            float sv0 = (float)f0 * scale2, sv1 = (float)f1 * scale2; \
            _Pragma("unroll") \
            for (int kp = 0; kp < 8; kp++) { \
                unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
                float lo = smem_LUT[packed & 0xF] * sv0; \
                float hi = smem_LUT[packed >> 4]  * sv0; \
                unsigned short fp8_pair; \
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                             : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
                *(unsigned short*)&smem_B_fp8[my_n][kp * 2] = fp8_pair; \
            } \
            _Pragma("unroll") \
            for (int kp = 8; kp < 16; kp++) { \
                unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
                float lo = smem_LUT[packed & 0xF] * sv1; \
                float hi = smem_LUT[packed >> 4]  * sv1; \
                unsigned short fp8_pair; \
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                             : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
                *(unsigned short*)&smem_B_fp8[my_n][kp * 2] = fp8_pair; \
            } \
        } \
    } while(0)

    // Compute. Each warp does 16 MMAs against its owned M-rows.
    // Chunk 0 warps (0-3) and chunk 1 warps (4-7) run in parallel, unlike
    // v1's serialized chunk-0-then-chunk-1 compute pass.
    #define V2_COMPUTE(a_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A[(a_buf)]; \
        unsigned int fr0, fr1, a0, a1, a2, a3; \
        fr0 = chunk * M_TILE + warp_m_offset + group_id; \
        fr1 = fr0 + 8; \
        a0 = v2_bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4]); \
        a1 = v2_bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid * 4]); \
        a2 = v2_bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid * 4]); \
        a3 = v2_bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8[nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8[nc][16 + 4 * tid]; \
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                :"=f"(acc[nt][0]),"=f"(acc[nt][1]),"=f"(acc[nt][2]),"=f"(acc[nt][3]) \
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
                 "f"(acc[nt][0]),"f"(acc[nt][1]),"f"(acc[nt][2]),"f"(acc[nt][3])); \
        } \
    } while(0)

    // 2-stage pipeline — same structure as v1's w4a16_gemm_t_m128.
    V2_LOADS(0, 0);
    v2_cp_async_commit();
    v2_cp_async_wait_all();
    __syncthreads();
    V2_DEQUANT(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        V2_LOADS(nxt, k_base);
        v2_cp_async_commit();
        V2_COMPUTE(cur);
        v2_cp_async_wait_all();
        __syncthreads();
        V2_DEQUANT(nxt);
        __syncthreads();
        cur = nxt;
    }
    V2_COMPUTE(cur);

    #undef V2_LOADS
    #undef V2_DEQUANT
    #undef V2_COMPUTE

    // Epilogue: each warp writes its own 16 M-rows × 128 N-cols (no shuffle).
    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt * 8 + tid * 2;
        unsigned int row_base = cta_m + chunk * M_TILE + warp_m_offset;
        unsigned int r0_lo = row_base + group_id;
        unsigned int r0_hi = row_base + group_id + 8;
        if (c0 + 1 < N) {
            if (r0_lo < M) {
                __nv_bfloat16 v_lo = __float2bfloat16_rn(acc[nt][0]);
                __nv_bfloat16 v_hi = __float2bfloat16_rn(acc[nt][1]);
                C[(unsigned long long)r0_lo * N + c0]     = v_lo;
                C[(unsigned long long)r0_lo * N + c0 + 1] = v_hi;
            }
            if (r0_hi < M) {
                __nv_bfloat16 v_lo = __float2bfloat16_rn(acc[nt][2]);
                __nv_bfloat16 v_hi = __float2bfloat16_rn(acc[nt][3]);
                C[(unsigned long long)r0_hi * N + c0]     = v_lo;
                C[(unsigned long long)r0_hi * N + c0 + 1] = v_hi;
            }
        }
    }
}
