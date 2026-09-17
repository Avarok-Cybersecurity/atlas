// SPDX-License-Identifier: AGPL-3.0-only

// Atlas W4A16 GEMM v3 — MiniMax-only shadow kernel.
//
// Relative to v2 (128×128×32 tile, 8 warps, 2-stage):
//   - K_STEP doubled from 32 → 64 (4 scale groups per iter, not 2)
//   - Iterations per CTA halved (K/32 → K/64, e.g. 96→48 for K=3072)
//   - Per-iter MMA count doubled: each warp runs 32 m16n8k32 MMAs
//     (16 N-tiles × 2 K-halves) instead of 16
//
// Tradeoff: SMEM grows to ~55 KB per CTA → occupancy drops from 3 CTAs/SM
// (v2) to 1 CTA/SM. We bet that halving sync/dequant overhead and doubling
// per-iter MMA work beats the occupancy loss. Outcome unknown — measure.
//
// SMEM layout (K_STEP_T=64, 2 stages):
//   A  [2][128][64+8] × 2 = 36,864 B
//   Bp [2][32][128+16]    =  9,216 B
//   Bs [2][4][128+16]     =  1,152 B   (4 scale groups for K=64)
//   B_fp8 [128][64]       =  8,192 B   (single reusable dequant buffer)
//   LUT [16]              =     64 B
//   Total ≈ 55,488 B → 1 CTA/SM
// ═══════════════════════════════════════════════════════════════════

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define V3_M_TILE       64
#define V3_N_TILE       128
#define V3_K_STEP       64      // was 32 in v2
#define V3_PAD          8       // (64+8)*2 = 144 ≡ 0 mod 16
#define V3_BP_PAD       16      // 128+16 = 144
#define V3_GROUP_SIZE   16
#define V3_NUM_GROUPS   4       // K_STEP / GROUP_SIZE = 64/16

__device__ __constant__ float V3_E2M1_LUT[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

__device__ __forceinline__ void v3_cp_async_pred_16(void* dst_smem, const void* src_gmem, bool pred) {
    unsigned int dst = __cvta_generic_to_shared(dst_smem);
    unsigned int src_bytes = pred ? 16 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
                 :: "r"(dst), "l"(src_gmem), "r"(src_bytes));
}

__device__ __forceinline__ void v3_cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}

__device__ __forceinline__ void v3_cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;");
}

__device__ __forceinline__ unsigned int v3_bf16x4_to_e4m3x4(const unsigned short* src) {
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
__launch_bounds__(256, 1)
void w4a16_gemm_t_m128_v3(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * V3_N_TILE;
    const unsigned int cta_m = blockIdx.y * (2 * V3_M_TILE);
    if (cta_m >= M) return;

    const unsigned int warp_id = threadIdx.x >> 5;     // 0..7
    const unsigned int lane_id = threadIdx.x & 31;
    const unsigned int chunk   = warp_id >> 2;         // 0 or 1
    const unsigned int sub     = warp_id & 3;          // 0..3 within chunk
    const unsigned int warp_m_offset = sub * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[2][2 * V3_M_TILE][V3_K_STEP + V3_PAD];
    __shared__ unsigned char smem_Bp[2][V3_K_STEP / 2][V3_N_TILE + V3_BP_PAD];
    __shared__ unsigned char smem_Bs[2][V3_NUM_GROUPS][V3_N_TILE + V3_BP_PAD];
    __shared__ unsigned char smem_B_fp8[V3_N_TILE][V3_K_STEP];
    __shared__ float smem_LUT[16];

    if (threadIdx.x < 16) smem_LUT[threadIdx.x] = V3_E2M1_LUT[threadIdx.x];

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.f; acc[i][1] = 0.f; acc[i][2] = 0.f; acc[i][3] = 0.f;
    }

    const unsigned int a_stride = V3_K_STEP + V3_PAD;

    // ── Load K=64 tile [buf] from global ────────────────────────────────
    // A: 128 rows × 64 cols BF16 = 16384 B. 256 threads × 4 rounds × 16 B.
    //    Each thread loads 4 different rows at 8 BF16 per load.
    // Bp: 32 K-pairs × 128 N = 4096 B. 256 threads × 1 round × 16 B.
    // Bs: 4 scale groups × 128 N = 512 B effective.
    #define V3_LOADS(buf, kb) do { \
        { \
            /* A load: 256 threads × 4 rounds covers 128 rows × 64 cols (16 B/load). */ \
            unsigned int a_row_base = threadIdx.x >> 3;        /* 0..31 */ \
            unsigned int a_col      = (threadIdx.x & 7) << 3;  /* 0/8/16/24/32/40/48/56 */ \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 4; rnd++) { \
                unsigned int row = (unsigned int)(rnd * 32) + a_row_base; \
                unsigned int gr  = cta_m + row; \
                v3_cp_async_pred_16(&smem_A[(buf)][row][a_col], \
                    &A[(unsigned long long)gr * K + gc], \
                    (gr < M) && (gc + 7 < K)); \
            } \
        } \
        { \
            /* Bp load: 256 threads × 1 round covers 32 K-pairs × 128 N. */ \
            unsigned int kp  = threadIdx.x >> 3;         /* 0..31 */ \
            unsigned int ns  = (threadIdx.x & 7) << 4;   /* 0/16/.../112 */ \
            unsigned int gke = (kb) + (kp << 1); \
            unsigned int gns = cta_n + ns; \
            v3_cp_async_pred_16(&smem_Bp[(buf)][kp][ns], \
                &B_packed[(unsigned long long)(gke >> 1) * N + gns], \
                (gke + 1 <= K) && (gns + 15 < N)); \
            /* Bs load: 4 scale groups × 128 N = 32 cp.asyncs, first 32 threads handle it. */ \
            if (kp < V3_NUM_GROUPS) { \
                unsigned int sg = (kb) / V3_GROUP_SIZE + kp; \
                v3_cp_async_pred_16(&smem_Bs[(buf)][kp][ns], \
                    &B_scale[(unsigned long long)sg * N + gns], \
                    (gns + 15 < N)); \
            } \
        } \
    } while(0)

    // Dequant B tile (buf) to smem_B_fp8. 128 threads do 128 N-columns.
    // K=64 has 4 scale groups: [0..15], [16..31], [32..47], [48..63].
    #define V3_DEQUANT(buf) do { \
        if (threadIdx.x < 128) { \
            unsigned int my_n = threadIdx.x; \
            unsigned char sb0 = smem_Bs[(buf)][0][my_n]; \
            unsigned char sb1 = smem_Bs[(buf)][1][my_n]; \
            unsigned char sb2 = smem_Bs[(buf)][2][my_n]; \
            unsigned char sb3 = smem_Bs[(buf)][3][my_n]; \
            __nv_fp8_e4m3 f0, f1, f2, f3; \
            *(unsigned char*)&f0 = sb0; *(unsigned char*)&f1 = sb1; \
            *(unsigned char*)&f2 = sb2; *(unsigned char*)&f3 = sb3; \
            float sv0 = (float)f0 * scale2, sv1 = (float)f1 * scale2; \
            float sv2 = (float)f2 * scale2, sv3 = (float)f3 * scale2; \
            /* Bp has K/2 rows × N cols; each "row" is 2 K values packed. \
             * K=64 → 32 Bp rows. Groups of 8 Bp rows == 16 K values == 1 scale group. */ \
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
            _Pragma("unroll") \
            for (int kp = 16; kp < 24; kp++) { \
                unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
                float lo = smem_LUT[packed & 0xF] * sv2; \
                float hi = smem_LUT[packed >> 4]  * sv2; \
                unsigned short fp8_pair; \
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                             : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
                *(unsigned short*)&smem_B_fp8[my_n][kp * 2] = fp8_pair; \
            } \
            _Pragma("unroll") \
            for (int kp = 24; kp < 32; kp++) { \
                unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
                float lo = smem_LUT[packed & 0xF] * sv3; \
                float hi = smem_LUT[packed >> 4]  * sv3; \
                unsigned short fp8_pair; \
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                             : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
                *(unsigned short*)&smem_B_fp8[my_n][kp * 2] = fp8_pair; \
            } \
        } \
    } while(0)

    // Compute. Each warp does 16 N-tiles × 2 K-halves = 32 MMAs per iter.
    #define V3_COMPUTE(a_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A[(a_buf)]; \
        unsigned int fr0, fr1, a0, a1, a2, a3, a0b, a1b, a2b, a3b; \
        fr0 = chunk * V3_M_TILE + warp_m_offset + group_id; \
        fr1 = fr0 + 8; \
        /* K first half: cols tid*4..tid*4+3 and 16+tid*4..16+tid*4+3 */ \
        a0  = v3_bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4]); \
        a1  = v3_bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid * 4]); \
        a2  = v3_bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid * 4]); \
        a3  = v3_bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid * 4]); \
        /* K second half: cols 32+... and 48+... */ \
        a0b = v3_bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 32 + tid * 4]); \
        a1b = v3_bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 32 + tid * 4]); \
        a2b = v3_bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 48 + tid * 4]); \
        a3b = v3_bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 48 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            /* First K-half MMA (K cols 0..31) */ \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8[nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8[nc][16 + 4 * tid]; \
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                :"=f"(acc[nt][0]),"=f"(acc[nt][1]),"=f"(acc[nt][2]),"=f"(acc[nt][3]) \
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
                 "f"(acc[nt][0]),"f"(acc[nt][1]),"f"(acc[nt][2]),"f"(acc[nt][3])); \
            /* Second K-half MMA (K cols 32..63) */ \
            unsigned int b0b = *(const unsigned int*)&smem_B_fp8[nc][32 + 4 * tid]; \
            unsigned int b1b = *(const unsigned int*)&smem_B_fp8[nc][48 + 4 * tid]; \
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                :"=f"(acc[nt][0]),"=f"(acc[nt][1]),"=f"(acc[nt][2]),"=f"(acc[nt][3]) \
                :"r"(a0b),"r"(a1b),"r"(a2b),"r"(a3b),"r"(b0b),"r"(b1b), \
                 "f"(acc[nt][0]),"f"(acc[nt][1]),"f"(acc[nt][2]),"f"(acc[nt][3])); \
        } \
    } while(0)

    // 2-stage pipeline — same structure as v2.
    V3_LOADS(0, 0);
    v3_cp_async_commit();
    v3_cp_async_wait_all();
    __syncthreads();
    V3_DEQUANT(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = V3_K_STEP; k_base < K; k_base += V3_K_STEP) {
        int nxt = 1 - cur;
        V3_LOADS(nxt, k_base);
        v3_cp_async_commit();
        V3_COMPUTE(cur);
        v3_cp_async_wait_all();
        __syncthreads();
        V3_DEQUANT(nxt);
        __syncthreads();
        cur = nxt;
    }
    V3_COMPUTE(cur);

    #undef V3_LOADS
    #undef V3_DEQUANT
    #undef V3_COMPUTE

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt * 8 + tid * 2;
        unsigned int row_base = cta_m + chunk * V3_M_TILE + warp_m_offset;
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
