// SPDX-License-Identifier: AGPL-3.0-only

// Atlas Grouped W4A16 GEMM for MoE — 35B model shadow.
//
// Optimizations over parent:
// - Transposed kernel: cp.async 2-stage double-buffered pipeline
// - Vectorized uint4 (128-bit) B_packed loads
// - Both-nibble extraction: one packed byte → 2 dequanted BF16 values
// - N_TILE=128 for reduced A bandwidth
// - K_STEP_T=32 (halves outer loop iterations)
// - Preloaded sorted_token_ids in smem for A indirection

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define M_TILE 64
#define N_TILE_SM 64
#define N_TILE_LG 128
#define K_STEP 16
#define K_STEP_T 32
#define PAD 2
#define PAD_T 8        // cp.async needs 16-byte aligned rows: (32+8)*2=80, 80%16=0
#define BP_PAD 16
#define GROUP_SIZE 16

__device__ __constant__ float E2M1_LUT_MOE[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// ═══════════════════════════════════════════════════════════════════
// Original N_TILE=64 pointer-table variant — kept for decode.
// ═══════════════════════════════════════════════════════════════════
extern "C" __global__ void moe_w4a16_grouped_gemm_ptrtable(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ B_packed_ptrs,
    const unsigned long long* __restrict__ B_scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    unsigned int num_experts,
    unsigned int N,
    unsigned int K
) {
    const unsigned int expert_id = blockIdx.z;
    if (expert_id >= num_experts) return;

    const int m_start = expert_offsets[expert_id];
    const int m_end = expert_offsets[expert_id + 1];
    const int M_expert = m_end - m_start;
    if (M_expert <= 0) return;

    const int cta_m_local = blockIdx.y * M_TILE;
    if (cta_m_local >= M_expert) return;

    const unsigned int cta_m = m_start + cta_m_local;
    const unsigned int cta_n = blockIdx.x * N_TILE_SM;

    const unsigned char* B_expert = (const unsigned char*)B_packed_ptrs[expert_id];
    const unsigned char* S_expert = (const unsigned char*)B_scale_ptrs[expert_id];
    const float scale2 = scale2_vals[expert_id];

    if (B_expert == 0) return;

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[M_TILE][K_STEP + PAD];
    __shared__ __nv_bfloat16 smem_B[K_STEP][N_TILE_SM + PAD];

    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int a_stride = K_STEP + PAD;
    const unsigned int b_stride = N_TILE_SM + PAD;
    const unsigned int M_eff = (unsigned int)M_expert;
    const unsigned int half_K = K / 2;
    const unsigned int num_groups = K / GROUP_SIZE;

    for (unsigned int k_base = 0; k_base < K; k_base += K_STEP) {
        {
            const unsigned int ept = (M_TILE * K_STEP) / 128;
            #pragma unroll
            for (unsigned int i = 0; i < ept; i++) {
                unsigned int idx = threadIdx.x * ept + i;
                unsigned int row = idx / K_STEP;
                unsigned int col = idx % K_STEP;
                unsigned int gc = k_base + col;
                bool valid = (cta_m_local + row) < M_eff && gc < K;
                if (valid) {
                    unsigned int a_row = sorted_token_ids
                        ? (unsigned int)sorted_token_ids[cta_m + row]
                        : (cta_m + row);
                    smem_A[row][col] = A[a_row * K + gc];
                } else {
                    smem_A[row][col] = __float2bfloat16(0.0f);
                }
            }
        }

        {
            const unsigned int ept = (K_STEP * N_TILE_SM) / 128;
            unsigned int scale_group = k_base / GROUP_SIZE;
            #pragma unroll
            for (unsigned int i = 0; i < ept; i++) {
                unsigned int idx = threadIdx.x * ept + i;
                unsigned int k = idx / N_TILE_SM;
                unsigned int n = idx % N_TILE_SM;
                unsigned int gk = k_base + k;
                unsigned int gn = cta_n + n;
                if (gk < K && gn < N) {
                    unsigned int k_pair = gk / 2;
                    unsigned char packed_byte = B_expert[(unsigned long long)gn * half_K + k_pair];
                    unsigned int nibble = (gk & 1) ? (packed_byte >> 4) : (packed_byte & 0xF);
                    unsigned char sb = S_expert[(unsigned long long)gn * num_groups + scale_group];
                    __nv_fp8_e4m3 fp8; *(unsigned char*)&fp8 = sb;
                    smem_B[k][n] = __float2bfloat16(E2M1_LUT_MOE[nibble] * (float)fp8 * scale2);
                } else {
                    smem_B[k][n] = __float2bfloat16(0.0f);
                }
            }
        }
        __syncthreads();

        const unsigned short* sA = (const unsigned short*)smem_A;
        const unsigned short* sB = (const unsigned short*)smem_B;
        unsigned int fr0 = warp_m_offset + group_id;
        unsigned int fr1 = fr0 + 8;
        unsigned int fc0 = tid * 2, fc1 = fc0 + 8;
        unsigned int a0 = ((unsigned int)sA[fr0*a_stride+fc0+1]<<16) | (unsigned int)sA[fr0*a_stride+fc0];
        unsigned int a1 = ((unsigned int)sA[fr1*a_stride+fc0+1]<<16) | (unsigned int)sA[fr1*a_stride+fc0];
        unsigned int a2 = ((unsigned int)sA[fr0*a_stride+fc1+1]<<16) | (unsigned int)sA[fr0*a_stride+fc1];
        unsigned int a3 = ((unsigned int)sA[fr1*a_stride+fc1+1]<<16) | (unsigned int)sA[fr1*a_stride+fc1];

        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            unsigned int nc = nt * 8 + group_id;
            unsigned int k0 = tid * 2, k1 = k0 + 8;
            unsigned int b0 = ((unsigned int)sB[(k0+1)*b_stride+nc]<<16) | (unsigned int)sB[k0*b_stride+nc];
            unsigned int b1 = ((unsigned int)sB[(k1+1)*b_stride+nc]<<16) | (unsigned int)sB[k1*b_stride+nc];
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                :"=f"(acc[nt][0]),"=f"(acc[nt][1]),"=f"(acc[nt][2]),"=f"(acc[nt][3])
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
                 "f"(acc[nt][0]),"f"(acc[nt][1]),"f"(acc[nt][2]),"f"(acc[nt][3]));
        }
        __syncthreads();
    }

    #pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        bool r0v = (int)(warp_m_offset + group_id + cta_m_local) < M_expert;
        bool r1v = (int)(warp_m_offset + group_id + 8 + cta_m_local) < M_expert;
        if (r0v && c0 < N) C[r0*N+c0] = __float2bfloat16(acc[nt][0]);
        if (r0v && c1 < N) C[r0*N+c1] = __float2bfloat16(acc[nt][1]);
        if (r1v && c0 < N) C[r1*N+c0] = __float2bfloat16(acc[nt][2]);
        if (r1v && c1 < N) C[r1*N+c1] = __float2bfloat16(acc[nt][3]);
    }
}

// ═══════════════════════════════════════════════════════════════════
// FP8-MMA transposed MoE GEMM.
//
// Dequant B to FP8 E4M3 (not BF16). Convert A from BF16→FP8 in
// registers. Use mma.sync.m16n8k32.e4m3.e4m3 — full K=32 per
// instruction (2x fewer MMA instructions vs BF16 m16n8k16).
//
// smem layout:
//   A:      2 × 64 × 40 × 2B = 10240B  (double-buffered)
//   Bp:     2 × 16 × 144     =  4608B  (double-buffered)
//   Bs:     2 × 2  × 144     =   576B  (double-buffered)
//   B_fp8:  128 × 32          =  4096B  (single, FP8 E4M3)
//   LUT: 64B  tok: 256B
//   Total: ~19.8KB
// ═══════════════════════════════════════════════════════════════════

__device__ __forceinline__ void moe_cp_async_pred_16(void* dst_smem, const void* src_gmem, bool pred) {
    unsigned int dst = __cvta_generic_to_shared(dst_smem);
    unsigned int src_bytes = pred ? 16 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
                 :: "r"(dst), "l"(src_gmem), "r"(src_bytes));
}

__device__ __forceinline__ void moe_cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}

__device__ __forceinline__ void moe_cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;");
}

// Convert 4 BF16 values from smem to packed uint32 of 4 E4M3 values
__device__ __forceinline__ unsigned int moe_bf16x4_to_e4m3x4(const unsigned short* src) {
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

extern "C" __global__ void moe_w4a16_grouped_gemm_ptrtable_t(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ B_packed_ptrs,
    const unsigned long long* __restrict__ B_scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    unsigned int num_experts,
    unsigned int N,
    unsigned int K
) {
    const unsigned int expert_id = blockIdx.z;
    if (expert_id >= num_experts) return;

    const int m_start = expert_offsets[expert_id];
    const int m_end = expert_offsets[expert_id + 1];
    const int M_expert = m_end - m_start;
    if (M_expert <= 0) return;

    const int cta_m_local = blockIdx.y * M_TILE;
    if (cta_m_local >= M_expert) return;

    const unsigned int cta_m = m_start + cta_m_local;
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;

    const unsigned char* B_expert = (const unsigned char*)B_packed_ptrs[expert_id];
    const unsigned char* S_expert = (const unsigned char*)B_scale_ptrs[expert_id];
    const float scale2 = scale2_vals[expert_id];

    if (B_expert == 0) return;

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[2][M_TILE][K_STEP_T + PAD_T];
    __shared__ unsigned char smem_Bp[2][K_STEP_T / 2][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_Bs[2][K_STEP_T / GROUP_SIZE][N_TILE_LG + BP_PAD];
    // +4 pad: row stride = 36 bytes (stride_words=9, gcd(9,32)=1).
    // Thread t's DEQUANT write: bank = (t*9 + kp/2) % 32 — all 32 distinct → 0 conflicts.
    // MMA reads: (nc*9 + tid) % 32 — 3 conflict pairs vs 16 without padding.
    __shared__ unsigned char smem_B_fp8[N_TILE_LG][K_STEP_T + 4];
    __shared__ float smem_LUT[16];
    __shared__ int smem_tok[M_TILE];

    if (threadIdx.x < 16) {
        smem_LUT[threadIdx.x] = E2M1_LUT_MOE[threadIdx.x];
    }
    if (threadIdx.x < M_TILE) {
        int local_row = threadIdx.x;
        if (sorted_token_ids && (cta_m_local + local_row) < (unsigned int)M_expert) {
            smem_tok[local_row] = sorted_token_ids[cta_m + local_row];
        } else {
            smem_tok[local_row] = (int)(cta_m + local_row);
        }
    }
    __syncthreads();

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int a_stride = K_STEP_T + PAD_T;
    const unsigned int M_eff = (unsigned int)M_expert;

    #define MOE_ISSUE_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 2; \
            unsigned int a_col = (threadIdx.x & 3) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 2; rnd++) { \
                unsigned int row = rnd * 32 + a_row_base; \
                bool valid = (cta_m_local + row) < M_eff && (gc + 7 < K); \
                unsigned int a_row = (unsigned int)smem_tok[row]; \
                moe_cp_async_pred_16(&smem_A[(buf)][row][a_col], \
                    &A[(unsigned long long)a_row * K + gc], valid); \
            } \
        } \
        { \
            unsigned int kp = threadIdx.x >> 3; \
            unsigned int ns = (threadIdx.x & 7) << 4; \
            unsigned int gke = (kb) + (kp << 1); \
            unsigned int gns = cta_n + ns; \
            moe_cp_async_pred_16(&smem_Bp[(buf)][kp][ns], \
                &B_expert[(unsigned long long)(gke >> 1) * N + gns], \
                (gke + 1 <= K) && (gns + 15 < N)); \
            if (kp < K_STEP_T / GROUP_SIZE) { \
                unsigned int sg = (kb) / GROUP_SIZE + kp; \
                moe_cp_async_pred_16(&smem_Bs[(buf)][kp][ns], \
                    &S_expert[(unsigned long long)sg * N + gns], \
                    (gns + 15 < N)); \
            } \
        } \
    } while(0)

    // Dequant B: FP4 → FP8 E4M3
    #define MOE_DEQUANT(buf) do { \
        unsigned int my_n = threadIdx.x; \
        unsigned char sb0 = smem_Bs[(buf)][0][my_n]; \
        unsigned char sb1 = smem_Bs[(buf)][1][my_n]; \
        __nv_fp8_e4m3 f0, f1; \
        *(unsigned char*)&f0 = sb0; \
        *(unsigned char*)&f1 = sb1; \
        float sv0 = (float)f0 * scale2; \
        float sv1 = (float)f1 * scale2; \
        _Pragma("unroll") \
        for (int kp = 0; kp < 8; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            float lo = smem_LUT[packed & 0xF] * sv0; \
            float hi = smem_LUT[packed >> 4] * sv0; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 8; kp < 16; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            float lo = smem_LUT[packed & 0xF] * sv1; \
            float hi = smem_LUT[packed >> 4] * sv1; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8[my_n][kp * 2] = fp8_pair; \
        } \
    } while(0)

    // FP8 MMA: convert A BF16→E4M3 in registers, single m16n8k32 per N-tile
    #define MOE_COMPUTE_MMA(a_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A[(a_buf)]; \
        unsigned int fr0 = warp_m_offset + group_id, fr1 = fr0 + 8; \
        unsigned int a0 = moe_bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4]); \
        unsigned int a1 = moe_bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid * 4]); \
        unsigned int a2 = moe_bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid * 4]); \
        unsigned int a3 = moe_bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8[nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8[nc][16 + 4 * tid]; \
            asm volatile( \
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                :"=f"(acc[nt][0]),"=f"(acc[nt][1]), \
                 "=f"(acc[nt][2]),"=f"(acc[nt][3]) \
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3), \
                 "r"(b0),"r"(b1), \
                 "f"(acc[nt][0]),"f"(acc[nt][1]), \
                 "f"(acc[nt][2]),"f"(acc[nt][3])); \
        } \
    } while(0)

    MOE_ISSUE_LOADS(0, 0);
    moe_cp_async_commit();
    moe_cp_async_wait_all();
    __syncthreads();
    MOE_DEQUANT(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;

        MOE_ISSUE_LOADS(nxt, k_base);
        moe_cp_async_commit();

        MOE_COMPUTE_MMA(cur);

        moe_cp_async_wait_all();
        __syncthreads();

        MOE_DEQUANT(nxt);
        __syncthreads();

        cur = nxt;
    }

    MOE_COMPUTE_MMA(cur);

    #undef MOE_ISSUE_LOADS
    #undef MOE_DEQUANT
    #undef MOE_COMPUTE_MMA

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        bool r0v = (int)(warp_m_offset + group_id + cta_m_local) < M_expert;
        bool r1v = (int)(warp_m_offset + group_id + 8 + cta_m_local) < M_expert;
        if (r0v && c0 < N) C[r0*N+c0] = __float2bfloat16(acc[nt][0]);
        if (r0v && c1 < N) C[r0*N+c1] = __float2bfloat16(acc[nt][1]);
        if (r1v && c0 < N) C[r1*N+c0] = __float2bfloat16(acc[nt][2]);
        if (r1v && c1 < N) C[r1*N+c1] = __float2bfloat16(acc[nt][3]);
    }
}

// ═══════════════════════════════════════════════════════════════════
// K64 variant of FP8-MMA transposed MoE GEMM.
//
// K_STEP_T=64: doubles the K tile size so compute time per step
// (2×16 MMAs × 4 cycles = 128 cycles) exceeds the cp.async load
// latency (~100 cycles), eliminating all pipeline stall.
//
// For the down GEMM (K=inter=512): 8 K-steps (vs 16 with K32).
// For gate/up (K=h=2048): 32 K-steps (vs 64 with K32).
//
// smem layout (K64):
//   A:      2 × 64 × 72 × 2B = 18432B  (double-buffered, BF16)
//   Bp:     2 × 32 × 144     =  9216B  (double-buffered, packed NVFP4)
//   Bs:     2 × 4  × 144     =  1152B  (double-buffered, FP8 scales)
//   B_fp8:  128 × 64          =  8192B  (single, FP8 E4M3)
//   LUT: 64B  tok: 256B
//   Total: ~37.4KB
// ═══════════════════════════════════════════════════════════════════

#define K_STEP_T64 64
#define PAD_T64 8  // (64+8)*2 = 144 bytes, 144%16 = 0 ✓

extern "C" __global__ void moe_w4a16_grouped_gemm_ptrtable_t_k64(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ B_packed_ptrs,
    const unsigned long long* __restrict__ B_scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    unsigned int num_experts,
    unsigned int N,
    unsigned int K
) {
    const unsigned int expert_id = blockIdx.z;
    if (expert_id >= num_experts) return;

    const int m_start = expert_offsets[expert_id];
    const int m_end = expert_offsets[expert_id + 1];
    const int M_expert = m_end - m_start;
    if (M_expert <= 0) return;

    const int cta_m_local = blockIdx.y * M_TILE;
    if (cta_m_local >= M_expert) return;

    const unsigned int cta_m = m_start + cta_m_local;
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;

    const unsigned char* B_expert = (const unsigned char*)B_packed_ptrs[expert_id];
    const unsigned char* S_expert = (const unsigned char*)B_scale_ptrs[expert_id];
    const float scale2 = scale2_vals[expert_id];

    if (B_expert == 0) return;

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    // B_fp8 padding: row stride 80 bytes (K64+16) gives zero bank conflicts.
    // Without pad: 64-byte rows (16 banks/row) → 4-way conflicts for nc spaced 2 apart.
    // With pad: 80-byte rows (20 banks/row) → nc*20 % 32 hits all distinct banks for nc=0..7.
    __shared__ __nv_bfloat16 smem_A_k64[2][M_TILE][K_STEP_T64 + PAD_T64];
    __shared__ unsigned char smem_Bp_k64[2][K_STEP_T64 / 2][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_Bs_k64[2][K_STEP_T64 / GROUP_SIZE][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_B_fp8_k64[N_TILE_LG][K_STEP_T64 + 16];
    __shared__ float smem_LUT_k64[16];
    __shared__ int smem_tok_k64[M_TILE];

    if (threadIdx.x < 16) smem_LUT_k64[threadIdx.x] = E2M1_LUT_MOE[threadIdx.x];
    if (threadIdx.x < M_TILE) {
        int local_row = threadIdx.x;
        if (sorted_token_ids && (cta_m_local + local_row) < (unsigned int)M_expert)
            smem_tok_k64[local_row] = sorted_token_ids[cta_m + local_row];
        else
            smem_tok_k64[local_row] = (int)(cta_m + local_row);
    }
    __syncthreads();

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int ast64 = K_STEP_T64 + PAD_T64;
    const unsigned int M_eff = (unsigned int)M_expert;

    // A: 4 rounds × 128 threads × 16 bytes = 8192B = 64×64 BF16
    // Bp: 2 rounds × 128 threads × 16 bytes = 4096B = 32×128 packed bytes
    // Bs: 4 scale groups × 8 ns per group × 16 bytes = 512B per buffer
    #define K64_ISSUE_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 3; \
            unsigned int a_col = (threadIdx.x & 7) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 4; rnd++) { \
                unsigned int row = rnd * 16 + a_row_base; \
                bool valid = (cta_m_local + row) < M_eff && (gc + 7 < K); \
                unsigned int a_row = (unsigned int)smem_tok_k64[row]; \
                moe_cp_async_pred_16(&smem_A_k64[(buf)][row][a_col], \
                    &A[(unsigned long long)a_row * K + gc], valid); \
            } \
        } \
        { \
            unsigned int kp = threadIdx.x >> 3; \
            unsigned int ns = (threadIdx.x & 7) << 4; \
            unsigned int gns = cta_n + ns; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 2; rnd++) { \
                unsigned int kp_cur = rnd * 16 + kp; \
                unsigned int gke = (kb) + (kp_cur << 1); \
                moe_cp_async_pred_16(&smem_Bp_k64[(buf)][kp_cur][ns], \
                    &B_expert[(unsigned long long)(gke >> 1) * N + gns], \
                    (gke + 1 < K) && (gns + 15 < N)); \
                if (kp_cur < K_STEP_T64 / GROUP_SIZE) { \
                    unsigned int sg = (kb) / GROUP_SIZE + kp_cur; \
                    moe_cp_async_pred_16(&smem_Bs_k64[(buf)][kp_cur][ns], \
                        &S_expert[(unsigned long long)sg * N + gns], \
                        (gns + 15 < N)); \
                } \
            } \
        } \
    } while(0)

    // Dequant B: FP4 → FP8 E4M3, 4 scale groups for K64
    #define K64_DEQUANT(buf) do { \
        unsigned int my_n = threadIdx.x; \
        __nv_fp8_e4m3 f0, f1, f2, f3; \
        *(unsigned char*)&f0 = smem_Bs_k64[(buf)][0][my_n]; \
        *(unsigned char*)&f1 = smem_Bs_k64[(buf)][1][my_n]; \
        *(unsigned char*)&f2 = smem_Bs_k64[(buf)][2][my_n]; \
        *(unsigned char*)&f3 = smem_Bs_k64[(buf)][3][my_n]; \
        float sv0 = (float)f0 * scale2; \
        float sv1 = (float)f1 * scale2; \
        float sv2 = (float)f2 * scale2; \
        float sv3 = (float)f3 * scale2; \
        _Pragma("unroll") \
        for (int kp = 0; kp < 8; kp++) { \
            unsigned char packed = smem_Bp_k64[(buf)][kp][my_n]; \
            float lo = smem_LUT_k64[packed & 0xF] * sv0; \
            float hi = smem_LUT_k64[packed >> 4] * sv0; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8_k64[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 8; kp < 16; kp++) { \
            unsigned char packed = smem_Bp_k64[(buf)][kp][my_n]; \
            float lo = smem_LUT_k64[packed & 0xF] * sv1; \
            float hi = smem_LUT_k64[packed >> 4] * sv1; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8_k64[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 16; kp < 24; kp++) { \
            unsigned char packed = smem_Bp_k64[(buf)][kp][my_n]; \
            float lo = smem_LUT_k64[packed & 0xF] * sv2; \
            float hi = smem_LUT_k64[packed >> 4] * sv2; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8_k64[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 24; kp < 32; kp++) { \
            unsigned char packed = smem_Bp_k64[(buf)][kp][my_n]; \
            float lo = smem_LUT_k64[packed & 0xF] * sv3; \
            float hi = smem_LUT_k64[packed >> 4] * sv3; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8_k64[my_n][kp * 2] = fp8_pair; \
        } \
    } while(0)

    // Two m16n8k32 MMA calls per N-tile: first covers k=0..31, second k=32..63.
    // a0..a3 loaded first, all N-tile first-half MMAs done, then a4..a7, then second-half.
    // This keeps max 4 A registers live at once (same as K32 variant).
    #define K64_COMPUTE_MMA(a_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A_k64[(a_buf)]; \
        unsigned int fr0 = warp_m_offset + group_id, fr1 = fr0 + 8; \
        unsigned int a0 = moe_bf16x4_to_e4m3x4(&sA[fr0 * ast64 + tid * 4]); \
        unsigned int a1 = moe_bf16x4_to_e4m3x4(&sA[fr1 * ast64 + tid * 4]); \
        unsigned int a2 = moe_bf16x4_to_e4m3x4(&sA[fr0 * ast64 + 16 + tid * 4]); \
        unsigned int a3 = moe_bf16x4_to_e4m3x4(&sA[fr1 * ast64 + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8_k64[nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8_k64[nc][16 + 4 * tid]; \
            asm volatile( \
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                :"=f"(acc[nt][0]),"=f"(acc[nt][1]),"=f"(acc[nt][2]),"=f"(acc[nt][3]) \
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
                 "f"(acc[nt][0]),"f"(acc[nt][1]),"f"(acc[nt][2]),"f"(acc[nt][3])); \
        } \
        unsigned int a4 = moe_bf16x4_to_e4m3x4(&sA[fr0 * ast64 + 32 + tid * 4]); \
        unsigned int a5 = moe_bf16x4_to_e4m3x4(&sA[fr1 * ast64 + 32 + tid * 4]); \
        unsigned int a6 = moe_bf16x4_to_e4m3x4(&sA[fr0 * ast64 + 48 + tid * 4]); \
        unsigned int a7 = moe_bf16x4_to_e4m3x4(&sA[fr1 * ast64 + 48 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8_k64[nc][32 + 4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8_k64[nc][48 + 4 * tid]; \
            asm volatile( \
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                :"=f"(acc[nt][0]),"=f"(acc[nt][1]),"=f"(acc[nt][2]),"=f"(acc[nt][3]) \
                :"r"(a4),"r"(a5),"r"(a6),"r"(a7),"r"(b0),"r"(b1), \
                 "f"(acc[nt][0]),"f"(acc[nt][1]),"f"(acc[nt][2]),"f"(acc[nt][3])); \
        } \
    } while(0)

    K64_ISSUE_LOADS(0, 0);
    moe_cp_async_commit();
    moe_cp_async_wait_all();
    __syncthreads();
    K64_DEQUANT(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T64; k_base < K; k_base += K_STEP_T64) {
        int nxt = 1 - cur;
        K64_ISSUE_LOADS(nxt, k_base);
        moe_cp_async_commit();
        K64_COMPUTE_MMA(cur);
        moe_cp_async_wait_all();
        __syncthreads();
        K64_DEQUANT(nxt);
        __syncthreads();
        cur = nxt;
    }
    K64_COMPUTE_MMA(cur);

    #undef K64_ISSUE_LOADS
    #undef K64_DEQUANT
    #undef K64_COMPUTE_MMA

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        bool r0v = (int)(warp_m_offset + group_id + cta_m_local) < M_expert;
        bool r1v = (int)(warp_m_offset + group_id + 8 + cta_m_local) < M_expert;
        if (r0v && c0 < N) C[r0*N+c0] = __float2bfloat16(acc[nt][0]);
        if (r0v && c1 < N) C[r0*N+c1] = __float2bfloat16(acc[nt][1]);
        if (r1v && c0 < N) C[r1*N+c0] = __float2bfloat16(acc[nt][2]);
        if (r1v && c1 < N) C[r1*N+c1] = __float2bfloat16(acc[nt][3]);
    }
}

// ═══════════════════════════════════════════════════════════════════
// K64 fused gate+up MoE GEMM — same K64 pipeline as down GEMM above.
//
// Replaces moe_w4a16_fused_gate_up_t for both gate+up projections
// when K=h=2048 (64 → 32 K-steps, zero pipeline stall).
// ═══════════════════════════════════════════════════════════════════
extern "C" __global__ void moe_w4a16_fused_gate_up_t_k64(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ gate_packed_ptrs,
    const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals,
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals,
    __nv_bfloat16* __restrict__ C_gate,
    __nv_bfloat16* __restrict__ C_up,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    unsigned int num_experts,
    unsigned int N,
    unsigned int K
) {
    const unsigned int expert_id = blockIdx.z;
    if (expert_id >= num_experts) return;

    const int m_start = expert_offsets[expert_id];
    const int m_end = expert_offsets[expert_id + 1];
    const int M_expert = m_end - m_start;
    if (M_expert <= 0) return;

    const int cta_m_local = blockIdx.y * M_TILE;
    if (cta_m_local >= M_expert) return;

    const unsigned int global_n = blockIdx.x * N_TILE_LG;
    const bool is_up = (global_n >= N);
    const unsigned int cta_n = is_up ? (global_n - N) : global_n;

    const unsigned char* B_expert;
    const unsigned char* S_expert;
    float scale2;
    __nv_bfloat16* C;
    if (is_up) {
        B_expert = (const unsigned char*)up_packed_ptrs[expert_id];
        S_expert = (const unsigned char*)up_scale_ptrs[expert_id];
        scale2 = up_scale2_vals[expert_id];
        C = C_up;
    } else {
        B_expert = (const unsigned char*)gate_packed_ptrs[expert_id];
        S_expert = (const unsigned char*)gate_scale_ptrs[expert_id];
        scale2 = gate_scale2_vals[expert_id];
        C = C_gate;
    }

    if (B_expert == 0) return;

    const unsigned int cta_m = m_start + cta_m_local;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A_fgu64[2][M_TILE][K_STEP_T64 + PAD_T64];
    __shared__ unsigned char smem_Bp_fgu64[2][K_STEP_T64 / 2][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_Bs_fgu64[2][K_STEP_T64 / GROUP_SIZE][N_TILE_LG + BP_PAD];
    // B_fp8 row stride 80 bytes → zero smem bank conflicts (see K64_DEQUANT comment above).
    __shared__ unsigned char smem_B_fp8_fgu64[N_TILE_LG][K_STEP_T64 + 16];
    __shared__ float smem_LUT_fgu64[16];
    __shared__ int smem_tok_fgu64[M_TILE];

    if (threadIdx.x < 16) smem_LUT_fgu64[threadIdx.x] = E2M1_LUT_MOE[threadIdx.x];
    if (threadIdx.x < M_TILE) {
        int local_row = threadIdx.x;
        if (sorted_token_ids && (cta_m_local + local_row) < (unsigned int)M_expert)
            smem_tok_fgu64[local_row] = sorted_token_ids[cta_m + local_row];
        else
            smem_tok_fgu64[local_row] = (int)(cta_m + local_row);
    }
    __syncthreads();

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int ast_fgu64 = K_STEP_T64 + PAD_T64;
    const unsigned int M_eff = (unsigned int)M_expert;

    #define FGU64_ISSUE_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 3; \
            unsigned int a_col = (threadIdx.x & 7) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 4; rnd++) { \
                unsigned int row = rnd * 16 + a_row_base; \
                bool valid = (cta_m_local + row) < M_eff && (gc + 7 < K); \
                unsigned int a_row = (unsigned int)smem_tok_fgu64[row]; \
                moe_cp_async_pred_16(&smem_A_fgu64[(buf)][row][a_col], \
                    &A[(unsigned long long)a_row * K + gc], valid); \
            } \
        } \
        { \
            unsigned int kp = threadIdx.x >> 3; \
            unsigned int ns = (threadIdx.x & 7) << 4; \
            unsigned int gns = cta_n + ns; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 2; rnd++) { \
                unsigned int kp_cur = rnd * 16 + kp; \
                unsigned int gke = (kb) + (kp_cur << 1); \
                moe_cp_async_pred_16(&smem_Bp_fgu64[(buf)][kp_cur][ns], \
                    &B_expert[(unsigned long long)(gke >> 1) * N + gns], \
                    (gke + 1 < K) && (gns + 15 < N)); \
                if (kp_cur < K_STEP_T64 / GROUP_SIZE) { \
                    unsigned int sg = (kb) / GROUP_SIZE + kp_cur; \
                    moe_cp_async_pred_16(&smem_Bs_fgu64[(buf)][kp_cur][ns], \
                        &S_expert[(unsigned long long)sg * N + gns], \
                        (gns + 15 < N)); \
                } \
            } \
        } \
    } while(0)

    #define FGU64_DEQUANT(buf) do { \
        unsigned int my_n = threadIdx.x; \
        __nv_fp8_e4m3 f0, f1, f2, f3; \
        *(unsigned char*)&f0 = smem_Bs_fgu64[(buf)][0][my_n]; \
        *(unsigned char*)&f1 = smem_Bs_fgu64[(buf)][1][my_n]; \
        *(unsigned char*)&f2 = smem_Bs_fgu64[(buf)][2][my_n]; \
        *(unsigned char*)&f3 = smem_Bs_fgu64[(buf)][3][my_n]; \
        float sv0 = (float)f0 * scale2; \
        float sv1 = (float)f1 * scale2; \
        float sv2 = (float)f2 * scale2; \
        float sv3 = (float)f3 * scale2; \
        _Pragma("unroll") \
        for (int kp = 0; kp < 8; kp++) { \
            unsigned char packed = smem_Bp_fgu64[(buf)][kp][my_n]; \
            float lo = smem_LUT_fgu64[packed & 0xF] * sv0; \
            float hi = smem_LUT_fgu64[packed >> 4] * sv0; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8_fgu64[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 8; kp < 16; kp++) { \
            unsigned char packed = smem_Bp_fgu64[(buf)][kp][my_n]; \
            float lo = smem_LUT_fgu64[packed & 0xF] * sv1; \
            float hi = smem_LUT_fgu64[packed >> 4] * sv1; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8_fgu64[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 16; kp < 24; kp++) { \
            unsigned char packed = smem_Bp_fgu64[(buf)][kp][my_n]; \
            float lo = smem_LUT_fgu64[packed & 0xF] * sv2; \
            float hi = smem_LUT_fgu64[packed >> 4] * sv2; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8_fgu64[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 24; kp < 32; kp++) { \
            unsigned char packed = smem_Bp_fgu64[(buf)][kp][my_n]; \
            float lo = smem_LUT_fgu64[packed & 0xF] * sv3; \
            float hi = smem_LUT_fgu64[packed >> 4] * sv3; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8_fgu64[my_n][kp * 2] = fp8_pair; \
        } \
    } while(0)

    #define FGU64_COMPUTE_MMA(a_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A_fgu64[(a_buf)]; \
        unsigned int fr0 = warp_m_offset + group_id, fr1 = fr0 + 8; \
        unsigned int a0 = moe_bf16x4_to_e4m3x4(&sA[fr0 * ast_fgu64 + tid * 4]); \
        unsigned int a1 = moe_bf16x4_to_e4m3x4(&sA[fr1 * ast_fgu64 + tid * 4]); \
        unsigned int a2 = moe_bf16x4_to_e4m3x4(&sA[fr0 * ast_fgu64 + 16 + tid * 4]); \
        unsigned int a3 = moe_bf16x4_to_e4m3x4(&sA[fr1 * ast_fgu64 + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8_fgu64[nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8_fgu64[nc][16 + 4 * tid]; \
            asm volatile( \
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                :"=f"(acc[nt][0]),"=f"(acc[nt][1]),"=f"(acc[nt][2]),"=f"(acc[nt][3]) \
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
                 "f"(acc[nt][0]),"f"(acc[nt][1]),"f"(acc[nt][2]),"f"(acc[nt][3])); \
        } \
        unsigned int a4 = moe_bf16x4_to_e4m3x4(&sA[fr0 * ast_fgu64 + 32 + tid * 4]); \
        unsigned int a5 = moe_bf16x4_to_e4m3x4(&sA[fr1 * ast_fgu64 + 32 + tid * 4]); \
        unsigned int a6 = moe_bf16x4_to_e4m3x4(&sA[fr0 * ast_fgu64 + 48 + tid * 4]); \
        unsigned int a7 = moe_bf16x4_to_e4m3x4(&sA[fr1 * ast_fgu64 + 48 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8_fgu64[nc][32 + 4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8_fgu64[nc][48 + 4 * tid]; \
            asm volatile( \
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                :"=f"(acc[nt][0]),"=f"(acc[nt][1]),"=f"(acc[nt][2]),"=f"(acc[nt][3]) \
                :"r"(a4),"r"(a5),"r"(a6),"r"(a7),"r"(b0),"r"(b1), \
                 "f"(acc[nt][0]),"f"(acc[nt][1]),"f"(acc[nt][2]),"f"(acc[nt][3])); \
        } \
    } while(0)

    FGU64_ISSUE_LOADS(0, 0);
    moe_cp_async_commit();
    moe_cp_async_wait_all();
    __syncthreads();
    FGU64_DEQUANT(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T64; k_base < K; k_base += K_STEP_T64) {
        int nxt = 1 - cur;
        FGU64_ISSUE_LOADS(nxt, k_base);
        moe_cp_async_commit();
        FGU64_COMPUTE_MMA(cur);
        moe_cp_async_wait_all();
        __syncthreads();
        FGU64_DEQUANT(nxt);
        __syncthreads();
        cur = nxt;
    }
    FGU64_COMPUTE_MMA(cur);

    #undef FGU64_ISSUE_LOADS
    #undef FGU64_DEQUANT
    #undef FGU64_COMPUTE_MMA

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        bool r0v = (int)(warp_m_offset + group_id + cta_m_local) < M_expert;
        bool r1v = (int)(warp_m_offset + group_id + 8 + cta_m_local) < M_expert;
        if (r0v && c0 < N) C[r0*N+c0] = __float2bfloat16(acc[nt][0]);
        if (r0v && c1 < N) C[r0*N+c1] = __float2bfloat16(acc[nt][1]);
        if (r1v && c0 < N) C[r1*N+c0] = __float2bfloat16(acc[nt][2]);
        if (r1v && c1 < N) C[r1*N+c1] = __float2bfloat16(acc[nt][3]);
    }
}

// ═══════════════════════════════════════════════════════════════════
// Fused gate+up MoE GEMM.
//
// Single launch replaces two separate gate/up GEMM launches.
// Grid x tiles across 2*N: first N columns use gate weights, last N
// columns use up weights. A is loaded once per (expert, m_tile) and
// shared across both projections via L2 cache reuse.
//
// Grid: (ceil(2*N / N_TILE_LG), max_m_tiles, num_experts)
// ═══════════════════════════════════════════════════════════════════
extern "C" __global__ void moe_w4a16_fused_gate_up_t(
    const __nv_bfloat16* __restrict__ A,
    // Gate weights
    const unsigned long long* __restrict__ gate_packed_ptrs,
    const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2_vals,
    // Up weights
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2_vals,
    // Two output buffers
    __nv_bfloat16* __restrict__ C_gate,
    __nv_bfloat16* __restrict__ C_up,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    unsigned int num_experts,
    unsigned int N,            // per-projection N (gate and up have same N)
    unsigned int K
) {
    const unsigned int expert_id = blockIdx.z;
    if (expert_id >= num_experts) return;

    const int m_start = expert_offsets[expert_id];
    const int m_end = expert_offsets[expert_id + 1];
    const int M_expert = m_end - m_start;
    if (M_expert <= 0) return;

    const int cta_m_local = blockIdx.y * M_TILE;
    if (cta_m_local >= M_expert) return;

    // Determine if this CTA handles gate (first N cols) or up (second N cols)
    const unsigned int global_n = blockIdx.x * N_TILE_LG;
    const bool is_up = (global_n >= N);
    const unsigned int cta_n = is_up ? (global_n - N) : global_n;

    // Select the right weight pointers and output buffer
    const unsigned char* B_expert;
    const unsigned char* S_expert;
    float scale2;
    __nv_bfloat16* C;
    if (is_up) {
        B_expert = (const unsigned char*)up_packed_ptrs[expert_id];
        S_expert = (const unsigned char*)up_scale_ptrs[expert_id];
        scale2 = up_scale2_vals[expert_id];
        C = C_up;
    } else {
        B_expert = (const unsigned char*)gate_packed_ptrs[expert_id];
        S_expert = (const unsigned char*)gate_scale_ptrs[expert_id];
        scale2 = gate_scale2_vals[expert_id];
        C = C_gate;
    }

    if (B_expert == 0) return;

    const unsigned int cta_m = m_start + cta_m_local;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[2][M_TILE][K_STEP_T + PAD_T];
    __shared__ unsigned char smem_Bp[2][K_STEP_T / 2][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_Bs[2][K_STEP_T / GROUP_SIZE][N_TILE_LG + BP_PAD];
    // Same +4 padding as grouped_gemm_ptrtable_t: eliminates 8-way DEQUANT write conflicts.
    __shared__ unsigned char smem_B_fp8[N_TILE_LG][K_STEP_T + 4];
    __shared__ float smem_LUT[16];
    __shared__ int smem_tok[M_TILE];

    if (threadIdx.x < 16) {
        smem_LUT[threadIdx.x] = E2M1_LUT_MOE[threadIdx.x];
    }
    if (threadIdx.x < M_TILE) {
        int local_row = threadIdx.x;
        if (sorted_token_ids && (cta_m_local + local_row) < (unsigned int)M_expert) {
            smem_tok[local_row] = sorted_token_ids[cta_m + local_row];
        } else {
            smem_tok[local_row] = (int)(cta_m + local_row);
        }
    }
    __syncthreads();

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int a_stride = K_STEP_T + PAD_T;
    const unsigned int M_eff = (unsigned int)M_expert;

    // Reuse exact same macros as moe_w4a16_grouped_gemm_ptrtable_t
    #define FGU_ISSUE_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 2; \
            unsigned int a_col = (threadIdx.x & 3) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 2; rnd++) { \
                unsigned int row = rnd * 32 + a_row_base; \
                bool valid = (cta_m_local + row) < M_eff && (gc + 7 < K); \
                unsigned int a_row = (unsigned int)smem_tok[row]; \
                moe_cp_async_pred_16(&smem_A[(buf)][row][a_col], \
                    &A[(unsigned long long)a_row * K + gc], valid); \
            } \
        } \
        { \
            unsigned int kp = threadIdx.x >> 3; \
            unsigned int ns = (threadIdx.x & 7) << 4; \
            unsigned int gke = (kb) + (kp << 1); \
            unsigned int gns = cta_n + ns; \
            moe_cp_async_pred_16(&smem_Bp[(buf)][kp][ns], \
                &B_expert[(unsigned long long)(gke >> 1) * N + gns], \
                (gke + 1 <= K) && (gns + 15 < N)); \
            if (kp < K_STEP_T / GROUP_SIZE) { \
                unsigned int sg = (kb) / GROUP_SIZE + kp; \
                moe_cp_async_pred_16(&smem_Bs[(buf)][kp][ns], \
                    &S_expert[(unsigned long long)sg * N + gns], \
                    (gns + 15 < N)); \
            } \
        } \
    } while(0)

    #define FGU_DEQUANT(buf) do { \
        unsigned int my_n = threadIdx.x; \
        unsigned char sb0 = smem_Bs[(buf)][0][my_n]; \
        unsigned char sb1 = smem_Bs[(buf)][1][my_n]; \
        __nv_fp8_e4m3 f0, f1; \
        *(unsigned char*)&f0 = sb0; \
        *(unsigned char*)&f1 = sb1; \
        float sv0 = (float)f0 * scale2; \
        float sv1 = (float)f1 * scale2; \
        _Pragma("unroll") \
        for (int kp = 0; kp < 8; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            float lo = smem_LUT[packed & 0xF] * sv0; \
            float hi = smem_LUT[packed >> 4] * sv0; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 8; kp < 16; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            float lo = smem_LUT[packed & 0xF] * sv1; \
            float hi = smem_LUT[packed >> 4] * sv1; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B_fp8[my_n][kp * 2] = fp8_pair; \
        } \
    } while(0)

    #define FGU_COMPUTE_MMA(a_buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A[(a_buf)]; \
        unsigned int fr0 = warp_m_offset + group_id, fr1 = fr0 + 8; \
        unsigned int a0 = moe_bf16x4_to_e4m3x4(&sA[fr0 * a_stride + tid * 4]); \
        unsigned int a1 = moe_bf16x4_to_e4m3x4(&sA[fr1 * a_stride + tid * 4]); \
        unsigned int a2 = moe_bf16x4_to_e4m3x4(&sA[fr0 * a_stride + 16 + tid * 4]); \
        unsigned int a3 = moe_bf16x4_to_e4m3x4(&sA[fr1 * a_stride + 16 + tid * 4]); \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B_fp8[nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B_fp8[nc][16 + 4 * tid]; \
            asm volatile( \
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                :"=f"(acc[nt][0]),"=f"(acc[nt][1]), \
                 "=f"(acc[nt][2]),"=f"(acc[nt][3]) \
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3), \
                 "r"(b0),"r"(b1), \
                 "f"(acc[nt][0]),"f"(acc[nt][1]), \
                 "f"(acc[nt][2]),"f"(acc[nt][3])); \
        } \
    } while(0)

    FGU_ISSUE_LOADS(0, 0);
    moe_cp_async_commit();
    moe_cp_async_wait_all();
    __syncthreads();
    FGU_DEQUANT(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        FGU_ISSUE_LOADS(nxt, k_base);
        moe_cp_async_commit();
        FGU_COMPUTE_MMA(cur);
        moe_cp_async_wait_all();
        __syncthreads();
        FGU_DEQUANT(nxt);
        __syncthreads();
        cur = nxt;
    }
    FGU_COMPUTE_MMA(cur);

    #undef FGU_ISSUE_LOADS
    #undef FGU_DEQUANT
    #undef FGU_COMPUTE_MMA

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        bool r0v = (int)(warp_m_offset + group_id + cta_m_local) < M_expert;
        bool r1v = (int)(warp_m_offset + group_id + 8 + cta_m_local) < M_expert;
        if (r0v && c0 < N) C[r0*N+c0] = __float2bfloat16(acc[nt][0]);
        if (r0v && c1 < N) C[r0*N+c1] = __float2bfloat16(acc[nt][1]);
        if (r1v && c0 < N) C[r1*N+c0] = __float2bfloat16(acc[nt][2]);
        if (r1v && c1 < N) C[r1*N+c1] = __float2bfloat16(acc[nt][3]);
    }
}

// ═══════════════════════════════════════════════════════════════════
// FP8-input MoE GEMM: A [M, K] FP8 × B_expert NVFP4 → C [M, N] BF16
//
// A is pre-converted to FP8 E4M3 — no BF16→FP8 in the inner loop.
// B is still NVFP4 (dequanted to FP8 in smem per tile).
// Same tiling as _t variant: M_TILE=64, N_TILE=128, K_STEP=32.
// A smem is FP8 (2 KB/buf vs 5 KB/buf BF16) — saves smem bandwidth.
// Grid: (ceil(N/128), max_m_tiles, num_experts)
// ═══════════════════════════════════════════════════════════════════
extern "C" __global__ void moe_fp8_grouped_gemm_ptrtable_t(
    const unsigned char* __restrict__ A_fp8,  // [total_tokens, K] FP8 E4M3
    const unsigned long long* __restrict__ B_packed_ptrs,
    const unsigned long long* __restrict__ B_scale_ptrs,
    const float* __restrict__ scale2_vals,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    unsigned int num_experts,
    unsigned int N,
    unsigned int K
) {
    const unsigned int expert_id = blockIdx.z;
    if (expert_id >= num_experts) return;

    const int m_start = expert_offsets[expert_id];
    const int m_end = expert_offsets[expert_id + 1];
    const int M_expert = m_end - m_start;
    if (M_expert <= 0) return;

    const int cta_m_local = blockIdx.y * M_TILE;
    if (cta_m_local >= M_expert) return;

    const unsigned int cta_m = m_start + cta_m_local;
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;

    const unsigned char* B_exp = (const unsigned char*)B_packed_ptrs[expert_id];
    const unsigned char* S_exp = (const unsigned char*)B_scale_ptrs[expert_id];
    const float scale2 = scale2_vals[expert_id];

    if (B_exp == 0) return;

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ unsigned char smem_Af2[2][M_TILE][K_STEP_T];
    __shared__ unsigned char smem_Bp2[2][K_STEP_T / 2][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_Bs2[2][K_STEP_T / GROUP_SIZE][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_B2_fp8[N_TILE_LG][K_STEP_T];
    __shared__ float smem_LUT2[16];
    __shared__ int smem_tok2[M_TILE];

    if (threadIdx.x < 16) {
        smem_LUT2[threadIdx.x] = E2M1_LUT_MOE[threadIdx.x];
    }
    if (threadIdx.x < M_TILE) {
        int local_row = threadIdx.x;
        if (sorted_token_ids && (cta_m_local + local_row) < (unsigned int)M_expert) {
            smem_tok2[local_row] = sorted_token_ids[cta_m + local_row];
        } else {
            smem_tok2[local_row] = (int)(cta_m + local_row);
        }
    }
    __syncthreads();

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int M_eff = (unsigned int)M_expert;

    #define MOE_FF_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 1; \
            unsigned int a_col = (threadIdx.x & 1) << 4; \
            unsigned int gc = (kb) + a_col; \
            unsigned int row = a_row_base; \
            bool valid = (cta_m_local + row) < M_eff && (gc + 15 < K); \
            unsigned int a_row = (unsigned int)smem_tok2[row]; \
            moe_cp_async_pred_16(&smem_Af2[(buf)][row][a_col], \
                &A_fp8[(unsigned long long)a_row * K + gc], valid); \
        } \
        { \
            unsigned int kp = threadIdx.x >> 3; \
            unsigned int ns = (threadIdx.x & 7) << 4; \
            unsigned int gke = (kb) + (kp << 1); \
            unsigned int gns = cta_n + ns; \
            moe_cp_async_pred_16(&smem_Bp2[(buf)][kp][ns], \
                &B_exp[(unsigned long long)(gke >> 1) * N + gns], \
                (gke + 1 <= K) && (gns + 15 < N)); \
            if (kp < K_STEP_T / GROUP_SIZE) { \
                unsigned int sg = (kb) / GROUP_SIZE + kp; \
                moe_cp_async_pred_16(&smem_Bs2[(buf)][kp][ns], \
                    &S_exp[(unsigned long long)sg * N + gns], \
                    (gns + 15 < N)); \
            } \
        } \
    } while(0)

    #define MOE_FF_DEQUANT(buf) do { \
        unsigned int my_n = threadIdx.x; \
        unsigned char sb0 = smem_Bs2[(buf)][0][my_n]; \
        unsigned char sb1 = smem_Bs2[(buf)][1][my_n]; \
        __nv_fp8_e4m3 f0, f1; \
        *(unsigned char*)&f0 = sb0; \
        *(unsigned char*)&f1 = sb1; \
        float sv0 = (float)f0 * scale2; \
        float sv1 = (float)f1 * scale2; \
        _Pragma("unroll") \
        for (int kp = 0; kp < 8; kp++) { \
            unsigned char packed = smem_Bp2[(buf)][kp][my_n]; \
            float lo = smem_LUT2[packed & 0xF] * sv0; \
            float hi = smem_LUT2[packed >> 4] * sv0; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B2_fp8[my_n][kp * 2] = fp8_pair; \
        } \
        _Pragma("unroll") \
        for (int kp = 8; kp < 16; kp++) { \
            unsigned char packed = smem_Bp2[(buf)][kp][my_n]; \
            float lo = smem_LUT2[packed & 0xF] * sv1; \
            float hi = smem_LUT2[packed >> 4] * sv1; \
            unsigned short fp8_pair; \
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" \
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo)); \
            *(unsigned short*)&smem_B2_fp8[my_n][kp * 2] = fp8_pair; \
        } \
    } while(0)

    #define MOE_FF_COMPUTE(a_buf) do { \
        unsigned int fr0 = warp_m_offset + group_id, fr1 = fr0 + 8; \
        unsigned int a0 = *(const unsigned int*)&smem_Af2[(a_buf)][fr0][4 * tid]; \
        unsigned int a1 = *(const unsigned int*)&smem_Af2[(a_buf)][fr1][4 * tid]; \
        unsigned int a2 = *(const unsigned int*)&smem_Af2[(a_buf)][fr0][16 + 4 * tid]; \
        unsigned int a3 = *(const unsigned int*)&smem_Af2[(a_buf)][fr1][16 + 4 * tid]; \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_B2_fp8[nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_B2_fp8[nc][16 + 4 * tid]; \
            asm volatile( \
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                :"=f"(acc[nt][0]),"=f"(acc[nt][1]), \
                 "=f"(acc[nt][2]),"=f"(acc[nt][3]) \
                :"r"(a0),"r"(a1),"r"(a2),"r"(a3), \
                 "r"(b0),"r"(b1), \
                 "f"(acc[nt][0]),"f"(acc[nt][1]), \
                 "f"(acc[nt][2]),"f"(acc[nt][3])); \
        } \
    } while(0)

    MOE_FF_LOADS(0, 0);
    moe_cp_async_commit();
    moe_cp_async_wait_all();
    __syncthreads();
    MOE_FF_DEQUANT(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        MOE_FF_LOADS(nxt, k_base);
        moe_cp_async_commit();
        MOE_FF_COMPUTE(cur);
        moe_cp_async_wait_all();
        __syncthreads();
        MOE_FF_DEQUANT(nxt);
        __syncthreads();
        cur = nxt;
    }
    MOE_FF_COMPUTE(cur);

    #undef MOE_FF_LOADS
    #undef MOE_FF_DEQUANT
    #undef MOE_FF_COMPUTE

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt*8 + tid*2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        bool r0v = (int)(warp_m_offset + group_id + cta_m_local) < M_expert;
        bool r1v = (int)(warp_m_offset + group_id + 8 + cta_m_local) < M_expert;
        if (r0v && c0 < N) C[r0*N+c0] = __float2bfloat16(acc[nt][0]);
        if (r0v && c1 < N) C[r0*N+c1] = __float2bfloat16(acc[nt][1]);
        if (r1v && c0 < N) C[r1*N+c0] = __float2bfloat16(acc[nt][2]);
        if (r1v && c1 < N) C[r1*N+c1] = __float2bfloat16(acc[nt][3]);
    }
}
