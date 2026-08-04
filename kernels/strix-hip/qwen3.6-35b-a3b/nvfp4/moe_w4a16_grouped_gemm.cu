// SPDX-License-Identifier: AGPL-3.0-only

// Atlas Grouped W4A16 GEMM for MoE — 35B/27B model shadow. HIP/gfx1151 (AMD WMMA) port.
//
// Ported from the NVIDIA/SCALE mma.sync version. Transforms (same as w4a16_gemm.cu port):
//   * mma.sync.m16n8k16.bf16  → __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32 (n8→n16).
//       N_TILE=64 → 4 WMMA n-sub-tiles; N_TILE=128 → 8 WMMA n-sub-tiles.
//   * FP8 m16n8k32 paths → decode E4M3→BF16 in registers + 2 (K32) or 4 (K64) WMMA K=16.
//   * cp.async → synchronous 16-byte uint4 smem copy; commit/wait dropped.
//   * Scale decode uses standard E4M3 (scl_fp8), matching the quantizer and the
//     validated reference — NOT SCALE's non-standard __nv_fp8_e4m3 cast.
//
// WMMA fragment layout (validated bit-exact in w4a16_wmma_ref.hip):
//   A load (M×K row-major smem): lane l → a[i] = smem_A[m_row + (l&15)][i], i=0..15
//   B load (N×K row-major smem): lane l → b[k] = smem_B[n_base + (l&15)][k], k=0..15
//   Store: lane l, acc elem e(0..7) → C[row + 2*e + (l>>4)][col + (l&15)]
//   M-tile row owned at store element e by this lane = warp_m_offset + 2*e + (l>>4).

#include <cuda_bf16.h>
#include <cuda_fp8.h>

typedef __bf16 v16bf __attribute__((ext_vector_type(16)));
typedef float  v8f   __attribute__((ext_vector_type(8)));

// Standard E4M3 (1-4-3, bias 7) decode via bit-math — matches the quantizer.
__device__ __forceinline__ float scl_fp8(unsigned char b) {
    unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u; float v;
    if (e == 0u)               v = (float)m * 0.001953125f;
    else if (e == 15u && m == 7u) v = 0.0f;
    else                       v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return s ? -v : v;
}
__device__ __forceinline__ float atlas_e4m3_to_f32(unsigned char b) { return scl_fp8(b); }

#define M_TILE 64
#define N_TILE_SM 64
#define N_TILE_LG 128
#define K_STEP 16
#define K_STEP_T 32
#define PAD 2
#define PAD_T 8
#define BP_PAD 16
#define GROUP_SIZE 16
#define K_STEP_T64 64
#define PAD_T64 8

__device__ __constant__ float E2M1_LUT_MOE[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// Synchronous 16-byte smem copy (cp.async replacement). Zero-fills when !pred.
__device__ __forceinline__ void sync_copy_16(void* dst_smem, const void* src_gmem, bool pred) {
    if (pred) *(uint4*)dst_smem = *(const uint4*)src_gmem;
    else      *(uint4*)dst_smem = uint4{0, 0, 0, 0};
}

// ═══════════════════════════════════════════════════════════════════
// N_TILE=64 pointer-table variant — BF16 WMMA, 4 n-sub-tiles. (decode path)
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

    __shared__ __nv_bfloat16 smem_A[M_TILE][K_STEP + PAD];
    __shared__ __nv_bfloat16 smem_B[K_STEP][N_TILE_SM + PAD];

    v8f acc[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) acc[i] = v8f{0, 0, 0, 0, 0, 0, 0, 0};

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
                    smem_B[k][n] = __float2bfloat16(E2M1_LUT_MOE[nibble] * scl_fp8(sb) * scale2);
                } else {
                    smem_B[k][n] = __float2bfloat16(0.0f);
                }
            }
        }
        __syncthreads();

        v16bf a;
        #pragma unroll
        for (int i = 0; i < 16; i++) a[i] = (__bf16)(float)smem_A[warp_m_offset + (lane_id & 15)][i];
        #pragma unroll
        for (int nb = 0; nb < 4; nb++) {
            v16bf b;
            #pragma unroll
            for (int k = 0; k < 16; k++) b[k] = (__bf16)(float)smem_B[k][nb * 16 + (lane_id & 15)];
            acc[nb] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc[nb]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int nb = 0; nb < 4; nb++)
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            unsigned int row_local = warp_m_offset + 2 * e + (lane_id >> 4);
            unsigned int c = cta_n + nb * 16 + (lane_id & 15);
            if ((int)(row_local + cta_m_local) < M_expert && c < N)
                C[(cta_m + row_local) * N + c] = __float2bfloat16(acc[nb][e]);
        }
}

// ═══════════════════════════════════════════════════════════════════
// N_TILE=128, K_STEP=32: NVFP4 B dequant→BF16, BF16 A, 2×WMMA K=16, 8 n-sub-tiles.
// ═══════════════════════════════════════════════════════════════════
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

    __shared__ __nv_bfloat16 smem_A[2][M_TILE][K_STEP_T + PAD_T];
    __shared__ unsigned char smem_Bp[2][K_STEP_T / 2][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_Bs[2][K_STEP_T / GROUP_SIZE][N_TILE_LG + BP_PAD];
    __shared__ __nv_bfloat16 smem_B_bf16[N_TILE_LG][K_STEP_T];
    __shared__ float smem_LUT[16];
    __shared__ int smem_tok[M_TILE];

    if (threadIdx.x < 16) smem_LUT[threadIdx.x] = E2M1_LUT_MOE[threadIdx.x];
    if (threadIdx.x < M_TILE) {
        int local_row = threadIdx.x;
        if (sorted_token_ids && (cta_m_local + local_row) < (unsigned int)M_expert)
            smem_tok[local_row] = sorted_token_ids[cta_m + local_row];
        else
            smem_tok[local_row] = (int)(cta_m + local_row);
    }
    __syncthreads();

    v8f acc[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = v8f{0, 0, 0, 0, 0, 0, 0, 0};

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
                sync_copy_16(&smem_A[(buf)][row][a_col], \
                    &A[(unsigned long long)a_row * K + gc], valid); \
            } \
        } \
        { \
            unsigned int kp = threadIdx.x >> 3; \
            unsigned int ns = (threadIdx.x & 7) << 4; \
            unsigned int gke = (kb) + (kp << 1); \
            unsigned int gns = cta_n + ns; \
            sync_copy_16(&smem_Bp[(buf)][kp][ns], \
                &B_expert[(unsigned long long)(gke >> 1) * N + gns], \
                (gke + 1 <= K) && (gns + 15 < N)); \
            if (kp < K_STEP_T / GROUP_SIZE) { \
                unsigned int sg = (kb) / GROUP_SIZE + kp; \
                sync_copy_16(&smem_Bs[(buf)][kp][ns], \
                    &S_expert[(unsigned long long)sg * N + gns], \
                    (gns + 15 < N)); \
            } \
        } \
    } while(0)

    #define MOE_DEQUANT(buf) do { \
        unsigned int my_n = threadIdx.x; \
        float sv0 = scl_fp8(smem_Bs[(buf)][0][my_n]) * scale2; \
        float sv1 = scl_fp8(smem_Bs[(buf)][1][my_n]) * scale2; \
        _Pragma("unroll") \
        for (int kp = 0; kp < 8; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            smem_B_bf16[my_n][kp * 2]     = __float2bfloat16(smem_LUT[packed & 0xF] * sv0); \
            smem_B_bf16[my_n][kp * 2 + 1] = __float2bfloat16(smem_LUT[packed >> 4] * sv0); \
        } \
        _Pragma("unroll") \
        for (int kp = 8; kp < 16; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            smem_B_bf16[my_n][kp * 2]     = __float2bfloat16(smem_LUT[packed & 0xF] * sv1); \
            smem_B_bf16[my_n][kp * 2 + 1] = __float2bfloat16(smem_LUT[packed >> 4] * sv1); \
        } \
    } while(0)

    #define MOE_COMPUTE_MMA(a_buf) do { \
        _Pragma("unroll") \
        for (int h = 0; h < 2; h++) { \
            v16bf a; \
            _Pragma("unroll") \
            for (int i = 0; i < 16; i++) \
                a[i] = (__bf16)(float)smem_A[(a_buf)][warp_m_offset + (lane_id & 15)][h * 16 + i]; \
            _Pragma("unroll") \
            for (int nb = 0; nb < 8; nb++) { \
                unsigned int nc = nb * 16 + (lane_id & 15); \
                v16bf b; \
                _Pragma("unroll") \
                for (int k = 0; k < 16; k++) b[k] = (__bf16)(float)smem_B_bf16[nc][h * 16 + k]; \
                acc[nb] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc[nb]); \
            } \
        } \
    } while(0)

    MOE_ISSUE_LOADS(0, 0);
    __syncthreads();
    MOE_DEQUANT(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        MOE_ISSUE_LOADS(nxt, k_base);
        MOE_COMPUTE_MMA(cur);
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
    for (int nb = 0; nb < 8; nb++)
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            unsigned int row_local = warp_m_offset + 2 * e + (lane_id >> 4);
            unsigned int c = cta_n + nb * 16 + (lane_id & 15);
            if ((int)(row_local + cta_m_local) < M_expert && c < N)
                C[(cta_m + row_local) * N + c] = __float2bfloat16(acc[nb][e]);
        }
}

// ═══════════════════════════════════════════════════════════════════
// K64 variant: N=128, K_STEP=64, NVFP4→BF16, 4×WMMA K=16, 8 n-sub-tiles.
// ═══════════════════════════════════════════════════════════════════
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

    __shared__ __nv_bfloat16 smem_A_k64[2][M_TILE][K_STEP_T64 + PAD_T64];
    __shared__ unsigned char smem_Bp_k64[2][K_STEP_T64 / 2][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_Bs_k64[2][K_STEP_T64 / GROUP_SIZE][N_TILE_LG + BP_PAD];
    __shared__ __nv_bfloat16 smem_B_bf16_k64[N_TILE_LG][K_STEP_T64];
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

    v8f acc[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = v8f{0, 0, 0, 0, 0, 0, 0, 0};

    const unsigned int M_eff = (unsigned int)M_expert;

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
                sync_copy_16(&smem_A_k64[(buf)][row][a_col], \
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
                sync_copy_16(&smem_Bp_k64[(buf)][kp_cur][ns], \
                    &B_expert[(unsigned long long)(gke >> 1) * N + gns], \
                    (gke + 1 < K) && (gns + 15 < N)); \
                if (kp_cur < K_STEP_T64 / GROUP_SIZE) { \
                    unsigned int sg = (kb) / GROUP_SIZE + kp_cur; \
                    sync_copy_16(&smem_Bs_k64[(buf)][kp_cur][ns], \
                        &S_expert[(unsigned long long)sg * N + gns], \
                        (gns + 15 < N)); \
                } \
            } \
        } \
    } while(0)

    #define K64_DEQUANT(buf) do { \
        unsigned int my_n = threadIdx.x; \
        float sv0 = scl_fp8(smem_Bs_k64[(buf)][0][my_n]) * scale2; \
        float sv1 = scl_fp8(smem_Bs_k64[(buf)][1][my_n]) * scale2; \
        float sv2 = scl_fp8(smem_Bs_k64[(buf)][2][my_n]) * scale2; \
        float sv3 = scl_fp8(smem_Bs_k64[(buf)][3][my_n]) * scale2; \
        _Pragma("unroll") \
        for (int kp = 0; kp < 8; kp++) { \
            unsigned char packed = smem_Bp_k64[(buf)][kp][my_n]; \
            smem_B_bf16_k64[my_n][kp * 2]     = __float2bfloat16(smem_LUT_k64[packed & 0xF] * sv0); \
            smem_B_bf16_k64[my_n][kp * 2 + 1] = __float2bfloat16(smem_LUT_k64[packed >> 4] * sv0); \
        } \
        _Pragma("unroll") \
        for (int kp = 8; kp < 16; kp++) { \
            unsigned char packed = smem_Bp_k64[(buf)][kp][my_n]; \
            smem_B_bf16_k64[my_n][kp * 2]     = __float2bfloat16(smem_LUT_k64[packed & 0xF] * sv1); \
            smem_B_bf16_k64[my_n][kp * 2 + 1] = __float2bfloat16(smem_LUT_k64[packed >> 4] * sv1); \
        } \
        _Pragma("unroll") \
        for (int kp = 16; kp < 24; kp++) { \
            unsigned char packed = smem_Bp_k64[(buf)][kp][my_n]; \
            smem_B_bf16_k64[my_n][kp * 2]     = __float2bfloat16(smem_LUT_k64[packed & 0xF] * sv2); \
            smem_B_bf16_k64[my_n][kp * 2 + 1] = __float2bfloat16(smem_LUT_k64[packed >> 4] * sv2); \
        } \
        _Pragma("unroll") \
        for (int kp = 24; kp < 32; kp++) { \
            unsigned char packed = smem_Bp_k64[(buf)][kp][my_n]; \
            smem_B_bf16_k64[my_n][kp * 2]     = __float2bfloat16(smem_LUT_k64[packed & 0xF] * sv3); \
            smem_B_bf16_k64[my_n][kp * 2 + 1] = __float2bfloat16(smem_LUT_k64[packed >> 4] * sv3); \
        } \
    } while(0)

    #define K64_COMPUTE_MMA(a_buf) do { \
        _Pragma("unroll") \
        for (int h = 0; h < 4; h++) { \
            v16bf a; \
            _Pragma("unroll") \
            for (int i = 0; i < 16; i++) \
                a[i] = (__bf16)(float)smem_A_k64[(a_buf)][warp_m_offset + (lane_id & 15)][h * 16 + i]; \
            _Pragma("unroll") \
            for (int nb = 0; nb < 8; nb++) { \
                unsigned int nc = nb * 16 + (lane_id & 15); \
                v16bf b; \
                _Pragma("unroll") \
                for (int k = 0; k < 16; k++) b[k] = (__bf16)(float)smem_B_bf16_k64[nc][h * 16 + k]; \
                acc[nb] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc[nb]); \
            } \
        } \
    } while(0)

    K64_ISSUE_LOADS(0, 0);
    __syncthreads();
    K64_DEQUANT(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T64; k_base < K; k_base += K_STEP_T64) {
        int nxt = 1 - cur;
        K64_ISSUE_LOADS(nxt, k_base);
        K64_COMPUTE_MMA(cur);
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
    for (int nb = 0; nb < 8; nb++)
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            unsigned int row_local = warp_m_offset + 2 * e + (lane_id >> 4);
            unsigned int c = cta_n + nb * 16 + (lane_id & 15);
            if ((int)(row_local + cta_m_local) < M_expert && c < N)
                C[(cta_m + row_local) * N + c] = __float2bfloat16(acc[nb][e]);
        }
}

// ═══════════════════════════════════════════════════════════════════
// K64 fused gate+up MoE GEMM. N=128, K=64. Dual output (C_gate/C_up).
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

    __shared__ __nv_bfloat16 smem_A_fgu64[2][M_TILE][K_STEP_T64 + PAD_T64];
    __shared__ unsigned char smem_Bp_fgu64[2][K_STEP_T64 / 2][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_Bs_fgu64[2][K_STEP_T64 / GROUP_SIZE][N_TILE_LG + BP_PAD];
    __shared__ __nv_bfloat16 smem_B_bf16_fgu64[N_TILE_LG][K_STEP_T64];
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

    v8f acc[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = v8f{0, 0, 0, 0, 0, 0, 0, 0};

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
                sync_copy_16(&smem_A_fgu64[(buf)][row][a_col], \
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
                sync_copy_16(&smem_Bp_fgu64[(buf)][kp_cur][ns], \
                    &B_expert[(unsigned long long)(gke >> 1) * N + gns], \
                    (gke + 1 < K) && (gns + 15 < N)); \
                if (kp_cur < K_STEP_T64 / GROUP_SIZE) { \
                    unsigned int sg = (kb) / GROUP_SIZE + kp_cur; \
                    sync_copy_16(&smem_Bs_fgu64[(buf)][kp_cur][ns], \
                        &S_expert[(unsigned long long)sg * N + gns], \
                        (gns + 15 < N)); \
                } \
            } \
        } \
    } while(0)

    #define FGU64_DEQUANT(buf) do { \
        unsigned int my_n = threadIdx.x; \
        float sv0 = scl_fp8(smem_Bs_fgu64[(buf)][0][my_n]) * scale2; \
        float sv1 = scl_fp8(smem_Bs_fgu64[(buf)][1][my_n]) * scale2; \
        float sv2 = scl_fp8(smem_Bs_fgu64[(buf)][2][my_n]) * scale2; \
        float sv3 = scl_fp8(smem_Bs_fgu64[(buf)][3][my_n]) * scale2; \
        _Pragma("unroll") \
        for (int kp = 0; kp < 8; kp++) { \
            unsigned char packed = smem_Bp_fgu64[(buf)][kp][my_n]; \
            smem_B_bf16_fgu64[my_n][kp * 2]     = __float2bfloat16(smem_LUT_fgu64[packed & 0xF] * sv0); \
            smem_B_bf16_fgu64[my_n][kp * 2 + 1] = __float2bfloat16(smem_LUT_fgu64[packed >> 4] * sv0); \
        } \
        _Pragma("unroll") \
        for (int kp = 8; kp < 16; kp++) { \
            unsigned char packed = smem_Bp_fgu64[(buf)][kp][my_n]; \
            smem_B_bf16_fgu64[my_n][kp * 2]     = __float2bfloat16(smem_LUT_fgu64[packed & 0xF] * sv1); \
            smem_B_bf16_fgu64[my_n][kp * 2 + 1] = __float2bfloat16(smem_LUT_fgu64[packed >> 4] * sv1); \
        } \
        _Pragma("unroll") \
        for (int kp = 16; kp < 24; kp++) { \
            unsigned char packed = smem_Bp_fgu64[(buf)][kp][my_n]; \
            smem_B_bf16_fgu64[my_n][kp * 2]     = __float2bfloat16(smem_LUT_fgu64[packed & 0xF] * sv2); \
            smem_B_bf16_fgu64[my_n][kp * 2 + 1] = __float2bfloat16(smem_LUT_fgu64[packed >> 4] * sv2); \
        } \
        _Pragma("unroll") \
        for (int kp = 24; kp < 32; kp++) { \
            unsigned char packed = smem_Bp_fgu64[(buf)][kp][my_n]; \
            smem_B_bf16_fgu64[my_n][kp * 2]     = __float2bfloat16(smem_LUT_fgu64[packed & 0xF] * sv3); \
            smem_B_bf16_fgu64[my_n][kp * 2 + 1] = __float2bfloat16(smem_LUT_fgu64[packed >> 4] * sv3); \
        } \
    } while(0)

    #define FGU64_COMPUTE_MMA(a_buf) do { \
        _Pragma("unroll") \
        for (int h = 0; h < 4; h++) { \
            v16bf a; \
            _Pragma("unroll") \
            for (int i = 0; i < 16; i++) \
                a[i] = (__bf16)(float)smem_A_fgu64[(a_buf)][warp_m_offset + (lane_id & 15)][h * 16 + i]; \
            _Pragma("unroll") \
            for (int nb = 0; nb < 8; nb++) { \
                unsigned int nc = nb * 16 + (lane_id & 15); \
                v16bf b; \
                _Pragma("unroll") \
                for (int k = 0; k < 16; k++) b[k] = (__bf16)(float)smem_B_bf16_fgu64[nc][h * 16 + k]; \
                acc[nb] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc[nb]); \
            } \
        } \
    } while(0)

    FGU64_ISSUE_LOADS(0, 0);
    __syncthreads();
    FGU64_DEQUANT(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T64; k_base < K; k_base += K_STEP_T64) {
        int nxt = 1 - cur;
        FGU64_ISSUE_LOADS(nxt, k_base);
        FGU64_COMPUTE_MMA(cur);
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
    for (int nb = 0; nb < 8; nb++)
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            unsigned int row_local = warp_m_offset + 2 * e + (lane_id >> 4);
            unsigned int c = cta_n + nb * 16 + (lane_id & 15);
            if ((int)(row_local + cta_m_local) < M_expert && c < N)
                C[(cta_m + row_local) * N + c] = __float2bfloat16(acc[nb][e]);
        }
}

// ═══════════════════════════════════════════════════════════════════
// Fused gate+up MoE GEMM. N=128, K=32. Dual output (C_gate/C_up).
// ═══════════════════════════════════════════════════════════════════
extern "C" __global__ void moe_w4a16_fused_gate_up_t(
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

    __shared__ __nv_bfloat16 smem_A[2][M_TILE][K_STEP_T + PAD_T];
    __shared__ unsigned char smem_Bp[2][K_STEP_T / 2][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_Bs[2][K_STEP_T / GROUP_SIZE][N_TILE_LG + BP_PAD];
    __shared__ __nv_bfloat16 smem_B_bf16[N_TILE_LG][K_STEP_T];
    __shared__ float smem_LUT[16];
    __shared__ int smem_tok[M_TILE];

    if (threadIdx.x < 16) smem_LUT[threadIdx.x] = E2M1_LUT_MOE[threadIdx.x];
    if (threadIdx.x < M_TILE) {
        int local_row = threadIdx.x;
        if (sorted_token_ids && (cta_m_local + local_row) < (unsigned int)M_expert)
            smem_tok[local_row] = sorted_token_ids[cta_m + local_row];
        else
            smem_tok[local_row] = (int)(cta_m + local_row);
    }
    __syncthreads();

    v8f acc[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = v8f{0, 0, 0, 0, 0, 0, 0, 0};

    const unsigned int M_eff = (unsigned int)M_expert;

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
                sync_copy_16(&smem_A[(buf)][row][a_col], \
                    &A[(unsigned long long)a_row * K + gc], valid); \
            } \
        } \
        { \
            unsigned int kp = threadIdx.x >> 3; \
            unsigned int ns = (threadIdx.x & 7) << 4; \
            unsigned int gke = (kb) + (kp << 1); \
            unsigned int gns = cta_n + ns; \
            sync_copy_16(&smem_Bp[(buf)][kp][ns], \
                &B_expert[(unsigned long long)(gke >> 1) * N + gns], \
                (gke + 1 <= K) && (gns + 15 < N)); \
            if (kp < K_STEP_T / GROUP_SIZE) { \
                unsigned int sg = (kb) / GROUP_SIZE + kp; \
                sync_copy_16(&smem_Bs[(buf)][kp][ns], \
                    &S_expert[(unsigned long long)sg * N + gns], \
                    (gns + 15 < N)); \
            } \
        } \
    } while(0)

    #define FGU_DEQUANT(buf) do { \
        unsigned int my_n = threadIdx.x; \
        float sv0 = scl_fp8(smem_Bs[(buf)][0][my_n]) * scale2; \
        float sv1 = scl_fp8(smem_Bs[(buf)][1][my_n]) * scale2; \
        _Pragma("unroll") \
        for (int kp = 0; kp < 8; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            smem_B_bf16[my_n][kp * 2]     = __float2bfloat16(smem_LUT[packed & 0xF] * sv0); \
            smem_B_bf16[my_n][kp * 2 + 1] = __float2bfloat16(smem_LUT[packed >> 4] * sv0); \
        } \
        _Pragma("unroll") \
        for (int kp = 8; kp < 16; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            smem_B_bf16[my_n][kp * 2]     = __float2bfloat16(smem_LUT[packed & 0xF] * sv1); \
            smem_B_bf16[my_n][kp * 2 + 1] = __float2bfloat16(smem_LUT[packed >> 4] * sv1); \
        } \
    } while(0)

    #define FGU_COMPUTE_MMA(a_buf) do { \
        _Pragma("unroll") \
        for (int h = 0; h < 2; h++) { \
            v16bf a; \
            _Pragma("unroll") \
            for (int i = 0; i < 16; i++) \
                a[i] = (__bf16)(float)smem_A[(a_buf)][warp_m_offset + (lane_id & 15)][h * 16 + i]; \
            _Pragma("unroll") \
            for (int nb = 0; nb < 8; nb++) { \
                unsigned int nc = nb * 16 + (lane_id & 15); \
                v16bf b; \
                _Pragma("unroll") \
                for (int k = 0; k < 16; k++) b[k] = (__bf16)(float)smem_B_bf16[nc][h * 16 + k]; \
                acc[nb] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc[nb]); \
            } \
        } \
    } while(0)

    FGU_ISSUE_LOADS(0, 0);
    __syncthreads();
    FGU_DEQUANT(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        FGU_ISSUE_LOADS(nxt, k_base);
        FGU_COMPUTE_MMA(cur);
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
    for (int nb = 0; nb < 8; nb++)
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            unsigned int row_local = warp_m_offset + 2 * e + (lane_id >> 4);
            unsigned int c = cta_n + nb * 16 + (lane_id & 15);
            if ((int)(row_local + cta_m_local) < M_expert && c < N)
                C[(cta_m + row_local) * N + c] = __float2bfloat16(acc[nb][e]);
        }
}

// ═══════════════════════════════════════════════════════════════════
// FP8-input MoE GEMM: A [M, K] FP8 × B_expert NVFP4 → C [M, N] BF16.
// N=128, K=32. A decoded E4M3→BF16, B dequant NVFP4→BF16, 2×WMMA K=16.
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

    __shared__ unsigned char smem_Af2[2][M_TILE][K_STEP_T];
    __shared__ unsigned char smem_Bp2[2][K_STEP_T / 2][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_Bs2[2][K_STEP_T / GROUP_SIZE][N_TILE_LG + BP_PAD];
    __shared__ __nv_bfloat16 smem_B2_bf16[N_TILE_LG][K_STEP_T];
    __shared__ float smem_LUT2[16];
    __shared__ int smem_tok2[M_TILE];

    if (threadIdx.x < 16) smem_LUT2[threadIdx.x] = E2M1_LUT_MOE[threadIdx.x];
    if (threadIdx.x < M_TILE) {
        int local_row = threadIdx.x;
        if (sorted_token_ids && (cta_m_local + local_row) < (unsigned int)M_expert)
            smem_tok2[local_row] = sorted_token_ids[cta_m + local_row];
        else
            smem_tok2[local_row] = (int)(cta_m + local_row);
    }
    __syncthreads();

    v8f acc[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = v8f{0, 0, 0, 0, 0, 0, 0, 0};

    const unsigned int M_eff = (unsigned int)M_expert;

    #define MOE_FF_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 1; \
            unsigned int a_col = (threadIdx.x & 1) << 4; \
            unsigned int gc = (kb) + a_col; \
            unsigned int row = a_row_base; \
            bool valid = (cta_m_local + row) < M_eff && (gc + 15 < K); \
            unsigned int a_row = (unsigned int)smem_tok2[row]; \
            sync_copy_16(&smem_Af2[(buf)][row][a_col], \
                &A_fp8[(unsigned long long)a_row * K + gc], valid); \
        } \
        { \
            unsigned int kp = threadIdx.x >> 3; \
            unsigned int ns = (threadIdx.x & 7) << 4; \
            unsigned int gke = (kb) + (kp << 1); \
            unsigned int gns = cta_n + ns; \
            sync_copy_16(&smem_Bp2[(buf)][kp][ns], \
                &B_exp[(unsigned long long)(gke >> 1) * N + gns], \
                (gke + 1 <= K) && (gns + 15 < N)); \
            if (kp < K_STEP_T / GROUP_SIZE) { \
                unsigned int sg = (kb) / GROUP_SIZE + kp; \
                sync_copy_16(&smem_Bs2[(buf)][kp][ns], \
                    &S_exp[(unsigned long long)sg * N + gns], \
                    (gns + 15 < N)); \
            } \
        } \
    } while(0)

    #define MOE_FF_DEQUANT(buf) do { \
        unsigned int my_n = threadIdx.x; \
        float sv0 = scl_fp8(smem_Bs2[(buf)][0][my_n]) * scale2; \
        float sv1 = scl_fp8(smem_Bs2[(buf)][1][my_n]) * scale2; \
        _Pragma("unroll") \
        for (int kp = 0; kp < 8; kp++) { \
            unsigned char packed = smem_Bp2[(buf)][kp][my_n]; \
            smem_B2_bf16[my_n][kp * 2]     = __float2bfloat16(smem_LUT2[packed & 0xF] * sv0); \
            smem_B2_bf16[my_n][kp * 2 + 1] = __float2bfloat16(smem_LUT2[packed >> 4] * sv0); \
        } \
        _Pragma("unroll") \
        for (int kp = 8; kp < 16; kp++) { \
            unsigned char packed = smem_Bp2[(buf)][kp][my_n]; \
            smem_B2_bf16[my_n][kp * 2]     = __float2bfloat16(smem_LUT2[packed & 0xF] * sv1); \
            smem_B2_bf16[my_n][kp * 2 + 1] = __float2bfloat16(smem_LUT2[packed >> 4] * sv1); \
        } \
    } while(0)

    // A is FP8 row-major [M_TILE][K_STEP_T]: decode E4M3→BF16 per lane.
    #define MOE_FF_COMPUTE(a_buf) do { \
        _Pragma("unroll") \
        for (int h = 0; h < 2; h++) { \
            v16bf a; \
            _Pragma("unroll") \
            for (int i = 0; i < 16; i++) \
                a[i] = (__bf16)(float)atlas_e4m3_to_f32(smem_Af2[(a_buf)][warp_m_offset + (lane_id & 15)][h * 16 + i]); \
            _Pragma("unroll") \
            for (int nb = 0; nb < 8; nb++) { \
                unsigned int nc = nb * 16 + (lane_id & 15); \
                v16bf b; \
                _Pragma("unroll") \
                for (int k = 0; k < 16; k++) b[k] = (__bf16)(float)smem_B2_bf16[nc][h * 16 + k]; \
                acc[nb] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc[nb]); \
            } \
        } \
    } while(0)

    MOE_FF_LOADS(0, 0);
    __syncthreads();
    MOE_FF_DEQUANT(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        MOE_FF_LOADS(nxt, k_base);
        MOE_FF_COMPUTE(cur);
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
    for (int nb = 0; nb < 8; nb++)
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            unsigned int row_local = warp_m_offset + 2 * e + (lane_id >> 4);
            unsigned int c = cta_n + nb * 16 + (lane_id & 15);
            if ((int)(row_local + cta_m_local) < M_expert && c < N)
                C[(cta_m + row_local) * N + c] = __float2bfloat16(acc[nb][e]);
        }
}
