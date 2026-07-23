// SPDX-License-Identifier: AGPL-3.0-only

// Atlas W4A16 GEMM — 35B/27B model shadow. HIP/gfx1151 (AMD WMMA) port.
//
// Ported from the NVIDIA/SCALE mma.sync version. Key transforms:
//   * mma.sync.m16n8k16.bf16  → __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32
//       NVIDIA n8 fragments become AMD n16: a 64-wide N tile = 4 WMMA ops
//       (was 8 n-tiles), a 128-wide N tile = 8 WMMA ops (was 16 n-tiles).
//   * mma.sync.m16n8k32.e4m3 (FP8) → decode E4M3→BF16 in registers + 2× WMMA K=16.
//       gfx1151 has no FP8 tensor cores; the SCALE path itself already decoded
//       FP8→BF16 then ran bf16 MMA, so this is numerically the same strategy.
//   * cp.async.ca.shared.global → synchronous 16-byte uint4 smem copy.
//   * commit_group / wait_group  → dropped (loads complete synchronously).
//
// WMMA fragment layout (validated bit-exact in w4a16_wmma_ref.hip):
//   A load (M×K row-major smem): lane l → a[i] = smem_A[m_row + (l&15)][i], i=0..15
//   B load (K×N smem):           lane l → b[k] = smem_B[k][n_base + (l&15)], k=0..15
//   B load (N×K row-major smem): lane l → b[k] = smem_B[n_base + (l&15)][k], k=0..15
//   Store: lane l, acc elem e(0..7) → C[row + 2*e + (l>>4)][col + (l&15)]

#include <cuda_bf16.h>
#include <cuda_fp8.h>

typedef __bf16 v16bf __attribute__((ext_vector_type(16)));
typedef float  v8f   __attribute__((ext_vector_type(8)));

// Standard E4M3 (1-4-3, bias 7) decode via pure bit-math. On real NVIDIA this is
// byte-identical to (float)__nv_fp8_e4m3; on SCALE/gfx1151 the built-in
// __nv_fp8_e4m3->float decode is a NON-STANDARD narrow format which mismatches
// the standard E4M3 scales written by quantize_bf16_to_nvfp4. Use this to match
// the encoder.
__device__ __forceinline__ float scl_fp8(unsigned char b) {
    unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u; float v;
    if (e == 0u)               v = (float)m * 0.001953125f;            // subnormal m*2^-9
    else if (e == 15u && m == 7u) v = 0.0f;                            // NaN -> 0
    else                       v = __uint_as_float(((e + 120u) << 23) | (m << 20)); // 2^(e-7)*(1+m/8)
    return s ? -v : v;
}

__device__ __forceinline__ unsigned char scl_enc_fp8(float v) {
    if (v != v) return 0x7F;
    unsigned int bb = __float_as_uint(v); unsigned int sign = (bb >> 31) & 1u;
    int e = (int)((bb >> 23) & 0xFF) - 127; unsigned int man = bb & 0x7FFFFFu;
    int ee = e + 7; unsigned int em;
    if (ee < 1) { ee = 0; em = 0; if (e >= -10) { float a = v < 0 ? -v : v; em = (unsigned int)(a / 0.001953125f + 0.5f); if (em > 7u) em = 7u; } }
    else if (ee > 15) { ee = 15; em = 6; }
    else { em = (man + (1u << 19)) >> 20; if (em > 7u) { em = 0; ee++; if (ee > 15) { ee = 15; em = 6; } } }
    return (unsigned char)((sign << 7) | ((unsigned)ee << 3) | em);
}

// FP8 E4M3 standard decode (matches the quantizer; SCALE's __NV_E4M3 is non-standard).
__device__ __forceinline__ float atlas_e4m3_to_f32(unsigned char b) { return scl_fp8(b); }

// Encode a pair of f32 → packed E4M3x2 (hi byte = e4m3(a_hi), lo byte = e4m3(b_lo)).
// Pure bit-math; matches the NVIDIA cvt.rn.satfinite.e4m3x2.f32 semantics.
__device__ __forceinline__ unsigned short atlas_cvt_e4m3x2_f32(float a_hi, float b_lo) {
    unsigned a8 = (unsigned)scl_enc_fp8(a_hi);
    unsigned b8 = (unsigned)scl_enc_fp8(b_lo);
    return (unsigned short)((a8 << 8) | (b8 & 0xFFu));
}

#define M_TILE 64
#define N_TILE_SM 64
#define N_TILE_LG 128
#define K_STEP 16
#define K_STEP_T 32
#define PAD 2
#define PAD_T 8        // 16-byte aligned smem rows: (32+8)*2=80, 80%16=0
#define BP_PAD 16      // smem_Bp row padding: stride 144 is 16-byte aligned
#define GROUP_SIZE 16

__device__ __constant__ float E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// ── Synchronous 16-byte smem copy (cp.async replacement) ────────────
// Copies 16 bytes gmem→smem when pred, else zero-fills, to preserve the
// predicated cp.async.16 semantics (out-of-bounds rows became zero).
__device__ __forceinline__ void sync_copy_16(void* dst_smem, const void* src_gmem, bool pred) {
    if (pred) {
        *(uint4*)dst_smem = *(const uint4*)src_gmem;
    } else {
        *(uint4*)dst_smem = uint4{0, 0, 0, 0};
    }
}

// ═══════════════════════════════════════════════════════════════════
// Original layout w4a16_gemm: N_TILE=64, BF16 WMMA. 4 WMMA n-sub-tiles.
// ═══════════════════════════════════════════════════════════════════
extern "C" __global__ void w4a16_gemm(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_m = blockIdx.y * M_TILE;
    const unsigned int cta_n = blockIdx.x * N_TILE_SM;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;

    __shared__ __nv_bfloat16 smem_A[M_TILE][K_STEP + PAD];
    __shared__ __nv_bfloat16 smem_B[K_STEP][N_TILE_SM + PAD];

    v8f acc[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) acc[i] = v8f{0, 0, 0, 0, 0, 0, 0, 0};

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
                unsigned int gr = cta_m + row;
                unsigned int gc = k_base + col;
                smem_A[row][col] = (gr < M && gc < K) ? A[gr * K + gc] : __float2bfloat16(0.0f);
            }
        }
        {
            #pragma unroll
            for (unsigned int i = 0; i < 8; i++) {
                unsigned int idx = threadIdx.x * 8 + i;
                unsigned int k = idx / N_TILE_SM;
                unsigned int n = idx % N_TILE_SM;
                unsigned int gk = k_base + k;
                unsigned int gn = cta_n + n;
                if (gk < K && gn < N) {
                    unsigned int k_pair = gk / 2;
                    unsigned char packed_byte = B_packed[(unsigned long long)gn * half_K + k_pair];
                    unsigned int nibble = (gk & 1) ? (packed_byte >> 4) : (packed_byte & 0xF);
                    unsigned int sg = gk / GROUP_SIZE;
                    unsigned char sb = B_scale[(unsigned long long)gn * num_groups + sg];
                    smem_B[k][n] = __float2bfloat16(E2M1_LUT[nibble] * scl_fp8(sb) * scale2);
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
            unsigned int r = cta_m + warp_m_offset + 2 * e + (lane_id >> 4);
            unsigned int c = cta_n + nb * 16 + (lane_id & 15);
            if (r < M && c < N) C[r * N + c] = __float2bfloat16(acc[nb][e]);
        }
}

// ═══════════════════════════════════════════════════════════════════
// w4a16_gemm_t: transposed B, N_TILE=128, 2-stage double-buffered pipeline.
// B_packed[K/2, N], B_scale[K/GROUP_SIZE, N]. Dequant NVFP4→BF16.
// 8 WMMA n-sub-tiles (128 N / 16). K_STEP_T=32 → 2 WMMA K=16 ops per step.
// ═══════════════════════════════════════════════════════════════════
extern "C" __global__ void w4a16_gemm_t(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * M_TILE;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;

    __shared__ __nv_bfloat16 smem_A[2][M_TILE][K_STEP_T + PAD_T];
    __shared__ unsigned char smem_Bp[2][K_STEP_T / 2][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_Bs[2][K_STEP_T / GROUP_SIZE][N_TILE_LG + BP_PAD];
    __shared__ __nv_bfloat16 smem_B_bf16[N_TILE_LG][K_STEP_T];
    __shared__ float smem_LUT[16];

    if (threadIdx.x < 16) smem_LUT[threadIdx.x] = E2M1_LUT[threadIdx.x];

    v8f acc[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = v8f{0, 0, 0, 0, 0, 0, 0, 0};


    #define ISSUE_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 2; \
            unsigned int a_col = (threadIdx.x & 3) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 2; rnd++) { \
                unsigned int row = rnd * 32 + a_row_base; \
                unsigned int gr = cta_m + row; \
                sync_copy_16(&smem_A[(buf)][row][a_col], \
                    &A[gr * K + gc], (gr < M) && (gc + 7 < K)); \
            } \
        } \
        { \
            unsigned int kp = threadIdx.x >> 3; \
            unsigned int ns = (threadIdx.x & 7) << 4; \
            unsigned int gke = (kb) + (kp << 1); \
            unsigned int gns = cta_n + ns; \
            sync_copy_16(&smem_Bp[(buf)][kp][ns], \
                &B_packed[(unsigned long long)(gke >> 1) * N + gns], \
                (gke + 1 <= K) && (gns + 15 < N)); \
            if (kp < K_STEP_T / GROUP_SIZE) { \
                unsigned int sg = (kb) / GROUP_SIZE + kp; \
                sync_copy_16(&smem_Bs[(buf)][kp][ns], \
                    &B_scale[(unsigned long long)sg * N + gns], \
                    (gns + 15 < N)); \
            } \
        } \
    } while(0)

    // Dequant B: NVFP4 -> BF16 directly into smem_B_bf16[n][k].
    #define DEQUANT_T(buf) do { \
        unsigned int my_n = threadIdx.x; \
        unsigned char sb0 = smem_Bs[(buf)][0][my_n]; \
        unsigned char sb1 = smem_Bs[(buf)][1][my_n]; \
        float sv0 = scl_fp8(sb0) * scale2, sv1 = scl_fp8(sb1) * scale2; \
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

    // BF16 WMMA: 2× K=16 over the 32-wide K step. 8 n-sub-tiles (128 N).
    // smem_A[buf] is [M_TILE][K_STEP_T+PAD_T]; smem_B_bf16 is [N][K_STEP_T].
    #define COMPUTE_MMA(a_buf) do { \
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
                for (int k = 0; k < 16; k++) \
                    b[k] = (__bf16)(float)smem_B_bf16[nc][h * 16 + k]; \
                acc[nb] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc[nb]); \
            } \
        } \
    } while(0)

    ISSUE_LOADS(0, 0);
    __syncthreads();
    DEQUANT_T(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        ISSUE_LOADS(nxt, k_base);
        COMPUTE_MMA(cur);
        __syncthreads();
        DEQUANT_T(nxt);
        __syncthreads();
        cur = nxt;
    }
    COMPUTE_MMA(cur);

    #undef ISSUE_LOADS
    #undef DEQUANT_T
    #undef COMPUTE_MMA

    #pragma unroll
    for (int nb = 0; nb < 8; nb++)
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            unsigned int r = cta_m + warp_m_offset + 2 * e + (lane_id >> 4);
            unsigned int c = cta_n + nb * 16 + (lane_id & 15);
            if (r < M && c < N) C[r * N + c] = __float2bfloat16(acc[nb][e]);
        }
}

// ═══════════════════════════════════════════════════════════════════
// fp8_gemm_t: BF16 A × FP8(E4M3) B (pre-dequanted), N_TILE=128, K_STEP=32.
// gfx1151: no FP8 tensor cores → decode A(BF16) and B(E4M3) to BF16, 2× WMMA K=16.
// B_fp8[N, K] row-major: smem_B[buf][n][k] holds the E4M3 weight byte.
// ═══════════════════════════════════════════════════════════════════
extern "C" __global__ void fp8_gemm_t(
    const __nv_bfloat16* __restrict__ A,       // [M, K] BF16
    const unsigned char* __restrict__ B_fp8,   // [N, K] FP8 E4M3
    __nv_bfloat16* __restrict__ C,             // [M, N] BF16
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * M_TILE;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;

    __shared__ __nv_bfloat16 smem_A[2][M_TILE][K_STEP_T + PAD_T];
    __shared__ unsigned char smem_B[2][N_TILE_LG][K_STEP_T];

    v8f acc[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = v8f{0, 0, 0, 0, 0, 0, 0, 0};


    #define FP8_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 2; \
            unsigned int a_col = (threadIdx.x & 3) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 2; rnd++) { \
                unsigned int row = rnd * 32 + a_row_base; \
                unsigned int gr = cta_m + row; \
                sync_copy_16(&smem_A[(buf)][row][a_col], \
                    &A[(unsigned long long)gr * K + gc], \
                    (gr < M) && (gc + 7 < K)); \
            } \
        } \
        { \
            unsigned int my_n = threadIdx.x; \
            unsigned int gn = cta_n + my_n; \
            bool valid = (gn < N) && ((kb) + 31 < K); \
            sync_copy_16(&smem_B[(buf)][my_n][0], \
                &B_fp8[(unsigned long long)gn * K + (kb)], valid); \
            sync_copy_16(&smem_B[(buf)][my_n][16], \
                &B_fp8[(unsigned long long)gn * K + (kb) + 16], valid); \
        } \
    } while(0)

    // Decode A(BF16) and B(E4M3) → BF16 per lane; 2× WMMA K=16. 8 n-sub-tiles.
    #define FP8_COMPUTE(a_buf, b_buf) do { \
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
                for (int k = 0; k < 16; k++) \
                    b[k] = (__bf16)(float)atlas_e4m3_to_f32(smem_B[(b_buf)][nc][h * 16 + k]); \
                acc[nb] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc[nb]); \
            } \
        } \
    } while(0)

    FP8_LOADS(0, 0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        FP8_LOADS(nxt, k_base);
        FP8_COMPUTE(cur, cur);
        __syncthreads();
        cur = nxt;
    }
    FP8_COMPUTE(cur, cur);

    #undef FP8_LOADS
    #undef FP8_COMPUTE

    #pragma unroll
    for (int nb = 0; nb < 8; nb++)
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            unsigned int r = cta_m + warp_m_offset + 2 * e + (lane_id >> 4);
            unsigned int c = cta_n + nb * 16 + (lane_id & 15);
            if (r < M && c < N) C[r * N + c] = __float2bfloat16(acc[nb][e]);
        }
}

// ═══════════════════════════════════════════════════════════════════
// Pre-dequant: NVFP4 [N, K/2] + scales [N, K/GROUP_SIZE] → FP8 [N, K].
// No tensor core — portable. One-time conversion at model load.
// ═══════════════════════════════════════════════════════════════════
extern "C" __global__ void predequant_nvfp4_to_fp8(
    const unsigned char* __restrict__ B_packed,  // [N, K/2]
    const unsigned char* __restrict__ B_scale,   // [N, K/GROUP_SIZE]
    float scale2,
    unsigned char* __restrict__ B_fp8,           // [N, K]
    unsigned int N, unsigned int K
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int half_K = K / 2;
    unsigned int total = N * half_K;
    if (idx >= total) return;

    unsigned int n = idx / half_K;
    unsigned int k_pair = idx % half_K;
    unsigned int k_even = k_pair * 2;

    unsigned char packed = B_packed[(unsigned long long)n * half_K + k_pair];
    unsigned int group = k_even / GROUP_SIZE;
    unsigned char sb = B_scale[(unsigned long long)n * (K / GROUP_SIZE) + group];
    float sv = scl_fp8(sb) * scale2;

    float val_lo = E2M1_LUT[packed & 0xF] * sv;
    float val_hi = E2M1_LUT[packed >> 4] * sv;

    unsigned short fp8_pair = atlas_cvt_e4m3x2_f32(val_hi, val_lo);
    *(unsigned short*)&B_fp8[(unsigned long long)n * K + k_even] = fp8_pair;
}

// ═══════════════════════════════════════════════════════════════════
// BF16 → FP8 E4M3 activation conversion. No tensor core — portable.
// Each thread converts 2 BF16 values → 2 FP8 values.
// ═══════════════════════════════════════════════════════════════════
extern "C" __global__ void bf16_to_fp8(
    const __nv_bfloat16* __restrict__ src,   // [M, K] BF16
    unsigned char* __restrict__ dst,          // [M, K] FP8 E4M3
    unsigned int total_elements               // M * K (must be even)
) {
    unsigned int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx >= total_elements) return;

    unsigned int p = *(const unsigned int*)&src[idx];
    unsigned short bf0 = (unsigned short)(p & 0xFFFFu);
    unsigned short bf1 = (unsigned short)(p >> 16);
    float f0 = __bfloat162float(__ushort_as_bfloat16(bf0));
    float f1 = __bfloat162float(__ushort_as_bfloat16(bf1));
    unsigned short fp8_pair = atlas_cvt_e4m3x2_f32(f1, f0);
    *(unsigned short*)&dst[idx] = fp8_pair;
}

// ═══════════════════════════════════════════════════════════════════
// fp8_fp8_gemm_t: FP8 A × FP8 B → BF16 C. N_TILE=128, K_STEP=32.
// gfx1151: decode both A and B (E4M3) to BF16, 2× WMMA K=16. 8 n-sub-tiles.
// A_fp8 smem [M_TILE][A_FP8_STRIDE] row-major; B_fp8 smem [N][K_STEP_T] row-major.
// ═══════════════════════════════════════════════════════════════════
#define A_FP8_STRIDE 32  // K_STEP_T = 32 bytes per row for FP8

extern "C" __global__ void fp8_fp8_gemm_t(
    const unsigned char* __restrict__ A_fp8,  // [M, K] FP8 E4M3
    const unsigned char* __restrict__ B_fp8,  // [N, K] FP8 E4M3
    __nv_bfloat16* __restrict__ C,            // [M, N] BF16
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * M_TILE;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;

    __shared__ unsigned char smem_Af[2][M_TILE][A_FP8_STRIDE];
    __shared__ unsigned char smem_Bf[2][N_TILE_LG][K_STEP_T];

    v8f acc[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = v8f{0, 0, 0, 0, 0, 0, 0, 0};

    #define FF_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 1; \
            unsigned int a_col = (threadIdx.x & 1) << 4; \
            unsigned int gc = (kb) + a_col; \
            unsigned int row = a_row_base; \
            unsigned int gr = cta_m + row; \
            sync_copy_16(&smem_Af[(buf)][row][a_col], \
                &A_fp8[(unsigned long long)gr * K + gc], \
                (gr < M) && (gc + 15 < K)); \
        } \
        { \
            unsigned int my_n = threadIdx.x; \
            unsigned int gn = cta_n + my_n; \
            bool valid = (gn < N) && ((kb) + 31 < K); \
            sync_copy_16(&smem_Bf[(buf)][my_n][0], \
                &B_fp8[(unsigned long long)gn * K + (kb)], valid); \
            sync_copy_16(&smem_Bf[(buf)][my_n][16], \
                &B_fp8[(unsigned long long)gn * K + (kb) + 16], valid); \
        } \
    } while(0)

    #define FF_COMPUTE(a_buf, b_buf) do { \
        _Pragma("unroll") \
        for (int h = 0; h < 2; h++) { \
            v16bf a; \
            _Pragma("unroll") \
            for (int i = 0; i < 16; i++) \
                a[i] = (__bf16)(float)atlas_e4m3_to_f32(smem_Af[(a_buf)][warp_m_offset + (lane_id & 15)][h * 16 + i]); \
            _Pragma("unroll") \
            for (int nb = 0; nb < 8; nb++) { \
                unsigned int nc = nb * 16 + (lane_id & 15); \
                v16bf b; \
                _Pragma("unroll") \
                for (int k = 0; k < 16; k++) \
                    b[k] = (__bf16)(float)atlas_e4m3_to_f32(smem_Bf[(b_buf)][nc][h * 16 + k]); \
                acc[nb] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc[nb]); \
            } \
        } \
    } while(0)

    FF_LOADS(0, 0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        FF_LOADS(nxt, k_base);
        FF_COMPUTE(cur, cur);
        __syncthreads();
        cur = nxt;
    }
    FF_COMPUTE(cur, cur);

    #undef FF_LOADS
    #undef FF_COMPUTE

    #pragma unroll
    for (int nb = 0; nb < 8; nb++)
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            unsigned int r = cta_m + warp_m_offset + 2 * e + (lane_id >> 4);
            unsigned int c = cta_n + nb * 16 + (lane_id & 15);
            if (r < M && c < N) C[r * N + c] = __float2bfloat16(acc[nb][e]);
        }
}

// ═══════════════════════════════════════════════════════════════════
// w4a16_gemm_t_k64: same as w4a16_gemm_t but K_STEP_T64=64 (halves outer loop).
// Dequant NVFP4→BF16 (gfx1151 has no FP8 tensor cores, so the K64 FP8 round-trip
// of the NVIDIA path is replaced by a direct BF16 staging buffer). 4 WMMA K=16
// ops per step × 8 n-sub-tiles. K must be divisible by 64.
// ═══════════════════════════════════════════════════════════════════
#define K_STEP_T64 64
#define PAD_T64    8   // (64+8)*2=144, 144%16=0

extern "C" __global__ void w4a16_gemm_t_k64(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * M_TILE;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;

    __shared__ __nv_bfloat16 smem_A_k64[2][M_TILE][K_STEP_T64 + PAD_T64];
    __shared__ unsigned char smem_Bp_k64[2][K_STEP_T64 / 2][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_Bs_k64[2][K_STEP_T64 / GROUP_SIZE][N_TILE_LG + BP_PAD];
    __shared__ __nv_bfloat16 smem_B_bf16_k64[N_TILE_LG][K_STEP_T64];
    __shared__ float smem_LUT_k64[16];

    if (threadIdx.x < 16) smem_LUT_k64[threadIdx.x] = E2M1_LUT[threadIdx.x];

    v8f acc[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = v8f{0, 0, 0, 0, 0, 0, 0, 0};


    #define K64_ISSUE_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 3; \
            unsigned int a_col = (threadIdx.x & 7) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 4; rnd++) { \
                unsigned int row = rnd * 16 + a_row_base; \
                unsigned int gr = cta_m + row; \
                sync_copy_16(&smem_A_k64[(buf)][row][a_col], \
                    &A[(unsigned long long)gr * K + gc], \
                    (gr < M) && (gc + 7 < K)); \
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
                    &B_packed[(unsigned long long)(gke >> 1) * N + gns], \
                    (gke + 1 <= K) && (gns + 15 < N)); \
                if (kp_cur < K_STEP_T64 / GROUP_SIZE) { \
                    unsigned int sg = (kb) / GROUP_SIZE + kp_cur; \
                    sync_copy_16(&smem_Bs_k64[(buf)][kp_cur][ns], \
                        &B_scale[(unsigned long long)sg * N + gns], \
                        (gns + 15 < N)); \
                } \
            } \
        } \
    } while(0)

    // 4 scale groups, 32 dequant iters: sv0→K{0..15}, sv1→K{16..31},
    // sv2→K{32..47}, sv3→K{48..63}. Dequant directly to BF16.
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

    // 4 WMMA K=16 ops (K=0..15,16..31,32..47,48..63) × 8 n-sub-tiles.
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
                for (int k = 0; k < 16; k++) \
                    b[k] = (__bf16)(float)smem_B_bf16_k64[nc][h * 16 + k]; \
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
            unsigned int r = cta_m + warp_m_offset + 2 * e + (lane_id >> 4);
            unsigned int c = cta_n + nb * 16 + (lane_id & 15);
            if (r < M && c < N) C[r * N + c] = __float2bfloat16(acc[nb][e]);
        }
}
#undef K_STEP_T64
#undef PAD_T64

// ═══════════════════════════════════════════════════════════════════
// w4a16_gemm_t_m128: 2 consecutive 64-row M-chunks per CTA. NVFP4→BF16.
// Two accumulator sets (acc0/acc1), each 8 WMMA n-sub-tiles.
// ═══════════════════════════════════════════════════════════════════
extern "C" __global__
__launch_bounds__(128, 3)
void w4a16_gemm_t_m128(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    const float scale2,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n  = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m  = blockIdx.y * (2 * M_TILE);
    if (cta_m >= M) return;

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;

    __shared__ __nv_bfloat16 smem_A[2][2 * M_TILE][K_STEP_T + PAD_T];
    __shared__ unsigned char smem_Bp[2][K_STEP_T / 2][N_TILE_LG + BP_PAD];
    __shared__ unsigned char smem_Bs[2][K_STEP_T / GROUP_SIZE][N_TILE_LG + BP_PAD];
    __shared__ __nv_bfloat16 smem_B_bf16[N_TILE_LG][K_STEP_T];
    __shared__ float smem_LUT[16];

    if (threadIdx.x < 16) smem_LUT[threadIdx.x] = E2M1_LUT[threadIdx.x];

    v8f acc0[8], acc1[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        acc0[i] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
        acc1[i] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
    }


    #define M128_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 2; \
            unsigned int a_col      = (threadIdx.x & 3) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 4; rnd++) { \
                unsigned int row = (unsigned int)(rnd * 32) + a_row_base; \
                unsigned int gr  = cta_m + row; \
                sync_copy_16(&smem_A[(buf)][row][a_col], \
                    &A[(unsigned long long)gr * K + gc], \
                    (gr < M) && (gc + 7 < K)); \
            } \
        } \
        { \
            unsigned int kp  = threadIdx.x >> 3; \
            unsigned int ns  = (threadIdx.x & 7) << 4; \
            unsigned int gke = (kb) + (kp << 1); \
            unsigned int gns = cta_n + ns; \
            sync_copy_16(&smem_Bp[(buf)][kp][ns], \
                &B_packed[(unsigned long long)(gke >> 1) * N + gns], \
                (gke + 1 <= K) && (gns + 15 < N)); \
            if (kp < K_STEP_T / GROUP_SIZE) { \
                unsigned int sg = (kb) / GROUP_SIZE + kp; \
                sync_copy_16(&smem_Bs[(buf)][kp][ns], \
                    &B_scale[(unsigned long long)sg * N + gns], \
                    (gns + 15 < N)); \
            } \
        } \
    } while(0)

    #define M128_DEQUANT(buf) do { \
        unsigned int my_n = threadIdx.x; \
        float sv0 = scl_fp8(smem_Bs[(buf)][0][my_n]) * scale2; \
        float sv1 = scl_fp8(smem_Bs[(buf)][1][my_n]) * scale2; \
        _Pragma("unroll") \
        for (int kp = 0; kp < 8; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            smem_B_bf16[my_n][kp * 2]     = __float2bfloat16(smem_LUT[packed & 0xF] * sv0); \
            smem_B_bf16[my_n][kp * 2 + 1] = __float2bfloat16(smem_LUT[packed >> 4]  * sv0); \
        } \
        _Pragma("unroll") \
        for (int kp = 8; kp < 16; kp++) { \
            unsigned char packed = smem_Bp[(buf)][kp][my_n]; \
            smem_B_bf16[my_n][kp * 2]     = __float2bfloat16(smem_LUT[packed & 0xF] * sv1); \
            smem_B_bf16[my_n][kp * 2 + 1] = __float2bfloat16(smem_LUT[packed >> 4]  * sv1); \
        } \
    } while(0)

    // Both M-chunks (ch=0 rows 0..63, ch=1 rows 64..127); B reused.
    #define M128_COMPUTE(a_buf) do { \
        _Pragma("unroll") \
        for (int ch = 0; ch < 2; ch++) { \
            v8f* acc = ch ? acc1 : acc0; \
            unsigned int m_row = ch * M_TILE + warp_m_offset + (lane_id & 15); \
            _Pragma("unroll") \
            for (int h = 0; h < 2; h++) { \
                v16bf a; \
                _Pragma("unroll") \
                for (int i = 0; i < 16; i++) \
                    a[i] = (__bf16)(float)smem_A[(a_buf)][m_row][h * 16 + i]; \
                _Pragma("unroll") \
                for (int nb = 0; nb < 8; nb++) { \
                    unsigned int nc = nb * 16 + (lane_id & 15); \
                    v16bf b; \
                    _Pragma("unroll") \
                    for (int k = 0; k < 16; k++) \
                        b[k] = (__bf16)(float)smem_B_bf16[nc][h * 16 + k]; \
                    acc[nb] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc[nb]); \
                } \
            } \
        } \
    } while(0)

    M128_LOADS(0, 0);
    __syncthreads();
    M128_DEQUANT(0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        M128_LOADS(nxt, k_base);
        M128_COMPUTE(cur);
        __syncthreads();
        M128_DEQUANT(nxt);
        __syncthreads();
        cur = nxt;
    }
    M128_COMPUTE(cur);

    #undef M128_LOADS
    #undef M128_DEQUANT
    #undef M128_COMPUTE

    // Write chunk 0: rows [cta_m..cta_m+63]
    #pragma unroll
    for (int nb = 0; nb < 8; nb++)
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            unsigned int r = cta_m + warp_m_offset + 2 * e + (lane_id >> 4);
            unsigned int c = cta_n + nb * 16 + (lane_id & 15);
            if (r < M && c < N) C[r * N + c] = __float2bfloat16(acc0[nb][e]);
        }
    // Write chunk 1: rows [cta_m+64..cta_m+127]
    #pragma unroll
    for (int nb = 0; nb < 8; nb++)
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            unsigned int r = cta_m + M_TILE + warp_m_offset + 2 * e + (lane_id >> 4);
            unsigned int c = cta_n + nb * 16 + (lane_id & 15);
            if (r < M && c < N) C[r * N + c] = __float2bfloat16(acc1[nb][e]);
        }
}

// ═══════════════════════════════════════════════════════════════════
// fp8_gemm_t_m128: BF16 A × FP8 B, 2 M-chunks per CTA. Decode FP8→BF16 + WMMA.
// ═══════════════════════════════════════════════════════════════════
extern "C" __global__
__launch_bounds__(128, 3)
void fp8_gemm_t_m128(
    const __nv_bfloat16* __restrict__ A,       // [M, K] BF16
    const unsigned char* __restrict__ B_fp8,   // [N, K] FP8 E4M3
    __nv_bfloat16* __restrict__ C,             // [M, N] BF16
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * (2 * M_TILE);
    if (cta_m >= M) return;

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;

    __shared__ __nv_bfloat16 smem_A[2][2 * M_TILE][K_STEP_T + PAD_T];
    __shared__ unsigned char  smem_B[2][N_TILE_LG][K_STEP_T];

    v8f acc0[8], acc1[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        acc0[i] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
        acc1[i] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
    }


    #define FGM128_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 2; \
            unsigned int a_col = (threadIdx.x & 3) << 3; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 4; rnd++) { \
                unsigned int row = (unsigned int)(rnd * 32) + a_row_base; \
                unsigned int gr  = cta_m + row; \
                sync_copy_16(&smem_A[(buf)][row][a_col], \
                    &A[(unsigned long long)gr * K + gc], \
                    (gr < M) && (gc + 7 < K)); \
            } \
        } \
        { \
            unsigned int my_n = threadIdx.x; \
            unsigned int gn = cta_n + my_n; \
            bool valid = (gn < N) && ((kb) + 31 < K); \
            sync_copy_16(&smem_B[(buf)][my_n][0], \
                &B_fp8[(unsigned long long)gn * K + (kb)], valid); \
            sync_copy_16(&smem_B[(buf)][my_n][16], \
                &B_fp8[(unsigned long long)gn * K + (kb) + 16], valid); \
        } \
    } while(0)

    #define FGM128_COMPUTE(a_buf, b_buf) do { \
        _Pragma("unroll") \
        for (int ch = 0; ch < 2; ch++) { \
            v8f* acc = ch ? acc1 : acc0; \
            unsigned int m_row = ch * M_TILE + warp_m_offset + (lane_id & 15); \
            _Pragma("unroll") \
            for (int h = 0; h < 2; h++) { \
                v16bf a; \
                _Pragma("unroll") \
                for (int i = 0; i < 16; i++) \
                    a[i] = (__bf16)(float)smem_A[(a_buf)][m_row][h * 16 + i]; \
                _Pragma("unroll") \
                for (int nb = 0; nb < 8; nb++) { \
                    unsigned int nc = nb * 16 + (lane_id & 15); \
                    v16bf b; \
                    _Pragma("unroll") \
                    for (int k = 0; k < 16; k++) \
                        b[k] = (__bf16)(float)atlas_e4m3_to_f32(smem_B[(b_buf)][nc][h * 16 + k]); \
                    acc[nb] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc[nb]); \
                } \
            } \
        } \
    } while(0)

    FGM128_LOADS(0, 0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        FGM128_LOADS(nxt, k_base);
        FGM128_COMPUTE(cur, cur);
        __syncthreads();
        cur = nxt;
    }
    FGM128_COMPUTE(cur, cur);

    #undef FGM128_LOADS
    #undef FGM128_COMPUTE

    #pragma unroll
    for (int nb = 0; nb < 8; nb++)
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            unsigned int r = cta_m + warp_m_offset + 2 * e + (lane_id >> 4);
            unsigned int c = cta_n + nb * 16 + (lane_id & 15);
            if (r < M && c < N) C[r * N + c] = __float2bfloat16(acc0[nb][e]);
        }
    #pragma unroll
    for (int nb = 0; nb < 8; nb++)
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            unsigned int r = cta_m + M_TILE + warp_m_offset + 2 * e + (lane_id >> 4);
            unsigned int c = cta_n + nb * 16 + (lane_id & 15);
            if (r < M && c < N) C[r * N + c] = __float2bfloat16(acc1[nb][e]);
        }
}

// ═══════════════════════════════════════════════════════════════════
// fp8_fp8_gemm_t_m128: FP8 A × FP8 B, 2 M-chunks per CTA. Decode both → BF16 + WMMA.
// ═══════════════════════════════════════════════════════════════════
extern "C" __global__
__launch_bounds__(128, 3)
void fp8_fp8_gemm_t_m128(
    const unsigned char* __restrict__ A_fp8,  // [M, K] FP8 E4M3
    const unsigned char* __restrict__ B_fp8,  // [N, K] FP8 E4M3
    __nv_bfloat16* __restrict__ C,            // [M, N] BF16
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * (2 * M_TILE);
    if (cta_m >= M) return;

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;

    __shared__ unsigned char smem_Af[2][2 * M_TILE][A_FP8_STRIDE];
    __shared__ unsigned char smem_Bf[2][N_TILE_LG][K_STEP_T];

    v8f acc0[8], acc1[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        acc0[i] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
        acc1[i] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
    }

    #define FFM128_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 1; \
            unsigned int a_col = (threadIdx.x & 1) << 4; \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 2; rnd++) { \
                unsigned int row = (unsigned int)(rnd * 64) + a_row_base; \
                unsigned int gr  = cta_m + row; \
                sync_copy_16(&smem_Af[(buf)][row][a_col], \
                    &A_fp8[(unsigned long long)gr * K + gc], \
                    (gr < M) && (gc + 15 < K)); \
            } \
        } \
        { \
            unsigned int my_n = threadIdx.x; \
            unsigned int gn = cta_n + my_n; \
            bool valid = (gn < N) && ((kb) + 31 < K); \
            sync_copy_16(&smem_Bf[(buf)][my_n][0], \
                &B_fp8[(unsigned long long)gn * K + (kb)], valid); \
            sync_copy_16(&smem_Bf[(buf)][my_n][16], \
                &B_fp8[(unsigned long long)gn * K + (kb) + 16], valid); \
        } \
    } while(0)

    #define FFM128_COMPUTE(a_buf, b_buf) do { \
        _Pragma("unroll") \
        for (int ch = 0; ch < 2; ch++) { \
            v8f* acc = ch ? acc1 : acc0; \
            unsigned int m_row = ch * M_TILE + warp_m_offset + (lane_id & 15); \
            _Pragma("unroll") \
            for (int h = 0; h < 2; h++) { \
                v16bf a; \
                _Pragma("unroll") \
                for (int i = 0; i < 16; i++) \
                    a[i] = (__bf16)(float)atlas_e4m3_to_f32(smem_Af[(a_buf)][m_row][h * 16 + i]); \
                _Pragma("unroll") \
                for (int nb = 0; nb < 8; nb++) { \
                    unsigned int nc = nb * 16 + (lane_id & 15); \
                    v16bf b; \
                    _Pragma("unroll") \
                    for (int k = 0; k < 16; k++) \
                        b[k] = (__bf16)(float)atlas_e4m3_to_f32(smem_Bf[(b_buf)][nc][h * 16 + k]); \
                    acc[nb] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc[nb]); \
                } \
            } \
        } \
    } while(0)

    FFM128_LOADS(0, 0);
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        FFM128_LOADS(nxt, k_base);
        FFM128_COMPUTE(cur, cur);
        __syncthreads();
        cur = nxt;
    }
    FFM128_COMPUTE(cur, cur);

    #undef FFM128_LOADS
    #undef FFM128_COMPUTE

    #pragma unroll
    for (int nb = 0; nb < 8; nb++)
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            unsigned int r = cta_m + warp_m_offset + 2 * e + (lane_id >> 4);
            unsigned int c = cta_n + nb * 16 + (lane_id & 15);
            if (r < M && c < N) C[r * N + c] = __float2bfloat16(acc0[nb][e]);
        }
    #pragma unroll
    for (int nb = 0; nb < 8; nb++)
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            unsigned int r = cta_m + M_TILE + warp_m_offset + 2 * e + (lane_id >> 4);
            unsigned int c = cta_n + nb * 16 + (lane_id & 15);
            if (r < M && c < N) C[r * N + c] = __float2bfloat16(acc1[nb][e]);
        }
}
