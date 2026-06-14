// SPDX-License-Identifier: AGPL-3.0-only
//
// Atlas W8A8 + FP32 epilogue GEMM — vLLM-equivalent FP8 numerics.
//
//   C[M, N] = bf16(   Σ_g  ( Σ_(k in g)  A_fp8[m, k] * B_fp8[n, k] ) * a_scale[m, g] * b_scale[n/128, g] )
//
// where g iterates over K-groups of 128 elements (FP8_GROUP_K=128).
// Layout matches vLLM's `apply_w8a8_block_fp8_linear`:
//   - A_fp8[M, K]      — per-token-per-128 FP8 quant (from `per_token_group_quant_fp8`)
//   - a_scale[M, K/128] FP32 — output scale of `per_token_group_quant_fp8`
//   - B_fp8[N, K]      — per-(128, 128) FP8 weight (block-scaled checkpoint)
//   - b_scale[N/128, K/128] FP32 — block-scaled weight scale (scale_inv widened
//                                  to FP32 at load; applied in full FP32)
//   - C[M, N] BF16
//
// MMA: mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 (sm_121 native FP8)
// Two-level accum: inner_acc runs 4 MMAs over K=128, then folds into outer_acc
// with a_scale * b_scale applied in FP32. Matches DeepGEMM / vLLM block-FP8.
//
// Tile: 64 × 128 × 32 (M_TILE × N_TILE_LG × K_STEP_T)  — same as fp8_fp8_gemm_t
// Block: 128 threads (4 warps × 16 M-rows each)
// Grid: (ceil(N/128), ceil(M/64), 1)

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define M_TILE 64
#define N_TILE_LG 128
#define K_STEP_T 32
#define K_BLOCK 128
#define K_STEPS_PER_BLOCK (K_BLOCK / K_STEP_T) // 4
#define A_FP8_STRIDE 32

// cp.async helpers (SM80+) — copied byte-for-byte from
// kernels/gb10/qwen3.6-35b-a3b/nvfp4/w4a16_gemm.cu:152-165.
// Uses the proven `__cvta_generic_to_shared` intrinsic + byte-count
// predication (src_bytes=0 makes cp.async a no-op). The earlier
// hand-rolled `cvta.to.shared.u64` with `@p` predication and a broken
// uint32←uint64 cast was the cause of CUDA_ERROR_ILLEGAL_ADDRESS in
// the first iteration of this kernel.
__device__ __forceinline__ void cp_async_pred_16(void* dst_smem, const void* src_gmem, bool pred) {
    unsigned int dst = __cvta_generic_to_shared(dst_smem);
    unsigned int src_bytes = pred ? 16 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
                 :: "r"(dst), "l"(src_gmem), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}
__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;");
}

// --- gfx1151/SCALE FP8-MMA software path (SSOT: w4a16_gemm.cu:24-132) ---
// SCALE/gfx1151 has no codegen for mma.sync.m16n8k32.e4m3. Replace it with two
// m16n8k16.bf16 MMAs (K split 0..15 / 16..31), decoding each E4M3 byte via the
// standard scl_fp8 (SCALE's __NV_E4M3 cast is non-standard). The #else is the
// verbatim e4m3 PTX, so NVIDIA codegen is byte-identical. Helpers duplicated
// per-module (same convention as scl_fp8 across the decode GEMVs / E4M3_LUT).
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ float scl_fp8(unsigned char b) {
    unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u; float v;
    if (e == 0u)               v = (float)m * 0.001953125f;            // subnormal m*2^-9
    else if (e == 15u && m == 7u) v = 0.0f;                            // NaN -> 0
    else                       v = __uint_as_float(((e + 120u) << 23) | (m << 20)); // 2^(e-7)*(1+m/8)
    return s ? -v : v;
}
#endif
#if defined(__SCALE__)
__device__ __forceinline__ float atlas_e4m3_to_f32(unsigned char b) {
    return scl_fp8(b);  // standard E4M3, matches per_token_group_quant_fp8 scl_enc_fp8
}
__device__ __forceinline__ unsigned atlas_bf2(float lo, float hi) {
    unsigned short l = __bfloat16_as_ushort(__float2bfloat16(lo));
    unsigned short h = __bfloat16_as_ushort(__float2bfloat16(hi));
    return ((unsigned)h << 16) | l;
}
#endif
__device__ __forceinline__ void atlas_mma_e4m3(float* acc,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3,
    unsigned b0, unsigned b1) {
#if defined(__SCALE__)
    unsigned lane = threadIdx.x & 31u, tig = lane & 3u, base = lane & ~3u;
    #pragma unroll
    for (int half = 0; half < 2; half++) {
        unsigned A_g = half ? a2 : a0, A_g8 = half ? a3 : a1, B_g = half ? b1 : b0;
        #define ATLAS_GA(reg, j) atlas_e4m3_to_f32((unsigned char)( \
            __shfl_sync(0xffffffffu, (reg), base + ((unsigned)(j) >> 2)) \
            >> (8 * ((j) & 3))))
        int j0 = 2 * (int)tig, j1 = 8 + 2 * (int)tig;
        unsigned A0 = atlas_bf2(ATLAS_GA(A_g, j0),  ATLAS_GA(A_g, j0 + 1));
        unsigned A1 = atlas_bf2(ATLAS_GA(A_g8, j0), ATLAS_GA(A_g8, j0 + 1));
        unsigned A2 = atlas_bf2(ATLAS_GA(A_g, j1),  ATLAS_GA(A_g, j1 + 1));
        unsigned A3 = atlas_bf2(ATLAS_GA(A_g8, j1), ATLAS_GA(A_g8, j1 + 1));
        unsigned B0 = atlas_bf2(ATLAS_GA(B_g, j0),  ATLAS_GA(B_g, j0 + 1));
        unsigned B1 = atlas_bf2(ATLAS_GA(B_g, j1),  ATLAS_GA(B_g, j1 + 1));
        #undef ATLAS_GA
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
            : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
            : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(B0), "r"(B1),
              "f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3]));
    }
#else
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
        : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3]));
#endif
}

extern "C" __global__ void fp8_gemm_t_blockscaled(
    const unsigned char* __restrict__ A_fp8,    // [M, K] FP8 E4M3
    const float* __restrict__ a_scale,          // [M, K/128] FP32
    const unsigned char* __restrict__ B_fp8,    // [N, K] FP8 E4M3
    const float* __restrict__ b_scale,          // [N/128, K/128] FP32
    __nv_bfloat16* __restrict__ C,              // [M, N] BF16
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * M_TILE;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ unsigned char smem_Af[2][M_TILE][A_FP8_STRIDE];
    __shared__ unsigned char smem_Bf[2][N_TILE_LG][K_STEP_T];

    // Two-level FP32 accumulation: inner_acc runs unscaled FP8×FP8 → FP32
    // across 4 MMAs (K=128). At each K-block boundary, scale inner_acc
    // by (a_scale[m_warp_row, k_block] × b_scale[n_block, k_block]) and
    // add to outer_acc. Reset inner_acc, advance.
    float inner_acc[16][4];
    float outer_acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        inner_acc[i][0] = 0.0f; inner_acc[i][1] = 0.0f;
        inner_acc[i][2] = 0.0f; inner_acc[i][3] = 0.0f;
        outer_acc[i][0] = 0.0f; outer_acc[i][1] = 0.0f;
        outer_acc[i][2] = 0.0f; outer_acc[i][3] = 0.0f;
    }

    // Per-CTA constants for scale indexing
    const unsigned int k_groups = K / K_BLOCK;
    // n_block constant per CTA (N_TILE_LG=128 == K_BLOCK; cta_n is 128-aligned
    // assuming N divisible by 128).
    const unsigned int n_block_lo = cta_n / K_BLOCK;
    // Within a 128-N tile, all N values share n_block_lo since cta_n is 128-aligned.

    // FP8 loads (both A and B as FP8 bytes) — mirrors fp8_fp8_gemm_t FF_LOADS
    #define FFB_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 1; \
            unsigned int a_col = (threadIdx.x & 1) << 4; \
            unsigned int gc = (kb) + a_col; \
            unsigned int row = a_row_base; \
            unsigned int gr = cta_m + row; \
            cp_async_pred_16(&smem_Af[(buf)][row][a_col], \
                &A_fp8[(unsigned long long)gr * K + gc], \
                (gr < M) && (gc + 15 < K)); \
        } \
        { \
            unsigned int my_n = threadIdx.x; \
            unsigned int gn = cta_n + my_n; \
            bool valid = (gn < N) && ((kb) + 31 < K); \
            cp_async_pred_16(&smem_Bf[(buf)][my_n][0], \
                &B_fp8[(unsigned long long)gn * K + (kb)], valid); \
            cp_async_pred_16(&smem_Bf[(buf)][my_n][16], \
                &B_fp8[(unsigned long long)gn * K + (kb) + 16], valid); \
        } \
    } while(0)

    // FP8×FP8 MMA — accumulates into inner_acc (NOT outer_acc directly).
    #define FFB_COMPUTE(a_buf, b_buf) do { \
        unsigned int fr0 = warp_m_offset + group_id, fr1 = fr0 + 8; \
        unsigned int a0 = *(const unsigned int*)&smem_Af[(a_buf)][fr0][4 * tid]; \
        unsigned int a1 = *(const unsigned int*)&smem_Af[(a_buf)][fr1][4 * tid]; \
        unsigned int a2 = *(const unsigned int*)&smem_Af[(a_buf)][fr0][16 + 4 * tid]; \
        unsigned int a3 = *(const unsigned int*)&smem_Af[(a_buf)][fr1][16 + 4 * tid]; \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_Bf[(b_buf)][nc][4 * tid]; \
            unsigned int b1 = *(const unsigned int*)&smem_Bf[(b_buf)][nc][16 + 4 * tid]; \
            atlas_mma_e4m3(inner_acc[nt], a0, a1, a2, a3, b0, b1); \
        } \
    } while(0)

    // Each warp's 16 N-tiles span N=[cta_n .. cta_n+127], i.e. ONE n_block
    // (n_block_lo). Each MMA tile's M rows = [warp_m_offset+group_id,
    // +group_id+8] — different m rows have different a_scale[m, k_block]
    // values. So we need per-row scale handling in the fold.
    // For the 4 acc[nt][i] outputs: i=0,1 are r0=warp_m_offset+group_id;
    // i=2,3 are r1=r0+8. We track both row scales.

    // Prolog
    FFB_LOADS(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    int cur = 0;
    unsigned int k_step_in_block = 0;
    for (unsigned int k_base = K_STEP_T; k_base < K; k_base += K_STEP_T) {
        int nxt = 1 - cur;
        FFB_LOADS(nxt, k_base);
        cp_async_commit();
        FFB_COMPUTE(cur, cur);
        cp_async_wait_all();
        __syncthreads();
        cur = nxt;

        k_step_in_block++;
        if (k_step_in_block == K_STEPS_PER_BLOCK) {
            const unsigned int k_block = (k_base - K_STEP_T) / K_BLOCK;
            const float bs = b_scale[n_block_lo * k_groups + k_block];
            const unsigned int r0_global = cta_m + warp_m_offset + group_id;
            const unsigned int r1_global = r0_global + 8;
            const float as0 = (r0_global < M) ? a_scale[r0_global * k_groups + k_block] : 0.0f;
            const float as1 = (r1_global < M) ? a_scale[r1_global * k_groups + k_block] : 0.0f;
            const float s0 = as0 * bs;
            const float s1 = as1 * bs;
            #pragma unroll
            for (int nt = 0; nt < 16; nt++) {
                outer_acc[nt][0] += inner_acc[nt][0] * s0;
                outer_acc[nt][1] += inner_acc[nt][1] * s0;
                outer_acc[nt][2] += inner_acc[nt][2] * s1;
                outer_acc[nt][3] += inner_acc[nt][3] * s1;
                inner_acc[nt][0] = 0.0f; inner_acc[nt][1] = 0.0f;
                inner_acc[nt][2] = 0.0f; inner_acc[nt][3] = 0.0f;
            }
            k_step_in_block = 0;
        }
    }
    // Last K_STEP outside the loop
    FFB_COMPUTE(cur, cur);
    k_step_in_block++;

    // Final fold (full block at the trailing edge).
    {
        const unsigned int k_block = (K - 1) / K_BLOCK;
        const float bs = b_scale[n_block_lo * k_groups + k_block];
        const unsigned int r0_global = cta_m + warp_m_offset + group_id;
        const unsigned int r1_global = r0_global + 8;
        const float as0 = (r0_global < M) ? a_scale[r0_global * k_groups + k_block] : 0.0f;
        const float as1 = (r1_global < M) ? a_scale[r1_global * k_groups + k_block] : 0.0f;
        const float s0 = as0 * bs;
        const float s1 = as1 * bs;
        #pragma unroll
        for (int nt = 0; nt < 16; nt++) {
            outer_acc[nt][0] += inner_acc[nt][0] * s0;
            outer_acc[nt][1] += inner_acc[nt][1] * s0;
            outer_acc[nt][2] += inner_acc[nt][2] * s1;
            outer_acc[nt][3] += inner_acc[nt][3] * s1;
        }
    }

    #undef FFB_LOADS
    #undef FFB_COMPUTE

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[r0 * N + c0] = __float2bfloat16(outer_acc[nt][0]);
        if (r0 < M && c1 < N) C[r0 * N + c1] = __float2bfloat16(outer_acc[nt][1]);
        if (r1 < M && c0 < N) C[r1 * N + c0] = __float2bfloat16(outer_acc[nt][2]);
        if (r1 < M && c1 < N) C[r1 * N + c1] = __float2bfloat16(outer_acc[nt][3]);
    }
}
