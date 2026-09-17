// SPDX-License-Identifier: AGPL-3.0-only

// Atlas W8A16 Transposed GEMM — M128 (2-chunk) fast prefill, FP8 E4M3 block-scaled.
//
// C[M,N] = A[M,K] (BF16) * dequant(B_t[K,N] (FP8 E4M3, transposed at load time))
//
// This is the FP8 analog of the NVFP4 `w4a16_gemm_t_m128_v2` (8-warp / 256-thread,
// parallel-chunk) fast-prefill template, but with two precision-correct changes:
//
//   1. Weights are FP8 E4M3 (1 byte/weight, full K) decoded LOSSLESSLY through the
//      256-entry E4M3 LUT (E4M3's 3 mantissa bits ⊂ BF16's 7) into MMA-ready
//      K-contiguous smem_B[n][k]. BF16 activations are consumed NATIVELY — they are
//      NOT requantized to E4M3 (which is the NVFP4 v2 template's lossy step).
//   2. The MMA is `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` (matching
//      `w8a16_gemm_t` / `w8a16_gemm_t_pipelined`), and the block scale is folded
//      onto a FP32 OUTER accumulator ONCE per 128-K block (two-level FP32
//      accumulation), NOT multiplied per-element. This is bit-identical in
//      K-order to `w8a16_gemm_t` and preserves the deep-layer FP8 floor.
//
// Geometry (mirrors w4a16_gemm_t_m128_v2): 128 (M) × 128 (N) CTA tile, two 64-row
// chunks, K_STEP=32 (2 m16n8k16 sub-MMAs per resident step). 8 warps: warps 0-3 own
// chunk-0 rows, warps 4-7 own chunk-1 rows → both chunks' MMAs run in parallel.
//
// Transposed-specific twist (from w8a16_gemm_t_pipelined): B_t[K,N] is N-contiguous,
// but the MMA fragment wants K-contiguous. Load N-contiguous 16-byte cp.async chunks
// into a raw [k][n] smem buffer (COALESCED), then TRANSPOSE-on-dequant into the
// MMA-ready K-contiguous smem_B[n][k]. Coalesced global loads AND fast 32-bit MMA
// fragment loads AND a shared-memory LUT, all at once.
//
// Grid: (ceil(N/128), ceil(M/128), 1)  Block: (256, 1, 1)

#include <cuda_bf16.h>

#include "e4m3_lut.cuh"   // shared E4M3_LUT SSOT (staged into smem — Lever 1)

#define WM128_M_TILE   64                 // half-tile (one chunk of 64 M-rows)
#define WM128_N_TILE   128                // full N width per CTA
#define WM128_K_STEP   32                 // 2 m16n8k16 sub-MMAs per resident step
#define WM128_K_SUB    16                 // one m16n8k16's K-width
#define WM128_K_SUBS   (WM128_K_STEP / WM128_K_SUB)   // = 2
#define WM128_PAD      8                  // A-tile row pad: (32+8)*2=80 B, 16-aligned
#define WM128_BPAD     2                  // smem_B K-pad (breaks bank conflicts)
#define WM128_FP8_BLOCK 128
#define WM128_WARPS    8
#define WM128_THREADS  (WM128_WARPS * 32) // 256

// cp.async.cg 16-byte (cache-global) copy: smem <- global. sm_80+; correct on sm_121.
__device__ __forceinline__ void wm128_cp_async_cg_16(void* smem_ptr, const void* gmem_ptr) {
    unsigned int s = (unsigned int)__cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem_ptr));
}
__device__ __forceinline__ void wm128_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
__device__ __forceinline__ void wm128_cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::);
}

/// W8A16 transposed M128 GEMM: B_t[K,N] N-contiguous FP8 E4M3 + 2D transposed
/// block scales. 128×128 (M×N) tile, two 64-row chunks, 8 warps, parallel-chunk
/// MMA, K_STEP=32, shared-memory LUT, transpose-on-dequant, two-level FP32 fold.
extern "C" __global__
__launch_bounds__(256, 2)
void w8a16_gemm_t_m128(
    const __nv_bfloat16* __restrict__ A,            // [M, K] BF16 activations
    const unsigned char* __restrict__ B_t,           // [K, N] FP8 E4M3 transposed
    const float* __restrict__ block_scale_t,         // [K/128, N/128] FP32 transposed
    __nv_bfloat16* __restrict__ C,                   // [M, N] BF16 output
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * WM128_N_TILE;
    const unsigned int cta_m = blockIdx.y * (2 * WM128_M_TILE);   // 128 rows
    if (cta_m >= M) return;

    const unsigned int warp_id = threadIdx.x >> 5;        // 0..7
    const unsigned int lane_id = threadIdx.x & 31;
    const unsigned int chunk   = warp_id >> 2;            // 0 or 1 (which 64-row half)
    const unsigned int sub     = warp_id & 3;             // 0..3 within chunk
    const unsigned int warp_m_offset = sub * 16;          // M-row offset within chunk
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    // Double-buffered smem. smem_A / smem_Braw are cp.async destinations → 16-byte
    // aligned rows. smem_B is the MMA-ready K-contiguous dequantized buffer.
    //   Per stage: smem_A 128*40*2 = 10240 B + smem_Braw 32*128 = 4096 B +
    //              smem_B 128*34*2 = 8704 B = 23040 B → 46080 B for 2 stages
    //              + 1 KB LUT ≈ 47 KB.
    __shared__ __align__(16) __nv_bfloat16 smem_A[2][2 * WM128_M_TILE][WM128_K_STEP + WM128_PAD];
    __shared__ __align__(16) unsigned char smem_Braw[2][WM128_K_STEP][WM128_N_TILE];
    __shared__ __nv_bfloat16 smem_B[2][WM128_N_TILE][WM128_K_STEP + WM128_BPAD];

    // ── Lever 1: stage the E4M3→FP32 LUT in SHARED memory (SSOT = E4M3_LUT) ──
    __shared__ float smem_lut[256];
    smem_lut[threadIdx.x] = E4M3_LUT[threadIdx.x];   // THREADS == 256, exact cover
    __syncthreads();

    // Two-level FP32 accumulation. Each warp owns 16 N-tiles × its 16 M-rows.
    // inner accumulates UNSCALED BF16-cast E4M3 across the K-steps of one 128-K
    // block; at each block boundary outer += inner * block_scale_t.
    float inner_acc[16][4];
    float outer_acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        inner_acc[i][0] = 0.f; inner_acc[i][1] = 0.f;
        inner_acc[i][2] = 0.f; inner_acc[i][3] = 0.f;
        outer_acc[i][0] = 0.f; outer_acc[i][1] = 0.f;
        outer_acc[i][2] = 0.f; outer_acc[i][3] = 0.f;
    }

    const unsigned int a_stride = WM128_K_STEP + WM128_PAD;     // 40
    const unsigned int b_stride = WM128_K_STEP + WM128_BPAD;    // 34
    const unsigned int n_scale_blocks = (N + WM128_FP8_BLOCK - 1) / WM128_FP8_BLOCK;
    const unsigned int k_steps_per_block = WM128_FP8_BLOCK / WM128_K_STEP;   // 4
    const unsigned int n_block = cta_n / WM128_FP8_BLOCK;       // constant per CTA
    const unsigned int n_steps = (K + WM128_K_STEP - 1) / WM128_K_STEP;

    // ── A loads: 128 rows × K_STEP K-cols BF16, contiguous along K (16-B chunks) ──
    //   256 threads × 16 B = 4096 B; 128*32*2 = 8192 B/tile → 2 rounds.
    #define WM128_LOADS(buf, kb) do { \
        { \
            unsigned int a_row_base = threadIdx.x >> 2;       /* 0..63 */ \
            unsigned int a_col      = (threadIdx.x & 3) << 3; /* 0/8/16/24 */ \
            unsigned int gc = (kb) + a_col; \
            _Pragma("unroll") \
            for (int rnd = 0; rnd < 2; rnd++) { \
                unsigned int row = (unsigned int)(rnd * 64) + a_row_base; \
                unsigned int gr  = cta_m + row; \
                __nv_bfloat16* dst = &smem_A[(buf)][row][a_col]; \
                /* 16-B cp.async needs a 16-B-aligned src: A[gr*K+gc] is aligned */ \
                /* iff K%8==0 (gc is already %8). Uniform per-CTA branch — a */ \
                /* misaligned K falls to the scalar path (cf. vision-716 fix). */ \
                if ((gr < M) && (gc + 7 < K) && ((K & 7) == 0)) { \
                    wm128_cp_async_cg_16(dst, &A[(unsigned long long)gr * K + gc]); \
                } else { \
                    _Pragma("unroll") \
                    for (int e = 0; e < 8; e++) { \
                        unsigned int gcol = gc + e; \
                        dst[e] = (gr < M && gcol < K) \
                            ? A[(unsigned long long)gr * K + gcol] : __float2bfloat16(0.0f); \
                    } \
                } \
            } \
        } \
        { \
            /* B_t raw: COALESCED 16-B (16 FP8-byte) chunks of N per K-row into */ \
            /* smem_Braw[k][n] mirroring global B_t[kb+k, n] contiguously.       */ \
            /* K_STEP*N_TILE/16 = 32*128/16 = 256 chunks → exactly 1/thread.     */ \
            unsigned int c = threadIdx.x; \
            unsigned int krow = (c * 16) / WM128_N_TILE;     /* 0..K_STEP-1 */ \
            unsigned int ncol = (c * 16) % WM128_N_TILE;     /* 0/16/.../112 */ \
            unsigned int gk = (kb) + krow; \
            unsigned int gn = cta_n + ncol; \
            unsigned char* dst = &smem_Braw[(buf)][krow][ncol]; \
            /* 16-B cp.async needs 16-B-aligned src: B_t[gk*N+gn] aligned iff */ \
            /* N%16==0 (gn is already %16). Uniform branch — misaligned N uses scalar. */ \
            if (gk < K && gn + 15 < N && ((N & 15) == 0)) { \
                wm128_cp_async_cg_16(dst, &B_t[(unsigned long long)gk * N + gn]); \
            } else { \
                _Pragma("unroll") \
                for (int e = 0; e < 16; e++) { \
                    unsigned int gne = gn + e; \
                    dst[e] = (gk < K && gne < N) ? B_t[(unsigned long long)gk * N + gne] : 0; \
                } \
            } \
        } \
    } while(0)

    // ── Dequant + TRANSPOSE buf's raw FP8 B into MMA-ready K-contiguous smem_B ──
    // Read N-contiguous smem_Braw[k][n], write K-contiguous smem_B[n][k]. No scale
    // (folded on the FP32 accumulator at the block boundary). K_STEP*N_TILE = 4096
    // elements, 256 threads → 16/thread.
    #define WM128_DEQUANT(buf) do { \
        _Pragma("unroll") \
        for (unsigned int idx = threadIdx.x; idx < WM128_K_STEP * WM128_N_TILE; idx += WM128_THREADS) { \
            unsigned int k = idx / WM128_N_TILE;     /* 0..K_STEP-1 */ \
            unsigned int n = idx % WM128_N_TILE;     /* 0..N_TILE-1 */ \
            unsigned char wb = smem_Braw[(buf)][k][n]; \
            smem_B[(buf)][n][k] = __float2bfloat16(smem_lut[wb]); \
        } \
    } while(0)

    // ── Compute: each warp does WM128_K_SUBS×16 MMAs against its owned M-rows. ──
    // Chunk-0 warps (0-3) and chunk-1 warps (4-7) run in parallel.
    #define WM128_COMPUTE(buf) do { \
        const unsigned short* sA = (const unsigned short*)smem_A[(buf)]; \
        const unsigned short* sB = (const unsigned short*)smem_B[(buf)]; \
        _Pragma("unroll") \
        for (int s = 0; s < WM128_K_SUBS; s++) { \
            unsigned int k_off = s * WM128_K_SUB; \
            unsigned int fr0 = chunk * WM128_M_TILE + warp_m_offset + group_id; \
            unsigned int fr1 = fr0 + 8; \
            unsigned int fc0 = k_off + tid * 2; \
            unsigned int fc1 = k_off + tid * 2 + 8; \
            unsigned int a0 = *(const unsigned int*)&sA[fr0 * a_stride + fc0]; \
            unsigned int a1 = *(const unsigned int*)&sA[fr1 * a_stride + fc0]; \
            unsigned int a2 = *(const unsigned int*)&sA[fr0 * a_stride + fc1]; \
            unsigned int a3 = *(const unsigned int*)&sA[fr1 * a_stride + fc1]; \
            _Pragma("unroll") \
            for (int nt = 0; nt < 16; nt++) { \
                unsigned int nc = nt * 8 + group_id; \
                unsigned int k0 = k_off + tid * 2; \
                unsigned int k1 = k_off + tid * 2 + 8; \
                unsigned int b0 = *(const unsigned int*)&sB[nc * b_stride + k0]; \
                unsigned int b1 = *(const unsigned int*)&sB[nc * b_stride + k1]; \
                asm volatile( \
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};" \
                    :"=f"(inner_acc[nt][0]),"=f"(inner_acc[nt][1]), \
                     "=f"(inner_acc[nt][2]),"=f"(inner_acc[nt][3]) \
                    :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1), \
                     "f"(inner_acc[nt][0]),"f"(inner_acc[nt][1]), \
                     "f"(inner_acc[nt][2]),"f"(inner_acc[nt][3])); \
            } \
        } \
    } while(0)

    // Fold the scaled inner accumulator into the outer and reset inner.
    #define WM128_FOLD(scale_val) do { \
        float _sc = (scale_val); \
        _Pragma("unroll") \
        for (int i = 0; i < 16; i++) { \
            outer_acc[i][0] += inner_acc[i][0] * _sc; \
            outer_acc[i][1] += inner_acc[i][1] * _sc; \
            outer_acc[i][2] += inner_acc[i][2] * _sc; \
            outer_acc[i][3] += inner_acc[i][3] * _sc; \
            inner_acc[i][0] = 0.f; inner_acc[i][1] = 0.f; \
            inner_acc[i][2] = 0.f; inner_acc[i][3] = 0.f; \
        } \
    } while(0)

    // ── 2-stage software pipeline (same structure as w4a16_gemm_t_m128_v2). ──
    WM128_LOADS(0, 0);
    wm128_cp_async_commit();
    wm128_cp_async_wait_all();
    __syncthreads();
    WM128_DEQUANT(0);
    __syncthreads();

    unsigned int k_step_in_block = 0;
    int cur = 0;
    for (unsigned int step = 1; step < n_steps; step++) {
        unsigned int k_base = step * WM128_K_STEP;
        int nxt = 1 - cur;
        WM128_LOADS(nxt, k_base);
        wm128_cp_async_commit();
        WM128_COMPUTE(cur);

        // K_BLOCK boundary for the step we just computed (step-1).
        k_step_in_block++;
        if (k_step_in_block == k_steps_per_block) {
            const unsigned int k_block = ((step - 1) * WM128_K_STEP) / WM128_FP8_BLOCK;
            WM128_FOLD(block_scale_t[k_block * n_scale_blocks + n_block]);
            k_step_in_block = 0;
        }

        wm128_cp_async_wait_all();
        __syncthreads();
        WM128_DEQUANT(nxt);
        __syncthreads();
        cur = nxt;
    }
    // Final compute for the last resident step.
    WM128_COMPUTE(cur);
    k_step_in_block++;
    if (k_step_in_block == k_steps_per_block) {
        const unsigned int k_block = ((n_steps - 1) * WM128_K_STEP) / WM128_FP8_BLOCK;
        WM128_FOLD(block_scale_t[k_block * n_scale_blocks + n_block]);
        k_step_in_block = 0;
    } else if (k_step_in_block != 0) {
        // Incomplete trailing K_BLOCK (K % FP8_BLOCK != 0).
        const unsigned int k_block = (K - 1) / WM128_FP8_BLOCK;
        WM128_FOLD(block_scale_t[k_block * n_scale_blocks + n_block]);
    }

    #undef WM128_LOADS
    #undef WM128_DEQUANT
    #undef WM128_COMPUTE
    #undef WM128_FOLD

    // ── Epilogue: each warp writes its own 16 M-rows × 128 N-cols (no shuffle). ──
    const unsigned int row_base = cta_m + chunk * WM128_M_TILE + warp_m_offset;
    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = row_base + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[(unsigned long long)r0 * N + c0] = __float2bfloat16(outer_acc[nt][0]);
        if (r0 < M && c1 < N) C[(unsigned long long)r0 * N + c1] = __float2bfloat16(outer_acc[nt][1]);
        if (r1 < M && c0 < N) C[(unsigned long long)r1 * N + c0] = __float2bfloat16(outer_acc[nt][2]);
        if (r1 < M && c1 < N) C[(unsigned long long)r1 * N + c1] = __float2bfloat16(outer_acc[nt][3]);
    }
}
