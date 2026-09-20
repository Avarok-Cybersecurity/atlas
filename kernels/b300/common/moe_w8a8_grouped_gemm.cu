// SPDX-License-Identifier: AGPL-3.0-only
//
// Atlas W8A8 + FP32 epilogue MoE Grouped GEMM — vLLM-equivalent numerics.
//
// Same shape/layout as `moe_fp8_grouped_gemm.cu` but:
//   - A is FP8 E4M3 (one byte per element), pre-quantized per-token-per-128
//     via `per_token_group_quant_fp8`. Dequanted to BF16 in smem via exact
//     bit arithmetic (lossless: FP8 3-bit mantissa fits in BF16 7-bit).
//   - a_scale[M_total, K/128] FP32 — looked up via sorted_token_ids[m_start + m_idx].
//   - b_scale[N/128, K/128] FP32 — scale_inv widened to FP32 at load (read once per fold).
//   - Two-level FP32 accumulation: inner_acc over K=128 block (4× K_STEP=16),
//     then outer_acc += inner_acc × (a_scale[row, kb] × b_scale[col, kb]).
//
//   C[M_expert, N] = bf16( Σ_kb ( Σ_(k∈kb) bf16(e4m3(A_fp8[m,k])) * bf16(e4m3(B_fp8[n,k])) )
//                       * a_scale[orig_token(m), kb] * b_scale[n/128, kb] )
//
// Grid: (ceil(N/64), max_m_tiles, num_experts)  Block: (128, 1, 1)

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#define M_TILE 64
#define N_TILE 64
#define K_STEP 16
#define PAD 2
#define FP8_BLOCK 128
#define K_PROMOTE 64

// Exact FP8 E4M3 -> BF16 dequant via bit arithmetic (replaces a 256-entry
// __constant__ LUT: 32 lanes indexing 32 different __constant__ addresses
// serialize the warp; this is pure ALU and issues at full width).
// An e4m3 magnitude placed in a half's bit layout ((b&0x7f)<<7) has bias 15
// instead of 7 and the same mantissa alignment, so value = half(bits) * 2^8.
// Denormals map to half denormals with the same 2^8 offset — exact for all
// codes. The NaN codes 0x7F/0xFF must be guarded to +/-0.0f to match the
// historical LUT mapping (otherwise they dequant to +/-480).
__device__ __forceinline__ __nv_bfloat16 e4m3_to_bf16_w8a8(unsigned char b) {
    unsigned short h = (unsigned short)((b & 0x7f) << 7);
    float f = __half2float(__ushort_as_half(h)) * 256.0f;
    if ((b & 0x7f) == 0x7f) f = 0.0f;
    f = (b & 0x80) ? -f : f;
    return __float2bfloat16(f);
}

__device__ __forceinline__ void fp8_w8a8_mma(
    __nv_bfloat16 smem_A[][K_STEP + PAD],
    __nv_bfloat16 smem_B[][N_TILE + PAD],
    float acc[8][4],
    unsigned int warp_m_offset, unsigned int group_id, unsigned int tid
) {
    const unsigned int a_stride = K_STEP + PAD;
    const unsigned int b_stride = N_TILE + PAD;
    const unsigned short* sA = (const unsigned short*)smem_A;
    const unsigned short* sB = (const unsigned short*)smem_B;

    unsigned int frag_r0 = warp_m_offset + group_id;
    unsigned int frag_r1 = warp_m_offset + group_id + 8;
    unsigned int frag_c0 = tid * 2;
    unsigned int frag_c1 = tid * 2 + 8;

    unsigned int a0 = *(const unsigned int*)&sA[frag_r0 * a_stride + frag_c0];
    unsigned int a1 = *(const unsigned int*)&sA[frag_r1 * a_stride + frag_c0];
    unsigned int a2 = *(const unsigned int*)&sA[frag_r0 * a_stride + frag_c1];
    unsigned int a3 = *(const unsigned int*)&sA[frag_r1 * a_stride + frag_c1];

    #pragma unroll
    for (int n_tile = 0; n_tile < 8; n_tile++) {
        unsigned int n_col = n_tile * 8 + group_id;
        unsigned int k0 = tid * 2;
        unsigned int k1 = tid * 2 + 8;

        unsigned int b0 = ((unsigned int)sB[(k0 + 1) * b_stride + n_col] << 16) |
                          (unsigned int)sB[k0 * b_stride + n_col];
        unsigned int b1 = ((unsigned int)sB[(k1 + 1) * b_stride + n_col] << 16) |
                          (unsigned int)sB[k1 * b_stride + n_col];

        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%10, %11, %12, %13};"
            : "=f"(acc[n_tile][0]), "=f"(acc[n_tile][1]),
              "=f"(acc[n_tile][2]), "=f"(acc[n_tile][3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1),
              "f"(acc[n_tile][0]), "f"(acc[n_tile][1]),
              "f"(acc[n_tile][2]), "f"(acc[n_tile][3])
        );
    }
}

extern "C" __global__ void moe_w8a8_grouped_gemm(
    const unsigned char* __restrict__ A_fp8,                // [total_tokens, K] FP8 E4M3
    const float* __restrict__ a_scale,                      // [total_tokens, K/128] FP32
    const unsigned long long* __restrict__ B_weight_ptrs,   // [num_experts] → [N, K] FP8
    const unsigned long long* __restrict__ B_scale_ptrs,    // [num_experts] → [N/128, K/128] FP32
    __nv_bfloat16* __restrict__ C,                          // [total_expanded, N] BF16
    const int* __restrict__ expert_offsets,                 // [num_experts + 1]
    const int* __restrict__ sorted_token_ids,               // [total_expanded] or NULL
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

    const unsigned int cta_n = blockIdx.x * N_TILE;

    const unsigned char* B_exp = (const unsigned char*)B_weight_ptrs[expert_id];
    const float* S_exp = (const float*)B_scale_ptrs[expert_id];
    if (B_exp == 0) return;

    const unsigned int k_blocks = (K + FP8_BLOCK - 1) / FP8_BLOCK;

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    __shared__ __nv_bfloat16 smem_A[M_TILE][K_STEP + PAD];
    __shared__ __nv_bfloat16 smem_B[K_STEP][N_TILE + PAD];
    // Per-warp cache of the original-token-id for each of its M-rows (group_id 0..7
    // → 8 row indices per warp). Stored once per CTA to avoid re-resolving every
    // K_STEP. We need rows [warp_m_offset .. warp_m_offset+15] for the warp's
    // two fragments (r0_global = warp_m_offset+group_id, r1 = +8).
    // CTA covers M_TILE=64 rows total — store all 64 here.
    __shared__ int smem_token_id[M_TILE];
    if (threadIdx.x < M_TILE) {
        int m_idx = threadIdx.x;
        if (m_idx + cta_m_local < M_expert) {
            int sorted_idx = m_start + cta_m_local + m_idx;
            smem_token_id[m_idx] = sorted_token_ids ? sorted_token_ids[sorted_idx] : sorted_idx;
        } else {
            smem_token_id[m_idx] = -1;
        }
    }
    __syncthreads();

    float outer_acc[8][4];
    float inner_acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        outer_acc[i][0] = 0.0f; outer_acc[i][1] = 0.0f;
        outer_acc[i][2] = 0.0f; outer_acc[i][3] = 0.0f;
        inner_acc[i][0] = 0.0f; inner_acc[i][1] = 0.0f;
        inner_acc[i][2] = 0.0f; inner_acc[i][3] = 0.0f;
    }

    const unsigned int n_block = cta_n / FP8_BLOCK;

    for (unsigned int k_base = 0; k_base < K; k_base += K_STEP) {
        // Load A tile: gather FP8 from sorted token positions, dequant to BF16 (lossless).
        {
            #pragma unroll
            for (unsigned int i = 0; i < 8; i++) {
                unsigned int idx = threadIdx.x * 8 + i;
                unsigned int row = idx / K_STEP;
                unsigned int col = idx % K_STEP;
                unsigned int m_idx = cta_m_local + row;
                unsigned int gc = k_base + col;

                if (m_idx < (unsigned int)M_expert && gc < K) {
                    int token_id = smem_token_id[row];
                    if (token_id >= 0) {
                        unsigned char a_byte = A_fp8[(unsigned long long)token_id * K + gc];
                        smem_A[row][col] = e4m3_to_bf16_w8a8(a_byte);
                    } else {
                        smem_A[row][col] = __float2bfloat16(0.0f);
                    }
                } else {
                    smem_A[row][col] = __float2bfloat16(0.0f);
                }
            }
        }

        // Dequant B tile: FP8 E4M3 → BF16 (lossless). Thread t < N_TILE loads
        // the 16 consecutive K bytes of row n = t as ONE 16-byte uint4 (fully
        // coalesced along K) instead of 8 single-byte loads strided by K
        // (32x sector amplification). Scalar tail for K % 16 != 0 or N edges
        // (production K = 2048 / 512 are both multiples of K_STEP = 16).
        if (threadIdx.x < N_TILE) {
            unsigned int n = threadIdx.x;
            unsigned int gn = cta_n + n;
            if (gn < N && k_base + K_STEP <= K) {
                uint4 v = *(const uint4*)&B_exp[(unsigned long long)gn * K + k_base];
                const unsigned char* b = (const unsigned char*)&v;
                #pragma unroll
                for (int k = 0; k < 16; k++)
                    smem_B[k][n] = e4m3_to_bf16_w8a8(b[k]);
            } else {
                for (int k = 0; k < K_STEP; k++) {
                    unsigned int gk = k_base + k;
                    smem_B[k][n] = (gk < K && gn < N)
                        ? e4m3_to_bf16_w8a8(B_exp[(unsigned long long)gn * K + gk])
                        : __float2bfloat16(0.0f);
                }
            }
        }

        __syncthreads();
        fp8_w8a8_mma(smem_A, smem_B, inner_acc, warp_m_offset, group_id, tid);
        __syncthreads();

        unsigned int next_k = k_base + K_STEP;
        if (next_k % K_PROMOTE == 0 || next_k >= K) {
            unsigned int k_block = k_base / FP8_BLOCK;
            const float bs = S_exp[n_block * k_blocks + k_block];
            // a_scale lookup per row. Row 0..7 use r0 = warp_m_offset+group_id,
            // row 8..15 use r1 = r0+8. For each n_tile, acc[][0,1] write to row r0,
            // acc[][2,3] to row r1.
            unsigned int r0 = warp_m_offset + group_id;
            unsigned int r1 = r0 + 8;
            int t0 = (r0 < M_TILE) ? smem_token_id[r0] : -1;
            int t1 = (r1 < M_TILE) ? smem_token_id[r1] : -1;
            const float as0 = (t0 >= 0)
                ? a_scale[(unsigned long long)t0 * k_blocks + k_block]
                : 0.0f;
            const float as1 = (t1 >= 0)
                ? a_scale[(unsigned long long)t1 * k_blocks + k_block]
                : 0.0f;
            const float s0 = as0 * bs;
            const float s1 = as1 * bs;
            #pragma unroll
            for (int n_tile = 0; n_tile < 8; n_tile++) {
                outer_acc[n_tile][0] += inner_acc[n_tile][0] * s0;
                outer_acc[n_tile][1] += inner_acc[n_tile][1] * s0;
                outer_acc[n_tile][2] += inner_acc[n_tile][2] * s1;
                outer_acc[n_tile][3] += inner_acc[n_tile][3] * s1;
                inner_acc[n_tile][0] = 0.0f; inner_acc[n_tile][1] = 0.0f;
                inner_acc[n_tile][2] = 0.0f; inner_acc[n_tile][3] = 0.0f;
            }
        }
    }

    #pragma unroll
    for (int n_tile = 0; n_tile < 8; n_tile++) {
        unsigned int base_n = cta_n + n_tile * 8;
        unsigned int col0 = base_n + (tid * 2);
        unsigned int col1 = col0 + 1;
        unsigned int row0 = cta_m_local + warp_m_offset + group_id;
        unsigned int row1 = row0 + 8;

        if (row0 < (unsigned int)M_expert) {
            unsigned int out_row = m_start + row0;
            if (col0 < N) C[(unsigned long long)out_row * N + col0] = __float2bfloat16(outer_acc[n_tile][0]);
            if (col1 < N) C[(unsigned long long)out_row * N + col1] = __float2bfloat16(outer_acc[n_tile][1]);
        }
        if (row1 < (unsigned int)M_expert) {
            unsigned int out_row = m_start + row1;
            if (col0 < N) C[(unsigned long long)out_row * N + col0] = __float2bfloat16(outer_acc[n_tile][2]);
            if (col1 < N) C[(unsigned long long)out_row * N + col1] = __float2bfloat16(outer_acc[n_tile][3]);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// moe_w8a8_grouped_gemm_pm4 — PM4-geometry W8A8 grouped GEMM (grid-compaction).
// ═══════════════════════════════════════════════════════════════════
//
// Port of the PM4 geometry proven by `moe_fp8_grouped_gemm.cu` (M_TILE=128,
// N_TILE=64, K_STEP=32 with 2 sub-MMAs, 2-stage cp.async.cg double buffering,
// K-contiguous smem_B[n][k], 256 threads, worklist-compacted 1D grid via
// `moe_build_tile_worklist`) onto the W8A8 path above, keeping the w8a8
// numerics EXACTLY:
//   - A is FP8 E4M3, gathered through sorted_token_ids, staged raw via
//     cp.async and dequanted to BF16 in smem (lossless);
//   - two-level FP32 accumulation with the per-row a_scale × b_scale fold at
//     K_PROMOTE=64 boundaries.
// The 2 sub-MMAs per K_STEP=32 accumulate in ascending K order — the same f32
// accumulation order as the K_STEP=16 kernel above — so on identical inputs
// the output is BIT-IDENTICAL to `moe_w8a8_grouped_gemm` (verified in the
// candidate bench: 0 differing bf16 outputs at both 36080- and 18040-row
// shapes, gate/up and down legs).
//
// Dequant is arithmetic (no LUT): the 7 magnitude bits placed in an f16
// exponent/mantissa frame decode E4M3 exactly after a ×2^8 rescale (bias 7 vs
// 15), subnormals included; NaN codes 0x7F/0xFF map to ±0.0 (LUT parity).
//
// Why it is ~4.7× the dense kernel above at the production shape (36080
// expanded rows, avg M≈141): M_TILE=128 halves expert-weight re-reads
// (ceil(141/64)=3 → 2), cp.async hides load latency behind MMA, and the
// worklist grid collapses the 99.5%-early-exit dense 3D launch.
//
// smem: 2 stages × (A 128×40 BF16 + A_raw 128×32 B + B 64×34 BF16 + B_raw
// 64×32 B) = 41472 B → 2 CTAs/SM. __launch_bounds__(256,2): 122 regs, 0 spills.
//
// SAME-STREAM INVARIANT: builder + this kernel MUST share a stream
// (read-after-write of total_tiles/worklist), as for moe_fp8_grouped_gemm.
//
// Grid: (wl_cap_items.clamp(1, MAX_GRID_CTAS), 1, 1)  Block: (256, 1, 1)
#define W8PM4_M_TILE 128
#define W8PM4_N_TILE 64
#define W8PM4_K_STEP 32
#define W8PM4_K_SUB 16
#define W8PM4_K_SUBS (W8PM4_K_STEP / W8PM4_K_SUB)
#define W8PM4_PAD 2
#define W8PM4_A_STRIDE (W8PM4_K_STEP + 8)
#define W8PM4_THREADS 256
#define W8PM4_NT_PER_WARP (W8PM4_N_TILE / 8)
#define W8PM4_STAGES 2
#define W8PM4_K_PROMOTE 64
#define W8PM4_FP8_BLOCK 128

__device__ __forceinline__ __nv_bfloat16 w8pm4_e4m3_to_bf16(unsigned char b) {
    // E4M3 -> f32 by bit arithmetic: place the 7 magnitude bits in an f16
    // exponent/mantissa frame and rescale by 2^8 (e4m3 bias 7 vs f16 bias 15).
    // Handles subnormals for free; NaN codes 0x7F/0xFF map to +/-0.0 (LUT parity).
    float f = __half2float(__ushort_as_half((unsigned short)((b & 0x7f) << 7))) * 256.0f;
    f = ((b & 0x7f) == 0x7f) ? 0.0f : f;
    return __float2bfloat16((b & 0x80) ? -f : f);
}

__device__ __forceinline__ void w8pm4_cp_async_cg_16(void* smem_ptr, const void* gmem_ptr) {
    unsigned int s = (unsigned int)__cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem_ptr));
}
__device__ __forceinline__ void w8pm4_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void w8pm4_cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void w8pm4_cp_async_wait_le(unsigned int n) {
    switch (n) {
        case 0:  w8pm4_cp_async_wait_group<0>(); break;
        case 1:  w8pm4_cp_async_wait_group<1>(); break;
        default: w8pm4_cp_async_wait_group<2>(); break;
    }
}

// MMA over one resident K_STEP (2 x m16n8k16 sub-MMAs in ascending K order —
// identical f32 accumulation order to the baseline's K_STEP=16 sequence).
// smem_B is [n][k] K-contiguous: the (k,k+1) B-fragment pair is one aligned u32.
__device__ __forceinline__ void w8pm4_mma_kstep(
    const __nv_bfloat16* smem_A,   // [W8PM4_M_TILE][W8PM4_A_STRIDE]
    const __nv_bfloat16* smem_B,   // [W8PM4_N_TILE][W8PM4_K_STEP + W8PM4_PAD]
    float inner[W8PM4_NT_PER_WARP][4],
    unsigned int warp_m_offset, unsigned int group_id, unsigned int tid
) {
    const unsigned int a_stride = W8PM4_A_STRIDE;
    const unsigned int b_stride = W8PM4_K_STEP + W8PM4_PAD;
    const unsigned short* sA = (const unsigned short*)smem_A;
    const unsigned short* sB = (const unsigned short*)smem_B;

    unsigned int frag_r0 = warp_m_offset + group_id;
    unsigned int frag_r1 = warp_m_offset + group_id + 8;

    #pragma unroll
    for (int s = 0; s < W8PM4_K_SUBS; s++) {
        const unsigned int k_off = s * W8PM4_K_SUB;
        unsigned int frag_c0 = k_off + tid * 2;
        unsigned int frag_c1 = k_off + tid * 2 + 8;

        unsigned int a0 = *(const unsigned int*)&sA[frag_r0 * a_stride + frag_c0];
        unsigned int a1 = *(const unsigned int*)&sA[frag_r1 * a_stride + frag_c0];
        unsigned int a2 = *(const unsigned int*)&sA[frag_r0 * a_stride + frag_c1];
        unsigned int a3 = *(const unsigned int*)&sA[frag_r1 * a_stride + frag_c1];

        #pragma unroll
        for (int n_tile = 0; n_tile < W8PM4_NT_PER_WARP; n_tile++) {
            unsigned int n_col = n_tile * 8 + group_id;
            unsigned int k0 = k_off + tid * 2;
            unsigned int k1 = k_off + tid * 2 + 8;

            unsigned int b0 = *(const unsigned int*)&sB[n_col * b_stride + k0];
            unsigned int b1 = *(const unsigned int*)&sB[n_col * b_stride + k1];

            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9}, "
                "{%10, %11, %12, %13};"
                : "=f"(inner[n_tile][0]), "=f"(inner[n_tile][1]),
                  "=f"(inner[n_tile][2]), "=f"(inner[n_tile][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                  "r"(b0), "r"(b1),
                  "f"(inner[n_tile][0]), "f"(inner[n_tile][1]),
                  "f"(inner[n_tile][2]), "f"(inner[n_tile][3])
            );
        }
    }
}

extern "C" __global__ void __launch_bounds__(W8PM4_THREADS, 2) moe_w8a8_grouped_gemm_pm4(
    const unsigned char* __restrict__ A_fp8,                // [total_tokens, K] FP8 E4M3
    const float* __restrict__ a_scale,                      // [total_tokens, K/128] FP32
    const unsigned long long* __restrict__ B_weight_ptrs,   // [num_experts] -> [N, K] FP8
    const unsigned long long* __restrict__ B_scale_ptrs,    // [num_experts] -> [N/128, K/128] FP32
    __nv_bfloat16* __restrict__ C,                          // [total_expanded, N] BF16
    const int* __restrict__ expert_offsets,                 // [num_experts + 1]
    const int* __restrict__ sorted_token_ids,               // [total_expanded] or NULL
    unsigned int num_experts,
    unsigned int N,
    unsigned int K,
    const unsigned int* __restrict__ worklist,              // [*total_tiles * 2]
    const int* __restrict__ total_tiles                     // [1]
) {
    __shared__ __align__(16) __nv_bfloat16 smem_A[W8PM4_STAGES][W8PM4_M_TILE][W8PM4_A_STRIDE];
    __shared__ __align__(16) unsigned char smem_Araw[W8PM4_STAGES][W8PM4_M_TILE][W8PM4_K_STEP];
    __shared__ __nv_bfloat16 smem_B[W8PM4_STAGES][W8PM4_N_TILE][W8PM4_K_STEP + W8PM4_PAD];
    __shared__ __align__(16) unsigned char smem_Braw[W8PM4_STAGES][W8PM4_N_TILE][W8PM4_K_STEP];

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    const int total = *total_tiles;

    for (int wid = blockIdx.x; wid < total; wid += (int)gridDim.x) {
        __syncthreads();   // fence smem reuse before re-priming the pipeline

        unsigned int expert_id = worklist[wid * 2 + 0];
        unsigned int packed    = worklist[wid * 2 + 1];
        unsigned int mt = packed >> 6;
        unsigned int nt = packed & 0x3F;

        const int m_start = expert_offsets[expert_id];
        const int M_expert = expert_offsets[expert_id + 1] - m_start;

        const unsigned char* B_exp = (const unsigned char*)B_weight_ptrs[expert_id];
        const float* S_exp = (const float*)B_scale_ptrs[expert_id];
        if (B_exp == 0) continue;

        const unsigned int cta_m_local = mt * W8PM4_M_TILE;
        const unsigned int cta_n = nt * W8PM4_N_TILE;

        float inner_acc[W8PM4_NT_PER_WARP][4];
        float outer_acc[W8PM4_NT_PER_WARP][4];
        #pragma unroll
        for (int i = 0; i < W8PM4_NT_PER_WARP; i++) {
            inner_acc[i][0] = 0.0f; inner_acc[i][1] = 0.0f;
            inner_acc[i][2] = 0.0f; inner_acc[i][3] = 0.0f;
            outer_acc[i][0] = 0.0f; outer_acc[i][1] = 0.0f;
            outer_acc[i][2] = 0.0f; outer_acc[i][3] = 0.0f;
        }

        const unsigned int k_blocks = (K + W8PM4_FP8_BLOCK - 1) / W8PM4_FP8_BLOCK;
        const unsigned int n_block = cta_n / W8PM4_FP8_BLOCK;
        const unsigned int n_steps = (K + W8PM4_K_STEP - 1) / W8PM4_K_STEP;
        const unsigned int steps_per_promote = W8PM4_K_PROMOTE / W8PM4_K_STEP;   // 2

        // Per-warp fragment rows: token ids resolved ONCE per tile for the
        // per-row a_scale fold (rows are fixed for the whole K loop).
        const unsigned int r0 = cta_m_local + warp_m_offset + group_id;
        const unsigned int r1 = r0 + 8;
        const int t0 = (r0 < (unsigned int)M_expert)
            ? (sorted_token_ids ? sorted_token_ids[m_start + (int)r0] : m_start + (int)r0) : -1;
        const int t1 = (r1 < (unsigned int)M_expert)
            ? (sorted_token_ids ? sorted_token_ids[m_start + (int)r1] : m_start + (int)r1) : -1;

        auto prefetch = [&](unsigned int step, unsigned int stage) {
            unsigned int k_base = step * W8PM4_K_STEP;

            // A raw: 128 rows x K_STEP FP8 bytes, 16-B chunks, gathered per row.
            const unsigned int a_chunks = (W8PM4_M_TILE * W8PM4_K_STEP) / 16;   // 256
            #pragma unroll
            for (unsigned int c = threadIdx.x; c < a_chunks; c += W8PM4_THREADS) {
                unsigned int row  = c / (W8PM4_K_STEP / 16);
                unsigned int kcol = (c % (W8PM4_K_STEP / 16)) * 16;
                unsigned int m_global = cta_m_local + row;
                unsigned int gk = k_base + kcol;
                unsigned char* dst = &smem_Araw[stage][row][kcol];
                if (m_global < (unsigned int)M_expert && gk + 16 <= K) {
                    int sorted_idx = m_start + (int)m_global;
                    int token_id = sorted_token_ids ? sorted_token_ids[sorted_idx] : sorted_idx;
                    w8pm4_cp_async_cg_16(dst, &A_fp8[(unsigned long long)token_id * K + gk]);
                } else {
                    #pragma unroll
                    for (unsigned int e = 0; e < 16; e++) {
                        unsigned int gke = gk + e;
                        if (m_global < (unsigned int)M_expert && gke < K) {
                            int sorted_idx = m_start + (int)m_global;
                            int token_id = sorted_token_ids ? sorted_token_ids[sorted_idx] : sorted_idx;
                            dst[e] = A_fp8[(unsigned long long)token_id * K + gke];
                        } else {
                            dst[e] = 0;   // dequants to +0.0
                        }
                    }
                }
            }

            // B raw: N_TILE rows x K_STEP FP8 bytes, 16-B chunks.
            const unsigned int b_chunks = (W8PM4_N_TILE * W8PM4_K_STEP) / 16;   // 128
            #pragma unroll
            for (unsigned int c = threadIdx.x; c < b_chunks; c += W8PM4_THREADS) {
                unsigned int nrow = (c * 16) / W8PM4_K_STEP;
                unsigned int kcol = (c * 16) % W8PM4_K_STEP;
                unsigned int gn = cta_n + nrow;
                unsigned int gk = k_base + kcol;
                unsigned char* dst = &smem_Braw[stage][nrow][kcol];
                if (gn < N && gk + 16 <= K) {
                    w8pm4_cp_async_cg_16(dst, &B_exp[(unsigned long long)gn * K + gk]);
                } else {
                    #pragma unroll
                    for (unsigned int e = 0; e < 16; e++) {
                        unsigned int gke = gk + e;
                        dst[e] = (gn < N && gke < K) ? B_exp[(unsigned long long)gn * K + gke] : 0;
                    }
                }
            }
            w8pm4_cp_async_commit();
        };

        // Arithmetic-dequant just-arrived raw A and B for `stage` into the
        // MMA-ready BF16 buffers (no scale — folded post-MMA at K_PROMOTE).
        auto dequant = [&](unsigned int stage) {
            #pragma unroll
            for (unsigned int idx = threadIdx.x; idx < W8PM4_M_TILE * W8PM4_K_STEP; idx += W8PM4_THREADS) {
                unsigned int row = idx / W8PM4_K_STEP;
                unsigned int k   = idx % W8PM4_K_STEP;
                smem_A[stage][row][k] = w8pm4_e4m3_to_bf16(smem_Araw[stage][row][k]);
            }
            #pragma unroll
            for (unsigned int idx = threadIdx.x; idx < W8PM4_N_TILE * W8PM4_K_STEP; idx += W8PM4_THREADS) {
                unsigned int n = idx / W8PM4_K_STEP;
                unsigned int k = idx % W8PM4_K_STEP;
                smem_B[stage][n][k] = w8pm4_e4m3_to_bf16(smem_Braw[stage][n][k]);
            }
        };

        #pragma unroll
        for (unsigned int p = 0; p < W8PM4_STAGES - 1; p++) {
            if (p < n_steps) prefetch(p, p % W8PM4_STAGES);
        }
        unsigned int k_step_in_prom = 0;

        for (unsigned int step = 0; step < n_steps; step++) {
            unsigned int cur = step % W8PM4_STAGES;

            unsigned int ahead = step + (W8PM4_STAGES - 1);
            if (ahead < n_steps) prefetch(ahead, ahead % W8PM4_STAGES);
            unsigned int committed = min(n_steps, W8PM4_STAGES + step);
            unsigned int target = committed - (step + 1);
            w8pm4_cp_async_wait_le(target);
            __syncthreads();   // raw A/B for `cur` resident for all threads

            dequant(cur);
            __syncthreads();   // smem_A/B[cur] fully written before MMA reads

            w8pm4_mma_kstep(&smem_A[cur][0][0], &smem_B[cur][0][0],
                         inner_acc, warp_m_offset, group_id, tid);
            __syncthreads();   // done reading smem_*[cur]; safe for reuse

            // K_PROMOTE boundary: fold per-row a_scale x b_scale, reset inner.
            k_step_in_prom++;
            if (k_step_in_prom == steps_per_promote || step + 1 == n_steps) {
                const unsigned int k_block = (step * W8PM4_K_STEP) / W8PM4_FP8_BLOCK;
                const float bs = S_exp[n_block * k_blocks + k_block];
                const float as0 = (t0 >= 0)
                    ? a_scale[(unsigned long long)t0 * k_blocks + k_block] : 0.0f;
                const float as1 = (t1 >= 0)
                    ? a_scale[(unsigned long long)t1 * k_blocks + k_block] : 0.0f;
                const float s0 = as0 * bs;
                const float s1 = as1 * bs;
                #pragma unroll
                for (int i = 0; i < W8PM4_NT_PER_WARP; i++) {
                    outer_acc[i][0] += inner_acc[i][0] * s0;
                    outer_acc[i][1] += inner_acc[i][1] * s0;
                    outer_acc[i][2] += inner_acc[i][2] * s1;
                    outer_acc[i][3] += inner_acc[i][3] * s1;
                    inner_acc[i][0] = 0.0f; inner_acc[i][1] = 0.0f;
                    inner_acc[i][2] = 0.0f; inner_acc[i][3] = 0.0f;
                }
                k_step_in_prom = 0;
            }
        }

        #pragma unroll
        for (int n_tile = 0; n_tile < W8PM4_NT_PER_WARP; n_tile++) {
            unsigned int base_n = cta_n + n_tile * 8;
            unsigned int col0 = base_n + (tid * 2);
            unsigned int col1 = col0 + 1;
            unsigned int row0 = cta_m_local + warp_m_offset + group_id;
            unsigned int row1 = row0 + 8;

            if (row0 < (unsigned int)M_expert) {
                unsigned int out_row = m_start + row0;
                if (col0 < N) C[(unsigned long long)out_row * N + col0] = __float2bfloat16(outer_acc[n_tile][0]);
                if (col1 < N) C[(unsigned long long)out_row * N + col1] = __float2bfloat16(outer_acc[n_tile][1]);
            }
            if (row1 < (unsigned int)M_expert) {
                unsigned int out_row = m_start + row1;
                if (col0 < N) C[(unsigned long long)out_row * N + col0] = __float2bfloat16(outer_acc[n_tile][2]);
                if (col1 < N) C[(unsigned long long)out_row * N + col1] = __float2bfloat16(outer_acc[n_tile][3]);
            }
        }
    }
}
