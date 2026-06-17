// SPDX-License-Identifier: AGPL-3.0-only

// Atlas FP8 Grouped MoE GEMM — HIP/gfx1151 (AMD WMMA) port of the NVIDIA
// mma.sync grid-compaction kernel (kernels/gb10/common/moe_fp8_grouped_gemm.cu).
//
//   C[M_expert, N] = A[M_expert, K] (BF16 acts) @ dequant(B_expert[N, K] (FP8 E4M3))
//
// THE routed-expert FP8 grouped-GEMM kernel for prefill. A persistent
// A grid-strided (stride = gridDim.x) launch over a COMPACTED (expert, m_tile, n_tile)
// work-list built by `moe_build_tile_worklist` (moe_permute.cu) so each
// work-item is exactly one real (non-early-exit) tile. Tokens are pre-sorted by
// expert (contiguous per expert); each expert's FP8 weight is loaded ONCE per
// (m_tile, n_tile) tile and matmul'd against ALL its assigned tokens via WMMA
// over the token (M) dimension — amortizing the LPDDR5X weight traffic.
//
// FP8 weight format: B[N,K] uint8 (E4M3) with block_scale[N/128, K/128] FP32.
//   Dequant: bf16_val = E4M3_LUT[byte] (scale folded post-WMMA per 128-K block).
//
// NUMERICS SSOT — must match the GB10 kernel AND the moe_microtest CPU oracle:
//   two-level FP32 accumulation — `inner` over a contiguous 128-K block, then
//   `outer += inner * block_scale` at the block boundary; the scale is NEVER
//   applied per-element. All f32 -> BF16 conversions use `__float2bfloat16`
//   (round-to-nearest-even), agreeing byte-exact with the CPU load-time dequant.
//
// HIP/gfx1151 transforms (mirror w8a16_gemm.cu / moe_w4a16_grouped_gemm.cu):
//   * mma.sync.m16n8k16.bf16 -> __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32
//     (one WMMA = two NVIDIA m16n8 n-tiles; N_TILE=64 -> 4 WMMA n-sub-tiles).
//   * cp.async -> register-prefetch ping-pong double-buffering: gfx1151 has NO
//     cp.async / global_load_lds, so VMEM reads for tile k+1 are issued into
//     VGPRs and stay in-flight across tile-k WMMA, then committed into the other
//     smem buffer.
//   * Grid-strided worklist launch (stride gridDim.x); the launcher sizes the
//     grid to the work-list tile-count bound (matches the Rust launcher
//     + moe_build_tile_worklist packing: (m_tile<<6)|n_tile, M_TILE=128, N_TILE=64).
//
// WMMA fragment layout (validated bit-exact in w8a16_gemm.cu / moe_w4a16):
//   A load (M x K row-major smem): lane l -> a[i] = smem_A[warp_m_offset+(l&15)][i], i=0..15
//   B load (K x N row-major smem): lane l -> b[k] = smem_B[k][nb*16 + (l&15)], k=0..15
//   Store: lane l, acc elem e(0..7) -> row = warp_m_offset + 2*e + (l>>4),
//          col = nb*16 + (l&15).
//
// LDS budget (gfx1151 cap = 64 KB/workgroup):
//   smem_A[2][128][18] bf16 = 2*128*18*2 = 9216 B
//   smem_B[2][16][66]  bf16 = 2*16*66*2  = 4224 B
//   lut_s[256] f32          = 1024 B
//   total ~= 14464 B  (well under 64 KB).
//
// Grid: (~tile_count CTAs, sized by the launcher, 1, 1)  Block: (PM4_THREADS=512, 1, 1)

#include <cuda_bf16.h>

typedef __bf16 v16bf __attribute__((ext_vector_type(16)));
typedef float  v8f   __attribute__((ext_vector_type(8)));
// ── Tile geometry ──────────────────────────────────────────────────────────
// A CTA computes a PM4_M_TILE×PM4_N_TILE output block. The warps form a 2-D
// (PM4_WARPS_M × PM4_WARPS_N) grid: PM4_WARPS_M warps tile the M dimension (each
// owns 16 token rows) and PM4_WARPS_N warps tile N (each owns
// PM4_SUBTILES_PER_WARP of the PM4_N_SUBTILES 16-wide WMMA n-sub-tiles).
//
// WHY 2-D (vs the GB10 1-D M-only warp grid): the GB10 geometry is 8 warps that
// all tile M (PM4_M_TILE=128), so adding warps means growing M — wasteful here
// because routed-expert tiles are SKINNY (top-8/256 routing -> ~47 tokens/expert
// average, far under 128). The long-K gate/up GEMM (K=2048, ~128 K-steps) is
// instead latency-bound and starved at 8 warps. The 2-D grid keeps the 128x64
// tile (work-list packing UNCHANGED) but splits the PM4_N_SUBTILES n-sub-tiles
// across PM4_WARPS_N warp-columns, so PM4_WARPS_M(=8) x PM4_WARPS_N(=2) = 16
// warps / 512 threads hide the K-reduction latency WITHOUT enlarging the tile.
// Measured (gfx1151, 256 experts, ~47 tok/expert): gate/up 2.25 -> 2.80 TFLOP/s
// (+24%), down ~flat, served 35B-A3B-FP8 TTFT (1710 tok) 3.08 -> 2.87 s (~7%).
// A 32-warp (PM4_WARPS_N=4) variant regressed gate/up (over-subscription) and
// a smaller PM4_M_TILE=64 helped only the short-K down GEMM, so 128x64/16-warp
// is the best single geometry for the 2x gate/up + 1x down per-layer mix.
#define FP8_BLOCK 128

#define PM4_M_TILE 128
#define PM4_N_TILE 64
#define PM4_K_STEP 16
#define PM4_PAD 2
#define PM4_WARPS_M (PM4_M_TILE / 16)            // warp-rows over M
#define PM4_WARPS_N 2                            // warp-cols over N
#define PM4_WARPS (PM4_WARPS_M * PM4_WARPS_N)
#define PM4_THREADS (PM4_WARPS * 32)
#define PM4_N_SUBTILES (PM4_N_TILE / 16)         // total WMMA n-sub-tiles
#define PM4_SUBTILES_PER_WARP (PM4_N_SUBTILES / PM4_WARPS_N)

__device__ __constant__ float E4M3_LUT_GMOE[256] = {
    0.0f, 0.001953125f, 0.00390625f, 0.005859375f,
    0.0078125f, 0.009765625f, 0.01171875f, 0.013671875f,
    0.015625f, 0.017578125f, 0.01953125f, 0.021484375f,
    0.0234375f, 0.025390625f, 0.02734375f, 0.029296875f,
    0.03125f, 0.03515625f, 0.0390625f, 0.04296875f,
    0.046875f, 0.05078125f, 0.0546875f, 0.05859375f,
    0.0625f, 0.0703125f, 0.078125f, 0.0859375f,
    0.09375f, 0.1015625f, 0.109375f, 0.1171875f,
    0.125f, 0.140625f, 0.15625f, 0.171875f,
    0.1875f, 0.203125f, 0.21875f, 0.234375f,
    0.25f, 0.28125f, 0.3125f, 0.34375f,
    0.375f, 0.40625f, 0.4375f, 0.46875f,
    0.5f, 0.5625f, 0.625f, 0.6875f,
    0.75f, 0.8125f, 0.875f, 0.9375f,
    1.0f, 1.125f, 1.25f, 1.375f,
    1.5f, 1.625f, 1.75f, 1.875f,
    2.0f, 2.25f, 2.5f, 2.75f,
    3.0f, 3.25f, 3.5f, 3.75f,
    4.0f, 4.5f, 5.0f, 5.5f,
    6.0f, 6.5f, 7.0f, 7.5f,
    8.0f, 9.0f, 10.0f, 11.0f,
    12.0f, 13.0f, 14.0f, 15.0f,
    16.0f, 18.0f, 20.0f, 22.0f,
    24.0f, 26.0f, 28.0f, 30.0f,
    32.0f, 36.0f, 40.0f, 44.0f,
    48.0f, 52.0f, 56.0f, 60.0f,
    64.0f, 72.0f, 80.0f, 88.0f,
    96.0f, 104.0f, 112.0f, 120.0f,
    128.0f, 144.0f, 160.0f, 176.0f,
    192.0f, 208.0f, 224.0f, 240.0f,
    256.0f, 288.0f, 320.0f, 352.0f,
    384.0f, 416.0f, 448.0f, 0.0f,
    -0.0f, -0.001953125f, -0.00390625f, -0.005859375f,
    -0.0078125f, -0.009765625f, -0.01171875f, -0.013671875f,
    -0.015625f, -0.017578125f, -0.01953125f, -0.021484375f,
    -0.0234375f, -0.025390625f, -0.02734375f, -0.029296875f,
    -0.03125f, -0.03515625f, -0.0390625f, -0.04296875f,
    -0.046875f, -0.05078125f, -0.0546875f, -0.05859375f,
    -0.0625f, -0.0703125f, -0.078125f, -0.0859375f,
    -0.09375f, -0.1015625f, -0.109375f, -0.1171875f,
    -0.125f, -0.140625f, -0.15625f, -0.171875f,
    -0.1875f, -0.203125f, -0.21875f, -0.234375f,
    -0.25f, -0.28125f, -0.3125f, -0.34375f,
    -0.375f, -0.40625f, -0.4375f, -0.46875f,
    -0.5f, -0.5625f, -0.625f, -0.6875f,
    -0.75f, -0.8125f, -0.875f, -0.9375f,
    -1.0f, -1.125f, -1.25f, -1.375f,
    -1.5f, -1.625f, -1.75f, -1.875f,
    -2.0f, -2.25f, -2.5f, -2.75f,
    -3.0f, -3.25f, -3.5f, -3.75f,
    -4.0f, -4.5f, -5.0f, -5.5f,
    -6.0f, -6.5f, -7.0f, -7.5f,
    -8.0f, -9.0f, -10.0f, -11.0f,
    -12.0f, -13.0f, -14.0f, -15.0f,
    -16.0f, -18.0f, -20.0f, -22.0f,
    -24.0f, -26.0f, -28.0f, -30.0f,
    -32.0f, -36.0f, -40.0f, -44.0f,
    -48.0f, -52.0f, -56.0f, -60.0f,
    -64.0f, -72.0f, -80.0f, -88.0f,
    -96.0f, -104.0f, -112.0f, -120.0f,
    -128.0f, -144.0f, -160.0f, -176.0f,
    -192.0f, -208.0f, -224.0f, -240.0f,
    -256.0f, -288.0f, -320.0f, -352.0f,
    -384.0f, -416.0f, -448.0f, -0.0f,
};

// Per-thread prefetch register staging.
//   A: (M_TILE*K_STEP)/THREADS elems/thread.   B raw FP8: (K_STEP*N_TILE)/THREADS bytes/thread.
#define PM4_A_EPT ((PM4_M_TILE * PM4_K_STEP) / PM4_THREADS)
#define PM4_B_EPT ((PM4_K_STEP * PM4_N_TILE) / PM4_THREADS)

// A prefetch: gather token rows through sorted_token_ids, contiguous K.
__device__ __forceinline__ void load_A_regs(
    const __nv_bfloat16* __restrict__ A,
    const int* __restrict__ sorted_token_ids,
    __nv_bfloat16 reg_A[PM4_A_EPT],
    int m_start, unsigned int cta_m_local, unsigned int k_base,
    unsigned int M_expert, unsigned int K
) {
    #pragma unroll
    for (unsigned int i = 0; i < PM4_A_EPT; i++) {
        unsigned int idx = threadIdx.x * PM4_A_EPT + i;
        unsigned int row = idx / PM4_K_STEP;
        unsigned int col = idx % PM4_K_STEP;
        unsigned int m_global = cta_m_local + row;
        unsigned int gc = k_base + col;
        if (m_global < M_expert && gc < K) {
            int sorted_idx = m_start + (int)m_global;
            int token_id = sorted_token_ids ? sorted_token_ids[sorted_idx] : sorted_idx;
            reg_A[i] = A[(unsigned long long)token_id * K + gc];
        } else {
            reg_A[i] = __float2bfloat16(0.0f);
        }
    }
}

__device__ __forceinline__ void store_A_regs(
    __nv_bfloat16 smem_A[][PM4_K_STEP + PM4_PAD], const __nv_bfloat16 reg_A[PM4_A_EPT]
) {
    #pragma unroll
    for (unsigned int i = 0; i < PM4_A_EPT; i++) {
        unsigned int idx = threadIdx.x * PM4_A_EPT + i;
        unsigned int row = idx / PM4_K_STEP;
        unsigned int col = idx % PM4_K_STEP;
        smem_A[row][col] = reg_A[i];
    }
}

// B prefetch: load raw FP8 bytes for the [K_STEP, N_TILE] tile into regs.
// Layout consumed: smem_B[k][n]. The E4M3 LUT dequant (NO scale — folded
// post-WMMA per 128-K block) happens on commit.
__device__ __forceinline__ void load_B_regs(
    const unsigned char* __restrict__ B_exp,
    unsigned char reg_B[PM4_B_EPT],
    unsigned int cta_n, unsigned int k_base,
    unsigned int N, unsigned int K
) {
    #pragma unroll
    for (unsigned int i = 0; i < PM4_B_EPT; i++) {
        unsigned int idx = threadIdx.x * PM4_B_EPT + i;
        unsigned int k = idx / PM4_N_TILE;
        unsigned int n = idx % PM4_N_TILE;
        unsigned int gk = k_base + k;
        unsigned int gn = cta_n + n;
        reg_B[i] = (gk < K && gn < N) ? B_exp[(unsigned long long)gn * K + gk] : 0;
    }
}

__device__ __forceinline__ void store_B_regs(
    __nv_bfloat16 smem_B[][PM4_N_TILE + PM4_PAD],
    const unsigned char reg_B[PM4_B_EPT], const float* lut_s
) {
    #pragma unroll
    for (unsigned int i = 0; i < PM4_B_EPT; i++) {
        unsigned int idx = threadIdx.x * PM4_B_EPT + i;
        unsigned int k = idx / PM4_N_TILE;
        unsigned int n = idx % PM4_N_TILE;
        smem_B[k][n] = __float2bfloat16(lut_s[reg_B[i]]);   // dequant, NO scale
    }
}

// WMMA over one resident K_STEP (16) into this warp's PM4_SUBTILES_PER_WARP v8f
// accumulators. The warp owns the N-sub-tiles [n_sub_base, n_sub_base+SPW).
__device__ __forceinline__ void mma_kstep(
    const __nv_bfloat16 smem_A[][PM4_K_STEP + PM4_PAD],
    const __nv_bfloat16 smem_B[][PM4_N_TILE + PM4_PAD],
    v8f inner[PM4_SUBTILES_PER_WARP],
    unsigned int warp_m_offset, unsigned int n_sub_base, unsigned int lane
) {
    v16bf a;
    #pragma unroll
    for (int i = 0; i < 16; i++) a[i] = (__bf16)smem_A[warp_m_offset + (lane & 15)][i];
    #pragma unroll
    for (int j = 0; j < PM4_SUBTILES_PER_WARP; j++) {
        unsigned int nb = n_sub_base + j;
        v16bf b;
        #pragma unroll
        for (int k = 0; k < 16; k++) b[k] = (__bf16)smem_B[k][nb * 16 + (lane & 15)];
        inner[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, inner[j]);
    }
}

extern "C" __global__ void __launch_bounds__(PM4_THREADS, 2) moe_fp8_grouped_gemm(
    const __nv_bfloat16* __restrict__ A,                    // [total_tokens, K] BF16
    const unsigned long long* __restrict__ B_weight_ptrs,    // [num_experts] -> [N, K] FP8
    const unsigned long long* __restrict__ B_scale_ptrs,     // [num_experts] -> [N/128, K/128] FP32
    __nv_bfloat16* __restrict__ C,                          // [total_expanded, N] BF16
    const int* __restrict__ expert_offsets,                  // [num_experts + 1]
    const int* __restrict__ sorted_token_ids,               // [total_expanded] or NULL
    unsigned int num_experts,
    unsigned int N,
    unsigned int K,
    const unsigned int* __restrict__ worklist,               // [*total_tiles * 2] (expert, packed m/n)
    const int* __restrict__ total_tiles                      // [1] (read-after-write on same stream)
) {
    (void)num_experts;

    __shared__ float lut_s[256];
    #pragma unroll
    for (unsigned int i = threadIdx.x; i < 256; i += PM4_THREADS) {
        lut_s[i] = E4M3_LUT_GMOE[i];
    }

    // Double-buffered (ping-pong) smem — overlaps tile-(k+1) global loads
    // (staged in VGPRs) with tile-k WMMA. Shared across all work-items.
    __shared__ __nv_bfloat16 smem_A[2][PM4_M_TILE][PM4_K_STEP + PM4_PAD];
    __shared__ __nv_bfloat16 smem_B[2][PM4_K_STEP][PM4_N_TILE + PM4_PAD];

    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    // 2-D warp grid: row owns 16 M-rows, col owns PM4_SUBTILES_PER_WARP n-tiles.
    const unsigned int warp_m_offset = (warp_id / PM4_WARPS_N) * 16;
    const unsigned int n_sub_base    = (warp_id % PM4_WARPS_N) * PM4_SUBTILES_PER_WARP;

    const int total = *total_tiles;

    for (int wid = blockIdx.x; wid < total; wid += (int)gridDim.x) {
        __syncthreads();   // fence smem reuse before re-priming the pipeline

        unsigned int expert_id = worklist[wid * 2 + 0];
        unsigned int packed    = worklist[wid * 2 + 1];
        unsigned int mt = packed >> 6;
        unsigned int nt = packed & 0x3F;

        const int m_start = expert_offsets[expert_id];
        const unsigned int M_expert = (unsigned int)(expert_offsets[expert_id + 1] - m_start);

        const unsigned char* B_exp = (const unsigned char*)B_weight_ptrs[expert_id];
        const float* S_exp = (const float*)B_scale_ptrs[expert_id];
        if (B_exp == 0) continue;   // NULL -> remote expert under EP

        const unsigned int cta_m_local = mt * PM4_M_TILE;
        const unsigned int cta_n = nt * PM4_N_TILE;

        // Two-level FP32 accumulation: inner over a 128-K block, fold
        // inner*block_scale into outer at the boundary; scale never per-element.
        // Each warp holds only its PM4_SUBTILES_PER_WARP n-sub-tiles.
        v8f inner_acc[PM4_SUBTILES_PER_WARP];
        v8f outer_acc[PM4_SUBTILES_PER_WARP];
        #pragma unroll
        for (int i = 0; i < PM4_SUBTILES_PER_WARP; i++) {
            inner_acc[i] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
            outer_acc[i] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
        }

        const unsigned int k_blocks = (K + FP8_BLOCK - 1) / FP8_BLOCK;
        const unsigned int k_steps_per_block = FP8_BLOCK / PM4_K_STEP;
        const unsigned int n_block = cta_n / FP8_BLOCK;
        const unsigned int n_steps = (K + PM4_K_STEP - 1) / PM4_K_STEP;

        // Prologue: stage tile 0 (global -> regs -> smem[0]).
        __nv_bfloat16 reg_A[PM4_A_EPT];
        unsigned char reg_B[PM4_B_EPT];
        load_A_regs(A, sorted_token_ids, reg_A, m_start, cta_m_local, 0, M_expert, K);
        load_B_regs(B_exp, reg_B, cta_n, 0, N, K);
        store_A_regs(smem_A[0], reg_A);
        store_B_regs(smem_B[0], reg_B, lut_s);
        __syncthreads();

        unsigned int k_step_in_block = 0;

        // Single barrier per K-step: the ping-pong double-buffer's publish
        // barrier (e) at the end of iteration `step` already fences every warp's
        // read of buf[step&1] in (b) before iteration step+1 overwrites that same
        // buffer in its own (e). The separate post-compute barrier is redundant.
        for (unsigned int step = 0; step < n_steps; step++) {
            const unsigned int cur = step & 1;

            // (a) issue global reads for the next tile into registers (no wait)
            if (step + 1 < n_steps) {
                unsigned int k_next = (step + 1) * PM4_K_STEP;
                load_A_regs(A, sorted_token_ids, reg_A, m_start, cta_m_local, k_next, M_expert, K);
                load_B_regs(B_exp, reg_B, cta_n, k_next, N, K);
            }
            // (b) compute the already-resident current tile
            mma_kstep(smem_A[cur], smem_B[cur], inner_acc, warp_m_offset, n_sub_base, lane_id);

            // (c) K_BLOCK boundary: fold scaled inner into outer, reset inner.
            k_step_in_block++;
            if (k_step_in_block == k_steps_per_block) {
                const unsigned int k_block = (step * PM4_K_STEP) / FP8_BLOCK;
                const float scale = S_exp[n_block * k_blocks + k_block];
                #pragma unroll
                for (int i = 0; i < PM4_SUBTILES_PER_WARP; i++) {
                    outer_acc[i] += inner_acc[i] * scale;
                    inner_acc[i] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
                }
                k_step_in_block = 0;
            }

            // (d) commit prefetched registers into the other buffer, then publish.
            if (step + 1 < n_steps) {
                store_A_regs(smem_A[(step + 1) & 1], reg_A);
                store_B_regs(smem_B[(step + 1) & 1], reg_B, lut_s);
                __syncthreads();   // publish next tile + fence this iter's reads
            }
        }

        if (k_step_in_block != 0) {
            const unsigned int k_block = (K - 1) / FP8_BLOCK;
            const float scale = S_exp[n_block * k_blocks + k_block];
            #pragma unroll
            for (int i = 0; i < PM4_SUBTILES_PER_WARP; i++) {
                outer_acc[i] += inner_acc[i] * scale;
            }
        }

        // Store C tile: f32 outer -> BF16, sorted output position. Each warp
        // writes only its own n-sub-tiles [n_sub_base, n_sub_base+SPW).
        #pragma unroll
        for (int j = 0; j < PM4_SUBTILES_PER_WARP; j++) {
            unsigned int nb = n_sub_base + j;
            #pragma unroll
            for (int e = 0; e < 8; e++) {
                unsigned int row_local = cta_m_local + warp_m_offset + 2 * e + (lane_id >> 4);
                unsigned int col = cta_n + nb * 16 + (lane_id & 15);
                if (row_local < M_expert && col < N) {
                    unsigned int out_row = (unsigned int)m_start + row_local;
                    C[(unsigned long long)out_row * N + col] = __float2bfloat16(outer_acc[j][e]);
                }
            }
        }
    }
}

// Compile-compat second entry (v2) — no-op alias kept so the registry links the
// byte-identical NVIDIA-source signature. The dense model never dispatches it.
extern "C" __global__ void moe_fp8_grouped_gemm_v2(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ B_weight_ptrs,
    const unsigned long long* __restrict__ B_scale_ptrs,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    unsigned int num_experts,
    unsigned int N,
    unsigned int K
) {
    (void)A; (void)B_weight_ptrs; (void)B_scale_ptrs; (void)C;
    (void)expert_offsets; (void)sorted_token_ids; (void)num_experts; (void)N; (void)K;
}
