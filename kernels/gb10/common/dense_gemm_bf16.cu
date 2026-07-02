// SPDX-License-Identifier: AGPL-3.0-only

// Atlas Dense GEMM kernel for SM121 (GB10).
//
// C = A * B^T  where:
//   A: [M, K] BF16 (activations, row-major)
//   B: [N, K] BF16 (weights, row-major — standard HuggingFace layout)
//   C: [M, N] BF16 (output, row-major)
//
// The kernel reads B transposed: B^T[k,n] = B[n,k] = B[n*K + k].
//
// Phase 1: Correct scalar implementation with shared memory tiling.
// Phase 2: Will add mma.sync.aligned.m16n8k16 BF16 tensor cores.

#include <cuda_bf16.h>

#define TILE_M 16
#define TILE_N 16
#define TILE_K 16

// Tiled GEMM: C[M,N] = A[M,K] * B[N,K]^T
// All matrices in BF16, accumulation in FP32.
//
// Grid: (ceil(N/TILE_N), ceil(M/TILE_M))
// Block: (TILE_N, TILE_M) — each thread computes one output element
extern "C" __global__ void dense_gemm_bf16(
    const __nv_bfloat16* __restrict__ A,  // [M, K] row-major
    const __nv_bfloat16* __restrict__ B,  // [N, K] row-major (read transposed)
    __nv_bfloat16* __restrict__ C,         // [M, N] row-major
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    // Each thread computes one element of C
    unsigned int row = blockIdx.y * TILE_M + threadIdx.y;
    unsigned int col = blockIdx.x * TILE_N + threadIdx.x;

    // Shared memory tiles
    __shared__ __nv_bfloat16 smem_A[TILE_M][TILE_K];
    __shared__ __nv_bfloat16 smem_B[TILE_K][TILE_N];

    float acc = 0.0f;

    // Loop over K in TILE_K chunks
    for (unsigned int k_base = 0; k_base < K; k_base += TILE_K) {
        // Load A tile
        if (row < M && (k_base + threadIdx.x) < K) {
            smem_A[threadIdx.y][threadIdx.x] = A[row * K + k_base + threadIdx.x];
        } else {
            smem_A[threadIdx.y][threadIdx.x] = __float2bfloat16(0.0f);
        }

        // Load B tile (B is [N,K] row-major, read as B^T[K,N])
        if ((k_base + threadIdx.y) < K && col < N) {
            smem_B[threadIdx.y][threadIdx.x] = B[(unsigned long long)col * K + (k_base + threadIdx.y)];
        } else {
            smem_B[threadIdx.y][threadIdx.x] = __float2bfloat16(0.0f);
        }

        __syncthreads();

        // Compute partial dot product
        for (unsigned int kk = 0; kk < TILE_K; kk++) {
            acc += __bfloat162float(smem_A[threadIdx.y][kk])
                 * __bfloat162float(smem_B[kk][threadIdx.x]);
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        C[row * N + col] = __float2bfloat16(acc);
    }
}

// FP32-output twin of dense_gemm_bf16: identical math, but writes the FP32
// accumulator directly instead of rounding to BF16. Used by the
// ATLAS_FP32_GATE routing path for the MoE router GEMM so the gate logits
// keep full precision into top-K (a BF16 store would round two near-tied
// experts to the same value and flip routing). Inputs stay BF16.
//
// Grid: (ceil(N/TILE_N), ceil(M/TILE_M))
// Block: (TILE_N, TILE_M)
extern "C" __global__ void dense_gemm_bf16_f32out(
    const __nv_bfloat16* __restrict__ A,  // [M, K] row-major
    const __nv_bfloat16* __restrict__ B,  // [N, K] row-major (read transposed)
    float* __restrict__ C,                 // [M, N] row-major, FP32
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    unsigned int row = blockIdx.y * TILE_M + threadIdx.y;
    unsigned int col = blockIdx.x * TILE_N + threadIdx.x;

    __shared__ __nv_bfloat16 smem_A[TILE_M][TILE_K];
    __shared__ __nv_bfloat16 smem_B[TILE_K][TILE_N];

    float acc = 0.0f;

    for (unsigned int k_base = 0; k_base < K; k_base += TILE_K) {
        if (row < M && (k_base + threadIdx.x) < K) {
            smem_A[threadIdx.y][threadIdx.x] = A[row * K + k_base + threadIdx.x];
        } else {
            smem_A[threadIdx.y][threadIdx.x] = __float2bfloat16(0.0f);
        }
        if ((k_base + threadIdx.y) < K && col < N) {
            smem_B[threadIdx.y][threadIdx.x] = B[(unsigned long long)col * K + (k_base + threadIdx.y)];
        } else {
            smem_B[threadIdx.y][threadIdx.x] = __float2bfloat16(0.0f);
        }
        __syncthreads();
        for (unsigned int kk = 0; kk < TILE_K; kk++) {
            acc += __bfloat162float(smem_A[threadIdx.y][kk])
                 * __bfloat162float(smem_B[kk][threadIdx.x]);
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// FP32-input, FP32-output variant: A is FP32 activations (the FP32 router_in from
// residual_add_rms_norm_gatef32), B is BF16 weights (the router gate), C is FP32.
// The ATLAS_FP32_ROUTING gate GEMM: full-precision activation × bf16 gate weight
// → unrounded gate logits, so top-K selection no longer flips on a bf16 store.
//
// Grid: (ceil(N/TILE_N), ceil(M/TILE_M))   Block: (TILE_N, TILE_M)
extern "C" __global__ void dense_gemm_f32in_f32out(
    const float* __restrict__ A,          // [M, K] row-major, FP32
    const __nv_bfloat16* __restrict__ B,  // [N, K] row-major (read transposed), BF16
    float* __restrict__ C,                 // [M, N] row-major, FP32
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    unsigned int row = blockIdx.y * TILE_M + threadIdx.y;
    unsigned int col = blockIdx.x * TILE_N + threadIdx.x;

    __shared__ float smem_A[TILE_M][TILE_K];
    __shared__ __nv_bfloat16 smem_B[TILE_K][TILE_N];

    float acc = 0.0f;

    for (unsigned int k_base = 0; k_base < K; k_base += TILE_K) {
        if (row < M && (k_base + threadIdx.x) < K) {
            smem_A[threadIdx.y][threadIdx.x] = A[row * K + k_base + threadIdx.x];
        } else {
            smem_A[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if ((k_base + threadIdx.y) < K && col < N) {
            smem_B[threadIdx.y][threadIdx.x] = B[(unsigned long long)col * K + (k_base + threadIdx.y)];
        } else {
            smem_B[threadIdx.y][threadIdx.x] = __float2bfloat16(0.0f);
        }
        __syncthreads();
        for (unsigned int kk = 0; kk < TILE_K; kk++) {
            acc += smem_A[threadIdx.y][kk] * __bfloat162float(smem_B[kk][threadIdx.x]);
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Fix-E: tensor-core pipelined BF16 dense GEMM (dense_gemm_bf16_pipelined).
//
// C[M,N] = A[M,K] (BF16, row-major) · B[N,K]^T (BF16, row-major)
//   i.e. C[m,n] = Σ_k A[m,k]·B[n,k].   Same math + same I/O layout as the
//   scalar dense_gemm_bf16 above; the original is left UNTOUCHED as the
//   production fallback.
//
// This is the w8a16_gemm_pipelined structure MINUS the FP8 dequant path: both
// A and B are BF16, loaded DIRECTLY into smem via cp.async (no E4M3 LUT, no
// smem_Braw staging, no block-scale, no two-level accumulation — a SINGLE FP32
// accumulator runs over all K). So it is strictly simpler than the W8A16 kernel.
//
// TILE SWEEP RESULT (BF16, kernel-only TFLOP/s on GB10/sm_121, K=4096):
// unlike the register-bound FP8 GEMMs (where N=32 was the occupancy sweet
// spot), this BF16 kernel is MMA-ISSUE / BARRIER bound — it has NO dequant
// (no LUT, no smem_Braw staging) so its base register pressure is far lower,
// and the win comes from putting MORE MMA work between the per-K-step
// __syncthreads barriers. A WIDER N-tile (more m16n8k16 N-sub-tiles per warp)
// amortizes each barrier over more MMAs and keeps winning even as occupancy
// falls:
//   N=32  → 47 regs, ~5 CTAs/SM → 24-26 TFLOP/s
//   N=64  → 63 regs, ~4 CTAs/SM → 40-43 TFLOP/s
//   N=96  → 80 regs, ~3 CTAs/SM → 48 TFLOP/s
//   N=128 → 98 regs, ~2 CTAs/SM → 51-58 TFLOP/s  ← chosen default
// 128×128 (M×N) is the shipped tile: on the production hot shape (the SSM
// out_proj, M=31311 N=2048 K=4096) it hits ~58 TFLOP/s = 27% of the 213
// TFLOP/s BF16 peak (a 40× speedup over the 1.42 TFLOP/s scalar kernel). A
// 2-stage cp.async pipeline prefetches the next K-step's A and B tiles
// (3 stages REGRESSED — not load-latency bound; K_STEP=64 was neutral).
// NOTE: the wide tile trades small-shape efficiency (few CTAs when M or N is
// small) for large-shape throughput — correct for this kernel, whose hot
// path is always large-M (token count = thousands). The original scalar
// dense_gemm_bf16 above stays the fallback for tiny shapes if ever needed.
//
// Uses mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 with FP32 accumulate.
// cp.async.cg (sm_80+) is correct on sm_121; TMA / cp.async.bulk are AVOIDED
// (they silently corrupt on sm_121). smem = 40960 B/CTA at the default tile,
// well under 101 KB.
//
// Grid: (ceil(N/DM_N_TILE), ceil(M/DM_M_TILE), 1), Block: (256,1,1) = 8 warps.

// Tile/pipeline params are #ifndef-guarded so a `-D` sweep can override them
// from nvcc without editing the file; the bodies below are the defaults that
// the production launch geometry in gemm_dense.rs mirrors.
#define DM_M_TILE 128
// N-tile = 128 (the BF16 sweep winner — see the file header). This kernel is
// MMA-issue/barrier bound, NOT register/occupancy bound (no dequant → low base
// reg pressure), so the WIDER tile that puts 16 m16n8k16 N-sub-tiles per warp
// between barriers wins despite dropping to ~2 CTAs/SM. 98 regs, 40960 B smem.
#ifndef DM_N_TILE
#define DM_N_TILE 128
#endif
// K_STEP=32 → each resident K-step carries TWO m16n8k16 MMAs (16-K each),
// amortizing the per-K-step barrier pair over 2× the MMA work.
#ifndef DM_K_STEP
#define DM_K_STEP 32
#endif
#define DM_K_SUB 16                            // one MMA's K-width
#define DM_K_SUBS (DM_K_STEP / DM_K_SUB)       // = 2 at K_STEP=32
// Smem row strides MUST be a multiple of 8 BF16 (16 bytes) so every 16-B
// cp.async.cg chunk lands on a 16-byte boundary. K_STEP K-cols + 8 pad; the
// pad also breaks smem bank conflicts on the MMA's u32 reads. Derived from
// K_STEP so a K_STEP sweep keeps alignment automatically.
#define DM_A_STRIDE (DM_K_STEP + 8)
#define DM_B_STRIDE (DM_K_STEP + 8)
#define DM_WARPS 8
#define DM_THREADS (DM_WARPS * 32)             // 256
#define DM_N_TILES_PER_WARP (DM_N_TILE / 8)    // m16n8k16 N-tiles per warp (=4)
// cp.async pipeline depth. 2 stages wins (BF16 sweep: 2 vs 3 within noise, 2
// uses least smem) — the kernel is occupancy/issue-bound, not load-latency
// bound, so a deeper prefetch does not help.
#ifndef DM_STAGES
#define DM_STAGES 2
#endif

// cp.async 16-byte (cg = cache-global) copy: smem <- global. sm_80+; correct
// on sm_121 (unlike TMA / cp.async.bulk). Requires 16-byte-aligned addresses.
__device__ __forceinline__ void dm_cp_async_cg_16(void* smem_ptr, const void* gmem_ptr) {
    unsigned int s = (unsigned int)__cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem_ptr));
}
__device__ __forceinline__ void dm_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void dm_cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
// Wait until at most `n` cp.async groups remain in flight (runtime n in
// [0, DM_STAGES-1]); cp.async.wait_group needs a compile-time immediate, so
// dispatch through a small switch over the legal values.
__device__ __forceinline__ void dm_cp_async_wait_le(unsigned int n) {
    switch (n) {
        case 0:  dm_cp_async_wait_group<0>(); break;
        case 1:  dm_cp_async_wait_group<1>(); break;
        case 2:  dm_cp_async_wait_group<2>(); break;
        default: dm_cp_async_wait_group<3>(); break;
    }
}

// MMA over one resident K_STEP (DM_K_STEP K-elements = DM_K_SUBS m16n8k16
// sub-MMAs of DM_K_SUB=16 K each). Accumulates into acc[DM_N_TILES_PER_WARP][4]
// (one accumulator quad per N-tile, summed across both sub-MMAs).
// A fragment: row-major BF16 m16k16. B fragment: smem_B [n][k] K-CONTIGUOUS, so
// the two consecutive-K BF16 (k, k+1) the MMA wants pack into a single aligned
// u32 load — exactly as in w8a16_gemm_pipelined's pm_mma_kstep.
__device__ __forceinline__ void dm_mma_kstep(
    const __nv_bfloat16* smem_A,   // [DM_M_TILE][DM_A_STRIDE]
    const __nv_bfloat16* smem_B,   // [DM_N_TILE][DM_B_STRIDE]  (K-contiguous)
    float acc[DM_N_TILES_PER_WARP][4],
    unsigned int warp_m_offset, unsigned int group_id, unsigned int tid
) {
    const unsigned int a_stride = DM_A_STRIDE;
    const unsigned int b_stride = DM_B_STRIDE;
    const unsigned short* sA = (const unsigned short*)smem_A;
    const unsigned short* sB = (const unsigned short*)smem_B;

    unsigned int frag_r0 = warp_m_offset + group_id;
    unsigned int frag_r1 = warp_m_offset + group_id + 8;

    #pragma unroll
    for (int s = 0; s < DM_K_SUBS; s++) {
        const unsigned int k_off = s * DM_K_SUB;     // K offset of this sub-MMA
        unsigned int frag_c0 = k_off + tid * 2;
        unsigned int frag_c1 = k_off + tid * 2 + 8;

        unsigned int a0 = *(const unsigned int*)&sA[frag_r0 * a_stride + frag_c0];
        unsigned int a1 = *(const unsigned int*)&sA[frag_r1 * a_stride + frag_c0];
        unsigned int a2 = *(const unsigned int*)&sA[frag_r0 * a_stride + frag_c1];
        unsigned int a3 = *(const unsigned int*)&sA[frag_r1 * a_stride + frag_c1];

        #pragma unroll
        for (int n_tile = 0; n_tile < DM_N_TILES_PER_WARP; n_tile++) {
            unsigned int n_col = n_tile * 8 + group_id;
            unsigned int k0 = k_off + tid * 2;
            unsigned int k1 = k_off + tid * 2 + 8;

            // [n][k] K-contiguous: (k, k+1) adjacent → single aligned u32.
            unsigned int b0 = *(const unsigned int*)&sB[n_col * b_stride + k0];
            unsigned int b1 = *(const unsigned int*)&sB[n_col * b_stride + k1];

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
}

/// Tensor-core pipelined BF16 dense GEMM: C[M,N] = A[M,K] · B[N,K]^T.
/// 128×32 tile (M×N), 8 warps, DM_STAGES-deep cp.async prefetch pipeline.
extern "C" __global__ void dense_gemm_bf16_pipelined(
    const __nv_bfloat16* __restrict__ A,   // [M, K] BF16 activations
    const __nv_bfloat16* __restrict__ B,   // [N, K] BF16 weights (read transposed)
    __nv_bfloat16* __restrict__ C,          // [M, N] BF16 output
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int cta_m = blockIdx.y * DM_M_TILE;
    const unsigned int cta_n = blockIdx.x * DM_N_TILE;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;   // 8 warps × 16 = 128 M-rows
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    // Pipelined smem. cp.async destinations (smem_A, smem_B) are __align__(16)
    // with 16-byte-aligned row strides so every 16-B cp.async.cg chunk is
    // naturally aligned. Per stage: smem_A 128*40*2 = 10240 B + smem_B
    // 32*40*2 = 2560 B = 12800 B → 25600 B for 2 stages. Well under 101 KB.
    __shared__ __align__(16) __nv_bfloat16 smem_A[DM_STAGES][DM_M_TILE][DM_A_STRIDE];
    __shared__ __align__(16) __nv_bfloat16 smem_B[DM_STAGES][DM_N_TILE][DM_B_STRIDE];

    // Single FP32 accumulator per N-tile quad (no two-level fold — pure BF16,
    // no block scale).
    float acc[DM_N_TILES_PER_WARP][4];
    #pragma unroll
    for (int i = 0; i < DM_N_TILES_PER_WARP; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f; acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    const unsigned int n_steps = (K + DM_K_STEP - 1) / DM_K_STEP;

    // A-tile cp.async: 128 rows × DM_K_STEP K-cols BF16. Each 16-B chunk = 8
    // BF16. 128×32/8 = 512 chunks → 2 chunks/thread (256 threads).
    const unsigned int a_chunks = (DM_M_TILE * DM_K_STEP) / 8;     // 512
    // B-tile cp.async: DM_N_TILE rows × DM_K_STEP K-cols BF16 = 32×32/8 = 128
    // chunks → ~0.5 chunk/thread.
    const unsigned int b_chunks = (DM_N_TILE * DM_K_STEP) / 8;     // 128

    // Issue cp.async loads for K-step `step` into double-buffer `stage`. The
    // copies are contiguous along K (the contiguous global axis for both A
    // [M,K] and B [N,K]). Bounds / K-tail fall back to a masked scalar copy.
    auto prefetch = [&](unsigned int step, unsigned int stage) {
        unsigned int k_base = step * DM_K_STEP;

        // ── A: contiguous 16-B (8 BF16) chunks along K ──
        #pragma unroll
        for (unsigned int c = threadIdx.x; c < a_chunks; c += DM_THREADS) {
            unsigned int row = (c * 8) / DM_K_STEP;          // 0..127
            unsigned int col = (c * 8) % DM_K_STEP;          // 0, 8, 16, 24
            unsigned int gr = cta_m + row;
            unsigned int gc = k_base + col;
            __nv_bfloat16* dst = &smem_A[stage][row][col];
            if (gr < M && gc + 8 <= K) {
                dm_cp_async_cg_16(dst, &A[(unsigned long long)gr * K + gc]);
            } else {
                #pragma unroll
                for (unsigned int e = 0; e < 8; e++) {
                    unsigned int gcol = gc + e;
                    dst[e] = (gr < M && gcol < K) ? A[(unsigned long long)gr * K + gcol]
                                                  : __float2bfloat16(0.0f);
                }
            }
        }

        // ── B: contiguous 16-B (8 BF16) chunks of K per N-row ──
        // smem_B[stage][n][k] mirrors global B[n, k_base + k] contiguously.
        #pragma unroll
        for (unsigned int c = threadIdx.x; c < b_chunks; c += DM_THREADS) {
            unsigned int nrow = (c * 8) / DM_K_STEP;         // 0..DM_N_TILE-1
            unsigned int kcol = (c * 8) % DM_K_STEP;         // 0, 8, 16, 24
            unsigned int gn = cta_n + nrow;
            unsigned int gk = k_base + kcol;
            __nv_bfloat16* dst = &smem_B[stage][nrow][kcol];
            if (gn < N && gk + 8 <= K) {
                dm_cp_async_cg_16(dst, &B[(unsigned long long)gn * K + gk]);
            } else {
                #pragma unroll
                for (unsigned int e = 0; e < 8; e++) {
                    unsigned int gke = gk + e;
                    dst[e] = (gn < N && gke < K) ? B[(unsigned long long)gn * K + gke]
                                                 : __float2bfloat16(0.0f);
                }
            }
        }
        dm_cp_async_commit();
    };

    // ── Software-pipelined main loop (DM_STAGES-deep cp.async) ──
    #pragma unroll
    for (unsigned int p = 0; p < DM_STAGES - 1; p++) {
        if (p < n_steps) {
            prefetch(p, p % DM_STAGES);
        }
    }

    for (unsigned int step = 0; step < n_steps; step++) {
        unsigned int cur = step % DM_STAGES;

        unsigned int ahead = step + (DM_STAGES - 1);
        if (ahead < n_steps) {
            prefetch(ahead, ahead % DM_STAGES);
        }
        unsigned int committed = min(n_steps, DM_STAGES + step);
        unsigned int target = committed - (step + 1);
        dm_cp_async_wait_le(target);
        __syncthreads();   // A/B for `cur` resident for all threads

        dm_mma_kstep(&smem_A[cur][0][0], &smem_B[cur][0][0],
                     acc, warp_m_offset, group_id, tid);
        __syncthreads();   // done reading smem_*[cur]; safe for next reuse
    }

    // ── Store C tile: f32 accumulators → BF16 output ──
    #pragma unroll
    for (int n_tile = 0; n_tile < DM_N_TILES_PER_WARP; n_tile++) {
        unsigned int base_n = cta_n + n_tile * 8;
        unsigned int col0 = base_n + (tid * 2);
        unsigned int col1 = col0 + 1;
        unsigned int row0 = cta_m + warp_m_offset + group_id;
        unsigned int row1 = row0 + 8;

        if (row0 < M && col0 < N) C[row0 * N + col0] = __float2bfloat16(acc[n_tile][0]);
        if (row0 < M && col1 < N) C[row0 * N + col1] = __float2bfloat16(acc[n_tile][1]);
        if (row1 < M && col0 < N) C[row1 * N + col0] = __float2bfloat16(acc[n_tile][2]);
        if (row1 < M && col1 < N) C[row1 * N + col1] = __float2bfloat16(acc[n_tile][3]);
    }
}

// Fused SiLU(gate) * up activation — vectorized 2-wide BF16 loads/stores.
// Input: [N, inter_size*2] where first half is gate, second half is up.
// Output: [N, inter_size]
// out[i] = silu(gate[i]) * up[i]  where silu(x) = x * sigmoid(x)
extern "C" __global__ void fused_silu_mul(
    const __nv_bfloat16* __restrict__ gate_up,  // [num_tokens, inter_size * 2]
    __nv_bfloat16* __restrict__ output,          // [num_tokens, inter_size]
    unsigned int num_tokens,
    unsigned int inter_size
) {
    // Each thread processes 2 elements (vectorized BF16x2)
    unsigned int idx2 = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int half_total = (num_tokens * inter_size) / 2;
    if (idx2 >= half_total) return;

    // Map linear index to (token, col_pair)
    unsigned int half_inter = inter_size / 2;
    unsigned int token = idx2 / half_inter;
    unsigned int col_pair = idx2 % half_inter;

    // Vectorized loads: 2 BF16 per 32-bit read
    const unsigned int* gate32 = (const unsigned int*)(gate_up + token * (inter_size * 2));
    const unsigned int* up32 = (const unsigned int*)(gate_up + token * (inter_size * 2) + inter_size);
    unsigned int* out32 = (unsigned int*)(output + token * inter_size);

    unsigned int g_packed = gate32[col_pair];
    unsigned int u_packed = up32[col_pair];

    float g0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(g_packed & 0xFFFF)));
    float g1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(g_packed >> 16)));
    float u0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(u_packed & 0xFFFF)));
    float u1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(u_packed >> 16)));

    // SiLU(gate) * up for both elements
    float sg0 = 1.0f / (1.0f + __expf(-g0));
    float sg1 = 1.0f / (1.0f + __expf(-g1));
    float r0 = g0 * sg0 * u0;
    float r1 = g1 * sg1 * u1;

    // Vectorized store
    unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(r0));
    unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(r1));
    out32[col_pair] = lo | (hi << 16);
}
