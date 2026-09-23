// SPDX-License-Identifier: AGPL-3.0-only

// Atlas BF16-weight small-M GEMV on TENSOR CORES — the energy-efficient sibling
// of `dense_gemv_bf16` / `dense_gemv_bf16_batchm` for the MTP drafter's
// M = 1..32 propose rows. Structure after `w4a16_gemv_tc.cu` (the NVFP4 TEP
// GEMV); BF16 weights need no dequant at all.
//
//   C[t, n] = sum_k A[t, k] * W[n, k]        t < M <= 8*NB
//
// ── Why this exists (energy, not speed) ─────────────────────────────────────
// The CUDA-core BF16 GEMV family converts every weight to FP32 and, per ROW,
// converts the activation and runs FMUL+FADD (the dir builds --fmad=false):
// ~13 thread-instructions per weight at M=4 and ~25 at M=8, so the GPU rail
// stays issue-bound while the kernel streams at the DRAM roofline. Here one
// `mma.sync.m16n8k16` consumes 256 weights for up to 8 tokens with no
// per-weight ALU work: the raw BF16 words go straight into the A fragment.
//
// ── Operand placement ───────────────────────────────────────────────────────
// WEIGHTS on the 16-row A side, TOKENS on the 8-wide B side. At M <= 8 that is
// one MMA per 16x16 weight block (tokens-on-A, as in the NVFP4 kernel, needs
// two: its B side is only 8 weight rows wide). M in 9..16 / 17..32 reuses each
// A fragment for 2 / 4 token tiles. A token tile that is entirely past M is
// skipped (warp-uniform), so the MMA count tracks the live rows.
//
// ── K permutation (why no repack) ───────────────────────────────────────────
// Thread (g = lane/4, t = lane%4) loads two 16-byte pieces (8 k each) of weight
// rows g and g+8 of its 16-row tile, at k offsets 8t and 32+8t of a 64-k block:
// each load instruction reads 64 contiguous bytes per row across the quad (two
// full 32-B sectors), and the pair covers the row's whole 128-byte line, in the
// checkpoint's own [N, K] layout. Its 8 BF16x2 words w0..w7 (piece 0 then
// piece 1) feed 4 MMAs: MMA j takes w(2j) in the k-slot pair {2t,2t+1} and
// w(2j+1) in {2t+8,2t+9}. The B fragment (token g) is built from the SAME
// 16 k of activation row g in the same order. A k slot is owned by lane%4 == t
// in both fragments, so the permutation is identical on both operands and the
// dot product is unchanged. No shuffles, no byte permutes.
//
// ── Numerics ────────────────────────────────────────────────────────────────
// BF16 x BF16 products are exact in FP32; accumulation is FP32 in tensor-core
// order. The result differs from the CUDA-core GEMV only in FP32 summation
// order, the same class of difference as `dense_gemm_bf16_pipelined` (also
// mma.sync m16n8k16), which the drafter already uses above 8 rows.
//
// ── Determinism ─────────────────────────────────────────────────────────────
// The TC_WARPS warps of a CTA split K (interleaved 64-k blocks) for the same
// 16*NT weight rows and reduce through shared memory in a FIXED warp order.
// There are no atomics, so the result is bitwise reproducible run to run and
// independent of scheduling, and it does not depend on M (a token's partials
// never mix with another token's).
//
// ── Layout / launch ─────────────────────────────────────────────────────────
// A:[M,K] BF16 contiguous, W:[N,K] BF16, C: M rows at C + t*out_stride — the
// `dense_gemv_bf16_batchm` contract, argument for argument.
// Requires K % 64 == 0 (the launcher checks; otherwise the caller keeps the
// CUDA-core kernel). Any N: weight rows past N load zeros and are never stored.
// Grid: (ceil(N / (16*NT)), 1, 1)  Block: (TC_WARPS*32, 1, 1) = 256.

#include <cuda_bf16.h>
#include <stdint.h>

#define DTC_WARPS 8
#define DTC_KB 64   // k per warp step: 16 per thread, 4 threads per quad

__device__ __forceinline__ void dtc_mma(float (&d)[4], uint32_t a0, uint32_t a1,
                                        uint32_t a2, uint32_t a3, uint32_t b0,
                                        uint32_t b1) {
    asm(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Streamed weight load: read once per launch, so keep it out of L1 and leave
// L1 to the activation rows every CTA re-reads.
__device__ __forceinline__ uint4 dtc_ld_stream(const void* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

__device__ __forceinline__ uint32_t dtc_word(const uint4& v, int i) {
    return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

template <int NB, int NT, int KU>
__device__ __forceinline__ void dense_gemv_bf16_tc_impl(
    const __nv_bfloat16* __restrict__ A,  // [M, K]
    const __nv_bfloat16* __restrict__ W,  // [N, K]
    __nv_bfloat16* __restrict__ C,        // rows at C + t*out_stride
    unsigned int M, unsigned int N, unsigned int K, unsigned int out_stride)
{
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int g = lane >> 2;
    const unsigned int t = lane & 3u;
    const unsigned int n0 = blockIdx.x * (16u * NT);
    const unsigned int num_kb = K / DTC_KB;

    // Weight rows g and g+8 of each 16-row tile; a row past N loads zeros.
    bool live[NT][2];
    const __nv_bfloat16* wrow[NT][2];
    #pragma unroll
    for (int i = 0; i < NT; i++) {
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            const unsigned int n = n0 + (unsigned int)i * 16u + (unsigned int)h * 8u + g;
            live[i][h] = n < N;
            wrow[i][h] = W + (unsigned long long)(live[i][h] ? n : 0u) * K + t * 8u;
        }
    }
    // Token tiles: tile b is live iff its first row is < M (warp-uniform).
    // Within a live tile, token row b*8+g past M feeds zeros.
    bool tile_on[NB];
    bool tok_live[NB];
    const __nv_bfloat16* arow[NB];
    #pragma unroll
    for (int b = 0; b < NB; b++) {
        const unsigned int tok = (unsigned int)b * 8u + g;
        tile_on[b] = (unsigned int)b * 8u < M;
        tok_live[b] = tok < M;
        arow[b] = A + (unsigned long long)(tok_live[b] ? tok : 0u) * K + t * 8u;
    }

    float acc[NT][NB][4];
    #pragma unroll
    for (int i = 0; i < NT; i++) {
        #pragma unroll
        for (int b = 0; b < NB; b++) {
            #pragma unroll
            for (int c = 0; c < 4; c++) acc[i][b][c] = 0.0f;
        }
    }

    // KU k-blocks per trip: all their weight loads are issued before any MMA,
    // so each thread keeps KU*NT*2 rows x 32 B of weights in flight.
    for (unsigned int kb0 = warp; kb0 < num_kb; kb0 += DTC_WARPS * KU) {
        uint4 w[KU][NT][2][2];
        #pragma unroll
        for (int u = 0; u < KU; u++) {
            const unsigned int kb = kb0 + (unsigned int)u * DTC_WARPS;
            #pragma unroll
            for (int i = 0; i < NT; i++) {
                #pragma unroll
                for (int h = 0; h < 2; h++) {
                    if (live[i][h] && kb < num_kb) {
                        const __nv_bfloat16* p = wrow[i][h] + kb * DTC_KB;
                        w[u][i][h][0] = dtc_ld_stream(p);
                        w[u][i][h][1] = dtc_ld_stream(p + 32);
                    } else {
                        w[u][i][h][0] = make_uint4(0u, 0u, 0u, 0u);
                        w[u][i][h][1] = make_uint4(0u, 0u, 0u, 0u);
                    }
                }
            }
        }
        #pragma unroll
        for (int u = 0; u < KU; u++) {
            const unsigned int kb = kb0 + (unsigned int)u * DTC_WARPS;
            if (kb >= num_kb) break;
            #pragma unroll
            for (int b = 0; b < NB; b++) {
                if (!tile_on[b]) continue;
                uint4 av[2];
                if (tok_live[b]) {
                    const uint4* ap = (const uint4*)(arow[b] + kb * DTC_KB);
                    av[0] = ap[0];
                    av[1] = ap[4];   // +32 k: the same split as the weights
                } else {
                    av[0] = make_uint4(0u, 0u, 0u, 0u);
                    av[1] = make_uint4(0u, 0u, 0u, 0u);
                }
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    // Words 2j and 2j+1 of this thread's 16 k (j<2: first piece).
                    const uint32_t b0 = dtc_word(av[j >> 1], (2 * j) & 3);
                    const uint32_t b1 = dtc_word(av[j >> 1], (2 * j + 1) & 3);
                    #pragma unroll
                    for (int i = 0; i < NT; i++) {
                        const uint32_t a0 = dtc_word(w[u][i][0][j >> 1], (2 * j) & 3);
                        const uint32_t a2 = dtc_word(w[u][i][0][j >> 1], (2 * j + 1) & 3);
                        const uint32_t a1 = dtc_word(w[u][i][1][j >> 1], (2 * j) & 3);
                        const uint32_t a3 = dtc_word(w[u][i][1][j >> 1], (2 * j + 1) & 3);
                        dtc_mma(acc[i][b], a0, a1, a2, a3, b0, b1);
                    }
                }
            }
        }
    }

    // Fixed-order split-K reduction across the CTA's warps.
    __shared__ float red[DTC_WARPS][NT * NB][4][32];
    #pragma unroll
    for (int i = 0; i < NT; i++) {
        #pragma unroll
        for (int b = 0; b < NB; b++) {
            #pragma unroll
            for (int c = 0; c < 4; c++) red[warp][i * NB + b][c][lane] = acc[i][b][c];
        }
    }
    __syncthreads();

    // D fragment: d0,d1 = (weight row g, tokens 2t, 2t+1); d2,d3 = row g+8.
    for (unsigned int f = warp; f < (unsigned int)(NT * NB); f += DTC_WARPS) {
        const unsigned int i = f / NB;
        const unsigned int b = f % NB;
        if (b * 8u >= M) continue;
        float r[4];
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            float v = red[0][f][c][lane];
            #pragma unroll
            for (int ww = 1; ww < DTC_WARPS; ww++) v += red[ww][f][c][lane];
            r[c] = v;
        }
        const unsigned int tok0 = b * 8u + t * 2u;
        const unsigned int n_lo = n0 + i * 16u + g;
        const unsigned int n_hi = n_lo + 8u;
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            const unsigned int tok = tok0 + (unsigned int)(c & 1);
            const unsigned int n = (c < 2) ? n_lo : n_hi;
            if (tok < M && n < N)
                C[(unsigned long long)tok * out_stride + n] = __float2bfloat16_rn(r[c]);
        }
    }
}

#define DTC_ENTRY(NAME, NB, NT, KU)                                                     \
    extern "C" __global__ __launch_bounds__(DTC_WARPS * 32) void NAME(                   \
        const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ W,         \
        __nv_bfloat16* __restrict__ C, unsigned int M, unsigned int N, unsigned int K,    \
        unsigned int out_stride) {                                                        \
        dense_gemv_bf16_tc_impl<NB, NT, KU>(A, W, C, M, N, K, out_stride);               \
    }

// Production entries, one per token-tile count (M <= 8 / 16 / 32). Each
// covers every M up to its cap: a token tile past M is skipped whole.
//
// Geometry measured on GB10 (dgx1, one full 27B draft position = 849 MB of
// cold BF16 weights, 2026-09-23). What moves the time is weight bytes in
// flight per thread, not occupancy: KU=8 (512 B/thread, 1 CTA/SM) beat KU=4
// (2 CTAs/SM) by 3.5-4.4%, while KU=2 at 4 CTAs/SM and KU=3/4 at 3 CTAs/SM
// were slower than KU=4, 32 weight rows per CTA (NT=2) was slower than KU=8,
// and an L2::256B prefetch hint changed nothing. Against the CUDA-core
// `dense_gemv_bf16_batchm` at M=2/4/8: 3554/3583/3583 us vs 3487/3564/3631 us,
// at 97/117/134 mJ vs 149/168/264 mJ per position. tc16 at KU=8: 3602 us
// (KU=4: 3737). tc32 at KU=4: 3803 us (KU=2: 3885; KU=8 does not fit).
DTC_ENTRY(dense_gemv_bf16_tc8, 1, 1, 8)
DTC_ENTRY(dense_gemv_bf16_tc16, 2, 1, 8)
DTC_ENTRY(dense_gemv_bf16_tc32, 4, 1, 4)
