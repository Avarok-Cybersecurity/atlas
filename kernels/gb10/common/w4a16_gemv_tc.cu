// SPDX-License-Identifier: AGPL-3.0-only

// Atlas W4A16 small-M GEMV on TENSOR CORES — the energy-efficient sibling of
// `w4a16_gemv_batch{4..8}` (w4a16_gemv.cu) for M = 1..16 decode / MTP-verify
// rows.
//
// C[m, n] = sum_k A[m, k] * dequant(B_fp4[n, k])        (m < M <= 16)
//
// ── Why this exists (energy, not speed) ─────────────────────────────────────
// The CUDA-core batch tiers do, per 8 packed weight bytes and per ROW, two
// 16-B activation loads and 17 FP32 FMAs, plus 16 shared-LUT loads per chunk.
// At M=4 that is ~160 thread-instructions and ~128 B of L1 activation traffic
// per 8 B of weight streamed from DRAM. On GB10 that regime draws 72-86 W on
// the GPU rail against ~50 W for the tensor-core paths used above 8 rows,
// for the same bytes streamed (TEP stream B/C, 2026-09-23).
//
// Here the per-weight work is M-INDEPENDENT: every 8 E2M1 nibbles become four
// BF16x2 MMA operands with ~3 integer ops per pair plus one BF16x2 multiply by
// the group scale, and one `mma.sync.m16n8k16` covers all (up to 16) rows.
// Activation fragments are loaded once per K block and reused across NT
// column tiles, so L1 activation traffic is ~1/NT of the weight traffic
// instead of ~16x it.
//
// ── Dequant (exact) ─────────────────────────────────────────────────────────
// An E2M1 nibble s|e1 e0|m placed at BF16 bits 15 | 8..7 | 6 reads as
// e2m1 * 2^-126 (e>=1 is a normal with exponent field e; e=0,m=1 is the BF16
// subnormal 2^-127). The FP8-E4M3 group scale is converted to BF16 times 2^100
// (exact: E4M3 has 3 mantissa bits and 2^100 keeps every value normal), and
// one BF16x2 multiply forms  w' = e2m1 * scale * 2^-26  — EXACT, because the
// product of a 2-bit and a 4-bit significand fits BF16's 8 bits and lands in
// the normal range. The MMA accumulates A*w' in FP32; the epilogue applies
// scale2 * 2^26. The only numeric difference from the CUDA-core tiers is the
// FP32 summation order (tensor-core reduction instead of the fmaf chain),
// i.e. the same class of difference as the tile GEMMs used above 8 rows.
//
// ── K permutation (why no weight repack is needed) ──────────────────────────
// A dot product is invariant under any permutation of k applied to BOTH
// operands. Thread (g = lane/4, t = lane%4) reads 16 contiguous weight bytes
// of row n = tile*8 + g (32 k values; the quad covers 128 contiguous k, so a
// quad's load is 64 contiguous bytes) in the checkpoint's own [N, K/2] layout.
// Each 32-bit word q (k0..k7) yields the operand pairs
//     P0=(k3,k7)  P1=(k2,k6)  P2=(k1,k5)  P3=(k0,k4)
// and the activation fragment for the same slot is built with the matching
// byte permutation of the thread's own row-g activations. The MMA's k slot
// 2t/2t+1/2t+8/2t+9 is owned by lane%4 == t in BOTH the A and B fragments,
// so the two permutations agree by construction.
//
// ── Layout / launch ─────────────────────────────────────────────────────────
// A:[M,K] BF16, B_packed:[N,K/2], B_scale:[N,K/16] FP8-E4M3, scale2 FP32,
// C:[M,N] BF16 — the `w4a16_gemv_batchm` contract, argument for argument.
// Requires K % 128 == 0 and N % 8 == 0 (the launcher checks and otherwise
// keeps the CUDA-core tier).
// Grid: (ceil(N / (8*NT)), 1, 1)  Block: (TC_WARPS*32, 1, 1) = 256
// (tc8: NT=1 -> ceil(N/8) CTAs; tc16: NT=2 -> ceil(N/16) CTAs).
// The TC_WARPS warps of a CTA split K (interleaved 128-k blocks) for the same
// 8*NT columns and reduce through shared memory in a FIXED order: the result
// is deterministic run to run and independent of scheduling.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#define TC_WARPS 8

__device__ __forceinline__ void w4tc_mma(float (&d)[4], uint32_t a0, uint32_t a1,
                                         uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
    asm(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Nibbles at bits 12..15 and 28..31 of q -> BF16x2 {lo, hi} = e2m1 * 2^-126.
__device__ __forceinline__ uint32_t w4tc_e2m1x2(uint32_t q) {
    return (q & 0x80008000u) | ((q & 0x70007000u) >> 6);
}

__device__ __forceinline__ uint32_t w4tc_bmul2(uint32_t a, uint32_t b) {
    uint32_t d;
    asm("mul.rn.bf16x2 %0, %1, %2;\n" : "=r"(d) : "r"(a), "r"(b));
    return d;
}

// FP8-E4M3 scale byte -> BF16 (scale * 2^100) broadcast to both halves. Exact.
__device__ __forceinline__ uint32_t w4tc_scale_x2(uint32_t sb) {
    __nv_fp8_e4m3 f;
    *(unsigned char*)&f = (unsigned char)sb;
    const float v = (float)f * 0x1p100f;
    const __nv_bfloat16 h = __float2bfloat16_rn(v);
    const uint32_t u = (uint32_t)(*(const unsigned short*)&h);
    return u | (u << 16);
}

template <int MT, int NT, int KU>
__device__ __forceinline__ void w4a16_gemv_tc_impl(
    const __nv_bfloat16* __restrict__ A,          // [M, K]
    const unsigned char* __restrict__ B_packed,   // [N, K/2]
    const unsigned char* __restrict__ B_scale,    // [N, K/16] FP8-E4M3
    const float scale2,
    __nv_bfloat16* __restrict__ C,                // [M, N]
    unsigned int M, unsigned int N, unsigned int K)
{
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int g = lane >> 2;
    const unsigned int t = lane & 3u;
    const unsigned int n0 = blockIdx.x * (8u * NT);
    const unsigned int half_K = K >> 1;
    const unsigned int num_groups = K >> 4;
    const unsigned int num_kb = K >> 7;

    const bool lo_live = g < M;
    const bool hi_live = (MT > 8) && (g + 8u < M);
    const uint4* a_lo_row = (const uint4*)(A + (unsigned long long)g * K);
    const uint4* a_hi_row = (const uint4*)(A + (unsigned long long)(g + 8u) * K);

    bool tile_live[NT];
    const unsigned char* wrow[NT];
    const unsigned char* srow[NT];
    #pragma unroll
    for (int i = 0; i < NT; i++) {
        const unsigned int n = n0 + (unsigned int)i * 8u + g;
        tile_live[i] = (n0 + (unsigned int)i * 8u) < N;   // N % 8 == 0: whole tile in or out
        wrow[i] = B_packed + (unsigned long long)n * half_K + t * 16u;
        srow[i] = B_scale + (unsigned long long)n * num_groups + t * 2u;
    }

    float acc[NT][4];
    #pragma unroll
    for (int i = 0; i < NT; i++) {
        #pragma unroll
        for (int c = 0; c < 4; c++) acc[i][c] = 0.0f;
    }

    // KU k-blocks per trip: all their weight/scale loads are issued before any
    // dequant, so each thread keeps KU*NT 16-B weight loads in flight.
    for (unsigned int kb0 = warp; kb0 < num_kb; kb0 += TC_WARPS * KU) {
        uint4 w[KU][NT];
        uint32_t sc[KU][NT];
        #pragma unroll
        for (int u = 0; u < KU; u++) {
            const unsigned int kb = kb0 + (unsigned int)u * TC_WARPS;
            #pragma unroll
            for (int i = 0; i < NT; i++) {
                if (tile_live[i] && kb < num_kb) {
                    w[u][i] = *(const uint4*)(wrow[i] + kb * 64u);
                    sc[u][i] = *(const unsigned short*)(srow[i] + kb * 8u);
                } else {
                    w[u][i] = make_uint4(0u, 0u, 0u, 0u);
                    sc[u][i] = 0u;
                }
            }
        }
        #pragma unroll
        for (int u = 0; u < KU; u++) {
            const unsigned int kb = kb0 + (unsigned int)u * TC_WARPS;
            if (kb >= num_kb) break;
            // This thread's 32 activations of row g (and g+8): k = kb*128 + t*32 ..
            uint4 al[4], ah[4];
            const unsigned int abase = kb * 16u + t * 4u;   // in uint4 (8 bf16) units
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                al[j] = lo_live ? a_lo_row[abase + j] : make_uint4(0u, 0u, 0u, 0u);
                if (MT > 8) ah[j] = hi_live ? a_hi_row[abase + j] : make_uint4(0u, 0u, 0u, 0u);
            }
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                // Activation pairs matching P0..P3 of weight word j.
                const uint32_t l37 = __byte_perm(al[j].y, al[j].w, 0x7632);  // (k3,k7)
                const uint32_t l26 = __byte_perm(al[j].y, al[j].w, 0x5410);  // (k2,k6)
                const uint32_t l15 = __byte_perm(al[j].x, al[j].z, 0x7632);  // (k1,k5)
                const uint32_t l04 = __byte_perm(al[j].x, al[j].z, 0x5410);  // (k0,k4)
                uint32_t h37 = 0u, h26 = 0u, h15 = 0u, h04 = 0u;
                if (MT > 8) {
                    h37 = __byte_perm(ah[j].y, ah[j].w, 0x7632);
                    h26 = __byte_perm(ah[j].y, ah[j].w, 0x5410);
                    h15 = __byte_perm(ah[j].x, ah[j].z, 0x7632);
                    h04 = __byte_perm(ah[j].x, ah[j].z, 0x5410);
                }
                #pragma unroll
                for (int i = 0; i < NT; i++) {
                    const uint32_t q = (j == 0) ? w[u][i].x : (j == 1) ? w[u][i].y
                                     : (j == 2) ? w[u][i].z : w[u][i].w;
                    const uint32_t s = w4tc_scale_x2((sc[u][i] >> ((j >> 1) * 8)) & 0xFFu);
                    const uint32_t p0 = w4tc_bmul2(w4tc_e2m1x2(q), s);
                    const uint32_t p1 = w4tc_bmul2(w4tc_e2m1x2(q << 4), s);
                    const uint32_t p2 = w4tc_bmul2(w4tc_e2m1x2(q << 8), s);
                    const uint32_t p3 = w4tc_bmul2(w4tc_e2m1x2(q << 12), s);
                    w4tc_mma(acc[i], l37, h37, l26, h26, p0, p1);
                    w4tc_mma(acc[i], l15, h15, l04, h04, p2, p3);
                }
            }
        }
    }

    // Fixed-order split-K reduction across the CTA's warps.
    __shared__ float red[TC_WARPS][NT][4][32];
    #pragma unroll
    for (int i = 0; i < NT; i++) {
        #pragma unroll
        for (int c = 0; c < 4; c++) red[warp][i][c][lane] = acc[i][c];
    }
    __syncthreads();

    const float sfin = scale2 * 0x1p26f;
    for (unsigned int i = warp; i < (unsigned int)NT; i += TC_WARPS) {
        if (n0 + i * 8u >= N) continue;
        float r[4];
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            float v = red[0][i][c][lane];
            #pragma unroll
            for (int ww = 1; ww < TC_WARPS; ww++) v += red[ww][i][c][lane];
            r[c] = v * sfin;
        }
        const unsigned int col = n0 + i * 8u + t * 2u;
        if (g < M) {
            *(__nv_bfloat162*)(C + (unsigned long long)g * N + col) =
                __floats2bfloat162_rn(r[0], r[1]);
        }
        if (MT > 8 && g + 8u < M) {
            *(__nv_bfloat162*)(C + (unsigned long long)(g + 8u) * N + col) =
                __floats2bfloat162_rn(r[2], r[3]);
        }
    }
}

#define W4TC_ENTRY(NAME, MT, NT, KU)                                                   \
    extern "C" __global__ __launch_bounds__(TC_WARPS * 32) void NAME(                    \
        const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ B_packed,  \
        const unsigned char* __restrict__ B_scale, const float scale2,                    \
        __nv_bfloat16* __restrict__ C, unsigned int M, unsigned int N, unsigned int K) {  \
        w4a16_gemv_tc_impl<MT, NT, KU>(A, B_packed, B_scale, scale2, C, M, N, K);        \
    }

// Shapes measured on GB10 (tcbench, cold weights, real 27B projections):
// M<=8 runs NT=1 x KU=8 (8 columns per CTA, 8 k-blocks of loads in flight per
// thread) — at parity with w4a16_gemv_batch4 in time at M=1..4 and 20% faster
// than batch8_rt2 at M=8, at 51 W instead of 77-82 W.
// 9<=M<=16 runs NT=2 x KU=2 — 2.4x faster than w4a16_gemv_batch16.
// M <= 8: the A fragment's rows 8..15 are constant zero.
W4TC_ENTRY(w4a16_gemv_tc8, 8, 1, 8)
W4TC_ENTRY(w4a16_gemv_tc16, 16, 2, 2)
