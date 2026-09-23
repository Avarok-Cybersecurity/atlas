// SPDX-License-Identifier: AGPL-3.0-only

// Atlas W4A4 small-M projection on FP4 TENSOR CORES (Blackwell block-scale
// MMA `kind::mxf4nvf4`, E2M1 x E2M1, UE4M3 per-16 scales). This is the energy
// sibling of `w4a16_gemv_tc.cu` for 1..32 decode / MTP-verify rows.
//
// ── Why ─────────────────────────────────────────────────────────────────────
// bf16 tensor-core MMA power on GB10 scales with LIVE rows (TEP 2026-09-23:
// tc8 draws 46/51/59 W at 1/4/8 rows, and tc16 with 9..16 live rows is +19%
// J/token at C=4). An FP4 MMA moves 4x fewer operand bits per MAC and needs
// NO weight dequant at all. The checkpoint's NVFP4 bytes ARE the MMA
// B/A operand, and its E4M3 group scales ARE the UE4M3 block scales. vLLM
// runs NVFP4 layers this way at every M.
//
// ── Operands ────────────────────────────────────────────────────────────────
// Weights sit on the MMA's 16-row side (A, m16), tokens on the 8-wide side
// (B, n8), so an 8-row verify step fills the B tile. m16n8k64 fragments
// (4-bit elements, 8 per 32-bit register; g = lane/4, t = lane%4):
//   A: a0 = row g   ints t, a1 = row g+8 ints t, a2 = row g ints t+4,
//      a3 = row g+8 ints t+4   (int c of a k64 block = k [8c, 8c+8))
//   B: b0 = col g ints t, b1 = col g ints t+4
// With the checkpoint's natural packing (byte j = k 2j low | 2j+1 high), int c
// of a row's 32-byte k64 block holds k [8c, 8c+8). The scale block i
// (k-slots [16i, 16i+16)) is then ints {2i, 2i+1}, i.e. exactly NVFP4 group
// i. Weights load straight from [N, K/2] with no repack, and the 4 group-scale
// bytes of a k64 block are one aligned 32-bit load from [N, K/16].
// Block-scale registers with selector {0,0} (as llama.cpp's
// mma_block_scaled_fp4): lane supplies A's scales for row
// g + (t & 1) * 8 and B's for column g.
//
// ── Activations (per-row dynamic NVFP4) ─────────────────────────────────────
// `w4a4_quant_rows` quantises each BF16 row with a per-ROW FP32 global scale
// gs = amax(row) / (6 * 448) (so the largest group scale lands at the E4M3
// maximum), per-16 E4M3 group scales s = e4m3_rn(amax(group) / 6 / gs), and
// E2M1 round-to-nearest-even of x / (s * gs). Output uses the weights'
// natural packing. This is DYNAMIC per-row scaling rather than the
// checkpoint's static per-tensor input_scale: the projections this serves
// (attention q/k/v/o, GDN in/out) are FP8 in the 27B checkpoint and are
// requantised to NVFP4 at load, so there is no input_scale for them.
//
// C[m, n] = scale2 * gs[m] * sum_k (qa * sa)[m, k] * (qw * sw)[n, k]
//
// Grid: (ceil(N/16), 1, 1), block 256 (8 warps split K over k64 blocks,
// fixed-order reduction, deterministic). Requires K % 128 == 0.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#define W4A4_WARPS 8

__device__ __forceinline__ void w4a4_mma(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2,
                                         uint32_t a3, uint32_t b0, uint32_t b1, uint32_t sa,
                                         uint32_t sb) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
        "%10, {0, 0}, %11, {0, 0};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
#endif
}

template <int MB, int KU>
__device__ __forceinline__ void w4a4_gemv_mx_impl(
    const unsigned char* __restrict__ Aq,   // [M, K/2]  NVFP4 activations, FRAGMENT order
    const unsigned char* __restrict__ As,   // [M, K/16] E4M3 group scales, FRAGMENT order
    const float* __restrict__ Ag,           // [M]       per-row global scale
    const unsigned char* __restrict__ Bq,   // [N, K/2]  NVFP4 weights (checkpoint layout)
    const unsigned char* __restrict__ Bs,   // [N, K/16] E4M3 group scales (checkpoint layout)
    const float scale2,
    __nv_bfloat16* __restrict__ C,          // [M, N]
    unsigned int M, unsigned int N, unsigned int K)
{
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int g = lane >> 2;
    const unsigned int t = lane & 3u;
    const bool odd = (t & 1u) != 0u;
    const unsigned int n0 = blockIdx.x * 16u;
    const unsigned int half_K = K >> 1;
    const unsigned int groups = K >> 4;
    const unsigned int num_c = K >> 7;     // k128 chunks: 64 weight bytes per row

    const unsigned int r0 = n0 + g, r1 = n0 + g + 8u, rs = n0 + g + (odd ? 8u : 0u);
    const bool l0 = r0 < N, l1 = r1 < N, ls = rs < N;
    const unsigned char* w0 = Bq + (unsigned long long)r0 * half_K + t * 16u;
    const unsigned char* w1 = Bq + (unsigned long long)r1 * half_K + t * 16u;
    const unsigned char* ws = Bs + (unsigned long long)rs * groups;

    bool tl[MB];
    const unsigned char* aq[MB];
    const unsigned char* as[MB];
    #pragma unroll
    for (int j = 0; j < MB; j++) {
        const unsigned int tok = (unsigned int)j * 8u + g;
        tl[j] = tok < M;
        aq[j] = Aq + (unsigned long long)tok * half_K + t * 16u;
        as[j] = As + (unsigned long long)tok * groups;
    }

    float acc[MB][4];
    #pragma unroll
    for (int j = 0; j < MB; j++) {
        #pragma unroll
        for (int c = 0; c < 4; c++) acc[j][c] = 0.0f;
    }

    for (unsigned int c0 = warp; c0 < num_c; c0 += W4A4_WARPS * KU) {
        uint4 wl[KU], wh[KU], b[KU][MB];
        uint2 sw[KU], sb[KU][MB];
        #pragma unroll
        for (int u = 0; u < KU; u++) {
            const unsigned int c = c0 + (unsigned int)u * W4A4_WARPS;
            const bool live = c < num_c;
            const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
            const uint2 z2 = make_uint2(0u, 0u);
            wl[u] = (live && l0) ? *(const uint4*)(w0 + c * 64u) : z4;
            wh[u] = (live && l1) ? *(const uint4*)(w1 + c * 64u) : z4;
            sw[u] = (live && ls) ? *(const uint2*)(ws + c * 8u) : z2;
            #pragma unroll
            for (int j = 0; j < MB; j++) {
                b[u][j] = (live && tl[j]) ? *(const uint4*)(aq[j] + c * 64u) : z4;
                sb[u][j] = (live && tl[j]) ? *(const uint2*)(as[j] + c * 8u) : z2;
            }
        }
        #pragma unroll
        for (int u = 0; u < KU; u++) {
            if (c0 + (unsigned int)u * W4A4_WARPS >= num_c) break;
            // Thread t holds groups 2t, 2t+1 of each row (ints 0,1 | 2,3).
            // MMA0 takes scale blocks {G0,G4,G2,G6}, MMA1 {G1,G5,G3,G7}: each
            // block's two ints are split across the pair (t, t^1), so ONE
            // shfl_xor(1) per row per MMA completes the fragment.
            const uint32_t x0l = __shfl_xor_sync(0xFFFFFFFFu, odd ? wl[u].x : wl[u].y, 1);
            const uint32_t x0h = __shfl_xor_sync(0xFFFFFFFFu, odd ? wh[u].x : wh[u].y, 1);
            const uint32_t x1l = __shfl_xor_sync(0xFFFFFFFFu, odd ? wl[u].z : wl[u].w, 1);
            const uint32_t x1h = __shfl_xor_sync(0xFFFFFFFFu, odd ? wh[u].z : wh[u].w, 1);
            const uint32_t a00 = odd ? x0l : wl[u].x, a02 = odd ? wl[u].y : x0l;
            const uint32_t a01 = odd ? x0h : wh[u].x, a03 = odd ? wh[u].y : x0h;
            const uint32_t a10 = odd ? x1l : wl[u].z, a12 = odd ? wl[u].w : x1l;
            const uint32_t a11 = odd ? x1h : wh[u].z, a13 = odd ? wh[u].w : x1h;
            const uint32_t s0 = __byte_perm(sw[u].x, sw[u].y, 0x6240);
            const uint32_t s1 = __byte_perm(sw[u].x, sw[u].y, 0x7351);
            #pragma unroll
            for (int j = 0; j < MB; j++) {
                w4a4_mma(acc[j], a00, a01, a02, a03, b[u][j].x, b[u][j].y, s0, sb[u][j].x);
                w4a4_mma(acc[j], a10, a11, a12, a13, b[u][j].z, b[u][j].w, s1, sb[u][j].y);
            }
        }
    }

    __shared__ float red[W4A4_WARPS][MB][4][32];
    #pragma unroll
    for (int j = 0; j < MB; j++) {
        #pragma unroll
        for (int c = 0; c < 4; c++) red[warp][j][c][lane] = acc[j][c];
    }
    __syncthreads();

    // D (16 x 8): c0,c1 = weight row g, tokens 2t, 2t+1; c2,c3 = row g+8.
    for (unsigned int j = warp; j < (unsigned int)MB; j += W4A4_WARPS) {
        float r[4];
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            float v = red[0][j][c][lane];
            #pragma unroll
            for (int ww = 1; ww < W4A4_WARPS; ww++) v += red[ww][j][c][lane];
            r[c] = v;
        }
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            const unsigned int n = (c < 2) ? r0 : r1;
            const unsigned int tok = j * 8u + t * 2u + (unsigned int)(c & 1);
            if (n < N && tok < M) {
                C[(unsigned long long)tok * N + n] = __float2bfloat16_rn(r[c] * (Ag[tok] * scale2));
            }
        }
    }
}

#define W4A4_ENTRY(NAME, MB, KU)                                                          \
    extern "C" __global__ __launch_bounds__(W4A4_WARPS * 32) void NAME(                    \
        const unsigned char* __restrict__ Aq, const unsigned char* __restrict__ As,       \
        const float* __restrict__ Ag, const unsigned char* __restrict__ Bq,               \
        const unsigned char* __restrict__ Bs, const float scale2,                         \
        __nv_bfloat16* __restrict__ C, unsigned int M, unsigned int N, unsigned int K) {  \
        w4a4_gemv_mx_impl<MB, KU>(Aq, As, Ag, Bq, Bs, scale2, C, M, N, K);                \
    }

W4A4_ENTRY(w4a4_gemv_mx8, 1, 4)    // M <= 8
W4A4_ENTRY(w4a4_gemv_mx16, 2, 4)   // M <= 16
W4A4_ENTRY(w4a4_gemv_mx32, 4, 2)   // M <= 32

// ── Per-row dynamic NVFP4 activation quantisation ───────────────────────────
__device__ __forceinline__ unsigned int w4a4_e2m1_rne(float x) {
    const float a = fabsf(x);
    unsigned int c;
    if (a <= 0.25f) c = 0u;
    else if (a < 0.75f) c = 1u;
    else if (a <= 1.25f) c = 2u;
    else if (a < 1.75f) c = 3u;
    else if (a <= 2.5f) c = 4u;
    else if (a < 3.5f) c = 5u;
    else if (a <= 5.0f) c = 6u;
    else c = 7u;
    return (x < 0.0f && c != 0u) ? (c | 8u) : c;
}

// One CTA (256 threads) per row. Aq/As/Ag as consumed above.
extern "C" __global__ __launch_bounds__(256) void w4a4_quant_rows(
    const __nv_bfloat16* __restrict__ A, unsigned char* __restrict__ Aq,
    unsigned char* __restrict__ As, float* __restrict__ Ag, unsigned int K)
{
    const unsigned int row = blockIdx.x;
    const __nv_bfloat16* x = A + (unsigned long long)row * K;
    float m = 0.0f;
    for (unsigned int k = threadIdx.x; k < K; k += 256u) m = fmaxf(m, fabsf(__bfloat162float(x[k])));
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, o));
    __shared__ float wm[8];
    if ((threadIdx.x & 31u) == 0u) wm[threadIdx.x >> 5] = m;
    __syncthreads();
    float amax = wm[0];
    #pragma unroll
    for (int w = 1; w < 8; w++) amax = fmaxf(amax, wm[w]);
    const float gs = amax > 0.0f ? amax / (6.0f * 448.0f) : 1.0f;
    if (threadIdx.x == 0) Ag[row] = gs;
    const float inv_gs = 1.0f / gs;

    for (unsigned int grp = threadIdx.x; grp < (K >> 4); grp += 256u) {
        const uint4* src = (const uint4*)(x + grp * 16u);
        const uint4 v0 = src[0], v1 = src[1];
        const unsigned int w[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
        float f[16];
        float gm = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            f[2 * i] = __uint_as_float(w[i] << 16);
            f[2 * i + 1] = __uint_as_float(w[i] & 0xFFFF0000u);
            gm = fmaxf(gm, fmaxf(fabsf(f[2 * i]), fabsf(f[2 * i + 1])));
        }
        const __nv_fp8_e4m3 s8(gm * (1.0f / 6.0f) * inv_gs);
        const float s = (float)s8;
        // FRAGMENT order (see the GEMV): within each k128 chunk the 8 group
        // scales are stored as [G0,G4,G2,G6, G1,G5,G3,G7] and group q's two
        // ints land in slot (2*(q>>2) + half)*4 + perm[q&3], perm = {0,2,1,3}.
        const unsigned int chunk = grp >> 3, q = grp & 7u;
        const unsigned int spos = ((q & 1u) << 2) | ((q >> 2) & 1u) | (q & 2u);
        As[(unsigned long long)row * (K >> 4) + chunk * 8u + spos] = *(const unsigned char*)&s8;
        const float inv = s > 0.0f ? 1.0f / (s * gs) : 0.0f;
        uint2 packed;
        unsigned int p[2] = {0u, 0u};
        #pragma unroll
        for (int i = 0; i < 16; i++) p[i >> 3] |= w4a4_e2m1_rne(f[i] * inv) << ((i & 7) * 4);
        (void)packed;
        const unsigned int pr = (q & 3u) == 1u ? 2u : (q & 3u) == 2u ? 1u : (q & 3u);
        unsigned char* dst = Aq + (unsigned long long)row * (K >> 1) + chunk * 64u;
        #pragma unroll
        for (unsigned int h = 0; h < 2u; h++) {
            const unsigned int slot = (2u * (q >> 2) + h) * 4u + pr;
            *(uint32_t*)(dst + slot * 4u) = p[h];
        }
    }
}
