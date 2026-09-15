// SPDX-License-Identifier: AGPL-3.0-only

// Shared inner machinery for the Hopper (sm_90a) GDN chunked-prefill twins
// (#928): `gdn_fwd_o_hopper.cu` and `gdn_recompute_wu_hopper.cu`.
//
// SSOT for why those two files exist — `GDN-PREFILL-ATTRIBUTION.md`, from the
// 1xH100 nsys round-9 capture (2026-09-11, Qwen/Qwen3.8-27B-FP8, nk=16, nv=48,
// kd=vd=128, CHUNK=64), whose closing paragraph names exactly the two scalar
// remnants these twins remove:
//
//   chunk_fwd_o    96 launches  T=1193  20.1 ms (5.5%), 209.8 us/launch
//                               T=4593  71.9 ms,        749.0 us/launch
//                               3.35 / 12.71 GFLOP -> 16.0 / 17.0 TFLOP/s
//   recompute_wu   96 launches  T=1193  14.9 ms (4.0%), 155.1 us/launch
//                               T=4593  47.5 ms,        494.9 us/launch
//                               1.90 / 7.19 GFLOP -> 12.2 / 14.5 TFLOP/s
//
// This header has no gb10 counterpart by design — it is the shared operand
// machinery of the two `.cu` twins and nothing else includes it, the same
// shape `w8a16_gemv_hopper.cuh` takes for the decode GEMV family.
//
// NOTHING HERE IS HOPPER-SPECIFIC AS AN INSTRUCTION SET. `mma.sync.m16n8k16`
// with bf16 operands exists on sm_80 and up, and the gb10 parents already use
// it. What is Hopper-specific is the SHAPE of the defect: 132 SMs against
// GB10's 48, so the parents' per-CTA serial tails are paid against a machine
// that has four warps idle for every one that is working, and the gb10 sources
// must not be edited to serve that (maintainer rule, 2026-09-11).

#ifndef ATLAS_GDN_PREFILL_HOPPER_CUH
#define ATLAS_GDN_PREFILL_HOPPER_CUH

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define GDNH_K_DIM 128
#define GDNH_V_DIM 128
#define GDNH_CHUNK 64

// Padded smem row strides, in bf16 elements.
//
// THE PARENTS ARE 8-WAY BANK-CONFLICTED AND THIS IS THE FIX. `mma_gram` in
// `kernels/gb10/common/gated_delta_rule_fla.cu` indexes its A operand as
// `sA[(m_base + grp) * 128 + ks + q*2]` and reads 4 bytes. In 4-byte words that
// is `(m_base + grp) * 64 + ks/2 + q`, so the eight `grp` rows a warp reads in
// one fragment land on `grp * 64 mod 32 == 0` — one bank group for all eight.
// A stride of 136 bf16 makes the row term `grp * 68 mod 32 == grp * 4`, i.e.
// eight distinct groups. Same argument for 72 against 64 on the short axis.
// This is a pure addressing change: it moves no arithmetic.
#define GDNH_SW 136 // 128-column tiles: q/k/S panels
#define GDNH_SC 72  //  64-column tiles: Gram limbs, uc^T, L limbs
#define GDNH_SX 24  //  16-column tiles: the per-warp triangular-solve panel

// Per-stream prefill geometry. Byte-for-byte the contract of `GDN_GEOM` in
// `gated_delta_rule_fla.cu` and `TCF_GEOM` in `gated_delta_rule_chunk_tc.cu`:
// varlen reads cu_seqlens/cu_chunks, uniform reduces to b*seq_len. Duplicated
// rather than shared because the gb10 sources must stay untouched and a
// `#include` across the symlink mirror would make this header a gb10 edit.
struct GdnhGeom {
    unsigned int seqlen, nchunks, choff;
    unsigned long long tokoff;
};
#define GDNH_GEOM(g)                                                           \
    GdnhGeom g;                                                                \
    (void)cu_chunks;                                                           \
    if (is_varlen) {                                                           \
        unsigned int _s0 = (unsigned int)cu_seqlens[b];                        \
        g.seqlen = (unsigned int)cu_seqlens[b + 1] - _s0;                      \
        g.tokoff = (unsigned long long)_s0;                                    \
        unsigned int _co = 0;                                                  \
        for (unsigned int _i = 0; _i < b; _i++)                                \
            _co += ((unsigned int)(cu_seqlens[_i + 1] - cu_seqlens[_i])        \
                    + GDNH_CHUNK - 1)                                          \
                   / GDNH_CHUNK;                                               \
        g.choff = _co;                                                         \
        g.nchunks = (g.seqlen + GDNH_CHUNK - 1) / GDNH_CHUNK;                  \
    } else {                                                                   \
        g.seqlen = seq_len;                                                    \
        g.tokoff = (unsigned long long)b * seq_len;                            \
        g.choff = b * num_chunks;                                              \
        g.nchunks = num_chunks;                                                \
    }

// One warp's slab of C[m][n] += SUM_k A[m][k] * B[n][k] on tensor cores.
// A is [.][SA] bf16 row-major, B is [.][SB] bf16 row-major (already the `.col`
// operand, i.e. indexed [n][k]), contraction extent KC (a multiple of 16). The
// warp owns m rows [m_base, m_base+16) and the NT n-tiles of 8 starting at
// n_base. `acc` is accumulated into and never zeroed here, so the caller
// decides between "fresh product", "scaled state" and "correction limb".
//
// Fragment addressing is copied verbatim from the production `mma_gram`
// (gated_delta_rule_fla.cu) and `tcf_mma` (gated_delta_rule_chunk_tc.cu) so
// the three helpers cannot drift in their reading of the m16n8k16 layout.
template <int NT, int KC, int SA, int SB>
__device__ __forceinline__ void gdnh_mma(const __nv_bfloat16* __restrict__ A,
                                         const __nv_bfloat16* __restrict__ B,
                                         unsigned int m_base, unsigned int n_base,
                                         unsigned int lane, float (&acc)[NT][4]) {
    const unsigned int grp = lane >> 2, q = lane & 3;
    const unsigned short* sA = (const unsigned short*)A;
    const unsigned short* sB = (const unsigned short*)B;
#pragma unroll
    for (int ks = 0; ks < KC; ks += 16) {
        const unsigned int fr0 = m_base + grp, fr1 = fr0 + 8;
        const unsigned int fc0 = ks + q * 2, fc1 = fc0 + 8;
        const unsigned int a0 = *(const unsigned int*)&sA[fr0 * SA + fc0];
        const unsigned int a1 = *(const unsigned int*)&sA[fr1 * SA + fc0];
        const unsigned int a2 = *(const unsigned int*)&sA[fr0 * SA + fc1];
        const unsigned int a3 = *(const unsigned int*)&sA[fr1 * SA + fc1];
#pragma unroll
        for (int nt = 0; nt < NT; nt++) {
            const unsigned int nc = n_base + nt * 8 + grp;
            const unsigned int k0 = ks + q * 2, k1 = k0 + 8;
            const unsigned int b0 =
                ((unsigned int)sB[nc * SB + k0 + 1] << 16) | (unsigned int)sB[nc * SB + k0];
            const unsigned int b1 =
                ((unsigned int)sB[nc * SB + k1 + 1] << 16) | (unsigned int)sB[nc * SB + k1];
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                : "=f"(acc[nt][0]), "=f"(acc[nt][1]), "=f"(acc[nt][2]), "=f"(acc[nt][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(acc[nt][0]),
                  "f"(acc[nt][1]), "f"(acc[nt][2]), "f"(acc[nt][3]));
        }
    }
}

// Split one f32 into two bf16 limbs: hi = bf16(x), lo = bf16(x - hi). The pair
// carries ~16 mantissa bits against one limb's 8, which is what lets an MMA
// stand in for an f32 multiply inside a difference or a solve. Both twins use
// it and both explain, at the call site, WHICH cancellation makes it necessary
// — a limb nobody can name the amplification for is MMA issue spent on nothing.
__device__ __forceinline__ void gdnh_split(float x, __nv_bfloat16& hi, __nv_bfloat16& lo) {
    hi = __float2bfloat16(x);
    lo = __float2bfloat16(x - (float)hi);
}

#endif // ATLAS_GDN_PREFILL_HOPPER_CUH
