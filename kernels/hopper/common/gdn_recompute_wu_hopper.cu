// SPDX-License-Identifier: AGPL-3.0-only

// Atlas GDN chunked-prefill WY pass — Hopper (sm_90a) twin of
// `gated_delta_rule_recompute_wu` (#928).
//
// SSOT for every number below: `GDN-PREFILL-ATTRIBUTION.md`, nsys round 9,
// 1xH100, Qwen/Qwen3.8-27B-FP8, nk=16 nv=48 kd=vd=128 CHUNK=64, 2026-09-11.
//   96 launches, T=1193: 14.9 ms total (4.0% of a 368.3 ms prefill),
//                        155.1 us/launch, 1.90 GFLOP -> 12.2 TFLOP/s,
//                        60.0 MB -> 387 GB/s
//   96 launches, T=4593: 47.5 ms total, 494.9 us/launch,
//                        7.19 GFLOP -> 14.5 TFLOP/s, 227.4 MB -> 459 GB/s
//
// THE REMNANT, priced. Per (chunk, head) the parent does:
//   K.K^T            M=N=64 K=128      = 1.049 M MAC   mma.sync (mma_gram)
//   (I+L)U = beta.V  SUM_i i * 128     = 0.258 M MAC   SCALAR   <- the remnant
//   (I+L)W = beta.e^gc.K  same         = 0.258 M MAC   SCALAR   <- the remnant
// = 2.081 MFLOP, of which the two forward substitutions are 49.6% of the
// arithmetic and — per the in-file measurement of 2026-08-22, a solve-removed
// probe against the whole kernel (14.3/16.4/20.5 us of prologue against
// 68.9/96.6/140.8 us total) — 79-85% OF THE TIME. One thread per right-hand-
// side column, 2016 dependent f32 FMAs deep, with `acc[64]` in LOCAL memory
// because it is indexed by a runtime row (ptxas: 256-byte stack frame). The
// parent already carries the two wins that are available inside that shape:
// a right-looking block of 16 (1.95-2.28x over left-looking) and running the
// two independent solves on two thread halves. What is left is the shape.
//
// WHAT THIS TWIN CHANGES — a BLOCKED triangular solve, 512 threads.
// Partition the 64 rows into four blocks of 16. For j = 0..3:
//     X_j <- T_jj . B_j              T_jj = (I + L_jj)^-1, 16x16
//     B_i <- B_i - L_ij . X_j        for every i > j
// Both steps are `mma.sync.m16n8k16`. The 256 right-hand-side columns (128 for
// U, 128 for W) are split 16 to a warp, so each of the 16 warps holds its whole
// [64 x 16] panel in ONE C fragment — 32 f32 registers, no local memory, no
// runtime-indexed array — and the entire solve is warp-local: `__syncwarp`,
// never `__syncthreads`. Arithmetic goes from 258 k MAC of scalar FMA per solve
// to 328 k MAC of MMA (1.27x the work: four diagonal applications and six
// off-diagonal updates of 16x16x128 against the triangle's 2016 x 128), at a
// unit rate the parent's own siblings measure at 4.4x the scalar path.
//
// The DIAGONAL blocks are inverted, not substituted, because a substitution
// walks rows in sequence and this fragment layout spreads the 16 rows of a
// block across eight lanes. `T_jj` is built ONCE per (chunk, head) by an EXACT
// f32 forward substitution — the same recurrence the parent runs — on one lane
// per column of four warps, then applied by MMA. It is shared by both solves,
// which is why the inversion costs 4 x 680 = 2720 MAC against 656 k MAC of
// apply: it is 0.4% of the kernel. This is FLA's own structure (it forms
// `T = (I + tril(diag(beta).K.K^T, -1))^-1` outright); no code was taken.
//
// Two more things change, both pure addressing:
//  * `K.K^T` and the `L` build FUSE. `kk` is symmetric, so `kk[l][i]` — the
//    element the parent re-reads out of a 16 KB f32 shared buffer — is the C
//    fragment's own `kk[i][l]`, bit-identical (same k-tree, commuted operands).
//    The f32 Gram buffer disappears, and with it the aliasing contract that
//    made `L` and `kk` share it.
//  * padded 136/72/24 operand strides. The parent's 128/64 strides put the
//    eight `grp` rows of every fragment read on one bank group. See
//    `gdn_prefill_hopper.cuh`.
//
// NUMERICS CONTRACT. `W_out`/`U_out` are bf16 and terminal (the sibling spine
// microtest measures the bf16 storage floor at 1.65e-3 for these tensors), and
// `gc_out` is f32 and BIT-IDENTICAL to the parent — the order-stable scan is
// copied unchanged, including its serial 64-add, because a tree scan over this
// sum cost 2.6 BFCL points once already. Relative to the parent:
//   * the substitution becomes a blocked solve with an explicit 16x16 diagonal
//     inverse. Algebraically identical; the rounding differs.
//   * every MMA operand is TWO bf16 limbs (hi, lo = bf16(x - hi)) and the
//     products are Ah.Bh + Ah.Bl + Al.Bh, i.e. ~16 mantissa bits. One limb is
//     not enough here and the amplification is nameable: the solve SUBTRACTS
//     `L.X` from `B`, so the operand error is multiplied by |L.X| / |B_final|
//     and compounds over four blocks. Two limbs put it at ~2^-17 per stage,
//     two orders inside the bf16 output floor.
//   * the k-sums are reassociated into the MMA's fixed 16-wide tree. The
//     campaign's standing warning applies (the SPLIT=4 note in
//     `gated_delta_rule_fla.cu`: cos = 1.0000 while losing 1.4 BFCL points),
//     which is why the whole family stays behind a lever.
// The gate is `native_gdn_prefill_remnants_microtest` (H100 only — this entry
// point exists in no other image) plus the host simulation in
// `crates/spark-model/src/layers/ops/ssm_gdn_remnants_tests.rs`, which runs the
// blocked solve, in limb arithmetic, against an f64 forward substitution with
// no GPU.
//
// GEOMETRY. Drop-in ABI == the parent (21 args) and grid [nchunks, nv, batch]
// unchanged. Block goes 256 -> 512 and shared memory 33 024 -> 79 104 B.
//
// OCCUPANCY, measured with ptxas (CUDA 13.0, `-arch=sm_90a --fmad=false`, the
// tree's own build flags, on gx10-a309 2026-09-11 — a cross-compile, no H100
// touched):
//   parent  gated_delta_rule_recompute_wu   87 regs, 512 B STACK FRAME, 256 thr
//   twin    ..._hopper, (512, 1)           114 regs,   0 B stack frame  <- shipped
//   twin    ..._hopper, (512, 2)            64 regs,  84 B spill stores
// The parent's 512-byte stack frame IS the `acc[64]` local-memory array this
// rewrite exists to delete; the twin has none. `(512, 2)` is NOT taken: forcing
// 64 registers reintroduces spill traffic, which is the same defect under a
// different name. At (512, 1) the twin runs 512 threads per SM against the
// parent's 2 CTAs x 256 — the same warp residency, with the solve on tensor
// cores instead of a 2016-deep dependent FMA chain through local memory.
// This is a compile-time receipt about register pressure and nothing more; no
// runtime measurement of either kernel on Hopper exists yet.

#include "gdn_prefill_hopper.cuh"

// Floor for the linear gate before the log-space cumsum — copied from the
// parent, where deep-layer gates underflow to 0 and log(0) makes the chunked
// form NaN where the per-token recurrence tolerates it.
#define GATE_FLOOR 1e-30f

// smem: sk[64][136] + Ld[64][24]f32 + Tf[64][24]f32 + Lh/Ll[64][72]
//       + Th/Tl[64][24] + Xh/Xl[16 warps][16][24] + gc[64]f32
//     = 17408 + 6144 + 6144 + 9216 + 9216 + 3072 + 3072 + 12288 + 12288 + 256
//     = 79 104 B. SSOT for the launcher (mirrored in ssm_gdn_a3.rs).
#define WUH_SMEM                                                                   \
    (GDNH_CHUNK * GDNH_SW * 2 + 2 * (GDNH_CHUNK * GDNH_SX * 4)                      \
     + 2 * (GDNH_CHUNK * GDNH_SC * 2) + 2 * (GDNH_CHUNK * GDNH_SX * 2)              \
     + 2 * (16 * 16 * GDNH_SX * 2) + GDNH_CHUNK * 4)

static_assert(WUH_SMEM == 79104, "WUH_SMEM must match ssm_gdn_a3.rs::GDN_WU_HOPPER_SMEM");

// Publish one 16x16 C fragment as the `.col` MMA operand `panel[col][row]`, in
// two bf16 limbs. The caller owns the `__syncwarp` on either side.
__device__ __forceinline__ void wuh_publish(const float (&p)[2][4], __nv_bfloat16* __restrict__ Xh,
                                            __nv_bfloat16* __restrict__ Xl, unsigned int grp,
                                            unsigned int q4) {
#pragma unroll
    for (int nt = 0; nt < 2; nt++) {
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const unsigned int r = grp + (e >= 2 ? 8u : 0u);
            const unsigned int cl = nt * 8 + q4 * 2 + (e & 1);
            __nv_bfloat16 hi, lo;
            gdnh_split(p[nt][e], hi, lo);
            Xh[cl * GDNH_SX + r] = hi;
            Xl[cl * GDNH_SX + r] = lo;
        }
    }
}

extern "C" __global__ void __launch_bounds__(512, 1) gated_delta_rule_recompute_wu_hopper(
    const __nv_bfloat16* __restrict__ key, const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate, const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ W_out, __nv_bfloat16* __restrict__ U_out,
    float* __restrict__ gc_out, unsigned int batch_size, unsigned int seq_len,
    unsigned int num_chunks, unsigned int num_k_heads, unsigned int num_v_heads,
    unsigned int k_dim, unsigned int v_dim, unsigned int qk_stride, unsigned int v_stride,
    unsigned int gb_stride, const int* __restrict__ cu_seqlens,
    const int* __restrict__ cu_chunks, unsigned int is_varlen) {
    const unsigned int c = blockIdx.x;
    const unsigned int vh = blockIdx.y;
    const unsigned int b = blockIdx.z;
    if (vh >= num_v_heads || b >= batch_size) return;
    GDNH_GEOM(g);
    if (c >= g.nchunks) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5, lane = tid & 31;
    const unsigned int grp = lane >> 2, q4 = lane & 3;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;
    const unsigned int cs = c * GDNH_CHUNK;
    const unsigned int ce = (g.seqlen - cs) < GDNH_CHUNK ? (g.seqlen - cs) : GDNH_CHUNK;
    const unsigned long long bs = ((unsigned long long)(g.choff + c) * num_v_heads + vh);
    key += g.tokoff * qk_stride;
    value += g.tokoff * v_stride;
    gate += g.tokoff * gb_stride;
    beta += g.tokoff * gb_stride;

    extern __shared__ __align__(16) char wuh_smem[];
    __nv_bfloat16* sk = (__nv_bfloat16*)wuh_smem;                 // [CHUNK][SW]
    float* Ld = (float*)(sk + GDNH_CHUNK * GDNH_SW);              // [CHUNK][SX] diag blocks
    float* Tf = Ld + GDNH_CHUNK * GDNH_SX;                        // [CHUNK][SX] their inverses
    __nv_bfloat16* Lh = (__nv_bfloat16*)(Tf + GDNH_CHUNK * GDNH_SX); // [CHUNK][SC] hi(-L)
    __nv_bfloat16* Ll = Lh + GDNH_CHUNK * GDNH_SC;                // [CHUNK][SC] lo(-L)
    __nv_bfloat16* Th = Ll + GDNH_CHUNK * GDNH_SC;                // [CHUNK][SX] hi(T_jj)
    __nv_bfloat16* Tl = Th + GDNH_CHUNK * GDNH_SX;                // [CHUNK][SX] lo(T_jj)
    __nv_bfloat16* Xh = Tl + GDNH_CHUNK * GDNH_SX;                // [16][16][SX] hi panel
    __nv_bfloat16* Xl = Xh + 16 * 16 * GDNH_SX;                   // [16][16][SX] lo panel
    float* gc = (float*)(Xl + 16 * 16 * GDNH_SX);                 // [CHUNK]

    for (unsigned int idx = tid; idx < GDNH_CHUNK * GDNH_K_DIM; idx += 512) {
        const unsigned int i = idx / GDNH_K_DIM, j = idx % GDNH_K_DIM;
        sk[i * GDNH_SW + j] =
            (i < ce && j < k_dim)
                ? key[(unsigned long long)(cs + i) * qk_stride + kh * k_dim + j]
                : __float2bfloat16(0.0f);
    }
    // ORDER-STABLE gc scan, copied from the parent verbatim: the 64 global
    // loads and logf calls in parallel, the 64 ADDITIONS serially on one thread
    // in the original order. A tree scan here re-associates a sum that feeds
    // exp(gc_i - gc_l) across the whole L matrix and cost 2.6 BFCL points on
    // 2026-08-23 while reading cos 0.999999.
    for (unsigned int idx = tid; idx < GDNH_CHUNK; idx += 512)
        gc[idx] = (idx < ce) ? logf(fmaxf(gate[(unsigned long long)(cs + idx) * gb_stride + vh],
                                          GATE_FLOOR))
                             : 0.0f;
    __syncthreads();
    if (tid == 0) {
        float a = 0.0f;
        for (unsigned int i = 0; i < ce; i++) {
            a += gc[i];
            gc[i] = a;
        }
    }
    __syncthreads();
    for (unsigned int idx = tid; idx < GDNH_CHUNK; idx += 512) {
        if (idx >= ce)
            gc[idx] = 0.0f;
        else
            gc_out[bs * GDNH_CHUNK + idx] = gc[idx];
    }
    __syncthreads();

    // ── (1) K.K^T and the L build, FUSED in the C fragment ───────────────────
    // L[i][l] = beta_i * exp(gc_i - gc_l) * <k_l, k_i> for l < i. The Gram is
    // symmetric, so <k_l, k_i> is this fragment's own element and the parent's
    // f32 round-trip is unnecessary. `-L` is stored, because an MMA only
    // accumulates and the update is a subtraction.
    {
        const unsigned int mg = (warp & 3u) * 16, ng = (warp >> 2) * 16;
        float kka[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
        gdnh_mma<2, GDNH_K_DIM, GDNH_SW, GDNH_SW>(sk, sk, mg, ng, lane, kka);
        const unsigned int i0 = mg + grp, i1 = i0 + 8;
        const float b0 =
            (i0 < ce) ? beta[(unsigned long long)(cs + i0) * gb_stride + vh] : 0.0f;
        const float b1 =
            (i1 < ce) ? beta[(unsigned long long)(cs + i1) * gb_stride + vh] : 0.0f;
        const float g0 = gc[i0], g1 = gc[i1];
#pragma unroll
        for (int nt = 0; nt < 2; nt++) {
            const unsigned int l0 = ng + nt * 8 + q4 * 2, l1 = l0 + 1;
            const unsigned int ii[4] = {i0, i0, i1, i1};
            const unsigned int ll[4] = {l0, l1, l0, l1};
            const float bb[4] = {b0, b0, b1, b1};
            const float gg[4] = {g0, g0, g1, g1};
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const float v = (ii[e] < ce && ll[e] < ii[e])
                                    ? bb[e] * expf(gg[e] - gc[ll[e]]) * kka[nt][e]
                                    : 0.0f;
                __nv_bfloat16 hi, lo;
                gdnh_split(-v, hi, lo);
                Lh[ii[e] * GDNH_SC + ll[e]] = hi;
                Ll[ii[e] * GDNH_SC + ll[e]] = lo;
                if ((ii[e] >> 4) == (ll[e] >> 4))
                    Ld[ii[e] * GDNH_SX + (ll[e] & 15u)] = v;
            }
        }
    }
    __syncthreads();

    // ── (2) T_jj = (I + L_jj)^-1, exact f32 forward substitution ─────────────
    // Warp j owns block j, lane c owns column c: every value it reads is one it
    // wrote, so the recurrence needs no barrier. Rows past `ce` have a zero L
    // row, which makes their T row the identity — the partial-tail case falls
    // out rather than being special-cased.
    if (warp < 4 && lane < 16) {
        const unsigned int r0 = warp * 16, cc = lane;
        for (unsigned int r = 0; r < 16; r++)
            Tf[(r0 + r) * GDNH_SX + cc] = (r == cc) ? 1.0f : 0.0f;
        for (unsigned int r = cc + 1; r < 16; r++) {
            float s = 0.0f;
            for (unsigned int m = cc; m < r; m++)
                s -= Ld[(r0 + r) * GDNH_SX + m] * Tf[(r0 + m) * GDNH_SX + cc];
            Tf[(r0 + r) * GDNH_SX + cc] = s;
        }
    }
    __syncthreads();
    for (unsigned int idx = tid; idx < GDNH_CHUNK * 16; idx += 512) {
        const unsigned int r = idx / 16, cc = idx % 16;
        __nv_bfloat16 hi, lo;
        gdnh_split(Tf[r * GDNH_SX + cc], hi, lo);
        Th[r * GDNH_SX + cc] = hi;
        Tl[r * GDNH_SX + cc] = lo;
    }
    __syncthreads();

    // ── (3) the blocked solve, warp-local ────────────────────────────────────
    // Warp w solves columns [16*(w&7), +16) of U (w < 8) or W (w >= 8).
    const unsigned int solve = warp >> 3;
    const unsigned int nb = (warp & 7u) * 16;
    const unsigned int lim = (solve == 0) ? v_dim : k_dim;
    __nv_bfloat16* Xhw = Xh + warp * 16 * GDNH_SX;
    __nv_bfloat16* Xlw = Xl + warp * 16 * GDNH_SX;

    float acc[4][2][4];
#pragma unroll
    for (int mt = 0; mt < 4; mt++) {
#pragma unroll
        for (int nt = 0; nt < 2; nt++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const unsigned int i = mt * 16 + grp + (e >= 2 ? 8u : 0u);
                const unsigned int col = nb + nt * 8 + q4 * 2 + (e & 1);
                float v = 0.0f;
                if (i < ce && col < lim) {
                    const float bi = beta[(unsigned long long)(cs + i) * gb_stride + vh];
                    v = (solve == 0)
                            ? bi
                                  * (float)value[(unsigned long long)(cs + i) * v_stride
                                                 + vh * v_dim + col]
                            : bi * expf(gc[i]) * (float)sk[i * GDNH_SW + col];
                }
                acc[mt][nt][e] = v;
            }
        }
    }

#pragma unroll
    for (int j = 0; j < 4; j++) {
        wuh_publish(acc[j], Xhw, Xlw, grp, q4);
        __syncwarp();
        float x[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
        const __nv_bfloat16* th = Th + (unsigned int)(j * 16) * GDNH_SX;
        const __nv_bfloat16* tl = Tl + (unsigned int)(j * 16) * GDNH_SX;
        gdnh_mma<2, 16, GDNH_SX, GDNH_SX>(th, Xhw, 0, 0, lane, x);
        gdnh_mma<2, 16, GDNH_SX, GDNH_SX>(th, Xlw, 0, 0, lane, x);
        gdnh_mma<2, 16, GDNH_SX, GDNH_SX>(tl, Xhw, 0, 0, lane, x);
#pragma unroll
        for (int nt = 0; nt < 2; nt++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[j][nt][e] = x[nt][e];
        __syncwarp();
        wuh_publish(acc[j], Xhw, Xlw, grp, q4);
        __syncwarp();
#pragma unroll
        for (int i = j + 1; i < 4; i++) {
            const __nv_bfloat16* ah = Lh + (unsigned int)(i * 16) * GDNH_SC + j * 16;
            const __nv_bfloat16* al = Ll + (unsigned int)(i * 16) * GDNH_SC + j * 16;
            gdnh_mma<2, 16, GDNH_SC, GDNH_SX>(ah, Xhw, 0, 0, lane, acc[i]);
            gdnh_mma<2, 16, GDNH_SC, GDNH_SX>(ah, Xlw, 0, 0, lane, acc[i]);
            gdnh_mma<2, 16, GDNH_SC, GDNH_SX>(al, Xhw, 0, 0, lane, acc[i]);
        }
        __syncwarp();
    }

#pragma unroll
    for (int mt = 0; mt < 4; mt++) {
#pragma unroll
        for (int nt = 0; nt < 2; nt++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const unsigned int i = mt * 16 + grp + (e >= 2 ? 8u : 0u);
                const unsigned int col = nb + nt * 8 + q4 * 2 + (e & 1);
                if (i >= ce || col >= lim) continue;
                const __nv_bfloat16 o = __float2bfloat16(acc[mt][nt][e]);
                if (solve == 0)
                    U_out[bs * GDNH_CHUNK * GDNH_V_DIM + i * v_dim + col] = o;
                else
                    W_out[bs * GDNH_CHUNK * GDNH_K_DIM + i * k_dim + col] = o;
            }
        }
    }
}
