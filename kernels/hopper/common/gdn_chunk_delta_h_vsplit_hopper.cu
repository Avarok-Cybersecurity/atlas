// SPDX-License-Identifier: AGPL-3.0-only

// Atlas GDN chunked-prefill state spine, VALUE-DIMENSION SPLIT — Hopper
// (sm_90a) twin of `gated_delta_rule_chunk_delta_h_tcfuse_x2` (#928).
//
// SSOT for why this file exists — `GDN-PREFILL-ATTRIBUTION.md`, from the
// 1xH100 nsys round-13 capture (2026-09-11, Qwen/Qwen3.8-27B-FP8,
// nk=16 nv=48 kd=vd=128 CHUNK=64), cell T1N:
//   gated_delta_rule_chunk_delta_h_tcfuse_x2  96 launches
//     52 060.3 us = 11.32% of the 4593-token prefill's 459.812 ms of GPU busy,
//     1 062.45 us per launch at the M=4576 chunk,
//     13.64 TFLOP/s = 1.38% of BF16 tensor-core peak,
//     327 GB/s      = 9.7%  of HBM3.
// Arithmetic intensity 41.8 FLOP/byte: bound by NEITHER roofline. And the
// engine already runs at the isolated kernel's own ceiling — the GB10
// microtest reports 1.0553 ms at T=4593 against the engine's 1.0625 ms, 0.7%
// apart — so there is no implementation slack left at this launch geometry.
//
// THE ONE AXIS LEFT IS CTA COUNT. The parent launches `grid = [nv, batch]`,
// which at nv=48 and C=1 is **48 CTAs on 132 SMs = 36% of the machine**. 84
// SMs are idle for the whole 52 ms. This file splits the value dimension
// across CTAs: 2-way -> 96 CTAs, 4-way -> 192.
//
// WHY THE SPLIT IS EXACT, AND WHY IT NEEDS NO CROSS-CTA REDUCTION. Read the
// parent's two per-chunk products with the value index held fixed:
//     Phase A   ws[i][v]   = SUM_k W[i][k] * S_c[k][v]
//               uc[i][v]   = U[i][v] - ws[i][v]
//               duc[i][v]  = exp(gc_last - gc_i) * uc[i][v]
//     Phase B   S_{c+1}[k][v] = exp(gc_last) * S_c[k][v] + SUM_i K[i][k]*duc[i][v]
// Every one contracts over `k` or over `i` and NEVER over `v`. So column block
// j of the state depends only on column block j of U (hence of V) plus the
// k-space operands W, K and the scalar decay row, which are SHARED and are
// re-read, not reduced. The downstream reader is the same: `chunk_fwd_o`
// computes O[:, v] = Q~ . S_c[:, v] + tril(decay . Q~ K^T) . uc[:, v], again
// column-separable, and it is untouched here — this kernel writes the same
// `S_out`, `uc_out` and `h_state` tensors at the same offsets, each CTA
// filling a disjoint set of v columns. Nothing is summed across CTAs, so
// there is no reduction tree to get wrong and no determinism question.
//
// BIT-IDENTITY, which is the contract and not an aspiration. An m16n8k16
// output element is a fixed k-tree over the `ks` loop, and the n-tiles of a
// warp's C fragment are INDEPENDENT accumulators — so which warp, which CTA
// and how many n-tiles at a time evaluate a given (m, n) element changes
// nothing about the order its terms are combined. This file preserves, per
// element, exactly the parent's sequence of operations:
//   * the same `ks` order (0, 16, ..., 112 in Phase A; 0, 16, 32, 48 in B);
//   * the same n-tile boundaries (multiples of 8 columns), so the 8 v columns
//     an `mma` covers are the parent's 8;
//   * the same f32 decay arithmetic (`expf(gl)`, `expf(gl - gc_i)`), computed
//     per CTA from the same inputs;
//   * the same two bf16 limbs of S_c (Phase A) and of duc (Phase B) — this is
//     a twin of the `_x2` entry ONLY, because `_x2` is the arm that meets the
//     spine's numerics contract (h rel_rms 3.0e-6..3.9e-6 against 1e-3, where
//     the 1-limb entry measures 2.0e-3..2.7e-3).
// The parent's own in-file V-split experiment agrees: the 2026-06-25 verdict
// recorded in `kernels/gb10/common/gated_delta_rule_chunk_tc.cu` measured
// **bit-parity 18/18** at VTILES=2/4/8 and rejected the split on SPEED
// (0.71x / 0.65x / 0.34x) — on GB10, a **48-SM** part where 48 CTAs already
// fill the device, so a split buys no SMs and pays the duplicated W/K reads
// twice over. That verdict's own closing line is "re-test that only after this
// changes the bound"; the bound changed twice since (the tensor-core spine, and
// a 132-SM part), which is what this file re-tests. It does NOT overturn it:
// `[defaults] gdn_spine_vsplit` is **1 on every target**, and round 16 runs
// the A/B with `ATLAS_GDN_SPINE_VSPLIT=2` / `=4`.
//
// WHAT IT COSTS. Per (chunk, head) the parent moves 98 560 B: W 16 384 + U
// 16 384 + K 16 384 + gc 256 + uc_out 16 384 + S_out 32 768. U, uc_out and
// S_out split with the columns; W, K and gc are re-read by every split. So
//   1-way  98 560 B   2-way 131 584 B (1.335x)   4-way 197 632 B (2.005x)
// at 9.7% of HBM today — the extra traffic is affordable exactly because the
// kernel is nowhere near the bandwidth roofline. It is the reason a 2-way
// split cannot be a 2x and an 8-way split would be pointless.
//
// SHARED MEMORY, and the 2-CTA/SM threshold. The parent is 88 324 B, so two
// resident CTAs need 176 648 B of H100's 228 KB — it fits, but at
// `__launch_bounds__(256, 1)` ptxas has no reason to try. Here the S tile, the
// U tile and the duc tile all shrink with the split while the W tile (k-space)
// and the K^T tile (k-space) do not:
//   split  St/Kt region   Wp      Up      ducT    dec    TOTAL
//     2      18 432     17 408   9 216   9 216   260   54 532 B
//     4      18 432     17 408   5 120   4 608   260   45 828 B
// i.e. 4 and 5 resident CTAs by shared memory, which is what makes
// `__launch_bounds__(256, 2)` reachable — needed at 4-way, where 192 CTAs must
// land on 132 SMs in one wave rather than two.
//
// GATES. `native_gdn_spine_vsplit_hopper_microtest` (H100 only — these entry
// points exist in no other image) asserts BYTE EQUALITY of `h`, `uc` and `S_c`
// against `..._tcfuse_x2` at T in {256, 1193, 4593} with a KNOWN_BAD control,
// plus the host simulation in
// `crates/spark-model/src/layers/ops/ssm_gdn_vsplit_tests.rs`, which runs the
// column-block index maps against the parent's with no GPU.
//
// GEOMETRY. Drop-in ABI == the parent (21 args) and block 256 unchanged. The
// ONLY launch difference is grid.y: `batch * VSPLIT` instead of `batch`, with
// `b = blockIdx.y / VSPLIT` and the column block `vs = blockIdx.y % VSPLIT`
// — the shape `gated_delta_rule_chunk_delta_h_tc_vblock` already uses for its
// DV blocks, so the launcher's grid arithmetic is not a new idea here.

#include "gdn_prefill_hopper.cuh"

// Residual-limb stride for `duc`'s lo limb, which ALIASES the dead `Wp`.
// 68 rather than GDNH_SC's 72 for the parent's reason: 128 * 68 * 2 is exactly
// Wp's size at the unsplit shape, and the parent pays the 2-way bank conflict
// on the limb that carries the small correction. Kept identical here so the
// two kernels differ in no operand address a fragment read can see.
#define GDNV_SCL 68

// Per-split shared-memory layout, as macros for the reason the parent's
// `TCF_SMEM` is one: it has to be usable in a `static_assert` and in pointer
// arithmetic in the same file, with no `--expt-relaxed-constexpr` question.
//
// `St` and `Kt` ALIAS (Phase A consumes St, then it is dead and the K
// transpose reuses its bytes), so that region is the LARGER of the two — and
// at every split it is `Kt`, because K^T is k-space and does not shrink with
// the value split. SSOT for the launcher's `shared_mem` argument, mirrored in
// `ops::gdn_spine_vsplit_smem`.
#define GDNV_VD_L(S) (GDNH_V_DIM / (S))
#define GDNV_ST_ELEMS(S)                                                               \
    (GDNV_VD_L(S) * GDNH_SW > GDNH_K_DIM * GDNH_SC ? GDNV_VD_L(S) * GDNH_SW            \
                                                   : GDNH_K_DIM * GDNH_SC)
#define GDNV_SMEM(S)                                                                   \
    (GDNV_ST_ELEMS(S) * 2 + GDNH_CHUNK * GDNH_SW * 2                                   \
     + GDNH_CHUNK * (GDNV_VD_L(S) + 8) * 2 + GDNV_VD_L(S) * GDNH_SC * 2                 \
     + (GDNH_CHUNK + 1) * 4)

static_assert(GDNV_SMEM(2) == 54532, "mirror of ops::gdn_spine_vsplit_smem(2)");
static_assert(GDNV_SMEM(4) == 45828, "mirror of ops::gdn_spine_vsplit_smem(4)");
// The duc lo limb must fit in the dead Wp, at every split this file emits.
static_assert(GDNV_VD_L(2) * GDNV_SCL <= GDNH_CHUNK * GDNH_SW, "ducL overruns Wp");
// ...and `Kt` must fit the region `St` sized, which is what the max above buys.
static_assert(GDNH_K_DIM * GDNH_SC <= GDNV_ST_ELEMS(4), "Kt overruns St");

// `cp.async` staging and the 16-byte K load, byte-for-byte the parent's
// (`tcf_cp_async16` / `TcfB8` in gated_delta_rule_chunk_tc.cu). Local to this
// file rather than hoisted into `gdn_prefill_hopper.cuh`: the other two twins
// stage through plain loads, so a shared copy would have no second caller.
__device__ __forceinline__ void gdnv_cp_async16(void* dst_smem, const void* src_gmem) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(
                     (unsigned int)__cvta_generic_to_shared(dst_smem)),
                 "l"(src_gmem));
}
__device__ __forceinline__ void gdnv_cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void gdnv_cp_wait() { asm volatile("cp.async.wait_group 0;\n" ::); }

union GdnvB8 {
    uint4 v;
    __nv_bfloat16 h[8];
};

// ── KERNEL ───────────────────────────────────────────────────────────────────
// 256 threads = 8 warps, exactly as the parent. Warp w owns state k-rows
// [16w, 16w+16) and ALL of this CTA's value columns, so the Phase-B C fragment
// is `NTB = VD_L/8` n-tiles instead of the parent's 16 — the same map with a
// narrower n extent, which is why the register budget falls with the split.
template <int VSPLIT>
__device__ __forceinline__ void gdnv_core(
    float* __restrict__ h_state, const __nv_bfloat16* __restrict__ W_in,
    const __nv_bfloat16* __restrict__ U_in, const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gc_in, __nv_bfloat16* __restrict__ S_out,
    __nv_bfloat16* __restrict__ uc_out, unsigned int seq_len, unsigned int num_chunks,
    unsigned int num_k_heads, unsigned int num_v_heads, unsigned int k_dim, unsigned int v_dim,
    unsigned int qk_stride, unsigned int h_state_is_table, const int* __restrict__ cu_seqlens,
    const int* __restrict__ cu_chunks, unsigned int is_varlen) {
    constexpr unsigned int VD_L = GDNV_VD_L(VSPLIT);     // value columns per CTA
    constexpr int NTA = (int)(VD_L / 16);               // Phase-A n-tiles per warp
    constexpr int NTB = (int)(VD_L / 8);                // Phase-B n-tiles per warp
    constexpr unsigned int SU = VD_L + 8;               // Up's padded row stride

    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y / VSPLIT;
    const unsigned int vs = blockIdx.y % VSPLIT;  // this CTA's value-column block
    const unsigned int vbase = vs * VD_L;
    if (vh >= num_v_heads) return;
    GDNH_GEOM(g);

    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5, lane = tid & 31;
    const unsigned int grp = lane >> 2, q = lane & 3;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    extern __shared__ __align__(16) char gdnv_smem_buf[];
    __nv_bfloat16* St = (__nv_bfloat16*)gdnv_smem_buf;          // [VD_L][GDNH_SW]
    __nv_bfloat16* Kt = St;                                     // [128][GDNH_SC] (alias)
    __nv_bfloat16* Wp = St + GDNV_ST_ELEMS(VSPLIT);             // [CHUNK][GDNH_SW]
    __nv_bfloat16* Up = Wp + GDNH_CHUNK * GDNH_SW;              // [CHUNK][SU]
    __nv_bfloat16* ducT = Up + GDNH_CHUNK * SU;                 // [VD_L][GDNH_SC]
    float* dec = (float*)(ducT + VD_L * GDNH_SC);               // [CHUNK+1], [0]=exp(gc_last)

    float* H = h_state_is_table
                   ? ((float* const*)h_state)[b] + (unsigned long long)vh * GDNH_K_DIM * GDNH_V_DIM
                   : h_state + ((unsigned long long)(b * num_v_heads + vh) * GDNH_K_DIM
                                * GDNH_V_DIM);

    // Phase-B accumulator == the recurrent state, for THIS CTA's columns only.
    const unsigned int m0 = warp * 16 + grp, m1 = m0 + 8;
    float acc[NTB][4];
#pragma unroll
    for (int nt = 0; nt < NTB; nt++) {
        const unsigned int n0 = vbase + nt * 8 + q * 2;
        acc[nt][0] = H[m0 * GDNH_V_DIM + n0];
        acc[nt][1] = H[m0 * GDNH_V_DIM + n0 + 1];
        acc[nt][2] = H[m1 * GDNH_V_DIM + n0];
        acc[nt][3] = H[m1 * GDNH_V_DIM + n0 + 1];
    }

    const __nv_bfloat16* key_b = key + g.tokoff * qk_stride;
    // Phase-A warp split: 4 m-tiles (i) x 2 halves of this CTA's columns.
    const unsigned int a_m = (warp & 3u) * 16, a_n = (warp >> 2) * (VD_L / 2);
    // K staging map: thread owns token row `krow` and the 32 k-columns at `kcol`.
    const unsigned int krow = tid >> 2, kcol = (tid & 3u) * 32;

    for (unsigned int c = 0; c < g.nchunks; c++) {
        const unsigned int cs = c * GDNH_CHUNK;
        const unsigned int ce =
            (g.seqlen - cs) < GDNH_CHUNK ? (g.seqlen - cs) : GDNH_CHUNK;
        const unsigned long long base = ((unsigned long long)(g.choff + c) * num_v_heads + vh);

        __syncthreads();  // previous chunk's Phase B is done reading Kt/ducT
        // (1) stage W (full, k-space) and this CTA's COLUMN BLOCK of U.
        for (unsigned int idx = tid * 8; idx < GDNH_CHUNK * GDNH_K_DIM; idx += 256 * 8)
            gdnv_cp_async16(&Wp[(idx / GDNH_K_DIM) * GDNH_SW + (idx % GDNH_K_DIM)],
                            &W_in[base * GDNH_CHUNK * GDNH_K_DIM + idx]);
        for (unsigned int idx = tid * 8; idx < GDNH_CHUNK * VD_L; idx += 256 * 8)
            gdnv_cp_async16(&Up[(idx / VD_L) * SU + (idx % VD_L)],
                            &U_in[base * GDNH_CHUNK * GDNH_V_DIM + (idx / VD_L) * GDNH_V_DIM
                                  + vbase + (idx % VD_L)]);
        gdnv_cp_commit();
        {  // exact f32 decay, recomputed per CTA from the same gc row.
            const float gl = gc_in[base * GDNH_CHUNK + ce - 1];
            if (tid == 0) dec[0] = expf(gl);
            if (tid < GDNH_CHUNK)
                dec[1 + tid] = (tid < ce) ? expf(gl - gc_in[base * GDNH_CHUNK + tid]) : 0.0f;
        }
        // (2) entry state S_c -> S_out (bf16, consumed by chunk_fwd_o) and the
        //     bf16 snapshot St[v][k] Phase A contracts against. Disjoint columns
        //     across the VSPLIT CTAs, so the two writes race with nothing.
#pragma unroll
        for (int nt = 0; nt < NTB; nt++) {
            const unsigned int n0 = vbase + nt * 8 + q * 2, n1 = n0 + 1;
            const unsigned int l0 = n0 - vbase, l1 = l0 + 1;
            const __nv_bfloat16 s00 = __float2bfloat16(acc[nt][0]);
            const __nv_bfloat16 s01 = __float2bfloat16(acc[nt][1]);
            const __nv_bfloat16 s10 = __float2bfloat16(acc[nt][2]);
            const __nv_bfloat16 s11 = __float2bfloat16(acc[nt][3]);
            S_out[base * GDNH_K_DIM * GDNH_V_DIM + m0 * GDNH_V_DIM + n0] = s00;
            S_out[base * GDNH_K_DIM * GDNH_V_DIM + m0 * GDNH_V_DIM + n1] = s01;
            S_out[base * GDNH_K_DIM * GDNH_V_DIM + m1 * GDNH_V_DIM + n0] = s10;
            S_out[base * GDNH_K_DIM * GDNH_V_DIM + m1 * GDNH_V_DIM + n1] = s11;
            St[l0 * GDNH_SW + m0] = s00;
            St[l1 * GDNH_SW + m0] = s01;
            St[l0 * GDNH_SW + m1] = s10;
            St[l1 * GDNH_SW + m1] = s11;
        }
        gdnv_cp_wait();
        __syncthreads();
        // Rows past the sequence end carry whatever recompute_wu left there.
        // Two loops rather than the parent's one because W and U no longer
        // share a row stride.
        if (ce < GDNH_CHUNK) {
            for (unsigned int e = tid; e < (GDNH_CHUNK - ce) * GDNH_SW; e += 256)
                Wp[ce * GDNH_SW + e] = __float2bfloat16(0.0f);
            for (unsigned int e = tid; e < (GDNH_CHUNK - ce) * SU; e += 256)
                Up[ce * SU + e] = __float2bfloat16(0.0f);
        }
        __syncthreads();

        // (3) K for THIS chunk into registers, issued before the Phase-A MMAs so
        //     its global latency hides behind them (Kt aliases St). Full k-space:
        //     every split re-reads it, which is the traffic this file pays.
        GdnvB8 kr[4];
        if (krow < ce) {
            const __nv_bfloat16* src =
                key_b + (unsigned long long)(cs + krow) * qk_stride + kh * k_dim + kcol;
#pragma unroll
            for (int j = 0; j < 4; j++) kr[j].v = *(const uint4*)(src + j * 8);
        } else {
#pragma unroll
            for (int j = 0; j < 4; j++) kr[j].v = make_uint4(0u, 0u, 0u, 0u);
        }

        // (4) PHASE A: ws[i][v] = <W_i, S_c[:,v]>, hi limb.
        float wsa[NTA][4];
#pragma unroll
        for (int nt = 0; nt < NTA; nt++) {
            wsa[nt][0] = 0.0f;
            wsa[nt][1] = 0.0f;
            wsa[nt][2] = 0.0f;
            wsa[nt][3] = 0.0f;
        }
        gdnh_mma<NTA, GDNH_K_DIM, GDNH_SW, GDNH_SW>(Wp, St, a_m, a_n, lane, wsa);
        {
            // The parent's second S_c limb, verbatim. `uc = U - W.S_c` is a
            // DIFFERENCE, so one bf16 limb of S_c lands on `uc` amplified by
            // |W.S_c| / |uc| — measured 2.56e-3 against the bf16 storage floor
            // of 1.65e-3. Two limbs recover ~16 mantissa bits and cost no
            // shared memory: the residual overwrites `St` in place, because the
            // f32 master is in the accumulator.
            __syncthreads();  // every warp is done reading the hi limb
#pragma unroll
            for (int nt = 0; nt < NTB; nt++) {
                const unsigned int l0 = nt * 8 + q * 2, l1 = l0 + 1;
                const float a0 = acc[nt][0], a1 = acc[nt][1];
                const float a2 = acc[nt][2], a3 = acc[nt][3];
                St[l0 * GDNH_SW + m0] = __float2bfloat16(a0 - (float)__float2bfloat16(a0));
                St[l1 * GDNH_SW + m0] = __float2bfloat16(a1 - (float)__float2bfloat16(a1));
                St[l0 * GDNH_SW + m1] = __float2bfloat16(a2 - (float)__float2bfloat16(a2));
                St[l1 * GDNH_SW + m1] = __float2bfloat16(a3 - (float)__float2bfloat16(a3));
            }
            __syncthreads();
            gdnh_mma<NTA, GDNH_K_DIM, GDNH_SW, GDNH_SW>(Wp, St, a_m, a_n, lane, wsa);
        }

        // (5) uc = U - ws ; duc = exp(gc_last - gc_i) * uc, written TRANSPOSED
        //     as Phase B's `.col` operand, in two bf16 limbs. Every (i, v) of
        //     this CTA's block is covered exactly once by the 8 warps.
        __syncthreads();           // Wp is dead; the duc residual limb takes it
        __nv_bfloat16* ducL = Wp;  // [VD_L][GDNV_SCL]
        const unsigned int i0 = a_m + grp, i1 = i0 + 8;
#define GDNV_EMIT(ii, vv, a)                                                            \
    do {                                                                                \
        const float uci = (float)Up[(ii) * SU + (vv)] - (a);                            \
        if ((ii) < ce)                                                                  \
            uc_out[base * GDNH_CHUNK * GDNH_V_DIM + (ii) * v_dim + vbase + (vv)] =       \
                __float2bfloat16(uci);                                                  \
        const float d = (ii) < ce ? dec[1 + (ii)] * uci : 0.0f;                          \
        __nv_bfloat16 dh, dl;                                                           \
        gdnh_split(d, dh, dl);                                                          \
        ducT[(vv) * GDNH_SC + (ii)] = dh;                                               \
        ducL[(vv) * GDNV_SCL + (ii)] = dl;                                              \
    } while (0)
#pragma unroll
        for (int nt = 0; nt < NTA; nt++) {
            const unsigned int v0 = a_n + nt * 8 + q * 2, v1 = v0 + 1;
            GDNV_EMIT(i0, v0, wsa[nt][0]);
            GDNV_EMIT(i0, v1, wsa[nt][1]);
            GDNV_EMIT(i1, v0, wsa[nt][2]);
            GDNV_EMIT(i1, v1, wsa[nt][3]);
        }
#undef GDNV_EMIT
        __syncthreads();  // St is dead past Phase A; Kt may overwrite it

        // (6) K^T into the freed St region: Kt[k][i] = K[i][k].
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int e = 0; e < 8; e++)
                Kt[(kcol + j * 8 + e) * GDNH_SC + krow] = kr[j].h[e];
        __syncthreads();

        // (7) PHASE B: S_{c+1} = edl*S_c + K^T . duc, both limbs. `edl` is an
        //     exact f32 multiply on the accumulator; the MMAs accumulate the
        //     correction into the same f32 registers.
        const float edl = dec[0];
#pragma unroll
        for (int nt = 0; nt < NTB; nt++) {
            acc[nt][0] *= edl;
            acc[nt][1] *= edl;
            acc[nt][2] *= edl;
            acc[nt][3] *= edl;
        }
        gdnh_mma<NTB, GDNH_CHUNK, GDNH_SC, GDNH_SC>(Kt, ducT, warp * 16, 0, lane, acc);
        // THE term: with gates ~0.9 the chunk decay is ~1e-3, so S_{c+1} is
        // almost entirely this correction and inherits duc's operand error
        // outright. See the parent's Phase-B note.
        gdnh_mma<NTB, GDNH_CHUNK, GDNH_SC, GDNV_SCL>(Kt, ducL, warp * 16, 0, lane, acc);
    }

#pragma unroll
    for (int nt = 0; nt < NTB; nt++) {
        const unsigned int n0 = vbase + nt * 8 + q * 2;
        H[m0 * GDNH_V_DIM + n0] = acc[nt][0];
        H[m0 * GDNH_V_DIM + n0 + 1] = acc[nt][1];
        H[m1 * GDNH_V_DIM + n0] = acc[nt][2];
        H[m1 * GDNH_V_DIM + n0 + 1] = acc[nt][3];
    }
}

// The two shipped entry points. Identical ABI (21 args) and block (256) to
// `..._tcfuse_x2`; grid.y is `batch * VSPLIT`, which is the ONLY launch
// difference and the whole point. `__launch_bounds__(256, 2)` on both: at
// 4-way the 192 CTAs must land on 132 SMs in one wave, and at 2-way the cap
// costs nothing ptxas reports (124 / 110 registers, zero spill — see the
// header).
//
// SPELLED OUT TWICE rather than emitted from a macro, unlike the parent's
// `TCF_ENTRY`. `build_shadow::entry_points` — which the shadow detector, the
// build's drop warning and `inherited_overrides.rs` all read — scrapes the
// kernel declarator and separately expands every function-like macro in the
// file, so an entry-emitting macro registers a PHANTOM entry named after its
// own parameter and a forwarding macro registers one named after itself. Both
// are measured facts on this file, not a worry: the first collided with the
// identical phantom the parent's macro produces and failed the addition test,
// and the second survived into the entry set. Twenty lines of duplication buys
// an entry set that is exactly the two names the launcher, the route line and
// the microtest resolve.
//
// The same hazard applies to PROSE. Naming the declarator keyword in a comment
// makes the scraper parse the next parenthesised token as a kernel — which is
// why this paragraph describes it rather than quoting it.
extern "C" __global__ void __launch_bounds__(256, 2) gated_delta_rule_chunk_delta_h_vsplit2_hopper(
    float* __restrict__ h_state, const __nv_bfloat16* __restrict__ W_in,
    const __nv_bfloat16* __restrict__ U_in, const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gate, const float* __restrict__ gc_in,
    __nv_bfloat16* __restrict__ S_out, __nv_bfloat16* __restrict__ uc_out,
    unsigned int batch_size, unsigned int seq_len, unsigned int num_chunks,
    unsigned int num_k_heads, unsigned int num_v_heads, unsigned int k_dim,
    unsigned int v_dim, unsigned int qk_stride, unsigned int gb_stride,
    unsigned int h_state_is_table, const int* __restrict__ cu_seqlens,
    const int* __restrict__ cu_chunks, unsigned int is_varlen) {
    (void)gate;
    (void)gb_stride;
    (void)batch_size;
    gdnv_core<2>(h_state, W_in, U_in, key, gc_in, S_out, uc_out, seq_len, num_chunks,
                 num_k_heads, num_v_heads, k_dim, v_dim, qk_stride, h_state_is_table,
                 cu_seqlens, cu_chunks, is_varlen);
}

extern "C" __global__ void __launch_bounds__(256, 2) gated_delta_rule_chunk_delta_h_vsplit4_hopper(
    float* __restrict__ h_state, const __nv_bfloat16* __restrict__ W_in,
    const __nv_bfloat16* __restrict__ U_in, const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gate, const float* __restrict__ gc_in,
    __nv_bfloat16* __restrict__ S_out, __nv_bfloat16* __restrict__ uc_out,
    unsigned int batch_size, unsigned int seq_len, unsigned int num_chunks,
    unsigned int num_k_heads, unsigned int num_v_heads, unsigned int k_dim,
    unsigned int v_dim, unsigned int qk_stride, unsigned int gb_stride,
    unsigned int h_state_is_table, const int* __restrict__ cu_seqlens,
    const int* __restrict__ cu_chunks, unsigned int is_varlen) {
    (void)gate;
    (void)gb_stride;
    (void)batch_size;
    gdnv_core<4>(h_state, W_in, U_in, key, gc_in, S_out, uc_out, seq_len, num_chunks,
                 num_k_heads, num_v_heads, k_dim, v_dim, qk_stride, h_state_is_table,
                 cu_seqlens, cu_chunks, is_varlen);
}
