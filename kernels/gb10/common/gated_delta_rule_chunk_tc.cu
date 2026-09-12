// SPDX-License-Identifier: AGPL-3.0-only

// Atlas GDN chunked-prefill state spine, TENSOR-CORE build (#928).
//
// SSOT for why this exists — `GDN-PREFILL-ATTRIBUTION.md`, from the 1xH100
// nsys round-9 capture (2026-09-11, Qwen3.8-27B-FP8, nv=48/nk=16/hd=128):
//   gated_delta_rule_chunk_delta_h_vfused  96 launches
//     T=1193:   97 830 us (26.6% of a 368.3 ms prefill),  1019.1 us/launch
//     T=4593:  376 054 us (32.3% of a 1163.5 ms prefill), 3917.2 us/launch
//   -> 3.75 / 3.70 TFLOP/s, 94 / 89 GB/s = 5.6% of H100 FP32 peak, 2.7% of HBM,
//      0.38% of BF16 tensor-core peak, and ZERO mma instructions issued.
//   -> per-chunk cost is FLAT in T (53.6 us at 19 chunks, 54.4 us at 72), i.e.
//      ~95 000 cycles against a one-SM FP32 floor of 16 384. Latency-bound.
// The bound is the scalar `wsp` reduction in `cdh_vtile_core`: with SPLIT=2 each
// thread walks a 64-DEEP DEPENDENT f32 FMA chain per token, 64 tokens per chunk,
// holding Sold[64]+Snew[64] = 128 live f32 registers with 8 warps to interleave.
// Its two siblings in `gated_delta_rule_fla.cu` are the control: same file, same
// data, big matmuls already on `mma.sync` -> 16-17 (chunk_fwd_o) and 12-14.5
// (recompute_wu) TFLOP/s, 4.4x this kernel's rate.
//
// WHAT THIS KERNEL CHANGES, AND ONLY THIS. Both per-chunk products move to
// `mma.sync.m16n8k16.row.col.f32.bf16.bf16.f32`:
//     Phase A   ws[i][v] = SUM_k W[i][k] * S_c[k][v]        (M=64,N=128,K=128)
//     Phase B   S_{c+1}  = edl*S_c + SUM_i K[i][k]*duc[i][v] (M=128,N=128,K=64)
// 1024 MMAs per CTA per chunk = 128 per warp, against 8192 scalar FMAs/thread.
// The recurrent state NEVER leaves the f32 MMA accumulator (64 regs/thread, half
// of vfused's 128), and the decay math stays exact f32.
//
// NUMERICS CONTRACT, MEASURED (`native_gdn_chunk_prefill_microtest`, GB10,
// nv=48, f64 CPU reference, 2026-09-11). Accumulation is f32 throughout, and
// `edl * acc` / `exp(gc_last - gc_i)` stay unchanged f32 ops on unrounded
// values. TWO operands are newly rounded to bf16 relative to the scalar spine:
// S_c (Phase A's B operand) and duc (Phase B's). W and K were ALREADY bf16 in
// memory on both paths.
//   ..._tcfuse     (1 limb):  h rel_rms 2.1e-3 / 2.0e-3 / 2.7e-3 at T=256/1193/
//                             4593 -- OVER the 1e-3 budget.
//   ..._tcfuse_x2  (2 limbs): h rel_rms 3.2e-6 / 3.0e-6 / 3.9e-6, and uc/S_c
//                             land on 1.660e-3 = the scalar spine's own bf16
//                             STORAGE floor to four digits. Drift over 72
//                             serial chunks 1.02x, identical to the spine.
// The x2 entry is what `ATLAS_GDN_PREFILL_TC` ships. Speed on GB10:
// 1.58x / 2.16x / 2.29x over vfused (3.65 -> 8.38 TFLOP/s at T=4593); the
// second limb is free at the two long shapes, which is what says this kernel
// has MMA issue to spare.
//   * the k-reduction is REASSOCIATED into the MMA's fixed 16-wide tree. The
//     campaign's standing warning applies (`gated_delta_rule_fla.cu`: SPLIT=4
//     read cos=1.0000 on the spine while losing 1.4 BFCL points) -> a cosine
//     check is NOT sufficient evidence to promote this to default. The
//     ssm-poisoning tripwire is.
//
// GEOMETRY. Drop-in ABI == `..._vfused` (21 args, same S_out/uc_out/h_state
// layouts, so `chunk_fwd_o` is untouched). Grid stays [nv, batch] and block
// stays 256: the in-file 2026-06-25 V-split verdict (0.71x/0.65x/0.34x at
// VTILES=2/4/8, bit-parity 18/18) says adding CTAs loses WHILE the kernel is
// latency-bound. Re-test that only after this changes the bound.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define K_DIM 128
#define V_DIM 128
#define CHUNK 64
// Padded smem row strides. 128+8 and 64+8 bf16 make the mma fragment reads
// conflict-free: the A/B lane address is (row*STRIDE/2 + ks/2 + q) mod 32, and
// a stride of 136 (resp. 72) bf16 contributes row*4 (resp. row*4) rather than
// row*0, so the 8 `grp` rows land on 8 distinct bank groups instead of one.
#define TCF_SW 136   // W, U, St : 128 columns + 8 pad
#define TCF_SC 72    // Kt, ducT :  64 columns + 8 pad
// Residual-limb stride for the x2 arm's `duc`. 68 instead of 72 because the lo
// limb ALIASES `Wp` (dead the moment Phase A's last MMA retires) and 128*68*2 =
// 17 408 B is exactly Wp's size — so the second limb costs ZERO shared memory.
// 68 is 2-way bank-conflicted where 72 is conflict-free; that is the price of
// the alias, and it is paid only on the limb that carries the small correction.
#define TCF_SCL 68
// SSOT for the launcher's `shared_mem` argument (mirrored in ssm_gdn_a3.rs).
#define TCF_SMEM (V_DIM * TCF_SW * 2 + 2 * (CHUNK * TCF_SW * 2) \
                  + V_DIM * TCF_SC * 2 + (CHUNK + 1) * 4)

// Per-stream prefill geometry — same contract as `gated_delta_rule_fla.cu`'s
// GDN_GEOM: varlen reads cu_seqlens/cu_chunks, uniform reduces to b*seq_len.
struct TcfGeom { unsigned int seqlen, nchunks, choff; unsigned long long tokoff; };
#define TCF_GEOM(g)                                                            \
    TcfGeom g;                                                                 \
    (void)cu_chunks;                                                           \
    if (is_varlen) {                                                           \
        unsigned int _s0 = (unsigned int)cu_seqlens[b];                        \
        g.seqlen  = (unsigned int)cu_seqlens[b + 1] - _s0;                     \
        g.tokoff  = (unsigned long long)_s0;                                   \
        unsigned int _co = 0;                                                  \
        for (unsigned int _i = 0; _i < b; _i++)                                \
            _co += ((unsigned int)(cu_seqlens[_i + 1] - cu_seqlens[_i])        \
                    + CHUNK - 1) / CHUNK;                                      \
        g.choff   = _co;                                                       \
        g.nchunks = (g.seqlen + CHUNK - 1) / CHUNK;                            \
    } else {                                                                   \
        g.seqlen  = seq_len;                                                   \
        g.tokoff  = (unsigned long long)b * seq_len;                           \
        g.choff   = b * num_chunks;                                            \
        g.nchunks = num_chunks;                                                \
    }

__device__ __forceinline__ void tcf_cp_async16(void* dst_smem, const void* src_gmem) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                 "r"((unsigned int)__cvta_generic_to_shared(dst_smem)), "l"(src_gmem));
}
__device__ __forceinline__ void tcf_cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void tcf_cp_wait() { asm volatile("cp.async.wait_group 0;\n" ::); }

// One warp's slab of C = A * B^T on tensor cores. A is [M][SA] bf16 row-major,
// B is [N][SB] bf16 row-major (i.e. the `.col` operand, already transposed),
// contraction extent KC. The warp owns m rows [m_base, m_base+16) and the NT
// n-tiles starting at n_base. `acc` is accumulated into, never zeroed here, so
// the caller decides between "fresh product" and "edl-scaled state".
//
// Fragment addressing is copied verbatim from the production `mma_gram` in
// gated_delta_rule_fla.cu (which serves chunk_fwd_o and recompute_wu) so the
// two helpers cannot drift in their reading of the m16n8k16 layout.
template <int NT, int KC, int SA, int SB>
__device__ __forceinline__ void tcf_mma(
    const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,
    unsigned int m_base, unsigned int n_base, unsigned int lane, float (&acc)[NT][4]
) {
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
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                  "f"(acc[nt][0]), "f"(acc[nt][1]), "f"(acc[nt][2]), "f"(acc[nt][3]));
        }
    }
}

union TcfB8 { uint4 v; __nv_bfloat16 h[8]; };

// ── KERNEL: chunk_delta_h_tcfuse ─────────────────────────────────────────────
// 256 threads = 8 warps. The f32 state S[k][v] lives ENTIRELY in the Phase-B
// MMA accumulator: warp w owns k-rows [16w, 16w+16) and all 16 n-tiles, i.e.
//   acc[nt][0..3] <-> S[16w+grp][8nt+2q], S[16w+grp][8nt+2q+1],
//                     S[16w+grp+8][8nt+2q], S[16w+grp+8][8nt+2q+1]
// with grp = lane>>2, q = lane&3. 64 f32 registers of state per thread.
template <bool X2>
__device__ __forceinline__ void tcf_core(
    float* __restrict__ h_state, const __nv_bfloat16* __restrict__ W_in,
    const __nv_bfloat16* __restrict__ U_in, const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gc_in,
    __nv_bfloat16* __restrict__ S_out, __nv_bfloat16* __restrict__ uc_out,
    unsigned int seq_len, unsigned int num_chunks,
    unsigned int num_k_heads, unsigned int num_v_heads, unsigned int k_dim,
    unsigned int v_dim, unsigned int qk_stride,
    unsigned int h_state_is_table,
    const int* __restrict__ cu_seqlens, const int* __restrict__ cu_chunks,
    unsigned int is_varlen
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads) return;
    TCF_GEOM(g);

    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5, lane = tid & 31;
    const unsigned int grp = lane >> 2, q = lane & 3;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;

    extern __shared__ __align__(16) char tcf_smem[];
    __nv_bfloat16* St = (__nv_bfloat16*)tcf_smem;              // [V_DIM][TCF_SW]
    __nv_bfloat16* Kt = St;                                    // [K_DIM][TCF_SC] (alias, Phase B)
    __nv_bfloat16* Wp = St + V_DIM * TCF_SW;                   // [CHUNK][TCF_SW]
    __nv_bfloat16* Up = Wp + CHUNK * TCF_SW;                   // [CHUNK][TCF_SW]
    __nv_bfloat16* ducT = Up + CHUNK * TCF_SW;                 // [V_DIM][TCF_SC]
    float* dec = (float*)(ducT + V_DIM * TCF_SC);              // [CHUNK+1], [0]=exp(gc_last)

    float* H = h_state_is_table
        ? ((float* const*)h_state)[b] + (unsigned long long)vh * K_DIM * V_DIM
        : h_state + ((unsigned long long)(b * num_v_heads + vh) * K_DIM * V_DIM);

    // Phase-B accumulator == the recurrent state. Load S_0 from the f32 pool.
    const unsigned int m0 = warp * 16 + grp, m1 = m0 + 8;
    float acc[16][4];
    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        const unsigned int n0 = nt * 8 + q * 2;
        acc[nt][0] = H[m0 * V_DIM + n0];     acc[nt][1] = H[m0 * V_DIM + n0 + 1];
        acc[nt][2] = H[m1 * V_DIM + n0];     acc[nt][3] = H[m1 * V_DIM + n0 + 1];
    }

    const __nv_bfloat16* key_b = key + g.tokoff * qk_stride;
    // Phase-A warp split: 4 m-tiles (i) x 2 halves of the 16 n-tiles (v).
    const unsigned int a_m = (warp & 3u) * 16, a_n = (warp >> 2) * 64;
    // K staging map: thread owns token row `krow` and the 32 k-columns at `kcol`.
    const unsigned int krow = tid >> 2, kcol = (tid & 3u) * 32;

    for (unsigned int c = 0; c < g.nchunks; c++) {
        const unsigned int cs = c * CHUNK;
        const unsigned int ce = (g.seqlen - cs) < CHUNK ? (g.seqlen - cs) : CHUNK;
        const unsigned long long base = ((unsigned long long)(g.choff + c) * num_v_heads + vh);

        __syncthreads();   // previous chunk's Phase B is done reading Kt/ducT
        // (1) stage W and U (cp.async, 16 B = 8 bf16 per issue) and the decay row.
        for (unsigned int idx = tid * 8; idx < CHUNK * K_DIM; idx += 256 * 8)
            tcf_cp_async16(&Wp[(idx / K_DIM) * TCF_SW + (idx % K_DIM)],
                           &W_in[base * CHUNK * K_DIM + idx]);
        for (unsigned int idx = tid * 8; idx < CHUNK * V_DIM; idx += 256 * 8)
            tcf_cp_async16(&Up[(idx / V_DIM) * TCF_SW + (idx % V_DIM)],
                           &U_in[base * CHUNK * V_DIM + idx]);
        tcf_cp_commit();
        {   // exact f32 decay, one thread per token instead of vfused's serial 64.
            const float gl = gc_in[base * CHUNK + ce - 1];
            if (tid == 0) dec[0] = expf(gl);
            if (tid < CHUNK)
                dec[1 + tid] = (tid < ce) ? expf(gl - gc_in[base * CHUNK + tid]) : 0.0f;
        }
        // (2) entry state S_c -> S_out (bf16, consumed by chunk_fwd_o) and the
        //     bf16 snapshot St[v][k] that Phase A contracts against. The f32
        //     master in `acc` is untouched: this is a per-chunk READ of it.
        #pragma unroll
        for (int nt = 0; nt < 16; nt++) {
            const unsigned int n0 = nt * 8 + q * 2, n1 = n0 + 1;
            const __nv_bfloat16 s00 = __float2bfloat16(acc[nt][0]);
            const __nv_bfloat16 s01 = __float2bfloat16(acc[nt][1]);
            const __nv_bfloat16 s10 = __float2bfloat16(acc[nt][2]);
            const __nv_bfloat16 s11 = __float2bfloat16(acc[nt][3]);
            S_out[base * K_DIM * V_DIM + m0 * V_DIM + n0] = s00;
            S_out[base * K_DIM * V_DIM + m0 * V_DIM + n1] = s01;
            S_out[base * K_DIM * V_DIM + m1 * V_DIM + n0] = s10;
            S_out[base * K_DIM * V_DIM + m1 * V_DIM + n1] = s11;
            St[n0 * TCF_SW + m0] = s00;  St[n1 * TCF_SW + m0] = s01;
            St[n0 * TCF_SW + m1] = s10;  St[n1 * TCF_SW + m1] = s11;
        }
        tcf_cp_wait();
        __syncthreads();
        // Rows past the sequence end carry whatever recompute_wu left there. The
        // MMA reads all 64 rows (row m of C depends only on row m of A), so they
        // are zeroed rather than relied on; `duc` is zeroed again below anyway.
        if (ce < CHUNK)
            for (unsigned int e = tid; e < (CHUNK - ce) * TCF_SW; e += 256) {
                Wp[ce * TCF_SW + e] = __float2bfloat16(0.0f);
                Up[ce * TCF_SW + e] = __float2bfloat16(0.0f);
            }
        __syncthreads();

        // (3) K for THIS chunk into registers, issued before the Phase-A MMAs so
        //     its global latency hides behind them (Kt aliases St, which Phase A
        //     is still reading).
        TcfB8 kr[4];
        if (krow < ce) {
            const __nv_bfloat16* src =
                key_b + (unsigned long long)(cs + krow) * qk_stride + kh * k_dim + kcol;
            #pragma unroll
            for (int j = 0; j < 4; j++) kr[j].v = *(const uint4*)(src + j * 8);
        } else {
            #pragma unroll
            for (int j = 0; j < 4; j++) kr[j].v = make_uint4(0u, 0u, 0u, 0u);
        }

        // (4) PHASE A (TENSOR CORE): ws[i][v] = <W_i, S_c[:,v]>.
        float wsa[8][4];
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            wsa[nt][0] = 0.0f; wsa[nt][1] = 0.0f; wsa[nt][2] = 0.0f; wsa[nt][3] = 0.0f;
        }
        tcf_mma<8, K_DIM, TCF_SW, TCF_SW>(Wp, St, a_m, a_n, lane, wsa);
        if (X2) {
            // SPLIT-bf16 (x2) LIMB on S_c. The plain arm's whole numerics cost is this
            // one rounding: `uc = U - W.S_c` is a DIFFERENCE, so S_c's bf16
            // operand error lands on `uc` amplified by |W.S_c| / |uc|. Measured
            // on GB10 (native_gdn_chunk_prefill_microtest, T=1193): plain-arm
            // uc rel_rms 2.56e-3 against the bf16-STORAGE floor of 1.65e-3 that
            // the scalar spine also pays, i.e. ~1.95e-3 of genuine excess, and
            // the final f32 state lands at 2.00e-3 — the same number, which is
            // what says this term dominates and `duc`'s own rounding does not.
            //
            // The fix is the classic two-limb product: S = hi + lo with
            // hi = bf16(S) and lo = bf16(S - hi), so W.S is recovered to ~16
            // mantissa bits. It costs a SECOND Phase-A pass and NOTHING in
            // shared memory, because the residual limb overwrites `St` in place
            // — the f32 master is in the accumulator, so the limb is recomputed,
            // not stored. That is why this is not the DV-split trade: no extra
            // smem, no extra CTA, no duplicated global traffic.
            __syncthreads();   // every warp is done reading the hi limb
            #pragma unroll
            for (int nt = 0; nt < 16; nt++) {
                const unsigned int n0 = nt * 8 + q * 2, n1 = n0 + 1;
                const float a0 = acc[nt][0], a1 = acc[nt][1];
                const float a2 = acc[nt][2], a3 = acc[nt][3];
                St[n0 * TCF_SW + m0] = __float2bfloat16(a0 - (float)__float2bfloat16(a0));
                St[n1 * TCF_SW + m0] = __float2bfloat16(a1 - (float)__float2bfloat16(a1));
                St[n0 * TCF_SW + m1] = __float2bfloat16(a2 - (float)__float2bfloat16(a2));
                St[n1 * TCF_SW + m1] = __float2bfloat16(a3 - (float)__float2bfloat16(a3));
            }
            __syncthreads();
            tcf_mma<8, K_DIM, TCF_SW, TCF_SW>(Wp, St, a_m, a_n, lane, wsa);
        }

        // (5) uc = U - ws ; duc = exp(gc_last - gc_i) * uc, written TRANSPOSED as
        //     Phase B's `.col` operand. Every (i, v) is covered exactly once by
        //     the 8 warps (4 m-tiles x 2 n-halves).
        if (X2) __syncthreads();   // Wp is dead; the duc residual limb takes it
        __nv_bfloat16* ducL = Wp;  // [V_DIM][TCF_SCL], x2 arm only
        const unsigned int i0 = a_m + grp, i1 = i0 + 8;
#define TCF_EMIT(ii, vv, a)                                                    \
            do {                                                                   \
                const float uci = (float)Up[(ii) * TCF_SW + (vv)] - (a);           \
                if ((ii) < ce)                                                     \
                    uc_out[base * CHUNK * V_DIM + (ii) * v_dim + (vv)] =            \
                        __float2bfloat16(uci);                                      \
                const float d = (ii) < ce ? dec[1 + (ii)] * uci : 0.0f;            \
                const __nv_bfloat16 dh = __float2bfloat16(d);                      \
                ducT[(vv) * TCF_SC + (ii)] = dh;                                    \
                if (X2)                                                             \
                    ducL[(vv) * TCF_SCL + (ii)] =                                   \
                        __float2bfloat16(d - (float)dh);                            \
            } while (0)
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            const unsigned int v0 = a_n + nt * 8 + q * 2, v1 = v0 + 1;
            TCF_EMIT(i0, v0, wsa[nt][0]);
            TCF_EMIT(i0, v1, wsa[nt][1]);
            TCF_EMIT(i1, v0, wsa[nt][2]);
            TCF_EMIT(i1, v1, wsa[nt][3]);
        }
#undef TCF_EMIT
        __syncthreads();   // St is dead past Phase A; Kt may overwrite it

        // (6) K^T into the freed St region: Kt[k][i] = K[i][k].
        #pragma unroll
        for (int j = 0; j < 4; j++)
            #pragma unroll
            for (int e = 0; e < 8; e++) Kt[(kcol + j * 8 + e) * TCF_SC + krow] = kr[j].h[e];
        __syncthreads();

        // (7) PHASE B (TENSOR CORE): S_{c+1} = edl*S_c + K^T * duc. The edl scale
        //     is an exact f32 multiply on the accumulator; the MMA accumulates
        //     the correction into the same f32 registers.
        const float edl = dec[0];
        #pragma unroll
        for (int nt = 0; nt < 16; nt++) {
            acc[nt][0] *= edl; acc[nt][1] *= edl; acc[nt][2] *= edl; acc[nt][3] *= edl;
        }
        tcf_mma<16, CHUNK, TCF_SC, TCF_SC>(Kt, ducT, warp * 16, 0, lane, acc);
        if (X2) {
            // THE term. Measured on GB10 (T=4593, 72 chunks): splitting S_c
            // alone moved the f32 state only 2.72e-3 -> 2.44e-3, because the
            // FIRST chunk is already exact (per-chunk rel_rms 1.66e-3 = the
            // bf16 storage floor) and the error is INJECTED here, then fed
            // forward. With gates ~0.9 the chunk decay exp(sum of 64 log-gates)
            // is ~1e-3, so S_{c+1} is almost entirely this correction term and
            // inherits `duc`'s bf16 operand error outright.
            tcf_mma<16, CHUNK, TCF_SC, TCF_SCL>(Kt, ducL, warp * 16, 0, lane, acc);
        }
    }

    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        const unsigned int n0 = nt * 8 + q * 2;
        H[m0 * V_DIM + n0] = acc[nt][0];     H[m0 * V_DIM + n0 + 1] = acc[nt][1];
        H[m1 * V_DIM + n0] = acc[nt][2];     H[m1 * V_DIM + n0 + 1] = acc[nt][3];
    }
}

// The two shipped entry points. Identical ABI (21 args), identical grid
// [nv, batch], block 256 and smem TCF_SMEM — they differ ONLY in whether
// Phase A contracts against one bf16 limb of S_c or two. See the SPLIT_S block
// above for the measurement that says which one meets the numerics contract.
// `_x2` carries a second bf16 limb of BOTH S_c (Phase A, in-place in `St`) and
// `duc` (Phase B, aliased over the dead `Wp`), so it costs no shared memory and
// no extra global traffic — only MMA issue, which this kernel has to spare.
#define TCF_ENTRY(NAME, SPLIT)                                                 \
    extern "C" __global__ void __launch_bounds__(256, 1) NAME(                 \
        float* __restrict__ h_state, const __nv_bfloat16* __restrict__ W_in,   \
        const __nv_bfloat16* __restrict__ U_in,                                \
        const __nv_bfloat16* __restrict__ key, const float* __restrict__ gate, \
        const float* __restrict__ gc_in, __nv_bfloat16* __restrict__ S_out,    \
        __nv_bfloat16* __restrict__ uc_out, unsigned int batch_size,           \
        unsigned int seq_len, unsigned int num_chunks, unsigned int num_k_heads,\
        unsigned int num_v_heads, unsigned int k_dim, unsigned int v_dim,      \
        unsigned int qk_stride, unsigned int gb_stride,                        \
        unsigned int h_state_is_table, const int* __restrict__ cu_seqlens,     \
        const int* __restrict__ cu_chunks, unsigned int is_varlen) {           \
        (void)gate;                                                            \
        (void)gb_stride;                                                       \
        (void)batch_size;                                                      \
        tcf_core<SPLIT>(h_state, W_in, U_in, key, gc_in, S_out, uc_out,        \
                        seq_len, num_chunks, num_k_heads, num_v_heads, k_dim,  \
                        v_dim, qk_stride, h_state_is_table, cu_seqlens,        \
                        cu_chunks, is_varlen);                                 \
    }

TCF_ENTRY(gated_delta_rule_chunk_delta_h_tcfuse, false)
TCF_ENTRY(gated_delta_rule_chunk_delta_h_tcfuse_x2, true)
