// SPDX-License-Identifier: AGPL-3.0-only

// Atlas GDN chunked-prefill output pass — Hopper (sm_90a) twin of
// `gated_delta_rule_chunk_fwd_o` (#928).
//
// SSOT for every number below: `GDN-PREFILL-ATTRIBUTION.md`, nsys round 9,
// 1xH100, Qwen/Qwen3.8-27B-FP8, nk=16 nv=48 kd=vd=128 CHUNK=64, 2026-09-11.
//   96 launches, T=1193: 20.1 ms total (5.5% of a 368.3 ms prefill),
//                        209.8 us/launch, 3.35 GFLOP -> 16.0 TFLOP/s,
//                        89.7 MB -> 427 GB/s
//   96 launches, T=4593: 71.9 ms total, 749.0 us/launch,
//                        12.71 GFLOP -> 17.0 TFLOP/s, 340.0 MB -> 454 GB/s
//
// THE REMNANT, priced. Per (chunk, head) the parent does three products:
//   q.k^T      M=64  N=64  K=128 = 0.524 M MAC   mma.sync   (mma_gram)
//   q.S_c^T    M=64  N=128 K=128 = 1.049 M MAC   mma.sync   (mma_gram)
//   tril(kq).uc  SUM_i (i+1) * 128 = 0.266 M MAC  SCALAR    <- the remnant
// = 3.678 MFLOP total, of which the triangular term is 0.53 MFLOP (14.5%).
// It runs as `for i { for l<=i { t2 += kq[i][l] * ucb[l][v] } }` on `tid <
// v_dim`, i.e. 128 of the block's 512 threads, as 2080 FMAs per thread whose
// inner `l` loop is a DEPENDENT f32 chain. The two `mma_gram` calls are fenced
// to `tid < 128` as well (the helper hardwires M=64 across 4 warps), so in the
// parent TWELVE OF SIXTEEN WARPS DO NO ARITHMETIC AT ALL — they exist to stage
// 96.5 KB of shared memory and then idle.
//
// WHAT THIS TWIN CHANGES
//  1. ALL 16 WARPS COMPUTE. Every product is re-tiled as 4 m-tiles x 4
//     n-quarters, so each of the 512 threads carries a C fragment. The two
//     Gram products are bit-identical to the parent's: an m16n8k16 output
//     element is a fixed k-tree over ks = 0..112, and which warp evaluates it
//     changes nothing.
//  2. THE TRIANGULAR PRODUCT BECOMES AN MMA. `kq` is masked to l <= i and
//     rounded to two bf16 limbs in the C fragment where it is produced (it
//     never round-trips through f32 shared memory), and `uc` is staged
//     TRANSPOSED as the `.col` operand. The masked square [64x64]x[64x128]
//     costs 0.524 M MAC against the triangle's 0.266 M — 2x the arithmetic, on
//     a unit that runs it at 60x the rate, with no dependent chain.
//  3. BANK CONFLICTS. The parent's operand strides are 128 and 64 bf16, which
//     put the eight `grp` rows of every fragment read on one bank group; the
//     padded 136/72 strides here do not. See `gdn_prefill_hopper.cuh`.
//  4. `o1` STAYS f32. The parent writes q.S_c^T to bf16 shared memory and
//     reads it back to combine; here both products land in the SAME f32
//     accumulator, so the epilogue rounds once instead of twice.
//
// NUMERICS CONTRACT. `output` is bf16 and terminal, so its floor is bf16
// storage (the sibling spine microtest measures that floor at 1.65e-3 for
// these tensors) and no 1e-3 claim is available on it. Relative to the parent:
//   * REMOVED rounding: `o1` is no longer rounded to bf16 before the combine.
//   * ADDED rounding: `kq~` = exp(gc_i - gc_l) * <q_i, k_l> becomes two bf16
//     limbs. Two limbs carry ~16 mantissa bits, so the operand error is ~2^-17
//     = 7.6e-6 relative — two orders inside the bf16 output floor. ONE limb was
//     not chosen on a hunch: `t2` is a sum of 64 signed terms and the storage
//     floor is already 1.65e-3, so a 2e-3 operand error would land on top of
//     it. `uc` was ALREADY bf16 in memory on both paths.
//   * REASSOCIATED: the l-sum moves from a sequential f32 chain to the MMA's
//     fixed 16-wide tree. The campaign's standing warning applies (the SPLIT=4
//     note in `gated_delta_rule_fla.cu`: cos = 1.0000 while losing 1.4 BFCL
//     points), which is why the whole family stays behind a lever.
// The gate is `native_gdn_prefill_remnants_microtest` (H100 only — these entry
// points exist in no other image) plus the host simulation in
// `crates/spark-model/src/layers/ops/ssm_gdn_remnants_tests.rs`, which runs the
// index maps and the limb arithmetic against an f64 reference with no GPU.
//
// GEOMETRY. Drop-in ABI == the parent (20 args), grid [nchunks, nv, batch] and
// block 512 unchanged; only the shared-memory footprint differs, and it is
// SMALLER (97 536 B against 98 816) because the `kq` lo limb aliases the dead
// `sk`.
//
// OCCUPANCY, measured with ptxas (CUDA 13.0, `-arch=sm_90a --fmad=false`, the
// tree's own build flags, on gx10-a309 2026-09-11 — a cross-compile, no H100
// touched). The parent needs 104 registers at 512 threads, which is 1 CTA per
// SM: 2 x 512 threads would need 64 or fewer. This twin fits in 64 with ZERO
// spill, so `__launch_bounds__(512, 2)` is free and the resident CTA count
// DOUBLES. 2 x 97 536 B = 195 072 B also fits H100's 228 KB of shared memory.
//   parent  gated_delta_rule_chunk_fwd_o          104 regs, 0 B spill
//   twin    ..._hopper, __launch_bounds__(512,1)   85 regs, 0 B spill
//   twin    ..._hopper, __launch_bounds__(512,2)   64 regs, 0 B spill  <- shipped
// This is a compile-time receipt about register pressure and nothing more; no
// runtime measurement of either kernel on Hopper exists yet.

#include "gdn_prefill_hopper.cuh"

// smem: sq[64][136] + sk[64][136] + Sb[128][136] + ucT[128][72] + kqh[64][72]
//       + gc[64] = 17408 + 17408 + 34816 + 18432 + 9216 + 256 = 97 536 B.
// The `kq` LO limb aliases `sk`, which is dead the moment the q.k^T Gram
// retires: 9216 <= 17408, and a `__syncthreads` separates the two uses.
// SSOT for the launcher's `shared_mem` argument (mirrored in ssm_gdn_a3.rs).
#define FOH_SMEM                                                               \
    (GDNH_CHUNK * GDNH_SW * 2 + GDNH_CHUNK * GDNH_SW * 2 + GDNH_V_DIM * GDNH_SW * 2 \
     + GDNH_V_DIM * GDNH_SC * 2 + GDNH_CHUNK * GDNH_SC * 2 + GDNH_CHUNK * 4)

static_assert(FOH_SMEM == 97536, "FOH_SMEM must match ssm_gdn_a3.rs::GDN_FWD_O_HOPPER_SMEM");

extern "C" __global__ void __launch_bounds__(512, 2) gated_delta_rule_chunk_fwd_o_hopper(
    const __nv_bfloat16* __restrict__ query, const __nv_bfloat16* __restrict__ key,
    const float* __restrict__ gate, const float* __restrict__ gc_in,
    const __nv_bfloat16* __restrict__ S_in, const __nv_bfloat16* __restrict__ uc_in,
    __nv_bfloat16* __restrict__ output, unsigned int batch_size, unsigned int seq_len,
    unsigned int num_chunks, unsigned int num_k_heads, unsigned int num_v_heads,
    unsigned int k_dim, unsigned int v_dim, unsigned int qk_stride, unsigned int gb_stride,
    const int* __restrict__ cu_seqlens, const int* __restrict__ cu_chunks,
    unsigned int is_varlen) {
    (void)gate;
    (void)gb_stride;
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
    const float inv_sqrt_d = rsqrtf((float)k_dim);
    const unsigned int cs = c * GDNH_CHUNK;
    const unsigned int ce = (g.seqlen - cs) < GDNH_CHUNK ? (g.seqlen - cs) : GDNH_CHUNK;
    const unsigned long long base = ((unsigned long long)(g.choff + c) * num_v_heads + vh);
    const unsigned long long out_base = (g.tokoff * num_v_heads + vh) * v_dim;
    query += g.tokoff * qk_stride;
    key += g.tokoff * qk_stride;

    extern __shared__ __align__(16) char foh_smem[];
    __nv_bfloat16* sq = (__nv_bfloat16*)foh_smem;                    // [CHUNK][SW]
    __nv_bfloat16* sk = sq + GDNH_CHUNK * GDNH_SW;                   // [CHUNK][SW]
    __nv_bfloat16* kql = sk;                                         // [CHUNK][SC] alias
    __nv_bfloat16* Sb = sk + GDNH_CHUNK * GDNH_SW;                   // [V_DIM][SW] = S_c^T
    __nv_bfloat16* ucT = Sb + GDNH_V_DIM * GDNH_SW;                  // [V_DIM][SC]
    __nv_bfloat16* kqh = ucT + GDNH_V_DIM * GDNH_SC;                 // [CHUNK][SC]
    float* gc = (float*)(kqh + GDNH_CHUNK * GDNH_SC);                // [CHUNK]

    // ── staging ──────────────────────────────────────────────────────────────
    // Tiles are staged COMPLETE (all 64x128, all 128 v-columns), zero-filled
    // past `ce` and past a narrow head: an MMA reads its whole fragment, so a
    // row or column the parent could simply not loop over has to be defined.
    for (unsigned int idx = tid; idx < GDNH_CHUNK * GDNH_K_DIM; idx += 512) {
        const unsigned int i = idx / GDNH_K_DIM, j = idx % GDNH_K_DIM;
        if (i < ce && j < k_dim) {
            const unsigned long long off =
                (unsigned long long)(cs + i) * qk_stride + kh * k_dim + j;
            sq[i * GDNH_SW + j] = query[off];
            sk[i * GDNH_SW + j] = key[off];
        } else {
            sq[i * GDNH_SW + j] = __float2bfloat16(0.0f);
            sk[i * GDNH_SW + j] = __float2bfloat16(0.0f);
        }
    }
    // S_c^T[v][k] = S_c[k][v]. Iterated in SOURCE order (k outer, v inner) so
    // the global read is coalesced and the transpose is paid in shared memory;
    // the parent iterates in destination order and pays it on HBM.
    for (unsigned int idx = tid; idx < GDNH_K_DIM * GDNH_V_DIM; idx += 512) {
        const unsigned int k = idx / GDNH_V_DIM, v = idx % GDNH_V_DIM;
        Sb[v * GDNH_SW + k] = S_in[base * GDNH_K_DIM * GDNH_V_DIM + idx];
    }
    for (unsigned int idx = tid; idx < GDNH_CHUNK * GDNH_V_DIM; idx += 512) {
        const unsigned int i = idx / GDNH_V_DIM, v = idx % GDNH_V_DIM;
        ucT[v * GDNH_SC + i] = (i < ce && v < v_dim)
                                   ? uc_in[base * GDNH_CHUNK * GDNH_V_DIM + i * v_dim + v]
                                   : __float2bfloat16(0.0f);
    }
    for (unsigned int i = tid; i < GDNH_CHUNK; i += 512)
        gc[i] = (i < ce) ? gc_in[base * GDNH_CHUNK + i] : 0.0f;
    __syncthreads();

    // ── (1) kq[i][l] = <q_i, k_l>, 16 warps: 4 m-tiles x 4 n-tiles of 16 ─────
    const unsigned int m_base = (warp & 3u) * 16;
    float kqa[2][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
    gdnh_mma<2, GDNH_K_DIM, GDNH_SW, GDNH_SW>(sq, sk, m_base, (warp >> 2) * 16, lane, kqa);
    __syncthreads(); // every warp is done reading `sk`; its bytes become kql

    // ── (2) fold the decay, apply the causal mask, split to two bf16 limbs ───
    // The parent folds `exp(gc_i - gc_l)` into a shared-memory f32 `kq` and
    // leaves the l > i half holding the raw Gram (its consumer loops l <= i).
    // An MMA has no such loop bound, so the mask is MATERIALISED here — and the
    // fold happens in the C fragment, where the value already lives.
    {
        const unsigned int i0 = m_base + grp, i1 = i0 + 8;
        const float g0 = gc[i0], g1 = gc[i1];
#pragma unroll
        for (int nt = 0; nt < 2; nt++) {
            const unsigned int l0 = (warp >> 2) * 16 + nt * 8 + q4 * 2, l1 = l0 + 1;
            const unsigned int ii[4] = {i0, i0, i1, i1};
            const unsigned int ll[4] = {l0, l1, l0, l1};
            const float gi[4] = {g0, g0, g1, g1};
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const float x = (ii[e] < ce && ll[e] <= ii[e])
                                    ? expf(gi[e] - gc[ll[e]]) * kqa[nt][e]
                                    : 0.0f;
                __nv_bfloat16 hi, lo;
                gdnh_split(x, hi, lo);
                kqh[ii[e] * GDNH_SC + ll[e]] = hi;
                kql[ii[e] * GDNH_SC + ll[e]] = lo;
            }
        }
    }
    __syncthreads();

    // ── (3) O_i = exp(gc_i) * <q_i, S_c[:,v]> + SUM_{l<=i} kq~[i][l] * uc[l][v]
    // Both products accumulate into ONE f32 fragment, so `o1` is never rounded
    // and the epilogue rounds exactly once, at the bf16 output.
    const unsigned int n_base = (warp >> 2) * 32;
    float acc[4][4];
#pragma unroll
    for (int nt = 0; nt < 4; nt++) {
        acc[nt][0] = 0.0f;
        acc[nt][1] = 0.0f;
        acc[nt][2] = 0.0f;
        acc[nt][3] = 0.0f;
    }
    gdnh_mma<4, GDNH_K_DIM, GDNH_SW, GDNH_SW>(sq, Sb, m_base, n_base, lane, acc);
    {   // exact f32 scale by exp(gc_i) on the accumulator, m-indexed
        const float e0 = expf(gc[m_base + grp]), e1 = expf(gc[m_base + grp + 8]);
#pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            acc[nt][0] *= e0;
            acc[nt][1] *= e0;
            acc[nt][2] *= e1;
            acc[nt][3] *= e1;
        }
    }
    gdnh_mma<4, GDNH_CHUNK, GDNH_SC, GDNH_SC>(kqh, ucT, m_base, n_base, lane, acc);
    gdnh_mma<4, GDNH_CHUNK, GDNH_SC, GDNH_SC>(kql, ucT, m_base, n_base, lane, acc);

    const unsigned int i0 = m_base + grp, i1 = i0 + 8;
#pragma unroll
    for (int nt = 0; nt < 4; nt++) {
        const unsigned int v0 = n_base + nt * 8 + q4 * 2, v1 = v0 + 1;
        const unsigned int ii[4] = {i0, i0, i1, i1};
        const unsigned int vv[4] = {v0, v1, v0, v1};
#pragma unroll
        for (int e = 0; e < 4; e++) {
            if (ii[e] < ce && vv[e] < v_dim)
                output[out_base + (unsigned long long)(cs + ii[e]) * num_v_heads * v_dim
                       + vv[e]] = __float2bfloat16(acc[nt][e] * inv_sqrt_d);
        }
    }
}
