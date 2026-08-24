// SPDX-License-Identifier: AGPL-3.0-only

// Atlas WY-Chunkwise Gated Delta Rule — K=17 verification (DFlash γ+1).
//
// Generalization of gated_delta_rule_wy4.cu to K=17 tokens. DFlash uses
// γ=16 drafts per step plus 1 prefix-bonus position, so the K=γ verify
// path runs 17 tokens through every SSM layer in one fused pass.
//
// Algorithm (same WY-chunkwise structure as wy4, scaled to K=17):
//   1. Load q[K], k[K] into SMEM (17 KB total at K=17, k_dim=128).
//   2. Compute K*(K-1)/2 = 136 inter-token k-dot products via block reduction.
//   3. PASS 1: read H once, compute hk[K] = pre-update H·k[t] dots.
//   4. WY correction (sequential over K tokens): produce vn[K].
//   5. PASS 2: apply K state updates in single fused loop, writing
//      Hi_t = state after token t for t=0..K-2, and final H = state
//      after token K-1.
//
// SMEM budget @ K=17, k_dim=128:
//   sk[17][128] + sq[17][128] = 17·128·2·4 = 17 KB
//   kdots[136]                             = 0.5 KB
//   gate/beta scalars[2·17]                = 0.1 KB
//   warp_sums[4]                           = 16 B
//   ────────────────────────────────────────
//   Total                                  ≈ 17.7 KB  (SM_120 cap: 100 KB)
//
// Grid: (num_v_heads, batch, 1)   Block: (128, 1, 1)
// Reduction primitives match per-token baseline bit-exactly.

#include <cuda_bf16.h>
#include "../../common/gdn_reduce.cuh"
#define BLOCK_SIZE 128
#define K_TOKENS 17

// `h_state_inter_base` is a contiguous pool of (K-1)=16 intermediate H
// states for this (layer, slot). Stride between intermediates is
// `inter_stride_floats` floats. Slot t's intermediate lives at
// `h_state_inter_base + t * inter_stride_floats` (per (b, vh) sub-region).
//
// `h_state` itself becomes the K-1=16th (final) intermediate.
//
// LAZY Hi-WRITES (ATLAS_WY17_LAZY=J, MISSION target #1)
// ─────────────────────────────────────────────────────
// The 16 per-token Hi writes in PASS 2 are 86% of this kernel's DRAM
// traffic, yet the commit consumer (async_chkpt.rs) reads at most ONE of
// them: inter[num_accepted-1] on a partial accept (full accept reads
// NONE — it uses the live h_state). To cut that traffic losslessly we
// take `lazy_j`:
//   * lazy_j == 1  → write ALL 16 slots. BIT-IDENTICAL to the historical
//                    kernel (the shared `gated_delta_rule_wy17` symbol
//                    keeps lazy_j==1 semantics via a thin wrapper below).
//   * lazy_j >  1  → write only CHECKPOINT slots: slot 0 (post-root), the
//                    last slot K-2 (highest partial-accept target), and
//                    every J-th slot. On a partial accept whose target
//                    slot was NOT written, the commit path must replay
//                    from the nearest written checkpoint <= target via
//                    `gated_delta_rule_wy17_replay` (below), which is
//                    bit-exact (same FP32 recurrence, same token order).
// The per-token OUTPUT (qd[t]) never depended on Hi being in DRAM — it is
// computed from the rolling H registers — so outputs + final h_state are
// bit-identical for ANY lazy_j. Only the checkpoint DRAM footprint shrinks.
//
// A slot s is a checkpoint iff:  s == 0 || s == K-2 || ((s+1) % lazy_j == 0)
__device__ __forceinline__ bool wy17_is_checkpoint(int s, unsigned int lazy_j) {
    if (lazy_j <= 1u) return true;
    return (s == 0) || (s == K_TOKENS - 2) || (((unsigned)(s + 1) % lazy_j) == 0u);
}

__device__ __forceinline__ void gated_delta_rule_wy17_impl(
    float* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ output,
    // Pool of 16 intermediates (Hi_0..Hi_15). Each Hi_t is at
    // h_state_inter_base + t * inter_stride_floats (per (b, vh)).
    float* __restrict__ h_state_inter_base,
    unsigned int inter_stride_floats,
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride,
    unsigned int lazy_j            // 1 = write all (bit-identical); >1 = checkpoints only
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int hr = num_v_heads / num_k_heads;
    const unsigned int kh = vh / hr;
    const unsigned int hv = k_dim * v_dim;

    float* H = h_state + ((b * num_v_heads + vh) * hv);
    // Per-(b, vh) offset into the intermediate pool. Each Hi_t base ptr =
    // h_state_inter_base + t * inter_stride_floats + ((b*nv+vh)*hv).
    float* Hi_base = h_state_inter_base + ((b * num_v_heads + vh) * hv);

    // ── Load q, k, v, gate, beta scalars into SMEM ──
    __shared__ float sk[K_TOKENS][128];
    __shared__ float sq[K_TOKENS][128];
    __shared__ float sg[K_TOKENS];   // gate clamped
    __shared__ float sbt[K_TOKENS];  // beta
    __shared__ float smem_warp[4];

    // Each token t has its q/k row at offset (b*K + t) * qk_stride + kh*k_dim.
    if (tid < k_dim) {
        #pragma unroll
        for (int t = 0; t < K_TOKENS; t++) {
            const __nv_bfloat16* q_t = query + (b * K_TOKENS + t) * qk_stride + kh * k_dim;
            const __nv_bfloat16* k_t = key   + (b * K_TOKENS + t) * qk_stride + kh * k_dim;
            sq[t][tid] = (float)q_t[tid];
            sk[t][tid] = (float)k_t[tid];
        }
    }
    if (tid < K_TOKENS) {
        // Gate clamp matches per-token gated_delta_rule_decode (see wy4 comment).
        float g_raw = gate[(b * K_TOKENS + tid) * gb_stride + vh];
        sg[tid] = fminf(fmaxf(g_raw, 1e-6f), 1.0f - 1e-6f);
        sbt[tid] = beta[(b * K_TOKENS + tid) * gb_stride + vh];
    }
    __syncthreads();

    // ── Compute K*(K-1)/2 = 136 k-dot products via block reduction ──
    // kd[t][s] = k_t · k_s for s < t. Stored sparsely: kd_flat[tri_idx(t,s)]
    // where tri_idx(t,s) = t*(t-1)/2 + s for s < t.
    __shared__ float kd_flat[K_TOKENS * (K_TOKENS - 1) / 2];

    #pragma unroll
    for (int t = 1; t < K_TOKENS; t++) {
        #pragma unroll
        for (int s = 0; s < t; s++) {
            float p = (tid < k_dim) ? sk[t][tid] * sk[s][tid] : 0.0f;
            float r = atlas_block_reduce_sum(p, smem_warp, tid);
            if (tid == 0) {
                kd_flat[t * (t - 1) / 2 + s] = r;
            }
            __syncthreads();
        }
    }

    if (tid < v_dim) {
        // Load v[K] for this thread's v_dim slot.
        float vi[K_TOKENS];
        #pragma unroll
        for (int t = 0; t < K_TOKENS; t++) {
            const __nv_bfloat16* v_t = value + (b * K_TOKENS + t) * v_stride + vh * v_dim;
            vi[t] = (float)v_t[tid];
        }

        // ── PASS 1: Read H once, compute K dot products hk[t] = H · k_t ──
        float hk[K_TOKENS];
        #pragma unroll
        for (int t = 0; t < K_TOKENS; t++) hk[t] = 0.0f;

        #pragma unroll 4
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = H[(j + 0) * v_dim + tid];
            float h1 = H[(j + 1) * v_dim + tid];
            float h2 = H[(j + 2) * v_dim + tid];
            float h3 = H[(j + 3) * v_dim + tid];
            #pragma unroll
            for (int t = 0; t < K_TOKENS; t++) {
                hk[t] += h0 * sk[t][j + 0] + h1 * sk[t][j + 1]
                       + h2 * sk[t][j + 2] + h3 * sk[t][j + 3];
            }
        }

        // ── WY Correction (sequential over K tokens) ──
        // hk_corrected[t] = product(g[0..t-1]) * hk_raw[t]
        //                 + sum_{s<t} (product(g[s+1..t-1])) * kd[t][s] * vn[s]
        // vn[t]           = (v[t] - g[t] * hk_corrected[t]) * beta[t]
        //
        // Build vn[K] sequentially. Carry running products of g.
        float vn[K_TOKENS];

        // Token 0: no correction
        vn[0] = (vi[0] - sg[0] * hk[0]) * sbt[0];

        // Tokens 1..K-1: WY correction
        for (int t = 1; t < K_TOKENS; t++) {
            // Compute g_product[s+1..t-1] * kd[t][s] * vn[s] for s=0..t-1
            // and the leading term product(g[0..t-1]) * hk[t].
            float corrected = 0.0f;
            // Leading term: prod g[0..t-1] * hk[t]
            float lead_prod = 1.0f;
            for (int u = 0; u < t; u++) lead_prod *= sg[u];
            corrected = lead_prod * hk[t];
            // Cross terms: sum_{s=0..t-1} (prod g[s+1..t-1]) * kd[t][s] * vn[s]
            for (int s = 0; s < t; s++) {
                float gprod = 1.0f;
                for (int u = s + 1; u < t; u++) gprod *= sg[u];
                corrected += gprod * kd_flat[t * (t - 1) / 2 + s] * vn[s];
            }
            vn[t] = (vi[t] - sg[t] * corrected) * sbt[t];
        }

        // ── PASS 2: Apply K state updates in fused loop ──
        // After update t: H_new[t] = g[t] * H_prev + k[t] * vn[t]
        // Write Hi_t = H_new[t] for t=0..K-2; final H = H_new[K-1].
        float qd[K_TOKENS];
        #pragma unroll
        for (int t = 0; t < K_TOKENS; t++) qd[t] = 0.0f;

        #pragma unroll 4
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = H[(j + 0) * v_dim + tid];
            float h1 = H[(j + 1) * v_dim + tid];
            float h2 = H[(j + 2) * v_dim + tid];
            float h3 = H[(j + 3) * v_dim + tid];

            // Apply update for each token sequentially, writing intermediates.
            #pragma unroll
            for (int t = 0; t < K_TOKENS; t++) {
                h0 = sg[t] * h0 + sk[t][j + 0] * vn[t];
                h1 = sg[t] * h1 + sk[t][j + 1] * vn[t];
                h2 = sg[t] * h2 + sk[t][j + 2] * vn[t];
                h3 = sg[t] * h3 + sk[t][j + 3] * vn[t];
                if (t < K_TOKENS - 1) {
                    // Hi_t at offset t * inter_stride_floats from base.
                    // LAZY: only persist checkpoint slots. Skipped slots are
                    // reconstructed bit-exactly by the replay kernel on the
                    // (rare) partial accept that targets them. The register
                    // recurrence (h0..h3) continues regardless, so the running
                    // state and qd[t] accumulation are unaffected.
                    if (wy17_is_checkpoint(t, lazy_j)) {
                        float* Hi_t = Hi_base + t * inter_stride_floats;
                        Hi_t[(j + 0) * v_dim + tid] = h0;
                        Hi_t[(j + 1) * v_dim + tid] = h1;
                        Hi_t[(j + 2) * v_dim + tid] = h2;
                        Hi_t[(j + 3) * v_dim + tid] = h3;
                    }
                } else {
                    // Final state goes to live H.
                    H[(j + 0) * v_dim + tid] = h0;
                    H[(j + 1) * v_dim + tid] = h1;
                    H[(j + 2) * v_dim + tid] = h2;
                    H[(j + 3) * v_dim + tid] = h3;
                }
                qd[t] += h0 * sq[t][j + 0] + h1 * sq[t][j + 1]
                       + h2 * sq[t][j + 2] + h3 * sq[t][j + 3];
            }
        }

        // ── Write outputs (K rows × v_dim) ──
        float s = rsqrtf((float)k_dim);
        #pragma unroll
        for (int t = 0; t < K_TOKENS; t++) {
            output[((b * K_TOKENS + t) * num_v_heads + vh) * v_dim + tid] =
                __float2bfloat16(qd[t] * s);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Entry points.
//
// `gated_delta_rule_wy17`      — historical symbol, writes ALL 16 Hi slots
//                                (lazy_j = 1). BIT-IDENTICAL to the prior
//                                kernel; existing callers are unchanged.
// `gated_delta_rule_wy17_lazy` — takes runtime lazy_j; lazy_j==1 is
//                                bit-identical to `gated_delta_rule_wy17`,
//                                lazy_j>1 writes only checkpoint slots.
// Both dispatch the identical FP32 recurrence body above, so outputs and
// the final h_state are byte-for-byte equal for every lazy_j.
// ─────────────────────────────────────────────────────────────────────
extern "C" __global__ void gated_delta_rule_wy17(
    float* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ h_state_inter_base,
    unsigned int inter_stride_floats,
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride
) {
    gated_delta_rule_wy17_impl(h_state, query, key, value, gate, beta, output,
        h_state_inter_base, inter_stride_floats, batch_size, num_k_heads,
        num_v_heads, k_dim, v_dim, qk_stride, v_stride, gb_stride, /*lazy_j=*/1u);
}

extern "C" __global__ void gated_delta_rule_wy17_lazy(
    float* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ h_state_inter_base,
    unsigned int inter_stride_floats,
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride,
    unsigned int lazy_j
) {
    gated_delta_rule_wy17_impl(h_state, query, key, value, gate, beta, output,
        h_state_inter_base, inter_stride_floats, batch_size, num_k_heads,
        num_v_heads, k_dim, v_dim, qk_stride, v_stride, gb_stride, lazy_j);
}

// ─────────────────────────────────────────────────────────────────────
// gated_delta_rule_wy17_replay — reconstruct ONE intermediate H slot.
//
// Used by the commit path (async_chkpt.rs) ONLY on a partial accept whose
// target slot `target_slot` was skipped under lazy_j>1. It re-seeds from
// the pre-verify ROOT state (`H_ckpt` = the same h_state the main kernel
// read in PASS 1) and re-runs the recurrence for tokens [0 .. target_slot]
// to reproduce H-after-token-`target_slot`, writing it into `out_h` (the
// caller points this at the live h_state directly).
//
// Why re-seed from ROOT (not a nearer J-checkpoint): the main kernel's WY
// correction vn[t] is computed against H_ROOT for the FULL prefix, so a
// bit-exact reproduction of state[target_slot] must recompute vn[0..target]
// from the root. That recompute is cheap — O(target^2) scalar FP32 over
// <=16 tokens, one CTA/head, ONE state read + ONE state write — versus the
// 16 full-state writes the main pass avoids. The J-checkpoints written by
// the main pass let the COMMON partial-accept slots skip replay entirely;
// replay only fires on a non-checkpoint miss.
//
// Bit-exactness: identical FP32 math, identical clamp, identical strict
// left-to-right token order as the main kernel → byte-identical state.
// `ckpt_first_token` is retained for ABI symmetry and MUST be 0 (root
// replay); `target_slot` = the slot to reproduce (inclusive).
// Grid/block identical to the main kernel: (num_v_heads, batch), 128.
extern "C" __global__ void gated_delta_rule_wy17_replay(
    const float* __restrict__ H_ckpt,           // nearest checkpoint state (RO)
    const __nv_bfloat16* __restrict__ query,     // unused (kept for arg symmetry)
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ out_h,                   // reconstructed H-after-target (WO)
    unsigned int ckpt_first_token,               // first token index to replay
    unsigned int target_slot,                    // last token index to replay (inclusive)
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;
    (void)query;

    const unsigned int tid = threadIdx.x;
    const unsigned int hr = num_v_heads / num_k_heads;
    const unsigned int kh = vh / hr;
    const unsigned int hv = k_dim * v_dim;

    const float* Hc = H_ckpt + ((b * num_v_heads + vh) * hv);
    float* Ho = out_h + ((b * num_v_heads + vh) * hv);

    // Only tokens [ckpt_first_token .. target_slot] participate. Load their
    // k/v/gate/beta. We MUST reproduce the same WY correction the main
    // kernel computed for these tokens — which depends on the FULL token
    // history 0..target_slot for kd cross-terms. But the main kernel's
    // corrected[t] uses H_ROOT (pre-verify), NOT the checkpoint. To stay
    // bit-exact we recompute vn[t] exactly as the main kernel did: over the
    // full prefix [0..target_slot] against the ROOT state. Hence the replay
    // is seeded with the ROOT h_state (ckpt_first_token=0, H_ckpt=root) for
    // correctness. The J-checkpoint optimization amortizes the PASS-2 write
    // traffic; PASS-1/WY recompute is cheap (O(target^2) scalar, <=16 tok).
    __shared__ float sk[K_TOKENS][128];
    __shared__ float sg[K_TOKENS];
    __shared__ float sbt[K_TOKENS];
    __shared__ float smem_warp[4];
    __shared__ float kd_flat[K_TOKENS * (K_TOKENS - 1) / 2];

    const unsigned int T = target_slot + 1;   // tokens 0..target_slot

    if (tid < k_dim) {
        for (unsigned int t = 0; t < T; t++) {
            const __nv_bfloat16* k_t = key + (b * K_TOKENS + t) * qk_stride + kh * k_dim;
            sk[t][tid] = (float)k_t[tid];
        }
    }
    if (tid < T) {
        float g_raw = gate[(b * K_TOKENS + tid) * gb_stride + vh];
        sg[tid] = fminf(fmaxf(g_raw, 1e-6f), 1.0f - 1e-6f);
        sbt[tid] = beta[(b * K_TOKENS + tid) * gb_stride + vh];
    }
    __syncthreads();

    for (unsigned int t = 1; t < T; t++) {
        for (unsigned int s = 0; s < t; s++) {
            float p = (tid < k_dim) ? sk[t][tid] * sk[s][tid] : 0.0f;
            float r = atlas_block_reduce_sum(p, smem_warp, tid);
            if (tid == 0) kd_flat[t * (t - 1) / 2 + s] = r;
            __syncthreads();
        }
    }

    if (tid < v_dim) {
        float vi[K_TOKENS];
        for (unsigned int t = 0; t < T; t++) {
            const __nv_bfloat16* v_t = value + (b * K_TOKENS + t) * v_stride + vh * v_dim;
            vi[t] = (float)v_t[tid];
        }

        // PASS 1 over the ROOT state (H_ckpt == root when replaying).
        float hk[K_TOKENS];
        for (unsigned int t = 0; t < T; t++) hk[t] = 0.0f;
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = Hc[(j + 0) * v_dim + tid];
            float h1 = Hc[(j + 1) * v_dim + tid];
            float h2 = Hc[(j + 2) * v_dim + tid];
            float h3 = Hc[(j + 3) * v_dim + tid];
            for (unsigned int t = 0; t < T; t++) {
                hk[t] += h0 * sk[t][j + 0] + h1 * sk[t][j + 1]
                       + h2 * sk[t][j + 2] + h3 * sk[t][j + 3];
            }
        }

        float vn[K_TOKENS];
        vn[0] = (vi[0] - sg[0] * hk[0]) * sbt[0];
        for (unsigned int t = 1; t < T; t++) {
            float lead_prod = 1.0f;
            for (unsigned int u = 0; u < t; u++) lead_prod *= sg[u];
            float corrected = lead_prod * hk[t];
            for (unsigned int s = 0; s < t; s++) {
                float gprod = 1.0f;
                for (unsigned int u = s + 1; u < t; u++) gprod *= sg[u];
                corrected += gprod * kd_flat[t * (t - 1) / 2 + s] * vn[s];
            }
            vn[t] = (vi[t] - sg[t] * corrected) * sbt[t];
        }

        // PASS 2: apply updates 0..target_slot, write only the final state.
        (void)ckpt_first_token;
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = Hc[(j + 0) * v_dim + tid];
            float h1 = Hc[(j + 1) * v_dim + tid];
            float h2 = Hc[(j + 2) * v_dim + tid];
            float h3 = Hc[(j + 3) * v_dim + tid];
            for (unsigned int t = 0; t < T; t++) {
                h0 = sg[t] * h0 + sk[t][j + 0] * vn[t];
                h1 = sg[t] * h1 + sk[t][j + 1] * vn[t];
                h2 = sg[t] * h2 + sk[t][j + 2] * vn[t];
                h3 = sg[t] * h3 + sk[t][j + 3] * vn[t];
            }
            Ho[(j + 0) * v_dim + tid] = h0;
            Ho[(j + 1) * v_dim + tid] = h1;
            Ho[(j + 2) * v_dim + tid] = h2;
            Ho[(j + 3) * v_dim + tid] = h3;
        }
    }
}
