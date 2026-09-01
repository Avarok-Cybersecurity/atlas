// SPDX-License-Identifier: AGPL-3.0-only

// Atlas WY-Chunkwise Gated Delta Rule — K∈{5..16} verification (wyN).
//
// K-templated generalization of gated_delta_rule_wy17.cu (which itself
// generalizes wy4). One __device__ impl, instantiated for the chain-verify
// widths between the dedicated wy4 and the DFlash wy17 — mirroring the
// w4a16_gemv_batchm_impl<MAX_M> instantiation pattern in w4a16_gemv.cu.
// Removes the serial per-token GDN fallback at chain-verify K=5..8.
//
// Algorithm (identical WY-chunkwise structure — "2 passes over H
// regardless of K"):
//   1. Load q[K], k[K] into SMEM (K KB at k_dim=128).
//   2. Compute K*(K-1)/2 inter-token k-dot products via block reduction.
//   3. PASS 1: read H once, compute hk[K] = pre-update H·k[t] dots.
//   4. WY correction (sequential over K tokens): produce vn[K].
//   5. PASS 2: apply K state updates in single fused loop, writing
//      Hi_t = state after token t for t=0..K-2, and final H = state
//      after token K-1.
//
// SMEM budget @ K=8, k_dim=128:
//   sk[8][128] + sq[8][128] = 8·128·2·4 = 8 KB
//   kdots[28] + gate/beta[16] + warp_sums[4]  < 0.25 KB
//   (SM_120 cap: 100 KB — trivially fits for every instantiation)
//
// Grid: (num_v_heads, batch, 1)   Block: (128, 1, 1)
// Reduction primitives (gdn_reduce.cuh) match the per-token baseline
// bit-exactly; the gate clamp MUST match per-token gated_delta_rule_decode
// (see gated_delta_rule_wy.cu — drift here flips argmax on long verifies).

#include <cuda_bf16.h>
#include "gdn_reduce.cuh"
#include "gdn_f16_state.cuh"
#define BLOCK_SIZE 128

// `h_state_inter_base` is a contiguous pool of (K-1) intermediate H states
// for this (layer, slot). Stride between intermediates is
// `inter_stride_floats` floats. Slot t's intermediate lives at
// `h_state_inter_base + t * inter_stride_floats` (per (b, vh) sub-region).
// `h_state` itself becomes the final (post token K-1) state — same
// pool-layout contract as gated_delta_rule_wy17.

template <int K_TOKENS>
__device__ __forceinline__ void gated_delta_rule_wyn_impl(
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
    // Same contract as wy2/wy3/wy4:
    // 0 = h_state / h_state_inter_base are CONTIGUOUS bases indexed by
    //     (b*num_v_heads+vh), Hi_t at t*inter_stride_floats — the C=1
    //     chain-verify form, byte-identical to the original;
    // 1 = both args are device POINTER TABLES: h_state is a slab of
    //     `batch_size` per-sequence H base pointers, h_state_inter_base
    //     points at the Hi0 slab, and `inter_stride_floats` is
    //     REINTERPRETED as the number of POINTER ENTRIES between
    //     consecutive Hi slabs (= VERIFY_WY_TABLE_SEQS). This reaches
    //     Hi_t for any t without adding K-1 arguments.
    unsigned int state_is_table
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int hr = num_v_heads / num_k_heads;
    const unsigned int kh = vh / hr;
    const unsigned int hv = k_dim * v_dim;

    const unsigned long long head_off = (unsigned long long)vh * hv;
    const unsigned long long flat_off =
        (unsigned long long)(b * num_v_heads + vh) * hv;
    float* H = state_is_table ? ((float* const*)h_state)[b] + head_off
                              : h_state + flat_off;
    // Per-t Hi pointers, hoisted once per block. Contiguous form: base +
    // t*stride at the (b, vh) offset (identical math to the original
    // Hi_base). Table form: slab t's per-sequence entry, plus head_off.
    float* Hi_ptrs[K_TOKENS - 1];
    #pragma unroll
    for (int t = 0; t < K_TOKENS - 1; t++) {
        Hi_ptrs[t] = state_is_table
            ? ((float* const*)h_state_inter_base)[b + (unsigned long long)t * inter_stride_floats]
                  + head_off
            : h_state_inter_base + flat_off
                  + (unsigned long long)t * inter_stride_floats;
    }

    // ── Load q, k, gate, beta into SMEM ──
    __shared__ float sk[K_TOKENS][128];
    __shared__ float sq[K_TOKENS][128];
    __shared__ float sg[K_TOKENS];   // gate clamped
    __shared__ float sbt[K_TOKENS];  // beta
    __shared__ float smem_warp[4];

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

    // ── Compute K*(K-1)/2 k-dot products via block reduction ──
    // kd[t][s] = k_t · k_s for s < t, stored sparsely at tri_idx(t,s) =
    // t*(t-1)/2 + s.
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
        float vn[K_TOKENS];
        vn[0] = (vi[0] - sg[0] * hk[0]) * sbt[0];
        for (int t = 1; t < K_TOKENS; t++) {
            float lead_prod = 1.0f;
            for (int u = 0; u < t; u++) lead_prod *= sg[u];
            float corrected = lead_prod * hk[t];
            for (int s = 0; s < t; s++) {
                float gprod = 1.0f;
                for (int u = s + 1; u < t; u++) gprod *= sg[u];
                corrected += gprod * kd_flat[t * (t - 1) / 2 + s] * vn[s];
            }
            vn[t] = (vi[t] - sg[t] * corrected) * sbt[t];
        }

        // ── PASS 2: Apply K state updates in fused loop ──
        // After update t: H_new[t] = g[t] * H_prev + k[t] * vn[t].
        // Write Hi_t for t=0..K-2; final H = H_new[K-1].
        float qd[K_TOKENS];
        #pragma unroll
        for (int t = 0; t < K_TOKENS; t++) qd[t] = 0.0f;

        #pragma unroll 4
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = H[(j + 0) * v_dim + tid];
            float h1 = H[(j + 1) * v_dim + tid];
            float h2 = H[(j + 2) * v_dim + tid];
            float h3 = H[(j + 3) * v_dim + tid];

            #pragma unroll
            for (int t = 0; t < K_TOKENS; t++) {
                h0 = sg[t] * h0 + sk[t][j + 0] * vn[t];
                h1 = sg[t] * h1 + sk[t][j + 1] * vn[t];
                h2 = sg[t] * h2 + sk[t][j + 2] * vn[t];
                h3 = sg[t] * h3 + sk[t][j + 3] * vn[t];
                if (t < K_TOKENS - 1) {
                    float* Hi_t = Hi_ptrs[t];
                    Hi_t[(j + 0) * v_dim + tid] = h0;
                    Hi_t[(j + 1) * v_dim + tid] = h1;
                    Hi_t[(j + 2) * v_dim + tid] = h2;
                    Hi_t[(j + 3) * v_dim + tid] = h3;
                } else {
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

// Instantiations for chain-verify K=5..8. The argument list is identical to
// gated_delta_rule_wy17; the Rust side selects the handle by num_tokens.
#define ATLAS_WYN_INSTANTIATE(K)                                              \
    extern "C" __global__ void gated_delta_rule_wy##K(                        \
        float* __restrict__ h_state,                                          \
        const __nv_bfloat16* __restrict__ query,                              \
        const __nv_bfloat16* __restrict__ key,                                \
        const __nv_bfloat16* __restrict__ value,                              \
        const float* __restrict__ gate,                                       \
        const float* __restrict__ beta,                                       \
        __nv_bfloat16* __restrict__ output,                                   \
        float* __restrict__ h_state_inter_base,                               \
        unsigned int inter_stride_floats,                                     \
        unsigned int batch_size,                                              \
        unsigned int num_k_heads,                                             \
        unsigned int num_v_heads,                                             \
        unsigned int k_dim,                                                   \
        unsigned int v_dim,                                                   \
        unsigned int qk_stride,                                               \
        unsigned int v_stride,                                                \
        unsigned int gb_stride                                                \
    ) {                                                                       \
        gated_delta_rule_wyn_impl<K>(                                         \
            h_state, query, key, value, gate, beta, output,                   \
            h_state_inter_base, inter_stride_floats, batch_size,              \
            num_k_heads, num_v_heads, k_dim, v_dim, qk_stride, v_stride,      \
            gb_stride, 0u);                                                   \
    }                                                                         \
    /* Cross-sequence batched-verify twin: same impl, state args are     */   \
    /* device POINTER TABLES (see the impl's state_is_table contract).   */   \
    /* ADDITIVE symbol so the contiguous form's signature (shared with   */   \
    /* the per-target wy17 launches via gdn_decode_wyn) never changes.   */   \
    extern "C" __global__ void gated_delta_rule_wy##K##_table(                \
        float* __restrict__ h_state,                                          \
        const __nv_bfloat16* __restrict__ query,                              \
        const __nv_bfloat16* __restrict__ key,                                \
        const __nv_bfloat16* __restrict__ value,                              \
        const float* __restrict__ gate,                                       \
        const float* __restrict__ beta,                                       \
        __nv_bfloat16* __restrict__ output,                                   \
        float* __restrict__ h_state_inter_base,                               \
        unsigned int inter_stride_floats,                                     \
        unsigned int batch_size,                                              \
        unsigned int num_k_heads,                                             \
        unsigned int num_v_heads,                                             \
        unsigned int k_dim,                                                   \
        unsigned int v_dim,                                                   \
        unsigned int qk_stride,                                               \
        unsigned int v_stride,                                                \
        unsigned int gb_stride                                                \
    ) {                                                                       \
        gated_delta_rule_wyn_impl<K>(                                         \
            h_state, query, key, value, gate, beta, output,                   \
            h_state_inter_base, inter_stride_floats, batch_size,              \
            num_k_heads, num_v_heads, k_dim, v_dim, qk_stride, v_stride,      \
            gb_stride, 1u);                                                   \
    }

ATLAS_WYN_INSTANTIATE(5)
ATLAS_WYN_INSTANTIATE(6)
ATLAS_WYN_INSTANTIATE(7)
ATLAS_WYN_INSTANTIATE(8)
// K=9..16 (2026-08-29): the γ>8 window class. Before these, K=9..16 was the
// ONLY un-served verify width band — it fell to the sequential per-token
// fallback (per token: conv launch + gdn launch + 2 state D2Ds, across every
// GDN layer; ~1200 serial launches/step at K=10). Measured motivation: γ10
// probe on Qwen3.8-27B DFlash2 chained accept past the trained block
// (tok_step 6.474 -> 7.661, +18%) but net tok/s LOST to that loop. SMEM
// scales as K KB (K=16: 16 KB q+k) against the 100 KB cap — trivially fits.
// provenance-id: 526f6e616c6420522e205374657369616b
ATLAS_WYN_INSTANTIATE(9)
ATLAS_WYN_INSTANTIATE(10)
ATLAS_WYN_INSTANTIATE(11)
ATLAS_WYN_INSTANTIATE(12)
ATLAS_WYN_INSTANTIATE(13)
ATLAS_WYN_INSTANTIATE(14)
ATLAS_WYN_INSTANTIATE(15)
ATLAS_WYN_INSTANTIATE(16)

#undef ATLAS_WYN_INSTANTIATE

// ── FP16 h-state twins — stage 2 of ATLAS_SSM_H_FP16, DFlash widths ──────
//
// MECHANICALLY DERIVED from gated_delta_rule_wyn_impl above: every float
// expression, gate clamp, accumulation order and reduction is the parent's,
// unchanged. The only edits are dtype ones — the h-state and its rollback
// intermediates are `__half` in memory, loaded through `__half2float` and
// stored through `gdn_f16_store` (gdn_f16_state.cuh). Arithmetic stays FP32
// in registers. PER-TOKEN ROUND-TRIP per the wy2/wy3/wy4 _f16 contract:
// each update is rounded to FP16 BEFORE it is stored, carried to the next
// token, or fed to the q-dot — so the forward chain's value is bit-for-bit
// what a rollback restores.
//
// Motivation (2026-08-29, #812): the FP16 h-pool is the lever that lets
// MTP serve bs=128; DFlash was refused it because these widths had no f16
// twins ("an FP32 kernel over an FP16 h-state emits fluent garbage").
// With wyn covering K=5..16, these twins give the Qwen3.8 DFlash target
// (gamma <= 16) full FP16 coverage and roughly halve the gamma-sized
// verify blobs behind the ~49 GB pool intercept measured there.
// `inter_stride_halves` is the pool pitch in HALF elements (h pitch bytes/2).
// provenance-id: 526f6e616c6420522e205374657369616b

template <int K_TOKENS>
__device__ __forceinline__ void gated_delta_rule_wyn_f16_impl(
    __half* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ output,
    __half* __restrict__ h_state_inter_base,
    unsigned int inter_stride_halves,
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride,
    // Same contract as the FP32 impl above: 0 = contiguous bases (C=1
    // form, byte-identical); 1 = h_state / h_state_inter_base are device
    // POINTER TABLES (per-sequence __half* entries), and
    // `inter_stride_halves` is REINTERPRETED as the number of POINTER
    // ENTRIES between consecutive Hi slabs (= VERIFY_WY_TABLE_SEQS).
    unsigned int state_is_table
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int hr = num_v_heads / num_k_heads;
    const unsigned int kh = vh / hr;
    const unsigned int hv = k_dim * v_dim;

    const unsigned long long head_off = (unsigned long long)vh * hv;
    const unsigned long long flat_off =
        (unsigned long long)(b * num_v_heads + vh) * hv;
    __half* H = state_is_table ? ((__half* const*)h_state)[b] + head_off
                               : h_state + flat_off;
    __half* Hi_ptrs[K_TOKENS - 1];
    #pragma unroll
    for (int t = 0; t < K_TOKENS - 1; t++) {
        Hi_ptrs[t] = state_is_table
            ? ((__half* const*)h_state_inter_base)[b + (unsigned long long)t * inter_stride_halves]
                  + head_off
            : h_state_inter_base + flat_off
                  + (unsigned long long)t * inter_stride_halves;
    }

    __shared__ float sk[K_TOKENS][128];
    __shared__ float sq[K_TOKENS][128];
    __shared__ float sg[K_TOKENS];
    __shared__ float sbt[K_TOKENS];
    __shared__ float smem_warp[4];

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
        float g_raw = gate[(b * K_TOKENS + tid) * gb_stride + vh];
        sg[tid] = fminf(fmaxf(g_raw, 1e-6f), 1.0f - 1e-6f);
        sbt[tid] = beta[(b * K_TOKENS + tid) * gb_stride + vh];
    }
    __syncthreads();

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
        float vi[K_TOKENS];
        #pragma unroll
        for (int t = 0; t < K_TOKENS; t++) {
            const __nv_bfloat16* v_t = value + (b * K_TOKENS + t) * v_stride + vh * v_dim;
            vi[t] = (float)v_t[tid];
        }

        float hk[K_TOKENS];
        #pragma unroll
        for (int t = 0; t < K_TOKENS; t++) hk[t] = 0.0f;

        #pragma unroll 4
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = __half2float(H[(j + 0) * v_dim + tid]);
            float h1 = __half2float(H[(j + 1) * v_dim + tid]);
            float h2 = __half2float(H[(j + 2) * v_dim + tid]);
            float h3 = __half2float(H[(j + 3) * v_dim + tid]);
            #pragma unroll
            for (int t = 0; t < K_TOKENS; t++) {
                hk[t] += h0 * sk[t][j + 0] + h1 * sk[t][j + 1]
                       + h2 * sk[t][j + 2] + h3 * sk[t][j + 3];
            }
        }

        float vn[K_TOKENS];
        vn[0] = (vi[0] - sg[0] * hk[0]) * sbt[0];
        for (int t = 1; t < K_TOKENS; t++) {
            float lead_prod = 1.0f;
            for (int u = 0; u < t; u++) lead_prod *= sg[u];
            float corrected = lead_prod * hk[t];
            for (int s = 0; s < t; s++) {
                float gprod = 1.0f;
                for (int u = s + 1; u < t; u++) gprod *= sg[u];
                corrected += gprod * kd_flat[t * (t - 1) / 2 + s] * vn[s];
            }
            vn[t] = (vi[t] - sg[t] * corrected) * sbt[t];
        }

        float qd[K_TOKENS];
        #pragma unroll
        for (int t = 0; t < K_TOKENS; t++) qd[t] = 0.0f;

        #pragma unroll 4
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = __half2float(H[(j + 0) * v_dim + tid]);
            float h1 = __half2float(H[(j + 1) * v_dim + tid]);
            float h2 = __half2float(H[(j + 2) * v_dim + tid]);
            float h3 = __half2float(H[(j + 3) * v_dim + tid]);

            #pragma unroll
            for (int t = 0; t < K_TOKENS; t++) {
                h0 = sg[t] * h0 + sk[t][j + 0] * vn[t];
                h1 = sg[t] * h1 + sk[t][j + 1] * vn[t];
                h2 = sg[t] * h2 + sk[t][j + 2] * vn[t];
                h3 = sg[t] * h3 + sk[t][j + 3] * vn[t];
                // Per-token FP16 round-trip BEFORE store, carry, and q-dot
                // (the wy3_f16 contract, verbatim).
                h0 = __half2float(gdn_f16_store(h0));
                h1 = __half2float(gdn_f16_store(h1));
                h2 = __half2float(gdn_f16_store(h2));
                h3 = __half2float(gdn_f16_store(h3));
                if (t < K_TOKENS - 1) {
                    __half* Hi_t = Hi_ptrs[t];
                    Hi_t[(j + 0) * v_dim + tid] = gdn_f16_store(h0);
                    Hi_t[(j + 1) * v_dim + tid] = gdn_f16_store(h1);
                    Hi_t[(j + 2) * v_dim + tid] = gdn_f16_store(h2);
                    Hi_t[(j + 3) * v_dim + tid] = gdn_f16_store(h3);
                } else {
                    H[(j + 0) * v_dim + tid] = gdn_f16_store(h0);
                    H[(j + 1) * v_dim + tid] = gdn_f16_store(h1);
                    H[(j + 2) * v_dim + tid] = gdn_f16_store(h2);
                    H[(j + 3) * v_dim + tid] = gdn_f16_store(h3);
                }
                qd[t] += h0 * sq[t][j + 0] + h1 * sq[t][j + 1]
                       + h2 * sq[t][j + 2] + h3 * sq[t][j + 3];
            }
        }

        float s = rsqrtf((float)k_dim);
        #pragma unroll
        for (int t = 0; t < K_TOKENS; t++) {
            output[((b * K_TOKENS + t) * num_v_heads + vh) * v_dim + tid] =
                __float2bfloat16(qd[t] * s);
        }
    }
}

#define ATLAS_WYN_F16_INSTANTIATE(K)                                          \
    extern "C" __global__ void gated_delta_rule_wy##K##_f16(                  \
        __half* __restrict__ h_state,                                         \
        const __nv_bfloat16* __restrict__ query,                              \
        const __nv_bfloat16* __restrict__ key,                                \
        const __nv_bfloat16* __restrict__ value,                              \
        const float* __restrict__ gate,                                       \
        const float* __restrict__ beta,                                       \
        __nv_bfloat16* __restrict__ output,                                   \
        __half* __restrict__ h_state_inter_base,                              \
        unsigned int inter_stride_halves,                                      \
        unsigned int batch_size,                                              \
        unsigned int num_k_heads,                                             \
        unsigned int num_v_heads,                                             \
        unsigned int k_dim,                                                   \
        unsigned int v_dim,                                                   \
        unsigned int qk_stride,                                               \
        unsigned int v_stride,                                                \
        unsigned int gb_stride                                                \
    ) {                                                                       \
        gated_delta_rule_wyn_f16_impl<K>(                                     \
            h_state, query, key, value, gate, beta, output,                   \
            h_state_inter_base, inter_stride_halves, batch_size,               \
            num_k_heads, num_v_heads, k_dim, v_dim, qk_stride, v_stride,      \
            gb_stride, 0u);                                                   \
    }                                                                         \
    /* Batched-verify pointer-table twin — see the FP32 macro's note. */      \
    extern "C" __global__ void gated_delta_rule_wy##K##_f16_table(            \
        __half* __restrict__ h_state,                                         \
        const __nv_bfloat16* __restrict__ query,                              \
        const __nv_bfloat16* __restrict__ key,                                \
        const __nv_bfloat16* __restrict__ value,                              \
        const float* __restrict__ gate,                                       \
        const float* __restrict__ beta,                                       \
        __nv_bfloat16* __restrict__ output,                                   \
        __half* __restrict__ h_state_inter_base,                              \
        unsigned int inter_stride_halves,                                      \
        unsigned int batch_size,                                              \
        unsigned int num_k_heads,                                             \
        unsigned int num_v_heads,                                             \
        unsigned int k_dim,                                                   \
        unsigned int v_dim,                                                   \
        unsigned int qk_stride,                                               \
        unsigned int v_stride,                                                \
        unsigned int gb_stride                                                \
    ) {                                                                       \
        gated_delta_rule_wyn_f16_impl<K>(                                     \
            h_state, query, key, value, gate, beta, output,                   \
            h_state_inter_base, inter_stride_halves, batch_size,               \
            num_k_heads, num_v_heads, k_dim, v_dim, qk_stride, v_stride,      \
            gb_stride, 1u);                                                   \
    }

ATLAS_WYN_F16_INSTANTIATE(5)
ATLAS_WYN_F16_INSTANTIATE(6)
ATLAS_WYN_F16_INSTANTIATE(7)
ATLAS_WYN_F16_INSTANTIATE(8)
ATLAS_WYN_F16_INSTANTIATE(9)
ATLAS_WYN_F16_INSTANTIATE(10)
ATLAS_WYN_F16_INSTANTIATE(11)
ATLAS_WYN_F16_INSTANTIATE(12)
ATLAS_WYN_F16_INSTANTIATE(13)
ATLAS_WYN_F16_INSTANTIATE(14)
ATLAS_WYN_F16_INSTANTIATE(15)
ATLAS_WYN_F16_INSTANTIATE(16)

#undef ATLAS_WYN_F16_INSTANTIATE
