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
    unsigned int gb_stride
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
                    float* Hi_t = Hi_base + t * inter_stride_floats;
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
            gb_stride);                                                       \
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
