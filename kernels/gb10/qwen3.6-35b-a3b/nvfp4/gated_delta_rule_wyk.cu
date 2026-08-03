// SPDX-License-Identifier: AGPL-3.0-only

// Atlas WY-Chunkwise Gated Delta Rule — GENERAL K family (K = 5..16).
//
// 2026-07-30. Atlas shipped fused GDN verify kernels for K ∈ {2,3,4,17} only
// (gated_delta_rule_wy.cu / _wy3 / _wy4 / _wy17). Every other K fell through to
// the SEQUENTIAL per-token loop — ~34 launches/layer × 30 SSM layers — which
// made intermediate γ unaffordable and left us choosing between cap4 and full
// width. Measured on the serial path, K=8 STILL beat cap4 on MinHeap steady
// (84.5 vs 79.1) with tokens/step 3.88 → 5.67 (+46%), i.e. wide K was winning
// while handicapped. AEON-7's published fastest runs K=11 for the same reason.
//
// This file is `gated_delta_rule_wy17.cu` with `#define K_TOKENS 17` promoted
// to a template parameter, instantiated for every missing K. The algorithm,
// the reduction primitives (gdn_reduce.cuh) and the gate clamp are UNCHANGED
// and therefore bit-exact with the shipped kernels — which is what makes the
// correctness oracle work: run this family at K=4 and diff accept against
// `gated_delta_rule_wy4`; at K=17 against `_wy17`. Any mismatch is a port bug,
// not a numerics tradeoff.
//
// WHY A TEMPLATE AND NOT A RUNTIME K: vi[], hk[], vn[], qd[] are per-thread
// REGISTER arrays, so their extent must be a compile-time constant. A pleasant
// side effect is that smaller K uses fewer registers than K=17 and therefore
// gets BETTER occupancy, so intermediate K is not merely possible but cheaper
// per row than the K=17 path.
//
// SMEM @ K, k_dim=128:  sk+sq = K·128·2·4 B, kd_flat = K(K-1)/2·4 B
//   K=8  → 8.2 KB    K=12 → 12.4 KB    K=16 → 16.6 KB   (SM_120 cap: 100 KB)
//
// Grid: (num_v_heads, batch, 1)   Block: (128, 1, 1)

#include <cuda_bf16.h>
#include "../../common/gdn_reduce.cuh"
#define BLOCK_SIZE 128

// `h_state_inter_base` is a contiguous pool of (K-1) intermediate H states for
// this (layer, slot); stride `inter_stride_floats`. Slot t's intermediate is at
// `h_state_inter_base + t * inter_stride_floats` (per (b, vh) sub-region).
// `h_state` itself receives the final (K-1'th) state.
template <int K_TOKENS>
__device__ __forceinline__ void atlas_gdr_wyk_body(
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

    // ── K*(K-1)/2 k-dot products via block reduction ──
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

        // ── PASS 1: read H once, hk[t] = H · k_t ──
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

        // ── WY correction (sequential over K tokens) ──
        float vn[K_TOKENS];
        vn[0] = (vi[0] - sg[0] * hk[0]) * sbt[0];

        for (int t = 1; t < K_TOKENS; t++) {
            float corrected = 0.0f;
            float lead_prod = 1.0f;
            for (int u = 0; u < t; u++) lead_prod *= sg[u];
            corrected = lead_prod * hk[t];
            for (int s = 0; s < t; s++) {
                float gprod = 1.0f;
                for (int u = s + 1; u < t; u++) gprod *= sg[u];
                corrected += gprod * kd_flat[t * (t - 1) / 2 + s] * vn[s];
            }
            vn[t] = (vi[t] - sg[t] * corrected) * sbt[t];
        }

        // ── PASS 2: apply K state updates, writing intermediates ──
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

// One `extern "C"` entry per K so the Rust side can resolve a symbol by name.
// K=4 and K=17 are instantiated too, purely as CORRECTNESS ORACLES: their
// accept must match `gated_delta_rule_wy4` / `_wy17` exactly.
#define ATLAS_GDR_WYK(N)                                                       \
    extern "C" __global__ void gated_delta_rule_wyk##N(                        \
        float* __restrict__ h_state,                                           \
        const __nv_bfloat16* __restrict__ query,                               \
        const __nv_bfloat16* __restrict__ key,                                 \
        const __nv_bfloat16* __restrict__ value,                               \
        const float* __restrict__ gate,                                        \
        const float* __restrict__ beta,                                        \
        __nv_bfloat16* __restrict__ output,                                    \
        float* __restrict__ h_state_inter_base,                                \
        unsigned int inter_stride_floats,                                      \
        unsigned int batch_size,                                               \
        unsigned int num_k_heads,                                              \
        unsigned int num_v_heads,                                              \
        unsigned int k_dim,                                                    \
        unsigned int v_dim,                                                    \
        unsigned int qk_stride,                                                \
        unsigned int v_stride,                                                 \
        unsigned int gb_stride                                                 \
    ) {                                                                        \
        atlas_gdr_wyk_body<N>(h_state, query, key, value, gate, beta, output,   \
                              h_state_inter_base, inter_stride_floats,         \
                              batch_size, num_k_heads, num_v_heads, k_dim,     \
                              v_dim, qk_stride, v_stride, gb_stride);          \
    }

ATLAS_GDR_WYK(4)
ATLAS_GDR_WYK(5)
ATLAS_GDR_WYK(6)
ATLAS_GDR_WYK(7)
ATLAS_GDR_WYK(8)
ATLAS_GDR_WYK(9)
ATLAS_GDR_WYK(10)
ATLAS_GDR_WYK(11)
ATLAS_GDR_WYK(12)
ATLAS_GDR_WYK(13)
ATLAS_GDR_WYK(14)
ATLAS_GDR_WYK(15)
ATLAS_GDR_WYK(16)
ATLAS_GDR_WYK(17)
