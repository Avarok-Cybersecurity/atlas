// SPDX-License-Identifier: AGPL-3.0-only

// Atlas WY-Chunkwise Gated Delta Rule: K-generic WRITE-ON-ACCEPT verify for
// the cross-sequence batched verify (Ada-owned, port lane L10, row
// `gdn_woa_mtp`).
//
// Design after the ReplaySSM record-and-fold idea (Cinference / NInfer,
// docs/maintainer/replayssm-gdn.md, Apache-2.0): verify keeps the state out
// of DRAM, writes only the small per-row update terms, and after the verdict
// folds the accepted prefix once. No upstream code is copied; the arithmetic
// below is Atlas's own `gated_delta_rule_wyn_impl` (kernels/gb10/common/
// gated_delta_rule_wyn.cu), reproduced expression for expression.
//
// The parents this pair stands in for (table form, one pointer entry per
// sequence, the slab layout `upload_verify_wy_tables` stages):
//   K = 2      gated_delta_rule_wy2 (and its register-resident twin)
//   K = 3      gated_delta_rule_wy3 (and its register-resident twin)
//   K = 5..16  gated_delta_rule_wy{K}_table (the wyN template)
// K = 4 keeps the existing gb10-common pair (gated_delta_rule_wy4_woa / _fold),
// so no target's K=4 behavior changes.
//
// Each parent reads H twice and writes K full state widths per launch (Hi_0 ..
// Hi_{K-2} plus the final H), and a partial accept then copies one of them
// back over H. This pair:
//
//   gated_delta_rule_wyn_woa_<K>: pass 1 reads the head's H slice ONCE into
//     dynamic shared memory (k_dim x v_dim floats, 64 KB at 128 x 128) and
//     computes the K k-dots; the WY correction; pass 2 applies the K
//     in-register updates from shared memory and computes the K q-dots.
//     Identical expressions, in the identical order, to the parent, so
//     `output` is byte-identical. It writes NO state. It stashes what the
//     fold needs: the K per-row vn vectors, the K clamped gate scalars and
//     the K key rows, as the same float values the parent used.
//
//   gated_delta_rule_wyn_fold: after the host knows how many rows each
//     sequence accepted (na in 1..=K), applies rows 0..na-1 to H with the
//     parent's update expression `h = g*h + k[j]*vn` in row order, reading H
//     once and writing it once. H_na equals the parent's Hi(na-1), or its
//     final H at na == K.
//
// Traffic per step and head: parent 2 reads + K writes (+1 restore copy on a
// partial accept) -> here 1 read (twin) + 1 read + 1 write (fold).
//
// BYTE IDENTITY rests on three facts, each checked by the GPU oracle
// (`crates/spark-model/examples/gdn_wyn_woa_oracle.rs`) before the row may
// flip:
//   1. the expression trees match the parent's (the wy2/wy3 explicit forms
//      are the wyN template at K = 2, 3: `1.0f * x` and the empty-product
//      gate terms are exact);
//   2. `common/KERNEL.toml` builds this directory with `--fmad=false`, so no
//      multiply-add is contracted differently in the twin than in the parent;
//   3. the shared-memory copy of H holds the same floats DRAM held.
//
// The ENGAGED word protocol is the K=4 pair's (spark-model qwen3_ssm/woa.rs):
// the twin sets the layer's word to 1 inside the captured graph, the host
// clears it at the top of every requesting batched verify, and the fold reads
// it: 1 folds the stash, 0 performs the parent's partial-accept restore from
// the Hi tables (the parent ran), so the host never has to know which kernel a
// graph replayed.
//
// Grid (num_v_heads, batch), block 128. Host-checked, never re-checked here:
// k_dim == v_dim == 128 (woa.rs `woa_decision`), K within the bound stash
// (the model sizes the stash for its widest batched-verify width).
//
// Shared memory at K = 16: sk + sq 16 KB static, H slice 64 KB dynamic, under
// the 99 KB sm_89 opt-in (the launch path sets the dynamic-size attribute for
// every launch over 48 KB, avarok-core registry.rs).

#include <cuda_bf16.h>
#include "gdn_reduce.cuh"

#define WYN_WOA_BLOCK 128
#define WYN_FOLD_MAX 16

// Stash layout per sequence for verify width K (floats):
//   vn[K][num_v_heads][v_dim] | g[K][num_v_heads] | sk[K][num_k_heads][k_dim]
// At K = 4 this is exactly the gb10 wy4 pair's layout. The host passes the
// per-sequence float count, K * (nv*vd + nv + nk*kd)
// (`spark_model::port::gdn_woa::stash_seq_floats_k`).
#define WYNW_VN(base, T, VH) \
    ((base) + ((unsigned long long)(T) * num_v_heads + (VH)) * v_dim)
#define WYNW_G(base, K, T, VH) \
    ((base) + (unsigned long long)(K) * num_v_heads * v_dim \
            + (unsigned long long)(T) * num_v_heads + (VH))
#define WYNW_SK(base, K, T, KH) \
    ((base) + (unsigned long long)(K) * num_v_heads * v_dim \
            + (unsigned long long)(K) * num_v_heads \
            + ((unsigned long long)(T) * num_k_heads + (KH)) * k_dim)

template <int K_TOKENS>
__device__ __forceinline__ void gated_delta_rule_wyn_woa_impl(
    const float* __restrict__ h_state_table,   // float* const[batch]
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ stash,
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride,
    unsigned int stash_seq_floats,
    unsigned int* __restrict__ engaged_flag
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;
    // Dims are the HOST's check: an in-kernel early return would leave
    // `output` unwritten with no kernel having run in its place.
    if (threadIdx.x == 0) *engaged_flag = 1u;

    const unsigned int tid = threadIdx.x;
    const unsigned int hr = num_v_heads / num_k_heads;
    const unsigned int kh = vh / hr;
    const unsigned int hv = k_dim * v_dim;
    const unsigned long long head_off = (unsigned long long)vh * hv;
    const float* H = ((const float* const*)h_state_table)[b] + head_off;
    float* sb = stash + (unsigned long long)b * stash_seq_floats;

    // ── Load q, k, gate, beta into SMEM (the parent's loads, verbatim) ──
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
        // Key rows for the fold: one writer per k-head (the first v-head that
        // maps to it), the same float values this thread just loaded.
        if (vh % hr == 0) {
            #pragma unroll
            for (int t = 0; t < K_TOKENS; t++) {
                WYNW_SK(sb, K_TOKENS, t, kh)[tid] = sk[t][tid];
            }
        }
    }
    if (tid < K_TOKENS) {
        // Gate clamp matches per-token gated_delta_rule_decode (see wy4).
        float g_raw = gate[(b * K_TOKENS + tid) * gb_stride + vh];
        sg[tid] = fminf(fmaxf(g_raw, 1e-6f), 1.0f - 1e-6f);
        sbt[tid] = beta[(b * K_TOKENS + tid) * gb_stride + vh];
        *WYNW_G(sb, K_TOKENS, tid, vh) = sg[tid];
    }
    __syncthreads();

    // ── K*(K-1)/2 k-dot products via block reduction (verbatim) ──
    __shared__ float kd_flat[K_TOKENS * (K_TOKENS - 1) / 2];

    #pragma unroll
    for (int t = 1; t < K_TOKENS; t++) {
        #pragma unroll
        for (int s = 0; s < t; s++) {
            float p = (tid < k_dim) ? sk[t][tid] * sk[s][tid] : 0.0f;
            float r = avarok_block_reduce_sum(p, smem_warp, tid);
            if (tid == 0) {
                kd_flat[t * (t - 1) / 2 + s] = r;
            }
            __syncthreads();
        }
    }

    // The head's H slice lives in DYNAMIC shared memory for the launch: pass 1
    // fills it from DRAM once, pass 2 reads it back. Column-per-thread
    // indexing (j * v_dim + tid) is bank-conflict-free.
    extern __shared__ float hs[];

    if (tid < v_dim) {
        float vi[K_TOKENS];
        #pragma unroll
        for (int t = 0; t < K_TOKENS; t++) {
            const __nv_bfloat16* v_t = value + (b * K_TOKENS + t) * v_stride + vh * v_dim;
            vi[t] = (float)v_t[tid];
        }

        // ── PASS 1: read H ONCE (DRAM -> smem), hk[t] = H . k_t ──
        float hk[K_TOKENS];
        #pragma unroll
        for (int t = 0; t < K_TOKENS; t++) hk[t] = 0.0f;

        #pragma unroll 4
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = H[(j + 0) * v_dim + tid];
            float h1 = H[(j + 1) * v_dim + tid];
            float h2 = H[(j + 2) * v_dim + tid];
            float h3 = H[(j + 3) * v_dim + tid];
            hs[(j + 0) * v_dim + tid] = h0;
            hs[(j + 1) * v_dim + tid] = h1;
            hs[(j + 2) * v_dim + tid] = h2;
            hs[(j + 3) * v_dim + tid] = h3;
            #pragma unroll
            for (int t = 0; t < K_TOKENS; t++) {
                hk[t] += h0 * sk[t][j + 0] + h1 * sk[t][j + 1]
                       + h2 * sk[t][j + 2] + h3 * sk[t][j + 3];
            }
        }

        // ── WY correction (sequential over K tokens, verbatim) ──
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

        // Stash the per-row update vectors for the fold.
        #pragma unroll
        for (int t = 0; t < K_TOKENS; t++) {
            WYNW_VN(sb, t, vh)[tid] = vn[t];
        }

        // ── PASS 2: the K updates from smem, NO state writes ──
        float qd[K_TOKENS];
        #pragma unroll
        for (int t = 0; t < K_TOKENS; t++) qd[t] = 0.0f;

        #pragma unroll 4
        for (unsigned int j = 0; j < k_dim; j += 4) {
            float h0 = hs[(j + 0) * v_dim + tid];
            float h1 = hs[(j + 1) * v_dim + tid];
            float h2 = hs[(j + 2) * v_dim + tid];
            float h3 = hs[(j + 3) * v_dim + tid];

            #pragma unroll
            for (int t = 0; t < K_TOKENS; t++) {
                h0 = sg[t] * h0 + sk[t][j + 0] * vn[t];
                h1 = sg[t] * h1 + sk[t][j + 1] * vn[t];
                h2 = sg[t] * h2 + sk[t][j + 2] * vn[t];
                h3 = sg[t] * h3 + sk[t][j + 3] * vn[t];
                qd[t] += h0 * sq[t][j + 0] + h1 * sq[t][j + 1]
                       + h2 * sq[t][j + 2] + h3 * sq[t][j + 3];
            }
        }

        // ── Outputs (K rows x v_dim), the parent's rounding ──
        float s = rsqrtf((float)k_dim);
        #pragma unroll
        for (int t = 0; t < K_TOKENS; t++) {
            output[((b * K_TOKENS + t) * num_v_heads + vh) * v_dim + tid] =
                __float2bfloat16(qd[t] * s);
        }
    }
}

// One table-form entry point per width. The argument list is exactly the K=4
// twin's (`gated_delta_rule_wy4_woa`), so spark-model launches every width
// through the same `ops::gdn_decode_wy4_woa` launcher with this handle.
#define AVAROK_WYN_WOA_INSTANTIATE(K)                                          \
    extern "C" __global__ void gated_delta_rule_wyn_woa_##K(                  \
        const float* __restrict__ h_state_table,                              \
        const __nv_bfloat16* __restrict__ query,                              \
        const __nv_bfloat16* __restrict__ key,                                \
        const __nv_bfloat16* __restrict__ value,                              \
        const float* __restrict__ gate,                                       \
        const float* __restrict__ beta,                                       \
        __nv_bfloat16* __restrict__ output,                                   \
        float* __restrict__ stash,                                            \
        unsigned int batch_size,                                              \
        unsigned int num_k_heads,                                             \
        unsigned int num_v_heads,                                             \
        unsigned int k_dim,                                                   \
        unsigned int v_dim,                                                   \
        unsigned int qk_stride,                                               \
        unsigned int v_stride,                                                \
        unsigned int gb_stride,                                               \
        unsigned int stash_seq_floats,                                        \
        unsigned int* __restrict__ engaged_flag                               \
    ) {                                                                       \
        gated_delta_rule_wyn_woa_impl<K>(                                     \
            h_state_table, query, key, value, gate, beta, output, stash,      \
            batch_size, num_k_heads, num_v_heads, k_dim, v_dim, qk_stride,    \
            v_stride, gb_stride, stash_seq_floats, engaged_flag);             \
    }

AVAROK_WYN_WOA_INSTANTIATE(2)
AVAROK_WYN_WOA_INSTANTIATE(3)
// K = 4 is the gb10-common pair (gated_delta_rule_wy4_woa.cu); not repeated.
AVAROK_WYN_WOA_INSTANTIATE(5)
AVAROK_WYN_WOA_INSTANTIATE(6)
AVAROK_WYN_WOA_INSTANTIATE(7)
AVAROK_WYN_WOA_INSTANTIATE(8)
AVAROK_WYN_WOA_INSTANTIATE(9)
AVAROK_WYN_WOA_INSTANTIATE(10)
AVAROK_WYN_WOA_INSTANTIATE(11)
AVAROK_WYN_WOA_INSTANTIATE(12)
AVAROK_WYN_WOA_INSTANTIATE(13)
AVAROK_WYN_WOA_INSTANTIATE(14)
AVAROK_WYN_WOA_INSTANTIATE(15)
AVAROK_WYN_WOA_INSTANTIATE(16)

#undef AVAROK_WYN_WOA_INSTANTIATE

// Apply the accepted rows 0..na-1 (na = na_tab[b], 0..=k_rows) to H, once.
// Same argument list as `gated_delta_rule_wy4_fold`, so the same launcher
// serves both; `k_rows` is the verify width K of the launch that ran and also
// selects the stash layout above.
//
// `hi_tables`: the verify's Hi0 slab; Hi(t) is `slab_entries` pointers further
// on (the parent's intermediates). When `engaged_flag` is 0 the parent ran and
// this launch performs the parent's partial-accept restore instead (H =
// Hi[na-1], nothing at na >= k_rows): byte for byte the wy4 fold's flag-0 arm.
extern "C" __global__ void gated_delta_rule_wyn_fold(
    float* __restrict__ h_state_table,          // float* const[batch]
    const float* __restrict__ stash,
    const unsigned int* __restrict__ na_tab,
    const float* __restrict__ hi_tables,        // float* const[slab_entries][batch]
    unsigned int slab_entries,
    const unsigned int* __restrict__ engaged_flag,
    unsigned int k_rows,
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int stash_seq_floats
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;
    unsigned int na = na_tab[b];
    if (na == 0) return;

    const unsigned int tid = threadIdx.x;
    const unsigned int hr = num_v_heads / num_k_heads;
    const unsigned int kh = vh / hr;
    const unsigned int hv = k_dim * v_dim;
    float* H = ((float* const*)h_state_table)[b] + (unsigned long long)vh * hv;
    const float* sb = stash + (unsigned long long)b * stash_seq_floats;

    if (*engaged_flag == 0u) {
        // The parent ran: it wrote H (final) and Hi0..Hi(K-2). A full accept
        // keeps H; a partial accept of na rows restores Hi(na-1).
        if (na >= k_rows) return;
        const float* src = ((const float* const*)hi_tables)[(na - 1) * slab_entries + b]
                           + (unsigned long long)vh * hv;
        for (unsigned int i = tid; i < hv; i += WYN_WOA_BLOCK) H[i] = src[i];
        return;
    }

    // A verdict never accepts more rows than the launch verified; the clamp
    // keeps a host slip inside the stash instead of reading the next row band.
    if (na > k_rows) na = k_rows;
    if (na > WYN_FOLD_MAX) na = WYN_FOLD_MAX;

    __shared__ float sk[WYN_FOLD_MAX][128];
    __shared__ float svn[WYN_FOLD_MAX][128];
    __shared__ float sg[WYN_FOLD_MAX];
    if (tid < k_dim) {
        for (unsigned int t = 0; t < na; t++) {
            sk[t][tid] = WYNW_SK(sb, k_rows, t, kh)[tid];
        }
    }
    if (tid < v_dim) {
        for (unsigned int t = 0; t < na; t++) {
            svn[t][tid] = WYNW_VN(sb, t, vh)[tid];
        }
    }
    if (tid < na) {
        sg[tid] = *WYNW_G(sb, k_rows, tid, vh);
    }
    __syncthreads();
    if (tid >= v_dim) return;

    // The parent's update expression, row by row, on each element (dims are
    // host-checked; no guard here either).
    for (unsigned int j = 0; j < k_dim; j += 4) {
        float h0 = H[(j + 0) * v_dim + tid];
        float h1 = H[(j + 1) * v_dim + tid];
        float h2 = H[(j + 2) * v_dim + tid];
        float h3 = H[(j + 3) * v_dim + tid];
        for (unsigned int t = 0; t < na; t++) {
            const float g = sg[t];
            const float v = svn[t][tid];
            h0 = g * h0 + sk[t][j + 0] * v;
            h1 = g * h1 + sk[t][j + 1] * v;
            h2 = g * h2 + sk[t][j + 2] * v;
            h3 = g * h3 + sk[t][j + 3] * v;
        }
        H[(j + 0) * v_dim + tid] = h0;
        H[(j + 1) * v_dim + tid] = h1;
        H[(j + 2) * v_dim + tid] = h2;
        H[(j + 3) * v_dim + tid] = h3;
    }
}

#undef WYNW_VN
#undef WYNW_G
#undef WYNW_SK
