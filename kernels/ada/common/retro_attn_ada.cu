// SPDX-License-Identifier: AGPL-3.0-only
// Algorithm: RetroInfer, arXiv 2505.02922 (VLDB 2026), Section 4.2, Eq 2-5 (estimation zone
// as weighted centroid pseudo-tokens, merged with the exact zone in one softmax).
// Translated for Atlas from microsoft/RetrievalAttention (RetroInfer), MIT License,
// Copyright (c) Microsoft Corporation. Upstream: cache_hub/retroinfer_cache.py:771-815
// (estimation-zone `weighted_flash_decoding` over centroids, value sums and cluster sizes,
// then the exact zone merged through `previous_out` / `previous_lse`). Changes: one single
// pass over the union instead of two attention calls and an LSE merge; centroids, value sums
// and sizes read in place through the estimation id list (no gather copy); the body is a fork
// of Atlas's own `kernels/gb10/common/paged_decode_attn_bf16_gqa.cu` (head_dim 256,
// register-resident Q, fixed per-warp partition and merge tree, no atomics) with a per-entry
// weight, a per-row trim of the shared exact buffer, and an explicit __syncwarp in the merge.

// Ada (sm_89) RetroInfer estimation attention (port lane L34b): five entries,
//
//   retro_est_attn_g{1,2,4,6,8}      G = query heads per kv head (GQA group)
//
// ── MATH (per row r, per query head j of kv head h = j / G) ────────────────
//
//   exact entries i < valid_len[h] - trim_r with position >= 0:   s_i = (q . k_i) * scale
//   estimation clusters c (counts[h * 2 + 1] ids, size n_c > 0):  s_c = (q . centroid_c) * scale
//
//   m   = max( max_i s_i , max_c s_c )
//   num = sum_i exp(s_i - m) v_i  +  sum_c exp(s_c - m) value_sum_c
//   den = sum_i exp(s_i - m)      +  sum_c n_c exp(s_c - m)
//   out = num / den                   (zeros when nothing is attended)
//   lse = m + ln(den)                 (-inf when nothing is attended)
//
// trim_r = row_trim + (rows - 1 - r): the exact buffer ends with the verify rows' own
// positions, so row r of a launch drops the entries of the rows after it (state.rs
// `ExactBuf::row_trim`; one row per launch gives trim = row_trim). The paper's two-call
// merge (estimation output and LSE fed to the exact call as previous_out / previous_lse)
// equals this single pass in exact arithmetic; with one token per cluster it is dense
// attention.
//
// ── LAYOUT (crates/spark-model/src/port/retro/state.rs) ────────────────────
//
//   Q          BF16 [rows][q_row_stride], query head j at j * 256
//   ek, ev     BF16 [kv_heads][cap][256]   the gather's exact buffer
//   epos       i32  [kv_heads][cap]        -1 = skip
//   valid      i32  [kv_heads]
//   est_ids    i32  [kv_heads][es_cap]     estimation-zone ids, rank order
//   counts     i32  [kv_heads][2]          retrieval count, estimation count
//   centroids  BF16 [kv_heads][cap_centroids][256]
//   value_sum  BF16 [kv_heads][cap_centroids][256]
//   csize      i32  [kv_heads][cap_centroids]   0 = empty, never attended
//   O          BF16 [rows][q_heads][256]
//   LSE        F32  [rows][q_heads], may be NULL
//
// ── DETERMINISM ────────────────────────────────────────────────────────────
//
// grid (kv_heads, rows), block 256 (8 warps). Warp w takes the w-th contiguous eighth of the
// row's exact entries, then the w-th eighth of the estimation ids, in index order, four per
// step (the parent's BC = 4 batching); dot products reduce with the parent's XOR shuffle tree;
// warps merge through the parent's fixed shared-memory tree. The partition depends only on
// device counts, so equal inputs give bitwise-equal outputs. Static shared memory:
// 8 * 256 * 4 + 64 bytes, far under 48 KiB.

#include <cuda_bf16.h>

#define RA_HDIM 256
#define RA_WARPS 8
#define RA_THREADS (RA_WARPS * 32)
#define RA_VEC (RA_HDIM / 32)
#define RA_U32 (RA_VEC / 2)
#define RA_BC 4
#define RA_FULL 0xffffffffu

__device__ __forceinline__ void ra_unpack2(unsigned int packed, float& v0, float& v1) {
    v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFF)));
    v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
}

__device__ __forceinline__ void ra_load(const __nv_bfloat16* row, unsigned int lane,
                                        float (&dst)[RA_VEC]) {
    const uint4 p = *reinterpret_cast<const uint4*>(row + lane * RA_VEC);
    ra_unpack2(p.x, dst[0], dst[1]);
    ra_unpack2(p.y, dst[2], dst[3]);
    ra_unpack2(p.z, dst[4], dst[5]);
    ra_unpack2(p.w, dst[6], dst[7]);
}

__device__ __forceinline__ void ra_zero(float (&dst)[RA_VEC]) {
    #pragma unroll
    for (int i = 0; i < RA_VEC; i++) dst[i] = 0.0f;
}

// One online-softmax step over RA_BC entries for all G query heads. `valid` and `w` are
// warp-uniform, so every lane takes the same branches and every shuffle is full-warp.
template <int G>
__device__ __forceinline__ void ra_step(
    const float (&q)[G][RA_VEC], const float (&kf)[RA_BC][RA_VEC],
    const float (&vf)[RA_BC][RA_VEC], const float (&w)[RA_BC], const bool (&valid)[RA_BC],
    float scale, float (&m)[G], float (&l)[G], float (&o)[G][RA_VEC]) {
    #pragma unroll
    for (int g = 0; g < G; g++) {
        float s[RA_BC];
        #pragma unroll
        for (int b = 0; b < RA_BC; b++) {
            float dot = 0.0f;
            #pragma unroll
            for (int i = 0; i < RA_VEC; i++) dot += q[g][i] * kf[b][i];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(RA_FULL, dot, off);
            s[b] = dot * scale;
        }
        float m_new = m[g];
        bool any = false;
        #pragma unroll
        for (int b = 0; b < RA_BC; b++) {
            if (valid[b]) {
                m_new = fmaxf(m_new, s[b]);
                any = true;
            }
        }
        if (!any) continue;
        const float exp_old = __expf(m[g] - m_new);
        l[g] *= exp_old;
        #pragma unroll
        for (int i = 0; i < RA_VEC; i++) o[g][i] *= exp_old;
        #pragma unroll
        for (int b = 0; b < RA_BC; b++) {
            if (!valid[b]) continue;
            const float ef = __expf(s[b] - m_new);
            l[g] += w[b] * ef;
            #pragma unroll
            for (int i = 0; i < RA_VEC; i++) o[g][i] += ef * vf[b][i];
        }
        m[g] = m_new;
    }
}

template <int G>
__device__ __forceinline__ void ra_body(
    const __nv_bfloat16* __restrict__ Q,
    unsigned int q_row_stride,
    unsigned int rows,
    unsigned int row_trim,
    const __nv_bfloat16* __restrict__ ek,
    const __nv_bfloat16* __restrict__ ev,
    const int* __restrict__ epos,
    const int* __restrict__ valid_len,
    unsigned int cap,
    const int* __restrict__ est_ids,
    const int* __restrict__ counts,
    unsigned int es_cap,
    const __nv_bfloat16* __restrict__ centroids,
    const __nv_bfloat16* __restrict__ vsum,
    const int* __restrict__ csize,
    unsigned int cap_centroids,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    unsigned int q_heads,
    unsigned int kv_heads,
    float scale) {
    const unsigned int h = blockIdx.x;
    const unsigned int row = blockIdx.y;
    if (h >= kv_heads) return;               // uniform per CTA
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid / 32;
    const unsigned int lane = tid % 32;
    const unsigned int qh0 = h * G;
    const long long trim = (long long)row_trim + (long long)(rows - 1u - row);

    float q[G][RA_VEC];
    #pragma unroll
    for (int g = 0; g < G; g++) {
        ra_load(Q + (unsigned long long)row * q_row_stride
                    + (unsigned long long)(qh0 + g) * RA_HDIM, lane, q[g]);
    }
    float m[G];
    float l[G];
    float o[G][RA_VEC];
    #pragma unroll
    for (int g = 0; g < G; g++) {
        m[g] = -1e30f;
        l[g] = 0.0f;
        ra_zero(o[g]);
    }

    // Exact zone: this warp's contiguous eighth of the row's entries.
    const long long n_valid = max((long long)min(valid_len[h], (int)cap) - trim, 0ll);
    const int ne = (int)n_valid;
    const int chunk = (ne + RA_WARPS - 1) / RA_WARPS;
    const int e_begin = min((int)warp * chunk, ne);
    const int e_end = min(e_begin + chunk, ne);
    const unsigned long long head_base = (unsigned long long)h * cap;
    for (int e = e_begin; e < e_end; e += RA_BC) {
        float kf[RA_BC][RA_VEC];
        float vf[RA_BC][RA_VEC];
        float w[RA_BC];
        bool valid[RA_BC];
        #pragma unroll
        for (int b = 0; b < RA_BC; b++) {
            const int idx = e + b;
            valid[b] = idx < e_end && epos[head_base + idx] >= 0;
            w[b] = 1.0f;
            if (valid[b]) {
                ra_load(ek + (head_base + idx) * RA_HDIM, lane, kf[b]);
                ra_load(ev + (head_base + idx) * RA_HDIM, lane, vf[b]);
            } else {
                ra_zero(kf[b]);
                ra_zero(vf[b]);
            }
        }
        ra_step<G>(q, kf, vf, w, valid, scale, m, l, o);
    }

    // Estimation zone: this warp's contiguous eighth of the estimation ids.
    const int nc = max(0, min(counts[h * 2 + 1], (int)es_cap));
    const int cchunk = (nc + RA_WARPS - 1) / RA_WARPS;
    const int c_begin = min((int)warp * cchunk, nc);
    const int c_end = min(c_begin + cchunk, nc);
    const int* ids = est_ids + (unsigned long long)h * es_cap;
    const unsigned long long idx_base = (unsigned long long)h * cap_centroids;
    for (int c = c_begin; c < c_end; c += RA_BC) {
        float kf[RA_BC][RA_VEC];
        float vf[RA_BC][RA_VEC];
        float w[RA_BC];
        bool valid[RA_BC];
        #pragma unroll
        for (int b = 0; b < RA_BC; b++) {
            const int idx = c + b;
            const int id = idx < c_end ? ids[idx] : -1;
            const bool in_range = id >= 0 && (unsigned int)id < cap_centroids;
            const int n = in_range ? csize[idx_base + (unsigned int)id] : 0;
            valid[b] = n > 0;
            w[b] = (float)max(n, 0);
            if (valid[b]) {
                ra_load(centroids + (idx_base + (unsigned int)id) * RA_HDIM, lane, kf[b]);
                ra_load(vsum + (idx_base + (unsigned int)id) * RA_HDIM, lane, vf[b]);
            } else {
                ra_zero(kf[b]);
                ra_zero(vf[b]);
            }
        }
        ra_step<G>(q, kf, vf, w, valid, scale, m, l, o);
    }

    // Per-head epilogue through one shared buffer: the parent kernel's fixed merge tree.
    __shared__ float smem_m[RA_WARPS];
    __shared__ float smem_l[RA_WARPS];
    __shared__ float smem_o[RA_WARPS][RA_HDIM];
    const unsigned int vec_offset = lane * RA_VEC;

    #pragma unroll
    for (int g = 0; g < G; g++) {
        __syncthreads();
        if (lane == 0) {
            smem_m[warp] = m[g];
            smem_l[warp] = l[g];
        }
        #pragma unroll
        for (int i = 0; i < RA_VEC; i++) smem_o[warp][vec_offset + i] = o[g][i];
        __syncthreads();

        #pragma unroll
        for (int stride = RA_WARPS / 2; stride > 0; stride >>= 1) {
            if (warp < (unsigned int)stride) {
                const unsigned int other = warp + stride;
                const float lw = smem_l[other];
                if (lw > 0.0f) {
                    const float mw = smem_m[other];
                    const float my_m = smem_m[warp];
                    const float my_l = smem_l[warp];
                    const float m_new = fmaxf(my_m, mw);
                    const float scale_me = __expf(my_m - m_new);
                    const float scale_w = __expf(mw - m_new);
                    // Every lane reads the running max and sum before lane 0 updates them.
                    __syncwarp();
                    if (lane == 0) {
                        smem_l[warp] = my_l * scale_me + lw * scale_w;
                        smem_m[warp] = m_new;
                    }
                    #pragma unroll
                    for (int i = 0; i < RA_VEC; i++) {
                        smem_o[warp][vec_offset + i] = smem_o[warp][vec_offset + i] * scale_me
                            + smem_o[other][vec_offset + i] * scale_w;
                    }
                }
            }
            __syncthreads();
        }

        if (warp == 0) {
            const float final_l = smem_l[0];
            const float inv_l = (final_l > 0.0f) ? (1.0f / final_l) : 0.0f;
            const unsigned long long qh = (unsigned long long)row * q_heads + qh0 + g;
            unsigned int* o32 = reinterpret_cast<unsigned int*>(O + qh * RA_HDIM + vec_offset);
            #pragma unroll
            for (int i = 0; i < RA_U32; i++) {
                const float v0 = smem_o[0][vec_offset + 2 * i] * inv_l;
                const float v1 = smem_o[0][vec_offset + 2 * i + 1] * inv_l;
                const unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
                const unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
                o32[i] = lo | (hi << 16);
            }
            if (LSE != nullptr && lane == 0) {
                LSE[qh] = (final_l > 0.0f) ? (smem_m[0] + logf(final_l))
                                           : __int_as_float(0xff800000);   // -inf
            }
        }
    }
}

#define RA_PARAMS                                                                              \
    const __nv_bfloat16* __restrict__ Q, unsigned int q_row_stride, unsigned int rows,         \
    unsigned int row_trim, const __nv_bfloat16* __restrict__ ek,                               \
    const __nv_bfloat16* __restrict__ ev, const int* __restrict__ epos,                        \
    const int* __restrict__ valid_len, unsigned int cap, const int* __restrict__ est_ids,      \
    const int* __restrict__ counts, unsigned int es_cap,                                       \
    const __nv_bfloat16* __restrict__ centroids, const __nv_bfloat16* __restrict__ vsum,       \
    const int* __restrict__ csize, unsigned int cap_centroids, __nv_bfloat16* __restrict__ O,  \
    float* __restrict__ LSE, unsigned int q_heads, unsigned int kv_heads, float scale

#define RA_ARGS                                                                                \
    Q, q_row_stride, rows, row_trim, ek, ev, epos, valid_len, cap, est_ids, counts, es_cap,    \
    centroids, vsum, csize, cap_centroids, O, LSE, q_heads, kv_heads, scale

#define RA_ENTRY(NAME, G)                                                                      \
    extern "C" __global__ void __launch_bounds__(RA_THREADS, 1) NAME(RA_PARAMS) {              \
        ra_body<G>(RA_ARGS);                                                                   \
    }

RA_ENTRY(retro_est_attn_g1, 1)
RA_ENTRY(retro_est_attn_g2, 2)
RA_ENTRY(retro_est_attn_g4, 4)
RA_ENTRY(retro_est_attn_g6, 6)
RA_ENTRY(retro_est_attn_g8, 8)
