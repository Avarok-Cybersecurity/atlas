// SPDX-License-Identifier: AGPL-3.0-only
// Technique described in arXiv:2609.04852 (KVMem). Independent implementation from the published description; no source copied from kvmem/kvmem-llama.cpp (that repository carries no license).

// kv_page_score_ada.cu: the page-softmax view-block scorer of the bounded KV view (port
// lane L31; host launcher `crates/spark-runtime/src/kv_select/score.rs`, f32 host model
// `kv_select/emulate.rs`, f64 oracle `kv_select/oracle.rs`). ADDITION: no gb10
// counterpart. Runs once per user turn, only when `kv_view_selection` is on.
//
// Per attention layer, for one row tile of the captured query and every view block b
// (pages_per_block consecutive pages of the sequence):
//   1. kv_block_merge_f32       M_b = count-weighted mean of the pages' BF16 sketch means
//                               (f32), wsum_b = tokens behind it (0 = not eligible).
//   2. kv_block_logits_f32      logit[r, h, b] = dot(q[l, r, h], M_b[h / G]) / sqrt(D), or
//                               -inf for an ineligible block. One warp per block: lanes
//                               stride the head dim, a warp xor tree folds them.
//   3. kv_logit_stats_f32       per (r, h): max and sum of exp over the blocks.
//   4. kv_group_norm_f32        (GQA max mode) per (r, g): sum over b of max_h p[r, h, b].
//   5. kv_score_accumulate_f32  scores[b] += weight * sum_r (sum_h p | sum_g max_h p / gnorm).
// Every reduction has one fixed order (thread stride, warp xor tree, warp 0 folds the
// warp sums) and every scores[b] has one writer per launch, so equal inputs give equal
// bits. Scratch is the caller's; nothing is allocated here.

#include <cuda_bf16.h>
#include <math.h>

#define KSC_MAX_ACC 32
#define KSC_BLOCKS_PER_CTA 8

__device__ __forceinline__ float ksc_warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

__device__ __forceinline__ float ksc_warp_max(float v) {
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}

// Block-wide sum for blockDim.x == 256 (8 warps); every thread gets the result.
__device__ __forceinline__ float ksc_block_sum(float v, float* sm) {
    const unsigned int lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    v = ksc_warp_sum(v);
    __syncthreads();
    if (lane == 0) sm[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float w = lane < (blockDim.x >> 5) ? sm[lane] : 0.0f;
        w = ksc_warp_sum(w);
        if (lane == 0) sm[0] = w;
    }
    __syncthreads();
    return sm[0];
}

__device__ __forceinline__ float ksc_block_max(float v, float* sm) {
    const unsigned int lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    v = ksc_warp_max(v);
    __syncthreads();
    if (lane == 0) sm[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float w = lane < (blockDim.x >> 5) ? sm[lane] : -INFINITY;
        w = ksc_warp_max(w);
        if (lane == 0) sm[0] = w;
    }
    __syncthreads();
    return sm[0];
}

// grid (chunk_blocks, kv_heads), block (128): one merged (view block, KV head).
extern "C" __global__ void kv_block_merge_f32(
    const __nv_bfloat16* __restrict__ means, const unsigned char* __restrict__ counts,
    const int* __restrict__ pages, unsigned int n_pages, unsigned int pages_per_block,
    unsigned int layer, unsigned int attn_layers, unsigned int kv_heads, unsigned int head_dim,
    unsigned int block0, float* __restrict__ merged, float* __restrict__ wsum)
{
    const unsigned int lb = blockIdx.x;
    const unsigned int g = blockIdx.y;
    const unsigned int b = block0 + lb;
    const unsigned int p0 = b * pages_per_block;
    const unsigned int p1 = min(p0 + pages_per_block, n_pages);
    float tot = 0.0f;
    for (unsigned int p = p0; p < p1; ++p) tot += (float)counts[pages[p]];
    float* out = merged + ((unsigned long long)lb * kv_heads + g) * head_dim;
    for (unsigned int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (unsigned int p = p0; p < p1; ++p) {
            const unsigned int page = (unsigned int)pages[p];
            const unsigned char w = counts[page];
            if (w != 0) {
                const unsigned long long at =
                    (((unsigned long long)page * attn_layers + layer) * kv_heads + g) * head_dim + d;
                acc += (float)w * __bfloat162float(means[at]);
            }
        }
        out[d] = tot > 0.0f ? acc / tot : 0.0f;
    }
    if (g == 0 && threadIdx.x == 0) wsum[lb] = tot;
}

// grid (ceil(chunk_len / 8), kv_heads, ceil(rows_t / cta_rows)), block (256),
// dynamic smem cta_rows * G * head_dim * 4. Warp w owns view block blockIdx.x * 8 + w.
extern "C" __global__ void kv_block_logits_f32(
    const __nv_bfloat16* __restrict__ q, unsigned int rows_cap, unsigned int layer,
    unsigned int row0, unsigned int rows_t, unsigned int q_heads, unsigned int kv_heads,
    unsigned int head_dim, unsigned int cta_rows, const float* __restrict__ merged,
    const float* __restrict__ wsum, unsigned int chunk0, unsigned int chunk_len,
    unsigned int n_blocks, float* __restrict__ logits, float scale)
{
    extern __shared__ float ksc_qs[];
    const unsigned int G = q_heads / kv_heads;
    const unsigned int g = blockIdx.y;
    const unsigned int r_base = blockIdx.z * cta_rows;
    if (r_base >= rows_t) return;  // uniform per CTA
    const unsigned int n_r = min(cta_rows, rows_t - r_base);
    const unsigned int n_acc = n_r * G;
    for (unsigned int i = threadIdx.x; i < n_acc * head_dim; i += blockDim.x) {
        const unsigned int a = i / head_dim, d = i % head_dim;
        const unsigned int r = a / G, h = g * G + a % G;
        const unsigned long long src =
            (((unsigned long long)layer * rows_cap + row0 + r_base + r) * q_heads + h) * head_dim + d;
        ksc_qs[i] = __bfloat162float(q[src]);
    }
    __syncthreads();
    const unsigned int lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const unsigned int lb = blockIdx.x * KSC_BLOCKS_PER_CTA + warp;
    if (lb >= chunk_len) return;  // uniform per warp, after the only barrier
    const float* m = merged + ((unsigned long long)lb * kv_heads + g) * head_dim;
    float acc[KSC_MAX_ACC];
#pragma unroll
    for (int a = 0; a < KSC_MAX_ACC; ++a) acc[a] = 0.0f;
    for (unsigned int d = lane; d < head_dim; d += 32) {
        const float mv = m[d];
#pragma unroll
        for (int a = 0; a < KSC_MAX_ACC; ++a) {
            if ((unsigned int)a < n_acc) acc[a] += ksc_qs[a * head_dim + d] * mv;
        }
    }
    const bool eligible = wsum[lb] > 0.0f;
    const unsigned int b = chunk0 + lb;
#pragma unroll
    for (int a = 0; a < KSC_MAX_ACC; ++a) {
        if ((unsigned int)a < n_acc) {
            const float v = ksc_warp_sum(acc[a]);
            if (lane == 0) {
                const unsigned int r = (unsigned int)a / G, h = g * G + (unsigned int)a % G;
                logits[((unsigned long long)(r_base + r) * q_heads + h) * n_blocks + b] =
                    eligible ? v * scale : -INFINITY;
            }
        }
    }
}

// grid (rows_t * q_heads), block (256): stats[2 rh] = max, stats[2 rh + 1] = sum exp.
extern "C" __global__ void kv_logit_stats_f32(
    const float* __restrict__ logits, unsigned int n_blocks, float* __restrict__ stats)
{
    __shared__ float sm[32];
    const unsigned int rh = blockIdx.x;
    const float* l = logits + (unsigned long long)rh * n_blocks;
    float mx = -INFINITY;
    for (unsigned int b = threadIdx.x; b < n_blocks; b += blockDim.x) mx = fmaxf(mx, l[b]);
    mx = ksc_block_max(mx, sm);
    float s = 0.0f;
    if (mx != -INFINITY) {
        for (unsigned int b = threadIdx.x; b < n_blocks; b += blockDim.x) s += expf(l[b] - mx);
    }
    s = ksc_block_sum(s, sm);
    if (threadIdx.x == 0) {
        stats[2 * rh] = mx;
        stats[2 * rh + 1] = mx != -INFINITY ? s : 0.0f;
    }
}

__device__ __forceinline__ float ksc_prob(const float* logits, const float* stats,
                                          unsigned int rh, unsigned int n_blocks, unsigned int b) {
    const float m = stats[2 * rh], s = stats[2 * rh + 1];
    return s > 0.0f ? expf(logits[(unsigned long long)rh * n_blocks + b] - m) / s : 0.0f;
}

// grid (rows_t * kv_heads), block (256): gnorm[r, g] = sum_b max_{h in g} p[r, h, b].
extern "C" __global__ void kv_group_norm_f32(
    const float* __restrict__ logits, const float* __restrict__ stats, unsigned int n_blocks,
    unsigned int q_heads, unsigned int kv_heads, float* __restrict__ gnorm)
{
    __shared__ float sm[32];
    const unsigned int G = q_heads / kv_heads;
    const unsigned int r = blockIdx.x / kv_heads, g = blockIdx.x % kv_heads;
    float s = 0.0f;
    for (unsigned int b = threadIdx.x; b < n_blocks; b += blockDim.x) {
        float mx = 0.0f;
        for (unsigned int i = 0; i < G; ++i) {
            mx = fmaxf(mx, ksc_prob(logits, stats, r * q_heads + g * G + i, n_blocks, b));
        }
        s += mx;
    }
    s = ksc_block_sum(s, sm);
    if (threadIdx.x == 0) gnorm[blockIdx.x] = s;
}

// grid (ceil(n_blocks / 256)), block (256): one thread per view block, rows then heads in
// order; mode 0 = mean over heads, 1 = per-group max renormalized by gnorm.
extern "C" __global__ void kv_score_accumulate_f32(
    const float* __restrict__ logits, const float* __restrict__ stats,
    const float* __restrict__ gnorm, unsigned int rows_t, unsigned int q_heads,
    unsigned int kv_heads, unsigned int n_blocks, unsigned int mode, float weight,
    float* __restrict__ scores)
{
    const unsigned int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;
    const unsigned int G = q_heads / kv_heads;
    float acc = 0.0f;
    for (unsigned int r = 0; r < rows_t; ++r) {
        if (mode == 0) {
            for (unsigned int h = 0; h < q_heads; ++h) {
                acc += ksc_prob(logits, stats, r * q_heads + h, n_blocks, b);
            }
        } else {
            for (unsigned int g = 0; g < kv_heads; ++g) {
                const float gn = gnorm[r * kv_heads + g];
                if (gn > 0.0f) {
                    float mx = 0.0f;
                    for (unsigned int i = 0; i < G; ++i) {
                        mx = fmaxf(mx, ksc_prob(logits, stats, r * q_heads + g * G + i, n_blocks, b));
                    }
                    acc += mx / gn;
                }
            }
        }
    }
    scores[b] += acc * weight;
}
