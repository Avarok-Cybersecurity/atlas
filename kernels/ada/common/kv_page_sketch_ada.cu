// SPDX-License-Identifier: AGPL-3.0-only
// Technique described in arXiv:2609.04852 (KVMem). Independent implementation from the published description; no source copied from kvmem/kvmem-llama.cpp (that repository carries no license).
// Block key-mean statistic per arXiv:2609.04852; independent implementation.

// kv_page_sketch_ada.cu: the per-page pre-RoPE key sketch of the bounded KV view
// (port lane L31; host side `crates/spark-runtime/src/kv_cache/sketch.rs`), and the
// in-place inverse RoPE of captured query rows (`port/kv_view/query_capture.rs`).
// ADDITION: no gb10 counterpart. Nothing here runs unless `kv_view_selection` is on.
//
// Sketch layout:
//   means  [num_blocks, attn_layers, kv_heads, head_dim]  BF16, the mean over the page's
//          stored tokens of the key after k-norm and BEFORE RoPE;
//   counts [num_blocks]                                   u8, the tokens behind the mean.
// Paged K layout (NHD, `reshape_and_cache.cu`): page `b` starts at `b * block_stride_bytes`
// and holds `[block_size, kv_heads, head_dim]` BF16, or FP8 E4M3 with `value = fp8 * scale`.
//
// One CTA recomputes one (page, KV head) from scratch over the page's `fill` stored tokens:
// each thread owns dims d = tid, tid + blockDim, ...; a rotary dim reads its NeoX partner
// (d +- rotary_dim/2) and undoes the rotation at the token's absolute position
// (logical page * block_size + offset), with the frequency evaluated in f64 exactly like
// `fused_k_norm_rope_cache.cu`; the sum is FP32 in token order, divided by `fill`, rounded
// to BF16 once. Recomputing (not accumulating) makes every update idempotent.
//
// Entry points:
//   kv_page_sketch_jobs_{bf16,fp8}   grid (n_jobs, kv_heads): a host-planned job list
//                                    (prefill, eager). Job = i32 {block or -1, stream,
//                                    logical page, fill}; -1 resolves through
//                                    block_table_ptrs[stream][logical page].
//   kv_page_sketch_tail_{bf16,fp8}   grid (n_seqs * pages_span, kv_heads): the pages the
//                                    last n_new tokens of each sequence landed in, from
//                                    the step's device block table and lengths (decode and
//                                    verify, graph-safe: no host value changes per step).
//   kv_query_unrope_bf16             grid (rows, heads): undo RoPE in place on captured
//                                    BF16 query rows at positions pos0, pos0 + 1, ...

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <math.h>

__device__ __forceinline__ float kps_load(const unsigned char* __restrict__ page,
                                          unsigned int idx, int is_fp8, float scale) {
    if (is_fp8) {
        const __half_raw hr = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)page[idx], __NV_E4M3);
        return __half2float(__half(hr)) * scale;
    }
    const __nv_bfloat16* p = reinterpret_cast<const __nv_bfloat16*>(page);
    return __bfloat162float(p[idx]);
}

// Same expression as fused_k_norm_rope_cache.cu (f64 pow, then one f32 round).
__device__ __forceinline__ float kps_freq(unsigned int pair, unsigned int rotary_dim, float theta) {
    const double e = (double)(2u * pair) / (double)rotary_dim;
    return (float)(1.0 / pow((double)theta, e));
}

__device__ void kps_sketch_page(
    const unsigned char* __restrict__ k_pool, unsigned long long block_stride_bytes,
    int is_fp8, float kv_scale, unsigned int block, unsigned int logical_page,
    unsigned int fill, unsigned int kv_head, __nv_bfloat16* __restrict__ means,
    unsigned char* __restrict__ counts, unsigned int attn_layer, unsigned int attn_layers,
    unsigned int kv_heads, unsigned int head_dim, unsigned int block_size,
    unsigned int rotary_dim, float theta)
{
    __nv_bfloat16* dst = means
        + (((unsigned long long)block * attn_layers + attn_layer) * kv_heads + kv_head) * head_dim;
    const unsigned char* page = k_pool + (unsigned long long)block * block_stride_bytes;
    const unsigned int half_rot = rotary_dim / 2;
    for (unsigned int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        if (d < rotary_dim) {
            const bool is_d0 = d < half_rot;
            const unsigned int pair = is_d0 ? d : d - half_rot;
            const unsigned int partner = is_d0 ? d + half_rot : d - half_rot;
            const float freq = kps_freq(pair, rotary_dim, theta);
            for (unsigned int o = 0; o < fill; ++o) {
                const unsigned int row = (o * kv_heads + kv_head) * head_dim;
                const float y = kps_load(page, row + d, is_fp8, kv_scale);
                const float yp = kps_load(page, row + partner, is_fp8, kv_scale);
                const unsigned int pos = logical_page * block_size + o;
                const float angle = (float)pos * freq;
                const float c = cosf(angle);
                const float s = sinf(angle);
                // Forward: y0 = x0 c - x1 s, y1 = x1 c + x0 s. Inverse below.
                acc += is_d0 ? (y * c + yp * s) : (y * c - yp * s);
            }
        } else {
            for (unsigned int o = 0; o < fill; ++o) {
                acc += kps_load(page, (o * kv_heads + kv_head) * head_dim + d, is_fp8, kv_scale);
            }
        }
        dst[d] = __float2bfloat16(fill > 0 ? acc / (float)fill : 0.0f);
    }
    if (threadIdx.x == 0 && kv_head == 0) {
        counts[block] = (unsigned char)fill;
    }
}

__device__ __forceinline__ void kps_jobs_body(
    const unsigned char* k_pool, unsigned long long block_stride_bytes, float kv_scale,
    const int* __restrict__ jobs, unsigned int n_jobs,
    const unsigned long long* __restrict__ block_table_ptrs, unsigned int max_blocks_per_seq,
    __nv_bfloat16* means, unsigned char* counts, unsigned int num_blocks,
    unsigned int attn_layer, unsigned int attn_layers, unsigned int kv_heads,
    unsigned int head_dim, unsigned int block_size, unsigned int rotary_dim, float theta,
    int is_fp8)
{
    const unsigned int j = blockIdx.x;
    if (j >= n_jobs) return;
    int block = jobs[4 * j + 0];
    const unsigned int stream = (unsigned int)jobs[4 * j + 1];
    const unsigned int lp = (unsigned int)jobs[4 * j + 2];
    unsigned int fill = (unsigned int)jobs[4 * j + 3];
    if (block < 0) {
        if (block_table_ptrs == nullptr || lp >= max_blocks_per_seq) return;
        const int* table = reinterpret_cast<const int*>(block_table_ptrs[stream]);
        if (table == nullptr) return;
        block = table[lp];
    }
    if (block < 0 || (unsigned int)block >= num_blocks) return;
    if (fill > block_size) fill = block_size;
    kps_sketch_page(k_pool, block_stride_bytes, is_fp8, kv_scale, (unsigned int)block, lp,
                    fill, blockIdx.y, means, counts, attn_layer, attn_layers, kv_heads,
                    head_dim, block_size, rotary_dim, theta);
}

#define KPS_JOBS_PARAMS                                                                     \
    const unsigned char* k_pool, unsigned long long block_stride_bytes, float kv_scale,    \
    const int* jobs, unsigned int n_jobs, const unsigned long long* block_table_ptrs,      \
    unsigned int max_blocks_per_seq, __nv_bfloat16* means, unsigned char* counts,          \
    unsigned int num_blocks, unsigned int attn_layer, unsigned int attn_layers,            \
    unsigned int kv_heads, unsigned int head_dim, unsigned int block_size,                 \
    unsigned int rotary_dim, float theta

#define KPS_JOBS_ARGS                                                                       \
    k_pool, block_stride_bytes, kv_scale, jobs, n_jobs, block_table_ptrs,                   \
    max_blocks_per_seq, means, counts, num_blocks, attn_layer, attn_layers, kv_heads,       \
    head_dim, block_size, rotary_dim, theta

extern "C" __global__ void kv_page_sketch_jobs_bf16(KPS_JOBS_PARAMS) {
    kps_jobs_body(KPS_JOBS_ARGS, 0);
}

extern "C" __global__ void kv_page_sketch_jobs_fp8(KPS_JOBS_PARAMS) {
    kps_jobs_body(KPS_JOBS_ARGS, 1);
}

__device__ __forceinline__ void kps_tail_body(
    const unsigned char* k_pool, unsigned long long block_stride_bytes, float kv_scale,
    const int* __restrict__ block_table, const int* __restrict__ seq_lens,
    unsigned int max_blocks_per_seq, unsigned int n_seqs, unsigned int rows_per_seq,
    unsigned int n_new, unsigned int pages_span, __nv_bfloat16* means, unsigned char* counts,
    unsigned int num_blocks, unsigned int attn_layer, unsigned int attn_layers,
    unsigned int kv_heads, unsigned int head_dim, unsigned int block_size,
    unsigned int rotary_dim, float theta, int is_fp8)
{
    const unsigned int s = blockIdx.x / pages_span;
    const unsigned int j = blockIdx.x % pages_span;
    if (s >= n_seqs) return;
    // The last row of a sequence's group carries its full extent (verify rows are one apart).
    const unsigned int row = s * rows_per_seq + rows_per_seq - 1;
    const int ext_i = seq_lens[row];
    if (ext_i <= 0) return;  // padding row
    const unsigned int ext = (unsigned int)ext_i;
    const unsigned int first = ext > n_new ? ext - n_new : 0u;
    const unsigned int lp = first / block_size + j;
    if (lp * block_size >= ext || lp >= max_blocks_per_seq) return;
    const unsigned int fill = min(block_size, ext - lp * block_size);
    const int block = block_table[(unsigned long long)row * max_blocks_per_seq + lp];
    if (block < 0 || (unsigned int)block >= num_blocks) return;
    kps_sketch_page(k_pool, block_stride_bytes, is_fp8, kv_scale, (unsigned int)block, lp,
                    fill, blockIdx.y, means, counts, attn_layer, attn_layers, kv_heads,
                    head_dim, block_size, rotary_dim, theta);
}

#define KPS_TAIL_PARAMS                                                                     \
    const unsigned char* k_pool, unsigned long long block_stride_bytes, float kv_scale,    \
    const int* block_table, const int* seq_lens, unsigned int max_blocks_per_seq,          \
    unsigned int n_seqs, unsigned int rows_per_seq, unsigned int n_new,                    \
    unsigned int pages_span, __nv_bfloat16* means, unsigned char* counts,                  \
    unsigned int num_blocks, unsigned int attn_layer, unsigned int attn_layers,            \
    unsigned int kv_heads, unsigned int head_dim, unsigned int block_size,                 \
    unsigned int rotary_dim, float theta

#define KPS_TAIL_ARGS                                                                       \
    k_pool, block_stride_bytes, kv_scale, block_table, seq_lens, max_blocks_per_seq,        \
    n_seqs, rows_per_seq, n_new, pages_span, means, counts, num_blocks, attn_layer,         \
    attn_layers, kv_heads, head_dim, block_size, rotary_dim, theta

extern "C" __global__ void kv_page_sketch_tail_bf16(KPS_TAIL_PARAMS) {
    kps_tail_body(KPS_TAIL_ARGS, 0);
}

extern "C" __global__ void kv_page_sketch_tail_fp8(KPS_TAIL_PARAMS) {
    kps_tail_body(KPS_TAIL_ARGS, 1);
}

// In place: q[r * row_stride + h * head_dim + d] for r < rows, h < heads. Thread i owns
// rotary pair (i, i + rotary_dim / 2) of row blockIdx.x, head blockIdx.y; position pos0 + r.
extern "C" __global__ void kv_query_unrope_bf16(
    __nv_bfloat16* __restrict__ q, unsigned int rows, unsigned int row_stride,
    unsigned int heads, unsigned int head_dim, unsigned int rotary_dim, unsigned int pos0,
    float theta)
{
    const unsigned int r = blockIdx.x;
    const unsigned int h = blockIdx.y;
    if (r >= rows || h >= heads) return;
    __nv_bfloat16* v = q + (unsigned long long)r * row_stride + (unsigned long long)h * head_dim;
    const unsigned int half_rot = rotary_dim / 2;
    const unsigned int pos = pos0 + r;
    for (unsigned int i = threadIdx.x; i < half_rot; i += blockDim.x) {
        const float freq = kps_freq(i, rotary_dim, theta);
        const float angle = (float)pos * freq;
        const float c = cosf(angle);
        const float s = sinf(angle);
        const float y0 = __bfloat162float(v[i]);
        const float y1 = __bfloat162float(v[i + half_rot]);
        v[i] = __float2bfloat16(y0 * c + y1 * s);
        v[i + half_rot] = __float2bfloat16(y1 * c - y0 * s);
    }
}
