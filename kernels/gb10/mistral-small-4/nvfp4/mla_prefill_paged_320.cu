// SPDX-License-Identifier: AGPL-3.0-only

// Paged MLA Prefill Attention — absorbed form, HDIM=320.
//
// Multi-chunk prefill for MLA models: Q tokens from the current chunk (in
// absorbed form, [q_len, nq, 320]) attend to the full KV history from the
// paged cache ([kv_len, 1, 320]) with causal masking.
//
// This kernel is used when seq_len_start > 0 (chunks 2+).  The KV cache has
// already been written for the current chunk before this kernel runs, so
// kv_len = seq_len_start + q_len includes both history and current tokens.
//
// Causal masking: Q at local position i (global position q_offset + i)
// attends to KV positions 0 .. q_offset + i (inclusive).
//
// Grid: (num_q_heads, ceil(q_len / MLA_BR), 1)  Block: (256, 1, 1)
// Thread layout within a block: 16 threads per Q row × MLA_BR Q rows.
// Each thread covers MLA_HDIM / 16 = 20 head-dim elements.

#include <cuda_bf16.h>
#include <float.h>

#define MLA_HDIM 320
#define MLA_BR   16      // query rows per block
#define MLA_LANES 16     // threads per query row (256 / MLA_BR)
#define MLA_ELEMS 20     // head-dim elements per lane (MLA_HDIM / MLA_LANES)

// Compute pointer into paged KV cache for a given logical position.
__device__ __forceinline__ const __nv_bfloat16* paged_kv_ptr_mla(
    const __nv_bfloat16* __restrict__ cache,
    const int* __restrict__ block_table,
    unsigned int pos,
    unsigned int cache_block_size,
    unsigned int num_kv_heads,
    unsigned int head_dim,
    unsigned int kv_head
) {
    unsigned int logical_block = pos / cache_block_size;
    unsigned int block_offset  = pos % cache_block_size;
    unsigned int physical_block = (unsigned int)block_table[logical_block];
    unsigned long long page_stride = (unsigned long long)cache_block_size * num_kv_heads * head_dim;
    return cache
        + (unsigned long long)physical_block * page_stride
        + (unsigned long long)block_offset   * num_kv_heads * head_dim
        + (unsigned long long)kv_head        * head_dim;
}

extern "C" __global__ void mla_prefill_paged_320(
    const __nv_bfloat16* __restrict__ Q,        // [q_len, num_q_heads, MLA_HDIM]
    const __nv_bfloat16* __restrict__ K_cache,  // paged: [num_blocks, block_size, 1, MLA_HDIM]
    const __nv_bfloat16* __restrict__ V_cache,  // paged: same layout as K_cache
    __nv_bfloat16* __restrict__ O,              // [q_len, num_q_heads, MLA_HDIM]
    const int* __restrict__ block_table,        // [max_blocks_per_seq]
    unsigned int q_len,
    unsigned int kv_len,                        // = seq_len_start + q_len
    unsigned int q_offset,                      // = seq_len_start
    unsigned int num_q_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,                      // MLA_HDIM = 320
    unsigned int cache_block_size,
    float        inv_sqrt_d
) {
    const unsigned int q_head  = blockIdx.x;
    const unsigned int q_block = blockIdx.y;
    const unsigned int tid     = threadIdx.x;

    if (q_head >= num_q_heads) return;

    const unsigned int q_start = q_block * MLA_BR;
    if (q_start >= q_len) return;
    const unsigned int q_end = min(q_start + (unsigned int)MLA_BR, q_len);

    const unsigned int gqa_ratio = num_q_heads / max(num_kv_heads, 1u);
    const unsigned int kv_head   = q_head / gqa_ratio;

    const unsigned int q_stride  = num_q_heads * head_dim;  // elements per Q token
    const unsigned int warp_lane = tid % 32;

    // Thread layout: 16 lanes per Q row, MLA_BR rows per block.
    // tid [0,255] → q_row = tid/16, lane = tid%16.
    const unsigned int q_row = tid / (unsigned int)MLA_LANES;
    const unsigned int lane  = tid % (unsigned int)MLA_LANES;

    // Half-warp mask: restrict shfl/shfl_down to the 16-thread sub-group that
    // shares the same q_row.  Using 0xFFFFFFFF when the opposite half-warp has
    // returned early (last tile, q_len % MLA_BR != 0) is CUDA UB per §B.15
    // (all threads named in the mask must be executing the same instruction).
    // warp_lane 0..15 → mask 0x0000FFFF, warp_lane 16..31 → mask 0xFFFF0000.
    const unsigned int lane_mask = (warp_lane < 16) ? 0x0000FFFFu : 0xFFFF0000u;

    if (q_row >= (q_end - q_start)) return;

    const unsigned int q_local  = q_start + q_row;
    const unsigned int q_global = q_offset + q_local;  // causal position

    const __nv_bfloat16* Q_row = Q
        + (unsigned long long)q_local * q_stride
        + (unsigned long long)q_head  * head_dim;

    float m_prev = -FLT_MAX;
    float l_prev = 0.0f;
    float acc_o[MLA_ELEMS];
    #pragma unroll
    for (int i = 0; i < MLA_ELEMS; i++) acc_o[i] = 0.0f;

    // Causal: attend to KV 0 .. q_global (inclusive).
    const unsigned int causal_kv_end = min(q_global + 1, kv_len);

    for (unsigned int kv_pos = 0; kv_pos < causal_kv_end; kv_pos++) {
        const __nv_bfloat16* K_row = paged_kv_ptr_mla(
            K_cache, block_table, kv_pos,
            cache_block_size, num_kv_heads, head_dim, kv_head);

        // Each lane handles MLA_ELEMS contiguous dims: lane*20 .. (lane+1)*20-1.
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < MLA_ELEMS; i++) {
            unsigned int d = lane * MLA_ELEMS + i;
            if (d < head_dim) {
                dot += __bfloat162float(Q_row[d]) * __bfloat162float(K_row[d]);
            }
        }

        // Reduce across 16 lanes (half a warp) using the half-warp mask.
        for (int offset = 8; offset > 0; offset >>= 1) {
            dot += __shfl_down_sync(lane_mask, dot, offset);
        }
        float score = dot * inv_sqrt_d;
        // Broadcast from lane 0 of each 16-lane group.
        score = __shfl_sync(lane_mask, score, (warp_lane / MLA_LANES) * MLA_LANES);

        float m_new = fmaxf(m_prev, score);
        float alpha = expf(m_prev - m_new);
        float p     = expf(score - m_new);
        float l_new = alpha * l_prev + p;

        const __nv_bfloat16* V_row = paged_kv_ptr_mla(
            V_cache, block_table, kv_pos,
            cache_block_size, num_kv_heads, head_dim, kv_head);

        #pragma unroll
        for (int i = 0; i < MLA_ELEMS; i++) {
            unsigned int d = lane * MLA_ELEMS + i;
            if (d < head_dim) {
                acc_o[i] = alpha * acc_o[i] + p * __bfloat162float(V_row[d]);
            }
        }
        m_prev = m_new;
        l_prev = l_new;
    }

    float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
    __nv_bfloat16* O_row = O
        + (unsigned long long)q_local * q_stride
        + (unsigned long long)q_head  * head_dim;

    #pragma unroll
    for (int i = 0; i < MLA_ELEMS; i++) {
        unsigned int d = lane * MLA_ELEMS + i;
        if (d < head_dim) {
            O_row[d] = __float2bfloat16(acc_o[i] * inv_l);
        }
    }
}
