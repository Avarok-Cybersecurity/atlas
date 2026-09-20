// SPDX-License-Identifier: AGPL-3.0-only

// KV Cache Append — write new K, V tokens into the flat cache.
//
// Appends num_new_tokens of K and V to the cache at position seq_len.
//
// K_cache layout: [batch, max_seq_len, num_kv_heads, head_dim] BF16
// V_cache layout: [batch, max_seq_len, num_kv_heads, head_dim] BF16
//
// K_new:  [batch, num_new_tokens, num_kv_heads, head_dim] BF16
// V_new:  [batch, num_new_tokens, num_kv_heads, head_dim] BF16
//
// Grid: (num_new_tokens, num_kv_heads, batch_size)
// Block: (256, 1, 1)
//
// Each block copies one (token, kv_head) pair's head_dim elements.

#include <cuda_bf16.h>

extern "C" __global__ void kv_cache_append(
    __nv_bfloat16* __restrict__ K_cache,     // [batch, max_seq_len, num_kv_heads, head_dim]
    __nv_bfloat16* __restrict__ V_cache,
    const __nv_bfloat16* __restrict__ K_new, // [batch, num_new_tokens, num_kv_heads, head_dim]
    const __nv_bfloat16* __restrict__ V_new,
    const unsigned int seq_len,               // Current cache position (write starts here)
    const unsigned int max_seq_len,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int num_new_tokens
) {
    const unsigned int token_idx = blockIdx.x;   // Which new token [0, num_new_tokens)
    const unsigned int kv_head = blockIdx.y;      // Which KV head
    const unsigned int batch = blockIdx.z;

    if (token_idx >= num_new_tokens || kv_head >= num_kv_heads) return;

    const unsigned int cache_pos = seq_len + token_idx;
    if (cache_pos >= max_seq_len) return;

    // Source: K_new[batch, token_idx, kv_head, :]
    const unsigned int src_stride = num_kv_heads * head_dim;
    const unsigned int src_offset = batch * num_new_tokens * src_stride
                                  + token_idx * src_stride
                                  + kv_head * head_dim;

    // Destination: K_cache[batch, cache_pos, kv_head, :]
    const unsigned int dst_stride = num_kv_heads * head_dim;
    const unsigned int dst_offset = batch * max_seq_len * dst_stride
                                  + cache_pos * dst_stride
                                  + kv_head * head_dim;

    // Copy head_dim elements with thread parallelism
    for (unsigned int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        K_cache[dst_offset + i] = K_new[src_offset + i];
        V_cache[dst_offset + i] = V_new[src_offset + i];
    }
}
