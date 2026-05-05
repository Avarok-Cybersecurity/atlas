// SPDX-License-Identifier: AGPL-3.0-only
//
// kv_lowrank_project: K_lr[block, kv_head, tok] = K_block[tok, kv_head] @ P
//
// Per-token projection (NOT the block mean). Stored at write time, read by
// `predictor_score` which reduces over (q_head, tok) at scoring time. Mean-
// reduction would systematically underweight needle tokens whose softmax
// dominance comes from a single high-alignment K, so we keep all tokens.
//
// Shapes (all BF16):
//   K_block : [block_size, num_kv_heads, head_dim]   one paged block
//   P       : [head_dim,   r]                          fixed Gaussian projection
//   K_lr    : [num_kv_heads, block_size, r]           output for this block
//             (kv_head outer, then token, then r — matches scorer reuse pattern)
//
// Launch:
//   grid  = (num_kv_heads, block_size, 1)
//   block = (r,            1,          1)
//
// Each thread handles one output element (one (kv_head, tok, r-index)).

#include <cuda_bf16.h>

extern "C" __global__ void kv_lowrank_project(
    const __nv_bfloat16* __restrict__ K_block,  // [block_size, num_kv_heads, head_dim]
    const __nv_bfloat16* __restrict__ P,        // [head_dim, r]
    __nv_bfloat16*       __restrict__ K_lr,     // [num_kv_heads, block_size, r]
    int block_size,
    int num_kv_heads,
    int head_dim,
    int r
) {
    const int kv_head = blockIdx.x;
    const int tok     = blockIdx.y;
    const int out_idx = threadIdx.x;
    if (kv_head >= num_kv_heads || tok >= block_size || out_idx >= r) return;

    const __nv_bfloat16* k_row = K_block
        + (size_t)tok * (size_t)num_kv_heads * (size_t)head_dim
        + (size_t)kv_head * (size_t)head_dim;

    float acc = 0.0f;
    #pragma unroll 8
    for (int i = 0; i < head_dim; ++i) {
        float k_val = __bfloat162float(k_row[i]);
        float p_val = __bfloat162float(P[(size_t)i * r + out_idx]);
        acc = fmaf(k_val, p_val, acc);
    }
    K_lr[((size_t)kv_head * block_size + tok) * r + out_idx] = __float2bfloat16(acc);
}
