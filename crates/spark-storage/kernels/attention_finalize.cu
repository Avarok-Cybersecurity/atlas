// SPDX-License-Identifier: AGPL-3.0-only
//
// attention_finalize: divide the running output by the running exp-sum to
// produce the final attention output. Run once per decode step after the
// last tile.
//
// Layout:
//   l_state : [num_seqs, num_q_heads]                   (fp32)
//   o_state : [num_seqs, num_q_heads, head_dim]         (fp32)
//   output  : [num_seqs, num_q_heads, head_dim]         (bf16)
//
// Launch: grid = (num_seqs, num_q_heads, 1); block = (head_dim, 1, 1).

#include <cuda_bf16.h>

extern "C" __global__ void attention_finalize(
    const float*         __restrict__ l_state,
    const float*         __restrict__ o_state,
    __nv_bfloat16*       __restrict__ output,
    int num_q_heads,
    int head_dim
) {
    const int seq = blockIdx.x;
    const int qh  = blockIdx.y;
    const int tid = threadIdx.x;
    if (tid >= head_dim) return;

    const float l = l_state[seq * num_q_heads + qh];
    const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    const float o = o_state[((size_t)seq * num_q_heads + qh) * head_dim + tid];
    output[((size_t)seq * num_q_heads + qh) * head_dim + tid]
        = __float2bfloat16(o * inv_l);
}
