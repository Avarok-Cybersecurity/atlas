// SPDX-License-Identifier: AGPL-3.0-only
//
// q_lowrank_project: Q @ P  for the Atlas high-speed-swap predictor.
//
// Decode-time projection of the current step's queries through the fixed
// Gaussian projection matrix P. Run once per layer per decode step.
//
// Shapes (all BF16):
//   Q     : [num_q_heads, head_dim]
//   P     : [head_dim,    r]
//   Q_proj: [num_q_heads, r]
//
// Launch:
//   grid  = (num_q_heads, 1, 1)
//   block = (r,           1, 1)
//
// Each thread handles one output element (one (q_head, r-index) pair) and
// reads the full head_dim dimension.

#include <cuda_bf16.h>

extern "C" __global__ void q_lowrank_project(
    const __nv_bfloat16* __restrict__ Q,       // [num_q_heads, head_dim]
    const __nv_bfloat16* __restrict__ P,       // [head_dim, r]
    __nv_bfloat16*       __restrict__ Q_proj,  // [num_q_heads, r]
    int num_q_heads,
    int head_dim,
    int r
) {
    const int q_head  = blockIdx.x;
    const int out_idx = threadIdx.x;
    if (q_head >= num_q_heads || out_idx >= r) return;

    const __nv_bfloat16* q_row = Q + (size_t)q_head * head_dim;
    float acc = 0.0f;
    #pragma unroll 8
    for (int i = 0; i < head_dim; ++i) {
        float q_val = __bfloat162float(q_row[i]);
        float p_val = __bfloat162float(P[(size_t)i * r + out_idx]);
        acc = fmaf(q_val, p_val, acc);
    }
    Q_proj[(size_t)q_head * r + out_idx] = __float2bfloat16(acc);
}
