// SPDX-License-Identifier: AGPL-3.0-only

// Gemma-4 E2B per-layer-embedding (PLE) slice multiply (SM121).
//
// The model-level PLE precompute builds the combined per-layer vectors as a
// single [num_tokens, num_layers * per_layer_dim] BF16 buffer (row-major:
// layer `i`'s 256-dim vector for token `t` lives at columns
// `[t*row_stride + i*256, t*row_stride + (i+1)*256)`). A decoder layer's PLE
// block needs its own 256-dim slice as a CONTIGUOUS [num_tokens, 256] vector
// to elementwise-multiply into `h`, but the slice is strided by `row_stride`
// between tokens — so this kernel reads the strided source directly instead
// of staging a transposed copy.
//
//   h[t*ple_dim + d] *= ple[t*row_stride + layer_col + d]
//
// Grid: (num_tokens, 1, 1)  Block: (ple_dim, 1, 1)

#include <cuda_bf16.h>

extern "C" __global__ void gemma4_ple_mul(
    __nv_bfloat16* __restrict__ h,              // [num_tokens, ple_dim] contiguous, in/out
    const __nv_bfloat16* __restrict__ ple,      // [num_tokens, row_stride] combined PLE buffer
    const unsigned int layer_col,               // column offset of this layer's slice (= i*ple_dim)
    const unsigned int row_stride,              // elements per token row in `ple` (= 35*ple_dim)
    const unsigned int num_tokens,
    const unsigned int ple_dim
) {
    const unsigned int t = blockIdx.x;
    const unsigned int d = threadIdx.x;
    if (t >= num_tokens) return;
    __nv_bfloat16* h_row = h + t * ple_dim;
    const __nv_bfloat16* ple_row = ple + t * row_stride;
    const float h_val = __bfloat162float(h_row[d]);
    const float ple_val = __bfloat162float(ple_row[layer_col + d]);
    h_row[d] = __float2bfloat16(h_val * ple_val);
}
