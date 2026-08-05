// SPDX-License-Identifier: AGPL-3.0-only

// Embedding scale for Gemma-4: output[i] *= scale
//
// Gemma models scale embeddings by sqrt(hidden_size) after lookup.
// In-place BF16 operation. Applied after embedding table copy.
//
// Grid: (ceil(N/256), 1, 1)  Block: (256, 1, 1)

#include <cuda_bf16.h>

extern "C" __global__ void bf16_scale_inplace(
    __nv_bfloat16* __restrict__ data,
    unsigned int N,
    float scale
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float x = __bfloat162float(data[idx]);
    data[idx] = __float2bfloat16(x * scale);
}

// FP32 in-place scale for Gemma-4 FP32-residual path.
// Used by `apply_layer_scalar` when hidden is kept in FP32 across layers
// to prevent BF16 underflow from cumulative tiny layer_scalar values
// (L0=0.089, L1=0.065 on 31B compound to ~0.006 in 2 layers).
extern "C" __global__ void f32_scale_inplace(
    float* __restrict__ data,
    unsigned int N,
    float scale
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    data[idx] = data[idx] * scale;
}
