// SPDX-License-Identifier: AGPL-3.0-only

// Widen an FP8 block-scale tensor (`weight_scale_inv`) to FP32 on the GPU.
//
// FP8 block-scaled checkpoints (Qwen3.x / DeepSeek-V3 store the scale BF16;
// MiniMax-M2 stores it FP32) carry a per-128x128-block scale. Atlas applies
// this scale in the FP32 epilogue of its W8A8 / W8A16 GEMM kernels — to match
// vLLM / DeepGEMM / HF block-FP8 numerics the scale must be held in FP32 end
// to end, not BF16. This kernel materialises a genuine FP32 device buffer once
// at load time (lossless BF16->FP32 widen, or a straight FP32 copy) so every
// downstream FP8 block-scale kernel can read `const float*` unconditionally.
//
// in_is_fp32 == 0:  src is `const __nv_bfloat16*`  -> widen each element.
// in_is_fp32 != 0:  src is `const float*`          -> straight copy.
//
// Grid: (ceil(total/256), 1, 1)  Block: (256, 1, 1)  — one element per thread.

#include <cuda_bf16.h>

extern "C" __global__ void widen_block_scale_f32(
    const void* __restrict__ src,    // [total] BF16 or FP32
    float* __restrict__ dst,         // [total] FP32
    unsigned int total,
    unsigned int in_is_fp32
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;

    if (in_is_fp32) {
        dst[i] = ((const float*)src)[i];
    } else {
        unsigned short raw = ((const unsigned short*)src)[i];
        dst[i] = __bfloat162float(*(const __nv_bfloat16*)&raw);
    }
}
