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
// input_dtype == 0: src is `const __nv_bfloat16*` -> widen each element.
// input_dtype == 1: src is `const float*`          -> straight copy.
// input_dtype == 2: src is F8_E8M0                 -> exact power-of-two widen.
//
// Grid: (ceil(total/256), 1, 1)  Block: (256, 1, 1)  — one element per thread.

#include <cuda_bf16.h>

extern "C" __global__ void widen_block_scale_f32(
    const void* __restrict__ src,    // [total] BF16 or FP32
    float* __restrict__ dst,         // [total] FP32
    unsigned int total,
    unsigned int input_dtype
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;

    if (input_dtype == 1) {
        dst[i] = ((const float*)src)[i];
    } else if (input_dtype == 2) {
        unsigned int exp = ((const unsigned char*)src)[i];
        // OCP MX E8M0 has NO zero encoding: exp==0 is the smallest scale,
        // 2^-127 (fp32 subnormal 0x00400000), and only exp==255 is NaN.
        // Mapping 0 -> 0.0f silently zeroed a legitimate block scale.
        dst[i] = (exp == 255u)
                     ? 0.0f
                     : (exp == 0u ? __uint_as_float(0x00400000u)
                                  : __uint_as_float(exp << 23));
    } else {
        unsigned short raw = ((const unsigned short*)src)[i];
        dst[i] = __bfloat162float(*(const __nv_bfloat16*)&raw);
    }
}
