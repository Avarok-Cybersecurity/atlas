// SPDX-License-Identifier: AGPL-3.0-only

// Widen an FP8 block-scale tensor to FP32 on the GPU.
//
// `src` is `[total]` BF16 (`in_is_fp32 == 0`) or FP32 (`in_is_fp32 == 1`);
// `dst` is `[total]` FP32. Lossless BF16->FP32 widen / straight copy. Run once
// at load so downstream FP8 block-scale kernels read `const float*`.
// Mirrors crates/spark-model/src/layers/ops/gemm_quant.rs::widen_block_scale_f32.
//
// Grid:  (ceil(total/256), 1, 1)
// Block: (256, 1, 1)

#include <cuda_bf16.h>

extern "C" __global__ void widen_block_scale_f32(
    const void* __restrict__ src,
    float* __restrict__ dst,
    unsigned int total,
    unsigned int in_is_fp32)
{
    unsigned int i = blockIdx.x * 256u + threadIdx.x;
    if (i >= total) return;
    if (in_is_fp32) {
        dst[i] = static_cast<const float*>(src)[i];
    } else {
        dst[i] = __bfloat162float(static_cast<const __nv_bfloat16*>(src)[i]);
    }
}
