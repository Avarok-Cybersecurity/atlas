// SPDX-License-Identifier: AGPL-3.0-only

// BF16 in-place vector addition: dst[i] += src[i]
// Used by 2-rank send/recv all-reduce in NcclBackend.

#include <cuda_bf16.h>

extern "C" __global__ void bf16_add_inplace(
    __nv_bfloat16* __restrict__ dst,
    const __nv_bfloat16* __restrict__ src,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __hadd(dst[i], src[i]);
    }
}
