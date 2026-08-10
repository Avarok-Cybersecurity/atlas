// SPDX-License-Identifier: AGPL-3.0-only

// SSM h-state storage-dtype converters (ATLAS_SSM_H_FP16).
//
// The GDN decode scan is state-traffic bound, so storing h as FP16 instead of
// FP32 halves its time. Prefill is left entirely FP32 — it writes h through 6
// different kernel families and is not traffic-bound — so the two formats meet
// at exactly two edges, and these kernels are that meeting point:
//
//   f32 -> f16   once per sequence, on its first decode step
//   f16 -> f32   when a decode-produced Marconi snapshot is written, so every
//                snapshot in the pool is FP32 and the restore path (which
//                always lands in a PREFILL) needs no dtype knowledge at all
//
// Both are out-of-place. In-place compaction is NOT safe: thread i reads
// src[4i..4i+4) and writes dst[2i..2i+2), so with src == dst thread 2i's write
// lands inside thread i's source with no ordering between them.
//
// Conversion is round-to-nearest-even in both directions (__float2half /
// __half2float), and FP16's range is never approached: the in-tree per-head
// Frobenius clamps bound every element at <= 1000 against FP16's 65504.

#include <cuda_fp16.h>

extern "C" __global__ void ssm_h_state_f32_to_f16(
    const float* __restrict__ src,
    __half* __restrict__ dst,
    unsigned long long n
) {
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        dst[i] = __float2half(src[i]);
    }
}

extern "C" __global__ void ssm_h_state_f16_to_f32(
    const __half* __restrict__ src,
    float* __restrict__ dst,
    unsigned long long n
) {
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += stride) {
        dst[i] = __half2float(src[i]);
    }
}
