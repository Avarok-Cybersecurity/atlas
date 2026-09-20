// SPDX-License-Identifier: AGPL-3.0-only
// K3 dense/shared MLP: original resident BF16 or FP32 weights, FP32 activations.
// Four warps per block, one independent output per warp. No padded reads.
#include <cuda_bf16.h>
#include <stdint.h>

__device__ __forceinline__ float k3_load_weight(const void* w, uint32_t i, uint32_t dtype) {
    return dtype == 0 ? static_cast<const float*>(w)[i]
                      : __bfloat162float(static_cast<const __nv_bfloat16*>(w)[i]);
}

extern "C" __global__ void k3_dense_gate_up_situ_f32io(
    const float* x, const void* gate, const void* up, float* mid,
    uint32_t n, uint32_t k, uint32_t dtype, float beta, float linear_beta) {
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t row = blockIdx.x * 4 + threadIdx.x / 32;
    if (row >= n) return;
    float g = 0.f, u = 0.f;
    for (uint32_t i = lane; i < k; i += 32) {
        const float a = x[i];
        g += a * k3_load_weight(gate, row * k + i, dtype);
        u += a * k3_load_weight(up, row * k + i, dtype);
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        g += __shfl_down_sync(0xffffffff, g, offset);
        u += __shfl_down_sync(0xffffffff, u, offset);
    }
    if (lane == 0) {
        const float activated = beta * tanhf(g / beta) / (1.f + expf(-g));
        mid[row] = activated * (linear_beta * tanhf(u / linear_beta));
    }
}

extern "C" __global__ void k3_dense_down_f32io(
    const float* x, const void* w, float* y,
    uint32_t n, uint32_t k, uint32_t dtype) {
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t row = blockIdx.x * 4 + threadIdx.x / 32;
    if (row >= n) return;
    float sum = 0.f;
    for (uint32_t i = lane; i < k; i += 32)
        sum += x[i] * k3_load_weight(w, row * k + i, dtype);
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (lane == 0) y[row] = sum;
}
