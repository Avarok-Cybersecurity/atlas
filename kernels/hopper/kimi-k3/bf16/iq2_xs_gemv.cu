// SPDX-License-Identifier: AGPL-3.0-only
// Fused IQ2_XS / IQ3_XXS GEMV. Packed expert stacks stay packed.
// One output row per warp. Activations FP32. Hopper kimi-k3 owned (no common/).
#include <cuda_fp16.h>
#include <stdint.h>
#include "iq2_xs_tables.cuh"

__device__ __forceinline__ float iq2_xs_dscale(const uint8_t* blk) {
    const uint16_t bits = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
    return __half2float(__ushort_as_half(bits));
}

__device__ __forceinline__ float dequant_iq2_at(const uint8_t* blk, uint32_t t) {
    const float d = iq2_xs_dscale(blk);
    const uint8_t* qs = blk + 2;
    const uint8_t* scales = blk + 66;
    const uint32_t ib32 = t / 32;
    const uint32_t t32 = t % 32;
    const uint32_t l = t32 / 8;
    const uint32_t j = t32 % 8;
    const uint8_t sb = scales[ib32];
    const float db = d * (0.5f + (float)((l < 2) ? (sb & 0xf) : (sb >> 4))) * 0.25f;
    const uint32_t byte_pos = 8 * ib32 + 2 * l;
    const uint16_t qs_val = (uint16_t)qs[byte_pos] | ((uint16_t)qs[byte_pos + 1] << 8);
    const uint64_t mag_bits = IQ2XS_GRID[qs_val & 511];
    const uint8_t mag = (uint8_t)(mag_bits >> (8 * j));
    const uint8_t sign_byte = KSIGNS_IQ2XS[qs_val >> 9];
    const float sign = (sign_byte & KMASK_IQ2XS[j]) ? -1.f : 1.f;
    return db * (float)mag * sign;
}

__device__ __forceinline__ float dequant_iq3_at(const uint8_t* blk, uint32_t t) {
    const float d = iq2_xs_dscale(blk);
    const uint8_t* qs = blk + 2;
    const uint8_t* scales = blk + 66;
    const uint32_t ib32 = t / 32;
    const uint32_t t32 = t % 32;
    const uint32_t l = t32 / 8;
    const uint32_t j = t32 % 8;
    const uint32_t aux = (uint32_t)scales[4 * ib32]
        | ((uint32_t)scales[4 * ib32 + 1] << 8)
        | ((uint32_t)scales[4 * ib32 + 2] << 16)
        | ((uint32_t)scales[4 * ib32 + 3] << 24);
    const float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
    const uint8_t signs = KSIGNS_IQ2XS[(aux >> (7 * l)) & 127];
    const uint8_t gidx = qs[8 * ib32 + 2 * l + (j >= 4 ? 1 : 0)];
    const uint32_t grid = IQ3XXS_GRID[gidx];
    const uint8_t mag = (uint8_t)(grid >> (8 * (j & 3)));
    const float sign = (signs & KMASK_IQ2XS[j]) ? -1.f : 1.f;
    return db * (float)mag * sign;
}

// y[n] = packed W[n, k_full] @ x[k_local] using covering IQ blocks.
// packed row is n_cover * block_bytes. skip = k0 - b0*256.
extern "C" __global__ void k3_iq2_xs_gemv_f32io(
    const float* x, const uint8_t* w, float* y,
    uint32_t n, uint32_t k_local, uint32_t skip, uint32_t n_cover, uint32_t block_bytes) {
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t row = blockIdx.x * 4u + threadIdx.x / 32;
    if (row >= n) return;
    const uint8_t* rowp = w + (size_t)row * (size_t)n_cover * (size_t)block_bytes;
    float sum = 0.f;
    for (uint32_t t = lane; t < k_local; t += 32) {
        const uint32_t src = t + skip;
        const uint32_t b = src / 256;
        const uint32_t off = src % 256;
        sum += x[t] * dequant_iq2_at(rowp + b * block_bytes, off);
    }
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (lane == 0) y[row] = sum;
}

extern "C" __global__ void k3_iq3_xxs_gemv_f32io(
    const float* x, const uint8_t* w, float* y,
    uint32_t n, uint32_t k_local, uint32_t skip, uint32_t n_cover, uint32_t block_bytes) {
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t row = blockIdx.x * 4u + threadIdx.x / 32;
    if (row >= n) return;
    const uint8_t* rowp = w + (size_t)row * (size_t)n_cover * (size_t)block_bytes;
    float sum = 0.f;
    for (uint32_t t = lane; t < k_local; t += 32) {
        const uint32_t src = t + skip;
        const uint32_t b = src / 256;
        const uint32_t off = src % 256;
        sum += x[t] * dequant_iq3_at(rowp + b * block_bytes, off);
    }
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (lane == 0) y[row] = sum;
}
