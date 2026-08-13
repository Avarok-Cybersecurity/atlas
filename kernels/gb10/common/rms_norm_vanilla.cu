// SPDX-License-Identifier: AGPL-3.0-only

// Vanilla RMS Normalization kernel for SM121.
//
// Standard RMSNorm formula (HF transformers / Qwen3RMSNorm convention):
//     out = x * weight / sqrt(mean(x²) + eps)
//
// Differs from `rms_norm` (in this same module) which uses Qwen3-Next's
// **offset-from-1** convention `out = x * (1 + weight) / sqrt(mean(x²) + eps)`.
// The DFlash drafter ships HF-vanilla weights and must NOT have +1 added,
// so it calls this kernel instead.
//
// Input/output: BF16, computation in FP32.
// Vectorized: 2 BF16 elements per 32-bit load/store.

#include <cuda_bf16.h>

__device__ __forceinline__ void unpack_bf16x2(unsigned int packed, float& v0, float& v1) {
    v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFF)));
    v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
}

__device__ __forceinline__ unsigned int pack_bf16x2(float v0, float v1) {
    unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
    unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
    return lo | (hi << 16);
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Grid: (num_tokens, 1, 1)
// Block: (min(hidden_size, 1024), 1, 1)
extern "C" __global__ void rms_norm_vanilla(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    unsigned int hidden_size,
    float eps
) {
    unsigned int token = blockIdx.x;
    unsigned int tid = threadIdx.x;

    const __nv_bfloat16* x = input + token * hidden_size;
    __nv_bfloat16* out = output + token * hidden_size;

    // Sum of squares — vectorized.
    float sum_sq = 0.0f;
    const unsigned int half_size = hidden_size / 2;
    const unsigned int* x32 = (const unsigned int*)x;

    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        float v0, v1;
        unpack_bf16x2(x32[i], v0, v1);
        sum_sq += v0 * v0 + v1 * v1;
    }
    if ((hidden_size & 1) && tid == 0) {
        float val = __bfloat162float(x[hidden_size - 1]);
        sum_sq += val * val;
    }

    sum_sq = warp_reduce_sum(sum_sq);
    __shared__ float warp_sums[32];
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;
    if (lane_id == 0) {
        warp_sums[warp_id] = sum_sq;
    }
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) {
            warp_sums[0] = val;
        }
    }
    __syncthreads();

    float rms = rsqrtf(warp_sums[0] / (float)hidden_size + eps);

    const unsigned int* w32 = (const unsigned int*)weight;
    unsigned int* out32 = (unsigned int*)out;

    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        float xv0, xv1, wv0, wv1;
        unpack_bf16x2(x32[i], xv0, xv1);
        unpack_bf16x2(w32[i], wv0, wv1);
        // Vanilla formula: out = x * w / RMS(x). NO `(1 + w)` offset.
        out32[i] = pack_bf16x2(xv0 * rms * wv0, xv1 * rms * wv1);
    }
    if ((hidden_size & 1) && tid == 0) {
        float val = __bfloat162float(x[hidden_size - 1]);
        float w = __bfloat162float(weight[hidden_size - 1]);
        out[hidden_size - 1] = __float2bfloat16(val * rms * w);
    }
}

// Warp-per-row vanilla RMS norm for SHORT rows (per-head q_norm / k_norm).
//
// `rms_norm_vanilla` assigns one BLOCK per row. For the Qwen3-style PER-HEAD
// norms the row is only head_dim (128) elements and the prefill call passes
// num_rows = num_heads * seq_len — at ISL 1K with 72 Q heads that is 70,560
// blocks of 128 threads, each taking a shared-memory barrier to reduce a
// 256-byte row. Measured 6.4 ms/layer against a ~0.15 ms bandwidth floor
// (Q+K = 40 MB moved): launch/occupancy bound, not bandwidth bound.
//
// One WARP owns one row here: the reduction is pure __shfl_xor_sync with no
// shared memory and no __syncthreads, and RMSV_ROWS_PER_BLOCK warps share a
// block so the grid shrinks by that factor.
//
// Same formula as the block kernel — FP32 accumulate, mean over hidden_size,
// rsqrtf(mean + eps), VANILLA `x * w` scaling with NO `(1 + w)` offset. The
// reduction ORDER differs, so results are equivalent but not bit-identical.
#define RMSV_ROWS_PER_BLOCK 8

extern "C" __global__ void rms_norm_vanilla_warp_row(
    const __nv_bfloat16* __restrict__ input,   // [num_rows, hidden_size]
    const __nv_bfloat16* __restrict__ weight,  // [hidden_size]
    __nv_bfloat16* __restrict__ output,        // [num_rows, hidden_size]
    unsigned int num_rows,
    unsigned int hidden_size,
    float eps
) {
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int row = blockIdx.x * RMSV_ROWS_PER_BLOCK + (threadIdx.x >> 5);
    if (row >= num_rows) return;

    const size_t base = (size_t)row * hidden_size;
    const unsigned int half_size = hidden_size >> 1;
    const unsigned int* x32 = (const unsigned int*)(input + base);
    const unsigned int* w32 = (const unsigned int*)weight;
    unsigned int* out32 = (unsigned int*)(output + base);

    float sum_sq = 0.0f;
    for (unsigned int i = lane; i < half_size; i += 32) {
        float v0, v1;
        unpack_bf16x2(x32[i], v0, v1);
        sum_sq += v0 * v0 + v1 * v1;
    }
    if ((hidden_size & 1) && lane == 0) {
        float val = __bfloat162float(input[base + hidden_size - 1]);
        sum_sq += val * val;
    }

    sum_sq = warp_reduce_sum(sum_sq);
    const float rms = rsqrtf(sum_sq / (float)hidden_size + eps);

    for (unsigned int i = lane; i < half_size; i += 32) {
        float xv0, xv1, wv0, wv1;
        unpack_bf16x2(x32[i], xv0, xv1);
        unpack_bf16x2(w32[i], wv0, wv1);
        out32[i] = pack_bf16x2(xv0 * rms * wv0, xv1 * rms * wv1);
    }
    if ((hidden_size & 1) && lane == 0) {
        float val = __bfloat162float(input[base + hidden_size - 1]);
        float w = __bfloat162float(weight[hidden_size - 1]);
        output[base + hidden_size - 1] = __float2bfloat16(val * rms * w);
    }
}
