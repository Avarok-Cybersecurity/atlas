// SPDX-License-Identifier: AGPL-3.0-only

// Laguna-DFlash per-captured-state aux RMSNorm (SM121).
//
// The DFlash captured-hidden accumulator packs each row as a stack of
// `n_states` contiguous slices:
//     row = [ s0(state_dim) | s1(state_dim) | ... | s_{n_states-1}(state_dim) ]
// Each slice is one captured target-layer hidden state and must be RMS-normed
// with its OWN `aux_hidden_norms.{state}` weight BEFORE the `fc`
// concat+projection.
//
// One block owns one (row, state) slice. The grid is launched as
// `n_rows * n_states` blocks, and because the accumulator is row-contiguous
// with slice width `state_dim`, block `b`'s slice base is simply
// `io + b * state_dim`. Its weight vector is `w + (b % n_states) * state_dim`
// (the concatenated `[n_states * state_dim]` aux-norm buffer).
//
// HF-vanilla formula: out = x * w / sqrt(mean(x^2) + eps), eps = 1e-6.
// NO Qwen3-Next `(1 + w)` offset. In-place (io is both input and output).
//
// BF16 I/O, FP32 compute, 2-wide vectorized loads/stores.

#include <cuda_bf16.h>

__device__ __forceinline__ void aux_unpack_bf16x2(unsigned int packed, float& v0, float& v1) {
    v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFF)));
    v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
}

__device__ __forceinline__ unsigned int aux_pack_bf16x2(float v0, float v1) {
    unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
    unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
    return lo | (hi << 16);
}

__device__ __forceinline__ float aux_warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Grid: (n_rows * n_states, 1, 1)   Block: (256, 1, 1)
extern "C" __global__ void rms_norm_aux_stack(
    __nv_bfloat16* __restrict__ io,          // [n_rows, n_states * state_dim] in/out
    const __nv_bfloat16* __restrict__ w,     // [n_states * state_dim] concatenated weights
    unsigned int state_dim,
    unsigned int n_states
) {
    const unsigned int block = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const float eps = 1e-6f;

    // Slice base: row-contiguous accumulator, slice width = state_dim, so
    // block b maps directly to element base b * state_dim.
    __nv_bfloat16* x = io + (size_t)block * state_dim;
    // Weight for this state: cycle through the n_states concatenated vectors.
    const __nv_bfloat16* wv = w + (size_t)(block % n_states) * state_dim;

    // Step 1: sum of squares — vectorized 2-wide BF16 loads.
    float sum_sq = 0.0f;
    const unsigned int half_size = state_dim / 2;
    const unsigned int* x32 = (const unsigned int*)x;

    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        float v0, v1;
        aux_unpack_bf16x2(x32[i], v0, v1);
        sum_sq += v0 * v0 + v1 * v1;
    }
    if ((state_dim & 1) && tid == 0) {
        float val = __bfloat162float(x[state_dim - 1]);
        sum_sq += val * val;
    }

    // Step 2: block-level reduction.
    sum_sq = aux_warp_reduce_sum(sum_sq);
    __shared__ float warp_sums[32];
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;
    if (lane_id == 0) {
        warp_sums[warp_id] = sum_sq;
    }
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane_id < (blockDim.x + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        val = aux_warp_reduce_sum(val);
        if (lane_id == 0) {
            warp_sums[0] = val;
        }
    }
    __syncthreads();

    // Step 3: normalization factor.
    float rms = rsqrtf(warp_sums[0] / (float)state_dim + eps);

    // Step 4: apply vanilla weight in place — out = x * rms * w.
    const unsigned int* w32 = (const unsigned int*)wv;
    unsigned int* out32 = (unsigned int*)x;
    for (unsigned int i = tid; i < half_size; i += blockDim.x) {
        float xv0, xv1, w0, w1;
        aux_unpack_bf16x2(x32[i], xv0, xv1);
        aux_unpack_bf16x2(w32[i], w0, w1);
        out32[i] = aux_pack_bf16x2(xv0 * rms * w0, xv1 * rms * w1);
    }
    if ((state_dim & 1) && tid == 0) {
        float val = __bfloat162float(x[state_dim - 1]);
        float wl = __bfloat162float(wv[state_dim - 1]);
        x[state_dim - 1] = __float2bfloat16(val * rms * wl);
    }
}
