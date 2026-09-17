// SPDX-License-Identifier: AGPL-3.0-only
//
// TurboQuant+ InnerQ — per-channel Q/K equalization.
//
// See CITATIONS.md for prior-art chain.
//
// Equalizes per-channel variance of K vectors before WHT rotation. Reduces
// the dynamic range each Lloyd-Max bin must cover by smoothing high-variance
// channels into the available codebook range. Preserves attention dot
// products via the identity <Q/s, s·K> = <Q, K>: Q gets multiplied by
// scale_inv = 1/s, K gets multiplied by s. The two scales cancel exactly
// at the inner product.
//
// Two-phase operation:
//   1. Calibration (first innerq_target_tokens K tensors): accumulate per-
//      channel sum-of-squares in d_innerq_sq_accum. d_innerq_calibrating = 1.
//   2. Finalize (host calls turbo_innerq_finalize once count >= target):
//      compute per-channel RMS, scale[i] = (mean_rms/rms[i])^strength clamped
//      to [0.5, 2.0]. Auto-disable if max ratio < 1.2 (already balanced).
//      Upload to d_innerq_scale / d_innerq_scale_inv, set d_innerq_active = 1.
//   3. Active (post-finalize, all subsequent tokens): Q multiplied by
//      d_innerq_scale_inv before WHT, K multiplied by d_innerq_scale before
//      quantization. Identity preservation holds at dot-product level.

#pragma once
#include <cuda_bf16.h>

namespace tq_plus {

// Max per-head channel count. Qwen3.6-A3B uses head_dim=128, so 128 is enough.
// Bump if porting to head_dim > 128 models without splitting the array.
#define INNERQ_MAX_CHANNELS 128

// Device state — driven by the host-side calibration controller
// (innerq_driver.rs). Defined in tq_plus_innerq_apply.cu, the SAME TU as the
// only kernels that read/write it: Atlas loads each .cu as its own PTX module
// with no -rdc device linking, so `extern __device__` never resolves across
// modules — state and its consumers must share one TU. If a new TU includes
// this header without carrying the definitions, nvcc #20044-D ("extern
// declaration treated as static definition") fires under the strict kernel
// build: that error means the new TU would get a private zero-init copy the
// host driver never uploads to. Don't suppress it — move the code needing the
// state into tq_plus_innerq_apply.cu (or give the new module its own uploads).
extern __device__ float d_innerq_scale[INNERQ_MAX_CHANNELS];
extern __device__ float d_innerq_scale_inv[INNERQ_MAX_CHANNELS];
extern __device__ float d_innerq_sq_accum[INNERQ_MAX_CHANNELS];
extern __device__ int   d_innerq_count;
extern __device__ int   d_innerq_active;       // 0 = identity scales, 1 = post-finalize
extern __device__ int   d_innerq_calibrating;  // 1 = accumulating K² stats

// Apply per-channel scale_inv to Q-side values. No-op when d_innerq_active=0.
// Caller: each thread holds 4 floats for hd=128 (lane*4 + i indexing).
__device__ __forceinline__ void apply_innerq_scale_inv_128(float vals[4], unsigned int lane) {
    if (d_innerq_active == 0) return;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        unsigned int ch = lane * 4 + i;
        vals[i] *= d_innerq_scale_inv[ch];
    }
}

// Apply per-channel scale to K-side values. No-op when d_innerq_active=0.
__device__ __forceinline__ void apply_innerq_scale_128(float vals[4], unsigned int lane) {
    if (d_innerq_active == 0) return;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        unsigned int ch = lane * 4 + i;
        vals[i] *= d_innerq_scale[ch];
    }
}

// Accumulate K² into d_innerq_sq_accum during calibration. Each thread adds
// its 4 values squared into the corresponding channel slot. Atomic adds
// because every token's 32 lanes write to the same 128 channel slots in
// parallel across many warps.
__device__ __forceinline__ void accumulate_innerq_calibration_128(float vals[4], unsigned int lane) {
    if (d_innerq_calibrating == 0) return;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        unsigned int ch = lane * 4 + i;
        atomicAdd(&d_innerq_sq_accum[ch], vals[i] * vals[i]);
    }
    if (lane == 0 && threadIdx.y == 0) {
        atomicAdd(&d_innerq_count, 1);
    }
}

}  // namespace tq_plus
