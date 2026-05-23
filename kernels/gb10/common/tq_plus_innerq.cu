// SPDX-License-Identifier: AGPL-3.0-only
//
// TurboQuant+ InnerQ — device-side state definitions + host-side calibration
// controller.
//
// Header: tq_plus_innerq.cuh (declarations).
// See CITATIONS.md for prior-art chain.

#include "tq_plus_innerq.cuh"
#include <cmath>

namespace tq_plus {

// Storage for InnerQ per-channel scales. Identity (1.0) by default; finalize
// uploads the calibrated values via cudaMemcpyToSymbol.
__device__ float d_innerq_scale[INNERQ_MAX_CHANNELS] = {
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
};
__device__ float d_innerq_scale_inv[INNERQ_MAX_CHANNELS] = {
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
};
__device__ float d_innerq_sq_accum[INNERQ_MAX_CHANNELS] = {0};
__device__ int   d_innerq_count = 0;
__device__ int   d_innerq_active = 0;
__device__ int   d_innerq_calibrating = 0;

// Host-side controller. Call once at startup with target_tokens > 0 to begin
// calibration. After target_tokens K vectors have flowed through, finalize
// auto-fires (driven by host checking d_innerq_count >= target).
//
// strength: exponent on the rms-ratio scale. 0.5 = sqrt (geometric mean),
//           1.0 = full equalization. Clamped to (0, 1].
// Returns 0 on success, nonzero on invalid arg.
extern "C" int turbo_innerq_start_calibration(int target_tokens, float strength) {
    if (target_tokens <= 0) return 1;
    if (strength <= 0.0f || strength > 1.0f) return 2;

    int zero = 0, one = 1;
    float zeros[INNERQ_MAX_CHANNELS] = {0};
    cudaMemcpyToSymbol(d_innerq_sq_accum, zeros, sizeof(zeros));
    cudaMemcpyToSymbol(d_innerq_count, &zero, sizeof(int));
    cudaMemcpyToSymbol(d_innerq_active, &zero, sizeof(int));
    cudaMemcpyToSymbol(d_innerq_calibrating, &one, sizeof(int));
    return 0;
}

// Host-side finalize. Reads sq_accum + count from device, computes per-channel
// RMS, scale[i] = (mean_rms/rms[i])^strength clamped [0.5, 2.0]. Auto-disables
// if max_ratio < 1.2 (channels already balanced).
// Returns 1 if scales were activated, 0 if auto-disabled.
extern "C" int turbo_innerq_finalize(int group_size, float strength) {
    float sq_accum[INNERQ_MAX_CHANNELS];
    int count = 0;
    cudaMemcpyFromSymbol(sq_accum, d_innerq_sq_accum, group_size * sizeof(float));
    cudaMemcpyFromSymbol(&count, d_innerq_count, sizeof(int));

    if (count <= 0) {
        int zero = 0;
        cudaMemcpyToSymbol(d_innerq_calibrating, &zero, sizeof(int));
        return 0;
    }

    float rms[INNERQ_MAX_CHANNELS];
    float mean_rms = 0.0f;
    for (int i = 0; i < group_size; i++) {
        rms[i] = sqrtf(sq_accum[i] / (float)count);
        mean_rms += rms[i];
    }
    mean_rms /= (float)group_size;

    float scale[INNERQ_MAX_CHANNELS];
    float scale_inv[INNERQ_MAX_CHANNELS];
    float max_ratio = 0.0f, min_ratio = 1e30f;
    for (int i = 0; i < group_size; i++) {
        float ratio = (rms[i] > 1e-10f) ? (mean_rms / rms[i]) : 1.0f;
        float s = powf(ratio, strength);
        if (s < 0.5f) s = 0.5f;
        if (s > 2.0f) s = 2.0f;
        scale[i] = s;
        scale_inv[i] = 1.0f / s;
        if (ratio > max_ratio) max_ratio = ratio;
        if (ratio < min_ratio) min_ratio = ratio;
    }

    int zero = 0, one = 1;
    cudaMemcpyToSymbol(d_innerq_calibrating, &zero, sizeof(int));
    if (max_ratio < 1.2f && min_ratio > (1.0f / 1.2f)) {
        return 0;  // auto-disabled
    }
    cudaMemcpyToSymbol(d_innerq_scale, scale, group_size * sizeof(float));
    cudaMemcpyToSymbol(d_innerq_scale_inv, scale_inv, group_size * sizeof(float));
    cudaDeviceSynchronize();
    cudaMemcpyToSymbol(d_innerq_active, &one, sizeof(int));
    return 1;
}

}  // namespace tq_plus
