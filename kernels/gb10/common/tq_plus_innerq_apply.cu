// SPDX-License-Identifier: AGPL-3.0-only
//
// TurboQuant+ InnerQ application kernels — per-channel Q/K equalization.
//
// Two tiny element-wise multiply kernels that fire BEFORE WHT(Q) and AFTER
// WHT(K) when InnerQ is active (d_innerq_active == 1). When inactive
// (d_innerq_active == 0), the kernels return immediately — zero cost.
//
// Math: <Q/s, s·K> = <Q, K>. Multiplying Q by 1/s pre-WHT and K by s
// post-WHT cancels at the inner product, restoring the original
// attention scores while shifting the variance burden from K into the
// codebook-friendly post-WHT distribution.

#include <cuda_bf16.h>
#include "tq_plus_innerq.cuh"

namespace tq_plus {

// InnerQ device state — defined HERE, in the same TU as the only kernels that
// read/write it. Atlas loads each .cu as its own PTX module (no -rdc device
// linking), so state and its consumers must share one module or the host
// driver (innerq_driver.rs, module "tq_plus_innerq_apply") uploads into a copy
// the kernels never see. Scales default to identity (1.0); finalize uploads
// calibrated values via cuModuleGetGlobal_v2 + cuMemcpyHtoDAsync.
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

}  // namespace tq_plus

// Apply scale_inv to Q (one bf16 vector per warp, hd=128 expected).
// Grid: (num_q_heads, 1, 1) for matching wht_bf16_inplace dispatch shape.
// Block: (32, 1, 1) — one warp per head.
extern "C" __global__ void tq_plus_innerq_apply_q(
    __nv_bfloat16* __restrict__ data,
    const unsigned int head_dim
) {
    if (tq_plus::d_innerq_active == 0) return;
    if (head_dim != 128) return;  // only hd=128 supported for now

    const unsigned int head = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (lane >= 32) return;

    __nv_bfloat16* head_data = data + (unsigned long long)head * head_dim;

    // 32 threads × 4 elements = 128.
    #pragma unroll
    for (unsigned int i = 0; i < 4; i++) {
        unsigned int ch = lane * 4 + i;
        float v = __bfloat162float(head_data[ch]);
        v *= tq_plus::d_innerq_scale_inv[ch];
        head_data[ch] = __float2bfloat16(v);
    }
}

// Apply scale to K (post-WHT bf16 vector per warp). Same dispatch shape as
// the Q variant. Also opportunistically accumulates K² stats into
// d_innerq_sq_accum during calibration phase (d_innerq_calibrating == 1).
extern "C" __global__ void tq_plus_innerq_apply_k(
    __nv_bfloat16* __restrict__ data,
    const unsigned int head_dim
) {
    if (head_dim != 128) return;

    const unsigned int head = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (lane >= 32) return;

    __nv_bfloat16* head_data = data + (unsigned long long)head * head_dim;

    // Calibration: accumulate K² before scaling. Only head 0 accumulates to
    // avoid double-counting per-head K (each token has multiple kv_heads but
    // the calibration is per-channel across all of them).
    if (tq_plus::d_innerq_calibrating == 1 && head == 0) {
        #pragma unroll
        for (unsigned int i = 0; i < 4; i++) {
            unsigned int ch = lane * 4 + i;
            float v = __bfloat162float(head_data[ch]);
            atomicAdd(&tq_plus::d_innerq_sq_accum[ch], v * v);
        }
        if (lane == 0) {
            atomicAdd(&tq_plus::d_innerq_count, 1);
        }
    }

    if (tq_plus::d_innerq_active == 0) return;

    // Apply scale (post-WHT).
    #pragma unroll
    for (unsigned int i = 0; i < 4; i++) {
        unsigned int ch = lane * 4 + i;
        float v = __bfloat162float(head_data[ch]);
        v *= tq_plus::d_innerq_scale[ch];
        head_data[ch] = __float2bfloat16(v);
    }
}
