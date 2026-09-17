// SPDX-License-Identifier: AGPL-3.0-only

// Atlas MoE element-wise SiLU activation + multiply, WITH a SwiGLU clamp.
// Shadows `common/moe_silu_mul.cu` for Step-3.7-Flash only.
//
// This file PRESERVES TODAY'S BEHAVIOUR; it does not claim to be correct.
//
// Step-3.7's `text_config` carries `swiglu_limits` and `swiglu_limits_shared`
// as PER-LAYER ARRAYS, and `config/parsers/step3p7.rs` deletes both of them —
// not as a numerics decision but because `ModelConfig` has no field to hold an
// array and serde would fail on the unknown key (the same treatment
// `partial_rotary_factors` gets two lines above). So the declared limits are
// discarded and the model has been served with whatever constant the shared
// kernel happened to hold: 10.0, which is DeepSeek-V4's value, not Step-3.7's.
//
// The clamp moved out of `common/` because it was wrong for the ~20 models that
// declare no limit at all. Step-3.7 DOES declare one, so removing the clamp
// here would be a second guess rather than a fix, and no Step-3.7 checkpoint is
// on any box to measure either guess against. The honest state is therefore:
// keep the existing constant, and record that it is a placeholder.
//
// TO FINISH THIS: read `swiglu_limits` / `swiglu_limits_shared` in
// `parsers/step3p7.rs` instead of dropping them, thread the per-layer limit
// into a kernel argument, and delete this file. Until then Step-3.7's
// activation is approximate in a way its config already tells us about.
//
// Grid: (ceil(total_elements / 256), 1, 1)  Block: (256, 1, 1)

#include <cuda_bf16.h>

extern "C" __global__ void moe_silu_mul(
    const __nv_bfloat16* __restrict__ gate,   // [total_expanded, inter_size]
    const __nv_bfloat16* __restrict__ up,     // [total_expanded, inter_size]
    __nv_bfloat16* __restrict__ output,        // [total_expanded, inter_size]
    unsigned int total_elements
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float g = __bfloat162float(gate[idx]);
    float u = __bfloat162float(up[idx]);
    // PLACEHOLDER, see above: the checkpoint declares per-layer limits that the
    // parser discards. 10.0 is what this model has always been served with.
    const float SWIGLU_LIMIT = 10.0f;
    g = fminf(g, SWIGLU_LIMIT);
    u = fminf(fmaxf(u, -SWIGLU_LIMIT), SWIGLU_LIMIT);
    float sigmoid_g = 1.0f / (1.0f + __expf(-g));
    float result = g * sigmoid_g * u;
    output[idx] = __float2bfloat16(result);
}
