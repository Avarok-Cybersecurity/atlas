// SPDX-License-Identifier: AGPL-3.0-only

// GLM-5.3-Flash FFN activation — clamped SwiGLU.
//
// 🔴 GLM clamps, and the clamp is ASYMMETRIC:
//     gate = clamp(gate, max=+swiglu_limit)          <- UPPER bound ONLY
//     up   = clamp(up,   min=-limit, max=+limit)     <- BOTH bounds
//     out  = silu(gate) * up
// Verbatim from HF `Glm5NextTextMLP.forward` and `Glm5NextTextExperts._apply_gate`; vLLM's
// `SiluAndMulWithClamp` implements the same asymmetry (its alpha=1.0/beta=0.0 defaults reduce
// it to exactly silu(gate)*up).
//
// Why a GLM-specific kernel rather than `moe_silu_mul`: common/'s entry point does NOT clamp —
// its own header notes that models declaring a `swiglu_limit` SHADOW that file, and that
// Qwen3.5-class models are a bare `act_fn(gate)*up`. GLM has no kernel target to shadow from,
// so the clamped form lives here, in common/, beside the other GLM kernels.
//
// 🪤 A symmetric clamp on `gate` is the easy wrong answer and is invisible on well-scaled
// activations: with |gate| < limit nothing fires at all. The gate for this kernel deliberately
// drives ~30k values above +10 AND ~30k below -10 per fixture, so a lower-bounded `gate`
// produces a different answer instead of an identical one.
//
// Separate gate/up buffers (not one interleaved [.., 2*I] tensor): GLM's checkpoint stores
// `gate_proj` and `up_proj` as separate per-expert tensors, so fusing them would mean an extra
// copy purely to satisfy a layout nothing else wants.

#include <cuda_bf16.h>

// out[i] = silu(min(gate[i], limit)) * clamp(up[i], -limit, limit), fp32 compute.
// Grid: enough blocks to cover `n`.  Block: (256,1,1).
extern "C" __global__ void glm5next_swiglu_clamp(
    const __nv_bfloat16* __restrict__ gate, // [n]
    const __nv_bfloat16* __restrict__ up,   // [n]
    __nv_bfloat16* __restrict__ out,        // [n]
    const unsigned int n,
    const float limit
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = (float)gate[i];
    float u = (float)up[i];
    g = fminf(g, limit);                    // upper bound only — NOT fmaxf(-limit, ...)
    u = fminf(fmaxf(u, -limit), limit);
    float s = g / (1.0f + expf(-g));        // silu
    out[i] = __float2bfloat16(s * u);
}

// Same, writing FP32 so an integrated FFN can keep the activation in fp32 between the
// up-projection and the down-projection when the caller wants floor-A-style accumulation.
extern "C" __global__ void glm5next_swiglu_clamp_f32out(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    float* __restrict__ out,
    const unsigned int n,
    const float limit
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = (float)gate[i];
    float u = (float)up[i];
    g = fminf(g, limit);
    u = fminf(fmaxf(u, -limit), limit);
    float s = g / (1.0f + expf(-g));
    out[i] = s * u;
}
