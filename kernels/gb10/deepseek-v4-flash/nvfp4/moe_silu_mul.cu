// SPDX-License-Identifier: AGPL-3.0-only

// Atlas MoE element-wise SiLU activation + multiply, WITH DeepSeek-V4's
// configured SwiGLU clamp. Shadows `common/moe_silu_mul.cu` for this model
// only.
//
//   output[i] = silu(clamp(gate[i])) * clamp(up[i])
//
// WHY THIS FILE EXISTS. The clamp used to live in `common/`, added there by the
// DeepSeek-V4 port (#186) as six lines on a kernel that was already shared. But
// `moe_silu_mul` is not a DeepSeek kernel: it is the SiLU activation for every
// dense model's decode and K-verify FFN (`DenseFfnLayer::act_mul`), for every
// MoE model's grouped prefill (`MoeLayer::moe_act_mul`), and for the MTP and
// DFlash draft heads. So one checkpoint's config value was applied to about
// twenty checkpoints, none of which declare a SwiGLU limit and none of whose
// reference implementations clamp at all (`Qwen3_5MLP.forward` and
// `Qwen3_5MoeExperts.forward` are a bare `act_fn(gate) * up`).
//
// `swiglu_limit` is genuinely model-specific — DeepSeek-V4 sets 10.0, GPT-OSS
// sets 7.0, Qwen sets nothing — so the value belongs with the model. Keeping it
// in a shadow is the smallest correct home for it until the limit is threaded
// from config into a kernel argument, which is what the checkpoint actually
// asks for and what Step-3.7's per-LAYER `swiglu_limits` array will require.
//
// The reference is `inference/model.py` in `deepseek-ai/DeepSeek-V4-Flash`:
//
//     if self.swiglu_limit > 0:
//         up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
//         gate = torch.clamp(gate, max=self.swiglu_limit)
//     x = F.silu(gate) * up
//
// Note the asymmetry — gate is bounded ABOVE only, up is bounded on both sides.
// The math below is byte-for-byte what `common/` computed before the move, so
// DeepSeek-V4's numerics are unchanged by relocating it.
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
    // swiglu_limit = 10.0, from the checkpoint's config.json. Hardcoded because
    // no Rust parser reads the field yet; `ModelConfig` has no home for it.
    const float SWIGLU_LIMIT = 10.0f;
    g = fminf(g, SWIGLU_LIMIT);
    u = fminf(fmaxf(u, -SWIGLU_LIMIT), SWIGLU_LIMIT);
    float sigmoid_g = 1.0f / (1.0f + __expf(-g));
    float result = g * sigmoid_g * u;
    output[idx] = __float2bfloat16(result);
}
