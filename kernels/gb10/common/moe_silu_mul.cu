// SPDX-License-Identifier: AGPL-3.0-only

// Atlas MoE element-wise SiLU activation + multiply.
//
// output[i] = silu(gate[i]) * up[i]
// where silu(x) = x * sigmoid(x)
//
// Grid: (ceil(total_elements / 256), 1, 1)  Block: (256, 1, 1)
//
// Used after grouped gate+up GEMMs to fuse activation before down GEMM.
//
// NO SWIGLU CLAMP HERE, deliberately. One lived here from #186 until wave 55: a
// `fminf(g, 10.0f)` / `clamp(u, ±10.0f)` labelled "DeepSeek-V4 routed-expert
// swiglu clamp (swiglu_limit = 10.0, config)". The label was right about where
// the number came from and wrong about who it reached. This kernel is not the
// DeepSeek routed path — it is the SiLU activation for every dense model's
// decode and K-verify FFN, every MoE model's grouped prefill, and the MTP and
// DFlash draft heads. `swiglu_limit` is a per-checkpoint config value
// (DeepSeek-V4 10.0, GPT-OSS 7.0, every Qwen/Gemma/Nemotron/Mistral checkpoint
// on the fleet: absent), and the references for the models that do not declare
// one do not clamp: `Qwen3_5MLP.forward` is a bare `act_fn(gate) * up`.
//
// It was not dormant. Instrumented on Qwen3.6-27B over a 20-sample BFCL draw it
// bound over 100,000 times in this kernel alone, with `up` reaching -21.78 and
// `gate` reaching 17.38 — truncations of more than 2x, on a checkpoint whose
// config declares no limit.
//
// The models that DO declare a limit shadow this file:
// `deepseek-v4-flash/nvfp4/moe_silu_mul.cu` and
// `step3p7-flash/nvfp4/moe_silu_mul.cu`. That is a holding pattern, not the
// destination — Step-3.7's limits are PER LAYER, so they cannot be a compile-
// time constant and eventually want a kernel argument fed from `ModelConfig`.
// Anything added here reaches the whole fleet; add it to a shadow instead.

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
    float sigmoid_g = 1.0f / (1.0f + __expf(-g));
    float result = g * sigmoid_g * u;
    output[idx] = __float2bfloat16(result);
}
