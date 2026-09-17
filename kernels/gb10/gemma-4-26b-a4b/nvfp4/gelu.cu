// SPDX-License-Identifier: AGPL-3.0-only

// Atlas GELU activation kernels for Gemma-4 (SM121).
//
// Gemma-4 uses GELU with tanh approximation (PyTorch convention) instead
// of SiLU used by Qwen/Llama models. The FFN computes:
//   output = down_proj(gelu(gate_proj(x)) * up_proj(x))
//
// Provides:
//   gelu_tanh     — standalone GELU activation
//   gelu_mul      — fused gelu(gate) * up for the gated FFN
//
// Input/output: BF16, computation in FP32.

#include <cuda_bf16.h>

// GELU with tanh approximation (PyTorch convention):
//   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// Grid: (ceil(N/256), 1, 1)  Block: (256, 1, 1)
extern "C" __global__ void gelu_tanh(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    unsigned int N
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float x = __bfloat162float(input[idx]);
    // sqrt(2/pi) = 0.7978845608...
    float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
    float gelu = 0.5f * x * (1.0f + tanhf(inner));
    output[idx] = __float2bfloat16(gelu);
}

// Fused GELU + multiply for gated FFN:
//   output[i] = gelu(gate[i]) * up[i]
//
// This matches Gemma-4's FFN: down_proj(gelu(gate_proj(x)) * up_proj(x)).
// Gate is the gate_proj output, up is the up_proj output.
//
// Grid: (ceil(N/256), 1, 1)  Block: (256, 1, 1)
extern "C" __global__ void gelu_mul(
    const __nv_bfloat16* __restrict__ gate,   // gate_proj output
    const __nv_bfloat16* __restrict__ up,     // up_proj output
    __nv_bfloat16* __restrict__ output,        // gelu(gate) * up
    unsigned int N
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float g = __bfloat162float(gate[idx]);
    float u = __bfloat162float(up[idx]);

    // gelu(g) with tanh approximation
    float inner = 0.7978845608f * (g + 0.044715f * g * g * g);
    float gelu_g = 0.5f * g * (1.0f + tanhf(inner));

    output[idx] = __float2bfloat16(gelu_g * u);
}
