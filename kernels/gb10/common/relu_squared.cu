// SPDX-License-Identifier: AGPL-3.0-only

// Atlas ReLU² (ReLU-squared) activation.
//
// output[i] = max(0, input[i])^2
//
// Used by Nemotron-H MoE experts which use relu2 instead of SiLU+gate.
// Grid: (ceil(total_elements / 256), 1, 1)  Block: (256, 1, 1)

#include <cuda_bf16.h>

extern "C" __global__ void relu_squared(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    unsigned int total_elements
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float x = __bfloat162float(input[idx]);
    float r = fmaxf(x, 0.0f);
    output[idx] = __float2bfloat16(r * r);
}

// In-place variant: input[i] = max(0, input[i])^2
extern "C" __global__ void relu_squared_inplace(
    __nv_bfloat16* __restrict__ data,
    unsigned int total_elements
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float x = __bfloat162float(data[idx]);
    float r = fmaxf(x, 0.0f);
    data[idx] = __float2bfloat16(r * r);
}

// ── Nemotron-H utility kernels ──

// Add F32 bias to BF16 values: output[i] = bf16_to_f32(input[i]) + bias[i], cast back to BF16.
// Used for e_score_correction_bias on gate logits before topK routing.
// Grid: (ceil(N / 256), 1, 1)  Block: (256, 1, 1)
extern "C" __global__ void bias_add_bf16_f32(
    __nv_bfloat16* __restrict__ data,    // [N] BF16 (in-place)
    const float* __restrict__ bias,       // [N] F32
    unsigned int N
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float val = __bfloat162float(data[idx]) + bias[idx];
    data[idx] = __float2bfloat16(val);
}

// Nemotron-H MoE weighted sum: output = scale * sum(weights[i] * expert_down[i]) + shared_down.
// No sigmoid gate — uses routed_scaling_factor directly.
// Grid: (ceil(H / 256), 1, 1)  Block: (256, 1, 1)
extern "C" __global__ void moe_weighted_sum_scale(
    __nv_bfloat16* __restrict__ output,            // [H] BF16
    const __nv_bfloat16* __restrict__ expert_down,  // [top_k, H] BF16
    const float* __restrict__ weights,               // [top_k] F32
    const __nv_bfloat16* __restrict__ shared_down,  // [H] BF16
    unsigned int H,
    unsigned int top_k,
    float routed_scaling_factor
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= H) return;

    float routed_sum = 0.0f;
    for (unsigned int k = 0; k < top_k; k++) {
        routed_sum += weights[k] * __bfloat162float(expert_down[k * H + idx]);
    }
    float shared_val = __bfloat162float(shared_down[idx]);
    output[idx] = __float2bfloat16(routed_scaling_factor * routed_sum + shared_val);
}

// Convert F32 tensor to BF16 in-place (overwrites lower half of buffer).
// Used to convert F32 gate weights to BF16 at load time.
// Grid: (ceil(N / 256), 1, 1)  Block: (256, 1, 1)
extern "C" __global__ void convert_f32_to_bf16(
    const float* __restrict__ input,      // [N] F32
    __nv_bfloat16* __restrict__ output,   // [N] BF16
    unsigned int N
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    output[idx] = __float2bfloat16(input[idx]);
}
