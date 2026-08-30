// SPDX-License-Identifier: AGPL-3.0-only

// Boot-time expert loading (BEL): mask the router's gate logits so top-k can
// only select experts whose weights are resident.
//
// `spark serve --expert-category <name>` loads a subset of each layer's
// experts; the pointer table stays full-length with NULL entries for the
// rest, so a router that selected one would dereference null. Adding -inf to
// those logits before top-k makes them unselectable, and the top-k kernels'
// existing normalize step then renormalizes over the survivors — the
// selected experts' weights still sum to 1, so the layer's output is a
// re-weighted blend of loaded experts rather than a partial one.
//
// One generic kernel in common/ rather than an edit to each model's gate
// kernel: this must apply to EVERY routing path (softmax, sigmoid+bias,
// sqrt-softplus), and a fork that missed one would produce a null deref
// under exactly the traffic the feature exists for.
//
// `mask` is [num_experts], uploaded once at boot: 0.0f for loaded experts,
// -inf for unloaded. Additive, so a category listing every expert is
// numerically a no-op (the negative control the tests rely on).

#include <cuda_bf16.h>
#include <cuda_runtime.h>

// BF16 logits: [n, num_experts], modified in place.
//
// -inf is written as the bf16 literal rather than added, because
// bf16(-inf) + finite saturates to -inf anyway and the direct write avoids
// an inf-minus-inf NaN if a logit were already -inf.
extern "C" __global__ void moe_bel_mask_bf16(
    __nv_bfloat16* __restrict__ logits,
    const float* __restrict__ mask,
    int n,
    int num_experts
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * num_experts;
    if (i >= total) {
        return;
    }
    int e = i % num_experts;
    if (mask[e] != 0.0f) {
        logits[i] = __float2bfloat16(-INFINITY);
    }
}

// FP32 logits: the ATLAS_FP32_GATE / ATLAS_FP32_ROUTING paths write the
// router output in f32, and they feed the same top-k selection.
extern "C" __global__ void moe_bel_mask_f32(
    float* __restrict__ logits,
    const float* __restrict__ mask,
    int n,
    int num_experts
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * num_experts;
    if (i >= total) {
        return;
    }
    int e = i % num_experts;
    if (mask[e] != 0.0f) {
        logits[i] = -INFINITY;
    }
}
