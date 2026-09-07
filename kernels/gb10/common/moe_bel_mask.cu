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

// ── Resident-mass rescale ─────────────────────────────────────────────────
//
// Masking before top-k and renormalizing over the survivors gives every
// selected expert w' = w / rho, where w is its weight in the FULL 256-way
// softmax and rho is the residents' share of that softmax mass. So the mass
// that belonged to absent experts does not vanish — it is handed to whichever
// resident experts were selected, including ones the router ranked far below
// the true top-k. A confident wrong expert at full strength, which is what a
// corrupted span in otherwise on-task text looks like.
//
// Multiplying the selected weights by rho undoes exactly that: w' * rho = w,
// so every selected expert carries its TRUE weight and the routed branch
// contributes rho of its usual total instead of all of it. The remainder is
// the mass that genuinely belonged to experts this serve does not hold, and
// it is left for a compensation term rather than silently reassigned.
//
// Softmax-routed models only. Sigmoid routing does not normalize across
// experts, so there is no shared denominator to take a ratio of, and the
// caller must not invoke this for those.

// One block per row. Computes rho = (sum of exp over RESIDENT experts) /
// (sum of exp over all experts), max-subtracted for stability, and writes it
// to `rho[row]`. Reads the logits BEFORE masking.
extern "C" __global__ void moe_bel_resident_mass_bf16(
    const __nv_bfloat16* __restrict__ logits,   // [n, num_experts], unmasked
    const float* __restrict__ mask,             // [num_experts] 0 = resident
    float* __restrict__ rho,                    // [n] out
    int num_experts
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const __nv_bfloat16* l = logits + (size_t)row * num_experts;

    __shared__ float s_max[256];
    __shared__ float s_all[256];
    __shared__ float s_res[256];

    float m = -INFINITY;
    for (int e = tid; e < num_experts; e += blockDim.x) {
        m = fmaxf(m, __bfloat162float(l[e]));
    }
    s_max[tid] = m;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        __syncthreads();
    }
    const float row_max = s_max[0];
    __syncthreads();

    float all = 0.0f, res = 0.0f;
    for (int e = tid; e < num_experts; e += blockDim.x) {
        const float v = __expf(__bfloat162float(l[e]) - row_max);
        all += v;
        // mask[e] == 0 marks a resident expert; anything else is -inf.
        if (mask[e] == 0.0f) res += v;
    }
    s_all[tid] = all;
    s_res[tid] = res;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_all[tid] += s_all[tid + s];
            s_res[tid] += s_res[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        // all >= res > 0 whenever at least one expert is resident, which the
        // boot-time top-k floor check guarantees.
        rho[row] = (s_all[0] > 0.0f) ? (s_res[0] / s_all[0]) : 1.0f;
    }
}

// Scale each row's selected top-k weights by that row's rho.
// `row_base` is the caller's first row in the pass. The per-token decode
// path reuses a single-row weights buffer and walks the batch itself, so it
// passes its loop index here; whole-pass callers pass 0. Reading rho[0] for
// every token would scale every row by the FIRST token's resident share.
extern "C" __global__ void moe_bel_scale_weights(
    float* __restrict__ expert_weights,   // [n, top_k], modified in place
    const float* __restrict__ rho,        // [row_base + n]
    int row_base,
    int n,
    int top_k
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * top_k) return;
    expert_weights[i] *= rho[row_base + i / top_k];
}
