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

// ── glm5next_router_topk ──
// GLM-5.3 MoE router selection, from `Glm5NextTextTopkRouter.forward`:
//   scores        = sigmoid(router_logits)
//   scores_choice = scores + e_score_correction_bias      <- SELECTION only
//   topk_ids      = topk(scores_choice, K)
//   topk_weights  = scores.gather(topk_ids)               <- the UNBIASED scores
//   if renormalize: topk_weights /= (sum + 1e-20)
//   topk_weights *= routed_scaling_factor
//
// 🪤 The correction bias steers SELECTION and NOTHING ELSE. Gathering the BIASED score into the
// weights is the easy wrong answer and still produces a plausible mixture.
// 🪤 The renormalisation epsilon is 1e-20, not the usual 1e-6.
// 🪤 `n_group`/`topk_group` group routing is a NO-OP on this checkpoint (both are 1, so the one
// group holds all experts and the mask is all-ones). This kernel therefore implements plain
// top-k and takes `n_group` only to REFUSE a checkpoint where the machinery would matter.
//
// `bf16_ladder` reproduces vLLM's current dtype ladder for glm5_next: its GateLinear resolves
// out_dtype = None, so the gate GEMM, the sigmoid, the bias add and the renormalisation all run
// in bf16. Production (HF semantics) passes 0 and stays in fp32 throughout.
//
// Sentinel discipline: all K slots are written unconditionally; an unfilled slot is -1.
// Grid: (T,1,1)  Block: (256,1,1) — thread 0 does the tiny top-k for a deterministic order.
extern "C" __global__ void glm5next_router_topk(
    const float* __restrict__ logits,   // [T, E]
    const float* __restrict__ bias,     // [E]
    int* __restrict__ topk_ids,         // [T, K]
    float* __restrict__ topk_weights,   // [T, K]
    const unsigned int num_experts,
    const unsigned int top_k,
    const unsigned int n_group,
    const float routed_scale,
    const unsigned int renormalize,
    const unsigned int bf16_ladder
) {
    const unsigned int t = blockIdx.x;
    if (threadIdx.x != 0) return;
    if (n_group != 1) return; // caller must refuse; grouped routing is not implemented here.

    const float* row = logits + (size_t)t * num_experts;
    int* ids = topk_ids + (size_t)t * top_k;
    float* wts = topk_weights + (size_t)t * top_k;

    // Full-write first: an early return must never leave a stale tail behind.
    for (unsigned int k = 0; k < top_k; ++k) {
        ids[k] = -1;
        wts[k] = 0.0f;
    }

    float best_w[16];
    for (unsigned int k = 0; k < top_k && k < 16; ++k) best_w[k] = 0.0f;

    for (unsigned int k = 0; k < top_k; ++k) {
        float best = -1e30f;
        int arg = -1;
        for (unsigned int e = 0; e < num_experts; ++e) {
            bool taken = false;
            for (unsigned int j = 0; j < k; ++j)
                if (ids[j] == (int)e) { taken = true; break; }
            if (taken) continue;
            float s = 1.0f / (1.0f + expf(-row[e]));
            if (bf16_ladder) s = (float)__float2bfloat16(s);
            float c = s + bias[e];
            if (bf16_ladder) c = (float)__float2bfloat16(c);
            if (c > best) { best = c; arg = (int)e; }
        }
        ids[k] = arg;
        // The WEIGHT is the unbiased score of the chosen expert.
        float s = 1.0f / (1.0f + expf(-row[arg]));
        if (bf16_ladder) s = (float)__float2bfloat16(s);
        best_w[k] = s;
    }

    float sum = 0.0f;
    for (unsigned int k = 0; k < top_k; ++k) {
        sum += best_w[k];
        if (bf16_ladder) sum = (float)__float2bfloat16(sum);
    }
    for (unsigned int k = 0; k < top_k; ++k) {
        float w = best_w[k];
        if (renormalize) {
            w = w / (sum + 1e-20f);
            if (bf16_ladder) w = (float)__float2bfloat16(w);
        }
        w *= routed_scale;
        if (bf16_ladder) w = (float)__float2bfloat16(w);
        wts[k] = w;
    }
}

// ── glm5next_moe_combine ──
// out[t,d] = shared[t,d] + sum_k weights[t,k] * expert_out[t,k,d]
//
// 🔴 `apply_routed_scale_to_output = False` in vLLM's Glm5NextMoE construction, and HF adds the
// shared expert OUTSIDE the weighted sum. So `routed_scale` is already folded into `weights`
// and the shared expert is NOT multiplied by it. Scaling `shared` here would be the same defect
// with a different sign — hence shared is added raw and this kernel takes no scale argument.
// Grid: (T,1,1)  Block: (256,1,1).
extern "C" __global__ void glm5next_moe_combine(
    const __nv_bfloat16* __restrict__ expert_out, // [T, K, H]
    const float* __restrict__ weights,            // [T, K]
    const __nv_bfloat16* __restrict__ shared,     // [T, H]
    __nv_bfloat16* __restrict__ out,              // [T, H]
    const unsigned int hidden,
    const unsigned int top_k
) {
    const unsigned int t = blockIdx.x;
    const __nv_bfloat16* eo = expert_out + (size_t)t * top_k * hidden;
    const float* w = weights + (size_t)t * top_k;
    for (unsigned int d = threadIdx.x; d < hidden; d += blockDim.x) {
        float acc = 0.0f;
        for (unsigned int k = 0; k < top_k; ++k) acc += w[k] * (float)eo[k * hidden + d];
        acc += (float)shared[(size_t)t * hidden + d];  // NOT routed-scaled
        out[(size_t)t * hidden + d] = __float2bfloat16(acc);
    }
}
