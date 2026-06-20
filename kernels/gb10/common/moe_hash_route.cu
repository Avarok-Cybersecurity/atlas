// SPDX-License-Identifier: AGPL-3.0-only

// Atlas MoE hash-routing kernel for DeepSeek-V4 (first `num_hash_layers` MoE
// layers, paper §2.1).
//
// Unlike the learned-gate path (moe_topk_sqrtsoftplus), expert SELECTION here
// is a static lookup `tid2eid[token_id]` (a frozen token-id → expert-id table),
// not a top-K of the gate scores. The learned gate still produces the per-expert
// scores that WEIGHT the selected experts; only *which* experts is static.
//
// Reference (transformers DeepseekV4HashRouter.forward):
//   logits  = gate(h)
//   scores  = sqrtsoftplus(logits) = sqrt(log(1 + exp(logits)))
//   indices = tid2eid[token_id]          // [top_k], static
//   weights = scores[indices]
//   weights = weights / (sum(weights) + 1e-20)   // if norm_topk_prob
//   weights = weights * routed_scaling_factor
//
// Grid: (1, 1, 1)   Block: (256, 1, 1)  — top_k is tiny (≤8), thread 0 does it.

#include <cuda_bf16.h>

#define MAX_TOP_K 32

extern "C" __global__ void moe_hash_route(
    const __nv_bfloat16* __restrict__ gate_logits,  // [num_experts] BF16
    const long* __restrict__ tid2eid,               // [vocab_size, top_k] i64
    const unsigned int* __restrict__ token_id_ptr,  // [1] u32 (device) — token id
    unsigned int* __restrict__ expert_indices,      // [top_k] output
    float* __restrict__ expert_weights,             // [top_k] output
    unsigned int num_experts,
    unsigned int top_k,
    unsigned int normalize,       // 1 = normalize weights to sum to 1
    float scaling_factor          // routed_scaling_factor
) {
    if (threadIdx.x != 0) return;

    const unsigned int tok = token_id_ptr[0];
    const long* row = tid2eid + (size_t)tok * (size_t)top_k;

    float w_local[MAX_TOP_K];
    unsigned int idx_local[MAX_TOP_K];
    float topk_sum = 0.0f;

    for (unsigned int t = 0; t < top_k && t < MAX_TOP_K; t++) {
        unsigned int e = (unsigned int)row[t];
        if (e >= num_experts) e = 0;  // defensive clamp
        idx_local[t] = e;
        float logit = __bfloat162float(gate_logits[e]);
        float score = sqrtf(logf(1.0f + __expf(logit)));
        w_local[t] = score;
        topk_sum += score;
    }

    if (normalize && topk_sum > 1e-20f) {
        for (unsigned int t = 0; t < top_k && t < MAX_TOP_K; t++) {
            w_local[t] /= topk_sum;
        }
    }

    for (unsigned int t = 0; t < top_k && t < MAX_TOP_K; t++) {
        expert_indices[t] = idx_local[t];
        expert_weights[t] = w_local[t] * scaling_factor;
    }
}

// Batched variant: process N tokens in parallel, one block per token.
//
// Grid: (N, 1, 1)   Block: (256, 1, 1)  — thread 0 of each block does it.
extern "C" __global__ void moe_hash_route_batched(
    const __nv_bfloat16* __restrict__ gate_logits,  // [N, num_experts] BF16
    const long* __restrict__ tid2eid,               // [vocab_size, top_k] i64
    const unsigned int* __restrict__ token_ids,     // [N] u32 (device)
    unsigned int* __restrict__ expert_indices,      // [N, top_k] output
    float* __restrict__ expert_weights,             // [N, top_k] output
    unsigned int num_experts,
    unsigned int top_k,
    unsigned int normalize,
    float scaling_factor
) {
    if (threadIdx.x != 0) return;

    const unsigned int token = blockIdx.x;
    const __nv_bfloat16* my_gate = gate_logits + (size_t)token * num_experts;
    unsigned int* my_indices = expert_indices + (size_t)token * top_k;
    float* my_weights = expert_weights + (size_t)token * top_k;

    const unsigned int tok = token_ids[token];
    const long* row = tid2eid + (size_t)tok * (size_t)top_k;

    float w_local[MAX_TOP_K];
    unsigned int idx_local[MAX_TOP_K];
    float topk_sum = 0.0f;

    for (unsigned int t = 0; t < top_k && t < MAX_TOP_K; t++) {
        unsigned int e = (unsigned int)row[t];
        if (e >= num_experts) e = 0;
        idx_local[t] = e;
        float logit = __bfloat162float(my_gate[e]);
        float score = sqrtf(logf(1.0f + __expf(logit)));
        w_local[t] = score;
        topk_sum += score;
    }

    if (normalize && topk_sum > 1e-20f) {
        for (unsigned int t = 0; t < top_k && t < MAX_TOP_K; t++) {
            w_local[t] /= topk_sum;
        }
    }

    for (unsigned int t = 0; t < top_k && t < MAX_TOP_K; t++) {
        my_indices[t] = idx_local[t];
        my_weights[t] = w_local[t] * scaling_factor;
    }
}
