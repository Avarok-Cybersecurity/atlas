// SPDX-License-Identifier: AGPL-3.0-only

// Atlas MoE Top-K Sigmoid kernel for Nemotron-H.
//
// Nemotron-H uses sigmoid routing (NOT softmax like Qwen3/DeepSeek):
//   scores = sigmoid(logits)
//   selection = scores + bias   (bias affects WHICH experts, not their weights)
//   indices  = topk(selection)
//   weights  = scores[indices]  (pre-bias sigmoid scores)
//   weights /= sum(weights)     (if norm_topk_prob)
//   weights *= scaling_factor   (routed_scaling_factor, e.g., 2.5)
//
// Grid: (1, 1, 1)   Block: (256, 1, 1)

#include <cuda_bf16.h>

#define BLOCK_SIZE 256
#define MAX_EXPERTS 512
#define MAX_TOP_K 24
#define WARP_SIZE 32

extern "C" __global__ void moe_topk_sigmoid(
    const __nv_bfloat16* __restrict__ gate_logits,  // [num_experts] BF16
    const float* __restrict__ bias,                  // [num_experts] F32
    unsigned int* __restrict__ expert_indices,       // [top_k] output
    float* __restrict__ expert_weights,              // [top_k] output
    unsigned int num_experts,
    unsigned int top_k,
    unsigned int normalize,       // 1 = normalize weights to sum to 1
    float scaling_factor          // routed_scaling_factor (applied to final weights)
) {
    __shared__ float s_sigmoid[MAX_EXPERTS];     // pre-bias sigmoid (for weights)
    __shared__ float s_selection[MAX_EXPERTS];   // sigmoid + bias (for top-K selection)
    __shared__ float s_top_vals[MAX_TOP_K];      // top-K selection scores
    __shared__ unsigned int s_top_idxs[MAX_TOP_K];
    __shared__ float s_warp_val[8];
    __shared__ unsigned int s_warp_idx[8];

    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane = tid % 32;
    const unsigned int num_warps = BLOCK_SIZE / 32;

    unsigned int actual_n = num_experts < MAX_EXPERTS ? num_experts : MAX_EXPERTS;

    // Phase 1: Load gate logits, compute sigmoid, add bias for selection
    for (unsigned int i = tid; i < actual_n; i += BLOCK_SIZE) {
        float logit = __bfloat162float(gate_logits[i]);
        float sig = 1.0f / (1.0f + __expf(-logit));
        s_sigmoid[i] = sig;
        s_selection[i] = sig + bias[i];
    }
    for (unsigned int i = actual_n + tid; i < MAX_EXPERTS; i += BLOCK_SIZE) {
        s_sigmoid[i] = -1e30f;
        s_selection[i] = -1e30f;
    }
    __syncthreads();

    // Phase 2: Parallel top-K from selection scores (sigmoid + bias)
    for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
        float local_max = -1e30f;
        unsigned int local_idx = 0;
        for (unsigned int i = tid; i < actual_n; i += BLOCK_SIZE) {
            float v = s_selection[i];
            if (v > local_max) {
                local_max = v;
                local_idx = i;
            }
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_val = __shfl_down_sync(0xFFFFFFFF, local_max, offset);
            unsigned int other_idx = __shfl_down_sync(0xFFFFFFFF, local_idx, offset);
            if (other_val > local_max) {
                local_max = other_val;
                local_idx = other_idx;
            }
        }

        if (lane == 0) {
            s_warp_val[warp_id] = local_max;
            s_warp_idx[warp_id] = local_idx;
        }
        __syncthreads();

        if (tid == 0) {
            float best_val = s_warp_val[0];
            unsigned int best_idx = s_warp_idx[0];
            for (unsigned int w = 1; w < num_warps; w++) {
                if (s_warp_val[w] > best_val) {
                    best_val = s_warp_val[w];
                    best_idx = s_warp_idx[w];
                }
            }
            s_top_vals[t] = best_val;
            s_top_idxs[t] = best_idx;
            s_selection[best_idx] = -1e30f;  // invalidate winner
        }
        __syncthreads();
    }

    // Phase 3: Gather pre-bias sigmoid weights, normalize, and scale
    if (tid == 0) {
        float topk_sum = 0.0f;
        for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
            unsigned int idx = s_top_idxs[t];
            float w = s_sigmoid[idx];  // pre-bias sigmoid score
            s_top_vals[t] = w;
            topk_sum += w;
        }

        if (normalize && topk_sum > 1e-20f) {
            for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
                s_top_vals[t] /= topk_sum;
            }
        }

        for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
            expert_indices[t] = s_top_idxs[t];
            expert_weights[t] = s_top_vals[t] * scaling_factor;
        }
    }
}

// Batched variant: process N tokens in parallel, one block per token.
//
// Grid: (N, 1, 1)   Block: (256, 1, 1)
extern "C" __global__ void moe_topk_sigmoid_batched(
    const __nv_bfloat16* __restrict__ gate_logits,  // [N, num_experts] BF16
    const float* __restrict__ bias,                  // [num_experts] F32
    unsigned int* __restrict__ expert_indices,       // [N, top_k] output
    float* __restrict__ expert_weights,              // [N, top_k] output
    unsigned int num_experts,
    unsigned int top_k,
    unsigned int normalize,
    float scaling_factor
) {
    __shared__ float s_sigmoid[MAX_EXPERTS];
    __shared__ float s_selection[MAX_EXPERTS];
    __shared__ float s_top_vals[MAX_TOP_K];
    __shared__ unsigned int s_top_idxs[MAX_TOP_K];
    __shared__ float s_warp_val[8];
    __shared__ unsigned int s_warp_idx[8];

    const unsigned int token = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane = tid % 32;
    const unsigned int num_warps = BLOCK_SIZE / 32;

    const __nv_bfloat16* my_gate = gate_logits + token * num_experts;
    unsigned int* my_indices = expert_indices + token * top_k;
    float* my_weights = expert_weights + token * top_k;

    unsigned int actual_n = num_experts < MAX_EXPERTS ? num_experts : MAX_EXPERTS;
    for (unsigned int i = tid; i < actual_n; i += BLOCK_SIZE) {
        float logit = __bfloat162float(my_gate[i]);
        float sig = 1.0f / (1.0f + __expf(-logit));
        s_sigmoid[i] = sig;
        s_selection[i] = sig + bias[i];
    }
    for (unsigned int i = actual_n + tid; i < MAX_EXPERTS; i += BLOCK_SIZE) {
        s_sigmoid[i] = -1e30f;
        s_selection[i] = -1e30f;
    }
    __syncthreads();

    for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
        float local_max = -1e30f;
        unsigned int local_idx = 0;
        for (unsigned int i = tid; i < actual_n; i += BLOCK_SIZE) {
            float v = s_selection[i];
            if (v > local_max) {
                local_max = v;
                local_idx = i;
            }
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_val = __shfl_down_sync(0xFFFFFFFF, local_max, offset);
            unsigned int other_idx = __shfl_down_sync(0xFFFFFFFF, local_idx, offset);
            if (other_val > local_max) {
                local_max = other_val;
                local_idx = other_idx;
            }
        }
        if (lane == 0) {
            s_warp_val[warp_id] = local_max;
            s_warp_idx[warp_id] = local_idx;
        }
        __syncthreads();
        if (tid == 0) {
            float best_val = s_warp_val[0];
            unsigned int best_idx = s_warp_idx[0];
            for (unsigned int w = 1; w < num_warps; w++) {
                if (s_warp_val[w] > best_val) {
                    best_val = s_warp_val[w];
                    best_idx = s_warp_idx[w];
                }
            }
            s_top_vals[t] = best_val;
            s_top_idxs[t] = best_idx;
            s_selection[best_idx] = -1e30f;
        }
        __syncthreads();
    }

    if (tid == 0) {
        float topk_sum = 0.0f;
        for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
            unsigned int idx = s_top_idxs[t];
            float w = s_sigmoid[idx];
            s_top_vals[t] = w;
            topk_sum += w;
        }
        if (normalize && topk_sum > 1e-20f) {
            for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
                s_top_vals[t] /= topk_sum;
            }
        }
        for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
            my_indices[t] = s_top_idxs[t];
            my_weights[t] = s_top_vals[t] * scaling_factor;
        }
    }
}
