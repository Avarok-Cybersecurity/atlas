// SPDX-License-Identifier: AGPL-3.0-only

// Atlas MoE Top-K Softmax kernel for SM121 (GB10).
//
// GPU-side replacement for CPU top-K routing.
// Eliminates D2H copy of gate logits + CPU sort + CPU softmax.
//
// Input:  gate_logits[num_experts] BF16 on device
// Output: expert_indices[top_k] u32 (sorted by logit value, descending)
//         expert_weights[top_k] f32 (softmax probabilities)
//
// Single block, 256 threads. Supports up to 512 experts (2 loads per thread).

#include <cuda_bf16.h>

#define BLOCK_SIZE 256
#define MAX_EXPERTS 512
#define MAX_TOP_K 32  // Must be >= num_experts_per_tok (22 for Super 120B)
#define WARP_SIZE 32

// moe_topk_softmax: find top-K experts and compute softmax weights.
//
// Grid: (1, 1, 1)   Block: (256, 1, 1)
//
// Algorithm:
// 1. Each thread loads up to 2 gate_logits (BF16->f32) for num_experts <= 512
// 2. Block-wide find top-K via parallel warp-shuffle + shared memory reduction
// 3. Compute softmax over top-K values
// 4. Write expert_indices and expert_weights
extern "C" __global__ void moe_topk_softmax(
    const __nv_bfloat16* __restrict__ gate_logits,  // [num_experts] BF16
    unsigned int* __restrict__ expert_indices,       // [top_k] output
    float* __restrict__ expert_weights,              // [top_k] output
    unsigned int num_experts,
    unsigned int top_k,
    unsigned int normalize  // 1 = normalize softmax weights to sum to 1
) {
    __shared__ float s_vals[MAX_EXPERTS];
    __shared__ float s_top_vals[MAX_TOP_K];
    __shared__ unsigned int s_top_idxs[MAX_TOP_K];
    __shared__ float s_warp_val[8];
    __shared__ unsigned int s_warp_idx[8];

    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane = tid % 32;
    const unsigned int num_warps = BLOCK_SIZE / 32;

    // Phase 1: Load gate logits (BF16 -> f32) — 2 elements per thread for 512 experts
    unsigned int actual_n = num_experts < MAX_EXPERTS ? num_experts : MAX_EXPERTS;
    for (unsigned int i = tid; i < actual_n; i += BLOCK_SIZE) {
        s_vals[i] = __bfloat162float(gate_logits[i]);
    }
    // Initialize remaining slots to -inf
    for (unsigned int i = actual_n + tid; i < MAX_EXPERTS; i += BLOCK_SIZE) {
        s_vals[i] = -1e30f;
    }
    __syncthreads();

    // Phase 2: Parallel top-K via iterative warp-shuffle max reduction
    for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
        // Each thread finds local max across its portion of experts
        float local_max = -1e30f;
        unsigned int local_idx = 0;
        for (unsigned int i = tid; i < actual_n; i += BLOCK_SIZE) {
            float v = s_vals[i];
            if (v > local_max) {
                local_max = v;
                local_idx = i;
            }
        }

        // Warp-level reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_val = __shfl_down_sync(0xFFFFFFFF, local_max, offset);
            unsigned int other_idx = __shfl_down_sync(0xFFFFFFFF, local_idx, offset);
            if (other_val > local_max) {
                local_max = other_val;
                local_idx = other_idx;
            }
        }

        // Cross-warp reduction via shared memory
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
            // Invalidate winner for next iteration
            s_vals[best_idx] = -1e30f;
        }
        __syncthreads();
    }

    // Phase 3: Compute softmax over ALL experts, then extract top-K weights.
    //
    // HF/vLLM compute softmax(all_logits)[top_k_indices], NOT softmax(top_k_logits).
    // The difference: with full softmax, each weight is divided by sum of exp over
    // ALL experts (not just top-K), producing smaller absolute weights but correct
    // relative magnitudes.
    //
    // Optimization: s_top_vals[0] IS the global max (first top-K = largest logit).
    // Skip Phase 3a (restore) and Phase 3b (find max). Compute exp+sum directly,
    // adding back top-K contributions that are currently -1e30 in s_vals[].

    float global_max = s_top_vals[0];  // First top-K IS the global max

    // Phase 3c: Compute exp(x - max) for non-top-K experts and parallel sum.
    // Top-K slots in s_vals[] are -1e30, so exp(-1e30 - max) ≈ 0 — correct.
    {
        float local_sum = 0.0f;
        for (unsigned int i = tid; i < actual_n; i += BLOCK_SIZE) {
            local_sum += __expf(s_vals[i] - global_max);
        }
        // Warp-level sum reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
        }
        if (lane == 0) s_warp_val[warp_id] = local_sum;
        __syncthreads();

        // Cross-warp sum + add back top-K contributions (thread 0)
        if (tid == 0) {
            float total = 0.0f;
            for (unsigned int w = 0; w < num_warps; w++) {
                total += s_warp_val[w];
            }
            // Add correct exp values for top-K (replacing the ~0 from -1e30)
            for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
                total += __expf(s_top_vals[t] - global_max);
            }
            s_warp_val[0] = total;
        }
        __syncthreads();
    }

    float exp_sum = s_warp_val[0];

    // Phase 3d: Extract top-K weights from full softmax and write results
    if (tid == 0) {
        for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
            expert_indices[t] = s_top_idxs[t];
            float softmax_weight = __expf(s_top_vals[t] - global_max) / exp_sum;

            if (normalize) {
                s_top_vals[t] = softmax_weight;
            } else {
                expert_weights[t] = softmax_weight;
            }
        }

        if (normalize) {
            float topk_sum = 0.0f;
            for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
                topk_sum += s_top_vals[t];
            }
            for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
                expert_weights[t] = s_top_vals[t] / topk_sum;
            }
        }
    }
}

// Batched variant: process N tokens in parallel, one block per token.
//
// Grid: (N, 1, 1)   Block: (256, 1, 1)
//
// gate_logits:    [N, num_experts] BF16
// expert_indices: [N, top_k] u32
// expert_weights: [N, top_k] f32
extern "C" __global__ void moe_topk_softmax_batched(
    const __nv_bfloat16* __restrict__ gate_logits,  // [N, num_experts]
    unsigned int* __restrict__ expert_indices,       // [N, top_k]
    float* __restrict__ expert_weights,              // [N, top_k]
    unsigned int num_experts,
    unsigned int top_k,
    unsigned int normalize
) {
    __shared__ float s_vals[MAX_EXPERTS];
    __shared__ float s_top_vals[MAX_TOP_K];
    __shared__ unsigned int s_top_idxs[MAX_TOP_K];
    __shared__ float s_warp_val[8];
    __shared__ unsigned int s_warp_idx[8];

    const unsigned int token = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane = tid % 32;
    const unsigned int num_warps = BLOCK_SIZE / 32;

    // Per-token pointer offsets
    const __nv_bfloat16* my_gate = gate_logits + token * num_experts;
    unsigned int* my_indices = expert_indices + token * top_k;
    float* my_weights = expert_weights + token * top_k;

    unsigned int actual_n = num_experts < MAX_EXPERTS ? num_experts : MAX_EXPERTS;
    for (unsigned int i = tid; i < actual_n; i += BLOCK_SIZE) {
        s_vals[i] = __bfloat162float(my_gate[i]);
    }
    for (unsigned int i = actual_n + tid; i < MAX_EXPERTS; i += BLOCK_SIZE) {
        s_vals[i] = -1e30f;
    }
    __syncthreads();

    // Phase 2: Parallel top-K
    for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
        float local_max = -1e30f;
        unsigned int local_idx = 0;
        for (unsigned int i = tid; i < actual_n; i += BLOCK_SIZE) {
            float v = s_vals[i];
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
            s_vals[best_idx] = -1e30f;
        }
        __syncthreads();
    }

    // Phase 3: Full softmax
    float global_max = s_top_vals[0];
    {
        float local_sum = 0.0f;
        for (unsigned int i = tid; i < actual_n; i += BLOCK_SIZE) {
            local_sum += __expf(s_vals[i] - global_max);
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
        }
        if (lane == 0) s_warp_val[warp_id] = local_sum;
        __syncthreads();
        if (tid == 0) {
            float total = 0.0f;
            for (unsigned int w = 0; w < num_warps; w++) {
                total += s_warp_val[w];
            }
            for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
                total += __expf(s_top_vals[t] - global_max);
            }
            s_warp_val[0] = total;
        }
        __syncthreads();
    }

    float exp_sum = s_warp_val[0];
    if (tid == 0) {
        for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
            my_indices[t] = s_top_idxs[t];
            float softmax_weight = __expf(s_top_vals[t] - global_max) / exp_sum;
            if (normalize) {
                s_top_vals[t] = softmax_weight;
            } else {
                my_weights[t] = softmax_weight;
            }
        }
        if (normalize) {
            float topk_sum = 0.0f;
            for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
                topk_sum += s_top_vals[t];
            }
            for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
                my_weights[t] = s_top_vals[t] / topk_sum;
            }
        }
    }
}
