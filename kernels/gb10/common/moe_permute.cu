// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_bf16.h>

// Atlas MoE token permutation kernels.
//
// Reorders tokens by expert assignment for batched GEMM,
// and scatters results back with weighted accumulation.
//
// Memory layout:
//   hidden_states: [num_tokens, hidden_size] BF16
//   topk_ids:      [num_tokens, topk] int32
//   topk_weights:  [num_tokens, topk] float32
//   permuted_out:  [num_tokens * topk, hidden_size] BF16
//   expert_offsets: [num_experts + 1] int32 (prefix sum of tokens per expert)

// Permute tokens: gather from hidden_states into expert-sorted order.
// Each thread handles one element of one token.
extern "C" __global__ void moe_permute_tokens(
    const __nv_bfloat16* __restrict__ hidden_states,  // [num_tokens, hidden_size]
    __nv_bfloat16* __restrict__ permuted,              // [total_expanded, hidden_size]
    const int* __restrict__ sorted_token_ids,          // [total_expanded] maps permuted row -> original token
    unsigned int hidden_size,
    unsigned int total_expanded
) {
    unsigned int row = blockIdx.x;
    unsigned int col = threadIdx.x;

    if (row >= total_expanded) return;

    int src_token = sorted_token_ids[row];
    // Copy hidden_size elements — each thread handles one element,
    // loop if hidden_size > blockDim.x
    for (unsigned int c = col; c < hidden_size; c += blockDim.x) {
        permuted[row * hidden_size + c] = hidden_states[src_token * hidden_size + c];
    }
}

// Unpermute and weighted reduce: scatter expert outputs back to original token order
// and accumulate with topk weights.
//
// For each original token, sums over its topk expert outputs:
//   output[token, :] = sum_k( topk_weight[token, k] * expert_output[permuted_idx[token,k], :] )
extern "C" __global__ void moe_unpermute_reduce(
    const __nv_bfloat16* __restrict__ expert_output,   // [total_expanded, hidden_size]
    __nv_bfloat16* __restrict__ output,                 // [num_tokens, hidden_size]
    const int* __restrict__ sorted_token_ids,           // [total_expanded]
    const float* __restrict__ topk_weights,             // [num_tokens, topk]
    unsigned int hidden_size,
    unsigned int num_tokens,
    unsigned int topk,
    unsigned int total_expanded
) {
    // Each block handles one original token
    unsigned int token = blockIdx.x;
    unsigned int col = threadIdx.x;

    if (token >= num_tokens) return;

    for (unsigned int c = col; c < hidden_size; c += blockDim.x) {
        float acc = 0.0f;
        for (unsigned int k = 0; k < topk; k++) {
            unsigned int perm_row = token * topk + k;
            if (perm_row < total_expanded) {
                float w = topk_weights[token * topk + k];
                float val = __bfloat162float(expert_output[perm_row * hidden_size + c]);
                acc += w * val;
            }
        }
        output[token * hidden_size + c] = __float2bfloat16(acc);
    }
}

// Count tokens per expert (for building sorted indices).
// Uses atomicAdd to count how many tokens are assigned to each expert.
extern "C" __global__ void moe_count_experts(
    const int* __restrict__ topk_ids,   // [num_tokens, topk]
    int* __restrict__ expert_counts,     // [num_experts] — zeroed before launch
    unsigned int num_tokens,
    unsigned int topk
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = num_tokens * topk;
    if (idx < total) {
        int expert_id = topk_ids[idx];
        atomicAdd(&expert_counts[expert_id], 1);
    }
}

// Unpermute with pre-built reverse map: token_to_perm[token, k] = row in expert_output.
// Each block handles one original token, accumulates topk expert outputs with weights.
//
// Grid: (num_tokens, 1, 1)  Block: (256, 1, 1)
extern "C" __global__ void moe_unpermute_reduce_indexed(
    const __nv_bfloat16* __restrict__ expert_output,  // [total_expanded, hidden_size]
    __nv_bfloat16* __restrict__ output,                // [num_tokens, hidden_size]
    const int* __restrict__ token_to_perm,             // [num_tokens, topk] reverse map
    const float* __restrict__ topk_weights,            // [num_tokens, topk]
    unsigned int hidden_size,
    unsigned int num_tokens,
    unsigned int topk
) {
    unsigned int token = blockIdx.x;
    if (token >= num_tokens) return;

    for (unsigned int c = threadIdx.x; c < hidden_size; c += blockDim.x) {
        float acc = 0.0f;
        for (unsigned int k = 0; k < topk; k++) {
            int perm_row = token_to_perm[token * topk + k];
            float w = topk_weights[token * topk + k];
            float val = __bfloat162float(expert_output[perm_row * hidden_size + c]);
            acc += w * val;
        }
        output[token * hidden_size + c] = __float2bfloat16(acc);
    }
}

// Batched blend: for each token, compute sigmoid(dot(normed, gate_weight)) and
// blend shared expert output into routed output.
//
// output[token] += sigmoid(gate_scalar) * shared_out[token]
// where gate_scalar = dot(normed[token], gate_weight)
//
// Grid: (num_tokens, 1, 1)  Block: (256, 1, 1)
extern "C" __global__ void moe_batched_blend(
    __nv_bfloat16* __restrict__ output,          // [num_tokens, hidden_size] (in/out)
    const __nv_bfloat16* __restrict__ shared_out, // [num_tokens, hidden_size]
    const __nv_bfloat16* __restrict__ normed,     // [num_tokens, hidden_size]
    const __nv_bfloat16* __restrict__ gate_weight, // [hidden_size] shared expert gate
    unsigned int hidden_size,
    unsigned int num_tokens
) {
    __shared__ float s_dot_partial[8]; // one per warp (256/32=8)

    unsigned int token = blockIdx.x;
    if (token >= num_tokens) return;

    unsigned int tid = threadIdx.x;
    unsigned int warp_id = tid / 32;
    unsigned int lane = tid % 32;

    const __nv_bfloat16* my_normed = normed + token * hidden_size;
    const __nv_bfloat16* my_shared = shared_out + token * hidden_size;
    __nv_bfloat16* my_output = output + token * hidden_size;

    // Phase 1: dot product normed[token] . gate_weight
    // NULL gate_weight = no gate modulation → sigmoid=1.0 (always include shared expert)
    float local_dot = 0.0f;
    if (gate_weight != 0) {
        for (unsigned int i = tid; i < hidden_size; i += blockDim.x) {
            float n = __bfloat162float(my_normed[i]);
            float g = __bfloat162float(gate_weight[i]);
            local_dot += n * g;
        }
    }

    // Warp-level sum
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_dot += __shfl_down_sync(0xFFFFFFFF, local_dot, offset);
    }
    if (lane == 0) s_dot_partial[warp_id] = local_dot;
    __syncthreads();

    // Cross-warp sum (thread 0)
    float gate_scalar;
    if (tid == 0) {
        if (gate_weight == 0) {
            // No gate: always include shared expert at full weight
            gate_scalar = 1.0f;
        } else {
            float total = 0.0f;
            for (unsigned int w = 0; w < blockDim.x / 32; w++) {
                total += s_dot_partial[w];
            }
            gate_scalar = 1.0f / (1.0f + __expf(-total));
        }
        s_dot_partial[0] = gate_scalar;
    }
    __syncthreads();
    gate_scalar = s_dot_partial[0];

    // Phase 2: output[i] += gate_scalar * shared_out[i]
    for (unsigned int i = tid; i < hidden_size; i += blockDim.x) {
        float o = __bfloat162float(my_output[i]);
        float s = __bfloat162float(my_shared[i]);
        my_output[i] = __float2bfloat16(o + gate_scalar * s);
    }
}

// Sort tokens by expert assignment: counting sort producing sorted indices,
// expert prefix sum offsets, and a reverse map for unpermute.
//
// Grid: (1, 1, 1)  Block: (256, 1, 1)
//
// Output layout: sorted_token_ids groups all slots for the same expert
// contiguously. expert_offsets[e] is the first sorted position for expert e.
// token_to_perm[slot] = sorted position of original slot.
extern "C" __global__ void moe_sort_by_expert(
    const unsigned int* __restrict__ topk_ids,      // [total_expanded] expert ids
    int* __restrict__ sorted_token_ids,              // [total_expanded] → original token index
    int* __restrict__ sorted_expert_ids,             // [total_expanded] → expert id per sorted pos
    int* __restrict__ expert_offsets,                 // [num_experts + 1] prefix sum
    int* __restrict__ token_to_perm,                  // [total_expanded] → sorted position
    unsigned int total_expanded,
    unsigned int num_experts,
    unsigned int topk
) {
    // Supports up to 512 experts (counts[512] + offsets[513] = 4100 bytes shared mem).
    __shared__ unsigned int counts[1024];
    __shared__ unsigned int offsets[1025];

    // Clear counts
    for (unsigned int i = threadIdx.x; i < num_experts; i += blockDim.x)
        counts[i] = 0;
    __syncthreads();

    // Phase 1: histogram
    for (unsigned int i = threadIdx.x; i < total_expanded; i += blockDim.x)
        atomicAdd(&counts[topk_ids[i]], 1);
    __syncthreads();

    // Phase 2: prefix sum (single thread)
    if (threadIdx.x == 0) {
        offsets[0] = 0;
        for (unsigned int e = 0; e < num_experts; e++)
            offsets[e + 1] = offsets[e] + counts[e];
        for (unsigned int e = 0; e <= num_experts; e++)
            expert_offsets[e] = (int)offsets[e];
    }
    __syncthreads();

    // Phase 3: reset counts for scatter
    for (unsigned int i = threadIdx.x; i < num_experts; i += blockDim.x)
        counts[i] = 0;
    __syncthreads();

    // Phase 4: scatter to sorted positions
    for (unsigned int i = threadIdx.x; i < total_expanded; i += blockDim.x) {
        unsigned int expert_id = topk_ids[i];
        unsigned int pos = offsets[expert_id] + atomicAdd(&counts[expert_id], 1);
        sorted_token_ids[pos] = (int)(i / topk);
        sorted_expert_ids[pos] = (int)expert_id;
        token_to_perm[i] = (int)pos;
    }
}
