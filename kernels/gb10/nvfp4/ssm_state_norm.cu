// SPDX-License-Identifier: AGPL-3.0-only

// Atlas Fused SSM State Normalization kernel.
//
// Normalizes the Frobenius norm of each head's h_state matrix across ALL
// SSM layers in a single kernel launch to prevent catastrophic state
// explosion during long chunked prefill sequences.
//
// Without normalization, SSM h_state grows unbounded during prefill
// (which lacks the per-token norm clamping of the decode GDN kernel).
// After 30K+ tokens of chunked prefill, the state saturates and the
// model "forgets" recent tokens — the SSM forgetting bug (Stuffed Mamba).
//
// Called between prefill chunks (every 8K tokens) to keep state bounded.
//
// State layout per layer: [num_heads, k_dim, v_dim] FP32 (v_dim contiguous)
// Pointer table: [num_layers] pointers to each layer's h_state
//
// Grid: (num_heads, num_layers, 1)   — fused across all SSM layers
// Block: (v_dim, 1, 1)              — typically 128 threads

#include <cuda_bf16.h>

// Softer clamp for inter-chunk prefill: allows h_state to carry full context
// between chunks without catastrophic loss. The decode GDN kernel has its own
// per-token norm clamping (50 model-specific, 1000 common) to prevent explosion
// during generation. This only affects the prefill→prefill boundary.
// Old value (50.0) destroyed ~50% of context at chunk boundaries, causing
// the model to output </think> immediately after >8K token chunked prefill.
#define MAX_NORM 200.0f

extern "C" __global__ void ssm_state_clamp_norm_fused(
    float** __restrict__ h_state_ptrs,  // [num_layers] device pointers
    unsigned int num_heads,
    unsigned int k_dim,
    unsigned int v_dim
) {
    const unsigned int head = blockIdx.x;
    const unsigned int layer = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    if (head >= num_heads || tid >= v_dim) return;

    float* H = h_state_ptrs[layer] + (unsigned long long)head * k_dim * v_dim;

    // Compute per-column sum of squares (each thread handles one v_dim column)
    float local_sq = 0.0f;
    for (unsigned int j = 0; j < k_dim; j++) {
        float v = H[j * v_dim + tid];
        local_sq += v * v;
    }

    // Block-wide reduction to get full head Frobenius norm²
    unsigned int mask = __activemask();
    float warp_sum = local_sq;
    warp_sum += __shfl_down_sync(mask, warp_sum, 16);
    warp_sum += __shfl_down_sync(mask, warp_sum, 8);
    warp_sum += __shfl_down_sync(mask, warp_sum, 4);
    warp_sum += __shfl_down_sync(mask, warp_sum, 2);
    warp_sum += __shfl_down_sync(mask, warp_sum, 1);

    __shared__ float norm_sums[4];
    unsigned int warp_id = tid / 32;
    unsigned int lane_id = tid % 32;
    if (lane_id == 0) norm_sums[warp_id] = warp_sum;
    __syncthreads();

    if (tid < 4) {
        float s = norm_sums[tid];
        s += __shfl_down_sync(0xf, s, 2);
        s += __shfl_down_sync(0xf, s, 1);
        norm_sums[0] = s;
    }
    __syncthreads();
    float head_norm_sq = norm_sums[0];

    // Clamp: if ||h[head]||_F > MAX_NORM, scale down
    if (head_norm_sq > MAX_NORM * MAX_NORM) {
        float scale = MAX_NORM * rsqrtf(head_norm_sq);
        for (unsigned int j = 0; j < k_dim; j++) {
            H[j * v_dim + tid] *= scale;
        }
    }
}
