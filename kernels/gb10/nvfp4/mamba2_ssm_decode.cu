// SPDX-License-Identifier: AGPL-3.0-only

// Atlas Mamba-2 Selective SSM — Decode kernel (single token per step).
//
// State equation per head h in group g:
//   dt = softplus(dt_raw + dt_bias), clamped to [dt_min, dt_max]
//   dA = exp(-exp(A_log) * dt)
//   H[hd, s] = dA * H[hd, s] + dt * x[hd] * B[g, s]   (outer product update)
//   y[hd] = sum_s(H[hd, s] * C[g, s]) + D * x[hd]       (output via C projection)
//
// Dimensions (Nemotron-3-Nano-30B):
//   num_heads: 64, head_dim: 64, state_size: 128, n_groups: 8
//   heads_per_group: 8 (heads sharing same B, C vectors)
//
// State H: [batch, num_heads, head_dim, state_size] FP32
//          = [B, 64, 64, 128] — state_size is contiguous (fast dimension).
//
// Grid: (num_heads, batch, 1)   Block: (128, 1, 1)
// Each block handles one (batch, head) pair.
// 128 threads = state_size → each thread handles one state column.

#include <cuda_bf16.h>

#define BLOCK_SIZE 128

extern "C" __global__ void mamba2_ssm_decode(
    // State (in/out): [batch, num_heads, head_dim, state_size] FP32
    float* __restrict__ h_state,
    // SSM input (after conv1d + SiLU): [batch, num_heads * head_dim] BF16
    const __nv_bfloat16* __restrict__ x,
    // B projection: [batch, n_groups * state_size] BF16
    const __nv_bfloat16* __restrict__ B_in,
    // C projection: [batch, n_groups * state_size] BF16
    const __nv_bfloat16* __restrict__ C_in,
    // Raw dt from in_proj: [batch, num_heads] BF16
    const __nv_bfloat16* __restrict__ dt_raw,
    // Learned parameters (static):
    const float* __restrict__ A_log,     // [num_heads] FP32
    const float* __restrict__ D_param,   // [num_heads] FP32
    const float* __restrict__ dt_bias,   // [num_heads] FP32
    // Output: [batch, num_heads * head_dim] BF16
    __nv_bfloat16* __restrict__ output,
    // Dimensions:
    unsigned int batch_size,
    unsigned int num_heads,
    unsigned int head_dim,
    unsigned int state_size,
    unsigned int n_groups,
    // dt clamping:
    float dt_min,
    float dt_max
) {
    const unsigned int head = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (head >= num_heads || b >= batch_size) return;

    const unsigned int tid = threadIdx.x;
    if (tid >= state_size) return;

    // Group index for shared B, C
    const unsigned int heads_per_group = num_heads / n_groups;
    const unsigned int group = head / heads_per_group;

    // ── Compute dt (fused softplus + clamp) ──
    float dt_val = (float)dt_raw[b * num_heads + head] + dt_bias[head];
    // softplus: log(1 + exp(x)), numerically stable
    dt_val = (dt_val > 20.0f) ? dt_val : logf(1.0f + expf(dt_val));
    // clamp
    dt_val = fminf(fmaxf(dt_val, dt_min), dt_max);

    // ── Compute dA = exp(-exp(A_log) * dt) ──
    float neg_A = expf(A_log[head]);  // A_log stores log(-A), so exp gives |A|
    float dA = expf(-neg_A * dt_val);

    float D_val = D_param[head];

    // ── Load B[group, tid] and C[group, tid] ──
    float B_val = (float)B_in[b * n_groups * state_size + group * state_size + tid];
    float C_val = (float)C_in[b * n_groups * state_size + group * state_size + tid];

    // Pre-compute dt * B for the outer product: H += dt * x[hd] * B[s]
    float dtB = dt_val * B_val;

    // ── Pointers ──
    float* H = h_state + ((unsigned long long)(b * num_heads + head) * head_dim * state_size);
    const __nv_bfloat16* x_ptr = x + (unsigned long long)(b * num_heads + head) * head_dim;

    // Shared memory for cross-warp reduction: [4 warps][head_dim]
    // Shared memory sized for max head_dim (128 for Super 120B, 64 for Nano 30B).
    // Using 128 covers all models — unused elements for smaller head_dim are benign.
    __shared__ float smem_warp[4][128];
    __shared__ float smem_x[128];

    // Load x[head_dim] into shared memory
    if (tid < head_dim) {
        smem_x[tid] = (float)x_ptr[tid];
    }
    __syncthreads();

    const unsigned int warp_id = tid / 32;
    const unsigned int lane = tid % 32;

    // ── Main loop: update state + accumulate output ──
    // Each thread handles column `tid` of H[head_dim, state_size].
    // For each row hd: update H[hd, tid], compute y_partial = H[hd, tid] * C[tid].
    // Warp-reduce y_partial, store in smem_warp for final cross-warp reduction.
    for (unsigned int hd = 0; hd < head_dim; hd++) {
        float x_hd = smem_x[hd];

        // State update: H[hd, tid] = dA * H[hd, tid] + dt * x[hd] * B[tid]
        unsigned int idx = hd * state_size + tid;
        float h_val = H[idx];
        h_val = dA * h_val + x_hd * dtB;
        h_val = fminf(fmaxf(h_val, -200.0f), 200.0f);
        H[idx] = h_val;

        // Output contribution: y_partial = H[hd, tid] * C[tid]
        float y_partial = h_val * C_val;

        // Warp-level reduction (128 threads = 4 warps of 32)
        for (int offset = 16; offset >= 1; offset >>= 1)
            y_partial += __shfl_down_sync(0xFFFFFFFF, y_partial, offset);

        // Lane 0 of each warp writes partial sum
        if (lane == 0) smem_warp[warp_id][hd] = y_partial;
    }

    __syncthreads();

    // ── Final cross-warp reduction + D skip connection + write output ──
    // Threads 0..head_dim-1 each handle one output element.
    if (tid < head_dim) {
        float y_val = smem_warp[0][tid] + smem_warp[1][tid]
                    + smem_warp[2][tid] + smem_warp[3][tid];
        // D skip connection: y += D * x
        y_val += D_val * smem_x[tid];
        output[(unsigned long long)(b * num_heads + head) * head_dim + tid] =
            __float2bfloat16(y_val);
    }
}

// ============================================================
// PREFILL: Sequential Mamba-2 (processes multiple tokens)
// ============================================================
// Same algorithm but loops over seq_len tokens.
// Grid: (num_heads, batch, 1)   Block: (128, 1, 1)
extern "C" __global__ void mamba2_ssm_prefill(
    float* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ x,      // [batch, seq_len, num_heads * head_dim]
    const __nv_bfloat16* __restrict__ B_in,   // [batch, seq_len, n_groups * state_size]
    const __nv_bfloat16* __restrict__ C_in,   // [batch, seq_len, n_groups * state_size]
    const __nv_bfloat16* __restrict__ dt_raw, // [batch, seq_len, num_heads]
    const float* __restrict__ A_log,
    const float* __restrict__ D_param,
    const float* __restrict__ dt_bias,
    __nv_bfloat16* __restrict__ output,       // [batch, seq_len, num_heads * head_dim]
    unsigned int batch_size,
    unsigned int seq_len,
    unsigned int num_heads,
    unsigned int head_dim,
    unsigned int state_size,
    unsigned int n_groups,
    float dt_min,
    float dt_max,
    // Strides (BF16 elements) between consecutive tokens:
    unsigned int x_stride,      // typically num_heads * head_dim
    unsigned int bc_stride,     // typically n_groups * state_size
    unsigned int dt_stride,     // typically num_heads
    unsigned int y_stride       // output stride (may differ from x_stride)
) {
    const unsigned int head = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (head >= num_heads || b >= batch_size) return;

    const unsigned int tid = threadIdx.x;
    if (tid >= state_size) return;

    const unsigned int heads_per_group = num_heads / n_groups;
    const unsigned int group = head / heads_per_group;
    float neg_A = expf(A_log[head]);
    float D_val = D_param[head];
    float dt_bias_val = dt_bias[head];

    float* H = h_state + ((unsigned long long)(b * num_heads + head) * head_dim * state_size);

    __shared__ float smem_warp[4][128];
    __shared__ float smem_x[128];

    const unsigned int warp_id = tid / 32;
    const unsigned int lane = tid % 32;

    for (unsigned int t = 0; t < seq_len; t++) {
        // Per-token pointers
        const __nv_bfloat16* x_t = x + (unsigned long long)t * x_stride
            + (unsigned long long)(b * num_heads + head) * head_dim;
        const __nv_bfloat16* B_t = B_in + (unsigned long long)t * bc_stride
            + b * n_groups * state_size + group * state_size;
        const __nv_bfloat16* C_t = C_in + (unsigned long long)t * bc_stride
            + b * n_groups * state_size + group * state_size;

        // dt for this token
        float dt_val = (float)dt_raw[(unsigned long long)t * dt_stride + b * num_heads + head]
                     + dt_bias_val;
        dt_val = (dt_val > 20.0f) ? dt_val : logf(1.0f + expf(dt_val));
        dt_val = fminf(fmaxf(dt_val, dt_min), dt_max);
        float dA = expf(-neg_A * dt_val);

        float B_val = (float)B_t[tid];
        float C_val = (float)C_t[tid];
        float dtB = dt_val * B_val;

        if (tid < head_dim) smem_x[tid] = (float)x_t[tid];
        __syncthreads();

        for (unsigned int hd = 0; hd < head_dim; hd++) {
            float x_hd = smem_x[hd];
            unsigned int idx = hd * state_size + tid;
            float h_val = H[idx];
            h_val = dA * h_val + x_hd * dtB;
            H[idx] = h_val;

            float y_partial = h_val * C_val;
            for (int offset = 16; offset >= 1; offset >>= 1)
                y_partial += __shfl_down_sync(0xFFFFFFFF, y_partial, offset);
            if (lane == 0) smem_warp[warp_id][hd] = y_partial;
        }
        __syncthreads();

        if (tid < head_dim) {
            float y_val = smem_warp[0][tid] + smem_warp[1][tid]
                        + smem_warp[2][tid] + smem_warp[3][tid];
            y_val += D_val * smem_x[tid];
            output[(unsigned long long)t * y_stride
                + (unsigned long long)(b * num_heads + head) * head_dim + tid] =
                __float2bfloat16(y_val);
        }
        __syncthreads();
    }
}
