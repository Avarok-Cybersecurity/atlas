// SPDX-License-Identifier: AGPL-3.0-only

// Atlas Causal Conv1d — Depthwise temporal convolution + SiLU activation.
//
// For Qwen3-Next Gated Delta Net layers:
//   Input: [batch, dim, seq_len]  BF16 (dim=8192, d_conv=4)
//   Weight: [dim, 1, d_conv]      BF16 (depthwise: groups=dim, converted to FP32 in-register)
//   Output: [batch, dim, seq_len]  BF16
//
// Causal: only past tokens used (left padding).
// Fused: conv1d + SiLU activation in one kernel.
//
// Two modes:
//   1. Prefill: full sequence convolution
//   2. Decode (update): single token, update sliding window state
//
// Grid: (dim, batch, 1)
// Block: (seq_len clamped to 1024, 1, 1)  [prefill]
//   or   (1, 1, 1)                         [decode]

#include <cuda_bf16.h>

// ============================================================
// PREFILL: Full-sequence causal depthwise conv1d + SiLU
// ============================================================
// Each block handles one (batch, channel) pair across all seq positions.
// Grid: (dim, batch, 1)
// Block: (min(seq_len, 1024), 1, 1)
// For seq_len > 1024, each thread handles multiple positions.
extern "C" __global__ void causal_conv1d_fwd(
    const __nv_bfloat16* __restrict__ input,   // [batch, dim, seq_len]
    const __nv_bfloat16* __restrict__ weight,  // [dim, d_conv]  BF16
    const float* __restrict__ bias,             // [dim] or nullptr
    __nv_bfloat16* __restrict__ output,         // [batch, dim, seq_len]
    unsigned int batch,
    unsigned int dim,
    unsigned int seq_len,
    unsigned int d_conv                         // typically 4
) {
    const unsigned int ch = blockIdx.x;         // channel index
    const unsigned int b = blockIdx.y;          // batch index
    if (ch >= dim || b >= batch) return;

    // Pointers for this (batch, channel)
    const __nv_bfloat16* in_ptr = input + (b * dim + ch) * seq_len;
    __nv_bfloat16* out_ptr = output + (b * dim + ch) * seq_len;
    const __nv_bfloat16* w = weight + ch * d_conv;
    const float b_val = (bias != nullptr) ? bias[ch] : 0.0f;

    // Load BF16 weights into FP32 registers (d_conv is small, typically 4)
    float w_reg[8];  // max d_conv=8
    for (unsigned int i = 0; i < d_conv && i < 8; i++) {
        w_reg[i] = (float)w[i];
    }

    // Each thread processes one or more sequence positions
    for (unsigned int t = threadIdx.x; t < seq_len; t += blockDim.x) {
        float acc = b_val;

        // Causal conv (correlation, matching PyTorch F.conv1d convention):
        // out[t] = sum_{k=0}^{d_conv-1} input[t - (d_conv-1) + k] * weight[k]
        // where input[idx] = 0 if idx < 0 (causal padding)
        // weight[0] applies to oldest, weight[d_conv-1] to newest (current)
        for (unsigned int k = 0; k < d_conv && k < 8; k++) {
            int idx = (int)t - (int)(d_conv - 1) + (int)k;
            if (idx >= 0) {
                acc += (float)in_ptr[idx] * w_reg[k];
            }
        }

        // SiLU activation: x * sigmoid(x)
        float sigmoid_acc = 1.0f / (1.0f + __expf(-acc));
        float silu = acc * sigmoid_acc;

        out_ptr[t] = __float2bfloat16(silu);
    }
}

// ============================================================
// DECODE (UPDATE): Single-token sliding window update + SiLU
// ============================================================
// Updates the conv state (sliding window) and computes one output token.
//
// conv_state: [batch, dim, d_conv]  FP32 (persistent between tokens)
// new_input:  [batch, dim]          BF16 (new token values)
// output:     [batch, dim]          BF16
//
// Algorithm:
//   1. Shift conv_state left by 1: state[:, :, i] = state[:, :, i+1]
//   2. Insert new token: state[:, :, d_conv-1] = new_input
//   3. Compute depthwise conv: out = sum(state * weight) + bias
//   4. Apply SiLU
//
// Grid: (ceil(dim/256), batch, 1)
// Block: (256, 1, 1)
extern "C" __global__ void causal_conv1d_update(
    float* __restrict__ conv_state,             // [batch, dim, d_conv] FP32 (in/out)
    const __nv_bfloat16* __restrict__ new_input, // [batch, dim] BF16
    const __nv_bfloat16* __restrict__ weight,   // [dim, d_conv] BF16
    const float* __restrict__ bias,             // [dim] or nullptr
    __nv_bfloat16* __restrict__ output,         // [batch, dim] BF16
    unsigned int batch,
    unsigned int dim,
    unsigned int d_conv
) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int b = blockIdx.y;
    if (ch >= dim || b >= batch) return;

    // Pointer to this channel's conv state: [d_conv] elements
    float* state = conv_state + (b * dim + ch) * d_conv;

    // 1. Shift state left by 1
    for (unsigned int i = 0; i < d_conv - 1; i++) {
        state[i] = state[i + 1];
    }

    // 2. Insert new token
    state[d_conv - 1] = (float)new_input[b * dim + ch];

    // 3. Depthwise convolution (BF16 weights converted to FP32 in-register)
    const __nv_bfloat16* w = weight + ch * d_conv;
    float acc = (bias != nullptr) ? bias[ch] : 0.0f;
    for (unsigned int k = 0; k < d_conv; k++) {
        acc += state[k] * (float)w[k];
    }

    // 4. SiLU activation
    float sigmoid_acc = 1.0f / (1.0f + __expf(-acc));
    float silu = acc * sigmoid_acc;

    output[b * dim + ch] = __float2bfloat16(silu);
}

// ============================================================
// FP32 OUTPUT variant — prevents BF16 truncation in SSM recurrent path.
// Identical to causal_conv1d_update but output is FP32 instead of BF16.
// Used for decode to prevent precision drift at 8k+ tokens.
// ============================================================
extern "C" __global__ void causal_conv1d_update_f32(
    float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ new_input,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,                     // FP32 output (was BF16)
    unsigned int batch,
    unsigned int dim,
    unsigned int d_conv
) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int b = blockIdx.y;
    if (ch >= dim || b >= batch) return;

    float* state = conv_state + (b * dim + ch) * d_conv;

    for (unsigned int i = 0; i < d_conv - 1; i++) {
        state[i] = state[i + 1];
    }
    state[d_conv - 1] = (float)new_input[b * dim + ch];

    const __nv_bfloat16* w = weight + ch * d_conv;
    float acc = (bias != nullptr) ? bias[ch] : 0.0f;
    for (unsigned int k = 0; k < d_conv; k++) {
        acc += state[k] * (float)w[k];
    }

    float sigmoid_acc = 1.0f / (1.0f + __expf(-acc));
    float silu = acc * sigmoid_acc;

    output[b * dim + ch] = silu;  // FP32 — no truncation!
}

// ============================================================
// PREFILL UPDATE: Multi-token sliding window update + SiLU
// ============================================================
// Processes N tokens sequentially per channel through the conv1d
// sliding window. State lives in registers for the entire loop —
// no global memory reads/writes for state until final writeback.
//
// Input tokens may be non-contiguous (stride between tokens in BF16 elements).
// Output tokens may also be non-contiguous (separate stride).
//
// Grid: (ceil(dim/256), 1, 1)  Block: (256, 1, 1)
extern "C" __global__ void causal_conv1d_update_prefill(
    float* __restrict__ conv_state,             // [dim, d_conv] FP32 (in/out)
    const __nv_bfloat16* __restrict__ input,    // N tokens, input[t * input_stride + ch]
    const __nv_bfloat16* __restrict__ weight,   // [dim, d_conv] BF16
    const float* __restrict__ bias,             // [dim] or nullptr
    __nv_bfloat16* __restrict__ output,         // N tokens, output[t * output_stride + ch]
    unsigned int dim,
    unsigned int d_conv,
    unsigned int seq_len,          // number of tokens
    unsigned int input_stride,     // BF16 elements between consecutive tokens in input
    unsigned int output_stride     // BF16 elements between consecutive tokens in output
) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= dim) return;

    float* state = conv_state + ch * d_conv;
    const __nv_bfloat16* w = weight + ch * d_conv;
    const float b_val = (bias != nullptr) ? bias[ch] : 0.0f;

    // Load weights into registers (d_conv = 4 for Qwen3-Next)
    float w_reg[4];
    for (unsigned int k = 0; k < d_conv && k < 4; k++) {
        w_reg[k] = (float)w[k];
    }

    // Load current sliding window state into registers
    float s[4];
    for (unsigned int k = 0; k < d_conv && k < 4; k++) {
        s[k] = state[k];
    }

    // Process all tokens sequentially — state stays in registers
    for (unsigned int t = 0; t < seq_len; t++) {
        float new_val = (float)input[(unsigned long long)t * input_stride + ch];

        // Shift state left by 1, insert new value
        s[0] = s[1]; s[1] = s[2]; s[2] = s[3]; s[3] = new_val;

        // Depthwise convolution
        float acc = b_val + s[0]*w_reg[0] + s[1]*w_reg[1] + s[2]*w_reg[2] + s[3]*w_reg[3];

        // SiLU activation
        float sigmoid_acc = 1.0f / (1.0f + __expf(-acc));
        output[(unsigned long long)t * output_stride + ch] = __float2bfloat16(acc * sigmoid_acc);
    }

    // Write final state back to global memory
    for (unsigned int k = 0; k < d_conv && k < 4; k++) {
        state[k] = s[k];
    }
}

// ============================================================
// CHUNK2: Fused 2-token conv1d update + SiLU
// ============================================================
// Processes 2 tokens through the conv1d sliding window in one kernel.
// Saves intermediate conv_state (after token 0) for rollback.
//
// Each thread handles one channel independently — the 2-token dependency
// (token 1's window includes token 0's input) is resolved in registers.
//
// Grid: (ceil(dim/256), batch, 1)
// Block: (256, 1, 1)
extern "C" __global__ void causal_conv1d_update_chunk2(
    float* __restrict__ conv_state,              // [batch, dim, d_conv] FP32 (in/out)
    const __nv_bfloat16* __restrict__ new_input, // [batch, 2, dim] BF16
    const __nv_bfloat16* __restrict__ weight,    // [dim, d_conv] BF16
    const float* __restrict__ bias,              // [dim] or nullptr
    __nv_bfloat16* __restrict__ output,          // [batch, 2, dim] BF16
    float* __restrict__ conv_state_intermediate, // [batch, dim, d_conv] FP32 (out)
    unsigned int batch,
    unsigned int dim,
    unsigned int d_conv
) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int b = blockIdx.y;
    if (ch >= dim || b >= batch) return;

    float* state = conv_state + (b * dim + ch) * d_conv;
    float* state_inter = conv_state_intermediate + (b * dim + ch) * d_conv;
    const __nv_bfloat16* w = weight + ch * d_conv;
    const float b_val = (bias != nullptr) ? bias[ch] : 0.0f;

    // Load weights into registers
    float w_reg[4]; // d_conv = 4 for Qwen3-Next
    for (unsigned int k = 0; k < d_conv && k < 4; k++) {
        w_reg[k] = (float)w[k];
    }

    // Load current state into registers
    float s[4];
    for (unsigned int k = 0; k < d_conv && k < 4; k++) {
        s[k] = state[k];
    }

    // ── Token 0: shift + insert + convolve ──
    float in0 = (float)new_input[b * 2 * dim + ch];
    // Shift left by 1, insert in0
    float s0_0 = s[1], s0_1 = s[2], s0_2 = s[3], s0_3 = in0;

    // Save intermediate state (after token 0)
    state_inter[0] = s0_0;
    state_inter[1] = s0_1;
    state_inter[2] = s0_2;
    state_inter[3] = s0_3;

    // Depthwise conv + SiLU for token 0
    float acc0 = b_val + s0_0*w_reg[0] + s0_1*w_reg[1] + s0_2*w_reg[2] + s0_3*w_reg[3];
    float sig0 = 1.0f / (1.0f + __expf(-acc0));
    output[b * 2 * dim + ch] = __float2bfloat16(acc0 * sig0);

    // ── Token 1: shift + insert + convolve ──
    float in1 = (float)new_input[(b * 2 + 1) * dim + ch];
    float s1_0 = s0_1, s1_1 = s0_2, s1_2 = s0_3, s1_3 = in1;

    // Write final conv_state
    state[0] = s1_0;
    state[1] = s1_1;
    state[2] = s1_2;
    state[3] = s1_3;

    // Depthwise conv + SiLU for token 1
    float acc1 = b_val + s1_0*w_reg[0] + s1_1*w_reg[1] + s1_2*w_reg[2] + s1_3*w_reg[3];
    float sig1 = 1.0f / (1.0f + __expf(-acc1));
    output[(b * 2 + 1) * dim + ch] = __float2bfloat16(acc1 * sig1);
}

// ============================================================
// FUSED: Conv1d update + SiLU + L2 normalization (decode)
// ============================================================
// Combines causal_conv1d_update and l2_norm_bf16 into a single kernel.
// For Q+K channels (first qk_channels), applies L2 normalization after SiLU.
// For V channels (remaining), just applies SiLU.
//
// Eliminates 1 kernel launch per SSM layer (36 layers = 36 launches/step)
// and avoids writing Q+K to DRAM only to re-read for normalization.
//
// L2 norm grouping: BLOCK_SIZE=256, head_dim=128 → exactly 2 heads per block.
// Warps 0-3 (tid 0-127) = head A, warps 4-7 (tid 128-255) = head B.
// Requires: qk_channels % 256 == 0 (always true: qk_channels = 2*key_dim = 4096).
//
// Grid: (ceil(dim/256), batch, 1)
// Block: (256, 1, 1)
extern "C" __global__ void causal_conv1d_update_l2norm(
    float* __restrict__ conv_state,             // [batch, dim, d_conv] FP32 (in/out)
    const __nv_bfloat16* __restrict__ new_input, // [batch, dim] BF16
    const __nv_bfloat16* __restrict__ weight,   // [dim, d_conv] BF16
    const float* __restrict__ bias,             // [dim] or nullptr
    __nv_bfloat16* __restrict__ output,         // [batch, dim] BF16
    unsigned int batch,
    unsigned int dim,
    unsigned int d_conv,
    unsigned int qk_channels,   // channels 0..qk_channels-1 get L2 normalized
    unsigned int head_dim,      // L2 norm group size (128)
    float l2_eps                // L2 norm epsilon (1e-6)
) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int b = blockIdx.y;
    const unsigned int tid = threadIdx.x;

    // Does this block contain Q/K channels that need L2 normalization?
    const unsigned int block_start = blockIdx.x * blockDim.x;
    const bool block_needs_l2 = (block_start < qk_channels);

    const bool valid = (ch < dim && b < batch);
    float silu = 0.0f;

    // ── Step 1: Conv1d update + SiLU (same as causal_conv1d_update) ──
    if (valid) {
        float* state = conv_state + (b * dim + ch) * d_conv;

        for (unsigned int i = 0; i < d_conv - 1; i++)
            state[i] = state[i + 1];
        state[d_conv - 1] = (float)new_input[b * dim + ch];

        const __nv_bfloat16* w = weight + ch * d_conv;
        float acc = (bias != nullptr) ? bias[ch] : 0.0f;
        for (unsigned int k = 0; k < d_conv; k++)
            acc += state[k] * (float)w[k];

        float sigmoid_acc = 1.0f / (1.0f + __expf(-acc));
        silu = acc * sigmoid_acc;
    }

    // ── Step 2: L2 normalization for Q+K channels ──
    // 256 threads / 128 head_dim = 2 heads per block.
    // Warps 0-3 (tid 0-127) = head A, warps 4-7 (tid 128-255) = head B.
    if (block_needs_l2) {
        float sq = valid ? (silu * silu) : 0.0f;

        // Warp-level reduction
        const unsigned int warp_id = tid / 32;
        const unsigned int lane = tid % 32;
        for (int offset = 16; offset >= 1; offset >>= 1)
            sq += __shfl_down_sync(0xFFFFFFFF, sq, offset);

        // Store per-warp partial sums
        __shared__ float warp_sums[8];
        if (lane == 0) warp_sums[warp_id] = sq;
        __syncthreads();

        // Cross-warp reduction: 4 warps per head
        // head 0: warps 0-3, head 1: warps 4-7
        const unsigned int head_in_block = tid / head_dim; // 0 or 1
        const unsigned int base_warp = head_in_block * (head_dim / 32);

        // First thread of each head computes inv_norm
        if (tid == 0 || tid == head_dim) {
            float total = warp_sums[base_warp] + warp_sums[base_warp + 1]
                        + warp_sums[base_warp + 2] + warp_sums[base_warp + 3];
            warp_sums[base_warp] = rsqrtf(total + l2_eps);
        }
        __syncthreads();

        // Apply normalization
        if (valid) {
            silu *= warp_sums[base_warp];
        }
    }

    if (valid) {
        output[b * dim + ch] = __float2bfloat16(silu);
    }
}

// ============================================================
// FP32 OUTPUT variant of fused conv1d + L2 norm.
// Prevents BF16 precision drift in the SSM recurrent path at 8k+ tokens.
// ============================================================
extern "C" __global__ void causal_conv1d_update_l2norm_f32(
    float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ new_input,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,                     // FP32 output
    unsigned int batch,
    unsigned int dim,
    unsigned int d_conv,
    unsigned int qk_channels,
    unsigned int head_dim,
    float l2_eps
) {
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int b = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int block_start = blockIdx.x * blockDim.x;
    const bool block_needs_l2 = (block_start < qk_channels);
    const bool valid = (ch < dim && b < batch);
    float silu = 0.0f;

    if (valid) {
        float* state = conv_state + (b * dim + ch) * d_conv;
        for (unsigned int i = 0; i < d_conv - 1; i++)
            state[i] = state[i + 1];
        state[d_conv - 1] = (float)new_input[b * dim + ch];
        const __nv_bfloat16* w = weight + ch * d_conv;
        float acc = (bias != nullptr) ? bias[ch] : 0.0f;
        for (unsigned int k = 0; k < d_conv; k++)
            acc += state[k] * (float)w[k];
        float sigmoid_acc = 1.0f / (1.0f + __expf(-acc));
        silu = acc * sigmoid_acc;
    }

    if (block_needs_l2) {
        float sq = valid ? (silu * silu) : 0.0f;
        const unsigned int warp_id = tid / 32;
        const unsigned int lane = tid % 32;
        for (int offset = 16; offset >= 1; offset >>= 1)
            sq += __shfl_down_sync(0xFFFFFFFF, sq, offset);
        __shared__ float warp_sums[8];
        if (lane == 0) warp_sums[warp_id] = sq;
        __syncthreads();
        const unsigned int head_in_block = tid / head_dim;
        const unsigned int base_warp = head_in_block * (head_dim / 32);
        if (tid == 0 || tid == head_dim) {
            float total = warp_sums[base_warp] + warp_sums[base_warp + 1]
                        + warp_sums[base_warp + 2] + warp_sums[base_warp + 3];
            warp_sums[base_warp] = rsqrtf(total + l2_eps);
        }
        __syncthreads();
        if (valid) {
            silu *= warp_sums[base_warp];
        }
    }

    if (valid) {
        output[b * dim + ch] = silu;  // FP32 — no truncation!
    }
}
