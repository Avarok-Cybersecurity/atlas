// SPDX-License-Identifier: AGPL-3.0-only

// Rotary Position Embedding (RoPE) for SM121.
//
// Applies rotary embeddings to Q and K tensors in-place.
// Supports partial rotary: only first `rotary_dim` dimensions are rotated.
//
// For Qwen3-Next: head_dim=256, partial_rotary_factor=0.25, rotary_dim=64, theta=10M
//
// Uses the "rotate_half" convention (matching HuggingFace):
//   Pairs (i, i + half_rot) where half_rot = rotary_dim / 2
//   x'_i          = x_i          * cos(pos * freq_i) - x_{i+half_rot} * sin(pos * freq_i)
//   x'_{i+half_rot} = x_{i+half_rot} * cos(pos * freq_i) + x_i          * sin(pos * freq_i)
// where freq_i = 1.0 / (theta ^ (2i / rotary_dim))
//
// Memory layout:
//   Q: [batch, seq_len, num_q_heads, head_dim]  BF16
//   K: [batch, seq_len, num_kv_heads, head_dim] BF16
//   positions: [batch, seq_len]  uint32  (absolute position for each token)
//
// Grid: (num_q_heads + num_kv_heads, ceil(seq_len / 4), batch)
// Block: (128, 1, 1)
//   - First num_q_heads blocks handle Q, remaining num_kv_heads blocks handle K
//   - Each block processes 4 sequence positions (128 threads / 32 pairs per pos)
//   - Each thread handles one (cos, sin) rotation pair

#include <cuda_bf16.h>

extern "C" __global__ void rope_forward(
    __nv_bfloat16* __restrict__ Q,          // [batch, seq_len, num_q_heads, head_dim]
    __nv_bfloat16* __restrict__ K,          // [batch, seq_len, num_kv_heads, head_dim]
    const unsigned int* __restrict__ positions,  // [batch, seq_len]
    const unsigned int seq_len,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int rotary_dim,           // Number of dims to rotate (e.g., 64)
    const float* __restrict__ inv_freq,       // [rotary_dim/2] precomputed YaRN frequencies
    const float theta                        // Base frequency (unused when inv_freq != NULL)
) {
    const unsigned int head_idx = blockIdx.x;     // Combined Q+K head index
    const unsigned int seq_block = blockIdx.y;    // Which group of seq positions
    const unsigned int batch = blockIdx.z;
    const unsigned int tid = threadIdx.x;

    // Determine if we're processing Q or K
    const bool is_q = (head_idx < num_q_heads);
    const unsigned int head = is_q ? head_idx : (head_idx - num_q_heads);
    const unsigned int num_heads = is_q ? num_q_heads : num_kv_heads;

    if (!is_q && head >= num_kv_heads) return;

    // Each block handles rotary_dim/2 pairs per position.
    // With rotary_dim=64: 32 pairs per position.
    // With 128 threads: 128 / 32 = 4 positions per block.
    const unsigned int pairs_per_pos = rotary_dim / 2;
    const unsigned int pos_per_block = 128 / pairs_per_pos;
    // Guard: if rotary_dim > 256, pairs_per_pos > 128, need different mapping
    // For rotary_dim=64: pairs_per_pos=32, pos_per_block=4

    const unsigned int local_pos = tid / pairs_per_pos;   // 0..3
    const unsigned int pair_idx = tid % pairs_per_pos;     // 0..31

    const unsigned int seq_pos = seq_block * pos_per_block + local_pos;
    if (seq_pos >= seq_len) return;

    // Get absolute position for this token
    const unsigned int abs_pos = positions[batch * seq_len + seq_pos];

    // Use precomputed YaRN inv_freq if available, else compute from theta
    const float freq = (inv_freq != 0) ? inv_freq[pair_idx]
        : (1.0f / powf(theta, (float)(2 * pair_idx) / (float)rotary_dim));
    const float angle = (float)abs_pos * freq;
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle);

    // Pointer to the head's data at this sequence position
    const unsigned int stride = num_heads * head_dim;
    __nv_bfloat16* ptr;
    if (is_q) {
        ptr = Q + batch * seq_len * (num_q_heads * head_dim)
                + seq_pos * (num_q_heads * head_dim)
                + head * head_dim;
    } else {
        ptr = K + batch * seq_len * (num_kv_heads * head_dim)
                + seq_pos * (num_kv_heads * head_dim)
                + head * head_dim;
    }

    // Load the pair (interleaved convention: pair (2i, 2i+1) matching Mistral weight storage)
    // Mistral uses rope_interleave=True: weights are stored as (d0,d1),(d2,d3),...
    // HF converts interleaved→split-half before rotate_half, but we rotate in-place.
    const unsigned int half_rot = rotary_dim / 2;
    const unsigned int d0 = 2 * pair_idx;              // Interleaved: (0,1), (2,3), ...
    const unsigned int d1 = 2 * pair_idx + 1;
    float x0 = (float)ptr[d0];
    float x1 = (float)ptr[d1];

    // Apply rotation
    float y0 = x0 * cos_val - x1 * sin_val;
    float y1 = x1 * cos_val + x0 * sin_val;

    // Store back
    ptr[d0] = __float2bfloat16(y0);
    ptr[d1] = __float2bfloat16(y1);
}
// YaRN RoPE variant (added 2026-04-05): uses pre-computed inverse frequency table instead
// of computing frequencies from theta. Required for models with
// non-standard RoPE scaling (Mistral Small 4 llama_4_scaling, etc.)
// where frequencies are NTK-aware interpolated at load time.
//
// Grid: (num_q_heads + num_kv_heads, seq_blocks, batch)
// Block: (128, 1, 1) — same as rope_forward
// ═══════════════════════════════════════════════════════════════════
extern "C" __global__ void rope_forward_yarn(
    __nv_bfloat16* __restrict__ Q,
    __nv_bfloat16* __restrict__ K,
    const unsigned int* __restrict__ positions,
    const unsigned int seq_len,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int rotary_dim,
    const float* __restrict__ inv_freq,       // [rotary_dim/2] pre-computed frequencies
    const float theta                          // unused (kept for API compat, freq from inv_freq)
) {
    (void)theta;  // frequencies come from inv_freq table

    const unsigned int head_idx = blockIdx.x;
    const unsigned int seq_block = blockIdx.y;
    const unsigned int batch = blockIdx.z;
    const unsigned int tid = threadIdx.x;

    const bool is_q = (head_idx < num_q_heads);
    const unsigned int head = is_q ? head_idx : (head_idx - num_q_heads);
    const unsigned int num_heads = is_q ? num_q_heads : num_kv_heads;

    if (!is_q && head >= num_kv_heads) return;

    const unsigned int pairs_per_pos = rotary_dim / 2;
    const unsigned int pos_per_block = 128 / pairs_per_pos;

    const unsigned int local_pos = tid / pairs_per_pos;
    const unsigned int pair_idx = tid % pairs_per_pos;

    const unsigned int seq_pos = seq_block * pos_per_block + local_pos;
    if (seq_pos >= seq_len) return;

    const unsigned int abs_pos = positions[batch * seq_len + seq_pos];

    // Use pre-computed frequency from inv_freq table (YaRN/NTK-aware)
    const float freq = inv_freq[pair_idx];
    const float angle = (float)abs_pos * freq;
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle);

    const unsigned int stride = num_heads * head_dim;
    __nv_bfloat16* ptr;
    if (is_q) {
        ptr = Q + batch * seq_len * (num_q_heads * head_dim)
                + seq_pos * (num_q_heads * head_dim)
                + head * head_dim;
    } else {
        ptr = K + batch * seq_len * (num_kv_heads * head_dim)
                + seq_pos * (num_kv_heads * head_dim)
                + head * head_dim;
    }

    // Interleaved pair convention matching rope_forward above.
    // Mistral uses rope_interleave=True: weights store rope dims as
    // (d0,d1),(d2,d3),... so pair_idx i rotates (d_2i, d_2i+1).
    // NOT split-half (d_i, d_i+half_rot) — that convention is used by
    // neox-style models in the common kernels/gb10/common/rope.cu.
    const unsigned int d0 = 2 * pair_idx;
    const unsigned int d1 = 2 * pair_idx + 1;
    float x0 = (float)ptr[d0];
    float x1 = (float)ptr[d1];

    float y0 = x0 * cos_val - x1 * sin_val;
    float y1 = x1 * cos_val + x0 * sin_val;

    ptr[d0] = __float2bfloat16(y0);
    ptr[d1] = __float2bfloat16(y1);
}
