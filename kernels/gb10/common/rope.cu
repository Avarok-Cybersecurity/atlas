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
    const float theta                        // Base frequency (e.g., 10000000.0)
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

    // freq depends ONLY on pair_idx, but the block covers pos_per_block positions, so
    // every freq was recomputed once per thread -- pos_per_block redundant FP64 pow()s.
    // FP64 runs at 1/64 rate on SM121, which left this kernel ~13x above its bandwidth
    // floor (1.16 ms for 4.4M elements). Evaluate each distinct freq once and share it.
    // Bit-identical (same FP64 pow) and it must happen BEFORE the divergent return
    // below, or the __syncthreads() is unreachable for the exited threads.
    __shared__ float s_freq[128];
    if (tid < pairs_per_pos) {
        const double fe = (double)(2 * tid) / (double)rotary_dim;
        s_freq[tid] = (float)(1.0 / pow((double)theta, fe));
    }
    __syncthreads();

    const unsigned int seq_pos = seq_block * pos_per_block + local_pos;
    if (seq_pos >= seq_len) return;

    // Get absolute position for this token
    const unsigned int abs_pos = positions[batch * seq_len + seq_pos];

    // Compute frequency for this pair in FP64 to prevent precision loss at high positions.
    // FP32 powf() has ~1e-6 relative error; at position 30K this causes ~0.03 rad drift.
    // freq_i = 1.0 / theta^(2*pair_idx / rotary_dim)
    const float freq = s_freq[pair_idx];
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

    // Load the pair (rotate_half convention: pair i with i + half_rot)
    const unsigned int half_rot = rotary_dim / 2;
    const unsigned int d0 = pair_idx;              // First half: 0..31
    const unsigned int d1 = pair_idx + half_rot;   // Second half: 32..63
    float x0 = (float)ptr[d0];
    float x1 = (float)ptr[d1];

    // Apply rotation
    float y0 = x0 * cos_val - x1 * sin_val;
    float y1 = x1 * cos_val + x0 * sin_val;

    // Store back
    ptr[d0] = __float2bfloat16(y0);
    ptr[d1] = __float2bfloat16(y1);
}

// ═══════════════════════════════════════════════════════════════════
// Proportional RoPE variant (Gemma-4 full-attention layers).
//
// Gemma-4's full-attention layers (rope_type="proportional",
// partial_rotary_factor=0.25, head_dim=512) use a different layout than
// standard partial RoPE. HF:
//   rope_angles = int(0.25 * head_dim / 2)                # 64
//   inv_freq = 1 / theta^(2i / head_dim) for i in [0, rope_angles)
//   inv_freq zero-padded to head_dim/2, then cat'd to head_dim length.
//   Rotation pairs: (i, i + head_dim/2) for i in [0, head_dim/2).
// Only the first `rope_angles` pairs get a non-zero frequency, the rest
// are passthrough.
//
// Atlas's `rope_forward` pairs (i, i + rotary_dim/2) and uses the wrong
// frequency denominator (`rotary_dim` instead of `head_dim`) — correct
// for Qwen3-style partial RoPE but incorrect for Gemma-4 proportional.
//
// This kernel implements the proportional layout directly:
//   - Each thread rotates one (i, i + head_dim/2) pair where i < rope_angles
//   - freq = 1 / theta^(2i / head_dim)
//
// Grid: (num_q_heads + num_kv_heads, seq_blocks, batch)
// Block: (128, 1, 1)
// ═══════════════════════════════════════════════════════════════════
extern "C" __global__ void rope_forward_proportional(
    __nv_bfloat16* __restrict__ Q,
    __nv_bfloat16* __restrict__ K,
    const unsigned int* __restrict__ positions,
    const unsigned int seq_len,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int rope_angles,  // Number of rotation pairs (e.g. 64)
    const float theta
) {
    const unsigned int head_idx = blockIdx.x;
    const unsigned int seq_block = blockIdx.y;
    const unsigned int batch = blockIdx.z;
    const unsigned int tid = threadIdx.x;

    const bool is_q = (head_idx < num_q_heads);
    const unsigned int head = is_q ? head_idx : (head_idx - num_q_heads);
    const unsigned int num_heads = is_q ? num_q_heads : num_kv_heads;

    if (!is_q && head >= num_kv_heads) return;

    // Each block handles `rope_angles` pairs per position.
    // For rope_angles=64, pairs_per_pos=64, pos_per_block=2 (128/64).
    const unsigned int pairs_per_pos = rope_angles;
    const unsigned int pos_per_block = (128 / pairs_per_pos) > 0 ? (128 / pairs_per_pos) : 1;

    const unsigned int local_pos = tid / pairs_per_pos;
    const unsigned int pair_idx = tid % pairs_per_pos;

    const unsigned int seq_pos = seq_block * pos_per_block + local_pos;
    if (seq_pos >= seq_len) return;
    if (local_pos >= pos_per_block) return;
    if (pair_idx >= rope_angles) return;

    const unsigned int abs_pos = positions[batch * seq_len + seq_pos];

    // freq = 1 / theta^(2 * pair_idx / head_dim)  (proportional: denom=head_dim)
    const double freq_exp_d = (double)(2 * pair_idx) / (double)head_dim;
    const float freq = (float)(1.0 / pow((double)theta, freq_exp_d));
    const float angle = (float)abs_pos * freq;
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle);

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

    // Proportional layout: pair (d0, d1) = (i, i + head_dim/2).
    const unsigned int half_hd = head_dim / 2;
    const unsigned int d0 = pair_idx;
    const unsigned int d1 = pair_idx + half_hd;
    float x0 = (float)ptr[d0];
    float x1 = (float)ptr[d1];

    float y0 = x0 * cos_val - x1 * sin_val;
    float y1 = x1 * cos_val + x0 * sin_val;

    ptr[d0] = __float2bfloat16(y0);
    ptr[d1] = __float2bfloat16(y1);
}

// ═══════════════════════════════════════════════════════════════════
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

    const unsigned int half_rot = rotary_dim / 2;
    const unsigned int d0 = pair_idx;
    const unsigned int d1 = pair_idx + half_rot;
    float x0 = (float)ptr[d0];
    float x1 = (float)ptr[d1];

    float y0 = x0 * cos_val - x1 * sin_val;
    float y1 = x1 * cos_val + x0 * sin_val;

    ptr[d0] = __float2bfloat16(y0);
    ptr[d1] = __float2bfloat16(y1);
}

// rope_forward_yarn_interleaved — GPT-J / is_neox_style=False variant.
//
// DeepSeek MLA (V2/V3/V4) stores the rope channels in INTERLEAVED layout:
// the rotation pairs are adjacent elements (2i, 2i+1), not (i, i+half).
// The HF reference de-interleaves via `q.view(d//2, 2).transpose(...)`
// before applying the standard rotate_half; rotating adjacent pairs in
// place is the algebraic equivalent on the original stored layout.
//
// Frequency mapping is identical to rope_forward_yarn: pair `pair_idx`
// uses inv_freq[pair_idx]. Only the channel pairing differs.
//
// Grid: (num_q_heads + num_kv_heads, seq_blocks, batch)
// Block: (128, 1, 1) — same as rope_forward_yarn
// ═══════════════════════════════════════════════════════════════════
extern "C" __global__ void rope_forward_yarn_interleaved(
    __nv_bfloat16* __restrict__ Q,
    __nv_bfloat16* __restrict__ K,
    const unsigned int* __restrict__ positions,
    const unsigned int seq_len,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int rotary_dim,
    const float* __restrict__ inv_freq,       // [rotary_dim/2] pre-computed frequencies
    const float mscale                         // YaRN attention-temperature _mscale (folded into cos/sin)
) {
    const unsigned int head_idx = blockIdx.x;
    const unsigned int seq_block = blockIdx.y;
    const unsigned int batch = blockIdx.z;
    const unsigned int tid = threadIdx.x;

    const bool is_q = (head_idx < num_q_heads);
    const unsigned int head = is_q ? head_idx : (head_idx - num_q_heads);
    const unsigned int num_heads = is_q ? num_q_heads : num_kv_heads;
    (void)num_heads;

    if (!is_q && head >= num_kv_heads) return;

    const unsigned int pairs_per_pos = rotary_dim / 2;
    const unsigned int pos_per_block = 128 / pairs_per_pos;

    const unsigned int local_pos = tid / pairs_per_pos;
    const unsigned int pair_idx = tid % pairs_per_pos;

    const unsigned int seq_pos = seq_block * pos_per_block + local_pos;
    if (seq_pos >= seq_len) return;

    const unsigned int abs_pos = positions[batch * seq_len + seq_pos];

    const float freq = inv_freq[pair_idx];
    const float angle = (float)abs_pos * freq;
    // DeepSeek YaRN folds _mscale into cos/sin: cos_cached = cos(angle)*_mscale.
    const float cos_val = cosf(angle) * mscale;
    const float sin_val = sinf(angle) * mscale;

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

    // Interleaved pairing: adjacent elements (2*pair_idx, 2*pair_idx + 1).
    const unsigned int d0 = 2 * pair_idx;
    const unsigned int d1 = 2 * pair_idx + 1;
    float x0 = (float)ptr[d0];
    float x1 = (float)ptr[d1];

    float y0 = x0 * cos_val - x1 * sin_val;
    float y1 = x1 * cos_val + x0 * sin_val;

    ptr[d0] = __float2bfloat16(y0);
    ptr[d1] = __float2bfloat16(y1);
}

// rope_forward_yarn_interleaved_inv — CONJUGATE (negated-sin) interleaved YaRN.
// DeepSeek-V4 de-rotates the attention OUTPUT by the query position
// (eq.26: apply_rotary(attn_output, cos, -sin)) so each value's contribution
// becomes relative-distance. Same launch signature as the forward kernel, so it
// reuses the rope_yarn wrapper with a separate kernel handle.
extern "C" __global__ void rope_forward_yarn_interleaved_inv(
    __nv_bfloat16* __restrict__ Q,
    __nv_bfloat16* __restrict__ K,
    const unsigned int* __restrict__ positions,
    const unsigned int seq_len,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int rotary_dim,
    const float* __restrict__ inv_freq,
    const float mscale
) {
    const unsigned int head_idx = blockIdx.x;
    const unsigned int seq_block = blockIdx.y;
    const unsigned int batch = blockIdx.z;
    const unsigned int tid = threadIdx.x;

    const bool is_q = (head_idx < num_q_heads);
    const unsigned int head = is_q ? head_idx : (head_idx - num_q_heads);
    if (!is_q && head >= num_kv_heads) return;

    const unsigned int pairs_per_pos = rotary_dim / 2;
    const unsigned int pos_per_block = 128 / pairs_per_pos;
    const unsigned int local_pos = tid / pairs_per_pos;
    const unsigned int pair_idx = tid % pairs_per_pos;
    const unsigned int seq_pos = seq_block * pos_per_block + local_pos;
    if (seq_pos >= seq_len) return;

    const unsigned int abs_pos = positions[batch * seq_len + seq_pos];
    const float freq = inv_freq[pair_idx];
    const float angle = (float)abs_pos * freq;
    const float cos_val = cosf(angle) * mscale;
    const float sin_val = sinf(angle) * mscale;

    __nv_bfloat16* ptr;
    if (is_q) {
        ptr = Q + batch * seq_len * (num_q_heads * head_dim)
                + seq_pos * (num_q_heads * head_dim) + head * head_dim;
    } else {
        ptr = K + batch * seq_len * (num_kv_heads * head_dim)
                + seq_pos * (num_kv_heads * head_dim) + head * head_dim;
    }

    const unsigned int d0 = 2 * pair_idx;
    const unsigned int d1 = 2 * pair_idx + 1;
    float x0 = (float)ptr[d0];
    float x1 = (float)ptr[d1];

    // Conjugate rotation (negated sin) = inverse of the forward rotation.
    float y0 = x0 * cos_val + x1 * sin_val;
    float y1 = x1 * cos_val - x0 * sin_val;

    ptr[d0] = __float2bfloat16(y0);
    ptr[d1] = __float2bfloat16(y1);
}
