// SPDX-License-Identifier: AGPL-3.0-only

// Interleaved Multi-modal RoPE (MRoPE) for SM121.
//
// Applies the Qwen3.6 / Qwen3-VL rotary variant where three position ID
// streams (temporal, height, width) are interleaved across the rotary
// channels. Each rotary pair is owned by one of the three sections, chosen
// by `pair_idx % 3`:
//
//   pair_idx % 3 == 0 → temporal (N_t pairs at indices 0, 3, 6, …)
//   pair_idx % 3 == 1 → height   (N_h pairs at indices 1, 4, 7, …)
//   pair_idx % 3 == 2 → width    (N_w pairs at indices 2, 5, 8, …)
//
// `mrope_section = [N_t, N_h, N_w]` must satisfy `(N_t + N_h + N_w) * 2 ==
// rotary_dim`. Qwen3.6 uses [11, 11, 10] → 32 pairs → rotary_dim = 64.
//
// For each owned pair, the kernel looks up the section's own position ID
// and applies the standard rotate-half rotation using the shared inv_freq
// schedule `freq_i = 1 / theta^(2*i / rotary_dim)`. When all three position
// streams agree (text-only serving), the result is bit-identical to scalar
// RoPE — this lets us reuse the same kernel for text-only without a
// fast-path branch.
//
// Layout (matches `rope_forward`):
//   Q: [batch, seq_len, num_q_heads, head_dim]  BF16
//   K: [batch, seq_len, num_kv_heads, head_dim] BF16
//   pos_t, pos_h, pos_w: [batch, seq_len]  uint32 each
//
// Grid: (num_q_heads + num_kv_heads, ceil(seq_len / pos_per_block), batch)
// Block: (128, 1, 1)

#include <cuda_bf16.h>

extern "C" __global__ void rope_forward_mrope_interleaved(
    __nv_bfloat16* __restrict__ Q,
    __nv_bfloat16* __restrict__ K,
    const unsigned int* __restrict__ pos_t,
    const unsigned int* __restrict__ pos_h,
    const unsigned int* __restrict__ pos_w,
    const unsigned int seq_len,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int rotary_dim,
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

    const unsigned int pairs_per_pos = rotary_dim / 2;
    const unsigned int pos_per_block = 128 / pairs_per_pos;
    if (pos_per_block == 0) return;

    const unsigned int local_pos = tid / pairs_per_pos;
    const unsigned int pair_idx = tid % pairs_per_pos;

    const unsigned int seq_pos = seq_block * pos_per_block + local_pos;
    if (seq_pos >= seq_len) return;
    if (local_pos >= pos_per_block) return;

    // Select position ID based on which section owns this pair.
    const unsigned int section = pair_idx % 3;
    const unsigned int tok_idx = batch * seq_len + seq_pos;
    unsigned int abs_pos;
    if (section == 0) abs_pos = pos_t[tok_idx];
    else if (section == 1) abs_pos = pos_h[tok_idx];
    else                   abs_pos = pos_w[tok_idx];

    // Shared inverse-frequency schedule (FP64 powf to avoid drift).
    const double freq_exp_d = (double)(2 * pair_idx) / (double)rotary_dim;
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

    // rotate-half convention: pair (i, i + half_rot)
    const unsigned int half_rot = rotary_dim / 2;
    const unsigned int d0 = pair_idx;
    const unsigned int d1 = pair_idx + half_rot;
    const float x0 = (float)ptr[d0];
    const float x1 = (float)ptr[d1];

    const float y0 = x0 * cos_val - x1 * sin_val;
    const float y1 = x1 * cos_val + x0 * sin_val;

    ptr[d0] = __float2bfloat16(y0);
    ptr[d1] = __float2bfloat16(y1);
}
