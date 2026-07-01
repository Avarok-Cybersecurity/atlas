// SPDX-License-Identifier: AGPL-3.0-only

// Simple MLA Prefill Attention for absorbed MLA (HDIM=576, GQA 64:1).
//
// No tensor core MMA — uses scalar BF16 dot products with FP32 accumulation.
// For typical prefill lengths (16-30 tokens), this is memory-bound and sufficient.
// Avoids the SM121 PTX JIT issue with the tensor-core inferspark_prefill at HDIM=320.
//
// Q: [batch, seq_len, num_q_heads, 576]
// K: [batch, seq_len, 1, 576] (single KV head, broadcast to all Q heads)
// V: [batch, seq_len, 1, 576]
// O: [batch, seq_len, num_q_heads, 576]
//
// Grid: (num_q_heads, ceil(seq_len/BR), batch)
// Block: (256, 1, 1)

#include <cuda_bf16.h>
#include <float.h>

#define MLA_HDIM 576
#define MLA_BR 16     // query tile size (smaller than 32 to reduce shared mem)
#define MLA_BC 16     // KV tile size

// NOTE: function name retains "320" for Rust symbol compatibility,
// but MLA_HDIM=576 makes this the HDIM=576 variant for DeepSeek-V4.
extern "C" __global__ void mla_prefill_attn_320(
    const __nv_bfloat16* __restrict__ Q,    // [batch, seq_len, num_q_heads, MLA_HDIM]
    const __nv_bfloat16* __restrict__ K,    // [batch, seq_len, 1, MLA_HDIM]
    const __nv_bfloat16* __restrict__ V,    // [batch, seq_len, 1, MLA_HDIM]
    __nv_bfloat16* __restrict__ O,           // [batch, seq_len, num_q_heads, MLA_HDIM]
    unsigned int seq_len,
    unsigned int num_q_heads,
    unsigned int num_kv_heads,
    unsigned int head_dim,                   // should be 576
    float inv_sqrt_d,
    unsigned int causal
) {
    const unsigned int q_head = blockIdx.x;
    const unsigned int q_block = blockIdx.y;
    const unsigned int batch = blockIdx.z;
    const unsigned int tid = threadIdx.x;

    if (q_head >= num_q_heads) return;

    const unsigned int q_start = q_block * MLA_BR;
    if (q_start >= seq_len) return;
    const unsigned int q_end = min(q_start + MLA_BR, seq_len);

    const unsigned int gqa_ratio = num_q_heads / max(num_kv_heads, 1u);
    const unsigned int kv_head = q_head / gqa_ratio;

    const unsigned int q_stride = num_q_heads * head_dim;
    const unsigned int kv_stride = num_kv_heads * head_dim;

    const __nv_bfloat16* Q_base = Q + (unsigned long long)batch * seq_len * q_stride;
    const __nv_bfloat16* K_base = K + (unsigned long long)batch * seq_len * kv_stride;
    const __nv_bfloat16* V_base = V + (unsigned long long)batch * seq_len * kv_stride;
    __nv_bfloat16* O_base = O + (unsigned long long)batch * seq_len * q_stride;

    // Each thread handles one query row within the tile.
    // 256 threads / 16 lanes per row = 16 query rows.
    // IMPORTANT: 16 lanes per row means 2 "sub-warps" per 32-thread warp.
    // Use full warp sync mask (0xFFFFFFFF) and restrict reduction to 16 lanes
    // by only doing offsets 1,2,4,8 (not 16).
    const unsigned int q_row = tid / 16;  // which query in the tile (0..15)
    const unsigned int lane = tid % 16;   // lane within query processing (0..15)
    const unsigned int warp_lane = tid % 32; // position within the 32-thread warp

    if (q_row >= (q_end - q_start)) return;

    const unsigned int q_pos = q_start + q_row;
    const __nv_bfloat16* Q_row = Q_base + (unsigned long long)q_pos * q_stride + q_head * head_dim;

    // Online softmax state
    float m_prev = -FLT_MAX;
    float l_prev = 0.0f;
    float acc_o[36];  // MLA_HDIM / 16 = 36 output elements per lane
    for (int i = 0; i < 36; i++) acc_o[i] = 0.0f;

    // Iterate over KV blocks
    unsigned int kv_end = causal ? min(q_pos + 1, seq_len) : seq_len;
    for (unsigned int kv_start = 0; kv_start < kv_end; kv_start += MLA_BC) {
        unsigned int kv_block_end = min(kv_start + MLA_BC, kv_end);

        for (unsigned int kv_pos = kv_start; kv_pos < kv_block_end; kv_pos++) {
            // Compute Q[q_pos] · K[kv_pos]^T
            const __nv_bfloat16* K_row = K_base + (unsigned long long)kv_pos * kv_stride + kv_head * head_dim;

            // Parallel dot product: each lane handles 36 elements (576/16=36)
            float dot = 0.0f;
            for (unsigned int d = lane * 36; d < min((lane + 1) * 36, head_dim); d++) {
                float q_val = __bfloat162float(Q_row[d]);
                float k_val = __bfloat162float(K_row[d]);
                dot += q_val * k_val;
            }
            // Warp reduce within 16 lanes (half a 32-thread warp).
            // Use full warp mask (0xFFFFFFFF) to avoid UB from partial mask.
            // Only reduce within 16-lane group by using offsets 1,2,4,8.
            for (int offset = 8; offset > 0; offset >>= 1) {
                dot += __shfl_down_sync(0xFFFFFFFF, dot, offset);
            }
            // Lane 0 of each 16-lane group has the reduction result.
            // For warp_lane < 16: lane 0 has the result.
            // For warp_lane >= 16: lane 16 has the result.
            float score = dot * inv_sqrt_d;

            // Causal mask
            if (causal && kv_pos > q_pos) score = -FLT_MAX;

            // Broadcast score from lane 0 of each 16-lane group
            // warp_lane % 16 == 0 has the correct value
            score = __shfl_sync(0xFFFFFFFF, score, (warp_lane / 16) * 16);

            // Online softmax update (all lanes compute uniformly)
            float m_new = fmaxf(m_prev, score);
            float alpha = expf(m_prev - m_new);
            float p = expf(score - m_new);
            float l_new = alpha * l_prev + p;

            // Update output accumulator: O = alpha * O + p * V[kv_pos]
            const __nv_bfloat16* V_row = V_base + (unsigned long long)kv_pos * kv_stride + kv_head * head_dim;
            for (int i = 0; i < 36; i++) {
                unsigned int d = lane * 36 + i;
                if (d < head_dim) {
                    float v_val = __bfloat162float(V_row[d]);
                    acc_o[i] = alpha * acc_o[i] + p * v_val;
                }
            }
            m_prev = m_new;
            l_prev = l_new;
        }
    }

    // Normalize by softmax denominator and write output
    float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
    __nv_bfloat16* O_row = O_base + (unsigned long long)q_pos * q_stride + q_head * head_dim;
    for (int i = 0; i < 36; i++) {
        unsigned int d = lane * 36 + i;
        if (d < head_dim) {
            O_row[d] = __float2bfloat16(acc_o[i] * inv_l);
        }
    }
}
