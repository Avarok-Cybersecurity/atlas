// SPDX-License-Identifier: AGPL-3.0-only
//
// Decode-path scaled-dot-product attention against a contiguous KV
// cache. One threadgroup per head; threads inside the group cooperate
// on the K-dot-product, the softmax, and the V-weighted sum.
//
//   scores[s] = (Q[h] · K[s, h_kv]) / sqrt(head_dim)
//   softmax over s
//   out[h, d] = sum_s(softmax_s * V[s, h_kv, d])
//
// Supports Grouped-Query Attention: `num_heads` queries map to
// `num_kv_heads` keys/values via integer division (`h / (num_heads /
// num_kv_heads)`).
//
// Layout:
//   q   : bfloat [num_heads,   head_dim]            (one token)
//   k   : bfloat [seq_len, num_kv_heads, head_dim]  (cache)
//   v   : bfloat [seq_len, num_kv_heads, head_dim]
//   out : bfloat [num_heads,   head_dim]
//
// `seq_len` is capped at MAX_SEQ_DECODE because the per-token score
// vector lives in threadgroup memory. Long-context decode goes
// through the paged variant (separate kernel, future PR).

#include <metal_stdlib>
using namespace metal;

constant uint MAX_SEQ_DECODE = 4096;

kernel void attention_decode(
    constant uint  &seq_len      [[buffer(0)]],
    constant uint  &num_heads    [[buffer(1)]],
    constant uint  &num_kv_heads [[buffer(2)]],
    constant uint  &head_dim     [[buffer(3)]],
    constant float &scale        [[buffer(4)]],
    device const bfloat *q       [[buffer(5)]],
    device const bfloat *k       [[buffer(6)]],
    device const bfloat *v       [[buffer(7)]],
    device bfloat       *out     [[buffer(8)]],
    uint h       [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup float scores[MAX_SEQ_DECODE];
    threadgroup float max_score;
    threadgroup float sum_exp;

    if (h >= num_heads) {
        return;
    }
    // The score vector lives in threadgroup memory: positions past the
    // cap would read/write out of bounds in stages 2-5, so clamp hard.
    // Long-context decode belongs to a future paged variant.
    uint seq = min(seq_len, MAX_SEQ_DECODE);
    uint group = num_heads / num_kv_heads;
    uint kv_h  = h / group;

    // Stage 1: scores[s] = (Q[h] · K[s, kv_h]) * scale.
    for (uint s = tid; s < seq; s += tg_size) {
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; ++d) {
            float qv = float(q[h * head_dim + d]);
            float kv = float(k[(s * num_kv_heads + kv_h) * head_dim + d]);
            dot += qv * kv;
        }
        scores[s] = dot * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage 2: max reduction (single-threaded for correctness on
    // exact scoring; small seq_len compared to head_dim so the
    // serial sweep doesn't dominate).
    if (tid == 0) {
        float m = -INFINITY;
        for (uint s = 0; s < seq; ++s) {
            if (scores[s] > m) {
                m = scores[s];
            }
        }
        max_score = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage 3: exp(score - max) in parallel.
    for (uint s = tid; s < seq; s += tg_size) {
        scores[s] = exp(scores[s] - max_score);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage 4: sum reduction.
    if (tid == 0) {
        float sum = 0.0f;
        for (uint s = 0; s < seq; ++s) {
            sum += scores[s];
        }
        sum_exp = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage 5: out[h, d] = sum_s(softmax_s * V[s, kv_h, d]).
    // Each thread handles a slice of d to amortize memory traffic.
    float inv_sum = 1.0f / sum_exp;
    for (uint d = tid; d < head_dim; d += tg_size) {
        float acc = 0.0f;
        for (uint s = 0; s < seq; ++s) {
            float vv = float(v[(s * num_kv_heads + kv_h) * head_dim + d]);
            acc += scores[s] * inv_sum * vv;
        }
        out[h * head_dim + d] = bfloat(acc);
    }
}
