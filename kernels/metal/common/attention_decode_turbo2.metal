// SPDX-License-Identifier: AGPL-3.0-only
//
// Decode-path SDPA against a contiguous Turbo2 KV cache (2-bit
// Lloyd-Max indices, 4 elems/byte + FP8 E4M3 group scales,
// WHT-rotated basis). Caller bookends: WHT(Q) before, iWHT(out) after.
//
// Stage 5 applies the sparse-V gate: V dequant + accumulation are
// skipped for positions whose unnormalized softmax weight
// exp(score - max) falls below 1e-3 — the attention distribution is
// known before V is touched, and at long context most weights are
// negligible (see docs: attention-gated value dequantization).
//
// Layout:
//   q        : bfloat [num_heads, head_dim]       (one token, rotated)
//   k_data   : uchar  [seq_len, num_kv_heads * head_dim / 4]
//   v_data   : uchar  [seq_len, num_kv_heads * head_dim / 4]
//   k_scales : uchar  [seq_len, num_kv_heads * head_dim / 16]  (E4M3)
//   v_scales : uchar  [seq_len, num_kv_heads * head_dim / 16]  (E4M3)
//   out      : bfloat [num_heads, head_dim]       (rotated-V basis)

#include <metal_stdlib>
using namespace metal;

constant uint MAX_SEQ_DECODE_TQ2 = 4096;
constant uint TQ2_GROUP_SIZE = 16;

constant float TURBO2_CODEBOOK[4] = { -1.5104f, -0.4528f, 0.4528f, 1.5104f };

static inline float e4m3_to_f32(uchar b) {
    float sign = (b & 0x80) ? -1.0f : 1.0f;
    uint e = (b >> 3) & 0xF;
    uint m = b & 7;
    if (e == 0) {
        return sign * float(m) * 0.001953125f;
    }
    return sign * (1.0f + float(m) * 0.125f) * exp2(float(int(e) - 7));
}

kernel void attention_decode_turbo2(
    constant uint  &seq_len      [[buffer(0)]],
    constant uint  &num_heads    [[buffer(1)]],
    constant uint  &num_kv_heads [[buffer(2)]],
    constant uint  &head_dim     [[buffer(3)]],
    constant float &scale        [[buffer(4)]],
    // Sparse-V gate: V rows with exp(score - max) <= sparse_v_threshold
    // skip dequant + accumulation. 0.0 disables the gate.
    constant float &sparse_v_threshold [[buffer(5)]],
    device const bfloat *q       [[buffer(6)]],
    device const uchar  *k_data  [[buffer(7)]],
    device const uchar  *v_data  [[buffer(8)]],
    device const uchar  *k_scales [[buffer(9)]],
    device const uchar  *v_scales [[buffer(10)]],
    device bfloat       *out     [[buffer(11)]],
    uint h       [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup float scores[MAX_SEQ_DECODE_TQ2];
    threadgroup float max_score;
    threadgroup float sum_exp;

    if (h >= num_heads) {
        return;
    }
    // The score vector lives in threadgroup memory: positions past the
    // cap would read/write out of bounds in stages 2-5, so clamp hard.
    // Long-context decode belongs to a future paged variant.
    uint seq = min(seq_len, MAX_SEQ_DECODE_TQ2);
    uint group = num_heads / num_kv_heads;
    uint kv_h  = h / group;
    uint n_elems = num_kv_heads * head_dim;
    uint row_bytes = n_elems / 4;
    uint num_groups = n_elems / TQ2_GROUP_SIZE;

    // Stage 1: scores[s] = (Q[h] · dequant(K[s, kv_h])) * scale.
    for (uint s = tid; s < seq; s += tg_size) {
        device const uchar *k_row = k_data + (ulong)s * row_bytes + kv_h * head_dim / 4;
        device const uchar *k_srow =
            k_scales + (ulong)s * num_groups + kv_h * head_dim / TQ2_GROUP_SIZE;
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; d += TQ2_GROUP_SIZE) {
            float gs = e4m3_to_f32(k_srow[d / TQ2_GROUP_SIZE]);
            for (uint i = 0; i < TQ2_GROUP_SIZE; i += 4) {
                uchar packed = k_row[(d + i) / 4];
                for (uint j = 0; j < 4; ++j) {
                    float qv = float(q[h * head_dim + d + i + j]);
                    dot += qv * TURBO2_CODEBOOK[(packed >> (2 * j)) & 3] * gs;
                }
            }
        }
        scores[s] = dot * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage 2: max reduction.
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

    // Stage 3: exp(score - max).
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

    // Stage 5: out[h, d] = sum_s(softmax_s * dequant(V[s, kv_h, d])),
    // skipping rows below the sparse-V threshold.
    float inv_sum = 1.0f / sum_exp;
    for (uint d = tid; d < head_dim; d += tg_size) {
        uint sg = (kv_h * head_dim + d) / TQ2_GROUP_SIZE;
        uint byte_idx = (kv_h * head_dim + d) / 4;
        uint shift = 2 * (d & 3);
        float acc = 0.0f;
        for (uint s = 0; s < seq; ++s) {
            if (scores[s] <= sparse_v_threshold) {
                continue;
            }
            uchar packed = v_data[(ulong)s * row_bytes + byte_idx];
            float vv = TURBO2_CODEBOOK[(packed >> shift) & 3]
                * e4m3_to_f32(v_scales[(ulong)s * num_groups + sg]);
            acc += scores[s] * inv_sum * vv;
        }
        out[h * head_dim + d] = bfloat(acc);
    }
}
