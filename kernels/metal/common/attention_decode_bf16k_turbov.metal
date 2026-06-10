// SPDX-License-Identifier: AGPL-3.0-only
//
// Safer-asym decode SDPA: K read as raw bf16 (original basis — Q is
// NOT rotated for these dtypes), V dequantized from a TurboQuant
// format in the WHT-rotated basis (caller applies
// `wht_bf16_inplace_inv` to the output). One entry point per V dtype.
//
// Stage 5 applies the sparse-V gate (skip V dequant + accumulation
// when exp(score - max) <= sparse_v_threshold; 0 disables).
//
// Layout:
//   q        : bfloat [num_heads, head_dim]       (un-rotated)
//   k_cache  : bfloat [seq_len, num_kv_heads * head_dim]
//   v_data   : uchar  [seq_len, packed V bytes]
//   v_scales : uchar  [seq_len, num_kv_heads * head_dim / 16]  (E4M3)
//   out      : bfloat [num_heads, head_dim]       (rotated-V basis)

#include <metal_stdlib>
using namespace metal;

constant uint MAX_SEQ_DECODE_ASYM = 4096;
constant uint ASYM_GROUP_SIZE = 16;

constant float TURBO4_CODEBOOK[16] = {
    -2.7326f, -2.0690f, -1.6180f, -1.2562f, -0.9423f, -0.6568f, -0.3880f, -0.1284f,
     0.1284f,  0.3880f,  0.6568f,  0.9423f,  1.2562f,  1.6180f,  2.0690f,  2.7326f
};
constant float TURBO3_CODEBOOK[8] = {
    -2.1520f, -1.3440f, -0.7560f, -0.2451f, 0.2451f, 0.7560f, 1.3440f, 2.1520f
};
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

static inline uchar unpack3(device const uchar *b, uint i) {
    switch (i) {
        case 0: return b[0] & 7;
        case 1: return (b[0] >> 3) & 7;
        case 2: return ((b[0] >> 6) | (b[1] << 2)) & 7;
        case 3: return (b[1] >> 1) & 7;
        case 4: return (b[1] >> 4) & 7;
        case 5: return ((b[1] >> 7) | (b[2] << 1)) & 7;
        case 6: return (b[2] >> 2) & 7;
        default: return (b[2] >> 5) & 7;
    }
}

#define ASYM_ATTN_PARAMS \
    constant uint  &seq_len      [[buffer(0)]], \
    constant uint  &num_heads    [[buffer(1)]], \
    constant uint  &num_kv_heads [[buffer(2)]], \
    constant uint  &head_dim     [[buffer(3)]], \
    constant float &scale        [[buffer(4)]], \
    constant float &sparse_v_threshold [[buffer(5)]], \
    device const bfloat *q       [[buffer(6)]], \
    device const bfloat *k_cache [[buffer(7)]], \
    device const uchar  *v_data  [[buffer(8)]], \
    device const uchar  *v_scales [[buffer(9)]], \
    device bfloat       *out     [[buffer(10)]], \
    uint h       [[threadgroup_position_in_grid]], \
    uint tid     [[thread_position_in_threadgroup]], \
    uint tg_size [[threads_per_threadgroup]]

// Stages 1-4 are identical for all three V dtypes: bf16 K scores +
// softmax staging into the shared threadgroup arrays.
#define ASYM_ATTN_STAGES_1_TO_4 \
    threadgroup float scores[MAX_SEQ_DECODE_ASYM]; \
    threadgroup float max_score; \
    threadgroup float sum_exp; \
    if (h >= num_heads) { return; } \
    uint seq = min(seq_len, MAX_SEQ_DECODE_ASYM); \
    uint group = num_heads / num_kv_heads; \
    uint kv_h  = h / group; \
    uint n_elems = num_kv_heads * head_dim; \
    uint num_groups = n_elems / ASYM_GROUP_SIZE; \
    for (uint s = tid; s < seq; s += tg_size) { \
        device const bfloat *k_row = k_cache + (ulong)s * n_elems + kv_h * head_dim; \
        float dot = 0.0f; \
        for (uint d = 0; d < head_dim; ++d) { \
            dot += float(q[h * head_dim + d]) * float(k_row[d]); \
        } \
        scores[s] = dot * scale; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (tid == 0) { \
        float m = -INFINITY; \
        for (uint s = 0; s < seq; ++s) { \
            if (scores[s] > m) { m = scores[s]; } \
        } \
        max_score = m; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    for (uint s = tid; s < seq; s += tg_size) { \
        scores[s] = exp(scores[s] - max_score); \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    if (tid == 0) { \
        float sum = 0.0f; \
        for (uint s = 0; s < seq; ++s) { sum += scores[s]; } \
        sum_exp = sum; \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    float inv_sum = 1.0f / sum_exp;

kernel void attention_decode_bf16k_turbo4v(ASYM_ATTN_PARAMS) {
    ASYM_ATTN_STAGES_1_TO_4
    uint row_bytes = n_elems / 2;
    for (uint d = tid; d < head_dim; d += tg_size) {
        uint elem = kv_h * head_dim + d;
        uint sg = elem / ASYM_GROUP_SIZE;
        uint byte_idx = elem / 2;
        bool high = (d & 1) != 0;
        float acc = 0.0f;
        for (uint s = 0; s < seq; ++s) {
            if (scores[s] <= sparse_v_threshold) { continue; }
            uchar packed = v_data[(ulong)s * row_bytes + byte_idx];
            uchar idx = high ? (packed >> 4) : (packed & 0xF);
            acc += scores[s] * inv_sum * TURBO4_CODEBOOK[idx]
                * e4m3_to_f32(v_scales[(ulong)s * num_groups + sg]);
        }
        out[h * head_dim + d] = bfloat(acc);
    }
}

kernel void attention_decode_bf16k_turbo3v(ASYM_ATTN_PARAMS) {
    ASYM_ATTN_STAGES_1_TO_4
    uint row_bytes = n_elems * 3 / 8;
    for (uint d = tid; d < head_dim; d += tg_size) {
        uint elem = kv_h * head_dim + d;
        uint sg = elem / ASYM_GROUP_SIZE;
        uint pack_base = (elem / 8) * 3;
        uint in_pack = elem & 7;
        float acc = 0.0f;
        for (uint s = 0; s < seq; ++s) {
            if (scores[s] <= sparse_v_threshold) { continue; }
            device const uchar *p = v_data + (ulong)s * row_bytes + pack_base;
            acc += scores[s] * inv_sum * TURBO3_CODEBOOK[unpack3(p, in_pack)]
                * e4m3_to_f32(v_scales[(ulong)s * num_groups + sg]);
        }
        out[h * head_dim + d] = bfloat(acc);
    }
}

kernel void attention_decode_bf16k_turbo2v(ASYM_ATTN_PARAMS) {
    ASYM_ATTN_STAGES_1_TO_4
    uint row_bytes = n_elems / 4;
    for (uint d = tid; d < head_dim; d += tg_size) {
        uint elem = kv_h * head_dim + d;
        uint sg = elem / ASYM_GROUP_SIZE;
        uint byte_idx = elem / 4;
        uint shift = 2 * (d & 3);
        float acc = 0.0f;
        for (uint s = 0; s < seq; ++s) {
            if (scores[s] <= sparse_v_threshold) { continue; }
            uchar packed = v_data[(ulong)s * row_bytes + byte_idx];
            acc += scores[s] * inv_sum * TURBO2_CODEBOOK[(packed >> shift) & 3]
                * e4m3_to_f32(v_scales[(ulong)s * num_groups + sg]);
        }
        out[h * head_dim + d] = bfloat(acc);
    }
}
