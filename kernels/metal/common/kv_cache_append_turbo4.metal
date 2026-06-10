// SPDX-License-Identifier: AGPL-3.0-only
//
// Append a single token's K and V projections into a contiguous
// Turbo4 KV cache at slot `cache_pos` — Metal counterpart of the CUDA
// `reshape_and_cache_flash_turbo4` write path, for the contiguous
// (non-paged) Metal cache.
//
// Turbo4 storage: 4-bit Lloyd-Max codebook indices (2 elems/byte,
// low nibble = even element) + FP8 E4M3 group scales (group of 16).
// Per-group matched-norm L2 scale: after indexing into the codebook,
// the raw amax scale is replaced with ||original|| / ||centroid_vec||
// so the dequantized group keeps the input's L2 norm (compensates the
// systematic shrinkage of rounding-to-centroid).
//
// The caller rotates K/V with `wht_bf16_inplace` BEFORE this kernel;
// the codebook is Lloyd-Max-optimal for the Gaussianized post-WHT
// distribution.
//
// Layout:
//   new_k    : bfloat [num_kv_heads, head_dim]   (WHT-rotated)
//   new_v    : bfloat [num_kv_heads, head_dim]   (WHT-rotated)
//   k_data   : uchar  [max_seq, num_kv_heads * head_dim / 2]
//   v_data   : uchar  [max_seq, num_kv_heads * head_dim / 2]
//   k_scales : uchar  [max_seq, num_kv_heads * head_dim / 16]  (E4M3)
//   v_scales : uchar  [max_seq, num_kv_heads * head_dim / 16]  (E4M3)
//
// Grid: (num_groups, 1, 1) threads — one thread per group of 16.

#include <metal_stdlib>
using namespace metal;

constant uint TQ4_GROUP_SIZE = 16;
constant float TURBO4_MAX = 2.7326f;
constant float FP8_E4M3_MAX = 448.0f;

// 16-level Lloyd-Max codebook for N(0,1) + decision boundaries.
constant float TURBO4_CODEBOOK[16] = {
    -2.7326f, -2.0690f, -1.6180f, -1.2562f, -0.9423f, -0.6568f, -0.3880f, -0.1284f,
     0.1284f,  0.3880f,  0.6568f,  0.9423f,  1.2562f,  1.6180f,  2.0690f,  2.7326f
};
constant float TURBO4_BOUNDS[15] = {
    -2.4008f, -1.8435f, -1.4371f, -1.0993f, -0.7996f, -0.5224f, -0.2582f, 0.0f,
     0.2582f,  0.5224f,  0.7996f,  1.0993f,  1.4371f,  1.8435f,  2.4008f
};

static inline uchar turbo4_quantize(float x) {
    uchar idx = 0;
    while (idx < 15 && x >= TURBO4_BOUNDS[idx]) {
        idx++;
    }
    return idx;
}

// float → FP8 E4M3 byte (saturating; round-half-away on the mantissa —
// same convention as kv_cache_append_turbo8.metal).
static inline uchar f32_to_e4m3(float f) {
    uchar sign = f < 0.0f ? 0x80 : 0x00;
    float a = fabs(f);
    if (a >= FP8_E4M3_MAX) return sign | 0x7E;
    if (a < 0.001953125f) {
        uint m = uint(round(a * 512.0f));
        return sign | uchar(m);
    }
    int e = int(floor(log2(a)));
    if (e < -6) e = -6;
    float man = a / exp2(float(e));
    uint m3 = uint(round((man - 1.0f) * 8.0f));
    if (m3 == 8) { e += 1; m3 = 0; }
    return sign | uchar((e + 7) << 3) | uchar(m3);
}

static inline float e4m3_to_f32(uchar b) {
    float sign = (b & 0x80) ? -1.0f : 1.0f;
    uint e = (b >> 3) & 0xF;
    uint m = b & 7;
    if (e == 0) {
        return sign * float(m) * 0.001953125f;
    }
    return sign * (1.0f + float(m) * 0.125f) * exp2(float(int(e) - 7));
}

kernel void kv_cache_append_turbo4(
    constant uint &num_kv_heads [[buffer(0)]],
    constant uint &head_dim     [[buffer(1)]],
    constant uint &cache_pos    [[buffer(2)]],
    device const bfloat *new_k  [[buffer(3)]],
    device const bfloat *new_v  [[buffer(4)]],
    device uchar  *k_data       [[buffer(5)]],
    device uchar  *v_data       [[buffer(6)]],
    device uchar  *k_scales     [[buffer(7)]],
    device uchar  *v_scales     [[buffer(8)]],
    uint g [[thread_position_in_grid]])
{
    uint n_elems = num_kv_heads * head_dim;
    uint num_groups = n_elems / TQ4_GROUP_SIZE;
    if (g >= num_groups) {
        return;
    }
    uint elem_off = g * TQ4_GROUP_SIZE;

    float kf[TQ4_GROUP_SIZE], vf[TQ4_GROUP_SIZE];
    float k_norm_sq = 0.0f, v_norm_sq = 0.0f;
    float k_max = 0.0f, v_max = 0.0f;
    for (uint i = 0; i < TQ4_GROUP_SIZE; ++i) {
        kf[i] = float(new_k[elem_off + i]);
        vf[i] = float(new_v[elem_off + i]);
        k_norm_sq += kf[i] * kf[i];
        v_norm_sq += vf[i] * vf[i];
        k_max = max(k_max, fabs(kf[i]));
        v_max = max(v_max, fabs(vf[i]));
    }

    float k_inv = (k_max > 1e-12f) ? (TURBO4_MAX / k_max) : 1.0f;
    float v_inv = (v_max > 1e-12f) ? (TURBO4_MAX / v_max) : 1.0f;

    uchar k_idx[TQ4_GROUP_SIZE], v_idx[TQ4_GROUP_SIZE];
    float k_recon_sq = 0.0f, v_recon_sq = 0.0f;
    for (uint i = 0; i < TQ4_GROUP_SIZE; ++i) {
        k_idx[i] = turbo4_quantize(kf[i] * k_inv);
        v_idx[i] = turbo4_quantize(vf[i] * v_inv);
        float kc = TURBO4_CODEBOOK[k_idx[i]];
        float vc = TURBO4_CODEBOOK[v_idx[i]];
        k_recon_sq += kc * kc;
        v_recon_sq += vc * vc;
    }
    float k_recon_norm = sqrt(k_recon_sq);
    float v_recon_norm = sqrt(v_recon_sq);

    // Matched-norm scale, amax fallback on degenerate groups.
    float ks = (k_recon_norm > 1e-10f) ? (sqrt(k_norm_sq) / k_recon_norm)
                                       : (k_max / TURBO4_MAX);
    float vs = (v_recon_norm > 1e-10f) ? (sqrt(v_norm_sq) / v_recon_norm)
                                       : (v_max / TURBO4_MAX);
    ks = min(ks, FP8_E4M3_MAX);
    vs = min(vs, FP8_E4M3_MAX);

    uint scale_row = cache_pos * num_groups;
    k_scales[scale_row + g] = f32_to_e4m3(ks);
    v_scales[scale_row + g] = f32_to_e4m3(vs);

    uint data_row = cache_pos * (n_elems / 2);
    for (uint i = 0; i < TQ4_GROUP_SIZE; i += 2) {
        k_data[data_row + (elem_off + i) / 2] = k_idx[i] | (k_idx[i + 1] << 4);
        v_data[data_row + (elem_off + i) / 2] = v_idx[i] | (v_idx[i + 1] << 4);
    }
}
