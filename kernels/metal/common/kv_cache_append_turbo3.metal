// SPDX-License-Identifier: AGPL-3.0-only
//
// Append a single token's K and V into a contiguous Turbo3 KV cache —
// Metal counterpart of the CUDA `reshape_and_cache_flash_turbo3` write
// path. 3-bit Lloyd-Max codebook indices packed 8 values → 3 bytes
// + FP8 E4M3 group scales with the matched-norm L2 correction.
// Caller rotates K/V with `wht_bf16_inplace` before this kernel.
//
// Layout:
//   new_k    : bfloat [num_kv_heads, head_dim]   (WHT-rotated)
//   new_v    : bfloat [num_kv_heads, head_dim]   (WHT-rotated)
//   k_data   : uchar  [max_seq, num_kv_heads * head_dim * 3 / 8]
//   v_data   : uchar  [max_seq, num_kv_heads * head_dim * 3 / 8]
//   k_scales : uchar  [max_seq, num_kv_heads * head_dim / 16]  (E4M3)
//   v_scales : uchar  [max_seq, num_kv_heads * head_dim / 16]  (E4M3)
//
// Grid: (num_groups, 1, 1) threads — one thread per group of 16.

#include <metal_stdlib>
using namespace metal;

constant uint TQ3_GROUP_SIZE = 16;
constant float TURBO3_MAX = 2.1520f;
constant float FP8_E4M3_MAX = 448.0f;

// 8-level Lloyd-Max codebook for N(0,1) + decision boundaries.
constant float TURBO3_CODEBOOK[8] = {
    -2.1520f, -1.3440f, -0.7560f, -0.2451f, 0.2451f, 0.7560f, 1.3440f, 2.1520f
};
constant float TURBO3_BOUNDS[7] = {
    -1.748f, -1.050f, -0.501f, 0.0f, 0.501f, 1.050f, 1.748f
};

static inline uchar turbo3_quantize(float x) {
    uchar idx = 0;
    while (idx < 7 && x >= TURBO3_BOUNDS[idx]) {
        idx++;
    }
    return idx;
}

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

// Pack 8 3-bit indices into 3 bytes (same layout as the CUDA kernel).
static inline void pack8x3(thread const uchar *idx, device uchar *out) {
    out[0] = idx[0] | (idx[1] << 3) | (idx[2] << 6);
    out[1] = (idx[2] >> 2) | (idx[3] << 1) | (idx[4] << 4) | (idx[5] << 7);
    out[2] = (idx[5] >> 1) | (idx[6] << 2) | (idx[7] << 5);
}

kernel void kv_cache_append_turbo3(
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
    uint num_groups = n_elems / TQ3_GROUP_SIZE;
    if (g >= num_groups) {
        return;
    }
    uint elem_off = g * TQ3_GROUP_SIZE;

    float kf[TQ3_GROUP_SIZE], vf[TQ3_GROUP_SIZE];
    float k_norm_sq = 0.0f, v_norm_sq = 0.0f;
    float k_max = 0.0f, v_max = 0.0f;
    for (uint i = 0; i < TQ3_GROUP_SIZE; ++i) {
        kf[i] = float(new_k[elem_off + i]);
        vf[i] = float(new_v[elem_off + i]);
        k_norm_sq += kf[i] * kf[i];
        v_norm_sq += vf[i] * vf[i];
        k_max = max(k_max, fabs(kf[i]));
        v_max = max(v_max, fabs(vf[i]));
    }

    float k_inv = (k_max > 1e-12f) ? (TURBO3_MAX / k_max) : 1.0f;
    float v_inv = (v_max > 1e-12f) ? (TURBO3_MAX / v_max) : 1.0f;

    uchar k_idx[TQ3_GROUP_SIZE], v_idx[TQ3_GROUP_SIZE];
    float k_recon_sq = 0.0f, v_recon_sq = 0.0f;
    for (uint i = 0; i < TQ3_GROUP_SIZE; ++i) {
        k_idx[i] = turbo3_quantize(kf[i] * k_inv);
        v_idx[i] = turbo3_quantize(vf[i] * v_inv);
        float kc = TURBO3_CODEBOOK[k_idx[i]];
        float vc = TURBO3_CODEBOOK[v_idx[i]];
        k_recon_sq += kc * kc;
        v_recon_sq += vc * vc;
    }
    float k_recon_norm = sqrt(k_recon_sq);
    float v_recon_norm = sqrt(v_recon_sq);

    float ks = (k_recon_norm > 1e-10f) ? (sqrt(k_norm_sq) / k_recon_norm)
                                       : (k_max / TURBO3_MAX);
    float vs = (v_recon_norm > 1e-10f) ? (sqrt(v_norm_sq) / v_recon_norm)
                                       : (v_max / TURBO3_MAX);
    ks = min(ks, FP8_E4M3_MAX);
    vs = min(vs, FP8_E4M3_MAX);

    uint scale_row = cache_pos * num_groups;
    k_scales[scale_row + g] = f32_to_e4m3(ks);
    v_scales[scale_row + g] = f32_to_e4m3(vs);

    // 16 × 3-bit → 6 bytes per group.
    uint data_row = cache_pos * (n_elems * 3 / 8);
    uint byte_base = elem_off * 3 / 8;
    pack8x3(k_idx, k_data + data_row + byte_base);
    pack8x3(k_idx + 8, k_data + data_row + byte_base + 3);
    pack8x3(v_idx, v_data + data_row + byte_base);
    pack8x3(v_idx + 8, v_data + data_row + byte_base + 3);
}
