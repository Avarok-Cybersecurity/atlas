// SPDX-License-Identifier: AGPL-3.0-only
//
// Safer-asym appends: K stays raw bf16, V is TurboQuant-compressed —
// Metal counterparts of the CUDA `reshape_and_cache_flash_bf16k_turbo*v`
// write paths, for the contiguous (non-paged) Metal cache. One entry
// point per V dtype, sharing the quantization helpers.
//
// Per-side rotation contract (mirrors the CUDA bookends): K is NOT
// rotated (it stays in the original basis, so the decode kernel scores
// raw Q against raw K); the caller rotates V with `wht_bf16_inplace`
// BEFORE this kernel and applies `wht_bf16_inplace_inv` to the
// attention output.
//
// Layout:
//   new_k    : bfloat [num_kv_heads, head_dim]   (un-rotated)
//   new_v    : bfloat [num_kv_heads, head_dim]   (WHT-rotated)
//   k_cache  : bfloat [max_seq, num_kv_heads * head_dim]
//   v_data   : uchar  [max_seq, packed V bytes]
//   v_scales : uchar  [max_seq, num_kv_heads * head_dim / 16]  (E4M3)
//
// Grid: (num_groups, 1, 1) threads — one thread per group of 16.

#include <metal_stdlib>
using namespace metal;

constant uint ASYM_GROUP_SIZE = 16;
constant float FP8_E4M3_MAX = 448.0f;

constant float TURBO4_CODEBOOK[16] = {
    -2.7326f, -2.0690f, -1.6180f, -1.2562f, -0.9423f, -0.6568f, -0.3880f, -0.1284f,
     0.1284f,  0.3880f,  0.6568f,  0.9423f,  1.2562f,  1.6180f,  2.0690f,  2.7326f
};
constant float TURBO4_BOUNDS[15] = {
    -2.4008f, -1.8435f, -1.4371f, -1.0993f, -0.7996f, -0.5224f, -0.2582f, 0.0f,
     0.2582f,  0.5224f,  0.7996f,  1.0993f,  1.4371f,  1.8435f,  2.4008f
};
constant float TURBO4_MAX = 2.7326f;

constant float TURBO3_CODEBOOK[8] = {
    -2.1520f, -1.3440f, -0.7560f, -0.2451f, 0.2451f, 0.7560f, 1.3440f, 2.1520f
};
constant float TURBO3_BOUNDS[7] = {
    -1.748f, -1.050f, -0.501f, 0.0f, 0.501f, 1.050f, 1.748f
};
constant float TURBO3_MAX = 2.1520f;

constant float TURBO2_CODEBOOK[4] = { -1.5104f, -0.4528f, 0.4528f, 1.5104f };
constant float TURBO2_BOUNDS[3] = { -0.9816f, 0.0f, 0.9816f };
constant float TURBO2_MAX = 1.5104f;

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

// Quantize one group of 16 rotated V values with the matched-norm L2
// scale; writes the E4M3 scale byte and returns the 16 indices.
static inline void quant_group(
    device const bfloat *src, uint elem_off,
    constant float *codebook, constant float *bounds, uint n_bounds, float cmax,
    thread uchar *idx, device uchar *scale_out)
{
    float vf[ASYM_GROUP_SIZE];
    float norm_sq = 0.0f, vmax = 0.0f;
    for (uint i = 0; i < ASYM_GROUP_SIZE; ++i) {
        vf[i] = float(src[elem_off + i]);
        norm_sq += vf[i] * vf[i];
        vmax = max(vmax, fabs(vf[i]));
    }
    float inv = (vmax > 1e-12f) ? (cmax / vmax) : 1.0f;
    float recon_sq = 0.0f;
    for (uint i = 0; i < ASYM_GROUP_SIZE; ++i) {
        float x = vf[i] * inv;
        uchar q = 0;
        while (q < n_bounds && x >= bounds[q]) {
            q++;
        }
        idx[i] = q;
        float c = codebook[q];
        recon_sq += c * c;
    }
    float recon_norm = sqrt(recon_sq);
    float s = (recon_norm > 1e-10f) ? (sqrt(norm_sq) / recon_norm) : (vmax / cmax);
    *scale_out = f32_to_e4m3(min(s, FP8_E4M3_MAX));
}

// Copy one group of raw bf16 K into the cache.
static inline void copy_k_group(
    device const bfloat *new_k, device bfloat *k_cache,
    uint cache_pos, uint n_elems, uint elem_off)
{
    device bfloat *dst = k_cache + (ulong)cache_pos * n_elems + elem_off;
    for (uint i = 0; i < ASYM_GROUP_SIZE; ++i) {
        dst[i] = new_k[elem_off + i];
    }
}

#define ASYM_APPEND_PROLOGUE \
    uint n_elems = num_kv_heads * head_dim; \
    uint num_groups = n_elems / ASYM_GROUP_SIZE; \
    if (g >= num_groups) { return; } \
    uint elem_off = g * ASYM_GROUP_SIZE; \
    copy_k_group(new_k, k_cache, cache_pos, n_elems, elem_off); \
    uchar idx[ASYM_GROUP_SIZE]; \
    uint scale_row = cache_pos * num_groups;

#define ASYM_APPEND_PARAMS \
    constant uint &num_kv_heads [[buffer(0)]], \
    constant uint &head_dim     [[buffer(1)]], \
    constant uint &cache_pos    [[buffer(2)]], \
    device const bfloat *new_k  [[buffer(3)]], \
    device const bfloat *new_v  [[buffer(4)]], \
    device bfloat *k_cache      [[buffer(5)]], \
    device uchar  *v_data       [[buffer(6)]], \
    device uchar  *v_scales     [[buffer(7)]], \
    uint g [[thread_position_in_grid]]

kernel void kv_cache_append_bf16k_turbo4v(ASYM_APPEND_PARAMS) {
    ASYM_APPEND_PROLOGUE
    quant_group(new_v, elem_off, TURBO4_CODEBOOK, TURBO4_BOUNDS, 15, TURBO4_MAX,
                idx, v_scales + scale_row + g);
    uint data_row = cache_pos * (n_elems / 2);
    for (uint i = 0; i < ASYM_GROUP_SIZE; i += 2) {
        v_data[data_row + (elem_off + i) / 2] = idx[i] | (idx[i + 1] << 4);
    }
}

kernel void kv_cache_append_bf16k_turbo3v(ASYM_APPEND_PARAMS) {
    ASYM_APPEND_PROLOGUE
    quant_group(new_v, elem_off, TURBO3_CODEBOOK, TURBO3_BOUNDS, 7, TURBO3_MAX,
                idx, v_scales + scale_row + g);
    uint data_row = cache_pos * (n_elems * 3 / 8);
    device uchar *out = v_data + data_row + elem_off * 3 / 8;
    for (uint half_g = 0; half_g < 2; ++half_g) {
        thread const uchar *p = idx + half_g * 8;
        out[half_g * 3 + 0] = p[0] | (p[1] << 3) | (p[2] << 6);
        out[half_g * 3 + 1] = (p[2] >> 2) | (p[3] << 1) | (p[4] << 4) | (p[5] << 7);
        out[half_g * 3 + 2] = (p[5] >> 1) | (p[6] << 2) | (p[7] << 5);
    }
}

kernel void kv_cache_append_bf16k_turbo2v(ASYM_APPEND_PARAMS) {
    ASYM_APPEND_PROLOGUE
    quant_group(new_v, elem_off, TURBO2_CODEBOOK, TURBO2_BOUNDS, 3, TURBO2_MAX,
                idx, v_scales + scale_row + g);
    uint data_row = cache_pos * (n_elems / 4);
    for (uint i = 0; i < ASYM_GROUP_SIZE; i += 4) {
        v_data[data_row + (elem_off + i) / 4] =
            idx[i] | (idx[i + 1] << 2) | (idx[i + 2] << 4) | (idx[i + 3] << 6);
    }
}
