// SPDX-License-Identifier: AGPL-3.0-only
//
// Walsh-Hadamard Transform (WHT) for BF16 vectors — Metal port of the
// CUDA kernel of the same name. Applied per-head to K/V before turbo
// quantization at cache-append time, to Q before turbo attention, and
// (inverse) to the attention output after.
//
// Grid: (num_heads, 1, 1) threadgroups × (32, 1, 1) threads — one
// simdgroup per head. head_dim 128 (32×4) and 256 (32×8) supported;
// 512 follows the CUDA kernel when a Metal model target needs it.
//
// With TQ_PLUS_SIGNS defined (extra_metal_flags), the rotation is the
// two-sided Rademacher form S2·H·S1 (seed=42 sign tables, identical to
// the CUDA tq_plus_signs.cuh vendoring). (S2·H·S1)·(S2·H·S1) ≠ I when
// S1 ≠ S2, so the inverse kernel reverses the sign order. Without the
// define, both kernels reduce to the plain self-inverse WHT.

#include <metal_stdlib>
using namespace metal;

#ifdef TQ_PLUS_SIGNS
// 128-element Rademacher signs (seed=42). Byte-identical values to the
// CUDA TQP_SIGNS{1,2}_128 tables.
constant float TQP_SIGNS1_128[128] = {
    -1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1,
    1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, -1,
    -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1,
    1, 1, 1, -1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, -1, 1,
    -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1,
    1, -1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, -1,
    -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, 1,
    1, -1, 1, 1, -1, -1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1
};
constant float TQP_SIGNS2_128[128] = {
    1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1,
    1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, 1,
    1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, -1,
    1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1,
    1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1, 1, 1,
    -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1,
    1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1,
    -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, -1
};
constant float TQP_SIGNS1_256[256] = {
    -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, -1,
    1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1,
    1, 1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1,
    1, 1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1,
    -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, -1, 1,
    -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1,
    1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1,
    1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1, 1,
    1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, -1, 1, -1, 1,
    -1, -1, 1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1,
    -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, 1, -1, -1,
    -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1,
    1, -1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1, 1, 1,
    1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1,
    1, -1, 1, -1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, -1, -1
};
constant float TQP_SIGNS2_256[256] = {
    -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1,
    1, 1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1,
    -1, 1, -1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1, -1, -1, 1,
    1, -1, 1, 1, 1, 1, 1, -1, 1, -1, -1, 1, -1, -1, -1, 1,
    1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1,
    -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, 1, 1, -1, 1, -1,
    -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1,
    -1, -1, 1, 1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1,
    -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, -1,
    1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1, -1, -1, -1,
    -1, 1, 1, 1, 1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1,
    -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1, -1, 1,
    -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1,
    1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1,
    1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1,
    -1, 1, -1, 1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1
};
#endif // TQ_PLUS_SIGNS

// In-place butterfly network over the values one simdgroup holds.
// VPT values per thread: stages within the thread first (stride 1..VPT/2),
// then across lanes via simd_shuffle_xor (mask 1..16). Normalizes by
// 1/sqrt(32 * VPT) at the end.
template <int VPT>
static inline void wht_simdgroup(thread float (&vals)[VPT], uint lane) {
    for (int stride = 1; stride <= VPT / 2; stride <<= 1) {
        for (int i = 0; i < VPT; i += stride * 2) {
            for (int j = 0; j < stride; j++) {
                float a = vals[i + j];
                float b = vals[i + j + stride];
                vals[i + j] = a + b;
                vals[i + j + stride] = a - b;
            }
        }
    }
    for (int xor_mask = 1; xor_mask <= 16; xor_mask <<= 1) {
        for (int i = 0; i < VPT; i++) {
            float other = simd_shuffle_xor(vals[i], (ushort)xor_mask);
            vals[i] = (lane & xor_mask) ? (other - vals[i]) : (vals[i] + other);
        }
    }
    float norm = 1.0f / sqrt(float(32 * VPT));
    for (int i = 0; i < VPT; i++) vals[i] *= norm;
}

// direction 0 = forward (signs1 → WHT → signs2), 1 = inverse (signs2 →
// WHT → signs1). Without TQ_PLUS_SIGNS both directions are the plain
// self-inverse WHT.
template <int VPT, int DIRECTION>
static inline void wht_head(device bfloat *head_data, uint lane
#ifdef TQ_PLUS_SIGNS
                            , constant float *signs1, constant float *signs2
#endif
) {
    float vals[VPT];
    for (int i = 0; i < VPT; i++) {
        vals[i] = float(head_data[lane * VPT + i]);
    }
#ifdef TQ_PLUS_SIGNS
    constant float *pre = (DIRECTION == 0) ? signs1 : signs2;
    for (int i = 0; i < VPT; i++) vals[i] *= pre[lane * VPT + i];
#endif
    wht_simdgroup<VPT>(vals, lane);
#ifdef TQ_PLUS_SIGNS
    constant float *post = (DIRECTION == 0) ? signs2 : signs1;
    for (int i = 0; i < VPT; i++) vals[i] *= post[lane * VPT + i];
#endif
    for (int i = 0; i < VPT; i++) {
        head_data[lane * VPT + i] = bfloat(vals[i]);
    }
}

#ifdef TQ_PLUS_SIGNS
#define WHT_DISPATCH(VPT, DIR, S1, S2) \
    wht_head<VPT, DIR>(head_data, lane, S1, S2)
#else
#define WHT_DISPATCH(VPT, DIR, S1, S2) wht_head<VPT, DIR>(head_data, lane)
#endif

kernel void wht_bf16_inplace(
    constant uint &head_dim   [[buffer(0)]],
    device bfloat *data       [[buffer(1)]],   // [num_heads, head_dim]
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    if (lane >= 32) return;
    device bfloat *head_data = data + (ulong)head * head_dim;
    if (head_dim >= 256) {
        WHT_DISPATCH(8, 0, TQP_SIGNS1_256, TQP_SIGNS2_256);
    } else {
        WHT_DISPATCH(4, 0, TQP_SIGNS1_128, TQP_SIGNS2_128);
    }
}

kernel void wht_bf16_inplace_inv(
    constant uint &head_dim   [[buffer(0)]],
    device bfloat *data       [[buffer(1)]],
    uint head [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    if (lane >= 32) return;
    device bfloat *head_data = data + (ulong)head * head_dim;
    if (head_dim >= 256) {
        WHT_DISPATCH(8, 1, TQP_SIGNS1_256, TQP_SIGNS2_256);
    } else {
        WHT_DISPATCH(4, 1, TQP_SIGNS1_128, TQP_SIGNS2_128);
    }
}
