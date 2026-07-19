// SPDX-License-Identifier: AGPL-3.0-only
//
// Shared MoE block-scale primitive — ONE E8M0 dequant-scale definition for
// every native-MXFP4 kernel entry (ARM-2 Phase-K, RIDER 1).
//
// Both the decode GEMV (common/moe_shared_expert_fused_t.cu, Family A) and the
// prefill W4A16 grouped GEMM (qwen3.6-35b-a3b/nvfp4/moe_w4a16_grouped_gemm.cu,
// Family B) include THIS header so the E8M0 scale is bit-identical across
// families. A mixed regime (one family bit-construct, the other exp2f/ex2.approx)
// would plant a family-dependent numeric skew inside the discriminator serve —
// the exact confound RIDER 1 forbids. Do not copy these functions into a .cu.
//
// Leg-2 fast-check greps every `_e8m0` entry for the E8M0 bit-construct
// (`shl … 23`) and the ABSENCE of a scale-side `cvt.f32.e4m3` (the NVFP4
// scale decode). The weight→FP8 recast `cvt.rn.satfinite.e4m3x2.f32` in the
// prefill MMA path is unrelated and stays.

#pragma once

#include <cuda_fp8.h>

// NVFP4 per-block FP8-E4M3 scale decode. SCALE/gfx1151 `(float)__nv_fp8_e4m3`
// is NON-STANDARD (same bug fixed in moe_sorted_prefill.cu / the decode GEMVs) —
// software scl_fp8 there; the NVIDIA path is the verbatim cast.
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ float atlas_dec_e4m3(unsigned char b) {
    unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u; float v;
    if (e == 0u)               v = (float)m * 0.001953125f;
    else if (e == 15u && m == 7u) v = 0.0f;
    else                       v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return s ? -v : v;
}
#else
__device__ __forceinline__ float atlas_dec_e4m3(unsigned char b) {
    __nv_fp8_e4m3 f; *(unsigned char*)&f = b; return (float)f;
}
#endif

// Per-block dequant-scale, parameterized on the weight's scale format.
//   E8M0=false → NVFP4: FP8-E4M3 per-16 scale byte × per-tensor global (s2).
//   E8M0=true  → native MXFP4 (ARM-2): the scale byte is a biased E8M0
//                exponent; effective scale = 2^(sb-127), NO global (s2 ignored).
//                sb==0 → 0, sb==0xFF (NaN sentinel) → 0. Bit-constructs the
//                power of two so it is BYTE-EXACT with the Rust host reference
//                `fp8_e8m0_to_f32` (weight_map/fp8_lut.rs) — the Leg-2 SSOT —
//                rather than `exp2f`, whose ex2.approx would drift.
template<bool E8M0>
__device__ __forceinline__ float mx_block_scale(unsigned char sb, float s2) {
    if (E8M0) {
        if (sb == 0u || sb == 255u) return 0.0f;
        return __uint_as_float((unsigned int)sb << 23);
    } else {
        return atlas_dec_e4m3(sb) * s2;
    }
}
