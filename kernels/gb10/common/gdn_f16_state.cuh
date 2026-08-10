// SPDX-License-Identifier: AGPL-3.0-only

// FP16 h-state storage helpers shared by the GDN speculative-verify twins
// (`ATLAS_SSM_H_FP16`, stage 2).
//
// Stage 1 (wave 32) narrowed the h-state of the NON-speculative decode scan to
// FP16 and proved the lever: the scan is pure state traffic, so halving the
// footprint halves the time (178.6 -> 82.9 ms/step at n=128). Stage 2 carries
// the same storage dtype through the MTP verify path, whose WY kernels are the
// h-state readers/writers whenever `--speculative` is on. Without these twins
// the flag and speculation are mutually exclusive, which is why preflight
// refused the combination.
//
// STORAGE-ONLY NARROWING. Every float expression, accumulation order, gate
// clamp and reduction in the twins is copied verbatim from the FP32 parent —
// the ONLY change is that the h-state (and its rollback intermediates) are
// loaded through `__half2float` and stored through `gdn_f16_store`. The
// arithmetic still runs in FP32 registers, so the twins differ from their
// parents by exactly the h round-trip rounding and nothing else.
//
// LAYOUT. The pool stays FP32-SIZED (prefill still writes FP32), and the FP16
// state occupies the FIRST HALF of each slot, densely packed — exactly the
// layout `ssm_h_state_f32_to_f16` produces, since that converter is a flat
// element-for-element compaction. So WITHIN a slot the head offset is still
// `vh * k_dim * v_dim` elements; only the SLOT-TO-SLOT stride would differ
// (slots are twice the dense FP16 footprint apart), and the verify path never
// needs it: the cross-sequence arm passes device POINTER TABLES with one
// explicit base per sequence, and the contiguous arm is only ever launched at
// batch_size == 1. Both twins keep the `state_is_table` contract of their
// parents, and the Rust launchers refuse the contiguous form at batch_size > 1
// under FP16 rather than silently applying FP32 slot arithmetic.

#ifndef ATLAS_GDN_F16_STATE_CUH
#define ATLAS_GDN_F16_STATE_CUH

#include <cuda_fp16.h>

// Saturating FP32 -> FP16 h-state store.
//
// Saturating rather than a raw `__float2half` for the reason wave 32 recorded:
// the per-head Frobenius clamp runs AFTER the state update and re-reads H, and
// `inf * scale` is still `inf`, so a single overflow would poison `hk_dot` for
// the rest of the sequence. Below 65504 this is bit-for-bit `__float2half`, and
// the in-tree clamps bound every element at <= 1000, so the saturation arm is
// unreachable in practice — it exists so that if it ever IS reached the state
// degrades instead of turning into NaN.
__device__ __forceinline__ __half gdn_f16_store(float v) {
    return __float2half(fminf(fmaxf(v, -65504.0f), 65504.0f));
}

// Widening h-state load. Named for symmetry with the store so the twins read
// as a dtype change and diff cleanly against their FP32 parents.
__device__ __forceinline__ float gdn_f16_load(const __half* __restrict__ p) {
    return __half2float(*p);
}

#endif  // ATLAS_GDN_F16_STATE_CUH
