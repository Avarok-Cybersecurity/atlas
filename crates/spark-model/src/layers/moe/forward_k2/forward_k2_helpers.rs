// SPDX-License-Identifier: AGPL-3.0-only

//! Pure dispatch helpers for the K=2-verify MoE path. Split out of
//! `forward_k2.rs` (500-LoC cap) — decisions only, no launches.

/// Block width for the NVFP4 batch2 MoE GEMVs — 128 (one warp per output pair)
/// or 256 (two warps joined through smem, which pays off once K is large). This
/// was inlined at the call site, where it read as a proxy for "is this model's
/// kernel one of the three 256-wide shadows" and over-fired for every other MoE
/// model at hidden_size ≥ 3072, launching them twice as wide as their `#define
/// BLOCK_SIZE 128`. The kernel reads `blockDim.x` now, so this is pure tuning.
pub(crate) fn batch2_block_width(hidden_size: usize) -> u32 {
    if hidden_size >= 3072 { 256 } else { 128 }
}

/// K=2-verify MoE dispatch guard. E8M0 (native MXFP4, per-32 E8M0 scale) routed
/// experts MUST take the per-token unified-T path (GS32 `_e8m0` kernel via
/// `e8m0_or`), NOT the GS16 NVFP4 `moe_expert_gate_up_shared_batch2_t` batch2
/// kernel: that kernel reads `inter·h/16` scale bytes from the correctly-sized
/// `inter·h/32` E8M0 scale buffer — a 2× over-read → CUDA_ERROR_ILLEGAL_ADDRESS.
/// Pure decision, unit-tested and wired at the top of `forward_k2`.
pub(crate) fn k2_e8m0_needs_per_token(scale_kind: crate::weight_map::WeightQuantFormat) -> bool {
    matches!(scale_kind, crate::weight_map::WeightQuantFormat::Mxfp4E8m0)
}
