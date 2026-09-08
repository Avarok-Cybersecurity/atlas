// SPDX-License-Identifier: AGPL-3.0-only

//! `impl Glm5NextVisionEncoder` body, split across sibling files for the ≤500
//! LoC cap, mirroring `vision_encoder/enc_impl/`.
//!
//! - `init`        — `new()`, scratch allocation, the downsample-weight permutation
//! - `rope`        — 2D axial cos/sin table (no learned position embedding exists)
//! - `patch_embed` — f32 pixels → BF16 → patch-embed GEMM
//! - `attn`        — the GEMM-with-bias helper and GEMM-based SDPA
//! - `block`       — one ViT block, batched over Σpatches
//! - `merger`      — post_layernorm → merge → downsample → merger MLP
//! - `forward`     — `forward` / `forward_batched` / oversized fallback

pub(super) mod attn;
pub(super) mod block;
pub(super) mod forward;
pub(crate) mod init;
pub(super) mod merger;
pub(super) mod patch_embed;
pub(super) mod rope;

/// Round-to-nearest-even f32 → BF16 bits.
///
/// Re-exported from the Qwen tower rather than copied: two implementations of
/// a rounding rule is two places for a rounding rule to drift, and the RoPE
/// tables of both towers are built with it.
pub(super) use crate::layers::vision_encoder::enc_impl::f32_to_bf16_bits;
