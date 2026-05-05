// SPDX-License-Identifier: AGPL-3.0-only

//! `impl VisionEncoder` body, split across sibling files for the ≤500
//! LoC cap. Each sibling adds methods to the `VisionEncoder` inherent
//! impl.
//!
//! - `init`        — `new()` constructor
//! - `pos_embed`   — `resample_pos_embed`, `build_rope_cossin`
//! - `patch_embed` — `patch_embed`
//! - `vit_block`   — `vit_block`
//! - `merger`      — `apply_merger`
//! - `forward`     — top-level `forward`
//! - `utils`       — `gpu_copy_bf16`, `maybe_dump_buf`

mod forward;
mod init;
mod merger;
mod patch_embed;
mod pos_embed;
mod utils;
mod vit_block;

/// Convert an f32 to BF16 bits using round-to-nearest-even. The input
/// domain for pos_embed and rotary cos/sin values is well within the
/// BF16 range, so the special-case handling for NaN / overflow is
/// inlined as standard bit-level rounding.
#[inline]
pub(super) fn f32_to_bf16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    if (bits & 0x7fff_ffff) > 0x7f80_0000 {
        // NaN → canonical quiet NaN in BF16.
        return 0x7fc0;
    }
    let rounding = 0x7fff + ((bits >> 16) & 1);
    ((bits.wrapping_add(rounding)) >> 16) as u16
}
