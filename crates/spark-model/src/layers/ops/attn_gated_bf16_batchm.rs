// SPDX-License-Identifier: AGPL-3.0-only

//! The `attn_gated_bf16_batchm` row: gated dense-BF16 attention Q/K/V on the
//! batched `dense_gemv_batchm` tier in multi-sequence decode.
//!
//! # What it replaces
//!
//! `multi_seq/qkv.rs::ms_phase_qkv` has a batched dense-BF16 tier
//! (`ms_qkv_batchm_bf16`, one weight pass for 2..=8 rows), but it required
//! `!self.gated`. Output-gated attention (Qwen3.5, `config.attn_gated`: q_proj
//! emits an interleaved `[Q|gate]` row) fell to the per-sequence loop: per
//! stream one `dense_gemv` of q_proj, one `deinterleave_qg`, one `dense_gemv`
//! each of k_proj and v_proj. So every stream re-read the whole Q/K/V weight.
//! On the Qwen3.5 4B BF16 load (8 attention layers, q 8192 x 2560, k and v
//! 1024 x 2560) that is 0.42 GB per pass, n passes per step.
//!
//! With the row on, gated layers take the same batched tier: one
//! `dense_gemv_batchm` per projection writing the raw `[Q|gate]` rows into
//! the per-sequence `qkv_buf` slots, then one in-place `deinterleave_qg` per
//! row (the same kernel and arguments the per-sequence loop runs). With a q
//! LoRA adapter resident the deinterleave is deferred to
//! `ms_qkv_deinterleave_q`, exactly as on the per-sequence path.
//!
//! # Why ON for ada: bit-identical
//!
//! `dense_gemv_bf16_batchm` keeps `dense_gemv_bf16`'s per-row K-iteration
//! order, lo-then-hi add order, 64-thread stride, warp-shuffle tree and 2-warp
//! fold; the extra rows are independent accumulators over the same loop
//! (kernel header; measured bit-identical on 64 shapes at M = 1..8). Both
//! kernels are the same source on ada (`kernels/ada/common/*.cu` symlink to
//! `kernels/gb10/common/`), and `kernels/ada/common/KERNEL.toml` builds the
//! directory with `--fmad=false`, which is what makes the identity exact. The
//! deinterleave is a permutation. So the Q/K/V values are the per-sequence
//! loop's bits; only the number of weight passes changes. Every other target
//! declares `false`, so nothing changes there.
//!
//! # Overrides
//!
//! `AVAROK_ATTN_GATED_BF16_BATCHM` under the 2026-09-11 toggle grammar. The
//! older `AVAROK_BF16_QKV_BATCHM=0` kill switch still turns off the whole
//! batched dense-BF16 tier, gated or not.

use super::target_defaults::Resolved;

/// The row's override variable.
pub const ENV: &str = "AVAROK_ATTN_GATED_BF16_BATCHM";

/// Resolve the row from the declaration and [`ENV`] (`None` = absent).
pub fn resolve(default_on: bool, row: Option<&str>) -> Resolved<bool> {
    super::target_defaults::resolve_toggle(default_on, row, false)
}

/// The process-wide resolved row (cached on each attention layer at build).
pub fn row() -> bool {
    super::target_defaults::resolved()
        .attn_gated_bf16_batchm
        .value
}

/// Whether the batched dense-BF16 Q/K/V tier admits a layer, given its
/// gating and the row. Ungated layers were always admitted.
pub fn admits(gated: bool, row: bool) -> bool {
    !gated || row
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ungated_layers_are_admitted_either_way() {
        assert!(admits(false, false));
        assert!(admits(false, true));
    }

    #[test]
    fn gated_layers_follow_the_row() {
        assert!(!admits(true, false));
        assert!(admits(true, true));
    }

    #[test]
    fn resolution_follows_the_toggle_grammar() {
        assert!(resolve(true, None).value);
        assert!(!resolve(false, None).value);
        assert!(!resolve(true, Some("0")).value);
        assert!(resolve(false, Some("1")).value);
        assert!(resolve(false, Some("1")).from_env());
    }
}
