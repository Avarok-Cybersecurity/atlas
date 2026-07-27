// SPDX-License-Identifier: AGPL-3.0-only

#![deny(warnings)]
#![deny(clippy::all)]
// Kernel-launch helpers and trait-impl wide signatures legitimately exceed
// clippy's 7-argument default. The same goes for the indexing-loop patterns
// that mirror the kernel grids we dispatch.
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
// Some FP/integer special-case branches return the same value but have
// distinct semantic meanings (NaN vs zero, etc.). Audit shows these are
// intentional.
#![allow(clippy::if_same_then_else)]
// The HSS / disk-spill plumbing threads `Vec<u32>` through trait methods so
// callers can grow them in place; converting to slices breaks the contract.
#![allow(clippy::ptr_arg)]
// HF safetensors index tuples are wide on purpose.
#![allow(clippy::type_complexity)]

pub mod engine;
pub mod factory;
pub mod forward;
pub mod layer;
pub mod layers;
pub mod lora;
pub mod mistral_loader;
pub mod model;
pub mod precision_schedule;
pub mod preflight;
pub mod quant_format;
pub mod speculative;
pub mod tp_shard;
pub mod traits;
pub mod vision_preprocess;
pub mod weight_loader;
pub mod weight_map;

/// True when the checkpoint ships **HF-vanilla** RMSNorm weights — i.e. the norm
/// weight is used as `out = x * w / rms`, not Qwen3-Next's offset-from-1
/// `out = x * (1 + w) / rms`.
///
/// Such a model must load its norm weights **exactly** and dispatch
/// `rms_norm_vanilla`. The alternative — pre-subtracting 1.0 and storing
/// `bf16(w - 1)` for the offset kernel — is only lossless when `w ≈ 1`.
/// DeepSeek-V4's norm weights are ≈ 0.03, so `w - 1 ≈ -0.97`, and BF16's
/// rounding error there (~1.9e-3 absolute) becomes a **1.8-3.4 % relative error
/// on the weight itself** once 1 is added back — catastrophic cancellation.
/// Measured over all 249 V4 norm tensors: up to 19 % on `q_norm`, and 100 %
/// with sign flips on the compressor norms.
///
/// This is an explicit model dispatch, NOT an inference from weight statistics.
pub fn ships_vanilla_norm_weights(config: &atlas_core::config::ModelConfig) -> bool {
    model_type_ships_vanilla_norm_weights(&config.model_type)
}

/// The dispatch predicate itself, on the bare `model_type`, so it is unit-testable
/// without constructing a full `ModelConfig`.
pub fn model_type_ships_vanilla_norm_weights(model_type: &str) -> bool {
    model_type == "deepseek_v4"
}

#[cfg(test)]
mod norm_convention_tests {
    use super::model_type_ships_vanilla_norm_weights as vanilla;
    use half::bf16;

    /// Only DeepSeek-V4 takes the vanilla path. Every other family keeps the
    /// offset-from-1 convention it was loaded and validated under.
    #[test]
    fn only_deepseek_v4_uses_vanilla_norm_weights() {
        assert!(vanilla("deepseek_v4"));
        for other in [
            "qwen3_next",
            "qwen3_5_moe",
            "qwen3_moe",
            "deepseek_v3",
            "llama",
            "mistral",
            "nemotron",
            "",
        ] {
            assert!(!vanilla(other), "{other} must keep offset-from-1 semantics");
        }
    }

    /// The norm weights ship as BF16 in the checkpoint, so the value the loader
    /// actually sees is `bf16(x)`. Model that exactly.
    fn ckpt(x: f32) -> f32 {
        bf16::from_f32(x).to_f32()
    }
    /// OLD loader: store `bf16(w - 1)`, so the offset-from-1 kernel applies
    /// `1 + bf16(w - 1)`.
    fn old_effective(w: f32) -> f32 {
        1.0 + bf16::from_f32(w - 1.0).to_f32()
    }
    fn rel_err(got: f32, want: f32) -> f32 {
        ((got - want) / want).abs()
    }

    /// The whole reason for this change: the offset round-trip is EXACTLY lossless
    /// when `w` is near 1 — which is why every other model family was unaffected —
    /// and badly lossy when `w` is near 0, which is what DeepSeek-V4 ships.
    #[test]
    fn offset_roundtrip_is_lossless_near_one_and_lossy_near_zero() {
        // Near 1 and above (every offset-convention model, and V4's own kv_norm in
        // most layers): `w - 1` stays exactly representable, so the round-trip is
        // bit-exact. This is the regression guarantee for other model families.
        for x in [0.63_f32, 0.75, 0.875, 0.99, 1.0, 1.5, 2.0, 3.828125] {
            let w = ckpt(x);
            assert_eq!(
                old_effective(w),
                w,
                "w={w} must round-trip EXACTLY under the offset convention"
            );
        }

        // Near 0 — real `attn_norm` values from DeepSeek-V4-Flash layers.0.
        // Catastrophic cancellation: a large RELATIVE error on the weight itself.
        for x in [0.028931_f32, 0.030762, 0.032471, 0.033447] {
            let w = ckpt(x);
            let old_err = rel_err(old_effective(w), w);
            assert!(
                old_err > 0.01,
                "attn_norm w={w}: the old path should be >1% wrong, got {:.2}%",
                old_err * 100.0
            );
            // The exact load is lossless by construction: the checkpoint is BF16.
            assert_eq!(ckpt(w), w);
        }
    }

    /// `q_norm`'s small tail — the old path is ~20 % wrong on the weight.
    #[test]
    fn q_norm_worst_case_is_badly_wrong_under_the_offset_convention() {
        let w = ckpt(0.009766);
        let old_err = rel_err(old_effective(w), w);
        assert!(
            old_err > 0.10,
            "expected >10% relative error, got {:.2}%",
            old_err * 100.0
        );
        assert_eq!(ckpt(w), w);
    }

    /// Compressor / indexer norms straddle zero. The offset round-trip does not
    /// merely degrade them — `bf16(w - 1)` rounds to exactly -1.0, so `1 + (-1.0)`
    /// collapses the weight to ZERO, destroying sign and magnitude together.
    /// No downstream precision can recover that.
    #[test]
    fn compressor_norm_sign_collapse_under_the_offset_convention() {
        let w = ckpt(-0.001);
        let old = old_effective(w);
        assert_eq!(old, 0.0, "expected collapse to zero, got {old}");
        assert!(w < 0.0, "the true weight is negative");
        assert_eq!(ckpt(w), w, "the exact load preserves it");
    }
}
