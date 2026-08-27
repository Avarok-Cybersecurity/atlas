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
pub mod ssm_reserve;
pub mod tp_shard;
pub mod traits;
pub mod video_decode_ffmpeg;
pub mod video_preprocess;
pub mod vision_item;
pub mod vision_preprocess;
pub use vision_item::VisionItem;

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
    matches!(model_type, "deepseek_v4" | "laguna")
}

#[cfg(test)]
mod norm_convention_tests {
    use super::model_type_ships_vanilla_norm_weights as vanilla;

    /// Only explicitly listed model families take the vanilla path. Every
    /// other family keeps the offset-from-1 convention it was validated under.
    #[test]
    fn vanilla_norm_models_are_explicit() {
        assert!(vanilla("deepseek_v4"));
        assert!(vanilla("laguna"));
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
}
