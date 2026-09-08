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
pub mod mtp_layout;
pub mod precision_schedule;
pub mod preflight;
pub mod quant_format;
mod rank_agree;
pub mod seq_state_reserve;
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
    // 🪤 `glm5_next` added 2026-08-27. GLM-5.3's norms are PLAIN `x * rms * w` — the same
    // trap `glm5next_layer` documents for its per-layer norms. This predicate additionally
    // picks the kernel for the MODEL-LEVEL final norm (`model/impl_a1.rs`), which is applied
    // outside any layer, so omitting GLM here silently normalises the final hidden state with
    // the `(1 + w)` offset and corrupts every token's logits. Nothing about the shapes says so.
    matches!(model_type, "deepseek_v4" | "laguna" | "glm5_next")
}

/// Must chunked prefill run as a SINGLE chunk for this model?
///
/// True only for models that reach the chunk-LOCAL MLA prefill in
/// `qwen3_attention/prefill.rs`, which attends over the current chunk's K/V alone —
/// multi-chunk there silently corrupts attention output (Mistral-Small-4, 2026-05-01: 8 K
/// collapses to "The\nThe…").
///
/// 🔴 `kv_lora_rank > 0` is a PROXY for that kernel and `glm5_next` breaks it: GLM-5.3 is
/// MLA (rank 512) but prefills through `Glm5NextLayer::prefill`, a per-token walk that
/// attends the whole paged prefix at each absolute position — chunk boundaries are
/// invisible to it. Answering true capped every GLM prompt at `2 × --max-prefill-tokens`,
/// because `prefill_a_step` splits the FIRST chunk at the cap regardless and this gate then
/// made the remainder one unsplit chunk the buffer arena refused. ANOMALIES A61.
pub fn requires_single_chunk_prefill(model_type: &str, kv_lora_rank: usize) -> bool {
    kv_lora_rank > 0 && model_type != "glm5_next"
}

#[cfg(test)]
mod single_chunk_prefill_tests {
    use super::requires_single_chunk_prefill as single;

    /// GLM-5.3 is MLA and must still be chunked — that is the whole of A61.
    #[test]
    fn glm5_next_is_mla_but_chunks_fine() {
        assert!(!single("glm5_next", 512));
        // Every other MLA family keeps the single-chunk guard.
        assert!(single("deepseek_v4", 512));
        assert!(single("mistral", 512));
        // Non-MLA models were never gated.
        assert!(!single("qwen3_5_moe", 0));
    }
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
        // GLM-5.3's norms are plain; the final norm is applied outside any layer.
        assert!(vanilla("glm5_next"));
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
