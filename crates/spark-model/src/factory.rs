// SPDX-License-Identifier: AGPL-3.0-only

//! Model factory: builds the right model from config + weights.
//!
//! Weight loader selection is registry-driven — add new models by implementing
//! [`ModelWeightLoader`] and registering in [`loader_for_config`]. No other
//! code changes needed.

use anyhow::{Result, bail};
use atlas_core::config::ModelConfig;
use spark_runtime::weights::WeightStore;

use crate::mistral_loader::MistralWeightLoader;
use crate::weight_loader::{
    DflashConfig, Gemma4WeightLoader, MinimaxM2WeightLoader, ModelWeightLoader,
    NemotronHWeightLoader, Qwen3VLWeightLoader, Qwen3WeightLoader, Qwen35DenseWeightLoader,
    Qwen35WeightLoader,
};

/// DFlash speculative-decoding build arguments. `None` for non-DFlash runs;
/// `Some(...)` carries the drafter's separate [`WeightStore`], parsed
/// `config.json`, and CLI overrides for γ and the sliding-window size.
///
/// Construction order: the caller (`spark-server::main`) loads the drafter
/// checkpoint into a fresh [`WeightStore`] via the same `WeightStore::load`
/// path used for the target, then parses `config.json` via
/// [`crate::weight_loader::dflash_loader::parse_dflash_config`]. Both inputs
/// flow through to [`build_model`] which validates dimensions against the
/// target before constructing [`crate::layers::BlockDiffusionDraftHead`].
pub struct DflashBuildArgs<'a> {
    pub drafter_store: &'a WeightStore,
    pub drafter_config: DflashConfig,
    pub gamma: Option<usize>,
    pub window_size: Option<usize>,
}

// ── Loader registry ─────────────────────────────────────────────────────────
// Adding a new model: implement ModelWeightLoader and add a match arm below.
// Everything else (KV cache, buffers, TransformerModel) is model-agnostic.

/// Select the weight loader for a given model config.
///
/// This is the ONLY place model_type strings are matched. All downstream
/// code is model-agnostic.
pub fn loader_for_config(config: &ModelConfig) -> Result<Box<dyn ModelWeightLoader>> {
    let normalized = config.model_type.to_lowercase().replace(['-', '.'], "_");
    match normalized.as_str() {
        // Qwen3 family: sub-dispatch by config predicates
        "qwen3_next" => Ok(Box::new(Qwen3WeightLoader)),
        "qwen3_vl_moe" => Ok(Box::new(Qwen3VLWeightLoader)),
        "qwen3_5_moe" | "qwen3_5" | "qwen35_moe" | "qwen35" => {
            // Dense check has to come first. Qwen3.6-27B-FP8 is the dense text
            // sibling of the Qwen3.6 VL family — its config declares the same
            // `vision_config` block as the MoE-VL siblings (so `is_qwen3_vl()`
            // returns true), but the checkpoint ships no vision tower and no
            // MoE router, so the VL loader panics on a missing `mlp.gate`.
            // `is_qwen35_dense()` requires `num_experts == 0`, which only the
            // dense text models satisfy — VL-MoE always has experts.
            if config.is_qwen35_dense() {
                Ok(Box::new(Qwen35DenseWeightLoader))
            } else if config.is_qwen3_vl() {
                Ok(Box::new(Qwen3VLWeightLoader))
            } else {
                Ok(Box::new(Qwen35WeightLoader))
            }
        }
        // Qwen3.6: identical architecture to Qwen3.5 MoE at the weight level
        // (GDN + full-attention + MoE hybrid, same expert layout, same MTP).
        // Only difference is MRoPE-interleaved layout + attn_output_gate on
        // full-attention layers — both handled at forward-pass layer time,
        // not during weight loading.
        "qwen3_6_moe" => Ok(Box::new(Qwen35WeightLoader)),
        // Nemotron-H family (Mamba-2 + MoE + Attention)
        "nemotron_h" => Ok(Box::new(NemotronHWeightLoader)),
        // Gemma-4 family (pure attention, GeGLU, sliding + full attention)
        "gemma4" | "gemma_4" => Ok(Box::new(Gemma4WeightLoader)),
        // Mistral family (MLA + MoE, GQA fallback for initial bring-up)
        "mistral" => Ok(Box::new(MistralWeightLoader)),
        // MiniMax M2 family (M2.1 / M2.7) — full attention + 256-expert
        // sigmoid-routed MoE + 3-module MTP.
        "minimax_m2" => Ok(Box::new(MinimaxM2WeightLoader)),
        _ => bail!(
            "Unsupported model type: '{}' (normalized: '{}'). \
             Supported: qwen3_next, qwen3_5_moe, qwen3_5, qwen3_6_moe, qwen3_vl_moe, nemotron_h, gemma4, mistral, minimax_m2",
            config.model_type,
            normalized,
        ),
    }
}

mod build;
mod lm_head_setup;
mod m2_setup;

pub use build::build_model;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::mtp_head::MtpQuantization;
    use spark_runtime::kv_cache::KvCacheDtype;
    use spark_runtime::prefix_cache::PrefixCache;

    #[test]
    fn test_unsupported_model_type() {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "llama".to_string();

        let gpu = spark_runtime::gpu::mock::MockGpuBackend::new();
        let store = WeightStore::empty();

        let prefix_cache: Box<dyn PrefixCache> =
            Box::new(spark_runtime::prefix_cache::NoPrefixCaching);
        let result = build_model(
            config,
            &store,
            Box::new(gpu),
            1,
            16,
            4096,
            8,
            MtpQuantization::Nvfp4,
            false,
            prefix_cache,
            0,
            None,
            false,
            1,
            KvCacheDtype::Fp8,
            1024 * 1024 * 1024,
            0.90,
            0,
            vec![],
            0,
            None,
            None, // dflash_args
        );
        match result {
            Err(e) => assert!(e.to_string().contains("Unsupported model type: 'llama'")),
            Ok(_) => panic!("Expected error for unsupported model type"),
        }
    }

    #[test]
    fn test_loader_selection() {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "qwen3_next".to_string();
        assert!(loader_for_config(&config).is_ok());

        config.model_type = "nemotron_h".to_string();
        assert!(loader_for_config(&config).is_ok());

        config.model_type = "unsupported_model".to_string();
        assert!(loader_for_config(&config).is_err());
    }
}
