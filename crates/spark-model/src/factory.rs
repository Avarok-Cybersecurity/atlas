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
    DeepSeekV4WeightLoader, DflashConfig, Gemma4WeightLoader, MinimaxM2WeightLoader,
    ModelWeightLoader, NemotronHWeightLoader, NllbWeightLoader, Qwen3VLWeightLoader,
    Qwen3WeightLoader, Qwen35DenseWeightLoader, Qwen35WeightLoader, Step3p7WeightLoader,
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

/// LoRA adapter build arguments (`--lora-adapter NAME=PATH`). `None` for
/// base-only runs; `Some(...)` carries the adapter's separate on-device
/// [`WeightStore`] (loaded via
/// `spark_runtime::weights::adapter::load_adapter_safetensors`), the parsed
/// `adapter_config.json`, and the pool-shape CLI knobs.
///
/// Unlike DFlash (loaded post-construction), the LoRA pool is allocated at
/// the TOP of `build_model` — before the buffer arena and the free-memory
/// snapshot — so its bytes are automatically debited from the KV budget.
pub struct LoraBuildArgs<'a> {
    /// One or more adapters to pack (repeated `--lora-adapter NAME=PATH`),
    /// each carrying its NAME, its on-device `WeightStore`, and its parsed
    /// `adapter_config.json`. Slot k = `adapters[k]`. A single element is
    /// byte-identical to the pre-multi-adapter path.
    pub adapters: Vec<crate::lora::LoraAdapterInput<'a>>,
    pub max_lora_rank: usize,
    pub max_loras: usize,
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
        "qwen3_6_moe" | "holo3_1_moe" => Ok(Box::new(Qwen35WeightLoader)),
        // Nemotron-H family (Mamba-2 + MoE + Attention), including Puzzle
        // (heterogeneous per-block MoE intermediate / top-k).
        "nemotron_h" | "nemotron_h_puzzle" => Ok(Box::new(NemotronHWeightLoader)),
        // NLLB / M2M-100 encoder-decoder translation family.
        "m2m_100" | "nllb" => Ok(Box::new(NllbWeightLoader)),
        // Gemma-4 family (pure attention, GeGLU, sliding + full attention)
        "gemma4" | "gemma_4" => Ok(Box::new(Gemma4WeightLoader)),
        // Mistral family (MLA + MoE, GQA fallback for initial bring-up)
        "mistral" => Ok(Box::new(MistralWeightLoader)),
        // MiniMax M2 family (M2.1 / M2.7) — full attention + 256-expert
        // sigmoid-routed MoE + 3-module MTP.
        "minimax_m2" => Ok(Box::new(MinimaxM2WeightLoader)),
        // Step 3.7 Flash — 288-expert sigmoid-routed MoE + shared expert +
        // mixed full/sliding attention + attention gate + 3 MTP modules.
        "step3p7" => Ok(Box::new(Step3p7WeightLoader)),
        // DeepSeek-V4 family (Flash) — MLA + MoE + CSA/HCA hybrid attention + mHC.
        "deepseek_v4" => Ok(Box::new(DeepSeekV4WeightLoader)),
        _ => bail!(
            "Unsupported model type: '{}' (normalized: '{}'). \
             Supported: qwen3_next, qwen3_5_moe, qwen3_5, qwen3_6_moe, holo3_1_moe, qwen3_vl_moe, nemotron_h, nemotron_h_puzzle, gemma4, mistral, minimax_m2, deepseek_v4, m2m_100",
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
            None, // lora_args
            None, // nllb_lang
            None, // nllb_lora_dir
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

        config.model_type = "holo3_1_moe".to_string();
        assert!(loader_for_config(&config).is_ok());

        config.model_type = "m2m_100".to_string();
        assert!(loader_for_config(&config).is_ok());

        config.model_type = "unsupported_model".to_string();
        assert!(loader_for_config(&config).is_err());
    }

    #[test]
    fn test_nllb_loader_fails_fast_until_encoder_decoder_runtime_exists() {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "nllb".to_string();
        let loader = loader_for_config(&config).unwrap();
        let store = WeightStore::empty();
        let gpu = spark_runtime::gpu::mock::MockGpuBackend::new();

        let err = loader.load_embedding(&store, &config, &gpu).unwrap_err();
        assert!(
            err.to_string().contains(
                "Atlas does not yet implement the encoder-decoder runtime required by facebook/nllb-200-3.3B"
            ),
            "{err}"
        );
    }
}
