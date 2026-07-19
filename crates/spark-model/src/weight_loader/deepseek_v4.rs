// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4 weight loader (MLA + MoE).
//!
//! Implements full layer loading for DeepSeek-V4-Flash, reusing the same
//! MLA attention pattern as Mistral Small 4 with DeepSeek weight naming.

mod assemble;
mod compute;
mod load_layers;
// MTP draft-module loader for nvidia/DeepSeek-V4-Flash-NVFP4.
mod mtp;
pub(crate) use mtp::{DeepseekV4MtpModule, load_v4_mtp_module};

use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use super::ModelWeightLoader;
use crate::layer::TransformerLayer;
use crate::weight_map::{DenseWeight, MtpWeights, dense, dense_minus_one};

pub struct DeepSeekV4WeightLoader;

impl ModelWeightLoader for DeepSeekV4WeightLoader {
    fn supports_tp(&self) -> bool {
        // DeepSeek-V4 uses num_key_value_heads=1 (MQA), which makes
        // head-parallel TP sharding impossible — 1 is not divisible by
        // any tp_size > 1.  Multi-spark deployments MUST use pure EP
        // (tp-size 1, ep-size 2/4/...) instead.
        false
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        load_layers::load_all_layers(store, config, gpu, layer_kv_dtypes)
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        // RedHatAI re-quant uses flattened naming; try it first, then standard HF names.
        if let Ok(w) = dense(store, "embed.weight") {
            return Ok(w);
        }
        if let Ok(w) = dense(store, "model.embed_tokens.weight") {
            return Ok(w);
        }
        dense(store, "embed_tokens.weight")
            .context("DeepSeek-V4: no embedding tensor found (tried embed.weight, model.embed_tokens.weight, embed_tokens.weight)")
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        // DeepSeek-V4 uses STANDARD RMSNorm (scale = weight); subtract 1.0 so the
        // offset-from-1 rms_norm kernel computes `1 + (w-1) = w`. See dense_minus_one.
        if let Ok(w) = dense_minus_one(store, "norm.weight", _gpu) {
            return Ok(w);
        }
        if let Ok(w) = dense_minus_one(store, "model.norm.weight", _gpu) {
            return Ok(w);
        }
        dense_minus_one(store, "final_norm.weight", _gpu)
            .context("DeepSeek-V4: no final norm tensor found (tried norm.weight, model.norm.weight, final_norm.weight)")
    }

    fn load_lm_head(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        // Try standard HF name first
        if store.contains("lm_head.weight") {
            return dense(store, "lm_head.weight");
        }
        // RedHatAI / consolidated checkpoints
        if store.contains("output.weight") {
            return dense(store, "output.weight");
        }
        if store.contains("head.weight") {
            return dense(store, "head.weight");
        }
        // Tied embeddings: either config says so, or no separate head exists
        if config.tie_word_embeddings
            || store.contains("embed.weight")
            || store.contains("model.embed_tokens.weight")
        {
            // Tied: reuse the embedding tensor. Inline the same dense lookups as
            // load_embedding (load_lm_head has no `gpu`, and they don't need it).
            if let Ok(w) = dense(store, "embed.weight") {
                return Ok(w);
            }
            if let Ok(w) = dense(store, "model.embed_tokens.weight") {
                return Ok(w);
            }
            return dense(store, "embed_tokens.weight")
                .context("DeepSeek-V4: tied lm_head — no embedding tensor found");
        }
        anyhow::bail!(
            "DeepSeek-V4: lm_head not found (tried lm_head.weight, output.weight, head.weight, and tied embeddings)"
        )
    }

    fn load_mtp_weights(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        Ok(None)
    }
}
