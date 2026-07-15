// SPDX-License-Identifier: AGPL-3.0-only

//! `impl ModelWeightLoader for MistralWeightLoader` — the trait body
//! delegates to a per-layer phased pipeline split across sibling files
//! to stay under the ≤500 LoC cap.
//!
//! Layer-loading phases:
//! - `phase_lora_qkv`     — wq_a/b, wkv_a/b LoRA + NVFP4 + TP shard
//! - `phase_per_head`     — W_UK_T / W_UV per-head transpose, wq_b_rope
//! - `phase_qk_absorbed`  — fused W_QK_absorbed CPU compute
//! - `phase_block_diag`   — block-diagonal W_UK_BD / W_UV_BD
//! - `phase_o_proj`       — output projection NVFP4
//! - `yarn`               — YaRN inv_freq table (computed once)
//! - `phase_assemble`     — MlaWeights + MoE + TransformerLayer

use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use super::MistralWeightLoader;
use crate::layer::TransformerLayer;
use crate::layers::vision_encoder::VisionEncoder;
use crate::weight_loader::ModelWeightLoader;
use crate::weight_map::{DenseWeight, MtpWeights, dense};

mod ctx;
mod phase_assemble;
mod phase_block_diag;
mod phase_lora_qkv;
mod phase_o_proj;
mod phase_per_head;
mod phase_qk_absorbed;
mod yarn;

impl ModelWeightLoader for MistralWeightLoader {
    fn supports_tp(&self) -> bool {
        // MLA TP: wq_b and wkv_b are sharded ColumnParallel on the
        // head-output axis. wq_a / wkv_a (LoRA down-projections to
        // latent) stay replicated since the latent dim isn't
        // head-dependent. q_a_norm / kv_a_norm (latent norms) also
        // replicated. The CPU-side wkv_b transpose for absorbed MLA
        // (W_UK / W_UV) iterates over `config.num_key_value_heads`
        // which is already TP-local after main.rs's head split, so
        // it naturally produces per-rank absorbed weights.
        true
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        self.load_layers_inner(store, config, gpu, layer_kv_dtypes)
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, "tok_embeddings.weight")
            .or_else(|_| dense(store, "model.embed_tokens.weight"))
            .context("Mistral: embedding not found")
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, "norm.weight")
            .or_else(|_| dense(store, "model.norm.weight"))
            .context("Mistral: final norm not found")
    }

    fn load_lm_head(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        if store.contains("output.weight") {
            dense(store, "output.weight")
        } else if store.contains("lm_head.weight") {
            dense(store, "lm_head.weight")
        } else if config.tie_word_embeddings {
            dense(store, "tok_embeddings.weight")
                .or_else(|_| dense(store, "model.embed_tokens.weight"))
                .context("Mistral: tied embedding lm_head not found")
        } else {
            anyhow::bail!("Mistral: lm_head/output weight not found")
        }
    }

    fn load_mtp_weights(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        Ok(None)
    }

    fn load_vision_encoder(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<VisionEncoder>> {
        Ok(None)
    }
}

/// Inherent helpers — outside the trait impl block.
impl MistralWeightLoader {
    pub(super) fn load_layers_inner(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        let n = config.num_hidden_layers;
        let q_lora = config.q_lora_rank;
        let kv_lora = config.kv_lora_rank;
        let nope = config.qk_nope_head_dim;
        let rope = config.qk_rope_head_dim;
        let v_dim = config.v_head_dim;

        tracing::info!(
            "Mistral MLA→GQA: expanding LoRA on GPU (q_lora={q_lora}, kv_lora={kv_lora}, \
             nope={nope}, rope={rope}, v_dim={v_dim})"
        );

        let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
        let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
        let stream = gpu.default_stream();

        let mut layers: Vec<Box<dyn TransformerLayer>> = Vec::with_capacity(n);
        let mut yarn_inv_freq_shared = spark_runtime::gpu::DevicePtr::NULL;

        for i in 0..n {
            let mut ctx =
                ctx::MistralLayerCtx::new(store, config, gpu, absmax_k, quantize_k, stream, i);
            phase_lora_qkv::load_lora_qkv(&mut ctx)?;
            phase_per_head::build_per_head_views(&mut ctx)?;
            phase_qk_absorbed::build_w_qk_absorbed(&mut ctx)?;
            phase_block_diag::build_block_diagonals(&mut ctx)?;
            phase_o_proj::load_o_proj(&mut ctx)?;
            let yarn_inv_freq =
                ctx::ensure_yarn_inv_freq(&mut yarn_inv_freq_shared, config, rope, gpu)?;
            let layer = phase_assemble::assemble_layer(ctx, yarn_inv_freq, layer_kv_dtypes)?;
            layers.push(layer);

            if (i + 1) % 6 == 0 || i == n - 1 {
                let free = gpu.free_memory().unwrap_or(0);
                tracing::info!("L{}/{n} done — {:.1} GB free", i + 1, free as f64 / 1e9);
            }
        }
        Ok(layers)
    }
}
