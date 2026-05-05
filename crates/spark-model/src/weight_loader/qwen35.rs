// SPDX-License-Identifier: AGPL-3.0-only

mod load_layers;

use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use super::ModelWeightLoader;
use crate::layer::TransformerLayer;
use crate::weight_map::{DenseWeight, MtpWeights, dense, detect_nvfp4_variant, load_mtp};

pub struct Qwen35WeightLoader;

impl ModelWeightLoader for Qwen35WeightLoader {
    fn supports_tp(&self) -> bool {
        // FullAttention layers are TP-sharded across all 3 quant paths
        // (FP8 native, NVFP4-from-disk, BF16 → NVFP4). LinearAttention
        // (GDN SSM) layers run full-replica per rank — functionally
        // correct (same hidden in → same SSM out across ranks) but
        // wastes SSM weight memory. Acceptable trade-off: SSM weights
        // are a small fraction of total model size for Qwen3.5-A3B
        // (most parameters are in routed MoE experts which are
        // EP-sharded). Future work: GDN HeadParallel sharding would
        // recover the per-rank SSM memory.
        true
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        load_layers::load_layers(self, store, config, gpu, layer_kv_dtypes)
    }

    fn load_embedding(&self, store: &WeightStore, config: &ModelConfig) -> Result<DenseWeight> {
        let prefix = &config.weight_prefix;
        dense(store, &format!("{prefix}.embed_tokens.weight"))
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        let prefix = &config.weight_prefix;
        dense(store, &format!("{prefix}.norm.weight"))
    }

    fn load_lm_head(&self, store: &WeightStore, config: &ModelConfig) -> Result<DenseWeight> {
        // lm_head location varies by quantizer:
        //   Sehyo: "lm_head.weight"
        //   Kbenkhaled: "language_model.lm_head.weight"
        for pattern in &[
            "lm_head.weight",
            "language_model.lm_head.weight",
            "model.lm_head.weight",
        ] {
            if store.contains(pattern) {
                return dense(store, pattern);
            }
        }
        self.load_embedding(store, config)
    }

    fn load_mtp_weights(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        if !store.contains("mtp.fc.weight") {
            tracing::info!("No MTP weights found — speculative decoding disabled");
            return Ok(None);
        }
        let variant = detect_nvfp4_variant(store, config);
        tracing::info!(
            "Loading MTP weights ({} experts, variant={:?})...",
            config.num_experts,
            variant
        );
        let mtp = load_mtp(store, config.num_experts, gpu, variant)?;
        tracing::info!(
            "MTP weights loaded: fc=[2048,4096], {} experts, attn layer",
            mtp.experts.len(),
        );
        Ok(Some(mtp))
    }

    /// Load the Qwen3.6 ViT tower. Returns `None` when `config.vision` is
    /// `None` (Qwen3.5 text-only). Otherwise matches the Qwen3-VL shape
    /// exactly (27 blocks, `model.visual.*` prefix, optional deepstack
    /// merger list + final merger) but auto-dequants FP8 per-channel
    /// weights to BF16 for blocks 4+. Blocks 0-3 are exempted in the
    /// checkpoint's `modules_to_not_convert` list and stay BF16 on disk.
    fn load_vision_encoder(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<crate::layers::VisionEncoder>> {
        use crate::weight_map::dense_auto_fp8_or_bf16;
        let vcfg = match &config.vision {
            Some(v) => v.clone(),
            None => return Ok(None),
        };
        // AEON-7's v2 NVFP4 re-quant (and other multimodal-preserved
        // checkpoints quantized via AutoModelForImageTextToText) keeps
        // the canonical nested layout `model.language_model.visual.*`
        // instead of the flat `model.visual.*` form. Probe the canonical
        // tensor under both prefixes; first hit wins.
        let vp = if store.contains("model.visual.patch_embed.proj.weight") {
            "model.visual"
        } else if store.contains("model.language_model.visual.patch_embed.proj.weight") {
            "model.language_model.visual"
        } else {
            tracing::warn!(
                "Vision encoder tensors absent under both `model.visual.*` and \
                 `model.language_model.visual.*`; skipping vision tower (text-only mode)"
            );
            return Ok(None);
        };

        // Patch embed + position embed are always BF16.
        let patch_embed_w = dense(store, &format!("{vp}.patch_embed.proj.weight"))?;
        let patch_embed_b = dense(store, &format!("{vp}.patch_embed.proj.bias"))?;
        let pos_embed = dense(store, &format!("{vp}.pos_embed.weight"))?;
        let pos_embed_shape = store.get(&format!("{vp}.pos_embed.weight"))?.shape.clone();
        let num_position_embeddings = pos_embed_shape
            .first()
            .copied()
            .context("pos_embed shape missing rows")?;

        let mut blocks = Vec::with_capacity(vcfg.depth);
        for i in 0..vcfg.depth {
            let bp = format!("{vp}.blocks.{i}");
            blocks.push(crate::layers::ViTBlock {
                norm1_w: dense(store, &format!("{bp}.norm1.weight"))?.weight,
                norm1_b: dense(store, &format!("{bp}.norm1.bias"))?.weight,
                qkv_w: dense_auto_fp8_or_bf16(store, &format!("{bp}.attn.qkv"), gpu)?.weight,
                qkv_b: dense(store, &format!("{bp}.attn.qkv.bias"))?.weight,
                proj_w: dense_auto_fp8_or_bf16(store, &format!("{bp}.attn.proj"), gpu)?.weight,
                proj_b: dense(store, &format!("{bp}.attn.proj.bias"))?.weight,
                norm2_w: dense(store, &format!("{bp}.norm2.weight"))?.weight,
                norm2_b: dense(store, &format!("{bp}.norm2.bias"))?.weight,
                fc1_w: dense_auto_fp8_or_bf16(store, &format!("{bp}.mlp.linear_fc1"), gpu)?.weight,
                fc1_b: dense(store, &format!("{bp}.mlp.linear_fc1.bias"))?.weight,
                fc2_w: dense_auto_fp8_or_bf16(store, &format!("{bp}.mlp.linear_fc2"), gpu)?.weight,
                fc2_b: dense(store, &format!("{bp}.mlp.linear_fc2.bias"))?.weight,
            });
        }

        let mut deepstack = Vec::with_capacity(vcfg.deepstack_visual_indexes.len());
        for i in 0..vcfg.deepstack_visual_indexes.len() {
            let mp = format!("{vp}.deepstack_merger_list.{i}");
            deepstack.push(crate::layers::MergerLayer {
                norm_w: dense(store, &format!("{mp}.norm.weight"))?.weight,
                norm_b: dense(store, &format!("{mp}.norm.bias"))?.weight,
                fc1_w: dense_auto_fp8_or_bf16(store, &format!("{mp}.linear_fc1"), gpu)?.weight,
                fc1_b: dense(store, &format!("{mp}.linear_fc1.bias"))?.weight,
                fc2_w: dense_auto_fp8_or_bf16(store, &format!("{mp}.linear_fc2"), gpu)?.weight,
                fc2_b: dense(store, &format!("{mp}.linear_fc2.bias"))?.weight,
            });
        }

        let mp = format!("{vp}.merger");
        let merger = crate::layers::MergerLayer {
            norm_w: dense(store, &format!("{mp}.norm.weight"))?.weight,
            norm_b: dense(store, &format!("{mp}.norm.bias"))?.weight,
            fc1_w: dense_auto_fp8_or_bf16(store, &format!("{mp}.linear_fc1"), gpu)?.weight,
            fc1_b: dense(store, &format!("{mp}.linear_fc1.bias"))?.weight,
            fc2_w: dense_auto_fp8_or_bf16(store, &format!("{mp}.linear_fc2"), gpu)?.weight,
            fc2_b: dense(store, &format!("{mp}.linear_fc2.bias"))?.weight,
        };

        let deepstack_indexes = vcfg.deepstack_visual_indexes.clone();
        let ve = crate::layers::VisionEncoder::new(
            patch_embed_w.weight,
            patch_embed_b.weight,
            pos_embed.weight,
            num_position_embeddings,
            blocks,
            deepstack,
            deepstack_indexes,
            merger,
            vcfg.hidden_size,
            vcfg.num_heads,
            vcfg.spatial_merge_size,
            vcfg.out_hidden_size,
            vcfg.intermediate_size,
            gpu,
        )?;
        tracing::info!(
            "Qwen3.6 vision encoder loaded: depth={}, hidden={}, heads={}, FP8-blocks>=4",
            vcfg.depth,
            vcfg.hidden_size,
            vcfg.num_heads,
        );
        Ok(Some(ve))
    }
}
