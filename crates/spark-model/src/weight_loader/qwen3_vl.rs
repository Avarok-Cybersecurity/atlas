// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use super::{ModelWeightLoader, WeightFormat};
use crate::layer::TransformerLayer;
use crate::layers::vision_encoder::{MergerLayer, ViTBlock};
use crate::layers::{FfnComponent, MoeLayer, Qwen3AttentionLayer, VisionEncoder};
use crate::tp_shard::{TpShardKind, load_qkvo_tp, shard_quantized_nvfp4};
use crate::weight_map::{
    AttentionWeights, DenseWeight, MtpWeights, dense, detect_nvfp4_variant, load_kv_scales,
    load_moe_no_shared, quantize_to_nvfp4, quantized_auto,
};

pub struct Qwen3VLWeightLoader;

impl ModelWeightLoader for Qwen3VLWeightLoader {
    fn supports_tp(&self) -> bool {
        // Q/K/V column-parallel + O row-parallel via shard_quantized_nvfp4
        // (single quant path: NVFP4 from disk). Per-head q_norm/k_norm
        // are replicated naturally — TP just shards heads, the norm
        // weights duplicate. MoE and vision encoder remain full-replica.
        true
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        let mut layers: Vec<Box<dyn TransformerLayer>> =
            Vec::with_capacity(config.num_hidden_layers);

        let variant = detect_nvfp4_variant(store, config);
        let weight_format = WeightFormat::detect(store, config);
        tracing::info!(
            "Weight format: {:?}, NVFP4 variant: {:?}",
            weight_format,
            variant
        );

        let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
        let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
        let stream = gpu.default_stream();
        let h = config.hidden_size;

        for i in 0..config.num_hidden_layers {
            let lp = config.layer_prefix(i);
            let input_norm = dense(store, &format!("{lp}.input_layernorm.weight"))?;
            let post_attn_norm = dense(store, &format!("{lp}.post_attention_layernorm.weight"))?;

            // MoE without shared experts
            let moe_weights =
                load_moe_no_shared(store, &lp, config.num_experts, gpu, config, variant)?;
            let gate_nvfp4 = quantize_to_nvfp4(
                &moe_weights.gate,
                config.num_experts,
                h,
                gpu,
                absmax_k,
                quantize_k,
                stream,
            )?;
            let ffn = FfnComponent::Moe(MoeLayer::new(
                moe_weights,
                config.num_experts,
                Some(gate_nvfp4),
                gpu,
                config,
            )?);

            // All layers are FullAttention with ungated Q projection.
            //
            // TP: Q/K/V column-parallel + O row-parallel via the shared
            // `load_qkvo_tp` helper. Each of the four projections is loaded
            // as NVFP4 from disk, then sliced on the matching axis.
            // NVFP4 group_size is 16 for Qwen3-VL.
            let p = format!("{lp}.self_attn");
            let tp_rank = config.tp_rank;
            let tp_size = config.tp_world_size.max(1);
            let group_size = 16usize;
            let load_proj = |name: &str,
                             full_n: usize,
                             full_k: usize,
                             kind: TpShardKind|
             -> Result<crate::weight_map::QuantizedWeight> {
                let src = quantized_auto(store, &format!("{p}.{name}"), gpu, variant)?;
                if tp_size == 1 {
                    return Ok(src);
                }
                let sharded = shard_quantized_nvfp4(
                    &src, full_n, full_k, kind, tp_rank, tp_size, group_size, gpu,
                )?;
                gpu.free(src.weight)?;
                gpu.free(src.weight_scale)?;
                Ok(sharded)
            };
            let [q, k, v, o] = load_qkvo_tp(config, load_proj)?;
            let dummy = DenseWeight {
                weight: spark_runtime::gpu::DevicePtr::NULL,
            };
            let (k_scale, v_scale) = load_kv_scales(store, &p, gpu);
            let attn = AttentionWeights {
                q_proj: dummy,
                k_proj: dummy,
                v_proj: dummy,
                o_proj: o,
                q_norm: dense(store, &format!("{p}.q_norm.weight"))?,
                k_norm: dense(store, &format!("{p}.k_norm.weight"))?,
                q_norm_full: None,
                k_norm_full: None,
                k_scale,
                v_scale,
            };

            layers.push(Box::new(Qwen3AttentionLayer::new_ungated(
                input_norm,
                attn,
                post_attn_norm,
                ffn,
                i, // attn_layer_idx = layer index (all layers are attention)
                Some(q),
                Some(k),
                Some(v),
                gpu,
                layer_kv_dtypes[i],
                config.fp8_kv_calibration_tokens,
                config,
            )?));

            if (i + 1) % 10 == 0 {
                tracing::info!("Loaded layers 0..{}", i + 1);
            }
        }

        tracing::info!(
            "Qwen3-VL weight loader: {} layers (all attention, ungated)",
            layers.len(),
        );

        Ok(layers)
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
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

    fn load_lm_head(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        for pattern in &[
            "lm_head.weight",
            "language_model.lm_head.weight",
            "model.lm_head.weight",
        ] {
            if store.contains(pattern) {
                return dense(store, pattern);
            }
        }
        let prefix = &config.weight_prefix;
        dense(store, &format!("{prefix}.embed_tokens.weight"))
    }

    fn load_mtp_weights(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        Ok(None) // VL model has no MTP
    }

    fn load_vision_encoder(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<VisionEncoder>> {
        let vcfg = match &config.vision {
            Some(v) => v.clone(),
            None => return Ok(None),
        };
        let vp = "model.visual";

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
            blocks.push(ViTBlock {
                norm1_w: dense(store, &format!("{bp}.norm1.weight"))?.weight,
                norm1_b: dense(store, &format!("{bp}.norm1.bias"))?.weight,
                qkv_w: dense(store, &format!("{bp}.attn.qkv.weight"))?.weight,
                qkv_b: dense(store, &format!("{bp}.attn.qkv.bias"))?.weight,
                proj_w: dense(store, &format!("{bp}.attn.proj.weight"))?.weight,
                proj_b: dense(store, &format!("{bp}.attn.proj.bias"))?.weight,
                norm2_w: dense(store, &format!("{bp}.norm2.weight"))?.weight,
                norm2_b: dense(store, &format!("{bp}.norm2.bias"))?.weight,
                fc1_w: dense(store, &format!("{bp}.mlp.linear_fc1.weight"))?.weight,
                fc1_b: dense(store, &format!("{bp}.mlp.linear_fc1.bias"))?.weight,
                fc2_w: dense(store, &format!("{bp}.mlp.linear_fc2.weight"))?.weight,
                fc2_b: dense(store, &format!("{bp}.mlp.linear_fc2.bias"))?.weight,
            });
        }

        let mut deepstack = Vec::with_capacity(vcfg.deepstack_visual_indexes.len());
        for i in 0..vcfg.deepstack_visual_indexes.len() {
            let mp = format!("{vp}.deepstack_merger_list.{i}");
            deepstack.push(MergerLayer {
                norm_w: dense(store, &format!("{mp}.norm.weight"))?.weight,
                norm_b: dense(store, &format!("{mp}.norm.bias"))?.weight,
                fc1_w: dense(store, &format!("{mp}.linear_fc1.weight"))?.weight,
                fc1_b: dense(store, &format!("{mp}.linear_fc1.bias"))?.weight,
                fc2_w: dense(store, &format!("{mp}.linear_fc2.weight"))?.weight,
                fc2_b: dense(store, &format!("{mp}.linear_fc2.bias"))?.weight,
            });
        }

        let mp = format!("{vp}.merger");
        let merger = MergerLayer {
            norm_w: dense(store, &format!("{mp}.norm.weight"))?.weight,
            norm_b: dense(store, &format!("{mp}.norm.bias"))?.weight,
            fc1_w: dense(store, &format!("{mp}.linear_fc1.weight"))?.weight,
            fc1_b: dense(store, &format!("{mp}.linear_fc1.bias"))?.weight,
            fc2_w: dense(store, &format!("{mp}.linear_fc2.weight"))?.weight,
            fc2_b: dense(store, &format!("{mp}.linear_fc2.bias"))?.weight,
        };

        let deepstack_indexes = vcfg.deepstack_visual_indexes.clone();
        let ve = VisionEncoder::new(
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
            "Vision encoder loaded: depth={}, hidden={}, heads={}, deepstack={:?}",
            vcfg.depth,
            vcfg.hidden_size,
            vcfg.num_heads,
            vcfg.deepstack_visual_indexes,
        );
        Ok(Some(ve))
    }
}
