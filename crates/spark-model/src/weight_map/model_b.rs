// SPDX-License-Identifier: AGPL-3.0-only

//! Auto-extracted from `weight_map.rs` during refactor wave 4a.

#![allow(unused_imports)]

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::*;

impl ModelWeights {
    /// Build typed weight references from a flat WeightStore.
    ///
    /// `layer_types` maps layer index → FullAttention or LinearAttention.
    /// `num_experts` is 512 for Qwen3-Next.
    pub fn from_store(
        store: &WeightStore,
        layer_types: &[atlas_core::config::LayerType],
        num_experts: usize,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<Self> {
        let embed_tokens = dense(store, "model.embed_tokens.weight")?;
        let final_norm = dense(store, "model.norm.weight")?;

        // LM head may be tied to embed_tokens.
        let lm_head = if store.contains("lm_head.weight") {
            dense(store, "lm_head.weight")?
        } else {
            embed_tokens
        };

        let mut layers = Vec::with_capacity(layer_types.len());
        for (i, lt) in layer_types.iter().enumerate() {
            let lp = config.layer_prefix(i);
            let input_norm = dense(store, &format!("{lp}.input_layernorm.weight"))?;
            let post_attn_norm = dense(store, &format!("{lp}.post_attention_layernorm.weight"))?;
            // ModelWeights::from_store is only used for Standard NVFP4 (test/legacy path).
            let dummy_qctx = QuantizeCtx {
                absmax_k: spark_runtime::gpu::KernelHandle(0),
                quantize_k: spark_runtime::gpu::KernelHandle(0),
                stream: 0,
            };
            let moe = load_moe(
                store,
                &lp,
                num_experts,
                gpu,
                config,
                Nvfp4Variant::Standard,
                dummy_qctx,
            )?;

            match lt {
                atlas_core::config::LayerType::FullAttention => {
                    let attn = load_attention(
                        store,
                        &lp,
                        gpu,
                        Nvfp4Variant::Standard,
                        dummy_qctx,
                        config,
                    )?;
                    layers.push(LayerWeights::FullAttention {
                        input_norm,
                        attn,
                        post_attn_norm,
                        moe,
                    });
                }
                atlas_core::config::LayerType::LinearAttention => {
                    let ssm =
                        load_ssm(store, &lp, gpu, Nvfp4Variant::Standard, dummy_qctx, config)?;
                    layers.push(LayerWeights::LinearAttention {
                        input_norm,
                        ssm,
                        post_attn_norm,
                        moe,
                    });
                }
                atlas_core::config::LayerType::Moe => {
                    unreachable!("Qwen3 has no standalone MoE layers")
                }
            }

            if (i + 1) % 12 == 0 {
                tracing::info!("Mapped weights for layers 0..{}", i + 1);
            }
        }

        tracing::info!(
            "Weight map: {} layers ({} attention, {} SSM)",
            layers.len(),
            layers
                .iter()
                .filter(|l| matches!(l, LayerWeights::FullAttention { .. }))
                .count(),
            layers
                .iter()
                .filter(|l| matches!(l, LayerWeights::LinearAttention { .. }))
                .count(),
        );

        Ok(Self {
            embed_tokens,
            final_norm,
            lm_head,
            layers,
        })
    }
}

// ── Nemotron-H weight types and loaders ──
