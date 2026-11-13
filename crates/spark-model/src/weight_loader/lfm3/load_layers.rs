// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from HuggingFace transformers (models/lfm3, Apache-2.0, Copyright 2026 The HuggingFace Team) and vLLM (vllm/model_executor/models/lfm3.py, Apache-2.0, Copyright contributors to the vLLM project). Changes: ported from Python/Triton to Rust/CUDA for Atlas. Layer loop after weight_loader/laguna/load_layers.rs and weight_loader/gemma4/loader_a.rs (Atlas).

//! Responsibility: build the 24 LFM3 layers: validate the checkpoint against
//! the config, resolve the FP8 route and the RoPE tables once, then walk the
//! layer schedule (`conv` -> L38's conv layer, `full_attention` and
//! `sliding_attention` -> `Qwen3AttentionLayer`), each with a dense SwiGLU FFN.

use anyhow::{Result, bail, ensure};
use avarok_core::config::{LayerType, ModelConfig};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use super::attention::{self, AttnArgs};
use super::fp8::Fp8Loader;
use super::names::{self, Geometry};
use super::rope_tables::RopeTables;
use super::{LICENSE_BANNER, SLIDING_ROPE_THETA, conv_bridge, ffn};
use crate::layer::TransformerLayer;

/// The three lever values the FP8 route reads, resolved from the target table
/// and the environment (the `lfm3_fp8_all` port row, and the two existing FP8
/// rows). Split from [`load_layers`] so a test can name them.
#[derive(Clone, Copy, Debug)]
pub(super) struct Levers {
    pub all: bool,
    pub ffn: bool,
    pub attn: bool,
}

impl Levers {
    pub fn resolved() -> Self {
        let target = crate::layers::ops::target_defaults::resolved();
        Self {
            all: crate::layers::ops::port_rows::lfm3_fp8_all::row(),
            ffn: target.dense_fp8_from_bf16.value,
            attn: target.attn_fp8_from_bf16.value,
        }
    }
}

pub(super) fn load_layers(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    layer_kv_dtypes: &[KvCacheDtype],
) -> Result<Vec<Box<dyn TransformerLayer>>> {
    load_with(store, config, gpu, layer_kv_dtypes, Levers::resolved())
}

/// [`load_layers`] with the levers given (tests drive this directly).
pub(super) fn load_with(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    layer_kv_dtypes: &[KvCacheDtype],
    levers: Levers,
) -> Result<Vec<Box<dyn TransformerLayer>>> {
    tracing::info!("{LICENSE_BANNER}");
    ensure!(
        config.layer_types.len() == config.num_hidden_layers,
        "lfm3 requires a layer schedule ({} entries for {} layers)",
        config.layer_types.len(),
        config.num_hidden_layers
    );
    ensure!(
        layer_kv_dtypes.len() == config.num_attention_layers(),
        "lfm3 requires one KV dtype per attention layer ({} given, {} attention layers)",
        layer_kv_dtypes.len(),
        config.num_attention_layers()
    );
    names::validate(store, config)?;

    let g = Geometry::of(config);
    let fp8 = Fp8Loader::resolve(store, gpu, config, levers.all, levers.ffn, levers.attn);
    let rope = RopeTables::upload(
        gpu,
        store,
        g.head_dim,
        config.rope_theta,
        config.partial_rotary_factor,
        SLIDING_ROPE_THETA,
    )?;

    let mut layers: Vec<Box<dyn TransformerLayer>> = Vec::with_capacity(config.num_hidden_layers);
    let mut attn_idx = 0usize;
    for i in 0..config.num_hidden_layers {
        let lp = config.layer_prefix(i);
        let ffn = ffn::build(&fp8, gpu, &g, &lp)?;
        match config.layer_type(i) {
            LayerType::LinearAttention => {
                layers.push(conv_bridge::build(store, gpu, &fp8, config, &g, i, ffn)?);
            }
            kind @ (LayerType::FullAttention | LayerType::SlidingAttention) => {
                let args = AttnArgs {
                    config,
                    geometry: g,
                    layer: i,
                    attn_idx,
                    kind,
                    kv_dtype: layer_kv_dtypes[attn_idx],
                    rope,
                };
                layers.push(Box::new(attention::build(store, gpu, &fp8, &args, ffn)?));
                attn_idx += 1;
            }
            other => bail!("lfm3 layer {i} has an unsupported kind {other:?}"),
        }
    }
    // Flush the load-time quantize launches before the model is used.
    gpu.synchronize(gpu.default_stream())?;

    let routes = fp8.routes();
    tracing::info!(
        "LFM3 loaded: {} layers ({} conv, {} sliding, {} full); FP8 conversion ffn={} attn={} conv={}",
        layers.len(),
        config.num_ssm_layers(),
        config
            .layer_types
            .iter()
            .filter(|t| **t == LayerType::SlidingAttention)
            .count(),
        config
            .layer_types
            .iter()
            .filter(|t| **t == LayerType::FullAttention)
            .count(),
        routes.ffn,
        routes.attn,
        routes.conv,
    );
    Ok(layers)
}
