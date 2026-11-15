// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from HuggingFace transformers (models/lfm3, Apache-2.0, Copyright 2026 The HuggingFace Team) and vLLM (vllm/model_executor/models/lfm3.py, Apache-2.0, Copyright contributors to the vLLM project). Changes: ported from Python/Triton to Rust/CUDA for Atlas.

//! Responsibility: the ONE place the loader touches lane L38's conv layer
//! (`crate::port::lfm3_conv`): gather the checkpoint tensors of a `conv` layer
//! into `Lfm3ConvWeights`, build the layer with `Lfm3ConvLayer::new`, and
//! install the optional E4M3 decode copies of its two projections with
//! `set_fp8_proj` (see L38's `init.rs`).
//!
//! What the layer expects of its weights, and what this file supplies:
//!
//! | `Lfm3ConvWeights` field | tensor | notes |
//! |---|---|---|
//! | `operator_norm`, `post_operator_norm`, `ffn_norm`, `post_ffn_norm` | `[hidden]` | plain weight (`x * w / rms`), the vanilla norm kernels |
//! | `in_proj` | `conv.in_proj.weight [3h, h]` | BF16, always (prefill reads it) |
//! | `conv` | `conv.conv.weight [h, 1, 7]` | the same bytes as `[h, 7]` |
//! | `out_proj` | `conv.out_proj.weight [h, h]` | BF16, always |
//! | `ffn` | `feed_forward.{w1, w3, w2}` | `FfnComponent::Dense`, built by `ffn.rs` |
//!
//! The E4M3 copies (`set_fp8_proj`) arrive only through `lfm3_fp8_all` or an
//! E4M3 conv tensor in the checkpoint; the BF16 weights stay installed.

use anyhow::Result;
use avarok_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use super::dense_norm;
use super::fp8::Fp8Loader;
use super::names::Geometry;
use crate::layer::TransformerLayer;
use crate::layers::FfnComponent;
use crate::port::lfm3_conv::{Lfm3ConvCfg, Lfm3ConvLayer, Lfm3ConvWeights};

/// Build the conv layer `layer` with its already-built dense FFN.
pub(super) fn build(
    store: &WeightStore,
    gpu: &dyn GpuBackend,
    fp8: &Fp8Loader<'_>,
    config: &ModelConfig,
    g: &Geometry,
    layer: usize,
    ffn: FfnComponent,
) -> Result<Box<dyn TransformerLayer>> {
    let lp = config.layer_prefix(layer);
    let cp = format!("{lp}.conv");
    let projs = fp8.load_conv_pair(
        &format!("{cp}.in_proj"),
        &format!("{cp}.out_proj"),
        g.hidden,
    )?;
    let weights = Lfm3ConvWeights {
        operator_norm: dense_norm(store, gpu, &format!("{lp}.operator_norm.weight"))?,
        in_proj: projs.in_bf16,
        // [hidden, 1, 7] and [hidden, 7] are the same bytes.
        conv: dense_norm(store, gpu, &format!("{cp}.conv.weight"))?,
        out_proj: projs.out_bf16,
        post_operator_norm: dense_norm(store, gpu, &format!("{lp}.post_operator_norm.weight"))?,
        ffn_norm: dense_norm(store, gpu, &format!("{lp}.ffn_norm.weight"))?,
        ffn,
        post_ffn_norm: dense_norm(store, gpu, &format!("{lp}.post_ffn_norm.weight"))?,
    };
    let cfg = Lfm3ConvCfg::from_model_config(config, layer);
    let mut conv = Lfm3ConvLayer::new(gpu, &cfg, weights)?;
    if let Some((in_proj, out_proj)) = projs.fp8 {
        conv.set_fp8_proj(in_proj, out_proj)?;
    }
    Ok(Box::new(conv))
}
