// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from HuggingFace transformers (models/lfm3, Apache-2.0, Copyright 2026 The HuggingFace Team) and vLLM (vllm/model_executor/models/lfm3.py, Apache-2.0, Copyright contributors to the vLLM project). Changes: ported from Python/Triton to Rust/CUDA for Atlas. Layer assembly follows weight_loader/laguna/load_layers.rs and weight_loader/step3p7/load_layers.rs (Atlas).

//! Responsibility: assemble one LFM3 attention layer (full or sliding) on the
//! existing `Qwen3AttentionLayer`.
//!
//! LFM3's attention block is `h + post_operator_norm(attn(operator_norm(h)))`
//! then `h + post_ffn_norm(ffn(ffn_norm(h)))`, which is the layer's four-norm
//! structure: `input_norm` = `operator_norm`, `post_attn_norm` (the norm in
//! front of the FFN) = `ffn_norm`, and `set_post_sublayer_norms` takes
//! `post_operator_norm` and `post_ffn_norm`. The attention itself is GQA 16/4
//! with head_dim 128, per-head RMS QK norm before RoPE, a per-head SIGMOID
//! gate from `gate_proj [heads, hidden]` applied to the head outputs before
//! `o_proj` (`HeadGateActivation::Sigmoid`, as Step 3.7's `g_proj`), and
//! scale `1 / sqrt(head_dim)`. Sliding layers carry window 256.
//!
//! RoPE goes through the table kernel (`set_yarn_rope`, attention factor 1):
//! the zero-tail table of `rope_tables.rs` is the only way to express the
//! full-attention layers' half rotation without a new kernel. The norm
//! kernels are the "vanilla" ones (`out = x * w / rms`), selected for the
//! model type by `crate::ships_vanilla_norm_weights` (`lfm3` is in that list).

use anyhow::{Result, bail};
use avarok_core::config::{LayerType, ModelConfig};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use super::fp8::{Class, Fp8Loader, Group};
use super::names::Geometry;
use super::rope_tables::RopeTables;
use super::{SLIDING_ROPE_THETA, dense_norm};
use crate::layers::qwen3_attention::HeadGateActivation;
use crate::layers::{FfnComponent, Qwen3AttentionLayer};
use crate::weight_map::{AttentionWeights, DenseWeight, QuantizedWeight, load_kv_scales};

/// Everything one attention layer needs besides the weight store.
pub(super) struct AttnArgs<'a> {
    pub config: &'a ModelConfig,
    pub geometry: Geometry,
    /// The decoder layer index (`model.layers.{i}`).
    pub layer: usize,
    /// The KV-cache layer index (a counter over attention layers only).
    pub attn_idx: usize,
    pub kind: LayerType,
    pub kv_dtype: KvCacheDtype,
    pub rope: RopeTables,
}

/// Build the layer; `ffn` is the layer's already-built dense FFN.
pub(super) fn build(
    store: &WeightStore,
    gpu: &dyn GpuBackend,
    fp8: &Fp8Loader<'_>,
    a: &AttnArgs<'_>,
    ffn: FfnComponent,
) -> Result<Qwen3AttentionLayer> {
    let g = a.geometry;
    let lp = a.config.layer_prefix(a.layer);
    let p = format!("{lp}.self_attn");

    let input_norm = dense_norm(store, gpu, &format!("{lp}.operator_norm.weight"))?;
    let ffn_norm = dense_norm(store, gpu, &format!("{lp}.ffn_norm.weight"))?;
    let post_operator_norm = dense_norm(store, gpu, &format!("{lp}.post_operator_norm.weight"))?;
    let post_ffn_norm = dense_norm(store, gpu, &format!("{lp}.post_ffn_norm.weight"))?;

    let (q_name, k_name, v_name, o_name) = (
        format!("{p}.q_proj"),
        format!("{p}.k_proj"),
        format!("{p}.v_proj"),
        format!("{p}.o_proj"),
    );
    let group = fp8.load_group(
        Class::Attn,
        [
            (q_name.as_str(), g.q_width(), g.hidden),
            (k_name.as_str(), g.kv_width(), g.hidden),
            (v_name.as_str(), g.kv_width(), g.hidden),
            (o_name.as_str(), g.hidden, g.q_width()),
        ],
    )?;

    let null = DenseWeight {
        weight: spark_runtime::gpu::DevicePtr::NULL,
    };
    let (q_proj, k_proj, v_proj) = match &group {
        Group::Bf16([q, k, v, _]) => (*q, *k, *v),
        // The FP8 overlay replaces these slots; the BF16 ones stay NULL.
        Group::Fp8(_) => (null, null, null),
    };
    let (k_scale, v_scale) = load_kv_scales(store, &p, gpu);
    let attn = AttentionWeights {
        q_proj,
        k_proj,
        v_proj,
        o_proj: QuantizedWeight::null(),
        q_norm: dense_norm(store, gpu, &format!("{p}.q_norm.weight"))?,
        k_norm: dense_norm(store, gpu, &format!("{p}.k_norm.weight"))?,
        q_norm_full: None,
        k_norm_full: None,
        k_scale,
        v_scale,
    };
    let mut layer = Qwen3AttentionLayer::new_ungated(
        input_norm,
        attn,
        ffn_norm,
        ffn,
        a.attn_idx,
        None,
        None,
        None,
        gpu,
        a.kv_dtype,
        a.config.fp8_kv_calibration_tokens,
        a.config,
    )?;
    layer.set_dimension_overrides(g.head_dim, g.heads, g.kv_heads);

    match group {
        Group::Bf16([_, _, _, o_dense]) => layer.set_o_dense_bf16(o_dense),
        Group::Fp8([qw, kw, vw, ow]) => {
            layer.set_fp8_weights(Some(qw), Some(kw), Some(vw), Some(ow));
            // No transposed FP8 twins are built (as the load-time attention
            // route): every prefill arm that would read one has a W8A8 or a
            // non-transposed W8A16 arm. The in-tree chunk-0 W8A8 arm is armed
            // only when its kernels resolved.
            if layer.has_w8a8_prefill_kernels() {
                layer.set_w8a8_qkv_prefill(true);
            }
        }
    }

    // Per-head sigmoid gate, BF16 [heads, hidden] (never quantized).
    let gate = dense_norm(store, gpu, &format!("{p}.gate_proj.weight"))?;
    layer.set_head_gate_weight(gate, HeadGateActivation::Sigmoid);
    layer.set_post_sublayer_norms(post_operator_norm, post_ffn_norm);

    match a.kind {
        LayerType::SlidingAttention => {
            layer.set_sliding_window(Some(a.config.sliding_window));
            layer.set_rope_overrides(SLIDING_ROPE_THETA as f32, a.rope.rotary_dim);
            layer.set_yarn_rope(a.rope.sliding, 1.0);
        }
        LayerType::FullAttention => {
            layer.set_sliding_window(None);
            layer.set_rope_overrides(a.config.rope_theta as f32, a.rope.rotary_dim);
            layer.set_yarn_rope(a.rope.full, 1.0);
        }
        other => bail!(
            "lfm3 layer {} is not an attention layer: {other:?}",
            a.layer
        ),
    }
    Ok(layer)
}
