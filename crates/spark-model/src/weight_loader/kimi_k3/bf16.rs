// SPDX-License-Identifier: AGPL-3.0-only

//! Bind 0.40B BF16 (or FP32) twin tensors. Packed MXFP4 is refused upstream.

use anyhow::Result;
use atlas_core::config::ModelConfig;
use atlas_core::kimi_k3::{K3Graph, MixerKind, MlpKind};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use crate::kimi_k3::bound::K3BoundLayer;
use crate::layer::TransformerLayer;
use crate::weight_map::{DenseWeight, dense};

pub fn text_key(config: &ModelConfig, rest: &str) -> String {
    let p = config.weight_prefix.trim_end_matches('.');
    if p.is_empty() {
        rest.to_string()
    } else {
        format!("{p}.{rest}")
    }
}

pub fn load_embedding(
    store: &WeightStore,
    config: &ModelConfig,
    _gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    dense(store, &text_key(config, "model.embed_tokens.weight"))
}

pub fn load_final_norm(
    store: &WeightStore,
    config: &ModelConfig,
    _gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    dense(store, &text_key(config, "model.norm.weight"))
}

pub fn load_lm_head(
    store: &WeightStore,
    config: &ModelConfig,
    _gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let a = text_key(config, "lm_head.weight");
    if store.contains(&a) {
        return dense(store, &a);
    }
    dense(store, "lm_head.weight")
}

pub fn load_layers(
    store: &WeightStore,
    config: &ModelConfig,
    _gpu: &dyn GpuBackend,
) -> Result<Vec<Box<dyn TransformerLayer>>> {
    let graph = K3Graph::from_config(config);
    let mut layers: Vec<Box<dyn TransformerLayer>> = Vec::with_capacity(graph.layers.len());
    for spec in &graph.layers {
        let keys = layer_keys(config, spec.index, spec.mixer, spec.mlp, config.num_experts);
        let mut weights = Vec::with_capacity(keys.len());
        for k in &keys {
            weights.push(dense(store, k)?);
        }
        layers.push(Box::new(K3BoundLayer {
            index: spec.index,
            weights,
        }));
    }
    Ok(layers)
}

pub fn layer_keys(
    config: &ModelConfig,
    i: usize,
    mixer: MixerKind,
    mlp: MlpKind,
    n_experts: usize,
) -> Vec<String> {
    let lp = text_key(config, &format!("model.layers.{i}"));
    let mut k = vec![
        format!("{lp}.input_layernorm.weight"),
        format!("{lp}.post_attention_layernorm.weight"),
        format!("{lp}.mlp_res_norm.weight"),
        format!("{lp}.mlp_res_proj.weight"),
        format!("{lp}.self_attention_res_norm.weight"),
        format!("{lp}.self_attention_res_proj.weight"),
        format!("{lp}.self_attn.g_proj.weight"),
        format!("{lp}.self_attn.o_proj.weight"),
    ];
    match mixer {
        MixerKind::Kda => {
            k.extend([
                format!("{lp}.self_attn.A_log"),
                format!("{lp}.self_attn.b_proj.weight"),
                format!("{lp}.self_attn.dt_bias"),
                format!("{lp}.self_attn.f_a_proj.weight"),
                format!("{lp}.self_attn.f_b_proj.weight"),
                format!("{lp}.self_attn.k_conv1d.weight"),
                format!("{lp}.self_attn.k_proj.weight"),
                format!("{lp}.self_attn.o_norm.weight"),
                format!("{lp}.self_attn.q_conv1d.weight"),
                format!("{lp}.self_attn.q_proj.weight"),
                format!("{lp}.self_attn.v_conv1d.weight"),
                format!("{lp}.self_attn.v_proj.weight"),
            ]);
        }
        MixerKind::Mla => {
            k.extend([
                format!("{lp}.self_attn.kv_a_layernorm.weight"),
                format!("{lp}.self_attn.kv_a_proj_with_mqa.weight"),
                format!("{lp}.self_attn.kv_b_proj.weight"),
                format!("{lp}.self_attn.q_a_layernorm.weight"),
                format!("{lp}.self_attn.q_a_proj.weight"),
                format!("{lp}.self_attn.q_b_proj.weight"),
            ]);
        }
    }
    match mlp {
        MlpKind::Dense => {
            k.extend([
                format!("{lp}.mlp.down_proj.weight"),
                format!("{lp}.mlp.gate_proj.weight"),
                format!("{lp}.mlp.up_proj.weight"),
            ]);
        }
        MlpKind::LatentMoe => {
            k.extend([
                format!("{lp}.block_sparse_moe.gate.e_score_correction_bias"),
                format!("{lp}.block_sparse_moe.gate.weight"),
                format!("{lp}.block_sparse_moe.routed_expert_down_proj.weight"),
                format!("{lp}.block_sparse_moe.routed_expert_norm.weight"),
                format!("{lp}.block_sparse_moe.routed_expert_up_proj.weight"),
                format!("{lp}.block_sparse_moe.shared_experts.down_proj.weight"),
                format!("{lp}.block_sparse_moe.shared_experts.gate_proj.weight"),
                format!("{lp}.block_sparse_moe.shared_experts.up_proj.weight"),
            ]);
            for e in 0..n_experts {
                k.extend([
                    format!("{lp}.block_sparse_moe.experts.{e}.w1.weight"),
                    format!("{lp}.block_sparse_moe.experts.{e}.w2.weight"),
                    format!("{lp}.block_sparse_moe.experts.{e}.w3.weight"),
                ]);
            }
        }
    }
    k
}
