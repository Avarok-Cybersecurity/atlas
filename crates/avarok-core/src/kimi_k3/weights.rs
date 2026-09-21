// SPDX-License-Identifier: AGPL-3.0-only

//! Text decoder inventory shared by checkpoint admission and layer binding.
use super::{K3Graph, MixerKind, MlpKind};
use crate::config::ModelConfig;

pub fn text_key(config: &ModelConfig, rest: &str) -> String {
    let p = config.weight_prefix.trim_end_matches('.');
    if p.is_empty() {
        rest.to_string()
    } else {
        format!("{p}.{rest}")
    }
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

/// Required logical keys. Routed expert `.weight` may instead be supplied by
/// both `.weight_packed` and `.weight_scale`; the loader validates those pairs.
/// A prefixed `lm_head.weight` may use the unprefixed checkpoint fallback.
pub fn required_names(config: &ModelConfig) -> Vec<String> {
    let mut names: Vec<_> = [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "model.output_attn_res_proj.weight",
        "model.output_attn_res_norm.weight",
        "lm_head.weight",
    ]
    .into_iter()
    .map(|name| text_key(config, name))
    .collect();
    for layer in K3Graph::from_config(config).layers {
        names.extend(layer_keys(
            config,
            layer.index,
            layer.mixer,
            layer.mlp,
            config.num_experts,
        ));
    }
    names
}
