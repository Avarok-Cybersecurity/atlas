// SPDX-License-Identifier: AGPL-3.0-only
//! Replicated checkpoint geometry checked before device allocation.
use crate::config::ModelConfig;
use anyhow::{Context, Result, bail, ensure};

/// Storage geometry for weights that remain full-width on every TP rank.
/// These are the matrix/vector dimensions consumed by `cpu_bind::assemble`.
/// Sharded projections are validated by `tp::plan_tensor_bytes` instead.
pub fn expected_replicated_shape(name: &str, config: &ModelConfig) -> Result<Vec<usize>> {
    let prefix = config.weight_prefix.trim_end_matches('.');
    let canonical = if prefix.is_empty() || name == "lm_head.weight" {
        name
    } else {
        name.strip_prefix(prefix)
            .and_then(|rest| rest.strip_prefix('.'))
            .context("K3 weight does not match configured text prefix")?
    };
    let h = config.hidden_size;
    let shape = match canonical {
        "model.embed_tokens.weight" | "lm_head.weight" => vec![config.vocab_size, h],
        "model.norm.weight" | "model.output_attn_res_norm.weight" => vec![h],
        "model.output_attn_res_proj.weight" => vec![1, h],
        _ => layer_shape(canonical, config)?,
    };
    shape.iter().try_fold(1usize, |elements, &dimension| {
        ensure!(dimension > 0, "{name}: zero replicated dimension");
        elements
            .checked_mul(dimension)
            .context("K3 replicated shape overflow")
    })?;
    Ok(shape)
}

fn layer_shape(name: &str, c: &ModelConfig) -> Result<Vec<usize>> {
    let (layer, suffix) = name
        .strip_prefix("model.layers.")
        .and_then(|rest| rest.split_once('.'))
        .context("unknown replicated K3 tensor")?;
    let layer = layer.parse::<usize>().context("invalid K3 layer index")?;
    ensure!(
        layer < c.layer_types.len(),
        "K3 replicated layer outside configuration"
    );
    let h = c.hidden_size;
    let lat = c.moe_latent_size;
    let shared = if c.shared_expert_intermediate_size > 0 {
        c.shared_expert_intermediate_size
    } else {
        c.moe_intermediate_size
    };
    Ok(match suffix {
        "input_layernorm.weight"
        | "post_attention_layernorm.weight"
        | "mlp_res_norm.weight"
        | "self_attention_res_norm.weight" => vec![h],
        "mlp_res_proj.weight" | "self_attention_res_proj.weight" => vec![1, h],
        "self_attn.f_a_proj.weight" => vec![c.linear_key_head_dim, h],
        "self_attn.o_norm.weight" => vec![c.linear_key_head_dim],
        "self_attn.kv_a_layernorm.weight" => vec![c.kv_lora_rank],
        "self_attn.kv_a_proj_with_mqa.weight" => vec![
            c.kv_lora_rank
                .checked_add(c.qk_rope_head_dim)
                .context("K3 KV width overflow")?,
            h,
        ],
        "self_attn.q_a_layernorm.weight" => vec![c.q_lora_rank],
        "self_attn.q_a_proj.weight" => vec![c.q_lora_rank, h],
        "block_sparse_moe.gate.weight" => vec![c.num_experts, h],
        "block_sparse_moe.gate.e_score_correction_bias" => vec![c.num_experts],
        "block_sparse_moe.routed_expert_down_proj.weight" => vec![lat, h],
        "block_sparse_moe.routed_expert_up_proj.weight" => vec![h, lat],
        "block_sparse_moe.routed_expert_norm.weight" => vec![lat],
        "block_sparse_moe.shared_experts.down_proj.weight" => vec![h, shared],
        "block_sparse_moe.shared_experts.gate_proj.weight"
        | "block_sparse_moe.shared_experts.up_proj.weight" => vec![shared, h],
        _ => bail!("{name}: unknown replicated K3 projection"),
    })
}

pub fn validate_replicated_shape(name: &str, shape: &[usize], config: &ModelConfig) -> Result<()> {
    let expected = expected_replicated_shape(name, config)?;
    // Residual scoring consumes one flat query row. Both vector and Linear
    // [1,H] serialization have identical order and exactly H elements.
    let flat_residual = name.ends_with("res_proj.weight")
        && expected == [1, config.hidden_size]
        && shape == [config.hidden_size];
    ensure!(
        shape == expected || flat_residual,
        "{name}: replicated shape {shape:?} != {expected:?}"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{LayerType, parse_config};

    fn config() -> ModelConfig {
        let mut c = parse_config(include_str!(
            "../../../../docs/k3/fixtures/Kimi-K3-0.40B-config.json"
        ))
        .unwrap();
        c.weight_prefix.clear();
        c.hidden_size = 8;
        c.vocab_size = 32;
        c.layer_types = vec![LayerType::LinearAttention, LayerType::FullAttention];
        c.moe_latent_size = 4;
        c.moe_intermediate_size = 16;
        c.shared_expert_intermediate_size = 12;
        c.num_experts = 3;
        c
    }

    #[test]
    fn embedding_and_norm_reject_tiny_or_transposed_storage() {
        let c = config();
        assert!(validate_replicated_shape("model.embed_tokens.weight", &[32, 8], &c).is_ok());
        for shape in [vec![1], vec![8, 32], vec![256]] {
            assert!(validate_replicated_shape("model.embed_tokens.weight", &shape, &c).is_err());
        }
        assert!(validate_replicated_shape("model.norm.weight", &[8], &c).is_ok());
        assert!(validate_replicated_shape("model.norm.weight", &[1], &c).is_err());
    }

    #[test]
    fn router_and_shared_projections_keep_full_width_at_tp2() {
        let mut c = config();
        c.tp_world_size = 2;
        c.tp_rank = 1;
        let p = "model.layers.1.block_sparse_moe.";
        for (suffix, shape) in [
            ("gate.weight", vec![3, 8]),
            ("gate.e_score_correction_bias", vec![3]),
            ("routed_expert_down_proj.weight", vec![4, 8]),
            ("routed_expert_up_proj.weight", vec![8, 4]),
            ("routed_expert_norm.weight", vec![4]),
            ("shared_experts.down_proj.weight", vec![8, 12]),
            ("shared_experts.gate_proj.weight", vec![12, 8]),
            ("shared_experts.up_proj.weight", vec![12, 8]),
        ] {
            let name = format!("{p}{suffix}");
            assert_eq!(expected_replicated_shape(&name, &c).unwrap(), shape);
            assert!(validate_replicated_shape(&name, &[1], &c).is_err());
        }
    }

    #[test]
    fn unknown_out_of_range_and_zero_dimensions_refuse() {
        let mut c = config();
        for name in [
            "model.layers.9.input_layernorm.weight",
            "model.layers.0.unknown.weight",
        ] {
            assert!(expected_replicated_shape(name, &c).is_err());
        }
        c.hidden_size = 0;
        assert!(expected_replicated_shape("model.norm.weight", &c).is_err());
    }

    #[test]
    fn prefixed_weights_and_residual_query_shapes_are_explicit() {
        let mut c = config();
        c.weight_prefix = "language_model".into();
        assert_eq!(
            expected_replicated_shape("language_model.model.norm.weight", &c).unwrap(),
            [8]
        );
        assert!(expected_replicated_shape("other.model.norm.weight", &c).is_err());
        assert_eq!(
            expected_replicated_shape("lm_head.weight", &c).unwrap(),
            [32, 8]
        );
        let name = "language_model.model.output_attn_res_proj.weight";
        assert!(validate_replicated_shape(name, &[1, 8], &c).is_ok());
        assert!(validate_replicated_shape(name, &[8], &c).is_ok());
        assert!(validate_replicated_shape(name, &[8, 1], &c).is_err());
    }

    #[test]
    fn every_replicated_required_name_has_a_geometry_matching_cpu_bind() {
        use super::super::{
            K3Graph, kda_from, mla_from, moe_from,
            tp::{TpAxis, tensor_plan},
        };
        let c = config();
        let graph = K3Graph::from_config(&c);
        let inventory = super::super::cpu_bind::inventory(
            &c,
            &graph,
            &kda_from(&c),
            &mla_from(&c),
            &moe_from(&c),
        );
        for (name, elements) in inventory {
            let mixer = name
                .split("model.layers.")
                .nth(1)
                .and_then(|tail| tail.split('.').next())
                .and_then(|index| index.parse::<usize>().ok())
                .map(|index| graph.layers[index].mixer)
                .unwrap_or(super::super::MixerKind::Kda);
            if tensor_plan(&name, mixer, &c).0 == TpAxis::Replicated {
                let shape = expected_replicated_shape(&name, &c).unwrap();
                assert_eq!(shape.iter().product::<usize>(), elements, "{name}");
            }
        }
    }
}
