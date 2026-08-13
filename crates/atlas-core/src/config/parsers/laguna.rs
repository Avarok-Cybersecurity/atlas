// SPDX-License-Identifier: AGPL-3.0-only

//! Poolside Laguna configuration parser.

use anyhow::{Context, Result, ensure};
use serde_json::Value;

use super::super::{ModelConfig, finalize_config};

fn required<'a>(raw: &'a Value, key: &str) -> Result<&'a Value> {
    raw.get(key)
        .with_context(|| format!("laguna config missing required field `{key}`"))
}

fn required_f64(raw: &Value, key: &str) -> Result<f64> {
    required(raw, key)?
        .as_f64()
        .with_context(|| format!("laguna config field `{key}` must be numeric"))
}

pub(crate) fn parse_laguna(raw: &Value) -> Result<ModelConfig> {
    let mut normalized = raw.clone();
    let object = normalized
        .as_object_mut()
        .context("laguna config.json must be an object")?;

    let eos = required(raw, "eos_token_id")?;
    let primary_eos = match eos {
        Value::Number(n) => n.as_u64(),
        Value::Array(ids) => ids.first().and_then(Value::as_u64),
        _ => None,
    }
    .context("laguna eos_token_id must be an integer or non-empty integer array")?;
    object.insert("eos_token_id".into(), Value::from(primary_eos));

    let mut config: ModelConfig =
        serde_json::from_value(normalized).context("Failed to parse laguna config.json")?;
    ensure!(
        config.hidden_size > 0,
        "laguna hidden_size must be non-zero"
    );
    ensure!(config.head_dim > 0, "laguna head_dim must be non-zero");
    ensure!(
        config.num_key_value_heads > 0,
        "laguna num_key_value_heads must be non-zero"
    );
    ensure!(
        config.num_experts > 0,
        "laguna num_experts must be non-zero"
    );
    ensure!(
        config.num_experts_per_tok > 0,
        "laguna num_experts_per_tok must be non-zero"
    );

    let max_heads = config
        .num_attention_heads_per_layer
        .iter()
        .copied()
        .max()
        .context("laguna num_attention_heads_per_layer must not be empty")?;
    ensure!(
        config
            .num_attention_heads_per_layer
            .iter()
            .all(|heads| heads.is_multiple_of(config.num_key_value_heads)),
        "every laguna Q-head count must be divisible by num_key_value_heads"
    );
    config.num_attention_heads = max_heads;

    let full_rope = required(raw, "rope_parameters")?
        .get("full_attention")
        .context("laguna rope_parameters missing full_attention")?;
    ensure!(
        full_rope.get("rope_type").and_then(Value::as_str) == Some("yarn"),
        "laguna full_attention rope_type must be yarn"
    );
    config.rope_theta = required_f64(full_rope, "rope_theta")?;
    config.partial_rotary_factor = required_f64(full_rope, "partial_rotary_factor")?;
    config.yarn_factor = required_f64(full_rope, "factor")? as f32;
    config.yarn_beta_slow = required_f64(full_rope, "beta_slow")? as f32;
    config.yarn_beta_fast = required_f64(full_rope, "beta_fast")? as f32;
    config.yarn_original_max_position_embeddings =
        required(full_rope, "original_max_position_embeddings")?
            .as_u64()
            .context("laguna original_max_position_embeddings must be an integer")?
            as usize;
    config.yarn_attention_factor = required_f64(full_rope, "attention_factor")? as f32;

    let sliding_rope = required(raw, "rope_parameters")?
        .get("sliding_attention")
        .context("laguna rope_parameters missing sliding_attention")?;
    ensure!(
        sliding_rope.get("rope_type").and_then(Value::as_str) == Some("default"),
        "laguna sliding_attention rope_type must be default"
    );
    ensure!(
        required_f64(sliding_rope, "partial_rotary_factor")? == 1.0,
        "laguna sliding_attention partial_rotary_factor must be 1.0"
    );

    config.routed_scaling_factor = required_f64(raw, "moe_routed_scaling_factor")?;
    config.scoring_func = "sigmoid".to_string();
    config.use_routing_bias = true;
    config.qk_norm_type = "per_head".to_string();
    config.attn_gated = false;
    config.weight_prefix.clear();
    config.mtp_num_hidden_layers = 0;
    config.num_mtp_modules = 0;
    config.mtp_transformer_layers = 0;

    finalize_config(&mut config, raw)?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use crate::config::LayerType;

    const CONFIG: &str = r#"{
        "model_type": "laguna",
        "hidden_size": 3072,
        "intermediate_size": 12288,
        "num_hidden_layers": 2,
        "vocab_size": 100352,
        "num_attention_heads": 48,
        "num_attention_heads_per_layer": [48, 72],
        "num_key_value_heads": 8,
        "head_dim": 128,
        "num_experts": 256,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 1024,
        "shared_expert_intermediate_size": 1024,
        "norm_topk_prob": true,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [0],
        "layer_types": ["full_attention", "sliding_attention"],
        "sliding_window": 512,
        "max_position_embeddings": 262144,
        "rms_norm_eps": 0.000001,
        "bos_token_id": 2,
        "eos_token_id": [2, 24],
        "tie_word_embeddings": false,
        "gating": "per-head",
        "moe_routed_scaling_factor": 2.5,
        "rope_parameters": {
            "full_attention": {
                "rope_type": "yarn",
                "rope_theta": 500000.0,
                "factor": 32.0,
                "original_max_position_embeddings": 8192,
                "beta_slow": 1.0,
                "beta_fast": 32.0,
                "attention_factor": 1.3465735902799727,
                "partial_rotary_factor": 0.5
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0
            }
        }
    }"#;

    #[test]
    fn parses_heterogeneous_attention_and_moe() {
        let config = crate::config::parse_config(CONFIG).expect("parse laguna");
        assert_eq!(config.model_type, "laguna");
        assert_eq!(config.num_attention_heads, 72);
        assert_eq!(config.num_attention_heads_per_layer, [48, 72]);
        assert_eq!(
            config.layer_types,
            [LayerType::FullAttention, LayerType::SlidingAttention]
        );
        assert_eq!(config.num_attention_layers(), 2);
        assert_eq!(config.num_ssm_layers(), 0);
        assert_eq!(config.mlp_only_layers, [0]);
        assert_eq!(config.rotary_dim(), 64);
        assert_eq!(config.yarn_factor, 32.0);
        assert_eq!(config.yarn_attention_factor, 1.3465736);
        assert_eq!(config.scoring_func, "sigmoid");
        assert!(config.use_routing_bias);
        assert_eq!(config.routed_scaling_factor, 2.5);
        assert_eq!(config.eos_token_id, 2);
    }

    #[test]
    fn rejects_missing_per_layer_heads() {
        let invalid = CONFIG.replace(
            "\"num_attention_heads_per_layer\": [48, 72]",
            "\"num_attention_heads_per_layer\": []",
        );
        let error = crate::config::parse_config(&invalid).expect_err("must reject missing heads");
        assert!(error.to_string().contains("must not be empty"));
    }
}
