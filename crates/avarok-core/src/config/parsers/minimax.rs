// SPDX-License-Identifier: AGPL-3.0-only

//! Split out of `config.rs` for file-size budget. Parser for a model family.

#![allow(unused_imports)]

use anyhow::{Context, Result};
use serde_json::Value;

use super::super::{
    LayerType, ModelConfig, QuantizationConfig, VisionConfig, default_conv_kernel, default_one,
    default_one_f64, default_partial_rotary, default_rms_eps, default_rope_theta, finalize_config,
    parse_quantization_config, parse_vision_config, validate_config,
};

pub(crate) fn parse_minimax_m2(raw: &serde_json::Value) -> Result<ModelConfig> {
    // Start from the flat deserialization — ModelConfig defaults cover the
    // stock fields (hidden_size, num_*_heads, etc.).
    let mut config: ModelConfig =
        serde_json::from_value(raw.clone()).context("Failed to parse minimax_m2 config.json")?;

    // rope_theta: the full 229B config puts it at the top level (flat
    // `rope_theta: 5000000`), but the tiny-random test variant nests it
    // under `rope_parameters: { rope_theta, rope_type }` — HF's newer
    // canonical layout. Honor the nested form when the flat default was
    // kept. Same shape as parse_config's qwen3_5_moe branch.
    if config.rope_theta == default_rope_theta()
        && let Some(rp) = raw.get("rope_parameters")
        && let Some(theta) = rp.get("rope_theta").and_then(serde_json::Value::as_f64)
    {
        config.rope_theta = theta;
    }

    // num_local_experts → num_experts (MiniMax uses the newer name).
    if config.num_experts == 0 {
        let n = raw
            .get("num_local_experts")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0) as usize;
        config.num_experts = n;
    }
    // MoE FFN width per expert: MiniMax ships `intermediate_size` for MoE
    // experts (not a shared dense FFN). If moe_intermediate_size is unset,
    // pull from the top-level intermediate_size.
    if config.moe_intermediate_size == 0 {
        config.moe_intermediate_size = config.intermediate_size;
    }
    // No shared expert on MiniMax M2 — `shared_intermediate_size: 0`.
    config.shared_expert_intermediate_size = 0;

    // Layer pattern: `attn_type_list` is `[1, 1, ..., 1]` of length
    // `num_hidden_layers`. 1 = full attention. (If MiniMax later ships a
    // hybrid variant with lightning/linear attention, this map needs to
    // handle the other type codes — for now M2.x is all-full.)
    if config.layer_types.is_empty()
        && let Some(list) = raw
            .get("attn_type_list")
            .and_then(serde_json::Value::as_array)
    {
        config.layer_types = list.iter().map(|v| {
                match v.as_u64().unwrap_or(1) {
                    1 => LayerType::FullAttention,
                    other => panic!(
                        "minimax_m2: unexpected attn_type_list entry {other} — only 1 (full) is supported in M1"
                    ),
                }
            }).collect();
    }

    // MTP: MiniMax exposes `use_mtp` + `num_mtp_modules` + `mtp_transformer_layers`.
    // We already deserialize those fields above via serde default. Reflect
    // `num_mtp_modules * mtp_transformer_layers` into the existing Atlas
    // `mtp_num_hidden_layers` counter so downstream buffer sizing still works.
    let use_mtp = raw
        .get("use_mtp")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    if use_mtp && config.mtp_num_hidden_layers == 0 {
        let layers_per = config.mtp_transformer_layers.max(1);
        config.mtp_num_hidden_layers = config.num_mtp_modules.max(1) * layers_per;
    }

    // Architecture flags
    config.attn_gated = false; // MiniMax uses ungated Q (like Mistral/Nemotron)
    config.nested_config = false;
    config.model_type = "minimax_m2".to_string();

    // MiniMax M2's `MiniMaxM2SparseMoeBlock.route_tokens_to_experts`
    // unconditionally normalizes the top-k weights
    // (`top_k_weights /= top_k_weights.sum(...)`). The config file omits
    // `norm_topk_prob`, which would otherwise land as `false` by serde
    // default and cause the MoE weighted-sum step to skip normalization —
    // producing gibberish output (routing weights don't sum to 1, so
    // expert contributions are wrongly scaled).
    config.norm_topk_prob = true;

    finalize_config(&mut config, raw)?;
    Ok(config)
}
