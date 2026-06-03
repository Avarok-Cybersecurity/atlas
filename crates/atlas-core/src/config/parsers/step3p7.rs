// SPDX-License-Identifier: AGPL-3.0-only

//! Step 3.7 Flash config parser.
//!
//! Step 3.7 is a nested-config model (top-level `model_type: "step3p7"` with
//! `text_config` containing the language model dimensions). Architecture:
//!   * 45 hidden layers: 3 dense FFN (layers 0-2) + 42 MoE (layers 3-44)
//!   * Mixed attention: full + sliding (window=512), pattern from layer_types
//!   * 288 experts top-8, sigmoid routing + correction bias
//!   * Shared expert per MoE layer (share_expert_dim=1280)
//!   * 3 MTP draft modules (num_nextn_predict_layers=3)
//!   * Head-wise attention gate (g_proj)
//!   * Partial RoPE 0.5 (64 of 128 dims)
//!
//! Field mapping from Step 3.7 config.json → Atlas ModelConfig:
//!   moe_num_experts       → num_experts
//!   moe_top_k             → num_experts_per_tok
//!   moe_intermediate_size → moe_intermediate_size
//!   share_expert_dim      → shared_expert_intermediate_size
//!   num_attention_groups   → num_key_value_heads
//!   moe_router_activation  → scoring_func
//!   moe_router_scaling_factor → routed_scaling_factor
//!   use_moe_router_bias   → use_routing_bias
//!   num_nextn_predict_layers → mtp_num_hidden_layers
//!   use_head_wise_attn_gate → attn_gated

#![allow(unused_imports)]

use anyhow::{Context, Result};
use serde_json::Value;

use super::super::{
    LayerType, ModelConfig, default_conv_kernel, default_one, default_one_f64,
    default_partial_rotary, default_rms_eps, default_rope_theta, finalize_config,
    parse_quantization_config, parse_vision_config, validate_config,
};

pub(crate) fn parse_step3p7(raw: &serde_json::Value) -> Result<ModelConfig> {
    // Step 3.7 uses nested config: text_config holds the language model params.
    let text_config = raw
        .get("text_config")
        .context("step3p7 config missing text_config")?;

    // Pre-process text_config: Step 3.7 has eos_token_id as an array [1, 2, 128007]
    // but ModelConfig expects a scalar u32. Fix before deserializing.
    let mut tc_value = text_config.clone();
    if let Some(obj) = tc_value.as_object_mut() {
        // Fix array-typed fields that serde expects as scalars
        // eos_token_id: [1, 2, 128007] → 1
        if let Some(eos) = obj.get("eos_token_id").cloned() {
            if eos.is_array() {
                let first = eos.as_array()
                    .and_then(|a| a.first())
                    .and_then(Value::as_u64)
                    .unwrap_or(1);
                obj.insert("eos_token_id".to_string(), Value::from(first));
            }
        }
        // rope_theta: [5000000.0, 10000.0, 10000.0] → 500000.0 (first element)
        if let Some(rt) = obj.get("rope_theta").cloned() {
            if rt.is_array() {
                let first = rt.as_array()
                    .and_then(|a| a.first())
                    .and_then(Value::as_f64)
                    .unwrap_or(500000.0);
                obj.insert("rope_theta".to_string(), Value::from(first));
            }
        }
        // partial_rotary_factors: array → remove (we handle via partial_rotary_factor scalar)
        obj.remove("partial_rotary_factors");
        // Remove other array fields that serde can't handle
        obj.remove("swiglu_limits");
        obj.remove("swiglu_limits_shared");
        obj.remove("use_rope_layers");
        obj.remove("yarn_only_types");
        obj.remove("architectures");
        // moe_layers_enum is a comma-separated string, remove it (we detect MoE by probing)
        obj.remove("moe_layers_enum");
        // Convert layer_types: "sliding_attention" → "full_attention" for serde
        // (Atlas treats both as FullAttention; sliding window is a separate config property)
        if let Some(lt) = obj.get("layer_types").cloned() {
            if let Some(arr) = lt.as_array() {
                let fixed: Vec<Value> = arr.iter().map(|v| {
                    match v.as_str() {
                        Some("sliding_attention") => Value::from("full_attention"),
                        _ => v.clone(),
                    }
                }).collect();
                obj.insert("layer_types".to_string(), Value::from(fixed));
            }
        }
        // Also handle moe_num_experts → num_experts for serde (Step 3.7 uses non-standard field names)
        if let Some(mne) = obj.get("moe_num_experts").cloned() {
            if !obj.contains_key("num_experts") {
                obj.insert("num_experts".to_string(), mne);
            }
        }
        if let Some(mtk) = obj.get("moe_top_k").cloned() {
            if !obj.contains_key("num_experts_per_tok") {
                obj.insert("num_experts_per_tok".to_string(), mtk);
            }
        }
        // Map num_attention_groups → num_key_value_heads
        if let Some(nag) = obj.get("num_attention_groups").cloned() {
            if !obj.contains_key("num_key_value_heads") {
                obj.insert("num_key_value_heads".to_string(), nag);
            }
        }
    }

    let mut config: ModelConfig = serde_json::from_value(tc_value)
        .context("Failed to parse step3p7 text_config")?;

    // Override model_type to the top-level one
    config.model_type = "step3p7".to_string();
    config.nested_config = true;
    // Weight prefix: Step 3.7 uses "model.language_model" for main layers
    config.weight_prefix = "model.language_model".to_string();

    // ── MoE field mapping ───────────────────────────────────────────────
    // Step 3.7 uses different field names than Atlas defaults
    if config.num_experts == 0 {
        config.num_experts = text_config
            .get("moe_num_experts")
            .and_then(Value::as_u64)
            .unwrap_or(288) as usize;
    }
    if config.num_experts_per_tok <= 1 {
        config.num_experts_per_tok = text_config
            .get("moe_top_k")
            .and_then(Value::as_u64)
            .unwrap_or(8) as usize;
    }
    if config.moe_intermediate_size == 0 {
        config.moe_intermediate_size = text_config
            .get("moe_intermediate_size")
            .and_then(Value::as_u64)
            .unwrap_or(1280) as usize;
    }

    // Shared expert: Step 3.7 uses `share_expert_dim` (or `share_expert_dims`)
    config.shared_expert_intermediate_size = text_config
        .get("share_expert_dim")
        .or_else(|| text_config.get("share_expert_dims"))
        .and_then(Value::as_u64)
        .unwrap_or(1280) as usize;

    // ── Attention field mapping ─────────────────────────────────────────
    // Step 3.7 uses `num_attention_groups` for KV heads (GQA groups)
    if config.num_key_value_heads == 0 {
        config.num_key_value_heads = text_config
            .get("num_attention_groups")
            .and_then(Value::as_u64)
            .unwrap_or(8) as usize;
    }

    // Head dim
    if config.head_dim == 0 {
        config.head_dim = text_config
            .get("head_dim")
            .and_then(Value::as_u64)
            .unwrap_or(128) as usize;
    }

    // ── RoPE configuration ──────────────────────────────────────────────
    if let Some(rope_params) = text_config.get("rope_scaling")
        .or_else(|| text_config.get("rope_parameters"))
    {
        if config.rope_theta == default_rope_theta() {
            if let Some(theta) = rope_params.get("rope_theta").and_then(Value::as_f64) {
                config.rope_theta = theta;
            }
        }
        if config.partial_rotary_factor == default_partial_rotary() {
            if let Some(prf) = rope_params.get("partial_rotary_factor").and_then(Value::as_f64) {
                config.partial_rotary_factor = prf;
            }
        }
    }
    // Also check top-level partial_rotary_factor in text_config
    if config.partial_rotary_factor == default_partial_rotary() {
        if let Some(prf) = text_config.get("partial_rotary_factor").and_then(Value::as_f64) {
            config.partial_rotary_factor = prf;
        }
    }

    // ── Routing configuration ───────────────────────────────────────────
    // Step 3.7: `moe_router_activation: "sigmoid"` → `scoring_func: "sigmoid"`
    let router_activation = text_config
        .get("moe_router_activation")
        .and_then(Value::as_str)
        .unwrap_or("sigmoid");
    config.scoring_func = router_activation.to_string();

    // Scaling factor for routed expert weights
    config.routed_scaling_factor = text_config
        .get("moe_router_scaling_factor")
        .and_then(Value::as_f64)
        .unwrap_or(3.0);

    // Router bias for sigmoid routing
    config.use_routing_bias = text_config
        .get("use_moe_router_bias")
        .and_then(Value::as_bool)
        .unwrap_or(true);

    // Normalize top-k expert weights (Step 3.7 has norm_expert_weight: true)
    config.norm_topk_prob = text_config
        .get("norm_expert_weight")
        .and_then(Value::as_bool)
        .unwrap_or(true);

    // ── Layer types ─────────────────────────────────────────────────────
    // Step 3.7 specifies layer_types as strings: "full_attention" / "sliding_attention"
    // Atlas treats both as FullAttention (sliding window is a property, not a layer type)
    if config.layer_types.is_empty() {
        if let Some(list) = text_config.get("layer_types").and_then(Value::as_array) {
            config.layer_types = list
                .iter()
                .map(|v| {
                    match v.as_str().unwrap_or("full_attention") {
                        "full_attention" | "sliding_attention" => LayerType::FullAttention,
                        other => panic!(
                            "step3p7: unexpected layer_type '{other}' — only full_attention and sliding_attention are supported"
                        ),
                    }
                })
                .collect();
        }
    }

    // Truncate layer_types to num_hidden_layers (Step 3.7 includes MTP layers in the array)
    if config.layer_types.len() > config.num_hidden_layers && config.num_hidden_layers > 0 {
        config.layer_types.truncate(config.num_hidden_layers);
    }

    // Sliding window size
    if let Some(sw) = text_config.get("sliding_window").and_then(Value::as_u64) {
        config.sliding_window = sw as u32;
    }

    // ── Attention gate ──────────────────────────────────────────────────
    // Step 3.7: `use_head_wise_attn_gate: true` means attention output is gated (g_proj)
    let use_attn_gate = text_config
        .get("use_head_wise_attn_gate")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    config.attn_gated = use_attn_gate;

    // ── MTP (Multi-Token Prediction) ────────────────────────────────────
    // Step 3.7: `num_nextn_predict_layers: 3` = 3 MTP draft modules
    let mtp_layers = text_config
        .get("num_nextn_predict_layers")
        .and_then(Value::as_u64)
        .unwrap_or(3) as usize;
    config.mtp_num_hidden_layers = mtp_layers;
    config.num_mtp_modules = mtp_layers;
    config.mtp_transformer_layers = 1; // Each MTP module is a single transformer layer

    // ── Vocab size (may be at top level) ────────────────────────────────
    if config.vocab_size == 0 {
        config.vocab_size = raw
            .get("vocab_size")
            .or_else(|| text_config.get("vocab_size"))
            .and_then(Value::as_u64)
            .unwrap_or(128896) as usize;
    }

    // ── EOS token ───────────────────────────────────────────────────────
    if config.eos_token_id == 0 {
        config.eos_token_id = text_config
            .get("eos_token_id")
            .and_then(|v| {
                // May be an int or an array
                v.as_u64().or_else(|| v.as_array().and_then(|a| a.first()).and_then(Value::as_u64))
            })
            .unwrap_or(1) as u32;
    }

    // ── Vision config (if present) ──────────────────────────────────────
    if raw.get("vision_config").is_some() || raw.get("image_token_id").is_some() {
        config.vision = parse_vision_config(raw);
    }

    finalize_config(&mut config, raw)?;
    Ok(config)
}
