// SPDX-License-Identifier: AGPL-3.0-only

//! Top-level model-type dispatch for [`super::parse_config`]. Split out of
//! `config.rs` for file-size budget — handles the JSON `model_type` field
//! and routes to the appropriate parser sub-module.

#![allow(unused_imports)]

use anyhow::{Context, Result};

use super::{
    LayerType, ModelConfig, default_conv_kernel, default_partial_rotary, default_rms_eps,
    default_rope_theta, finalize_config, parse_gemma4_params, parse_minimax_m2,
    parse_mistral_params, parse_quantization_config, parse_step3p7, parse_vision_config,
    validate_config,
};

pub fn parse_config(json: &str) -> Result<ModelConfig> {
    // First, probe the top-level model_type.
    let raw: serde_json::Value =
        serde_json::from_str(json).context("Invalid JSON in config.json")?;

    let top_model_type = raw
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");

    match top_model_type {
        "qwen3_vl_moe" | "qwen3_5_moe" | "qwen3_5" => {
            let text_config = raw
                .get("text_config")
                .context("qwen3_5_moe config missing text_config")?;
            let mut config: ModelConfig = serde_json::from_value(text_config.clone())
                .context("Failed to parse text_config")?;
            // Override model_type to the top-level one (text_config has "*_text" suffix)
            config.model_type = top_model_type.to_string();
            // Weight prefix is auto-detected from store keys in main.rs after loading
            // (different quantizers use different prefixes)
            // eos_token_id from text_config
            if config.eos_token_id == 0 {
                config.eos_token_id = text_config
                    .get("eos_token_id")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0) as u32;
            }
            // Vocab size can also be at top level
            if config.vocab_size == 0 {
                config.vocab_size = raw
                    .get("vocab_size")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0) as usize;
            }
            // rope_theta and partial_rotary_factor from nested rope_parameters
            if let Some(rope_params) = text_config.get("rope_parameters") {
                if config.rope_theta == default_rope_theta()
                    && let Some(theta) = rope_params
                        .get("rope_theta")
                        .and_then(serde_json::Value::as_f64)
                {
                    config.rope_theta = theta;
                }
                // FP8 checkpoints store partial_rotary_factor inside rope_parameters
                if config.partial_rotary_factor == default_partial_rotary()
                    && let Some(prf) = rope_params
                        .get("partial_rotary_factor")
                        .and_then(serde_json::Value::as_f64)
                {
                    config.partial_rotary_factor = prf;
                }
            }
            // Qwen3.5 MoE unconditionally normalizes top-K expert weights
            // (hardcoded in HF's Qwen3_5MoeTopKRouter, no config toggle).
            config.norm_topk_prob = true;
            // Architecture flags
            config.nested_config = true;
            config.attn_gated = top_model_type != "qwen3_vl_moe";
            // Parse vision_config for VL models. Qwen3.6 also ships a ViT
            // tower (detected via the mrope_interleaved flag set below,
            // but we don't have that until after this block, so also
            // trigger when the raw config has a `vision_config` key).
            if top_model_type == "qwen3_vl_moe" || raw.get("vision_config").is_some() {
                config.vision = parse_vision_config(&raw);
            }
            // MRoPE detection: Qwen3.6 MoE sets mrope_interleaved + mrope_section
            // inside text_config.rope_parameters. When present on a MoE
            // variant, rewrite model_type to "qwen3_6_moe" so kernel-target
            // resolution picks the right directory (Qwen3.5-MoE and
            // Qwen3.6-MoE share hidden_size=2048 and would otherwise collide).
            // The backing weight loader stays in the qwen3_5 family — MoE
            // architecture is identical except for MRoPE layout and the
            // full-attention layer gate.
            //
            // Kbenkhaled's Qwen3.5-27B-NVFP4 is dense (top_model_type="qwen3_5",
            // no experts) but also enables MRoPE. For dense, do NOT rewrite:
            // the Qwen35 MoE weight loader would fail looking for mlp.gate.
            // The qwen3.5-27b kernel target handles MRoPE at runtime via the
            // mrope_interleaved / mrope_section flags.
            if let Some(rope_params) = text_config.get("rope_parameters") {
                if let Some(ms) = rope_params.get("mrope_section").and_then(|v| v.as_array())
                    && ms.len() == 3
                {
                    config.mrope_section = [
                        ms[0].as_u64().unwrap_or(0) as usize,
                        ms[1].as_u64().unwrap_or(0) as usize,
                        ms[2].as_u64().unwrap_or(0) as usize,
                    ];
                }
                config.mrope_interleaved = rope_params
                    .get("mrope_interleaved")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let is_moe = top_model_type == "qwen3_5_moe" || top_model_type == "qwen3_vl_moe";
                if is_moe
                    && config.mrope_interleaved
                    && config.mrope_section.iter().sum::<usize>() > 0
                {
                    config.model_type = "qwen3_6_moe".to_string();
                }
            }
            finalize_config(&mut config, &raw)?;
            Ok(config)
        }
        "nemotron_h" => {
            let mut config: ModelConfig =
                serde_json::from_str(json).context("Failed to parse nemotron_h config.json")?;
            // Map Nemotron-H field names → Atlas canonical names
            if config.num_experts == 0 && config.n_routed_experts > 0 {
                config.num_experts = config.n_routed_experts;
            }
            if config.rms_norm_eps == default_rms_eps() && config.norm_eps > 0.0 {
                config.rms_norm_eps = config.norm_eps;
            }
            if config.linear_conv_kernel_dim == default_conv_kernel() && config.conv_kernel > 0 {
                config.linear_conv_kernel_dim = config.conv_kernel;
            }
            if config.shared_expert_intermediate_size == 0
                && config.moe_shared_expert_intermediate_size > 0
            {
                config.shared_expert_intermediate_size = config.moe_shared_expert_intermediate_size;
            }
            // Architecture flags
            config.attn_gated = false;
            config.weight_prefix = "backbone".to_string();
            // Parse hybrid_override_pattern → layer_types
            if !config.hybrid_override_pattern.is_empty() && config.layer_types.is_empty() {
                config.layer_types = config
                    .hybrid_override_pattern
                    .chars()
                    .map(|c| match c {
                        'M' => LayerType::LinearAttention,
                        'E' => LayerType::Moe,
                        '*' => LayerType::FullAttention,
                        other => panic!("Unknown hybrid_override_pattern char: '{other}'"),
                    })
                    .collect();
            }
            finalize_config(&mut config, &raw)?;
            Ok(config)
        }
        "gemma4" => parse_gemma4_params(&raw),
        "minimax_m2" => parse_minimax_m2(&raw),
        "step3p7" => parse_step3p7(&raw),
        _ => {
            // Flat config (qwen3_next, etc.)
            let mut config: ModelConfig =
                serde_json::from_str(json).context("Failed to parse config.json")?;
            config.attn_gated = true;
            finalize_config(&mut config, &raw)?;
            Ok(config)
        }
    }
}
