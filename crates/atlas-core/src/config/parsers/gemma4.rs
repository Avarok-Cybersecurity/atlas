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

pub(crate) fn parse_gemma4_params(raw: &serde_json::Value) -> Result<ModelConfig> {
    let tc = raw
        .get("text_config")
        .context("gemma4 config missing text_config")?;

    let get_usize = |key: &str| -> usize {
        tc.get(key).and_then(serde_json::Value::as_u64).unwrap_or(0) as usize
    };
    let get_f64 = |key: &str, default: f64| -> f64 {
        tc.get(key)
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(default)
    };

    let hidden_size = get_usize("hidden_size");
    let num_hidden_layers = get_usize("num_hidden_layers");
    let num_attention_heads = get_usize("num_attention_heads");
    let num_key_value_heads = get_usize("num_key_value_heads");
    let head_dim = get_usize("head_dim"); // Explicit, NOT hidden_size / num_attention_heads
    let intermediate_size = get_usize("intermediate_size");
    let vocab_size = get_usize("vocab_size");
    let rms_norm_eps = get_f64("rms_norm_eps", 1e-6);
    let max_position_embeddings = get_usize("max_position_embeddings");

    // Sliding attention uses rope_theta=10000, full attention uses 1000000.
    // Store sliding theta as the default; full theta is handled per-layer at runtime.
    // Gemma-4 stores RoPE params in rope_parameters.{sliding_attention, full_attention},
    // NOT in sliding_attention_config / full_attention_config (which are empty).
    let rope_params = tc.get("rope_parameters");
    let sliding_rope = rope_params.and_then(|r| r.get("sliding_attention"));
    let full_rope = rope_params.and_then(|r| r.get("full_attention"));
    // Fallback to legacy config keys
    let sliding_config = tc.get("sliding_attention_config");
    let full_config = tc.get("full_attention_config");

    let rope_theta = sliding_rope
        .or(sliding_config)
        .and_then(|c| c.get("rope_theta"))
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(10000.0);

    let partial_rotary_factor = full_rope
        .or(full_config)
        .and_then(|c| c.get("partial_rotary_factor"))
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(default_partial_rotary());

    // Parse attention_pattern → layer_types.
    // Both "sliding_attention" and "full_attention" are attention (not SSM).
    let layer_types: Vec<LayerType> = if let Some(pattern) = tc
        .get("attention_pattern")
        .and_then(serde_json::Value::as_array)
    {
        pattern
            .iter()
            .map(|v| {
                // All attention types map to FullAttention (no SSM in Gemma-4)
                match v.as_str().unwrap_or("full_attention") {
                    "sliding_attention" | "full_attention" => LayerType::FullAttention,
                    other => panic!("Unknown Gemma-4 attention_pattern entry: '{other}'"),
                }
            })
            .collect()
    } else {
        vec![LayerType::FullAttention; num_hidden_layers]
    };

    let tie_word_embeddings = raw
        .get("tie_word_embeddings")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);

    // Build config from a clean template, resetting all SSM/MoE fields.
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    // Reset inherited SSM fields to zero
    config.linear_num_key_heads = 0;
    config.linear_key_head_dim = 0;
    config.linear_num_value_heads = 0;
    config.linear_value_head_dim = 0;
    config.linear_conv_kernel_dim = 0;
    // MoE fields: conditionally parse from text_config (26B MoE variant has experts).
    let num_experts = tc.get("num_experts").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let top_k_experts = tc
        .get("top_k_experts")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let moe_intermediate_size = tc
        .get("moe_intermediate_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    if num_experts > 0 {
        config.num_experts = num_experts;
        config.num_experts_per_tok = top_k_experts;
        config.moe_intermediate_size = moe_intermediate_size;
        config.norm_topk_prob = true;
        config.shared_expert_intermediate_size = 0; // dense MLP is separate, not a shared expert
    } else {
        config.num_experts = 0;
        config.num_experts_per_tok = 1;
        config.moe_intermediate_size = 0;
        config.shared_expert_intermediate_size = 0;
    }
    // Reset inherited MTP
    config.mtp_num_hidden_layers = 0;

    // Populate from Gemma-4 config
    config.hidden_size = hidden_size;
    config.num_hidden_layers = num_hidden_layers;
    config.intermediate_size = intermediate_size;
    config.vocab_size = vocab_size;
    // Gemma-4 heterogeneous attention:
    // Sliding layers: 32 Q × 256dim, 16 KV × 256dim → Q_proj=[8192], K_proj=[4096]
    // Full layers:    32 Q × 512dim,  4 KV × 512dim → Q_proj=[16384], K_proj=[2048]
    // Global config uses sliding dims (majority). Buffer sizing uses max across layers.
    // Use max(head_dim, global_head_dim) for buffer sizing.
    // Per-layer overrides handle the actual dimensions at runtime.
    let global_head_dim = tc
        .get("global_head_dim")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    config.num_attention_heads = num_attention_heads; // 32
    config.num_key_value_heads = num_key_value_heads; // 16
    config.head_dim = if global_head_dim > 0 {
        global_head_dim
    } else {
        head_dim
    }; // 512 for buffers
    config.partial_rotary_factor = partial_rotary_factor;
    config.layer_types = layer_types;
    // Sliding-window size for hybrid attention. Gemma-4 config.json has
    // `sliding_window: 1024`. Only sliding layers use it; full layers ignore.
    config.sliding_window = tc
        .get("sliding_window")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0) as u32;
    config.max_position_embeddings = max_position_embeddings;
    config.rope_theta = rope_theta;
    config.rms_norm_eps = rms_norm_eps;
    config.tie_word_embeddings = tie_word_embeddings;
    config.model_type = "gemma4".to_string();
    config.attn_gated = false;
    config.nested_config = true;
    config.norm_topk_prob = false; // No MoE

    // Embedding scale: embeddings *= sqrt(hidden_size) — Gemma family convention
    config.embed_scale = (hidden_size as f32).sqrt();

    // Logit softcapping: cap * tanh(logits / cap). Required for some Gemma models.
    // Gemma-4 31B dense: needs softcap=30 (moderate final norm weights ~4.5).
    // Gemma-4 26B MoE: large final norm weights (~29) make raw logits huge (thousands).
    //   Softcap=30 compresses ALL logits to ±30 destroying discrimination.
    //   MoE variants should NOT use final logit softcapping.
    config.final_logit_softcapping = raw
        .get("text_config")
        .and_then(|tc| tc.get("final_logit_softcapping"))
        .or_else(|| raw.get("final_logit_softcapping"))
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(30.0) as f32;
    // MoE Gemma-4 (e.g. 26B) has large final norm weights (~29) so raw logits
    // are huge (thousands). softcap=30 collapses ALL logits to ±30, destroying
    // discrimination. Disable softcap for MoE variants per the comment above.
    if config.num_experts > 0 && config.model_type == "gemma4" {
        config.final_logit_softcapping = 0.0;
    }
    // Override softcap via env for A/B testing (llama.cpp #21390: 25.0 fixes
    // creative diversity collapse for NVFP4 Gemma-4 at BF16 precision).
    if let Ok(v) = std::env::var("ATLAS_SOFTCAP_OVERRIDE")
        && let Ok(cap) = v.parse::<f32>()
    {
        config.final_logit_softcapping = cap;
    }
    let _softcap_from_config = raw
        .get("text_config")
        .and_then(|tc| tc.get("final_logit_softcapping"))
        .or_else(|| raw.get("final_logit_softcapping"))
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(30.0) as f32;

    finalize_config(&mut config, raw)?;
    Ok(config)
}
