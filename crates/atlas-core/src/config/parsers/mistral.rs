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

pub fn parse_mistral_params(json: &str) -> Result<ModelConfig> {
    let raw: serde_json::Value =
        serde_json::from_str(json).context("Invalid JSON in params.json")?;

    let dim = raw.get("dim").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let n_heads = raw.get("n_heads").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let n_kv_heads = raw
        .get("n_kv_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(n_heads as u64) as usize;
    let n_layers = raw.get("n_layers").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let head_dim = raw.get("head_dim").and_then(|v| v.as_u64()).unwrap_or(128) as usize;
    let vocab_size = raw.get("vocab_size").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let hidden_dim = raw.get("hidden_dim").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let rope_theta = raw
        .get("rope_theta")
        .and_then(|v| v.as_f64())
        .unwrap_or(10000.0);
    let norm_eps = raw.get("norm_eps").and_then(|v| v.as_f64()).unwrap_or(1e-6);

    // MLA fields
    let kv_lora_rank = raw
        .get("kv_lora_rank")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let q_lora_rank = raw.get("q_lora_rank").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let qk_nope_head_dim = raw
        .get("qk_nope_head_dim")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let qk_rope_head_dim = raw
        .get("qk_rope_head_dim")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let v_head_dim = raw
        .get("v_head_dim")
        .and_then(|v| v.as_u64())
        .unwrap_or(head_dim as u64) as usize;

    // MoE config
    let moe = raw.get("moe");
    let num_experts = moe
        .and_then(|m| m.get("num_experts"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let num_experts_per_tok = moe
        .and_then(|m| m.get("num_experts_per_tok"))
        .and_then(|v| v.as_u64())
        .unwrap_or(1) as usize;
    let expert_hidden_dim = moe
        .and_then(|m| m.get("expert_hidden_dim"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let num_shared_experts = moe
        .and_then(|m| m.get("num_shared_experts"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    let _shared_expert_size = if num_shared_experts > 0 {
        expert_hidden_dim
    } else {
        0
    };

    // All layers are attention+MoE for Mistral (no SSM)
    let layer_types = vec![LayerType::FullAttention; n_layers];

    // Start from a known-good template and override all fields.
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    // Reset all fields to zero/empty before populating from params.json.
    // CRITICAL: must reset ALL SSM fields inherited from the Qwen3 template,
    // otherwise linear_num_key_heads > 0 triggers FP32 residual → BF16 kernel mismatch.
    config.num_hidden_layers = 0;
    config.intermediate_size = 0;
    config.vocab_size = 0;
    config.num_attention_heads = 0;
    config.num_key_value_heads = 0;
    config.head_dim = 0;
    config.num_experts = 0;
    config.num_experts_per_tok = 1;
    config.moe_intermediate_size = 0;
    config.shared_expert_intermediate_size = 0;
    config.mtp_num_hidden_layers = 0;
    config.linear_num_key_heads = 0;
    config.linear_key_head_dim = 0;
    config.linear_num_value_heads = 0;
    config.linear_value_head_dim = 0;
    config.linear_conv_kernel_dim = 0;
    // MLA: RoPE is applied ONLY to the rope portion (qk_rope_head_dim dims per head).
    // Our RoPE kernel rotates dims 0..rotary_dim-1, so we swap Q/K to [rope|nope]
    // before RoPE, matching the kernel's expectation.
    config.partial_rotary_factor = if qk_rope_head_dim > 0 && head_dim > 0 {
        qk_rope_head_dim as f64 / head_dim as f64
    } else {
        1.0
    };
    config.hidden_size = dim;
    config.num_hidden_layers = n_layers;
    config.intermediate_size = hidden_dim;
    config.vocab_size = vocab_size;
    config.num_attention_heads = n_heads;
    config.num_key_value_heads = n_kv_heads;
    config.head_dim = head_dim;
    config.num_experts = num_experts;
    config.num_experts_per_tok = num_experts_per_tok;
    config.moe_intermediate_size = expert_hidden_dim;
    // Mistral has a shared expert with intermediate_size = expert_hidden_dim.
    // The CUDA kernels have NULL guards for the shared expert slot, so it's safe
    // to set this non-zero even during EP (NULL guard writes zeros for missing experts).
    config.shared_expert_intermediate_size = expert_hidden_dim;
    config.layer_types = layer_types;
    config.max_position_embeddings = raw
        .get("max_position_embeddings")
        .and_then(|v| v.as_u64())
        .unwrap_or(8192) as usize;
    config.rope_theta = rope_theta;
    config.rms_norm_eps = norm_eps;
    config.model_type = "mistral".to_string();
    config.attn_gated = false;
    config.kv_lora_rank = kv_lora_rank;
    config.q_lora_rank = q_lora_rank;
    config.qk_nope_head_dim = qk_nope_head_dim;
    config.qk_rope_head_dim = qk_rope_head_dim;
    config.v_head_dim = v_head_dim;

    // YaRN RoPE scaling — Mistral exposes this under `params.json::yarn` with
    // its own naming: `alpha` is the low-rotation cutoff (HF `beta_slow`),
    // `beta` is the high-rotation cutoff (HF `beta_fast`).
    if let Some(yarn) = raw.get("yarn") {
        config.yarn_factor = yarn.get("factor").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
        config.yarn_beta_slow = yarn.get("alpha").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
        config.yarn_beta_fast = yarn.get("beta").and_then(|v| v.as_f64()).unwrap_or(32.0) as f32;
        config.yarn_original_max_position_embeddings = yarn
            .get("original_max_position_embeddings")
            .and_then(|v| v.as_u64())
            .unwrap_or(8192) as usize;
    }
    // llama_4_scaling Q temperature multiplier (separate from YaRN).
    if let Some(l4) = raw.get("llama_4_scaling") {
        config.llama_4_scaling_beta = l4.get("beta").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
        config.llama_4_scaling_original_max_position_embeddings =
            l4.get("original_max_position_embeddings")
                .and_then(|v| v.as_u64())
                .unwrap_or(8192) as usize;
    }

    // Detect tied embeddings
    config.tie_word_embeddings = raw
        .get("tied_embeddings")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    // Mistral uses standard BOS=1, EOS=2 (</s>)
    config.eos_token_id = 2;
    config.bos_token_id = 1;

    finalize_config(&mut config, &raw)?;
    Ok(config)
}
