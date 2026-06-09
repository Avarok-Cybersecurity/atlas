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
        // eos_token_id: [1, 2, 128007] → 128007 (last = special stop token)
        // Step 3.7 lists multiple EOS tokens; the last is typically the
        // meaningful generation-stopping special token (<|end|> etc.).
        if let Some(arr) = obj.get("eos_token_id").and_then(Value::as_array) {
            let last = arr.last().and_then(Value::as_u64).unwrap_or(1);
            obj.insert("eos_token_id".to_string(), Value::from(last));
        }
        // rope_theta: per-layer array [5e6, 1e4, 1e4, 1e4, ...] → 5000000.0
        // KNOWN LIMITATION: collapsing per-layer theta to scalar. Full-attention
        // layers use θ=5e6, sliding layers use θ=1e4. We take the full-attention
        // value (first element). See RoPE section below for documentation.
        if let Some(arr) = obj.get("rope_theta").and_then(Value::as_array) {
            let first = arr.first().and_then(Value::as_f64).unwrap_or(5000000.0);
            obj.insert("rope_theta".to_string(), Value::from(first));
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
        // layer_types: serde handles "full_attention" and "sliding_attention"
        // natively via the LayerType enum, so no pre-processing needed.
        // We just remove the field here to avoid serde errors from unknown
        // variants ("moe" layers are detected by probing, not from this array).
        // The actual layer_types are populated below from text_config.
        obj.remove("layer_types");
        // Also handle moe_num_experts → num_experts for serde (Step 3.7 uses non-standard field names)
        if let Some(mne) = obj.get("moe_num_experts").cloned() {
            obj.entry("num_experts".to_string()).or_insert(mne);
        }
        if let Some(mtk) = obj.get("moe_top_k").cloned() {
            obj.entry("num_experts_per_tok".to_string()).or_insert(mtk);
        }
        // Map num_attention_groups → num_key_value_heads
        if let Some(nag) = obj.get("num_attention_groups").cloned() {
            obj.entry("num_key_value_heads".to_string()).or_insert(nag);
        }
    }

    let mut config: ModelConfig =
        serde_json::from_value(tc_value).context("Failed to parse step3p7 text_config")?;

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
    // KNOWN LIMITATION: Step 3.7 uses per-layer rope_theta and
    // partial_rotary_factors arrays (theta=5e6 for full-attention layers,
    // theta=1e4 for sliding layers; prf=0.5 for full, 1.0 for sliding).
    // Atlas ModelConfig currently supports only a single scalar for each.
    // We take the first element (full-attention value). This means sliding
    // layers will use incorrect RoPE parameters — acceptable for initial
    // bring-up but will need per-layer support for correct output.
    if let Some(rt) = text_config.get("rope_theta") {
        if let Some(theta) = rt.as_f64() {
            config.rope_theta = theta;
        } else if let Some(theta) = rt
            .as_array()
            .and_then(|a| a.first())
            .and_then(Value::as_f64)
        {
            // Per-layer array — take first (full-attention) value
            config.rope_theta = theta;
        }
    }
    if let Some(rope_params) = text_config
        .get("rope_scaling")
        .or_else(|| text_config.get("rope_parameters"))
    {
        if config.rope_theta == default_rope_theta()
            && let Some(theta) = rope_params.get("rope_theta").and_then(Value::as_f64)
        {
            config.rope_theta = theta;
        }
        if config.partial_rotary_factor == default_partial_rotary()
            && let Some(prf) = rope_params
                .get("partial_rotary_factor")
                .and_then(Value::as_f64)
        {
            config.partial_rotary_factor = prf;
        }
    }
    // Also check top-level partial_rotary_factor in text_config
    if config.partial_rotary_factor == default_partial_rotary()
        && let Some(prf) = text_config
            .get("partial_rotary_factor")
            .and_then(Value::as_f64)
    {
        config.partial_rotary_factor = prf;
    }

    // Compute rotary_dim from partial_rotary_factor if not set explicitly.
    // Step 3.7: partial_rotary_factor=0.5, head_dim=128 → rotary_dim=64.
    if config.rotary_dim == 0 && config.partial_rotary_factor < 1.0 {
        config.rotary_dim = (config.head_dim as f64 * config.partial_rotary_factor) as usize;
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
    // KNOWN LIMITATION: Step 3.7 has mixed attention (12 full + 33 sliding
    // in 45 hidden layers). Atlas currently maps both to FullAttention.
    // The sliding_window value (512) is set globally but not applied
    // per-layer. For correct behaviour, Atlas would need per-layer
    // attention type dispatch. Acceptable for initial bring-up.
    if config.layer_types.is_empty()
        && let Some(list) = text_config.get("layer_types").and_then(Value::as_array)
    {
        config.layer_types = list
            .iter()
            .map(|v| match v.as_str().unwrap_or("full_attention") {
                "full_attention" => LayerType::FullAttention,
                "sliding_attention" => LayerType::SlidingAttention,
                other => panic!(
                    "step3p7: unexpected layer_type '{other}' — \
                     only full_attention and sliding_attention are supported"
                ),
            })
            .collect();
    }

    // Truncate layer_types to num_hidden_layers (Step 3.7 includes MTP layers in the array)
    if config.layer_types.len() > config.num_hidden_layers && config.num_hidden_layers > 0 {
        config.layer_types.truncate(config.num_hidden_layers);
    }

    // Sliding window size
    if let Some(sw) = text_config.get("sliding_window").and_then(Value::as_u64) {
        config.sliding_window = sw as u32;
    }

    // ── Per-layer-type attention head count ─────────────────────────────
    // Step 3.7 has heterogeneous attention: full-attention layers use 64 Q heads,
    // sliding-attention layers use 96 Q heads (from attention_other_setting).
    // Set num_attention_heads to the MAX so buffer sizing accommodates all layers.
    if let Some(other_heads) = text_config
        .get("attention_other_setting")
        .and_then(|o| o.get("num_attention_heads"))
        .and_then(Value::as_u64)
    {
        let other_heads = other_heads as usize;
        if other_heads > config.num_attention_heads {
            config.num_attention_heads = other_heads;
        }
    }

    // ── Attention gate ──────────────────────────────────────────────────
    // Step 3.7 has `use_head_wise_attn_gate: true` with a separate `g_proj`
    // weight [num_q_heads, hidden_size]. This is a PER-HEAD gate (one scalar
    // per head), unlike Qwen 3.5's interleaved Q+G pattern where the gate
    // has the same dimension as Q.
    //
    // Atlas's gated attention pipeline assumes Q+G are interleaved in a
    // single [2*q_dim, hidden] weight, and the deinterleave+sigmoid_gate_mul
    // kernels work element-wise. Step 3.7's per-head gate would require a
    // different kernel (broadcast over head_dim) or weight tiling.
    //
    // For now: disable gating. The model will produce slightly different
    // output without the attention gate, but should still be coherent.
    // TODO: Implement per-head g_proj gating for Step 3.7.
    config.attn_gated = false;

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
                v.as_u64()
                    .or_else(|| v.as_array().and_then(|a| a.first()).and_then(Value::as_u64))
            })
            .unwrap_or(1) as u32;
    }

    // ── Vision config (if present) ──────────────────────────────────────
    if raw.get("vision_config").is_some() || raw.get("image_token_id").is_some() {
        config.vision = parse_vision_config(raw);
    }

    // Step 3.7 uses FP32 residual accumulation for precision across 81 layers.
    config.use_fp32_residual = true;

    finalize_config(&mut config, raw)?;
    Ok(config)
}
