// SPDX-License-Identifier: AGPL-3.0-only

//! Split out of `config.rs` for file-size budget. Parser for DeepSeek-V4
//! family (DeepSeek-V4-Flash, DeepSeek-V4-Pro).
//!
//! DeepSeek-V4 is an MLA + MoE architecture with novel features:
//! - Hybrid attention (CSA + HCA) with per-layer compress_ratios
//! - Manifold-Constrained Hyper-Connections (mHC)
//! - sqrtsoftplus routing (fallback: sigmoid)
//! - FP4 experts + FP8 other weights
//! - YaRN rope scaling
//!
//! Fallback strategy: parse config correctly, register model type, and
//! populate standard Atlas fields. Novel features (CSA/HCA, mHC) are
//! stored in config but ignored by the initial fallback loader.

use anyhow::{Context, Result};

use super::super::{LayerType, ModelConfig, finalize_config, parse_quantization_config};

pub fn parse_deepseek_v4(json: &str) -> Result<ModelConfig> {
    let mut raw: serde_json::Value =
        serde_json::from_str(json).context("Invalid JSON in DeepSeek-V4 config.json")?;

    // Some DeepSeek-V4 checkpoints have `null` for numeric fields instead of
    // omitting the key. Serde's #[serde(default)] only handles missing keys,
    // not null. Sanitize all null top-level values to 0.
    if let Some(obj) = raw.as_object_mut() {
        for v in obj.values_mut() {
            if v.is_null() {
                *v = serde_json::Value::Number(serde_json::Number::from(0));
            }
        }
    }

    // DeepSeek-V4 ships a flat config.json (no nested text_config).
    let json_fixed =
        serde_json::to_string(&raw).context("Failed to re-serialize DeepSeek-V4 config")?;
    let mut config: ModelConfig =
        serde_json::from_str(&json_fixed).context("Failed to parse deepseek_v4 config.json")?;

    // Map DeepSeek field names → Atlas canonical names
    if config.num_experts == 0 && config.n_routed_experts > 0 {
        config.num_experts = config.n_routed_experts;
    }

    // DeepSeek-V4 uses `moe_intermediate_size` for both routed and shared experts.
    // `n_shared_experts` is the count; total shared FFN width = count * intermediate.
    let n_shared_experts = raw
        .get("n_shared_experts")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    if config.shared_expert_intermediate_size == 0 && n_shared_experts > 0 {
        config.shared_expert_intermediate_size = n_shared_experts * config.moe_intermediate_size;
    }

    // kv_lora_rank is not present in V4 config.json but is required for MLA
    // paths. DeepSeek-V3 used 512; V4-Flash likely uses a similar value.
    // Fallback: infer from kv_a_proj_with_mqa shape or default to 512.
    if config.kv_lora_rank == 0 {
        config.kv_lora_rank = 512;
    }

    // head_dim may be absent; compute from hidden_size / num_attention_heads
    if config.head_dim == 0 && config.hidden_size > 0 && config.num_attention_heads > 0 {
        config.head_dim = config.hidden_size / config.num_attention_heads;
    }
    // DeepSeek-V4 uses MLA with head_dim=512, NOT hidden_size/num_attention_heads.
    // If the checkpoint lacks head_dim, the computed fallback (4096/64=64) breaks
    // qk_nope_head_dim, kv_dim, and all attention kernels. Force the correct value.
    if config.head_dim == 64
        && config.hidden_size == 4096
        && config.num_attention_heads == 64
        && config.kv_lora_rank > 0
        && config.q_lora_rank > 0
    {
        config.head_dim = 512;
    }

    // q_lora_rank may be absent; DeepSeek-V4-Flash uses 1024 for q_a latent dim
    if config.q_lora_rank == 0 {
        config.q_lora_rank = 1024;
    }

    // qk_nope_head_dim is not in V4 config; compute from head_dim - qk_rope_head_dim
    if config.qk_nope_head_dim == 0 && config.head_dim > 0 && config.qk_rope_head_dim > 0 {
        config.qk_nope_head_dim = config.head_dim - config.qk_rope_head_dim;
    }

    // v_head_dim defaults to head_dim when absent
    if config.v_head_dim == 0 && config.head_dim > 0 {
        config.v_head_dim = config.head_dim;
    }

    // CRITICAL FIX: DeepSeek V4 Flash RoPE parameters
    // The HF config.json has TWO rope_theta values:
    // - rope_theta: 10000 (WRONG - at top level)
    // - compress_rope_theta: 160000 (CORRECT - this is what we need)
    //
    // Previous hardcoded values were WRONG and caused complete corruption of position encoding → gibberish output.
    //
    // HF config.json values (correct):
    // - compress_rope_theta: 160000 (NOT rope_theta which is 10000)
    // - qk_rope_head_dim: 64
    // - qk_nope_head_dim: 448
    //
    // Previous hardcoded values (WRONG):
    // - rope_theta: 1000000
    // - qk_rope_head_dim: 128
    // - qk_nope_head_dim: 384
    //
    // Read from HF config if present (for DeepSeek V4 Flash detected via o_lora_rank > 0)
    // NOTE: Must read compress_rope_theta, NOT rope_theta!
    if config.o_lora_rank > 0 {
        // Read compress_rope_theta (160000) instead of rope_theta (10000)
        if let Some(theta) = raw.get("compress_rope_theta").and_then(|v| v.as_f64()) {
            config.rope_theta = theta;
        }
        if let Some(rope_dim) = raw.get("qk_rope_head_dim").and_then(|v| v.as_u64()) {
            config.qk_rope_head_dim = rope_dim as usize;
        }
        if let Some(nope_dim) = raw.get("qk_nope_head_dim").and_then(|v| v.as_u64()) {
            config.qk_nope_head_dim = nope_dim as usize;
        }
        // Block-diagonal grouped O projection: n_heads*head_dim split into
        // `o_groups` independent groups (see ModelConfig::o_groups).
        if let Some(g) = raw.get("o_groups").and_then(|v| v.as_u64()) {
            config.o_groups = g as usize;
        }
    }

    // partial_rotary_factor for MLA: only the rope portion gets rotated
    if config.qk_rope_head_dim > 0 && config.head_dim > 0 {
        config.partial_rotary_factor = config.qk_rope_head_dim as f64 / config.head_dim as f64;
    }

    // All layers are full attention in fallback (CSA/HCA ignored)
    config.layer_types = vec![LayerType::FullAttention; config.num_hidden_layers];

    // Architecture flags
    config.model_type = "deepseek_v4".to_string();
    config.attn_gated = false; // DeepSeek-V4 uses ungated Q
    config.nested_config = false;
    config.weight_prefix = "model".to_string();

    // Loss-free balancing (noaux_tc) implies correction bias
    let topk_method = raw
        .get("topk_method")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if topk_method == "noaux_tc" {
        config.use_routing_bias = true;
    }

    // Expert routing score function. DeepSeek-V4 uses `sqrtsoftplus`
    // (`sqrt(softplus(logits))`), NOT sigmoid — the MoE forward paths dispatch
    // on `config.scoring_func == "sqrtsoftplus"`. Leaving this unset routes every
    // MoE layer through the sigmoid kernel → wrong expert weights → incoherent
    // generation. (The parser builds config manually, so this is not auto-read.)
    config.scoring_func = raw
        .get("scoring_func")
        .and_then(|v| v.as_str())
        .unwrap_or("sqrtsoftplus")
        .to_string();

    // Routed-expert output scaling (DeepSeek-V4: 1.5). Consumed by the topk
    // kernels; an unset/wrong value mis-scales every routed MoE contribution.
    if let Some(s) = raw.get("routed_scaling_factor").and_then(|v| v.as_f64()) {
        config.routed_scaling_factor = s;
    }

    // MTP: DeepSeek-V4 uses multi-module MTP (num_nextn_predict_layers)
    if let Some(n) = raw.get("num_nextn_predict_layers").and_then(|v| v.as_u64()) {
        config.num_mtp_modules = n as usize;
        config.mtp_transformer_layers = 1;
        config.mtp_num_hidden_layers = n as usize;
    }

    // Parse quantization_config if present
    if config.quantization_config.is_none() {
        config.quantization_config = parse_quantization_config(&raw);
    }

    // Parse compress_ratios from the raw JSON (not in ModelConfig serde)
    if let Some(ratios) = raw.get("compress_ratios").and_then(|v| v.as_array()) {
        config.compress_ratios = ratios
            .iter()
            .filter_map(|v| v.as_u64().map(|x| x as usize))
            .collect();
    }

    // Parse num_hash_layers from raw JSON
    if let Some(n) = raw.get("num_hash_layers").and_then(|v| v.as_u64()) {
        config.num_hash_layers = n as usize;
    }

    // Manifold-Constrained Hyper-Connections (mHC). Every block maintains
    // `hc_mult` residual streams mixed by a per-block Sinkhorn matrix. These
    // are load-bearing: a single-stream residual flow diverges from the
    // trained model. Defaults match DeepSeek-V4 (hc_mult=4, iters=20).
    //
    // NOTE: the null→0 sanitization above breaks `unwrap_or` because a null
    // `hc_mult` becomes `Some(0)` instead of `None`. We therefore fall back
    // when the *parsed* value is 0, not when the key is missing.
    if config.hc_mult == 0 {
        config.hc_mult = 4;
    }
    if config.hc_sinkhorn_iters == 0 {
        config.hc_sinkhorn_iters = 20;
    }
    if config.hc_eps == 0.0 {
        config.hc_eps = 1e-6;
    }

    // YaRN rope scaling. DeepSeek-V4 checkpoints use the `rope_scaling` key
    // (HF transformers naming); some pre-release configs used `rope_parameters`.
    // Accept either so the YaRN params are actually populated (SSOT: the
    // config, not compute.rs defaults).
    if let Some(rp) = raw
        .get("rope_scaling")
        .or_else(|| raw.get("rope_parameters"))
    {
        if let Some(f) = rp.get("factor").and_then(|v| v.as_f64()) {
            config.yarn_factor = f as f32;
        }
        if let Some(bf) = rp.get("beta_fast").and_then(|v| v.as_f64()) {
            config.yarn_beta_fast = bf as f32;
        }
        if let Some(bs) = rp.get("beta_slow").and_then(|v| v.as_f64()) {
            config.yarn_beta_slow = bs as f32;
        }
        if let Some(om) = rp
            .get("original_max_position_embeddings")
            .and_then(|v| v.as_u64())
        {
            config.yarn_original_max_position_embeddings = om as usize;
        }
        // Attention-temperature mscale. HF defaults: mscale=1.0, mscale_all_dim=0.0
        // when absent. `_mscale = get_mscale(factor, mscale) / get_mscale(factor,
        // mscale_all_dim)` is folded into the rope cos/sin (see the V4 forward).
        config.yarn_mscale = rp.get("mscale").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
        config.yarn_mscale_all_dim = rp
            .get("mscale_all_dim")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as f32;
    }

    // DEBUG: Log YaRN parameters to verify they're being read correctly
    println!(
        "DeepSeek-V4 YaRN parameters: factor={:?}, beta_fast={:?}, beta_slow={:?}, original_max_pos={:?}, mscale={:?}, mscale_all_dim={:?}",
        config.yarn_factor,
        config.yarn_beta_fast,
        config.yarn_beta_slow,
        config.yarn_original_max_position_embeddings,
        config.yarn_mscale,
        config.yarn_mscale_all_dim,
    );

    finalize_config(&mut config, &raw)?;
    Ok(config)
}
